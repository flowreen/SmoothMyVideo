"""
GMFSS pipe interpolation engine for SmoothMyVideo.
ffmpeg decode -> GMFSS anime union model -> ffmpeg encode (audio copied).
Streams frames so there is no PNG folder. Prints "PROGRESS k/total" to stderr for the GUI.

Performance first, no quality knobs: the pipeline always runs fp16, always targets visually
lossless, and always uses the fastest backend the machine supports.
- Backend: TensorRT engines by default (built+cached per resolution on first run), with
  automatic eager fallback when TensorRT is unavailable. Pass --no-trt to force eager.
- Encoder: HEVC (hevc_nvenc) for every source when the device has a usable NVENC session,
  otherwise an automatic CPU fallback to SVT-AV1 (the strongest visually lossless software
  encoder in the bundled LGPL ffmpeg; x264/x265 are not built in). The output codec does not
  echo the source: HEVC at the same visually lossless CQ is far smaller than H.264, and an
  interpolated clip is a new artifact, so matching an H.264 source would only bloat it. Source
  bit depth (8/10 bit), chroma and colour signalling are preserved either way.

Uniform look (every frame generated): the output is fully smoothed. NO emitted frame is a source
frame and none sits on a source timestamp. The output grid is shifted by half an output step, so
every frame is an interior blend: timesteps 1/2M, 3/2M, ... (2M-1)/2M within each source pair for
an integer multi M (symmetric around 0.5, spacing 1/M), and the analogous half step offset in
--fps mode. The first and last output frames are therefore generated too. This is deliberate. A
passthrough interpolator interleaves byte exact source frames (sharp, full real detail) with the
softer generated tweens, so on every Nth frame fine detail snaps in and out (sharp original, soft
tween, sharp original ...), a periodic shimmer that breaks immersion. Note that running a source
frame back through GMFSS at timestep 0 does NOT fix this: at t=0 the model reconstructs it about
as sharply as the original (measured ~equal), so the pop survives. Keeping every output frame off
the source grid is what makes the whole timeline one consistent softness. This mirrors the
"generate every displayed frame, never pass a real one through" behaviour requested from Lossless
Scaling. The cost: the true pixels that sat on the source grid are dropped, so the clip is
uniformly a touch softer than a passthrough render (that uniformity is the goal). The output frame
count is multi*frames (true doubling for 2x, etc., matching target fps tools like Topaz and the
GUI's own total): every source frame gets multi output frames, and the last source frame's own
time slot, which has no frame after it to interpolate toward, is filled by holding the last
generated frame. Duration therefore matches the source.

Usage: gmfss_interp.py <input> <multi> [output] [--scale 1.0] [--fps TARGET] [--no-trt]
       [--sharpen S] [--no-interp] [--upscale F] [--rtx-vsr] [--rtx-hdr] [--hdr-nits N]
       --fps overrides <multi>, resampling the timeline to TARGET output fps.
       --sharpen S applies FSR-style RCAS sharpening (strength 0..1) to every output frame to
       offset the uniform-look softness; omit it (or 0) to leave the frames untouched.
       --no-interp skips interpolation entirely: the clip is only re-encoded at its source fps
       with --sharpen applied, for users who just want the sharpening and not the smoothing.
       --upscale F spatially upscales every output frame before encode by an arbitrary factor
       (1.0 = off). With --rtx-vsr it runs NVIDIA RTX Video Super Resolution (real AI SR, any
       target resolution); otherwise a high-quality bicubic resize. Decode and interpolation
       stay at the source resolution.
       --rtx-hdr converts the output to HDR10 (BT.2020 PQ) via the RTX Video TrueHDR model and writes
       HDR10 static metadata (mastering-display + measured MaxCLL/MaxFALL, see hdr10_meta.py);
       --hdr-nits sets the mastering peak luminance (400..2000, default 1000).
"""
import os
import sys
import math
import json
import argparse
import subprocess
import threading
import queue
import numpy as np
import torch
from torch.nn import functional as F

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(ENGINE_DIR, "GMFSS_Fortuna")
# Prefer ffmpeg/ffprobe bundled at engine/bin so a packaged build needs no system
# ffmpeg on PATH; fall back to the bare PATH names for dev.
_BIN = os.path.join(ENGINE_DIR, "bin")
def _tool(name):
    exe = os.path.join(_BIN, name + ".exe")
    return exe if os.path.isfile(exe) else name
FFMPEG, FFPROBE = _tool("ffmpeg"), _tool("ffprobe")
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

ap = argparse.ArgumentParser()
ap.add_argument("input")
ap.add_argument("multi", type=int)
ap.add_argument("output", nargs="?", default=None)
ap.add_argument("--scale", type=float, default=1.0)
ap.add_argument("--fps", type=float, default=None,
                help="target output fps; overrides <multi> via timeline resampling")
ap.add_argument("--no-trt", action="store_true",
                help="disable the default TensorRT backend and run the eager pipeline")
ap.add_argument("--sharpen", type=float, nargs="?", const=0.8, default=0.0,
                help="FSR-style RCAS sharpening strength 0..1 applied to every output frame "
                     "(0 = off, the default). A bare --sharpen uses 0.8. Offsets the uniform-look "
                     "softness; RCAS self-limits, so even 1.0 keeps texture (unlike plain CAS).")
ap.add_argument("--no-interp", action="store_true",
                help="skip GMFSS interpolation: only re-encode at the source fps with --sharpen "
                     "applied (sharpening without smoothing). The model and TRT are never loaded.")
ap.add_argument("--upscale", type=float, nargs="?", const=1.5, default=1.0,
                help="spatially upscale every output frame before encode by this factor (1.0 = off, "
                     "the default; a bare --upscale uses 1.5). With --rtx-vsr this is RTX Video "
                     "Super Resolution (AI, any target resolution), otherwise a bicubic resize. "
                     "Decode and interpolation stay at the source resolution.")
ap.add_argument("--rtx-vsr", action="store_true",
                help="use the NVIDIA RTX Video SDK (real RTX VSR) for the --upscale step. Requires "
                     "--upscale and the engine/rtxvideo bridge + feature DLLs; falls back to "
                     "bicubic if unavailable.")
ap.add_argument("--rtx-hdr", action="store_true",
                help="convert the output to HDR10 with the RTX Video SDK TrueHDR model (SDR to HDR): "
                     "10-bit BT.2020 PQ. Combines with --upscale (the RTX bridge does VSR then "
                     "TrueHDR in one pass). Needs engine/rtxvideo; falls back to an SDR render if "
                     "the bridge is unavailable.")
ap.add_argument("--hdr-nits", type=int, default=1000,
                help="HDR10 peak luminance in nits for --rtx-hdr (the TrueHDR MaxLuminance and the "
                     "stream's mastering-display / MaxCLL metadata). Clamped to 400..2000; "
                     "default 1000.")
ap.add_argument("--hdr-saturation", type=int, default=0,
                help="TrueHDR Saturation for --rtx-hdr (SDK range 0..200). Default 0 = faithful, no "
                     "added saturation: the SDK's own 'neutral' 100 measurably oversaturates vs the "
                     "SDR source, while 0 matches it. 100 restores the vivid look; in between trades.")
ap.add_argument("--hdr-contrast", type=int, default=100,
                help="TrueHDR Contrast for --rtx-hdr (SDK range 0..200, default 100 = neutral).")
ap.add_argument("--hdr-middlegray", type=int, default=50,
                help="TrueHDR MiddleGray for --rtx-hdr (SDK range 10..100, default 50). Midtone "
                     "anchor; affects brightness, not colour.")
ap.add_argument("--hdr-mastering-prim", choices=["display-p3", "dci-p3", "bt2020", "bt709"],
                default="display-p3",
                help="mastering-display colorspace stamped into the HDR10 mdcv box for --rtx-hdr: "
                     "display-p3 (P3 gamut + D65, the default and what normal HDR masters carry, so a "
                     "player reports real chromaticities), dci-p3 (same P3 gamut, DCI theatrical white, "
                     "SMPTE RP431-2), bt2020 (full nominal BT.2020), or bt709 (the true SDR-source "
                     "gamut). Cosmetic gamut hint; the stream stays BT.2020 PQ and the frames decode "
                     "identically.")
ap.add_argument("--hdr-color", choices=["vivid", "faithful", "raw"], default="vivid",
                help="colour handling for --rtx-hdr (the TrueHDR model rotates hues - it greens/cyans the "
                     "blues - even at saturation 0). vivid (default): keep TrueHDR luminance, take the SDR "
                     "source's hue, and scale chroma in ICtCp by --hdr-vividness - a hue-linear saturation "
                     "gain (HDR pop, no hue shift). faithful: source hue AND saturation, accurate but never "
                     "richer than the source. raw: TrueHDR colour unmodified.")
ap.add_argument("--hdr-vividness", type=float, default=1.3,
                help="for --hdr-color vivid: chroma gain over the colorimetric source, applied in ICtCp so "
                     "hue never shifts. 1.0 == faithful, >1 richer; default 1.3. Clamped 0..4.")
args = ap.parse_args()

inp = os.path.abspath(args.input)
SHARPEN = max(0.0, min(1.0, args.sharpen))   # CAS strength on every output frame; 0 = off
NO_INTERP = args.no_interp                   # sharpen/re-encode only, no frame generation
UPSCALE_F = max(1.0, min(8.0, args.upscale)) # output spatial upscale factor (clamped); 1.0 = off.
                                             # RTX VSR has no integer-scale limit (probed clean past
                                             # 8K), so any factor is allowed up to an 8x sanity cap.
UPSCALE = UPSCALE_F > 1.0
RTX_VSR = args.rtx_vsr                        # use the RTX Video SDK (real RTX VSR) for --upscale
RTX_HDR = args.rtx_hdr                         # convert the output to HDR10 via RTX Video TrueHDR
HDR_NITS = max(400, min(2000, args.hdr_nits)) # HDR10 peak luminance (TrueHDR target + metadata)
HDR_SAT = max(0, min(200, args.hdr_saturation))  # TrueHDR Saturation; default 0 = faithful to source
HDR_CON = max(0, min(200, args.hdr_contrast))    # TrueHDR Contrast (100 = SDK neutral)
HDR_MG = max(10, min(100, args.hdr_middlegray))  # TrueHDR MiddleGray midtone anchor (50 = SDK default)
HDR_MASTER_PRIM = args.hdr_mastering_prim         # mdcv mastering-display colorspace (display-p3/dci-p3/bt2020/bt709)
HDR_COLOR = args.hdr_color                         # vivid (default) / faithful / raw colour handling
HDR_VIVIDNESS = max(0.0, min(4.0, args.hdr_vividness))  # vivid chroma gain over source (1.0 == faithful)

def probe(path):
    # JSON (not csv) so the extra color/format fields stay robust when any of them is
    # absent or "unknown" rather than shifting column positions.
    out = subprocess.check_output(
        [FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,nb_frames,codec_name,pix_fmt,"
         "bits_per_raw_sample,color_space,color_transfer,color_primaries,color_range",
         "-of", "json", path], text=True, creationflags=NO_WINDOW)
    st = (json.loads(out).get("streams") or [{}])[0]
    w, h = int(st["width"]), int(st["height"])
    num, den = (str(st.get("r_frame_rate") or "0/1").split("/") + ["1"])[:2]
    nb = int(st["nb_frames"]) if str(st.get("nb_frames") or "").isdigit() else 0
    return w, h, int(num), int(den or "1"), nb, st

W, H, num, den, NB, ST = probe(inp)

# --- source characteristics, for matched "passthrough quality" encoding ---
def _tag(key):
    v = ST.get(key)
    return v if v and v not in ("unknown", "reserved", "N/A") else None

SRC_CODEC = (ST.get("codec_name") or "").lower()
SRC_PIX = ST.get("pix_fmt") or "yuv420p"
_bpr = ST.get("bits_per_raw_sample")
if _bpr and str(_bpr).isdigit():
    SRC_BITS = int(_bpr)
elif any(s in SRC_PIX for s in ("p10", "10le", "10be")):
    SRC_BITS = 10
elif any(s in SRC_PIX for s in ("p12", "12le", "12be")):
    SRC_BITS = 12
else:
    SRC_BITS = 8
TEN_BIT = SRC_BITS >= 10
CHROMA444 = "444" in SRC_PIX
FPS_MODE = args.fps is not None and args.fps > 0
src_fps = num / den
if not NB:
    # MKV and other streaming containers routinely report nb_frames as N/A. Without a
    # real count, total_pairs would be 0 and the GUI's PROGRESS bar divides by ~zero
    # (runs off to millions of %). Estimate the frame count from the container duration.
    try:
        dur = float(subprocess.check_output(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", inp], text=True, creationflags=NO_WINDOW).strip())
        NB = max(0, int(round(dur * src_fps)))
    except (subprocess.CalledProcessError, ValueError):
        NB = 0
if NO_INTERP:
    # Sharpen-only: keep the source frame rate, emit one frame per source frame.
    rate_str = f"{num}/{den}"
    out_label = int(round(src_fps))
elif FPS_MODE:
    ratio = args.fps / src_fps                 # output frames per source frame
    rate_str = f"{args.fps:g}"
    out_label = int(round(args.fps))
else:
    rate_str = f"{num * args.multi}/{den}"
    out_label = int(round(src_fps * args.multi))
out_path = os.path.abspath(args.output) if args.output else \
    os.path.splitext(inp)[0] + (("_sharpened" if NO_INTERP else f"_{out_label}fps") + ".mp4")

# Decode/encode bit depth follows the source. 8 bit clips keep the original fast rgb24
# path byte for byte; 10 bit and up are carried as 16 bit rgb (rgb48le) so the model
# (fp16, which holds 10 bit precision) never truncates them to 8 bit. GMFSS_Fortuna,
# GMFSS_union and enhancr all emit 8 bit from this point; carrying 10 bit through is the
# one step past them, and it is free because the pipe was already fp16.
if TEN_BIT:
    DEC_FMT, NP_DT, MAXV, BPP = "rgb48le", np.uint16, 65535.0, 6
else:
    DEC_FMT, NP_DT, MAXV, BPP = "rgb24", np.uint8, 255.0, 3
fsize = W * H * BPP
# Output spatial resolution. Decode and GMFSS interpolation stay at the source W x H; each finished
# frame is upscaled to OUT_W x OUT_H just before encode (in to_bytes), so only the encoder input
# size changes. RTX VSR upscales to any output rectangle (no integer-scale restriction), so the
# requested factor is applied directly; both source dimensions scale by the same factor, so the
# aspect ratio is preserved, and each is rounded down to an even number (required by yuv420p /
# p010le). The bicubic fallback targets the same dims.
if UPSCALE:
    OUT_W, OUT_H = (round(W * UPSCALE_F) // 2) * 2, (round(H * UPSCALE_F) // 2) * 2
else:
    OUT_W, OUT_H = W, H
total_pairs = max(1, NB - 1) if NB else 0
# Progress denominator: interpolation steps over source pairs (NB-1); the sharpen-only pass
# instead processes one unit per source frame (NB).
total_units = NB if NO_INTERP else total_pairs

sys.path.insert(0, REPO)
os.chdir(REPO)
torch.set_grad_enabled(False)
device = torch.device("cuda")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def _add_cuda_dll_dirs():
    # let cupy find NVRTC + its builtins DLL from the nvidia-*-cu12 wheels
    for base in list(sys.path):
        nv = os.path.join(base, "nvidia")
        if not os.path.isdir(nv):
            continue
        for sub in os.listdir(nv):
            b = os.path.join(nv, sub, "bin")
            if os.path.isdir(b):
                try:
                    os.add_dll_directory(b)
                except OSError:
                    pass
                os.environ["PATH"] = b + os.pathsep + os.environ.get("PATH", "")
if NO_INTERP:
    # Sharpen-only / re-encode: the GMFSS model and the TensorRT backend are never loaded, so
    # there is no warmup and no first-run engine build. Each frame just gets the RCAS pass.
    model = None
    sys.stderr.write("no-interp mode: GMFSS interpolation disabled "
                     "(re-encode at source fps with optional FSR sharpen)\n"); sys.stderr.flush()
else:
    _add_cuda_dll_dirs()
    from model.GMFSS_infer_u import Model
    model = Model()
    if not hasattr(model, "version"):
        model.version = 0
    model.load_model("train_log", -1)
    model.eval()
    model.device()
    sys.stderr.write("GMFSS union model loaded (fp16)\n"); sys.stderr.flush()

    if not args.no_trt:
        # Default backend: swap sub nets for TensorRT engines (built+cached per resolution on
        # first run, per-subnet eager fallback on any build/run failure). trt_runtime imports
        # tensorrt at module load, so an environment without a working TensorRT is caught here
        # and the eager pipeline is used instead. trt_runtime lives next to this script.
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import trt_runtime
            trt_runtime.trtify(model)
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"[trt] unavailable, using eager pipeline: {repr(e)[:200]}\n")
            sys.stderr.flush()

# Upscale / HDR backend, loaded once for this resolution, then run per frame. Independent of
# NO_INTERP, so sharpen-only runs can also upscale or convert to HDR. The AI backend is the NVIDIA
# RTX Video SDK (engine/rtxvideo bridges its CUDA path; see rtxvideo.py). VSR and TrueHDR run as
# SEPARATE passes so the RCAS sharpen lands at the OUTPUT resolution between them (the max-quality
# order: upscale the clean frame, sharpen at final res, then expand to HDR), instead of the SDK's
# fused VSR->THDR pass which would force the sharpen at the source resolution ahead of VSR.
#
# One bridge instance holds whatever features are needed (the SDK bridge is a single processor):
#   _RTX.run_vsr - RTX Video Super Resolution (--rtx-vsr); _upscale() falls back to bicubic if absent.
#   _RTX.run_hdr - RTX TrueHDR (--rtx-hdr): SDR -> HDR10 at the output resolution.
# Any setup failure leaves _RTX None and the render degrades gracefully (bicubic upscale, SDR render),
# so a missing/blocked RTX runtime never breaks a render.
_RTX = None
_need_vsr = UPSCALE and RTX_VSR              # AI upscale via RTX VSR (else bicubic / no upscale)
_need_hdr = RTX_HDR                          # SDR -> HDR10 via RTX TrueHDR
RTX_VSR_ACTIVE = False                       # True once the RTX VSR pass is confirmed available
HDR_ACTIVE = False                           # drives the 10-bit BT.2020 PQ encode path below
if _need_vsr or _need_hdr:
    try:
        sys.path.insert(0, ENGINE_DIR)
        import rtxvideo
        # OUT_W x OUT_H is the final resolution: VSR's target and the size TrueHDR runs at. It equals
        # W x H when not upscaling, so a no-/bicubic-upscale HDR run still runs TrueHDR at the right
        # size. VSR and TrueHDR features both live on this one instance when both are requested.
        _RTX = rtxvideo.RTXVideo(W, H, OUT_W, OUT_H, vsr=_need_vsr, hdr=_need_hdr,
                                 hdr_max_nits=HDR_NITS, hdr_contrast=HDR_CON, hdr_saturation=HDR_SAT,
                                 hdr_middlegray=HDR_MG, hdr_color=HDR_COLOR, hdr_vividness=HDR_VIVIDNESS)
        RTX_VSR_ACTIVE = _need_vsr
        HDR_ACTIVE = _need_hdr
        if _need_vsr:
            sys.stderr.write(f"RTX Video Super Resolution ready (Ultra) -> {OUT_W}x{OUT_H}\n")
        if _need_hdr:
            _viv = f" {HDR_VIVIDNESS:g}" if HDR_COLOR == "vivid" else ""
            sys.stderr.write(f"RTX HDR ready (TrueHDR {HDR_NITS} nits, sat {HDR_SAT}, con {HDR_CON}, "
                             f"mg {HDR_MG}, colour {HDR_COLOR}{_viv}) HDR10 (BT.2020 PQ) @ {OUT_W}x{OUT_H}\n")
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"[rtx] unavailable, falling back (bicubic upscale / SDR): {repr(e)[:200]}\n")
        _RTX = None
        RTX_VSR_ACTIVE = False
        HDR_ACTIVE = False
    sys.stderr.flush()

scale = args.scale
tmp = max(64, int(64 / scale))
ph = ((H - 1) // tmp + 1) * tmp
pw = ((W - 1) // tmp + 1) * tmp

def amp():
    return torch.autocast("cuda", dtype=torch.float16)

def to_tensor(buf):
    a = np.frombuffer(buf, NP_DT).reshape(H, W, 3)
    # Pinned host buffer + non_blocking H2D: a pageable .to() blocks the main thread until the
    # copy finishes, but a page-locked (pinned) source lets .to() return immediately, so the
    # thread races ahead to queue this frame's inference and fetch the next while the upload runs
    # on the CUDA stream. PyTorch's caching pinned allocator reuses the locked block across frames
    # and defers its reuse until the copy's event completes, so dropping the Python-side reference
    # right after is safe; the per-frame cost is one host copy, not a fresh page-lock.
    t = torch.from_numpy(a.copy()).pin_memory()                  # HWC, one flat memcpy (no strided CPU transpose)
    t = t.to(device, non_blocking=True).permute(2, 0, 1).unsqueeze(0).float() / MAXV  # HWC->CHW folds into the GPU cast
    # Reach the multiple of 64 the model needs by padding the bottom/right edge, not by
    # resizing the whole frame up and back. A bilinear resize (the old path here and in
    # to_bytes) resamples every pixel and softens the entire image; padding then cropping
    # leaves all real content bit-untouched, so the generated frames are strictly sharper.
    # replicate (vs zero) extends the edge smoothly so the flow net has no hard border to track.
    return F.pad(t, (0, pw - W, 0, ph - H), mode="replicate")

RCAS_LIMIT = 0.1875 - 1e-6   # AMD FSR_RCAS_LIMIT = 0.25 - 1/16

def _rcas(img, con):
    """AMD FidelityFX RCAS (Robust Contrast-Adaptive Sharpening) on a [1,3,H,W] float image in [0,1].

    This is the sharpen AMD FSR and Lossless Scaling's FSR mode use, not the plain CAS the ffmpeg
    `cas` filter applies. RCAS limits its sharpening lobe to the 4-neighbour min/max (no overshoot or
    ringing) and attenuates it in noisy regions (the FSR_RCAS_DENOISE term), so it crisps real edges
    without amplifying fine texture/grain into mush the way CAS does at high strength. The lobe is one
    scalar per pixel applied to all three channels, so it cannot decorrelate them into colour speckle.
    con in (0, 1] scales the strength (1 = AMD's max, sharpness 0); con <= 0 is a no-op.
    """
    p = F.pad(img, (1, 1, 1, 1), mode="replicate")        # replicate so edges don't sharpen to black
    b, h = p[..., 0:-2, 1:-1], p[..., 2:, 1:-1]           # up, down
    d, f = p[..., 1:-1, 0:-2], p[..., 1:-1, 2:]           # left, right
    e = img                                               # centre
    def luma(x): return x[:, 1:2] + 0.5 * (x[:, 0:1] + x[:, 2:3])   # green-weighted, as in FSR
    bL, dL, eL, fL, hL = luma(b), luma(d), luma(e), luma(f), luma(h)
    rng = torch.maximum(torch.maximum(torch.maximum(bL, dL), torch.maximum(fL, hL)), eL) \
        - torch.minimum(torch.minimum(torch.minimum(bL, dL), torch.minimum(fL, hL)), eL)
    nz = (0.25 * (bL + dL + fL + hL) - eL).abs() / (rng + 1e-4)
    nz = -0.5 * nz.clamp(0.0, 1.0) + 1.0                   # ~1 on flat areas, ->0.5 in noise
    mn4 = torch.minimum(torch.minimum(b, d), torch.minimum(f, h))
    mx4 = torch.maximum(torch.maximum(b, d), torch.maximum(f, h))
    hit_min = mn4 / (4.0 * mx4 + 1e-4)
    hit_max = (1.0 - mx4) / (4.0 * mn4 - 4.0 - 1e-4)       # denom strictly < 0: no divide-by-zero
    lobe = torch.maximum(-hit_min, hit_max).amax(dim=1, keepdim=True)   # most-limiting channel
    lobe = lobe.clamp(max=0.0).clamp(min=-RCAS_LIMIT) * con * nz
    return ((e + lobe * (b + d + f + h)) / (1.0 + 4.0 * lobe)).clamp(0.0, 1.0)

def _upscale(t, ow, oh):
    """Spatially upscale a [1,3,H,W] RGB float image in [0,1] to (oh, ow), per output frame.

    Uses RTX VSR (the RTX Video SDK, _RTX.run_vsr) when it loaded (--rtx-vsr + the runtime present),
    which outputs exactly (ow, oh); otherwise a high-quality bicubic resize. If VSR fails mid-run it
    is dropped (RTX_VSR_ACTIVE set False) and the rest of the clip falls back to bicubic; bicubic is
    clamped because it can overshoot past [0,1].
    """
    global RTX_VSR_ACTIVE
    if RTX_VSR_ACTIVE and _RTX is not None:
        try:
            return _RTX.run_vsr(t)
        except Exception as e:  # noqa: BLE001 - degrade to bicubic for the rest of the run
            sys.stderr.write(f"[rtx] VSR run failed, using bicubic: {repr(e)[:160]}\n")
            sys.stderr.flush()
            RTX_VSR_ACTIVE = False
    return F.interpolate(t, size=(oh, ow), mode="bicubic", align_corners=False).clamp(0.0, 1.0)

def to_bytes(t):
    t = t.float()[..., :H, :W]            # crop off the padding added in to_tensor
    # Pipeline order, chosen for max quality (VSR -> RCAS -> TrueHDR):
    #   1. upscale the clean interpolated frame (the AI upscaler gets an unsharpened, in-distribution
    #      input);
    #   2. sharpen at the OUTPUT resolution (RCAS crisps the final image instead of being blurred up
    #      by the upscaler);
    #   3. expand to HDR last (sharpening stays in SDR, where RCAS's luma weighting is valid).
    # VSR and TrueHDR are separate RTX passes (see rtxvideo.py) so RCAS can sit between them.
    if UPSCALE:
        t = _upscale(t, OUT_W, OUT_H)     # RTX VSR (_RTX.run_vsr) or bicubic; -> OUT_W x OUT_H
    if SHARPEN > 0:
        t = _rcas(t, SHARPEN)             # FSR-style RCAS sharpen at the output res (GUI "FSR"); see _rcas
    if HDR_ACTIVE:
        # RTX TrueHDR: SDR -> packed 10-bit BT.2020 PQ RGB (x2rgb10le) at OUT_W x OUT_H. This drives
        # the HDR encode path below, so it bypasses the SDR quantise.
        return _RTX.run_hdr(t)
    # Round to nearest, not truncate: numpy's float->uint cast floors, which biases every frame
    # ~0.5 LSB low (a uniform darkening, and the wrong quantisation of the model output). Rounding
    # is the unbiased mapping back to integer samples; every emitted frame goes through here.
    a = (t[0] * MAXV).round().clamp(0, MAXV).permute(1, 2, 0).contiguous().cpu().numpy()  # CHW->HWC on GPU
    return a.astype(NP_DT).tobytes()

def read_exact(stream, nbytes):
    buf = bytearray()
    while len(buf) < nbytes:
        chunk = stream.read(nbytes - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)

dec = subprocess.Popen(
    [FFMPEG, "-v", "error", "-i", inp, "-f", "rawvideo", "-pix_fmt", DEC_FMT, "-"],
    stdout=subprocess.PIPE, creationflags=NO_WINDOW)

# Always encode HEVC, whatever the source codec is. The interpolated clip is a brand new artifact
# (many times the source's frame count), and HEVC at the same visually lossless CQ is far smaller
# than H.264, so echoing an H.264 source would roughly quadruple the file for no quality gain. It
# is also what the polished interpolation apps steer toward (enhancr, Topaz, Flowframes all offer
# a codec menu and lean on HEVC/AV1); GMFSS_Fortuna's own script only dumps mp4v. HEVC carries
# 10 bit (main10) and 4:4:4 cleanly. NVENC is used when the device has a usable HEVC session, with
# the automatic CPU fallback below when it does not.
venc = "hevc_nvenc"

def _enc_works(name):
    # Real availability check: actually open the encoder on a tiny frame. NVENC fails fast
    # here when the device has no usable encode session (no NVIDIA GPU, a GPU too old for
    # this codec, or no driver), which is exactly the case the software fallback covers.
    try:
        return subprocess.run(
            [FFMPEG, "-hide_banner", "-v", "error", "-f", "lavfi",
             "-i", "color=c=black:s=256x256:d=1:r=24", "-frames:v", "1",
             "-c:v", name, "-f", "null", "-"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=NO_WINDOW).returncode == 0
    except Exception:  # noqa: BLE001
        return False

USE_NVENC = _enc_works(venc)
if not USE_NVENC:
    # No usable NVENC on this device: fall back to the best visually lossless software
    # encoder in the bundled (LGPL) ffmpeg. SVT-AV1 has true CRF rate control and clean
    # 8/10 bit support; libx264/libx265 are GPL and not compiled into this build.
    sys.stderr.write(f"NVENC ({venc}) unavailable on this device; "
                     f"falling back to CPU libsvtav1\n"); sys.stderr.flush()
    venc = "libsvtav1"

# Output pixel format: preserve 10 bit and 4:4:4 where the encoder allows it, otherwise the
# standard 8 bit 4:2:0. NVENC takes 10 bit as p010le; SVT-AV1 wants planar yuv420p10le and
# has no 4:4:4 path, so the fallback stays 4:2:0.
if HDR_ACTIVE or TEN_BIT:
    # HDR10 (TrueHDR) is always 10-bit; a 10-bit source is also carried as 10-bit.
    out_pix = "p010le" if USE_NVENC else "yuv420p10le"
elif CHROMA444 and venc in ("h264_nvenc", "hevc_nvenc"):
    out_pix = "yuv444p"
else:
    out_pix = "yuv420p"

# Always visually lossless. NVENC: constant quality VBR around the point the linked H.264
# guide calls visually lossless (CQ 17, CQ 20 for AV1), AQ on, a small chroma QP boost.
# SVT-AV1 CPU fallback: CRF 20 (its visually lossless range) at preset 8, fast enough not
# to starve the GPU frame pipe.
if USE_NVENC:
    cq = "20" if venc == "av1_nvenc" else "17"
    qargs = ["-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", cq, "-b:v", "0",
             "-spatial_aq", "1", "-temporal_aq", "1"]
    if venc in ("h264_nvenc", "hevc_nvenc"):
        qargs += ["-qp_cb_offset", "-2", "-qp_cr_offset", "-2"]
else:
    qargs = ["-crf", "20", "-preset", "8"]

prof = ["-profile:v", "main10"] if ((TEN_BIT or HDR_ACTIVE) and venc == "hevc_nvenc") else []

# Carry the source colour signalling through. NVENC ignores the bare -color_* output flags
# for transfer/primaries (verified: only matrix and range stick), which would strip HDR
# signalling, so the values are stamped onto the frames with setparams before the pixel
# conversion, and the -color_* flags are kept too so the mp4 'colr' atom is written. This
# also makes the RGB -> YUV conversion use the source matrix instead of swscale's guess.
# The values come straight from ffprobe of this same ffmpeg, so they are valid filter input.
if HDR_ACTIVE:
    # TrueHDR output is full-range BT.2020 PQ RGB10 (x2bgr10le). Force HDR10 signalling regardless of
    # the source: stamp BT.2020 / PQ (smpte2084) and convert to limited-range BT.2020 YUV (the matrix
    # is applied to the PQ-encoded signal as-is, no tone map), the same setparams-before-format idiom
    # the SDR path uses to carry an HDR source through.
    sp = ["range=tv", "colorspace=bt2020nc", "color_trc=smpte2084", "color_primaries=bt2020"]
    color = ["-color_range", "tv", "-colorspace", "bt2020nc",
             "-color_trc", "smpte2084", "-color_primaries", "bt2020"]
else:
    sp, color = [], []
    for sp_opt, flag, key in (("range", "-color_range", "color_range"),
                              ("colorspace", "-colorspace", "color_space"),
                              ("color_trc", "-color_trc", "color_transfer"),
                              ("color_primaries", "-color_primaries", "color_primaries")):
        v = _tag(key)
        if v:
            sp.append(f"{sp_opt}={v}")
            color += [flag, v]
# Sharpening (the GUI "FSR" toggle / --sharpen) is applied in-engine on the GPU by _rcas() inside
# to_bytes, NOT here. It uses AMD FidelityFX RCAS, the exact sharpen AMD FSR and Lossless Scaling's
# FSR mode use; ffmpeg only ships the older, blunter `cas`, which over-sharpens fine texture into
# mush at high strength (RCAS limits its lobe to the local min/max and eases off in noisy areas, so
# it crisps edges without destroying texture). So the encode vf is just the colour-tag passthrough.
vf = ",".join((["setparams=" + ":".join(sp)] if sp else []) + [f"format={out_pix}"])

# The piped frames are DEC_FMT (rgb24/rgb48le) normally, but the RTX HDR pass emits packed 10-bit
# BT.2020 PQ RGB, so the encoder input format switches when HDR is active. The TrueHDR output is
# x2rgb10le (B in the low 10 bits): the model's channel 0 is blue, which lands in the low bits of the
# CUDA 101010_2 packing, verified by an R/B swap when read as x2bgr10le.
ENC_IN_FMT = "x2rgb10le" if HDR_ACTIVE else DEC_FMT
enc_cmd = [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", ENC_IN_FMT,
           "-s", f"{OUT_W}x{OUT_H}", "-r", rate_str, "-i", "-", "-i", inp,
           "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "copy",
           "-c:v", venc, "-vf", vf]
enc_cmd += qargs + prof + color + [out_path]
enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, creationflags=NO_WINDOW)
_sharp_note = f"  sharpen(rcas)={SHARPEN:g}" if SHARPEN > 0 else ""
_up_note = f"  upscale={UPSCALE_F:g}x->{OUT_W}x{OUT_H}" if UPSCALE else ""
_hdr_note = "  HDR10(TrueHDR,BT.2020 PQ)" if HDR_ACTIVE else ""
sys.stderr.write(f"encode: {venc} visually-lossless -> {out_pix}  "
                 f"(source {SRC_CODEC or '?'} {SRC_BITS}bit {SRC_PIX}){_sharp_note}{_up_note}{_hdr_note}\n"); sys.stderr.flush()

# Encode on a background thread so ffmpeg writes overlap the next frame's GPU work instead of
# stalling the single pipe. One writer pulling a bounded FIFO preserves frame order, so the
# output bytes are exactly what the serial path produced; the bound caps buffered frames so a
# slow encoder applies backpressure rather than growing memory. On a write failure the writer
# keeps draining (so the producer never blocks on a full queue) and the error is surfaced after join.
wq = queue.Queue(maxsize=8)
_werr = []
def _writer():
    while True:
        buf = wq.get()
        if buf is None:
            break
        if _werr:
            continue
        try:
            enc.stdin.write(buf)
        except Exception as e:  # noqa: BLE001
            _werr.append(e)
wt = threading.Thread(target=_writer, daemon=True)
wt.start()

# Symmetrically, read the decode pipe on its own thread into a bounded queue so the next frame
# is prefetched while the GPU works the current one. Frames stay ordered (one reader, FIFO);
# the bound caps prefetch so a fast decoder applies backpressure. EOF is signalled by None.
rq = queue.Queue(maxsize=8)
def _reader():
    try:
        while True:
            buf = read_exact(dec.stdout, fsize)
            rq.put(buf)
            if buf is None:
                break
    except Exception:  # noqa: BLE001 - pipe closed during shutdown, stop quietly
        pass
rt = threading.Thread(target=_reader, daemon=True)
rt.start()


def _write_hdr10_metadata():
    """Stamp HDR10 static metadata into the finished mp4 (mastering display + content light level).

    The bundled LGPL ffmpeg cannot write it on the hevc_nvenc path (no encoder/BSF option exists),
    so hdr10_meta adds the ISOBMFF boxes directly: the mastering display peak is the TrueHDR target
    (HDR_NITS), and MaxCLL/MaxFALL are measured from the actual frames. With this, one PQ/BT.2020
    file tone-maps correctly on both a 1000-nit and a 400-nit display with no per-display setting.
    Best-effort: a failure logs a note but never fails the render."""
    if not HDR_ACTIVE or not str(out_path).lower().endswith(".mp4"):
        return
    try:
        import hdr10_meta
        cll = int(getattr(_RTX, "maxcll", 0) or 0)
        fall = int(getattr(_RTX, "maxfall", 0) or 0)
        if hdr10_meta.inject_hdr10(out_path, max_nits=HDR_NITS, maxcll=cll, maxfall=fall,
                                   colorspace=HDR_MASTER_PRIM):
            sys.stderr.write(f"HDR10 metadata: mastered {HDR_NITS} nits ({HDR_MASTER_PRIM}), "
                             f"measured MaxCLL {cll} / MaxFALL {fall} nits\n"); sys.stderr.flush()
    except Exception as e:  # noqa: BLE001 - container metadata is a finishing touch, not load-bearing
        sys.stderr.write(f"HDR10 metadata: skipped ({e})\n"); sys.stderr.flush()


# Run the whole interpolation/encode pipeline on one non-default CUDA stream so TRT, softsplat's cupy
# kernel and the torch glue all share it: same-stream ordering then makes each op's output ready for the
# next with no per-call host sync (the old per-engine synchronize in trt_runtime is gone), leaving just
# the implicit drain at each frame's .cpu() download. The non-default stream also keeps TensorRT's
# default-stream warning away. Sync first so model weights and RTX init (issued on the default stream)
# are visible on the new stream.
torch.cuda.synchronize()
_infer_stream = torch.cuda.Stream()
torch.cuda.set_stream(_infer_stream)

if NO_INTERP:
    # Sharpen-only pass: no interpolation, one output frame per source frame at the source fps.
    # Each decoded frame is RCAS-sharpened on the GPU when --sharpen > 0, or passed straight
    # through (a plain re-encode) when it is 0. Shares the same encode pipe/threads as the
    # interpolation path, so colour signalling, bit depth and audio are handled identically.
    k = 0
    try:
        while True:
            buf = rq.get()
            if buf is None:
                break
            # Route through to_bytes (decode->tensor->process->bytes) when there is any per-frame
            # GPU work to do (sharpen and/or upscale); otherwise pass the raw frame straight to the
            # encoder as a plain re-encode.
            wq.put(to_bytes(to_tensor(buf)) if (SHARPEN > 0 or UPSCALE or HDR_ACTIVE) else buf)
            k += 1
            if k % 10 == 0:
                sys.stderr.write(f"PROGRESS {k}/{total_units}\n"); sys.stderr.flush()
    finally:
        wq.put(None)            # sentinel: let the writer drain its queue and exit
        wt.join()
        enc.stdin.close()
        dec.stdout.close()
        enc.wait()
        dec.wait()
    if _werr:
        raise _werr[0]          # surface a failed encode pipe as a nonzero exit
    _write_hdr10_metadata()
    sys.stderr.write(f"done {k} frames "
                     f"({'RCAS-sharpened' if SHARPEN > 0 else 're-encoded'}) -> {out_path}\n")
    sys.exit(0)

prev = rq.get()
if prev is None:
    sys.exit("no frames decoded")
I0 = to_tensor(prev)
k = 0
i = 0
dups = 0
last_out = None         # bytes of the most recent emitted frame, held across the final slot
try:
    while True:
        cur = rq.get()
        if cur is None:
            break
        I1 = to_tensor(cur)
        # Held cels: anime is drawn on twos/threes, so repeated frames decode byte for byte alike.
        # Every timestep between two identical frames renders the same still, so render it once and
        # reuse those bytes for all of this pair's slots; this also avoids the shimmer GMFSS can add
        # on identical input. The bytes compare short circuits, so it is ~free to detect.
        dup = cur == prev
        if dup:
            dups += 1
        if FPS_MODE:
            # Emit the output frames whose time falls in [i, i+1) source frame units, but on a grid
            # shifted by half an output step ((j + 0.5)/ratio, not j/ratio) so no frame lands on a
            # source timestamp. Every frame is therefore an interior blend with the same softness as
            # its neighbours, instead of a sharp source frame that pops; see the module docstring.
            lo = math.ceil(i * ratio - 0.5)
            hi = math.ceil((i + 1) * ratio - 0.5)
            fracs = [(j + 0.5) / ratio - i for j in range(lo, hi)]
            if fracs:
                with amp():
                    reuse = model.reuse(I0, I1, scale)
                    held = to_bytes(model.inference(I0, I1, reuse, 0.5)) if dup else None
                    for fr in fracs:
                        last_out = held if dup else to_bytes(model.inference(I0, I1, reuse, fr))
                        wq.put(last_out)
        else:
            # Multi mode: emit multi frames per source frame on a grid offset by half a step,
            # timesteps 1/2M, 3/2M, ... (2M-1)/2M (symmetric around 0.5, spacing 1/M). None is at 0
            # or 1, so no output frame is a copy (or near copy) of a source frame and the clip shares
            # one look. The matching M frames for the last source frame are the held tail below.
            M = args.multi
            with amp():
                reuse = model.reuse(I0, I1, scale)
                if dup:
                    held = to_bytes(model.inference(I0, I1, reuse, 0.5))
                    last_out = held
                    for _ in range(M):
                        wq.put(held)
                else:
                    for j in range(M):
                        last_out = to_bytes(model.inference(I0, I1, reuse, (2 * j + 1) / (2 * M)))
                        wq.put(last_out)
        prev, I0 = cur, I1
        i += 1
        k += 1
        if k % 10 == 0:
            sys.stderr.write(f"PROGRESS {k}/{total_pairs}\n"); sys.stderr.flush()
    # Closing slot for the last source frame (i is now its index): its own time interval [i, i+1),
    # which has no frame after it to interpolate toward. Hold the last generated frame across it so
    # the output covers the full source duration and lands on exactly multi*frames (true doubling,
    # the target fps behaviour Topaz uses) instead of stopping one slot short. It is held, so it
    # stays soft and does not pop. A single decoded frame has no pair at all, so it just passes
    # through (still routed through to_bytes when sharpening/upscaling so its dims match the encoder).
    if last_out is None:
        wq.put(to_bytes(I0) if (SHARPEN > 0 or UPSCALE or HDR_ACTIVE) else prev)
    else:
        tail = (math.ceil((i + 1) * ratio - 0.5) - math.ceil(i * ratio - 0.5)) if FPS_MODE else args.multi
        for _ in range(tail):
            wq.put(last_out)
finally:
    wq.put(None)            # sentinel: let the writer drain its queue and exit
    wt.join()
    enc.stdin.close()
    dec.stdout.close()
    enc.wait()
    dec.wait()
if _werr:
    raise _werr[0]          # surface a failed encode pipe as a nonzero exit
_write_hdr10_metadata()
sys.stderr.write(f"done {k} pairs ({dups} held as duplicates) -> {out_path}\n")
