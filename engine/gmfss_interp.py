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
  interpolated clip is a new artifact, so matching an H.264 source would only bloat it. Chroma
  and colour signalling are preserved either way; the output is 10 bit by default whatever the
  source depth (see --out-bits below), never less than the source.

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
       [--out-bits {8,10}] [--no-scene-detect] [--no-near-dup]
       --fps overrides <multi>, resampling the timeline to TARGET output fps.
       --sharpen S applies FSR-style RCAS sharpening (strength 0..1) to every output frame to
       offset the uniform-look softness; omit it (or 0) to leave the frames untouched.
       --no-interp skips interpolation entirely: the clip is only re-encoded at its source fps
       with --sharpen applied, for users who just want the sharpening and not the smoothing.
       --no-scene-detect disables hard-cut detection (on by default: a true scene cut is held
       across the boundary instead of interpolated, which would morph the two shots together;
       fast pans are unaffected, see the scene cut detection block below).
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
import time
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
                     "softness; RCAS self-limits, so even 1.0 keeps texture.")
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
ap.add_argument("--codec", choices=["hevc", "av1", "vvc"], default="hevc",
                help="output codec. hevc (default): hevc_nvenc, the smallest widely-supported "
                     "visually lossless choice. av1: av1_nvenc (RTX 40/50 hardware encode). "
                     "vvc: H.266 via CPU libvvenc (best compression, slow, limited player support; "
                     "always 10-bit). The NVENC choices fall back to CPU libsvtav1 when no usable "
                     "session exists; vvc falls back to HEVC if libvvenc is absent.")
ap.add_argument("--no-near-dup", action="store_true",
                help="disable near-duplicate detection. By default a pair whose frames differ only "
                     "by compression noise (anime drawn on twos/threes re-encoded lossily, so held "
                     "cels are no longer byte-identical) is held like an exact duplicate instead of "
                     "interpolated, skipping wasted compute and the shimmer GMFSS adds when it "
                     "interpolates pure noise. Real motion is never held: the detector reacts to "
                     "spatially coherent change (even a tiny blink or a 1px pan), not noise.")
ap.add_argument("--no-scene-detect", action="store_true",
                help="disable hard-cut detection. By default the engine detects true scene cuts "
                     "(via forward/backward flow consistency plus warp residual, reusing the flows "
                     "GMFSS already computes) and holds the boundary frames across the cut instead "
                     "of interpolating, which would morph one shot into the next as a smeared "
                     "ghost. Fast pans and action are not affected: their flow is large but "
                     "consistent, which is exactly what the check separates.")
ap.add_argument("--out-bits", type=int, choices=[8, 10], default=10,
                help="output bit depth. 10 (default): encode 10-bit (HEVC main10 / 10-bit AV1) even "
                     "from an 8-bit source - every emitted frame is computed in floating point, so "
                     "the two extra bits carry real sub-8-bit precision instead of re-quantising the "
                     "blend to 8 bit, which is what bands gradients (skies, glows). Any modern "
                     "HEVC/AV1 decoder plays 10-bit. 8: legacy 8-bit output for maximum device "
                     "compatibility (the only reason to pick it). HDR and VVC are always 10-bit.")
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
ap.add_argument("--hdr-color", choices=["vivid", "rtx", "raw"], default="vivid",
                help="colour handling for --rtx-hdr (the TrueHDR model rotates hues - it greens/cyans the "
                     "blues - even at saturation 0). vivid (default): keep TrueHDR's luminance but take the "
                     "SDR source's hue AND chroma in ICtCp - faithful colour, cyan removed; the SDK "
                     "--hdr-saturation is inert here (the model's chroma is dropped entirely). rtx: drive "
                     "saturation with the SDK --hdr-saturation like real RTX TrueHDR, hue-corrected (chroma "
                     "magnitude from TrueHDR, hue from the source, floored at source), the familiar NVIDIA "
                     "slider without the cyan cast. raw: TrueHDR colour unmodified (a debug/reference mode "
                     "with the cyan cast).")
ap.add_argument("--hdr-vibrance", type=float, default=0.0,
                help="Dynamic Vibrance Intensity for --rtx-hdr (vivid/rtx colour modes): boost muted "
                     "colours without touching already-saturated ones or hue (applied in ICtCp). "
                     "0 (default) = off, 1 = full boost; inert in raw mode.")
ap.add_argument("--hdr-satboost", type=float, default=0.0,
                help="Dynamic Vibrance Saturation boost for --rtx-hdr: uniform extra saturation on top "
                     "of the colour mode (0..1 = +0..100%%, hue-safe in ICtCp). Independent of "
                     "--hdr-saturation, which is RTX HDR's own TrueHDR knob, mirroring NVIDIA's two "
                     "separate filters. 0 (default) = off; inert in raw mode.")
args = ap.parse_args()

inp = os.path.abspath(args.input)
SHARPEN = max(0.0, min(1.0, args.sharpen))   # RCAS strength on every output frame; 0 = off
NO_INTERP = args.no_interp                   # sharpen/re-encode only, no frame generation
SCENE_DETECT = not args.no_scene_detect      # hold frames across true cuts instead of morphing
NEAR_DUP = not args.no_near_dup              # hold noise-only pairs like exact duplicates
UPSCALE_F = max(1.0, min(16.0, args.upscale)) # output spatial upscale factor (clamped); 1.0 = off.
                                             # RTX VSR has no integer-scale limit (probed clean to
                                             # 16K), so any factor is allowed up to a 16x sanity cap
                                             # (16K from 720p); the encoder pick below handles the
                                             # >8192px sizes NVENC cannot encode.
UPSCALE = UPSCALE_F > 1.0
RTX_VSR = args.rtx_vsr                        # use the RTX Video SDK (real RTX VSR) for --upscale
RTX_HDR = args.rtx_hdr                         # convert the output to HDR10 via RTX Video TrueHDR
CODEC = args.codec                             # output codec family: hevc (default) / av1 / vvc
HDR_NITS = max(400, min(2000, args.hdr_nits)) # HDR10 peak luminance (TrueHDR target + metadata)
HDR_SAT = max(0, min(200, args.hdr_saturation))  # TrueHDR Saturation; default 0 = faithful to source
HDR_CON = max(0, min(200, args.hdr_contrast))    # TrueHDR Contrast (100 = SDK neutral)
HDR_MG = max(10, min(100, args.hdr_middlegray))  # TrueHDR MiddleGray midtone anchor (50 = SDK default)
HDR_MASTER_PRIM = args.hdr_mastering_prim         # mdcv mastering-display colorspace (display-p3/dci-p3/bt2020/bt709)
HDR_COLOR = args.hdr_color                         # vivid (default) / faithful / raw colour handling
HDR_VIBRANCE = max(0.0, min(1.0, args.hdr_vibrance))    # vibrance Intensity in ICtCp (0 = off)
HDR_SATBOOST = max(0.0, min(1.0, args.hdr_satboost))    # vibrance uniform Saturation boost (0 = off)

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
TEN_BIT_OUT = args.out_bits >= 10 or TEN_BIT   # 10-bit output (the default; --out-bits 8 = legacy
                                               # compat, honoured for 8-bit sources only: output
                                               # never drops below the source depth)
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

# Decode bit depth follows the source: 8 bit clips decode as rgb24 byte for byte; 10 bit and
# up are carried as 16 bit rgb (rgb48le) so the model (fp16, which holds 10 bit precision)
# never truncates them to 8 bit. The OUTPUT side is decoupled and defaults to 10 bit whatever
# the source (--out-bits): every emitted frame is a floating point blend, so quantising back
# to 8 bit would throw away real sub-8-bit precision the interpolation just created and band
# the gradients. Frames leave to_bytes as 16 bit rgb whenever either side is >8 bit and the
# encoder dithers down to its 10-bit format; --out-bits 8 restores the legacy 8-bit output.
if TEN_BIT:
    DEC_FMT, NP_DT, MAXV, BPP = "rgb48le", np.uint16, 65535.0, 6
else:
    DEC_FMT, NP_DT, MAXV, BPP = "rgb24", np.uint8, 255.0, 3
if TEN_BIT_OUT:
    OUT_RAW_FMT, OUT_NP_DT, OUT_MAXV = "rgb48le", np.uint16, 65535.0
else:
    OUT_RAW_FMT, OUT_NP_DT, OUT_MAXV = "rgb24", np.uint8, 255.0
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
# TrueHDR is an SDR-to-HDR model: feeding it an already-PQ/HLG source would expand PQ-encoded
# pixels it assumes are SDR gamma. Skip the conversion and carry the source HDR through unchanged
# (the SDR colour path below already stamps the source transfer/primaries onto the output).
SRC_HDR_IN = (_tag("color_transfer") or "") in ("smpte2084", "arib-std-b67")
# TrueHDR is also capped at 8192 px: at 16K its ICtCp colour math alone needs ~10 GB of
# intermediate tensors on top of the SDK and VSR buffers, which oversubscribes a 24 GB card.
# On Windows, VRAM oversubscription does not fail cleanly - WDDM starts evicting under load and
# the NVIDIA driver can stall in kernel mode long enough to trip the DPC watchdog (bugcheck
# 0x133, a hard reboot). Refusing up front is the only safe behaviour.
if RTX_HDR and (OUT_W > 8192 or OUT_H > 8192):
    sys.stderr.write(f"[rtx] RTX HDR is limited to 8192px outputs ({OUT_W}x{OUT_H} requested): "
                     "TrueHDR at this size oversubscribes GPU memory, which can hard-crash the "
                     "system (DPC watchdog). Rendering SDR; lower the upscale target to combine "
                     "it with HDR.\n"); sys.stderr.flush()
    RTX_HDR = False
if RTX_HDR and SRC_HDR_IN:
    sys.stderr.write(f"[rtx] source is already HDR (transfer {_tag('color_transfer')}); TrueHDR "
                     "converts SDR only, skipping the HDR conversion (source HDR signalling is "
                     "carried through as-is)\n"); sys.stderr.flush()
    RTX_HDR = False
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
                                 hdr_middlegray=HDR_MG, hdr_color=HDR_COLOR, hdr_vibrance=HDR_VIBRANCE,
                                 hdr_satboost=HDR_SATBOOST)
        RTX_VSR_ACTIVE = _need_vsr
        HDR_ACTIVE = _need_hdr
        if _need_vsr:
            sys.stderr.write(f"RTX Video Super Resolution ready (Ultra) -> {OUT_W}x{OUT_H}\n")
        if _need_hdr:
            _vib = f", vib {HDR_VIBRANCE:g}" if HDR_VIBRANCE > 0 else ""
            _sb = f", sb {HDR_SATBOOST:g}" if HDR_SATBOOST > 0 else ""
            sys.stderr.write(f"RTX HDR ready (TrueHDR {HDR_NITS} nits, sat {HDR_SAT}, con {HDR_CON}, "
                             f"mg {HDR_MG}, colour {HDR_COLOR}{_vib}{_sb}) HDR10 (BT.2020 PQ) @ {OUT_W}x{OUT_H}\n")
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

# AMD FidelityFX RCAS, shared with the single-frame preview (preview.py) so the preview shows exactly
# the sharpen a full render applies; the implementation moved verbatim to rcas.py (same folder).
from rcas import rcas as _rcas  # noqa: E402

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

# --- live output preview ----------------------------------------------------------------------
# When SMV_LIVE_PREVIEW is set (the GUI passes a path under its userData dir), a small JPEG of
# the most recently produced frame is dropped there about once a second and the GUI shows it as
# a "what is being written right now" thumbnail. Time-gated, so the cost is one 480p download +
# JPEG encode per second (~ms) and exactly zero when the env var is absent (CLI runs). Written
# tmp-then-replace so the poller never reads a half-written file. The thumbnail shows what the
# OUTPUT will look like: an HDR render's frame is the graded TrueHDR result (unpacked from the
# packed PQ bytes and tonemapped for the sRGB canvas with the SAME source-anchored tonemap the
# before/after pane uses - preview.py's _tonemap - so colour mode, vibrance and contrast all
# show); an HDR SOURCE carried through gets the pane's self-anchored _tonemap_pq (raw PQ code
# values would read flat and washed out). preview.py guards its main(), so it imports clean.
LIVE_PREVIEW = os.environ.get("SMV_LIVE_PREVIEW")
_live_last = 0.0

def _live_small(img):
    """[3,H,W] float in [0,1] -> HxWx3 uint8 numpy, downscaled to <=480 tall (even width)."""
    h, w = img.shape[-2], img.shape[-1]
    if h > 480:
        img = F.interpolate(img[None], size=(480, max(2, round(w * 480 / h / 2) * 2)),
                            mode="bilinear", align_corners=False)[0]
    return (img.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()

_live_q = None      # single-slot handoff to the formatting worker; created on the first tick
_live_thread = None

def _live_worker():
    """Formats and writes the thumbnails off the render thread. The heavy part of a tick is CPU
    formatting (PQ unpack + tonemap + JPEG, ~100 ms for an HDR frame), and the render pipeline is
    GPU-compute-bound with idle cores, so moving it here makes the render-thread cost of a tick
    just the ~ms GPU downscale in _live_small. One worker + a 1-slot queue: ticks that arrive
    while it is busy are simply skipped (put_nowait below), never queued up."""
    import cv2
    import preview as _prev                   # importable: preview.py guards main(); the pane's tonemaps
    while True:
        item = _live_q.get()
        if item is None:                      # flush sentinel: drain finished, exit
            return
        kind, rgb, packed = item
        try:
            if kind == "hdr":
                # Graded HDR frame: unpack run_hdr's x2rgb10le (R>>20, G>>10, B low bits), decode
                # PQ to linear and tonemap anchored to the SDR frame it was graded from, exactly
                # like the before/after pane. Stride-subsample the packed words before any math:
                # the shifts/stack/resize then see only a thumbnail-sized grid (nearest-pixel
                # subsampling is invisible at this size).
                import rtxvideo
                u = np.frombuffer(packed, "<u4").reshape(OUT_H, OUT_W)
                step = max(1, OUT_H // 480)
                if step > 1:
                    u = u[::step, ::step]
                code = np.stack([(u >> 20) & 1023, (u >> 10) & 1023, u & 1023], -1).astype(np.float32) / 1023.0
                if code.shape[0] > 480:
                    code = cv2.resize(code, (max(2, round(code.shape[1] * 480 / code.shape[0] / 2) * 2), 480),
                                      interpolation=cv2.INTER_AREA)
                lin = rtxvideo._pq_to_linear(torch.from_numpy(code)).numpy()
                rgb = _prev._tonemap(lin, rgb)
            elif kind == "pq":
                rgb = _prev._tonemap_pq(rgb)  # PQ source carried through: display-map it like the pane
            tmp = LIVE_PREVIEW + ".tmp.jpg"
            if cv2.imwrite(tmp, np.ascontiguousarray(rgb[:, :, ::-1])):   # RGB -> BGR for cv2
                os.replace(tmp, LIVE_PREVIEW)
        except Exception:  # noqa: BLE001 - a thumbnail must never break (or stall) the render
            pass

def _live_flush():
    """Let the worker finish the tick in flight (and the render's final thumbnail) before exit;
    bounded so a wedged worker can never hold the render open (it is a daemon thread)."""
    if _live_q is not None:
        try:
            _live_q.put(None, timeout=2)
            _live_thread.join(timeout=5)
        except Exception:  # noqa: BLE001
            pass

def _live_preview(t, hdr_packed=None):
    global _live_last, _live_q, _live_thread
    if not LIVE_PREVIEW or time.time() - _live_last < 1.0:
        return
    _live_last = time.time()
    try:
        if _live_q is None:
            _live_q = queue.Queue(maxsize=1)
            _live_thread = threading.Thread(target=_live_worker, daemon=True)
            _live_thread.start()
        kind = "hdr" if hdr_packed is not None else ("pq" if SRC_HDR_IN else "sdr")
        # The render thread only snapshots: a ~480p GPU downscale + download (_live_small) and,
        # for HDR, a reference to the already-host packed bytes. All decoding/tonemapping/JPEG
        # happens on the worker.
        _live_q.put_nowait((kind, _live_small(t[0]), hdr_packed))
    except queue.Full:
        pass               # worker still formatting the previous tick: skip this one
    except Exception:  # noqa: BLE001 - a thumbnail must never break the render
        pass

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
        # the HDR encode path below, so it bypasses the SDR quantise. The live thumbnail is made
        # from the graded output (tonemapped), so the GUI shows the actual HDR look.
        out = _RTX.run_hdr(t)
        _live_preview(t, out)
        return out
    _live_preview(t)                      # GUI live thumbnail (time-gated; no-op without the env var)
    # Round to nearest, not truncate: numpy's float->uint cast floors, which biases every frame
    # ~0.5 LSB low (a uniform darkening, and the wrong quantisation of the model output). Rounding
    # is the unbiased mapping back to integer samples; every emitted frame goes through here.
    # Quantise at the OUTPUT depth (OUT_MAXV, 16 bit unless --out-bits 8 on an 8-bit source), not
    # the decode depth: this is where the float precision either survives into the 10-bit encode
    # or gets flattened to 8-bit steps.
    a = (t[0] * OUT_MAXV).round().clamp(0, OUT_MAXV).permute(1, 2, 0).contiguous().cpu().numpy()  # CHW->HWC on GPU
    return a.astype(OUT_NP_DT).tobytes()

# --- scene cut detection --------------------------------------------------------------------
# Interpolating across a hard cut morphs one shot into the next (a smeared ghost frame), so cut
# pairs hold the boundary frames instead. Detection reuses the flows model.reuse() has already
# computed, so it costs a few grid_samples per pair. A raw pixel difference cannot be the signal
# here: a fast pan also produces a huge frame difference and would be falsely flagged, killing
# interpolation exactly where it is most wanted. Two flow-based checks separate the cases, and
# BOTH must fire (a false cut is worse than a missed one, which just keeps today's behaviour):
#   occ   - forward/backward consistency: for real motion flow01(x) + flow10(x + flow01(x)) ~ 0;
#           on a cut the two flows are unrelated, so most pixels fail the check. A pan fails only
#           in its disocclusion band.
#   photo - warp residual: reconstruct each frame from the other by backward-warping with the
#           matching flow; on a cut even the best flow cannot make the content match.
# Set SMV_SCENE_DEBUG=1 to log both metrics for every pair (used to calibrate the thresholds).
# Calibration (2026-07-02, samples/test.mp4 family; SMV_SCENE_DEBUG sweeps):
#   within-shot anime pairs      occ 0.042..0.063   photo 0.034..0.047
#   fast pan 25 px/frame         occ 0.000          photo ~0.005
#   whip pan 100 px/frame        occ 0.000          photo 0.006..0.022   (flow tracks it: no cut)
#   animated gradient morph      occ <=0.063        photo 0.0002
#   same-shot crop-zoom reframe  occ 0.196          photo 0.056          (a coherent zoom: leave it)
#   true content cut             occ 1.000          photo 0.260          (fires)
# The thresholds sit in that gap with ~8x margin on the false-positive side (occ) and 2..3x on
# the detection side. Raising sensitivity enough to also catch the same-shot reframe would sit
# only ~1.2x above normal-content photo noise, so borderline reframes deliberately interpolate
# (GMFlow matches them and renders a coherent zoom, not a smear).
SCENE_DEBUG = bool(os.environ.get("SMV_SCENE_DEBUG"))
SCENE_OCC_TH = 0.5      # fraction of pixels failing the fwd/bwd consistency check
SCENE_PHOTO_TH = 0.08   # mean abs warp-reconstruction error, [0,1] scale
_scene_grid = None      # cached base pixel grid, [1,h,w,2]; flows keep one shape per run

def _flow_grid(flow):
    """grid_sample coordinates that sample position x + flow(x) (align_corners=True)."""
    global _scene_grid
    _, _, h, w = flow.shape
    if _scene_grid is None or _scene_grid.shape[1:3] != (h, w):
        gy, gx = torch.meshgrid(
            torch.arange(h, device=flow.device, dtype=torch.float32),
            torch.arange(w, device=flow.device, dtype=torch.float32), indexing="ij")
        _scene_grid = torch.stack((gx, gy), dim=-1).unsqueeze(0)
    g = _scene_grid + flow.permute(0, 2, 3, 1)
    return torch.stack((g[..., 0] * (2.0 / (w - 1)) - 1.0,
                        g[..., 1] * (2.0 / (h - 1)) - 1.0), dim=-1)

def _cut_metrics(I0, I1, flow01, flow10):
    """(occ, photo) for one pair; flows are the half-resolution ones out of model.reuse()."""
    f01, f10 = flow01.float(), flow10.float()
    g01, g10 = _flow_grid(f01), _flow_grid(f10)
    f10w = F.grid_sample(f10, g01, mode="bilinear", padding_mode="border", align_corners=True)
    res = (f01 + f10w).square().sum(1).sqrt()
    mag = f01.square().sum(1).sqrt() + f10w.square().sum(1).sqrt()
    # occlusion-style test: inconsistent if the residual exceeds 5% of the motion, floored at
    # 1.5 half-res px so near-static content is not judged on sub-pixel noise
    occ = (res > torch.clamp(0.05 * mag, min=1.5)).float().mean()
    i0h = F.interpolate(I0, scale_factor=0.5, mode="bilinear", align_corners=False).float()
    i1h = F.interpolate(I1, scale_factor=0.5, mode="bilinear", align_corners=False).float()
    r1 = F.grid_sample(i0h, g10, mode="bilinear", padding_mode="border", align_corners=True)
    r0 = F.grid_sample(i1h, g01, mode="bilinear", padding_mode="border", align_corners=True)
    photo = 0.5 * ((r1 - i1h).abs().mean() + (r0 - i0h).abs().mean())
    return occ.item(), photo.item()

def _pair_is_cut(i, I0, I1, reuse):
    """Decide whether source pair i is a hard cut (and log it); False when detection is off."""
    if not SCENE_DETECT:
        return False
    occ, photo = _cut_metrics(I0, I1, reuse[0], reuse[1])
    if SCENE_DEBUG:
        sys.stderr.write(f"SCENE {i} occ={occ:.3f} photo={photo:.4f}\n"); sys.stderr.flush()
    cut = occ > SCENE_OCC_TH and photo > SCENE_PHOTO_TH
    if cut:
        sys.stderr.write(f"cut detected at pair {i} (occ {occ:.2f}, photo {photo:.3f}): "
                         "holding boundary frames instead of interpolating\n"); sys.stderr.flush()
    return cut

# --- near-duplicate detection ---------------------------------------------------------------
# Anime is drawn on twos and threes, so held cels repeat; after a lossy encode the repeats are
# no longer byte-identical (the exact-equality check above misses them) and GMFSS interpolates
# pure compression noise: M inferences per pair wasted, plus a subtle shimmer on what should be
# a rock-still shot. A pair whose difference is noise-only is therefore held like an exact dup.
# The detector must never hold visible motion (that would create judder, the thing the app
# exists to remove), so it keys on spatial coherence, not amount: the SIGNED difference is
# averaged per 16 px block (random encode noise cancels inside a block; real change, however
# small or local - a blink, a 1 px pan - is coherent and survives), and the maximum block wins.
# A plain mean-abs difference would do the opposite: its noise floor never averages away, and a
# tiny blinking dot drowns in it. The threshold is a HARM BOUND rather than a noise/motion
# classifier (at these amplitudes the two are not separable even in principle: an ultra-slow
# morph measures BELOW dither noise): holding emits the pair midpoint, so the worst-case error
# vs true interpolation is half the pair difference, i.e. <= ~1.5 8-bit levels in the single
# most-changed block at this threshold - at or below the source's own quantisation noise,
# invisible whether the sub-threshold change was noise or drift.
# Calibration (2026-07-02, SMV_SCENE_DEBUG sweeps; maxblock values):
#   duplicated cels + temporal noise alls=1/2/3   0.0068..0.0099 / 0.008..0.012 / 0.0075..0.016
#   ultra-slow gradient morph                     ~0.002           (held: harmless per the bound)
#   1 px/frame photo pan, 1 s fade (+ noise)      0.033..0.047 / 0.035..0.052   (~3x margin)
#   24 px blink toggle, normal anime motion       ~0.82 / 0.71..0.85            (60x+ margin)
NEARDUP_TH = 0.012      # max block-mean signed diff, [0,1] scale (~3 8-bit levels)

def _near_dup(i, I0, I1):
    """True when pair i differs only by compression noise; logged under SMV_SCENE_DEBUG."""
    if not NEAR_DUP:
        return False
    d = F.avg_pool2d((I0 - I1).float(), 16).abs().max().item()
    if SCENE_DEBUG:
        sys.stderr.write(f"DUP {i} maxblock={d:.5f}\n"); sys.stderr.flush()
    return d < NEARDUP_TH

def _soft_still(I):
    """One source frame rendered the way held duplicate cels already are: GMFSS on the (I, I)
    pair at t=0.5. Emitting the raw source frame here would pop (sharp against the soft tweens
    around it, see Uniform look); the model's own reconstruction keeps the clip's one look."""
    r = model.reuse(I, I, scale)
    return to_bytes(model.inference(I, I, r, 0.5))

def _emit_cut(I0, I1, fracs):
    """Frames for a cut pair: slots before the boundary (t<0.5) hold shot A's still, the rest
    hold shot B's, so the cut lands sharp between two output frames instead of as a morph."""
    a = b = None
    outs = []
    for fr in fracs:
        if fr < 0.5:
            if a is None:
                a = _soft_still(I0)
            outs.append(a)
        else:
            if b is None:
                b = _soft_still(I1)
            outs.append(b)
    return outs

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

# The output codec never echoes the source: the interpolated clip is a brand new artifact (many
# times the source's frame count), so it gets a modern efficient codec. --codec picks the family:
# HEVC (default; smallest widely-supported visually lossless choice, carries 10 bit and 4:4:4
# cleanly), AV1 (RTX 40/50 have AV1 NVENC hardware), or H.266/VVC (libvvenc on the CPU: the best
# compression of the three, slow, limited player support, always 10-bit main10). Hardware picks
# degrade gracefully: a missing NVENC session falls back to CPU libsvtav1, a missing libvvenc to
# HEVC.

def _enc_works(name, size="256x256", fast=False):
    # Real availability check: actually open the encoder on a frame of the given size. NVENC
    # fails fast here when the device has no usable encode session (no NVIDIA GPU, a GPU too
    # old for this codec, or no driver), which is exactly the case the software fallback
    # covers. With an output-sized `size` this doubles as a resolution-capability probe (the
    # CPU encoders' ceilings are build/machine dependent); `fast` switches to each encoder's
    # fastest preset there, since open/size failures do not depend on the preset.
    try:
        args = [FFMPEG, "-hide_banner", "-v", "error", "-f", "lavfi",
                "-i", f"color=c=black:s={size}:d=1:r=24", "-frames:v", "1"]
        if fast:
            args += ["-vf", "format=yuv420p10le"]
            if name == "libsvtav1":
                args += ["-preset", "12"]
            elif name == "libvvenc":
                args += ["-preset", "faster"]
        return subprocess.run(
            args + ["-c:v", name, "-f", "null", "-"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=NO_WINDOW).returncode == 0
    except Exception:  # noqa: BLE001
        return False

venc = {"av1": "av1_nvenc", "vvc": "libvvenc"}.get(CODEC, "hevc_nvenc")
NVENC_MAX = 8192   # hevc_nvenc AND av1_nvenc refuse anything larger in either dimension (probed
                   # on the RTX 5090: 8192 passes, 8704 "No capable devices"); HEVC as a format
                   # tops out at 8192 anyway, so past this the codec family must change.
if OUT_W > NVENC_MAX or OUT_H > NVENC_MAX:
    # Beyond NVENC: only the CPU encoders can take it, and their ceilings are not fixed numbers
    # (this build's SVT-AV1 passed 12288x6912 and refused 14336x8064; vvenc took 15360x8640), so
    # probe candidates AT the real output size. Order by measured probe cost: above ~90 MP
    # SVT-AV1 is expected to refuse and its failing probe costs ~40 s, while a passing vvenc
    # probe costs ~3 s, so VVC goes first there; below that SVT-AV1 (AV1 plays far more widely
    # than VVC) gets the first shot. --codec vvc keeps VVC first at any size.
    order = ["libvvenc", "libsvtav1"] if (CODEC == "vvc" or OUT_W * OUT_H > 90_000_000) \
        else ["libsvtav1", "libvvenc"]
    sys.stderr.write(f"{OUT_W}x{OUT_H} exceeds the {NVENC_MAX}px NVENC/HEVC limit; probing CPU "
                     "encoders at the output size (one-time, up to ~1 min)...\n"); sys.stderr.flush()
    for cand in order:
        if _enc_works(cand, f"{OUT_W}x{OUT_H}", fast=True):
            venc = cand
            break
    else:
        sys.exit(f"no bundled encoder can encode {OUT_W}x{OUT_H}; lower the upscale target")
    sys.stderr.write(f"encoding {OUT_W}x{OUT_H} with {venc} "
                     f"({'H.266/VVC' if venc == 'libvvenc' else 'AV1'}, CPU)\n"); sys.stderr.flush()

    # RAM preflight (fail closed). The CPU encoders keep dozens of pictures in flight at these
    # sizes. MEASURED IN-PIPELINE at 15360x8640 (per-process RSS during a real render): the
    # encode-side ffmpeg peaks at ~42 GB (vvenc's own ~30 GB working set - GOP/threads knobs do
    # not shrink it; maxparallelframes=2 in qargs below saves ~4 GB at no wall cost - plus
    # ffmpeg's fixed inter-stage frame queues carrying ~0.8 GB raw frames), the engine python
    # ~8 GB, the decode ffmpeg ~5 GB. SVT-AV1 standalone is ~33.5 GB at 12288x6912 (its banner
    # calls >=8K support a work-in-progress). Running out of physical RAM here does not fail
    # cleanly: with a small pagefile the machine reaches commit exhaustion, kernel drivers
    # stall, and the DPC watchdog hard-reboots the box (bugcheck 0x133, observed 2026-07-02 on
    # a 64 GB machine, reproduced under a monitored rerun). Refusing up front with an
    # actionable message is the only safe behaviour.
    def _avail_ram_gb():
        try:
            import ctypes
            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + \
                           [(n, ctypes.c_ulonglong) for n in
                            ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
                             "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]
            ms = _MS()
            ms.dwLength = ctypes.sizeof(_MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            return ms.ullAvailPhys / 1e9
        except Exception:  # noqa: BLE001 - non-Windows or API failure: skip the gate
            return 0.0
    _mp = OUT_W * OUT_H / 1e6
    # Coefficients calibrated on the monitored 15360x8640 run (133 MP): the whole pipeline had
    # consumed ~47 GB of available RAM at 87% progress when the watchdog killed it, so ~48 GB
    # true peak + margin -> 0.36 GB/MP for the vvenc path. SVT-AV1 measures fatter per pixel
    # (33.5 GB standalone at 85 MP before pipeline overhead) -> 0.55 GB/MP. Mid sizes stay
    # practical (9600x5400 -> ~35 GB); true 16K honestly needs a ~64 GB machine with nearly
    # everything closed, or more RAM.
    _need = (0.36 if venc == "libvvenc" else 0.55) * _mp + 6.0
    _avail = _avail_ram_gb()
    if _avail and _avail < _need:
        sys.exit(f"{OUT_W}x{OUT_H} needs ~{_need:.0f} GB of free RAM (the CPU encoder alone "
                 f"holds ~40 GB of frames in flight at this size) but only {_avail:.0f} GB is "
                 "available. Close other applications, lower the upscale target, or run on a "
                 "machine with more memory; proceeding anyway can freeze or hard-crash the "
                 "whole system (DPC watchdog).")
    sys.stderr.write(f"RAM preflight: ~{_need:.0f} GB needed, {_avail:.0f} GB available\n")
    sys.stderr.flush()
elif venc == "libvvenc" and not _enc_works(venc):
    sys.stderr.write("libvvenc (H.266/VVC) unavailable in this ffmpeg; using HEVC instead\n")
    sys.stderr.flush()
    venc = "hevc_nvenc"
USE_NVENC = venc.endswith("_nvenc")
if USE_NVENC and not _enc_works(venc):
    # No usable NVENC session for this codec on this device: fall back to the best visually
    # lossless software encoder in the bundled (LGPL) ffmpeg. SVT-AV1 has true CRF rate control
    # and clean 8/10 bit support; libx264/libx265 are GPL and not compiled into this build.
    sys.stderr.write(f"NVENC ({venc}) unavailable on this device; "
                     f"falling back to CPU libsvtav1\n"); sys.stderr.flush()
    venc = "libsvtav1"
    USE_NVENC = False

# Output pixel format: 10 bit by default (--out-bits), preserving 4:4:4 where the encoder
# allows it. NVENC takes 10 bit as p010le (4:2:0) or yuv444p16le (4:4:4, rext profile; the
# high 10 of 16 bits are used); SVT-AV1 wants planar yuv420p10le and has no 4:4:4 path, so
# the CPU fallback stays 4:2:0.
if venc == "libvvenc":
    out_pix = "yuv420p10le"                  # libvvenc's only supported input format (always main10)
elif HDR_ACTIVE:
    # HDR10 (TrueHDR) is always 10-bit 4:2:0 regardless of --out-bits.
    out_pix = "p010le" if USE_NVENC else "yuv420p10le"
elif CHROMA444 and venc in ("h264_nvenc", "hevc_nvenc"):
    out_pix = "yuv444p16le" if TEN_BIT_OUT else "yuv444p"
elif TEN_BIT_OUT:
    out_pix = "p010le" if USE_NVENC else "yuv420p10le"
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
elif venc == "libvvenc":
    # vvenc has no CRF; QP 21 with its perceptual QP adaptation (qpa, on by default) sits in the
    # visually lossless range, and the fast preset keeps 1080p from being glacial on the CPU.
    qargs = ["-qp", "21", "-preset", "fast"]
    if OUT_W > NVENC_MAX or OUT_H > NVENC_MAX:
        # Ultra sizes: cap frame-level parallelism. Measured at 15360x8640 (24 frames): default
        # peaks ~34 GB RAM, maxparallelframes=2 peaks ~30 GB at the SAME wall time.
        qargs += ["-vvenc-params", "maxparallelframes=2"]
else:
    qargs = ["-crf", "20", "-preset", "8"]

# HEVC profile follows the pixel format: main10 for 10-bit 4:2:0, rext (range extensions) for
# 4:4:4 at 10 bit. Other encoders pick their profile from the input format on their own.
if venc == "hevc_nvenc" and out_pix == "yuv444p16le":
    prof = ["-profile:v", "rext"]
elif venc == "hevc_nvenc" and out_pix == "p010le":
    prof = ["-profile:v", "main10"]
else:
    prof = []

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
# FSR mode use (it limits its lobe to the local min/max and eases off in noisy areas, so it crisps
# edges without destroying texture). So the encode vf is just the colour-tag passthrough.
vf = ",".join((["setparams=" + ":".join(sp)] if sp else []) + [f"format={out_pix}"])

# The encoder pipe format. The RTX HDR pass emits packed 10-bit BT.2020 PQ RGB (x2rgb10le: B in
# the low 10 bits of the CUDA 101010_2 packing - the model's channel 0 is blue, verified by an
# R/B swap when read as x2bgr10le). Otherwise frames come out of to_bytes at the OUTPUT raw
# depth (OUT_RAW_FMT). The one path that bypasses to_bytes - the plain no-interp re-encode with
# no sharpen/upscale - pipes the decoded bytes straight through at DEC_FMT; the encoder's format
# filter still raises an 8-bit source to the 10-bit out_pix there, which softens encoder-side
# banding even though a plain re-encode has no float precision to preserve.
if HDR_ACTIVE:
    ENC_IN_FMT = "x2rgb10le"
elif NO_INTERP and not (SHARPEN > 0 or UPSCALE):
    ENC_IN_FMT = DEC_FMT
else:
    ENC_IN_FMT = OUT_RAW_FMT
# At ultra sizes, strip ffmpeg's input-side buffering: the demux thread queue defaults to 8
# packets and the rawvideo decoder frame-threads across every core, each slot holding a full
# raw frame - invisible at 1080p (~50 MB total) but ~0.8 GB per slot at 16K, i.e. tens of GB
# of silent pooling on top of the encoder's own ~30 GB working set (measured: the encode-side
# ffmpeg peaked at 42.8 GB in-pipeline vs 29.8 GB standalone before this cap).
_TQ = ["-threads", "1", "-thread_queue_size", "1"] if (OUT_W > NVENC_MAX or OUT_H > NVENC_MAX) else []
enc_cmd = [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", ENC_IN_FMT,
           "-s", f"{OUT_W}x{OUT_H}", "-r", rate_str] + _TQ + ["-i", "-", "-i", inp,
           "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "copy",
           "-c:v", venc, "-vf", vf]
# VVC-in-MP4 muxing is gated behind -strict experimental on some ffmpeg versions; harmless otherwise.
enc_cmd += qargs + prof + color + (["-strict", "experimental"] if venc == "libvvenc" else []) + [out_path]
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
# keeps draining (so the producer never blocks on a full queue) and the error is surfaced after
# join. The bound is byte-aware: 8 frames was sized for ~10 MB frames, but a 16K rgb48le frame
# is ~0.8 GB, so cap the buffered bytes at ~1.5 GB instead of a fixed frame count.
_ENC_BPP = 4 if ENC_IN_FMT == "x2rgb10le" else (6 if ENC_IN_FMT == "rgb48le" else 3)
wq = queue.Queue(maxsize=max(1, min(8, int(1.5e9 // max(1, OUT_W * OUT_H * _ENC_BPP)))))
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
# Byte-aware like wq, for very large sources.
rq = queue.Queue(maxsize=max(2, min(8, int(1.5e9 // max(1, fsize)))))
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
    _live_flush()               # let the final live thumbnail land before exit
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
neardups = 0            # subset of dups: noise-only pairs caught by _near_dup, not byte equality
cuts = 0                # hard cuts held instead of interpolated (see scene cut detection)
last_out = None         # bytes of the most recent emitted frame, held across the final slot
try:
    while True:
        cur = rq.get()
        if cur is None:
            break
        I1 = to_tensor(cur)
        # Held cels: anime is drawn on twos/threes, so repeated frames decode byte for byte alike
        # (exact bytes compare, ~free) or, after a lossy re-encode, alike up to compression noise
        # (_near_dup). Every timestep between two such frames renders the same still, so render it
        # once and reuse those bytes for all of this pair's slots; this also avoids the shimmer
        # GMFSS can add on (near-)identical input.
        dup = cur == prev
        if not dup and _near_dup(i, I0, I1):
            dup = True
            neardups += 1
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
                    if dup:
                        held = to_bytes(model.inference(I0, I1, reuse, 0.5))
                        for _ in fracs:
                            last_out = held
                            wq.put(held)
                    elif _pair_is_cut(i, I0, I1, reuse):
                        cuts += 1
                        for out in _emit_cut(I0, I1, fracs):
                            last_out = out
                            wq.put(out)
                    else:
                        for fr in fracs:
                            last_out = to_bytes(model.inference(I0, I1, reuse, fr))
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
                elif _pair_is_cut(i, I0, I1, reuse):
                    cuts += 1
                    for out in _emit_cut(I0, I1, [(2 * j + 1) / (2 * M) for j in range(M)]):
                        last_out = out
                        wq.put(out)
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
    # through (still routed through to_bytes when sharpening/upscaling changes its dims, or when
    # the pipe carries the output depth and the raw decode bytes would be the wrong size).
    if last_out is None:
        wq.put(to_bytes(I0) if (SHARPEN > 0 or UPSCALE or HDR_ACTIVE
                                or DEC_FMT != OUT_RAW_FMT) else prev)
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
_live_flush()               # let the final live thumbnail land before exit
sys.stderr.write(f"done {k} pairs ({dups} held as duplicates ({neardups} near), {cuts} cuts held) "
                 f"-> {out_path}\n")
