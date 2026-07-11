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

Uniform look, on the source grid (GMFSS integer --multi, the default GUI path): the FIRST and LAST
output frames ARE the real source frames (there is nothing before frame 0 / after frame N-1 to
interpolate from, so they are kept pristine), and every INTERIOR output frame is model-generated so
the whole interior carries one consistent softness. The problem being solved is the passthrough
shimmer: interleaving byte-exact source frames (sharp) with softer tweens makes fine detail snap in
and out every Nth frame (sharp, soft, sharp ...), a periodic pop that breaks immersion (it is what
RIFE/DAIN do). So no interior source frame is passed through pristine. The subtle case is the
interior slot that lands ON a source timestamp t=k: it is NOT frame k pristine, and NOT frame k
re-run through the model at t=0 (measured: t=0 reconstructs about as sharply as the original, so
the pop survives) - it is the BRACKET MIDPOINT inference(f[k-1], f[k+1], 0.5), a genuine
interpolation that lands at k's instant with the same generated softness as the tweens around it
(it spans 2x the per-pair motion, so it is the interior's softest slot, but temporally correct and
visually consistent - the design goal). The interior tweens sit at on-grid timesteps j/M within
each source pair. Output frame count is multi*(N-1)+1 (2N-1 at 2x): the honest count for reaching
the target fps with real endpoints, so a 2-frame clip at 2x is 3 frames (real, tween, real). This
matches how RIFE/DAIN report 2N-1; duration is ~(M-1)/(M*fps) shorter than the source (the last
frame has no slot after it to fill). See the on-grid loop near the bottom of this file.

Legacy off-grid look (GMFSS --fps mode): here NO emitted frame sits on a source
timestamp - the grid is shifted by half an output step (timesteps 1/2M, 3/2M ... (2M-1)/2M for
integer M; the analogous offset in --fps mode), so every frame incl. the first/last is a generated
interior blend and the last source frame's slot is filled by holding the last generated frame.
Count is multi*frames (true doubling), duration matches the source. This is the older "generate
every displayed frame, never pass a real one through" scheme (Lossless-Scaling style); the on-grid
path above supersedes it for the common GMFSS multiplier case because real endpoints + correctly
timed on-grid interior frames read as higher quality than an all-synthetic half-step-shifted grid.

Usage: gmfss_interp.py <input> <multi> [output] [--scale 1.0] [--fps TARGET] [--no-trt]
       [--sharpen S] [--no-interp] [--upscale F] [--rtx-vsr] [--rtx-hdr] [--hdr-nits N]
       [--out-bits {8,10}] [--restore]
       --fps overrides <multi>, resampling the timeline to TARGET output fps.
       --sharpen S applies FSR-style RCAS sharpening (strength 0..1) to every output frame to
       offset the uniform-look softness; omit it (or 0) to leave the frames untouched.
       --no-interp skips interpolation entirely: the clip is only re-encoded at its source fps
       with --sharpen applied, for users who just want the sharpening and not the smoothing.
       --restore runs Real-ESRGAN's anime-video model (bundled realesr-animevideov3) on every
       output frame to clean noise and redraw linework (fine texture can flatten), before the
       upscale (without RTX VSR its 4x output directly feeds the upscale). Off by default;
       works with --no-interp too.
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
ap.add_argument("--scale", type=float, default=None,
                help="optical-flow resolution factor (GMFlow already runs at half the source "
                     "resolution; this scales it further, flow is upsampled back afterwards). "
                     "Default: AUTO - 1.0 below 4K, 0.5 for 4K+ sources, where flow at quarter "
                     "resolution still carries 1080p-class motion detail and roughly quarters the "
                     "dominant GMFlow cost (the standard GMFSS practice for UHD). Pass an explicit "
                     "value to override the auto rule.")
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
ap.add_argument("--dv", action="store_true",
                help="also export a Dolby Vision Profile 8.1 MP4. Requires --rtx-hdr (the HDR10 render "
                     "is the DV base layer), an MP4 output, and the bundled dovi_tool in "
                     "engine/dvtools. Collects per-frame DV metadata (L1) from the HDR frames, builds "
                     "the RPU with dovi_tool, muxes with the bundled ffmpeg and writes the DV "
                     "configuration box in-engine (no GPAC/MP4Box). The base stays HDR10, so non-DV "
                     "players fall back to HDR10. Skipped with a notice if any requirement is missing.")
ap.add_argument("--hdr10plus", action="store_true",
                help="also embed HDR10+ (SMPTE ST 2094-40) dynamic metadata into the HDR10 render. "
                     "Requires --rtx-hdr, HEVC, an MP4 output, and the user-installed hdr10plus_tool "
                     "in engine/hptools. Collects per-frame brightness statistics (maxSCL, average "
                     "and a maxRGB percentile distribution) from the HDR frames and injects the SEI "
                     "with hdr10plus_tool after the encode. The base stays HDR10, so players without "
                     "HDR10+ fall back to HDR10; combinable with --dv (both metadata ride the same "
                     "stream). Skipped with a notice if any requirement is missing.")
ap.add_argument("--codec", choices=["hevc", "av1", "vvc"], default="hevc",
                help="output codec. hevc (default): hevc_nvenc, the smallest widely-supported "
                     "visually lossless choice. av1: av1_nvenc (RTX 40/50 hardware encode). "
                     "vvc: H.266 via CPU libvvenc (best compression, slow, limited player support; "
                     "always 10-bit). The NVENC choices fall back to CPU libsvtav1 when no usable "
                     "session exists; vvc falls back to HEVC if libvvenc is absent.")
ap.add_argument("--restore", action="store_true",
                help="AI detail restoration: run Real-ESRGAN's anime-video model "
                     "(realesr-animevideov3, bundled - see realesr.py) on every output frame "
                     "to clean compression noise and redraw the linework that interpolation "
                     "and lossy sources soften (a generative repaint - fine texture can "
                     "flatten). Runs before the upscale, so RTX VSR receives the "
                     "restored frame; without RTX VSR the model's own 4x output directly "
                     "feeds the upscale. Works with --no-interp too (restore without "
                     "smoothing). Off by default; adds a second model pass per OUTPUT frame "
                     "(TensorRT-cached; roughly +50%% wall on a 2x 1080p render, more at "
                     "higher multipliers since every emitted frame pays it).")
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
                     "colours without touching already-saturated ones or hue (applied in ICtCp; the "
                     "boost is luminance-coupled - full in midtones, eased in shadows/highlights - so "
                     "bright lights stay light instead of turning neon). "
                     "0 (default) = off, 1 = full boost; inert in raw mode.")
ap.add_argument("--hdr-satboost", type=float, default=0.0,
                help="Dynamic Vibrance Saturation boost for --rtx-hdr: extra saturation on top "
                     "of the colour mode (0..1 = +0..100%%, hue-safe in ICtCp, luminance-coupled like "
                     "--hdr-vibrance). Independent of "
                     "--hdr-saturation, which is RTX HDR's own TrueHDR knob, mirroring NVIDIA's two "
                     "separate filters. 0 (default) = off; inert in raw mode.")
args = ap.parse_args()

inp = os.path.abspath(args.input)
SHARPEN = max(0.0, min(1.0, args.sharpen))   # RCAS strength on every output frame; 0 = off
NO_INTERP = args.no_interp                   # sharpen/re-encode only, no frame generation
UPSCALE_F = max(1.0, min(16.0, args.upscale)) # output spatial upscale factor (clamped); 1.0 = off.
                                             # RTX VSR has no integer-scale limit (probed clean to
                                             # 16K), so any factor is allowed up to a 16x sanity cap
                                             # (16K from 720p); the encoder pick below handles the
                                             # >8192px sizes NVENC cannot encode.
UPSCALE = UPSCALE_F > 1.0
RTX_VSR = args.rtx_vsr                        # use the RTX Video SDK (real RTX VSR) for --upscale
RTX_HDR = args.rtx_hdr                         # convert the output to HDR10 via RTX Video TrueHDR
DV_EXPORT = args.dv                            # also emit a Dolby Vision Profile 8.1 MP4 (needs --rtx-hdr)
HP_EXPORT = args.hdr10plus                     # also embed HDR10+ dynamic metadata (needs --rtx-hdr)
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
         "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,codec_name,pix_fmt,"
         "bits_per_raw_sample,color_space,color_transfer,color_primaries,color_range",
         "-of", "json", path], text=True, creationflags=NO_WINDOW)
    st = (json.loads(out).get("streams") or [{}])[0]
    w, h = int(st["width"]), int(st["height"])
    num, den = (str(st.get("r_frame_rate") or "0/1").split("/") + ["1"])[:2]
    nb = int(st["nb_frames"]) if str(st.get("nb_frames") or "").isdigit() else 0
    return w, h, int(num), int(den or "1"), nb, st

W, H, num, den, NB, ST = probe(inp)

# VFR sources (phone recordings, screen captures, stream rips): r_frame_rate is the container's
# NOMINAL rate - for VFR it is typically the maximum instantaneous rate (often the timebase, e.g.
# 1000/1), not the real pace - while avg_frame_rate is total_frames/duration, the rate the stream
# actually plays at. The raw decode pipe emits frames 1:1 with no timing, and the encoder re-times
# them as CFR at multi*rate, so deriving the rate from a wild r_frame_rate would change the
# duration and desync every passthrough audio track. Detect the mismatch and (a) take avg as the
# source rate, (b) have ffmpeg dup/drop to a constant avg rate on decode (-fps_mode cfr) so the
# frame stream matches that rate exactly. CFR sources: avg == r, nothing changes.
VFR_DEC = []
_an, _ad = (str(ST.get("avg_frame_rate") or "0/1").split("/") + ["1"])[:2]
_an, _ad = int(_an or "0"), int(_ad or "1")
if _an > 0 and _ad > 0 and num > 0 and abs(_an / _ad - num / den) / (num / den) > 0.005:
    sys.stderr.write(f"VFR source: container rate {num}/{den} ({num / den:g} fps) but the stream "
                     f"averages {_an}/{_ad} ({_an / _ad:g} fps); decoding at the constant average "
                     "rate so duration and audio sync are preserved\n"); sys.stderr.flush()
    num, den = _an, _ad
    VFR_DEC = ["-fps_mode", "cfr", "-r", f"{num}/{den}"]


def _power_notice():
    """One-line heads-up when the GPU is power-limited (a laptop Silent/quiet profile): the
    pipeline is GPU-compute-bound at 98-99% utilisation (measured 2026-07-08), so a 55 W cap on a
    175 W part multiplies wall time directly - and the cap is invisible unless someone thinks to
    check nvidia-smi. Reads the driver's power limits once at startup; best-effort (no nvidia-smi,
    no NVIDIA GPU, or N/A fields all stay silent), never affects the render."""
    try:
        q = subprocess.check_output(["nvidia-smi", "-q", "-d", "POWER"], text=True,
                                    creationflags=NO_WINDOW, timeout=5)
        vals = {}
        for line in q.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                k, v = k.strip(), v.strip().split(" ")[0]
                if k in ("Current Power Limit", "Default Power Limit", "Max Power Limit") \
                        and k not in vals:
                    try:
                        vals[k] = float(v)
                    except ValueError:
                        pass
        cur, dflt, mx = (vals.get(k) for k in
                         ("Current Power Limit", "Default Power Limit", "Max Power Limit"))
        if cur and dflt and cur < 0.9 * dflt:
            sys.stderr.write(
                f"note: GPU power limit is {cur:.0f} W (board default {dflt:.0f} W"
                + (f", max {mx:.0f} W" if mx else "") + ") - a quiet/Silent power profile is "
                "active, so this render runs proportionally slower; switch the Windows/vendor "
                "power mode for full speed\n")
            sys.stderr.flush()
    except Exception:  # noqa: BLE001 - a missing/odd nvidia-smi must never break a render
        pass


_power_notice()

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
# --- track passthrough (translations) --------------------------------------------------------
# Older builds output mp4 with only the first audio track: every subtitle track, extra audio
# language, chapter list and font attachment was silently dropped. Now they are all copied
# through unconditionally. Container rule: mp4 cannot hold styled ASS/PGS subtitles or some
# audio codecs, so the DEFAULT output name switches to .mkv whenever the source has subtitles or
# mp4-incompatible audio; an EXPLICIT output path keeps its extension and the mapping adapts
# (subtitles/exotic audio are dropped from an explicit .mp4, with a notice). mov_text subtitles
# (mp4-native) cannot be *copied* into mkv, so those streams are converted to SRT.
MP4_AUDIO_OK = {"aac", "ac3", "eac3", "mp3", "alac", "opus", "flac"}   # flac/opus verified in this build
MKV_SUB_COPY_OK = {"ass", "ssa", "subrip", "srt", "hdmv_pgs_subtitle", "dvd_subtitle", "webvtt"}
AUD_STREAMS, SUB_STREAMS, HAS_ATTACH = [], [], False   # (input-1 absolute index, codec) per stream
try:
    _ts = json.loads(subprocess.check_output(
        [FFPROBE, "-v", "error", "-show_entries", "stream=index,codec_type,codec_name",
         "-of", "json", inp], text=True, creationflags=NO_WINDOW)).get("streams", [])
    for _s in _ts:
        _ty, _c = _s.get("codec_type", ""), (_s.get("codec_name") or "").lower()
        if _ty == "audio":
            AUD_STREAMS.append((_s.get("index"), _c))
        elif _ty == "subtitle":
            SUB_STREAMS.append((_s.get("index"), _c))
        elif _ty == "attachment":
            HAS_ATTACH = True
except Exception:  # noqa: BLE001 - probe failure: fall back to no extra tracks
    pass
NEED_MKV = bool(SUB_STREAMS) or any(c not in MP4_AUDIO_OK for _, c in AUD_STREAMS)
out_path = os.path.abspath(args.output) if args.output else \
    os.path.splitext(inp)[0] + (("_sharpened" if NO_INTERP else f"_{out_label}fps")
                                + (".mkv" if NEED_MKV else ".mp4"))
OUT_IS_MKV = out_path.lower().endswith(".mkv")
# Render into a sibling ".part" file and promote it with os.replace only at success (_finish): a
# cancelled/crashed/failed render leaves <name>.part.<ext> behind instead of a silently truncated
# file at the final path, and an existing good file at out_path is never destroyed until the new
# render actually completed. The GUI deletes the .part remnant after a Cancel.
_ob, _oe = os.path.splitext(out_path)
WORK_PATH = _ob + ".part" + _oe

# --- crash/exit resume ------------------------------------------------------------------------
# Resumable renders (hevc/av1 families) encode the video ALONE into a FRAGMENTED-MP4 stage-1
# file: unlike a plain mp4 (whose moov index is only written at the end, so a killed encoder
# leaves an unreadable file), an fmp4 stays readable up to the last complete moof/mdat fragment,
# and frag_keyframe puts those boundaries exactly on encoder keyframes. (Matroska would also
# survive truncation but quantises timestamps to 1 ms, which jitters high-fps timing; fmp4
# keeps the exact stream timescale.) A tiny sidecar json carries the settings signature (plus
# the HDR light-level maxima, which live in engine state and would otherwise be lost). On the
# next run with the SAME source and settings the engine salvages the partial video, trims it
# back to the last encoder keyframe (a clean closed-GOP stream-copy cut), maps that
# output-frame index back onto the source pair/slot grid (closed form in all three loop
# modes), and continues from there; _finalize_output concatenates the banked prefix with the
# continuation stream before the usual track remux. All codecs are resumable; vvc additionally
# stays fragmented through every mp4 step (see FRAG_COPY below). The legacy direct-encode
# paths survive only behind the SMV_NO_RESUME env opt-out.
VID_PART = WORK_PATH + ".video.mp4"       # stage-1 video-only stream (the banked prefix on resume)
VID_PART2 = WORK_PATH + ".video2.mp4"     # continuation stream written by a resumed run
VID_FULL = WORK_PATH + ".videofull.mp4"   # concat of the two, made at finalize
RESUME_JSON = WORK_PATH + ".resume.json"  # settings signature + HDR maxima sidecar
RESUMABLE = not os.environ.get("SMV_NO_RESUME")
FRAG_FLAGS = ["-movflags", "+frag_keyframe+empty_moov+default_base_moof"]
# VVC must stay FRAGMENTED through every mp4 it touches: this ffmpeg's REGULAR-mp4 muxer mangles
# vvc composition offsets on stream copy (a plain `-c copy` remux of a clean 96-frame vvc file
# left only 73 decodable frames with negative/disordered pts; measured 2026-07-10), while
# frag->frag and ->mkv copies roundtrip losslessly (the moof/trun offset path is correct, only
# the ctts path is broken). So for vvc, every intermediate copy AND a final .mp4 target are
# written fragmented (fmp4 = the CMAF/DASH layout; fine in modern players, and vvc playback is
# specialist territory anyway). The HDR10 box injector was verified on fragmented mp4 too
# (default_base_moof keeps moof/mdat offsets self-relative, so growing the init moov is safe).
FRAG_COPY = args.codec == "vvc"
RESUME_ACTIVE = False       # this run continues a previous partial render
RESUME_OUT_BASE = 0         # output frames already banked in VID_PART
RESUME_SKIP_SRC = 0         # source frames the decoder must skip
RESUME_PAIR_SKIP = 0        # already-banked output slots of the first resumed pair
RESUME_BASE_BYTES = 0       # size of the banked prefix, for the SIZE projection


def _resume_sig():
    """Settings signature guarding resume: every CLI arg plus the source file identity (and the
    hidden SMV_CQ knob, which changes the bitstream). Any mismatch means the partial video was
    rendered with different settings and must be discarded, not continued."""
    import hashlib
    d = dict(vars(args))
    # Normalize the path args: the same file passed with different slash styles or casing (GUI vs
    # shell) must not read as different settings.
    d["input"] = os.path.normcase(os.path.abspath(args.input))
    d["output"] = os.path.normcase(os.path.abspath(args.output)) if args.output else None
    d["__src"] = [d["input"], os.path.getsize(inp), int(os.path.getmtime(inp))]
    d["__cq"] = os.environ.get("SMV_CQ") or ""
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()


RESUME_SIG = _resume_sig()


def _resume_cleanup():
    """Drop every resume artifact (stale settings, unusable salvage, or normal end-of-run)."""
    for _p in (VID_PART, VID_PART2, VID_FULL, RESUME_JSON, WORK_PATH + ".concat.txt",
               WORK_PATH + ".salv.mp4", WORK_PATH + ".salv2.mp4", WORK_PATH + ".trim.mp4"):
        try:
            os.remove(_p)
        except OSError:
            pass


def _write_resume_sidecar(pair, total):
    """Refresh the resume sidecar: the settings signature (validated on the next run), the
    chosen encoder (a continuation stream MUST come from the same encoder or the concat would
    mix incompatible bitstreams), the pair counter (the GUI's "will resume from" hint), and the
    HDR light-level running maxima (engine state that would die with the process). Atomic
    replace so a kill mid-write can't leave a torn json."""
    _rtx = globals().get("_RTX")
    try:
        _tmp = RESUME_JSON + ".tmp"
        with open(_tmp, "w", encoding="utf-8") as _f:
            json.dump({"sig": RESUME_SIG, "pair": pair, "total": total,
                       "venc": globals().get("venc", ""),
                       "maxcll": float(getattr(_rtx, "maxcll", 0) or 0),
                       "maxfall": float(getattr(_rtx, "maxfall", 0) or 0)}, _f)
        os.replace(_tmp, RESUME_JSON)
    except OSError:
        pass

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
    OUT_RAW_FMT, OUT_MAXV, OUT_TORCH_DT = "rgb48le", 65535.0, torch.uint16
else:
    OUT_RAW_FMT, OUT_MAXV, OUT_TORCH_DT = "rgb24", 255.0, torch.uint8
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
# Deterministic renders (2026-07-09): benchmark mode re-times conv algorithms per process, so two
# processes could pick different reduction orders and produce (invisibly) different frames. With
# it off + deterministic algorithm selection, and softsplat's fixed-point accumulation (the one
# true nondeterminism source, see softsplat.py), identical runs produce byte-identical output
# files on the GMFSS path - which is what lets smoke.py assert md5s. Perf measured neutral on the
# eager path (the TRT default path never used cudnn autotuning anyway).
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if NO_INTERP:
    # Sharpen-only / re-encode: the GMFSS model and the TensorRT backend are never loaded, so
    # there is no warmup and no first-run engine build. Each frame just gets the RCAS pass.
    model = None
    sys.stderr.write("no-interp mode: GMFSS interpolation disabled "
                     "(re-encode at source fps with optional FSR sharpen)\n"); sys.stderr.flush()
else:
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
# Dolby Vision export (--dv) sits on top of the HDR10 render: it needs the HDR path active, an MP4
# output, and the user-installed dovi_tool. Decide up front whether to collect per-frame L1 metadata
# during the render (RTXVideo collect_l1); the actual export runs after the encode in _dv_export.
DOVI_EXE = os.path.join(ENGINE_DIR, "dvtools", "dovi_tool.exe")
# DV Profile 8.1 is HEVC-only (the RPU rides in the HEVC bitstream), MP4-only, and needs dovi_tool.
_want_dv = DV_EXPORT and CODEC == "hevc" and not OUT_IS_MKV and os.path.isfile(DOVI_EXE)
# HDR10+ (--hdr10plus) mirrors the DV gating: the ST 2094-40 SEI rides the HEVC stream, the export
# re-muxes through an MP4, and the injector is the user-installed hdr10plus_tool (see _hp_export).
HP_EXE = os.path.join(ENGINE_DIR, "hptools", "hdr10plus_tool.exe")
_want_hp = HP_EXPORT and CODEC == "hevc" and not OUT_IS_MKV and os.path.isfile(HP_EXE)
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
                                 hdr_satboost=HDR_SATBOOST, collect_l1=_want_dv, collect_hp=_want_hp)
        RTX_VSR_ACTIVE = _need_vsr
        HDR_ACTIVE = _need_hdr
        if _need_vsr:
            sys.stderr.write(f"RTX Video Super Resolution ready (Ultra) -> {OUT_W}x{OUT_H}\n")
        if _need_hdr:
            _vib = f", vib {HDR_VIBRANCE:g}" if HDR_VIBRANCE > 0 else ""
            _sb = f", sb {HDR_SATBOOST:g}" if HDR_SATBOOST > 0 else ""
            sys.stderr.write(f"RTX HDR ready (TrueHDR {HDR_NITS} nits, con {HDR_CON}, sat {HDR_SAT} "
                             f"[SDK 0-200, 100=neutral], mg {HDR_MG}, colour {HDR_COLOR}{_vib}{_sb}) "
                             f"HDR10 (BT.2020 PQ) @ {OUT_W}x{OUT_H}\n")
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"[rtx] unavailable, falling back (bicubic upscale / SDR): {repr(e)[:200]}\n")
        _RTX = None
        RTX_VSR_ACTIVE = False
        HDR_ACTIVE = False
    sys.stderr.flush()

# Dolby Vision export is active only if the HDR path actually came up (collect_l1 was already gated on
# an MP4 output + dovi_tool present). The export runs in _dv_export at finalize; note the state now.
DV_ACTIVE = _want_dv and HDR_ACTIVE
if DV_EXPORT and not DV_ACTIVE:
    _why = ("output is MKV (DV export needs MP4)" if OUT_IS_MKV
            else "dovi_tool not found in engine/dvtools" if not os.path.isfile(DOVI_EXE)
            else "RTX HDR is not active")
    sys.stderr.write(f"[dv] Dolby Vision export skipped: {_why}\n"); sys.stderr.flush()
elif DV_ACTIVE:
    sys.stderr.write("Dolby Vision Profile 8.1 export ON (plays as HDR10 where DV is unsupported)\n")
    sys.stderr.flush()
# HDR10+ mirrors the DV activation rule (stats collection was gated the same way via collect_hp).
HP_ACTIVE = _want_hp and HDR_ACTIVE
if HP_EXPORT and not HP_ACTIVE:
    _why = ("output is MKV (HDR10+ export needs MP4)" if OUT_IS_MKV
            else "hdr10plus_tool not found in engine/hptools" if not os.path.isfile(HP_EXE)
            else "RTX HDR is not active")
    sys.stderr.write(f"[hdr10+] HDR10+ export skipped: {_why}\n"); sys.stderr.flush()
elif HP_ACTIVE:
    sys.stderr.write("HDR10+ dynamic metadata ON (plays as HDR10 where HDR10+ is unsupported)\n")
    sys.stderr.flush()

# Detail restoration (--restore): Real-ESRGAN's anime-video model reconstructs linework and
# texture on every output frame - the "chain a restoration model after interpolation" option
# from the README's sharpness bullet. The compact arch keeps every conv at the SOURCE
# resolution (only the final PixelShuffle emits the 4x image), so the per-frame cost is small.
# It runs BEFORE the upscale: RTX VSR then receives a restored, in-distribution frame, and
# without RTX VSR the model's 4x output serves as the upscale source directly (see _restore).
# Any load failure just drops the pass with a notice - never a dead render.
RESTORE_ACTIVE = False
_restore_net = None
if args.restore:
    try:
        sys.path.insert(0, ENGINE_DIR)
        import realesr
        _restore_net = realesr.load(device)
        RESTORE_ACTIVE = True
        _restore_be = "eager fp16"
        if not args.no_trt:
            # Route the net through the same TRT engine cache as the GMFSS sub nets (build on
            # first frame ~40 s, cached per resolution, eager fallback on any failure). The
            # pass is real work either way - ~2.6 TFLOP per 1080p frame - measured on the dev
            # laptop (power-capped RTX 5090 Laptop) at ~190 ms/frame TRT vs ~330 ms eager
            # cudnn, i.e. about +50% wall on a 2x 1080p render; it scales with the OUTPUT
            # frame count, so high multipliers pay it per emitted frame.
            try:
                import trt_runtime
                _restore_net = trt_runtime.RestoreEngine(_restore_net, realesr.weights_hash())
                _restore_be = "TensorRT"
            except Exception as e:  # noqa: BLE001
                sys.stderr.write(f"[restore] TensorRT unavailable, eager pass: {repr(e)[:160]}\n")
        sys.stderr.write(f"detail restore ready (Real-ESRGAN animevideov3, {_restore_be})\n")
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"[restore] unavailable, skipping: {repr(e)[:200]}\n")
    sys.stderr.flush()

# Flow scale: GMFlow dominates the interpolation wall (~70% of a 2x pair, measured 2026-07-08) and
# its cost grows with source area, so 4K+ sources default to computing flow at an extra 0.5 factor
# (quarter resolution net of the built-in half, still 1080p-class motion detail at 4K - the
# standard GMFSS/SVFI setting for UHD). Explicit --scale always wins; sub-4K sources stay at 1.0.
if args.scale is not None:
    scale = args.scale
elif W * H >= 3840 * 2160:
    scale = 0.5
    sys.stderr.write("flow scale auto: 0.5 for this 4K+ source (motion estimated at quarter "
                     "resolution, ~4x cheaper flow; override with --scale 1.0)\n"); sys.stderr.flush()
else:
    scale = 1.0
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

def _restore(t):
    """Real-ESRGAN animevideov3 on a [1,3,h,w] RGB float frame in [0,1] (explicit fp16 pass).

    The net emits 4x; the result is resized (antialiased) to wherever the pipeline needs it
    next: straight to the output size when this pass is also the upscaler (--upscale without
    RTX VSR - one resize from the 4x reconstruction beats restore -> downscale -> bicubic
    re-upscale), else back to the source size so RTX VSR upscales the restored frame. A
    mid-run failure drops the pass for the rest of the clip; restore must never kill a render.
    """
    global RESTORE_ACTIVE
    try:
        out = _restore_net(t.half()).clamp(0.0, 1.0)     # fp16 4x reconstruction
        oh, ow = (OUT_H, OUT_W) if (UPSCALE and not RTX_VSR_ACTIVE) else (H, W)
        return realesr.fit(out, oh, ow).float()          # box/area/bicubic per target (shared w/ preview)
    except Exception as e:  # noqa: BLE001 - degrade to the unrestored frame for the rest
        sys.stderr.write(f"[restore] failed, dropping the pass: {repr(e)[:160]}\n"); sys.stderr.flush()
        RESTORE_ACTIVE = False
        return t

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
LIVE_OFF_FILE = os.environ.get("SMV_LIVE_OFF_FILE")   # GUI Hide button; present = skip generation
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
    # The GUI's Hide button drops SMV_LIVE_OFF_FILE to stop preview generation entirely (reclaim the
    # snapshot cost), Show removes it. Checked AFTER the 1s gate + _live_last bump so the os.path.exists
    # stat stays rate-limited to ~1/s; checking it before the bump would run it on every output frame.
    if LIVE_OFF_FILE and os.path.exists(LIVE_OFF_FILE):
        return
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

# Cooperative pause: when SMV_PAUSE_FILE is set (the GUI passes a path under its userData dir),
# the GUI creates that file to pause and deletes it to resume. Checked at the top of each
# generation-loop iteration, so a pause takes effect at the next source-pair boundary: the frames
# already handed to the encode queue (wq) keep draining to disk on the writer thread while this
# call blocks, then generation stops before starting the next pair. Zero cost when the env var is
# absent (CLI runs). The GUI's Pause/Resume button drives it; PAUSED/RESUMED land in the log.
PAUSE_FILE = os.environ.get("SMV_PAUSE_FILE")
_paused = False

def _check_pause():
    global _paused
    if not PAUSE_FILE:
        return
    while os.path.exists(PAUSE_FILE):
        if not _paused:
            _paused = True
            sys.stderr.write("PAUSED\n"); sys.stderr.flush()
        time.sleep(0.2)
    if _paused:
        _paused = False
        sys.stderr.write("RESUMED\n"); sys.stderr.flush()

def to_bytes(t):
    t = t.float()[..., :H, :W]            # crop off the padding added in to_tensor
    # Pipeline order, chosen for max quality (restore -> VSR -> RCAS -> TrueHDR):
    #   0. restore first (--restore): detail reconstruction at the source resolution, so the
    #      upscaler receives a restored, in-distribution frame (see _restore; without RTX VSR
    #      its 4x output IS the upscale source and the bicubic step below is skipped);
    #   1. upscale the clean interpolated frame (the AI upscaler gets an unsharpened, in-distribution
    #      input);
    #   2. sharpen at the OUTPUT resolution (RCAS crisps the final image instead of being blurred up
    #      by the upscaler);
    #   3. expand to HDR last (sharpening stays in SDR, where RCAS's luma weighting is valid).
    # VSR and TrueHDR are separate RTX passes (see rtxvideo.py) so RCAS can sit between them.
    if RESTORE_ACTIVE:
        t = _restore(t)                   # may already land at OUT size (restore-as-upscaler)
    if UPSCALE and t.shape[-1] != OUT_W:
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
    # Cast to the integer output dtype ON the GPU (after round+clamp, so values are already exact
    # integers in range and the truncating cast is lossless - bit-identical to the old CPU astype).
    # This halves the device->host copy for the default 10-bit path (uint16, 2 bytes/channel, vs
    # downloading float32 at 4) - e.g. ~12 MB not ~25 MB per 1080p frame, ~200 MB not ~400 at 8K -
    # so the per-frame PCIe cost drops with resolution and output-frame count.
    a = (t[0] * OUT_MAXV).round().clamp(0, OUT_MAXV).to(OUT_TORCH_DT) \
        .permute(1, 2, 0).contiguous().cpu().numpy()   # CHW->HWC, quantise on GPU
    return a.tobytes()

# (scene-cut detection removed 2026-07-05 per request: the app always interpolates every pair)

def _pair_fracs(p):
    """Output slot fractions within source pair interval [p, p+1), for --fps mode.

    The grid is offset by half an output step so no slot lands on a source timestamp: the
    output times (j+0.5)/ratio that fall inside [p, p+1), which can be zero slots for a pair
    when downsampling the timeline. Every emitted frame is therefore an interior blend with
    the same softness as its neighbours, instead of a sharp source frame that pops; see the
    module docstring (Legacy off-grid look). Only --fps reaches this; integer --multi uses the
    on-grid _MTW loop instead."""
    lo = math.ceil(p * ratio - 0.5)
    hi = math.ceil((p + 1) * ratio - 0.5)
    return [(j + 0.5) / ratio - p for j in range(lo, hi)]

def read_exact(stream, nbytes):
    # Read one full raw frame into a single pre-allocated buffer with readinto (no per-chunk
    # append + final bytes() copy - that was a whole extra frame-sized memcpy per frame, ~0.8 GB
    # at 16K). The bytearray is bytes-like, so np.frombuffer (to_tensor) and enc.stdin.write (the
    # plain-passthrough path) both take it directly, and each call allocates a fresh buffer so
    # nothing downstream aliases. None on EOF (short/empty read before the frame completes).
    buf = bytearray(nbytes)
    mv = memoryview(buf)
    off = 0
    while off < nbytes:
        n = stream.readinto(mv[off:])
        if not n:
            return None
        off += n
    return buf

def _ff_copy(src, dst, extra=None):
    """Error-tolerant video-only stream copy; returns True on success. Writes fragmented mp4
    for vvc (see FRAG_COPY: the regular-mp4 copy path corrupts vvc)."""
    frag = FRAG_FLAGS if (FRAG_COPY and dst.lower().endswith(".mp4")) else []
    cmd = [FFMPEG, "-v", "error", "-y", "-err_detect", "ignore_err", "-i", src,
           "-map", "0:v:0", "-c", "copy"] + (extra or []) + frag + [dst]
    return subprocess.run(cmd, creationflags=NO_WINDOW,
                          stderr=subprocess.DEVNULL).returncode == 0 and os.path.exists(dst)


def _ff_packets(path):
    """(pts list in FILE/decode order, keyframe packet indices, stream time_base as a float)
    of the sole video stream."""
    out = subprocess.check_output(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "packet=pts,flags", "-show_entries", "stream=time_base",
         "-of", "csv=p=0", path],
        text=True, creationflags=NO_WINDOW)
    pts, keys, tb = [], [], 0.0
    for ln in out.splitlines():
        f = ln.strip().split(",")
        if not f or not f[0]:
            continue
        if "/" in f[0]:                       # the stream=time_base row, e.g. "1/60000"
            _n, _d = f[0].split("/")
            tb = int(_n) / max(1, int(_d))
        elif f[0] != "N/A":
            if len(f) > 1 and "K" in f[1]:
                keys.append(len(pts))
            pts.append(int(f[0]))
    return pts, keys, tb


def _decoded_count(path, start_time):
    """How many frames actually DECODE from start_time to EOF (err_detect explode stops at the
    first corrupt packet). Frames are counted as raw bytes on a tiny grayscale scale-down, so
    the pipe cost is negligible while the decode still exercises every packet."""
    r = subprocess.run(
        [FFMPEG, "-v", "error", "-err_detect", "explode", "-ss", f"{start_time:.6f}",
         "-i", path, "-map", "0:v:0", "-vf", "scale=64:36", "-f", "rawvideo",
         "-pix_fmt", "gray", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW)
    return len(r.stdout) // (64 * 36)


def _clean_cut(pts, cap):
    """Largest packet count c <= cap whose decode-order prefix displays gaplessly.

    The stage-1 stream is CFR, so its pts form an arithmetic sequence; a prefix of c packets is
    a valid stream cut iff it contains exactly display frames 0..c-1 (references always point
    earlier in decode order, so decodability is free). With B-frames the decode order permutes
    display order, hence the running-max test rather than per-packet equality. This needs no
    keyframe at the cut: complete fragments (frag_keyframe flushes whole GOPs) pass wholesale,
    and a torn tail fragment simply shrinks the prefix back to its last gapless point."""
    if len(pts) < 2:
        return 0
    s = sorted(pts)
    step = min(b - a for a, b in zip(s, s[1:]) if b > a)
    base, runmax, best = s[0], -1, 0
    for idx, p in enumerate(pts, 1):
        runmax = max(runmax, p)
        if runmax - base == (idx - 1) * step and idx <= cap:
            best = idx
    return best


def _concat_copy(parts, dst):
    """Stream-copy concat of same-parameter video parts (the banked prefix + continuation)."""
    lst = WORK_PATH + ".concat.txt"
    with open(lst, "w", encoding="utf-8") as f:
        for p in parts:
            f.write("file '" + os.path.abspath(p).replace("\\", "/").replace("'", "'\\''") + "'\n")
    frag = FRAG_FLAGS if (FRAG_COPY and dst.lower().endswith(".mp4")) else []
    cmd = [FFMPEG, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
           "-map", "0:v:0", "-c", "copy"] + frag + [dst]
    ok = subprocess.run(cmd, creationflags=NO_WINDOW).returncode == 0 and os.path.exists(dst)
    try:
        os.remove(lst)
    except OSError:
        pass
    return ok


def _try_resume():
    """Detect and prepare a resumable partial render. On success VID_PART holds a clean banked
    prefix ending exactly at an encoder keyframe on the output grid, and the returned dict maps
    that boundary back to (source pair, in-pair slot). Any doubt -> cleanup and render fresh."""
    if not (RESUMABLE and os.path.exists(RESUME_JSON)
            and (os.path.exists(VID_PART) or os.path.exists(VID_PART2))):
        return None
    try:
        with open(RESUME_JSON, encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, ValueError):
        meta = {}
    if meta.get("sig") != RESUME_SIG:
        sys.stderr.write("resume: found a partial render from DIFFERENT settings/source; "
                         "starting fresh\n"); sys.stderr.flush()
        _resume_cleanup()
        return None
    if meta.get("venc") == "libvvenc":
        # The banked prefix is a VVC bitstream even if args.codec says hevc/av1 (the encoder
        # auto-switches to libvvenc past NVENC's 8192px ceiling and above libsvtav1's 240 fps
        # cap). Every salvage/trim below is an mp4 stream copy, and the regular-mp4 muxer
        # corrupts vvc composition offsets (see FRAG_COPY), so flip it before the first copy.
        global FRAG_COPY
        FRAG_COPY = True
    try:
        # A resumed run that itself crashed leaves prefix + continuation: merge them back into a
        # single prefix first, so every crash depth reduces to the same single-file case.
        if os.path.exists(VID_PART2):
            salv2 = WORK_PATH + ".salv2.mp4"
            if _ff_copy(VID_PART2, salv2):
                if os.path.exists(VID_PART):
                    if not _concat_copy([VID_PART, salv2], VID_FULL):
                        raise RuntimeError("concat of previous resume parts failed")
                    os.replace(VID_FULL, VID_PART)
                    os.remove(salv2)
                else:
                    os.replace(salv2, VID_PART)
            os.remove(VID_PART2)
        salv = WORK_PATH + ".salv.mp4"
        if not _ff_copy(VID_PART, salv):
            raise RuntimeError("partial video unreadable")
        pts, keys, tb = _ff_packets(salv)
        nf = len(pts)
        if not nf or not keys or not tb:
            raise RuntimeError("no usable frames in the partial video")
        # The killed encoder can leave its FINAL packet(s) with truncated data that the salvage
        # copy passes through untouched (the sizes come from the moof, the bytes ran out
        # mid-mdat), invisible to any container-level check. Decode-verify from the last
        # keyframe (at most one GOP, cheap at any length) and only trust packets that decode.
        good = keys[-1] + min(_decoded_count(salv, pts[keys[-1]] * tb), nf - keys[-1])
        # Latest safe cut: a display-gapless decode prefix (see _clean_cut) of the verified
        # packets, and early enough that the render loop still emits at least one more frame
        # (the --fps tail hold needs a live last_out).
        if NO_INTERP:
            cap = NB
        elif not FPS_MODE:
            cap = 1 + args.multi * (total_pairs - 1)
        else:
            cap = max(0, math.ceil(total_pairs * ratio - 0.5) - 1)
        c = _clean_cut(pts[:good], cap)
        if c < 1:
            raise RuntimeError("no usable frames in the partial video")
        if c == nf:
            os.replace(salv, VID_PART)
        else:
            trim = WORK_PATH + ".trim.mp4"
            if not _ff_copy(salv, trim, ["-frames:v", str(c)]):
                raise RuntimeError("trim to the cut point failed")
            got = len(_ff_packets(trim)[0])
            if got != c:
                raise RuntimeError(f"trim produced {got} frames, wanted {c}")
            os.replace(trim, VID_PART)
            os.remove(salv)
        # Map output-frame index c back onto the source grid (see each loop's emission scheme).
        if NO_INTERP:
            p, skip = c, 0
        elif not FPS_MODE:
            p, skip = (c - 1) // args.multi, (c - 1) % args.multi
        else:
            p = int((c + 0.5) / ratio)
            while p > 0 and math.ceil(p * ratio - 0.5) > c:
                p -= 1
            while math.ceil((p + 1) * ratio - 0.5) <= c:
                p += 1
            skip = c - math.ceil(p * ratio - 0.5)
        return {"c": c, "p": p, "skip": skip, "venc": meta.get("venc", ""),
                "maxcll": float(meta.get("maxcll", 0) or 0),
                "maxfall": float(meta.get("maxfall", 0) or 0)}
    except Exception as e:  # noqa: BLE001 - a failed salvage must never block a fresh render
        sys.stderr.write(f"resume: could not continue the partial render ({e}); "
                         "starting fresh\n"); sys.stderr.flush()
        _resume_cleanup()
        return None


RESUME_VENC = ""            # encoder that produced the banked prefix (must match this run's)
RESUME_DVHP_NOTE = None     # set when a resumed run cannot carry DV/HDR10+ (stats rebuild failed)


def _rebuild_hdr_stats(expect):
    """Recompute the per-frame DV L1 / HDR10+ stats (and the light-level maxima) of the banked
    prefix by decoding it back to the packed PQ RGB the measurements originally ran on. The
    banked video IS the PQ frames (visually lossless encode), so feeding each decoded frame
    through _RTX._measure_light rebuilds the in-RAM state the killed process took with it -
    within encode/4:2:0 roundtrip noise, far inside the SEI's own precision. Returns the number
    of frames measured (must equal the banked frame count for the exports to be usable)."""
    fsz = OUT_W * OUT_H * 4
    pr = subprocess.Popen(
        [FFMPEG, "-v", "error", "-i", VID_PART, "-map", "0:v:0",
         "-f", "rawvideo", "-pix_fmt", "x2rgb10le", "-"],
        stdout=subprocess.PIPE, creationflags=NO_WINDOW)
    n = 0
    try:
        while True:
            buf = read_exact(pr.stdout, fsz)
            if buf is None:
                break
            t = torch.from_numpy(np.frombuffer(buf, dtype=np.uint8).reshape(OUT_H, OUT_W, 4))
            _RTX._measure_light(t.cuda() if torch.cuda.is_available() else t)
            n += 1
            if n % 2000 == 0:
                sys.stderr.write(f"resume: rebuilding HDR dynamic-metadata stats "
                                 f"{n}/{expect} frames\n"); sys.stderr.flush()
    finally:
        pr.stdout.close()
        pr.wait()
    return n


_rz = _try_resume()
if _rz:
    RESUME_ACTIVE = True
    RESUME_OUT_BASE = _rz["c"]
    RESUME_SKIP_SRC = _rz["p"]
    RESUME_PAIR_SKIP = _rz["skip"]
    RESUME_VENC = _rz["venc"]
    RESUME_BASE_BYTES = os.path.getsize(VID_PART)
    if HDR_ACTIVE and _RTX is not None:
        # The light-level maxima are running maxima over the whole render; re-seed the banked
        # prefix's values (the private accumulators behind the read-only maxcll/maxfall
        # properties, in nits) so the HDR10 metadata still covers part 1's frames.
        _RTX._cll = max(getattr(_RTX, "_cll", 0.0), _rz["maxcll"])
        _RTX._fall = max(getattr(_RTX, "_fall", 0.0), _rz["maxfall"])
        if DV_ACTIVE or HP_ACTIVE:
            # DV/HDR10+ need per-frame stats for EVERY output frame; part 1's died with its
            # process, so rebuild them from the banked video itself before rendering part 2
            # (the render loop appends part 2's stats after these, keeping frame order).
            sys.stderr.write(f"resume: rebuilding Dolby Vision/HDR10+ per-frame stats from the "
                             f"{RESUME_OUT_BASE} banked frames...\n"); sys.stderr.flush()
            try:
                _got = _rebuild_hdr_stats(RESUME_OUT_BASE)
            except Exception as _e:  # noqa: BLE001 - never block the resume over the exports
                _got = -1
                sys.stderr.write(f"resume: stats rebuild error ({_e})\n"); sys.stderr.flush()
            if _got != RESUME_OUT_BASE:
                RESUME_DVHP_NOTE = (f"stats rebuild got {_got}/{RESUME_OUT_BASE} banked frames; "
                                    "Dolby Vision / HDR10+ export will be skipped (the HDR10 "
                                    "output itself is complete)")
                sys.stderr.write(f"resume: {RESUME_DVHP_NOTE}\n"); sys.stderr.flush()
    sys.stderr.write(f"resume: continuing the previous render from source frame "
                     f"{RESUME_SKIP_SRC}/{NB or '?'} ({RESUME_OUT_BASE} output frames already "
                     f"rendered, {RESUME_BASE_BYTES / 1e6:.0f} MB banked)\n")
    # Jump the GUI's bar/taskbar to the banked position right away.
    sys.stderr.write(f"PROGRESS {RESUME_SKIP_SRC}/{total_units}\n"); sys.stderr.flush()

# Decoder-side skip for resume: on CFR sources drop the first p frames inside ffmpeg (select
# runs pre-pipe, so skipped frames never cross the pipe); VFR sources are conformed to CFR by
# VFR_DEC's fps_mode AFTER filtering, where a select would count the wrong (pre-conform) frames,
# so they fall back to draining the pipe (exact, just slower).
# setpts rebases the survivors to t=0: without it the post-select timestamp gap makes ffmpeg's
# output vsync duplicate frames back up to the original count (measured: select alone re-emitted
# all 383 frames of a 383-frame source instead of the requested 259).
_DEC_SKIP = ["-vf", f"select=gte(n\\,{RESUME_SKIP_SRC}),setpts=PTS-STARTPTS"] \
    if (RESUME_SKIP_SRC and not VFR_DEC) else []
_PIPE_DISCARD = RESUME_SKIP_SRC if (RESUME_SKIP_SRC and VFR_DEC) else 0

dec = subprocess.Popen(
    [FFMPEG, "-v", "error", "-i", inp] + VFR_DEC + _DEC_SKIP
    + ["-f", "rawvideo", "-pix_fmt", DEC_FMT, "-"],
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
if venc == "libsvtav1" and out_label > 240:
    # SVT-AV1 hard-rejects high frame rates at encoder open ("Svt[error]: The maximum allowed
    # frame rate is 240 fps", verified 2026-07-10 on a 360fps stream), so a >240 fps render on
    # this path would die at the first frame. libvvenc has no such cap; switch if it exists.
    if _enc_works("libvvenc", f"{OUT_W}x{OUT_H}", fast=True):
        sys.stderr.write(f"libsvtav1 caps at 240 fps ({out_label} fps requested); "
                         "encoding H.266/VVC instead\n"); sys.stderr.flush()
        venc = "libvvenc"
    else:
        sys.exit(f"no available encoder can write {out_label} fps: libsvtav1 caps at 240 fps "
                 "and libvvenc is unavailable. Lower the output multiplier/fps.")
if RESUME_ACTIVE and RESUME_VENC and RESUME_VENC != venc:
    # A continuation stream must come from the SAME encoder as the banked prefix: even within
    # one codec family, different encoders write different sequence headers and the concat would
    # mix incompatible bitstreams. This only happens when encoder availability changed between
    # the runs (e.g. NVENC lost to a driver problem), so stop with instructions rather than
    # silently produce a corrupt file or throw away hours of banked work.
    sys.exit(f"resume: the interrupted render used {RESUME_VENC} but this run would encode with "
             f"{venc} (encoder availability changed). Fix the encoder (e.g. the NVIDIA driver) "
             f"to continue, or delete '{os.path.basename(VID_PART)}' and its .resume.json next "
             "to the output to render fresh.")
if venc == "libvvenc":
    # venc is final here. FRAG_COPY was computed from args.codec, but the encoder auto-switches
    # to libvvenc past NVENC's 8192px ceiling and above libsvtav1's 240 fps cap; the fragmented
    # rule follows the BITSTREAM, not the requested codec (see the FRAG_COPY comment).
    FRAG_COPY = True

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

# QUALITY-FIRST POLICY (2026-07-10, user decision: professional-grade fidelity regardless of
# size, one standard at every frame rate). The CQ values are VERIFIED against a lossless 8K
# master (2026-07-03; exact args, frame-aligned VMAF/PSNR/SSIM): HEVC CQ 17 = the app's quality
# reference, VMAF 99.78 / 57.0 dB / SSIM 0.9986; AV1 CQ 22 exceeds it (99.84 / 58.2 / 0.9988).
# CQ is a SATURATED knob at max effort (hevc CQ 14-21 byte-identical on easy content, and 14
# vs 17 stayed byte-identical under the p7 ladder), so the remaining quality levers are the
# effort/foresight flags, measured 2026-07-10 on the 1080p sample vs p5 single-pass:
#   -preset p7 -multipass fullres -rc-lookahead 32: hevc 50.8->52.5 dB avg, worst frame
#   49.3->51.6 dB, SSIM 0.9956->0.9968 at +7% size; av1 51.2->52.2 dB, worst 50.1->51.6,
#   SSIM 0.9955->0.9962 at +2.6%. Throughput drops 480->154 fps (hevc, 1080p; av1 273) but
#   GMFSS generates output frames far slower than that, so the pipeline bottleneck never moves.
# -tune uhq was measured and REJECTED: its temporal filtering rewrites frame content (fidelity
# DROPPED to 50.2 dB / SSIM 0.9946 while shrinking the file); it is a streaming-perception
# tune, not a fidelity tune. The former high-fps CQ relief (+3 above 120 fps, a pure size
# optimization measured 2026-07-09) was REMOVED under this policy: tween-dense streams now
# spend whatever the base CQ costs (the known extreme: ~77 Mbps for 360fps 1440p).
# NVENC runs AQ + a small chroma QP boost. SVT-AV1 CPU fallback: CRF 17 preset 6 (measured
# 2026-07-10: 48.9 dB / SSIM 0.9932 vs 47.9 / 0.9916 for the old CRF 20 preset 8 at similar
# size; 98 fps at 1080p, still far above what the interpolator can feed it).
if USE_NVENC:
    # SMV_CQ overrides the tuned constant-quality value (hidden knob for measurement work like
    # the 2026-07-03 codec tuning; not a user-facing setting).
    cq = os.environ.get("SMV_CQ") or ("22" if venc == "av1_nvenc" else "17")
    # Lookahead depth, measured 2026-07-10 (md5 + frame-aligned PSNR, 24-frame sample AND a
    # 5405-frame real clip). The fidelity gain needs lookahead AND multipass TOGETHER (either
    # alone measures ~0 over the old p5 baseline; p7 adds +0.25 dB), and the DEPTH response is
    # stepped, not monotonic: 1/2/4 encode byte-identically and measure BEST (hevc 53.5 dB avg
    # / 52.3 worst; av1 53.1/52.5), 8/16/32 encode byte-identically at -1.1 dB (the deeper
    # queue turns on B-frame restructuring that trades fidelity for size - the wrong trade
    # here) while a depth-32 queue also costs real VRAM (~81 MB/slot at 8K = 2.6 GB measured).
    # So: depth 1 = full gain, best fidelity, one ~5-81 MB slot. Above 120 fps output the sign
    # FLIPS: on wall-to-wall tweens ANY lookahead measured -1 dB AND bigger files (la1==la8
    # there), so high-fps renders drop the queue entirely.
    la = "0" if out_label > 120 else "1"
    qargs = ["-preset", "p7", "-tune", "hq", "-rc", "vbr", "-cq", cq, "-b:v", "0",
             "-multipass", "fullres", "-rc-lookahead", la,
             "-spatial-aq", "1", "-temporal-aq", "1"]
    if venc in ("h264_nvenc", "hevc_nvenc"):
        qargs += ["-qp_cb_offset", "-2", "-qp_cr_offset", "-2"]
    if DV_ACTIVE or HP_ACTIVE:
        # Dolby Vision / HDR10+ export re-muxes the encoded HEVC through a raw elementary stream
        # (dovi_tool / hdr10plus_tool need Annex B), and a B-frame reorder buffer makes that
        # raw->MP4 copy assign non-monotonic DTS and silently drop the tail frames. Coding order ==
        # display order with no B-frames, so the remux is exact and the per-frame metadata aligns
        # 1:1. The size cost is small at our CQ.
        qargs += ["-bf", "0"]
elif venc == "libvvenc":
    # vvenc has no CRF; QP mode. QUALITY-FIRST 2026-07-10: QP 17 (was 20), measured on the
    # 1080p sample: 49.6 dB / SSIM 0.9939 vs 47.9 / 0.9919 at QP 20, +1.8 dB for +47% size,
    # still ~55% of the HEVC ladder's size (VVC was the thinnest-margin codec of the three,
    # this buys it real headroom). Preset stays fast: a slower preset at fixed QP is a size
    # optimizer, not a fidelity gain (medium measured SMALLER and slightly LOWER PSNR than
    # fast at the same QP), so under quality-first the QP is the lever, not the preset.
    # QPA (perceptual QP adaptation, vvenc default on) is now ALWAYS off: it deliberately
    # spends fewer bits where it predicts the eye won't look (its ~27% size saving on normal
    # content) and it INVERTS outright on 120+fps tween streams (measured 2026-07-03: bloats
    # 360fps files, at 8K it made VVC larger than HEVC). A fidelity-first pipeline wants
    # uniform quality, not psychovisual bit-robbing.
    qargs = ["-qp", "17", "-preset", "fast", "-qpa", "0"]
    if OUT_W > NVENC_MAX or OUT_H > NVENC_MAX:
        # Ultra sizes: cap frame-level parallelism. Measured at 15360x8640 (24 frames): default
        # peaks ~34 GB RAM, maxparallelframes=2 peaks ~30 GB at the SAME wall time.
        qargs += ["-vvenc-params", "maxparallelframes=2"]
else:
    # SVT-AV1 fallback, quality-first 2026-07-10: CRF 17 preset 6 beat the old CRF 20 preset 8
    # on BOTH axes on the 1080p sample (48.9 dB / SSIM 0.9932 vs 47.9 / 0.9916, and preset 6 at
    # CRF 17 came out SMALLER than preset 8 at the same CRF). Throughput measured 98 fps at
    # 1080p (984-frame run) vs 140 for preset 8: both far above what GMFSS can feed it, so the
    # "don't starve the frame pipe" constraint that justified preset 8 still holds with margin.
    qargs = ["-crf", "17", "-preset", "6"]

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
# Track mapping (see the passthrough block above). Audio: every track into mkv; into mp4 only
# the compatible ones (auto-mp4 implies all are compatible, so selective mapping only bites
# when an explicit .mp4 path overrode the mkv choice). Subtitles/fonts: mkv only, with
# mov_text converted to SRT per stream. Chapters ride along in both containers.
_maps, _drops = [], []
if OUT_IS_MKV:
    _maps += ["-map", "1:a?"]
else:
    _good = [i for i, c in AUD_STREAMS if c in MP4_AUDIO_OK]
    for _i in _good:
        _maps += ["-map", f"1:{_i}"]
    if len(_good) < len(AUD_STREAMS):
        _drops.append(f"{len(AUD_STREAMS) - len(_good)} audio track(s) (codec not mp4-compatible)")
_maps += ["-c:a", "copy"]
if OUT_IS_MKV and SUB_STREAMS:
    _maps += ["-map", "1:s?", "-c:s", "copy"]
    for _j, (_i, _c) in enumerate(SUB_STREAMS):
        if _c not in MKV_SUB_COPY_OK:
            _maps += [f"-c:s:{_j}", "srt"]     # mov_text etc: mp4-native, transcode for mkv
elif SUB_STREAMS:
    _drops.append(f"{len(SUB_STREAMS)} subtitle track(s) (mp4 output; use .mkv to keep them)")
if OUT_IS_MKV and HAS_ATTACH:
    _maps += ["-map", "1:t?"]
_maps += ["-map_chapters", "1"]
_n_sub = len(SUB_STREAMS) if OUT_IS_MKV else 0
_n_aud = len(AUD_STREAMS) if OUT_IS_MKV else len([1 for _, c in AUD_STREAMS if c in MP4_AUDIO_OK])
if _n_aud > 1 or _n_sub or (OUT_IS_MKV and HAS_ATTACH):
    sys.stderr.write(f"passthrough: {_n_aud} audio, {_n_sub} subtitle track(s)"
                     f"{', fonts' if (OUT_IS_MKV and HAS_ATTACH) else ''}, chapters -> "
                     f"{'mkv' if OUT_IS_MKV else 'mp4'}\n")
for _d in _drops:
    sys.stderr.write(f"passthrough: dropping {_d}\n")
sys.stderr.flush()
# HDR into MKV runs TWO-STAGE so the HDR10 static metadata survives: the injector writes ISOBMFF
# boxes (mp4-only), but ffmpeg's mov demuxer reads mdcv/clli into stream side data and the
# Matroska muxer writes them as MKV's NATIVE MasteringMetadata/MaxCLL elements (verified: values
# map exactly). So: stage 1 encodes the video alone into a temp .mp4 (no source input at all, so
# not even chapters sneak in), the boxes are injected there, and stage 2 stream-copy remuxes the
# temp video plus the source's tracks into the final .mkv (see _finalize_output).
HDR_MKV_2STAGE = HDR_ACTIVE and OUT_IS_MKV
if RESUMABLE:
    # Resumable renders always run two-stage: encode the video ALONE into a truncation-tolerant
    # fragmented mp4 (the crash/exit resume asset - see the resume block), then _finalize_output
    # remuxes it with the source's tracks (and the HDR10 boxes when HDR is on) into the real
    # container. A resumed run appends into a separate continuation file, concatenated at finalize.
    _ENC_TARGET = VID_PART2 if RESUME_ACTIVE else VID_PART
    _stage2_maps, _maps = _maps, []
    _in2 = []
elif HDR_MKV_2STAGE:
    _ENC_TARGET = WORK_PATH + ".video.tmp.mp4"
    _stage2_maps, _maps = _maps, []
    _in2 = []
else:
    _ENC_TARGET = WORK_PATH
    _stage2_maps = []
    _in2 = ["-i", inp]
enc_cmd = [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", ENC_IN_FMT,
           "-s", f"{OUT_W}x{OUT_H}", "-r", rate_str] + _TQ + ["-i", "-"] + _in2 + \
          ["-map", "0:v:0"] + _maps + ["-c:v", venc, "-vf", vf] + \
          ["-max_interleave_delta", "0"]
# -max_interleave_delta 0 forces true packet-by-packet interleave when muxing our video with the
# source's audio/subs. The default (10 s) lets the muxer write the passthrough tracks in ~10 s
# bursts, which is invisible at normal bitrates but FATAL at high-fps rates: 10 s of 360 fps video
# is ~100+ MB of audio-free data, mpv's demuxer queue (~150 MB) overflows hunting for the next
# audio burst after a seek and declares the audio track EOF (sound dies until a from-zero restart;
# diagnosed 2026-07-10 on a real 360 fps episode - '[mkv] Too many packets in the demuxer packet
# queues ... audio/1: 0 packets'). Zero-delta buffering is bounded here because the video pipe is
# the pacing stream.
# VVC-in-MP4 muxing is gated behind -strict experimental on some ffmpeg versions; harmless otherwise.
# Resumable stage-1 files are FRAGMENTED mp4 (see the resume block): a killed encoder leaves all
# complete moof/mdat fragments readable, and frag_keyframe puts those boundaries on encoder
# keyframes - exactly the clean closed-GOP stream-copy cut points the resume trim needs.
_frag = ["-movflags", "+frag_keyframe+empty_moov+default_base_moof"] if RESUMABLE else []
enc_cmd += qargs + prof + color + (["-strict", "experimental"] if venc == "libvvenc" else []) \
    + _frag + [_ENC_TARGET]
enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, creationflags=NO_WINDOW)
if RESUMABLE:
    # Sidecar from frame one (not just from the first heartbeat), so even a crash in the first
    # seconds leaves a valid resume pair once a keyframe has been flushed.
    _write_resume_sidecar(RESUME_SKIP_SRC, total_units)
_sharp_note = f"  sharpen(rcas)={SHARPEN:g}" if SHARPEN > 0 else ""
_up_note = f"  upscale={UPSCALE_F:g}x->{OUT_W}x{OUT_H}" if UPSCALE else ""
_hdr_note = "  HDR10(TrueHDR,BT.2020 PQ)" if HDR_ACTIVE else ""
_res_note = "  restore(animevideov3)" if RESTORE_ACTIVE else ""
sys.stderr.write(f"encode: {venc} visually-lossless -> {out_pix}  "
                 f"(source {SRC_CODEC or '?'} {SRC_BITS}bit {SRC_PIX}){_res_note}{_sharp_note}{_up_note}{_hdr_note}\n"); sys.stderr.flush()

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


def _write_hdr10_metadata(target):
    """Stamp HDR10 static metadata into an mp4 (mastering display + content light level).

    The bundled LGPL ffmpeg cannot write it on the hevc_nvenc path (no encoder/BSF option exists),
    so hdr10_meta adds the ISOBMFF boxes directly: the mastering display peak is the TrueHDR target
    (HDR_NITS), and MaxCLL/MaxFALL are measured from the actual frames. With this, one PQ/BT.2020
    file tone-maps correctly on both a 1000-nit and a 400-nit display with no per-display setting.
    Best-effort: a failure logs a note but never fails the render."""
    if not HDR_ACTIVE or not str(target).lower().endswith(".mp4"):
        return
    try:
        import hdr10_meta
        cll = int(getattr(_RTX, "maxcll", 0) or 0)
        fall = int(getattr(_RTX, "maxfall", 0) or 0)
        if hdr10_meta.inject_hdr10(target, max_nits=HDR_NITS, maxcll=cll, maxfall=fall,
                                   colorspace=HDR_MASTER_PRIM):
            sys.stderr.write(f"HDR10 metadata: mastered {HDR_NITS} nits ({HDR_MASTER_PRIM}), "
                             f"measured MaxCLL {cll} / MaxFALL {fall} nits\n"); sys.stderr.flush()
    except Exception as e:  # noqa: BLE001 - container metadata is a finishing touch, not load-bearing
        sys.stderr.write(f"HDR10 metadata: skipped ({e})\n"); sys.stderr.flush()


def _remux_tracks(vid_src, dst):
    """Stream-copy remux of the finished video with the source's passthrough tracks.
    -max_interleave_delta 0: same interleave fix as enc_cmd (see there) - this is exactly the
    two-input mux that produced the audio-burst layout mpv chokes on at high fps.
    A vvc video into an .mp4 destination is written FRAGMENTED (see FRAG_COPY: the regular-mp4
    copy path corrupts vvc; fmp4 needs no faststart, its index is already up front)."""
    frag = FRAG_FLAGS if (FRAG_COPY and dst.lower().endswith(".mp4")) else []
    cmd = [FFMPEG, "-v", "error", "-y", "-i", vid_src, "-i", inp,
           "-map", "0:v:0", "-c:v", "copy"] + _stage2_maps + \
          ["-max_interleave_delta", "0"] + frag + [dst]
    subprocess.run(cmd, check=True, creationflags=NO_WINDOW)
    if frag:
        sys.stderr.write("vvc: final mp4 written fragmented (resume-safe layout; plays in "
                         "modern players, streams without faststart)\n"); sys.stderr.flush()


def _finalize_output():
    """Finish the container.

    Resumable renders (hevc/av1): the encoder wrote a video-only fragmented mp4 (plus a
    continuation stream when this run resumed a previous one). Concat the parts, then remux
    with the source's tracks into the real container - via a temp regular mp4 carrying the
    injected HDR10 boxes when the final container is mkv (ffmpeg maps mdcv/clli onto Matroska's
    native MasteringMetadata/MaxCLL elements, verified value-exact), or injecting straight into
    the final mp4 otherwise. Resume artifacts are cleaned only on success, so a failed finalize
    stays resumable.

    Legacy path (vvc): plain renders inject the HDR10 boxes into the finished mp4; HDR-into-MKV
    renders inject into the stage-1 temp mp4 and remux (the original HDR_MKV_2STAGE flow)."""
    if not RESUMABLE:
        if not HDR_MKV_2STAGE:
            _write_hdr10_metadata(WORK_PATH)
            if HP_ACTIVE:
                _hp_export(WORK_PATH)
            if DV_ACTIVE:
                _dv_export(WORK_PATH)
            return
        _write_hdr10_metadata(_ENC_TARGET)
        try:
            _remux_tracks(_ENC_TARGET, WORK_PATH)
            os.remove(_ENC_TARGET)
            sys.stderr.write("HDR10 metadata carried into the MKV as native "
                             "MasteringMetadata/MaxCLL elements\n"); sys.stderr.flush()
        except Exception as e:  # noqa: BLE001 - keep the finished video rather than fail the render
            try:
                os.remove(WORK_PATH)   # a half-written remux must never be promoted to out_path
            except OSError:
                pass
            sys.stderr.write(f"final MKV remux failed ({e}); the HDR video (with metadata, without "
                             f"the extra tracks) was kept at {_ENC_TARGET}\n"); sys.stderr.flush()
        return
    vid = VID_PART
    try:
        if RESUME_ACTIVE:
            if not _concat_copy([VID_PART, VID_PART2], VID_FULL):
                raise RuntimeError("concat of the banked prefix + continuation failed")
            vid = VID_FULL
        if HDR_ACTIVE and OUT_IS_MKV:
            # The HDR10 boxes are ISOBMFF: hop through an mp4 so the mkv remux maps them onto
            # Matroska's native elements (the proven HDR_MKV_2STAGE route). Fragmented for vvc
            # (FRAG_COPY; the injector was verified to work on fmp4 too).
            tmp = WORK_PATH + ".video.tmp.mp4"
            subprocess.run([FFMPEG, "-v", "error", "-y", "-i", vid, "-map", "0:v:0",
                            "-c", "copy"] + (FRAG_FLAGS if FRAG_COPY else []) + [tmp],
                           check=True, creationflags=NO_WINDOW)
            _write_hdr10_metadata(tmp)
            _remux_tracks(tmp, WORK_PATH)
            os.remove(tmp)
            sys.stderr.write("HDR10 metadata carried into the MKV as native "
                             "MasteringMetadata/MaxCLL elements\n"); sys.stderr.flush()
        else:
            _remux_tracks(vid, WORK_PATH)
            if HDR_ACTIVE:
                _write_hdr10_metadata(WORK_PATH)
                if RESUME_DVHP_NOTE and (HP_ACTIVE or DV_ACTIVE):
                    # Normally a resumed run rebuilds the per-frame stats from the banked video
                    # (see _rebuild_hdr_stats) and exports as usual; this only fires when that
                    # rebuild failed or came up short.
                    sys.stderr.write(f"resume: {RESUME_DVHP_NOTE}\n"); sys.stderr.flush()
                else:
                    # HDR10+ first, DV second: the HDR10+ SEI NALs ride inside the HEVC samples,
                    # so the DV export's extract -> inject-rpu -> remux passes them through
                    # untouched, while running DV first would drop the dvvC box. See _dv_export.
                    if HP_ACTIVE:
                        _hp_export(WORK_PATH)
                    if DV_ACTIVE:
                        _dv_export(WORK_PATH)
        _resume_cleanup()
    except Exception as e:  # noqa: BLE001 - keep the rendered video (and resumability) on failure
        try:
            os.remove(WORK_PATH)   # a half-written remux must never be promoted to out_path
        except OSError:
            pass
        sys.stderr.write(f"final remux failed ({e}); the rendered video stream was kept at "
                         f"{vid} and the render stays resumable\n"); sys.stderr.flush()


def _dv_export(mp4_path):
    """Turn the finished HDR10 MP4 into a Dolby Vision Profile 8.1 MP4 in place, GPAC-free: dovi_tool
    builds the RPU from the per-frame L1 collected during the render (dovi_tool handles B-frame
    reordering), the bundled ffmpeg muxes it, and hdr10_meta writes the DV configuration box (dvvC)
    plus the HDR10 fallback boxes ourselves. The base layer stays HDR10, so non-DV players fall back.
    Best-effort: any failure keeps the finished HDR10 file."""
    l1 = list(getattr(_RTX, "l1", []) or [])
    if not l1:
        sys.stderr.write("[dv] no per-frame metadata collected; kept the HDR10 file\n"); sys.stderr.flush()
        return
    base = mp4_path + ".dvwork"
    hevc, rpu, dvhevc, cfg, tmp_out = (base + s for s in (".hevc", ".rpu", ".dv.hevc", ".json", ".mp4"))
    cll = int(getattr(_RTX, "maxcll", 0) or 0)
    fall = int(getattr(_RTX, "maxfall", 0) or 0)
    _q = dict(check=True, creationflags=NO_WINDOW, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        import hdr10_meta
        # 1. pull the HDR10 HEVC elementary stream out of the finished mp4
        subprocess.run([FFMPEG, "-v", "error", "-y", "-i", mp4_path, "-map", "0:v:0", "-c", "copy",
                        "-bsf:v", "hevc_mp4toannexb", "-f", "hevc", hevc], **_q)
        # 2. dovi_tool generate config: one L1 shot per output frame + L6 (mastering + measured light)
        gen = {"cm_version": "V29", "length": len(l1),
               "level6": {"max_display_mastering_luminance": HDR_NITS, "min_display_mastering_luminance": 1,
                          "max_content_light_level": cll, "max_frame_average_light_level": fall},
               "shots": [{"start": i, "duration": 1,
                          "metadata_blocks": [{"Level1": {"min_pq": mn, "avg_pq": av, "max_pq": mx}}]}
                         for i, (mn, av, mx) in enumerate(l1)]}
        with open(cfg, "w") as f:
            json.dump(gen, f)
        # 3. build the RPU, then 4. inject it in-band into the elementary stream
        subprocess.run([DOVI_EXE, "generate", "-j", cfg, "-o", rpu], **_q)
        subprocess.run([DOVI_EXE, "inject-rpu", "-i", hevc, "--rpu-in", rpu, "-o", dvhevc], **_q)
        # 5. remux the DV video with the original audio/subtitle tracks, re-tagging BT.2020 PQ
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "hevc", "-r", rate_str, "-i", dvhevc,
                        "-i", mp4_path, "-map", "0:v:0", "-map", "1:a?", "-map", "1:s?", "-c", "copy",
                        "-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc",
                        "-color_range", "tv", "-tag:v", "hvc1",
                        "-max_interleave_delta", "0", tmp_out],   # interleave fix, see enc_cmd
                       check=True, creationflags=NO_WINDOW)
        # 6. stamp the DV configuration box (dvvC) + the HDR10 mastering/CLL fallback boxes ourselves
        hdr10_meta.inject_dv_config(tmp_out, OUT_W, OUT_H, out_label)
        hdr10_meta.inject_hdr10(tmp_out, max_nits=HDR_NITS, maxcll=cll, maxfall=fall,
                                colorspace=HDR_MASTER_PRIM)
        os.replace(tmp_out, mp4_path)
        sys.stderr.write("Dolby Vision: Profile 8.1 written (HDR10-compatible)\n")
        sys.stderr.flush()
    except Exception as e:  # noqa: BLE001 - never lose the HDR10 render over the DV step
        sys.stderr.write(f"[dv] export failed ({repr(e)[:200]}); kept the HDR10 file at {mp4_path}\n")
        sys.stderr.flush()
    finally:
        for p in (hevc, rpu, dvhevc, cfg, tmp_out):
            try:
                os.remove(p)
            except OSError:
                pass


def _hp_export(mp4_path):
    """Embed HDR10+ (SMPTE ST 2094-40) dynamic metadata into the finished HDR10 MP4 in place:
    the per-frame brightness statistics collected during the render (RTXVideo collect_hp) become
    the tool's metadata JSON, hdr10plus_tool interleaves the SEI into the extracted HEVC stream,
    and the bundled ffmpeg re-muxes it with the audio. The SEI rides inside the samples, so no
    container box is needed (unlike DV's dvvC) and a later DV export passes it through. The JSON
    layout follows the format hdr10plus_tool itself extracts (Profile A, per-frame SceneInfo):
    DistributionValues carries [1st pct, 99.98th pct, bright-pixel fraction, 25/50/75/90/95/99th
    pct] - the slot-2 near-peak / slot-3 fraction convention observed in real HDR10+ masters -
    with luminance values in 0.1-nit units. Best-effort: any failure keeps the HDR10 file."""
    hp = list(getattr(_RTX, "hp", []) or [])
    if not hp:
        sys.stderr.write("[hdr10+] no per-frame metadata collected; kept the HDR10 file\n"); sys.stderr.flush()
        return
    base = mp4_path + ".hpwork"
    hevc, hphevc, cfg, tmp_out = (base + s for s in (".hevc", ".hp.hevc", ".json", ".mp4"))
    cll = int(getattr(_RTX, "maxcll", 0) or 0)
    fall = int(getattr(_RTX, "maxfall", 0) or 0)
    _q = dict(check=True, creationflags=NO_WINDOW, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        import hdr10_meta
        # 1. pull the HDR10 HEVC elementary stream out of the finished mp4
        subprocess.run([FFMPEG, "-v", "error", "-y", "-i", mp4_path, "-map", "0:v:0", "-c", "copy",
                        "-bsf:v", "hevc_mp4toannexb", "-f", "hevc", hevc], **_q)
        # 2. build the metadata JSON: one SceneInfo entry per output frame (single scene, frame-
        #    accurate statistics; TargetedSystemDisplayMaximumLuminance 0 = Profile A convention).
        gen = {"JSONInfo": {"HDR10plusProfile": "A", "Version": "1.0"},
               "SceneInfo": [
                   {"LuminanceParameters": {
                        "AverageRGB": f["avg"],
                        "LuminanceDistributions": {
                            "DistributionIndex": [1, 5, 10, 25, 50, 75, 90, 95, 99],
                            "DistributionValues": f["dist"]},
                        "MaxScl": f["maxscl"]},
                    "NumberOfWindows": 1, "TargetedSystemDisplayMaximumLuminance": 0,
                    "SceneFrameIndex": i, "SceneId": 0, "SequenceFrameIndex": i}
                   for i, f in enumerate(hp)],
               "SceneInfoSummary": {"SceneFirstFrameIndex": [0], "SceneFrameNumbers": [len(hp)]},
               "ToolInfo": {"Tool": "SmoothMyVideo", "Version": "1.0"}}   # required by the parser
        with open(cfg, "w") as f:
            json.dump(gen, f)
        # 3. interleave the ST 2094-40 SEI messages before each frame's slices. The tool's stderr
        # is captured and surfaced on failure (a silent exit-1 here is undebuggable otherwise).
        _r = subprocess.run([HP_EXE, "inject", "-i", hevc, "-j", cfg, "-o", hphevc],
                            creationflags=NO_WINDOW, capture_output=True, text=True)
        if _r.returncode != 0:
            raise RuntimeError("hdr10plus_tool inject: "
                               + ((_r.stderr or "") + (_r.stdout or "")).strip()[-300:])
        # 4. remux with the original audio/subtitle tracks, re-tagging BT.2020 PQ, and re-stamp the
        #    HDR10 static boxes (the remux rebuilds the container, dropping the injected ones)
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "hevc", "-r", rate_str, "-i", hphevc,
                        "-i", mp4_path, "-map", "0:v:0", "-map", "1:a?", "-map", "1:s?", "-c", "copy",
                        "-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc",
                        "-color_range", "tv", "-tag:v", "hvc1",
                        "-max_interleave_delta", "0", tmp_out],   # interleave fix, see enc_cmd
                       check=True, creationflags=NO_WINDOW)
        hdr10_meta.inject_hdr10(tmp_out, max_nits=HDR_NITS, maxcll=cll, maxfall=fall,
                                colorspace=HDR_MASTER_PRIM)
        os.replace(tmp_out, mp4_path)
        sys.stderr.write("HDR10+: dynamic metadata written (HDR10-compatible)\n")
        sys.stderr.flush()
    except Exception as e:  # noqa: BLE001 - never lose the HDR10 render over the HDR10+ step
        sys.stderr.write(f"[hdr10+] export failed ({repr(e)[:200]}); kept the HDR10 file at {mp4_path}\n")
        sys.stderr.flush()
    finally:
        for p in (hevc, hphevc, cfg, tmp_out):
            try:
                os.remove(p)
            except OSError:
                pass


_disk_warned = False


def _progress(k, total):
    """One PROGRESS heartbeat (the GUI parses `PROGRESS k/total`; emitted every 10 pairs/frames)
    plus a `SIZE cur projected` line: the encoder's bytes so far and their linear projection to
    100%, accurate to a few percent once a minute of content is in (CQ bitrate is stationary
    enough per title). The GUI shows the projection next to the ETA - so a 10-hour render tells
    you it will be ~13 GB in its first minutes, not at the end. A one-time warning fires when the
    remaining bytes (doubled for the HDR-into-MKV two-stage, whose temp and final coexist) exceed
    the free space on the output drive."""
    global _disk_warned
    sys.stderr.write(f"PROGRESS {k}/{total}\n")
    if RESUMABLE:
        _write_resume_sidecar(k, total)
    frac = k / total if total else 0.0
    if frac >= 0.01:
        try:
            # Not os.path.getsize: on Windows the directory-entry size it reads is updated LAZILY
            # while another process (the encoder) holds the file open, so it can report ~0 minutes
            # into a render. Opening our own handle and seeking to the end forces the true current
            # size. Below 1 MB the projection is still meaningless (header + first flush), skip it.
            with open(_ENC_TARGET, "rb") as _f:
                _f.seek(0, 2)
                cur = _f.tell()
            # A resumed run only writes the continuation stream; count the banked prefix too so
            # the projection covers the whole output.
            cur += RESUME_BASE_BYTES
            if cur < (1 << 20):
                sys.stderr.flush()
                return
            proj = int(cur / frac)
            sys.stderr.write(f"SIZE {cur} {proj}\n")
            if not _disk_warned and frac >= 0.05:
                _disk_warned = True
                import shutil
                free = shutil.disk_usage(os.path.dirname(out_path) or ".").free
                # Two-stage renders (all resumable ones, and HDR-into-MKV) transiently hold the
                # video stream twice: the stage-1 file plus the remuxed final.
                need = (proj - cur) + (proj if (HDR_MKV_2STAGE or RESUMABLE) else 0) + (1 << 30)
                if need > free:
                    sys.stderr.write(
                        f"warning: projected output ~{proj / 1e9:.1f} GB"
                        + (" (x2 transiently for the HDR MKV remux)" if HDR_MKV_2STAGE else "")
                        + f" but only {free / 1e9:.1f} GB free on the output drive - "
                        "the render may fail; free up space or change the output location\n")
        except OSError:
            pass
    sys.stderr.flush()


def _drain_pipes():
    """Teardown shared by all three render loops (runs in each loop's finally, so on error too):
    flush the encode queue and close the pipes in order - writer sentinel, join the writer, close
    the encode stdin, close the decode stdout, wait on both ffmpeg processes."""
    wq.put(None)            # sentinel: let the writer drain its queue and exit
    wt.join()
    enc.stdin.close()
    dec.stdout.close()
    enc.wait()
    dec.wait()


def _finish(done_msg, out_frames=None):
    """Success tail shared by all three render loops: surface a failed encode pipe as a nonzero
    exit, finalize the container (HDR10 boxes / MKV remux), flush the last live thumbnail, log the
    done line and exit 0. `out_frames` is the EXACT number of frames actually written (each loop
    knows it); emitted as OUTFRAMES so the GUI's frame counter is right instead of estimating from
    the container's (sometimes off-by-one) nb_frames."""
    if _werr:
        raise _werr[0]          # surface a failed encode pipe as a nonzero exit
    _finalize_output()
    # Promote the finished .part file to the real output name (see WORK_PATH). The 2-stage MKV
    # remux failure path removes WORK_PATH and keeps its own temp, so promotion is conditional.
    if os.path.exists(WORK_PATH):
        os.replace(WORK_PATH, out_path)
    _live_flush()               # let the final live thumbnail land before exit
    if out_frames is not None:
        sys.stderr.write(f"OUTFRAMES {out_frames}\n")
    sys.stderr.write(done_msg); sys.stderr.flush()
    sys.exit(0)


# Run the whole interpolation/encode pipeline on one non-default CUDA stream so TRT, softsplat's cupy
# kernel and the torch glue all share it: same-stream ordering then makes each op's output ready for the
# next with no per-call host sync (the old per-engine synchronize in trt_runtime is gone), leaving just
# the implicit drain at each frame's .cpu() download. The non-default stream also keeps TensorRT's
# default-stream warning away. Sync first so model weights and RTX init (issued on the default stream)
# are visible on the new stream.
torch.cuda.synchronize()
_infer_stream = torch.cuda.Stream()
torch.cuda.set_stream(_infer_stream)

# Resume on a VFR source: the decoder could not pre-skip with a select filter (it would count
# pre-conform frames; see _DEC_SKIP), so drain the already-conformed pipe here instead.
for _ in range(_PIPE_DISCARD):
    if rq.get() is None:
        sys.exit("resume: the source ended before the resume point (source changed?); "
                 "delete the .part files next to the output and render fresh")

if NO_INTERP:
    # Sharpen-only pass: no interpolation, one output frame per source frame at the source fps.
    # Each decoded frame is RCAS-sharpened on the GPU when --sharpen > 0, or passed straight
    # through (a plain re-encode) when it is 0. Shares the same encode pipe/threads as the
    # interpolation path, so colour signalling, bit depth and audio are handled identically.
    k = RESUME_OUT_BASE            # frames banked by a resumed run (0 on a fresh one)
    try:
        while True:
            _check_pause()      # block here (queued frames keep encoding) while the GUI holds Pause
            buf = rq.get()
            if buf is None:
                break
            # Route through to_bytes (decode->tensor->process->bytes) when there is any per-frame
            # GPU work to do (sharpen and/or upscale); otherwise pass the raw frame straight to the
            # encoder as a plain re-encode.
            wq.put(to_bytes(to_tensor(buf))
                   if (SHARPEN > 0 or UPSCALE or HDR_ACTIVE or RESTORE_ACTIVE) else buf)
            k += 1
            if k % 10 == 0:
                _progress(k, total_units)
    finally:
        _drain_pipes()
    _finish(f"done {k} frames "
            f"({'RCAS-sharpened' if SHARPEN > 0 else 're-encoded'}) -> {out_path}\n", out_frames=k)

# =============================================================================================
# On-grid passthrough interpolation (GMFSS; integer --multi; not --fps). The
# output timeline lands ON the source grid: EVERY real source frame passes through at its integer
# timestamp (t=0,1,2,...,N-1) at full quality, and the M-1 generated tweens are inserted at the half
# positions t=k+j/M between each consecutive pair. Total frames = multi*(N-1)+1 (2N-1 at 2x), the
# honest on-grid count for reaching the target fps with real endpoints, so a 2-frame clip at 2x gives
# 3 frames (real, tween, real), not 4. Duration is ~(M-1)/(M*fps) shorter than the source (the last
# frame has no slot after it to fill), exactly how RIFE/DAIN report 2N-1.
#
# The interior frames that sit ON a source timestamp t=k are the REAL frame f[k], kept at full
# quality. This is a deliberate choice (2026-07-05, user): a real frame can pop a little against the
# softer tweens, but it is the max-quality frame we already have, and the alternatives that avoid the
# pop are worse. A bracket inference(f[k-1], f[k+1], 0.5) skips f[k] entirely, so it smooths past f[k]'s
# pose on non-linear motion (a wing at an extreme the neighbours do not bracket). Interpolating the two
# neighbour tweens keeps f[k]'s motion but double-fades (generating between two already-generated
# frames compounds GMFSS's softening). So we keep f[k].
# Only --fps (arbitrary timeline, off the source grid) uses the legacy path below.
if not FPS_MODE:
    _MTW = [j / args.multi for j in range(1, args.multi)]   # interior tween fractions j/M, on-grid

    def _gen_tweens(a, b, fracs):
        with amp():
            reuse = model.reuse(a, b, scale)
            return [to_bytes(model.inference(a, b, reuse, f)) for f in fracs]

    prev0 = rq.get()
    if prev0 is None:
        sys.exit("no frames decoded")
    f_cur = to_tensor(prev0)  # f[0] (f[p] when resuming: the decoder skipped to the resume pair)
    k = RESUME_SKIP_SRC       # pairs processed (PROGRESS counter, matches total_pairs)
    last_out = None
    # Resume lands mid-pair when the banked prefix's last keyframe does: the first resumed pair
    # only re-generates its still-missing tween slots (never the banked ones).
    _skip = RESUME_PAIR_SKIP
    try:
        # Passthrough scheme: every REAL source frame is emitted at its integer timestamp at full
        # quality, with the M-1 generated tweens inserted at the half positions between each pair. The
        # real frames can pop a little against the softer tweens, but the alternatives lose quality: a
        # bracket interp(f[k-1], f[k+1]) skips f[k] and can smooth past its pose, and interpolating two
        # already-generated tweens double-fades. So we keep the frames we already have, at max quality.
        if not RESUME_ACTIVE:           # t=0 : real f[0] (banked already when resuming)
            last_out = to_bytes(f_cur)
            wq.put(last_out)
        while True:
            _check_pause()      # block here (queued frames keep encoding) while the GUI holds Pause
            cur = rq.get()
            if cur is None:
                break
            f_next = to_tensor(cur)                    # f[k+1]
            for tb in _gen_tweens(f_cur, f_next,       # tweens at t = k + j/M (strictly interior)
                                  _MTW[_skip:] if _skip else _MTW):
                last_out = tb
                wq.put(tb)
            _skip = 0
            last_out = to_bytes(f_next)                # real f[k+1] at its integer timestamp t = k+1
            wq.put(last_out)
            f_cur = f_next
            k += 1
            if k % 10 == 0:
                _progress(k, total_pairs)
    finally:
        _drain_pipes()
    # On-grid output is the real first frame plus M tweens/real per processed pair: multi*k + 1.
    _finish(f"done {k} pairs -> {out_path}\n", out_frames=args.multi * k + 1)

# ---------------------------------------------------------------------------------------------
# Legacy off-grid path: --fps mode only (GMFSS). --fps resamples to an arbitrary target fps
# whose output times do not line up on the source grid, so it keeps the offset scheme: every emitted
# frame is an interior blend on the _pair_fracs offset grid (no frame on a source timestamp), and
# the last source frame's slot is filled by holding the last generated frame. Integer --multi is
# handled on-grid above. See _pair_fracs and the module docstring.
prev = rq.get()
if prev is None:
    sys.exit("no frames decoded")
I0 = to_tensor(prev)    # f[0] (f[p] when resuming: the decoder skipped to the resume pair)
k = RESUME_SKIP_SRC
i = RESUME_SKIP_SRC
nout = RESUME_OUT_BASE  # exact count of frames written, for the GUI's OUTFRAMES total
_skip = RESUME_PAIR_SKIP  # banked slots of the first resumed pair (resume lands mid-pair)
last_out = None         # bytes of the most recent emitted frame, held across the final slot

try:
    while True:
        _check_pause()          # block here (queued frames keep encoding) while the GUI holds Pause
        cur = rq.get()
        if cur is None:
            break
        I1 = to_tensor(cur)
        # Uniform smoothing: every source pair is interpolated on the _pair_fracs slot grid, so
        # near-identical drawings (anime on twos/threes, held cels blurred only by encode noise,
        # or exact repeats) get exactly the same even motion as real motion. Nothing is held or
        # retimed - the source's own frame timings are preserved by construction, and the gap
        # between every pair of consecutive source frames is smoothed the same way. (--fps mode
        # can still leave a pair zero slots.)
        fracs = _pair_fracs(i)
        if _skip:
            fracs = fracs[_skip:]
            _skip = 0
        if fracs:
            with amp():
                reuse = model.reuse(I0, I1, scale)
                for fr in fracs:
                    last_out = to_bytes(model.inference(I0, I1, reuse, fr))
                    wq.put(last_out)
        nout += len(fracs)
        prev, I0 = cur, I1
        i += 1
        k += 1
        if k % 10 == 0:
            _progress(k, total_pairs)
    # Closing slot for the last source frame (i is now its index): its own time interval [i, i+1),
    # which has no frame after it to interpolate toward. Hold the last generated frame across it so
    # the output covers the full source duration and lands on exactly multi*frames (true doubling,
    # the target fps behaviour Topaz uses) instead of stopping one slot short. It is held, so it
    # stays soft and does not pop. A single decoded frame has no pair at all, so it just passes
    # through (still routed through to_bytes when sharpening/upscaling changes its dims, or when
    # the pipe carries the output depth and the raw decode bytes would be the wrong size).
    if last_out is None:
        # Single decoded frame, no pair to interpolate: pass it through. Never on a resumed run
        # (that frame is already banked; reaching here would mean the source delivered fewer
        # frames than the original run saw, and emitting would duplicate a banked frame).
        if not RESUME_ACTIVE:
            wq.put(to_bytes(I0) if (SHARPEN > 0 or UPSCALE or HDR_ACTIVE or RESTORE_ACTIVE
                                    or DEC_FMT != OUT_RAW_FMT) else prev)
            nout += 1
    else:
        tail = math.ceil((i + 1) * ratio - 0.5) - math.ceil(i * ratio - 0.5)  # this path is --fps only
        for _ in range(tail):
            wq.put(last_out)
        nout += tail
finally:
    _drain_pipes()
_finish(f"done {k} pairs -> {out_path}\n", out_frames=nout)
