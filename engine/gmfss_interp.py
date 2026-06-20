"""
GMFSS pipe interpolation engine for SmoothMyVideo.
ffmpeg decode -> GMFSS anime union model -> ffmpeg encode (audio copied).
Streams frames so there is no PNG folder. Prints "PROGRESS k/total" to stderr for the GUI.

Performance first, no quality knobs: the pipeline always runs fp16, always targets visually
lossless, and always uses the fastest backend the machine supports.
- Backend: TensorRT engines by default (built+cached per resolution on first run), with
  automatic eager fallback when TensorRT is unavailable. Pass --no-trt to force eager.
- Encoder: the matching NVENC for the source codec (h264/hevc/av1) when the device has a
  usable NVENC session, otherwise an automatic CPU fallback to SVT-AV1 (the strongest
  visually lossless software encoder in the bundled LGPL ffmpeg; x264/x265 are not built in).
  Source bit depth (8/10 bit), chroma and colour signalling are preserved either way.

Usage: gmfss_interp.py <input> <multi> [output] [--scale 1.0] [--fps TARGET] [--no-trt]
       --fps overrides <multi>, resampling the timeline to TARGET output fps.
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

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GMFSS_Fortuna")
# Prefer ffmpeg/ffprobe bundled at engine/bin so a packaged build needs no system
# ffmpeg on PATH; fall back to the bare PATH names for dev.
_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin")
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
args = ap.parse_args()

inp = os.path.abspath(args.input)

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
if FPS_MODE:
    ratio = args.fps / src_fps                 # output frames per source frame
    rate_str = f"{args.fps:g}"
    out_label = int(round(args.fps))
else:
    rate_str = f"{num * args.multi}/{den}"
    out_label = int(round(src_fps * args.multi))
out_path = os.path.abspath(args.output) if args.output else \
    os.path.splitext(inp)[0] + f"_{out_label}fps.mp4"

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
total_pairs = max(1, NB - 1) if NB else 0

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

scale = args.scale
tmp = max(64, int(64 / scale))
ph = ((H - 1) // tmp + 1) * tmp
pw = ((W - 1) // tmp + 1) * tmp

def amp():
    return torch.autocast("cuda", dtype=torch.float16)

def to_tensor(buf):
    a = np.frombuffer(buf, NP_DT).reshape(H, W, 3)
    t = torch.from_numpy(a.transpose(2, 0, 1).copy()).to(device).unsqueeze(0).float() / MAXV
    # Reach the multiple of 64 the model needs by padding the bottom/right edge, not by
    # resizing the whole frame up and back. A bilinear resize (the old path here and in
    # to_bytes) resamples every pixel and softens the entire image; padding then cropping
    # leaves all real content bit-untouched, so the generated frames are strictly sharper.
    # replicate (vs zero) extends the edge smoothly so the flow net has no hard border to track.
    return F.pad(t, (0, pw - W, 0, ph - H), mode="replicate")

def to_bytes(t):
    t = t.float()[..., :H, :W]            # crop off the padding added in to_tensor
    # Round to nearest, not truncate: numpy's float->uint cast floors, which biases every
    # interpolated frame ~0.5 LSB low against the byte-exact source frames it is interleaved
    # with, a structured source-vs-tween brightness step. Rounding removes that bias.
    a = (t[0] * MAXV).round().clamp(0, MAXV).cpu().numpy().transpose(1, 2, 0)
    return a.astype(NP_DT).tobytes()

def make_inference(I0, I1, reuse, n):
    if model.version >= 3.9:
        return [model.inference(I0, I1, reuse, (i + 1) * 1. / (n + 1)) for i in range(n)]
    middle = model.inference(I0, I1, scale)
    if n == 1:
        return [middle]
    first = make_inference(I0, middle, reuse, n // 2)
    second = make_inference(middle, I1, reuse, n // 2)
    return [*first, middle, *second] if n % 2 else [*first, *second]

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

# Match the encoder family to the source codec so output stays in kind (h264 -> h264,
# hevc -> hevc, av1 -> av1); anything else defaults to hevc. NVENC H.264 is 8 bit only,
# so a 10 bit source that came in as h264 is promoted to HEVC main10.
_CODEC_ENC = {"h264": "h264_nvenc", "avc": "h264_nvenc", "hevc": "hevc_nvenc",
              "h265": "hevc_nvenc", "av1": "av1_nvenc"}
venc = _CODEC_ENC.get(SRC_CODEC, "hevc_nvenc")
if TEN_BIT and venc == "h264_nvenc":
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
if TEN_BIT:
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

prof = ["-profile:v", "main10"] if (TEN_BIT and venc == "hevc_nvenc") else []

# Carry the source colour signalling through. NVENC ignores the bare -color_* output flags
# for transfer/primaries (verified: only matrix and range stick), which would strip HDR
# signalling, so the values are stamped onto the frames with setparams before the pixel
# conversion, and the -color_* flags are kept too so the mp4 'colr' atom is written. This
# also makes the RGB -> YUV conversion use the source matrix instead of swscale's guess.
# The values come straight from ffprobe of this same ffmpeg, so they are valid filter input.
sp, color = [], []
for sp_opt, flag, key in (("range", "-color_range", "color_range"),
                          ("colorspace", "-colorspace", "color_space"),
                          ("color_trc", "-color_trc", "color_transfer"),
                          ("color_primaries", "-color_primaries", "color_primaries")):
    v = _tag(key)
    if v:
        sp.append(f"{sp_opt}={v}")
        color += [flag, v]
vf = ",".join((["setparams=" + ":".join(sp)] if sp else []) + [f"format={out_pix}"])

enc_cmd = [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", DEC_FMT,
           "-s", f"{W}x{H}", "-r", rate_str, "-i", "-", "-i", inp,
           "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "copy",
           "-c:v", venc, "-vf", vf]
enc_cmd += qargs + prof + color + [out_path]
enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, creationflags=NO_WINDOW)
sys.stderr.write(f"encode: {venc} visually-lossless -> {out_pix}  "
                 f"(source {SRC_CODEC or '?'} {SRC_BITS}bit {SRC_PIX})\n"); sys.stderr.flush()

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

n = args.multi - 1
prev = rq.get()
if prev is None:
    sys.exit("no frames decoded")
I0 = to_tensor(prev)
k = 0
i = 0
dups = 0
try:
    while True:
        cur = rq.get()
        if cur is None:
            break
        I1 = to_tensor(cur)
        # Exact duplicate source frames (anime is drawn on twos/threes, so held cels decode
        # byte-identically) need no interpolation: the correct tween of a still is the still
        # itself. Holding the frame skips the wasted flow+inference and the shimmer GMFSS can
        # add when fed two identical frames. The bytes compare short-circuits, so it is ~free
        # on real motion.
        dup = cur == prev
        if dup:
            dups += 1
        if FPS_MODE:
            # emit the output frames whose time falls in [i, i+1) source-frame units
            fracs = [j / ratio - i for j in range(math.ceil(i * ratio), math.ceil((i + 1) * ratio))]
            need = any(fr > 1e-6 for fr in fracs)
            with amp():
                reuse = model.reuse(I0, I1, scale) if (need and not dup) else None
                for fr in fracs:
                    if dup or fr <= 1e-6:
                        wq.put(prev)            # held frame, or coincides with source i
                    else:
                        wq.put(to_bytes(model.inference(I0, I1, reuse, fr)))
        else:
            wq.put(prev)
            if dup:
                for _ in range(n):
                    wq.put(prev)
            else:
                with amp():
                    reuse = model.reuse(I0, I1, scale)
                    mids = make_inference(I0, I1, reuse, n)
                for m in mids:
                    wq.put(to_bytes(m))
        prev, I0 = cur, I1
        i += 1
        k += 1
        if k % 10 == 0:
            sys.stderr.write(f"PROGRESS {k}/{total_pairs}\n"); sys.stderr.flush()
    wq.put(prev)
finally:
    wq.put(None)            # sentinel: let the writer drain its queue and exit
    wt.join()
    enc.stdin.close()
    dec.stdout.close()
    enc.wait()
    dec.wait()
if _werr:
    raise _werr[0]          # surface a failed encode pipe as a nonzero exit
sys.stderr.write(f"done {k} pairs ({dups} held as duplicates) -> {out_path}\n")
