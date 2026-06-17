"""
GMFSS pipe interpolation engine for SmoothMyVideo.
ffmpeg decode (rgb24) -> GMFSS anime union model -> ffmpeg encode (hevc_nvenc, audio copied).
Streams frames so there is no PNG folder. Prints "PROGRESS k/total" to stderr for the GUI.

Usage: gmfss_interp.py <input> <multi> [output] [--scale 1.0]   (always runs fp16)
"""
import os
import sys
import argparse
import subprocess
import numpy as np
import torch
from torch.nn import functional as F

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GMFSS_Fortuna")
FFMPEG, FFPROBE = "ffmpeg", "ffprobe"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

ap = argparse.ArgumentParser()
ap.add_argument("input")
ap.add_argument("multi", type=int)
ap.add_argument("output", nargs="?", default=None)
ap.add_argument("--scale", type=float, default=1.0)
args = ap.parse_args()

inp = os.path.abspath(args.input)

def probe(path):
    out = subprocess.check_output(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
         "-of", "csv=p=0", path], text=True, creationflags=NO_WINDOW).strip().split(",")
    w, h, rate = int(out[0]), int(out[1]), out[2]
    num, den = (rate.split("/") + ["1"])[:2]
    nb = int(out[3]) if len(out) > 3 and out[3].isdigit() else 0
    return w, h, int(num), int(den), nb

W, H, num, den, NB = probe(inp)
out_path = os.path.abspath(args.output) if args.output else \
    os.path.splitext(inp)[0] + f"_{int(round(num / den * args.multi))}fps.mp4"
out_num = num * args.multi
fsize = W * H * 3
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

scale = args.scale
tmp = max(64, int(64 / scale))
ph = ((H - 1) // tmp + 1) * tmp
pw = ((W - 1) // tmp + 1) * tmp

def amp():
    return torch.autocast("cuda", dtype=torch.float16)

def to_tensor(buf):
    a = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    t = torch.from_numpy(a.transpose(2, 0, 1).copy()).to(device).unsqueeze(0).float() / 255.
    return F.interpolate(t, (ph, pw), mode="bilinear", align_corners=False)

def to_bytes(t):
    t = F.interpolate(t.float(), (H, W), mode="bilinear", align_corners=False)
    return (t[0] * 255.).clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0).tobytes()

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
    [FFMPEG, "-v", "error", "-i", inp, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
    stdout=subprocess.PIPE, creationflags=NO_WINDOW)
enc_cmd = [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", f"{out_num}/{den}", "-i", "-", "-i", inp,
           "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "copy",
           "-c:v", "hevc_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "18",
           "-b:v", "0", "-pix_fmt", "yuv420p", out_path]
enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, creationflags=NO_WINDOW)

n = args.multi - 1
prev = read_exact(dec.stdout, fsize)
if prev is None:
    sys.exit("no frames decoded")
I0 = to_tensor(prev)
k = 0
try:
    while True:
        cur = read_exact(dec.stdout, fsize)
        if cur is None:
            break
        I1 = to_tensor(cur)
        with amp():
            reuse = model.reuse(I0, I1, scale)
            mids = make_inference(I0, I1, reuse, n)
        enc.stdin.write(prev)
        for m in mids:
            enc.stdin.write(to_bytes(m))
        prev, I0 = cur, I1
        k += 1
        if k % 10 == 0:
            sys.stderr.write(f"PROGRESS {k}/{total_pairs}\n"); sys.stderr.flush()
    enc.stdin.write(prev)
finally:
    enc.stdin.close()
    dec.stdout.close()
    enc.wait()
    dec.wait()
sys.stderr.write(f"done {k} pairs -> {out_path}\n")
