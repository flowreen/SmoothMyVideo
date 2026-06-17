"""
GMFSS speed benchmark for SmoothMyVideo.
Times the core per-frame inference at 1080p (warmup + cuda.synchronize), for fp32 and fp16,
on a real frame pair from samples/gtest.mp4. Appends a dated entry to BENCHMARKS.md so we can
track speedups over time (fp16 -> cupy -> torch.compile -> TensorRT).

Metric: pure GMFSS reuse() + inference() time. Excludes model load and ffmpeg I/O on purpose.
Run: engine/.venv/Scripts/python.exe engine/benchmark.py
"""
import os
import sys
import time
import json
import datetime
import contextlib
import subprocess
import numpy as np
import torch
from torch.nn import functional as F
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "GMFSS_Fortuna")
SAMPLE = os.path.normpath(os.path.join(HERE, "..", "samples", "gtest.mp4"))
LOG = os.path.normpath(os.path.join(HERE, "..", "BENCHMARKS.md"))
N, WARMUP = 20, 3

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

t0 = time.time()
model = Model()
if not hasattr(model, "version"):
    model.version = 0
model.load_model("train_log", -1)
model.eval()
model.device()
load_s = time.time() - t0

cap = cv2.VideoCapture(SAMPLE)
_, f0 = cap.read()
_, f1 = cap.read()
cap.release()
H, W = f0.shape[:2]
tmp = 64
ph = ((H - 1) // tmp + 1) * tmp
pw = ((W - 1) // tmp + 1) * tmp

def prep(bgr):
    rgb = bgr[:, :, ::-1]
    t = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).to(device).unsqueeze(0).float() / 255.
    return F.interpolate(t, (ph, pw), mode="bilinear", align_corners=False)

I0, I1 = prep(f0), prep(f1)

def amp(fp16):
    return torch.autocast("cuda", dtype=torch.float16) if fp16 else contextlib.nullcontext()

def bench(fp16):
    for _ in range(WARMUP):
        with amp(fp16):
            r = model.reuse(I0, I1, 1.0)
            _ = model.inference(I0, I1, r, 0.5)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(N):
        with amp(fp16):
            r = model.reuse(I0, I1, 1.0)
    torch.cuda.synchronize()
    reuse_ms = (time.time() - t) / N * 1000
    with amp(fp16):
        r = model.reuse(I0, I1, 1.0)
    torch.cuda.synchronize()
    t = time.time()
    for i in range(N):
        with amp(fp16):
            _ = model.inference(I0, I1, r, (i + 1) / (N + 1))
    torch.cuda.synchronize()
    inf_ms = (time.time() - t) / N * 1000
    return reuse_ms, inf_ms

results = {}
for name, fp in [("fp16", True)]:
    try:
        rms, ims = bench(fp)
        per_pair_16x = rms + 15 * ims          # one pair at 16x = 1 reuse + 15 interp frames
        results[name] = {
            "reuse_ms": round(rms, 1),
            "inference_ms_per_frame": round(ims, 1),
            "per_pair_16x_ms": round(per_pair_16x, 1),
            "est_360frame_16x_min": round(359 * per_pair_16x / 1000 / 60, 1),
        }
    except Exception as e:
        results[name] = {"error": repr(e)[:200]}

try:
    commit = subprocess.check_output(["git", "-C", os.path.join(HERE, ".."), "rev-parse", "--short", "HEAD"],
                                     text=True, stderr=subprocess.DEVNULL).strip()
except Exception:
    commit = "n/a"

gpu = torch.cuda.get_device_name(0)
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
print(json.dumps(results, indent=2))

entry = [f"\n## {now}  (commit {commit})",
         f"- GPU: {gpu} | torch {torch.__version__} | model load {load_s:.1f}s | sample {W}x{H}"]
for k, v in results.items():
    if "error" in v:
        entry.append(f"- **{k}**: ERROR {v['error']}")
    else:
        entry.append(f"- **{k}**: reuse {v['reuse_ms']}ms, inference {v['inference_ms_per_frame']}ms/frame, "
                     f"pair@16x {v['per_pair_16x_ms']}ms, est 360f@16x ~{v['est_360frame_16x_min']}min")

header = ("# SmoothMyVideo GMFSS speed benchmarks\n\n"
          "Core GMFSS inference timing at 1080p (warmup + cuda.synchronize), excludes model load and ffmpeg I/O.\n"
          "Lower is better. Each entry is a progress point as speedups land (fp16, cupy, torch.compile, TensorRT).\n")
if not os.path.exists(LOG):
    open(LOG, "w", encoding="utf-8").write(header)
open(LOG, "a", encoding="utf-8").write("\n".join(entry) + "\n")
print("logged to", LOG)
