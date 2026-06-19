"""
Isolated end to end check of the TRT integration: run one frame pair through
reuse() + inference() in eager and in TRT mode, compare the interpolated frame,
and time both. Exercises all 5 sub net wrappers + the eager softsplat glue.

Run: engine/runtime/python.exe engine/trt_integration_test.py [H W]
Default 1088x1920 (padded 1080p).
"""
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "GMFSS_Fortuna")
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
os.chdir(REPO)
torch.set_grad_enabled(False)
device = torch.device("cuda")


def _add_cuda_dll_dirs():
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
from model.GMFSS_infer_u import Model  # noqa: E402
import trt_runtime  # noqa: E402

H = int(sys.argv[1]) if len(sys.argv) > 1 else 1088
W = int(sys.argv[2]) if len(sys.argv) > 2 else 1920
I0 = torch.rand(1, 3, H, W, device=device)
I1 = (torch.roll(I0, shifts=(3, 5), dims=(2, 3)) + 0.02 * torch.rand_like(I0)).clamp(0, 1)


def make_model():
    m = Model()
    if not hasattr(m, "version"):
        m.version = 0
    m.load_model("train_log", -1)
    m.eval()
    m.device()
    return m


def run(m):
    with torch.autocast("cuda", dtype=torch.float16):
        reuse = m.reuse(I0, I1, 1.0)
        return m.inference(I0, I1, reuse, 0.5)


def bench(fn, n=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000.0


print("=== eager ===")
me = make_model()
out_e = run(me)
torch.cuda.synchronize()
te = bench(lambda: run(me))
print(f"eager reuse+inference: {te:.1f} ms")

print("=== tensorrt (first call builds + caches engines) ===")
mt = make_model()
trt_runtime.trtify(mt)
t0 = time.time()
out_t = run(mt)
torch.cuda.synchronize()
print(f"first TRT pass (incl. build): {time.time() - t0:.0f} s")
tt = bench(lambda: run(mt))

d = (out_e.float() - out_t.float()).abs()
print("=== result ===")
print(f"interpolated frame: maxdiff {d.max().item():.4e}  mean {d.mean().item():.4e}  "
      f"NaN={torch.isnan(out_t).any().item()}")
print(f"eager   : {te:.1f} ms / pair-timestep")
print(f"tensorrt: {tt:.1f} ms / pair-timestep")
print(f"speedup : {te / tt:.2f}x")
