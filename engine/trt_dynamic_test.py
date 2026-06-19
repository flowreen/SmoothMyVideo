"""
Measure the dynamic vs static TensorRT penalty on GMFlow itself.

Builds a static engine (fixed 544x960) and a dynamic engine (optimization profile
min/opt/max, opt=544x960) from the same autocast ONNX, then benchmarks both at the
opt shape, plus the dynamic one at an off-opt shape. Also tells us whether GMFlow
even exports with dynamic H/W (the risky part).

Run: engine/runtime/python.exe engine/trt_dynamic_test.py
"""
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "GMFSS_Fortuna")
sys.path.insert(0, REPO)
os.chdir(REPO)
torch.set_grad_enabled(False)
device = torch.device("cuda")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")


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
import tensorrt as trt  # noqa: E402
from model.gmflow.gmflow import GMFlow  # noqa: E402
from torch.export import Dim  # noqa: E402

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
_T2T = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16,
        trt.DataType.INT32: torch.int32, trt.DataType.INT64: torch.int64,
        trt.DataType.BOOL: torch.bool, trt.DataType.BF16: torch.bfloat16}

net = GMFlow().to(device).eval()
net.load_state_dict(torch.load("train_log/flownet.pkl", map_location=device))


def mk(h, w):
    a = torch.rand(1, 3, h, w, device=device)
    b = (torch.roll(a, (3, 5), (2, 3)) + 0.02 * torch.rand_like(a)).clamp(0, 1)
    return a, b


OPT = (544, 960)
a, b = mk(*OPT)
ONNX_S = os.path.join(HERE, "dyn_static.onnx")
ONNX_D = os.path.join(HERE, "dyn_dynamic.onnx")


def export(path, dynamic):
    kw = dict(input_names=["img0", "img1"], output_names=["flow"], dynamo=True, opset_version=18)
    if dynamic:
        H = Dim("H", min=256, max=1536)
        W = Dim("W", min=256, max=2048)
        kw["dynamic_shapes"] = ({2: H, 3: W}, {2: H, 3: W})
    with torch.autocast("cuda", dtype=torch.float16):
        torch.onnx.export(net, (a, b), path, **kw)


def build(onnx_path, profile=None):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    if not parser.parse_from_file(onnx_path):
        raise RuntimeError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    if profile:
        mn, opt, mx = profile
        prof = builder.create_optimization_profile()
        for n in ("img0", "img1"):
            prof.set_shape(n, mn, opt, mx)
        config.add_optimization_profile(prof)
    t0 = time.time()
    ser = builder.build_serialized_network(network, config)
    if ser is None:
        raise RuntimeError("build returned None")
    return bytes(ser), time.time() - t0


class Mod:
    def __init__(self, ser):
        self.rt = trt.Runtime(TRT_LOGGER)
        self.eng = self.rt.deserialize_cuda_engine(ser)
        self.ctx = self.eng.create_execution_context()
        self.ins, self.out = [], None
        for i in range(self.eng.num_io_tensors):
            n = self.eng.get_tensor_name(i)
            if self.eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                self.ins.append((n, _T2T[self.eng.get_tensor_dtype(n)]))
            else:
                self.out = (n, _T2T[self.eng.get_tensor_dtype(n)])

    def run(self, x0, x1):
        vals = {"img0": x0, "img1": x1}
        for n, d in self.ins:
            t = vals[n].to(d).contiguous()
            vals[n] = t
            self.ctx.set_input_shape(n, tuple(t.shape))
            self.ctx.set_tensor_address(n, t.data_ptr())
        oname, odt = self.out
        oshape = tuple(self.ctx.get_tensor_shape(oname))  # concrete after set_input_shape
        o = torch.empty(oshape, dtype=odt, device=device)
        self.ctx.set_tensor_address(oname, o.data_ptr())
        st = torch.cuda.current_stream()
        self.ctx.execute_async_v3(st.cuda_stream)
        st.synchronize()
        return o


def bench(fn, n=50, warmup=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000.0


print("=== static engine (fixed 544x960) ===")
export(ONNX_S, dynamic=False)
ser_s, bt = build(ONNX_S)
print(f"  built in {bt:.0f}s")
ms = Mod(ser_s)
s_opt = bench(lambda: ms.run(a, b))
print(f"  static @544x960: {s_opt:.1f} ms")

print("=== dynamic engine (profile min 256x256 / opt 544x960 / max 1536x2048) ===")
try:
    export(ONNX_D, dynamic=True)
    ser_d, bt = build(ONNX_D, profile=((1, 3, 256, 256), (1, 3, 544, 960), (1, 3, 1536, 2048)))
    print(f"  built in {bt:.0f}s")
    md = Mod(ser_d)
    d_opt = bench(lambda: md.run(a, b))
    a2, b2 = mk(384, 640)
    d_off = bench(lambda: md.run(a2, b2))
    print(f"  dynamic @544x960 (opt): {d_opt:.1f} ms   ->  {(d_opt / s_opt - 1) * 100:+.1f}% vs static")
    print(f"  dynamic @384x640 (off-opt): {d_off:.1f} ms")
except Exception as e:  # noqa: BLE001
    import traceback
    traceback.print_exc()
    print("\n[!] dynamic export/build FAILED -> GMFlow does not go dynamic cleanly:")
    print("   ", repr(e)[:400])
finally:
    for p in (ONNX_S, ONNX_S + ".data", ONNX_D, ONNX_D + ".data"):
        try:
            os.remove(p)
        except OSError:
            pass
