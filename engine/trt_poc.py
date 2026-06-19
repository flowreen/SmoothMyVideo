"""
TensorRT proof of concept for SmoothMyVideo.

Validates the native TRT path on the simplest GMFSS sub network (FeatureNet,
all Conv2d + PReLU): export to fp16 ONNX, build a strongly typed TRT engine,
check the output matches eager fp16, and benchmark both. This proves the
plumbing (ONNX -> TRT 11 build -> zero copy GPU bindings against torch tensors)
before porting the heavy nets (GMFlow especially).

TensorRT 11 notes: the FP16/INT8 BuilderFlags are gone; precision now comes from
the network types, so we export an fp16 ONNX and build a STRONGLY_TYPED network.

Run: engine/runtime/python.exe engine/trt_poc.py [H W]
Default shape is padded 1080p (1088x1920).
"""
import os
import sys
import time
import copy

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "GMFSS_Fortuna")
sys.path.insert(0, REPO)
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
import tensorrt as trt  # noqa: E402
from model.FeatureNet import FeatureNet  # noqa: E402

print("TensorRT", trt.__version__, "| torch", torch.__version__,
      "| dev", torch.cuda.get_device_name())

H = int(sys.argv[1]) if len(sys.argv) > 1 else 1088
W = int(sys.argv[2]) if len(sys.argv) > 2 else 1920
SHAPE = (1, 3, H, W)
ONNX = os.path.join(HERE, f"feat_{H}x{W}.onnx")
ENGINE = os.path.join(HERE, f"feat_{H}x{W}_fp16.engine")
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

_T2T = {
    trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16,
    trt.DataType.BF16: torch.bfloat16, trt.DataType.INT32: torch.int32,
    trt.DataType.INT64: torch.int64, trt.DataType.BOOL: torch.bool,
}

# ---- model (fp32 weights, eager path uses autocast to match the app) ----
net = FeatureNet().to(device).eval()
net.load_state_dict(torch.load("train_log/feat.pkl", map_location=device))
base = torch.randn(SHAPE)
x32 = base.to(device)                 # eager input (autocast handles fp16)
x16 = base.to(device).half()          # trt input (engine IO is fp16)


# ---- 1) export fp16 ONNX (strongly typed TRT needs fp16 in the graph) ----
def export_onnx():
    half_net = copy.deepcopy(net).half().eval()
    names = dict(input_names=["x"], output_names=["f1", "f2", "f3"], opset_version=17)
    try:
        torch.onnx.export(half_net, (x16,), ONNX, dynamo=False, **names)
        return "legacy"
    except Exception as e:  # noqa: BLE001
        print("legacy export failed -> trying dynamo:", repr(e)[:200])
        torch.onnx.export(half_net, (x16,), ONNX, dynamo=True, **names)
        return "dynamo"


if not os.path.isfile(ONNX):
    print("onnx export via", export_onnx(), "->", ONNX)
else:
    print("onnx cached ->", ONNX)


# ---- 2) build strongly typed TRT engine ----
def build_engine():
    builder = trt.Builder(TRT_LOGGER)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(ONNX, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("  parser error:", parser.get_error(i))
            raise RuntimeError("onnx parse failed")
    config = builder.create_builder_config()
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    except Exception as e:  # noqa: BLE001
        print("  workspace limit skipped:", repr(e)[:120])
    t0 = time.time()
    ser = builder.build_serialized_network(network, config)
    if ser is None:
        raise RuntimeError("build_serialized_network returned None")
    print(f"  engine built in {time.time() - t0:.1f}s")
    data = bytes(ser)
    with open(ENGINE, "wb") as f:
        f.write(data)
    return data


if os.path.isfile(ENGINE):
    with open(ENGINE, "rb") as f:
        serialized = f.read()
    print("engine cached ->", ENGINE)
else:
    serialized = build_engine()

# ---- 3) load + bind ----
runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(serialized)
context = engine.create_execution_context()

inputs, outputs = [], []
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    spec = (name, _T2T[engine.get_tensor_dtype(name)], tuple(engine.get_tensor_shape(name)))
    (inputs if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else outputs).append(spec)
print("inputs:", inputs)
print("outputs:", outputs)
out_t = {n: torch.empty(s, dtype=d, device=device) for n, d, s in outputs}


def run_trt(x):
    in_name, in_dtype, in_shape = inputs[0]
    context.set_input_shape(in_name, in_shape)
    context.set_tensor_address(in_name, x.to(in_dtype).contiguous().data_ptr())
    for n, _, _ in outputs:
        context.set_tensor_address(n, out_t[n].data_ptr())
    s = torch.cuda.current_stream()
    context.execute_async_v3(s.cuda_stream)
    s.synchronize()
    return [out_t[n] for n, _, _ in outputs]


# ---- 4) numeric check vs eager fp16 (autocast, matching the app) ----
def eager_call():
    with torch.autocast("cuda", dtype=torch.float16):
        return net(x32)


eager = eager_call()
trt_out = run_trt(x16)
print("--- numeric check (eager fp16 vs trt fp16) ---")
for e, (n, _, _) in zip(eager, outputs):
    t = out_t[n]
    d = (e.float() - t.float()).abs()
    print(f"  {n}: maxdiff {d.max().item():.4e}  mean {d.mean().item():.4e}  (signal max {e.float().abs().max().item():.3f})")


# ---- 5) benchmark ----
def bench(fn, n=60, warmup=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000.0


e_ms = bench(eager_call)
t_ms = bench(lambda: run_trt(x16))
print("--- benchmark FeatureNet @ %dx%d ---" % (H, W))
print(f"  eager fp16 : {e_ms:.2f} ms/call")
print(f"  tensorrt   : {t_ms:.2f} ms/call")
print(f"  speedup    : {e_ms / t_ms:.2f}x")
