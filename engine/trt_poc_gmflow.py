"""
TensorRT feasibility test for GMFlow (the heavy optical flow net, run twice per
pair). Structural export is already proven; this nails down fp16.

fp16 strategy: a blanket .half() crashes because normalize_img injects fp32
constants, and would also risk fp16 overflow in softmax/correlation. So we export
the fp32 model UNDER autocast(fp16) via the dynamo exporter, which bakes the app's
exact mixed precision (fp16 convs/matmuls, fp32 softmax) into the ONNX. A strongly
typed TRT engine then honors that.

MODE = "autocast" (mixed fp16) or "fp32" (baseline).
Run: engine/runtime/python.exe engine/trt_poc_gmflow.py [H W]
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

MODE = os.environ.get("TRT_MODE", "autocast")  # "autocast" or "fp32"
print("TensorRT", trt.__version__, "| torch", torch.__version__, "| MODE", MODE)

H = int(sys.argv[1]) if len(sys.argv) > 1 else 544
W = int(sys.argv[2]) if len(sys.argv) > 2 else 960
SHAPE = (1, 3, H, W)
ONNX = os.path.join(HERE, f"gmflow_{H}x{W}_{MODE}.onnx")
ENGINE = os.path.join(HERE, f"gmflow_{H}x{W}_{MODE}.engine")
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
_T2T = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16,
        trt.DataType.INT32: torch.int32, trt.DataType.INT64: torch.int64,
        trt.DataType.BOOL: torch.bool, trt.DataType.BF16: torch.bfloat16}

net = GMFlow().to(device).eval()
net.load_state_dict(torch.load("train_log/flownet.pkl", map_location=device))

# correlated inputs so optical flow is well defined (a small shift), making the
# numeric check meaningful (random noise -> unstable garbage flow).
img0 = torch.rand(SHAPE)
img1 = (torch.roll(img0, shifts=(3, 5), dims=(2, 3)) + 0.02 * torch.rand(SHAPE)).clamp(0, 1)
a32, c32 = img0.to(device), img1.to(device)
in_tensors = {"img0": a32, "img1": c32}


def export_onnx():
    names = dict(input_names=["img0", "img1"], output_names=["flow"], opset_version=18)
    if MODE == "autocast":
        with torch.autocast("cuda", dtype=torch.float16):
            torch.onnx.export(net, (a32, c32), ONNX, dynamo=True, **names)
    else:
        torch.onnx.export(net, (a32, c32), ONNX, dynamo=True, **names)


if not os.path.isfile(ONNX):
    export_onnx()
    print("onnx exported ->", ONNX)
else:
    print("onnx cached ->", ONNX)


def build_engine():
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    if not parser.parse_from_file(ONNX):  # resolves external .onnx.data weights
        for i in range(parser.num_errors):
            print("  parser error:", parser.get_error(i))
        raise RuntimeError("onnx parse failed")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 6 << 30)
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


def run_trt():
    for n, d, s in inputs:
        context.set_input_shape(n, s)
        context.set_tensor_address(n, in_tensors[n].to(d).contiguous().data_ptr())
    for n, _, _ in outputs:
        context.set_tensor_address(n, out_t[n].data_ptr())
    st = torch.cuda.current_stream()
    context.execute_async_v3(st.cuda_stream)
    st.synchronize()
    return out_t[outputs[0][0]]


def eager_call():
    with torch.autocast("cuda", dtype=torch.float16):
        return net(a32, c32)


eager = eager_call()
trt_out = run_trt()
d = (eager.float() - trt_out.float()).abs()
print("--- numeric check (eager autocast vs trt %s) ---" % MODE)
print(f"  flow: maxdiff {d.max().item():.4e}  mean {d.mean().item():.4e}  "
      f"(flow range {eager.float().min().item():.2f}..{eager.float().max().item():.2f})  "
      f"NaN={torch.isnan(trt_out).any().item()} Inf={torch.isinf(trt_out).any().item()}")


def bench(fn, n=40, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000.0


e_ms = bench(eager_call)
t_ms = bench(run_trt)
print("--- benchmark GMFlow @ %dx%d (one direction, MODE=%s) ---" % (H, W, MODE))
print(f"  eager fp16 : {e_ms:.2f} ms/call")
print(f"  tensorrt   : {t_ms:.2f} ms/call")
print(f"  speedup    : {e_ms / t_ms:.2f}x")
