"""
TensorRT backend for the GMFSS sub networks.

Strategy (validated in trt_poc.py / trt_poc_gmflow.py): each sub net is exported
to ONNX under autocast(fp16) via the dynamo exporter (mixed fp16/fp32 matching the
app's precision), then built into a strongly typed TRT engine. softsplat (cupy) and
the F.interpolate glue stay in eager. Engines are built on first use for a given
input resolution and cached on disk per (net, shapes, gpu, trt version).

trtify(model) swaps model.feat_ext / flownet / metricnet / ifnet / fusionnet for
wrappers with identical call signatures, so GMFSS_infer_u is untouched. Any export
or build failure falls back to the original eager module, so the app never breaks.
"""
import os
import sys
import time

import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import tensorrt as trt

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.environ.get("SMV_TRT_CACHE") or os.path.join(HERE, "trt_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
_T2T = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16,
        trt.DataType.INT32: torch.int32, trt.DataType.INT64: torch.int64,
        trt.DataType.BOOL: torch.bool, trt.DataType.BF16: torch.bfloat16}


def _gpu_tag():
    name = torch.cuda.get_device_name().replace(" ", "")
    return f"{name}-trt{trt.__version__}".replace(".", "_")


def _shape_tag(tensors):
    return "_".join("x".join(map(str, t.shape)) for t in tensors)


def _log(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


class TRTModule:
    """A built engine; binds torch cuda tensors zero copy and runs it."""

    def __init__(self, serialized):
        self.runtime = trt.Runtime(TRT_LOGGER)
        self.engine = self.runtime.deserialize_cuda_engine(serialized)
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()  # dedicated non-default stream (TRT warns on the default one)
        self.inputs, self.outputs = [], []
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            spec = (n, _T2T[self.engine.get_tensor_dtype(n)], tuple(self.engine.get_tensor_shape(n)))
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                self.inputs.append(spec)
            else:
                self.outputs.append(spec)

    def __call__(self, *args):
        held = []  # keep cast/contiguous temporaries alive through execution
        for (n, d, s), a in zip(self.inputs, args):
            a = a.to(d).contiguous()
            held.append(a)
            self.context.set_input_shape(n, s)
            self.context.set_tensor_address(n, a.data_ptr())
        outs = []
        for n, d, s in self.outputs:
            o = torch.empty(s, dtype=d, device="cuda")  # fresh each call; outputs may persist
            outs.append(o)
            self.context.set_tensor_address(n, o.data_ptr())
        self.stream.wait_stream(torch.cuda.current_stream())  # inputs prepared on the caller's stream are ready first
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()                             # block until outputs are materialized (as before)
        return outs[0] if len(outs) == 1 else tuple(outs)


def _build_serialized(onnx_path):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    if not parser.parse_from_file(onnx_path):  # resolves external .onnx.data weights
        errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError("onnx parse failed: " + errs)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 6 << 30)
    ser = builder.build_serialized_network(network, config)
    if ser is None:
        raise RuntimeError("build_serialized_network returned None")
    return bytes(ser)


def build_or_load(name, export_module, example_inputs, input_names, output_names):
    """Return a TRTModule for export_module at these input shapes, building and
    caching (autocast fp16 ONNX -> strongly typed engine) on a cache miss."""
    tag = f"{name}_{_shape_tag(example_inputs)}_{_gpu_tag()}"
    engine_path = os.path.join(CACHE_DIR, tag + ".engine")
    if os.path.isfile(engine_path):
        with open(engine_path, "rb") as f:
            return TRTModule(f.read())
    onnx_path = os.path.join(CACHE_DIR, tag + ".onnx")
    _log(f"[trt] building {name} {_shape_tag(example_inputs)} (one time for this resolution)...")
    t0 = time.time()
    with torch.autocast("cuda", dtype=torch.float16):
        torch.onnx.export(export_module, tuple(example_inputs), onnx_path,
                          input_names=input_names, output_names=output_names,
                          dynamo=True, opset_version=18)
    serialized = _build_serialized(onnx_path)
    with open(engine_path, "wb") as f:
        f.write(serialized)
    _log(f"[trt] {name} built in {time.time() - t0:.0f}s")
    for p in (onnx_path, onnx_path + ".data"):  # engine is self contained; drop the onnx
        try:
            os.remove(p)
        except OSError:
            pass
    return TRTModule(serialized)


class _Engine:
    """Caches built TRTModules per input shape; falls back to eager on failure."""

    def __init__(self, name, eager, input_names, output_names):
        self.name = name
        self.eager = eager
        self.input_names = input_names
        self.output_names = output_names
        self.cache = {}
        self.eager_only = False

    def _run(self, export_module, tensors, call_args):
        if self.eager_only:
            return self.eager(*call_args)
        try:
            key = _shape_tag(tensors)
            mod = self.cache.get(key)
            if mod is None:
                mod = build_or_load(self.name, export_module, tensors,
                                    self.input_names, self.output_names)
                self.cache[key] = mod
            return mod(*tensors)
        except Exception as e:  # noqa: BLE001
            _log(f"[trt] {self.name} fell back to eager: {repr(e)[:240]}")
            self.eager_only = True
            return self.eager(*call_args)


class FeatEngine(_Engine):
    def __init__(self, eager):
        super().__init__("featurenet", eager, ["x"], ["f1", "f2", "f3"])

    def __call__(self, x):
        return self._run(self.eager, (x,), (x,))


class FlowEngine(_Engine):
    def __init__(self, eager):
        super().__init__("gmflow", eager, ["img0", "img1"], ["flow"])

    def __call__(self, i0, i1):
        return self._run(self.eager, (i0, i1), (i0, i1))


class MetricEngine(_Engine):
    def __init__(self, eager):
        super().__init__("metricnet", eager, ["i0", "i1", "f01", "f10"], ["m0", "m1"])

    def __call__(self, i0, i1, f01, f10):
        return self._run(self.eager, (i0, i1, f01, f10), (i0, i1, f01, f10))


class FusionEngine(_Engine):
    def __init__(self, eager):
        super().__init__("fusionnet", eager, ["a", "b", "c", "d"], ["out"])

    def __call__(self, a, b, c, d):
        return self._run(self.eager, (a, b, c, d), (a, b, c, d))


class _IFNetExport(nn.Module):
    """IFNet with scale_list baked and timestep as a (1,1,1,1) tensor input."""

    def __init__(self, ifnet, scale_list):
        super().__init__()
        self.ifnet = ifnet
        self.scale_list = scale_list

    def forward(self, x, timestep):
        return self.ifnet(x, timestep, scale_list=self.scale_list)


class IFNetEngine(_Engine):
    def __init__(self, eager):
        super().__init__("ifnet", eager, ["x", "timestep"], ["merged"])

    def __call__(self, x, timestep, scale_list=(8, 4, 2, 1)):
        if self.eager_only:
            return self.eager(x, timestep, scale_list=list(scale_list))
        ts = x.new_full((1, 1, 1, 1), float(timestep))
        export_mod = _IFNetExport(self.eager, list(scale_list))
        return self._run(export_mod, (x, ts), (x, timestep))

    def _run(self, export_module, tensors, call_args):  # eager fallback needs the kwarg
        if self.eager_only:
            return self.eager(call_args[0], call_args[1], scale_list=[8, 4, 2, 1])
        try:
            key = _shape_tag(tensors)
            mod = self.cache.get(key)
            if mod is None:
                mod = build_or_load(self.name, export_module, tensors,
                                    self.input_names, self.output_names)
                self.cache[key] = mod
            return mod(*tensors)
        except Exception as e:  # noqa: BLE001
            _log(f"[trt] {self.name} fell back to eager: {repr(e)[:240]}")
            self.eager_only = True
            return self.eager(call_args[0], call_args[1], scale_list=[8, 4, 2, 1])


def trtify(model):
    """Swap a loaded GMFSS Model's sub nets for TRT wrappers (in place)."""
    model.feat_ext = FeatEngine(model.feat_ext)
    model.flownet = FlowEngine(model.flownet)
    model.metricnet = MetricEngine(model.metricnet)
    model.ifnet = IFNetEngine(model.ifnet)
    model.fusionnet = FusionEngine(model.fusionnet)
    _log("[trt] model wrapped with TensorRT engines (build on first frame)")
    return model
