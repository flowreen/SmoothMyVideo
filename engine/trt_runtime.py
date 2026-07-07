"""
TensorRT backend for the GMFSS sub networks.

Strategy: each sub net is exported
to ONNX under autocast(fp16) via the dynamo exporter (mixed fp16/fp32 matching the
app's precision), then built into a strongly typed TRT engine. softsplat (cupy) and
the F.interpolate glue stay in eager. Engines are built on first use for a given
input resolution and cached on disk per (net, shapes, gpu, trt version, weights hash);
the weights fingerprint in the name means a train_log swap invalidates the cache
automatically (stale engines are deleted at startup) instead of silently serving
engines compiled from the old model.

trtify(model) swaps model.feat_ext / flownet / metricnet / ifnet / fusionnet for
wrappers with identical call signatures, so GMFSS_infer_u is untouched. Any export
or build failure falls back to the original eager module, so the app never breaks.
"""
import hashlib
import logging
import os
import sys
import time
import warnings

import torch
import torch.nn as nn

# torch 2.12's dynamo exporter unpickles pytree TreeSpecs internally and trips torch's OWN
# LeafSpec deprecation shim - a FutureWarning surfacing through copyreg once per one-time
# engine build. Torch-internal, nothing this code calls; silence it so builds don't spam the
# GUI log (same benign family as the documented torch.cuda.amp FutureWarnings).
warnings.filterwarnings("ignore", message=r".*LeafSpec.*", category=FutureWarning)
# The exporter also lazily imports torch.utils.flop_counter, whose import-time "triton not
# found" logger warning is meaningless here (Triton is deliberately not installed on Windows,
# see the README's torch.compile note). Raise that logger's threshold before the lazy import
# fires; this module is imported ahead of every build, so it always lands in time.
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)

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


def _weights_tag():
    """Fingerprint of the train_log weights, baked into every engine filename so a compiled
    engine can never outlive the weights it was exported from: swapped .pkl files change the
    tag, which is a cache miss (fresh build) plus garbage collection of the stale engines
    below. Hashes the file CONTENTS, not mtimes - a fresh unzip/copy of identical weights
    must not throw away ~6 min of builds per resolution. The ~75 MB read costs ~0.2 s once
    per engine start, and the same files are about to be loaded by torch anyway."""
    h = hashlib.md5()
    wdir = os.path.join(HERE, "GMFSS_Fortuna", "train_log")
    for n in sorted(os.listdir(wdir)):
        if n.endswith(".pkl"):
            with open(os.path.join(wdir, n), "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
    return "w" + h.hexdigest()[:10]




def _shape_tag(tensors):
    return "_".join("x".join(map(str, t.shape)) for t in tensors)


def _log(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


WEIGHTS_TAG = _weights_tag()

# Reconcile the cache with the current weights, once per engine start:
#   - engines named for a DIFFERENT weights fingerprint are stale by definition: delete them
#     (this is the "force delete the ancient engines" step - without it a weight swap would
#     silently keep serving the old model, because build_or_load finds engines by name and
#     never consults the .pkl files again);
#   - engines from before fingerprinting (no _w suffix) are migrated by RENAME to the current
#     tag instead of deleted: every historical build came from the only weights this app has
#     ever shipped (hash-verified against upstream), so they are known good and rebuilding
#     them would cost ~6 min per resolution for nothing;
#   - stray .onnx/.onnx.data intermediates (a crashed build) are junk either way.
for _fn in os.listdir(CACHE_DIR):
    _p = os.path.join(CACHE_DIR, _fn)
    try:
        if _fn.endswith((".onnx", ".onnx.data")):
            os.remove(_p)
        elif _fn.endswith(".engine") and f"_{WEIGHTS_TAG}" not in _fn:
            if "_w" not in _fn:  # pre-fingerprint name: adopt it for the current weights
                os.rename(_p, _p[:-len(".engine")] + f"_{WEIGHTS_TAG}.engine")
                _log(f"[trt] adopted cached engine for current weights: {_fn}")
            else:                # fingerprinted for other weights: stale, remove
                os.remove(_p)
                _log(f"[trt] removed stale engine (weights changed): {_fn}")
    except OSError:
        pass  # cache hygiene is best effort; a locked file just stays until next start


class TRTModule:
    """A built engine; binds torch cuda tensors zero copy and runs it."""

    def __init__(self, serialized):
        self.runtime = trt.Runtime(TRT_LOGGER)
        self.engine = self.runtime.deserialize_cuda_engine(serialized)
        self.context = self.engine.create_execution_context()
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
        # Enqueue on the caller's current stream and do NOT host-sync. The whole pipeline (this engine,
        # softsplat's cupy kernel, and the torch glue) runs on one shared stream the caller sets, so
        # same-stream ordering makes the outputs ready for the next op with no per-call GPU drain. The
        # caller using a non-default stream is also what keeps TensorRT's default-stream warning away.
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
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
    tag = f"{name}_{_shape_tag(example_inputs)}_{_gpu_tag()}_{WEIGHTS_TAG}"
    engine_path = os.path.join(CACHE_DIR, tag + ".engine")
    if os.path.isfile(engine_path):
        with open(engine_path, "rb") as f:
            return TRTModule(f.read())
    onnx_path = os.path.join(CACHE_DIR, tag + ".onnx")
    _log(f"[trt] building {name} {_shape_tag(example_inputs)} (one time for this resolution)...")
    t0 = time.time()
    # Freshly constructed wrapper modules (e.g. _IFNetExport) default to training=True even
    # when every weight inside is already eval, and the exporter checks (and warns on) the TOP
    # module's flag. An export here is always for inference, so force eval unconditionally.
    export_module.eval()
    with torch.autocast("cuda", dtype=torch.float16):
        # verbose=False drops the exporter's per-phase progress chatter (each phase printed
        # twice: a start line, then the same line again with a checkmark on completion) - the
        # "[trt] building..." line above is the user-facing signal for the one-time build.
        torch.onnx.export(export_module, tuple(example_inputs), onnx_path,
                          input_names=input_names, output_names=output_names,
                          dynamo=True, opset_version=18, verbose=False)
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


class RestoreEngine(_Engine):
    """TRT wrapper for the --restore Real-ESRGAN pass (realesr.py). The realesr weights hash
    rides in the NAME (not the _w tag): a realesr weight swap changes the name and is a plain
    cache miss (the old file lingers harmlessly, a few MB), while the global _w tag keeps the
    startup GC from deleting these engines - it only ties them to the train_log fingerprint,
    so a GMFSS weight swap also rebuilds them (seconds; the net is tiny)."""

    def __init__(self, eager, whash):
        super().__init__(f"restore_{whash}", eager, ["x"], ["y"])

    def __call__(self, x):
        return self._run(self.eager, (x,), (x,))


def trtify(model):
    """Swap a loaded GMFSS Model's sub nets for TRT wrappers (in place)."""
    model.feat_ext = FeatEngine(model.feat_ext)
    model.flownet = FlowEngine(model.flownet)
    model.metricnet = MetricEngine(model.metricnet)
    model.ifnet = IFNetEngine(model.ifnet)
    model.fusionnet = FusionEngine(model.fusionnet)
    _log("[trt] model wrapped with TensorRT engines (build on first frame)")
    return model
