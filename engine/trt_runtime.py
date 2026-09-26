"""
The size-free ONNX export of every graph the native host builds TensorRT-RTX engines from (a dev
tool, run by engine/onnx_export.py; never shipped).

Each graph is exported under autocast(fp16) via the dynamo exporter (mixed fp16/fp32 matching the
app's precision) with H and W symbolic, once for every size; the host builds one engine per
resolution from that file (H and W pinned, only the live RIFE timestep batch axis dynamic). The
engine classes below carry only what the export needs from each graph: the engine base name, the
input / output names and the dynamic batch range. The engines are built and run by the native
host (smv-live.exe); nothing here builds or runs them.
"""
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

# Cache naming (weights fingerprints, the ONNX paths) lives in the torch-free trt_lookup module.
import trt_lookup


def _log(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def _rm(*paths):
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


def _export_onnx(export_module, example_inputs, input_names, output_names, onnx_path,
                 dynamic_shapes=None):
    # Freshly constructed wrapper modules (e.g. _IFNetExport) default to training=True even
    # when every weight inside is already eval, and the exporter checks (and warns on) the TOP
    # module's flag. An export here is always for inference, so force eval unconditionally.
    export_module.eval()
    # cudnn OFF for the trace: after a failed dynamic export earlier in the process (e.g.
    # gmflow_bidir on this stack), grid_sampler decomposition starts picking
    # aten.cudnn_grid_sampler, which has no fake-tensor support, and every later export of a
    # grid_sample-using net (metricnet's backwarp) dies with ConversionError/PassError.
    # Disabling cudnn for the export forces the exportable grid_sampler_2d choice; the built
    # engine and the eager nets are untouched.
    with torch.backends.cudnn.flags(enabled=False), \
            torch.autocast("cuda", dtype=torch.float16):
        # verbose=False drops the exporter's per-phase progress chatter (each phase printed
        # twice: a start line, then the same line again with a checkmark on completion) - the
        # "[trt] exporting..." line is the user-facing signal for the one-time export.
        torch.onnx.export(export_module, tuple(example_inputs), onnx_path,
                          input_names=input_names, output_names=output_names,
                          dynamo=True, opset_version=18, verbose=False,
                          dynamic_shapes=dynamic_shapes)


def _fuse_prelu(onnx_path):
    """Rewrite every PRelu into an exact form TensorRT-RTX fuses into the conv before it: it runs
    PRelu as a separate elementwise kernel (22 % of fusionnet, 26 % of Restore) but fuses
    LeakyRelu and Max. Scalar slope a: LeakyRelu(alpha=a); per-channel slopes all <= 1:
    Max(x, x * a); any per-channel slope: Max(x, 0) + a * Min(x, 0). Each is exact for its case,
    and the engines built from the rewrite give bit-identical output (Restore 1.30x, fusionnet
    1.06x / 1.09x faster at 1080p / 4K).
    Only the graph is rewritten; the external weight file stays as it is."""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    g = onnx.load(onnx_path, load_external_data=False)
    if not any(n.op_type == "PRelu" for n in g.graph.node):
        return
    vals = {i.name: numpy_helper.to_array(i) for i in onnx.load(onnx_path).graph.initializer}
    nodes = []
    for nd in g.graph.node:
        if nd.op_type != "PRelu":
            nodes.append(nd)
            continue
        x, s, y = nd.input[0], nd.input[1], nd.output[0]
        a = vals[s]
        if a.size == 1:
            nodes.append(helper.make_node("LeakyRelu", [x], [y], alpha=float(a.reshape(-1)[0]),
                                          name=nd.name + "_lrelu"))
        elif float(a.max()) <= 1.0:
            nodes += [helper.make_node("Mul", [x, s], [y + "_ax"], name=nd.name + "_mul"),
                      helper.make_node("Max", [x, y + "_ax"], [y], name=nd.name + "_max")]
        else:
            z = nd.name + "_zero"
            g.graph.initializer.append(numpy_helper.from_array(np.zeros((1,), a.dtype), z))
            nodes += [helper.make_node("Max", [x, z], [y + "_p"], name=nd.name + "_pos"),
                      helper.make_node("Min", [x, z], [y + "_n"], name=nd.name + "_neg"),
                      helper.make_node("Mul", [s, y + "_n"], [y + "_an"], name=nd.name + "_mul"),
                      helper.make_node("Add", [y + "_p", y + "_an"], [y], name=nd.name + "_add")]
    used = {i for n in nodes for i in n.input}
    keep = [i for i in g.graph.initializer if i.name in used]
    del g.graph.initializer[:]
    g.graph.initializer.extend(keep)
    del g.graph.node[:]
    g.graph.node.extend(nodes)
    with open(onnx_path + ".tmp", "wb") as fh:
        fh.write(g.SerializeToString())
    os.replace(onnx_path + ".tmp", onnx_path)
    onnx.checker.check_model(onnx_path)   # by path: the external data resolves beside the file


def _half_features(onnx_path, name):
    """RIFE IFNet / block0: take the feature encodes f0 / f1 as fp16 inputs and widen them inside
    the graph. The encode engine outputs fp16, so the host used to widen it to fp32 (k_h2f) only
    for the engine to read it back; fp16 -> fp32 is exact, so the graph sees the same values and
    the engines give bit-identical output (the live batched IFNet at 1472x2560 1.020x faster, and
    the host's widen pass is gone). Graph only."""
    import onnx
    from onnx import helper, TensorProto

    if not name.startswith(("rife_ifnet_", "rife_block0_")):
        return
    g = onnx.load(onnx_path, load_external_data=False)
    feats = [i for i in g.graph.input if i.name in ("f0", "f1")]
    if len(feats) != 2 or any(i.type.tensor_type.elem_type != TensorProto.FLOAT for i in feats):
        raise RuntimeError(f"{name}: expected fp32 inputs f0 / f1")
    for inp in feats:
        wide = inp.name + "_f32"
        for nd in g.graph.node:
            for k, x in enumerate(nd.input):
                if x == inp.name:
                    nd.input[k] = wide
        g.graph.node.insert(0, helper.make_node("Cast", [inp.name], [wide], to=TensorProto.FLOAT,
                                                name=inp.name + "_widen"))
        inp.type.tensor_type.elem_type = TensorProto.FLOAT16
    with open(onnx_path + ".tmp", "wb") as fh:
        fh.write(g.SerializeToString())
    os.replace(onnx_path + ".tmp", onnx_path)
    onnx.checker.check_model(onnx_path)


def _half_frames(onnx_path, name):
    """RIFE IFNet x / the encode's img: take the frames as fp16 inputs. The IFNet widens x inside
    the graph (its warps still run fp32, on the fp16-rounded pixels); the encode's first node
    casts img to fp16 anyway, so that Cast goes and its features stay identical. The host fills an
    fp16 copy of the frame pair once per frame and the IFNet reads half the bytes (1.05x per call
    at 1080p). Not bit-identical: the rounding reaches the flow (tweens 56 to 65 dB against fp32
    frames on anime, every real frame unchanged). Graph only."""
    import onnx
    from onnx import helper, TensorProto

    if name.startswith("rife_ifnet_"):
        key = "x"
    elif name.startswith("rife_encode_"):
        key = "img"
    else:
        return
    g = onnx.load(onnx_path, load_external_data=False)
    inp = [i for i in g.graph.input if i.name == key]
    if len(inp) != 1 or inp[0].type.tensor_type.elem_type != TensorProto.FLOAT:
        raise RuntimeError(f"{name}: expected an fp32 input {key}")
    inp = inp[0]
    readers = [nd for nd in g.graph.node if key in nd.input]
    narrow = (len(readers) == 1 and readers[0].op_type == "Cast"
              and [a.i for a in readers[0].attribute if a.name == "to"] == [TensorProto.FLOAT16])
    if narrow:
        old = readers[0].output[0]
        g.graph.node.remove(readers[0])
        for nd in g.graph.node:
            for k, x in enumerate(nd.input):
                if x == old:
                    nd.input[k] = key
    else:
        wide = key + "_f32"
        for nd in readers:
            for k, x in enumerate(nd.input):
                if x == key:
                    nd.input[k] = wide
        g.graph.node.insert(0, helper.make_node("Cast", [key], [wide], to=TensorProto.FLOAT,
                                                name=key + "_widen"))
    inp.type.tensor_type.elem_type = TensorProto.FLOAT16
    with open(onnx_path + ".tmp", "wb") as fh:
        fh.write(g.SerializeToString())
    os.replace(onnx_path + ".tmp", onnx_path)
    onnx.checker.check_model(onnx_path)


# --- size-free ONNX -----------------------------------------------------------------------------
# The graphs below export ONCE with H / W symbolic (trt_lookup.onnx_path), and every engine size is
# built from that file, pinned to the example shape exactly like a per-size export (one engine per
# resolution still holds, the profile pins every axis but the declared batch range). Built from
# these files, 8 graphs are bit-exact with the per-size
# engines; gmflow_bidir runs its size-free branch (multiply + sum local ops) and is equivalent or
# better (closer to eager fp32). Unit = the alignment of the FIRST input's H and W (the /64 frame
# pad, 32 for GMFSS's half-size nets, 1 for Restore at the model size); every other input's
# spatial dims are an integer multiple of the same symbols. Any other name or an unaligned or
# oversized example does not qualify (ensure_onnx returns None).
_SIZE_FREE_UNIT = {"featurenet": 64, "gmflow_bidir": 32, "metricnet": 32, "ifnet": 32,
                   "fusionnet": 32}
_SIZE_FREE_PREFIX = (("rife_ifnet_", 64), ("rife_encode_", 64), ("rife_block0_", 64),
                     ("restore_", 1))
_SIZE_FREE_MAX = (4352, 7680)   # H, W: an 8K frame padded to /64


def _size_free_unit(name):
    for pre, unit in _SIZE_FREE_PREFIX:
        if name.startswith(pre):
            return unit
    return _SIZE_FREE_UNIT.get(name)


def _export_key(name, export_module, example_inputs):
    """The ONNX key: the engine base name, plus what that name leaves out but the graph bakes in:
    the GMFSS IFNet's scale_list (its engine is named plain `ifnet`) and the input dtypes (a
    strongly typed engine takes its input types from the ONNX; `_dt` + one letter per input
    only when an input is not fp32, e.g. Restore's fp16 `_dth`)."""
    key = name
    if isinstance(export_module, _IFNetExport):
        key += "_sl" + "-".join(f"{s:g}" for s in export_module.scale_list)
    dts = "".join({torch.float32: "f", torch.float16: "h"}.get(t.dtype, "x") for t in example_inputs)
    if dts.strip("f"):
        key += "_dt" + dts
    return key


def _size_free_shapes(name, example_inputs, input_names, dyn_batch):
    """torch.export dynamic_shapes for the size-free export, or None when this graph / example
    does not qualify."""
    unit = _size_free_unit(name)
    if unit is None:
        return None
    h0, w0 = int(example_inputs[0].shape[2]), int(example_inputs[0].shape[3])
    lo = 16 if unit == 1 else 1
    mh, mw = _SIZE_FREE_MAX[0] // unit, _SIZE_FREE_MAX[1] // unit
    if h0 % unit or w0 % unit or not (lo <= h0 // unit <= mh and lo <= w0 // unit <= mw):
        return None
    D = torch.export.Dim
    sh, sw = D("sh", min=lo, max=mh), D("sw", min=lo, max=mw)
    b = None
    ds = []
    for n, t in zip(input_names, example_inputs):
        d = {}
        if t.dim() != 4:
            return None
        hi, wi = int(t.shape[2]), int(t.shape[3])
        if (hi, wi) != (1, 1):
            if (hi * unit) % h0 or (wi * unit) % w0:
                return None
            d[2], d[3] = (hi * unit // h0) * sh, (wi * unit // w0) * sw
        if dyn_batch and n in dyn_batch:
            if b is None:
                b = D("b", min=dyn_batch[n][0], max=dyn_batch[n][2])
            d[0] = b
        ds.append(d)
    return tuple(ds)


def _size_free_onnx(key, name, export_module, example_inputs, input_names, output_names,
                    dyn_batch):
    """Path of this graph's size-free ONNX, exported now when missing; None when the graph or
    example does not qualify. The export lands in a per-process temp folder and is moved in only
    once complete (weights first), so a concurrent process never builds from half a file."""
    # qualify the example FIRST, also when the file exists: TensorRT reads an axis named
    # `64*sh` as plain -1 and builds any size from it, so only this check keeps an unaligned
    # example (the export's divisibility assumption broken) out
    ds = _size_free_shapes(name, example_inputs, input_names, dyn_batch)
    if ds is None:
        return None
    path = trt_lookup.onnx_path(key)
    if os.path.isfile(path):
        return path
    tmpdir = os.path.join(trt_lookup.ONNX_DIR, f"tmp_{os.getpid()}")
    os.makedirs(tmpdir, exist_ok=True)
    tmp = os.path.join(tmpdir, os.path.basename(path))
    try:
        _log(f"[trt] exporting the size-free {name} graph (one time for every size)...")
        t0 = time.time()
        _export_onnx(export_module, example_inputs, input_names, output_names, tmp,
                     dynamic_shapes=ds)
        import onnx

        got = {i.name: [d.dim_param if d.HasField("dim_param") else d.dim_value
                        for d in i.type.tensor_type.shape.dim]
               for i in onnx.load(tmp, load_external_data=False).graph.input}
        for n, d in zip(input_names, ds):
            dims = got.get(n)
            if dims is None:
                raise RuntimeError(f"exporter renamed input {n} (graph has {sorted(got)})")
            if any(not isinstance(dims[a], str) or not dims[a] for a in d):
                raise RuntimeError(f"exporter specialized an axis of {n}: {dims}")
        _fuse_prelu(tmp)
        _half_features(tmp, name)
        _half_frames(tmp, name)
        if os.path.isfile(tmp + ".data"):
            os.replace(tmp + ".data", path + ".data")
        os.replace(tmp, path)
        _log(f"[trt] size-free {name} graph exported in {time.time() - t0:.0f}s")
        return path
    finally:
        _rm(tmp, tmp + ".data")
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass


def ensure_onnx(name, export_module, example_inputs, input_names, output_names, dyn_batch=None):
    """Export this graph's size-free ONNX if it qualifies and is missing (scripts/export-onnx.js
    through engine/onnx_export.py); returns the path or None. Raises on a failed export."""
    return _size_free_onnx(_export_key(name, export_module, example_inputs), name, export_module,
                           example_inputs, input_names, output_names, dyn_batch)


class _Engine:
    """One graph's engine contract: the base name the host's engine and ONNX names start from,
    the input / output tensor names, and the dynamic batch range ({input name: (min, opt, max)}
    for the live RIFE class, None for every other graph)."""

    def __init__(self, name, input_names, output_names):
        self.name = name
        self.input_names = input_names
        self.output_names = output_names
        self.dyn_batch = None


class FeatEngine(_Engine):
    def __init__(self):
        super().__init__("featurenet", ["x"], ["f1", "f2", "f3"])


class _BidirFlowExport(nn.Module):
    """gmflow with pred_bidir_flow=True baked: one call returns both flow directions
    stacked on the batch dim ([2,2,H,W]); the backbone runs once for both."""

    def __init__(self, flownet):
        super().__init__()
        self.flownet = flownet

    def forward(self, img0, img1):
        return self.flownet(img0, img1, pred_bidir_flow=True)


class BidirFlowEngine(_Engine):
    def __init__(self):
        super().__init__("gmflow_bidir", ["img0", "img1"], ["flow"])


class MetricEngine(_Engine):
    def __init__(self):
        super().__init__("metricnet", ["i0", "i1", "f01", "f10"], ["m0", "m1"])


class FusionEngine(_Engine):
    def __init__(self):
        super().__init__("fusionnet", ["a", "b", "c", "d"], ["out"])


class _IFNetExport(nn.Module):
    """IFNet with scale_list baked and timestep as a (1,1,1,1) tensor input."""

    def __init__(self, ifnet, scale_list):
        super().__init__()
        self.ifnet = ifnet
        self.scale_list = scale_list

    def forward(self, x, timestep):
        return self.ifnet(x, timestep, scale_list=self.scale_list)


class IFNetEngine(_Engine):
    def __init__(self):
        super().__init__("ifnet", ["x", "timestep"], ["merged"])


class RestoreEngine(_Engine):
    """The --restore Real-ESRGAN pass (realesr.py). The realesr weights hash rides in the NAME
    (not the _w tag): a realesr weight swap changes the name and is a plain cache miss."""

    def __init__(self, whash):
        super().__init__(f"restore_{whash}", ["x"], ["y"])


# --- RIFE 4.26 (rife_backend.py) ----------------------------------------------------------------
# The RIFE IFNet_HDv3 is a different network from GMFSS's IFNet above, so it gets its own graphs:
# the full IFNet forward (five IFBlocks + the interleaved grid_sample warps), the feature Head
# (encode) and DRBA's block0 flow.


class _RifeIFNetExport(nn.Module):
    """RIFE IFNet full forward with scale_list baked; timestep and the two feature encodes (f0,f1)
    are tensor inputs and the output is just the fused frame merged[4] (flow_list is dropped)."""

    def __init__(self, ifnet, scale_list):
        super().__init__()
        self.ifnet = ifnet
        self.scale_list = scale_list

    def forward(self, x, timestep, f0, f1):
        return self.ifnet(x, timestep=timestep, scale_list=self.scale_list, f0=f0, f1=f1)[0]


def _rife_ifnet_base(ifnet, scale_list, whash):
    """The rife checkpoint fingerprint plus the baked scale_list are in the NAME (4K's 0.5 flow
    scale changes it)."""
    scale_tag = "-".join(f"{s:g}" for s in scale_list)
    return f"rife_ifnet_{whash}_{scale_tag}"


class RifeIFNetEngine(_Engine):
    """One RIFE IFNet forward (the unbatched class). timestep rides as a (1,1,H,W) tensor so both
    the scalar plain-RIFE path and DRBA's spatial DistanceRatioMap run the same engine."""

    def __init__(self, ifnet, scale_list, whash):
        super().__init__(_rife_ifnet_base(ifnet, scale_list, whash),
                         ["x", "timestep", "f0", "f1"], ["merged"])


class _RifeIFNetBatchExport(nn.Module):
    """Same forward as _RifeIFNetExport, but up to B timesteps in ONE call: the timestep input
    carries the batch ((B,1,H,W), a dynamic axis) and the pair-constant inputs (x, f0, f1) stay
    batch-1 and are EXPANDED inside the graph, so the caller uploads one pair's data no matter
    how many tweens it asks for."""

    def __init__(self, ifnet, scale_list):
        super().__init__()
        self.ifnet = ifnet
        self.scale_list = scale_list

    def forward(self, x, timestep, f0, f1):
        b = timestep.shape[0]
        if getattr(self.ifnet, "batch_broadcast", False):
            # the net expands the SMALL tensors itself, after the downsample, so the
            # pair constant full resolution downsample runs once instead of once per tween.
            return self.ifnet(x, timestep=timestep, scale_list=self.scale_list,
                              f0=f0, f1=f1)[0]
        return self.ifnet(x.expand(b, -1, -1, -1), timestep=timestep,
                          scale_list=self.scale_list,
                          f0=f0.expand(b, -1, -1, -1),
                          f1=f1.expand(b, -1, -1, -1))[0]


class RifeIFNetBatchEngine(_Engine):
    """1 to b tweens of one pair per enqueue (the live `_bd{b}` class; the host builds the offline
    fixed-batch `_b{B}` engines from the same graph). `_bd{b}` in the name keeps it apart from the
    unbatched class."""

    def __init__(self, ifnet, scale_list, whash, b):
        super().__init__(_rife_ifnet_base(ifnet, scale_list, whash) + f"_bd{b}",
                         ["x", "timestep", "f0", "f1"], ["merged"])
        # (min, opt, max) on the batch axis. opt = b: the full batch is the throughput case
        # worth tuning for.
        self.dyn_batch = {"timestep": (1, b, b)}


class _RifeEncodeExport(nn.Module):
    """The IFNet feature Head as its own graph: one padded frame in, its 16-channel encode out."""

    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, img):
        return self.head(img)


class RifeEncodeEngine(_Engine):
    """The RIFE IFNet Head (f0/f1). The name carries the rife checkpoint fingerprint exactly like
    the IFNet engines."""

    def __init__(self, whash):
        super().__init__(f"rife_encode_{whash}", ["img"], ["feat"])


class _RifeBlock0Export(nn.Module):
    """DRBA's coarsest-level flow as its own graph: exactly
    rife_backend.RIFE.calc_flow's block0 call, timestep 0.5 and scale_list[0] baked, the
    4-channel flow (0->0.5 | 1->0.5, full size) out. The avg splats after it are the host's
    kernels, so the graph stops at the flow."""

    def __init__(self, block0, scale):
        super().__init__()
        self.block0 = block0
        self.scale = scale

    def forward(self, img0, img1, f0, f1):
        ts = img0[:, :1] * 0 + 0.5
        flow, _, _ = self.block0(torch.cat((img0, img1, f0, f1, ts), 1), None, scale=self.scale)
        return flow.float()   # the native host's contract: fp32 in, fp32 out


class RifeBlock0Engine(_Engine):
    """DRBA's block0 flow (calc_flow). The name carries the rife checkpoint fingerprint and the
    baked scale (trt_lookup.rife_block0_base is the one naming rule)."""

    def __init__(self, scale, whash):
        super().__init__(trt_lookup.rife_block0_base(scale, whash),
                         ["img0", "img1", "f0", "f1"], ["flow"])
