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
import io
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
    for _stream in (sys.stdout, sys.stderr):
        if isinstance(_stream, io.TextIOWrapper):
            _stream.reconfigure(encoding="utf-8", errors="replace")
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


def _half_flow_mask(onnx_path, name):
    """RIFE IFNet: hand the final flow and the blend mask out as fp16. An output the graph already
    makes in fp16 stays as it is; an fp32 one gets one Cast at the end, and any reader inside the
    graph keeps the fp32 tensor. Graph only."""
    import onnx
    from onnx import helper, TensorProto

    if not name.startswith("rife_ifnet_"):
        return
    g = onnx.load(onnx_path, load_external_data=False)
    for key in ("flow", "mask"):
        out = [o for o in g.graph.output if o.name == key]
        prod = [nd for nd in g.graph.node if key in nd.output]
        if len(out) != 1 or len(prod) != 1:
            raise RuntimeError(f"{name}: expected one output {key}")
        et = out[0].type.tensor_type.elem_type
        if et == TensorProto.FLOAT16:
            continue
        if et != TensorProto.FLOAT:
            raise RuntimeError(f"{name}: output {key} is neither fp32 nor fp16")
        prod[0].output[list(prod[0].output).index(key)] = key + "_f32"
        for nd in g.graph.node:
            for k, x in enumerate(nd.input):
                if x == key:
                    nd.input[k] = key + "_f32"
        g.graph.node.append(helper.make_node("Cast", [key + "_f32"], [key], to=TensorProto.FLOAT16,
                                             name=key + "_narrow"))
        out[0].type.tensor_type.elem_type = TensorProto.FLOAT16
    with open(onnx_path + ".tmp", "wb") as fh:
        fh.write(g.SerializeToString())
    os.replace(onnx_path + ".tmp", onnx_path)
    onnx.checker.check_model(onnx_path)


def _half_timestep(onnx_path, name):
    """RIFE IFNet: take the timestep as an fp16 input and widen it inside the graph. Every timestep
    path resizes it in fp32 and narrows it to fp16 before a conv, so a constant t (RIFE, Frame Blend)
    gives the same tweens bit for bit; DRBA's spatial map is rounded to fp16 before those resizes
    (its tweens move by up to 31 codes at 8 bits on thin moving edges, 55.7 dB at the worst frame).
    The host fills half the bytes and the timestep buffer halves. Graph only."""
    import onnx
    from onnx import helper, TensorProto

    if not name.startswith("rife_ifnet_"):
        return
    g = onnx.load(onnx_path, load_external_data=False)
    inp = [i for i in g.graph.input if i.name == "timestep"]
    if len(inp) != 1 or inp[0].type.tensor_type.elem_type != TensorProto.FLOAT:
        raise RuntimeError(f"{name}: expected an fp32 input timestep")
    for nd in g.graph.node:
        for k, x in enumerate(nd.input):
            if x == "timestep":
                nd.input[k] = "timestep_f32"
    g.graph.node.insert(0, helper.make_node("Cast", ["timestep"], ["timestep_f32"], to=TensorProto.FLOAT,
                                            name="timestep_widen"))
    inp[0].type.tensor_type.elem_type = TensorProto.FLOAT16
    with open(onnx_path + ".tmp", "wb") as fh:
        fh.write(g.SerializeToString())
    os.replace(onnx_path + ".tmp", onnx_path)
    onnx.checker.check_model(onnx_path)


def _half_blockin(onnx_path, name):
    """RIFE IFNet: build each block's input in fp16. A block concatenates its four warps (GridSample
    stays fp32: TensorRT-RTX takes one type for its input and grid) with fp16 tensors widened for
    the concat, resizes that to the block's scale and narrows it to fp16 for the convs. Here every
    warp output is narrowed at once, the fp16 tensors join as they are, the concat and resize run in
    fp16 and the final narrowing goes. Those fp32 stages were 27 % of the IFNet's time at 1080p: it
    runs 1.12x to 1.13x faster per call at 1080p and 4K. Not bit-identical: the tweens move by up to
    29 codes at 8 bits on thin fast-moving edges (PSNR 58.2 dB at the worst of 96 tweens at 480p,
    1080p and 4K, median 74 to 82 dB); the real frames do not. Graph only."""
    import onnx
    from onnx import helper, TensorProto

    if not name.startswith("rife_ifnet_"):
        return
    g = onnx.load(onnx_path, load_external_data=False)
    typed = onnx.shape_inference.infer_shapes(g).graph
    dtype = {v.name: v.type.tensor_type.elem_type
             for v in list(typed.value_info) + list(typed.input) + list(typed.output)}
    prod = {o: nd for nd in g.graph.node for o in nd.output}
    cons = {}
    for nd in g.graph.node:
        for x in nd.input:
            cons.setdefault(x, []).append(nd)

    def only(t, op):
        # the one reader of op type `op`; Shape readers (the batched graph sizes its Resize from them)
        # are allowed beside it and later read the fp16 tensor, its shape is the same
        c = [nd for nd in cons.get(t, []) if nd.op_type != "Shape"]
        if len(c) != 1 or c[0].op_type != op:
            raise RuntimeError(f"{name}: expected one {op} reading {t}")
        return c[0]

    blocks = []
    for c1 in g.graph.node:
        if c1.op_type != "Concat" or not any(x in prod and prod[x].op_type == "GridSample" for x in c1.input):
            continue
        rs = only(c1.output[0], "Resize")
        c2 = only(rs.output[0], "Concat")
        fc = only(c2.output[0], "Cast")
        if [a.i for a in fc.attribute if a.name == "to"] != [TensorProto.FLOAT16]:
            raise RuntimeError(f"{name}: block input {c2.name} is not narrowed to fp16")
        blocks.append((c1, rs, c2, fc))
    if not blocks:
        raise RuntimeError(f"{name}: no warped block input found")
    # every new node takes the place of the node it replaces, so readers in between stay in order
    at = {}
    for k, (c1, rs, c2, fc) in enumerate(blocks, 1):
        at[c1.name], at[rs.name], at[c2.name], at[fc.name] = (k, "c1"), (k, "rs"), (k, "c2"), (k, "fc")
    new = {}   # old fp32 tensor -> its fp16 replacement, for the Shape readers
    out = []
    for nd in g.graph.node:
        if nd.name not in at:
            out.append(nd)
            continue
        k, role = at[nd.name]
        c1, rs, c2, fc = blocks[k - 1]

        def to_half(t):
            p = prod.get(t)
            if p is not None and p.op_type == "Cast" and dtype.get(p.input[0]) == TensorProto.FLOAT16:
                return p.input[0]
            h = f"{t}_h{k}"
            out.append(helper.make_node("Cast", [t], [h], name=f"{t}_tohalf{k}", to=TensorProto.FLOAT16))
            return h

        if role == "c1":
            c1h = helper.make_node("Concat", [to_half(x) for x in c1.input], [f"{c1.output[0]}_h"],
                                   name=f"{c1.name}_h")
            c1h.attribute.extend(c1.attribute)
            out.append(c1h)
            new[c1.output[0]] = c1h.output[0]
        elif role == "rs":
            rsh = helper.make_node("Resize", [f"{c1.output[0]}_h"] + list(rs.input[1:]), [f"{rs.output[0]}_h"],
                                   name=f"{rs.name}_h")
            rsh.attribute.extend(rs.attribute)
            out.append(rsh)
            new[rs.output[0]] = rsh.output[0]
        elif role == "c2":
            c2h = helper.make_node("Concat", [f"{rs.output[0]}_h" if x == rs.output[0] else to_half(x)
                                              for x in c2.input], [fc.output[0]], name=f"{c2.name}_h")
            c2h.attribute.extend(c2.attribute)
            out.append(c2h)
            new[c2.output[0]] = fc.output[0]
    for nd in out:
        if nd.op_type == "Shape" and nd.input[0] in new:
            nd.input[0] = new[nd.input[0]]
    keep = {o.name for o in g.graph.output}
    while True:   # drop the fp32 chain and the widening casts nothing reads any more
        used = {x for nd in out for x in nd.input} | keep
        alive = [nd for nd in out if any(o in used for o in nd.output)]
        if len(alive) == len(out):
            break
        out = alive
    made = {o for nd in out for o in nd.output}
    vis = [v for v in g.graph.value_info if v.name in made]
    del g.graph.node[:]
    g.graph.node.extend(out)
    del g.graph.value_info[:]
    g.graph.value_info.extend(vis)
    with open(onnx_path + ".tmp", "wb") as fh:
        fh.write(g.SerializeToString())
    os.replace(onnx_path + ".tmp", onnx_path)
    onnx.checker.check_model(onnx_path)


def _b0_slice_first(onnx_path, name):
    """RIFE block0 (DRBA's flow): slice the 4 flow channels out of the last layer's 13 BEFORE the
    resize to the full size instead of after it. A bilinear resize works per channel, so the flow is
    the same bit for bit; the 9 mask and feature channels block0 drops are no longer resized. block0
    runs 1.33x to 1.36x faster per call at 1080p and 4K (unchanged at 480p). Graph only."""
    import onnx
    from onnx import helper, numpy_helper, TensorProto

    if not name.startswith("rife_block0_"):
        return
    g = onnx.load(onnx_path, load_external_data=False)
    prod = {o: nd for nd in g.graph.node for o in nd.output}
    readers = {}
    for nd in g.graph.node:
        for x in nd.input:
            readers[x] = readers.get(x, 0) + 1
    small = {i.name: i for i in g.graph.initializer if i.data_location != TensorProto.EXTERNAL}

    def const(t):   # a small constant (the weights stay in the .data file)
        return numpy_helper.to_array(small[t]) if t in small else None

    cast = prod.get(g.graph.output[0].name)
    mul = prod.get(cast.input[0]) if cast is not None else None
    sl = prod.get(mul.input[0]) if mul is not None else None
    rs = prod.get(sl.input[0]) if sl is not None else None
    chain = [nd.op_type if nd is not None else None for nd in (cast, mul, sl, rs)]
    if (cast is None or mul is None or sl is None or rs is None or chain != ["Cast", "Mul", "Slice", "Resize"]
            or readers[rs.output[0]] != 1 or len(rs.input) != 4 or not rs.name or not sl.name):
        raise RuntimeError(f"{name}: expected Resize -> Slice -> Mul -> Cast at the flow output, got {chain}")
    sizes = prod.get(rs.input[3])
    if sizes is None or sizes.op_type != "Concat":
        raise RuntimeError(f"{name}: expected the resize sized by a Concat")
    lead = const(sizes.input[0])
    start, end, axes = (const(x) for x in sl.input[1:4])
    if lead is None or start is None or end is None or axes is None or list(axes) != [1] or list(lead) != [1, 13]:
        raise RuntimeError(f"{name}: expected the resize sized [1, 13, H, W] and a channel slice")
    g.graph.initializer.append(helper.make_tensor("flow_sizes_lead", TensorProto.INT64, [2],
                                                  [1, int(end[0]) - int(start[0])]))
    fsizes = helper.make_node("Concat", ["flow_sizes_lead", sizes.input[1]], ["flow_sizes"], name="flow_sizes",
                              axis=0)
    moved = {rs.output[0], sl.output[0]}   # both tensors change shape
    sl.input[0] = rs.input[0]
    rs.input[0] = sl.output[0]
    rs.input[3] = "flow_sizes"
    mul.input[0] = rs.output[0]
    # the Slice takes the Resize's place in the order, the new sizes and the Resize the Slice's
    out = []
    for nd in g.graph.node:
        if nd.name == rs.name:
            out.append(sl)
        elif nd.name == sl.name:
            out.extend([fsizes, rs])
        else:
            out.append(nd)
    vis = [v for v in g.graph.value_info if v.name not in moved]
    del g.graph.node[:]
    g.graph.node.extend(out)
    del g.graph.value_info[:]
    g.graph.value_info.extend(vis)
    with open(onnx_path + ".tmp", "wb") as fh:
        fh.write(g.SerializeToString())
    os.replace(onnx_path + ".tmp", onnx_path)
    onnx.checker.check_model(onnx_path)


def _blockin_resize_parts(onnx_path, name):
    """RIFE IFNet: resize each part of a warped block's input on its own and concatenate the
    results, instead of concatenating the parts at the full size and resizing that. A bilinear
    resize works per channel, so the block input is the same bit for bit and the full-size concat
    goes. Only the blocks whose scale is not 1 (8, 4 and 2 of 16-8-4-2-1); block 0's input stays as
    it is, because split the same way it makes TensorRT-RTX 1.6.1.120 build an engine whose flow and
    mask are NaN. The IFNet runs about 3 % faster per call at 1080p and 4K. Graph only."""
    import re

    import onnx
    from onnx import helper, numpy_helper, TensorProto

    if not name.startswith("rife_ifnet_"):
        return
    sl = re.search(r"_(\d+(?:-\d+)+)(?:_|\.|$)", name)
    if sl is None:
        raise RuntimeError(f"{name}: no scale list in the name")
    scales = [int(s) for s in sl.group(1).split("-")]
    g = onnx.load(onnx_path, load_external_data=False)
    typed = onnx.shape_inference.infer_shapes(g).graph
    shp = {v.name: v.type.tensor_type.shape for v in list(typed.value_info) + list(typed.input)}
    prod = {o: nd for nd in g.graph.node for o in nd.output}
    cons = {}
    for nd in g.graph.node:
        for x in nd.input:
            cons.setdefault(x, []).append(nd)
    small = {i.name: i for i in g.graph.initializer if i.data_location != TensorProto.EXTERNAL}

    def warped(t):
        p = prod.get(t)
        while p is not None and p.op_type == "Cast":
            p = prod.get(p.input[0])
        return p is not None and p.op_type == "GridSample"

    blocks = []   # (concat, resize) of each warped block input, in graph order
    for cat in g.graph.node:
        if cat.op_type != "Concat" or not any(warped(x) for x in cat.input):
            continue
        rs = [nd for nd in cons.get(cat.output[0], []) if nd.op_type != "Shape"]
        if len(rs) != 1 or rs[0].op_type != "Resize" or len(rs[0].input) != 4 or not rs[0].name:
            raise RuntimeError(f"{name}: expected one Resize reading the block input {cat.name}")
        blocks.append((cat, rs[0]))
    if len(blocks) != len(scales) - 1:
        raise RuntimeError(f"{name}: {len(blocks)} warped block inputs for the scales {scales}")
    new_init = []
    split = {}   # resize name -> the nodes that replace it
    for (cat, rs), scale in zip(blocks, scales[1:]):
        if scale == 1:
            continue
        sizes = prod.get(rs.input[3])
        if sizes is None or sizes.op_type != "Concat" or len(sizes.input) != 2:
            raise RuntimeError(f"{name}: expected {rs.name} sized by Concat(lead, hw)")
        lead_t, hw = sizes.input
        lead = numpy_helper.to_array(small[lead_t]) if lead_t in small else None
        shape_nd = prod.get(lead_t)   # the batched graph takes the lead dims from the concat's shape
        shape_attrs = list(shape_nd.attribute) if shape_nd is not None else []
        attrs = {a.name: a.i for a in shape_attrs}
        if lead is None and (shape_nd is None or shape_nd.op_type != "Shape" or shape_nd.input[0] != cat.output[0]
                             or attrs != {"start": 0, "end": 2}):
            raise RuntimeError(f"{name}: {rs.name}'s lead dims are neither a constant nor the concat's shape")
        nodes, parts = [], []
        for k, p in enumerate(cat.input):
            lk = f"{rs.name}_lead{k}"
            if lead is not None:
                c = shp[p].dim[1].dim_value if p in shp else 0
                if c <= 0:
                    raise RuntimeError(f"{name}: no static channel count for {p}")
                new_init.append(helper.make_tensor(lk, TensorProto.INT64, [2], [int(lead[0]), c]))
            else:
                s = helper.make_node("Shape", [p], [lk], name=lk)
                s.attribute.extend(shape_attrs)
                nodes.append(s)
            nodes.append(helper.make_node("Concat", [lk, hw], [f"{rs.name}_sizes{k}"], name=f"{rs.name}_sizes{k}",
                                          axis=0))
            r = helper.make_node("Resize", [p] + list(rs.input[1:3]) + [f"{rs.name}_sizes{k}"],
                                 [f"{rs.name}_part{k}"], name=f"{rs.name}_part{k}")
            r.attribute.extend(rs.attribute)
            nodes.append(r)
            parts.append(f"{rs.name}_part{k}")
        cc = helper.make_node("Concat", parts, [rs.output[0]], name=f"{rs.name}_cat")
        cc.attribute.extend(cat.attribute)
        split[rs.name] = (cat.output[0], nodes + [cc])
    out = []
    for nd in g.graph.node:
        out.extend(split[nd.name][1] if nd.name in split else [nd])
    keep = {o.name for o in g.graph.output}
    while True:   # the full-size concats (and the batched graph's Shape readers) nothing reads any more
        used = {x for nd in out for x in nd.input} | keep
        alive = [nd for nd in out if any(o in used for o in nd.output)]
        if len(alive) == len(out):
            break
        out = alive
    made = {o for nd in out for o in nd.output}
    left = [c for c, _ in split.values() if c in made]
    if left:
        raise RuntimeError(f"{name}: the full-size block inputs {left} are still read")
    vis = [v for v in g.graph.value_info if v.name in made]
    del g.graph.node[:]
    g.graph.node.extend(out)
    g.graph.initializer.extend(new_init)
    del g.graph.value_info[:]
    g.graph.value_info.extend(vis)
    with open(onnx_path + ".tmp", "wb") as fh:
        fh.write(g.SerializeToString())
    os.replace(onnx_path + ".tmp", onnx_path)
    onnx.checker.check_model(onnx_path)


def _last_block_prune(onnx_path, name):
    """RIFE IFNet: the last block's transposed conv computes only the channels its flow and mask
    read. Its 52 output channels become 13 at the full size through the pixel shuffle, and the last
    block reads only the first 5 (4 flow + 1 mask; the other 8 are the features the earlier blocks
    hand on); the shuffle's output channel c comes from input channels 4c to 4c + 3, so the weight
    and bias are sliced to the first 20. A transposed conv's output channels are independent, so the
    flow and mask stay bit-identical. The IFNet runs about 5 % faster per call at 1080p and 4K.
    Graph only (the slices read the weights where they are)."""
    import onnx
    from onnx import helper, numpy_helper, TensorProto

    if not name.startswith("rife_ifnet_"):
        return
    g = onnx.load(onnx_path, load_external_data=False)
    prod = {o: nd for nd in g.graph.node for o in nd.output}
    cons = {}
    for nd in g.graph.node:
        for x in nd.input:
            cons.setdefault(x, []).append(nd)
    small = {i.name: numpy_helper.to_array(i) for i in g.graph.initializer if i.data_location != TensorProto.EXTERNAL}
    cts = [nd for nd in g.graph.node if nd.op_type == "ConvTranspose"]
    if not cts:
        raise RuntimeError(f"{name}: no transposed conv")
    ct = cts[-1]
    d2s = cons.get(ct.output[0], [])
    attrs = {a.name: helper.get_attribute_value(a) for a in d2s[0].attribute} if len(d2s) == 1 else {}
    if len(d2s) != 1 or d2s[0].op_type != "DepthToSpace" or attrs != {"blocksize": 2, "mode": b"CRD"}:
        raise RuntimeError(f"{name}: expected the last transposed conv read by one DepthToSpace(2, CRD)")
    d2s = d2s[0]
    rs = [nd for nd in cons[d2s.output[0]] if nd.op_type == "Resize"]
    others = [nd for nd in cons[d2s.output[0]] if nd.op_type != "Resize"]
    if len(rs) != 1 or len(rs[0].input) != 4 or not rs[0].name:
        raise RuntimeError(f"{name}: expected one sized Resize reading the last block's shuffle")
    rs = rs[0]
    sizes = prod.get(rs.input[3])
    if sizes is None or sizes.op_type != "Concat" or len(sizes.input) != 2:
        raise RuntimeError(f"{name}: expected {rs.name} sized by Concat(lead, hw)")
    lead = small.get(sizes.input[0])
    lead_nd = prod.get(sizes.input[0])   # the batched graph takes the lead dims from the shuffle's shape
    shape_lead = (lead_nd is not None and lead_nd.op_type == "Shape" and lead_nd.input[0] == d2s.output[0]
                  and {a.name: a.i for a in lead_nd.attribute} == {"start": 0, "end": 2})
    if lead is None and not shape_lead:
        raise RuntimeError(f"{name}: {rs.name}'s lead dims are neither a constant nor the shuffle's shape")
    if any(lead_nd is None or list(nd.output) != list(lead_nd.output) for nd in others):
        raise RuntimeError(f"{name}: the last block's shuffle has readers besides its Resize")
    used = 0
    for nd in cons.get(rs.output[0], []):
        slice_in = nd.op_type == "Slice" and len(nd.input) in (4, 5)
        end = small.get(nd.input[2]) if slice_in else None
        axes = small.get(nd.input[3]) if slice_in else None
        steps = small.get(nd.input[4]) if len(nd.input) == 5 else None
        if (not slice_in or small.get(nd.input[1]) is None or end is None or axes is None or list(axes) != [1]
                or (len(nd.input) == 5 and (steps is None or list(steps) != [1]))):
            raise RuntimeError(f"{name}: expected only constant channel slices to read {rs.name}")
        used = max(used, int(end[0]))
    channels = 13 if lead is None else int(lead[1])
    if lead is not None and (len(lead) != 2 or channels != 13):
        raise RuntimeError(f"{name}: expected {rs.name} sized [1, 13, H, W], got the lead {list(lead)}")
    if not 0 < used < channels:
        raise RuntimeError(f"{name}: the last block reads {used} of its {channels} channels")
    keep = 4 * used
    init = [helper.make_tensor("last_prune_lo", TensorProto.INT64, [1], [0]),
            helper.make_tensor("last_prune_hi", TensorProto.INT64, [1], [keep]),
            helper.make_tensor("last_prune_ax0", TensorProto.INT64, [1], [0]),
            helper.make_tensor("last_prune_ax1", TensorProto.INT64, [1], [1])]
    new = [helper.make_node("Slice", [ct.input[1], "last_prune_lo", "last_prune_hi", "last_prune_ax1"],
                            ["last_prune_weight"], name="last_prune_weight"),
           helper.make_node("Slice", [ct.input[2], "last_prune_lo", "last_prune_hi", "last_prune_ax0"],
                            ["last_prune_bias"], name="last_prune_bias")]
    if lead is not None:
        init.append(helper.make_tensor("last_prune_lead", TensorProto.INT64, [2], [int(lead[0]), used]))
        new.append(helper.make_node("Concat", ["last_prune_lead", sizes.input[1]], ["last_prune_sizes"],
                                    name="last_prune_sizes", axis=0))
        rs.input[3] = "last_prune_sizes"
    ct.input[1], ct.input[2] = "last_prune_weight", "last_prune_bias"
    moved = {ct.output[0], d2s.output[0], rs.output[0]}   # their channel counts change
    out = []
    for nd in g.graph.node:
        if nd.name == ct.name:
            out.extend(new)
        out.append(nd)
    vis = [v for v in g.graph.value_info if v.name not in moved]
    del g.graph.node[:]
    g.graph.node.extend(out)
    g.graph.initializer.extend(init)
    del g.graph.value_info[:]
    g.graph.value_info.extend(vis)
    with open(onnx_path + ".tmp", "wb") as fh:
        fh.write(g.SerializeToString())
    os.replace(onnx_path + ".tmp", onnx_path)
    onnx.checker.check_model(onnx_path)


def _last_block_conv_split(onnx_path, name):
    """RIFE IFNet: the last block's first conv reads its 52-channel input in parts instead of
    from one full-size concat. That block runs at scale 1, so its input is Concat(the 48-channel
    block input, the flow), both through identity resizes, and only this conv reads it. A conv is a
    sum over its input channels, so each part with 16 channels or more (the two feature warps) gets
    a conv of its own over its slice of the weight, the remaining 20 channels one more over a
    20-channel concat, and Adds join them (the bias on the first). The full-size 52-channel copy
    goes and the IFNet runs about 5.5 % faster per call at 480p, 1080p and 4K. NOT bit-identical:
    the fp16 sum runs in another order, the flow moves by up to one fp16 step and the tweens stay
    above 62 dB from the unsplit conv's at 480p and 1080p. Parts under 16 channels stay joined,
    because a conv that small runs slowly on TensorRT-RTX 1.6.1.120. The three weight slices are
    written into the graph (about 15 KB); the shared weight file stays as it is."""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    if not name.startswith("rife_ifnet_"):
        return
    g = onnx.load(onnx_path, load_external_data=False)
    typed = onnx.shape_inference.infer_shapes(g).graph
    shp = {v.name: [d.dim_value or d.dim_param for d in v.type.tensor_type.shape.dim]
           for v in list(typed.value_info) + list(typed.input)}
    nodes = list(g.graph.node)
    prod = {o: nd for nd in nodes for o in nd.output}
    cons = {}
    for nd in nodes:
        for x in nd.input:
            cons.setdefault(x, []).append(nd)
    cts = [i for i, nd in enumerate(nodes) if nd.op_type == "ConvTranspose"]
    if len(cts) < 2:
        raise RuntimeError(f"{name}: expected a transposed conv per block")
    conv = next((nd for nd in nodes[cts[-2] + 1:] if nd.op_type == "Conv"), None)
    if conv is None:
        raise RuntimeError(f"{name}: no conv after the last but one transposed conv")
    outer = prod.get(conv.input[0])
    if (outer is None or outer.op_type != "Concat" or helper.get_attribute_value(outer.attribute[0]) != 1
            or cons[conv.input[0]] != [conv] or len(conv.input) != 3):
        raise RuntimeError(f"{name}: expected the last block's first conv (with a bias) as the only reader of a "
                           "channel concat")
    full = shp["x"][2:4]
    gone = [conv, outer]
    parts = []
    for x in outer.input:
        rs = prod.get(x)
        attrs = {a.name: helper.get_attribute_value(a) for a in rs.attribute} if rs is not None else {}
        if (rs is None or rs.op_type != "Resize" or cons[x] != [outer] or attrs.get("mode") != b"linear"
                or attrs.get("coordinate_transformation_mode") != b"half_pixel" or shp.get(rs.input[0], [])[2:4] != full):
            raise RuntimeError(f"{name}: expected each part of the last block's input through an identity Resize")
        gone.append(rs)
        inner = prod.get(rs.input[0])
        others = [nd for nd in cons[rs.input[0]] if nd is not rs]   # the batched graph: Shapes for the size lead
        if inner is not None and inner.op_type == "Concat" and all(nd.op_type == "Shape" for nd in others):
            if helper.get_attribute_value(inner.attribute[0]) != 1:
                raise RuntimeError(f"{name}: {inner.name} is not a channel concat")
            gone.append(inner)
            parts += list(inner.input)
        else:
            parts.append(rs.input[0])
    dims = [shp.get(p, [0, 0])[1] for p in parts]
    if not all(isinstance(c, int) and c > 0 for c in dims):
        raise RuntimeError(f"{name}: no static channel count for a part of the last block's input ({dims})")
    chans = [int(c) for c in dims]
    init = {i.name: i for i in g.graph.initializer}
    if conv.input[1] not in init:
        raise RuntimeError(f"{name}: {conv.name}'s weight is not an initializer")
    wt = numpy_helper.to_array(init[conv.input[1]], base_dir=os.path.dirname(onnx_path))
    if sum(chans) != wt.shape[1]:
        raise RuntimeError(f"{name}: the parts' channels {chans} do not add up to the weight {list(wt.shape)}")
    big = [i for i, c in enumerate(chans) if c >= 16]
    rest = [i for i, c in enumerate(chans) if c < 16]
    groups = [[i] for i in big] + ([rest] if rest else [])
    if len(groups) < 2:
        raise RuntimeError(f"{name}: no part to split off ({chans})")
    offs = np.cumsum([0] + chans)
    new, new_init = [], []
    total = ""
    for k, grp in enumerate(groups):
        xin = parts[grp[0]]
        if len(grp) > 1:
            xin = f"last_split_cat{k}"
            new.append(helper.make_node("Concat", [parts[i] for i in grp], [xin], name=xin, axis=1))
        w = np.concatenate([wt[:, offs[i]:offs[i + 1]] for i in grp], axis=1)
        new_init.append(numpy_helper.from_array(np.ascontiguousarray(w), f"last_split_w{k}"))
        out = f"last_split_conv{k}"
        c = helper.make_node("Conv", [xin, f"last_split_w{k}"] + ([conv.input[2]] if k == 0 else []), [out],
                             name=out)
        c.attribute.extend(conv.attribute)
        new.append(c)
        if k > 0:
            out_sum = conv.output[0] if k == len(groups) - 1 else f"last_split_sum{k}"
            new.append(helper.make_node("Add", [total, out], [out_sum], name=f"last_split_sum{k}"))
            out = out_sum
        total = out
    at = nodes.index(conv)
    keep = [nd for nd in nodes[:at] if nd not in gone] + new + [nd for nd in nodes[at:] if nd not in gone]
    outs = {o.name for o in g.graph.output}
    while True:   # the Resizes' size math and the batched graph's Shape readers nothing reads any more
        used = {x for nd in keep for x in nd.input} | outs
        alive = [nd for nd in keep if any(o in used for o in nd.output)]
        if len(alive) == len(keep):
            break
        keep = alive
    removed = {o for nd in gone for o in nd.output} - {o for nd in new for o in nd.output}
    if [x for nd in keep for x in nd.input if x in removed]:
        raise RuntimeError(f"{name}: a node still reads the last block's old input")
    made = {o for nd in keep for o in nd.output}
    used = {x for nd in keep for x in nd.input}
    inits = [i for i in g.graph.initializer if i.name in used] + new_init
    vis = [v for v in g.graph.value_info if v.name in made and v.name != conv.output[0]]
    del g.graph.node[:]
    g.graph.node.extend(keep)
    del g.graph.initializer[:]
    g.graph.initializer.extend(inits)
    del g.graph.value_info[:]
    g.graph.value_info.extend(vis)
    with open(onnx_path + ".tmp", "wb") as fh:
        fh.write(g.SerializeToString())
    os.replace(onnx_path + ".tmp", onnx_path)
    onnx.checker.check_model(onnx_path)


def _block3_conv_split(onnx_path, name):
    """RIFE IFNet: block 3's first conv (52 -> 32 channels, 3x3, stride 2) reads its input in parts, as the last
    block's does. Its input is Concat(Concat(the seven resized parts: picture warps 3 + 3, feature warps 16 + 16,
    timestep 1, mask 1, features 8), the resized flow 4) at half the size, read only by this conv; the two 16-channel
    feature warps each get a conv over their slice of the weight, the other 20 channels one more over a 20-channel
    concat, and Adds join them (the bias on the first). The weight is an fp32 initializer behind a Cast to fp16, so
    each slice keeps a Cast of its own. The IFNet runs about 1.8 % faster per call at 1080p and 1.9 % at 4K (nothing
    measurable at 480p). NOT bit-identical (the fp16 sum runs in another order): the tweens move by 63 to 65 dB against the unsplit
    conv's and stay as close to an fp32 IFNet as before (53 to 57 dB pooled on RIFE x2 / x4, Frame Blend and DRBA). The
    graph's shape annotations (value_info) are dropped, the parser infers them. The three weight slices are written
    into the graph (about 60 KB)."""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    if not name.startswith("rife_ifnet_"):
        return
    g = onnx.load(onnx_path, load_external_data=False)
    typed = onnx.shape_inference.infer_shapes(g).graph
    shp = {v.name: [d.dim_value or d.dim_param for d in v.type.tensor_type.shape.dim]
           for v in list(typed.value_info) + list(typed.input)}
    nodes = list(g.graph.node)
    prod = {o: nd for nd in nodes for o in nd.output}
    cons = {}
    for nd in nodes:
        for x in nd.input:
            cons.setdefault(x, []).append(nd)
    init = {i.name: i for i in g.graph.initializer}
    base = os.path.dirname(onnx_path)

    def const(t):
        if t in init:
            return numpy_helper.to_array(init[t], base_dir=base)
        nd = prod.get(t)
        return numpy_helper.to_array(nd.attribute[0].t) if nd is not None and nd.op_type == "Constant" else None

    def chan(x):
        c = shp.get(x, [0, 0])[1]
        if isinstance(c, int) and c > 0:
            return c
        rs = prod.get(x)   # a Resize keeps the channels; the `_bd8` class's sizes lead with a Shape
        if rs is None or rs.op_type != "Resize" or len(rs.input) != 4:
            raise RuntimeError(f"{name}: no channel count for {x}")
        c = shp.get(rs.input[0], [0, 0])[1]
        if isinstance(c, int) and c > 0:
            return c
        sizes = prod.get(rs.input[3])
        lead = const(sizes.input[0]) if sizes is not None and sizes.op_type == "Concat" else None
        if lead is None or len(lead) != 2:
            raise RuntimeError(f"{name}: no channel count for {x}")
        return int(lead[1])

    cts = [i for i, nd in enumerate(nodes) if nd.op_type == "ConvTranspose"]
    if len(cts) < 3:
        raise RuntimeError(f"{name}: expected a transposed conv per block")
    conv = next((nd for nd in nodes[cts[2] + 1:] if nd.op_type == "Conv"), None)
    if conv is None:
        raise RuntimeError(f"{name}: no conv after block 3's transposed conv")
    outer = prod.get(conv.input[0])
    if (outer is None or outer.op_type != "Concat" or helper.get_attribute_value(outer.attribute[0]) != 1
            or cons[conv.input[0]] != [conv] or len(conv.input) != 3):
        raise RuntimeError(f"{name}: expected block 3's first conv (with a bias) as the only reader of a channel concat")
    gone = [conv, outer]
    parts = []
    for x in outer.input:
        inner = prod.get(x)
        if inner is not None and inner.op_type == "Concat" and cons[x] == [outer]:
            if helper.get_attribute_value(inner.attribute[0]) != 1:
                raise RuntimeError(f"{name}: {inner.name} is not a channel concat")
            gone.append(inner)
            parts += list(inner.input)
        else:
            parts.append(x)
    chans = [chan(p) for p in parts]
    cast = prod.get(conv.input[1])
    if cast is None or cast.op_type != "Cast" or cast.input[0] not in init:
        raise RuntimeError(f"{name}: expected block 3's weight as an initializer behind a Cast")
    wt = numpy_helper.to_array(init[cast.input[0]], base_dir=base)
    if chans != [3, 3, 16, 16, 1, 1, 8, 4] or wt.shape[1] != 52:
        raise RuntimeError(f"{name}: block 3's input parts {chans} / weight {list(wt.shape)} are not the expected 52")
    groups = [[i] for i, c in enumerate(chans) if c >= 16] + [[i for i, c in enumerate(chans) if c < 16]]
    offs = np.cumsum([0] + chans)
    new, new_init = [], []
    total = ""
    for k, grp in enumerate(groups):
        xin = parts[grp[0]]
        if len(grp) > 1:
            xin = f"fb3_cat{k}"
            new.append(helper.make_node("Concat", [parts[i] for i in grp], [xin], axis=1, name=xin))
        w = np.concatenate([wt[:, offs[i]:offs[i + 1]] for i in grp], axis=1)
        new_init.append(numpy_helper.from_array(np.ascontiguousarray(w), f"fb3_w{k}"))
        wc = helper.make_node("Cast", [f"fb3_w{k}"], [f"fb3_w{k}h"], name=f"fb3_cast{k}")
        wc.attribute.extend(cast.attribute)
        new.append(wc)
        out = f"fb3_conv{k}"
        c = helper.make_node("Conv", [xin, f"fb3_w{k}h"] + ([conv.input[2]] if k == 0 else []), [out], name=out)
        c.attribute.extend(conv.attribute)
        new.append(c)
        if k > 0:
            s = conv.output[0] if k == len(groups) - 1 else f"fb3_sum{k}"
            new.append(helper.make_node("Add", [total, out], [s], name=f"fb3_sum{k}"))
            out = s
        total = out
    keep = []
    for nd in nodes:
        if nd is conv:
            keep += new
        elif nd not in gone:
            keep.append(nd)
    outs = {o.name for o in g.graph.output}
    while True:   # the old weight Cast nothing reads any more
        used = {x for nd in keep for x in nd.input} | outs
        alive = [nd for nd in keep if any(o in used for o in nd.output)]
        if len(alive) == len(keep):
            break
        keep = alive
    removed = {o for nd in gone for o in nd.output} - {o for nd in new for o in nd.output}
    if [x for nd in keep for x in nd.input if x in removed]:
        raise RuntimeError(f"{name}: a node still reads block 3's old input")
    used = {x for nd in keep for x in nd.input}
    inits = [i for i in g.graph.initializer if i.name in used] + new_init
    del g.graph.node[:]
    g.graph.node.extend(keep)
    del g.graph.initializer[:]
    g.graph.initializer.extend(inits)
    del g.graph.value_info[:]
    with open(onnx_path + ".tmp", "wb") as fh:
        fh.write(g.SerializeToString())
    os.replace(onnx_path + ".tmp", onnx_path)
    onnx.checker.check_model(onnx_path)


def _blocks12_flat_concat(onnx_path, name):
    """RIFE IFNet: blocks 1 and 2's first conv reads one channel Concat of its eight parts instead of a Concat of the
    seven resized parts nested in a Concat with the resized flow. The parts and their order stay, so the conv reads
    the same tensor and the engines give bit-identical output; one Concat a block fewer for the builder to fuse (its
    speed sits inside the noise at 240p to 4K). Block b's conv = the first Conv after the b-th transposed conv. Graph
    only."""
    import onnx
    from onnx import helper

    if not name.startswith("rife_ifnet_"):
        return
    g = onnx.load(onnx_path, load_external_data=False)
    nodes = list(g.graph.node)
    prod = {o: nd for nd in nodes for o in nd.output}
    cons = {}
    for nd in nodes:
        for x in nd.input:
            cons.setdefault(x, []).append(nd)
    cts = [i for i, nd in enumerate(nodes) if nd.op_type == "ConvTranspose"]
    if len(cts) < 2:
        raise RuntimeError(f"{name}: expected a transposed conv per block")
    replace, gone = {}, []
    for b in (1, 2):
        conv = next((nd for nd in nodes[cts[b - 1] + 1:] if nd.op_type == "Conv"), None)
        if conv is None:
            raise RuntimeError(f"{name}: no conv after block {b}'s transposed conv")
        outer = prod.get(conv.input[0])
        if (outer is None or outer.op_type != "Concat" or helper.get_attribute_value(outer.attribute[0]) != 1
                or cons[conv.input[0]] != [conv]):
            raise RuntimeError(f"{name}: block {b}'s first conv does not read a channel concat of its own")
        parts, inner_n = [], 0
        for x in outer.input:
            inner = prod.get(x)
            if inner is not None and inner.op_type == "Concat" and cons[x] == [outer]:
                if helper.get_attribute_value(inner.attribute[0]) != 1:
                    raise RuntimeError(f"{name}: {inner.name} is not a channel concat")
                gone.append(inner)
                parts += list(inner.input)
                inner_n += 1
            else:
                parts.append(x)
        if inner_n != 1 or len(parts) != 8:
            raise RuntimeError(f"{name}: block {b}: expected one nested concat and 8 parts ({inner_n}, {len(parts)})")
        replace[id(outer)] = helper.make_node("Concat", parts, list(outer.output), axis=1, name=f"fb{b}_cat")
    keep = []
    for nd in nodes:
        if id(nd) in replace:
            keep.append(replace[id(nd)])
        elif all(nd is not x for x in gone):
            keep.append(nd)
    del g.graph.node[:]
    g.graph.node.extend(keep)
    with open(onnx_path + ".tmp", "wb") as fh:
        fh.write(g.SerializeToString())
    os.replace(onnx_path + ".tmp", onnx_path)
    onnx.checker.check_model(onnx_path)


def _encode_subpixel(onnx_path, name):
    """RIFE encode: its last layer, a transposed conv (4x4, stride 2, pad 1) from half the frame
    size to the full size, becomes its sub-pixel form: one Conv (3x3, pad 1) with four times the
    output channels at the half size, then DepthToSpace(2, DCR). Output pixel (2m + i, 2n + j)
    reads input rows m - 1 .. m + 1 through the transposed kernel's rows 3 - 2r + i (window row r,
    two of the three in range), columns alike, so each of the four phases keeps its 2x2 taps in a
    3x3 window, the other taps are exact zeros and the bias repeats per phase: the same products.
    TensorRT-RTX 1.6.1.120 runs the transposed conv as one slow kernel plus a full-size layout
    pass; the encode at 1472x2560 takes 1.19 instead of 1.76 ms. NOT bit-identical: the fp16 sum
    runs in another order, about a quarter of the features move by one fp16 step and RIFE's tweens
    stay above 64 dB from the transposed conv's (72 dB pooled on 2560x1440 anime), as close to an
    fp32 run of the model as before. The new weight (18 KB) is written into the graph."""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    if not name.startswith("rife_encode_"):
        return
    g = onnx.load(onnx_path, load_external_data=False)
    outs = {o.name for o in g.graph.output}
    cts = [nd for nd in g.graph.node if nd.op_type == "ConvTranspose"]
    if len(cts) != 1 or len(cts[0].input) != 3 or cts[0].output[0] not in outs:
        raise RuntimeError(f"{name}: expected one transposed conv with a bias writing the output")
    ct = cts[0]
    a = {x.name: helper.get_attribute_value(x) for x in ct.attribute}
    if (list(a.get("kernel_shape", [])) != [4, 4] or list(a.get("strides", [])) != [2, 2]
            or list(a.get("pads", [])) != [1, 1, 1, 1] or a.get("group", 1) != 1
            or list(a.get("dilations", [1, 1])) != [1, 1] or list(a.get("output_padding", [0, 0])) != [0, 0]):
        raise RuntimeError(f"{name}: expected a 4x4, stride 2, pad 1 transposed conv ({a})")
    init = {i.name: i for i in g.graph.initializer}
    if ct.input[1] not in init or ct.input[2] not in init:
        raise RuntimeError(f"{name}: the transposed conv's weight or bias is not an initializer")
    if any(x in ct.input[1:] for nd in g.graph.node if nd is not ct for x in nd.input):
        raise RuntimeError(f"{name}: another node reads the transposed conv's weight or bias")
    base = os.path.dirname(onnx_path)
    wt = numpy_helper.to_array(init[ct.input[1]], base_dir=base)
    bt = numpy_helper.to_array(init[ct.input[2]], base_dir=base)
    ci, co = wt.shape[:2]
    if wt.shape != (ci, co, 4, 4) or bt.shape != (co,):
        raise RuntimeError(f"{name}: unexpected weight {list(wt.shape)} / bias {list(bt.shape)}")
    w = np.zeros((4 * co, ci, 3, 3), wt.dtype)
    for i in range(2):
        for j in range(2):
            ph = i * 2 + j   # DCR: the phase is the outer channel index
            for ry in range(3):
                for rx in range(3):
                    ky, kx = 3 - 2 * ry + i, 3 - 2 * rx + j
                    if 0 <= ky <= 3 and 0 <= kx <= 3:
                        w[ph * co:(ph + 1) * co, :, ry, rx] = wt[:, :, ky, kx].T
    sub = "enc_subpixel"
    conv = helper.make_node("Conv", [ct.input[0], sub + "_w", sub + "_b"], [sub], name=sub + "_conv",
                            kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1])
    d2s = helper.make_node("DepthToSpace", [sub], [ct.output[0]], name=sub + "_d2s", blocksize=2, mode="DCR")
    nodes = list(g.graph.node)
    at = nodes.index(ct)
    nodes[at:at + 1] = [conv, d2s]
    inits = [i for i in g.graph.initializer if i.name not in ct.input[1:]]
    inits += [numpy_helper.from_array(w, sub + "_w"), numpy_helper.from_array(np.tile(bt, 4), sub + "_b")]
    del g.graph.node[:]
    g.graph.node.extend(nodes)
    del g.graph.initializer[:]
    g.graph.initializer.extend(inits)
    with open(onnx_path + ".tmp", "wb") as fh:
        fh.write(g.SerializeToString())
    os.replace(onnx_path + ".tmp", onnx_path)
    onnx.checker.check_model(onnx_path)


def _fusion_fp16_inputs(onnx_path, name):
    """GMFSS fusionnet: take the splatted feature levels b / c / d as fp16 inputs. Each one's only
    reader is a Cast to fp16, so that Cast goes; the host's splat normalization rounds to nearest
    even as the Cast did, the engine gives bit-identical output and reads half the bytes (fusionnet
    at the 1080p half 0.96x the time). `a` stays fp32: the GMFSS IFNet writes its middle planes in
    fp32. Graph only."""
    import onnx
    from onnx import TensorProto

    if name != "fusionnet":
        return
    g = onnx.load(onnx_path, load_external_data=False)
    for key in ("b", "c", "d"):
        inp = [i for i in g.graph.input if i.name == key]
        if len(inp) != 1 or inp[0].type.tensor_type.elem_type != TensorProto.FLOAT:
            raise RuntimeError(f"{name}: expected an fp32 input {key}")
        readers = [nd for nd in g.graph.node if key in nd.input]
        if (len(readers) != 1 or readers[0].op_type != "Cast"
                or [a.i for a in readers[0].attribute if a.name == "to"] != [TensorProto.FLOAT16]):
            raise RuntimeError(f"{name}: {key} is not read by exactly one Cast to fp16")
        old = readers[0].output[0]
        g.graph.node.remove(readers[0])
        for nd in g.graph.node:
            for k, x in enumerate(nd.input):
                if x == old:
                    nd.input[k] = key
        inp[0].type.tensor_type.elem_type = TensorProto.FLOAT16
    with open(onnx_path + ".tmp", "wb") as fh:
        fh.write(g.SerializeToString())
    os.replace(onnx_path + ".tmp", onnx_path)
    onnx.checker.check_model(onnx_path)


# the host's names for what crosses the gmflow split, in its order (smv-live kGmA after q, f, coords)
_GMFLOW_CROSS = ("win", "valid", "flow0", "feat", "flow1")


def gmflow_split(full_path, a_key="gmflow_bidir_a", with_b=True):
    """gmflow_bidir as two graphs around its quarter-scale local correlation, which the host computes between them
    (k_localCorr: never the [2, 128, HW, 81] fp32 product the graph writes and sums back, ~11 % of a call at 960x544).
    The correlation = the one ReduceSum of Mul(GridSample(f, coords), q). A = img0, img1 -> q [2, HW, 128] and
    f [2, H, W, 128] channels-last (one coalesced read a tap), coords [2, HW, K, 2], then every tensor made before
    the correlation and read after it that derives from the frames (Shape included) and is not an integer shape
    vector (B recomputes those from its own img0 / img1 inputs, and weight-only tensors from its initializers),
    renamed by role: valid = the bool one, feat = the 128-channel map, win = the [2, HW, K, 2] window, flow0 = the
    [2, 2, H, W] one derived from the frames' values, flow1 = the one made from the frame size alone. B = corr
    [2, HW, K] + those + img0 / img1 -> flow. The exporter's own names change from export to export, so the cut is
    found by structure and any other shape refuses. Writes trt_lookup.onnx_path(a_key) and, with_b, ("gmflow_bidir_b")
    with their own weights; returns the paths written."""
    import onnx
    from onnx import helper, shape_inference, TensorProto

    m = onnx.load(full_path)
    g = m.graph
    prod = {o: n for n in g.node for o in n.output}
    found = []
    for n in g.node:
        mul = prod.get(n.input[0]) if n.op_type == "ReduceSum" else None
        if mul is None or mul.op_type != "Mul":
            continue
        gs = [prod[x] for x in mul.input if x in prod and prod[x].op_type == "GridSample"]
        if len(gs) == 1:
            found.append((n, mul, gs[0]))
    if len(found) != 1:
        raise RuntimeError(f"gmflow_bidir: {len(found)} local correlations (ReduceSum of GridSample x q), expected 1")
    rs, mul, gs = found[0]
    q = next(x for x in mul.input if x != gs.output[0])
    f, coords = gs.input[0], gs.input[1]
    inf = shape_inference.infer_shapes(m)
    vi = {v.name: v.type.tensor_type for v in list(inf.graph.value_info) + list(inf.graph.output)}

    def kind(x):
        t = vi[x]
        return t.elem_type, [d.dim_value if d.HasField("dim_value") else None for d in t.shape.dim]

    data, vals = {i.name for i in g.input}, {i.name for i in g.input}
    for n in g.node:   # an exported graph is topologically sorted
        if any(x in data for x in n.input):
            data.update(n.output)
        if n.op_type not in ("Shape", "Size") and any(x in vals for x in n.input):
            vals.update(n.output)
    at = {o: i for i, n in enumerate(g.node) for o in n.output}
    c = at[rs.output[0]]
    cross = []
    for n in list(g.node)[c + 1:]:
        for x in n.input:
            if x in data and x in at and at[x] < c and x not in cross:
                dt, dims = kind(x)
                if not (dt in (TensorProto.INT64, TensorProto.INT32) and len(dims) <= 1):
                    cross.append(x)
    if set(cross) & {q, f, coords}:
        raise RuntimeError("gmflow_bidir: a correlation input is read after the correlation too")
    cross.sort(key=lambda x: at[x])
    names, flows = {}, []
    for x in cross:
        dt, dims = kind(x)
        if dt == TensorProto.BOOL and len(dims) == 3:
            role = "valid"
        elif dt == TensorProto.FLOAT and len(dims) == 4 and dims[1] == 128:
            role = "feat"
        elif dt == TensorProto.FLOAT and len(dims) == 4 and dims[0] == 2 and dims[3] == 2:
            role = "win"
        elif dt == TensorProto.FLOAT and len(dims) == 4 and dims[:2] == [2, 2]:
            flows.append(x)
            continue
        else:
            raise RuntimeError(f"gmflow_bidir: crossing tensor {x} ({dims}, dtype {dt}) has no role in the split")
        if role in names.values():
            raise RuntimeError(f"gmflow_bidir: two crossing tensors read as {role}")
        names[x] = role
    if len(flows) != 2 or len(names) != 3 or sum(x in vals for x in flows) != 1:
        raise RuntimeError(f"gmflow_bidir: crossing tensors {cross} do not match the split contract")
    f0 = flows[0] if flows[0] in vals else flows[1]
    names[f0], names[flows[1] if f0 == flows[0] else flows[0]] = "flow0", "flow1"
    cross.sort(key=lambda x: _GMFLOW_CROSS.index(names[x]))
    ex = onnx.utils.Extractor(m)
    a = ex.extract_model(["img0", "img1"], [q, f, coords] + cross)
    b = ex.extract_model([rs.output[0]] + cross + ["img0", "img1"], ["flow"])
    ga = a.graph
    ga.initializer.append(helper.make_tensor("cl_qshape", TensorProto.INT64, [3], [0, 0, -1]))
    ga.node.extend([helper.make_node("Reshape", [q, "cl_qshape"], ["cl_q3"], name="cl_q3"),
                    helper.make_node("Transpose", ["cl_q3"], ["q"], name="cl_qt", perm=[0, 2, 1]),
                    helper.make_node("Transpose", [f], ["f"], name="cl_ft", perm=[0, 2, 3, 1])])
    outs = list(ga.output)
    del ga.output[:]
    ga.output.extend([helper.make_tensor_value_info("q", TensorProto.FLOAT, [2, "hw4", 128]),
                      helper.make_tensor_value_info("f", TensorProto.FLOAT, [2, "h4", "w4", 128])] + outs[2:])
    paths = []
    for mm, key, mapping in ((a, a_key, {coords: "coords", **names}),
                             (b, "gmflow_bidir_b" if with_b else None, {rs.output[0]: "corr", **names})):
        if key is None:
            continue
        for n in mm.graph.node:
            n.input[:] = [mapping.get(x, x) for x in n.input]
            n.output[:] = [mapping.get(x, x) for x in n.output]
        for v in list(mm.graph.input) + list(mm.graph.output) + list(mm.graph.value_info):
            v.name = mapping.get(v.name, v.name)
        path = trt_lookup.onnx_path(key)
        tmp = path + ".tmp.onnx"
        onnx.save(mm, tmp, save_as_external_data=True, location=os.path.basename(path) + ".data")
        onnx.checker.check_model(tmp)
        # the weights file is named for the final graph, so only the graph moves
        os.replace(tmp, path)
        paths.append(path)
    _log(f"[trt] gmflow_bidir split into {' + '.join(os.path.basename(p) for p in paths)}")
    return paths


def gmflow_backbone_cut(a_path, with_bone=True):
    """gmflow_bidir_a's CNN backbone as its own graph for ONE frame, so the host computes each frame's maps once (the
    previous frame's come from the last pair) instead of both frames' every pair. Found by structure: the maps = the
    inputs of the two batch Splits (axis 0, two ways) fed straight by a Conv, named map4 / map8 by their size; the
    batch = the one axis-0 Concat of a tensor that depends on img0 alone and one on img1 alone (the normalized frames).
    gmflow_backbone = that Concat replaced by its img0 branch, img0 -> map4, map8 [1, C, H, W]; gmflow_bidir_a is
    rewritten to take map4, map8 [2, C, H, W] (+ img0 / img1, read for shapes). Not bit-identical to the batch-2
    backbone (other fp16 kernels at batch 1): GMFSS tweens vs an fp32 reference stay at the whole graph's level.
    Writes a_path in place and, with_bone, trt_lookup.onnx_path("gmflow_backbone"); returns the backbone's path (None
    without it)."""
    import onnx
    from onnx import helper, shape_inference

    a = onnx.load(a_path)
    g = a.graph
    prod = {o: n for n in g.node for o in n.output}

    def attr(n, k, d=None):
        for x in n.attribute:
            if x.name == k:
                return helper.get_attribute_value(x)
        return d

    splits = [n for n in g.node if n.op_type == "Split" and attr(n, "axis", 0) == 0 and len(n.output) == 2
              and n.input[0] in prod and prod[n.input[0]].op_type == "Conv"]
    if len(splits) != 2:
        raise RuntimeError(f"gmflow_bidir_a: {len(splits)} Conv-fed batch Splits, expected the backbone's 2")
    maps = [s.input[0] for s in splits]
    deps = {i.name: {i.name} for i in g.input}
    for n in g.node:   # an exported graph is topologically sorted
        d = set().union(*(deps.get(x, set()) for x in n.input))
        for o in n.output:
            deps[o] = d
    cats = [n for n in g.node if n.op_type == "Concat" and attr(n, "axis") == 0 and len(n.input) == 2
            and deps.get(n.input[0]) == {"img0"} and deps.get(n.input[1]) == {"img1"}]
    if len(cats) != 1:
        raise RuntimeError(f"gmflow_bidir_a: {len(cats)} frame batch Concats, expected 1")
    vi = {v.name: v.type.tensor_type for v in shape_inference.infer_shapes(a).graph.value_info}
    lead = {x: vi[x].shape.dim[2].dim_param.split("*")[0] for x in maps}   # '8*s8' (1/4) vs '4*s8' (1/8)
    if not all(v.isdigit() for v in lead.values()) or len(set(lead.values())) != 2:
        raise RuntimeError(f"gmflow_bidir_a: the backbone maps' heights {lead} do not tell the two scales apart")
    maps.sort(key=lambda x: -int(lead[x]))
    outs = [o.name for o in g.output]
    ex = onnx.utils.Extractor(a)
    rest = ex.extract_model(maps + ["img0", "img1"], outs)
    bone = ex.extract_model(["img0", "img1"], maps)
    cat = next(n for n in bone.graph.node if n.output[0] == cats[0].output[0])
    at = list(bone.graph.node).index(cat)
    bone.graph.node.remove(cat)
    bone.graph.node.insert(at, helper.make_node("Identity", [cat.input[0]], [cat.output[0]], name="one_frame"))
    del bone.graph.value_info[:]
    for o in bone.graph.output:
        o.type.tensor_type.shape.dim[0].dim_value = 1
    bone = onnx.utils.Extractor(bone).extract_model(["img0"], maps)
    names = {maps[0]: "map4", maps[1]: "map8"}
    paths = []
    for mm, path in ((bone, trt_lookup.onnx_path("gmflow_backbone") if with_bone else None), (rest, a_path)):
        if path is None:
            paths.append(None)
            continue
        for n in mm.graph.node:
            n.input[:] = [names.get(x, x) for x in n.input]
            n.output[:] = [names.get(x, x) for x in n.output]
        for v in list(mm.graph.input) + list(mm.graph.output) + list(mm.graph.value_info):
            v.name = names.get(v.name, v.name)
        tmp = path + ".tmp.onnx"
        onnx.save(mm, tmp, save_as_external_data=True, location=os.path.basename(path) + ".data")
        onnx.checker.check_model(tmp)
        os.replace(tmp, path)
        paths.append(path)
    _log(f"[trt] gmflow's backbone cut out{' as ' + os.path.basename(paths[0]) if paths[0] else ''}")
    return paths[0]


def gmflow_matching_cut(a_path, key):
    """gmflow_bidir_a cut around its two attentions with a 2-channel value (the global matching: Softmax over
    Concat(X, X^T), X = Reshape(Div(Reshape(MatMul(f0, f1)))), times the pixel grid; the flow propagation:
    Softmax(Div(MatMul(q, k^T))) times the flow), which the host runs as k_attn2 between two engines: fp32 scores,
    softmax and value product, and the N x N matrices never exist (the graph's own fp16 probabilities over N keys put
    its flow 8x further from the fp32 model at the 1440p DLAA half). Writes `key`_m beside a_path: the same inputs ->
    mf (f0 and f1 token-major, [2, N, C]), mq (the propagation's queries) and mk (its keys, [2, N, C] each), the graph's
    dtype; and rewrites a_path to take `prop` (the propagation's output [2, N, 2]) instead of computing it. Found by
    structure (gmflow's transformer heads carry 128-channel values); any other count refuses. Both graphs keep the
    external weights file; nodes nothing reads any more are dropped. Returns the _m path."""
    import copy

    import onnx
    from onnx import helper, shape_inference

    m = onnx.load(a_path, load_external_data=False)
    g = m.graph
    inferred = {v.name: v for v in shape_inference.infer_shapes(m).graph.value_info}
    vi = {k: v.type.tensor_type.shape for k, v in inferred.items()}
    prod = {o: n for n in g.node for o in n.output}
    cons = {}
    for n in g.node:
        for x in n.input:
            cons.setdefault(x, []).append(n)

    def two_channel(t):
        s = vi.get(t)
        return s is not None and len(s.dim) > 0 and s.dim[-1].dim_value == 2

    chains = {"prop": [], "global": []}
    for sm in (n for n in g.node if n.op_type == "Softmax"):
        users = cons.get(sm.output[0], [])
        if len(users) != 1 or users[0].op_type != "MatMul" or not two_channel(users[0].input[1]):
            continue
        mm_v = users[0]
        src = prod[sm.input[0]]
        if src.op_type == "Div" and prod[src.input[0]].op_type == "MatMul":
            mm = prod[src.input[0]]
            chains["prop"].append((mm.input[0], mm.input[1], mm_v.output[0]))
        elif src.op_type == "Concat" and len(src.input) == 2:
            x, xt = src.input
            r2 = prod[x]
            if r2.op_type != "Reshape" or prod[xt].op_type != "Transpose" or prod[xt].input[0] != x:
                continue
            div = prod[r2.input[0]]
            if div.op_type != "Div":
                continue
            r1 = prod[div.input[0]]
            if r1.op_type != "Reshape":
                continue
            mm = prod[r1.input[0]]
            if mm.op_type == "MatMul":
                chains["global"].append((mm.input[0], mm.input[1], mm_v.output[0]))
    for kind, found in chains.items():
        if len(found) != 1:
            raise RuntimeError(f"gmflow_bidir_a: {len(found)} {kind} attentions with a 2-channel value, expected 1")
    f0, f1t, _ = chains["global"][0]   # f0 tokens [1, N, C], f1 channels [1, C, N]
    q, kt, pout = chains["prop"][0]   # q [2, N, C], k^T [2, C, N], the output [2, N, 2]

    def dims(t):
        return [d.dim_value or d.dim_param for d in vi[t].dim]

    if dims(f0)[-1] != 128 or dims(f1t)[1] != 128 or dims(q)[-1] != 128 or dims(kt)[1] != 128:
        raise RuntimeError(f"gmflow_bidir_a: the attentions' operands are {dims(f0)} / {dims(f1t)} / {dims(q)} / "
                           f"{dims(kt)}, expected tokens x 128 and 128 x tokens")
    dt = inferred[q].type.tensor_type.elem_type
    cut = [helper.make_node("Transpose", [f1t], ["cut_f1"], perm=[0, 2, 1], name="cut_f1"),
           helper.make_node("Concat", [f0, "cut_f1"], ["mf"], axis=0, name="cut_mf"),
           helper.make_node("Identity", [q], ["mq"], name="cut_mq"),
           helper.make_node("Transpose", [kt], ["mk"], perm=[0, 2, 1], name="cut_mk")]

    def tokens(name):
        """[2, N, 128] in the graph's dtype, N the propagation queries' own (symbolic) dim"""
        v = helper.make_tensor_value_info(name, dt, [2, 1, 128])
        v.type.tensor_type.shape.dim[1].CopyFrom(vi[q].dim[1])
        return v

    def reachable(mm, nodes, outputs):
        """keep the nodes (and initializers, value infos) the outputs read"""
        gg = mm.graph
        need = set(outputs)
        keep = []
        for n in reversed(nodes):
            if any(o in need for o in n.output):
                keep.append(n)
                need.update(x for x in n.input if x)
        del gg.node[:]
        gg.node.extend(reversed(keep))
        live = {x for n in gg.node for x in list(n.input) + list(n.output)}
        inits = [t for t in gg.initializer if t.name in live]
        del gg.initializer[:]
        gg.initializer.extend(inits)
        vis = [v for v in gg.value_info if v.name in live]
        del gg.value_info[:]
        gg.value_info.extend(vis)

    def write(mm, path):
        tmp = path + ".tmp.onnx"
        with open(tmp, "wb") as fh:
            fh.write(mm.SerializeToString())
        onnx.checker.check_model(tmp)
        os.replace(tmp, path)

    m_path = trt_lookup.onnx_path(key + "_m")
    mm = copy.deepcopy(m)
    del mm.graph.output[:]
    mm.graph.output.extend([tokens("mf"), tokens("mq"), tokens("mk")])
    reachable(mm, list(mm.graph.node) + cut, ["mf", "mq", "mk"])
    write(mm, m_path)
    for n in g.node:
        n.input[:] = ["prop" if x == pout else x for x in n.input]
    p = copy.deepcopy(inferred[pout])
    p.name = "prop"
    g.input.append(p)
    reachable(m, list(g.node), [o.name for o in g.output])
    write(m, a_path)
    _log(f"[trt] gmflow's matching cut out: {os.path.basename(m_path)} + {os.path.basename(a_path)} (input prop)")
    return m_path


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
        _half_flow_mask(tmp, name)
        _half_timestep(tmp, name)
        _half_blockin(tmp, name)
        _b0_slice_first(tmp, name)
        _blockin_resize_parts(tmp, name)
        _last_block_prune(tmp, name)
        _last_block_conv_split(tmp, name)
        _block3_conv_split(tmp, name)
        _blocks12_flat_concat(tmp, name)
        _encode_subpixel(tmp, name)
        _fusion_fp16_inputs(tmp, name)
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
    """Export this graph's size-free ONNX if it qualifies and is missing (scripts/export-onnx.ts
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
    """RIFE IFNet forward with scale_list baked; timestep and the two feature encodes (f0,f1) are
    tensor inputs. Outputs: the final flow (0->t | 1->t) and the blend mask; the host makes the
    tween with RIFE's own last step (warp each frame along its flow, blend by the mask), on the
    frames the flow came from or on other pictures of the same pair (flow_list's coarser levels
    and the fused frame are dropped)."""

    def __init__(self, ifnet, scale_list):
        super().__init__()
        self.ifnet = ifnet
        self.scale_list = scale_list

    def forward(self, x, timestep, f0, f1):
        _, flows, mask = self.ifnet(x, timestep=timestep, scale_list=self.scale_list, f0=f0, f1=f1,
                                    return_mask=True)
        return flows[-1], mask


def _rife_ifnet_base(scale_list, whash):
    """The rife checkpoint fingerprint plus the baked scale_list are in the NAME (4K's 0.5 flow
    scale changes it)."""
    scale_tag = "-".join(f"{s:g}" for s in scale_list)
    return f"rife_ifnet_{whash}_{scale_tag}"


class RifeIFNetEngine(_Engine):
    """One RIFE IFNet forward (the unbatched class). timestep rides as a (1,1,H,W) tensor so both
    the scalar plain-RIFE path and DRBA's spatial DistanceRatioMap run the same engine."""

    def __init__(self, scale_list, whash):
        super().__init__(_rife_ifnet_base(scale_list, whash),
                         ["x", "timestep", "f0", "f1"], ["flow", "mask"])


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
            _, flows, mask = self.ifnet(x, timestep=timestep, scale_list=self.scale_list,
                                        f0=f0, f1=f1, return_mask=True)
        else:
            _, flows, mask = self.ifnet(x.expand(b, -1, -1, -1), timestep=timestep,
                                        scale_list=self.scale_list,
                                        f0=f0.expand(b, -1, -1, -1),
                                        f1=f1.expand(b, -1, -1, -1), return_mask=True)
        return flows[-1], mask


class RifeIFNetBatchEngine(_Engine):
    """1 to b tweens of one pair per enqueue (the live `_bd{b}` class; the host builds the offline
    fixed-batch `_b{B}` engines from the same graph). `_bd{b}` in the name keeps it apart from the
    unbatched class."""

    def __init__(self, scale_list, whash, b):
        super().__init__(_rife_ifnet_base(scale_list, whash) + f"_bd{b}",
                         ["x", "timestep", "f0", "f1"], ["flow", "mask"])
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
