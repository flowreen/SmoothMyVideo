"""Export the size-free ONNX of every live graph into engine/onnx (trt_lookup.onnx_path), so an
engine at a new window size is a build from that file instead of a torch export. Run by
scripts/export-onnx.js (npm setup and dist). While the export stamp (trt_lookup.export_stamp: the
sources, the exporter packages, the weights) matches the one in weights_tags.txt, every file
already present is skipped, so a rerun is cheap; a changed stamp removes the folder's graphs and
exports every one again. Graphs: the RIFE IFNet (live `_bd8` and the
unbatched class) and its encode, plain and as the flow-warp class at k 1 (Frame Blend), 2 and 4,
DRBA's block0, Restore, and the GMFSS nets (gmflow_bidir as its backbone and two halves, trt_runtime.gmflow_split +
gmflow_backbone_cut, the first half cut around its two N x N attentions by gmflow_matching_cut (`_m` before them, the
host's k_attn2 between), and that half again with its quarter-scale transformer in 2 / 4 / 8 window groups,
GMFLOW_CHUNKS, and in 8 with the shifted
blocks' attention per mask region, GMFLOW_REGION_CHUNKS; every transformer attention with its projections folded,
transformer.fold_projections). The host
builds the offline fixed-batch RIFE
classes (`_b{B}`) from `_bd8`. Then ship_tidy() and write_tags().
Usage: runtime python engine/onnx_export.py"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "GMFSS_Fortuna"))

import torch  # noqa: E402

import realesr  # noqa: E402
import trt_lookup  # noqa: E402
import trt_runtime as tr  # noqa: E402
from rife_backend import RIFE  # noqa: E402

PH, PW = 576, 960   # any /64 size inside the symbolic range works as the example
GMFLOW_CHUNKS = (2, 4, 8)   # gmflow_bidir_a's window-group variants (the host's kGmFlowA)
GMFLOW_REGION_CHUNKS = (8,)   # the same with transformer.EXPORT_REGION_ATTENTION, `_g<N>r` (the host's large-size pick)
# the scales (attn_num_splits) each gmflow_bidir_a variant exports with transformer.EXPORT_TOKEN_OUTER, as the 160 W
# A/Bs read them, outputs bit-identical: token-outer at the quarter scale x0.86 to x0.94 at every variant's size; at the
# coarse scale (the `_m` graph) token-outer x0.80 / x0.97 / x0.99 on the 1 / 2 / 8r-group graphs, window order on the
# 4-group graph (x0.97) and the 8-group one (x0.89 / x0.86 at Balanced / Quality)
GMFLOW_TOKEN_OUTER = {"gmflow_bidir_a": (2, 8), "gmflow_bidir_a_g2": (2, 8), "gmflow_bidir_a_g4": (8,),
                      "gmflow_bidir_a_g8": (8,), "gmflow_bidir_a_g8r": (2, 8)}
# the token-outer scales that add the position encoding after the layout change (transformer.EXPORT_TOKEN_POSITION):
# the entry kernel at Balanced's half 1.36 -> 0.72 ms (55 W map), outputs bit-identical; the whole graph's quarter
# scale keeps it before the layout change (its engine's outputs moved, the 360p tweens 58.6 dB off byte identity)
GMFLOW_TOKEN_POSITION = {"gmflow_bidir_a": (2,), "gmflow_bidir_a_g2": (2, 8), "gmflow_bidir_a_g4": (8,),
                         "gmflow_bidir_a_g8": (8,), "gmflow_bidir_a_g8r": (2, 8)}
# every token-outer scale keeps its stream in fp16 (transformer.EXPORT_FP16_STREAM): the 160 W A call x0.96 to x0.97
# from Ultra to DLAA (each block's glue moves half the bytes), the GMFSS tweens' PSNR vs eager fp32 within -0.3 / +0.2
# dB of the fp32 stream's, real frames identical
GMFLOW_FP16_STREAM = True


def main():
    t0 = time.time()
    stamp = trt_lookup.export_stamp()
    have = read_tag("export")
    if have != stamp:
        clear_graphs()
        print(f"export stamp {have or 'none'} -> {stamp}: exporting every graph", flush=True)
    torch.cuda.set_stream(torch.cuda.Stream())
    g = torch.Generator(device="cuda").manual_seed(0)
    a = torch.rand((1, 3, PH, PW), device="cuda", generator=g)
    b = torch.rand((1, 3, PH, PW), device="cuda", generator=g)
    x = torch.cat((a, b), 1)
    rh = trt_lookup.rife_weights_tag()
    done = []

    def ensure(eng, export_module, ins):
        p = tr.ensure_onnx(eng.name, export_module, ins, eng.input_names, eng.output_names,
                           eng.dyn_batch)
        if p is None:
            raise RuntimeError(f"{eng.name}: this graph does not qualify for a size-free export")
        done.append(os.path.basename(p))

    # RIFE (plain RIFE, Frame Blend and DRBA share these graphs)
    rife = RIFE()
    net, sl = rife.ifnet, list(rife.scale_list)
    with torch.inference_mode():
        f0, f1 = net.encode(a).float(), net.encode(b).float()
    ts = torch.cat([x[:, :1] * 0 + t for t in (0.25, 0.5, 0.75)], 0)
    e = tr.RifeIFNetBatchEngine(sl, rh, 8)
    ensure(e, tr._RifeIFNetBatchExport(net, sl), (x, ts, f0, f1))
    e = tr.RifeEncodeEngine(rh)
    ensure(e, tr._RifeEncodeExport(net.encode), (a,))
    # the unbatched class: the `.nofit` fallback and the offline single-tween calls
    e = tr.RifeIFNetEngine(sl, rh)
    ensure(e, tr._RifeIFNetExport(net, sl), (x, x[:, :1] * 0 + 0.5, f0, f1))
    e = tr.RifeBlock0Engine(float(sl[0]), rh)
    ensure(e, tr._RifeBlock0Export(net.block0, float(sl[0])), (a, b, f0, f1))
    del rife, net

    # Restore: fp16 at the model size (the caller halves the frame)
    rnet = realesr.load("cuda")
    ensure(tr.RestoreEngine(realesr.weights_hash()), rnet, (a.half(),))
    del rnet

    # GMFSS: the exact tensors an eager reuse + inference hands each sub net (forward pre-hooks)
    cwd = os.getcwd()
    os.chdir(os.path.join(HERE, "GMFSS_Fortuna"))
    try:
        from model.GMFSS_infer_u import Model

        gm = Model()
        gm.load_model("train_log", -1)
        gm.eval()
        gm.device()
    finally:
        os.chdir(cwd)
    from model.gmflow import transformer as gmt

    # gmflow's attention exports with its projections folded (two of four GEMMs an attention, transformer.py)
    gmt.fold_projections(gm.flownet)
    rec = {}
    hooks = []
    for nm in ("feat_ext", "flownet", "metricnet", "ifnet", "fusionnet"):
        def pre(mod, args, nm=nm):
            rec.setdefault(nm, tuple(q.detach().clone() if torch.is_tensor(q) else q for q in args))
        hooks.append(getattr(gm, nm).register_forward_pre_hook(pre))
    with torch.inference_mode():
        gm.inference(a, b, gm.reuse(a, b, 1.0), 0.5)
    for hk in hooks:
        hk.remove()
    ensure(tr.FeatEngine(), gm.feat_ext, rec["feat_ext"][:1])
    # the host runs gmflow_bidir as its backbone (one frame a call) and two halves around its own local correlation,
    # the first half itself as `_m` + the rest around its own global matching and propagation (k_attn2): the whole
    # graph is exported only to be cut (ship_tidy then removes it), and parts already present skip all of it
    parts = [trt_lookup.onnx_path(k) for k in
             ("gmflow_backbone", "gmflow_bidir_a_m", "gmflow_bidir_a", "gmflow_bidir_b")]
    if not all(os.path.isfile(p) for p in parts):
        gmt.EXPORT_TOKEN_OUTER = GMFLOW_TOKEN_OUTER.get("gmflow_bidir_a", ())
        gmt.EXPORT_TOKEN_POSITION = GMFLOW_TOKEN_POSITION.get("gmflow_bidir_a", ())
        gmt.EXPORT_FP16_STREAM = GMFLOW_FP16_STREAM
        try:
            ensure(tr.BidirFlowEngine(), tr._BidirFlowExport(gm.flownet), rec["flownet"][:2])
        finally:
            gmt.EXPORT_TOKEN_OUTER = ()
            gmt.EXPORT_TOKEN_POSITION = ()
            gmt.EXPORT_FP16_STREAM = False
        a_path, _ = tr.gmflow_split(os.path.join(trt_lookup.ONNX_DIR, done.pop()))
        tr.gmflow_backbone_cut(a_path)
        tr.gmflow_matching_cut(a_path, "gmflow_bidir_a")
    done.extend(os.path.basename(p) for p in parts)
    # gmflow_bidir_a again with its quarter-scale transformer in 2 / 4 / 8 window groups (transformer.py
    # EXPORT_WINDOW_CHUNKS), and in 8 with the shifted blocks' attention per mask region (EXPORT_REGION_ATTENTION): the
    # host picks one by the quarter-scale token count (lkGmFlowA); _b and the backbone are the same graphs at every
    # count, and ship_tidy merges the variants' identical weights into one file
    for k, regions in [(k, False) for k in GMFLOW_CHUNKS] + [(k, True) for k in GMFLOW_REGION_CHUNKS]:
        key = f"gmflow_bidir_a_g{k}{'r' if regions else ''}"
        p = trt_lookup.onnx_path(key)
        if not os.path.isfile(p) or not os.path.isfile(trt_lookup.onnx_path(key + "_m")):
            whole = trt_lookup.onnx_path(tr.BidirFlowEngine().name)
            for f in (whole, whole + ".data"):
                if os.path.isfile(f):
                    os.remove(f)
            gmt.EXPORT_WINDOW_CHUNKS = k
            gmt.EXPORT_REGION_ATTENTION = regions
            gmt.EXPORT_TOKEN_OUTER = GMFLOW_TOKEN_OUTER.get(key, ())
            gmt.EXPORT_TOKEN_POSITION = GMFLOW_TOKEN_POSITION.get(key, ())
            gmt.EXPORT_FP16_STREAM = GMFLOW_FP16_STREAM
            try:
                ensure(tr.BidirFlowEngine(), tr._BidirFlowExport(gm.flownet), rec["flownet"][:2])
            finally:
                gmt.EXPORT_WINDOW_CHUNKS = 1
                gmt.EXPORT_REGION_ATTENTION = False
                gmt.EXPORT_TOKEN_OUTER = ()
                gmt.EXPORT_TOKEN_POSITION = ()
                gmt.EXPORT_FP16_STREAM = False
            a_path, = tr.gmflow_split(os.path.join(trt_lookup.ONNX_DIR, done.pop()), a_key=key, with_b=False)
            tr.gmflow_backbone_cut(a_path, with_bone=False)
            tr.gmflow_matching_cut(a_path, key)
        done.extend((os.path.basename(p), os.path.basename(trt_lookup.onnx_path(key + "_m"))))
    ensure(tr.MetricEngine(), gm.metricnet, rec["metricnet"][:4])
    xi, tsv = rec["ifnet"][0], rec["ifnet"][1]
    ensure(tr.IFNetEngine(), tr._IFNetExport(gm.ifnet, [8, 4, 2, 1]),
           (xi, xi.new_full((1, 1, 1, 1), float(tsv))))
    ensure(tr.FusionEngine(), gm.fusionnet, rec["fusionnet"][:4])
    print(f"size-free ONNX ready in {trt_lookup.ONNX_DIR}: {len(done)} graphs, "
          f"{time.time() - t0:.0f} s", flush=True)
    ship_tidy(done)
    write_tags(stamp)


def tags_path():
    return os.path.join(trt_lookup.ONNX_DIR, "weights_tags.txt")


def read_tag(key):
    """One `key value` line of weights_tags.txt, or None."""
    try:
        with open(tags_path(), encoding="ascii") as fh:
            for line in fh:
                k, _, v = line.strip().partition(" ")
                if k == key:
                    return v
    except OSError:
        pass
    return None


def clear_graphs():
    """Before a full export: the tags file first (a run that dies midway leaves no stamp, so the next
    run starts over), then every graph and weight file of the folder."""
    d = trt_lookup.ONNX_DIR
    if not os.path.isdir(d):
        return
    if os.path.isfile(tags_path()):
        os.remove(tags_path())
    for f in sorted(os.listdir(d)):
        if f.endswith((".onnx", ".data")):
            os.remove(os.path.join(d, f))


def ship_tidy(done):
    """The shipped folder holds exactly these graphs, each weight blob once: (1) an
    .onnx this export does not produce is removed with its .data (the python route's offline
    `_b{B}` classes: the host builds those engines from `_bd8`); (2) external data files with the
    same bytes are merged into one `weights_<md5 12>.data` and every graph's `location` points at
    it (the offsets stay valid: same bytes, same layout); the RIFE family carried one 23 MB blob
    twelve times. (3) a .data no kept graph references is removed. Idempotent: a merged folder has
    nothing left to merge. The TensorRT parser resolves `location` next to the .onnx it parses."""
    import hashlib

    import onnx
    from onnx.external_data_helper import _get_all_tensors

    d = trt_lookup.ONNX_DIR
    keep = set(done)
    for f in sorted(os.listdir(d)):
        if f.endswith(".onnx") and f not in keep:
            for p in (os.path.join(d, f), os.path.join(d, f + ".data")):
                if os.path.isfile(p):
                    os.remove(p)
            print(f"removed {f}: not produced by this export", flush=True)
    models, users = {}, {}
    for f in sorted(keep):
        m = onnx.load(os.path.join(d, f), load_external_data=False)
        models[f] = m
        for t in _get_all_tensors(m):
            for kv in t.external_data:
                if kv.key == "location":
                    users.setdefault(kv.value, set()).add(f)

    def md5(p):
        h = hashlib.md5()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    groups = {}
    for loc in sorted(users):
        groups.setdefault(md5(os.path.join(d, loc)), []).append(loc)
    saved = 0
    for h, locs in groups.items():
        if len(locs) < 2:
            continue
        shared = f"weights_{h[:12]}.data"
        sp = os.path.join(d, shared)
        if not os.path.isfile(sp):
            src = next(l for l in locs if l != shared)
            tmp = sp + ".tmp"
            with open(os.path.join(d, src), "rb") as a, open(tmp, "wb") as b:
                for chunk in iter(lambda: a.read(1 << 20), b""):
                    b.write(chunk)
            os.replace(tmp, sp)
        for loc in locs:
            if loc == shared:
                continue
            for f in sorted(users[loc]):
                m = models[f]
                for t in _get_all_tensors(m):
                    for kv in t.external_data:
                        if kv.key == "location" and kv.value == loc:
                            kv.value = shared
                tmp = os.path.join(d, f + ".tmp")
                with open(tmp, "wb") as fh:
                    fh.write(m.SerializeToString())
                os.replace(tmp, os.path.join(d, f))
                users.setdefault(shared, set()).add(f)
            saved += os.path.getsize(os.path.join(d, loc))
            os.remove(os.path.join(d, loc))
            del users[loc]
        print(f"merged {len(locs)} identical weight files into {shared}", flush=True)
    for f in sorted(os.listdir(d)):
        if f.endswith(".data") and f not in users:
            saved += os.path.getsize(os.path.join(d, f))
            os.remove(os.path.join(d, f))
            print(f"removed {f}: no graph references it", flush=True)
    total = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d))
    print(f"ONNX folder {total / 1e6:.0f} MB ({saved / 1e6:.0f} MB of duplicates removed)", flush=True)


def write_tags(stamp):
    """weights_tags.txt: the weight tags every engine name carries, for a shipped tree without the
    weight files (the dist leaves the .pkl / .pth out; the ONNX carry the weights).
    The host hashes the weight files when they exist (a dev tree) and reads this file otherwise.
    `export` = the export stamp these graphs were made under (the host ignores it): the file is half
    of the engine cache stamp, so a new export empties the cache once (src/render/cache.ts)."""
    p = tags_path()
    txt = (f"w {trt_lookup.weights_tag()}\nr {trt_lookup.rife_weights_tag()}\n"
           f"rest {realesr.weights_hash()}\nexport {stamp}\n")
    with open(p + ".tmp", "w", encoding="ascii", newline="\n") as fh:
        fh.write(txt)
    os.replace(p + ".tmp", p)
    print("weights_tags.txt: " + txt.replace("\n", "  ").strip(), flush=True)


if __name__ == "__main__":
    main()
