"""Torch-free TensorRT-RTX cache naming: the engine names, the weights and TRT tags and the size-free
ONNX paths, used by onnx_export.py and trt_runtime.py (the size-free ONNX export) and by the harness
gates.

The native host (smv-live.exe) finds, builds and warms its own engines for live and offline
(smv-live-native.inl: lkSession, lkOfflineRife); it reads the same names.

Never import torch here: a name lookup must not pay the torch import (test harnesses import this
module too). Nothing here
writes to stdout. The naming and the hashes have a C++ twin in
engine/live/build_src/smv-live-native.inl: a change to a name here needs the same change there.
"""
import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__))
# The folder NAME is the documentation: everything in it is a rebuildable compilation cache (the
# app empties it only when the engine stamp changes, src/render/cache.ts), and the user reclaims the
# disk by deleting the folder whenever they want. It lives in the app's top folder (the repo root here,
# src/render/cache.ts), never under AppData.
CACHE_DIR = os.environ.get("SMV_TRT_CACHE") or os.path.join(os.path.dirname(HERE), "model_cache_safe_to_delete")

_tags = {}


def trt_tag():
    """TensorRT-RTX's serialized engine is the hardware-agnostic AOT blob (the GPU-specific kernels
    are JIT-compiled at load and cached separately), so its filename needs only the TRT version,
    no GPU name: one built engine is reusable across any RTX GPU (and survives a card swap). The
    version still rides in the name, so a TRT-RTX upgrade is a clean rebuild."""
    t = _tags.get("trt")
    if t is None:
        import tensorrt_rtx as trt
        t = _tags["trt"] = f"trt{trt.__version__}".replace(".", "_")
    return t


def _md5_files(paths):
    h = hashlib.md5()
    for p in paths:
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()[:10]


def weights_tag():
    """Fingerprint of the train_log weights, baked into every engine filename so a compiled
    engine can never outlive the weights it was exported from: swapped .pkl files change the
    tag, which is a cache miss (fresh build); engines for the old weights just stop being
    loaded. Hashes the file CONTENTS, not mtimes: a fresh unzip/copy of identical weights
    must not throw away ~6 min of builds per resolution. The ~75 MB read costs ~0.2 s once
    per process."""
    t = _tags.get("w")
    if t is None:
        wdir = os.path.join(HERE, "GMFSS_Fortuna", "train_log")
        t = _tags["w"] = "w" + _md5_files(
            os.path.join(wdir, n) for n in sorted(os.listdir(wdir)) if n.endswith(".pkl"))
    return t


def rife_weights_tag():
    """md5 of rife/flownet.pkl contents, baked into RIFE engine names so a checkpoint swap is a
    plain cache miss (parallel to weights_tag for GMFSS). Read lazily: only RIFE pays it."""
    t = _tags.get("r")
    if t is None:
        t = _tags["r"] = "r" + _md5_files([os.path.join(HERE, "rife", "flownet.pkl")])
    return t


def engine_name(name, shapes, input_names=None, dyn_batch=None):
    """Cache name of the engine the host builds for this sub net at these example shapes:
    every input's shape, H and W pinned (one engine per resolution: a dynamic-shape profile's
    execution context reserves memory for the profile MAXIMUM, 8 GB for a 960x540 live session,
    and its kernel cache grows with every size).
    An input whose batch axis is dynamic (the live RIFE timestep stack, dyn_batch = {input:
    (min, opt, max)}) writes its range as "1to8" in place of the example batch, so no session's
    first group size leaks into the name and the fixed-batch classes can never collide."""
    parts = []
    names = input_names if input_names is not None else [None] * len(shapes)
    for n, s in zip(names, shapes):
        dims = [str(int(v)) for v in s]
        if dyn_batch and n in dyn_batch:
            lo, _opt, hi = dyn_batch[n]
            dims[0] = f"{lo}to{hi}"
        parts.append("x".join(dims))
    return f"{name}_{'_'.join(parts)}_{trt_tag()}_{weights_tag()}"


# Size-free ONNX graphs: one file per live graph with every H /
# W symbolic, generated from the committed weights by scripts/export-onnx.js (npm setup and dist),
# gitignored, shipped in the release archive. An engine at a new size is then a build from this file
# (about 1 s) instead of a torch export (20 to 40 s). export_stamp() fingerprints everything an export
# reads: when it changes, onnx_export.py exports every graph again and writes the new stamp into
# weights_tags.txt, and a changed tags file makes the app / CLI empty the engine cache once
# (src/render/cache.ts), so no engine built from an older graph is reused.
ONNX_DIR = os.environ.get("SMV_ONNX_DIR") or os.path.join(HERE, "onnx")
# the packages that write or rewrite the graphs (the dynamo exporter runs on onnxscript and onnx-ir)
EXPORT_PACKAGES = ("torch", "onnx", "onnxscript", "onnx-ir")
# the host's builder settings that change an engine but not its graph (smv-live.exe lkBuild): part of
# export_stamp(), so a change here empties the engine cache once, like a new export
HOST_BUILD = "optimization level 5; aux stream caps: gmflow_bidir_b 0, metricnet 0, gmflow_backbone 0, fusionnet 2"


def onnx_path(key):
    """The size-free ONNX of a graph: key = the engine's base name (plus the export tag of a class
    whose baked arguments are not in that name), then the weights tag."""
    return os.path.join(ONNX_DIR, f"{key}_{weights_tag()}.onnx")


def export_stamp():
    """Fingerprint of everything a size-free export reads: the export and model sources (engine's own
    .py files, engine/rife, engine/GMFSS_Fortuna; CRLF read as LF, so a checkout's line endings do not
    count), the versions of EXPORT_PACKAGES, the weight files and HOST_BUILD. Any change means every
    graph is exported again (onnx_export.py). Reads no torch, so it costs well under a second."""
    from importlib import metadata

    files = [n for n in os.listdir(HERE) if n.endswith(".py")]
    for sub in ("rife", "GMFSS_Fortuna"):
        for root, dirs, names in os.walk(os.path.join(HERE, sub)):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            files += [os.path.relpath(os.path.join(root, n), HERE) for n in names if n.endswith(".py")]
    h = hashlib.md5()
    for rel in sorted(f.replace("\\", "/") for f in files):
        with open(os.path.join(HERE, rel), "rb") as f:
            h.update(rel.encode() + b"\0" + f.read().replace(b"\r\n", b"\n") + b"\0")
    for pkg in EXPORT_PACKAGES:
        try:
            v = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            v = "-"
        h.update(f"{pkg} {v}\n".encode())
    rest = _md5_files([os.path.join(HERE, "realesr-animevideov3.pth")])
    h.update(f"{weights_tag()} {rife_weights_tag()} {rest}\n".encode())
    h.update(f"{HOST_BUILD}\n".encode())
    return "e" + h.hexdigest()[:10]


def rife_scale_tag(flow_scale):
    """rife_backend.RIFE.set_scale's list as RifeIFNetEngine bakes it into the engine name."""
    return "-".join(f"{v:g}" for v in (16 / flow_scale, 8 / flow_scale, 4 / flow_scale,
                                         2 / flow_scale, 1 / flow_scale))


def rife_block0_base(scale, whash=None):
    """Base name of DRBA's block0 flow engine (trt_runtime.RifeBlock0Engine): the rife
    checkpoint fingerprint plus the baked block0 scale (scale_list[0] = 16 / flow scale)."""
    return f"rife_block0_{whash or rife_weights_tag()}_{float(scale):g}"
