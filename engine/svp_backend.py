"""SVP interpolation backend: generates and services the standalone VapourSynth host process.

A sibling process on the same bundled runtime imports the pinned `vapoursynth` wheel (svpflow is
a VS API3 plugin; re-verify a render before any version bump), decodes the source with the
bundled ffmpeg, feeds svpflow through a sliding-window frame cache (no VS source plugin needed)
and writes y4m to stdout for the engine's downstream conversion ffmpeg. SVP 4 only provides the
two plugins64 svpflow DLLs.

The svpflow parameter strings are a MAX-QUALITY OFFLINE profile (export quality regardless of
processing cost), each derived from SVP's own generate.js mappings: frames interpolation mode
"Uniform (max fluidity)" -> scene.limits {m1:0,m2:0} (every pair interpolated); scene changes
repeat-frame OFF -> scene.blend true (cuts blend instead of holding - the only non-repeat option
svpflow offers); shader "13. Standard" -> algo 13 (svpflow's default); artifacts masking
Disabled -> no mask.area; MV precision: half-pixel = svpflow's maximum (pel 2 default) +
search.type 2; MV grid smallest "6 px" -> block {w:8} (finest vectors); search radius largest ->
coarse.distance -12; wide search strongest -> coarse.bad {sad:2000} (range stays svpflow's
exhaustive default); decrease grid step -> refine [{thsad:250}] (generate.js supports one
level); NVOF variant: accuracy high = default quality (the nvof.q key only exists to LOWER it)
+ "4 px" vector grid. The ONE deviation: the rate is ours (a relative num/den so any source fps
multiplies right; a playback profile's fixed Hz target does not apply offline).

CAUTION (bisected offline, this hosting): block:{w:32} degenerates SmoothFps into holding pairs
as repeats instead of tweens (~20% alone, worse combined with other search knobs); block w:8 and
algo 2 are verified clean. Never reintroduce block:{w:32} offline.
"""
import os
import sys
import tempfile
import threading

NO_WINDOW = 0x08000000


def gpuid():
    """SVP's OpenCL device id for svpflow GPU rendering, read from the user's own SVP settings
    (frc.cfg 'gpu': 11 = first NVIDIA, 21 = first Intel in SVP's numbering). SMV_SVP_GPUID
    overrides; falls back to 11 (the value SVP uses for the primary NVIDIA GPU)."""
    if os.environ.get("SMV_SVP_GPUID"):
        return int(os.environ["SMV_SVP_GPUID"])
    try:
        import re as _re
        cfg = os.path.join(os.environ.get("APPDATA", ""), "SVP4", "settings", "frc.cfg")
        with open(cfg, encoding="utf-8") as _f:
            m = _re.search(r'"gpu"\s*:\s*(\d+)\s*,', _f.read())
        if m:
            return int(m.group(1))
    except Exception:  # noqa: BLE001 - missing/odd SVP settings: use the NVIDIA default
        pass
    return 11


def write_host_script(*, inp, w, h, nb, fps_num, fps_den, vfr_dec,
                      rate_num, rate_den, nvof, svp_dir, ffmpeg, resume_out_base):
    """Write the VapourSynth host program and return its path (see the module docstring for the
    profile it encodes). `resume_out_base` > 0 trims the host to start after the banked frames."""
    if nb <= 0:
        sys.exit("SVP backend: the source frame count is unknown (no nb_frames and no container "
                 "duration), and the VapourSynth host needs a fixed clip length")
    dev = gpuid()
    # SVP shader for SmoothFps rendering; 13 "Standard" is svpflow's default. SMV_SVP_ALGO is a
    # debug/compare override (env-only). Note SmoothFps_NVOF ignores the "Complicated" shaders
    # (>=21): they only take effect on the classic path.
    algo = int(os.environ.get("SMV_SVP_ALGO", "13"))
    # SVP itself hosts NVOF with a fixed 8 VS threads (the OFA hardware does the vector search);
    # the classic path gets cpu+1 for the CPU block-matching.
    threads = 8 if nvof else (os.cpu_count() or 8) + 1
    # Cache window for the host's frame feeder: comfortably above VS's concurrent request span,
    # byte-capped for huge frames (a 16-bit 8K frame is ~100 MB) with the concurrency floor
    # winning so requests can never outrun the window.
    fsz = w * h * 3   # yuv420p16le bytes per frame: (w*h*3//2)*2
    keep = max(threads + 8, min(2 * threads + 8, int(2.5e9 // fsz)))
    plug = os.path.join(svp_dir, "plugins64")
    if nvof:
        # NVOF variant (SmoothFps_NVOF): the vectors come from the NVIDIA Optical Flow hardware,
        # fed a vector clip sized as generate.js does for nvof_grid "4 px" - the densest setting
        # (vec clip = w//4*4 x h//4*4 P8, near full res). 4 is also the floor of generate.js's
        # small-source shrink loop, so no shrink is needed.
        grid = 4
        interp = f"""vec8    = clip.resize.Bicubic(clip.width//{grid}*4, clip.height//{grid}*4,
                             src_width=clip.width-(clip.width % {grid}),
                             src_height=clip.height-(clip.height % {grid}), format=vs.YUV420P8)
smooth  = core.svp2.SmoothFps_NVOF(clip, smoothfps_params, vec_src=vec8, src=clip, fps=src_fps)"""
    else:
        # Analysis runs on an 8-bit clip (svp1's input, as in SVP's own scripts); SmoothFps
        # renders the 16-bit clip, so the tween precision is unaffected by the vector search.
        interp = """clip8   = clip.resize.Point(format=vs.YUV420P8)
sup     = core.svp1.Super(clip8, super_params)
vectors = core.svp1.Analyse(sup["clip"], sup["data"], clip8, analyse_params)
smooth  = core.svp2.SmoothFps(clip, sup["clip"], sup["data"], vectors["clip"], vectors["data"],
                              smoothfps_params, src=clip, fps=src_fps)"""
    # NVOF needs only svpflow2 (SVP's own generated script skips svpflow1 there too).
    plugin_loads = f'core.std.LoadPlugin(r"{plug}\\svpflow2_vs.dll")' if nvof else (
        f'core.std.LoadPlugin(r"{plug}\\svpflow1_vs.dll")\n'
        f'core.std.LoadPlugin(r"{plug}\\svpflow2_vs.dll")')
    # Crash-resume: the banked prefix already holds the first resume_out_base output frames, so
    # the host starts emitting at that index (VS renders on demand - the skipped output frames
    # are never computed; the feeder just decodes past the unused source prefix once).
    resume_trim = (f"smooth = smooth.std.Trim(first={resume_out_base})\n"
                   if resume_out_base else "")
    script = f"""# Generated by SmoothMyVideo (--svp{'-nvof' if nvof else ''}); max-quality svpflow profile, see engine/svp_backend.py.
# Standalone VapourSynth host: bundled ffmpeg decode -> sliding-window frame cache -> svpflow
# -> y4m on stdout.
import subprocess, sys, threading
import numpy as np
import vapoursynth as vs

W, H, N = {w}, {h}, {nb}
YSZ, CSZ = W * H, (W // 2) * (H // 2)
FSZ = (YSZ + 2 * CSZ) * 2   # yuv420p16le: 2 bytes per sample
KEEP = {keep}    # cache window; comfortably above VS's concurrent request span
NO_WINDOW = 0x08000000

core = vs.core
core.num_threads = {threads}
{plugin_loads}

# The source frames arrive as a sequential 16-bit yuv420 pipe (any source depth upshifts
# losslessly; svpflow then renders SmoothFps at 16-bit precision so tween blends never band -
# only the analysis/vector clips drop to 8-bit, exactly like SVP's own scripts). VS pulls
# frames on demand (mildly out of order across worker threads), so serve them from a sliding
# dict guarded by one lock: a request beyond what is read so far decodes up to it, EOF short
# of N repeats the last real frame (the clip length is the container's own count; drift is a
# frame or two at most).
_dec = subprocess.Popen([{ffmpeg!r}, "-v", "error", "-i", {inp!r}] + {vfr_dec!r}
                        + ["-f", "rawvideo", "-pix_fmt", "yuv420p16le", "-"],
                        stdout=subprocess.PIPE, creationflags=NO_WINDOW)
_buf, _next, _eof, _last = {{}}, 0, False, None
_lock = threading.Lock()

def _read_exact():
    data = b""
    while len(data) < FSZ:
        c = _dec.stdout.read(FSZ - len(data))
        if not c:
            return None
        data += c
    return data

def _fill(n, f):
    global _next, _eof, _last
    with _lock:
        while n >= _next and not _eof:
            data = _read_exact()
            if data is None:
                _eof = True
                if _next < N:
                    sys.stderr.write(f"svp host: decode ended at frame {{_next}}/{{N}}; "
                                     "repeating the last frame\\n"); sys.stderr.flush()
                break
            _buf[_next] = (np.frombuffer(data, np.uint16, YSZ).reshape(H, W),
                           np.frombuffer(data, np.uint16, CSZ, YSZ * 2).reshape(H // 2, W // 2),
                           np.frombuffer(data, np.uint16, CSZ, (YSZ + CSZ) * 2).reshape(H // 2, W // 2))
            _last = _buf[_next]
            _next += 1
            for k in [k for k in _buf if k < _next - KEEP]:
                del _buf[k]
        planes = _buf.get(n) if n in _buf else (_last if n >= _next else None)
        if planes is None:
            raise RuntimeError(f"svp host: frame {{n}} left the cache window (KEEP={{KEEP}})")
    fout = f.copy()
    for p in range(3):
        np.asarray(fout[p])[:] = planes[p]
    return fout

_blank = core.std.BlankClip(width=W, height=H, format=vs.YUV420P16, length=N,
                            fpsnum={fps_num}, fpsden={fps_den})
clip = _blank.std.ModifyFrame(clips=_blank, selector=_fill)
super_params     = "{{scale:{{up:0}},gpu:1,rc:true}}"
analyse_params   = "{{block:{{w:8}},main:{{search:{{type:2,coarse:{{distance:-12,bad:{{sad:2000}}}}}}}},refine:[{{thsad:250}}]}}"
smoothfps_params = "{{gpuid:{dev},gpu_qn:2,rate:{{num:{rate_num},den:{rate_den}}},algo:{algo},scene:{{blend:true,limits:{{m1:0,m2:0}}}}}}"
src_fps = {fps_num} / {fps_den}
{interp}
{resume_trim}smooth.output(sys.stdout.buffer, y4m=True)
_dec.kill()
"""
    path = os.path.join(tempfile.gettempdir(), f"smv_svp_{os.getpid()}.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(script)
    return path


def start_stderr_pump(proc):
    """Forward the host's stderr on a daemon thread, dropping VapourSynth's "svpflow*.dll is
    using API3 which is deprecated" warning: the user can do nothing about it (the pinned
    vapoursynth wheel in requirements.txt is what manages that risk). The VS core prints it
    straight to fd 2, so it cannot be intercepted inside the host itself."""
    def _pump():
        for line in iter(proc.stderr.readline, b""):
            if b"is using API3 which is deprecated" in line:
                continue
            sys.stderr.buffer.write(line)
            sys.stderr.flush()
    threading.Thread(target=_pump, daemon=True).start()
