"""One-command render verification matrix for the SMV engine.

Codifies the manual checks every change used to be verified with: real renders on
a lossless 854x480 copy of samples/test.mp4 made at start (small_sample; 25 decoded frames,
so 2x on-grid = 49 frames, 3x = 73, 5x = 121,
--fps 60 = 63, --no-interp = 25), a synthetic VFR clip (duration preservation), the
.part rename-on-success hygiene, and - when their runtimes are installed - the RTX HDR
metadata boxes, Dolby Vision configuration record and HDR10+ dynamic-metadata SEI.

    python scripts\\smoke.py            # quick set (first run per resolution builds engines)
    python scripts\\smoke.py --full     # + 5x, av1/vvc, HDR/DV/HDR10+, the live cases
                                        # (--trt is accepted and ignored: every render is TensorRT)

Pure stdlib, any python 3 (a dev tool; the product ships no python). Runs the render CLI
exactly like the GUI does: dist/render/cli.js under the Electron binary in node mode (run
`npx tsc` first). Cases whose runtime is not installed (RTX feature DLLs, dovi_tool,
hdr10plus_tool, engine/live) report SKIP, not FAIL. The live cases drive
engine/live/smv-live.exe against its --testsrc window (parked + --no-hud, nothing appears on
screen) and assert the handshake, the zero-copy transports and the stats line shape. Exits
nonzero if any case FAILs. Assertions are structural (frame counts, duration, metadata
presence): TensorRT-RTX output is not run-to-run bit-stable, so there is no md5 case.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.join(ROOT, "engine")
CLI = os.path.join(ROOT, "dist", "render", "cli.js")
ELECTRON = os.path.join(ROOT, "node_modules", "electron", "dist", "electron.exe")
NODE = [ELECTRON] if os.path.isfile(ELECTRON) else ["node"]   # the app's own binary in node mode
SAMPLE = os.path.join(ROOT, "samples", "test.mp4")
FFMPEG = os.path.join(ENGINE, "bin", "ffmpeg.exe")
FFPROBE = os.path.join(ENGINE, "bin", "ffprobe.exe")
if not os.path.isfile(FFMPEG):
    FFMPEG, FFPROBE = "ffmpeg", "ffprobe"

results = []


def probe_json(path, *args):
    out = subprocess.check_output([FFPROBE, "-v", "error", *args, "-of", "json", path], text=True)
    return json.loads(out)


def frames(path):
    j = probe_json(path, "-count_frames", "-select_streams", "v:0",
                   "-show_entries", "stream=nb_read_frames")
    return int(j["streams"][0]["nb_read_frames"])


def duration(path):
    j = probe_json(path, "-show_entries", "format=duration")
    return float(j["format"]["duration"])


def stream_side_data(path):
    j = probe_json(path, "-select_streams", "v:0", "-show_streams")
    return [d.get("side_data_type", "") for d in j["streams"][0].get("side_data_list", [])]


def first_frame_side_data(path):
    j = probe_json(path, "-select_streams", "v:0", "-read_intervals", "%+#1", "-show_frames")
    fr = j.get("frames") or [{}]
    return [d.get("side_data_type", "") for d in fr[0].get("side_data_list", [])]


def render(inp, out, *args, multi="2"):
    """Run one render through the CLI the app runs; returns (returncode, stderr_text)."""
    env = dict(os.environ, ELECTRON_RUN_AS_NODE="1", SMV_ENGINE_DIR=ENGINE)
    p = subprocess.run(NODE + [CLI, inp, multi, out, *args], cwd=ENGINE, env=env, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    return p.returncode, (p.stderr or "") + (p.stdout or "")


def offline_host_quit():
    """End the resident offline host a CLI render leaves behind (cli.js names its pipe from
    the exe path when the app's SMV_OFFLINE_HOST_PIPE is not set); no host = nothing to do."""
    import hashlib
    exe = os.path.join(ENGINE, "live", "smv-live.exe")
    name = "smv-offline-" + hashlib.md5(os.path.normcase(os.path.abspath(exe)).encode("utf-8", "replace")).hexdigest()[:8]
    try:
        fd = os.open("\\\\.\\pipe\\" + name, os.O_RDWR | os.O_BINARY)
    except OSError:
        return
    try:
        os.write(fd, b"quit\n")
    except OSError:
        pass
    finally:
        os.close(fd)


def case(name):
    def deco(fn):
        fn.smoke_name = name
        return fn
    return deco


def run_case(fn, tmp, trt):
    t0 = time.time()
    try:
        verdict = fn(tmp, trt) or "PASS"
    except AssertionError as e:
        verdict = "FAIL: " + str(e)
    except Exception as e:  # noqa: BLE001 - a crashed case is a failed case
        verdict = "FAIL: " + repr(e)[:200]
    results.append((fn.smoke_name, verdict, time.time() - t0))
    print(f"  {verdict.split(':')[0]:<5} {fn.smoke_name}  ({time.time() - t0:.0f}s)"
          + ("" if verdict.startswith(("PASS", "SKIP")) else "\n        " + verdict))


def expect_part_promoted(out):
    b, e = os.path.splitext(out)
    assert os.path.isfile(out), "output missing: " + out
    assert not os.path.exists(b + ".part" + e), ".part remnant left behind"


# --------------------------------------------------------------------------- quick set
@case("2x on-grid -> 49 frames")
def c_2x(tmp, trt):
    out = os.path.join(tmp, "s_2x.mp4")
    rc, log = render(SAMPLE, out)
    assert rc == 0, "engine exit " + str(rc)
    expect_part_promoted(out)
    assert frames(out) == 49, f"frames {frames(out)} != 49"


@case("--fps 60 -> 63 frames")
def c_fps60(tmp, trt):
    out = os.path.join(tmp, "s_60.mp4")
    rc, log = render(SAMPLE, out, "--fps", "60")
    assert rc == 0, "engine exit " + str(rc)
    expect_part_promoted(out)
    assert frames(out) == 63, f"frames {frames(out)} != 63"


@case("--no-interp -> 25 frames")
def c_noint(tmp, trt):
    out = os.path.join(tmp, "s_ni.mp4")
    rc, log = render(SAMPLE, out, "--no-interp")
    assert rc == 0, "engine exit " + str(rc)
    expect_part_promoted(out)
    assert frames(out) == 25, f"frames {frames(out)} != 25"


@case("--lsfg (Frame Blend) 2x -> 49 frames")
def c_blend(tmp, trt):
    # the flow-warp Frame Blend path: RIFE's flow at the blend scale, full-resolution
    # warps
    out = os.path.join(tmp, "s_blend.mp4")
    rc, err = render(SAMPLE, out, "--lsfg")
    assert rc == 0, "engine exit " + str(rc)
    expect_part_promoted(out)
    assert frames(out) == 49, f"frames {frames(out)} != 49"
    assert "Using the Frame Blend backend" in err, "Frame Blend backend line missing from stderr"


@case("--nvof (NVIDIA Optical Flow) 2x -> 49 frames, native host, identical pair held")
def c_nvof(tmp, trt):
    # the NVIDIA Optical Flow model: native host only, torch-free, no engines;
    # the sample's frames 0 and 1 are byte identical, so one pair must pass through held
    out = os.path.join(tmp, "s_nvof.mp4")
    rc, err = render(SAMPLE, out, "--nvof")
    offline_host_quit()
    assert rc == 0, "engine exit " + str(rc)
    expect_part_promoted(out)
    assert frames(out) == 49, f"frames {frames(out)} != 49"
    assert "Using the NVIDIA Optical Flow (direct) backend" in err, "nvof backend line missing"
    assert "model nvof" in err, "the native host did not run the nvof model"
    assert "static pairs held: 1" in err, "the identical head pair was not held"


@case("--flow-scale is refused (the Flow scale control is gone)")
def c_rife_flow(tmp, trt):
    # the Flow scale control ran the motion estimation smaller; it was removed because GMFSS at
    # 25 % wobbled static frames and lost small fast objects at 50 %. The flag must be refused,
    # never silently ignored.
    out = os.path.join(tmp, "s_rife_flow.mp4")
    rc, err = render(SAMPLE, out, "--rife", "--flow-scale", "0.5")
    assert rc != 0, "a --flow-scale render must fail"
    assert "unrecognized arguments: --flow-scale" in err, "the refusal line is missing"
    assert not os.path.exists(out), "a refused render must write no output"


@case("VFR source -> constant-avg decode, duration preserved")
def c_vfr(tmp, trt):
    vfr = os.path.join(tmp, "vfr.mp4")
    # Irregular frame drops with original timestamps: r_frame_rate stays 24000/1001 while the
    # average drops to ~12.5 fps, the VFR signature the engine must detect.
    p = subprocess.run([FFMPEG, "-v", "error", "-y", "-i", SAMPLE,
                        "-vf", "select='not(mod(n,3))+not(mod(n,4))-not(mod(n,12))'",
                        "-fps_mode", "vfr", "-c:v", "hevc_nvenc", "-cq", "18", vfr],
                       capture_output=True, text=True)
    if p.returncode != 0:
        return "SKIP (no NVENC to synthesize the VFR clip)"
    out = os.path.join(tmp, "s_vfr.mp4")
    rc, log = render(vfr, out)
    assert rc == 0, "engine exit " + str(rc)
    assert "VFR source" in log, "VFR notice missing from engine output"
    src_d, out_d = duration(vfr), duration(out)
    # On-grid 2x is up to ~2 output frames shorter/longer at the tail; 15% covers that on a 1s clip.
    assert abs(out_d - src_d) / src_d < 0.15, f"duration {out_d:.2f}s vs source {src_d:.2f}s"


# --------------------------------------------------------------------------- --full extras
@case("rife 2x -> 49 frames through the native offline host")
def c_rife_native(tmp, trt):
    # the plain RIFE render runs inside smv-live.exe --offline
    out = os.path.join(tmp, "s_rife_native.mp4")
    rc, err = render(SAMPLE, out, "--rife")
    assert rc == 0, "engine exit " + str(rc)
    assert frames(out) == 49, f"frames {frames(out)} != 49"
    assert "done 24 pairs (native host)" in err, "native host not used"
    # the resident offline host: the second render of the same size and
    # multiplier must land on the host the first one left behind, engines reused
    assert "resident host on" in err, "first render did not use the resident offline host"
    rc, err = render(SAMPLE, out, "--rife")
    assert rc == 0, "engine exit " + str(rc) + " (second render)"
    assert frames(out) == 49, f"frames {frames(out)} != 49 (second render)"
    assert "resident engines reused" in err, "second render did not reuse the resident engines"


@case("5x on-grid -> 121 frames")
def c_5x(tmp, trt):
    out = os.path.join(tmp, "s_5x.mp4")
    rc, _ = render(SAMPLE, out, multi="5")
    assert rc == 0, "engine exit " + str(rc)
    assert frames(out) == 121, f"frames {frames(out)} != 121"


@case("--codec av1 2x -> 49 frames")
def c_av1(tmp, trt):
    out = os.path.join(tmp, "s_av1.mp4")
    rc, _ = render(SAMPLE, out, "--codec", "av1")
    assert rc == 0, "engine exit " + str(rc)
    assert frames(out) == 49, f"frames {frames(out)} != 49"


def slices_per_frame(path):
    """Slice segments per picture of an HEVC file (split-frame encoding writes one per strip)."""
    r = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "debug", "-i", path, "-frames:v", "2", "-c", "copy",
                        "-bsf:v", "trace_headers", "-f", "null", "-"], capture_output=True, text=True, errors="replace")
    firsts = len(re.findall(r"first_slice_segment_in_pic_flag\s+\d+\s*=\s*1", r.stderr))
    total = len(re.findall(r"first_slice_segment_in_pic_flag\s+\d+\s*=\s*[01]", r.stderr))
    return total / firsts if firsts else 0


@case("--enc-speed is refused (the Encoder speed selector is gone); --rife 2x -> 49 frames, ONE slice per frame (split-frame encoding stays off)")
def c_encsplit(tmp, trt):
    out = os.path.join(tmp, "s_encsplit.mp4")
    rc, err = render(SAMPLE, out, "--rife", "--enc-speed", "fast")
    assert rc != 0, "an --enc-speed render must fail"
    assert "unrecognized arguments: --enc-speed" in err, "the refusal line is missing"
    rc, err = render(SAMPLE, out, "--rife")
    assert rc == 0, "engine exit " + str(rc)
    assert frames(out) == 49, f"frames {frames(out)} != 49"
    assert "split frame encoding" not in err, "a split-frame note appeared without SMV_NVENC_SPLIT"
    spf = slices_per_frame(out)
    assert spf == 1, f"{spf} slices per frame: split-frame encoding is on (it leaves a seam line)"


@case("--codec vvc --no-interp -> 25 frames")
def c_vvc(tmp, trt):
    out = os.path.join(tmp, "s_vvc.mp4")
    rc, _ = render(SAMPLE, out, "--codec", "vvc", "--no-interp")
    assert rc == 0, "engine exit " + str(rc)
    assert frames(out) == 25, f"frames {frames(out)} != 25"


def _rtx_ready():
    d = os.path.join(ENGINE, "rtxvideo")
    return all(os.path.isfile(os.path.join(d, n))
               for n in ("rtxvideo_cuda.dll", "nvngx_truehdr.dll"))


@case("--rtx-hdr --no-interp -> PQ + mastering/CLL boxes")
def c_hdr(tmp, trt):
    if not _rtx_ready():
        return "SKIP (RTX runtime not installed)"
    out = os.path.join(tmp, "s_hdr.mp4")
    rc, log = render(SAMPLE, out, "--rtx-hdr", "--no-interp")
    assert rc == 0, "engine exit " + str(rc)
    assert frames(out) == 25, f"frames {frames(out)} != 25"
    j = probe_json(out, "-select_streams", "v:0", "-show_entries", "stream=color_transfer")
    assert j["streams"][0].get("color_transfer") == "smpte2084", "not PQ"
    sd = " ".join(stream_side_data(out))
    assert "Mastering display metadata" in sd, "mdcv box missing"
    assert "Content light level metadata" in sd, "clli box missing"


@case("--rtx-hdr --dv -> DOVI configuration record")
def c_dv(tmp, trt):
    if not _rtx_ready():
        return "SKIP (RTX runtime not installed)"
    if not os.path.isfile(os.path.join(ENGINE, "dvtools", "dovi_tool.exe")):
        return "SKIP (dovi_tool not installed)"
    out = os.path.join(tmp, "s_dv.mp4")
    rc, log = render(SAMPLE, out, "--rtx-hdr", "--dv", "--no-interp")
    assert rc == 0, "engine exit " + str(rc)
    assert frames(out) == 25, f"frames {frames(out)} != 25"
    assert any("DOVI" in s for s in stream_side_data(out)), "DOVI configuration record missing"


@case("--rtx-hdr(rtx) --hdr10plus -> ST 2094-40 frame SEI")
def c_hp(tmp, trt):
    if not _rtx_ready():
        return "SKIP (RTX runtime not installed)"
    if not os.path.isfile(os.path.join(ENGINE, "hptools", "hdr10plus_tool.exe")):
        return "SKIP (hdr10plus_tool not installed)"
    out = os.path.join(tmp, "s_hp.mp4")
    # rtx colour mode on purpose: its magnitude-ratio math is the path that once NaN'd on
    # zero-chroma (black) pixels under fp16 autocast and crashed the HDR10+ histogram.
    rc, log = render(SAMPLE, out, "--rtx-hdr", "--hdr-color", "rtx", "--hdr-saturation", "100",
                     "--hdr10plus", "--no-interp")
    assert rc == 0, "engine exit " + str(rc)
    assert "HDR10+: dynamic metadata written" in log, "export fell back: " + log[-300:]
    assert frames(out) == 25, f"frames {frames(out)} != 25"
    fsd = " ".join(first_frame_side_data(out))
    assert "SMPTE2094-40" in fsd or "HDR10+" in fsd, "HDR10+ dynamic metadata SEI missing"


def _dlssnr_ready():
    d = os.path.join(ENGINE, "dlssnr")
    return all(os.path.isfile(os.path.join(d, n))
               for n in ("dlssnr.exe", "nvngx.dll", "nvngx_dlssnr.dll"))


@case("--dlssg --dlssnr 2x -> 49 frames through the automatic two-pass split")
def c_dlss_twopass(tmp, trt):
    # DLSS-G plus a second NGX user (here DLSS 5) splits into pass 1 (frame generation into an
    # intermediate) and pass 2 (the per-frame passes over it); the intermediate is spent on
    # success. The relayed stderr carries both passes' backend lines.
    if not os.path.isfile(os.path.join(ENGINE, "dlssg", "dlssg2f.exe")):
        return "SKIP (DLSS-G host not present)"
    if not _dlssnr_ready():
        return "SKIP (DLSS 5 runtime not installed)"
    out = os.path.join(tmp, "s_dlss2p.mp4")
    rc, err = render(SAMPLE, out, "--dlssg", "--dlssnr")
    assert rc == 0, "engine exit " + str(rc) + ": " + err[-300:]
    expect_part_promoted(out)
    assert frames(out) == 49, f"frames {frames(out)} != 49"
    assert "Using the DLSS Frame Generation backend" in err, "pass 1 DLSS-G line missing from stderr"
    assert "DLSS 5 Neural Rendering ready" in err, "pass 2 DLSS 5 line missing from stderr"
    left = [n for n in os.listdir(tmp) if ".dlss-interp-" in n]
    assert not left, "intermediate not spent: " + ", ".join(left)


LIVE_EXE = os.path.join(ENGINE, "live", "smv-live.exe")


def _live_ticks(err):
    """The stats lines that actually captured frames (over 2 fps: the 60 fps testsrc reads
    30 or more, the static-hold refresh reads exactly 1.0). A tick at or near 0 captured fps
    is the testsrc window not being delivered by WGC (its start can come seconds late, and a
    user alt-tabbing at the PC while the case runs covers it, e.g. one frame in the first tick
    and none after); such ticks say nothing about the route and are skipped."""
    import re
    out = []
    for ln in err.splitlines():
        m = re.match(r"live: ([\d.]+) captured fps", ln)
        if m and float(m.group(1)) > 2.0:
            out.append(ln)
    return out


def _live_session(backend, gen, seconds, ticks=3):
    """testsrc + a parked live session (park bypasses the alt-tab pause, --no-hud keeps the
    screen clean); returns the live process's stderr. Structural checks only, like the rest
    of this file: handshake, transport mode, stats lines. stderr goes to a temp FILE, never
    subprocess.PIPE: an undrained pipe caps at ~4KB, so a 60s session's stats stall and the
    last line comes back torn mid-write (the rife case failed on '(ratio 3' exactly there).
    `ticks` counts LIVE ticks (captured fps > 0, _live_ticks): starved ticks are waited out
    within `seconds`, so a late WGC start or a user at the PC cannot fail the case by itself;
    a capture that never delivers still fails on the cases' own "no live tick" assertion."""
    live_dir = os.path.join(ENGINE, "live")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    ts = subprocess.Popen([LIVE_EXE, "--testsrc", "16"], cwd=live_dir,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(2)
        # --no-adapt: these cases assert a FIXED ratio; adaptive smoothness would (correctly)
        # step the 60fps testsrc down on a loaded GPU and break the assertion
        fd, epath = tempfile.mkstemp(suffix=".smvlive.log")
        try:
            with os.fdopen(fd, "r+", encoding="utf-8", errors="replace") as ef:
                lv = subprocess.Popen([LIVE_EXE, "--live", "TestSrc", "--backend", backend,
                                       "--gen", str(gen), "--park", "--no-hud", "--no-adapt"],
                                      cwd=live_dir, stdout=subprocess.DEVNULL,
                                      stderr=ef, creationflags=flags)
                try:
                    # `seconds` is a CAP, not a sleep: startup (engine load, JIT cache) eats
                    # ~20 s before the first stats tick and the first tick is the warm-up one
                    # (ratio 2.9, a drop), so we stop as soon as `ticks` stats lines exist.
                    # Measured: a flat 20 s sleep saw only the warm-up tick and
                    # failed the rife ratio; ticks 2..20 of a 60 s run were all 3.00 +- 0.05.
                    deadline = time.time() + seconds
                    while time.time() < deadline and lv.poll() is None:
                        time.sleep(0.5)
                        with open(epath, "r", encoding="utf-8", errors="replace") as rf:
                            if len(_live_ticks(rf.read())) >= ticks:
                                break
                finally:
                    lv.kill()
                    lv.wait()
                ef.seek(0)
                err = ef.read()
        finally:
            try:
                os.remove(epath)
            except OSError:
                pass
    finally:
        ts.kill()
    return err


@case("live: echo transport at capture rate")
def c_live_echo(tmp, trt):
    if not os.path.isfile(LIVE_EXE):
        return "SKIP (engine/live/smv-live.exe missing)"
    import re
    err = _live_session("echo", 1, 90, ticks=2)
    assert "LIVE READY" in err, "no LIVE READY handshake: " + err[-300:]
    # the native host is the only live route: its no-engine mode, answered by the
    # host's own lookup
    assert "engine=none (effects only)" in err, "echo not on the native host: " + err[-300:]
    assert "no python process" in err, "the handoff started python: " + err[-300:]
    stats = _live_ticks(err)
    assert stats, "no live tick (WGC delivered nothing in 90 s): " + err[-300:]
    m = re.search(r"live: ([\d.]+) captured fps -> ([\d.]+) presented", stats[-1])
    assert m and float(m.group(1)) > 5, "capture rate too low: " + stats[-1]


@case("live: rife 3x -> ratio 3.00 + plausible latency")
def c_live_rife(tmp, trt):
    if not os.path.isfile(LIVE_EXE):
        return "SKIP (engine/live/smv-live.exe missing)"
    import re
    err = _live_session("rife", 2, 90, ticks=3)
    assert "LIVE READY" in err, "no LIVE READY handshake: " + err[-300:]
    assert "native host ready" in err, "rife not on the native host: " + err[-300:]
    stats = _live_ticks(err)
    assert stats, "no live tick (model load too slow, or WGC delivered nothing in 90 s): " + err[-300:]
    last = stats[-1]
    m = re.search(r"\(ratio ([\d.]+),.*latency ~(\d+)ms", last)
    assert m, "stats format changed: " + last
    assert abs(float(m.group(1)) - 3.0) < 0.15, "ratio off 3.00: " + last
    assert 0 < int(m.group(2)) < 500, "latency implausible: " + last


QUICK = [c_2x, c_fps60, c_noint, c_blend, c_nvof, c_rife_flow, c_vfr, c_live_echo]
FULL = [c_rife_native, c_5x, c_av1, c_encsplit, c_vvc, c_hdr, c_dv, c_hp, c_dlss_twopass,
        c_live_rife]


def small_sample(tmp):
    """samples/test.mp4 (1920x1080) scaled to 854x480, NVENC lossless in MP4 (the source's
    timestamps, so the rate probe and the VFR case read it like the original), audio copied: every case costs
    about a quarter of the pixels while the frame count (25), the rate, the audio track and the
    byte-identical first pair (the static-hold cases) stay. 480p is the floor: TensorRT-RTX
    below 480p needs the user's confirmation first (a GPU TDR once followed such runs)."""
    out = os.path.join(tmp, "test_480.mp4")
    subprocess.run([FFMPEG, "-v", "error", "-y", "-i", SAMPLE, "-map", "0", "-vf", "scale=854:480",
                    "-c:v", "hevc_nvenc", "-tune", "lossless", "-pix_fmt", "yuv420p",
                    "-c:a", "copy", out], check=True)   # the bundled LGPL ffmpeg has no libx264
    return out


def main():
    global SAMPLE
    ap = argparse.ArgumentParser(description="SMV engine smoke tests (real renders)")
    ap.add_argument("--full", action="store_true",
                    help="also run 5x, av1/vvc, HDR, DV, HDR10+, the DLSS two-pass split")
    ap.add_argument("--trt", action="store_true", help="ignored (every render runs on TensorRT-RTX)")
    ap.add_argument("--keep", action="store_true", help="keep the rendered outputs")
    args = ap.parse_args()
    assert os.path.isfile(SAMPLE), "samples/test.mp4 missing"
    assert os.path.isfile(CLI), "dist/render/cli.js missing (run npx tsc)"
    tmp = tempfile.mkdtemp(prefix="smv-smoke-")
    t0 = time.time()
    SAMPLE = small_sample(tmp)
    print(f"smoke: sample={SAMPLE}\n       outputs={tmp}  cli={' '.join(NODE)} {CLI}")
    for fn in QUICK + (FULL if args.full else []):
        run_case(fn, tmp, True)
    offline_host_quit()     # the renders above leave the resident offline host idle; end it
    fails = [r for r in results if r[1].startswith("FAIL")]
    skips = [r for r in results if r[1].startswith("SKIP")]
    print(f"\n{len(results) - len(fails) - len(skips)} passed, {len(skips)} skipped, "
          f"{len(fails)} failed  ({time.time() - t0:.0f}s)")
    if not args.keep and not fails:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    elif fails:
        print("outputs kept for inspection:", tmp)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
