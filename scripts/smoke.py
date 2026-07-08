"""One-command render verification matrix for the SMV engine.

Codifies the manual checks every change used to be verified with: real renders on
samples/test.mp4 (25 decoded frames, so 2x on-grid = 49 frames, 3x = 73, 5x = 121,
--fps 60 = 63, --no-interp = 25), a synthetic VFR clip (duration preservation), the
.part rename-on-success hygiene, and - when their runtimes are installed - the RTX HDR
metadata boxes, Dolby Vision configuration record and HDR10+ dynamic-metadata SEI.

    engine\\runtime\\python.exe scripts\\smoke.py            # quick set (~2 min, eager GMFSS)
    engine\\runtime\\python.exe scripts\\smoke.py --full     # + 3x/5x, av1/vvc, HDR/DV/HDR10+
    engine\\runtime\\python.exe scripts\\smoke.py --trt      # default TRT path instead of --no-trt
                                                             # (first run per resolution builds engines)

Pure stdlib; runs the engine CLI exactly like the GUI does. Cases whose runtime is not
installed (RTX feature DLLs, dovi_tool, hdr10plus_tool) report SKIP, not FAIL. Exits
nonzero if any case FAILs. Since 2026-07-09 the GMFSS path is fully deterministic
(fixed-point softsplat + cudnn deterministic mode), so identical runs produce
byte-identical files - the --full set asserts exactly that with an md5 case; the other
assertions stay structural (frame counts, duration, metadata presence).
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.join(ROOT, "engine")
PY = sys.executable
SCRIPT = os.path.join(ENGINE, "gmfss_interp.py")
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


def render(inp, out, *args, trt=False):
    """Run one engine render; returns (returncode, stderr_text)."""
    cmd = [PY, SCRIPT, inp, "2", out, *args]
    if not trt:
        cmd.append("--no-trt")
    p = subprocess.run(cmd, cwd=ENGINE, capture_output=True, text=True)
    return p.returncode, (p.stderr or "") + (p.stdout or "")


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
    rc, log = render(SAMPLE, out, trt=trt)
    assert rc == 0, "engine exit " + str(rc)
    expect_part_promoted(out)
    assert frames(out) == 49, f"frames {frames(out)} != 49"


@case("--fps 60 -> 63 frames")
def c_fps60(tmp, trt):
    out = os.path.join(tmp, "s_60.mp4")
    rc, log = render(SAMPLE, out, "--fps", "60", trt=trt)
    assert rc == 0, "engine exit " + str(rc)
    expect_part_promoted(out)
    assert frames(out) == 63, f"frames {frames(out)} != 63"


@case("--no-interp -> 25 frames")
def c_noint(tmp, trt):
    out = os.path.join(tmp, "s_ni.mp4")
    rc, log = render(SAMPLE, out, "--no-interp", trt=trt)
    assert rc == 0, "engine exit " + str(rc)
    expect_part_promoted(out)
    assert frames(out) == 25, f"frames {frames(out)} != 25"


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
    rc, log = render(vfr, out, trt=trt)
    assert rc == 0, "engine exit " + str(rc)
    assert "VFR source" in log, "VFR notice missing from engine output"
    src_d, out_d = duration(vfr), duration(out)
    # On-grid 2x is up to ~2 output frames shorter/longer at the tail; 15% covers that on a 1s clip.
    assert abs(out_d - src_d) / src_d < 0.15, f"duration {out_d:.2f}s vs source {src_d:.2f}s"


# --------------------------------------------------------------------------- --full extras
@case("3x on-grid -> 73 frames")
def c_3x(tmp, trt):
    out = os.path.join(tmp, "s_3x.mp4")
    p = subprocess.run([PY, SCRIPT, SAMPLE, "3", out] + ([] if trt else ["--no-trt"]),
                       cwd=ENGINE, capture_output=True, text=True)
    assert p.returncode == 0, "engine exit " + str(p.returncode)
    assert frames(out) == 73, f"frames {frames(out)} != 73"


@case("5x on-grid -> 121 frames")
def c_5x(tmp, trt):
    out = os.path.join(tmp, "s_5x.mp4")
    p = subprocess.run([PY, SCRIPT, SAMPLE, "5", out] + ([] if trt else ["--no-trt"]),
                       cwd=ENGINE, capture_output=True, text=True)
    assert p.returncode == 0, "engine exit " + str(p.returncode)
    assert frames(out) == 121, f"frames {frames(out)} != 121"


@case("--codec av1 2x -> 49 frames")
def c_av1(tmp, trt):
    out = os.path.join(tmp, "s_av1.mp4")
    rc, _ = render(SAMPLE, out, "--codec", "av1", trt=trt)
    assert rc == 0, "engine exit " + str(rc)
    assert frames(out) == 49, f"frames {frames(out)} != 49"


@case("--codec vvc --no-interp -> 25 frames")
def c_vvc(tmp, trt):
    out = os.path.join(tmp, "s_vvc.mp4")
    rc, _ = render(SAMPLE, out, "--codec", "vvc", "--no-interp", trt=trt)
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
    rc, log = render(SAMPLE, out, "--rtx-hdr", "--no-interp", trt=trt)
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
    rc, log = render(SAMPLE, out, "--rtx-hdr", "--dv", "--no-interp", trt=trt)
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
                     "--hdr10plus", "--no-interp", trt=trt)
    assert rc == 0, "engine exit " + str(rc)
    assert "HDR10+: dynamic metadata written" in log, "export fell back: " + log[-300:]
    assert frames(out) == 25, f"frames {frames(out)} != 25"
    fsd = " ".join(first_frame_side_data(out))
    assert "SMPTE2094-40" in fsd or "HDR10+" in fsd, "HDR10+ dynamic metadata SEI missing"


@case("determinism: two identical renders -> identical md5")
def c_det(tmp, trt):
    import hashlib
    outs = []
    for i in range(2):
        out = os.path.join(tmp, f"s_det{i}.mp4")
        rc, _ = render(SAMPLE, out, trt=trt)
        assert rc == 0, "engine exit " + str(rc)
        outs.append(hashlib.md5(open(out, "rb").read()).hexdigest())
    assert outs[0] == outs[1], f"md5 mismatch: {outs[0]} vs {outs[1]}"


QUICK = [c_2x, c_fps60, c_noint, c_vfr]
FULL = [c_3x, c_5x, c_av1, c_vvc, c_hdr, c_dv, c_hp, c_det]


def main():
    ap = argparse.ArgumentParser(description="SMV engine smoke tests (real renders)")
    ap.add_argument("--full", action="store_true", help="also run 3x/5x, av1/vvc, HDR, DV, HDR10+")
    ap.add_argument("--trt", action="store_true",
                    help="use the default TensorRT path (first run per resolution builds engines); "
                         "default is --no-trt so a cold machine stays fast")
    ap.add_argument("--keep", action="store_true", help="keep the rendered outputs")
    args = ap.parse_args()
    assert os.path.isfile(SAMPLE), "samples/test.mp4 missing"
    tmp = tempfile.mkdtemp(prefix="smv-smoke-")
    print(f"smoke: sample={SAMPLE}\n       outputs={tmp}  backend={'TRT' if args.trt else 'eager'}")
    t0 = time.time()
    for fn in QUICK + (FULL if args.full else []):
        run_case(fn, tmp, args.trt)
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
