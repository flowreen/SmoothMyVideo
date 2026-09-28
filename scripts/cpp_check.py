"""C++ static checks of the native host (engine/live/build_src/smv-live.cpp and the DLSS 5 core
engine/dlssnr/build_src/nr_host.cpp), compiled from build.bat's own cl line into a temporary folder, so the
shipped engine/live/smv-live.exe is never touched:
  analyze  MSVC /analyze (the C6xxx / C26xxx / C28xxx checks; the SDK headers are external, not analysed)
  w4       warning level 4 (the ship build uses /W3), the SDK headers external
Needs what build.bat needs: SL_SDK, TRT_RTX_SDK and NGX_SDK (SMV_CU defaults to the dev runtime's CUDA wheel)
and Visual Studio 2026 (vcvars64 through vswhere). Prints the warnings grouped by code and every warning line of
our own sources; exit 0 = none in our sources. --keep DIR writes the objects and the log there instead.
Usage: python scripts/cpp_check.py analyze|w4 [--keep DIR]"""
import collections
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BSRC = os.path.join(REPO, "engine", "live", "build_src")
NRSRC = os.path.join(REPO, "engine", "dlssnr", "build_src")
OWN = ("smv-live.cpp", "smv-live-", "nr_host.", "sl_focus_shim.h")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in ("analyze", "w4"):
        print(__doc__)
        return 2
    env = {k: os.environ.get(k, "").strip() for k in ("SL_SDK", "TRT_RTX_SDK", "NGX_SDK")}
    missing = [k for k, v in env.items() if not v]
    if missing:
        print("set " + ", ".join(missing) + " first (the same variables build.bat needs, see Building the native host)")
        return 2
    env["SMV_CU"] = os.environ.get("SMV_CU") or os.path.join(REPO, "engine", "runtime", "Lib", "site-packages", "nvidia", "cu13")
    env["NRSRC"] = NRSRC
    bat = open(os.path.join(BSRC, "build.bat"), encoding="utf-8").read()
    m = re.search(r"^cl /nologo .*?/OUT:\.\.\\smv-live\.exe", bat, re.S | re.M)
    if not m:
        print("build.bat: the cl line was not found")
        return 2
    cl = re.sub(r"\^\r?\n\s*", " ", m.group(0)).replace("%~dp0", BSRC + "\\")
    for k, v in env.items():
        cl = cl.replace("%" + k + "%", v)
    if "%" in cl:
        print("build.bat: an unresolved variable in the cl line: " + cl)
        return 2
    keep = sys.argv[sys.argv.index("--keep") + 1] if "--keep" in sys.argv else None
    work = os.path.abspath(keep) if keep else tempfile.mkdtemp(prefix="smv_cpp_check_")
    os.makedirs(work, exist_ok=True)
    external = [env["SL_SDK"] + r"\include", env["TRT_RTX_SDK"] + r"\include", env["SMV_CU"] + r"\include",
                env["NGX_SDK"] + r"\include", os.path.join(BSRC, "nvofa"), os.path.join(BSRC, "cuda_shim")]
    ext = " ".join('/external:I "' + d + '"' for d in external)
    # C6262 at 32 KB instead of 16: every host thread runs on the default 1 MB stack
    extra = "/analyze /analyze:external- /analyze:stacksize 32768" if mode == "analyze" else "/W4"
    cl = cl[:cl.index(" /link ")].replace("cl /nologo", f"cl /nologo /c /Fo{work}\\ {extra} "
                                                          f"/external:anglebrackets /external:W0 {ext}", 1)
    if mode == "w4":
        cl = cl.replace(" /W3 ", " ")
    cmd, log = os.path.join(work, mode + ".cmd"), os.path.join(work, mode + ".log")
    with open(cmd, "w", encoding="ascii") as f:
        f.write("@echo off\r\nsetlocal\r\n"
                "where cl >nul 2>nul\r\nif not errorlevel 1 goto compile\r\n"
                'set "VSWHERE=%ProgramFiles(x86)%\\Microsoft Visual Studio\\Installer\\vswhere.exe"\r\n'
                'for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -property installationPath`) do set "VSDIR=%%i"\r\n'
                'call "%VSDIR%\\VC\\Auxiliary\\Build\\vcvars64.bat" >nul || exit /b 1\r\n'
                ":compile\r\n"
                'cd /d "' + BSRC + '"\r\n' + cl + "\r\nexit /b %errorlevel%\r\n")
    with open(log, "w", encoding="utf-8") as lf:
        rc = subprocess.run(["cmd", "/c", cmd], stdout=lf, stderr=subprocess.STDOUT).returncode
    txt = open(log, encoding="utf-8", errors="replace").read()
    warn = sorted({ln.strip() for ln in txt.splitlines() if re.search(r"\bwarning [A-Z]+\d+", ln)})
    own = [ln for ln in warn if any(o in ln for o in OWN)]
    codes = collections.Counter(m.group(1) for ln in warn if (m := re.search(r"\bwarning ([A-Z]+\d+)", ln)))
    print(f"{mode}: compiler rc {rc}, {len(warn)} distinct warning line(s), {len(own)} in our sources")
    for c, n in codes.most_common():
        print(f"  {c}: {n}")
    for ln in own:
        print("  " + ln)
    if rc:
        print("compile failed, log: " + log)
    elif not keep:
        shutil.rmtree(work, ignore_errors=True)
    return rc if rc else (1 if own else 0)


if __name__ == "__main__":
    sys.exit(main())
