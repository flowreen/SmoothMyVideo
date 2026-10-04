r"""Builds engine\fsrup\smv_fsrup_bridge.dll (fsrup_bridge.cpp: the host's C API over AMD's FidelityFX upscaler) and copies
AMD's signed FidelityFX loader + upscaler DLLs from the FSR SDK v2.3.0 beside it (both MIT: Kits/FidelityFX/docs/license.md
lists Kits\FidelityFX\signedbin\amd_fidelityfx_loader_dx12.dll and amd_fidelityfx_upscaler_dx12.dll in its MIT section).

Environment:
  FSR_SDK  the FSR SDK v2.3.0 checkout (holds Kits\FidelityFX: api\include, upscalers\include, signedbin)
Usage (any python 3.8+, Visual Studio 2026 with the C++ workload): python build.py"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
WORK = os.path.join(HERE, "work")
SIGNED = ("amd_fidelityfx_loader_dx12.dll", "amd_fidelityfx_upscaler_dx12.dll")


def vcvars() -> str:
    vswhere = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "Microsoft Visual Studio",
                           "Installer", "vswhere.exe")
    vs = subprocess.run([vswhere, "-latest", "-property", "installationPath"], capture_output=True,
                        text=True).stdout.strip()
    return os.path.join(vs, "VC", "Auxiliary", "Build", "vcvars64.bat")


def main() -> int:
    sdk = os.path.join(os.environ.get("FSR_SDK", ""), "Kits", "FidelityFX")
    if not all(os.path.isfile(os.path.join(sdk, "signedbin", d)) for d in SIGNED):
        raise SystemExit("set FSR_SDK (the FSR SDK v2.3.0 checkout with Kits\\FidelityFX\\signedbin)")
    os.makedirs(WORK, exist_ok=True)
    cl = ["cl", "/nologo", "/O2", "/EHsc", "/std:c++17", "/W3", "/MD", "/LD", "/D_WINDOWS", "/DUNICODE", "/D_UNICODE",
          "/DNOMINMAX", "/I", os.path.join(sdk, "api", "include"), "/I", os.path.join(sdk, "upscalers", "include"),
          f"/Fo{WORK}\\", f"/Fe{os.path.join(WORK, 'smv_fsrup_bridge.dll')}", os.path.join(HERE, "fsrup_bridge.cpp"),
          "/link", "d3d12.lib", "dxgi.lib"]
    cmdfile = os.path.join(WORK, "build_dll.cmd")
    with open(cmdfile, "w", encoding="utf-8") as f:
        f.write(f'@call "{vcvars()}" > nul\r\n' + subprocess.list2cmdline(cl) + "\r\nexit /b %ERRORLEVEL%\r\n")
    log = os.path.join(WORK, "build_dll.log")
    with open(log, "w", encoding="utf-8") as lf:
        rc = subprocess.run(["cmd", "/c", cmdfile], stdout=lf, stderr=subprocess.STDOUT).returncode
    if rc == 0:
        shutil.copyfile(os.path.join(WORK, "smv_fsrup_bridge.dll"), os.path.join(OUT, "smv_fsrup_bridge.dll"))
        for d in SIGNED:
            shutil.copyfile(os.path.join(sdk, "signedbin", d), os.path.join(OUT, d))
    print(f"dll: cl rc {rc} ({log}){', ' + OUT + ' (+ the two signed AMD DLLs)' if rc == 0 else ''}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
