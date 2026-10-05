r"""Builds engine\fsrfg: smv_fsrfg_bridge.dll (fsrfg_bridge.cpp) and amdxc64.dll (amdxc64_shim.cpp), and copies beside
them AMD's signed FSR SDK v2.3.0 loader + frame generation DLLs (signedbin, unmodified) and vkd3d-proton's two DLLs.

vkd3d-proton (LGPL-2.1, built separately, MSYS2 UCRT64 with gcc, meson and ninja): github.com/HansKristian-Work/
vkd3d-proton at commit b206eb6680fb92a64ae57bfccc7454a138f76887 with its submodules, dxil-spirv at
ab47c3df1a4746f36c958f0262bc9e314eb566eb, then
  git apply ..\source\vkd3d-proton-b206eb6-smv.patch            (in the vkd3d-proton checkout)
  git apply ..\source\dxil-spirv-ab47c3d-smv.patch              (in subprojects\dxil-spirv)
  meson setup build --buildtype=release -Denable_extended_emulation=true
  ninja -C build
The patches: VKD3D_FP8_EMULATION=1 runs the FP8 cooperative matrices of AMD's ML shaders as FP16 ones, dxil-spirv's
DXIL_SPIRV_CONFIG=wmma_fp8_staging stages FP8 loads / stores through shared memory as FP16 and reads its configuration
from the process environment, and the two DLLs are named smv_vkd3d_d3d12.dll / smv_vkd3d_d3d12core.dll.

Environment:
  FSR_SDK      the FSR SDK v2.3.0 checkout (holds Kits\FidelityFX: api\include, framegeneration\include, signedbin)
  VKD3D_BUILD  the vkd3d-proton build folder above (holds libs\d3d12\smv_vkd3d_d3d12.dll, libs\d3d12core\...)
Usage (any python 3.8+, Visual Studio 2026 with the C++ workload): python build.py [output folder, default engine\fsrfg]
"""
import os
import shutil
import subprocess
import sys
from typing import List

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "work")


def log_run(cmd: List[str], logf: str) -> int:
    with open(logf, "a", encoding="utf-8") as f:
        f.write("> " + " ".join(cmd) + "\n")
        f.flush()
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        f.write(f"rc {rc}\n")
    return rc


def vcvars() -> str:
    vswhere = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "Microsoft Visual Studio",
                           "Installer", "vswhere.exe")
    vs = subprocess.run([vswhere, "-latest", "-property", "installationPath"], capture_output=True,
                        text=True).stdout.strip()
    return os.path.join(vs, "VC", "Auxiliary", "Build", "vcvars64.bat")


def main() -> int:
    out = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.dirname(HERE)
    sdk = os.path.join(os.environ.get("FSR_SDK", ""), "Kits", "FidelityFX")
    vk = os.environ.get("VKD3D_BUILD", "")
    vk_dlls = [os.path.join(vk, "libs", "d3d12", "smv_vkd3d_d3d12.dll"),
               os.path.join(vk, "libs", "d3d12core", "smv_vkd3d_d3d12core.dll")]
    amd = [os.path.join(sdk, "signedbin", n) for n in ("amd_fidelityfx_loader_dx12.dll",
                                                        "amd_fidelityfx_framegeneration_dx12.dll")]
    missing = [p for p in vk_dlls + amd if not os.path.isfile(p)]
    if missing:
        raise SystemExit("set FSR_SDK (the FSR SDK v2.3.0 checkout) and VKD3D_BUILD (the vkd3d-proton build folder); "
                         f"missing: {', '.join(missing)}")
    if os.path.exists(WORK):
        shutil.rmtree(WORK)
    os.makedirs(WORK)
    os.makedirs(out, exist_ok=True)
    common = ["cl", "/nologo", "/O2", "/EHsc", "/W3", "/MT", "/LD", "/D_WINDOWS", "/DNOMINMAX", "/DWIN32_LEAN_AND_MEAN",
              f"/Fo{WORK}\\"]
    bridge = (common + ["/std:c++20", "/I", os.path.join(sdk, "api", "include"), "/I",
                        os.path.join(sdk, "framegeneration", "include"), f"/Fe{os.path.join(WORK, 'bridge.dll')}",
                        os.path.join(HERE, "fsrfg_bridge.cpp"), "/link", "dxgi.lib"])
    shim = common + ["/std:c++17", f"/Fe{os.path.join(WORK, 'amdxc64.dll')}", os.path.join(HERE, "amdxc64_shim.cpp")]
    cmdfile = os.path.join(WORK, "build_dll.cmd")
    with open(cmdfile, "w", encoding="utf-8") as f:
        f.write(f'@call "{vcvars()}" > nul\r\n' + subprocess.list2cmdline(bridge) + "\r\nif errorlevel 1 exit /b 1\r\n"
                + subprocess.list2cmdline(shim) + "\r\nexit /b %ERRORLEVEL%\r\n")
    clog = os.path.join(WORK, "build_dll.log")
    rc = log_run(["cmd", "/c", cmdfile], clog)
    print(f"dll: cl rc {rc} ({clog})")
    if rc:
        return rc
    shutil.copyfile(os.path.join(WORK, "bridge.dll"), os.path.join(out, "smv_fsrfg_bridge.dll"))
    shutil.copyfile(os.path.join(WORK, "amdxc64.dll"), os.path.join(out, "amdxc64.dll"))
    for p in vk_dlls + amd:
        shutil.copyfile(p, os.path.join(out, os.path.basename(p)))
    print(f"installed into {out}: smv_fsrfg_bridge.dll, amdxc64.dll, " + ", ".join(os.path.basename(p)
                                                                                 for p in vk_dlls + amd))
    return 0


if __name__ == "__main__":
    sys.exit(main())
