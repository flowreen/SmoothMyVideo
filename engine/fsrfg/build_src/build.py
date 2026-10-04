r"""Builds engine\fsrfg\smv_fsrfg_bridge.dll: AMD FSR 3.1.6 frame generation from the FSR SDK v2.3.0 source (MIT, listed
in Kits/FidelityFX/docs/license.md) plus fsrfg_bridge.cpp, in one DLL. One change to AMD's source:
ffx_frameinterpolation_setup.h's `if(Reset() || HasSceneChanged())` -> `if(Reset())`, so the optical flow's
scene-change detection never makes FSR copy the current frame in place of a tween (video has no scene-change signal
from a game engine; a repeated frame is a hold). Two pieces AMD does not publish are written here: the driver-provider
header (a stub that is never Valid(): no AMD driver provider) and the DLL's provider list (only the FSR 3 frame
generation). Shaders: FidelityFX_SC.exe + DXC from the MIT FidelityFX SDK v1.1.4 (its tools folder), with the
permutation arguments of AMD's BuildFrameInterpolationShaders.bat / BuildOpticalFlowShaders.bat.

Environment:
  FSR_SDK     the FSR SDK v2.3.0 checkout (holds Kits\FidelityFX)
  FFX_SC_DIR  a folder with FidelityFX_SC.exe, dxcompiler.dll and dxil.dll (FidelityFX SDK v1.1.4)
Usage (any python 3.8+, Visual Studio 2026 with the C++ workload): python build.py"""
import glob
import os
import shutil
import subprocess
import sys
from typing import List

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DLL = os.path.join(os.path.dirname(HERE), "smv_fsrfg_bridge.dll")
# beside this file: FidelityFX_SC v1.1.4 found none of its includes with the copied sources under %TEMP% (C:), from any
# working folder; the same sources beside the checkout compile
WORK = os.path.join(HERE, "work")
SRC = os.path.join(WORK, "src", "Kits", "FidelityFX")
OUT_SC = os.path.join(WORK, "ffx_sc_output")

SCD_LINE = "if(Reset() || HasSceneChanged()) {"
# build fixes on the copied sources, each must match exactly once: AMD's own build forces things this one does not have
# (AGS and PIX tooling, a precompiled ffx_util.h) and its FidelityFX_SC writes an entryName member v1.1.4's tool lacks
# (every pass compiles with -E CS)
FIXES = [
    (os.path.join("backend", "dx12", "ffx_dx12.cpp"), "#define ENABLE_AGS 1",
     "// ENABLE_AGS off: no AMD GPU Services in this build"),
    (os.path.join("backend", "dx12", "ffx_dx12.cpp"), "#define ENABLE_PIX_CAPTURES 1",
     "// ENABLE_PIX_CAPTURES off: no PIX runtime in this build"),
    (os.path.join("api", "internal", "ffx_message.cpp"), '#include "ffx_message.h"',
     '#include "ffx_message.h"\n#include "ffx_util.h"'),
    (os.path.join("api", "internal", "ffx_internal_types.h"), "info[index].entryName,", '"CS",'),
]
STUB = """#pragma once
// Stand-in for AMD's unpublished driver-provider header: no AMD driver provider is ever offered (Valid() false), so
// GetProvider always picks the built-in FSR 3 frame generation.
#include <d3d12.h>
class ffxProviderExternal : public ffxProvider
{
public:
    ffxProviderExternal(ID3D12Device*, ffxStructType_t) : ffxProvider(0, 0, "none") {}
    bool Valid() const { return false; }
    bool CanProvide(uint64_t) const override { return false; }
    bool IsSupported(void*) const override { return false; }
    ffxReturnCode_t CreateContext(ffxContext*, ffxCreateContextDescHeader*, Allocator&) override { return FFX_API_RETURN_NO_PROVIDER; }
    ffxReturnCode_t DestroyContext(ffxContext*, Allocator&) override { return FFX_API_RETURN_NO_PROVIDER; }
    ffxReturnCode_t Configure(ffxContext*, const ffxConfigureDescHeader*) const override { return FFX_API_RETURN_NO_PROVIDER; }
    ffxReturnCode_t Query(ffxContext*, ffxQueryDescHeader*) const override { return FFX_API_RETURN_NO_PROVIDER; }
    ffxReturnCode_t Dispatch(ffxContext*, const ffxDispatchDescHeader*) const override { return FFX_API_RETURN_NO_PROVIDER; }
};
"""
PROVIDERS = """// The provider list of this DLL: only the FSR 3 frame generation (AMD's per-DLL list is not published).
#include "Kits/FidelityFX/api/internal/ffx_provider.h"
#include "Kits/FidelityFX/framegeneration/fsr3/include/ffx_provider_fsr3framegeneration.h"

static ffxProvider* const kProviders[] = {&ffxProvider_Fsr3FrameGeneration::GetInstance()};

ffxProvider* GetProvider(ffxStructType_t descType, uint64_t overrideId, void* device,
                         std::optional<ffxProviderExternal>& extProviderSlot)
{
    return GetProvider(descType, overrideId, device, extProviderSlot, kProviders);
}

uint64_t GetProviderVersions(ffxStructType_t descType, void* device, uint64_t capacity, uint64_t* versionIds,
                             const char** versionNames, std::optional<ffxProviderExternal>& extProviderSlot)
{
    return GetProviderVersions(descType, device, capacity, versionIds, versionNames, extProviderSlot, kProviders);
}
"""
BASE = ["-reflection", "-deps=gcc", "-DFFX_GPU=1", "-DFFX_IMPLICIT_SHADER_REGISTER_BINDING_HLSL=0"]
API = ["-embed-arguments", "-E", "CS", "-Wno-for-redefinition", "-Wno-ambig-lit-shift", "-DFFX_HLSL=1"]
WAVE64 = ["-DFFX_PREFER_WAVE64=[WaveSize(64)]", "-DFFX_HLSL_SM=66", "-T", "cs_6_6"]
WAVE32 = ["-DFFX_HLSL_SM=62", "-T", "cs_6_2"]
HALF = ["-DFFX_HALF=1", "-enable-16bit-types"]
FI = (["-DFFX_FRAMEINTERPOLATION_OPTION_UPSAMPLE_SAMPLERS_USE_DATA_HALF=0",
       "-DFFX_FRAMEINTERPOLATION_OPTION_ACCUMULATE_SAMPLERS_USE_DATA_HALF=0",
       "-DFFX_FRAMEINTERPOLATION_OPTION_REPROJECT_SAMPLERS_USE_DATA_HALF=1",
       "-DFFX_FRAMEINTERPOLATION_OPTION_POSTPROCESSLOCKSTATUS_SAMPLERS_USE_DATA_HALF=0",
       "-DFFX_FRAMEINTERPOLATION_OPTION_UPSAMPLE_USE_LANCZOS_TYPE=2"] + BASE + API
      + ["-DFFX_FRAMEINTERPOLATION_EMBED_ROOTSIG=0", "-DFFX_FRAMEINTERPOLATION_OPTION_LOW_RES_MOTION_VECTORS={0,1}",
         "-DFFX_FRAMEINTERPOLATION_OPTION_JITTER_MOTION_VECTORS={0,1}",
         "-DFFX_FRAMEINTERPOLATION_OPTION_INVERTED_DEPTH={0,1}"])
OF = BASE + API + ["-DFFX_OPTICALFLOW_EMBED_ROOTSIG=0", "-DFFX_OPTICALFLOW_OPTION_HDR_COLOR_INPUT={0,1}"]


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
    sdk = os.path.join(os.environ.get("FSR_SDK", ""), "Kits", "FidelityFX")
    tools = os.environ.get("FFX_SC_DIR", "")
    sc, dxc = os.path.join(tools, "FidelityFX_SC.exe"), os.path.join(tools, "dxcompiler.dll")
    if not os.path.isdir(sdk) or not os.path.isfile(sc) or not os.path.isfile(dxc):
        raise SystemExit("set FSR_SDK (the FSR SDK v2.3.0 checkout) and FFX_SC_DIR (FidelityFX_SC.exe + DXC)")
    if os.path.exists(WORK):
        shutil.rmtree(WORK)
    for sub in ("api", "backend", "framegeneration", os.path.join("upscalers", "fsr3", "include")):
        shutil.copytree(os.path.join(sdk, sub), os.path.join(SRC, sub))
    stub = os.path.join(SRC, "amdinternal", "api", "internal", "dx12", "ffx_provider_external.h")
    os.makedirs(os.path.dirname(stub))
    open(stub, "w", encoding="utf-8").write(STUB)
    open(os.path.join(WORK, "src", "providers.cpp"), "w", encoding="utf-8").write(PROVIDERS)
    for rel, old, new in FIXES:
        path = os.path.join(SRC, rel)
        t = open(path, encoding="utf-8").read()
        if t.count(old) != 1:
            raise SystemExit(f"{path}: '{old}' occurs {t.count(old)} times, want 1")
        open(path, "w", encoding="utf-8").write(t.replace(old, new))
    setup = os.path.join(SRC, "framegeneration", "fsr3", "include", "gpu", "frameinterpolation",
                         "ffx_frameinterpolation_setup.h")
    text = open(setup, encoding="utf-8").read()
    if text.count(SCD_LINE) != 1:
        raise SystemExit(f"{setup}: the scene-change line occurs {text.count(SCD_LINE)} times, want 1")
    open(setup, "w", encoding="utf-8").write(text.replace(SCD_LINE, "if(Reset()) {"))
    print(f"sources copied to {SRC}, the scene-change reset removed")

    # shaders: every pass x {wave32, wave64} x {fp32, 16-bit}, FidelityFX_SC's permutations per pass
    os.makedirs(OUT_SC)
    sclog = os.path.join(WORK, "shaders.log")
    inc = ["-I", os.path.join(SRC, "api", "internal", "gpu"), "-I",
           os.path.join(SRC, "framegeneration", "fsr3", "include", "gpu")]
    shaders = sorted(glob.glob(os.path.join(SRC, "framegeneration", "fsr3", "internal", "shaders", "ffx_*.hlsl")))
    fails = 0
    for path in shaders:
        name = os.path.splitext(os.path.basename(path))[0]
        args = FI if name.startswith("ffx_frameinterpolation") else OF
        for suffix, extra in (("", ["-DFFX_HALF=0"] + WAVE32), ("_wave64", ["-DFFX_HALF=0"] + WAVE64),
                              ("_16bit", HALF + WAVE32), ("_wave64_16bit", HALF + WAVE64)):
            cmd = [sc, "-Zs"] + args + [f"-name={name}{suffix}"] + extra + inc + [f"-dxcdll={dxc}",
                                                                                  f"-output={OUT_SC}", path]
            fails += log_run(cmd, sclog) != 0
    print(f"shaders: {len(shaders)} passes x 4, {fails} failed compiles ({sclog})")
    if fails:
        return 1

    cpp = ([os.path.join(SRC, "api", "internal", f) for f in ("ffx_api.cpp", "ffx_assert.cpp", "ffx_message.cpp",
                                                              "ffx_object_management.cpp", "ffx_query_fallback.cpp")]
           + glob.glob(os.path.join(SRC, "backend", "dx12", "*.cpp"))
           + glob.glob(os.path.join(SRC, "framegeneration", "fsr3", "internal", "*.cpp"))
           + glob.glob(os.path.join(SRC, "framegeneration", "fsr3", "dx12", "*.cpp"))
           + [os.path.join(WORK, "src", "providers.cpp"), os.path.join(HERE, "fsrfg_bridge.cpp")])
    cl = (["cl", "/nologo", "/O2", "/EHsc", "/std:c++20", "/W3", "/MD", "/LD", "/D_WINDOWS", "/DUNICODE", "/D_UNICODE",
           "/DNOMINMAX", "/DWIN32_LEAN_AND_MEAN", "/DFFX_BACKEND_DX12=1", "/DFFX_FRAMEGENERATION=1",
           "/I", os.path.join(WORK, "src"), "/I", OUT_SC, f"/Fo{WORK}\\", f"/Fe{os.path.join(WORK, 'out.dll')}"]
          + cpp + ["/link", "d3d12.lib", "dxgi.lib", "dxguid.lib"])
    cmdfile = os.path.join(WORK, "build_dll.cmd")
    with open(cmdfile, "w", encoding="utf-8") as f:
        f.write(f'@call "{vcvars()}" > nul\r\n' + subprocess.list2cmdline(cl) + "\r\nexit /b %ERRORLEVEL%\r\n")
    clog = os.path.join(WORK, "build_dll.log")
    rc = log_run(["cmd", "/c", cmdfile], clog)
    if rc == 0:
        shutil.copyfile(os.path.join(WORK, "out.dll"), OUT_DLL)
    print(f"dll: cl rc {rc} ({clog}){', ' + OUT_DLL if rc == 0 else ''}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
