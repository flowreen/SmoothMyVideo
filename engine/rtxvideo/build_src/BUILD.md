# Rebuilding the RTX Video bridge (rtxvideo_cuda.dll)

`rtxvideo_cuda.dll` is a small CUDA bridge that lets the Python engine (`engine/rtxvideo.py`) drive
NVIDIA's RTX Video SDK (RTX VSR + TrueHDR) by `ctypes`. It is built locally and is **not
redistributable** (it statically links the SDK's `nvsdk_ngx_s.lib`), which is why all of
`engine/rtxvideo/` is gitignored and excluded from the packaged build.

You only need to rebuild it after updating the RTX Video SDK or moving to a different CUDA runtime.

## Sources here
- `rtx_video_api_cuda_impl.cpp` - the SDK's CUDA convenience layer (`samples/RTX_Video_API/`),
  copied so its `#include "utils.h"` picks up our override below.
- `utils.h` - overrides the SDK's hardcoded `APP_PATH` with an extern global so the model path can
  be set at runtime (`g_rtxv_model_path`).
- `rtxvideo_pathshim.cpp` - defines that global and exports `rtxv_set_model_path(const wchar_t*)`.
- `rtxvideo.def` - the exported C symbols (extern "C", undecorated on x64).

## Toolchain (verified on this machine)
- MSVC v142 (VS2019 Build Tools), `cl.exe` 19.29 via `VC\Auxiliary\Build\vcvars64.bat`.
- RTX Video SDK at `D:\AIStuff\RTX_Video_SDK` (headers in `include/`, `nvsdk_ngx_s.lib` in
  `lib\Windows\x64`, feature DLLs in `bin\Windows\x64\rel`).
- CUDA headers/libs come from the bundled torch runtime wheel (no separate CUDA Toolkit needed):
  `engine\runtime\Lib\site-packages\nvidia\cuda_runtime\{include, lib\x64}` (cuda.h is the driver
  API, CUDA 12.9; `nvcuda.dll` is in System32, `cudart64_12.dll` ships with torch).

## Recipe
From an `x64 Native Tools` prompt (or after running vcvars64.bat), in this folder:

```
set SDK=D:\AIStuff\RTX_Video_SDK
set RT=..\..\runtime\Lib\site-packages\nvidia\cuda_runtime
cl /nologo /LD /EHsc /MT /DNDEBUG ^
   /I"%SDK%\include" /I"%SDK%\samples\RTX_Video_API" /I"%RT%\include" ^
   rtx_video_api_cuda_impl.cpp rtxvideo_pathshim.cpp ^
   /Fe:rtxvideo_cuda.dll ^
   /link /DEF:rtxvideo.def /LIBPATH:"%SDK%\lib\Windows\x64" /LIBPATH:"%RT%\lib\x64" ^
   nvsdk_ngx_s.lib cuda.lib cudart.lib user32.lib shell32.lib advapi32.lib
```

Then copy `rtxvideo_cuda.dll` up into `engine/rtxvideo/` next to `nvngx_vsr.dll` +
`nvngx_truehdr.dll` (NGX resolves the feature DLLs relative to the loading module, so co-location is
what matters).

### Gotchas
- `/MT` (static CRT) is required - `nvsdk_ngx_s.lib` uses the static CRT; `/MD` gives LNK4098 +
  unresolved CRT symbols.
- `cudart.lib` is required (the NGX static lib references `cudaGetDevice`/`cudaGetDeviceProperties`
  to map the CUDA device to an adapter LUID); `user32`/`shell32`/`advapi32` are also needed.
- The feature DLLs (`nvngx_vsr.dll`, `nvngx_truehdr.dll`) are obtained from the RTX Video SDK and
  placed in `engine/rtxvideo/` - in the app, the in-GUI "Install runtime" button does this.

## CUDA 13 runtimes (do NOT rebuild the bridge against CUDA 13)

NVIDIA's `nvsdk_ngx_s.lib` (the static NGX lib linked into the bridge) is built for the **CUDA 12**
runtime ABI: internally it calls `cudaGetDeviceProperties` with a CUDA 12-sized `cudaDeviceProp`. Link
a CUDA 13 `cudart` and the runtime writes the larger CUDA 13 struct into that smaller buffer, overrunning
the stack - the process dies with `0xC0000409` (STATUS_STACK_BUFFER_OVERRUN) inside NGX `create()`.
Verified: a bridge relinked against `cudart64_13` (whether via the wheel's static `cudart.lib` or a
synthesized dynamic import lib) loads fine but crashes in `create()`. We cannot recompile NVIDIA's lib.

So to run under a **CUDA 13** runtime (torch `cu130` etc.), keep the **cu12 bridge exactly as built**
and just drop `cudart64_12.dll` next to it in `engine/rtxvideo/`. A cu13 runtime ships only
`cudart64_13.dll`, so the bridge's `cudart64_12` import would otherwise be unresolved. With it present,
the bridge uses cu12 `cudart` for its read-only device-property / LUID query (matching NGX's ABI) while
torch uses cu13 `cudart` separately; the CUDA **driver** context is shared (driver API, version
agnostic), so VSR and TrueHDR run correctly. Validated end to end on torch 2.12.1+cu130 + cupy-cuda13x.
