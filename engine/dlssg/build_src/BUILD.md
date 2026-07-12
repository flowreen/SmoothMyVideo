# Building dlssg2f.exe (the DLSS Frame Generation host)

`dlssg2f.exe` is the offline D3D12 presentation loop that lets DLSS-FG (a game-only
SDK) interpolate two video frames. It is built from the single `main.cpp` here and
ships prebuilt in this folder's parent, together with NVIDIA's redistributable
Streamline runtime (sl.*.dll + nvngx_dlssg.dll, from the Streamline SDK's
`bin/x64` production set; licenses alongside).

Requirements:
* Visual Studio 2022+ with the C++ workload (MSVC x64)
* The NVIDIA Streamline SDK, v2.12.0 or later: https://github.com/NVIDIA-RTX/Streamline/releases
  (the `streamline-sdk-v*.zip` release asset, extracted anywhere)

Build:

```bat
set SL_SDK=D:\path\to\extracted\streamline-sdk
build.bat
```

`build.bat` compiles `main.cpp` against `%SL_SDK%\include`, links
`%SL_SDK%\lib\x64\sl.interposer.lib` INSTEAD of d3d12.lib/dxgi.lib (that is how
Streamline interposes the API; verify with `dumpbin /dependents dlssg2f.exe` that
neither d3d12.dll nor dxgi.dll appears), and writes `..\dlssg2f.exe`.

Runtime notes (why the exe is shaped the way it is):
* The swap chain must be created with `FRAME_LATENCY_WAITABLE_OBJECT | ALLOW_TEARING`;
  Streamline's frame pacer calls `SetMaximumFrameLatency` and presents with
  `DXGI_PRESENT_ALLOW_TEARING`, and without the flags the very first Present fails
  with DXGI_ERROR_INVALID_CALL inside the SL present hook.
* `DLSSGFlags::eShowOnlyInterpolatedFrame` makes every native present a generated
  frame; the host reads it back from the native swap chain (`slGetNativeInterface`)
  after polling `GetLastPresentCount`, so no window ever needs to be on screen.
* DLSS-FG requires Windows hardware-accelerated GPU scheduling ON and an RTX 40/50
  GPU; the host exits with code 2 when unsupported.
