@echo off
setlocal
if "%SL_SDK%"=="" (
  echo Set SL_SDK to the extracted Streamline SDK first, see DEVELOPMENT.md
  exit /b 1
)
rem "set SL_SDK=path && build.bat" leaves a trailing space that corrupts /I, trim it
:trimsdk
if "%SL_SDK:~-1%"==" " set "SL_SDK=%SL_SDK:~0,-1%"
if "%SL_SDK:~-1%"==" " goto trimsdk
where cl >nul 2>nul
if not errorlevel 1 goto compile
rem no parens block here: the ^) in "Program Files (x86)" would close it early
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
  echo vswhere.exe not found, run from a VS x64 developer prompt instead
  exit /b 1
)
set VSDIR=
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -property installationPath`) do set "VSDIR=%%i"
if "%VSDIR%"=="" (
  echo vswhere found no Visual Studio installation
  exit /b 1
)
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
:compile
cd /d %~dp0
rem WO-15 native RIFE host. TRT_RTX_SDK = the extracted TensorRT-RTX SDK (include\ + lib\).
rem SMV_CU = the CUDA headers and import libs that already ship in the app's python runtime;
rem it defaults to this checkout's own engine\runtime, so a plain build needs neither a CUDA
rem toolkit nor any absolute path. cuda_shim supplies the one internal header the runtime wheel
rem leaves out (see cuda_shim\crt\host_defines.h).
if "%TRT_RTX_SDK%"=="" (
  echo Set TRT_RTX_SDK to the extracted TensorRT-RTX SDK first, see DEVELOPMENT.md
  exit /b 1
)
:trimtrt
if "%TRT_RTX_SDK:~-1%"==" " set "TRT_RTX_SDK=%TRT_RTX_SDK:~0,-1%"
if "%TRT_RTX_SDK:~-1%"==" " goto trimtrt
if "%SMV_CU%"=="" set "SMV_CU=%~dp0..\..\runtime\Lib\site-packages\nvidia\cu13"
rem WO-42: the DLSS 5 Neural Rendering core (engine\dlssnr\build_src\nr_host.cpp) is compiled
rem into the exe; NGX_SDK = the public NVIDIA DLSS SDK folder holding include\nvsdk_ngx.h, the
rem same variable the dlssnr build uses (see DEVELOPMENT.md). Build time only, nothing shipped.
if "%NGX_SDK%"=="" (
  echo Set NGX_SDK to the NVIDIA DLSS SDK folder that contains include\nvsdk_ngx.h, see DEVELOPMENT.md
  exit /b 1
)
:trimngx
if "%NGX_SDK:~-1%"==" " set "NGX_SDK=%NGX_SDK:~0,-1%"
if "%NGX_SDK:~-1%"==" " goto trimngx
set "NRSRC=%~dp0..\..\dlssnr\build_src"
rem Nothing below is a hard dependency of the exe: tensorrt_rtx_1_6.dll is delay loaded, and
rem CUDA 13's cudart.lib and cuda.lib are lazy LOADER stubs that open their DLLs on first call,
rem so the exe still starts on a machine that has none of them and no non-native route changes.
rem nvofa\ = the MIT NVIDIA Optical Flow interface headers (the nvof model); nvofapi64.dll is
rem driver-installed and opened at run time by full path, nothing to link or ship.
rem No d3d12.lib on purpose: D3D12CreateDevice must keep resolving from sl.interposer.lib.
cl /nologo /std:c++20 /EHsc /permissive- /O2 /W3 smv-live.cpp "%NRSRC%\nr_host.cpp" ^
   /I "%SL_SDK%\include" /I "%TRT_RTX_SDK%\include" /I "%SMV_CU%\include" /I cuda_shim ^
   /I "%NRSRC%" /I "%NGX_SDK%\include" /I nvofa ^
   /link /LIBPATH:"%SL_SDK%\lib\x64" /LIBPATH:"%TRT_RTX_SDK%\lib" /LIBPATH:"%SMV_CU%\lib\x64" ^
   sl.interposer.lib tensorrt_rtx_1_6.lib tensorrt_onnxparser_rtx_1_6.lib cudart.lib cuda.lib delayimp.lib ^
   user32.lib gdi32.lib ole32.lib windowscodecs.lib dxguid.lib advapi32.lib ^
   d3d11.lib d3dcompiler.lib dwmapi.lib shlwapi.lib winmm.lib windowsapp.lib ^
   /DELAYLOAD:tensorrt_rtx_1_6.dll /DELAYLOAD:tensorrt_onnxparser_rtx_1_6.dll ^
   /SUBSYSTEM:CONSOLE /OUT:..\smv-live.exe
exit /b %errorlevel%
