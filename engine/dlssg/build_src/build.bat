@echo off
setlocal
rem NGX_SDK = the public NVIDIA DLSS SDK folder (include\nvsdk_ngx.h, lib\Windows_x86_64\x64\nvsdk_ngx_s.lib), the
rem same variable the live host's and the dlssnr build use: the helper calls NGX's frame generation itself
if "%NGX_SDK%"=="" (
  echo Set NGX_SDK to the NVIDIA DLSS SDK folder that contains include\nvsdk_ngx.h, see DEVELOPMENT.md
  exit /b 1
)
if not exist "%NGX_SDK%\lib\Windows_x86_64\x64\nvsdk_ngx_s.lib" (
  echo %NGX_SDK%\lib\Windows_x86_64\x64\nvsdk_ngx_s.lib not found, see DEVELOPMENT.md
  exit /b 1
)
where cl >nul 2>nul
if not errorlevel 1 goto compile
rem no parens block here: the ^) in "Program Files (x86)" would close it early (same as the live host's build.bat)
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
rem the object file goes to %TEMP%, so the source folder holds only sources
rem nvsdk_ngx_s.lib is NGX's static loader, it opens the driver's NGX at run time
cl /nologo /std:c++17 /EHsc /O2 /W3 main.cpp /Fo"%TEMP%\dlssg2f_main.obj" /I "%NGX_SDK%\include" ^
   /link /LIBPATH:"%NGX_SDK%\lib\Windows_x86_64\x64" nvsdk_ngx_s.lib ^
   d3d12.lib dxgi.lib user32.lib gdi32.lib ole32.lib dxguid.lib advapi32.lib ^
   /SUBSYSTEM:CONSOLE /OUT:..\dlssg2f.exe
exit /b %errorlevel%
