@echo off
rem Builds engine\xess\smv_xess_bridge.dll (xess_bridge.cpp: the host's C API over Intel's XeSS-SR) and
rem engine\xess\smv_xefg_bridge.dll (xefg_bridge.cpp: the host's frame generation C API over Intel's XeSS-FG + XeLL).
rem Intel's DLLs are NOT copied: the app's XeSS setup downloads the pinned XeSS SDK release and installs them beside
rem these DLLs (each bridge imports Intel's DLLs, so the host loads a bridge only when they are there).
rem
rem Environment:
rem   XESS_SDK  the extracted XeSS SDK 3.0.2 (holds inc\xess\xess_d3d12.h, inc\xess_fg\, inc\xell\ and lib\libxess.lib,
rem             lib\libxess_fg.lib, lib\libxell.lib)
rem Usage (Visual Studio 2026 with the C++ workload): build.bat [fg]   (fg = smv_xefg_bridge.dll only)
rem No delayed expansion (a `!` in a path stays) and no ( ) blocks around paths (a `)` in one stays).
setlocal
set "OUT=%~dp0.."
for %%i in ("%OUT%") do set "OUT=%%~fi"
set "WORK=%~dp0work"
set "LOG=%WORK%\build_dll.log"
set "LOGFG=%WORK%\build_fg_dll.log"
if not exist "%XESS_SDK%\inc\xess\xess_d3d12.h" goto nosdk
if not exist "%XESS_SDK%\inc\xess_fg\xefg_swapchain_d3d12.h" goto nosdk
if not exist "%XESS_SDK%\lib\libxess.lib" goto nosdk
if not exist "%XESS_SDK%\lib\libxess_fg.lib" goto nosdk
if not exist "%XESS_SDK%\lib\libxell.lib" goto nosdk
rem vcvars64 every time: a Developer Command Prompt's own cl can target x86
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -property installationPath`) do set "VSDIR=%%i"
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
if not exist "%WORK%" mkdir "%WORK%" || exit /b 1
if /i "%~1"=="fg" goto fg

cl /nologo /O2 /EHsc /std:c++17 /W4 /MD /LD /D_WINDOWS /DUNICODE /D_UNICODE /DNOMINMAX /I "%XESS_SDK%\inc" ^
   /Fo"%WORK%\\" /Fe"%WORK%\smv_xess_bridge.dll" "%~dp0xess_bridge.cpp" ^
   /link d3d12.lib dxgi.lib "%XESS_SDK%\lib\libxess.lib" > "%LOG%" 2>&1
if %errorlevel% neq 0 goto failed
copy /y "%WORK%\smv_xess_bridge.dll" "%OUT%\smv_xess_bridge.dll" >nul || exit /b 1
echo dll: cl rc 0 (%LOG%), %OUT%\smv_xess_bridge.dll

:fg
cl /nologo /O2 /EHsc /std:c++17 /W4 /MD /LD /D_WINDOWS /DUNICODE /D_UNICODE /DNOMINMAX /I "%XESS_SDK%\inc" ^
   /Fo"%WORK%\\" /Fe"%WORK%\smv_xefg_bridge.dll" "%~dp0xefg_bridge.cpp" ^
   /link d3d12.lib dxgi.lib user32.lib "%XESS_SDK%\lib\libxess_fg.lib" "%XESS_SDK%\lib\libxell.lib" > "%LOGFG%" 2>&1
if %errorlevel% neq 0 goto failedfg
copy /y "%WORK%\smv_xefg_bridge.dll" "%OUT%\smv_xefg_bridge.dll" >nul || exit /b 1
echo dll: cl rc 0 (%LOGFG%), %OUT%\smv_xefg_bridge.dll
exit /b 0

:nosdk
echo set XESS_SDK (the extracted XeSS SDK 3.0.2 with inc\xess\, inc\xess_fg\, inc\xell\ and lib\libxess.lib,
echo lib\libxess_fg.lib, lib\libxell.lib)
exit /b 1

:failed
set "RC=%errorlevel%"
echo dll: cl rc %RC% (%LOG%)
exit /b %RC%

:failedfg
set "RC=%errorlevel%"
echo dll: cl rc %RC% (%LOGFG%)
exit /b %RC%
