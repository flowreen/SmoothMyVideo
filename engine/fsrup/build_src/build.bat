@echo off
rem Builds engine\fsrup\smv_fsrup_bridge.dll (fsrup_bridge.cpp: the host's C API over AMD's FidelityFX upscaler) and copies
rem AMD's signed FidelityFX loader + upscaler DLLs from the FSR SDK v2.3.0 beside it (both MIT: Kits/FidelityFX/docs/license.md
rem lists Kits\FidelityFX\signedbin\amd_fidelityfx_loader_dx12.dll and amd_fidelityfx_upscaler_dx12.dll in its MIT section).
rem
rem Environment:
rem   FSR_SDK  the FSR SDK v2.3.0 checkout (holds Kits\FidelityFX: api\include, upscalers\include, signedbin)
rem Usage (Visual Studio 2026 with the C++ workload): build.bat
rem No delayed expansion (a `!` in a path stays) and no ( ) blocks around paths (a `)` in one stays).
setlocal
set "OUT=%~dp0.."
for %%i in ("%OUT%") do set "OUT=%%~fi"
set "SDK=%FSR_SDK%\Kits\FidelityFX"
set "WORK=%~dp0work"
set "LOG=%WORK%\build_dll.log"
if not exist "%SDK%\signedbin\amd_fidelityfx_loader_dx12.dll" goto nosdk
if not exist "%SDK%\signedbin\amd_fidelityfx_upscaler_dx12.dll" goto nosdk
rem vcvars64 every time: a Developer Command Prompt's own cl can target x86
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -property installationPath`) do set "VSDIR=%%i"
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
if not exist "%WORK%" mkdir "%WORK%" || exit /b 1

cl /nologo /O2 /EHsc /std:c++17 /W3 /MD /LD /D_WINDOWS /DUNICODE /D_UNICODE /DNOMINMAX /I "%SDK%\api\include" ^
   /I "%SDK%\upscalers\include" /Fo"%WORK%\\" /Fe"%WORK%\smv_fsrup_bridge.dll" "%~dp0fsrup_bridge.cpp" ^
   /link d3d12.lib dxgi.lib > "%LOG%" 2>&1
if %errorlevel% neq 0 goto failed
copy /y "%WORK%\smv_fsrup_bridge.dll" "%OUT%\smv_fsrup_bridge.dll" >nul || exit /b 1
copy /y "%SDK%\signedbin\amd_fidelityfx_loader_dx12.dll" "%OUT%\amd_fidelityfx_loader_dx12.dll" >nul || exit /b 1
copy /y "%SDK%\signedbin\amd_fidelityfx_upscaler_dx12.dll" "%OUT%\amd_fidelityfx_upscaler_dx12.dll" >nul || exit /b 1
echo dll: cl rc 0 (%LOG%), %OUT% (+ the two signed AMD DLLs)
exit /b 0

:nosdk
echo set FSR_SDK (the FSR SDK v2.3.0 checkout with Kits\FidelityFX\signedbin)
exit /b 1

:failed
set "RC=%errorlevel%"
echo dll: cl rc %RC% (%LOG%)
exit /b %RC%
