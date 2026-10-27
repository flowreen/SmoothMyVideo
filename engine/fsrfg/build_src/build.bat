@echo off
rem Builds engine\fsrfg: smv_fsrfg_bridge.dll (fsrfg_bridge.cpp) and amdxc64.dll (amdxc64_shim.cpp), and copies beside
rem them AMD's signed FSR SDK v2.3.0 loader + frame generation DLLs (signedbin, unmodified) and vkd3d-proton's two DLLs.
rem
rem vkd3d-proton (LGPL-2.1, built separately, MSYS2 UCRT64 with gcc, meson and ninja): github.com/HansKristian-Work/
rem vkd3d-proton at commit f5443fe51c99363ad02a3e7dbead2f3ec25f6ded with its submodules, dxil-spirv at
rem 8b48f0e0392d76be4fdcea2a3ba3ea99dddaae5b, then
rem   git apply ..\source\vkd3d-proton-f5443fe-smv.patch            (in the vkd3d-proton checkout)
rem   git apply ..\source\dxil-spirv-8b48f0e-smv.patch              (in subprojects\dxil-spirv)
rem   meson setup build --buildtype=release -Denable_extended_emulation=true
rem   ninja -C build
rem The patches: VKD3D_FP8_EMULATION=1 runs the FP8 cooperative matrices of AMD's ML shaders as FP16 ones, dxil-spirv's
rem DXIL_SPIRV_CONFIG=wmma_fp8_staging loads FP8 tiles as coopmat2 tensor loads with a decode function where the device
rem has VK_NV_cooperative_matrix2 tensor addressing + block loads (else through shared memory as FP16, like the stores) and
rem reads its configuration from the process environment, and the two DLLs are named smv_vkd3d_d3d12.dll /
rem smv_vkd3d_d3d12core.dll.
rem
rem Environment:
rem   FSR_SDK      the FSR SDK v2.3.0 checkout (holds Kits\FidelityFX: api\include, framegeneration\include, signedbin)
rem   VKD3D_BUILD  the vkd3d-proton build folder above (holds libs\d3d12\smv_vkd3d_d3d12.dll, libs\d3d12core\...)
rem Usage (Visual Studio 2026 with the C++ workload): build.bat [output folder, default engine\fsrfg]
rem No delayed expansion (a `!` in a path stays) and no ( ) blocks around paths (a `)` in one stays).
setlocal
set "OUT=%~1"
if "%OUT%"=="" set "OUT=%~dp0.."
for %%i in ("%OUT%") do set "OUT=%%~fi"
set "SDK=%FSR_SDK%\Kits\FidelityFX"
set "WORK=%~dp0work"
set "LOG=%WORK%\build_dll.log"
set "VKD3D_D3D12=%VKD3D_BUILD%\libs\d3d12\smv_vkd3d_d3d12.dll"
set "VKD3D_CORE=%VKD3D_BUILD%\libs\d3d12core\smv_vkd3d_d3d12core.dll"
set "AMD_LOADER=%SDK%\signedbin\amd_fidelityfx_loader_dx12.dll"
set "AMD_FG=%SDK%\signedbin\amd_fidelityfx_framegeneration_dx12.dll"
set "MISSING="
if not exist "%VKD3D_D3D12%" set "MISSING=%MISSING% %VKD3D_D3D12%"
if not exist "%VKD3D_CORE%" set "MISSING=%MISSING% %VKD3D_CORE%"
if not exist "%AMD_LOADER%" set "MISSING=%MISSING% %AMD_LOADER%"
if not exist "%AMD_FG%" set "MISSING=%MISSING% %AMD_FG%"
if defined MISSING goto missing
rem vcvars64 every time: a Developer Command Prompt's own cl can target x86
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -property installationPath`) do set "VSDIR=%%i"
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
if exist "%WORK%" rmdir /s /q "%WORK%"
mkdir "%WORK%" || exit /b 1
if not exist "%OUT%" mkdir "%OUT%" || exit /b 1

cl /nologo /O2 /EHsc /W3 /MT /LD /D_WINDOWS /DNOMINMAX /DWIN32_LEAN_AND_MEAN /Fo"%WORK%\\" /std:c++20 ^
   /I "%SDK%\api\include" /I "%SDK%\framegeneration\include" /Fe"%WORK%\bridge.dll" "%~dp0fsrfg_bridge.cpp" ^
   /link dxgi.lib > "%LOG%" 2>&1
if %errorlevel% neq 0 goto failed
cl /nologo /O2 /EHsc /W3 /MT /LD /D_WINDOWS /DNOMINMAX /DWIN32_LEAN_AND_MEAN /Fo"%WORK%\\" /std:c++17 ^
   /Fe"%WORK%\amdxc64.dll" "%~dp0amdxc64_shim.cpp" >> "%LOG%" 2>&1
if %errorlevel% neq 0 goto failed
echo dll: cl rc 0 (%LOG%)

copy /y "%WORK%\bridge.dll" "%OUT%\smv_fsrfg_bridge.dll" >nul || exit /b 1
copy /y "%WORK%\amdxc64.dll" "%OUT%\amdxc64.dll" >nul || exit /b 1
for %%f in ("%VKD3D_D3D12%" "%VKD3D_CORE%" "%AMD_LOADER%" "%AMD_FG%") do copy /y "%%~f" "%OUT%\%%~nxf" >nul || exit /b 1
echo installed into %OUT%: smv_fsrfg_bridge.dll, amdxc64.dll, smv_vkd3d_d3d12.dll, smv_vkd3d_d3d12core.dll,
echo   amd_fidelityfx_loader_dx12.dll, amd_fidelityfx_framegeneration_dx12.dll
exit /b 0

:missing
echo set FSR_SDK (the FSR SDK v2.3.0 checkout) and VKD3D_BUILD (the vkd3d-proton build folder); missing:%MISSING%
exit /b 1

:failed
set "RC=%errorlevel%"
echo dll: cl rc %RC% (%LOG%)
exit /b %RC%
