@echo off
rem Builds engine\update\smv-update.exe (smv_update.cpp: the automatic update's swap helper, a GUI-subsystem exe the app
rem copies to %TEMP% and runs as it exits; docs\DEVELOPMENT.md). Static CRT: nothing to find beside it in %TEMP%.
rem Then packs engine\update\probe.7z with the 7zr.exe beside it, the release archive's way (LZMA2 level 9, solid, BCJ2
rem on the exe): the app unpacks it before any download to prove the unpacker runs on this PC.
rem Usage (Visual Studio 2026 with the C++ workload): build.bat
rem No delayed expansion (a `!` in a path stays) and no ( ) blocks around paths (a `)` in one stays).
setlocal
set "OUT=%~dp0.."
for %%i in ("%OUT%") do set "OUT=%%~fi"
set "WORK=%~dp0work"
set "LOG=%WORK%\build_exe.log"
rem a shell with cl already set up (the host's build environment) builds as it is; any other finds VS and calls vcvars64
where cl >nul 2>&1 && goto havecl
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -property installationPath`) do set "VSDIR=%%i"
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
:havecl
if not exist "%WORK%" mkdir "%WORK%" || exit /b 1

cl /nologo /O2 /EHsc /std:c++17 /W4 /WX /analyze /MT /D_WINDOWS /DUNICODE /D_UNICODE /DNOMINMAX /Fo"%WORK%\\" ^
   /Fe"%WORK%\smv-update.exe" "%~dp0smv_update.cpp" /link /SUBSYSTEM:WINDOWS shell32.lib > "%LOG%" 2>&1
if %errorlevel% neq 0 goto failed
copy /y "%WORK%\smv-update.exe" "%OUT%\smv-update.exe" >nul || exit /b 1
echo exe: cl rc 0 (%LOG%), %OUT%\smv-update.exe
if exist "%OUT%\probe.7z" del "%OUT%\probe.7z"
"%OUT%\7zr.exe" a -t7z -m0=lzma2 -mx=9 -ms=on -bd -bso0 -bsp0 "%OUT%\probe.7z" "%OUT%\smv-update.exe" >> "%LOG%" 2>&1
if %errorlevel% neq 0 goto failed
echo probe: %OUT%\probe.7z
exit /b 0

:failed
set "RC=%errorlevel%"
echo exe: cl rc %RC% (%LOG%)
exit /b %RC%
