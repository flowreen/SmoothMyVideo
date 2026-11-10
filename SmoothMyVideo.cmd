@echo off
rem Double-click launcher for SmoothMyVideo from its source folder. A folder that is set up starts the app at once:
rem a hidden second run of this file (:run) builds the app and starts it, so no console window stays open. A build
rem that fails opens a window with its last lines (:startfailed) instead of nothing.
rem A fresh download is set up first, in this window: the app's packages (npm install), ffmpeg and NVIDIA's CUDA
rem runtime DLLs. It needs Node.js 24 or later and nothing else. SMV_LAUNCH_DRY=1 does everything but the start;
rem SMV_LAUNCH_ARGS = extra arguments for the app (a test's own profile folder).
setlocal
cd /d "%~dp0"
set "LAUNCH_DIR=%~dp0."
set "LAUNCH_SELF=%~f0"
set "START_LOG=%TEMP%\smv-start.log"
if "%~1"==":run" goto run
if "%~1"==":startfailed" goto startfailed

rem npm's command shims split a path at an ampersand, so nothing can be built from such a folder.
if not "%LAUNCH_DIR:&=%"=="%LAUNCH_DIR%" goto badpath

where node >nul 2>nul
if errorlevel 1 goto nonode
set "NODE_MAJOR=0"
for /f "tokens=1 delims=." %%v in ('node -p "process.versions.node"') do set "NODE_MAJOR=%%v"
if %NODE_MAJOR% LSS 24 goto oldnode

if not exist "node_modules\electron\dist\electron.exe" goto setup
if not exist "node_modules\typescript" goto setup
if not exist "engine\bin\ffmpeg.exe" goto setup
if not exist "engine\gpu_runtime\cudart64_13.dll" goto setup
goto start

:setup
title Smooth My Video: first-time setup
echo.
echo   Smooth My Video: first-time setup of this folder
echo   -------------------------------------------------
echo   Downloading the app's packages, ffmpeg and NVIDIA's CUDA runtime (under 1 GB on disk).
echo   This runs once and takes a few minutes. The app starts by itself when it is done.
echo.
if exist "node_modules\electron\dist\electron.exe" if exist "node_modules\typescript" goto engine
echo   [1/3] the app's packages (npm install) ...
call npm install
if errorlevel 1 goto failed
if not exist "node_modules\electron\dist\electron.exe" goto failed
:engine
echo.
echo   [2/3] ffmpeg ...
call node scripts\fetch-ffmpeg.ts
echo.
echo   [3/3] NVIDIA's CUDA runtime ...
call node scripts\stage-gpu-runtime.ts
echo.
if not exist "engine\bin\ffmpeg.exe" echo   NOTE: ffmpeg did not download. The app starts, and its setup box can install ffmpeg later.
if not exist "engine\gpu_runtime\cudart64_13.dll" echo   NOTE: the CUDA runtime did not download. The app starts, but renders need it: run this file again when online.
echo   Setup done. Starting Smooth My Video ...
if not exist "engine\bin\ffmpeg.exe" timeout /t 8 >nul
if not exist "engine\gpu_runtime\cudart64_13.dll" timeout /t 8 >nul

:start
if defined SMV_LAUNCH_DRY echo   SMV_LAUNCH_DRY: the app would start now.
if defined SMV_LAUNCH_DRY goto end
powershell -NoProfile -WindowStyle Hidden -Command "Start-Process -FilePath cmd.exe -ArgumentList '/c call \"%%LAUNCH_SELF%%\" :run' -WorkingDirectory $env:LAUNCH_DIR -WindowStyle Hidden"
goto end

rem The hidden half of a start: the build, then the app. Only the build's result is judged: a failed build opens the
rem :startfailed window, the app's own exit code never does.
:run
call npm run build > "%START_LOG%" 2>&1
if errorlevel 1 goto runfailed
start "" "node_modules\electron\dist\electron.exe" . %SMV_LAUNCH_ARGS%
goto end

:runfailed
start "Smooth My Video: start failed" cmd /c call "%LAUNCH_SELF%" :startfailed
goto end

:startfailed
title Smooth My Video: start failed
echo.
echo   Smooth My Video did not start: building it from this folder failed.
echo   The build's last lines:
echo.
powershell -NoProfile -Command "Get-Content -LiteralPath $env:START_LOG -Tail 25"
echo.
echo   The whole text: %START_LOG%
echo   If a file in this folder was changed, undo the change or download the folder again.
echo   The ready-made app needs no build:
echo     https://github.com/flowreen/SmoothMyVideo/releases/latest
echo.
pause
goto end

:badpath
echo.
echo   Smooth My Video cannot run from this folder: its path holds the character ^&
echo     "%LAUNCH_DIR%"
echo   The tools that build the app cannot handle that character. Move or rename the folder
echo   so that its path has no ^&, then double-click this file again.
echo.
echo   The ready-made app runs from any folder:
echo     https://github.com/flowreen/SmoothMyVideo/releases/latest
echo.
pause
goto end

:nonode
echo.
echo   Smooth My Video: this folder is the SOURCE CODE, and it needs Node.js to run.
echo.
echo   The ready-made app needs nothing installed. Download SmoothMyVideo-win.7z here:
echo     https://github.com/flowreen/SmoothMyVideo/releases/latest
echo.
echo   To run from this folder instead: install Node.js 24 or later from https://nodejs.org
echo   then double-click this file again.
echo.
pause
goto end

:oldnode
echo.
echo   Smooth My Video needs Node.js 24 or later; this PC has version %NODE_MAJOR%.
echo   Install the current Node.js from https://nodejs.org then double-click this file again.
echo.
echo   The ready-made app needs nothing installed:
echo     https://github.com/flowreen/SmoothMyVideo/releases/latest
echo.
pause
goto end

:failed
echo.
echo   Setup stopped: the app's packages did not install (see the lines above).
echo   Check the internet connection, then double-click this file again.
echo.
pause

:end
endlocal
