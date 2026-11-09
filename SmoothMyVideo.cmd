@echo off
rem Double-click launcher for SmoothMyVideo from its source folder. A folder that is set up starts the app at once:
rem PowerShell starts `npm start` (build, then the Electron app) hidden, so no console window stays open.
rem A fresh download is set up first, in this window: the app's packages (npm install), ffmpeg and NVIDIA's CUDA
rem runtime DLLs. It needs Node.js 24 or later and nothing else. SMV_LAUNCH_DRY=1 does everything but the start.
setlocal
cd /d "%~dp0"
set "LAUNCH_DIR=%~dp0."

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
powershell -NoProfile -WindowStyle Hidden -Command "Start-Process -FilePath cmd.exe -ArgumentList '/c npm start' -WorkingDirectory $env:LAUNCH_DIR -WindowStyle Hidden"
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
