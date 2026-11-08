@echo off
rem Double-click launcher for SmoothMyVideo: builds (tsc) and opens the Electron app. PowerShell (Microsoft's
rem replacement for VBScript) starts `npm start` hidden from this folder, so no console window stays open.
set "LAUNCH_DIR=%~dp0."
powershell -NoProfile -WindowStyle Hidden -Command "Start-Process -FilePath cmd.exe -ArgumentList '/c npm start' -WorkingDirectory $env:LAUNCH_DIR -WindowStyle Hidden"
