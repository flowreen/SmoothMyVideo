@echo off
rem Side-by-side fps comparison export.
rem Best: select BOTH videos in Explorer (Ctrl+click) and drag the pair onto
rem this file; the lower-fps video takes the left pane automatically, in
rem whatever order the pair arrives. Dropping one file (or double-clicking)
rem also works: the window stays open and asks you to drag the missing video
rem into it. On success the window closes itself; it stays open only when
rem something failed, so the error stays readable.
"%~dp0..\engine\runtime\python.exe" "%~dp0compare.py" %*
if errorlevel 1 pause
