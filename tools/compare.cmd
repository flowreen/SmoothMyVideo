@echo off
rem Side-by-side fps comparison export.
rem Best: select BOTH videos in Explorer (Ctrl+click) and drag the pair onto
rem this file. Dropping one file (or double-clicking) also works: the window
rem stays open and asks you to drag the missing video into it.
"%~dp0..\engine\runtime\python.exe" "%~dp0compare.py" %*
pause
