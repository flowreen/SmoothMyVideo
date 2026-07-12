@echo off
setlocal
if "%SL_SDK%"=="" (
  echo Set SL_SDK to the extracted Streamline SDK first, see BUILD.md
  exit /b 1
)
where cl >nul 2>nul
if errorlevel 1 (
  for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath`) do set VSDIR=%%i
  call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
)
cd /d %~dp0
cl /nologo /std:c++17 /EHsc /O2 /W3 main.cpp /I "%SL_SDK%\include" ^
   /link /LIBPATH:"%SL_SDK%\lib\x64" sl.interposer.lib user32.lib gdi32.lib ole32.lib windowscodecs.lib dxguid.lib ^
   /SUBSYSTEM:CONSOLE /OUT:..\dlssg2f.exe
exit /b %errorlevel%
