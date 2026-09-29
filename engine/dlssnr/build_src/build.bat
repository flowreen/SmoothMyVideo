@echo off
setlocal enabledelayedexpansion
if "%NGX_SDK%"=="" (
  echo Set NGX_SDK to the NVIDIA DLSS SDK folder that contains include\nvsdk_ngx.h, see DEVELOPMENT.md
  exit /b 1
)
if not exist "%NGX_SDK%\include\nvsdk_ngx.h" (
  echo %NGX_SDK%\include\nvsdk_ngx.h not found, see DEVELOPMENT.md
  exit /b 1
)
where cl >nul 2>nul
if errorlevel 1 (
  for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath`) do set VSDIR=%%i
  call "!VSDIR!\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
)
cd /d %~dp0

rem the NR core (nr_host.cpp) is compiled into smv-live.exe by engine\live\build_src\build.bat
echo caller shim -^> ..\nvngx.dll
cl /nologo /std:c++17 /EHsc /O2 /W3 /LD shim.cpp /I "%NGX_SDK%\include" ^
   /Fo.\ /Fe..\nvngx.dll || exit /b 1

del /q *.obj *.exp 2>nul
del /q "..\nvngx.exp" "..\nvngx.lib" 2>nul
exit /b 0
