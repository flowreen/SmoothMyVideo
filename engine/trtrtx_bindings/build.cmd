@echo off
rem Build tensorrt_rtx.pyd (custom CPython 3.14 bindings) - see BUILD.md.
rem Set these three paths for your machine before running from a normal cmd prompt:
setlocal
set SDK=%~1
set PYDIR=%~dp0..\runtime
set VSVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat
if "%SDK%"=="" echo usage: build.cmd ^<TensorRT-RTX SDK dir^> & exit /b 1

set SRC=%~dp0
set SITE=%PYDIR%\Lib\site-packages
rem CUDA headers from the pip wheels (no CUDA Toolkit needed): runtime headers + crt/ from nvidia-cuda-crt
"%PYDIR%\python.exe" -m pip install --quiet pybind11 nvidia-cuda-crt || exit /b 1
set CUDAINC=%SITE%\nvidia\cu13\include

call "%VSVARS%" >nul || exit /b 1
cl /nologo /std:c++17 /EHsc /O2 /MD /LD /bigobj /utf-8 ^
   /I"%PYDIR%\include" /I"%SITE%\pybind11\include" /I"%SDK%\include" /I"%CUDAINC%" ^
   "%SRC%bindings.cpp" ^
   /Fo"%SRC%bindings.obj" /Fe"%SITE%\tensorrt_rtx.pyd" ^
   /link /IMPLIB:"%SRC%tensorrt_rtx.lib" /LIBPATH:"%PYDIR%\libs" /LIBPATH:"%SDK%\lib" ^
   tensorrt_rtx_1_5.lib tensorrt_onnxparser_rtx_1_5.lib python314.lib || exit /b 1
copy /y "%SDK%\bin\tensorrt_rtx_1_5.dll" "%SITE%" >nul
copy /y "%SDK%\bin\tensorrt_onnxparser_rtx_1_5.dll" "%SITE%" >nul
copy /y "%SITE%\nvidia\cu13\bin\x86_64\cudart64_13.dll" "%SITE%" >nul
echo built and installed into %SITE%
exit /b 0
