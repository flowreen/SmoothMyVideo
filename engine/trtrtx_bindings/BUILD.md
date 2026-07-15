# Custom `tensorrt_rtx` bindings for CPython 3.14

The bundled runtime is Python 3.14, but NVIDIA ships `tensorrt_rtx` wheels only for
cp38..cp313 (and publishes no binding source - github.com/NVIDIA/TensorRT-RTX is
samples-only). `bindings.cpp` is a ~250-line pybind11 module that reimplements exactly
the API subset `engine/trt_runtime.py` uses, with the official wheel's names and
semantics, so `import tensorrt_rtx as trt` works unchanged. Validated against the
official-wheel behaviour: same engine files, same JIT-cache mechanism, TF32/fp16
numerics (see the covered-surface list at the top of `bindings.cpp`).

**Retire this the day NVIDIA ships a cp314 `tensorrt_rtx_cu13_bindings` wheel on PyPI**:
delete `tensorrt_rtx.pyd` + the three DLLs from site-packages, `pip install
tensorrt-rtx-cu13`, and remove this directory.

## Building

1. Download the TensorRT-RTX **1.5.0.114** SDK zip (Windows, CUDA 13) from
   developer.nvidia.com and extract it anywhere.
2. From a normal cmd prompt:

```
engine\trtrtx_bindings\build.cmd <path-to-extracted-SDK>
```

The script pip-installs pybind11 + `nvidia-cuda-crt` (CUDA headers come from the pip
wheels; no CUDA Toolkit install needed), compiles with MSVC (VS 2026 vcvars64, edit
`VSVARS` in the script for other versions), and drops `tensorrt_rtx.pyd` plus its three
DLL dependencies (`tensorrt_rtx_1_5.dll`, `tensorrt_onnxparser_rtx_1_5.dll`,
`cudart64_13.dll`) into `engine/runtime/Lib/site-packages` - CPython resolves a .pyd's
DLLs from the module's own directory.

## Gotchas
* The runtime wheel's top-level `host_defines.h` is only a deprecation forwarder to
  `crt/host_defines.h`, which ships in the separate `nvidia-cuda-crt` wheel.
* pybind11 requires enums to be registered before use as argument defaults (the Logger
  severity default trips this as an `ImportError` at import time).
* A TRT-RTX version bump changes the `.lib`/`.dll` names (`tensorrt_rtx_1_5`) and the
  engine-cache tag (`trt_runtime._trt_tag`); rebuild against the matching SDK.
