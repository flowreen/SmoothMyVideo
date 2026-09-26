/* Build shim.
 *
 * The CUDA 13 runtime WHEEL that ships inside engine\runtime carries the public CUDA headers
 * but not the internal crt\ subdirectory, and cuda_runtime_api.h opens with
 * #include "crt/host_defines.h". smv-live compiles for the HOST only (every kernel is built
 * by NVRTC at run time, there is no nvcc anywhere in this project), so the only thing that
 * header has to provide is the attribute macro set, all of which are empty or trivial off the
 * CUDA compiler. This file is that set, and nothing else in crt\ is reachable from the
 * headers this exe includes (cuda_runtime_api.h, cuda_d3d11_interop.h, cuda.h, nvrtc.h).
 *
 * Do NOT extend this into a general CUDA toolkit stand-in: if a future include needs more of
 * crt\, install the real toolkit headers instead of growing the shim.
 */
#ifndef SMV_CRT_HOST_DEFINES_H
#define SMV_CRT_HOST_DEFINES_H

#define __no_return__
#define __noinline__
#define __forceinline__ __forceinline
#define __align__(n) __declspec(align(n))
#define __thread__ __declspec(thread)
#define __import__
#define __export__
#define __annotate__(a)
#define __location__(a)
#define __specialization_static static

#define CUDARTAPI __stdcall
#define CUDARTAPI_CDECL __cdecl
#define CUDAAPI __stdcall

#define __host__
#define __device__
#define __global__
#define __shared__
#define __constant__
#define __managed__
#define __grid_constant__
#define __restrict__

#define __device_builtin__
#define __device_builtin_texture_type__
#define __device_builtin_surface_type__
#define __cudart_builtin__
#define __inline_hint__
#define __device_builtin_forceinline__ __forceinline

#define __CUDA_DEVICE_BUILTIN(a)

#define __builtin_align__(a) __align__(a)

#endif /* SMV_CRT_HOST_DEFINES_H */
