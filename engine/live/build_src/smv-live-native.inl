// part of smv-live.cpp (unity include, see the file map there); not a standalone translation unit
// ---------------------------------------------------------------- the native model host
//
// Every live model runs INSIDE this process: the host's own CUDA kernels for every net (NVRTC-compiled,
// their cubins cached by source hash), the driver's Optical Flow API for nvof, the shipped bridge DLL for
// fruc.
//
// NativeRife sits behind the PipeServer boundary: writeFull() hands it one group message per
// captured frame ([u32 N][N f32 fractions]) and readFullRaw() hands back the tokens the
// reader thread consumes (slot (i+1), the capture-release token 0x7FFFFFFF, the bare
// 0x80000000 end marker).
//
// Everything CUDA happens on ONE compute thread and ONE non-default stream:
//   * the default stream is never used (it synchronises with every other stream, a
//     permanent-hang risk);
//   * the kernel chains replay as CUDA graphs captured on that stream (chainGraphRun).
//
// cudart64_13.dll and nvrtc64_130_0.dll ship in engine\gpu_runtime (staged from NVIDIA's
// two pinned wheels by scripts/stage-gpu-runtime.ts).
//
// The code lives in parts, smv-live-native-<name>.inl, #included further down in source order (the repo's
// CLAUDE.md lists them), one concern each: a vendor bridge, one ported net (its kernel text, loader and launch
// code), one stage of a session. This file keeps the shared kernel text kNativeKernels, the CUDA runtime's entry
// points and the small glue between parts. The order matters: a later part uses names an earlier one defines.

// cuda_runtime.h pulls the internal crt\ headers the runtime WHEEL does not ship; the host
// side only ever needs the API declarations, so cuda_runtime_api.h plus the one-macro shim in
// cuda_shim\crt is the whole build dependency (see that file).
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuda_d3d11_interop.h>
#include <nvrtc.h>
#include "nvOpticalFlowCuda.h" // the nvof model: MIT interface headers in build_src\nvofa

// ---- the two device kernels, JIT-compiled by NVRTC at first native start ------------------
// Half-to-float is done with the PTX instruction rather than cuda_fp16.h so the kernel source
// needs no include path at all. The planes are (R, G, B) on every route: the live SDR packers
// read the BGRA capture bytes in reverse and its slot stores write them back as BGRA.
static const char kNativeKernels[] = R"CUDASRC(
extern "C" {

__device__ __forceinline__ float h2f(unsigned short h)
{
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

// ---- every resize the host does itself (RTX VSR does the others): Lanczos -------------------
// The radius a (the lobes) per axis: kLzUp on an axis that enlarges or keeps its size, kLzDown on
// one that shrinks = the universal resizer test's picks, the best no-sharpen PSNR over anime, real
// camera and CGI at every ratio (up) and working size (down). These defines are the one place they
// are set: the cubin cache key hashes this text, the host reads them for the tap tables' sizes
// (lzTaps), a harness may pass -D to try others (sizing its own tables to match).
#ifndef kLzUp
#define kLzUp 4
#endif
#ifndef kLzDown
#define kLzDown 7
#endif
// sinc(x) sinc(x / a) for |x| < a, placed the way zimg places it: output index o of an in -> out
// axis sits at (o + 0.5) in / out on the input grid, a shrinking axis widens the filter by in /
// out (the antialiasing), the fs = 2 ceil(a max(in, out) / out) taps start at round_halfup(that
// position - fs / 2), a tap past an edge reads the mirrored pixel, and the taps are normalised by
// their sum. The window start is an integer floor division and every tap argument ((2j + 1) out -
// (2o + 1) in) / (2 max(in, out)) comes from integers.
__device__ __forceinline__ float lanczosA(float x, int a)
{
    x = fabsf(x);
    if (x >= (float)a) return 0.0f;
    if (x == 0.0f) return 1.0f;
    const float px = 3.14159265358979f * x;
    return (float)a * sinpif(x) * sinpif(x / (float)a) / (px * px);
}
__device__ __forceinline__ int rsTaps(int in, int out)
{
    return in > out ? 2 * (int)(((long long)kLzDown * in + out - 1) / out) : 2 * kLzUp;
}
__device__ __forceinline__ int rsBegin(int o, int in, int out, int fs)
{
    const long long num = (2LL * o + 1) * in - (long long)(fs - 1) * out, den = 2LL * out;
    long long q = num / den;
    if (num - q * den < 0) q--;   // floor: the numerator is negative near the left edge
    return (int)q;
}
__device__ __forceinline__ float rsArg(int j, int o, int in, int out)
{
    const long long n = (2LL * j + 1) * out - (2LL * o + 1) * in, d = 2LL * (in >= out ? in : out);
    const long long q = n / d;   // a small whole part plus an exact remainder: one fp32 division
    return (float)q + (float)(n - q * d) / (float)d;
}
// the unnormalised weight of tap j for output index o, at the axis's radius
__device__ __forceinline__ float rsW(int j, int o, int in, int out)
{
    return lanczosA(rsArg(j, o, in, out), in > out ? kLzDown : kLzUp);
}
// a tap past an edge reads the mirrored pixel (half-sample symmetric), clamped for a window wider
// than the axis
__device__ __forceinline__ int rsMirror(int j, int n)
{
    if (j < 0) j = -j - 1;
    if (j >= n) j = 2 * n - j - 1;
    return j < 0 ? 0 : (j >= n ? n - 1 : j);
}
// 1 / the sum of output index o's fs taps from mn; a tap's weight is rsW times this
__device__ __forceinline__ float rsInv(int o, int in, int out, int mn, int fs)
{
    float s = 0.0f;
    for (int k = 0; k < fs; k++) s += rsW(mn + k, o, in, out);
    return 1.0f / s;
}
// one resize axis's taps (in -> out), built once by k_rsTab and read by every resize kernel: tap k of output index o
// at k * out + o = its normalised weight (rsW times rsInv, the rounded product) and its mirrored input index, so a
// pixel reads its taps instead of evaluating fs sines; every ratio fits (the table lives in global memory)
struct __align__(8) RsTap
{
    float w;
    int i;
};
__global__ void k_rsTab(int in, int out, RsTap* __restrict__ tab)
{
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= out) return;
    const int fs = rsTaps(in, out);
    const int mn = rsBegin(o, in, out, fs);
    const float inv = rsInv(o, in, out, mn, fs);
    for (int k = 0; k < fs; k++)
    {
        RsTap t;
        t.w = __fmul_rn(rsW(mn + k, o, in, out), inv);
        t.i = rsMirror(mn + k, in);
        tab[(size_t)k * out + o] = t;
    }
}
// ---- no fp64 in these kernels (consumer GPUs run fp64 at a small fraction of fp32). Where
// accuracy would call for double, fp32 keeps it by construction: tap centres are integer ratios (rsBegin / rsArg, nvUpTap), a
// landing point is the pixel plus the floor and the exact fraction of its offset (land, nvAxis),
// a hole or mask decision reads the integer accumulator (DRBA_HOLE, k_velNorm), and the one
// rounding that feeds a discontinuous consumer (k_nvofLuma -> the Optical Flow Accelerator)
// carries its error in a second float.
// a * b = p + e and a + b = s + e EXACTLY in fp32 (FMA TwoProduct, Knuth TwoSum; the _rn
// intrinsics keep the compiler from contracting the rounded product or sum into an FMA)
__device__ __forceinline__ void ffMul(float a, float b, float& p, float& e)
{
    p = __fmul_rn(a, b);
    e = fmaf(a, b, -p);
}
__device__ __forceinline__ void ffAdd(float a, float b, float& s, float& e)
{
    s = __fadd_rn(a, b);
    const float bb = __fsub_rn(s, a);
    e = __fadd_rn(__fsub_rn(a, __fsub_rn(s, bb)), __fsub_rn(b, bb));
}
// the landing of pixel x moved by a * b px: the cell n and the bilinear fraction l in [0, 1],
// from the EXACT product p + e: n = x + floor(p), l = (p - floor(p)) + e (p - floor(p) is exact
// outside (-1, 0), within 6e-8 inside), so l is within one fp32 ulp of the fp64 form's x + a * b;
// an fp32 x + a * b carries one ulp of x (1.2e-4 px at x ~ 2000) and the rounded product alone
// 1.7e-5 px at 280 px, which DRBA's timestep map read as 1e-4. false (and no landing) for a
// product of 1e8 px or more or a non-finite one.
__device__ __forceinline__ bool land(int x, float a, float b, int& n, float& l)
{
    float p, e;
    ffMul(a, b, p, e);
    if (!(fabsf(p) < 1e8f)) { n = 0; l = 0.0f; return false; }
    const float fl = floorf(p);
    n = x + (int)fl;
    l = (p - fl) + e;
    if (l < 0.0f) { l += 1.0f; n--; }
    return true;
}

// scale == 1.0 fast path: BGRA8 -> planar (R, G, B) float [0,1], replicate-padded to (ph, pw).
__global__ void k_packInDirect(const unsigned char* __restrict__ src, int cw, int ch,
                               float* __restrict__ dst, int ph, int pw, int planeStride)
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= pw || oy >= ph) return;
    const int sx = ox < cw ? ox : cw - 1;          // replicate pad
    const int sy = oy < ch ? oy : ch - 1;
    const unsigned char* p = src + ((size_t)sy * cw + sx) * 4;
    const int o = oy * pw + ox;
    dst[o] = p[2] * (1.0f / 255.0f);
    dst[planeStride + o] = p[1] * (1.0f / 255.0f);
    dst[2 * planeStride + o] = p[0] * (1.0f / 255.0f);
}

// the live capture resize to the working size, horizontal pass: BGRA8 (cw x ch) -> planar (R, G, B) float (3, ch, w),
// Lanczos with the placement above (it shrinks, or enlarges when the working size exceeds the capture)
__global__ void k_resizeH(const unsigned char* __restrict__ src, int cw, int ch,
                          float* __restrict__ tmp, int w, const RsTap* __restrict__ tab)
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int sy = blockIdx.y * blockDim.y + threadIdx.y;
    const int fs = rsTaps(cw, w);
    if (ox >= w || sy >= ch) return;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
    for (int k = 0; k < fs; k++)
    {
        const RsTap t = tab[(size_t)k * w + ox];
        const float wt = t.w;
        const int ix = t.i;
        const unsigned char* p = src + ((size_t)sy * cw + ix) * 4;
        a0 += wt * p[0];
        a1 += wt * p[1];
        a2 += wt * p[2];
    }
    const size_t plane = (size_t)ch * w;
    const size_t o = (size_t)sy * w + ox;
    tmp[o] = a2 * (1.0f / 255.0f);
    tmp[plane + o] = a1 * (1.0f / 255.0f);
    tmp[2 * plane + o] = a0 * (1.0f / 255.0f);
}

// vertical pass plus the replicate pad, writing straight into the (3, ph, pw) model half; Lanczos
// rings slightly past [0, 1] at hard edges, so the model input is clamped back into range
__global__ void k_resizeV(const float* __restrict__ tmp, int w, int ch,
                          float* __restrict__ dst, int h, int ph, int pw, int planeStride,
                          const RsTap* __restrict__ tab)
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    const int fs = rsTaps(ch, h);
    if (ox >= pw || oy >= ph) return;
    const int tx = ox < w ? ox : w - 1;            // replicate pad in x
    const int ty = oy < h ? oy : h - 1;            // replicate pad in y
    const size_t plane = (size_t)ch * w;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
    for (int k = 0; k < fs; k++)
    {
        const RsTap t = tab[(size_t)k * h + ty];
        const float wt = t.w;
        const int iy = t.i;
        const size_t o = (size_t)iy * w + tx;
        a0 += wt * tmp[o];
        a1 += wt * tmp[plane + o];
        a2 += wt * tmp[2 * plane + o];
    }
    const int o = oy * pw + ox;
    dst[o] = a0 < 0.0f ? 0.0f : (a0 > 1.0f ? 1.0f : a0);
    dst[planeStride + o] = a1 < 0.0f ? 0.0f : (a1 > 1.0f ? 1.0f : a1);
    dst[2 * planeStride + o] = a2 < 0.0f ? 0.0f : (a2 > 1.0f ? 1.0f : a2);
}

// ---- HDR. Verbatim math of live_server.py _scrgb_to_pq2020 (~48-64) and
// _pack_r10a2 (~66-73): the BT.2087 709 to 2020 matrix runs FIRST, the >= 0 clamp is AFTER it
// (negative scRGB is valid wide-gamut colour that only goes out of range once in BT.2020),
// scRGB 1.0 = 80 nits against PQ 1.0 = 10000, then the ST 2084 inverse EOTF. The capture
// texture is R16G16B16A16_FLOAT, so the channels come in as (R,G,B); the SDR packers read
// the BGRA bytes in reverse for the same plane order.
__device__ __forceinline__ void scrgb_to_pq2020(float r, float g, float b,
                                                float& o0, float& o1, float& o2)
{
    const float M1 = 0.1593017578125f, M2 = 78.84375f;
    const float C1 = 0.8359375f, C2 = 18.8515625f, C3 = 18.6875f;
    float l0 = 0.6274f * r + 0.3293f * g + 0.0433f * b;
    float l1 = 0.0691f * r + 0.9195f * g + 0.0114f * b;
    float l2 = 0.0164f * r + 0.0880f * g + 0.8956f * b;
    l0 = (l0 < 0.0f ? 0.0f : l0) * (80.0f / 10000.0f);
    l1 = (l1 < 0.0f ? 0.0f : l1) * (80.0f / 10000.0f);
    l2 = (l2 < 0.0f ? 0.0f : l2) * (80.0f / 10000.0f);
    const float p0 = powf(l0, M1), p1 = powf(l1, M1), p2 = powf(l2, M1);
    o0 = powf((C1 + C2 * p0) / (1.0f + C3 * p0), M2);
    o1 = powf((C1 + C2 * p1) / (1.0f + C3 * p1), M2);
    o2 = powf((C1 + C2 * p2) / (1.0f + C3 * p2), M2);
}

// HDR, image scale 1.00: FP16 scRGB -> planar PQ fp32, replicate-padded to (ph, pw).
__global__ void k_packInDirectHdr(const unsigned short* __restrict__ src, int cw, int ch,
                                  float* __restrict__ dst, int ph, int pw, int planeStride)
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= pw || oy >= ph) return;
    const int sx = ox < cw ? ox : cw - 1;
    const int sy = oy < ch ? oy : ch - 1;
    const unsigned short* p = src + ((size_t)sy * cw + sx) * 4;
    float a, b, c;
    scrgb_to_pq2020(h2f(p[0]), h2f(p[1]), h2f(p[2]), a, b, c);
    const int o = oy * pw + ox;
    dst[o] = a;
    dst[planeStride + o] = b;
    dst[2 * planeStride + o] = c;
}

// HDR, image scale under 1.00: convert the WHOLE capture to planar PQ first, because the HDR
// fill rule rescales ON PQ, then the shared resize pair runs on those values.
__global__ void k_pqPlanar(const unsigned short* __restrict__ src, int cw, int ch,
                           float* __restrict__ dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cw || y >= ch) return;
    const unsigned short* p = src + ((size_t)y * cw + x) * 4;
    float a, b, c;
    scrgb_to_pq2020(h2f(p[0]), h2f(p[1]), h2f(p[2]), a, b, c);
    const size_t plane = (size_t)ch * cw;
    const size_t o = (size_t)y * cw + x;
    dst[o] = a;
    dst[plane + o] = b;
    dst[2 * plane + o] = c;
}

// the horizontal pass of an ALREADY planar float source (the HDR input path); the same taps as
// k_resizeH, which reads BGRA8 instead.
__global__ void k_resizeHf(const float* __restrict__ src, int cw, int ch,
                           float* __restrict__ tmp, int w, const RsTap* __restrict__ tab)
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int sy = blockIdx.y * blockDim.y + threadIdx.y;
    const int fs = rsTaps(cw, w);
    if (ox >= w || sy >= ch) return;
    const size_t splane = (size_t)ch * cw;
    const size_t dplane = (size_t)ch * w;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
    for (int k = 0; k < fs; k++)
    {
        const RsTap t = tab[(size_t)k * w + ox];
        const float wt = t.w;
        const int ix = t.i;
        const size_t o = (size_t)sy * cw + ix;
        a0 += wt * src[o];
        a1 += wt * src[splane + o];
        a2 += wt * src[2 * splane + o];
    }
    const size_t o = (size_t)sy * w + ox;
    tmp[o] = a0;
    tmp[dplane + o] = a1;
    tmp[2 * dplane + o] = a2;
}

__global__ void k_h2f(const unsigned short* __restrict__ src, float* __restrict__ dst, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = h2f(src[i]);
}

// one element of a planar source that is fp32, or fp16 (half != 0: the IFNet's tweens, read as
// the engine wrote them, so no pass widens them first)
__device__ __forceinline__ float ldS(const void* __restrict__ src, int half, size_t o)
{
    return half ? h2f(((const unsigned short*)src)[o]) : ((const float*)src)[o];
}

// one output pixel of a slot or a planar frame: the model pad cropped, copied 1:1 (every size change
// runs as the separable Lanczos pair below); the SDR and HDR slot packers and k_fitPlanar share it and
// differ only in the store
__device__ __forceinline__ void sampleOut(const void* __restrict__ src, int half, int planeStride,
                                          int rowStride, int ox, int oy, float c[3])
{
    const size_t o = (size_t)oy * rowStride + ox;
    c[0] = ldS(src, half, o);
    c[1] = ldS(src, half, planeStride + o);
    c[2] = ldS(src, half, 2 * planeStride + o);
}

// one output slot: sampleOut, clamp, round, and write the (R, G, B) planes as BGRA8 into the
// content rect of a pitched ring slot.
__global__ void k_packOut(const void* __restrict__ src, int half, int planeStride, int rowStride,
                          int w, int h, unsigned char* __restrict__ dst, int pitch,
                          int x0, int y0, int dw, int dh)
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= dw || oy >= dh || ox >= w || oy >= h) return;
    float c[3];
    sampleOut(src, half, planeStride, rowStride, ox, oy, c);
    unsigned char* p = dst + (size_t)(y0 + oy) * pitch + (size_t)(x0 + ox) * 4;
    for (int ci = 0; ci < 3; ci++)
    {
        float v = c[ci];
        v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        p[2 - ci] = (unsigned char)(int)rintf(v * 255.0f);
    }
    p[3] = 255;
}

// HDR output slot: the same sampleOut and clamp, but the canvas is R10G10B10A2 packed into
// int32 with A = 3, exactly what _pack_r10a2 produces and what the R10A2 present reads back.
__global__ void k_packOutHdr(const void* __restrict__ src, int half, int planeStride, int rowStride,
                             int w, int h, unsigned char* __restrict__ dst, int pitch,
                             int x0, int y0, int dw, int dh)
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= dw || oy >= dh || ox >= w || oy >= h) return;
    float c[3];
    sampleOut(src, half, planeStride, rowStride, ox, oy, c);
    unsigned int q[3];
    for (int ci = 0; ci < 3; ci++)
    {
        float v = c[ci];
        v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        q[ci] = (unsigned int)(int)(v * 1023.0f + 0.5f);
        if (q[ci] > 1023u) q[ci] = 1023u;
    }
    unsigned int* p = (unsigned int*)(dst + (size_t)(y0 + oy) * pitch) + (x0 + ox);
    *p = q[0] | (q[1] << 10) | (q[2] << 20) | (3u << 30);
}

// ---- live Sharpen and RTX VSR inside the native host ---------------------------------------
// Both effects run at the PRESENTED size, after the fit, exactly like live_server.py's compose
// (upscale, then the sharpen): the fit lands in a planar staging frame at dw x dh (the Lanczos pair, or
// the VSR bridge output unpacked by k_unpackBgraRgb), and the sharpen kernels apply Adaptive Sharpen
// on it while doing the slot store that k_packOut / k_packOutHdr do without effects.

// crop the model pad and quantise to tight 8-bit pixels at the model size, plane c into byte c,
// alpha 255: the (R, G, B) planes as the offline DLSS 4.5 host's RGBA8 frame and RTX VSR's input
__global__ void k_packBgra(const void* __restrict__ src, int half, int planeStride, int rowStride,
                           int w, int h, unsigned char* __restrict__ dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t o = (size_t)y * rowStride + x;
    unsigned char* p = dst + ((size_t)y * w + x) * 4;
    for (int ci = 0; ci < 3; ci++)
    {
        float v = ldS(src, half, (size_t)ci * planeStride + o);
        v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        p[ci] = (unsigned char)(int)rintf(v * 255.0f);
    }
    p[3] = 255;
}

// the (R, G, B) planes of either route as the B G R A bytes of NvOFFRUC and AMD FSR upscaling, and
// back (back with a true / 255)
__global__ void k_packBgraRgb(const void* __restrict__ src, int half, int planeStride, int rowStride,
                              int w, int h, unsigned char* __restrict__ dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t o = (size_t)y * rowStride + x;
    unsigned char* p = dst + ((size_t)y * w + x) * 4;
    for (int ci = 0; ci < 3; ci++)
    {
        float v = ldS(src, half, (size_t)ci * planeStride + o);
        v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        p[2 - ci] = (unsigned char)(int)rintf(v * 255.0f);
    }
    p[3] = 255;
}
__global__ void k_unpackBgraRgb(const unsigned char* __restrict__ src, int dw, int dh,
                                float* __restrict__ dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    const size_t plane = (size_t)dw * dh;
    const size_t o = (size_t)y * dw + x;
    const unsigned char* p = src + o * 4;
    dst[o] = p[2] / 255.0f;
    dst[plane + o] = p[1] / 255.0f;
    dst[2 * plane + o] = p[0] / 255.0f;
}

// RGBA8 back to the offline (R, G, B) planes: a DLSS 4.5 host frame or RTX VSR's output,
// dlssg.py _recv (uint8 / 255.0, a division, as python and k_unpackBgraRgb)
__global__ void k_unpackRgba(const unsigned char* __restrict__ src, int dw, int dh,
                             float* __restrict__ dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    const size_t plane = (size_t)dw * dh;
    const size_t o = (size_t)y * dw + x;
    const unsigned char* p = src + o * 4;
    dst[o] = p[0] / 255.0f;
    dst[plane + o] = p[1] / 255.0f;
    dst[2 * plane + o] = p[2] / 255.0f;
}

// sampleOut's 1:1 crop copy into a planar frame (the Restore input's fp16 widening, Upscale to at the model size)
__global__ void k_fitPlanar(const void* __restrict__ src, int half, int planeStride, int rowStride,
                            int w, int h, float* __restrict__ dst, int dw, int dh)
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= dw || oy >= dh || ox >= w || oy >= h) return;
    float c[3];
    sampleOut(src, half, planeStride, rowStride, ox, oy, c);
    const size_t plane = (size_t)dw * dh;
    const size_t o = (size_t)oy * dw + ox;
    dst[o] = c[0];
    dst[plane + o] = c[1];
    dst[2 * plane + o] = c[2];
}

// a planar frame (w x h, planes ps apart, rows rs apart) into the padded model frame (pw x ph,
// planes dps apart) with the edge replicated as the packers pad it: the decoded frame,
// upscaled before the model
__global__ void k_padPlanar(const float* __restrict__ src, int ps, int rs, int w, int h,
                            float* __restrict__ dst, int pw, int ph, int dps)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= pw || y >= ph) return;
    const int sx = x < w ? x : w - 1, sy = y < h ? y : h - 1;
    const size_t s = (size_t)sy * rs + sx, o = (size_t)y * pw + x;
    dst[o] = src[s];
    dst[(size_t)dps + o] = src[(size_t)ps + s];
    dst[2 * (size_t)dps + o] = src[2 * (size_t)ps + s];
}

// ---- every planar resize: the Lanczos pair --------------------------------------------------
// The resize as a separable pair (the placement at the top of this block, each axis's taps read
// from its k_rsTab table): 2 kLzUp taps on an enlarging axis, 2 ceil(kLzDown in / out) on a
// shrinking one, a pixel reading the sum of its two axes' taps.

// horizontal pass: the model pad cropped to (w, h) -> tmp (3, h, dw)
__global__ void k_fitAaH(const void* __restrict__ src, int half, int planeStride, int rowStride,
                         int w, int h, float* __restrict__ tmp, int dw, const RsTap* __restrict__ tab)
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int fs = rsTaps(w, dw);
    if (ox >= dw || y >= h) return;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
    for (int k = 0; k < fs; k++)
    {
        const RsTap t = tab[(size_t)k * dw + ox];
        const float wt = t.w;
        const int ix = t.i;
        const size_t o = (size_t)y * rowStride + ix;
        a0 += wt * ldS(src, half, o);
        a1 += wt * ldS(src, half, planeStride + o);
        a2 += wt * ldS(src, half, 2 * planeStride + o);
    }
    const size_t plane = (size_t)h * dw;
    const size_t o = (size_t)y * dw + ox;
    tmp[o] = a0;
    tmp[plane + o] = a1;
    tmp[2 * plane + o] = a2;
}

// the vertical taps of output row oy, column x, from a horizontal pass's tmp (3, h, dw): shared by
// the vertical pass and the fused vertical pass + slot store, so both give the same values
__device__ __forceinline__ void aaVSum(const float* __restrict__ tmp, int dw, int h, int dh, int x, int oy,
                                       const RsTap* __restrict__ tab, float a[3])
{
    const int fs = rsTaps(h, dh);
    const size_t splane = (size_t)h * dw;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
    for (int k = 0; k < fs; k++)
    {
        const RsTap t = tab[(size_t)k * dh + oy];
        const float wt = t.w;
        const int iy = t.i;
        const size_t o = (size_t)iy * dw + x;
        a0 += wt * tmp[o];
        a1 += wt * tmp[splane + o];
        a2 += wt * tmp[2 * splane + o];
    }
    a[0] = a0;
    a[1] = a1;
    a[2] = a2;
}

// vertical pass: tmp (3, h, dw) -> the planar staging frame (3, dh, dw)
__global__ void k_fitAaV(const float* __restrict__ tmp, int dw, int h,
                         float* __restrict__ dst, int dh, const RsTap* __restrict__ tab)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || oy >= dh) return;
    float a[3];
    aaVSum(tmp, dw, h, dh, x, oy, tab, a);
    const size_t plane = (size_t)dw * dh;
    const size_t o = (size_t)oy * dw + x;
    dst[o] = a[0];
    dst[plane + o] = a[1];
    dst[2 * plane + o] = a[2];
}

// the vertical pass fused with the slot store (a live fit with no effect after it): tmp (3, h, dw)
// -> k_packOut's BGRA8 store into the content rect of a pitched ring slot, the same bytes as the
// vertical pass into the staging frame followed by the 1:1 store, without the staging round trip
__global__ void k_packOutV(const float* __restrict__ tmp, int dw, int h, unsigned char* __restrict__ dst,
                           int pitch, int x0, int y0, int dh, const RsTap* __restrict__ tab)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || oy >= dh) return;
    float c[3];
    aaVSum(tmp, dw, h, dh, x, oy, tab, c);
    unsigned char* p = dst + (size_t)(y0 + oy) * pitch + (size_t)(x0 + x) * 4;
    for (int ci = 0; ci < 3; ci++)
    {
        float v = c[ci];
        v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        p[2 - ci] = (unsigned char)(int)rintf(v * 255.0f);
    }
    p[3] = 255;
}

// the HDR twin: k_packOutHdr's R10G10B10A2 store
__global__ void k_packOutHdrV(const float* __restrict__ tmp, int dw, int h, unsigned char* __restrict__ dst,
                              int pitch, int x0, int y0, int dh, const RsTap* __restrict__ tab)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || oy >= dh) return;
    float c[3];
    aaVSum(tmp, dw, h, dh, x, oy, tab, c);
    unsigned int q[3];
    for (int ci = 0; ci < 3; ci++)
    {
        float v = c[ci];
        v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        q[ci] = (unsigned int)(int)(v * 1023.0f + 0.5f);
        if (q[ci] > 1023u) q[ci] = 1023u;
    }
    unsigned int* p = (unsigned int*)(dst + (size_t)(y0 + oy) * pitch) + (x0 + x);
    *p = q[0] | (q[1] << 10) | (q[2] << 20) | (3u << 30);
}

// ---- live Restore ---------------------------------------------------------------------------
// live_server._Fit._restore: the model (the bundled Real-ESRGAN or AnimeJaNai's form on the host's kernels below; x =
// the model frame as fp16 NCHW [1,3,h,w], y = its s x reconstruction) then realesr.fit to the
// restore target: `out.clamp(0,1)` first, then the Lanczos pair at every ratio (the pair above with the clamp folded in
// and the output's own dtype read at the taps).
__device__ __forceinline__ unsigned short f2h(float f)
{
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}
__device__ __forceinline__ float restRaw(const void* __restrict__ src, int half, size_t o)
{
    return half ? h2f(((const unsigned short*)src)[o]) : ((const float*)src)[o];
}

__device__ __forceinline__ float restTap(const void* __restrict__ src, int half, size_t o)
{
    const float v = half ? h2f(((const unsigned short*)src)[o]) : ((const float*)src)[o];
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

// the engine input: the model pad cropped to (w, h), fp32 (or fp16) planar -> fp16 NCHW (torch
// .half(); an fp16 source passes through unchanged)
__global__ void k_restIn(const void* __restrict__ src, int half, int planeStride, int rowStride,
                         int w, int h, unsigned short* __restrict__ dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t o = (size_t)y * w + x, plane = (size_t)w * h;
    const size_t s = (size_t)y * rowStride + x;
    dst[o] = f2h(ldS(src, half, s));
    dst[plane + o] = f2h(ldS(src, half, planeStride + s));
    dst[2 * plane + o] = f2h(ldS(src, half, 2 * planeStride + s));
}

// fp32 -> fp16 element for element, round to nearest even: the frames the RIFE engines take in
// fp16 (the IFNet's x, the encode's img)
__global__ void k_f2h(const float* __restrict__ src, unsigned short* __restrict__ dst, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = f2h(src[i]);
}

// the fold, horizontal pass: the 4x output (fp16 or fp32 planar, clamped per tap)
// -> tmp (3, h, dw); the taps are k_fitAaH's
__global__ void k_restFoldH(const void* __restrict__ src, int half, int planeStride, int rowStride,
                            int w, int h, float* __restrict__ tmp, int dw, const RsTap* __restrict__ tab)
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int fs = rsTaps(w, dw);
    if (ox >= dw || y >= h) return;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
    for (int k = 0; k < fs; k++)
    {
        const RsTap t = tab[(size_t)k * dw + ox];
        const float wt = t.w;
        const int ix = t.i;
        const size_t o = (size_t)y * rowStride + ix;
        a0 += wt * restTap(src, half, o);
        a1 += wt * restTap(src, half, planeStride + o);
        a2 += wt * restTap(src, half, 2 * planeStride + o);
    }
    const size_t plane = (size_t)h * dw;
    const size_t o = (size_t)y * dw + ox;
    tmp[o] = a0;
    tmp[plane + o] = a1;
    tmp[2 * plane + o] = a2;
}

// the fold's horizontal pass on an x s net's sub-pixel planes (k_restConvLast writes them; their PixelShuffle and, with
// in, the nearest-base add happen here): the s x value at (Y, X) = plane c s^2 + (Y % s) s + X % s at (Y / s, X / s)
// of the w x h plane grid (+ the input's pixel there when in != nullptr: Real-ESRGAN), added in fp32 and clamped per
// tap like k_restFoldH -> tmp (3, oh, dw); ow x oh = the model's output size, at most s w x s h (AnimeJaNai's half grid
// x 4 = the frame x 2, cropped where the frame is odd)
__global__ void k_restFoldHSub(const void* __restrict__ sub, int half, const void* __restrict__ in, int inHalf, int s,
                               int w, int h, int ow, int oh, float* __restrict__ tmp, int dw,
                               const RsTap* __restrict__ tab)
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int Y = blockIdx.y * blockDim.y + threadIdx.y;
    const int fs = rsTaps(ow, dw);
    if (ox >= dw || Y >= oh) return;
    const int y = Y / s;
    const size_t ps = (size_t)w * h, row = (size_t)y * w, cs = (size_t)s * s * ps;
    const size_t sy = (size_t)(Y - y * s) * s * ps;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
    for (int k = 0; k < fs; k++)
    {
        const RsTap t = tab[(size_t)k * dw + ox];
        const int x = t.i / s;
        const size_t o = row + x;
        const size_t p = sy + (size_t)(t.i - x * s) * ps + o;
        const float v0 = restRaw(sub, half, p) + (in ? restRaw(in, inHalf, o) : 0.0f);
        const float v1 = restRaw(sub, half, cs + p) + (in ? restRaw(in, inHalf, ps + o) : 0.0f);
        const float v2 = restRaw(sub, half, 2 * cs + p) + (in ? restRaw(in, inHalf, 2 * ps + o) : 0.0f);
        a0 += t.w * (v0 < 0.0f ? 0.0f : (v0 > 1.0f ? 1.0f : v0));
        a1 += t.w * (v1 < 0.0f ? 0.0f : (v1 > 1.0f ? 1.0f : v1));
        a2 += t.w * (v2 < 0.0f ? 0.0f : (v2 > 1.0f ? 1.0f : v2));
    }
    const size_t plane = (size_t)oh * dw;
    const size_t o = (size_t)Y * dw + ox;
    tmp[o] = a0;
    tmp[plane + o] = a1;
    tmp[2 * plane + o] = a2;
}

// Real-ESRGAN animevideov3 on the host's own kernels (restNat): k_restConv0 (3 -> 64 + PReLU),
// 16 x k_restConvMid (64 -> 64 + PReLU), k_restConvLast (64 -> 48 sub-pixel planes = k_restFoldHSub's input); between
// them channels-last fp16 (64 halves = 128 B a pixel). The 64 -> 64 layers are implicit GEMMs on the tensor cores
// (mma.sync m16n8k16, M = pixels, N = 64 outputs, K = 9 taps x 64 channels) with fp16 ACCUMULATORS (the GeForce rate:
// 1.4x fp32-accumulate convs, the image within one 8-bit code of fp32 sums); the last layer keeps fp32
// accumulators. A block = 4 warps on an 8 x 32 pixel tile, its 10 x 34 halo in shared memory (16-byte chunks XOR-
// swizzled by pixel & 7, so each ldmatrix reads 8 bank groups), two blocks an SM so one block's loads and stores
// overlap the other's math; the weights come from global memory already in fragment order (one uint4 a lane = the four
// B registers of two n-tiles), loaded a k-step ahead.
#define RN_TW 32
#define RN_TH 8
#define RN_HW (RN_TW + 2)
#define RN_HH (RN_TH + 2)

// the tensor-core and async-copy instructions: sm_80+ (RTX 30 and newer) as written; sm_75 (RTX 20) has no cp.async
// and no k16 mma: the product runs as its two k8 halves (a0 a1 with b0, then a2 a3 with b1), the trailing `//` turns
// the call site's k16 operand list into a PTX comment, rnCp16 copies in place and commit / wait are empty
#if __CUDA_ARCH__ >= 800
#define SMV_MMA_F32 "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
#define SMV_MMA_F16 "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
#define SMV_CP_COMMIT "cp.async.commit_group;\n"
#define SMV_CP_WAIT0 "cp.async.wait_group 0;\n"
#else
#define SMV_MMA_F32                                                                                                    \
    "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%8}, {%0,%1,%2,%3};\n"                 \
    "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%6,%7}, {%9}, {%0,%1,%2,%3}; // "
// fp16 sums: both halves on fp32 accumulators, rounded to fp16 once (as the sm_80 instruction rounds)
#define SMV_MMA_F16                                                                                                    \
    "{\n.reg .f16 smvh<4>;\n.reg .f32 smvf<4>;\nmov.b32 {smvh0, smvh1}, %0;\nmov.b32 {smvh2, smvh3}, %1;\n"           \
    "cvt.f32.f16 smvf0, smvh0;\ncvt.f32.f16 smvf1, smvh1;\ncvt.f32.f16 smvf2, smvh2;\ncvt.f32.f16 smvf3, smvh3;\n"    \
    "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {smvf0,smvf1,smvf2,smvf3}, {%2,%3}, {%6}, "                     \
    "{smvf0,smvf1,smvf2,smvf3};\n"                                                                                     \
    "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {smvf0,smvf1,smvf2,smvf3}, {%4,%5}, {%7}, "                     \
    "{smvf0,smvf1,smvf2,smvf3};\n"                                                                                     \
    "cvt.rn.f16.f32 smvh0, smvf0;\ncvt.rn.f16.f32 smvh1, smvf1;\ncvt.rn.f16.f32 smvh2, smvf2;\n"                      \
    "cvt.rn.f16.f32 smvh3, smvf3;\nmov.b32 %0, {smvh0, smvh1};\nmov.b32 %1, {smvh2, smvh3};\n} // "
#define SMV_CP_COMMIT ""
#define SMV_CP_WAIT0 ""
#endif

__device__ __forceinline__ void rnCp16(unsigned dst, const void* src, int bytes)
{
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes));
#else
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (bytes)
        v = *reinterpret_cast<const uint4*>(src);
    asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" ::"r"(dst), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
#endif
}

__device__ __forceinline__ void rnLdm4(unsigned addr, unsigned* r)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

// a PReLU on an fp16 value: the fp32 product of two fp16 values is exact, so its one rounding = the fp16 multiply
__device__ __forceinline__ unsigned rnPrelu(unsigned v, unsigned short a)
{
    const float y = h2f((unsigned short)v);
    return y < 0.0f ? (unsigned)f2h(h2f(a) * y) : v;
}

// the 10 x 34 halo of an 8 x 32 tile (zero outside the frame) into shared memory, cp.async; the caller waits
__device__ __forceinline__ void rnHalo(unsigned halo, const unsigned short* __restrict__ x, int tx0, int ty0, int H,
                                       int W)
{
    for (int i = threadIdx.x; i < RN_HH * RN_HW * 8; i += 128)
    {
        const int p = i >> 3, k = i & 7;
        const int hr = p / RN_HW, hc = p - hr * RN_HW;
        const int gy = ty0 - 1 + hr, gx = tx0 - 1 + hc;
        const bool in = gy >= 0 && gy < H && gx >= 0 && gx < W;
        const unsigned short* src = in ? x + ((size_t)gy * W + gx) * 64 + k * 8 : x;
        rnCp16(halo + ((p * 8 + (k ^ (p & 7))) << 4), src, in ? 16 : 0);
    }
    asm volatile(SMV_CP_COMMIT);
}

// the first layer: three fp16 planes (k_restIn's) -> channels-last + PReLU; one pixel a thread, fp32 sums of 27 terms;
// w0[o][c][tap] and b0 fp32 in shared memory
__global__ void __launch_bounds__(256) k_restConv0(const unsigned short* __restrict__ src, unsigned short* __restrict__ y,
                                                    const float* __restrict__ w0, const float* __restrict__ b0,
                                                    const unsigned short* __restrict__ slope, int H, int W)
{
    __shared__ float sw[64 * 27 + 64];
    for (int i = threadIdx.x; i < 64 * 27 + 64; i += 256)
        sw[i] = i < 64 * 27 ? w0[i] : b0[i - 64 * 27];
    __syncthreads();
    const int xx = blockIdx.x * 32 + (threadIdx.x & 31), yy = blockIdx.y * 8 + (threadIdx.x >> 5);
    if (xx >= W || yy >= H) return;
    const size_t ps = (size_t)W * H;
    float v[27];
#pragma unroll
    for (int c = 0; c < 3; ++c)
#pragma unroll
        for (int t = 0; t < 9; ++t)
        {
            const int gy = yy + t / 3 - 1, gx = xx + t % 3 - 1;
            v[c * 9 + t] = (gy >= 0 && gy < H && gx >= 0 && gx < W) ? h2f(src[c * ps + (size_t)gy * W + gx]) : 0.0f;
        }
    unsigned short* o = y + ((size_t)yy * W + xx) * 64;
#pragma unroll
    for (int n = 0; n < 64; n += 8)
    {
        unsigned r[4];
#pragma unroll
        for (int j = 0; j < 8; j += 2)
        {
            float a0 = sw[64 * 27 + n + j], a1 = sw[64 * 27 + n + j + 1];
#pragma unroll
            for (int k = 0; k < 27; ++k)
            {
                a0 = fmaf(sw[(n + j) * 27 + k], v[k], a0);
                a1 = fmaf(sw[(n + j + 1) * 27 + k], v[k], a1);
            }
            r[j / 2] = rnPrelu(f2h(a0), slope[n + j]) | (rnPrelu(f2h(a1), slope[n + j + 1]) << 16);
        }
        *reinterpret_cast<uint4*>(o + n) = make_uint4(r[0], r[1], r[2], r[3]);
    }
}

// a 64 -> 64 layer + PReLU, channels-last in and out; fp16 accumulators started at the bias; wf[tap][kc][q][lane]
__global__ void __launch_bounds__(128, 2) k_restConvMid(const unsigned short* __restrict__ x,
                                                         unsigned short* __restrict__ y, const uint4* __restrict__ wf,
                                                         const unsigned short* __restrict__ bias,
                                                         const unsigned short* __restrict__ slope, int H, int W)
{
    extern __shared__ __align__(128) unsigned char rnSm[];
    const unsigned halo = (unsigned)__cvta_generic_to_shared(rnSm);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int tx0 = blockIdx.x * RN_TW, ty0 = blockIdx.y * RN_TH;
    rnHalo(halo, x, tx0, ty0, H, W);
    uint4 bc[4], bn[4];
#pragma unroll
    for (int q = 0; q < 4; ++q)
        bc[q] = __ldg(wf + q * 32 + lane);
    const int g = lane >> 2, t4 = lane & 3;
    unsigned acc[4][8][2];
#pragma unroll
    for (int nj = 0; nj < 8; ++nj)
    {
        const unsigned b2 = (unsigned)bias[nj * 8 + t4 * 2] | ((unsigned)bias[nj * 8 + t4 * 2 + 1] << 16);
#pragma unroll
        for (int mi = 0; mi < 4; ++mi)
            acc[mi][nj][0] = acc[mi][nj][1] = b2;
    }
    asm volatile(SMV_CP_WAIT0);
    __syncthreads();
    const int arow = (lane & 7) + ((lane >> 3) & 1) * 8, akc = lane >> 4;
#pragma unroll 1
    for (int tap = 0; tap < 9; ++tap)
    {
        const int dy = tap / 3, dx = tap - dy * 3;
        int pa[4];
#pragma unroll
        for (int mi = 0; mi < 4; ++mi)
            pa[mi] = (2 * warp + (mi >> 1) + dy) * RN_HW + (mi & 1) * 16 + arow + dx;
#pragma unroll
        for (int kc = 0; kc < 4; ++kc)
        {
            const int step = tap * 4 + kc;
            if (step + 1 < 36)
#pragma unroll
                for (int q = 0; q < 4; ++q)
                    bn[q] = __ldg(wf + ((step + 1) * 4 + q) * 32 + lane);
#pragma unroll
            for (int mi = 0; mi < 4; ++mi)
            {
                unsigned a[4];
                const int p = pa[mi], ch = kc * 2 + akc;
                rnLdm4(halo + ((p * 8 + (ch ^ (p & 7))) << 4), a);
#pragma unroll
                for (int q = 0; q < 8; ++q)
                {
                    const unsigned b0 = (q & 1) ? bc[q >> 1].z : bc[q >> 1].x;
                    const unsigned b1 = (q & 1) ? bc[q >> 1].w : bc[q >> 1].y;
                    asm volatile(
                        SMV_MMA_F16 "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                        : "+r"(acc[mi][q][0]), "+r"(acc[mi][q][1])
                        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
                }
            }
#pragma unroll
            for (int q = 0; q < 4; ++q)
                bc[q] = bn[q];
        }
    }
#pragma unroll
    for (int nj = 0; nj < 8; ++nj)
    {
        const int n0 = nj * 8 + t4 * 2;
        const unsigned short s0 = slope[n0], s1 = slope[n0 + 1];
#pragma unroll
        for (int mi = 0; mi < 4; ++mi)
        {
            const int gy = ty0 + 2 * warp + (mi >> 1);
            if (gy >= H) continue;
#pragma unroll
            for (int h = 0; h < 2; ++h)
            {
                const int gx = tx0 + (mi & 1) * 16 + g + h * 8;
                if (gx >= W) continue;
                const unsigned v = acc[mi][nj][h];
                *reinterpret_cast<unsigned*>(y + ((size_t)gy * W + gx) * 64 + n0) =
                    rnPrelu(v & 0xffffu, s0) | (rnPrelu(v >> 16, s1) << 16);
            }
        }
    }
}

// the last layer: 64 -> 48, channels-last in -> 48 fp16 planes out (plane n at out[n * ps]), fp32 accumulators started
// at the bias, no activation; wf[tap][kc][q 3][lane]
__global__ void __launch_bounds__(128, 2) k_restConvLast(const unsigned short* __restrict__ x,
                                                          unsigned short* __restrict__ out, const uint4* __restrict__ wf,
                                                          const unsigned short* __restrict__ bias, int H, int W)
{
    extern __shared__ __align__(128) unsigned char rnSm[];
    const unsigned halo = (unsigned)__cvta_generic_to_shared(rnSm);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int tx0 = blockIdx.x * RN_TW, ty0 = blockIdx.y * RN_TH;
    rnHalo(halo, x, tx0, ty0, H, W);
    uint4 bc[3], bn[3];
#pragma unroll
    for (int q = 0; q < 3; ++q)
        bc[q] = __ldg(wf + q * 32 + lane);
    const int g = lane >> 2, t4 = lane & 3;
    float acc[4][6][4];
#pragma unroll
    for (int nj = 0; nj < 6; ++nj)
    {
        const float b0 = h2f(bias[nj * 8 + t4 * 2]), b1 = h2f(bias[nj * 8 + t4 * 2 + 1]);
#pragma unroll
        for (int mi = 0; mi < 4; ++mi)
        {
            acc[mi][nj][0] = acc[mi][nj][2] = b0;
            acc[mi][nj][1] = acc[mi][nj][3] = b1;
        }
    }
    asm volatile(SMV_CP_WAIT0);
    __syncthreads();
    const int arow = (lane & 7) + ((lane >> 3) & 1) * 8, akc = lane >> 4;
#pragma unroll 1
    for (int tap = 0; tap < 9; ++tap)
    {
        const int dy = tap / 3, dx = tap - dy * 3;
        int pa[4];
#pragma unroll
        for (int mi = 0; mi < 4; ++mi)
            pa[mi] = (2 * warp + (mi >> 1) + dy) * RN_HW + (mi & 1) * 16 + arow + dx;
#pragma unroll
        for (int kc = 0; kc < 4; ++kc)
        {
            const int step = tap * 4 + kc;
            if (step + 1 < 36)
#pragma unroll
                for (int q = 0; q < 3; ++q)
                    bn[q] = __ldg(wf + ((step + 1) * 3 + q) * 32 + lane);
#pragma unroll
            for (int mi = 0; mi < 4; ++mi)
            {
                unsigned a[4];
                const int p = pa[mi], ch = kc * 2 + akc;
                rnLdm4(halo + ((p * 8 + (ch ^ (p & 7))) << 4), a);
#pragma unroll
                for (int q = 0; q < 6; ++q)
                {
                    const unsigned b0 = (q & 1) ? bc[q >> 1].z : bc[q >> 1].x;
                    const unsigned b1 = (q & 1) ? bc[q >> 1].w : bc[q >> 1].y;
                    float* d = acc[mi][q];
                    asm volatile(SMV_MMA_F32 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, "
                                 "{%8,%9}, {%0,%1,%2,%3};\n"
                                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
                }
            }
#pragma unroll
            for (int q = 0; q < 3; ++q)
                bc[q] = bn[q];
        }
    }
    const size_t ps = (size_t)W * H;
#pragma unroll
    for (int nj = 0; nj < 6; ++nj)
    {
        const int n0 = nj * 8 + t4 * 2;
#pragma unroll
        for (int mi = 0; mi < 4; ++mi)
        {
            const int gy = ty0 + 2 * warp + (mi >> 1);
            if (gy >= H) continue;
#pragma unroll
            for (int h = 0; h < 2; ++h)
            {
                const int gx = tx0 + (mi & 1) * 16 + g + h * 8;
                if (gx >= W) continue;
                const size_t pix = (size_t)gy * W + gx;
                out[n0 * ps + pix] = f2h(acc[mi][nj][2 * h]);
                out[(n0 + 1) * ps + pix] = f2h(acc[mi][nj][2 * h + 1]);
            }
        }
    }
}

// AnimeJaNai (SPAN-F3, 64 features, the --restore-model ONNX's weights) on the host's kernels: k_ajFirst, then per
// block 3 x k_restConvMid (ReLU = slope 0, none = slope 1) + k_ajAtt, k_ajCat, k_restConvLast -> the 48 sub-pixel
// planes at the half grid (k_restFoldHSub at s 4, no base). Every activation channels-last fp16 at the half grid.

// the first layer: three fp16 planes at the source size sw x sh (k_restIn's) -> reflect pad to even -> PixelUnshuffle(2)
// (channel c * 4 + dy * 2 + dx = plane c at (2 y + dy, 2 x + dx)) -> conv 12 -> 64 3 x 3 (zero pad 1 on the half grid),
// no activation -> channels-last fp16 at hw x hh; one half-grid pixel a thread, fp32 sums; w[o][12][9] + b fp32 in
// shared memory
__global__ void __launch_bounds__(256) k_ajFirst(const unsigned short* __restrict__ src, int sw, int sh,
                                                  unsigned short* __restrict__ y, const float* __restrict__ wt,
                                                  const float* __restrict__ bias, int hw, int hh)
{
    __shared__ float sm[64 * 108 + 64];
    for (int i = threadIdx.x; i < 64 * 108 + 64; i += 256)
        sm[i] = i < 64 * 108 ? wt[i] : bias[i - 64 * 108];
    __syncthreads();
    const int x2 = blockIdx.x * 32 + (threadIdx.x & 31), y2 = blockIdx.y * 8 + (threadIdx.x >> 5);
    if (x2 >= hw || y2 >= hh) return;
    const size_t ps = (size_t)sw * sh;
    float acc[64];
#pragma unroll
    for (int o = 0; o < 64; o++)
        acc[o] = sm[64 * 108 + o];
    for (int c = 0; c < 12; c++)
    {
        const int pl = c >> 2, dy = (c >> 1) & 1, dx = c & 1;
        float v[9];
#pragma unroll
        for (int t = 0; t < 9; t++)
        {
            const int yy = y2 + t / 3 - 1, xx = x2 + t % 3 - 1;
            if (yy < 0 || yy >= hh || xx < 0 || xx >= hw)
                v[t] = 0.0f;
            else
            {
                int fy = 2 * yy + dy, fx = 2 * xx + dx;
                fy = fy < sh ? fy : 2 * sh - 2 - fy; // the reflect pad of an odd height / width
                fx = fx < sw ? fx : 2 * sw - 2 - fx;
                v[t] = h2f(src[pl * ps + (size_t)fy * sw + fx]);
            }
        }
#pragma unroll
        for (int o = 0; o < 64; o++)
        {
            float a = acc[o];
#pragma unroll
            for (int t = 0; t < 9; t++)
                a = fmaf(sm[o * 108 + c * 9 + t], v[t], a);
            acc[o] = a;
        }
    }
    unsigned short* out = y + ((size_t)y2 * hw + x2) * 64;
#pragma unroll
    for (int n = 0; n < 64; n += 8)
        *reinterpret_cast<uint4*>(out + n) =
            make_uint4((unsigned)f2h(acc[n]) | ((unsigned)f2h(acc[n + 1]) << 16),
                       (unsigned)f2h(acc[n + 2]) | ((unsigned)f2h(acc[n + 3]) << 16),
                       (unsigned)f2h(acc[n + 4]) | ((unsigned)f2h(acc[n + 5]) << 16),
                       (unsigned)f2h(acc[n + 6]) | ((unsigned)f2h(acc[n + 7]) << 16));
}

// 128 pixels' 64 channels (a channels-last run) into shared memory, 16-byte chunks XOR-swizzled by pixel & 7; pixels
// past n = zeros; the caller commits / waits
__device__ __forceinline__ void ajTile(unsigned smem, const unsigned short* __restrict__ x, size_t p0, size_t n)
{
    for (int i = threadIdx.x; i < 128 * 8; i += 128)
    {
        const int p = i >> 3, k = i & 7;
        const bool in = p0 + p < n;
        rnCp16(smem + ((p * 8 + (k ^ (p & 7))) << 4), in ? x + (p0 + p) * 64 + k * 8 : x, in ? 16 : 0);
    }
}

// one 1 x 1 64 -> 64 GEMM on a staged tile: K = 64 (4 k-steps), the warp's 32 pixels (2 m16 tiles) x 64 outputs, fp32
// accumulators; wf[kc][q][lane] (B registers of n-tiles 2q, 2q + 1)
__device__ __forceinline__ void ajGemm64(unsigned smem, const uint4* __restrict__ wf, float (*acc)[8][4])
{
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int arow = (lane & 7) + ((lane >> 3) & 1) * 8, akc = lane >> 4;
#pragma unroll
    for (int kc = 0; kc < 4; kc++)
    {
        uint4 b[4];
#pragma unroll
        for (int q = 0; q < 4; q++)
            b[q] = __ldg(wf + (kc * 4 + q) * 32 + lane);
#pragma unroll
        for (int mi = 0; mi < 2; mi++)
        {
            unsigned a[4];
            const int p = warp * 32 + mi * 16 + arow, ch = kc * 2 + akc;
            rnLdm4(smem + ((p * 8 + (ch ^ (p & 7))) << 4), a);
#pragma unroll
            for (int q = 0; q < 8; q++)
            {
                const unsigned b0 = (q & 1) ? b[q >> 1].z : b[q >> 1].x;
                const unsigned b1 = (q & 1) ? b[q >> 1].w : b[q >> 1].y;
                float* d = acc[mi][q];
                asm volatile(SMV_MMA_F32 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, "
                             "{%8,%9}, {%0,%1,%2,%3};\n"
                             : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
            }
        }
    }
}

// a SPAN block's tail on n pixels: out = (t + c3) * tanh(conv1x1(c3) + b), fp32 math, one rounding to fp16; 128 pixels a
// block of 4 warps; wf = the 1 x 1's fragment-order weights, bias fp32
__global__ void __launch_bounds__(128) k_ajAtt(const unsigned short* __restrict__ c3,
                                                const unsigned short* __restrict__ t, unsigned short* __restrict__ out,
                                                const uint4* __restrict__ wf, const float* __restrict__ bias, int n)
{
    __shared__ __align__(128) unsigned char sm[128 * 128];
    const unsigned smem = (unsigned)__cvta_generic_to_shared(sm);
    const size_t p0 = (size_t)blockIdx.x * 128;
    ajTile(smem, c3, p0, (size_t)n);
    asm volatile(SMV_CP_COMMIT);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t4 = lane & 3;
    float acc[2][8][4];
#pragma unroll
    for (int q = 0; q < 8; q++)
    {
        const float b0 = bias[q * 8 + t4 * 2], b1 = bias[q * 8 + t4 * 2 + 1];
#pragma unroll
        for (int mi = 0; mi < 2; mi++)
        {
            acc[mi][q][0] = acc[mi][q][2] = b0;
            acc[mi][q][1] = acc[mi][q][3] = b1;
        }
    }
    asm volatile(SMV_CP_WAIT0);
    __syncthreads();
    ajGemm64(smem, wf, acc);
#pragma unroll
    for (int mi = 0; mi < 2; mi++)
#pragma unroll
        for (int h = 0; h < 2; h++)
        {
            const size_t p = p0 + warp * 32 + mi * 16 + g + h * 8;
            if (p >= (size_t)n) continue;
#pragma unroll
            for (int q = 0; q < 8; q++)
            {
                const size_t o = p * 64 + q * 8 + t4 * 2;
                const unsigned cv = *reinterpret_cast<const unsigned*>(c3 + o);
                const unsigned tv = *reinterpret_cast<const unsigned*>(t + o);
                const float r0 = (h2f((unsigned short)tv) + h2f((unsigned short)cv)) * tanhf(acc[mi][q][2 * h]);
                const float r1 = (h2f((unsigned short)(tv >> 16)) + h2f((unsigned short)(cv >> 16))) *
                                 tanhf(acc[mi][q][2 * h + 1]);
                *reinterpret_cast<unsigned*>(out + o) = (unsigned)f2h(r0) | ((unsigned)f2h(r1) << 16);
            }
        }
}

// the tail's 1 x 1 256 -> 64 on n pixels of the concat [s0, s1, s2, s3] (each channels-last 64, read in place, one
// staged tile at a time), fp32 accumulators from the bias, no activation; wf[src][kc][q][lane]
__global__ void __launch_bounds__(128) k_ajCat(const unsigned short* __restrict__ s0,
                                                const unsigned short* __restrict__ s1,
                                                const unsigned short* __restrict__ s2,
                                                const unsigned short* __restrict__ s3, unsigned short* __restrict__ out,
                                                const uint4* __restrict__ wf, const float* __restrict__ bias, int n)
{
    __shared__ __align__(128) unsigned char sm[128 * 128];
    const unsigned smem = (unsigned)__cvta_generic_to_shared(sm);
    const size_t p0 = (size_t)blockIdx.x * 128;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t4 = lane & 3;
    float acc[2][8][4];
#pragma unroll
    for (int q = 0; q < 8; q++)
    {
        const float b0 = bias[q * 8 + t4 * 2], b1 = bias[q * 8 + t4 * 2 + 1];
#pragma unroll
        for (int mi = 0; mi < 2; mi++)
        {
            acc[mi][q][0] = acc[mi][q][2] = b0;
            acc[mi][q][1] = acc[mi][q][3] = b1;
        }
    }
    const unsigned short* src[4] = {s0, s1, s2, s3};
    for (int s = 0; s < 4; s++)
    {
        ajTile(smem, src[s], p0, (size_t)n);
        asm volatile(SMV_CP_COMMIT);
        asm volatile(SMV_CP_WAIT0);
        __syncthreads();
        ajGemm64(smem, wf + (size_t)s * 4 * 4 * 32, acc);
        __syncthreads();
    }
#pragma unroll
    for (int mi = 0; mi < 2; mi++)
#pragma unroll
        for (int h = 0; h < 2; h++)
        {
            const size_t p = p0 + warp * 32 + mi * 16 + g + h * 8;
            if (p >= (size_t)n) continue;
#pragma unroll
            for (int q = 0; q < 8; q++)
                *reinterpret_cast<unsigned*>(out + p * 64 + q * 8 + t4 * 2) =
                    (unsigned)f2h(acc[mi][q][2 * h]) | ((unsigned)f2h(acc[mi][q][2 * h + 1]) << 16);
        }
}

// Anime4K's fold on its three last textures (the exported graph ends before the shader's Depth-to-Space pass, which
// happens here): the 2x value at (Y, X) = texture c's channel (Y % 2) 2 + X % 2 at (Y / 2, X / 2) + the source's
// bilinear 2x there (align_corners False, edge clamp: 3/4 of the nearest source pixel, 1/4 of the next one out on each
// axis), added in fp32 and NOT clipped (the shader's rgba16f textures keep a net's overshoot until the output); stat
// set (a chain's last pass with Clamp_Highlights on) = the clamp on this 2x frame, before any resize (the shader's
// PREKERNEL hook): the luma lowered to the sw x sh maximum's bilinear sample at the texel centre (clamp-to-edge) on all
// three channels, k_a4kClamp's arithmetic; in = the engine's fp16 input
__device__ __forceinline__ void a4kFoldPx(const void* __restrict__ sub, int half, const unsigned short* __restrict__ in,
                                          int w, int h, int Y, int X, float* v, const unsigned short* __restrict__ stat,
                                          int sw, int sh)
{
    const int y = Y >> 1, dy = Y & 1;
    const int yn = dy ? (y + 1 < h ? y + 1 : h - 1) : (y > 0 ? y - 1 : 0);
    const size_t ps = (size_t)w * h, r0 = (size_t)y * w, r1 = (size_t)yn * w;
    const int x = X >> 1, dx = X & 1;
    const int xn = dx ? (x + 1 < w ? x + 1 : w - 1) : (x > 0 ? x - 1 : 0);
    const size_t sp = (size_t)(dy * 2 + dx) * ps + r0 + x;
    for (int c = 0; c < 3; c++)
    {
        const unsigned short* p = in + (size_t)c * ps;
        const float base = 0.75f * (0.75f * h2f(p[r0 + x]) + 0.25f * h2f(p[r0 + xn])) +
                           0.25f * (0.75f * h2f(p[r1 + x]) + 0.25f * h2f(p[r1 + xn]));
        v[c] = restRaw(sub, half, (size_t)c * 4 * ps + sp) + base;
    }
    if (!stat) return;
    const float cur = 0.299f * v[0] + 0.587f * v[1] + 0.114f * v[2];
    const float fx = ((float)X + 0.5f) * (float)sw / (float)(2 * w) - 0.5f;
    const float fy = ((float)Y + 0.5f) * (float)sh / (float)(2 * h) - 0.5f;
    const float x0f = floorf(fx), y0f = floorf(fy);
    const float ax = fx - x0f, ay = fy - y0f;
    const int x0 = min(max((int)x0f, 0), sw - 1), x1 = min(max((int)x0f + 1, 0), sw - 1);
    const int y0 = min(max((int)y0f, 0), sh - 1), y1 = min(max((int)y0f + 1, 0), sh - 1);
    const float s00 = h2f(stat[(size_t)y0 * sw + x0]), s10 = h2f(stat[(size_t)y0 * sw + x1]);
    const float s01 = h2f(stat[(size_t)y1 * sw + x0]), s11 = h2f(stat[(size_t)y1 * sw + x1]);
    const float mx = (s00 + (s10 - s00) * ax) + ((s01 + (s11 - s01) * ax) - (s00 + (s10 - s00) * ax)) * ay;
    const float d = cur - fminf(cur, mx);
    v[0] -= d;
    v[1] -= d;
    v[2] -= d;
}

// the horizontal pass, 2w -> dw along x -> tmp (3, 2h, dw); 16 x 16 blocks: each 2x value the block's outputs tap is
// computed ONCE into shared memory (the block's 16 rows x its taps' column range, at most AF_CAP columns; a wider
// range, a far shrink, computes per tap), then every output sums its taps in the table's order
#define AF_CAP 128
__global__ void __launch_bounds__(256) k_a4kFoldHSub(const void* __restrict__ sub, int half,
                                                     const unsigned short* __restrict__ in, int w, int h,
                                                     float* __restrict__ tmp, int dw, const RsTap* __restrict__ tab,
                                                     const unsigned short* __restrict__ stat, int sw, int sh)
{
    __shared__ float sv[3][16][AF_CAP];
    __shared__ int lo, hi;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int ox = blockIdx.x * 16 + tx, Y0 = blockIdx.y * 16, Y = Y0 + ty;
    const int fs = rsTaps(2 * w, dw);
    if (tx == 0 && ty == 0)
    {
        lo = 0x7fffffff;
        hi = -1;
    }
    __syncthreads();
    if (ty == 0 && ox < dw)
    {
        int mn = 0x7fffffff, mx = -1;
        for (int k = 0; k < fs; k++)
        {
            const int i = tab[(size_t)k * dw + ox].i;
            mn = min(mn, i);
            mx = max(mx, i);
        }
        atomicMin(&lo, mn);
        atomicMax(&hi, mx);
    }
    __syncthreads();
    const int c0 = lo, n = hi - lo + 1;
    const bool staged = n <= AF_CAP;
    if (staged)
        for (int q = ty * 16 + tx; q < 16 * n; q += 256)
        {
            const int r = q / n, cc = q - r * n;
            if (Y0 + r < 2 * h)
            {
                float v[3];
                a4kFoldPx(sub, half, in, w, h, Y0 + r, c0 + cc, v, stat, sw, sh);
                sv[0][r][cc] = v[0];
                sv[1][r][cc] = v[1];
                sv[2][r][cc] = v[2];
            }
        }
    __syncthreads();
    if (ox >= dw || Y >= 2 * h) return;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
    for (int k = 0; k < fs; k++)
    {
        const RsTap t = tab[(size_t)k * dw + ox];
        float v[3];
        if (staged)
        {
            const int cc = t.i - c0;
            v[0] = sv[0][ty][cc];
            v[1] = sv[1][ty][cc];
            v[2] = sv[2][ty][cc];
        }
        else
            a4kFoldPx(sub, half, in, w, h, Y, t.i, v, stat, sw, sh);
        a0 += t.w * v[0];
        a1 += t.w * v[1];
        a2 += t.w * v[2];
    }
    const size_t plane = (size_t)2 * h * dw;
    const size_t o = (size_t)Y * dw + ox;
    tmp[o] = a0;
    tmp[plane + o] = a1;
    tmp[2 * plane + o] = a2;
}

// an exact 2x frame (2w x 2h): Lanczos at 1:1 weighs the centre tap 1 and every other 0 (sinpi at an integer is 0), so
// the two passes reproduce the 2x values exactly; this one pass writes them straight into the planar target: clip 1 =
// the chain's output (0..1, after the clamp when stat is set), clip 0 = the next pass's input (dA4kMid, as the net made it)
__global__ void k_a4kFold2x(const void* __restrict__ sub, int half, const unsigned short* __restrict__ in, int w, int h,
                            float* __restrict__ dst, const unsigned short* __restrict__ stat, int sw, int sh, int clip)
{
    const int X = blockIdx.x * blockDim.x + threadIdx.x;
    const int Y = blockIdx.y * blockDim.y + threadIdx.y;
    if (X >= 2 * w || Y >= 2 * h) return;
    float v[3];
    a4kFoldPx(sub, half, in, w, h, Y, X, v, stat, sw, sh);
    if (clip)
        for (int c = 0; c < 3; c++)
            v[c] = v[c] < 0.0f ? 0.0f : (v[c] > 1.0f ? 1.0f : v[c]);
    const size_t plane = (size_t)4 * w * h, o = (size_t)Y * (2 * w) + X;
    dst[o] = v[0];
    dst[plane + o] = v[1];
    dst[2 * plane + o] = v[2];
}

// vertical pass with realesr.fit's final clamp: tmp (3, h, dw) -> the planar target (3, dh, dw)
__global__ void k_restFoldV(const float* __restrict__ tmp, int dw, int h,
                            float* __restrict__ dst, int dh, const RsTap* __restrict__ tab)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    const int fs = rsTaps(h, dh);
    if (x >= dw || oy >= dh) return;
    const size_t splane = (size_t)h * dw;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
    for (int k = 0; k < fs; k++)
    {
        const RsTap t = tab[(size_t)k * dh + oy];
        const float wt = t.w;
        const int iy = t.i;
        const size_t o = (size_t)iy * dw + x;
        a0 += wt * tmp[o];
        a1 += wt * tmp[splane + o];
        a2 += wt * tmp[2 * splane + o];
    }
    const size_t plane = (size_t)dw * dh;
    const size_t o = (size_t)oy * dw + x;
    dst[o] = a0 < 0.0f ? 0.0f : (a0 > 1.0f ? 1.0f : a0);
    dst[plane + o] = a1 < 0.0f ? 0.0f : (a1 > 1.0f ? 1.0f : a1);
    dst[2 * plane + o] = a2 < 0.0f ? 0.0f : (a2 > 1.0f ? 1.0f : a2);
}

// an engine's output clamped to fp32 planar, python's torch.clamp(out, 0, 1) (GMFSS's output)
__global__ void k_restToF(const void* __restrict__ src, int half, int n, float* __restrict__ dst)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = restTap(src, half, (size_t)i);
}

// one RGBA16F texel of a CUDA array through its surface object, the raw halves (the .b forms take x
// in bytes): the NR core's own Color / Output textures on the zero-copy route
__device__ __forceinline__ void nrStore(unsigned long long s, int x, int y, const unsigned short* h)
{
    asm volatile("sust.b.2d.v4.b16.trap [%0, {%1, %2}], {%3, %4, %5, %6};" ::"l"(s), "r"(x * 8), "r"(y), "h"(h[0]),
                 "h"(h[1]), "h"(h[2]), "h"(h[3]) : "memory");
}
__device__ __forceinline__ void nrLoad(unsigned long long s, int x, int y, unsigned short* h)
{
    asm volatile("suld.b.2d.v4.b16.trap {%0, %1, %2, %3}, [%4, {%5, %6}];" : "=h"(h[0]), "=h"(h[1]), "=h"(h[2]),
                 "=h"(h[3]) : "l"(s), "r"(x * 8), "r"(y));
}
// the RGBA16F frame's texel: zero-copy (surf != 0) the core's texture, CPU staging rows ioPitch halves apart
__device__ __forceinline__ void nrPut(unsigned long long surf, unsigned short* dst, int ioPitch, int x, int y,
                                      const unsigned short* h)
{
    if (surf)
        nrStore(surf, x, y, h);
    else
    {
        unsigned short* d = dst + (size_t)y * ioPitch + (size_t)x * 4;
        for (int c = 0; c < 4; c++) d[c] = h[c];
    }
}
__device__ __forceinline__ void nrGet(unsigned long long surf, const unsigned short* src, int ioPitch, int x, int y,
                                      unsigned short* h)
{
    if (surf)
        nrLoad(surf, x, y, h);
    else
    {
        const unsigned short* p = src + (size_t)y * ioPitch + (size_t)x * 4;
        for (int c = 0; c < 3; c++) h[c] = p[c];
    }
}

// offline DLSS 5: dlssnr.py process() around the NR host's RGBA16F frame. In: the dw x dh
// picture of a planar (R, G, B) frame, clamp(0, 1).to(float16) (round to nearest even, torch's
// cast) plus alpha 1.0. Out: the first three channels, .float().clamp(0, 1), written back over
// the whole pw x ph padded frame with the edge replicated into the pad, as k_packInRaw8 / 16
// pad a decoded frame. Zero-copy (surf != 0): the texels go straight into the core's Color texture
// and come straight out of its Output texture, no copy between the kernels and the core; CPU staging:
// dst / src rows ioPitch halves apart (dw x 4).
__global__ void k_nrIn(const float* __restrict__ src, int planeStride, int rowStride, int dw, int dh,
                       unsigned short* __restrict__ dst, int ioPitch, unsigned long long surf)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    const long long i = (long long)y * rowStride + x;
    unsigned short d[4];
    for (int c = 0; c < 3; c++)
    {
        const float v = src[(long long)c * planeStride + i];
        d[c] = f2h(v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v));
    }
    d[3] = 0x3C00;
    nrPut(surf, dst, ioPitch, x, y, d);
}
__global__ void k_nrOut(const unsigned short* __restrict__ src, int ioPitch, int dw, int dh,
                        float* __restrict__ dst, int ph, int pw, int planeStride, unsigned long long surf)
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= pw || oy >= ph) return;
    const int sx = ox < dw ? ox : dw - 1;
    const int sy = oy < dh ? oy : dh - 1;
    unsigned short p[4];
    nrGet(surf, src, ioPitch, sx, sy, p);
    const long long o = (long long)oy * pw + ox;
    for (int c = 0; c < 3; c++)
    {
        const float v = h2f(p[c]);
        dst[(long long)c * planeStride + o] = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
    }
}

// DLSS 5 on HDR planes (PQ BT.2020, or HLG BT.2020 when hlg; (R, G, B)): live's HDR capture and
// offline HDR video. Only the SDR range of the picture goes through the pass. In: back to scRGB (the
// ST 2084 EOTF, 10000 / 80, the inverse of scrgb_to_pq2020's BT.2087 matrix; HLG: hlg2020_to_scrgb),
// normalised by the SDR reference white (scRGB units), clamped to 0..1, the sRGB inverse EOTF: the
// SDR range of the picture. Out: the model's result through the sRGB EOTF times the SDR white, plus
// each channel's light above SDR white (the part the clamp kept from the pass), back to PQ (or HLG).
// So the SDR range gets the pass's result as an SDR picture would, and a highlight keeps its light
// above white on top of the edited picture under it: the output follows the source continuously
// across SDR white, whatever the size of the edit. In place, the edge replicated into the pad as
// k_nrOut does; the RGBA16F rows ioPitch halves apart, as k_nrIn / k_nrOut.
__device__ __forceinline__ void pq2020_to_scrgb(float p0, float p1, float p2, float& r, float& g, float& b)
{
    const float M1 = 0.1593017578125f, M2 = 78.84375f;
    const float C1 = 0.8359375f, C2 = 18.8515625f, C3 = 18.6875f;
    const float p[3] = { p0, p1, p2 };
    float l[3];
    for (int c = 0; c < 3; c++)
    {
        const float n = powf(p[c] < 0.0f ? 0.0f : p[c], 1.0f / M2);
        const float d = n - C1;
        l[c] = powf((d > 0.0f ? d : 0.0f) / (C2 - C3 * n), 1.0f / M1) * (10000.0f / 80.0f);
    }
    r = 1.66051121f * l[0] - 0.58771059f * l[1] - 0.07280062f * l[2];   // the exact inverse of the
    g = -0.12456141f * l[0] + 1.13296051f * l[1] - 0.00839911f * l[2];  // 4-digit matrix above
    b = -0.01816769f * l[0] - 0.10056060f * l[1] + 1.11872828f * l[2];
}
// HLG on BT.2100's reference display (1000 nits peak, system gamma 1.2), so 75 % HLG = 203 nits,
// BT.2408's reference white. In: the inverse OETF (scene light), the OOTF on the scene luminance,
// the BT.2020 -> 709 matrix of pq2020_to_scrgb, scRGB units. Out: the reverse (the 4-digit matrix,
// the inverse OOTF on the display luminance, the OETF).
__device__ __forceinline__ void hlg2020_to_scrgb(float e0, float e1, float e2, float& r, float& g, float& b)
{
    const float A = 0.17883277f, B = 0.28466892f, C = 0.55991073f;
    const float e[3] = { e0, e1, e2 };
    float s[3];
    for (int c = 0; c < 3; c++)
    {
        const float v = e[c] < 0.0f ? 0.0f : (e[c] > 1.0f ? 1.0f : e[c]);
        s[c] = v <= 0.5f ? v * v / 3.0f : (expf((v - C) / A) + B) / 12.0f;
    }
    const float ys = 0.2627f * s[0] + 0.6780f * s[1] + 0.0593f * s[2];
    const float k = ys > 0.0f ? (1000.0f / 80.0f) * powf(ys, 0.2f) : 0.0f;
    const float l[3] = { s[0] * k, s[1] * k, s[2] * k };
    r = 1.66051121f * l[0] - 0.58771059f * l[1] - 0.07280062f * l[2];
    g = -0.12456141f * l[0] + 1.13296051f * l[1] - 0.00839911f * l[2];
    b = -0.01816769f * l[0] - 0.10056060f * l[1] + 1.11872828f * l[2];
}
__device__ __forceinline__ void scrgb_to_hlg2020(float r, float g, float b, float& o0, float& o1, float& o2)
{
    const float A = 0.17883277f, B = 0.28466892f, C = 0.55991073f;
    float l[3] = { 0.6274f * r + 0.3293f * g + 0.0433f * b, 0.0691f * r + 0.9195f * g + 0.0114f * b,
                   0.0164f * r + 0.0880f * g + 0.8956f * b };
    for (int c = 0; c < 3; c++)
        l[c] = (l[c] < 0.0f ? 0.0f : l[c]) * (80.0f / 1000.0f);
    const float yd = 0.2627f * l[0] + 0.6780f * l[1] + 0.0593f * l[2];
    const float k = yd > 0.0f ? powf(yd, -0.2f / 1.2f) : 0.0f;
    float o[3];
    for (int c = 0; c < 3; c++)
    {
        const float s = l[c] * k > 1.0f ? 1.0f : l[c] * k;
        o[c] = s <= 1.0f / 12.0f ? sqrtf(3.0f * s) : A * logf(12.0f * s - B) + C;
    }
    o0 = o[0];
    o1 = o[1];
    o2 = o[2];
}
__device__ __forceinline__ void hdr2020_to_scrgb(float p0, float p1, float p2, int hlg, float& r, float& g, float& b)
{
    if (hlg) hlg2020_to_scrgb(p0, p1, p2, r, g, b);
    else pq2020_to_scrgb(p0, p1, p2, r, g, b);
}
__device__ __forceinline__ float srgbOetf(float c)
{ return c <= 0.0031308f ? c * 12.92f : 1.055f * powf(c > 1e-6f ? c : 1e-6f, 1.0f / 2.4f) - 0.055f; }
__device__ __forceinline__ float srgbEotf(float g)
{ return g <= 0.04045f ? g / 12.92f : powf((g + 0.055f) / 1.055f, 2.4f); }
__global__ void k_nrInPq(const float* __restrict__ src, int planeStride, int rowStride, int dw, int dh,
                         float sdrWhite, unsigned short* __restrict__ dst, int ioPitch, int hlg,
                         unsigned long long surf)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    const long long i = (long long)y * rowStride + x;
    float v[3];
    hdr2020_to_scrgb(src[i], src[(long long)planeStride + i], src[2LL * planeStride + i], hlg, v[0], v[1], v[2]);
    unsigned short d[4];
    for (int c = 0; c < 3; c++)
    {
        const float s = v[c] / sdrWhite;
        d[c] = f2h(srgbOetf(s < 0.0f ? 0.0f : (s > 1.0f ? 1.0f : s)));
    }
    d[3] = 0x3C00;
    nrPut(surf, dst, ioPitch, x, y, d);
}
__global__ void k_nrOutPq(const unsigned short* __restrict__ src, int ioPitch, int dw, int dh, float sdrWhite,
                          float* __restrict__ dst, int ph, int pw, int planeStride, int hlg, unsigned long long surf)
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= pw || oy >= ph) return;
    const long long o = (long long)oy * pw + ox;
    float in[3];
    hdr2020_to_scrgb(dst[o], dst[(long long)planeStride + o], dst[2LL * planeStride + o], hlg, in[0], in[1], in[2]);
    const int sx = ox < dw ? ox : dw - 1;
    const int sy = oy < dh ? oy : dh - 1;
    unsigned short p[4];
    nrGet(surf, src, ioPitch, sx, sy, p);
    float q[3];
    for (int c = 0; c < 3; c++)
    {
        const float v = h2f(p[c]);
        q[c] = srgbEotf(v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v)) * sdrWhite;
        if (in[c] > sdrWhite) q[c] += in[c] - sdrWhite;   // the light above SDR white the pass never saw
    }
    float a0, a1, a2;
    if (hlg) scrgb_to_hlg2020(q[0], q[1], q[2], a0, a1, a2);
    else scrgb_to_pq2020(q[0], q[1], q[2], a0, a1, a2);
    dst[o] = a0;
    dst[(long long)planeStride + o] = a1;
    dst[2LL * planeStride + o] = a2;
}

// live, RTX TrueHDR failed on this frame: the SDR model frame (sRGB-encoded against the SDR reference white,
// the whole padded frame, in place) to PQ BT.2020 the faithful way, as the capture path converts
__global__ void k_sdrPq(float* __restrict__ dst, int pw, int ph, int planeStride, float sdrWhite)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= pw || y >= ph) return;
    const long long o = (long long)y * pw + x;
    float q[3];
    for (int c = 0; c < 3; c++)
    {
        const float v = dst[(long long)c * planeStride + o];
        q[c] = srgbEotf(v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v)) * sdrWhite;
    }
    float a0, a1, a2;
    scrgb_to_pq2020(q[0], q[1], q[2], a0, a1, a2);
    dst[o] = a0;
    dst[(long long)planeStride + o] = a1;
    dst[2LL * planeStride + o] = a2;
}

// offline DLSS 5 motion (DLSSNR.MVec): the Optical Flow field current -> previous of the NR input
// frames (px, two w x h planes, k_nvofUp's output) as R16G16_FLOAT texels stored into the core's
// MVec texture (its surface object dst, x in bytes). A vector is kept where it explains its 5x5 window of the current
// frame better than no motion by more than margin (the summed 8-bit luma error: the previous frame bilinear at the
// moved position against the same window unmoved, border clamped), or where the window really changed (its no-motion
// error above realChange) and the vector removes at least keepShare of that change: a dark, low-contrast moving edge
// changes its window by less than the margin, the absolute test alone zeroed its right vector, and DLSS 5 then held
// its history where the edge was (a trail per pass in a chain). An estimated field reports small vectors on still
// content and grain, and DLSS 5 would pull its history along them: a still window does not change (no relative keep)
// and grain's change is not explained by a vector. Zero elsewhere: what a still pixel gets from a game engine. The
// moved tap is the pixel plus the floor and the exact fraction of the vector, never an fp32 absolute coordinate.
__global__ void k_nrMv(const float* __restrict__ flow, const unsigned char* __restrict__ cur,
                       const unsigned char* __restrict__ prev, int lumaPitch, int w, int h,
                       float margin, float realChange, float keepShare, float flatMin, float cohTol,
                       unsigned long long dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t o = (size_t)y * w + x;
    const float u = flow[o], v = flow[(size_t)w * h + o];
    unsigned int out = 0;
    if (u != 0.0f || v != 0.0f)
    {
        const float fu = floorf(u), fv = floorf(v);
        const int iu = (int)fu, iv = (int)fv;
        const float lx = u - fu, ly = v - fv;
        float e0 = 0.0f, e1 = 0.0f;
        for (int dy = -2; dy <= 2; dy++)
        {
            const int yy = min(max(y + dy, 0), h - 1);
            const size_t r0 = (size_t)min(max(yy + iv, 0), h - 1) * lumaPitch;
            const size_t r1 = (size_t)min(max(yy + iv + 1, 0), h - 1) * lumaPitch;
            for (int dx = -2; dx <= 2; dx++)
            {
                const int xx = min(max(x + dx, 0), w - 1);
                const int x0 = min(max(xx + iu, 0), w - 1), x1 = min(max(xx + iu + 1, 0), w - 1);
                const float c = (float)cur[(size_t)yy * lumaPitch + xx];
                const float a = (1.0f - lx) * (float)prev[r0 + x0] + lx * (float)prev[r0 + x1];
                const float b = (1.0f - lx) * (float)prev[r1 + x0] + lx * (float)prev[r1 + x1];
                e0 += fabsf(c - (float)prev[(size_t)yy * lumaPitch + xx]);
                e1 += fabsf(c - ((1.0f - ly) * a + ly * b));
            }
        }
        // a window neither test decides (flat content: the sky behind a moving edge changes too little to tell) keeps
        // a vector of flatMin px or more unless no motion explains it clearly better (still text over a pan), when
        // the field agrees with it 8 px away on all four sides within cohTol px (a pan moves its neighbours alike;
        // the vectors the flow reports on grain are as long but scattered): the history there still holds what moved
        // through it, and zero motion would keep that image in place
        const bool vecWins = e1 + margin < e0 || (keepShare > 0.0f && e0 > realChange && e1 <= (1.0f - keepShare) * e0);
        const bool zeroWins = e0 + margin < e1;
        bool flatKeep = flatMin > 0.0f && !zeroWins && u * u + v * v >= flatMin * flatMin;
        if (flatKeep && cohTol > 0.0f)
        {
            const int nx[4] = {max(x - 8, 0), min(x + 8, w - 1), x, x};
            const int ny[4] = {y, y, max(y - 8, 0), min(y + 8, h - 1)};
            for (int k = 0; k < 4; k++)
            {
                const size_t q = (size_t)ny[k] * w + nx[k];
                const float du = flow[q] - u, dv = flow[(size_t)w * h + q] - v;
                if (du * du + dv * dv > cohTol * cohTol) flatKeep = false;
            }
        }
        if (vecWins || flatKeep)
            out = (unsigned int)f2h(u) | ((unsigned int)f2h(v) << 16);
    }
    asm volatile("sust.b.2d.b32.trap [%0, {%1, %2}], {%3};" ::"l"(dst), "r"(x * 4), "r"(y), "r"(out) : "memory");
}
// a zero MVec texture: a stream's first frame, a Reset, motion that could not start
__global__ void k_nrMvClear(unsigned long long dst, int w, int h)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    asm volatile("sust.b.2d.b32.trap [%0, {%1, %2}], {%3};" ::"l"(dst), "r"(x * 4), "r"(y), "r"(0u) : "memory");
}

__global__ void k_clamp01(float* __restrict__ p, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { const float v = p[i]; p[i] = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }
}

// ---- live GMFSS glue ------------------------------------------------------------------------
// The eager glue of GMFSS_infer_u.Model.inference() as four kernels (the chain in
// nativeGmfssPair / nativeGmfssTween calls them). Every buffer is planar NCHW with N = 1, contiguous unless a stride is passed.
//   k_half      F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False) on an
//               even-sized frame = the 2x2 box (torch's lambdas are exactly 0.5, and scaling by
//               a power of two is exact, so the sum order below reproduces torch's fp32 result
//               bit for bit). The caller offsets dst so img0's and img1's halves land as the
//               adjacent planes of one (6, hh, hw) buffer: ifnet's x as is, the gmflow and
//               metricnet inputs by pointer offset, the two image splats likewise.
//   k_pyr       the per-PAIR flow and metric pyramids: level 2 = the 2x2 box of taps (2o, 2o+1)
//               (scale_factor 0.5), level 4 = the 2x2 box of taps (4o+1, 4o+2) (scale_factor
//               0.25: torch's source centre 4o+1.5 sits exactly between them). The flow is
//               multiplied by the level's 0.5 / 0.25 (python's `* 0.5`, `* 0.25`), a metric by
//               1. The taps are integers, no fp centre exists to drift. The timestep is NOT
//               folded in: python's `interp(t * flow) * 0.5` equals `t * (interp(flow) * 0.5)`,
//               so the pyramids are built once per pair and k_splatSoft applies s.
//   k_splatSoft softsplat.py's softsplat_out_det (the SMV int64 fixed-point forward) with the
//               'soft' mode's cat(in * exp(Z), exp(Z)) computed on the fly in fp32: Z = s *
//               metric, the flow = s * flow, s = t for the 0 -> 1 splat, 1 - t for the 1 -> 0
//               splat. Same four corner weights, same `llrintf(v * w * 2^26)` product order,
//               same int64 atomics; the cell and the weights come from the offset s * flow
//               (floor + exact fraction, below), so the accumulators match python's kernel bit
//               for bit only where the position is exactly representable (the harness's
//               quantized-flow control) and sit nearer the exact splat than python's fp32
//               `x + s * flow` everywhere else; python also takes exp and the product in fp16
//               under autocast (~1e-3 off fp64), this kernel in fp32.
//   k_splatNorm python's tail: acc.to(float32) * 2^-26 per plane, out = v / (vN + 1e-7),
//               written as C planes of the target (the fusionnet input planes: a channel concat
//               is adjacent planes, so no cat).
// Checked against fp64 references on real-motion inputs through the engines.
__device__ __forceinline__ float gmTap(const void* __restrict__ src, int half, size_t o)
{
    return half ? h2f(((const unsigned short*)src)[o]) : ((const float*)src)[o];
}
__device__ __forceinline__ void gmStore(void* __restrict__ dst, int half, size_t o, float v)
{
    if (half) ((unsigned short*)dst)[o] = f2h(v); else ((float*)dst)[o] = v;
}
// fusionnet's input LeakyRelu (slope lr, lr < 0 = none) on a stored fp16 value: the fp32 product of two fp16 values is
// exact, so its one rounding = an fp16 multiply's result bit for bit
__device__ __forceinline__ unsigned short gmLrelu16(unsigned short h, float lr)
{
    const float f = h2f(h);
    return lr >= 0.0f && f < 0.0f ? f2h(f * lr) : h;
}
__device__ __forceinline__ void gmStoreLr(void* __restrict__ dst, int half, size_t o, float v, float lr)
{
    if (half) ((unsigned short*)dst)[o] = gmLrelu16(f2h(v), lr);
    else ((float*)dst)[o] = lr >= 0.0f && v < 0.0f ? v * lr : v;
}

// C planes of (2 hh, 2 hw) with strides (planeStride, rowStride) -> C contiguous planes of (hh, hw)
__global__ void k_half(const void* __restrict__ src, int srcHalf, int planeStride, int rowStride,
                       int C, int hw, int hh, void* __restrict__ dst, int dstHalf)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= hw || y >= hh) return;
    const size_t s0 = (size_t)(2 * y) * rowStride + 2 * x, s1 = s0 + rowStride;
    const size_t plane = (size_t)hw * hh, o = (size_t)y * hw + x;
    for (int c = 0; c < C; c++)
    {
        const size_t p = (size_t)c * planeStride;
        const float v = ((gmTap(src, srcHalf, p + s0) + gmTap(src, srcHalf, p + s0 + 1))
                       + (gmTap(src, srcHalf, p + s1) + gmTap(src, srcHalf, p + s1 + 1))) * 0.25f;
        gmStore(dst, dstHalf, (size_t)c * plane + o, v);
    }
}

// C planes of (sh, sw) (the fp32 flow or the fp16 metric) -> C fp32 planes of (sh / level,
// sw / level); level 2 or 4, scale = the flow's 0.5 / 0.25, 1 for a metric
__global__ void k_pyr(const void* __restrict__ src, int srcHalf, int C, int sw, int sh, int level,
                      float scale, float* __restrict__ dst, int dw, int dh)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    const int sx = level == 4 ? 4 * x + 1 : 2 * x;
    const int sy = level == 4 ? 4 * y + 1 : 2 * y;
    const size_t splane = (size_t)sw * sh, s0 = (size_t)sy * sw + sx, s1 = s0 + sw;
    const size_t plane = (size_t)dw * dh, o = (size_t)y * dw + x;
    for (int c = 0; c < C; c++)
    {
        const size_t p = (size_t)c * splane;
        const float v = ((gmTap(src, srcHalf, p + s0) + gmTap(src, srcHalf, p + s0 + 1))
                       + (gmTap(src, srcHalf, p + s1) + gmTap(src, srcHalf, p + s1 + 1))) * 0.25f;
        dst[(size_t)c * plane + o] = v * scale;
    }
}

// one source pixel of (w, h): C planes of `in` (fp16 features or the fp32 image half), the flow
// (2, h, w) fp32 (plane 0 = x), the metric (1, h, w) (fp16 from the engine, fp32 from k_pyr),
// both scaled by s, splatted into C + 1 int64 planes (zeroed by the caller; plane C = exp(Z)).
// Neighbour combine: products bound for the SAME target pixel are summed
// before one atomic. Horizontal: lane i's NW / SW corner is lane i - 1's NE / SE corner (same
// target coordinates) -> lane i adds both, lane i - 1 skips (warp shuffle). Vertical: the
// thread one block row below whose NW / NE corners are my SW / SE corners takes my bottom row
// (shared memory) and I skip it. Integer adds mod 2^64 are order-free, so the accumulators stay
// bit-identical; smooth flow drops from 4 to ~1 atomic per pixel and channel. Keyed on the
// target coordinates, not on the thread layout, so any block shape is correct; the host
// launches 32 x 8 (one warp = 32 consecutive x of one row). The block must be a multiple of 32
// threads and at most 256. No early return: every thread joins the shuffles and barriers.
// Checked: the accumulators equal a one-atomic-per-corner splat's bit for bit.
// `in` channel c of pixel o at c * cs + o * ps (planar: cs = the plane, ps = 1; channels-last: cs = 1, ps = C)
__global__ void k_splatSoft(const void* __restrict__ in, int inHalf, int C, int cs, int ps,
                            const float* __restrict__ flow, const void* __restrict__ metric,
                            int metricHalf, float s, int w, int h, long long* __restrict__ acc)
{
    __shared__ int shX[256], shY[256], shA[256];
    __shared__ unsigned long long shV[2][2][256];     // [channel parity][SW, SE][thread]
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned lane = (unsigned)tid & 31u;
    const size_t plane = (size_t)w * h;
    bool act = x < w && y < h;
    const size_t o = act ? (size_t)y * w + x : 0;
    // the target cell and the corner weights from land(), never an fp32 absolute position:
    // python's fp32 `x + s * flow` carries a 6e-5 px error at x ~ 600..960 (one fp32 ulp), which
    // measures up to 8.7e-4 of the normalized output where the weight sum is small
    int nwX = 0, nwY = 0;
    float dx = 0.0f, dy = 0.0f;
    if (act)
    {
        act = land(x, s, flow[o], nwX, dx) && land(y, s, flow[plane + o], nwY, dy);
        if (!act) { nwX = 0; nwY = 0; dx = 0.0f; dy = 0.0f; }
    }
    const float e = act ? expf(__fmul_rn(s, gmTap(metric, metricHalf, o))) : 0.0f;
    const float wNW = (1.0f - dx) * (1.0f - dy), wNE = dx * (1.0f - dy);
    const float wSW = (1.0f - dx) * dy, wSE = dx * dy;
    const bool okW = act && nwX >= 0 && nwX < w, okE = act && nwX + 1 >= 0 && nwX + 1 < w;
    const bool okN = nwY >= 0 && nwY < h, okS = nwY + 1 >= 0 && nwY + 1 < h;
    const long long oNW = (long long)nwY * w + nwX;   // only dereferenced behind the ok flags
    // take: my NW / SW corners are lane - 1's NE / SE corners (same pixels, so the same ok
    // flags); give: lane + 1 takes mine
    const int pX = __shfl_up_sync(0xffffffffu, nwX, 1), pY = __shfl_up_sync(0xffffffffu, nwY, 1);
    const int pAct = __shfl_up_sync(0xffffffffu, act ? 1 : 0, 1);
    const bool take = act && lane > 0 && pAct && pX + 1 == nwX && pY == nwY;
    const bool give = __shfl_down_sync(0xffffffffu, take ? 1 : 0, 1) && lane < 31;
    // vtake: my NW / NE corners are the SW / SE corners of the thread one block row above;
    // vgive: the thread below takes my bottom row (the same test from both ends)
    shX[tid] = nwX;
    shY[tid] = nwY;
    shA[tid] = act ? 1 : 0;
    __syncthreads();
    const int up = tid - (int)blockDim.x, dn = tid + (int)blockDim.x;
    const int nT = (int)(blockDim.x * blockDim.y);
    const bool vtake = act && up >= 0 && shA[up] && shX[up] == nwX && shY[up] + 1 == nwY;
    const bool vgive = act && dn < nT && shA[dn] && shX[dn] == nwX && shY[dn] == nwY + 1;
    for (int c = 0; c <= C; c++)
    {
        const float v = !act ? 0.0f : c < C ? __fmul_rn(gmTap(in, inHalf, (size_t)c * cs + o * ps), e) : e;
        unsigned long long qNW = (unsigned long long)(long long)llrintf(v * wNW * 67108864.0f);
        unsigned long long qNE = (unsigned long long)(long long)llrintf(v * wNE * 67108864.0f);
        const unsigned long long qSW = (unsigned long long)(long long)llrintf(v * wSW * 67108864.0f);
        const unsigned long long qSE = (unsigned long long)(long long)llrintf(v * wSE * 67108864.0f);
        // bottom row, combined across the row first: my SW pixel also holds lane - 1's SE
        const unsigned long long lSE = __shfl_up_sync(0xffffffffu, qSE, 1);
        const unsigned long long sSW = take ? qSW + lSE : qSW, sSE = give ? 0ull : qSE;
        const int b = c & 1;
        shV[b][0][tid] = sSW;
        shV[b][1][tid] = sSE;
        __syncthreads();
        // top row: add the row above's bottom row, then combine across the row
        if (vtake) { qNW += shV[b][0][up]; qNE += shV[b][1][up]; }
        const unsigned long long lNE = __shfl_up_sync(0xffffffffu, qNE, 1);
        unsigned long long* p = (unsigned long long*)(acc + (size_t)c * plane);
        if (okN && okW) atomicAdd(p + oNW, take ? qNW + lNE : qNW);
        if (okN && okE && !give) atomicAdd(p + oNW + 1, qNE);
        if (!vgive)
        {
            if (okS && okW) atomicAdd(p + oNW + w, sSW);
            if (okS && okE && !give) atomicAdd(p + oNW + w + 1, sSE);
        }
    }
}

// acc (C + 1 int64 planes of n) -> C channels of the target (fp32, or fp16 when dstHalf), channel c of cell i at
// c * cs + i * ps (planar: cs = n, ps = 1; channels-last: cs = 1, ps = the target's channel count); lr = gmStoreLr's
__global__ void k_splatNorm(const long long* __restrict__ acc, int C, int n,
                            void* __restrict__ dst, int dstHalf, int cs, int ps, float lr)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float norm = __fadd_rn((float)acc[(size_t)C * n + i] * (1.0f / 67108864.0f), 0.0000001f);
    for (int c = 0; c < C; c++)
        gmStoreLr(dst, dstHalf, (size_t)c * cs + (size_t)i * ps,
                  __fdiv_rn((float)acc[(size_t)c * n + i] * (1.0f / 67108864.0f), norm), lr);
}

// The gather form of k_splatSoft + k_splatNorm, the same output bit for bit without accumulator planes: a list per
// target cell of the (source pixel, corner) pairs landing on it, then one thread per cell adds the SAME int64
// fixed-point products k_splatSoft adds (integer adds are order-free, so the list order does not matter) and divides
// like k_splatNorm. One list serves every input splatted with one flow, metric and s (the image and the features at
// the half). k_gsCount: per source pixel k_splatSoft's landing and e, kept as e / dx / dy planes, and a count per
// landed cell; k_gsScanA / B / C: the counts -> exclusive list offsets (off[n] = the total) and a fill cursor;
// k_gsFill: entry = source * 4 + corner (0 NW, 1 NE, 2 SW, 3 SE).
__device__ __forceinline__ bool gsLand(int x, int y, float s, const float* __restrict__ flow, int w, int h, int& nwX,
                                       int& nwY, float& dx, float& dy)
{
    const size_t plane = (size_t)w * h, o = (size_t)y * w + x;
    return land(x, s, flow[o], nwX, dx) && land(y, s, flow[plane + o], nwY, dy);
}

__global__ void k_gsCount(const float* __restrict__ flow, const void* __restrict__ metric, int metricHalf, float s,
                          int w, int h, int* __restrict__ cnt, float* __restrict__ src)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t plane = (size_t)w * h, o = (size_t)y * w + x;
    int nwX, nwY;
    float dx, dy;
    if (!gsLand(x, y, s, flow, w, h, nwX, nwY, dx, dy)) return;
    src[o] = expf(__fmul_rn(s, gmTap(metric, metricHalf, o)));
    src[plane + o] = dx;
    src[2 * plane + o] = dy;
    const bool okW = nwX >= 0 && nwX < w, okE = nwX + 1 >= 0 && nwX + 1 < w;
    const bool okN = nwY >= 0 && nwY < h, okS = nwY + 1 >= 0 && nwY + 1 < h;
    const long long oNW = (long long)nwY * w + nwX;
    if (okN && okW) atomicAdd(cnt + oNW, 1);
    if (okN && okE) atomicAdd(cnt + oNW + 1, 1);
    if (okS && okW) atomicAdd(cnt + oNW + w, 1);
    if (okS && okE) atomicAdd(cnt + oNW + w + 1, 1);
}

// the exclusive prefix sum of v over a block of 1024 threads; *total = the block's sum (sh = 32 ints)
__device__ int gsBlockScan(int v, int* sh, int* total)
{
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    int x = v;
    for (int d = 1; d < 32; d <<= 1)
    {
        const int t = __shfl_up_sync(0xffffffffu, x, d);
        if (lane >= d) x += t;
    }
    if (lane == 31) sh[wid] = x;
    __syncthreads();
    if (wid == 0)
    {
        int y = sh[lane];
        for (int d = 1; d < 32; d <<= 1)
        {
            const int t = __shfl_up_sync(0xffffffffu, y, d);
            if (lane >= d) y += t;
        }
        sh[lane] = y;
    }
    __syncthreads();
    const int r = (wid ? sh[wid - 1] : 0) + x - v;
    *total = sh[31];
    __syncthreads();
    return r;
}

// 2048 counts a block of 1024 threads -> block-local offsets + the block's sum
__global__ void k_gsScanA(const int* __restrict__ cnt, int n, int* __restrict__ off, int* __restrict__ blk)
{
    __shared__ int sh[32];
    const int i = blockIdx.x * 2048 + 2 * threadIdx.x;
    const int a = i < n ? cnt[i] : 0, b = i + 1 < n ? cnt[i + 1] : 0;
    int total;
    const int ex = gsBlockScan(a + b, sh, &total);
    if (i < n) off[i] = ex;
    if (i + 1 < n) off[i + 1] = ex + a;
    if (threadIdx.x == 0) blk[blockIdx.x] = total;
}

// one block of 1024 threads: up to 4096 block sums -> their exclusive offsets; off[n] = the total
__global__ void k_gsScanB(int* __restrict__ blk, int nb, int* __restrict__ off, int n)
{
    __shared__ int sh[32];
    const int i = 4 * threadIdx.x;
    int v[4], sum = 0;
    for (int k = 0; k < 4; k++)
    {
        v[k] = i + k < nb ? blk[i + k] : 0;
        sum += v[k];
    }
    int total;
    int ex = gsBlockScan(sum, sh, &total);
    for (int k = 0; k < 4; k++)
    {
        if (i + k < nb) blk[i + k] = ex;
        ex += v[k];
    }
    if (threadIdx.x == 0) off[n] = total;
}

// the block offsets added: off = the list starts, cursor = a copy the fill advances
__global__ void k_gsScanC(int* __restrict__ off, const int* __restrict__ blk, int n, int* __restrict__ cursor)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int v = off[i] + blk[i / 2048];
    off[i] = v;
    cursor[i] = v;
}

__global__ void k_gsFill(const float* __restrict__ flow, float s, int w, int h, int* __restrict__ cursor,
                         int* __restrict__ list)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int nwX, nwY;
    float dx, dy;
    if (!gsLand(x, y, s, flow, w, h, nwX, nwY, dx, dy)) return;
    const bool okW = nwX >= 0 && nwX < w, okE = nwX + 1 >= 0 && nwX + 1 < w;
    const bool okN = nwY >= 0 && nwY < h, okS = nwY + 1 >= 0 && nwY + 1 < h;
    const long long oNW = (long long)nwY * w + nwX;
    const int e0 = (y * w + x) * 4;
    if (okN && okW) list[atomicAdd(cursor + oNW, 1)] = e0;
    if (okN && okE) list[atomicAdd(cursor + oNW + 1, 1)] = e0 + 1;
    if (okS && okW) list[atomicAdd(cursor + oNW + w, 1)] = e0 + 2;
    if (okS && okE) list[atomicAdd(cursor + oNW + w + 1, 1)] = e0 + 3;
}

// llrintf(x) through the 32-bit conversion where it fits: __float2int_rn rounds to nearest even like llrintf, so
// |x| < 2^31 gives the same integer; larger values and NaN take llrintf
__device__ __forceinline__ long long gsQ(float x)
{
    return fabsf(x) < 2147483648.0f ? (long long)__float2int_rn(x) : llrintf(x);
}

// one list entry: its source pixel, e and corner weight (k_splatSoft's wNW / wNE / wSW / wSE expressions)
__device__ __forceinline__ void gsEntry(int ent, const float* __restrict__ src, size_t plane, int& o, float& e,
                                        float& wc)
{
    o = ent >> 2;
    e = src[o];
    const float dx = src[plane + o], dy = src[2 * plane + o];
    const int k = ent & 3;
    wc = k == 0 ? (1.0f - dx) * (1.0f - dy) : k == 1 ? dx * (1.0f - dy) : k == 2 ? (1.0f - dx) * dy : dx * dy;
}

// one target cell and up to 8 channels (blockIdx.y = the channel group): k_splatSoft's int64 sums over the cell's list
// (the normalizer = e, then its channels of `in`), divided like k_splatNorm into those planes of dst (lr = gmStoreLr's);
// the first 8 entries stay in registers across the channels
__global__ void k_gsGather(const void* __restrict__ in, int inHalf, int C, const float* __restrict__ src, int w, int h,
                           const int* __restrict__ off, const int* __restrict__ list, void* __restrict__ dst,
                           int dstHalf, float lr)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = w * h;
    const int c0 = (int)blockIdx.y * 8, c1 = c0 + 8 < C ? c0 + 8 : C;
    if (i >= n) return;
    const size_t plane = (size_t)n;
    const int b = off[i], m = off[i + 1] - b;
    int so[8];
    float se[8], sw[8];
    unsigned long long accN = 0;
#pragma unroll
    for (int k = 0; k < 8; k++)
    {
        so[k] = 0;
        se[k] = 0.0f;
        sw[k] = 0.0f;
        if (k < m)
        {
            gsEntry(list[b + k], src, plane, so[k], se[k], sw[k]);
            accN += (unsigned long long)gsQ(se[k] * sw[k] * 67108864.0f);
        }
    }
    for (int k = 8; k < m; k++)
    {
        int o;
        float e, wc;
        gsEntry(list[b + k], src, plane, o, e, wc);
        accN += (unsigned long long)gsQ(e * wc * 67108864.0f);
    }
    const float norm = __fadd_rn((float)(long long)accN * (1.0f / 67108864.0f), 0.0000001f);
    for (int c = c0; c < c1; c++)
    {
        const size_t pc = (size_t)c * plane;
        unsigned long long acc = 0;
#pragma unroll
        for (int k = 0; k < 8; k++)
            if (k < m)
            {
                const float v = __fmul_rn(gmTap(in, inHalf, pc + so[k]), se[k]);
                acc += (unsigned long long)gsQ(v * sw[k] * 67108864.0f);
            }
        for (int k = 8; k < m; k++)
        {
            int o;
            float e, wc;
            gsEntry(list[b + k], src, plane, o, e, wc);
            const float v = __fmul_rn(gmTap(in, inHalf, pc + o), e);
            acc += (unsigned long long)gsQ(v * wc * 67108864.0f);
        }
        gmStoreLr(dst, dstHalf, pc + i, __fdiv_rn((float)(long long)acc * (1.0f / 67108864.0f), norm), lr);
    }
}

// k_gsGather on channels-last fp16 features (`in` [n][C], C a multiple of 8) into a channels-last fp16 target ([n][dstC],
// this splat's channels from dstOff): a thread = one cell's 8 channels, a cell's channel groups on consecutive lanes
// (blockDim.x = C / 8, cells on threadIdx.y), so a warp reads whole feature vectors; per list entry ONE 16-byte load, per
// cell ONE 16-byte store. The int64 sums, the per-term products and the normalizer are k_gsGather's: the same values
// bit for bit (gmStore's f2h rounding, then gmStoreLr's slope lr).
__global__ void k_gsGatherNhwc(const unsigned short* __restrict__ in, int C, const float* __restrict__ src, int w, int h,
                               const int* __restrict__ off, const int* __restrict__ list,
                               unsigned short* __restrict__ dst, int dstC, int dstOff, float lr)
{
    const int i = blockIdx.x * blockDim.y + threadIdx.y;
    if (i >= w * h)
        return;
    const int c0 = (int)threadIdx.x * 8;
    const size_t plane = (size_t)w * h;
    const int b = off[i], m = off[i + 1] - b;
    int so[8];
    float se[8], sw[8];
    unsigned long long accN = 0;
#pragma unroll
    for (int k = 0; k < 8; k++)
    {
        so[k] = 0;
        se[k] = 0.0f;
        sw[k] = 0.0f;
        if (k < m)
        {
            gsEntry(list[b + k], src, plane, so[k], se[k], sw[k]);
            accN += (unsigned long long)gsQ(se[k] * sw[k] * 67108864.0f);
        }
    }
    for (int k = 8; k < m; k++)
    {
        int o;
        float e, wc;
        gsEntry(list[b + k], src, plane, o, e, wc);
        accN += (unsigned long long)gsQ(e * wc * 67108864.0f);
    }
    const float norm = __fadd_rn((float)(long long)accN * (1.0f / 67108864.0f), 0.0000001f);
    unsigned long long acc[8];
#pragma unroll
    for (int j = 0; j < 8; j++)
        acc[j] = 0;
#pragma unroll
    for (int k = 0; k < 8; k++)
        if (k < m)
        {
            const uint4 q = *(const uint4*)(in + (size_t)so[k] * C + c0);
            const unsigned int wd[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
            for (int j = 0; j < 8; j++)
            {
                const float v = __fmul_rn(h2f((unsigned short)(wd[j >> 1] >> (16 * (j & 1)))), se[k]);
                acc[j] += (unsigned long long)gsQ(v * sw[k] * 67108864.0f);
            }
        }
    for (int k = 8; k < m; k++)
    {
        int o;
        float e, wc;
        gsEntry(list[b + k], src, plane, o, e, wc);
        const uint4 q = *(const uint4*)(in + (size_t)o * C + c0);
        const unsigned int wd[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            const float v = __fmul_rn(h2f((unsigned short)(wd[j >> 1] >> (16 * (j & 1)))), e);
            acc[j] += (unsigned long long)gsQ(v * wc * 67108864.0f);
        }
    }
    unsigned int pk[4];
#pragma unroll
    for (int j = 0; j < 4; j++)
        pk[j] = (unsigned int)gmLrelu16(f2h(__fdiv_rn((float)(long long)acc[2 * j] * (1.0f / 67108864.0f), norm)), lr) |
                ((unsigned int)gmLrelu16(f2h(__fdiv_rn((float)(long long)acc[2 * j + 1] * (1.0f / 67108864.0f), norm)),
                                         lr)
                 << 16);
    *(uint4*)(dst + (size_t)i * dstC + dstOff + c0) = make_uint4(pk[0], pk[1], pk[2], pk[3]);
}

// Adaptive Sharpen: bacondither's DX11 2-pass HQ (version 2021-09-10, quality_mode 1; Copyright (c) 2015-2021,
// bacondither, BSD-2-Clause, see THIRD_PARTY_NOTICES.md). Both passes run in one kernel per 16 x 16 block through
// shared memory: the colour tile with a 5 px border, then the edge (pass 1) and the luma of every position pass 2
// reads (a 3 px border), each computed once. Taps clamp to the frame like the shader's texture reads, so an edge read
// past the border is the border pixel's edge. The edge is kept raw (the shader's + 2.0 alpha offset only kept its
// texture value positive), the bounds-check debug guard is left out, the result is clamped to 0..1.
// curve = the strength (curve_height; 1 = the author's default). Launch with 16 x 16 blocks.
#define AS_T 16
#define AS_C (AS_T + 10)
#define AS_E (AS_T + 6)

struct AsTile
{
    float c[3][AS_C][AS_C];
    float e[AS_E][AS_E];
    float l[AS_E][AS_E];
};

__device__ __forceinline__ float asSat(float x)
{
    return fminf(fmaxf(x, 0.0f), 1.0f);
}

// the math uses the GPU's fast exp / pow / divide / sqrt: within 1 code of the precise form on 0.001 % of the samples,
// about 1.8x faster
__device__ __forceinline__ float asSqrtA(float x)
{
    float r;
    asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ float asSmooth(float e0, float e1, float x)
{
    const float t = asSat(__fdividef(x - e0, e1 - e0));
    return t * t * (3.0f - 2.0f * t);
}

__device__ __forceinline__ float asSoftLim(float v, float s)
{
    const float z = __expf(__fdividef(2.0f * fminf(fabsf(v), s * 24.0f), s));
    return __fdividef(z - 1.0f, z + 1.0f) * s;
}

__device__ __forceinline__ float asWpmean(float a, float b, float w)
{
    return __powf(w * __powf(fabsf(a), 0.7f) + fabsf(1.0f - w) * __powf(fabsf(b), 0.7f), 1.0f / 0.7f);
}

// every thread of the block: the colour tile, then pass 1's edge and the luma at each position pass 2 reads. src =
// planes ps apart, rows rs apart, dw x dh used (a padded frame reads in place: the clamp never reaches its pad)
__device__ void asLoad(AsTile& s, const float* __restrict__ src, int ps, int rs, int dw, int dh)
{
    const int x0 = blockIdx.x * AS_T, y0 = blockIdx.y * AS_T;
    const int tid = threadIdx.y * AS_T + threadIdx.x;
    const size_t plane = (size_t)ps;
    for (int i = tid; i < AS_C * AS_C; i += AS_T * AS_T)
    {
        const int ty = i / AS_C, tx = i - ty * AS_C;
        const int gx = min(max(x0 - 5 + tx, 0), dw - 1), gy = min(max(y0 - 5 + ty, 0), dh - 1);
        const size_t o = (size_t)gy * rs + gx;
        s.c[0][ty][tx] = src[o];
        s.c[1][ty][tx] = src[plane + o];
        s.c[2][ty][tx] = src[2 * plane + o];
    }
    __syncthreads();
    // pass 1 taps:  [ 9 ] / [ 1 2 3 ] / [ 10 4 0 5 11 ] / [ 6 7 8 ] / [ 12 ]
    const int px[13] = {0, -1, 0, 1, -1, 1, -1, 0, 1, 0, -2, 2, 0};
    const int py[13] = {0, -1, -1, -1, 0, 0, 1, 1, 1, -2, 0, 0, 2};
    for (int i = tid; i < AS_E * AS_E; i += AS_T * AS_T)
    {
        const int ty = i / AS_E, tx = i - ty * AS_E;
        const int cx = min(max(x0 - 3 + tx, 0), dw - 1), cy = min(max(y0 - 3 + ty, 0), dh - 1);
        float c[13][3];
#pragma unroll
        for (int k = 0; k < 13; k++)
        {
            const int sx = min(max(cx + px[k], 0), dw - 1) - (x0 - 5);
            const int sy = min(max(cy + py[k], 0), dh - 1) - (y0 - 5);
#pragma unroll
            for (int ch = 0; ch < 3; ch++)
                c[k][ch] = asSat(s.c[ch][sy][sx]);
        }
        float ev[3], bsum = 0.0f;
#pragma unroll
        for (int ch = 0; ch < 3; ch++)
        {
            const float bl = (2.0f * (c[2][ch] + c[4][ch] + c[5][ch] + c[7][ch]) +
                              (c[1][ch] + c[3][ch] + c[6][ch] + c[8][ch]) + 4.0f * c[0][ch]) /
                             16.0f;
            bsum += bl;
            ev[ch] = 1.38f * fabsf(bl - c[0][ch]) +
                     1.15f * (fabsf(bl - c[2][ch]) + fabsf(bl - c[4][ch]) + fabsf(bl - c[5][ch]) + fabsf(bl - c[7][ch])) +
                     0.92f * (fabsf(bl - c[1][ch]) + fabsf(bl - c[3][ch]) + fabsf(bl - c[6][ch]) + fabsf(bl - c[8][ch])) +
                     0.23f * (fabsf(bl - c[9][ch]) + fabsf(bl - c[10][ch]) + fabsf(bl - c[11][ch]) + fabsf(bl - c[12][ch]));
        }
        const float c_comp = asSat(4.0f / 15.0f + 0.9f * exp2f(bsum * (-37.0f / 15.0f)));
        s.e[ty][tx] = asSqrtA(ev[0] * ev[0] + ev[1] * ev[1] + ev[2] * ev[2]) * c_comp;
        // CtL: pass 2 reads the colour unclamped, then clamps c * |c|
        const int lx = cx - (x0 - 5), ly = cy - (y0 - 5);
        const float r = s.c[0][ly][lx], g = s.c[1][ly][lx], b = s.c[2][ly][lx];
        s.l[ty][tx] = asSqrtA(0.2558f * asSat(r * fabsf(r)) + 0.6511f * asSat(g * fabsf(g)) + 0.0931f * asSat(b * fabsf(b)));
    }
    __syncthreads();
}

// pass 2 for the block's pixel (tx, ty)
__device__ void asPixel(const AsTile& s, int tx, int ty, float curve, float out[3])
{
    // [ 22 ] / [ 24 9 23 ] / [ 21 1 2 3 18 ] / [ 19 10 4 0 5 11 16 ] / [ 20 6 7 8 17 ] / [ 15 12 14 ] / [ 13 ]
    const int px[25] = {0, -1, 0, 1, -1, 1, -1, 0, 1, 0, -2, 2, 0, 0, 1, -1, 3, 2, 2, -3, -2, -2, 0, 1, -1};
    const int py[25] = {0, -1, -1, -1, 0, 0, 1, 1, 1, -2, 0, 0, 2, 3, 2, 2, 0, 1, -1, 0, 1, -1, -3, -2, -2};
    float e[25], L[25];
#pragma unroll
    for (int k = 0; k < 25; k++)
    {
        e[k] = s.e[ty + 3 + py[k]][tx + 3 + px[k]];
        L[k] = s.l[ty + 3 + py[k]][tx + 3 + px[k]];
    }
    float c0[3];
#pragma unroll
    for (int ch = 0; ch < 3; ch++)
        c0[ch] = asSat(s.c[ch][ty + 5][tx + 5]);
    const float c0_Y = L[0]; // CtL of the clamped centre = CtL of the raw one
    const float c_edge = e[0];
    float maxedge = e[0];
#pragma unroll
    for (int k = 1; k < 13; k++)
        maxedge = fmaxf(maxedge, e[k]);
    const float den = fabsf(maxedge) + 0.03f;
#define AS_SOFT_IF(a, b, c) asSat(__fdividef((a) + (b) + (c) + 0.056f, den) - 0.85f)
    const float sbe = AS_SOFT_IF(e[2], e[9], e[22]) * AS_SOFT_IF(e[7], e[12], e[13]) +
                      AS_SOFT_IF(e[4], e[10], e[19]) * AS_SOFT_IF(e[5], e[11], e[16]) +
                      AS_SOFT_IF(e[1], e[24], e[21]) * AS_SOFT_IF(e[8], e[14], e[17]) +
                      AS_SOFT_IF(e[3], e[23], e[18]) * AS_SOFT_IF(e[6], e[20], e[15]);
#undef AS_SOFT_IF
    const float kcs = asSmooth(2.0f, 3.1f, sbe);
    const float cs_x = 0.167f + (0.334f - 0.167f) * kcs, cs_y = 0.250f + (0.500f - 0.250f) * kcs;
    const float sw = asSmooth(0.3f, 0.8f, c_edge);
    float dW0 = 0.5f + (0.86602540378f - 0.5f) * sw;
    float dW2 = 1.41421356237f + (0.54772255751f - 1.41421356237f) * sw;
    dW0 *= dW0;
    dW2 *= dW2;
    const float dW1 = 1.0f;
    const float mdiff_c0 =
        0.02f + 3.0f * (fabsf(L[0] - L[2]) + fabsf(L[0] - L[4]) + fabsf(L[0] - L[5]) + fabsf(L[0] - L[7]) +
                        0.25f * (fabsf(L[0] - L[1]) + fabsf(L[0] - L[3]) + fabsf(L[0] - L[6]) + fabsf(L[0] - L[8])));
#define AS_MDIFF(a, b, c, d, e_, f, g)                                                                             \
    (fabsf(L[g] - L[a]) + fabsf(L[g] - L[b]) + fabsf(L[g] - L[c]) + fabsf(L[g] - L[d]) +                           \
     0.5f * (fabsf(L[g] - L[e_]) + fabsf(L[g] - L[f])))
    float w[12] = {fminf(__fdividef(mdiff_c0, AS_MDIFF(24, 21, 2, 4, 9, 10, 1)), dW1),  dW0,
                   fminf(__fdividef(mdiff_c0, AS_MDIFF(23, 18, 5, 2, 9, 11, 3)), dW1),  dW0,
                   dW0,
                   fminf(__fdividef(mdiff_c0, AS_MDIFF(4, 20, 15, 7, 10, 12, 6)), dW1),  dW0,
                   fminf(__fdividef(mdiff_c0, AS_MDIFF(5, 7, 17, 14, 12, 11, 8)), dW1),
                   fminf(__fdividef(mdiff_c0, AS_MDIFF(2, 24, 23, 22, 1, 3, 9)), dW2),
                   fminf(__fdividef(mdiff_c0, AS_MDIFF(20, 19, 21, 4, 1, 6, 10)), dW2),
                   fminf(__fdividef(mdiff_c0, AS_MDIFF(17, 5, 18, 16, 3, 8, 11)), dW2),
                   fminf(__fdividef(mdiff_c0, AS_MDIFF(13, 15, 7, 14, 6, 8, 12)), dW2)};
#undef AS_MDIFF
    w[0] = (fmaxf(fmaxf((w[8] + w[9]) / 4.0f, w[0]), 0.25f) + w[0]) / 2.0f;
    w[2] = (fmaxf(fmaxf((w[8] + w[10]) / 4.0f, w[2]), 0.25f) + w[2]) / 2.0f;
    w[5] = (fmaxf(fmaxf((w[9] + w[11]) / 4.0f, w[5]), 0.25f) + w[5]) / 2.0f;
    w[7] = (fmaxf(fmaxf((w[10] + w[11]) / 4.0f, w[7]), 0.25f) + w[7]) / 2.0f;
    float lowthrsum = 0.0f, weightsum = 0.0f, neg_laplace = 0.0f;
#pragma unroll
    for (int p = 0; p < 12; p++)
    {
        const float t = asSat(__fdividef(e[p + 1] - 0.01f, 0.09f));
        const float lowthr = t * t * (2.97f - 1.98f * t) + 0.01f;
        neg_laplace += __powf(fabsf(L[p + 1]) + 0.06f, 2.4f) * (fabsf(w[p]) * lowthr);
        weightsum += fabsf(w[p]) * lowthr;
        lowthrsum += __fdividef(lowthr, 12.0f);
    }
    neg_laplace = asSat(__powf(__fdividef(neg_laplace, weightsum), 1.0f / 2.4f) - 0.06f);
    const float sharpen_val = __fdividef(curve, curve * 0.5f * __powf(fabsf(c_edge), 3.5f) + 0.625f);
    float sharpdiff = (c0_Y - neg_laplace) * (lowthrsum * sharpen_val + 0.01f);
    // the local near min and max: the shader's partial sort, 3 iterations
#pragma unroll
    for (int it = 0; it < 3; it++)
    {
#pragma unroll
        for (int i = it; i < 24 - it; i += 2)
        {
            const float a = L[i], b = L[i + 1];
            L[i] = fminf(a, b);
            L[i + 1] = fmaxf(a, b);
        }
#pragma unroll
        for (int ii = 24 - it; ii > it; ii -= 2)
        {
            float a = L[it], b = L[ii];
            L[it] = fminf(a, b);
            L[ii] = fmaxf(a, b);
            a = L[24 - it];
            b = L[ii - 1];
            L[24 - it] = fmaxf(a, b);
            L[ii - 1] = fminf(a, b);
        }
    }
    const float nmax = (fmaxf(L[22] + L[23] * 2.0f, c0_Y * 3.0f) + L[24]) / 4.0f;
    const float nmin = (fminf(L[2] + L[1] * 2.0f, c0_Y * 3.0f) + L[0]) / 4.0f;
    const float min_dist = fminf(fabsf(nmax - c0_Y), fabsf(c0_Y - nmin));
    float pn_x = fminf(0.003f + min_dist, 1.0001f - c0_Y), pn_y = fminf(0.009f + min_dist, 0.0001f + c0_Y);
    pn_x = fminf(pn_x, 0.1f * (1.0f - 0.056f) + pn_x * 0.056f);
    pn_y = fminf(pn_y, 0.1f * (1.0f - 0.056f) + pn_y * 0.056f);
    const float pos = fmaxf(sharpdiff, 0.0f), neg = fminf(sharpdiff, 0.0f);
    sharpdiff = asWpmean(pos, asSoftLim(pos, pn_x), cs_x) - asWpmean(neg, asSoftLim(neg, pn_y), cs_y);
    const float sharpdiff_lim = asSat(c0_Y + sharpdiff) - c0_Y;
    const float satmul = __fdividef(c0_Y + fmaxf(sharpdiff_lim * 0.9f, sharpdiff_lim) * 1.03f + 0.03f, c0_Y + 0.03f);
    const float base = c0_Y + (sharpdiff_lim * 3.0f + sharpdiff) / 4.0f;
#pragma unroll
    for (int ch = 0; ch < 3; ch++)
        out[ch] = asSat(base + (c0[ch] - c0_Y) * satmul);
}

// Adaptive Sharpen on a planar frame (planes ps apart, rows rs apart) into a tight one (dw x dh): the sharpen after
// DLSS 5 at the working size, before RTX HDR and the model (post-processing before frame generation); 16 x 16 blocks
__global__ void k_sharpPlanar(const float* __restrict__ src, int ps, int rs, int dw, int dh, float curve,
                              float* __restrict__ dst)
{
    __shared__ AsTile s;
    asLoad(s, src, ps, rs, dw, dh);
    const int ox = blockIdx.x * AS_T + threadIdx.x, oy = blockIdx.y * AS_T + threadIdx.y;
    if (ox >= dw || oy >= dh) return;
    float c[3];
    asPixel(s, threadIdx.x, threadIdx.y, curve, c);
    const size_t plane = (size_t)dw * dh, o = (size_t)oy * dw + ox;
    dst[o] = c[0];
    dst[plane + o] = c[1];
    dst[2 * plane + o] = c[2];
}

// ---- live RTX TrueHDR inside the native host ----------------------------------------------
// Verbatim math of live_server.py _cap_to_pq2020 (~88) and rtxvideo.py _eval_thdr (~248),
// run_hdr_pq (~280), _ictcp_linear (~330), _rtx_linear (~355), _vibrance_gain and
// _luma_weight. TrueHDR runs ONCE PER REAL FRAME at capture resolution; the PQ result is
// resized on PQ afterwards, exactly as python does.

// ST 2084 EOTF (_pq_to_linear): PQ code' in [0,1] -> display-linear in [0,1].
__device__ __forceinline__ float pq_eotf(float e)
{
    const float M1 = 0.1593017578125f, M2 = 78.84375f;
    const float C1 = 0.8359375f, C2 = 18.8515625f, C3 = 18.6875f;
    float ec = e < 0.0f ? 0.0f : (e > 1.0f ? 1.0f : e);
    const float ep = powf(ec, 1.0f / M2);
    float num = ep - C1;
    if (num < 0.0f) num = 0.0f;
    float den = C2 - C3 * ep;
    if (den < 1e-6f) den = 1e-6f;
    return powf(num / den, 1.0f / M1);
}

// ST 2084 inverse EOTF (_linear_to_pq): display-linear in [0,1] -> PQ code' in [0,1].
__device__ __forceinline__ float pq_oetf(float lin)
{
    const float M1 = 0.1593017578125f, M2 = 78.84375f;
    const float C1 = 0.8359375f, C2 = 18.8515625f, C3 = 18.6875f;
    const float l = lin < 0.0f ? 0.0f : lin;
    const float lm = powf(l, M1);
    return powf((C1 + C2 * lm) / (1.0f + C3 * lm), M2);
}

// ---- offline DLSS 4.5 in HDR10: the DLSS-G guide's HDR input (section 11.0), R10G10B10A2 words of PQ BT.2020 (the
// dlssg2f --hdr10 frames). An HLG picture goes as PQ on BT.2100's reference display (1000 nits peak, system gamma 1.2:
// the OOTF on the scene luminance, as hlg2020_to_scrgb) and comes back through the exact inverse (the inverse OOTF on
// the display luminance, as scrgb_to_hlg2020), both per channel in BT.2020.
__device__ __forceinline__ void hlg2020_to_pq2020(float e0, float e1, float e2, float& p0, float& p1, float& p2)
{
    const float A = 0.17883277f, B = 0.28466892f, C = 0.55991073f;
    const float e[3] = { e0, e1, e2 };
    float s[3];
    for (int c = 0; c < 3; c++)
    {
        const float v = e[c] < 0.0f ? 0.0f : (e[c] > 1.0f ? 1.0f : e[c]);
        s[c] = v <= 0.5f ? v * v / 3.0f : (expf((v - C) / A) + B) / 12.0f;
    }
    const float ys = 0.2627f * s[0] + 0.6780f * s[1] + 0.0593f * s[2];
    const float k = ys > 0.0f ? 0.1f * powf(ys, 0.2f) : 0.0f; // 1000 nits = 0.1 of PQ's 10000
    p0 = pq_oetf(s[0] * k);
    p1 = pq_oetf(s[1] * k);
    p2 = pq_oetf(s[2] * k);
}
__device__ __forceinline__ void pq2020_to_hlg2020(float p0, float p1, float p2, float& e0, float& e1, float& e2)
{
    const float A = 0.17883277f, B = 0.28466892f, C = 0.55991073f;
    const float l[3] = { pq_eotf(p0) * 10.0f, pq_eotf(p1) * 10.0f, pq_eotf(p2) * 10.0f }; // units of 1000 nits
    const float yd = 0.2627f * l[0] + 0.6780f * l[1] + 0.0593f * l[2];
    const float k = yd > 0.0f ? powf(yd, -0.2f / 1.2f) : 0.0f;
    float o[3];
    for (int c = 0; c < 3; c++)
    {
        const float s = l[c] * k > 1.0f ? 1.0f : l[c] * k;
        o[c] = s <= 1.0f / 12.0f ? sqrtf(3.0f * s) : A * logf(12.0f * s - B) + C;
    }
    e0 = o[0];
    e1 = o[1];
    e2 = o[2];
}

// the model frame ((R, G, B) planes, fp32 or half) to tight R10G10B10A2 words at the padded size: round(x * 1023) per
// field, alpha 3 (an HLG picture as PQ first)
__global__ void k_packR10(const void* __restrict__ src, int half, int planeStride, int rowStride, int w, int h,
                          int hlg, unsigned int* __restrict__ dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t o = (size_t)y * rowStride + x;
    float v[3];
    for (int ci = 0; ci < 3; ci++)
        v[ci] = ldS(src, half, (size_t)ci * planeStride + o);
    if (hlg) hlg2020_to_pq2020(v[0], v[1], v[2], v[0], v[1], v[2]);
    unsigned int word = 3u << 30;
    for (int ci = 0; ci < 3; ci++)
    {
        const float c = v[ci] < 0.0f ? 0.0f : (v[ci] > 1.0f ? 1.0f : v[ci]);
        word |= (unsigned int)rintf(c * 1023.0f) << (10 * ci);
    }
    dst[(size_t)y * w + x] = word;
}

// a DLSS 4.5 host frame in HDR10 back to the offline (R, G, B) planes: each 10-bit code / 1023 (an HLG picture back
// from PQ)
__global__ void k_unpackR10(const unsigned int* __restrict__ src, int dw, int dh, int hlg, float* __restrict__ dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    const size_t plane = (size_t)dw * dh;
    const size_t o = (size_t)y * dw + x;
    const unsigned int word = src[o];
    float v[3];
    for (int ci = 0; ci < 3; ci++)
        v[ci] = (float)((word >> (10 * ci)) & 1023u) / 1023.0f;
    if (hlg) pq2020_to_hlg2020(v[0], v[1], v[2], v[0], v[1], v[2]);
    dst[o] = v[0];
    dst[plane + o] = v[1];
    dst[2 * plane + o] = v[2];
}

// the two numerically inverted matrices (python inverts in float64 then casts to fp32); the
// host computes them in double at init and uploads them here.
__device__ float g_ictcp2lms[9];
__device__ float g_lms2rgb[9];

__device__ __forceinline__ void mat3(const float* m, float a, float b, float c,
                                     float& o0, float& o1, float& o2)
{
    o0 = m[0] * a + m[1] * b + m[2] * c;
    o1 = m[3] * a + m[4] * b + m[5] * c;
    o2 = m[6] * a + m[7] * b + m[8] * c;
}

// linear BT.2020 -> ICtCp (BT.2100): Mlms2ictcp * PQ_OETF(Mrgb2lms * max(x, 0))
__device__ __forceinline__ void to_ictcp(float r, float g, float b,
                                         float& i, float& ct, float& cp)
{
    const float R = r < 0.0f ? 0.0f : r, G = g < 0.0f ? 0.0f : g, B = b < 0.0f ? 0.0f : b;
    const float l = (1688.0f * R + 2146.0f * G + 262.0f * B) / 4096.0f;
    const float m = (683.0f * R + 2951.0f * G + 462.0f * B) / 4096.0f;
    const float s = (99.0f * R + 309.0f * G + 3688.0f * B) / 4096.0f;
    const float lp = pq_oetf(l), mp = pq_oetf(m), sp = pq_oetf(s);
    i  = (2048.0f * lp + 2048.0f * mp + 0.0f * sp) / 4096.0f;
    ct = (6610.0f * lp - 13613.0f * mp + 7003.0f * sp) / 4096.0f;
    cp = (17933.0f * lp - 17390.0f * mp - 543.0f * sp) / 4096.0f;
}

// _luma_weight: smoothstep 0 to 1 over PQ 0.10..0.25, then 1 to 0.25 over PQ 0.50..0.75
__device__ __forceinline__ float luma_weight(float i)
{
    float t0 = (i - 0.10f) / 0.15f;
    t0 = t0 < 0.0f ? 0.0f : (t0 > 1.0f ? 1.0f : t0);
    const float s0 = t0 * t0 * (3.0f - 2.0f * t0);
    float t1 = (i - 0.50f) / 0.25f;
    t1 = t1 < 0.0f ? 0.0f : (t1 > 1.0f ? 1.0f : t1);
    const float s1 = t1 * t1 * (3.0f - 2.0f * t1);
    return s0 * (1.0f - 0.75f * s1);
}

// _vibrance_gain: exact no-op when both controls are 0
__device__ __forceinline__ void vibrance_gain(float i, float ct, float cp, float vib, float sb,
                                              float& oct, float& ocp)
{
    if (vib <= 0.0f && sb <= 0.0f) { oct = ct; ocp = cp; return; }
    const float c = hypotf(ct, cp);
    float t = c / 0.12f;
    t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    float g = (1.0f + sb) * (1.0f + vib * (1.0f - t));
    g = 1.0f + luma_weight(i) * (g - 1.0f);
    oct = ct * g;
    ocp = cp * g;
}

// steps 1 to 3 of _cap_to_pq2020: scRGB half4 to the SDR-encoded g as unquantised planar fp32 (the chain's SDR
// picture); the TrueHDR bridge input is packed later from the finished frame (k_thdrIn / k_sharpThdrIn).
__global__ void k_sdrEncode(const unsigned short* __restrict__ src, int cw, int ch, float sdrScale,
                            float* __restrict__ dstG)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cw || y >= ch) return;
    const unsigned short* p = src + ((size_t)y * cw + x) * 4;
    float g[3];
    for (int c = 0; c < 3; c++)
    {
        float lin = h2f(p[c]) / sdrScale;
        lin = lin < 0.0f ? 0.0f : (lin > 1.0f ? 1.0f : lin);
        float v = lin <= 0.0031308f
                      ? lin * 12.92f
                      : 1.055f * powf(lin < 1e-6f ? 1e-6f : lin, 1.0f / 2.4f) - 0.055f;
        v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        g[c] = v;
    }
    const size_t plane = (size_t)ch * cw;
    const size_t o = (size_t)y * cw + x;
    dstG[o] = g[0];
    dstG[plane + o] = g[1];
    dstG[2 * plane + o] = g[2];
}

// the vivid / rtx colour step of one pixel (_source_at_thdr_luma + _ictcp_linear / _rtx_linear +
// _pack_out's nan_to_num): the bridge word u (10:10:10:2, R in the LOW 10 bits) and the
// unquantised SDR source s0..s2 (gamma BT.709, R, G, B) -> the corrected linear BT.2020 in
// [0,1]. mode 0 = vivid, 1 = rtx. k_thdrColor's colour step (both routes).
__device__ __forceinline__ void thdr_linear(unsigned int u, float s0, float s1, float s2,
                                            int mode, float vib, float sb, float* v)
{
    const float tr = (float)(u & 1023u) * (1.0f / 1023.0f);
    const float tg = (float)((u >> 10) & 1023u) * (1.0f / 1023.0f);
    const float tb = (float)((u >> 20) & 1023u) * (1.0f / 1023.0f);
    {
        const float lt0 = pq_eotf(tr), lt1 = pq_eotf(tg), lt2 = pq_eotf(tb);
        const float y_t = 0.2627f * lt0 + 0.6780f * lt1 + 0.0593f * lt2;
        float s[3] = { s0, s1, s2 };
        float l709[3];
        for (int c = 0; c < 3; c++)
        {
            const float sv = s[c] < 0.0f ? 0.0f : (s[c] > 1.0f ? 1.0f : s[c]);
            if (sv < 0.081f) l709[c] = sv / 4.5f;
            else
            {
                float t = (sv + 0.099f) / 1.099f;
                if (t < 0.0f) t = 0.0f;
                l709[c] = powf(t, 1.0f / 0.45f);
            }
        }
        const float li0 = 0.6274f * l709[0] + 0.3293f * l709[1] + 0.0433f * l709[2];
        const float li1 = 0.0691f * l709[0] + 0.9195f * l709[1] + 0.0114f * l709[2];
        const float li2 = 0.0164f * l709[0] + 0.0880f * l709[1] + 0.8956f * l709[2];
        float y_s = 0.2627f * li0 + 0.6780f * li1 + 0.0593f * li2;
        if (y_s < 1e-6f) y_s = 1e-6f;
        const float k = y_t / y_s;
        float fi, fct, fcp;
        to_ictcp(li0 * k, li1 * k, li2 * k, fi, fct, fcp);
        if (mode == 1)
        {
            float ti, tct, tcp;
            to_ictcp(lt0, lt1, lt2, ti, tct, tcp);
            float mag_f = hypotf(fct, fcp);
            if (mag_f < 1e-6f) mag_f = 1e-6f;
            float mag = hypotf(tct, tcp);
            if (mag < mag_f) mag = mag_f;
            const float dd = mag / mag_f;
            fct *= dd;
            fcp *= dd;
        }
        float ct, cp;
        vibrance_gain(fi, fct, fcp, vib, sb, ct, cp);
        float l0, l1, l2;
        mat3(g_ictcp2lms, fi, ct, cp, l0, l1, l2);
        l0 = pq_eotf(l0); l1 = pq_eotf(l1); l2 = pq_eotf(l2);
        float r0, r1, r2;
        mat3(g_lms2rgb, l0, l1, l2, r0, r1, r2);
        v[0] = r0; v[1] = r1; v[2] = r2;
        for (int c = 0; c < 3; c++)
        {
            float t = v[c];
            t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
            if (!(t == t)) t = 0.0f;             // nan_to_num_(0.0)
            v[c] = t;
        }
    }
}

// Dynamic Vibrance on TrueHDR's own result (PQ R, G, B in place): the gain of the vivid / rtx step, in the result's
// ICtCp
__device__ __forceinline__ void thdr_vibrance(float& p0, float& p1, float& p2, float vib, float sb)
{
    float i, ct, cp, oct, ocp;
    to_ictcp(pq_eotf(p0), pq_eotf(p1), pq_eotf(p2), i, ct, cp);
    vibrance_gain(i, ct, cp, vib, sb, oct, ocp);
    float l0, l1, l2;
    mat3(g_ictcp2lms, i, oct, ocp, l0, l1, l2);
    l0 = pq_eotf(l0); l1 = pq_eotf(l1); l2 = pq_eotf(l2);
    float v[3];
    mat3(g_lms2rgb, l0, l1, l2, v[0], v[1], v[2]);
    for (int c = 0; c < 3; c++)
    {
        float t = v[c];
        t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
        if (!(t == t)) t = 0.0f;
        v[c] = t;
    }
    p0 = pq_oetf(v[0]); p1 = pq_oetf(v[1]); p2 = pq_oetf(v[2]);
}

// Unpack the bridge output (10:10:10:2, R in the LOW 10 bits), apply the
// colour mode against the unquantised source, and write planar PQ into the padded model frame
// (pw x ph, planes dps apart) with the edge replicated as k_padPlanar pads it: a pad pixel takes
// the result of the edge pixel it repeats.
// mode 0 = vivid (_ictcp_linear), 1 = rtx (_rtx_linear), 2 = raw (TrueHDR's result as is, Dynamic Vibrance when on).
__global__ void k_thdrColor(const unsigned int* __restrict__ thdrOut,
                            const float* __restrict__ srcG, int cw, int ch,
                            float* __restrict__ dst, int pw, int ph, int dps,
                            int mode, float vib, float sb)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= pw || y >= ph) return;
    const size_t plane = (size_t)ch * cw;
    const int sx = x < cw ? x : cw - 1, sy = y < ch ? y : ch - 1;
    const size_t s = (size_t)sy * cw + sx, o = (size_t)y * pw + x;
    const unsigned int u = thdrOut[s];
    float p0, p1, p2;
    if (mode == 2)
    {
        p0 = (float)(u & 1023u) * (1.0f / 1023.0f);
        p1 = (float)((u >> 10) & 1023u) * (1.0f / 1023.0f);
        p2 = (float)((u >> 20) & 1023u) * (1.0f / 1023.0f);
        if (vib > 0.0f || sb > 0.0f) thdr_vibrance(p0, p1, p2, vib, sb);
    }
    else
    {
        float v[3];
        thdr_linear(u, srcG[s], srcG[plane + s], srcG[2 * plane + s], mode, vib, sb, v);
        p0 = pq_oetf(v[0]); p1 = pq_oetf(v[1]); p2 = pq_oetf(v[2]);
    }
    dst[o] = p0;
    dst[(size_t)dps + o] = p1;
    dst[2 * (size_t)dps + o] = p2;
}

// ---- offline RTX TrueHDR (rtxvideo.run_hdr on every output frame) -------------------------
// The output-size SDR frame after the pass chain -> the bridge's RGBA8 input (_eval_thdr: clamp,
// * 255, round) and the unquantised planar source the colour step reads (R, G, B, tight).
__device__ __forceinline__ void thdr_in_store(float r, float g, float b, size_t o, size_t plane,
                                              unsigned char* __restrict__ rgba, float* __restrict__ dstG)
{
    const float c[3] = { r, g, b };
    unsigned char q[3];
    for (int ci = 0; ci < 3; ci++)
    {
        const float v = c[ci] < 0.0f ? 0.0f : (c[ci] > 1.0f ? 1.0f : c[ci]);
        q[ci] = (unsigned char)(int)rintf(v * 255.0f);
    }
    unsigned char* d = rgba + o * 4;
    d[0] = q[0]; d[1] = q[1]; d[2] = q[2]; d[3] = 255;
    dstG[o] = r;
    dstG[plane + o] = g;
    dstG[2 * plane + o] = b;
}

// the frame straight from the model or the staging buffer (planar with its own strides)
__global__ void k_thdrIn(const void* __restrict__ src, int half, int planeStride, int rowStride, int dw, int dh,
                         unsigned char* __restrict__ rgba, float* __restrict__ dstG)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    const size_t i = (size_t)y * rowStride + x;
    thdr_in_store(ldS(src, half, i), ldS(src, half, (size_t)planeStride + i),
                  ldS(src, half, 2 * (size_t)planeStride + i),
                  (size_t)y * dw + x, (size_t)dw * dh, rgba, dstG);
}

// Adaptive Sharpen on the staging frame first (the last SDR pass before TrueHDR; src planes ps apart, rows rs
// apart); 16 x 16 blocks
__global__ void k_sharpThdrIn(const float* __restrict__ src, int ps, int rs, int dw, int dh, float curve,
                              unsigned char* __restrict__ rgba, float* __restrict__ dstG)
{
    __shared__ AsTile s;
    asLoad(s, src, ps, rs, dw, dh);
    const int ox = blockIdx.x * AS_T + threadIdx.x, oy = blockIdx.y * AS_T + threadIdx.y;
    if (ox >= dw || oy >= dh) return;
    float c[3];
    asPixel(s, threadIdx.x, threadIdx.y, curve, c);
    thdr_in_store(c[0], c[1], c[2], (size_t)oy * dw + ox, (size_t)dw * dh, rgba, dstG);
}

// the 10-bit PQ code -> display-linear table of _accum_hp (_pq_to_linear(arange(1024) / 1023))
__global__ void k_pqLut(float* __restrict__ dst)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 1024) dst[i] = pq_eotf((float)i / 1023.0f);
}

// the offline RTX HDR emit: the PQ planes (R, G, B: TrueHDR's result carried through the model and the
// final resize; fp16 when half) -> the x2rgb10le words and the light statistics of the frame. Per frame:
// hist[1024] = the maxRGB 10-bit code histogram (DV L1, HDR10+ percentiles and average), misc[0..2] = the
// per-channel max code (HDR10+ MaxScl), misc[3] = the float bits of the brightest maxRGB value (never
// negative, so the bits order like the floats) and vSum its sum (int64, 2^24 fixed point): the linear
// maxRGB in vivid / rtx (MaxCLL / MaxFALL = * 10000), the maxRGB nits in raw (mode 2). The block stages
// the histogram in shared memory.
__global__ void k_pqOut(const void* __restrict__ src, int half, int planeStride, int rowStride, int dw, int dh,
                        int mode, unsigned int* __restrict__ dst, unsigned int* __restrict__ hist,
                        unsigned int* __restrict__ misc, unsigned long long* __restrict__ vSum)
{
    __shared__ unsigned int sh[1024];
    __shared__ unsigned int shMax[4];
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int nt = blockDim.x * blockDim.y;
    for (int i = tid; i < 1024; i += nt) sh[i] = 0;
    if (tid < 4) shMax[tid] = 0;
    __syncthreads();
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    long long vv = 0;   // vSum in 2^24 fixed point (no fp64; integer adds are order-free, so the
                        // sum is deterministic): nits <= 10000 in raw mode, 8K frame < 2^63
    if (x < dw && y < dh)
    {
        const size_t i = (size_t)y * rowStride + x;
        const float p[3] = { ldS(src, half, i), ldS(src, half, (size_t)planeStride + i),
                             ldS(src, half, 2 * (size_t)planeStride + i) };
        unsigned int q[3];
        float v[3];
        for (int c = 0; c < 3; c++)
        {
            const float pc = p[c] < 0.0f ? 0.0f : (p[c] > 1.0f ? 1.0f : p[c]);
            q[c] = (unsigned int)rintf(pc * 1023.0f);
            v[c] = pq_eotf(pc);
        }
        dst[(size_t)y * dw + x] = q[2] | (q[1] << 10) | (q[0] << 20) | 0xC0000000u;
        const unsigned int m = q[0] > q[1] ? (q[0] > q[2] ? q[0] : q[2]) : (q[1] > q[2] ? q[1] : q[2]);
        float val = mode == 2 ? pq_eotf((float)m / 1023.0f) * 10000.0f : fmaxf(fmaxf(v[0], v[1]), v[2]);
        val = val + 0.0f;   // a -0.0 would carry the sign bit into the bit-pattern max
        atomicAdd(&sh[m], 1u);
        atomicMax(&shMax[0], q[0]);
        atomicMax(&shMax[1], q[1]);
        atomicMax(&shMax[2], q[2]);
        atomicMax(&shMax[3], __float_as_uint(val));
        vv = llrintf(val * 16777216.0f);
    }
    for (int off = 16; off > 0; off >>= 1) vv += __shfl_down_sync(0xffffffffu, vv, off);
    if ((tid & 31) == 0 && vv != 0) atomicAdd(vSum, (unsigned long long)vv);
    __syncthreads();
    for (int i = tid; i < 1024; i += nt)
        if (sh[i]) atomicAdd(&hist[i], sh[i]);
    if (tid < 4 && shMax[tid]) atomicMax(&misc[tid], shMax[tid]);
}

// Offline render: raw interleaved RGB frames (rgb48le or rgb24, the decoder pipe format
// render.py uses) <-> the model's planar fp32 [0,1] with replicate pad, channel order R,G,B exactly
// as render.py's to_tensor / to_bytes (the live route packs BGRA8 and is a different contract).
__global__ void k_packInRaw16(const unsigned short* __restrict__ src, int w, int h,
                              float* __restrict__ dst, int ph, int pw, int planeStride)
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= pw || oy >= ph) return;
    const int sx = ox < w ? ox : w - 1;
    const int sy = oy < h ? oy : h - 1;
    const unsigned short* p = src + ((size_t)sy * w + sx) * 3;
    const int o = oy * pw + ox;
    dst[o] = p[0] * (1.0f / 65535.0f);
    dst[planeStride + o] = p[1] * (1.0f / 65535.0f);
    dst[2 * planeStride + o] = p[2] * (1.0f / 65535.0f);
}
__global__ void k_packInRaw8(const unsigned char* __restrict__ src, int w, int h,
                             float* __restrict__ dst, int ph, int pw, int planeStride)
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= pw || oy >= ph) return;
    const int sx = ox < w ? ox : w - 1;
    const int sy = oy < h ? oy : h - 1;
    const unsigned char* p = src + ((size_t)sy * w + sx) * 3;
    const int o = oy * pw + ox;
    dst[o] = p[0] * (1.0f / 255.0f);
    dst[planeStride + o] = p[1] * (1.0f / 255.0f);
    dst[2 * planeStride + o] = p[2] * (1.0f / 255.0f);
}
// crop the pad, clamp, round to the pipe depth (to_bytes: clamp(0,1) * maxv, round, uint)
__global__ void k_packOutRaw16(const void* __restrict__ src, int half, int planeStride, int rowStride,
                               int w, int h, unsigned short* __restrict__ dst)
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= w || oy >= h) return;
    const size_t o = (size_t)oy * rowStride + ox;
    unsigned short* p = dst + ((size_t)oy * w + ox) * 3;
    for (int ci = 0; ci < 3; ci++)
    {
        float v = ldS(src, half, (size_t)ci * planeStride + o);
        v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        p[ci] = (unsigned short)(int)rintf(v * 65535.0f);
    }
}
__global__ void k_expand8to16(const unsigned char* __restrict__ src, int n, unsigned short* __restrict__ dst)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (unsigned short)(src[i] * 257);
}
// identical-pair test: one pass over the pair's two packed inputs, any differing element sets
// the flag. EXACT equality, no tolerance anywhere. The flag read is a monotonic 0 -> 1 hint
// that lets later blocks give up early; only the host's read after the sync is authoritative.
__global__ void k_pairDiff(const float* __restrict__ a, const float* __restrict__ b, int n,
                           int* __restrict__ flag)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || *flag) return;
    if (a[i] != b[i]) *flag = 1;
}
// DLSS 5 reuse test: two decoded frames as the decoder handed them over, byte for byte, 16 bytes
// per thread (cudaMalloc buffers are 256-byte aligned); any difference sets the flag, same
// early-out contract as k_pairDiff
__global__ void k_rawDiff(const unsigned char* __restrict__ a, const unsigned char* __restrict__ b,
                          long long n, int* __restrict__ flag)
{
    const long long i = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * 16;
    if (i >= n || *flag) return;
    if (i + 16 <= n)
    {
        const uint4 x = *(const uint4*)(a + i), y = *(const uint4*)(b + i);
        if (x.x != y.x || x.y != y.y || x.z != y.z || x.w != y.w) *flag = 1;
        return;
    }
    for (long long k = i; k < n; k++)
        if (a[k] != b[k]) { *flag = 1; return; }
}
__global__ void k_packOutRaw8(const void* __restrict__ src, int half, int planeStride, int rowStride,
                              int w, int h, unsigned char* __restrict__ dst)
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= w || oy >= h) return;
    const size_t o = (size_t)oy * rowStride + ox;
    unsigned char* p = dst + ((size_t)oy * w + ox) * 3;
    for (int ci = 0; ci < 3; ci++)
    {
        float v = ldS(src, half, (size_t)ci * planeStride + o);
        v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        p[ci] = (unsigned char)(int)rintf(v * 255.0f);
    }
}

// ---- native NVOF model glue -----------------------------------------------------------------
// The tween maths of the NVOF model: the NVIDIA Optical Flow Accelerator gives both flow fields
// of a pair in ONE Execute (grid 4, S10.5 vectors in INPUT pixels, uint8 cost per cell), these
// kernels turn them into tweens by a bidirectional softmax FORWARD splat (the GMFSS glue without
// the networks), gated against a float64 reference implementation.
// Every buffer is planar with N = 1; an image is 3
// fp32 planes addressed by (planeStride, rowStride) like the host's padded model planes, a flow
// or metric plane is contiguous (w, h).
//   k_nvofLuma      the model planes (0..1) -> GRAYSCALE8 NVOF input, BT.709 luma, x255,
//                   rounded half to even.
//   k_nvofUp        one direction: the interleaved short2 field (raw / 32 = input px, NO rescale)
//                   and its uint8 cost, bilinear from the cell grid to (w, h). Source index from
//                   the GRID, (x + 0.5) / grid - 0.5 (a cell covers grid x grid px), clamped at 0,
//                   the far tap held at the last cell, like k_flowUp; the taps exact in integers.
//   k_nvofMetric    Z = -a * cost - b * |F(x) + G(x + F(x))|, G = the other direction sampled
//                   bilinear at the landing point (border clamped): forward-backward consistency,
//                   large where the pixel is occluded in the other frame.
//   k_splatNvof     k_splatSoft with the flow scale and the weight split: landing x + s * F,
//                   e = expf(Z + bias) (bias = log of the time weight, 1 - t for frame 0, t for
//                   frame 1), the same four corner weights and int64 atomics (2^40, below).
//   k_splatNvofNorm acc -> the tween; where the plain coverage is below `hole` (nothing landed),
//                   the plain blend (1 - t) I0 + t I1, and the hole is counted.
// All fp32 (fp64 cost 2 to 3 ms per 1080p tween plus 1.4 ms per pair): a splat lands by land(), a sample by
// nvAxis (the pixel plus the floor and the fraction of its offset), never at an fp32 absolute
// coordinate (one ulp at x ~ 2000 is 1.2e-4 px); the k_nvofUp taps are integer ratios, exact at
// the power-of-two grids; k_nvofLuma rounds exactly like the fp64 form.
// fixed point of the splat accumulators: 2^40, not k_splatSoft's 2^26. e = exp(Z + bias) <= 1
// and Z is clamped at -16, so a contribution is at most 2^40 and at least ~1.2e5 units (2^26
// would leave ~7 units there, a 13% colour quantization exactly in the occluded regions); even a
// thousand contributions per pixel stay under 2^50 of the int64 range
#define NV_FIX 1099511627776.0f

// one axis of a bilinear tap at pixel x plus the offset u px, the point clamped into [0, n - 1]:
// the cell from floor(u) and the weight from its exact fraction (x + u in fp32 would carry one
// ulp of x); an offset beyond +-(n + 1) clamps to the edge anyway, so it is limited first and the
// int conversion cannot overflow (a NaN offset lands on the low edge)
__device__ __forceinline__ void nvAxis(int x, float u, int n, int& x0, int& x1, float& l)
{
    u = fminf(fmaxf(u, -(float)(n + 1)), (float)(n + 1));
    const float fl = floorf(u);
    x0 = x + (int)fl;
    l = u - fl;
    if (x0 < 0) { x0 = 0; l = 0.0f; }
    else if (x0 >= n - 1) { x0 = n - 1; l = 0.0f; }
    x1 = x0 < n - 1 ? x0 + 1 : x0;
}
struct NvTap { int x0, x1, y0, y1; float lx, ly; };
__device__ __forceinline__ NvTap nvTap(int x, int y, float ux, float uy, int w, int h)
{
    NvTap T;
    nvAxis(x, ux, w, T.x0, T.x1, T.lx);
    nvAxis(y, uy, h, T.y0, T.y1, T.ly);
    return T;
}
__device__ __forceinline__ float nvLerp(const float* __restrict__ p, int rowStride, const NvTap& T)
{
    const float a = (1.0f - T.lx) * p[(size_t)T.y0 * rowStride + T.x0] + T.lx * p[(size_t)T.y0 * rowStride + T.x1];
    const float b = (1.0f - T.lx) * p[(size_t)T.y1 * rowStride + T.x0] + T.lx * p[(size_t)T.y1 * rowStride + T.x1];
    return (1.0f - T.ly) * a + T.ly * b;
}

// ---- RIFE: the IFNet hands out its final flow and blend mask, the host makes the tween ---------
// The motion frame the IFNet and its encode read: a planar source (fp32 codes, sw x sh) padded to
// dpw x dph by replicating the edge (zp: zeros, RIFE's author's pad), fp16 out (k_f2h's rounding). mode 0 = the codes as they are
// (an SDR picture: bit for bit what k_f2h makes of it); 1 PQ / 2 HLG = the BT.2020 codes as BT.709
// light over SDR white in sRGB, the encoding RIFE was trained on: up to the knee exactly an SDR
// source's values, above it the largest channel takes PQ's spacing, cap of extended sRGB from the
// knee to 10000 nits (the sRGB of the knee + cap (PQ(m) - PQ(knee)) / (1 - PQ(knee))), every channel
// scaled alike so the hue stays.
__device__ __forceinline__ void motionEnc(float v[3], int mode, float sdrWhite, float knee, float cap)
{
    if (!mode)
        return;
    float l[3], m = 0.0f;
    hdr2020_to_scrgb(v[0], v[1], v[2], mode == 2, l[0], l[1], l[2]);
    for (int c = 0; c < 3; c++)
    {
        l[c] = l[c] > 0.0f ? l[c] / sdrWhite : 0.0f;
        m = fmaxf(m, l[c]);
    }
    float k = 1.0f;
    if (m > knee)
    {
        const float wn = sdrWhite * (80.0f / 10000.0f);
        const float p0 = pq_oetf(knee * wn);
        k = srgbEotf(srgbOetf(knee) + cap * (pq_oetf(fminf(m * wn, 1.0f)) - p0) / (1.0f - p0)) / m;
    }
    for (int c = 0; c < 3; c++)
        v[c] = srgbOetf(l[c] * k);
}
// one axis of PyTorch's bilinear resize (F.interpolate, align_corners False, no antialias) as DRBA's author stretches
// with it: output index d of m from n source samples = cell i0 / i1 and the weight l1 of i1
__device__ __forceinline__ void blAxis(int d, int n, int m, int& i0, int& i1, float& l1)
{
    float s = ((float)n / (float)m) * ((float)d + 0.5f) - 0.5f;
    s = s < 0.0f ? 0.0f : s;
    i0 = (int)s;
    i1 = i0 + (i0 < n - 1 ? 1 : 0);
    l1 = s - (float)i0;
}
__device__ __forceinline__ float blAt(const float* __restrict__ p, int rs, int x0, int x1, int y0, int y1, float lx,
                                      float ly)
{
    const float lx0 = 1.0f - lx, ly0 = 1.0f - ly;
    return ly0 * (lx0 * p[(size_t)y0 * rs + x0] + lx * p[(size_t)y0 * rs + x1]) +
           ly * (lx0 * p[(size_t)y1 * rs + x0] + lx * p[(size_t)y1 * rs + x1]);
}
// st (DRBA as its author, NativeRife::drAuthor): the source stretched bilinearly over the whole dpw x dph frame and
// handed over as B, G, R (the author's frames come from OpenCV)
__global__ void k_motionIn(const float* __restrict__ src, int sps, int srs, int sw, int sh,
                           unsigned short* __restrict__ dst, int dps, int drs, int dpw, int dph, int mode,
                           float sdrWhite, float knee, float cap, int zp, int st)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dpw || y >= dph) return;
    const size_t s = (size_t)(y < sh ? y : sh - 1) * srs + (x < sw ? x : sw - 1);
    const size_t d = (size_t)y * drs + x;
    if (zp && (x >= sw || y >= sh))
    {
        dst[d] = dst[(size_t)dps + d] = dst[2 * (size_t)dps + d] = 0;
        return;
    }
    float v[3] = { src[s], src[(size_t)sps + s], src[2 * (size_t)sps + s] };
    if (st)
    {
        int x0, x1, y0, y1;
        float lx, ly;
        blAxis(x, sw, dpw, x0, x1, lx);
        blAxis(y, sh, dph, y0, y1, ly);
        for (int c = 0; c < 3; c++)
            v[c] = blAt(src + (size_t)c * sps, srs, x0, x1, y0, y1, lx, ly);
    }
    motionEnc(v, mode, sdrWhite, knee, cap);
    if (st)
    {
        const float t = v[0];
        v[0] = v[2];
        v[2] = t;
    }
    dst[d] = f2h(v[0]);
    dst[(size_t)dps + d] = f2h(v[1]);
    dst[2 * (size_t)dps + d] = f2h(v[2]);
}
// three planes resized bilinearly as PyTorch's F.interpolate(mode='bilinear', align_corners=False) does (DRBA as its
// author: every frame to the network's frame and every output back): sw x sh fp32 -> dw x dh, fp16 out when dHalf
__global__ void k_bilin(const float* __restrict__ src, int sps, int srs, int sw, int sh, void* __restrict__ dst, int dps,
                        int drs, int dw, int dh, int dHalf)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    int x0, x1, y0, y1;
    float lx, ly;
    blAxis(x, sw, dw, x0, x1, lx);
    blAxis(y, sh, dh, y0, y1, ly);
    const size_t o = (size_t)y * drs + x;
    for (int c = 0; c < 3; c++)
    {
        const float v = blAt(src + (size_t)c * sps, srs, x0, x1, y0, y1, lx, ly);
        if (dHalf)
            ((unsigned short*)dst)[(size_t)c * dps + o] = f2h(v);
        else
            ((float*)dst)[(size_t)c * dps + o] = v;
    }
}

// ---- GMFSS and Restore on HDR planes: the SDR view at the model boundary ---------------------------
// Both synthesise pixels (GMFSS the tweens, Restore the frame) and were trained on SDR sRGB pictures: on PQ codes they
// fall short of their SDR quality, and any value outside 0..1 breaks their synthesis (highlights, wide gamut; both
// measured). hdrToModel = a pixel of HDR planes (PQ BT.2020, or HLG when mode 2) as its BT.709 light over SDR white in
// sRGB, what an SDR source hands a model: the largest channel's light above the knee rolled off toward knee + head
// (knee + head e / (e + head), every channel scaled alike; head 0 = scaled to the knee), light below 0 (a colour
// outside BT.709) clipped. Restore reads it with head 0 and gets back what it never held (k_hdrRestOut); GMFSS reads it
// on frame pairs inside the SDR range (k_hdrRange) and comes back through the sRGB EOTF (k_hdrFromSdr), otherwise only
// its motion nets read it (kMotionHead's roll-off) and the rest read the pictures. The real frames never pass the map.
__device__ __forceinline__ void hdrToModel(float v[3], int mode, float white, float knee, float head)
{
    float l[3], m = 0.0f;
    hdr2020_to_scrgb(v[0], v[1], v[2], mode == 2, l[0], l[1], l[2]);
    for (int c = 0; c < 3; c++)
    {
        l[c] /= white;
        m = fmaxf(m, l[c]);
    }
    const float e = m - knee;
    const float k = e > 0.0f ? (knee + head * e / (e + head)) / m : 1.0f;
    for (int c = 0; c < 3; c++)
        v[c] = srgbOetf(fmaxf(l[c] * k, 0.0f));
}
// a planar HDR frame (fp32, or fp16 when half; w x h, planes sps apart, rows srs apart) -> its SDR view, fp32 or
// fp16 (dstHalf, k_f2h's rounding), planes dps apart, rows drs apart
__global__ void k_hdrEnc(const void* __restrict__ src, int half, int sps, int srs, int w, int h, void* __restrict__ dst,
                         int dstHalf, int dps, int drs, int mode, float white, float knee, float head)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t s = (size_t)y * srs + x, d = (size_t)y * drs + x;
    float v[3] = { ldS(src, half, s), ldS(src, half, (size_t)sps + s), ldS(src, half, 2 * (size_t)sps + s) };
    hdrToModel(v, mode, white, knee, head);
    for (int c = 0; c < 3; c++)
    {
        const size_t o = (size_t)c * dps + d;
        if (dstHalf) ((unsigned short*)dst)[o] = f2h(v[c]);
        else ((float*)dst)[o] = v[c];
    }
}
// the SDR view's way back (a GMFSS tween of an SDR pair, fp32 or fp16 when half): clamp(0, 1) as the SDR route, the
// sRGB EOTF, times white, to the BT.2020 codes, fp32 planar
__global__ void k_hdrFromSdr(const void* __restrict__ src, int half, int sps, int srs, int w, int h,
                             float* __restrict__ dst, int dps, int drs, int mode, float white)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t s = (size_t)y * srs + x, d = (size_t)y * drs + x;
    float q[3];
    for (int c = 0; c < 3; c++)
    {
        const float v = ldS(src, half, (size_t)c * sps + s);
        q[c] = srgbEotf(v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v)) * white;
    }
    if (mode == 2) scrgb_to_hlg2020(q[0], q[1], q[2], q[0], q[1], q[2]);
    else scrgb_to_pq2020(q[0], q[1], q[2], q[0], q[1], q[2]);
    dst[d] = q[0];
    dst[(size_t)dps + d] = q[1];
    dst[2 * (size_t)dps + d] = q[2];
}
// whether a frame of HDR planes stays inside the SDR range: a pixel whose BT.709 light over SDR white is above 1 + tol
// or below -tol sets *flag, one atomic per block (GMFSS's SDR pairs)
__global__ void k_hdrRange(const float* __restrict__ src, int sps, int srs, int w, int h, int mode, float white,
                           float tol, int* __restrict__ flag)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    int out = 0;
    if (x < w && y < h)
    {
        const size_t s = (size_t)y * srs + x;
        float l[3];
        hdr2020_to_scrgb(src[s], src[(size_t)sps + s], src[2 * (size_t)sps + s], mode == 2, l[0], l[1], l[2]);
        out = fmaxf(l[0], fmaxf(l[1], l[2])) > (1.0f + tol) * white || fminf(l[0], fminf(l[1], l[2])) < -tol * white;
    }
    if (__syncthreads_or(out) && threadIdx.x == 0 && threadIdx.y == 0)
        atomicOr(flag, 1);
}
// FRUC reads 8-bit BGRA only: on a pair inside the SDR range (k_hdrRange) it gets the frames' SDR view (hdrToModel,
// knee 1, head 0), 256 codes for the SDR range where the 8-bit HDR codes spend ~148 on it; fp32 (R, G, B) planes, w x h,
// planes sps apart, rows srs apart, into true BGRA8 with k_packBgraRgb's rounding
__global__ void k_packBgraSdr(const float* __restrict__ src, int sps, int srs, int w, int h, int mode, float white,
                              unsigned char* __restrict__ dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t s = (size_t)y * srs + x;
    float v[3] = { src[s], src[(size_t)sps + s], src[2 * (size_t)sps + s] };
    hdrToModel(v, mode, white, 1.0f, 0.0f);
    unsigned char* p = dst + ((size_t)y * w + x) * 4;
    for (int ci = 0; ci < 3; ci++)
    {
        const float c = v[ci] < 0.0f ? 0.0f : (v[ci] > 1.0f ? 1.0f : v[ci]);
        p[2 - ci] = (unsigned char)(int)rintf(c * 255.0f);
    }
    p[3] = 255;
}
// such a pair's tween (true BGRA8 of the SDR view) back to the HDR codes, fp32 (R, G, B) planes dw x dh: k_unpackBgraRgb's
// / 255, then k_hdrFromSdr's way (the sRGB EOTF, times white, to the BT.2020 codes)
__global__ void k_unpackBgraSdr(const unsigned char* __restrict__ src, int dw, int dh, int mode, float white,
                                float* __restrict__ dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    const size_t plane = (size_t)dw * dh;
    const size_t o = (size_t)y * dw + x;
    const unsigned char* p = src + o * 4;
    float q[3] = { srgbEotf(p[2] / 255.0f) * white, srgbEotf(p[1] / 255.0f) * white, srgbEotf(p[0] / 255.0f) * white };
    if (mode == 2) scrgb_to_hlg2020(q[0], q[1], q[2], q[0], q[1], q[2]);
    else scrgb_to_pq2020(q[0], q[1], q[2], q[0], q[1], q[2]);
    dst[o] = q[0];
    dst[plane + o] = q[1];
    dst[2 * plane + o] = q[2];
}
// AMD FSR frame generation (the bridge's RGBA16F frames, rows dstPitch bytes apart): fp32 (R, G, B) planes, w x h,
// planes sps apart, rows srs apart, clamped to 0..1 as the 8-bit packers do, without their rounding; sdr = the SDR view
// of HDR planes (k_packBgraSdr's hdrToModel), else the planes as stored
__global__ void k_packRgbaH(const float* __restrict__ src, int sps, int srs, int w, int h, int sdr, int mode,
                            float white, unsigned short* __restrict__ dst, int dstPitch)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t s = (size_t)y * srs + x;
    float v[3] = { src[s], src[(size_t)sps + s], src[2 * (size_t)sps + s] };
    if (sdr) hdrToModel(v, mode, white, 1.0f, 0.0f);
    unsigned short* p = (unsigned short*)((unsigned char*)dst + (size_t)y * dstPitch) + 4 * (size_t)x;
    for (int c = 0; c < 3; c++)
        p[c] = f2h(v[c] < 0.0f ? 0.0f : (v[c] > 1.0f ? 1.0f : v[c]));
    p[3] = 0x3C00; // 1.0
}
// an RGBA16F frame as the Optical Flow engine's ABGR8 input (R, G, B, A bytes, k_packBgraRgb's rounding)
__global__ void k_rgbaHAbgr(const unsigned short* __restrict__ src, int srcPitch, int w, int h,
                            unsigned char* __restrict__ dst, int dstPitch)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const unsigned short* p = (const unsigned short*)((const unsigned char*)src + (size_t)y * srcPitch) + 4 * (size_t)x;
    unsigned char* q = dst + (size_t)y * dstPitch + 4 * (size_t)x;
    for (int c = 0; c < 3; c++)
    {
        const float v = h2f(p[c]);
        q[c] = (unsigned char)(int)rintf((v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v)) * 255.0f);
    }
    q[3] = 255;
}
// an FSR frame generation output (RGBA16F rows) into fp32 (R, G, B) planes, w x h, planes dps apart, rows drs apart; an
// SDR pair's tween back to the HDR codes (k_hdrFromSdr's way)
__global__ void k_unpackRgbaH(const unsigned short* __restrict__ src, int srcPitch, int w, int h, int sdr, int mode,
                              float white, float* __restrict__ dst, int dps, int drs)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const unsigned short* p = (const unsigned short*)((const unsigned char*)src + (size_t)y * srcPitch) + 4 * (size_t)x;
    float q[3] = { h2f(p[0]), h2f(p[1]), h2f(p[2]) };
    if (sdr)
    {
        for (int c = 0; c < 3; c++)
            q[c] = srgbEotf(q[c] < 0.0f ? 0.0f : (q[c] > 1.0f ? 1.0f : q[c])) * white;
        if (mode == 2) scrgb_to_hlg2020(q[0], q[1], q[2], q[0], q[1], q[2]);
        else scrgb_to_pq2020(q[0], q[1], q[2], q[0], q[1], q[2]);
    }
    const size_t d = (size_t)y * drs + x;
    dst[d] = q[0];
    dst[(size_t)dps + d] = q[1];
    dst[2 * (size_t)dps + d] = q[2];
}
// FSR's motion vectors: a field in px (two w x h planes, k_nvofUp's output) as R16G16_FLOAT texels, rows dstPitch bytes
// apart, every vector as it is (k_nrMv without its test)
// gate: a vector shorter than it (px) is written as exactly zero (0 = off): FSR upscaling's history holds still
// detail only under exact zeros (nativeFsrupEval)
__global__ void k_mvHalf(const float* __restrict__ flow, int w, int h, unsigned int* __restrict__ dst, int dstPitch,
                         float gate)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t o = (size_t)y * w + x;
    const float fx = flow[o], fy = flow[(size_t)w * h + o];
    dst[(size_t)y * (dstPitch >> 2) + x] =
        fx * fx + fy * fy < gate * gate ? 0u : (unsigned int)f2h(fx) | ((unsigned int)f2h(fy) << 16);
}
// the same texels from the engine's grid-1 field in one pass: vec = w x h short2 (raw / 32 = px, rows vecPitch bytes),
// k_nvofUp at grid 1 (weights 0 and 1, an exact copy) then k_mvHalf, byte for byte
__global__ void k_mvHalfG1(const short* __restrict__ vec, int vecPitch, int w, int h, unsigned int* __restrict__ dst,
                           int dstPitch, float gate)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const short* r = (const short*)((const char*)vec + (size_t)y * vecPitch);
    const float fx = (float)r[2 * x] * (1.0f / 32.0f), fy = (float)r[2 * x + 1] * (1.0f / 32.0f);
    dst[(size_t)y * (dstPitch >> 2) + x] =
        fx * fx + fy * fy < gate * gate ? 0u : (unsigned int)f2h(fx) | ((unsigned int)f2h(fy) << 16);
}
// a chained upscale pass's vectors (AMD FSR / Intel XeSS-SR past 3x, fuChainPlan): the first pass's R16G16_FLOAT
// texels (sw x sh, its render size) at this pass's render size dw x dh, the nearest texel times the size ratio (kx,
// ky) = render pixels of this pass; an exact zero (k_mvHalf's gate) stays zero
__global__ void k_fuMvScale(const unsigned int* __restrict__ src, int srcPitch, int sw, int sh,
                            unsigned int* __restrict__ dst, int dstPitch, int dw, int dh, float kx, float ky)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    const int sx = min(sw - 1, (int)(((float)x + 0.5f) * (float)sw / (float)dw));
    const int sy = min(sh - 1, (int)(((float)y + 0.5f) * (float)sh / (float)dh));
    const unsigned int v = src[(size_t)sy * (srcPitch >> 2) + sx];
    dst[(size_t)y * (dstPitch >> 2) + x] =
        v == 0u ? 0u
                : (unsigned int)f2h(kx * h2f((unsigned short)(v & 0xFFFFu))) |
                      ((unsigned int)f2h(ky * h2f((unsigned short)(v >> 16))) << 16);
}
// the engine's grid-1 field (short2, raw / 32 = px) bilinear at (x, y), clamped to the frame
__device__ __forceinline__ void mvAt(const short* __restrict__ vec, int vecPitch, int w, int h, float x, float y,
                                     float& vx, float& vy)
{
    x = fminf(fmaxf(x, 0.0f), (float)(w - 1));
    y = fminf(fmaxf(y, 0.0f), (float)(h - 1));
    const int x0 = (int)x, y0 = (int)y, x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
    const float fx = x - (float)x0, fy = y - (float)y0;
    const short* r0 = (const short*)((const char*)vec + (size_t)y0 * vecPitch);
    const short* r1 = (const short*)((const char*)vec + (size_t)y1 * vecPitch);
    const float a = (1.0f - fx) * (1.0f - fy), b = fx * (1.0f - fy), c = (1.0f - fx) * fy, d = fx * fy;
    vx = (a * r0[2 * x0] + b * r0[2 * x1] + c * r1[2 * x0] + d * r1[2 * x1]) * (1.0f / 32.0f);
    vy = (a * r0[2 * x0 + 1] + b * r0[2 * x1 + 1] + c * r1[2 * x0 + 1] + d * r1[2 * x1 + 1]) * (1.0f / 32.0f);
}
// AMD FSR 4's level-2+ call (its two frames at t0 and t1 of the pair) as R16G16_FLOAT texels from the pair's level-1
// field (B -> A at B's positions, grid 1): under the pair's straight-line motion the content at p in the frame at t1
// sits at the B position q with p = q + (1 - t1) v(q) (a 3-step fixed point) and at t0 at q + (1 - t0) v(q), so the
// call's field (its frame -> its previous, FSR's convention) = (t1 - t0) v(q)
__global__ void k_mvDerive(const short* __restrict__ vec, int vecPitch, int w, int h, float t0, float t1,
                           unsigned int* __restrict__ dst, int dstPitch)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float vx, vy, qx = (float)x, qy = (float)y;
    for (int it = 0; it < 3; it++)
    {
        mvAt(vec, vecPitch, w, h, qx, qy, vx, vy);
        qx = (float)x - (1.0f - t1) * vx;
        qy = (float)y - (1.0f - t1) * vy;
    }
    mvAt(vec, vecPitch, w, h, qx, qy, vx, vy);
    dst[(size_t)y * (dstPitch >> 2) + x] =
        (unsigned int)f2h((t1 - t0) * vx) | ((unsigned int)f2h((t1 - t0) * vy) << 16);
}
// AMD FSR upscaling: its tight BGRA8 input frame as RGBA8 rows twice, the bridge's (aPitch bytes apart) and the Optical
// Flow engine's ABGR8 input (bPitch); and an RGBA8 frame (srcPitch) as BGRA8 rows dstPitch apart: FSR's result into its
// tight buffer, RTX VSR's output into a live slot
__global__ void k_bgraRgbaFsr(const unsigned char* __restrict__ src, int w, int h, unsigned char* __restrict__ a,
                              int aPitch, unsigned char* __restrict__ b, int bPitch)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const unsigned int p = ((const unsigned int*)src)[(size_t)y * w + x];
    const unsigned int q = (p & 0xFF00FF00u) | ((p & 0xFFu) << 16) | ((p >> 16) & 0xFFu); // bytes 0 and 2 swapped
    if (a) *(unsigned int*)(a + (size_t)y * aPitch + 4 * (size_t)x) = q; // null on HDR planes (k_fsrHdrIn's frame)
    *(unsigned int*)(b + (size_t)y * bPitch + 4 * (size_t)x) = q;
}
// an odd output's last column and row, which AMD FSR upscaling leaves untouched: a copy of their neighbours (tight
// 32-bit pixels, w x h)
__global__ void k_oddEdge(unsigned int* dst, int w, int h)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const int sx = (w & 1) && x == w - 1 ? x - 1 : x, sy = (h & 1) && y == h - 1 ? y - 1 : y;
    if (sx != x || sy != y) dst[(size_t)y * w + x] = dst[(size_t)sy * w + sx];
}
__global__ void k_rgbaBgra(const unsigned char* __restrict__ src, int srcPitch, int w, int h,
                           unsigned char* __restrict__ dst, int dstPitch)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const unsigned int p = *(const unsigned int*)(src + (size_t)y * srcPitch + 4 * (size_t)x);
    *(unsigned int*)(dst + (size_t)y * dstPitch + 4 * (size_t)x) =
        (p & 0xFF00FF00u) | ((p & 0xFFu) << 16) | ((p >> 16) & 0xFFu);
}
// ---- AMD FSR upscaling on HDR planes: the bridge's HDR mode takes linear light (FP16 RGBA), so the PQ codes go in
// through the PQ EOTF per channel with SDR white at 1.0 (BT.2020 light as it is: no matrix, nothing clipped; scale =
// 10000 nits over SDR white) and come back through the inverse; an HLG picture (offline HLG video) goes as PQ on
// BT.2100's reference display first and comes back through the exact inverse (k_packR10's rule). The output's odd
// last column / row, which the upscaler leaves untouched, read their neighbours (k_oddEdge's rule). ----
__global__ void k_fsrHdrIn(const void* __restrict__ src, int half, int planeStride, int rowStride, int w, int h,
                           float scale, int hlg, unsigned char* __restrict__ dst, int dstPitch)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    unsigned short* p = (unsigned short*)(dst + (size_t)y * dstPitch) + 4 * (size_t)x;
    const size_t o = (size_t)y * rowStride + x;
    float v[3] = { ldS(src, half, o), ldS(src, half, planeStride + o), ldS(src, half, 2 * planeStride + o) };
    if (hlg) hlg2020_to_pq2020(v[0], v[1], v[2], v[0], v[1], v[2]);
    p[0] = f2h(pq_eotf(v[0]) * scale);
    p[1] = f2h(pq_eotf(v[1]) * scale);
    p[2] = f2h(pq_eotf(v[2]) * scale);
    p[3] = f2h(1.0f);
}
__global__ void k_fsrHdrOut(const unsigned char* __restrict__ src, int srcPitch, int w, int h, float inv, int hlg,
                            float* __restrict__ dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const int sx = (w & 1) && x == w - 1 ? x - 1 : x, sy = (h & 1) && y == h - 1 ? y - 1 : y;
    const unsigned short* p = (const unsigned short*)(src + (size_t)sy * srcPitch) + 4 * (size_t)sx;
    const size_t plane = (size_t)w * h;
    const size_t o = (size_t)y * w + x;
    float q[3] = { pq_oetf(fmaxf(h2f(p[0]), 0.0f) * inv), pq_oetf(fmaxf(h2f(p[1]), 0.0f) * inv),
                   pq_oetf(fmaxf(h2f(p[2]), 0.0f) * inv) };
    if (hlg) pq2020_to_hlg2020(q[0], q[1], q[2], q[0], q[1], q[2]);
    dst[o] = q[0];
    dst[plane + o] = q[1];
    dst[2 * plane + o] = q[2];
}
// AMD FSR upscaling on HDR planes in one pass over the frame: the frame's SDR view (hdrToModel, knee 1, head 0) packed
// to the vectors' BGRA8 (k_hdrEnc's view then k_packBgraRgb's bytes, the fp32 view never stored) and the codes in
// linear light into the bridge's FP16 buffer (k_fsrHdrIn's halves): one read of the planes in place of three
__global__ void k_fsrHdrPack(const void* __restrict__ src, int half, int planeStride, int rowStride, int w, int h,
                             int mode, float white, float scale, int hlg, unsigned char* __restrict__ view,
                             unsigned char* __restrict__ dst, int dstPitch)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t o = (size_t)y * rowStride + x;
    float v[3] = { ldS(src, half, o), ldS(src, half, planeStride + o), ldS(src, half, 2 * planeStride + o) };
    float s[3] = { v[0], v[1], v[2] };
    hdrToModel(s, mode, white, 1.0f, 0.0f);
    unsigned char* b = view + ((size_t)y * w + x) * 4;
    for (int ci = 0; ci < 3; ci++)
    {
        const float c = s[ci] < 0.0f ? 0.0f : (s[ci] > 1.0f ? 1.0f : s[ci]);
        b[2 - ci] = (unsigned char)(int)rintf(c * 255.0f);
    }
    b[3] = 255;
    unsigned short* p = (unsigned short*)(dst + (size_t)y * dstPitch) + 4 * (size_t)x;
    if (hlg) hlg2020_to_pq2020(v[0], v[1], v[2], v[0], v[1], v[2]);
    p[0] = f2h(pq_eotf(v[0]) * scale);
    p[1] = f2h(pq_eotf(v[1]) * scale);
    p[2] = f2h(pq_eotf(v[2]) * scale);
    p[3] = f2h(1.0f);
}
// Restore on HDR planes reads the frame's SDR view (hdrToModel, knee 1, head 0: the SDR range exactly, a brighter pixel
// scaled to white with its hue); the way back = the restored view (sRGB codes, the fold's 0..1) through the sRGB EOTF
// plus the light the view never held (ref's light minus its view: above white, below 0), times white, to the BT.2020
// codes (DLSS 5's way on HDR). ref = the HDR frame at the restored size; in place when rest is dst.
__global__ void k_hdrRestOut(const float* rest, int rps, int rrs, const void* __restrict__ ref, int half, int fps,
                             int frs, int w, int h, float* dst, int dps, int drs, int mode, float white)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t r = (size_t)y * rrs + x, f = (size_t)y * frs + x, d = (size_t)y * drs + x;
    float l[3], q[3], m = 0.0f;
    hdr2020_to_scrgb(ldS(ref, half, f), ldS(ref, half, (size_t)fps + f), ldS(ref, half, 2 * (size_t)fps + f), mode == 2,
                     l[0], l[1], l[2]);
    for (int c = 0; c < 3; c++)
    {
        l[c] /= white;
        m = fmaxf(m, l[c]);
    }
    const float k = m > 1.0f ? 1.0f / m : 1.0f;
    for (int c = 0; c < 3; c++)
    {
        const float v = rest[(size_t)c * rps + r];
        q[c] = (srgbEotf(v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v)) + l[c] - fmaxf(l[c] * k, 0.0f)) * white;
    }
    if (mode == 2) scrgb_to_hlg2020(q[0], q[1], q[2], q[0], q[1], q[2]);
    else scrgb_to_pq2020(q[0], q[1], q[2], q[0], q[1], q[2]);
    dst[d] = q[0];
    dst[(size_t)dps + d] = q[1];
    dst[2 * (size_t)dps + d] = q[2];
}

// RIFE's last step (IFNet_HDv3's merged[4]): tween = warp(frame 0, flow 0->t) x mask + warp(frame 1,
// flow 1->t) x (1 - mask), warp = grid_sample(bilinear, border, align_corners=True) = a bilinear tap
// at x + f clamped into [0, cw - 1] x [0, ch - 1] (nvTap). Per batch entry blockIdx.z: the flow (4
// fp16 planes: 0->t x, y, 1->t x, y) and the mask (1 fp16 plane) at the model's frame (fw x fh
// content, plane stride mps, row stride mrs). A picture of another size reads them the way RIFE's
// scale mode resizes a flow (bilinear, align_corners=False: the output pixel centre at (x + 0.5) fw /
// w - 0.5 in the model's frame, clamped at 0, as an exact integer ratio; the flow times w / fw and
// h / fh). The pictures: frame 0 in planes 0..2, frame 1 in 3..5 (plane stride pps, row stride prs,
// fp32 or fp16 by half); the tween: 3 fp16 planes per batch entry at the pictures' layout.
__device__ __forceinline__ float rbLerp(const void* __restrict__ p, int half, size_t base, int rs, const NvTap& T)
{
    const float a = (1.0f - T.lx) * ldS(p, half, base + (size_t)T.y0 * rs + T.x0) +
                    T.lx * ldS(p, half, base + (size_t)T.y0 * rs + T.x1);
    const float b = (1.0f - T.lx) * ldS(p, half, base + (size_t)T.y1 * rs + T.x0) +
                    T.lx * ldS(p, half, base + (size_t)T.y1 * rs + T.x1);
    return (1.0f - T.ly) * a + T.ly * b;
}
// rbLerp over a zero pad (RIFE's author pads the frames with zeros and warps the padded frame): a tap at or past the
// picture's w x h reads 0
__device__ __forceinline__ float rbAtZ(const void* __restrict__ p, int half, size_t base, int rs, int x, int y, int w,
                                       int h)
{
    return x < w && y < h ? ldS(p, half, base + (size_t)y * rs + x) : 0.0f;
}
__device__ __forceinline__ float rbLerpZ(const void* __restrict__ p, int half, size_t base, int rs, const NvTap& T, int w,
                                         int h)
{
    const float a = (1.0f - T.lx) * rbAtZ(p, half, base, rs, T.x0, T.y0, w, h) +
                    T.lx * rbAtZ(p, half, base, rs, T.x1, T.y0, w, h);
    const float b = (1.0f - T.lx) * rbAtZ(p, half, base, rs, T.x0, T.y1, w, h) +
                    T.lx * rbAtZ(p, half, base, rs, T.x1, T.y1, w, h);
    return (1.0f - T.ly) * a + T.ly * b;
}
__device__ __forceinline__ void rbAxis(int x, int w, int fw, int& a0, int& a1, float& l)
{
    const long long num = (2LL * x + 1) * fw - w, den = 2LL * w;
    a0 = 0;
    l = 0.0f;
    if (num > 0)
    {
        a0 = (int)(num / den);
        l = (float)(num % den) / (float)den;
    }
    if (a0 >= fw - 1)
    {
        a0 = fw - 1;
        l = 0.0f;
    }
    a1 = a0 < fw - 1 ? a0 + 1 : a0;
}
__device__ __forceinline__ float rbBil(const unsigned short* __restrict__ p, size_t o00, size_t o01, size_t o10,
                                       size_t o11, float lx, float ly)
{
    return (1.0f - ly) * ((1.0f - lx) * h2f(p[o00]) + lx * h2f(p[o01])) +
           ly * ((1.0f - lx) * h2f(p[o10]) + lx * h2f(p[o11]));
}
// one tween's flow F and mask M at picture pixel (x, y) (read at the model frame's fw x fh: bilinear, the
// flow times the ratio, when it differs from the pictures') as the two border taps and the mask
__device__ __forceinline__ void rbTaps(int x, int y, int w, int h, int cw, int ch, const unsigned short* __restrict__ F,
                                       const unsigned short* __restrict__ M, int mps, int mrs, int fw, int fh,
                                       NvTap& T0, NvTap& T1, float& m)
{
    float f[4];
    if (fw == w && fh == h)
    {
        const size_t o = (size_t)y * mrs + x;
        for (int c = 0; c < 4; c++)
            f[c] = h2f(F[c * (size_t)mps + o]);
        m = h2f(M[o]);
    }
    else
    {
        int x0, x1, y0, y1;
        float lx, ly;
        rbAxis(x, w, fw, x0, x1, lx);
        rbAxis(y, h, fh, y0, y1, ly);
        const size_t o00 = (size_t)y0 * mrs + x0, o01 = (size_t)y0 * mrs + x1;
        const size_t o10 = (size_t)y1 * mrs + x0, o11 = (size_t)y1 * mrs + x1;
        const float sx = (float)w / (float)fw, sy = (float)h / (float)fh;
        for (int c = 0; c < 4; c++)
            f[c] = rbBil(F + c * (size_t)mps, o00, o01, o10, o11, lx, ly) * ((c & 1) ? sy : sx);
        m = rbBil(M, o00, o01, o10, o11, lx, ly);
    }
    T0 = nvTap(x, y, f[0], f[1], cw, ch);
    T1 = nvTap(x, y, f[2], f[3], cw, ch);
}
// RIFE's last step: frame 0 in pic's planes 0..2 and frame 1 in 3..5, or the reverse (sw)
__global__ void k_rifeBlend(const void* __restrict__ pic, int sw, int half, int pps, int prs, int w, int h, int cw,
                            int ch, const unsigned short* __restrict__ flow, const unsigned short* __restrict__ mask,
                            int mps, int mrs, int fw, int fh, unsigned short* __restrict__ out, int zp)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t z = blockIdx.z;
    NvTap T0, T1;
    float m;
    rbTaps(x, y, w, h, cw, ch, flow + z * 4 * (size_t)mps, mask + z * (size_t)mps, mps, mrs, fw, fh, T0, T1, m);
    unsigned short* O = out + z * 3 * (size_t)pps;
    const size_t o = (size_t)y * prs + x;
    for (int c = 0; c < 3; c++)
    {
        const size_t b0 = (size_t)((sw ? 3 : 0) + c) * pps, b1 = (size_t)((sw ? 0 : 3) + c) * pps;
        const float a = zp ? rbLerpZ(pic, half, b0, prs, T0, w, h) : rbLerp(pic, half, b0, prs, T0);
        const float b = zp ? rbLerpZ(pic, half, b1, prs, T1, w, h) : rbLerp(pic, half, b1, prs, T1);
        O[(size_t)c * pps + o] = f2h(a * m + b * (1.0f - m));
    }
}
// k_rifeBlend with an fp32 store (outHalf false): the tween keeps the blend's fp32 value, so the reader that quantises
// it rounds once
__global__ void k_rifeBlendF(const void* __restrict__ pic, int sw, int half, int pps, int prs, int w, int h, int cw,
                             int ch, const unsigned short* __restrict__ flow, const unsigned short* __restrict__ mask,
                             int mps, int mrs, int fw, int fh, float* __restrict__ out, int zp)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t z = blockIdx.z;
    NvTap T0, T1;
    float m;
    rbTaps(x, y, w, h, cw, ch, flow + z * 4 * (size_t)mps, mask + z * (size_t)mps, mps, mrs, fw, fh, T0, T1, m);
    float* O = out + z * 3 * (size_t)pps;
    const size_t o = (size_t)y * prs + x;
    for (int c = 0; c < 3; c++)
    {
        const size_t b0 = (size_t)((sw ? 3 : 0) + c) * pps, b1 = (size_t)((sw ? 0 : 3) + c) * pps;
        const float a = zp ? rbLerpZ(pic, half, b0, prs, T0, w, h) : rbLerp(pic, half, b0, prs, T0);
        const float b = zp ? rbLerpZ(pic, half, b1, prs, T1, w, h) : rbLerp(pic, half, b1, prs, T1);
        O[(size_t)c * pps + o] = a * m + b * (1.0f - m);
    }
}
// RIFE's last step fused with a 1:1 slot store (live, the model frame = the presented rect): the fp32
// blend of one tween clamped and quantised straight into the slot as k_packOut (hdr 0: BGRA8) and
// k_packOutHdr (hdr 1: R10G10B10A2) store it, one rounding and no tween buffer
__global__ void k_rifeBlendOut(const void* __restrict__ pic, int sw, int half, int pps, int prs, int w, int h,
                               int cw, int ch, const unsigned short* __restrict__ flow,
                               const unsigned short* __restrict__ mask, int mps, int mrs, int fw, int fh, int hdr,
                               unsigned char* __restrict__ dst, int pitch, int x0, int y0, int zp)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    NvTap T0, T1;
    float m;
    rbTaps(x, y, w, h, cw, ch, flow, mask, mps, mrs, fw, fh, T0, T1, m);
    unsigned char* p = dst + (size_t)(y0 + y) * pitch + (size_t)(x0 + x) * 4;
    unsigned int word = 3u << 30;
    for (int c = 0; c < 3; c++)
    {
        const size_t b0 = (size_t)((sw ? 3 : 0) + c) * pps, b1 = (size_t)((sw ? 0 : 3) + c) * pps;
        const float a = zp ? rbLerpZ(pic, half, b0, prs, T0, w, h) : rbLerp(pic, half, b0, prs, T0);
        const float b = zp ? rbLerpZ(pic, half, b1, prs, T1, w, h) : rbLerp(pic, half, b1, prs, T1);
        float v = a * m + b * (1.0f - m);
        v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        if (hdr)
        {
            unsigned int q = (unsigned int)(int)(v * 1023.0f + 0.5f);
            if (q > 1023u) q = 1023u;
            word |= q << (10 * c);
        }
        else
            p[2 - c] = (unsigned char)(int)rintf(v * 255.0f);
    }
    if (hdr)
        *(unsigned int*)p = word;
    else
        p[3] = 255;
}
// k_rifeBlendOut for fp32 pictures: with the picture type known at compile time the 24 tap loads carry no per-load
// type test, ~10 to 15 % faster on smooth flows; the compiler pairs a few products into FMAs differently, so a rare
// code differs by one, and the codes sit as close to an fp64 blend as k_rifeBlendOut's
__global__ void k_rifeBlendOut32(const void* __restrict__ pic, int sw, int pps, int prs, int w, int h, int cw, int ch,
                                 const unsigned short* __restrict__ flow, const unsigned short* __restrict__ mask,
                                 int mps, int mrs, int fw, int fh, int hdr, unsigned char* __restrict__ dst, int pitch,
                                 int x0, int y0, int zp)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    NvTap T0, T1;
    float m;
    rbTaps(x, y, w, h, cw, ch, flow, mask, mps, mrs, fw, fh, T0, T1, m);
    unsigned char* p = dst + (size_t)(y0 + y) * pitch + (size_t)(x0 + x) * 4;
    unsigned int word = 3u << 30;
    for (int c = 0; c < 3; c++)
    {
        const size_t b0 = (size_t)((sw ? 3 : 0) + c) * pps, b1 = (size_t)((sw ? 0 : 3) + c) * pps;
        const float a = zp ? rbLerpZ(pic, 0, b0, prs, T0, w, h) : rbLerp(pic, 0, b0, prs, T0);
        const float b = zp ? rbLerpZ(pic, 0, b1, prs, T1, w, h) : rbLerp(pic, 0, b1, prs, T1);
        float v = a * m + b * (1.0f - m);
        v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        if (hdr)
        {
            unsigned int q = (unsigned int)(int)(v * 1023.0f + 0.5f);
            if (q > 1023u) q = 1023u;
            word |= q << (10 * c);
        }
        else
            p[2 - c] = (unsigned char)(int)rintf(v * 255.0f);
    }
    if (hdr)
        *(unsigned int*)p = word;
    else
        p[3] = 255;
}

extern "C" __global__ void k_nvofLuma(const float* __restrict__ src, int planeStride, int rowStride,
                                      int w, int h, unsigned char* __restrict__ dst, int pitch)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t o = (size_t)y * rowStride + x;
    // The host's planes are (R, G, B), the order of the BT.709 weights below. The luma is
    // rounded EXACTLY as an fp64 clip(v, 0, 1) * 255 would round: the constants, products and
    // sums carry their fp32 rounding error as a second float (ffMul / ffAdd), so the value is
    // known to ~1e-12 and the rounding takes the same side of every x.5 an fp64 sum takes. A
    // plain fp32 sum sits up to ~2.5e-5 off at 255 and moves up to 127 codes of a 1080p frame by
    // one; these codes feed the Optical Flow Accelerator, whose block matching can turn one
    // moved code into another vector.
    const float s[3] = { src[o], src[(long long)planeStride + (long long)o],
                         src[2 * (long long)planeStride + (long long)o] };
    const float kh[3] = { 0.2126f, 0.7152f, 0.0722f };                 // the double constants =
    const float kl[3] = { 7.247925e-09f, -6.9618227e-09f, -2.861023e-10f };  // kh + kl (1e-16)
    float H = 0.0f, L = 0.0f, p, e, t, f;
    for (int c = 0; c < 3; c++)
    {
        ffMul(s[c], kh[c], p, e);
        ffAdd(H, p, t, f);
        H = t;
        L += f + e + s[c] * kl[c];
    }
    ffAdd(H, L, t, f);
    ffMul(t, 255.0f, p, e);
    ffAdd(p, e + f * 255.0f, H, L);
    int code = (int)rintf(H);           // half to even on H; H + L decides an exact H tie
    const float d = H - (float)code;    // exact
    if (d == 0.5f && L > 0.0f) code++;
    else if (d == -0.5f && L < 0.0f) code--;
    dst[(size_t)y * pitch + x] = (unsigned char)(code < 0 ? 0 : (code > 255 ? 255 : code));
}

// one axis of k_nvofUp: the grid cell i0 / i1 and the lerp weight l of pixel o, f = (o + 0.5) /
// grid - 0.5 = (2o + 1 - grid) / (2 grid) clamped at 0, in integers; l = the remainder over
// 2 grid, exact in fp32 (and the lerps below exact) at the power-of-two grids
__device__ __forceinline__ void nvUpTap(int o, int grid, int g, int& i0, int& i1, float& l)
{
    const int num = 2 * o + 1 - grid, den = 2 * grid;
    int i = num > 0 ? num / den : 0;
    if (i > g - 1) i = g - 1;
    const int rem = num > 0 ? num - i * den : 0;
    l = rem >= den ? 1.0f : (float)rem / (float)den;
    i0 = i;
    i1 = i < g - 1 ? i + 1 : i;
}

// vec: gw x gh short2 (x, y) with a row pitch of vecPitch BYTES; cost: gw x gh uint8, costPitch
// bytes; out: flow (2 planes of w x h, px) and cost (1 plane, 0..255 as float); costOut null = the
// flow alone (cost is then never read)
extern "C" __global__ void k_nvofUp(const short* __restrict__ vec, int vecPitch,
                                    const unsigned char* __restrict__ cost, int costPitch,
                                    int gw, int gh, int grid, int w, int h,
                                    float* __restrict__ flow, float* __restrict__ costOut)
{
    // the taps depend on x alone (x0, x1, lx) or y alone (y0, y1, ly), so lane t of the block
    // fills column t and row t ONCE with nvUpTap, the pixels
    // read them; a block side above 32 = per pixel
    __shared__ int sX0[32], sX1[32], sY0[32], sY1[32];
    __shared__ float sLx[32], sLy[32];
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const bool tab = blockDim.x <= 32 && blockDim.y <= 32;
    if (tab)
    {
        const int t = threadIdx.y * blockDim.x + threadIdx.x;
        if (t < blockDim.x) nvUpTap(blockIdx.x * blockDim.x + t, grid, gw, sX0[t], sX1[t], sLx[t]);
        if (t < blockDim.y) nvUpTap(blockIdx.y * blockDim.y + t, grid, gh, sY0[t], sY1[t], sLy[t]);
        __syncthreads();
    }
    if (x >= w || y >= h) return;
    int x0, x1, y0, y1;
    float lx, ly;
    if (tab)
    {
        x0 = sX0[threadIdx.x]; x1 = sX1[threadIdx.x]; lx = sLx[threadIdx.x];
        y0 = sY0[threadIdx.y]; y1 = sY1[threadIdx.y]; ly = sLy[threadIdx.y];
    }
    else
    {
        nvUpTap(x, grid, gw, x0, x1, lx);
        nvUpTap(y, grid, gh, y0, y1, ly);
    }
    // exact in fp32 at grid <= 8: the weights carry at most 4 fraction bits, the shorts 15 bits
    const short* r0 = (const short*)((const char*)vec + (size_t)y0 * vecPitch);
    const short* r1 = (const short*)((const char*)vec + (size_t)y1 * vecPitch);
    const size_t plane = (size_t)w * h, o = (size_t)y * w + x;
    for (int c = 0; c < 2; c++)
    {
        const float a = (1.0f - lx) * (float)r0[2 * x0 + c] + lx * (float)r0[2 * x1 + c];
        const float b = (1.0f - lx) * (float)r1[2 * x0 + c] + lx * (float)r1[2 * x1 + c];
        flow[(size_t)c * plane + o] = ((1.0f - ly) * a + ly * b) * (1.0f / 32.0f);
    }
    if (!costOut) return;
    const unsigned char* c0 = cost + (size_t)y0 * costPitch;
    const unsigned char* c1 = cost + (size_t)y1 * costPitch;
    const float a = (1.0f - lx) * (float)c0[x0] + lx * (float)c0[x1];
    const float b = (1.0f - lx) * (float)c1[x0] + lx * (float)c1[x1];
    costOut[o] = (1.0f - ly) * a + ly * b;
}

// fA / costA: this direction (2 + 1 planes), fB: the other direction (2 planes); Z out (1 plane)
extern "C" __global__ void k_nvofMetric(const float* __restrict__ fA, const float* __restrict__ costA,
                                        const float* __restrict__ fB, int w, int h, float a, float b,
                                        float* __restrict__ Z)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t plane = (size_t)w * h, o = (size_t)y * w + x;
    const float ux = fA[o], uy = fA[plane + o];
    const NvTap T = nvTap(x, y, ux, uy, w, h);
    const float dx = ux + nvLerp(fB, w, T);
    const float dy = uy + nvLerp(fB + plane, w, T);
    // clamped at -16: a landing whose every candidate is bad keeps a weight (exp(-16) * NV_FIX
    // ~ 1.2e5 units) instead of vanishing; beyond a gap of 16 the softmax is winner-take-all anyway
    const float z = -a * costA[o] - b * sqrtf(dx * dx + dy * dy);
    Z[o] = z < -16.0f ? -16.0f : z;
}

// one source pixel: 3 image planes (planeStride, rowStride), flow (2, h, w), Z (1, h, w); splat
// along s * flow with weight exp(Z + bias) into 5 int64 planes of (w, h) (zeroed by the caller):
// RGB * e, e, and the plain bilinear COVERAGE (weight 1 per source pixel), which alone decides a
// hole, so the hole test does not depend on the metric's scale
extern "C" __global__ void k_splatNvof(const float* __restrict__ img, int planeStride, int rowStride,
                                       const float* __restrict__ flow, const float* __restrict__ Z,
                                       float s, float bias, int w, int h, long long* __restrict__ acc)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t plane = (size_t)w * h, o = (size_t)y * w + x;
    int nwX, nwY;
    float dx, dy;
    if (!land(x, s, flow[o], nwX, dx) || !land(y, s, flow[plane + o], nwY, dy)) return;
    const float e = expf(__fadd_rn(Z[o], bias));
    const float wNW = (1.0f - dx) * (1.0f - dy), wNE = dx * (1.0f - dy);
    const float wSW = (1.0f - dx) * dy, wSE = dx * dy;
    const bool okW = nwX >= 0 && nwX < w, okE = nwX + 1 >= 0 && nwX + 1 < w;
    const bool okN = nwY >= 0 && nwY < h, okS = nwY + 1 >= 0 && nwY + 1 < h;
    const long long oNW = (long long)nwY * w + nwX;   // only dereferenced behind the ok flags
    const size_t io = (size_t)y * rowStride + x;
    for (int c = 0; c <= 4; c++)
    {
        const float v = c < 3 ? __fmul_rn(img[(size_t)c * planeStride + io], e) : (c == 3 ? e : 1.0f);
        unsigned long long* p = (unsigned long long*)(acc + (size_t)c * plane);
        if (okN && okW) atomicAdd(p + oNW, (unsigned long long)(long long)llrintf(v * wNW * NV_FIX));
        if (okN && okE) atomicAdd(p + oNW + 1, (unsigned long long)(long long)llrintf(v * wNE * NV_FIX));
        if (okS && okW) atomicAdd(p + oNW + w, (unsigned long long)(long long)llrintf(v * wSW * NV_FIX));
        if (okS && okE) atomicAdd(p + oNW + w + 1, (unsigned long long)(long long)llrintf(v * wSE * NV_FIX));
    }
}

// acc (5 int64 planes of (w, h)) -> the tween (3 planes, planeStride / rowStride); a pixel whose
// coverage (plane 4, both splats summed, so up to 2 on a clean landing) is below `hole` blends
extern "C" __global__ void k_splatNvofNorm(const long long* __restrict__ acc, int w, int h,
                                           const float* __restrict__ i0, const float* __restrict__ i1,
                                           int planeStride, int rowStride, float t, float hole,
                                           float* __restrict__ dst, unsigned int* __restrict__ holes)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t plane = (size_t)w * h, o = (size_t)y * w + x, io = (size_t)y * rowStride + x;
    const float norm = (float)acc[3 * plane + o] * (1.0f / NV_FIX);
    const float cover = (float)acc[4 * plane + o] * (1.0f / NV_FIX);
    if (cover < hole || norm <= 0.0f)
    {
        for (int c = 0; c < 3; c++)
        {
            const size_t p = (size_t)c * planeStride + io;
            dst[p] = __fadd_rn(__fmul_rn(1.0f - t, i0[p]), __fmul_rn(t, i1[p]));
        }
        if (holes) atomicAdd(holes, 1u);   // null = uncounted (one contended address per hole)
        return;
    }
    for (int c = 0; c < 3; c++)
        dst[(size_t)c * planeStride + io] = __fdiv_rn((float)acc[(size_t)c * plane + o] * (1.0f / NV_FIX), norm);
}

// ---- the PULL warp with a confidence fallback (the tween the product runs; the forward splat
// above is not launched) ----------------------------------------------------------------------
// Per tween: (1) k_splatVel splats the VELOCITY (frame 0 -> 1 px) of both frames to time t with
// the softmax weights exp(Z + bias): frame 0 along t * F01 (velocity F01), frame 1 along
// (1 - t) * F10 (velocity -F10), each frame's plain coverage in its own plane (acc planes: vx e,
// vy e, e, cov0, cov1). (2) k_velNorm: the merged velocity, its confidence (the summed
// coverage, clamped to 1) premultiplied for the push-pull fill, the two visibilities (a frame
// whose pixels land here shows this spot) and the raw fallback mask: log of the mean weight
// q = e / (cov0 (1 - t) + cov1 t) mapped linearly from `lo` (blend) to `hi` (warp), 0 where
// nothing landed. (3) the push-pull pyramid fills the velocity gaps (k_ppDown, k_ppTop,
// k_ppUp). (4) k_blurH / k_blurV: the mask Gaussian-blurred (radius 4 sigma, reflect edges,
// scipy's gaussian_filter) so it carries no speckle of its own. (5) k_nvofCompose samples
// frame 0 at x - t V and frame 1 at x + (1 - t) V (bilinear, clamped), weighs them by time and
// visibility, and mixes the result with the plain blend by the mask: failed vectors become
// ghosting, never speckle. Checked against a reference implementation of the same tween.
// Launched 16 x 16 only: each block sums its landings in a shared window (the block's tile plus kSvMg px a side,
// moved by the block's mean landing offset) and adds each non-zero sum to global once; a landing outside the
// window goes to global directly. The sums are integers, so every total equals one global add a landing.
#define kSvMg 4
extern "C" __global__ void __launch_bounds__(256) k_splatVel(const float* __restrict__ flow,
                                                             const float* __restrict__ Z, float s, float bias,
                                                             float sign, int covPlane, int w, int h,
                                                             long long* __restrict__ acc)
{
    constexpr int RW = 16 + 2 * kSvMg, RN = RW * RW;
    __shared__ unsigned long long sa[4 * RN];
    __shared__ int sSum[3];
    const int tid = threadIdx.y * 16 + threadIdx.x;
    for (int i = tid; i < 4 * RN; i += 256) sa[i] = 0ull;
    if (tid < 3) sSum[tid] = 0;
    const int x = blockIdx.x * 16 + threadIdx.x;
    const int y = blockIdx.y * 16 + threadIdx.y;
    const size_t plane = (size_t)w * h, o = (size_t)y * w + x;
    int nwX = 0, nwY = 0;
    float dx = 0.0f, dy = 0.0f;
    const bool ok = x < w && y < h && land(x, s, flow[o], nwX, dx) && land(y, s, flow[plane + o], nwY, dy);
    __syncthreads();
    if (ok)
    {
        atomicAdd(&sSum[0], min(max(nwX - x, -64), 64));
        atomicAdd(&sSum[1], min(max(nwY - y, -64), 64));
        atomicAdd(&sSum[2], 1);
    }
    __syncthreads();
    int offX = 0, offY = 0;
    if (sSum[2])
    {
        offX = (int)floorf((float)sSum[0] / (float)sSum[2] + 0.5f);
        offY = (int)floorf((float)sSum[1] / (float)sSum[2] + 0.5f);
    }
    const int rx0 = blockIdx.x * 16 - kSvMg + offX, ry0 = blockIdx.y * 16 - kSvMg + offY;
    unsigned long long* a = (unsigned long long*)acc;
    if (ok)
    {
        const float e = expf(Z[o] + bias);
        const float vx = sign * flow[o], vy = sign * flow[plane + o];
        const float wt[4] = { (1.0f - dx) * (1.0f - dy), dx * (1.0f - dy), (1.0f - dx) * dy, dx * dy };
        const int ox[4] = { 0, 1, 0, 1 }, oy[4] = { 0, 0, 1, 1 };
        for (int k = 0; k < 4; k++)
        {
            const int tx = nwX + ox[k], ty = nwY + oy[k];
            if (tx < 0 || tx >= w || ty < 0 || ty >= h) continue;
            const float we = e * wt[k];
            const unsigned long long a0 = (unsigned long long)llrintf(vx * we * NV_FIX);
            const unsigned long long a1 = (unsigned long long)llrintf(vy * we * NV_FIX);
            const unsigned long long a2 = (unsigned long long)llrintf(we * NV_FIX);
            const unsigned long long a3 = (unsigned long long)llrintf(wt[k] * NV_FIX);
            const int lx = tx - rx0, ly = ty - ry0;
            if ((unsigned)lx < (unsigned)RW && (unsigned)ly < (unsigned)RW)
            {
                const int li = ly * RW + lx;
                atomicAdd(&sa[li], a0);
                atomicAdd(&sa[RN + li], a1);
                atomicAdd(&sa[2 * RN + li], a2);
                atomicAdd(&sa[3 * RN + li], a3);
            }
            else
            {
                const size_t t = (size_t)ty * w + tx;
                atomicAdd(a + t, a0);
                atomicAdd(a + plane + t, a1);
                atomicAdd(a + 2 * plane + t, a2);
                atomicAdd(a + (size_t)covPlane * plane + t, a3);
            }
        }
    }
    __syncthreads();
    for (int i = tid; i < RN; i += 256)
    {
        const int tx = rx0 + i % RW, ty = ry0 + i / RW;
        if (tx < 0 || tx >= w || ty < 0 || ty >= h) continue;
        const size_t t = (size_t)ty * w + tx;
        unsigned long long v;
        if ((v = sa[i]) != 0ull) atomicAdd(a + t, v);
        if ((v = sa[RN + i]) != 0ull) atomicAdd(a + plane + t, v);
        if ((v = sa[2 * RN + i]) != 0ull) atomicAdd(a + 2 * plane + t, v);
        if ((v = sa[3 * RN + i]) != 0ull) atomicAdd(a + (size_t)covPlane * plane + t, v);
    }
}

// acc (5 int64 planes) -> n0 = V * conf (2 planes), d0 = conf, vis (2 planes), mraw (1 plane)
extern "C" __global__ void k_velNorm(const long long* __restrict__ acc, int w, int h, float t,
                                     float lo, float hi, float* __restrict__ n0, float* __restrict__ d0,
                                     float* __restrict__ vis, float* __restrict__ mraw)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t plane = (size_t)w * h, o = (size_t)y * w + x;
    const float inv = 1.0f / NV_FIX;
    const float vxe = (float)acc[o] * inv, vye = (float)acc[plane + o] * inv;
    const float e = (float)acc[2 * plane + o] * inv;
    const long long a0 = acc[3 * plane + o], a1 = acc[4 * plane + o];
    const float c0 = (float)a0 * inv, c1 = (float)a1 * inv;
    // the velocity only where the weight sum is resolvable: a landing whose weight rounded to 0
    // in the fixed point while its velocity product did not would divide to infinity (and the
    // push-pull turns inf * 0 into NaN); a real landing weighs at least ~5e-8 (Z clamp -16, time
    // weight >= 0.25 at the grids used), so 1e-10 drops only corner crumbs and makes them holes
    // (e is exact in fp32 up to 2^24 units and 1e-10 * 2^40 = 109.95, so the test decides exactly
    // like the fp64 form did)
    const bool okE = e > 1e-10f;
    float conf = okE ? c0 + c1 : 0.0f;
    conf = conf < 0.0f ? 0.0f : (conf > 1.0f ? 1.0f : conf);
    n0[o] = okE ? vxe / e * conf : 0.0f;
    n0[plane + o] = okE ? vye / e * conf : 0.0f;
    d0[o] = conf;
    vis[o] = c0 < 0.0f ? 0.0f : (c0 > 1.0f ? 1.0f : c0);
    vis[plane + o] = c1 < 0.0f ? 0.0f : (c1 > 1.0f ? 1.0f : c1);
    float den = c0 * (1.0f - t) + c1 * t;
    den = den > 1e-12f ? den : 1e-12f;
    float q = e / den;
    q = q > 1e-30f ? q : 1e-30f;
    float m = (logf(q) - lo) / (hi - lo);
    m = m < 0.0f ? 0.0f : (m > 1.0f ? 1.0f : m);
    if (20LL * (a0 + a1) < (1LL << 40)) m = 0.0f;   // c0 + c1 < 0.05, exact on the integers
    mraw[o] = m;
}

// push-pull, down: level (w, h) -> (w2, h2) = ceil halves, 2x2 sums (zero outside the level)
extern "C" __global__ void k_ppDown(const float* __restrict__ n, const float* __restrict__ d, int w, int h,
                                    float* __restrict__ n2, float* __restrict__ d2, int w2, int h2)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w2 || y >= h2) return;
    const size_t plane = (size_t)w * h, plane2 = (size_t)w2 * h2, o2 = (size_t)y * w2 + x;
    float a = 0.0f, b = 0.0f, c = 0.0f;
    for (int dy = 0; dy < 2; dy++)
        for (int dx = 0; dx < 2; dx++)
        {
            const int sx = 2 * x + dx, sy = 2 * y + dy;
            if (sx >= w || sy >= h) continue;
            const size_t o = (size_t)sy * w + sx;
            a += n[o]; b += n[plane + o]; c += d[o];
        }
    n2[o2] = a; n2[plane2 + o2] = b; d2[o2] = c;
}

// push-pull, the coarsest level: out = n / max(d, 1e-12)
extern "C" __global__ void k_ppTop(const float* __restrict__ n, const float* __restrict__ d, int w, int h,
                                   float* __restrict__ out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t plane = (size_t)w * h, o = (size_t)y * w + x;
    const float dd = d[o] > 1e-12f ? d[o] : 1e-12f;
    out[o] = n[o] / dd;
    out[plane + o] = n[plane + o] / dd;
}

// push-pull, up: out = own * a + coarse(x / 2, y / 2) * (1 - a), own = n / max(d, 1e-12), a = clip(d)
extern "C" __global__ void k_ppUp(const float* __restrict__ n, const float* __restrict__ d, int w, int h,
                                  const float* __restrict__ coarse, int w2, int h2, float* __restrict__ out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t plane = (size_t)w * h, plane2 = (size_t)w2 * h2, o = (size_t)y * w + x;
    const size_t oc = (size_t)(y >> 1) * w2 + (x >> 1);
    const float dv = d[o];
    const float dd = dv > 1e-12f ? dv : 1e-12f;
    const float a = dv < 0.0f ? 0.0f : (dv > 1.0f ? 1.0f : dv);
    out[o] = n[o] / dd * a + coarse[oc] * (1.0f - a);
    out[plane + o] = n[plane + o] / dd * a + coarse[plane2 + oc] * (1.0f - a);
}

// Gaussian blur, one axis (dir 0 = x, 1 = y), radius r = int(4 sigma + 0.5), reflect edges
// (scipy's mode 'reflect': d c b a | a b c d), weights exp(-x^2 / (2 sigma^2)) normalised.
// The weights and their sum depend only on k, so each block computes them ONCE into shared
// memory (the sum in k order, as the per-pixel form adds them) instead of 2r + 1 exps per
// pixel, and loads its source span (16 lines of 16 + 2r samples, the reflection applied at the
// load) into shared memory once; every pixel then accumulates in fp32 in k order. Launched
// 16 x 16 only. A radius above kBlurMaxR keeps the per-pixel form.
#define kBlurMaxR 64
extern "C" __global__ void __launch_bounds__(256) k_blur1(const float* __restrict__ src, int w, int h, float sigma,
                                                          int dir, float* __restrict__ dst)
{
    __shared__ float wtab[2 * kBlurMaxR + 1];
    __shared__ float wsum;
    __shared__ float tile[16 * (16 + 2 * kBlurMaxR + 1)];
    const int x = blockIdx.x * 16 + threadIdx.x;
    const int y = blockIdx.y * 16 + threadIdx.y;
    const int r = (int)(4.0f * sigma + 0.5f);
    const bool tab = r <= kBlurMaxR;
    const float g = -0.5f / (sigma * sigma);
    const int n = dir ? h : w;
    if (tab)
    {
        const int tid = threadIdx.y * 16 + threadIdx.x;
        for (int k = tid; k <= 2 * r; k += 256)
            wtab[k] = expf(g * (float)((k - r) * (k - r)));
        // the x pass holds 16 rows of L samples (row stride Lp, odd), the y pass L rows of 16 columns
        const int L = 16 + 2 * r, Lp = L | 1;
        const int base = (dir ? blockIdx.y : blockIdx.x) * 16 - r;
        for (int q = tid; q < 16 * L; q += 256)
        {
            const int j = dir ? q / 16 : q % L, line = dir ? q % 16 : q / L;
            int i = base + j;
            // reflect about the edge sample boundary, repeated for kernels wider than the axis
            while (i < 0 || i >= n) i = i < 0 ? -i - 1 : 2 * n - i - 1;
            const int sx = dir ? blockIdx.x * 16 + line : i, sy = dir ? i : blockIdx.y * 16 + line;
            tile[dir ? j * 16 + line : line * Lp + j] = sx < w && sy < h ? src[(size_t)sy * w + sx] : 0.0f;
        }
        __syncthreads();
        if (tid == 0)
        {
            float ws = 0.0f;
            for (int k = 0; k <= 2 * r; k++) ws += wtab[k];
            wsum = ws;
        }
        __syncthreads();
        if (x >= w || y >= h) return;
        const float* t = dir ? tile + threadIdx.y * 16 + threadIdx.x : tile + threadIdx.y * Lp + threadIdx.x;
        const int st = dir ? 16 : 1;
        float acc = 0.0f;
        for (int k = -r; k <= r; k++)
            acc += wtab[k + r] * t[(k + r) * st];
        dst[(size_t)y * w + x] = acc / wsum;
        return;
    }
    if (x >= w || y >= h) return;
    const int c = dir ? y : x;
    float acc = 0.0f, ws = 0.0f;
    for (int k = -r; k <= r; k++)
    {
        int i = c + k;
        while (i < 0 || i >= n) i = i < 0 ? -i - 1 : 2 * n - i - 1;
        const float wk = expf(g * (float)(k * k));
        acc += wk * (dir ? src[(size_t)i * w + x] : src[(size_t)y * w + i]);
        ws += wk;
    }
    dst[(size_t)y * w + x] = acc / ws;
}

// the tween: V (2 planes, filled), vis (2 planes), m (the blurred mask), both frames at
// (planeStride, rowStride); out = m * warp + (1 - m) * blend into dst at the same strides.
// Frame 0 is sampled bilinear at x - t V, frame 1 at x + (1 - t) V, each point clamped into the
// frame (map_coordinates mode nearest) and taken as the pixel plus its offset (nvTap); a
// non-finite V samples the origin instead of indexing out of bounds. The taps serve all three
// channels.
extern "C" __global__ void k_nvofCompose(const float* __restrict__ i0, const float* __restrict__ i1,
                                         int planeStride, int rowStride, const float* __restrict__ V,
                                         const float* __restrict__ vis, const float* __restrict__ m,
                                         int w, int h, float t, float* __restrict__ dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t plane = (size_t)w * h, o = (size_t)y * w + x, io = (size_t)y * rowStride + x;
    const float vx = V[o], vy = V[plane + o];
    const bool fin = isfinite(vx) && isfinite(vy);
    const NvTap T0 = fin ? nvTap(x, y, -t * vx, -t * vy, w, h) : nvTap(0, 0, 0.0f, 0.0f, w, h);
    const NvTap T1 = fin ? nvTap(x, y, (1.0f - t) * vx, (1.0f - t) * vy, w, h) : nvTap(0, 0, 0.0f, 0.0f, w, h);
    const float w0 = (1.0f - t) * (vis[o] + 1e-3f), w1 = t * (vis[plane + o] + 1e-3f);
    const float mm = m[o];
    for (int c = 0; c < 3; c++)
    {
        const float* p0 = i0 + (size_t)c * planeStride;
        const float* p1 = i1 + (size_t)c * planeStride;
        const float c0 = nvLerp(p0, rowStride, T0);
        const float c1 = nvLerp(p1, rowStride, T1);
        const float warp = (w0 * c0 + w1 * c1) / (w0 + w1);
        const float blend = (1.0f - t) * p0[io] + t * p1[io];
        dst[(size_t)c * planeStride + io] = mm * warp + (1.0f - mm) * blend;
    }
}

// ---- native DRBA glue -----------------------------------------------------------------------
// The two pieces DRBA adds around the IFNet, ported from the python reference:
//   rife_backend.RIFE.calc_flow's tail: block0's flow (4 planes, 0->0.5 | 1->0.5, sampled from
//     block0's 1/16 planes, drbaFlowAt) splatted by
//     itself with softsplat 'avg', negated, holes (splatted ones < 0.999) filled with
//     max(H, W), times 2 = flow05 * 2 | flow15 * 2.
//   rife.drm.calc_drm_rife(tt, flow10, flow12, linear=True) for ONE side (each tween needs one):
//     d = |flow| + 1e-4, drm10 = d10 / (d10 + d12), drm12 = d12 / (d10 + d12); side -1 (between
//     I0 and I1) = drm_t1_t01 = avg splat of drm12 * tt * 2 along flow10 * that map, side +1 =
//     drm_t1_t12 = avg splat of drm10 * tt * 2 along flow12 * that map; holes take the unaligned
//     map. The result is the (1, 1, H, W) timestep map the IFNet engine takes.
// softsplat 'avg' = bilinear forward splat of [v, 1], out = v / (w + 1e-7); the ones splat has the
// same weights, so its mask is w / (w + 1e-7) and a hole is where that is < 0.999. Accumulators are
// int64 fixed point at 2^32 (the GMFSS pattern, order-independent, deterministic); a value that
// lands inside the frame is bounded by its own displacement (|v| <= max(W, H) + 1 for the flow
// splat, <= 2 for the DRM), so 2^32 leaves 1e5 overlapping sources of headroom at 8K. All fp32
// (fp64 cost 0.7 ms of the DRBA glue): a landing is land()'s (the pixel plus the floor and the
// exact fraction of the offset), and the hole test runs on the integer weight accumulator
// (DRBA_HOLE), so it takes exactly the fp64 form's decisions (an fp32 ratio test flips rare
// pixels between the fill and the splat, which jumps their timestep). Planes are contiguous
// (w * h each). Checked against fp64 references on real block0 flows.

#define DRBA_FIX 4294967296.0f
// hole = w / (w + 1e-7) < 0.999 = w < 9.99e-5 = acc < 9.99e-5 * 2^32 = 429067.23
#define DRBA_HOLE 429068LL

// the nv value(s) v of pixel (x, y) splatted at (x + ax * m, y + ay * m) (land) into nv value planes + the weight
// plane, through the 16 x 16 block's shared window sa ((nv + 1) planes of the tile plus kDrMg px a side, moved by the
// block's mean landing offset): every thread of the block calls it (ok = this pixel takes part); each non-zero window
// sum goes to global once, a landing outside the window goes to global directly. The sums are integers, so every total
// equals one global add a landing.
#define kDrMg 4
__device__ __forceinline__ void drbaSplatW(unsigned long long* p, size_t plane, int nv, const float* v, bool ok, int x,
                                           int y, float ax, float ay, float m, int w, int h, unsigned long long* sa,
                                           int* sSum)
{
    constexpr int RW = 16 + 2 * kDrMg, RN = RW * RW;
    const int tid = threadIdx.y * 16 + threadIdx.x;
    for (int i = tid; i < (nv + 1) * RN; i += 256) sa[i] = 0ull;
    if (tid < 3) sSum[tid] = 0;
    int nwX = 0, nwY = 0;
    float dx = 0.0f, dy = 0.0f;
    ok = ok && land(x, ax, m, nwX, dx) && land(y, ay, m, nwY, dy);
    __syncthreads();
    if (ok)
    {
        atomicAdd(&sSum[0], min(max(nwX - x, -64), 64));
        atomicAdd(&sSum[1], min(max(nwY - y, -64), 64));
        atomicAdd(&sSum[2], 1);
    }
    __syncthreads();
    int offX = 0, offY = 0;
    if (sSum[2])
    {
        offX = (int)floorf((float)sSum[0] / (float)sSum[2] + 0.5f);
        offY = (int)floorf((float)sSum[1] / (float)sSum[2] + 0.5f);
    }
    const int rx0 = blockIdx.x * 16 - kDrMg + offX, ry0 = blockIdx.y * 16 - kDrMg + offY;
    if (ok)
    {
        const float wt[4] = {(1.0f - dx) * (1.0f - dy), dx * (1.0f - dy), (1.0f - dx) * dy, dx * dy};
        const int cx[4] = {nwX, nwX + 1, nwX, nwX + 1}, cy[4] = {nwY, nwY, nwY + 1, nwY + 1};
        for (int k = 0; k < 4; k++)
        {
            if (cx[k] < 0 || cx[k] >= w || cy[k] < 0 || cy[k] >= h) continue;
            const int lx = cx[k] - rx0, ly = cy[k] - ry0;
            const bool in = (unsigned)lx < (unsigned)RW && (unsigned)ly < (unsigned)RW;
            const int li = ly * RW + lx;
            const size_t o = (size_t)cy[k] * w + cx[k];
            for (int c = 0; c < nv; c++)
            {
                const unsigned long long a = (unsigned long long)(long long)llrintf(v[c] * wt[k] * DRBA_FIX);
                if (in) atomicAdd(&sa[c * RN + li], a);
                else atomicAdd(p + (size_t)c * plane + o, a);
            }
            const unsigned long long a = (unsigned long long)(long long)llrintf(wt[k] * DRBA_FIX);
            if (in) atomicAdd(&sa[nv * RN + li], a);
            else atomicAdd(p + (size_t)nv * plane + o, a);
        }
    }
    __syncthreads();
    for (int i = tid; i < RN; i += 256)
    {
        const int tx = rx0 + i % RW, ty = ry0 + i / RW;
        if (tx < 0 || tx >= w || ty < 0 || ty >= h) continue;
        const size_t o = (size_t)ty * w + tx;
        for (int c = 0; c <= nv; c++)
        {
            const unsigned long long s = sa[c * RN + i];
            if (s != 0ull) atomicAdd(p + (size_t)c * plane + o, s);
        }
    }
    __syncthreads();
}

// block0's flow at pixel (x, y) of the w x h motion frame: block0 hands over its 4 planes at w / 16 x h / 16 (lo),
// and calc_flow's F.interpolate(size = (h, w), bilinear, align_corners = False) * 16 runs here, where its one reader
// samples it
__device__ __forceinline__ void drbaFlowAt(const float* __restrict__ lo, int w, int h, int x, int y, float* f)
{
    const int w16 = w >> 4, h16 = h >> 4;
    float sy = 0.0625f * ((float)y + 0.5f) - 0.5f, sx = 0.0625f * ((float)x + 0.5f) - 0.5f;
    sy = sy < 0.0f ? 0.0f : sy;
    sx = sx < 0.0f ? 0.0f : sx;
    const int y0 = (int)sy, x0 = (int)sx;
    const int y1 = y0 + (y0 < h16 - 1 ? 1 : 0), x1 = x0 + (x0 < w16 - 1 ? 1 : 0);
    const float ly1 = sy - (float)y0, ly0 = 1.0f - ly1, lx1 = sx - (float)x0, lx0 = 1.0f - lx1;
    const size_t po = (size_t)w16 * h16;
    for (int c = 0; c < 4; c++)
    {
        const float* p = lo + c * po;
        const float v = ly0 * (lx0 * p[y0 * w16 + x0] + lx1 * p[y0 * w16 + x1]) +
                        ly1 * (lx0 * p[y1 * w16 + x0] + lx1 * p[y1 * w16 + x1]);
        f[c] = v * 16.0f;
    }
}

// block0's flow (its 4 planes at w / 16 x h / 16, drbaFlowAt) -> acc (6 int64 planes, zeroed by the caller): half j
// = planes 3j, 3j+1 (x, y) + 3j+2 (weight); launched 16 x 16 only, the two halves through one window in turn
extern "C" __global__ void __launch_bounds__(256) k_drbaFlowSplat(const float* __restrict__ lo, int w, int h,
                                                                  long long* __restrict__ acc)
{
    __shared__ unsigned long long sa[3 * (16 + 2 * kDrMg) * (16 + 2 * kDrMg)];
    __shared__ int sSum[3];
    const int x = blockIdx.x * 16 + threadIdx.x;
    const int y = blockIdx.y * 16 + threadIdx.y;
    const bool in = x < w && y < h;
    const size_t plane = (size_t)w * h;
    float f[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (in) drbaFlowAt(lo, w, h, x, y, f);
    for (int j = 0; j < 2; j++)
    {
        const float v[2] = {f[2 * j], f[2 * j + 1]};
        drbaSplatW((unsigned long long*)(acc + (size_t)(3 * j) * plane), plane, 2, v, in, x, y, v[0], v[1], 1.0f, w, h,
                   sa, sSum);
    }
}

// acc (6 planes) -> out (4 planes): flow05 * 2 (x, y), flow15 * 2 (x, y)
extern "C" __global__ void k_drbaFlowNorm(const long long* __restrict__ acc, int w, int h,
                                          float* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t plane = (size_t)w * h;
    if (i >= (int)plane) return;
    const float fill = 2.0f * (float)(w > h ? w : h);
    for (int j = 0; j < 2; j++)
    {
        const long long aw = acc[(size_t)(3 * j + 2) * plane + i];
        const bool hole = aw < DRBA_HOLE;
        const float wsum = (float)aw / DRBA_FIX;
        for (int c = 0; c < 2; c++)
            out[(size_t)(2 * j + c) * plane + i] = hole ? fill
                : -2.0f * ((float)acc[(size_t)(3 * j + c) * plane + i] / DRBA_FIX) / (wsum + 1e-7f);
    }
}

// the unaligned DRM value at one pixel: side < 0 -> drm12 * tt * 2, side > 0 -> drm10 * tt * 2
__device__ __forceinline__ float drbaDrmUn(const float* f10, const float* f12, size_t plane,
                                           size_t o, int side, float tt)
{
    const float u0 = f10[o], v0 = f10[plane + o], u2 = f12[o], v2 = f12[plane + o];
    const float d10 = sqrtf(u0 * u0 + v0 * v0) + 1e-4f, d12 = sqrtf(u2 * u2 + v2 * v2) + 1e-4f;
    return (side < 0 ? d12 : d10) / (d10 + d12) * tt * 2.0f;
}

// f10, f12 (2 planes each, calc_flow's flow05 * 2) -> acc (2 int64 planes, zeroed): drm, weight; launched 16 x 16 only
extern "C" __global__ void __launch_bounds__(256) k_drbaDrmSplat(const float* __restrict__ f10,
                                                                 const float* __restrict__ f12, int side, float tt,
                                                                 int w, int h, long long* __restrict__ acc)
{
    __shared__ unsigned long long sa[2 * (16 + 2 * kDrMg) * (16 + 2 * kDrMg)];
    __shared__ int sSum[3];
    const int x = blockIdx.x * 16 + threadIdx.x;
    const int y = blockIdx.y * 16 + threadIdx.y;
    const bool in = x < w && y < h;
    const size_t plane = (size_t)w * h, o = in ? (size_t)y * w + x : 0;
    const float d = in ? drbaDrmUn(f10, f12, plane, o, side, tt) : 0.0f;
    const float* fa = side < 0 ? f10 : f12;
    const float v[1] = {d};
    drbaSplatW((unsigned long long*)acc, plane, 1, v, in, x, y, in ? fa[o] : 0.0f, in ? fa[plane + o] : 0.0f, d, w, h,
               sa, sSum);
}

// acc (2 planes) -> the (1, H, W) timestep map for the IFNet, in the engine's timestep dtype: fp32,
// or fp16 when half (rounded to nearest even, f2h's cvt; this block also compiles on its own in the
// harness, without f2h)
extern "C" __global__ void k_drbaDrmNorm(const long long* __restrict__ acc, const float* __restrict__ f10,
                                         const float* __restrict__ f12, int side, float tt, int w, int h,
                                         void* __restrict__ out, int half)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t plane = (size_t)w * h;
    if (i >= (int)plane) return;
    const long long aw = acc[plane + i];
    const float wsum = (float)aw / DRBA_FIX;
    const float v = aw < DRBA_HOLE ? drbaDrmUn(f10, f12, plane, i, side, tt)
                                   : ((float)acc[i] / DRBA_FIX) / (wsum + 1e-7f);
    if (half)
    {
        unsigned short hv;
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(hv) : "f"(v));
        ((unsigned short*)out)[i] = hv;
    }
    else
        ((float*)out)[i] = v;
}

// GMFlow's local correlation (gmflow_bidir's 1/4-scale window, between its two engines): out[b, p, k] = the dot of
// q[b, p, :] with f bilinearly sampled at window sample k of pixel p (GridSample's rules: align_corners, zero padding).
// The window position is GMFlow's normalize_coords of the pixel's grid point plus the sample's offset, computed here:
// pixel p = y W + x, sample k = (dy + r) side + (dx + r) with side x side = K, g = ((x + dx) - c) / c, c = (W - 1) / 2
// (and y alike), with normalize_coords' IEEE divide (bit for bit PyTorch's fp32 run of that graph at every quarter
// grid). The window's samples share one fractional offset, so their taps lie on a small integer grid: one warp a
// pixel takes the grid's dot products once (float4 lanes over the 128 channels, kept in shared memory, one cell of
// margin for rounding), then each sample combines its 4 corners with its own weights. Never builds the
// [B, 128, HW, K] product the engine wrote. Reads A's one fp16 exit qf [2B, H, W, 128] channels-last as it is: rows
// 0..B-1 = q (pixel p at y W + x), rows B..2B-1 = f; each lane widens its 4 channels exactly (one uint2 load), so the
// sums are the fp32 ones; 128 threads a block, one warp a pixel.
__device__ __forceinline__ float4 lcLoad4(const unsigned short* p)
{
    const uint2 u = *reinterpret_cast<const uint2*>(p);
    return make_float4(h2f((unsigned short)(u.x & 0xffffu)), h2f((unsigned short)(u.x >> 16)),
                       h2f((unsigned short)(u.y & 0xffffu)), h2f((unsigned short)(u.y >> 16)));
}

extern "C" __global__ void k_localCorr(const unsigned short* __restrict__ qf, float* __restrict__ out, int B, int H,
                                       int W, int HW, int K)
{
    __shared__ float dots[4][144];
    const int wib = threadIdx.x >> 5;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= B * HW) return;
    const int b = warp / HW, p = warp % HW;
    const float4 qv = lcLoad4(qf + ((size_t)b * HW + p) * 128 + lane * 4);
    const unsigned short* fb = qf + (size_t)(B + b) * H * W * 128;
    int side = 1;
    while (side * side < K) side++;
    const int r = side / 2, px = p % W, py = p / W;
    const float cxn = 0.5f * (float)(W - 1), cyn = 0.5f * (float)(H - 1);
    float mnx = 1e30f, mny = 1e30f;
    for (int k = lane; k < K; k += 32)
    {
        const float gx = __fdiv_rn((float)(px + k % side - r) - cxn, cxn);
        const float gy = __fdiv_rn((float)(py + k / side - r) - cyn, cyn);
        mnx = fminf(mnx, floorf((gx + 1.f) * 0.5f * (float)(W - 1)));
        mny = fminf(mny, floorf((gy + 1.f) * 0.5f * (float)(H - 1)));
    }
    for (int o = 16; o; o >>= 1)
    {
        mnx = fminf(mnx, __shfl_xor_sync(0xffffffffu, mnx, o));
        mny = fminf(mny, __shfl_xor_sync(0xffffffffu, mny, o));
    }
    const int ox = (int)mnx, oy = (int)mny;
    const int gs = side + 2;
    for (int t = 0; t < gs * gs; t++)
    {
        const int xx = ox + t % gs, yy = oy + t / gs;
        float s = 0.f;
        if (xx >= 0 && xx < W && yy >= 0 && yy < H)
        {
            const float4 v = lcLoad4(fb + ((size_t)yy * W + xx) * 128 + lane * 4);
            s = qv.x * v.x + qv.y * v.y + qv.z * v.z + qv.w * v.w;
        }
        for (int o = 16; o; o >>= 1)
            s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) dots[wib][t] = s;
    }
    __syncwarp();
    for (int k = lane; k < K; k += 32)
    {
        const float gx = __fdiv_rn((float)(px + k % side - r) - cxn, cxn);
        const float gy = __fdiv_rn((float)(py + k / side - r) - cyn, cyn);
        const float x = (gx + 1.f) * 0.5f * (float)(W - 1);
        const float y = (gy + 1.f) * 0.5f * (float)(H - 1);
        const float x0f = floorf(x), y0f = floorf(y);
        const float ax = x - x0f, ay = y - y0f;
        const int cx = (int)x0f - ox, cy = (int)y0f - oy;
        const float* d = dots[wib] + cy * gs + cx;
        // the bilinear combine with its roundings pinned (which product is rounded, which one fused), so the
        // correlation does not move with the compiler's contraction choices
        const float bx = 1.f - ax, by = 1.f - ay;
        const float top = __fmaf_rn(d[1], ax, __fmul_rn(d[0], bx));
        const float bot = __fmaf_rn(ax, d[gs + 1], __fmul_rn(d[gs], bx));
        out[((size_t)b * HW + p) * K + k] = __fmaf_rn(by, top, __fmul_rn(ay, bot));
    }
}

// GMFlow's global matching and its flow propagation (between gmflow_bidir_a_m and A, trt_runtime.gmflow_matching_cut):
// O[b, i] = sum_j softmax_j(Q[b, i] . K[b ^ kswap, j] * scale) V[b, j] with 128 channels and a 2-channel value. The
// matching: Q = K = mf (f0, f1 token-major), kswap 1 (each frame's tokens attend to the other's), V = the pixel grid
// (mode 1: x = j % gw, y = j / gw) and O minus the query's own grid point = the flow (mode 2); the propagation: mq / mk,
// kswap 0, V = that flow, stored in A's prop dtype (mode 4 = fp16). Q / K fp16 as half pairs in 32-bit words, V / O
// [B, N, 2]. A warp = 16 query rows, a block = 4 warps, grid (ceil(N / 64), B); keys in tiles of 64 through shared
// memory (rows padded to 68 words: the fragment loads hit 32 distinct banks); Q K^T on the tensor cores (mma.sync
// m16n8k16, fp32 accumulation), the softmax online in fp32 in base 2 (scale2 = scale * log2 e), the value product in
// fp32. Keys past N score -inf, rows past N are not stored.
#define AT_W 64
#define AT_KT 64
#define AT_SP 68
__device__ __forceinline__ void attnMma(float* d, const unsigned* a, unsigned b0, unsigned b1)
{
    asm volatile(SMV_MMA_F32 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float attnEx2(float x)
{
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ float attnQuad(float x, bool mx)
{
    float a = __shfl_xor_sync(0xffffffffu, x, 1);
    x = mx ? fmaxf(x, a) : x + a;
    a = __shfl_xor_sync(0xffffffffu, x, 2);
    return mx ? fmaxf(x, a) : x + a;
}
extern "C" __global__ void __launch_bounds__(128) k_attn2(const unsigned* __restrict__ q,
                                                          const unsigned* __restrict__ k,
                                                          const float* __restrict__ v, void* __restrict__ o, int n,
                                                          int kswap, float scale2, int mode, int gw)
{
    __shared__ __align__(16) unsigned sk[AT_KT * AT_SP];
    __shared__ float sv[AT_KT * 2];
    const float ninf = __int_as_float(0xff800000);
    const int b = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t4 = lane & 3;
    const int r0 = blockIdx.x * 64 + warp * 16 + g, r1 = r0 + 8;
    const unsigned* qb = q + (size_t)b * n * AT_W;
    const unsigned* kb = k + (size_t)(b ^ kswap) * n * AT_W;
    const float* vb = v + (size_t)b * n * 2;
    unsigned qa[8][4];
#pragma unroll
    for (int s = 0; s < 8; ++s)
    {
        qa[s][0] = r0 < n ? qb[(size_t)r0 * AT_W + s * 8 + t4] : 0u;
        qa[s][1] = r1 < n ? qb[(size_t)r1 * AT_W + s * 8 + t4] : 0u;
        qa[s][2] = r0 < n ? qb[(size_t)r0 * AT_W + s * 8 + 4 + t4] : 0u;
        qa[s][3] = r1 < n ? qb[(size_t)r1 * AT_W + s * 8 + 4 + t4] : 0u;
    }
    float m0 = ninf, m1 = ninf, l0 = 0.f, l1 = 0.f, x0 = 0.f, y0 = 0.f, x1 = 0.f, y1 = 0.f;
    for (int j0 = 0; j0 < n; j0 += AT_KT)
    {
        __syncthreads();
        for (int e = threadIdx.x; e < AT_KT * (AT_W / 4); e += 128)
        {
            const int row = e / (AT_W / 4), c4 = e % (AT_W / 4);
            uint4 w = make_uint4(0u, 0u, 0u, 0u);
            if (j0 + row < n)
                w = reinterpret_cast<const uint4*>(kb + (size_t)(j0 + row) * AT_W)[c4];
            reinterpret_cast<uint4*>(sk + row * AT_SP)[c4] = w;
        }
        if (threadIdx.x < AT_KT * 2)
        {
            const int j = j0 + (int)(threadIdx.x >> 1), c = threadIdx.x & 1;
            sv[threadIdx.x] = j >= n ? 0.f : (mode & 1) ? (float)(c ? j / gw : j % gw) : vb[(size_t)j * 2 + c];
        }
        __syncthreads();
        float acc[8][4];
#pragma unroll
        for (int t = 0; t < 8; ++t)
        {
            acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
            const unsigned* kr = sk + (t * 8 + g) * AT_SP + t4;
#pragma unroll
            for (int s = 0; s < 8; ++s)
                attnMma(acc[t], qa[s], kr[s * 8], kr[s * 8 + 4]);
        }
        float mx0 = m0, mx1 = m1;
#pragma unroll
        for (int t = 0; t < 8; ++t)
#pragma unroll
            for (int c = 0; c < 2; ++c)
            {
                const bool in = j0 + t * 8 + t4 * 2 + c < n;
                acc[t][c] = in ? acc[t][c] * scale2 : ninf;
                acc[t][2 + c] = in ? acc[t][2 + c] * scale2 : ninf;
                mx0 = fmaxf(mx0, acc[t][c]);
                mx1 = fmaxf(mx1, acc[t][2 + c]);
            }
        mx0 = attnQuad(mx0, true);
        mx1 = attnQuad(mx1, true);
        const float c0 = attnEx2(m0 - mx0), c1 = attnEx2(m1 - mx1);
        l0 *= c0;
        x0 *= c0;
        y0 *= c0;
        l1 *= c1;
        x1 *= c1;
        y1 *= c1;
#pragma unroll
        for (int t = 0; t < 8; ++t)
#pragma unroll
            for (int c = 0; c < 2; ++c)
            {
                const int key = t * 8 + t4 * 2 + c;
                const float vx = sv[key * 2], vy = sv[key * 2 + 1];
                const float p0 = attnEx2(acc[t][c] - mx0), p1 = attnEx2(acc[t][2 + c] - mx1);
                l0 += p0;
                x0 += p0 * vx;
                y0 += p0 * vy;
                l1 += p1;
                x1 += p1 * vx;
                y1 += p1 * vy;
            }
        m0 = mx0;
        m1 = mx1;
    }
    l0 = attnQuad(l0, false);
    x0 = attnQuad(x0, false);
    y0 = attnQuad(y0, false);
    l1 = attnQuad(l1, false);
    x1 = attnQuad(x1, false);
    y1 = attnQuad(y1, false);
    if (t4 != 0)
        return;
    for (int h = 0; h < 2; ++h)
    {
        const int r = h ? r1 : r0;
        if (r >= n)
            continue;
        float ox = (h ? x1 / l1 : x0 / l0), oy = (h ? y1 / l1 : y0 / l0);
        if (mode & 2)
        {
            ox -= (float)(r % gw);
            oy -= (float)(r / gw);
        }
        const size_t at = ((size_t)b * n + r) * 2;
        if (mode & 4)
        {
            unsigned short hx, hy;
            asm("cvt.rn.f16.f32 %0, %1;" : "=h"(hx) : "f"(ox));
            asm("cvt.rn.f16.f32 %0, %1;" : "=h"(hy) : "f"(oy));
            ((unsigned short*)o)[at] = hx;
            ((unsigned short*)o)[at + 1] = hy;
        }
        else
        {
            ((float*)o)[at] = ox;
            ((float*)o)[at + 1] = oy;
        }
    }
}
#undef AT_W
#undef AT_KT
#undef AT_SP

// GMFSS fusion's tail on the host: fusionnet ends at its 64 -> 256 conv (`out`, the pre-shuffle tensor at the half,
// fp16 [256][hh][hw]); pixel_shuffle(2) (CRD: channel 4c + 2 sy + sx = phase (sy, sx) of channel c) and conv_last
// (64 -> 3, 3x3, zero padding at the full size) run here in one pass, fp32 sums over fusionnet.pkl's conv_last in fp16
// (w_last [3][64][3][3] / b_last [3]), the frame written fp32 planar [3][2 hh][2 hw]. One thread = 2 x 2
// half pixels = a 4 x 4 full block; per input channel its 6 x 6 full window (full row 2 y0 - 1 + r = half row
// y0 - 1 + (r + 1) / 2 of phase plane sy = r even, columns alike) comes from 24 aligned pairs (half2), two a phase
// plane a row; a full row or column outside the frame reads 0 (zero padding; hh and hw are multiples of 4, the setup's
// check, so every pair inside the frame is whole). Every fixed-size loop unrolls, so the window, the sums and the
// weights stay in registers. clamp01 != 0 = python's torch.clamp(out, 0, 1) on the way out (restTap's form).
__global__ void k_fuTail(const unsigned short* __restrict__ pre, const unsigned short* __restrict__ wl,
                         const unsigned short* __restrict__ bl, float* __restrict__ out, int hh, int hw, int clamp01)
{
    __shared__ float sw[64 * 27]; // [c][o * 9 + ky * 3 + kx]: one channel's 27 weights side by side
    __shared__ float sb[3];
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    for (int i = tid; i < 3 * 64 * 9; i += blockDim.x * blockDim.y)
    {
        const int o = i / 576, c = (i % 576) / 9, k = i % 9;
        sw[c * 27 + o * 9 + k] = h2f(wl[i]);
    }
    if (tid < 3)
        sb[tid] = h2f(bl[tid]);
    __syncthreads();
    const int x0 = 2 * (blockIdx.x * blockDim.x + threadIdx.x), y0 = 2 * (blockIdx.y * blockDim.y + threadIdx.y);
    if (x0 >= hw || y0 >= hh)
        return;
    const int fh = 2 * hh;
    const size_t plane = (size_t)hh * hw;
    size_t rowOff[6];
    bool rowOk[6];
#pragma unroll
    for (int r = 0; r < 6; ++r)
    {
        const int R = 2 * y0 - 1 + r;
        rowOk[r] = R >= 0 && R < fh;
        rowOff[r] = rowOk[r] ? (size_t)(2 * (R & 1)) * plane + (size_t)(R >> 1) * hw : 0;
    }
    const bool left = x0 >= 2, right = x0 + 2 < hw;
    float acc[3][4][4];
#pragma unroll
    for (int o = 0; o < 3; ++o)
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j)
                acc[o][i][j] = sb[o];
    for (int c = 0; c < 64; ++c)
    {
        const unsigned short* p = pre + (size_t)(4 * c) * plane;
        float v[6][6];
#pragma unroll
        for (int r = 0; r < 6; ++r)
        {
            unsigned int a = 0, b = 0, cc = 0, d = 0;
            if (rowOk[r])
            {
                const unsigned short* q0 = p + rowOff[r] + x0; // phase sx = 0: half columns x0, x0 + 1, x0 + 2
                const unsigned short* q1 = q0 + plane;         // phase sx = 1: half columns x0 - 1, x0, x0 + 1
                a = *(const unsigned int*)q0;
                d = *(const unsigned int*)q1;
                if (right)
                    b = *(const unsigned int*)(q0 + 2);
                if (left)
                    cc = *(const unsigned int*)(q1 - 2);
            }
            v[r][0] = h2f((unsigned short)(cc >> 16));
            v[r][1] = h2f((unsigned short)(a & 0xFFFFu));
            v[r][2] = h2f((unsigned short)(d & 0xFFFFu));
            v[r][3] = h2f((unsigned short)(a >> 16));
            v[r][4] = h2f((unsigned short)(d >> 16));
            v[r][5] = h2f((unsigned short)(b & 0xFFFFu));
        }
        float w[27];
#pragma unroll
        for (int k = 0; k < 27; ++k)
            w[k] = sw[c * 27 + k];
#pragma unroll
        for (int o = 0; o < 3; ++o)
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j)
#pragma unroll
                    for (int ky = 0; ky < 3; ++ky)
#pragma unroll
                        for (int kx = 0; kx < 3; ++kx)
                            acc[o][i][j] += v[i + ky][j + kx] * w[o * 9 + ky * 3 + kx];
    }
    if (clamp01)
#pragma unroll
        for (int o = 0; o < 3; ++o)
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    acc[o][i][j] = acc[o][i][j] < 0.0f ? 0.0f : (acc[o][i][j] > 1.0f ? 1.0f : acc[o][i][j]);
    const int fw = 2 * hw;
#pragma unroll
    for (int o = 0; o < 3; ++o)
#pragma unroll
        for (int i = 0; i < 4; ++i)
            *(float4*)(out + (size_t)o * fh * fw + (size_t)(2 * y0 + i) * fw + 2 * x0) =
                make_float4(acc[o][i][0], acc[o][i][1], acc[o][i][2], acc[o][i][3]);
}

// k_fuTail on a channels-last `out` (plain NHWC fp16 [hh][hw][256] at 256
// channels, half pixel (y, x)'s channel 4c + 2 sy + sx = phase (sy, sx) of channel c): the same windows, sums and sum
// order, so the same frame bit for bit. Read straight from `out`, a warp's neighbours sit 1 KB apart, so the block's
// tile (its 16 x 32 half pixels, the 3 x 3 windows' border and one more column left so the pairs stay aligned: 18 x 36)
// comes in 4 channels at a time as whole 16-byte pieces (2 channels x 4 phases; two threads on one half pixel's 32
// bytes), each piece = 4 shared words, one a channel and phase row ((sx 0, sx 1): the pair the sums read), 0 outside
// the frame. The tile is sized for 16 x 8 threads.
#define FU_TR 18
#define FU_TC 36
#define FU_K 4
#define FU_P (FU_TR * FU_TC)
__global__ void k_fuTailNhwc(const unsigned short* __restrict__ pre, const unsigned short* __restrict__ wl,
                             const unsigned short* __restrict__ bl, float* __restrict__ out, int hh, int hw,
                             int clamp01)
{
    __shared__ float sw[64 * 27];
    __shared__ float sb[3];
    __shared__ __align__(16) unsigned int st[FU_K * 2 * FU_P]; // [k][sy][row][col] = (sx 0, sx 1)
    const int tid = threadIdx.y * blockDim.x + threadIdx.x, nt = blockDim.x * blockDim.y;
    for (int i = tid; i < 3 * 64 * 9; i += nt)
    {
        const int o = i / 576, c = (i % 576) / 9, k = i % 9;
        sw[c * 27 + o * 9 + k] = h2f(wl[i]);
    }
    if (tid < 3)
        sb[tid] = h2f(bl[tid]);
    const int x0 = 2 * (blockIdx.x * blockDim.x + threadIdx.x), y0 = 2 * (blockIdx.y * blockDim.y + threadIdx.y);
    const int bx = 2 * blockIdx.x * blockDim.x - 2, by = 2 * blockIdx.y * blockDim.y - 1; // the tile's first column / row
    const bool live = x0 < hw && y0 < hh;
    const int lx = 2 * threadIdx.x + 2, ly = 2 * threadIdx.y;
    __syncthreads();
    float acc[3][4][4];
#pragma unroll
    for (int o = 0; o < 3; ++o)
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j)
                acc[o][i][j] = sb[o];
    for (int c0 = 0; c0 < 64; c0 += FU_K)
    {
        if (c0)
            __syncthreads();
        for (int it = tid; it < FU_P * (FU_K / 2); it += nt)
        {
            const int seg = it % (FU_K / 2), pix = it / (FU_K / 2), rx = pix % FU_TC, ry = pix / FU_TC;
            const int X = bx + rx, Y = by + ry;
            uint4 u = make_uint4(0, 0, 0, 0);
            if (X >= 0 && X < hw && Y >= 0 && Y < hh)
                u = *(const uint4*)(pre + ((size_t)Y * hw + X) * 256 + 4 * c0 + 8 * seg);
            unsigned int* d = st + 4 * seg * FU_P + ry * FU_TC + rx;
            d[0] = u.x;
            d[FU_P] = u.y;
            d[2 * FU_P] = u.z;
            d[3 * FU_P] = u.w;
        }
        __syncthreads();
        if (live)
#pragma unroll
            for (int k = 0; k < FU_K; ++k)
            {
                const unsigned int* sk = st + 2 * k * FU_P;
                float v[6][6];
#pragma unroll
                for (int r = 0; r < 6; ++r)
                {
                    const int j = (r + 1) >> 1, sy = (r + 1) & 1;
                    const unsigned int* q = sk + sy * FU_P + (ly + j) * FU_TC + lx; // half column x0
                    const uint2 m = *(const uint2*)q;
                    const unsigned int cc = q[-1], d = q[2];
                    v[r][0] = h2f((unsigned short)(cc >> 16));
                    v[r][1] = h2f((unsigned short)(m.x & 0xFFFFu));
                    v[r][2] = h2f((unsigned short)(m.x >> 16));
                    v[r][3] = h2f((unsigned short)(m.y & 0xFFFFu));
                    v[r][4] = h2f((unsigned short)(m.y >> 16));
                    v[r][5] = h2f((unsigned short)(d & 0xFFFFu));
                }
                float w[27];
#pragma unroll
                for (int q = 0; q < 27; ++q)
                    w[q] = sw[(c0 + k) * 27 + q];
#pragma unroll
                for (int o = 0; o < 3; ++o)
#pragma unroll
                    for (int i = 0; i < 4; ++i)
#pragma unroll
                        for (int j = 0; j < 4; ++j)
#pragma unroll
                            for (int ky = 0; ky < 3; ++ky)
#pragma unroll
                                for (int kx = 0; kx < 3; ++kx)
                                    acc[o][i][j] += v[i + ky][j + kx] * w[o * 9 + ky * 3 + kx];
            }
    }
    if (!live)
        return;
    if (clamp01)
#pragma unroll
        for (int o = 0; o < 3; ++o)
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    acc[o][i][j] = acc[o][i][j] < 0.0f ? 0.0f : (acc[o][i][j] > 1.0f ? 1.0f : acc[o][i][j]);
    const int fh = 2 * hh, fw = 2 * hw;
#pragma unroll
    for (int o = 0; o < 3; ++o)
#pragma unroll
        for (int i = 0; i < 4; ++i)
            *(float4*)(out + (size_t)o * fh * fw + (size_t)(2 * y0 + i) * fw + 2 * x0) =
                make_float4(acc[o][i][0], acc[o][i][1], acc[o][i][2], acc[o][i][3]);
}
#undef FU_TR
#undef FU_TC
#undef FU_K
#undef FU_P

// ---- RIFE's encode (the IFNet Head) on fp32 FMAs: k_encC01 + k_encC2Up --------------------------------------------
// cnn0 3 -> 16 3x3 stride 2, cnn1 / cnn2 16 -> 16 3x3, each + LeakyReLU 0.2 at half size, then cnn3 = the 4x4 stride 2
// pad 1 transposed conv 16 -> kEncCo back to the full size (no activation). Both kernels are written after this text
// by encKernelText, every weight of kRifeEncW a float literal (an FFMA immediate: a constant-bank operand per FMA ran
// 1.8x slower). k_encC01 = cnn0 on a 32 x 8 block's 34 x 10 tile straight from the frame into shared memory, then
// cnn1 on it; k_encC2Up = cnn1's planes (16 fp32 at half size, the one scratch set) -> cnn2 on up's 34 x 10 tile in
// shared memory -> cnn3. The frame is even in both axes (the /64 pad). The host's copy: kEncCo = kRifeFeatC.
#define kEncCo 4

__device__ __forceinline__ float encLrelu(float v)
{
    return v >= 0.0f ? v : 0.2f * v;
}

// cnn0's input at full-size (y, x) of plane c: zero above / left of the frame (the stride 2 window never passes its
// right or bottom edge)
__device__ __forceinline__ float encC0Tap(const void* __restrict__ img, int inHalf, int W, size_t plane, int c, int y,
                                          int x)
{
    if (y < 0 || x < 0)
        return 0.0f;
    const size_t i = c * plane + (size_t)y * W + x;
    return inHalf ? h2f(((const unsigned short*)img)[i]) : ((const float*)img)[i];
}

// cnn1's planes -> s = [16][12][36]: tile row c * 12 + r = plane row by + r - 2, columns bx - 2 .. bx + 33 (zero
// outside); each warp loads whole rows, eight loads in flight a thread before their shared stores
__device__ __forceinline__ void encLoad36(float* __restrict__ s, const float* __restrict__ in, int w, int h, int bx,
                                          int by)
{
    const int lane = threadIdx.x, wp = threadIdx.y;
    const size_t plane = (size_t)w * h;
    const int xa = bx - 2 + lane, xb = bx + 30 + lane;
    const bool ina = xa >= 0 && xa < w, inb = lane < 4 && xb < w;
#pragma unroll
    for (int k0 = 0; k0 < 24; k0 += 8)
    {
        float va[8], vb[8];
#pragma unroll
        for (int k = 0; k < 8; k++)
        {
            const int rr = wp + (k0 + k) * 8, c = rr / 12, y = by + rr % 12 - 2;
            va[k] = vb[k] = 0.0f;
            if (y >= 0 && y < h)
            {
                const float* row = in + c * plane + (size_t)y * w;
                if (ina)
                    va[k] = row[xa];
                if (inb)
                    vb[k] = row[xb];
            }
        }
#pragma unroll
        for (int k = 0; k < 8; k++)
        {
            const int rr = wp + (k0 + k) * 8;
            s[rr * 36 + lane] = va[k];
            if (lane < 4)
                s[rr * 36 + 32 + lane] = vb[k];
        }
    }
    __syncthreads();
}

}

)CUDASRC";

// the kernel text's Lanczos radii (its #define lines, the one place they are set), for the tap tables' sizes
static int lzDefine(const char* name)
{
    const char* p = strstr(kNativeKernels, name);
    return p ? atoi(p + strlen(name)) : 0;
}
static const int kLzUp = lzDefine("#define kLzUp "), kLzDown = lzDefine("#define kLzDown ");
static int lzTaps(long long in, long long out)
{
    return in > out ? 2 * (int)((kLzDown * in + out - 1) / out) : 2 * kLzUp;
}

// ---- dynamically resolved entry points ----------------------------------------------------
// nvrtc ships no import library in the runtime wheel, so it is resolved by hand; cudart and the
// driver API are lazily loaded imports whose DLLs are pre-loaded by full path in
// nativeLoadDlls() (which is why no process-wide DLL search order is touched).
typedef int (*PFN_nvrtcCreateProgram)(void**, const char*, const char*, int, const char* const*, const char* const*);
typedef int (*PFN_nvrtcCompileProgram)(void*, int, const char* const*);
typedef int (*PFN_nvrtcGetCUBINSize)(void*, size_t*);
typedef int (*PFN_nvrtcGetCUBIN)(void*, char*);
typedef int (*PFN_nvrtcGetProgramLogSize)(void*, size_t*);
typedef int (*PFN_nvrtcGetProgramLog)(void*, char*);
typedef int (*PFN_nvrtcDestroyProgram)(void**);

struct NvrtcApi
{
    HMODULE mod = nullptr;
    PFN_nvrtcCreateProgram create = nullptr;
    PFN_nvrtcCompileProgram compile = nullptr;
    PFN_nvrtcGetCUBINSize cubinSize = nullptr;
    PFN_nvrtcGetCUBIN cubin = nullptr;
    PFN_nvrtcGetProgramLogSize logSize = nullptr;
    PFN_nvrtcGetProgramLog log = nullptr;
    PFN_nvrtcDestroyProgram destroy = nullptr;
    bool ok() const
    {
        return create && compile && cubinSize && cubin && destroy;
    }
};

static std::wstring g_nativeRuntimeDir; // engine\gpu_runtime: the CUDA 13 DLLs
static std::wstring g_nativeEngineDir;  // ...\engine itself, the parent of the folders above

#include "smv-live-native-rtxvideo.inl"

static bool nativeLoadDlls(const std::wstring& scriptPath)
{
    static int state = 0; // 0 untried, 1 ok, -1 failed
    if (state)
        return state > 0;
    // The shipped layout is engine\live\smv-live.exe next to engine\gpu_runtime, so the runtime
    // is resolved RELATIVE TO THIS EXE and never from a build-time absolute path. A dev exe that
    // lives outside the app tree (bin\smv-live.exe with an explicit --script) falls back to the
    // directory of the --script path it was pointed at. The chosen folder is always logged.
    auto dirOf = [](const std::wstring& p) -> std::wstring {
        const size_t sl = p.find_last_of(L"\\/");
        return sl == std::wstring::npos ? std::wstring() : p.substr(0, sl);
    };
    std::wstring engineDir;
    {
        wchar_t exePath[MAX_PATH]{};
        GetModuleFileNameW(nullptr, exePath, MAX_PATH);
        const std::wstring liveDir = dirOf(exePath); // ...\engine\live
        const std::wstring cand = dirOf(liveDir);    // ...\engine
        if (!cand.empty() && GetFileAttributesW((cand + L"\\gpu_runtime").c_str()) != INVALID_FILE_ATTRIBUTES)
            engineDir = cand;
    }
    if (engineDir.empty())
        engineDir = dirOf(scriptPath); // dev exe outside the app tree
    if (engineDir.empty())
    {
        LOG("native: cannot locate the app runtime folder\n");
        state = -1;
        return false;
    }
    g_nativeEngineDir = engineDir;
    g_nativeRuntimeDir = engineDir + L"\\gpu_runtime";
    LOG("native: runtime folder %s\n", wideToUtf8(g_nativeRuntimeDir).c_str());
    const std::wstring cudaBin = g_nativeRuntimeDir;
    // nvrtc opens nvrtc-builtins64_<ver>.dll BY NAME at compile time, and a module loaded with
    // LOAD_WITH_ALTERED_SEARCH_PATH does not make its own directory searchable for that later
    // call (measured: "nvrtc: error: failed to open nvrtc-builtins64_133.dll"), so the builtins
    // are pre-loaded by full path too and the by-name open then finds the loaded module.
    std::wstring builtins;
    {
        WIN32_FIND_DATAW fd{};
        const std::wstring pat = cudaBin + L"\\nvrtc-builtins64_*.dll";
        HANDLE hf = FindFirstFileW(pat.c_str(), &fd);
        if (hf != INVALID_HANDLE_VALUE)
        {
            builtins = fd.cFileName;
            FindClose(hf);
        }
    }
    struct
    {
        const wchar_t* dir;
        const wchar_t* dll;
        bool required;
    } want[] = {
        {cudaBin.c_str(), L"cudart64_13.dll", true},
        {cudaBin.c_str(), builtins.empty() ? L"" : builtins.c_str(), false},
        {cudaBin.c_str(), L"nvrtc64_130_0.dll", true},
    };
    for (auto& wdl : want)
    {
        if (!wdl.dll[0])
            continue;
        std::wstring full = std::wstring(wdl.dir) + L"\\" + wdl.dll;
        if (!LoadLibraryExW(full.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH) && wdl.required)
        {
            LOG("native: %s not loadable (err %lu), the native host cannot start\n", wideToUtf8(full).c_str(),
                GetLastError());
            state = -1;
            return false;
        }
    }
    // AddDllDirectory for anything either DLL loads later by name; harmless when the process
    // never opts into the safe search order.
    AddDllDirectory(cudaBin.c_str());
    state = 1;
    return true;
}

#include "smv-live-native-state.inl"
#include "smv-live-native-handoff.inl"

#include "smv-live-native-weights.inl"

#include "smv-live-native-engine-load.inl"

#include "smv-live-native-kernel-build.inl"
#include "smv-live-native-cuda-init.inl"

#include "smv-live-native-nvof.inl"

#include "smv-live-native-fruc.inl"

#include "smv-live-native-fsrfg.inl"

#include "smv-live-native-nvvfg.inl"

#include "smv-live-native-fsrup.inl"

#include "smv-live-native-anime4k.inl"

#include "smv-live-native-artcnn.inl"

#include "smv-live-native-anime4k-restore.inl"

#include "smv-live-native-model-setup.inl"

#include "smv-live-native-gmfss-featnet.inl"

#include "smv-live-native-drba-block0.inl"

#include "smv-live-native-gmfss-metricnet.inl"

static size_t giNatBytes(const NativeRife& nr);
static size_t fuNatBytes(const NativeRife& nr);
static size_t gbNatBytes(const NativeRife& nr);
static size_t bbNatBytes(const NativeRife& nr);
static size_t gaNatBytes(const NativeRife& nr);

// GMFSS's one activation block: Restore (live and offline, before the model on the same stream), feat_ext's scratch
// (the first of the pair's work on it), metricnet's, the GMFSS IFNet's, fusionnet's, gmflow B's, the backbone's and the
// transformer's activations, all serial on nr.stream; it takes the largest
static bool nativeGmShareMemory(NativeRife& nr)
{
    const int64_t rb = (int64_t)restNatBytes(nr), fb = (int64_t)fnNatBytes(nr, true), mb = (int64_t)mnNatBytes(nr),
                  gb = (int64_t)giNatBytes(nr), ub = (int64_t)fuNatBytes(nr), bb = (int64_t)gbNatBytes(nr),
                  kb = (int64_t)bbNatBytes(nr), ab = (int64_t)gaNatBytes(nr),
                  blk = (std::max)((std::max)((std::max)(rb, (std::max)(fb, mb)), (std::max)((std::max)(gb, ub), bb)),
                                   (std::max)(kb, ab));
    if (blk > 0 && cudaMalloc(&nr.dGmCtxMem, (size_t)blk) != cudaSuccess)
    {
        LOG("native: gmfss shared activation block (%.0f MB) alloc failed\n", blk / 1048576.0);
        return false;
    }
    nr.gmCtxMemB = blk;
    LOG("native: gmfss nets share one %.0f MB activation block\n", blk / 1048576.0);
    return true;
}

// a chain's graph mode (live on, offline only with SMV_OFFLINE_GRAPH=1); its switch env = 0 turns it off, = fail
// throws every capture away
static int chainGraphMode(const char* env)
{
    const std::string v = lkEnv(env);
    return v == "0" || (g_offline && !g_offlineGraph) ? 0 : v == "fail" ? 2 : 1;
}

static void chainGraphsFree(ChainGraphs& cg)
{
    for (ChainGraphs::Entry& x : cg.e)
    {
        if (x.exec)
            cudaGraphExecDestroy(x.exec);
        x = ChainGraphs::Entry();
    }
    cg.next = cg.mode = cg.captures = 0;
    cg.logged = false;
}

// the chain `launch` (its kernels on st, nothing else) through cg: the graph captured for key (the chain's buffers, up to
// kKey - 2) at w x h replays; a new key captures one (ThreadLocal stream capture); a failed capture = direct launches
// for the rest of the session (`native: NAME graph capture failed: direct launches`). Short kernels at small sizes leave
// the GPU waiting between single launches; a graph replays them back to back
template <class F>
static bool chainGraphRun(ChainGraphs& cg, std::initializer_list<const void*> bufs, int w, int h, cudaStream_t st,
                          const char* name, F&& launch)
{
    if (!cg.mode)
        return launch();
    const void* key[ChainGraphs::kKey] = {};
    int n = 0;
    for (const void* b : bufs)
        if (n < ChainGraphs::kKey - 2)
            key[n++] = b;
    key[ChainGraphs::kKey - 2] = (const void*)(intptr_t)w;
    key[ChainGraphs::kKey - 1] = (const void*)(intptr_t)h;
    for (const ChainGraphs::Entry& x : cg.e)
        if (x.exec && memcmp(x.key, key, sizeof(key)) == 0)
            return cudaGraphLaunch(x.exec, st) == cudaSuccess;
    if (++cg.captures > ChainGraphs::kMaxCaptures)
    {
        chainGraphsFree(cg);
        LOG("native: %s keeps new buffers: direct launches\n", name);
        return launch();
    }
    cudaGraph_t g = nullptr;
    cudaGraphExec_t ex = nullptr;
    bool ok = false;
    if (cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal) == cudaSuccess)
    {
        const bool launched = launch();
        ok = cudaStreamEndCapture(st, &g) == cudaSuccess && launched && g;
    }
    ok = ok && cg.mode != 2 && cudaGraphInstantiate(&ex, g, 0) == cudaSuccess;
    if (g)
        cudaGraphDestroy(g);
    if (!ok)
    {
        chainGraphsFree(cg);
        cudaGetLastError();
        LOG("native: %s graph capture failed: direct launches\n", name);
        return launch();
    }
    ChainGraphs::Entry& slot = cg.e[cg.next];
    cg.next = (cg.next + 1) % ChainGraphs::kN;
    if (slot.exec)
        cudaGraphExecDestroy(slot.exec);
    memcpy(slot.key, key, sizeof(key));
    slot.exec = ex;
    if (!cg.logged)
        LOG("native: %s as one CUDA graph (%dx%d)\n", name, w, h);
    cg.logged = true;
    return cudaGraphLaunch(ex, st) == cudaSuccess;
}

#include "smv-live-native-rife-ifnet.inl"

// RIFE's encode on the host's kernels: cnn1's half-size planes (16 fp32) at the encode's frame (the motion frame, else
// the pictures' pad), 0 = no IFNet in this session
static size_t encNatBytes(const NativeRife& nr)
{
    if (!nr.ifnNat)
        return 0;
    const size_t w = (size_t)(nr.mph ? nr.mpw : nr.pw), h = (size_t)(nr.mph ? nr.mph : nr.ph);
    return 16 * (w / 2) * (h / 2) * sizeof(float);
}

// The RIFE family's nets (the IFNet, Restore) enqueue one after another on nr.stream too, so they share ONE
// activation block, the largest one's, set before any warm-up (a graph capture records its address); the encode's
// planes (nativeEncodeRun, before the IFNet on the same stream) and DRBA block0's scratch (nativeB0Run, its flow read
// by the splat right after it) live in it too. SMV_CTXSHARE=0 = each net its own.
static bool nativeModelShareMemory(NativeRife& nr)
{
    int64_t need = 0, sum = 0;
    int n = 0;
    if (nr.restNat) // Restore on the host's kernels: its ping-pong takes the block too
    {
        const int64_t b = (int64_t)restNatBytes(nr);
        need = (std::max)(need, b);
        sum += b;
        n++;
    }
    if (const int64_t b = (int64_t)encNatBytes(nr))
    {
        need = (std::max)(need, b);
        sum += b;
        n++;
    }
    if (const int64_t b = (int64_t)b0NatBytes(nr))
    {
        need = (std::max)(need, b);
        sum += b;
        n++;
    }
    if (const int64_t b = (int64_t)ifnNatBytes(nr))
    {
        need = (std::max)(need, b);
        sum += b;
        n++;
    }
    if (!n)
        return true;
    if (need > 0 && cudaMalloc(&nr.dGmCtxMem, (size_t)need) != cudaSuccess)
    {
        LOG("native: shared activation block (%.0f MB) alloc failed\n", need / 1048576.0);
        return false;
    }
    nr.gmCtxMemB = need;
    LOG("native: the model contexts share one %.0f MB activation block (%.0f MB as %d)\n", need / 1048576.0,
        sum / 1048576.0, n);
    return true;
}

#include "smv-live-native-gmfss-ifnet.inl"

#include "smv-live-native-gmfss-fusionnet.inl"

#include "smv-live-native-gmflow-b.inl"

#include "smv-live-native-gmflow-backbone.inl"

#include "smv-live-native-gmflow-a.inl"

#include "smv-live-native-model-init.inl"

#include "smv-live-native-gmfss-chain.inl"
#include "smv-live-native-drba.inl"

#include "smv-live-native-restore-run.inl"

#include "smv-live-native-dlss5.inl"
#include "smv-live-native-video-memory.inl"

#include "smv-live-native-dlss5-frame.inl"

#include "smv-live-native-emit.inl"

#include "smv-live-native-group.inl"
#include "smv-live-native-thread.inl"
