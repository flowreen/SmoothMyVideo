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
// Anime4K and each ported net live in their own part, smv-live-native-<name>.inl (kernel text, loader and
// launch code together), #included further down in this order: anime4k, artcnn, anime4k-restore,
// gmfss-featnet, drba-block0, gmfss-metricnet, rife-ifnet, gmfss-ifnet, gmfss-fusionnet, gmflow-b,
// gmflow-backbone, gmflow-a. The order matters: a later part uses names an earlier one defines.

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

// ---- the RTX Video SDK CUDA bridge (engine\rtxvideo\rtxvideo_cuda.dll) --------------------
// The TrueHDR bridge is NOT an NGX D3D11 bridge: it is a plain C ABI over CUDA device
// pointers. The NGX feature DLLs
// (nvngx_truehdr.dll, nvngx_vsr.dll) resolve relative to the LOADING module's directory, so
// the bridge is loaded by full path out of its own folder and nothing is copied or split.
struct RtxRect
{
    uint32_t left, top, right, bottom;
};
struct RtxThdrSetting
{
    uint32_t Contrast, Saturation, MiddleGray, MaxLuminance;
};
struct RtxVsrSetting
{
    uint32_t QualityLevel;
}; // 0 bicubic .. 4 Ultra; the host asks --vsr-quality's level, 1..4 (vsrSet, default 4)
typedef void (*PFN_rtxvSetModelPath)(const wchar_t*);
typedef unsigned int (*PFN_rtxCreate)(void*, void*, int, unsigned int, unsigned int);
typedef unsigned int (*PFN_rtxEvalThdr)(void*, void*, RtxRect, RtxRect, RtxThdrSetting*);
typedef unsigned int (*PFN_rtxEvalVsr)(void*, void*, RtxRect, RtxRect, RtxVsrSetting*);
typedef unsigned int (*PFN_rtxEvalVsrAsync)(void*, void*, RtxRect, RtxRect, RtxVsrSetting*, void*);
typedef void (*PFN_rtxShutdown)();

struct RtxBridge
{
    HMODULE mod = nullptr;
    PFN_rtxvSetModelPath setModelPath = nullptr;
    PFN_rtxCreate create = nullptr;
    PFN_rtxEvalThdr evalThdr = nullptr;
    PFN_rtxEvalVsr evalVsr = nullptr; // live RTX VSR: 8-bit BGRA in and out
    // the same eval ordered against the caller's stream by events instead of blocking the thread (optional: an
    // older bridge lacks it and every caller keeps evalVsr)
    PFN_rtxEvalVsrAsync evalVsrAsync = nullptr;
    PFN_rtxShutdown shutdown = nullptr;
    bool created = false;
    bool vsrMade = false; // the one create held RTX VSR's feature (an upscaler switch back to RTX VSR reuses it)
    std::wstring dir;
};
static RtxBridge g_rtxb;

// Resolve the bridge folder (SMV_RTXVIDEO_DIR, else
// <engine>\rtxvideo) and bind the four entry points. Any failure logs one line naming the
// file.
static bool rtxBridgeLoad()
{
    if (g_rtxb.mod)
        return true;
    std::wstring dir;
    wchar_t ov[MAX_PATH]{};
    if (GetEnvironmentVariableW(L"SMV_RTXVIDEO_DIR", ov, MAX_PATH) && ov[0])
        dir = ov;
    else if (!g_nativeEngineDir.empty())
        dir = g_nativeEngineDir + L"\\rtxvideo";
    if (dir.empty())
    {
        LOG("native: cannot locate the rtxvideo folder\n");
        return false;
    }
    const std::wstring dll = dir + L"\\rtxvideo_cuda.dll";
    HMODULE m = LoadLibraryExW(dll.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!m)
    {
        LOG("native: %s not loadable (err %lu)\n", wideToUtf8(dll).c_str(), GetLastError());
        return false;
    }
    g_rtxb.setModelPath = (PFN_rtxvSetModelPath)GetProcAddress(m, "rtxv_set_model_path");
    g_rtxb.create = (PFN_rtxCreate)GetProcAddress(m, "rtx_video_api_cuda_create");
    g_rtxb.evalThdr = (PFN_rtxEvalThdr)GetProcAddress(m, "rtx_video_api_cuda_evaluate_thdr_deviceptr");
    g_rtxb.evalVsr = (PFN_rtxEvalVsr)GetProcAddress(m, "rtx_video_api_cuda_evaluate_vsr_deviceptr");
    g_rtxb.shutdown = (PFN_rtxShutdown)GetProcAddress(m, "rtx_video_api_cuda_shutdown");
    if (!g_rtxb.setModelPath || !g_rtxb.create || !g_rtxb.evalThdr || !g_rtxb.evalVsr || !g_rtxb.shutdown)
    {
        LOG("native: %s is missing an entry point\n", wideToUtf8(dll).c_str());
        FreeLibrary(m);
        g_rtxb.setModelPath = nullptr;
        g_rtxb.create = nullptr;
        g_rtxb.evalThdr = nullptr;
        g_rtxb.evalVsr = nullptr;
        g_rtxb.shutdown = nullptr;
        return false;
    }
    g_rtxb.evalVsrAsync = (PFN_rtxEvalVsrAsync)GetProcAddress(m, "rtx_video_api_cuda_evaluate_vsr_deviceptr_async");
    g_rtxb.mod = m;
    g_rtxb.dir = dir;
    g_rtxb.setModelPath(dir.c_str()); // NGX APP_PATH, must precede create
    LOG("native: RTX Video bridge %s\n", wideToUtf8(dll).c_str());
    return true;
}

// Offline TrueHDR's light statistics (MaxCLL, MaxFALL, the Dolby Vision L1 triples, the HDR10+
// records), accumulated per output
// frame from k_pqOut's block (hist[1024] u32, misc[4] u32, vSum int64 2^24 fixed point) and written as one
// JSON file the render reads at the finalize: maxcll / maxfall always, the Dolby Vision L1
// triples and the HDR10+ records when asked for (--hdr-dv / --hdr-hp).
static const size_t kThdrStatsBytes = 1024 * 4 + 4 * 4 + 8;
struct ThdrL1
{
    int v[3];
};
struct ThdrHp
{
    int avg;
    int maxscl[3];
    int dist[9];
};
struct ThdrAcc
{
    float lut[1024]{};  // the light of every PQ code (k_pqLut, the kernel's own PQ EOTF)
    int brightCode = 0; // HDR10+: the first code brighter than 100 nits
    double cll = 0.0, fall = 0.0;
    bool raw = false; // --hdr-color raw: the stats read nits, not linear
    bool wantL1 = false, wantHp = false;
    std::vector<ThdrL1> l1;
    std::vector<ThdrHp> hp;
    // --hdr-frames: one line per output frame, appended and flushed as the
    // record lands, "<frame MaxCLL> <frame MaxFALL>[ L <l1 x3>][ H <avg> <maxscl x3> <dist x9>]":
    // a killed render keeps every record of the frames it wrote, and a resumed render reads the
    // banked prefix back from it instead of decoding the banked video
    FILE* lines = nullptr;
    double lastCll = 0.0, lastFall = 0.0;
    bool haveLast = false;
    ThdrAcc() = default;
    ThdrAcc(const ThdrAcc&) = delete;
    ThdrAcc& operator=(const ThdrAcc&) = delete;
    ~ThdrAcc()
    {
        if (lines)
            fclose(lines);
    }

    void writeLine(double c, double f)
    {
        lastCll = c;
        lastFall = f;
        haveLast = true;
        if (!lines)
            return;
        fprintf(lines, "%.17g %.17g", c, f);
        if (wantL1 && !l1.empty())
            fprintf(lines, " L %d %d %d", l1.back().v[0], l1.back().v[1], l1.back().v[2]);
        if (wantHp && !hp.empty())
        {
            const ThdrHp& r = hp.back();
            fprintf(lines, " H %d %d %d %d %d %d %d %d %d %d %d %d %d", r.avg, r.maxscl[0], r.maxscl[1], r.maxscl[2],
                    r.dist[0], r.dist[1], r.dist[2], r.dist[3], r.dist[4], r.dist[5], r.dist[6], r.dist[7], r.dist[8]);
        }
        fprintf(lines, "\n");
        fflush(lines);
    }

    void add(const uint8_t* s, uint64_t n)
    {
        const uint32_t* hist = (const uint32_t*)s;
        const uint32_t* misc = hist + 1024;
        long long vFix = 0;
        memcpy(&vFix, s + 1024 * 4 + 16, 8);
        const double vSum = (double)vFix * (1.0 / 16777216.0);
        float vMax = 0.0f;
        memcpy(&vMax, &misc[3], 4);
        // the frame's largest value and its mean, x 10000 to nits (the mean rounded to fp32
        // first); --hdr-color raw reads nits already
        const double sc = raw ? 1.0 : 10000.0;
        const double fc = (double)vMax * sc;
        if (fc > cll)
            cll = fc;
        const double fa = (double)(float)(vSum / (double)n) * sc;
        if (fa > fall)
            fall = fa;
        if (wantL1)
        {
            // _accum_l1: [amin, mean, amax] of the maxRGB code, * 4095 / 1023, round, clamp
            int mn = -1, mx = 0;
            double sum = 0.0;
            for (int c = 0; c < 1024; c++)
                if (hist[c])
                {
                    if (mn < 0)
                        mn = c;
                    mx = c;
                    sum += (double)hist[c] * c;
                }
            if (mn < 0)
                mn = 0;
            const float k = (float)(4095.0 / 1023.0);
            const float st[3] = {(float)mn * k, (float)(sum / (double)n) * k, (float)mx * k};
            ThdrL1 r{};
            for (int i = 0; i < 3; i++)
            {
                float v = rintf(st[i]);
                r.v[i] = (int)(v < 0.0f ? 0.0f : (v > 4095.0f ? 4095.0f : v));
            }
            l1.push_back(r);
        }
        if (wantHp)
        {
            // _accum_hp: fp32 cumsum, searchsorted (the smallest code with cum >= q * n)
            static const float qs[8] = {0.01f, 0.9998f, 0.25f, 0.50f, 0.75f, 0.90f, 0.95f, 0.99f};
            float cum[1024];
            float acc = 0.0f;
            double lsum = 0.0;
            for (int c = 0; c < 1024; c++)
            {
                acc += (float)hist[c];
                cum[c] = acc;
                lsum += (double)((float)hist[c] * lut[c]);
            }
            const float nf = (float)n;
            int pv[8];
            for (int i = 0; i < 8; i++)
            {
                const float t = qs[i] * nf;
                int c = 0;
                while (c < 1024 && cum[c] < t)
                    c++;
                if (c > 1023)
                    c = 1023;
                pv[i] = (int)rintf(lut[c] * 100000.0f);
            }
            ThdrHp r{};
            for (int c = 0; c < 3; c++)
                r.maxscl[c] = (int)rintf(lut[misc[c] > 1023u ? 1023u : misc[c]] * 100000.0f);
            r.avg = (int)nearbyint((double)(float)lsum / (double)n * 100000.0);
            const double bright = (double)n - (brightCode > 0 ? (double)cum[brightCode - 1] : 0.0);
            const int d[9] = {pv[0], pv[1], (int)nearbyint(1000.0 * bright / (double)n), pv[2], pv[3], pv[4], pv[5],
                              pv[6], pv[7]};
            memcpy(r.dist, d, sizeof(d));
            hp.push_back(r);
        }
        writeLine(fc, fa);
    }

    // a held slot that carries the previous frame's finished bytes (DLSS 5 on): the
    // same picture, so the same record (MaxCLL / MaxFALL are unchanged by a repeat)
    void repeat()
    {
        if (wantL1 && !l1.empty())
            l1.push_back(l1.back());
        if (wantHp && !hp.empty())
            hp.push_back(hp.back());
        if (haveLast)
            writeLine(lastCll, lastFall);
    }

    bool writeJson(const std::wstring& path) const
    {
        FILE* f = nullptr;
        if (_wfopen_s(&f, path.c_str(), L"wb") || !f)
            return false;
        const int mc = cll > 65535.0 ? 65535 : (int)ceil(cll);
        const int mf = fall > 65535.0 ? 65535 : (int)ceil(fall);
        fprintf(f, "{\"maxcll\": %d, \"maxfall\": %d, \"frames\": %zu", mc, mf, wantL1 ? l1.size() : hp.size());
        if (wantL1)
        {
            fprintf(f, ", \"l1\": [");
            for (size_t i = 0; i < l1.size(); i++)
                fprintf(f, "%s[%d, %d, %d]", i ? ", " : "", l1[i].v[0], l1[i].v[1], l1[i].v[2]);
            fprintf(f, "]");
        }
        if (wantHp)
        {
            fprintf(f, ", \"hp\": [");
            for (size_t i = 0; i < hp.size(); i++)
            {
                const ThdrHp& r = hp[i];
                fprintf(f, "%s{\"avg\": %d, \"maxscl\": [%d, %d, %d], \"dist\": [%d, %d, %d, %d, %d, %d, %d, %d, %d]}",
                        i ? ", " : "", r.avg, r.maxscl[0], r.maxscl[1], r.maxscl[2], r.dist[0], r.dist[1], r.dist[2],
                        r.dist[3], r.dist[4], r.dist[5], r.dist[6], r.dist[7], r.dist[8]);
            }
            fprintf(f, "]");
        }
        fprintf(f, "}\n");
        const bool ok = !ferror(f);
        fclose(f);
        return ok;
    }
};

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

#define NCHK(call, what)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        const cudaError_t _e = (call);                                                                                 \
        if (_e != cudaSuccess)                                                                                         \
        {                                                                                                              \
            LOG("native: %s failed (cuda %d)\n", what, (int)_e);                                                       \
            return false;                                                                                              \
        }                                                                                                              \
    } while (0)

// gmflow A's outputs (the transformer on the host's kernels, nativeGaRun): qf (the transformer's fp16 stream as it is:
// q and feat = rows 0..1, f = rows 2..3) feeds k_localCorr (which computes the window positions itself) and B, flow0
// goes to B; B takes both beside the correlation
constexpr int kGmAn = 2;
// the quarter-scale local correlation's window (gmflow's radius 4: 9 x 9 taps), k_localCorr's K and gmflow B's
constexpr int kGmCorrK = 81;

// the midpoint tree of Smooth Motion and AMD FSR 4 (one bridge instance a level, the bridges hold 16): its deepest
// level (x512) and its node grid, kFrGrid steps a pair. The ring's memory fit prices the level the slots reach, so a
// GPU without the room for a level never reaches it
constexpr int kFrDepth = 9, kFrGrid = 1 << kFrDepth;
static_assert(kFrDepth <= 9, "a slot's fraction below 0.999f is a tween: x1024's last node (1023 / 1024) is not");
// the groups at a pipelined spell's start whose prices over the direct call before it give the pipelined route's ratios
constexpr int kFrPipeLearn = 8;

// RIFE's feature planes: the encode's output and the IFNet / block0 inputs f0 / f1 (RIFE 4.25: 4)
constexpr int kRifeFeatC = 4;

// a kernel chain on the host's kernels replayed as CUDA graphs (chainGraphRun): up to kN executable graphs, each
// captured on the first call with its key (the chain's buffers, then its w and h), replaced oldest first; mode 0 =
// direct launches, 1 = graphs, 2 = every capture thrown away (the fallback's trigger); logged = the graph line written;
// captures past kMaxCaptures (a chain whose buffers keep changing) = direct launches
struct ChainGraphs
{
    static constexpr int kN = 16, kKey = 10, kMaxCaptures = 4 * kN;
    struct Entry
    {
        const void* key[kKey] = {};
        cudaGraphExec_t exec = nullptr;
    } e[kN];
    int next = 0, mode = 0, captures = 0;
    bool logged = false;
};

// one launch of GMFSS's fusionnet on the host's kernels (fuPlan's step at the same index): its kernel (kFuKern), the
// channels, the PReLU slope before the conv (0 = none), the offsets of its fragment-order weights and fp32 bias in the
// weight block (fuNatLoad)
struct FuOp
{
    int k = 0, cin = 0, cout = 0;
    float slope = 0.0f;
    size_t wOff = 0, bOff = 0;
};

struct NativeRife
{
    // ---- handoff facts (nativeHandoff)
    std::string ifnetPath;
    int ph = 0, pw = 0, w = 0, h = 0, cw = 0, ch = 0;
    int dw = 0, dh = 0, x0 = 0, y0 = 0;
    int batchMax = 1;
    bool identity = false;
    bool hdr = false; // FP16 scRGB capture in, PQ R10A2 ring out
    int dev = 0;      // the CUDA device of the capture adapter (bound per thread)
    // live RTX TrueHDR through the CUDA bridge, once per real frame at capture size
    bool rtxHdr = false;    // the session asked for it and the bridge loaded
    bool rtxFailed = false; // a mid-run eval failed: faithful PQ for the rest of the run
    RtxThdrSetting thdr{};  // Contrast, Saturation, MiddleGray 50, MaxLuminance (thdrPeakNits)
    float sdrScale = 3.0f;  // SDR reference white as an scRGB scale (nits / 80)
    int rtxMode = 0;        // 0 vivid, 1 rtx, 2 raw
    float rtxVib = 0.0f, rtxSb = 0.0f;
    double thdrMs = 0.0, thdrMaxMs = 0.0; // host-side eval cost, reported at teardown
    uint64_t thdrN = 0;
    // live Sharpen and RTX VSR, the app's --sharpen / --rtx-vsr on this route
    float sharpen = 0.0f;   // Adaptive Sharpen strength 0..2 (curve_height) at the presented size (0 = off)
    bool vsrWant = false;   // --rtx-vsr on an SDR session with the bridge and its DLL present
    bool vsr = false;       // ...and the fit enlarges in both axes (decided after the handoff)
    bool vsrFailed = false; // a mid-run eval failed: Lanczos for the rest of the run
    RtxVsrSetting vsrSet{4};
    double vsrMs = 0.0, vsrMaxMs = 0.0;
    bool vsrAsync = false; // the evals ran through the bridge's event-ordered export (vsrMs not timed)
    uint64_t vsrN = 0;
    // the shrinking fit: the Lanczos pair (k_fitAaH / V) into the staging frame
    bool fitAa = false; // the fit to (dw, dh) changes the size: the separable Lanczos pair
    // Upscale to: the app's --upscale H = the
    // INTERNAL render size (uw, uh), derived from the exe's own flag. The
    // model frame goes there first (RTX VSR when it enlarges, else Lanczos), then the fit to
    // (dw, dh) like any frame. 0 = off (also when it equals the
    // fit rect: one resize, not two).
    int uw = 0, uh = 0;
    bool upAa = false; // the first resize shrinks the height (uh < h)
    // live Restore: the handoff names the bundled Real-ESRGAN or a --restore-model in AnimeJaNai's
    // SPAN-F3 form, both on the host's kernels; the host runs it on every presented frame before the
    // upscale, then folds the s x output to the first resize target (the
    // rule: back to the model size when RTX VSR follows, else straight to the internal
    // render size or the fit rect). A mid-run failure drops the pass for the rest of the session with
    // one line.
    std::string restorePath;
    bool restore = false; // the handoff named a Restore model (the session asked for it)
    // no-engine mode: the effects-only route (the
    // app's `echo` backend, no model ticked) runs here too. The handoff answers with the
    // geometry only (`engine=none`, plus the restore engine when asked for): no IFNet, no
    // encode, no tween; every group stores its one real frame through the same effects
    // chain. The cubin cache folder comes
    // from `NATIVE-PATH cache=` since no jit path names it.
    bool noEngine = false;
    std::string cachePath;
    // GMFSS: the handoff answers `engine=gmfss` with the half size and the kernel cache; this host runs the whole chain
    // on its own kernels (feat_ext, metricnet, the GMFSS IFNet, fusionnet, gmflow's backbone, A, `_m` and B,
    // nativeGmfssPair / nativeGmfssTween with the glue kernels above): no engine, so gmfss is a native backend like rife.
    bool gmfss = false;
    int hh = 0, hw = 0; // the half frame (the fusion grid)
    // the one shared activation block, gmCtxMemB bytes: feat_ext's, metricnet's, the GMFSS IFNet's, fusionnet's and
    // gmflow's scratch + Restore's (nativeGmShareMemory) or
    // the IFNet's, the encode's, Restore's and DRBA block0's scratch (nativeModelShareMemory)
    void* dGmCtxMem = nullptr;
    int64_t gmCtxMemB = 0;
    // between gmflow's halves: A's outputs (kGmA order), the correlation k_localCorr writes for B, and the shapes it
    // runs at (the quarter grid gmH x gmW, gmK window samples a pixel)
    void* dGmA[kGmAn] = {};
    // A's inputs map4 / map8 [2, 128, ...] in the backbone's dtype: batch slot 0 = the previous frame, 1 = the new one;
    // gmMapSlot = one frame's bytes of each; gmMapsNext = slot 1 holds the frame the next call reads as its previous
    // one, taken on view gmMapsKind (0 the pictures, 1 the SDR view, 2 the motion view)
    void* dGmMap[2] = {};
    size_t gmMapSlot[2] = {};
    bool gmMapsNext = false;
    int gmMapsKind = 0;
    float* dGmCorr = nullptr;
    int gmH = 0, gmW = 0, gmK = 0;
    // between `_m` and A: its outputs mf / mq / mk (fp16 [2, N, 128] each), the matching's flow (fp32 [2, N, 2]) and
    // A's prop input; N = gmN8 tokens on the eighth grid gmW8 wide, gmPropHalf = prop is fp16
    void* dGmM[3] = {};
    float* dGmFlow8 = nullptr;
    void* dGmProp = nullptr;
    int gmN8 = 0, gmW8 = 0;
    bool gmPropHalf = false;
    // the chain's contract, read off the engines in nativeGmfssSetup (never from the handoff
    // line, the 5a rule), and the planar NCHW buffers it sizes from that. Levels: the half
    // (hh, hw), the quarter and the eighth, exactly GMFSS_infer_u's feature pyramid.
    int gmC[3] = {}; // feat_ext channels at the half / quarter / eighth
    bool gmFeatHalf = true, gmMetricHalf = true, gmOutHalf = true, gmFuseHalf = true;
    int gmCur = 0;             // which dGmFeat set holds the CURRENT frame's features
    void* dGmFeat[2][3] = {};  // the two frames' feature sets (cur is the next pair's prev)
    float* dGmHalf = nullptr;  // (6, hh, hw): img0's half in planes 0..2, img1's in 3..5
    float* dGmFlow = nullptr;  // the gmflow output (2, 2, hh, hw) = flow01 then flow10
    void* dGmMetric = nullptr; // (2, hh, hw) in the engine's dtype: m0 then m1
    float* dGmFlowP[2] = {};   // the flow pyramids at the quarter / eighth, pre-scaled
    float* dGmMetP[2] = {};    // the metric pyramids at the quarter / eighth (fp32)
    // the scatter splats' accumulator, max(C) + 1 planes of the half; with gmTail it also holds the cut fusionnet's
    // out (256 fp16 planes, free from the splats' end to the next clear); the gather splats never use it: then only
    // that out (64 planes), or null
    long long* dGmAcc = nullptr;
    // the pair scratch (dGmA, dGmCorr, dGmM, dGmFlow8, dGmProp, dGmEnc, dGmHalfM) carved from dGmAcc, not allocated
    bool gmScratchInAcc = false;
    // the gather splats (k_gs*): two list sets (a level's two directions), each the counts / fill cursor, the list
    // offsets, the entries and the sources' e / dx / dy planes, sized for the half; gmGather false (SMV_GM_SPLAT=old,
    // or a half above the scan's 2048 x 4096 cells) = k_splatSoft + k_splatNorm on dGmAcc
    int* dGsCnt[2] = {};
    int* dGsOff[2] = {};
    int* dGsList[2] = {};
    float* dGsSrc[2] = {};
    int* dGsBlk[2] = {};
    bool gmGather = false;
    bool gmTail = false;    // the cut fusionnet (out = the pre-shuffle tensor): k_fuTail does shuffle + conv_last
    bool gmTailHwc = false; // gmTail with out channels-last (the engine's HWC8 / HWC16): k_fuTailNhwc reads it
    bool gmFeatHwc = false; // feat_ext's levels and fusionnet's b / c / d channels-last: the feature splats' layout
    // feat_ext on the host's kernels (nativeFnModule): its module and six kernels, the weight block (fnOff = per conv
    // 1..5 its fragment-order weights, then its fp32 bias), the six LeakyRelu slopes, the scratch (in the shared
    // activation block, else dFnOwn)
    CUmodule fnMod = nullptr;
    CUfunction fFn[6] = {};
    void* dFnW = nullptr;
    size_t fnOff[10] = {};
    float fnSlope[6] = {};
    void* dFnScr = nullptr;
    void* dFnOwn = nullptr;
    // metricnet on the host's kernels (nativeMnModule): its module and five kernels, the weight block (mnOff, see
    // mnNatLoad), the four PReLU slopes, the scratch (in the shared activation block, else dMnOwn)
    CUmodule mnMod = nullptr;
    CUfunction fMn[5] = {};
    void* dMnW = nullptr;
    size_t mnOff[10] = {};
    float mnSlope[4] = {};
    void* dMnOwn = nullptr;
    // the GMFSS IFNet on the host's kernels (nativeGiModule): its module and kernels, the weight block (giOff, see
    // giNatLoad), the scratch (in the shared activation block, else dGiOwn)
    CUmodule giMod = nullptr;
    CUfunction fGi[11] = {};
    void* dGiW = nullptr;
    std::vector<size_t> giOff;
    void* dGiOwn = nullptr;
    ChainGraphs giGraphs; // its launches as CUDA graphs (nativeGiRun; SMV_GI_GRAPH)
    // fusionnet on the host's kernels (nativeFuModule): its module and kernels, one FuOp a launch, the weight block, the
    // activations (fuArena; in the shared activation block, else dFuOwn)
    CUmodule fuMod = nullptr;
    CUfunction fFu[11] = {};
    int fuR4Min = 0; // the 8-row grid's CTAs from which a PReLU wide conv runs as its 4-row twin (8 an SM; 0 = never)
    void* dFuW = nullptr;
    std::vector<FuOp> fuOps;
    void* dFuOwn = nullptr;
    ChainGraphs fuGraphs; // its launches as CUDA graphs (nativeFuRun; SMV_FU_GRAPH)
    // gmflow's second half B on the host's kernels (nativeGbModule): its module and kernels, the weight block (gbOff, see
    // gbNatLoad), the activations (gbArena; in the shared activation block, else dGbOwn)
    CUmodule gbMod = nullptr;
    CUfunction fGb[6] = {};
    void* dGbW = nullptr;
    size_t gbOff[6] = {};
    void* dGbOwn = nullptr;
    ChainGraphs gbGraphs; // its launches as CUDA graphs (nativeGbRun; SMV_GB_GRAPH)
    // gmflow's backbone on the host's kernels (nativeBbModule): its module and kernels, the weight block (bbOff, see
    // bbNatLoad), the activations (bbArena; in the shared activation block, else dBbOwn)
    CUmodule bbMod = nullptr;
    CUfunction fBb[16] = {};
    void* dBbW = nullptr;
    size_t bbOff[22] = {};
    void* dBbOwn = nullptr;
    ChainGraphs bbGraphs; // its launches as CUDA graphs (nativeBbRun; SMV_BB_GRAPH)
    // gmflow's transformer at both scales on the host's kernels (nativeGaModule): its module and kernels, the weight
    // block (gaOff, see gaNatLoad), the window position tables (dGaPos: A's posY | posX, then `_m`'s), the activations
    // (gaArena; in the shared activation block, else dGaOwn)
    CUmodule gaMod = nullptr;
    CUfunction fGa[7] = {};
    int gaMlp2Min = 0;    // the CTAs from which the mlp runs as k_gaMlp2 (3 an SM; 0 = never: RTX 20, SMV_GA_MLP2=0)
    bool gaAttn2 = false; // gmflow A's attention as k_gaAttn2 (never on RTX 20; SMV_GA_ATTN2=0 = never)
    void* dGaW = nullptr;
    size_t gaOff[46] = {};
    float* dGaPos = nullptr;
    void* dGaOwn = nullptr;
    ChainGraphs gaGraphs[2]; // `_m` (0) and A (1) as CUDA graphs (nativeGaRun; SMV_GA_GRAPH)
    void* dGmWl = nullptr;   // conv_last's weight (3, 64, 3, 3) fp16 (fusionnet.pkl, k_fuTailNhwc reads it)
    void* dGmBl = nullptr;   // its bias (3) fp16
    float gmLr[3] = {-1.0f, -1.0f,
                     -1.0f}; // the heads' first PReLU slopes on b / c / d: the feature splats' stores apply them
    float* dGmFa = nullptr;  // fusionnet a (9, half): I1t, merged (ifnet writes it), I2t
    void* dGmFb = nullptr;   // fusionnet b (2 * gmC[0], half), channels-last fp16
    void* dGmFc = nullptr;   // fusionnet c (2 * gmC[1], quarter), the same
    void* dGmFd = nullptr;   // fusionnet d (2 * gmC[2], eighth), the same
    float* dGmT = nullptr;   // the timestep, one device float (ifnet's 1x1x1x1 input)
    void* dGmOut = nullptr;  // the full graph's fusionnet output (3, ph, pw) in the engine's dtype (gmTail: null, the
                             // tail writes dGmF)
    float* dGmF = nullptr;   // that output clamped to fp32: what storeSlot consumes
    // HDR planes (nativeGmfssPair): the SDR view of one padded frame (hdrToModel, the scratch), the motion view's
    // halves gmflow and metricnet read on a pair outside the SDR range, the range flag (device, pinned host copy);
    // per feature slot: its frame stays inside the SDR range, its features came from the SDR view; the pair's mode
    float* dGmEnc = nullptr;
    float* dGmHalfM = nullptr;
    int* dGmRange = nullptr;
    int* hGmRange = nullptr;
    bool gmSdrPairs = true; // SMV_GMFSS_SDR_PAIRS=0: every pair the way outside the SDR range
    bool gmFrameSdr[2] = {}, gmFeatSdr[2] = {}, gmPairSdr = false;
    // gmFeatOk[s] = dGmFeat[s] holds its frame's features: a call without tweens skips feat_ext (gmFeatLazy;
    // SMV_GM_FEAT_LAZY=0 = never) and the next pair that makes tweens takes the previous frame's again (the refeat
    // branch). gmCatchUp = the last pair ran such a catch-up the eager route would not have (its span between
    // gmCatchEv[0] and [1], live only), which the throttle's price takes out of that group's pre
    bool gmFeatOk[2] = {}, gmFeatLazy = true, gmCatchUp = false;
    cudaEvent_t gmCatchEv[2] = {};
    uint32_t gmFeatSkips = 0, gmCatchUps = 0;
    int gmModeLogs = 0; // the mode lines this session printed (the first pair's and up to 3 switches)
    // SMV_LIVE_GMFSS_PROF=1: per-phase CUDA-event breakdown of the chain, printed every 32
    // groups (nine spans, the tween ones
    // measured on the group's FIRST tween). The pair span splits into half | flow | metric |
    // pyr, with the CPU (QPC) enqueue spans beside the GPU ones: the pair
    // block, the gmflow enqueue inside it, and every tween. The exe runs a group ahead of the
    // GPU, so those CPU numbers are launch cost, not the GPU pace.
    bool gmProf = false, gmProfTween = false;
    cudaEvent_t gmEv[11] = {};
    double gmAcc[9] = {};
    uint32_t gmProfN = 0;
    double gmCpuPair = 0.0, gmCpuFlow = 0.0, gmCpuTween = 0.0;
    uint32_t gmCpuPairN = 0, gmCpuTweenN = 0;
    // NVIDIA Smooth Motion (fruc): the handoff answers
    // `engine=fruc` with the /64 padded geometry and `NATIVE-PATH fruc=` (the folder holding the
    // shipped nvoffruc_bridge.dll beside the user-installed NvOFFRUC.dll + cudart64_110.dll);
    // this host drives that bridge's flat C API:
    // the (R, G, B) model planes packed to true BGRA8 at pw x ph (k_packBgraRgb),
    // one bridge call per tween, the BGRA8 result unpacked (k_unpackBgraRgb).
    bool fruc = false;
    bool dlssg = false;           // offline DLSS 4.5: no IFNet, a child server
    std::string frucDir;          // the bridge folder the handoff named
    uint8_t* dFrSurf[3] = {};     // BGRA8 pw x ph: the packed frames, rotated per group
    uint8_t* dFrOutB = nullptr;   // BGRA8 pw x ph: the bridge's output
    float* dFrOut = nullptr;      // (3, ph, pw): the tween, the layout storeSlot reads
    int frPrev = -1, frLast = -1; // surfaces: the previous frame, the last tweened pair's end
    int frA = -1, frB = -1;       // this group's pair
    // the feed-once bridge (nvoffruc_step), per FRUC instance: the surface whose frame it was fed
    // last (-1 = none, or that surface was repacked since), whether the pair's next tween on it is
    // its first, and how many calls it has served
    int frFed[kFrDepth] = {}; // -1 each from nativeFrucSetup on
    bool frFirst[kFrDepth] = {};
    uint64_t frCalls[kFrDepth] = {};
    bool frCreated = false;
    uint64_t frPrimed = 0, frRepeats = 0, frTweens = 0;
    // parallel instances (the direct-t scheme, bridge nvoffruc_step_i): tween k of a pair's plan
    // runs on instance k % frInst, a round of frInst tweens at a time; instance 0 on the compute
    // thread, the others on their own workers (g_frW), each into its own BGRA8 output
    int frInst = 1, frInstMax = 1;
    uint8_t* dFrOutBI[kFrDepth] = {}; // [0] = dFrOutB
    double frPlan[64] = {};
    uint32_t frPlanN = 0, frPlanK = 0;
    // recursive midpoints (the default): NvOFFRUC damages content that does not move at every
    // t != 0.5, even called exactly as NVIDIA documents it, so every tween is node k / 2^L of the
    // pair's midpoint tree:
    // FRUC at t = 0.5 between the node's two parents, level L on instance L - 1, which then sees
    // one continuous stream of new frames (NVIDIA's one call per new frame). A t that is no node
    // takes the nearest node of depth frMpCap. SMV_FRUC_MIDPOINTS=0 = the direct-t scheme above.
    bool frMp = false;
    int frMpCap = 2;                // off-node snap depth: offline 3, live 2 (cost); SMV_FRUC_DEPTH=1..9 sets both
    int frMpL = 0;                  // this pair's depth, 0 = not chosen yet
    bool frMpBuilt = false;         // this pair's whole tree to frMpL is computed
    uint64_t frSerial = 0;          // frames packed this session: the node keys
    uint8_t* dFrNode[kFrGrid] = {}; // BGRA8 pw x ph: node j / kFrGrid of this pair (j = 1..kFrGrid - 1)
    bool frNodeOk[kFrGrid] = {};
    uint64_t frLastKey[kFrDepth] = {}; // per instance: the key of the frame it was fed last, 0 = none
    uint64_t frNodeCalls = 0;
    // pipelined across pairs (live Smooth Motion, the exe's route bit on a group of x4 or deeper): a group shows the
    // PREVIOUS pair, whose levels 1..k instance 0 made during the previous group on its own worker (g_frW[0], k = the
    // depth less one: nativeFrucPipeSpec) while the compute thread ran the deeper levels of the pair before; every
    // instance still sees its frames in time order. frLag* = the pair shown (its serial, surfaces, SDR view, nodes,
    // held = identical or never split); frKeep = the surface the next pack spares (the shown pair's first frame);
    // frPipeDepth = the depth the last pipelined group's tweens asked (the worker's k)
    bool frPipePrev = false, frLagHave = false, frLagSdr = false, frLagStatic = false, frCurStatic = false;
    bool frSpecRun = false, frRepacked = false;
    int frLagA = -1, frLagB = -1, frKeep = -1, frLagL = 0, frPipeDepth = 0;
    uint64_t frLagSerial = 0, frPipeGroups = 0;
    uint8_t* dFrLagNode[kFrGrid] = {};
    bool frLagOk[kFrGrid] = {};
    // this group's midpoint-tree calls on the compute thread (Smooth Motion's bridge or AMD FSR 4's: calls, their ms,
    // the calls and ms of those made before the slots = FSR 4's level 1, and the chain inside the slots incl. a
    // pipelined group's worker join) for the exe's price (gFrCallUs / gFrStoreUs); a pipelined group's route
    // (grPiped) and instance 0's worker's calls + ms (grFrWk*) for the pipelined prices (gFrPipeMainUs / gFrPipeWkUs)
    uint32_t grFrCalls = 0, grFrPreCalls = 0, grFrWkCalls = 0;
    double grFrCallMs = 0.0, grFrPreMs = 0.0, grFrChainMs = 0.0, gFrCallMs = 0.0, gFrStoreMs = 0.0, grFrWkMs = 0.0;
    double gFrPipeMainMs = 0.0, gFrPipeWkMs = 0.0, gFrPipeRm = 0.0, gFrPipeRw = 0.0, gFrCallEntryMs = 0.0;
    int grPipeLearn = 0; // the groups left of a pipelined spell's start that learn gFrPipeRm / gFrPipeRw
    bool grPiped = false;
    // groups whose midpoint-tree price is not sampled: one that made instances (their creation and first, cold calls
    // land in its time: a first x4 group billed AMD FSR 4's call at 20.3 ms at 854x480) and the next
    int grPriceSkip = 0;
    // HDR planes (encPost): a pair whose two frames stay inside the SDR range goes to FRUC as their SDR view
    // (k_packBgraSdr) and its tweens come back through k_unpackBgraSdr; any other pair as the 8-bit HDR codes. Per
    // surface: its frame inside the range, whether it holds the SDR view. SMV_FRUC_SDR_PAIRS=0 = every pair as the codes
    bool frSdrOn = false, frPairSdr = false;
    bool frIn[3] = {}, frSurfSdr[3] = {};
    int* dFrRange = nullptr;
    int* hFrRange = nullptr;
    int frModeLogs = 0;
    // AMD FSR frame generation (fsrfg): the FRUC route above (fruc is set too: its ladders, its midpoint tree, its
    // surfaces) with our smv_fsrfg_bridge.dll (AMD's FSR 4 ML frame generation on vkd3d-proton) in place of NvOFFRUC.
    // The handoff answers `engine=fsrfg` with the unpadded geometry and `NATIVE-PATH fsrfg=` (the bridge folder). Every
    // frame (real or a node) is RGBA16F at the bridge's row pitch in dFrSurf / dFrNode; a call feeds instance i one frame
    // and its vectors (NVIDIA optical flow, frame -> the instance's last frame, px) and returns the frame halfway between
    // them. The bridge's three textures (frame, vectors, generated frame) are Vulkan exports CUDA maps as arrays, its fence
    // a Vulkan timeline semaphore: the stream copies in, signals an odd value, the bridge's queue waits, generates,
    // signals the next even value, the stream waits for it and copies out. Instance 0 is fed every real frame in order
    // (FSR is built for one continuous stream); level L of a pair's tree on instance L - 1, as FRUC's. frLastKey 0 =
    // never fed (its first call resets). Intel XeSS-FG (--xess-fg) runs on this route through its own bridge (fgKind 1:
    // smv_xefg_bridge.dll with Intel's libxess_fg.dll + libxell.dll in engine\xess): D3D12 textures and a D3D12 fence
    // instead of the Vulkan exports, the same calls.
    bool fsrfg = false;
    int fgKind =
        0; // the bridge on this route: 0 = AMD FSR 4 (smv_fsrfg_bridge.dll), 1 = Intel XeSS-FG (smv_xefg_bridge.dll)
    struct FgInst
    {
        cudaExternalMemory_t emIn = nullptr, emMv = nullptr, emOut = nullptr;
        cudaMipmappedArray_t mmIn = nullptr, mmMv = nullptr, mmOut = nullptr;
        cudaArray_t in = nullptr, mv = nullptr, out = nullptr; // the three textures' level 0
        cudaExternalSemaphore_t sem = nullptr;
        uint64_t val = 0; // the fence's last value
        bool made = false;
    };
    FgInst fg[kFrDepth];
    LUID fgLuid = {};
    int fgPitch = 0, fgMvPitch = 0; // the bridge's row pitches (bytes), the same for every instance
    uint8_t* dFgMv = nullptr;       // a call's vectors (RG16F rows fgMvPitch apart) before their copy into its texture
    // SMV_FSRFG_PROF=1: CUDA events between a call's stages (copy in, the Optical Flow inputs, the Optical Flow, the
    // vector kernels, the vector copy, FSR on vkd3d, the copy out), means at the session's end; every call then waits
    // for its last event, so the stages do not overlap the next call's. A derived call's k_mvDerive counts apart
    // (fgDeriveMs over fgProfDerN), the three Optical Flow stages over the calls that ran a pass (fgProfVecN)
    bool fgProf = false;
    cudaEvent_t fgEv[8] = {};
    double fgStageMs[7] = {}, fgCpuMs = 0.0, fgDeriveMs = 0.0;
    uint64_t fgProfN = 0, fgProfVecN = 0, fgProfDerN = 0;
    // FSR's motion vectors (frame generation and upscaling): an Optical Flow session at w x h, ABGR8 (R, G, B, A
    // bytes), one direction, fast, its output grid: the field from slot `cur` to slot `prev` = the current frame -> its
    // predecessor (FSR's convention), in px: two planes (FSR reads no cost: no cost buffers, k_nvofUp writes none)
    struct OfVec
    {
        NvOFHandle of = nullptr;
        NvOFGPUBufferHandle in[2] = {}, out[2] = {};
        CUdeviceptr inP[2] = {}, outP[2] = {};
        uint32_t inPitch = 0, outPitch[2] = {};
        int w = 0, h = 0, grid = 1, gw = 0, gh = 0;
        float* dFlow = nullptr;
        bool both = false; // SMV_OFVEC_BOTH=1: both directions, the backward field read (else forward, frames swapped)
        NV_OF_PERF_LEVEL level = NV_OF_PERF_LEVEL_SLOW; // nativeOfLevel()
        cudaStream_t st = nullptr; // the stream its Optical Flow and kernels run on, null = the session's
        int wr = 1; // one direction: the field goes to out[wr] (FSR 4's pipelined route alternates out[0] / out[1])
        // FSR 4's level-1 chain (nativeFsrfgOfIn): the input slot still holding frame surface keepSrc as the last pass
        // converted it, -1 = none; a pack into that surface (nativeFsrfgPack) and every other pass clear it
        int keepSlot = -1;
        const uint8_t* keepSrc = nullptr;
    };
    OfVec fgOf;
    // a tree level with 2+ calls makes its vectors ahead (nativeFsrfgLevel): call j + 1's Optical Flow on fgOfSt (its own
    // session fgOf2, into dFgMvR[(j + 1) & 1]) while call j's FSR runs. fgAhead: 0 = not made yet, 1 = on, -1 = off
    // (SMV_FSRFG_OVERLAP=0, the profiler, or a failed setup: serial vectors)
    int fgAhead = 0;
    cudaStream_t fgOfSt = nullptr;
    OfVec fgOf2;
    uint8_t* dFgMvR[2] = {};
    float fgChildT0 = 0.0f, fgChildT1 = 0.0f; // a serial level-2+ call's two frames in its pair (k_mvDerive's t0, t1)
    cudaEvent_t fgEvLevel = nullptr, fgEvVec[2] = {}, fgEvFree[2] = {};
    // the level-1 chain's two streams: fgOfL1St runs fgOf's Optical Flow and its kernels (the session's stream waits
    // for them on the direct route), fgL1St on the pipelined route (SMV_FSRFG_PIPE, nativeFsrfgPipeAdvance) instance
    // 0's copies and whole call of the new pair, beside the shown pair's level-2+ calls on the session's stream, so the
    // next pair's Optical Flow runs beside instance 0's call; FSR's calls keep the serial route's order on the device
    // (instance 0's waits for fgEvDeep, the shown pair's last deeper call). fgOfWr / fgOfRd = the field fgOf writes /
    // the derived vectors read (out[0] / out[1] alternate by pair while pipelined, else 1); dFgMvL1 = the calls'
    // vectors on fgL1St, dFgL1Node = the new pair's level-1 node (swapped into dFrNode when that pair is shown);
    // fgEvL1 = fgL1St's last work, fgEvS = the session stream's point the two streams wait for, fgEvOf = fgOfL1St's
    // last pass, fgEvMv = fgL1St's last read of dFgMvL1 (the next pass writes it after), fgEvSurf[n] = frame surface
    // n's last read off the session's stream (a pack into it waits for it); fgPiped = the last group ran pipelined,
    // fgL1Pend = the new pair's call still to issue (nativeFsrfgL1Issue), fgL1Ok = its node is made
    cudaStream_t fgL1St = nullptr, fgOfL1St = nullptr;
    cudaEvent_t fgEvL1 = nullptr, fgEvS = nullptr, fgEvDeep = nullptr, fgEvOf = nullptr, fgEvMv = nullptr;
    cudaEvent_t fgEvSurf[3] = {};
    int fgOfWr = 1, fgOfRd = 1;
    uint8_t* dFgMvL1 = nullptr;
    uint8_t* dFgL1Node = nullptr;
    bool fgPiped = false, fgL1Pend = false, fgL1Ok = false, fgPipeLogged = false;
    // AMD FSR upscaling (fsrUp, `--fsr-upscale`): RTX VSR's resize (vsr is set too, so the slot rules, the staging and
    // nativeVsrEval's callers stay one path; nativeVsrEval hands dVsrIn to smv_fsrup_bridge.dll instead of NGX). The
    // bridge runs AMD's signed FidelityFX upscaler (FSR 4.1.1's INT8 model) on its own D3D12 device, frames in and out
    // through shared buffers + one fence like FSR FG's; every frame of the slot's stream is fed in order (the first
    // resets), its vectors from the previous one (fuOf, the two input slots alternate)
    bool fsrUp = false;
    bool fuHdr = false; // live HDR planes: the bridge in its HDR mode (linear light; off with RTX HDR before the model)
    std::string fsrUpDir;
    int fuKind =
        0; // the bridge on this route: 0 = AMD FSR (smv_fsrup_bridge.dll), 1 = Intel XeSS-SR (smv_xess_bridge.dll)
    cudaExternalMemory_t fuEmIn = nullptr, fuEmMv = nullptr, fuEmOut = nullptr;
    uint8_t* dFuIn = nullptr;
    uint8_t* dFuMv = nullptr;
    uint8_t* dFuOut = nullptr;
    cudaExternalSemaphore_t fuSem = nullptr;
    uint64_t fuVal = 0;
    int fuInPitch = 0, fuMvPitch = 0, fuOutPitch = 0;
    bool fuMade = false;
    int fuCur = 0;    // the vector session's slot that holds this frame
    uint64_t fuN = 0; // frames upscaled
    // past 3x a side (fuChainPlan): pass k = fuPw[k] x fuPh[k] -> fuPw[k + 1] x fuPh[k + 1] on bridge instance k; pass 0
    // = the fields above (its output dFu0Out when chained), passes 1 .. fuPasses - 1 = fuX; dFuOut = the last pass's
    // output, what the callers read; a pass's vectors = pass 0's scaled to its size (k_fuMvScale)
    static constexpr int kFuMaxPass = 6;
    struct FuPass
    {
        cudaExternalMemory_t emIn = nullptr, emMv = nullptr, emOut = nullptr;
        uint8_t* dIn = nullptr;
        uint8_t* dMv = nullptr;
        uint8_t* dOut = nullptr; // null on the last pass (its output = dFuOut)
        cudaExternalSemaphore_t sem = nullptr;
        uint64_t val = 0;
        int inPitch = 0, mvPitch = 0, outPitch = 0;
        bool made = false;
    };
    int fuPasses = 1;
    int fuPw[kFuMaxPass + 1] = {}, fuPh[kFuMaxPass + 1] = {};
    FuPass fuX[kFuMaxPass - 1];
    cudaExternalMemory_t fu0EmOut = nullptr;
    uint8_t* dFu0Out = nullptr;
    int fu0OutPitch = 0;
    // SMV_FSRUP_PROF=1: CUDA events between a frame's stages (the input pack, the Optical Flow, the vector kernels, FSR
    // on D3D12, the output unpack), means at the session's end; every frame then waits for its last event
    bool fuProf = false;
    cudaEvent_t fuEv[6] = {};
    double fuStageMs[5] = {};
    uint64_t fuProfN = 0, fuProfVecN = 0;
    char fuVersion[32] = {};
    OfVec fuOf;
    // an FSR / XeSS upscaler ticked mid-session (nativeUpSwitch): its first two frames run on a worker thread
    // (nativeFsrupWarmProc), so the driver's shader compile inside them (up to kFsrCompileMs on an exe location and
    // upscale ratio it has not seen) blocks the worker, not the native thread, while the session presents without the
    // upscaler. fuWarm: 1 = warming, 2 = warm, 3 = failed; nativeUpWarmEnd turns it on at the next group when the tick
    // still stands (fuWarmWant: an `up 0` meanwhile clears it)
    volatile LONG fuWarm = 0;
    HANDLE fuWarmTh = nullptr;
    bool fuWarmWant = false;
    ULONGLONG fuWarmT0 = 0;
    // ArtCNN (acnUp, `--artcnn`): RTX VSR's resize too (vsr is set), on planar floats like Anime4K: C4F32_DN v1.6.2's
    // luma x2 in its own module (acnMod, kAcnA + kAcnB), the weights from engine\artcnn\artcnn_c4f32_dn_v1.6.2_fp16.pkl,
    // the colour from the frame's Lanczos enlargement (nativeAcnRun); x2 passes as Anime4K's (a4kPassesFor)
    bool acnUp = false;
    CUmodule acnMod = nullptr;
    CUfunction fAcn[6] = {}; // k_acnBodyR, k_acnBodyA, k_acnFirst, k_acnLast, k_acnY, k_acnMerge
    void* dAcnWf = nullptr;  // conv1 .. conv5's fp16 B fragments, conv after conv (9 x 32 x 32 halves each)
    float* dAcnB = nullptr;  // conv1 .. conv5's fp32 biases, 32 each
    float* dAcnW0 = nullptr; // conv0's weights [32][9], then its 32 biases (fp32)
    float* dAcnWz = nullptr; // conv6's weights [4][32][9], then its 4 biases (fp32)
    void* dAcnAct = nullptr; // three activation buffers (channels-last fp16) of acnActN halves each
    size_t acnActN = 0;
    int acnCore = 0;                  // the tile core (0 = whole frames)
    int acnW = 0, acnH = 0, acnN = 1; // the first pass's input size, the x2 passes a frame
    int acnOw = 0;                    // the widest fit target dAcnTmp holds
    float* dAcnY = nullptr;           // the luma of a pass's input (the largest pass's size)
    float* dAcnY2 = nullptr;          // the net's luma (the largest pass's output)
    float* dAcnLz = nullptr;          // the pass's input Lanczos x2: three planes (the largest pass's output)
    float* dAcnOut = nullptr;         // a pass's RGB, three planes = the next pass's input
    float* dAcnTmp = nullptr;         // the Lanczos pairs' horizontal pass
    size_t acnTmpN = 0;               // floats dAcnTmp holds
    // Anime4K (a4kUp, `--anime4k`): RTX VSR's resize too (vsr is set), on planar floats instead of the bridge's BGRA:
    // Upscale CNN x2 UL's conv passes in CUDA (its own module a4kMod, a4kKernelText: layers 1..6 on the tensor cores,
    // layer 0 inside layer 1's kernel and the last 1 x 1 inside layer 6's, both with literal weights) at the slot's
    // input size, then the fold (k_a4kFoldHSub: the shader's Depth-to-Space pass, and the Lanczos pair from 2w x 2h to
    // the slot's target); HDR frames go in as their SDR view and come back through RTX VSR's way to the codes
    bool a4kUp = false;
    CUmodule a4kMod = nullptr;
    // the module build and the statistics buffers are shared by Restore (nativeModelInit) and the resize
    // (nativeRtxInit), which the live early init runs on two threads at once: one builds, the other finds it
    std::mutex mA4k;
    // k_a4kConvM16First, k_a4kConvM16, k_a4kConvM16Last (the Upscale net), k_a4kRestFirst, k_a4kRestLast (the Restore
    // net), k_a4kStatH, k_a4kStatV, k_a4kClamp (Clamp_Highlights)
    CUfunction fA4k[8] = {};
    int a4kW = 0, a4kH = 0;   // the first pass's input size (the slot's)
    int a4kN = 1;             // x2 passes a frame: the fewest whose 2^n reaches the slot's ratio, memory permitting
    bool a4kAA = false;       // Mode A+A (--anime4k-aa, memory permitting): Restore on the first pass's 2x frame
    int a4kOw = 0, a4kOh = 0; // the fold target dA4kTmp holds (the widest; a live A+A switch sets up again at it)
    void* dA4kIn = nullptr;   // NCHW fp16, the largest pass's input size
    void* dA4kT = nullptr;    // four layer slots, 3 textures x 4 fp16 a pixel each (nativeA4kRun), the largest pass's
    void* dA4kOut = nullptr;  // the three last textures as 12 fp16 planes, the largest pass's
    float* dA4kMid = nullptr; // a pass's exact 2x frame = the next pass's input, planar fp32 (a4kN > 1 only)
    float* dA4kTmp = nullptr; // the fold's horizontal pass: planar fp32, target width x 2 x the last pass's height
    // Clamp_Highlights, the shader set's de-ringing (part of every Anime4K mode): the 5 x 5 maximum of the luma, taken
    // before a frame's first Anime4K pass (Restore's input when Restore is Anime4K, else the resize's) and kept fp16 at
    // that size; the chain's last output is lowered where its luma exceeds the bilinear sample of that maximum
    unsigned short* dA4kStat = nullptr;    // the maximum, a4kStatW x a4kStatH
    unsigned short* dA4kStatTmp = nullptr; // its horizontal pass
    size_t a4kStatCap = 0;                 // pixels the two hold
    int a4kStatW = 0, a4kStatH = 0;
    bool a4kStatFresh = false; // this frame's Restore (Anime4K) took the maximum and leaves the clamp to the resize
    // Restore = Anime4K's Restore CNN UL (restNatKind 3, `--restore-anime4k`): the same kind of passes at the source
    // size (weights 6..12 of the module, four layer slots of its own), out as planar fp32 (the shader adds the source
    // back; clamped 0..1 like every Restore), then Restore's Lanczos pair when the target size differs
    void* dA4kRT = nullptr;    // four layer slots at the source size
    float* dA4kRMid = nullptr; // the restored frame, planar fp32 at the source size
    float* dA4kRTmp = nullptr; // the fold's horizontal pass (planar fp32, a4kRTmpW x the source height)
    int a4kRTmpW = 0;
    // RIFE with DRBA timing (rifedrba): the RIFE
    // handoff plus `engine=drba lag=1` (calc_flow's block0 runs on the host's kernels,
    // nativeB0Module): a four-frame history of padded frames and their
    // encodes, windows centred on a frame id (two kept, chained: a window's left flow is the
    // previous window's right one, reversed), the lag-1 group (every slot shows time (k-2) + f,
    // the real slot is frame k-1), one IFNet enqueue per tween with its own DRM timestep map
    // (the kernels k_drbaFlowSplat / FlowNorm / DrmSplat / DrmNorm above, gated against a float64
    // reference), the head fallback (plain pair RIFE on the lagged pair).
    // RIFE's two domains (mph set): the ring keeps each frame's motion frame beside the picture, and
    // the encodes, block0 and its flows, the windows, the DRM map and the IFNet's x live at the
    // motion frame (M x N = mph x mpw, else the pictures' ph x pw); k_rifeBlend warps the pictures.
    bool drba = false;
    // DRBA's block0 on the host's kernels (nativeB0Module): its module and five kernels, the weight block (b0Off, see
    // b0NatLoad), the scratch (in the shared activation block, else dB0Own)
    CUmodule b0Mod = nullptr;
    CUfunction fB0[5] = {};
    void* dB0W = nullptr;
    size_t b0Off[28] = {};
    void* dB0Own = nullptr;
    // RIFE's IFNet on the host's kernels (ifnetPath = kIfnNative, nativeIfnModule): its module and kernels, the weight
    // block (ifnOff, see ifnNatLoad), the scratch (in the shared activation block, else dIfnOwn)
    bool ifnNat = false;
    CUmodule ifnMod = nullptr;
    CUfunction fIfn[12] = {};
    void* dIfnW = nullptr;
    std::vector<size_t> ifnOff;
    void* dIfnOwn = nullptr;
    ChainGraphs ifnGraphs;        // its launches as CUDA graphs (nativeIfnRun; SMV_IFN_GRAPH)
    uint32_t drFid = 0;           // frames pushed this session; frame id i sits in ring slot i & 3
    float* dDrIBase = nullptr;    // the four ring slots in ONE block, slot i at i * (3, ph, pw): two ring neighbours
                                  // are one (6, ph, pw) frame pair
    float* dDrI[4] = {};          // (3, ph, pw) the padded frames (the pictures), slot i of dDrIBase
    uint32_t drPicFix = 0;        // two domains: the k whose wrap pair copied frame k - 1 into slot (k + 1) & 3
    uint16_t* dDrMBase = nullptr; // two domains: the motion ring, five (3, M, N) fp16 slots in ONE block: frame i
                                  // in slot 3 - (i & 3) (descending, so frame i - 1 follows frame i), slot 4 a spare
    uint16_t* dDrM[4] = {};       // two domains: their motion frames, ring slot i & 3 = block slot 3 - (i & 3)
    float* dDrF[4] = {};          // (kRifeFeatC, M, N) their encodes in the IFNet's feature dtype (nativeTweenBuffers)
    float* dDrX[2] = {};          // one domain: (6, ph, pw) the frame pair: [0] = [k-1, k-2] (side -1), [1] = [k-2,
                                  // k-1]; the IFNet's x (fp16 in place when xHalf)
    uint16_t* dDrMX[2] = {};      // two domains: the IFNet's x, two neighbouring slots of dDrMBase (not owned)
    uint32_t drXFor[2] = {};      // the newest frame id each x was built for (0 = none)
    struct DrWin
    {
        uint32_t c;
        float* f10;
        float* r;
    }; // r = (4, M, N): flow12 | flow21
    DrWin drWin[2] = {};         // c = 0: empty
    long long* dDrAcc = nullptr; // (6, M, N) the splat accumulator
    uint64_t drTweens = 0, drHeads = 0, drBlock0 = 0;
    // DRBA as its author runs it (routineLife1/DRBA infer.py; drbaAuthorWanted): the network's frames stretched
    // bilinearly over its whole /64 frame as B, G, R (k_motionIn st), RIFE's last step at that frame on the pair's
    // stretched pictures, every output frame (tweens and real frames alike) bilinearly back to the pictures' size
    bool drAuthor = false;
    float* dDrS = nullptr; // (6, M, N) the pair's pictures stretched (nativeDrbaAuthorBlend): two (3, M, N) slots,
                           // frame i in slot i & 1, each frame stretched once (both sides of a group and the next
                           // group's left frame read it)
    int64_t drSFid[2] = {-1, -1}; // the frame id in each slot of dDrS (-1 = none)
    float* dDrB = nullptr;  // (3, M, N) the blend at the network's frame; a real frame's stretch on its way back: the
                            // accumulator's last 12 B a pixel (nativeDrbaAlloc, not owned)
    float* dDrRT = nullptr; // (3, ph, pw) a real frame stretched and back (nativeDrbaReal): the accumulator's first
                            // 36 B a motion pixel when the picture fits there, else its own block (drRtOwn)
    bool drRtOwn = false;
    // NVIDIA Optical Flow model: the handoff answers
    // `engine=nvof` with the geometry only (like echo); this host runs
    // the Optical Flow Accelerator itself through the driver's nvofapi64.dll (grid 4, fast,
    // BOTH directions in one Execute, gray8 of the two model frames at the true w x h) and the
    // glue kernels k_nvofLuma / k_nvofUp / k_nvofMetric (checked against fp64 references;
    // k_splatNvof / k_splatNvofNorm in the kernel text are a forward-splat tween with no host
    // launch, kept because the kernel text is shared byte for byte with that check).
    bool nvof = false;
    NvOFHandle ofH = nullptr;
    NvOFGPUBufferHandle ofIn[2] = {}, ofOut[2] = {}, ofCost[2] = {}; // [0] forward, [1] backward
    CUdeviceptr ofInP[2] = {}, ofOutP[2] = {}, ofCostP[2] = {};
    uint32_t ofInPitch = 0, ofOutPitch[2] = {}, ofCostPitch[2] = {};
    int ofGw = 0, ofGh = 0;
    float* dNvFlow[2] = {};      // (2, h, w) px: [0] = F01 (frame 0 -> 1), [1] = F10
    float* dNvCost[2] = {};      // (h, w) the upsampled cost per direction
    float* dNvZ[2] = {};         // (h, w) the splat metric per direction
    long long* dNvAcc = nullptr; // (5, h, w): RGB * e, e, coverage
    float* dNvOut = nullptr;     // (3, ph, pw): the tween, the layout storeSlot reads
    bool nvProf = false;         // SMV_LIVE_NVOF_PROF=1: per-pair GPU ms
    cudaEvent_t nvEv[2] = {};
    double nvPairMs = 0.0, nvTweenMs = 0.0;
    uint32_t nvPairN = 0, nvTweenN = 0;
    // the tween: the pull warp + confidence fallback. The velocity
    // at time t (n / d = the push-pull level 0 inputs), the two visibilities, the raw and the
    // blurred fallback mask, the blur's pass, and the push-pull pyramid levels 1.. (n, d, out)
    float* dNvN0 = nullptr;      // (2, h, w) velocity * confidence
    float* dNvD0 = nullptr;      // (h, w) confidence
    float* dNvVis = nullptr;     // (2, h, w) visibility in frame 0 / frame 1
    float* dNvMraw = nullptr;    // (h, w) the fallback mask before the blur
    float* dNvMask = nullptr;    // (h, w) the blurred mask (1 = warp, 0 = blend)
    float* dNvBlurTmp = nullptr; // (h, w) the blur's horizontal pass
    float* dNvV = nullptr;       // (2, h, w) the filled velocity (push-pull level 0 out)
    struct NvLevel
    {
        int w, h;
        float* n;
        float* d;
        float* out;
    };
    std::vector<NvLevel> nvPyr;                 // levels 1.. down to 1 px on the short side
    bool restHalfIn = true, restHalfOut = true; // Restore's input / output dtypes (fp16 on the host's kernels)
    int restScale = 4;                          // the model's scale: Real-ESRGAN 4, AnimeJaNai 2
    bool restSub = false; // the output = 3 s^2 sub-pixel planes at the input size (Real-ESRGAN): k_restFoldHSub
    bool restFailed = false;
    int restTw = 0, restTh = 0; // the fold target (decided once VSR is known)
    // Real-ESRGAN on the host's kernels (restorePath = kRestNative, no engine): dRestA / dRestB = the channels-last fp16
    // ping-pong (64 halves a pixel each; inside the shared activation block when it holds them, else restNatOwn),
    // dRestW = the weights (restNatOff offsets)
    bool restNat = false, restNatOwn = false;
    int restNatKind = 0;   // 1 = the bundled Real-ESRGAN, 2 = AnimeJaNai (SPAN-F3 64, the --restore-model file),
                           // 3 = Anime4K's Restore CNN UL (--restore-anime4k: the Anime4K module's passes)
    int restNatBlocks = 0; // AnimeJaNai's block count
    void *dRestA = nullptr, *dRestB = nullptr, *dRestW = nullptr; // dRestA = the whole activation region
    std::vector<size_t> restNatOff;
    // the fold's sub-pixel input (restSub): s, the plane grid w x h, the nearest-base add (Real-ESRGAN) or none
    int restSubS = 4, restSubW = 0, restSubH = 0;
    bool restSubBase = true;
    void* dRestIn = nullptr;   // the input: NCHW at the source size (srcW x srcH), fp16
    void* dRestOut = nullptr;  // the output: the model's fp16 sub-pixel planes (restSub)
    float* dRestTmp = nullptr; // the fold's horizontal pass, planar fp32 at (4h, the largest scale) x max target width
    float* dRest = nullptr;    // the fold back to the model size (VSR follows), planar fp32 w x h
    float* dRestRem = nullptr; // HDR planes: the source fitted to Restore's target (k_hdrRestOut's remainder)

    // ---- ring geometry (mirrors the exe's own slot layout)
    uint32_t slots = 0, pitch = 0;
    size_t slotBytes = 0;

    // ---- the models' dtypes
    bool featHalf = false; // IFNet / block0 take f0 / f1 in fp16: the encode (nativeEncodeRun) writes that dtype
    // the IFNet's x in fp16 (the native IFNet takes the fp16 copy of the pair, dXh, or the motion frames)
    bool xHalf = false;
    // the tweens (merged) are fp16 (k_rifeBlend, SMV_TWEEN_FP16=1), else fp32 (k_rifeBlendF): every reader of a
    // tween gets this as its half flag (mergedAt)
    bool outHalf = false;
    // the IFNet takes its timestep in fp16: the constant fills and DRBA's map are written in fp16
    bool tHalf = false;

    // ---- CUDA
    cudaStream_t stream = nullptr;
    cudaExternalMemory_t emCap = nullptr, emOut = nullptr;
    cudaExternalSemaphore_t semCap = nullptr;
    cudaMipmappedArray_t capMip = nullptr;
    cudaArray_t capArr = nullptr;
    uint8_t* dOutRing = nullptr;
    uint8_t* dCap = nullptr;
    float* dTmp = nullptr; // the capture resize's horizontal pass, and nativeFitPlanar's (tmpN floats)
    size_t tmpN = 0;
    // the Lanczos tap tables, one per resize axis in -> out (rsTabFor), freed by nativeFree
    struct RsTab
    {
        int in, out;
        void* d;
    };
    std::vector<RsTab> rsTabs;
    float* dCapF = nullptr;       // HDR + image scale under 1: planar PQ at capture resolution
    uint8_t* dThdrIn = nullptr;   // RGBA8 bridge input, pitch 4*cw
    uint32_t* dThdrOut = nullptr; // packed 10:10:10:2 bridge output, B in the low bits
    float* dSrcG = nullptr;       // the unquantised sRGB-encoded source, planar fp32
    float* dShIn = nullptr;       // the Sharpen's tight SDR view of HDR planes (w x h)
    float* dShOut = nullptr;      // and the sharpen's tight result padded back into the model frame (Sharpen on only)
    // offline: the TrueHDR buffers above are sized dw x dh (TrueHDR runs at the working size before the
    // model), and the emit's per-frame statistics block (nativeOfflinePqOut), read back one frame late
    uint8_t* dThdrStats = nullptr; // k_pqOut's hist / misc / vSum (kThdrStatsBytes)
    uint8_t* hThdrStats = nullptr; // its pinned copy, valid after the next stream sync
    bool thdrStatsPending = false;
    ThdrAcc* thdrAcc = nullptr; // the render's accumulators (runOfflineSession owns them)
    uint32_t thdrRepeat = 0;    // held slots that reuse the pending frame's record (DLSS 5)
    // offline DLSS 5: the NR core on its own D3D12 device, one RGBA16F frame at nrW x nrH, the
    // decoded frame before the interpolation (nativeNrFrame). Zero-copy by default: the frame
    // moves through the core's two shared D3D12 buffers, which the stream writes and reads
    // directly, and the core's shared fence orders both sides on the GPU. CPU staging
    // (SMV_NR_STAGED=1, the preview's change map, or a handoff that cannot start): the core's
    // renderFrame through pinned host copies.
    nr::Host* nrHost = nullptr; // null = DLSS 5 off or unavailable for this render
    bool nrFailed = false;      // an evaluate failed: off for the rest of the render
    bool nrFirst = true;        // the stream's first frame evaluates with Reset
    bool nrResetEvery = false;  // SMV_NR_RESET_EVERY=1: every frame (the equivalence gate's lever)
    bool nrReuse = false;       // a decoded frame identical to the previous one takes its DLSS 5 output
    uint64_t nrReused = 0;      // frames that did, this render
    bool nrZeroCopy = false;    // the shared-buffer route is up
    int nrW = 0, nrH = 0;       // the decoded picture's size, the size DLSS 5 runs at
    float* dRawPrev = nullptr;  // the previous decoded frame, packed: the identical-pair test's
                                // reference once DLSS 5 has rewritten the frames in dX (SMV_NR_REUSE=0 only)
    uint16_t* dNrIo = nullptr;  // CPU staging: device RGBA16F, both directions (zero-copy uses the core's textures)
    uint16_t* hNrIn = nullptr;  // pinned host copies handed to renderFrame (CPU staging)
    uint16_t* hNrOut = nullptr;
    // zero-copy: the core's own Color / Output / MVec textures as CUDA arrays (dedicated imports, mapped
    // arrays, surface objects the kernels store into and load from)
    cudaExternalMemory_t emNrIn = nullptr, emNrOut = nullptr, emNrMv = nullptr;
    cudaMipmappedArray_t mmNrIn = nullptr, mmNrOut = nullptr, mmNrMv = nullptr;
    unsigned long long sfNrIn = 0, sfNrOut = 0, sfNrMv = 0;
    cudaExternalSemaphore_t semNr = nullptr; // the core's shared fence
    uint64_t nrFenceV = 0;                   // last value put on it: odd = the stream's, even = the queue's
    // DLSS 5 motion (zero-copy route, nativeNrMotion): its own Optical Flow session at nrW x nrH on
    // the NR input frames, current -> previous, into the core's MVec texture (DLSSNR.MVec).
    // Off = SMV_NR_MV=0 or CPU staging: the pass then gets no motion
    bool nrMotion = false;
    NvOFHandle nrOfH = nullptr;
    NvOFGPUBufferHandle nrOfIn[2] = {}, nrOfOut = nullptr;
    CUdeviceptr nrOfInP[2] = {}, nrOfOutP = 0;
    uint32_t nrOfInPitch = 0, nrOfOutPitch = 0;
    int nrOfGw = 0, nrOfGh = 0;
    int nrOfCur = 0;          // the input slot this frame's luma goes into (they alternate)
    float* dNrFlow = nullptr; // (2, nrH, nrW): the field in px (k_nrMv reads no cost)
    std::wstring nrDeltaPath; // --nr-delta PATH: the pass's change map (the preview's mask)
    double nrMs = 0.0, nrMaxMs = 0.0;
    uint64_t nrN = 0;
    // live DLSS 5: the same core, handoff and motion once per captured frame on the
    // model frame after Restore / the resize (nativeLiveNrInit, nativeGroup); a capture byte-identical
    // to the previous one keeps the last output in the cur half instead of an evaluate
    bool liveNr = false;
    bool nrHaveLast = false;     // the frame before holds an evaluated output
    uint8_t* dCapPrev = nullptr; // the previous raw capture, the reuse test's reference; trades places with dCap
    float nrSdrWhite = 3.0f;     // HDR: the SDR reference white in scRGB units (nits / 80)
    int nrSrcHdr = 0;            // offline: HDR video (1 PQ, 2 HLG), the pass sees its SDR range
    // GMFSS and Restore on HDR planes (hdrToModel): the decoded frame's encoding (offline Restore reads it; 1 PQ,
    // 2 HLG, 0 SDR planes or off) and the model frames' (after RTX HDR: GMFSS, offline Restore after the model, live
    // Restore on PQ), the SDR white in scRGB units
    int encPre = 0, encPost = 0;
    float encWhite = 203.0f / 80.0f;
    // Sharpen on HDR planes without RTX HDR: the sharpen edits their SDR view (knee 1 head 0) as an SDR source's, clamped
    // at SDR white, and the light outside the view is added back; false = SMV_HDR_SHARPEN_VIEW=0, the sharpen on the codes
    bool shHdrView = true;
    // a resize of HDR planes before the model (live the capture to the working size, offline the decoded frame to it,
    // Restore's reference at a size change): a source frame inside the SDR range is resized as its SDR view (the SDR
    // route's Lanczos, clamped at SDR white) and comes back through k_hdrFromSdr, any other on the codes; false =
    // SMV_HDR_RESIZE_VIEW=0. dRsView = the source's SDR view (planar fp32 at the source size), set only when used.
    bool rsHdrView = true;
    float* dRsView = nullptr;
    int* dRsRange = nullptr;
    int* hRsRange = nullptr;
    int rsModeLogs = 0;
    bool rsLastView = false;
    bool nrSdrIn = false;  // live: DLSS 5's input is SDR this frame (RTX HDR converts after it)
    bool nrPaused = false; // live DLSS 5 switched off in a running session (stdin set): the core stays, the pass skips
    // a DLSS 5 pass growth in progress (stdin nrpasses above the passes built, nativeNrGrowStep): nrGrowWant = the run
    // count asked (0 = none), nrGrowFrom = the passes before it, the build time so far, the set's value and the note
    int nrGrowWant = 0, nrGrowFrom = 0;
    double nrGrowMs = 0.0;
    std::string nrGrowVal, nrGrowWhy;
    // an effect switched off in a running session (stdin set) through its own fallback (vsrFailed = Lanczos, restFailed =
    // no Restore, rtxFailed = faithful PQ): set when the switch set that flag, so on again clears only a switch, never a
    // real failure
    bool vsrUserOff = false, restUserOff = false, rtxUserOff = false;
    // offline: Restore and the resize run on the decoded frame (the pre-model
    // SOURCE, sw x sh, in dSrcPl) before DLSS 5 and the model, which then run at the working
    // size w x h, and the emit's final resize takes it to dw x dh; sw / sh = 0 = the source is the
    // model frame (no working size, and live)
    int sw = 0, sh = 0;
    bool nvPre = false;
    bool vsrPost = false;       // RTX VSR runs the final resize (it enlarges), not the first
    bool vsrPre = false;        // live: RTX VSR runs the capture -> working-size resize before the model, not the fit
    bool restPre = false;       // live: Restore on the captured frame, folded to the model size, before the model
    float* dSrcPl = nullptr;    // the decoded frame, planar fp32 at sw x sh
    float* dPres = nullptr;     // a resize stage's staging frame, planar fp32 at dw x dh (nativeOfflineStage)
    float* dFitTmp = nullptr;   // fitAa: the horizontal pass, planar fp32 at dw x (uh or h)
    float* dUp = nullptr;       // Upscale to: the internal render frame, planar fp32 at uw x uh
    float* dUpTmp = nullptr;    // upAa: the horizontal pass of the first resize, planar fp32 at uw x h
    uint8_t* dVsrIn = nullptr;  // VSR: tight RGBA8 (FSR upscaling: BGRA8) at the source size (srcW x srcH)
    uint8_t* dVsrOut = nullptr; // VSR: tight RGBA8 (FSR upscaling: BGRA8) at the presented size dw x dh
    float* dVsrView = nullptr;  // live HDR planes: the SDR view RTX VSR reads, planar fp32 at its input size
    float* dVsrRem = nullptr;   // live HDR planes: the Lanczos fit at VSR's target (k_hdrRestOut's remainder)
    unsigned hdrInBits = 0;     // live HDR planes: one bit a capture, newest lowest: it stayed inside the SDR range
    float* dX = nullptr;        // (1,6,ph,pw): prev in planes 0..2, cur in 3..5 (xSwap: the reverse)
    bool xSwap = false;         // live: dX's halves trade roles instead of the cur -> prev copy (nativeXPrev / Cur)
    uint16_t* dXh = nullptr;    // dX in fp16 for an IFNet that takes fp16 frames (xHalf)
    float* dF[2]{};             // f_prev / f_cur, (1,kRifeFeatC,ph,pw) in the IFNet's feature dtype (featHalf: fp16)
    float* dEncOwn = nullptr;   // the encode's half-size planes when no shared activation block holds them
    // the IFNet's features, timestep, flow and mask and the tweens, allocated once the engine's dtypes are
    // known (nativeTweenBuffers)
    void* dT = nullptr;        // (B,1,mph,mpw), fp32 or fp16 by tHalf
    uint32_t tHeld[64] = {};   // the t bits timestep plane j holds when bit j of tHeldMask is set: nativeFillT skips
    uint64_t tHeldMask = 0;    // a refill of the same t; cleared whenever anything else writes dT
    uint16_t* dFlow = nullptr; // (B,4,mph,mpw) fp16: the IFNet's final flow 0->t | 1->t
    uint16_t* dMask = nullptr; // (B,1,mph,mpw) fp16: its blend mask
    float* dMerged = nullptr;  // (B,3,ph,pw) fp32 or fp16 by outHalf: the tweens RIFE's last step makes from them
    // live with the model frame = the presented rect (no resize after the model): k_rifeBlendOut stores each
    // tween straight into its slot and dMerged is never allocated
    bool blendOut = false;
    // RIFE's two domains (offline and live RIFE): the IFNet and its
    // encode run on the motion frames dM, the finished picture in SDR sRGB at the decoded size
    // (offline) or at the working size, the capture size when the working size enlarges it (live)
    // (mw x mh, padded to mph x mpw), and k_rifeBlend applies their flow and mask to the pictures dX
    // at the working size. mph = 0: the model's frame is the picture's (ph x pw), the IFNet reads dX / dXh.
    int mph = 0, mpw = 0, mw = 0, mh = 0;
    // plain RIFE (two domains, both routes) as its author pads: zeros to a multiple of 128 (Practical-RIFE at scale 1)
    // in the motion frames and under the blend's taps, instead of the edge replicated to a multiple of 64 (DRBA's pad);
    // live reads it off the handoff's zpad=1; SMV_RIFE_AUTHOR_PAD=0 = the old pad
    bool zeroPad = false;
    uint16_t* dM = nullptr; // (1,6,mph,mpw) fp16: prev in planes 0..2, cur in 3..5
    float* dMFit = nullptr; // live: the working-size picture shrunk to mw x mh (3 fp32 planes)
    int fCur = 0;
    cudaEvent_t capEv = nullptr;
    std::vector<cudaEvent_t> slotEv;
    // live: the capture read's GPU time stamp, where a group's own GPU time starts (grEv below)
    cudaEvent_t pfEv[1] = {};
    // live: a group's own GPU time for the present loop's throttle, from pfEv[0] (the capture read): grEv[0] where the
    // slots start (after the pair's model work), grEv[1] after the last slot. Smoothed per kind of group, in
    // microseconds: gIntUs = the work before the slots of a group with tweens, gTweenUs = one tween with its store
    cudaEvent_t grEv[2] = {};
    bool grArmed = false;
    uint32_t grTween = 0, grSlots = 0;
    double gIntMs = 0.0, gTweenMs = 0.0;
    std::atomic<uint32_t> gIntUs{0}, gTweenUs{0};
    // Smooth Motion's midpoint tree: one bridge call, and one slot's unpack + store (the slots' time less the calls'),
    // so the exe prices the direct tree ((M - 1) calls + M stores) and the pipelined one (M stores + the longer of
    // the compute thread's calls and instance 0's worker's, each at the pipelined groups' own price: gFrPipeMainUs =
    // the compute thread's chain a call incl. the join, gFrPipeWkUs = a worker call) apart; AMD FSR 4's: one call, one
    // slot, and an x2 group's whole time (gFrBaseUs); 0 = not timed yet
    std::atomic<uint32_t> gFrCallUs{0}, gFrStoreUs{0}, gFrBaseUs{0}, gFrPipeMainUs{0}, gFrPipeWkUs{0};
    double gFrBaseMs = 0.0;
    // AMD FSR 4's level-1 call (made before the slots at every ladder): its FSR span on the stream (fgL1Ev, from the
    // fence signal to the result's wait), that span in the x2 groups that set the base (gFrL1BaseMs) and the call
    // price over it as x4 and deeper groups show it (gFrL1Ratio), so an x2 group keeps the call price current; priced
    // by x4 and deeper groups alone, a load that passed while x4 fell would keep pricing x4 out for the session. Two
    // sets by group (fgEvSet = the one this group records): a pipelined group's call may still run when the next
    // group starts (its pack does not wait for it), so its spans are read once done, without a wait
    cudaEvent_t fgL1Ev[2][2] = {};
    bool grL1Armed[2] = {};
    int fgEvSet = 0;
    double gFrL1Ratio = 0.0, gFrL1BaseMs = 0.0;
    // AMD FSR 4's pipelined route (SMV_FSRFG_PIPE) is priced off three parts every group times on either route: the
    // level-1 Optical Flow on fgOfL1St (fgOfEv: the direct route hops there too), the level-1 call's FSR span
    // (fgL1Ev) and the session stream's pre outside that chain (the pipelined pre, or a direct pre less the two); the
    // exe's pipelined cost = the longest of the Optical Flow chain, the session stream's chain (the deeper calls + the
    // later of the stores + pre and the next level-1 call) and half the loop through both. 0 = not timed yet
    cudaEvent_t fgOfEv[2][2] = {};
    bool grOfArmed[2] = {};
    // a pipelined group's slots start with the wait for the shown pair's level-1 node (grLagEv after it, armed in
    // nativeFsrfgLagTween): its deeper calls are priced from there
    cudaEvent_t grLagEv = nullptr;
    bool grLagArmed = false;
    double gFgPreMs = 0.0, gFgOfMs = 0.0, gFgL1Ms = 0.0;
    std::atomic<uint32_t> gFgPreUs{0}, gFgOfUs{0}, gFgL1Us{0};

    CUmodule cuMod = nullptr;
    std::wstring kernCacheDir; // the cubin folder nativeBuildKernels was given (Anime4K's module caches there too)
    CUfunction fPackInDirect = nullptr, fResizeH = nullptr, fResizeV = nullptr, fH2f = nullptr, fF2h = nullptr,
               fPackOut = nullptr;
    CUfunction fPackInDirectHdr = nullptr, fPqPlanar = nullptr, fResizeHf = nullptr, fPackOutHdr = nullptr;
    CUfunction fSdrEncode = nullptr, fThdrColor = nullptr;                                    // live TrueHDR
    CUfunction fThdrIn = nullptr, fSharpThdrIn = nullptr, fPqOut = nullptr, fPqLut = nullptr, // TrueHDR in, the
        fSdrPq = nullptr;                                                                     // offline PQ emit
    CUfunction fFitPlanar = nullptr, fSharpPlanar = nullptr,                                  // sharpen
        fPackBgraRgb = nullptr, fUnpackBgraRgb = nullptr,                   // FSR upscaling / FRUC: (R, G, B) <-> BGRA8
        fPackRgbaH = nullptr, fRgbaHAbgr = nullptr, fUnpackRgbaH = nullptr, // FSR frame generation: RGBA16F frames,
        fMvHalf = nullptr, fMvHalfG1 = nullptr, fMvDerive = nullptr,        // the Optical Flow input, the vectors
        fBgraRgbaFsr = nullptr, fRgbaBgra = nullptr, fOddEdge = nullptr, // FSR upscaling: BGRA8 <-> RGBA8; VSR's slot
        fFuMvScale = nullptr,                                            // its chained passes' vectors
        fPackBgra = nullptr, fUnpackRgba = nullptr, fPackR10 = nullptr,  // offline DLSS 4.5 frames, RTX VSR
        fUnpackR10 = nullptr;
    CUfunction fFitAaH = nullptr, fFitAaV = nullptr, fRsTab = nullptr;           // the separable fit, its tap tables
    CUfunction fFsrHdrIn = nullptr, fFsrHdrOut = nullptr, fFsrHdrPack = nullptr; // AMD FSR upscaling on HDR planes
    CUfunction fPackOutV = nullptr, fPackOutHdrV = nullptr;                      // its vertical pass + the slot store
    CUfunction fRestIn = nullptr, fRestFoldH = nullptr, fRestFoldV = nullptr,    // live Restore
        fRestFoldHSub = nullptr, fA4kFoldHSub = nullptr, fA4kFold2x = nullptr, fRestToF = nullptr, fClamp01 = nullptr;
    CUfunction fRestConv0 = nullptr, fRestConvMid = nullptr, fRestConvLast = nullptr; // Restore on the host's kernels
    CUfunction fAjFirst = nullptr, fAjAtt = nullptr, fAjCat = nullptr;                // AnimeJaNai's own three
    CUfunction fPadPlanar = nullptr; // the decoded frame into the model frame
    CUfunction fNrIn = nullptr, fNrOut = nullptr, fNrMv = nullptr, fNrMvClear = nullptr; // DLSS 5
    CUfunction fNrInPq = nullptr, fNrOutPq = nullptr;                 // live DLSS 5 on HDR (PQ) planes
    CUfunction fHalf = nullptr, fPyr = nullptr, fSplatSoft = nullptr, // live GMFSS glue (5b)
        fSplatNorm = nullptr;
    CUfunction fGsCount = nullptr, fGsScanA = nullptr, fGsScanB = nullptr, fGsScanC = nullptr, fGsFill = nullptr,
               fGsGather = nullptr;                           // the gather splats
    CUfunction fGsGatherNhwc = nullptr;                       // the feature splats on channels-last features
    CUfunction fPackInRaw16 = nullptr, fPackInRaw8 = nullptr, // offline
        fPackOutRaw16 = nullptr, fPackOutRaw8 = nullptr, fExpand8to16 = nullptr;
    CUfunction fPairDiff = nullptr, fRawDiff = nullptr;                       // identical-pair test, DLSS 5 reuse test
    CUfunction fNvofLuma = nullptr, fNvofUp = nullptr, fNvofMetric = nullptr; // the nvof model
    CUfunction fSplatVel = nullptr, fVelNorm = nullptr, fPpDown = nullptr,    // its pull-warp tween
        fPpTop = nullptr, fPpUp = nullptr, fBlur1 = nullptr, fNvofCompose = nullptr;
    CUfunction fDrFlowSplat = nullptr, fDrFlowNorm = nullptr, // native DRBA
        fDrDrmSplat = nullptr, fDrDrmNorm = nullptr;
    CUfunction fMotionIn = nullptr, fRifeBlend = nullptr, // RIFE's motion frame, its last step (the tween: fp16
        fRifeBlendF = nullptr;                            // store, fp32 store)
    CUfunction fRifeBlendOut = nullptr;                   // its last step fused with a 1:1 slot store (blendOut)
    CUfunction fRifeBlendOut32 = nullptr;                 // the same for fp32 pictures
    CUfunction fBilin = nullptr;                          // PyTorch's bilinear resize (DRBA as its author)
    CUfunction fHdrEnc = nullptr, fHdrFromSdr = nullptr, fHdrRange = nullptr; // GMFSS / Restore on HDR planes
    CUfunction fPackBgraSdr = nullptr, fUnpackBgraSdr = nullptr;              // FRUC's SDR pairs on HDR planes
    CUfunction fHdrRestOut = nullptr;                                         // Restore's way back on HDR planes
    CUfunction fLocalCorr = nullptr;                                          // gmflow's local correlation, fused
    CUfunction fAttn2 = nullptr;      // gmflow's global matching and propagation (k_attn2)
    CUfunction fFuTail = nullptr;     // GMFSS fusion's pixel shuffle + conv_last on the cut graph's out (k_fuTail)
    CUfunction fFuTailNhwc = nullptr; // the same on a channels-last out (k_fuTailNhwc)
    CUfunction fEncC01 = nullptr, fEncC2Up = nullptr; // RIFE's encode (nativeEncodeRun)
    int* dStaticFlag = nullptr;                       // device flag k_pairDiff sets when the pair differs
    int* hStaticFlag = nullptr;                       // pinned readback of it, one int per group
    uint64_t staticN = 0;                             // identical pairs held this session

    // ---- protocol plumbing
    std::mutex mMsg, mTok;
    std::condition_variable cvMsg, cvTok;
    std::deque<std::vector<uint8_t>> msgs;
    std::deque<uint32_t> toks;
    std::thread th;
    volatile LONG dead = 0;
    bool started = false;
    std::mutex mInit;
    std::condition_variable cvInit;
    int initState = 0; // 0 pending, 1 ready, -1 failed
    uint32_t seq = 0;
    bool havePrev = false;

    // ---------------- lifetime ----------------
    void pushTok(uint32_t v)
    {
        {
            std::lock_guard<std::mutex> lk(mTok);
            toks.push_back(v);
        }
        cvTok.notify_one();
    }
    void die(const char* why)
    {
        LOG("native: %s\n", why);
        InterlockedExchange(&dead, 1);
        // pass through both locks before the wake: a waiter that has tested isDead but not
        // yet blocked would miss a notify sent in that gap
        {
            std::lock_guard<std::mutex> lk(mTok);
        }
        {
            std::lock_guard<std::mutex> lk(mMsg);
        }
        cvTok.notify_all();
        cvMsg.notify_all();
    }
    bool isDead() const
    {
        return InterlockedCompareExchange((volatile LONG*)&dead, 0, 0) != 0;
    }

    bool write(const void* p, DWORD n)
    {
        if (isDead())
            return false;
        const uint8_t* b = (const uint8_t*)p;
        {
            std::lock_guard<std::mutex> lk(mMsg);
            msgs.emplace_back(b, b + n);
        }
        cvMsg.notify_one();
        return true;
    }
    bool read(void* p, DWORD n)
    {
        uint8_t* d = (uint8_t*)p;
        while (n)
        {
            std::unique_lock<std::mutex> lk(mTok);
            cvTok.wait(lk, [&] { return !toks.empty() || isDead(); });
            if (toks.empty())
                return false;
            const uint32_t v = toks.front();
            toks.pop_front();
            lk.unlock();
            if (n < 4)
                return false;
            memcpy(d, &v, 4);
            d += 4;
            n -= 4;
        }
        return true;
    }
};
// the previous and the current frame's planes in dX (offline never swaps: prev in planes 0..2)
static float* nativeXPrev(const NativeRife& nr)
{
    return nr.dX + (nr.xSwap ? 3 * (size_t)nr.ph * nr.pw : 0);
}
static float* nativeXCur(const NativeRife& nr)
{
    return nr.dX + (nr.xSwap ? 0 : 3 * (size_t)nr.ph * nr.pw);
}
// ---- part 2: handoff, engine load, kernels, buffers, the compute thread -------------------

// What a --resident process keeps between sessions: the NVRTC kernel module and the models' dtypes,
// plus the last successful handoff's facts so the same window starts again without a new handoff.
// Everything else in NativeRife (stream, buffers, imports, events) is per session. Sessions are
// strictly sequential (the previous compute thread is joined before the next early thread starts),
// so no lock guards this.
struct NativeResident
{
    std::string ifnetPath;
    bool ready = false; // a session's model state (the dtypes below) is held for the next one
    CUmodule cuMod = nullptr;
    bool featHalf = false;
    bool xHalf = false, outHalf = false, tHalf = false;
    int dev = 0;
    // memoised handoff: nativeHandoff's session key (gen left out: the fast path ignores gen and
    // the cold path builds the same engines whatever the gen) and the facts it answered with
    std::wstring handoffKey;
    NativeRife facts;
    bool haveFacts = false;
};
static NativeResident g_res;

// drop the resident engine pair (a different pair is needed for a new window size). The
// kernel module and the memoised handoff are engine independent and stay: the session that
// calls this already adopted the module in nativeBuildKernels and stored the new handoff
// facts (unloading the module here broke the first kernel launch of a later session). The
// calling thread must have the device bound.
static void residentFreeEngines()
{
    if (!g_res.ready)
        return;
    g_res.ready = false;
    g_res.ifnetPath.clear();
    LOG("native: resident engines released\n");
}

// everything, at host exit (after the last session's compute thread is gone)
static void residentFree()
{
    residentFreeEngines();
    if (g_res.cuMod)
    {
        cuModuleUnload(g_res.cuMod);
        g_res.cuMod = nullptr;
    }
    g_res.handoffKey.clear();
    g_res.haveFacts = false;
}

static bool fileExistsA(const std::string& p)
{
    const DWORD a = GetFileAttributesA(p.c_str());
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
}

static bool lkXefgDir(const std::string& engDir, std::string& dir, const char* tag); // Intel XeSS-FG's folder, below

// the server backends this host runs (PipeServer::nativeRefusal asks here): rife, echo (the
// effects-only route, the no-engine mode above), gmfss (its chain on the host's kernels), nvof (the NVIDIA
// Optical Flow model), fruc (Nvidia Smooth Motion), fsrfg (AMD FSR frame generation; Intel XeSS-FG with --xess-fg)
// and rifedrba (RIFE with DRBA timing). Any other name is refused.
static bool nativeBackendOk(const std::wstring& backend)
{
    return backend == L"rife" || backend == L"echo" || backend == L"gmfss" || backend == L"nvof" ||
           backend == L"fruc" || backend == L"fsrfg" || backend == L"rifedrba";
}

// the smallest GMFSS pad a side (lkSession): gmflow's quarter-scale windows are the pad / 64 a side and its shifted
// blocks roll by half a window, so a window needs 2 tokens a side (a 1-token window rolls by 0: an empty slice in the
// export's roll, transformer._roll_hw, untested); the coarse scale's per-region blocks (`_m`) need 2 at the pad / 32
static constexpr int kGmMinPad = 128;

// Upscale to: the
// internal render size from the exe's own --upscale H and the handoff geometry: the factor
// up_h / h clamped to 1/16..16, even dims, dropped when it
// equals the fit rect (one resize, not two). lround rounds half away from zero; the & ~1 lands a
// tie on the same even number as rounding half to even. Runs after (w, h, dw, dh) are known,
// on the resident facts too (uw is never a stored fact, the flag is per session).
static void nativeDeriveUpscale(NativeRife& nr)
{
    nr.uw = nr.uh = 0;
    nr.upAa = false;
    nr.fitAa = nr.dw != nr.w || nr.dh != nr.h;
    if (g_upscaleH > 0 && g_upscaleH != nr.h && nr.h > 0)
    {
        double f = (double)g_upscaleH / (double)nr.h;
        f = f < 1.0 / 16 ? 1.0 / 16 : (f > 16.0 ? 16.0 : f);
        int uw = (int)lround(nr.w * f) & ~1, uh = (int)lround(nr.h * f) & ~1;
        if (uw < 2)
            uw = 2;
        if (uh < 2)
            uh = 2;
        if (uw != nr.dw || uh != nr.dh)
        {
            nr.uw = uw;
            nr.uh = uh;
            nr.upAa = uw != nr.w || uh != nr.h;
            nr.fitAa = nr.dw != uw || nr.dh != uh; // the second resize, internal frame -> the fit rect
        }
    }
}

// ---- the session handoff --------------------------------------------------------------------
// Every net runs on the host's kernels, so the host answers the handoff itself from the session's
// arguments: NATIVE-PATH lines plus one LIVE READY line, the format nativeHandoff parses.

static std::string lkEnv(const char* name)
{
    char buf[512];
    size_t n = 0;
    if (getenv_s(&n, buf, sizeof(buf), name) || !n)
        return std::string();
    return std::string(buf);
}

// SMV_HANDOFF_DUMP=1: log every handoff line the host produces as `handoff-line ...`
static const bool g_handoffDump = lkEnv("SMV_HANDOFF_DUMP") == "1";

// round half to even (the default FP rounding mode)
static int lkRound(double x)
{
    return (int)std::nearbyint(x);
}

static bool lkFile(const std::string& p)
{
    return fileExistsA(p);
}

// RIFE's IFNet on the host's kernels (nativeIfnModule): the path every RIFE / DRBA lookup hands out, nothing to build
static const char* const kIfnNative = "native:rife-4.25-ifnet";

static std::string lkF4(double v)
{
    char b[64];
    sprintf_s(b, "%.4f", v);
    return b;
}

// Everything the lookup and the cold-start check derive from a session's arguments, in ONE
// place: both must see the same names and sizes.
struct LkSession
{
    std::string backend, engDir, cacheDir;
    std::wstring engDirW;
    double imgScale = 1.0;    // live: the share the working size came from (the DLSS mode's, or --scale's)
    int cw = 0, ch = 0;       // the capture
    int presW = 0, presH = 0; // live: the presented rect (the capture aspect-fit into the canvas)
    int mode = 0;             // live: the DLSS mode the size came from (1 + kDlssModeName index, 0 = a share)
    int mw = 0, mh = 0;       // the model frame = the working size (live: liveWorkSize)
    int ph = 0, pw = 0;       // the /64 pad (SMV_LIVE_SAFEPAD=1 on the RIFE family)
    // gmfss: the half size (every net on the host's kernels: no engine)
    int hh = 0, hw = 0;
    // rife / rifedrba
    bool drba = false;
    // live rife: RIFE's motion frame (mmw x mmh, padded mph x mpw), the frame the IFNet and
    // its encode run at (iph x ipw = the motion pad, else the pictures' pad)
    int mmw = 0, mmh = 0, mph = 0, mpw = 0, iph = 0, ipw = 0;
    bool zpad = false; // live rife: the motion frame padded as RIFE's author pads (NativeRife::zeroPad)
};

// Create a folder and its missing parents. Users delete model_cache_safe_to_delete whenever
// they like (the name invites it), and every engine, JIT and cubin write below assumes the folder
// exists: a missing one used to fail each save silently and rebuild every session.
static void ensureDirW(const std::wstring& dir)
{
    if (dir.empty() || dir == L".")
        return;
    const DWORD a = GetFileAttributesW(dir.c_str());
    if (a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY))
        return;
    const size_t sl = dir.find_last_of(L"\\/");
    if (sl != std::wstring::npos && sl > 0 && dir[sl - 1] != L':')
        ensureDirW(dir.substr(0, sl));
    if (!CreateDirectoryW(dir.c_str(), nullptr) && GetLastError() != ERROR_ALREADY_EXISTS)
        LOG("cache: cannot create %ls (error %lu)\n", dir.c_str(), GetLastError());
}

// the rect a w x h frame takes in the outW x outH canvas: aspect-fit, even, centred
static void lkFitRect(int w, int h, int outW, int outH, int& dw, int& dh, int& x0, int& y0)
{
    if (outW == w && outH == h)
    {
        dw = w;
        dh = h;
        x0 = 0;
        y0 = 0;
        return;
    }
    const double s = (std::min)((double)outW / w, (double)outH / h);
    dw = (std::max)(2, lkRound(w * s) & ~1);
    dh = (std::max)(2, lkRound(h * s) & ~1);
    x0 = (outW - dw) / 2;
    y0 = (outH - dh) / 2; // both non-negative: the division floors
}

// NVIDIA's DLSS modes as plan.ts has them (DLSS_MODES): the working size's share of the output per kDlssModeName entry
static const double kDlssModeShare[5] = {1.0, 1.0 / 1.5, 1.0 / 1.724, 1.0 / 2, 1.0 / 3};

// DLSS 5's GPU work never completes on a 64 px wide frame (the GPU resets; 102 and 128 px wide run, and so does a
// 64 px high one): with DLSS 5 on, a working size below the presented rect is at least this wide, and a frame that is
// itself smaller than kNrMinWorkW x kNrMinWorkH runs without DLSS 5 (both routes' NR init)
static const int kNrMinWorkW = 128, kNrMinWorkH = 64;

// Live DLSS 4.5 at the working size runs its frames at the working size itself, so the working size keeps the shorter
// side of the canvas at least this long (the helpers' floor, above DLSS-G's own: 160x90 refused, 192x108 and 128x128
// run)
static const int kDlssgMinWork = 128;

// Live's working size (Restore, the resize, DLSS 5, Sharpen, RTX HDR and the model run at it, the fit
// takes it to the canvas after the model): the DLSS mode's share of the PRESENTED rect (the capture aspect-fit into
// the canvas W x H: the window itself, or the Fill rect), even, at most the presented rect a side, the share raised
// (aspect kept) to the largest minimum of the session's pipeline, so the capture is shrunk once and never below what a
// model needs: the shorter side 64 px, with DLSS 5 the width kNrMinWorkW, with DLSS 4.5 at the working size the
// canvas's shorter side kDlssgMinWork (*raised: 1 = the 64 px floor, 2 = DLSS 5's width, 3 = DLSS 4.5's minimum);
// capped at 3840x2160 keeping the aspect (offline's workPlan). DLAA = the presented rect itself (in window
// mode the capture, odd sizes kept). liveWorkSizeFor = the size a mode gives the presented presW x presH (mode = 1 +
// the kDlssModeName index, 0 = --scale's share; fDg = DLSS 4.5's minimum share, 0 = none); liveWorkSize = the
// session's, and the mode it came from.
static void liveWorkSizeFor(int mode, int presW, int presH, int& mw, int& mh, int* raised = nullptr, double fDg = 0.0)
{
    double f = 1.0;
    if (mode)
        f = kDlssModeShare[mode - 1];
    else
    {
        char sb[32];
        sprintf_s(sb, "%.2f", g_flowScale);
        f = (std::max)(0.01, (std::min)(1.0, strtod(sb, nullptr)));
    }
    mw = presW;
    mh = presH;
    if (raised)
        *raised = 0;
    if (f < 1.0)
    {
        const double fFloor = (std::max)(64.0 / presW, 64.0 / presH);
        const double fNr = g_dlssnr ? (double)kNrMinWorkW / presW : 0.0;
        const double fModel = (std::max)(fFloor, fNr);
        const double k = (std::max)((std::max)(f, fDg), fModel);
        if (raised && k > f)
            *raised = fDg >= fModel ? 3 : fNr >= fFloor ? 2 : 1;
        mw = (std::min)(presW, (std::max)(g_dlssnr ? kNrMinWorkW : 64, lkRound(presW * k) & ~1));
        mh = (std::min)(presH, (std::max)(64, lkRound(presH * k) & ~1));
    }
    if ((double)mw * mh > 3840.0 * 2160)
    {
        const double k = std::sqrt(3840.0 * 2160 / ((double)mw * mh));
        mw = (int)std::floor(mw * k) & ~1;
        mh = (int)std::floor(mh * k) & ~1;
    }
}
static void liveWorkSize(int cw, int ch, int& presW, int& presH, int& mode, int& mw, int& mh, int* raised = nullptr)
{
    const int outW = g_presW ? (int)g_presW : W ? (int)W : cw, outH = g_presW ? (int)g_presH : W ? (int)H : ch;
    int x0, y0;
    lkFitRect(cw, ch, outW, outH, presW, presH, x0, y0);
    mode = g_dlssMode;
    liveWorkSizeFor(mode, presW, presH, mw, mh, raised,
                    g_dlssgWorkFloor ? (double)kDlssgMinWork / (std::min)(outW, outH) : 0.0);
}

// The model cache sits in the app's top folder: the repo root in a dev tree, the folder holding
// SmoothMyVideo.exe in an install (whose engine folder is resources\engine). src/render/cache.ts
// derives the same folder for its stamp check and the cubin folder.
static std::string lkModelCacheDir(const std::string& engDir)
{
    auto parentOf = [](const std::string& p) {
        const size_t sl = p.find_last_of('\\');
        return sl == std::string::npos ? std::string(".") : p.substr(0, sl);
    };
    // engDir can be relative or end in `..` (the default script is <exe dir>\..\live_server.py):
    // resolve it the way the file system does before taking its parent
    const std::wstring engW = utf8ToWide(engDir);
    wchar_t full[4096];
    const DWORD n = GetFullPathNameW(engW.c_str(), (DWORD)(sizeof(full) / sizeof(full[0])), full, nullptr);
    std::string resolved = n > 0 && n < sizeof(full) / sizeof(full[0]) ? wideToUtf8(full) : engDir;
    while (resolved.size() > 3 && resolved.back() == '\\')
        resolved.pop_back();
    std::string top = parentOf(resolved);
    const size_t sl = top.find_last_of('\\');
    std::string last = sl == std::string::npos ? top : top.substr(sl + 1);
    for (auto& c : last)
        c = (char)tolower((unsigned char)c);
    if (last == "resources")
        top = parentOf(top);
    return top + "\\model_cache_safe_to_delete";
}

static bool lkSession(const std::wstring& script, const std::wstring& backendW, uint32_t capW, uint32_t capH,
                      LkSession& s, int padMult = 64)
{
    s = LkSession();
    s.backend = wideToUtf8(backendW);
    const size_t sl = script.find_last_of(L"\\/");
    if (sl == std::wstring::npos)
        return false;
    s.engDirW = script.substr(0, sl);
    for (auto& c : s.engDirW)
        if (c == L'/')
            c = L'\\'; // the dev harnesses pass forward slashes
    s.engDir = wideToUtf8(s.engDirW);
    const std::string envCache = lkEnv("SMV_TRT_CACHE");
    s.cacheDir = envCache.empty() ? lkModelCacheDir(s.engDir) : envCache;
    ensureDirW(utf8ToWide(s.cacheDir));
    s.cw = (int)capW;
    s.ch = (int)capH;
    if (g_offline)
    {
        // the offline host names its model size as the capture
        s.mw = s.cw;
        s.mh = s.ch;
    }
    else
    {
        liveWorkSize(s.cw, s.ch, s.presW, s.presH, s.mode, s.mw, s.mh);
        char sb[32];
        sprintf_s(sb, "%.2f", g_flowScale);
        s.imgScale = s.mode ? kDlssModeShare[s.mode - 1] : (std::max)(0.01, (std::min)(1.0, strtod(sb, nullptr)));
    }
    s.ph = (s.mh + padMult - 1) / padMult * padMult;
    s.pw = (s.mw + padMult - 1) / padMult * padMult;
    if (s.backend == "gmfss")
    {
        // a 64 px pad side would give gmflow's quarter-scale windows 1 token a side and its shifted blocks a roll of 0
        // (kGmMinPad): such a side runs at kGmMinPad (the packers replicate the edge into the pad)
        s.ph = (std::max)(s.ph, kGmMinPad);
        s.pw = (std::max)(s.pw, kGmMinPad);
        // every net on the host's kernels (feat_ext at the padded frame; metricnet, the GMFSS IFNet, fusionnet and
        // gmflow (backbone, `_m` + A, the scale-0 matching and propagation between them, the local correlation, B) at
        // the half): no engine to look up, build or warm
        s.hh = s.ph / 2;
        s.hw = s.pw / 2;
        return true;
    }
    s.drba = s.backend == "rifedrba";
    const bool safePad = lkEnv(g_offline ? "SMV_RIFE_SAFEPAD" : "SMV_LIVE_SAFEPAD") == "1";
    if (safePad)
    {
        s.pw = (std::max)(s.pw, 1152);
        s.ph = (std::max)(s.ph, 640);
    }
    s.iph = s.ph;
    s.ipw = s.pw;
    // RIFE's two domains live (rife / rifedrba): the IFNet, its encode and DRBA's block0 find
    // the motion on the finished picture at the working size, or at the capture size when the working
    // size enlarges it (Fill at DLAA: an enlarged picture costs RIFE its motion); the warp + blend runs
    // on the pictures at the working size. SMV_RIFE_TWO_DOMAIN=0: the IFNet reads the pictures.
    if (!g_offline && (s.backend == "rife" || s.drba) && lkEnv("SMV_RIFE_TWO_DOMAIN") != "0")
    {
        const bool up = s.mw > s.cw || s.mh > s.ch;
        s.mmw = up ? s.cw : s.mw;
        s.mmh = up ? s.ch : s.mh;
        // plain RIFE pads as its author does (zeros to a multiple of 128, NativeRife::zeroPad); DRBA keeps 64
        s.zpad = s.backend == "rife" && lkEnv("SMV_RIFE_AUTHOR_PAD") != "0";
        const int pm = s.zpad ? 128 : 64;
        s.mph = (s.mmh + pm - 1) / pm * pm;
        s.mpw = (s.mmw + pm - 1) / pm * pm;
        if (safePad)
        {
            s.mpw = (std::max)(s.mpw, 1152);
            s.mph = (std::max)(s.mph, 640);
        }
        s.iph = s.mph;
        s.ipw = s.mpw;
    }
    return true;
}

// the session's handoff answer (the NATIVE-PATH lines and the LIVE READY line nativeHandoff
// parses), or false
static bool lkBaseHandoff(const std::wstring& script, const std::wstring& backendW, uint32_t capW, uint32_t capH,
                          std::vector<std::string>& lines)
{
    if (g_offline)
        return false; // the offline host has its own lookups (lkOffline*)
    LkSession s;
    if (!lkSession(script, backendW, capW, capH, s))
        return false;
    const std::string& backend = s.backend;
    const bool geoOnly = backend == "nvof" || backend == "fruc" || backend == "fsrfg"; // no engine: geometry only
    if (!geoOnly && backend != "rife" && backend != "echo" && backend != "gmfss" && backend != "rifedrba")
        return false;
    const std::string& engDir = s.engDir;
    const std::string& cacheDir = s.cacheDir;
    const double imgScale = s.imgScale;
    const int cw = s.cw, ch = s.ch, mw = s.mw, mh = s.mh;
    const int outW = W ? (int)W : cw, outH = W ? (int)H : ch;
    // the presented rect comes from the capture, never the working size: each of its axes rounds to even on its own,
    // so its aspect can miss the capture's and a fit from it would leave black rows or columns
    // (DLSS 4.5 at the working size, g_presW: the presented rect's fit scaled to the smaller output; a picture that
    // fills the presented rect fills it whole, since lkFitRect's even rounding there leaves a line DXGI then stretches)
    auto fitRect = [&](int& dw, int& dh, int& x0, int& y0) {
        if (!g_presW)
        {
            lkFitRect(cw, ch, outW, outH, dw, dh, x0, y0);
            return;
        }
        int pw = 0, ph = 0, px = 0, py = 0;
        lkFitRect(cw, ch, (int)g_presW, (int)g_presH, pw, ph, px, py);
        const bool whole = px <= 1 && py <= 1;
        dw = whole ? outW : (std::max)(2, lkRound((double)pw * outW / g_presW) & ~1);
        dh = whole ? outH : (std::max)(2, lkRound((double)ph * outH / g_presH) & ~1);
        x0 = (outW - dw) / 2;
        y0 = (outH - dh) / 2;
    };
    auto geo = [&](int ph, int pw, int w, int h, int dw, int dh, int x0, int y0) -> std::string {
        char b[512];
        sprintf_s(b, "ph=%d pw=%d w=%d h=%d cw=%d ch=%d outw=%d outh=%d dw=%d dh=%d x0=%d y0=%d", ph, pw, w, h, cw, ch,
                  outW, outH, dw, dh, x0, y0);
        return b;
    };
    if (backend == "echo")
    {
        // the effects run at the working size like the models' (no /64 pad: no engine reads the frame)
        int dw, dh, x0, y0;
        fitRect(dw, dh, x0, y0);
        lines.push_back("NATIVE-PATH cache=" + cacheDir);
        lines.push_back("LIVE READY native=1 " + geo(mh, mw, mw, mh, dw, dh, x0, y0) + " scale=" + lkF4(imgScale) +
                        " batch=0 batchpad=0 effects=0 restore=0 engine=none fast=1");
        return true;
    }
    if (geoOnly)
    {
        const int gw = mw, gh = mh;
        int dw, dh, x0, y0;
        fitRect(dw, dh, x0, y0);
        const std::string tail =
            " scale=" + lkF4(imgScale) + " batch=0 batchpad=0 effects=0 restore=0 engine=" + backend + " fast=1";
        if (backend == "nvof")
        {
            // no /64 pad: the Optical Flow engine takes any size from 32x32
            lines.push_back("NATIVE-PATH cache=" + cacheDir);
            lines.push_back("LIVE READY native=1 " + geo(gh, gw, gw, gh, dw, dh, x0, y0) + tail);
            return true;
        }
        if (backend == "fsrfg")
        {
            // AMD FSR frame generation: our bridge folder (SMV_FSRFG_DIR, else engine\fsrfg), no /64 pad (FSR and the
            // Optical Flow engine take any size from 32x32); --xess-fg: Intel XeSS-FG's folder (lkXefgDir)
            std::string gdir;
            if (g_xessFg)
            {
                if (!lkXefgDir(engDir, gdir, "native"))
                    return false;
            }
            else
            {
                gdir = lkEnv("SMV_FSRFG_DIR");
                if (gdir.empty())
                    gdir = engDir + "\\fsrfg";
                if (!lkFile(gdir + "\\smv_fsrfg_bridge.dll"))
                {
                    LOG("native: AMD FSR frame generation needs smv_fsrfg_bridge.dll in %s\n", gdir.c_str());
                    return false;
                }
            }
            lines.push_back("NATIVE-PATH cache=" + cacheDir);
            lines.push_back("NATIVE-PATH fsrfg=" + gdir);
            lines.push_back("LIVE READY native=1 " + geo(gh, gw, gw, gh, dw, dh, x0, y0) + tail);
            return true;
        }
        // fruc: the bridge folder (SMV_NVOFFRUC_DIR, else engine\nvoffruc) and its three DLLs
        std::string fdir = lkEnv("SMV_NVOFFRUC_DIR");
        if (fdir.empty())
            fdir = engDir + "\\nvoffruc";
        for (const char* dll : {"nvoffruc_bridge.dll", "NvOFFRUC.dll", "cudart64_110.dll"})
            if (!lkFile(fdir + "\\" + dll))
            {
                LOG("native: Nvidia Smooth Motion needs %s in %s\n", dll, fdir.c_str());
                return false;
            }
        const int ph = (gh + 63) / 64 * 64, pw = (gw + 63) / 64 * 64; // never the safe pad
        lines.push_back("NATIVE-PATH cache=" + cacheDir);
        lines.push_back("NATIVE-PATH fruc=" + fdir);
        lines.push_back("LIVE READY native=1 " + geo(ph, pw, gw, gh, dw, dh, x0, y0) + tail);
        return true;
    }
    if (!nativeLoadDlls(script))
        return false;
    const int ph = s.ph, pw = s.pw;
    // rife / rifedrba: the IFNet on the host's kernels (one in-between frame a call, offline's class too): nothing to
    // find, build or warm; its kernel cubin goes into the model cache
    if (backend == "rife" || backend == "rifedrba")
    {
        int dw, dh, x0, y0;
        fitRect(dw, dh, x0, y0);
        lines.push_back(std::string("NATIVE-PATH ifnet=") + kIfnNative);
        lines.push_back("NATIVE-PATH cache=" + cacheDir);
        char tail[512], motion[96] = "";
        if (s.mph)
            sprintf_s(motion, " mph=%d mpw=%d motw=%d moth=%d%s", s.mph, s.mpw, s.mmw, s.mmh, s.zpad ? " zpad=1" : "");
        sprintf_s(tail, " batch=0 batchpad=0 xdtype=fp16 tdtype=fp16 fdtype=fp16 outdtype=fp16 effects=0 fast=1%s%s",
                  motion, s.drba ? " engine=drba lag=1" : "");
        lines.push_back("LIVE READY native=1 " + geo(ph, pw, mw, mh, dw, dh, x0, y0) + " scale=" + lkF4(imgScale) +
                        tail);
        return true;
    }
    if (backend == "gmfss")
    {
        // every GMFSS net runs on the host's kernels: the session's half size and the kernel cache, no engine
        int dw, dh, x0, y0;
        fitRect(dw, dh, x0, y0);
        lines.push_back("NATIVE-PATH cache=" + cacheDir);
        char tail[256];
        sprintf_s(tail, " batch=0 batchpad=0 hh=%d hw=%d effects=0 restore=0 engine=gmfss fast=1", s.hh, s.hw);
        lines.push_back("LIVE READY native=1 " + geo(ph, pw, mw, mh, dw, dh, x0, y0) + " scale=" + lkF4(imgScale) +
                        tail);
        return true;
    }
    return false;
}

static bool lkReadFile(const std::string& path, std::vector<char>& out);

static int lkField(const std::string& line, const char* key)
{
    const std::string k = std::string(" ") + key + "=";
    const size_t p = line.find(k);
    return p == std::string::npos ? -1 : atoi(line.c_str() + p + k.size());
}

// ---- Restore on the host's own kernels -------------------------------------------------------------------------
// The bundled Real-ESRGAN (realesr-animevideov3, BSD-3, engine\realesrgan\realesr-animevideov3.pkl) runs as
// k_restConv0, 16 x k_restConvMid and k_restConvLast. The handoff
// names it as restorePath = kRestNative; weights that are missing or not this model = no Restore (offline: the pass
// is dropped with its line).
static const char* const kRestNative = "native:realesr-animevideov3";
// a --restore-model file runs on the host's kernels in AnimeJaNai's SPAN-F3 form (64 features, ajNatLoad) only; any
// other graph = no Restore, the same way (ajNatWanted's line names why)
static const char* const kRestNativeAj = "native:animejanai";
// --restore-anime4k: Anime4K's Restore CNN UL on the Anime4K module's passes (kA4kRestWeights, no file to look up); a
// --restore-model on the same command line wins over it
static const char* const kRestNativeA4k = "native:anime4k";

// fp32 -> fp16 on the CPU, round to nearest even (torch's .half())
static uint16_t f2hRne(float f)
{
    uint32_t x = 0;
    memcpy(&x, &f, sizeof x);
    const uint32_t sign = (x >> 16) & 0x8000u, ax = x & 0x7fffffffu;
    if (ax >= 0x7f800000u)
        return (uint16_t)(sign | (ax > 0x7f800000u ? 0x7e00u : 0x7c00u));
    if (ax >= 0x477ff000u) // rounds to 65520 or more
        return (uint16_t)(sign | 0x7c00u);
    if (ax < 0x38800000u) // below 2^-14: a subnormal half = the value x 2^24 rounded to an even integer
    {
        float v = 0.0f;
        memcpy(&v, &ax, sizeof v);
        return (uint16_t)(sign | (uint32_t)std::nearbyint(v * 16777216.0f));
    }
    const uint32_t r = ax - 0x38000000u;
    return (uint16_t)(sign | ((r + 0xfffu + ((r >> 13) & 1u)) >> 13));
}

static float h2fHost(uint16_t h)
{
    const uint32_t e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
    float f = e == 0 ? std::ldexp((float)m, -24)
                     : (e == 31 ? (m ? NAN : INFINITY) : std::ldexp((float)(m | 0x400u), (int)e - 25));
    return (h & 0x8000u) ? -f : f;
}

// torch.save's file = a stored zip: <dir>/data.pkl, a protocol-2 pickle of the saved object (each tensor =
// torch._utils._rebuild_tensor_v2(its persistent storage ('storage', <dtype class>, key, location, numel), offset,
// size, stride, ...)), and <dir>/data/<key> = each storage's little-endian bytes. Read: the opcodes a state dict uses.
struct PkObj
{
    enum Kind
    {
        None,
        Int,
        Str,
        Tuple,
        List,
        Dict,
        Global,
        Storage,
        Tensor,
        Other
    } k = None;
    int64_t i = 0;
    std::string s;                         // Str; Global = "module.name"; Storage / Tensor = the storage key
    std::vector<std::shared_ptr<PkObj>> v; // Tuple / List items; Dict = key, value, key, value, ...
    int bytes = 0;                         // Storage / Tensor: 4 = fp32, 2 = fp16
    int64_t off = 0;
    std::vector<int64_t> size, stride;
};
using PkP = std::shared_ptr<PkObj>;

static PkP pkLoad(const std::string& b, std::string& why)
{
    std::vector<PkP> st;
    std::vector<size_t> marks;
    std::map<uint32_t, PkP> memo;
    size_t p = 0;
    auto mk = [](PkObj::Kind k) {
        auto o = std::make_shared<PkObj>();
        o->k = k;
        return o;
    };
    auto need = [&](size_t n) { return p + n <= b.size(); };
    auto le = [&](size_t n) {
        uint64_t v = 0;
        for (size_t j = 0; j < n; j++)
            v |= (uint64_t)(unsigned char)b[p + j] << (8 * j);
        p += n;
        return v;
    };
    auto popMark = [&](std::vector<PkP>& items) {
        if (marks.empty() || marks.back() > st.size())
            return false;
        items.assign(st.begin() + (ptrdiff_t)marks.back(), st.end());
        st.resize(marks.back());
        marks.pop_back();
        return true;
    };
    while (p < b.size())
    {
        const unsigned char op = (unsigned char)b[p++];
        bool ok = true;
        switch (op)
        {
        case 0x80: // PROTO
            ok = need(1);
            p += 1;
            break;
        case 0x95: // FRAME
            ok = need(8);
            p += 8;
            break;
        case '}':
            st.push_back(mk(PkObj::Dict));
            break;
        case ']':
            st.push_back(mk(PkObj::List));
            break;
        case ')':
            st.push_back(mk(PkObj::Tuple));
            break;
        case 'N':
            st.push_back(mk(PkObj::None));
            break;
        case 0x88:
        case 0x89: {
            auto o = mk(PkObj::Int);
            o->i = op == 0x88;
            st.push_back(o);
            break;
        }
        case 'K':
        case 'M':
        case 'J': {
            const size_t n = op == 'K' ? 1 : op == 'M' ? 2 : 4;
            if (!(ok = need(n)))
                break;
            const uint64_t u = le(n);
            auto o = mk(PkObj::Int);
            o->i = op == 'J' ? (int64_t)(int32_t)(uint32_t)u : (int64_t)u;
            st.push_back(o);
            break;
        }
        case 0x8a: // LONG1
        {
            if (!(ok = need(1)))
                break;
            const size_t n = (unsigned char)b[p++];
            if (!(ok = n <= 8 && need(n)))
                break;
            const uint64_t u = le(n);
            auto o = mk(PkObj::Int);
            o->i = (n && n < 8 && ((u >> (8 * n - 1)) & 1u)) ? (int64_t)(u | (~0ull << (8 * n))) : (int64_t)u;
            st.push_back(o);
            break;
        }
        case 'X':  // BINUNICODE
        case 0x8c: // SHORT_BINUNICODE
        {
            const size_t ln = op == 'X' ? 4 : 1;
            if (!(ok = need(ln)))
                break;
            const size_t n = (size_t)le(ln);
            if (!(ok = need(n)))
                break;
            auto o = mk(PkObj::Str);
            o->s = b.substr(p, n);
            p += n;
            st.push_back(o);
            break;
        }
        case 'c': // GLOBAL module\nname\n
        {
            const size_t e1 = b.find('\n', p);
            const size_t e2 = e1 == std::string::npos ? e1 : b.find('\n', e1 + 1);
            if (!(ok = e2 != std::string::npos))
                break;
            auto o = mk(PkObj::Global);
            o->s = b.substr(p, e1 - p) + "." + b.substr(e1 + 1, e2 - e1 - 1);
            p = e2 + 1;
            st.push_back(o);
            break;
        }
        case '(':
            marks.push_back(st.size());
            break;
        case 't': {
            std::vector<PkP> it;
            if (!(ok = popMark(it)))
                break;
            auto o = mk(PkObj::Tuple);
            o->v = it;
            st.push_back(o);
            break;
        }
        case 0x85: // TUPLE1 .. TUPLE3
        case 0x86:
        case 0x87: {
            const size_t n = (size_t)(op - 0x84);
            if (!(ok = st.size() >= n))
                break;
            auto o = mk(PkObj::Tuple);
            o->v.assign(st.end() - (ptrdiff_t)n, st.end());
            st.resize(st.size() - n);
            st.push_back(o);
            break;
        }
        case 'q': // BINPUT, LONG_BINPUT
        case 'r': {
            const size_t n = op == 'q' ? 1 : 4;
            if (!(ok = need(n) && !st.empty()))
                break;
            memo[(uint32_t)le(n)] = st.back();
            break;
        }
        case 0x94: // MEMOIZE
            if ((ok = !st.empty()))
                memo[(uint32_t)memo.size()] = st.back();
            break;
        case 'h': // BINGET, LONG_BINGET
        case 'j': {
            const size_t n = op == 'h' ? 1 : 4;
            if (!(ok = need(n)))
                break;
            const auto it = memo.find((uint32_t)le(n));
            if (!(ok = it != memo.end()))
                break;
            st.push_back(it->second);
            break;
        }
        case 'Q': // BINPERSID: ('storage', <dtype class>, key, location, numel)
        {
            if (!(ok = !st.empty()))
                break;
            const PkP id = st.back();
            st.pop_back();
            if (!(ok = id->k == PkObj::Tuple && id->v.size() >= 3 && id->v[1]->k == PkObj::Global &&
                       id->v[2]->k == PkObj::Str))
                break;
            auto o = mk(PkObj::Storage);
            o->s = id->v[2]->s;
            o->bytes = id->v[1]->s == "torch.FloatStorage" ? 4 : id->v[1]->s == "torch.HalfStorage" ? 2 : 0;
            st.push_back(o);
            break;
        }
        case 'R': // REDUCE
        {
            if (!(ok = st.size() >= 2))
                break;
            const PkP args = st.back();
            st.pop_back();
            const PkP fn = st.back();
            st.pop_back();
            PkP o;
            if (fn->k == PkObj::Global && fn->s == "torch._utils._rebuild_tensor_v2" && args->k == PkObj::Tuple &&
                args->v.size() >= 4 && args->v[0]->k == PkObj::Storage && args->v[2]->k == PkObj::Tuple &&
                args->v[3]->k == PkObj::Tuple)
            {
                o = mk(PkObj::Tensor);
                o->s = args->v[0]->s;
                o->bytes = args->v[0]->bytes;
                o->off = args->v[1]->i;
                for (const PkP& d : args->v[2]->v)
                    o->size.push_back(d->i);
                for (const PkP& d : args->v[3]->v)
                    o->stride.push_back(d->i);
            }
            else
                o = mk(fn->k == PkObj::Global && fn->s == "collections.OrderedDict" ? PkObj::Dict : PkObj::Other);
            st.push_back(o);
            break;
        }
        case 's': // SETITEM
        {
            if (!(ok = st.size() >= 3))
                break;
            const PkP val = st.back();
            st.pop_back();
            const PkP key = st.back();
            st.pop_back();
            st.back()->v.push_back(key);
            st.back()->v.push_back(val);
            break;
        }
        case 'u': // SETITEMS
        case 'e': // APPENDS
        {
            std::vector<PkP> it;
            if (!(ok = popMark(it) && !st.empty()))
                break;
            for (const PkP& o : it)
                st.back()->v.push_back(o);
            break;
        }
        case 'a': // APPEND
        {
            if (!(ok = st.size() >= 2))
                break;
            const PkP val = st.back();
            st.pop_back();
            st.back()->v.push_back(val);
            break;
        }
        case 'b': // BUILD: the state is not needed
            if ((ok = st.size() >= 2))
                st.pop_back();
            break;
        case '.':
            if (st.empty())
            {
                why = "an empty pickle";
                return nullptr;
            }
            return st.back();
        default: {
            char h[48];
            snprintf(h, sizeof h, "pickle opcode 0x%02x not read", op);
            why = h;
            return nullptr;
        }
        }
        if (!ok)
        {
            why = "a malformed pickle";
            return nullptr;
        }
    }
    why = "the pickle has no end";
    return nullptr;
}

// a stored zip's entries: name -> (data offset, size); false = not one (`why`: a compressed or zip64 entry, ...)
// the stored zip's directory over the file's n bytes at f: each entry's data offset and size by name
static bool pthZip(const unsigned char* f, size_t fn, std::map<std::string, std::pair<size_t, size_t>>& ent,
                   std::string& why)
{
    auto u16 = [&](size_t o) { return (uint32_t)f[o] | ((uint32_t)f[o + 1] << 8); };
    auto u32 = [&](size_t o) { return u16(o) | (u16(o + 2) << 16); };
    size_t eocd = SIZE_MAX;
    for (size_t o = fn >= 22 ? fn - 22 : 0; fn >= 22 && fn - o <= 65557; o--)
    {
        if (u32(o) == 0x06054b50u)
        {
            eocd = o;
            break;
        }
        if (o == 0)
            break;
    }
    if (eocd == SIZE_MAX)
    {
        why = "not a zip";
        return false;
    }
    const uint32_t n = u16(eocd + 10);
    size_t o = u32(eocd + 16);
    for (uint32_t i = 0; i < n; i++)
    {
        if (o + 46 > fn || u32(o) != 0x02014b50u)
        {
            why = "a bad zip directory";
            return false;
        }
        const uint32_t method = u16(o + 10), csize = u32(o + 20), usize = u32(o + 24), nl = u16(o + 28);
        const uint32_t el = u16(o + 30), cl = u16(o + 32), lho = u32(o + 42);
        if (o + 46 + nl > fn)
        {
            why = "a bad zip directory";
            return false;
        }
        const std::string name((const char*)f + o + 46, nl);
        if (method != 0 || csize != usize || csize == 0xffffffffu || lho == 0xffffffffu)
        {
            why = "a compressed or zip64 entry (" + name + ")";
            return false;
        }
        if ((size_t)lho + 30 > fn || u32(lho) != 0x04034b50u)
        {
            why = "a bad zip entry (" + name + ")";
            return false;
        }
        const size_t data = (size_t)lho + 30 + u16(lho + 26) + u16(lho + 28);
        if (data + csize > fn)
        {
            why = "a cut zip entry (" + name + ")";
            return false;
        }
        ent[name] = {data, (size_t)csize};
        o += 46 + (size_t)nl + el + cl;
    }
    return true;
}

// A torch.save file's state dict (BasicSR's {"params": ...} unwrapped) read in place: the file mapped read-only, each
// tensor in t a view of its stored fp32 values (an fp16 or misaligned tensor converted into own), its shape in S; every
// tensor contiguous. The views live as long as the PthFile.
struct PthFile
{
    HANDLE fh = INVALID_HANDLE_VALUE, mh = nullptr;
    const unsigned char* base = nullptr;
    std::map<std::string, std::span<const float>> t;
    std::map<std::string, std::vector<int64_t>> S;
    std::list<std::vector<float>> own;
    PthFile() = default;
    PthFile(const PthFile&) = delete;
    PthFile& operator=(const PthFile&) = delete;
    ~PthFile()
    {
        if (base)
            UnmapViewOfFile(base);
        if (mh)
            CloseHandle(mh);
        if (fh != INVALID_HANDLE_VALUE)
            CloseHandle(fh);
    }
};

// false = missing or malformed (`why`)
static bool pthStateDict(const std::wstring& path, PthFile& wf, std::string& why)
{
    wf.fh = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN,
                        nullptr);
    LARGE_INTEGER sz = {};
    if (wf.fh == INVALID_HANDLE_VALUE || !GetFileSizeEx(wf.fh, &sz))
    {
        why = "its weights " + wideToUtf8(path) + " are missing";
        return false;
    }
    if (sz.QuadPart < 22)
    {
        why = "not a zip";
        return false;
    }
    const size_t fn = (size_t)sz.QuadPart;
    wf.mh = CreateFileMappingW(wf.fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
    wf.base = wf.mh ? (const unsigned char*)MapViewOfFile(wf.mh, FILE_MAP_READ, 0, 0, 0) : nullptr;
    if (!wf.base)
    {
        why = "its weights " + wideToUtf8(path) + " could not be mapped";
        return false;
    }
    const unsigned char* f = wf.base;
    std::map<std::string, std::pair<size_t, size_t>> ent;
    if (!pthZip(f, fn, ent, why))
        return false;
    std::string pkl, dir;
    for (const auto& e : ent)
        if (e.first.size() > 9 && e.first.compare(e.first.size() - 9, 9, "/data.pkl") == 0)
        {
            dir = e.first.substr(0, e.first.size() - 9);
            pkl.assign((const char*)f + e.second.first, e.second.second);
        }
    if (pkl.empty())
    {
        why = "no data.pkl in its weights";
        return false;
    }
    const PkP top = pkLoad(pkl, why);
    if (!top)
        return false;
    PkP sd = top; // BasicSR's save = {"params": the state dict}
    if (top->k == PkObj::Dict)
        for (size_t i = 0; i + 1 < top->v.size(); i += 2)
            if (top->v[i]->k == PkObj::Str && top->v[i]->s == "params")
                sd = top->v[i + 1];
    if (sd->k != PkObj::Dict)
    {
        why = "no state dict in its weights";
        return false;
    }
    for (size_t i = 0; i + 1 < sd->v.size(); i += 2)
    {
        const PkObj& k = *sd->v[i];
        const PkObj& t = *sd->v[i + 1];
        if (k.k != PkObj::Str || t.k != PkObj::Tensor || (t.bytes != 4 && t.bytes != 2) ||
            t.size.size() != t.stride.size())
        {
            why = "a state dict entry that is not an fp32 / fp16 tensor";
            return false;
        }
        int64_t n = 1, want = 1;
        for (int64_t d : t.size)
            n *= d;
        for (size_t d = t.size.size(); d-- > 0;)
        {
            if (t.size[d] != 1 && t.stride[d] != want)
            {
                why = "a tensor that is not contiguous (" + k.s + ")";
                return false;
            }
            want *= t.size[d];
        }
        const auto e = ent.find(dir + "/data/" + t.s);
        if (n <= 0 || t.off < 0 || e == ent.end() || (uint64_t)(t.off + n) * (uint64_t)t.bytes > e->second.second)
        {
            why = "a tensor's storage is missing or short (" + k.s + ")";
            return false;
        }
        const unsigned char* src = f + e->second.first + (size_t)t.off * (size_t)t.bytes;
        if (t.bytes == 4 && ((uintptr_t)src & 3) == 0)
            wf.t[k.s] = std::span<const float>(reinterpret_cast<const float*>(src), (size_t)n);
        else
        {
            std::vector<float>& v = wf.own.emplace_back((size_t)n);
            for (size_t j = 0; j < (size_t)n; j++)
            {
                if (t.bytes == 4)
                    memcpy(&v[j], src + 4 * j, 4);
                else
                {
                    uint16_t h = 0;
                    memcpy(&h, src + 2 * j, 2);
                    v[j] = h2fHost(h);
                }
            }
            wf.t[k.s] = std::span<const float>(v);
        }
        wf.S[k.s] = t.size;
    }
    return true;
}

// The bundled model's weights for the kernels, in one block (restNatOff's offsets, 256-byte aligned): k_restConv0's
// w0[64][27] and b0 fp32 + its slopes fp16; per middle layer its fragment-order weights wf[tap][kc][q 4][lane][8 halves]
// (lane's B registers for n-tiles 2q, 2q + 1 at channels 16 kc .. + 15: n = 16 q + lane / 4 (+ 8), k = 2 (lane % 4)
// (+ 1, + 8, + 9)), bias and slopes fp16; the last layer's wf[tap][kc][q 3][lane][8] and bias fp16. Every value is
// first rounded to fp16, as the model runs (.half()). false = missing, or not this model (`why`).
static std::vector<uint16_t> restFragOf(std::span<const float> w, int nout, int cin, int coff, int taps);
static bool restNatLoad(std::vector<uint8_t>& blob, std::vector<size_t>& off, std::string& why)
{
    off.assign(3 + 16 * 3 + 2, 0);
    PthFile pf;
    if (!pthStateDict(g_nativeEngineDir + L"\\realesrgan\\realesr-animevideov3.pkl", pf, why))
        return false;
    auto& wts = pf.t;
    const auto& S = pf.S;
    // the form: body.0 conv 3 -> 64, body.1 PReLU 64, 16 x (conv 64 -> 64, PReLU 64), body.34 conv 64 -> 48, 3 x 3
    auto has = [&](const std::string& n, const std::vector<int64_t>& s) {
        const auto it = S.find(n);
        return it != S.end() && it->second == s;
    };
    bool form = wts.size() == 53 && has("body.0.weight", {64, 3, 3, 3}) && has("body.0.bias", {64}) &&
                has("body.1.weight", {64}) && has("body.34.weight", {48, 64, 3, 3}) && has("body.34.bias", {48});
    for (int L = 0; L < 16 && form; L++)
    {
        const std::string c = "body." + std::to_string(2 + 2 * L), a = "body." + std::to_string(3 + 2 * L);
        form = has(c + ".weight", {64, 64, 3, 3}) && has(c + ".bias", {64}) && has(a + ".weight", {64});
    }
    if (!form)
    {
        why = "its weights are not realesr-animevideov3's form";
        return false;
    }
    blob.clear();
    auto put = [&](const void* p, size_t n) {
        const size_t o = (blob.size() + 255) & ~(size_t)255;
        blob.resize(o + n);
        memcpy(blob.data() + o, p, n);
        return o;
    };
    auto halves = [](std::span<const float> v) {
        std::vector<uint16_t> h(v.size());
        for (size_t i = 0; i < v.size(); i++)
            h[i] = f2hRne(v[i]);
        return h;
    };
    // conv 0 keeps fp32 storage of the fp16-rounded values; every other weight goes through f2hRne once (rounding an
    // fp16 value again changes nothing)
    auto rounded = [](std::span<const float> v) {
        std::vector<float> r(v.begin(), v.end());
        for (float& x : r)
            x = h2fHost(f2hRne(x));
        return r;
    };
    auto fragOf = [](std::span<const float> w, int nout) { return restFragOf(w, nout, 64, 0, 9); };
    const std::vector<float> w0 = rounded(wts["body.0.weight"]), b0 = rounded(wts["body.0.bias"]);
    off[0] = put(w0.data(), w0.size() * sizeof(float));
    off[1] = put(b0.data(), b0.size() * sizeof(float));
    const std::vector<uint16_t> s0 = halves(wts["body.1.weight"]);
    off[2] = put(s0.data(), s0.size() * 2);
    for (int L = 0; L < 16; L++)
    {
        const std::string c = "body." + std::to_string(2 + 2 * L), a = "body." + std::to_string(3 + 2 * L);
        const std::vector<uint16_t> wf = fragOf(wts[c + ".weight"], 64), bs = halves(wts[c + ".bias"]),
                                    sl = halves(wts[a + ".weight"]);
        off[3 + 3 * L] = put(wf.data(), wf.size() * 2);
        off[4 + 3 * L] = put(bs.data(), bs.size() * 2);
        off[5 + 3 * L] = put(sl.data(), sl.size() * 2);
    }
    const std::vector<uint16_t> wl = fragOf(wts["body.34.weight"], 48), bl = halves(wts["body.34.bias"]);
    off[51] = put(wl.data(), wl.size() * 2);
    off[52] = put(bl.data(), bl.size() * 2);
    return true;
}

// a conv's fp16 fragment-order weights for the tensor-core kernels: w[o][cin][taps] (PyTorch's layout) -> per tap, per
// 16-channel step kc of the 64 channels from coff, per n-tile pair q of nout, per lane the 8 halves of its B registers
// (n = 16 q + lane / 4 (+ 8), k = 2 (lane % 4) (+ 1, + 8, + 9))
static std::vector<uint16_t> restFragOf(std::span<const float> w, int nout, int cin, int coff, int taps)
{
    const int nq = nout / 16;
    std::vector<uint16_t> fr((size_t)taps * 4 * nq * 32 * 8);
    size_t i = 0;
    for (int t = 0; t < taps; t++)
        for (int kc = 0; kc < 4; kc++)
            for (int q = 0; q < nq; q++)
                for (int l = 0; l < 32; l++)
                {
                    const int n0 = 16 * q + (l >> 2), c0 = coff + 16 * kc + 2 * (l & 3);
                    const int ns[8] = {n0, n0, n0, n0, n0 + 8, n0 + 8, n0 + 8, n0 + 8};
                    const int cs[8] = {c0, c0 + 1, c0 + 8, c0 + 9, c0, c0 + 1, c0 + 8, c0 + 9};
                    for (int j = 0; j < 8; j++)
                        fr[i++] = f2hRne(w[((size_t)ns[j] * cin + (size_t)cs[j]) * taps + (size_t)t]);
                }
    return fr;
}

// the native route for this session: the bundled model (no --restore-model), its weights readable
static bool restNatWanted()
{
    if (!g_restoreModel.empty())
        return false;
    static int ok = -1; // the file is read once a process
    if (ok < 0)
    {
        std::vector<uint8_t> blob;
        std::vector<size_t> off;
        std::string why;
        ok = restNatLoad(blob, off, why) ? 1 : 0;
        if (!ok)
            LOG("native: Restore unavailable (%s)\n", why.c_str());
    }
    return ok == 1;
}

// one protobuf field: its number, wire type and value (a varint / fixed one, or the bytes of a length-delimited one)
struct PbField
{
    uint32_t num = 0, wire = 0;
    uint64_t v = 0;
    const uint8_t* p = nullptr;
    size_t n = 0;
};
static bool pbVarint(const uint8_t*& p, const uint8_t* end, uint64_t& v)
{
    v = 0;
    for (int s = 0; s < 64 && p < end; s += 7)
    {
        const uint8_t b = *p++;
        v |= (uint64_t)(b & 0x7fu) << s;
        if (!(b & 0x80u))
            return true;
    }
    return false;
}
static bool pbNext(const uint8_t*& p, const uint8_t* end, PbField& f)
{
    uint64_t key = 0;
    if (p >= end || !pbVarint(p, end, key))
        return false;
    f.num = (uint32_t)(key >> 3);
    f.wire = (uint32_t)(key & 7u);
    f.v = 0;
    f.p = nullptr;
    f.n = 0;
    const size_t left = (size_t)(end - p);
    switch (f.wire)
    {
    case 0:
        return pbVarint(p, end, f.v);
    case 1:
        if (left < 8)
            return false;
        memcpy(&f.v, p, 8);
        p += 8;
        return true;
    case 5: {
        if (left < 4)
            return false;
        uint32_t u = 0;
        memcpy(&u, p, 4);
        f.v = u;
        p += 4;
        return true;
    }
    case 2: {
        uint64_t n = 0;
        if (!pbVarint(p, end, n) || n > (uint64_t)(end - p))
            return false;
        f.p = p;
        f.n = (size_t)n;
        p += f.n;
        return true;
    }
    default:
        return false;
    }
}

// the ONNX subset a conv net's weights and wiring need: the graph's nodes (op, inputs, outputs, int / string
// attributes) and its float / fp16 initializers as floats (shape constants: their dims only)
struct OnnxNode
{
    std::string op;
    std::vector<std::string> in, out;
    std::map<std::string, std::vector<int64_t>> ints;
    std::map<std::string, std::string> strs;
};
struct OnnxInit
{
    std::vector<int64_t> dims;
    std::vector<float> f;
};
struct OnnxGraph
{
    std::vector<OnnxNode> nodes;
    std::map<std::string, OnnxInit> inits;
    std::vector<std::string> inputs, outputs;
};

static bool onnxRead(const std::vector<char>& file, OnnxGraph& g, std::string& why)
{
    const uint8_t* p = (const uint8_t*)file.data();
    const uint8_t* end = p + file.size();
    PbField f;
    const uint8_t *gp = nullptr, *ge = nullptr;
    while (pbNext(p, end, f))
        if (f.num == 7 && f.wire == 2) // ModelProto.graph
        {
            gp = f.p;
            ge = f.p + f.n;
        }
    if (!gp)
    {
        why = "no ONNX graph";
        return false;
    }
    auto ints = [](const PbField& x, std::vector<int64_t>& v) { // repeated int64: packed or one at a time
        if (x.wire == 0)
            v.push_back((int64_t)x.v);
        else if (x.wire == 2)
        {
            const uint8_t* q = x.p;
            uint64_t u = 0;
            while (q < x.p + x.n && pbVarint(q, x.p + x.n, u))
                v.push_back((int64_t)u);
        }
    };
    auto name = [](const PbField& x) { return std::string((const char*)x.p, x.n); };
    while (pbNext(gp, ge, f))
    {
        if (f.wire != 2)
            continue;
        const uint8_t *q = f.p, *qe = f.p + f.n;
        PbField a;
        if (f.num == 1) // NodeProto
        {
            OnnxNode nd;
            while (pbNext(q, qe, a))
                if (a.num == 1 && a.wire == 2)
                    nd.in.push_back(name(a));
                else if (a.num == 2 && a.wire == 2)
                    nd.out.push_back(name(a));
                else if (a.num == 4 && a.wire == 2)
                    nd.op = name(a);
                else if (a.num == 5 && a.wire == 2) // AttributeProto: name 1, i 3, s 4, ints 8
                {
                    const uint8_t *r = a.p, *re = a.p + a.n;
                    PbField b;
                    std::string an;
                    std::vector<int64_t> iv;
                    std::string sv;
                    bool hasS = false;
                    while (pbNext(r, re, b))
                        if (b.num == 1 && b.wire == 2)
                            an = name(b);
                        else if (b.num == 3 && b.wire == 0)
                            iv.push_back((int64_t)b.v);
                        else if (b.num == 8)
                            ints(b, iv);
                        else if (b.num == 4 && b.wire == 2)
                        {
                            sv = name(b);
                            hasS = true;
                        }
                    if (hasS)
                        nd.strs[an] = sv;
                    else
                        nd.ints[an] = iv;
                }
            g.nodes.push_back(std::move(nd));
        }
        else if (f.num == 5) // TensorProto: dims 1, data_type 2, float_data 4, int32_data 5, name 8, raw_data 9,
        {                    // data_location 14 (1 = external)
            OnnxInit t;
            std::string tn;
            int64_t type = 0, loc = 0;
            const uint8_t* raw = nullptr;
            size_t rawN = 0;
            std::vector<int64_t> i32;
            std::vector<float> f32;
            while (pbNext(q, qe, a))
                if (a.num == 1)
                    ints(a, t.dims);
                else if (a.num == 2 && a.wire == 0)
                    type = (int64_t)a.v;
                else if (a.num == 8 && a.wire == 2)
                    tn = name(a);
                else if (a.num == 9 && a.wire == 2)
                {
                    raw = a.p;
                    rawN = a.n;
                }
                else if (a.num == 5)
                    ints(a, i32);
                else if (a.num == 4 && a.wire == 2)
                    for (size_t j = 0; j + 4 <= a.n; j += 4)
                    {
                        float v = 0.0f;
                        memcpy(&v, a.p + j, 4);
                        f32.push_back(v);
                    }
                else if (a.num == 4 && a.wire == 5)
                {
                    float v = 0.0f;
                    const uint32_t u = (uint32_t)a.v;
                    memcpy(&v, &u, 4);
                    f32.push_back(v);
                }
                else if (a.num == 14 && a.wire == 0)
                    loc = (int64_t)a.v;
            if (loc == 1)
            {
                why = "an initializer in an external file (" + tn + ")";
                return false;
            }
            if (type == 1) // FLOAT
            {
                if (raw)
                    for (size_t j = 0; j + 4 <= rawN; j += 4)
                    {
                        float v = 0.0f;
                        memcpy(&v, raw + j, 4);
                        t.f.push_back(v);
                    }
                else
                    t.f = f32;
            }
            else if (type == 10) // FLOAT16: raw bytes, or one value per int32_data entry
            {
                if (raw)
                    for (size_t j = 0; j + 2 <= rawN; j += 2)
                    {
                        uint16_t h = 0;
                        memcpy(&h, raw + j, 2);
                        t.f.push_back(h2fHost(h));
                    }
                else
                    for (int64_t v : i32)
                        t.f.push_back(h2fHost((uint16_t)v));
            }
            g.inits[tn] = std::move(t);
        }
        else if (f.num == 11 || f.num == 12) // ValueInfoProto: name 1
            while (pbNext(q, qe, a))
                if (a.num == 1 && a.wire == 2)
                    (f.num == 11 ? g.inputs : g.outputs).push_back(name(a));
    }
    return true;
}

// AnimeJaNai's SPAN-F3 net (64 features, n blocks) from a --restore-model ONNX, checked node by node along its data
// path: Pad (reflect) -> PixelUnshuffle(2) as Reshape / Transpose [0,1,3,5,2,4] / Reshape -> Conv 12 -> 64 3 x 3 (x0)
// -> per block Conv 3 x 3 -> Relu -> Conv 3 x 3 -> Relu -> Conv 3 x 3 (c3), Add(block input, c3), Conv 1 x 1 (c3) ->
// Tanh, Mul(Add, Tanh) -> Concat [x0, the last block, the first block, the last block's first Relu] -> Conv 1 x 1
// 256 -> 64 -> Conv 3 x 3 64 -> 48 -> DepthToSpace(4, CRD) -> Slices -> the output. Every conv: bias, stride 1, group 1,
// pad 1 (3 x 3) / 0 (1 x 1). Anything else = not this form (`why`). The weights for the kernels, in one block (off: the
// first conv's w[64][108] + b fp32; per block c1 / c2 / c3 fragment-order + bias + slope (0 = ReLU, 1 = none) fp16,
// the 1 x 1's fragments + bias fp32; the concat conv's fragments (4 sources) + bias fp32; the last conv's + bias fp16).
// AnimeJaNai's SPAN-F3 weights by layer, fp32 values (a weight stored as fp16 reads back exactly): conv_1 (64 x 12 x 3
// x 3), per block three 3 x 3 convs and the 1 x 1 attention conv, conv_cat (64 x 256 x 1 x 1), upsampler (48 x 64 x 3
// x 3); the .pkl the app keeps (--convert-restore-model) names them conv_1, block_K.c1 .. c3, block_K.att, conv_cat,
// upsampler (.weight / .bias)
struct AjT
{
    std::vector<float> w, b;
};
struct AjBlock
{
    AjT c[3], att;
};
struct AjWeights
{
    AjT first, cat, last;
    std::vector<AjBlock> blocks;
};

// the weights out of the author's ONNX, after the graph's form check
static bool ajFromOnnx(const std::string& path, AjWeights& aw, std::string& why)
{
    std::vector<char> file;
    if (!lkReadFile(path, file) || file.empty())
    {
        why = "its model file cannot be read";
        return false;
    }
    OnnxGraph g;
    if (!onnxRead(file, g, why))
        return false;
    std::map<std::string, int> prod;
    std::map<std::string, std::vector<int>> cons;
    for (int i = 0; i < (int)g.nodes.size(); i++)
    {
        for (const std::string& o : g.nodes[i].out)
            prod[o] = i;
        for (const std::string& x : g.nodes[i].in)
            cons[x].push_back(i);
    }
    auto bad = [&](const std::string& where) {
        why = "not AnimeJaNai's SPAN-F3 form (" + where + ")";
        return false;
    };
    auto node = [&](const std::string& t) -> const OnnxNode* {
        const auto it = prod.find(t);
        return it == prod.end() ? nullptr : &g.nodes[it->second];
    };
    auto dims = [&](const std::string& t) {
        const auto it = g.inits.find(t);
        return it == g.inits.end() ? std::vector<int64_t>() : it->second.dims;
    };
    // the one consumer of tensor t with op `op` (and, for a Conv, weight dims wd), or -1
    auto take = [&](const std::string& t, const char* op, const std::vector<int64_t>& wd) {
        int hit = -1, n = 0;
        for (int i : cons[t])
            if (g.nodes[i].op == op && (wd.empty() || (g.nodes[i].in.size() == 3 && dims(g.nodes[i].in[1]) == wd)))
            {
                hit = i;
                n++;
            }
        return n == 1 ? hit : -1;
    };
    auto convOk = [&](int i, int k) {
        const OnnxNode& c = g.nodes[i];
        auto at = [&](const char* a, const std::vector<int64_t>& want) {
            const auto it = c.ints.find(a);
            return it == c.ints.end() || it->second == want;
        };
        const std::vector<int64_t> pd(4, k == 3 ? 1 : 0);
        return c.in.size() == 3 && dims(c.in[2]).size() == 1 && at("strides", {1, 1}) && at("dilations", {1, 1}) &&
               at("group", {1}) && at("kernel_shape", {k, k}) && at("pads", pd) && c.out.size() == 1;
    };
    // the head: input -> Pad (reflect) -> Reshape -> Transpose [0,1,3,5,2,4] -> Reshape -> the first conv
    const std::vector<int64_t> w12 = {64, 12, 3, 3}, w64 = {64, 64, 3, 3}, w1 = {64, 64, 1, 1}, wc = {64, 256, 1, 1},
                               wl = {48, 64, 3, 3};
    int first = -1, nFirst = 0;
    for (int i = 0; i < (int)g.nodes.size(); i++)
        if (g.nodes[i].op == "Conv" && g.nodes[i].in.size() == 3 && dims(g.nodes[i].in[1]) == w12)
        {
            first = i;
            nFirst++;
        }
    if (nFirst != 1 || !convOk(first, 3) || g.inputs.empty() || g.outputs.size() != 1)
        return bad("the first conv");
    {
        const OnnxNode* r2 = node(g.nodes[first].in[0]);
        const OnnxNode* tr = r2 && r2->op == "Reshape" ? node(r2->in[0]) : nullptr;
        const OnnxNode* r1 = tr && tr->op == "Transpose" ? node(tr->in[0]) : nullptr;
        const OnnxNode* pad = r1 && r1->op == "Reshape" ? node(r1->in[0]) : nullptr;
        const auto perm = tr ? tr->ints.find("perm") : std::map<std::string, std::vector<int64_t>>::const_iterator();
        const auto mode = pad ? pad->strs.find("mode") : std::map<std::string, std::string>::const_iterator();
        if (!pad || pad->op != "Pad" || pad->in.empty() || pad->in[0] != g.inputs[0] || perm == tr->ints.end() ||
            perm->second != std::vector<int64_t>{0, 1, 3, 5, 2, 4} || mode == pad->strs.end() ||
            mode->second != "reflect")
            return bad("the pad and pixel unshuffle");
    }
    struct Block
    {
        int c[3], att;
        std::string relu1, out;
    };
    std::vector<Block> blocks;
    std::string cur = g.nodes[first].out[0];
    for (;;)
    {
        Block b{};
        b.c[0] = take(cur, "Conv", w64);
        if (b.c[0] < 0)
            break;
        const int r1 = take(g.nodes[b.c[0]].out[0], "Relu", {});
        b.c[1] = r1 < 0 ? -1 : take(g.nodes[r1].out[0], "Conv", w64);
        const int r2 = b.c[1] < 0 ? -1 : take(g.nodes[b.c[1]].out[0], "Relu", {});
        b.c[2] = r2 < 0 ? -1 : take(g.nodes[r2].out[0], "Conv", w64);
        if (b.c[2] < 0 || !convOk(b.c[0], 3) || !convOk(b.c[1], 3) || !convOk(b.c[2], 3))
            return bad("block " + std::to_string(blocks.size() + 1) + "'s convs");
        const std::string c3 = g.nodes[b.c[2]].out[0];
        const int add = take(c3, "Add", {});
        b.att = take(c3, "Conv", w1);
        const int th = b.att < 0 ? -1 : take(g.nodes[b.att].out[0], "Tanh", {});
        const int mul = add < 0 || th < 0 ? -1 : take(g.nodes[add].out[0], "Mul", {});
        if (add < 0 || th < 0 || mul < 0 || !convOk(b.att, 1) || cons[c3].size() != 2 ||
            !((g.nodes[add].in[0] == cur && g.nodes[add].in[1] == c3) ||
              (g.nodes[add].in[1] == cur && g.nodes[add].in[0] == c3)) ||
            std::find(g.nodes[mul].in.begin(), g.nodes[mul].in.end(), g.nodes[th].out[0]) == g.nodes[mul].in.end())
            return bad("block " + std::to_string(blocks.size() + 1) + "'s attention");
        b.relu1 = g.nodes[r1].out[0];
        b.out = cur = g.nodes[mul].out[0];
        blocks.push_back(b);
        if (blocks.size() > 16)
            return bad("more than 16 blocks");
    }
    if (blocks.empty())
        return bad("no block");
    const int cat = take(cur, "Concat", {});
    const std::vector<std::string> want = {g.nodes[first].out[0], blocks.back().out, blocks.front().out,
                                           blocks.back().relu1};
    const int cc = cat < 0 ? -1 : take(g.nodes[cat].out[0], "Conv", wc);
    const int last = cc < 0 ? -1 : take(g.nodes[cc].out[0], "Conv", wl);
    const int d2s = last < 0 ? -1 : take(g.nodes[last].out[0], "DepthToSpace", {});
    if (cat < 0 || g.nodes[cat].in != want || cc < 0 || last < 0 || d2s < 0 || !convOk(cc, 1) || !convOk(last, 3))
        return bad("the tail");
    {
        const OnnxNode& d = g.nodes[d2s];
        const auto bs = d.ints.find("blocksize");
        const auto md = d.strs.find("mode");
        if (bs == d.ints.end() || bs->second != std::vector<int64_t>{4} || (md != d.strs.end() && md->second != "CRD"))
            return bad("the depth to space");
        std::string t = g.outputs[0];
        for (int s = 0; s < 3 && node(t) && node(t)->op == "Slice"; s++)
            t = node(t)->in[0];
        if (t != d.out[0])
            return bad("the output's crop");
    }
    int nconv = 0, nrelu = 0, ntanh = 0;
    for (const OnnxNode& nd : g.nodes)
    {
        nconv += nd.op == "Conv";
        nrelu += nd.op == "Relu";
        ntanh += nd.op == "Tanh";
    }
    const int n = (int)blocks.size();
    if (nconv != 3 + 4 * n || nrelu != 2 * n || ntanh != n)
        return bad("extra layers");
    // the weights
    auto wOf = [&](int i) -> const std::vector<float>& { return g.inits[g.nodes[i].in[1]].f; };
    auto bOf = [&](int i) -> const std::vector<float>& { return g.inits[g.nodes[i].in[2]].f; };
    auto tOf = [&](int i) { return AjT{wOf(i), bOf(i)}; };
    aw.first = tOf(first);
    aw.blocks.clear();
    for (const Block& b : blocks)
        aw.blocks.push_back(AjBlock{{tOf(b.c[0]), tOf(b.c[1]), tOf(b.c[2])}, tOf(b.att)});
    aw.cat = tOf(cc);
    aw.last = tOf(last);
    return true;
}

// the kernels' block (offsets 256-byte aligned) from the weights; false = a weight of the wrong size (`why`)
static bool ajBlob(const AjWeights& aw, std::vector<uint8_t>& blob, std::vector<size_t>& off, int& nblocks,
                   std::string& why)
{
    const int n = (int)aw.blocks.size();
    auto bad = [&](const std::string& where) {
        why = "not AnimeJaNai's SPAN-F3 form (" + where + ")";
        return false;
    };
    if (n < 1 || n > 16)
        return bad("the blocks");
    blob.clear();
    off.assign((size_t)(6 + 11 * n), 0);
    auto put = [&](const void* p, size_t bytes) {
        const size_t o = (blob.size() + 255) & ~(size_t)255;
        blob.resize(o + bytes);
        memcpy(blob.data() + o, p, bytes);
        return o;
    };
    auto halves = [](const std::vector<float>& v) {
        std::vector<uint16_t> h(v.size());
        for (size_t i = 0; i < v.size(); i++)
            h[i] = f2hRne(v[i]);
        return h;
    };
    auto sizesOk = [&](const AjT& t, size_t nw, size_t nb) { return t.w.size() == nw && t.b.size() == nb; };
    if (!sizesOk(aw.first, 64 * 108, 64) || !sizesOk(aw.cat, 64 * 256, 64) || !sizesOk(aw.last, 48 * 64 * 9, 48))
        return bad("a weight's size");
    off[0] = put(aw.first.w.data(), aw.first.w.size() * sizeof(float));
    off[1] = put(aw.first.b.data(), 64 * sizeof(float));
    const std::vector<uint16_t> zeros(64, 0), ones(64, 0x3c00);
    for (int k = 0; k < n; k++)
    {
        const AjBlock& b = aw.blocks[(size_t)k];
        for (int j = 0; j < 3; j++)
        {
            if (!sizesOk(b.c[j], 64 * 64 * 9, 64))
                return bad("a weight's size");
            const std::vector<uint16_t> wf = restFragOf(b.c[j].w, 64, 64, 0, 9), bs = halves(b.c[j].b);
            off[(size_t)(2 + 11 * k + 3 * j)] = put(wf.data(), wf.size() * 2);
            off[(size_t)(3 + 11 * k + 3 * j)] = put(bs.data(), bs.size() * 2);
            off[(size_t)(4 + 11 * k + 3 * j)] = put(j < 2 ? zeros.data() : ones.data(), 64 * 2);
        }
        if (!sizesOk(b.att, 64 * 64, 64))
            return bad("a weight's size");
        const std::vector<uint16_t> wa = restFragOf(b.att.w, 64, 64, 0, 1);
        off[(size_t)(11 + 11 * k)] = put(wa.data(), wa.size() * 2);
        off[(size_t)(12 + 11 * k)] = put(b.att.b.data(), 64 * sizeof(float));
    }
    std::vector<uint16_t> wcat;
    for (int s = 0; s < 4; s++)
    {
        const std::vector<uint16_t> part = restFragOf(aw.cat.w, 64, 256, 64 * s, 1);
        wcat.insert(wcat.end(), part.begin(), part.end());
    }
    off[(size_t)(2 + 11 * n)] = put(wcat.data(), wcat.size() * 2);
    off[(size_t)(3 + 11 * n)] = put(aw.cat.b.data(), 64 * sizeof(float));
    const std::vector<uint16_t> wlf = restFragOf(aw.last.w, 48, 64, 0, 9), bl = halves(aw.last.b);
    off[(size_t)(4 + 11 * n)] = put(wlf.data(), wlf.size() * 2);
    off[(size_t)(5 + 11 * n)] = put(bl.data(), bl.size() * 2);
    nblocks = n;
    return true;
}

// the weights out of the .pkl the converter writes: every name with its exact shape, nothing else
static bool ajFromPkl(const std::wstring& path, AjWeights& aw, std::string& why)
{
    PthFile pf;
    if (!pthStateDict(path, pf, why))
        return false;
    auto bad = [&](const std::string& where) {
        why = "not AnimeJaNai's SPAN-F3 form (" + where + ")";
        return false;
    };
    auto get = [&](const std::string& n, const std::vector<int64_t>& wd, AjT& t) {
        const auto w = pf.S.find(n + ".weight"), b = pf.S.find(n + ".bias");
        if (w == pf.S.end() || b == pf.S.end() || w->second != wd || b->second != std::vector<int64_t>{wd[0]})
            return false;
        const std::span<const float> tw = pf.t.at(n + ".weight"), tb = pf.t.at(n + ".bias");
        t.w.assign(tw.begin(), tw.end());
        t.b.assign(tb.begin(), tb.end());
        return true;
    };
    const std::vector<int64_t> w64 = {64, 64, 3, 3};
    if (!get("conv_1", {64, 12, 3, 3}, aw.first))
        return bad("conv_1");
    aw.blocks.clear();
    for (int k = 1; k <= 16 && pf.S.count("block_" + std::to_string(k) + ".c1.weight"); k++)
    {
        const std::string p = "block_" + std::to_string(k);
        AjBlock b;
        if (!get(p + ".c1", w64, b.c[0]) || !get(p + ".c2", w64, b.c[1]) || !get(p + ".c3", w64, b.c[2]) ||
            !get(p + ".att", {64, 64, 1, 1}, b.att))
            return bad(p);
        aw.blocks.push_back(std::move(b));
    }
    if (aw.blocks.empty())
        return bad("no block");
    if (!get("conv_cat", {64, 256, 1, 1}, aw.cat) || !get("upsampler", {48, 64, 3, 3}, aw.last))
        return bad("the tail");
    if (pf.t.size() != 2 * (3 + 4 * aw.blocks.size()))
        return bad("extra tensors");
    return true;
}

// the .pkl (torch.save's zip, stored entries: archive/data.pkl + archive/data/<i>, version, byteorder): an OrderedDict
// of _rebuild_tensor_v2 tensors, each fp16 when every value round-trips exactly (the author's fp16 ONNX: all), else
// fp32; pthStateDict and torch.load read it; no timestamps, so one ONNX always gives the same bytes
static bool ajWritePkl(const AjWeights& aw, const std::wstring& path, std::string& why)
{
    struct Ent
    {
        std::string name;
        const std::vector<float>* v;
        std::vector<int64_t> dims;
    };
    std::vector<Ent> ts;
    auto add = [&](const std::string& n, const AjT& t, const std::vector<int64_t>& wd) {
        ts.push_back({n + ".weight", &t.w, wd});
        ts.push_back({n + ".bias", &t.b, {wd[0]}});
    };
    add("conv_1", aw.first, {64, 12, 3, 3});
    for (size_t k = 0; k < aw.blocks.size(); k++)
    {
        const std::string p = "block_" + std::to_string(k + 1);
        for (int j = 0; j < 3; j++)
            add(p + ".c" + std::to_string(j + 1), aw.blocks[k].c[j], {64, 64, 3, 3});
        add(p + ".att", aw.blocks[k].att, {64, 64, 1, 1});
    }
    add("conv_cat", aw.cat, {64, 256, 1, 1});
    add("upsampler", aw.last, {48, 64, 3, 3});
    std::string p = "\x80\x02";
    auto glob = [&](const char* m, const char* n) {
        p += 'c';
        p += m;
        p += '\n';
        p += n;
        p += '\n';
    };
    auto str = [&](const std::string& s) {
        const uint32_t l = (uint32_t)s.size();
        p += 'X';
        p.append((const char*)&l, 4);
        p += s;
    };
    auto i32 = [&](int64_t v) {
        const int32_t x = (int32_t)v;
        p += 'J';
        p.append((const char*)&x, 4);
    };
    glob("collections", "OrderedDict");
    p += ")R(";
    std::vector<std::string> data;
    for (size_t i = 0; i < ts.size(); i++)
    {
        const std::vector<float>& v = *ts[i].v;
        int64_t numel = 1;
        for (int64_t d : ts[i].dims)
            numel *= d;
        if (numel != (int64_t)v.size())
        {
            why = "a weight's size (" + ts[i].name + ")";
            return false;
        }
        bool half = true;
        for (float x : v)
            if (h2fHost(f2hRne(x)) != x)
            {
                half = false;
                break;
            }
        std::string d;
        if (half)
        {
            d.resize(v.size() * 2);
            for (size_t j = 0; j < v.size(); j++)
            {
                const uint16_t h = f2hRne(v[j]);
                memcpy(&d[2 * j], &h, 2);
            }
        }
        else
            d.assign((const char*)v.data(), v.size() * sizeof(float));
        data.push_back(std::move(d));
        str(ts[i].name);
        glob("torch._utils", "_rebuild_tensor_v2");
        p += "((";
        str("storage");
        glob("torch", half ? "HalfStorage" : "FloatStorage");
        str(std::to_string(i));
        str("cpu");
        i32(numel);
        p += "tQ";
        i32(0);
        p += '(';
        for (int64_t dd : ts[i].dims)
            i32(dd);
        p += "t(";
        std::vector<int64_t> stride(ts[i].dims.size(), 1);
        for (size_t s = stride.size(); s-- > 1;)
            stride[s - 1] = stride[s] * ts[i].dims[s];
        for (int64_t s : stride)
            i32(s);
        p += "t\x89";
        glob("collections", "OrderedDict");
        p += ")RtR";
    }
    p += "u.";
    std::vector<std::pair<std::string, std::string>> files = {
        {"archive/data.pkl", p}, {"archive/byteorder", "little"}, {"archive/version", "3\n"}};
    for (size_t i = 0; i < data.size(); i++)
        files.push_back({"archive/data/" + std::to_string(i), data[i]});
    static uint32_t crcTab[256];
    if (!crcTab[1])
        for (uint32_t i = 0; i < 256; i++)
        {
            uint32_t c = i;
            for (int k = 0; k < 8; k++)
                c = c & 1 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            crcTab[i] = c;
        }
    auto crc32 = [](const std::string& s) {
        uint32_t c = 0xFFFFFFFFu;
        for (unsigned char ch : s)
            c = crcTab[(c ^ ch) & 0xFF] ^ (c >> 8);
        return c ^ 0xFFFFFFFFu;
    };
    std::string z, cd;
    auto le = [](std::string& o, uint32_t v, int bytes) {
        for (int i = 0; i < bytes; i++)
            o += (char)((v >> (8 * i)) & 0xFF);
    };
    for (const auto& [name, body] : files)
    {
        const uint32_t at = (uint32_t)z.size(), crc = crc32(body), sz = (uint32_t)body.size();
        le(z, 0x04034b50u, 4);
        le(z, 20, 2);
        le(z, 0, 2);
        le(z, 0, 2);
        le(z, 0, 2);
        le(z, 0x21, 2);
        le(z, crc, 4);
        le(z, sz, 4);
        le(z, sz, 4);
        le(z, (uint32_t)name.size(), 2);
        le(z, 0, 2);
        z += name;
        z += body;
        le(cd, 0x02014b50u, 4);
        le(cd, 20, 2);
        le(cd, 20, 2);
        le(cd, 0, 2);
        le(cd, 0, 2);
        le(cd, 0, 2);
        le(cd, 0x21, 2);
        le(cd, crc, 4);
        le(cd, sz, 4);
        le(cd, sz, 4);
        le(cd, (uint32_t)name.size(), 2);
        le(cd, 0, 2);
        le(cd, 0, 2);
        le(cd, 0, 2);
        le(cd, 0, 2);
        le(cd, 0, 4);
        le(cd, at, 4);
        cd += name;
    }
    const uint32_t cdAt = (uint32_t)z.size();
    z += cd;
    le(z, 0x06054b50u, 4);
    le(z, 0, 2);
    le(z, 0, 2);
    le(z, (uint32_t)files.size(), 2);
    le(z, (uint32_t)files.size(), 2);
    le(z, (uint32_t)cd.size(), 4);
    le(z, cdAt, 4);
    le(z, 0, 2);
    HANDLE fh = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    DWORD wrote = 0;
    const bool ok = fh != INVALID_HANDLE_VALUE && WriteFile(fh, z.data(), (DWORD)z.size(), &wrote, nullptr) &&
                    wrote == (DWORD)z.size();
    if (fh != INVALID_HANDLE_VALUE)
        CloseHandle(fh);
    if (!ok)
    {
        DeleteFileW(path.c_str());
        why = "the .pkl could not be written (" + wideToUtf8(path) + ")";
    }
    return ok;
}

// a --restore-model file's weights for the kernels: the .pkl the app keeps, or the author's ONNX
static bool ajNatLoad(const std::string& path, std::vector<uint8_t>& blob, std::vector<size_t>& off, int& nblocks,
                      std::string& why)
{
    AjWeights aw;
    const bool pkl = path.size() > 4 && _stricmp(path.c_str() + path.size() - 4, ".pkl") == 0;
    return (pkl ? ajFromPkl(utf8ToWide(path), aw, why) : ajFromOnnx(path, aw, why)) &&
           ajBlob(aw, blob, off, nblocks, why);
}

// --convert-restore-model IN.onnx OUT.pkl: AnimeJaNai's ONNX after the form check as the .pkl the app keeps, read
// back and checked to give the kernels the ONNX's block byte for byte; 0 = written, 2 = refused (the line says why)
static int ajConvert(const std::string& in, const std::wstring& out)
{
    AjWeights aw, back;
    std::vector<uint8_t> blob, blob2;
    std::vector<size_t> off, off2;
    int n = 0, n2 = 0;
    std::string why;
    if (!ajFromOnnx(in, aw, why) || !ajBlob(aw, blob, off, n, why) || !ajWritePkl(aw, out, why))
    {
        printf("convert: refused: %s\n", why.c_str());
        return 2;
    }
    if (!ajFromPkl(out, back, why) || !ajBlob(back, blob2, off2, n2, why) || blob != blob2 || off != off2)
    {
        DeleteFileW(out.c_str());
        printf("convert: refused: the .pkl does not read back as the ONNX's weights%s%s\n", why.empty() ? "" : ": ",
               why.c_str());
        return 2;
    }
    printf("convert: %d blocks, %zu tensors -> %s\n", n, (size_t)(2 * (3 + 4 * n)), wideToUtf8(out).c_str());
    return 0;
}

// a --restore-model file on the host's kernels: AnimeJaNai's form, else no Restore (read once a path, the line once)
static bool ajNatWanted()
{
    if (g_restoreModel.empty())
        return false;
    static std::string path;
    static bool ok = false;
    if (path != g_restoreModel)
    {
        path = g_restoreModel;
        std::vector<uint8_t> blob;
        std::vector<size_t> off;
        std::string why;
        int nb = 0;
        ok = ajNatLoad(path, blob, off, nb, why);
        if (ok)
            LOG("native: the Restore model runs on the host's kernels (SPAN-F3, %d blocks)\n", nb);
        else
            LOG("native: Restore unavailable (%s; a --restore-model runs only in AnimeJaNai HD's SPAN-F3 form, the "
                "Balanced model's)\n",
                why.c_str());
    }
    return ok;
}

// The session's full handoff answer: the base answer plus, for live Restore (every backend:
// the host restores the captured frame before any model reads it), the Restore model on the host's kernels.
static bool nativeLocalHandoff(const std::wstring& script, const std::wstring& backendW, uint32_t capW, uint32_t capH,
                               std::vector<std::string>& lines)
{
    if (!lkBaseHandoff(script, backendW, capW, capH, lines))
        return false;
    if (!g_restore)
        return true;
    LkSession s;
    const int w = lkField(lines.back(), "w"), h = lkField(lines.back(), "h");
    if (!lkSession(script, backendW, capW, capH, s) || w <= 0 || h <= 0)
    {
        lines.clear();
        return false;
    }
    if (g_restoreA4k && g_restoreModel.empty()) // Anime4K's Restore CNN UL: the Anime4K module, nothing to look up
    {
        const std::string ready = lines.back();
        lines.pop_back();
        lines.push_back(std::string("NATIVE-PATH restore=") + kRestNativeA4k);
        lines.push_back(ready);
        return true;
    }
    if (g_restoreModel.empty()) // the bundled Real-ESRGAN on the host's kernels
    {
        if (!restNatWanted())
        {
            lines.clear();
            return false;
        }
        const std::string ready = lines.back();
        lines.pop_back();
        lines.push_back(std::string("NATIVE-PATH restore=") + kRestNative);
        lines.push_back(ready);
        return true;
    }
    if (!ajNatWanted()) // a --restore-model in another form: no Restore (ajNatWanted's line says why)
    {
        lines.clear();
        return false;
    }
    const std::string ready = lines.back();
    lines.pop_back();
    lines.push_back(std::string("NATIVE-PATH restore=") + kRestNativeAj);
    lines.push_back(ready);
    return true;
}

static bool lkReadFile(const std::string& path, std::vector<char>& out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return false;
    out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return !out.empty();
}

// Restore on the host's kernels for a session: the bundled Real-ESRGAN = kRestNative (false when its weights are
// unusable), a --restore-model file = kRestNativeAj in AnimeJaNai's SPAN-F3 form, any other graph = false
// (ajNatWanted's line says why). The live cold start and the offline host share it. true = ready, the path in rpath.
static bool lkEnsureRestore(const std::wstring& script, std::string& rpath)
{
    if (!nativeLoadDlls(script))
        return false;
    if (g_restoreA4k && !g_restoreModel.empty())
        LOG("native: --restore-anime4k ignored: the --restore-model holds Restore\n");
    if (g_restoreA4k && g_restoreModel.empty())
    {
        rpath = kRestNativeA4k; // the Anime4K module builds at the session's setup (nativeA4kRestSetup)
        return true;
    }
    if (g_restoreModel.empty() ? !restNatWanted() : !ajNatWanted())
        return false;
    rpath = g_restoreModel.empty() ? kRestNative : kRestNativeAj;
    return true;
}

// the cold start: every backend's nets run on the host's kernels (echo, nvof and fruc have none) and so does live
// Restore, so nothing is built; true = the session can start (worth a second lookup)
static bool nativeLocalBuild(const std::wstring& script, const std::wstring& backendW, uint32_t capW, uint32_t capH)
{
    if (g_offline)
        return false;
    std::vector<std::string> base;
    if (!lkBaseHandoff(script, backendW, capW, capH, base))
        return false;
    std::string rpath;
    return !g_restore || lkEnsureRestore(script, rpath);
}

// ---- offline RIFE -------------------------------------------------------------------------------------------------
// The offline host's IFNet runs on the host's kernels (kIfnNative, one in-between frame a call at every multiplier):
// the session's pad of the source only (/64, the author's /128 for plain RIFE), nothing to find or build.
struct OfflineEngines
{
    std::string ifnet;
    int ph = 0, pw = 0;
};

static bool lkOfflineRife(const std::wstring& script, int w, int h, OfflineEngines& o, int padMult = 64)
{
    LkSession s;
    if (!lkSession(script, L"rife", (uint32_t)w, (uint32_t)h, s, padMult))
        return false;
    if (!nativeLoadDlls(script))
        return false;
    o.ifnet = kIfnNative;
    o.ph = s.ph;
    o.pw = s.pw;
    LOG("offline: RIFE's IFNet on the host's kernels at %dx%d\n", s.pw, s.ph);
    return true;
}

// ---- offline GMFSS ----------------------------------------------------------------------------
// lkSession "gmfss"'s sizes (the padded frame, the half) for nr: every GMFSS net runs on the host's kernels, so there
// is no engine to find or build
static bool lkOfflineGmfss(const std::wstring& script, int w, int h, NativeRife& nr)
{
    LkSession s;
    if (!lkSession(script, L"gmfss", (uint32_t)w, (uint32_t)h, s))
        return false;
    if (!nativeLoadDlls(script))
        return false;
    nr.gmfss = true;
    nr.ph = s.ph;
    nr.pw = s.pw;
    nr.hh = s.hh;
    nr.hw = s.hw;
    LOG("offline: GMFSS at %dx%d on the host's kernels (no engine)\n", s.pw, s.ph);
    return true;
}

// ---- offline Smooth Motion ------------------------------------------------------------------
// no engine: the bridge folder the live handoff names (SMV_NVOFFRUC_DIR, else engine\nvoffruc) with its three DLLs; nativeFrucSetup loads the bridge from it
static bool lkOfflineFruc(const std::wstring& script, std::string& dir)
{
    dir = lkEnv("SMV_NVOFFRUC_DIR");
    if (dir.empty())
    {
        std::wstring eng = script.substr(0, script.find_last_of(L"\\/"));
        for (auto& c : eng)
            if (c == L'/')
                c = L'\\';
        dir = wideToUtf8(eng) + "\\nvoffruc";
    }
    for (const char* dll : {"nvoffruc_bridge.dll", "NvOFFRUC.dll", "cudart64_110.dll"})
        if (!lkFile(dir + "\\" + dll))
        {
            LOG("offline: Nvidia Smooth Motion needs %s in %s\n", dll, dir.c_str());
            return false;
        }
    return true;
}

// Intel XeSS-FG's folder (SMV_XESS_DIR, else engDir\xess) holds the bridge and Intel's two DLLs (the app's XeSS setup
// installs them); tag names the route in the refusal line
static bool lkXefgDir(const std::string& engDir, std::string& dir, const char* tag)
{
    dir = lkEnv("SMV_XESS_DIR");
    if (dir.empty())
        dir = engDir + "\\xess";
    for (const char* f : {"smv_xefg_bridge.dll", "libxess_fg.dll", "libxell.dll"})
        if (!lkFile(dir + "\\" + f))
        {
            LOG("%s: Intel XeSS frame generation unavailable (smv_xefg_bridge.dll and Intel's libxess_fg.dll + "
                "libxell.dll needed in %s; the app's XeSS setup installs them)\n",
                tag, dir.c_str());
            return false;
        }
    return true;
}

// offline AMD FSR frame generation (kind 0): the bridge folder the live handoff names (SMV_FSRFG_DIR, else
// engine\fsrfg); Intel XeSS-FG (kind 1): lkXefgDir's
static bool lkOfflineFsrfg(const std::wstring& script, std::string& dir, int kind)
{
    std::wstring eng = script.substr(0, script.find_last_of(L"\\/"));
    for (auto& c : eng)
        if (c == L'/')
            c = L'\\';
    if (kind == 1)
        return lkXefgDir(wideToUtf8(eng), dir, "offline");
    dir = lkEnv("SMV_FSRFG_DIR");
    if (dir.empty())
        dir = wideToUtf8(eng) + "\\fsrfg";
    if (!lkFile(dir + "\\smv_fsrfg_bridge.dll"))
    {
        LOG("offline: AMD FSR frame generation needs smv_fsrfg_bridge.dll in %s\n", dir.c_str());
        return false;
    }
    return true;
}

// The engine handoff: the session's geometry and model paths, from the host's own lookup
// (nativeLocalHandoff), after the cold-start check (nativeLocalBuild) on a worker thread when the
// lookup misses. Blocking (the caller is srv.start() on the loader thread or the early handoff
// thread). One check at a time: a new handoff waits for a running one first.
struct BgHandoff
{
    std::atomic<bool> pending{false};
};
static BgHandoff g_bgHandoff;
static bool nativeBuildPending()
{
    return g_bgHandoff.pending.load();
}
// the session is ending (stdin "stop", or the load-time resize); polled by the handoff wait
static bool nativeLoadAbort()
{
    return g_stopReq.load() || g_resizeReq.load();
}

static bool nativeHandoff(const std::wstring& script, const std::wstring& backend, uint32_t capW, uint32_t capH,
                          NativeRife& nr)
{
    // resident host: the same window as the previous session (same overlay size, capture
    // size, DLSS mode or share and HDR mode) gets the previous answer without a lookup,
    // as long as the engine files still exist (the user may empty the cache folder by hand);
    // live Restore rides in the key: its engine path is a fact of a restore session only; DLSS 5 too: it raises a
    // small working size (liveWorkSizeFor); --xess-fg too: its bridge folder is a fact of an fsrfg session
    wchar_t key[1024];
    swprintf_s(key, L"%s|%s|%.2f|%d|%u|%u|%u|%u|%d|%d|%d|%d|%d", script.c_str(), backend.c_str(), g_flowScale,
               g_dlssMode, W, H, capW, capH, g_hdr ? 1 : 0, g_restore ? 1 : 0, g_restoreA4k ? 1 : 0, g_dlssnr ? 1 : 0,
               g_xessFg ? 1 : 0);
    bool factsOk = g_resident && g_res.haveFacts && g_res.handoffKey == key;
    if (factsOk)
    {
        const NativeRife& f = g_res.facts;
        if (!f.gmfss)
            factsOk = f.ifnetPath == kIfnNative;
        factsOk = factsOk && (f.restorePath.empty() || f.restorePath == kRestNative || f.restorePath == kRestNativeAj ||
                              f.restorePath == kRestNativeA4k);
    }
    if (factsOk)
    {
        const NativeRife& f = g_res.facts;
        nr.ifnetPath = f.ifnetPath;
        nr.restorePath = f.restorePath;
        nr.restore = f.restore;
        nr.noEngine = f.noEngine;
        nr.cachePath = f.cachePath;
        nr.nvof = f.nvof;
        nr.fruc = f.fruc;
        nr.fsrfg = f.fsrfg;
        nr.fgKind = f.fgKind;
        nr.frucDir = f.frucDir;
        nr.drba = f.drba;
        nr.gmfss = f.gmfss;
        nr.hh = f.hh;
        nr.hw = f.hw;
        nr.ph = f.ph;
        nr.pw = f.pw;
        nr.w = f.w;
        nr.h = f.h;
        nr.cw = f.cw;
        nr.ch = f.ch;
        nr.dw = f.dw;
        nr.dh = f.dh;
        nr.x0 = f.x0;
        nr.y0 = f.y0;
        nr.mph = f.mph;
        nr.mpw = f.mpw;
        nr.mw = f.mw;
        nr.mh = f.mh;
        nr.zeroPad = f.zeroPad;
        nr.batchMax = f.batchMax;
        nr.identity = f.identity;
        nr.fitAa = f.fitAa;
        nr.hdr = g_hdr;
        nativeDeriveUpscale(nr); // from this session's own --upscale, never a stored fact
        LOG("native: engine handoff skipped, same window as the resident session\n");
        return true;
    }
    // an earlier session's build is still running: wait for it (its engines may be the ones
    // this session needs), unless this session is ending too
    if (g_bgHandoff.pending.load())
    {
        LOG("native: an earlier engine build is still running, waiting for it\n");
        while (g_bgHandoff.pending.load())
        {
            if (nativeLoadAbort())
            {
                LOG("native: session ended while waiting for the earlier engine build\n");
                return false;
            }
            Sleep(50);
        }
    }
    std::string all;
    bool ready = false;
    // one handoff line of the lookup's answer
    auto onLine = [&](const std::string& line) {
        if (line.rfind("NATIVE-PATH ifnet=", 0) == 0)
            nr.ifnetPath = line.substr(18);
        else if (line.rfind("NATIVE-PATH restore=", 0) == 0)
            nr.restorePath = line.substr(20);
        else if (line.rfind("NATIVE-PATH cache=", 0) == 0)
            nr.cachePath = line.substr(18);
        else if (line.rfind("NATIVE-PATH fruc=", 0) == 0)
            nr.frucDir = line.substr(17);
        else if (line.rfind("NATIVE-PATH fsrfg=", 0) == 0)
            nr.frucDir = line.substr(18);
        else if (line.rfind("LIVE READY", 0) == 0)
        {
            all = line;
            ready = true;
        }
    };
    {
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<std::string> local;
        bool hit = nativeLocalHandoff(script, backend, capW, capH, local);
        if (!hit)
        {
            // the lookup missed: the cold-start check (nativeLocalBuild) on a worker thread; a session
            // ending meanwhile leaves it running under g_bgHandoff (see the comment above)
            auto state = std::make_shared<std::atomic<int>>(0); // 0 running, 1 built, 2 not
            g_bgHandoff.pending.store(true);
            std::thread([state, script, backend, capW, capH] {
                const bool ok = nativeLocalBuild(script, backend, capW, capH);
                state->store(ok ? 1 : 2);
                g_bgHandoff.pending.store(false);
            }).detach();
            while (state->load() == 0)
            {
                if (nativeLoadAbort())
                {
                    LOG("native: session ended during the host engine build, the build continues in the background\n");
                    return false;
                }
                Sleep(20);
            }
            local.clear();
            hit = state->load() == 1 && nativeLocalHandoff(script, backend, capW, capH, local);
        }
        if (hit)
        {
            for (const auto& l : local)
            {
                onLine(l);
                if (g_handoffDump)
                    LOG("handoff-line %s\n", l.c_str());
            }
            LOG("native: warm engines found by the host (%.2fs)\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
    }
    if (!ready)
    {
        // the lookup and the cold-start check cover every product case; what is left (a lookup
        // that failed: its own line says why)
        // ends the session here
        LOG("native: the host could not find or build the engines for this session (see the lines above; "
            "the engine folder's weights and engine\\gpu_runtime must be intact)\n");
        return false;
    }
    LOG("%s\n", all.c_str());
    auto num = [&](const char* key, int dflt) -> int {
        const std::string k = std::string(" ") + key + "=";
        const size_t p = all.find(k);
        return p == std::string::npos ? dflt : atoi(all.c_str() + p + k.size());
    };
    nr.ph = num("ph", 0);
    nr.pw = num("pw", 0);
    nr.w = num("w", 0);
    nr.h = num("h", 0);
    nr.cw = num("cw", 0);
    nr.ch = num("ch", 0);
    nr.dw = num("dw", 0);
    nr.dh = num("dh", 0);
    nr.x0 = num("x0", 0);
    nr.y0 = num("y0", 0);
    // RIFE's two domains: the motion frame's pad and content size (absent = the IFNet reads the pictures)
    nr.mph = num("mph", 0);
    nr.mpw = num("mpw", 0);
    nr.mw = num("motw", 0);
    nr.mh = num("moth", 0);
    nr.zeroPad = nr.mph && num("zpad", 0) == 1;
    if (nr.mph && (nr.mpw <= 0 || nr.mw <= 0 || nr.mh <= 0 || nr.mw > nr.mpw || nr.mh > nr.mph))
    {
        LOG("native: handoff line names an incomplete motion frame (mph / mpw / motw / moth)\n");
        return false;
    }
    nr.batchMax = num("batch", 0);
    if (nr.batchMax < 1)
        nr.batchMax = 1;
    // no-engine mode: the echo handoff answers `engine=none` (geometry only, no paths); the
    // flow factor and the engine paths are not facts of such a session
    nr.noEngine = all.find(" engine=none") != std::string::npos;
    if (nr.noEngine && nr.cachePath.empty())
    {
        LOG("native: no-engine handoff named no cache folder\n");
        return false;
    }
    // the nvof model: the same geometry-only answer with `engine=nvof` (the
    // cache folder holds the kernel cubin)
    nr.nvof = all.find(" engine=nvof") != std::string::npos;
    if (nr.nvof && nr.cachePath.empty())
    {
        LOG("native: nvof handoff named no cache folder\n");
        return false;
    }
    // Smooth Motion: the same geometry-only shape, plus the bridge folder
    nr.fruc = all.find(" engine=fruc") != std::string::npos;
    if (nr.fruc && (nr.cachePath.empty() || nr.frucDir.empty()))
    {
        LOG("native: fruc handoff named no cache or bridge folder\n");
        return false;
    }
    // AMD FSR frame generation: the same shape with its own bridge, on the FRUC route (NativeRife's fsrfg notes)
    nr.fsrfg = all.find(" engine=fsrfg") != std::string::npos;
    if (nr.fsrfg && (nr.cachePath.empty() || nr.frucDir.empty()))
    {
        LOG("native: fsrfg handoff named no cache or bridge folder\n");
        return false;
    }
    nr.fgKind = nr.fsrfg && g_xessFg ? 1 : 0; // the folder above is XeSS-FG's then (nativeLocalHandoff)
    nr.fruc = nr.fruc || nr.fsrfg;
    // GMFSS: the half-frame grid and the cache folder for the kernel cubins (no engine, no jit path)
    nr.gmfss = all.find(" engine=gmfss") != std::string::npos;
    if (nr.gmfss)
    {
        nr.hh = num("hh", 0);
        nr.hw = num("hw", 0);
        if (nr.hh <= 0 || nr.hw <= 0 || nr.cachePath.empty())
        {
            LOG("native: gmfss handoff line incomplete (hh / hw / cache)\n");
            return false;
        }
    }
    // native DRBA: the RIFE handoff (block0 runs on the host's kernels); the lag is one capture
    nr.drba = all.find(" engine=drba") != std::string::npos;
    if (nr.drba && num("lag", 0) != 1)
    {
        LOG("native: drba handoff named no lag=1\n");
        return false;
    }
    const int outW = num("outw", 0), outH = num("outh", 0);
    nr.identity = (outW == nr.w && outH == nr.h);
    // every fit that changes the size runs as the separable Lanczos pair (the sum of the two axes'
    // taps a pixel); decided here so the early helper knows to run nativeRtxInit
    // (its staging buffers) before any effect flag is looked at. Upscale to re-derives it
    // against the internal render size in nativeDeriveUpscale.
    nr.fitAa = nr.dw != nr.w || nr.dh != nr.h;
    nr.hdr = g_hdr;
    nativeDeriveUpscale(nr);
    if (nr.ph <= 0 || nr.pw <= 0 || nr.w <= 0 || nr.h <= 0 || nr.cw <= 0 || nr.ch <= 0 ||
        (!nr.noEngine && !nr.gmfss && !nr.nvof && !nr.fruc && nr.ifnetPath.empty()))
    {
        LOG("native: handoff line incomplete\n");
        return false;
    }
    if (num("effects", 0))
    {
        LOG("native: live effects are on, phase 1 has no sharpen or VSR path\n");
        return false;
    }
    // live Restore: the session asked for it, so the handoff must have named its model
    nr.restore = !nr.restorePath.empty();
    if (g_restore && !nr.restore)
    {
        LOG("native: Restore is on but the handoff named no Restore model\n");
        return false;
    }
    // a downscaling fit runs natively (k_fitAaH / k_fitAaV, nativeRtxInit)
    if ((uint32_t)nr.cw != capW || (uint32_t)nr.ch != capH)
    {
        LOG("native: handoff capture size %dx%d != %ux%u\n", nr.cw, nr.ch, capW, capH);
        return false;
    }
    if (g_resident)
    {
        NativeRife& f = g_res.facts;
        f.ifnetPath = nr.ifnetPath;
        f.restorePath = nr.restorePath;
        f.restore = nr.restore;
        f.noEngine = nr.noEngine;
        f.cachePath = nr.cachePath;
        f.nvof = nr.nvof;
        f.fruc = nr.fruc;
        f.fsrfg = nr.fsrfg;
        f.fgKind = nr.fgKind;
        f.frucDir = nr.frucDir;
        f.drba = nr.drba;
        f.gmfss = nr.gmfss;
        f.hh = nr.hh;
        f.hw = nr.hw;
        f.ph = nr.ph;
        f.pw = nr.pw;
        f.w = nr.w;
        f.h = nr.h;
        f.cw = nr.cw;
        f.ch = nr.ch;
        f.dw = nr.dw;
        f.dh = nr.dh;
        f.x0 = nr.x0;
        f.y0 = nr.y0;
        f.mph = nr.mph;
        f.mpw = nr.mpw;
        f.mw = nr.mw;
        f.mh = nr.mh;
        f.zeroPad = nr.zeroPad;
        f.batchMax = nr.batchMax;
        f.identity = nr.identity;
        f.fitAa = nr.fitAa;
        g_res.handoffKey = key;
        g_res.haveFacts = true;
    }
    return true;
}

// RIFE 4.25's encode weights (the literals encKernelText writes), fp32 bits in PyTorch's layouts: cnn0 [16][3][3][3]
// + bias, cnn1 / cnn2 [16][16][3][3] + bias, cnn3 [16][kRifeFeatC][4][4] + bias = the module.encode.* tensors of
// engine\rife\flownet.pkl (its md5 below); a new checkpoint brings a new table
constexpr int kRifeEncWN = 5088 + 16 * kRifeFeatC * 16 + kRifeFeatC;
// flownet.pkl md5 a4715e33279bed4b7dab5bdafd2f138c
static const uint32_t kRifeEncW[kRifeEncWN] = {
    0x3dd5f696, 0x3bec66b3, 0xbd2ec985, 0x3ce7f93b, 0xbe0922f4, 0xbd4d8f03, 0x3db26d6f, 0xbd8feca5, 0x3d8f072f,
    0x3e31354a, 0x3d2c7be6, 0xbcd8f39f, 0x3c67feb3, 0xbe4809d2, 0xbdad23cc, 0x3db983d1, 0xbdad181a, 0x3d99421a,
    0x3e21928b, 0x3d9d1a73, 0x39052d1e, 0xbc6d0770, 0xbe23621a, 0xbda38d95, 0x3d1b1a51, 0xbd99d256, 0x3d5bb2b6,
    0xbe11c2ab, 0xbbf0de07, 0xbda86f58, 0x3dcf5e1d, 0x3e780d67, 0x3db71c13, 0xbda51e7f, 0xbdf9cca6, 0xbbb56310,
    0xbe14d870, 0x3b94c667, 0xbded8f18, 0x3dfe6ba8, 0x3e92da11, 0x3daed482, 0xbd92ee65, 0xbdee337c, 0xbd50b3b2,
    0xbe0de954, 0x3bc7ac3e, 0xbdb9c14a, 0x3da5cedc, 0x3e69af94, 0x3d838021, 0xbd2c3ba8, 0xbdba8c29, 0xbcc729f0,
    0x3cd63417, 0x3c1036f9, 0x3d5d69fb, 0xbc02510b, 0xbd4d9210, 0xbd298701, 0xbc60ff4f, 0xbd40b773, 0xbd685725,
    0x3cbfc6f8, 0xbcb0e0d2, 0x3d2d98cc, 0xbc297b20, 0xbdbf0e76, 0xbdab7682, 0x3cfac825, 0xbd1ad87d, 0xbd322707,
    0xbb816b44, 0xbd82f0f6, 0xbd055883, 0xbc12ec13, 0xbd9b0f3b, 0xbdb851be, 0x3d8bd039, 0x3ce4112a, 0x3afb6b17,
    0x3e5e5e3a, 0x3c531f55, 0x3dab7ec3, 0x3ca5501e, 0xbe8ad2d2, 0xbd73f211, 0xbd714b21, 0xbd9b129f, 0x3e14eeb4,
    0x3e6553ba, 0xbd86072d, 0x3d387ade, 0x3d2545ad, 0xbeb28ae6, 0xbd8029f3, 0x3c3e3552, 0xbda708b2, 0x3e574de4,
    0x3dd73cf1, 0xbdd18fb0, 0xbcc59c0d, 0x3d730422, 0xbe673997, 0xbcb0a8fc, 0x3cbb12a3, 0xbbef5b92, 0x3e68373d,
    0x3d698749, 0xbbab6f0f, 0xbdafcc67, 0x3dc5ad21, 0xbab09994, 0xbe126f04, 0x3de1dcfd, 0x3d2710a3, 0xbd77d4fb,
    0x3d87f429, 0xbc5f64e6, 0xbe06a7d8, 0x3df7d4fd, 0xbc610123, 0xbe467a2d, 0x3e301411, 0x3d8de93b, 0xbdad1666,
    0x3d0e79be, 0xbc48a460, 0xbdab91fe, 0x3d9ec855, 0xbca6d261, 0xbe058da8, 0x3deddd94, 0x3d799fb9, 0xbd073c52,
    0xbda0c726, 0xbc930f5f, 0x3db8096c, 0xbdc9c79f, 0x3b8b0e75, 0x3e089fc2, 0xbda43ab6, 0xbca42734, 0x3d9dc6fc,
    0xbde97ef4, 0xbc83d124, 0x3de7c09d, 0xbdffc33d, 0x3c36210e, 0x3e3aacea, 0xbe00900c, 0xbd1d4483, 0x3dcb602b,
    0xbdacc08b, 0xbc7bac1e, 0x3db1ec46, 0xbd8d4254, 0x3c4aa198, 0x3e0de8cd, 0xbdb51244, 0xbd166583, 0x3d937a86,
    0xbdbfe6ac, 0x3d99639e, 0xbd181c07, 0xbdc23578, 0x3e6a266b, 0xbda4ae97, 0xbd5b0bf7, 0x3e6e09b7, 0xbe1e8476,
    0xbe1e96b3, 0x3dc0d54a, 0xbda01272, 0xbdc6f61f, 0x3ea0b9f9, 0xbd76f3c8, 0xbda4f76e, 0x3e90f086, 0xbe4a788d,
    0xbdc80523, 0x3d7f3e00, 0xbd336665, 0xbd993989, 0x3e4bbc10, 0xbd61e7cc, 0xbd848402, 0x3e51697c, 0xbe19e920,
    0xbda6f4ec, 0x3bfa092a, 0x3d98edd5, 0xbdb257be, 0xbe4e9f7b, 0x3c577a2e, 0x3e80b676, 0x3ca8aec1, 0x3b974127,
    0xbd1d4736, 0x3d495722, 0x3e365eb7, 0xbdd2ad41, 0xbe887714, 0x3c8ce115, 0x3e6fe060, 0xbd01a874, 0xbd1db42a,
    0x3cd3146a, 0x3ddef199, 0x3e462000, 0xbda63fde, 0xbe4e568c, 0x3c35dac4, 0x3dfe0449, 0xbdad793f, 0xbdc12432,
    0x3c59dba3, 0x3dbf129a, 0x3e01ff66, 0xbda7ae48, 0xbc32d551, 0x3d4411bf, 0xbdf07448, 0xbd866bfe, 0xbb9fa518,
    0x3cf8dc51, 0x3e014f5a, 0x3e2a8e75, 0xbdbdece1, 0xbb641c9d, 0x3d99e8ae, 0xbe1be907, 0xbddb818d, 0xbccc0cae,
    0x3caf68da, 0x3db9d54a, 0x3de6bb30, 0xbd889294, 0xbb46e0b9, 0x3d307860, 0xbde7957e, 0xbd8bf107, 0xbcbe9207,
    0xbda8ba97, 0x3aded85e, 0xbc6029cc, 0xbe3f2352, 0x3c4128b6, 0x3ca32b3e, 0xbc8923a4, 0x3e21e1fc, 0x3dce9550,
    0xbdff3486, 0xbb9251ce, 0xbd004b6c, 0xbe4edd61, 0x3d2e414e, 0x3d546daf, 0xbcc3ef57, 0x3e39793d, 0x3df32578,
    0xbda945a3, 0xba14a43f, 0xbcc0ac6c, 0xbe0c452c, 0x3d20a977, 0x3d20242d, 0xbd209e9e, 0x3e027a62, 0x3da18cb2,
    0x3de34ace, 0x3deed649, 0x3d67598f, 0x3d1bf965, 0x3c077d03, 0xbcc3f918, 0xbd560a5e, 0xbdfb40db, 0xbde4d54a,
    0x3e0767d1, 0x3e02a805, 0x3d821ec5, 0x3d55f73b, 0x3c138179, 0xbd4bc8a8, 0xbd7ae8df, 0xbe20f827, 0xbe16fc45,
    0x3dc71c78, 0x3dccfe2f, 0x3d22af91, 0x3d40a7d2, 0xbb185691, 0xbd42e1e1, 0xbcf61072, 0xbdb73e06, 0xbdcac192,
    0x3cf869a4, 0x3d59b170, 0x3dcac968, 0xbd8bc940, 0xbd56be11, 0x3dcf650b, 0xbd5f047e, 0xbe310979, 0x3d77fa22,
    0x3d6e3993, 0x3d9d1f3e, 0x3e13fd65, 0xbda28169, 0xbda5335e, 0x3e19d586, 0xbde47779, 0xbe627849, 0x3d78d6fb,
    0x3d1956de, 0x3d2e5fed, 0x3d71b4f5, 0xbd2078e5, 0xbcd39b0a, 0x3ddf72ce, 0xbda73a7b, 0xbe14ce84, 0x3d35775c,
    0xbdb0fbb9, 0xbddbff26, 0xbc091099, 0xbd7a5549, 0xbd6fb765, 0x3d8a1c26, 0x3d168881, 0x3db1db67, 0x3e16caa2,
    0xbe08707d, 0xbe113179, 0xbc957e33, 0xbd962413, 0xbd8227df, 0x3dae2d1b, 0x3d4d3c4d, 0x3dd25460, 0x3e3e601e,
    0xbd9f3981, 0xbdcf31fb, 0x3c25e174, 0xbd50c152, 0xbd518b12, 0x3dadd293, 0x3ca1b6de, 0x3d623fa2, 0x3de82bef,
    0x3d12aa0e, 0xbcf408ab, 0xbd5ba50c, 0x3d4c23bf, 0x3d918e1d, 0x3d32eb2e, 0xbd4d9c5f, 0xbc9b796d, 0xbca227f6,
    0x3d241147, 0xbc98d03a, 0xbd16a9c0, 0x3cb44a7c, 0x3db76722, 0x3d7f3739, 0xbd680044, 0x3b50f80d, 0x3b640b24,
    0x3d97813c, 0x3c4d6225, 0xbc463427, 0x3be08eb7, 0x3d05d7bd, 0xb9337fd1, 0xbd49e510, 0xbce05524, 0xbd0a5187,
    0x3e0308f4, 0x3c3a5f4e, 0xbca8ef6c, 0x3de04bd6, 0xbc4a1691, 0xbdc0a44b, 0x3d2091be, 0xbd315bbf, 0xbdf3f14c,
    0x3e2d0ef9, 0x3d12c469, 0xbd017dba, 0x3e0d84ad, 0x3b3f6830, 0xbe010128, 0x3d3773e2, 0xbd82e1be, 0xbe1f23bd,
    0x3de51d78, 0x3c0fcc46, 0xbd134388, 0x3dcd7b3b, 0xbbc241f3, 0xbdaa71f9, 0x3d348333, 0xbd0c7133, 0xbdd72126,
    0xbcfd3246, 0xbdecca75, 0xbde7187d, 0x3d9c2b19, 0x3bbb491e, 0xbd2fba24, 0x3ddd1942, 0x3dc19a95, 0x3ccfe921,
    0xbd39f2a5, 0xbe1aa037, 0xbe103595, 0x3da5cd61, 0x3badf214, 0xbd81e3f1, 0x3e1fc67a, 0x3df3bb34, 0x3d24371e,
    0xbd0ab0c9, 0xbdd8e15d, 0xbdd0d3a2, 0x3d681d65, 0xbbac8a52, 0xbd154eff, 0x3dc3ec83, 0x3dc574c2, 0x3d12fb9d,
    0xbb982148, 0xba3cf84c, 0x3d96b68c, 0x3b9501cd, 0x3c3dadb1, 0x3c1e79cc, 0xbb054ffa, 0xbb628e4b, 0x3c828c23,
    0x3b306871, 0x3be4416e, 0x3b7e6ad5, 0x3c92e08e, 0xbd7fa8ab, 0x3c37e924, 0x3b82a0f4, 0xbbdc51d4, 0xbb8cab3f,
    0x3da9f23b, 0x3c4e8eb8, 0xbd71502d, 0x3bff8e08, 0x3c28f591, 0x3d37e06e, 0x3b51390c, 0xbb07a7a7, 0xbe509d5d,
    0x3d9a1e0d, 0xbb55c72b, 0x3f0c7019, 0xbf07a5fa, 0x3d2f7c1f, 0x3b29dab7, 0x3d891df0, 0xbd4b2cb8, 0x3daa0d7a,
    0x3a97beba, 0xbd27f65b, 0x3d7875b6, 0x3c1eea9c, 0xbd85029d, 0x3d762ee1, 0xbd1e3c4d, 0xbd6debf9, 0x3d6ed787,
    0x3da2bbbe, 0x3ba80301, 0xbbace022, 0x3e3c6b29, 0xbe3578c8, 0x3de7aef5, 0xbe26d2d6, 0x3c82144e, 0x3e2e06ab,
    0xbcc48f3c, 0xbc5e0193, 0x3e679b16, 0xbe84a01b, 0x3d35eb9e, 0xbd0e7382, 0xbd0d2aff, 0x3cfd0379, 0xbc71c398,
    0x3dbd7e78, 0xbcab47d8, 0xbe80dce3, 0x3e85ff81, 0xbd92bdb0, 0x3dd65a30, 0x3d991e75, 0x3be96bd5, 0x3e20df7b,
    0xbe83a763, 0xbd740e67, 0x3e34833a, 0xbd083e08, 0x3d9b6f3b, 0xbd678542, 0x3de142b0, 0x3cef42dd, 0xbc38fcc7,
    0xbd77e757, 0x3d8d744b, 0xbe376a6c, 0x3eabee20, 0x3d3abcc7, 0xbe48d053, 0xbe2366ae, 0x3d1601cc, 0xbd8b1674,
    0xbd487797, 0x3d2029a7, 0x3e2ac0bd, 0x3d3fcc5a, 0xbcc63f50, 0xbd687f6a, 0x3cc19d6b, 0xbcabb2dd, 0x3e13011d,
    0xbe5f4392, 0x3a31e43b, 0xbe0ad412, 0x3eb48fc1, 0x3cf8f59e, 0x3d77bc3a, 0xbccc09b4, 0x3d2edc30, 0xbe30f795,
    0x3d88a253, 0xbcdf48f4, 0x3ec31487, 0xbec8f7f9, 0x3d9abb5e, 0xbdaafae4, 0xbc845878, 0xbc5b41a0, 0xbe19f20c,
    0x3de9bdf1, 0xba3fec2d, 0xbccc7340, 0x3d5a5898, 0xbd9b1331, 0x3cb04bb7, 0xbc9b0715, 0xbcecdbe0, 0x3d7ed313,
    0x3d814faa, 0xbd91d5ff, 0xbec5a681, 0x3e79f3a1, 0xbcde2561, 0x3e131cb6, 0x3d18d8f4, 0x3d1a6c04, 0xbc9ad2fb,
    0x3cd52927, 0x3cde9639, 0xbdc94a67, 0x3b072708, 0x3d408523, 0xbd447531, 0x3c2686a5, 0x3cdae430, 0x3c51cf4f,
    0x3da214d5, 0xbb02d78d, 0x3e8b3fc2, 0xbea412bb, 0x3d2bdcf4, 0x3d8bce5c, 0xbd17cc10, 0xbda9522f, 0x3e2eb580,
    0xbd153b0d, 0xbd15f27b, 0xbe4a8155, 0x3ab775f9, 0xbd076134, 0x3e02fe5b, 0xbd02f180, 0xbdd38ca9, 0xbd5ff853,
    0xbd9dd19a, 0xbda0df5d, 0xbd856031, 0x3b082546, 0x3e2fc1b4, 0x3e193e5b, 0x3d3b635e, 0x3e0919ae, 0x3e031626,
    0x3d5a2f79, 0x3e792f30, 0x3eb44bf0, 0x3de2e36a, 0xbe30e6dd, 0xbefdbb45, 0xbeaef84f, 0xbe01d8e3, 0xbe0d0008,
    0xbdf47c28, 0x3c70a85e, 0x3a211417, 0x3c2a1e51, 0x3deed921, 0x3e0714be, 0x3e07a8a4, 0xbd8d8e3a, 0xbcb69673,
    0xbd58afe4, 0xbd453fbe, 0xbd231265, 0xbdb830b8, 0x3e25d128, 0x3e16b387, 0x3d332394, 0x3bb820ee, 0xbcf3643e,
    0xbdf585fb, 0xbd8dfd8b, 0xbe0b598f, 0xbd403530, 0xbdb8e991, 0xbcb952f2, 0xbbdd4b6b, 0x3ca2914c, 0x3ba7d718,
    0x3b4b62cb, 0xbb166bae, 0x3d35089b, 0xbd938315, 0xbde075f3, 0xbd798c93, 0xbc32b257, 0x3dda2168, 0x3df72003,
    0xbb17f96a, 0xbd4f1ab2, 0xbd89a470, 0x3d008fff, 0xbcec81e9, 0xbd73356a, 0xbd429e6f, 0xbe07ba4c, 0xbde3cb46,
    0x3d94e66c, 0x3e463754, 0xbe23b435, 0xbe9bb64b, 0xbd883daf, 0x3ed623f7, 0x3e836da6, 0x3d893080, 0xbc942be0,
    0x3cbbd412, 0x3e74f42f, 0x3e903cd2, 0x3dd295f0, 0x3e048dba, 0x3dec8c6d, 0x3e49fb78, 0xbdba11ce, 0xbddead51,
    0xbdf04d62, 0xbdcd7a81, 0xbdbb1d49, 0xbe1ac363, 0x3dda68da, 0x3e131be9, 0xbbfbddbb, 0x3df75c80, 0x3ac0f428,
    0x3dde5d17, 0x3ea63a0d, 0x3ecb9143, 0x3e5995ad, 0x3e64f189, 0x3d8ad51c, 0x3dd867b7, 0x3d1144b9, 0xbe0d04e6,
    0x3d874b09, 0x3e5ec31f, 0x3e80b801, 0xbcee3f1c, 0x3cfe18a1, 0x3d4523a3, 0x3e885352, 0xbe4fd41b, 0xbddf6ab7,
    0x3c854664, 0xbe96f736, 0xbe8e99a7, 0xbe0103e5, 0xbd9b413c, 0xbd8ced76, 0xbdf106e1, 0x3db782bb, 0x3d998d86,
    0x3d6a8e9a, 0x3c4b272b, 0x3d3f50e3, 0x3cb1682c, 0xbdd5daa1, 0xbe011b50, 0xbd617608, 0x3d9cf6ef, 0x3d590a71,
    0x3d4421c9, 0x3e2b0c8e, 0x3e4068d1, 0x3e156e21, 0x3c56ea94, 0xbc81599c, 0xbd555041, 0x3d36dece, 0x3c467db6,
    0x3ce2fe7a, 0xbe8ccd2b, 0xbeb38223, 0xbe8559df, 0xbe5bb200, 0xbe07b1ec, 0xbda6f314, 0xbd735919, 0x3e29ba05,
    0x3d98729a, 0xbdd36e6f, 0x3b86f288, 0xbcaa093f, 0xbd91e855, 0xbd9fa195, 0x3d265fc0, 0xbc8c7c27, 0xbe1e66c4,
    0x3e1aea67, 0xbd2ba290, 0x3e8cead4, 0xbce719da, 0x3e24e9cd, 0x3e407b3d, 0xbc21efc2, 0xbd698ad4, 0xbd27472d,
    0xbd51a370, 0x3cf4bb71, 0x3cb04fb3, 0x3cf90411, 0xbd76d268, 0x3dab4c7c, 0x3d04ed56, 0x3dc15d8c, 0xbdc70dd7,
    0x3cf2160c, 0xbd089397, 0xbeaf4523, 0x3ca91cfd, 0xbdba31ee, 0x3e99a87e, 0x3e0c7ccf, 0x3d3cb2e5, 0x3dd04757,
    0x3c8b094a, 0x3e1556c0, 0xbef42d7c, 0xbd1e34d6, 0xbd8ab7c5, 0x3e169065, 0x3d5c707d, 0x39e2677c, 0xbd7e56a5,
    0xbcbf8665, 0xbe1fa209, 0x3ec8c4f1, 0x3cde3d63, 0x3d9b26f9, 0xbe120d8b, 0xbdbe9334, 0xbb95521b, 0x3e118279,
    0xbe252c51, 0x3e664307, 0xbaee185a, 0xbd23ea51, 0xbd36d744, 0xbe0b1d71, 0x3d316908, 0x3d1c4c6d, 0xbdd7fff0,
    0xbe125c28, 0xbd552b2c, 0x3dc2a209, 0xb982eb41, 0x3e343bd1, 0xbdfa63b8, 0x3e144153, 0xbd8703ba, 0x3d3de10b,
    0x3d017629, 0xbdcbb859, 0x3e8d3ceb, 0xbdd32ca2, 0x3c775cd5, 0xbe4d13d7, 0x3d97aa79, 0xbd8957f0, 0xbd57d7e0,
    0xbdaa05e9, 0x3dd9b27b, 0x3d64e6fe, 0xbddf4666, 0x3b5ee66e, 0xbe0dca61, 0x3e1742e2, 0xbcac2e43, 0xbd09e061,
    0x3d0558ae, 0xbd3af144, 0x3e08345e, 0xbd8b1d02, 0x3c274706, 0xbbd21041, 0x3d231358, 0xbc912ccb, 0xbe2701b2,
    0x3dcb2b14, 0xbe1a20d2, 0x3ec587bd, 0xbd8f1ae4, 0x3d18ce30, 0xbe105d71, 0x3c80a445, 0x3d3b250d, 0xbcc16df0,
    0xbcd3d06b, 0xbd520437, 0xbda6d342, 0x3e06a82c, 0x3c5b0bd1, 0x3d513106, 0xbd8bac75, 0x3d43e18e, 0x3c0df2b3,
    0x3d969d6c, 0xbd43174e, 0xbd1537c1, 0xbd2abd7b, 0x3c6a1e2e, 0xbbb06ea9, 0xbbafcd0e, 0x3d720c6e, 0x3d46855a,
    0x3d508f93, 0x3dc58e2c, 0xbe9c6c59, 0x3c8c33e4, 0xbdbd679e, 0x3e20e1c1, 0xbd1ab6ce, 0x3defd0bd, 0x3c1716a1,
    0xbd360ea8, 0x3dc85891, 0xbec873be, 0x3dc0444b, 0xbce63f9b, 0x3e4c7ff7, 0x3c65ff6a, 0xbd5d13f4, 0x3d014e41,
    0xbe0087a8, 0x3e0e44c2, 0xbdad4ee7, 0xbe9da8da, 0xbd339771, 0xba30d30f, 0x3de6831b, 0xbdb0f3eb, 0x3c75a763,
    0x3e50fa54, 0xbe18a053, 0xbe30f8d5, 0xbe23f4fe, 0x3e18c5f3, 0x3d3b39e5, 0x3e0f4eec, 0x3e19f565, 0xbaa7b4ad,
    0xbd627773, 0x3dbeeaf6, 0xbd4f4ec2, 0xbe0f8cbf, 0x3e3c4cfd, 0x3c375c26, 0xbd6694a8, 0x3a80ac0c, 0x3c0db02d,
    0xbdcb0b1e, 0xbd977e23, 0xbe301ca0, 0xbccd4dc3, 0xbd72c7e5, 0x3d8047f4, 0x3db25854, 0xbe059f18, 0xbddbffab,
    0x3d191e89, 0x3c788ce2, 0xbe9dab00, 0xbe643d5a, 0xbe22c69d, 0xbe5a4bdf, 0xbd4f3bc3, 0x3c3f8645, 0x3e412ebc,
    0x3dd8c8b6, 0x3dde7ab1, 0xbc870d51, 0x3e9551b2, 0x3d8819ef, 0x3d269846, 0x3e73fb30, 0xbe333d17, 0xbe62834b,
    0x3e110a27, 0xbe1e5159, 0xbe530adb, 0x3e62f78b, 0xbdda469a, 0xbe72ebe9, 0x3e4358ff, 0xbc99389b, 0x3e588696,
    0x3d267e79, 0xbde3375a, 0xbebca603, 0xbd9855cf, 0x3d67c8fe, 0x3d248213, 0x3cc7277a, 0x3d295ea0, 0x3e1fe334,
    0x395ba25a, 0x3d63cd32, 0xbd0583cb, 0x3e38589d, 0x3daa4a4e, 0x3b906277, 0x3d098854, 0xbd8464b8, 0x3d420d47,
    0xbcd4c6bc, 0xbd0ea5c7, 0xbdc1e459, 0x3e6ee42f, 0xbd54b1cf, 0x3cd7ed80, 0x3e08030f, 0xbdd58f7d, 0x3b690cb3,
    0xbded8afe, 0xbda23383, 0xbde60534, 0xbdbda1cb, 0x3d5ff801, 0x3b9a0891, 0xbe062c61, 0x3dc1f519, 0x3e748fe3,
    0xbcddddb6, 0x3dfc54fe, 0xbde17a6f, 0x3e769e3f, 0x3e02070d, 0x3dc00718, 0x3dce4f67, 0x3da2e3b5, 0x3d2f063a,
    0xbd450494, 0x3dfca1ca, 0x3c1eafbd, 0x3e0a03d8, 0x3da211a7, 0x3e2f5548, 0x3db4dcac, 0xbe12057d, 0x3d110885,
    0x3d064b63, 0xbe273717, 0xbdb3a3b0, 0x3d0bf1ea, 0xbda3126e, 0x3cc45429, 0x3d60b665, 0xbdb59ee8, 0xbdea4a17,
    0xbe2e8188, 0x3d11581c, 0xbdb3b6e6, 0xbe1f0465, 0xbdb0ac75, 0xbddd08c5, 0xbddda54d, 0xbdb21049, 0xbd921ffc,
    0x3db60d2c, 0xbded3ef0, 0xbe42f80d, 0xbdbbb9bd, 0xbd4e33dd, 0x3dffa3c6, 0xbda1a56c, 0x3e52be5a, 0x3d2ad1a5,
    0x3d75afc0, 0x3d4f141a, 0xbe01fdb2, 0xbe6fcc2d, 0x3e012c8b, 0xbd29718a, 0xbdd60c48, 0xbee5eb1b, 0xbea52b24,
    0xbea19018, 0xbe34b09a, 0xbb8f6cf5, 0xbce124b0, 0xbd0ec8df, 0x3ddbb8ec, 0x3ec2cf16, 0x3df95fc4, 0x3d378e61,
    0x3d9968ad, 0xbc6b6536, 0xbdcb4a57, 0xbdab75d6, 0x3bc9fa5d, 0xbd72c65a, 0xbd2cb6d8, 0xbc0fdd58, 0xbab0972e,
    0x3e1eafd3, 0xbdf479c0, 0xbe1883cb, 0xbd36d982, 0x3c63ce34, 0x3b4df17c, 0x3c170860, 0xbd638293, 0x3da2abfe,
    0x3ea517ef, 0xbe28b379, 0xbcace091, 0x3e097e02, 0xbe9308ad, 0xbdeaaa28, 0x3caed46d, 0x3d8d2711, 0xbc9a6037,
    0xbe7b80d8, 0x3dce5600, 0x3d5dc747, 0xbddc7bc4, 0x3e0abdcc, 0x3d97778b, 0xbda5ecae, 0xbdc318e6, 0x3c2f51c8,
    0x3d9c1ebc, 0xbd445e38, 0x3cdbf6c7, 0x3d234171, 0xbd8629ac, 0xbd7d85d9, 0x3d36dd6f, 0x3d2afdb2, 0xbd7b4101,
    0x3da68292, 0xbe550c52, 0xbe104ee3, 0xbe5d9038, 0xbdcaa4d9, 0x3e12ecae, 0x3e93dcaf, 0xbb7bfe19, 0xbe02299b,
    0xbea3fecb, 0xbd7f22da, 0xbd57c5ca, 0xbdec2ef1, 0x3e5a189a, 0x3e310c08, 0x3e187338, 0xbd755d75, 0x3dcb04c3,
    0x3e496188, 0xbd42f0dd, 0xbdb15e58, 0x3c9d703b, 0xbe05bfb0, 0xbe2a252e, 0x3cc1dccc, 0xbd6187bb, 0x3d0635cf,
    0xbc803395, 0xbdc6bc5d, 0xbe0f02d1, 0xbe2cd490, 0x3d978008, 0xbcc0b6f5, 0x3e41f716, 0x3e05b2be, 0x3bc2d99c,
    0x3d50c73c, 0xbe02fc70, 0xbdd9cbf8, 0x3d3a4d43, 0xbd88837c, 0xbd099c2e, 0x3d83f6f7, 0x3dfddcc0, 0x3df796a7,
    0x3d6f0da5, 0x3e02b380, 0x3d850628, 0xbc319ffd, 0xbd8aa9d2, 0xbd33cd8c, 0xbe50a979, 0xbdede3d6, 0xbd848cdb,
    0xbd9cd591, 0x3923a24c, 0x3d70c951, 0x3d49fd2f, 0x3d3a1ab1, 0x3d917779, 0x3d7d0dda, 0xbe6cbe32, 0xbdab78ce,
    0x3d2b00d1, 0xbdf81f12, 0xbdacb81f, 0xbc97e805, 0x3bc81926, 0x3d33ea0f, 0x3e815765, 0x3dd2691f, 0x3dc0b897,
    0x3e84ccec, 0xbd96c644, 0x3cbb8851, 0x3e211ad4, 0xbea72646, 0xbe019003, 0xbe1e9385, 0xbd5b4952, 0x3c83b7a1,
    0xbcefc1fb, 0x3e88b1dd, 0xbda189d3, 0xbe481dec, 0x3d6e18a2, 0x3e110dee, 0xbc854a2a, 0x3deeba40, 0xbd80b926,
    0x3e2124c7, 0xbed8da9f, 0x3eea8d64, 0xbe0bbc24, 0xbd823b81, 0xbe49fe55, 0x3e21c5ec, 0x3ccef38c, 0x3c8ad047,
    0x3cd8faae, 0x3ccac017, 0xbd68abd1, 0xbc5dc7d1, 0x3cd47e7f, 0xbc6470ea, 0xbc37e2c7, 0xbe0a1a34, 0x3e94eb79,
    0x3e22fab9, 0x3f12d06e, 0xbf2bc619, 0xbe5d6d3f, 0x3c8962be, 0x3dbf614e, 0xbe24b1e4, 0x3d349541, 0xbd318209,
    0x3d7f99f8, 0xbdb05042, 0x3d8c1342, 0x3d6e122b, 0xbd2de03c, 0x3d9ccc22, 0x3b2ec9d3, 0x3ce178af, 0x3ce95f48,
    0xbdc007b6, 0x3d72a174, 0xbbbd89bf, 0xbac15faa, 0x3e2041ac, 0xbde616af, 0x3d425591, 0x3e6a0235, 0xbe0d2da3,
    0xbe1fa2e6, 0xbeff86fa, 0x3ebe3aef, 0x3e2b0dba, 0x3d2f4902, 0xbd40b823, 0x3d4c3736, 0x3e059da1, 0xbda3c8aa,
    0x3dabcc75, 0x3e371567, 0x3a08477f, 0xbdad3dad, 0x3c0a0010, 0xbe2173f8, 0x3c5fb3d8, 0x3b5d077a, 0x3ce3f04f,
    0xbae67bc1, 0x3c102d8c, 0xbe01fbe7, 0x3d40a270, 0x3d8fb7d1, 0xbd82902e, 0x3cbde065, 0x3d875812, 0x3cfc421e,
    0x3d76e060, 0xbceef306, 0x3c8a3e53, 0xbd61c473, 0x3cab9482, 0xbc915138, 0xbc0215aa, 0x3d552385, 0xbbaf52af,
    0x3e033e5d, 0x3d7f1f0c, 0xbcc49bf9, 0xbc125d7b, 0xbd9302b3, 0xbdad6c90, 0xbd24ace2, 0x3ccdbc20, 0x3db2be88,
    0x3cbe2e3f, 0x3e48c2e6, 0xbe6988f2, 0xbd23df94, 0x3d726657, 0xbd881e52, 0x3e196bd7, 0xbd8a2b2b, 0xbbce6637,
    0xbd1e7212, 0x3db9cbbe, 0xbd9c7820, 0xbd1cad70, 0x3d18ba7c, 0x3cceb7ad, 0x3cce3508, 0x3c3d3f7e, 0x3aa1c5b9,
    0x3c7aa167, 0xbcdc0c67, 0xbc74690e, 0x3bf2fa8a, 0x3d1a52d4, 0x3b3cc6b7, 0xbd69d9b6, 0x39591e4d, 0x3ba2857e,
    0xbbde3ffc, 0xbc84a0ef, 0xbd594f23, 0xbcd281d4, 0x3dc43d1b, 0x3d8c51d2, 0xbb3bca52, 0x3dbb930d, 0xbdcaabf4,
    0xbb3adb82, 0xbdc40e8a, 0x3dd0345a, 0x3d40ea87, 0xbd7e59a4, 0x3d1d6ad1, 0xbd1e1e0a, 0x3d40ba4a, 0x3c536754,
    0x3cd11efc, 0x3db52a07, 0xbd03ab54, 0x3c8b6965, 0x3dcb70c5, 0xbe3e0446, 0xbd7a6714, 0xbda08ad5, 0xbcefb7a3,
    0x3d385bcd, 0xbe0a6c52, 0x3e168914, 0x3c9152fb, 0xbe7eedb5, 0x3e871364, 0xbd74bf79, 0x3ddd4c88, 0xbcfa2a78,
    0xbd4a59b1, 0x3db06a71, 0xbd815eed, 0xbd443c25, 0x3dd6c32e, 0xbd324f06, 0xbd333097, 0xbd7e280f, 0x3dd474c6,
    0xbc25763d, 0x3ddca1b5, 0xbd2fdf23, 0x3d01becc, 0x3de24ba6, 0xbe384b98, 0x3d69690a, 0xbd935ede, 0xbe1d2e0f,
    0xbca18627, 0xbe44ee32, 0xbe06c355, 0xbd8e9145, 0xbeae4e7d, 0xbe765b0c, 0xbd1089c5, 0x3e091436, 0x3de576cb,
    0x3da374af, 0x3e49079b, 0x3e078914, 0x3da5ff20, 0x3e6f3a11, 0x3e637511, 0x3e1ab9d1, 0xbda7821a, 0x3d4a7f2d,
    0x3dd4c6f6, 0xbe7549c1, 0x3e33e7bc, 0x3e06dd34, 0xbe86e6d4, 0x3d903c3b, 0x3da7c496, 0xbd0de82d, 0xbddc48b2,
    0x3d22d6ae, 0x3cea8ee0, 0xbe52dab3, 0x3e18bb88, 0x3e5f4288, 0x3b8f6772, 0xbd29439b, 0x3d2906ab, 0xbbfb0c57,
    0xbca81198, 0x3dbbd6fc, 0x3de1227f, 0x3d024085, 0x3dc2b6ba, 0x3e82369a, 0xbd2f9fcb, 0xbe21ab68, 0x3e1aff45,
    0x3dbf8a21, 0xbe1aa267, 0x3e7c912b, 0x3dbe431d, 0xbd4f2c30, 0x3e3f2290, 0x3e6a5f6b, 0x3ae26dac, 0xbda6f102,
    0x3cb7f19e, 0xbd46b061, 0xbdf85788, 0xbda25065, 0x3d4ccf90, 0x3b641ae7, 0xbe7e3ec8, 0x3e2e5779, 0xbd9fe90a,
    0x3d789232, 0x3e6eaca3, 0xbe16d333, 0x3d09a1c4, 0x3e1e66b2, 0x3da9ac1c, 0x3c1a1474, 0xbdaa103b, 0x3db574f4,
    0x3db88d95, 0x3c52d211, 0x3dffaf92, 0x3e0846b3, 0x3d3b4149, 0x3d3c774e, 0x3e4ffde6, 0xbdcbf68e, 0xb9fd4d8e,
    0x3cd3dd1b, 0xbda4a8db, 0x3d315b2a, 0x3dad4594, 0xbddcb438, 0x3d962ba9, 0x3dad0856, 0xbc35f003, 0xbd35a946,
    0xbd5d3f49, 0xbdd0b36f, 0xbe0c507a, 0xbd544314, 0xbdc5fb29, 0xbeab3426, 0xbe63ef0e, 0xbd9c7b2c, 0xba88132b,
    0x3c0d655e, 0xbe290dde, 0xbc91b019, 0xbdb95522, 0xbe6503bc, 0xbcf5d52f, 0x3c3533a4, 0xbc53a5d3, 0x3d384f78,
    0x3d83fb9d, 0x3dc40574, 0xbd8689b7, 0xbdfa1285, 0xbe09415c, 0xbdc36b3b, 0x3d973a40, 0xbe073df9, 0xbea1e14f,
    0xbe9ffe63, 0xbe9547b2, 0xbe2fb373, 0xbe4c8998, 0x3f032d55, 0x3e8595c5, 0x3e9bf33d, 0x3d8975f9, 0x3d4b9256,
    0x3df94d3d, 0xbd02a17a, 0xbd97d86c, 0xbcf01829, 0xbd72451c, 0xbc7a75e2, 0xbbf64c3f, 0x3d3e69e2, 0xbd3bcc8f,
    0xbd7f7620, 0x3d589569, 0xbe8ac938, 0xbc724155, 0x3d8a90c8, 0xbadf3768, 0x3e7796e2, 0x3d1987e5, 0x3d2b6cf0,
    0x3de73ab5, 0x3e8c2e70, 0xbdc1fd1d, 0x3d38428e, 0x3e537557, 0x3e2428bb, 0xb8b4358f, 0x3c914934, 0xbdacc63b,
    0xbe10f158, 0xbd2a718c, 0x3cd9f8a8, 0x3d1a8e31, 0xbcc62919, 0xbe1db5bc, 0x3e281d17, 0x3d5bdaaf, 0xbded19ad,
    0xba09d919, 0x3dbed3ba, 0xbd59c9e8, 0x3df2f560, 0x3e093ddd, 0xbd98165e, 0x3cef41e2, 0xbd2e0179, 0x3d856f6c,
    0xbc79b7fc, 0xbcaa2959, 0xbeadc2a1, 0xbd8c9e81, 0x3b830ca4, 0x3db6b2a4, 0x3e9256e9, 0xbcfdb919, 0xbdf4ee00,
    0xbe5d550a, 0xbe534710, 0xbdb4cc46, 0xbe251f97, 0xbe3666c4, 0xbe737d84, 0xbe3abe32, 0x3e104c42, 0x3ccee83e,
    0xb6d80284, 0x3dc9f182, 0xbe058e19, 0x3dbc1295, 0x3e4a5237, 0x3d26d822, 0x3de8d802, 0xbd9bf201, 0xbe18c4bd,
    0xbdf47efe, 0xbe6afb64, 0xbe4aacf7, 0xbe6abd5c, 0xbd945492, 0xbe7c5046, 0xbdc3201c, 0xbde46395, 0xbcd6ba8f,
    0xbd4c0377, 0xbe9f469a, 0xbe98ee37, 0xbcb1ec20, 0xbe6cdb6f, 0xbe9a327c, 0xbdc11dde, 0xbd210748, 0x3db2a119,
    0x3db3a96e, 0xb852e247, 0x3d6281c3, 0x3dd56062, 0x3d04bf72, 0x3e8041f2, 0x3e7e2381, 0xbdb311ba, 0xbd863c5d,
    0xbe1870f6, 0xbc4ed629, 0x3c22e9fa, 0x3b3ee6fc, 0x3e137b9d, 0x3db7f67d, 0xbb314e9e, 0xba837fb7, 0xbd38c8dc,
    0xbd9a0250, 0x3db5d6a8, 0xbdb7ed17, 0xbc96831e, 0x3de49137, 0xbe4de4cf, 0xbe2459ba, 0x3d749586, 0x3dccd009,
    0x3dbdc1c4, 0x3e07f2f0, 0xbe044fd3, 0x3d04569c, 0x3e73eae4, 0x3ed66c04, 0x3e0726aa, 0xbd95aaf0, 0xbd9ef349,
    0x3d707aa6, 0xbde56f8c, 0x3dbb900f, 0x3e96750b, 0xbccd5d78, 0xbd25f107, 0xbdf16002, 0x3e190ec0, 0x3e7d4273,
    0x3e9d8973, 0x3d9cbcdf, 0xbe902094, 0xbe8002a6, 0x3e0ef4f0, 0x3e5ab32c, 0x3ec6c135, 0x3ddcb21a, 0x3d990a8c,
    0x3d9b18a7, 0x3bfabeda, 0xb6596c36, 0xbba7c681, 0xbcdd90fe, 0xbda39b09, 0xbd87020f, 0x3d610700, 0xbc480736,
    0xbd55914e, 0x3a6405b9, 0x3e26fb37, 0x3e6d0c05, 0xbbeb29db, 0xbd5019cd, 0xbea53b71, 0xbcb205e4, 0xbd98d918,
    0xbcfe798a, 0x3daed9df, 0x3e5ed868, 0x3c373d80, 0xbdbfe31b, 0xbe0b15d6, 0x3d58c474, 0xbd945536, 0xbc4b169c,
    0xbdacec5c, 0xbdd36907, 0xbe1b525a, 0xbd331dd3, 0x3c893b02, 0x3de74435, 0x3d48249e, 0xbd546b5e, 0x3d70469f,
    0x3d1efdd7, 0xbc05c1c1, 0xbdb60cfe, 0xbe54c5a3, 0xbccd5ac6, 0xbc912cf7, 0x3e045a75, 0x3e343a83, 0xbd2dfcc1,
    0xbd0b2949, 0x3dbf80a3, 0x3e469e23, 0x3d91d287, 0x3d41ac8f, 0xbe259f98, 0x3c688de6, 0x3e010922, 0x3dd25586,
    0x3d94e11a, 0xbc80cb42, 0xbce888c9, 0xbd80a827, 0xbddea2fd, 0xbdb3b2dc, 0xbdaa40dd, 0x3c47e852, 0xbad7045e,
    0xbe197f63, 0x3c880d5f, 0xbd04dd0b, 0x3e689cf7, 0x3d8a10d5, 0x3d8a7348, 0x3db1abd0, 0x3daf19fa, 0x3d9ffed3,
    0x3e80f349, 0xbd3ba09b, 0xbb9009a2, 0x3dd79722, 0x3e2d408d, 0xbe14a8ed, 0xbec3c97e, 0x3dc03292, 0xbd80dfe2,
    0x3c48b020, 0xbd56243b, 0xbdf175b3, 0x3d7ecffe, 0x3b7d1b91, 0x3d5caeaf, 0x3dcc38f2, 0xbe02b825, 0xbdec0a93,
    0xbe3433e5, 0x3c3e2821, 0x3d30feb8, 0x3e2083b6, 0x3e4ad230, 0x3e7a27aa, 0x3e6842d5, 0xbd7b8c0a, 0xbd91757b,
    0xbd8f6aac, 0x3ad30ce2, 0xbd0f220d, 0xbcdfb904, 0x3d9fda26, 0x3df9dc1e, 0x3e145f7c, 0x3d810343, 0x3d55d2a2,
    0x3e329a25, 0xbd102044, 0xbc08306e, 0xbd73345e, 0xbd03abf1, 0xbe565764, 0xbe96e58d, 0x3d8b2254, 0xbc32fd9c,
    0xbce0fd00, 0x3dbc78eb, 0x3e0d86d7, 0x3c34a3ab, 0x3d9028c0, 0x3e052558, 0x3ea74a92, 0xbe1e6010, 0x3bedeed6,
    0x3dabc42d, 0xbdbe048b, 0x3d84c49e, 0xbda783ba, 0xbccdb79c, 0xbe1dcfc6, 0x3ed76b97, 0x3c02dd27, 0xbcc760ec,
    0xbd357c24, 0x3e5f04f7, 0xbebd1252, 0xbc8fb2d8, 0xbc2fedaf, 0x3e833a55, 0x3e3e25b4, 0x3cc7f07a, 0x3d1b5d83,
    0xbcec5b95, 0x3d12f141, 0x3dd20171, 0xbd504ea3, 0xbbdc22a9, 0xbd5a017b, 0xbcfaf049, 0xbd400b32, 0x3dac1f9c,
    0xbe830b5f, 0x3d6490ee, 0x3e9fb379, 0xbe4dddae, 0x3cb5bc09, 0x3d15f1c4, 0x3e3ea13d, 0x3da71c00, 0xbd30acd7,
    0xbd3ccce9, 0x3dd7a425, 0x3dba9beb, 0xbd6848b1, 0xbd7ea19f, 0xbe9260c7, 0x3e20dc56, 0xbdbe86d7, 0x3e0f7c72,
    0x3dbfe74d, 0xbd61d70e, 0xbc16fa19, 0x3d49e119, 0xbd398807, 0x3daff317, 0xbe7ca8a3, 0x3d56fa9c, 0xbe080663,
    0x3de64f13, 0xbdf97b55, 0xbefdb4b2, 0x3ef0c7fd, 0x3d0cea2f, 0x3e88a6d3, 0xbd97183c, 0xbcf93d5c, 0x3c0d52d8,
    0xbd52f280, 0x3d802563, 0x3ee2f634, 0x3bbd0dba, 0x3c14b4ea, 0xbe63be47, 0xbda04d40, 0x3d19cf40, 0x3ce44aa1,
    0xbbde36a4, 0x3cf3517b, 0x3c8d75e8, 0xbc709498, 0x3ce02fa5, 0xbd44784f, 0x3d86dcf1, 0xbccbf7ec, 0xbd7047d5,
    0x3d0682fd, 0xbc0ebee1, 0xbe44a23c, 0x3f0c6858, 0x3e00c11b, 0x3e58673d, 0xbe922404, 0xb94c2360, 0xbcd608c3,
    0xbdc6c433, 0xbd95a855, 0xbc3315c1, 0xbe275383, 0x3da28c5f, 0xbc70bf39, 0x3e74165d, 0x3c329680, 0x3e29028e,
    0xbdd2d8ea, 0xbc48e555, 0x3e2b997b, 0xbe37f487, 0xbd1cb330, 0xbd2ace66, 0x3d9f1d60, 0xbd4c319e, 0x3dcf1615,
    0x3d0af0a1, 0x3c78c226, 0x3d91ecde, 0x3dac5ddd, 0x3c936c76, 0x3e1f2b98, 0xbef3f5a6, 0x3ca130df, 0xbb1d83cf,
    0x3c958b3e, 0xbcc34aa0, 0xbde9c6f5, 0x3c380fe9, 0xbc042b7b, 0x3cc383a4, 0x3d1803cf, 0x3dc01b38, 0x3a836b95,
    0x3d7706b3, 0xbc85912d, 0x3d2c28b9, 0xbe5854ec, 0xbdb3d98f, 0xbc2b2924, 0x3e14e88a, 0xbc721343, 0xbc3105fb,
    0xbcaf0106, 0x3d041efb, 0xbd9380d5, 0xbda0f82c, 0x3d8f2fd6, 0x3e084d2f, 0xbdf81d88, 0x3cb9fb78, 0xbdacbabd,
    0xba98ee8d, 0xbdb09edf, 0x3e730c49, 0xbd31527f, 0x3dc8cbf2, 0x3cf3cce1, 0xbd124c2c, 0xbe0ee51d, 0x3cfd7f41,
    0x3df4a263, 0x3e3a5f58, 0x3e45897a, 0xbe09d89b, 0xbd2a14b8, 0x3e8ab2dc, 0x3db5ffac, 0x3e662163, 0x3dfd0574,
    0x3e0f277a, 0x3e276292, 0x3defd629, 0x3e1e1a31, 0x3e471276, 0x3dbfd497, 0x3d692a6c, 0xbd966347, 0xbdb1e607,
    0x3ce93cf8, 0xbc3adaa7, 0x3ea6508d, 0x3cc16a55, 0xbdd9eb91, 0xbe2f435b, 0x3cbb8eee, 0x3d11027d, 0x3d6d7445,
    0xbd7d283b, 0x3d810a78, 0xbd8d08a7, 0x3e55d850, 0x3d896f4a, 0xbdd951e6, 0x3db8209a, 0xbc07daf9, 0x3d171539,
    0xbd83b232, 0xbcaf2215, 0xbe8c2c7f, 0xbbcf1ff4, 0x3d020364, 0xbb1b9115, 0x3d142258, 0xbd24c0c9, 0x3b7a863b,
    0xbd72beaa, 0x3dfd1a9e, 0xbeab0338, 0x3d3777eb, 0x3e0cb506, 0xbc51a6de, 0x3de97a31, 0xbceed162, 0xbdc7f5b1,
    0xbe2a7e3a, 0x3e08e985, 0x3e63c50f, 0x3ed0480e, 0xbccf7235, 0x3dc6a22a, 0xbd083d6b, 0x3d9f156b, 0xbd84cd0c,
    0xbdbb72b0, 0x3c794144, 0x3d58b5f4, 0xbd7c0906, 0xbd9ab31d, 0x3d341f71, 0xbdea2afe, 0xbe17893d, 0x3e788d65,
    0x3e491d08, 0xbcc47781, 0x3da06721, 0xbdbe80d6, 0xbd0c2e8f, 0x3c158d0c, 0xbd15d011, 0xbdea1da9, 0x39f766be,
    0x3e55a843, 0x3e2d3ad5, 0x3eae6d7a, 0x3cb0d6a8, 0xbc8b851b, 0xbd9ea1dd, 0x3bbc7819, 0xbcce6b77, 0xbc2934b5,
    0x3cace259, 0xbd038e13, 0x3a85c389, 0x3e173e6c, 0x3d98e539, 0xbd94ed8c, 0x3dfb39d7, 0x3d9949eb, 0xbc5b7bb8,
    0x3da0575d, 0x3d3d9c6d, 0xbd5ede3d, 0xbdf4e31d, 0xbd0df039, 0xbc0b7cc4, 0xbdda61aa, 0xbb8a6596, 0x3d603579,
    0x3dfcbcdd, 0xbb540461, 0x3cb5fd1a, 0x3dc47be1, 0xbd5836c5, 0x3ced9b92, 0x3c2b8cf2, 0xbdf62bbf, 0x3d9f7c1b,
    0x3c8fc79b, 0x3d47c651, 0x3d9d5132, 0xbd4bded1, 0x3dfba924, 0xbde79795, 0x3e2a82c3, 0xbac5bc81, 0xbda82988,
    0xbe290c31, 0x3d89eef8, 0x3d4bba33, 0x3e299885, 0xbdc8f7dc, 0x3de17c70, 0xba8a93d3, 0x3da9edbf, 0xbd0b92ad,
    0x3d94fdf2, 0xbe910842, 0x3d9c515b, 0x3e6f5f3c, 0xbd56ff45, 0x3e25ec19, 0xbe0dfa00, 0x3d6ecf6a, 0x3c75b362,
    0x3daf73f3, 0x3e71acad, 0x3da8b180, 0xbe6bd5d4, 0x3ceec095, 0xbe410ef1, 0x3dfd11fa, 0xbdc4f794, 0x3ca5dc70,
    0x3df00ac9, 0xbe14be2c, 0xbd115c32, 0x3dbe95c0, 0xbdb7ed57, 0x3d08c7c3, 0x3dc7db13, 0x3dbc912b, 0xbe36badf,
    0x3dc0be34, 0xbe71f4ef, 0xbcb2dd2b, 0x3e885f38, 0x3d12c60f, 0x3e25d46a, 0xbd96666e, 0x3e1e771f, 0x3e8daf5c,
    0x3e0a74cf, 0x3d2b0891, 0x3e81e1d7, 0x3e281fb7, 0x3e0bdeb2, 0x3e32f22a, 0x3dce9b03, 0xbd666e6b, 0xbea1fb37,
    0xbe903f97, 0xbd9dc869, 0xbed69017, 0xbe50a900, 0xbdc74995, 0xbe00ce2e, 0x3cc2d06b, 0x3d905717, 0x3cca9581,
    0xbe08048b, 0x3e3fde0a, 0x3d422830, 0xbe8c9212, 0x3d386ff6, 0xbd08dbe0, 0x3c9ac2b4, 0xbcccba48, 0x3db6e3b3,
    0xbdfc4d31, 0xbc3d1db3, 0xbeb04160, 0x3e4fdb8e, 0x3d838655, 0xbda28868, 0xbce591ca, 0xbb98aa70, 0xbe245a45,
    0xbe43d4c6, 0x3d6f988f, 0xbe0a7df8, 0xbe00c979, 0x3c9e075e, 0xbdbb8bfc, 0xbdb469eb, 0x3d16dd8a, 0xbea53802,
    0xbe079f81, 0x3d4dbec7, 0xbe0dd344, 0xbe112faf, 0x3e0e925d, 0xbde49f5c, 0xbde90858, 0x3dd74ca4, 0x3c9c9329,
    0x3dfd6798, 0x3c2f907d, 0x3e7f42ff, 0x3d104ab3, 0x3de69728, 0x3d601aa8, 0xbcde0885, 0xbd02e045, 0xbd96302d,
    0x3d2f0216, 0xbe0d228f, 0xbe8194a8, 0x3b1b8e79, 0xbdb73906, 0xbdca827a, 0xbe04f073, 0xbe2614c7, 0xbe3a3c7c,
    0xbdaefebe, 0xbe25cd46, 0xbe31a41f, 0x3ceb2c68, 0xbe3a44fc, 0xbd8708d3, 0x3d640757, 0x3d0bfc0a, 0xbd3b8815,
    0xbd990d10, 0x3dbed85a, 0xbc13925c, 0xbd87eb4d, 0x3d6173f6, 0xbcd80661, 0xbc15f597, 0x3e18a414, 0x3e2ff031,
    0x3e3c3fb6, 0x3d4ccf25, 0x3eada15a, 0x3d975874, 0x3e33c31f, 0x3e2ee281, 0x3d684900, 0x3d97f595, 0x3e254547,
    0x3dd78651, 0x3cb15c56, 0xbc027129, 0x3e2d07af, 0xbd2bb5f3, 0xbce3c6d8, 0x3d383dad, 0xbd564c92, 0x3d5b7dc4,
    0x3d0a134d, 0x3ddbb268, 0xbe2f0627, 0x3d3628ad, 0x3e3512a9, 0xbcdef117, 0xbc95e765, 0x3d33adad, 0xbe33592f,
    0x3d1cd347, 0xbe2ec5f6, 0xbd4f1378, 0xbe834dbc, 0xbd1bd1d5, 0x3e320f4c, 0xbd49d183, 0xbd19a655, 0x3e296ab7,
    0x3dabb0f9, 0xbe0f6286, 0xbd86f928, 0x3dcd6a5e, 0xbd9e6acd, 0xbd39f8b9, 0x3d6778b5, 0xbc37be21, 0xbd8eef01,
    0x3e6c996b, 0x3d6489a2, 0xbedb986f, 0x3ce24ec6, 0x3d4b956f, 0xbe9e8e64, 0x3e8d114c, 0x3de6e511, 0x3eb87927,
    0xbd942744, 0xbdc6d498, 0x3e6900a9, 0x3eb09234, 0x3aea85d3, 0xbdf0961a, 0x3ead3448, 0xbd3a7ad9, 0xbd77b291,
    0x3e2e1b3c, 0x3d9330e2, 0xbdae02e2, 0xbe9fd1c5, 0x3ceddf20, 0x3d820b41, 0xbd9a0ce3, 0x3c23d394, 0xbca003f2,
    0xbc9f0e4b, 0x3cccdf8e, 0x3e038e20, 0xbe8607e1, 0x3d0fce48, 0x3e34ec01, 0xbd9e2412, 0x3e16bf0c, 0x3e91a3fa,
    0xbe8fccc0, 0x3da011c1, 0xbdf39b40, 0x3ec26f21, 0xbdc99064, 0xbd9ad23f, 0x3e412849, 0xbcfd93f7, 0xbe1b1a75,
    0xbd96d358, 0x3db672ec, 0xbe175dcf, 0xbe44abff, 0xbc20c7e8, 0x3a1af1db, 0xbebbd0f7, 0x3bfe302a, 0xbd9bb0d6,
    0xbd68f266, 0xbcde078f, 0x3dc1ad09, 0xbed2e43a, 0x3c95ee33, 0x3de067c5, 0xbd0b1e62, 0xbc1ae913, 0xbc945a67,
    0x3de22b1c, 0xbc3e39d8, 0xbe9d2aef, 0x3e699c8f, 0xbd6f54f8, 0x3dda235b, 0xbd1a97e8, 0xbdef2039, 0xbcd89b8f,
    0x3da7db7c, 0xbcfd2b30, 0x3b4e92c9, 0x3eba12dc, 0xbd98ebe0, 0xbdffdbae, 0x3e968c05, 0xbca221e9, 0x3c4400d1,
    0xbca81b77, 0x3cec4f7b, 0x3c6aca22, 0xbdcf9190, 0x3d37f68c, 0xbe230b39, 0xbdef6714, 0x3d8f42fb, 0xbdc5fdce,
    0x3d39c43e, 0x3d863495, 0x3d218ac0, 0x3c03b450, 0x3dd7d34f, 0xbde78b89, 0x3d8b2e92, 0xbd0b3ae9, 0x3d56793c,
    0x3d90223e, 0xbdc02fc9, 0xbd1fa783, 0x3e989779, 0xbdf8d8b1, 0x3dbe2790, 0x3eb44385, 0x3e0ea30b, 0x3e324209,
    0xbe6249f7, 0xbc388448, 0x3ea98c29, 0x3e60b105, 0x3d515311, 0xbe4af0de, 0x3e43b080, 0x3d838295, 0xbd832de5,
    0xbd7333f0, 0xbbefbfa9, 0xbd187388, 0xbc89864e, 0x3d21c623, 0x3c1b3fca, 0x3d059ec8, 0xbe47d116, 0x395e949e,
    0x3e207beb, 0x3de15b58, 0xbd3de185, 0xbcab362d, 0x3e1122d2, 0xbcf0543b, 0xbdbfd72b, 0x3d06a6fa, 0xbcb24eea,
    0xbd1132b7, 0x3ae37953, 0xbcd4f0fe, 0xbbd33422, 0x3c9f18a8, 0x3b10ab7c, 0x3d16696a, 0xbcd0b8af, 0xbcd47adb,
    0xbdcc9184, 0xbc12b28e, 0xbad3753c, 0xbd94df8e, 0x3daeb012, 0x3db52e85, 0x3d13a0dc, 0xbdb6eb04, 0xbe8406b8,
    0xbe925b66, 0x3d2580ca, 0x3d81662b, 0xbcb20ab9, 0x3e43cfe4, 0x3e3332cd, 0x3e541402, 0x3de31337, 0x3e9a5ddc,
    0x3e695bff, 0xbd85515c, 0xbc21df6d, 0x3cf8a2b9, 0xbe2bf2ee, 0xbe6f250d, 0xbe1bd1b1, 0xbdf7f1d7, 0xb7e9dfd5,
    0x3daed848, 0x3c834b7c, 0xbc06fb91, 0xbd2ba3c1, 0x3dc0e91f, 0x3a73fa40, 0xbd0c5c93, 0x3dfc7c72, 0x3dd9ac5c,
    0xbd8c5a3e, 0xbd511824, 0x3e2af13d, 0x3d590577, 0xbe31ccc7, 0xbe6ea71e, 0x3df56f9d, 0xbd41c711, 0x3e155912,
    0x3e2591a1, 0xbe13748c, 0x3d519ea6, 0x3e5c32f7, 0xbe7117fd, 0xbe0b058d, 0x3ce73057, 0x3d4c3560, 0x3dd08d88,
    0x3dd0b0e8, 0x3dab5ffd, 0xbd827e4b, 0xbe0a569f, 0x3e4f8aa7, 0xbde60bcb, 0xbe829264, 0xbe4684d1, 0xbd93f281,
    0x3dab411d, 0xbe1951ce, 0xbd207aea, 0x3e05fe5f, 0x3d85fd62, 0x3d778806, 0x3df4053c, 0x3dce999b, 0x3e10f0c1,
    0x3d203a0e, 0xbe2490f6, 0xbd942ef3, 0x3dfa0a99, 0xbd972bff, 0xbe18fc1d, 0x3d10b858, 0x3dbf7293, 0x3da2d7ae,
    0x3cdc8a86, 0x3dfca38a, 0x3c97eda0, 0xbe0742d9, 0x3d79f7ac, 0xbd8f7aa6, 0xbe67e11d, 0xbd963c0d, 0xbc853f81,
    0x3d3a20ea, 0xbcb2d7b9, 0xbc8ce091, 0x3cccdb12, 0x3d1577eb, 0x3c86c1e5, 0x3a46c1b0, 0xbe6adbd7, 0xbe40c04a,
    0xbda364a5, 0xbd48b451, 0x3d0362b6, 0x3ca97b19, 0x3e319ebe, 0x3e2569dd, 0x3e2a2cad, 0x3d861152, 0xbd2731d3,
    0xbe20dc79, 0x3de3712c, 0x3d18a04f, 0xbe191256, 0x3e0fd26e, 0x3d6a5c69, 0xbd428d61, 0x3d1af23b, 0x3d6da0ad,
    0xbc71ce15, 0x3c44f0d4, 0xbe3fd0d8, 0x3d88e0a1, 0x3e31c3bf, 0xbd0f4b66, 0xbd9040c4, 0x3d877589, 0xbe1f0faa,
    0x3e277a64, 0xbcf3732f, 0x3ee9523e, 0xbe0abf37, 0x3d5db940, 0xbeb1559f, 0x3c786533, 0xbcf20d98, 0xbc2b971b,
    0xbc6f8a0f, 0x3c97318b, 0xbd7c6984, 0x3d7c1eff, 0x3c6db5d4, 0x3ca27c4a, 0xbbf94ecd, 0x3de0a6cf, 0xbe295df0,
    0x3d4d0f4c, 0xbcd31823, 0xbed7f8d5, 0x3e851e62, 0xbac7e036, 0x3df3d17b, 0x3d864eb6, 0xbdc3cb76, 0x3e68a4d6,
    0xbd260cc1, 0xbe2de658, 0xbcc456fb, 0x3d8df81a, 0x3cd958b7, 0xbe3ff032, 0x3cd2523a, 0x3d8b7bdd, 0xbe1a78f7,
    0x3ca91c7b, 0x3d810781, 0x3d7d0b46, 0xbdf5fb4d, 0xbde68a96, 0x3e48cfc8, 0xbdb0a0a1, 0xbe2b59fe, 0x3dcc5186,
    0xbdb99004, 0x3c182718, 0x3e9b6929, 0xbe63e0a5, 0x3d11210f, 0xbd032580, 0xbd026d66, 0xbc201986, 0x3da7b53e,
    0xbe89e5bc, 0x3e1e0e42, 0xbf352e10, 0x3f35f630, 0xbe8b09b3, 0x3ef34eda, 0xbb531d7b, 0x3d774f0c, 0xbdec78ba,
    0x3d832db5, 0x3de0ccf7, 0x3d0003c1, 0xbe6f7069, 0xbd96f0a6, 0x3e0e2f5b, 0xbc948799, 0x3c970c2c, 0xbc8fed36,
    0xbdc31c66, 0xbdc73a91, 0x3e0fe66f, 0xbce5bcf3, 0x3cec44a3, 0x3cc16b04, 0xbdbeb59c, 0xbcb03244, 0xbc522216,
    0x3debed51, 0x3e110a16, 0xbd1826ba, 0xbdf53090, 0xbcbc5c33, 0xbcc6753d, 0xbc46e416, 0x3d8602e5, 0xbe91cb09,
    0x3cae695a, 0x3d43c186, 0x3c85a9b4, 0xbdf11c70, 0xbd4db269, 0x3e2927c5, 0x3bbc5375, 0x3d4f04fc, 0xbc4d45c0,
    0xbd955e20, 0xbd94e648, 0xbd8b11ee, 0x3e19221a, 0x3c7c9001, 0x3de8ccb4, 0xbdb9759b, 0x3d112865, 0xbc3a8194,
    0x3d51decb, 0xbd27d52f, 0x3d792073, 0xbdaa97ec, 0x3b3da54b, 0xbd771b88, 0x3dac021c, 0xbdd052c8, 0x3deb2ace,
    0x3dad02fb, 0x3d308f65, 0xbe10ab38, 0xbd098f1b, 0xbc9d4beb, 0xbe413cac, 0x3dd6965f, 0xbc803e83, 0x3dce503c,
    0xbd7365ca, 0xbd4feea5, 0x3d136eeb, 0x3e17c88d, 0x3d6d46ab, 0xbe92eb0a, 0x3d95601b, 0x3ccdc740, 0x3dde25cd,
    0x3c8290dd, 0xbe9f74b4, 0x3d3134bb, 0x3d9716a2, 0x3d715d04, 0x3da72dab, 0xbde918c7, 0xbda041a3, 0xbd16f57e,
    0xbde7bee1, 0xbc09f696, 0x3e8d28de, 0x3e157575, 0x3d5cf78e, 0xbf384bfa, 0xbd094eed, 0xbbeba33b, 0xbd28ef23,
    0xbd0b1a1a, 0x3d2533e9, 0xbd3f6a42, 0xbcfd9c9b, 0x3dba6636, 0x3c5a0fdf, 0x3d177244, 0x3de16a16, 0xbca0af8d,
    0xbcba3731, 0xbe724167, 0xbd156b97, 0xbd6b366c, 0x3d89699c, 0x3de438dd, 0xbe1e7eef, 0x3d6e23f6, 0x3e058b5d,
    0x3d78bdf8, 0xbb6e6b87, 0xbd47536f, 0xbc82e7e6, 0x3e368045, 0x3d506d6a, 0xbcf88ecf, 0x3d86c83c, 0x3d5bac16,
    0x3d0c48a5, 0x3d7242a3, 0x3e1c5f09, 0x3b1e09ae, 0xbe43255b, 0x3d9d3a02, 0x3e1961fe, 0xbcbefb5a, 0x3d1202b3,
    0xbd890510, 0x3e00ac4f, 0xbe447348, 0x3d0ccc27, 0x3da719a4, 0xbeb6ae50, 0x3e009da5, 0x3c75c695, 0xbda49275,
    0x3d398b56, 0x3cf5d192, 0xbe7275f6, 0xbedd7daa, 0xbd749c8c, 0x3d8998f8, 0xbd70d28e, 0xbca0063c, 0xbcb2748b,
    0xbd09b504, 0x3c8e1ccf, 0x3e402f6c, 0x3a40d0b9, 0xbd919d6a, 0x3d9a5989, 0x3e342020, 0x3ce8596b, 0x3dd6adb1,
    0xbbd17dd9, 0x3de53e9b, 0x3d842f6a, 0x3d0760e5, 0x3d98f739, 0x3d147c56, 0x3d969c57, 0xbca052bc, 0xbe09dfdc,
    0xbd0bea62, 0x3b6d9979, 0x3e553de7, 0x3def7417, 0x3e26c24b, 0x3caf6940, 0xbd7a23e5, 0x3c8c5fe1, 0xbc8f4238,
    0x3cc9a59b, 0x3abe121d, 0x3e777a72, 0xbdfbf0de, 0xbd10b114, 0x3df0d6ea, 0x3e21e7c2, 0xbd0540df, 0xbca672ea,
    0xbc1d0ee0, 0x3ce3ff29, 0x3d8bc05f, 0x3df9c241, 0xbdfb61fa, 0x3cf7d15d, 0x3ccc8bd5, 0x3cd936d7, 0x3b2684f7,
    0xbd670d27, 0x3d8a0317, 0x3e1466f2, 0x3cae1461, 0xbd76c289, 0xbe03a4c0, 0xbd20c107, 0xbc155e32, 0x3b1ae270,
    0xbbb90f5a, 0x3d571de9, 0x3e07e49f, 0x3d90f982, 0x3e6685de, 0x3db95d00, 0xbdcf9e55, 0xbcdec8d4, 0x3dafa88b,
    0xbc4d5f65, 0x3dd14553, 0x3dcedeb0, 0x3e0fd4b8, 0x3b9a2d74, 0xbdf6fc9a, 0xbd63efa7, 0xbc10d020, 0xbb56c97b,
    0x3a21bd12, 0x3b590324, 0x3bfb9297, 0xbb92fe15, 0xbbddcf34, 0x3c3f97b6, 0xbc12b63e, 0xbb91c031, 0x3d00aaca,
    0x3c08dbc9, 0x3a3b44f3, 0xba89e4f2, 0x3be22421, 0xbd45317a, 0x3d682064, 0xbbff5f1c, 0xbcf09fb5, 0x3d5d7a3a,
    0xbd83ea2a, 0xbb62fc23, 0xbd0f4fba, 0x3da7b7af, 0x3e011221, 0xbc9b7e40, 0x3da48f7e, 0x3c175a05, 0xbda649ed,
    0xbd85dbad, 0xbd94d7a4, 0xbe979fd3, 0xbea0dc9c, 0xbedd6e40, 0xbd835654, 0xbc7de632, 0x3e039eb9, 0xbe29c5bd,
    0xbe0e5aeb, 0x3d40f826, 0x3df0f231, 0xbd3d5d21, 0xbd9e383d, 0x3cd8ae8d, 0x3d0e22ce, 0x3da6ac9f, 0xbd38f705,
    0x3de8b8f7, 0x3d437a05, 0xba1db2e5, 0x3da3a398, 0x3cf6f502, 0xbd16f326, 0xbe75fe42, 0xbe6caecc, 0x3e1250c1,
    0x3e903ba5, 0xbd19f4f2, 0xbe8d2519, 0xbe5afc4c, 0xbf0394bd, 0x3d6056d6, 0x3ce1782a, 0x3d1d295b, 0x3e507b57,
    0x3deb3d9a, 0x3e45ee00, 0x3e148d0b, 0x3dffda96, 0x3e580816, 0x3d9baaf9, 0x3cfb485c, 0xbd8abe91, 0x3d7b5add,
    0xbbb45704, 0xbe2d6dda, 0x3ce992ec, 0xbb284ee0, 0xbd62bfab, 0xbe091cee, 0xbe20a897, 0xbe3b2d0e, 0x3dc37f82,
    0xbb15b219, 0xbdfcf4b0, 0xbdf9977a, 0xbd1f3000, 0xbe163b5b, 0xbdeee28f, 0xbdae81fa, 0xbe72119b, 0xbea3e168,
    0xbe87596f, 0xbe3d0208, 0xbf071b0b, 0xbe766b47, 0xbf0f4b93, 0x3e21a23d, 0x3de94967, 0xbda6a921, 0xbdd9c04c,
    0xbe3a4bfc, 0xbc7ffc65, 0xbda932ba, 0xbd78372f, 0x3d8085f6, 0xbc8858ec, 0x3d4a2e9e, 0x3de69d31, 0xbbdc1355,
    0x3d98164a, 0xbdf2c903, 0xbdaaa451, 0xbdeedf35, 0xbd036f57, 0x3ca0ef97, 0x3de15b9d, 0x3ccd53af, 0x3e138744,
    0x3d3bd6db, 0xbd79219b, 0x3d1e7718, 0xbd8b30c7, 0xbcee7b98, 0x3d741197, 0xbce82ecd, 0xbe341a62, 0xbe555e58,
    0x3d590e17, 0x3d87979e, 0xbd166a65, 0xbd4aad73, 0xbd7b79c2, 0xbd975162, 0x394bad77, 0xbd2006ca, 0x3d8d46e9,
    0x3dbd018f, 0xbd9caf52, 0xbe0d41ef, 0xbe30012f, 0xbe961c98, 0xbd19debd, 0xbdb69139, 0x3def214b, 0xbe007b11,
    0xbdb2a7c2, 0x3e339cd9, 0x3e4fbf5e, 0x3df7006c, 0x3d678b4d, 0x3e13f170, 0x3e1e2228, 0x3e09d8b1, 0x3c31b39a,
    0xbc82efbf, 0x3b8a65e8, 0x3d3aa6ad, 0xbd28b97f, 0x3d903fb5, 0xbde0cbee, 0x3d85fb53, 0xbd2cb887, 0xbe044702,
    0xbe0e70d3, 0xbdca663b, 0xbde114b7, 0x3e1e5637, 0x3ded3740, 0xbe162590, 0x3d963856, 0x3d1a8753, 0xbe271328,
    0xbda68cc6, 0x3c0b67f5, 0xbe002297, 0x3d20a1cc, 0xbd6e6377, 0xbd412f2c, 0x3e90ede9, 0x3e3e159b, 0xbd76bd74,
    0xbddf96dd, 0x3d1470aa, 0x3e683656, 0x3e32df5d, 0x3da2a6de, 0x3e09b4b2, 0x3ce354bc, 0x3e0d76ad, 0x3adc8904,
    0x3d80c30a, 0xbe0d2b61, 0xbdaeae92, 0xbc82a4e6, 0x3e3e5b4f, 0x3b9c81e7, 0xbd9c07b4, 0x3e3ad237, 0xbe94f3b1,
    0xbc3ab9b0, 0x3db10d61, 0xbdc074a1, 0xbd3dc7cc, 0x3ba7b62a, 0xbdcfcff2, 0xbda32bb7, 0x3db4539a, 0xbbe5bf5f,
    0x3e479111, 0x3d0d05bf, 0x3d31e577, 0x3ca61296, 0x3e2450c8, 0xbe473986, 0xbe78132a, 0x3d38c037, 0xbdbb21ad,
    0xbde382be, 0xbdadca9a, 0xbe034f74, 0xbcf69cab, 0xbd538afd, 0xbd41db55, 0x3d807323, 0x3e0cea16, 0xbd360c8a,
    0x3dcb7d86, 0xbdbf14c2, 0xbe4d02d9, 0x3e00ea14, 0x3d360a5a, 0xbcaf6827, 0x3ddc5352, 0x3cbe08bf, 0xbdbe4fa6,
    0xbe39f7bd, 0x3cf291cc, 0x3d523849, 0x3c395837, 0xbdd3a979, 0xbb8e2d94, 0x3cefcb1e, 0xbd1c9545, 0xbe144545,
    0xbe4eeb5e, 0xbc1ae811, 0x3d4013a3, 0x3e149623, 0x3e012906, 0x3d7f9c31, 0x3d531b3f, 0x3ba8524b, 0x3e09668b,
    0x3ec82af2, 0xbda0f53a, 0x3cd91404, 0x3d21cd6a, 0xbc0c29d0, 0xbe917e9d, 0xbd3397ed, 0x3c4c4531, 0x3d181d93,
    0x3db035f5, 0xbdb01322, 0xbd98f8a8, 0xbc4e1d95, 0x3e15c579, 0xbd0a9caa, 0x3c8e1a27, 0xbd1a8565, 0xbed9b2d4,
    0x3df2912d, 0xbd633591, 0x3c68eb71, 0xbe75e027, 0x3cf3618d, 0xbeaa13e6, 0xbdb1475a, 0x3aa66c3e, 0xbe75be15,
    0xbd529103, 0x3d5d147f, 0xbe5c6fd8, 0xbd4f1b29, 0x3cae85d9, 0x3d10b0d8, 0x3d7a7a73, 0x3d91aec7, 0xbe0a39a3,
    0xbe0324ce, 0x3e8191fc, 0x3d87979b, 0xbd7c497a, 0xbdb6b0f7, 0xbe20dd13, 0x3d60c294, 0xbd8cb2c7, 0xbe9eaf0b,
    0xbe9e0096, 0xbdb114ec, 0xbd27b8dd, 0xbe22eace, 0xbb75da95, 0x3c30fbcc, 0xbd61b988, 0xbbb6485a, 0xbe11c655,
    0xbda00299, 0x3b84219c, 0xbe13a91a, 0xbd0b30ac, 0xbc209c10, 0x3e148bd5, 0x3d1980e2, 0x3d207f60, 0x3e0bdf3b,
    0x3d9f4017, 0x3d66b3a0, 0x3e6b610b, 0x3e131c9b, 0x3e32f59a, 0xbd293115, 0xbd34f59f, 0x3d1f50ef, 0x3c32615e,
    0xbe2cb16b, 0xbd9d3f85, 0x3d7200d0, 0x3d1d2027, 0xbd996d66, 0xbb99f60a, 0xbd06541c, 0x3db96235, 0xbc41448d,
    0x3da3ea30, 0xbe030f16, 0x3e104c03, 0x3de16823, 0x3d3522f9, 0x3e1f8c99, 0x3b7c4f94, 0x3d9e6603, 0x3d722ad3,
    0x3c0d5e34, 0x3c92448d, 0x3e75850e, 0x3dbb9d21, 0x3e0f74b1, 0x3d8fe519, 0x3d6dd1f0, 0xbd8e8800, 0x3da21c72,
    0xbd254961, 0xbd095b3d, 0x3d9fd7f3, 0xbe1acb38, 0xbdcaef76, 0xbcc441bf, 0xbce576a8, 0xbdad86ef, 0x3d1b1666,
    0x3cf161a8, 0xbd966032, 0x3d43df42, 0xbd1eb50b, 0xbcaa1224, 0x3cf76334, 0x3c1410c7, 0xbd1c682a, 0x3d0000ea,
    0x3d6dd7d0, 0x3acd6207, 0x3d6943e9, 0xbc293207, 0x3e0796de, 0x3e2340aa, 0x3e403026, 0x3ddab89e, 0x3d8425fd,
    0x3d90c879, 0x3da93e19, 0x3d41dd3d, 0x3dc16286, 0x3d2def56, 0x3d1c46c4, 0x3da9db69, 0xbbb45cb5, 0xbe4444ad,
    0xbd02bcb7, 0xbc8124b4, 0x3def67d9, 0x3d326883, 0x3c9fb8bf, 0xb9881d4d, 0x3b82b306, 0xbc6593b1, 0x3a74a069,
    0x3dece0df, 0xbc808134, 0x3dc30f30, 0xbb8138e9, 0x3dc3a628, 0x3d9135a2, 0xbd37b295, 0x3d7d0f38, 0x3e07bc70,
    0xbc8c75de, 0x3cba5831, 0x3cdeceee, 0x3c9caa28, 0xbbeb0cf5, 0xbdf2d87b, 0xbccf13ad, 0xbdd224c7, 0xbe0aab5d,
    0x3c767dd2, 0xbdad7173, 0xbde8a6cc, 0xbdca4e56, 0xbbf35692, 0x3d8b7cc1, 0x3df6b4ff, 0x3dc83e6b, 0x3e23863e,
    0x3d8d15b9, 0x3e0de2d5, 0x3e629f58, 0x3e2ece5f, 0x3e11d797, 0x3d3e727b, 0x3d19a2b8, 0xbca71626, 0xbd9ffb0f,
    0xbda9c56a, 0xbc632233, 0x3d87b019, 0xbd107b88, 0xbd451bbb, 0xbbefc306, 0x3d2f3d65, 0xbcb38802, 0xbc10db7a,
    0xbd106ede, 0xbdc05337, 0xbd6ec9d5, 0x3d3e5d81, 0xbcceeab4, 0xbcc31cb6, 0xbd9a3c54, 0xbc7c191a, 0xbc9c1a32,
    0x3e263590, 0xbdf70968, 0xbb2699ac, 0x3e380231, 0xbebe6f81, 0xbe05317e, 0xbdd84d46, 0xbe9f8d9d, 0x3cc21968,
    0x3e122042, 0x3dd64b0f, 0x3ce0e993, 0x3cf6e7aa, 0xbbfb959d, 0x3db0b720, 0x3e061550, 0xbe155b8c, 0x3d88da0b,
    0x3e430806, 0x3e5ae34e, 0x3c57c2da, 0xbdd2f858, 0x3e27d08d, 0x3d60a3f1, 0x3cc70850, 0x39a6d13c, 0x3d96f275,
    0xbca61313, 0xbe64e2ec, 0x3e2f7ed5, 0x3dfddeae, 0xbdf8e5fc, 0xbe2ec830, 0xbe40c875, 0xbec83342, 0x3daf556d,
    0xbdc38fd5, 0xbcfc4c27, 0x3c2dec78, 0x3c9c9aef, 0x3dc68682, 0x3cb7b2dc, 0xbdf3a6c3, 0xbe59f3fe, 0x3dbfa037,
    0xbdfb8b14, 0x3ce69f0f, 0x3d771991, 0x3dd9c3ae, 0x3e2efb50, 0x3bc31d2b, 0xbd892dc3, 0xbd0209a9, 0xbd04211a,
    0xbe2d73c8, 0x3e42965c, 0xbd096754, 0x3d9715c6, 0x3d89e99b, 0xbc6460d7, 0xbd6e48e0, 0xbb5772f1, 0x3cbde84c,
    0xbdcb8b14, 0x3d1f1aa0, 0x3a9ba426, 0x3deed359, 0x3d72503c, 0xbe3ec726, 0xbe42c05f, 0xbe971cfd, 0x3cc52859,
    0x3d714e5c, 0xbd63d6fb, 0x3ccdc451, 0xbd8df419, 0x3c43ccfa, 0xbd01f15c, 0x3c875db0, 0xbdb7d172, 0x3e14139f,
    0x3ea2435a, 0xbe2c6a02, 0x3dea2dee, 0x3d90d37b, 0xbe2a51b1, 0x3d44c9ed, 0xbb38aa76, 0xbd85d524, 0x3dbf1173,
    0x3c873bd4, 0x3d3e8da7, 0xbda56977, 0x3d162503, 0xbc05f6e6, 0x3d967e4d, 0xbe265f69, 0xbcfb767d, 0x3d5ada63,
    0x3cc1d74c, 0x3e026da0, 0x3cfa1c95, 0x3e1ecbc0, 0x3cf70bf7, 0xbe0caced, 0xbdf90e32, 0xbe4e80e9, 0xbddf0820,
    0x3eb06da4, 0xbe8f8a9f, 0x3d0ddf63, 0x3b2bc3d7, 0x3d28c6fd, 0xbd99b8ba, 0xbe0a76f2, 0xbeb9b849, 0x3c92a6db,
    0x3d78d5e9, 0xbe3d36c9, 0x3d22252a, 0x3cc99373, 0xbc9603f6, 0x3d9bb765, 0x3db884d5, 0xbe328682, 0x3e245832,
    0x3e3bbde5, 0xbce464ae, 0x3bbb92fc, 0xbe00d065, 0x3e319eba, 0xbd9a0c30, 0xbe523306, 0xbeb5da14, 0xbd169916,
    0xbe227a92, 0xbd308c64, 0x3da5bc09, 0xbd889d48, 0x3c39feaf, 0xbc9b0526, 0xbc58f6e5, 0x3cbe610c, 0x3d67a3b7,
    0x3c4c24a8, 0x3bfaf8cc, 0x3c61152b, 0xbde50b48, 0xbb84a06a, 0xbd6a90c5, 0xbbdc2253, 0x3d85a68e, 0x3d4146c4,
    0xbc8e0c28, 0x3d0a142a, 0x3bdd9935, 0x3cd52ef8, 0x3cb211bc, 0xbca2dfd3, 0x3da2facb, 0x3e1b755b, 0xbdb49062,
    0xbbeab638, 0x3df8e21d, 0xbdecaf1b, 0x3afbbb18, 0xbd1d04fe, 0xbc9493b5, 0xbd0fd0e5, 0xbe184519, 0xbbf9fe8c,
    0x3c61358e, 0xbdb3447f, 0x3d06c163, 0xbd9324df, 0xbe33490e, 0xbca90e44, 0x3dd010cb, 0xbd0d1736, 0xbc104e64,
    0x3dff4e71, 0x3d6c080f, 0xbd93b7f4, 0x3c887a13, 0x3bcb574d, 0xbdcade65, 0xbe5551f4, 0xbe28064a, 0x3d9dfd27,
    0x3d9bb9e3, 0x3e14153d, 0x3d8d2c21, 0xbcc85c85, 0x3d85021c, 0xbe9bb97a, 0xbd2925c6, 0xbe8b2b17, 0xbe93bf6b,
    0x3e085de0, 0xbe89a852, 0xbf14ebf6, 0xbdd8f59c, 0xbf06c7f8, 0xbc5a3ecc, 0x3d4d0b15, 0x3b85bb52, 0xbda41703,
    0x3dcdf6ca, 0xbd1c3d41, 0xbda8ba36, 0x3caadade, 0x3d981307, 0xbce4d0b9, 0x3d41ab9e, 0x3dd946c4, 0x3de9a8ca,
    0x3d90a53a, 0x3d459864, 0x3bf20ad6, 0x3dbe9c72, 0x3d874ef6, 0x3d2f188b, 0x3d8885f5, 0x3b10e900, 0x3d87577f,
    0x3dda377e, 0xbd397c2e, 0x3d3cb89d, 0x3badeefb, 0xbccfca75, 0x3c35b557, 0x3d73d99d, 0xbd0a2307, 0xbd043d5a,
    0x3aa75642, 0xbcefab2f, 0xbc0d9054, 0x3d08a328, 0x3b45028e, 0xbef06f75, 0xbe6ec5a8, 0xbee40855, 0xbe507322,
    0x3d51f99a, 0xbe4630da, 0xbe60c42b, 0xbdcf5a02, 0xbe7cf32d, 0x3df08199, 0x3e01c1ac, 0xbe3d76b8, 0x3e1d21af,
    0xbde481f6, 0xbca0b1ef, 0x3dc387a1, 0xbe2e39d8, 0xbe79e887, 0xbedb9b5c, 0xbe3dc6bb, 0xbed45698, 0xbe4cccd5,
    0xbd217956, 0xbd8ba2ae, 0xbee17dda, 0xbe92f6b2, 0xbead822d, 0x3d215afe, 0x3c12f118, 0xbc9b1a31, 0x3d2f4788,
    0xbc830e58, 0xbcb1b887, 0xbde14561, 0x3cdee327, 0x3c9dafbd, 0x3d05c425, 0x3cba21d7, 0x3d38bad3, 0xbd6df2d8,
    0xbcde93ed, 0x39070ca3, 0x3d05a70d, 0x3e0b7b08, 0xbd555537, 0xbe25520f, 0x3d1c09e2, 0xbda594a1, 0x3cf47c0b,
    0x3de272b9, 0xbcdea7a3, 0xbe13f974, 0xbe08a500, 0x3b61cd1a, 0x3d14b4ec, 0x3d74a9d8, 0xbd0c5c28, 0xbcd9897c,
    0xbcffa81b, 0x3abe61a6, 0x3d35bef1, 0x3d85fe41, 0x3d549cb2, 0x3e0395b0, 0x3edb8564, 0x3b96dfc0, 0xbe628ec4,
    0x3ebba024, 0x3e95537c, 0x3e672250, 0xbedc6d2f, 0x3d7721d0, 0x3d3095ae, 0x3dc7e61b, 0x3d8d80dc, 0xbd8b9a6f,
    0xbe285de5, 0xbd1dd978, 0xbdae32b0, 0x3dc03d9a, 0x3d15e9f9, 0x3dada3df, 0x3da45476, 0xbc0683fb, 0x3d697229,
    0xbc36b369, 0xbaf6547b, 0x3d7f2cf7, 0x3d35563a, 0xbc9a3ab1, 0x3d4ef445, 0xbc8fd12e, 0x3e2089ce, 0xbd709bd9,
    0x3e530277, 0x3ee649ae, 0x3b3045d0, 0x3e814de6, 0x3dd7b8a2, 0x3d1e9a8d, 0xbd134aeb, 0x3cf36cc9, 0x3d71a004,
    0x3badcad5, 0x3d7d1dae, 0x3d1ffb8d, 0x3c6da06b, 0x3d7d0339, 0x3cc3a0af, 0x3debe8cc, 0x3cec1dff, 0x3d673204,
    0xbe177d89, 0xbc16284d, 0x3cbf77af, 0x3d9f52af, 0x3b8956c2, 0xbe06cb7e, 0x3d131ae7, 0x3c100cf5, 0xbd33d6b8,
    0xbc634f75, 0xbc9e5f4a, 0xbc123a84, 0x3c9df1cf, 0xbc904573, 0xbc5b4793, 0x3e607242, 0xbdb0a3d3, 0x3c1f359e,
    0xbded4c89, 0xbd723c80, 0xbe125e08, 0x3dd3fe53, 0x3d8086ac, 0x3e15d4a9, 0xbe6cd9b5, 0xbcd110ee, 0x3e28ee71,
    0xbcfc1b23, 0x3dc768a0, 0x3dc13347, 0xbd02ace0, 0xbe18c1f9, 0x3d469a80, 0x3c4ba007, 0x3d6fcf39, 0x3b89ff43,
    0x3d7d111a, 0xbdb8cbfd, 0x3d952a46, 0xbceefcca, 0x3cc661b8, 0xbdcd65c5, 0xbcabbdae, 0x3cccde5b, 0x3c6d5556,
    0xbb44bfda, 0xbc2c290d, 0x3dbf07ef, 0xbd992e79, 0x3d344e22, 0x3caedeaf, 0xbb713161, 0xbbbee658, 0x3c67b8af,
    0xbd347f77, 0x3b6a03b1, 0xba7c0b1e, 0x3d47745c, 0xbc43b03c, 0x3d3817f5, 0x3c7d7b60, 0xbd19c991, 0xbe126ab7,
    0x3e977272, 0xbe687d58, 0xbdf9e88f, 0xbe9ed62d, 0x3db9d1a8, 0xbe0cc417, 0xbe9c673e, 0xbe152410, 0xbdacad50,
    0xbe137a2c, 0xbc9f7b5c, 0xbd97184c, 0xbde8eaec, 0xbdc7e55f, 0x3ccf17cd, 0x3cd243a3, 0xbc2f934a, 0x3dfb095c,
    0x3d2e4441, 0x3db0c40c, 0x3d9451ac, 0xbd0878c8, 0x3db01282, 0x3c35c78d, 0x3dd48bed, 0x3cfe0a2c, 0xbc69141c,
    0x3cb60960, 0x3cce26dd, 0xbd1577f8, 0xbd911390, 0xbc78ed62, 0xbca52cfa, 0xbb8f466c, 0x3c31e9ab, 0xbc13aad0,
    0xb9d4907c, 0xbdbda8b6, 0xbe176971, 0xbc069a84, 0xbcdb81bf, 0x3e5aacc7, 0x3d190a2e, 0x3d646c59, 0x3de68372,
    0xbdaf2c70, 0x3e0408c6, 0x3e3baae4, 0xbc9099e0, 0x3decb316, 0x3e25eae2, 0x3d90da58, 0x3e3cc149, 0x3e21a72b,
    0x3cd1caa4, 0x3dcd12a2, 0x3d3b02aa, 0x3d4819ca, 0x3dbb5007, 0x3b0dcf20, 0x3d93a188, 0x3d80cc42, 0xbd3e8c1f,
    0xbbbb82e8, 0xbd2184f2, 0xbbe5a6ce, 0x3d1aeb4f, 0xbcda9aa6, 0x3d54bf6a, 0x3b3fd656, 0x3e1c71de, 0xbbedbf0e,
    0x3d2a0a14, 0x3e35bc13, 0x3dd72232, 0x3de908de, 0x3e0c168d, 0x3e34159d, 0x3dc372d7, 0x3e82164a, 0x3dd8d36a,
    0xbdb7d9a1, 0x3dc08222, 0x3e3ac2eb, 0x3da8f83e, 0x3cc78932, 0xbdd2fd8d, 0x3cb0bc35, 0xbd09b058, 0xbdd7ce27,
    0xbc8bc513, 0xbdebc38d, 0xbda1a4c2, 0xbbe4eaa2, 0xbde7dfcd, 0x3beac753, 0xbdaf812f, 0x3db7e658, 0x3d9186fa,
    0xbdc59268, 0xbc631c33, 0xbd6f0dfc, 0xbe1194cf, 0x3d9d5dd8, 0x3e250c9f, 0x3db5525a, 0x3d9d9887, 0xbd4ba81d,
    0xbe0d8c99, 0x3ca37641, 0xbc5fe0ce, 0x3d3d5fa5, 0x3db11748, 0x3d9decb3, 0x3d91762d, 0x3e58e8ff, 0xbc3734e0,
    0x3c7ea5ff, 0x3e0d08e1, 0x3cff90b1, 0x3d9e9d91, 0x3e423171, 0x3d8dd7a8, 0xbc9565e4, 0x3dfb68a3, 0x3df16706,
    0xbdc9f98d, 0x3e1cfee3, 0xbbd64d4c, 0x3c9dc080, 0x3dc6855b, 0x3dc7677c, 0x3dc80677, 0x3df2882b, 0x3df2f00e,
    0xbca245b9, 0x3d7ae832, 0xbc6e8526, 0x3dac11e4, 0x3d7fcb17, 0xbcc858c5, 0xbe3c8a00, 0x3c7c48b5, 0x3bba15cc,
    0xbdc830e9, 0xbdcb4231, 0xbd50938c, 0x3d0eb3b9, 0xbdab753b, 0x3d03c3bf, 0xbcbe06d1, 0x3d368f6f, 0xbc37311f,
    0x3d798871, 0x3c6e7a5b, 0xbd015df7, 0x3d82a215, 0x3d1e6cef, 0x3d497ffb, 0xbdaeab27, 0x3d1015ac, 0xbe5c3d3b,
    0xbe0b4509, 0x3e28112d, 0xbd5b9b2a, 0x3ce40764, 0x3e7c66a7, 0xbd6fa165, 0xbc92bd0e, 0x3d2cf3cf, 0xbc7d085a,
    0xbdec0bc1, 0xbe5bec7a, 0xbbec1f38, 0x3ce6419d, 0x3d0325f2, 0x3d60ea1f, 0xbe761f4f, 0xbc56c1f2, 0x3e766952,
    0xbd885e4e, 0xbdf148a4, 0x3e2d6ef1, 0x3ea0f4ff, 0x3d8f65e0, 0xbdef0b19, 0xbd474ecf, 0x3d66fcee, 0xbe1443ae,
    0x3d616d40, 0x3e24322c, 0xbd4d3dd7, 0xbd659f6e, 0x3d8953e5, 0x3d56314c, 0xbcd917d3, 0x3dca2b22, 0xbceb3db9,
    0xbb6152f7, 0xbd5bb0b3, 0x3d163137, 0xbd7c2f68, 0x3ca3b5b5, 0x3e6126fc, 0x3e3b79bc, 0x3dee4097, 0x3dd0f5ba,
    0x3d8da7b4, 0x3cd26092, 0x3d3a7959, 0xbcfefcee, 0xbd3bdfd4, 0xbc8baa44, 0xbd8f70aa, 0xbe5ada72, 0x3c3cc5ba,
    0x3b398ef1, 0xbe4704ef, 0xbd14ac82, 0x3c9fba63, 0xbc9a4a6a, 0xbd41f77c, 0xbde6507e, 0xbe39fc63, 0xbc717d97,
    0x3e2c2059, 0xbd5c5301, 0x3a8226df, 0x3ca6d0d0, 0xbdc1c345, 0xbd292191, 0xbe035008, 0xbd206a3e, 0xbe40fabe,
    0xbd5dae36, 0xbd8e0c86, 0xbd3b4ea3, 0x3d196004, 0xbb87eb73, 0x3b950258, 0xbea943d4, 0x3d17aced, 0xbd826d04,
    0xbdf273c5, 0x3e844326, 0xbb986143, 0x3d0ac02f, 0x3decb80c, 0xbd5815a4, 0x3e1a0844, 0x3cfefe15, 0xbddced44,
    0xbe104012, 0xbe78589e, 0xbcb16ec9, 0x3cfee171, 0x3d8098ad, 0x3ca8cc80, 0xbc1c0781, 0xbde1698e, 0x3da5d58e,
    0xbe640db9, 0xbd08bd62, 0x3d61c21a, 0xbd8feb40, 0x3d8c20e3, 0xbe25f1b4, 0xbdeb3130, 0x3e0e2e6f, 0xbe390c01,
    0xbe423a8f, 0x3da45d3f, 0x3c8aacaa, 0xbc89651c, 0xbd542567, 0x3bb2d7a6, 0xbc47cae9, 0xbd1b7b52, 0xbd3d7b46,
    0xbd31d822, 0xbd8f97d1, 0xbc8cd50c, 0xbd3bdf07, 0xbde46b40, 0xbca04d74, 0xbdc931b2, 0x3e0963d6, 0x3e13b2f9,
    0xbd109a01, 0xbcdb7cab, 0x3df94991, 0x3d29b697, 0xbd96a31c, 0xbc2c55dd, 0x3ba3d276, 0xbd3af7fc, 0xbc40335d,
    0xbe67a0af, 0xbdfe07e2, 0xbcd338f8, 0xbddc978b, 0xbde81b49, 0x3e44353b, 0x3e47f894, 0xbbc750a9, 0x3e028143,
    0xbe18c27b, 0xbe0629b8, 0x3e97e5ac, 0x3da2ebfa, 0xbe76653e, 0xbea3c7ff, 0xbe6d9774, 0xbe7ecc84, 0xbd6c89ac,
    0x3e0e1f27, 0x3d1b16c4, 0xbe49a73c, 0xbdf50dec, 0xbdf319a7, 0x3d7ebf4e, 0x3db1ffb9, 0x3deb2b5b, 0x3dc28f3f,
    0xbdcd97e3, 0x3e2824ab, 0xbe2a5d0a, 0x3d66c13b, 0x3b3bcc1b, 0x3a853fa3, 0x3d9e0e43, 0xbe6b9036, 0xbcde6295,
    0x3a784e05, 0xbe378ea4, 0xbd08c2f5, 0x3dca2378, 0xbe9821ca, 0xbc0d4bbc, 0x3ca0e8f4, 0xbd773dad, 0x3d9aca8c,
    0xbd0022ba, 0x3deca07e, 0xbdaa49c2, 0x3c6de8c7, 0xbdaffd70, 0xbe9ebf96, 0xbe35b5e5, 0xbc073843, 0xbd4a94d7,
    0x3dfb5c81, 0x3e2e43ff, 0xbe18c7c1, 0x3dd06e3f, 0x3e022f3f, 0x3d096c2b, 0x3c74271f, 0xbe39864c, 0xbde664e5,
    0x3be05e19, 0xbd01a650, 0xbd6ba31d, 0xbd911ac1, 0xbe3ee89a, 0xbe4d4802, 0x3dca140c, 0xbda50b3f, 0xbe6cf1c7,
    0xbdc0f45e, 0x3a20e35d, 0xbebf777a, 0xbe212ea5, 0xbe692c37, 0xbe417b5c, 0xbd26131e, 0xbdf1affe, 0x3d96943c,
    0x3e11b55b, 0xbd5cc288, 0xbe86fc31, 0xbd9ca6f5, 0xbe902aba, 0x3b9ca6ce, 0x3e1a9b0f, 0x3e07c766, 0x3e7a14f3,
    0x3d571afa, 0x3dfc538e, 0xbde097b7, 0xbde9da10, 0xbd274ed6, 0xbe32878e, 0xbd574364, 0x3c91c8d5, 0xbe31b61b,
    0xbd663443, 0x3d3dec82, 0xbe172cce, 0x3c1d65f4, 0x3d3bf02e, 0xbe82e361, 0xbd7ff711, 0x3cb474b5, 0xbdf6aa12,
    0x3dbec100, 0x3c055fd2, 0xbe0014c7, 0x3c9cf24a, 0xbd285874, 0x3c082f64, 0xbd0d10a4, 0xbd0ffd21, 0x3e677893,
    0x3cf77bda, 0xbdb4c47e, 0xbd5edc1f, 0x3e206c35, 0xbdaab6db, 0xbdeff381, 0xbd11d10d, 0xbe181a6a, 0x3d5bdb77,
    0xbd45521e, 0xbd0fe650, 0xbe39d436, 0x3cd00e73, 0xbe4c3a93, 0x3ddc82d8, 0x3c328f03, 0xbbc68ac0, 0x3d4698a4,
    0x3e1000e5, 0x3dc8a15b, 0x3c966358, 0x3d6199a4, 0x3e233e62, 0xbcbd6d6c, 0x3a9a017f, 0x3d6fdd2a, 0xbe3544e4,
    0xbc05314c, 0xbe337465, 0xbe675f65, 0xbd9319df, 0xbe0360e8, 0x3dbdc298, 0x3e117611, 0x3def88cd, 0x3e4a2513,
    0x3db4e0dd, 0x3d693bfa, 0x3e632903, 0x3e15ea5e, 0xbb02d63c, 0xbd056139, 0xbc23f842, 0xbcdcc537, 0x3ae890c7,
    0x3da372f5, 0x3d82105c, 0xbc8364ec, 0xbd698e21, 0xbd467dbc, 0x3e665d59, 0x3e43e9ca, 0x3e6e9566, 0x3e2d3ab1,
    0x3e988eaf, 0x3e9fac98, 0x3be7deb8, 0x3a66b5cd, 0x3e517f36, 0x3e0621b8, 0xbd2653aa, 0x3d3d1d7e, 0x3d2d386c,
    0xbb971b27, 0xbe18d06b, 0x3da3c0b8, 0xbc9de6d6, 0x3d53a5d5, 0xbd433a9b, 0xbbb720a0, 0x3d60352f, 0x3cb25cd2,
    0xbdf71438, 0x3d861acf, 0xbcc6f70f, 0x3d943772, 0xbd656788, 0x3dc69efc, 0xbba32ecd, 0x3e07679c, 0x3da16caa,
    0x3e1f3317, 0x3e33f42b, 0x3e462fda, 0x3e9576bd, 0x3e955fd4, 0xbd499fcc, 0xbc4a2d7d, 0x3d17582d, 0xbdec8fb9,
    0xbd5153b7, 0xbb1eb85c, 0xbd52988f, 0xbdb2241f, 0x3c1db8b6, 0xbafb8170, 0x3d8b9ee8, 0x3ccf39e9, 0x3bf713fb,
    0xbe0cf3ca, 0xbbe59f6a, 0xbb2d9aa2, 0xbd87de2f, 0xbc5f3b4a, 0x3b57e1c7, 0x3d2f64a1, 0xbd5d8bc5, 0x3d7f1ca4,
    0xbd13fdf5, 0x3cb6641c, 0xbce6e9f1, 0x3b54102c, 0xbc8f24f9, 0x3e1ffa32, 0x3e58d327, 0x3e2dfc22, 0x3e9632fc,
    0x3e62eee7, 0x3da48cdf, 0x3da49fd9, 0xbd1384ce, 0x3d55e41f, 0xbd2aa113, 0xbdd2a1e2, 0x3da0f442, 0xbd3264d7,
    0x3db6cc8d, 0x3cf75a5a, 0xbb8dd921, 0xbcb76199, 0xbdbb9a93, 0xbd3cfc3d, 0xbd89d9dc, 0x3d2e4ee3, 0xbd978f86,
    0x3d07af70, 0xbd4c6b43, 0x3d292e55, 0x3d840d81, 0xbdf121bb, 0x3d9578d2, 0x3d03fdc3, 0xbae0c4ea, 0x3db120af,
    0x3e0bc9c0, 0xbdc8226e, 0xbcba043b, 0x3db29247, 0x3b0c17b1, 0x3bae0b1b, 0xbc227da0, 0x3caf88bb, 0xbcaa7bf4,
    0xbd57fe6a, 0xbdab0092, 0xbd41aaf3, 0x3c65ec6a, 0xbd6b1fd1, 0x3e2bfc9f, 0x3e441ac8, 0x3c395ba4, 0x3e398b9e,
    0x3e6749f4, 0x3e366d36, 0x3d6e4d40, 0x3e0e7795, 0x3eaeadd5, 0xbd96fb20, 0xbdc729b5, 0xbd3926ee, 0xbdb7640c,
    0xbba556b8, 0xbd501349, 0x3c34c8ad, 0xbbbc27bc, 0x3c7892c9, 0xbde75a25, 0x3d9332a0, 0x3d91ee54, 0xbced8cbc,
    0x3e967f0f, 0xbda5d7f8, 0x3d4bf61e, 0x3d978f3d, 0xbd2385d7, 0xbcc785c2, 0x3de033b0, 0x3de45430, 0xbc853909,
    0x3e02e6ac, 0x3e3b8aa2, 0xbdffbc51, 0xbd6ceaad, 0xbe1a37e4, 0x3e1668aa, 0x3e9e0295, 0x3d9c0a71, 0x3cde12ee,
    0xbdb2271d, 0x3eb9b2b6, 0xbd5fc27a, 0xbd9b2493, 0x3d904fe6, 0xbd900c33, 0x3d383cef, 0xbb99570d, 0xbd135d94,
    0xbde37e26, 0xbca82a71, 0xbe6f0506, 0x3d1109b5, 0xbe027eb2, 0xbe4997ff, 0xbd680a9a, 0xbe3129ac, 0xbd95fc64,
    0xbd812cf1, 0x3bc70e35, 0xbe86bb10, 0x3ce05e23, 0xbe7ad80b, 0xbda4e792, 0xbd994910, 0x3dc47c12, 0xbd4b8986,
    0xbdd8c6b0, 0x3df923d4, 0xbd121ea3, 0x3e77cbd6, 0x3e1bb77e, 0x3ceeb916, 0x3cdc4c0b, 0x3e430d51, 0x3b410ecb,
    0xbd6c7bfa, 0x3e276b59, 0x3c28995f, 0x3d9a190b, 0x3d1a88ba, 0xbd997b7c, 0xbd7ea817, 0x3da209a8, 0x3da2c128,
    0xbe132bce, 0x3cf797da, 0x3d227847, 0x3d895317, 0xbb3ff812, 0xbd966eea, 0x3d87458b, 0xbc9be13e, 0x3d0a2ce3,
    0x3de91125, 0x3c820f63, 0xbdc4289c, 0xbe4746e9, 0xbe0da6ff, 0x3d6b24ac, 0x3e435936, 0x3e0a7974, 0x3e28a609,
    0x3e8a6968, 0xbe8e7146, 0xbdbda852, 0xbe71164e, 0xbcb1d8af, 0xbdc6dd53, 0xbe99df28, 0xbdd2f841, 0x3ad92b94,
    0xbdcf781d, 0x3d9cd86e, 0xbdf0d788, 0xbe1c8ef6, 0xbe278dea, 0x3d313e47, 0xbd145ada, 0x3d4370e8, 0x3bf9731a,
    0x3e12314a, 0xbe02b6d7, 0x3d02992e, 0x3d1f0c28, 0xbdf1d976, 0x3d36617b, 0x3ddc1030, 0x3c9bcd3a, 0x3e12e5b2,
    0x3dfc76b5, 0xbbbccdee, 0xbc79108f, 0x3d19c293, 0x3c93fed1, 0xbd1cc573, 0xbd1ea97a, 0xbd1113d8, 0xbe015c29,
    0x3db641ec, 0xbcf6b76a, 0xbe1c137a, 0xbdfc82e0, 0xbd9c6d78, 0xbce1d2ee, 0x3d661cca, 0xbe6e238a, 0xbcc88b23,
    0x3d9d6eb0, 0x3e80bed5, 0xbd9c350a, 0x3d8a9597, 0x3d8198f2, 0xbe0a96a0, 0xbe711927, 0xbd270c5a, 0x3d46ed6f,
    0x3d537a15, 0x3dd487f9, 0xbdf363bf, 0xbe16334e, 0xbcb6c825, 0xbc80b8cf, 0xbd52d48b, 0xbe5f7834, 0x3c6662f4,
    0xbe92f147, 0x3e085d87, 0xbc638c87, 0xbe124623, 0xbcfeea55, 0xbebe6448, 0xbea00304, 0xbeb16349, 0xbdf308ff,
    0xbbdc056d, 0xbd3f1218, 0xbeaf75c7, 0xbe8b3390, 0xbe8a9761, 0xbd71977b, 0xbf098206, 0xbe150bf0, 0x3c09fc61,
    0x3c360634, 0xbe6d1205, 0xbe0ebcf3, 0xbe19c705, 0xbe4cdd6a, 0x3da15d6e, 0xbd068c7c, 0x3bd2767e, 0xbd0c625f,
    0xbd9efd9d, 0xbd1ebbae, 0xbcbe9287, 0xbbfc371c, 0xbe38e797, 0xbf040c1a, 0xbe8b7e40, 0xbebd75e7, 0xbe82c8de,
    0xbd0364b8, 0xbe3f94e5, 0xbeef7f49, 0xbe26e418, 0xbe3ffaa7, 0xbe8c30d1, 0xbe64ce40, 0x3dc8e51d, 0xbe208b68,
    0xbe42f42c, 0xbe80e1b1, 0xbe92ff87, 0xbe268254, 0xbde9a0a1, 0x3cf77695, 0xbe0a0018, 0xbe0cf925, 0x3b5fee13,
    0xbde65d0d, 0xbdbf554b, 0xbdb4a067, 0xbe2bd490, 0xbe22073e, 0xbec720c7, 0xbe77e859, 0xbeba23bc, 0xbe87c17d,
    0xbcb1b2c6, 0xbe700833, 0xbebb923e, 0xbd2a31f8, 0xbe93328a, 0xbe5d8301, 0xbdaf5187, 0xbe9286b1, 0xbe810b07,
    0xbe5f12ca, 0xbe71dbeb, 0xbe493f02, 0xbcac22dc, 0xbe7f86a3, 0x3d3bac9d, 0x3e809f02, 0x3ccfe7b8, 0xbd9808a1,
    0xbd77b5b9, 0xbe9a8fbd, 0xbe6db6cb, 0x3b87f6c5, 0xbe30d755, 0x3d3b6f79, 0x3dc7489a, 0xbcf9019e, 0x3da5c93c,
    0x3da02be7, 0xbd48f213, 0xbc96e685, 0xbd0b4e51, 0xbd8f2b4b, 0xbe60e906, 0x3d2273d6, 0xbe42332c, 0xbdc870e9,
    0xbd382c71, 0xbdd3a5d5, 0xbe0b95e9, 0xbd416aa7, 0xbd7ebffe, 0x3d82befb, 0x3d455e27, 0x3dae4156, 0xbc569d5a,
    0x3aba9b91, 0x3bf1bf19, 0xbd2b235f, 0x3d68e747, 0xbd637b1b, 0xbee11b28, 0xbe8e05b0, 0xbecf7679, 0xbe2c8339,
    0xbe0539e8, 0xbea2a105, 0xbeea063f, 0xbe31e896, 0xbedf0372, 0xbd594a79, 0xbe3c39ae, 0xbec96693, 0x3c9cccd2,
    0xbca42ce6, 0x3e3170b8, 0xbde042b3, 0xbecfbda1, 0xbe226fb4, 0x3d86aa13, 0xbdff099b, 0xbe59e091, 0x3db7681f,
    0x3e335c29, 0x3cef0ad9, 0xbbca509c, 0xbc98f27c, 0xbdaf8099, 0xbe9ae277, 0xbe81a8db, 0xbea41e02, 0xbed5fefa,
    0xbda6fd20, 0xbe8f9d06, 0xbf024ebe, 0xbe4d0ab7, 0xbe8846c9, 0x3daf8f6f, 0x3d89afe1, 0x3e024197, 0x3cf0a5c4,
    0xbd8e5375, 0xbd64a1cf, 0x3d9cca0f, 0x3e114c53, 0x3e0db863, 0xbf1b8d6d, 0xbebffd8c, 0xbe8f181f, 0xbf05ef9c,
    0xbeb700f0, 0xbeaf98dd, 0x3c0712bf, 0xbe234385, 0xbe805434, 0x3d3bd67c, 0xbd4d7619, 0x3e14b2cd, 0x3cdbd626,
    0x3c7eb7f2, 0x3db069c7, 0x3ce960a5, 0xbc783972, 0x3d42cb7f, 0x3d97794c, 0x3dbf0c25, 0x3c59087e, 0xbd5ff61a,
    0x3e8fc179, 0xbd3ed23d, 0x3dde5166, 0xbd12509b, 0x3da8afd5, 0xbdd89d33, 0x3d9b0e25, 0xbdff575b, 0xbe4f7be6,
    0xbe7106e0, 0xbec4b8a5, 0xbeaadb18, 0xbee3bc30, 0xbedc7842, 0x3e2d7244, 0xbd68fab3, 0x3dd1ec77, 0x3e73514b,
    0x3d65f7cf, 0x3d00a6c9, 0x3d7fcc9a, 0x3da8e57e, 0x3da02119, 0x3e147077, 0x3b130e8d, 0x3dc2c279, 0x3d0d9ba5,
    0x3de715c9, 0xbd7eea3b, 0x3db93e87, 0x3c8d64e4, 0x3d2aa04a, 0xbc182bd9, 0xbcf593a6, 0x3d3d31fe, 0xbd7e1f3c,
    0xbdfe06b8, 0x39547c8e, 0x3d8022b8, 0x3d9a0314, 0x3c9dc225, 0xbeb8e073, 0xbe8ebf64, 0xbf1f70f6, 0xbed2aa06,
    0xbeeedeba, 0xbec90ba3, 0xbdc88907, 0x3d5824c8, 0xbdb98f8f, 0x3cee08e3, 0x3d90034b, 0xbcab3e2e, 0x3e03b3c2,
    0xbd742438, 0xbdbaadf0, 0x3d8b6cba, 0xbe14afbe, 0xbc234582, 0x3e309509, 0x3dee51f5, 0x3c7e219a, 0x3e3058b6,
    0xbdcf1e0c, 0x3cdd0284, 0x3e0e446f, 0xbd03778c, 0x3e5f8e16, 0xbd41e52f, 0xbc91ed47, 0x3d4ab9d2, 0xbe6ab573,
    0xbe809d8e, 0x3e4c3587, 0x3d392336, 0xbe318ef7, 0xbd3fb814, 0x3c79fe90, 0xbc9289b5, 0xbd9910fc, 0xbd9d9e39,
    0x3d310e72, 0x3d2d2bd4, 0x3db3d34d, 0xbd4458b5, 0x3dfa6fe7, 0xbeb1791c, 0xbed24c77, 0xbeab87de, 0xbf2062af,
    0xbf066a9f, 0xbf62cd70, 0xbeaadaa4, 0xbebf5693, 0xbf001bf5, 0xbe1d3aa7, 0xbd2df943, 0xbe56790a, 0xbded32ca,
    0xbebfae24, 0xbe23d87d, 0xbe44144c, 0xbe682e1c, 0xbe7b314e, 0xbe940316, 0x3de51cc6, 0xbce7b0c5, 0x3e9aac96,
    0xbe58564b, 0xbcfc7fbb, 0x3e11154c, 0x3e91ac27, 0xbde15729, 0xbb98ad3a, 0x3db9dbb1, 0x3d66e50a, 0xbb545c45,
    0xbdd72df9, 0x3caebbe4, 0x3d04ad89, 0x3da62bfe, 0x3dda6d46, 0xbe7d3a0c, 0x3d60f9c5, 0x3d028e2c, 0x3d9a9481,
    0x3e9a40ae, 0x3e64ce9f, 0xbe3386a2, 0x3d2d36f3, 0x3e81b526, 0x3d5cf95d, 0x3d0c726e, 0x3ad72bfc, 0x3e06ec76,
    0xbe581797, 0x3d28f8a8, 0x3d285117, 0xbd6a02e8, 0x3ddafb6b, 0xbdf99e31, 0x3dc7bf59, 0x3cc107af, 0xbda273fb,
    0xbd20ea8b, 0xbcc69720, 0x3d11f635, 0xbd7dc2da, 0x3c2ea0f9, 0xbcb46a53, 0x3e0e6b2d, 0x3d8b90ce, 0xbcf692a7,
    0xbe712b11, 0x3da9997e, 0x3cd24bbe, 0x3dc04424, 0xbcdb5206, 0x3d9d2c2f, 0xbdf326a0, 0x3d8e9afa, 0x3d490073,
    0xbdd7d8da, 0xbc56d01f, 0xbbea4153, 0x3d768115, 0xbbf5a4ca, 0x3ddc7369, 0x3d88751d, 0xbd3e7248, 0x3dec47de,
    0xbd774c53, 0x3d8361c2, 0x3dbdc79c, 0x3caba33b, 0xbc92c923, 0xbd057bad, 0xbd51553a, 0x3cf2975f, 0xbc9bda5c,
    0x3d2ccca2, 0xbdb3e35b, 0xbddce724, 0xbd53491e, 0x3d2ee942, 0xbe758a2a, 0x3eb06d97, 0xbd60c58b, 0x3dc33975,
    0xbe9dbdaf, 0x3de18e9e, 0x3cedfd22, 0x3c3f0d9e, 0xbc225e3b, 0x3d8f6a7c, 0x3c0646a5, 0x3aa10883, 0xbd1d4601,
    0xbe72c1d6, 0x3e460bcf, 0x3c425dd9, 0xbcd31ff5, 0xbd10105f, 0x3cbe6ef1, 0xbc46575b, 0xbd7314a9, 0xbd21aab0,
    0x3e244c22, 0xbbc7a063, 0xbd6e8820, 0x3da04430, 0xbc730bf5, 0xbd54e05d, 0xbc5aa5cc, 0xbd1600d6, 0xbe1dd80e,
    0x3d2a2af0, 0x3d9dffac, 0xbd26c4fb, 0x3dd562c2, 0xbd4c7063, 0xbcd613cc, 0x3bbd6781, 0xbb803f9c, 0xbd9354d0,
    0xbd4294cb, 0xbcf7e809, 0x3c100d23, 0xbc0fbdce, 0xbd752bd2, 0xbda88c9d, 0x3e16c5d8, 0x3c7523fb, 0x3c219d7b,
    0x3e0277bc, 0xbb416c26, 0xbdb1e7de, 0x3df6f675, 0xbde858ae, 0x3d738446, 0x3c8de63c, 0xbdc43101, 0xbd968dbc,
    0x3d66c6cd, 0xbd86587b, 0xbd8b5a4a, 0xbe0b26f1, 0xbcf9abb5, 0x3d0e61ec, 0xbd51e613, 0xbe22f63b, 0x3cab4f28,
    0x3be5ce23, 0x3e0d0d25, 0xbdb44494, 0x3d555c5a, 0xbe3530fb, 0x3debcea0, 0xbd96e93d, 0xbd1b120d, 0xba354dbf,
    0xbcfe3908, 0xbc879199, 0x3dc7875c, 0x3ba1c13b, 0x3d908a56, 0x3db6d71d, 0x3d182b6a, 0xbd76769b, 0x3d18e855,
    0x3ec1676b, 0x3e4f0128, 0xbda60389, 0xbdc8a597, 0x3e2f5ae1, 0x3e6eeeb4, 0xbcfcd459, 0xbd483c8e, 0x3e1f4653,
    0xbe256433, 0x3d8275b5, 0x3ea83f78, 0x3d8cc745, 0xbd96dfab, 0x3e4a630d, 0xbd56d380, 0x3e0da4c4, 0x3d9f31b1,
    0xbc26ec96, 0xbd23d886, 0x3de6867f, 0xbd33f24f, 0x3dbe6110, 0x3ded0325, 0x3de5034e, 0x3e0011d0, 0x3cfbfd5b,
    0x3cb99530, 0xbccb27fe, 0x3d725e2b, 0x3e25ee51, 0x3dde8d49, 0xbd899af3, 0x3cd4b5c4, 0xbd561583, 0xbc68d233,
    0xbe0483b8, 0x3cd09397, 0xbdad8094, 0xbd8828a2, 0xbd3dcb38, 0x3da35ca5, 0x3d7215a4, 0x3dc7dc73, 0x3cc5c1f7,
    0xbddf769b, 0x3dce30f8, 0xbd55dfa3, 0x3d804977, 0xbde30f15, 0x3d986161, 0x3c1bbff3, 0x3d3a3bf9, 0xbdb25e29,
    0x3bf0dd02, 0xbd383577, 0xbdd8a684, 0xbdca23bf, 0x3cc59f04, 0x3e23205b, 0x3e0ed889, 0x3e2317cb, 0x3e648e43,
    0x3cb2bcbf, 0x3d8eb77f, 0x3d83da02, 0xbc8d32b9, 0x3d47ae63, 0x3dd40646, 0x3e4ecfd4, 0x3e988135, 0x3e8725e4,
    0x3bf60bf9, 0x3e0b42dc, 0x3ddcc962, 0x3e2b0f32, 0x3e64ba3f, 0xbd495792, 0x3d335c85, 0x3cbb2c28, 0xbd1c5265,
    0xbe1bd7d8, 0x3b90a165, 0xbd4d86a3, 0x3c57b21c, 0xbd345503, 0x3dc41ec0, 0xbd99c49d, 0x3e47bd11, 0xbc85676a,
    0xbce4eac2, 0xbcf3130f, 0xbe12c175, 0x3cc003fb, 0xbd162687, 0x3d1a12af, 0x3c819924, 0x3d93b3c3, 0x3d231832,
    0x3d717bc5, 0xbd4924d8, 0xbd72086a, 0xbc0440ad, 0x3c169bd1, 0x3e20a475, 0xbd66bf10, 0xbdb871be, 0x3e0a6d2f,
    0x3e802005, 0xbca77174, 0xbd9701ed, 0x3e1992a1, 0x3e00450d, 0xbd0dadbb, 0xbdef35c9, 0xbd0c6ee7, 0xbd26c582,
    0x3bf31d76, 0xbce4b992, 0xbd78109d, 0xbd0babd7, 0x3c92dd0f, 0xbe9ac187, 0xbe4e51a6, 0xbf000625, 0xbedace5b,
    0xbd88662a, 0xbeb91fc7, 0xbf08ae78, 0xbe8ecc2a, 0xbecbd3c3, 0xbde24641, 0xbd04a313, 0x3df74c1a, 0x3d4cb736,
    0xbdae236e, 0xbddeb4ee, 0xbaf51a42, 0x3e0d639c, 0x3e4205af, 0xbeefd731, 0xbe61514f, 0xbe7364c7, 0xbeeec252,
    0xbddeda0c, 0xbeb92910, 0xbe30ca30, 0xbdda4b8f, 0xbf31c82f, 0xbd0fa5a9, 0xbbcc9e66, 0x3e4a5c52, 0xbe46caad,
    0xbcbde117, 0x3d9b9ae8, 0xba53b7d8, 0xbce58bce, 0xbcaa387d, 0xbeb032ec, 0xbe589aec, 0xbe8f5a26, 0xbe1b0ed2,
    0xbc884df5, 0xbe93d57d, 0xbd862fff, 0xbeaa424c, 0xbe6af753, 0xbeb7b8e6, 0xbe348349, 0xbe923aeb, 0xbe8feb0d,
    0xbdd46aa0, 0xbecf0075, 0xbef98116, 0xbeace927, 0xbead44fa, 0x3e413006, 0x3cf7b501, 0xbe117950, 0x3e48187f,
    0x3e112aa5, 0xbdf9c0c7, 0xbb92ebf5, 0x3dc6330c, 0xbc46df47, 0x3db5469e, 0x3d0c7ee3, 0x3d8aa12b, 0x3d0bd4bd,
    0x3ea1dff8, 0x3dfbabdf, 0x3ddbc758, 0x3d7261ab, 0xbd107d45, 0xbe58626c, 0xbea791a5, 0xbe2c9581, 0xbe3f4aa5,
    0x3c4738d5, 0xbd0b472f, 0xbe2996d7, 0x3c4039c1, 0xbe7c6a80, 0xbdfccd14, 0xbeb61cd9, 0xbeaf1da1, 0xbec05deb,
    0xbec1c363, 0xbe4fb6f9, 0xbeb22f5f, 0xbd9d03ec, 0xbd91b06f, 0x3bd21321, 0x3deb755b, 0x3d854853, 0x3d80a66b,
    0xbd8048c4, 0xbe0c2327, 0xbc104d98, 0xbdc21dea, 0x3db9d025, 0xbd122632, 0x3dff71f7, 0xbd87011c, 0x3d48632d,
    0xbe644ab4, 0xbd38a53c, 0x3d013f04, 0xbdbdd538, 0x3ebb838b, 0xbdb72db3, 0xbcea2bf3, 0x3e3dba90, 0xbe656bdf,
    0xbebaecce, 0x3e5a6dab, 0x3dbe1303, 0xbd94a360, 0xbde3b433, 0xbe8330ed, 0xbddca9d0, 0xbf016558, 0xbe844210,
    0xbd807051, 0x3bfda8af, 0xbe84d894, 0xbe718cd6, 0xbe1696b4, 0xbd35db24, 0xbdce579f, 0xbb539680, 0xbd6119f2,
    0x3d1e4085, 0xbe98c668, 0xbe90c747, 0xbe476760, 0xbf5e5b62, 0x3d83c5ce, 0x3dd906b3, 0x3e2e31a1, 0x3d16ca07,
    0x3da3110a, 0x3e5cba7e, 0xbdec39a8, 0xbe83b51c, 0xbd6471bc, 0x3e0af570, 0x3db8a5b0, 0xbd949f5f, 0x3d669d12,
    0x3d852fe1, 0xbc7435c4, 0xbdc14f6e, 0x3e28246a, 0x3e0d0871, 0xbe45d5c0, 0x3e0c0e4e, 0x3d87305e, 0x3dd0287f,
    0xbc086ea7, 0x3e3f4369, 0x3daf636a, 0xbe37368c, 0xbe489618, 0xbde9f7b8, 0xbd8aabb1, 0xbd8aca37, 0xbe1be702,
    0xbcbac744, 0xbcd15aa8, 0xbdd089ec, 0x3d2cfea6, 0x3d66287b, 0x3e008c7a, 0x3c9c202e, 0xbc32fa01, 0x3dc41a26,
    0x3dd196b1, 0x3dc7726e, 0x3ed33c13, 0x3e71f44e, 0x3e879998, 0x3ecadcab, 0x3ed3c343, 0x3ec2f24f, 0x3ed6cf42,
    0x3c83f04a, 0x3ead05a7, 0x3df87ea9, 0x3e4f1091, 0x3eb67dbe, 0x3eb600fa, 0x3e92ed2c, 0x3eaa5a08, 0xbcf09f4e,
    0x3e19500e, 0xbe0ef14c, 0x3e535ff2, 0xbd44cafb, 0x3d24c70c, 0x3ce4a5d1, 0x3e1661ae, 0xbd64d3f6, 0xbdb08825,
    0xbe484720, 0x3e28b3a4, 0x3bf0e0f4, 0x3dc13c27, 0xbc91c51a, 0x3e10e1c6, 0xbe49fae7, 0xbebdc732, 0xbe8f5f95,
    0xbe87e495, 0xbe47fcb3, 0xbe85027a, 0xbe7b107b, 0xbdbd9a58, 0x3cc46eae, 0x3e80cbbc, 0x3dfa9e8a, 0xbd3628bf,
    0xbca7f0e7, 0xbc92e15e, 0xbdb68373, 0xbdf19cf4, 0x3d4a9b58, 0x3dc6c1d1, 0xbdc25082, 0x3ca3db9e, 0x3e38b167,
    0x3e5456d7, 0x3d080111, 0xbc6be41a, 0x3dcb6b3a, 0x3da9eaf2, 0x3d8ceb5a, 0x3db29f8d, 0x3e17e7e5, 0x3e4b00e0,
    0x3dbb4c5f, 0x3d2d4b75, 0x3ddb62ec, 0x3e0e19db, 0x3e1eec10, 0x3e226e67, 0x3d26fe89, 0xbde5963a, 0x3e0ed687,
    0x3de2ffb4, 0x3e323f4f, 0xbaa7635e, 0x3e35ff8b, 0xbc8adc3d, 0xbd88af30, 0xbe25ee74, 0x3df80a23, 0x3d95552d,
    0x3e1ddf2e, 0x3e28139a, 0x3e0f8282, 0xbd1c317f, 0xbd0b08d9, 0xbe489f45, 0xbe1fa03c, 0xbe07f95e, 0xbdafccaf,
    0xbec801f6, 0xbe84ae51, 0xbe3845ed, 0x3d913cae, 0xbe3289e0, 0xbe7befd8, 0xbd4c994f, 0xbe6c8ddf, 0xbe86ffb1,
    0xbef4fbd0, 0xbd8efd05, 0x3d368340, 0xbec4f44e, 0xbe9f0e8c, 0xbeadf5cf, 0xbcced737, 0x3e849e93, 0xbe79593c,
    0xbea63ce3, 0x3cdaaf49, 0x3e0d86f8, 0x3dd4e0f3, 0xbe254c97, 0xbcc898fb, 0x3cd05252, 0xbc669ceb, 0xbbb326a5,
    0xbd95a654, 0xbd9dfabd, 0xbd54072e, 0xbd743bfd, 0x3d2c48da, 0x3cd29410, 0x3d995305, 0x3d75b257, 0x3cd0a975,
    0xbd9ba084, 0x3c7a1505, 0xbd4514bf, 0x3de110cf, 0x3e3bf87c, 0x3e36fe6b, 0x3e390a0e, 0x3e88d264, 0x3e1888e1,
    0x3e6a94db, 0x3e4125ed, 0x3d409bc2, 0x3e70d199, 0x3dc561bc, 0x3e2346d3, 0x3e217970, 0x3e519c11, 0x3e1076ee,
    0x3e1d1cae, 0x3d8816eb, 0xbe54a9c2, 0xbd4b4e90, 0xbd6bef4e, 0xbdfd7336, 0xbed1a844, 0xbe474e8d, 0xbe29b98c,
    0xbd541d0b, 0xbea8ffd1, 0xbeaa9d92, 0xbdfe8ab0, 0xbc802a3d, 0xbe26e781, 0x3cb372e5, 0xbcfc2351, 0xbda6f06a,
    0xbe54fcbd, 0xbe558399, 0xbe57b0ed, 0x3e137585, 0x3d10aad6, 0xbd557a5e, 0xbe0c29ec, 0x3dd93c7a, 0x3e758442,
    0xbd369f2d, 0xbc8d16f1, 0x3e61f23f, 0x3d188dcf, 0x3dcc8bce, 0xbd749b89, 0xbcea9e0d, 0x3ca94377, 0x3d0c72c0,
    0x3dc77018, 0xbd8197df, 0xbd0a9ce4, 0x3d4d3211, 0x3d724ec0, 0xbe2bf33c, 0xbe007892, 0xbe43db75, 0xbd83d554,
    0xbd863a62, 0xbe3926dc, 0xbdbfee6b, 0xbd95da52, 0xbd1518dc, 0x3e892b89, 0x3c3ef34d, 0x3e12dcd4, 0xbcf2d50f,
    0x3e348bb8, 0x3d6781bd, 0x3d611302, 0x3c13b926, 0x3ed84993, 0x3e5cbf5b, 0x3ec9d18b, 0x3e5ec1b1, 0x3eca422a,
    0x3e7649c0, 0x3e623a26, 0xbd19d850, 0xbd52d939, 0xbe438074, 0x3d20dc99, 0xbe54dd6c, 0xbdae57a7, 0x3cbb05cc,
    0x3e3b4596, 0xbe84387c, 0xbd40b20d, 0xbe780378, 0x3ec8d144, 0xbe92e927, 0xbe068802, 0xbdc8a6ae, 0x3d75ad02,
    0x3ca0a16a, 0xbd1887df, 0xbda49477, 0x3e5c180c, 0x3cade570, 0xbe3748bb, 0xbf249a12, 0xbe9927b1, 0x3daa1343,
    0xbef21f45, 0xbf5e7130, 0xbeee3cde, 0xbbad27ba, 0xbe669a25, 0xbeeea742, 0xbe88c3e4, 0xbbc52ed7, 0xbd7df7b1,
    0xbda24b9d, 0xbd19b3ef, 0xbbc15e54, 0xbdf42bb0, 0xbca54ff4, 0xbd45c337, 0xbc39d15a, 0xbdacccfe, 0xbd082469,
    0x3d8bb35d, 0xbda93131, 0xbe0d05e3, 0xbce9d655, 0x3d967340, 0x3ec0d12d, 0x3ee0aea6, 0x3f02e2d1, 0x3e60bada,
    0x3ebb8e09, 0x3eb3aa7d, 0x3eb29100, 0x3e23d1d7, 0x3e80cd87, 0x3e91cb90, 0x3efb3247, 0x3e380078, 0x3e870e28,
    0x3e88060d, 0x3eb2995e, 0x3e51574d, 0x3e1ecd28, 0xbdcaec2c, 0xbccfd1b7, 0xbd1da91f, 0xbe8ce13c, 0xbe770c69,
    0xbc0479f4, 0xbe2e088c, 0x3dba2b3a, 0xbec31919, 0xbe7cdef3, 0xbebfe6a2, 0xbe34439e, 0xbe3cd690, 0xbccaef0e,
    0xbea91f78, 0xbdf3401c, 0xbceec03a, 0xbee28372, 0x3c8cc3ba, 0x3db28b4a, 0xbe5ebcbf, 0xbc99cb86, 0xbe857d08,
    0xbeb6ada9, 0xbdda7d9c, 0xbe7e1bbf, 0x3a40c9e0, 0xbeb06052, 0xbeb4f4b8, 0x3debd1f7, 0xbc3da524, 0x3e08e6d0,
    0x3e7fac58, 0x3dbdbae8, 0x3e50e160, 0x3e252572, 0x3e51676e, 0x3e1c2357, 0x3e08801d, 0x3d0f392b, 0x3e455f51,
    0x3db8b68d, 0x3e6398ef, 0x3e1a363f, 0x3e877f17, 0x3e941407, 0x3eb7d23b, 0xbd135bdf, 0x3e38334b, 0xbba9c345,
    0x3db50af5, 0x3e4dd41f, 0x3de77caa, 0x3e831ef8, 0x3dd69fb1, 0x3e27febb, 0x3e650853, 0x3e2aa865, 0x3e20187a,
    0x3e21ee74, 0xbbfdaee9, 0x3e03988b, 0x3c99275d, 0xbe08489d, 0x3e7f6e80, 0x3d9f6ccd, 0x3dd29e32, 0x3ddf0a7d,
    0x3e0433d6, 0x3e8d3f44, 0xbdb3b20f, 0x3c8b0920, 0x3ed1bb8a, 0xbd50396a, 0xbcd00da9, 0x3cf39b2b, 0x3e031fb7,
    0x3e08a288, 0xbcc3ab9f, 0x3c48e700, 0xbf069d18, 0xbe3642fd, 0xbe96591d, 0xbf6fbdbb, 0xbf5e5f5e, 0xbf840343,
    0xbecec9d9, 0xbedc6cc8, 0xbf2e5e5a, 0xbf08ecb2, 0xbec13087, 0xbe829d14, 0xbe769f3d, 0xbed1d4d8, 0xbe41da51,
    0xbd164aa6, 0xbdbf667c, 0xbd2da403, 0x3c272e1b, 0x3d19b7ae, 0xbdc46470, 0xbd13a9dc, 0xbce37033, 0x3d0607e0,
    0xbcc209ba, 0x3be63460, 0x3d38b867, 0xb9ae02ea, 0xbd76c1e9, 0xbd226dbd, 0x3b8d9333, 0x3ce0aacd, 0x3e9dddaf,
    0x3d5a2fc0, 0x3e5aa0c2, 0x3e99a312, 0x3e93d400, 0x3e94660a, 0x3e620ea8, 0xbccf88e8, 0x3e817827, 0x3db40cfa,
    0x3e285f6b, 0x3e8a65de, 0x3e80bcfa, 0x3e5ce4b2, 0x3dd4e9f9, 0xbdf8bc68, 0xbdec9d9b, 0xbe8db36f, 0x3c969ce6,
    0xbae76b83, 0xbdba1e62, 0xbdf7bfe7, 0xbe30af18, 0xbe0727af, 0xbe20d688, 0xbecc2895, 0xbec628a5, 0xbd85ca48,
    0xbdaafec8, 0xbe5e7a72, 0xbe766293, 0x3d5497ae, 0x3dad8f8d, 0xbc7682f1, 0xbd9b3187, 0xbe75d4aa, 0xbd8910f5,
    0x3c9e4cac, 0x3d9954fe, 0xbde9bb90, 0xbebda767, 0x3cd4008b, 0xbc591ba0, 0xbe81e34a, 0xbdda157a, 0xbd33856b,
    0x3d7c8307, 0xbcd105b2, 0x3dea4b15, 0x3ddb81b1, 0x3e19d361, 0x3d44848c, 0x3e5016b2, 0x3e59c016, 0x3e4b04dd,
    0xbb6388ea, 0x3e01debf, 0x3ddb1599, 0x3e35935d, 0x3d4c73df, 0x3debede5, 0x3e14b358, 0x3dcb4806, 0x3ebacc76,
    0x3e8db08f, 0x3e9595ae, 0x3e88cf53, 0x3e4b3a4c, 0x3d79dbdc, 0x3e487377, 0x3e13d76e, 0x3ed88087, 0x3e888555,
    0x3eaff2e8, 0x3e87e2d4, 0x3e939fc0, 0x3e3f8490, 0x3eb01150, 0x3ea52343, 0x3e6a820b, 0x3cbe3a4a, 0x3e3045f2,
    0x3e3f1fc7, 0x3c86772c, 0xbe5d009b, 0xbdeee93f, 0x3d9795b0, 0x3deeb8f1, 0x3ccc867c, 0x3cd68688, 0x3e6ad18c,
    0xbc01b6ef, 0xbd09bd16, 0x3c21377a, 0x3decadf9, 0xbdd41eeb, 0x3dfddc81, 0x3d12f51d, 0x3d1eb617, 0x3dc87382,
    0x3e77a99c, 0x3ee66910, 0x3cf4ae98, 0xbe312397, 0x3e90b167, 0xbd8e3739, 0x3d6cd3b8, 0xbe4ba8c6, 0xbe8e06d2,
    0xbdf60c3e, 0xbe6ba300, 0x3d9c7249, 0xbd8de785, 0x3d076f72, 0xbc67f883, 0xbe3e73af, 0xbdeea836, 0xbe3ad7a1,
    0x3c3d285a, 0x3da38315, 0xbdafc793, 0x3c90a28b, 0xbccca502, 0xbd9b2a27, 0xbd96bba2, 0xbd8885e1, 0x3cfaaeb1,
    0x3ec853e4, 0x3c8f279a, 0x3ee08801, 0x3d597e25, 0x3e5f5d2e, 0x3e9b121d, 0x3e802b20, 0x3e6ba08d, 0x3eb68096,
    0x3cd1b90b, 0x3edcc559, 0x3dd20b21, 0x3da53599, 0x3e37a893, 0x3dec69c4, 0x3ddc4713, 0x3e196d46, 0xbc90d995,
    0x3e2f7cfc, 0x3e2fbc51, 0x3d34f933, 0x3b0bd370, 0x3df844cf, 0x3d4e570c, 0x3ea34771, 0x3dcbbeac, 0x3e966f64,
    0x3d74b8d0, 0xbdf00209, 0xbe168aa1, 0xbc5d0aee, 0xbd6ded21, 0x3e20ef8f, 0xbe07e6b4, 0x3e01ce34, 0xbe159519,
    0xbec06742, 0x3d64d88e, 0xbeacd755, 0x3e1683a2, 0xbcef10a6, 0xbf067ea5, 0x3d1dfa6a, 0xbde880b6, 0xbe25b92b,
    0x3dd81ee7, 0xbea3711b, 0x3db0a083, 0x3d91e498, 0xbdbe8797, 0x3d673f73, 0xbd03e67b, 0xbda41087, 0xbe40652b,
    0xbd248197, 0xbdae6ce9, 0xbc6e4f55, 0xbdb85afd, 0xbcf0b264, 0xbd783dfb, 0xbdc95bcb, 0xbe1fdacf, 0xbd9ba07e,
    0xbdc82996, 0xbe31dd03, 0x3c0760c7, 0xbd2466ce, 0x3c8e4e41, 0x3d461d63, 0x3e9b1579, 0x3e0dd201, 0x3e2dfa61,
    0xbde44c72, 0x3e1b15b8, 0x3d5e595a, 0x3dcd0dde, 0xbdf79bbc, 0x3df24198, 0x3cd7641c, 0x3de74e02, 0xbd976881,
    0xbe93cf7c, 0xbe823ff9, 0xbe695cd0, 0x3a7d1c63, 0xbd7c7ed2, 0xbdfaa79a, 0x3cad3146, 0xbd378122, 0xbec97f22,
    0xbe240290, 0xbe516a42, 0x3dd11f88, 0xbd186547, 0x3d57127c, 0x3d660128, 0x3e13fd07, 0x3e337ee7, 0x3e101870,
    0xbd928573, 0x3f22af62, 0x3f136044, 0x3ecd7bab, 0x3c6d216f, 0x3ed31fbf, 0x3eb0fcdb, 0x3edfd14e, 0x3cd54e16,
    0x3e854c3c, 0x3ebfad15, 0x3ec1756a, 0xbb110835, 0xbd965f98, 0xbd4a565f, 0xbdf3e16a, 0xbc948e1a, 0x3d4129b5,
    0xbd42c631, 0xbccc9f13, 0xbd930d62, 0xbdcc7486, 0x3c18fad0, 0xbdcefbf3, 0xbc5e16a1, 0x3cd8950b, 0x3d24f2e3,
    0x3d758cb7, 0xbba16a98, 0xbc6b7bf5, 0x3e9ab523, 0x3d186352, 0x3e632dc5, 0x3e0e34ff, 0x3e015369, 0x3eb62776,
    0x3e82a89b, 0x3c13cbe7, 0x3e7c4a2d, 0x3cc535ee, 0x3e6aac22, 0x3d19d54d, 0x3c978737, 0x3e3f9516, 0x3dbe824d,
    0xbe3a0ce1, 0x3e8bb3ab, 0xbd86bb21, 0x3e6c1fa3, 0xbd376c8d, 0x3eabff96, 0x3e9d6dd9, 0x3ea9d270, 0xbda71695,
    0x3edf6b4a, 0x3df92002, 0x3ece5b77, 0xbde05f59, 0x3baf6952, 0x3c2697f2, 0x3df44845, 0x3e310a97, 0x3e0bf011,
    0xbdb7cffe, 0x3e802ec2, 0xbd59f85f, 0xbee86dc4, 0xbdf2b3b3, 0xbcc03284, 0xbe0edd69, 0x3cb8f3c0, 0xbe968210,
    0x3e70c49d, 0xbde444cf, 0xbdd1d9e5, 0x3d881f8c, 0xbda535ee, 0x3d6ade63, 0xbd889309, 0x3d90664d, 0xbd23feaa,
    0xbd6bfee4, 0xbda0949a, 0xbd0e2817, 0xbd4dd80c, 0x3d0ee222, 0xbd01b428, 0x3d3fd751, 0x3b070628, 0xbd4d10f0,
    0xbd8a0c16, 0xbd409808, 0xbd3ba3f1, 0x3db43495, 0x3e2eee66, 0x3e6b1bc7, 0x3d83e5f3, 0x3e82fe93, 0x3ecae6a8,
    0x3e9a4650, 0x3e8399dc, 0x3e971ddc, 0x3ec40729, 0x3ed6bd62, 0x3e51aa96, 0x3e5a8e3d, 0x3ea88883, 0x3e9efa2f,
    0x3e476c33, 0x3c295e87, 0xbf11720d, 0xbecca3e8, 0xbf4d4577, 0xbdb96ded, 0xbe006bdf, 0xbeb33bd2, 0xbebfdb22,
    0x3e33a7dd, 0xbebf8267, 0xbe6d73a7, 0xbef4ec75, 0x3cbc86f0, 0x3d91776a, 0xbe32aea5, 0xbd9ee3a8, 0xbe721155,
    0xbea1977d, 0xbe922574, 0xbe66e32d, 0xbc3e0037, 0xbe00dbb3, 0xbdc2671e, 0xbdb01aed, 0xbd5b1479, 0xbce0ecb2,
    0xbe748ac6, 0xbe021e2e, 0x3dd97800, 0xbc09b521, 0xbc8e3a35, 0xbde41a3e, 0xbdee1cdb, 0xbda6f073, 0xbb6814e6,
    0xbd941776, 0xbdea966d, 0xbdb9af0b, 0x3d1a0bb7, 0xbdc2cce7, 0xbc222661, 0xbc226dcf, 0x3e006f1d, 0xbc35d65d,
    0xbcca9159, 0xbdd6e2f7, 0x3d6a2312, 0xbdbf5190, 0xbefaf38b, 0xbe43ac1e, 0xbeea23a4, 0xbe34d39a, 0xbdd0df86,
    0x3c4eb2c8, 0xbddfc16a, 0xbd07cec9, 0xbec47ad4, 0xbd9602c4, 0xbeafee06, 0xbdd3031f, 0xbe5d184d, 0xbde68195,
    0xbe6a391f, 0xbded74e5, 0xbe9e6229, 0xbdc66ace, 0xbe8d6422, 0xbe1ec6bf, 0xbd736097, 0xbe402c17, 0xbd3b0c47,
    0xbdbf53b4, 0xbeac66a2, 0xbe4692cf, 0xbec93a69, 0xbe3b538e, 0xbdb17a25, 0xbd19986f, 0x3c79623f, 0xbd86ccc5,
    0x3edd7c27, 0x3d682734, 0x3ec43946, 0x3e0cbff8, 0x3ebe0a95, 0x3ef8e908, 0x3eea72b3, 0x3e916269, 0x3f14fd84,
    0x3ef00467, 0x3f226c0c, 0x3e5956f4, 0x3ec88cdf, 0x3eb4f503, 0x3ec12c82, 0x3e95cdb6, 0x3c3d75d5, 0x3e1d7d32,
    0x3c34de7e, 0x3d657642, 0x3e3190e5, 0x3e3d7996, 0x3daa9937, 0x3d9435d5, 0x3d998739, 0x3e1cc2d1, 0x3d12db31,
    0x3dbecd4e, 0x3d5dae3a, 0x3e0be868, 0x3d74dd21, 0x3da0a708, 0xbd39c92e, 0x3e7d2783, 0x3c2a1f0a, 0x3e433397,
    0x3db5e0ae, 0x3e002259, 0x3e5a836a, 0xbc3afc34, 0xbd750f0e, 0x3e3a3e09, 0x3ceea922, 0x3dcf4972, 0x3e153e57,
    0x3e5a2a8c, 0x3e93ce8e, 0x3e06da70, 0xbd2fd40c, 0x3edb560d, 0x3d3f8e37, 0x3e386011, 0xbdd25099, 0x3e59305e,
    0xbd2ab3a1, 0xbd50687c, 0xbe790b1d, 0x3ec82529, 0xbb031a0c, 0x3e538447, 0xbb62c909, 0x3e18bc22, 0x3e3036bc,
    0xbd1f8948, 0x3dc4e0b5, 0xbf1dba25, 0xbe818af8, 0xbee6d3a4, 0xbf0525ac, 0xbf1671f5, 0xbf488a6f, 0xbe956066,
    0xbec26f7a, 0xbf0070ba, 0xbec48f8b, 0x3cfd4cc9, 0xbf26cc85, 0xbf0c9d9f, 0xbefae485, 0x3dae2096, 0xbc783612,
    0xbbb4af01, 0xbd54bee2, 0x3cdabb86, 0xbbbfb1cb, 0x3cf9dceb, 0xbcbcdd9d, 0x3d94e499, 0x3d11a884, 0x3ccc54e6,
    0xbcac3a98, 0x3ca9fae5, 0x3d20af29, 0x3d5dd563, 0xbb434c2f, 0x3d3d8755, 0x3d8504bf, 0xbe04f6bc, 0x3c9fa050,
    0xbe045efc, 0xbe05eab2, 0xbe433b46, 0xbe21bbb7, 0xbe6c89fd, 0xbd27ba46, 0xbe9bd388, 0xbd5dfcc8, 0xbe987ec3,
    0xbe5c3701, 0xbe6785d7, 0xbe7110ca, 0xbe7dd07d, 0x3c8dcf06, 0xbde5ce75, 0x3cf82d47, 0xbe417719, 0xbe47cfd1,
    0xbe241f18, 0xbe7e6256, 0xbe67cf0f, 0xbdfbfaf8, 0x3c46f756, 0x3dc11c4e, 0x3da84844, 0xbe90f685, 0xbe01bb1d,
    0xbdd7f601, 0xbc40087e, 0xbd90ed01, 0xbb1d31d4, 0x3e3fcb54, 0x3e028057, 0xbea8a95b, 0xbdb1a74d, 0xbe249c61,
    0x3d222d04, 0xbe19a93c, 0xbe6e8912, 0xbe37c62a, 0xbdbfb059, 0xbd13b7eb, 0xbdb16d26, 0xbea5f10e, 0xbdd8623f,
    0x3c91a192, 0x3d39a815, 0x3d615af7, 0xbce90427, 0xbd13ae0c, 0xbd29b0dd, 0x3e0b90d8, 0xbd432106, 0xbc74e375,
    0x3d15e3aa, 0xbc80600d, 0xbcf79ca4, 0xbd553c7d, 0xbdfd2f57, 0xbcae7411, 0xbe0f6f4d, 0xbea4c786, 0x3e445dd6,
    0xbdff7947, 0x3e453b5c, 0x3dcd19fc, 0x3ee96bc1, 0x3e734fa6, 0x3e9b48f0, 0xbe4450fa, 0x3eaa88f7, 0x3c648368,
    0x3e903ada, 0x3d9489c1, 0x3e8fce32, 0x3e12ef56, 0x3e428355, 0xbe42fdfb, 0xbeaee3cb, 0xbeba79fb, 0x3cbd4ef8,
    0x3dc043d0, 0xbe88d76d, 0x3b37fdc0, 0x3d6bdccf, 0xbe1a0f4d, 0xbee6c097, 0xbefd3787, 0xbe132521, 0x3d19444d,
    0xbdad705f, 0x3dd03772, 0x3debfcea, 0x3e80411b, 0x3e0346b3, 0x3ec0cadf, 0x3d772e3f, 0x3e8d923a, 0x3f2d8a6a,
    0x3f4ad0d7, 0x3e77f435, 0x3ed8bc61, 0x3ef4bc24, 0x3f2b67e6, 0x3e45f462, 0x3e2eac50, 0x3e77bfd0, 0x3e997aaf,
    0xbcc1430a, 0x3d79a10f, 0xbde07dbc, 0x3da5a4f4, 0x3e2bdcbc};

static void a4kLit(std::string& s, uint32_t bits);

// RIFE's encode kernels after the main text, in their own extern "C" block, every weight of kRifeEncW an exact float
// literal (a4kLit): encC0At (cnn0 at one half-size position into the tile), k_encC01, encC2At (cnn2 at one tile
// position), k_encC2Up. Each output's FMAs run in PyTorch's order (input channel, kernel row, kernel column).
static void encKernelText(std::string& s)
{
    const uint32_t* w = kRifeEncW;
    constexpr int W0 = 0, B0 = 432, W1 = 448, B1 = 2752, W2 = 2768, B2 = 5072, WT = 5088;
    constexpr int BT = WT + 16 * kRifeFeatC * 16;
    char ln[192];
    // "<pre>a = fmaf(v, LIT, a);" for an accumulator name a
    auto fma = [&](const char* pre, const char* acc, const char* v, uint32_t bits) {
        sprintf_s(ln, "%s%s = fmaf(%s, ", pre, acc, v);
        s += ln;
        a4kLit(s, bits);
        sprintf_s(ln, ", %s);\n", acc);
        s += ln;
    };
    char acc[16], v[8];
    s +=
        "\nextern \"C\" {\n"
        "__device__ __forceinline__ void encC0At(const void* __restrict__ img, int inHalf, int W, int H, int X, int Y,\n"
        "                                        float* __restrict__ d)\n{\n"
        "    const size_t plane = (size_t)W * H;\n"
        "    const int y0 = 2 * Y - 1, x0 = 2 * X - 1;\n"
        "    float v;\n";
    for (int k = 0; k < 16; k++)
    {
        sprintf_s(ln, "    float a%d = ", k);
        s += ln;
        a4kLit(s, w[B0 + k]);
        s += ";\n";
    }
    for (int c = 0; c < 3; c++)
        for (int ky = 0; ky < 3; ky++)
            for (int kx = 0; kx < 3; kx++)
            {
                sprintf_s(ln, "    v = encC0Tap(img, inHalf, W, plane, %d, y0 + %d, x0 + %d);\n", c, ky, kx);
                s += ln;
                for (int k = 0; k < 16; k++)
                {
                    sprintf_s(acc, "a%d", k);
                    fma("    ", acc, "v", w[W0 + ((k * 3 + c) * 3 + ky) * 3 + kx]);
                }
            }
    for (int k = 0; k < 16; k++)
    {
        sprintf_s(ln, "    d[%d] = encLrelu(a%d);\n", k * 340, k);
        s += ln;
    }
    s += "}\n"
         "__global__ void __launch_bounds__(256)\n"
         "    k_encC01(const void* __restrict__ img, int inHalf, int W, int H, float* __restrict__ out)\n{\n"
         "    __shared__ float s[16 * 10 * 34];\n"
         "    const int w = W >> 1, h = H >> 1, bx = blockIdx.x * 32, by = blockIdx.y * 8;\n"
         "    for (int p = threadIdx.y * 32 + threadIdx.x; p < 340; p += 256)\n"
         "    {\n"
         "        const int X = bx - 1 + p % 34, Y = by - 1 + p / 34;\n"
         "        if (X >= 0 && X < w && Y >= 0 && Y < h)\n"
         "            encC0At(img, inHalf, W, H, X, Y, s + p);\n"
         "        else\n"
         "        {\n"
         "#pragma unroll\n"
         "            for (int k = 0; k < 16; k++)\n"
         "                s[k * 340 + p] = 0.0f;\n"
         "        }\n"
         "    }\n"
         "    __syncthreads();\n"
         "    const int x = bx + threadIdx.x, y = by + threadIdx.y;\n"
         "    const float* t = s + threadIdx.y * 34 + threadIdx.x;\n"
         "    float v00, v01, v02, v10, v11, v12, v20, v21, v22;\n";
    for (int k = 0; k < 16; k++)
    {
        sprintf_s(ln, "    float a%d = ", k);
        s += ln;
        a4kLit(s, w[B1 + k]);
        s += ";\n";
    }
    for (int c = 0; c < 16; c++)
    {
        for (int ky = 0; ky < 3; ky++)
        {
            sprintf_s(ln, "    v%d0 = t[%d]; v%d1 = t[%d]; v%d2 = t[%d];\n", ky, c * 340 + ky * 34, ky,
                      c * 340 + ky * 34 + 1, ky, c * 340 + ky * 34 + 2);
            s += ln;
        }
        for (int ky = 0; ky < 3; ky++)
            for (int kx = 0; kx < 3; kx++)
            {
                sprintf_s(v, "v%d%d", ky, kx);
                for (int k = 0; k < 16; k++)
                {
                    sprintf_s(acc, "a%d", k);
                    fma("    ", acc, v, w[W1 + ((k * 16 + c) * 3 + ky) * 3 + kx]);
                }
            }
    }
    s += "    if (x >= w || y >= h)\n"
         "        return;\n"
         "    const size_t plane = (size_t)w * h, oi = (size_t)y * w + x;\n";
    for (int k = 0; k < 16; k++)
    {
        sprintf_s(ln, "    out[%d * plane + oi] = encLrelu(a%d);\n", k, k);
        s += ln;
    }
    s +=
        "}\n"
        "__device__ __forceinline__ void encC2At(const float* __restrict__ s, int r, int q, float* __restrict__ a)\n{\n"
        "    const float* t = s + r * 36 + q;\n"
        "    float v00, v01, v02, v10, v11, v12, v20, v21, v22;\n";
    for (int k = 0; k < 16; k++)
    {
        sprintf_s(ln, "    a[%d] = ", k);
        s += ln;
        a4kLit(s, w[B2 + k]);
        s += ";\n";
    }
    for (int c = 0; c < 16; c++)
    {
        for (int ky = 0; ky < 3; ky++)
        {
            sprintf_s(ln, "    v%d0 = t[%d]; v%d1 = t[%d]; v%d2 = t[%d];\n", ky, c * 432 + ky * 36, ky,
                      c * 432 + ky * 36 + 1, ky, c * 432 + ky * 36 + 2);
            s += ln;
        }
        for (int ky = 0; ky < 3; ky++)
            for (int kx = 0; kx < 3; kx++)
            {
                sprintf_s(v, "v%d%d", ky, kx);
                for (int k = 0; k < 16; k++)
                {
                    sprintf_s(acc, "a[%d]", k);
                    fma("    ", acc, v, w[W2 + ((k * 16 + c) * 3 + ky) * 3 + kx]);
                }
            }
    }
    s += "#pragma unroll\n"
         "    for (int k = 0; k < 16; k++)\n"
         "        a[k] = encLrelu(a[k]);\n"
         "}\n"
         "__global__ void __launch_bounds__(256)\n"
         "    k_encC2Up(const float* __restrict__ in, int w2, int h2, void* __restrict__ out, int outHalf)\n{\n"
         "    __shared__ float s[16 * 12 * 36];\n"
         "    const int bx = blockIdx.x * 32, by = blockIdx.y * 8, tid = threadIdx.y * 32 + threadIdx.x;\n"
         "    encLoad36(s, in, w2, h2, bx, by);\n"
         "    float a0[16], a1[16];\n"
         "    {\n"
         "        const int r = tid / 34, q = tid % 34, X = bx - 1 + q, Y = by - 1 + r;\n"
         "        encC2At(s, r, q, a0);\n"
         "        if (X < 0 || X >= w2 || Y < 0 || Y >= h2)\n"
         "        {\n"
         "#pragma unroll\n"
         "            for (int k = 0; k < 16; k++)\n"
         "                a0[k] = 0.0f;\n"
         "        }\n"
         "    }\n"
         "    const bool two = tid + 256 < 340;\n"
         "    if (two)\n"
         "    {\n"
         "        const int p = tid + 256, r = p / 34, q = p % 34, X = bx - 1 + q, Y = by - 1 + r;\n"
         "        encC2At(s, r, q, a1);\n"
         "        if (X < 0 || X >= w2 || Y < 0 || Y >= h2)\n"
         "        {\n"
         "#pragma unroll\n"
         "            for (int k = 0; k < 16; k++)\n"
         "                a1[k] = 0.0f;\n"
         "        }\n"
         "    }\n"
         "    __syncthreads();\n"
         "#pragma unroll\n"
         "    for (int k = 0; k < 16; k++)\n"
         "    {\n"
         "        s[k * 340 + tid] = a0[k];\n"
         "        if (two)\n"
         "            s[k * 340 + tid + 256] = a1[k];\n"
         "    }\n"
         "    __syncthreads();\n"
         "    const int tx = threadIdx.x, ty = threadIdx.y, n = bx + tx, m = by + ty;\n"
         "    if (n >= w2 || m >= h2)\n"
         "        return;\n"
         "    const int W = 2 * w2;\n"
         "    const size_t plane = (size_t)W * (2 * h2);\n"
         "    const float* rp;\n"
         "    float vl, vc, vr;\n";
    // cnn3: output row 2m + i reads tile rows ty + 1 + dy (kernel row ky); p = column 2n, q = column 2n + 1
    for (int i = 0; i < 2; i++)
    {
        s += "    {\n";
        for (int k = 0; k < kRifeFeatC; k++)
        {
            sprintf_s(ln, "        float p%d = ", k);
            s += ln;
            a4kLit(s, w[BT + k]);
            sprintf_s(ln, ", q%d = ", k);
            s += ln;
            a4kLit(s, w[BT + k]);
            s += ";\n";
        }
        for (int c = 0; c < 16; c++)
            for (int t = 0; t < 2; t++)
            {
                const int dy = i == 0 ? -t : 1 - t, ky = i == 0 ? 1 + 2 * t : 2 * t;
                sprintf_s(ln,
                          "        rp = s + (%d + ty + %d) * 34 + tx;\n        vl = rp[0]; vc = rp[1]; vr = rp[2];\n",
                          c * 10, 1 + dy);
                s += ln;
                for (int k = 0; k < kRifeFeatC; k++)
                {
                    const int b = WT + ((c * kRifeFeatC + k) * 4 + ky) * 4;
                    char pa[8], qa[8];
                    sprintf_s(pa, "p%d", k);
                    sprintf_s(qa, "q%d", k);
                    fma("        ", pa, "vc", w[b + 1]);
                    fma("        ", pa, "vl", w[b + 3]);
                    fma("        ", qa, "vr", w[b + 0]);
                    fma("        ", qa, "vc", w[b + 2]);
                }
            }
        sprintf_s(ln, "        const size_t row = (size_t)(2 * m + %d) * W + 2 * n;\n        if (outHalf)\n        {\n",
                  i);
        s += ln;
        for (int k = 0; k < kRifeFeatC; k++)
        {
            sprintf_s(
                ln,
                "            *(unsigned int*)((unsigned short*)out + %d * plane + row) = (unsigned int)f2h(p%d) | "
                "((unsigned int)f2h(q%d) << 16);\n",
                k, k, k);
            s += ln;
        }
        s += "        }\n        else\n        {\n";
        for (int k = 0; k < kRifeFeatC; k++)
        {
            sprintf_s(ln, "            *(float2*)((float*)out + %d * plane + row) = make_float2(p%d, q%d);\n", k, k, k);
            s += ln;
        }
        s += "        }\n    }\n";
    }
    s += "}\n} // extern \"C\"\n";
}

// NVRTC once, then the cubin is cached on disk in the model cache, keyed by source hash and
// device arch (no nvcc anywhere).
static bool nativeBindKernels(NativeRife& nr)
{
    struct
    {
        CUfunction* fn;
        const char* nm;
    } fns[] = {
        {&nr.fPackInDirect, "k_packInDirect"},
        {&nr.fResizeH, "k_resizeH"},
        {&nr.fResizeV, "k_resizeV"},
        {&nr.fH2f, "k_h2f"},
        {&nr.fF2h, "k_f2h"},
        {&nr.fPackOut, "k_packOut"},
        {&nr.fPackInDirectHdr, "k_packInDirectHdr"},
        {&nr.fPqPlanar, "k_pqPlanar"},
        {&nr.fResizeHf, "k_resizeHf"},
        {&nr.fPackOutHdr, "k_packOutHdr"},
        {&nr.fSdrEncode, "k_sdrEncode"},
        {&nr.fThdrColor, "k_thdrColor"},
        {&nr.fThdrIn, "k_thdrIn"},
        {&nr.fSharpThdrIn, "k_sharpThdrIn"},
        {&nr.fPqOut, "k_pqOut"},
        {&nr.fSdrPq, "k_sdrPq"},
        {&nr.fPqLut, "k_pqLut"},
        {&nr.fPackBgra, "k_packBgra"},
        {&nr.fFitPlanar, "k_fitPlanar"},
        {&nr.fSharpPlanar, "k_sharpPlanar"},
        {&nr.fPackBgraRgb, "k_packBgraRgb"},
        {&nr.fUnpackBgraRgb, "k_unpackBgraRgb"},
        {&nr.fUnpackRgba, "k_unpackRgba"},
        {&nr.fPackR10, "k_packR10"},
        {&nr.fUnpackR10, "k_unpackR10"},
        {&nr.fFitAaH, "k_fitAaH"},
        {&nr.fFitAaV, "k_fitAaV"},
        {&nr.fFsrHdrIn, "k_fsrHdrIn"},
        {&nr.fFsrHdrOut, "k_fsrHdrOut"},
        {&nr.fFsrHdrPack, "k_fsrHdrPack"},
        {&nr.fRsTab, "k_rsTab"},
        {&nr.fPackOutV, "k_packOutV"},
        {&nr.fPackOutHdrV, "k_packOutHdrV"},
        {&nr.fRestIn, "k_restIn"},
        {&nr.fRestFoldH, "k_restFoldH"},
        {&nr.fRestFoldV, "k_restFoldV"},
        {&nr.fRestFoldHSub, "k_restFoldHSub"},
        {&nr.fA4kFoldHSub, "k_a4kFoldHSub"},
        {&nr.fA4kFold2x, "k_a4kFold2x"},
        {&nr.fRestConv0, "k_restConv0"},
        {&nr.fRestConvMid, "k_restConvMid"},
        {&nr.fRestConvLast, "k_restConvLast"},
        {&nr.fAjFirst, "k_ajFirst"},
        {&nr.fAjAtt, "k_ajAtt"},
        {&nr.fAjCat, "k_ajCat"},
        {&nr.fRestToF, "k_restToF"},
        {&nr.fClamp01, "k_clamp01"},
        {&nr.fPadPlanar, "k_padPlanar"},
        {&nr.fNrIn, "k_nrIn"},
        {&nr.fNrOut, "k_nrOut"},
        {&nr.fNrMv, "k_nrMv"},
        {&nr.fNrMvClear, "k_nrMvClear"},
        {&nr.fNrInPq, "k_nrInPq"},
        {&nr.fNrOutPq, "k_nrOutPq"},
        {&nr.fHalf, "k_half"},
        {&nr.fPyr, "k_pyr"},
        {&nr.fSplatSoft, "k_splatSoft"},
        {&nr.fSplatNorm, "k_splatNorm"},
        {&nr.fGsCount, "k_gsCount"},
        {&nr.fGsScanA, "k_gsScanA"},
        {&nr.fGsScanB, "k_gsScanB"},
        {&nr.fGsScanC, "k_gsScanC"},
        {&nr.fGsFill, "k_gsFill"},
        {&nr.fGsGather, "k_gsGather"},
        {&nr.fGsGatherNhwc, "k_gsGatherNhwc"},
        {&nr.fPackInRaw16, "k_packInRaw16"},
        {&nr.fPackInRaw8, "k_packInRaw8"},
        {&nr.fPackOutRaw16, "k_packOutRaw16"},
        {&nr.fPackOutRaw8, "k_packOutRaw8"},
        {&nr.fExpand8to16, "k_expand8to16"},
        {&nr.fPairDiff, "k_pairDiff"},
        {&nr.fRawDiff, "k_rawDiff"},
        {&nr.fNvofLuma, "k_nvofLuma"},
        {&nr.fNvofUp, "k_nvofUp"},
        {&nr.fNvofMetric, "k_nvofMetric"},
        {&nr.fSplatVel, "k_splatVel"},
        {&nr.fVelNorm, "k_velNorm"},
        {&nr.fPpDown, "k_ppDown"},
        {&nr.fPpTop, "k_ppTop"},
        {&nr.fPpUp, "k_ppUp"},
        {&nr.fBlur1, "k_blur1"},
        {&nr.fNvofCompose, "k_nvofCompose"},
        {&nr.fDrFlowSplat, "k_drbaFlowSplat"},
        {&nr.fDrFlowNorm, "k_drbaFlowNorm"},
        {&nr.fDrDrmSplat, "k_drbaDrmSplat"},
        {&nr.fDrDrmNorm, "k_drbaDrmNorm"},
        {&nr.fMotionIn, "k_motionIn"},
        {&nr.fRifeBlend, "k_rifeBlend"},
        {&nr.fRifeBlendF, "k_rifeBlendF"},
        {&nr.fRifeBlendOut, "k_rifeBlendOut"},
        {&nr.fRifeBlendOut32, "k_rifeBlendOut32"},
        {&nr.fBilin, "k_bilin"},
        {&nr.fHdrEnc, "k_hdrEnc"},
        {&nr.fHdrFromSdr, "k_hdrFromSdr"},
        {&nr.fHdrRange, "k_hdrRange"},
        {&nr.fPackBgraSdr, "k_packBgraSdr"},
        {&nr.fUnpackBgraSdr, "k_unpackBgraSdr"},
        {&nr.fPackRgbaH, "k_packRgbaH"},
        {&nr.fRgbaHAbgr, "k_rgbaHAbgr"},
        {&nr.fUnpackRgbaH, "k_unpackRgbaH"},
        {&nr.fMvHalf, "k_mvHalf"},
        {&nr.fMvHalfG1, "k_mvHalfG1"},
        {&nr.fMvDerive, "k_mvDerive"},
        {&nr.fBgraRgbaFsr, "k_bgraRgbaFsr"},
        {&nr.fFuMvScale, "k_fuMvScale"},
        {&nr.fRgbaBgra, "k_rgbaBgra"},
        {&nr.fOddEdge, "k_oddEdge"},
        {&nr.fHdrRestOut, "k_hdrRestOut"},
        {&nr.fLocalCorr, "k_localCorr"},
        {&nr.fAttn2, "k_attn2"},
        {&nr.fFuTail, "k_fuTail"},
        {&nr.fFuTailNhwc, "k_fuTailNhwc"},
        {&nr.fEncC01, "k_encC01"},
        {&nr.fEncC2Up, "k_encC2Up"},
    };
    for (auto& e : fns)
        if (cuModuleGetFunction(e.fn, nr.cuMod, e.nm) != CUDA_SUCCESS)
        {
            LOG("native: kernel %s missing\n", e.nm);
            return false;
        }
    return true;
}

// one resize axis's tap table (in -> out: lzTaps x out entries of 8 bytes), built by k_rsTab on the reader's stream at
// its first use (so it is ready before that read) and kept for the session; nullptr = the allocation or the launch failed
static const void* rsTabFor(NativeRife& nr, int in, int out, cudaStream_t st)
{
    for (const NativeRife::RsTab& e : nr.rsTabs)
        if (e.in == in && e.out == out)
            return e.d;
    void* d = nullptr;
    if (cudaMalloc(&d, (size_t)lzTaps(in, out) * out * 8) != cudaSuccess)
        return nullptr;
    void* a[] = {&in, &out, &d};
    if (cuLaunchKernel(nr.fRsTab, (out + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, a, nullptr) != CUDA_SUCCESS)
    {
        cudaFree(d);
        return nullptr;
    }
    nr.rsTabs.push_back({in, out, d});
    return d;
}

// one resize axis's tap table on the CPU (the session's final DLSS 4.5 enlarge, Host::lzTabFn): k_rsTab on a stream
// and into a buffer of its own (the compute thread's tables and streams stay untouched) once the native pipeline is
// up = 1; else k_rsTab's arithmetic ported (sin in double rounded once where the kernel has sinpif) = 2
static int nativeRsTabHost(NativeRife* nrp, int in, int out, std::vector<uint8_t>& tab)
{
    if (in <= 0 || out <= 0)
        return 0;
    const size_t bytes = (size_t)lzTaps(in, out) * out * 8;
    bool up = false;
    if (nrp)
    {
        std::lock_guard<std::mutex> lk(nrp->mInit);
        up = nrp->initState == 1 && nrp->fRsTab;
    }
    if (up && cudaSetDevice(nrp->dev) == cudaSuccess)
    {
        cudaStream_t st = nullptr;
        void* d = nullptr;
        bool ok = cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking) == cudaSuccess &&
                  cudaMalloc(&d, bytes) == cudaSuccess;
        if (ok)
        {
            void* a[] = {&in, &out, &d};
            ok = cuLaunchKernel(nrp->fRsTab, (out + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, a, nullptr) ==
                 CUDA_SUCCESS;
        }
        if (ok)
        {
            tab.resize(bytes);
            ok = cudaMemcpyAsync(tab.data(), d, bytes, cudaMemcpyDeviceToHost, st) == cudaSuccess &&
                 cudaStreamSynchronize(st) == cudaSuccess;
        }
        if (d)
            cudaFree(d);
        if (st)
            cudaStreamDestroy(st);
        if (ok)
            return 1;
    }
    const int fs = lzTaps(in, out), a = in > out ? kLzDown : kLzUp;
    auto arg = [&](int j, int o) {
        const long long n = (2LL * j + 1) * out - (2LL * o + 1) * in, d = 2LL * (in >= out ? in : out);
        const long long q = n / d;
        return (float)q + (float)(n - q * d) / (float)d;
    };
    auto sinpi = [](float x) { return (float)sin(3.14159265358979323846 * (double)x); };
    auto wt = [&](int j, int o) {
        float x = fabsf(arg(j, o));
        if (x >= (float)a)
            return 0.0f;
        if (x == 0.0f)
            return 1.0f;
        const float px = 3.14159265358979f * x;
        return (float)a * sinpi(x) * sinpi(x / (float)a) / (px * px);
    };
    tab.assign(bytes, 0);
    for (int o = 0; o < out; o++)
    {
        const long long num = (2LL * o + 1) * in - (long long)(fs - 1) * out, den = 2LL * out;
        long long q = num / den;
        if (num - q * den < 0)
            q--;
        const int mn = (int)q;
        float s = 0.0f;
        for (int k = 0; k < fs; k++)
            s += wt(mn + k, o);
        const float inv = 1.0f / s;
        for (int k = 0; k < fs; k++)
        {
            int j = mn + k;
            if (j < 0)
                j = -j - 1;
            if (j >= in)
                j = 2 * in - j - 1;
            j = j < 0 ? 0 : (j >= in ? in - 1 : j);
            const float w = wt(mn + k, o) * inv;
            uint8_t* e = tab.data() + ((size_t)k * out + o) * 8;
            memcpy(e, &w, 4);
            memcpy(e + 4, &j, 4);
        }
    }
    return 2;
}

// SMV_NATIVE_ARCH=75 (a test lever): every module compiled as compute_75 PTX, which the driver JIT-compiles for the
// running card, so the RTX 20 code paths (no cp.async, the k8 mma halves) run on a newer GPU
static int nativeArchLever()
{
    const std::string v = lkEnv("SMV_NATIVE_ARCH");
    return v.empty() ? 0 : atoi(v.c_str());
}
static const int g_nativeArch = nativeArchLever();

// one NVRTC program's cubin: read from cacheDir\<tag>_<source hash>_sm<arch>.cubin, else compiled (the arch the only
// option) and saved there; `what` names it in the lines. Under SMV_NATIVE_ARCH the buffer holds PTX instead
// (<tag>_<hash>_ptx<arch>.ptx, NUL-terminated), which cuModuleLoadData JIT-compiles the same way it loads a cubin.
static bool nativeCubin(const char* src, const char* file, const wchar_t* tag, const char* what,
                        const std::wstring& cacheDir, std::vector<char>& cubin)
{
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess)
        return false;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess)
        return false;
    uint64_t hash = 1469598103934665603ull;
    for (const char* p = src; *p; p++)
    {
        hash ^= (unsigned char)*p;
        hash *= 1099511628211ull;
    }
    wchar_t nm[128];
    if (g_nativeArch)
        swprintf_s(nm, L"\\%s_%016llx_ptx%d.ptx", tag, (unsigned long long)hash, g_nativeArch);
    else
        swprintf_s(nm, L"\\%s_%016llx_sm%d%d.cubin", tag, (unsigned long long)hash, prop.major, prop.minor);
    const std::wstring cubinPath = cacheDir + nm;
    ensureDirW(cacheDir);
    cubin.clear();
    bool haveCubin = false;
    {
        FILE* f = nullptr;
        if (!_wfopen_s(&f, cubinPath.c_str(), L"rb") && f)
        {
            fseek(f, 0, SEEK_END);
            const long n = ftell(f);
            fseek(f, 0, SEEK_SET);
            if (n > 0)
            {
                cubin.resize((size_t)n);
                haveCubin = fread(cubin.data(), 1, (size_t)n, f) == (size_t)n;
            }
            fclose(f);
        }
    }
    if (!haveCubin)
    {
        NvrtcApi nv;
        nv.mod = GetModuleHandleW(L"nvrtc64_130_0.dll");
        if (!nv.mod)
        {
            LOG("native: nvrtc not loaded\n");
            return false;
        }
        nv.create = (PFN_nvrtcCreateProgram)GetProcAddress(nv.mod, "nvrtcCreateProgram");
        nv.compile = (PFN_nvrtcCompileProgram)GetProcAddress(nv.mod, "nvrtcCompileProgram");
        // the PTX getters share the CUBIN getters' signatures
        nv.cubinSize =
            (PFN_nvrtcGetCUBINSize)GetProcAddress(nv.mod, g_nativeArch ? "nvrtcGetPTXSize" : "nvrtcGetCUBINSize");
        nv.cubin = (PFN_nvrtcGetCUBIN)GetProcAddress(nv.mod, g_nativeArch ? "nvrtcGetPTX" : "nvrtcGetCUBIN");
        nv.logSize = (PFN_nvrtcGetProgramLogSize)GetProcAddress(nv.mod, "nvrtcGetProgramLogSize");
        nv.log = (PFN_nvrtcGetProgramLog)GetProcAddress(nv.mod, "nvrtcGetProgramLog");
        nv.destroy = (PFN_nvrtcDestroyProgram)GetProcAddress(nv.mod, "nvrtcDestroyProgram");
        if (!nv.ok())
        {
            LOG("native: nvrtc entry points missing\n");
            return false;
        }
        void* prog = nullptr;
        if (nv.create(&prog, src, file, 0, nullptr, nullptr) != 0)
        {
            LOG("native: nvrtcCreateProgram failed (%s)\n", what);
            return false;
        }
        char arch[64];
        if (g_nativeArch)
            sprintf_s(arch, "--gpu-architecture=compute_%d", g_nativeArch);
        else
            sprintf_s(arch, "--gpu-architecture=sm_%d%d", prop.major, prop.minor);
        const char* opts[] = {arch, "--use_fast_math=false", "-default-device"};
        const int crc = nv.compile(prog, 1, opts); // arch only; the other two are informational
        if (crc != 0)
        {
            size_t ls = 0;
            std::string lg;
            if (nv.logSize && !nv.logSize(prog, &ls) && ls > 1)
            {
                lg.resize(ls);
                nv.log(prog, lg.data());
            }
            LOG("native: nvrtc compile failed (%s, %d): %s\n", what, crc, lg.c_str());
            nv.destroy(&prog);
            return false;
        }
        size_t cs = 0;
        if (nv.cubinSize(prog, &cs) != 0 || !cs)
        {
            LOG("native: nvrtcGetCUBINSize failed (%s)\n", what);
            nv.destroy(&prog);
            return false;
        }
        cubin.resize(cs);
        if (nv.cubin(prog, cubin.data()) != 0)
        {
            LOG("native: nvrtcGetCUBIN failed (%s)\n", what);
            nv.destroy(&prog);
            return false;
        }
        nv.destroy(&prog);
        FILE* f = nullptr;
        const std::wstring tmp = cubinPath + L".tmp";
        if (!_wfopen_s(&f, tmp.c_str(), L"wb") && f)
        {
            fwrite(cubin.data(), 1, cubin.size(), f);
            fclose(f);
            _wunlink(cubinPath.c_str());
            if (_wrename(tmp.c_str(), cubinPath.c_str()) != 0)
            {
                LOG("native: kernel cache not saved (%s, rename failed)\n", what);
                _wunlink(tmp.c_str());
            }
        }
        if (g_nativeArch)
            LOG("native: %s compiled as compute_%d PTX for the driver's JIT on sm_%d%d (SMV_NATIVE_ARCH, %zu bytes)\n",
                what, g_nativeArch, prop.major, prop.minor, cubin.size());
        else
            LOG("native: %s compiled for sm_%d%d (%zu bytes cubin)\n", what, prop.major, prop.minor, cubin.size());
    }
    return true;
}

static bool nativeBuildKernels(NativeRife& nr, const std::wstring& cacheDir)
{
    nr.kernCacheDir = cacheDir;
    // resident host: the module is per process (same device, same cubin), only rebind
    if (g_resident && g_res.cuMod)
    {
        nr.cuMod = g_res.cuMod;
        return nativeBindKernels(nr);
    }
    std::string text = kNativeKernels;
    encKernelText(text);
    std::vector<char> cubin;
    if (!nativeCubin(text.c_str(), "smv_native.cu", L"smv_native", "kernels", cacheDir, cubin))
        return false;
    if (cuInit(0) != CUDA_SUCCESS)
    {
        LOG("native: cuInit failed\n");
        return false;
    }
    if (cuModuleLoadData(&nr.cuMod, cubin.data()) != CUDA_SUCCESS)
    {
        LOG("native: cuModuleLoadData failed\n");
        return false;
    }
    if (!nativeBindKernels(nr))
        return false;
    if (g_resident)
        g_res.cuMod = nr.cuMod;
    return true;
}
// ---- part 3: CUDA init, the compute thread, shutdown --------------------------------------

// the two ICtCp matrices are inverted in double and uploaded as fp32
static bool inv3d(const double* m, float* out)
{
    const double d =
        m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if (d == 0.0)
        return false;
    const double id = 1.0 / d;
    out[0] = (float)((m[4] * m[8] - m[5] * m[7]) * id);
    out[1] = (float)((m[2] * m[7] - m[1] * m[8]) * id);
    out[2] = (float)((m[1] * m[5] - m[2] * m[4]) * id);
    out[3] = (float)((m[5] * m[6] - m[3] * m[8]) * id);
    out[4] = (float)((m[0] * m[8] - m[2] * m[6]) * id);
    out[5] = (float)((m[2] * m[3] - m[0] * m[5]) * id);
    out[6] = (float)((m[3] * m[7] - m[4] * m[6]) * id);
    out[7] = (float)((m[1] * m[6] - m[0] * m[7]) * id);
    out[8] = (float)((m[0] * m[4] - m[1] * m[3]) * id);
    return true;
}

// The native init is split in three so that everything except
// the output ring can run BEFORE the source-rate measurement (the ring is sized from the
// measured slot count):
//  * nativeCudaDeviceInit: device, context, stream, capture texture + fence import. Depends on
//    the capture size only, so the early thread runs it in parallel with the engine handoff
//    (it writes only dev / stream / emCap / capMip / capArr / semCap, the handoff writes the
//    geometry fields; distinct members, no race).
//  * nativeCudaInitEarly: kernels and the model buffers (needs the handoff geometry).
//  * nativeRtxInit: the live TrueHDR / RTX VSR bridge, the sharpen staging and the warm-ups
//    (needs the kernel module and the handoff geometry), on a helper thread beside the engine
//    load in the early thread.
//  * nativeCudaInitLate: the output ring and the slot events (needs the slot count), always
//    on the compute thread.
// The CUDA runtime binds the device's PRIMARY context on every thread that calls
// cudaSetDevice, so streams, allocations and the kernel modules created on
// the early thread are valid on the compute thread once it binds the same device.
static bool nativeCudaDeviceInit(NativeRife& nr, IDXGIAdapter1* adapter, HANDLE hTex, HANDLE hFence, uint32_t capW,
                                 uint32_t capH, bool hdr)
{
    int dev = 0;
    if (cudaD3D11GetDevice(&dev, adapter) != cudaSuccess)
    {
        LOG("native: cudaD3D11GetDevice failed, using device 0\n");
        dev = 0;
    }
    nr.dev = dev;
    NCHK(cudaSetDevice(dev), "cudaSetDevice");
    NCHK(cudaFree(nullptr), "cuda context init");
    NCHK(cudaStreamCreateWithFlags(&nr.stream, cudaStreamNonBlocking), "cudaStreamCreate");

    // capture texture: D3D11 resource, DEDICATED (mandatory for this handle type), read as a
    // mipmapped array.
    cudaExternalMemoryHandleDesc md{};
    md.type = cudaExternalMemoryHandleTypeD3D11Resource;
    md.handle.win32.handle = hTex;
    // Capture format: BGRA8 in SDR, R16G16B16A16_FLOAT scRGB in HDR (8 bytes per pixel)
    const int capBpp = hdr ? 8 : 4;
    md.size = (size_t)capW * capH * capBpp;
    md.flags = cudaExternalMemoryDedicated;
    NCHK(cudaImportExternalMemory(&nr.emCap, &md), "import capture texture");
    cudaExternalMemoryMipmappedArrayDesc ad{};
    ad.offset = 0;
    // the capture's format: uchar4 in SDR, half4 with channel kind FLOAT in
    // HDR (PG p.143's sample table has no float16 four-channel case; the working half4 import is
    // the truth, do not "fix" it to match the table)
    ad.formatDesc = hdr ? cudaChannelFormatDesc{16, 16, 16, 16, cudaChannelFormatKindFloat}
                        : cudaChannelFormatDesc{8, 8, 8, 8, cudaChannelFormatKindUnsigned};
    ad.extent = cudaExtent{(size_t)capW, (size_t)capH, 0};
    ad.flags = 0;
    ad.numLevels = 1;
    if (cudaExternalMemoryGetMappedMipmappedArray(&nr.capMip, nr.emCap, &ad) != cudaSuccess)
    {
        ad.flags = cudaArrayColorAttachment;
        NCHK(cudaExternalMemoryGetMappedMipmappedArray(&nr.capMip, nr.emCap, &ad), "map capture mipmapped array");
    }
    NCHK(cudaGetMipmappedArrayLevel(&nr.capArr, nr.capMip, 0), "capture array level");

    cudaExternalSemaphoreHandleDesc sd{};
    sd.type = cudaExternalSemaphoreHandleTypeD3D11Fence;
    sd.handle.win32.handle = hFence;
    NCHK(cudaImportExternalSemaphore(&nr.semCap, &sd), "import capture fence");
    return true;
}

// bind the device's primary context on the CALLING thread: cudaSetDevice alone only records
// the device, the context becomes current at the first runtime call that needs one, and the
// driver API (cuModuleLoadData, cuLaunchKernel) never triggers that, so force it here
static bool nativeBindDevice(const NativeRife& nr)
{
    NCHK(cudaSetDevice(nr.dev), "cudaSetDevice (bind)");
    NCHK(cudaFree(nullptr), "cuda context bind");
    return true;
}

static bool nativeCudaInitLate(NativeRife& nr, HANDLE hOutBuf, uint64_t outBytes)
{
    // output ring: the exe's own shared D3D12 committed buffer, imported as a linear buffer
    cudaExternalMemoryHandleDesc od{};
    od.type = cudaExternalMemoryHandleTypeD3D12Resource;
    od.handle.win32.handle = hOutBuf;
    od.size = outBytes;
    od.flags = cudaExternalMemoryDedicated;
    NCHK(cudaImportExternalMemory(&nr.emOut, &od), "import output buffer");
    cudaExternalMemoryBufferDesc bd{};
    bd.offset = 0;
    bd.size = outBytes;
    bd.flags = 0;
    NCHK(cudaExternalMemoryGetMappedBuffer((void**)&nr.dOutRing, nr.emOut, &bd), "map output buffer");
    // letterbox bars and alpha, written once: opaque black is 0xFF000000 in BGRA8 and
    // 0xC0000000 in R10G10B10A2 with A = 3
    if (cuMemsetD32Async((CUdeviceptr)nr.dOutRing, nr.hdr ? 0xC0000000u : 0xFF000000u, (size_t)(outBytes / 4),
                         (CUstream)nr.stream) != CUDA_SUCCESS)
    {
        LOG("native: output ring clear failed\n");
        return false;
    }
    nr.slotEv.resize(nr.slots);
    for (auto& e : nr.slotEv)
        NCHK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "create slot event");
    return true;
}

// Live TrueHDR setup: the bridge buffers, the ICtCp matrices, the NGX feature and its
// warm-up eval (~0.6 s). Needs the capture size, the stream and the kernel module only, so the
// early thread runs it on a helper beside the engine load; the non-early
// compute thread runs it after the engine load. The caller's thread must have bound the device.
// the frame Restore and the first resize read: the model frame, or with a working size the decoded one
static int srcW(const NativeRife& nr)
{
    return nr.sw ? nr.sw : nr.w;
}
static int srcH(const NativeRife& nr)
{
    return nr.sh ? nr.sh : nr.h;
}

// AMD FSR upscaling at RTX VSR's resize (NativeRife's fsrUp notes; defined after nativeFsrfgRelease)
static bool nativeFsrupSetup(NativeRife& nr, int w, int h, int ow, int oh);
static bool nativeFsrupEval(NativeRife& nr, const RtxRect& ri, const RtxRect& ro, cudaStream_t st, unsigned int& rv);
static void nativeFsrupRelease(NativeRife& nr);
static uint64_t nativeVideoMemoryRoom();
static bool nativeA4kSetup(NativeRife& nr, int w, int h, int ow, int oh, int upEnc);
static bool nativeA4kRun(NativeRife& nr, const void* s, int sHalf, int ps_, int rs_, float* dst, int tw, int th);
static void nativeA4kRelease(NativeRife& nr);
static bool nativeAcnSetup(NativeRife& nr, int w, int h, int ow, int oh, int upEnc);
static bool nativeAcnRun(NativeRife& nr, const void* s, int sHalf, int ps_, int rs_, float* dst, int tw, int th);
static void nativeAcnRelease(NativeRife& nr);

static bool nativeRtxInit(NativeRife& nr)
{
    // live Sharpen and RTX VSR: the fit is known now, so the rule "VSR
    // only when the resize enlarges in both axes" decides here
    // Upscale to: the first resize target is the internal render size when set, else
    // the fit rect; VSR and the aa rule read that target
    const int tw = nr.uw ? nr.uw : nr.dw, th = nr.uw ? nr.uh : nr.dh;
    const int sw = srcW(nr), sh = srcH(nr);
    // the resize RTX VSR runs (vw x vh -> vtw x vth): the first one; offline with a working size there
    // are two, the pre-model one (the decoded frame to the working size) and the final one (the
    // working size to the output): VSR is ONE bridge instance, so it takes the final one when that
    // enlarges, else the pre-model one, and the other is Lanczos. Live has the same two: the capture
    // to the working size before the model, and the fit after it
    int vw = g_offline ? sw : nr.w, vh = g_offline ? sh : nr.h, vtw = tw, vth = th;
    nr.vsrPre = false;
    if (!g_offline && nr.vsrWant && !(tw > nr.w && th > nr.h) && nr.w > nr.cw && nr.h > nr.ch)
    {
        nr.vsrPre = true;
        vw = nr.cw;
        vh = nr.ch;
        vtw = nr.w;
        vth = nr.h;
    }
    if (g_offline && nr.nvPre)
    {
        // with RTX HDR the frames after the model are PQ and RTX VSR takes SDR only (NVIDIA: TrueHDR after
        // VSR), so VSR can only take the pre-model resize
        nr.vsrPost = nr.dw > nr.w && nr.dh > nr.h && !nr.rtxHdr;
        if (nr.vsrWant && nr.rtxHdr && nr.dw > nr.w && nr.dh > nr.h)
            LOG("native: %s takes SDR only and RTX HDR runs before the model: the resize after it is Lanczos\n",
                nr.fsrUp   ? (nr.fuKind ? "Intel XeSS upscaling" : "AMD FSR upscaling")
                : nr.a4kUp ? "Anime4K"
                : nr.acnUp ? "ArtCNN"
                           : "RTX VSR");
        if (nr.vsrPost)
        {
            vw = nr.w;
            vh = nr.h;
        }
        else
        {
            vtw = nr.w;
            vth = nr.h;
        }
    }
    if (nr.vsrWant)
    {
        const char* up = nr.fsrUp   ? (nr.fuKind ? "Intel XeSS upscaling" : "AMD FSR upscaling")
                         : nr.a4kUp ? "Anime4K"
                         : nr.acnUp ? "ArtCNN"
                                    : "RTX VSR";
        if (g_offline && !nr.nvPre && nr.rtxHdr)
            LOG("native: %s skipped (it takes SDR only, and the one resize follows RTX HDR and the model), Lanczos\n",
                up);
        else if (vtw > vw && vth > vh)
            nr.vsr = true;
        else
            LOG("native: live %s skipped (upscales only; this resize does not enlarge), Lanczos\n", up);
    }
    nr.fsrUp = nr.fsrUp && nr.vsr;
    nr.a4kUp = nr.a4kUp && nr.vsr;
    nr.acnUp = nr.acnUp && nr.vsr;
    nr.vsrPre = nr.vsrPre && nr.vsr;
    // offline HDR video (--src-hdr): the upscaler's frames are the source's PQ / HLG codes at either stage (RTX HDR,
    // which would make them PQ after the model, keeps the upscaler off that resize), so RTX VSR reads their SDR view
    // and AMD FSR upscaling runs in its HDR mode, as live does on HDR planes
    const int upEnc = g_offline ? nr.encPre : 0;
    if (nr.uw)
    {
        NCHK(cudaMalloc((void**)&nr.dUp, (size_t)3 * nr.uw * nr.uh * sizeof(float)), "alloc internal render frame");
        if (nr.upAa)
            NCHK(cudaMalloc((void**)&nr.dUpTmp, (size_t)3 * nr.uw * nr.h * sizeof(float)),
                 "alloc internal render pass");
        LOG("native: live upscale to: model %dx%d -> %dx%d first, then %s to %dx%d\n", nr.w, nr.h, nr.uw, nr.uh,
            nr.identity ? "1:1" : "fit", nr.dw, nr.dh);
    }
    // the fit as the separable pair (nr.fitAa decided in the handoff parse): the horizontal
    // pass buffer and the staging frame; its source is the internal render frame under Upscale to
    if (nr.fitAa)
    {
        const int fh = nr.uw ? nr.uh : nr.h;
        NCHK(cudaMalloc((void**)&nr.dFitTmp, (size_t)3 * nr.dw * fh * sizeof(float)), "alloc fit pass");
        LOG("native: fit: %dx%d -> %dx%d, Lanczos (%s)\n", nr.uw ? nr.uw : nr.w, fh, nr.dw, nr.dh,
            nr.dh < fh ? "downscale" : "upscale");
    }
    // a resize stage's staging frame (nativeOfflineStage): Restore after the model, or RTX VSR before it (the
    // working-size frame, from the capture's planes) or after it on HDR planes (VSR's way back to the codes), and live
    // Anime4K / ArtCNN after the model without Upscale to (their nets write planes, the slot's store reads them back);
    // live's other stores never stage (RTX VSR straight into the slot, the Lanczos fit fused into the store, Upscale
    // to through dUp), offline's emit allocates its own
    if ((nr.restore && !nr.restPre) || nr.vsrPre || (nr.vsr && nr.hdr) ||
        (!g_offline && nr.vsr && (nr.a4kUp || nr.acnUp) && !nr.uw))
        NCHK(cudaMalloc((void**)&nr.dPres,
                        (size_t)3 * (std::max)(nr.dw * nr.dh, nr.vsrPre ? nr.w * nr.h : 0) * sizeof(float)),
             "alloc fit staging");
    if (nr.vsrPre && !nr.dCapF)
        NCHK(cudaMalloc((void**)&nr.dCapF, (size_t)3 * nr.ch * nr.cw * sizeof(float)), "alloc capture planes");
    if (nr.sharpen > 0.0f)
        NCHK(cudaMalloc((void**)&nr.dShOut, (size_t)3 * nr.w * nr.h * sizeof(float)), "alloc after-DLSS 5 output");
    if (nr.sharpen > 0.0f && nr.encPost && nr.shHdrView)
        NCHK(cudaMalloc((void**)&nr.dShIn, (size_t)3 * nr.w * nr.h * sizeof(float)), "alloc sharpen SDR view");
    if (nr.sharpen > 0.0f)
    {
        LOG("native: sharpen: Adaptive Sharpen %.2f at %dx%d, after DLSS 5 and before RTX HDR and the model\n",
            nr.sharpen, nr.w, nr.h);
        if (nr.encPost && !nr.rtxHdr)
            LOG("native: sharpen: HDR planes: %s\n",
                nr.shHdrView
                    ? "the sharpen edits their SDR view as an SDR source's, clamped at SDR white, and the light "
                      "outside the view is added back"
                    : "the sharpen on the PQ / HLG codes (SMV_HDR_SHARPEN_VIEW=0)");
    }
    // Restore's fold target. Offline with a
    // working size, and live, Restore runs before the model: the target is the working / model size,
    // or the source (the capture live) when VSR runs the pre-model resize right after it. Offline without a
    // working size restores the model output: back to its size when VSR follows (so VSR sees the
    // restored frame), else the first resize target directly (restore-as-upscaler, one resize)
    if (nr.restore)
    {
        const bool vsrAfter = nr.vsr && (g_offline ? !nr.vsrPost : nr.vsrPre);
        const bool toModel = (g_offline && nr.nvPre) || nr.restPre;
        nr.restTw = vsrAfter ? sw : (toModel ? nr.w : tw);
        nr.restTh = vsrAfter ? sh : (toModel ? nr.h : th);
        // HDR planes: the source fitted to the target, the remainder Restore's way back adds (nativeHdrRestOut)
        if ((nr.encPre || nr.encPost) && (nr.restTw != sw || nr.restTh != sh))
            NCHK(cudaMalloc((void**)&nr.dRestRem, (size_t)3 * nr.restTw * nr.restTh * sizeof(float)),
                 "alloc restore HDR fit");
        const std::string rm = !g_restoreModel.empty() ? g_restoreModel.substr(g_restoreModel.find_last_of("\\/") + 1)
                               : g_restoreA4k          ? std::string("Anime4K Restore CNN UL")
                                                       : std::string("Real-ESRGAN animevideov3");
        LOG("native: live restore: %s (the host's kernels) at %dx%d -> %dx%d%s\n", rm.c_str(), sw, sh, nr.restTw,
            nr.restTh, nr.restPre ? ", on the captured frame before the model" : "");
    }
    // a resize of HDR planes before the model: live the capture to the working size (with Restore, its reference),
    // offline the pre-model stage (the resize, or Restore at a size change)
    const int rsw = g_offline ? sw : nr.cw, rsh = g_offline ? sh : nr.ch;
    const bool rsHere = g_offline
                            ? nr.encPre && nr.nvPre &&
                                  (nr.w != sw || nr.h != sh || (nr.restore && (nr.restTw != sw || nr.restTh != sh)))
                            : nr.encPost && (nr.w != nr.cw || nr.h != nr.ch);
    if (rsHere && nr.rsHdrView)
    {
        NCHK(cudaMalloc((void**)&nr.dRsView, (size_t)3 * rsw * rsh * sizeof(float)), "alloc resize SDR view");
        NCHK(cudaMalloc((void**)&nr.dRsRange, sizeof(int)), "alloc resize range flag");
        NCHK(cudaHostAlloc((void**)&nr.hRsRange, sizeof(int), cudaHostAllocDefault), "alloc resize range readback");
        LOG("native: resize: HDR planes: a frame inside the SDR range is resized as its SDR view (the SDR route's "
            "Lanczos, clamped at SDR white), any other on the codes\n");
    }
    else if (rsHere)
        LOG("native: resize: HDR planes: resized on the codes (SMV_HDR_RESIZE_VIEW=0)\n");
    if (upEnc && nr.vsr && !nr.fsrUp && !nr.dRsRange)
    {
        // RTX VSR's way back to the codes tests each frame's range itself at a stage with no resize view (the final one)
        NCHK(cudaMalloc((void**)&nr.dRsRange, sizeof(int)), "alloc resize range flag");
        NCHK(cudaHostAlloc((void**)&nr.hRsRange, sizeof(int), cudaHostAllocDefault), "alloc resize range readback");
    }
    if (nr.vsr)
    {
        if (!nr.a4kUp && !nr.acnUp) // Anime4K / ArtCNN read and write planar floats, never the bridge's BGRA buffers
        {
            NCHK(cudaMalloc((void**)&nr.dVsrIn, (size_t)vw * vh * 4), "alloc VSR input");
            NCHK(cudaMalloc((void**)&nr.dVsrOut, (size_t)vtw * vth * 4), "alloc VSR output");
        }
        if (nr.hdr || upEnc)
        {
            NCHK(cudaMalloc((void**)&nr.dVsrView, (size_t)3 * vw * vh * sizeof(float)), "alloc VSR SDR view");
            NCHK(cudaMalloc((void**)&nr.dVsrRem, (size_t)3 * vtw * vth * sizeof(float)), "alloc VSR HDR fit");
        }
    }
    // RTX TrueHDR runs once per real frame at the working size, after Restore, DLSS 5 and Sharpen and before the
    // model (nativePreModelPost): its input and output (and the bridge's arrays, which the warm-up makes) are
    // working-size, dSrcG too offline; live dSrcG first carries the capture's SDR planes (k_sdrEncode), so it takes
    // the capture or the working size, whichever is bigger
    const bool workBig = (size_t)nr.w * nr.h > (size_t)nr.cw * nr.ch;
    const int gw = g_offline || workBig ? nr.w : nr.cw, gh = g_offline || workBig ? nr.h : nr.ch;
    if (nr.rtxHdr)
    {
        NCHK(cudaMalloc((void**)&nr.dThdrIn, (size_t)nr.w * nr.h * 4), "alloc TrueHDR input");
        NCHK(cudaMalloc((void**)&nr.dThdrOut, (size_t)nr.w * nr.h * 4), "alloc TrueHDR output");
        NCHK(cudaMalloc((void**)&nr.dSrcG, (size_t)3 * gw * gh * sizeof(float)), "alloc source gamma planes");
        if (g_offline)
        {
            NCHK(cudaMalloc((void**)&nr.dThdrStats, kThdrStatsBytes), "alloc TrueHDR stats");
            NCHK(cudaHostAlloc((void**)&nr.hThdrStats, kThdrStatsBytes, cudaHostAllocDefault),
                 "alloc TrueHDR stats copy");
            if (nr.thdrAcc)
            {
                // the _pq_lut of _accum_hp and its 100-nit code, from the kernels' own EOTF
                float* dLut = nullptr;
                NCHK(cudaMalloc((void**)&dLut, 1024 * sizeof(float)), "alloc PQ table");
                void* al[] = {&dLut};
                const bool lok = cuLaunchKernel(nr.fPqLut, 4, 1, 1, 256, 1, 1, 0, (CUstream)nr.stream, al, nullptr) ==
                                     CUDA_SUCCESS &&
                                 cudaMemcpyAsync(nr.thdrAcc->lut, dLut, sizeof(nr.thdrAcc->lut), cudaMemcpyDeviceToHost,
                                                 nr.stream) == cudaSuccess &&
                                 cudaStreamSynchronize(nr.stream) == cudaSuccess;
                cudaFree(dLut);
                if (!lok)
                {
                    LOG("native: the PQ table failed\n");
                    return false;
                }
                nr.thdrAcc->brightCode = 0;
                while (nr.thdrAcc->brightCode < 1024 && !(nr.thdrAcc->lut[nr.thdrAcc->brightCode] > 0.01f))
                    nr.thdrAcc->brightCode++;
            }
        }
        // the two numerically inverted ICtCp matrices, into the kernel module's globals
        const double rgb2lms[9] = {1688.0 / 4096.0, 2146.0 / 4096.0, 262.0 / 4096.0, 683.0 / 4096.0, 2951.0 / 4096.0,
                                   462.0 / 4096.0,  99.0 / 4096.0,   309.0 / 4096.0, 3688.0 / 4096.0};
        const double lms2ictcp[9] = {2048.0 / 4096.0,  2048.0 / 4096.0,   0.0,
                                     6610.0 / 4096.0,  -13613.0 / 4096.0, 7003.0 / 4096.0,
                                     17933.0 / 4096.0, -17390.0 / 4096.0, -543.0 / 4096.0};
        float lms2rgb[9], ictcp2lms[9];
        if (!inv3d(rgb2lms, lms2rgb) || !inv3d(lms2ictcp, ictcp2lms))
        {
            LOG("native: ICtCp matrix inversion failed\n");
            return false;
        }
        CUdeviceptr gp = 0;
        size_t gsz = 0;
        if (cuModuleGetGlobal(&gp, &gsz, nr.cuMod, "g_lms2rgb") != CUDA_SUCCESS ||
            cuMemcpyHtoD(gp, lms2rgb, sizeof(lms2rgb)) != CUDA_SUCCESS ||
            cuModuleGetGlobal(&gp, &gsz, nr.cuMod, "g_ictcp2lms") != CUDA_SUCCESS ||
            cuMemcpyHtoD(gp, ictcp2lms, sizeof(ictcp2lms)) != CUDA_SUCCESS)
        {
            LOG("native: uploading the ICtCp matrices failed\n");
            return false;
        }
    }
    const bool rtxVsrNow = nr.vsr && !nr.fsrUp && !nr.a4kUp && !nr.acnUp; // NVIDIA's RTX VSR holds the slot
    if (nr.rtxHdr || rtxVsrNow)
    {
        // the bridge retains the PRIMARY context (cuContext = NULL), which is the context this
        // thread just bound with cudaSetDevice. cuStream is NULL:
        // the bridge wraps every eval in host-synchronous cuMemcpy2D calls on the legacy
        // default stream, and the host's own stream is cudaStreamNonBlocking, so an NGX eval
        // placed on it would not be ordered against the bridge's own output copy. ONE create
        // per process (NGX is single-instance), so both features are asked for at once.
        if (g_rtxb.create(nullptr, nullptr, 0, nr.rtxHdr ? 1u : 0u, rtxVsrNow ? 1u : 0u) != 1u)
        {
            LOG("native: rtx_video_api_cuda_create failed\n");
            return false;
        }
        g_rtxb.created = true;
        g_rtxb.vsrMade = rtxVsrNow;
        g_rtxUsed = true; // resident host: NGX is single-instance per process, end it after this session
    }
    // AMD FSR upscaling on HDR planes runs in the bridge's HDR mode (linear light); with RTX HDR on, the frame before
    // the model is the SDR planes (NVIDIA's order: the upscale, then TrueHDR), the SDR mode
    nr.fuHdr = nr.fsrUp && (g_offline ? upEnc != 0 : nr.hdr && !(nr.vsrPre && nr.rtxHdr));
    if (nr.fsrUp && !nativeFsrupSetup(nr, vw, vh, vtw, vth))
        return false;
    if (nr.a4kUp && !nativeA4kSetup(nr, vw, vh, vtw, vth, upEnc))
        return false;
    if (nr.acnUp && !nativeAcnSetup(nr, vw, vh, vtw, vth, upEnc))
        return false;
    if (rtxVsrNow)
    {
        // warm-up eval on an opaque black frame, so the first presented frame pays nothing
        if (cuMemsetD32Async((CUdeviceptr)nr.dVsrIn, 0xFF000000u, (size_t)vw * vh, (CUstream)nr.stream) != CUDA_SUCCESS)
        {
            LOG("native: RTX VSR warm-up clear failed\n");
            return false;
        }
        NCHK(cudaStreamSynchronize(nr.stream), "VSR warm-up stream sync");
        const RtxRect ri{0, 0, (uint32_t)vw, (uint32_t)vh};
        const RtxRect ro{0, 0, (uint32_t)vtw, (uint32_t)vth};
        if (g_rtxb.evalVsr(nr.dVsrIn, nr.dVsrOut, ri, ro, &nr.vsrSet) != 1u)
        {
            LOG("native: the RTX VSR warm-up eval failed\n");
            return false;
        }
        NCHK(cudaDeviceSynchronize(), "VSR warm-up eval sync");
        LOG("native: live upscale: RTX VSR %dx%d -> %dx%d%s%s\n", vw, vh, vtw, vth,
            nr.vsrPre ? ", the captured frame before the model" : "",
            nr.hdr ? ", on the HDR planes' SDR view" : (upEnc ? ", on the HDR video's SDR view" : ""));
    }
    if (nr.rtxHdr)
    {
        // warm-up eval on a zero frame (opaque black RGBA)
        if (cuMemsetD32Async((CUdeviceptr)nr.dThdrIn, 0xFF000000u, (size_t)nr.w * nr.h, (CUstream)nr.stream) !=
            CUDA_SUCCESS)
        {
            LOG("native: TrueHDR warm-up clear failed\n");
            return false;
        }
        NCHK(cudaStreamSynchronize(nr.stream), "warm-up stream sync");
        const RtxRect wr{0, 0, (uint32_t)nr.w, (uint32_t)nr.h};
        if (g_rtxb.evalThdr(nr.dThdrIn, nr.dThdrOut, wr, wr, &nr.thdr) != 1u)
        {
            LOG("native: the TrueHDR warm-up eval failed\n");
            return false;
        }
        NCHK(cudaDeviceSynchronize(), "warm-up eval sync");
        if (g_offline)
            LOG("native: RTX TrueHDR on: %dx%d per decoded frame, after Restore, DLSS 5 and Sharpen and before the "
                "model, colour %ls, contrast %u saturation %u, %u nits\n",
                nr.w, nr.h, g_hdrColor, nr.thdr.Contrast, nr.thdr.Saturation, nr.thdr.MaxLuminance);
        else
            LOG("native: live RTX TrueHDR on: %dx%d per real frame, after Restore, DLSS 5 and Sharpen and before the "
                "model, colour %ls, SDR white %.0f nits, contrast %u saturation %u, %u nits\n",
                nr.w, nr.h, g_hdrColor, nr.sdrScale * 80.0f, nr.thdr.Contrast, nr.thdr.Saturation,
                nr.thdr.MaxLuminance);
    }
    return true;
}

// ---- the nvof model -------------------------------------------------------------------------
// The NVIDIA Optical Flow Accelerator through the driver's nvofapi64.dll (System32, opened by
// full path, never bundled; the process keeps it loaded once opened). One session per NativeRife
// at the true model size w x h: grid 4, preset fast, BOTH directions in one Execute, output cost
// on, temporal hints off (a paused or seeked source is not a continuous sequence). The defaults
// below are a sweep's pick (the metric weights barely move the result, gray input was best on
// the in-range clip). Every failure here is a REFUSAL.
static NV_OF_CUDA_API_FUNCTION_LIST g_nvofApi{};
static bool g_nvofLoaded = false;
static const float kNvofA = 0.1f, kNvofB = 1.0f;
// the pull-warp tween's fallback: log of the mean landing weight at or above kNvofHi keeps the warp,
// at or below kNvofLo takes the plain blend (linear between), the mask blurred by kNvofSigma px
// (picked by eye)
static const float kNvofLo = -9.0f, kNvofHi = -5.0f, kNvofSigma = 6.0f;
static const int kNvofGrid = 4;

// every Optical Flow session (the nvof model, DLSS 5's vectors, FSR's vectors) runs the engine's highest quality level,
// slow (the user's call: "we're not allowed to ship inferior quality"); SMV_OF_LEVEL=fast|medium|slow = A/B only
static NV_OF_PERF_LEVEL nativeOfLevel()
{
    char ev[16] = {};
    if (GetEnvironmentVariableA("SMV_OF_LEVEL", ev, sizeof(ev)) > 0)
    {
        if (!strcmp(ev, "fast"))
            return NV_OF_PERF_LEVEL_FAST;
        if (!strcmp(ev, "medium"))
            return NV_OF_PERF_LEVEL_MEDIUM;
    }
    return NV_OF_PERF_LEVEL_SLOW;
}

static const char* nativeOfLevelName(NV_OF_PERF_LEVEL l)
{
    return l == NV_OF_PERF_LEVEL_FAST ? "fast" : l == NV_OF_PERF_LEVEL_MEDIUM ? "medium" : "slow";
}

static bool nativeNvofLoad()
{
    if (g_nvofLoaded)
        return true;
    wchar_t sys[MAX_PATH]{};
    const UINT n = GetSystemDirectoryW(sys, MAX_PATH);
    if (!n || n >= MAX_PATH)
    {
        LOG("native: nvof: the system folder is unknown\n");
        return false;
    }
    const std::wstring p = std::wstring(sys) + L"\\nvofapi64.dll";
    HMODULE m = LoadLibraryExW(p.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!m)
    {
        LOG("native: nvof: nvofapi64.dll not found in the system folder (err %lu); the NVIDIA display driver installs it\n",
            GetLastError());
        return false;
    }
    typedef NV_OF_STATUS(NVOFAPI * PFN_CreateInstanceCuda)(uint32_t, NV_OF_CUDA_API_FUNCTION_LIST*);
    auto create = (PFN_CreateInstanceCuda)GetProcAddress(m, "NvOFAPICreateInstanceCuda");
    if (!create)
    {
        LOG("native: nvof: NvOFAPICreateInstanceCuda is not exported by nvofapi64.dll\n");
        return false;
    }
    const NV_OF_STATUS s = create(NV_OF_API_VERSION, &g_nvofApi);
    if (s != NV_OF_SUCCESS)
    {
        LOG("native: nvof: NvOFAPICreateInstanceCuda(0x%x) refused (status %d), driver too old for the Optical Flow SDK 5 interface\n",
            (unsigned)NV_OF_API_VERSION, (int)s);
        return false;
    }
    g_nvofLoaded = true;
    return true;
}

static bool nativeNvofBufOn(NvOFHandle ofh, uint32_t w, uint32_t h, NV_OF_BUFFER_USAGE usage, NV_OF_BUFFER_FORMAT fmt,
                            NvOFGPUBufferHandle& hb, CUdeviceptr& ptr, uint32_t& pitch, const char* what)
{
    NV_OF_BUFFER_DESCRIPTOR d{};
    d.width = w;
    d.height = h;
    d.bufferUsage = usage;
    d.bufferFormat = fmt;
    NV_OF_STATUS s = g_nvofApi.nvOFCreateGPUBufferCuda(ofh, &d, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &hb);
    if (s != NV_OF_SUCCESS)
    {
        LOG("native: nvof: %s buffer creation failed (status %d)\n", what, (int)s);
        return false;
    }
    ptr = g_nvofApi.nvOFGPUBufferGetCUdeviceptr(hb);
    NV_OF_CUDA_BUFFER_STRIDE_INFO si{};
    s = g_nvofApi.nvOFGPUBufferGetStrideInfo(hb, &si);
    if (s != NV_OF_SUCCESS || !ptr)
    {
        LOG("native: nvof: %s buffer stride query failed (status %d)\n", what, (int)s);
        return false;
    }
    pitch = si.strideInfo[0].strideXInBytes;
    return true;
}
static bool nativeNvofBuf(NativeRife& nr, uint32_t w, uint32_t h, NV_OF_BUFFER_USAGE usage, NV_OF_BUFFER_FORMAT fmt,
                          NvOFGPUBufferHandle& hb, CUdeviceptr& ptr, uint32_t& pitch, const char* what)
{
    return nativeNvofBufOn(nr.ofH, w, h, usage, fmt, hb, ptr, pitch, what);
}

// the session and the glue buffers; the calling thread has the device bound
static bool nativeNvofSetup(NativeRife& nr)
{
    if (!nativeNvofLoad())
        return false;
    if (nr.w < 32 || nr.h < 32)
    {
        LOG("native: nvof: the model frame %dx%d is below the Optical Flow minimum 32x32\n", nr.w, nr.h);
        return false;
    }
    CUcontext ctx = nullptr;
    if (cuCtxGetCurrent(&ctx) != CUDA_SUCCESS || !ctx)
    {
        LOG("native: nvof: no current CUDA context\n");
        return false;
    }
    NV_OF_STATUS s = g_nvofApi.nvCreateOpticalFlowCuda(ctx, &nr.ofH);
    if (s != NV_OF_SUCCESS)
    {
        nr.ofH = nullptr;
        LOG("native: nvof: nvCreateOpticalFlowCuda failed (status %d)\n", (int)s);
        return false;
    }
    NV_OF_INIT_PARAMS ip{};
    ip.width = (uint32_t)nr.w;
    ip.height = (uint32_t)nr.h;
    ip.outGridSize = (NV_OF_OUTPUT_VECTOR_GRID_SIZE)kNvofGrid;
    ip.hintGridSize = NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED;
    ip.mode = NV_OF_MODE_OPTICALFLOW;
    ip.perfLevel = nativeOfLevel();
    ip.enableExternalHints = NV_OF_FALSE;
    ip.enableOutputCost = NV_OF_TRUE;
    ip.disparityRange = NV_OF_STEREO_DISPARITY_RANGE_UNDEFINED;
    ip.enableRoi = NV_OF_FALSE;
    ip.predDirection = NV_OF_PRED_DIRECTION_BOTH;
    ip.enableGlobalFlow = NV_OF_FALSE;
    ip.inputBufferFormat = NV_OF_BUFFER_FORMAT_GRAYSCALE8;
    s = g_nvofApi.nvOFInit(nr.ofH, &ip);
    if (s != NV_OF_SUCCESS)
    {
        LOG("native: nvof: nvOFInit %dx%d grid %d refused (status %d)\n", nr.w, nr.h, kNvofGrid, (int)s);
        return false;
    }
    // the output grid is ceil(size / grid): verified at 1914x1078 -> 479x270
    nr.ofGw = (nr.w + kNvofGrid - 1) / kNvofGrid;
    nr.ofGh = (nr.h + kNvofGrid - 1) / kNvofGrid;
    for (int k = 0; k < 2; k++)
    {
        uint32_t inPitch = 0;
        if (!nativeNvofBuf(nr, nr.w, nr.h, NV_OF_BUFFER_USAGE_INPUT, NV_OF_BUFFER_FORMAT_GRAYSCALE8, nr.ofIn[k],
                           nr.ofInP[k], inPitch, "input") ||
            !nativeNvofBuf(nr, nr.ofGw, nr.ofGh, NV_OF_BUFFER_USAGE_OUTPUT, NV_OF_BUFFER_FORMAT_SHORT2, nr.ofOut[k],
                           nr.ofOutP[k], nr.ofOutPitch[k], "flow") ||
            !nativeNvofBuf(nr, nr.ofGw, nr.ofGh, NV_OF_BUFFER_USAGE_COST, NV_OF_BUFFER_FORMAT_UINT8, nr.ofCost[k],
                           nr.ofCostP[k], nr.ofCostPitch[k], "cost"))
            return false;
        if (k == 0)
            nr.ofInPitch = inPitch;
        else if (inPitch != nr.ofInPitch)
        {
            LOG("native: nvof: the two input buffers differ in pitch\n");
            return false;
        }
    }
    s = g_nvofApi.nvOFSetIOCudaStreams(nr.ofH, (CUstream)nr.stream, (CUstream)nr.stream);
    if (s != NV_OF_SUCCESS)
    {
        LOG("native: nvof: nvOFSetIOCudaStreams failed (status %d)\n", (int)s);
        return false;
    }
    const size_t mp = (size_t)nr.w * nr.h;
    for (int k = 0; k < 2; k++)
    {
        NCHK(cudaMalloc((void**)&nr.dNvFlow[k], 2 * mp * sizeof(float)), "alloc nvof flow");
        NCHK(cudaMalloc((void**)&nr.dNvCost[k], mp * sizeof(float)), "alloc nvof cost");
        NCHK(cudaMalloc((void**)&nr.dNvZ[k], mp * sizeof(float)), "alloc nvof metric");
    }
    NCHK(cudaMalloc((void**)&nr.dNvAcc, 5 * mp * sizeof(long long)), "alloc nvof accumulator");
    // the pull-warp tween's planes and the push-pull pyramid (levels halve, ceil, until the short
    // side is 1 px; level 0 is dNvN0 / dNvD0 in, dNvV out)
    NCHK(cudaMalloc((void**)&nr.dNvN0, 2 * mp * sizeof(float)), "alloc nvof velocity");
    NCHK(cudaMalloc((void**)&nr.dNvD0, mp * sizeof(float)), "alloc nvof confidence");
    NCHK(cudaMalloc((void**)&nr.dNvVis, 2 * mp * sizeof(float)), "alloc nvof visibility");
    NCHK(cudaMalloc((void**)&nr.dNvMraw, mp * sizeof(float)), "alloc nvof mask");
    NCHK(cudaMalloc((void**)&nr.dNvMask, mp * sizeof(float)), "alloc nvof blurred mask");
    NCHK(cudaMalloc((void**)&nr.dNvBlurTmp, mp * sizeof(float)), "alloc nvof blur pass");
    NCHK(cudaMalloc((void**)&nr.dNvV, 2 * mp * sizeof(float)), "alloc nvof filled velocity");
    for (int lw = nr.w, lh = nr.h; (lw < lh ? lw : lh) > 1;)
    {
        lw = (lw + 1) / 2;
        lh = (lh + 1) / 2;
        NativeRife::NvLevel L{lw, lh, nullptr, nullptr, nullptr};
        const size_t lp = (size_t)lw * lh;
        nr.nvPyr.push_back(L);
        NativeRife::NvLevel& B = nr.nvPyr.back();
        NCHK(cudaMalloc((void**)&B.n, 2 * lp * sizeof(float)), "alloc nvof pyramid");
        NCHK(cudaMalloc((void**)&B.d, lp * sizeof(float)), "alloc nvof pyramid");
        NCHK(cudaMalloc((void**)&B.out, 2 * lp * sizeof(float)), "alloc nvof pyramid");
    }
    // the tween in the model layout (3, ph, pw); the pad outside w x h is never written, zero it
    // once so no store path can ever read uninitialised memory there
    const size_t plane = (size_t)nr.ph * nr.pw;
    NCHK(cudaMalloc((void**)&nr.dNvOut, 3 * plane * sizeof(float)), "alloc nvof tween");
    NCHK(cudaMemsetAsync(nr.dNvOut, 0, 3 * plane * sizeof(float), nr.stream), "clear nvof tween");
    nr.nvProf = GetEnvironmentVariableW(L"SMV_LIVE_NVOF_PROF", nullptr, 0) != 0; // the GMFSS lever's rule
    if (nr.nvProf)
    {
        NCHK(cudaEventCreate(&nr.nvEv[0]), "create nvof event");
        NCHK(cudaEventCreate(&nr.nvEv[1]), "create nvof event");
    }
    LOG("native: nvof session %dx%d, grid %d (%dx%d), %s, both directions, gray8\n", nr.w, nr.h, kNvofGrid, nr.ofGw,
        nr.ofGh, nativeOfLevelName(ip.perfLevel));
    return true;
}

// the pair: both frames to gray8, one Execute for both fields, both upsampled, both metrics.
// dPrev / dCur are the model planes at (ph * pw, pw) strides.
static bool nativeNvofPair(NativeRife& nr, const float* dPrev, const float* dCur)
{
    cudaStream_t st = nr.stream;
    const size_t plane = (size_t)nr.ph * nr.pw;
    if (nr.nvProf)
        cudaEventRecord(nr.nvEv[0], st);
    int nps = (int)plane, pitch = (int)nr.ofInPitch;
    for (int k = 0; k < 2; k++)
    {
        const float* r = k ? dCur : dPrev;
        CUdeviceptr dst = nr.ofInP[k];
        void* a[] = {(void*)&r, &nps, &nr.pw, &nr.w, &nr.h, &dst, &pitch};
        if (cuLaunchKernel(nr.fNvofLuma, (nr.w + 15) / 16, (nr.h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a,
                           nullptr) != CUDA_SUCCESS)
        {
            nr.die("nvofLuma launch failed");
            return false;
        }
    }
    NV_OF_EXECUTE_INPUT_PARAMS ei{};
    ei.inputFrame = nr.ofIn[0]; // frame 0 -> "inputFrame": the forward field is F01
    ei.referenceFrame = nr.ofIn[1];
    ei.disableTemporalHints = NV_OF_TRUE;
    NV_OF_EXECUTE_OUTPUT_PARAMS eo{};
    eo.outputBuffer = nr.ofOut[0];
    eo.outputCostBuffer = nr.ofCost[0];
    eo.bwdOutputBuffer = nr.ofOut[1];
    eo.bwdOutputCostBuffer = nr.ofCost[1];
    const NV_OF_STATUS s = g_nvofApi.nvOFExecute(nr.ofH, &ei, &eo);
    if (s != NV_OF_SUCCESS)
    {
        LOG("native: nvof: nvOFExecute failed (status %d)\n", (int)s);
        nr.die("nvof execute failed");
        return false;
    }
    int grid = kNvofGrid;
    for (int k = 0; k < 2; k++)
    {
        CUdeviceptr v = nr.ofOutP[k], c = nr.ofCostP[k];
        int vp = (int)nr.ofOutPitch[k], cp = (int)nr.ofCostPitch[k];
        void* a[] = {&v, &vp, &c, &cp, &nr.ofGw, &nr.ofGh, &grid, &nr.w, &nr.h, &nr.dNvFlow[k], &nr.dNvCost[k]};
        if (cuLaunchKernel(nr.fNvofUp, (nr.w + 15) / 16, (nr.h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a, nullptr) !=
            CUDA_SUCCESS)
        {
            nr.die("nvofUp launch failed");
            return false;
        }
    }
    float ma = kNvofA, mb = kNvofB;
    for (int k = 0; k < 2; k++)
    {
        void* a[] = {&nr.dNvFlow[k], &nr.dNvCost[k], &nr.dNvFlow[k ^ 1], &nr.w, &nr.h, &ma, &mb, &nr.dNvZ[k]};
        if (cuLaunchKernel(nr.fNvofMetric, (nr.w + 15) / 16, (nr.h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a,
                           nullptr) != CUDA_SUCCESS)
        {
            nr.die("nvofMetric launch failed");
            return false;
        }
    }
    if (nr.nvProf)
    {
        cudaEventRecord(nr.nvEv[1], st);
        float ms = 0.0f;
        if (cudaEventSynchronize(nr.nvEv[1]) == cudaSuccess &&
            cudaEventElapsedTime(&ms, nr.nvEv[0], nr.nvEv[1]) == cudaSuccess)
        {
            nr.nvPairMs += ms;
            nr.nvPairN++;
        }
        if (nr.nvPairN && nr.nvPairN % 64 == 0)
            LOG("[nvof-prof] pair (luma + execute + upsample + metric) %.3f ms mean over %u pairs, tween %.3f ms mean\n",
                nr.nvPairMs / nr.nvPairN, nr.nvPairN, nr.nvTweenN ? nr.nvTweenMs / nr.nvTweenN : 0.0);
    }
    return true;
}

// one tween at t into dNvOut, the pull-warp chain: the
// velocity of both frames splatted to time t, its gaps push-pull filled, both frames SAMPLED
// along it, and where the vectors are untrustworthy (the blurred fallback mask) the plain blend
// instead: failed vectors show as ghosting, never as speckle
static bool nativeNvofTween(NativeRife& nr, float t)
{
    cudaStream_t st = nr.stream;
    const size_t plane = (size_t)nr.ph * nr.pw, mp = (size_t)nr.w * nr.h;
    const unsigned gx = (unsigned)(nr.w + 15) / 16, gy = (unsigned)(nr.h + 15) / 16;
    auto launch = [&](CUfunction f, unsigned bx, unsigned by, void** a, const char* what) -> bool {
        if (cuLaunchKernel(f, bx, by, 1, 16, 16, 1, 0, (CUstream)st, a, nullptr) == CUDA_SUCCESS)
            return true;
        nr.die(what);
        return false;
    };
    if (nr.nvProf)
        cudaEventRecord(nr.nvEv[0], st);
    if (cudaMemsetAsync(nr.dNvAcc, 0, 5 * mp * sizeof(long long), st) != cudaSuccess)
    {
        nr.die("nvof accumulator clear failed");
        return false;
    }
    // (1) the velocity (frame 0 -> 1 px) of both frames at time t, coverage per frame
    float s0 = t, s1 = 1.0f - t;
    float b0 = logf(1.0f - t), b1 = logf(t);
    float sg0 = 1.0f, sg1 = -1.0f;
    int cp0 = 3, cp1 = 4;
    for (int k = 0; k < 2; k++)
    {
        void* a[] = {&nr.dNvFlow[k],  &nr.dNvZ[k], k ? &s1 : &s0, k ? &b1 : &b0, k ? &sg1 : &sg0,
                     k ? &cp1 : &cp0, &nr.w,       &nr.h,         &nr.dNvAcc};
        if (!launch(nr.fSplatVel, gx, gy, a, "splatVel launch failed"))
            return false;
    }
    // (2) the merged velocity, confidence, visibilities, the raw fallback mask
    float lo = kNvofLo, hi = kNvofHi;
    {
        void* a[] = {&nr.dNvAcc, &nr.w, &nr.h, &t, &lo, &hi, &nr.dNvN0, &nr.dNvD0, &nr.dNvVis, &nr.dNvMraw};
        if (!launch(nr.fVelNorm, gx, gy, a, "velNorm launch failed"))
            return false;
    }
    // (3) push-pull: down to 1 px, the top normalised, back up to level 0 (dNvV)
    float* pn = nr.dNvN0;
    float* pd = nr.dNvD0;
    int pw_ = nr.w, ph_ = nr.h;
    for (auto& L : nr.nvPyr)
    {
        void* a[] = {&pn, &pd, &pw_, &ph_, &L.n, &L.d, &L.w, &L.h};
        if (!launch(nr.fPpDown, (unsigned)(L.w + 15) / 16, (unsigned)(L.h + 15) / 16, a, "ppDown launch failed"))
            return false;
        pn = L.n;
        pd = L.d;
        pw_ = L.w;
        ph_ = L.h;
    }
    if (nr.nvPyr.empty())
    {
        nr.die("nvof pyramid empty");
        return false;
    }
    {
        NativeRife::NvLevel& T = nr.nvPyr.back();
        void* a[] = {&T.n, &T.d, &T.w, &T.h, &T.out};
        if (!launch(nr.fPpTop, (unsigned)(T.w + 15) / 16, (unsigned)(T.h + 15) / 16, a, "ppTop launch failed"))
            return false;
    }
    for (int i = (int)nr.nvPyr.size() - 2; i >= -1; i--)
    {
        NativeRife::NvLevel& C = nr.nvPyr[i + 1]; // the coarser level, already filled
        float* n = i >= 0 ? nr.nvPyr[i].n : nr.dNvN0;
        float* d = i >= 0 ? nr.nvPyr[i].d : nr.dNvD0;
        float* o = i >= 0 ? nr.nvPyr[i].out : nr.dNvV;
        int lw = i >= 0 ? nr.nvPyr[i].w : nr.w, lh = i >= 0 ? nr.nvPyr[i].h : nr.h;
        void* a[] = {&n, &d, &lw, &lh, &C.out, &C.w, &C.h, &o};
        if (!launch(nr.fPpUp, (unsigned)(lw + 15) / 16, (unsigned)(lh + 15) / 16, a, "ppUp launch failed"))
            return false;
    }
    // (4) the mask blurred (x then y, scipy's gaussian_filter)
    float sig = kNvofSigma;
    int d0 = 0, d1 = 1;
    {
        void* a[] = {&nr.dNvMraw, &nr.w, &nr.h, &sig, &d0, &nr.dNvBlurTmp};
        if (!launch(nr.fBlur1, gx, gy, a, "blur1 (x) launch failed"))
            return false;
        void* b[] = {&nr.dNvBlurTmp, &nr.w, &nr.h, &sig, &d1, &nr.dNvMask};
        if (!launch(nr.fBlur1, gx, gy, b, "blur1 (y) launch failed"))
            return false;
    }
    // (5) sample both frames along the velocity, mix with the plain blend by the mask
    int ps = (int)plane;
    const float* i0 = nativeXPrev(nr);
    const float* i1 = nativeXCur(nr);
    {
        void* a[] = {(void*)&i0,  (void*)&i1, &ps,   &nr.pw, &nr.dNvV,  &nr.dNvVis,
                     &nr.dNvMask, &nr.w,      &nr.h, &t,     &nr.dNvOut};
        if (!launch(nr.fNvofCompose, gx, gy, a, "nvofCompose launch failed"))
            return false;
    }
    if (nr.nvProf)
    {
        cudaEventRecord(nr.nvEv[1], st);
        float ms = 0.0f;
        if (cudaEventSynchronize(nr.nvEv[1]) == cudaSuccess &&
            cudaEventElapsedTime(&ms, nr.nvEv[0], nr.nvEv[1]) == cudaSuccess)
        {
            nr.nvTweenMs += ms;
            nr.nvTweenN++;
        }
    }
    return true;
}

static void nativeNvofFree(NativeRife& nr)
{
    if (nr.nvProf && nr.nvPairN)
        LOG("[nvof-prof] session: pair %.3f ms mean over %u pairs, tween %.3f ms mean over %u tweens\n",
            nr.nvPairMs / nr.nvPairN, nr.nvPairN, nr.nvTweenN ? nr.nvTweenMs / nr.nvTweenN : 0.0, nr.nvTweenN);
    if (g_nvofLoaded)
    {
        for (int k = 0; k < 2; k++)
            for (NvOFGPUBufferHandle* b : {&nr.ofIn[k], &nr.ofOut[k], &nr.ofCost[k]})
                if (*b)
                {
                    g_nvofApi.nvOFDestroyGPUBufferCuda(*b);
                    *b = nullptr;
                }
        if (nr.ofH)
        {
            g_nvofApi.nvOFDestroy(nr.ofH);
            nr.ofH = nullptr;
        }
    }
    for (int k = 0; k < 2; k++)
    {
        for (float** p : {&nr.dNvFlow[k], &nr.dNvCost[k], &nr.dNvZ[k]})
            if (*p)
            {
                cudaFree(*p);
                *p = nullptr;
            }
        nr.ofInP[k] = nr.ofOutP[k] = nr.ofCostP[k] = 0;
    }
    if (nr.dNvAcc)
    {
        cudaFree(nr.dNvAcc);
        nr.dNvAcc = nullptr;
    }
    if (nr.dNvOut)
    {
        cudaFree(nr.dNvOut);
        nr.dNvOut = nullptr;
    }
    for (float** p : {&nr.dNvN0, &nr.dNvD0, &nr.dNvVis, &nr.dNvMraw, &nr.dNvMask, &nr.dNvBlurTmp, &nr.dNvV})
        if (*p)
        {
            cudaFree(*p);
            *p = nullptr;
        }
    for (auto& L : nr.nvPyr)
        for (float* p : {L.n, L.d, L.out})
            if (p)
                cudaFree(p);
    nr.nvPyr.clear();
    for (auto& e : nr.nvEv)
        if (e)
        {
            cudaEventDestroy(e);
            e = nullptr;
        }
}

// AMD FSR frame generation (fsrfg) runs on the FRUC route below with its own bridge: these four take over FRUC's setup,
// pair, tween and free when nr.fsrfg (defined after nativeFrucFree)
static bool nativeFsrfgSetup(NativeRife& nr);
static bool nativeFsrfgPair(NativeRife& nr, const float* dCur, uint32_t nTween);
static bool nativeFsrfgTween(NativeRife& nr, double t);
static void nativeFsrfgRelease(NativeRife& nr);

// NVIDIA Smooth Motion (fruc): the shipped nvoffruc_bridge.dll,
// loaded once per process by full path from the folder the handoff named (NvOFFRUC.dll and its
// cudart64_110.dll are user-installed beside it; the bridge loads NvOFFRUC.dll signature-checked
// itself). Its state is process-global (up to 16 FRUC instances), so a
// session creates instance 0 at setup, the others at the first pair that needs them, and
// destroys them all in nativeFree. The bridge runs each FRUC instance in its own CUDA context and brackets
// every call with cuCtxSynchronize on the caller's context (its SYNC FENCE notes), so the
// packs queued on nr.stream have landed when it copies, and its output has landed on return.
struct FrucBridge
{
    HMODULE mod = nullptr;
    const char* (*lastError)() = nullptr;
    int (*create)(unsigned, unsigned) = nullptr;
    int (*interpolate)(void*, void*, void*, double, int*) = nullptr;
    // the feed-once call: mode 0 = prime prev + feed cur, 1 = prev was fed
    // last, feed cur only, 2 = the same pair again; an older bridge lacks it (interpolate then)
    int (*step)(void*, void*, void*, double, int, int*) = nullptr;
    void (*destroy)() = nullptr;
    // the instance calls: up to 16 FRUC instances in the one module, each
    // with its own context; absent = an older bridge = one instance
    int (*createI)(int, unsigned, unsigned) = nullptr;
    int (*stepI)(int, void*, void*, void*, double, int, int*) = nullptr;
};
static FrucBridge g_fruc;

// one worker per extra instance (1..kFrDepth - 1): the compute thread hands it one step at a time and
// collects the result in tween order. The worker binds the host's device first, so the bridge's
// entry fence (cuCtxSynchronize on the caller's context) covers the host's packs and unpacks.
// Instance 0's worker serves the pipelined route instead: a job of levels > 0 = the next pair's levels 1..levels
// (nativeFrucSpecLevels), its calls, ms and counters kept here until the compute thread's join adds them
struct FrucWorker
{
    std::thread th;
    std::mutex m;
    std::condition_variable cv;
    bool job = false, done = true, quit = false;
    void* a = nullptr;
    void* b = nullptr;
    void* out = nullptr;
    double t = 0.0;
    int mode = 0, rc = 0, rep = 0;
    std::string err;
    int levels = 0;
    NativeRife* nr = nullptr;
    uint32_t calls = 0;
    uint64_t nodeCalls = 0, repeats = 0, primed = 0;
    double ms = 0.0;
};
// never destroyed: no std::thread destructor runs at process exit (nativeFrucFree joins them)
static FrucWorker* const g_frW = new FrucWorker[kFrDepth];
static int nativeFrucSpecLevels(NativeRife& nr, FrucWorker& w);

// the pipelined route's level gate: the deepest level of the shown pair the compute thread has made (kFrDepth once it
// has no more to make), so instance 0's worker starts a level of the next pair only after the shown pair's same level
// (one instance a level: no instance fed by two threads or out of time order when a ladder deepens)
static std::mutex g_frLagM;
static std::condition_variable g_frLagCv;
static int g_frLagDone = kFrDepth;
// each instance's calls in flight (nativeFrucNode): a second thread entering one logs `entered by two threads` once a
// session; SMV_FRUC_PIPE_GATE=0 = the worker never waits on the gate (that line's trigger, a test lever)
static std::atomic<int> g_frInUse[kFrDepth];
static std::atomic<bool> g_frInUseLogged{false};
static bool frucPipeGateOn()
{
    static const bool on = [] {
        char v[8] = "";
        return !(GetEnvironmentVariableA("SMV_FRUC_PIPE_GATE", v, sizeof(v)) > 0 && v[0] == '0');
    }();
    return on;
}
static void frucLagDone(int l)
{
    {
        std::lock_guard<std::mutex> lk(g_frLagM);
        g_frLagDone = (std::max)(g_frLagDone, l);
    }
    g_frLagCv.notify_all();
}

// the levels instance 0's worker may take of a pipelined pair: SMV_FRUC_PIPE_K caps them (1 = the level-1 call alone)
static int frucPipeK()
{
    static const int k = [] {
        char v[8] = "";
        return GetEnvironmentVariableA("SMV_FRUC_PIPE_K", v, sizeof(v)) > 0 ? (std::max)(1, atoi(v)) : kFrDepth;
    }();
    return k;
}

static void frucWorkerLoop(int i, int dev)
{
    FrucWorker& w = g_frW[i];
    const bool bound = cudaSetDevice(dev) == cudaSuccess && cudaFree(nullptr) == cudaSuccess;
    std::unique_lock<std::mutex> lk(w.m);
    for (;;)
    {
        w.cv.wait(lk, [&] { return w.job || w.quit; });
        if (w.quit)
            return;
        w.job = false;
        void* a = w.a;
        void* b = w.b;
        void* out = w.out;
        const double t = w.t;
        const int mode = w.mode;
        const int levels = w.levels;
        NativeRife* nr = w.nr;
        lk.unlock();
        int rep = 0, rc = -100;
        std::string err = "worker CUDA bind failed";
        if (bound && levels > 0)
        {
            rc = nativeFrucSpecLevels(*nr, w); // its calls, counters and error text go to w
            err = rc ? w.err : std::string();
        }
        else if (bound)
        {
            rc = g_fruc.stepI(i, a, b, out, t, mode, &rep);
            err = rc ? g_fruc.lastError() : "";
        }
        lk.lock();
        w.rc = rc;
        w.rep = rep;
        w.err = std::move(err);
        w.done = true;
        w.cv.notify_all();
    }
}

static bool nativeFrucLoad(const std::string& dir)
{
    if (g_fruc.mod)
        return true;
    const std::wstring wdir = utf8ToWide(dir);
    const std::wstring p = wdir + L"\\nvoffruc_bridge.dll";
    HMODULE m = LoadLibraryExW(p.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!m)
    {
        LOG("native: fruc: nvoffruc_bridge.dll did not load from %s (err %lu)\n", dir.c_str(), GetLastError());
        return false;
    }
    g_fruc.lastError = (const char* (*)())GetProcAddress(m, "nvoffruc_last_error");
    g_fruc.create = (int (*)(unsigned, unsigned))GetProcAddress(m, "nvoffruc_create");
    g_fruc.interpolate = (int (*)(void*, void*, void*, double, int*))GetProcAddress(m, "nvoffruc_interpolate");
    g_fruc.destroy = (void (*)())GetProcAddress(m, "nvoffruc_destroy");
    g_fruc.step = (int (*)(void*, void*, void*, double, int, int*))GetProcAddress(m, "nvoffruc_step");
    g_fruc.createI = (int (*)(int, unsigned, unsigned))GetProcAddress(m, "nvoffruc_create_i");
    g_fruc.stepI = (int (*)(int, void*, void*, void*, double, int, int*))GetProcAddress(m, "nvoffruc_step_i");
    if (!g_fruc.step)
        g_fruc.createI = nullptr, g_fruc.stepI = nullptr;
    if (!g_fruc.lastError || !g_fruc.create || !g_fruc.interpolate || !g_fruc.destroy)
    {
        LOG("native: fruc: nvoffruc_bridge.dll lacks an expected export\n");
        FreeLibrary(m);
        return false;
    }
    g_fruc.mod = m;
    return true;
}

// the FRUC instance at the padded model size (pw x ph) and the
// BGRA8 surfaces; the calling thread has the device bound
static bool nativeFrucSetup(NativeRife& nr)
{
    if (nr.fsrfg)
        return nativeFsrfgSetup(nr);
    if (!nativeFrucLoad(nr.frucDir))
        return false;
    const int rc = g_fruc.create((unsigned)nr.pw, (unsigned)nr.ph);
    // the bridge points the DLL search at its folder while it loads NvOFFRUC.dll (its
    // secure_load); give the process its default search back
    SetDllDirectoryW(nullptr);
    if (rc != 0)
    {
        LOG("native: fruc: nvoffruc_create %dx%d failed: %s (rc %d)\n", nr.pw, nr.ph, g_fruc.lastError(), rc);
        return false;
    }
    nr.frCreated = true;
    const size_t bytes = (size_t)nr.pw * nr.ph * 4, plane = (size_t)nr.ph * nr.pw;
    for (auto& s : nr.dFrSurf)
        NCHK(cudaMalloc((void**)&s, bytes), "alloc fruc surface");
    NCHK(cudaMalloc((void**)&nr.dFrOutB, bytes), "alloc fruc output");
    NCHK(cudaMalloc((void**)&nr.dFrOut, 3 * plane * sizeof(float)), "alloc fruc tween");
    {
        char ev[8] = {};
        nr.frSdrOn = nr.encPost && !(GetEnvironmentVariableA("SMV_FRUC_SDR_PAIRS", ev, sizeof(ev)) > 0 && ev[0] == '0');
    }
    if (nr.frSdrOn)
    {
        NCHK(cudaMalloc((void**)&nr.dFrRange, sizeof(int)), "alloc fruc range flag");
        NCHK(cudaHostAlloc((void**)&nr.hFrRange, sizeof(int), cudaHostAllocDefault), "alloc fruc range readback");
    }
    for (int i = 0; i < 3; i++)
        nr.frIn[i] = nr.frSurfSdr[i] = false;
    nr.frPairSdr = false;
    nr.frModeLogs = 0;
    nr.frPrev = nr.frLast = nr.frA = nr.frB = -1;
    for (int i = 0; i < kFrDepth; i++)
    {
        nr.frFed[i] = -1;
        nr.frFirst[i] = false;
        nr.frCalls[i] = 0;
    }
    nr.dFrOutBI[0] = nr.dFrOutB;
    nr.frInst = 1;
    nr.frPlanN = nr.frPlanK = 0;
    // recursive midpoints are the default wherever the bridge has the instance calls, one
    // instance per tree level, up to kFrDepth on either route; SMV_FRUC_MIDPOINTS=0 keeps the direct-t
    // scheme (parallel instances up to 4 live, 1 offline: offline FRUC is paced by the encoder,
    // where they measured 0.99x / ~0.94x at x3 / x5 1080p), SMV_FRUC_DEPTH=1..9 the snap depth.
    // SMV_FRUC_INSTANCES caps the instances on either scheme (1 = one instance, 5 = the tree to x32); the extra ones
    // are created by the first pair that needs them (nativeFrucGrow)
    nr.frMp = false;
    nr.frInstMax = 1;
    if (g_fruc.stepI)
    {
        char ev[8] = {};
        nr.frMp = !(GetEnvironmentVariableA("SMV_FRUC_MIDPOINTS", ev, sizeof(ev)) > 0 && ev[0] == '0');
        nr.frMpCap = g_offline ? 3 : 2;
        if (GetEnvironmentVariableA("SMV_FRUC_DEPTH", ev, sizeof(ev)) > 0)
            nr.frMpCap = (std::max)(1, (std::min)(kFrDepth, atoi(ev)));
        nr.frInstMax = nr.frMp ? kFrDepth : g_offline ? 1 : 4;
        if (GetEnvironmentVariableA("SMV_FRUC_INSTANCES", ev, sizeof(ev)) > 0)
            nr.frInstMax = (std::max)(1, (std::min)(kFrDepth, atoi(ev)));
    }
    nr.frMpL = 0;
    nr.frSerial = 0;
    nr.frNodeCalls = 0;
    for (bool& ok : nr.frNodeOk)
        ok = false;
    for (uint64_t& k : nr.frLastKey)
        k = 0;
    nr.frPipePrev = nr.frLagHave = nr.frLagSdr = nr.frLagStatic = nr.frCurStatic = false;
    nr.frSpecRun = nr.frRepacked = false;
    nr.frLagA = nr.frLagB = nr.frKeep = -1;
    nr.frLagL = nr.frPipeDepth = 0;
    nr.frLagSerial = nr.frPipeGroups = 0;
    for (bool& ok : nr.frLagOk)
        ok = false;
    nr.grFrCalls = nr.grFrPreCalls = nr.grFrWkCalls = 0;
    nr.grPriceSkip = 0;
    nr.grFrCallMs = nr.grFrPreMs = nr.grFrChainMs = nr.gFrCallMs = nr.gFrStoreMs = nr.gFrBaseMs = nr.gFrL1Ratio =
        nr.gFrL1BaseMs = nr.grFrWkMs = nr.gFrPipeMainMs = nr.gFrPipeWkMs = nr.gFrPipeRm = nr.gFrPipeRw =
            nr.gFrCallEntryMs = nr.gFgPreMs = nr.gFgOfMs = nr.gFgL1Ms = 0.0;
    nr.grPipeLearn = 0;
    nr.grPiped = nr.grLagArmed = false;
    for (int k = 0; k < 2; k++)
        nr.grL1Armed[k] = nr.grOfArmed[k] = false;
    nr.gFrCallUs.store(0);
    nr.gFrStoreUs.store(0);
    nr.gFrBaseUs.store(0);
    nr.gFrPipeMainUs.store(0);
    nr.gFrPipeWkUs.store(0);
    nr.gFgPreUs.store(0);
    nr.gFgOfUs.store(0);
    nr.gFgL1Us.store(0);
    g_frInUseLogged = false;
    if (nr.frMp)
        LOG("native: fruc session %dx%d (model %dx%d), NvOFFRUC through nvoffruc_bridge.dll, 8-bit BGRA, feed-once, recursive midpoints (depth cap %d)\n",
            nr.pw, nr.ph, nr.w, nr.h, nr.frMpCap);
    else
        LOG("native: fruc session %dx%d (model %dx%d), NvOFFRUC through nvoffruc_bridge.dll, 8-bit BGRA%s, up to %d instance%s\n",
            nr.pw, nr.ph, nr.w, nr.h, g_fruc.step ? ", feed-once" : "", nr.frInstMax, nr.frInstMax > 1 ? "s" : "");
    if (nr.encPost)
        LOG("native: fruc: HDR planes: %s\n",
            nr.frSdrOn
                ? "frame pairs inside the SDR range go to FRUC as their SDR view, any other pair as the 8-bit HDR codes"
                : "every pair as the 8-bit HDR codes (SMV_FRUC_SDR_PAIRS=0)");
    return true;
}

// the extra FRUC instances up to `want` (instance 0 is the session's own), each with its output
// buffer and its worker; a failed create (VRAM, most likely) keeps the instances made so far
static void nativeFrucGrow(NativeRife& nr, int want)
{
    want = (std::min)(want, (int)(sizeof(nr.dFrOutBI) / sizeof(nr.dFrOutBI[0])));
    const int64_t t0 = nowQpc100();
    const int had = nr.frInst;
    const size_t bytes = (size_t)nr.pw * nr.ph * 4;
    // SMV_FRUC_INST_FAILAT=i (harness lever): instance i fails to create, the trigger of the
    // fallback below
    char ev[8] = {};
    const int failAt = GetEnvironmentVariableA("SMV_FRUC_INST_FAILAT", ev, sizeof(ev)) > 0 ? atoi(ev) : -1;
    for (int i = (std::max)(nr.frInst, 0); i < want; i++)
    {
        const int rc = i == failAt ? -99 : g_fruc.createI(i, (unsigned)nr.pw, (unsigned)nr.ph);
        if (rc != 0 || cudaMalloc((void**)&nr.dFrOutBI[i], bytes) != cudaSuccess)
        {
            LOG("native: fruc: instance %d not created (%s, rc %d), staying at %d\n", i,
                rc == -99 ? "SMV_FRUC_INST_FAILAT"
                : rc      ? g_fruc.lastError()
                          : "output alloc failed",
                rc, nr.frInst);
            if (nr.dFrOutBI[i])
            {
                cudaFree(nr.dFrOutBI[i]);
                nr.dFrOutBI[i] = nullptr;
            }
            nr.frInstMax = nr.frInst;
            break;
        }
        FrucWorker& w = g_frW[i];
        w.job = w.quit = false;
        w.done = true;
        w.th = std::thread(frucWorkerLoop, i, nr.dev);
        nr.frInst = i + 1;
    }
    if (nr.frInst > had)
    {
        nr.grPriceSkip = 2;
        LOG("native: fruc: %d instances (%d new in %.0f ms)\n", nr.frInst, nr.frInst - had,
            (double)(nowQpc100() - t0) / 10000.0);
    }
}

// wait until every worker is idle (a round the caller abandoned, a session's end); instance 0's worker exists once a
// pipelined group ran (the level gate opened first: a failed group never reached its join)
static void nativeFrucDrain(NativeRife& nr)
{
    frucLagDone(kFrDepth);
    for (int i = 0; i < nr.frInst; i++)
    {
        FrucWorker& w = g_frW[i];
        if (i == 0 && !w.th.joinable())
            continue;
        std::unique_lock<std::mutex> lk(w.m);
        w.cv.wait(lk, [&] { return w.done; });
    }
}

// a pipelined group's job on instance 0's worker (made by the first such group): the next pair's levels 1..levels
static void nativeFrucSpecStart(NativeRife& nr, int levels)
{
    FrucWorker& w = g_frW[0];
    if (!w.th.joinable())
    {
        w.job = w.quit = false;
        w.done = true;
        w.th = std::thread(frucWorkerLoop, 0, nr.dev);
    }
    std::lock_guard<std::mutex> lk(w.m);
    w.levels = levels;
    w.nr = &nr;
    w.calls = 0;
    w.nodeCalls = w.repeats = w.primed = 0;
    w.ms = 0.0;
    w.err.clear();
    w.done = false;
    w.job = true;
    w.cv.notify_all();
    nr.frSpecRun = true;
}

// its result: the gate opened (a level the shown pair no longer needs), the job waited for, its counters added and its
// calls kept for the pipelined price (its nodes and its instances' keys were set by nativeFrucNode)
static bool nativeFrucSpecJoin(NativeRife& nr)
{
    if (!nr.frSpecRun)
        return true;
    nr.frSpecRun = false;
    frucLagDone(kFrDepth);
    FrucWorker& w = g_frW[0];
    std::unique_lock<std::mutex> lk(w.m);
    w.cv.wait(lk, [&] { return w.done; });
    nr.frNodeCalls += w.nodeCalls;
    nr.frRepeats += w.repeats;
    nr.frPrimed += w.primed;
    nr.grFrWkCalls = w.calls;
    nr.grFrWkMs = w.ms;
    if (w.rc != 0)
    {
        LOG("native: fruc: the next pair's levels 1..%d on instance 0's worker (pipelined) failed: %s (rc %d)\n",
            w.levels, w.err.c_str(), w.rc);
        nr.die("fruc midpoint failed");
        return false;
    }
    return true;
}

// the depth of a pair's midpoint tree: the smallest L that holds every t as a node k / 2^L (x2 / x4 /
// .. / x512 are exact on both routes); a pair with a t that is no node takes frMpCap (its nearest node
// there), offline at least its deepest node's depth. Live never deepens such a pair: a grid phase that
// lands on a deep node by chance would cost the whole deep tree for the pair's few snapped tweens. A node is
// within 1e-6 (a ladder's k / 2^L is exact in float), so a phase near a deep node by chance is no node
static int nativeFrucDepth(const NativeRife& nr, const double* ts, uint32_t n)
{
    int L = 1;
    bool off = false;
    for (uint32_t i = 0; i < n; i++)
    {
        int d = 1;
        for (; d <= kFrDepth; d++)
        {
            const double x = std::ldexp(ts[i], d);
            if (fabs(x - floor(x + 0.5)) < 1e-6)
                break;
        }
        if (d <= kFrDepth)
            L = (std::max)(L, d);
        else
            off = true;
    }
    if (off)
        L = g_offline ? (std::max)(L, nr.frMpCap) : nr.frMpCap;
    return (std::min)(L, kFrDepth);
}

// the pair's tweens in the order the caller asks for them (every nativeFrucTween call then takes
// the next one); fewer than two, or one instance, = the one-instance path
static void nativeFrucPlan(NativeRife& nr, const double* ts, uint32_t n)
{
    if (nr.frMp)
    {
        // recursive midpoints: only the pair's depth; the first nativeFrucTween builds the tree
        nr.frMpL = nativeFrucDepth(nr, ts, n);
        return;
    }
    nativeFrucDrain(nr);
    nr.frPlanN = nr.frPlanK = 0;
    if (nr.frInstMax < 2 || n < 2 || n > 64)
        return;
    const int want = (std::min)((int)n, nr.frInstMax);
    if (want > nr.frInst)
        nativeFrucGrow(nr, want);
    if (nr.frInst < 2)
        return;
    memcpy(nr.frPlan, ts, n * sizeof(double));
    nr.frPlanN = n;
}

// the pair: the new frame packed into a surface that holds neither the previous frame nor the
// last tweened pair's end. Only when this group has tweens: a pair that
// does not continue the last interpolated one first gets one dropped warp at 0.5 of the
// skipped pair (last end, previous frame), so the OFA hints stay consecutive
// one frame's planes into surface n: the SDR view (sdr, k_packBgraSdr) or the planes as stored (true BGRA)
static bool nativeFrucPack(NativeRife& nr, const float* src, int n, bool sdr)
{
    int ps = nr.ph * nr.pw, f32 = 0;
    void* a[] = {(void*)&src, &f32, &ps, &nr.pw, &nr.pw, &nr.ph, &nr.dFrSurf[n]};
    void* b[] = {(void*)&src, &ps, &nr.pw, &nr.pw, &nr.ph, &nr.encPost, &nr.encWhite, &nr.dFrSurf[n]};
    const CUfunction f = sdr ? nr.fPackBgraSdr : nr.fPackBgraRgb;
    if (cuLaunchKernel(f, (nr.pw + 15) / 16, (nr.ph + 15) / 16, 1, 16, 16, 1, 0, (CUstream)nr.stream, sdr ? b : a,
                       nullptr) != CUDA_SUCCESS)
    {
        nr.die("packBgra (fruc) launch failed");
        return false;
    }
    nr.frSurfSdr[n] = sdr;
    for (int& fed : nr.frFed)
        if (fed == n)
            fed = -1; // the frame FRUC was fed last is gone from its surface
    return true;
}

static bool nativeFrucPair(NativeRife& nr, const float* dCur, uint32_t nTween)
{
    if (nr.fsrfg)
        return nativeFsrfgPair(nr, dCur, nTween);
    nativeFrucDrain(nr); // no worker still reads a surface this pair may repack
    // a pipelined group spares the shown pair's first frame (frKeep) where the others spare the last fed one
    int n = 0;
    while (n == nr.frPrev || n == (nr.frKeep >= 0 ? nr.frKeep : nr.frLast))
        n++;
    nr.frRepacked = false;
    // HDR planes: the new frame's range (one flag read back); the pair takes the SDR view only when both frames stay
    // inside the SDR range. A previous frame's surface in the other encoding is packed again from its planes (dX)
    // and every instance primes it again (its node key would still match)
    bool in = false;
    if (nr.frSdrOn)
    {
        cudaStream_t st = nr.stream;
        int ps = nr.ph * nr.pw;
        float tol = 2e-3f;
        void* ar[] = {(void*)&dCur, &ps, &nr.pw, &nr.pw, &nr.ph, &nr.encPost, &nr.encWhite, &tol, (void*)&nr.dFrRange};
        if (cudaMemsetAsync(nr.dFrRange, 0, sizeof(int), st) != cudaSuccess ||
            cuLaunchKernel(nr.fHdrRange, (nr.pw + 15) / 16, (nr.ph + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, ar,
                           nullptr) != CUDA_SUCCESS ||
            cudaMemcpyAsync(nr.hFrRange, nr.dFrRange, sizeof(int), cudaMemcpyDeviceToHost, st) != cudaSuccess ||
            cudaStreamSynchronize(st) != cudaSuccess)
        {
            nr.die("fruc range test failed");
            return false;
        }
        in = *nr.hFrRange == 0;
    }
    const bool sdr = in && (nr.frPrev < 0 || nr.frIn[nr.frPrev]);
    if (!nativeFrucPack(nr, dCur, n, sdr))
        return false;
    nr.frIn[n] = in;
    if (nr.frPrev >= 0 && nr.frSurfSdr[nr.frPrev] != sdr)
    {
        if (!nativeFrucPack(nr, nativeXPrev(nr), nr.frPrev, sdr))
            return false;
        nr.frRepacked = true;
        for (uint64_t& k : nr.frLastKey)
            k = 0;
    }
    if (nr.frSdrOn && nTween && nr.frPrev >= 0 && nr.frModeLogs < 4 && (nr.frModeLogs == 0 || sdr != nr.frPairSdr))
    {
        LOG("native: fruc: %s\n", sdr ? "frame pairs inside the SDR range, their SDR view"
                                      : "a frame pair outside the SDR range, the 8-bit HDR codes");
        nr.frModeLogs++;
    }
    nr.frPairSdr = sdr;
    nr.frA = nr.frPrev;
    nr.frB = n;
    nr.frPrev = n;
    nr.frPlanN = nr.frPlanK = 0; // the caller plans this pair's tweens after this (nativeFrucPlan)
    // recursive midpoints: this frame's serial names it in the node keys; the pair's tree starts empty
    nr.frSerial++;
    nr.frMpL = 0;
    nr.frMpBuilt = false;
    for (bool& ok : nr.frNodeOk)
        ok = false;
    if (!nTween || nr.frA < 0)
        return true;
    if (g_fruc.step)
    {
        // feed-once bridge: no priming warp; an instance's first tween of the pair says whether
        // it was fed frA last (mode 1) or must prime it (mode 0), its other tweens reuse the pair (mode 2)
        for (bool& f : nr.frFirst)
            f = true;
        nr.frLast = nr.frB;
        return true;
    }
    if (nr.frLast >= 0 && nr.frLast != nr.frA)
    {
        int rep = 0;
        const int rc = g_fruc.interpolate(nr.dFrSurf[nr.frLast], nr.dFrSurf[nr.frA], nr.dFrOutB, 0.5, &rep);
        if (rc != 0)
        {
            LOG("native: fruc: priming warp failed: %s (rc %d)\n", g_fruc.lastError(), rc);
            nr.die("fruc priming failed");
            return false;
        }
        nr.frPrimed++;
    }
    nr.frLast = nr.frB;
    return true;
}

// one tween at t into dFrOut (3, ph, pw) (t is a
// double, the bridge's own type: offline --fps hands its double fraction over unrounded)
// the feed-once mode of instance i's next call on this pair; a mode 0 after the instance's first
// call is a pair that does not continue the one it was fed last (a repack of the fed surface
// clears frFed, so frFed alone would miss those primes)
static int nativeFrucMode(NativeRife& nr, int i)
{
    const int mode = !nr.frFirst[i] ? 2 : (nr.frFed[i] >= 0 && nr.frFed[i] == nr.frA) ? 1 : 0;
    if (mode == 0 && nr.frCalls[i] > 0)
        nr.frPrimed++;
    return mode;
}

static void nativeFrucFedOk(NativeRife& nr, int i)
{
    nr.frFirst[i] = false;
    nr.frFed[i] = nr.frB;
    nr.frCalls[i]++;
}

// a pair's midpoint tree: this group's pair, or the previous one a pipelined group shows (frLag*)
struct FrTree
{
    uint64_t serial; // its second frame's serial
    int a, b;        // its frames' surfaces
    uint8_t** node;  // [kFrGrid]
    bool* ok;        // [kFrGrid]
};
static FrTree nativeFrucCur(NativeRife& nr)
{
    return {nr.frSerial, nr.frA, nr.frB, nr.dFrNode, nr.frNodeOk};
}
static FrTree nativeFrucLag(NativeRife& nr)
{
    return {nr.frLagSerial, nr.frLagA, nr.frLagB, nr.dFrLagNode, nr.frLagOk};
}

// recursive midpoints: the key of node k / 2^L of a pair (k = 0 / 2^L = the pair's first /
// second frame): a real frame = its serial, an interior node = the pair's first serial and its
// position on the kFrGrid grid; an instance fed the left parent of a node last feeds only the right one
static uint64_t nativeFrucKey(const FrTree& tr, int k, int L)
{
    while (L > 0 && !(k & 1))
    {
        k >>= 1;
        L--;
    }
    const uint64_t a = tr.serial - 1;
    if (L == 0)
        return (k ? tr.serial : a) * kFrGrid + 1;
    return a * kFrGrid + ((uint64_t)k << (kFrDepth - L)) + 1;
}

// node k / 2^L of a pair's midpoint tree, once per pair: FRUC at t = 0.5 between the node's two
// parents (computed first) on instance L - 1, mode 1 when that instance was fed the left parent
// last, else mode 0 (prime the left parent, feed the right one). On instance 0's worker (w) the calls, counters and a
// failure's text go to w (the compute thread runs the other instances meanwhile; its join adds them)
static uint8_t* nativeFrucNode(NativeRife& nr, const FrTree& tr, int k, int L, FrucWorker* w = nullptr)
{
    while (L > 0 && !(k & 1))
    {
        k >>= 1;
        L--;
    }
    if (L <= 0)
        return nr.dFrSurf[k ? tr.b : tr.a];
    const int p = k << (kFrDepth - L);
    if (tr.ok[p])
        return tr.node[p];
    uint8_t* left = nativeFrucNode(nr, tr, k - 1, L, w);
    uint8_t* right = left ? nativeFrucNode(nr, tr, k + 1, L, w) : nullptr;
    if (!right)
        return nullptr;
    if (!tr.node[p] && cudaMalloc((void**)&tr.node[p], (size_t)nr.pw * nr.ph * 4) != cudaSuccess)
    {
        tr.node[p] = nullptr;
        if (w)
            w->err = "fruc midpoint buffer alloc failed";
        else
            nr.die("fruc midpoint buffer alloc failed");
        return nullptr;
    }
    const int i = L - 1;
    const int mode = nr.frLastKey[i] == nativeFrucKey(tr, k - 1, L) ? 1 : 0;
    if (mode == 0 && nr.frCalls[i] > 0)
        (w ? w->primed : nr.frPrimed)++;
    int rep = 0;
    if (g_frInUse[i].fetch_add(1) && !g_frInUseLogged.exchange(true))
        LOG("native: fruc: instance %d entered by two threads (pipelined level %d)\n", i, L);
    const int64_t c0 = nowQpc100();
    const int rc = g_fruc.stepI(i, left, right, tr.node[p], 0.5, mode, &rep);
    const double ms = (double)(nowQpc100() - c0) / 10000.0;
    g_frInUse[i].fetch_sub(1);
    if (w)
    {
        w->calls++;
        w->ms += ms;
    }
    else
    {
        nr.grFrCalls++;
        nr.grFrCallMs += ms;
    }
    if (rc != 0)
    {
        if (w)
        {
            char e[256];
            snprintf(e, sizeof(e), "midpoint %d/%d on instance %d: %s (bridge rc %d)", k, 1 << L, i, g_fruc.lastError(),
                     rc);
            w->err = e;
            return nullptr;
        }
        LOG("native: fruc: midpoint %d/%d on instance %d failed: %s (rc %d)\n", k, 1 << L, i, g_fruc.lastError(), rc);
        nr.die("fruc midpoint failed");
        return nullptr;
    }
    nr.frLastKey[i] = nativeFrucKey(tr, k + 1, L);
    nr.frCalls[i]++;
    (w ? w->nodeCalls : nr.frNodeCalls)++;
    if (rep)
        (w ? w->repeats : nr.frRepeats)++;
    tr.ok[p] = true;
    return tr.node[p];
}

// the levels of the shown pair already made (each level's every node), the level gate's start
static int nativeFrucLagMade(const NativeRife& nr)
{
    int l = 0;
    for (int L = 1; L <= kFrDepth; L++)
    {
        for (int k = 1; k < (1 << L); k += 2)
            if (!nr.frLagOk[k << (kFrDepth - L)])
                return l;
        l = L;
    }
    return l;
}

// instance 0's worker on a pipelined group: the next pair's levels 1..w.levels (instances 0..levels - 1), level by
// level as nativeFrucTween builds a tree, each level after the level gate passed it (the shown pair's same level made)
static int nativeFrucSpecLevels(NativeRife& nr, FrucWorker& w)
{
    const FrTree tr = nativeFrucCur(nr);
    for (int l = 1; l <= w.levels; l++)
    {
        if (frucPipeGateOn())
        {
            std::unique_lock<std::mutex> lk(g_frLagM);
            g_frLagCv.wait(lk, [&] { return g_frLagDone >= l; });
        }
        for (int k = 1; k < (1 << l); k += 2)
            if (!nativeFrucNode(nr, tr, k, l, &w))
                return -1;
    }
    return 0;
}

// a FRUC output (true BGRA8) into dFrOut (3, ph, pw) as (R, G, B) planes; an SDR pair's tween back to the HDR
// codes (k_unpackBgraSdr)
static bool nativeFrucUnpack(NativeRife& nr, uint8_t* src, bool sdr)
{
    void* a[] = {&src, &nr.pw, &nr.ph, &nr.dFrOut};
    void* b[] = {&src, &nr.pw, &nr.ph, &nr.encPost, &nr.encWhite, &nr.dFrOut};
    const CUfunction f = sdr ? nr.fUnpackBgraSdr : nr.fUnpackBgraRgb;
    if (cuLaunchKernel(f, (nr.pw + 15) / 16, (nr.ph + 15) / 16, 1, 16, 16, 1, 0, (CUstream)nr.stream, sdr ? b : a,
                       nullptr) != CUDA_SUCCESS)
    {
        nr.die("unpackBgra (fruc) launch failed");
        return false;
    }
    return true;
}

static bool nativeFrucTween(NativeRife& nr, double t)
{
    if (nr.fsrfg)
        return nativeFsrfgTween(nr, t);
    if (nr.frMp)
    {
        // recursive midpoints: the pair's WHOLE tree to its depth first, level by level, left to
        // right, so level L's instance sees the continuous stream A, 2 / 2^L, 4 / 2^L .. B of new
        // frames. A node computed alone needs a prime whose optical flow spans the skipped nodes,
        // NvOFFRUC seeds the next flow from it, and that damaged every such node (up to 28 % of a
        // held frame). Then the nearest node (never an end frame); a failed instance
        // create caps the depth at the instances made
        if (!nr.frMpL)
            nr.frMpL = nativeFrucDepth(nr, &t, 1);
        // a failed create lowers frInstMax (nativeFrucGrow), so it is not retried every pair
        if (nr.frInst < nr.frMpL && nr.frInst < nr.frInstMax)
            nativeFrucGrow(nr, (std::min)(nr.frMpL, nr.frInstMax));
        if (nr.frInst < nr.frMpL)
            nr.frMpL = nr.frInst;
        const FrTree tr = nativeFrucCur(nr);
        if (!nr.frMpBuilt)
        {
            for (int l = 1; l <= nr.frMpL; l++)
                for (int k = 1; k < (1 << l); k += 2)
                    if (!nativeFrucNode(nr, tr, k, l))
                        return false;
            nr.frMpBuilt = true;
        }
        const int den = 1 << nr.frMpL;
        const int k = (std::max)(1, (std::min)(den - 1, (int)floor(t * den + 0.5)));
        uint8_t* node = nativeFrucNode(nr, tr, k, nr.frMpL);
        if (!node)
            return false;
        nr.frTweens++;
        if (!nativeFrucUnpack(nr, node, nr.frPairSdr))
            return false;
        return true;
    }
    int rep = 0;
    int rc;
    uint8_t* outB = nr.dFrOutB;
    if (nr.frPlanK < nr.frPlanN && nr.frPlan[nr.frPlanK] != t)
    {
        // the caller asked for a tween its plan did not name: finish what runs, then one instance
        LOG("native: fruc: tween %.4f is not the planned %.4f, one instance for the rest of the pair\n", t,
            nr.frPlan[nr.frPlanK]);
        nativeFrucDrain(nr);
        nr.frPlanN = nr.frPlanK = 0;
    }
    if (nr.frPlanK < nr.frPlanN)
    {
        // parallel instances: tween k runs on instance k % frInst; the first tween of a round
        // hands the round's others to their workers and runs its own here, the others collect
        const uint32_t k = nr.frPlanK++;
        const int inst = (int)(k % (uint32_t)nr.frInst);
        if (inst == 0)
        {
            for (uint32_t j = k + 1; j < nr.frPlanN && j < k + (uint32_t)nr.frInst; j++)
            {
                const int wi = (int)(j - k);
                FrucWorker& w = g_frW[wi];
                std::lock_guard<std::mutex> lk(w.m);
                w.a = nr.dFrSurf[nr.frA];
                w.b = nr.dFrSurf[nr.frB];
                w.out = nr.dFrOutBI[wi];
                w.t = nr.frPlan[j];
                w.mode = nativeFrucMode(nr, wi);
                w.done = false;
                w.job = true;
                w.cv.notify_all();
            }
            rc = g_fruc.stepI(0, nr.dFrSurf[nr.frA], nr.dFrSurf[nr.frB], nr.dFrOutB, t, nativeFrucMode(nr, 0), &rep);
            if (rc == 0)
                nativeFrucFedOk(nr, 0);
            else
                LOG("native: fruc: interpolate failed: %s (rc %d)\n", g_fruc.lastError(), rc);
        }
        else
        {
            FrucWorker& w = g_frW[inst];
            std::unique_lock<std::mutex> lk(w.m);
            w.cv.wait(lk, [&] { return w.done; });
            rc = w.rc;
            rep = w.rep;
            if (rc == 0)
                nativeFrucFedOk(nr, inst);
            else
                LOG("native: fruc: interpolate (instance %d) failed: %s (rc %d)\n", inst, w.err.c_str(), rc);
            outB = nr.dFrOutBI[inst];
        }
        if (rc != 0)
        {
            nativeFrucDrain(nr);
            nr.die("fruc interpolate failed");
            return false;
        }
    }
    else if (g_fruc.step)
    {
        rc = g_fruc.step(nr.dFrSurf[nr.frA], nr.dFrSurf[nr.frB], nr.dFrOutB, t, nativeFrucMode(nr, 0), &rep);
        if (rc == 0)
            nativeFrucFedOk(nr, 0);
    }
    else
        rc = g_fruc.interpolate(nr.dFrSurf[nr.frA], nr.dFrSurf[nr.frB], nr.dFrOutB, t, &rep);
    if (rc != 0)
    {
        LOG("native: fruc: interpolate failed: %s (rc %d)\n", g_fruc.lastError(), rc);
        nr.die("fruc interpolate failed");
        return false;
    }
    if (rep)
        nr.frRepeats++;
    nr.frTweens++;
    return nativeFrucUnpack(nr, outB, nr.frPairSdr);
}

// a pipelined group's pair step: the pair the last group made current becomes the shown one (its levels 1..k, made by
// instance 0's worker, are among its nodes), then the new frame is packed, sparing the shown pair's first
// frame. The shown pair is held (its second frame in every slot) when it was identical, when the last group was not
// pipelined (it showed that pair itself: the route's entry costs one source period's hold) or when its second frame
// was packed again in the other encoding
static bool nativeFrucPipeAdvance(NativeRife& nr, const float* dCur, bool curStatic)
{
    std::swap(nr.dFrNode, nr.dFrLagNode);
    std::swap(nr.frNodeOk, nr.frLagOk);
    nr.frLagSerial = nr.frSerial;
    nr.frLagA = nr.frA;
    nr.frLagB = nr.frB;
    nr.frLagSdr = nr.frPairSdr;
    nr.frLagStatic = nr.frCurStatic;
    nr.frLagHave = nr.frPipePrev && nr.frLagA >= 0 && nr.frLagOk[kFrGrid / 2];
    nr.frLagL = 0;
    nr.frKeep = nr.frLagA;
    const bool ok = nativeFrucPair(nr, dCur, curStatic ? 0 : 1);
    nr.frKeep = -1;
    if (nr.frRepacked)
        nr.frLagHave = false;
    nr.frCurStatic = curStatic;
    nr.frPipePrev = true;
    nr.frPipeGroups++;
    return ok;
}

// the new pair's levels 1..k on instance 0's worker (the next group shows the pair), unless the pair is identical or has
// no first frame yet: k = the depth of this group's tweens less one (x4: the level-1 call alone, x8 two levels, x32
// four), so the worker's 2^k - 1 calls run beside the compute thread's 2^L - 2^k of the shown pair (its deepest
// level); the worker's calls run slower beside them, so depth - 1 is the best split at x8 to x64. The level gate starts
// at the shown pair's levels already made (kFrDepth when it is held: the compute thread makes none)
static bool nativeFrucPipeSpec(NativeRife& nr)
{
    if (nr.frCurStatic || nr.frA < 0)
        return true;
    const int k = (std::max)(1, (std::min)({nr.frPipeDepth - 1, nr.frInst - 1, frucPipeK()}));
    {
        std::lock_guard<std::mutex> lk(g_frLagM);
        g_frLagDone = nr.frLagHave && !nr.frLagStatic ? nativeFrucLagMade(nr) : kFrDepth;
    }
    nativeFrucSpecStart(nr, k);
    return true;
}

// a pipelined group's tween: the shown pair's levels the worker did not make (level by level, as nativeFrucTween) on
// their instances beside instance 0's worker's levels of the new pair, each level opening the gate for its twin; the
// worker joined before the slots' stores; then the nearest node
static bool nativeFrucLagTween(NativeRife& nr, double t)
{
    const FrTree tr = nativeFrucLag(nr);
    if (!nr.frLagL)
    {
        nr.frLagL = (std::min)(nativeFrucDepth(nr, &t, 1), nr.frInst);
        const int64_t c0 = nowQpc100();
        for (int l = 2; l <= nr.frLagL; l++)
        {
            for (int k = 1; k < (1 << l); k += 2)
                if (!nativeFrucNode(nr, tr, k, l))
                    return false;
            frucLagDone(l);
        }
        if (!nativeFrucSpecJoin(nr))
            return false;
        nr.grFrChainMs = (double)(nowQpc100() - c0) / 10000.0;
    }
    const int den = 1 << nr.frLagL;
    const int k = (std::max)(1, (std::min)(den - 1, (int)floor(t * den + 0.5)));
    uint8_t* node = nativeFrucNode(nr, tr, k, nr.frLagL);
    if (!node)
        return false;
    nr.frTweens++;
    return nativeFrucUnpack(nr, node, nr.frLagSdr);
}

static void nativeFrucFree(NativeRife& nr)
{
    if (nr.fsrfg)
        nativeFsrfgRelease(nr); // the bridge's instances and the Optical Flow session; the frames below are shared
    nativeFrucDrain(nr);
    for (int i = 0; i < kFrDepth; i++)
    {
        FrucWorker& w = g_frW[i];
        if (!w.th.joinable())
            continue;
        {
            std::lock_guard<std::mutex> lk(w.m);
            w.quit = true;
            w.cv.notify_all();
        }
        w.th.join();
    }
    if (nr.frCreated)
    {
        char mp[96] = "";
        if (nr.frMp)
            snprintf(mp, sizeof(mp), ", %llu midpoint calls%s", (unsigned long long)nr.frNodeCalls,
                     nr.frPipeGroups ? (", " + std::to_string(nr.frPipeGroups) + " groups pipelined").c_str() : "");
        LOG("native: fruc session: %llu tweens, %llu primed pairs, %llu frame repeats%s, %d instance%s\n",
            (unsigned long long)nr.frTweens, (unsigned long long)nr.frPrimed, (unsigned long long)nr.frRepeats, mp,
            nr.frInst, nr.frInst > 1 ? "s" : "");
        g_fruc.destroy(); // every instance
        nr.frCreated = false;
    }
    for (int i = 1; i < kFrDepth; i++)
        if (nr.dFrOutBI[i])
        {
            cudaFree(nr.dFrOutBI[i]);
            nr.dFrOutBI[i] = nullptr;
        }
    nr.dFrOutBI[0] = nullptr;
    nr.frInst = 1;
    nr.frPlanN = nr.frPlanK = 0;
    for (auto& s : nr.dFrSurf)
        if (s)
        {
            cudaFree(s);
            s = nullptr;
        }
    for (auto& s : nr.dFrNode)
        if (s)
        {
            cudaFree(s);
            s = nullptr;
        }
    for (auto& s : nr.dFrLagNode)
        if (s)
        {
            cudaFree(s);
            s = nullptr;
        }
    for (bool& ok : nr.frNodeOk)
        ok = false;
    for (bool& ok : nr.frLagOk)
        ok = false;
    nr.frLagHave = nr.frPipePrev = false;
    if (nr.dFrOutB)
    {
        cudaFree(nr.dFrOutB);
        nr.dFrOutB = nullptr;
    }
    if (nr.dFrOut)
    {
        cudaFree(nr.dFrOut);
        nr.dFrOut = nullptr;
    }
    if (nr.dFrRange)
    {
        cudaFree(nr.dFrRange);
        nr.dFrRange = nullptr;
    }
    if (nr.hFrRange)
    {
        cudaFreeHost(nr.hFrRange);
        nr.hFrRange = nullptr;
    }
    nr.frSdrOn = nr.frPairSdr = false;
}

// ---- AMD FSR / Intel XeSS frame generation (fsrfg) ----------------------------------------------
// One C API, two bridges (fgKind): smv_fsrfg_bridge.dll (source engine\fsrfg\build_src, exports fsrfg_*; SMV_FSRFG_DIR,
// else engine\fsrfg) and smv_xefg_bridge.dll (source engine\xess\build_src, exports xefg_*; SMV_XESS_DIR, else
// engine\xess, Intel's libxess_fg.dll + libxell.dll beside it). Each loads once per process by full path from the folder
// the handoff named; up to 16 instances, each its own context: FSR's on vkd3d-proton's device, XeSS-FG's on the
// adapter's D3D12 device (with its own swap chain and parked window), both for the CUDA device's adapter
struct FsrfgShared
{
    HANDLE in, mv, out, fence;           // three textures and a fence: Vulkan opaque Win32 exports and a timeline
                                         // semaphore (FSR), D3D12 resources and a D3D12 fence (XeSS-FG)
    uint64_t inBytes, mvBytes, outBytes; // the textures' allocation sizes
    uint32_t rowPitch, mvRowPitch;       // the linear rows a caller keeps its frames / vectors in
};
struct FsrfgBridge
{
    HMODULE mod = nullptr;
    const char* (*lastError)() = nullptr;
    void (*setCacheDir)(const char*) = nullptr;
    const char* (*provider)() = nullptr;
    int (*createI)(int, uint32_t, uint32_t, uint32_t, int32_t, int, FsrfgShared*) = nullptr;
    int (*frameI)(int, uint64_t, uint64_t, int, double) = nullptr;
    void (*destroy)() = nullptr;
    int (*devices)() = nullptr; // 1 = an own vkd3d device for every instance (optional; an older bridge has none)
    // XeSS-FG's counters of instance i: calls, calls with no frame presented, presents with no call, frames out
    int (*stats)(int, uint64_t*, uint64_t*, uint64_t*, uint64_t*) = nullptr; // optional (the XeSS-FG bridge's)
};
static FsrfgBridge g_fsrfgK[2];
static const char* const kFgDll[2] = {"smv_fsrfg_bridge.dll", "smv_xefg_bridge.dll"};
static const char* const kFgPrefix[2] = {"fsrfg_", "xefg_"};

// the bridge of this kind; FSR's vkd3d-proton shader cache in cacheDir\fsr4_shaders (the model cache: users may delete
// it); FSR's instances share vkd3d's one device (SMV_FSRFG_DEVICE=own = an own device each, ~235 MiB + ~0.3 s an
// instance)
static bool nativeFsrfgLoad(const std::string& dir, const std::string& cacheDir, int kind)
{
    FsrfgBridge& b = g_fsrfgK[kind];
    if (b.mod)
        return true;
    const std::wstring p = utf8ToWide(dir) + L"\\" + utf8ToWide(kFgDll[kind]);
    HMODULE m = LoadLibraryExW(p.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!m)
    {
        LOG("native: fsrfg: %s did not load from %s (err %lu)\n", kFgDll[kind], dir.c_str(), GetLastError());
        return false;
    }
    const std::string pre = kFgPrefix[kind];
    auto fn = [&](const char* name) { return GetProcAddress(m, (pre + name).c_str()); };
    b.lastError = (const char* (*)())fn("last_error");
    b.setCacheDir = (void (*)(const char*))fn("set_cache_dir");
    b.provider = (const char* (*)())fn("provider");
    b.createI = (int (*)(int, uint32_t, uint32_t, uint32_t, int32_t, int, FsrfgShared*))fn("create_i");
    b.frameI = (int (*)(int, uint64_t, uint64_t, int, double))fn("frame_i");
    b.destroy = (void (*)())fn("destroy");
    b.devices = (int (*)())fn("devices");
    b.stats = (int (*)(int, uint64_t*, uint64_t*, uint64_t*, uint64_t*))fn("stats");
    if (!b.lastError || !b.setCacheDir || !b.provider || !b.createI || !b.frameI || !b.destroy)
    {
        LOG("native: fsrfg: %s lacks an expected export\n", kFgDll[kind]);
        FreeLibrary(m);
        b = FsrfgBridge{};
        return false;
    }
    if (!cacheDir.empty())
        b.setCacheDir((cacheDir + "\\fsr4_shaders").c_str());
    b.mod = m;
    return true;
}

// instance i at w x h (RGBA16F) with its shared textures (as CUDA arrays) and fence imported into this CUDA context
static bool nativeFsrfgCreate(NativeRife& nr, int i)
{
    NativeRife::FgInst& g = nr.fg[i];
    FsrfgShared sh{};
    const FsrfgBridge& br = g_fsrfgK[nr.fgKind];
    const int rc = br.createI(i, (uint32_t)nr.w, (uint32_t)nr.h, nr.fgLuid.LowPart, nr.fgLuid.HighPart, 1, &sh);
    if (rc != 0)
    {
        LOG("native: fsrfg: instance %d at %dx%d not created: %s (rc %d)\n", i, nr.w, nr.h, br.lastError(), rc);
        return false;
    }
    g.made = true;
    if (nr.fgPitch && ((int)sh.rowPitch != nr.fgPitch || (int)sh.mvRowPitch != nr.fgMvPitch))
    {
        LOG("native: fsrfg: instance %d's row pitches %u / %u differ from instance 0's %d / %d\n", i, sh.rowPitch,
            sh.mvRowPitch, nr.fgPitch, nr.fgMvPitch);
        return false;
    }
    nr.fgPitch = (int)sh.rowPitch;
    nr.fgMvPitch = (int)sh.mvRowPitch;
    const int w = nr.w, h = nr.h;
    const bool d3d = nr.fgKind == 1;
    auto imp = [w, h, d3d](HANDLE hd, uint64_t bytes, const cudaChannelFormatDesc& fd, cudaExternalMemory_t& em,
                           cudaMipmappedArray_t& mm, cudaArray_t& a) {
        cudaExternalMemoryHandleDesc md{};
        md.type = d3d ? cudaExternalMemoryHandleTypeD3D12Resource : cudaExternalMemoryHandleTypeOpaqueWin32;
        md.size = bytes;
        md.flags = cudaExternalMemoryDedicated;
        md.handle.win32.handle = hd;
        if (cudaImportExternalMemory(&em, &md) != cudaSuccess)
        {
            em = nullptr;
            return false;
        }
        cudaExternalMemoryMipmappedArrayDesc ad{};
        ad.formatDesc = fd;
        ad.extent.width = (size_t)w;
        ad.extent.height = (size_t)h;
        ad.numLevels = 1;
        return cudaExternalMemoryGetMappedMipmappedArray(&mm, em, &ad) == cudaSuccess &&
               cudaGetMipmappedArrayLevel(&a, mm, 0) == cudaSuccess;
    };
    const cudaChannelFormatDesc rgba = cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindFloat);
    const cudaChannelFormatDesc rg = cudaCreateChannelDesc(16, 16, 0, 0, cudaChannelFormatKindFloat);
    cudaExternalSemaphoreHandleDesc sd{};
    sd.type = d3d ? cudaExternalSemaphoreHandleTypeD3D12Fence : cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
    sd.handle.win32.handle = sh.fence;
    if (!imp(sh.in, sh.inBytes, rgba, g.emIn, g.mmIn, g.in) || !imp(sh.mv, sh.mvBytes, rg, g.emMv, g.mmMv, g.mv) ||
        !imp(sh.out, sh.outBytes, rgba, g.emOut, g.mmOut, g.out) ||
        cudaImportExternalSemaphore(&g.sem, &sd) != cudaSuccess)
    {
        LOG("native: fsrfg: instance %d's shared textures or fence did not import into CUDA\n", i);
        return false;
    }
    if (!nr.dFgMv && cudaMalloc((void**)&nr.dFgMv, (size_t)nr.fgMvPitch * nr.h) != cudaSuccess)
    {
        nr.dFgMv = nullptr;
        LOG("native: fsrfg: the vector buffer did not allocate\n");
        return false;
    }
    g.val = 0;
    nr.frLastKey[i] = 0;
    nr.frCalls[i] = 0;
    return true;
}

// an OfVec session at w x h (FSR's vectors: ABGR8, one direction, grid 1 = the measurement harness's vectors, or grid 4
// with k_nvofUp's bilinear taps where the engine refuses grid 1) at nativeOfLevel(); tag names
// it in the log lines; st = the stream it runs on (null = nr.stream)
static bool nativeOfVecSetup(NativeRife& nr, NativeRife::OfVec& o, int w, int h, const char* tag,
                             cudaStream_t st = nullptr)
{
    if (!nativeNvofLoad())
        return false;
    if (w < 32 || h < 32)
    {
        LOG("native: %s: the frame %dx%d is below the Optical Flow minimum 32x32\n", tag, w, h);
        return false;
    }
    CUcontext ctx = nullptr;
    if (cuCtxGetCurrent(&ctx) != CUDA_SUCCESS || !ctx)
    {
        LOG("native: %s: no current CUDA context\n", tag);
        return false;
    }
    o.w = w;
    o.h = h;
    o.st = st;
    {
        char ev[8] = {};
        o.both = GetEnvironmentVariableA("SMV_OFVEC_BOTH", ev, sizeof(ev)) > 0 && ev[0] == '1';
    }
    o.level = nativeOfLevel();
    for (int grid : {1, 4})
    {
        if (g_nvofApi.nvCreateOpticalFlowCuda(ctx, &o.of) != NV_OF_SUCCESS)
        {
            o.of = nullptr;
            LOG("native: %s: nvCreateOpticalFlowCuda failed\n", tag);
            return false;
        }
        NV_OF_INIT_PARAMS ip{};
        ip.width = (uint32_t)w;
        ip.height = (uint32_t)h;
        ip.outGridSize = (NV_OF_OUTPUT_VECTOR_GRID_SIZE)grid;
        ip.hintGridSize = NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED;
        ip.mode = NV_OF_MODE_OPTICALFLOW;
        ip.perfLevel = o.level;
        ip.enableExternalHints = NV_OF_FALSE;
        // FSR reads no cost: off
        ip.enableOutputCost = NV_OF_FALSE;
        ip.disparityRange = NV_OF_STEREO_DISPARITY_RANGE_UNDEFINED;
        ip.enableRoi = NV_OF_FALSE;
        // FSR reads one field, current -> previous: the forward field with the two frames swapped (half the work of
        // both directions); SMV_OFVEC_BOTH=1 = both directions, the backward one read
        ip.predDirection = o.both ? NV_OF_PRED_DIRECTION_BOTH : NV_OF_PRED_DIRECTION_FORWARD;
        ip.enableGlobalFlow = NV_OF_FALSE;
        ip.inputBufferFormat = NV_OF_BUFFER_FORMAT_ABGR8;
        if (g_nvofApi.nvOFInit(o.of, &ip) == NV_OF_SUCCESS)
        {
            o.grid = grid;
            break;
        }
        g_nvofApi.nvOFDestroy(o.of);
        o.of = nullptr;
    }
    if (!o.of)
    {
        LOG("native: %s: the Optical Flow engine refused %dx%d at grid 1 and 4\n", tag, w, h);
        return false;
    }
    if (o.grid != 1)
        LOG("native: %s: the Optical Flow engine refused grid 1, vectors at grid 4 (bilinear)\n", tag);
    o.gw = (w + o.grid - 1) / o.grid;
    o.gh = (h + o.grid - 1) / o.grid;
    for (int k = 0; k < 2; k++)
    {
        uint32_t inPitch = 0;
        if (!nativeNvofBufOn(o.of, w, h, NV_OF_BUFFER_USAGE_INPUT, NV_OF_BUFFER_FORMAT_ABGR8, o.in[k], o.inP[k],
                             inPitch, "FSR vector input") ||
            !nativeNvofBufOn(o.of, o.gw, o.gh, NV_OF_BUFFER_USAGE_OUTPUT, NV_OF_BUFFER_FORMAT_SHORT2, o.out[k],
                             o.outP[k], o.outPitch[k], "FSR vector flow"))
            return false;
        if (k == 0)
            o.inPitch = inPitch;
        else if (inPitch != o.inPitch)
        {
            LOG("native: %s: the two Optical Flow inputs differ in pitch\n", tag);
            return false;
        }
    }
    const CUstream ios = (CUstream)(st ? st : nr.stream);
    if (g_nvofApi.nvOFSetIOCudaStreams(o.of, ios, ios) != NV_OF_SUCCESS)
    {
        LOG("native: %s: nvOFSetIOCudaStreams failed\n", tag);
        return false;
    }
    if (o.grid != 1) // grid 1 writes the texels straight from the field (k_mvHalfG1)
        NCHK(cudaMalloc((void**)&o.dFlow, 2 * (size_t)w * h * sizeof(float)), "alloc FSR vectors");
    return true;
}

// the vectors between slots `prev` and `cur`, both holding their ABGR8 frames: the field cur -> prev in px (the backward
// field of the pair), into dst as R16G16_FLOAT texels, rows dstPitch bytes apart
static bool nativeOfVecRun(NativeRife& nr, NativeRife::OfVec& o, int prev, int cur, uint8_t* dst, int dstPitch,
                           const char* tag, cudaEvent_t evOf = nullptr, float gate = 0.0f)
{
    cudaStream_t st = o.st ? o.st : nr.stream;
    NV_OF_EXECUTE_INPUT_PARAMS ei{};
    NV_OF_EXECUTE_OUTPUT_PARAMS eo{};
    ei.disableTemporalHints = NV_OF_TRUE;
    if (o.both)
    {
        ei.inputFrame = o.in[prev];
        ei.referenceFrame = o.in[cur];
        eo.outputBuffer = o.out[0];
        eo.bwdOutputBuffer = o.out[1];
    }
    else
    {
        // the forward field from `cur` to `prev` = the backward one of (prev, cur), into the buffers it is read from
        ei.inputFrame = o.in[cur];
        ei.referenceFrame = o.in[prev];
        eo.outputBuffer = o.out[o.wr];
    }
    const NV_OF_STATUS s = g_nvofApi.nvOFExecute(o.of, &ei, &eo);
    if (s != NV_OF_SUCCESS)
    {
        LOG("native: %s: nvOFExecute failed (status %d)\n", tag, (int)s);
        nr.die("FSR optical flow failed");
        return false;
    }
    if (evOf) // a profiler's mark between the Optical Flow and the vector kernels (the session's stream)
        cudaEventRecord(evOf, st);
    const unsigned gx = (unsigned)(o.w + 15) / 16, gy = (unsigned)(o.h + 15) / 16;
    const int f = o.both ? 1 : o.wr;
    int grid = o.grid, vp = (int)o.outPitch[f], cp = 0, mp = dstPitch;
    CUdeviceptr v = o.outP[f], c = 0;
    if (grid == 1) // the field is the texels' grid: one pass, no float planes in between
    {
        void* g1[] = {&v, &vp, &o.w, &o.h, &dst, &mp, &gate};
        if (cuLaunchKernel(nr.fMvHalfG1, gx, gy, 1, 16, 16, 1, 0, (CUstream)st, g1, nullptr) != CUDA_SUCCESS)
        {
            nr.die("FSR vector launch failed");
            return false;
        }
        return true;
    }
    float* flow = o.dFlow;
    float* cost = nullptr; // k_nvofUp: the flow alone, the cost never read
    void* b[] = {&v, &vp, &c, &cp, &o.gw, &o.gh, &grid, &o.w, &o.h, &flow, &cost};
    void* m[] = {&flow, &o.w, &o.h, &dst, &mp, &gate};
    if (cuLaunchKernel(nr.fNvofUp, gx, gy, 1, 16, 16, 1, 0, (CUstream)st, b, nullptr) != CUDA_SUCCESS ||
        cuLaunchKernel(nr.fMvHalf, gx, gy, 1, 16, 16, 1, 0, (CUstream)st, m, nullptr) != CUDA_SUCCESS)
    {
        nr.die("FSR vector launch failed");
        return false;
    }
    return true;
}

static void nativeOfVecFree(NativeRife::OfVec& o)
{
    for (NvOFGPUBufferHandle* b : {&o.in[0], &o.in[1], &o.out[0], &o.out[1]})
        if (*b)
            g_nvofApi.nvOFDestroyGPUBufferCuda(*b);
    if (o.of)
        g_nvofApi.nvOFDestroy(o.of);
    if (o.dFlow)
        cudaFree(o.dFlow);
    o = NativeRife::OfVec{};
}

// prev and frame (RGBA16F) into OfVec o's two inputs as ABGR8, on o's stream; the slots they are in come back in
// slotPrev / slotCur. chain = the level-1 pass, whose prev is the last level-1 pass's frame: when a slot still holds
// that surface (keepSrc), only the new frame is converted, into the other slot
static bool nativeFsrfgOfIn(NativeRife& nr, NativeRife::OfVec& o, const uint8_t* prev, const uint8_t* frame, bool chain,
                            int& slotPrev, int& slotCur)
{
    const unsigned gx = (unsigned)(nr.w + 15) / 16, gy = (unsigned)(nr.h + 15) / 16;
    int sp = nr.fgPitch, ip = (int)o.inPitch;
    const bool kept = chain && o.keepSlot >= 0 && o.keepSrc == prev;
    slotPrev = kept ? o.keepSlot : 0;
    slotCur = 1 - slotPrev;
    for (int k = kept ? 1 : 0; k < 2; k++)
    {
        const uint8_t* s = k ? frame : prev;
        CUdeviceptr d = o.inP[k ? slotCur : slotPrev];
        void* a[] = {(void*)&s, &sp, &nr.w, &nr.h, &d, &ip};
        if (cuLaunchKernel(nr.fRgbaHAbgr, gx, gy, 1, 16, 16, 1, 0, (CUstream)(o.st ? o.st : nr.stream), a, nullptr) !=
            CUDA_SUCCESS)
        {
            nr.die("rgbaHAbgr (fsrfg) launch failed");
            return false;
        }
    }
    o.keepSlot = chain ? slotCur : -1;
    o.keepSrc = chain ? frame : nullptr;
    return true;
}

// AMD FSR 4's level-2+ calls take vectors derived from their pair's level-1 field (k_mvDerive) instead of an optical-flow
// pass each: one SLOW pass a call bounds FSR 4 on the optical-flow engine (72-86 % busy, the SMs 17-22 %), and on real
// 1000 fps footage the derived tweens score closer to the true frames. The field must be grid 1
// (an engine that refuses grid 1 keeps a pass a call). On by default, every instance on the one shared vkd3d device
// (calls of different instances back to back stay exact because our vkd3d does not offer the coopmat2 conversions);
// SMV_FSRFG_CHILDMV=0 = a pass a call
static bool fsrfgChildEnvOn()
{
    static const bool on = [] {
        char v[8] = "";
        return !(GetEnvironmentVariableA("SMV_FSRFG_CHILDMV", v, sizeof(v)) > 0 && v[0] == '0');
    }();
    return on;
}

static bool fsrfgDeriveOn(const NativeRife& nr)
{
    return fsrfgChildEnvOn() && nr.fgOf.grid == 1;
}

// a level-2+ call's derived vectors (its frames at t0 and t1 of the pair) into dst (rows fgMvPitch apart) on stream st,
// from fgOf's level-1 output: no level-2+ call runs fgOf once its vectors are derived, so the field holds for the pair
static bool nativeFsrfgChildVec(NativeRife& nr, uint8_t* dst, cudaStream_t st, float t0, float t1)
{
    CUdeviceptr v = nr.fgOf.outP[nr.fgOfRd];
    int vp = (int)nr.fgOf.outPitch[nr.fgOfRd], mp = nr.fgMvPitch;
    void* a[] = {&v, &vp, &nr.w, &nr.h, &t0, &t1, &dst, &mp};
    if (cuLaunchKernel(nr.fMvDerive, (unsigned)(nr.w + 15) / 16, (unsigned)(nr.h + 15) / 16, 1, 16, 16, 1, 0,
                       (CUstream)st, a, nullptr) != CUDA_SUCCESS)
    {
        nr.die("fsrfg derived vector launch failed");
        return false;
    }
    return true;
}

// one bridge call: instance i gets `frame` (RGBA16F rows fgPitch apart) with its vectors toward `prev` (the frame the
// instance was fed last; null = zero vectors: a reset, or a re-prime nobody reads) and returns into `out` (null =
// dropped) the frame halfway between them. The vectors as the harness made them: the field this frame -> the previous
// (px) = FSR's convention. ring >= 0: the vectors were made ahead into dFgMvR[ring] (nativeFsrfgAheadVec). l1 = the
// group's level-1 call: its FSR span goes into fgL1Ev for the x2 call price (nativeGroup). part 0 = the whole call on
// the session's stream; on fgL1St (the pipelined route's instance-0 calls, their vectors in dFgMvL1): 1 = the level-1
// Optical Flow on fgOfL1St, then the frame and its vectors into the instance's textures, 2 = after fgEvDeep the call
// and its copy out, 3 = the whole call (a reset or a re-prime, in a group that runs no deeper call)
static bool nativeFsrfgFeed(NativeRife& nr, int i, const uint8_t* frame, const uint8_t* prev, bool reset, uint8_t* out,
                            int ring = -1, bool l1 = false, int part = 0)
{
    NativeRife::FgInst& g = nr.fg[i];
    cudaStream_t st = part ? nr.fgL1St : nr.stream;
    uint8_t* mvBuf = part ? nr.dFgMvL1 : nr.dFgMv;
    const size_t row = (size_t)nr.w * 8;
    auto mark = [&nr, st](int e) {
        if (nr.fgProf)
            cudaEventRecord(nr.fgEv[e], st);
    };
    bool derived = false;
    if (part != 2)
    {
        // the pipelined level-1 pass (part 1) starts from the pack, not behind the instance's last call on fgL1St:
        // the frame goes into the instance's texture after the pass
        const bool pass1 = part == 1;
        mark(0);
        if (!pass1 && cudaMemcpy2DToArrayAsync(g.in, 0, 0, frame, nr.fgPitch, row, nr.h, cudaMemcpyDeviceToDevice,
                                               st) != cudaSuccess)
        {
            nr.die("fsrfg frame copy failed");
            return false;
        }
        mark(1);
        const uint8_t* mv = ring >= 0 ? nr.dFgMvR[ring] : mvBuf;
        if (ring >= 0)
        {
            if (cudaStreamWaitEvent(st, nr.fgEvVec[ring], 0) != cudaSuccess)
            {
                nr.die("fsrfg vector wait failed");
                return false;
            }
        }
        else if (prev && (i == 0 || !fsrfgDeriveOn(nr)))
        {
            // fgOf runs on fgOfL1St: from another stream the pass waits for that stream's work so far (the pipelined
            // pass: for the pack, nativeFsrfgPipeAdvance, and for fgL1St's last read of dFgMvL1), and it for the pass
            const bool hop = nr.fgOf.st && nr.fgOf.st != st;
            if (hop && (pass1 ? cudaStreamWaitEvent(nr.fgOf.st, nr.fgEvMv, 0) != cudaSuccess
                              : (cudaEventRecord(nr.fgEvS, st) != cudaSuccess ||
                                 cudaStreamWaitEvent(nr.fgOf.st, nr.fgEvS, 0) != cudaSuccess)))
            {
                nr.die("fsrfg vector event failed");
                return false;
            }
            // the level-1 pass's span for the pipelined route's price (nativeGroup)
            cudaStream_t os = nr.fgOf.st ? nr.fgOf.st : nr.stream;
            const int es = nr.fgEvSet;
            if (l1 && nr.fgOfEv[es][0])
                nr.grOfArmed[es] = cudaEventRecord(nr.fgOfEv[es][0], os) == cudaSuccess;
            int slotPrev = 0, slotCur = 1;
            if (!nativeFsrfgOfIn(nr, nr.fgOf, prev, frame, i == 0, slotPrev, slotCur))
                return false;
            mark(2);
            if (!nativeOfVecRun(nr, nr.fgOf, slotPrev, slotCur, mvBuf, nr.fgMvPitch, "fsrfg",
                                nr.fgProf ? nr.fgEv[3] : nullptr))
                return false;
            if (l1 && nr.grOfArmed[es])
                nr.grOfArmed[es] = cudaEventRecord(nr.fgOfEv[es][1], os) == cudaSuccess;
            if (hop && (cudaEventRecord(nr.fgEvOf, nr.fgOf.st) != cudaSuccess ||
                        cudaStreamWaitEvent(st, nr.fgEvOf, 0) != cudaSuccess))
            {
                nr.die("fsrfg vector event failed");
                return false;
            }
        }
        else if (prev)
        {
            derived = true;
            mark(2);
            if (!nativeFsrfgChildVec(nr, mvBuf, st, nr.fgChildT0, nr.fgChildT1))
                return false;
            mark(3);
        }
        else if (cudaMemset2DAsync(mvBuf, nr.fgMvPitch, 0, (size_t)nr.w * 4, nr.h, st) != cudaSuccess)
        {
            nr.die("fsrfg vector clear failed");
            return false;
        }
        else
        {
            mark(2);
            mark(3);
        }
        if (pass1 && cudaMemcpy2DToArrayAsync(g.in, 0, 0, frame, nr.fgPitch, row, nr.h, cudaMemcpyDeviceToDevice, st) !=
                         cudaSuccess)
        {
            nr.die("fsrfg frame copy failed");
            return false;
        }
        mark(4);
        // fgEvMv: dFgMvL1 is read, the next level-1 pass may write it
        if (cudaMemcpy2DToArrayAsync(g.mv, 0, 0, mv, nr.fgMvPitch, (size_t)nr.w * 4, nr.h, cudaMemcpyDeviceToDevice,
                                     st) != cudaSuccess ||
            (ring >= 0 && cudaEventRecord(nr.fgEvFree[ring], st) != cudaSuccess) ||
            (part && cudaEventRecord(nr.fgEvMv, st) != cudaSuccess))
        {
            nr.die("fsrfg vector copy failed");
            return false;
        }
        mark(5);
        if (part == 1)
            return true;
    }
    else if (cudaStreamWaitEvent(st, nr.fgEvDeep, 0) != cudaSuccess)
    {
        nr.die("fsrfg call wait failed");
        return false;
    }
    const int es = nr.fgEvSet;
    if (l1 && nr.fgL1Ev[es][0])
        nr.grL1Armed[es] = cudaEventRecord(nr.fgL1Ev[es][0], st) == cudaSuccess;
    cudaExternalSemaphoreSignalParams sp{};
    sp.params.fence.value = ++g.val;
    if (cudaSignalExternalSemaphoresAsync(&g.sem, &sp, 1, st) != cudaSuccess)
    {
        nr.die("fsrfg fence signal failed");
        return false;
    }
    const uint64_t ready = g.val, done = ++g.val;
    const int64_t c0 = nr.fgProf ? nowQpc100() : 0;
    // the frame time changes neither FSR 4's output (a 24 and a 60 fps run byte for byte the same) nor XeSS-FG's (33 and
    // 0 ms the same): one 60 Hz frame
    const FsrfgBridge& br = g_fsrfgK[nr.fgKind];
    const int rc = br.frameI(i, ready, done, reset ? 1 : 0, 1000.0 / 60.0);
    if (nr.fgProf)
        nr.fgCpuMs += (double)(nowQpc100() - c0) / 10000.0;
    if (rc != 0)
    {
        LOG("native: fsrfg: instance %d call failed: %s (rc %d)\n", i, br.lastError(), rc);
        nr.die("fsrfg call failed");
        return false;
    }
    cudaExternalSemaphoreWaitParams wp{};
    wp.params.fence.value = done;
    if (cudaWaitExternalSemaphoresAsync(&g.sem, &wp, 1, st) != cudaSuccess)
    {
        nr.die("fsrfg result failed");
        return false;
    }
    mark(6);
    if (l1 && nr.grL1Armed[es])
        nr.grL1Armed[es] = cudaEventRecord(nr.fgL1Ev[es][1], st) == cudaSuccess;
    if (out && cudaMemcpy2DFromArrayAsync(out, nr.fgPitch, g.out, 0, 0, row, nr.h, cudaMemcpyDeviceToDevice, st) !=
                   cudaSuccess)
    {
        nr.die("fsrfg result failed");
        return false;
    }
    mark(7);
    if (nr.fgProf && cudaEventSynchronize(nr.fgEv[7]) == cudaSuccess)
    {
        for (int s = 0; s < 7; s++)
        {
            float ms = 0.0f;
            if (cudaEventElapsedTime(&ms, nr.fgEv[s], nr.fgEv[s + 1]) == cudaSuccess)
                (derived && s >= 1 && s <= 3 ? nr.fgDeriveMs : nr.fgStageMs[s]) += ms;
        }
        nr.fgProfN++;
        nr.fgProfVecN += prev && !derived ? 1 : 0;
        nr.fgProfDerN += derived ? 1 : 0;
    }
    nr.frCalls[i]++;
    return true;
}

// the extra instances up to `want` (instance 0 is the session's own); a failed create keeps the ones made so far
static void nativeFsrfgGrow(NativeRife& nr, int want)
{
    want = (std::min)(want, kFrDepth);
    const int64_t t0 = nowQpc100();
    const int had = nr.frInst;
    for (int i = nr.frInst; i < want; i++)
    {
        if (!nativeFsrfgCreate(nr, i))
        {
            LOG("native: fsrfg: staying at %d instance%s\n", nr.frInst, nr.frInst > 1 ? "s" : "");
            nr.frInstMax = nr.frInst;
            break;
        }
        nr.frInst = i + 1;
    }
    if (nr.frInst > had)
    {
        nr.grPriceSkip = 2;
        LOG("native: fsrfg: %d instances (%d new in %.0f ms)\n", nr.frInst, nr.frInst - had,
            (double)(nowQpc100() - t0) / 10000.0);
    }
}

// node k / 2^L of this pair's midpoint tree (FRUC's nativeFrucNode): FSR on instance L - 1 fed the right parent
// after the left one. An instance fed another frame last (a pair of another depth, a new encoding) gets the left
// parent first, its output dropped: a reset only for its very first frame (after a reset FSR's next 10 tweens follow
// the vectors alone, ffx_frameinterpolation.h fFrameIndexFactor), else a plain call. ring >= 0: the node's vectors were
// made ahead into dFgMvR[ring] (nativeFsrfgLevel). The tree = the pair shown: this group's, or the previous one while
// pipelined (its nodes in dFrNode either way, nativeFsrfgPipeAdvance)
static FrTree nativeFsrfgShown(NativeRife& nr)
{
    return nr.fgPiped ? FrTree{nr.frLagSerial, nr.frLagA, nr.frLagB, nr.dFrNode, nr.frNodeOk} : nativeFrucCur(nr);
}

static uint8_t* nativeFsrfgNode(NativeRife& nr, int k, int L, int ring = -1)
{
    while (L > 0 && !(k & 1))
    {
        k >>= 1;
        L--;
    }
    const FrTree tr = nativeFsrfgShown(nr);
    if (L <= 0)
        return nr.dFrSurf[k ? tr.b : tr.a];
    const int p = k << (kFrDepth - L);
    if (nr.frNodeOk[p])
        return nr.dFrNode[p];
    uint8_t* left = nativeFsrfgNode(nr, k - 1, L);
    uint8_t* right = left ? nativeFsrfgNode(nr, k + 1, L) : nullptr;
    if (!right)
        return nullptr;
    if (!nr.dFrNode[p] && cudaMalloc((void**)&nr.dFrNode[p], (size_t)nr.fgPitch * nr.h) != cudaSuccess)
    {
        nr.dFrNode[p] = nullptr;
        nr.die("fsrfg midpoint buffer alloc failed");
        return nullptr;
    }
    const int i = L - 1;
    const uint64_t lk = nativeFrucKey(tr, k - 1, L);
    if (nr.frLastKey[i] != lk)
    {
        if (!nativeFsrfgFeed(nr, i, left, nullptr, nr.frLastKey[i] == 0, nullptr))
            return nullptr;
        if (nr.frLastKey[i] != 0)
            nr.frPrimed++;
        nr.frLastKey[i] = lk;
    }
    const int64_t c0 = nowQpc100();
    nr.fgChildT0 = (float)(k - 1) / (float)(1 << L);
    nr.fgChildT1 = (float)(k + 1) / (float)(1 << L);
    if (!nativeFsrfgFeed(nr, i, right, left, false, nr.dFrNode[p], ring, i == 0))
        return nullptr;
    nr.grFrCalls++;
    nr.grFrCallMs += (double)(nowQpc100() - c0) / 10000.0;
    nr.frLastKey[i] = nativeFrucKey(tr, k + 1, L);
    nr.frNodeCalls++;
    nr.frNodeOk[p] = true;
    return nr.dFrNode[p];
}

static void nativeFsrfgAheadFree(NativeRife& nr)
{
    if (nr.fgOfSt)
        cudaStreamSynchronize(nr.fgOfSt);
    nativeOfVecFree(nr.fgOf2);
    for (int s = 0; s < 2; s++)
    {
        if (nr.dFgMvR[s])
            cudaFree(nr.dFgMvR[s]);
        nr.dFgMvR[s] = nullptr;
        for (cudaEvent_t* e : {&nr.fgEvVec[s], &nr.fgEvFree[s]})
            if (*e)
            {
                cudaEventDestroy(*e);
                *e = nullptr;
            }
    }
    if (nr.fgEvLevel)
        cudaEventDestroy(nr.fgEvLevel);
    nr.fgEvLevel = nullptr;
    if (nr.fgOfSt)
        cudaStreamDestroy(nr.fgOfSt);
    nr.fgOfSt = nullptr;
    nr.fgAhead = 0;
}

// the side stream, its Optical Flow session and the two vector buffers, made at the session's first level with 2+
// calls; false = serial vectors (SMV_FSRFG_OVERLAP=0, the profiler, or a failed setup, which says so;
// SMV_FSRFG_OVERLAP=fail = a failed setup after a whole one, the fallback's trigger)
static bool nativeFsrfgAhead(NativeRife& nr)
{
    if (nr.fgAhead)
        return nr.fgAhead > 0;
    char ev[8] = {};
    if (nr.fgProf || (GetEnvironmentVariableA("SMV_FSRFG_OVERLAP", ev, sizeof(ev)) > 0 && ev[0] == '0'))
    {
        nr.fgAhead = -1;
        return false;
    }
    const size_t bytes = (size_t)nr.fgMvPitch * nr.h;
    bool ok = cudaStreamCreateWithFlags(&nr.fgOfSt, cudaStreamNonBlocking) == cudaSuccess &&
              cudaEventCreateWithFlags(&nr.fgEvLevel, cudaEventDisableTiming) == cudaSuccess;
    for (int s = 0; ok && s < 2; s++)
    {
        uint8_t* b = nullptr;
        ok = cudaEventCreateWithFlags(&nr.fgEvVec[s], cudaEventDisableTiming) == cudaSuccess &&
             cudaEventCreateWithFlags(&nr.fgEvFree[s], cudaEventDisableTiming) == cudaSuccess &&
             cudaMalloc((void**)&b, bytes) == cudaSuccess;
        nr.dFgMvR[s] = ok ? b : nullptr;
    }
    // derived level-2+ vectors need no side Optical Flow session (fsrfgDeriveOn)
    const bool derive = fsrfgDeriveOn(nr);
    if (!ok || (!derive && !nativeOfVecSetup(nr, nr.fgOf2, nr.w, nr.h, "fsrfg side vectors", nr.fgOfSt)) ||
        strcmp(ev, "fail") == 0)
    {
        LOG(derive ? "native: fsrfg: the side stream did not start, every call's vectors stay in its turn\n"
                   : "native: fsrfg: the side Optical Flow session did not start, every call's vectors stay in its "
                     "turn\n");
        nativeFsrfgAheadFree(nr);
        nr.fgAhead = -1;
        return false;
    }
    nr.fgAhead = 1;
    LOG(derive ? "native: fsrfg: tree levels with 2+ calls derive the next call's vectors during each FSR call (a side "
                 "stream)\n"
               : "native: fsrfg: tree levels with 2+ calls make the next call's vectors during each FSR call (a side "
                 "stream, its own Optical Flow session)\n");
    return true;
}

// a call's vectors (the field frame -> prev; its frames at t0 / t1 of the pair) on the side stream into dFgMvR[s] once
// that buffer's last copy is done, then fgEvVec[s]: derived (fsrfgDeriveOn) or the side session's optical flow. The
// copy's event is recorded before this is enqueued: a stream waits for the event's last record
static bool nativeFsrfgAheadVec(NativeRife& nr, int s, const uint8_t* prev, const uint8_t* frame, float t0, float t1)
{
    if (cudaStreamWaitEvent(nr.fgOfSt, nr.fgEvFree[s], 0) != cudaSuccess)
    {
        nr.die("fsrfg side vector wait failed");
        return false;
    }
    if (fsrfgDeriveOn(nr))
    {
        if (!nativeFsrfgChildVec(nr, nr.dFgMvR[s], nr.fgOfSt, t0, t1))
            return false;
    }
    else
    {
        int slotPrev = 0, slotCur = 1;
        if (!nativeFsrfgOfIn(nr, nr.fgOf2, prev, frame, false, slotPrev, slotCur) ||
            !nativeOfVecRun(nr, nr.fgOf2, slotPrev, slotCur, nr.dFgMvR[s], nr.fgMvPitch, "fsrfg side vectors"))
            return false;
    }
    if (cudaEventRecord(nr.fgEvVec[s], nr.fgOfSt) != cudaSuccess)
    {
        nr.die("fsrfg side vector event failed");
        return false;
    }
    return true;
}

// level l (2^(l-1) calls on instance l - 1) with its vectors made ahead: every call's two frames are coarser nodes,
// done before the level starts, so call j + 1's Optical Flow (the OFA engine) runs while call j's FSR (vkd3d's queue)
// does; each instance still gets nativeFsrfgNode's calls in the same order
static bool nativeFsrfgLevel(NativeRife& nr, int l)
{
    const int n = 1 << (l - 1);
    uint8_t* lf[kFrGrid / 2] = {};
    uint8_t* rt[kFrGrid / 2] = {};
    for (int j = 0; j < n; j++)
        if (!(lf[j] = nativeFsrfgNode(nr, 2 * j, l)) || !(rt[j] = nativeFsrfgNode(nr, 2 * j + 2, l)))
            return false;
    if (cudaEventRecord(nr.fgEvLevel, nr.stream) != cudaSuccess ||
        cudaStreamWaitEvent(nr.fgOfSt, nr.fgEvLevel, 0) != cudaSuccess)
    {
        nr.die("fsrfg level event failed");
        return false;
    }
    const float den = (float)(1 << l); // call j's frames = nodes 2 j and 2 j + 2 of 2^l
    if (!nativeFsrfgAheadVec(nr, 0, lf[0], rt[0], 0.0f, 2.0f / den))
        return false;
    for (int j = 0; j < n; j++)
    {
        if (j + 1 < n && !nativeFsrfgAheadVec(nr, (j + 1) & 1, lf[j + 1], rt[j + 1], (2.0f * j + 2.0f) / den,
                                              (2.0f * j + 4.0f) / den))
            return false;
        if (!nativeFsrfgNode(nr, 2 * j + 1, l, j & 1))
            return false;
    }
    // the stream waits for the side stream's last work too (a node made earlier leaves its vectors unread), so the
    // next pair's frames never overwrite a node the side stream still reads
    if (cudaEventRecord(nr.fgEvLevel, nr.fgOfSt) != cudaSuccess ||
        cudaStreamWaitEvent(nr.stream, nr.fgEvLevel, 0) != cudaSuccess)
    {
        nr.die("fsrfg level event failed");
        return false;
    }
    return true;
}

// one frame's planes into frame surface n as RGBA16F: the SDR view (sdr) or the planes as stored
static bool nativeFsrfgPack(NativeRife& nr, const float* src, int n, bool sdr)
{
    int ps = nr.ph * nr.pw, sdrI = sdr ? 1 : 0, pitch = nr.fgPitch;
    void* a[] = {(void*)&src, &ps, &nr.pw, &nr.w, &nr.h, &sdrI, &nr.encPost, &nr.encWhite, &nr.dFrSurf[n], &pitch};
    if (cuLaunchKernel(nr.fPackRgbaH, (nr.w + 15) / 16, (nr.h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)nr.stream, a,
                       nullptr) != CUDA_SUCCESS)
    {
        nr.die("packRgbaH (fsrfg) launch failed");
        return false;
    }
    nr.frSurfSdr[n] = sdr;
    if (nr.fgOf.keepSrc == nr.dFrSurf[n]) // its ABGR8 copy in the Optical Flow's input slot is stale now
    {
        nr.fgOf.keepSlot = -1;
        nr.fgOf.keepSrc = nullptr;
    }
    return true;
}

// the level-1 chain's two streams and their events (nativeFsrfgSetup makes them)
static void nativeFsrfgChainFree(NativeRife& nr)
{
    for (cudaEvent_t* e : {&nr.fgEvL1, &nr.fgEvS, &nr.fgEvDeep, &nr.fgEvOf, &nr.fgEvMv, &nr.fgEvSurf[0],
                           &nr.fgEvSurf[1], &nr.fgEvSurf[2]})
        if (*e)
        {
            cudaEventDestroy(*e);
            *e = nullptr;
        }
    for (cudaStream_t* s : {&nr.fgL1St, &nr.fgOfL1St})
        if (*s)
        {
            cudaStreamDestroy(*s);
            *s = nullptr;
        }
}

static bool nativeFsrfgSetup(NativeRife& nr)
{
    if (!nativeFsrfgLoad(nr.frucDir, nr.cachePath, nr.fgKind))
        return false;
    int dev = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess)
    {
        LOG("native: fsrfg: the CUDA device's properties are unavailable\n");
        return false;
    }
    memcpy(&nr.fgLuid, prop.luid, sizeof(nr.fgLuid));
    nr.fgPitch = nr.fgMvPitch = 0;
    for (NativeRife::FgInst& g : nr.fg)
        g = NativeRife::FgInst{};
    // fgOf on its own stream fgOfL1St, instance 0's pipelined calls on fgL1St (the pipelined route's level-1 chain);
    // the profiler keeps both on the session's
    nr.fgOfWr = nr.fgOfRd = 1;
    nr.fgPiped = nr.fgL1Pend = nr.fgL1Ok = nr.fgPipeLogged = false;
    {
        char ev[8] = {};
        if (!(GetEnvironmentVariableA("SMV_FSRFG_PROF", ev, sizeof(ev)) > 0 && ev[0] == '1'))
        {
            bool ok = cudaStreamCreateWithFlags(&nr.fgL1St, cudaStreamNonBlocking) == cudaSuccess &&
                      cudaStreamCreateWithFlags(&nr.fgOfL1St, cudaStreamNonBlocking) == cudaSuccess;
            for (cudaEvent_t* e : {&nr.fgEvL1, &nr.fgEvS, &nr.fgEvDeep, &nr.fgEvOf, &nr.fgEvMv, &nr.fgEvSurf[0],
                                   &nr.fgEvSurf[1], &nr.fgEvSurf[2]})
                ok = ok && cudaEventCreateWithFlags(e, cudaEventDisableTiming) == cudaSuccess;
            if (!ok)
            {
                LOG("native: fsrfg: the level-1 streams did not start: the Optical Flow on the session's stream, no "
                    "pipelined route\n");
                nativeFsrfgChainFree(nr);
            }
        }
    }
    // never the fast perf level here: at fast, grid 1, the engine's vectors can go wrong after ~320 calls of a 1080p
    // session with vkd3d-proton in the process (garbage tweens from then on, run to run different)
    if (!nativeFsrfgCreate(nr, 0) || !nativeOfVecSetup(nr, nr.fgOf, nr.w, nr.h, "fsrfg", nr.fgOfL1St))
        return false;
    if (nr.fgL1St)
        NCHK(cudaMalloc((void**)&nr.dFgMvL1, (size_t)nr.fgMvPitch * nr.h), "alloc fsrfg level-1 vectors");
    const size_t bytes = (size_t)nr.fgPitch * nr.h, plane = (size_t)nr.ph * nr.pw;
    for (auto& s : nr.dFrSurf)
        NCHK(cudaMalloc((void**)&s, bytes), "alloc fsrfg frame");
    // the tween in the model layout; the pad outside w x h is never written, zero it once
    NCHK(cudaMalloc((void**)&nr.dFrOut, 3 * plane * sizeof(float)), "alloc fsrfg tween");
    NCHK(cudaMemsetAsync(nr.dFrOut, 0, 3 * plane * sizeof(float), nr.stream), "clear fsrfg tween");
    {
        char ev[8] = {};
        nr.frSdrOn =
            nr.encPost && !(GetEnvironmentVariableA("SMV_FSRFG_SDR_PAIRS", ev, sizeof(ev)) > 0 && ev[0] == '0');
    }
    if (nr.frSdrOn)
    {
        NCHK(cudaMalloc((void**)&nr.dFrRange, sizeof(int)), "alloc fsrfg range flag");
        NCHK(cudaHostAlloc((void**)&nr.hFrRange, sizeof(int), cudaHostAllocDefault), "alloc fsrfg range readback");
    }
    {
        char ev[8] = {};
        nr.fgProf = GetEnvironmentVariableA("SMV_FSRFG_PROF", ev, sizeof(ev)) > 0 && ev[0] == '1';
        for (cudaEvent_t& e : nr.fgEv)
            if (nr.fgProf && !e && cudaEventCreate(&e) != cudaSuccess)
                nr.fgProf = false;
        for (double& s : nr.fgStageMs)
            s = 0.0;
        nr.fgCpuMs = nr.fgDeriveMs = 0.0;
        nr.fgProfN = nr.fgProfVecN = nr.fgProfDerN = 0;
    }
    for (int i = 0; i < 3; i++)
        nr.frIn[i] = nr.frSurfSdr[i] = false;
    nr.frPairSdr = false;
    nr.frModeLogs = 0;
    nr.frPrev = nr.frLast = nr.frA = nr.frB = -1;
    nr.frInst = 1;
    nr.frInstMax = kFrDepth;
    {
        // FRUC's lever: SMV_FRUC_INSTANCES caps the instances (5 = the tree to x32)
        char ev[8] = {};
        if (GetEnvironmentVariableA("SMV_FRUC_INSTANCES", ev, sizeof(ev)) > 0)
            nr.frInstMax = (std::max)(1, (std::min)(kFrDepth, atoi(ev)));
    }
    nr.frPlanN = nr.frPlanK = 0;
    nr.frMp = true;
    nr.frMpCap = g_offline ? 3 : 2; // FRUC's snap depths: offline x8's nodes, live x4's
    nr.frMpL = 0;
    nr.frMpBuilt = false;
    nr.frSerial = 0;
    nr.frNodeCalls = nr.frTweens = nr.frPrimed = nr.frRepeats = 0;
    for (bool& ok : nr.frNodeOk)
        ok = false;
    if (nr.fgKind == 1)
        LOG("native: fsrfg session %dx%d, Intel %s frame generation through smv_xefg_bridge.dll on D3D12, RGBA16F, NVIDIA optical flow vectors (grid %d, %s; level-2+ calls %s), recursive midpoints (depth cap %d)\n",
            nr.w, nr.h, g_fsrfgK[1].provider(), nr.fgOf.grid, nativeOfLevelName(nr.fgOf.level),
            fsrfgDeriveOn(nr) ? "derived from level 1" : "a pass a call", nr.frMpCap);
    else
        LOG("native: fsrfg session %dx%d, AMD FSR %s frame generation (ML) through smv_fsrfg_bridge.dll on vkd3d-proton, RGBA16F, NVIDIA optical flow vectors (grid %d, %s; level-2+ calls %s), recursive midpoints (depth cap %d)\n",
            nr.w, nr.h, g_fsrfgK[0].provider(), nr.fgOf.grid, nativeOfLevelName(nr.fgOf.level),
            fsrfgDeriveOn(nr) ? "derived from level 1" : "a pass a call", nr.frMpCap);
    if (nr.encPost)
        LOG("native: fsrfg: HDR planes: %s\n",
            nr.frSdrOn ? "frame pairs inside the SDR range go to FSR as their SDR view, any other pair as the HDR codes"
                       : "every pair as the HDR codes (SMV_FSRFG_SDR_PAIRS=0)");
    return true;
}

// the new frame packed into a frame surface (never the last one, nor `keep`) and the pair advanced to it (frA / frB /
// frSerial, its nodes cleared); repacked = the last frame was packed again in the other encoding
static bool nativeFsrfgPairStep(NativeRife& nr, const float* dCur, uint32_t nTween, int keep, bool& repacked)
{
    int n = 0;
    while (n == nr.frPrev || n == keep)
        n++;
    repacked = false;
    // HDR planes: FRUC's rule (nativeFrucPair): the SDR view only when both frames stay inside the SDR range
    bool in = false;
    if (nr.frSdrOn)
    {
        cudaStream_t st = nr.stream;
        int ps = nr.ph * nr.pw;
        float tol = 2e-3f;
        void* ar[] = {(void*)&dCur, &ps, &nr.pw, &nr.w, &nr.h, &nr.encPost, &nr.encWhite, &tol, (void*)&nr.dFrRange};
        if (cudaMemsetAsync(nr.dFrRange, 0, sizeof(int), st) != cudaSuccess ||
            cuLaunchKernel(nr.fHdrRange, (nr.w + 15) / 16, (nr.h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, ar,
                           nullptr) != CUDA_SUCCESS ||
            cudaMemcpyAsync(nr.hFrRange, nr.dFrRange, sizeof(int), cudaMemcpyDeviceToHost, st) != cudaSuccess ||
            cudaStreamSynchronize(st) != cudaSuccess)
        {
            nr.die("fsrfg range test failed");
            return false;
        }
        in = *nr.hFrRange == 0;
    }
    const bool sdr = in && (nr.frPrev < 0 || nr.frIn[nr.frPrev]);
    // pipelined, the level-1 pass and instance 0's copies read the surfaces off the session's stream: a pack waits for
    // its surface's last such read (fgEvSurf), never for instance 0's call
    auto surfFree = [&nr](int k) {
        if (nr.fgPiped && cudaStreamWaitEvent(nr.stream, nr.fgEvSurf[k], 0) != cudaSuccess)
        {
            nr.die("fsrfg frame wait failed");
            return false;
        }
        return true;
    };
    if (!surfFree(n) || !nativeFsrfgPack(nr, dCur, n, sdr))
        return false;
    nr.frIn[n] = in;
    if (nr.frPrev >= 0 && nr.frSurfSdr[nr.frPrev] != sdr)
    {
        // the previous frame in the other encoding: packed again, and every fed instance restarts there (a key that
        // names no frame: a plain call, never a reset)
        if (!surfFree(nr.frPrev) || !nativeFsrfgPack(nr, nativeXPrev(nr), nr.frPrev, sdr))
            return false;
        for (uint64_t& k : nr.frLastKey)
            if (k)
                k = ~0ull;
        repacked = true;
    }
    if (nr.frSdrOn && nTween && nr.frPrev >= 0 && nr.frModeLogs < 4 && (nr.frModeLogs == 0 || sdr != nr.frPairSdr))
    {
        LOG("native: fsrfg: %s\n", sdr ? "frame pairs inside the SDR range, their SDR view"
                                       : "a frame pair outside the SDR range, the HDR codes");
        nr.frModeLogs++;
    }
    nr.frPairSdr = sdr;
    nr.frA = nr.frPrev;
    nr.frB = n;
    nr.frPrev = n;
    nr.frSerial++;
    nr.frMpL = 0;
    nr.frMpBuilt = false;
    for (bool& ok : nr.frNodeOk)
        ok = false;
    return true;
}

static bool nativeFsrfgPair(NativeRife& nr, const float* dCur, uint32_t nTween)
{
    bool repacked = false;
    if (!nativeFsrfgPairStep(nr, dCur, nTween, -1, repacked))
        return false;
    if (nr.frA < 0)
    {
        // the stream's first frame: instance 0 starts there with a reset
        if (!nativeFsrfgFeed(nr, 0, nr.dFrSurf[nr.frB], nullptr, true, nullptr))
            return false;
        nr.frLastKey[0] = nativeFrucKey(nativeFrucCur(nr), 1, 0);
        return true;
    }
    // every real frame goes to instance 0 in order, tweens or not (a gap would cost a re-prime): that call is the
    // pair's midpoint
    return nativeFsrfgNode(nr, 1, 1) != nullptr;
}

// the pipelined route (SMV_FSRFG_PIPE, nativeGroup's route bit) is open: the derived level-2+ vectors (a pair's
// deeper calls read only its own level-1 field), one field a pair (not SMV_OFVEC_BOTH), fgOf on fgL1St (not the
// profiler)
static bool fsrfgPipeOk(const NativeRife& nr)
{
    return nr.fgL1St && nr.dFgMvL1 && fsrfgDeriveOn(nr) && !nr.fgOf.both && !nr.fgProf;
}

// SMV_FSRFG_PIPE=serial (an A/B lever): the pipelined route with the new pair's Optical Flow too after the shown pair's
// deeper calls (its bookkeeping without its overlap)
static bool fsrfgPipeSerial()
{
    static const bool on = [] {
        char v[16] = "";
        return GetEnvironmentVariableA("SMV_FSRFG_PIPE", v, sizeof(v)) > 0 && strcmp(v, "serial") == 0;
    }();
    return on;
}

// the new pair's level-1 Optical Flow and instance 0's copies (nativeFsrfgFeed part 1): the pair's two surfaces are
// read off the session's stream until fgL1St's copies are done (fgEvSurf, the next packs wait for it)
static bool nativeFsrfgL1Vec(NativeRife& nr)
{
    if (!nativeFsrfgFeed(nr, 0, nr.dFrSurf[nr.frB], nr.dFrSurf[nr.frA], false, nullptr, -1, true, 1))
        return false;
    if (cudaEventRecord(nr.fgEvSurf[nr.frA], nr.fgL1St) != cudaSuccess ||
        cudaEventRecord(nr.fgEvSurf[nr.frB], nr.fgL1St) != cudaSuccess)
    {
        nr.die("fsrfg level-1 event failed");
        return false;
    }
    return true;
}

// a pipelined group's pair step: the pair the last group made current becomes the one shown (its level-1 node, made
// on fgL1St then, swapped into dFrNode), the new frame is packed sparing the shown pair's first frame (each pack waits
// only for its surface's last read off the session's stream, nativeFsrfgPairStep), and the new pair's level-1 chain
// starts: its Optical Flow on fgOfL1St into the other field now, beside instance 0's last call, the call itself on
// fgL1St after the shown pair's deeper calls (nativeFsrfgL1Issue). The shown pair is held (its second frame in every
// slot) when it was identical, when the last group was not pipelined (it showed that pair itself: the route's entry
// costs one source period's hold) or when its second frame was packed again in the other encoding
static bool nativeFsrfgPipeAdvance(NativeRife& nr, const float* dCur, bool curStatic)
{
    nr.frLagSerial = nr.frSerial;
    nr.frLagA = nr.frA;
    nr.frLagB = nr.frB;
    nr.frLagSdr = nr.frPairSdr;
    nr.frLagStatic = nr.frCurStatic;
    const bool had = nr.fgPiped && nr.frLagA >= 0 && nr.fgL1Ok;
    nr.fgOfRd = nr.fgOfWr;
    bool repacked = false;
    if (!nativeFsrfgPairStep(nr, dCur, curStatic ? 0 : 1, nr.frLagA, repacked))
        return false;
    nr.frLagHave = had && !repacked;
    if (nr.frLagHave)
    {
        std::swap(nr.dFrNode[kFrGrid / 2], nr.dFgL1Node);
        nr.frNodeOk[kFrGrid / 2] = true;
    }
    nr.frLagL = 0;
    nr.fgL1Ok = false;
    nr.frCurStatic = curStatic;
    nr.frPipePrev = true;
    nr.frPipeGroups++;
    nr.fgPiped = true;
    if (!nr.fgPipeLogged)
    {
        LOG("native: fsrfg: pipelined across pairs: the new pair's level-1 Optical Flow and call on their own streams "
            "beside the shown pair's level-2+ calls\n");
        nr.fgPipeLogged = true;
    }
    if (!nr.dFgL1Node && cudaMalloc((void**)&nr.dFgL1Node, (size_t)nr.fgPitch * nr.h) != cudaSuccess)
    {
        nr.dFgL1Node = nullptr;
        nr.die("fsrfg level-1 node alloc failed");
        return false;
    }
    // the level-1 pass and instance 0's stream after the pack
    if (cudaEventRecord(nr.fgEvS, nr.stream) != cudaSuccess ||
        cudaStreamWaitEvent(nr.fgOfL1St, nr.fgEvS, 0) != cudaSuccess ||
        cudaStreamWaitEvent(nr.fgL1St, nr.fgEvS, 0) != cudaSuccess)
    {
        nr.die("fsrfg level-1 event failed");
        return false;
    }
    const FrTree cur = nativeFrucCur(nr);
    if (nr.frA < 0)
    {
        // the stream's first frame: instance 0 starts there with a reset
        if (!nativeFsrfgFeed(nr, 0, nr.dFrSurf[nr.frB], nullptr, true, nullptr, -1, false, 3) ||
            cudaEventRecord(nr.fgEvL1, nr.fgL1St) != cudaSuccess ||
            cudaEventRecord(nr.fgEvSurf[nr.frB], nr.fgL1St) != cudaSuccess)
            return false;
        nr.frLastKey[0] = nativeFrucKey(cur, 1, 0);
        return true;
    }
    const uint64_t lk = nativeFrucKey(cur, 0, 1);
    if (nr.frLastKey[0] != lk)
    {
        // a re-prime (after a repack): the shown pair is held, so no deeper call runs beside this whole call
        nr.frLagHave = false;
        if (!nativeFsrfgFeed(nr, 0, nr.dFrSurf[nr.frA], nullptr, nr.frLastKey[0] == 0, nullptr, -1, false, 3))
            return false;
        if (nr.frLastKey[0] != 0)
            nr.frPrimed++;
        nr.frLastKey[0] = lk;
    }
    nr.fgOfWr = nr.fgOf.wr = 1 - nr.fgOfRd;
    nr.fgL1Pend = true;
    return fsrfgPipeSerial() || nativeFsrfgL1Vec(nr);
}

// the pipelined route's instance-0 call of the new pair (nativeFsrfgPipeAdvance made its vectors) on fgL1St after
// fgEvDeep = all the session's stream queued so far (the shown pair's deeper calls), so FSR's calls keep the serial
// route's order on the device; its node into dFgL1Node, fgEvL1 after it
static bool nativeFsrfgL1Issue(NativeRife& nr)
{
    if (!nr.fgL1Pend)
        return true;
    nr.fgL1Pend = false;
    if (cudaEventRecord(nr.fgEvDeep, nr.stream) != cudaSuccess)
    {
        nr.die("fsrfg level-1 event failed");
        return false;
    }
    if (fsrfgPipeSerial() && (cudaStreamWaitEvent(nr.fgOfL1St, nr.fgEvDeep, 0) != cudaSuccess || !nativeFsrfgL1Vec(nr)))
    {
        nr.die("fsrfg level-1 call failed");
        return false;
    }
    if (!nativeFsrfgFeed(nr, 0, nr.dFrSurf[nr.frB], nr.dFrSurf[nr.frA], false, nr.dFgL1Node, -1, true, 2))
        return false;
    if (cudaEventRecord(nr.fgEvL1, nr.fgL1St) != cudaSuccess)
    {
        nr.die("fsrfg level-1 event failed");
        return false;
    }
    nr.frLastKey[0] = nativeFrucKey(nativeFrucCur(nr), 2, 1);
    nr.frNodeCalls++;
    nr.fgL1Ok = true;
    return true;
}

// a pipelined group's tween: the shown pair's levels 2 to its depth (level by level, as nativeFsrfgTween) once its
// level-1 node is done on fgL1St, then the new pair's level-1 call (nativeFsrfgL1Issue), then the nearest node
static bool nativeFsrfgLagTween(NativeRife& nr, double t)
{
    if (!nr.frMpL)
    {
        nr.frMpL = (std::min)(nativeFrucDepth(nr, &t, 1), nr.frInst);
        if (cudaStreamWaitEvent(nr.stream, nr.fgEvL1, 0) != cudaSuccess)
        {
            nr.die("fsrfg level-1 wait failed");
            return false;
        }
        // the deeper calls' price starts here (nativeGroup): the wait above belongs to the level-1 chain
        if (nr.grLagEv)
            nr.grLagArmed = cudaEventRecord(nr.grLagEv, nr.stream) == cudaSuccess;
        for (int l = 2; l <= nr.frMpL; l++)
            if (nativeFsrfgAhead(nr))
            {
                if (!nativeFsrfgLevel(nr, l))
                    return false;
            }
            else
                for (int k = 1; k < (1 << l); k += 2)
                    if (!nativeFsrfgNode(nr, k, l))
                        return false;
        if (!nativeFsrfgL1Issue(nr))
            return false;
        nr.frMpBuilt = true;
    }
    const int den = 1 << nr.frMpL;
    const int k = (std::max)(1, (std::min)(den - 1, (int)floor(t * den + 0.5)));
    uint8_t* node = nativeFsrfgNode(nr, k, nr.frMpL);
    if (!node)
        return false;
    nr.frTweens++;
    int pitch = nr.fgPitch, sdr = nr.frLagSdr ? 1 : 0, ps = nr.ph * nr.pw;
    void* a[] = {&node, &pitch, &nr.w, &nr.h, &sdr, &nr.encPost, &nr.encWhite, &nr.dFrOut, &ps, &nr.pw};
    if (cuLaunchKernel(nr.fUnpackRgbaH, (nr.w + 15) / 16, (nr.h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)nr.stream, a,
                       nullptr) != CUDA_SUCCESS)
    {
        nr.die("unpackRgbaH (fsrfg) launch failed");
        return false;
    }
    return true;
}

// the direct route after a pipelined group: the session's stream waits for fgL1St's last work (the level-1 call of
// the pair the last group made current, which this route never shows; it waited for fgOfL1St's last pass), one field
// again
static bool nativeFsrfgPipeExit(NativeRife& nr)
{
    if (!nr.fgPiped)
        return true;
    nr.fgPiped = nr.fgL1Ok = nr.fgL1Pend = false;
    nr.fgOfWr = nr.fgOfRd = nr.fgOf.wr = 1;
    if (cudaStreamWaitEvent(nr.stream, nr.fgEvL1, 0) != cudaSuccess)
    {
        nr.die("fsrfg level-1 wait failed");
        return false;
    }
    return true;
}

// one tween at t into dFrOut: the pair's whole tree to its depth first, level by level (FRUC's order: level L's
// instance sees the continuous stream A, 2 / 2^L, .. B), then the nearest node
static bool nativeFsrfgTween(NativeRife& nr, double t)
{
    if (!nr.frMpL)
        nr.frMpL = nativeFrucDepth(nr, &t, 1);
    if (nr.frInst < nr.frMpL && nr.frInst < nr.frInstMax)
        nativeFsrfgGrow(nr, (std::min)(nr.frMpL, nr.frInstMax));
    if (nr.frInst < nr.frMpL)
        nr.frMpL = nr.frInst;
    if (!nr.frMpBuilt)
    {
        for (int l = 1; l <= nr.frMpL; l++)
            if (l >= 2 && nativeFsrfgAhead(nr))
            {
                if (!nativeFsrfgLevel(nr, l))
                    return false;
            }
            else
                for (int k = 1; k < (1 << l); k += 2)
                    if (!nativeFsrfgNode(nr, k, l))
                        return false;
        nr.frMpBuilt = true;
    }
    const int den = 1 << nr.frMpL;
    const int k = (std::max)(1, (std::min)(den - 1, (int)floor(t * den + 0.5)));
    uint8_t* node = nativeFsrfgNode(nr, k, nr.frMpL);
    if (!node)
        return false;
    nr.frTweens++;
    int pitch = nr.fgPitch, sdr = nr.frPairSdr ? 1 : 0, ps = nr.ph * nr.pw;
    void* a[] = {&node, &pitch, &nr.w, &nr.h, &sdr, &nr.encPost, &nr.encWhite, &nr.dFrOut, &ps, &nr.pw};
    if (cuLaunchKernel(nr.fUnpackRgbaH, (nr.w + 15) / 16, (nr.h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)nr.stream, a,
                       nullptr) != CUDA_SUCCESS)
    {
        nr.die("unpackRgbaH (fsrfg) launch failed");
        return false;
    }
    return true;
}

// the instances' CUDA imports, then the bridge's instances, then the Optical Flow session (nativeFrucFree frees the
// frames after this)
static void nativeFsrfgRelease(NativeRife& nr)
{
    for (cudaStream_t s : {nr.fgOfL1St, nr.fgL1St})
        if (s)
            cudaStreamSynchronize(s);
    if (nr.stream)
        cudaStreamSynchronize(nr.stream);
    if (nr.fg[0].made)
        LOG("native: fsrfg session: %llu tweens, %llu midpoint calls, %llu re-primed streams%s, %d instance%s\n",
            (unsigned long long)nr.frTweens, (unsigned long long)nr.frNodeCalls, (unsigned long long)nr.frPrimed,
            nr.frPipeGroups ? (", " + std::to_string(nr.frPipeGroups) + " groups pipelined").c_str() : "", nr.frInst,
            nr.frInst > 1 ? "s" : "");
    // XeSS-FG's own counts per instance: a call with no frame presented keeps the previous output (none expected after
    // a sequence's first call)
    if (nr.fgKind == 1 && g_fsrfgK[1].stats)
        for (int i = 0; i < nr.frInst && i < (int)(sizeof(nr.fg) / sizeof(nr.fg[0])); i++)
        {
            uint64_t c = 0, nf = 0, ex = 0, cap = 0;
            if (nr.fg[i].made && g_fsrfgK[1].stats(i, &c, &nf, &ex, &cap) == 0)
                LOG("native: fsrfg: XeSS-FG instance %d: %llu calls, %llu frames out, %llu calls without a frame, %llu "
                    "presents without a call\n",
                    i, (unsigned long long)c, (unsigned long long)cap, (unsigned long long)nf, (unsigned long long)ex);
        }
    if (nr.fgProfN)
    {
        const double n = (double)nr.fgProfN, nv = nr.fgProfVecN ? (double)nr.fgProfVecN : 1.0,
                     nd = nr.fgProfDerN ? (double)nr.fgProfDerN : 1.0;
        double sum = nr.fgDeriveMs;
        for (double s : nr.fgStageMs)
            sum += s;
        LOG("[fsrfg-prof] %llu calls (%llu with an Optical Flow pass, %llu derived) %dx%d, ms a call: copy in %.3f, "
            "Optical Flow inputs %.3f, Optical Flow %.3f, vector kernels %.3f (the three a pass call), derived vectors "
            "%.3f (a derived call), vector copy %.3f, FSR on vkd3d %.3f, copy out %.3f, all %.3f; the bridge's CPU "
            "call %.3f\n",
            (unsigned long long)nr.fgProfN, (unsigned long long)nr.fgProfVecN, (unsigned long long)nr.fgProfDerN, nr.w,
            nr.h, nr.fgStageMs[0] / n, nr.fgStageMs[1] / nv, nr.fgStageMs[2] / nv, nr.fgStageMs[3] / nv,
            nr.fgDeriveMs / nd, nr.fgStageMs[4] / n, nr.fgStageMs[5] / n, nr.fgStageMs[6] / n, sum / n, nr.fgCpuMs / n);
    }
    for (cudaEvent_t& e : nr.fgEv)
        if (e)
        {
            cudaEventDestroy(e);
            e = nullptr;
        }
    nr.fgProfN = 0;
    bool any = false;
    for (NativeRife::FgInst& g : nr.fg)
    {
        for (cudaMipmappedArray_t* mm : {&g.mmIn, &g.mmMv, &g.mmOut})
            if (*mm)
            {
                cudaFreeMipmappedArray(*mm);
                *mm = nullptr;
            }
        if (g.sem)
            cudaDestroyExternalSemaphore(g.sem);
        for (cudaExternalMemory_t* em : {&g.emIn, &g.emMv, &g.emOut})
            if (*em)
            {
                cudaDestroyExternalMemory(*em);
                *em = nullptr;
            }
        any = any || g.made;
        g = NativeRife::FgInst{};
    }
    if (any && g_fsrfgK[nr.fgKind].destroy)
        g_fsrfgK[nr.fgKind].destroy(); // every instance
    if (nr.dFgMv)
    {
        cudaFree(nr.dFgMv);
        nr.dFgMv = nullptr;
    }
    nativeFsrfgAheadFree(nr);
    nativeOfVecFree(nr.fgOf);
    for (uint8_t** b : {&nr.dFgMvL1, &nr.dFgL1Node})
        if (*b)
        {
            cudaFree(*b);
            *b = nullptr;
        }
    nativeFsrfgChainFree(nr);
    nr.fgPiped = nr.fgL1Pend = nr.fgL1Ok = false;
    nr.frInst = 1;
}

// ---- AMD FSR / Intel XeSS-SR upscaling (fsrUp) ---------------------------------------------------
// One C API, two bridges (fuKind): smv_fsrup_bridge.dll (source engine\fsrup\build_src, exports fsrup_*; SMV_FSRUP_DIR,
// else engine\fsrup, AMD's signed loader + upscaler beside it) and smv_xess_bridge.dll (source engine\xess\build_src,
// exports xessup_*; SMV_XESS_DIR, else engine\xess, Intel's libxess.dll beside it). Each loads once per process by full
// path from its folder
struct FsrupShared
{
    HANDLE in, mv, out, fence;
    uint64_t inBytes, mvBytes, outBytes;
    uint32_t inPitch, mvPitch, outPitch;
    char version[32];
};
struct FsrupBridge
{
    HMODULE mod = nullptr;
    const char* (*lastError)() = nullptr;
    int (*create)(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, int32_t, const char*, int, FsrupShared*) = nullptr;
    int (*frame)(uint64_t, uint64_t, int, double) = nullptr;
    void (*destroy)() = nullptr;
    // instance k of the same (the chained passes past 3x, fuChainPlan); an older bridge lacks them
    int (*createI)(int, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, int32_t, const char*, int,
                   FsrupShared*) = nullptr;
    int (*frameI)(int, uint64_t, uint64_t, int, double) = nullptr;
    void (*destroyI)(int) = nullptr;
};
static FsrupBridge g_fsrupK[2];
static const char* const kFuDll[2] = {"smv_fsrup_bridge.dll", "smv_xess_bridge.dll"};
static const char* const kFuPrefix[2] = {"fsrup_", "xessup_"};
static const char* const kFuName[2] = {"AMD FSR", "Intel"};

static bool nativeFsrupLoad(const std::string& dir, int kind)
{
    FsrupBridge& b = g_fsrupK[kind];
    if (b.mod)
        return true;
    const std::wstring p = utf8ToWide(dir) + L"\\" + utf8ToWide(kFuDll[kind]);
    HMODULE m = LoadLibraryExW(p.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!m)
    {
        LOG("native: upscale bridge: %s did not load from %s (err %lu)\n", kFuDll[kind], dir.c_str(), GetLastError());
        return false;
    }
    const std::string pre = kFuPrefix[kind];
    b.lastError = (const char* (*)())GetProcAddress(m, (pre + "last_error").c_str());
    b.create = (int (*)(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, int32_t, const char*, int,
                        FsrupShared*))GetProcAddress(m, (pre + "create").c_str());
    b.frame = (int (*)(uint64_t, uint64_t, int, double))GetProcAddress(m, (pre + "frame").c_str());
    b.destroy = (void (*)())GetProcAddress(m, (pre + "destroy").c_str());
    if (!b.lastError || !b.create || !b.frame || !b.destroy)
    {
        LOG("native: upscale bridge: %s lacks an expected export\n", kFuDll[kind]);
        FreeLibrary(m);
        b = FsrupBridge{};
        return false;
    }
    b.createI = (int (*)(int, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, int32_t, const char*, int,
                         FsrupShared*))GetProcAddress(m, (pre + "create_i").c_str());
    b.frameI = (int (*)(int, uint64_t, uint64_t, int, double))GetProcAddress(m, (pre + "frame_i").c_str());
    b.destroyI = (void (*)(int))GetProcAddress(m, (pre + "destroy_i").c_str());
    b.mod = m;
    return true;
}

// the reason an early (handoff-time) setup failed, named by the pipe's "the early engine handoff failed" line (cleared
// when the early thread starts)
static char g_earlyWhy[240];
static void earlyWhy(const char* fmt, ...)
{
    va_list a;
    va_start(a, fmt);
    vsnprintf(g_earlyWhy, sizeof(g_earlyWhy), fmt, a);
    va_end(a);
}

// AMD FSR / Intel XeSS-SR past 3x a side (FSR's named presets end at 3.0x and past about 9x it leaves one dot an input
// pixel; XeSS-SR refuses past 3.0x): the fewest equal passes of at most 3x, the last landing on ow x oh, the sizes in
// between even (the upscaler leaves an odd last column / row untouched). pw / ph [0 .. n] = the sizes; returns n (1 =
// one call: within 3x, or a shrink), 0 = no chain within kFuMaxPass. SMV_FU_CHAIN=0: one call at any ratio (A/B)
static int fuChainPlan(int w, int h, int ow, int oh, int* pw, int* ph)
{
    pw[0] = w;
    ph[0] = h;
    pw[1] = ow;
    ph[1] = oh;
    static const bool off = [] {
        char v[8] = {};
        return GetEnvironmentVariableA("SMV_FU_CHAIN", v, sizeof(v)) > 0 && v[0] == '0';
    }();
    if (off || ow < w || oh < h || ((double)ow <= 3.0 * w && (double)oh <= 3.0 * h))
        return 1;
    const double rx = (double)ow / w, ry = (double)oh / h;
    const int n0 = (std::max)(2, (int)std::ceil(std::log((std::max)(rx, ry)) / std::log(3.0) - 1e-9));
    for (int n = n0; n <= NativeRife::kFuMaxPass; n++)
    {
        pw[n] = ow;
        ph[n] = oh;
        for (int k = 1; k < n; k++)
        {
            pw[k] = (std::min)(ow, ((int)std::ceil(w * std::pow(rx, (double)k / n)) + 1) & ~1);
            ph[k] = (std::min)(oh, ((int)std::ceil(h * std::pow(ry, (double)k / n)) + 1) & ~1);
        }
        for (int k = n - 1; k >= 1; k--) // the step into k + 1 at most 3x a side
        {
            pw[k] = (std::max)(pw[k], ((pw[k + 1] + 2) / 3 + 1) & ~1);
            ph[k] = (std::max)(ph[k], ((ph[k + 1] + 2) / 3 + 1) & ~1);
        }
        bool ok = true;
        for (int k = 1; k <= n; k++)
            ok = ok && pw[k] >= pw[k - 1] && ph[k] >= ph[k - 1] && pw[k] <= 3 * pw[k - 1] && ph[k] <= 3 * ph[k - 1];
        if (ok)
            return n;
    }
    return 0;
}

// the upscaler at RTX VSR's resize (w x h -> ow x oh, nativeRtxInit's slot; past 3x a side in fuChainPlan's passes),
// its shared buffers and fences imported, and the vectors' Optical Flow session at the render size
static bool nativeFsrupSetup(NativeRife& nr, int w, int h, int ow, int oh)
{
    if (!nativeFsrupLoad(nr.fsrUpDir, nr.fuKind))
        return false;
    FsrupBridge& br = g_fsrupK[nr.fuKind];
    const int n = fuChainPlan(w, h, ow, oh, nr.fuPw, nr.fuPh);
    if (!n || (n > 1 && (!br.createI || !br.frameI || !br.destroyI)))
    {
        earlyWhy("%s upscaling %dx%d -> %dx%d: %s", kFuName[nr.fuKind], w, h, ow, oh,
                 n ? "its bridge has no chained passes (an older one)" : "no chain of passes of at most 3x reaches it");
        LOG("native: fsr upscale: %s\n", g_earlyWhy);
        return false;
    }
    nr.fuPasses = n;
    int dev = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess)
    {
        LOG("native: fsr upscale: the CUDA device's properties are unavailable\n");
        return false;
    }
    LUID luid;
    memcpy(&luid, prop.luid, sizeof(luid));
    auto imp = [](HANDLE hd, uint64_t bytes, cudaExternalMemory_t& em, uint8_t*& p) {
        cudaExternalMemoryHandleDesc md{};
        md.type = cudaExternalMemoryHandleTypeD3D12Resource;
        md.size = bytes;
        md.flags = cudaExternalMemoryDedicated;
        md.handle.win32.handle = hd;
        if (cudaImportExternalMemory(&em, &md) != cudaSuccess)
        {
            em = nullptr;
            return false;
        }
        cudaExternalMemoryBufferDesc bd{};
        bd.size = bytes;
        return cudaExternalMemoryGetMappedBuffer((void**)&p, em, &bd) == cudaSuccess;
    };
    // pass k on bridge instance k (0 = the plain exports), its shared buffers and fence into the given fields
    auto make = [&](int k, cudaExternalMemory_t& emIn, uint8_t*& dIn, int& inP, cudaExternalMemory_t& emMv,
                    uint8_t*& dMv, int& mvP, cudaExternalMemory_t& emOut, uint8_t*& dOut, int& outP,
                    cudaExternalSemaphore_t& sem, bool& made) {
        const int pw = nr.fuPw[k], ph = nr.fuPh[k], qw = nr.fuPw[k + 1], qh = nr.fuPh[k + 1];
        FsrupShared sh{};
        // live HDR planes: the bridge's HDR mode (FP16 linear light in and out, k_fsrHdrIn / k_fsrHdrOut)
        const int rc = k ? br.createI(k, (uint32_t)pw, (uint32_t)ph, (uint32_t)qw, (uint32_t)qh, luid.LowPart,
                                      luid.HighPart, "4.", nr.fuHdr ? 1 : 0, &sh)
                         : br.create((uint32_t)pw, (uint32_t)ph, (uint32_t)qw, (uint32_t)qh, luid.LowPart,
                                     luid.HighPart, "4.", nr.fuHdr ? 1 : 0, &sh);
        if (rc != 0)
        {
            earlyWhy("%s upscaling %dx%d -> %dx%d not created: %s (rc %d)", kFuName[nr.fuKind], pw, ph, qw, qh,
                     br.lastError(), rc);
            LOG("native: fsr upscale: %dx%d -> %dx%d not created: %s (rc %d)\n", pw, ph, qw, qh, br.lastError(), rc);
            return false;
        }
        made = true;
        cudaExternalSemaphoreHandleDesc sd{};
        sd.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
        sd.handle.win32.handle = sh.fence;
        if (!imp(sh.in, sh.inBytes, emIn, dIn) || !imp(sh.mv, sh.mvBytes, emMv, dMv) ||
            !imp(sh.out, sh.outBytes, emOut, dOut) || cudaImportExternalSemaphore(&sem, &sd) != cudaSuccess)
        {
            LOG("native: fsr upscale: the shared buffers or fence did not import into CUDA\n");
            return false;
        }
        inP = (int)sh.inPitch;
        mvP = (int)sh.mvPitch;
        outP = (int)sh.outPitch;
        if (!k)
        {
            memcpy(nr.fuVersion, sh.version, sizeof(nr.fuVersion));
            nr.fuVersion[sizeof(nr.fuVersion) - 1] = 0;
        }
        return true;
    };
    const bool chain = n > 1;
    if (!make(0, nr.fuEmIn, nr.dFuIn, nr.fuInPitch, nr.fuEmMv, nr.dFuMv, nr.fuMvPitch, chain ? nr.fu0EmOut : nr.fuEmOut,
              chain ? nr.dFu0Out : nr.dFuOut, chain ? nr.fu0OutPitch : nr.fuOutPitch, nr.fuSem, nr.fuMade))
        return false;
    for (int k = 1; k < n; k++)
    {
        NativeRife::FuPass& p = nr.fuX[k - 1];
        const bool last = k == n - 1;
        p.val = 0;
        if (!make(k, p.emIn, p.dIn, p.inPitch, p.emMv, p.dMv, p.mvPitch, last ? nr.fuEmOut : p.emOut,
                  last ? nr.dFuOut : p.dOut, last ? nr.fuOutPitch : p.outPitch, p.sem, p.made))
            return false;
    }
    nr.fuVal = 0;
    nr.fuN = 0;
    nr.fuCur = 0;
    {
        char ev[8] = {};
        nr.fuProf = GetEnvironmentVariableA("SMV_FSRUP_PROF", ev, sizeof(ev)) > 0 && ev[0] == '1';
        for (cudaEvent_t& e : nr.fuEv)
            if (nr.fuProf && !e && cudaEventCreate(&e) != cudaSuccess)
                nr.fuProf = false;
        for (double& s : nr.fuStageMs)
            s = 0.0;
        nr.fuProfN = nr.fuProfVecN = 0;
    }
    if (!nativeOfVecSetup(nr, nr.fuOf, w, h, "fsr upscale"))
        return false;
    char via[160] = "";
    if (chain)
    {
        int o = snprintf(via, sizeof(via), " in %d passes (", n);
        for (int k = 1; k < n && o > 0 && o < (int)sizeof(via); k++)
            o += snprintf(via + o, sizeof(via) - o, "%s%dx%d", k > 1 ? ", " : "", nr.fuPw[k], nr.fuPh[k]);
        if (o > 0 && o < (int)sizeof(via))
            snprintf(via + o, sizeof(via) - o, " between)");
    }
    LOG("native: upscale: %s %s %dx%d -> %dx%d%s through %s, NVIDIA optical flow vectors (grid %d, %s)%s\n",
        kFuName[nr.fuKind], nr.fuVersion, w, h, ow, oh, via, kFuDll[nr.fuKind], nr.fuOf.grid,
        nativeOfLevelName(nr.fuOf.level), nr.fuHdr ? ", on the HDR planes' linear light (the bridge's HDR mode)" : "");
    return true;
}

// the tick the stall watchdog (PipeServer::wdogProc) waits for. The driver compiles FSR's shaders inside a session's first
// dispatch, per exe location and upscale ratio (47 s measured, a warm-up covers its own ratio only), and that call blocks this
// thread; a real hang there still ends at kFsrCompileMs, and the watchdog's 10 s start over when it returns
static volatile LONGLONG g_wdogHold = 0;
static constexpr int kFsrCompileMs = 180000;

// one frame of the slot's stream: nr.dVsrIn (tight BGRA8 ri) upscaled into nr.dVsrOut (tight BGRA8 ro), queued on st;
// the frame into the bridge's buffer and the vector session's slot, the vectors from the previous frame's slot (the
// stream's first frame: zero vectors and a reset), the bridge's queue waits on the stream and the stream on the bridge
static bool nativeFsrupEval(NativeRife& nr, const RtxRect& ri, const RtxRect& ro, cudaStream_t st, unsigned int& rv)
{
    int w = (int)ri.right, h = (int)ri.bottom, ow = (int)ro.right, oh = (int)ro.bottom;
    const int cur = nr.fuCur, prev = cur ^ 1;
    int fip = nr.fuInPitch, oip = (int)nr.fuOf.inPitch;
    CUdeviceptr ofIn = nr.fuOf.inP[cur];
    auto mark = [&nr, st](int e) {
        if (nr.fuProf)
            cudaEventRecord(nr.fuEv[e], st);
    };
    mark(0);
    // HDR planes (fuHdr): the callers packed the frame in linear light into dFuIn (k_fsrHdrIn) and take the result back
    // themselves (k_fsrHdrOut); dVsrIn (the frame's SDR view) feeds the vectors only
    uint8_t* in8 = nr.fuHdr ? nullptr : nr.dFuIn;
    void* a[] = {&nr.dVsrIn, &w, &h, &in8, &fip, &ofIn, &oip};
    if (cuLaunchKernel(nr.fBgraRgbaFsr, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a, nullptr) !=
        CUDA_SUCCESS)
    {
        nr.die("bgraRgbaFsr launch failed");
        return false;
    }
    mark(1);
    // The vectors between consecutive emitted frames are mostly near zero on still content (a tween step moves a
    // fraction of a pixel; the flow's noise is a few 1/32 px), and FSR resamples its history by any nonzero vector:
    // still detail then drifts and flips by whole output pixels while exact zeros hold it. So a vector shorter than
    // kFuVecGate px is written as exactly zero; real motion is far above it (a 1 px / frame pan tracks to 0.02 px),
    // and FSR follows a sub-gate tween step no better with the vector than without. SMV_FSRUP_TEST=gateN = N
    // thousandths of a px (gate0 = off, the A/B); =zerovec = zero vectors every frame (A/B only)
    static constexpr float kFuVecGate = 0.125f;
    static const float gate = [] {
        wchar_t v[16]{};
        if (!GetEnvironmentVariableW(L"SMV_FSRUP_TEST", v, 16) || wcsncmp(v, L"gate", 4))
            return kFuVecGate;
        return (float)_wtoi(v + 4) / 1000.0f;
    }();
    static const bool zeroVec = [] {
        wchar_t v[16]{};
        return GetEnvironmentVariableW(L"SMV_FSRUP_TEST", v, 16) && !wcscmp(v, L"zerovec");
    }();
    const bool first = nr.fuN == 0;
    if (first || zeroVec)
    {
        if (cudaMemset2DAsync(nr.dFuMv, nr.fuMvPitch, 0, (size_t)w * 4, h, st) != cudaSuccess)
        {
            nr.die("fsr upscale vector clear failed");
            return false;
        }
        mark(2);
    }
    else if (!nativeOfVecRun(nr, nr.fuOf, prev, cur, nr.dFuMv, nr.fuMvPitch, "fsr upscale",
                             nr.fuProf ? nr.fuEv[2] : nullptr, gate))
        return false;
    mark(3);
    cudaExternalSemaphoreSignalParams sp{};
    sp.params.fence.value = ++nr.fuVal;
    if (cudaSignalExternalSemaphoresAsync(&nr.fuSem, &sp, 1, st) != cudaSuccess)
    {
        nr.die("fsr upscale fence signal failed");
        return false;
    }
    const uint64_t ready = nr.fuVal, done = ++nr.fuVal;
    // SMV_FSRUP_TEST=hang1 / hang2: that frame's call blocks past the watchdog's bound, the watchdog's trigger
    static const int testHang = [] {
        wchar_t v[8]{};
        return GetEnvironmentVariableW(L"SMV_FSRUP_TEST", v, 8) && wcsncmp(v, L"hang", 4) == 0 ? _wtoi(v + 4) : 0;
    }();
    if (first)
        InterlockedExchange64(&g_wdogHold, (LONGLONG)GetTickCount64() + kFsrCompileMs);
    if (testHang && nr.fuN + 1 == (uint64_t)testHang)
        Sleep(kFsrCompileMs + 20000);
    // FSR 4.1.1 gives byte-identical frames at any frame time (measured, harness fsr4_int8_upscaler): one 60 Hz frame
    const int rc = g_fsrupK[nr.fuKind].frame(ready, done, first ? 1 : 0, 1000.0 / 60.0);
    if (first)
        InterlockedExchange64(&g_wdogHold, (LONGLONG)GetTickCount64() + 10000);
    if (rc != 0)
    {
        LOG("native: fsr upscale: frame %llu failed: %s (rc %d)\n", (unsigned long long)nr.fuN,
            g_fsrupK[nr.fuKind].lastError(), rc);
        nr.die("fsr upscale failed");
        return false;
    }
    cudaExternalSemaphoreWaitParams wp{};
    wp.params.fence.value = done;
    int op = nr.fuOutPitch, otp = ow * 4;
    void* b[] = {&nr.dFuOut, &op, &ow, &oh, &nr.dVsrOut, &otp};
    if (cudaWaitExternalSemaphoresAsync(&nr.fuSem, &wp, 1, st) != cudaSuccess)
    {
        nr.die("fsr upscale result failed");
        return false;
    }
    // past 3x (fuChainPlan): passes 1.. on their own instances, each on the previous pass's output with pass 0's
    // vectors scaled to its size (the stream's first frame: zero vectors and a reset, as pass 0)
    for (int k = 1; k < nr.fuPasses; k++)
    {
        NativeRife::FuPass& p = nr.fuX[k - 1];
        int kw = nr.fuPw[k], kh = nr.fuPh[k], mvSrcPitch = nr.fuMvPitch, dp = p.mvPitch;
        float kx = (float)kw / (float)w, ky = (float)kh / (float)h;
        const uint8_t* src = k == 1 ? nr.dFu0Out : nr.fuX[k - 2].dOut;
        const size_t srcPitch = (size_t)(k == 1 ? nr.fu0OutPitch : nr.fuX[k - 2].outPitch);
        void* v[] = {&nr.dFuMv, &mvSrcPitch, &w, &h, &p.dMv, &dp, &kw, &kh, &kx, &ky};
        const bool vecOk = first || zeroVec
                               ? cudaMemset2DAsync(p.dMv, p.mvPitch, 0, (size_t)kw * 4, kh, st) == cudaSuccess
                               : cuLaunchKernel(nr.fFuMvScale, (kw + 15) / 16, (kh + 15) / 16, 1, 16, 16, 1, 0,
                                                (CUstream)st, v, nullptr) == CUDA_SUCCESS;
        if (cudaMemcpy2DAsync(p.dIn, p.inPitch, src, srcPitch, (size_t)kw * (nr.fuHdr ? 8 : 4), kh,
                              cudaMemcpyDeviceToDevice, st) != cudaSuccess ||
            !vecOk)
        {
            nr.die("fsr upscale pass input failed");
            return false;
        }
        cudaExternalSemaphoreSignalParams ps{};
        ps.params.fence.value = ++p.val;
        if (cudaSignalExternalSemaphoresAsync(&p.sem, &ps, 1, st) != cudaSuccess)
        {
            nr.die("fsr upscale fence signal failed");
            return false;
        }
        const uint64_t pReady = p.val, pDone = ++p.val;
        if (first)
            InterlockedExchange64(&g_wdogHold, (LONGLONG)GetTickCount64() + kFsrCompileMs);
        const int prc = g_fsrupK[nr.fuKind].frameI(k, pReady, pDone, first ? 1 : 0, 1000.0 / 60.0);
        if (first)
            InterlockedExchange64(&g_wdogHold, (LONGLONG)GetTickCount64() + 10000);
        if (prc != 0)
        {
            LOG("native: fsr upscale: pass %d, frame %llu failed: %s (rc %d)\n", k + 1, (unsigned long long)nr.fuN,
                g_fsrupK[nr.fuKind].lastError(), prc);
            nr.die("fsr upscale failed");
            return false;
        }
        cudaExternalSemaphoreWaitParams pw{};
        pw.params.fence.value = pDone;
        if (cudaWaitExternalSemaphoresAsync(&p.sem, &pw, 1, st) != cudaSuccess)
        {
            nr.die("fsr upscale result failed");
            return false;
        }
    }
    mark(4);
    if (!nr.fuHdr)
    {
        if (cuLaunchKernel(nr.fRgbaBgra, (ow + 15) / 16, (oh + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, b, nullptr) !=
            CUDA_SUCCESS)
        {
            nr.die("fsr upscale result failed");
            return false;
        }
        void* e[] = {&nr.dVsrOut, &ow, &oh};
        if (((ow | oh) & 1) && cuLaunchKernel(nr.fOddEdge, (ow + 15) / 16, (oh + 15) / 16, 1, 16, 16, 1, 0,
                                              (CUstream)st, e, nullptr) != CUDA_SUCCESS)
        {
            nr.die("fsr upscale edge failed");
            return false;
        }
    }
    mark(5);
    if (nr.fuProf && cudaEventSynchronize(nr.fuEv[5]) == cudaSuccess)
    {
        for (int s = 0; s < 5; s++)
        {
            float ms = 0.0f;
            if (cudaEventElapsedTime(&ms, nr.fuEv[s], nr.fuEv[s + 1]) == cudaSuccess)
                nr.fuStageMs[s] += ms;
        }
        nr.fuProfN++;
        nr.fuProfVecN += first ? 0 : 1;
    }
    nr.fuCur = prev;
    nr.fuN++;
    rv = 1u;
    return true;
}

// nativeUpSwitch's warm-up (fuWarm): two black frames with zero vectors through every pass, the first a reset, on its
// own stream, as the app's fsrWarm session runs them. The native thread touches none of these fields while fuWarm is 1
// (every per-frame reader waits for nr.vsr, set by nativeUpWarmEnd); the session's first real frame resets again (fuN 0)
static DWORD WINAPI nativeFsrupWarmProc(void* p)
{
    NativeRife& nr = *(NativeRife*)p;
    FsrupBridge& br = g_fsrupK[nr.fuKind];
    const int bpp = nr.fuHdr ? 8 : 4;
    cudaStream_t st = nullptr;
    bool ok = nativeBindDevice(nr) && cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking) == cudaSuccess;
    InterlockedExchange64(&g_wdogHold, (LONGLONG)GetTickCount64() + kFsrCompileMs);
    for (int f = 0; f < 2 && ok; f++)
        for (int k = 0; k < nr.fuPasses && ok; k++)
        {
            NativeRife::FuPass* x = k ? &nr.fuX[k - 1] : nullptr;
            uint8_t* in = k ? x->dIn : nr.dFuIn;
            uint8_t* mv = k ? x->dMv : nr.dFuMv;
            const size_t inP = (size_t)(k ? x->inPitch : nr.fuInPitch), mvP = (size_t)(k ? x->mvPitch : nr.fuMvPitch);
            cudaExternalSemaphore_t sem = k ? x->sem : nr.fuSem;
            uint64_t& val = k ? x->val : nr.fuVal;
            const int kw = nr.fuPw[k], kh = nr.fuPh[k];
            cudaExternalSemaphoreSignalParams sp{};
            sp.params.fence.value = ++val;
            const uint64_t ready = val, done = ++val;
            cudaExternalSemaphoreWaitParams wp{};
            wp.params.fence.value = done;
            const int reset = f ? 0 : 1;
            ok = cudaMemset2DAsync(in, inP, 0, (size_t)kw * bpp, kh, st) == cudaSuccess &&
                 cudaMemset2DAsync(mv, mvP, 0, (size_t)kw * 4, kh, st) == cudaSuccess &&
                 cudaSignalExternalSemaphoresAsync(&sem, &sp, 1, st) == cudaSuccess &&
                 (k ? br.frameI(k, ready, done, reset, 1000.0 / 60.0) : br.frame(ready, done, reset, 1000.0 / 60.0)) ==
                     0 &&
                 cudaWaitExternalSemaphoresAsync(&sem, &wp, 1, st) == cudaSuccess;
            if (!ok)
                LOG("native: fsr upscale: warm-up frame %d, pass %d failed: %s\n", f + 1, k + 1, br.lastError());
        }
    ok = ok && cudaStreamSynchronize(st) == cudaSuccess;
    if (st)
        cudaStreamDestroy(st);
    InterlockedExchange64(&g_wdogHold, (LONGLONG)GetTickCount64() + 10000);
    InterlockedExchange(&nr.fuWarm, ok ? 2 : 3);
    return 0;
}

// the warm-up's end (nativeLiveSets, on the native thread): warm = the upscaler on, as nativeUpSwitch turns the other
// kinds on; an `up 0` meanwhile = built and off (an `up N` of its kind turns it on); failed = Lanczos stays, and the
// built bridge stays with the session (nativeUpSwitch refuses another start while fuMade)
static void nativeUpWarmEnd(NativeRife& nr)
{
    WaitForSingleObject(nr.fuWarmTh, INFINITE);
    CloseHandle(nr.fuWarmTh);
    nr.fuWarmTh = nullptr;
    const bool ok = InterlockedExchange(&nr.fuWarm, 0) == 2;
    const char* name = nr.fuKind ? "Intel XeSS upscaling" : "AMD FSR upscaling";
    const double ms = (double)(GetTickCount64() - nr.fuWarmT0);
    if (!ok)
    {
        LOG("native: live upscale: %s's first frames failed after %.0f ms, Lanczos stays\n", name, ms);
        return;
    }
    const int tw = nr.uw ? nr.uw : nr.dw, th = nr.uw ? nr.uh : nr.dh;
    nr.vsrWant = nr.fsrUp = nr.vsr = true;
    nr.vsrFailed = nr.vsrUserOff = !nr.fuWarmWant;
    LOG("native: live upscale: %s %dx%d -> %dx%d, started in place (its first frames on a worker: %.0f ms)%s\n", name,
        nr.w, nr.h, tw, th, ms, nr.fuWarmWant ? "" : ", off: unticked meanwhile");
}

static void nativeFsrupRelease(NativeRife& nr)
{
    if (nr.fuWarmTh)
    {
        // a start in place still warming (nativeUpSwitch): its frames end before the bridge goes; past the compile's
        // bound the worker is still inside the bridge, so its state is left to the process exit
        const DWORD w = WaitForSingleObject(nr.fuWarmTh, (DWORD)kFsrCompileMs + 10000);
        if (w != WAIT_OBJECT_0)
        {
            LOG("native: fsr upscale: its warm-up still runs after %d s, its state is left to the process exit\n",
                kFsrCompileMs / 1000 + 10);
            return;
        }
        CloseHandle(nr.fuWarmTh);
        nr.fuWarmTh = nullptr;
        InterlockedExchange(&nr.fuWarm, 0);
    }
    if (nr.stream)
        cudaStreamSynchronize(nr.stream);
    if (nr.fuMade)
        LOG("native: fsr upscale: %llu frames (%s %s)\n", (unsigned long long)nr.fuN, kFuName[nr.fuKind], nr.fuVersion);
    if (nr.fuProfN)
    {
        const double n = (double)nr.fuProfN, nv = nr.fuProfVecN ? (double)nr.fuProfVecN : 1.0;
        double sum = 0.0;
        for (double s : nr.fuStageMs)
            sum += s;
        LOG("[fsrup-prof] %llu frames (%llu with vectors), ms a frame: input pack %.3f, Optical Flow %.3f, vector "
            "kernels %.3f (the two a vector frame), FSR on D3D12 %.3f, output unpack %.3f, all %.3f\n",
            (unsigned long long)nr.fuProfN, (unsigned long long)nr.fuProfVecN, nr.fuStageMs[0] / n,
            nr.fuStageMs[1] / nv, nr.fuStageMs[2] / nv, nr.fuStageMs[3] / n, nr.fuStageMs[4] / n, sum / n);
    }
    for (cudaEvent_t& e : nr.fuEv)
        if (e)
        {
            cudaEventDestroy(e);
            e = nullptr;
        }
    nr.fuProfN = 0;
    for (int k = nr.fuPasses - 1; k >= 1; k--) // the chained passes past 3x (fuChainPlan), then pass 0's own output
    {
        NativeRife::FuPass& p = nr.fuX[k - 1];
        for (uint8_t** b : {&p.dIn, &p.dMv, &p.dOut})
            if (*b)
            {
                cudaFree(*b);
                *b = nullptr;
            }
        if (p.sem)
        {
            cudaDestroyExternalSemaphore(p.sem);
            p.sem = nullptr;
        }
        for (cudaExternalMemory_t* em : {&p.emIn, &p.emMv, &p.emOut})
            if (*em)
            {
                cudaDestroyExternalMemory(*em);
                *em = nullptr;
            }
        if (p.made && g_fsrupK[nr.fuKind].destroyI)
            g_fsrupK[nr.fuKind].destroyI(k);
        p.made = false;
    }
    nr.fuPasses = 1;
    if (nr.dFu0Out)
    {
        cudaFree(nr.dFu0Out);
        nr.dFu0Out = nullptr;
    }
    if (nr.fu0EmOut)
    {
        cudaDestroyExternalMemory(nr.fu0EmOut);
        nr.fu0EmOut = nullptr;
    }
    for (uint8_t** p : {&nr.dFuIn, &nr.dFuMv, &nr.dFuOut})
        if (*p)
        {
            cudaFree(*p);
            *p = nullptr;
        }
    if (nr.fuSem)
    {
        cudaDestroyExternalSemaphore(nr.fuSem);
        nr.fuSem = nullptr;
    }
    for (cudaExternalMemory_t* em : {&nr.fuEmIn, &nr.fuEmMv, &nr.fuEmOut})
        if (*em)
        {
            cudaDestroyExternalMemory(*em);
            *em = nullptr;
        }
    if (nr.fuMade && g_fsrupK[nr.fuKind].destroy)
        g_fsrupK[nr.fuKind].destroy();
    nr.fuMade = false;
    nativeOfVecFree(nr.fuOf);
}

#include "smv-live-native-anime4k.inl"

#include "smv-live-native-artcnn.inl"

#include "smv-live-native-anime4k-restore.inl"

// native DRBA's buffers (both routes): the ring of pictures at the pictures' pad, everything the
// motion search uses at the model's frame (RIFE's two domains: the motion frame, where the ring also
// keeps the motion frames and the IFNet's x; else the pictures' pad, the encode's fp16 frame in dXh's
// cur half)
// DRBA as its author runs it (NativeRife::drAuthor) in two domains, the default; SMV_DRBA_AUTHOR=0 = the network's
// frames as SMV pads RIFE's (the edge replicated to a multiple of 64, R, G, B) and the blend on the pictures
static bool drbaAuthorWanted(const NativeRife& nr)
{
    return nr.drba && nr.mph && lkEnv("SMV_DRBA_AUTHOR") != "0";
}

static bool nativeDrbaAlloc(NativeRife& nr)
{
    const size_t plane = (size_t)nr.ph * nr.pw, mplane = nr.mph ? (size_t)nr.mph * nr.mpw : plane;
    nr.drAuthor = drbaAuthorWanted(nr);
    nr.drSFid[0] = nr.drSFid[1] = -1;
    if (nr.drAuthor)
        NCHK(cudaMalloc((void**)&nr.dDrS, 6 * mplane * sizeof(float)), "alloc drba stretched pair");
    if (!nr.mph)
        NCHK(cudaMalloc((void**)&nr.dXh, 6 * plane * sizeof(uint16_t)), "alloc x fp16");
    NCHK(cudaMalloc((void**)&nr.dDrIBase, 4 * 3 * plane * sizeof(float)), "alloc drba frame ring");
    if (nr.mph)
        NCHK(cudaMalloc((void**)&nr.dDrMBase, 5 * 3 * mplane * sizeof(uint16_t)), "alloc drba motion ring");
    for (int i = 0; i < 4; i++)
    {
        nr.dDrI[i] = nr.dDrIBase + (size_t)i * 3 * plane;
        if (nr.mph)
            nr.dDrM[i] = nr.dDrMBase + (size_t)(3 - i) * 3 * mplane;
    }
    for (int i = 0; i < 2; i++)
    {
        // two domains: the blend reads its pictures straight from the ring (nativeDrbaPics), the IFNet its
        // motion frames (nativeDrbaTween)
        if (!nr.mph)
            NCHK(cudaMalloc((void**)&nr.dDrX[i], 6 * plane * sizeof(float)), "alloc drba x");
        NCHK(cudaMalloc((void**)&nr.drWin[i].f10, 2 * mplane * sizeof(float)), "alloc drba window");
        NCHK(cudaMalloc((void**)&nr.drWin[i].r, 4 * mplane * sizeof(float)), "alloc drba window");
    }
    NCHK(cudaMalloc((void**)&nr.dDrAcc, 6 * mplane * sizeof(long long)), "alloc drba accumulator");
    // DRBA as its author: the accumulator is idle from a tween's DrmNorm (and block0's FlowNorm) to its next clear,
    // and every reader of these two runs on nr.stream before that clear (the blend's bilinear back, the pack of a real
    // frame): the blend at the network's frame, also a real frame's stretch on its way back, takes the accumulator's
    // last 12 B a pixel; the real frame stretched back (12 B a picture pixel) takes its first 36 B a motion pixel when
    // the picture fits there (the pictures sit at the working size, the motion frame at the decoded size), else its
    // own block
    if (nr.drAuthor)
    {
        nr.dDrB = (float*)nr.dDrAcc + 9 * mplane;
        if (plane <= 3 * mplane)
            nr.dDrRT = (float*)nr.dDrAcc;
        else
        {
            NCHK(cudaMalloc((void**)&nr.dDrRT, 3 * plane * sizeof(float)), "alloc drba real frame");
            nr.drRtOwn = true;
        }
        LOG("native: DRBA as its author runs it: the network's %dx%d frame stretched bilinearly as B, G, R, every "
            "frame back to %dx%d (SMV_DRBA_AUTHOR=0 = the padded frame); the blend%s in the accumulator\n",
            nr.mpw, nr.mph, nr.w, nr.h, nr.drRtOwn ? "" : " and the real frame");
    }
    return true;
}

static void nativeHdrModelSetup(NativeRife& nr, int pre, int post, float white);
static bool nativeCudaInitEarly(NativeRife& nr, const std::wstring& cacheDir)
{
    if (!nativeBuildKernels(nr, cacheDir))
        return false;
    // live: GMFSS and Restore read the HDR desktop's PQ planes as SDR sRGB over its SDR white, decided before
    // the engines and the RTX Video setup (a helper thread) size their buffers; offline sets it from the source
    if (!g_offline)
        nativeHdrModelSetup(nr, nr.hdr ? 1 : 0, nr.hdr ? 1 : 0, (float)(g_sdrWhite / 80.0));
    // the live working size the handoff sized the model frame with (liveWorkSize), and where it came from
    if (!g_offline)
    {
        int presW, presH, mode, mw, mh, raised;
        liveWorkSize(nr.cw, nr.ch, presW, presH, mode, mw, mh, &raised);
        const int shown = mode ? mode : (g_flowScale >= 0.995 ? 1 : 0); // no --scale = DLAA
        char how[64];
        if (shown)
            sprintf_s(how, "DLSS mode %ls", kDlssModeName[shown - 1]);
        else
            sprintf_s(how, "%.2f", g_flowScale);
        LOG("native: live working size %dx%d = %s of the %dx%d presented, capture %dx%d%s\n", nr.w, nr.h, how, presW,
            presH, nr.cw, nr.ch,
            raised == 3   ? " (raised to DLSS 4.5's minimum, 128 px a side, aspect kept)"
            : raised == 2 ? " (raised to DLSS 5's minimum width, 128 px, aspect kept)"
            : raised == 1 ? " (raised to the 64 px floor, aspect kept)"
                          : "");
    }
    // live: Restore reads the captured frame and folds it to the model size before the model runs, so srcW /
    // srcH name the capture from here on (Restore's setup in nativeModelInit and its target in nativeRtxInit read
    // them, possibly on two threads)
    if (nr.restore)
    {
        nr.restPre = true;
        nr.sw = nr.cw;
        nr.sh = nr.ch;
    }

    const int capBpp = nr.hdr ? 8 : 4;
    const size_t plane = (size_t)nr.ph * nr.pw;
    NCHK(cudaMalloc((void**)&nr.dCap, (size_t)nr.cw * nr.ch * capBpp), "alloc capture staging");
    if (!nr.drba) // DRBA builds each frame in its ring slot: no dX
        NCHK(cudaMalloc((void**)&nr.dX, 6 * plane * sizeof(float)), "alloc x");
    // the identical-pair flag and its pinned readback (one int per group)
    NCHK(cudaMalloc((void**)&nr.dStaticFlag, sizeof(int)), "alloc static flag");
    NCHK(cudaHostAlloc((void**)&nr.hStaticFlag, sizeof(int), cudaHostAllocDefault), "alloc static flag readback");
    // no-engine mode: the packed frame (the cur half of x) is the whole model side; a GMFSS
    // session shares only dX (its chain's own buffers are sized off the engines in
    // nativeGmfssSetup), so neither needs the RIFE encode buffers (the timestep and merged
    // buffers follow the engine's dtypes: nativeTweenBuffers)
    // RIFE's two domains: the features and the motion frames (instead of the fp16 copy of x) at the
    // motion frame, plus the shrink of an enlarged working-size picture back to it
    if (!nr.noEngine && !nr.gmfss && !nr.nvof && !nr.fruc && !nr.drba)
    {
        const size_t mplane = nr.mph ? (size_t)nr.mph * nr.mpw : plane;
        if (nr.mph)
            NCHK(cudaMalloc((void**)&nr.dM, 6 * mplane * sizeof(uint16_t)), "alloc motion frames");
        else
            NCHK(cudaMalloc((void**)&nr.dXh, 6 * plane * sizeof(uint16_t)), "alloc x fp16");
        if (nr.mph && (nr.mw != nr.w || nr.mh != nr.h))
            NCHK(cudaMalloc((void**)&nr.dMFit, (size_t)3 * nr.mw * nr.mh * sizeof(float)), "alloc motion fit");
    }
    // native DRBA: the history ring of padded frames and encodes replaces f0 / f1
    if (nr.drba)
    {
        if (!nativeDrbaAlloc(nr))
            return false;
        if (nr.mph && (nr.mw != nr.w || nr.mh != nr.h))
            NCHK(cudaMalloc((void**)&nr.dMFit, (size_t)3 * nr.mw * nr.mh * sizeof(float)), "alloc motion fit");
    }
    // live Restore: the fold's horizontal pass at the model's output height (4x, the largest scale) and the model-size
    // target, and the folded model-size frame (the model's input / output follow in nativeModelInit's restoreReady)
    if (nr.restore)
    {
        const size_t mp = (size_t)nr.w * nr.h;
        NCHK(cudaMalloc((void**)&nr.dRestTmp, (size_t)3 * 4 * srcH(nr) * nr.w * sizeof(float)),
             "alloc restore fold pass");
        NCHK(cudaMalloc((void**)&nr.dRest, 3 * mp * sizeof(float)), "alloc restore model-size frame");
    }
    // With live TrueHDR the PQ frame always exists at capture resolution first, so
    // the resize pair runs at EVERY image scale (at 1.00 the triangle filter sits on identity
    // positions and is an exact copy) and both staging planes are needed even at scale 1.00.
    if (nr.w != nr.cw || nr.h != nr.ch || nr.rtxHdr)
    {
        // also nativeFitPlanar's pass (3, source height, target width): the capture -> working size of Restore's
        // HDR reference and of the stage RTX VSR falls back from (ch x w), the motion frame's shrink (h x mw)
        nr.tmpN = (size_t)3 * (std::max)((size_t)nr.ch * nr.w, nr.dMFit ? (size_t)nr.h * nr.mw : (size_t)0);
        NCHK(cudaMalloc((void**)&nr.dTmp, nr.tmpN * sizeof(float)), "alloc resize temp");
        // HDR rescales ON PQ, so the conversion runs at capture resolution first
        if (nr.hdr)
            NCHK(cudaMalloc((void**)&nr.dCapF, (size_t)3 * nr.ch * nr.cw * sizeof(float)), "alloc PQ capture plane");
    }
    // Restore reads the captured frame as capture-size planes (PQ in HDR)
    if (nr.restPre && !nr.dCapF)
        NCHK(cudaMalloc((void**)&nr.dCapF, (size_t)3 * nr.ch * nr.cw * sizeof(float)), "alloc capture planes");
    NCHK(cudaEventCreateWithFlags(&nr.capEv, cudaEventDisableTiming), "create capture event");
    if (!g_offline)
    {
        for (cudaEvent_t& e : nr.pfEv)
            NCHK(cudaEventCreate(&e), "create frame timing event");
        for (cudaEvent_t& e : nr.grEv)
            NCHK(cudaEventCreate(&e), "create group timing event");
        for (auto& s : nr.fgL1Ev)
            for (cudaEvent_t& e : s)
                NCHK(cudaEventCreate(&e), "create fsrfg call timing event");
        for (auto& s : nr.fgOfEv)
            for (cudaEvent_t& e : s)
                NCHK(cudaEventCreate(&e), "create fsrfg Optical Flow timing event");
        NCHK(cudaEventCreate(&nr.grLagEv), "create fsrfg slot timing event");
    }
    // the nvof model: the Optical Flow session and the glue buffers (the kernels are bound above)
    if (nr.nvof && !nativeNvofSetup(nr))
        return false;
    // Smooth Motion: the bridge's FRUC instance and its BGRA8 surfaces
    if (nr.fruc && !nativeFrucSetup(nr))
        return false;
    return true;
}

// the model cache folder the handoff named (the kernels' cubins live there)
static std::wstring nativeCacheDir(const NativeRife& nr)
{
    return std::wstring(nr.cachePath.begin(), nr.cachePath.end());
}

// Live RTX TrueHDR runs in this process. Everything comes from the exe's own
// globals, parsed from the app's RTX HDR controls, so no new flag exists. Runs after
// nativeLoadDlls (the bridge folder is derived from the runtime folder found there).
// srcHdr = offline HDR video (1 PQ, 2 HLG): the upscalers' HDR levers apply to it as to live's HDR planes
static bool nativeConfigHdr(NativeRife& nr, int srcHdr = 0)
{
    // live Sharpen and RTX VSR read the exe's own globals; VSR needs the bridge and nvngx_vsr.dll present (else
    // Lanczos, never a route change), and the fit must enlarge (decided in nativeRtxInit once the handoff geometry is
    // known). It takes SDR, so on HDR planes it reads their SDR view (SMV_VSR_HDR=0 = Lanczos there)
    nr.sharpen = (float)(g_sharpen < 0.0 ? 0.0 : (g_sharpen > 2.0 ? 2.0 : g_sharpen));
    // RTX VSR's quality level (--vsr-quality, the app's slider): 1..4, 4 = Ultra by default; level 0 (bicubic) is left
    // out, since an unticked box already gives Lanczos. The network runs at the input size, so a level's price is the same
    // at any output size
    nr.vsrSet.QualityLevel = (uint32_t)(g_vsrQuality < 1 ? 1 : (g_vsrQuality > 4 ? 4 : g_vsrQuality));
    if (g_rtxVsr)
    {
        if (g_vsrQuality != (int)nr.vsrSet.QualityLevel)
            LOG("native: --vsr-quality %d is outside 1..4, RTX VSR runs quality %u\n", g_vsrQuality,
                nr.vsrSet.QualityLevel);
        else if (nr.vsrSet.QualityLevel != 4)
            LOG("native: RTX VSR quality %u of 4\n", nr.vsrSet.QualityLevel);
        wchar_t vh[8]{};
        if ((g_hdr || srcHdr) && GetEnvironmentVariableW(L"SMV_VSR_HDR", vh, 8) && vh[0] == L'0')
            LOG(g_hdr ? "native: live RTX VSR off on HDR planes (SMV_VSR_HDR=0), Lanczos\n"
                      : "native: RTX VSR off on HDR video (SMV_VSR_HDR=0), Lanczos\n");
        else
        {
            std::wstring dir;
            wchar_t ov[MAX_PATH]{};
            if (GetEnvironmentVariableW(L"SMV_RTXVIDEO_DIR", ov, MAX_PATH) && ov[0])
                dir = ov;
            else if (!g_nativeEngineDir.empty())
                dir = g_nativeEngineDir + L"\\rtxvideo";
            const bool have = !dir.empty() &&
                              GetFileAttributesW((dir + L"\\rtxvideo_cuda.dll").c_str()) != INVALID_FILE_ATTRIBUTES &&
                              GetFileAttributesW((dir + L"\\nvngx_vsr.dll").c_str()) != INVALID_FILE_ATTRIBUTES;
            if (!have)
                LOG("native: live RTX VSR unavailable (no rtxvideo bridge or nvngx_vsr.dll), Lanczos\n");
            else if (!rtxBridgeLoad())
                return false;
            else
                nr.vsrWant = true;
        }
    }
    // AMD FSR upscaling (--fsr-upscale) takes RTX VSR's resize under its rules (enlarging only); the two exclude each
    // other (the app ticks one), RTX VSR wins a command line with both. On HDR planes it runs in the bridge's HDR mode
    // (linear light, nativeFsrupSetup; SMV_FSRUP_HDR=0 = Lanczos there)
    wchar_t fh[8]{};
    if (g_fsrUp && g_rtxVsr)
        LOG("native: --fsr-upscale ignored: RTX VSR holds the resize (the two exclude each other)\n");
    else if (g_fsrUp && (g_hdr || srcHdr) && GetEnvironmentVariableW(L"SMV_FSRUP_HDR", fh, 8) && fh[0] == L'0')
        LOG(g_hdr ? "native: AMD FSR upscaling off on HDR planes (SMV_FSRUP_HDR=0), Lanczos\n"
                  : "native: AMD FSR upscaling off on HDR video (SMV_FSRUP_HDR=0), Lanczos\n");
    else if (g_fsrUp)
    {
        std::string dir = lkEnv("SMV_FSRUP_DIR");
        if (dir.empty() && !g_nativeEngineDir.empty())
            dir = wideToUtf8(g_nativeEngineDir) + "\\fsrup";
        bool have = !dir.empty();
        for (const char* f :
             {"smv_fsrup_bridge.dll", "amd_fidelityfx_loader_dx12.dll", "amd_fidelityfx_upscaler_dx12.dll"})
            have = have && lkFile(dir + "\\" + f);
        if (!have)
            LOG("native: AMD FSR upscaling unavailable (smv_fsrup_bridge.dll and AMD's FidelityFX loader + upscaler "
                "needed in %s), Lanczos\n",
                dir.c_str());
        else
        {
            nr.vsrWant = true;
            nr.fsrUp = true;
            nr.fsrUpDir = dir;
        }
    }
    // Anime4K (--anime4k) takes RTX VSR's resize under the same rules; it excludes RTX VSR and AMD FSR upscaling the
    // way those two exclude each other (either of them wins a command line with both). Its kernels build at the
    // session's setup (nativeA4kSetup)
    if (g_a4kUp && (g_rtxVsr || g_fsrUp))
        LOG("native: --anime4k ignored: %s holds the resize (the upscalers exclude each other)\n",
            g_rtxVsr ? "RTX VSR" : "AMD FSR upscaling");
    else if (g_a4kUp)
    {
        nr.vsrWant = true;
        nr.a4kUp = true;
    }
    // ArtCNN (--artcnn) the same way: any of the other three on the command line wins over it
    if (g_acnUp && (g_rtxVsr || g_fsrUp || g_a4kUp))
        LOG("native: --artcnn ignored: %s holds the resize (the upscalers exclude each other)\n",
            g_rtxVsr  ? "RTX VSR"
            : g_fsrUp ? "AMD FSR upscaling"
                      : "Anime4K");
    else if (g_acnUp)
    {
        nr.vsrWant = true;
        nr.acnUp = true;
    }
    // Intel XeSS-SR (--xess-upscale) runs on AMD FSR upscaling's route through its own bridge (fuKind 1:
    // smv_xess_bridge.dll with Intel's libxess.dll, which the app's XeSS setup installs into engine\xess); every other
    // upscaler on the command line wins over it; SMV_FSRUP_HDR=0 turns the route off on HDR for both bridges
    if (g_xessUp && (g_rtxVsr || nr.fsrUp || nr.a4kUp || nr.acnUp))
        LOG("native: --xess-upscale ignored: %s holds the resize (the upscalers exclude each other)\n",
            g_rtxVsr   ? "RTX VSR"
            : nr.fsrUp ? "AMD FSR upscaling"
            : nr.a4kUp ? "Anime4K"
                       : "ArtCNN");
    else if (g_xessUp && (g_hdr || srcHdr) && GetEnvironmentVariableW(L"SMV_FSRUP_HDR", fh, 8) && fh[0] == L'0')
        LOG(g_hdr ? "native: Intel XeSS upscaling off on HDR planes (SMV_FSRUP_HDR=0), Lanczos\n"
                  : "native: Intel XeSS upscaling off on HDR video (SMV_FSRUP_HDR=0), Lanczos\n");
    else if (g_xessUp)
    {
        std::string dir = lkEnv("SMV_XESS_DIR");
        if (dir.empty() && !g_nativeEngineDir.empty())
            dir = wideToUtf8(g_nativeEngineDir) + "\\xess";
        bool have = !dir.empty();
        for (const char* f : {"smv_xess_bridge.dll", "libxess.dll"})
            have = have && lkFile(dir + "\\" + f);
        if (!have)
            LOG("native: Intel XeSS upscaling unavailable (smv_xess_bridge.dll and Intel's libxess.dll needed in %s; the "
                "app's XeSS setup installs them), Lanczos\n",
                dir.c_str());
        else
        {
            nr.vsrWant = true;
            nr.fsrUp = true;
            nr.fsrUpDir = dir;
            nr.fuKind = 1;
        }
    }
    if (!(g_hdr && g_rtxHdr))
        return true;
    if (!rtxBridgeLoad())
        return false;
    nr.rtxHdr = true;
    nr.thdr.Contrast = (uint32_t)(g_hdrCon < 0 ? 0 : (g_hdrCon > 200 ? 200 : g_hdrCon));
    nr.thdr.Saturation = (uint32_t)(g_hdrSat < 0 ? 0 : (g_hdrSat > 200 ? 200 : g_hdrSat));
    nr.thdr.MiddleGray = 50; // the live default
    nr.thdr.MaxLuminance = g_thdrPeak;
    nr.sdrScale = (float)(g_sdrWhite / 80.0 < 0.5 ? 0.5 : g_sdrWhite / 80.0);
    nr.rtxMode = !wcscmp(g_hdrColor, L"rtx") ? 1 : (!wcscmp(g_hdrColor, L"raw") ? 2 : 0);
    nr.rtxVib = (float)(g_hdrVib < 0.0 ? 0.0 : (g_hdrVib > 1.0 ? 1.0 : g_hdrVib));
    nr.rtxSb = (float)(g_hdrSb < 0.0 ? 0.0 : (g_hdrSb > 1.0 ? 1.0 : g_hdrSb));
    return true;
}

// gmflow split: the sizes of the contract between `_m`, the scale-0 matching and propagation (k_attn2), A, k_localCorr
// and gmflow B, all on the host's kernels: the backbone writes each frame's map4 / map8 as planar fp16 [128, H, W] (A /
// `_m` take both frames' as [2, 128, H, W]); A hands out qf [4, H, W, 128] fp16 (channels-last; q = rows 0..1, f = rows
// 2..3) and flow0 fp32 [2, 2, H, W] on the quarter (gmH x gmW) of the half frame; `_m` hands out mf / mq / mk fp16 [2, N,
// 128] on the eighth (N = gmN8 tokens, gmW8 wide), the propagation writes prop fp16 [2, N, 2]; k_localCorr computes the
// window of K = kGmCorrK samples itself and writes corr [2, HW, K] for B. A's 8 x 8 windows need a quarter in eighths.
static bool nativeGmSplitCheck(NativeRife& nr)
{
    if (nr.hh <= 0 || nr.hw <= 0 || nr.hh % 32 || nr.hw % 32)
    {
        LOG("native: gmfss split: the half %dx%d is not a multiple of 32\n", nr.hw, nr.hh);
        return false;
    }
    nr.gmH = nr.hh / 4;
    nr.gmW = nr.hw / 4;
    nr.gmN8 = (nr.hh / 8) * (nr.hw / 8);
    nr.gmW8 = nr.hw / 8;
    nr.gmPropHalf = true;
    nr.gmK = kGmCorrK;
    return true;
}

// GMFSS: the chain's contract and its buffers: one pair's and one tween's planar buffers at the
// nets' channel counts and dtypes. Runs inside nativeModelInit, so a mismatch refuses the session
// before any group. The accumulator is one
// buffer for all eight splats (they are strictly sequential on the one stream): the widest is
// the half level's gmC[0] + 1 planes, and the quarter / eighth levels need less
// ((2 * gmC[1] + 1) / 4 and (2 * gmC[2] + 1) / 16 of a half plane at the stock 64/128/192).
// GMFSS's eight contexts enqueue one after another on nr.stream (their aux streams join it at each enqueue's end), so
// they share ONE activation block, the largest net's, instead of one each: set before the warm-up, so a whole-graph
// capture records its address. SMV_GM_CTXSHARE=0 = each net its own.
// Restore on the host's kernels: its channels-last activations (Real-ESRGAN: two 64-half frames at the source size;
// AnimeJaNai: eight at the half grid: x0, the first block's output, two block ping-pong frames, c1, c2, c3, the concat
// conv's output), 0 = not this route
static size_t restNatBytes(const NativeRife& nr)
{
    if (nr.restNatKind == 3)
        return 0; // Anime4K's Restore: its own layer slots (nativeA4kRestSetup)
    if (nr.restNatKind == 2)
        return 8 * (size_t)((srcW(nr) + 1) / 2) * (size_t)((srcH(nr) + 1) / 2) * 128;
    return nr.restNat ? 2 * (size_t)srcW(nr) * (size_t)srcH(nr) * 128 : 0;
}

// AnimeJaNai on st: dRestIn -> k_ajFirst -> per block 3 x k_restConvMid + k_ajAtt -> k_ajCat -> k_restConvLast ->
// dRestOut (48 fp16 sub-pixel planes at the half grid); false = a launch failed
static bool ajNatRun(NativeRife& nr, cudaStream_t st)
{
    int sw = srcW(nr), sh = srcH(nr), hw = nr.restSubW, hh = nr.restSubH, n = hw * hh;
    const size_t nb = (size_t)n * 128;
    auto at = [&](size_t i) { return (void*)((char*)nr.dRestW + nr.restNatOff[i]); };
    auto buf = [&](int i) { return (void*)((char*)nr.dRestA + (size_t)i * nb); };
    void *x0 = buf(0), *out1 = buf(1), *tA = buf(2), *tB = buf(3), *c1 = buf(4), *c2 = buf(5), *c3 = buf(6),
         *cat = buf(7);
    const unsigned gx = (unsigned)(hw + 31) / 32, gy = (unsigned)(hh + 7) / 8, gp = (unsigned)(n + 127) / 128,
                   smem = 10 * 34 * 128;
    void *w0 = at(0), *b0 = at(1);
    void* a0[] = {&nr.dRestIn, &sw, &sh, &x0, &w0, &b0, &hw, &hh};
    if (cuLaunchKernel(nr.fAjFirst, gx, gy, 1, 256, 1, 1, 0, (CUstream)st, a0, nullptr) != CUDA_SUCCESS)
        return false;
    void* t = x0;
    for (int k = 0; k < nr.restNatBlocks; k++)
    {
        void* io[3][2] = {{t, c1}, {c1, c2}, {c2, c3}};
        for (int j = 0; j < 3; j++)
        {
            const size_t o = (size_t)(2 + 11 * k + 3 * j);
            void *wf = at(o), *bs = at(o + 1), *sl = at(o + 2);
            void* a[] = {&io[j][0], &io[j][1], &wf, &bs, &sl, &hh, &hw};
            if (cuLaunchKernel(nr.fRestConvMid, gx, gy, 1, 128, 1, 1, smem, (CUstream)st, a, nullptr) != CUDA_SUCCESS)
                return false;
        }
        void* dst = k == 0 ? out1 : (t == tA ? tB : tA);
        void *wa = at((size_t)(11 + 11 * k)), *ba = at((size_t)(12 + 11 * k));
        void* aa[] = {&c3, &t, &dst, &wa, &ba, &n};
        if (cuLaunchKernel(nr.fAjAtt, gp, 1, 1, 128, 1, 1, 0, (CUstream)st, aa, nullptr) != CUDA_SUCCESS)
            return false;
        t = dst;
    }
    const size_t tail = (size_t)(2 + 11 * nr.restNatBlocks);
    void *wc = at(tail), *bc = at(tail + 1), *wl = at(tail + 2), *bl = at(tail + 3);
    void* ac[] = {&x0, &t, &out1, &c1, &cat, &wc, &bc, &n};
    if (cuLaunchKernel(nr.fAjCat, gp, 1, 1, 128, 1, 1, 0, (CUstream)st, ac, nullptr) != CUDA_SUCCESS)
        return false;
    void* al[] = {&cat, &nr.dRestOut, &wl, &bl, &hh, &hw};
    return cuLaunchKernel(nr.fRestConvLast, gx, gy, 1, 128, 1, 1, smem, (CUstream)st, al, nullptr) == CUDA_SUCCESS;
}

// one Restore on the host's kernels on st: dRestIn (three fp16 planes) -> k_restConv0 -> 16 x k_restConvMid ->
// k_restConvLast -> dRestOut (48 fp16 sub-pixel planes); AnimeJaNai = ajNatRun; false = a launch failed
static bool restNatRun(NativeRife& nr, cudaStream_t st)
{
    if (nr.restNatKind == 2)
        return ajNatRun(nr, st);
    int w = srcW(nr), h = srcH(nr);
    auto at = [&](int i) { return (void*)((char*)nr.dRestW + nr.restNatOff[(size_t)i]); };
    const unsigned gx = (unsigned)(w + 31) / 32, gy = (unsigned)(h + 7) / 8, smem = 10 * 34 * 128;
    void *w0 = at(0), *b0 = at(1), *s0 = at(2);
    void* a0[] = {&nr.dRestIn, &nr.dRestA, &w0, &b0, &s0, &h, &w};
    if (cuLaunchKernel(nr.fRestConv0, gx, gy, 1, 256, 1, 1, 0, (CUstream)st, a0, nullptr) != CUDA_SUCCESS)
        return false;
    void *s = nr.dRestA, *d = nr.dRestB;
    for (int L = 0; L < 16; L++)
    {
        void *wf = at(3 + 3 * L), *bs = at(4 + 3 * L), *sl = at(5 + 3 * L);
        void* a[] = {&s, &d, &wf, &bs, &sl, &h, &w};
        if (cuLaunchKernel(nr.fRestConvMid, gx, gy, 1, 128, 1, 1, smem, (CUstream)st, a, nullptr) != CUDA_SUCCESS)
            return false;
        std::swap(s, d);
    }
    void *wl = at(51), *bl = at(52);
    void* al[] = {&s, &nr.dRestOut, &wl, &bl, &h, &w};
    return cuLaunchKernel(nr.fRestConvLast, gx, gy, 1, 128, 1, 1, smem, (CUstream)st, al, nullptr) == CUDA_SUCCESS;
}

// Restore on the host's kernels for this session: the weights uploaded, dRestIn / dRestOut, the ping-pong (inside the
// shared activation block when that holds it: every enqueue there runs one after another on nr.stream, Restore
// first), one run on zeros; the engine route's facts it reads (fp16 in and out, 48 sub-pixel planes at x4)
static bool restNatSetup(NativeRife& nr)
{
    if (nr.restNatKind == 3)
        return nativeA4kRestSetup(nr);
    std::vector<uint8_t> blob;
    std::string why;
    const bool aj = nr.restNatKind == 2;
    if (!(aj ? ajNatLoad(g_restoreModel, blob, nr.restNatOff, nr.restNatBlocks, why)
             : restNatLoad(blob, nr.restNatOff, why)))
    {
        LOG("native: Restore on the host's kernels failed (%s)\n", why.c_str());
        return false;
    }
    nr.restHalfIn = nr.restHalfOut = true;
    nr.restSub = true;
    nr.restScale = aj ? 2 : 4;
    nr.restSubS = 4;
    nr.restSubW = aj ? (srcW(nr) + 1) / 2 : srcW(nr);
    nr.restSubH = aj ? (srcH(nr) + 1) / 2 : srcH(nr);
    nr.restSubBase = !aj;
    const size_t sp = (size_t)srcW(nr) * srcH(nr), pp = restNatBytes(nr);
    if (!nr.dRestW)
        NCHK(cudaMalloc(&nr.dRestW, blob.size()), "alloc restore weights");
    NCHK(cudaMemcpy(nr.dRestW, blob.data(), blob.size(), cudaMemcpyHostToDevice), "restore weights upload");
    if (!nr.dRestIn)
        NCHK(cudaMalloc(&nr.dRestIn, 3 * sp * sizeof(uint16_t)), "alloc restore input");
    if (!nr.dRestOut)
        NCHK(cudaMalloc(&nr.dRestOut, 48 * (size_t)nr.restSubW * nr.restSubH * sizeof(uint16_t)),
             "alloc restore output");
    if (nr.dGmCtxMem && nr.gmCtxMemB >= (int64_t)pp)
    {
        nr.dRestA = nr.dGmCtxMem;
        nr.restNatOwn = false;
    }
    else if (!nr.dRestA)
    {
        nr.restNatOwn = true;
        NCHK(cudaMalloc(&nr.dRestA, pp), "alloc restore activations");
    }
    nr.dRestB = (char*)nr.dRestA + pp / 2; // Real-ESRGAN's pong
    NCHK(cudaMemsetAsync(nr.dRestIn, 0, 3 * sp * sizeof(uint16_t), nr.stream), "restore warm-up clear");
    if (!restNatRun(nr, nr.stream))
    {
        LOG("native: restore warm-up launch failed\n");
        return false;
    }
    NCHK(cudaStreamSynchronize(nr.stream), "restore warm-up sync");
    LOG("native: Restore = %s on the host's kernels at %dx%d (fp16 accumulate; its %.0f MB of activations %s)\n",
        aj ? "AnimeJaNai (SPAN-F3)" : "Real-ESRGAN", srcW(nr), srcH(nr), pp / 1048576.0,
        nr.restNatOwn ? "on their own" : "in the shared activation block");
    return true;
}

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

static bool nativeGmfssSetup(NativeRife& nr)
{
    auto bad = [](const char* what) {
        LOG("native: gmfss chain: %s\n", what);
        return false;
    };
    if (nr.hh <= 0 || nr.hw <= 0 || (nr.hh & 3) || (nr.hw & 3))
        return bad("the half frame is not a multiple of 4");
    if (nr.hh * 2 != nr.ph || nr.hw * 2 != nr.pw)
        return bad("the half frame is not half of the padded frame");
    if (nr.ph < kGmMinPad || nr.pw < kGmMinPad)
        return bad("the padded frame is below 128 a side (gmflow's quarter-scale windows need 2 tokens a side)");
    const int hh = nr.hh, hw = nr.hw;
    // feat_ext on the host's kernels: the padded frame in, fp16 levels of 64 / 128 / 192 channels out
    if (nr.ph % 64 || nr.pw % 64)
        return bad("the padded frame is not a multiple of 64 (feat_ext's three stride-2 convs and the tiles)");
    nr.gmC[0] = kFnCh[1];
    nr.gmC[1] = kFnCh[2];
    nr.gmC[2] = kFnCh[3];
    nr.gmFeatHalf = true;
    // the fused bidir GMFlow at the half, all on the host's kernels; B hands the flow out: row 0 = flow01, row 1 = flow10
    // (so the pyramids take both at once)
    if (!nativeGmSplitCheck(nr))
        return false;
    // metricnet on the host's kernels: the motion halves and both flows in, m0 / m1 fp16 out
    nr.gmMetricHalf = true;
    // the GMFSS IFNet on the host's kernels (the half = the /64 pad / 2, so its 1 / 32 level is whole): the 6-plane half
    // buffer and the fp32 timestep in, `rife` (3 fp32 planes) out; its weights and module are loaded below
    // fusionnet on the host's kernels (its module and weights are loaded below): the concats are adjacent planes, so b /
    // c / d are exactly 2x the feature levels, written channels-last fp16 by the feature splats (k_gsGatherNhwc, the
    // heads' first PReLU in their stores) beside feat_ext's channels-last levels; its out = the 64 -> 256 conv before
    // the pixel shuffle, channels-last fp16, which k_fuTailNhwc shuffles and runs conv_last on (fp32 sums, the frame in
    // fp32, straight into dGmF)
    nr.gmFuseHalf = true;
    nr.gmFeatHwc = true;
    nr.gmTail = true;
    nr.gmTailHwc = true;
    nr.gmOutHalf = false;

    const size_t hp = (size_t)hh * hw, qp = hp / 4, ep = hp / 16;
    const size_t plane = (size_t)nr.ph * nr.pw;
    const size_t fe = nr.gmFeatHalf ? 2 : 4, me = nr.gmMetricHalf ? 2 : 4, oe = nr.gmOutHalf ? 2 : 4,
                 be = nr.gmFuseHalf ? 2 : 4;
    size_t accPlanes = (size_t)nr.gmC[0] + 1;
    if (((size_t)nr.gmC[1] + 1) * qp > accPlanes * hp)
        accPlanes = (((size_t)nr.gmC[1] + 1) * qp + hp - 1) / hp;
    if (((size_t)nr.gmC[2] + 1) * ep > accPlanes * hp)
        accPlanes = (((size_t)nr.gmC[2] + 1) * ep + hp - 1) / hp;
    nr.gmGather = lkEnv("SMV_GM_SPLAT") != "old" && hp <= (size_t)2048 * 4096;
    // the gather splats never touch the accumulator: it then holds only the cut fusionnet's out (256 fp16 a half
    // pixel = 64 planes of 8 bytes), and nothing with the full graph
    if (nr.gmGather)
        accPlanes = nr.gmTail ? 256 * 2 / sizeof(long long) : 0;
    if (nr.gmTail && accPlanes * sizeof(long long) < 256 * 2)
        return bad("the splat accumulator cannot hold the cut fusionnet's out");
    size_t freeB = 0, freeA = 0, totB = 0;
    cudaMemGetInfo(&freeB, &totB);
    for (int s = 0; s < 2; s++)
        for (int l = 0; l < 3; l++)
            NCHK(cudaMalloc(&nr.dGmFeat[s][l], (size_t)nr.gmC[l] * (l == 0 ? hp : (l == 1 ? qp : ep)) * fe),
                 "alloc gmfss features");
    // feat_ext's module and weights, its scratch in the shared activation block when that holds it
    {
        std::string why;
        if (!nativeFnModule(nr, why))
        {
            LOG("native: gmfss chain: feat_ext unavailable (%s)\n", why.c_str());
            return false;
        }
        const size_t fb = fnNatBytes(nr, nr.gmFeatHwc);
        if (nr.dGmCtxMem && (size_t)nr.gmCtxMemB >= fb)
            nr.dFnScr = nr.dGmCtxMem;
        else
        {
            NCHK(cudaMalloc(&nr.dFnOwn, fb), "alloc gmfss feat_ext scratch");
            nr.dFnScr = nr.dFnOwn;
        }
        LOG("native: gmfss feat_ext on the host's kernels (fp16 sums, %s levels; its %.0f MB of activations %s)\n",
            nr.gmFeatHwc ? "channels-last" : "planar", fb / 1048576.0,
            nr.dFnOwn ? "on their own" : "in the shared activation block");
    }
    // metricnet's module and weights, its scratch in the shared activation block when that holds it
    {
        std::string why;
        if (!nativeMnModule(nr, why))
        {
            LOG("native: gmfss chain: metricnet unavailable (%s)\n", why.c_str());
            return false;
        }
        const size_t mb = mnNatBytes(nr);
        if (!(nr.dGmCtxMem && (size_t)nr.gmCtxMemB >= mb))
            NCHK(cudaMalloc(&nr.dMnOwn, mb), "alloc gmfss metricnet scratch");
        LOG("native: gmfss metricnet on the host's kernels (fp16 sums; its %.0f MB of activations %s)\n",
            mb / 1048576.0, nr.dMnOwn ? "on their own" : "in the shared activation block");
    }
    // the GMFSS IFNet's module and weights, its scratch in the shared activation block when that holds it
    {
        std::string why;
        if (!nativeGiModule(nr, why))
        {
            LOG("native: gmfss chain: the GMFSS IFNet unavailable (%s)\n", why.c_str());
            return false;
        }
        const size_t gb = giNatBytes(nr);
        if (!(nr.dGmCtxMem && (size_t)nr.gmCtxMemB >= gb))
            NCHK(cudaMalloc(&nr.dGiOwn, gb), "alloc gmfss ifnet scratch");
        LOG("native: gmfss ifnet on the host's kernels (fp32 sums; its %.0f MB of activations %s)\n", gb / 1048576.0,
            nr.dGiOwn ? "on their own" : "in the shared activation block");
        nr.giGraphs.mode = chainGraphMode("SMV_GI_GRAPH");
    }
    // fusionnet's module, weights and conv_last, its activations in the shared activation block when that holds them
    {
        std::string why;
        if (!nativeFuModule(nr, why))
        {
            LOG("native: gmfss chain: fusionnet unavailable (%s)\n", why.c_str());
            return false;
        }
        const size_t ub = fuNatBytes(nr);
        if (!(nr.dGmCtxMem && (size_t)nr.gmCtxMemB >= ub))
            NCHK(cudaMalloc(&nr.dFuOwn, ub), "alloc gmfss fusionnet activations");
        LOG("native: gmfss fusionnet on the host's kernels (fp16 sums, head0's first conv fp32; its %.0f MB of "
            "activations %s)\n",
            ub / 1048576.0, nr.dFuOwn ? "on their own" : "in the shared activation block");
        nr.fuGraphs.mode = chainGraphMode("SMV_FU_GRAPH");
    }
    // gmflow B's module and weights, its activations in the shared activation block when that holds them
    {
        std::string why;
        if (!nativeGbModule(nr, why))
        {
            LOG("native: gmfss chain: gmflow B unavailable (%s)\n", why.c_str());
            return false;
        }
        const size_t bb = gbNatBytes(nr);
        if (!(nr.dGmCtxMem && (size_t)nr.gmCtxMemB >= bb))
            NCHK(cudaMalloc(&nr.dGbOwn, bb), "alloc gmfss gmflow B activations");
        LOG("native: gmfss gmflow B on the host's kernels (fp16 sums in its 3 x 3 conv; its %.0f MB of activations %s)\n",
            bb / 1048576.0, nr.dGbOwn ? "on their own" : "in the shared activation block");
        nr.gbGraphs.mode = chainGraphMode("SMV_GB_GRAPH");
    }
    // gmflow backbone's module and weights, its activations in the shared activation block when that holds them
    {
        std::string why;
        if (!nativeBbModule(nr, why))
        {
            LOG("native: gmfss chain: gmflow's backbone unavailable (%s)\n", why.c_str());
            return false;
        }
        const size_t kb = bbNatBytes(nr);
        if (!(nr.dGmCtxMem && (size_t)nr.gmCtxMemB >= kb))
            NCHK(cudaMalloc(&nr.dBbOwn, kb), "alloc gmfss gmflow backbone activations");
        LOG("native: gmfss gmflow backbone on the host's kernels (fp16 sums in its 3 x 3 convs; its %.0f MB of "
            "activations %s)\n",
            kb / 1048576.0, nr.dBbOwn ? "on their own" : "in the shared activation block");
        nr.bbGraphs.mode = chainGraphMode("SMV_BB_GRAPH");
    }
    // gmflow's transformer (A and `_m`): its module, weights and window tables; its activations in the shared activation
    // block when that holds them; A takes the propagation's output in fp16
    {
        std::string why;
        if (!nr.gmPropHalf)
            why = "the propagated flow is not fp16";
        if (!nr.gmPropHalf || !nativeGaModule(nr, why))
        {
            LOG("native: gmfss chain: gmflow's transformer unavailable (%s)\n", why.c_str());
            return false;
        }
        const size_t ab = gaNatBytes(nr);
        if (!(nr.dGmCtxMem && (size_t)nr.gmCtxMemB >= ab))
            NCHK(cudaMalloc(&nr.dGaOwn, ab), "alloc gmfss gmflow transformer activations");
        LOG("native: gmfss gmflow A and _m on the host's kernels (fp16 sums but in P v'; %s; their %.0f MB of activations "
            "%s)\n",
            gaTuring()   ? "32-key tiles, a 2-stage weight ring"
            : nr.gaAttn2 ? "64-key tiles (A's attention 32-key, two CTAs an SM), a 3-stage weight ring"
                         : "64-key tiles, a 3-stage weight ring",
            ab / 1048576.0, nr.dGaOwn ? "on their own" : "in the shared activation block");
        nr.gaGraphs[0].mode = nr.gaGraphs[1].mode = chainGraphMode("SMV_GA_GRAPH");
    }
    NCHK(cudaMalloc((void**)&nr.dGmHalf, 6 * hp * sizeof(float)), "alloc gmfss halves");
    NCHK(cudaMalloc((void**)&nr.dGmFlow, 4 * hp * sizeof(float)), "alloc gmfss flow");
    NCHK(cudaMalloc(&nr.dGmMetric, 2 * hp * me), "alloc gmfss metric");
    NCHK(cudaMalloc((void**)&nr.dGmFlowP[0], 4 * qp * sizeof(float)), "alloc gmfss flow pyramid");
    NCHK(cudaMalloc((void**)&nr.dGmFlowP[1], 4 * ep * sizeof(float)), "alloc gmfss flow pyramid");
    NCHK(cudaMalloc((void**)&nr.dGmMetP[0], 2 * qp * sizeof(float)), "alloc gmfss metric pyramid");
    NCHK(cudaMalloc((void**)&nr.dGmMetP[1], 2 * ep * sizeof(float)), "alloc gmfss metric pyramid");
    if (accPlanes)
        NCHK(cudaMalloc((void**)&nr.dGmAcc, accPlanes * hp * sizeof(long long)), "alloc gmfss splat accumulator");
    if (nr.gmGather)
        for (int s = 0; s < 2; s++)
        {
            NCHK(cudaMalloc((void**)&nr.dGsCnt[s], hp * sizeof(int)), "alloc gmfss gather counts");
            NCHK(cudaMalloc((void**)&nr.dGsOff[s], (hp + 1) * sizeof(int)), "alloc gmfss gather offsets");
            NCHK(cudaMalloc((void**)&nr.dGsList[s], 4 * hp * sizeof(int)), "alloc gmfss gather lists");
            NCHK(cudaMalloc((void**)&nr.dGsSrc[s], 3 * hp * sizeof(float)), "alloc gmfss gather sources");
            NCHK(cudaMalloc((void**)&nr.dGsBlk[s], 4096 * sizeof(int)), "alloc gmfss gather block sums");
        }
    LOG("native: gmfss splats by %s\n", nr.gmGather ? "gather (lists)" : "scatter (atomics)");
    NCHK(cudaMalloc((void**)&nr.dGmFa, 9 * hp * sizeof(float)), "alloc gmfss fusion a");
    NCHK(cudaMalloc(&nr.dGmFb, (size_t)2 * nr.gmC[0] * hp * be), "alloc gmfss fusion b");
    NCHK(cudaMalloc(&nr.dGmFc, (size_t)2 * nr.gmC[1] * qp * be), "alloc gmfss fusion c");
    NCHK(cudaMalloc(&nr.dGmFd, (size_t)2 * nr.gmC[2] * ep * be), "alloc gmfss fusion d");
    NCHK(cudaMalloc((void**)&nr.dGmT, sizeof(float)), "alloc gmfss timestep");
    if (!nr.gmTail)
        NCHK(cudaMalloc(&nr.dGmOut, 3 * plane * oe), "alloc gmfss fusion out");
    NCHK(cudaMalloc((void**)&nr.dGmF, 3 * plane * sizeof(float)), "alloc gmfss model frame");
    // the pair's scratch (A's qf / flow0, the local correlation, the matching operands and flow, the propagated flow;
    // with encPost the SDR view and the motion halves) is written and read inside nativeGmfssPair only, and the
    // accumulator under the gather splats inside nativeGmfssTween only (the cut out, written before it is read), both
    // on nr.stream: so the scratch is carved from the accumulator when it fits (256-B aligned), else it gets its own
    // buffers (SMV_GM_SCRATCH=own)
    struct Scratch
    {
        void** p;
        size_t bytes;
        const char* what;
    };
    const size_t q4 = (size_t)nr.gmH * nr.gmW;
    std::vector<Scratch> scr;
    scr.push_back({&nr.dGmA[0], 4 * q4 * 128 * sizeof(uint16_t), "alloc gmfss split tensors"});
    scr.push_back({&nr.dGmA[1], 2 * 2 * q4 * sizeof(float), "alloc gmfss split tensors"});
    scr.push_back(
        {(void**)&nr.dGmCorr, (size_t)2 * nr.gmH * nr.gmW * nr.gmK * sizeof(float), "alloc gmfss local correlation"});
    for (int i = 0; i < 3; i++)
        scr.push_back({&nr.dGmM[i], (size_t)2 * nr.gmN8 * 128 * sizeof(uint16_t), "alloc gmfss matching operands"});
    scr.push_back({(void**)&nr.dGmFlow8, (size_t)2 * nr.gmN8 * 2 * sizeof(float), "alloc gmfss matching flow"});
    scr.push_back({&nr.dGmProp, (size_t)2 * nr.gmN8 * 2 * sizeof(uint16_t), "alloc gmfss propagated flow"});
    if (nr.encPost)
    {
        scr.push_back({(void**)&nr.dGmEnc, 3 * plane * sizeof(float), "alloc gmfss SDR view"});
        scr.push_back({(void**)&nr.dGmHalfM, 6 * hp * sizeof(float), "alloc gmfss motion halves"});
    }
    size_t scrB = 0;
    for (const Scratch& s : scr)
        scrB += (s.bytes + 255) & ~(size_t)255;
    nr.gmScratchInAcc =
        nr.gmTail && nr.gmGather && lkEnv("SMV_GM_SCRATCH") != "own" && scrB <= accPlanes * hp * sizeof(long long);
    size_t scrOff = 0;
    for (const Scratch& s : scr)
        if (nr.gmScratchInAcc)
        {
            *s.p = (uint8_t*)nr.dGmAcc + scrOff;
            scrOff += (s.bytes + 255) & ~(size_t)255;
        }
        else
            NCHK(cudaMalloc(s.p, s.bytes), s.what);
    LOG("native: gmfss chain: pair scratch %.0f MB %s\n", (double)scrB / 1048576.0,
        nr.gmScratchInAcc ? "inside the splat accumulator" : "in its own buffers");
    for (int i = 0; i < 2; i++)
    {
        // map4 / map8: both frames' [2, 128, H, W] fp16 at the quarter / the eighth
        const size_t bytes = (size_t)2 * 128 * (i ? q4 / 4 : q4) * sizeof(uint16_t);
        NCHK(cudaMalloc(&nr.dGmMap[i], bytes), "alloc gmfss backbone maps");
        nr.gmMapSlot[i] = bytes / 2;
    }
    nr.gmMapsNext = false;
    if (nr.encPost)
    {
        NCHK(cudaMalloc((void**)&nr.dGmRange, sizeof(int)), "alloc gmfss range flag");
        NCHK(cudaHostAlloc((void**)&nr.hGmRange, sizeof(int), cudaHostAllocDefault), "alloc gmfss range readback");
        nr.gmFrameSdr[0] = nr.gmFrameSdr[1] = nr.gmFeatSdr[0] = nr.gmFeatSdr[1] = nr.gmPairSdr = false;
        nr.gmModeLogs = 0;
    }
    cudaMemGetInfo(&freeA, &totB);
    LOG("native: gmfss chain: half %dx%d, features %d/%d/%d %s, metric %s, fusion features %s, out %s, accumulator"
        " %zu planes, buffers %.0f MB\n",
        hw, hh, nr.gmC[0], nr.gmC[1], nr.gmC[2], nr.gmFeatHalf ? "fp16" : "fp32", nr.gmMetricHalf ? "fp16" : "fp32",
        nr.gmFuseHalf ? "fp16" : "fp32", nr.gmOutHalf ? "fp16" : "fp32", accPlanes,
        ((double)freeB - (double)freeA) / 1048576.0);
    LOG("native: gmfss chain: fusion tail %s\n",
        !nr.gmTail     ? "in the engine"
        : nr.gmTailHwc ? "on the host (k_fuTailNhwc: pixel shuffle + conv_last, fp32; out channels-last)"
                       : "on the host (k_fuTail: pixel shuffle + conv_last, fp32)");
    LOG("native: gmfss chain: feature splats %s\n",
        nr.gmFeatHwc ? "channels-last (k_gsGatherNhwc: feat_ext's levels and fusionnet's b / c / d NHWC)" : "planar");
    LOG("native: gmfss chain: fusionnet's input LeakyRelu in the feature splats' stores (slopes %.9g / %.9g / %.9g)\n",
        nr.gmLr[0], nr.gmLr[1], nr.gmLr[2]);
    LOG("native: gmfss chain: gmflow split, its local correlation in one kernel (%dx%d, %d samples a pixel)\n", nr.gmW,
        nr.gmH, nr.gmK);
    LOG("native: gmfss chain: the global matching by k_attn2 (%d tokens a frame)\n", nr.gmN8);
    nr.gmProf = GetEnvironmentVariableW(L"SMV_LIVE_GMFSS_PROF", nullptr, 0) != 0;
    if (nr.gmProf)
        for (auto& e : nr.gmEv)
            NCHK(cudaEventCreate(&e), "create gmfss profile event");
    nr.gmFeatLazy = lkEnv("SMV_GM_FEAT_LAZY") != "0";
    nr.gmFeatOk[0] = nr.gmFeatOk[1] = nr.gmCatchUp = false;
    nr.gmFeatSkips = nr.gmCatchUps = 0;
    if (!g_offline && nr.gmFeatLazy)
        for (auto& e : nr.gmCatchEv)
            NCHK(cudaEventCreate(&e), "create gmfss catch-up event");
    LOG("native: gmfss chain: %s\n", nr.gmFeatLazy
                                         ? "feat_ext only for pairs that make tweens (the next such pair takes a "
                                           "skipped frame's)"
                                         : "feat_ext on every new frame (SMV_GM_FEAT_LAZY=0)");
    return true;
}

// fp32 -> fp16 bits rounded to nearest even, the value the graph's own Cast (and f2h's cvt.rn) gives:
// the constant timestep fills of an fp16 timestep (subnormal halves and the overflow to infinity included)
static uint16_t halfBitsRne(float f)
{
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t s = (x >> 16) & 0x8000u, a = x & 0x7fffffffu;
    if (a > 0x7f800000u)
        return (uint16_t)(s | 0x7e00u); // NaN
    if (a >= 0x477ff000u)
        return (uint16_t)(s | 0x7c00u); // 65520 and up round to infinity
    if (a < 0x38800000u)
    {
        // below 2^-14: a subnormal half, in units of 2^-24; up to 2^-25 rounds to zero (the tie to
        // the even zero)
        if (a <= 0x33000000u)
            return (uint16_t)s;
        const uint32_t m = 0x800000u | (a & 0x7fffffu), sh = 126u - (a >> 23);
        const uint32_t rem = m & ((1u << sh) - 1u), halfway = 1u << (sh - 1u);
        uint32_t h = m >> sh;
        if (rem > halfway || (rem == halfway && (h & 1u)))
            h++;
        return (uint16_t)(s | h);
    }
    // a normal half: the exponent rebiased (127 -> 15), the 13 dropped bits rounded to nearest even
    // (a carry out of the mantissa steps the exponent, which is the right result)
    const uint32_t r = a - 0x38000000u;
    return (uint16_t)(s | ((r + 0x0fffu + ((r >> 13) & 1u)) >> 13));
}

// timestep plane j of the IFNet's batch: a constant map of t in the engine's timestep dtype; a plane that already
// holds this t is left as it is (x2 asks for 0.5 every group)
static bool nativeFillT(NativeRife& nr, size_t j, float t, size_t plane, cudaStream_t st)
{
    unsigned int bits;
    memcpy(&bits, &t, 4);
    const bool track = j < 64;
    const uint64_t bit = track ? 1ull << j : 0;
    if (track && (nr.tHeldMask & bit) && nr.tHeld[j] == bits)
        return true;
    const bool ok = nr.tHalf ? cuMemsetD16Async((CUdeviceptr)((uint16_t*)nr.dT + j * plane), halfBitsRne(t), plane,
                                                (CUstream)st) == CUDA_SUCCESS
                             : cuMemsetD32Async((CUdeviceptr)((float*)nr.dT + j * plane), bits, plane, (CUstream)st) ==
                                   CUDA_SUCCESS;
    if (ok && track)
    {
        nr.tHeld[j] = bits;
        nr.tHeldMask |= bit;
    }
    else
        nr.tHeldMask &= ~bit;
    return ok;
}

// the IFNet's timestep, flow and mask at the model's frame (the timestep sized by the dtype read
// off the engine) and the tweens RIFE's last step writes at the pictures' layout (fp32, fp16 when outHalf): batchMax
// entries each
// (DRBA: one, every tween is its own enqueue with its own map)
static bool nativeTweenBuffers(NativeRife& nr)
{
    const size_t plane = (size_t)nr.ph * nr.pw, b = nr.drba ? 1 : (size_t)(nr.batchMax < 1 ? 1 : nr.batchMax);
    const size_t mplane = nr.mph ? (size_t)nr.mph * nr.mpw : plane;
    // the features in the IFNet's own dtype (f0 / f1, DRBA's encode ring), the encode's fp16 output only where it is
    // widened for an fp32 IFNet
    const size_t fb = nr.featHalf ? sizeof(uint16_t) : sizeof(float);
    if (nr.drba)
        for (int i = 0; i < 4; i++)
            NCHK(cudaMalloc((void**)&nr.dDrF[i], kRifeFeatC * mplane * fb), "alloc drba encode ring");
    else
    {
        NCHK(cudaMalloc((void**)&nr.dF[0], kRifeFeatC * mplane * fb), "alloc f0");
        NCHK(cudaMalloc((void**)&nr.dF[1], kRifeFeatC * mplane * fb), "alloc f1");
    }
    NCHK(cudaMalloc(&nr.dT, b * mplane * (nr.tHalf ? sizeof(uint16_t) : sizeof(float))), "alloc timestep");
    nr.tHeldMask = 0;
    NCHK(cudaMalloc((void**)&nr.dFlow, b * 4 * mplane * sizeof(uint16_t)), "alloc flow");
    NCHK(cudaMalloc((void**)&nr.dMask, b * mplane * sizeof(uint16_t)), "alloc mask");
    nr.blendOut = !g_offline && !nr.gmfss && nr.w == nr.dw && nr.h == nr.dh && !nr.uw && !drbaAuthorWanted(nr);
    if (nr.blendOut)
        LOG("native: RIFE's last step stores each tween straight into its slot (the %dx%d model frame is the "
            "presented one)\n",
            nr.w, nr.h);
    else
    {
        NCHK(cudaMalloc((void**)&nr.dMerged, b * 3 * plane * (nr.outHalf ? sizeof(uint16_t) : sizeof(float))),
             "alloc tweens");
        if (nr.outHalf)
            LOG("native: RIFE's tweens are stored in fp16 (SMV_TWEEN_FP16=1)\n");
    }
    return true;
}

// k_motionIn's HDR curve, in units of SDR white: up to the knee the light passes exactly (what an SDR
// source hands RIFE; the knee sits k_hdrRange's tolerance above white, where the decode of SDR white's
// 16-bit PQ code can land), above it PQ's spacing with 10000 nits at kMotionPqCap above the knee's
// sRGB, so every stop of highlight keeps a similar share of contrast while the values stay near the
// range RIFE was trained on (measured on exact pans of highlight-heavy pictures: a smaller cap squeezes
// the highlights' motion, a larger one takes RIFE out of its range; 0.75 measured best)
constexpr float kMotionPqKnee = 1.002f, kMotionPqCap = 0.75f;
// GMFSS's motion view on a frame pair outside the SDR range (hdrToModel): the SDR range exactly, the
// light above white rolled off toward 5 x white
constexpr float kMotionKnee = 1.0f, kMotionHead = 4.0f;

// GMFSS and Restore on HDR planes (hdrToModel, the kernels' comment): pre = the decoded frame's encoding, post = the
// model frames' (1 PQ, 2 HLG, 0 SDR planes), white = SDR white in scRGB units. SMV_HDR_MODEL_ENC=0 = both read the HDR
// codes as they are; SMV_GMFSS_SDR_PAIRS=0 = GMFSS treats every pair as one outside the SDR range.
static void nativeHdrModelSetup(NativeRife& nr, int pre, int post, float white)
{
    wchar_t v[8]{};
    const bool off = GetEnvironmentVariableW(L"SMV_HDR_MODEL_ENC", v, 8) && v[0] == L'0';
    nr.encPre = off ? 0 : pre;
    nr.encPost = off ? 0 : post;
    nr.encWhite = white;
    nr.gmSdrPairs = !(GetEnvironmentVariableW(L"SMV_GMFSS_SDR_PAIRS", v, 8) && v[0] == L'0');
    nr.shHdrView = !(GetEnvironmentVariableW(L"SMV_HDR_SHARPEN_VIEW", v, 8) && v[0] == L'0');
    nr.rsHdrView = !(GetEnvironmentVariableW(L"SMV_HDR_RESIZE_VIEW", v, 8) && v[0] == L'0');
    if (nr.encPre || nr.encPost)
        LOG("native: HDR planes: Restore reads their SDR view over %.0f nits and gets back the light outside it; %s\n",
            white * 80.0,
            nr.gmSdrPairs ? "GMFSS reads it on frame pairs inside the SDR range, its motion nets on any other pair"
                          : "GMFSS's motion nets read it on every pair (SMV_GMFSS_SDR_PAIRS=0)");
}

// one planar frame (w x h, strides in elements) into its SDR view (fp32, or fp16 when dstHalf; hdrToModel's knee and
// head given), and a GMFSS tween of an SDR pair back to the HDR codes, on the render stream; false = the launch failed
static bool nativeHdrEnc(NativeRife& nr, const void* src, int half, int sps, int srs, int w, int h, void* dst,
                         int dstHalf, int dps, int drs, int mode, float knee, float head)
{
    void* a[] = {(void*)&src, &half, &sps, &srs, &w, &h, &dst, &dstHalf, &dps, &drs, &mode, &nr.encWhite, &knee, &head};
    return cuLaunchKernel(nr.fHdrEnc, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)nr.stream, a, nullptr) ==
           CUDA_SUCCESS;
}
static bool nativeHdrFromSdr(NativeRife& nr, const void* src, int half, int sps, int srs, int w, int h, float* dst,
                             int dps, int drs, int mode)
{
    void* a[] = {(void*)&src, &half, &sps, &srs, &w, &h, (void*)&dst, &dps, &drs, &mode, &nr.encWhite};
    return cuLaunchKernel(nr.fHdrFromSdr, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)nr.stream, a,
                          nullptr) == CUDA_SUCCESS;
}
// a resize's source frame (HDR planes, fp32, w x h, planes ps apart, rows rs apart): 1 inside the SDR range (k_hdrRange,
// read back), 0 outside, -1 when a launch or the readback failed; the mode is logged on the first frame and on up to 3
// switches
static int nativeRsInRange(NativeRife& nr, const float* src, int ps, int rs, int w, int h, int mode)
{
    cudaStream_t st = nr.stream;
    float tol = 2e-3f;
    void* a[] = {(void*)&src, &ps, &rs, &w, &h, &mode, &nr.encWhite, &tol, (void*)&nr.dRsRange};
    if (cudaMemsetAsync(nr.dRsRange, 0, sizeof(int), st) != cudaSuccess ||
        cuLaunchKernel(nr.fHdrRange, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a, nullptr) !=
            CUDA_SUCCESS ||
        cudaMemcpyAsync(nr.hRsRange, nr.dRsRange, sizeof(int), cudaMemcpyDeviceToHost, st) != cudaSuccess ||
        cudaStreamSynchronize(st) != cudaSuccess)
        return -1;
    const bool in = *nr.hRsRange == 0;
    if (nr.rsModeLogs < 4 && (nr.rsModeLogs == 0 || in != nr.rsLastView))
    {
        LOG("native: resize: %s\n", in ? "frames inside the SDR range, resized as their SDR view"
                                       : "a frame outside the SDR range, resized on "
                                         "the codes");
        nr.rsModeLogs++;
    }
    nr.rsLastView = in;
    return in ? 1 : 0;
}
// one planar Lanczos resize: the separable pair, k_fitAaH into dTmp (3, sh, tw) then k_fitAaV (a size kept on
// both axes = an exact copy); dTmp holds the session's largest such pass (tmpN floats: live nativeCudaInitEarly,
// offline runOfflineSession). false = dTmp cannot hold the pass or a launch failed.
static bool nativeFitPlanar(NativeRife& nr, const void* src, int half, int ps, int rs, int sw, int sh, float* dst,
                            int tw, int th, cudaStream_t st)
{
    if (!nr.dTmp || (size_t)3 * sh * tw > nr.tmpN)
    {
        LOG("native: the resize pass buffer cannot hold %dx%d -> %dx%d\n", sw, sh, tw, th);
        return false;
    }
    const void* tabH = rsTabFor(nr, sw, tw, st);
    const void* tabV = rsTabFor(nr, sh, th, st);
    if (!tabH || !tabV)
    {
        LOG("native: the resize tap tables for %dx%d -> %dx%d failed\n", sw, sh, tw, th);
        return false;
    }
    void* ah[] = {(void*)&src, &half, &ps, &rs, &sw, &sh, &nr.dTmp, &tw, (void*)&tabH};
    void* av[] = {&nr.dTmp, &tw, &sh, &dst, &th, (void*)&tabV};
    return cuLaunchKernel(nr.fFitAaH, (tw + 15) / 16, (sh + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, ah, nullptr) ==
               CUDA_SUCCESS &&
           cuLaunchKernel(nr.fFitAaV, (tw + 15) / 16, (th + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, av, nullptr) ==
               CUDA_SUCCESS;
}

// the resize of an SDR-range frame: its SDR view (knee 1, head 0) into dRsView, Lanczos (nativeFitPlanar) to tw x th
// into dst clamped at SDR white as on the SDR route, back to the HDR codes in place; false = a launch failed
static bool nativeViewFit(NativeRife& nr, const void* src, int half, int ps, int rs, int sw, int sh, float* dst, int tw,
                          int th, int mode)
{
    cudaStream_t st = nr.stream;
    int vps = sw * sh, n = 3 * tw * th;
    void* b[] = {(void*)&dst, &n};
    return nativeHdrEnc(nr, src, half, ps, rs, sw, sh, nr.dRsView, 0, vps, sw, mode, 1.0f, 0.0f) &&
           nativeFitPlanar(nr, nr.dRsView, 0, vps, sw, sw, sh, dst, tw, th, st) &&
           cuLaunchKernel(nr.fClamp01, (n + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, b, nullptr) == CUDA_SUCCESS &&
           nativeHdrFromSdr(nr, dst, 0, tw * th, tw, tw, th, dst, tw * th, tw, mode);
}

// RIFE's last step (k_rifeBlendF, k_rifeBlend when outHalf) for the n tweens of the last IFNet enqueue, into dMerged:
// pic = frame 0 in planes 0..2 and frame 1 in 3..5 at the pictures' layout (ph x pw), fp32 or fp16 by
// half, or the reverse (sw). The model's frame is the pictures' (the engine's own x: the engine's warp clamps at its
// padded frame) unless mph is set (RIFE's two domains: the flow is read at the picture size and a
// tap clamps at the picture's content). zeroPad: a tap in the picture's pad reads 0 (the author's zero pad).
static bool nativeRifeBlend(NativeRife& nr, const void* pic, int half, int n, cudaStream_t st, int sw = 0)
{
    int pps = nr.ph * nr.pw, prs = nr.pw, w = nr.w, h = nr.h;
    int mps = nr.mph ? nr.mph * nr.mpw : pps, mrs = nr.mph ? nr.mpw : nr.pw;
    int fw = nr.mph ? nr.mw : w, fh = nr.mph ? nr.mh : h;
    int cw = (fw == w && fh == h) ? nr.pw : w, ch = (fw == w && fh == h) ? nr.ph : h;
    int zp = nr.zeroPad ? 1 : 0;
    void* a[] = {(void*)&pic, &sw,       &half, &pps, &prs, &w,  &h,          &cw, &ch,
                 &nr.dFlow,   &nr.dMask, &mps,  &mrs, &fw,  &fh, &nr.dMerged, &zp};
    return cuLaunchKernel(nr.outHalf ? nr.fRifeBlend : nr.fRifeBlendF, (w + 15) / 16, (h + 15) / 16, (unsigned)n, 16,
                          16, 1, 0, (CUstream)st, a, nullptr) == CUDA_SUCCESS;
}

// blendOut: RIFE's last step for tween k of the last IFNet enqueue straight into its output slot
// (k_rifeBlendOut, the content rect at (x0, y0) as k_packOut / k_packOutHdr store it)
static bool nativeRifeBlendOut(NativeRife& nr, const void* pic, int half, int k, uint8_t* slot, cudaStream_t st,
                               int sw = 0)
{
    int pps = nr.ph * nr.pw, prs = nr.pw, w = nr.w, h = nr.h;
    int mps = nr.mph ? nr.mph * nr.mpw : pps, mrs = nr.mph ? nr.mpw : nr.pw;
    int fw = nr.mph ? nr.mw : w, fh = nr.mph ? nr.mh : h;
    int cw = (fw == w && fh == h) ? nr.pw : w, ch = (fw == w && fh == h) ? nr.ph : h;
    const uint16_t* flow = nr.dFlow + (size_t)k * 4 * mps;
    const uint16_t* mask = nr.dMask + (size_t)k * mps;
    int hdr = nr.hdr ? 1 : 0, pitch = (int)nr.pitch, zp = nr.zeroPad ? 1 : 0;
    void* a[] = {(void*)&pic, &sw,  &half, &pps, &prs, &w,    &h,     &cw,    &ch,    (void*)&flow, (void*)&mask,
                 &mps,        &mrs, &fw,   &fh,  &hdr, &slot, &pitch, &nr.x0, &nr.y0, &zp};
    void* a32[] = {(void*)&pic, &sw,  &pps, &prs, &w,   &h,           &cw,    &ch,    (void*)&flow, (void*)&mask,
                   &mps,        &mrs, &fw,  &fh,  &hdr, (void*)&slot, &pitch, &nr.x0, &nr.y0,       &zp};
    return cuLaunchKernel(half ? nr.fRifeBlendOut : nr.fRifeBlendOut32, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0,
                          (CUstream)st, half ? a : a32, nullptr) == CUDA_SUCCESS;
}

static bool nativeModelInit(NativeRife& nr)
{
    // every net runs on the host's kernels: per session their modules and weights (the ready steps below), the tween
    // buffers and the shared activation block; the resident host keeps the dtypes across sessions
    // Restore on the host's kernels: the bundled Real-ESRGAN or AnimeJaNai's form (restNatSetup)
    nr.restNatKind = nr.restorePath == kRestNative      ? 1
                     : nr.restorePath == kRestNativeAj  ? 2
                     : nr.restorePath == kRestNativeA4k ? 3
                                                        : 0;
    nr.restNat = nr.restNatKind != 0;
    // RIFE's IFNet on the host's kernels: no engine either
    nr.ifnNat = nr.ifnetPath == kIfnNative;
    auto restoreReady = [&]() -> bool {
        if (nr.restNat)
            return restNatSetup(nr);
        if (nr.restorePath.empty())
            return true;
        LOG("native: the Restore model %s does not run on the host's kernels\n", nr.restorePath.c_str());
        return false;
    };
    // native DRBA: block0's module and weights (nativeB0Module), per session; it reads the IFNet's encode ring in the
    // IFNet's feature dtype
    auto block0Ready = [&]() -> bool {
        if (!nr.drba)
            return true;
        std::string why;
        if (!nativeB0Module(nr, why))
        {
            LOG("native: drba: block0 unavailable (%s)\n", why.c_str());
            return false;
        }
        LOG("native: drba: block0 on the host's kernels\n");
        return true;
    };
    // RIFE's IFNet: its module and weights (nativeIfnModule), per session
    auto ifnReady = [&]() -> bool {
        if (!nr.ifnNat)
            return true;
        std::string why;
        if (!nativeIfnModule(nr, why))
        {
            LOG("native: rife: the IFNet unavailable (%s)\n", why.c_str());
            return false;
        }
        LOG("native: rife: the IFNet on the host's kernels\n");
        nr.ifnGraphs.mode = chainGraphMode("SMV_IFN_GRAPH");
        return true;
    };
    // the RIFE family's nets on one activation block, set once they all exist
    const bool modelShare = lkEnv("SMV_CTXSHARE") != "0";
    auto shareReady = [&]() -> bool { return !modelShare || nativeModelShareMemory(nr); };
    // GMFSS: every net on the host's kernels. nativeGmfssSetup allocates the chain's buffers (per session, freed in
    // nativeFree).
    if (nr.gmfss)
    {
        const int64_t t0 = nowQpc100();
        size_t freeB = 0, freeA = 0, totB = 0;
        cudaMemGetInfo(&freeB, &totB);
        const bool gmShare = lkEnv("SMV_GM_CTXSHARE") != "0";
        if (gmShare && !nativeGmShareMemory(nr))
            return false;
        if (!restoreReady())
            return false;
        cudaMemGetInfo(&freeA, &totB);
        LOG("native: gmfss ready in %.2f s, VRAM %+.0f MB\n", (nowQpc100() - t0) / 1e7,
            ((double)freeB - (double)freeA) / 1048576.0);
        if (!nativeGmfssSetup(nr))
            return false;
        if (g_resident)
        {
            g_res.ready = true; // the host stays resident after a GMFSS session too (no IFNet to reuse)
            g_res.dev = nr.dev;
            g_res.ifnetPath.clear();
        }
        return true;
    }
    if (g_resident && g_res.ready && g_res.ifnetPath == nr.ifnetPath)
    {
        const int64_t t0 = nowQpc100();
        nr.featHalf = g_res.featHalf;
        nr.xHalf = g_res.xHalf;
        nr.outHalf = g_res.outHalf;
        nr.tHalf = g_res.tHalf;
        if (!nr.noEngine && !nr.nvof && !nr.fruc && !nr.dlssg && !nativeTweenBuffers(nr))
            return false;
        if (!block0Ready() || !ifnReady() || !shareReady() || !restoreReady())
            return false;
        LOG("native: resident engines reused, contexts recreated in %.2f s\n", (nowQpc100() - t0) / 1e7);
        return true;
    }
    if (g_resident && g_res.ready)
        residentFreeEngines(); // a different pair (new window size)
    if (!nr.noEngine && !nr.nvof && !nr.fruc && !nr.dlssg)
    {
        if (!nr.ifnNat)
        {
            LOG("native: rife: the handoff named no IFNet on the host's kernels (%s)\n", nr.ifnetPath.c_str());
            return false;
        }
        // the native IFNet's dtypes: the frame pair x, the timestep (the constant fills and DRBA's map are written in
        // fp16) and the encodes f0 / f1 (nativeEncodeRun writes that dtype) in fp16; flow / mask out in fp16, the tween
        // made by RIFE's last step (nativeRifeBlend)
        nr.tHalf = nr.xHalf = nr.featHalf = true;
        // the tweens in fp32, each reader rounds once; SMV_TWEEN_FP16=1 = the fp16 store (a second rounding)
        nr.outHalf = lkEnv("SMV_TWEEN_FP16") == "1";
        if (!nativeTweenBuffers(nr))
            return false;
    }
    if (!g_offline && nr.mph)
        LOG("native: RIFE finds the motion on the picture at %dx%d%s, its warp + blend runs on the %dx%d pictures\n",
            nr.mw, nr.mh,
            nr.hdr ? (nr.rtxHdr ? " (the SDR picture before RTX HDR)" : " (the HDR desktop's PQ read as SDR sRGB)")
                   : "",
            nr.w, nr.h);
    if (!block0Ready() || !ifnReady() || !shareReady() || !restoreReady())
        return false;
    if (g_resident)
    {
        g_res.ready = true;
        g_res.featHalf = nr.featHalf;
        g_res.xHalf = nr.xHalf;
        g_res.outHalf = nr.outHalf;
        g_res.tHalf = nr.tHalf;
        g_res.dev = nr.dev;
        g_res.ifnetPath = nr.ifnetPath;
    }
    return true;
}

static void nativeFree(NativeRife& nr)
{
    if (nr.stream)
        cudaStreamSynchronize(nr.stream);
    nativeNvofFree(nr);     // the Optical Flow session and its buffers (nvof sessions only)
    nativeFrucFree(nr);     // the bridge's FRUC instance and its surfaces (fruc sessions only)
    nativeFsrupRelease(nr); // the FSR upscaler and its vector session (fsrUp sessions only)
    nativeA4kRelease(nr);   // Anime4K's buffers: the resize's, Restore's, the statistics (Anime4K sessions only)
    nativeAcnRelease(nr);   // ArtCNN's weights and frame buffers (ArtCNN sessions only)
    // release the TrueHDR feature on the compute thread, before the CUDA teardown
    if (nr.thdrN && !g_offline) // offline logs its own line with the light levels
        LOG("native: TrueHDR eval %.2f ms mean, %.2f ms max, over %llu real frames\n", nr.thdrMs / (double)nr.thdrN,
            nr.thdrMaxMs, (unsigned long long)nr.thdrN);
    if (nr.vsrN && nr.vsrAsync)
        LOG("native: RTX VSR eval ordered by events over %llu %s frames (SMV_VSR_SYNC=1 times each call)\n",
            (unsigned long long)nr.vsrN,
            g_offline ? (nr.nvPre && !nr.vsrPost ? "decoded" : "output") : (nr.vsrPre ? "captured" : "presented"));
    else if (nr.vsrN)
        LOG("native: RTX VSR eval %.2f ms mean, %.2f ms max, over %llu %s frames\n", nr.vsrMs / (double)nr.vsrN,
            nr.vsrMaxMs, (unsigned long long)nr.vsrN,
            g_offline ? (nr.nvPre && !nr.vsrPost ? "decoded" : "output") : (nr.vsrPre ? "captured" : "presented"));
    if (nr.liveNr)
        LOG("native: live DLSS 5 %.2f ms mean, %.2f ms max over %llu captured frames (host submit, zero-copy), "
            "its last output reused on %llu identical captures\n",
            nr.nrN ? nr.nrMs / (double)nr.nrN : 0.0, nr.nrMaxMs, (unsigned long long)nr.nrN,
            (unsigned long long)nr.nrReused);
    if (g_rtxb.created && g_rtxb.shutdown)
    {
        g_rtxb.shutdown();
        g_rtxb.created = g_rtxb.vsrMade = false;
    }
    if (nr.drTweens || nr.drHeads)
        LOG("native: drba %llu tweens (%llu head pairs), %llu block0 runs\n", (unsigned long long)nr.drTweens,
            (unsigned long long)nr.drHeads, (unsigned long long)nr.drBlock0);
    for (void* p : {(void*)nr.dDrIBase, (void*)nr.dDrF[0], (void*)nr.dDrF[1], (void*)nr.dDrF[2], (void*)nr.dDrF[3],
                    (void*)nr.dDrX[0], (void*)nr.dDrX[1], (void*)nr.drWin[0].f10, (void*)nr.drWin[0].r,
                    (void*)nr.drWin[1].f10, (void*)nr.drWin[1].r, (void*)nr.dDrAcc, (void*)nr.dDrMBase, (void*)nr.dDrS,
                    (void*)(nr.drRtOwn ? nr.dDrRT : nullptr)})
        if (p)
            cudaFree(p);
    nr.dDrIBase = nullptr;
    nr.dDrMBase = nullptr;
    nr.dDrS = nr.dDrB = nr.dDrRT = nullptr;
    nr.drRtOwn = false;
    nr.drAuthor = false;
    nr.drSFid[0] = nr.drSFid[1] = -1;
    for (int i = 0; i < 4; i++)
    {
        nr.dDrI[i] = nr.dDrF[i] = nullptr;
        nr.dDrM[i] = nullptr;
    }
    nr.dDrX[0] = nr.dDrX[1] = nullptr;
    nr.dDrMX[0] = nr.dDrMX[1] = nullptr;
    for (auto& wn : nr.drWin)
        wn = NativeRife::DrWin{};
    nr.dDrAcc = nullptr;
    if (nr.dGmCtxMem) // after the contexts that ran on it
    {
        cudaFree(nr.dGmCtxMem);
        nr.dGmCtxMem = nullptr;
        nr.gmCtxMemB = 0;
    }
    for (auto& e : nr.slotEv)
        if (e)
            cudaEventDestroy(e);
    nr.slotEv.clear();
    for (auto& e : nr.gmEv)
        if (e)
        {
            cudaEventDestroy(e);
            e = nullptr;
        }
    for (auto& e : nr.gmCatchEv)
        if (e)
        {
            cudaEventDestroy(e);
            e = nullptr;
        }
    if (nr.gmFeatSkips)
        LOG("native: gmfss: feat_ext skipped on %u frames without tweens, %u taken by a later pair\n", nr.gmFeatSkips,
            nr.gmCatchUps);
    nr.gmFeatSkips = nr.gmCatchUps = 0;
    nr.gmCatchUp = false;
    if (nr.capEv)
    {
        cudaEventDestroy(nr.capEv);
        nr.capEv = nullptr;
    }
    for (cudaEvent_t& e : nr.pfEv)
        if (e)
        {
            cudaEventDestroy(e);
            e = nullptr;
        }
    for (cudaEvent_t& e : nr.grEv)
        if (e)
        {
            cudaEventDestroy(e);
            e = nullptr;
        }
    for (auto* s : {&nr.fgL1Ev, &nr.fgOfEv})
        for (auto& set : *s)
            for (cudaEvent_t& e : set)
                if (e)
                {
                    cudaEventDestroy(e);
                    e = nullptr;
                }
    if (nr.grLagEv)
    {
        cudaEventDestroy(nr.grLagEv);
        nr.grLagEv = nullptr;
    }
    nr.grArmed = nr.grLagArmed = false;
    for (int k = 0; k < 2; k++)
        nr.grL1Armed[k] = nr.grOfArmed[k] = false;
    // PG p.124: mappings must go before the external memory objects, and every outstanding
    // wait must have completed before the semaphore is destroyed (the stream sync above).
    if (nr.dOutRing)
    {
        cudaFree(nr.dOutRing);
        nr.dOutRing = nullptr;
    }
    if (nr.capMip)
    {
        cudaFreeMipmappedArray(nr.capMip);
        nr.capMip = nullptr;
        nr.capArr = nullptr;
    }
    if (nr.semCap)
    {
        cudaDestroyExternalSemaphore(nr.semCap);
        nr.semCap = nullptr;
    }
    if (nr.emCap)
    {
        cudaDestroyExternalMemory(nr.emCap);
        nr.emCap = nullptr;
    }
    if (nr.emOut)
    {
        cudaDestroyExternalMemory(nr.emOut);
        nr.emOut = nullptr;
    }
    for (void* p : {(void*)nr.dFlow, (void*)nr.dMask, (void*)nr.dM, (void*)nr.dMFit, (void*)nr.dRestRem,
                    (void*)nr.dRsView, (void*)nr.dRsRange})
        if (p)
            cudaFree(p);
    if (nr.hRsRange)
        cudaFreeHost(nr.hRsRange);
    nr.dFlow = nr.dMask = nr.dM = nullptr;
    nr.dMFit = nr.dRestRem = nr.dRsView = nullptr;
    nr.dRsRange = nr.hRsRange = nullptr;
    nr.rsModeLogs = 0;
    for (const NativeRife::RsTab& e : nr.rsTabs)
        cudaFree(e.d);
    nr.rsTabs.clear();
    for (void* p :
         {(void*)nr.dCap,    (void*)nr.dX,          (void*)nr.dXh,        (void*)nr.dF[0],   (void*)nr.dF[1],
          (void*)nr.dEncOwn, (void*)nr.dT,          (void*)nr.dMerged,    (void*)nr.dTmp,    (void*)nr.dCapF,
          (void*)nr.dThdrIn, (void*)nr.dThdrOut,    (void*)nr.dSrcG,      (void*)nr.dPres,   (void*)nr.dSrcPl,
          (void*)nr.dVsrIn,  (void*)nr.dVsrOut,     (void*)nr.dVsrView,   (void*)nr.dVsrRem, (void*)nr.dFitTmp,
          (void*)nr.dUp,     (void*)nr.dUpTmp,      nr.dRestIn,           nr.dRestOut,       (void*)nr.dRestTmp,
          (void*)nr.dRest,   (void*)nr.dStaticFlag, (void*)nr.dThdrStats, (void*)nr.dShIn,   (void*)nr.dShOut})
        if (p)
            cudaFree(p);
    if (nr.restNatOwn && nr.dRestA) // else the activations sit in the shared activation block
        cudaFree(nr.dRestA);
    if (nr.dRestW)
        cudaFree(nr.dRestW);
    nr.dRestA = nr.dRestB = nr.dRestW = nullptr;
    nr.restNat = nr.restNatOwn = false;
    nr.restNatKind = 0;
    if (nr.hStaticFlag)
    {
        cudaFreeHost(nr.hStaticFlag);
        nr.hStaticFlag = nullptr;
    }
    if (nr.hThdrStats)
    {
        cudaFreeHost(nr.hThdrStats);
        nr.hThdrStats = nullptr;
    }
    nr.dStaticFlag = nullptr;
    nr.dThdrStats = nullptr;
    nr.thdrStatsPending = false;
    // the GMFSS chain's buffers: per session like every other model buffer, so a
    // backend switch on the resident host gives the VRAM back even though the engines stay
    if (nr.gmScratchInAcc)
    {
        // the pair scratch lives inside dGmAcc: the accumulator's free returns it
        for (auto& p : nr.dGmA)
            p = nullptr;
        for (auto& p : nr.dGmM)
            p = nullptr;
        nr.dGmCorr = nr.dGmFlow8 = nr.dGmEnc = nr.dGmHalfM = nullptr;
        nr.dGmProp = nullptr;
        nr.gmScratchInAcc = false;
    }
    for (void* p : {nr.dGmFeat[0][0],
                    nr.dGmFeat[0][1],
                    nr.dGmFeat[0][2],
                    nr.dGmFeat[1][0],
                    nr.dGmFeat[1][1],
                    nr.dGmFeat[1][2],
                    (void*)nr.dGmHalf,
                    (void*)nr.dGmFlow,
                    nr.dGmMetric,
                    (void*)nr.dGmFlowP[0],
                    (void*)nr.dGmFlowP[1],
                    (void*)nr.dGmMetP[0],
                    (void*)nr.dGmMetP[1],
                    (void*)nr.dGmAcc,
                    (void*)nr.dGmFa,
                    nr.dGmFb,
                    nr.dGmFc,
                    nr.dGmFd,
                    (void*)nr.dGmT,
                    nr.dGmOut,
                    nr.dGmWl,
                    nr.dGmBl,
                    (void*)nr.dGmF,
                    (void*)nr.dGmEnc,
                    (void*)nr.dGmHalfM,
                    (void*)nr.dGmRange,
                    nr.dGmA[0],
                    nr.dGmA[1],
                    (void*)nr.dGmCorr,
                    nr.dGmM[0],
                    nr.dGmM[1],
                    nr.dGmM[2],
                    (void*)nr.dGmFlow8,
                    nr.dGmProp,
                    nr.dGmMap[0],
                    nr.dGmMap[1],
                    (void*)nr.dGsCnt[0],
                    (void*)nr.dGsCnt[1],
                    (void*)nr.dGsOff[0],
                    (void*)nr.dGsOff[1],
                    (void*)nr.dGsList[0],
                    (void*)nr.dGsList[1],
                    (void*)nr.dGsSrc[0],
                    (void*)nr.dGsSrc[1],
                    (void*)nr.dGsBlk[0],
                    (void*)nr.dGsBlk[1],
                    nr.dFnW,
                    nr.dFnOwn,
                    nr.dMnW,
                    nr.dMnOwn,
                    nr.dGiW,
                    nr.dGiOwn,
                    nr.dFuW,
                    nr.dFuOwn,
                    nr.dGbW,
                    nr.dGbOwn,
                    nr.dBbW,
                    nr.dBbOwn,
                    nr.dGaW,
                    nr.dGaOwn,
                    (void*)nr.dGaPos,
                    nr.dB0W,
                    nr.dB0Own,
                    nr.dIfnW,
                    nr.dIfnOwn})
        if (p)
            cudaFree(p);
    nr.dFnW = nr.dFnOwn = nr.dFnScr = nullptr;
    nr.dMnW = nr.dMnOwn = nullptr;
    nr.dGiW = nr.dGiOwn = nullptr;
    nr.dFuW = nr.dFuOwn = nullptr;
    nr.fuOps.clear();
    nr.dGbW = nr.dGbOwn = nullptr;
    nr.dBbW = nr.dBbOwn = nullptr;
    nr.dGaW = nr.dGaOwn = nullptr;
    nr.dGaPos = nullptr;
    nr.dB0W = nr.dB0Own = nullptr;
    nr.dIfnW = nr.dIfnOwn = nullptr;
    for (int s = 0; s < 2; s++)
    {
        nr.dGsCnt[s] = nr.dGsOff[s] = nr.dGsList[s] = nr.dGsBlk[s] = nullptr;
        nr.dGsSrc[s] = nullptr;
    }
    nr.gmGather = false;
    for (auto& p : nr.dGmA)
        p = nullptr;
    for (auto& p : nr.dGmM)
        p = nullptr;
    nr.dGmFlow8 = nullptr;
    nr.dGmProp = nullptr;
    nr.dGmCorr = nullptr;
    nr.dGmMap[0] = nr.dGmMap[1] = nullptr;
    nr.gmMapsNext = false;
    if (nr.hGmRange)
        cudaFreeHost(nr.hGmRange);
    nr.hGmRange = nullptr;
    nr.dGmRange = nullptr;
    for (int s = 0; s < 2; s++)
        for (int l = 0; l < 3; l++)
            nr.dGmFeat[s][l] = nullptr;
    nr.dGmHalf = nullptr;
    nr.dGmFlow = nullptr;
    nr.dGmMetric = nullptr;
    nr.dGmFlowP[0] = nr.dGmFlowP[1] = nullptr;
    nr.dGmMetP[0] = nr.dGmMetP[1] = nullptr;
    nr.dGmAcc = nullptr;
    nr.dGmFa = nullptr;
    nr.dGmFb = nullptr;
    nr.dGmFc = nullptr;
    nr.dGmFd = nullptr;
    nr.dGmT = nullptr;
    nr.dGmOut = nullptr;
    nr.dGmWl = nr.dGmBl = nullptr;
    nr.gmLr[0] = nr.gmLr[1] = nr.gmLr[2] = -1.0f;
    nr.gmTail = nr.gmTailHwc = nr.gmFeatHwc = false;
    nr.dGmF = nullptr;
    nr.dGmEnc = nullptr;
    nr.dGmHalfM = nullptr;
    nr.dCap = nullptr;
    nr.dX = nullptr;
    nr.dXh = nullptr;
    nr.dF[0] = nr.dF[1] = nullptr;
    nr.dEncOwn = nullptr;
    nr.dT = nullptr;
    nr.tHeldMask = 0;
    nr.dMerged = nullptr;
    nr.dTmp = nullptr;
    nr.tmpN = 0;
    nr.dCapF = nullptr;
    nr.dThdrIn = nullptr;
    nr.dThdrOut = nullptr;
    nr.dSrcG = nullptr;
    nr.dShIn = nullptr;
    nr.dShOut = nullptr;
    nr.dPres = nullptr;
    nr.dSrcPl = nullptr;
    nr.dVsrIn = nullptr;
    nr.dVsrOut = nullptr;
    nr.dVsrView = nullptr;
    nr.dVsrRem = nullptr;
    nr.dFitTmp = nullptr;
    nr.dUp = nullptr;
    nr.dUpTmp = nullptr;
    nr.dRestIn = nullptr;
    nr.dRestOut = nullptr;
    nr.dRestTmp = nullptr;
    nr.dRest = nullptr;
    if (nr.cuMod && nr.cuMod != g_res.cuMod)
        cuModuleUnload(nr.cuMod);
    nr.cuMod = nullptr;
    if (nr.a4kMod)
        cuModuleUnload(nr.a4kMod);
    nr.a4kMod = nullptr;
    for (CUfunction& f : nr.fA4k)
        f = nullptr;
    if (nr.acnMod)
        cuModuleUnload(nr.acnMod);
    nr.acnMod = nullptr;
    for (CUfunction& f : nr.fAcn)
        f = nullptr;
    if (nr.fnMod)
        cuModuleUnload(nr.fnMod);
    nr.fnMod = nullptr;
    for (CUfunction& f : nr.fFn)
        f = nullptr;
    if (nr.mnMod)
        cuModuleUnload(nr.mnMod);
    nr.mnMod = nullptr;
    for (CUfunction& f : nr.fMn)
        f = nullptr;
    chainGraphsFree(nr.giGraphs);
    if (nr.giMod)
        cuModuleUnload(nr.giMod);
    nr.giMod = nullptr;
    for (CUfunction& f : nr.fGi)
        f = nullptr;
    chainGraphsFree(nr.fuGraphs);
    if (nr.fuMod)
        cuModuleUnload(nr.fuMod);
    nr.fuMod = nullptr;
    for (CUfunction& f : nr.fFu)
        f = nullptr;
    chainGraphsFree(nr.gbGraphs);
    if (nr.gbMod)
        cuModuleUnload(nr.gbMod);
    nr.gbMod = nullptr;
    for (CUfunction& f : nr.fGb)
        f = nullptr;
    chainGraphsFree(nr.bbGraphs);
    if (nr.bbMod)
        cuModuleUnload(nr.bbMod);
    nr.bbMod = nullptr;
    for (CUfunction& f : nr.fBb)
        f = nullptr;
    for (ChainGraphs& g : nr.gaGraphs)
        chainGraphsFree(g);
    if (nr.gaMod)
        cuModuleUnload(nr.gaMod);
    nr.gaMod = nullptr;
    for (CUfunction& f : nr.fGa)
        f = nullptr;
    if (nr.b0Mod)
        cuModuleUnload(nr.b0Mod);
    nr.b0Mod = nullptr;
    for (CUfunction& f : nr.fB0)
        f = nullptr;
    chainGraphsFree(nr.ifnGraphs);
    if (nr.ifnMod)
        cuModuleUnload(nr.ifnMod);
    nr.ifnMod = nullptr;
    for (CUfunction& f : nr.fIfn)
        f = nullptr;
    nr.ifnNat = false;
    if (nr.stream)
    {
        cudaStreamDestroy(nr.stream);
        nr.stream = nullptr;
    }
}

// ---- the GMFSS chain ------------------------------------------------------------------------
// GMFSS_infer_u.Model.reuse() and .inference() as native launches on the group's stream: the
// four engines, the native nets and the glue kernels (k_half / k_pyr / k_splatSoft / k_splatNorm, gated against a
// float64 reference). Two differences from
// the author's code, both exact:
//   * feat_ext runs ONCE per group (on the new frame) and the previous group's output is this
//     pair's feat0: the same reuse the native RIFE route does with the encode (feat_ext is a
//     pure function of the frame); the author's code recomputes both sets per pair.
//   * the flow and metric pyramids are built once per PAIR and the timestep is applied by the
//     splat: the author's interp(t * flow) * 0.5 equals t * (interp(flow) * 0.5).
// Everything else follows the author's op order literally, including which splat takes s = t and
// which takes s = 1 - t. Shapes come from the engines (nativeGmfssSetup),
// and the warm-up there left every static input shape set on these contexts, so only addresses
// are set here.
static bool nativeGmfssPair(NativeRife& nr, float* dPrev, float* dCur, bool needFlow)
{
    cudaStream_t st = nr.stream;
    const size_t plane = (size_t)nr.ph * nr.pw, hp = (size_t)nr.hh * nr.hw;
    int ps = (int)plane, rs = nr.pw, zero = 0, three = 3, four = 4, two = 2, lvl2 = 2, lvl4 = 4;
    int qw = nr.hw / 2, qh = nr.hh / 2, ew = nr.hw / 4, eh = nr.hh / 4;
    int mHalf = nr.gmMetricHalf ? 1 : 0;
    float s05 = 0.5f, s025 = 0.25f, s1 = 1.0f;
    nr.gmProfTween = false;
    // the CPU cost of enqueueing this pair block (QPC, 100 ns units), taken only
    // on a pair that really runs the flow so the average is over comparable groups
    const bool profPair = nr.gmProf && needFlow;
    const int64_t cpu0 = profPair ? nowQpc100() : 0;
    // the backbone maps in batch slot 1 belong to this call's previous frame only when the last call was a pair
    const bool mapsPrev = nr.gmMapsNext;
    nr.gmMapsNext = false;
    if (profPair)
        cudaEventRecord(nr.gmEv[0], st);
    // feat_ext of the NEW frame, into the set the NEXT pair will read as feat0. HDR planes (the kernels' comment): a
    // pair whose two frames stay inside the SDR range (k_hdrRange, read back here) runs every net on the SDR view
    // (hdrToModel, knee 1, head 0: an SDR source's values) through one scratch frame, its tweens come back through
    // k_hdrFromSdr; any other pair runs on the pictures with gmflow and metricnet on the motion view (kMotionHead's
    // roll-off). A previous frame's features taken in the other mode are taken again in this pair's.
    nr.gmCur ^= 1;
    void** fs = nr.dGmFeat[nr.gmCur];
    bool sdr = false;
    if (nr.encPost && nr.gmSdrPairs)
    {
        float tol = 2e-3f;
        void* ar[] = {(void*)&dCur, &ps, &rs, &nr.pw, &nr.ph, &nr.encPost, &nr.encWhite, &tol, (void*)&nr.dGmRange};
        if (cudaMemsetAsync(nr.dGmRange, 0, sizeof(int), st) != cudaSuccess ||
            cuLaunchKernel(nr.fHdrRange, (nr.pw + 15) / 16, (nr.ph + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, ar,
                           nullptr) != CUDA_SUCCESS ||
            cudaMemcpyAsync(nr.hGmRange, nr.dGmRange, sizeof(int), cudaMemcpyDeviceToHost, st) != cudaSuccess ||
            cudaStreamSynchronize(st) != cudaSuccess)
        {
            nr.die("gmfss range test failed");
            return false;
        }
        nr.gmFrameSdr[nr.gmCur] = *nr.hGmRange == 0;
        sdr = nr.gmFrameSdr[nr.gmCur] && nr.gmFrameSdr[nr.gmCur ^ 1];
        if (needFlow && nr.gmModeLogs < 4 && (nr.gmModeLogs == 0 || sdr != nr.gmPairSdr))
        {
            LOG("native: gmfss: %s\n", sdr ? "frame pairs inside the SDR range, every net on their SDR view"
                                           : "a frame pair outside the SDR range, the motion nets on the motion view");
            nr.gmModeLogs++;
        }
    }
    nr.gmPairSdr = sdr;
    // one frame's SDR view into the scratch (head 0 for an SDR pair, kMotionHead for the motion nets)
    auto view = [&](float* src, float head) {
        return nativeHdrEnc(nr, src, 0, ps, rs, nr.pw, nr.ph, nr.dGmEnc, 0, ps, rs, nr.encPost, kMotionKnee, head);
    };
    auto featExt = [&](float* x, void** f) { return nativeFnRun(nr, x, ps, rs, f, st); };
    float* xCur = sdr ? nr.dGmEnc : dCur;
    if (!needFlow && nr.gmFeatLazy)
    {
        // no tween in this group: its frame's features wait for the next pair that makes tweens, which takes them
        // from dPrev (the same bytes) in its own mode; gmFeatSdr keeps the mode the eager route would have used
        nr.gmFeatOk[nr.gmCur] = false;
        nr.gmFeatSdr[nr.gmCur] = sdr;
        nr.gmFeatSkips++;
        return true;
    }
    if (sdr && !view(dCur, 0.0f))
    {
        nr.die("gmfss SDR view launch failed");
        return false;
    }
    if (!featExt(xCur, fs))
    {
        nr.die("gmfss feat_ext launch failed");
        return false;
    }
    nr.gmFeatSdr[nr.gmCur] = sdr;
    nr.gmFeatOk[nr.gmCur] = true;
    if (!needFlow)
        return true; // no tween in this group: the pair-level work is skipped
    if (profPair)
        cudaEventRecord(nr.gmEv[1], st);
    // both halves into one (6, hh, hw) buffer: the ifnet's x as it is, gmflow's and
    // metricnet's inputs and the two image splats by pointer offset (an SDR pair: the new frame's view first, then
    // the previous frame's takes the scratch for its half and, when they came from the pictures, its features)
    float* h0 = nr.dGmHalf;
    float* h1 = nr.dGmHalf + 3 * hp;
    float* x0 = sdr ? nr.dGmEnc : dPrev;
    void* a0[] = {(void*)&x0, &zero, &ps, &rs, &three, &nr.hw, &nr.hh, (void*)&h0, &zero};
    void* a1[] = {(void*)&xCur, &zero, &ps, &rs, &three, &nr.hw, &nr.hh, (void*)&h1, &zero};
    auto halfOf = [&](void** a) {
        return cuLaunchKernel(nr.fHalf, (nr.hw + 15) / 16, (nr.hh + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a,
                              nullptr) == CUDA_SUCCESS;
    };
    // the previous frame's features again: taken in the other mode, or skipped by its call (a catch-up, timed apart
    // live when the eager route would not have taken them again)
    const bool stale = !nr.gmFeatOk[nr.gmCur ^ 1];
    const bool modeAgain = nr.gmFeatSdr[nr.gmCur ^ 1] != sdr;
    const bool refeat = stale || modeAgain;
    const bool catchUp = stale && !modeAgain && nr.gmCatchEv[0];
    if (!halfOf(a1) || (sdr && !view(dPrev, 0.0f)) || !halfOf(a0) ||
        (catchUp && cudaEventRecord(nr.gmCatchEv[0], st) != cudaSuccess) ||
        (refeat && !featExt(x0, nr.dGmFeat[nr.gmCur ^ 1])) ||
        (catchUp && cudaEventRecord(nr.gmCatchEv[1], st) != cudaSuccess))
    {
        nr.die("gmfss half launch failed");
        return false;
    }
    nr.gmFeatSdr[nr.gmCur ^ 1] = sdr;
    nr.gmFeatOk[nr.gmCur ^ 1] = true;
    nr.gmCatchUp = catchUp;
    if (stale)
        nr.gmCatchUps++;
    // the motion nets: an SDR pair's halves, else (HDR planes) the motion view's
    const bool mview = nr.encPost && !sdr;
    float* mv0 = mview ? nr.dGmHalfM : h0;
    float* mv1 = mview ? nr.dGmHalfM + 3 * hp : h1;
    float* e = nr.dGmEnc;
    void* am0[] = {(void*)&e, &zero, &ps, &rs, &three, &nr.hw, &nr.hh, (void*)&mv0, &zero};
    void* am1[] = {(void*)&e, &zero, &ps, &rs, &three, &nr.hw, &nr.hh, (void*)&mv1, &zero};
    if (mview && (!view(dCur, kMotionHead) || !halfOf(am1) || !view(dPrev, kMotionHead) || !halfOf(am0)))
    {
        nr.die("gmfss motion halves launch failed");
        return false;
    }
    if (profPair)
        cudaEventRecord(nr.gmEv[2], st);
    // gmflow's backbone once a frame: the previous frame's maps are the last pair's new ones (batch slot 1 -> 0) when
    // that pair took them on the same view of it (the pictures, the SDR view or the motion view), else taken again;
    // the new frame's into slot 1
    const int kind = mview ? 2 : (sdr ? 1 : 0);
    auto backbone = [&](float* x, int slot) {
        return nativeBbRun(nr, x, (uint8_t*)nr.dGmMap[0] + slot * nr.gmMapSlot[0],
                           (uint8_t*)nr.dGmMap[1] + slot * nr.gmMapSlot[1], st);
    };
    bool mapsOk = true;
    if (mapsPrev && nr.gmMapsKind == kind)
        for (int i = 0; i < 2 && mapsOk; i++)
            mapsOk = cudaMemcpyAsync(nr.dGmMap[i], (uint8_t*)nr.dGmMap[i] + nr.gmMapSlot[i], nr.gmMapSlot[i],
                                     cudaMemcpyDeviceToDevice, st) == cudaSuccess;
    else
        mapsOk = backbone(mv0, 0);
    if (!mapsOk || !backbone(mv1, 1))
    {
        nr.die("gmfss gmflow backbone's kernels (or the maps' copy) failed to launch");
        return false;
    }
    nr.gmMapsKind = kind;
    // the fused bidir GMFlow (both directions: `_m` on both frames' maps, the scale-0 global matching and its propagation
    // (k_attn2 twice: the matching with the pixel grid as its value writes the flow, the propagation reads it), A, the
    // local correlation (k_localCorr: one warp a pixel, 4 a block) and B) and metricnet on the same halves
    const int64_t cpuF0 = profPair ? nowQpc100() : 0;
    if (!nativeGaRun(nr, 0, nr.dGmMap[1], nullptr, nr.dGmM[0], nullptr, nr.dGmM[1], nr.dGmM[2], st))
    {
        nr.die("gmfss gmflow _m's kernels failed to launch");
        return false;
    }
    {
        float scale2 = 1.44269504f / 11.3137085f; // log2 e / sqrt(128): the softmax in base 2
        int n8 = nr.gmN8, kswapMatch = 1, kswapProp = 0, modeMatch = 3, modeProp = nr.gmPropHalf ? 4 : 0;
        const float* noV = nullptr;
        void* am[] = {(void*)&nr.dGmM[0],
                      (void*)&nr.dGmM[0],
                      (void*)&noV,
                      (void*)&nr.dGmFlow8,
                      &n8,
                      &kswapMatch,
                      &scale2,
                      &modeMatch,
                      &nr.gmW8};
        void* ap[] = {(void*)&nr.dGmM[1],
                      (void*)&nr.dGmM[2],
                      (void*)&nr.dGmFlow8,
                      (void*)&nr.dGmProp,
                      &n8,
                      &kswapProp,
                      &scale2,
                      &modeProp,
                      &nr.gmW8};
        const unsigned gx = (unsigned)((n8 + 63) / 64);
        const CUfunction fa = nr.fAttn2;
        if (cuLaunchKernel(fa, gx, 2, 1, 128, 1, 1, 0, (CUstream)st, am, nullptr) != CUDA_SUCCESS ||
            cuLaunchKernel(fa, gx, 2, 1, 128, 1, 1, 0, (CUstream)st, ap, nullptr) != CUDA_SUCCESS)
        {
            nr.die("gmfss global matching launch failed");
            return false;
        }
    }
    if (!nativeGaRun(nr, 1, nr.dGmMap[0], nr.dGmProp, nr.dGmA[0], nr.dGmA[1], nullptr, nullptr, st))
    {
        nr.die("gmfss gmflow A's kernels failed to launch");
        return false;
    }
    int nb = 2, hw4 = nr.gmH * nr.gmW;
    void* ac[] = {(void*)&nr.dGmA[0], (void*)&nr.dGmCorr, &nb, &nr.gmH, &nr.gmW, &hw4, &nr.gmK};
    if (cuLaunchKernel(nr.fLocalCorr, (unsigned)((2LL * hw4 * 32 + 127) / 128), 1, 1, 128, 1, 1, 0, (CUstream)st, ac,
                       nullptr) != CUDA_SUCCESS)
    {
        nr.die("gmfss local correlation launch failed");
        return false;
    }
    if (!nativeGbRun(nr, st))
    {
        nr.die("gmfss gmflow B's kernels failed to launch");
        return false;
    }
    nr.gmMapsNext = true; // slot 1 = the new frame's maps = the next pair's previous frame
    if (profPair)
        nr.gmCpuFlow += (double)(nowQpc100() - cpuF0) / 10000.0;
    if (profPair)
        cudaEventRecord(nr.gmEv[3], st);
    float* f10 = nr.dGmFlow + 2 * hp;
    void* m1 = (uint8_t*)nr.dGmMetric + hp * (nr.gmMetricHalf ? 2 : 4);
    if (!nativeMnRun(nr, mv0, mv1, nr.dGmFlow, f10, nr.dGmMetric, m1, st))
    {
        nr.die("gmfss metricnet's kernels failed to launch");
        return false;
    }
    if (profPair)
        cudaEventRecord(nr.gmEv[4], st);
    // the pyramids, once per pair: both flow directions in one launch (gmflow writes them as
    // one (2, 2, hh, hw) buffer) and both metrics in one (their planes are adjacent by the
    // addresses above). The flow carries the level's 0.5 / 0.25, a metric is only resampled.
    void* p0[] = {(void*)&nr.dGmFlow, &zero, &four, &nr.hw, &nr.hh, &lvl2, &s05, (void*)&nr.dGmFlowP[0], &qw, &qh};
    void* p1[] = {(void*)&nr.dGmFlow, &zero, &four, &nr.hw, &nr.hh, &lvl4, &s025, (void*)&nr.dGmFlowP[1], &ew, &eh};
    void* p2[] = {(void*)&nr.dGmMetric, &mHalf, &two, &nr.hw, &nr.hh, &lvl2, &s1, (void*)&nr.dGmMetP[0], &qw, &qh};
    void* p3[] = {(void*)&nr.dGmMetric, &mHalf, &two, &nr.hw, &nr.hh, &lvl4, &s1, (void*)&nr.dGmMetP[1], &ew, &eh};
    const int pw2[4] = {qw, ew, qw, ew}, ph2[4] = {qh, eh, qh, eh};
    void** pa[4] = {p0, p1, p2, p3};
    for (int i = 0; i < 4; i++)
        if (cuLaunchKernel(nr.fPyr, (pw2[i] + 15) / 16, (ph2[i] + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, pa[i],
                           nullptr) != CUDA_SUCCESS)
        {
            nr.die("gmfss pyramid launch failed");
            return false;
        }
    if (profPair)
    {
        cudaEventRecord(nr.gmEv[5], st);
        nr.gmCpuPair += (double)(nowQpc100() - cpu0) / 10000.0;
        nr.gmCpuPairN++;
    }
    return true;
}

// one tween at timestep t, into nr.dGmF (the clamped fp32 model frame storeSlot consumes).
// The author's inference(): I1t, I2t, the ifnet, the six feature splats, fusionnet, clamp.
static bool nativeGmfssTween(NativeRife& nr, float t)
{
    cudaStream_t st = nr.stream;
    const size_t plane = (size_t)nr.ph * nr.pw, hp = (size_t)nr.hh * nr.hw, qp = hp / 4, ep = hp / 16;
    const size_t me = nr.gmMetricHalf ? 2 : 4;
    // the engines' metric keeps their dtype, a pyramid level is always fp32 (k_pyr's output)
    const int mHalf = nr.gmMetricHalf ? 1 : 0, fHalf = nr.gmFeatHalf ? 1 : 0;
    float sA = t, sB = 1.0f - t;
    // one softsplat into channels co .. co + C of the fusionnet input `dst` (dc channels; a channel concat = this splat's
    // own channel range, so no copy follows); hwc = `in` and `dst` channels-last fp16 (nr.gmFeatHwc), else planar.
    // Gather form (nr.gmGather): list set ls built for (flow, metric, s) when `fresh` (k_gsCount, the three scans,
    // k_gsFill), then k_gsGather / k_gsGatherNhwc splats `in` through it; a level's image and feature splats share their
    // direction's set. Scatter form: zero the accumulator, scatter every source pixel, normalize.
    auto splat = [&](int ls, bool fresh, void* in, int inHalf, int C, float* flow, void* metric, int metricHalf,
                     float s, int w, int h, void* dst, int dstHalf, int dc, int co, bool hwc, float lr) -> bool {
        int cc = C, ww = w, hh2 = h, ih = inHalf, mh = metricHalf, dh = dstHalf, dcc = dc, dco = co;
        float slope = lr; // the target's LeakyRelu (fusionnet's b / c / d without theirs), -1 = none
        const int n = w * h;
        int ni = n;
        // planar: channel c of cell i at c * n + i of the range's start; channels-last: c + i * dc after co
        void* dstAt = (uint8_t*)dst + (size_t)co * (hwc ? 1 : (size_t)n) * (dstHalf ? 2 : 4);
        int isc = hwc ? 1 : n, isp = hwc ? C : 1, dsc = hwc ? 1 : n, dsp = hwc ? dc : 1;
        if (nr.gmGather)
        {
            int nb = (n + 2047) / 2048;
            const unsigned gx = (unsigned)(w + 31) / 32, gy = (unsigned)(h + 7) / 8;
            void* k0[] = {(void*)&flow, &metric, &mh, &s, &ww, &hh2, (void*)&nr.dGsCnt[ls], (void*)&nr.dGsSrc[ls]};
            void* k1[] = {(void*)&nr.dGsCnt[ls], &ni, (void*)&nr.dGsOff[ls], (void*)&nr.dGsBlk[ls]};
            void* k2[] = {(void*)&nr.dGsBlk[ls], &nb, (void*)&nr.dGsOff[ls], &ni};
            void* k3[] = {(void*)&nr.dGsOff[ls], (void*)&nr.dGsBlk[ls], &ni, (void*)&nr.dGsCnt[ls]};
            void* k4[] = {(void*)&flow, &s, &ww, &hh2, (void*)&nr.dGsCnt[ls], (void*)&nr.dGsList[ls]};
            void* k5[] = {
                &in,    &ih, &cc,   (void*)&nr.dGsSrc[ls], &ww, &hh2, (void*)&nr.dGsOff[ls], (void*)&nr.dGsList[ls],
                &dstAt, &dh, &slope};
            void* k5n[] = {&in,
                           &cc,
                           (void*)&nr.dGsSrc[ls],
                           &ww,
                           &hh2,
                           (void*)&nr.dGsOff[ls],
                           (void*)&nr.dGsList[ls],
                           &dst,
                           &dcc,
                           &dco,
                           &slope};
            const CUstream cs = (CUstream)st;
            // k_gsGatherNhwc: a cell's C / 8 channel groups on consecutive lanes, 256 / (C / 8) cells a block
            const unsigned ng = (unsigned)C / 8, nc = 256 / ng;
            const bool built =
                !fresh ||
                (cudaMemsetAsync(nr.dGsCnt[ls], 0, (size_t)n * sizeof(int), st) == cudaSuccess &&
                 cuLaunchKernel(nr.fGsCount, gx, gy, 1, 32, 8, 1, 0, cs, k0, nullptr) == CUDA_SUCCESS &&
                 cuLaunchKernel(nr.fGsScanA, (unsigned)nb, 1, 1, 1024, 1, 1, 0, cs, k1, nullptr) == CUDA_SUCCESS &&
                 cuLaunchKernel(nr.fGsScanB, 1, 1, 1, 1024, 1, 1, 0, cs, k2, nullptr) == CUDA_SUCCESS &&
                 cuLaunchKernel(nr.fGsScanC, (unsigned)(n + 255) / 256, 1, 1, 256, 1, 1, 0, cs, k3, nullptr) ==
                     CUDA_SUCCESS &&
                 cuLaunchKernel(nr.fGsFill, gx, gy, 1, 32, 8, 1, 0, cs, k4, nullptr) == CUDA_SUCCESS);
            if (!built || (hwc ? cuLaunchKernel(nr.fGsGatherNhwc, ((unsigned)n + nc - 1) / nc, 1, 1, ng, nc, 1, 0, cs,
                                                k5n, nullptr)
                               : cuLaunchKernel(nr.fGsGather, (unsigned)(n + 255) / 256, (unsigned)(C + 7) / 8, 1, 256,
                                                1, 1, 0, cs, k5, nullptr)) != CUDA_SUCCESS)
            {
                nr.die("gmfss gather splat launch failed");
                return false;
            }
            return true;
        }
        if (cudaMemsetAsync(nr.dGmAcc, 0, (size_t)(C + 1) * (size_t)n * sizeof(long long), st) != cudaSuccess)
        {
            nr.die("gmfss accumulator clear failed");
            return false;
        }
        void* a[] = {&in, &ih, &cc, &isc, &isp, (void*)&flow, &metric, &mh, &s, &ww, &hh2, (void*)&nr.dGmAcc};
        // 32 x 8: one warp = 32 consecutive x of a row, which k_splatSoft's neighbour combine
        // pairs (any shape of <= 256 threads, a multiple of 32, gives the same accumulators)
        if (cuLaunchKernel(nr.fSplatSoft, (w + 31) / 32, (h + 7) / 8, 1, 32, 8, 1, 0, (CUstream)st, a, nullptr) !=
            CUDA_SUCCESS)
        {
            nr.die("gmfss splatSoft launch failed");
            return false;
        }
        void* b[] = {(void*)&nr.dGmAcc, &cc, &ni, &dstAt, &dh, &dsc, &dsp, &slope};
        if (cuLaunchKernel(nr.fSplatNorm, (ni + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, b, nullptr) !=
            CUDA_SUCCESS)
        {
            nr.die("gmfss splatNorm launch failed");
            return false;
        }
        return true;
    };
    void** f0 = nr.dGmFeat[nr.gmCur ^ 1]; // the PREV frame's features (the author's feat1x)
    void** f1 = nr.dGmFeat[nr.gmCur];     // the new frame's (feat2x)
    void* met0 = nr.dGmMetric;
    void* met1 = (uint8_t*)nr.dGmMetric + hp * me;
    float* flow0 = nr.dGmFlow;
    float* flow1 = nr.dGmFlow + 2 * hp;
    const bool prof = nr.gmProf && !nr.gmProfTween; // the group's first tween carries the profile
    // the CPU enqueue span of EVERY tween (the ~20 launches below), the number the
    // per-tween CUDA graph decision rests on
    const int64_t cpu0 = nr.gmProf ? nowQpc100() : 0;
    if (prof)
    {
        nr.gmProfTween = true;
        cudaEventRecord(nr.gmEv[6], st);
    }
    // I1t and I2t into the fusionnet's a planes 0..2 and 6..8 (the ifnet fills 3..5 below)
    if (!splat(0, true, nr.dGmHalf, 0, 3, flow0, met0, mHalf, sA, nr.hw, nr.hh, nr.dGmFa, 0, 9, 0, false, -1.0f))
        return false;
    if (!splat(1, true, nr.dGmHalf + 3 * hp, 0, 3, flow1, met1, mHalf, sB, nr.hw, nr.hh, nr.dGmFa, 0, 9, 6, false,
               -1.0f))
        return false;
    if (prof)
        cudaEventRecord(nr.gmEv[7], st);
    // the GMFSS IFNet, straight into a's middle three planes (the author's cat([I1t, rife, I2t]))
    unsigned int bits;
    memcpy(&bits, &t, 4);
    if (cuMemsetD32Async((CUdeviceptr)nr.dGmT, bits, 1, (CUstream)st) != CUDA_SUCCESS)
    {
        nr.die("gmfss timestep fill failed");
        return false;
    }
    if (!nativeGiRun(nr, nr.dGmHalf, nr.dGmT, nr.dGmFa + 3 * hp, st))
    {
        nr.die("gmfss ifnet's kernels failed to launch");
        return false;
    }
    if (prof)
        cudaEventRecord(nr.gmEv[8], st);
    // the three feature levels, both directions: the half from the engines' own flow and
    // metric, the quarter and the eighth from the pair's pyramids
    // b / c / d channels-last fp16 (fusionnet's layout): the second direction starts gmC channels in
    const int bHalf = nr.gmFuseHalf ? 1 : 0;
    const bool fx = nr.gmFeatHwc;
    const int c0 = nr.gmC[0], c1 = nr.gmC[1], c2 = nr.gmC[2];
    // the half reuses the image splats' list sets (the same flow, metric and s); the quarter and the eighth build theirs
    const float* lr = nr.gmLr;
    if (!splat(0, false, f0[0], fHalf, c0, flow0, met0, mHalf, sA, nr.hw, nr.hh, nr.dGmFb, bHalf, 2 * c0, 0, fx, lr[0]))
        return false;
    if (!splat(1, false, f1[0], fHalf, c0, flow1, met1, mHalf, sB, nr.hw, nr.hh, nr.dGmFb, bHalf, 2 * c0, c0, fx,
               lr[0]))
        return false;
    if (!splat(0, true, f0[1], fHalf, c1, nr.dGmFlowP[0], nr.dGmMetP[0], 0, sA, nr.hw / 2, nr.hh / 2, nr.dGmFc, bHalf,
               2 * c1, 0, fx, lr[1]))
        return false;
    if (!splat(1, true, f1[1], fHalf, c1, nr.dGmFlowP[0] + 2 * qp, nr.dGmMetP[0] + qp, 0, sB, nr.hw / 2, nr.hh / 2,
               nr.dGmFc, bHalf, 2 * c1, c1, fx, lr[1]))
        return false;
    if (!splat(0, true, f0[2], fHalf, c2, nr.dGmFlowP[1], nr.dGmMetP[1], 0, sA, nr.hw / 4, nr.hh / 4, nr.dGmFd, bHalf,
               2 * c2, 0, fx, lr[2]))
        return false;
    if (!splat(1, true, f1[2], fHalf, c2, nr.dGmFlowP[1] + 2 * ep, nr.dGmMetP[1] + ep, 0, sB, nr.hw / 4, nr.hh / 4,
               nr.dGmFd, bHalf, 2 * c2, c2, fx, lr[2]))
        return false;
    if (prof)
        cudaEventRecord(nr.gmEv[9], st);
    if (!nativeFuRun(nr, st))
    {
        nr.die("gmfss fusionnet's kernels failed to launch");
        return false;
    }
    const int ps = (int)plane;
    if (nr.gmTail)
    {
        // the cut graph's out (in the splat accumulator) -> k_fuTail into dGmF: the frame clamped (the author's
        // torch.clamp(out, 0, 1)), or for an SDR pair unclamped, which k_hdrFromSdr then clamps and takes back to the
        // HDR codes in place (each pixel's three planes are read before it writes them)
        const void* pre = nr.dGmAcc;
        void* dst = nr.dGmF;
        int th = nr.hh, tw = nr.hw, cl = nr.gmPairSdr ? 0 : 1;
        void* at[] = {(void*)&pre, (void*)&nr.dGmWl, (void*)&nr.dGmBl, (void*)&dst, &th, &tw, &cl};
        // k_fuTailNhwc's shared tile is sized for these 16 x 8 threads
        if (cuLaunchKernel(nr.gmTailHwc ? nr.fFuTailNhwc : nr.fFuTail, (unsigned)((tw / 2 + 15) / 16),
                           (unsigned)((th / 2 + 7) / 8), 1, 16, 8, 1, 0, (CUstream)st, at, nullptr) != CUDA_SUCCESS ||
            (nr.gmPairSdr &&
             !nativeHdrFromSdr(nr, nr.dGmF, 0, ps, nr.pw, nr.pw, nr.ph, nr.dGmF, ps, nr.pw, nr.encPost)))
        {
            nr.die("gmfss fusion tail launch failed");
            return false;
        }
    }
    else
    {
        // the author's torch.clamp(out, 0, 1) on the fp16 output: k_restToF is exactly that; an SDR pair's tween
        // (the SDR view) goes back to the HDR codes in the same step (k_hdrFromSdr)
        int n = (int)(3 * plane), half = nr.gmOutHalf ? 1 : 0;
        void* ac[] = {(void*)&nr.dGmOut, &half, &n, (void*)&nr.dGmF};
        if (nr.gmPairSdr
                ? !nativeHdrFromSdr(nr, nr.dGmOut, half, ps, nr.pw, nr.pw, nr.ph, nr.dGmF, ps, nr.pw, nr.encPost)
                : cuLaunchKernel(nr.fRestToF, (n + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, ac, nullptr) !=
                      CUDA_SUCCESS)
        {
            nr.die("gmfss clamp launch failed");
            return false;
        }
    }
    if (prof)
        cudaEventRecord(nr.gmEv[10], st);
    if (nr.gmProf)
    {
        nr.gmCpuTween += (double)(nowQpc100() - cpu0) / 10000.0;
        nr.gmCpuTweenN++;
    }
    return true;
}

// SMV_LIVE_GMFSS_PROF=1: the nine spans of the group just finished (the stream is synced by the
// caller's final drain), averaged and printed every 32 groups, plus the CPU enqueue averages
// (the pair block, the gmflow enqueue inside it, and one tween)
static void nativeGmfssProfile(NativeRife& nr)
{
    static const int kSpan[9][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {6, 7}, {7, 8}, {8, 9}, {9, 10}};
    for (int i = 0; i < 9; i++)
    {
        float ms = 0.0f;
        if (cudaEventElapsedTime(&ms, nr.gmEv[kSpan[i][0]], nr.gmEv[kSpan[i][1]]) == cudaSuccess)
            nr.gmAcc[i] += ms;
    }
    if (++nr.gmProfN % 32)
        return;
    const double n = 32.0;
    LOG("[gmfss] feat_ext %.2f | half %.2f | flow %.2f | metric %.2f | pyr %.2f || per tween: "
        "image splats %.2f | ifnet %.2f | feature splats %.2f | fusionnet+clamp %.2f ms "
        "(avg over %u groups)\n",
        nr.gmAcc[0] / n, nr.gmAcc[1] / n, nr.gmAcc[2] / n, nr.gmAcc[3] / n, nr.gmAcc[4] / n, nr.gmAcc[5] / n,
        nr.gmAcc[6] / n, nr.gmAcc[7] / n, nr.gmAcc[8] / n, nr.gmProfN);
    LOG("[gmfss-cpu] enqueue: pair block %.3f (gmflow %.3f) over %u pairs | tween %.3f ms over "
        "%u tweens\n",
        nr.gmCpuPair / (double)(nr.gmCpuPairN ? nr.gmCpuPairN : 1),
        nr.gmCpuFlow / (double)(nr.gmCpuPairN ? nr.gmCpuPairN : 1), nr.gmCpuPairN,
        nr.gmCpuTween / (double)(nr.gmCpuTweenN ? nr.gmCpuTweenN : 1), nr.gmCpuTweenN);
    for (double& a : nr.gmAcc)
        a = 0.0;
    nr.gmCpuPair = nr.gmCpuFlow = nr.gmCpuTween = 0.0;
    nr.gmCpuPairN = nr.gmCpuTweenN = 0;
}

// one group: capture wait and read, the
// capture-release announcement, the pair encode, the tween chunks through the dynamic-batch
// engine, one pack-out per slot with its own event, then the bare end marker.
// ---- native DRBA ----------------------------------------------------------------------------
// DRBA as native launches on the group's stream. Frame ids count from 1 per
// session (drFid = the newest, k); id i sits in ring slot i & 3, four kept.
// Every frame is encoded once when pushed (encode is a pure function of the
// frame). A window centred on c holds flow10 (2 planes) and
// r = calc_flow(I1 = c, I2 = c + 1)'s flow12 | flow21 (4 planes); two slots, the one holding
// c - 1 is kept when c is built, and a kept left
// neighbour chains: this window's flow10 = its flow21, else block0 on
// (c, c - 1) computes it. The group presents time (k - 2) + f: f >= 0.5 from window k - 1 side
// -1 (x = [k-1, k-2], tt = 1 - f), f < 0.5 from window k - 2 side +1 (x = [k-2, k-1], tt = f);
// the head (three frames) runs plain pair RIFE on (k-2, k-1), fewer frames hold the lagged real
// frame k - 1.
// fp32 -> fp16 of n elements on the host's stream (the frames of the RIFE engines that take fp16)
static bool nativeF2h(NativeRife& nr, const float* src, uint16_t* dst, size_t n, cudaStream_t st)
{
    int k = (int)n;
    void* a[] = {&src, &dst, &k};
    return cuLaunchKernel(nr.fF2h, (k + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, a, nullptr) == CUDA_SUCCESS;
}

static float* drbaFrame(NativeRife& nr, uint32_t id)
{
    return nr.dDrI[id & 3];
}
static float* drbaEnc(NativeRife& nr, uint32_t id)
{
    return nr.dDrF[id & 3];
}
static uint16_t* drbaMotion(NativeRife& nr, uint32_t id)
{
    return nr.dDrM[id & 3];
}
// two domains: where k_motionIn writes the motion frame of the frame the next push takes
static uint16_t* drbaMotionNext(NativeRife& nr)
{
    return drbaMotion(nr, nr.drFid + 1);
}
// the model's frame: RIFE's motion frame in two domains, else the pictures' pad
static int drbaMh(const NativeRife& nr)
{
    return nr.mph ? nr.mph : nr.ph;
}
static int drbaMw(const NativeRife& nr)
{
    return nr.mph ? nr.mpw : nr.pw;
}

// RIFE's encode on the host's kernels, k_encC01 -> k_encC2Up on st (nr.stream: the planes share the activation block
// with the contexts that enqueue there): img = a planar frame at the encode's frame w x h (even, the /64 pad), fp16
// (imgHalf) or fp32; out = its kRifeFeatC planes in the IFNet's feature dtype (featHalf). cnn1's half-size planes live
// in the shared block, else in dEncOwn (SMV_CTXSHARE=0)
static bool nativeEncodeRun(NativeRife& nr, const void* img, int imgHalf, int w, int h, void* out, cudaStream_t st)
{
    const size_t need = encNatBytes(nr);
    float* a = nr.dGmCtxMem && (size_t)nr.gmCtxMemB >= need ? (float*)nr.dGmCtxMem : nr.dEncOwn;
    if (!a && cudaMalloc((void**)&nr.dEncOwn, need) == cudaSuccess)
        a = nr.dEncOwn;
    if (!a)
    {
        nr.die("encode planes alloc failed");
        return false;
    }
    int w2 = w / 2, h2 = h / 2, oh = nr.featHalf ? 1 : 0;
    const unsigned gx = (unsigned)(w2 + 31) / 32, gy = (unsigned)(h2 + 7) / 8;
    void* p0[] = {(void*)&img, &imgHalf, &w, &h, &a};
    void* p1[] = {&a, &w2, &h2, &out, &oh};
    if (cuLaunchKernel(nr.fEncC01, gx, gy, 1, 32, 8, 1, 0, (CUstream)st, p0, nullptr) != CUDA_SUCCESS ||
        cuLaunchKernel(nr.fEncC2Up, gx, gy, 1, 32, 8, 1, 0, (CUstream)st, p1, nullptr) != CUDA_SUCCESS)
    {
        nr.die("encode launch failed");
        return false;
    }
    return true;
}

// the new packed frame into the ring, and its encode beside it. Two domains: its motion frame sits in
// the ring already (drbaMotionNext), the encode reads that; same = a frame that reuses the previous
// frame's picture takes the previous motion frame too.
static bool nativeDrbaPush(NativeRife& nr, const float* dCur, bool same = false)
{
    cudaStream_t st = nr.stream;
    const size_t plane = (size_t)nr.ph * nr.pw, mplane = (size_t)drbaMh(nr) * drbaMw(nr);
    const uint32_t id = ++nr.drFid;
    float* dst = drbaFrame(nr, id);
    float* enc = drbaEnc(nr, id);
    // a frame built straight in its ring slot (dCur == dst) needs no copy
    if ((dCur != dst &&
         cudaMemcpyAsync(dst, dCur, 3 * plane * sizeof(float), cudaMemcpyDeviceToDevice, st) != cudaSuccess) ||
        (nr.mph && same && id > 1 &&
         cudaMemcpyAsync(drbaMotion(nr, id), drbaMotion(nr, id - 1), 3 * mplane * sizeof(uint16_t),
                         cudaMemcpyDeviceToDevice, st) != cudaSuccess))
    {
        nr.die("drba frame ring copy failed");
        return false;
    }
    // the encode reads the motion frame (fp16), else the new frame itself (fp32), and writes the ring slot in the
    // IFNet's feature dtype (read by block0 / IFNet)
    return nr.mph ? nativeEncodeRun(nr, drbaMotion(nr, id), 1, drbaMw(nr), drbaMh(nr), enc, st)
                  : nativeEncodeRun(nr, dst, 0, drbaMw(nr), drbaMh(nr), enc, st);
}

// calc_flow(frame a, frame b): block0 on the pair (nativeB0Run, its flow at 1/16), then its tail (the DRBA glue
// kernels) into out (4 planes: flow05 * 2 | flow15 * 2): the splat samples block0's planes at the motion frame into the
// accumulator, FlowNorm writes every value of out. Two domains: block0 reads the two motion frames (fp16).
static bool nativeDrbaFlow(NativeRife& nr, uint32_t a, uint32_t b, float* out)
{
    cudaStream_t st = nr.stream;
    int mw = drbaMw(nr), mh = drbaMh(nr);
    const size_t plane = (size_t)mh * mw;
    const void* img0 = nr.mph ? (const void*)drbaMotion(nr, a) : (const void*)drbaFrame(nr, a);
    const void* img1 = nr.mph ? (const void*)drbaMotion(nr, b) : (const void*)drbaFrame(nr, b);
    float* lo = nullptr;
    if (!nativeB0Run(nr, img0, img1, nr.mph ? 1 : 0, drbaEnc(nr, a), drbaEnc(nr, b), mw, mh, &lo, st))
    {
        nr.die("block0 launch failed");
        return false;
    }
    if (cudaMemsetAsync(nr.dDrAcc, 0, 6 * plane * sizeof(long long), st) != cudaSuccess)
    {
        nr.die("drba accumulator clear failed");
        return false;
    }
    void* as[] = {&lo, &mw, &mh, &nr.dDrAcc};
    if (cuLaunchKernel(nr.fDrFlowSplat, (mw + 15) / 16, (mh + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, as, nullptr) !=
        CUDA_SUCCESS)
    {
        nr.die("drbaFlowSplat launch failed");
        return false;
    }
    const int n = (int)plane;
    void* an[] = {&nr.dDrAcc, &mw, &mh, &out};
    if (cuLaunchKernel(nr.fDrFlowNorm, (n + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, an, nullptr) != CUDA_SUCCESS)
    {
        nr.die("drbaFlowNorm launch failed");
        return false;
    }
    nr.drBlock0++;
    return true;
}

// the window centred on c: cached, or built (chained on a kept c - 1 window); nullptr = died
static NativeRife::DrWin* nativeDrbaWindow(NativeRife& nr, uint32_t c)
{
    for (auto& wn : nr.drWin)
        if (wn.c == c)
            return &wn;
    const size_t plane = (size_t)drbaMh(nr) * drbaMw(nr);
    // keep the left neighbour (the chain), else a right one (every window >= c - 1 is needed;
    // the group builds k - 2 before k - 1 because the exe's fractions ascend, so both fit)
    const int s = (c > 1 && nr.drWin[0].c == c - 1)   ? 1
                  : (c > 1 && nr.drWin[1].c == c - 1) ? 0
                  : (nr.drWin[0].c == c + 1)          ? 1
                                                      : 0;
    NativeRife::DrWin& wn = nr.drWin[s];
    const NativeRife::DrWin* left = (c > 1 && nr.drWin[1 - s].c == c - 1) ? &nr.drWin[1 - s] : nullptr;
    wn.c = 0;
    if (left)
    {
        if (cudaMemcpyAsync(wn.f10, left->r + 2 * plane, 2 * plane * sizeof(float), cudaMemcpyDeviceToDevice,
                            nr.stream) != cudaSuccess)
        {
            nr.die("drba window chain copy failed");
            return nullptr;
        }
    }
    else
    {
        // the left flow lands in wn.r, which the right flow below overwrites (wn.c = 0: nothing reads it between)
        if (!nativeDrbaFlow(nr, c, c - 1, wn.r))
            return nullptr;
        if (cudaMemcpyAsync(wn.f10, wn.r, 2 * plane * sizeof(float), cudaMemcpyDeviceToDevice, nr.stream) !=
            cudaSuccess)
        {
            nr.die("drba window left flow copy failed");
            return nullptr;
        }
    }
    if (!nativeDrbaFlow(nr, c, c + 1, wn.r))
        return nullptr;
    wn.c = c;
    return &wn;
}

// one tween of the group at fraction f into dMerged; held = no tween (the slot shows the
// lagged real frame). nHist = frames in the history (at most 4). plain =
// plain pair RIFE on (k-2, k-1) at t = f whatever f is (the offline tail window:
// the last window has no right frame). slot = blendOut's output slot: the tween
// goes straight into it instead of dMerged.
// two domains: the blend's frame pair [k-2, k-1] read straight from the ring, whose slots are one block: slots
// (k-2) & 3 and (k-1) & 3 are neighbours unless (k-2) & 3 == 3; then frame k - 1 is copied once per k into slot
// (k+1) & 3, the slot of frame k - 3, which no picture reader uses after push(k) (the emits and the live lag read k - 1
// and k - 2), giving [k-1, k-2]. Frame k + 1 is then built in that same slot (both loops build each frame in its ring
// slot), so its decode / pack must be enqueued after group k's tweens on the same stream. pic = the pair's first slot, sw = the blend's frame order (1: frame 0 is the second
// slot); the IFNet's frame 0 is k - 1 on side -1, k - 2 on side +1 and the head's plain pair.
static bool nativeDrbaPics(NativeRife& nr, uint32_t k, int side, const float*& pic, int& sw)
{
    const size_t plane = (size_t)nr.ph * nr.pw;
    if (((k - 2) & 3) != 3)
    {
        pic = drbaFrame(nr, k - 2);
        sw = side < 0 ? 1 : 0;
        return true;
    }
    float* fix = nr.dDrI[(k + 1) & 3];
    if (nr.drPicFix != k)
    {
        if (cudaMemcpyAsync(fix, drbaFrame(nr, k - 1), 3 * plane * sizeof(float), cudaMemcpyDeviceToDevice,
                            nr.stream) != cudaSuccess)
        {
            nr.die("drba wrap pair copy failed");
            return false;
        }
        nr.drPicFix = k;
    }
    pic = fix;
    sw = side < 0 ? 0 : 1;
    return true;
}

// k_bilin: three planes (sw x sh at sps / srs) to dw x dh (at dps / drs), fp16 out when dHalf
static bool nativeBilin(NativeRife& nr, const float* src, int sps, int srs, int sw, int sh, void* dst, int dps, int drs,
                        int dw, int dh, int dHalf, cudaStream_t st)
{
    void* a[] = {(void*)&src, &sps, &srs, &sw, &sh, &dst, &dps, &drs, &dw, &dh, &dHalf};
    return cuLaunchKernel(nr.fBilin, (dw + 15) / 16, (dh + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a, nullptr) ==
           CUDA_SUCCESS;
}

// DRBA as its author (drAuthor): the pair's pictures (pic, frame 0 / frame 1 swapped by sw as k_rifeBlend reads them)
// stretched to the network's frame once per FRAME (to_inp; dDrS slot i & 1 holds frame i: both sides of a group and
// the next group's left frame find theirs there), RIFE's last step there with the IFNet's flow and mask as they are,
// the tween back to the pictures' size into dMerged (to_out)
static bool nativeDrbaAuthorBlend(NativeRife& nr, const float* pic, int sw, uint32_t k, int side, cudaStream_t st)
{
    int mw = nr.mpw, mh = nr.mph, mps = mw * mh, pps = nr.ph * nr.pw, half = 0, zp = 0;
    // frame k - 2 sits at pic's slot k2 (nativeDrbaPics: sw = 1 puts the IFNet's frame 0 second; frame 0 is k - 1 on
    // side -1, k - 2 on side +1), frame k - 1 at the other
    const int k2 = side < 0 ? 1 - sw : sw;
    for (int f = 0; f < 2; f++)
    {
        const int64_t fid = (int64_t)k - 2 + f;
        const int slot = (int)(fid & 1), src = f ? 1 - k2 : k2;
        if (nr.drSFid[slot] == fid)
            continue;
        if (!nativeBilin(nr, pic + (size_t)src * 3 * pps, pps, nr.pw, nr.w, nr.h, nr.dDrS + (size_t)slot * 3 * mps, mps,
                         mw, mw, mh, 0, st))
            return false;
        nr.drSFid[slot] = fid;
    }
    // the blend's frame 0 (k - 1 on side -1, k - 2 on side +1) is second in dDrS when its id is odd
    int ssw = (int)(((int64_t)k - (side < 0 ? 1 : 2)) & 1);
    const float* sp = nr.dDrS;
    void* a[] = {(void*)&sp, &ssw,      &half, &mps, &mw, &mw, &mh,      &mw, &mh,
                 &nr.dFlow,  &nr.dMask, &mps,  &mw,  &mw, &mh, &nr.dDrB, &zp};
    return cuLaunchKernel(nr.fRifeBlendF, (mw + 15) / 16, (mh + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a, nullptr) ==
               CUDA_SUCCESS &&
           nativeBilin(nr, nr.dDrB, mps, mw, mw, mh, nr.dMerged, pps, nr.pw, nr.w, nr.h, nr.outHalf ? 1 : 0, st);
}

// a real frame of a DRBA session as its author outputs it (drAuthor: stretched to the network's frame and back, as
// infer.py's to_out does to every frame it writes), else the picture itself
static const float* nativeDrbaReal(NativeRife& nr, const float* pic)
{
    if (!nr.drAuthor)
        return pic;
    const int mps = nr.mph * nr.mpw, pps = nr.ph * nr.pw;
    if (!nativeBilin(nr, pic, pps, nr.pw, nr.w, nr.h, nr.dDrB, mps, nr.mpw, nr.mpw, nr.mph, 0, nr.stream) ||
        !nativeBilin(nr, nr.dDrB, mps, nr.mpw, nr.mpw, nr.mph, nr.dDrRT, pps, nr.pw, nr.w, nr.h, 0, nr.stream))
    {
        nr.die("drba real frame stretch failed");
        return nullptr;
    }
    return nr.dDrRT;
}

static bool nativeDrbaTween(NativeRife& nr, float f, int nHist, bool& held, bool plain = false, uint8_t* slot = nullptr)
{
    cudaStream_t st = nr.stream;
    int mw = drbaMw(nr), mh = drbaMh(nr);
    const size_t plane = (size_t)nr.ph * nr.pw, mplane = (size_t)mh * mw;
    const uint32_t k = nr.drFid;
    held = false;
    NativeRife::DrWin* wn = nullptr;
    int side = +1;
    float tt = f;
    if (plain)
    {
    }
    else if (f >= 0.5f)
    {
        if (nHist < 3)
        {
            held = true;
            return true;
        }
        if (!(wn = nativeDrbaWindow(nr, k - 1)))
            return false;
        side = -1;
        tt = 1.0f - f;
    }
    else if (nHist >= 4)
    {
        if (!(wn = nativeDrbaWindow(nr, k - 2)))
            return false;
    }
    else if (nHist != 3)
    {
        held = true;
        return true;
    }
    // x = cat(I1, I0) on side -1, cat(I1, I2) on side +1 and for the head's plain pair
    const int xi = side < 0 ? 0 : 1;
    const uint32_t i1 = side < 0 ? k - 1 : k - 2, i0 = side < 0 ? k - 2 : k - 1;
    if (nr.drXFor[xi] != k)
    {
        if (nr.mph)
        {
            // two domains: the IFNet's x is the motion ring itself from frame i1's slot on (its pictures:
            // nativeDrbaPics). The ring descends, so on side -1 frame k - 2 already follows frame k - 1 unless
            // frame k - 1 holds the last slot (then frame k - 2 is copied into the spare). Side +1 and the head's
            // plain pair copy frame k - 1 into the slot after frame k - 2: the spare, or frame k - 3's, whose last
            // reader is window k - 2's block0, built above before this copy (the offline tail's plain tweens come
            // after the group's windowed ones). Frame k + 1's motion frame lands in that slot next, so its pack
            // must be enqueued after group k's tweens on the same stream, like its picture.
            uint16_t* x = drbaMotion(nr, i1);
            if (x + 3 * mplane != drbaMotion(nr, i0) &&
                cudaMemcpyAsync(x + 3 * mplane, drbaMotion(nr, i0), 3 * mplane * sizeof(uint16_t),
                                cudaMemcpyDeviceToDevice, st) != cudaSuccess)
            {
                nr.die("drba x copy failed");
                return false;
            }
            nr.dDrMX[xi] = x;
        }
        else if (nr.xHalf)
        {
            // the IFNet takes x in fp16: the two ring frames narrowed into dDrX (fp32-sized)
            uint16_t* x16 = (uint16_t*)nr.dDrX[xi];
            if (!nativeF2h(nr, drbaFrame(nr, i1), x16, 3 * plane, st) ||
                !nativeF2h(nr, drbaFrame(nr, i0), x16 + 3 * plane, 3 * plane, st))
            {
                nr.die("drba x f2h failed");
                return false;
            }
        }
        else if (cudaMemcpyAsync(nr.dDrX[xi], drbaFrame(nr, i1), 3 * plane * sizeof(float), cudaMemcpyDeviceToDevice,
                                 st) != cudaSuccess ||
                 cudaMemcpyAsync(nr.dDrX[xi] + 3 * plane, drbaFrame(nr, i0), 3 * plane * sizeof(float),
                                 cudaMemcpyDeviceToDevice, st) != cudaSuccess)
        {
            nr.die("drba x copy failed");
            return false;
        }
        nr.drXFor[xi] = k;
    }
    if (!wn)
    {
        // the head: plain pair RIFE at t = f, a constant timestep map
        if (!nativeFillT(nr, 0, f, mplane, st))
        {
            nr.die("timestep fill failed");
            return false;
        }
        nr.drHeads++;
    }
    else
    {
        if (cudaMemsetAsync(nr.dDrAcc, 0, 2 * mplane * sizeof(long long), st) != cudaSuccess)
        {
            nr.die("drba accumulator clear failed");
            return false;
        }
        void* as[] = {&wn->f10, &wn->r, &side, &tt, &mw, &mh, &nr.dDrAcc};
        if (cuLaunchKernel(nr.fDrDrmSplat, (mw + 15) / 16, (mh + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, as,
                           nullptr) != CUDA_SUCCESS)
        {
            nr.die("drbaDrmSplat launch failed");
            return false;
        }
        const int n = (int)mplane, th = nr.tHalf ? 1 : 0;
        nr.tHeldMask = 0; // the DRM map overwrites the timestep plane
        void* an[] = {&nr.dDrAcc, &wn->f10, &wn->r, &side, &tt, &mw, &mh, &nr.dT, (void*)&th};
        if (cuLaunchKernel(nr.fDrDrmNorm, (n + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, an, nullptr) !=
            CUDA_SUCCESS)
        {
            nr.die("drbaDrmNorm launch failed");
            return false;
        }
    }
    if (!nativeIfnRun(nr, nr.mph ? (const void*)nr.dDrMX[xi] : (const void*)nr.dDrX[xi], nr.dT, drbaEnc(nr, i1),
                      drbaEnc(nr, i0), mw, mh, nr.dFlow, nr.dMask, st))
    {
        nr.die("IFNet launch failed");
        return false;
    }
    // the tween: RIFE's last step on the frames this enqueue read, in two domains on their pictures
    const float* pic = nr.dDrX[xi];
    int sw = 0;
    if (nr.mph && !nativeDrbaPics(nr, k, side, pic, sw))
        return false;
    if (nr.drAuthor)
    {
        if (!nativeDrbaAuthorBlend(nr, pic, sw, k, side, st))
        {
            nr.die("drba stretched blend failed");
            return false;
        }
    }
    else if (slot ? !nativeRifeBlendOut(nr, pic, nr.mph ? 0 : (int)nr.xHalf, 0, slot, st, sw)
                  : !nativeRifeBlend(nr, pic, nr.mph ? 0 : (int)nr.xHalf, 1, st, sw))
    {
        nr.die(slot ? "rifeBlendOut launch failed" : "rifeBlend launch failed");
        return false;
    }
    nr.drTweens++;
    return true;
}

// tween k of the last batch (RIFE's last step's, 3 planes of `plane` elements each, fp32, fp16 when outHalf)
static const void* mergedAt(const NativeRife& nr, size_t k, size_t plane)
{
    const size_t n = k * 3 * plane;
    return nr.outHalf ? (const void*)((const uint16_t*)nr.dMerged + n) : (const void*)(nr.dMerged + n);
}

// Restore (live and offline) on one
// source-size planar frame (srcW x srcH: live the captured frame, offline the decoded one with
// a working size, else the model frame; fp16 when sHalf: a tween as the IFNet wrote it), the
// result folded into dst (tw x th).
// Restore's way back on HDR planes (k_hdrRestOut, in place on dst, tw x th): the remainder comes from the source
// (sw x sh, planes ps_ apart, rows rs_ apart, fp16 when sHalf), fitted to the target first when the sizes differ
static bool nativeHdrRestOut(NativeRife& nr, float* dst, int tw, int th, const void* s, int sHalf, int ps_, int rs_,
                             int sw, int sh, int mode, bool viewRef = false)
{
    cudaStream_t st = nr.stream;
    const void* ref = s;
    int half = sHalf, fps = ps_, frs = rs_, rps = tw * th;
    if (tw != sw || th != sh)
    {
        if (!nr.dRestRem)
        {
            nr.die("restore HDR fit buffer missing");
            return false;
        }
        // a source inside the SDR range: the reference is its SDR view's fit, so it carries no ringing above white
        if (viewRef ? !nativeViewFit(nr, s, sHalf, ps_, rs_, sw, sh, nr.dRestRem, tw, th, mode)
                    : !nativeFitPlanar(nr, s, sHalf, ps_, rs_, sw, sh, nr.dRestRem, tw, th, st))
        {
            nr.die("restore HDR fit launch failed");
            return false;
        }
        ref = nr.dRestRem;
        half = 0;
        fps = rps;
        frs = tw;
    }
    void* a[] = {(void*)&dst, &rps, &tw,         (void*)&ref, &half, &fps,  &frs,
                 &tw,         &th,  (void*)&dst, &rps,        &tw,   &mode, &nr.encWhite};
    if (cuLaunchKernel(nr.fHdrRestOut, (tw + 15) / 16, (th + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a, nullptr) !=
        CUDA_SUCCESS)
    {
        nr.die("restore HDR codes launch failed");
        return false;
    }
    return true;
}

// false = a launch failed (die was called); an engine enqueue refusal drops the pass for the rest
// of the session instead, the caller then continues with the unrestored source.
// HDR planes (enc 1 PQ, 2 HLG): the engine reads the frame's SDR view (hdrToModel, knee 1, head 0, the sign
// clipped), the fold is the SDR one, and k_hdrRestOut adds back the light the view never held.
// Restore = Anime4K (restNatKind 3): its passes on the same input, then its de-ringing clamp, unless the resize's Anime4K
// pass closes this frame's chain (a4kAfter: the clamp and its statistics move there).
static bool nativeRestoreRun(NativeRife& nr, const void* s, int ps_, int rs_, float* dst, int tw, int th, int sHalf = 0,
                             int enc = 0, bool viewRef = false, bool a4kAfter = false)
{
    cudaStream_t st = nr.stream;
    int sw = srcW(nr), sh = srcH(nr);
    if (enc)
    {
        if (!nativeHdrEnc(nr, s, sHalf, ps_, rs_, sw, sh, nr.dRestIn, nr.restHalfIn ? 1 : 0, sw * sh, sw, enc, 1.0f,
                          0.0f))
        {
            nr.die("restore model values launch failed");
            return false;
        }
    }
    else if (nr.restHalfIn)
    {
        void* a[] = {(void*)&s, &sHalf, &ps_, &rs_, &sw, &sh, &nr.dRestIn};
        if (cuLaunchKernel(nr.fRestIn, (sw + 15) / 16, (sh + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a, nullptr) !=
            CUDA_SUCCESS)
        {
            nr.die("restIn launch failed");
            return false;
        }
    }
    else if (sHalf)
    {
        // an fp32 engine input from an fp16 source: the 1:1 fit is the widening crop copy
        void* a[] = {(void*)&s, &sHalf, &ps_, &rs_, &sw, &sh, &nr.dRestIn, &sw, &sh};
        if (cuLaunchKernel(nr.fFitPlanar, (sw + 15) / 16, (sh + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a, nullptr) !=
            CUDA_SUCCESS)
        {
            nr.die("restore input widen failed");
            return false;
        }
    }
    else
    {
        for (int c = 0; c < 3; c++)
            if (cudaMemcpy2DAsync((float*)nr.dRestIn + (size_t)c * sw * sh, (size_t)sw * 4,
                                  (const float*)s + (ptrdiff_t)c * ps_, (size_t)rs_ * 4, (size_t)sw * 4, sh,
                                  cudaMemcpyDeviceToDevice, st) != cudaSuccess)
            {
                nr.die("restore input copy failed");
                return false;
            }
    }
    if (nr.restNatKind == 3)
    {
        if (!nativeA4kRestoreRun(nr, dst, tw, th, a4kAfter, enc != 0))
            return false;
        return !enc || nativeHdrRestOut(nr, dst, tw, th, s, sHalf, ps_, rs_, sw, sh, enc, viewRef);
    }
    if (!restNatRun(nr, st))
    {
        nr.die("restore kernel launch failed");
        return false;
    }
    // w4 / h4 = the model's output size (restScale x the input: Real-ESRGAN 4, AnimeJaNai 2)
    int half = nr.restHalfOut ? 1 : 0, w4 = nr.restScale * sw, h4 = nr.restScale * sh, ps4 = w4 * h4;
    // realesr.fit: the Lanczos pair at every ratio (an exact copy at the model's own scale), with `out.clamp(0,1)`
    // folded into the taps and the final clamp in the vertical pass
    const void* tabH = rsTabFor(nr, w4, tw, st);
    const void* tabV = rsTabFor(nr, h4, th, st);
    if (!tabH || !tabV)
    {
        nr.die("restore fold tap tables failed");
        return false;
    }
    int inHalf = nr.restHalfIn ? 1 : 0;
    const void* base = nr.restSubBase ? nr.dRestIn : nullptr; // Real-ESRGAN adds the nearest base, AnimeJaNai none
    void* ah[] = {&nr.dRestOut, &half, &ps4, &w4, &w4, &h4, &nr.dRestTmp, &tw, (void*)&tabH};
    void* ahs[] = {&nr.dRestOut, &half, (void*)&base, &inHalf,      &nr.restSubS, &nr.restSubW,
                   &nr.restSubH, &w4,   &h4,          &nr.dRestTmp, &tw,          (void*)&tabH};
    if (cuLaunchKernel(nr.restSub ? nr.fRestFoldHSub : nr.fRestFoldH, (tw + 15) / 16, (h4 + 15) / 16, 1, 16, 16, 1, 0,
                       (CUstream)st, nr.restSub ? ahs : ah, nullptr) != CUDA_SUCCESS)
    {
        nr.die("restFoldH launch failed");
        return false;
    }
    void* av[] = {&nr.dRestTmp, &tw, &h4, &dst, &th, (void*)&tabV};
    if (cuLaunchKernel(nr.fRestFoldV, (tw + 15) / 16, (th + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, av, nullptr) !=
        CUDA_SUCCESS)
    {
        nr.die("restFoldV launch failed");
        return false;
    }
    return !enc || nativeHdrRestOut(nr, dst, tw, th, s, sHalf, ps_, rs_, sw, sh, enc, viewRef);
}

// Offline TrueHDR: the previous frame's statistics block reaches the
// accumulators once the stream is known idle (the TrueHDR sync below, or the render's end).
static void nativeOfflineThdrDrain(NativeRife& nr)
{
    if (!nr.thdrStatsPending)
        return;
    nr.thdrStatsPending = false;
    if (nr.thdrAcc)
        nr.thdrAcc->add(nr.hThdrStats, (uint64_t)nr.dw * nr.dh);
    for (; nr.thdrRepeat; nr.thdrRepeat--)
        if (nr.thdrAcc)
            nr.thdrAcc->repeat();
}

// a held slot re-sent from the previous real frame's finished bytes: its statistics record
// is that frame's, still pending (read one frame late), so the repeat waits for the drain
static void nativeOfflineThdrRepeat(NativeRife& nr)
{
    if (nr.thdrStatsPending)
        nr.thdrRepeat++;
    else if (nr.thdrAcc)
        nr.thdrAcc->repeat();
}

static float halfToFloat(uint16_t h); // main.cpp, after the parts

// --nr-delta PATH (the preview's change mask): the
// largest channel change of the DLSS 5 pass per pixel, |after - before| with `before` the pass's
// fp32 input planes (tight, tw x th each) and `after` its fp16 output clamped to 0..1 like
// k_nrOut, or the given planes (HDR video: the frame after k_nrOutPq), tw x th float32, rewritten
// per frame (the last frame's). A failed write costs only the mask.
static void nativeOfflineNrDelta(const NativeRife& nr, const std::vector<float>& before, int tw, int th,
                                 const std::vector<float>* after = nullptr)
{
    const size_t n = (size_t)tw * th;
    std::vector<float> d(n);
    for (size_t o = 0; o < n; o++)
    {
        float m = 0.0f;
        for (int c = 0; c < 3; c++)
        {
            float a = after ? (*after)[c * n + o] : halfToFloat(nr.hNrOut[o * 4 + c]);
            if (!after)
                a = a < 0.0f ? 0.0f : (a > 1.0f ? 1.0f : a);
            const float v = fabsf(a - before[c * n + o]);
            if (v > m)
                m = v;
        }
        d[o] = m > 1.0f ? 1.0f : m;
    }
    FILE* f = nullptr;
    if (_wfopen_s(&f, nr.nrDeltaPath.c_str(), L"wb") || !f || fwrite(d.data(), sizeof(float), d.size(), f) != d.size())
        LOG("offline: cannot write the DLSS 5 change map %s\n", wideToUtf8(nr.nrDeltaPath).c_str());
    if (f)
        fclose(f);
}

// The zero-copy imports of the DLSS 5 handoff, released in the order CUDA asks for (the surface
// objects and mapped arrays first). The caller has drained the stream.
static void nativeNrReleaseImports(NativeRife& nr)
{
    for (unsigned long long* s : {&nr.sfNrIn, &nr.sfNrOut, &nr.sfNrMv})
        if (*s)
        {
            cudaDestroySurfaceObject(*s);
            *s = 0;
        }
    for (cudaMipmappedArray_t* m : {&nr.mmNrIn, &nr.mmNrOut, &nr.mmNrMv})
        if (*m)
        {
            cudaFreeMipmappedArray(*m);
            *m = nullptr;
        }
    if (nr.semNr)
    {
        cudaDestroyExternalSemaphore(nr.semNr);
        nr.semNr = nullptr;
    }
    for (cudaExternalMemory_t* e : {&nr.emNrIn, &nr.emNrOut, &nr.emNrMv})
        if (*e)
        {
            cudaDestroyExternalMemory(*e);
            *e = nullptr;
        }
}

// the current CUDA device's adapter LUID: the NR core creates its D3D12 device on that adapter, so the
// zero-copy handoff shares one GPU even when another GPU drives the main display
static bool nativeCudaLuid(LUID& out)
{
    int dev = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess)
        return false;
    memcpy(&out, prop.luid, sizeof(out));
    return true;
}

// one of the NR core's textures as a CUDA array (a dedicated import of its whole allocation, mapped
// with surface load / store) and its surface object
static bool nativeNrImportTex(HANDLE h, uint64_t bytes, bool rg, int w, int h2, cudaExternalMemory_t& em,
                              cudaMipmappedArray_t& mm, unsigned long long& sf)
{
    cudaExternalMemoryHandleDesc md{};
    md.type = cudaExternalMemoryHandleTypeD3D12Resource;
    md.size = bytes;
    md.flags = cudaExternalMemoryDedicated;
    md.handle.win32.handle = h;
    if (cudaImportExternalMemory(&em, &md) != cudaSuccess)
        return false;
    cudaExternalMemoryMipmappedArrayDesc ad{};
    ad.formatDesc = rg ? cudaCreateChannelDesc(16, 16, 0, 0, cudaChannelFormatKindFloat)
                       : cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindFloat);
    ad.extent = cudaExtent{(size_t)w, (size_t)h2, 0};
    ad.numLevels = 1;
    ad.flags = cudaArraySurfaceLoadStore;
    cudaArray_t arr = nullptr;
    if (cudaExternalMemoryGetMappedMipmappedArray(&mm, em, &ad) != cudaSuccess ||
        cudaGetMipmappedArrayLevel(&arr, mm, 0) != cudaSuccess)
        return false;
    cudaResourceDesc rd{};
    rd.resType = cudaResourceTypeArray;
    rd.res.array.array = arr;
    cudaSurfaceObject_t s = 0;
    if (cudaCreateSurfaceObject(&s, &rd) != cudaSuccess)
        return false;
    sf = (unsigned long long)s;
    return true;
}

// Zero-copy DLSS 5: import the NR core's Color / Output / MVec textures and fence
// (nr::Host::startShared) into this CUDA context. Refused when the core's D3D12 adapter is not the
// current CUDA device; the caller then keeps the CPU staging route.
static bool nativeNrImport(NativeRife& nr, nr::Host& host, int w, int h, std::string& err)
{
    int dev = 0;
    cudaDeviceProp prop{};
    const LUID luid = host.adapterLuid();
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess ||
        memcmp(prop.luid, &luid, sizeof(luid)) != 0)
    {
        err = "the NR device is not the CUDA device";
        return false;
    }
    if (!nativeNrImportTex(host.colorHandle(), host.colorBytes(), false, w, h, nr.emNrIn, nr.mmNrIn, nr.sfNrIn) ||
        !nativeNrImportTex(host.outputHandle(), host.outputBytes(), false, w, h, nr.emNrOut, nr.mmNrOut, nr.sfNrOut))
    {
        cudaGetLastError();
        err = "Color / Output texture import failed";
        return false;
    }
    cudaExternalSemaphoreHandleDesc sd{};
    sd.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
    sd.handle.win32.handle = host.sharedFenceHandle();
    if (cudaImportExternalSemaphore(&nr.semNr, &sd) != cudaSuccess)
    {
        err = "fence import failed";
        return false;
    }
    if (host.mvHandle() &&
        !nativeNrImportTex(host.mvHandle(), host.mvBytes(), true, w, h, nr.emNrMv, nr.mmNrMv, nr.sfNrMv))
    {
        cudaGetLastError();
        err = "MVec texture import failed";
        return false;
    }
    return true;
}

// a zero DLSSNR.MVec (k_nrMvClear) on the session's stream
static bool nativeNrMvClear(NativeRife& nr)
{
    int w = nr.nrW, h = nr.nrH;
    void* a[] = {&nr.sfNrMv, &w, &h};
    return cuLaunchKernel(nr.fNrMvClear, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)nr.stream, a,
                          nullptr) == CUDA_SUCCESS;
}

// DLSS 5 motion, set up once the core's textures are imported: an Optical Flow session at the NR
// size (forward only: the current frame is the input, the previous one the reference), its two
// luma slots, the field buffer and the upsampled field (no cost output: k_nrMv reads none). false = no
// motion for this render (the caller says why and frees what was made).
static bool nativeNrMotionSetup(NativeRife& nr, std::string& err)
{
    const int w = nr.nrW, h = nr.nrH;
    if (!nr.sfNrMv)
    {
        err = "the core shares no MVec texture";
        return false;
    }
    if (!nativeNvofLoad())
    {
        err = "the Optical Flow runtime is unavailable";
        return false;
    }
    if (w < 32 || h < 32)
    {
        err = "below the Optical Flow minimum 32x32";
        return false;
    }
    CUcontext ctx = nullptr;
    if (cuCtxGetCurrent(&ctx) != CUDA_SUCCESS || !ctx)
    {
        err = "no current CUDA context";
        return false;
    }
    if (g_nvofApi.nvCreateOpticalFlowCuda(ctx, &nr.nrOfH) != NV_OF_SUCCESS)
    {
        nr.nrOfH = nullptr;
        err = "nvCreateOpticalFlowCuda failed";
        return false;
    }
    NV_OF_INIT_PARAMS ip{};
    ip.width = (uint32_t)w;
    ip.height = (uint32_t)h;
    ip.outGridSize = (NV_OF_OUTPUT_VECTOR_GRID_SIZE)kNvofGrid;
    ip.hintGridSize = NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED;
    ip.mode = NV_OF_MODE_OPTICALFLOW;
    ip.perfLevel = nativeOfLevel();
    ip.enableExternalHints = NV_OF_FALSE;
    ip.enableOutputCost = NV_OF_FALSE;
    ip.disparityRange = NV_OF_STEREO_DISPARITY_RANGE_UNDEFINED;
    ip.enableRoi = NV_OF_FALSE;
    ip.predDirection = NV_OF_PRED_DIRECTION_FORWARD;
    ip.enableGlobalFlow = NV_OF_FALSE;
    ip.inputBufferFormat = NV_OF_BUFFER_FORMAT_GRAYSCALE8;
    if (g_nvofApi.nvOFInit(nr.nrOfH, &ip) != NV_OF_SUCCESS)
    {
        err = "nvOFInit refused the NR size";
        return false;
    }
    nr.nrOfGw = (w + kNvofGrid - 1) / kNvofGrid;
    nr.nrOfGh = (h + kNvofGrid - 1) / kNvofGrid;
    uint32_t p1 = 0;
    if (!nativeNvofBufOn(nr.nrOfH, w, h, NV_OF_BUFFER_USAGE_INPUT, NV_OF_BUFFER_FORMAT_GRAYSCALE8, nr.nrOfIn[0],
                         nr.nrOfInP[0], nr.nrOfInPitch, "dlss5 input") ||
        !nativeNvofBufOn(nr.nrOfH, w, h, NV_OF_BUFFER_USAGE_INPUT, NV_OF_BUFFER_FORMAT_GRAYSCALE8, nr.nrOfIn[1],
                         nr.nrOfInP[1], p1, "dlss5 input") ||
        !nativeNvofBufOn(nr.nrOfH, nr.nrOfGw, nr.nrOfGh, NV_OF_BUFFER_USAGE_OUTPUT, NV_OF_BUFFER_FORMAT_SHORT2,
                         nr.nrOfOut, nr.nrOfOutP, nr.nrOfOutPitch, "dlss5 flow"))
    {
        err = "Optical Flow buffer creation failed";
        return false;
    }
    if (p1 != nr.nrOfInPitch)
    {
        err = "the two luma slots differ in pitch";
        return false;
    }
    if (g_nvofApi.nvOFSetIOCudaStreams(nr.nrOfH, (CUstream)nr.stream, (CUstream)nr.stream) != NV_OF_SUCCESS)
    {
        err = "nvOFSetIOCudaStreams failed";
        return false;
    }
    if (cudaMalloc((void**)&nr.dNrFlow, 2 * (size_t)w * h * sizeof(float)) != cudaSuccess)
    {
        nr.dNrFlow = nullptr;
        err = "motion field allocation failed";
        return false;
    }
    nr.nrOfCur = 0;
    nr.nrMotion = true;
    return true;
}

// Everything nativeNrMotionSetup made; the caller has drained the stream.
static void nativeNrMotionFree(NativeRife& nr)
{
    for (NvOFGPUBufferHandle* b : {&nr.nrOfIn[0], &nr.nrOfIn[1], &nr.nrOfOut})
        if (*b)
        {
            g_nvofApi.nvOFDestroyGPUBufferCuda(*b);
            *b = nullptr;
        }
    if (nr.nrOfH)
    {
        g_nvofApi.nvOFDestroy(nr.nrOfH);
        nr.nrOfH = nullptr;
    }
    if (nr.dNrFlow)
    {
        cudaFree(nr.dNrFlow);
        nr.dNrFlow = nullptr;
    }
    nr.nrMotion = false;
}

// the motion field of one frame for DLSS 5 (zero-copy route, before the input fence): the frame's
// BT.709 luma into this frame's slot; from the second frame on NVOFA current -> previous, upsampled
// (k_nvofUp) and validated (k_nrMv) into the core's MVec texture; the stream's first frame
// (a Reset) gets a zero field. The slots alternate, so the previous frame's luma stays in place.
static const float kNrMvMargin = 25.0f; // one 8-bit level per pixel of the 5x5 window (k_nrMv)
static const float kNrMvReal = 0.5f;    // a window that changed by more than half a level a pixel ...
static const float kNrMvShare = 0.4f;   // ... keeps a vector that removes at least 40 % of that change (k_nrMv)
static const float kNrMvFlat = 1.0f;    // a window neither test decides keeps a vector of 1 px or more ...
static const float kNrMvCoh = 1.0f;     // ... when the field 8 px away agrees within 1 px (k_nrMv)
static bool nativeNrMotion(NativeRife& nr, const float* frame, int pw, int ps)
{
    cudaStream_t st = nr.stream;
    int w = nr.nrW, h = nr.nrH;
    const int cur = nr.nrOfCur, prev = cur ^ 1;
    nr.nrOfCur = prev;
    const float* r0 = frame;
    int nps = ps, lpitch = (int)nr.nrOfInPitch;
    CUdeviceptr lc = nr.nrOfInP[cur], lp = nr.nrOfInP[prev];
    void* a[] = {(void*)&r0, &nps, &pw, &w, &h, &lc, &lpitch};
    if (cuLaunchKernel(nr.fNvofLuma, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a, nullptr) !=
        CUDA_SUCCESS)
    {
        nr.die("DLSS 5 motion: luma launch failed");
        return false;
    }
    if (nr.nrFirst || nr.nrResetEvery)
    {
        if (!nativeNrMvClear(nr))
        {
            nr.die("DLSS 5 motion: clear failed");
            return false;
        }
        return true;
    }
    NV_OF_EXECUTE_INPUT_PARAMS ei{};
    ei.inputFrame = nr.nrOfIn[cur];      // the forward field of the current frame points into
    ei.referenceFrame = nr.nrOfIn[prev]; // the previous one: current -> previous, as DLSS 5 reads it
    ei.disableTemporalHints = NV_OF_TRUE;
    NV_OF_EXECUTE_OUTPUT_PARAMS eo{};
    eo.outputBuffer = nr.nrOfOut;
    const NV_OF_STATUS s = g_nvofApi.nvOFExecute(nr.nrOfH, &ei, &eo);
    if (s != NV_OF_SUCCESS)
    {
        LOG("native: DLSS 5 motion: nvOFExecute failed (status %d)\n", (int)s);
        nr.die("DLSS 5 motion: execute failed");
        return false;
    }
    int grid = kNvofGrid, vp = (int)nr.nrOfOutPitch, cp = 0;
    CUdeviceptr v = nr.nrOfOutP, c = 0;
    float* flow = nr.dNrFlow;
    float* cost = nullptr; // flow only
    // SMV_NR_MV_MARGIN=F: the validation's margin for an A/B (a large negative value keeps every Optical Flow vector)
    static const float marginSet = [] {
        const std::string m = lkEnv("SMV_NR_MV_MARGIN");
        return m.empty() ? kNrMvMargin : (float)atof(m.c_str());
    }();
    float margin = marginSet;
    // the relative keep (k_nrMv): a window that changed by more than kNrMvReal 8-bit levels a pixel keeps a vector that
    // removes at least kNrMvShare of that change; SMV_NR_MV_REAL=F / SMV_NR_MV_SHARE=F for an A/B (SHARE 0 = off: the
    // absolute test alone, the earlier field)
    static const float realSet = [] {
        const std::string m = lkEnv("SMV_NR_MV_REAL");
        return 25.0f * (m.empty() ? kNrMvReal : (float)atof(m.c_str()));
    }();
    static const float shareSet = [] {
        const std::string m = lkEnv("SMV_NR_MV_SHARE");
        return m.empty() ? kNrMvShare : (float)atof(m.c_str());
    }();
    // a window neither test decides keeps a vector of kNrMvFlat px or more (k_nrMv); SMV_NR_MV_FLAT=F for an A/B (0 =
    // off: such a window gets zero)
    static const float flatSet = [] {
        const std::string m = lkEnv("SMV_NR_MV_FLAT");
        return m.empty() ? kNrMvFlat : (float)atof(m.c_str());
    }();
    // ... when the field 8 px away agrees within kNrMvCoh px (k_nrMv); SMV_NR_MV_COH=F for an A/B (0 = no agreement test)
    static const float cohSet = [] {
        const std::string m = lkEnv("SMV_NR_MV_COH");
        return m.empty() ? kNrMvCoh : (float)atof(m.c_str());
    }();
    float real = realSet, share = shareSet, flat = flatSet, coh = cohSet;
    void* b[] = {&v, &vp, &c, &cp, &nr.nrOfGw, &nr.nrOfGh, &grid, &w, &h, &flow, &cost};
    void* m[] = {&flow, &lc, &lp, &lpitch, &w, &h, &margin, &real, &share, &flat, &coh, &nr.sfNrMv};
    if (cuLaunchKernel(nr.fNvofUp, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, b, nullptr) !=
            CUDA_SUCCESS ||
        cuLaunchKernel(nr.fNrMv, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, m, nullptr) !=
            CUDA_SUCCESS)
    {
        nr.die("DLSS 5 motion: field launch failed");
        return false;
    }
    return true;
}

// DLSS 5 on one frame, in place, before anything reads it: the padded planar model-input frame
// (pw x ph, plane stride ps) whose top-left nrW x nrH is the picture; offline a decoded frame,
// live a captured one after Restore and the resize ((R, G, B) planes, PQ in
// HDR). Every model and every output frame then reads DLSS 5 output: DLSS 5 on
// the rendered frame and frame generation after it. Reset on the stream's first frame (live: and
// after a pause).
// Zero-copy: k_nrIn's frame goes into the core's shared input buffer at its row pitch, the stream
// signals an odd fence value, the core's queue waits for it, evaluates, writes the shared output
// buffer and signals the next even value, and the stream waits for that before k_nrOut reads the
// result: no CPU wait. CPU staging: the core's renderFrame through its upload / readback
// buffers. An evaluate that fails turns the pass off for the rest of the render with a line and
// leaves the frame untouched (nr.nrFailed); there is no retry: NGX has no teardown, so the NR core
// cannot restart in this process.
// SMV_VRAM_MIB: the GPU's memory as the app's memory checks see it, in MiB (a smaller card's trigger test on a
// card everything fits in); 0 = the GPU's own
static uint64_t nativeVideoCardLever()
{
    const std::string lever = lkEnv("SMV_VRAM_MIB");
    const long long mib = lever.empty() ? 0 : atoll(lever.c_str());
    return mib > 0 ? (uint64_t)mib << 20 : 0;
}

// The GPU's video memory as its driver counts it, every process included (NvAPI_GPU_GetMemoryInfoEx, the figures
// nvidia-smi shows; cudaMemGetInfo and the DXGI budget describe this process alone): cardB = its size, freeB = what
// is free right now, moved = how many allocations the driver has moved out of video memory to make room since the
// system started. The GPU is the CUDA device's (matched by its adapter LUID), the first NVIDIA one when that cannot
// be told. SMV_VRAM_MIB shrinks the card and the free memory with it. false = the driver does not say.
static bool nativeVideoMemory(uint64_t& freeB, uint64_t& cardB, uint64_t* moved = nullptr)
{
    static const NvPhysicalGpuHandle gpu = [] {
        NvPhysicalGpuHandle all[NVAPI_MAX_PHYSICAL_GPUS] = {};
        NvU32 n = 0;
        if (NvAPI_Initialize() != NVAPI_OK || NvAPI_EnumPhysicalGPUs(all, &n) != NVAPI_OK || !n)
            return (NvPhysicalGpuHandle) nullptr;
        LUID want{};
        if (n > 1 && nativeCudaLuid(want))
            for (NvU32 i = 0; i < n; i++)
            {
                NvLogicalGpuHandle lg = nullptr;
                LUID id{};
                NV_LOGICAL_GPU_DATA d = {};
                d.version = NV_LOGICAL_GPU_DATA_VER;
                d.pOSAdapterId = &id;
                if (NvAPI_GetLogicalGPUFromPhysicalGPU(all[i], &lg) == NVAPI_OK &&
                    NvAPI_GPU_GetLogicalGpuInfo(lg, &d) == NVAPI_OK && memcmp(&id, &want, sizeof(id)) == 0)
                    return all[i];
            }
        return all[0];
    }();
    NV_GPU_MEMORY_INFO_EX m = {};
    m.version = NV_GPU_MEMORY_INFO_EX_VER;
    if (!gpu || NvAPI_GPU_GetMemoryInfoEx(gpu, &m) != NVAPI_OK || !m.dedicatedVideoMemory)
        return false;
    cardB = m.dedicatedVideoMemory;
    freeB = m.curAvailableDedicatedVideoMemory;
    const uint64_t lever = nativeVideoCardLever();
    if (lever && lever < cardB)
    {
        freeB = freeB > cardB - lever ? freeB - (cardB - lever) : 0;
        cardB = lever;
    }
    if (moved)
        *moved = m.dedicatedVideoMemoryEvictionCount;
    return true;
}

// The video memory a session may still take, in bytes: what is free right now less a reserve of 4 % of the card
// (512 MiB at least) for what the desktop and other apps take next. Memory past it would not fail: Windows moves
// what does not fit into system memory, which slows the session and every other app on the GPU. 0 = no limit
// (--no-gpu-fit, SMV_VRAM_CAP=0, or the driver does not report its memory); 1 = nothing is free.
static uint64_t nativeVideoMemoryRoom()
{
    uint64_t freeB = 0, cardB = 0;
    if (g_noGpuFit || lkEnv("SMV_VRAM_CAP") == "0" || !nativeVideoMemory(freeB, cardB))
        return 0;
    const uint64_t reserve = (std::max)((uint64_t)512 << 20, cardB / 25);
    return freeB > reserve ? freeB - reserve : 1;
}

// What a live session still takes after its output ring, in bytes. The ring is its last allocation (the engines,
// contexts, RTX Video and DLSS 5's passes exist by then), so only what the first frames create comes after it
// (NVAPI's free figure from the ring's creation to the steady session, a quarter on top of the measured): DLSS 5's
// first frames, 120 MiB + 16.6 a megapixel a pass (157 / 226 / 465 MiB at 1920x1080 with 1 / 3 / 10 passes, 139 / 190
// at 854x480 with 3 / 10: its per-size buffers exist before the ring on this route, so they are not added), and
// FRUC's NvOFFRUC buffers with its second midpoint instance, 233 MiB + 138 a padded megapixel of the model
// frame (296 / 521 MiB at 896x512 / 1920x1088), and AMD FSR 4 frame generation's second midpoint instance with its node
// frames and derived level-2+ vectors (fsrfgDeriveOn: the side stream's two vector buffers), 1249 MiB at any size (the
// whole GPU's rise, x4 against x2, at five sizes from 854x480 to 3840x2160: 538 to 999 MiB, 161 to 230 at 2560x1440,
// a quarter on top; the card's figure, not the process's own memory counter, which counts a D3D12 buffer CUDA imports
// twice and reads up to 1.7x the card's rise: a session whose own counter passes the free memory while the card's
// figure fits moves nothing out), or without derived vectors its side Optical Flow session instead of the vector
// buffers (nativeFsrfgAhead, x4 and up, 2 + 98 a megapixel with a quarter less the vectors' 10: 1251 + 88); Intel
// XeSS-FG's, fgKind 1, by the card's figure at the same five sizes: 247 + 265 a megapixel (without derived vectors
// 249 + 353),
// plus, while the bridge gives every instance its own vkd3d device (fsrfg_devices), 300 MiB an instance past the first
// (x4 parked, shared vs own devices: +243 / +49 MiB at 854x480 / 1708x960, x8 at 1280x720 +471 for two: the largest,
// flat, a quarter on top); the other routes add nothing the reserve does not cover. The x8 /
// x16 / x32 ladders' third, fourth and fifth tree levels (an instance and twice the node frames a level) come on top
// when the session's slots reach them (FRUC: the whole GPU's rise over x4 at five sizes, 854x480 to 3840x2160 with the
// fit off, a quarter on top, less the ring's slots past x32's and the pipelined second tree priced below, as a line by
// the padded megapixel that covers every size and never falls with depth: x8 884 + 224, x16 985 + 378, x32 1207 + 784,
// x64 1678 + 879, x128 1873 + 1363, x256 1873 + 2234 a megapixel). x512 (level 9): x128 to x256's step twice over,
// raised to cover its measures (FRUC 854x480 / 1280x720: 1873 + 3977). AMD FSR 4 frame generation's levels
// come from the whole GPU's rise over x2 at five sizes, 854x480 to 3840x2160 (x2 varies by up to ~330 MiB run to run,
// so a size takes the largest of its runs; 3840x2160 to x64 with the fit off): a level's term = that rise, a quarter on
// top, less the late term and the ring's slots past x32's, as a line that covers every size and never falls with depth
// (x8 155 + 86, x16 441 + 260, x32 441 + 626, x64 636 + 1023, x128 996 + 1712, x256 1134 + 3245 a megapixel; x512 by
// the step rule above, raised to cover its 854x480 measure: 1555 + 6309). Intel XeSS-FG's levels by the same measure
// and rule from its own rows (3840x2160 to x64, 2560x1440 to x128, with the fit off): 362 + 103, 362 + 511, 362 + 928,
// 362 + 1506, 362 + 2589, 596 + 4063, x512 1275 + 7011 a megapixel.
constexpr double kNrLiveLateBase = 150.0, kFrucLateBase = 290.0, kFrucLateMp = 173.0;
constexpr double kFsrfgLateBase = 1251.0, kFsrfgLateMp = 88.0, kFsrfgLateDerBase = 1249.0, kFsrfgLateDerMp = 0.0,
                 kFsrfgOwnDevice = 300.0;
constexpr double kXefgLateBase = 249.0, kXefgLateMp = 353.0, kXefgLateDerBase = 247.0, kXefgLateDerMp = 265.0;
constexpr double kXefgDeepBase[] = {362.0, 362.0, 362.0, 362.0, 362.0, 596.0, 1275.0},
                 kXefgDeepMp[] = {103.0, 511.0, 928.0, 1506.0, 2589.0, 4063.0, 7011.0};
constexpr double kFrucDeepBase[] = {884.0, 985.0, 1207.0, 1678.0, 1873.0, 1873.0, 1873.0},
                 kFrucDeepMp[] = {224.0, 378.0, 784.0, 879.0, 1363.0, 2234.0, 3977.0};
constexpr double kFsrfgDeepBase[] = {155.0, 441.0, 441.0, 636.0, 996.0, 1134.0, 1555.0},
                 kFsrfgDeepMp[] = {86.0, 260.0, 626.0, 1023.0, 1712.0, 3245.0, 6309.0};
static_assert(sizeof(kFrucDeepBase) == sizeof(double) * (kFrDepth - 2) &&
                  sizeof(kFrucDeepMp) == sizeof(kFrucDeepBase) && sizeof(kFsrfgDeepBase) == sizeof(kFrucDeepBase) &&
                  sizeof(kFsrfgDeepMp) == sizeof(kFrucDeepBase) && sizeof(kXefgDeepBase) == sizeof(kFrucDeepBase) &&
                  sizeof(kXefgDeepMp) == sizeof(kFrucDeepBase),
              "one memory term a tree level from x8 to the deepest");

// the deepest midpoint-tree level `slots` output frames a source frame reach (x2^L needs 2^L slots; 2 at least, the
// session's instances at most); 0 = no tree
static int nativeTreeLevels(const NativeRife& nr, uint32_t slots)
{
    if (!nr.fruc || !nr.frMp)
        return 0;
    int lv = 2;
    while (lv < kFrDepth && slots >= (2u << lv))
        lv++;
    return (std::min)(nr.frInstMax, lv);
}

static uint64_t nativeLiveLateNeed(const NativeRife& nr, uint32_t slots)
{
    const double mib = 1048576.0;
    double need = 0.0;
    if (nr.liveNr && nr.nrHost)
    {
        const double mp = (double)nr.nrW * nr.nrH / 1e6;
        const int p = g_nrPasses < 1 ? 1 : (g_nrPasses > nr::kMaxPasses ? nr::kMaxPasses : g_nrPasses);
        need += kNrLiveLateBase + nr::kFrameMp * mp * p;
    }
    if (nr.fruc)
    {
        const double mp = (double)nr.pw * nr.ph / 1e6;
        const bool xe = nr.fsrfg && nr.fgKind == 1;
        if (!nr.fsrfg)
            need += kFrucLateBase + kFrucLateMp * mp;
        else if (fsrfgDeriveOn(nr))
            need += xe ? kXefgLateDerBase + kXefgLateDerMp * mp : kFsrfgLateDerBase + kFsrfgLateDerMp * mp;
        else
            need += xe ? kXefgLateBase + kXefgLateMp * mp : kFsrfgLateBase + kFsrfgLateMp * mp;
        // the deepest ladder the slots and the instances reach: x8 = 3 levels, x16 = 4, .. x512 = 9
        const int levels = nativeTreeLevels(nr, slots);
        if (nr.frMp && levels > 2)
            need += !nr.fsrfg ? kFrucDeepBase[levels - 3] + kFrucDeepMp[levels - 3] * mp
                    : xe      ? kXefgDeepBase[levels - 3] + kXefgDeepMp[levels - 3] * mp
                              : kFsrfgDeepBase[levels - 3] + kFsrfgDeepMp[levels - 3] * mp;
        // an own vkd3d device for every FSR 4 instance past the first (x4 = 2 instances, x2^L = L)
        if (nr.fsrfg && g_fsrfgK[nr.fgKind].devices && g_fsrfgK[nr.fgKind].devices() == 1)
            need += ((std::max)(levels, (std::min)(2, nr.frInstMax)) - 1) * kFsrfgOwnDevice;
        // a pipelined group's second tree: the shown pair's nodes beside the new pair's (BGRA8, 4 bytes a pixel)
        if (nr.frMp && !nr.fsrfg)
            need += ((1 << levels) - 1) * 4.0 * mp * 1e6 / mib;
    }
    return (uint64_t)(need * mib);
}

// The output ring's slots the free video memory holds: the room less what the session takes after the ring, over the
// ring's two sets of slotBytes; 2 at least (the session then runs anyway, and says `video memory ran out` when the
// driver moves memory out). ringB = the bytes the ring may take. 0 = no limit (SMV_VRAM_CAP=0, or the driver does not
// report its memory). slots = the ring's asked slots (the deepest midpoint tree they reach is priced).
static uint32_t nativeRingFit(const NativeRife& nr, size_t slotBytes, uint32_t slots, uint64_t& ringB)
{
    ringB = 0;
    const uint64_t room = nativeVideoMemoryRoom();
    if (!room || !slotBytes)
        return 0;
    // a midpoint tree's level the room cannot hold beside the ring's slots for it is given up, one level at a time:
    // the slots then stop below the next power of two, so the ladder never reaches a level that was not priced
    uint32_t s = slots;
    for (;;)
    {
        const uint64_t late = nativeLiveLateNeed(nr, s);
        ringB = room > late ? room - late : 0;
        const uint64_t n = (std::min)((uint64_t)s, ringB / (2ull * slotBytes));
        const int lv = nativeTreeLevels(nr, s);
        if (lv <= 2 || n >= (1ull << lv))
        {
            if (s < slots)
            {
                uint32_t top = 2;
                while ((uint64_t)top * 2 <= n)
                    top *= 2;
                const uint32_t up = top * 2;
                LOG("native: video memory: the midpoint tree stops at x%u (x%u with its ring slots needs about %.0f "
                    "MiB, %.0f MiB are free beside the session)\n",
                    top, up, (double)(nativeLiveLateNeed(nr, up) + 2ull * up * slotBytes) / 1048576.0,
                    (double)room / 1048576.0);
            }
            return (uint32_t)(n < 2 ? 2 : (n > 4096 ? 4096 : n));
        }
        s = (1u << lv) - 1;
    }
}

// A session's video memory check, for every route: nativeVideoMemoryMark at its start (before the engines and
// buffers), nativeVideoMemoryNote once its first frames are out. One line when the driver moved allocations out of
// video memory meanwhile: the session did not fit in what was free, so part of it, or of another app, now lives in
// system memory and runs slower there (the pixels are the same). A live session keeps allocating after its first
// frames (AMD FSR 4 frame generation's second instance and deeper tree levels, a ladder the throttle raises), so when
// the first check found nothing, nativeVideoMemoryLate repeats it at every 2 s stat line until the line is said once.
static uint64_t g_vramMoved = 0, g_vramFree = 0;
static bool g_vramMarked = false, g_vramLate = false;
static void nativeVideoMemoryMark()
{
    uint64_t cardB = 0;
    g_vramLate = false;
    g_vramMarked = nativeVideoMemory(g_vramFree, cardB, &g_vramMoved);
}
// true = the driver moved memory out since the mark (the line said)
static bool nativeVideoMemoryCheck()
{
    uint64_t freeB = 0, cardB = 0, moved = 0;
    if (!nativeVideoMemory(freeB, cardB, &moved) || !(moved > g_vramMoved || (nativeVideoCardLever() && !freeB)))
        return false;
    const double gb = 1024.0 * 1024.0 * 1024.0;
    LOG("%s: video memory ran out: %.1f GB of the GPU's %.1f GB were free when this %s started and it needs "
        "more, so the driver moved memory out to system memory, which slows it and other apps down (the pixels "
        "stay the same); a smaller picture, a lower DLSS mode, fewer DLSS 5 passes or closing other apps leave "
        "more room\n",
        g_offline ? "offline" : "native", (double)g_vramFree / gb, (double)cardB / gb,
        g_offline ? "render" : "session");
    return true;
}
static void nativeVideoMemoryNote()
{
    if (!g_vramMarked)
        return;
    g_vramMarked = false;
    g_vramLate = !nativeVideoMemoryCheck() && !g_offline;
}
static void nativeVideoMemoryLate()
{
    if (g_vramLate && nativeVideoMemoryCheck())
        g_vramLate = false;
}

static bool nativeNrFrame(NativeRife& nr, float* frame, int pw, int ph, int ps)
{
    cudaStream_t st = nr.stream;
    int tw = nr.nrW, th = nr.nrH;
    const size_t bytes = (size_t)tw * th * 8;
    // live HDR planes are PQ, so the SDR-range math runs around the pass (k_nrInPq / k_nrOutPq), unless
    // RTX HDR converts after the pass (nrSdrIn: the planes are the capture's SDR range already); offline
    // HDR video (PQ or HLG codes) takes the same math
    const bool pq = (nr.liveNr && nr.hdr && !nr.nrSdrIn) || nr.nrSrcHdr != 0;
    int hlg = nr.nrSrcHdr == 2 ? 1 : 0;
    float* r0 = frame;
    const float* src = r0;
    int sps = ps;
    // the RGBA16F frame: zero-copy, the core's Color / Output textures (surface objects); staging, dNrIo tight
    uint16_t* io = nr.dNrIo;
    int iop = tw * 4;
    unsigned long long sIn = nr.nrZeroCopy ? nr.sfNrIn : 0, sOut = nr.nrZeroCopy ? nr.sfNrOut : 0;
    void* a[] = {(void*)&src, &sps, &pw, &tw, &th, &io, &iop, &sIn};
    void* b[] = {&io, &iop, &tw, &th, &r0, &ph, &pw, &sps, &sOut};
    void* aq[] = {(void*)&src, &sps, &pw, &tw, &th, &nr.nrSdrWhite, &io, &iop, &hlg, &sIn};
    void* bq[] = {&io, &iop, &tw, &th, &nr.nrSdrWhite, &r0, &ph, &pw, &sps, &hlg, &sOut};
    const unsigned gx = (unsigned)(pw + 15) / 16, gy = (unsigned)(ph + 15) / 16;
    if (nr.nrZeroCopy)
    {
        if (nr.nrMotion && !nativeNrMotion(nr, frame, pw, ps))
            return false;
        cudaExternalSemaphoreSignalParams sp{};
        sp.params.fence.value = ++nr.nrFenceV;
        if (cuLaunchKernel(pq ? nr.fNrInPq : nr.fNrIn, (tw + 15) / 16, (th + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st,
                           pq ? aq : a, nullptr) != CUDA_SUCCESS ||
            cudaSignalExternalSemaphoresAsync(&nr.semNr, &sp, 1, st) != cudaSuccess)
        {
            nr.die("DLSS 5 input handoff failed");
            return false;
        }
        const uint64_t ready = ++nr.nrFenceV;
        const int64_t t0 = nowQpc100();
        std::string err;
        if (!nr.nrHost->submitShared(nr.nrFirst || nr.nrResetEvery, sp.params.fence.value, ready, err))
        {
            LOG("[dlss5] %s; DLSS 5 disabled for the rest of this %s\n", err.c_str(), g_offline ? "render" : "session");
            nr.nrFailed = true;
            return true;
        }
        const double ms = (double)(nowQpc100() - t0) / 10000.0;
        nr.nrMs += ms;
        nr.nrN++;
        if (ms > nr.nrMaxMs)
            nr.nrMaxMs = ms;
        nr.nrFirst = false;
        cudaExternalSemaphoreWaitParams wp{};
        wp.params.fence.value = ready;
        if (cudaWaitExternalSemaphoresAsync(&nr.semNr, &wp, 1, st) != cudaSuccess ||
            cuLaunchKernel(pq ? nr.fNrOutPq : nr.fNrOut, gx, gy, 1, 16, 16, 1, 0, (CUstream)st, pq ? bq : b, nullptr) !=
                CUDA_SUCCESS)
        {
            nr.die("DLSS 5 output handoff failed");
            return false;
        }
        return true;
    }
    if (cuLaunchKernel(pq ? nr.fNrInPq : nr.fNrIn, (tw + 15) / 16, (th + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st,
                       pq ? aq : a, nullptr) != CUDA_SUCCESS ||
        cudaMemcpyAsync(nr.hNrIn, nr.dNrIo, bytes, cudaMemcpyDeviceToHost, st) != cudaSuccess ||
        cudaStreamSynchronize(st) != cudaSuccess)
    {
        nr.die("DLSS 5 input staging failed");
        return false;
    }
    std::vector<float> before; // --nr-delta: the pass's fp32 input (the stream is idle here)
    if (!nr.nrDeltaPath.empty())
    {
        before.resize((size_t)3 * tw * th);
        for (int c = 0; c < 3 && !before.empty(); c++)
            if (cudaMemcpy2D(before.data() + (size_t)c * tw * th, (size_t)tw * sizeof(float), src + (size_t)c * ps,
                             (size_t)pw * sizeof(float), (size_t)tw * sizeof(float), th,
                             cudaMemcpyDeviceToHost) != cudaSuccess)
            {
                LOG("offline: the DLSS 5 change map input download failed, no map\n");
                before.clear();
            }
    }
    const int64_t t0 = nowQpc100();
    std::string err;
    if (!nr.nrHost->renderFrame(nr.hNrIn, nr.hNrOut, nr.nrFirst || nr.nrResetEvery, err))
    {
        LOG("[dlss5] %s; DLSS 5 disabled for the rest of this render\n", err.c_str());
        nr.nrFailed = true;
        return true;
    }
    const double ms = (double)(nowQpc100() - t0) / 10000.0;
    nr.nrMs += ms;
    nr.nrN++;
    if (ms > nr.nrMaxMs)
        nr.nrMaxMs = ms;
    nr.nrFirst = false;
    if (!before.empty() && !pq)
        nativeOfflineNrDelta(nr, before, tw, th);
    if (cudaMemcpyAsync(nr.dNrIo, nr.hNrOut, bytes, cudaMemcpyHostToDevice, st) != cudaSuccess ||
        cuLaunchKernel(pq ? nr.fNrOutPq : nr.fNrOut, gx, gy, 1, 16, 16, 1, 0, (CUstream)st, pq ? bq : b, nullptr) !=
            CUDA_SUCCESS)
    {
        nr.die("DLSS 5 output staging failed");
        return false;
    }
    // HDR video: the change map compares the frame's own codes before and after the pass
    if (!before.empty() && pq)
    {
        std::vector<float> after((size_t)3 * tw * th);
        bool ok = cudaStreamSynchronize(st) == cudaSuccess;
        for (int c = 0; c < 3 && ok; c++)
            ok = cudaMemcpy2D(after.data() + (size_t)c * tw * th, (size_t)tw * sizeof(float), src + (size_t)c * ps,
                              (size_t)pw * sizeof(float), (size_t)tw * sizeof(float), th,
                              cudaMemcpyDeviceToHost) == cudaSuccess;
        if (ok)
            nativeOfflineNrDelta(nr, before, tw, th, &after);
        else
            LOG("offline: the DLSS 5 change map output download failed, no map\n");
    }
    return true;
}

// Live DLSS 5: the NR core brought up as offline does it (a private D3D12 device on
// the CUDA device's adapter) at the model size, with the zero-copy handoff and the CUDA Optical Flow
// motion; nativeGroup runs it once per captured frame on the model frame after Restore and the
// resize. A core that cannot start skips the pass for the session with a line and the session
// runs without it. false = an allocation failed.
static bool nativeLiveNrInit(NativeRife& nr)
{
    if (!g_liveNrCuda)
        return true;
    const int w = nr.w, h = nr.h;
    if ((uint64_t)w * h > 3840ull * 2160ull)
    {
        LOG("live DLSS 5 skipped for this session: the model frame %dx%d is above 3840x2160, the largest size the "
            "DLSS 5 host was probed at\n",
            w, h);
        return true;
    }
    if (w < kNrMinWorkW || h < kNrMinWorkH)
    {
        LOG("live DLSS 5 skipped for this session: the model frame %dx%d is below DLSS 5's minimum, %d px wide and %d "
            "high (a smaller frame resets the GPU)\n",
            w, h, kNrMinWorkW, kNrMinWorkH);
        return true;
    }
    wchar_t exePath[MAX_PATH]{};
    GetModuleFileNameW(nullptr, exePath, MAX_PATH);
    std::wstring nrDir(exePath);
    nrDir.resize(nrDir.find_last_of(L'\\'));
    nrDir += L"\\..\\dlssnr"; // shipped layout: engine\live\smv-live.exe beside engine\dlssnr
    wchar_t ov[MAX_PATH]{};
    if (GetEnvironmentVariableW(L"SMV_DLSSNR_DIR", ov, MAX_PATH) && ov[0])
        nrDir = ov;
    wchar_t full[MAX_PATH]{};
    if (GetFullPathNameW(nrDir.c_str(), MAX_PATH, full, nullptr))
        nrDir = full;
    nr::setModuleDir(nrDir.c_str());
    nr::Settings set;
    set.structure = (float)(g_nrStructure < 0.0 ? 0.0 : (g_nrStructure > 2.0 ? 2.0 : g_nrStructure));
    set.tone = (float)(g_nrTone < 0.0 ? 0.0 : (g_nrTone > 2.0 ? 2.0 : g_nrTone));
    set.skin = (float)(g_nrSkin < 0.0 ? -1.0 : (g_nrSkin > 2.0 ? 2.0 : g_nrSkin));
    set.automask = g_nrAutoMask ? 1 : 0;
    set.style = (g_nrStyle >= 0 && g_nrStyle <= 2) ? g_nrStyle : 0;
    set.passes = g_nrPasses < 1 ? 1 : (g_nrPasses > nr::kMaxPasses ? nr::kMaxPasses : g_nrPasses);
    // SMV_NR_MV=0 = no motion (a measurement lever); without motion every frame is a Reset, since kept
    // history with no motion slides; SMV_NR_AUTOMASK=0 = the mask off (read by the core)
    wchar_t mvv[8]{}, amv[8]{}, ru[8]{}, re[8]{};
    set.motion = !(GetEnvironmentVariableW(L"SMV_NR_MV", mvv, 8) && mvv[0] == L'0');
    const bool autoMaskEnvOff = GetEnvironmentVariableW(L"SMV_NR_AUTOMASK", amv, 8) && amv[0] == L'0';
    const bool autoMask = g_nrAutoMask && !autoMaskEnvOff;
    LUID luid{};
    const bool haveLuid = nativeCudaLuid(luid);
    std::string err;
    nr::Host* host = new nr::Host();
    const int nrc = host->startup((uint32_t)w, (uint32_t)h, set, err, true, haveLuid ? &luid : nullptr);
    if (nrc != 0)
    {
        LOG("live DLSS 5 skipped for this session: the NR core did not start: %s (exit %d)\n", err.c_str(), nrc);
        host->abandon(); // leaked on purpose, like every NGX object here
        return true;
    }
    if (!host->startShared(err) || !nativeNrImport(nr, *host, w, h, err))
    {
        nativeNrReleaseImports(nr);
        LOG("live DLSS 5 skipped for this session: the zero-copy handoff is unavailable (%s)\n", err.c_str());
        host->abandon();
        return true;
    }
    nr.nrHost = host;
    nr.nrW = w;
    nr.nrH = h;
    nr.nrZeroCopy = true;
    if (set.motion && !nativeNrMotionSetup(nr, err))
    {
        nativeNrMotionFree(nr);
        LOG("live DLSS 5: motion vectors unavailable (%s)\n", err.c_str());
    }
    NCHK(cudaMalloc((void**)&nr.dCapPrev, (size_t)nr.cw * nr.ch * (nr.hdr ? 8 : 4)), "alloc dlss5 reuse reference");
    nr.nrResetEvery = !nr.nrMotion || (GetEnvironmentVariableW(L"SMV_NR_RESET_EVERY", re, 8) && re[0] == L'1');
    nr.nrReuse = !(GetEnvironmentVariableW(L"SMV_NR_REUSE", ru, 8) && ru[0] == L'0');
    nr.nrSdrWhite = (float)(g_sdrWhite / 80.0);
    nr.nrFirst = true;
    nr.liveNr = true;
    char skinTxt[24] = ""; // the skin value when it is set apart from the structure
    if (set.skin >= 0.0f)
        sprintf_s(skinTxt, " skin %.2f", set.skin);
    LOG("live DLSS 5 native: on, %dx%d per captured frame on the model frame after Restore and the resize, "
        "zero-copy, structure %.2f tone %.2f%s style %d, passes %d, %s, %s, %s%s\n",
        w, h, set.structure, set.tone, skinTxt, set.style, host->passes(),
        nr.nrMotion ? "motion vectors (NVOFA grid 4), history kept"
                    : (set.motion ? "no motion vectors, every frame a Reset"
                                  : "no motion vectors (SMV_NR_MV=0), every frame a Reset"),
        autoMask ? "auto mask" : (autoMaskEnvOff ? "no auto mask (SMV_NR_AUTOMASK=0)" : "no auto mask"),
        nr.hdr ? "SDR range of the window (HDR highlights keep their light above white)" : "SDR window",
        nr.nrReuse ? ", identical frames reuse the last output" : "");
    if (host->passes() < set.passes)
        LOG("live DLSS 5 native: %d of %d passes (%s)\n", host->passes(), set.passes, host->passNote().c_str());
    if (nr.hdr)
        LOG("live DLSS 5 native: SDR reference white %.0f nits\n", g_sdrWhite);
    return true;
}

// After DLSS 5, once per real frame on the model frame dCur (pw x ph, planes dps apart, nr.w x nr.h
// used) before any model reads it: Adaptive Sharpen (post-processing at the working size), then RTX TrueHDR (SDR -> PQ
// BT.2020, after RTX VSR and every SDR pass), so the model interpolates the finished frames (frame generation last).
// thdr = TrueHDR runs this frame. A failed TrueHDR eval fails an offline render; live it turns the pass off for the
// session and this frame goes to PQ the faithful way. false = a launch failed (die was called).
static bool nativePreModelPost(NativeRife& nr, float* dCur, int dps, bool thdr)
{
    cudaStream_t st = nr.stream;
    int w = nr.w, h = nr.h, pw = nr.pw, ph = nr.ph;
    const bool sharpen = nr.sharpen > 0.0f;
    if (!sharpen && !thdr)
        return true;
    const size_t tight = (size_t)w * h;
    int fps = (int)tight;
    // HDR planes without RTX HDR: the sharpen reads their SDR view, so the SDR range is sharpened exactly as an SDR source's
    // (clamped at SDR white) and a frame of SDR content stays inside the SDR range; the light the view never held is
    // added back after it (k_hdrRestOut, the way Restore and DLSS 5 treat HDR planes)
    const bool hdrView = sharpen && !thdr && nr.encPost && nr.shHdrView;
    if (hdrView)
    {
        if (!nativeHdrEnc(nr, dCur, 0, dps, pw, w, h, nr.dShIn, 0, fps, w, nr.encPost, 1.0f, 0.0f))
        {
            nr.die("sharpen SDR view launch failed");
            return false;
        }
    }
    // the sharpen reads the HDR planes' tight SDR view, else the model frame itself with its pad strides
    float* shSrc = hdrView ? nr.dShIn : dCur;
    int shPs = hdrView ? fps : dps, shRs = hdrView ? w : pw;
    // the sharpen's tight result, padded back into dCur below (TrueHDR's colour step pads itself)
    float* out = nr.dShOut;
    if (thdr)
    {
        if (sharpen)
        {
            void* a[] = {&shSrc, &shPs, &shRs, &w, &h, &nr.sharpen, &nr.dThdrIn, &nr.dSrcG};
            if (cuLaunchKernel(nr.fSharpThdrIn, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a,
                               nullptr) != CUDA_SUCCESS)
            {
                nr.die("sharpThdrIn launch failed");
                return false;
            }
        }
        else
        {
            const void* src = dCur;
            int half = 0, rs = pw;
            void* a[] = {(void*)&src, &half, &dps, &rs, &w, &h, &nr.dThdrIn, &nr.dSrcG};
            if (cuLaunchKernel(nr.fThdrIn, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a, nullptr) !=
                CUDA_SUCCESS)
            {
                nr.die("thdrIn launch failed");
                return false;
            }
        }
        // the bridge reads dThdrIn with a synchronous cuMemcpy2D on the legacy default stream, which is
        // not ordered against this non-blocking stream: sync first
        if (cudaStreamSynchronize(st) != cudaSuccess)
        {
            nr.die("stream sync before the TrueHDR eval failed");
            return false;
        }
        const RtxRect rc{0, 0, (uint32_t)w, (uint32_t)h};
        const int64_t t0 = nowQpc100();
        const unsigned int rv = g_rtxb.evalThdr(nr.dThdrIn, nr.dThdrOut, rc, rc, &nr.thdr);
        // MEASURED, do not remove: the bridge's output copy is a device to device cuMemcpy2D on the
        // legacy default stream, asynchronous to the host, so the eval returning proves nothing about
        // dThdrOut; without this wait the colour step read the PREVIOUS frame's output
        if (cudaDeviceSynchronize() != cudaSuccess)
        {
            nr.die("TrueHDR eval sync failed");
            return false;
        }
        const double ms = (double)(nowQpc100() - t0) / 10000.0;
        nr.thdrMs += ms;
        nr.thdrN++;
        if (ms > nr.thdrMaxMs)
            nr.thdrMaxMs = ms;
        if (rv != 1u)
        {
            if (g_offline)
            {
                LOG("[rtx] TrueHDR eval failed (rc %u)\n", rv);
                nr.die("TrueHDR eval failed");
                return false;
            }
            LOG("native: live TrueHDR eval failed (%u), faithful PQ for the rest of the session\n", rv);
            nr.rtxFailed = true;
            if (sharpen)
            {
                void* a[] = {&shSrc, &shPs, &shRs, &w, &h, &nr.sharpen, &out};
                void* b[] = {&out, &fps, &w, &w, &h, &dCur, &pw, &ph, &dps};
                if (cuLaunchKernel(nr.fSharpPlanar, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a,
                                   nullptr) != CUDA_SUCCESS ||
                    cuLaunchKernel(nr.fPadPlanar, (pw + 15) / 16, (ph + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, b,
                                   nullptr) != CUDA_SUCCESS)
                {
                    nr.die("sharpPlanar launch failed");
                    return false;
                }
            }
            void* a[] = {&dCur, &pw, &ph, &dps, &nr.sdrScale};
            if (cuLaunchKernel(nr.fSdrPq, (pw + 15) / 16, (ph + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a, nullptr) !=
                CUDA_SUCCESS)
            {
                nr.die("sdrPq launch failed");
                return false;
            }
            return true;
        }
        // the colour step writes the padded model frame itself, its pad included
        void* a[] = {&nr.dThdrOut, &nr.dSrcG, &w, &h, &dCur, &pw, &ph, &dps, &nr.rtxMode, &nr.rtxVib, &nr.rtxSb};
        if (cuLaunchKernel(nr.fThdrColor, (pw + 15) / 16, (ph + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a, nullptr) !=
            CUDA_SUCCESS)
        {
            nr.die("thdrColor launch failed");
            return false;
        }
        return true;
    }
    else
    {
        void* a[] = {&shSrc, &shPs, &shRs, &w, &h, &nr.sharpen, &out};
        if (cuLaunchKernel(nr.fSharpPlanar, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a, nullptr) !=
            CUDA_SUCCESS)
        {
            nr.die("sharpPlanar launch failed");
            return false;
        }
        if (hdrView)
        {
            int half = 0;
            void* r[] = {&out, &fps, &w,   (void*)&dCur, &half, &dps,        &pw,
                         &w,   &h,   &out, &fps,         &w,    &nr.encPost, &nr.encWhite};
            if (cuLaunchKernel(nr.fHdrRestOut, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, r,
                               nullptr) != CUDA_SUCCESS)
            {
                nr.die("sharpen HDR codes launch failed");
                return false;
            }
        }
    }
    void* b[] = {&out, &fps, &w, &w, &h, &dCur, &pw, &ph, &dps};
    if (cuLaunchKernel(nr.fPadPlanar, (pw + 15) / 16, (ph + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, b, nullptr) !=
        CUDA_SUCCESS)
    {
        nr.die("padPlanar (after DLSS 5) launch failed");
        return false;
    }
    return true;
}

// the offline RTX HDR emit: the PQ frame (TrueHDR ran before the model) -> the x2rgb10le words in dO and the frame's
// statistics. The sync first makes the stream idle, so the previous frame's statistics copy is complete and reaches
// the accumulators before this frame's copy reuses the host block. false = a launch failed.
static bool nativeOfflinePqOut(NativeRife& nr, const void* src, int half, int ps, int rs, uint8_t* dO)
{
    cudaStream_t st = nr.stream;
    int tw = nr.dw, th = nr.dh;
    if (cudaStreamSynchronize(st) != cudaSuccess)
    {
        nr.die("stream sync before the HDR statistics failed");
        return false;
    }
    nativeOfflineThdrDrain(nr);
    uint32_t* dW = (uint32_t*)dO;
    uint32_t* dHist = (uint32_t*)nr.dThdrStats;
    uint32_t* dMisc = dHist + 1024;
    double* dSum = (double*)(nr.dThdrStats + 1024 * 4 + 16);
    void* a[] = {(void*)&src, &half, &ps, &rs, &tw, &th, &nr.rtxMode, &dW, &dHist, &dMisc, &dSum};
    if (cudaMemsetAsync(nr.dThdrStats, 0, kThdrStatsBytes, st) != cudaSuccess ||
        cuLaunchKernel(nr.fPqOut, (tw + 15) / 16, (th + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a, nullptr) !=
            CUDA_SUCCESS ||
        cudaMemcpyAsync(nr.hThdrStats, nr.dThdrStats, kThdrStatsBytes, cudaMemcpyDeviceToHost, st) != cudaSuccess)
    {
        nr.die("pqOut launch failed");
        return false;
    }
    nr.thdrStatsPending = true;
    return true;
}

// One offline resize stage on a planar frame of sw x sh (ps / rs strides) to tw x th: Restore
// first when withRestore (back to the source size when RTX VSR follows, else folded straight to
// the target), then the resize (RTX VSR when this stage owns it, else clamped Lanczos; offline
// only enlarges, a downscale is folded into the decode). The stages: without a working size the
// model output to the output size; with one the decoded frame to the working size before
// the model, and the final resize (no Restore) from the working size to the output after it.
// staged = nr.dPres holds the tw x th frame; src / ps / rs / srcHalf follow a Restore that hands on
// its fp32 frame; enc = the frame's HDR encoding for Restore (nativeRestoreRun). false = a launch failed.
// one RTX VSR eval of nr.dVsrIn into nr.dVsrOut for the work queued on st: through the bridge's event-ordered export
// when it has one (the thread keeps enqueuing while VSR runs), else (an older bridge, SMV_VSR_SYNC=1) the blocking
// export between a drain of st and a device-wide sync, because its copies ride the legacy default stream. The bridge
// takes and returns the R, G, B, A bytes both buffers hold
static bool nativeVsrEval(NativeRife& nr, const RtxRect& ri, const RtxRect& ro, cudaStream_t st, unsigned int& rv)
{
    if (nr.fsrUp)
        return nativeFsrupEval(nr, ri, ro, st, rv); // AMD FSR upscaling holds RTX VSR's resize, on B, G, R, A buffers
    static const bool sync = [] {
        wchar_t v[8]{};
        return GetEnvironmentVariableW(L"SMV_VSR_SYNC", v, 8) && v[0] != L'0';
    }();
    if (g_rtxb.evalVsrAsync && !sync)
    {
        rv = g_rtxb.evalVsrAsync(nr.dVsrIn, nr.dVsrOut, ri, ro, &nr.vsrSet, (void*)st);
        nr.vsrN++;
        nr.vsrAsync = true;
        return true;
    }
    if (cudaStreamSynchronize(st) != cudaSuccess)
    {
        nr.die("VSR input sync failed");
        return false;
    }
    const int64_t t0 = nowQpc100();
    rv = g_rtxb.evalVsr(nr.dVsrIn, nr.dVsrOut, ri, ro, &nr.vsrSet);
    if (cudaDeviceSynchronize() != cudaSuccess)
    {
        nr.die("VSR eval sync failed");
        return false;
    }
    const double ms = (nowQpc100() - t0) / 1e4;
    nr.vsrMs += ms;
    nr.vsrN++;
    if (ms > nr.vsrMaxMs)
        nr.vsrMaxMs = ms;
    return true;
}

// Restore and the resize of one frame into nr.dPres (staged) or in place. enc = the frame's encoding (0 SDR planes; 1 PQ,
// 2 HLG = offline HDR video, src holds the codes); live hands an SDR view as src and the HDR planes it came from as pl.
// inRange = the caller's range test of src (1 inside the SDR range, 0 outside, -1 none).
static bool nativeOfflineStage(NativeRife& nr, const void*& src, int& ps, int& rs, int& srcHalf, bool& staged, int sw,
                               int sh, int tw, int th, bool withRestore, bool vsrHere, int enc = 0,
                               bool viewFit = false, const void* pl = nullptr, int plHalf = 0, int plPs = 0,
                               int plRs = 0, int inRange = -1)
{
    cudaStream_t st = nr.stream;
    const bool resize = tw != sw || th != sh;
    const bool vsrNow = vsrHere && nr.vsr && !nr.vsrFailed && (g_rtxb.created || nr.fsrUp || nr.a4kUp || nr.acnUp);
    staged = false;
    if (withRestore && nr.restore && nr.restNat && !nr.restFailed)
    {
        if (vsrNow)
        {
            if (!nativeRestoreRun(nr, src, ps, rs, nr.dRest, sw, sh, srcHalf, enc, viewFit, nr.a4kUp && resize))
                return false;
            if (!nr.restFailed)
            {
                src = nr.dRest;
                ps = sw * sh;
                rs = sw;
                srcHalf = 0;
            }
        }
        else
        {
            if (!nativeRestoreRun(nr, src, ps, rs, nr.dPres, tw, th, srcHalf, enc, viewFit))
                return false;
            staged = !nr.restFailed;
        }
    }
    if (!staged && resize)
    {
        bool haveVsr = false;
        const int hlg = (enc ? enc : nr.encPost) == 2 ? 1 : 0;
        if (vsrNow && (nr.a4kUp || nr.acnUp))
        {
            // Anime4K / ArtCNN: the planes (HDR video: their SDR view, as RTX VSR reads it) through its x2 net and the
            // Lanczos pair into nr.dPres; HDR video comes back to the codes in RTX VSR's block below
            const void* a4kSrc = src;
            int aHalf = srcHalf, aps = ps, ars = rs;
            if (enc)
            {
                if (!nativeHdrEnc(nr, src, srcHalf, ps, rs, sw, sh, nr.dVsrView, 0, sw * sh, sw, enc, 1.0f, 0.0f))
                {
                    nr.die(nr.acnUp ? "ArtCNN SDR view launch failed" : "Anime4K SDR view launch failed");
                    return false;
                }
                a4kSrc = nr.dVsrView;
                aHalf = 0;
                aps = sw * sh;
                ars = sw;
            }
            if (!(nr.acnUp ? nativeAcnRun(nr, a4kSrc, aHalf, aps, ars, nr.dPres, tw, th)
                           : nativeA4kRun(nr, a4kSrc, aHalf, aps, ars, nr.dPres, tw, th)))
                return false;
            haveVsr = true;
        }
        else if (vsrNow)
        {
            // HDR video (enc: src holds the codes, restored or not): RTX VSR takes SDR, so it reads the frame's SDR
            // view (knee 1, head 0) and comes back to the codes below; AMD FSR upscaling takes the codes in linear
            // light (k_fsrHdrIn) and the view feeds only its vectors, as live's planes and view do
            // (HDR video through AMD FSR upscaling: the view and the linear frame in ONE pass over the codes, below)
            const bool onePass = enc && nr.fuHdr;
            const void* vsrSrc = src;
            int vHalf = srcHalf, vps = ps, vrs = rs;
            if (enc && !onePass)
            {
                if (!nativeHdrEnc(nr, src, srcHalf, ps, rs, sw, sh, nr.dVsrView, 0, sw * sh, sw, enc, 1.0f, 0.0f))
                {
                    nr.die("VSR SDR view launch failed");
                    return false;
                }
                vsrSrc = nr.dVsrView;
                vHalf = 0;
                vps = sw * sh;
                vrs = sw;
            }
            void* a[] = {(void*)&vsrSrc, &vHalf, &vps, &vrs, &sw, &sh, &nr.dVsrIn};
            if (!onePass && cuLaunchKernel(nr.fsrUp ? nr.fPackBgraRgb : nr.fPackBgra, (sw + 15) / 16, (sh + 15) / 16, 1,
                                           16, 16, 1, 0, (CUstream)st, a, nullptr) != CUDA_SUCCESS)
            {
                nr.die("packBgraRgb launch failed");
                return false;
            }
            if (onePass)
            {
                // the SDR view packed for the vectors (k_hdrEnc + k_packBgraRgb's bytes) and the codes in linear
                // light for the bridge (k_fsrHdrIn's halves) from one read of the frame (k_fsrHdrPack)
                const float scale = 125.0f / nr.encWhite;
                void* f[] = {(void*)&src,  &srcHalf,      &ps,         &rs,        &sw,       &sh,          (void*)&enc,
                             &nr.encWhite, (void*)&scale, (void*)&hlg, &nr.dVsrIn, &nr.dFuIn, &nr.fuInPitch};
                if (cuLaunchKernel(nr.fFsrHdrPack, (sw + 15) / 16, (sh + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, f,
                                   nullptr) != CUDA_SUCCESS)
                {
                    nr.die("FSR HDR pack launch failed");
                    return false;
                }
            }
            else if (nr.fuHdr)
            {
                // AMD FSR upscaling on HDR planes: the frame in linear light into the bridge's buffer. Live hands the
                // planes (pl, the SDR view src came from; after Restore on the view, the restored view goes back to
                // the codes first: dVsrRem as the scratch, the planes as the reference); offline src is the codes
                const void* fp = enc ? src : pl;
                int fh = enc ? srcHalf : plHalf, fps = enc ? ps : plPs, frs = enc ? rs : plRs;
                if (!enc && src == nr.dRest)
                {
                    const int n = sw * sh;
                    if (cudaMemcpyAsync(nr.dVsrRem, nr.dRest, (size_t)3 * n * sizeof(float), cudaMemcpyDeviceToDevice,
                                        st) != cudaSuccess ||
                        !nativeHdrRestOut(nr, nr.dVsrRem, sw, sh, fp, fh, fps, frs, sw, sh, nr.encPost))
                    {
                        nr.die("restored HDR codes (FSR) launch failed");
                        return false;
                    }
                    fp = nr.dVsrRem;
                    fh = 0;
                    fps = n;
                    frs = sw;
                }
                const float scale = 125.0f / nr.encWhite;
                void* f[] = {(void*)&fp, &fh,           &fps,        &frs,      &sw,
                             &sh,        (void*)&scale, (void*)&hlg, &nr.dFuIn, &nr.fuInPitch};
                if (!fp || cuLaunchKernel(nr.fFsrHdrIn, (sw + 15) / 16, (sh + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st,
                                          f, nullptr) != CUDA_SUCCESS)
                {
                    nr.die("FSR HDR input launch failed");
                    return false;
                }
            }
            const RtxRect ri{0, 0, (uint32_t)sw, (uint32_t)sh};
            const RtxRect ro{0, 0, (uint32_t)tw, (uint32_t)th};
            unsigned int rv = 0;
            if (!nativeVsrEval(nr, ri, ro, st, rv))
                return false;
            if (rv == 1u)
                haveVsr = true;
            else
            {
                LOG("[rtx] VSR run failed (rc %u), using Lanczos for the rest of the render\n", rv);
                nr.vsrFailed = true;
            }
        }
        if (haveVsr && nr.fuHdr)
        {
            // AMD FSR upscaling on HDR planes: the bridge's linear-light result back to the codes (k_fsrHdrOut)
            const float inv = nr.encWhite / 125.0f;
            void* a[] = {&nr.dFuOut, &nr.fuOutPitch, (void*)&tw, (void*)&th, (void*)&inv, (void*)&hlg, &nr.dPres};
            if (cuLaunchKernel(nr.fFsrHdrOut, (tw + 15) / 16, (th + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a,
                               nullptr) != CUDA_SUCCESS)
            {
                nr.die("FSR HDR output launch failed");
                return false;
            }
        }
        else if (haveVsr)
        {
            void* a[] = {&nr.dVsrOut, (void*)&tw, (void*)&th, &nr.dPres};
            if (!nr.a4kUp && !nr.acnUp && // Anime4K / ArtCNN wrote nr.dPres itself
                cuLaunchKernel(nr.fsrUp ? nr.fUnpackBgraRgb : nr.fUnpackRgba, (tw + 15) / 16, (th + 15) / 16, 1, 16, 16,
                               1, 0, (CUstream)st, a, nullptr) != CUDA_SUCCESS)
            {
                nr.die("unpackBgraRgb launch failed");
                return false;
            }
            if (enc)
            {
                // HDR video, back to the codes: a frame inside the SDR range takes the view's way back (the caller's
                // range test, else one here on fp32 codes: an fp16 frame widened into dVsrView, free once the view is
                // packed), any other gets the light the view never held from the frame's Lanczos fit (k_hdrRestOut)
                int in = inRange;
                if (in < 0 && nr.dRsRange)
                {
                    const float* f32 = (const float*)src;
                    int fps = ps, frs = rs;
                    if (srcHalf)
                    {
                        void* wa[] = {(void*)&src, &srcHalf, &ps, &rs, &sw, &sh, &nr.dVsrView, &sw, &sh};
                        if (cuLaunchKernel(nr.fFitPlanar, (sw + 15) / 16, (sh + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st,
                                           wa, nullptr) != CUDA_SUCCESS)
                        {
                            nr.die("VSR range widen failed");
                            return false;
                        }
                        f32 = nr.dVsrView;
                        fps = sw * sh;
                        frs = sw;
                    }
                    if ((in = nativeRsInRange(nr, f32, fps, frs, sw, sh, enc)) < 0)
                    {
                        nr.die("VSR range test failed");
                        return false;
                    }
                }
                const int tps = tw * th;
                if (in == 1 ? !nativeHdrFromSdr(nr, nr.dPres, 0, tps, tw, tw, th, nr.dPres, tps, tw, enc)
                            : !nativeFitPlanar(nr, src, srcHalf, ps, rs, sw, sh, nr.dVsrRem, tw, th, st) ||
                                  !nativeHdrRestOut(nr, nr.dPres, tw, th, nr.dVsrRem, 0, tps, tw, tw, th, enc))
                {
                    nr.die("VSR HDR codes launch failed");
                    return false;
                }
            }
        }
        else if (viewFit && enc)
        {
            // an HDR frame inside the SDR range: resized as its SDR view, as the SDR route resizes
            if (!nativeViewFit(nr, src, srcHalf, ps, rs, sw, sh, nr.dPres, tw, th, enc))
            {
                nr.die("SDR view fit launch failed");
                return false;
            }
        }
        else
        {
            if (!nativeFitPlanar(nr, src, srcHalf, ps, rs, sw, sh, nr.dPres, tw, th, st))
            {
                nr.die("fitPlanar launch failed");
                return false;
            }
            int n = 3 * tw * th;
            void* a2[] = {&nr.dPres, &n};
            if (cuLaunchKernel(nr.fClamp01, (n + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, a2, nullptr) !=
                CUDA_SUCCESS)
            {
                nr.die("clamp01 launch failed");
                return false;
            }
        }
        staged = true;
    }
    return true;
}

// Before DLSS 5 and the model: the decoded frame in nr.dSrcPl (srcW x srcH)
// through Restore and RTX VSR / Lanczos to the working size, which is the model size, then into
// the model frame dCur (pw x ph, planes dps apart) with the packers' replicate pad
static bool nativeOfflinePreModel(NativeRife& nr, float* dCur, int dps)
{
    const void* src = nr.dSrcPl;
    int ps = srcW(nr) * srcH(nr), rs = srcW(nr), half = 0;
    bool staged = false;
    // HDR planes: the resize and Restore's reference take the decoded frame's SDR view when it stays inside the range
    int in = 0;
    if (nr.dRsView && (in = nativeRsInRange(nr, nr.dSrcPl, ps, rs, srcW(nr), srcH(nr), nr.encPre)) < 0)
    {
        nr.die("resize range test failed");
        return false;
    }
    if (!nativeOfflineStage(nr, src, ps, rs, half, staged, srcW(nr), srcH(nr), nr.w, nr.h, true, !nr.vsrPost, nr.encPre,
                            in == 1, nullptr, 0, 0, 0, nr.dRsView ? in : -1))
        return false;
    // unstaged = no resize and Restore dropped: the source already has the model size
    const float* from = staged ? nr.dPres : (const float*)src;
    int fps = staged ? nr.w * nr.h : ps, frs = staged ? nr.w : rs;
    void* a[] = {&from, &fps, &frs, &nr.w, &nr.h, &dCur, &nr.pw, &nr.ph, &dps};
    if (cuLaunchKernel(nr.fPadPlanar, (nr.pw + 15) / 16, (nr.ph + 15) / 16, 1, 16, 16, 1, 0, (CUstream)nr.stream, a,
                       nullptr) != CUDA_SUCCESS)
    {
        nr.die("padPlanar launch failed");
        return false;
    }
    return true;
}

// The offline pass chain on one planar model-size frame (ps / rs strides, nr.w x nr.h), then
// the quantisation, into dO as tight rgb48le (out16) or rgb24 at the output size nr.dw x nr.dh:
// nativeOfflineStage (Restore and the resize; with a working size Restore ran before the model and
// only the final resize from the working size is left). DLSS 5, Sharpen and RTX TrueHDR already ran on
// the decoded frame before the interpolation (nativeNrFrame, nativePreModelPost): with RTX HDR the
// frame is PQ, the resize is Lanczos (RTX VSR takes SDR only) and nativeOfflinePqOut writes the
// x2rgb10le words and the frame's statistics. A pass that fails is dropped for the rest of the render
// with a line. false = a launch failed.
static bool nativeOfflineEmit(NativeRife& nr, const void* src, int ps, int rs, uint8_t* dO, bool out16, int srcHalf = 0)
{
    cudaStream_t st = nr.stream;
    const int tw = nr.dw, th = nr.dh;
    bool staged = false; // nr.dPres holds the output-size frame
    if (!nr.nvPre)
    {
        if (!nativeOfflineStage(nr, src, ps, rs, srcHalf, staged, srcW(nr), srcH(nr), tw, th, true, !nr.rtxHdr,
                                nr.encPost))
            return false;
    }
    else if ((nr.w != tw || nr.h != th) &&
             !nativeOfflineStage(nr, src, ps, rs, srcHalf, staged, nr.w, nr.h, tw, th, false, nr.vsrPost, nr.encPost))
        return false;
    const void* fin = staged ? (const void*)nr.dPres : src;
    int fps = staged ? tw * th : ps, frs = staged ? tw : rs, fhalf = staged ? 0 : srcHalf;
    if (nr.rtxHdr)
        return nativeOfflinePqOut(nr, fin, fhalf, fps, frs, dO);
    void* a[] = {(void*)&fin, &fhalf, &fps, &frs, (void*)&tw, (void*)&th, &dO};
    if (cuLaunchKernel(out16 ? nr.fPackOutRaw16 : nr.fPackOutRaw8, (tw + 15) / 16, (th + 15) / 16, 1, 16, 16, 1, 0,
                       (CUstream)st, a, nullptr) != CUDA_SUCCESS)
    {
        nr.die("packOutRaw (passes) launch failed");
        return false;
    }
    return true;
}

static bool nativeGroup(NativeRife& nr, const std::vector<uint8_t>& msg)
{
    if (msg.size() < 4)
        return false;
    uint32_t nfr = 0;
    memcpy(&nfr, msg.data(), 4);
    // the exe's route bit (sendPairStream): this Smooth Motion / AMD FSR 4 group shows the previous pair, pipelined
    const bool pipeAsk = (nfr & 0x80000000u) != 0;
    nfr &= 0x7fffffffu;
    const float* fr = (const float*)(msg.data() + 4);
    if (msg.size() < 4 + 4ull * nfr)
        return false;
    if (nfr > nr.slots)
        nfr = nr.slots;
    const bool pipe =
        pipeAsk && nr.fruc && nr.frMp && nr.havePrev && (nr.fsrfg ? fsrfgPipeOk(nr) : g_fruc.stepI != nullptr);
    // no-engine mode: an empty group stays empty (the pair advances,
    // nothing presented), any other group is exactly ONE real frame through the effects
    // chain, whatever fractions the pair clock asked for
    static const float kReal = 1.0f;
    if (nr.noEngine && nfr)
    {
        nfr = 1;
        fr = &kReal;
    }

    ++nr.seq;
    const uint32_t set = nr.seq % 2;
    const size_t plane = (size_t)nr.ph * nr.pw;
    cudaStream_t st = nr.stream;

    // the previous group's own GPU time (the group ended in drainReady(true), so its events are done)
    if (nr.grArmed && cudaEventQuery(nr.grEv[1]) == cudaSuccess)
    {
        float pre = 0.0f, slots = 0.0f;
        if (cudaEventElapsedTime(&pre, nr.pfEv[0], nr.grEv[0]) == cudaSuccess &&
            cudaEventElapsedTime(&slots, nr.grEv[0], nr.grEv[1]) == cudaSuccess)
        {
            auto ema = [](double& v, double x) { v = v > 0.0 ? v * 0.8 + x * 0.2 : x; };
            if (nr.grTween)
            {
                // a GMFSS catch-up (a skipped frame's feat_ext) is no part of a tweening group's steady price
                float cu = 0.0f;
                if (nr.gmCatchUp && cudaEventElapsedTime(&cu, nr.gmCatchEv[0], nr.gmCatchEv[1]) == cudaSuccess)
                    pre = (std::max)(0.0f, pre - cu);
                ema(nr.gIntMs, pre);
                ema(nr.gTweenMs, slots / nr.grTween);
                nr.gIntUs.store((uint32_t)(nr.gIntMs * 1000.0));
                nr.gTweenUs.store((uint32_t)(nr.gTweenMs * 1000.0));
                // the midpoint tree (Smooth Motion, AMD FSR 4): a call = the compute thread's calls' mean; a slot = the
                // slots' time less the calls' chain in them (a direct group: its calls in the slots; a pipelined one:
                // its levels and the worker's join); a group that made instances is skipped (grPriceSkip)
                if (nr.fruc && nr.frMp && nr.grPriceSkip > 0)
                    nr.grPriceSkip--;
                else if (nr.fruc && nr.frMp && nr.fsrfg && nr.grSlots)
                {
                    // the level-1 chain's parts, on either route (the pipelined route's price, see fgOfEv): its
                    // Optical Flow, its call's FSR span and the session stream's pre outside them. A pipelined group's
                    // level-1 call may still run (the next pack does not wait for it): the spans come from the newest
                    // set whose call is done (the last group's, else the one before), read once, never waited for
                    float l1 = 0.0f, of = 0.0f;
                    int ls = -1;
                    for (int k : {(int)((nr.seq - 1) % 2), (int)(nr.seq % 2)})
                        if (ls < 0 && nr.grL1Armed[k] && cudaEventQuery(nr.fgL1Ev[k][1]) == cudaSuccess)
                            ls = k;
                    const bool l1Ok = ls >= 0 &&
                                      cudaEventElapsedTime(&l1, nr.fgL1Ev[ls][0], nr.fgL1Ev[ls][1]) == cudaSuccess &&
                                      l1 > 0.0f;
                    if (l1Ok && nr.grOfArmed[ls] &&
                        cudaEventElapsedTime(&of, nr.fgOfEv[ls][0], nr.fgOfEv[ls][1]) == cudaSuccess && of > 0.0f)
                    {
                        ema(nr.gFgOfMs, of);
                        ema(nr.gFgL1Ms, l1);
                        ema(nr.gFgPreMs, nr.grPiped ? (double)pre : (std::max)(0.0, (double)pre - of - l1));
                        nr.gFgOfUs.store((uint32_t)(nr.gFgOfMs * 1000.0));
                        nr.gFgL1Us.store((uint32_t)(nr.gFgL1Ms * 1000.0));
                        nr.gFgPreUs.store((uint32_t)(nr.gFgPreMs * 1000.0));
                    }
                    if (ls >= 0)
                        nr.grL1Armed[ls] = nr.grOfArmed[ls] = false;
                    const uint32_t in = nr.grFrCalls - nr.grFrPreCalls;
                    if (nr.grPiped)
                    {
                        // pipelined: the slots = the wait for the shown pair's level-1 node (the level-1 chain's
                        // time), the pair's deeper calls + the stores, so a call = the slots from grLagEv less the
                        // stores, per call; an x2 group (no deeper call) prices the call by its level-1 span, as a
                        // direct x2 group does. The direct price takes it too (its base and store stay the direct x2
                        // groups'): the route's exit reads the call as the GPU runs it now
                        float lag = 0.0f;
                        const double sl =
                            nr.grLagArmed && cudaEventElapsedTime(&lag, nr.grLagEv, nr.grEv[1]) == cudaSuccess ? lag
                                                                                                               : slots;
                        if (in && nr.gFrStoreMs > 0.0)
                        {
                            ema(nr.gFrCallMs, (std::max)(0.0, sl - nr.grSlots * nr.gFrStoreMs) / in);
                            nr.gFrCallUs.store((uint32_t)(nr.gFrCallMs * 1000.0));
                        }
                        else if (!in && l1Ok && nr.gFrL1Ratio > 0.0)
                        {
                            ema(nr.gFrCallMs, nr.gFrL1Ratio * l1);
                            nr.gFrCallUs.store((uint32_t)(nr.gFrCallMs * 1000.0));
                        }
                    }
                    else
                    {
                        // AMD FSR 4 queues its calls on the stream (its CPU call returns at once) and the stream waits
                        // for a node where it is first read, so a call's GPU time lands before or inside the slots: the
                        // prices come off a group's WHOLE time. A group with no call in its slots (x2: level 1 is in
                        // pre) gives the base and a slot's store; a group with calls in them gives a call = its whole
                        // time less the base and its extra stores, per call in the slots. The level-1 call's FSR span
                        // (fgL1Ev) is timed in every group: one with calls in its slots learns the call price over that
                        // span (gFrL1Ratio), an x2 group then prices its call by it, so a load that slowed FSR as it
                        // slowed the rest leaves no stale price once it passed. The base and the store are not
                        // re-measured at x4 and deeper, so the ratio takes them scaled by the span's load (this span
                        // over the x2 groups' span): that load's share of them stays out of the ratio. The call keeps
                        // the share: a held x4 is priced at its cost
                        const double whole = (double)pre + slots;
                        if (!in)
                        {
                            ema(nr.gFrStoreMs, (double)slots / nr.grSlots);
                            ema(nr.gFrBaseMs, whole);
                            if (l1Ok)
                            {
                                ema(nr.gFrL1BaseMs, l1);
                                if (nr.gFrL1Ratio > 0.0)
                                    ema(nr.gFrCallMs, nr.gFrL1Ratio * l1);
                            }
                        }
                        else if (nr.gFrBaseMs > 0.0)
                        {
                            const double rest = nr.gFrBaseMs + (nr.grSlots - 2.0) * nr.gFrStoreMs;
                            ema(nr.gFrCallMs, (std::max)(0.0, whole - rest) / in);
                            if (l1Ok && nr.gFrL1BaseMs > 0.0)
                            {
                                const double own = (whole - rest * l1 / nr.gFrL1BaseMs) / in;
                                if (own > 0.0)
                                    ema(nr.gFrL1Ratio, own / l1);
                            }
                        }
                        nr.gFrCallUs.store((uint32_t)(nr.gFrCallMs * 1000.0));
                        nr.gFrStoreUs.store((uint32_t)(nr.gFrStoreMs * 1000.0));
                        nr.gFrBaseUs.store((uint32_t)(nr.gFrBaseMs * 1000.0));
                    }
                }
                else if (nr.fruc && nr.frMp && nr.grFrCalls && nr.grSlots)
                {
                    // a call = the compute thread's calls' mean in every group (on a pipelined one its calls beside the
                    // worker's read a little dearer: the route's exit stays on the safe side). The pipelined route's
                    // two prices (the compute thread's chain a call incl. the join's wait, a worker call) are measured
                    // while it runs and, while the direct tree runs, follow the call by their ratios to the call before
                    // the last pipelined spell (gFrPipeRm / gFrPipeRw, its first kFrPipeLearn groups): a load that passed
                    // while the direct tree ran leaves no stale pipelined price
                    if (nr.grPiped && nr.grPipeLearn == kFrPipeLearn)
                        nr.gFrCallEntryMs = nr.gFrCallMs; // the call before this pipelined spell
                    ema(nr.gFrCallMs, nr.grFrCallMs / nr.grFrCalls);
                    if (!nr.grPiped)
                    {
                        nr.grPipeLearn = kFrPipeLearn;
                        if (nr.gFrPipeRm > 0.0)
                        {
                            nr.gFrPipeMainMs = nr.gFrCallMs * nr.gFrPipeRm;
                            nr.gFrPipeWkMs = nr.gFrCallMs * nr.gFrPipeRw;
                        }
                    }
                    else
                    {
                        const double main = nr.grFrChainMs / nr.grFrCalls;
                        const double wk = nr.grFrWkCalls ? nr.grFrWkMs / nr.grFrWkCalls : 0.0;
                        ema(nr.gFrPipeMainMs, main);
                        if (wk > 0.0)
                            ema(nr.gFrPipeWkMs, wk);
                        if (nr.grPipeLearn > 0 && nr.gFrCallEntryMs > 0.0 && wk > 0.0)
                        {
                            ema(nr.gFrPipeRm, main / nr.gFrCallEntryMs);
                            ema(nr.gFrPipeRw, wk / nr.gFrCallEntryMs);
                            nr.grPipeLearn--;
                        }
                    }
                    ema(nr.gFrStoreMs, (std::max)(0.0, ((double)slots - nr.grFrChainMs) / nr.grSlots));
                    nr.gFrCallUs.store((uint32_t)(nr.gFrCallMs * 1000.0));
                    nr.gFrStoreUs.store((uint32_t)(nr.gFrStoreMs * 1000.0));
                    nr.gFrPipeMainUs.store((uint32_t)(nr.gFrPipeMainMs * 1000.0));
                    nr.gFrPipeWkUs.store((uint32_t)(nr.gFrPipeWkMs * 1000.0));
                }
            }
        }
    }
    // the level-1 spans stay armed until read (the alternate set is the next group's); this group records into its own
    nr.grArmed = nr.grLagArmed = false;
    nr.fgEvSet = (int)(nr.seq % 2);
    nr.grFrCalls = nr.grFrPreCalls = nr.grFrWkCalls = 0;
    nr.grFrCallMs = nr.grFrPreMs = nr.grFrChainMs = nr.grFrWkMs = 0.0;

    // (1) the exe signalled the capture fence with this sequence number before writing us
    cudaExternalSemaphoreWaitParams wp{};
    wp.params.fence.value = nr.seq;
    if (cudaWaitExternalSemaphoresAsync(&nr.semCap, &wp, 1, st) != cudaSuccess)
    {
        nr.die("wait external semaphore failed");
        return false;
    }
    if (nr.pfEv[0])
        cudaEventRecord(nr.pfEv[0], st);
    // (2) read the shared capture texture; the event marks the exe's texture as free again. Live DLSS 5's reuse
    // test keeps the previous capture: the two buffers trade places and the new one lands in the older
    if (nr.dCapPrev)
        std::swap(nr.dCap, nr.dCapPrev);
    const size_t capRow = (size_t)nr.cw * (nr.hdr ? 8 : 4);
    if (cudaMemcpy2DFromArrayAsync(nr.dCap, capRow, nr.capArr, 0, 0, capRow, nr.ch, cudaMemcpyDeviceToDevice, st) !=
        cudaSuccess)
    {
        nr.die("capture memcpy2DFromArray failed");
        return false;
    }
    cudaEventRecord(nr.capEv, st);
    bool capSent = false;

    // (3) the previous cur becomes prev, then pack the new frame into the cur half
    // (native DRBA keeps its own history ring, so it skips this). dX's halves trade roles (xSwap) unless an IFNet reads
    // dX itself as its x, one tensor with prev in planes 0..2: then the cur half is copied into the prev half
    const size_t mplane = nr.mph ? (size_t)nr.mph * nr.mpw : plane;
    bool halvesTraded = false; // the cur half now holds the frame before the previous one
    if (nr.havePrev && !nr.noEngine && !nr.drba)
    {
        if (!nr.gmfss && !nr.nvof && !nr.fruc && !nr.mph && !nr.xHalf)
            cudaMemcpyAsync(nr.dX, nr.dX + 3 * plane, 3 * plane * sizeof(float), cudaMemcpyDeviceToDevice, st);
        else
        {
            nr.xSwap = !nr.xSwap;
            halvesTraded = true;
        }
        // the frames the IFNet reads keep the same two frames: the motion frames, or the fp16 copy an
        // fp16-x IFNet reads
        if (nr.mph)
            cudaMemcpyAsync(nr.dM, nr.dM + 3 * mplane, 3 * mplane * sizeof(uint16_t), cudaMemcpyDeviceToDevice, st);
        else if (nr.xHalf)
            cudaMemcpyAsync(nr.dXh, nr.dXh + 3 * plane, 3 * plane * sizeof(uint16_t), cudaMemcpyDeviceToDevice, st);
    }
    // DRBA: the ring slot of the frame this group pushes (the previous group's tweens, on st before it, are done with it)
    float* dCur = nr.drba ? drbaFrame(nr, nr.drFid + 1) : nativeXCur(nr);
    // live DLSS 5's identical-frame reuse: a capture byte-identical to the previous one takes the
    // last DLSS 5 output and skips TrueHDR, Restore, the resize and the evaluate (the history stays
    // untouched, so a paused picture stays exactly still)
    bool nrSame = false;
    if (nr.liveNr && !nr.nrPaused && !nr.nrFailed)
    {
        if (g_liveNrReset.exchange(false))
            nr.nrFirst = true; // the overlay was hidden: a new stream
        const long long nb = (long long)nr.cw * nr.ch * (nr.hdr ? 8 : 4);
        if (nr.nrReuse && nr.nrHaveLast && !nr.nrFirst)
        {
            void* ad[] = {&nr.dCapPrev, &nr.dCap, (void*)&nb, &nr.dStaticFlag};
            const unsigned blocks = (unsigned)((nb + 16LL * 256 - 1) / (16LL * 256));
            if (cudaMemsetAsync(nr.dStaticFlag, 0, sizeof(int), st) != cudaSuccess ||
                cuLaunchKernel(nr.fRawDiff, blocks, 1, 1, 256, 1, 1, 0, (CUstream)st, ad, nullptr) != CUDA_SUCCESS ||
                cudaMemcpyAsync(nr.hStaticFlag, nr.dStaticFlag, sizeof(int), cudaMemcpyDeviceToHost, st) !=
                    cudaSuccess ||
                cudaStreamSynchronize(st) != cudaSuccess)
            {
                nr.die("DLSS 5 reuse test failed");
                return false;
            }
            nrSame = *nr.hStaticFlag == 0;
        }
        if (nrSame)
        {
            // the last output is the previous frame's cur half, read only since: it is still the cur half where
            // nothing moved it (the copy shift, echo) and the prev half where the halves traded places; DRBA's is
            // the ring slot it was built in. The motion frame of an identical capture is the previous one (the shift
            // put it in the prev half; DRBA's push copies it inside its ring)
            if ((halvesTraded && cudaMemcpyAsync(dCur, nativeXPrev(nr), 3 * plane * sizeof(float),
                                                 cudaMemcpyDeviceToDevice, st) != cudaSuccess) ||
                (nr.drba && cudaMemcpyAsync(dCur, drbaFrame(nr, nr.drFid), 3 * plane * sizeof(float),
                                            cudaMemcpyDeviceToDevice, st) != cudaSuccess) ||
                (nr.mph && nr.havePrev && !nr.drba &&
                 cudaMemcpyAsync(nr.dM + 3 * mplane, nr.dM, 3 * mplane * sizeof(uint16_t), cudaMemcpyDeviceToDevice,
                                 st) != cudaSuccess))
            {
                nr.die("DLSS 5 reuse copy failed");
                return false;
            }
            nr.nrReused++;
        }
    }
    // live with RTX HDR: Restore, the resize and DLSS 5 work on the SDR picture and RTX TrueHDR converts it
    // after them (nativePreModelPost). The capture's SDR range as planes (R, G, B, sRGB-encoded
    // against the SDR reference white) in dSrcG; without RTX HDR, or once it failed, the chain works on PQ from
    // the capture on.
    const bool sdrPre = !nrSame && nr.rtxHdr && !nr.rtxFailed && g_rtxb.created;
    nr.nrSdrIn = sdrPre;
    if (sdrPre)
    {
        void* ae[] = {&nr.dCap, &nr.cw, &nr.ch, &nr.sdrScale, &nr.dSrcG};
        if (cuLaunchKernel(nr.fSdrEncode, (nr.cw + 15) / 16, (nr.ch + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, ae,
                           nullptr) != CUDA_SUCCESS)
        {
            nr.die("sdrEncode launch failed");
            return false;
        }
    }
    // RTX VSR before the model (the working size above the capture while the fit after the model does not
    // enlarge): the capture's planes through Restore (folded back to the capture size) and VSR to the working
    // size, offline's pre-model stage. It replaces Restore and the resize below; after a VSR failure those
    // run instead. VSR takes SDR: RTX HDR's SDR planes as they are (TrueHDR follows, NVIDIA's order), other HDR
    // planes as their SDR view, back to the codes after it with the light the view never held.
    bool restored = false;
    bool hdrIn = false; // this capture stayed inside the SDR range (the range test ran and said so)
    if (!nrSame && nr.vsrPre && !nr.vsrFailed)
    {
        const int cps = nr.cw * nr.ch;
        const bool viewHere = nr.hdr && !sdrPre;
        if (viewHere)
        {
            void* a0[] = {&nr.dCap, &nr.cw, &nr.ch, &nr.dCapF};
            if (cuLaunchKernel(nr.fPqPlanar, (nr.cw + 15) / 16, (nr.ch + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a0,
                               nullptr) != CUDA_SUCCESS ||
                !nativeHdrEnc(nr, nr.dCapF, 0, cps, nr.cw, nr.cw, nr.ch, nr.dVsrView, 0, cps, nr.cw, 1, 1.0f, 0.0f))
            {
                nr.die("capture SDR view (RTX VSR) launch failed");
                return false;
            }
            int in = 0;
            if (nr.dRsView && (in = nativeRsInRange(nr, nr.dCapF, cps, nr.cw, nr.cw, nr.ch, 1)) < 0)
            {
                nr.die("resize range test failed");
                return false;
            }
            hdrIn = in == 1;
        }
        else if (!sdrPre)
        {
            void* ap[] = {&nr.dCap, &nr.cw, &nr.ch, &nr.dCapF, &nr.ch, &nr.cw, (void*)&cps};
            if (cuLaunchKernel(nr.fPackInDirect, (nr.cw + 15) / 16, (nr.ch + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st,
                               ap, nullptr) != CUDA_SUCCESS)
            {
                nr.die("capture planes (RTX VSR) launch failed");
                return false;
            }
        }
        const void* src = viewHere ? nr.dVsrView : (sdrPre ? nr.dSrcG : nr.dCapF);
        int ps = cps, rs = nr.cw, half = 0;
        bool staged = false;
        if (!nativeOfflineStage(nr, src, ps, rs, half, staged, nr.cw, nr.ch, nr.w, nr.h, true, true, 0, false, nr.dCapF,
                                0, cps, nr.cw))
            return false;
        if (staged)
        {
            const int wps = nr.w * nr.h;
            // (AMD FSR upscaling on HDR planes staged the codes itself: k_fsrHdrOut; after a failure the view's fit)
            if (viewHere && !(nr.fuHdr && !nr.vsrFailed) &&
                (hdrIn ? !nativeHdrFromSdr(nr, nr.dPres, 0, wps, nr.w, nr.w, nr.h, nr.dPres, wps, nr.w, 1)
                       : !nativeFitPlanar(nr, nr.dCapF, 0, cps, nr.cw, nr.cw, nr.ch, nr.dVsrRem, nr.w, nr.h, st) ||
                             !nativeHdrRestOut(nr, nr.dPres, nr.w, nr.h, nr.dVsrRem, 0, wps, nr.w, nr.w, nr.h, 1)))
            {
                nr.die("capture HDR codes (RTX VSR) launch failed");
                return false;
            }
            const float* from = nr.dPres;
            int fps = wps, dps = (int)plane;
            void* a[] = {&from, &fps, &nr.w, &nr.w, &nr.h, &dCur, &nr.pw, &nr.ph, &dps};
            if (cuLaunchKernel(nr.fPadPlanar, (nr.pw + 15) / 16, (nr.ph + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a,
                               nullptr) != CUDA_SUCCESS)
            {
                nr.die("padPlanar (RTX VSR) launch failed");
                return false;
            }
            restored = true;
        }
    }
    // Restore: once per captured frame, on the capture itself (its SDR planes when
    // RTX HDR converts later), folded straight to the model size and into the model frame with the packers'
    // replicate pad, so every model reads restored frames; it replaces the resize below. When the
    // engine refuses, the pass is dropped for the session and the resize below runs instead.
    if (!restored && !nrSame && nr.restPre && nr.restNat && !nr.restFailed)
    {
        const int cps = nr.cw * nr.ch;
        if (!sdrPre)
        {
            // the capture as capture-size planes in the chain's colour space (PQ in HDR)
            void* ap[] = {&nr.dCap, &nr.cw, &nr.ch, &nr.dCapF, &nr.ch, &nr.cw, (void*)&cps};
            void* aq[] = {&nr.dCap, &nr.cw, &nr.ch, &nr.dCapF};
            if (cuLaunchKernel(nr.hdr ? nr.fPqPlanar : nr.fPackInDirect, (nr.cw + 15) / 16, (nr.ch + 15) / 16, 1, 16,
                               16, 1, 0, (CUstream)st, nr.hdr ? aq : ap, nullptr) != CUDA_SUCCESS)
            {
                nr.die("capture planes launch failed");
                return false;
            }
        }
        float* capPl = sdrPre ? nr.dSrcG : nr.dCapF;
        // HDR planes: Restore's reference at the working size takes the capture's SDR view when it stays in range
        int in = 0;
        if (!sdrPre && nr.dRsView && (in = nativeRsInRange(nr, nr.dCapF, cps, nr.cw, nr.cw, nr.ch, nr.encPost)) < 0)
        {
            nr.die("resize range test failed");
            return false;
        }
        hdrIn = in == 1;
        if (!nativeRestoreRun(nr, capPl, cps, nr.cw, nr.dRest, nr.w, nr.h, 0, sdrPre ? 0 : nr.encPost, in == 1,
                              nr.a4kUp && nr.vsr && !nr.vsrFailed))
            return false;
        if (!nr.restFailed)
        {
            int fps = nr.w * nr.h, dps = (int)plane;
            const float* from = nr.dRest;
            void* a[] = {&from, &fps, &nr.w, &nr.w, &nr.h, &dCur, &nr.pw, &nr.ph, &dps};
            if (cuLaunchKernel(nr.fPadPlanar, (nr.pw + 15) / 16, (nr.ph + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a,
                               nullptr) != CUDA_SUCCESS)
            {
                nr.die("padPlanar (restore) launch failed");
                return false;
            }
            restored = true;
        }
    }
    if (!restored && !nrSame)
    {
        const int ps = (int)plane;
        if (sdrPre)
        {
            // the SDR planes already exist at capture resolution, so resize them at every
            // image scale; at 1.00 the Lanczos taps sit on identity positions (weight 1 on the
            // pixel itself, 0 elsewhere) and the pair is an exact copy plus the replicate pad.
            const void* tabH = rsTabFor(nr, nr.cw, nr.w, st);
            const void* tabV = rsTabFor(nr, nr.ch, nr.h, st);
            if (!tabH || !tabV)
            {
                nr.die("resize tap tables (RTX HDR) failed");
                return false;
            }
            void* a1[] = {&nr.dSrcG, &nr.cw, &nr.ch, &nr.dTmp, &nr.w, (void*)&tabH};
            if (cuLaunchKernel(nr.fResizeHf, (nr.w + 15) / 16, (nr.ch + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a1,
                               nullptr) != CUDA_SUCCESS)
            {
                nr.die("resizeHf (RTX HDR) launch failed");
                return false;
            }
            void* a2[] = {&nr.dTmp, &nr.w, &nr.ch, &dCur, &nr.h, &nr.ph, &nr.pw, (void*)&ps, (void*)&tabV};
            if (cuLaunchKernel(nr.fResizeV, (nr.pw + 15) / 16, (nr.ph + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a2,
                               nullptr) != CUDA_SUCCESS)
            {
                nr.die("resizeV (RTX HDR) launch failed");
                return false;
            }
        }
        else if (nr.w == nr.cw && nr.h == nr.ch)
        {
            void* a[] = {&nr.dCap, &nr.cw, &nr.ch, &dCur, &nr.ph, &nr.pw, (void*)&ps};
            if (cuLaunchKernel(nr.hdr ? nr.fPackInDirectHdr : nr.fPackInDirect, (nr.pw + 15) / 16, (nr.ph + 15) / 16, 1,
                               16, 16, 1, 0, (CUstream)st, a, nullptr) != CUDA_SUCCESS)
            {
                nr.die("packInDirect launch failed");
                return false;
            }
        }
        else
        {
            const void* rsrc = nr.hdr ? (const void*)nr.dCapF : (const void*)nr.dCap;
            bool view = false;
            if (nr.hdr)
            {
                // convert the whole capture to PQ first, then resize ON PQ (the fill rule:
                // rescale on PQ, never rescale scRGB and convert after)
                void* a0[] = {&nr.dCap, &nr.cw, &nr.ch, &nr.dCapF};
                if (cuLaunchKernel(nr.fPqPlanar, (nr.cw + 15) / 16, (nr.ch + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st,
                                   a0, nullptr) != CUDA_SUCCESS)
                {
                    nr.die("pqPlanar launch failed");
                    return false;
                }
                // a capture inside the SDR range is resized as its SDR view (the SDR route's resize, clamped at SDR
                // white) and comes back to PQ after it, so its ringing never crosses white
                const int cps = nr.cw * nr.ch;
                int in = 0;
                if (nr.dRsView && (in = nativeRsInRange(nr, nr.dCapF, cps, nr.cw, nr.cw, nr.ch, nr.encPost)) < 0)
                {
                    nr.die("resize range test failed");
                    return false;
                }
                view = in == 1;
                hdrIn = view;
                if (view && !nativeHdrEnc(nr, nr.dCapF, 0, cps, nr.cw, nr.cw, nr.ch, nr.dRsView, 0, cps, nr.cw,
                                          nr.encPost, 1.0f, 0.0f))
                {
                    nr.die("resize SDR view launch failed");
                    return false;
                }
                if (view)
                    rsrc = nr.dRsView;
            }
            const void* tabH = rsTabFor(nr, nr.cw, nr.w, st);
            const void* tabV = rsTabFor(nr, nr.ch, nr.h, st);
            if (!tabH || !tabV)
            {
                nr.die("resize tap tables failed");
                return false;
            }
            void* a1[] = {(void*)&rsrc, &nr.cw, &nr.ch, &nr.dTmp, &nr.w, (void*)&tabH};
            if (cuLaunchKernel(nr.hdr ? nr.fResizeHf : nr.fResizeH, (nr.w + 15) / 16, (nr.ch + 15) / 16, 1, 16, 16, 1,
                               0, (CUstream)st, a1, nullptr) != CUDA_SUCCESS)
            {
                nr.die("resizeH launch failed");
                return false;
            }
            void* a2[] = {&nr.dTmp, &nr.w, &nr.ch, &dCur, &nr.h, &nr.ph, &nr.pw, (void*)&ps, (void*)&tabV};
            if (cuLaunchKernel(nr.fResizeV, (nr.pw + 15) / 16, (nr.ph + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a2,
                               nullptr) != CUDA_SUCCESS)
            {
                nr.die("resizeV launch failed");
                return false;
            }
            if (view && !nativeHdrFromSdr(nr, dCur, 0, ps, nr.pw, nr.pw, nr.ph, dCur, ps, nr.pw, nr.encPost))
            {
                nr.die("resize HDR codes launch failed");
                return false;
            }
        }
    }
    if (!nrSame)
        nr.hdrInBits = (nr.hdrInBits << 1) | (hdrIn ? 1u : 0u);
    // DLSS 5: once per captured frame on the model frame, after Restore and the
    // resize and before any model reads it; then Sharpen and RTX TrueHDR (nativePreModelPost), and the
    // frame the model reads is kept for the reuse above
    if (nr.liveNr && !nr.nrPaused && !nr.nrFailed && !nrSame && !nativeNrFrame(nr, dCur, nr.pw, nr.ph, (int)plane))
        return false;
    // RIFE's motion frame (two domains): the finished picture as the IFNet would have read it, minus
    // what costs RIFE its motion: the HDR desktop's PQ encoding (read as SDR sRGB over the desktop's
    // SDR white) and an enlarge (Lanczos back to the capture size), padded to the motion frame. Taken
    // after Sharpen, or before the post with RTX HDR (the picture is still SDR there, Sharpen is fused
    // into TrueHDR's input). An identical capture copied it above.
    auto motionFrame = [&]() -> bool {
        const float* msrc = dCur;
        int sps = (int)plane, srs = nr.pw, sw = nr.w, sh = nr.h;
        if (nr.dMFit)
        {
            int n = 3 * nr.mw * nr.mh;
            void* b[] = {&nr.dMFit, &n};
            if (!nativeFitPlanar(nr, dCur, 0, sps, nr.pw, nr.w, nr.h, nr.dMFit, nr.mw, nr.mh, st) ||
                cuLaunchKernel(nr.fClamp01, (n + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, b, nullptr) !=
                    CUDA_SUCCESS)
                return false;
            msrc = nr.dMFit;
            sps = nr.mw * nr.mh;
            srs = nr.mw;
            sw = nr.mw;
            sh = nr.mh;
        }
        int dps = (int)mplane, mode = nr.hdr && !sdrPre ? 1 : 0;
        float white = (float)(g_sdrWhite / 80.0), knee = kMotionPqKnee, cap = kMotionPqCap;
        uint16_t* mdst = nr.drba ? drbaMotionNext(nr) : nr.dM + 3 * mplane;
        int zp = nr.zeroPad ? 1 : 0, stb = nr.drAuthor ? 1 : 0;
        void* a[] = {(void*)&msrc, &sps,    &srs,  &sw,    &sh,   &mdst, &dps, &nr.mpw,
                     &nr.mpw,      &nr.mph, &mode, &white, &knee, &cap,  &zp,  &stb};
        return cuLaunchKernel(nr.fMotionIn, (nr.mpw + 15) / 16, (nr.mph + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a,
                              nullptr) == CUDA_SUCCESS;
    };
    const bool motionHere = nr.mph && !nrSame;
    if (motionHere && sdrPre && !motionFrame())
    {
        nr.die("motion frame launch failed");
        return false;
    }
    if (!nrSame && !nativePreModelPost(nr, dCur, (int)plane, sdrPre))
        return false;
    if (motionHere && !sdrPre && !motionFrame())
    {
        nr.die("motion frame launch failed");
        return false;
    }
    if (nr.liveNr && !nr.nrPaused && !nr.nrFailed && !nrSame)
        nr.nrHaveLast = true;

    // native DRBA: the new frame and its encode into the history ring; the group then shows
    // the LAGGED frame k - 1 as its real frame (the first capture: itself, one slot:
    // DRBA's first group)
    int drHist = 0;
    float* drLag = dCur;
    if (nr.drba)
    {
        if (!nativeDrbaPush(nr, dCur, nrSame))
            return false;
        drHist = nr.drFid < 4 ? (int)nr.drFid : 4;
        if (drHist >= 2)
            drLag = drbaFrame(nr, nr.drFid - 1);
        else if (nfr > 1)
            nfr = 1;
    }

    // (4) the pair. How many tweens this group asks for, in fracs order, decides whether the
    // pair-level work is needed at all.
    uint32_t nTween = 0;
    for (uint32_t i = 0; i < nfr; i++)
        if (fr[i] < 0.999f)
            nTween++;
    if (!nr.havePrev)
        nTween = 0; // first pair: nothing to interpolate toward
    // IDENTICAL PAIR: the two packed inputs are compared element by element on
    // the device and the flag is read back once. Equal = no motion exists in this pair, so the
    // tween work is dropped (exactly the tween-less group the adaptive ladder already
    // produces: the per-frame chain state, the encode or feat_ext, still runs for the next
    // pair) and every slot presents the real frame below. The readback sync costs only the
    // overlap between this group's pack and its own tweens: the compute thread already waits
    // for the whole group in drainReady(true) before it takes the next message.
    bool pairStatic = false;
    // native DRBA tests the LAGGED pair (k-2, k-1): every tween of its group lies inside it
    // (DRBA's static rule, three frames of history needed)
    if (nTween && g_staticHold && nr.fPairDiff && nr.dStaticFlag && nr.hStaticFlag && (!nr.drba || drHist >= 3))
    {
        const int n = (int)(3 * plane);
        float* dPrev = nr.drba ? drbaFrame(nr, nr.drFid - 2) : nativeXPrev(nr);
        float* dNext = nr.drba ? drLag : dCur;
        if (cudaMemsetAsync(nr.dStaticFlag, 0, sizeof(int), st) != cudaSuccess)
        {
            nr.die("static flag clear failed");
            return false;
        }
        void* ad[] = {&dPrev, &dNext, (void*)&n, &nr.dStaticFlag};
        if (cuLaunchKernel(nr.fPairDiff, (n + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, ad, nullptr) !=
            CUDA_SUCCESS)
        {
            nr.die("pairDiff launch failed");
            return false;
        }
        if (cudaMemcpyAsync(nr.hStaticFlag, nr.dStaticFlag, sizeof(int), cudaMemcpyDeviceToHost, st) != cudaSuccess ||
            cudaStreamSynchronize(st) != cudaSuccess)
        {
            nr.die("static flag readback failed");
            return false;
        }
        if (*nr.hStaticFlag == 0)
        {
            pairStatic = true;
            nTween = 0;
            g_staticHeld = ++nr.staticN;
        }
    }
    // RIFE: the pair encode. The previous frame's encode is REUSED (encode is a pure function
    // of the frame, so this is exact and halves the Head work per pair).
    nr.fCur ^= 1;
    if (nr.gmfss)
    {
        // GMFSS: feat_ext of the new frame, then the halves, the bidir flow,
        // the metrics and the two pyramid levels when this group interpolates
        if (!nativeGmfssPair(nr, nativeXPrev(nr), dCur, nTween > 0))
            return false;
    }
    else if (nr.nvof)
    {
        // the nvof model: no per-frame state (both frames go to gray8 per pair), so a
        // tween-less or identical pair costs nothing; the identical-pair gate above already
        // zeroed nTween, an identical pair never reaches the Optical Flow engine
        if (nTween && !nativeNvofPair(nr, nativeXPrev(nr), dCur))
            return false;
    }
    else if (nr.fruc && pipe)
    {
        // Smooth Motion / AMD FSR 4 pipelined: the static test above judged the NEW pair, which the next group shows;
        // this group shows the previous pair (nativeFrucPipeAdvance / nativeFsrfgPipeAdvance), held when it was
        // identical or never split. The levels' instances are made here, before instance 0's worker / call runs
        const bool curStatic = pairStatic;
        if (!(nr.fsrfg ? nativeFsrfgPipeAdvance(nr, dCur, curStatic) : nativeFrucPipeAdvance(nr, dCur, curStatic)))
            return false;
        pairStatic = !nr.frLagHave || nr.frLagStatic;
        nTween = 0;
        if (!pairStatic)
        {
            double tw[64];
            for (uint32_t i = 0; i < nfr && nTween < 64; i++)
                if (fr[i] < 0.999f)
                    tw[nTween++] = fr[i];
            const int want = nTween ? (std::min)(nativeFrucDepth(nr, tw, nTween), nr.frInstMax) : 0;
            if (nr.frInst < want)
            {
                if (nr.fsrfg)
                    nativeFsrfgGrow(nr, want);
                else
                    nativeFrucGrow(nr, want);
            }
            if (want)
                nr.frPipeDepth = want;
        }
    }
    else if (nr.fruc)
    {
        // Smooth Motion: every frame is packed (the next pair's start), the bridge itself
        // runs per tween below; an identical pair got nTween = 0 above and never reaches FRUC
        if (nr.fsrfg && !nativeFsrfgPipeExit(nr))
            return false;
        nr.frPipePrev = false;
        nr.frCurStatic = pairStatic;
        if (!nativeFrucPair(nr, dCur, nTween))
            return false;
        if (nTween > 1)
        {
            // the tweens in slot order, for the parallel instances
            double tw[64];
            uint32_t n = 0;
            for (uint32_t i = 0; i < nfr && n < 64; i++)
                if (fr[i] < 0.999f)
                    tw[n++] = fr[i];
            nativeFrucPlan(nr, tw, n);
        }
    }
    else if (nr.drba)
    {
        // native DRBA: the encode already went into the ring (nativeDrbaPush), the windows
        // are built lazily by the tweens below
    }
    else if (!nr.noEngine)
    {
        // fp16 frames: the new frame into the fp16 copy's cur half for an fp16-x IFNet (every new frame, static pairs
        // too); RIFE's two domains: the motion frame
        if (!nr.mph && nr.xHalf && !nativeF2h(nr, dCur, nr.dXh + 3 * plane, 3 * plane, st))
        {
            nr.die("f2h launch failed");
            return false;
        }
        // the encode reads the motion frame (fp16), else the new frame itself (fp32), into dF in the IFNet's dtype
        if (!(nr.mph ? nativeEncodeRun(nr, nr.dM + 3 * mplane, 1, nr.mpw, nr.mph, nr.dF[nr.fCur], st)
                     : nativeEncodeRun(nr, dCur, 0, nr.pw, nr.ph, nr.dF[nr.fCur], st)))
            return false;
    }

    // (5) slots. Tokens go out in slot-index order, exactly like _drain_ready.
    if (nr.grEv[0])
        cudaEventRecord(nr.grEv[0], st);
    nr.grFrPreMs = nr.grFrCallMs;
    nr.grFrPreCalls = nr.grFrCalls;
    // a pipelined group: the new pair's levels 1..k start on instance 0's worker beside this group's slots (joined in
    // nativeFrucLagTween, or below when the shown pair is held)
    if (pipe && !nr.fsrfg && !nativeFrucPipeSpec(nr))
        return false;
    const int pitchI = (int)nr.pitch;
    uint32_t sent = 0;
    auto drainReady = [&](bool block) {
        if (!capSent &&
            (block ? cudaEventSynchronize(nr.capEv) == cudaSuccess : cudaEventQuery(nr.capEv) == cudaSuccess))
        {
            capSent = true;
            nr.pushTok(0x7FFFFFFFu);
        }
        while (sent < nfr && (block ? cudaEventSynchronize(nr.slotEv[sent]) == cudaSuccess
                                    : cudaEventQuery(nr.slotEv[sent]) == cudaSuccess))
        {
            nr.pushTok(sent + 1);
            sent++;
        }
    };

    uint32_t twDone = 0;                  // tweens already computed
    uint32_t chunkBase = 0, chunkLen = 0; // current chunk's [base, base+len) in tween order
    bool fail = false;

    // one presented frame from a model-size planar source into its slot. Without effects a 1:1
    // slot takes the plain packer, and a fit that changes the size runs the Lanczos pair's
    // horizontal pass into dFitTmp, then its vertical pass fused with the slot store
    // (k_packOutV / k_packOutHdrV). With live effects: RTX VSR (the bridge, model size ->
    // presented size, 8-bit in and out, host-synchronous like every bridge call) or the Lanczos
    // pair into the planar staging frame, then the sharpen in the slot store. A VSR eval failure demotes
    // the rest of the run to Lanczos with one line.
    // (half != 0 on any of these = an fp16 source: a tween as the IFNet wrote it)
    auto packFrom = [&](const void* src, int ps, int rs, int sw, int sh, uint8_t* slot, const char* what,
                        int half = 0) -> bool {
        void* a[] = {(void*)&src, &half,          (void*)&ps, (void*)&rs, &sw,    &sh,
                     &slot,       (void*)&pitchI, &nr.x0,     &nr.y0,     &nr.dw, &nr.dh};
        if (cuLaunchKernel(nr.hdr ? nr.fPackOutHdr : nr.fPackOut, (nr.dw + 15) / 16, (nr.dh + 15) / 16, 1, 16, 16, 1, 0,
                           (CUstream)st, a, nullptr) != CUDA_SUCCESS)
        {
            nr.die(what);
            return false;
        }
        return true;
    };
    auto plainPack = [&](const void* src, uint8_t* slot, const char* what, int half = 0) -> bool {
        return packFrom(src, (int)plane, nr.pw, nr.w, nr.h, slot, what, half);
    };
    // a fit with no effect after it: the pair's horizontal pass into dFitTmp (3, sh, dw), then the
    // vertical pass fused with the slot store
    auto fitPack = [&](const void* src, int ps, int rs, int sw, int sh, uint8_t* slot, const char* what,
                       int half = 0) -> bool {
        const void* tabH = rsTabFor(nr, sw, nr.dw, st);
        const void* tabV = rsTabFor(nr, sh, nr.dh, st);
        if (!tabH || !tabV)
        {
            nr.die(what);
            return false;
        }
        void* ah[] = {(void*)&src, &half, &ps, &rs, &sw, &sh, &nr.dFitTmp, &nr.dw, (void*)&tabH};
        if (cuLaunchKernel(nr.fFitAaH, (nr.dw + 15) / 16, (sh + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, ah, nullptr) !=
            CUDA_SUCCESS)
        {
            nr.die(what);
            return false;
        }
        void* av[] = {&nr.dFitTmp, &nr.dw, &sh, &slot, (void*)&pitchI, &nr.x0, &nr.y0, &nr.dh, (void*)&tabV};
        if (cuLaunchKernel(nr.hdr ? nr.fPackOutHdrV : nr.fPackOutV, (nr.dw + 15) / 16, (nr.dh + 15) / 16, 1, 16, 16, 1,
                           0, (CUstream)st, av, nullptr) != CUDA_SUCCESS)
        {
            nr.die(what);
            return false;
        }
        return true;
    };
    // one resize of a planar source (ps / rs / sw x sh) into a planar target tw x th: the Lanczos pair through tmp
    // (3, sh, tw) when aa, else k_fitPlanar's 1:1 crop copy
    auto resizePlanar = [&](const void* src, int ps, int rs, int sw, int sh, bool aa, float* tmp, float* dst, int tw,
                            int th, int half = 0) -> bool {
        if (aa)
        {
            const void* tabH = rsTabFor(nr, sw, tw, st);
            const void* tabV = rsTabFor(nr, sh, th, st);
            if (!tabH || !tabV)
            {
                nr.die("fitAa tap tables failed");
                return false;
            }
            void* ah[] = {(void*)&src, &half, &ps, &rs, &sw, &sh, &tmp, &tw, (void*)&tabH};
            if (cuLaunchKernel(nr.fFitAaH, (tw + 15) / 16, (sh + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, ah,
                               nullptr) != CUDA_SUCCESS)
            {
                nr.die("fitAaH launch failed");
                return false;
            }
            void* av[] = {&tmp, &tw, &sh, &dst, &th, (void*)&tabV};
            if (cuLaunchKernel(nr.fFitAaV, (tw + 15) / 16, (th + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, av,
                               nullptr) != CUDA_SUCCESS)
            {
                nr.die("fitAaV launch failed");
                return false;
            }
            return true;
        }
        void* a[] = {(void*)&src, &half, &ps, &rs, &sw, &sh, &dst, &tw, &th};
        if (cuLaunchKernel(nr.fFitPlanar, (tw + 15) / 16, (th + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a, nullptr) !=
            CUDA_SUCCESS)
        {
            nr.die("fitPlanar launch failed");
            return false;
        }
        return true;
    };
    // one output slot's chain after the model (Restore, DLSS 5, Sharpen and RTX HDR already ran on the
    // captured frame): RTX VSR or the Lanczos resize, Upscale to
    auto storeSlot = [&](const void* src, uint8_t* slot, const char* what, int half = 0) -> bool {
        if (!src)
            return false; // a source that failed before (its line is logged)
        int ps = (int)plane, rs = nr.pw;
        const bool vsrNow =
            nr.vsr && !nr.vsrPre && !nr.vsrFailed && (g_rtxb.created || nr.fsrUp || nr.a4kUp || nr.acnUp);
        if (!vsrNow && !nr.fitAa && !nr.uw)
            return plainPack(src, slot, what, half);
        if (!vsrNow && nr.fitAa && !nr.uw)
            return fitPack(src, ps, rs, nr.w, nr.h, slot, what, half);
        // Upscale to: the first resize lands in the internal render frame (uw x uh)
        // instead of the staging frame, then the fit takes it to (dw, dh) below
        const int tw = nr.uw ? nr.uw : nr.dw, th = nr.uw ? nr.uh : nr.dh;
        float* stage = nr.uw ? nr.dUp : nr.dPres;
        bool haveVsr = false;
        if (vsrNow && (nr.a4kUp || nr.acnUp))
        {
            // Anime4K / ArtCNN: the model frame (HDR planes: its SDR view, as RTX VSR reads it) through its x2 net and
            // the Lanczos pair into the stage; HDR planes come back to the codes in RTX VSR's block below
            const void* a4kSrc = src;
            int aHalf = half, aps = ps, ars = rs;
            if (nr.hdr)
            {
                aps = nr.w * nr.h;
                ars = nr.w;
                if (!nativeHdrEnc(nr, src, half, ps, rs, nr.w, nr.h, nr.dVsrView, 0, aps, ars, 1, 1.0f, 0.0f))
                {
                    nr.die(nr.acnUp ? "ArtCNN SDR view launch failed" : "Anime4K SDR view launch failed");
                    return false;
                }
                a4kSrc = nr.dVsrView;
                aHalf = 0;
            }
            if (!(nr.acnUp ? nativeAcnRun(nr, a4kSrc, aHalf, aps, ars, stage, tw, th)
                           : nativeA4kRun(nr, a4kSrc, aHalf, aps, ars, stage, tw, th)))
                return false;
            haveVsr = true;
        }
        else if (vsrNow)
        {
            // HDR planes: RTX VSR takes SDR, so it reads the frame's SDR view; the way back to the codes is below
            // (HDR planes through AMD FSR upscaling: the view and the linear frame in ONE pass over the planes, below)
            const bool onePass = nr.hdr && nr.fuHdr;
            const void* vsrSrc = nr.hdr ? (const void*)nr.dVsrView : src;
            int vHalf = nr.hdr ? 0 : half, vps = nr.hdr ? nr.w * nr.h : ps, vrs = nr.hdr ? nr.w : rs;
            if (nr.hdr && !onePass &&
                !nativeHdrEnc(nr, src, half, ps, rs, nr.w, nr.h, nr.dVsrView, 0, vps, vrs, 1, 1.0f, 0.0f))
            {
                nr.die("VSR SDR view launch failed");
                return false;
            }
            void* a[] = {(void*)&vsrSrc, &vHalf, (void*)&vps, (void*)&vrs, &nr.w, &nr.h, &nr.dVsrIn};
            if (!onePass && cuLaunchKernel(nr.fsrUp ? nr.fPackBgraRgb : nr.fPackBgra, (nr.w + 15) / 16,
                                           (nr.h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a, nullptr) != CUDA_SUCCESS)
            {
                nr.die("packBgra launch failed");
                return false;
            }
            if (onePass)
            {
                // the SDR view packed for the vectors (k_hdrEnc + k_packBgraRgb's bytes) and the planes in linear
                // light for the bridge (k_fsrHdrIn's halves) from one read of the frame (k_fsrHdrPack); the way back
                // is k_fsrHdrOut below
                const float scale = 125.0f / nr.encWhite;
                int mode = 1, hlg = nr.encPost == 2 ? 1 : 0;
                void* f[] = {(void*)&src,  &half,         (void*)&ps, (void*)&rs, &nr.w,     &nr.h,        &mode,
                             &nr.encWhite, (void*)&scale, &hlg,       &nr.dVsrIn, &nr.dFuIn, &nr.fuInPitch};
                if (cuLaunchKernel(nr.fFsrHdrPack, (nr.w + 15) / 16, (nr.h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, f,
                                   nullptr) != CUDA_SUCCESS)
                {
                    nr.die("FSR HDR pack launch failed");
                    return false;
                }
            }
            else if (nr.fuHdr)
            {
                // AMD FSR upscaling on HDR planes: the frame in linear light into the bridge's buffer (the view above
                // feeds the vectors only); the way back is k_fsrHdrOut below
                const float scale = 125.0f / nr.encWhite;
                int hlg = nr.encPost == 2 ? 1 : 0;
                void* f[] = {(void*)&src, &half,         (void*)&ps, (void*)&rs, &nr.w,
                             &nr.h,       (void*)&scale, &hlg,       &nr.dFuIn,  &nr.fuInPitch};
                if (cuLaunchKernel(nr.fFsrHdrIn, (nr.w + 15) / 16, (nr.h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, f,
                                   nullptr) != CUDA_SUCCESS)
                {
                    nr.die("FSR HDR input launch failed");
                    return false;
                }
            }
            const RtxRect ri{0, 0, (uint32_t)nr.w, (uint32_t)nr.h};
            const RtxRect ro{0, 0, (uint32_t)tw, (uint32_t)th};
            unsigned int rv = 0;
            if (!nativeVsrEval(nr, ri, ro, st, rv))
                return false;
            if (rv == 1u)
                haveVsr = true;
            else
            {
                LOG("native: RTX VSR eval failed (rc %u), Lanczos for the rest of the run\n", rv);
                nr.vsrFailed = true;
                if (!nr.uw) // VSR = enlarging fit
                    return fitPack(src, ps, rs, nr.w, nr.h, slot, what, half);
            }
        }
        if (haveVsr && !nr.uw && !nr.hdr && !nr.a4kUp && !nr.acnUp)
        {
            // the bridge output is the presented frame: into the content rect, RTX VSR's R, G, B, A as the slot's
            // B, G, R, A in one pass; AMD FSR upscaling's buffer already holds the slot's order
            uint8_t* to = slot + (size_t)nr.y0 * nr.pitch + (size_t)nr.x0 * 4;
            int srcPitch = nr.dw * 4;
            void* a[] = {&nr.dVsrOut, &srcPitch, &nr.dw, &nr.dh, &to, (void*)&pitchI};
            const bool ok = nr.fsrUp ? cudaMemcpy2DAsync(to, nr.pitch, nr.dVsrOut, (size_t)nr.dw * 4, (size_t)nr.dw * 4,
                                                         nr.dh, cudaMemcpyDeviceToDevice, st) == cudaSuccess
                                     : cuLaunchKernel(nr.fRgbaBgra, (nr.dw + 15) / 16, (nr.dh + 15) / 16, 1, 16, 16, 1,
                                                      0, (CUstream)st, a, nullptr) == CUDA_SUCCESS;
            if (!ok)
            {
                nr.die("VSR slot copy failed");
                return false;
            }
            return true;
        }
        // the first resize target (the internal render frame, or the staging frame at the
        // presented size): the VSR output unpacked, or the Lanczos fit
        if (haveVsr && nr.fuHdr)
        {
            // AMD FSR upscaling on HDR planes: the bridge's linear-light result back to the codes (k_fsrHdrOut), no
            // SDR view to come back from
            const float inv = nr.encWhite / 125.0f;
            int hlg = nr.encPost == 2 ? 1 : 0;
            void* a[] = {&nr.dFuOut, &nr.fuOutPitch, (void*)&tw, (void*)&th, (void*)&inv, &hlg, &stage};
            if (cuLaunchKernel(nr.fFsrHdrOut, (tw + 15) / 16, (th + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a,
                               nullptr) != CUDA_SUCCESS)
            {
                nr.die("FSR HDR output launch failed");
                return false;
            }
        }
        else if (haveVsr)
        {
            void* a[] = {&nr.dVsrOut, (void*)&tw, (void*)&th, &stage};
            if (!nr.a4kUp && !nr.acnUp && // Anime4K / ArtCNN wrote the stage itself
                cuLaunchKernel(nr.fsrUp ? nr.fUnpackBgraRgb : nr.fUnpackRgba, (tw + 15) / 16, (th + 15) / 16, 1, 16, 16,
                               1, 0, (CUstream)st, a, nullptr) != CUDA_SUCCESS)
            {
                nr.die("unpackBgra launch failed");
                return false;
            }
            // HDR planes: back to the codes. The frames this one came from all inside the SDR range take the view's way
            // back; any other gets the light the view never held from the frame's Lanczos fit (k_hdrRestOut). A DRBA
            // frame is 3 captures old, so the last 4 count
            const int fps = tw * th;
            if (nr.hdr && ((nr.hdrInBits & 15u) == 15u
                               ? !nativeHdrFromSdr(nr, stage, 0, fps, tw, tw, th, stage, fps, tw, 1)
                               : !resizePlanar(src, ps, rs, nr.w, nr.h, true, nr.uw ? nr.dUpTmp : nr.dFitTmp,
                                               nr.dVsrRem, tw, th, half) ||
                                     !nativeHdrRestOut(nr, stage, tw, th, nr.dVsrRem, 0, fps, tw, tw, th, 1)))
            {
                nr.die("VSR HDR codes launch failed");
                return false;
            }
        }
        else if (!resizePlanar(src, ps, rs, nr.w, nr.h, nr.uw ? nr.upAa : nr.fitAa, nr.uw ? nr.dUpTmp : nr.dFitTmp,
                               stage, tw, th, half))
            return false;
        // the second resize, internal render frame -> the fit rect: the pair with the fused store (the
        // plain packer's 1:1 copy when the size stays)
        if (nr.uw)
            return nr.fitAa ? fitPack(nr.dUp, nr.uw * nr.uh, nr.uw, nr.uw, nr.uh, slot, what)
                            : packFrom(nr.dUp, nr.uw * nr.uh, nr.uw, nr.uw, nr.uh, slot, what);
        // the staging frame is the presented frame, stored 1:1 (sampleOut reads it as a dw x dh source with no pad)
        return packFrom(nr.dPres, nr.dw * nr.dh, nr.dw, nr.dw, nr.dh, slot, what);
    };
    uint8_t* heldSlot = nullptr; // a held pair's first real slot, copied into its other slots
    for (uint32_t i = 0; i < nfr && !fail; i++)
    {
        uint8_t* slot = nr.dOutRing + ((size_t)set * nr.slots + i) * nr.slotBytes;
        const bool real = !(fr[i] < 0.999f) || !nr.havePrev || pairStatic;
        if (real)
        {
            if (pairStatic && heldSlot)
            {
                // the held pair's remaining slots: the real frame's finished slot, copied as
                // it is, so the per-frame passes (VSR, the fit, the sharpen) run ONCE
                if (cudaMemcpyAsync(slot, heldSlot, nr.slotBytes, cudaMemcpyDeviceToDevice, st) != cudaSuccess)
                {
                    nr.die("held slot copy failed");
                    fail = true;
                    break;
                }
            }
            else if (!storeSlot(nr.drba ? nativeDrbaReal(nr, drLag)
                                : pipe  ? nativeXPrev(nr)
                                        : dCur,
                                slot, "packOut (real) launch failed"))
            {
                fail = true;
                break;
            }
            else
                heldSlot = slot;
        }
        else if (nr.drba)
        {
            // native DRBA: one IFNet enqueue per tween with its own DRM timestep map, or the
            // lagged real frame where the group holds (a head without enough history)
            bool held = false;
            if (!nativeDrbaTween(nr, fr[i], drHist, held, false, nr.blendOut ? slot : nullptr))
            {
                fail = true;
                break;
            }
            if ((held || !nr.blendOut) &&
                !storeSlot(held ? (const void*)nativeDrbaReal(nr, drLag) : mergedAt(nr, 0, plane), slot,
                           "packOut (drba) launch failed", held ? 0 : (int)nr.outHalf))
            {
                fail = true;
                break;
            }
            if (!held)
                twDone++;
        }
        else if (nr.gmfss)
        {
            // GMFSS: one tween at a time (no batch axis anywhere in its four engines), the
            // clamped fusionnet output then rides the shared effects and slot chain
            if (!nativeGmfssTween(nr, fr[i]))
            {
                fail = true;
                break;
            }
            if (!storeSlot(nr.dGmF, slot, "packOut (gmfss) launch failed"))
            {
                fail = true;
                break;
            }
            twDone++;
        }
        else if (nr.nvof)
        {
            // the nvof model: one splat pair per tween into the model layout, then the shared
            // effects and slot chain like every other model's output
            if (!nativeNvofTween(nr, fr[i]))
            {
                fail = true;
                break;
            }
            if (!storeSlot(nr.dNvOut, slot, "packOut (nvof) launch failed"))
            {
                fail = true;
                break;
            }
            twDone++;
        }
        else if (nr.fruc)
        {
            // Smooth Motion: one bridge call per tween (it syncs the context on entry, so the
            // previous tween's store has read dFrOut before the next unpack overwrites it)
            if (!(pipe ? (nr.fsrfg ? nativeFsrfgLagTween(nr, fr[i]) : nativeFrucLagTween(nr, fr[i]))
                       : nativeFrucTween(nr, fr[i])))
            {
                fail = true;
                break;
            }
            if (!storeSlot(nr.dFrOut, slot, "packOut (fruc) launch failed"))
            {
                fail = true;
                break;
            }
            twDone++;
        }
        else
        {
            if (twDone >= chunkBase + chunkLen)
            {
                // next chunk of tweens, up to the engine's batch (1: the unbatched class)
                chunkBase = twDone;
                chunkLen = nTween - chunkBase;
                if (chunkLen > (uint32_t)nr.batchMax)
                    chunkLen = (uint32_t)nr.batchMax;
                // timestep planes are constant maps of t
                // collect the chunk's t values in tween order
                float ts[64];
                if (chunkLen > 64)
                    chunkLen = 64;
                uint32_t k = 0, seen = 0;
                for (uint32_t j = 0; j < nfr; j++)
                {
                    if (!(fr[j] < 0.999f))
                        continue;
                    if (seen >= chunkBase && k < chunkLen)
                        ts[k++] = fr[j];
                    seen++;
                }
                for (uint32_t j = 0; j < chunkLen; j++)
                {
                    if (!nativeFillT(nr, j, ts[j], mplane, st))
                    {
                        nr.die("timestep fill failed");
                        fail = true;
                        break;
                    }
                }
                if (fail)
                    break;
                // the IFNet's frame: the motion frame (two domains), else the pictures' pad
                const int iph = nr.mph ? nr.mph : nr.ph, ipw = nr.mph ? nr.mpw : nr.pw;
                if (!nativeIfnRun(nr, nr.mph ? (const void*)nr.dM : (const void*)nr.dXh, nr.dT, nr.dF[nr.fCur ^ 1],
                                  nr.dF[nr.fCur], ipw, iph, nr.dFlow, nr.dMask, st))
                {
                    nr.die("IFNet launch failed");
                    fail = true;
                    break;
                }
                // the chunk's tweens: RIFE's last step on the frames this enqueue read, or on the fp32
                // pictures at the working size (two domains); blendOut makes each one in its slot below
                const bool picHalf = !nr.mph && nr.xHalf;
                if (!nr.blendOut && !nativeRifeBlend(nr, picHalf ? (const void*)nr.dXh : (const void*)nr.dX,
                                                     (int)picHalf, (int)chunkLen, st, !picHalf && nr.xSwap))
                {
                    nr.die("rifeBlend launch failed");
                    fail = true;
                    break;
                }
            }
            const uint32_t off = twDone - chunkBase;
            if (nr.blendOut)
            {
                const bool picHalf = !nr.mph && nr.xHalf;
                if (!nativeRifeBlendOut(nr, picHalf ? (const void*)nr.dXh : (const void*)nr.dX, (int)picHalf, (int)off,
                                        slot, st, !picHalf && nr.xSwap))
                {
                    nr.die("rifeBlendOut launch failed");
                    fail = true;
                    break;
                }
            }
            else if (!storeSlot(mergedAt(nr, off, plane), slot, "packOut launch failed", (int)nr.outHalf))
            {
                fail = true;
                break;
            }
            twDone++;
        }
        cudaEventRecord(nr.slotEv[i], st);
        drainReady(false);
    }
    // a pipelined group's last step: Smooth Motion's worker joined, AMD FSR 4's level-1 call issued (when no tween did)
    if (fail || (pipe && !(nr.fsrfg ? nativeFsrfgL1Issue(nr) : nativeFrucSpecJoin(nr))))
        return false;
    if (!pipe)
        nr.grFrChainMs = nr.grFrCallMs - nr.grFrPreMs;
    // the first group has no pair and a group with no slot stores nothing: their times say nothing about either kind
    if (nr.grEv[1] && nr.havePrev && nfr)
    {
        cudaEventRecord(nr.grEv[1], st);
        nr.grArmed = true;
        nr.grPiped = pipe;
        nr.grTween = twDone;
        nr.grSlots = nfr;
    }
    drainReady(true); // sync the stragglers, exactly like _finish
    if (nr.gmProf && nr.gmProfTween)
        nativeGmfssProfile(nr);
    nr.pushTok(0x80000000u); // bare end marker closes the group
    nr.havePrev = true;
    nativeVideoMemoryNote(); // once a session, after its first group
    return true;
}
// ---- part 4: the compute thread and the public entry point --------------------------------

// the stdin `set` lines that arrived since the last frame group (g_liveSets), on the native thread that reads these
// values: the Sharpen strength (0 = off, on = its buffers first), RTX VSR's quality while RTX VSR is the upscaler, the RTX HDR
// sliders while RTX HDR runs, DLSS 5's Structure / Tone and its passes (more are built in place, memory permitting) while
// DLSS 5 runs (held for its start while it does not), DLSS 5 itself on / off (nativeNrSwitch), the HUD (the loop
// thread's), the upscaler / Restore / RTX HDR kind this session runs off and on again (their fallbacks); each read
// again at its next launch / evaluate; the HUD's latency segment (the loop thread's next paint). Any other key, or one
// whose effect this session does not run, waits for the next session's arguments (the app passes the new value there)
// Sharpen turned on in a running session: its tight result (and the HDR planes' SDR view) at the model frame's size,
// when the video memory room holds them; false = the next session takes it (its own fit gives the picked effects room)
static bool nativeSharpenOn(NativeRife& nr)
{
    const bool view = nr.encPost && nr.shHdrView;
    const size_t one = (size_t)3 * nr.w * nr.h * sizeof(float);
    const uint64_t need = (nr.dShOut ? 0 : one) + (view && !nr.dShIn ? one : 0);
    const uint64_t room = nativeVideoMemoryRoom();
    if (need && room && room < need)
        return false;
    if (!nr.dShOut && cudaMalloc((void**)&nr.dShOut, one) != cudaSuccess)
    {
        nr.dShOut = nullptr;
        return false;
    }
    if (view && !nr.dShIn && cudaMalloc((void**)&nr.dShIn, one) != cudaSuccess)
    {
        nr.dShIn = nullptr;
        return false;
    }
    return true;
}

// DLSS 5 switched in a running session: off = the pass skips (the core and its memory stay for the session); on again
// = it resumes from a Reset (the history stopped with it); on in a session that started without it = the core brought
// up as the session start does (nativeLiveNrInit, at the held Structure / Tone / passes / Style), when the video memory
// room holds its working memory. false = this session cannot (why says it); the next session takes it
static bool nativeNrSwitch(NativeRife& nr, bool on, std::string& why)
{
    if (!on)
    {
        nr.nrPaused = nr.liveNr;
        return true;
    }
    if (nr.liveNr && nr.nrHost && !nr.nrFailed)
    {
        if (nr.nrPaused)
        {
            nr.nrPaused = false;
            nr.nrFirst = true;
            nr.nrHaveLast = false;
        }
        return true;
    }
    if (nr.nrFailed)
    {
        why = "DLSS 5 stopped earlier in this session";
        return false;
    }
    if (g_backend != BK_SERVER)
    {
        why = "this route runs no native pass";
        return false;
    }
    const int p = g_nrPasses < 1 ? 1 : (g_nrPasses > nr::kMaxPasses ? nr::kMaxPasses : g_nrPasses);
    const uint64_t need = (uint64_t)((kNrLiveLateBase + nr::kFrameMp * nr.w * nr.h / 1e6 * p) * 1048576.0);
    const uint64_t room = nativeVideoMemoryRoom();
    if (room && room < need)
    {
        why = "not enough free video memory";
        return false;
    }
    g_liveNrCuda = g_nrNative = g_nrAttempted = true;
    if (!nativeLiveNrInit(nr))
    {
        why = "an allocation failed";
        return false;
    }
    if (!nr.liveNr)
    {
        why = "the core did not start (the line above)";
        return false;
    }
    return true;
}

// the running upscaler's own state released between two groups before another kind starts in place (nativeUpSwitch),
// the GPU idle first (the slot store's streams read its buffers): AMD FSR / Intel XeSS free their instance, imports and
// vector session (the DLL stays loaded; a bridge whose start failed too), Anime4K its resize buffers (the statistics
// and Restore's stay), ArtCNN its buffers and weights; RTX VSR keeps NGX (one create a process, its VSR feature reused
// by a switch back); the buffers every kind shares at this geometry (dVsrIn / dVsrOut, dPres, dVsrView / dVsrRem) stay
static void nativeUpRelease(NativeRife& nr)
{
    const char* name = !nr.vsr    ? nullptr
                       : nr.fsrUp ? (nr.fuKind ? "Intel XeSS upscaling" : "AMD FSR upscaling")
                       : nr.a4kUp ? "Anime4K"
                       : nr.acnUp ? "ArtCNN"
                                  : "RTX VSR";
    cudaDeviceSynchronize();
    if (nr.fuMade)
        nativeFsrupRelease(nr);
    if (nr.a4kUp)
        nativeA4kUpFree(nr);
    if (nr.acnUp)
        nativeAcnRelease(nr);
    nr.vsr = nr.vsrWant = nr.fsrUp = nr.fuHdr = nr.a4kUp = nr.acnUp = nr.vsrFailed = nr.vsrUserOff = false;
    if (name)
        LOG("native: live upscale: %s released in place\n", name);
}

// an upscaler ticked in a running session that started without one, or another kind than the one it runs (stdin `set
// up N`, N = the kind the renderer picked: 1 RTX VSR, 2 AMD FSR 4, 3 Anime4K, 4 ArtCNN, 5 Intel XeSS): the live case
// where the fit after the model enlarges (the upscaler's resize under nativeRtxInit's rule), on SDR or HDR planes, set
// up on the native thread between two groups the way the session start does (the DLL checks and HDR levers, the
// buffers the slot store reads, the bridge or module, the warm-up; FSR / XeSS's first frames on a worker), when the
// video memory room holds it; a running kind is released first (nativeUpRelease), after the checks that leave it
// running. false = this session cannot (why says it); the next session takes it. Left to the next session, as at the
// start: a resize that does not enlarge, the resize before the model (vsrPre), RTX VSR once NGX was created without it
// (single-instance a process: TrueHDR holds it), another route; a setup that fails after the release = Lanczos
static bool nativeUpSwitch(NativeRife& nr, int want, std::string& why)
{
    if (g_offline || g_backend != BK_SERVER)
    {
        why = "this route runs no native pass";
        return false;
    }
    if (nr.hdr)
    {
        // HDR planes as the session start runs them (nativeConfigHdr's levers): RTX VSR, Anime4K and ArtCNN on the
        // frame's SDR view and back to the codes, AMD FSR / Intel XeSS in the bridge's HDR mode
        wchar_t ev[8]{};
        const bool vsrOff = want == 1 && GetEnvironmentVariableW(L"SMV_VSR_HDR", ev, 8) && ev[0] == L'0';
        const bool fuOff =
            (want == 2 || want == 5) && GetEnvironmentVariableW(L"SMV_FSRUP_HDR", ev, 8) && ev[0] == L'0';
        if (vsrOff || fuOff)
        {
            why = vsrOff ? "HDR planes (SMV_VSR_HDR=0)" : "HDR planes (SMV_FSRUP_HDR=0)";
            return false;
        }
    }
    const int tw = nr.uw ? nr.uw : nr.dw, th = nr.uw ? nr.uh : nr.dh;
    if (!(tw > nr.w && th > nr.h))
    {
        why = nr.w > nr.cw && nr.h > nr.ch ? "the resize before the model" : "this resize does not enlarge";
        return false;
    }
    const char* name = want == 1   ? "RTX VSR"
                       : want == 2 ? "AMD FSR upscaling"
                       : want == 3 ? "Anime4K"
                       : want == 4 ? "ArtCNN"
                       : want == 5 ? "Intel XeSS upscaling"
                                   : nullptr;
    if (!name)
    {
        why = "an unknown upscaler";
        return false;
    }
    std::string dir;
    if (want == 1)
    {
        if (g_rtxb.created && !g_rtxb.vsrMade)
        {
            why = "RTX Video is already created in this process without RTX VSR (RTX HDR holds it)";
            return false;
        }
        std::wstring d;
        wchar_t ov[MAX_PATH]{};
        if (GetEnvironmentVariableW(L"SMV_RTXVIDEO_DIR", ov, MAX_PATH) && ov[0])
            d = ov;
        else if (!g_nativeEngineDir.empty())
            d = g_nativeEngineDir + L"\\rtxvideo";
        if (d.empty() || GetFileAttributesW((d + L"\\rtxvideo_cuda.dll").c_str()) == INVALID_FILE_ATTRIBUTES ||
            GetFileAttributesW((d + L"\\nvngx_vsr.dll").c_str()) == INVALID_FILE_ATTRIBUTES || !rtxBridgeLoad())
        {
            why = "no rtxvideo bridge or nvngx_vsr.dll";
            return false;
        }
    }
    else if (want == 2 || want == 5)
    {
        dir = lkEnv(want == 2 ? "SMV_FSRUP_DIR" : "SMV_XESS_DIR");
        if (dir.empty() && !g_nativeEngineDir.empty())
            dir = wideToUtf8(g_nativeEngineDir) + (want == 2 ? "\\fsrup" : "\\xess");
        bool have = !dir.empty();
        if (want == 2)
            for (const char* f :
                 {"smv_fsrup_bridge.dll", "amd_fidelityfx_loader_dx12.dll", "amd_fidelityfx_upscaler_dx12.dll"})
                have = have && lkFile(dir + "\\" + f);
        else
            for (const char* f : {"smv_xess_bridge.dll", "libxess.dll"})
                have = have && lkFile(dir + "\\" + f);
        if (!have)
        {
            why = want == 2 ? "AMD's FidelityFX DLLs are missing" : "Intel's XeSS DLLs are missing";
            return false;
        }
    }
    if (nr.vsr || nr.fuMade)
        nativeUpRelease(nr); // the running kind (or a bridge whose start failed) first: its memory counts below
    // the buffers the slot store reads at this geometry (nativeRtxInit's): Anime4K / ArtCNN write planar floats into
    // the staging frame (under Upscale to the internal render frame, allocated), the bridges read and write BGRA
    // buffers; HDR planes add the staging frame for every kind (the way back to the codes lands there), the frame's SDR
    // view and the light the view never held at the target (dVsrView / dVsrRem)
    const bool planar = want == 3 || want == 4;
    const bool pres = (planar && !nr.uw) || nr.hdr;
    const uint64_t fl = sizeof(float);
    const uint64_t need = (pres && !nr.dPres ? 3 * fl * nr.dw * nr.dh : 0) +
                          (planar ? 0 : (uint64_t)nr.w * nr.h * 4 + (uint64_t)tw * th * 4) +
                          (nr.hdr && !nr.dVsrView ? 3 * fl * nr.w * nr.h : 0) +
                          (nr.hdr && !nr.dVsrRem ? 3 * fl * tw * th : 0);
    const uint64_t room = nativeVideoMemoryRoom();
    if (room && room < need)
    {
        why = "not enough free video memory";
        return false;
    }
    auto alloc = [](void** p, size_t bytes) {
        if (*p)
            return true;
        if (cudaMalloc(p, bytes) == cudaSuccess)
            return true;
        *p = nullptr;
        cudaGetLastError();
        return false;
    };
    if ((pres && !alloc((void**)&nr.dPres, (size_t)3 * nr.dw * nr.dh * sizeof(float))) ||
        (!planar &&
         (!alloc((void**)&nr.dVsrIn, (size_t)nr.w * nr.h * 4) || !alloc((void**)&nr.dVsrOut, (size_t)tw * th * 4))) ||
        (nr.hdr && (!alloc((void**)&nr.dVsrView, (size_t)3 * nr.w * nr.h * sizeof(float)) ||
                    !alloc((void**)&nr.dVsrRem, (size_t)3 * tw * th * sizeof(float)))))
    {
        why = "an allocation failed";
        return false;
    }
    nr.vsrWant = true;
    nr.vsrPre = false;
    nr.fsrUp = want == 2 || want == 5;
    nr.fuHdr = nr.fsrUp && nr.hdr; // the start's rule (nativeRtxInit) with no resize before the model
    nr.fuKind = want == 5 ? 1 : 0;
    if (nr.fsrUp)
        nr.fsrUpDir = dir;
    nr.a4kUp = want == 3;
    nr.acnUp = want == 4;
    nr.vsrFailed = nr.vsrUserOff = false;
    bool ok = true;
    if (want == 1 && g_rtxb.vsrMade)
        LOG("native: live upscale: RTX VSR runs the feature of this process's create (warmed then)\n");
    else if (want == 1)
    {
        if (g_rtxb.create(nullptr, nullptr, 0, 0u, 1u) != 1u)
        {
            why = "rtx_video_api_cuda_create failed";
            ok = false;
        }
        else
        {
            g_rtxb.created = g_rtxb.vsrMade = true;
            g_rtxUsed = true; // resident host: NGX is single-instance per process, end it after this session
            // the warm-up eval on an opaque black frame, as the session start runs it
            const RtxRect ri{0, 0, (uint32_t)nr.w, (uint32_t)nr.h};
            const RtxRect ro{0, 0, (uint32_t)tw, (uint32_t)th};
            if (cuMemsetD32Async((CUdeviceptr)nr.dVsrIn, 0xFF000000u, (size_t)nr.w * nr.h, (CUstream)nr.stream) !=
                    CUDA_SUCCESS ||
                cudaStreamSynchronize(nr.stream) != cudaSuccess ||
                g_rtxb.evalVsr(nr.dVsrIn, nr.dVsrOut, ri, ro, &nr.vsrSet) != 1u ||
                cudaDeviceSynchronize() != cudaSuccess)
            {
                why = "the RTX VSR warm-up eval failed";
                ok = false;
            }
        }
    }
    else if (nr.fsrUp)
        ok = nativeFsrupSetup(nr, nr.w, nr.h, tw, th);
    else
        ok = (nr.a4kUp ? nativeA4kSetup(nr, nr.w, nr.h, tw, th, 0) : nativeAcnSetup(nr, nr.w, nr.h, tw, th, 0)) &&
             !nr.vsrFailed;
    if (!ok)
    {
        // Lanczos stays: the flags off (the kind's buffers go with the session's teardown)
        nr.vsrWant = nr.fsrUp = nr.a4kUp = nr.acnUp = nr.vsrFailed = false;
        if (!nr.fuMade) // a bridge built before the failure is destroyed at the teardown by its kind
            nr.fuKind = 0;
        if (why.empty())
            why = "its setup failed (the line above)";
        return false;
    }
    if (nr.fsrUp)
    {
        // the first frames on a worker (nativeFsrupWarmProc); the session presents without the upscaler meanwhile,
        // nativeUpWarmEnd turns it on
        nr.vsrWant = nr.fsrUp = false;
        nr.fuWarmWant = true;
        nr.fuWarmT0 = GetTickCount64();
        InterlockedExchange(&nr.fuWarm, 1);
        nr.fuWarmTh = CreateThread(nullptr, 0, nativeFsrupWarmProc, &nr, 0, nullptr);
        if (!nr.fuWarmTh)
        {
            InterlockedExchange(&nr.fuWarm, 0);
            why = "its warm-up thread did not start";
            return false;
        }
        LOG("native: live upscale: %s %dx%d -> %dx%d, its first frames on a worker (the driver may compile its "
            "shaders), the picture goes on without it\n",
            name, nr.w, nr.h, tw, th);
        return true;
    }
    nr.vsr = true;
    LOG("native: live upscale: %s %dx%d -> %dx%d, started in place\n", name, nr.w, nr.h, tw, th);
    return true;
}

// Restore ticked in a running session that started without it (stdin `set restore N`: 1 Real-ESRGAN, 2 AnimeJaNai =
// the file the `restoremodel` key holds, 3 Anime4K's Restore CNN UL), set up on the native thread between two groups
// the way the session start does: the capture as Restore's source (restPre, srcW / srcH = the capture; every reader
// of those is Restore's own), the fold to the model size, the buffers the handoff allocates (dRestTmp, dRest, dCapF,
// dRestRem on HDR planes at a size change), then restNatSetup (weights, activations, warm-up), when the video memory
// room holds them. Left to the next session: another route, an upscaler before the model (it moves Restore's fold
// target), a start in place that failed earlier (its buffers stay for the teardown). false = why says it
static bool nativeRestoreSwitch(NativeRife& nr, int want, std::string& why)
{
    if (g_offline || g_backend != BK_SERVER)
    {
        why = "this route runs no native pass";
        return false;
    }
    if (want < 1 || want > 3)
    {
        why = "an unknown Restore kind";
        return false;
    }
    if (nr.vsrPre)
    {
        why = "an upscaler runs before the model";
        return false;
    }
    if (nr.dRest)
    {
        why = "its start in place failed earlier in this session";
        return false;
    }
    // the kind's model: Real-ESRGAN's weights readable, the held AnimeJaNai file in its form, the Anime4K module
    if (want == 1)
    {
        std::vector<uint8_t> blob;
        std::vector<size_t> off;
        if (!restNatLoad(blob, off, why))
            return false;
    }
    else if (want == 2 && g_restoreModel.empty())
    {
        why = "no AnimeJaNai file named (restoremodel)";
        return false;
    }
    else if (want == 2 && !ajNatWanted())
    {
        why = "the AnimeJaNai file is not in its SPAN-F3 form (the line above)";
        return false;
    }
    else if (want == 3 && !nativeA4kModule(nr, why))
        return false;
    const int cw = nr.cw, ch = nr.ch;
    const uint64_t fl = sizeof(float), cp = (uint64_t)cw * ch, mp = (uint64_t)nr.w * nr.h;
    const bool rem = (nr.encPre || nr.encPost) && (nr.w != cw || nr.h != ch);
    // the kind's buffers at the capture size (restNatSetup / nativeA4kRestSetup reads srcW / srcH): the input planes,
    // Real-ESRGAN / AnimeJaNai's sub-pixel output and activations (none where the shared activation block holds
    // them), Anime4K's layer slots, output and statistics; the weights (a few MB) sit inside the room's reserve
    nr.restNatKind = want;
    nr.restNat = true;
    nr.sw = cw;
    nr.sh = ch;
    auto undo = [&nr]() {
        nr.restNatKind = 0;
        nr.restNat = false;
        nr.sw = nr.sh = 0;
    };
    uint64_t kindB = nr.dRestIn ? 0 : 3 * cp * sizeof(uint16_t);
    if (want == 3)
        kindB += 4 * 3 * cp * 8 + 3 * cp * fl + (nr.a4kStatCap >= cp ? 0 : 2 * cp * sizeof(uint16_t));
    else
    {
        const uint64_t sub = want == 2 ? (uint64_t)((cw + 1) / 2) * ((ch + 1) / 2) : cp;
        const uint64_t pp = restNatBytes(nr);
        kindB += 48 * sub * sizeof(uint16_t) + (nr.dGmCtxMem && nr.gmCtxMemB >= (int64_t)pp ? 0 : pp);
    }
    const uint64_t need = 3 * 4 * (uint64_t)ch * nr.w * fl + 3 * mp * fl + (nr.dCapF ? 0 : 3 * cp * fl) +
                          (rem && !nr.dRestRem ? 3 * mp * fl : 0) + kindB;
    const uint64_t room = nativeVideoMemoryRoom();
    if (room && room < need)
    {
        undo();
        why = "not enough free video memory";
        return false;
    }
    auto alloc = [](void** p, size_t bytes) {
        if (*p)
            return true;
        if (cudaMalloc(p, bytes) == cudaSuccess)
            return true;
        *p = nullptr;
        cudaGetLastError();
        return false;
    };
    if (!alloc((void**)&nr.dRestTmp, (size_t)3 * 4 * ch * nr.w * sizeof(float)) ||
        !alloc((void**)&nr.dRest, (size_t)3 * mp * sizeof(float)) ||
        !alloc((void**)&nr.dCapF, (size_t)3 * cp * sizeof(float)) ||
        (rem && !alloc((void**)&nr.dRestRem, (size_t)3 * mp * sizeof(float))))
    {
        undo();
        why = "an allocation failed";
        return false;
    }
    nr.restTw = nr.w;
    nr.restTh = nr.h;
    nr.restFailed = nr.restUserOff = false;
    // the frames in flight first: Anime4K's statistics buffers, shared with its upscaler, may grow to the capture size
    cudaStreamSynchronize(nr.stream);
    if (!restNatSetup(nr))
    {
        cudaGetLastError();
        undo();
        why = "its setup failed (the line above)";
        return false;
    }
    nr.restore = nr.restPre = true;
    const std::string name = want == 1   ? std::string("Real-ESRGAN animevideov3")
                             : want == 3 ? std::string("Anime4K Restore CNN UL")
                                         : g_restoreModel.substr(g_restoreModel.find_last_of("\\/") + 1);
    LOG("native: live restore: %s at %dx%d -> %dx%d, started in place on the captured frame before the model\n",
        name.c_str(), cw, ch, nr.w, nr.h);
    return true;
}

// Anime4K's Mode A+A in a running session (stdin `set a4kaa 0|1`): off = the per-pass flag (the buffers stay for the
// session), on = Anime4K's resize buffers freed and set up again at the session's geometry with g_a4kAA set, the way
// the session start runs it (the room first: short = Mode A, nativeA4kSetup's line); Anime4K not set up in this
// session = held for its start in place (nativeUpSwitch's setup reads g_a4kAA). An Anime4K the user unticked stays
// off; a setup that fails = Lanczos for the session (vsrFailed, its buffers freed). The `live set` line's note, "" =
// applied
static std::string nativeA4kAaSwitch(NativeRife& nr, bool on)
{
    g_a4kAA = on;
    if (!nr.a4kUp || !nr.a4kW)
        return " (held for Anime4K's start)";
    if (!on || nr.a4kAA)
    {
        nr.a4kAA = on;
        return "";
    }
    if (nr.vsrFailed && !nr.vsrUserOff)
        return " (the next session takes it: Anime4K stopped earlier in this session)";
    const int w = nr.a4kW, h = nr.a4kH, ow = nr.a4kOw, oh = nr.a4kOh;
    const bool userOff = nr.vsrUserOff;
    cudaDeviceSynchronize();
    nativeA4kUpFree(nr);
    nr.vsrFailed = false;
    if (!nativeA4kSetup(nr, w, h, ow, oh, 0) || nr.vsrFailed)
    {
        cudaDeviceSynchronize();
        nativeA4kUpFree(nr);
        nr.vsrFailed = true;
        nr.vsrUserOff = false;
        return " (Anime4K stopped: its setup failed, the line above; Lanczos for the session)";
    }
    nr.vsrFailed = userOff;
    return nr.a4kAA ? "" : " (the next session takes it: not enough free video memory, Mode A runs)";
}

// the end of a DLSS 5 pass growth: the lines a growth in one go logs (the passes built, then the set), the run count
// set once to the passes asked (as many as were built); cut = why it ended before its last pass, nullptr = it ran out
static void nativeNrGrowEnd(NativeRife& nr, const char* cut)
{
    const int now = nr.nrHost ? nr.nrHost->passes() : nr.nrGrowFrom;
    if (now > nr.nrGrowFrom)
        LOG("live DLSS 5: %d more passes built in %.0f ms (%d now)%s%s\n", now - nr.nrGrowFrom, nr.nrGrowMs, now,
            nr.nrGrowWhy.empty() ? "" : ": ", nr.nrGrowWhy.c_str());
    std::string why = cut ? std::string(cut) : nr.nrGrowWhy;
    if (why.empty() && now < nr.nrGrowWant)
        why = "no video memory for more";
    const int run = nr.nrHost ? nr.nrHost->runPasses(nr.nrGrowWant) : 0;
    if (run < nr.nrGrowWant)
        LOG("live set: nrpasses %s (%d run: %s)\n", nr.nrGrowVal.c_str(), run, why.c_str());
    else
        LOG("live set: nrpasses %s\n", nr.nrGrowVal.c_str());
    nr.nrGrowWant = 0;
    nr.nrGrowVal.clear();
    nr.nrGrowWhy.clear();
}

// one step of a DLSS 5 pass growth, on the native thread before a frame group: ONE more pass (its feature and frame,
// nr::Host::growPasses) when the video memory room holds its working memory (nr::kFrameMp a megapixel), so the picture
// moves between the builds (a pass's build holds it ~100 ms at 640x360, 17 in one go ~1.9 s); the run count stays
// until the end (nativeNrGrowEnd), so the picture changes once, as a growth in one go does
static void nativeNrGrowStep(NativeRife& nr)
{
    if (!nr.nrGrowWant)
        return;
    if (!nr.liveNr || !nr.nrHost || nr.nrFailed)
    {
        nativeNrGrowEnd(nr, "DLSS 5 stopped");
        return;
    }
    const int had = nr.nrHost->passes();
    if (had < nr.nrGrowWant)
    {
        const uint64_t per = (uint64_t)(nr::kFrameMp * nr.nrW * nr.nrH / 1e6 * 1048576.0);
        const uint64_t room = nativeVideoMemoryRoom();
        if (!room || !per || room >= per)
        {
            const auto t0 = std::chrono::steady_clock::now();
            const int now = nr.nrHost->growPasses(had + 1, nr.nrGrowWhy);
            nr.nrGrowMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (now > had && now < nr.nrGrowWant && nr.nrGrowWhy.empty())
                return; // the next group builds the next pass
        }
    }
    nativeNrGrowEnd(nr, nullptr);
}

static void nativeLiveSets(NativeRife& nr)
{
    std::vector<std::pair<std::string, std::string>> q;
    if (nr.fuWarmTh && InterlockedCompareExchange(&nr.fuWarm, 0, 0) >= 2)
        nativeUpWarmEnd(nr); // an FSR / XeSS start in place: its first frames ended on the worker
    nativeNrGrowStep(nr);    // a DLSS 5 pass growth: one pass a frame group
    {
        std::lock_guard<std::mutex> lk(g_liveSets.m);
        if (g_liveSets.q.empty())
            return;
        q.swap(g_liveSets.q);
    }
    for (const auto& [key, val] : q)
    {
        const double v = atof(val.c_str());
        auto clampU = [](double x, double lo, double hi) { return (uint32_t)(x < lo ? lo : (x > hi ? hi : x)); };
        auto clampF = [](double x, double lo, double hi) { return (float)(x < lo ? lo : (x > hi ? hi : x)); };
        bool done = true;
        const bool nrOn = nr.liveNr && nr.nrHost && !nr.nrFailed;
        if (key == "sharpen" && v > 0.0 && (nr.sharpen > 0.0f || nativeSharpenOn(nr)))
            nr.sharpen = clampF(v, 0.0, 2.0);
        else if (key == "sharpen" && v <= 0.0)
            nr.sharpen = 0.0f;
        else if (key == "nrstruct" && nrOn)
            nr.nrHost->setStructure(clampF(v, 0.0, 2.0));
        else if (key == "nrtone" && nrOn)
            nr.nrHost->setTone(clampF(v, 0.0, 2.0));
        else if (key == "nrskin" && nrOn) // < 0 = follows the structure value
            nr.nrHost->setSkin(v < 0.0 ? -1.0f : clampF(v, 0.0, 2.0));
        else if (key == "nrautomask" && nrOn)
            nr.nrHost->setAutoMask(v != 0.0);
        else if (key == "nrstyle" && nrOn)
            nr.nrHost->setStyle((int)clampU(v, 0.0, 2.0));
        else if (key == "nrpasses" && nrOn)
        {
            // more passes than the session built: their features and frames one pass a frame group
            // (nativeNrGrowStep), as many as the video memory room holds; the growth's end logs this set's line. A new
            // value during a growth ends it first (its line names the cut)
            const int want = (int)clampU(v, 1.0, (double)nr::kMaxPasses);
            if (nr.nrGrowWant)
                nativeNrGrowEnd(nr, "another nrpasses arrived");
            if (want > nr.nrHost->passes())
            {
                nr.nrGrowWant = want;
                nr.nrGrowFrom = nr.nrHost->passes();
                nr.nrGrowMs = 0.0;
                nr.nrGrowVal = val;
                nr.nrGrowWhy.clear();
                nativeNrGrowStep(nr); // the first pass before this group
                continue;
            }
            nr.nrHost->runPasses(want);
        }
        else if ((key == "nrstruct" || key == "nrtone" || key == "nrskin" || key == "nrautomask" || key == "nrpasses" ||
                  key == "nrstyle") &&
                 !nrOn)
        {
            // DLSS 5 not running here: held for its start in place (key dlssnr)
            if (key == "nrstruct")
                g_nrStructure = clampF(v, 0.0, 2.0);
            else if (key == "nrtone")
                g_nrTone = clampF(v, 0.0, 2.0);
            else if (key == "nrskin")
                g_nrSkin = v < 0.0 ? -1.0 : (double)clampF(v, 0.0, 2.0);
            else if (key == "nrautomask")
                g_nrAutoMask = v != 0.0 ? 1 : 0;
            else if (key == "nrpasses")
                g_nrPasses = (int)clampU(v, 1.0, (double)nr::kMaxPasses);
            else
                g_nrStyle = (int)clampU(v, 0.0, 2.0);
            LOG("live set: %s %s (held for DLSS 5's start)\n", key.c_str(), val.c_str());
            continue;
        }
        else if (key == "dlssnr")
        {
            std::string why;
            if (!nativeNrSwitch(nr, v != 0.0, why))
            {
                LOG("live set: dlssnr %s (the next session takes it: %s)\n", val.c_str(), why.c_str());
                continue;
            }
        }
        else if (key == "hudlat" && !g_noHud)
            g_noHudLat.store(v == 0.0);
        else if (key == "hud")
            g_hudWant.store(v != 0.0 ? 1 : 0); // the loop thread owns the window (applyHudWant)
        else if (key == "restoremodel")
        {
            // AnimeJaNai's file (a path): held for Restore's start in place (`restore 2`); a running AnimeJaNai keeps
            // the file it loaded
            const bool ajRuns = nr.restore && nr.restNatKind == 2;
            if (ajRuns && val != g_restoreModel)
            {
                LOG("live set: restoremodel %s (the next session takes it: AnimeJaNai runs its file)\n", val.c_str());
                continue;
            }
            g_restoreModel = val;
            if (!ajRuns)
            {
                LOG("live set: restoremodel %s (held for Restore's start)\n", val.c_str());
                continue;
            }
        }
        else if (key == "a4kaa")
        {
            const std::string note = nativeA4kAaSwitch(nr, v != 0.0);
            if (!note.empty())
            {
                LOG("live set: a4kaa %s%s\n", val.c_str(), note.c_str());
                continue;
            }
        }
        else if (key == "up" || key == "restore" || key == "rtxhdr")
        {
            // the session's own kind (up: 1 RTX VSR, 2 AMD FSR 4, 3 Anime4K, 4 ArtCNN, 5 Intel XeSS; restore: 1
            // Real-ESRGAN, 2 AnimeJaNai, 3 Anime4K; rtxhdr: 1), 0 = it runs none
            const int kind = key == "up"        ? (!nr.vsr    ? 0
                                                   : nr.fsrUp ? (nr.fuKind ? 5 : 2)
                                                   : nr.a4kUp ? 3
                                                   : nr.acnUp ? 4
                                                              : 1)
                             : key == "restore" ? (nr.restore && nr.restNat ? nr.restNatKind : 0)
                                                : (nr.rtxHdr ? 1 : 0);
            bool& failed = key == "up" ? nr.vsrFailed : key == "restore" ? nr.restFailed : nr.rtxFailed;
            bool& userOff = key == "up" ? nr.vsrUserOff : key == "restore" ? nr.restUserOff : nr.rtxUserOff;
            const int want = (int)clampU(v, 0.0, 9.0);
            if (key == "up" && nr.fuWarmTh)
            {
                // an FSR / XeSS start in place still warming: the tick's last state decides at its end
                const int pend = nr.fuKind ? 5 : 2;
                if (want == 0 || want == pend)
                {
                    nr.fuWarmWant = want != 0;
                    LOG("live set: up %s (%s)\n", val.c_str(),
                        want ? "its start in place is under way" : "off once its start in place ends");
                }
                else
                    LOG("live set: up %s (the next session takes it: another kind is starting in place)\n",
                        val.c_str());
                continue;
            }
            const char* why = nullptr;
            if (want == 0 && kind)
            {
                if (!failed)
                    failed = userOff = true;
            }
            else if (want && want == kind)
            {
                if (userOff)
                    failed = userOff = false;
                else if (failed)
                    why = "it stopped earlier in this session";
            }
            else if (want != kind)
                why = kind ? "another kind than this session runs" : "this session started without it";
            // RTX HDR on stays a session start: its start sets the picture path (the HDR route, an upscaler before the
            // model, the model's colour domain, the one NGX create)
            if (why && key == "rtxhdr" && want && !kind)
                why = "this session started without it: RTX HDR sets the picture path at the session's start";
            // an upscaler the session started without, or another kind than it runs: its setup in place when this
            // session can run it (the running kind released first, nativeUpRelease)
            std::string whyUp;
            const bool other = key == "up" && want && want != kind;
            const bool restIn = key == "restore" && want && !kind; // Restore in a session that started without it
            if (why && other && nativeUpSwitch(nr, want, whyUp))
                why = nullptr;
            else if (why && restIn && nativeRestoreSwitch(nr, want, whyUp))
                why = nullptr;
            else if (why && (other || restIn))
            {
                whyUp = std::string(why) + ": " + whyUp;
                why = whyUp.c_str();
            }
            if (why)
            {
                LOG("live set: %s %s (the next session takes it: %s)\n", key.c_str(), val.c_str(), why);
                continue;
            }
        }
        else if (key == "vsrq" && !nr.fsrUp && !nr.a4kUp && !nr.acnUp)
        {
            // RTX VSR's level: read at every evaluate while it runs; held for its start in place (nativeUpSwitch)
            nr.vsrSet.QualityLevel = clampU(v, 1.0, 4.0);
            if (!nr.vsr)
            {
                LOG("live set: vsrq %s (held for RTX VSR's start)\n", val.c_str());
                continue;
            }
        }
        else if (key == "hdrsat" && nr.rtxHdr)
            nr.thdr.Saturation = clampU(v, 0.0, 200.0);
        else if (key == "hdrcon" && nr.rtxHdr)
            nr.thdr.Contrast = clampU(v, 0.0, 200.0);
        else if (key == "hdrvib" && nr.rtxHdr)
            nr.rtxVib = clampF(v, 0.0, 1.0);
        else if (key == "hdrsb" && nr.rtxHdr)
            nr.rtxSb = clampF(v, 0.0, 1.0);
        else
            done = false;
        LOG("live set: %s %s%s\n", key.c_str(), val.c_str(), done ? "" : " (the next session takes it)");
    }
}

static void nativeThread(NativeRife* nrp, IDXGIAdapter1* adapter, HANDLE hTex, HANDLE hFence, HANDLE hOutBuf,
                         uint64_t outBytes, bool earlyDone)
{
    NativeRife& nr = *nrp;
    // earlyDone: PipeServer::beginNativeHandoff built everything but the output ring on its
    // own thread; this thread only binds the same device and finishes the ring
    if (!earlyDone)
        nativeVideoMemoryMark();
    bool ok = earlyDone ? nativeBindDevice(nr)
                        : nativeCudaDeviceInit(nr, adapter, hTex, hFence, (uint32_t)nr.cw, (uint32_t)nr.ch, nr.hdr) &&
                              nativeCudaInitEarly(nr, nativeCacheDir(nr)) && nativeModelInit(nr) && nativeRtxInit(nr) &&
                              nativeLiveNrInit(nr);
    ok = ok && nativeCudaInitLate(nr, hOutBuf, outBytes);
    {
        std::lock_guard<std::mutex> lk(nr.mInit);
        nr.initState = ok ? 1 : -1;
    }
    nr.cvInit.notify_all();
    if (!ok)
    {
        InterlockedExchange(&nr.dead, 1);
        nr.cvTok.notify_all();
        nativeFree(nr);
        return;
    }
    LOG("native host ready: model %dx%d padded %dx%d, out %dx%d at (%d,%d), batch max %d%s%s%s%s%s%s%s%s%s%s%s\n", nr.w,
        nr.h, nr.pw, nr.ph, nr.dw, nr.dh, nr.x0, nr.y0, nr.batchMax, nr.noEngine ? ", engine=none (effects only)" : "",
        nr.gmfss ? ", engine=gmfss" : "", nr.nvof ? ", engine=nvof" : "",
        nr.fruc ? (nr.fsrfg ? ", engine=fsrfg" : ", engine=fruc") : "", nr.drba ? ", engine=drba (lag 1)" : "",
        nr.rtxHdr ? ", rtxhdr=native" : "", nr.sharpen > 0.0f ? ", sharpen=native" : "",
        nr.vsr ? (nr.fsrUp   ? (nr.fuKind ? ", xessup=native" : ", fsrup=native")
                  : nr.a4kUp ? ", anime4k=native"
                  : nr.acnUp ? ", artcnn=native"
                             : ", vsr=native")
               : "",
        nr.uw ? ", upscale=native" : "", nr.fitAa ? ", fit=native-aa" : "",
        nr.restore && nr.restNat ? (nr.restNatKind == 3 ? ", restore=anime4k" : ", restore=native") : "");
    for (;;)
    {
        std::vector<uint8_t> msg;
        {
            std::unique_lock<std::mutex> lk(nr.mMsg);
            nr.cvMsg.wait(lk, [&] { return !nr.msgs.empty() || nr.isDead(); });
            if (nr.msgs.empty())
                break;
            msg = std::move(nr.msgs.front());
            nr.msgs.pop_front();
        }
        nativeLiveSets(nr);
        if (!nativeGroup(nr, msg))
            break;
    }
    {
        std::lock_guard<std::mutex> lk(nr.mTok);
        InterlockedExchange(&nr.dead, 1);
    }
    nr.cvTok.notify_all();
    nativeFree(nr);
}

// Start the in-process host. Returns false on ANY problem (one log line says why) and the
// session ends.
static bool nativeStart(NativeRife& nr, const std::wstring& script, const std::wstring& backend, uint32_t capW,
                        uint32_t capH, IDXGIAdapter1* adapter, HANDLE hTex, HANDLE hFence, HANDLE hOutBuf,
                        uint64_t outBytes, uint32_t slots, uint32_t pitch, size_t slotBytes, bool earlyDone)
{
    // earlyDone: PipeServer::beginNativeHandoff already ran the DLL load, the HDR config, the
    // engine handoff, the CUDA device init, the kernels, the buffers and the engine load on
    // its own thread; only the output ring (sized from the slot count) is left
    if (!nativeLoadDlls(script))
        return false;
    if (!earlyDone)
    {
        if (!nativeConfigHdr(nr))
            return false;
        if (!nativeHandoff(script, backend, capW, capH, nr))
            return false;
    }
    nr.slots = slots;
    nr.pitch = pitch;
    nr.slotBytes = slotBytes;
    nr.th = std::thread(nativeThread, &nr, adapter, hTex, hFence, hOutBuf, outBytes, earlyDone);
    std::unique_lock<std::mutex> lk(nr.mInit);
    nr.cvInit.wait(lk, [&] { return nr.initState != 0; });
    const bool ok = nr.initState > 0;
    lk.unlock();
    if (!ok)
    {
        if (nr.th.joinable())
            nr.th.join();
        LOG("native host init failed\n");
        return false;
    }
    nr.started = true;
    return true;
}

static void nativeStop(NativeRife& nr)
{
    if (!nr.started)
        return;
    nr.die("stopping");
    if (nr.th.joinable())
        nr.th.join();
    nr.started = false;
}

// a NativeRife the early thread initialised but no compute thread ever owned (a refusal in
// startNative, or a session that ends before it): free its CUDA state from the calling
// thread. Without a stream nothing CUDA was created (the stream is the first CUDA object), and
// the runtime DLL may not even be loaded, so no CUDA call is made at all then.
static void nativeDropEarly(NativeRife& nr)
{
    if (!nr.stream)
        return;
    nativeBindDevice(nr);
    nativeFree(nr);
}
