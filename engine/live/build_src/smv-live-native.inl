// part of smv-live.cpp (unity include, see the file map there); not a standalone translation unit
// ---------------------------------------------------------------- the native model host
//
// Every live model runs INSIDE this process: the TensorRT-RTX C++ runtime for the engine
// models (the host looks its engines up by name and builds missing ones from engine\onnx), the
// driver's Optical Flow API for nvof, the shipped bridge DLL for fruc. No python process is
// part of a live session. The python live server this host replaced (engine/live_server.py, in
// the git history) is where most of the math below was ported from; comments that name its
// classes and functions record that origin.
//
// NativeRife sits behind the PipeServer boundary: writeFull() hands it one group message per
// captured frame ([u32 N][N f32 fractions]) and readFullRaw() hands back the tokens the
// reader thread consumes (slot (i+1), the capture-release token 0x7FFFFFFF, the bare
// 0x80000000 end marker).
//
// Everything CUDA happens on ONE compute thread and ONE non-default stream:
//   * a TRT-RTX execution context per engine (contexts are not documented thread safe);
//   * the default stream is never used (TRT inserts a device sync there, SMV's permanent-hang
//     class, see NvInferRuntime.h :5039 and live_server.py's stream comment);
//   * the engine's own kWHOLE_GRAPH_CAPTURE is the ONLY graph capture (never capture a TRT
//     enqueue from outside, per the CUDA study 01 section B1).
//
// Delay-loaded DLLs: tensorrt_rtx_1_6.dll, cudart64_13.dll and nvrtc64_130_0.dll ship in
// engine\gpu_runtime (staged from the dev python's wheels by scripts/stage-gpu-runtime.js).

// cuda_runtime.h pulls the internal crt\ headers the runtime WHEEL does not ship; the host
// side only ever needs the API declarations, so cuda_runtime_api.h plus the one-macro shim in
// cuda_shim\crt is the whole build dependency (see that file).
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuda_d3d11_interop.h>
#include <nvrtc.h>
#include <NvInferRuntime.h>
#include <NvInfer.h>           // the builder: engines built from engine\onnx
#include <NvOnnxParser.h>      // tensorrt_onnxparser_rtx_1_6.dll, delay-loaded like the runtime
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
// are set: the host reads them for its launch sizes (lzTaps), the cubin cache key hashes this text,
// a harness may pass -D to try others.
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
// One axis of a separable pass (the shrinking fit, Restore's fold, the live capture resize): the
// taps depend on the output index alone, so each block computes them ONCE into shared memory,
// every thread of the block sharing the work: entry (lane, tap) holds the tap's mirrored source
// index and its weight, tap-major so a warp reads them without bank conflicts, then one thread per
// lane normalises its taps in tap order (rsInv's order, so the per-pixel form gives the same
// bytes). Lanes past the last output index repeat it (last). aaTabOk false (a shrink beyond
// kAaMaxTaps / (2 kLzDown), 4.57x at seven lobes, or a block side above kAaSide) = the per-pixel form.
#define kAaSide 32
#define kAaMaxTaps 64
struct AaTab { float wt[kAaMaxTaps * kAaSide]; int ix[kAaMaxTaps * kAaSide]; };
__device__ __forceinline__ bool aaTabOk(int fs, int side)
{
    return side <= kAaSide && fs <= kAaMaxTaps;
}
// the table at any stride (tap k of lane at k * stride + lane): AaTab's is kAaSide, the tiled
// fit's tables in dynamic shared memory use their block side
__device__ __forceinline__ void aaTabFillS(float* wt, int* ix, int stride, int o0, int side, int in, int out, int fs,
                                           int last)
{
    const int nt = blockDim.x * blockDim.y, t = threadIdx.y * blockDim.x + threadIdx.x;
    for (int e = t; e < side * fs; e += nt)
    {
        const int lane = e / fs, k = e - lane * fs;
        const int o = o0 + lane < last ? o0 + lane : last;
        const int mn = rsBegin(o, in, out, fs);
        wt[k * stride + lane] = rsW(mn + k, o, in, out);
        ix[k * stride + lane] = rsMirror(mn + k, in);
    }
    __syncthreads();
    if (t < side)
    {
        float s = 0.0f;
        for (int k = 0; k < fs; k++) s += wt[k * stride + t];
        const float inv = 1.0f / s;
        for (int k = 0; k < fs; k++) wt[k * stride + t] *= inv;
    }
    __syncthreads();
}
__device__ __forceinline__ void aaTabFill(AaTab& T, int o0, int side, int in, int out, int fs, int last)
{
    aaTabFillS(T.wt, T.ix, kAaSide, o0, side, in, out, fs, last);
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
                          float* __restrict__ tmp, int w)
{
    __shared__ AaTab T;
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int sy = blockIdx.y * blockDim.y + threadIdx.y;
    const int fs = rsTaps(cw, w);
    const bool tab = aaTabOk(fs, blockDim.x);
    if (tab) aaTabFill(T, blockIdx.x * blockDim.x, blockDim.x, cw, w, fs, w - 1);
    if (ox >= w || sy >= ch) return;
    const int mn = tab ? 0 : rsBegin(ox, cw, w, fs);
    const float inv = tab ? 0.0f : rsInv(ox, cw, w, mn, fs);
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
    for (int k = 0; k < fs; k++)
    {
        const float wt = tab ? T.wt[k * kAaSide + threadIdx.x] : rsW(mn + k, ox, cw, w) * inv;
        const int ix = tab ? T.ix[k * kAaSide + threadIdx.x] : rsMirror(mn + k, cw);
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
                          float* __restrict__ dst, int h, int ph, int pw, int planeStride)
{
    __shared__ AaTab T;
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    const int fs = rsTaps(ch, h);
    const bool tab = aaTabOk(fs, blockDim.y);
    if (tab) aaTabFill(T, blockIdx.y * blockDim.y, blockDim.y, ch, h, fs, h - 1);
    if (ox >= pw || oy >= ph) return;
    const int tx = ox < w ? ox : w - 1;            // replicate pad in x
    const int ty = oy < h ? oy : h - 1;            // replicate pad in y (the table's lanes repeat h - 1)
    const int mn = tab ? 0 : rsBegin(ty, ch, h, fs);
    const float inv = tab ? 0.0f : rsInv(ty, ch, h, mn, fs);
    const size_t plane = (size_t)ch * w;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
    for (int k = 0; k < fs; k++)
    {
        const float wt = tab ? T.wt[k * kAaSide + threadIdx.y] : rsW(mn + k, ty, ch, h) * inv;
        const int iy = tab ? T.ix[k * kAaSide + threadIdx.y] : rsMirror(mn + k, ch);
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
                           float* __restrict__ tmp, int w)
{
    __shared__ AaTab T;
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int sy = blockIdx.y * blockDim.y + threadIdx.y;
    const int fs = rsTaps(cw, w);
    const bool tab = aaTabOk(fs, blockDim.x);
    if (tab) aaTabFill(T, blockIdx.x * blockDim.x, blockDim.x, cw, w, fs, w - 1);
    if (ox >= w || sy >= ch) return;
    const int mn = tab ? 0 : rsBegin(ox, cw, w, fs);
    const float inv = tab ? 0.0f : rsInv(ox, cw, w, mn, fs);
    const size_t splane = (size_t)ch * cw;
    const size_t dplane = (size_t)ch * w;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
    for (int k = 0; k < fs; k++)
    {
        const float wt = tab ? T.wt[k * kAaSide + threadIdx.x] : rsW(mn + k, ox, cw, w) * inv;
        const int ix = tab ? T.ix[k * kAaSide + threadIdx.x] : rsMirror(mn + k, cw);
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

// one output pixel of a slot: crop the model pad, then copy 1:1 when the slot matches the
// model size, else resize with Lanczos on both axes (the placement at the top of this block).
// The SDR and HDR slot packers below share this sampler and differ only in the store.
// The taps depend on ox alone (x) or oy alone (y), so each block computes them ONCE into shared
// memory, every thread sharing the work: the mirrored source index and the weight per (lane, tap),
// tap-major so a warp reads them without bank conflicts, then one thread per column lane and one
// per row lane normalises its taps in tap order (rsInv's order, so the per-pixel form gives the
// same bytes). false = the 1:1 copy, a block side above kOutSide, an axis with more than kOutTaps
// taps (an enlarge, or a shrink by at most 4 / 3) or a block with fewer threads than column and
// row lanes together (the per-pixel form).
#define kOutSide 32
#define kOutTaps (2 * ((4 * kLzDown + 2) / 3) > 2 * kLzUp ? 2 * ((4 * kLzDown + 2) / 3) : 2 * kLzUp)
struct OutTab
{
    float wx[kOutTaps * kOutSide], wy[kOutTaps * kOutSide];
    int cx[kOutTaps * kOutSide], cy[kOutTaps * kOutSide];
};
__device__ __forceinline__ bool outTabFill(OutTab& T, int w, int h, int dw, int dh)
{
    const int fsx = rsTaps(w, dw), fsy = rsTaps(h, dh);
    const int bx = blockDim.x, by = blockDim.y, nt = bx * by, t = threadIdx.y * bx + threadIdx.x;
    if ((dw == w && dh == h) || bx > kOutSide || by > kOutSide || fsx > kOutTaps || fsy > kOutTaps ||
        nt < bx + by)
        return false;
    const int nx = bx * fsx, ny = by * fsy;
    for (int e = t; e < nx + ny; e += nt)
    {
        const bool col = e < nx;
        const int fs = col ? fsx : fsy, ee = col ? e : e - nx, lane = ee / fs, k = ee - lane * fs;
        const int in = col ? w : h, out = col ? dw : dh;
        int o = (col ? blockIdx.x * bx : blockIdx.y * by) + lane;
        if (o > out - 1) o = out - 1;
        const int mn = rsBegin(o, in, out, fs);
        (col ? T.wx : T.wy)[k * kOutSide + lane] = rsW(mn + k, o, in, out);
        (col ? T.cx : T.cy)[k * kOutSide + lane] = rsMirror(mn + k, in);
    }
    __syncthreads();
    if (t < bx + by)
    {
        const bool col = t < bx;
        float* wv = col ? T.wx : T.wy;
        const int lane = col ? t : t - bx, fs = col ? fsx : fsy;
        float s = 0.0f;
        for (int k = 0; k < fs; k++) s += wv[k * kOutSide + lane];
        const float inv = 1.0f / s;
        for (int k = 0; k < fs; k++) wv[k * kOutSide + lane] *= inv;
    }
    __syncthreads();
    return true;
}

// one element of a planar source that is fp32, or fp16 (half != 0: the IFNet's tweens, read as
// the engine wrote them, so no pass widens them first)
__device__ __forceinline__ float ldS(const void* __restrict__ src, int half, size_t o)
{
    return half ? h2f(((const unsigned short*)src)[o]) : ((const float*)src)[o];
}

// the table form's gather: the column's taps in registers (a fixed NT loop the compiler unrolls,
// NT >= fsx); the sums run in the per-pixel form's order, so both forms give the same bytes (C++
// linkage: the text sits in one extern "C" block, and a template cannot have C linkage)
extern "C++" template <int NT>
__device__ __forceinline__ void sampleOutTab(const void* __restrict__ src, int half, size_t p1, size_t p2,
                                             int rowStride, int fsx, int fsy, const OutTab* T, float& a0,
                                             float& a1, float& a2)
{
    float wxr[NT];
    int cxr[NT];
#pragma unroll
    for (int i = 0; i < NT; i++)
    {
        wxr[i] = i < fsx ? T->wx[i * kOutSide + threadIdx.x] : 0.0f;
        cxr[i] = i < fsx ? T->cx[i * kOutSide + threadIdx.x] : 0;
    }
    for (int j = 0; j < fsy; j++)
    {
        const float wy = T->wy[j * kOutSide + threadIdx.y];
        const size_t row = (size_t)T->cy[j * kOutSide + threadIdx.y] * rowStride;
        float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f;
#pragma unroll
        for (int i = 0; i < NT; i++)
        {
            if (i < fsx)
            {
                const size_t o = row + cxr[i];
                r0 += wxr[i] * ldS(src, half, o);
                r1 += wxr[i] * ldS(src, half, p1 + o);
                r2 += wxr[i] * ldS(src, half, p2 + o);
            }
        }
        a0 += wy * r0;
        a1 += wy * r1;
        a2 += wy * r2;
    }
}

// T = the block's filled table, or nullptr for the per-pixel taps
__device__ __forceinline__ void sampleOut(const void* __restrict__ src, int half, int planeStride,
                                          int rowStride, int w, int h, int ox, int oy,
                                          int dw, int dh, const OutTab* T, float c[3])
{
    if (dw == w && dh == h)
    {
        const size_t o = (size_t)oy * rowStride + ox;
        c[0] = ldS(src, half, o);
        c[1] = ldS(src, half, planeStride + o);
        c[2] = ldS(src, half, 2 * planeStride + o);
        return;
    }
    const int fsx = rsTaps(w, dw), fsy = rsTaps(h, dh);
    const size_t p1 = planeStride, p2 = 2 * (size_t)planeStride;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
    if (T)
    {
        // an enlarging (or same-size) x axis has 2 kLzUp taps: unrolling kOutTaps there (a 4 / 3
        // shrink's count) only adds predicated-off slots and registers
        if (fsx <= 2 * kLzUp)
            sampleOutTab<2 * kLzUp>(src, half, p1, p2, rowStride, fsx, fsy, T, a0, a1, a2);
        else
            sampleOutTab<kOutTaps>(src, half, p1, p2, rowStride, fsx, fsy, T, a0, a1, a2);
    }
    else
    {
        const int mx = rsBegin(ox, w, dw, fsx), my = rsBegin(oy, h, dh, fsy);
        const float invx = rsInv(ox, w, dw, mx, fsx), invy = rsInv(oy, h, dh, my, fsy);
        for (int j = 0; j < fsy; j++)
        {
            const float wy = rsW(my + j, oy, h, dh) * invy;
            const size_t row = (size_t)rsMirror(my + j, h) * rowStride;
            float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f;
            for (int i = 0; i < fsx; i++)
            {
                const float wx = rsW(mx + i, ox, w, dw) * invx;
                const size_t o = row + rsMirror(mx + i, w);
                r0 += wx * ldS(src, half, o);
                r1 += wx * ldS(src, half, p1 + o);
                r2 += wx * ldS(src, half, p2 + o);
            }
            a0 += wy * r0;
            a1 += wy * r1;
            a2 += wy * r2;
        }
    }
    c[0] = a0;
    c[1] = a1;
    c[2] = a2;
}

// one output slot: sampleOut, clamp, round, and write the (R, G, B) planes as BGRA8 into the
// content rect of a pitched ring slot.
__global__ void k_packOut(const void* __restrict__ src, int half, int planeStride, int rowStride,
                          int w, int h, unsigned char* __restrict__ dst, int pitch,
                          int x0, int y0, int dw, int dh)
{
    __shared__ OutTab T;
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    const bool tab = outTabFill(T, w, h, dw, dh);
    if (ox >= dw || oy >= dh) return;
    float c[3];
    sampleOut(src, half, planeStride, rowStride, w, h, ox, oy, dw, dh, tab ? &T : nullptr, c);
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
    __shared__ OutTab T;
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    const bool tab = outTabFill(T, w, h, dw, dh);
    if (ox >= dw || oy >= dh) return;
    float c[3];
    sampleOut(src, half, planeStride, rowStride, w, h, ox, oy, dw, dh, tab ? &T : nullptr, c);
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
// (upscale, then the sharpen): the fit lands in a planar staging frame at dw x dh (k_fitPlanar, or
// the VSR bridge output unpacked by k_unpackBgraRgb), and the sharpen kernels apply Adaptive Sharpen
// on it while doing the slot store that k_packOut / k_packOutHdr do without effects.

// crop the model pad and quantise to tight 8-bit pixels at the model size, plane c into byte c,
// alpha 255: the (R, G, B) planes as the offline DLSS 4.5 host's RGBA8 frame
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

// the (R, G, B) planes of either route as the B G R A bytes of the VSR bridge and NvOFFRUC, and
// back, as rtxvideo.py run_vsr packs and unpacks them (back with a true / 255)
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

// RGBA8 back to the offline (R, G, B) planes: a DLSS 4.5 host frame,
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

// sampleOut's fit into the planar staging frame (the sharpen input when VSR is off)
__global__ void k_fitPlanar(const void* __restrict__ src, int half, int planeStride, int rowStride,
                            int w, int h, float* __restrict__ dst, int dw, int dh)
{
    __shared__ OutTab T;
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    const bool tab = outTabFill(T, w, h, dw, dh);
    if (ox >= dw || oy >= dh) return;
    float c[3];
    sampleOut(src, half, planeStride, rowStride, w, h, ox, oy, dw, dh, tab ? &T : nullptr, c);
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

// ---- the shrinking fit ----------------------------------------------------------------------
// The Lanczos resize as a separable pair into the planar staging frame (the AaTab tables at the
// top of this block): 2 kLzUp taps per axis on an enlarging axis, 2 ceil(kLzDown in / out) on a
// shrinking one, so a pixel reads the sum of its two axes' taps where sampleOut's 2D gather reads
// their product. Every live fit that changes the size runs through it; the offline stages keep
// sampleOut (NVENC paces them).

// horizontal pass: the model pad cropped to (w, h) -> tmp (3, h, dw)
__global__ void k_fitAaH(const void* __restrict__ src, int half, int planeStride, int rowStride,
                         int w, int h, float* __restrict__ tmp, int dw)
{
    __shared__ AaTab T;
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int fs = rsTaps(w, dw);
    const bool tab = aaTabOk(fs, blockDim.x);
    if (tab) aaTabFill(T, blockIdx.x * blockDim.x, blockDim.x, w, dw, fs, dw - 1);
    if (ox >= dw || y >= h) return;
    const int mn = tab ? 0 : rsBegin(ox, w, dw, fs);
    const float inv = tab ? 0.0f : rsInv(ox, w, dw, mn, fs);
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
    for (int k = 0; k < fs; k++)
    {
        const float wt = tab ? T.wt[k * kAaSide + threadIdx.x] : rsW(mn + k, ox, w, dw) * inv;
        const int ix = tab ? T.ix[k * kAaSide + threadIdx.x] : rsMirror(mn + k, w);
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
                                       const AaTab& T, bool tab, int fs, float a[3])
{
    const int mn = tab ? 0 : rsBegin(oy, h, dh, fs);
    const float inv = tab ? 0.0f : rsInv(oy, h, dh, mn, fs);
    const size_t splane = (size_t)h * dw;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
    for (int k = 0; k < fs; k++)
    {
        const float wt = tab ? T.wt[k * kAaSide + threadIdx.y] : rsW(mn + k, oy, h, dh) * inv;
        const int iy = tab ? T.ix[k * kAaSide + threadIdx.y] : rsMirror(mn + k, h);
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
                         float* __restrict__ dst, int dh)
{
    __shared__ AaTab T;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    const int fs = rsTaps(h, dh);
    const bool tab = aaTabOk(fs, blockDim.y);
    if (tab) aaTabFill(T, blockIdx.y * blockDim.y, blockDim.y, h, dh, fs, dh - 1);
    if (x >= dw || oy >= dh) return;
    float a[3];
    aaVSum(tmp, dw, h, dh, x, oy, T, tab, fs, a);
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
                           int pitch, int x0, int y0, int dh)
{
    __shared__ AaTab T;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    const int fs = rsTaps(h, dh);
    const bool tab = aaTabOk(fs, blockDim.y);
    if (tab) aaTabFill(T, blockIdx.y * blockDim.y, blockDim.y, h, dh, fs, dh - 1);
    if (x >= dw || oy >= dh) return;
    float c[3];
    aaVSum(tmp, dw, h, dh, x, oy, T, tab, fs, c);
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
                              int pitch, int x0, int y0, int dh)
{
    __shared__ AaTab T;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    const int fs = rsTaps(h, dh);
    const bool tab = aaTabOk(fs, blockDim.y);
    if (tab) aaTabFill(T, blockIdx.y * blockDim.y, blockDim.y, h, dh, fs, dh - 1);
    if (x >= dw || oy >= dh) return;
    float c[3];
    aaVSum(tmp, dw, h, dh, x, oy, T, tab, fs, c);
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

// the pair and the slot store in ONE kernel, for a fit that shrinks the height (a block's source
// rows then barely overlap its neighbours'): both tap tables and the horizontal pass of the source
// rows the block's vertical taps read go into dynamic shared memory (k_fitAaH's sums in its
// order), then the vertical taps in aaVSum's order and k_packOutV's (hdr 0) or k_packOutHdrV's
// (hdr 1) store: the pair's bytes without the global tmp. Shared memory (the launch sizes it):
// fsh x blockDim.x and fsv x blockDim.y table entries, two ints, the tile (3, rcap, blockDim.x);
// rcap = ceil((blockDim.y - 1) h / dh) + fsv + 2 bounds a block's row span (rsBegin is a floor)
__global__ void k_packOutTile(const void* __restrict__ src, int half, int planeStride, int rowStride, int w, int h,
                              unsigned char* __restrict__ dst, int pitch, int x0, int y0, int dw, int dh, int hdr,
                              int rcap)
{
    extern __shared__ float sm[];
    const int bx = blockDim.x, by = blockDim.y;
    const int nt = bx * by, t = threadIdx.y * bx + threadIdx.x;
    const int fsh = rsTaps(w, dw), fsv = rsTaps(h, dh);
    float* wtH = sm;
    int* ixH = (int*)(wtH + fsh * bx);
    float* wtV = (float*)(ixH + fsh * bx);
    int* ixV = (int*)(wtV + fsv * by);
    int* lim = ixV + fsv * by;
    float* tile = (float*)(lim + 2);
    if (t == 0) { lim[0] = h; lim[1] = -1; }
    aaTabFillS(wtH, ixH, bx, blockIdx.x * bx, bx, w, dw, fsh, dw - 1);
    aaTabFillS(wtV, ixV, by, blockIdx.y * by, by, h, dh, fsv, dh - 1);
    for (int e = t; e < by * fsv; e += nt)
    {
        atomicMin(&lim[0], ixV[e]);
        atomicMax(&lim[1], ixV[e]);
    }
    __syncthreads();
    const int r0 = lim[0], nr = lim[1] - lim[0] + 1;
    if (nr > rcap) return;
    for (int e = t; e < nr * bx; e += nt)
    {
        const int r = e / bx, c = e - r * bx;
        const size_t row = (size_t)(r0 + r) * rowStride;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
        for (int k = 0; k < fsh; k++)
        {
            const float wt = wtH[k * bx + c];
            const size_t o = row + ixH[k * bx + c];
            a0 += wt * ldS(src, half, o);
            a1 += wt * ldS(src, half, planeStride + o);
            a2 += wt * ldS(src, half, 2 * planeStride + o);
        }
        tile[r * bx + c] = a0;
        tile[(nr + r) * bx + c] = a1;
        tile[(2 * nr + r) * bx + c] = a2;
    }
    __syncthreads();
    const int x = blockIdx.x * bx + threadIdx.x, oy = blockIdx.y * by + threadIdx.y;
    if (x >= dw || oy >= dh) return;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
    for (int k = 0; k < fsv; k++)
    {
        const float wt = wtV[k * by + threadIdx.y];
        const int r = ixV[k * by + threadIdx.y] - r0;
        a0 += wt * tile[r * bx + threadIdx.x];
        a1 += wt * tile[(nr + r) * bx + threadIdx.x];
        a2 += wt * tile[(2 * nr + r) * bx + threadIdx.x];
    }
    const float c[3] = {a0, a1, a2};
    if (!hdr)
    {
        unsigned char* p = dst + (size_t)(y0 + oy) * pitch + (size_t)(x0 + x) * 4;
        for (int ci = 0; ci < 3; ci++)
        {
            float v = c[ci];
            v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
            p[2 - ci] = (unsigned char)(int)rintf(v * 255.0f);
        }
        p[3] = 255;
        return;
    }
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

// the pair into a planar target in ONE kernel (Upscale to's first resize where it beats the pair),
// k_packOutTile's layout: both tap tables and k_fitAaH's sums of the source rows the block's vertical
// taps read in dynamic shared memory, then aaVSum's vertical sum into the planar target (3, dh, dw):
// k_fitAaV's bytes without the global tmp
__global__ void k_fitAaTile(const void* __restrict__ src, int half, int planeStride, int rowStride, int w, int h,
                            float* __restrict__ dst, int dw, int dh, int rcap)
{
    extern __shared__ float sm[];
    const int bx = blockDim.x, by = blockDim.y;
    const int nt = bx * by, t = threadIdx.y * bx + threadIdx.x;
    const int fsh = rsTaps(w, dw), fsv = rsTaps(h, dh);
    float* wtH = sm;
    int* ixH = (int*)(wtH + fsh * bx);
    float* wtV = (float*)(ixH + fsh * bx);
    int* ixV = (int*)(wtV + fsv * by);
    int* lim = ixV + fsv * by;
    float* tile = (float*)(lim + 2);
    if (t == 0) { lim[0] = h; lim[1] = -1; }
    aaTabFillS(wtH, ixH, bx, blockIdx.x * bx, bx, w, dw, fsh, dw - 1);
    aaTabFillS(wtV, ixV, by, blockIdx.y * by, by, h, dh, fsv, dh - 1);
    for (int e = t; e < by * fsv; e += nt)
    {
        atomicMin(&lim[0], ixV[e]);
        atomicMax(&lim[1], ixV[e]);
    }
    __syncthreads();
    const int r0 = lim[0], nr = lim[1] - lim[0] + 1;
    if (nr > rcap) return;
    for (int e = t; e < nr * bx; e += nt)
    {
        const int r = e / bx, c = e - r * bx;
        const size_t row = (size_t)(r0 + r) * rowStride;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
        for (int k = 0; k < fsh; k++)
        {
            const float wt = wtH[k * bx + c];
            const size_t o = row + ixH[k * bx + c];
            a0 += wt * ldS(src, half, o);
            a1 += wt * ldS(src, half, planeStride + o);
            a2 += wt * ldS(src, half, 2 * planeStride + o);
        }
        tile[r * bx + c] = a0;
        tile[(nr + r) * bx + c] = a1;
        tile[(2 * nr + r) * bx + c] = a2;
    }
    __syncthreads();
    const int x = blockIdx.x * bx + threadIdx.x, oy = blockIdx.y * by + threadIdx.y;
    if (x >= dw || oy >= dh) return;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
    for (int k = 0; k < fsv; k++)
    {
        const float wt = wtV[k * by + threadIdx.y];
        const int r = ixV[k * by + threadIdx.y] - r0;
        a0 += wt * tile[r * bx + threadIdx.x];
        a1 += wt * tile[(nr + r) * bx + threadIdx.x];
        a2 += wt * tile[(2 * nr + r) * bx + threadIdx.x];
    }
    const size_t plane = (size_t)dw * dh;
    const size_t o = (size_t)oy * dw + x;
    dst[o] = a0;
    dst[plane + o] = a1;
    dst[2 * plane + o] = a2;
}

// ---- live Restore ---------------------------------------------------------------------------
// live_server._Fit._restore: the Real-ESRGAN TensorRT engine (x = the model frame as fp16
// NCHW [1,3,h,w], y = its 4x reconstruction [1,3,4h,4w]) then realesr.fit to the restore
// target: `out.clamp(0,1)` first, the Lanczos pair when the target height shrinks (the pair
// above with the clamp folded in and the engine's own dtype read at the taps), sampleOut's
// Lanczos when it enlarges (k_restToF + k_fitPlanar + k_clamp01), identity when equal.
__device__ __forceinline__ unsigned short f2h(float f)
{
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
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

// the shrinking fold, horizontal pass: the 4x output (fp16 or fp32 planar, clamped per tap)
// -> tmp (3, h, dw); the taps are k_fitAaH's
__global__ void k_restFoldH(const void* __restrict__ src, int half, int planeStride, int rowStride,
                            int w, int h, float* __restrict__ tmp, int dw)
{
    __shared__ AaTab T;
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int fs = rsTaps(w, dw);
    const bool tab = aaTabOk(fs, blockDim.x);
    if (tab) aaTabFill(T, blockIdx.x * blockDim.x, blockDim.x, w, dw, fs, dw - 1);
    if (ox >= dw || y >= h) return;
    const int mn = tab ? 0 : rsBegin(ox, w, dw, fs);
    const float inv = tab ? 0.0f : rsInv(ox, w, dw, mn, fs);
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
    for (int k = 0; k < fs; k++)
    {
        const float wt = tab ? T.wt[k * kAaSide + threadIdx.x] : rsW(mn + k, ox, w, dw) * inv;
        const int ix = tab ? T.ix[k * kAaSide + threadIdx.x] : rsMirror(mn + k, w);
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

// vertical pass with realesr.fit's final clamp: tmp (3, h, dw) -> the planar target (3, dh, dw)
__global__ void k_restFoldV(const float* __restrict__ tmp, int dw, int h,
                            float* __restrict__ dst, int dh)
{
    __shared__ AaTab T;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    const int fs = rsTaps(h, dh);
    const bool tab = aaTabOk(fs, blockDim.y);
    if (tab) aaTabFill(T, blockIdx.y * blockDim.y, blockDim.y, h, dh, fs, dh - 1);
    if (x >= dw || oy >= dh) return;
    const int mn = tab ? 0 : rsBegin(oy, h, dh, fs);
    const float inv = tab ? 0.0f : rsInv(oy, h, dh, mn, fs);
    const size_t splane = (size_t)h * dw;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
    for (int k = 0; k < fs; k++)
    {
        const float wt = tab ? T.wt[k * kAaSide + threadIdx.y] : rsW(mn + k, oy, h, dh) * inv;
        const int iy = tab ? T.ix[k * kAaSide + threadIdx.y] : rsMirror(mn + k, h);
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

// the fold's pair in ONE kernel for a shrinking height, k_packOutTile's layout: both tap tables and
// k_restFoldH's sums of the source rows the block's vertical taps read in dynamic shared memory, then
// k_restFoldV's vertical sum and clamp into the planar target: the pair's bytes without the global tmp
__global__ void k_restFoldTile(const void* __restrict__ src, int half, int planeStride, int rowStride, int w, int h,
                               float* __restrict__ dst, int dw, int dh, int rcap)
{
    extern __shared__ float sm[];
    const int bx = blockDim.x, by = blockDim.y;
    const int nt = bx * by, t = threadIdx.y * bx + threadIdx.x;
    const int fsh = rsTaps(w, dw), fsv = rsTaps(h, dh);
    float* wtH = sm;
    int* ixH = (int*)(wtH + fsh * bx);
    float* wtV = (float*)(ixH + fsh * bx);
    int* ixV = (int*)(wtV + fsv * by);
    int* lim = ixV + fsv * by;
    float* tile = (float*)(lim + 2);
    if (t == 0) { lim[0] = h; lim[1] = -1; }
    aaTabFillS(wtH, ixH, bx, blockIdx.x * bx, bx, w, dw, fsh, dw - 1);
    aaTabFillS(wtV, ixV, by, blockIdx.y * by, by, h, dh, fsv, dh - 1);
    for (int e = t; e < by * fsv; e += nt)
    {
        atomicMin(&lim[0], ixV[e]);
        atomicMax(&lim[1], ixV[e]);
    }
    __syncthreads();
    const int r0 = lim[0], nr = lim[1] - lim[0] + 1;
    if (nr > rcap) return;
    for (int e = t; e < nr * bx; e += nt)
    {
        const int r = e / bx, c = e - r * bx;
        const size_t row = (size_t)(r0 + r) * rowStride;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
        for (int k = 0; k < fsh; k++)
        {
            const float wt = wtH[k * bx + c];
            const size_t o = row + ixH[k * bx + c];
            a0 += wt * restTap(src, half, o);
            a1 += wt * restTap(src, half, planeStride + o);
            a2 += wt * restTap(src, half, 2 * planeStride + o);
        }
        tile[r * bx + c] = a0;
        tile[(nr + r) * bx + c] = a1;
        tile[(2 * nr + r) * bx + c] = a2;
    }
    __syncthreads();
    const int x = blockIdx.x * bx + threadIdx.x, oy = blockIdx.y * by + threadIdx.y;
    if (x >= dw || oy >= dh) return;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
    for (int k = 0; k < fsv; k++)
    {
        const float wt = wtV[k * by + threadIdx.y];
        const int r = ixV[k * by + threadIdx.y] - r0;
        a0 += wt * tile[r * bx + threadIdx.x];
        a1 += wt * tile[(nr + r) * bx + threadIdx.x];
        a2 += wt * tile[(2 * nr + r) * bx + threadIdx.x];
    }
    const size_t plane = (size_t)dw * dh;
    const size_t o = (size_t)oy * dw + x;
    dst[o] = a0 < 0.0f ? 0.0f : (a0 > 1.0f ? 1.0f : a0);
    dst[plane + o] = a1 < 0.0f ? 0.0f : (a1 > 1.0f ? 1.0f : a1);
    dst[2 * plane + o] = a2 < 0.0f ? 0.0f : (a2 > 1.0f ? 1.0f : a2);
}

// the enlarging fold's source: the 4x output clamped to fp32 planar (then k_fitPlanar)
__global__ void k_restToF(const void* __restrict__ src, int half, int n, float* __restrict__ dst)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = restTap(src, half, (size_t)i);
}

// offline DLSS 5: dlssnr.py process() around the NR host's RGBA16F frame. In: the dw x dh
// picture of a planar (R, G, B) frame, clamp(0, 1).to(float16) (round to nearest even, torch's
// cast) plus alpha 1.0. Out: the first three channels, .float().clamp(0, 1), written back over
// the whole pw x ph padded frame with the edge replicated into the pad, as k_packInRaw8 / 16
// pad a decoded frame.
__global__ void k_nrIn(const float* __restrict__ src, int planeStride, int rowStride, int dw, int dh,
                       unsigned short* __restrict__ dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    const long long i = (long long)y * rowStride + x;
    unsigned short* d = dst + ((size_t)y * dw + x) * 4;
    for (int c = 0; c < 3; c++)
    {
        const float v = src[(long long)c * planeStride + i];
        d[c] = f2h(v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v));
    }
    d[3] = 0x3C00;
}
__global__ void k_nrOut(const unsigned short* __restrict__ src, int dw, int dh,
                        float* __restrict__ dst, int ph, int pw, int planeStride)
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= pw || oy >= ph) return;
    const int sx = ox < dw ? ox : dw - 1;
    const int sy = oy < dh ? oy : dh - 1;
    const unsigned short* p = src + ((size_t)sy * dw + sx) * 4;
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
// k_nrOut does.
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
                         float sdrWhite, unsigned short* __restrict__ dst, int hlg)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    const long long i = (long long)y * rowStride + x;
    float v[3];
    hdr2020_to_scrgb(src[i], src[(long long)planeStride + i], src[2LL * planeStride + i], hlg, v[0], v[1], v[2]);
    unsigned short* d = dst + ((size_t)y * dw + x) * 4;
    for (int c = 0; c < 3; c++)
    {
        const float s = v[c] / sdrWhite;
        d[c] = f2h(srgbOetf(s < 0.0f ? 0.0f : (s > 1.0f ? 1.0f : s)));
    }
    d[3] = 0x3C00;
}
__global__ void k_nrOutPq(const unsigned short* __restrict__ src, int dw, int dh, float sdrWhite,
                          float* __restrict__ dst, int ph, int pw, int planeStride, int hlg)
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= pw || oy >= ph) return;
    const long long o = (long long)oy * pw + ox;
    float in[3];
    hdr2020_to_scrgb(dst[o], dst[(long long)planeStride + o], dst[2LL * planeStride + o], hlg, in[0], in[1], in[2]);
    const int sx = ox < dw ? ox : dw - 1;
    const int sy = oy < dh ? oy : dh - 1;
    const unsigned short* p = src + ((size_t)sy * dw + sx) * 4;
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
// frames (px, two w x h planes, k_nvofUp's output) as R16G16_FLOAT texels, rows dstPitch bytes
// apart. A vector is kept only where it explains its 5x5 window of the current frame better than
// no motion by more than margin (the summed 8-bit luma error: the previous frame bilinear at the
// moved position against the same window unmoved, border clamped); an estimated field reports
// small vectors on still content and grain, and DLSS 5 would pull its history along them. Zero
// elsewhere: what a still pixel gets from a game engine. The moved tap is the pixel plus the
// floor and the exact fraction of the vector, never an fp32 absolute coordinate.
__global__ void k_nrMv(const float* __restrict__ flow, const unsigned char* __restrict__ cur,
                       const unsigned char* __restrict__ prev, int lumaPitch, int w, int h,
                       float margin, unsigned int* __restrict__ dst, int dstPitch)
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
        if (e1 + margin < e0) out = (unsigned int)f2h(u) | ((unsigned int)f2h(v) << 16);
    }
    dst[(size_t)y * (dstPitch >> 2) + x] = out;
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
__global__ void k_splatSoft(const void* __restrict__ in, int inHalf, int C,
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
        const float v = !act ? 0.0f : c < C ? __fmul_rn(gmTap(in, inHalf, (size_t)c * plane + o), e) : e;
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

// acc (C + 1 int64 planes of n) -> C planes of the target (fp32, or fp16 when dstHalf)
__global__ void k_splatNorm(const long long* __restrict__ acc, int C, int n,
                            void* __restrict__ dst, int dstHalf)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float norm = __fadd_rn((float)acc[(size_t)C * n + i] * (1.0f / 67108864.0f), 0.0000001f);
    for (int c = 0; c < C; c++)
        gmStore(dst, dstHalf, (size_t)c * n + i,
                __fdiv_rn((float)acc[(size_t)c * n + i] * (1.0f / 67108864.0f), norm));
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

// every thread of the block: the colour tile, then pass 1's edge and the luma at each position pass 2 reads
__device__ void asLoad(AsTile& s, const float* __restrict__ src, int dw, int dh)
{
    const int x0 = blockIdx.x * AS_T, y0 = blockIdx.y * AS_T;
    const int tid = threadIdx.y * AS_T + threadIdx.x;
    const size_t plane = (size_t)dw * dh;
    for (int i = tid; i < AS_C * AS_C; i += AS_T * AS_T)
    {
        const int ty = i / AS_C, tx = i - ty * AS_C;
        const int gx = min(max(x0 - 5 + tx, 0), dw - 1), gy = min(max(y0 - 5 + ty, 0), dh - 1);
        const size_t o = (size_t)gy * dw + gx;
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

// Adaptive Sharpen on a tight planar frame into another (dw x dh each): the sharpen after DLSS 5 at the
// working size, before RTX HDR and the model (post-processing before frame generation); 16 x 16 blocks
__global__ void k_sharpPlanar(const float* __restrict__ src, int dw, int dh, float curve, float* __restrict__ dst)
{
    __shared__ AsTile s;
    asLoad(s, src, dw, dh);
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

// steps 1 to 3 of _cap_to_pq2020: scRGB half4 to the SDR-encoded g, written twice: quantised
// BGRA8 for the TrueHDR bridge input, and unquantised planar fp32 for the colour step.
__global__ void k_sdrEncode(const unsigned short* __restrict__ src, int cw, int ch,
                            float sdrScale, unsigned char* __restrict__ dstBgra,
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
    unsigned char* d = dstBgra + o * 4;
    d[0] = (unsigned char)(int)rintf(g[2] * 255.0f);   // B
    d[1] = (unsigned char)(int)rintf(g[1] * 255.0f);   // G
    d[2] = (unsigned char)(int)rintf(g[0] * 255.0f);   // R
    d[3] = 255;
    dstG[o] = g[0];
    dstG[plane + o] = g[1];
    dstG[2 * plane + o] = g[2];
}

// the vivid / rtx colour step of one pixel (_source_at_thdr_luma + _ictcp_linear / _rtx_linear +
// _pack_out's nan_to_num): the bridge word u (10:10:10:2, B in the LOW 10 bits) and the
// unquantised SDR source s0..s2 (gamma BT.709, R, G, B) -> the corrected linear BT.2020 in
// [0,1]. mode 0 = vivid, 1 = rtx. k_thdrColor's colour step (both routes).
__device__ __forceinline__ void thdr_linear(unsigned int u, float s0, float s1, float s2,
                                            int mode, float vib, float sb, float* v)
{
    const float tr = (float)((u >> 20) & 1023u) * (1.0f / 1023.0f);
    const float tg = (float)((u >> 10) & 1023u) * (1.0f / 1023.0f);
    const float tb = (float)(u & 1023u) * (1.0f / 1023.0f);
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

// Unpack the bridge output (10:10:10:2, B in the LOW 10 bits), apply the
// colour mode against the unquantised source, and write planar PQ at capture resolution.
// mode 0 = vivid (_ictcp_linear), 1 = rtx (_rtx_linear), 2 = raw.
__global__ void k_thdrColor(const unsigned int* __restrict__ thdrOut,
                            const float* __restrict__ srcG, int cw, int ch,
                            float* __restrict__ dst, int mode, float vib, float sb)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cw || y >= ch) return;
    const size_t plane = (size_t)ch * cw;
    const size_t o = (size_t)y * cw + x;
    const unsigned int u = thdrOut[o];
    float p0, p1, p2;
    if (mode == 2)
    {
        p0 = (float)((u >> 20) & 1023u) * (1.0f / 1023.0f);
        p1 = (float)((u >> 10) & 1023u) * (1.0f / 1023.0f);
        p2 = (float)(u & 1023u) * (1.0f / 1023.0f);
    }
    else
    {
        float v[3];
        thdr_linear(u, srcG[o], srcG[plane + o], srcG[2 * plane + o], mode, vib, sb, v);
        p0 = pq_oetf(v[0]); p1 = pq_oetf(v[1]); p2 = pq_oetf(v[2]);
    }
    dst[o] = p0;
    dst[plane + o] = p1;
    dst[2 * plane + o] = p2;
}

// ---- offline RTX TrueHDR (rtxvideo.run_hdr on every output frame) -------------------------
// The output-size SDR frame after the pass chain -> the bridge's BGRA8 input (_eval_thdr: clamp,
// * 255, round) and the unquantised planar source the colour step reads (R, G, B, tight).
__device__ __forceinline__ void thdr_in_store(float r, float g, float b, size_t o, size_t plane,
                                              unsigned char* __restrict__ bgra, float* __restrict__ dstG)
{
    const float c[3] = { r, g, b };
    unsigned char q[3];
    for (int ci = 0; ci < 3; ci++)
    {
        const float v = c[ci] < 0.0f ? 0.0f : (c[ci] > 1.0f ? 1.0f : c[ci]);
        q[ci] = (unsigned char)(int)rintf(v * 255.0f);
    }
    unsigned char* d = bgra + o * 4;
    d[0] = q[2]; d[1] = q[1]; d[2] = q[0]; d[3] = 255;
    dstG[o] = r;
    dstG[plane + o] = g;
    dstG[2 * plane + o] = b;
}

// the frame straight from the model or the staging buffer (planar with its own strides)
__global__ void k_thdrIn(const void* __restrict__ src, int half, int planeStride, int rowStride, int dw, int dh,
                         unsigned char* __restrict__ bgra, float* __restrict__ dstG)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    const size_t i = (size_t)y * rowStride + x;
    thdr_in_store(ldS(src, half, i), ldS(src, half, (size_t)planeStride + i),
                  ldS(src, half, 2 * (size_t)planeStride + i),
                  (size_t)y * dw + x, (size_t)dw * dh, bgra, dstG);
}

// Adaptive Sharpen on the staging frame first (the last SDR pass before TrueHDR); 16 x 16 blocks
__global__ void k_sharpThdrIn(const float* __restrict__ src, int dw, int dh, float curve,
                              unsigned char* __restrict__ bgra, float* __restrict__ dstG)
{
    __shared__ AsTile s;
    asLoad(s, src, dw, dh);
    const int ox = blockIdx.x * AS_T + threadIdx.x, oy = blockIdx.y * AS_T + threadIdx.y;
    if (ox >= dw || oy >= dh) return;
    float c[3];
    asPixel(s, threadIdx.x, threadIdx.y, curve, c);
    thdr_in_store(c[0], c[1], c[2], (size_t)oy * dw + ox, (size_t)dw * dh, bgra, dstG);
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
// dpw x dph by replicating the edge, fp16 out (k_f2h's rounding). mode 0 = the codes as they are
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
__global__ void k_motionIn(const float* __restrict__ src, int sps, int srs, int sw, int sh,
                           unsigned short* __restrict__ dst, int dps, int drs, int dpw, int dph, int mode,
                           float sdrWhite, float knee, float cap)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dpw || y >= dph) return;
    const size_t s = (size_t)(y < sh ? y : sh - 1) * srs + (x < sw ? x : sw - 1);
    const size_t d = (size_t)y * drs + x;
    float v[3] = { src[s], src[(size_t)sps + s], src[2 * (size_t)sps + s] };
    motionEnc(v, mode, sdrWhite, knee, cap);
    dst[d] = f2h(v[0]);
    dst[(size_t)dps + d] = f2h(v[1]);
    dst[2 * (size_t)dps + d] = f2h(v[2]);
}
// live, the working size = the capture and nothing between the pack and the motion frame: k_packInDirect (hdr 0) or
// k_packInDirectHdr (hdr 1), k_motionIn (mode = hdr) on the values it packs and, with prev set, k_pairDiff of the
// planes against prev (its flag contract) in one pass; the motion frame shares the planes' pad (ph x pw, planeStride),
// every output bit for bit what the three kernels make. tex != 0: the capture is read from the capture array itself
// (a point-sampled texture object, element reads: SDR the four bytes as integers, HDR the four halves widened to fp32
// like h2f), not from src; raw (SDR) then gets each capture texel's four bytes, the copy the raw passthrough reads.
__global__ void k_packInMotion(const void* __restrict__ src, unsigned long long tex, unsigned int* __restrict__ raw,
                               int hdr, int cw, int ch, float* __restrict__ dst, unsigned short* __restrict__ mdst,
                               int ph, int pw, int planeStride, float sdrWhite, float knee, float cap,
                               const float* __restrict__ prev, int* __restrict__ flag)
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= pw || oy >= ph) return;
    const int sx = ox < cw ? ox : cw - 1;
    const int sy = oy < ch ? oy : ch - 1;
    const size_t s = ((size_t)sy * cw + sx) * 4;
    float v[3];
    if (tex)
    {
        const float fx = sx + 0.5f, fy = sy + 0.5f;
        if (hdr)
        {
            float r, g, b, a;
            asm("tex.2d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
                : "=f"(r), "=f"(g), "=f"(b), "=f"(a) : "l"(tex), "f"(fx), "f"(fy));
            scrgb_to_pq2020(r, g, b, v[0], v[1], v[2]);
        }
        else
        {
            unsigned int q[4];
            asm("tex.2d.v4.u32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
                : "=r"(q[0]), "=r"(q[1]), "=r"(q[2]), "=r"(q[3]) : "l"(tex), "f"(fx), "f"(fy));
            for (int c = 0; c < 3; c++)
                v[c] = q[2 - c] * (1.0f / 255.0f);
            if (raw && ox < cw && oy < ch)
                raw[(size_t)oy * cw + ox] = q[0] | q[1] << 8 | q[2] << 16 | q[3] << 24;
        }
    }
    else if (hdr)
    {
        const unsigned short* p = (const unsigned short*)src + s;
        scrgb_to_pq2020(h2f(p[0]), h2f(p[1]), h2f(p[2]), v[0], v[1], v[2]);
    }
    else
    {
        const unsigned char* p = (const unsigned char*)src + s;
        for (int c = 0; c < 3; c++)
            v[c] = p[2 - c] * (1.0f / 255.0f);
    }
    const size_t o = (size_t)oy * pw + ox;
    for (int c = 0; c < 3; c++)
        dst[(size_t)c * planeStride + o] = v[c];
    if (prev && !*flag &&
        (prev[o] != v[0] || prev[(size_t)planeStride + o] != v[1] || prev[2 * (size_t)planeStride + o] != v[2]))
        *flag = 1;
    motionEnc(v, hdr, sdrWhite, knee, cap);
    for (int c = 0; c < 3; c++)
        mdst[(size_t)c * planeStride + o] = f2h(v[c]);
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
__global__ void k_mvHalf(const float* __restrict__ flow, int w, int h, unsigned int* __restrict__ dst, int dstPitch)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t o = (size_t)y * w + x;
    dst[(size_t)y * (dstPitch >> 2) + x] = (unsigned int)f2h(flow[o]) | ((unsigned int)f2h(flow[(size_t)w * h + o]) << 16);
}
// the same texels from the engine's grid-1 field in one pass: vec = w x h short2 (raw / 32 = px, rows vecPitch bytes),
// k_nvofUp at grid 1 (weights 0 and 1, an exact copy) then k_mvHalf, byte for byte
__global__ void k_mvHalfG1(const short* __restrict__ vec, int vecPitch, int w, int h, unsigned int* __restrict__ dst,
                           int dstPitch)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const short* r = (const short*)((const char*)vec + (size_t)y * vecPitch);
    const float fx = (float)r[2 * x] * (1.0f / 32.0f), fy = (float)r[2 * x + 1] * (1.0f / 32.0f);
    dst[(size_t)y * (dstPitch >> 2) + x] = (unsigned int)f2h(fx) | ((unsigned int)f2h(fy) << 16);
}
// AMD FSR upscaling: RTX VSR's tight BGRA8 input frame as RGBA8 rows twice, the bridge's (aPitch bytes apart) and the
// Optical Flow engine's ABGR8 input (bPitch); and the bridge's RGBA8 result (srcPitch) back to tight BGRA8
__global__ void k_bgraRgbaFsr(const unsigned char* __restrict__ src, int w, int h, unsigned char* __restrict__ a,
                              int aPitch, unsigned char* __restrict__ b, int bPitch)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const unsigned int p = ((const unsigned int*)src)[(size_t)y * w + x];
    const unsigned int q = (p & 0xFF00FF00u) | ((p & 0xFFu) << 16) | ((p >> 16) & 0xFFu); // bytes 0 and 2 swapped
    *(unsigned int*)(a + (size_t)y * aPitch + 4 * (size_t)x) = q;
    *(unsigned int*)(b + (size_t)y * bPitch + 4 * (size_t)x) = q;
}
__global__ void k_rgbaBgra(const unsigned char* __restrict__ src, int srcPitch, int w, int h,
                           unsigned char* __restrict__ dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const unsigned int p = *(const unsigned int*)(src + (size_t)y * srcPitch + 4 * (size_t)x);
    ((unsigned int*)dst)[(size_t)y * w + x] = (p & 0xFF00FF00u) | ((p & 0xFFu) << 16) | ((p >> 16) & 0xFFu);
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
                            int mps, int mrs, int fw, int fh, unsigned short* __restrict__ out)
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
        const float a = rbLerp(pic, half, (size_t)((sw ? 3 : 0) + c) * pps, prs, T0);
        const float b = rbLerp(pic, half, (size_t)((sw ? 0 : 3) + c) * pps, prs, T1);
        O[(size_t)c * pps + o] = f2h(a * m + b * (1.0f - m));
    }
}
// k_rifeBlend with an fp32 store (outHalf false): the tween keeps the blend's fp32 value, so the reader that quantises
// it rounds once
__global__ void k_rifeBlendF(const void* __restrict__ pic, int sw, int half, int pps, int prs, int w, int h, int cw,
                             int ch, const unsigned short* __restrict__ flow, const unsigned short* __restrict__ mask,
                             int mps, int mrs, int fw, int fh, float* __restrict__ out)
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
        const float a = rbLerp(pic, half, (size_t)((sw ? 3 : 0) + c) * pps, prs, T0);
        const float b = rbLerp(pic, half, (size_t)((sw ? 0 : 3) + c) * pps, prs, T1);
        O[(size_t)c * pps + o] = a * m + b * (1.0f - m);
    }
}
// RIFE's last step fused with a 1:1 slot store (live, the model frame = the presented rect): the fp32
// blend of one tween clamped and quantised straight into the slot as k_packOut (hdr 0: BGRA8) and
// k_packOutHdr (hdr 1: R10G10B10A2) store it, one rounding and no tween buffer
__global__ void k_rifeBlendOut(const void* __restrict__ pic, int sw, int half, int pps, int prs, int w, int h,
                               int cw, int ch, const unsigned short* __restrict__ flow,
                               const unsigned short* __restrict__ mask, int mps, int mrs, int fw, int fh, int hdr,
                               unsigned char* __restrict__ dst, int pitch, int x0, int y0)
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
        const float a = rbLerp(pic, half, (size_t)((sw ? 3 : 0) + c) * pps, prs, T0);
        const float b = rbLerp(pic, half, (size_t)((sw ? 0 : 3) + c) * pps, prs, T1);
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
                                 int x0, int y0)
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
        const float a = rbLerp(pic, 0, (size_t)((sw ? 3 : 0) + c) * pps, prs, T0);
        const float b = rbLerp(pic, 0, (size_t)((sw ? 0 : 3) + c) * pps, prs, T1);
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
// bytes; out: flow (2 planes of w x h, px) and cost (1 plane, 0..255 as float)
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
extern "C" __global__ void k_splatVel(const float* __restrict__ flow, const float* __restrict__ Z,
                                      float s, float bias, float sign, int covPlane, int w, int h,
                                      long long* __restrict__ acc)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t plane = (size_t)w * h, o = (size_t)y * w + x;
    int nwX, nwY;
    float dx, dy;
    if (!land(x, s, flow[o], nwX, dx) || !land(y, s, flow[plane + o], nwY, dy)) return;
    const float e = expf(Z[o] + bias);
    const float vx = sign * flow[o], vy = sign * flow[plane + o];
    const float wt[4] = { (1.0f - dx) * (1.0f - dy), dx * (1.0f - dy), (1.0f - dx) * dy, dx * dy };
    const int ox[4] = { 0, 1, 0, 1 }, oy[4] = { 0, 0, 1, 1 };
    for (int k = 0; k < 4; k++)
    {
        const int tx = nwX + ox[k], ty = nwY + oy[k];
        if (tx < 0 || tx >= w || ty < 0 || ty >= h) continue;
        const size_t t = (size_t)ty * w + tx;
        const float we = e * wt[k];
        atomicAdd((unsigned long long*)acc + t, (unsigned long long)llrintf(vx * we * NV_FIX));
        atomicAdd((unsigned long long*)acc + plane + t, (unsigned long long)llrintf(vy * we * NV_FIX));
        atomicAdd((unsigned long long*)acc + 2 * plane + t, (unsigned long long)llrintf(we * NV_FIX));
        atomicAdd((unsigned long long*)acc + (size_t)covPlane * plane + t, (unsigned long long)llrintf(wt[k] * NV_FIX));
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
// pixel, then every pixel accumulates in fp32. A radius above kBlurMaxR keeps the per-pixel
// form.
#define kBlurMaxR 64
extern "C" __global__ void k_blur1(const float* __restrict__ src, int w, int h, float sigma, int dir,
                                   float* __restrict__ dst)
{
    __shared__ float wtab[2 * kBlurMaxR + 1];
    __shared__ float wsum;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int r = (int)(4.0f * sigma + 0.5f);
    const bool tab = r <= kBlurMaxR;
    const float g = -0.5f / (sigma * sigma);
    if (tab)
    {
        const int tid = threadIdx.y * blockDim.x + threadIdx.x, nt = blockDim.x * blockDim.y;
        for (int k = tid; k <= 2 * r; k += nt)
            wtab[k] = expf(g * (float)((k - r) * (k - r)));
        __syncthreads();
        if (tid == 0)
        {
            float ws = 0.0f;
            for (int k = 0; k <= 2 * r; k++) ws += wtab[k];
            wsum = ws;
        }
        __syncthreads();
    }
    if (x >= w || y >= h) return;
    const int n = dir ? h : w;
    const int c = dir ? y : x;
    float acc = 0.0f, ws = 0.0f;
    for (int k = -r; k <= r; k++)
    {
        int i = c + k;
        // reflect about the edge sample boundary, repeated for kernels wider than the axis
        while (i < 0 || i >= n) i = i < 0 ? -i - 1 : 2 * n - i - 1;
        const float wk = tab ? wtab[k + r] : expf(g * (float)(k * k));
        acc += wk * (dir ? src[(size_t)i * w + x] : src[(size_t)y * w + i]);
        if (!tab) ws += wk;
    }
    dst[(size_t)y * w + x] = acc / (tab ? wsum : ws);
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
//   rife_backend.RIFE.calc_flow's tail: block0's flow (4 planes, 0->0.5 | 1->0.5) splatted by
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

// the value(s) v of pixel (x, y) splatted at (x + ax * m, y + ay * m) (land) into nv value planes +
// the weight plane
__device__ __forceinline__ void drbaSplat4(unsigned long long* p, size_t plane, int nv,
                                           const float* v, int x, int y, float ax, float ay, float m,
                                           int w, int h)
{
    int nwX, nwY;
    float dx, dy;
    if (!land(x, ax, m, nwX, dx) || !land(y, ay, m, nwY, dy)) return;
    const float wt[4] = {(1.0f - dx) * (1.0f - dy), dx * (1.0f - dy), (1.0f - dx) * dy, dx * dy};
    const int cx[4] = {nwX, nwX + 1, nwX, nwX + 1}, cy[4] = {nwY, nwY, nwY + 1, nwY + 1};
    for (int k = 0; k < 4; k++)
    {
        if (cx[k] < 0 || cx[k] >= w || cy[k] < 0 || cy[k] >= h) continue;
        const size_t o = (size_t)cy[k] * w + cx[k];
        for (int c = 0; c < nv; c++)
            atomicAdd(p + (size_t)c * plane + o,
                      (unsigned long long)(long long)llrintf(v[c] * wt[k] * DRBA_FIX));
        atomicAdd(p + (size_t)nv * plane + o, (unsigned long long)(long long)llrintf(wt[k] * DRBA_FIX));
    }
}

// flow (4 planes) -> acc (6 int64 planes, zeroed by the caller): half j = planes 3j, 3j+1 (x, y) + 3j+2 (weight)
extern "C" __global__ void k_drbaFlowSplat(const float* __restrict__ flow, int w, int h,
                                           long long* __restrict__ acc)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t plane = (size_t)w * h, o = (size_t)y * w + x;
    for (int j = 0; j < 2; j++)
    {
        const float v[2] = {flow[(2 * j) * plane + o], flow[(2 * j + 1) * plane + o]};
        drbaSplat4((unsigned long long*)(acc + (size_t)(3 * j) * plane), plane, 2, v,
                   x, y, v[0], v[1], 1.0f, w, h);
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

// f10, f12 (2 planes each, calc_flow's flow05 * 2) -> acc (2 int64 planes, zeroed): drm, weight
extern "C" __global__ void k_drbaDrmSplat(const float* __restrict__ f10, const float* __restrict__ f12,
                                          int side, float tt, int w, int h, long long* __restrict__ acc)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t plane = (size_t)w * h, o = (size_t)y * w + x;
    const float d = drbaDrmUn(f10, f12, plane, o, side, tt);
    const float* fa = side < 0 ? f10 : f12;
    const float v[1] = {d};
    drbaSplat4((unsigned long long*)acc, plane, 1, v, x, y, fa[o], fa[plane + o], d, w, h);
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

// GMFlow's local correlation (gmflow_bidir's 1/4-scale window around the current flow, between its two engines):
// out[b, p, k] = the dot of q[b, p, :] with f bilinearly sampled at the window position g[b, p, k] (GridSample's
// rules: align_corners, zero padding). The window's samples share one fractional offset, so their taps lie on a small
// integer grid: one warp a pixel takes the grid's dot products once (float4 lanes over the 128 channels, kept in
// shared memory, one cell of margin for rounding), then each sample combines its 4 corners with its own weights.
// Never builds the [B, 128, HW, K] product the engine wrote. q [B, HW, 128] and f [B, H, W, 128] channels-last,
// g [B, HW, K, 2] normalized; 128 threads a block, one warp a pixel.
extern "C" __global__ void k_localCorr(const float* __restrict__ q, const float* __restrict__ f,
                                       const float* __restrict__ g, float* __restrict__ out, int B, int H, int W,
                                       int HW, int K)
{
    __shared__ float dots[4][144];
    const int wib = threadIdx.x >> 5;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= B * HW) return;
    const int b = warp / HW, p = warp % HW;
    const float4 qv = reinterpret_cast<const float4*>(q + ((size_t)b * HW + p) * 128)[lane];
    const float* fb = f + (size_t)b * H * W * 128;
    const float* gp = g + ((size_t)b * HW + p) * K * 2;
    float mnx = 1e30f, mny = 1e30f;
    for (int k = lane; k < K; k += 32)
    {
        mnx = fminf(mnx, floorf((gp[2 * k] + 1.f) * 0.5f * (float)(W - 1)));
        mny = fminf(mny, floorf((gp[2 * k + 1] + 1.f) * 0.5f * (float)(H - 1)));
    }
    for (int o = 16; o; o >>= 1)
    {
        mnx = fminf(mnx, __shfl_xor_sync(0xffffffffu, mnx, o));
        mny = fminf(mny, __shfl_xor_sync(0xffffffffu, mny, o));
    }
    const int ox = (int)mnx, oy = (int)mny;
    int side = 1;
    while (side * side < K) side++;
    const int gs = side + 2;
    for (int t = 0; t < gs * gs; t++)
    {
        const int xx = ox + t % gs, yy = oy + t / gs;
        float s = 0.f;
        if (xx >= 0 && xx < W && yy >= 0 && yy < H)
        {
            const float4 v = reinterpret_cast<const float4*>(fb + ((size_t)yy * W + xx) * 128)[lane];
            s = qv.x * v.x + qv.y * v.y + qv.z * v.z + qv.w * v.w;
        }
        for (int o = 16; o; o >>= 1)
            s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) dots[wib][t] = s;
    }
    __syncwarp();
    for (int k = lane; k < K; k += 32)
    {
        const float x = (gp[2 * k] + 1.f) * 0.5f * (float)(W - 1);
        const float y = (gp[2 * k + 1] + 1.f) * 0.5f * (float)(H - 1);
        const float x0f = floorf(x), y0f = floorf(y);
        const float ax = x - x0f, ay = y - y0f;
        const int cx = (int)x0f - ox, cy = (int)y0f - oy;
        const float* d = dots[wib] + cy * gs + cx;
        out[((size_t)b * HW + p) * K + k] = (1.f - ay) * ((1.f - ax) * d[0] + ax * d[1]) +
                                            ay * ((1.f - ax) * d[gs] + ax * d[gs + 1]);
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
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
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
// k_attn2b: k_attn2 with the key tiles double-buffered through cp.async (the next tile loads while this one computes)
// and the key fragments read by ldmatrix.x4 (four 8 x 8 matrices a load: b0 / b1 of two k-steps); bit-identical to
// k_attn2, faster at small N only (the host's kAttn2bMaxN)
__device__ __forceinline__ unsigned attnSmem(const void* p)
{
    unsigned a;
    asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(a) : "l"(p));
    return a;
}
// one key tile (rows j0 .. j0 + 63) and its values into a stage: a 16-byte cp.async per (row, 16-byte column), rows
// past n zero-filled; the values as k_attn2 reads them
__device__ __forceinline__ void attnLoad(unsigned* sk, float* sv, const unsigned* kb, const float* vb, int j0, int n,
                                         int mode, int gw)
{
    for (int e = threadIdx.x; e < AT_KT * (AT_W / 4); e += 128)
    {
        const int row = e / (AT_W / 4), c4 = e % (AT_W / 4);
        const unsigned dst = attnSmem(sk + row * AT_SP + c4 * 4);
        const unsigned* src = kb + (size_t)(j0 + (j0 + row < n ? row : 0)) * AT_W + c4 * 4;
        const int bytes = j0 + row < n ? 16 : 0;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(bytes));
    }
    if (threadIdx.x < AT_KT * 2)
    {
        const int j = j0 + (int)(threadIdx.x >> 1), c = threadIdx.x & 1;
        sv[threadIdx.x] = j >= n ? 0.f : (mode & 1) ? (float)(c ? j / gw : j % gw) : vb[(size_t)j * 2 + c];
    }
    asm volatile("cp.async.commit_group;");
}
extern "C" __global__ void __launch_bounds__(128) k_attn2b(const unsigned* __restrict__ q,
                                                           const unsigned* __restrict__ k,
                                                           const float* __restrict__ v, void* __restrict__ o, int n,
                                                           int kswap, float scale2, int mode, int gw)
{
    __shared__ __align__(16) unsigned sk[2][AT_KT * AT_SP];
    __shared__ float sv[2][AT_KT * 2];
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
    // ldmatrix: lane L addresses row L & 7 of matrix L >> 3 = the half columns s * 16 + (L >> 3) * 8 of key t * 8 +
    // (L & 7); the four matrices = b0 / b1 of k-step s and of s + 1
    const int lrow = lane & 7, lcol = (lane >> 3) * 4;
    float m0 = ninf, m1 = ninf, l0 = 0.f, l1 = 0.f, x0 = 0.f, y0 = 0.f, x1 = 0.f, y1 = 0.f;
    attnLoad(sk[0], sv[0], kb, vb, 0, n, mode, gw);
    int st = 0;
    for (int j0 = 0; j0 < n; j0 += AT_KT, st ^= 1)
    {
        if (j0 + AT_KT < n)
        {
            attnLoad(sk[st ^ 1], sv[st ^ 1], kb, vb, j0 + AT_KT, n, mode, gw);
            asm volatile("cp.async.wait_group 1;");
        }
        else
            asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        float acc[8][4];
#pragma unroll
        for (int t = 0; t < 8; ++t)
        {
            acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
            const unsigned base = attnSmem(sk[st] + (t * 8 + lrow) * AT_SP + lcol);
#pragma unroll
            for (int s = 0; s < 8; s += 2)
            {
                unsigned f0, f1, f2, f3;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(f0), "=r"(f1), "=r"(f2), "=r"(f3)
                             : "r"(base + s * 32));
                attnMma(acc[t], qa[s], f0, f1);
                attnMma(acc[t], qa[s + 1], f2, f3);
            }
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
        const float* svs = sv[st];
#pragma unroll
        for (int t = 0; t < 8; ++t)
#pragma unroll
            for (int c = 0; c < 2; ++c)
            {
                const int key = t * 8 + t4 * 2 + c;
                const float vx = svs[key * 2], vy = svs[key * 2 + 1];
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
        __syncthreads();
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

// ---- live GMFSS at a flow scale below 1 (--gmfss-flow, GMFSS_infer_u.reuse's scale branch) ----------------------
// Both halves shrink to the /32 flow grid the way F.interpolate(size=(fh, fw), mode='bilinear', align_corners=False,
// antialias=True) does (torch's _compute_weights_aa: the triangle filter over a window of support = the ratio,
// normalised by the tap sum), gmflow runs there, and k_gmFlowUp brings its two flows back to the half the way
// F.interpolate(size=(hh, hw), mode='bilinear', align_corners=False) * (hw / fw, hh / fh) does (channel 0 = x). The
// axes only shrink (in >= out). No fp64: a tap's distance from the centre is the integer ratio
// ((2j + 1) out - (2o + 1) in) / (2 in), one rounding; the window ends are exact integer floors. Both kernels are gated
// against an fp64 reference and against torch.
__device__ __forceinline__ void gmAaAxis(int o, int in, int out, int& mn, int& mx)
{
    // torch: centre = in / out * (o + 0.5), support = in / out; min = max(int(centre - support + 0.5), 0), max =
    // min(int(centre + support + 0.5), in)
    const long long den = 2LL * out, lo = (2LL * o - 1) * in + out, hi = (2LL * o + 3) * in + out;
    mn = lo > 0 ? (int)(lo / den) : 0;
    mx = (int)(hi / den);
    if (mx > in) mx = in;
}
__device__ __forceinline__ float gmAaW(int j, int o, int in, int out)
{
    const float a = fabsf((float)((2LL * j + 1) * out - (2LL * o + 1) * in) / (float)(2LL * in));
    return a < 1.0f ? 1.0f - a : 0.0f;
}

// C contiguous fp32 planes of (h, w) -> C contiguous planes of (dh, dw): the horizontal taps of each window row, then
// the vertical sum, over the product of the two axes' tap sums
__global__ void k_gmShrinkAa(const float* __restrict__ src, int C, int w, int h, float* __restrict__ dst, int dw, int dh)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    int x0, x1, y0, y1;
    gmAaAxis(x, w, dw, x0, x1);
    gmAaAxis(y, h, dh, y0, y1);
    float sx = 0.0f, sy = 0.0f;
    for (int i = x0; i < x1; i++) sx += gmAaW(i, x, w, dw);
    for (int j = y0; j < y1; j++) sy += gmAaW(j, y, h, dh);
    const float norm = sx > 0.0f && sy > 0.0f ? 1.0f / (sx * sy) : 0.0f;
    const size_t splane = (size_t)w * h, dplane = (size_t)dw * dh, o = (size_t)y * dw + x;
    for (int c = 0; c < C; c++)
    {
        const float* s = src + (size_t)c * splane;
        float acc = 0.0f;
        for (int j = y0; j < y1; j++)
        {
            const float* r = s + (size_t)j * w;
            float row = 0.0f;
            for (int i = x0; i < x1; i++) row += gmAaW(i, x, w, dw) * r[i];
            acc += gmAaW(j, y, h, dh) * row;
        }
        dst[(size_t)c * dplane + o] = acc * norm;
    }
}

// the flow grid's planes (sh, sw) -> (dh, dw): bilinear at rbAxis's exact source coordinate, plane p times dw / sw when
// p is even (x) and dh / sh when it is odd (y)
__global__ void k_gmFlowUp(const float* __restrict__ src, int planes, int sw, int sh, float* __restrict__ dst, int dw,
                           int dh)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    int x0, x1, y0, y1;
    float lx, ly;
    rbAxis(x, dw, sw, x0, x1, lx);
    rbAxis(y, dh, sh, y0, y1, ly);
    const float vx = (float)dw / (float)sw, vy = (float)dh / (float)sh;
    const size_t splane = (size_t)sw * sh, dplane = (size_t)dw * dh;
    const size_t r0 = (size_t)y0 * sw, r1 = (size_t)y1 * sw, o = (size_t)y * dw + x;
    for (int p = 0; p < planes; p++)
    {
        const float* s = src + (size_t)p * splane;
        const float a = (1.0f - lx) * s[r0 + x0] + lx * s[r0 + x1];
        const float b = (1.0f - lx) * s[r1 + x0] + lx * s[r1 + x1];
        dst[(size_t)p * dplane + o] = ((1.0f - ly) * a + ly * b) * ((p & 1) ? vy : vx);
    }
}

}

)CUDASRC";

// the Lanczos radii of the host's resizes, read once from the kernel text's #defines (the one
// place they are set), and the kernels' rsTaps for the host's launch sizes
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
// nvrtc ships no import library in the runtime wheel, so it is resolved by hand; cudart, the
// driver API and TRT-RTX are delay-loaded imports whose DLLs are pre-loaded by full path in
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

static std::wstring g_nativeRuntimeDir; // engine\gpu_runtime: the CUDA 13 + TensorRT-RTX DLLs
static std::wstring g_nativeEngineDir;  // ...\engine itself, the parent of the folders above

// ---- the RTX Video SDK CUDA bridge (engine\rtxvideo\rtxvideo_cuda.dll) --------------------
// The TrueHDR bridge is NOT an NGX D3D11 bridge: it is a plain C ABI over CUDA device
// pointers, the same DLL and the same entry points rtxvideo.py drives. The NGX feature DLLs
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
}; // 0 bicubic .. 4 Ultra; the host always asks for 4 (vsrSet)
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
    std::wstring dir;
};
static RtxBridge g_rtxb;

// Resolve the bridge folder exactly as rtxvideo.py does (SMV_RTXVIDEO_DIR, else
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

// Offline TrueHDR's light statistics, python's RTXVideo _cll / _fall / _l1
// / _hp (rtxvideo.py _pack_out, _measure_light, _accum_l1, _accum_hp), accumulated per output
// frame from k_pqOut's block (hist[1024] u32, misc[4] u32, vSum int64 2^24 fixed point) and written as one
// JSON file render.py reads at the finalize: maxcll / maxfall always, the Dolby Vision L1
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
    float lut[1024]{};  // _pq_lut (k_pqLut, the kernel's own PQ EOTF)
    int brightCode = 0; // _hp_bright_code: the first code brighter than 100 nits
    double cll = 0.0, fall = 0.0;
    bool raw = false; // --hdr-color raw: the stats read nits, not linear
    bool wantL1 = false, wantHp = false;
    std::vector<ThdrL1> l1;
    std::vector<ThdrHp> hp;
    // --hdr-frames: one line per output frame, appended and flushed as the
    // record lands, "<frame MaxCLL> <frame MaxFALL>[ L <l1 x3>][ H <avg> <maxscl x3> <dist x9>]":
    // a killed render keeps every record of the frames it wrote, and a resumed render reads the
    // banked prefix back from it instead of decoding the banked video (python's _rebuild_hdr_stats)
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
        // _pack_out: float(mx.max()) * 10000 and float(mx.mean()) * 10000 (the mean is fp32 in
        // torch); _measure_light (raw) reads nits already
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

static bool g_onnxParserOk = false; // tensorrt_onnxparser_rtx_1_6.dll loaded (nativeLoadDlls)

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
    const std::wstring trtBin = g_nativeRuntimeDir;
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
        {trtBin.c_str(), L"tensorrt_rtx_1_6.dll", true},
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
    // the ONNX parser: optional, without it a cold size cannot build
    g_onnxParserOk = LoadLibraryExW((trtBin + L"\\tensorrt_onnxparser_rtx_1_6.dll").c_str(), nullptr,
                                    LOAD_WITH_ALTERED_SEARCH_PATH) != nullptr;
    if (!g_onnxParserOk)
        LOG("native: ONNX parser not loadable, a window size without warm engines cannot start\n");
    // AddDllDirectory for anything either DLL loads later by name; harmless when the process
    // never opts into the safe search order.
    AddDllDirectory(cudaBin.c_str());
    AddDllDirectory(trtBin.c_str());
    state = 1;
    return true;
}

class NativeTrtLogger : public nvinfer1::ILogger
{
  public:
    void log(Severity s, const char* msg) noexcept override
    {
        if (s <= Severity::kWARNING)
            LOG("[trt-native] %s\n", msg);
    }
};
static NativeTrtLogger g_nativeTrtLogger;

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

// the GMFSS engine slots: feat_ext, gmflow_bidir's first half (A), metricnet, the GMFSS IFNet, fusionnet, gmflow_bidir's
// second half (B), gmflow's backbone (one frame), then A's front (`_m`: up to the scale-0 global matching); the host
// computes the matching and its propagation between `_m` and A (k_attn2) and gmflow's local correlation between A and B,
// and hands `_m` and A each frame's backbone maps once
constexpr int kGmN = 8;
// A's outputs in this order (trt_runtime.gmflow_split's contract): q and f feed k_localCorr with coords, the rest
// go to B beside the correlation
static const char* const kGmA[8] = {"q", "f", "coords", "win", "valid", "flow0", "feat", "flow1"};

struct NativeRife
{
    // ---- handoff facts, straight off live_server.py's --native-handoff lines
    std::string ifnetPath, encodePath, jitPath, ejitPath;
    int ph = 0, pw = 0, w = 0, h = 0, cw = 0, ch = 0;
    int dw = 0, dh = 0, x0 = 0, y0 = 0;
    int batchMax = 1;
    bool identity = false;
    bool hdr = false; // FP16 scRGB capture in, PQ R10A2 ring out
    int dev = 0;      // the CUDA device of the capture adapter (bound per thread)
    // live RTX TrueHDR through the CUDA bridge, once per real frame at capture size
    bool rtxHdr = false;    // the session asked for it and the bridge loaded
    bool rtxFailed = false; // a mid-run eval failed: faithful PQ for the rest of the run
    RtxThdrSetting thdr{};  // Contrast, Saturation, MiddleGray 50, MaxLuminance 1000
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
    bool vsrAsync = false;    // the evals ran through the bridge's event-ordered export (vsrMs not timed)
    bool fitUpLogged = false; // the fit's enlarge-in-one-kernel line was logged
    uint64_t vsrN = 0;
    // the shrinking fit: the Lanczos pair (k_fitAaH / V) into the staging frame
    bool fitAa = false; // the fit to (dw, dh) changes the size: the separable Lanczos pair
    // Upscale to: the app's --upscale H = the
    // INTERNAL render size (uw, uh), derived from the exe's own flag. The
    // model frame goes there first (RTX VSR when it enlarges, else Lanczos), then the fit to
    // (dw, dh) like any frame. 0 = off (also when it equals the
    // fit rect: one resize, not two).
    int uw = 0, uh = 0;
    bool upAa = false;   // the first resize shrinks the height (uh < h)
    bool upTile = false; // the first resize runs as k_fitAaTile (decided with dUpTmp)
    // live Restore: the handoff builds the
    // Real-ESRGAN TensorRT engine into the shared cache and hands its path over like the
    // IFNet's; the host runs it on every presented frame before the upscale, then folds the 4x
    // output to the first resize target (_Fit._load_restore's restore_target rule: back to the
    // model size when RTX VSR follows, else straight to the internal render size or the fit
    // rect). A mid-run failure drops the pass for the rest of the session with one line.
    std::string restorePath, rjitPath;
    bool restore = false; // the handoff named an engine (the session asked for it)
    // no-engine mode: the effects-only route (the
    // app's `echo` backend, no model ticked) runs here too. The handoff answers with the
    // geometry only (`engine=none`, plus the restore engine when asked for): no IFNet, no
    // encode, no tween; every group stores its one real frame through the same effects
    // chain, exactly live_server.py's Identity.process_shm. The cubin cache folder comes
    // from `NATIVE-PATH cache=` since no jit path names it.
    bool noEngine = false;
    std::string cachePath;
    // GMFSS: the handoff answers
    // `engine=gmfss` with the six engines of the set (feat_ext at the padded size, the fused
    // bidir GMFlow's two halves, metricnet, the GMFSS IFNet and fusionnet at the half) and their
    // jit caches; this host loads, warms and keeps the set resident (one model set at a time)
    // and runs the chain itself (nativeGmfssPair / nativeGmfssTween with the glue kernels
    // above), so gmfss is a native backend like rife.
    bool gmfss = false;
    int hh = 0, hw = 0;   // the half frame (the fusion grid)
    int gfh = 0, gfw = 0; // live --gmfss-flow below 1: gmflow's /32 grid inside the half (0 = the half itself)
    std::string gmPath[kGmN], gmJit[kGmN];
    nvinfer1::ICudaEngine* engGm[kGmN] = {};
    nvinfer1::IRuntimeConfig* cfgGm[kGmN] = {};
    nvinfer1::IExecutionContext* ctxGm[kGmN] = {};
    // between gmflow's halves: A's outputs (kGmA order), the correlation k_localCorr writes for B, and the shapes it
    // runs at (the quarter grid gmH x gmW, gmK window samples a pixel)
    void* dGmA[8] = {};
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
    bool gmAttnB = false; // the matching and propagation run k_attn2b (N <= kAttn2bMaxN)
    // the chain's contract, read off the engines in nativeGmfssSetup (never from the handoff
    // line, the 5a rule), and the planar NCHW buffers it sizes from that. Levels: the half
    // (hh, hw), the quarter and the eighth, exactly GMFSS_infer_u's feature pyramid.
    int gmC[3] = {}; // feat_ext channels at the half / quarter / eighth
    bool gmFeatHalf = true, gmMetricHalf = true, gmOutHalf = true, gmFuseHalf = true;
    int gmCur = 0;               // which dGmFeat set holds the CURRENT frame's features
    void* dGmFeat[2][3] = {};    // the two frames' feature sets (cur is the next pair's prev)
    float* dGmHalf = nullptr;    // (6, hh, hw): img0's half in planes 0..2, img1's in 3..5
    float* dGmFlow = nullptr;    // the gmflow output (2, 2, hh, hw) = flow01 then flow10
    float* dGmHalfF = nullptr;   // at a flow grid (gfh): the six planes gmflow reads, shrunk to (gfh, gfw)
    float* dGmFlowF = nullptr;   // at a flow grid: gmflow's output (2, 2, gfh, gfw) before k_gmFlowUp
    void* dGmMetric = nullptr;   // (2, hh, hw) in the engine's dtype: m0 then m1
    float* dGmFlowP[2] = {};     // the flow pyramids at the quarter / eighth, pre-scaled
    float* dGmMetP[2] = {};      // the metric pyramids at the quarter / eighth (fp32)
    long long* dGmAcc = nullptr; // the splat accumulator, max(C) + 1 planes of the half
    float* dGmFa = nullptr;      // fusionnet a (9, half): I1t, merged (ifnet writes it), I2t
    void* dGmFb = nullptr;       // fusionnet b (2 * gmC[0], half) in the engine's dtype
    void* dGmFc = nullptr;       // fusionnet c (2 * gmC[1], quarter), the same dtype
    void* dGmFd = nullptr;       // fusionnet d (2 * gmC[2], eighth), the same dtype
    float* dGmT = nullptr;       // the timestep, one device float (ifnet's 1x1x1x1 input)
    void* dGmOut = nullptr;      // the fusionnet output (3, ph, pw) in the engine's dtype
    float* dGmF = nullptr;       // that output clamped to fp32: what storeSlot consumes
    // HDR planes (nativeGmfssPair): the SDR view of one padded frame (hdrToModel, the scratch), the motion view's
    // halves gmflow and metricnet read on a pair outside the SDR range, the range flag (device, pinned host copy);
    // per feature slot: its frame stays inside the SDR range, its features came from the SDR view; the pair's mode
    float* dGmEnc = nullptr;
    float* dGmHalfM = nullptr;
    int* dGmRange = nullptr;
    int* hGmRange = nullptr;
    bool gmSdrPairs = true; // SMV_GMFSS_SDR_PAIRS=0: every pair the way outside the SDR range
    bool gmFrameSdr[2] = {}, gmFeatSdr[2] = {}, gmPairSdr = false;
    int gmModeLogs = 0; // the mode lines this session printed (the first pair's and up to 3 switches)
    // SMV_LIVE_GMFSS_PROF=1: per-phase CUDA-event breakdown of the chain, printed every 32
    // groups (the native answer to the python route's [timing] line; nine spans, the tween ones
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
    // this host drives that bridge's flat C API exactly as live_server.py's Fruc class does
    // through ctypes: the (R, G, B) model planes packed to true BGRA8 at pw x ph (k_packBgraRgb,
    // the VSR kernel), one bridge call per tween, the BGRA8 result unpacked (k_unpackBgraRgb).
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
    int frFed[4] = {-1, -1, -1, -1};
    bool frFirst[4] = {};
    uint64_t frCalls[4] = {};
    bool frCreated = false;
    uint64_t frPrimed = 0, frRepeats = 0, frTweens = 0;
    // parallel instances (the direct-t scheme, bridge nvoffruc_step_i): tween k of a pair's plan
    // runs on instance k % frInst, a round of frInst tweens at a time; instance 0 on the compute
    // thread, the others on their own workers (g_frW), each into its own BGRA8 output
    int frInst = 1, frInstMax = 1;
    uint8_t* dFrOutBI[4] = {}; // [0] = dFrOutB
    double frPlan[64] = {};
    uint32_t frPlanN = 0, frPlanK = 0;
    // recursive midpoints (the default): NvOFFRUC damages content that does not move at every
    // t != 0.5, even called exactly as NVIDIA documents it, so every tween is node k / 2^L of the
    // pair's midpoint tree:
    // FRUC at t = 0.5 between the node's two parents, level L on instance L - 1, which then sees
    // one continuous stream of new frames (NVIDIA's one call per new frame). A t that is no node
    // takes the nearest node of depth frMpCap. SMV_FRUC_MIDPOINTS=0 = the direct-t scheme above.
    bool frMp = false;
    int frMpCap = 2;           // off-node snap depth: offline 3, live 2 (cost); SMV_FRUC_DEPTH=1..4 sets both
    int frMpL = 0;             // this pair's depth, 0 = not chosen yet
    bool frMpBuilt = false;    // this pair's whole tree to frMpL is computed
    uint64_t frSerial = 0;     // frames packed this session: the node keys
    uint8_t* dFrNode[16] = {}; // BGRA8 pw x ph: node j / 16 of this pair (j = 1..15)
    bool frNodeOk[16] = {};
    uint64_t frLastKey[4] = {}; // per instance: the key of the frame it was fed last, 0 = none
    uint64_t frNodeCalls = 0;
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
    // never fed (its first call resets).
    bool fsrfg = false;
    struct FgInst
    {
        cudaExternalMemory_t emIn = nullptr, emMv = nullptr, emOut = nullptr;
        cudaMipmappedArray_t mmIn = nullptr, mmMv = nullptr, mmOut = nullptr;
        cudaArray_t in = nullptr, mv = nullptr, out = nullptr; // the three textures' level 0
        cudaExternalSemaphore_t sem = nullptr;
        uint64_t val = 0; // the fence's last value
        bool made = false;
    };
    FgInst fg[4];
    LUID fgLuid = {};
    int fgPitch = 0, fgMvPitch = 0; // the bridge's row pitches (bytes), the same for every instance
    uint8_t* dFgMv = nullptr;       // a call's vectors (RG16F rows fgMvPitch apart) before their copy into its texture
    // SMV_FSRFG_PROF=1: CUDA events between a call's stages (copy in, the Optical Flow inputs, the Optical Flow, the
    // vector kernels, the vector copy, FSR on vkd3d, the copy out), means at the session's end; every call then waits
    // for its last event, so the stages do not overlap the next call's
    bool fgProf = false;
    cudaEvent_t fgEv[8] = {};
    double fgStageMs[7] = {}, fgCpuMs = 0.0;
    uint64_t fgProfN = 0, fgProfVecN = 0;
    // FSR's motion vectors (frame generation and upscaling): an Optical Flow session at w x h, ABGR8 (R, G, B, A
    // bytes), one direction, fast, its output grid: the field from slot `cur` to slot `prev` = the current frame -> its
    // predecessor (FSR's convention), in px: two planes + k_nvofUp's cost plane
    struct OfVec
    {
        NvOFHandle of = nullptr;
        NvOFGPUBufferHandle in[2] = {}, out[2] = {}, cost[2] = {};
        CUdeviceptr inP[2] = {}, outP[2] = {}, costP[2] = {};
        uint32_t inPitch = 0, outPitch[2] = {}, costPitch[2] = {};
        int w = 0, h = 0, grid = 1, gw = 0, gh = 0;
        float* dFlow = nullptr;
        bool both = false; // SMV_OFVEC_BOTH=1: both directions, the backward field read (else forward, frames swapped)
        bool costOn = false; // SMV_OFVEC_COST=1: the engine's cost output (FSR reads none; k_nvofUp's plane unread)
        bool grid4 = false;  // SMV_OFVEC_GRID4=1: grid 4 (+ k_nvofUp's bilinear) where grid 1 is offered too
        NV_OF_PERF_LEVEL level = NV_OF_PERF_LEVEL_SLOW; // nativeOfLevel()
        cudaStream_t st = nullptr; // the stream its Optical Flow and kernels run on, null = the session's
    };
    OfVec fgOf;
    // a tree level with 2+ calls makes its vectors ahead (nativeFsrfgLevel): call j + 1's Optical Flow on fgOfSt (its own
    // session fgOf2, into dFgMvR[(j + 1) & 1]) while call j's FSR runs. fgAhead: 0 = not made yet, 1 = on, -1 = off
    // (SMV_FSRFG_OVERLAP=0, the profiler, or a failed setup: serial vectors)
    int fgAhead = 0;
    cudaStream_t fgOfSt = nullptr;
    OfVec fgOf2;
    uint8_t* dFgMvR[2] = {};
    cudaEvent_t fgEvLevel = nullptr, fgEvVec[2] = {}, fgEvFree[2] = {};
    // AMD FSR upscaling (fsrUp, `--fsr-upscale`): RTX VSR's resize (vsr is set too, so the slot rules, the staging and
    // nativeVsrEval's callers stay one path; nativeVsrEval hands dVsrIn to smv_fsrup_bridge.dll instead of NGX). The
    // bridge runs AMD's signed FidelityFX upscaler (FSR 4.1.1's INT8 model) on its own D3D12 device, frames in and out
    // through shared buffers + one fence like FSR FG's; every frame of the slot's stream is fed in order (the first
    // resets), its vectors from the previous one (fuOf, the two input slots alternate)
    bool fsrUp = false;
    std::string fsrUpDir;
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
    // SMV_FSRUP_PROF=1: CUDA events between a frame's stages (the input pack, the Optical Flow, the vector kernels, FSR
    // on D3D12, the output unpack), means at the session's end; every frame then waits for its last event
    bool fuProf = false;
    cudaEvent_t fuEv[6] = {};
    double fuStageMs[5] = {};
    uint64_t fuProfN = 0, fuProfVecN = 0;
    char fuVersion[32] = {};
    OfVec fuOf;
    // RIFE with DRBA timing (rifedrba): the RIFE
    // handoff plus `NATIVE-PATH block0=` (calc_flow's block0 as its own engine) and `engine=drba
    // lag=1`. live_server.RifeDrba natively: a four-frame history of padded frames and their
    // encodes, windows centred on a frame id (two kept, chained: a window's left flow is the
    // previous window's right one, reversed), the lag-1 group (every slot shows time (k-2) + f,
    // the real slot is frame k-1), one IFNet enqueue per tween with its own DRM timestep map
    // (the kernels k_drbaFlowSplat / FlowNorm / DrmSplat / DrmNorm above, gated against a float64
    // reference), the head fallback (plain pair RIFE on the lagged pair).
    // RIFE's two domains (mph set): the ring keeps each frame's motion frame beside the picture, and
    // the encodes, block0 and its flows, the windows, the DRM map and the IFNet's x live at the
    // motion frame (M x N = mph x mpw, else the pictures' ph x pw); k_rifeBlend warps the pictures.
    bool drba = false;
    std::string block0Path, block0Jit;
    nvinfer1::ICudaEngine* engB0 = nullptr;
    nvinfer1::IRuntimeConfig* cfgB0 = nullptr;
    nvinfer1::IExecutionContext* ctxB0 = nullptr;
    uint32_t drFid = 0;      // frames pushed this session; frame id i sits in ring slot i & 3
    float* dDrI[4] = {};     // (3, ph, pw) the padded frames (the pictures)
    uint16_t* dDrM[4] = {};  // two domains: (3, M, N) fp16, their motion frames
    float* dDrF[4] = {};     // (16, M, N) their encodes, fp32 (fp16 in place when featHalf)
    float* dDrX[2] = {};     // (6, ph, pw) the frame pair: [0] = [k-1, k-2] (side -1), [1] = [k-2, k-1]; the
                             // IFNet's x (fp16 in place when xHalf), in two domains the pictures (fp32)
    uint16_t* dDrMX[2] = {}; // two domains: (6, M, N) fp16, the IFNet's x (the motion frames of that pair)
    float* dDrB0 = nullptr;  // two domains: (6, M, N) block0's frame pair, the motion frames in fp32
    uint32_t drXFor[2] = {}; // the newest frame id each x was built for (0 = none)
    struct DrWin
    {
        uint32_t c;
        float* f10;
        float* r;
    }; // r = (4, M, N): flow12 | flow21
    DrWin drWin[2] = {};         // c = 0: empty
    float* dDrFlow = nullptr;    // (4, M, N) block0's output
    float* dDrFlowN = nullptr;   // (4, M, N) a non-chained left flow's FlowNorm output
    long long* dDrAcc = nullptr; // (6, M, N) the splat accumulator
    uint64_t drTweens = 0, drHeads = 0, drBlock0 = 0;
    // NVIDIA Optical Flow model: the handoff answers
    // `engine=nvof` with the geometry only (like echo, no TensorRT engine at all); this host runs
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
    bool restHalfIn = true, restHalfOut = true; // the engine's x / y dtypes (read off it)
    bool restFailed = false;
    int restTw = 0, restTh = 0; // the fold target (decided once VSR is known)
    nvinfer1::ICudaEngine* engRest = nullptr;
    nvinfer1::IRuntimeConfig* cfgRest = nullptr;
    nvinfer1::IExecutionContext* ctxRest = nullptr;
    void* dRestIn = nullptr;   // x: NCHW at the source size (srcW x srcH), fp16 (or fp32)
    void* dRestOut = nullptr;  // y: NCHW 4x, the engine's dtype
    float* dRestTmp = nullptr; // the fold's horizontal pass, planar fp32 at (4h) x max target width
    float* dRestF = nullptr;   // the enlarging fold only: y as clamped fp32 planar
    float* dRest = nullptr;    // the fold back to the model size (VSR follows), planar fp32 w x h
    float* dRestRem = nullptr; // HDR planes: the source fitted to Restore's target (k_hdrRestOut's remainder)

    // ---- ring geometry (mirrors the exe's own slot layout)
    uint32_t slots = 0, pitch = 0;
    size_t slotBytes = 0;

    // ---- TRT
    nvinfer1::IRuntime* rt = nullptr;
    nvinfer1::ICudaEngine* engIf = nullptr;
    nvinfer1::ICudaEngine* engEnc = nullptr;
    nvinfer1::IRuntimeConfig* cfgIf = nullptr;
    nvinfer1::IRuntimeConfig* cfgEnc = nullptr;
    nvinfer1::IRuntimeCache* jit = nullptr;
    nvinfer1::IExecutionContext* ctxIf = nullptr;
    nvinfer1::IExecutionContext* ctxEnc = nullptr;
    bool encHalf = true;
    bool featHalf = false; // IFNet / block0 take f0 / f1 in fp16: the encode writes dF directly
    // the IFNet's x / the encode's img in fp16 (read off each engine on its own, so a cache that
    // mixes the two revisions still works: the encode features are the same either way)
    bool xHalf = false, imgHalf = false;
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
    // capArr as a point-sampled texture: the live 1:1 pack reads the capture through it, no copy to dCap first (0 = the
    // copy, SMV_CAP_DIRECT=0 or a refused texture)
    cudaTextureObject_t capTex = 0;
    uint8_t* dOutRing = nullptr;
    uint8_t* dCap = nullptr;
    float* dTmp = nullptr;
    float* dCapF = nullptr;       // HDR + image scale under 1: planar PQ at capture resolution
    uint8_t* dThdrIn = nullptr;   // BGRA8 bridge input, pitch 4*cw
    uint32_t* dThdrOut = nullptr; // packed 10:10:10:2 bridge output, B in the low bits
    float* dSrcG = nullptr;       // the unquantised sRGB-encoded source, planar fp32
    float* dShIn = nullptr;       // Sharpen / TrueHDR after DLSS 5: the model frame without its pad (w x h)
    float* dShOut = nullptr;      // and the tight result (the sharpen, or TrueHDR's PQ) padded back into it
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
                                // reference once DLSS 5 has rewritten the frames in dX
    uint16_t* dNrIo = nullptr;  // device RGBA16F, both directions
    uint16_t* hNrIn = nullptr;  // pinned host copies handed to renderFrame (CPU staging)
    uint16_t* hNrOut = nullptr;
    uint8_t* dNrShIn = nullptr; // zero-copy: the core's shared buffers, rows nrPitch bytes apart
    uint8_t* dNrShOut = nullptr;
    size_t nrPitch = 0;
    cudaExternalMemory_t emNrIn = nullptr, emNrOut = nullptr;
    cudaExternalSemaphore_t semNr = nullptr; // the core's shared fence
    uint64_t nrFenceV = 0;                   // last value put on it: odd = the stream's, even = the queue's
    // DLSS 5 motion (zero-copy route, nativeNrMotion): its own Optical Flow session at nrW x nrH on
    // the NR input frames, current -> previous, into the core's shared motion buffer (DLSSNR.MVec).
    // Off = SMV_NR_MV=0 or CPU staging: the pass then gets no motion
    bool nrMotion = false;
    NvOFHandle nrOfH = nullptr;
    NvOFGPUBufferHandle nrOfIn[2] = {}, nrOfOut = nullptr, nrOfCost = nullptr;
    CUdeviceptr nrOfInP[2] = {}, nrOfOutP = 0, nrOfCostP = 0;
    uint32_t nrOfInPitch = 0, nrOfOutPitch = 0, nrOfCostPitch = 0;
    int nrOfGw = 0, nrOfGh = 0;
    int nrOfCur = 0;            // the input slot this frame's luma goes into (they alternate)
    float* dNrFlow = nullptr;   // (3, nrH, nrW): the field in px (2 planes) + its cost
    uint8_t* dNrShMv = nullptr; // the core's shared motion buffer, rows nrMvPitch bytes apart
    size_t nrMvPitch = 0;
    cudaExternalMemory_t emNrMv = nullptr;
    std::wstring nrDeltaPath; // --nr-delta PATH: the pass's change map (the preview's mask)
    double nrMs = 0.0, nrMaxMs = 0.0;
    uint64_t nrN = 0;
    // live DLSS 5: the same core, handoff and motion once per captured frame on the
    // model frame after Restore / the resize (nativeLiveNrInit, nativeGroup); a capture byte-identical
    // to the previous one takes dNrLast instead of an evaluate
    bool liveNr = false;
    bool nrHaveLast = false;     // dNrLast holds an evaluated frame
    float* dNrLast = nullptr;    // the last evaluated model frame, 3 padded planes
    uint8_t* dCapPrev = nullptr; // the previous raw capture, the reuse test's reference
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
    bool nrSdrIn = false; // live: DLSS 5's input is SDR this frame (RTX HDR converts after it)
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
    float* dPres = nullptr;     // sharpen / fitAa: the fitted frame, planar fp32 at dw x dh
    float* dFitTmp = nullptr;   // fitAa: the horizontal pass, planar fp32 at dw x (uh or h)
    float* dUp = nullptr;       // Upscale to: the internal render frame, planar fp32 at uw x uh
    float* dUpTmp = nullptr;    // upAa: the horizontal pass of the first resize, planar fp32 at uw x h
    uint8_t* dVsrIn = nullptr;  // VSR: tight BGRA8 at the source size (srcW x srcH)
    uint8_t* dVsrOut = nullptr; // VSR: tight BGRA8 at the presented size dw x dh
    float* dX = nullptr;        // (1,6,ph,pw): prev in planes 0..2, cur in 3..5 (xSwap: the reverse)
    bool xSwap = false;         // live: dX's halves trade roles instead of the cur -> prev copy (nativeXPrev / Cur)
    uint16_t* dXh = nullptr;    // dX in fp16 for the RIFE engines that take fp16 frames (xHalf /
                                // imgHalf); DRBA uses its cur half for the encode only
    float* dF[2]{};             // f_prev / f_cur, (1,16,ph,pw)
    uint16_t* dEncHalf = nullptr;
    // the IFNet's timestep, flow and mask and the tweens, allocated once the engine's dtypes are
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
    // RIFE's two domains (offline RIFE / Frame Blend, live RIFE / Frame Blend): the IFNet and its
    // encode run on the motion frames dM, the finished picture in SDR sRGB at the decoded size
    // (offline) or at the working size, the capture size when the working size enlarges it (live)
    // (mw x mh, padded to mph x mpw), and k_rifeBlend applies their flow and mask to the pictures dX
    // at the working size. mph = 0: the model's frame is the picture's (ph x pw), the IFNet reads dX / dXh.
    int mph = 0, mpw = 0, mw = 0, mh = 0;
    uint16_t* dM = nullptr; // (1,6,mph,mpw) fp16: prev in planes 0..2, cur in 3..5
    float* dMFit = nullptr; // live: the working-size picture shrunk to mw x mh (3 fp32 planes)
    int fCur = 0;
    cudaEvent_t capEv = nullptr;
    std::vector<cudaEvent_t> slotEv;
    // live: a captured frame's own passes timed on the GPU (the capture read through Sharpen / RTX HDR and the motion
    // frame; pfEv[1] = where the part at the working size starts, after Restore / the resize), smoothed, in
    // microseconds for the present loop's Auto step (it reads them at its stats tick)
    cudaEvent_t pfEv[3] = {};
    bool pfArmed = false;
    double pfAllMs = 0.0, pfWorkMs = 0.0;
    std::atomic<uint32_t> pfAllUs{0}, pfWorkUs{0};
    // live: a group's own GPU time for the present loop's throttle, from pfEv[0] (the capture read): grEv[0] where the
    // slots start (after the pair's model work), grEv[1] after the last slot. Smoothed per kind of group, in
    // microseconds: gIntUs = the work before the slots of a group with tweens, gTweenUs = one tween with its store,
    // gBaseUs = a whole group without a tween (the real frame alone)
    cudaEvent_t grEv[2] = {};
    bool grArmed = false;
    uint32_t grTween = 0;
    double gIntMs = 0.0, gTweenMs = 0.0, gBaseMs = 0.0;
    std::atomic<uint32_t> gIntUs{0}, gTweenUs{0}, gBaseUs{0};

    CUmodule cuMod = nullptr;
    CUfunction fPackInDirect = nullptr, fResizeH = nullptr, fResizeV = nullptr, fH2f = nullptr, fF2h = nullptr,
               fPackOut = nullptr;
    CUfunction fPackInDirectHdr = nullptr, fPqPlanar = nullptr, fResizeHf = nullptr, fPackOutHdr = nullptr;
    CUfunction fSdrEncode = nullptr, fThdrColor = nullptr;                                    // live TrueHDR
    CUfunction fThdrIn = nullptr, fSharpThdrIn = nullptr, fPqOut = nullptr, fPqLut = nullptr, // TrueHDR in, the
        fSdrPq = nullptr;                                                                     // offline PQ emit
    CUfunction fFitPlanar = nullptr, fSharpPlanar = nullptr,                                  // sharpen
        fPackBgraRgb = nullptr, fUnpackBgraRgb = nullptr,                   // VSR / FRUC: (R, G, B) <-> BGRA8
        fPackRgbaH = nullptr, fRgbaHAbgr = nullptr, fUnpackRgbaH = nullptr, // FSR frame generation: RGBA16F frames,
        fMvHalf = nullptr, fMvHalfG1 = nullptr,                             // the Optical Flow input, the vectors
        fBgraRgbaFsr = nullptr, fRgbaBgra = nullptr,                        // FSR upscaling: VSR's BGRA8 <-> RGBA8
        fPackBgra = nullptr, fUnpackRgba = nullptr, fPackR10 = nullptr,     // offline DLSS 4.5 frames
        fUnpackR10 = nullptr;
    CUfunction fFitAaH = nullptr, fFitAaV = nullptr;        // the separable fit
    CUfunction fPackOutV = nullptr, fPackOutHdrV = nullptr; // its vertical pass + the slot store
    CUfunction fPackOutTile = nullptr;                      // both passes + the store, a shrinking height
    CUfunction fFitAaTile = nullptr;                        // both passes into a planar target
    CUfunction fRestIn = nullptr, fRestFoldH = nullptr, fRestFoldV = nullptr, // live Restore
        fRestFoldTile = nullptr, fRestToF = nullptr, fClamp01 = nullptr;
    CUfunction fPadPlanar = nullptr;                                  // the decoded frame into the model frame
    CUfunction fNrIn = nullptr, fNrOut = nullptr, fNrMv = nullptr;    // DLSS 5
    CUfunction fNrInPq = nullptr, fNrOutPq = nullptr;                 // live DLSS 5 on HDR (PQ) planes
    CUfunction fHalf = nullptr, fPyr = nullptr, fSplatSoft = nullptr, // live GMFSS glue (5b)
        fSplatNorm = nullptr, fGmShrink = nullptr, fGmFlowUp = nullptr;
    CUfunction fPackInRaw16 = nullptr, fPackInRaw8 = nullptr, // offline
        fPackOutRaw16 = nullptr, fPackOutRaw8 = nullptr, fExpand8to16 = nullptr;
    CUfunction fPairDiff = nullptr, fRawDiff = nullptr;                       // identical-pair test, DLSS 5 reuse test
    CUfunction fNvofLuma = nullptr, fNvofUp = nullptr, fNvofMetric = nullptr; // the nvof model
    CUfunction fSplatVel = nullptr, fVelNorm = nullptr, fPpDown = nullptr,    // its pull-warp tween
        fPpTop = nullptr, fPpUp = nullptr, fBlur1 = nullptr, fNvofCompose = nullptr;
    CUfunction fDrFlowSplat = nullptr, fDrFlowNorm = nullptr, // native DRBA
        fDrDrmSplat = nullptr, fDrDrmNorm = nullptr;
    CUfunction fPackInMotion = nullptr; // live 1:1: the pack, the motion frame and the identical-pair test in one pass
    CUfunction fMotionIn = nullptr, fRifeBlend = nullptr, // RIFE's motion frame, its last step (the tween: fp16
        fRifeBlendF = nullptr;                            // store, fp32 store)
    CUfunction fRifeBlendOut = nullptr;                   // its last step fused with a 1:1 slot store (blendOut)
    CUfunction fRifeBlendOut32 = nullptr;                 // the same for fp32 pictures
    CUfunction fHdrEnc = nullptr, fHdrFromSdr = nullptr, fHdrRange = nullptr; // GMFSS / Restore on HDR planes
    CUfunction fPackBgraSdr = nullptr, fUnpackBgraSdr = nullptr;              // FRUC's SDR pairs on HDR planes
    CUfunction fHdrRestOut = nullptr;                                         // Restore's way back on HDR planes
    CUfunction fLocalCorr = nullptr;                                          // gmflow's local correlation, fused
    CUfunction fAttn2 = nullptr;  // gmflow's global matching and propagation (k_attn2)
    CUfunction fAttn2b = nullptr; // the same, double-buffered (k_attn2b, gmAttnB)
    int* dStaticFlag = nullptr;   // device flag k_pairDiff sets when the pair differs
    int* hStaticFlag = nullptr;   // pinned readback of it, one int per group
    uint64_t staticN = 0;         // identical pairs held this session

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

// What a --resident process keeps between sessions: the TensorRT runtime, the two deserialized
// engines, the JIT kernel cache and the NVRTC kernel module, plus the last successful
// handoff's facts so the same window starts again without a new handoff.
// Everything else in NativeRife (stream, execution contexts, buffers, imports, events) is per
// session. Sessions are strictly sequential (the previous compute thread is joined before
// the next early thread starts), so no lock guards this.
struct NativeResident
{
    std::string ifnetPath, encodePath, jitPath, restorePath;
    nvinfer1::IRuntime* rt = nullptr;
    nvinfer1::ICudaEngine* engIf = nullptr;
    nvinfer1::ICudaEngine* engEnc = nullptr;
    nvinfer1::ICudaEngine* engRest = nullptr; // live Restore: kept beside the pair, per path
    std::string block0Path;                   // native DRBA: kept beside the pair, per path
    nvinfer1::ICudaEngine* engB0 = nullptr;
    std::string gmPath[kGmN]; // the GMFSS set, exclusive with the pair
    nvinfer1::ICudaEngine* engGm[kGmN] = {};
    nvinfer1::IRuntimeCache* jit = nullptr;
    CUmodule cuMod = nullptr;
    bool encHalf = true;
    bool featHalf = false;
    bool xHalf = false, imgHalf = false, outHalf = false, tHalf = false;
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
    if (!g_res.rt)
        return;
    if (g_res.jit)
    {
        delete g_res.jit;
        g_res.jit = nullptr;
    }
    if (g_res.engIf)
    {
        delete g_res.engIf;
        g_res.engIf = nullptr;
    }
    if (g_res.engEnc)
    {
        delete g_res.engEnc;
        g_res.engEnc = nullptr;
    }
    if (g_res.engRest)
    {
        delete g_res.engRest;
        g_res.engRest = nullptr;
    }
    if (g_res.engB0)
    {
        delete g_res.engB0;
        g_res.engB0 = nullptr;
    }
    g_res.block0Path.clear();
    for (int i = 0; i < kGmN; i++)
    {
        if (g_res.engGm[i])
        {
            delete g_res.engGm[i];
            g_res.engGm[i] = nullptr;
        }
        g_res.gmPath[i].clear();
    }
    delete g_res.rt;
    g_res.rt = nullptr;
    g_res.ifnetPath.clear();
    g_res.encodePath.clear();
    g_res.jitPath.clear();
    g_res.restorePath.clear();
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

// the server backends this host runs (PipeServer::nativeRefusal asks here): rife, blend (Frame
// Blend live = the RIFE engines under its own name), echo (the effects-only route, the
// no-engine mode above), gmfss (the five-engine chain), nvof (the NVIDIA Optical Flow model),
// fruc (Nvidia Smooth Motion), fsrfg (AMD FSR frame generation) and rifedrba (RIFE with DRBA timing). Any other name
// is refused.
static bool nativeBackendOk(const std::wstring& backend)
{
    return backend == L"rife" || backend == L"blend" || backend == L"echo" || backend == L"gmfss" ||
           backend == L"nvof" || backend == L"fruc" || backend == L"fsrfg" || backend == L"rifedrba";
}

// the GMFSS engine set, in handoff order (the NATIVE-PATH keys and the log names)
static const char* const kGmKey[kGmN] = {"gfeat", "gflow", "gmetric", "gifnet", "gfusion", "gflowb", "gbone", "gflowm"};
static const char* const kGmName[kGmN] = {"feat_ext",  "gmflow_bidir_a", "metricnet",       "ifnet",
                                          "fusionnet", "gmflow_bidir_b", "gmflow_backbone", "gmflow_bidir_a_m"};

// gmflow_bidir_a with its quarter-scale transformer in 1 / 2 / 4 / 8 groups of whole windows (onnx_export.py
// GMFLOW_CHUNKS): the most groups up to 8 that keep >= kGmChunkTokens quarter-scale tokens (both frames) a group, so a
// group's tensors stay near the L2; fewer tokens a group cost more than they save
static const char* const kGmFlowA[4] = {"gmflow_bidir_a", "gmflow_bidir_a_g2", "gmflow_bidir_a_g4",
                                        "gmflow_bidir_a_g8"};
static constexpr int64_t kGmChunkTokens = 4500;
// from kGmRegionTokens a group up, the 8-group graph whose shifted blocks attend per mask region (onnx_export.py
// GMFLOW_REGION_CHUNKS) instead of through the masked kernel: big windows gain (the 1440p DLAA half x0.94, the 4K half
// x0.83), small ones lose to the extra calls (5,376 a group x1.03)
static constexpr int64_t kGmRegionTokens = 9000;
// the scale-0 matching's token count up to which k_attn2b runs instead of k_attn2 (bit-identical; 160 W, matching +
// propagation: Ultra's 1,792 x0.88, Performance's 3,840 x1.07 and Balanced's 5,376 x1.16 slower); SMV_GM_ATTN2B=0 =
// k_attn2 at every N
static constexpr int kAttn2bMaxN = 1792;
static const char* lkGmFlowA(int gh, int gw)
{
    const int64_t tokens = 2 * (int64_t)(gh / 4) * (gw / 4);
    int i = 3;
    while (i > 0 && (tokens >> i) < kGmChunkTokens)
        --i;
    if (i == 3 && (tokens >> 3) >= kGmRegionTokens)
        return "gmflow_bidir_a_g8r";
    return kGmFlowA[i];
}

// the `_m` graph cut from the same export as lkGmFlowA's pick (trt_runtime.gmflow_matching_cut)
static const char* lkGmFlowM(int gh, int gw)
{
    static const char* const a[5] = {"gmflow_bidir_a", "gmflow_bidir_a_g2", "gmflow_bidir_a_g4", "gmflow_bidir_a_g8",
                                     "gmflow_bidir_a_g8r"};
    static const char* const m[5] = {"gmflow_bidir_a_m", "gmflow_bidir_a_g2_m", "gmflow_bidir_a_g4_m",
                                     "gmflow_bidir_a_g8_m", "gmflow_bidir_a_g8r_m"};
    const char* pick = lkGmFlowA(gh, gw);
    for (int i = 0; i < 5; i++)
        if (!strcmp(pick, a[i]))
            return m[i];
    return nullptr;
}

// Upscale to: _Fit.__init__'s derivation of the
// internal render size from the exe's own --upscale H and the handoff geometry: the factor
// up_h / h clamped to 1/16..16 (render.py's UPSCALE_F clamp), even dims, dropped when it
// equals the fit rect (one resize, not two). Python rounds half to even and this rounds half
// away, but the & ~1 lands both on the same even number. Runs after (w, h, dw, dh) are known,
// on the resident facts too (uw is never a stored fact, the flag is per session).
static void nativeDeriveUpscale(NativeRife& nr)
{
    nr.uw = nr.uh = 0;
    nr.upAa = nr.upTile = false;
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

// ---- warm engine lookup ---------------------------------------------------------------------
// trt_lookup.py's naming and find_* checks in C++ (trt_lookup.engine_name stays the naming
// reference of the dev tools, so the two must agree): when a previous session built AND warmed the engines
// for this window, the lookup answers the handoff as NATIVE-PATH lines plus one LIVE READY line,
// the format nativeHandoff parses. A miss returns false and nativeHandoff builds the engines.
struct LookupTags
{
    std::string trt, w, r, rest;
    bool ok = false;
}; // rest = realesr.weights_hash()

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

// md5 of the concatenated file contents, first 10 hex digits (trt_lookup._md5_files)
static std::string lkMd5Files(const std::vector<std::wstring>& files)
{
    HCRYPTPROV prov = 0;
    HCRYPTHASH h = 0;
    std::string out;
    if (!CryptAcquireContextW(&prov, nullptr, nullptr, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT))
        return out;
    if (CryptCreateHash(prov, CALG_MD5, 0, 0, &h))
    {
        bool ok = true;
        std::vector<BYTE> buf(1 << 20);
        for (const auto& f : files)
        {
            HANDLE fh = CreateFileW(f.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                    FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
            if (fh == INVALID_HANDLE_VALUE)
            {
                ok = false;
                break;
            }
            DWORD got = 0;
            while (ok && ReadFile(fh, buf.data(), (DWORD)buf.size(), &got, nullptr) && got)
                ok = CryptHashData(h, buf.data(), got, 0) != FALSE;
            CloseHandle(fh);
            if (!ok)
                break;
        }
        BYTE dig[16];
        DWORD dl = sizeof(dig);
        if (ok && CryptGetHashParam(h, HP_HASHVAL, dig, &dl, 0))
        {
            char hex[33];
            for (int i = 0; i < 16; i++)
                sprintf_s(hex + 2 * i, 3, "%02x", dig[i]);
            out.assign(hex, 10);
        }
        CryptDestroyHash(h);
    }
    CryptReleaseContext(prov, 0);
    return out;
}

// the name tags trt_lookup bakes into every engine: trt<version> (the tensorrt_rtx package
// version, read from the same dist-info folder python reports), w<md5 of GMFSS train_log/*.pkl>
// and r<md5 of rife/flownet.pkl>. Hashed once per process (the resident host keeps them).
static const LookupTags& lkTags(const std::wstring& engDir)
{
    static LookupTags t;
    static std::wstring forDir;
    if (forDir == engDir)
        return t;
    forDir = engDir;
    t = LookupTags();
    // the version stage-gpu-runtime.js copied from the wheel's dist-info name (1.6.1.120)
    FILE* vf = nullptr;
    if (!_wfopen_s(&vf, (engDir + L"\\gpu_runtime\\tensorrt_rtx_version.txt").c_str(), L"rb") && vf)
    {
        char buf[64]{};
        const size_t n = fread(buf, 1, sizeof(buf) - 1, vf);
        fclose(vf);
        std::string v(buf, n);
        while (!v.empty() && (v.back() == '\n' || v.back() == '\r' || v.back() == ' '))
            v.pop_back();
        for (char& c : v)
            if (c == '.')
                c = '_';
        if (!v.empty())
            t.trt = "trt" + v;
    }
    WIN32_FIND_DATAW fd{};
    HANDLE hf;
    std::vector<std::wstring> pkls;
    const std::wstring tl = engDir + L"\\GMFSS_Fortuna\\train_log";
    hf = FindFirstFileW((tl + L"\\*.pkl").c_str(), &fd);
    if (hf != INVALID_HANDLE_VALUE)
    {
        do
        {
            pkls.push_back(fd.cFileName);
        } while (FindNextFileW(hf, &fd));
        FindClose(hf);
    }
    // python: sorted(os.listdir(...)), code point order on the names
    std::sort(pkls.begin(), pkls.end());
    for (auto& p : pkls)
        p = tl + L"\\" + p;
    const std::string w = pkls.empty() ? std::string() : lkMd5Files(pkls);
    const std::string r = lkMd5Files({engDir + L"\\rife\\flownet.pkl"});
    t.w = w.empty() ? std::string() : "w" + w;
    t.r = r.empty() ? std::string() : "r" + r;
    // the Restore engine's name part: md5[:8] of the bundled Real-ESRGAN weights (optional)
    const std::string rs = lkMd5Files({engDir + L"\\realesr-animevideov3.pth"});
    t.rest = rs.size() >= 8 ? rs.substr(0, 8) : std::string();
    // a shipped tree carries no weight files (the ONNX hold the weights): the tags
    // onnx_export.py wrote next to the ONNX ("w <tag>", "r <tag>", "rest <tag>"); a tag whose
    // weight file exists is always the hash above, so a dev tree never reads a stale file
    if (t.w.empty() || t.r.empty() || t.rest.empty())
    {
        std::string od = lkEnv("SMV_ONNX_DIR");
        std::wstring odW = od.empty() ? engDir + L"\\onnx" : std::wstring(od.begin(), od.end());
        FILE* tf = nullptr;
        if (!_wfopen_s(&tf, (odW + L"\\weights_tags.txt").c_str(), L"rb") && tf)
        {
            char line[128];
            while (fgets(line, sizeof(line), tf))
            {
                std::string s(line);
                while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' '))
                    s.pop_back();
                const size_t sp = s.find(' ');
                if (sp == std::string::npos)
                    continue;
                const std::string k = s.substr(0, sp), v = s.substr(sp + 1);
                if (k == "w" && t.w.empty())
                    t.w = v;
                else if (k == "r" && t.r.empty())
                    t.r = v;
                else if (k == "rest" && t.rest.empty())
                    t.rest = v;
            }
            fclose(tf);
        }
    }
    t.ok = !t.trt.empty() && !t.w.empty() && !t.r.empty();
    return t;
}

// python's round(): ties to even (the default FP rounding mode), then int()
static int lkRound(double x)
{
    return (int)std::nearbyint(x);
}

static std::string lkG(double v) // f"{v:g}"
{
    char b[64];
    sprintf_s(b, "%g", v);
    return b;
}

struct LkShape
{
    const char* name;
    std::vector<int64_t> dims;
};

// engine_name(): <base>_<shape per input joined by x, inputs by _>_<trt>_<w>, a dynamic batch
// axis written lo"to"hi
static std::string lkEngineName(const std::string& base, const std::vector<LkShape>& set, const char* dynInput,
                                int dynLo, int dynHi, const LookupTags& t)
{
    std::string s = base;
    for (const auto& in : set)
    {
        s += '_';
        for (size_t i = 0; i < in.dims.size(); i++)
        {
            if (i)
                s += 'x';
            if (i == 0 && dynInput && !strcmp(in.name, dynInput))
                s += std::to_string(dynLo) + "to" + std::to_string(dynHi);
            else
                s += std::to_string(in.dims[i]);
        }
    }
    return s + "_" + t.trt + "_" + t.w;
}

static bool lkFile(const std::string& p)
{
    return fileExistsA(p);
}

static std::string lkJit(const std::string& enginePath)
{
    const std::string kind = lkEnv("SMV_TRT_CACHE_KIND");
    return enginePath + (kind.empty() ? "" : "." + kind) + ".jit";
}

static bool lkWarm(const std::string& jit, const std::string& key)
{
    if (!lkFile(jit))
        return false;
    std::ifstream f(jit + ".warm");
    std::string ln;
    while (std::getline(f, ln))
    {
        while (!ln.empty() && (ln.back() == '\r' || ln.back() == ' ' || ln.back() == '\t'))
            ln.pop_back();
        size_t a = 0;
        while (a < ln.size() && (ln[a] == ' ' || ln[a] == '\t'))
            a++;
        if (ln.compare(a, std::string::npos, key) == 0)
            return true;
    }
    return false;
}

static const char* lkDtype(nvinfer1::DataType d)
{
    switch (d)
    {
    case nvinfer1::DataType::kFLOAT:
        return "fp32";
    case nvinfer1::DataType::kHALF:
        return "fp16";
    case nvinfer1::DataType::kBF16:
        return "bf16";
    case nvinfer1::DataType::kINT32:
        return "i32";
    case nvinfer1::DataType::kINT64:
        return "i64";
    case nvinfer1::DataType::kBOOL:
        return "bool";
    case nvinfer1::DataType::kUINT8:
        return "u8";
    default:
        return "?";
    }
}

struct LkIo
{
    std::vector<std::pair<std::string, std::string>> ins, outs;
};

// find_engine() + _inspect(): the file by name, deserialized without a context, every input
// admitting every shape set (static dims exact, dynamic ones inside profile 0)
static bool lkFind(nvinfer1::IRuntime* rt, const std::string& cacheDir, const std::string& base,
                   const std::vector<std::vector<LkShape>>& sets, const char* dynInput, int dynLo, int dynHi,
                   const LookupTags& t, std::string& path, LkIo& io)
{
    path = cacheDir + "\\" + lkEngineName(base, sets[0], dynInput, dynLo, dynHi, t) + ".engine";
    if (!lkFile(path))
        return false;
    std::ifstream f(path, std::ios::binary);
    std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (blob.empty())
        return false;
    std::unique_ptr<nvinfer1::ICudaEngine> eng(rt->deserializeCudaEngine(blob.data(), blob.size()));
    if (!eng)
        return false;
    for (const auto& set : sets)
        for (const auto& in : set)
        {
            const nvinfer1::Dims d = eng->getTensorShape(in.name);
            if (d.nbDims != (int)in.dims.size())
                return false;
            bool dyn = false;
            for (int k = 0; k < d.nbDims; k++)
                dyn = dyn || d.d[k] < 0;
            if (!dyn)
            {
                for (int k = 0; k < d.nbDims; k++)
                    if (d.d[k] != in.dims[k])
                        return false;
                continue;
            }
            const nvinfer1::Dims lo = eng->getProfileShape(in.name, 0, nvinfer1::OptProfileSelector::kMIN);
            const nvinfer1::Dims hi = eng->getProfileShape(in.name, 0, nvinfer1::OptProfileSelector::kMAX);
            if (lo.nbDims != d.nbDims || hi.nbDims != d.nbDims)
                return false;
            for (int k = 0; k < d.nbDims; k++)
                if (in.dims[k] < lo.d[k] || in.dims[k] > hi.d[k])
                    return false;
        }
    io = LkIo();
    for (int i = 0; i < eng->getNbIOTensors(); i++)
    {
        const char* n = eng->getIOTensorName(i);
        auto e = std::make_pair(std::string(n), std::string(lkDtype(eng->getTensorDataType(n))));
        (eng->getTensorIOMode(n) == nvinfer1::TensorIOMode::kINPUT ? io.ins : io.outs).push_back(e);
    }
    return true;
}

// the RIFE IFNet / block0 class: fp32 everywhere except the feature encodes f0 / f1 (fp32 or fp16, the same
// for both; nativeTrtInit pairs them with the encode output), the IFNet's frame pair x and timestep (fp32 or
// fp16) and its outputs flow / mask (fp16); block0's frames and flow stay fp32
static bool lkRifeIo(const LkIo& io, bool ifnet)
{
    std::string fd;
    for (auto& p : io.ins)
    {
        if (p.first == "f0" || p.first == "f1")
        {
            if ((p.second != "fp32" && p.second != "fp16") || (!fd.empty() && fd != p.second))
                return false;
            fd = p.second;
        }
        else if (ifnet && (p.first == "x" || p.first == "timestep"))
        {
            if (p.second != "fp32" && p.second != "fp16")
                return false;
        }
        else if (p.second != "fp32")
            return false;
    }
    for (auto& p : io.outs)
        if (p.second != "fp32" && !(ifnet && (p.first == "flow" || p.first == "mask") && p.second == "fp16"))
            return false;
    return true;
}

static std::string lkDt(const LkIo& io, bool in, const char* name)
{
    for (auto& p : (in ? io.ins : io.outs))
        if (p.first == name)
            return p.second;
    return "None";
}

static std::string lkF4(double v)
{
    char b[64];
    sprintf_s(b, "%.4f", v);
    return b;
}

// One net of the GMFSS set: the handoff key, the engine base name, the ONNX file key, the
// build workspace multiplier, the input shapes and how many of them the engine's name carries (0 = all)
struct LkGmNet
{
    const char* key;
    const char* base;
    const char* onnx;
    int ws;
    std::vector<LkShape> set;
    size_t nameIn = 0;
};

// the inputs that name a GMFSS engine (and that its lookup checks): gmflow_bidir_b's corr and img0 fix every other
// input's shape, and all eight would push its cache paths toward MAX_PATH
static std::vector<LkShape> lkGmNameSet(const LkGmNet& n)
{
    return n.nameIn ? std::vector<LkShape>(n.set.begin(), n.set.begin() + n.nameIn) : n.set;
}

// GMFSS's flow grid floor at a --gmfss-flow below 1 (lkSession)
constexpr int kGmFlowMinW = 320, kGmFlowMinH = 192;

// Everything the lookup and the build derive from a session's arguments, in ONE place: a name
// or a size that differed between the two would be an engine the build writes and the lookup
// never finds. Sizes follow trt_lookup / the engine classes in trt_runtime.py.
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
    std::string warmKey;      // the line a warmed .jit.warm marker carries
    // gmfss: the half size, gmflow's /32 grid at a live --gmfss-flow below 1 (0 = the half), the nets
    int hh = 0, hw = 0, fh = 0, fw = 0;
    std::vector<LkGmNet> gm;
    // rife / blend / rifedrba
    bool drba = false;
    double fs = 1.0;           // SMV_RIFE_FLOW_SCALE snapped to a power-of-two rung
    std::string scaleTag;      // name part: the scale list
    std::vector<int64_t> x, f; // the IFNet's frame pair and feature shapes
    // live rife / blend: RIFE's motion frame (mmw x mmh, padded mph x mpw), the frame the IFNet and
    // its encode run at (iph x ipw = the motion pad, else the pictures' pad)
    int mmw = 0, mmh = 0, mph = 0, mpw = 0, iph = 0, ipw = 0;
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

// the rect a w x h frame takes in the outW x outH canvas: aspect-fit, even, centred (_Fit's fit rect)
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
    y0 = (outH - dh) / 2; // python // on non-negative values
}

// NVIDIA's DLSS modes as plan.ts has them (DLSS_MODES, autoMode): the working size's share of the output per
// kDlssModeName entry, and Auto's pick by the output's pixel count (below 1920x1080 DLAA, up to 2560x1440
// Quality, up to 3840x2160 Performance, else Ultra Performance), as 1 + the kDlssModeName index
static const double kDlssModeShare[6] = {1.0, 1.0, 1.0 / 1.5, 1.0 / 1.724, 1.0 / 2, 1.0 / 3};
static int liveAutoMode(int w, int h)
{
    const double px = (double)w * h;
    return px < 1920.0 * 1080 ? 2 : (px <= 2560.0 * 1440 ? 3 : (px <= 3840.0 * 2160 ? 5 : 6));
}

// DLSS 5's GPU work never completes on a 64 px wide frame (the GPU resets; 102 and 128 px wide run, and so does a
// 64 px high one): with DLSS 5 on, a working size below the presented rect is at least this wide, and a frame that is
// itself smaller than kNrMinWorkW x kNrMinWorkH runs without DLSS 5 (both routes' NR init)
static const int kNrMinWorkW = 128, kNrMinWorkH = 64;

// Live's working size (Restore, the resize, DLSS 5, Sharpen, RTX HDR and the model run at it, the fit
// takes it to the canvas after the model): the DLSS mode's share of the PRESENTED rect (the capture aspect-fit into
// the canvas W x H: the window itself, or the Fill rect), even, at most the presented rect a side, the share raised
// (aspect kept) until the shorter side is 64 px and, with DLSS 5, the width kNrMinWorkW (*raised: 1 = the 64 px
// floor, 2 = DLSS 5's width), capped at 3840x2160 keeping the aspect (offline's workPlan). DLAA = the presented rect itself (in window
// mode the capture, odd sizes kept). liveWorkSizeFor = the size a mode gives the presented presW x presH (mode = 1 +
// the kDlssModeName index, 0 = --scale's share); liveWorkSize = the session's: Auto picks the mode by the presented
// pixel count, no larger than the GPU-time step's --auto-floor, or takes the one the free video memory fits
// (g_liveAutoFit, picked once a session before anything loads), and says which mode it came from.
static void liveWorkSizeFor(int mode, int presW, int presH, int& mw, int& mh, int* raised = nullptr)
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
        const double k = (std::max)(f, (std::max)(fFloor, fNr));
        if (raised && k > f)
            *raised = fNr >= fFloor ? 2 : 1;
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
    const int outW = W ? (int)W : cw, outH = W ? (int)H : ch;
    int x0, y0;
    lkFitRect(cw, ch, outW, outH, presW, presH, x0, y0);
    mode = g_dlssMode == 1 ? (g_liveAutoFit ? g_liveAutoFit : (std::max)(liveAutoMode(presW, presH), g_liveAutoFloor))
                           : g_dlssMode;
    liveWorkSizeFor(mode, presW, presH, mw, mh, raised);
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
                      LkSession& s)
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
    s.ph = (s.mh + 63) / 64 * 64;
    s.pw = (s.mw + 63) / 64 * 64;
    char key[64];
    if (s.backend == "gmfss")
    {
        const int hh = s.hh = s.ph / 2, hw = s.hw = s.pw / 2;
        // --gmfss-flow below 1 (live; offline a gate lever): gmflow (backbone, A, B) on the /32 grid nearest the share
        // of the half, GMFSS_infer_u.reuse's round(n * scale / 32) * 32 (Python's round: half to even = nearbyint), never
        // below kGmFlowMinW x kGmFlowMinH (or the half when it is smaller): at a 256x128 grid (1080p at 25 %, 540p at
        // 50 %) GMFlow moved 28 % of a static frame's pixels by more than 16 levels and lost a known pan (20 dB vs 46);
        // 320x192 and up are clean
        // (SMV_GMFSS_FLOW_FLOOR=0: 32 px, the A/B lever of the floor)
        if (g_gmFlow < 1.0)
        {
            const bool floor = lkEnv("SMV_GMFSS_FLOW_FLOOR") != "0";
            s.fh = (std::max)((std::min)(hh, floor ? kGmFlowMinH : 32), (int)std::nearbyint(hh * g_gmFlow / 32.0) * 32);
            s.fw = (std::max)((std::min)(hw, floor ? kGmFlowMinW : 32), (int)std::nearbyint(hw * g_gmFlow / 32.0) * 32);
            if (s.fh >= hh && s.fw >= hw)
                s.fh = s.fw = 0;
        }
        const int gh = s.fh ? s.fh : hh, gw = s.fh ? s.fw : hw;
        const int64_t q4 = (int64_t)(gh / 4) * (gw / 4), q8 = (int64_t)(gh / 8) * (gw / 8), K = 81;
        // featurenet at the padded frame, the rest at the half, gmflow at its grid; gmflow_bidir as `_m` + A + B: the
        // scale-0 global matching and propagation between `_m` and A (k_attn2) and the quarter-scale local correlation
        // between A and B (k_localCorr, never the [2, 128, HW, 81] product), A with the doubled workspace ceiling; the
        // IFNet's ONNX key carries its baked scale list
        s.gm = {
            {"gfeat", "featurenet", "featurenet", 1, {{"x", {1, 3, s.ph, s.pw}}}},
            {"gflow",
             lkGmFlowA(gh, gw),
             lkGmFlowA(gh, gw),
             2,
             {{"map4", {2, 128, gh / 4, gw / 4}},
              {"map8", {2, 128, gh / 8, gw / 8}},
              {"img0", {1, 3, gh, gw}},
              {"img1", {1, 3, gh, gw}},
              {"prop", {2, q8, 2}}}},
            {"gmetric",
             "metricnet",
             "metricnet",
             1,
             {{"i0", {1, 3, hh, hw}}, {"i1", {1, 3, hh, hw}}, {"f01", {1, 2, hh, hw}}, {"f10", {1, 2, hh, hw}}}},
            {"gifnet", "ifnet", "ifnet_sl8-4-2-1", 1, {{"x", {1, 6, hh, hw}}, {"timestep", {1, 1, 1, 1}}}},
            {"gfusion",
             "fusionnet",
             "fusionnet",
             1,
             {{"a", {1, 9, hh, hw}},
              {"b", {1, 128, hh, hw}},
              {"c", {1, 256, hh / 2, hw / 2}},
              {"d", {1, 384, hh / 4, hw / 4}}}},
            {"gflowb",
             "gmflow_bidir_b",
             "gmflow_bidir_b",
             1,
             {{"corr", {2, q4, K}},
              {"img0", {1, 3, gh, gw}},
              {"img1", {1, 3, gh, gw}},
              {"win", {2, q4, K, 2}},
              {"valid", {2, q4, K}},
              {"flow0", {2, 2, gh / 4, gw / 4}},
              {"feat", {2, 128, gh / 4, gw / 4}},
              {"flow1", {2, 2, gh / 4, gw / 4}}},
             2},
            {"gbone", "gmflow_backbone", "gmflow_backbone", 1, {{"img0", {1, 3, gh, gw}}}},
            {"gflowm",
             lkGmFlowM(gh, gw),
             lkGmFlowM(gh, gw),
             1,
             {{"map4", {2, 128, gh / 4, gw / 4}},
              {"map8", {2, 128, gh / 8, gw / 8}},
              {"img0", {1, 3, gh, gw}},
              {"img1", {1, 3, gh, gw}}}},
        };
        // on the fusionnet jit; the `|0x0` tail is a legacy flow-grid field, kept so existing
        // warm markers still match
        sprintf_s(key, "%dx%d|0x0", s.ph, s.pw);
        s.warmKey = key;
        return true;
    }
    s.drba = s.backend == "rifedrba";
    const std::string e = lkEnv("SMV_RIFE_FLOW_SCALE");
    if (!e.empty())
        s.fs = strtod(e.c_str(), nullptr);
    s.fs =
        (std::min)(1.0,
                   (std::max)(0.25, std::pow(2.0, std::nearbyint(std::log2((std::min)(1.0, (std::max)(0.25, s.fs)))))));
    const bool safePad = lkEnv(g_offline ? "SMV_RIFE_SAFEPAD" : "SMV_LIVE_SAFEPAD") == "1";
    if (safePad)
    {
        s.pw = (std::max)(s.pw, 1152);
        s.ph = (std::max)(s.ph, 640);
    }
    s.iph = s.ph;
    s.ipw = s.pw;
    // RIFE's two domains live (rife / blend / rifedrba): the IFNet, its encode and DRBA's block0 find
    // the motion on the finished picture at the working size, or at the capture size when the working
    // size enlarges it (Fill at DLAA: an enlarged picture costs RIFE its motion); the warp + blend runs
    // on the pictures at the working size. SMV_RIFE_TWO_DOMAIN=0: the IFNet reads the pictures.
    if (!g_offline && (s.backend == "rife" || s.backend == "blend" || s.drba) && lkEnv("SMV_RIFE_TWO_DOMAIN") != "0")
    {
        const bool up = s.mw > s.cw || s.mh > s.ch;
        s.mmw = up ? s.cw : s.mw;
        s.mmh = up ? s.ch : s.mh;
        s.mph = (s.mmh + 63) / 64 * 64;
        s.mpw = (s.mmw + 63) / 64 * 64;
        if (safePad)
        {
            s.mpw = (std::max)(s.mpw, 1152);
            s.mph = (std::max)(s.mph, 640);
        }
        s.iph = s.mph;
        s.ipw = s.mpw;
    }
    s.scaleTag = lkG(16 / s.fs) + "-" + lkG(8 / s.fs) + "-" + lkG(4 / s.fs) + "-" + lkG(2 / s.fs) + "-" + lkG(1 / s.fs);
    s.x = {1, 6, s.iph, s.ipw};
    s.f = {1, 16, s.iph, s.ipw};
    sprintf_s(key, "%dx%d", s.iph, s.ipw);
    s.warmKey = key;
    return true;
}

// the RIFE family's engine base names (they carry the weights tag)
static std::string lkIfnetBase(const LkSession& s, const LookupTags& t)
{
    return "rife_ifnet_" + t.r + "_" + s.scaleTag;
}
static std::string lkEncodeBase(const LkSession& s, const LookupTags& t)
{
    (void)s;
    return "rife_encode_" + t.r;
}
static std::string lkBlock0Base(const LkSession& s, const LookupTags& t)
{
    return "rife_block0_" + t.r + "_" + lkG(16 / s.fs);
}
// the IFNet input set at a timestep batch b
static std::vector<LkShape> lkIfnetSet(const LkSession& s, int b)
{
    return {{"x", s.x}, {"timestep", {b, 1, s.iph, s.ipw}}, {"f0", s.f}, {"f1", s.f}};
}
static std::vector<LkShape> lkBlock0Set(const LkSession& s)
{
    const std::vector<int64_t> x = {1, 3, s.iph, s.ipw}, f = {1, 16, s.iph, s.ipw};
    return {{"img0", x}, {"img1", x}, {"f0", f}, {"f1", f}};
}

// one TensorRT runtime for every lookup and build (deserialize checks and builds never overlap:
// nativeHandoff waits for a running build first); process lifetime, like the engines' own g_res.rt
static nvinfer1::IRuntime* lkRuntime()
{
    static nvinfer1::IRuntime* rt = nvinfer1::createInferRuntime(g_nativeTrtLogger);
    return rt;
}

// the session's handoff answer (the NATIVE-PATH lines and the LIVE READY line nativeHandoff
// parses) when its engines exist and were warmed, or false
static bool lkBaseHandoff(const std::wstring& script, const std::wstring& backendW, uint32_t capW, uint32_t capH,
                          std::vector<std::string>& lines)
{
    if (g_offline)
        return false; // the offline host gets its engine paths from render.py
    LkSession s;
    if (!lkSession(script, backendW, capW, capH, s))
        return false;
    const std::string& backend = s.backend;
    const bool geoOnly = backend == "nvof" || backend == "fruc" || backend == "fsrfg"; // no engine: geometry only
    if (!geoOnly && backend != "rife" && backend != "blend" && backend != "echo" && backend != "gmfss" &&
        backend != "rifedrba")
        return false;
    const std::string& engDir = s.engDir;
    const std::string& cacheDir = s.cacheDir;
    const double imgScale = s.imgScale;
    const int cw = s.cw, ch = s.ch, mw = s.mw, mh = s.mh;
    const int outW = W ? (int)W : cw, outH = W ? (int)H : ch;
    auto fitRect = [&](int w, int h, int& dw, int& dh, int& x0, int& y0) {
        lkFitRect(w, h, outW, outH, dw, dh, x0, y0);
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
        fitRect(mw, mh, dw, dh, x0, y0);
        lines.push_back("NATIVE-PATH cache=" + cacheDir);
        lines.push_back("LIVE READY native=1 " + geo(mh, mw, mw, mh, dw, dh, x0, y0) + " scale=" + lkF4(imgScale) +
                        " batch=0 batchpad=0 effects=0 restore=0 engine=none fast=1");
        return true;
    }
    if (geoOnly)
    {
        const int gw = mw, gh = mh;
        int dw, dh, x0, y0;
        fitRect(gw, gh, dw, dh, x0, y0);
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
            // Optical Flow engine take any size from 32x32)
            std::string gdir = lkEnv("SMV_FSRFG_DIR");
            if (gdir.empty())
                gdir = engDir + "\\fsrfg";
            if (!lkFile(gdir + "\\smv_fsrfg_bridge.dll"))
            {
                LOG("native: AMD FSR frame generation needs smv_fsrfg_bridge.dll in %s\n", gdir.c_str());
                return false;
            }
            lines.push_back("NATIVE-PATH cache=" + cacheDir);
            lines.push_back("NATIVE-PATH fsrfg=" + gdir);
            lines.push_back("LIVE READY native=1 " + geo(gh, gw, gw, gh, dw, dh, x0, y0) + tail);
            return true;
        }
        // fruc: the bridge folder (SMV_NVOFFRUC_DIR as nvoffruc.py reads it) and its three DLLs
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
    const LookupTags& t = lkTags(s.engDirW);
    if (!t.ok)
    {
        LOG("native: host lookup could not read the engine name tags\n");
        return false;
    }
    nvinfer1::IRuntime* rt = lkRuntime();
    if (!rt)
        return false;
    const int ph = s.ph, pw = s.pw;
    if (backend == "gmfss")
    {
        const int hh = s.hh, hw = s.hw, n = (int)s.gm.size();
        std::string path[kGmN], jit[kGmN];
        for (int i = 0; i < n; i++)
        {
            LkIo io;
            if (!lkFind(rt, cacheDir, s.gm[i].base, {lkGmNameSet(s.gm[i])}, nullptr, 0, 0, t, path[i], io))
                return false;
            jit[i] = lkJit(path[i]);
        }
        if (!lkWarm(jit[4], s.warmKey))
            return false;
        for (int i = 0; i < n; i++)
            if (i != 4 && !lkFile(jit[i]))
                return false;
        int dw, dh, x0, y0;
        fitRect(mw, mh, dw, dh, x0, y0);
        for (int i = 0; i < n; i++)
        {
            lines.push_back(std::string("NATIVE-PATH ") + s.gm[i].key + "=" + path[i]);
            lines.push_back(std::string("NATIVE-PATH ") + s.gm[i].key + "jit=" + jit[i]);
        }
        lines.push_back("NATIVE-PATH cache=" + cacheDir);
        char tail[256];
        sprintf_s(tail, " batch=0 batchpad=0 hh=%d hw=%d fh=%d fw=%d effects=0 restore=0 engine=gmfss fast=1", hh, hw,
                  s.fh, s.fw);
        lines.push_back("LIVE READY native=1 " + geo(ph, pw, mw, mh, dw, dh, x0, y0) + " scale=" + lkF4(imgScale) +
                        tail);
        return true;
    }
    // rife / blend / rifedrba: the unbatched IFNet, one in-between frame a call (the class offline runs
    // for a single tween): faster than the batched class at 1080p, within a few percent of it at 540p and
    // below, at a fraction of its video memory and without the wait for a batch before a pair's first frame
    const bool drba = s.drba;
    const int batch = 0;
    std::string ipath, epath;
    LkIo iio, eio;
    if (!lkFind(rt, cacheDir, lkIfnetBase(s, t), {lkIfnetSet(s, 1)}, nullptr, 1, batch, t, ipath, iio))
        return false;
    if (!lkRifeIo(iio, true))
        return false;
    const std::string ijit = lkJit(ipath);
    if (!lkWarm(ijit, s.warmKey))
        return false;
    if (!lkFind(rt, cacheDir, lkEncodeBase(s, t), {{{"img", {1, 3, s.iph, s.ipw}}}}, nullptr, 0, 0, t, epath, eio))
        return false;
    std::string bpath, bjit;
    if (drba)
    {
        LkIo bio;
        if (!lkFind(rt, cacheDir, lkBlock0Base(s, t), {lkBlock0Set(s)}, nullptr, 0, 0, t, bpath, bio))
            return false;
        if (!lkRifeIo(bio, false))
            return false;
        bjit = lkJit(bpath);
        if (!lkWarm(bjit, s.warmKey))
            return false;
    }
    int dw, dh, x0, y0;
    fitRect(mw, mh, dw, dh, x0, y0);
    if (drba)
    {
        lines.push_back("NATIVE-PATH block0=" + bpath);
        lines.push_back("NATIVE-PATH block0jit=" + bjit);
    }
    lines.push_back("NATIVE-PATH ifnet=" + ipath);
    lines.push_back("NATIVE-PATH encode=" + epath);
    lines.push_back("NATIVE-PATH jit=" + ijit);
    lines.push_back("NATIVE-PATH ejit=" + lkJit(epath));
    const std::string en = eio.outs.empty() ? "?" : eio.outs[0].first, ein = eio.ins.empty() ? "?" : eio.ins[0].first;
    const std::string ed = eio.outs.empty() ? "?" : eio.outs[0].second, eid = eio.ins.empty() ? "?" : eio.ins[0].second;
    char tail[512], motion[96] = "";
    if (s.mph)
        sprintf_s(motion, " mph=%d mpw=%d motw=%d moth=%d", s.mph, s.mpw, s.mmw, s.mmh);
    sprintf_s(
        tail,
        " batch=%d batchpad=%d xdtype=%s tdtype=%s fdtype=%s outdtype=%s ename=%s einame=%s edtype=%s eidtype=%s effects=0 fast=1%s%s",
        batch, 0, lkDt(iio, true, "x").c_str(), lkDt(iio, true, "timestep").c_str(), lkDt(iio, true, "f0").c_str(),
        lkDt(iio, false, "flow").c_str(), en.c_str(), ein.c_str(), ed.c_str(), eid.c_str(), motion,
        drba ? " engine=drba lag=1" : "");
    lines.push_back("LIVE READY native=1 " + geo(ph, pw, mw, mh, dw, dh, x0, y0) + " scale=" + lkF4(imgScale) + tail);
    return true;
}

// the Restore engine for a model size, as trt_runtime.RestoreEngine names it: restore_<md5[:8] of
// the weights>, input x (1,3,h,w) at the unpadded model size (the fit's w x h)
static std::string lkRestorePath(const std::string& cacheDir, const LookupTags& t, int w, int h)
{
    return cacheDir + "\\" + lkEngineName("restore_" + t.rest, {{"x", {1, 3, h, w}}}, nullptr, 0, 0, t) + ".engine";
}

static int lkField(const std::string& line, const char* key)
{
    const std::string k = std::string(" ") + key + "=";
    const size_t p = line.find(k);
    return p == std::string::npos ? -1 : atoi(line.c_str() + p + k.size());
}

// The session's full handoff answer: the base answer plus, for live Restore (every backend:
// the host restores the captured frame before any model reads it), the Restore engine and its
// jit once both exist (nativeLocalBuild creates them; the engine has no warm marker).
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
    const LookupTags& t = lkTags(s.engDirW);
    if (!t.ok || t.rest.empty())
    {
        lines.clear();
        return false;
    }
    // Restore reads the captured frame, so its engine has the capture size
    const std::string rpath = lkRestorePath(s.cacheDir, t, (int)capW, (int)capH);
    const std::string rjit = lkJit(rpath);
    if (!lkFile(rpath) || !lkFile(rjit))
    {
        lines.clear();
        return false;
    }
    const std::string ready = lines.back();
    lines.pop_back();
    lines.push_back("NATIVE-PATH restore=" + rpath);
    lines.push_back("NATIVE-PATH rjit=" + rjit);
    lines.push_back(ready);
    return true;
}

// ---- cold engine build inside the host ------------------------------------------------------
// A window size with no warm engines: the host builds them itself from the size-free ONNX in
// engine\onnx (strongly typed, workspace
// SMV_TRT_WORKSPACE_GB (8) x the per-graph multiplier, optimization level 5, one profile pinning
// every input to its shape, only a declared batch axis spanning), writes the engine file only
// after it deserializes, then warms every shape the session will run with a FRESH runtime cache
// (EAGER specialization, so the saved kernels are the specialized ones) and writes that cache as
// the engine's .jit and the warm key. After the build the host lookup runs again and must
// hit.

class LkBuildLogger : public nvinfer1::ILogger
{
  public:
    std::string errs;
    void log(Severity s, const char* msg) noexcept override
    {
        if (s <= Severity::kWARNING)
            LOG("[TRT] %s\n", msg);
        if (s <= Severity::kERROR)
        {
            errs += msg;
            errs += "\n";
        }
    }
};

static bool lkOom(const std::string& s)
{
    std::string l = s;
    for (auto& c : l)
        c = (char)tolower((unsigned char)c);
    return l.find("out of memory") != std::string::npos || l.find("outofmemory") != std::string::npos ||
           l.find("not enough gpu memory") != std::string::npos;
}

static bool lkWriteFile(const std::string& path, const void* data, size_t n)
{
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary);
        f.write(static_cast<const char*>(data), (std::streamsize)n);
        if (!f)
            return false;
    }
    return MoveFileExA(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING) != FALSE;
}

// The cap on TensorRT-RTX's auxiliary streams for the GMFSS nets where its heuristic (3 on fusionnet and metricnet, 1
// on gmflow_bidir_b and gmflow_backbone) is slower or larger than a lower cap, outputs identical: gmflow_bidir_b 0
// (x0.73 at 960x544, less memory), metricnet 0 (x0.95, 15 to 56 MiB more), gmflow_backbone 0 (x0.97 to 1.00, less
// memory), fusionnet 2 (x0.97 to 0.99, 7 to 10 % less memory; 0 or 1 is 1 to 2 % slower). -1 = the heuristic's choice.
// trt_lookup.HOST_BUILD names this table, so a change empties the engine cache once.
static int lkAuxStreamCap(const std::string& onnx)
{
    static const struct
    {
        const char* prefix;
        int cap;
    } kCaps[] = {{"gmflow_bidir_b_", 0}, {"metricnet_", 0}, {"gmflow_backbone_", 0}, {"fusionnet_", 2}};
    const size_t sl = onnx.find_last_of("\\/");
    const std::string base = sl == std::string::npos ? onnx : onnx.substr(sl + 1);
    for (const auto& c : kCaps)
        if (!base.compare(0, strlen(c.prefix), c.prefix))
            return c.cap;
    return -1;
}

// one engine from a size-free ONNX; *oom = the failure was memory-shaped (python's .nofit rule:
// a null build or an out-of-memory message)
static bool lkBuild(nvinfer1::IRuntime* rt, const std::string& onnx, const std::string& out,
                    const std::vector<LkShape>& set, const char* dynInput, int dynLo, int dynHi, int wsMult, bool* oom)
{
    *oom = false;
    LkBuildLogger lg;
    const auto t0 = std::chrono::steady_clock::now();
    std::unique_ptr<nvinfer1::IBuilder> b(nvinfer1::createInferBuilder(lg));
    if (!b)
        return false;
    const auto flags = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    std::unique_ptr<nvinfer1::INetworkDefinition> net(b->createNetworkV2(flags));
    if (!net)
        return false;
    std::unique_ptr<nvonnxparser::IParser> p(nvonnxparser::createParser(*net, lg));
    if (!p || !p->parseFromFile(onnx.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
    {
        if (p)
            for (int i = 0; i < p->getNbErrors(); i++)
                LOG("native: onnx parse: %s\n", p->getError(i)->desc());
        return false;
    }
    std::unique_ptr<nvinfer1::IBuilderConfig> cfg(b->createBuilderConfig());
    nvinfer1::IOptimizationProfile* prof = b->createOptimizationProfile();
    if (!cfg || !prof)
        return false;
    double ws = 8.0;
    const std::string wse = lkEnv("SMV_TRT_WORKSPACE_GB");
    if (!wse.empty())
    {
        const double v = strtod(wse.c_str(), nullptr);
        if (v > 0)
            ws = v;
    }
    cfg->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, (size_t)(ws * (1ull << 30)) * wsMult);
    cfg->setBuilderOptimizationLevel(5);
    const int auxCap = lkAuxStreamCap(onnx);
    if (auxCap >= 0)
        cfg->setMaxAuxStreams(auxCap);
    for (int i = 0; i < net->getNbInputs(); i++)
    {
        nvinfer1::ITensor* t = net->getInput(i);
        const LkShape* s = nullptr;
        for (const auto& in : set)
            if (!strcmp(in.name, t->getName()))
                s = &in;
        if (!s || (int)s->dims.size() != t->getDimensions().nbDims)
        {
            LOG("native: onnx input %s does not match the engine class\n", t->getName());
            return false;
        }
        nvinfer1::Dims lo, hi;
        lo.nbDims = hi.nbDims = (int)s->dims.size();
        for (int k = 0; k < lo.nbDims; k++)
            lo.d[k] = hi.d[k] = s->dims[k];
        if (dynInput && !strcmp(s->name, dynInput))
        {
            lo.d[0] = dynLo;
            hi.d[0] = dynHi;
        }
        if (!prof->setDimensions(t->getName(), nvinfer1::OptProfileSelector::kMIN, lo) ||
            !prof->setDimensions(t->getName(), nvinfer1::OptProfileSelector::kOPT, hi) ||
            !prof->setDimensions(t->getName(), nvinfer1::OptProfileSelector::kMAX, hi))
            return false;
    }
    cfg->addOptimizationProfile(prof);
    std::unique_ptr<nvinfer1::IHostMemory> ser(b->buildSerializedNetwork(*net, *cfg));
    if (!ser)
    {
        *oom = true; // python: "returned None" counts as memory-shaped
        LOG("native: engine build returned nothing for %s\n", out.c_str());
        return false;
    }
    // deserialize BEFORE the file reaches the cache: a file that cannot load must not be left
    std::unique_ptr<nvinfer1::ICudaEngine> eng(rt->deserializeCudaEngine(ser->data(), ser->size()));
    if (!eng)
    {
        *oom = lkOom(lg.errs);
        return false;
    }
    eng.reset();
    if (!lkWriteFile(out, ser->data(), ser->size()))
    {
        LOG("native: cannot write %s\n", out.c_str());
        return false;
    }
    LOG("native: built %s from %s in %.1fs\n", out.substr(out.find_last_of("\\/") + 1).c_str(),
        onnx.substr(onnx.find_last_of("\\/") + 1).c_str(),
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
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

static size_t lkDtypeBytes(nvinfer1::DataType d)
{
    switch (d)
    {
    case nvinfer1::DataType::kHALF:
    case nvinfer1::DataType::kBF16:
        return 2;
    case nvinfer1::DataType::kINT64:
        return 8;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kUINT8:
    case nvinfer1::DataType::kINT8:
        return 1;
    default:
        return 4;
    }
}

// warm an engine for every input shape set with a fresh runtime cache (the existing .jit file
// merged first, as python's context does), then write the cache as <engine>[.kind].jit and add
// the warm key (none = the file only). Inputs are zeros: the kernels depend on shapes, not data.
static bool lkWarmEngine(nvinfer1::IRuntime* rt, const std::string& enginePath,
                         const std::vector<std::vector<LkShape>>& sets, const std::string& warmKey, cudaStream_t st,
                         bool* oom)
{
    *oom = false;
    std::vector<char> blob;
    if (!lkReadFile(enginePath, blob))
        return false;
    std::unique_ptr<nvinfer1::ICudaEngine> eng(rt->deserializeCudaEngine(blob.data(), blob.size()));
    if (!eng)
        return false;
    std::unique_ptr<nvinfer1::IRuntimeConfig> cfg(eng->createRuntimeConfig());
    if (!cfg)
        return false;
    std::unique_ptr<nvinfer1::IRuntimeCache> cache(cfg->createRuntimeCache());
    if (!cache)
        return false;
    const std::string jit = lkJit(enginePath);
    std::vector<char> cb;
    if (lkReadFile(jit, cb) && !cache->deserialize(cb.data(), cb.size()))
    {
        // python's rule: a rejected cache starts fresh and nothing in it is warm any more
        DeleteFileA((jit + ".warm").c_str());
        cache.reset(cfg->createRuntimeCache());
        if (!cache)
            return false;
    }
    cfg->setRuntimeCache(*cache);
    cfg->setDynamicShapesKernelSpecializationStrategy(nvinfer1::DynamicShapesKernelSpecializationStrategy::kEAGER);
    std::unique_ptr<nvinfer1::IExecutionContext> ctx(eng->createExecutionContext(cfg.get()));
    if (!ctx)
    {
        *oom = true;
        LOG("native: warm-up context failed for %s\n", enginePath.c_str());
        return false;
    }
    std::vector<void*> bufs;
    bool ok = true;
    for (const auto& set : sets)
    {
        for (const auto& in : set)
        {
            nvinfer1::Dims d;
            d.nbDims = (int)in.dims.size();
            for (int k = 0; k < d.nbDims; k++)
                d.d[k] = in.dims[k];
            if (!ctx->setInputShape(in.name, d))
            {
                ok = false;
                break;
            }
        }
        for (int i = 0; ok && i < eng->getNbIOTensors(); i++)
        {
            const char* n = eng->getIOTensorName(i);
            const nvinfer1::Dims d = ctx->getTensorShape(n);
            size_t bytes = lkDtypeBytes(eng->getTensorDataType(n));
            for (int k = 0; k < d.nbDims; k++)
                bytes *= (size_t)(d.d[k] > 0 ? d.d[k] : 1);
            void* ptr = nullptr;
            if (cudaMalloc(&ptr, bytes) != cudaSuccess)
            {
                ok = false;
                *oom = true;
                break;
            }
            bufs.push_back(ptr);
            cudaMemsetAsync(ptr, 0, bytes, st);
            ctx->setTensorAddress(n, ptr);
        }
        ok = ok && ctx->enqueueV3(st) && cudaStreamSynchronize(st) == cudaSuccess;
        for (void* ptr : bufs)
            cudaFree(ptr);
        bufs.clear();
        if (!ok)
            break;
    }
    if (!ok)
    {
        LOG("native: warm-up enqueue failed for %s\n", enginePath.c_str());
        return false;
    }
    ctx.reset();
    std::unique_ptr<nvinfer1::IHostMemory> ser(cache->serialize());
    if (!ser || !lkWriteFile(jit, ser->data(), ser->size()))
    {
        LOG("native: cannot write %s\n", jit.c_str());
        return false;
    }
    if (!warmKey.empty())
    {
        // trt_lookup.mark_warm: the key set, sorted, one per line
        std::vector<std::string> keys;
        {
            std::ifstream f(jit + ".warm");
            std::string ln;
            while (std::getline(f, ln))
            {
                while (!ln.empty() && (ln.back() == '\r' || ln.back() == ' '))
                    ln.pop_back();
                if (!ln.empty())
                    keys.push_back(ln);
            }
        }
        keys.push_back(warmKey);
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
        std::string txt;
        for (auto& k : keys)
            txt += k + "\n";
        if (!lkWriteFile(jit + ".warm", txt.data(), txt.size()))
            return false;
    }
    return true;
}

// trt_lookup.onnx_path(): <key>_<w>.onnx; onnx_export.py exports every graph again whenever its export stamp
// changes, so the file under this name is always the current graph
static std::string lkOnnxPath(const std::string& engDir, const std::string& key, const LookupTags& t)
{
    const std::string env = lkEnv("SMV_ONNX_DIR");
    return (env.empty() ? engDir + "\\onnx" : env) + "\\" + key + "_" + t.w + ".onnx";
}

// a CUDA stream for a build on the calling thread, on the device the sessions run on
static bool lkBuildStream(cudaStream_t& st)
{
    st = nullptr;
    NCHK(cudaSetDevice(0), "cudaSetDevice (build)");
    NCHK(cudaFree(nullptr), "cuda context (build)");
    NCHK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "cudaStreamCreate (build)");
    return true;
}

// one engine of a build: the file when it exists, else built from its ONNX in engine\onnx, then
// warmed for every shape set (a warm key, when given, marks the jit as warmed for the session's
// size). *oom = the failure was memory-shaped.
static bool lkEnsure(nvinfer1::IRuntime* rt, const LkSession& s, const LookupTags& t, const std::string& path,
                     const std::string& onnxKey, const std::vector<LkShape>& set, const char* dynInput, int dynLo,
                     int dynHi, int wsMult, const std::vector<std::vector<LkShape>>& warmSets,
                     const std::string& warmKey, cudaStream_t st, bool* oom)
{
    *oom = false;
    if (!lkFile(path))
    {
        const std::string onnx = lkOnnxPath(s.engDir, onnxKey, t);
        if (!lkFile(onnx))
        {
            LOG("native: %s is missing (node scripts/export-onnx.js writes it)\n", onnx.c_str());
            return false;
        }
        if (!lkBuild(rt, onnx, path, set, dynInput, dynLo, dynHi, wsMult, oom))
            return false;
    }
    return lkWarmEngine(rt, path, warmSets, warmKey, st, oom);
}

// the RIFE family: the unbatched IFNet, the encoder, and DRBA's block0 flow engine
static bool lkBuildRife(nvinfer1::IRuntime* rt, const LkSession& s, const LookupTags& t, cudaStream_t st)
{
    auto path = [&](const std::string& base, const std::vector<LkShape>& set) {
        return s.cacheDir + "\\" + lkEngineName(base, set, nullptr, 0, 0, t) + ".engine";
    };
    bool oom = false;
    const std::string ibase = lkIfnetBase(s, t);
    const std::vector<LkShape> uset = lkIfnetSet(s, 1);
    if (!lkEnsure(rt, s, t, path(ibase, uset), ibase, uset, nullptr, 0, 0, 1, {uset}, s.warmKey, st, &oom))
        return false;
    const std::vector<LkShape> eset = {{"img", {1, 3, s.iph, s.ipw}}};
    const std::string ebase = lkEncodeBase(s, t);
    if (!lkEnsure(rt, s, t, path(ebase, eset), ebase, eset, nullptr, 0, 0, 1, {eset}, std::string(), st, &oom))
        return false;
    if (s.drba)
    {
        const std::vector<LkShape> b0set = lkBlock0Set(s);
        const std::string b0base = lkBlock0Base(s, t);
        if (!lkEnsure(rt, s, t, path(b0base, b0set), b0base, b0set, nullptr, 0, 0, 1, {b0set}, s.warmKey, st, &oom))
            return false;
    }
    return true;
}

// build + warm what lkBaseHandoff needs for this session; true = worth a second lookup
static bool lkBuildBackend(const std::wstring& script, const std::wstring& backendW, uint32_t capW, uint32_t capH)
{
    if (g_offline)
        return false;
    LkSession s;
    if (!lkSession(script, backendW, capW, capH, s))
        return false;
    const bool gmfss = s.backend == "gmfss";
    if (s.backend != "rife" && s.backend != "blend" && !gmfss && !s.drba)
        return false;
    if (!nativeLoadDlls(script) || !g_onnxParserOk)
        return false;
    const LookupTags& t = lkTags(s.engDirW);
    if (!t.ok)
        return false;
    const auto t0 = std::chrono::steady_clock::now();
    cudaStream_t st = nullptr;
    if (!lkBuildStream(st))
        return false;
    nvinfer1::IRuntime* rt = lkRuntime();
    bool done = rt != nullptr;
    if (done && gmfss)
    {
        // the warm key sits on fusionnet's jit
        for (size_t i = 0; done && i < s.gm.size(); i++)
        {
            bool oom = false;
            const LkGmNet& n = s.gm[i];
            const std::string path =
                s.cacheDir + "\\" + lkEngineName(n.base, lkGmNameSet(n), nullptr, 0, 0, t) + ".engine";
            done = lkEnsure(rt, s, t, path, n.onnx, n.set, nullptr, 0, 0, n.ws, {n.set},
                            !strcmp(n.key, "gfusion") ? s.warmKey : std::string(), st, &oom);
        }
    }
    else if (done)
        done = lkBuildRife(rt, s, t, st);
    cudaStreamDestroy(st);
    LOG("native: host %sengine build for %dx%d %s (%.1fs)\n", gmfss ? "GMFSS " : "", s.pw, s.ph,
        done ? "done" : "failed", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return done;
}

// the cold-start build: the backend's engines when its warm answer misses (echo, nvof and fruc
// have none), then for live Restore the Real-ESRGAN engine from engine\onnx (restore_<hash>_dth:
// fp16 input) at the model size the answer names, warmed once so its .jit exists. A backend that
// is already warm is never rewarmed. true = worth a second lookup.
static bool lkEnsureRestore(const std::wstring& script, const LkSession& s, int w, int h, std::string& rpath,
                            std::string& rjit);
static bool nativeLocalBuild(const std::wstring& script, const std::wstring& backendW, uint32_t capW, uint32_t capH)
{
    if (g_offline)
        return false;
    std::vector<std::string> base;
    if (!lkBaseHandoff(script, backendW, capW, capH, base))
    {
        if (!lkBuildBackend(script, backendW, capW, capH))
            return false;
        base.clear();
        if (!lkBaseHandoff(script, backendW, capW, capH, base))
            return false;
    }
    if (!g_restore)
        return true;
    LkSession s;
    const int w = lkField(base.back(), "w"), h = lkField(base.back(), "h");
    if (!lkSession(script, backendW, capW, capH, s) || w <= 0 || h <= 0)
        return false;
    std::string rpath, rjit;
    return lkEnsureRestore(script, s, (int)capW, (int)capH, rpath, rjit); // the captured frame's size
}

// the Real-ESRGAN Restore engine at its input size w x h (restore_<hash>_dth: fp16 input; live the
// captured frame, offline the decoded one) from engine\onnx when it is missing, warmed once so its
// .jit exists; the live cold build and the offline host share it. true = both files exist, paths
// in rpath / rjit.
static bool lkEnsureRestore(const std::wstring& script, const LkSession& s, int w, int h, std::string& rpath,
                            std::string& rjit)
{
    if (!nativeLoadDlls(script))
        return false;
    const LookupTags& t = lkTags(s.engDirW);
    if (!t.ok || t.rest.empty())
        return false;
    rpath = lkRestorePath(s.cacheDir, t, w, h);
    rjit = lkJit(rpath);
    if (lkFile(rpath) && lkFile(rjit))
        return true;
    if (!g_onnxParserOk)
    {
        LOG("native: the ONNX parser DLL is missing, cannot build the Restore engine\n");
        return false;
    }
    const auto t0 = std::chrono::steady_clock::now();
    cudaStream_t st = nullptr;
    if (!lkBuildStream(st))
        return false;
    nvinfer1::IRuntime* rt = lkRuntime();
    bool oom = false;
    const std::vector<LkShape> rset = {{"x", {1, 3, h, w}}};
    const bool done = rt && lkEnsure(rt, s, t, rpath, "restore_" + t.rest + "_dth", rset, nullptr, 0, 0, 1, {rset},
                                     std::string(), st, &oom);
    cudaStreamDestroy(st);
    LOG("native: host Restore engine build for %dx%d %s (%.1fs)\n", w, h, done ? "done" : "failed",
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return done;
}

static uint64_t nativeVideoMemoryRoom();

// ---- offline RIFE engines -------------------------------------------------------------------
// The offline host finds, builds and warms its own IFNet + Head engines. The classes are the ones
// trt_lookup._rife_ifnet_spec names for offline: x2 = the unbatched IFNet, xN = the FIXED batch
// class `_b{N-1}` (every group exactly full), pinned to the /64 pad of the source. The per-B
// ONNX never ships, so a fixed class is built from the shipped `_bd8` graph with its batch axis
// pinned to B (bit-exact with an engine built from a per-B `_b{B}` export). A batched
// build that runs out of memory is marked `.nofit` and the unbatched engine serves.
// The batched class is also taken only while its extra timesteps fit in the free video memory
// beside the render's other needs (others, nativeOfflineNeed): the unbatched engine makes the
// same in-between frames in more calls, where a spill into system memory slows the whole PC.
struct OfflineEngines
{
    std::string ifnet, encode, jit;
    int ph = 0, pw = 0, batch = 1;
};

// What each timestep of a batched IFNet class takes beyond the first, in MiB a padded megapixel: its activations and
// buffers at the IFNet's frame, kIfnetStepMp (offline x4 renders: 189 / 857 MiB at 854x480 / 1920x1080 with the
// working size at the decode, less their fp16 tween's 6 bytes a pixel), and its fp32 tween at the pictures' frame,
// kTweenStepMp (3 channels x 4 bytes a pixel)
constexpr double kIfnetStepMp = 405.0, kTweenStepMp = 11.4;

// picW x picH = the pictures' frame (the working size: RIFE's last step and its tweens), 0 = the IFNet's w x h
static bool lkOfflineRife(const std::wstring& script, int w, int h, int multi, OfflineEngines& o, uint64_t others = 0,
                          int picW = 0, int picH = 0)
{
    LkSession s;
    if (!lkSession(script, L"rife", (uint32_t)w, (uint32_t)h, s))
        return false;
    if (!nativeLoadDlls(script))
        return false;
    const LookupTags& t = lkTags(s.engDirW);
    if (!t.ok)
    {
        LOG("offline: could not read the engine name tags\n");
        return false;
    }
    nvinfer1::IRuntime* rt = lkRuntime();
    if (!rt)
        return false;
    const auto t0 = std::chrono::steady_clock::now();
    auto enginePath = [&](const std::string& base, const std::vector<LkShape>& set) {
        return s.cacheDir + "\\" + lkEngineName(base, set, nullptr, 0, 0, t) + ".engine";
    };
    // one class: its engine exists and is warm at this size, else build (when missing) and warm
    auto ensure = [&](const std::string& base, const std::string& onnxKey, const std::vector<LkShape>& set,
                      const std::string& warmKey, cudaStream_t& st, bool* oom) -> bool {
        *oom = false;
        const std::string p = enginePath(base, set);
        if (lkFile(p) && (warmKey.empty() ? lkFile(lkJit(p)) : lkWarm(lkJit(p), warmKey)))
            return true;
        if (!lkFile(p) && !g_onnxParserOk)
        {
            LOG("offline: the ONNX parser DLL is missing, cannot build %s\n", base.c_str());
            return false;
        }
        if (!st && !lkBuildStream(st))
            return false;
        return lkEnsure(rt, s, t, p, onnxKey, set, nullptr, 0, 0, 1, {set}, warmKey, st, oom);
    };
    cudaStream_t st = nullptr;
    const std::string ibase = lkIfnetBase(s, t);
    int B = lkEnv("SMV_RIFE_BATCH") == "1" ? 1 : multi - 1;
    const uint64_t room = B > 1 && others ? nativeVideoMemoryRoom() : 0;
    if (room)
    {
        const double mib = 1048576.0;
        const double picMp =
            picW ? (double)((picW + 63) / 64 * 64) * ((picH + 63) / 64 * 64) / 1e6 : (double)s.pw * s.ph / 1e6;
        const uint64_t extra = (uint64_t)((B - 1) * (kIfnetStepMp * s.pw * s.ph / 1e6 + kTweenStepMp * picMp) * mib);
        if (others + extra > room)
        {
            LOG("offline: video memory: the model runs 1 of %d in-between frames a call; a batch of %d needs about "
                "%.0f MiB more, %.0f MiB are free beside the render's other %.0f MiB\n",
                B, B, extra / mib, room > others ? (room - others) / mib : 0.0, others / mib);
            B = 1;
        }
    }
    bool ok = false, oom = false;
    std::string base = ibase;
    std::vector<LkShape> set = lkIfnetSet(s, 1);
    if (B > 1)
    {
        const std::string bbase = ibase + "_b" + std::to_string(B);
        const std::vector<LkShape> bset = lkIfnetSet(s, B);
        const std::string bpath = enginePath(bbase, bset);
        if (lkFile(bpath) || !lkFile(bpath + ".nofit"))
        {
            ok = ensure(bbase, ibase + "_bd8", bset, s.warmKey, st, &oom);
            if (ok)
            {
                base = bbase;
                set = bset;
            }
            else if (oom)
            {
                char why[160];
                sprintf_s(why, "%dx%d: the batched engine did not fit (offline host build)\n", s.pw, s.ph);
                if (!lkWriteFile(bpath + ".nofit", why, strlen(why)))
                    LOG("offline: cannot write %s.nofit, the next render retries the batched build\n", bpath.c_str());
                LOG("offline: batched engine did not fit at %dx%d, marked; using the unbatched engine\n", s.pw, s.ph);
            }
            else
            {
                if (st)
                    cudaStreamDestroy(st);
                return false;
            }
        }
        else
            LOG("offline: batched engine marked \"did not fit\" at %dx%d, using the unbatched engine\n", s.pw, s.ph);
    }
    if (base == ibase)
        ok = ensure(ibase, ibase, set, s.warmKey, st, &oom);
    const std::string ebase = lkEncodeBase(s, t);
    const std::vector<LkShape> eset = {{"img", {1, 3, s.ph, s.pw}}};
    ok = ok && ensure(ebase, ebase, eset, std::string(), st, &oom);
    const bool built = st != nullptr;
    if (st)
        cudaStreamDestroy(st);
    if (!ok)
    {
        LOG("offline: engine build for %dx%d failed\n", s.pw, s.ph);
        return false;
    }
    LkIo iio, eio;
    if (!lkFind(rt, s.cacheDir, base, {set}, nullptr, 0, 0, t, o.ifnet, iio) || !lkRifeIo(iio, true) ||
        !lkFind(rt, s.cacheDir, ebase, {eset}, nullptr, 0, 0, t, o.encode, eio))
    {
        LOG("offline: the engines at %dx%d do not match their class (delete them from the cache to rebuild)\n", s.pw,
            s.ph);
        return false;
    }
    o.jit = lkJit(o.ifnet);
    o.ph = s.ph;
    o.pw = s.pw;
    o.batch = base == ibase ? 1 : B;
    LOG("offline: engines for %dx%d %s%s (%.1fs)\n", s.pw, s.ph,
        o.batch > 1 ? ("batched B=" + std::to_string(o.batch)).c_str() : "single tween",
        built ? " built by the host" : " warm",
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return true;
}

// ---- offline GMFSS engines ------------------------------------------------------------------
// The live host's five-engine set (lkSession "gmfss": featurenet at the /64 pad, the fused bidir
// gmflow, metricnet, the GMFSS IFNet and fusionnet at the half), found warm or built and warmed from engine\onnx exactly as
// lkBuildBackend does for live, then handed to nr like the live handoff's NATIVE-PATH lines.
static bool lkOfflineGmfss(const std::wstring& script, int w, int h, NativeRife& nr)
{
    LkSession s;
    if (!lkSession(script, L"gmfss", (uint32_t)w, (uint32_t)h, s))
        return false;
    if (!nativeLoadDlls(script))
        return false;
    const LookupTags& t = lkTags(s.engDirW);
    if (!t.ok)
    {
        LOG("offline: could not read the engine name tags\n");
        return false;
    }
    nvinfer1::IRuntime* rt = lkRuntime();
    if (!rt)
        return false;
    const auto t0 = std::chrono::steady_clock::now();
    cudaStream_t st = nullptr;
    for (size_t i = 0; i < s.gm.size(); i++)
    {
        const LkGmNet& n = s.gm[i];
        const std::string p = s.cacheDir + "\\" + lkEngineName(n.base, lkGmNameSet(n), nullptr, 0, 0, t) + ".engine";
        // the warm key sits on fusionnet's jit, the others need their .jit
        const std::string wk = !strcmp(n.key, "gfusion") ? s.warmKey : std::string();
        if (lkFile(p) && (wk.empty() ? lkFile(lkJit(p)) : lkWarm(lkJit(p), wk)))
            continue;
        if (!lkFile(p) && !g_onnxParserOk)
        {
            LOG("offline: the ONNX parser DLL is missing, cannot build %s\n", n.base);
            return false;
        }
        bool oom = false;
        if ((!st && !lkBuildStream(st)) ||
            !lkEnsure(rt, s, t, p, n.onnx, n.set, nullptr, 0, 0, n.ws, {n.set}, wk, st, &oom))
        {
            if (st)
                cudaStreamDestroy(st);
            LOG("offline: GMFSS engine build for %dx%d failed%s\n", s.pw, s.ph, oom ? " (out of memory)" : "");
            return false;
        }
    }
    const bool built = st != nullptr;
    if (st)
        cudaStreamDestroy(st);
    for (size_t i = 0; i < s.gm.size(); i++)
    {
        LkIo io;
        if (!lkFind(rt, s.cacheDir, s.gm[i].base, {lkGmNameSet(s.gm[i])}, nullptr, 0, 0, t, nr.gmPath[i], io))
        {
            LOG("offline: the GMFSS engines at %dx%d do not match their class (delete them from the cache to rebuild)\n",
                s.pw, s.ph);
            return false;
        }
        nr.gmJit[i] = lkJit(nr.gmPath[i]);
    }
    nr.gmfss = true;
    nr.ph = s.ph;
    nr.pw = s.pw;
    nr.hh = s.hh;
    nr.hw = s.hw;
    nr.gfh = s.fh;
    nr.gfw = s.fw;
    LOG("offline: GMFSS engines for %dx%d%s (%.1fs)\n", s.pw, s.ph, built ? ", built by the host" : ", warm",
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return true;
}

// ---- offline DRBA block0 engine -------------------------------------------------------------
// DRBA's calc_flow engine (lkBlock0Base / lkBlock0Set, the live names) at the /64 pad of the
// source, warm or built + warmed from engine\onnx; the IFNet and encode come from lkOfflineRife
// (x2 = the unbatched class: DRBA enqueues one tween at a time with its own DRM map).
static bool lkOfflineBlock0(const std::wstring& script, int w, int h, std::string& path, std::string& jit)
{
    LkSession s;
    if (!lkSession(script, L"rifedrba", (uint32_t)w, (uint32_t)h, s))
        return false;
    if (!nativeLoadDlls(script))
        return false;
    const LookupTags& t = lkTags(s.engDirW);
    if (!t.ok)
    {
        LOG("offline: could not read the engine name tags\n");
        return false;
    }
    nvinfer1::IRuntime* rt = lkRuntime();
    if (!rt)
        return false;
    const std::string base = lkBlock0Base(s, t);
    const std::vector<LkShape> set = lkBlock0Set(s);
    const std::string p = s.cacheDir + "\\" + lkEngineName(base, set, nullptr, 0, 0, t) + ".engine";
    const bool warm = lkFile(p) && lkWarm(lkJit(p), s.warmKey);
    if (!warm)
    {
        if (!lkFile(p) && !g_onnxParserOk)
        {
            LOG("offline: the ONNX parser DLL is missing, cannot build %s\n", base.c_str());
            return false;
        }
        cudaStream_t st = nullptr;
        bool oom = false;
        const bool ok =
            lkBuildStream(st) && lkEnsure(rt, s, t, p, base, set, nullptr, 0, 0, 1, {set}, s.warmKey, st, &oom);
        if (st)
            cudaStreamDestroy(st);
        if (!ok)
        {
            LOG("offline: DRBA block0 engine build for %dx%d failed%s\n", s.pw, s.ph, oom ? " (out of memory)" : "");
            return false;
        }
    }
    LkIo io;
    if (!lkFind(rt, s.cacheDir, base, {set}, nullptr, 0, 0, t, path, io) || !lkRifeIo(io, false))
    {
        LOG("offline: the DRBA block0 engine at %dx%d does not match its class\n", s.pw, s.ph);
        return false;
    }
    jit = lkJit(path);
    LOG("offline: DRBA block0 engine for %dx%d %s\n", s.pw, s.ph, warm ? "warm" : "built by the host");
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

// offline AMD FSR frame generation: the bridge folder the live handoff names (SMV_FSRFG_DIR, else engine\fsrfg)
static bool lkOfflineFsrfg(const std::wstring& script, std::string& dir)
{
    dir = lkEnv("SMV_FSRFG_DIR");
    if (dir.empty())
    {
        std::wstring eng = script.substr(0, script.find_last_of(L"\\/"));
        for (auto& c : eng)
            if (c == L'/')
                c = L'\\';
        dir = wideToUtf8(eng) + "\\fsrfg";
    }
    if (!lkFile(dir + "\\smv_fsrfg_bridge.dll"))
    {
        LOG("offline: AMD FSR frame generation needs smv_fsrfg_bridge.dll in %s\n", dir.c_str());
        return false;
    }
    return true;
}

// The engine handoff: the session's geometry and engine paths, from the host's own lookup
// (nativeLocalHandoff), after a cold build on a worker thread when the lookup misses.
// Blocking (the caller is srv.start() on the loader thread or the early handoff thread).
// A build whose session was ended meanwhile (the hotkey toggled off, or the target window
// resized during the load) keeps running while the resident host stays alive
// (residentMain's stay rule counts it): its engines land in the cache, so the next start at
// that size is warm. A host that
// exits anyway ends the build with it; engine files are written whole or not at all
// (lkWriteFile). One builder at a time: a new handoff waits for a running one first.
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
    // size, DLSS mode or share, Auto's memory pick and floor and HDR mode) gets the previous answer without a lookup,
    // as long as the engine files still exist (the user may empty the cache folder by hand);
    // live Restore rides in the key: its engine path is a fact of a restore session only; DLSS 5 too: it raises a
    // small working size (liveWorkSizeFor)
    wchar_t key[1024];
    swprintf_s(key, L"%s|%s|%.2f|%d|%d|%d|%u|%u|%u|%u|%d|%d|%d|%.2f", script.c_str(), backend.c_str(), g_flowScale,
               g_dlssMode, g_liveAutoFit, g_liveAutoFloor, W, H, capW, capH, g_hdr ? 1 : 0, g_restore ? 1 : 0,
               g_dlssnr ? 1 : 0, g_gmFlow);
    bool factsOk = g_resident && g_res.haveFacts && g_res.handoffKey == key;
    if (factsOk)
    {
        const NativeRife& f = g_res.facts;
        if (f.gmfss)
        {
            for (int i = 0; i < kGmN; i++)
                factsOk = factsOk && fileExistsA(f.gmPath[i]);
        }
        else
            factsOk = fileExistsA(f.ifnetPath) && fileExistsA(f.encodePath);
        factsOk = factsOk && (f.restorePath.empty() || fileExistsA(f.restorePath));
        factsOk = factsOk && (!f.drba || fileExistsA(f.block0Path));
    }
    if (factsOk)
    {
        const NativeRife& f = g_res.facts;
        nr.ifnetPath = f.ifnetPath;
        nr.encodePath = f.encodePath;
        nr.jitPath = f.jitPath;
        nr.ejitPath = f.ejitPath;
        nr.restorePath = f.restorePath;
        nr.rjitPath = f.rjitPath;
        nr.restore = f.restore;
        nr.noEngine = f.noEngine;
        nr.cachePath = f.cachePath;
        nr.nvof = f.nvof;
        nr.fruc = f.fruc;
        nr.fsrfg = f.fsrfg;
        nr.frucDir = f.frucDir;
        nr.drba = f.drba;
        nr.block0Path = f.block0Path;
        nr.block0Jit = f.block0Jit;
        nr.gmfss = f.gmfss;
        nr.hh = f.hh;
        nr.hw = f.hw;
        nr.gfh = f.gfh;
        nr.gfw = f.gfw;
        for (int i = 0; i < kGmN; i++)
        {
            nr.gmPath[i] = f.gmPath[i];
            nr.gmJit[i] = f.gmJit[i];
        }
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
        else if (line.rfind("NATIVE-PATH encode=", 0) == 0)
            nr.encodePath = line.substr(19);
        else if (line.rfind("NATIVE-PATH jit=", 0) == 0)
            nr.jitPath = line.substr(16);
        else if (line.rfind("NATIVE-PATH ejit=", 0) == 0)
            nr.ejitPath = line.substr(17);
        else if (line.rfind("NATIVE-PATH restore=", 0) == 0)
            nr.restorePath = line.substr(20);
        else if (line.rfind("NATIVE-PATH rjit=", 0) == 0)
            nr.rjitPath = line.substr(17);
        else if (line.rfind("NATIVE-PATH cache=", 0) == 0)
            nr.cachePath = line.substr(18);
        else if (line.rfind("NATIVE-PATH fruc=", 0) == 0)
            nr.frucDir = line.substr(17);
        else if (line.rfind("NATIVE-PATH fsrfg=", 0) == 0)
            nr.frucDir = line.substr(18);
        else if (line.rfind("NATIVE-PATH block0=", 0) == 0)
            nr.block0Path = line.substr(19);
        else if (line.rfind("NATIVE-PATH block0jit=", 0) == 0)
            nr.block0Jit = line.substr(22);
        else if (line.rfind("LIVE READY", 0) == 0)
        {
            all = line;
            ready = true;
        }
        else if (line.rfind("NATIVE-PATH g", 0) == 0)
        {
            // the GMFSS set: `NATIVE-PATH <key>=` and `NATIVE-PATH <key>jit=` per engine
            for (int i = 0; i < kGmN; i++)
            {
                const std::string kp = std::string("NATIVE-PATH ") + kGmKey[i] + "=";
                const std::string kj = std::string("NATIVE-PATH ") + kGmKey[i] + "jit=";
                if (line.rfind(kp, 0) == 0)
                    nr.gmPath[i] = line.substr(kp.size());
                else if (line.rfind(kj, 0) == 0)
                    nr.gmJit[i] = line.substr(kj.size());
            }
        }
    };
    {
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<std::string> local;
        bool hit = nativeLocalHandoff(script, backend, capW, capH, local);
        if (!hit)
        {
            // no warm engines: build them from engine\onnx on a worker thread; a session
            // ending mid-build leaves it running under g_bgHandoff (see the comment above)
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
            LOG("native: warm engines found by the host, no python process (%.2fs)\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
    }
    if (!ready)
    {
        // the lookup and the build cover every product case; what is left (a missing
        // engine\onnx file or ONNX parser, a non-memory build failure, unreadable name tags)
        // ends the session here
        LOG("native: the host could not find or build the engines for this session (see the lines above; "
            "engine\\onnx and engine\\gpu_runtime must be intact)\n");
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
    // the nvof model: the same geometry-only answer with `engine=nvof` (no TensorRT engine; the
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
    nr.fruc = nr.fruc || nr.fsrfg;
    // GMFSS: five engine paths, the half-frame grid, the cache folder for the
    // kernel cubin (no RIFE jit path names it)
    nr.gmfss = all.find(" engine=gmfss") != std::string::npos;
    if (nr.gmfss)
    {
        nr.hh = num("hh", 0);
        nr.hw = num("hw", 0);
        nr.gfh = num("fh", 0);
        nr.gfw = num("fw", 0);
        if ((nr.gfh || nr.gfw) &&
            (nr.gfh < 32 || nr.gfw < 32 || (nr.gfh & 31) || (nr.gfw & 31) || nr.gfh > nr.hh || nr.gfw > nr.hw))
        {
            LOG("native: gmfss handoff flow grid %dx%d is not a /32 grid inside the %dx%d half\n", nr.gfw, nr.gfh,
                nr.hw, nr.hh);
            return false;
        }
        for (int i = 0; i < kGmN; i++)
            if (nr.gmPath[i].empty())
            {
                LOG("native: gmfss handoff named no %s engine\n", kGmName[i]);
                return false;
            }
        if (nr.hh <= 0 || nr.hw <= 0 || nr.cachePath.empty())
        {
            LOG("native: gmfss handoff line incomplete (hh / hw / cache)\n");
            return false;
        }
    }
    // native DRBA: the RIFE handoff plus the block0 engine; the lag is one capture
    nr.drba = all.find(" engine=drba") != std::string::npos;
    if (nr.drba && (nr.block0Path.empty() || num("lag", 0) != 1))
    {
        LOG("native: drba handoff named no block0 engine or no lag=1\n");
        return false;
    }
    const int outW = num("outw", 0), outH = num("outh", 0);
    nr.identity = (outW == nr.w && outH == nr.h);
    // every fit that changes the size runs as the separable Lanczos pair (the sum of the two axes'
    // taps a pixel instead of sampleOut's product); decided here so the early helper knows to run nativeRtxInit
    // (its staging buffers) before any effect flag is looked at. Upscale to re-derives it
    // against the internal render size in nativeDeriveUpscale.
    nr.fitAa = nr.dw != nr.w || nr.dh != nr.h;
    nr.hdr = g_hdr;
    nativeDeriveUpscale(nr);
    if (nr.ph <= 0 || nr.pw <= 0 || nr.w <= 0 || nr.h <= 0 || nr.cw <= 0 || nr.ch <= 0 ||
        (!nr.noEngine && !nr.gmfss && !nr.nvof && !nr.fruc && (nr.ifnetPath.empty() || nr.encodePath.empty())))
    {
        LOG("native: handoff line incomplete\n");
        return false;
    }
    if (num("effects", 0))
    {
        LOG("native: live effects are on, phase 1 has no sharpen or VSR path\n");
        return false;
    }
    // live Restore: the session asked for it, so the handoff must have named the engine
    nr.restore = !nr.restorePath.empty();
    if (g_restore && !nr.restore)
    {
        LOG("native: Restore is on but the handoff named no restore engine\n");
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
        f.encodePath = nr.encodePath;
        f.jitPath = nr.jitPath;
        f.ejitPath = nr.ejitPath;
        f.restorePath = nr.restorePath;
        f.rjitPath = nr.rjitPath;
        f.restore = nr.restore;
        f.noEngine = nr.noEngine;
        f.cachePath = nr.cachePath;
        f.nvof = nr.nvof;
        f.fruc = nr.fruc;
        f.fsrfg = nr.fsrfg;
        f.frucDir = nr.frucDir;
        f.drba = nr.drba;
        f.block0Path = nr.block0Path;
        f.block0Jit = nr.block0Jit;
        f.gmfss = nr.gmfss;
        f.hh = nr.hh;
        f.hw = nr.hw;
        for (int i = 0; i < kGmN; i++)
        {
            f.gmPath[i] = nr.gmPath[i];
            f.gmJit[i] = nr.gmJit[i];
        }
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
        f.batchMax = nr.batchMax;
        f.identity = nr.identity;
        f.fitAa = nr.fitAa;
        g_res.handoffKey = key;
        g_res.haveFacts = true;
    }
    return true;
}

static bool nativeReadFile(const std::string& path, std::vector<char>& out)
{
    FILE* f = nullptr;
    if (fopen_s(&f, path.c_str(), "rb") || !f)
        return false;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    out.resize(n > 0 ? (size_t)n : 0);
    const bool ok = n > 0 && fread(out.data(), 1, (size_t)n, f) == (size_t)n;
    fclose(f);
    return ok;
}

// runtime config with the SHARED JIT cache and the engine's own whole-graph capture strategy,
// then the execution context (the per-session part: a context is sized for the engine's
// maximum shape and is what the resident host drops between sessions; the engine itself stays)
static bool nativeMakeContext(NativeRife& nr, nvinfer1::ICudaEngine* eng, nvinfer1::IRuntimeConfig** cfgOut,
                              nvinfer1::IExecutionContext** ctxOut)
{
    nvinfer1::IRuntimeConfig* cfg = eng->createRuntimeConfig();
    if (!cfg)
    {
        LOG("native: createRuntimeConfig failed\n");
        return false;
    }
    if (!nr.jit)
    {
        nr.jit = cfg->createRuntimeCache();
        std::vector<char> cb;
        if (nr.jit && !nr.jitPath.empty() && nativeReadFile(nr.jitPath, cb))
            LOG("native: jit cache deserialize -> %d (%zu bytes)\n", (int)nr.jit->deserialize(cb.data(), cb.size()),
                cb.size());
        // every engine keeps its own cache file, so the encoder's kernels sit in its own
        // .jit; merged into this cache the encode context takes 7 ms instead of recompiling
        // them for 0.5 s on every first session of a pair
        if (nr.jit && !nr.ejitPath.empty() && nativeReadFile(nr.ejitPath, cb))
            LOG("native: encode jit cache merged -> %d (%zu bytes)\n", (int)nr.jit->deserialize(cb.data(), cb.size()),
                cb.size());
        // the restore engine's kernels too (live Restore), same reason
        if (nr.jit && !nr.rjitPath.empty() && nativeReadFile(nr.rjitPath, cb))
            LOG("native: restore jit cache merged -> %d (%zu bytes)\n", (int)nr.jit->deserialize(cb.data(), cb.size()),
                cb.size());
        // native DRBA's block0 engine, same reason
        if (nr.jit && !nr.block0Jit.empty() && nativeReadFile(nr.block0Jit, cb))
            LOG("native: block0 jit cache merged -> %d (%zu bytes)\n", (int)nr.jit->deserialize(cb.data(), cb.size()),
                cb.size());
        // the GMFSS set's caches, one file per engine
        for (int i = 0; i < kGmN; i++)
            if (nr.jit && !nr.gmJit[i].empty() && nativeReadFile(nr.gmJit[i], cb))
                LOG("native: gmfss %s jit cache merged -> %d (%zu bytes)\n", kGmName[i],
                    (int)nr.jit->deserialize(cb.data(), cb.size()), cb.size());
    }
    if (nr.jit)
        cfg->setRuntimeCache(*nr.jit);
    // offline keeps TRT-RTX's graph capture OFF by default (no gain above 480p, one
    // cudaErrorInvalidValue in ten runs, a possible GPU TDR link); SMV_OFFLINE_GRAPH=1 turns it on
    bool wantGraph = !g_offline || g_offlineGraph;
    // SMV_LIVE_GMFSS_GRAPH=0: the GMFSS session's contexts without the whole-graph strategy
    // (a diagnostic)
    if (nr.gmfss)
    {
        wchar_t v[8]{};
        if (GetEnvironmentVariableW(L"SMV_LIVE_GMFSS_GRAPH", v, 8) && !wcscmp(v, L"0"))
            wantGraph = false;
    }
    if (wantGraph)
        cfg->setCudaGraphStrategy(nvinfer1::CudaGraphStrategy::kWHOLE_GRAPH_CAPTURE);
    nvinfer1::IExecutionContext* ctx = eng->createExecutionContext(cfg);
    if (!ctx)
    {
        LOG("native: createExecutionContext failed\n");
        delete cfg;
        return false;
    }
    *cfgOut = cfg;
    *ctxOut = ctx;
    return true;
}

// exact sequence from NATIVE-HOST-API-MAP.md (a): validity pre-check, deserialize, then the
// config + context through nativeMakeContext.
static nvinfer1::ICudaEngine* nativeLoadEngine(NativeRife& nr, const std::string& path,
                                               nvinfer1::IRuntimeConfig** cfgOut, nvinfer1::IExecutionContext** ctxOut)
{
    std::vector<char> blob;
    if (!nativeReadFile(path, blob))
    {
        LOG("native: cannot read %s\n", path.c_str());
        return nullptr;
    }
    uint64_t diag = 0;
    const auto v = nr.rt->getEngineValidity(blob.data(), blob.size(), &diag);
    if (v == nvinfer1::EngineValidity::kINVALID)
    {
        LOG("native: engine rejected by getEngineValidity (reasons 0x%llx): %s\n", (unsigned long long)diag,
            path.c_str());
        return nullptr;
    }
    if (v == nvinfer1::EngineValidity::kSUBOPTIMAL)
        LOG("native: engine reports kSUBOPTIMAL (reasons 0x%llx), continuing\n", (unsigned long long)diag);
    nvinfer1::ICudaEngine* eng = nr.rt->deserializeCudaEngine(blob.data(), blob.size());
    if (!eng)
    {
        LOG("native: deserializeCudaEngine failed for %s\n", path.c_str());
        return nullptr;
    }
    if (!nativeMakeContext(nr, eng, cfgOut, ctxOut))
    {
        delete eng;
        return nullptr;
    }
    return eng;
}

// NVRTC once, then the cubin is cached on disk next to the TRT cache, keyed by source hash and
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
        {&nr.fPackOutV, "k_packOutV"},
        {&nr.fPackOutHdrV, "k_packOutHdrV"},
        {&nr.fPackOutTile, "k_packOutTile"},
        {&nr.fFitAaTile, "k_fitAaTile"},
        {&nr.fRestIn, "k_restIn"},
        {&nr.fRestFoldH, "k_restFoldH"},
        {&nr.fRestFoldV, "k_restFoldV"},
        {&nr.fRestFoldTile, "k_restFoldTile"},
        {&nr.fRestToF, "k_restToF"},
        {&nr.fClamp01, "k_clamp01"},
        {&nr.fPadPlanar, "k_padPlanar"},
        {&nr.fNrIn, "k_nrIn"},
        {&nr.fNrOut, "k_nrOut"},
        {&nr.fNrMv, "k_nrMv"},
        {&nr.fNrInPq, "k_nrInPq"},
        {&nr.fNrOutPq, "k_nrOutPq"},
        {&nr.fHalf, "k_half"},
        {&nr.fPyr, "k_pyr"},
        {&nr.fSplatSoft, "k_splatSoft"},
        {&nr.fSplatNorm, "k_splatNorm"},
        {&nr.fGmShrink, "k_gmShrinkAa"},
        {&nr.fGmFlowUp, "k_gmFlowUp"},
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
        {&nr.fPackInMotion, "k_packInMotion"},
        {&nr.fRifeBlend, "k_rifeBlend"},
        {&nr.fRifeBlendF, "k_rifeBlendF"},
        {&nr.fRifeBlendOut, "k_rifeBlendOut"},
        {&nr.fRifeBlendOut32, "k_rifeBlendOut32"},
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
        {&nr.fBgraRgbaFsr, "k_bgraRgbaFsr"},
        {&nr.fRgbaBgra, "k_rgbaBgra"},
        {&nr.fHdrRestOut, "k_hdrRestOut"},
        {&nr.fLocalCorr, "k_localCorr"},
        {&nr.fAttn2, "k_attn2"},
        {&nr.fAttn2b, "k_attn2b"},
    };
    for (auto& e : fns)
        if (cuModuleGetFunction(e.fn, nr.cuMod, e.nm) != CUDA_SUCCESS)
        {
            LOG("native: kernel %s missing\n", e.nm);
            return false;
        }
    return true;
}

static bool nativeBuildKernels(NativeRife& nr, const std::wstring& cacheDir)
{
    // resident host: the module is per process (same device, same cubin), only rebind
    if (g_resident && g_res.cuMod)
    {
        nr.cuMod = g_res.cuMod;
        return nativeBindKernels(nr);
    }
    int dev = 0;
    NCHK(cudaGetDevice(&dev), "cudaGetDevice");
    cudaDeviceProp prop{};
    NCHK(cudaGetDeviceProperties(&prop, dev), "cudaGetDeviceProperties");
    uint64_t hash = 1469598103934665603ull;
    for (const char* p = kNativeKernels; *p; p++)
    {
        hash ^= (unsigned char)*p;
        hash *= 1099511628211ull;
    }
    wchar_t nm[128];
    swprintf_s(nm, L"\\smv_native_%016llx_sm%d%d.cubin", (unsigned long long)hash, prop.major, prop.minor);
    const std::wstring cubinPath = cacheDir + nm;
    ensureDirW(cacheDir);
    std::vector<char> cubin;
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
        nv.cubinSize = (PFN_nvrtcGetCUBINSize)GetProcAddress(nv.mod, "nvrtcGetCUBINSize");
        nv.cubin = (PFN_nvrtcGetCUBIN)GetProcAddress(nv.mod, "nvrtcGetCUBIN");
        nv.logSize = (PFN_nvrtcGetProgramLogSize)GetProcAddress(nv.mod, "nvrtcGetProgramLogSize");
        nv.log = (PFN_nvrtcGetProgramLog)GetProcAddress(nv.mod, "nvrtcGetProgramLog");
        nv.destroy = (PFN_nvrtcDestroyProgram)GetProcAddress(nv.mod, "nvrtcDestroyProgram");
        if (!nv.ok())
        {
            LOG("native: nvrtc entry points missing\n");
            return false;
        }
        void* prog = nullptr;
        if (nv.create(&prog, kNativeKernels, "smv_native.cu", 0, nullptr, nullptr) != 0)
        {
            LOG("native: nvrtcCreateProgram failed\n");
            return false;
        }
        char arch[64];
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
            LOG("native: nvrtc compile failed (%d): %s\n", crc, lg.c_str());
            nv.destroy(&prog);
            return false;
        }
        size_t cs = 0;
        if (nv.cubinSize(prog, &cs) != 0 || !cs)
        {
            LOG("native: nvrtcGetCUBINSize failed\n");
            nv.destroy(&prog);
            return false;
        }
        cubin.resize(cs);
        if (nv.cubin(prog, cubin.data()) != 0)
        {
            LOG("native: nvrtcGetCUBIN failed\n");
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
                LOG("native: kernel cache not saved (rename failed)\n");
                _wunlink(tmp.c_str());
            }
        }
        LOG("native: kernels compiled for sm_%d%d (%zu bytes cubin)\n", prop.major, prop.minor, cubin.size());
    }
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

// python inverts the two ICtCp matrices in float64 and casts the result to fp32
// (rtxvideo.py ~226), so the inverse is computed in double here and uploaded as fp32.
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
//    (it writes only dev / stream / emCap / capMip / capArr / capTex / semCap, the handoff writes the
//    geometry fields; distinct members, no race).
//  * nativeCudaInitEarly: kernels and the model buffers (needs the handoff geometry).
//  * nativeRtxInit: the live TrueHDR / RTX VSR bridge, the sharpen staging and the warm-ups
//    (needs the kernel module and the handoff geometry), on a helper thread beside the engine
//    load in the early thread.
//  * nativeCudaInitLate: the output ring and the slot events (needs the slot count), always
//    on the compute thread.
// The CUDA runtime binds the device's PRIMARY context on every thread that calls
// cudaSetDevice, so streams, allocations, the kernel module and the TRT contexts created on
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
    // mipmapped array exactly as live_server.py's _CapTex does.
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
    // exactly what _CapTex asks cudart for: uchar4 in SDR, half4 with channel kind FLOAT in
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
    const bool capDirect = lkEnv("SMV_CAP_DIRECT") != "0";
    if (capDirect)
    {
        cudaResourceDesc rd{};
        rd.resType = cudaResourceTypeArray;
        rd.res.array.array = nr.capArr;
        cudaTextureDesc td{};
        td.addressMode[0] = td.addressMode[1] = cudaAddressModeClamp;
        td.filterMode = cudaFilterModePoint;
        td.readMode = cudaReadModeElementType;
        td.normalizedCoords = 0;
        if (cudaCreateTextureObject(&nr.capTex, &rd, &td, nullptr) != cudaSuccess)
        {
            cudaGetLastError();
            nr.capTex = 0;
        }
    }
    if (!nr.capTex)
        LOG("native: the 1:1 pack reads a copy of the capture (%s)\n",
            capDirect ? "the capture array has no texture object" : "SMV_CAP_DIRECT=0");

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
    // 0xC0000000 in R10G10B10A2 with A = 3 (python's -1073741824 canvas fill)
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

static bool nativeRtxInit(NativeRife& nr)
{
    // live Sharpen and RTX VSR: the fit is known now, so the rule "VSR
    // only when the resize enlarges in both axes" (_Fit._setup_resize) decides here
    // Upscale to: the first resize target is the internal render size when set, else
    // the fit rect (_Fit._setup_resize); VSR and the aa rule read that target
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
                nr.fsrUp ? "AMD FSR upscaling" : "RTX VSR");
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
        const char* up = nr.fsrUp ? "AMD FSR upscaling" : "RTX VSR";
        if (g_offline && !nr.nvPre && nr.rtxHdr)
            LOG("native: %s skipped (it takes SDR only, and the one resize follows RTX HDR and the model), Lanczos\n",
                up);
        else if (vtw > vw && vth > vh)
            nr.vsr = true;
        else
            LOG("native: live %s skipped (upscales only; this resize does not enlarge), Lanczos\n", up);
    }
    nr.fsrUp = nr.fsrUp && nr.vsr;
    nr.vsrPre = nr.vsrPre && nr.vsr;
    if (nr.uw)
    {
        NCHK(cudaMalloc((void**)&nr.dUp, (size_t)3 * nr.uw * nr.uh * sizeof(float)), "alloc internal render frame");
        if (nr.upAa)
        {
            NCHK(cudaMalloc((void**)&nr.dUpTmp, (size_t)3 * nr.uw * nr.h * sizeof(float)),
                 "alloc internal render pass");
            // k_fitAaTile saves the pair's round trip through dUpTmp; measured, it pays on every shrinking
            // height and on every enlarge of at most 3x, and is slower at 4.5x (its blocks re-run the
            // horizontal taps of the rows they share)
            nr.upTile = nr.uh < nr.h || (long long)nr.uh <= 3LL * nr.h;
        }
        LOG("native: live upscale to: model %dx%d -> %dx%d first, then %s to %dx%d%s\n", nr.w, nr.h, nr.uw, nr.uh,
            nr.identity ? "1:1" : "fit", nr.dw, nr.dh, nr.upTile ? " (the first resize in one kernel)" : "");
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
    // the fit's staging frame; with RTX VSR before the model it first stages the working-size frame, from the
    // capture's planes
    if (nr.sharpen > 0.0f || nr.fitAa || (nr.restore && !nr.restPre) || nr.vsrPre)
        NCHK(cudaMalloc((void**)&nr.dPres,
                        (size_t)3 * (std::max)(nr.dw * nr.dh, nr.vsrPre ? nr.w * nr.h : 0) * sizeof(float)),
             "alloc fit staging");
    if (nr.vsrPre && !nr.dCapF)
        NCHK(cudaMalloc((void**)&nr.dCapF, (size_t)3 * nr.ch * nr.cw * sizeof(float)), "alloc capture planes");
    if (nr.sharpen > 0.0f || nr.rtxHdr)
    {
        NCHK(cudaMalloc((void**)&nr.dShIn, (size_t)3 * nr.w * nr.h * sizeof(float)), "alloc after-DLSS 5 input");
        NCHK(cudaMalloc((void**)&nr.dShOut, (size_t)3 * nr.w * nr.h * sizeof(float)), "alloc after-DLSS 5 output");
    }
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
    // Restore's fold target (_Fit._load_restore's restore_target). Offline with a
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
        if (nr.restTh > 4 * sh)
            NCHK(cudaMalloc((void**)&nr.dRestF, (size_t)3 * 16 * sw * sh * sizeof(float)), "alloc restore fp32 output");
        // HDR planes: the source fitted to the target, the remainder Restore's way back adds (nativeHdrRestOut)
        if ((nr.encPre || nr.encPost) && (nr.restTw != sw || nr.restTh != sh))
            NCHK(cudaMalloc((void**)&nr.dRestRem, (size_t)3 * nr.restTw * nr.restTh * sizeof(float)),
                 "alloc restore HDR fit");
        LOG("native: live restore: Real-ESRGAN animevideov3 (TensorRT) at %dx%d -> %dx%d%s\n", sw, sh, nr.restTw,
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
    if (nr.vsr)
    {
        NCHK(cudaMalloc((void**)&nr.dVsrIn, (size_t)vw * vh * 4), "alloc VSR input");
        NCHK(cudaMalloc((void**)&nr.dVsrOut, (size_t)vtw * vth * 4), "alloc VSR output");
    }
    // RTX TrueHDR runs once per real frame at the working size, after Restore, DLSS 5 and Sharpen and before the
    // model (nativePreModelPost); its buffers take the largest frame of the route (live the capture or the working
    // size, whichever is bigger; the output offline), and live they first carry the capture's SDR planes
    // (k_sdrEncode)
    const bool workBig = !g_offline && (size_t)nr.w * nr.h > (size_t)nr.cw * nr.ch;
    const int hw = g_offline ? nr.dw : (workBig ? nr.w : nr.cw), hh = g_offline ? nr.dh : (workBig ? nr.h : nr.ch);
    if (nr.rtxHdr)
    {
        NCHK(cudaMalloc((void**)&nr.dThdrIn, (size_t)hw * hh * 4), "alloc TrueHDR input");
        NCHK(cudaMalloc((void**)&nr.dThdrOut, (size_t)hw * hh * 4), "alloc TrueHDR output");
        NCHK(cudaMalloc((void**)&nr.dSrcG, (size_t)3 * hw * hh * sizeof(float)), "alloc source gamma planes");
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
    if (nr.rtxHdr || (nr.vsr && !nr.fsrUp))
    {
        // the bridge retains the PRIMARY context (cuContext = NULL), which is the context this
        // thread just bound with cudaSetDevice. cuStream is NULL exactly as rtxvideo.py passes
        // it: the bridge wraps every eval in host-synchronous cuMemcpy2D calls on the legacy
        // default stream, and the host's own stream is cudaStreamNonBlocking, so an NGX eval
        // placed on it would not be ordered against the bridge's own output copy. ONE create
        // per process (NGX is single-instance), so both features are asked for at once.
        if (g_rtxb.create(nullptr, nullptr, 0, nr.rtxHdr ? 1u : 0u, nr.vsr && !nr.fsrUp ? 1u : 0u) != 1u)
        {
            LOG("native: rtx_video_api_cuda_create failed\n");
            return false;
        }
        g_rtxb.created = true;
        g_rtxUsed = true; // resident host: NGX is single-instance per process, end it after this session
    }
    if (nr.fsrUp && !nativeFsrupSetup(nr, vw, vh, vtw, vth))
        return false;
    if (nr.vsr && !nr.fsrUp)
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
        LOG("native: live upscale: RTX VSR %dx%d -> %dx%d%s\n", vw, vh, vtw, vth,
            nr.vsrPre ? ", the captured frame before the model" : "");
    }
    if (nr.rtxHdr)
    {
        // warm-up eval on a zero frame (opaque black BGRA), python live_server.py ~1859
        if (cuMemsetD32Async((CUdeviceptr)nr.dThdrIn, 0xFF000000u, (size_t)hw * hh, (CUstream)nr.stream) !=
            CUDA_SUCCESS)
        {
            LOG("native: TrueHDR warm-up clear failed\n");
            return false;
        }
        NCHK(cudaStreamSynchronize(nr.stream), "warm-up stream sync");
        const RtxRect wr{0, 0, (uint32_t)hw, (uint32_t)hh};
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
                "model, colour %ls, SDR white %.0f nits, contrast %u saturation %u\n",
                nr.w, nr.h, g_hdrColor, nr.sdrScale * 80.0f, nr.thdr.Contrast, nr.thdr.Saturation);
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
// itself). Its state is process-global (up to four FRUC instances), so a
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
    // the instance calls: up to 4 FRUC instances in the one module, each
    // with its own context; absent = an older bridge = one instance
    int (*createI)(int, unsigned, unsigned) = nullptr;
    int (*stepI)(int, void*, void*, void*, double, int, int*) = nullptr;
};
static FrucBridge g_fruc;

// one worker per extra instance (1..3): the compute thread hands it one step at a time and
// collects the result in tween order. The worker binds the host's device first, so the bridge's
// entry fence (cuCtxSynchronize on the caller's context) covers the host's packs and unpacks.
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
};
// never destroyed: no std::thread destructor runs at process exit (nativeFrucFree joins them)
static FrucWorker* const g_frW = new FrucWorker[4];

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
        lk.unlock();
        int rep = 0;
        const int rc = bound ? g_fruc.stepI(i, a, b, out, t, mode, &rep) : -100;
        std::string err = rc == -100 ? std::string("worker CUDA bind failed") : (rc ? g_fruc.lastError() : "");
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

// the FRUC instance at the padded model size (python sizes it to pw x ph the same way) and the
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
    for (int i = 0; i < 4; i++)
    {
        nr.frFed[i] = -1;
        nr.frFirst[i] = false;
        nr.frCalls[i] = 0;
    }
    nr.dFrOutBI[0] = nr.dFrOutB;
    nr.frInst = 1;
    nr.frPlanN = nr.frPlanK = 0;
    // recursive midpoints are the default wherever the bridge has the instance calls, one
    // instance per tree level, up to 4 on either route; SMV_FRUC_MIDPOINTS=0 keeps the direct-t
    // scheme (parallel instances up to 4 live, 1 offline: offline FRUC is paced by the encoder,
    // where they measured 0.99x / ~0.94x at x3 / x5 1080p), SMV_FRUC_DEPTH=1..4 the snap depth.
    // SMV_FRUC_INSTANCES caps the instances on either scheme (1 = one instance); the extra ones
    // are created by the first pair that needs them (nativeFrucGrow)
    nr.frMp = false;
    nr.frInstMax = 1;
    if (g_fruc.stepI)
    {
        char ev[8] = {};
        nr.frMp = !(GetEnvironmentVariableA("SMV_FRUC_MIDPOINTS", ev, sizeof(ev)) > 0 && ev[0] == '0');
        nr.frMpCap = g_offline ? 3 : 2;
        if (GetEnvironmentVariableA("SMV_FRUC_DEPTH", ev, sizeof(ev)) > 0)
            nr.frMpCap = (std::max)(1, (std::min)(4, atoi(ev)));
        nr.frInstMax = (g_offline && !nr.frMp) ? 1 : 4;
        if (GetEnvironmentVariableA("SMV_FRUC_INSTANCES", ev, sizeof(ev)) > 0)
            nr.frInstMax = (std::max)(1, (std::min)(4, atoi(ev)));
    }
    nr.frMpL = 0;
    nr.frSerial = 0;
    nr.frNodeCalls = 0;
    for (bool& ok : nr.frNodeOk)
        ok = false;
    for (uint64_t& k : nr.frLastKey)
        k = 0;
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
        LOG("native: fruc: %d instances (%d new in %.0f ms)\n", nr.frInst, nr.frInst - had,
            (double)(nowQpc100() - t0) / 10000.0);
}

// wait until every worker is idle (a round the caller abandoned, a session's end)
static void nativeFrucDrain(NativeRife& nr)
{
    for (int i = 1; i < nr.frInst; i++)
    {
        FrucWorker& w = g_frW[i];
        std::unique_lock<std::mutex> lk(w.m);
        w.cv.wait(lk, [&] { return w.done; });
    }
}

// the depth of a pair's midpoint tree: the smallest L that holds every t as a node k / 2^L (x2 / x4 /
// x8 / x16 are exact on both routes); a pair with a t that is no node takes frMpCap (its nearest node
// there), offline at least its deepest node's depth. Live never deepens such a pair: a grid phase that
// lands on a deep node by chance would cost the whole deep tree for the pair's few snapped tweens
static int nativeFrucDepth(const NativeRife& nr, const double* ts, uint32_t n)
{
    int L = 1;
    bool off = false;
    for (uint32_t i = 0; i < n; i++)
    {
        int d = 1;
        for (; d <= 4; d++)
        {
            const double x = std::ldexp(ts[i], d);
            if (fabs(x - floor(x + 0.5)) < 1e-4)
                break;
        }
        if (d <= 4)
            L = (std::max)(L, d);
        else
            off = true;
    }
    if (off)
        L = g_offline ? (std::max)(L, nr.frMpCap) : nr.frMpCap;
    return (std::min)(L, 4);
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
// last tweened pair's end. python's Fruc._reuse, only when this group has tweens: a pair that
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
    int n = 0;
    while (n == nr.frPrev || n == nr.frLast)
        n++;
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

// one tween at t into dFrOut (3, ph, pw), python's Fruc._infer + NvOFFRUC.interpolate (a
// double, the bridge's own type: offline --fps hands python's double fraction over unrounded)
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

// recursive midpoints: the key of node k / 2^L of this pair (k = 0 / 2^L = the pair's first /
// second frame): a real frame = its serial, an interior node = the pair's first serial and its
// position on the 16-grid; an instance fed the left parent of a node last feeds only the right one
static uint64_t nativeFrucKey(const NativeRife& nr, int k, int L)
{
    while (L > 0 && !(k & 1))
    {
        k >>= 1;
        L--;
    }
    const uint64_t a = nr.frSerial - 1;
    if (L == 0)
        return (k ? nr.frSerial : a) * 16 + 1;
    return a * 16 + ((uint64_t)k << (4 - L)) + 1;
}

// node k / 2^L of this pair's midpoint tree, once per pair: FRUC at t = 0.5 between the node's two
// parents (computed first) on instance L - 1, mode 1 when that instance was fed the left parent
// last, else mode 0 (prime the left parent, feed the right one)
static uint8_t* nativeFrucNode(NativeRife& nr, int k, int L)
{
    while (L > 0 && !(k & 1))
    {
        k >>= 1;
        L--;
    }
    if (L <= 0)
        return nr.dFrSurf[k ? nr.frB : nr.frA];
    const int p = k << (4 - L);
    if (nr.frNodeOk[p])
        return nr.dFrNode[p];
    uint8_t* left = nativeFrucNode(nr, k - 1, L);
    uint8_t* right = left ? nativeFrucNode(nr, k + 1, L) : nullptr;
    if (!right)
        return nullptr;
    if (!nr.dFrNode[p] && cudaMalloc((void**)&nr.dFrNode[p], (size_t)nr.pw * nr.ph * 4) != cudaSuccess)
    {
        nr.dFrNode[p] = nullptr;
        nr.die("fruc midpoint buffer alloc failed");
        return nullptr;
    }
    const int i = L - 1;
    const int mode = nr.frLastKey[i] == nativeFrucKey(nr, k - 1, L) ? 1 : 0;
    if (mode == 0 && nr.frCalls[i] > 0)
        nr.frPrimed++;
    int rep = 0;
    const int rc = g_fruc.stepI(i, left, right, nr.dFrNode[p], 0.5, mode, &rep);
    if (rc != 0)
    {
        LOG("native: fruc: midpoint %d/%d on instance %d failed: %s (rc %d)\n", k, 1 << L, i, g_fruc.lastError(), rc);
        nr.die("fruc midpoint failed");
        return nullptr;
    }
    nr.frLastKey[i] = nativeFrucKey(nr, k + 1, L);
    nr.frCalls[i]++;
    nr.frNodeCalls++;
    if (rep)
        nr.frRepeats++;
    nr.frNodeOk[p] = true;
    return nr.dFrNode[p];
}

// a FRUC output (true BGRA8) into dFrOut (3, ph, pw) as (R, G, B) planes; an SDR pair's tween back to the HDR
// codes (k_unpackBgraSdr)
static bool nativeFrucUnpack(NativeRife& nr, uint8_t* src)
{
    void* a[] = {&src, &nr.pw, &nr.ph, &nr.dFrOut};
    void* b[] = {&src, &nr.pw, &nr.ph, &nr.encPost, &nr.encWhite, &nr.dFrOut};
    const CUfunction f = nr.frPairSdr ? nr.fUnpackBgraSdr : nr.fUnpackBgraRgb;
    if (cuLaunchKernel(f, (nr.pw + 15) / 16, (nr.ph + 15) / 16, 1, 16, 16, 1, 0, (CUstream)nr.stream,
                       nr.frPairSdr ? b : a, nullptr) != CUDA_SUCCESS)
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
        if (!nr.frMpBuilt)
        {
            for (int l = 1; l <= nr.frMpL; l++)
                for (int k = 1; k < (1 << l); k += 2)
                    if (!nativeFrucNode(nr, k, l))
                        return false;
            nr.frMpBuilt = true;
        }
        const int den = 1 << nr.frMpL;
        const int k = (std::max)(1, (std::min)(den - 1, (int)floor(t * den + 0.5)));
        uint8_t* node = nativeFrucNode(nr, k, nr.frMpL);
        if (!node)
            return false;
        nr.frTweens++;
        if (!nativeFrucUnpack(nr, node))
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
    return nativeFrucUnpack(nr, outB);
}

static void nativeFrucFree(NativeRife& nr)
{
    if (nr.fsrfg)
        nativeFsrfgRelease(nr); // the bridge's instances and the Optical Flow session; the frames below are shared
    nativeFrucDrain(nr);
    for (int i = 1; i < 4; i++)
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
        char mp[64] = "";
        if (nr.frMp)
            snprintf(mp, sizeof(mp), ", %llu midpoint calls", (unsigned long long)nr.frNodeCalls);
        LOG("native: fruc session: %llu tweens, %llu primed pairs, %llu frame repeats%s, %d instance%s\n",
            (unsigned long long)nr.frTweens, (unsigned long long)nr.frPrimed, (unsigned long long)nr.frRepeats, mp,
            nr.frInst, nr.frInst > 1 ? "s" : "");
        g_fruc.destroy(); // every instance
        nr.frCreated = false;
    }
    for (int i = 1; i < 4; i++)
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
    for (bool& ok : nr.frNodeOk)
        ok = false;
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

// ---- AMD FSR frame generation (fsrfg) ----------------------------------------------------------
// smv_fsrfg_bridge.dll's C API (source: engine\fsrfg\build_src), loaded once per process by full path from the folder
// the handoff named; up to four instances, each its own FSR context on vkd3d-proton's device for the CUDA device's
// adapter
struct FsrfgShared
{
    HANDLE in, mv, out, fence;           // three textures (Vulkan opaque Win32 exports) and a timeline semaphore
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
};
static FsrfgBridge g_fsrfg;

// the bridge, its vkd3d-proton shader cache in cacheDir\fsr4_shaders (the model cache: users may delete it)
static bool nativeFsrfgLoad(const std::string& dir, const std::string& cacheDir)
{
    if (g_fsrfg.mod)
        return true;
    const std::wstring p = utf8ToWide(dir) + L"\\smv_fsrfg_bridge.dll";
    HMODULE m = LoadLibraryExW(p.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!m)
    {
        LOG("native: fsrfg: smv_fsrfg_bridge.dll did not load from %s (err %lu)\n", dir.c_str(), GetLastError());
        return false;
    }
    g_fsrfg.lastError = (const char* (*)())GetProcAddress(m, "fsrfg_last_error");
    g_fsrfg.setCacheDir = (void (*)(const char*))GetProcAddress(m, "fsrfg_set_cache_dir");
    g_fsrfg.provider = (const char* (*)())GetProcAddress(m, "fsrfg_provider");
    g_fsrfg.createI =
        (int (*)(int, uint32_t, uint32_t, uint32_t, int32_t, int, FsrfgShared*))GetProcAddress(m, "fsrfg_create_i");
    g_fsrfg.frameI = (int (*)(int, uint64_t, uint64_t, int, double))GetProcAddress(m, "fsrfg_frame_i");
    g_fsrfg.destroy = (void (*)())GetProcAddress(m, "fsrfg_destroy");
    if (!g_fsrfg.lastError || !g_fsrfg.setCacheDir || !g_fsrfg.provider || !g_fsrfg.createI || !g_fsrfg.frameI ||
        !g_fsrfg.destroy)
    {
        LOG("native: fsrfg: smv_fsrfg_bridge.dll lacks an expected export\n");
        FreeLibrary(m);
        return false;
    }
    if (!cacheDir.empty())
        g_fsrfg.setCacheDir((cacheDir + "\\fsr4_shaders").c_str());
    g_fsrfg.mod = m;
    return true;
}

// FSR instance i at w x h (RGBA16F) with its shared textures (as CUDA arrays) and fence imported into this CUDA context
static bool nativeFsrfgCreate(NativeRife& nr, int i)
{
    NativeRife::FgInst& g = nr.fg[i];
    FsrfgShared sh{};
    const int rc = g_fsrfg.createI(i, (uint32_t)nr.w, (uint32_t)nr.h, nr.fgLuid.LowPart, nr.fgLuid.HighPart, 1, &sh);
    if (rc != 0)
    {
        LOG("native: fsrfg: instance %d at %dx%d not created: %s (rc %d)\n", i, nr.w, nr.h, g_fsrfg.lastError(), rc);
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
    auto imp = [w, h](HANDLE hd, uint64_t bytes, const cudaChannelFormatDesc& fd, cudaExternalMemory_t& em,
                      cudaMipmappedArray_t& mm, cudaArray_t& a) {
        cudaExternalMemoryHandleDesc md{};
        md.type = cudaExternalMemoryHandleTypeOpaqueWin32;
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
    sd.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
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
// with k_nvofUp's bilinear taps where the engine refuses grid 1, or with SMV_OFVEC_GRID4=1) at nativeOfLevel(); tag names
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
        ev[0] = 0;
        o.costOn = GetEnvironmentVariableA("SMV_OFVEC_COST", ev, sizeof(ev)) > 0 && ev[0] == '1';
        ev[0] = 0;
        o.grid4 = GetEnvironmentVariableA("SMV_OFVEC_GRID4", ev, sizeof(ev)) > 0 && ev[0] == '1';
    }
    o.level = nativeOfLevel();
    for (int grid : {1, 4})
    {
        if (o.grid4 && grid == 1)
            continue;
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
        // FSR reads no cost (k_nvofUp's cost plane goes unread): off; SMV_OFVEC_COST=1 = on
        ip.enableOutputCost = o.costOn ? NV_OF_TRUE : NV_OF_FALSE;
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
        LOG(o.grid4 ? "native: %s: vectors at grid 4 (bilinear, SMV_OFVEC_GRID4=1)\n"
                    : "native: %s: the Optical Flow engine refused grid 1, vectors at grid 4 (bilinear)\n",
            tag);
    o.gw = (w + o.grid - 1) / o.grid;
    o.gh = (h + o.grid - 1) / o.grid;
    for (int k = 0; k < 2; k++)
    {
        uint32_t inPitch = 0;
        if (!nativeNvofBufOn(o.of, w, h, NV_OF_BUFFER_USAGE_INPUT, NV_OF_BUFFER_FORMAT_ABGR8, o.in[k], o.inP[k],
                             inPitch, "FSR vector input") ||
            !nativeNvofBufOn(o.of, o.gw, o.gh, NV_OF_BUFFER_USAGE_OUTPUT, NV_OF_BUFFER_FORMAT_SHORT2, o.out[k],
                             o.outP[k], o.outPitch[k], "FSR vector flow") ||
            !nativeNvofBufOn(o.of, o.gw, o.gh, NV_OF_BUFFER_USAGE_COST, NV_OF_BUFFER_FORMAT_UINT8, o.cost[k],
                             o.costP[k], o.costPitch[k], "FSR vector cost"))
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
        NCHK(cudaMalloc((void**)&o.dFlow, 3 * (size_t)w * h * sizeof(float)), "alloc FSR vectors");
    return true;
}

// the vectors between slots `prev` and `cur`, both holding their ABGR8 frames: the field cur -> prev in px (the backward
// field of the pair), into dst as R16G16_FLOAT texels, rows dstPitch bytes apart
static bool nativeOfVecRun(NativeRife& nr, NativeRife::OfVec& o, int prev, int cur, uint8_t* dst, int dstPitch,
                           const char* tag, cudaEvent_t evOf = nullptr)
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
        eo.outputCostBuffer = o.costOn ? o.cost[0] : nullptr;
        eo.bwdOutputBuffer = o.out[1];
        eo.bwdOutputCostBuffer = o.costOn ? o.cost[1] : nullptr;
    }
    else
    {
        // the forward field from `cur` to `prev` = the backward one of (prev, cur), into the buffers it is read from
        ei.inputFrame = o.in[cur];
        ei.referenceFrame = o.in[prev];
        eo.outputBuffer = o.out[1];
        eo.outputCostBuffer = o.costOn ? o.cost[1] : nullptr;
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
    int grid = o.grid, vp = (int)o.outPitch[1], cp = (int)o.costPitch[1], mp = dstPitch;
    CUdeviceptr v = o.outP[1], c = o.costP[1];
    if (grid == 1) // the field is the texels' grid: one pass, no float planes in between
    {
        void* g1[] = {&v, &vp, &o.w, &o.h, &dst, &mp};
        if (cuLaunchKernel(nr.fMvHalfG1, gx, gy, 1, 16, 16, 1, 0, (CUstream)st, g1, nullptr) != CUDA_SUCCESS)
        {
            nr.die("FSR vector launch failed");
            return false;
        }
        return true;
    }
    float* flow = o.dFlow;
    float* cost = o.dFlow + 2 * (size_t)o.w * o.h;
    void* b[] = {&v, &vp, &c, &cp, &o.gw, &o.gh, &grid, &o.w, &o.h, &flow, &cost};
    void* m[] = {&flow, &o.w, &o.h, &dst, &mp};
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
    for (NvOFGPUBufferHandle* b : {&o.in[0], &o.in[1], &o.out[0], &o.out[1], &o.cost[0], &o.cost[1]})
        if (*b)
            g_nvofApi.nvOFDestroyGPUBufferCuda(*b);
    if (o.of)
        g_nvofApi.nvOFDestroy(o.of);
    if (o.dFlow)
        cudaFree(o.dFlow);
    o = NativeRife::OfVec{};
}

// prev and frame (RGBA16F) into OfVec o's two inputs as ABGR8, on o's stream
static bool nativeFsrfgOfIn(NativeRife& nr, NativeRife::OfVec& o, const uint8_t* prev, const uint8_t* frame)
{
    const unsigned gx = (unsigned)(nr.w + 15) / 16, gy = (unsigned)(nr.h + 15) / 16;
    int sp = nr.fgPitch, ip = (int)o.inPitch;
    for (int k = 0; k < 2; k++)
    {
        const uint8_t* s = k ? frame : prev;
        CUdeviceptr d = o.inP[k];
        void* a[] = {(void*)&s, &sp, &nr.w, &nr.h, &d, &ip};
        if (cuLaunchKernel(nr.fRgbaHAbgr, gx, gy, 1, 16, 16, 1, 0, (CUstream)(o.st ? o.st : nr.stream), a, nullptr) !=
            CUDA_SUCCESS)
        {
            nr.die("rgbaHAbgr (fsrfg) launch failed");
            return false;
        }
    }
    return true;
}

// one bridge call: instance i gets `frame` (RGBA16F rows fgPitch apart) with its vectors toward `prev` (the frame the
// instance was fed last; null = zero vectors: a reset, or a re-prime nobody reads) and returns into `out` (null =
// dropped) the frame halfway between them. The vectors as the harness made them: the field this frame -> the previous
// (px) = FSR's convention. ring >= 0: the vectors were made ahead into dFgMvR[ring] (nativeFsrfgAheadVec)
static bool nativeFsrfgFeed(NativeRife& nr, int i, const uint8_t* frame, const uint8_t* prev, bool reset, uint8_t* out,
                            int ring = -1)
{
    NativeRife::FgInst& g = nr.fg[i];
    cudaStream_t st = nr.stream;
    const size_t row = (size_t)nr.w * 8;
    auto mark = [&nr, st](int e) {
        if (nr.fgProf)
            cudaEventRecord(nr.fgEv[e], st);
    };
    mark(0);
    if (cudaMemcpy2DToArrayAsync(g.in, 0, 0, frame, nr.fgPitch, row, nr.h, cudaMemcpyDeviceToDevice, st) != cudaSuccess)
    {
        nr.die("fsrfg frame copy failed");
        return false;
    }
    mark(1);
    const uint8_t* mv = ring >= 0 ? nr.dFgMvR[ring] : nr.dFgMv;
    if (ring >= 0)
    {
        if (cudaStreamWaitEvent(st, nr.fgEvVec[ring], 0) != cudaSuccess)
        {
            nr.die("fsrfg vector wait failed");
            return false;
        }
    }
    else if (prev)
    {
        if (!nativeFsrfgOfIn(nr, nr.fgOf, prev, frame))
            return false;
        mark(2);
        if (!nativeOfVecRun(nr, nr.fgOf, 0, 1, nr.dFgMv, nr.fgMvPitch, "fsrfg", nr.fgProf ? nr.fgEv[3] : nullptr))
            return false;
    }
    else if (cudaMemset2DAsync(nr.dFgMv, nr.fgMvPitch, 0, (size_t)nr.w * 4, nr.h, st) != cudaSuccess)
    {
        nr.die("fsrfg vector clear failed");
        return false;
    }
    else
    {
        mark(2);
        mark(3);
    }
    mark(4);
    if (cudaMemcpy2DToArrayAsync(g.mv, 0, 0, mv, nr.fgMvPitch, (size_t)nr.w * 4, nr.h, cudaMemcpyDeviceToDevice, st) !=
            cudaSuccess ||
        (ring >= 0 && cudaEventRecord(nr.fgEvFree[ring], st) != cudaSuccess))
    {
        nr.die("fsrfg vector copy failed");
        return false;
    }
    mark(5);
    cudaExternalSemaphoreSignalParams sp{};
    sp.params.fence.value = ++g.val;
    if (cudaSignalExternalSemaphoresAsync(&g.sem, &sp, 1, st) != cudaSuccess)
    {
        nr.die("fsrfg fence signal failed");
        return false;
    }
    const uint64_t ready = g.val, done = ++g.val;
    const int64_t c0 = nr.fgProf ? nowQpc100() : 0;
    // the frame time does not change FSR 4's output (a 24 and a 60 fps run byte for byte the same): one 60 Hz frame
    const int rc = g_fsrfg.frameI(i, ready, done, reset ? 1 : 0, 1000.0 / 60.0);
    if (nr.fgProf)
        nr.fgCpuMs += (double)(nowQpc100() - c0) / 10000.0;
    if (rc != 0)
    {
        LOG("native: fsrfg: instance %d call failed: %s (rc %d)\n", i, g_fsrfg.lastError(), rc);
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
                nr.fgStageMs[s] += ms;
        }
        nr.fgProfN++;
        nr.fgProfVecN += prev ? 1 : 0;
    }
    nr.frCalls[i]++;
    return true;
}

// the extra instances up to `want` (instance 0 is the session's own); a failed create keeps the ones made so far
static void nativeFsrfgGrow(NativeRife& nr, int want)
{
    want = (std::min)(want, 4);
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
        LOG("native: fsrfg: %d instances (%d new in %.0f ms)\n", nr.frInst, nr.frInst - had,
            (double)(nowQpc100() - t0) / 10000.0);
}

// node k / 2^L of this pair's midpoint tree (FRUC's nativeFrucNode): FSR on instance L - 1 fed the right parent
// after the left one. An instance fed another frame last (a pair of another depth, a new encoding) gets the left
// parent first, its output dropped: a reset only for its very first frame (after a reset FSR's next 10 tweens follow
// the vectors alone, ffx_frameinterpolation.h fFrameIndexFactor), else a plain call. ring >= 0: the node's vectors were
// made ahead into dFgMvR[ring] (nativeFsrfgLevel)
static uint8_t* nativeFsrfgNode(NativeRife& nr, int k, int L, int ring = -1)
{
    while (L > 0 && !(k & 1))
    {
        k >>= 1;
        L--;
    }
    if (L <= 0)
        return nr.dFrSurf[k ? nr.frB : nr.frA];
    const int p = k << (4 - L);
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
    const uint64_t lk = nativeFrucKey(nr, k - 1, L);
    if (nr.frLastKey[i] != lk)
    {
        if (!nativeFsrfgFeed(nr, i, left, nullptr, nr.frLastKey[i] == 0, nullptr))
            return nullptr;
        if (nr.frLastKey[i] != 0)
            nr.frPrimed++;
        nr.frLastKey[i] = lk;
    }
    if (!nativeFsrfgFeed(nr, i, right, left, false, nr.dFrNode[p], ring))
        return nullptr;
    nr.frLastKey[i] = nativeFrucKey(nr, k + 1, L);
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
    if (!ok || !nativeOfVecSetup(nr, nr.fgOf2, nr.w, nr.h, "fsrfg side vectors", nr.fgOfSt) || strcmp(ev, "fail") == 0)
    {
        LOG("native: fsrfg: the side Optical Flow session did not start, every call's vectors stay in its turn\n");
        nativeFsrfgAheadFree(nr);
        nr.fgAhead = -1;
        return false;
    }
    nr.fgAhead = 1;
    LOG("native: fsrfg: tree levels with 2+ calls make the next call's vectors during each FSR call (a side stream, its "
        "own Optical Flow session)\n");
    return true;
}

// a call's vectors (the field frame -> prev) on the side stream into dFgMvR[s] once that buffer's last copy is done,
// then fgEvVec[s]. The copy's event is recorded before this is enqueued: a stream waits for the event's last record
static bool nativeFsrfgAheadVec(NativeRife& nr, int s, const uint8_t* prev, const uint8_t* frame)
{
    if (cudaStreamWaitEvent(nr.fgOfSt, nr.fgEvFree[s], 0) != cudaSuccess)
    {
        nr.die("fsrfg side vector wait failed");
        return false;
    }
    if (!nativeFsrfgOfIn(nr, nr.fgOf2, prev, frame) ||
        !nativeOfVecRun(nr, nr.fgOf2, 0, 1, nr.dFgMvR[s], nr.fgMvPitch, "fsrfg side vectors"))
        return false;
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
    uint8_t* lf[8] = {};
    uint8_t* rt[8] = {};
    for (int j = 0; j < n; j++)
        if (!(lf[j] = nativeFsrfgNode(nr, 2 * j, l)) || !(rt[j] = nativeFsrfgNode(nr, 2 * j + 2, l)))
            return false;
    if (cudaEventRecord(nr.fgEvLevel, nr.stream) != cudaSuccess ||
        cudaStreamWaitEvent(nr.fgOfSt, nr.fgEvLevel, 0) != cudaSuccess)
    {
        nr.die("fsrfg level event failed");
        return false;
    }
    if (!nativeFsrfgAheadVec(nr, 0, lf[0], rt[0]))
        return false;
    for (int j = 0; j < n; j++)
    {
        if (j + 1 < n && !nativeFsrfgAheadVec(nr, (j + 1) & 1, lf[j + 1], rt[j + 1]))
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
    return true;
}

static bool nativeFsrfgSetup(NativeRife& nr)
{
    if (!nativeFsrfgLoad(nr.frucDir, nr.cachePath))
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
    // never the fast perf level here: at fast, grid 1, the engine's vectors can go wrong after ~320 calls of a 1080p
    // session with vkd3d-proton in the process (garbage tweens from then on, run to run different)
    if (!nativeFsrfgCreate(nr, 0) || !nativeOfVecSetup(nr, nr.fgOf, nr.w, nr.h, "fsrfg"))
        return false;
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
        nr.fgCpuMs = 0.0;
        nr.fgProfN = nr.fgProfVecN = 0;
    }
    for (int i = 0; i < 3; i++)
        nr.frIn[i] = nr.frSurfSdr[i] = false;
    nr.frPairSdr = false;
    nr.frModeLogs = 0;
    nr.frPrev = nr.frLast = nr.frA = nr.frB = -1;
    nr.frInst = 1;
    nr.frInstMax = 4;
    nr.frPlanN = nr.frPlanK = 0;
    nr.frMp = true;
    nr.frMpCap = g_offline ? 3 : 2; // FRUC's snap depths: offline x8's nodes, live x4's
    nr.frMpL = 0;
    nr.frMpBuilt = false;
    nr.frSerial = 0;
    nr.frNodeCalls = nr.frTweens = nr.frPrimed = nr.frRepeats = 0;
    for (bool& ok : nr.frNodeOk)
        ok = false;
    LOG("native: fsrfg session %dx%d, AMD FSR %s frame generation (ML) through smv_fsrfg_bridge.dll on vkd3d-proton, RGBA16F, NVIDIA optical flow vectors (grid %d, %s), recursive midpoints (depth cap %d)\n",
        nr.w, nr.h, g_fsrfg.provider(), nr.fgOf.grid, nativeOfLevelName(nr.fgOf.level), nr.frMpCap);
    if (nr.encPost)
        LOG("native: fsrfg: HDR planes: %s\n",
            nr.frSdrOn ? "frame pairs inside the SDR range go to FSR as their SDR view, any other pair as the HDR codes"
                       : "every pair as the HDR codes (SMV_FSRFG_SDR_PAIRS=0)");
    return true;
}

static bool nativeFsrfgPair(NativeRife& nr, const float* dCur, uint32_t nTween)
{
    int n = 0;
    while (n == nr.frPrev)
        n++;
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
    if (!nativeFsrfgPack(nr, dCur, n, sdr))
        return false;
    nr.frIn[n] = in;
    if (nr.frPrev >= 0 && nr.frSurfSdr[nr.frPrev] != sdr)
    {
        // the previous frame in the other encoding: packed again, and every fed instance restarts there (a key that
        // names no frame: a plain call, never a reset)
        if (!nativeFsrfgPack(nr, nativeXPrev(nr), nr.frPrev, sdr))
            return false;
        for (uint64_t& k : nr.frLastKey)
            if (k)
                k = ~0ull;
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
    if (nr.frA < 0)
    {
        // the stream's first frame: instance 0 starts there with a reset
        if (!nativeFsrfgFeed(nr, 0, nr.dFrSurf[n], nullptr, true, nullptr))
            return false;
        nr.frLastKey[0] = nativeFrucKey(nr, 1, 0);
        return true;
    }
    // every real frame goes to instance 0 in order, tweens or not (a gap would cost a re-prime): that call is the
    // pair's midpoint
    return nativeFsrfgNode(nr, 1, 1) != nullptr;
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
    if (nr.stream)
        cudaStreamSynchronize(nr.stream);
    if (nr.fg[0].made)
        LOG("native: fsrfg session: %llu tweens, %llu midpoint calls, %llu re-primed streams, %d instance%s\n",
            (unsigned long long)nr.frTweens, (unsigned long long)nr.frNodeCalls, (unsigned long long)nr.frPrimed,
            nr.frInst, nr.frInst > 1 ? "s" : "");
    if (nr.fgProfN)
    {
        const double n = (double)nr.fgProfN, nv = nr.fgProfVecN ? (double)nr.fgProfVecN : 1.0;
        double sum = 0.0;
        for (double s : nr.fgStageMs)
            sum += s;
        LOG("[fsrfg-prof] %llu calls (%llu with vectors) %dx%d, ms a call: copy in %.3f, Optical Flow inputs %.3f, "
            "Optical Flow %.3f, vector kernels %.3f (the three a vector call), vector copy %.3f, FSR on vkd3d %.3f, "
            "copy out %.3f, all %.3f; the bridge's CPU call %.3f\n",
            (unsigned long long)nr.fgProfN, (unsigned long long)nr.fgProfVecN, nr.w, nr.h, nr.fgStageMs[0] / n,
            nr.fgStageMs[1] / nv, nr.fgStageMs[2] / nv, nr.fgStageMs[3] / nv, nr.fgStageMs[4] / n, nr.fgStageMs[5] / n,
            nr.fgStageMs[6] / n, sum / n, nr.fgCpuMs / n);
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
    if (any && g_fsrfg.destroy)
        g_fsrfg.destroy(); // every instance
    if (nr.dFgMv)
    {
        cudaFree(nr.dFgMv);
        nr.dFgMv = nullptr;
    }
    nativeFsrfgAheadFree(nr);
    nativeOfVecFree(nr.fgOf);
    nr.frInst = 1;
}

// ---- AMD FSR upscaling (fsrUp) -------------------------------------------------------------------
// smv_fsrup_bridge.dll's C API (source: engine\fsrup\build_src), loaded once per process by full path from the folder
// the config named (SMV_FSRUP_DIR, else engine\fsrup), AMD's signed loader + upscaler beside it
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
    int (*create)(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, int32_t, const char*, FsrupShared*) = nullptr;
    int (*frame)(uint64_t, uint64_t, int, double) = nullptr;
    void (*destroy)() = nullptr;
};
static FsrupBridge g_fsrup;

static bool nativeFsrupLoad(const std::string& dir)
{
    if (g_fsrup.mod)
        return true;
    const std::wstring p = utf8ToWide(dir) + L"\\smv_fsrup_bridge.dll";
    HMODULE m = LoadLibraryExW(p.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!m)
    {
        LOG("native: fsr upscale: smv_fsrup_bridge.dll did not load from %s (err %lu)\n", dir.c_str(), GetLastError());
        return false;
    }
    g_fsrup.lastError = (const char* (*)())GetProcAddress(m, "fsrup_last_error");
    g_fsrup.create = (int (*)(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, int32_t, const char*,
                              FsrupShared*))GetProcAddress(m, "fsrup_create");
    g_fsrup.frame = (int (*)(uint64_t, uint64_t, int, double))GetProcAddress(m, "fsrup_frame");
    g_fsrup.destroy = (void (*)())GetProcAddress(m, "fsrup_destroy");
    if (!g_fsrup.lastError || !g_fsrup.create || !g_fsrup.frame || !g_fsrup.destroy)
    {
        LOG("native: fsr upscale: smv_fsrup_bridge.dll lacks an expected export\n");
        FreeLibrary(m);
        return false;
    }
    g_fsrup.mod = m;
    return true;
}

// the upscaler at RTX VSR's resize (w x h -> ow x oh, nativeRtxInit's slot), its shared buffers and fence imported, and
// the vectors' Optical Flow session at the render size
static bool nativeFsrupSetup(NativeRife& nr, int w, int h, int ow, int oh)
{
    if (!nativeFsrupLoad(nr.fsrUpDir))
        return false;
    int dev = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess)
    {
        LOG("native: fsr upscale: the CUDA device's properties are unavailable\n");
        return false;
    }
    LUID luid;
    memcpy(&luid, prop.luid, sizeof(luid));
    FsrupShared sh{};
    const int rc =
        g_fsrup.create((uint32_t)w, (uint32_t)h, (uint32_t)ow, (uint32_t)oh, luid.LowPart, luid.HighPart, "4.", &sh);
    if (rc != 0)
    {
        LOG("native: fsr upscale: %dx%d -> %dx%d not created: %s (rc %d)\n", w, h, ow, oh, g_fsrup.lastError(), rc);
        return false;
    }
    nr.fuMade = true;
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
    cudaExternalSemaphoreHandleDesc sd{};
    sd.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
    sd.handle.win32.handle = sh.fence;
    if (!imp(sh.in, sh.inBytes, nr.fuEmIn, nr.dFuIn) || !imp(sh.mv, sh.mvBytes, nr.fuEmMv, nr.dFuMv) ||
        !imp(sh.out, sh.outBytes, nr.fuEmOut, nr.dFuOut) || cudaImportExternalSemaphore(&nr.fuSem, &sd) != cudaSuccess)
    {
        LOG("native: fsr upscale: the shared buffers or fence did not import into CUDA\n");
        return false;
    }
    nr.fuInPitch = (int)sh.inPitch;
    nr.fuMvPitch = (int)sh.mvPitch;
    nr.fuOutPitch = (int)sh.outPitch;
    memcpy(nr.fuVersion, sh.version, sizeof(nr.fuVersion));
    nr.fuVersion[sizeof(nr.fuVersion) - 1] = 0;
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
    LOG("native: upscale: AMD FSR %s %dx%d -> %dx%d through smv_fsrup_bridge.dll, NVIDIA optical flow vectors (grid "
        "%d, %s)\n",
        nr.fuVersion, w, h, ow, oh, nr.fuOf.grid, nativeOfLevelName(nr.fuOf.level));
    return true;
}

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
    void* a[] = {&nr.dVsrIn, &w, &h, &nr.dFuIn, &fip, &ofIn, &oip};
    if (cuLaunchKernel(nr.fBgraRgbaFsr, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a, nullptr) !=
        CUDA_SUCCESS)
    {
        nr.die("bgraRgbaFsr launch failed");
        return false;
    }
    mark(1);
    const bool first = nr.fuN == 0;
    if (first)
    {
        if (cudaMemset2DAsync(nr.dFuMv, nr.fuMvPitch, 0, (size_t)w * 4, h, st) != cudaSuccess)
        {
            nr.die("fsr upscale vector clear failed");
            return false;
        }
        mark(2);
    }
    else if (!nativeOfVecRun(nr, nr.fuOf, prev, cur, nr.dFuMv, nr.fuMvPitch, "fsr upscale",
                             nr.fuProf ? nr.fuEv[2] : nullptr))
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
    // FSR 4.1.1 gives byte-identical frames at any frame time (measured, harness fsr4_int8_upscaler): one 60 Hz frame
    const int rc = g_fsrup.frame(ready, done, first ? 1 : 0, 1000.0 / 60.0);
    if (rc != 0)
    {
        LOG("native: fsr upscale: frame %llu failed: %s (rc %d)\n", (unsigned long long)nr.fuN, g_fsrup.lastError(),
            rc);
        nr.die("fsr upscale failed");
        return false;
    }
    cudaExternalSemaphoreWaitParams wp{};
    wp.params.fence.value = done;
    int op = nr.fuOutPitch;
    void* b[] = {&nr.dFuOut, &op, &ow, &oh, &nr.dVsrOut};
    if (cudaWaitExternalSemaphoresAsync(&nr.fuSem, &wp, 1, st) != cudaSuccess)
    {
        nr.die("fsr upscale result failed");
        return false;
    }
    mark(4);
    if (cuLaunchKernel(nr.fRgbaBgra, (ow + 15) / 16, (oh + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, b, nullptr) !=
        CUDA_SUCCESS)
    {
        nr.die("fsr upscale result failed");
        return false;
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

static void nativeFsrupRelease(NativeRife& nr)
{
    if (nr.stream)
        cudaStreamSynchronize(nr.stream);
    if (nr.fuMade)
        LOG("native: fsr upscale: %llu frames (AMD FSR %s)\n", (unsigned long long)nr.fuN, nr.fuVersion);
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
    if (nr.fuMade && g_fsrup.destroy)
        g_fsrup.destroy();
    nr.fuMade = false;
    nativeOfVecFree(nr.fuOf);
}

// native DRBA's buffers (both routes): the ring of pictures at the pictures' pad, everything the
// motion search uses at the model's frame (RIFE's two domains: the motion frame, where the ring also
// keeps the motion frames, the IFNet's x and block0's fp32 frame pair; else the pictures' pad, the
// encode's fp16 frame in dXh's cur half)
static bool nativeDrbaAlloc(NativeRife& nr)
{
    const size_t plane = (size_t)nr.ph * nr.pw, mplane = nr.mph ? (size_t)nr.mph * nr.mpw : plane;
    NCHK(cudaMalloc((void**)&nr.dEncHalf, 16 * mplane * sizeof(uint16_t)), "alloc encode out");
    if (!nr.mph)
        NCHK(cudaMalloc((void**)&nr.dXh, 6 * plane * sizeof(uint16_t)), "alloc x fp16");
    else
        NCHK(cudaMalloc((void**)&nr.dDrB0, 6 * mplane * sizeof(float)), "alloc drba block0 frames");
    for (int i = 0; i < 4; i++)
    {
        NCHK(cudaMalloc((void**)&nr.dDrI[i], 3 * plane * sizeof(float)), "alloc drba frame ring");
        NCHK(cudaMalloc((void**)&nr.dDrF[i], 16 * mplane * sizeof(float)), "alloc drba encode ring");
        if (nr.mph)
            NCHK(cudaMalloc((void**)&nr.dDrM[i], 3 * mplane * sizeof(uint16_t)), "alloc drba motion ring");
    }
    for (int i = 0; i < 2; i++)
    {
        NCHK(cudaMalloc((void**)&nr.dDrX[i], 6 * plane * sizeof(float)), "alloc drba x");
        if (nr.mph)
            NCHK(cudaMalloc((void**)&nr.dDrMX[i], 6 * mplane * sizeof(uint16_t)), "alloc drba motion x");
        NCHK(cudaMalloc((void**)&nr.drWin[i].f10, 2 * mplane * sizeof(float)), "alloc drba window");
        NCHK(cudaMalloc((void**)&nr.drWin[i].r, 4 * mplane * sizeof(float)), "alloc drba window");
    }
    NCHK(cudaMalloc((void**)&nr.dDrFlow, 4 * mplane * sizeof(float)), "alloc drba flow");
    NCHK(cudaMalloc((void**)&nr.dDrFlowN, 4 * mplane * sizeof(float)), "alloc drba flow");
    NCHK(cudaMalloc((void**)&nr.dDrAcc, 6 * mplane * sizeof(long long)), "alloc drba accumulator");
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
        const int shown = mode ? mode : (g_flowScale >= 0.995 ? 2 : 0); // no --scale = DLAA
        char how[64];
        if (shown)
            sprintf_s(how, "DLSS mode %ls%s", kDlssModeName[shown - 1], g_dlssMode == 1 ? " (Auto)" : "");
        else
            sprintf_s(how, "%.2f", g_flowScale);
        LOG("native: live working size %dx%d = %s of the %dx%d presented, capture %dx%d%s\n", nr.w, nr.h, how, presW,
            presH, nr.cw, nr.ch,
            raised == 2   ? " (raised to DLSS 5's minimum width, 128 px, aspect kept)"
            : raised == 1 ? " (raised to the 64 px floor, aspect kept)"
                          : "");
    }
    // live: Restore reads the captured frame (its engine has the capture size) and folds it to
    // the model size before the model runs, so srcW / srcH name the capture from here on (the Restore warm-up
    // in nativeTrtInit and its target in nativeRtxInit read them, possibly on two threads)
    if (nr.restore)
    {
        nr.restPre = true;
        nr.sw = nr.cw;
        nr.sh = nr.ch;
    }

    const int capBpp = nr.hdr ? 8 : 4;
    const size_t plane = (size_t)nr.ph * nr.pw;
    NCHK(cudaMalloc((void**)&nr.dCap, (size_t)nr.cw * nr.ch * capBpp), "alloc capture staging");
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
        NCHK(cudaMalloc((void**)&nr.dF[0], 16 * mplane * sizeof(float)), "alloc f0");
        NCHK(cudaMalloc((void**)&nr.dF[1], 16 * mplane * sizeof(float)), "alloc f1");
        NCHK(cudaMalloc((void**)&nr.dEncHalf, 16 * mplane * sizeof(uint16_t)), "alloc encode out");
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
    // live Restore: the engine's x / y at its input size, the captured frame (sized for fp32 so
    // either dtype fits), the fold's horizontal pass at the 4x height and the model-size target,
    // and the folded model-size frame; the enlarging fold's fp32 copy is allocated in
    // nativeRtxInit once the target is known (live: a working size above 4x the capture)
    if (nr.restore)
    {
        const size_t mp = (size_t)nr.w * nr.h, sp = (size_t)srcW(nr) * srcH(nr);
        NCHK(cudaMalloc(&nr.dRestIn, 3 * sp * sizeof(float)), "alloc restore input");
        NCHK(cudaMalloc(&nr.dRestOut, 3 * 16 * sp * sizeof(float)), "alloc restore output");
        NCHK(cudaMalloc((void**)&nr.dRestTmp, (size_t)3 * 4 * srcH(nr) * nr.w * sizeof(float)),
             "alloc restore fold pass");
        NCHK(cudaMalloc((void**)&nr.dRest, 3 * mp * sizeof(float)), "alloc restore model-size frame");
    }
    // With live TrueHDR the PQ frame always exists at capture resolution first, so
    // the resize pair runs at EVERY image scale (at 1.00 the triangle filter sits on identity
    // positions and is an exact copy) and both staging planes are needed even at scale 1.00.
    if (nr.w != nr.cw || nr.h != nr.ch || nr.rtxHdr)
    {
        NCHK(cudaMalloc((void**)&nr.dTmp, (size_t)3 * nr.ch * nr.w * sizeof(float)), "alloc resize temp");
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
    }
    // the nvof model: the Optical Flow session and the glue buffers (the kernels are bound above)
    if (nr.nvof && !nativeNvofSetup(nr))
        return false;
    // Smooth Motion: the bridge's FRUC instance and its BGRA8 surfaces
    if (nr.fruc && !nativeFrucSetup(nr))
        return false;
    return true;
}

// the TRT cache folder = the folder of the jit cache the handoff named (kernels cubin lives there)
static std::wstring nativeCacheDir(const NativeRife& nr)
{
    if (!nr.cachePath.empty())
        return std::wstring(nr.cachePath.begin(), nr.cachePath.end());
    if (nr.jitPath.empty())
        return std::wstring();
    std::wstring w(nr.jitPath.begin(), nr.jitPath.end());
    const size_t sl = w.find_last_of(L"\\/");
    return sl == std::wstring::npos ? std::wstring(L".") : w.substr(0, sl);
}

// Live RTX TrueHDR runs in this process. Everything comes from the exe's own
// globals, parsed from the app's RTX HDR controls, so no new flag exists. Runs after
// nativeLoadDlls (the bridge folder is derived from the runtime folder found there).
static bool nativeConfigHdr(NativeRife& nr)
{
    // live Sharpen and RTX VSR read the exe's own globals; VSR needs SDR (Lanczos in HDR),
    // the bridge
    // and nvngx_vsr.dll present (else Lanczos, never a route change), and the fit must
    // enlarge (decided in nativeRtxInit once the handoff geometry is known)
    nr.sharpen = (float)(g_sharpen < 0.0 ? 0.0 : (g_sharpen > 2.0 ? 2.0 : g_sharpen));
    if (g_rtxVsr)
    {
        if (g_hdr)
            LOG("native: live RTX VSR demoted to Lanczos in HDR mode (SDR-only feature)\n");
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
    // AMD FSR upscaling (--fsr-upscale) takes RTX VSR's resize under its rules (SDR only, enlarging only); the two
    // exclude each other (the app ticks one), RTX VSR wins a command line with both
    if (g_fsrUp && g_rtxVsr)
        LOG("native: --fsr-upscale ignored: RTX VSR holds the resize (the two exclude each other)\n");
    else if (g_fsrUp && g_hdr)
        LOG("native: AMD FSR upscaling demoted to Lanczos in HDR mode (SDR only, as RTX VSR)\n");
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
    if (!(g_hdr && g_rtxHdr))
        return true;
    if (!rtxBridgeLoad())
        return false;
    nr.rtxHdr = true;
    nr.thdr.Contrast = (uint32_t)(g_hdrCon < 0 ? 0 : (g_hdrCon > 200 ? 200 : g_hdrCon));
    nr.thdr.Saturation = (uint32_t)(g_hdrSat < 0 ? 0 : (g_hdrSat > 200 ? 200 : g_hdrSat));
    nr.thdr.MiddleGray = 50; // the live defaults, rtxvideo.py ~173
    nr.thdr.MaxLuminance = 1000;
    nr.sdrScale = (float)(g_sdrWhite / 80.0 < 0.5 ? 0.5 : g_sdrWhite / 80.0);
    nr.rtxMode = !wcscmp(g_hdrColor, L"rtx") ? 1 : (!wcscmp(g_hdrColor, L"raw") ? 2 : 0);
    nr.rtxVib = (float)(g_hdrVib < 0.0 ? 0.0 : (g_hdrVib > 1.0 ? 1.0 : g_hdrVib));
    nr.rtxSb = (float)(g_hdrSb < 0.0 ? 0.0 : (g_hdrSb > 1.0 ? 1.0 : g_hdrSb));
    return true;
}

// GMFSS: one warm enqueue of an engine on zeroed scratch buffers sized from its own tensor
// shapes (every GMFSS engine is H/W pinned, so the shapes are static), which also logs the
// tensor contract the chain binds to. Buffers are freed again here.
static const char* trtDtypeName(nvinfer1::DataType t)
{
    switch (t)
    {
    case nvinfer1::DataType::kFLOAT:
        return "fp32";
    case nvinfer1::DataType::kHALF:
        return "fp16";
    case nvinfer1::DataType::kBF16:
        return "bf16";
    case nvinfer1::DataType::kINT32:
        return "i32";
    case nvinfer1::DataType::kINT64:
        return "i64";
    case nvinfer1::DataType::kBOOL:
        return "bool";
    default:
        return "other";
    }
}

static size_t trtElemSize(nvinfer1::DataType t)
{
    switch (t)
    {
    case nvinfer1::DataType::kFLOAT:
    case nvinfer1::DataType::kINT32:
        return 4;
    case nvinfer1::DataType::kHALF:
    case nvinfer1::DataType::kBF16:
        return 2;
    case nvinfer1::DataType::kINT64:
        return 8;
    default:
        return 1;
    }
}

static bool nativeWarmEngine(NativeRife& nr, nvinfer1::ICudaEngine* eng, nvinfer1::IExecutionContext* ctx,
                             const char* tag, const char* model = "gmfss")
{
    std::vector<void*> bufs;
    std::string contract;
    bool ok = true;
    const int n = eng->getNbIOTensors();
    for (int i = 0; i < n && ok; i++)
    {
        const char* nm = eng->getIOTensorName(i);
        const nvinfer1::Dims d = eng->getTensorShape(nm);
        size_t vol = 1;
        std::string dims;
        for (int k = 0; k < d.nbDims; k++)
        {
            if (d.d[k] < 0)
            {
                LOG("native: %s %s tensor %s has a dynamic dim\n", model, tag, nm);
                ok = false;
                break;
            }
            vol *= (size_t)d.d[k];
            dims += (k ? "x" : "") + std::to_string(d.d[k]);
        }
        if (!ok)
            break;
        const bool in = eng->getTensorIOMode(nm) == nvinfer1::TensorIOMode::kINPUT;
        if (in && !ctx->setInputShape(nm, d))
        {
            LOG("native: %s %s setInputShape rejected on %s\n", model, tag, nm);
            ok = false;
            break;
        }
        void* p = nullptr;
        const size_t bytes = vol * trtElemSize(eng->getTensorDataType(nm));
        if (cudaMalloc(&p, bytes) != cudaSuccess)
        {
            LOG("native: %s %s warm-up alloc failed (%zu bytes)\n", model, tag, bytes);
            ok = false;
            break;
        }
        bufs.push_back(p);
        cudaMemsetAsync(p, 0, bytes, nr.stream);
        ctx->setTensorAddress(nm, p);
        contract += std::string(i ? ", " : "") + (in ? "" : "-> ") + nm + " " +
                    trtDtypeName(eng->getTensorDataType(nm)) + " " + dims;
    }
    if (ok && !ctx->enqueueV3(nr.stream))
    {
        LOG("native: %s %s warm-up enqueue failed\n", model, tag);
        ok = false;
    }
    if (ok && cudaStreamSynchronize(nr.stream) != cudaSuccess)
    {
        LOG("native: %s %s warm-up sync failed\n", model, tag);
        ok = false;
    }
    for (void* p : bufs)
        cudaFree(p);
    if (ok)
        LOG("native: %s %s: %s\n", model, tag, contract.c_str());
    return ok;
}

// gmflow split: the contract between A (slot 1), k_localCorr and B (slot 5), read off the engines. The kernel takes
// q [2, HW, 128] and f [2, H, W, 128] (channels-last: 32 lanes of float4) with coords [2, HW, K, 2] on the quarter of
// the half frame and writes corr [2, HW, K] (K <= 100: its shared grid holds 12 x 12 taps); A hands out exactly the
// kGmA tensors and B takes corr, img0, img1 and the other A outputs at their exact dims and dtype. The backbone (slot 6)
// makes one frame's map4 / map8 [1, C, H, W] and A takes both frames' as [2, C, H, W] in the same dtype.
static bool nativeGmSplitCheck(NativeRife& nr)
{
    nvinfer1::ICudaEngine* a = nr.engGm[1];
    nvinfer1::ICudaEngine* b = nr.engGm[5];
    auto bad = [](const char* nm, const char* what) {
        LOG("native: gmfss split: %s %s\n", nm, what);
        return false;
    };
    auto same = [](const nvinfer1::Dims& x, const nvinfer1::Dims& y) {
        if (x.nbDims != y.nbDims)
            return false;
        for (int k = 0; k < x.nbDims; k++)
            if (x.d[k] != y.d[k])
                return false;
        return true;
    };
    if (a->getNbIOTensors() != 13 || b->getNbIOTensors() != 9)
        return bad("gmflow_bidir_a / _b", "do not have the split's 13 / 9 tensors");
    nvinfer1::ICudaEngine* bb = nr.engGm[6];
    if (bb->getNbIOTensors() != 3)
        return bad("gmflow_backbone", "does not have its 3 tensors (img0 -> map4, map8)");
    for (const char* nm : {"map4", "map8"})
    {
        const nvinfer1::Dims o = bb->getTensorShape(nm), in = a->getTensorShape(nm);
        if (bb->getTensorIOMode(nm) != nvinfer1::TensorIOMode::kOUTPUT ||
            a->getTensorIOMode(nm) != nvinfer1::TensorIOMode::kINPUT)
            return bad(nm, "is not an output of gmflow_backbone and an input of gmflow_bidir_a");
        if (o.nbDims != 4 || in.nbDims != 4 || o.d[0] != 1 || in.d[0] != 2 || o.d[1] != in.d[1] || o.d[2] != in.d[2] ||
            o.d[3] != in.d[3] || bb->getTensorDataType(nm) != a->getTensorDataType(nm))
            return bad(nm,
                       "is not [1, C, H, W] out of gmflow_backbone and [2, C, H, W] into gmflow_bidir_a, one dtype");
    }
    for (const char* nm : kGmA)
        if (a->getTensorIOMode(nm) != nvinfer1::TensorIOMode::kOUTPUT)
            return bad(nm, "is not an output of gmflow_bidir_a");
    for (const char* nm : {"q", "f", "coords"})
        if (a->getTensorDataType(nm) != nvinfer1::DataType::kFLOAT)
            return bad(nm, "is not fp32");
    if (b->getTensorIOMode("corr") != nvinfer1::TensorIOMode::kINPUT ||
        b->getTensorDataType("corr") != nvinfer1::DataType::kFLOAT)
        return bad("corr", "is not an fp32 input of gmflow_bidir_b");
    const nvinfer1::Dims q = a->getTensorShape("q"), f = a->getTensorShape("f"), g = a->getTensorShape("coords"),
                         c = b->getTensorShape("corr");
    const int gh = nr.gfh ? nr.gfh : nr.hh, gw = nr.gfh ? nr.gfw : nr.hw;
    if (f.nbDims != 4 || f.d[0] != 2 || f.d[1] != gh / 4 || f.d[2] != gw / 4 || f.d[3] != 128)
        return bad("f", "is not [2, H, W, 128] on the quarter of gmflow's frame (the half, or its flow grid)");
    nr.gmH = (int)f.d[1];
    nr.gmW = (int)f.d[2];
    // `_m` (slot 7): A's four inputs -> mf / mq / mk, fp16 [2, N, 128] on the eighth grid (k_attn2 reads only fp16);
    // A takes the propagation's output as prop [2, N, 2], fp16 or fp32
    nvinfer1::ICudaEngine* m = nr.engGm[7];
    if (m->getNbIOTensors() != 7)
        return bad("gmflow_bidir_a_m", "does not have its 7 tensors (map4, map8, img0, img1 -> mf, mq, mk)");
    for (const char* nm : {"map4", "map8", "img0", "img1"})
        if (m->getTensorIOMode(nm) != nvinfer1::TensorIOMode::kINPUT ||
            !same(m->getTensorShape(nm), a->getTensorShape(nm)) || m->getTensorDataType(nm) != a->getTensorDataType(nm))
            return bad(nm, "is not the same input of gmflow_bidir_a_m and gmflow_bidir_a");
    const nvinfer1::Dims mt = m->getTensorShape("mf");
    for (const char* nm : {"mf", "mq", "mk"})
        if (m->getTensorIOMode(nm) != nvinfer1::TensorIOMode::kOUTPUT ||
            m->getTensorDataType(nm) != nvinfer1::DataType::kHALF || !same(m->getTensorShape(nm), mt))
            return bad(nm, "is not an fp16 output of gmflow_bidir_a_m shaped like mf");
    if (mt.nbDims != 3 || mt.d[0] != 2 || mt.d[2] != 128 || mt.d[1] != (int64_t)(gh / 8) * (gw / 8))
        return bad("mf", "is not [2, N, 128] on the eighth of gmflow's frame");
    const nvinfer1::Dims pr = a->getTensorShape("prop");
    const nvinfer1::DataType pt = a->getTensorDataType("prop");
    if (a->getTensorIOMode("prop") != nvinfer1::TensorIOMode::kINPUT || pr.nbDims != 3 || pr.d[0] != 2 ||
        pr.d[1] != mt.d[1] || pr.d[2] != 2 || (pt != nvinfer1::DataType::kHALF && pt != nvinfer1::DataType::kFLOAT))
        return bad("prop", "is not a [2, N, 2] fp16 / fp32 input of gmflow_bidir_a");
    nr.gmN8 = (int)mt.d[1];
    nr.gmW8 = gw / 8;
    nr.gmPropHalf = pt == nvinfer1::DataType::kHALF;
    char attnEv[4] = {};
    nr.gmAttnB = nr.gmN8 <= kAttn2bMaxN &&
                 !(GetEnvironmentVariableA("SMV_GM_ATTN2B", attnEv, sizeof(attnEv)) > 0 && attnEv[0] == '0');
    const int64_t hw = (int64_t)nr.gmH * nr.gmW;
    if (q.nbDims != 3 || q.d[0] != 2 || q.d[1] != hw || q.d[2] != 128)
        return bad("q", "is not [2, H * W, 128]");
    if (g.nbDims != 4 || g.d[0] != 2 || g.d[1] != hw || g.d[2] < 1 || g.d[2] > 100 || g.d[3] != 2)
        return bad("coords", "is not [2, H * W, K, 2] with K <= 100");
    nr.gmK = (int)g.d[2];
    if (c.nbDims != 3 || c.d[0] != 2 || c.d[1] != hw || c.d[2] != nr.gmK)
        return bad("corr", "is not [2, H * W, K]");
    for (int i = 0; i < b->getNbIOTensors(); i++)
    {
        const char* nm = b->getIOTensorName(i);
        if (b->getTensorIOMode(nm) != nvinfer1::TensorIOMode::kINPUT)
        {
            if (strcmp(nm, "flow"))
                return bad(nm, "is an output of gmflow_bidir_b other than flow");
            continue;
        }
        if (!strcmp(nm, "corr") || !strcmp(nm, "img0") || !strcmp(nm, "img1"))
            continue;
        int k = 3;
        while (k < 8 && strcmp(nm, kGmA[k]))
            k++;
        if (k == 8)
            return bad(nm, "is a gmflow_bidir_b input gmflow_bidir_a does not hand over");
        if (!same(a->getTensorShape(nm), b->getTensorShape(nm)) || a->getTensorDataType(nm) != b->getTensorDataType(nm))
            return bad(nm, "differs between gmflow_bidir_a and gmflow_bidir_b");
    }
    return true;
}

// one engine tensor's bytes at its static shape
static size_t trtTensorBytes(nvinfer1::ICudaEngine* eng, const char* nm)
{
    const nvinfer1::Dims d = eng->getTensorShape(nm);
    size_t n = trtElemSize(eng->getTensorDataType(nm));
    for (int k = 0; k < d.nbDims; k++)
        n *= (size_t)(d.d[k] > 0 ? d.d[k] : 0);
    return n;
}

// GMFSS: the chain's contract and its buffers. Every channel count and dtype is
// READ OFF THE ENGINES (the handoff line never types them), then one pair's and
// one tween's planar buffers are allocated from it. Runs inside nativeTrtInit right after the
// five warm-ups, so a mismatch refuses the session before any group. The accumulator is one
// buffer for all eight splats (they are strictly sequential on the one stream): the widest is
// the half level's gmC[0] + 1 planes, and the quarter / eighth levels need less
// ((2 * gmC[1] + 1) / 4 and (2 * gmC[2] + 1) / 16 of a half plane at the stock 64/128/192).
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
    // one tensor: 4 dims, the expected dtype (kFLOAT unless dtOut takes the answer) and the
    // expected shape (a negative expectation means "read it", n = the leading dim)
    int dims[4]{};
    auto tensor = [&](int e, const char* nm, int n, int c, int h, int w, bool* dtOut) -> bool {
        const nvinfer1::Dims s = nr.engGm[e]->getTensorShape(nm);
        if (s.nbDims != 4)
        {
            LOG("native: gmfss chain: %s %s is not a 4D tensor\n", kGmName[e], nm);
            return false;
        }
        for (int i = 0; i < 4; i++)
            dims[i] = (int)s.d[i];
        const nvinfer1::DataType t = nr.engGm[e]->getTensorDataType(nm);
        if (dtOut)
            *dtOut = t == nvinfer1::DataType::kHALF;
        if (!dtOut && t != nvinfer1::DataType::kFLOAT)
        {
            LOG("native: gmfss chain: %s %s is %s, the chain needs fp32\n", kGmName[e], nm, trtDtypeName(t));
            return false;
        }
        if (dtOut && t != nvinfer1::DataType::kHALF && t != nvinfer1::DataType::kFLOAT)
        {
            LOG("native: gmfss chain: %s %s dtype %s unsupported\n", kGmName[e], nm, trtDtypeName(t));
            return false;
        }
        if (dims[0] != n || (c >= 0 && dims[1] != c) || (h >= 0 && dims[2] != h) || (w >= 0 && dims[3] != w))
        {
            LOG("native: gmfss chain: %s %s is %dx%dx%dx%d, expected %dx%dx%dx%d\n", kGmName[e], nm, dims[0], dims[1],
                dims[2], dims[3], n, c, h, w);
            return false;
        }
        return true;
    };
    const int hh = nr.hh, hw = nr.hw;
    bool fh2 = false, fh3 = false, mh1 = false;
    // feat_ext: the padded frame in, the three feature levels out (64 / 128 / 192 at stock)
    if (!tensor(0, "x", 1, 3, nr.ph, nr.pw, nullptr))
        return false;
    if (!tensor(0, "f1", 1, -1, hh, hw, &nr.gmFeatHalf))
        return false;
    nr.gmC[0] = dims[1];
    if (!tensor(0, "f2", 1, -1, hh / 2, hw / 2, &fh2))
        return false;
    nr.gmC[1] = dims[1];
    if (!tensor(0, "f3", 1, -1, hh / 4, hw / 4, &fh3))
        return false;
    nr.gmC[2] = dims[1];
    if (fh2 != nr.gmFeatHalf || fh3 != nr.gmFeatHalf)
        return bad("the feature levels have mixed dtypes");
    // the fused bidir GMFlow: both halves take the frames, B hands the flow out: row 0 = flow01, row 1 = flow10 (so
    // the pyramids take both at once); at a flow grid (live --gmfss-flow) the three gmflow engines run on it
    if (nr.gfh && (nr.gfh > hh || nr.gfw > hw || (nr.gfh & 31) || (nr.gfw & 31)))
        return bad("the flow grid is not a /32 grid inside the half frame");
    const int gh = nr.gfh ? nr.gfh : hh, gw = nr.gfh ? nr.gfw : hw;
    if (!tensor(1, "img0", 1, 3, gh, gw, nullptr) || !tensor(1, "img1", 1, 3, gh, gw, nullptr) ||
        !tensor(5, "img0", 1, 3, gh, gw, nullptr) || !tensor(5, "img1", 1, 3, gh, gw, nullptr) ||
        !tensor(6, "img0", 1, 3, gh, gw, nullptr))
        return false;
    if (!tensor(5, "flow", 2, 2, gh, gw, nullptr) || !nativeGmSplitCheck(nr))
        return false;
    if (!tensor(2, "i0", 1, 3, hh, hw, nullptr) || !tensor(2, "i1", 1, 3, hh, hw, nullptr) ||
        !tensor(2, "f01", 1, 2, hh, hw, nullptr) || !tensor(2, "f10", 1, 2, hh, hw, nullptr))
        return false;
    if (!tensor(2, "m0", 1, 1, hh, hw, &nr.gmMetricHalf))
        return false;
    if (!tensor(2, "m1", 1, 1, hh, hw, &mh1))
        return false;
    if (mh1 != nr.gmMetricHalf)
        return bad("the two metrics have different dtypes");
    // the GMFSS IFNet: the 6-plane half buffer and the scalar timestep in, `rife` out
    if (!tensor(3, "x", 1, 6, hh, hw, nullptr) || !tensor(3, "merged", 1, 3, hh, hw, nullptr))
        return false;
    {
        const nvinfer1::Dims s = nr.engGm[3]->getTensorShape("timestep");
        if (s.nbDims != 4 || s.d[0] != 1 || s.d[1] != 1 || s.d[2] != 1 || s.d[3] != 1 ||
            nr.engGm[3]->getTensorDataType("timestep") != nvinfer1::DataType::kFLOAT)
            return bad("the ifnet timestep is not a 1x1x1x1 fp32 value");
    }
    // fusionnet: the concats are adjacent planes, so b / c / d are exactly 2x the feature levels; k_splatNorm writes
    // them in the engine's dtype (fp16: its store rounds to nearest even, as an ONNX Cast to fp16 does)
    bool ch = false, dh = false;
    if (!tensor(4, "a", 1, 9, hh, hw, nullptr) || !tensor(4, "b", 1, 2 * nr.gmC[0], hh, hw, &nr.gmFuseHalf) ||
        !tensor(4, "c", 1, 2 * nr.gmC[1], hh / 2, hw / 2, &ch) ||
        !tensor(4, "d", 1, 2 * nr.gmC[2], hh / 4, hw / 4, &dh))
        return false;
    if (ch != nr.gmFuseHalf || dh != nr.gmFuseHalf)
        return bad("the fusionnet feature inputs have mixed dtypes");
    if (!tensor(4, "out", 1, 3, nr.ph, nr.pw, &nr.gmOutHalf))
        return false;

    const size_t hp = (size_t)hh * hw, qp = hp / 4, ep = hp / 16;
    const size_t plane = (size_t)nr.ph * nr.pw;
    const size_t fe = nr.gmFeatHalf ? 2 : 4, me = nr.gmMetricHalf ? 2 : 4, oe = nr.gmOutHalf ? 2 : 4,
                 be = nr.gmFuseHalf ? 2 : 4;
    size_t accPlanes = (size_t)nr.gmC[0] + 1;
    if (((size_t)nr.gmC[1] + 1) * qp > accPlanes * hp)
        accPlanes = (((size_t)nr.gmC[1] + 1) * qp + hp - 1) / hp;
    if (((size_t)nr.gmC[2] + 1) * ep > accPlanes * hp)
        accPlanes = (((size_t)nr.gmC[2] + 1) * ep + hp - 1) / hp;
    size_t freeB = 0, freeA = 0, totB = 0;
    cudaMemGetInfo(&freeB, &totB);
    for (int s = 0; s < 2; s++)
        for (int l = 0; l < 3; l++)
            NCHK(cudaMalloc(&nr.dGmFeat[s][l], (size_t)nr.gmC[l] * (l == 0 ? hp : (l == 1 ? qp : ep)) * fe),
                 "alloc gmfss features");
    NCHK(cudaMalloc((void**)&nr.dGmHalf, 6 * hp * sizeof(float)), "alloc gmfss halves");
    NCHK(cudaMalloc((void**)&nr.dGmFlow, 4 * hp * sizeof(float)), "alloc gmfss flow");
    if (nr.gfh)
    {
        const size_t gp = (size_t)nr.gfh * nr.gfw;
        NCHK(cudaMalloc((void**)&nr.dGmHalfF, 6 * gp * sizeof(float)), "alloc gmfss flow grid halves");
        NCHK(cudaMalloc((void**)&nr.dGmFlowF, 4 * gp * sizeof(float)), "alloc gmfss flow grid flow");
        LOG("native: gmfss flow grid %dx%d of the %dx%d half (--gmfss-flow)\n", nr.gfw, nr.gfh, hw, hh);
    }
    NCHK(cudaMalloc(&nr.dGmMetric, 2 * hp * me), "alloc gmfss metric");
    NCHK(cudaMalloc((void**)&nr.dGmFlowP[0], 4 * qp * sizeof(float)), "alloc gmfss flow pyramid");
    NCHK(cudaMalloc((void**)&nr.dGmFlowP[1], 4 * ep * sizeof(float)), "alloc gmfss flow pyramid");
    NCHK(cudaMalloc((void**)&nr.dGmMetP[0], 2 * qp * sizeof(float)), "alloc gmfss metric pyramid");
    NCHK(cudaMalloc((void**)&nr.dGmMetP[1], 2 * ep * sizeof(float)), "alloc gmfss metric pyramid");
    NCHK(cudaMalloc((void**)&nr.dGmAcc, accPlanes * hp * sizeof(long long)), "alloc gmfss splat accumulator");
    NCHK(cudaMalloc((void**)&nr.dGmFa, 9 * hp * sizeof(float)), "alloc gmfss fusion a");
    NCHK(cudaMalloc(&nr.dGmFb, (size_t)2 * nr.gmC[0] * hp * be), "alloc gmfss fusion b");
    NCHK(cudaMalloc(&nr.dGmFc, (size_t)2 * nr.gmC[1] * qp * be), "alloc gmfss fusion c");
    NCHK(cudaMalloc(&nr.dGmFd, (size_t)2 * nr.gmC[2] * ep * be), "alloc gmfss fusion d");
    NCHK(cudaMalloc((void**)&nr.dGmT, sizeof(float)), "alloc gmfss timestep");
    NCHK(cudaMalloc(&nr.dGmOut, 3 * plane * oe), "alloc gmfss fusion out");
    NCHK(cudaMalloc((void**)&nr.dGmF, 3 * plane * sizeof(float)), "alloc gmfss model frame");
    for (int i = 0; i < 8; i++)
        NCHK(cudaMalloc(&nr.dGmA[i], trtTensorBytes(nr.engGm[1], kGmA[i])), "alloc gmfss split tensors");
    NCHK(cudaMalloc((void**)&nr.dGmCorr, (size_t)2 * nr.gmH * nr.gmW * nr.gmK * sizeof(float)),
         "alloc gmfss local correlation");
    static const char* const kGmM[3] = {"mf", "mq", "mk"};
    for (int i = 0; i < 3; i++)
        NCHK(cudaMalloc(&nr.dGmM[i], trtTensorBytes(nr.engGm[7], kGmM[i])), "alloc gmfss matching operands");
    NCHK(cudaMalloc((void**)&nr.dGmFlow8, (size_t)2 * nr.gmN8 * 2 * sizeof(float)), "alloc gmfss matching flow");
    NCHK(cudaMalloc(&nr.dGmProp, trtTensorBytes(nr.engGm[1], "prop")), "alloc gmfss propagated flow");
    for (int i = 0; i < 2; i++)
    {
        const size_t bytes = trtTensorBytes(nr.engGm[1], i ? "map8" : "map4");
        NCHK(cudaMalloc(&nr.dGmMap[i], bytes), "alloc gmfss backbone maps");
        nr.gmMapSlot[i] = bytes / 2;
    }
    nr.gmMapsNext = false;
    if (nr.encPost)
    {
        NCHK(cudaMalloc((void**)&nr.dGmEnc, 3 * plane * sizeof(float)), "alloc gmfss SDR view");
        NCHK(cudaMalloc((void**)&nr.dGmHalfM, 6 * hp * sizeof(float)), "alloc gmfss motion halves");
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
    LOG("native: gmfss chain: gmflow split, its local correlation in one kernel (%dx%d, %d samples a pixel)\n", nr.gmW,
        nr.gmH, nr.gmK);
    LOG("native: gmfss chain: the global matching by %s (%d tokens a frame)\n", nr.gmAttnB ? "k_attn2b" : "k_attn2",
        nr.gmN8);
    nr.gmProf = GetEnvironmentVariableW(L"SMV_LIVE_GMFSS_PROF", nullptr, 0) != 0;
    if (nr.gmProf)
        for (auto& e : nr.gmEv)
            NCHK(cudaEventCreate(&e), "create gmfss profile event");
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
// holds this t is left as it is (x2 asks for 0.5 every group, an offline batch for the same t per plane)
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
    NCHK(cudaMalloc(&nr.dT, b * mplane * (nr.tHalf ? sizeof(uint16_t) : sizeof(float))), "alloc timestep");
    nr.tHeldMask = 0;
    NCHK(cudaMalloc((void**)&nr.dFlow, b * 4 * mplane * sizeof(uint16_t)), "alloc flow");
    NCHK(cudaMalloc((void**)&nr.dMask, b * mplane * sizeof(uint16_t)), "alloc mask");
    nr.blendOut = !g_offline && !nr.gmfss && nr.w == nr.dw && nr.h == nr.dh && !nr.uw;
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

// RTX HDR's and RTX VSR's video memory in MiB, by the megapixels of the model frame they take in (the
// bridge's NGX features and their frames, a quarter on top of the measured: a live session's own
// dedicated memory with and without the feature, 82 / 226 MiB for RTX HDR at 854x480 / 1920x1080,
// 174 / 524 MiB for RTX VSR from those into a 2560x1440 Fill)
constexpr double kRtxHdrBase = 58.0, kRtxHdrMp = 109.0, kRtxVsrBase = 109.0, kRtxVsrMp = 264.0;
// AMD FSR upscaling (FSR 4.1.1): RTX VSR's row moved by a quarter over the process's own memory against VSR's, at
// 854x480 and 1708x960 upscaled 2.25x (+243 / +793 MiB against VSR's +108 / +402 over no upscaler)
constexpr double kFsrUpBase = 171.0, kFsrUpMp = 524.0;

// Restore's video memory in MiB a megapixel of the decoded frame (its engine 145 MiB at 854x480, a quarter on top)
constexpr double kRestoreMp = 445.0;

// megapixels of a frame padded to 64 a side (the engines' and the model buffers' shape)
static double padMp64(int a, int b)
{
    return (double)((a + 63) / 64 * 64) * ((b + 63) / 64 * 64) / 1e6;
}

// An offline RIFE-family render with one tween a call, in MiB: 276 + 427 a padded megapixel of the IFNet frame (its
// activations and the encoder) + 224 a padded megapixel of the pictures at the working size (572 / 1624 MiB measured
// at 854x480 / 1920x1080 with the tween stored in fp16, plus 6 for its fp32 store: 3 channels x 2 more bytes a pixel).
// ifw x ifh = the IFNet's frame, w x h = the working size.
static double nativeOfflineRifeMiB(int ifw, int ifh, int w, int h)
{
    return 276.0 + 427.0 * padMp64(ifw, ifh) + 224.0 * padMp64(w, h);
}

// The passes around an offline model, in MiB: DLSS 5's at the working size w x h (nr_host's per-pass constants), RTX
// HDR, RTX VSR, and Restore at the decoded frame sw x sh.
static double nativeOfflineEffectsMiB(int w, int h, int sw, int sh)
{
    const double mp = (double)w * h / 1e6;
    double need = 0.0;
    if (g_dlssnr)
    {
        const int p = g_nrPasses < 1 ? 1 : (g_nrPasses > nr::kMaxPasses ? nr::kMaxPasses : g_nrPasses);
        need += p * (nr::kPassBase + nr::kPassMp * mp) + nr::kAfterBase + nr::kAfterMp * mp + nr::kFrameMp * mp * p;
    }
    if (g_rtxHdr)
        need += kRtxHdrBase + kRtxHdrMp * mp;
    if (g_rtxVsr)
        need += kRtxVsrBase + kRtxVsrMp * mp;
    else if (g_fsrUp)
        need += kFsrUpBase + kFsrUpMp * mp;
    if (g_restore)
        need += kRestoreMp * (double)sw * sh / 1e6;
    return need;
}

// What an offline RIFE-family render takes besides its IFNet's extra timesteps, in bytes, before any engine or
// buffer exists (lkOfflineRife's batch choice): the render with one tween a call and the passes around it.
// ifw x ifh = the IFNet's frame, w x h = the working size, sw x sh = the decoded frame.
static uint64_t nativeOfflineNeed(int ifw, int ifh, int w, int h, int sw, int sh)
{
    return (uint64_t)((nativeOfflineRifeMiB(ifw, ifh, w, h) + nativeOfflineEffectsMiB(w, h, sw, sh)) * 1048576.0);
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
// the Lanczos pair of a planar source into a planar target in ONE kernel (k_fitAaTile: the pair's bytes, no global
// tmp), 32 columns by the tallest of 32 / 16 / 8 rows whose tables and tile fit the 48 KB of shared memory a launch
// gets without opting in; launched = false when none fits (the caller's own path then). false = a launch failed.
static bool nativeFitAaTile(NativeRife& nr, const void* src, int half, int ps, int rs, int sw, int sh, float* dst,
                            int tw, int th, cudaStream_t st, bool& launched)
{
    launched = false;
    const int fsh = lzTaps(sw, tw), fsv = lzTaps(sh, th);
    for (int by = 32; by >= 8; by /= 2)
    {
        int rcap = (int)(((long long)(by - 1) * sh + th - 1) / th) + fsv + 2;
        const size_t smem = ((size_t)fsh * 32 + (size_t)fsv * by) * 8 + 8 + (size_t)12 * rcap * 32;
        if (smem > 48 * 1024)
            continue;
        void* at[] = {(void*)&src, &half, &ps, &rs, &sw, &sh, &dst, &tw, &th, &rcap};
        if (cuLaunchKernel(nr.fFitAaTile, (tw + 31) / 32, (th + by - 1) / by, 1, 32, by, 1, (unsigned)smem,
                           (CUstream)st, at, nullptr) != CUDA_SUCCESS)
        {
            nr.die("fitAaTile launch failed");
            return false;
        }
        launched = true;
        return true;
    }
    return true;
}

// one planar Lanczos resize: a size change of at most 3x per axis = k_fitAaTile (the same bytes as sampleOut's 2D
// gather, 2 to 2.5x faster at 1.2x to 3x: 2 x 2 kLzUp taps a pixel on an enlarge where the gather reads
// (2 kLzUp)^2), else k_fitPlanar (sampleOut). false = a launch failed.
static bool nativeFitPlanar(NativeRife& nr, const void* src, int half, int ps, int rs, int sw, int sh, float* dst,
                            int tw, int th, cudaStream_t st)
{
    bool tiled = false;
    if ((tw != sw || th != sh) && (long long)tw <= 3LL * sw && (long long)th <= 3LL * sh &&
        !nativeFitAaTile(nr, src, half, ps, rs, sw, sh, dst, tw, th, st, tiled))
        return false;
    if (tiled)
        return true;
    void* a[] = {(void*)&src, &half, &ps, &rs, &sw, &sh, &dst, &tw, &th};
    return cuLaunchKernel(nr.fFitPlanar, (tw + 15) / 16, (th + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a, nullptr) ==
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
// tap clamps at the picture's content).
static bool nativeRifeBlend(NativeRife& nr, const void* pic, int half, int n, cudaStream_t st, int sw = 0)
{
    int pps = nr.ph * nr.pw, prs = nr.pw, w = nr.w, h = nr.h;
    int mps = nr.mph ? nr.mph * nr.mpw : pps, mrs = nr.mph ? nr.mpw : nr.pw;
    int fw = nr.mph ? nr.mw : w, fh = nr.mph ? nr.mh : h;
    int cw = (fw == w && fh == h) ? nr.pw : w, ch = (fw == w && fh == h) ? nr.ph : h;
    void* a[] = {(void*)&pic, &sw,       &half,     &pps, &prs, &w,  &h,  &cw,
                 &ch,         &nr.dFlow, &nr.dMask, &mps, &mrs, &fw, &fh, &nr.dMerged};
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
    int hdr = nr.hdr ? 1 : 0, pitch = (int)nr.pitch;
    void* a[] = {(void*)&pic,  &sw,  &half, &pps, &prs, &w,   &h,    &cw,    &ch,    (void*)&flow,
                 (void*)&mask, &mps, &mrs,  &fw,  &fh,  &hdr, &slot, &pitch, &nr.x0, &nr.y0};
    void* a32[] = {(void*)&pic, &sw,  &pps, &prs, &w,   &h,           &cw,    &ch,    (void*)&flow, (void*)&mask,
                   &mps,        &mrs, &fw,  &fh,  &hdr, (void*)&slot, &pitch, &nr.x0, &nr.y0};
    return cuLaunchKernel(half ? nr.fRifeBlendOut : nr.fRifeBlendOut32, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0,
                          (CUstream)st, half ? a : a32, nullptr) == CUDA_SUCCESS;
}

static bool nativeTrtInit(NativeRife& nr)
{
    // resident host: the same engine pair stays loaded across sessions, only the two
    // execution contexts are recreated (measured: engine deserialize 0.10 s +
    // jit cache 1.0 to 3.5 s per start against 0.3 to 0.8 s for the contexts alone)
    // live Restore: the Real-ESRGAN engine loads beside the pair when the handoff
    // named one, its dtypes read off the engine (x / y are fp16 as python exports the .half()
    // net; fp32 is accepted too), and one warm enqueue on zeros so the first presented frame
    // pays no kernel specialisation (the merged jit cache makes that cheap)
    auto dt = [](nvinfer1::ICudaEngine* e, const char* n) { return e->getTensorDataType(n); };
    auto restoreReady = [&]() -> bool {
        if (!nr.engRest)
            return true;
        const auto xd = dt(nr.engRest, "x"), yd = dt(nr.engRest, "y");
        if (xd == nvinfer1::DataType::kHALF)
            nr.restHalfIn = true;
        else if (xd == nvinfer1::DataType::kFLOAT)
            nr.restHalfIn = false;
        else
        {
            LOG("native: restore engine input dtype unsupported\n");
            return false;
        }
        if (yd == nvinfer1::DataType::kHALF)
            nr.restHalfOut = true;
        else if (yd == nvinfer1::DataType::kFLOAT)
            nr.restHalfOut = false;
        else
        {
            LOG("native: restore engine output dtype unsupported\n");
            return false;
        }
        if (!nr.dRestIn || !nr.dRestOut)
        {
            LOG("native: restore buffers missing\n");
            return false;
        }
        NCHK(cudaMemsetAsync(nr.dRestIn, 0, (size_t)3 * srcW(nr) * srcH(nr) * 4, nr.stream), "restore warm-up clear");
        nvinfer1::Dims4 din{1, 3, srcH(nr), srcW(nr)};
        if (!nr.ctxRest->setInputShape("x", din))
        {
            LOG("native: restore setInputShape rejected (engine built for another size)\n");
            return false;
        }
        nr.ctxRest->setTensorAddress("x", nr.dRestIn);
        nr.ctxRest->setTensorAddress("y", nr.dRestOut);
        if (!nr.ctxRest->enqueueV3(nr.stream))
        {
            LOG("native: restore warm-up enqueue failed\n");
            return false;
        }
        NCHK(cudaStreamSynchronize(nr.stream), "restore warm-up sync");
        return true;
    };
    // native DRBA: the block0 engine loads beside the pair like the restore
    // engine (resident per path; any session without DRBA drops it, nothing of another route
    // idles in VRAM), its fp32 contract read off the engine, one warm enqueue on zeros
    auto block0Ready = [&]() -> bool {
        if (!nr.drba)
        {
            if (g_res.engB0)
            {
                delete g_res.engB0;
                g_res.engB0 = nullptr;
                g_res.block0Path.clear();
            }
            return true;
        }
        if (g_resident && g_res.engB0 && g_res.block0Path == nr.block0Path)
        {
            nr.engB0 = g_res.engB0;
            if (!nativeMakeContext(nr, nr.engB0, &nr.cfgB0, &nr.ctxB0))
                return false;
        }
        else
        {
            if (g_res.engB0)
            {
                delete g_res.engB0;
                g_res.engB0 = nullptr;
                g_res.block0Path.clear();
            }
            nr.engB0 = nativeLoadEngine(nr, nr.block0Path, &nr.cfgB0, &nr.ctxB0);
            if (!nr.engB0)
                return false;
            if (g_resident)
            {
                g_res.engB0 = nr.engB0;
                g_res.block0Path = nr.block0Path;
            }
        }
        for (const char* n : {"img0", "img1", "flow"})
            if (dt(nr.engB0, n) != nvinfer1::DataType::kFLOAT)
            {
                LOG("native: block0 tensor %s is not fp32\n", n);
                return false;
            }
        // block0 reads the same encode ring as the IFNet: its f0 / f1 dtype must match
        const auto bfd = nr.featHalf ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT;
        if (dt(nr.engB0, "f0") != bfd || dt(nr.engB0, "f1") != bfd)
        {
            LOG("native: block0 f0 / f1 dtype differs from the IFNet's\n");
            return false;
        }
        return nativeWarmEngine(nr, nr.engB0, nr.ctxB0, "block0", "drba");
    };
    // GMFSS: the five-engine set, resident like the pair (the same paths =
    // reuse, contexts recreated; anything else frees whatever set was resident and loads this
    // one), each engine warmed once on zeros; the restore engine is per session here. The warm
    // enqueue also leaves every static input shape set on the context, which is why the chain
    // itself only sets tensor ADDRESSES per group. nativeGmfssSetup then reads the contract off
    // the engines and allocates the chain's buffers (per session, freed in nativeFree).
    if (nr.gmfss)
    {
        const int64_t t0 = nowQpc100();
        size_t freeB = 0, freeA = 0, totB = 0;
        cudaMemGetInfo(&freeB, &totB);
        bool same = g_resident && g_res.rt;
        for (int i = 0; i < kGmN && same; i++)
            same = g_res.engGm[i] && g_res.gmPath[i] == nr.gmPath[i];
        if (same)
        {
            nr.rt = g_res.rt;
            nr.jit = g_res.jit;
            for (int i = 0; i < kGmN; i++)
            {
                nr.engGm[i] = g_res.engGm[i];
                if (!nativeMakeContext(nr, nr.engGm[i], &nr.cfgGm[i], &nr.ctxGm[i]))
                    return false;
            }
        }
        else
        {
            if (g_resident && g_res.rt)
                residentFreeEngines(); // the RIFE pair or another set
            nr.rt = nvinfer1::createInferRuntime(g_nativeTrtLogger);
            if (!nr.rt)
            {
                LOG("native: createInferRuntime failed\n");
                return false;
            }
            for (int i = 0; i < kGmN; i++)
            {
                nr.engGm[i] = nativeLoadEngine(nr, nr.gmPath[i], &nr.cfgGm[i], &nr.ctxGm[i]);
                if (!nr.engGm[i])
                    return false;
            }
        }
        if (g_res.engRest)
        {
            delete g_res.engRest;
            g_res.engRest = nullptr;
            g_res.restorePath.clear();
        }
        if (!nr.restorePath.empty())
        {
            nr.engRest = nativeLoadEngine(nr, nr.restorePath, &nr.cfgRest, &nr.ctxRest);
            if (!nr.engRest)
                return false;
        }
        for (int i = 0; i < kGmN; i++)
            if (!nativeWarmEngine(nr, nr.engGm[i], nr.ctxGm[i], kGmName[i]))
                return false;
        if (!restoreReady())
            return false;
        cudaMemGetInfo(&freeA, &totB);
        LOG("native: gmfss engine set %s in %.2f s, VRAM %+.0f MB\n",
            same ? "reused (resident), contexts recreated" : "loaded", (nowQpc100() - t0) / 1e7,
            ((double)freeB - (double)freeA) / 1048576.0);
        if (!nativeGmfssSetup(nr))
            return false;
        if (g_resident)
        {
            g_res.rt = nr.rt;
            g_res.jit = nr.jit;
            g_res.dev = nr.dev;
            for (int i = 0; i < kGmN; i++)
            {
                g_res.engGm[i] = nr.engGm[i];
                g_res.gmPath[i] = nr.gmPath[i];
            }
            g_res.ifnetPath.clear();
            g_res.encodePath.clear();
            g_res.jitPath.clear();
        }
        return true;
    }
    if (g_resident && g_res.rt && g_res.ifnetPath == nr.ifnetPath && g_res.encodePath == nr.encodePath)
    {
        const int64_t t0 = nowQpc100();
        nr.rt = g_res.rt;
        nr.engIf = g_res.engIf;
        nr.engEnc = g_res.engEnc;
        nr.jit = g_res.jit;
        nr.encHalf = g_res.encHalf;
        nr.featHalf = g_res.featHalf;
        nr.xHalf = g_res.xHalf;
        nr.imgHalf = g_res.imgHalf;
        nr.outHalf = g_res.outHalf;
        nr.tHalf = g_res.tHalf;
        // no-engine mode: the resident holds no pair (both paths empty on both sides), only
        // the runtime, the jit cache and, per path, the restore engine
        if (!nr.noEngine && !nr.nvof && !nr.fruc && !nr.dlssg)
        {
            if (!nativeMakeContext(nr, nr.engIf, &nr.cfgIf, &nr.ctxIf))
                return false;
            if (!nativeMakeContext(nr, nr.engEnc, &nr.cfgEnc, &nr.ctxEnc))
                return false;
            if (!nativeTweenBuffers(nr))
                return false;
        }
        // the restore engine follows its own path: kept when the same one is asked for again,
        // dropped for a session without Restore, loaded for one that brings a different path
        if (g_res.restorePath != nr.restorePath)
        {
            if (g_res.engRest)
            {
                delete g_res.engRest;
                g_res.engRest = nullptr;
            }
            g_res.restorePath.clear();
            if (!nr.restorePath.empty())
            {
                nr.engRest = nativeLoadEngine(nr, nr.restorePath, &nr.cfgRest, &nr.ctxRest);
                if (!nr.engRest)
                    return false;
                g_res.engRest = nr.engRest;
                g_res.restorePath = nr.restorePath;
            }
        }
        else if (g_res.engRest)
        {
            nr.engRest = g_res.engRest;
            if (!nativeMakeContext(nr, nr.engRest, &nr.cfgRest, &nr.ctxRest))
                return false;
        }
        if (!restoreReady() || !block0Ready())
            return false;
        LOG("native: resident engines reused, contexts recreated in %.2f s\n", (nowQpc100() - t0) / 1e7);
        return true;
    }
    if (g_resident && g_res.rt)
        residentFreeEngines(); // a different pair (new window size)
    nr.rt = nvinfer1::createInferRuntime(g_nativeTrtLogger);
    if (!nr.rt)
    {
        LOG("native: createInferRuntime failed\n");
        return false;
    }
    if (!nr.noEngine && !nr.nvof && !nr.fruc && !nr.dlssg)
    {
        nr.engIf = nativeLoadEngine(nr, nr.ifnetPath, &nr.cfgIf, &nr.ctxIf);
        if (!nr.engIf)
            return false;
        nr.engEnc = nativeLoadEngine(nr, nr.encodePath, &nr.cfgEnc, &nr.ctxEnc);
        if (!nr.engEnc)
            return false;
    }
    if (!nr.restorePath.empty())
    {
        nr.engRest = nativeLoadEngine(nr, nr.restorePath, &nr.cfgRest, &nr.ctxRest);
        if (!nr.engRest)
            return false;
    }
    if (!nr.noEngine && !nr.nvof && !nr.fruc && !nr.dlssg)
    {
        // dtype contract, read off the engines rather than trusted from the handoff line
        // timestep: fp32 or fp16, read off the engine (fp16: the constant fills and DRBA's map are written in fp16)
        const auto td = dt(nr.engIf, "timestep");
        if (td != nvinfer1::DataType::kFLOAT && td != nvinfer1::DataType::kHALF)
        {
            LOG("native: IFNet timestep dtype unsupported\n");
            return false;
        }
        nr.tHalf = td == nvinfer1::DataType::kHALF;
        // flow / mask: fp16, the tween is made on the host by RIFE's last step (nativeRifeBlend); an engine
        // without them is refused (a missing name reads as fp32)
        if (dt(nr.engIf, "flow") != nvinfer1::DataType::kHALF || dt(nr.engIf, "mask") != nvinfer1::DataType::kHALF)
        {
            LOG("native: the IFNet engine hands out no fp16 flow / mask (built from an ONNX before rev 7): empty "
                "the engine cache\n");
            return false;
        }
        // the tweens in fp32, each reader rounds once; SMV_TWEEN_FP16=1 = the fp16 store (a second rounding)
        nr.outHalf = lkEnv("SMV_TWEEN_FP16") == "1";
        // x: fp32 or fp16, read off the engine (fp16: bound to the fp16 copy of the frame pair)
        const auto xd = dt(nr.engIf, "x");
        if (xd != nvinfer1::DataType::kFLOAT && xd != nvinfer1::DataType::kHALF)
        {
            LOG("native: IFNet x dtype unsupported\n");
            return false;
        }
        nr.xHalf = xd == nvinfer1::DataType::kHALF;
        // f0 / f1: fp32 or fp16, read off the engine (fp16: the encode output fed as is)
        const auto fd = dt(nr.engIf, "f0");
        if (dt(nr.engIf, "f1") != fd || (fd != nvinfer1::DataType::kFLOAT && fd != nvinfer1::DataType::kHALF))
        {
            LOG("native: IFNet f0 / f1 dtype unsupported\n");
            return false;
        }
        nr.featHalf = fd == nvinfer1::DataType::kHALF;
        const auto ed = dt(nr.engEnc, "feat");
        if (ed == nvinfer1::DataType::kHALF)
            nr.encHalf = true;
        else if (ed == nvinfer1::DataType::kFLOAT)
            nr.encHalf = false;
        else
        {
            LOG("native: encode output dtype unsupported\n");
            return false;
        }
        if (nr.featHalf && !nr.encHalf)
        {
            LOG("native: IFNet takes fp16 features but the encode engine outputs fp32\n");
            return false;
        }
        // img: fp32 or fp16, read off the engine; either pairs with either x
        const auto id = dt(nr.engEnc, "img");
        if (id != nvinfer1::DataType::kFLOAT && id != nvinfer1::DataType::kHALF)
        {
            LOG("native: encode input dtype unsupported\n");
            return false;
        }
        nr.imgHalf = id == nvinfer1::DataType::kHALF;
        if (!nativeTweenBuffers(nr))
            return false;
    }
    // RIFE's two domains live: the motion frames are fp16, the engines must take fp16 frames
    if (!g_offline && nr.mph)
    {
        if (!nr.xHalf || !nr.imgHalf)
        {
            LOG("native: RIFE's motion frames are fp16 but the engines take fp32 frames (an ONNX before rev 4)\n");
            return false;
        }
        LOG("native: RIFE finds the motion on the picture at %dx%d%s, its warp + blend runs on the %dx%d pictures\n",
            nr.mw, nr.mh,
            nr.hdr ? (nr.rtxHdr ? " (the SDR picture before RTX HDR)" : " (the HDR desktop's PQ read as SDR sRGB)")
                   : "",
            nr.w, nr.h);
    }
    if (!restoreReady() || !block0Ready())
        return false;
    if (g_resident)
    {
        g_res.rt = nr.rt;
        g_res.engIf = nr.engIf;
        g_res.engEnc = nr.engEnc;
        g_res.engRest = nr.engRest;
        g_res.jit = nr.jit;
        g_res.encHalf = nr.encHalf;
        g_res.featHalf = nr.featHalf;
        g_res.xHalf = nr.xHalf;
        g_res.imgHalf = nr.imgHalf;
        g_res.outHalf = nr.outHalf;
        g_res.tHalf = nr.tHalf;
        g_res.dev = nr.dev;
        g_res.ifnetPath = nr.ifnetPath;
        g_res.encodePath = nr.encodePath;
        g_res.jitPath = nr.jitPath;
        g_res.restorePath = nr.restorePath;
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
        g_rtxb.created = false;
    }
    if (nr.ctxIf)
    {
        delete nr.ctxIf;
        nr.ctxIf = nullptr;
    }
    if (nr.ctxEnc)
    {
        delete nr.ctxEnc;
        nr.ctxEnc = nullptr;
    }
    if (nr.ctxRest)
    {
        delete nr.ctxRest;
        nr.ctxRest = nullptr;
    }
    if (nr.cfgIf)
    {
        delete nr.cfgIf;
        nr.cfgIf = nullptr;
    }
    if (nr.cfgEnc)
    {
        delete nr.cfgEnc;
        nr.cfgEnc = nullptr;
    }
    if (nr.cfgRest)
    {
        delete nr.cfgRest;
        nr.cfgRest = nullptr;
    }
    if (nr.engRest && nr.engRest != g_res.engRest)
        delete nr.engRest;
    nr.engRest = nullptr;
    if (nr.ctxB0)
    {
        delete nr.ctxB0;
        nr.ctxB0 = nullptr;
    }
    if (nr.cfgB0)
    {
        delete nr.cfgB0;
        nr.cfgB0 = nullptr;
    }
    if (nr.engB0 && nr.engB0 != g_res.engB0)
        delete nr.engB0;
    nr.engB0 = nullptr;
    if (nr.drTweens || nr.drHeads)
        LOG("native: drba %llu tweens (%llu head pairs), %llu block0 runs\n", (unsigned long long)nr.drTweens,
            (unsigned long long)nr.drHeads, (unsigned long long)nr.drBlock0);
    for (void* p :
         {(void*)nr.dDrI[0],      (void*)nr.dDrI[1],    (void*)nr.dDrI[2],      (void*)nr.dDrI[3],    (void*)nr.dDrF[0],
          (void*)nr.dDrF[1],      (void*)nr.dDrF[2],    (void*)nr.dDrF[3],      (void*)nr.dDrX[0],    (void*)nr.dDrX[1],
          (void*)nr.drWin[0].f10, (void*)nr.drWin[0].r, (void*)nr.drWin[1].f10, (void*)nr.drWin[1].r, (void*)nr.dDrFlow,
          (void*)nr.dDrFlowN,     (void*)nr.dDrAcc,     (void*)nr.dDrM[0],      (void*)nr.dDrM[1],    (void*)nr.dDrM[2],
          (void*)nr.dDrM[3],      (void*)nr.dDrMX[0],   (void*)nr.dDrMX[1],     (void*)nr.dDrB0})
        if (p)
            cudaFree(p);
    for (int i = 0; i < 4; i++)
    {
        nr.dDrI[i] = nr.dDrF[i] = nullptr;
        nr.dDrM[i] = nullptr;
    }
    nr.dDrX[0] = nr.dDrX[1] = nullptr;
    nr.dDrMX[0] = nr.dDrMX[1] = nullptr;
    nr.dDrB0 = nullptr;
    for (auto& wn : nr.drWin)
        wn = NativeRife::DrWin{};
    nr.dDrFlow = nr.dDrFlowN = nullptr;
    nr.dDrAcc = nullptr;
    for (int i = 0; i < kGmN; i++)
    {
        if (nr.ctxGm[i])
        {
            delete nr.ctxGm[i];
            nr.ctxGm[i] = nullptr;
        }
        if (nr.cfgGm[i])
        {
            delete nr.cfgGm[i];
            nr.cfgGm[i] = nullptr;
        }
        if (nr.engGm[i] && nr.engGm[i] != g_res.engGm[i])
            delete nr.engGm[i];
        nr.engGm[i] = nullptr;
    }
    // the resident cache owns these in --resident mode (residentFree drops them); the
    // pointer compare keeps the non-resident teardown byte for byte
    if (nr.jit && nr.jit != g_res.jit)
        delete nr.jit;
    nr.jit = nullptr;
    if (nr.engIf && nr.engIf != g_res.engIf)
        delete nr.engIf;
    nr.engIf = nullptr;
    if (nr.engEnc && nr.engEnc != g_res.engEnc)
        delete nr.engEnc;
    nr.engEnc = nullptr;
    if (nr.rt && nr.rt != g_res.rt)
        delete nr.rt;
    nr.rt = nullptr;
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
    nr.pfArmed = false;
    for (cudaEvent_t& e : nr.grEv)
        if (e)
        {
            cudaEventDestroy(e);
            e = nullptr;
        }
    nr.grArmed = false;
    // PG p.124: mappings must go before the external memory objects, and every outstanding
    // wait must have completed before the semaphore is destroyed (the stream sync above).
    if (nr.dOutRing)
    {
        cudaFree(nr.dOutRing);
        nr.dOutRing = nullptr;
    }
    if (nr.capTex)
    {
        cudaDestroyTextureObject(nr.capTex);
        nr.capTex = 0;
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
    for (void* p : {(void*)nr.dCap,    (void*)nr.dX,          (void*)nr.dXh,        (void*)nr.dF[0],
                    (void*)nr.dF[1],   (void*)nr.dEncHalf,    (void*)nr.dT,         (void*)nr.dMerged,
                    (void*)nr.dTmp,    (void*)nr.dCapF,       (void*)nr.dThdrIn,    (void*)nr.dThdrOut,
                    (void*)nr.dSrcG,   (void*)nr.dPres,       (void*)nr.dSrcPl,     (void*)nr.dVsrIn,
                    (void*)nr.dVsrOut, (void*)nr.dFitTmp,     (void*)nr.dUp,        (void*)nr.dUpTmp,
                    nr.dRestIn,        nr.dRestOut,           (void*)nr.dRestTmp,   (void*)nr.dRestF,
                    (void*)nr.dRest,   (void*)nr.dStaticFlag, (void*)nr.dThdrStats, (void*)nr.dShIn,
                    (void*)nr.dShOut})
        if (p)
            cudaFree(p);
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
    for (void* p : {nr.dGmFeat[0][0],
                    nr.dGmFeat[0][1],
                    nr.dGmFeat[0][2],
                    nr.dGmFeat[1][0],
                    nr.dGmFeat[1][1],
                    nr.dGmFeat[1][2],
                    (void*)nr.dGmHalf,
                    (void*)nr.dGmFlow,
                    (void*)nr.dGmHalfF,
                    (void*)nr.dGmFlowF,
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
                    (void*)nr.dGmF,
                    (void*)nr.dGmEnc,
                    (void*)nr.dGmHalfM,
                    (void*)nr.dGmRange,
                    nr.dGmA[0],
                    nr.dGmA[1],
                    nr.dGmA[2],
                    nr.dGmA[3],
                    nr.dGmA[4],
                    nr.dGmA[5],
                    nr.dGmA[6],
                    nr.dGmA[7],
                    (void*)nr.dGmCorr,
                    nr.dGmM[0],
                    nr.dGmM[1],
                    nr.dGmM[2],
                    (void*)nr.dGmFlow8,
                    nr.dGmProp,
                    nr.dGmMap[0],
                    nr.dGmMap[1]})
        if (p)
            cudaFree(p);
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
    nr.dGmHalfF = nullptr;
    nr.dGmFlowF = nullptr;
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
    nr.dGmF = nullptr;
    nr.dGmEnc = nullptr;
    nr.dGmHalfM = nullptr;
    nr.dCap = nullptr;
    nr.dX = nullptr;
    nr.dXh = nullptr;
    nr.dF[0] = nr.dF[1] = nullptr;
    nr.dEncHalf = nullptr;
    nr.dT = nullptr;
    nr.tHeldMask = 0;
    nr.dMerged = nullptr;
    nr.dTmp = nullptr;
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
    nr.dFitTmp = nullptr;
    nr.dUp = nullptr;
    nr.dUpTmp = nullptr;
    nr.dRestIn = nullptr;
    nr.dRestOut = nullptr;
    nr.dRestTmp = nullptr;
    nr.dRestF = nullptr;
    nr.dRest = nullptr;
    if (nr.cuMod && nr.cuMod != g_res.cuMod)
        cuModuleUnload(nr.cuMod);
    nr.cuMod = nullptr;
    if (nr.stream)
    {
        cudaStreamDestroy(nr.stream);
        nr.stream = nullptr;
    }
}

// ---- the GMFSS chain ------------------------------------------------------------------------
// GMFSS_infer_u.Model.reuse() and .inference() as native launches on the group's stream: the
// five engines and the glue kernels (k_half / k_pyr / k_splatSoft / k_splatNorm, gated against a
// float64 reference). Two differences from
// live_server.Gmfss, both exact:
//   * feat_ext runs ONCE per group (on the new frame) and the previous group's output is this
//     pair's feat0: the same reuse the native RIFE route does with the encode (feat_ext is a
//     pure function of the frame); python recomputes both sets per pair.
//   * the flow and metric pyramids are built once per PAIR and the timestep is applied by the
//     splat: python's interp(t * flow) * 0.5 equals t * (interp(flow) * 0.5).
// Everything else follows python's op order literally, including which splat takes s = t and
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
    auto featExt = [&](float* x, void** f) {
        nvinfer1::IExecutionContext* c = nr.ctxGm[0];
        c->setTensorAddress("x", x);
        c->setTensorAddress("f1", f[0]);
        c->setTensorAddress("f2", f[1]);
        c->setTensorAddress("f3", f[2]);
        return c->enqueueV3(st);
    };
    float* xCur = sdr ? nr.dGmEnc : dCur;
    if (sdr && !view(dCur, 0.0f))
    {
        nr.die("gmfss SDR view launch failed");
        return false;
    }
    if (!featExt(xCur, fs))
    {
        nr.die("gmfss feat_ext enqueueV3 returned false");
        return false;
    }
    nr.gmFeatSdr[nr.gmCur] = sdr;
    if (!needFlow)
        return true; // no tween in this group: python skips reuse() as well
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
    const bool refeat = nr.gmFeatSdr[nr.gmCur ^ 1] != sdr;
    if (!halfOf(a1) || (sdr && !view(dPrev, 0.0f)) || !halfOf(a0) || (refeat && !featExt(x0, nr.dGmFeat[nr.gmCur ^ 1])))
    {
        nr.die("gmfss half launch failed");
        return false;
    }
    nr.gmFeatSdr[nr.gmCur ^ 1] = sdr;
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
    // at a flow grid (live --gmfss-flow): gmflow reads both halves shrunk to it (one launch: mv1 = mv0 + 3 planes), B's
    // flow comes back to the half through k_gmFlowUp; metricnet and everything after it read the halves
    float* g0 = mv0;
    float* g1 = mv1;
    if (nr.gfh)
    {
        int six = 6;
        void* as[] = {(void*)&mv0, &six, &nr.hw, &nr.hh, (void*)&nr.dGmHalfF, &nr.gfw, &nr.gfh};
        if (cuLaunchKernel(nr.fGmShrink, (nr.gfw + 15) / 16, (nr.gfh + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, as,
                           nullptr) != CUDA_SUCCESS)
        {
            nr.die("gmfss flow grid shrink launch failed");
            return false;
        }
        g0 = nr.dGmHalfF;
        g1 = nr.dGmHalfF + 3 * (size_t)nr.gfh * nr.gfw;
    }
    nvinfer1::IExecutionContext* ctx = nullptr;
    if (profPair)
        cudaEventRecord(nr.gmEv[2], st);
    // gmflow's backbone once a frame: the previous frame's maps are the last pair's new ones (batch slot 1 -> 0) when
    // that pair took them on the same view of it (the pictures, the SDR view or the motion view), else taken again;
    // the new frame's into slot 1
    const int kind = mview ? 2 : (sdr ? 1 : 0);
    auto backbone = [&](float* x, int slot) {
        nvinfer1::IExecutionContext* c = nr.ctxGm[6];
        c->setTensorAddress("img0", x);
        c->setTensorAddress("map4", (uint8_t*)nr.dGmMap[0] + slot * nr.gmMapSlot[0]);
        c->setTensorAddress("map8", (uint8_t*)nr.dGmMap[1] + slot * nr.gmMapSlot[1]);
        return c->enqueueV3(st);
    };
    bool mapsOk = true;
    if (mapsPrev && nr.gmMapsKind == kind)
        for (int i = 0; i < 2 && mapsOk; i++)
            mapsOk = cudaMemcpyAsync(nr.dGmMap[i], (uint8_t*)nr.dGmMap[i] + nr.gmMapSlot[i], nr.gmMapSlot[i],
                                     cudaMemcpyDeviceToDevice, st) == cudaSuccess;
    else
        mapsOk = backbone(g0, 0);
    if (!mapsOk || !backbone(g1, 1))
    {
        nr.die("gmfss gmflow_backbone enqueueV3 returned false");
        return false;
    }
    nr.gmMapsKind = kind;
    // the fused bidir GMFlow (both directions: `_m` on both frames' maps, the scale-0 global matching and its propagation
    // (k_attn2 twice: the matching with the pixel grid as its value writes the flow, the propagation reads it), A, the
    // local correlation (k_localCorr: one warp a pixel, 4 a block) and B) and metricnet on the same halves
    const int64_t cpuF0 = profPair ? nowQpc100() : 0;
    ctx = nr.ctxGm[7];
    ctx->setTensorAddress("map4", nr.dGmMap[0]);
    ctx->setTensorAddress("map8", nr.dGmMap[1]);
    ctx->setTensorAddress("img0", g0);
    ctx->setTensorAddress("img1", g1);
    ctx->setTensorAddress("mf", nr.dGmM[0]);
    ctx->setTensorAddress("mq", nr.dGmM[1]);
    ctx->setTensorAddress("mk", nr.dGmM[2]);
    if (!ctx->enqueueV3(st))
    {
        nr.die("gmfss gmflow_bidir_a_m enqueueV3 returned false");
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
        const CUfunction fa = nr.gmAttnB ? nr.fAttn2b : nr.fAttn2;
        if (cuLaunchKernel(fa, gx, 2, 1, 128, 1, 1, 0, (CUstream)st, am, nullptr) != CUDA_SUCCESS ||
            cuLaunchKernel(fa, gx, 2, 1, 128, 1, 1, 0, (CUstream)st, ap, nullptr) != CUDA_SUCCESS)
        {
            nr.die("gmfss global matching launch failed");
            return false;
        }
    }
    ctx = nr.ctxGm[1];
    ctx->setTensorAddress("map4", nr.dGmMap[0]);
    ctx->setTensorAddress("map8", nr.dGmMap[1]);
    ctx->setTensorAddress("img0", g0);
    ctx->setTensorAddress("img1", g1);
    ctx->setTensorAddress("prop", nr.dGmProp);
    for (int i = 0; i < 8; i++)
        ctx->setTensorAddress(kGmA[i], nr.dGmA[i]);
    if (!ctx->enqueueV3(st))
    {
        nr.die("gmfss gmflow_bidir_a enqueueV3 returned false");
        return false;
    }
    int nb = 2, hw4 = nr.gmH * nr.gmW;
    void* ac[] = {(void*)&nr.dGmA[0],
                  (void*)&nr.dGmA[1],
                  (void*)&nr.dGmA[2],
                  (void*)&nr.dGmCorr,
                  &nb,
                  &nr.gmH,
                  &nr.gmW,
                  &hw4,
                  &nr.gmK};
    if (cuLaunchKernel(nr.fLocalCorr, (unsigned)((2LL * hw4 * 32 + 127) / 128), 1, 1, 128, 1, 1, 0, (CUstream)st, ac,
                       nullptr) != CUDA_SUCCESS)
    {
        nr.die("gmfss local correlation launch failed");
        return false;
    }
    ctx = nr.ctxGm[5];
    ctx->setTensorAddress("corr", nr.dGmCorr);
    for (int i = 3; i < 8; i++)
        ctx->setTensorAddress(kGmA[i], nr.dGmA[i]);
    ctx->setTensorAddress("img0", g0);
    ctx->setTensorAddress("img1", g1);
    ctx->setTensorAddress("flow", nr.gfh ? nr.dGmFlowF : nr.dGmFlow);
    if (!ctx->enqueueV3(st))
    {
        nr.die("gmfss gmflow_bidir_b enqueueV3 returned false");
        return false;
    }
    if (nr.gfh)
    {
        int planes = 4;
        void* au[] = {(void*)&nr.dGmFlowF, &planes, &nr.gfw, &nr.gfh, (void*)&nr.dGmFlow, &nr.hw, &nr.hh};
        if (cuLaunchKernel(nr.fGmFlowUp, (nr.hw + 15) / 16, (nr.hh + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, au,
                           nullptr) != CUDA_SUCCESS)
        {
            nr.die("gmfss flow grid upsample launch failed");
            return false;
        }
    }
    nr.gmMapsNext = true; // slot 1 = the new frame's maps = the next pair's previous frame
    if (profPair)
        nr.gmCpuFlow += (double)(nowQpc100() - cpuF0) / 10000.0;
    if (profPair)
        cudaEventRecord(nr.gmEv[3], st);
    float* f10 = nr.dGmFlow + 2 * hp;
    void* m1 = (uint8_t*)nr.dGmMetric + hp * (nr.gmMetricHalf ? 2 : 4);
    ctx = nr.ctxGm[2];
    ctx->setTensorAddress("i0", mv0);
    ctx->setTensorAddress("i1", mv1);
    ctx->setTensorAddress("f01", nr.dGmFlow);
    ctx->setTensorAddress("f10", f10);
    ctx->setTensorAddress("m0", nr.dGmMetric);
    ctx->setTensorAddress("m1", m1);
    if (!ctx->enqueueV3(st))
    {
        nr.die("gmfss metricnet enqueueV3 returned false");
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
// python's inference(): I1t, I2t, the ifnet, the six feature splats, fusionnet, clamp.
static bool nativeGmfssTween(NativeRife& nr, float t)
{
    cudaStream_t st = nr.stream;
    const size_t plane = (size_t)nr.ph * nr.pw, hp = (size_t)nr.hh * nr.hw, qp = hp / 4, ep = hp / 16;
    const size_t me = nr.gmMetricHalf ? 2 : 4;
    // the engines' metric keeps their dtype, a pyramid level is always fp32 (k_pyr's output)
    const int mHalf = nr.gmMetricHalf ? 1 : 0, fHalf = nr.gmFeatHalf ? 1 : 0;
    float sA = t, sB = 1.0f - t;
    // one softsplat: zero the accumulator, scatter every source pixel, normalize into the
    // fusionnet plane (a channel concat = adjacent planes, so no copy follows)
    auto splat = [&](void* in, int inHalf, int C, float* flow, void* metric, int metricHalf, float s, int w, int h,
                     void* dst, int dstHalf) -> bool {
        int cc = C, ww = w, hh2 = h, ih = inHalf, mh = metricHalf, dh = dstHalf;
        const int n = w * h;
        int ni = n;
        if (cudaMemsetAsync(nr.dGmAcc, 0, (size_t)(C + 1) * (size_t)n * sizeof(long long), st) != cudaSuccess)
        {
            nr.die("gmfss accumulator clear failed");
            return false;
        }
        void* a[] = {&in, &ih, &cc, (void*)&flow, &metric, &mh, &s, &ww, &hh2, (void*)&nr.dGmAcc};
        // 32 x 8: one warp = 32 consecutive x of a row, which k_splatSoft's neighbour combine
        // pairs (any shape of <= 256 threads, a multiple of 32, gives the same accumulators)
        if (cuLaunchKernel(nr.fSplatSoft, (w + 31) / 32, (h + 7) / 8, 1, 32, 8, 1, 0, (CUstream)st, a, nullptr) !=
            CUDA_SUCCESS)
        {
            nr.die("gmfss splatSoft launch failed");
            return false;
        }
        void* b[] = {(void*)&nr.dGmAcc, &cc, &ni, (void*)&dst, &dh};
        if (cuLaunchKernel(nr.fSplatNorm, (ni + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, b, nullptr) !=
            CUDA_SUCCESS)
        {
            nr.die("gmfss splatNorm launch failed");
            return false;
        }
        return true;
    };
    void** f0 = nr.dGmFeat[nr.gmCur ^ 1]; // the PREV frame's features (python's feat1x)
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
    if (!splat(nr.dGmHalf, 0, 3, flow0, met0, mHalf, sA, nr.hw, nr.hh, nr.dGmFa, 0))
        return false;
    if (!splat(nr.dGmHalf + 3 * hp, 0, 3, flow1, met1, mHalf, sB, nr.hw, nr.hh, nr.dGmFa + 6 * hp, 0))
        return false;
    if (prof)
        cudaEventRecord(nr.gmEv[7], st);
    // the GMFSS IFNet, straight into a's middle three planes (python's cat([I1t, rife, I2t]))
    unsigned int bits;
    memcpy(&bits, &t, 4);
    if (cuMemsetD32Async((CUdeviceptr)nr.dGmT, bits, 1, (CUstream)st) != CUDA_SUCCESS)
    {
        nr.die("gmfss timestep fill failed");
        return false;
    }
    nvinfer1::IExecutionContext* ctx = nr.ctxGm[3];
    ctx->setTensorAddress("x", nr.dGmHalf);
    ctx->setTensorAddress("timestep", nr.dGmT);
    ctx->setTensorAddress("merged", nr.dGmFa + 3 * hp);
    if (!ctx->enqueueV3(st))
    {
        nr.die("gmfss ifnet enqueueV3 returned false");
        return false;
    }
    if (prof)
        cudaEventRecord(nr.gmEv[8], st);
    // the three feature levels, both directions: the half from the engines' own flow and
    // metric, the quarter and the eighth from the pair's pyramids
    // b / c / d in the engine's dtype: the second direction starts gmC planes in
    const int bHalf = nr.gmFuseHalf ? 1 : 0;
    const size_t be = nr.gmFuseHalf ? 2 : 4;
    uint8_t* fb = (uint8_t*)nr.dGmFb;
    uint8_t* fc = (uint8_t*)nr.dGmFc;
    uint8_t* fd = (uint8_t*)nr.dGmFd;
    if (!splat(f0[0], fHalf, nr.gmC[0], flow0, met0, mHalf, sA, nr.hw, nr.hh, fb, bHalf))
        return false;
    if (!splat(f1[0], fHalf, nr.gmC[0], flow1, met1, mHalf, sB, nr.hw, nr.hh, fb + (size_t)nr.gmC[0] * hp * be, bHalf))
        return false;
    if (!splat(f0[1], fHalf, nr.gmC[1], nr.dGmFlowP[0], nr.dGmMetP[0], 0, sA, nr.hw / 2, nr.hh / 2, fc, bHalf))
        return false;
    if (!splat(f1[1], fHalf, nr.gmC[1], nr.dGmFlowP[0] + 2 * qp, nr.dGmMetP[0] + qp, 0, sB, nr.hw / 2, nr.hh / 2,
               fc + (size_t)nr.gmC[1] * qp * be, bHalf))
        return false;
    if (!splat(f0[2], fHalf, nr.gmC[2], nr.dGmFlowP[1], nr.dGmMetP[1], 0, sA, nr.hw / 4, nr.hh / 4, fd, bHalf))
        return false;
    if (!splat(f1[2], fHalf, nr.gmC[2], nr.dGmFlowP[1] + 2 * ep, nr.dGmMetP[1] + ep, 0, sB, nr.hw / 4, nr.hh / 4,
               fd + (size_t)nr.gmC[2] * ep * be, bHalf))
        return false;
    if (prof)
        cudaEventRecord(nr.gmEv[9], st);
    ctx = nr.ctxGm[4];
    ctx->setTensorAddress("a", nr.dGmFa);
    ctx->setTensorAddress("b", nr.dGmFb);
    ctx->setTensorAddress("c", nr.dGmFc);
    ctx->setTensorAddress("d", nr.dGmFd);
    ctx->setTensorAddress("out", nr.dGmOut);
    if (!ctx->enqueueV3(st))
    {
        nr.die("gmfss fusionnet enqueueV3 returned false");
        return false;
    }
    // python's torch.clamp(out, 0, 1) on the fp16 engine output: k_restToF is exactly that; an SDR pair's tween
    // (the SDR view) goes back to the HDR codes in the same step (k_hdrFromSdr)
    int n = (int)(3 * plane), half = nr.gmOutHalf ? 1 : 0, ps = (int)plane;
    void* ac[] = {(void*)&nr.dGmOut, &half, &n, (void*)&nr.dGmF};
    if (nr.gmPairSdr ? !nativeHdrFromSdr(nr, nr.dGmOut, half, ps, nr.pw, nr.pw, nr.ph, nr.dGmF, ps, nr.pw, nr.encPost)
                     : cuLaunchKernel(nr.fRestToF, (n + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, ac, nullptr) !=
                           CUDA_SUCCESS)
    {
        nr.die("gmfss clamp launch failed");
        return false;
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

// one group. Mirrors live_server.py's process_shm step for step: capture wait and read, the
// capture-release announcement, the pair encode, the tween chunks through the dynamic-batch
// engine, one pack-out per slot with its own event, then the bare end marker.
// ---- native DRBA ----------------------------------------------------------------------------
// live_server.RifeDrba as native launches on the group's stream. Frame ids count from 1 per
// session (drFid = the newest, python's k); id i sits in ring slot i & 3, four kept like
// python's hist. Every frame is encoded once when pushed (encode is a pure function of the
// frame; python encodes inside calc_flow). A window centred on c holds flow10 (2 planes) and
// r = calc_flow(I1 = c, I2 = c + 1)'s flow12 | flow21 (4 planes); two slots, the one holding
// c - 1 is kept when c is built (python deletes every window below c - 1), and a kept left
// neighbour chains: this window's flow10 = its flow21 (python: left['flow21']), else block0 on
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
    if (cudaMemcpyAsync(dst, dCur, 3 * plane * sizeof(float), cudaMemcpyDeviceToDevice, st) != cudaSuccess ||
        (nr.mph && same && id > 1 &&
         cudaMemcpyAsync(drbaMotion(nr, id), drbaMotion(nr, id - 1), 3 * mplane * sizeof(uint16_t),
                         cudaMemcpyDeviceToDevice, st) != cudaSuccess))
    {
        nr.die("drba frame ring copy failed");
        return false;
    }
    nvinfer1::Dims4 din{1, 3, drbaMh(nr), drbaMw(nr)};
    if (!nr.ctxEnc->setInputShape("img", din))
    {
        nr.die("encode setInputShape rejected (shape outside the engine profile)");
        return false;
    }
    // an fp16 encode input reads the motion frame, or an fp16 copy of the new frame (the fp16 x
    // copy's cur half, unused otherwise on this route)
    uint16_t* img16 = nr.mph ? drbaMotion(nr, id) : nr.dXh + 3 * plane;
    if (!nr.mph && nr.imgHalf && !nativeF2h(nr, dCur, img16, 3 * plane, st))
    {
        nr.die("f2h launch failed");
        return false;
    }
    nr.ctxEnc->setTensorAddress("img", nr.imgHalf ? (void*)img16 : (void*)dst);
    // fp16 features: the encode writes the ring slot directly (read as fp16 by block0 / IFNet)
    const bool widen = nr.encHalf && !nr.featHalf;
    nr.ctxEnc->setTensorAddress("feat", widen ? (void*)nr.dEncHalf : (void*)enc);
    if (!nr.ctxEnc->enqueueV3(st))
    {
        nr.die("encode enqueueV3 returned false");
        return false;
    }
    if (widen)
    {
        int n = (int)(16 * mplane);
        void* a[] = {&nr.dEncHalf, &enc, &n};
        if (cuLaunchKernel(nr.fH2f, (n + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, a, nullptr) != CUDA_SUCCESS)
        {
            nr.die("h2f launch failed");
            return false;
        }
    }
    return true;
}

// calc_flow(frame a, frame b): block0 on the pair, then its tail (the DRBA glue kernels) into out
// (4 planes: flow05 * 2 | flow15 * 2). Two domains: block0 takes fp32 frames, so the two motion
// frames are widened into dDrB0 first.
static bool nativeDrbaFlow(NativeRife& nr, uint32_t a, uint32_t b, float* out)
{
    cudaStream_t st = nr.stream;
    int mw = drbaMw(nr), mh = drbaMh(nr);
    const size_t plane = (size_t)mh * mw;
    nvinfer1::Dims4 di{1, 3, mh, mw}, df{1, 16, mh, mw};
    if (!nr.ctxB0->setInputShape("img0", di) || !nr.ctxB0->setInputShape("img1", di) ||
        !nr.ctxB0->setInputShape("f0", df) || !nr.ctxB0->setInputShape("f1", df))
    {
        nr.die("block0 setInputShape rejected");
        return false;
    }
    float *img0 = drbaFrame(nr, a), *img1 = drbaFrame(nr, b);
    if (nr.mph)
    {
        img0 = nr.dDrB0;
        img1 = nr.dDrB0 + 3 * plane;
        int n = (int)(3 * plane);
        const uint16_t *m0 = drbaMotion(nr, a), *m1 = drbaMotion(nr, b);
        void* a0[] = {(void*)&m0, &img0, &n};
        void* a1[] = {(void*)&m1, &img1, &n};
        if (cuLaunchKernel(nr.fH2f, (n + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, a0, nullptr) != CUDA_SUCCESS ||
            cuLaunchKernel(nr.fH2f, (n + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, a1, nullptr) != CUDA_SUCCESS)
        {
            nr.die("block0 frame h2f launch failed");
            return false;
        }
    }
    nr.ctxB0->setTensorAddress("img0", img0);
    nr.ctxB0->setTensorAddress("img1", img1);
    nr.ctxB0->setTensorAddress("f0", drbaEnc(nr, a));
    nr.ctxB0->setTensorAddress("f1", drbaEnc(nr, b));
    nr.ctxB0->setTensorAddress("flow", nr.dDrFlow);
    if (!nr.ctxB0->enqueueV3(st))
    {
        nr.die("block0 enqueueV3 returned false");
        return false;
    }
    if (cudaMemsetAsync(nr.dDrAcc, 0, 6 * plane * sizeof(long long), st) != cudaSuccess)
    {
        nr.die("drba accumulator clear failed");
        return false;
    }
    void* as[] = {&nr.dDrFlow, &mw, &mh, &nr.dDrAcc};
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

// RifeDrba._window(c): cached, or built (chained on a kept c - 1 window); nullptr = died
static NativeRife::DrWin* nativeDrbaWindow(NativeRife& nr, uint32_t c)
{
    for (auto& wn : nr.drWin)
        if (wn.c == c)
            return &wn;
    const size_t plane = (size_t)drbaMh(nr) * drbaMw(nr);
    // keep the left neighbour (the chain), else a right one (python keeps every window >= c - 1;
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
        if (!nativeDrbaFlow(nr, c, c - 1, nr.dDrFlowN))
            return nullptr;
        if (cudaMemcpyAsync(wn.f10, nr.dDrFlowN, 2 * plane * sizeof(float), cudaMemcpyDeviceToDevice, nr.stream) !=
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

// one tween of the group at fraction f into dMerged; held = python's None (the slot shows the
// lagged real frame). nHist = frames in the history (python's len(hist), at most 4). plain =
// plain pair RIFE on (k-2, k-1) at t = f whatever f is (the offline tail window:
// render_loops.drba_loop's last window has no right frame). slot = blendOut's output slot: the tween
// goes straight into it instead of dMerged.
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
            // two domains: the pair's motion frames for the IFNet, its pictures for the blend
            if (cudaMemcpyAsync(nr.dDrMX[xi], drbaMotion(nr, i1), 3 * mplane * sizeof(uint16_t),
                                cudaMemcpyDeviceToDevice, st) != cudaSuccess ||
                cudaMemcpyAsync(nr.dDrMX[xi] + 3 * mplane, drbaMotion(nr, i0), 3 * mplane * sizeof(uint16_t),
                                cudaMemcpyDeviceToDevice, st) != cudaSuccess ||
                cudaMemcpyAsync(nr.dDrX[xi], drbaFrame(nr, i1), 3 * plane * sizeof(float), cudaMemcpyDeviceToDevice,
                                st) != cudaSuccess ||
                cudaMemcpyAsync(nr.dDrX[xi] + 3 * plane, drbaFrame(nr, i0), 3 * plane * sizeof(float),
                                cudaMemcpyDeviceToDevice, st) != cudaSuccess)
            {
                nr.die("drba x copy failed");
                return false;
            }
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
        // the head: plain pair RIFE at t = f, a constant timestep map (python: base + float(t))
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
    nvinfer1::Dims4 dx{1, 6, mh, mw}, dts{1, 1, mh, mw}, df{1, 16, mh, mw};
    if (!nr.ctxIf->setInputShape("x", dx) || !nr.ctxIf->setInputShape("timestep", dts) ||
        !nr.ctxIf->setInputShape("f0", df) || !nr.ctxIf->setInputShape("f1", df))
    {
        nr.die("IFNet setInputShape rejected (shape outside the engine profile)");
        return false;
    }
    nr.ctxIf->setTensorAddress("x", nr.mph ? (void*)nr.dDrMX[xi] : (void*)nr.dDrX[xi]);
    nr.ctxIf->setTensorAddress("timestep", nr.dT);
    nr.ctxIf->setTensorAddress("f0", drbaEnc(nr, i1));
    nr.ctxIf->setTensorAddress("f1", drbaEnc(nr, i0));
    nr.ctxIf->setTensorAddress("flow", nr.dFlow);
    nr.ctxIf->setTensorAddress("mask", nr.dMask);
    if (!nr.ctxIf->enqueueV3(st))
    {
        nr.die("IFNet enqueueV3 returned false (outputs would be garbage)");
        return false;
    }
    // the tween: RIFE's last step on the frames this enqueue read, in two domains on their pictures
    if (slot ? !nativeRifeBlendOut(nr, nr.dDrX[xi], nr.mph ? 0 : (int)nr.xHalf, 0, slot, st)
             : !nativeRifeBlend(nr, nr.dDrX[xi], nr.mph ? 0 : (int)nr.xHalf, 1, st))
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

// Restore (live and offline): _Fit._restore / render_passes.restore on one
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
// of the session instead (python's rule), the caller then continues with the unrestored source.
// HDR planes (enc 1 PQ, 2 HLG): the engine reads the frame's SDR view (hdrToModel, knee 1, head 0, the sign
// clipped), the fold is the SDR one, and k_hdrRestOut adds back the light the view never held.
static bool nativeRestoreRun(NativeRife& nr, const void* s, int ps_, int rs_, float* dst, int tw, int th, int sHalf = 0,
                             int enc = 0, bool viewRef = false)
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
    nvinfer1::Dims4 din{1, 3, sh, sw};
    nr.ctxRest->setTensorAddress("x", nr.dRestIn);
    nr.ctxRest->setTensorAddress("y", nr.dRestOut);
    if (!nr.ctxRest->setInputShape("x", din) || !nr.ctxRest->enqueueV3(st))
    {
        LOG("native: restore failed, dropping the pass for the rest of the %s\n", g_offline ? "render" : "session");
        nr.restFailed = true;
        return true;
    }
    int half = nr.restHalfOut ? 1 : 0, w4 = 4 * sw, h4 = 4 * sh, ps4 = w4 * h4;
    if (th <= h4)
    {
        // realesr.fit: the antialiased pair when the target height shrinks (an exact copy
        // at 4x itself), with `out.clamp(0,1)` folded into the taps. A shrinking height runs both
        // passes in k_restFoldTile (the same bytes, no dRestTmp round trip) at blocks of 32 columns by
        // the tallest of 32 / 16 / 8 rows that fit 48 KB of shared memory (fitPack's rule); none = the pair
        if (th < h4)
        {
            const int fsh = lzTaps(w4, tw), fsv = lzTaps(h4, th);
            for (int by = 32; by >= 8; by /= 2)
            {
                int rcap = (int)(((long long)(by - 1) * h4 + th - 1) / th) + fsv + 2;
                const size_t smem = ((size_t)fsh * 32 + (size_t)fsv * by) * 8 + 8 + (size_t)12 * rcap * 32;
                if (smem > 48 * 1024)
                    continue;
                void* at[] = {&nr.dRestOut, &half, &ps4, &w4, &w4, &h4, &dst, &tw, &th, &rcap};
                if (cuLaunchKernel(nr.fRestFoldTile, (tw + 31) / 32, (th + by - 1) / by, 1, 32, by, 1, (unsigned)smem,
                                   (CUstream)st, at, nullptr) != CUDA_SUCCESS)
                {
                    nr.die("restFoldTile launch failed");
                    return false;
                }
                return !enc || nativeHdrRestOut(nr, dst, tw, th, s, sHalf, ps_, rs_, sw, sh, enc, viewRef);
            }
        }
        void* ah[] = {&nr.dRestOut, &half, &ps4, &w4, &w4, &h4, &nr.dRestTmp, &tw};
        if (cuLaunchKernel(nr.fRestFoldH, (tw + 15) / 16, (h4 + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, ah, nullptr) !=
            CUDA_SUCCESS)
        {
            nr.die("restFoldH launch failed");
            return false;
        }
        void* av[] = {&nr.dRestTmp, &tw, &h4, &dst, &th};
        if (cuLaunchKernel(nr.fRestFoldV, (tw + 15) / 16, (th + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, av, nullptr) !=
            CUDA_SUCCESS)
        {
            nr.die("restFoldV launch failed");
            return false;
        }
        return !enc || nativeHdrRestOut(nr, dst, tw, th, s, sHalf, ps_, rs_, sw, sh, enc, viewRef);
    }
    // an enlarging target (above 4x): sampleOut's Lanczos from the clamped fp32 copy
    int n = 3 * ps4, f32 = 0;
    void* a0[] = {&nr.dRestOut, &half, &n, &nr.dRestF};
    if (cuLaunchKernel(nr.fRestToF, (n + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, a0, nullptr) != CUDA_SUCCESS)
    {
        nr.die("restToF launch failed");
        return false;
    }
    if (!nativeFitPlanar(nr, nr.dRestF, f32, ps4, w4, w4, h4, dst, tw, th, st))
    {
        nr.die("fitPlanar (restore) launch failed");
        return false;
    }
    int n2 = 3 * tw * th;
    void* a2[] = {&dst, &n2};
    if (cuLaunchKernel(nr.fClamp01, (n2 + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, a2, nullptr) != CUDA_SUCCESS)
    {
        nr.die("clamp01 launch failed");
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

// The zero-copy imports of the offline DLSS 5 handoff, released in the order CUDA asks for (the
// mapped buffers first). The caller has drained the stream.
static void nativeNrReleaseImports(NativeRife& nr)
{
    if (nr.dNrShIn)
    {
        cudaFree(nr.dNrShIn);
        nr.dNrShIn = nullptr;
    }
    if (nr.dNrShOut)
    {
        cudaFree(nr.dNrShOut);
        nr.dNrShOut = nullptr;
    }
    if (nr.dNrShMv)
    {
        cudaFree(nr.dNrShMv);
        nr.dNrShMv = nullptr;
    }
    if (nr.semNr)
    {
        cudaDestroyExternalSemaphore(nr.semNr);
        nr.semNr = nullptr;
    }
    if (nr.emNrIn)
    {
        cudaDestroyExternalMemory(nr.emNrIn);
        nr.emNrIn = nullptr;
    }
    if (nr.emNrOut)
    {
        cudaDestroyExternalMemory(nr.emNrOut);
        nr.emNrOut = nullptr;
    }
    if (nr.emNrMv)
    {
        cudaDestroyExternalMemory(nr.emNrMv);
        nr.emNrMv = nullptr;
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

// Zero-copy DLSS 5: import the NR core's shared buffers and fence (nr::Host::startShared)
// into this CUDA context. Refused when the core's D3D12 adapter is not the current CUDA device;
// the caller then keeps the CPU staging route.
static bool nativeNrImport(NativeRife& nr, nr::Host& host, std::string& err)
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
    cudaExternalMemoryHandleDesc md{};
    md.type = cudaExternalMemoryHandleTypeD3D12Resource;
    md.size = host.sharedBytes();
    md.flags = cudaExternalMemoryDedicated;
    md.handle.win32.handle = host.sharedInHandle();
    if (cudaImportExternalMemory(&nr.emNrIn, &md) != cudaSuccess)
    {
        err = "input buffer import failed";
        return false;
    }
    md.handle.win32.handle = host.sharedOutHandle();
    if (cudaImportExternalMemory(&nr.emNrOut, &md) != cudaSuccess)
    {
        err = "output buffer import failed";
        return false;
    }
    cudaExternalMemoryBufferDesc bd{};
    bd.offset = 0;
    bd.size = host.sharedBytes();
    bd.flags = 0;
    if (cudaExternalMemoryGetMappedBuffer((void**)&nr.dNrShIn, nr.emNrIn, &bd) != cudaSuccess ||
        cudaExternalMemoryGetMappedBuffer((void**)&nr.dNrShOut, nr.emNrOut, &bd) != cudaSuccess)
    {
        err = "shared buffer mapping failed";
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
    nr.nrPitch = (size_t)host.rowPitch();
    if (host.sharedMvHandle())
    {
        md.size = host.sharedMvBytes();
        md.handle.win32.handle = host.sharedMvHandle();
        bd.size = host.sharedMvBytes();
        if (cudaImportExternalMemory(&nr.emNrMv, &md) != cudaSuccess ||
            cudaExternalMemoryGetMappedBuffer((void**)&nr.dNrShMv, nr.emNrMv, &bd) != cudaSuccess)
        {
            err = "motion buffer import failed";
            return false;
        }
        nr.nrMvPitch = (size_t)host.mvRowPitch();
    }
    return true;
}

// DLSS 5 motion, set up once the shared buffers are imported: an Optical Flow session at the NR
// size (forward only: the current frame is the input, the previous one the reference), its two
// luma slots, the field and cost buffers, and the upsampled field. false = no motion for this
// render (the caller says why and frees what was made).
static bool nativeNrMotionSetup(NativeRife& nr, std::string& err)
{
    const int w = nr.nrW, h = nr.nrH;
    if (!nr.dNrShMv)
    {
        err = "the core shares no motion buffer";
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
    ip.enableOutputCost = NV_OF_TRUE;
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
                         nr.nrOfOut, nr.nrOfOutP, nr.nrOfOutPitch, "dlss5 flow") ||
        !nativeNvofBufOn(nr.nrOfH, nr.nrOfGw, nr.nrOfGh, NV_OF_BUFFER_USAGE_COST, NV_OF_BUFFER_FORMAT_UINT8,
                         nr.nrOfCost, nr.nrOfCostP, nr.nrOfCostPitch, "dlss5 cost"))
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
    if (cudaMalloc((void**)&nr.dNrFlow, 3 * (size_t)w * h * sizeof(float)) != cudaSuccess)
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
    for (NvOFGPUBufferHandle* b : {&nr.nrOfIn[0], &nr.nrOfIn[1], &nr.nrOfOut, &nr.nrOfCost})
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
// (k_nvofUp) and validated (k_nrMv) into the core's shared motion buffer; the stream's first frame
// (a Reset) gets a zero field. The slots alternate, so the previous frame's luma stays in place.
static const float kNrMvMargin = 25.0f; // one 8-bit level per pixel of the 5x5 window (k_nrMv)
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
        if (cudaMemset2DAsync(nr.dNrShMv, nr.nrMvPitch, 0, (size_t)w * 4, h, st) != cudaSuccess)
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
    eo.outputCostBuffer = nr.nrOfCost;
    const NV_OF_STATUS s = g_nvofApi.nvOFExecute(nr.nrOfH, &ei, &eo);
    if (s != NV_OF_SUCCESS)
    {
        LOG("native: DLSS 5 motion: nvOFExecute failed (status %d)\n", (int)s);
        nr.die("DLSS 5 motion: execute failed");
        return false;
    }
    int grid = kNvofGrid, vp = (int)nr.nrOfOutPitch, cp = (int)nr.nrOfCostPitch, mp = (int)nr.nrMvPitch;
    CUdeviceptr v = nr.nrOfOutP, c = nr.nrOfCostP;
    float* flow = nr.dNrFlow;
    float* cost = nr.dNrFlow + 2 * (size_t)w * h;
    float margin = kNrMvMargin;
    void* b[] = {&v, &vp, &c, &cp, &nr.nrOfGw, &nr.nrOfGh, &grid, &w, &h, &flow, &cost};
    void* m[] = {&flow, &lc, &lp, &lpitch, &w, &h, &margin, &nr.dNrShMv, &mp};
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
// be told or the CUDA runtime is not loaded (g_vramNoCuda: offline Auto's fit query). SMV_VRAM_MIB shrinks the card
// and the free memory with it. false = the driver does not say.
static bool g_vramNoCuda = false;
static bool nativeVideoMemory(uint64_t& freeB, uint64_t& cardB, uint64_t* moved = nullptr)
{
    static const NvPhysicalGpuHandle gpu = [] {
        NvPhysicalGpuHandle all[NVAPI_MAX_PHYSICAL_GPUS] = {};
        NvU32 n = 0;
        if (NvAPI_Initialize() != NVAPI_OK || NvAPI_EnumPhysicalGPUs(all, &n) != NVAPI_OK || !n)
            return (NvPhysicalGpuHandle) nullptr;
        LUID want{};
        if (n > 1 && !g_vramNoCuda && nativeCudaLuid(want))
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
// at 854x480 with 3 / 10: its per-size buffers exist before the ring on this route, so nr_host's kAfterMp is not
// added), and FRUC's NvOFFRUC buffers with its second midpoint instance, 233 MiB + 138 a padded megapixel of the model
// frame (296 / 521 MiB at 896x512 / 1920x1088), and AMD FSR 4 frame generation's second midpoint instance with its node
// frames, 45 MiB + 335 a megapixel of the (unpadded) model frame (a parked live session's own memory, x4 against x2: 146
// / 476 MiB at 854x480 / 1708x960, a quarter on top) + its side Optical Flow session (nativeFsrfgAhead, x4 and up: 34 /
// 74 MiB at 854x480 / 1280x720 = 2 + 98 a megapixel with a quarter, where both sides hold x4); the other routes add
// nothing the reserve does not cover. The x8 /
// x16 ladders' third and fourth tree levels (an instance and twice the node frames a level) come on top when the
// session's slots reach them (the process's own memory over x4 at 854x480 / 1708x960, a quarter on top, FRUC by the
// padded megapixel: FRUC x8 +332 / +620 MiB, x16 +711 / +1423; FSR 4 x8 +282 / +462, x16 +627 / +1115, but FSR 4 held
// only ~x3.7 at 1708x960 and leaves those levels' node frames partly unmade: the per-megapixel term is FSR 3.1's larger
// fit).
constexpr double kNrLiveLateBase = 150.0, kFrucLateBase = 290.0, kFrucLateMp = 173.0;
constexpr double kFsrfgLateBase = 47.0, kFsrfgLateMp = 433.0;
constexpr double kFrucDeepBase[2] = {277.0, 549.0}, kFrucDeepMp[2] = {300.0, 742.0};
constexpr double kFsrfgDeepBase[2] = {278.0, 580.0}, kFsrfgDeepMp[2] = {333.0, 820.0};
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
        need += nr.fsrfg ? kFsrfgLateBase + kFsrfgLateMp * mp : kFrucLateBase + kFrucLateMp * mp;
        // the deepest ladder the slots and the instances reach: x8 = 3 levels, x16 = 4
        const int levels = (std::min)(nr.frInstMax, slots >= 16 ? 4 : (slots >= 8 ? 3 : 2));
        if (nr.frMp && levels > 2)
            need += nr.fsrfg ? kFsrfgDeepBase[levels - 3] + kFsrfgDeepMp[levels - 3] * mp
                             : kFrucDeepBase[levels - 3] + kFrucDeepMp[levels - 3] * mp;
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
    const uint64_t late = nativeLiveLateNeed(nr, slots);
    ringB = room > late ? room - late : 0;
    const uint64_t n = ringB / (2ull * slotBytes);
    return (uint32_t)(n < 2 ? 2 : (n > 4096 ? 4096 : n));
}

// A live session's video memory at the working size mw x mh before anything loads, in bytes. The model's share per
// backend (the CUDA context, the runtime, the kernels, the engines and every buffer: a base, a padded megapixel of the
// working size and a megapixel of the capture; NVAPI's free figure from the session's start to its steady state at
// 854x480 and 1920x1080, and at 960x540 of a 1920x1080 capture for the capture's share, a quarter on top; rife, blend
// and rifedrba measured with the tween stored in fp16, + 7 for its fp32 store: 6 bytes a pixel more; gmfss moved by a
// quarter over what its graph set changes against the whole gmflow graph in the process's own memory at 854x480 and
// 1708x960: gmflow's two halves -76 MiB - 772 a padded megapixel, its backbone once a frame +9 + 50, fusionnet's fp16
// feature inputs -1 - 102; fsrfg = fruc's row moved by a quarter over the process's own memory against fruc's, one
// instance, at 854x480 and 1708x960: 475 / 996 MiB against 608 / 928), DLSS 5's
// passes (nr_host's creation and first-frame shares plus this route's own buffers, kNrLiveMp: within 2 % of the
// measured 1526 / 828 MiB for 3 passes at 1920x1080 / 960x540), RTX HDR, RTX VSR, Restore at the capture, and the
// output ring's 2 slots (the ring gives way down to them, nativeRingFit).
struct LiveModelMem
{
    const char* backend;
    double base, padMp, capMp;
};
static const LiveModelMem kLiveModelMem[] = {
    {"rife", 345, 730, 103}, {"blend", 345, 730, 103}, {"rifedrba", 388, 1210, 103}, {"gmfss", 254, 2036, 103},
    {"nvof", 297, 116, 103}, {"fruc", 577, 122, 103},  {"fsrfg", 315, 331, 103},     {"echo", 282, 0, 48}};
constexpr double kNrLiveMp = 30.0;
static uint64_t nativeLiveNeed(const std::string& backend, int cw, int ch, int mw, int mh)
{
    const double mib = 1048576.0, mp = (double)mw * mh / 1e6, capMp = (double)cw * ch / 1e6;
    const double padMp = (double)((mw + 63) / 64 * 64) * ((mh + 63) / 64 * 64) / 1e6;
    const LiveModelMem* m = &kLiveModelMem[0];
    for (const LiveModelMem& e : kLiveModelMem)
        if (backend == e.backend)
            m = &e;
    double need = m->base + m->padMp * padMp + m->capMp * capMp;
    if (g_liveNrCuda)
    {
        const int p = g_nrPasses < 1 ? 1 : (g_nrPasses > nr::kMaxPasses ? nr::kMaxPasses : g_nrPasses);
        need += p * (nr::kPassBase + nr::kPassMp * mp) + nr::kAfterBase + nr::kAfterMp * mp + nr::kFrameMp * mp * p +
                kNrLiveMp * mp;
    }
    if (g_rtxHdr && g_hdr) // RTX HDR runs on an HDR desktop only, RTX VSR on SDR only (nativeConfigHdr)
        need += kRtxHdrBase + kRtxHdrMp * (std::max)(mp, capMp);
    if (g_rtxVsr && !g_hdr)
        need += kRtxVsrBase + kRtxVsrMp * mp;
    else if (g_fsrUp && !g_hdr)
        need += kFsrUpBase + kFsrUpMp * mp;
    if (g_restore)
        need += kRestoreMp * capMp;
    const double slot = (double)(((size_t)((W * 4 + 255) & ~255u) * H + 511) & ~(size_t)511);
    return (uint64_t)(need * mib + 4.0 * slot);
}

// Live Auto's working size by the free video memory (DLSS 5's passes are never cut; Auto may lower its working size):
// Auto's own pick while the session fits, else the largest lower mode that fits, Ultra Performance when none does
// (the session then runs anyway and says `video memory ran out` if the driver moves memory out). Picked ONCE a
// session, before anything loads: every lkSession of the session reads it through liveWorkSize, so the lookups, the
// builds and the resident handoff key agree. 0 = Auto's own pick (it fits, Auto is off, or there is no room figure:
// SMV_VRAM_CAP=0 or no driver figure).
static int liveAutoFitMemory(uint32_t capW, uint32_t capH, const std::wstring& backendW)
{
    if (g_dlssMode != 1)
        return 0;
    const uint64_t room = nativeVideoMemoryRoom();
    if (!room)
        return 0;
    const int cw = (int)capW, ch = (int)capH;
    int presW, presH, x0, y0;
    lkFitRect(cw, ch, W ? (int)W : cw, W ? (int)H : ch, presW, presH, x0, y0);
    const int pick = (std::max)(liveAutoMode(presW, presH), g_liveAutoFloor);
    if (pick < 2 || pick > 6)
        return 0;
    const std::string backend = wideToUtf8(backendW);
    int pickW = 0, pickH = 0;
    uint64_t pickNeed = 0;
    for (int m = pick; m <= 6; m++)
    {
        int mw, mh;
        liveWorkSizeFor(m, presW, presH, mw, mh);
        const uint64_t need = nativeLiveNeed(backend, cw, ch, mw, mh);
        if (m == pick)
        {
            pickW = mw;
            pickH = mh;
            pickNeed = need;
        }
        if (need > room && m < 6)
            continue;
        if (m == pick)
            return 0;
        const double mib = 1048576.0;
        LOG("native: video memory: Auto runs %ls (%dx%d) instead of %ls (%dx%d): %ls needs about %.0f MiB, %.0f MiB "
            "are free%s\n",
            kDlssModeName[m - 1], mw, mh, kDlssModeName[pick - 1], pickW, pickH, kDlssModeName[pick - 1],
            pickNeed / mib, room == 1 ? 0.0 : room / mib,
            need > room ? " (it does not fit at any mode: the session runs anyway)" : "");
        return m;
    }
    return 0;
}

// A session's video memory check, for every route: nativeVideoMemoryMark at its start (before the engines and
// buffers), nativeVideoMemoryNote once its first frames are out. One line when the driver moved allocations out of
// video memory meanwhile: the session did not fit in what was free, so part of it, or of another app, now lives in
// system memory and runs slower there (the pixels are the same).
static uint64_t g_vramMoved = 0, g_vramFree = 0;
static bool g_vramMarked = false;
static void nativeVideoMemoryMark()
{
    uint64_t cardB = 0;
    g_vramMarked = nativeVideoMemory(g_vramFree, cardB, &g_vramMoved);
}
static void nativeVideoMemoryNote()
{
    uint64_t freeB = 0, cardB = 0, moved = 0;
    if (!g_vramMarked || !nativeVideoMemory(freeB, cardB, &moved))
        return;
    g_vramMarked = false;
    const double gb = 1024.0 * 1024.0 * 1024.0;
    if (moved > g_vramMoved || (nativeVideoCardLever() && !freeB))
        LOG("%s: video memory ran out: %.1f GB of the GPU's %.1f GB were free when this %s started and it needs "
            "more, so the driver moved memory out to system memory, which slows it and other apps down (the pixels "
            "stay the same); a smaller picture, a lower DLSS mode, fewer DLSS 5 passes or closing other apps leave "
            "more room\n",
            g_offline ? "offline" : "native", (double)g_vramFree / gb, (double)cardB / gb,
            g_offline ? "render" : "session");
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
    void* a[] = {(void*)&src, &sps, &pw, &tw, &th, &nr.dNrIo};
    void* b[] = {&nr.dNrIo, &tw, &th, &r0, &ph, &pw, &sps};
    void* aq[] = {(void*)&src, &sps, &pw, &tw, &th, &nr.nrSdrWhite, &nr.dNrIo, &hlg};
    void* bq[] = {&nr.dNrIo, &tw, &th, &nr.nrSdrWhite, &r0, &ph, &pw, &sps, &hlg};
    const unsigned gx = (unsigned)(pw + 15) / 16, gy = (unsigned)(ph + 15) / 16;
    if (nr.nrZeroCopy)
    {
        const size_t row = (size_t)tw * 8;
        if (nr.nrMotion && !nativeNrMotion(nr, frame, pw, ps))
            return false;
        cudaExternalSemaphoreSignalParams sp{};
        sp.params.fence.value = ++nr.nrFenceV;
        if (cuLaunchKernel(pq ? nr.fNrInPq : nr.fNrIn, (tw + 15) / 16, (th + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st,
                           pq ? aq : a, nullptr) != CUDA_SUCCESS ||
            cudaMemcpy2DAsync(nr.dNrShIn, nr.nrPitch, nr.dNrIo, row, row, th, cudaMemcpyDeviceToDevice, st) !=
                cudaSuccess ||
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
            cudaMemcpy2DAsync(nr.dNrIo, row, nr.dNrShOut, nr.nrPitch, row, th, cudaMemcpyDeviceToDevice, st) !=
                cudaSuccess ||
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
    set.style = (g_nrStyle >= 0 && g_nrStyle <= 2) ? g_nrStyle : 1;
    set.passes = g_nrPasses < 1 ? 1 : (g_nrPasses > nr::kMaxPasses ? nr::kMaxPasses : g_nrPasses);
    // SMV_NR_MV=0 = no motion (a measurement lever); without motion every frame is a Reset, since kept
    // history with no motion slides; SMV_NR_AUTOMASK=0 = the mask off (read by the core)
    wchar_t mvv[8]{}, amv[8]{}, ru[8]{}, re[8]{};
    set.motion = !(GetEnvironmentVariableW(L"SMV_NR_MV", mvv, 8) && mvv[0] == L'0');
    const bool autoMask = !(GetEnvironmentVariableW(L"SMV_NR_AUTOMASK", amv, 8) && amv[0] == L'0');
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
    if (!host->startShared(err) || !nativeNrImport(nr, *host, err))
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
    const size_t plane = (size_t)nr.ph * nr.pw;
    NCHK(cudaMalloc((void**)&nr.dNrIo, (size_t)w * h * 8), "alloc dlss5 frame");
    NCHK(cudaMalloc((void**)&nr.dNrLast, 3 * plane * sizeof(float)), "alloc dlss5 last output");
    NCHK(cudaMalloc((void**)&nr.dCapPrev, (size_t)nr.cw * nr.ch * (nr.hdr ? 8 : 4)), "alloc dlss5 reuse reference");
    nr.nrResetEvery = !nr.nrMotion || (GetEnvironmentVariableW(L"SMV_NR_RESET_EVERY", re, 8) && re[0] == L'1');
    nr.nrReuse = !(GetEnvironmentVariableW(L"SMV_NR_REUSE", ru, 8) && ru[0] == L'0');
    nr.nrSdrWhite = (float)(g_sdrWhite / 80.0);
    nr.nrFirst = true;
    nr.liveNr = true;
    LOG("live DLSS 5 native: on, %dx%d per captured frame on the model frame after Restore and the resize, "
        "zero-copy, structure %.2f tone %.2f style %d, passes %d, %s, %s, %s%s\n",
        w, h, set.structure, set.tone, set.style, host->passes(),
        nr.nrMotion ? "motion vectors (NVOFA grid 4), history kept"
                    : (set.motion ? "no motion vectors, every frame a Reset"
                                  : "no motion vectors (SMV_NR_MV=0), every frame a Reset"),
        autoMask ? "auto mask" : "no auto mask (SMV_NR_AUTOMASK=0)",
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
    else if (sharpen)
    {
        // the sharpen reads a tight w x h frame: the model frame without its pad
        for (int c = 0; c < 3; c++)
            if (cudaMemcpy2DAsync(nr.dShIn + (size_t)c * tight, (size_t)w * sizeof(float), dCur + (size_t)c * dps,
                                  (size_t)pw * sizeof(float), (size_t)w * sizeof(float), h, cudaMemcpyDeviceToDevice,
                                  st) != cudaSuccess)
            {
                nr.die("sharpen input copy failed");
                return false;
            }
    }
    float* out = nr.dShOut; // the tight result (the sharpen, or TrueHDR's PQ), padded back into dCur below
    if (thdr)
    {
        if (sharpen)
        {
            void* a[] = {&nr.dShIn, &w, &h, &nr.sharpen, &nr.dThdrIn, &nr.dSrcG};
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
                void* a[] = {&nr.dShIn, &w, &h, &nr.sharpen, &out};
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
        void* a[] = {&nr.dThdrOut, &nr.dSrcG, &w, &h, &out, &nr.rtxMode, &nr.rtxVib, &nr.rtxSb};
        if (cuLaunchKernel(nr.fThdrColor, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a, nullptr) !=
            CUDA_SUCCESS)
        {
            nr.die("thdrColor launch failed");
            return false;
        }
    }
    else
    {
        void* a[] = {&nr.dShIn, &w, &h, &nr.sharpen, &out};
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
// export between a drain of st and a device-wide sync, because its copies ride the legacy default stream
static bool nativeVsrEval(NativeRife& nr, const RtxRect& ri, const RtxRect& ro, cudaStream_t st, unsigned int& rv)
{
    if (nr.fsrUp)
        return nativeFsrupEval(nr, ri, ro, st, rv); // AMD FSR upscaling holds RTX VSR's resize
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

static bool nativeOfflineStage(NativeRife& nr, const void*& src, int& ps, int& rs, int& srcHalf, bool& staged, int sw,
                               int sh, int tw, int th, bool withRestore, bool vsrHere, int enc = 0,
                               bool viewFit = false)
{
    cudaStream_t st = nr.stream;
    const bool resize = tw != sw || th != sh;
    const bool vsrNow = vsrHere && nr.vsr && !nr.vsrFailed && (g_rtxb.created || nr.fsrUp);
    staged = false;
    if (withRestore && nr.restore && nr.ctxRest && !nr.restFailed)
    {
        if (vsrNow)
        {
            if (!nativeRestoreRun(nr, src, ps, rs, nr.dRest, sw, sh, srcHalf, enc, viewFit))
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
        if (vsrNow)
        {
            void* a[] = {(void*)&src, &srcHalf, (void*)&ps, (void*)&rs, &sw, &sh, &nr.dVsrIn};
            if (cuLaunchKernel(nr.fPackBgraRgb, (sw + 15) / 16, (sh + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a,
                               nullptr) != CUDA_SUCCESS)
            {
                nr.die("packBgraRgb launch failed");
                return false;
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
        if (haveVsr)
        {
            void* a[] = {&nr.dVsrOut, (void*)&tw, (void*)&th, &nr.dPres};
            if (cuLaunchKernel(nr.fUnpackBgraRgb, (tw + 15) / 16, (th + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a,
                               nullptr) != CUDA_SUCCESS)
            {
                nr.die("unpackBgraRgb launch failed");
                return false;
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
                            in == 1))
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
             !nativeOfflineStage(nr, src, ps, rs, srcHalf, staged, nr.w, nr.h, tw, th, false, nr.vsrPost))
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
    const float* fr = (const float*)(msg.data() + 4);
    if (msg.size() < 4 + 4ull * nfr)
        return false;
    if (nfr > nr.slots)
        nfr = nr.slots;
    // no-engine mode (Identity.process_shm): an empty group stays empty (the pair advances,
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

    // the previous captured frame's own passes, once the GPU is past them (a frame that reused the last DLSS 5
    // output ran none and is not counted)
    if (nr.pfArmed && cudaEventQuery(nr.pfEv[2]) == cudaSuccess)
    {
        float all = 0.0f, work = 0.0f;
        if (cudaEventElapsedTime(&all, nr.pfEv[0], nr.pfEv[2]) == cudaSuccess &&
            cudaEventElapsedTime(&work, nr.pfEv[1], nr.pfEv[2]) == cudaSuccess)
        {
            nr.pfAllMs = nr.pfAllMs > 0.0 ? nr.pfAllMs * 0.8 + all * 0.2 : all;
            nr.pfWorkMs = nr.pfWorkMs > 0.0 ? nr.pfWorkMs * 0.8 + work * 0.2 : work;
            nr.pfAllUs.store((uint32_t)(nr.pfAllMs * 1000.0));
            nr.pfWorkUs.store((uint32_t)(nr.pfWorkMs * 1000.0));
        }
    }
    nr.pfArmed = false;
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
                ema(nr.gIntMs, pre);
                ema(nr.gTweenMs, slots / nr.grTween);
                nr.gIntUs.store((uint32_t)(nr.gIntMs * 1000.0));
                nr.gTweenUs.store((uint32_t)(nr.gTweenMs * 1000.0));
            }
            else
            {
                ema(nr.gBaseMs, pre + slots);
                nr.gBaseUs.store((uint32_t)(nr.gBaseMs * 1000.0));
            }
        }
    }
    nr.grArmed = false;

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
    // the 1:1 frame with nothing between the pack and the motion frame (no DLSS 5, no Sharpen, no RTX HDR, no Restore, no
    // RTX VSR before the model): k_packInMotion packs it, makes its motion frame and, when the identical-pair test below
    // runs on this pair, that test too; through nr.capTex it reads the capture array itself
    const bool packMotion = !(nr.rtxHdr && !nr.rtxFailed && g_rtxb.created) && !(nr.vsrPre && !nr.vsrFailed) &&
                            !(nr.restPre && nr.ctxRest && !nr.restFailed) && nr.w == nr.cw && nr.h == nr.ch &&
                            nr.mph == nr.ph && nr.mpw == nr.pw && !nr.dMFit && !(nr.liveNr && !nr.nrFailed) &&
                            !(nr.sharpen > 0.0f);
    const bool capDirect = packMotion && nr.capTex;
    // the real slots present the raw capture bit for bit: the passthrough keys on identity AND no effect, never on
    // geometry alone (with sharpen or Upscale to on, real frames go through the store like every other frame, with
    // Restore or DLSS 5 on the real frame is the one they wrote in dCur, never on DRBA: its real frame is the lagged
    // k - 1), and the model frame must BE the capture (a working size above it, Fill, equals the canvas too)
    const bool rawPass = nr.identity && nr.w == nr.cw && nr.h == nr.ch && !nr.drba && !nr.hdr && nr.sharpen <= 0.0f &&
                         !nr.uw && !nr.restore && !nr.liveNr;
    // (2) read the shared capture texture; the event marks the exe's texture as free again (capDirect: after the pack)
    const size_t capRow = (size_t)nr.cw * (nr.hdr ? 8 : 4);
    if (!capDirect)
    {
        if (cudaMemcpy2DFromArrayAsync(nr.dCap, capRow, nr.capArr, 0, 0, capRow, nr.ch, cudaMemcpyDeviceToDevice, st) !=
            cudaSuccess)
        {
            nr.die("capture memcpy2DFromArray failed");
            return false;
        }
        cudaEventRecord(nr.capEv, st);
    }
    bool capSent = false;

    // (3) the previous cur becomes prev, then pack the new frame into the cur half
    // (native DRBA keeps its own history ring, so it skips this). dX's halves trade roles (xSwap) unless an IFNet reads
    // dX itself as its x, one tensor with prev in planes 0..2: then the cur half is copied into the prev half
    const size_t mplane = nr.mph ? (size_t)nr.mph * nr.mpw : plane;
    if (nr.havePrev && !nr.noEngine && !nr.drba)
    {
        if (!nr.gmfss && !nr.nvof && !nr.fruc && !nr.mph && !nr.xHalf)
            cudaMemcpyAsync(nr.dX, nr.dX + 3 * plane, 3 * plane * sizeof(float), cudaMemcpyDeviceToDevice, st);
        else
            nr.xSwap = !nr.xSwap;
        // the frames the IFNet reads keep the same two frames: the motion frames, or the fp16 copy an
        // fp16-x IFNet reads
        if (nr.mph)
            cudaMemcpyAsync(nr.dM, nr.dM + 3 * mplane, 3 * mplane * sizeof(uint16_t), cudaMemcpyDeviceToDevice, st);
        else if (nr.xHalf)
            cudaMemcpyAsync(nr.dXh, nr.dXh + 3 * plane, 3 * plane * sizeof(uint16_t), cudaMemcpyDeviceToDevice, st);
    }
    float* dCur = nativeXCur(nr);
    // live DLSS 5's identical-frame reuse: a capture byte-identical to the previous one takes the
    // last DLSS 5 output and skips TrueHDR, Restore, the resize and the evaluate (the history stays
    // untouched, so a paused picture stays exactly still)
    bool nrSame = false;
    if (nr.liveNr && !nr.nrFailed)
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
        if (cudaMemcpyAsync(nr.dCapPrev, nr.dCap, (size_t)nb, cudaMemcpyDeviceToDevice, st) != cudaSuccess)
        {
            nr.die("DLSS 5 reuse reference copy failed");
            return false;
        }
        if (nrSame)
        {
            // the motion frame of an identical capture is the previous one (the shift put it in the prev
            // half; DRBA's push copies it inside its ring)
            if (cudaMemcpyAsync(dCur, nr.dNrLast, 3 * plane * sizeof(float), cudaMemcpyDeviceToDevice, st) !=
                    cudaSuccess ||
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
        void* ae[] = {&nr.dCap, &nr.cw, &nr.ch, &nr.sdrScale, &nr.dThdrIn, &nr.dSrcG};
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
    // run instead.
    bool restored = false;
    if (!nrSame && nr.vsrPre && !nr.vsrFailed && !sdrPre)
    {
        const int cps = nr.cw * nr.ch;
        void* ap[] = {&nr.dCap, &nr.cw, &nr.ch, &nr.dCapF, &nr.ch, &nr.cw, (void*)&cps};
        if (cuLaunchKernel(nr.fPackInDirect, (nr.cw + 15) / 16, (nr.ch + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, ap,
                           nullptr) != CUDA_SUCCESS)
        {
            nr.die("capture planes (RTX VSR) launch failed");
            return false;
        }
        const void* src = nr.dCapF;
        int ps = cps, rs = nr.cw, half = 0;
        bool staged = false;
        if (!nativeOfflineStage(nr, src, ps, rs, half, staged, nr.cw, nr.ch, nr.w, nr.h, true, true))
            return false;
        if (staged)
        {
            const float* from = nr.dPres;
            int fps = nr.w * nr.h, dps = (int)plane;
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
    if (!restored && !nrSame && nr.restPre && nr.ctxRest && !nr.restFailed)
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
        if (!nativeRestoreRun(nr, capPl, cps, nr.cw, nr.dRest, nr.w, nr.h, 0, sdrPre ? 0 : nr.encPost, in == 1))
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
    bool diffDone = false;
    if (!restored && !nrSame)
    {
        const int ps = (int)plane;
        if (sdrPre)
        {
            // the SDR planes already exist at capture resolution, so resize them at every
            // image scale; at 1.00 the triangle filter sits on identity positions (single
            // tap, weight 1) and the pair is an exact copy plus the replicate pad.
            void* a1[] = {&nr.dSrcG, &nr.cw, &nr.ch, &nr.dTmp, &nr.w};
            if (cuLaunchKernel(nr.fResizeHf, (nr.w + 15) / 16, (nr.ch + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a1,
                               nullptr) != CUDA_SUCCESS)
            {
                nr.die("resizeHf (RTX HDR) launch failed");
                return false;
            }
            void* a2[] = {&nr.dTmp, &nr.w, &nr.ch, &dCur, &nr.h, &nr.ph, &nr.pw, (void*)&ps};
            if (cuLaunchKernel(nr.fResizeV, (nr.pw + 15) / 16, (nr.ph + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a2,
                               nullptr) != CUDA_SUCCESS)
            {
                nr.die("resizeV (RTX HDR) launch failed");
                return false;
            }
        }
        else if (packMotion)
        {
            uint32_t asked = 0;
            for (uint32_t i = 0; i < nfr; i++)
                if (fr[i] < 0.999f)
                    asked++;
            const float* prev = nullptr;
            if (asked && nr.havePrev && !nr.drba && g_staticHold && nr.fPairDiff && nr.dStaticFlag && nr.hStaticFlag)
            {
                if (cudaMemsetAsync(nr.dStaticFlag, 0, sizeof(int), st) != cudaSuccess)
                {
                    nr.die("static flag clear failed");
                    return false;
                }
                prev = nativeXPrev(nr);
            }
            int hdr = nr.hdr ? 1 : 0;
            float white = (float)(g_sdrWhite / 80.0), knee = kMotionPqKnee, cap = kMotionPqCap;
            uint16_t* mdst = nr.drba ? drbaMotionNext(nr) : nr.dM + 3 * mplane;
            unsigned long long tex = capDirect ? (unsigned long long)nr.capTex : 0ULL;
            uint8_t* raw = capDirect && rawPass ? nr.dCap : nullptr;
            void* a[] = {&nr.dCap, &tex,   &raw,       &hdr,   &nr.cw, &nr.ch, &dCur,        &mdst,
                         &nr.ph,   &nr.pw, (void*)&ps, &white, &knee,  &cap,   (void*)&prev, &nr.dStaticFlag};
            if (cuLaunchKernel(nr.fPackInMotion, (nr.pw + 15) / 16, (nr.ph + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a,
                               nullptr) != CUDA_SUCCESS)
            {
                nr.die("packInMotion launch failed");
                return false;
            }
            if (capDirect)
                cudaEventRecord(nr.capEv, st);
            diffDone = prev != nullptr;
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
                if (view && !nativeHdrEnc(nr, nr.dCapF, 0, cps, nr.cw, nr.cw, nr.ch, nr.dRsView, 0, cps, nr.cw,
                                          nr.encPost, 1.0f, 0.0f))
                {
                    nr.die("resize SDR view launch failed");
                    return false;
                }
                if (view)
                    rsrc = nr.dRsView;
            }
            void* a1[] = {(void*)&rsrc, &nr.cw, &nr.ch, &nr.dTmp, &nr.w};
            if (cuLaunchKernel(nr.hdr ? nr.fResizeHf : nr.fResizeH, (nr.w + 15) / 16, (nr.ch + 15) / 16, 1, 16, 16, 1,
                               0, (CUstream)st, a1, nullptr) != CUDA_SUCCESS)
            {
                nr.die("resizeH launch failed");
                return false;
            }
            void* a2[] = {&nr.dTmp, &nr.w, &nr.ch, &dCur, &nr.h, &nr.ph, &nr.pw, (void*)&ps};
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
    // DLSS 5: once per captured frame on the model frame, after Restore and the
    // resize and before any model reads it; then Sharpen and RTX TrueHDR (nativePreModelPost), and the
    // frame the model reads is kept for the reuse above
    if (nr.pfEv[1])
        cudaEventRecord(nr.pfEv[1], st);
    if (nr.liveNr && !nr.nrFailed && !nrSame && !nativeNrFrame(nr, dCur, nr.pw, nr.ph, (int)plane))
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
        void* a[] = {(void*)&msrc, &sps,    &srs,    &sw,   &sh,    &mdst, &dps,
                     &nr.mpw,      &nr.mpw, &nr.mph, &mode, &white, &knee, &cap};
        return cuLaunchKernel(nr.fMotionIn, (nr.mpw + 15) / 16, (nr.mph + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a,
                              nullptr) == CUDA_SUCCESS;
    };
    const bool motionHere = nr.mph && !nrSame && !packMotion;
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
    if (nr.liveNr && !nr.nrFailed && !nrSame)
    {
        if (cudaMemcpyAsync(nr.dNrLast, dCur, 3 * plane * sizeof(float), cudaMemcpyDeviceToDevice, st) != cudaSuccess)
        {
            nr.die("DLSS 5 last output copy failed");
            return false;
        }
        nr.nrHaveLast = true;
    }
    if (nr.pfEv[2] && !nrSame)
    {
        cudaEventRecord(nr.pfEv[2], st);
        nr.pfArmed = true;
    }

    // native DRBA: the new frame and its encode into the history ring; the group then shows
    // the LAGGED frame k - 1 as its real frame (the first capture: itself, one slot, exactly
    // RifeDrba's first group)
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
    // pair-level work is needed at all (python skips reuse() for a tween-less group).
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
    // (RifeDrba's static rule, three frames of history needed)
    if (nTween && g_staticHold && nr.fPairDiff && nr.dStaticFlag && nr.hStaticFlag && (!nr.drba || drHist >= 3))
    {
        const int n = (int)(3 * plane);
        float* dPrev = nr.drba ? drbaFrame(nr, nr.drFid - 2) : nativeXPrev(nr);
        float* dNext = nr.drba ? drLag : dCur;
        // the 1:1 pack ran the test already (k_packInMotion)
        if (!diffDone && cudaMemsetAsync(nr.dStaticFlag, 0, sizeof(int), st) != cudaSuccess)
        {
            nr.die("static flag clear failed");
            return false;
        }
        void* ad[] = {&dPrev, &dNext, (void*)&n, &nr.dStaticFlag};
        if (!diffDone && cuLaunchKernel(nr.fPairDiff, (n + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, ad, nullptr) !=
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
    // of the frame, so this is exact and halves the Head work python does per pair).
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
    else if (nr.fruc)
    {
        // Smooth Motion: every frame is packed (the next pair's start), the bridge itself
        // runs per tween below; an identical pair got nTween = 0 above and never reaches FRUC
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
        nvinfer1::Dims4 din{1, 3, nr.mph ? nr.mph : nr.ph, nr.mph ? nr.mpw : nr.pw};
        if (!nr.ctxEnc->setInputShape("img", din))
        {
            nr.die("encode setInputShape rejected (shape outside the engine profile)");
            return false;
        }
        // fp16 frames: the new frame into the fp16 copy's cur half, read by an
        // fp16-x IFNet and / or an fp16-img encode (every new frame, static pairs too); RIFE's two
        // domains: the motion frame
        uint16_t* cur16 = nr.mph ? nr.dM + 3 * mplane : nr.dXh + 3 * plane;
        if (!nr.mph && (nr.xHalf || nr.imgHalf) && !nativeF2h(nr, dCur, cur16, 3 * plane, st))
        {
            nr.die("f2h launch failed");
            return false;
        }
        nr.ctxEnc->setTensorAddress("img", nr.imgHalf ? (void*)cur16 : (void*)dCur);
        // fp16 features: the encode writes dF directly, no widen pass
        const bool widen = nr.encHalf && !nr.featHalf;
        void* encOut = widen ? (void*)nr.dEncHalf : (void*)nr.dF[nr.fCur];
        nr.ctxEnc->setTensorAddress("feat", encOut);
        if (!nr.ctxEnc->enqueueV3(st))
        {
            nr.die("encode enqueueV3 returned false");
            return false;
        }
        if (widen)
        {
            int n = (int)(16 * mplane);
            void* a[] = {&nr.dEncHalf, &nr.dF[nr.fCur], &n};
            if (cuLaunchKernel(nr.fH2f, (n + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, a, nullptr) != CUDA_SUCCESS)
            {
                nr.die("h2f launch failed");
                return false;
            }
        }
    }

    // (5) slots. Tokens go out in slot-index order, exactly like _drain_ready.
    if (nr.grEv[0])
        cudaEventRecord(nr.grEv[0], st);
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
    // vertical pass fused with the slot store. k_packOutTile runs both instead (the same bytes, no dFitTmp
    // round trip) where it measured faster: a shrinking height, in blocks of 32 columns by the tallest of
    // 32 / 16 / 8 rows whose tables and tile fit the 48 KB of shared memory a launch gets without opting in
    // (tap counts = lzTaps), and an enlarge of at most 2.75x, in blocks of 16 x 32 (at radius 4: 5 to 27 %
    // faster a call; a tie at 3x and 3.56x, slower at 4x); none fits = the pair
    auto fitPack = [&](const void* src, int ps, int rs, int sw, int sh, uint8_t* slot, const char* what,
                       int half = 0) -> bool {
        auto tileLaunch = [&](int bx, int by, bool& launched) -> bool {
            const int fsh = lzTaps(sw, nr.dw), fsv = lzTaps(sh, nr.dh);
            int hdr = nr.hdr ? 1 : 0;
            int rcap = (int)(((long long)(by - 1) * sh + nr.dh - 1) / nr.dh) + fsv + 2;
            const size_t smem = ((size_t)fsh * bx + (size_t)fsv * by) * 8 + 8 + (size_t)12 * rcap * bx;
            launched = false;
            if (smem > 48 * 1024)
                return true;
            void* at[] = {(void*)&src,    &half,  &ps,    &rs,    &sw,    &sh,  &slot,
                          (void*)&pitchI, &nr.x0, &nr.y0, &nr.dw, &nr.dh, &hdr, &rcap};
            if (cuLaunchKernel(nr.fPackOutTile, (nr.dw + bx - 1) / bx, (nr.dh + by - 1) / by, 1, bx, by, 1,
                               (unsigned)smem, (CUstream)st, at, nullptr) != CUDA_SUCCESS)
            {
                nr.die(what);
                return false;
            }
            launched = true;
            return true;
        };
        bool launched = false;
        if (nr.dh < sh)
        {
            for (int by = 32; by >= 8 && !launched; by /= 2)
                if (!tileLaunch(32, by, launched))
                    return false;
        }
        else if (nr.dh > sh && 4LL * nr.dh <= 11LL * sh)
        {
            if (!tileLaunch(16, 32, launched))
                return false;
            if (launched && !nr.fitUpLogged)
            {
                nr.fitUpLogged = true;
                LOG("native: the fit's enlarge %dx%d -> %dx%d runs in one kernel\n", sw, sh, nr.dw, nr.dh);
            }
        }
        if (launched)
            return true;
        void* ah[] = {(void*)&src, &half, &ps, &rs, &sw, &sh, &nr.dFitTmp, &nr.dw};
        if (cuLaunchKernel(nr.fFitAaH, (nr.dw + 15) / 16, (sh + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, ah, nullptr) !=
            CUDA_SUCCESS)
        {
            nr.die(what);
            return false;
        }
        void* av[] = {&nr.dFitTmp, &nr.dw, &sh, &slot, (void*)&pitchI, &nr.x0, &nr.y0, &nr.dh};
        if (cuLaunchKernel(nr.hdr ? nr.fPackOutHdrV : nr.fPackOutV, (nr.dw + 15) / 16, (nr.dh + 15) / 16, 1, 16, 16, 1,
                           0, (CUstream)st, av, nullptr) != CUDA_SUCCESS)
        {
            nr.die(what);
            return false;
        }
        return true;
    };
    // one resize of a planar source (ps / rs / sw x sh) into a planar target tw x th: the
    // Lanczos pair through tmp (3, sh, tw) when aa, else sampleOut's Lanczos (an
    // enlarging or 1:1 resize). With tile (Upscale to's nr.upTile) the pair runs as k_fitAaTile (the same
    // bytes, no tmp round trip) at fitPack's blocks; none fits = the pair
    auto resizePlanar = [&](const void* src, int ps, int rs, int sw, int sh, bool aa, float* tmp, float* dst, int tw,
                            int th, int half = 0, bool tile = false) -> bool {
        if (aa && tile)
        {
            bool launched = false;
            if (!nativeFitAaTile(nr, src, half, ps, rs, sw, sh, dst, tw, th, st, launched))
                return false;
            if (launched)
                return true;
        }
        if (aa)
        {
            void* ah[] = {(void*)&src, &half, &ps, &rs, &sw, &sh, &tmp, &tw};
            if (cuLaunchKernel(nr.fFitAaH, (tw + 15) / 16, (sh + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, ah,
                               nullptr) != CUDA_SUCCESS)
            {
                nr.die("fitAaH launch failed");
                return false;
            }
            void* av[] = {&tmp, &tw, &sh, &dst, &th};
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
        int ps = (int)plane, rs = nr.pw;
        const bool vsrNow = nr.vsr && !nr.vsrPre && !nr.vsrFailed && (g_rtxb.created || nr.fsrUp);
        if (!vsrNow && !nr.fitAa && !nr.uw)
            return plainPack(src, slot, what, half);
        if (!vsrNow && nr.fitAa && !nr.uw)
            return fitPack(src, ps, rs, nr.w, nr.h, slot, what, half);
        // Upscale to: the first resize lands in the internal render frame (uw x uh)
        // instead of the staging frame, then the fit takes it to (dw, dh) below
        const int tw = nr.uw ? nr.uw : nr.dw, th = nr.uw ? nr.uh : nr.dh;
        float* stage = nr.uw ? nr.dUp : nr.dPres;
        bool haveVsr = false;
        if (vsrNow)
        {
            void* a[] = {(void*)&src, &half, (void*)&ps, (void*)&rs, &nr.w, &nr.h, &nr.dVsrIn};
            if (cuLaunchKernel(nr.fPackBgraRgb, (nr.w + 15) / 16, (nr.h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a,
                               nullptr) != CUDA_SUCCESS)
            {
                nr.die("packBgra launch failed");
                return false;
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
                    return packFrom(src, ps, rs, nr.w, nr.h, slot, what, half);
            }
        }
        if (haveVsr && !nr.uw)
        {
            // the bridge output is the presented frame: into the content rect as it is
            if (cudaMemcpy2DAsync(slot + (size_t)nr.y0 * nr.pitch + (size_t)nr.x0 * 4, nr.pitch, nr.dVsrOut,
                                  (size_t)nr.dw * 4, (size_t)nr.dw * 4, nr.dh, cudaMemcpyDeviceToDevice,
                                  st) != cudaSuccess)
            {
                nr.die("VSR slot copy failed");
                return false;
            }
            return true;
        }
        // the first resize target (the internal render frame, or the staging frame at the
        // presented size): the VSR output unpacked, or the Lanczos fit
        if (haveVsr)
        {
            void* a[] = {&nr.dVsrOut, (void*)&tw, (void*)&th, &stage};
            if (cuLaunchKernel(nr.fUnpackBgraRgb, (tw + 15) / 16, (th + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a,
                               nullptr) != CUDA_SUCCESS)
            {
                nr.die("unpackBgra launch failed");
                return false;
            }
        }
        else if (!resizePlanar(src, ps, rs, nr.w, nr.h, nr.uw ? nr.upAa : nr.fitAa, nr.uw ? nr.dUpTmp : nr.dFitTmp,
                               stage, tw, th, half, nr.uw && nr.upTile))
            return false;
        // the second resize, internal render frame -> the fit rect: the pair with the fused store (the
        // plain packer's 1:1 copy when the size stays)
        if (nr.uw)
            return nr.fitAa ? fitPack(nr.dUp, nr.uw * nr.uh, nr.uw, nr.uw, nr.uh, slot, what)
                            : packFrom(nr.dUp, nr.uw * nr.uh, nr.uw, nr.uw, nr.uh, slot, what);
        // the staging frame is the presented frame, stored 1:1 (sampleOut's identity branch reads it as a
        // dw x dh source with no pad)
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
            else if (rawPass)
            {
                // bit-exact passthrough of the raw capture (rawPass above; capDirect: the pack wrote dCap)
                if (cudaMemcpy2DAsync(slot, nr.pitch, nr.dCap, (size_t)nr.cw * 4, (size_t)nr.cw * 4, nr.ch,
                                      cudaMemcpyDeviceToDevice, st) != cudaSuccess)
                {
                    nr.die("real slot copy failed");
                    fail = true;
                    break;
                }
                heldSlot = slot;
            }
            else if (!storeSlot(nr.drba ? drLag : dCur, slot, "packOut (real) launch failed"))
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
            // lagged real frame where python's group holds (a head without enough history)
            bool held = false;
            if (!nativeDrbaTween(nr, fr[i], drHist, held, false, nr.blendOut ? slot : nullptr))
            {
                fail = true;
                break;
            }
            if ((held || !nr.blendOut) && !storeSlot(held ? (const void*)drLag : mergedAt(nr, 0, plane), slot,
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
            // GMFSS: one tween at a time (no batch axis anywhere in its five engines), the
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
            if (!nativeFrucTween(nr, fr[i]))
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
                // next chunk of tweens, up to the engine's batch (1 for the unbatched class; offline's
                // fixed class takes a whole group)
                chunkBase = twDone;
                chunkLen = nTween - chunkBase;
                if (chunkLen > (uint32_t)nr.batchMax)
                    chunkLen = (uint32_t)nr.batchMax;
                // timestep planes are constant maps of t (python: base + float(t))
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
                nvinfer1::Dims4 dx{1, 6, iph, ipw};
                nvinfer1::Dims4 dtst{(int)chunkLen, 1, iph, ipw};
                nvinfer1::Dims4 df{1, 16, iph, ipw};
                if (!nr.ctxIf->setInputShape("x", dx) || !nr.ctxIf->setInputShape("timestep", dtst) ||
                    !nr.ctxIf->setInputShape("f0", df) || !nr.ctxIf->setInputShape("f1", df))
                {
                    nr.die("IFNet setInputShape rejected (shape outside the engine profile)");
                    fail = true;
                    break;
                }
                nr.ctxIf->setTensorAddress("x", nr.mph ? (void*)nr.dM : nr.xHalf ? (void*)nr.dXh : (void*)nr.dX);
                nr.ctxIf->setTensorAddress("timestep", nr.dT);
                nr.ctxIf->setTensorAddress("f0", nr.dF[nr.fCur ^ 1]);
                nr.ctxIf->setTensorAddress("f1", nr.dF[nr.fCur]);
                nr.ctxIf->setTensorAddress("flow", nr.dFlow);
                nr.ctxIf->setTensorAddress("mask", nr.dMask);
                if (!nr.ctxIf->enqueueV3(st))
                {
                    nr.die("IFNet enqueueV3 returned false (outputs would be garbage)");
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
    if (fail)
        return false;
    // the first group has no pair and a group with no slot stores nothing: their times say nothing about either kind
    if (nr.grEv[1] && nr.havePrev && nfr)
    {
        cudaEventRecord(nr.grEv[1], st);
        nr.grArmed = true;
        nr.grTween = twDone;
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
                              nativeCudaInitEarly(nr, nativeCacheDir(nr)) && nativeTrtInit(nr) && nativeRtxInit(nr) &&
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
        nr.vsr ? (nr.fsrUp ? ", fsrup=native" : ", vsr=native") : "", nr.uw ? ", upscale=native" : "",
        nr.fitAa ? ", fit=native-aa" : "", nr.restore && nr.ctxRest ? ", restore=native" : "");
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
// startNative, or a session that ends before it): free its CUDA and TRT state from the calling
// thread. Without a stream nothing CUDA was created (the stream is the first CUDA object), and
// the runtime DLL may not even be loaded, so no CUDA call is made at all then.
static void nativeDropEarly(NativeRife& nr)
{
    if (!nr.stream)
        return;
    nativeBindDevice(nr);
    nativeFree(nr);
}
