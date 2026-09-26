// part of smv-live.cpp (unity include, see the file map there); not a standalone translation unit
// ---------------------------------------------------------------- the native model host
//
// Every live model runs INSIDE this process: the TensorRT-RTX C++ runtime for the engine
// models (the host looks its engines up by name and builds missing ones from engine\onnx), the
// driver's Optical Flow API for nvof, the shipped bridge DLL for fruc. No python process is
// part of a live session. The python live server this host replaced (engine/live_server.py,
// deleted 2026-09-21, in the git history) is where most of the math below was ported from;
// comments that name its classes and functions record that origin.
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
#include <NvInfer.h>              // the builder: engines built from engine\onnx (ONNX-in-exe 3c-2)
#include <NvOnnxParser.h>         // tensorrt_onnxparser_rtx_1_6.dll, delay-loaded like the runtime
#include "nvOpticalFlowCuda.h"   // the nvof model: MIT interface headers in build_src\nvofa

// ---- the two device kernels, JIT-compiled by NVRTC at first native start ------------------
// Half-to-float is done with the PTX instruction rather than cuda_fp16.h so the kernel source
// needs no include path at all. Channel order is (B,G,R) end to end, exactly as the python
// route sees it (_CapTex hands the model a BGRA texture and permutes the first three planes).
static const char kNativeKernels[] = R"CUDASRC(
extern "C" {

__device__ __forceinline__ float h2f(unsigned short h)
{
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

// triangle (bilinear) filter kernel
__device__ __forceinline__ float tri(float x)
{
    x = x < 0.0f ? -x : x;
    return x < 1.0f ? 1.0f - x : 0.0f;
}

// torch's cubic convolution coefficients, A = -0.75, align_corners = false
__device__ __forceinline__ float cc1(float x, float A)
{
    return ((A + 2.0f) * x - (A + 3.0f)) * x * x + 1.0f;
}
__device__ __forceinline__ float cc2(float x, float A)
{
    return ((A * x - 5.0f * A) * x + 8.0f * A) * x - 4.0f * A;
}

// scale == 1.0 fast path: BGRA8 -> planar float [0,1], replicate-padded to (ph, pw).
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
    dst[o] = p[0] * (1.0f / 255.0f);
    dst[planeStride + o] = p[1] * (1.0f / 255.0f);
    dst[2 * planeStride + o] = p[2] * (1.0f / 255.0f);
}

// antialiased bilinear downscale, horizontal pass: BGRA8 (cw x ch) -> planar float (3, ch, w).
// Weight construction is torch's _compute_weights_aa for mode='bilinear', antialias=True.
__global__ void k_resizeH(const unsigned char* __restrict__ src, int cw, int ch,
                          float* __restrict__ tmp, int w)
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int sy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= w || sy >= ch) return;
    const float scale = (float)cw / (float)w;      // > 1 (this pass only ever downscales)
    const float support = scale;
    const float center = scale * (ox + 0.5f);
    int xmin = (int)floorf(center - support + 0.5f);
    if (xmin < 0) xmin = 0;
    int xmax = (int)floorf(center + support + 0.5f);
    if (xmax > cw) xmax = cw;
    const float inv = 1.0f / scale;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, wsum = 0.0f;
    for (int j = xmin; j < xmax; j++)
    {
        const float wt = tri((j - center + 0.5f) * inv);
        if (wt == 0.0f) continue;
        const unsigned char* p = src + ((size_t)sy * cw + j) * 4;
        a0 += wt * p[0];
        a1 += wt * p[1];
        a2 += wt * p[2];
        wsum += wt;
    }
    const float n = wsum > 0.0f ? 1.0f / (wsum * 255.0f) : 0.0f;
    const size_t plane = (size_t)ch * w;
    const size_t o = (size_t)sy * w + ox;
    tmp[o] = a0 * n;
    tmp[plane + o] = a1 * n;
    tmp[2 * plane + o] = a2 * n;
}

// vertical pass plus the replicate pad, writing straight into the (3, ph, pw) model half.
__global__ void k_resizeV(const float* __restrict__ tmp, int w, int ch,
                          float* __restrict__ dst, int h, int ph, int pw, int planeStride)
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= pw || oy >= ph) return;
    const int tx = ox < w ? ox : w - 1;            // replicate pad in x
    const int ty = oy < h ? oy : h - 1;            // replicate pad in y
    const float scale = (float)ch / (float)h;
    const float support = scale;
    const float center = scale * (ty + 0.5f);
    int ymin = (int)floorf(center - support + 0.5f);
    if (ymin < 0) ymin = 0;
    int ymax = (int)floorf(center + support + 0.5f);
    if (ymax > ch) ymax = ch;
    const float inv = 1.0f / scale;
    const size_t plane = (size_t)ch * w;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, wsum = 0.0f;
    for (int j = ymin; j < ymax; j++)
    {
        const float wt = tri((j - center + 0.5f) * inv);
        if (wt == 0.0f) continue;
        const size_t o = (size_t)j * w + tx;
        a0 += wt * tmp[o];
        a1 += wt * tmp[plane + o];
        a2 += wt * tmp[2 * plane + o];
        wsum += wt;
    }
    const float n = wsum > 0.0f ? 1.0f / wsum : 0.0f;
    const int o = oy * pw + ox;
    dst[o] = a0 * n;
    dst[planeStride + o] = a1 * n;
    dst[2 * planeStride + o] = a2 * n;
}

// ---- WO-15 phase 2, HDR. Verbatim math of live_server.py _scrgb_to_pq2020 (~48-64) and
// _pack_r10a2 (~66-73): the BT.2087 709 to 2020 matrix runs FIRST, the >= 0 clamp is AFTER it
// (negative scRGB is valid wide-gamut colour that only goes out of range once in BT.2020),
// scRGB 1.0 = 80 nits against PQ 1.0 = 10000, then the ST 2084 inverse EOTF. The capture
// texture is R16G16B16A16_FLOAT so the channels really are (R,G,B) here, unlike the SDR path
// where the BGRA order rides through untouched.
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

// HDR, image scale under 1.00: convert the WHOLE capture to planar PQ first, because python
// rescales ON PQ (the WO-8 fill rule), then the shared resize pair runs on those values.
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

// horizontal antialiased downscale of an ALREADY planar float source (the HDR input path);
// identical weights to k_resizeH, which reads BGRA8 instead.
__global__ void k_resizeHf(const float* __restrict__ src, int cw, int ch,
                           float* __restrict__ tmp, int w)
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int sy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= w || sy >= ch) return;
    const float scale = (float)cw / (float)w;
    const float support = scale;
    const float center = scale * (ox + 0.5f);
    int xmin = (int)floorf(center - support + 0.5f);
    if (xmin < 0) xmin = 0;
    int xmax = (int)floorf(center + support + 0.5f);
    if (xmax > cw) xmax = cw;
    const float inv = 1.0f / scale;
    const size_t splane = (size_t)ch * cw;
    const size_t dplane = (size_t)ch * w;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, wsum = 0.0f;
    for (int j = xmin; j < xmax; j++)
    {
        const float wt = tri((j - center + 0.5f) * inv);
        if (wt == 0.0f) continue;
        const size_t o = (size_t)sy * cw + j;
        a0 += wt * src[o];
        a1 += wt * src[splane + o];
        a2 += wt * src[2 * splane + o];
        wsum += wt;
    }
    const float n = wsum > 0.0f ? 1.0f / wsum : 0.0f;
    const size_t o = (size_t)sy * w + ox;
    tmp[o] = a0 * n;
    tmp[dplane + o] = a1 * n;
    tmp[2 * dplane + o] = a2 * n;
}

__global__ void k_h2f(const unsigned short* __restrict__ src, float* __restrict__ dst, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = h2f(src[i]);
}

// one output pixel of a slot: crop the model pad, then copy 1:1 when the slot matches the
// model size, else upscale (bicubic, torch's align_corners=false form, A = -0.75, edges
// clamped). The SDR and HDR slot packers below share this sampler and differ only in the
// store.
__device__ __forceinline__ void sampleOut(const float* __restrict__ src, int planeStride,
                                          int rowStride, int w, int h, int ox, int oy,
                                          int dw, int dh, float c[3])
{
    if (dw == w && dh == h)
    {
        const size_t o = (size_t)oy * rowStride + ox;
        c[0] = src[o];
        c[1] = src[planeStride + o];
        c[2] = src[2 * planeStride + o];
        return;
    }
    // the tap centre in DOUBLE (2026-09-15, Upscale to gate): torch's fp32 kernel computes
    // scale * (o + 0.5) - 0.5 in float and drifts 1e-4 from its own fp64 result by output
    // index 2500, so an fp32 centre here tracked the fp32 drift, not the filter; same rule as
    // aaWindow below, two double ops per pixel
    const double rx = (double)w / (double)dw * (ox + 0.5) - 0.5;
    const double ry = (double)h / (double)dh * (oy + 0.5) - 0.5;
    const double fxd = floor(rx), fyd = floor(ry);
    const float tx = (float)(rx - fxd), ty = (float)(ry - fyd);
    const int ix = (int)fxd, iy = (int)fyd;
    const float A = -0.75f;
    float wx[4], wy[4];
    wx[0] = cc2(tx + 1.0f, A); wx[1] = cc1(tx, A);
    wx[2] = cc1(1.0f - tx, A); wx[3] = cc2(2.0f - tx, A);
    wy[0] = cc2(ty + 1.0f, A); wy[1] = cc1(ty, A);
    wy[2] = cc1(1.0f - ty, A); wy[3] = cc2(2.0f - ty, A);
    for (int ci = 0; ci < 3; ci++)
    {
        const float* pl = src + (size_t)ci * planeStride;
        float acc = 0.0f;
        for (int j = 0; j < 4; j++)
        {
            int yy = iy - 1 + j;
            yy = yy < 0 ? 0 : (yy > h - 1 ? h - 1 : yy);
            float row = 0.0f;
            for (int i = 0; i < 4; i++)
            {
                int xx = ix - 1 + i;
                xx = xx < 0 ? 0 : (xx > w - 1 ? w - 1 : xx);
                row += wx[i] * pl[(size_t)yy * rowStride + xx];
            }
            acc += wy[j] * row;
        }
        c[ci] = acc;
    }
}

// one output slot: sampleOut, clamp, round, and write BGRA8 into the content rect of a
// pitched ring slot.
__global__ void k_packOut(const float* __restrict__ src, int planeStride, int rowStride,
                          int w, int h, unsigned char* __restrict__ dst, int pitch,
                          int x0, int y0, int dw, int dh)
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= dw || oy >= dh) return;
    float c[3];
    sampleOut(src, planeStride, rowStride, w, h, ox, oy, dw, dh, c);
    unsigned char* p = dst + (size_t)(y0 + oy) * pitch + (size_t)(x0 + ox) * 4;
    for (int ci = 0; ci < 3; ci++)
    {
        float v = c[ci];
        v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        p[ci] = (unsigned char)(int)rintf(v * 255.0f);
    }
    p[3] = 255;
}

// HDR output slot: the same sampleOut and clamp, but the canvas is R10G10B10A2 packed into
// int32 with A = 3, exactly what _pack_r10a2 produces and what the R10A2 present reads back.
__global__ void k_packOutHdr(const float* __restrict__ src, int planeStride, int rowStride,
                             int w, int h, unsigned char* __restrict__ dst, int pitch,
                             int x0, int y0, int dw, int dh)
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= dw || oy >= dh) return;
    float c[3];
    sampleOut(src, planeStride, rowStride, w, h, ox, oy, dw, dh, c);
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

// ---- live Sharpen and RTX VSR inside the native host (2026-09-15) --------------------------
// Both effects run at the PRESENTED size, after the fit, exactly like live_server.py's compose
// (upscale, then RCAS): the fit lands in a planar staging frame at dw x dh (k_fitPlanar, or
// the VSR bridge output unpacked by k_unpackBgra), and k_rcasOut / k_rcasOutHdr apply rcas.py
// on it while doing the slot store that k_packOut / k_packOutHdr do without effects.

// crop the model pad and quantise to tight BGRA8 at the model size: the VSR bridge input,
// rtxvideo.py run_vsr (round(x * 255), B G R A = 255)
__global__ void k_packBgra(const float* __restrict__ src, int planeStride, int rowStride,
                           int w, int h, unsigned char* __restrict__ dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t o = (size_t)y * rowStride + x;
    unsigned char* p = dst + ((size_t)y * w + x) * 4;
    for (int ci = 0; ci < 3; ci++)
    {
        float v = src[(size_t)ci * planeStride + o];
        v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        p[ci] = (unsigned char)(int)rintf(v * 255.0f);
    }
    p[3] = 255;
}

// the VSR bridge output (tight BGRA8 at the presented size) back to planar float [0,1]
__global__ void k_unpackBgra(const unsigned char* __restrict__ src, int dw, int dh,
                             float* __restrict__ dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    const size_t plane = (size_t)dw * dh;
    const size_t o = (size_t)y * dw + x;
    const unsigned char* p = src + o * 4;
    dst[o] = p[0] * (1.0f / 255.0f);
    dst[plane + o] = p[1] * (1.0f / 255.0f);
    dst[2 * plane + o] = p[2] * (1.0f / 255.0f);
}

// the OFFLINE route's planes are (R, G, B) (the decoder's rgb, k_packInRaw8 / 16), not the live
// (B, G, R): the VSR bridge input and output as rtxvideo.py run_vsr packs and unpacks them
// (R, G, B planes into B G R A bytes, back with a true / 255)
__global__ void k_packBgraRgb(const float* __restrict__ src, int planeStride, int rowStride,
                              int w, int h, unsigned char* __restrict__ dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t o = (size_t)y * rowStride + x;
    unsigned char* p = dst + ((size_t)y * w + x) * 4;
    for (int ci = 0; ci < 3; ci++)
    {
        float v = src[(size_t)ci * planeStride + o];
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

// RGBA8 back to the offline (R, G, B) planes (priority 24 step 5): a DLSS 4.5 host frame,
// dlssg.py _recv (uint8 / 255.0, a division, as python; k_unpackBgra multiplies)
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

// the bicubic fit into the planar staging frame (the RCAS input when VSR is off)
__global__ void k_fitPlanar(const float* __restrict__ src, int planeStride, int rowStride,
                            int w, int h, float* __restrict__ dst, int dw, int dh)
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= dw || oy >= dh) return;
    float c[3];
    sampleOut(src, planeStride, rowStride, w, h, ox, oy, dw, dh, c);
    const size_t plane = (size_t)dw * dh;
    const size_t o = (size_t)oy * dw + ox;
    dst[o] = c[0];
    dst[plane + o] = c[1];
    dst[2 * plane + o] = c[2];
}

// ---- the downscaling fit (2026-09-15, full native migration item 1) ------------------------
// torch F.interpolate(mode='bicubic', antialias=True, align_corners=False) as a separable
// pair into the planar staging frame. Weight construction is torch's _compute_weights_aa
// with its bicubic_filter (A = -0.5, the PIL-compatible kernel, NOT sampleOut's -0.75): a
// shrinking axis widens the support to 2 * scale and rescales the taps by 1 / scale, an
// enlarging axis keeps support 2 and unit taps. Python turns antialias on whenever the fit
// HEIGHT shrinks (_Fit._upscale), so both axes run this filter then, whatever the width does.
__device__ __forceinline__ float bcaa(float x)
{
    const float A = -0.5f;
    x = x < 0.0f ? -x : x;
    if (x < 1.0f) return ((A + 2.0f) * x - (A + 3.0f)) * x * x + 1.0f;
    if (x < 2.0f) return (((x - 5.0f) * x + 8.0f) * x - 4.0f) * A;
    return 0.0f;
}

// one axis: the tap window [mn, mx) of output index o, its centre and the tap rescale. In
// DOUBLE: an fp32 centre at output index 2500 carries a 1e-4 error, which is exactly how far
// torch's own fp32 CUDA kernel drifts from its fp64 result on a near-1:1 fit (2576 -> 2560:
// 6.9e-5, 2026-09-15 probe); this pair tracks the exact filter instead, a few double ops per
// tap against the memory traffic.
__device__ __forceinline__ void aaWindow(int o, int in, int out, double& center, double& inv,
                                         int& mn, int& mx)
{
    const double scale = (double)in / (double)out;
    const double support = scale >= 1.0 ? 2.0 * scale : 2.0;
    inv = scale >= 1.0 ? 1.0 / scale : 1.0;
    center = scale * (o + 0.5);
    mn = (int)(center - support + 0.5);
    if (mn < 0) mn = 0;
    mx = (int)(center + support + 0.5);
    if (mx > in) mx = in;
}

// horizontal pass: the model pad cropped to (w, h) -> tmp (3, h, dw)
__global__ void k_fitAaH(const float* __restrict__ src, int planeStride, int rowStride,
                         int w, int h, float* __restrict__ tmp, int dw)
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= dw || y >= h) return;
    double center, inv;
    int mn, mx;
    aaWindow(ox, w, dw, center, inv, mn, mx);
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, wsum = 0.0f;
    for (int j = mn; j < mx; j++)
    {
        const float wt = bcaa((float)((j - center + 0.5) * inv));
        const size_t o = (size_t)y * rowStride + j;
        a0 += wt * src[o];
        a1 += wt * src[planeStride + o];
        a2 += wt * src[2 * planeStride + o];
        wsum += wt;
    }
    const float n = wsum != 0.0f ? 1.0f / wsum : 1.0f;   // torch normalises unless the sum is 0
    const size_t plane = (size_t)h * dw;
    const size_t o = (size_t)y * dw + ox;
    tmp[o] = a0 * n;
    tmp[plane + o] = a1 * n;
    tmp[2 * plane + o] = a2 * n;
}

// vertical pass: tmp (3, h, dw) -> the planar staging frame (3, dh, dw)
__global__ void k_fitAaV(const float* __restrict__ tmp, int dw, int h,
                         float* __restrict__ dst, int dh)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || oy >= dh) return;
    double center, inv;
    int mn, mx;
    aaWindow(oy, h, dh, center, inv, mn, mx);
    const size_t splane = (size_t)h * dw;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, wsum = 0.0f;
    for (int j = mn; j < mx; j++)
    {
        const float wt = bcaa((float)((j - center + 0.5) * inv));
        const size_t o = (size_t)j * dw + x;
        a0 += wt * tmp[o];
        a1 += wt * tmp[splane + o];
        a2 += wt * tmp[2 * splane + o];
        wsum += wt;
    }
    const float n = wsum != 0.0f ? 1.0f / wsum : 1.0f;
    const size_t plane = (size_t)dw * dh;
    const size_t o = (size_t)oy * dw + x;
    dst[o] = a0 * n;
    dst[plane + o] = a1 * n;
    dst[2 * plane + o] = a2 * n;
}

// ---- live Restore (2026-09-15, full native migration item 3) -------------------------------
// live_server._Fit._restore: the Real-ESRGAN TensorRT engine (x = the model frame as fp16
// NCHW [1,3,h,w], y = its 4x reconstruction [1,3,4h,4w]) then realesr.fit to the restore
// target: `out.clamp(0,1)` first, an antialiased bicubic when the target height shrinks
// (the pair above with the clamp folded in and the engine's own dtype read at the taps),
// plain bicubic when it enlarges (k_restToF + k_fitPlanar + k_clamp01), identity when equal.
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

// the engine input: the model pad cropped to (w, h), fp32 planar -> fp16 NCHW (torch .half())
__global__ void k_restIn(const float* __restrict__ src, int planeStride, int rowStride,
                         int w, int h, unsigned short* __restrict__ dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t o = (size_t)y * w + x, plane = (size_t)w * h;
    const size_t s = (size_t)y * rowStride + x;
    dst[o] = f2h(src[s]);
    dst[plane + o] = f2h(src[planeStride + s]);
    dst[2 * plane + o] = f2h(src[2 * planeStride + s]);
}

// the shrinking fold, horizontal pass: the 4x output (fp16 or fp32 planar, clamped per tap)
// -> tmp (3, h, dw); the window maths is aaWindow's
__global__ void k_restFoldH(const void* __restrict__ src, int half, int planeStride, int rowStride,
                            int w, int h, float* __restrict__ tmp, int dw)
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= dw || y >= h) return;
    double center, inv;
    int mn, mx;
    aaWindow(ox, w, dw, center, inv, mn, mx);
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, wsum = 0.0f;
    for (int j = mn; j < mx; j++)
    {
        const float wt = bcaa((float)((j - center + 0.5) * inv));
        const size_t o = (size_t)y * rowStride + j;
        a0 += wt * restTap(src, half, o);
        a1 += wt * restTap(src, half, planeStride + o);
        a2 += wt * restTap(src, half, 2 * planeStride + o);
        wsum += wt;
    }
    const float n = wsum != 0.0f ? 1.0f / wsum : 1.0f;
    const size_t plane = (size_t)h * dw;
    const size_t o = (size_t)y * dw + ox;
    tmp[o] = a0 * n;
    tmp[plane + o] = a1 * n;
    tmp[2 * plane + o] = a2 * n;
}

// vertical pass with realesr.fit's final clamp: tmp (3, h, dw) -> the planar target (3, dh, dw)
__global__ void k_restFoldV(const float* __restrict__ tmp, int dw, int h,
                            float* __restrict__ dst, int dh)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || oy >= dh) return;
    double center, inv;
    int mn, mx;
    aaWindow(oy, h, dh, center, inv, mn, mx);
    const size_t splane = (size_t)h * dw;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, wsum = 0.0f;
    for (int j = mn; j < mx; j++)
    {
        const float wt = bcaa((float)((j - center + 0.5) * inv));
        const size_t o = (size_t)j * dw + x;
        a0 += wt * tmp[o];
        a1 += wt * tmp[splane + o];
        a2 += wt * tmp[2 * splane + o];
        wsum += wt;
    }
    const float n = wsum != 0.0f ? 1.0f / wsum : 1.0f;
    const size_t plane = (size_t)dw * dh;
    const size_t o = (size_t)oy * dw + x;
    a0 *= n; a1 *= n; a2 *= n;
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

// offline DLSS 5 (priority 24 step 2d): dlssnr.py process() around the NR host's RGBA16F
// frame. In: the planar (R, G, B) output-size frame, clamp(0, 1).to(float16) (round to
// nearest even, torch's cast) plus alpha 1.0. Out: the first three channels, .float().clamp(0,
// 1), back to tight planar.
__global__ void k_nrIn(const float* __restrict__ src, int planeStride, int rowStride, int dw, int dh,
                       unsigned short* __restrict__ dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    const size_t i = (size_t)y * rowStride + x;
    unsigned short* d = dst + ((size_t)y * dw + x) * 4;
    for (int c = 0; c < 3; c++)
    {
        const float v = src[(size_t)c * planeStride + i];
        d[c] = f2h(v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v));
    }
    d[3] = 0x3C00;
}
__global__ void k_nrOut(const unsigned short* __restrict__ src, int dw, int dh, float* __restrict__ dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    const size_t o = (size_t)y * dw + x;
    for (int c = 0; c < 3; c++)
    {
        const float v = h2f(src[o * 4 + c]);
        dst[(size_t)c * dw * dh + o] = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
    }
}

__global__ void k_clamp01(float* __restrict__ p, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { const float v = p[i]; p[i] = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }
}

// ---- live GMFSS glue (2026-09-15, full native migration item 5 sub-step 5b) ---------------
// The eager glue of GMFSS_infer_u.Model.inference() as four kernels; no chain calls them yet
// (sub-step 5c). Every buffer is planar NCHW with N = 1, contiguous unless a stride is passed.
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
//               same int64 atomics; the position and the weights are formed in double (below),
//               so the accumulators match python's kernel bit for bit only where the position
//               is exactly representable (the harness's quantized-flow control) and sit nearer
//               the exact splat than python everywhere else; python also takes exp and the
//               product in fp16 under autocast (~1e-3 off fp64), this kernel in fp32.
//   k_splatNorm python's tail: acc.to(float32) * 2^-26 per plane, out = v / (vN + 1e-7),
//               written as C planes of the target (the fusionnet input planes: a channel concat
//               is adjacent planes, so no cat).
// Gate: harness\eff\gmfss_equiv.py (fp64 references, real-motion inputs through the engines).
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
// both scaled by s, splatted into C + 1 int64 planes (zeroed by the caller; plane C = exp(Z))
__global__ void k_splatSoft(const void* __restrict__ in, int inHalf, int C,
                            const float* __restrict__ flow, const void* __restrict__ metric,
                            int metricHalf, float s, int w, int h, long long* __restrict__ acc)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t plane = (size_t)w * h, o = (size_t)y * w + x;
    // the target position and the four corner weights in DOUBLE: python's fp32 `x + s * flow`
    // carries a 6e-5 px error at x ~ 600..960 (one fp32 ulp), which the harness measured as
    // up to 8.7e-4 of the normalized output where the weight sum is small; the double position
    // tracks the exact splat and the weights round once to fp32 for the fixed-point product
    // (third occurrence of the tap-centre lesson, 2026-09-15)
    const double fx = (double)x + (double)s * (double)flow[o];
    const double fy = (double)y + (double)s * (double)flow[plane + o];
    if (!isfinite(fx) || !isfinite(fy)) return;
    const float e = expf(__fmul_rn(s, gmTap(metric, metricHalf, o)));
    const int nwX = (int)floor(fx), nwY = (int)floor(fy);
    const double dx = fx - (double)nwX, dy = fy - (double)nwY;
    const float wNW = (float)((1.0 - dx) * (1.0 - dy)), wNE = (float)(dx * (1.0 - dy));
    const float wSW = (float)((1.0 - dx) * dy), wSE = (float)(dx * dy);
    const bool okW = nwX >= 0 && nwX < w, okE = nwX + 1 >= 0 && nwX + 1 < w;
    const bool okN = nwY >= 0 && nwY < h, okS = nwY + 1 >= 0 && nwY + 1 < h;
    const long long oNW = (long long)nwY * w + nwX;   // only dereferenced behind the ok flags
    for (int c = 0; c <= C; c++)
    {
        const float v = c < C ? __fmul_rn(gmTap(in, inHalf, (size_t)c * plane + o), e) : e;
        unsigned long long* p = (unsigned long long*)(acc + (size_t)c * plane);
        if (okN && okW) atomicAdd(p + oNW, (unsigned long long)(long long)llrintf(v * wNW * 67108864.0f));
        if (okN && okE) atomicAdd(p + oNW + 1, (unsigned long long)(long long)llrintf(v * wNE * 67108864.0f));
        if (okS && okW) atomicAdd(p + oNW + w, (unsigned long long)(long long)llrintf(v * wSW * 67108864.0f));
        if (okS && okE) atomicAdd(p + oNW + w + 1, (unsigned long long)(long long)llrintf(v * wSE * 67108864.0f));
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

// rcas.py verbatim on one pixel of the planar staging frame: replicate-padded cross taps,
// green-weighted luma (symmetric in the two chroma planes, so BGR planes need no swap), one
// lobe per pixel (the max over the three channels), clamped to [-RCAS_LIMIT, 0] and scaled
// by the strength and the noise term.
__device__ __forceinline__ void rcasPixel(const float* __restrict__ src, int dw, int dh,
                                          int x, int y, float con, float out[3])
{
    const size_t plane = (size_t)dw * dh;
    const int xl = x > 0 ? x - 1 : 0, xr = x < dw - 1 ? x + 1 : dw - 1;
    const int yu = y > 0 ? y - 1 : 0, yd = y < dh - 1 ? y + 1 : dh - 1;
    float b[3], d[3], e[3], f[3], hh[3];
    for (int ci = 0; ci < 3; ci++)
    {
        const float* pl = src + (size_t)ci * plane;
        b[ci] = pl[(size_t)yu * dw + x];
        d[ci] = pl[(size_t)y * dw + xl];
        e[ci] = pl[(size_t)y * dw + x];
        f[ci] = pl[(size_t)y * dw + xr];
        hh[ci] = pl[(size_t)yd * dw + x];
    }
    const float bL = b[1] + 0.5f * (b[0] + b[2]);
    const float dL = d[1] + 0.5f * (d[0] + d[2]);
    const float eL = e[1] + 0.5f * (e[0] + e[2]);
    const float fL = f[1] + 0.5f * (f[0] + f[2]);
    const float hL = hh[1] + 0.5f * (hh[0] + hh[2]);
    const float mx = fmaxf(fmaxf(fmaxf(bL, dL), fmaxf(fL, hL)), eL);
    const float mn = fminf(fminf(fminf(bL, dL), fminf(fL, hL)), eL);
    float nz = fabsf(0.25f * (bL + dL + fL + hL) - eL) / ((mx - mn) + 1e-4f);
    nz = nz < 0.0f ? 0.0f : (nz > 1.0f ? 1.0f : nz);
    nz = -0.5f * nz + 1.0f;
    float lobe = -3.0e38f;
    for (int ci = 0; ci < 3; ci++)
    {
        const float mn4 = fminf(fminf(b[ci], d[ci]), fminf(f[ci], hh[ci]));
        const float mx4 = fmaxf(fmaxf(b[ci], d[ci]), fmaxf(f[ci], hh[ci]));
        const float hitMin = mn4 / (4.0f * mx4 + 1e-4f);
        const float hitMax = (1.0f - mx4) / (4.0f * mn4 - 4.0f - 1e-4f);
        lobe = fmaxf(lobe, fmaxf(-hitMin, hitMax));
    }
    const float RCAS_LIMIT = 0.1875f - 1e-6f;
    lobe = lobe > 0.0f ? 0.0f : lobe;
    lobe = lobe < -RCAS_LIMIT ? -RCAS_LIMIT : lobe;
    lobe = lobe * con * nz;
    for (int ci = 0; ci < 3; ci++)
    {
        float v = (e[ci] + lobe * (b[ci] + d[ci] + f[ci] + hh[ci])) / (1.0f + 4.0f * lobe);
        out[ci] = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
    }
}

// RCAS on the staging frame, then the same BGRA8 slot store as k_packOut
__global__ void k_rcasOut(const float* __restrict__ src, int dw, int dh, float con,
                          unsigned char* __restrict__ dst, int pitch, int x0, int y0)
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= dw || oy >= dh) return;
    float c[3];
    rcasPixel(src, dw, dh, ox, oy, con, c);
    unsigned char* p = dst + (size_t)(y0 + oy) * pitch + (size_t)(x0 + ox) * 4;
    for (int ci = 0; ci < 3; ci++) p[ci] = (unsigned char)(int)rintf(c[ci] * 255.0f);
    p[3] = 255;
}

// RCAS on the PQ staging frame, then the same R10G10B10A2 slot store as k_packOutHdr
__global__ void k_rcasOutHdr(const float* __restrict__ src, int dw, int dh, float con,
                             unsigned char* __restrict__ dst, int pitch, int x0, int y0)
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= dw || oy >= dh) return;
    float c[3];
    rcasPixel(src, dw, dh, ox, oy, con, c);
    unsigned int q[3];
    for (int ci = 0; ci < 3; ci++)
    {
        q[ci] = (unsigned int)(int)(c[ci] * 1023.0f + 0.5f);
        if (q[ci] > 1023u) q[ci] = 1023u;
    }
    unsigned int* p = (unsigned int*)(dst + (size_t)(y0 + oy) * pitch) + (x0 + ox);
    *p = q[0] | (q[1] << 10) | (q[2] << 20) | (3u << 30);
}

// RCAS on the staging frame, then the offline raw store (priority 24 step 2): tight rgb48le
// (out16) or rgb24 at dw x dh, render.py to_bytes' rounding (k_packOutRaw16 / 8's)
__global__ void k_rcasOutRaw(const float* __restrict__ src, int dw, int dh, float con,
                             unsigned char* __restrict__ dst, int out16)
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= dw || oy >= dh) return;
    float c[3];
    rcasPixel(src, dw, dh, ox, oy, con, c);
    const size_t o = ((size_t)oy * dw + ox) * 3;
    if (out16)
        for (int ci = 0; ci < 3; ci++) ((unsigned short*)dst)[o + ci] = (unsigned short)(int)rintf(c[ci] * 65535.0f);
    else
        for (int ci = 0; ci < 3; ci++) dst[o + ci] = (unsigned char)(int)rintf(c[ci] * 255.0f);
}

// ---- WO-23: live RTX TrueHDR inside the native host ---------------------------------------
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
// [0,1]. mode 0 = vivid, 1 = rtx. Shared by live (k_thdrColor) and offline (k_thdrOut).
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

// step 5 of the WO: unpack the bridge output (10:10:10:2, B in the LOW 10 bits), apply the
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

// ---- priority 24 step 2c: offline RTX TrueHDR (rtxvideo.run_hdr on every output frame) ----
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
__global__ void k_thdrIn(const float* __restrict__ src, int planeStride, int rowStride, int dw, int dh,
                         unsigned char* __restrict__ bgra, float* __restrict__ dstG)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    const size_t i = (size_t)y * rowStride + x;
    thdr_in_store(src[i], src[(size_t)planeStride + i], src[2 * (size_t)planeStride + i],
                  (size_t)y * dw + x, (size_t)dw * dh, bgra, dstG);
}

// RCAS on the staging frame first (render_passes: RCAS is the last SDR pass before TrueHDR)
__global__ void k_rcasThdrIn(const float* __restrict__ src, int dw, int dh, float con,
                             unsigned char* __restrict__ bgra, float* __restrict__ dstG)
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= dw || oy >= dh) return;
    float c[3];
    rcasPixel(src, dw, dh, ox, oy, con, c);
    thdr_in_store(c[0], c[1], c[2], (size_t)oy * dw + ox, (size_t)dw * dh, bgra, dstG);
}

// the 10-bit PQ code -> display-linear table of _accum_hp (_pq_to_linear(arange(1024) / 1023))
__global__ void k_pqLut(float* __restrict__ dst)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 1024) dst[i] = pq_eotf((float)i / 1023.0f);
}

// the colour step, the x2rgb10le words and the light statistics of one frame (_pack_out, or
// _measure_light in raw mode). Per frame: hist[1024] = the maxRGB 10-bit code histogram (DV L1,
// HDR10+ percentiles and average), misc[0..2] = the per-channel max code (HDR10+ MaxScl),
// misc[3] = the float bits of the brightest maxRGB value (never negative, so the bits order like
// the floats) and vSum its sum: the corrected linear maxRGB in vivid / rtx (MaxCLL / MaxFALL
// = * 10000), the maxRGB nits in raw. The block stages the histogram in shared memory.
__global__ void k_thdrOut(const unsigned int* __restrict__ thdrOut, const float* __restrict__ srcG,
                          int dw, int dh, int mode, float vib, float sb, unsigned int* __restrict__ dst,
                          unsigned int* __restrict__ hist, unsigned int* __restrict__ misc,
                          double* __restrict__ vSum)
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
    double vv = 0.0;
    if (x < dw && y < dh)
    {
        const size_t plane = (size_t)dh * dw;
        const size_t o = (size_t)y * dw + x;
        const unsigned int u = thdrOut[o];
        unsigned int q[3];
        float val;
        if (mode == 2)
        {
            // raw: the bridge words unchanged; the stats from the maxRGB code in nits
            q[0] = (u >> 20) & 1023u; q[1] = (u >> 10) & 1023u; q[2] = u & 1023u;
            dst[o] = u;
            const unsigned int m = q[0] > q[1] ? (q[0] > q[2] ? q[0] : q[2]) : (q[1] > q[2] ? q[1] : q[2]);
            val = pq_eotf((float)m / 1023.0f) * 10000.0f;
        }
        else
        {
            float v[3];
            thdr_linear(u, srcG[o], srcG[plane + o], srcG[2 * plane + o], mode, vib, sb, v);
            for (int c = 0; c < 3; c++)
            {
                float p = rintf(pq_oetf(v[c]) * 1023.0f);
                p = p < 0.0f ? 0.0f : (p > 1023.0f ? 1023.0f : p);
                q[c] = (unsigned int)p;
            }
            dst[o] = q[2] | (q[1] << 10) | (q[0] << 20) | 0xC0000000u;
            val = fmaxf(fmaxf(v[0], v[1]), v[2]);
        }
        val = val + 0.0f;   // a -0.0 would carry the sign bit into the bit-pattern max
        const unsigned int m = q[0] > q[1] ? (q[0] > q[2] ? q[0] : q[2]) : (q[1] > q[2] ? q[1] : q[2]);
        atomicAdd(&sh[m], 1u);
        atomicMax(&shMax[0], q[0]);
        atomicMax(&shMax[1], q[1]);
        atomicMax(&shMax[2], q[2]);
        atomicMax(&shMax[3], __float_as_uint(val));
        vv = (double)val;
    }
    for (int off = 16; off > 0; off >>= 1) vv += __shfl_down_sync(0xffffffffu, vv, off);
    if ((tid & 31) == 0 && vv != 0.0) atomicAdd(vSum, vv);
    __syncthreads();
    for (int i = tid; i < 1024; i += nt)
        if (sh[i]) atomicAdd(&hist[i], sh[i]);
    if (tid < 4 && shMax[tid]) atomicMax(&misc[tid], shMax[tid]);
}

// WO-32 offline render: raw interleaved RGB frames (rgb48le or rgb24, the decoder pipe format
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
__global__ void k_packOutRaw16(const float* __restrict__ src, int planeStride, int rowStride,
                               int w, int h, unsigned short* __restrict__ dst)
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= w || oy >= h) return;
    const size_t o = (size_t)oy * rowStride + ox;
    unsigned short* p = dst + ((size_t)oy * w + ox) * 3;
    for (int ci = 0; ci < 3; ci++)
    {
        float v = src[(size_t)ci * planeStride + o];
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
__global__ void k_packOutRaw8(const float* __restrict__ src, int planeStride, int rowStride,
                              int w, int h, unsigned char* __restrict__ dst)
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= w || oy >= h) return;
    const size_t o = (size_t)oy * rowStride + ox;
    unsigned char* p = dst + ((size_t)oy * w + ox) * 3;
    for (int ci = 0; ci < 3; ci++)
    {
        float v = src[(size_t)ci * planeStride + o];
        v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        p[ci] = (unsigned char)(int)rintf(v * 255.0f);
    }
}

// ---- native NVOF model glue (2026-09-21, memory priority 20 step 2) -------------------------
// The tween maths of the NVOF model: the NVIDIA Optical Flow Accelerator gives both flow fields
// of a pair in ONE Execute (grid 4, S10.5 vectors in INPUT pixels, uint8 cost per cell), these
// kernels turn them into tweens by a bidirectional softmax FORWARD splat (the GMFSS glue without
// the networks). Step 2 compiles this text through NVRTC in harness\nvof\warp (cupy), step 3
// pastes it into smv-live-native.inl unchanged (gate: harness\nvof\warp\nvof_equiv.py).
// Every buffer is planar with N = 1; an image is 3
// fp32 planes addressed by (planeStride, rowStride) like the host's padded model planes, a flow
// or metric plane is contiguous (w, h).
//   k_nvofLuma      the model planes (0..1) -> GRAYSCALE8 NVOF input, BT.709 luma, x255,
//                   rounded half to even, in double (so the fp64 reference matches exactly).
//   k_nvofUp        one direction: the interleaved short2 field (raw / 32 = input px, NO rescale)
//                   and its uint8 cost, bilinear from the cell grid to (w, h). Source index from
//                   the GRID, (x + 0.5) / grid - 0.5 (a cell covers grid x grid px), clamped at 0,
//                   the far tap held at the last cell, like k_flowUp; centre and lambdas DOUBLE.
//   k_nvofMetric    Z = -a * cost - b * |F(x) + G(x + F(x))|, G = the other direction sampled
//                   bilinear at the landing point (border clamped): forward-backward consistency,
//                   large where the pixel is occluded in the other frame. Position in double.
//   k_splatNvof     k_splatSoft with the flow scale and the weight split: position x + s * F in
//                   double, e = expf(Z + bias) (bias = log of the time weight, 1 - t for frame 0,
//                   t for frame 1), the same four corner weights and int64 atomics (2^40, below).
//   k_splatNvofNorm acc -> the tween; where the plain coverage is below `hole` (nothing landed),
//                   the plain blend (1 - t) I0 + t I1, and the hole is counted.
// fixed point of the splat accumulators: 2^40, not k_splatSoft's 2^26. e = exp(Z + bias) <= 1
// and Z is clamped at -16, so a contribution is at most 2^40 and at least ~1.2e5 units (2^26
// would leave ~7 units there, a 13% colour quantization exactly in the occluded regions); even a
// thousand contributions per pixel stay under 2^50 of the int64 range
#define NV_FIX 1099511627776.0f

__device__ __forceinline__ double nvLerp2(const float* __restrict__ p, int w, int h, double fx, double fy)
{
    // bilinear with the sample point clamped into [0, w - 1] x [0, h - 1]
    if (fx < 0.0) fx = 0.0;
    if (fy < 0.0) fy = 0.0;
    if (fx > (double)(w - 1)) fx = (double)(w - 1);
    if (fy > (double)(h - 1)) fy = (double)(h - 1);
    const int x0 = (int)fx, y0 = (int)fy;
    const int x1 = x0 < w - 1 ? x0 + 1 : x0, y1 = y0 < h - 1 ? y0 + 1 : y0;
    const double lx = fx - (double)x0, ly = fy - (double)y0;
    const double a = (1.0 - lx) * (double)p[(size_t)y0 * w + x0] + lx * (double)p[(size_t)y0 * w + x1];
    const double b = (1.0 - lx) * (double)p[(size_t)y1 * w + x0] + lx * (double)p[(size_t)y1 * w + x1];
    return (1.0 - ly) * a + ly * b;
}

extern "C" __global__ void k_nvofLuma(const float* __restrict__ src, int planeStride, int rowStride,
                                      int w, int h, unsigned char* __restrict__ dst, int pitch)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t o = (size_t)y * rowStride + x;
    // planeStride is SIGNED: the live host's planes are (B, G, R), so it passes the R plane with
    // a negative stride and the BT.709 weights still meet R, G, B in this order
    double v = 0.2126 * (double)src[o] + 0.7152 * (double)src[(long long)planeStride + (long long)o]
             + 0.0722 * (double)src[2 * (long long)planeStride + (long long)o];
    v = v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
    dst[(size_t)y * pitch + x] = (unsigned char)rint(v * 255.0);
}

// vec: gw x gh short2 (x, y) with a row pitch of vecPitch BYTES; cost: gw x gh uint8, costPitch
// bytes; out: flow (2 planes of w x h, px) and cost (1 plane, 0..255 as float)
extern "C" __global__ void k_nvofUp(const short* __restrict__ vec, int vecPitch,
                                    const unsigned char* __restrict__ cost, int costPitch,
                                    int gw, int gh, int grid, int w, int h,
                                    float* __restrict__ flow, float* __restrict__ costOut)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    double fx = ((double)x + 0.5) / (double)grid - 0.5;
    double fy = ((double)y + 0.5) / (double)grid - 0.5;
    if (fx < 0.0) fx = 0.0;
    if (fy < 0.0) fy = 0.0;
    int x0 = (int)fx, y0 = (int)fy;
    if (x0 > gw - 1) x0 = gw - 1;
    if (y0 > gh - 1) y0 = gh - 1;
    const double lx = fx - (double)x0 > 1.0 ? 1.0 : fx - (double)x0;
    const double ly = fy - (double)y0 > 1.0 ? 1.0 : fy - (double)y0;
    const int x1 = x0 < gw - 1 ? x0 + 1 : x0, y1 = y0 < gh - 1 ? y0 + 1 : y0;
    const short* r0 = (const short*)((const char*)vec + (size_t)y0 * vecPitch);
    const short* r1 = (const short*)((const char*)vec + (size_t)y1 * vecPitch);
    const size_t plane = (size_t)w * h, o = (size_t)y * w + x;
    for (int c = 0; c < 2; c++)
    {
        const double a = (1.0 - lx) * (double)r0[2 * x0 + c] + lx * (double)r0[2 * x1 + c];
        const double b = (1.0 - lx) * (double)r1[2 * x0 + c] + lx * (double)r1[2 * x1 + c];
        flow[(size_t)c * plane + o] = (float)(((1.0 - ly) * a + ly * b) * (1.0 / 32.0));
    }
    const unsigned char* c0 = cost + (size_t)y0 * costPitch;
    const unsigned char* c1 = cost + (size_t)y1 * costPitch;
    const double a = (1.0 - lx) * (double)c0[x0] + lx * (double)c0[x1];
    const double b = (1.0 - lx) * (double)c1[x0] + lx * (double)c1[x1];
    costOut[o] = (float)((1.0 - ly) * a + ly * b);
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
    const double ux = (double)fA[o], uy = (double)fA[plane + o];
    const double px = (double)x + ux, py = (double)y + uy;
    const double dx = ux + nvLerp2(fB, w, h, px, py);
    const double dy = uy + nvLerp2(fB + plane, w, h, px, py);
    // clamped at -16: a landing whose every candidate is bad keeps a weight (exp(-16) * NV_FIX
    // ~ 1.2e5 units) instead of vanishing; beyond a gap of 16 the softmax is winner-take-all anyway
    const double z = -(double)a * (double)costA[o] - (double)b * sqrt(dx * dx + dy * dy);
    Z[o] = (float)(z < -16.0 ? -16.0 : z);
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
    const double fx = (double)x + (double)s * (double)flow[o];
    const double fy = (double)y + (double)s * (double)flow[plane + o];
    if (!isfinite(fx) || !isfinite(fy)) return;
    const float e = expf(__fadd_rn(Z[o], bias));
    const int nwX = (int)floor(fx), nwY = (int)floor(fy);
    const double dx = fx - (double)nwX, dy = fy - (double)nwY;
    const float wNW = (float)((1.0 - dx) * (1.0 - dy)), wNE = (float)(dx * (1.0 - dy));
    const float wSW = (float)((1.0 - dx) * dy), wSE = (float)(dx * dy);
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

// ---- the PULL warp with a confidence fallback (2026-09-21 fix round, variant E2; the product
// since then, the forward splat above stays for the fp64 gate's history) ---------------------
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
// ghosting, never speckle. Gate: harness\nvof\warp\nvof_equiv_e.py against variants.variant_e.
extern "C" __global__ void k_splatVel(const float* __restrict__ flow, const float* __restrict__ Z,
                                      float s, float bias, float sign, int covPlane, int w, int h,
                                      long long* __restrict__ acc)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t plane = (size_t)w * h, o = (size_t)y * w + x;
    const double fx = (double)x + (double)s * (double)flow[o];
    const double fy = (double)y + (double)s * (double)flow[plane + o];
    if (!isfinite(fx) || !isfinite(fy)) return;
    const double e = exp((double)Z[o] + (double)bias);
    const double vx = (double)sign * (double)flow[o], vy = (double)sign * (double)flow[plane + o];
    const int nwX = (int)floor(fx), nwY = (int)floor(fy);
    const double dx = fx - (double)nwX, dy = fy - (double)nwY;
    const double wt[4] = { (1.0 - dx) * (1.0 - dy), dx * (1.0 - dy), (1.0 - dx) * dy, dx * dy };
    const int ox[4] = { 0, 1, 0, 1 }, oy[4] = { 0, 0, 1, 1 };
    for (int k = 0; k < 4; k++)
    {
        const int tx = nwX + ox[k], ty = nwY + oy[k];
        if (tx < 0 || tx >= w || ty < 0 || ty >= h) continue;
        const size_t t = (size_t)ty * w + tx;
        const double we = e * wt[k];
        atomicAdd((unsigned long long*)acc + t, (unsigned long long)llrint(vx * we * (double)NV_FIX));
        atomicAdd((unsigned long long*)acc + plane + t, (unsigned long long)llrint(vy * we * (double)NV_FIX));
        atomicAdd((unsigned long long*)acc + 2 * plane + t, (unsigned long long)llrint(we * (double)NV_FIX));
        atomicAdd((unsigned long long*)acc + (size_t)covPlane * plane + t, (unsigned long long)llrint(wt[k] * (double)NV_FIX));
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
    const double inv = 1.0 / (double)NV_FIX;
    const double vxe = (double)acc[o] * inv, vye = (double)acc[plane + o] * inv;
    const double e = (double)acc[2 * plane + o] * inv;
    const double c0 = (double)acc[3 * plane + o] * inv, c1 = (double)acc[4 * plane + o] * inv;
    // the velocity only where the weight sum is resolvable: a landing whose weight rounded to 0
    // in the fixed point while its velocity product did not would divide to infinity (and the
    // push-pull turns inf * 0 into NaN); a real landing weighs at least ~5e-8 (Z clamp -16, time
    // weight >= 0.25 at the grids used), so 1e-10 drops only corner crumbs and makes them holes
    const bool okE = e > 1e-10;
    double conf = okE ? c0 + c1 : 0.0;
    conf = conf < 0.0 ? 0.0 : (conf > 1.0 ? 1.0 : conf);
    n0[o] = okE ? (float)(vxe / e * conf) : 0.0f;
    n0[plane + o] = okE ? (float)(vye / e * conf) : 0.0f;
    d0[o] = (float)conf;
    vis[o] = (float)(c0 < 0.0 ? 0.0 : (c0 > 1.0 ? 1.0 : c0));
    vis[plane + o] = (float)(c1 < 0.0 ? 0.0 : (c1 > 1.0 ? 1.0 : c1));
    double den = c0 * (1.0 - (double)t) + c1 * (double)t;
    den = den > 1e-12 ? den : 1e-12;
    double q = e / den;
    q = q > 1e-30 ? q : 1e-30;
    double m = (log(q) - (double)lo) / ((double)hi - (double)lo);
    m = m < 0.0 ? 0.0 : (m > 1.0 ? 1.0 : m);
    if (c0 + c1 < 0.05) m = 0.0;
    mraw[o] = (float)m;
}

// push-pull, down: level (w, h) -> (w2, h2) = ceil halves, 2x2 sums (zero outside the level)
extern "C" __global__ void k_ppDown(const float* __restrict__ n, const float* __restrict__ d, int w, int h,
                                    float* __restrict__ n2, float* __restrict__ d2, int w2, int h2)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w2 || y >= h2) return;
    const size_t plane = (size_t)w * h, plane2 = (size_t)w2 * h2, o2 = (size_t)y * w2 + x;
    double a = 0.0, b = 0.0, c = 0.0;
    for (int dy = 0; dy < 2; dy++)
        for (int dx = 0; dx < 2; dx++)
        {
            const int sx = 2 * x + dx, sy = 2 * y + dy;
            if (sx >= w || sy >= h) continue;
            const size_t o = (size_t)sy * w + sx;
            a += (double)n[o]; b += (double)n[plane + o]; c += (double)d[o];
        }
    n2[o2] = (float)a; n2[plane2 + o2] = (float)b; d2[o2] = (float)c;
}

// push-pull, the coarsest level: out = n / max(d, 1e-12)
extern "C" __global__ void k_ppTop(const float* __restrict__ n, const float* __restrict__ d, int w, int h,
                                   float* __restrict__ out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t plane = (size_t)w * h, o = (size_t)y * w + x;
    const double dd = (double)d[o] > 1e-12 ? (double)d[o] : 1e-12;
    out[o] = (float)((double)n[o] / dd);
    out[plane + o] = (float)((double)n[plane + o] / dd);
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
    const double dv = (double)d[o];
    const double dd = dv > 1e-12 ? dv : 1e-12;
    const double a = dv < 0.0 ? 0.0 : (dv > 1.0 ? 1.0 : dv);
    out[o] = (float)((double)n[o] / dd * a + (double)coarse[oc] * (1.0 - a));
    out[plane + o] = (float)((double)n[plane + o] / dd * a + (double)coarse[plane2 + oc] * (1.0 - a));
}

// Gaussian blur, one axis (dir 0 = x, 1 = y), radius r = int(4 sigma + 0.5), reflect edges
// (scipy's mode 'reflect': d c b a | a b c d), weights exp(-x^2 / (2 sigma^2)) normalised
extern "C" __global__ void k_blur1(const float* __restrict__ src, int w, int h, float sigma, int dir,
                                   float* __restrict__ dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const int r = (int)(4.0 * (double)sigma + 0.5);
    const int n = dir ? h : w;
    const int c = dir ? y : x;
    double acc = 0.0, ws = 0.0;
    for (int k = -r; k <= r; k++)
    {
        int i = c + k;
        // reflect about the edge sample boundary, repeated for kernels wider than the axis
        while (i < 0 || i >= n) i = i < 0 ? -i - 1 : 2 * n - i - 1;
        const double wk = exp(-0.5 * (double)k * (double)k / ((double)sigma * (double)sigma));
        acc += wk * (double)(dir ? src[(size_t)i * w + x] : src[(size_t)y * w + i]);
        ws += wk;
    }
    dst[(size_t)y * w + x] = (float)(acc / ws);
}

__device__ __forceinline__ double nvSample(const float* __restrict__ p, int rowStride, int w, int h,
                                           double fx, double fy)
{
    // bilinear at (fx, fy) with the point clamped into the frame (map_coordinates mode nearest);
    // a non-finite point samples the origin instead of indexing out of bounds
    if (!isfinite(fx) || !isfinite(fy)) { fx = 0.0; fy = 0.0; }
    if (fx < 0.0) fx = 0.0;
    if (fy < 0.0) fy = 0.0;
    if (fx > (double)(w - 1)) fx = (double)(w - 1);
    if (fy > (double)(h - 1)) fy = (double)(h - 1);
    const int x0 = (int)fx, y0 = (int)fy;
    const int x1 = x0 < w - 1 ? x0 + 1 : x0, y1 = y0 < h - 1 ? y0 + 1 : y0;
    const double lx = fx - (double)x0, ly = fy - (double)y0;
    const double a = (1.0 - lx) * (double)p[(size_t)y0 * rowStride + x0] + lx * (double)p[(size_t)y0 * rowStride + x1];
    const double b = (1.0 - lx) * (double)p[(size_t)y1 * rowStride + x0] + lx * (double)p[(size_t)y1 * rowStride + x1];
    return (1.0 - ly) * a + ly * b;
}

// the tween: V (2 planes, filled), vis (2 planes), m (the blurred mask), both frames at
// (planeStride, rowStride); out = m * warp + (1 - m) * blend into dst at the same strides
extern "C" __global__ void k_nvofCompose(const float* __restrict__ i0, const float* __restrict__ i1,
                                         int planeStride, int rowStride, const float* __restrict__ V,
                                         const float* __restrict__ vis, const float* __restrict__ m,
                                         int w, int h, float t, float* __restrict__ dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t plane = (size_t)w * h, o = (size_t)y * w + x, io = (size_t)y * rowStride + x;
    const double td = (double)t;
    const double vx = (double)V[o], vy = (double)V[plane + o];
    const double w0 = (1.0 - td) * ((double)vis[o] + 1e-3), w1 = td * ((double)vis[plane + o] + 1e-3);
    const double mm = (double)m[o];
    for (int c = 0; c < 3; c++)
    {
        const float* p0 = i0 + (size_t)c * planeStride;
        const float* p1 = i1 + (size_t)c * planeStride;
        const double c0 = nvSample(p0, rowStride, w, h, (double)x - td * vx, (double)y - td * vy);
        const double c1 = nvSample(p1, rowStride, w, h, (double)x + (1.0 - td) * vx, (double)y + (1.0 - td) * vy);
        const double warp = (w0 * c0 + w1 * c1) / (w0 + w1);
        const double blend = (1.0 - td) * (double)p0[io] + td * (double)p1[io];
        dst[(size_t)c * planeStride + io] = (float)(mm * warp + (1.0 - mm) * blend);
    }
}

// ---- native DRBA glue (priority 21 (b) step 2, 2026-09-21) ---------------------------------
// The two python pieces around the IFNet that DRBA adds on live (live_server.RifeDrba):
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
// splat, <= 2 for the DRM), so 2^32 leaves 1e5 overlapping sources of headroom at 8K. Positions,
// corner weights and every normalisation are DOUBLE, rounded once to fp32 at the store
// ([[gate-kernel-port-against-fp64]]). Planes are contiguous (w * h each), fp32.
// Gate: harness\drba\drba_equiv.py (fp64 references on real block0 flows).

#define DRBA_FIX 4294967296.0

__device__ __forceinline__ void drbaSplat4(unsigned long long* p, size_t plane, int nv,
                                           const double* v, double fx, double fy, int w, int h)
{
    if (!isfinite(fx) || !isfinite(fy)) return;
    const int nwX = (int)floor(fx), nwY = (int)floor(fy);
    const double dx = fx - (double)nwX, dy = fy - (double)nwY;
    const double wt[4] = {(1.0 - dx) * (1.0 - dy), dx * (1.0 - dy), (1.0 - dx) * dy, dx * dy};
    const int cx[4] = {nwX, nwX + 1, nwX, nwX + 1}, cy[4] = {nwY, nwY, nwY + 1, nwY + 1};
    for (int k = 0; k < 4; k++)
    {
        if (cx[k] < 0 || cx[k] >= w || cy[k] < 0 || cy[k] >= h) continue;
        const size_t o = (size_t)cy[k] * w + cx[k];
        for (int c = 0; c < nv; c++)
            atomicAdd(p + (size_t)c * plane + o,
                      (unsigned long long)(long long)llrint(v[c] * wt[k] * DRBA_FIX));
        atomicAdd(p + (size_t)nv * plane + o, (unsigned long long)(long long)llrint(wt[k] * DRBA_FIX));
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
        const double v[2] = {(double)flow[(2 * j) * plane + o], (double)flow[(2 * j + 1) * plane + o]};
        drbaSplat4((unsigned long long*)(acc + (size_t)(3 * j) * plane), plane, 2, v,
                   (double)x + v[0], (double)y + v[1], w, h);
    }
}

// acc (6 planes) -> out (4 planes): flow05 * 2 (x, y), flow15 * 2 (x, y)
extern "C" __global__ void k_drbaFlowNorm(const long long* __restrict__ acc, int w, int h,
                                          float* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t plane = (size_t)w * h;
    if (i >= (int)plane) return;
    const double fill = 2.0 * (double)(w > h ? w : h);
    for (int j = 0; j < 2; j++)
    {
        const double wsum = (double)acc[(size_t)(3 * j + 2) * plane + i] / DRBA_FIX;
        const bool hole = wsum / (wsum + 1e-7) < 0.999;
        for (int c = 0; c < 2; c++)
            out[(size_t)(2 * j + c) * plane + i] = hole ? (float)fill
                : (float)(-2.0 * ((double)acc[(size_t)(3 * j + c) * plane + i] / DRBA_FIX) / (wsum + 1e-7));
    }
}

// the unaligned DRM value at one pixel: side < 0 -> drm12 * tt * 2, side > 0 -> drm10 * tt * 2
__device__ __forceinline__ double drbaDrmUn(const float* f10, const float* f12, size_t plane,
                                            size_t o, int side, double tt)
{
    const double u0 = f10[o], v0 = f10[plane + o], u2 = f12[o], v2 = f12[plane + o];
    const double d10 = sqrt(u0 * u0 + v0 * v0) + 1e-4, d12 = sqrt(u2 * u2 + v2 * v2) + 1e-4;
    return (side < 0 ? d12 : d10) / (d10 + d12) * tt * 2.0;
}

// f10, f12 (2 planes each, calc_flow's flow05 * 2) -> acc (2 int64 planes, zeroed): drm, weight
extern "C" __global__ void k_drbaDrmSplat(const float* __restrict__ f10, const float* __restrict__ f12,
                                          int side, float tt, int w, int h, long long* __restrict__ acc)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t plane = (size_t)w * h, o = (size_t)y * w + x;
    const double d = drbaDrmUn(f10, f12, plane, o, side, (double)tt);
    const float* fa = side < 0 ? f10 : f12;
    const double v[1] = {d};
    drbaSplat4((unsigned long long*)acc, plane, 1, v, (double)x + (double)fa[o] * d,
               (double)y + (double)fa[plane + o] * d, w, h);
}

// acc (2 planes) -> the (1, H, W) timestep map for the IFNet
extern "C" __global__ void k_drbaDrmNorm(const long long* __restrict__ acc, const float* __restrict__ f10,
                                         const float* __restrict__ f12, int side, float tt, int w, int h,
                                         float* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t plane = (size_t)w * h;
    if (i >= (int)plane) return;
    const double wsum = (double)acc[plane + i] / DRBA_FIX;
    out[i] = wsum / (wsum + 1e-7) < 0.999 ? (float)drbaDrmUn(f10, f12, plane, i, side, (double)tt)
                                          : (float)(((double)acc[i] / DRBA_FIX) / (wsum + 1e-7));
}

}

)CUDASRC";

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
    bool ok() const { return create && compile && cubinSize && cubin && destroy; }
};

static std::wstring g_nativeRuntimeDir;   // engine\gpu_runtime: the CUDA 13 + TensorRT-RTX DLLs
static std::wstring g_nativeEngineDir;    // ...\engine itself, the parent of the folders above

// ---- WO-23: the RTX Video SDK CUDA bridge (engine\rtxvideo\rtxvideo_cuda.dll) -------------
// The TrueHDR bridge is NOT an NGX D3D11 bridge: it is a plain C ABI over CUDA device
// pointers, the same DLL and the same entry points rtxvideo.py drives. The NGX feature DLLs
// (nvngx_truehdr.dll, nvngx_vsr.dll) resolve relative to the LOADING module's directory, so
// the bridge is loaded by full path out of its own folder and nothing is copied or split.
struct RtxRect { uint32_t left, top, right, bottom; };
struct RtxThdrSetting { uint32_t Contrast, Saturation, MiddleGray, MaxLuminance; };
struct RtxVsrSetting { uint32_t QualityLevel; };   // 0 bicubic .. 4 Ultra, python always 4
typedef void (*PFN_rtxvSetModelPath)(const wchar_t*);
typedef unsigned int (*PFN_rtxCreate)(void*, void*, int, unsigned int, unsigned int);
typedef unsigned int (*PFN_rtxEvalThdr)(void*, void*, RtxRect, RtxRect, RtxThdrSetting*);
typedef unsigned int (*PFN_rtxEvalVsr)(void*, void*, RtxRect, RtxRect, RtxVsrSetting*);
typedef void (*PFN_rtxShutdown)();

struct RtxBridge
{
    HMODULE mod = nullptr;
    PFN_rtxvSetModelPath setModelPath = nullptr;
    PFN_rtxCreate create = nullptr;
    PFN_rtxEvalThdr evalThdr = nullptr;
    PFN_rtxEvalVsr evalVsr = nullptr;   // live RTX VSR (2026-09-15): 8-bit BGRA in and out
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
    if (g_rtxb.mod) return true;
    std::wstring dir;
    wchar_t ov[MAX_PATH]{};
    if (GetEnvironmentVariableW(L"SMV_RTXVIDEO_DIR", ov, MAX_PATH) && ov[0]) dir = ov;
    else if (!g_nativeEngineDir.empty()) dir = g_nativeEngineDir + L"\\rtxvideo";
    if (dir.empty()) { LOG("native: cannot locate the rtxvideo folder\n"); return false; }
    const std::wstring dll = dir + L"\\rtxvideo_cuda.dll";
    HMODULE m = LoadLibraryExW(dll.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!m)
    {
        LOG("native: %s not loadable (err %lu)\n",
            wideToUtf8(dll).c_str(), GetLastError());
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
        g_rtxb.setModelPath = nullptr; g_rtxb.create = nullptr;
        g_rtxb.evalThdr = nullptr; g_rtxb.evalVsr = nullptr; g_rtxb.shutdown = nullptr;
        return false;
    }
    g_rtxb.mod = m;
    g_rtxb.dir = dir;
    g_rtxb.setModelPath(dir.c_str());   // NGX APP_PATH, must precede create
    LOG("native: RTX Video bridge %s\n", wideToUtf8(dll).c_str());
    return true;
}

// priority 24 step 2c: offline TrueHDR's light statistics, python's RTXVideo _cll / _fall / _l1
// / _hp (rtxvideo.py _pack_out, _measure_light, _accum_l1, _accum_hp), accumulated per output
// frame from k_thdrOut's block (hist[1024] u32, misc[4] u32, vSum double) and written as one
// JSON file render.py reads at the finalize: maxcll / maxfall always, the Dolby Vision L1
// triples and the HDR10+ records when asked for (--hdr-dv / --hdr-hp).
static const size_t kThdrStatsBytes = 1024 * 4 + 4 * 4 + 8;
struct ThdrL1 { int v[3]; };
struct ThdrHp { int avg; int maxscl[3]; int dist[9]; };
struct ThdrAcc
{
    float lut[1024]{};          // _pq_lut (k_pqLut, the kernel's own PQ EOTF)
    int brightCode = 0;         // _hp_bright_code: the first code brighter than 100 nits
    double cll = 0.0, fall = 0.0;
    bool raw = false;           // --hdr-color raw: the stats read nits, not linear
    bool wantL1 = false, wantHp = false;
    std::vector<ThdrL1> l1;
    std::vector<ThdrHp> hp;
    // --hdr-frames (priority 24 step 6d): one line per output frame, appended and flushed as the
    // record lands, "<frame MaxCLL> <frame MaxFALL>[ L <l1 x3>][ H <avg> <maxscl x3> <dist x9>]":
    // a killed render keeps every record of the frames it wrote, and a resumed render reads the
    // banked prefix back from it instead of decoding the banked video (python's _rebuild_hdr_stats)
    FILE* lines = nullptr;
    double lastCll = 0.0, lastFall = 0.0;
    bool haveLast = false;
    ThdrAcc() = default;
    ThdrAcc(const ThdrAcc&) = delete;
    ThdrAcc& operator=(const ThdrAcc&) = delete;
    ~ThdrAcc() { if (lines) fclose(lines); }

    void writeLine(double c, double f)
    {
        lastCll = c; lastFall = f; haveLast = true;
        if (!lines) return;
        fprintf(lines, "%.17g %.17g", c, f);
        if (wantL1 && !l1.empty()) fprintf(lines, " L %d %d %d", l1.back().v[0], l1.back().v[1], l1.back().v[2]);
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
        double vSum = 0.0;
        memcpy(&vSum, s + 1024 * 4 + 16, 8);
        float vMax = 0.0f;
        memcpy(&vMax, &misc[3], 4);
        // _pack_out: float(mx.max()) * 10000 and float(mx.mean()) * 10000 (the mean is fp32 in
        // torch); _measure_light (raw) reads nits already
        const double sc = raw ? 1.0 : 10000.0;
        const double fc = (double)vMax * sc;
        if (fc > cll) cll = fc;
        const double fa = (double)(float)(vSum / (double)n) * sc;
        if (fa > fall) fall = fa;
        if (wantL1)
        {
            // _accum_l1: [amin, mean, amax] of the maxRGB code, * 4095 / 1023, round, clamp
            int mn = -1, mx = 0;
            double sum = 0.0;
            for (int c = 0; c < 1024; c++)
                if (hist[c]) { if (mn < 0) mn = c; mx = c; sum += (double)hist[c] * c; }
            if (mn < 0) mn = 0;
            const float k = (float)(4095.0 / 1023.0);
            const float st[3] = { (float)mn * k, (float)(sum / (double)n) * k, (float)mx * k };
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
            static const float qs[8] = { 0.01f, 0.9998f, 0.25f, 0.50f, 0.75f, 0.90f, 0.95f, 0.99f };
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
                while (c < 1024 && cum[c] < t) c++;
                if (c > 1023) c = 1023;
                pv[i] = (int)rintf(lut[c] * 100000.0f);
            }
            ThdrHp r{};
            for (int c = 0; c < 3; c++)
                r.maxscl[c] = (int)rintf(lut[misc[c] > 1023u ? 1023u : misc[c]] * 100000.0f);
            r.avg = (int)nearbyint((double)(float)lsum / (double)n * 100000.0);
            const double bright = (double)n - (brightCode > 0 ? (double)cum[brightCode - 1] : 0.0);
            const int d[9] = { pv[0], pv[1], (int)nearbyint(1000.0 * bright / (double)n),
                               pv[2], pv[3], pv[4], pv[5], pv[6], pv[7] };
            memcpy(r.dist, d, sizeof(d));
            hp.push_back(r);
        }
        writeLine(fc, fa);
    }

    // a held slot that carries the previous frame's finished bytes (DLSS 5 on, step 2d): the
    // same picture, so the same record (MaxCLL / MaxFALL are unchanged by a repeat)
    void repeat()
    {
        if (wantL1 && !l1.empty()) l1.push_back(l1.back());
        if (wantHp && !hp.empty()) hp.push_back(hp.back());
        if (haveLast) writeLine(lastCll, lastFall);
    }

    bool writeJson(const std::wstring& path) const
    {
        FILE* f = nullptr;
        if (_wfopen_s(&f, path.c_str(), L"wb") || !f) return false;
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
                        i ? ", " : "", r.avg, r.maxscl[0], r.maxscl[1], r.maxscl[2],
                        r.dist[0], r.dist[1], r.dist[2], r.dist[3], r.dist[4], r.dist[5], r.dist[6], r.dist[7], r.dist[8]);
            }
            fprintf(f, "]");
        }
        fprintf(f, "}\n");
        const bool ok = !ferror(f);
        fclose(f);
        return ok;
    }
};

static bool g_onnxParserOk = false;   // tensorrt_onnxparser_rtx_1_6.dll loaded (nativeLoadDlls)

static bool nativeLoadDlls(const std::wstring& scriptPath)
{
    static int state = 0;   // 0 untried, 1 ok, -1 failed
    if (state) return state > 0;
    // The shipped layout is engine\live\smv-live.exe next to engine\gpu_runtime, so the runtime
    // is resolved RELATIVE TO THIS EXE and never from a build-time absolute path. A dev exe that
    // lives outside the app tree (bin\smv-live.exe with an explicit --script) falls back to the
    // directory of the --script path it was pointed at. The chosen folder is always logged.
    auto dirOf = [](const std::wstring& p) -> std::wstring
    {
        const size_t sl = p.find_last_of(L"\\/");
        return sl == std::wstring::npos ? std::wstring() : p.substr(0, sl);
    };
    std::wstring engineDir;
    {
        wchar_t exePath[MAX_PATH]{};
        GetModuleFileNameW(nullptr, exePath, MAX_PATH);
        const std::wstring liveDir = dirOf(exePath);           // ...\engine\live
        const std::wstring cand = dirOf(liveDir);              // ...\engine
        if (!cand.empty() && GetFileAttributesW((cand + L"\\gpu_runtime").c_str()) != INVALID_FILE_ATTRIBUTES)
            engineDir = cand;
    }
    if (engineDir.empty()) engineDir = dirOf(scriptPath);       // dev exe outside the app tree
    if (engineDir.empty()) { LOG("native: cannot locate the app runtime folder\n"); state = -1; return false; }
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
        if (hf != INVALID_HANDLE_VALUE) { builtins = fd.cFileName; FindClose(hf); }
    }
    struct { const wchar_t* dir; const wchar_t* dll; bool required; } want[] = {
        { cudaBin.c_str(), L"cudart64_13.dll", true },
        { cudaBin.c_str(), builtins.empty() ? L"" : builtins.c_str(), false },
        { cudaBin.c_str(), L"nvrtc64_130_0.dll", true },
        { trtBin.c_str(),  L"tensorrt_rtx_1_6.dll", true },
    };
    for (auto& wdl : want)
    {
        if (!wdl.dll[0]) continue;
        std::wstring full = std::wstring(wdl.dir) + L"\\" + wdl.dll;
        if (!LoadLibraryExW(full.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH) && wdl.required)
        {
            LOG("native: %s not loadable (err %lu), the native host cannot start\n",
                wideToUtf8(full).c_str(), GetLastError());
            state = -1;
            return false;
        }
    }
    // the ONNX parser (ONNX-in-exe 3c-2): optional, without it a cold size cannot build
    g_onnxParserOk = LoadLibraryExW((trtBin + L"\\tensorrt_onnxparser_rtx_1_6.dll").c_str(), nullptr,
                                    LOAD_WITH_ALTERED_SEARCH_PATH) != nullptr;
    if (!g_onnxParserOk) LOG("native: ONNX parser not loadable, a window size without warm engines cannot start\n");
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
        if (s <= Severity::kWARNING) LOG("[trt-native] %s\n", msg);
    }
};
static NativeTrtLogger g_nativeTrtLogger;

#define NCHK(call, what)                                                                   \
    do {                                                                                   \
        const cudaError_t _e = (call);                                                     \
        if (_e != cudaSuccess)                                                             \
        { LOG("native: %s failed (cuda %d)\n", what, (int)_e); return false; }             \
    } while (0)

struct NativeRife
{
    // ---- handoff facts, straight off live_server.py's --native-handoff lines
    std::string ifnetPath, encodePath, jitPath, ejitPath;
    int ph = 0, pw = 0, w = 0, h = 0, cw = 0, ch = 0;
    int dw = 0, dh = 0, x0 = 0, y0 = 0;
    int batchMax = 1;
    bool identity = false;
    bool hdr = false;           // WO-15 phase 2: FP16 scRGB capture in, PQ R10A2 ring out
    int dev = 0;                // the CUDA device of the capture adapter (bound per thread)
    // WO-23: live RTX TrueHDR through the CUDA bridge, once per real frame at capture size
    bool rtxHdr = false;        // the session asked for it and the bridge loaded
    bool rtxFailed = false;     // a mid-run eval failed: faithful PQ for the rest of the run
    RtxThdrSetting thdr{};      // Contrast, Saturation, MiddleGray 50, MaxLuminance 1000
    float sdrScale = 3.0f;      // SDR reference white as an scRGB scale (nits / 80)
    int rtxMode = 0;            // 0 vivid, 1 rtx, 2 raw
    float rtxVib = 0.0f, rtxSb = 0.0f;
    double thdrMs = 0.0, thdrMaxMs = 0.0;   // host-side eval cost, reported at teardown
    uint64_t thdrN = 0;
    // live Sharpen and RTX VSR (2026-09-15), the app's --sharpen / --rtx-vsr on this route
    float sharpen = 0.0f;       // RCAS strength 0..1 at the presented size (0 = off)
    bool vsrWant = false;       // --rtx-vsr on an SDR session with the bridge and its DLL present
    bool vsr = false;           // ...and the fit enlarges in both axes (decided after the handoff)
    bool vsrFailed = false;     // a mid-run eval failed: bicubic for the rest of the run
    RtxVsrSetting vsrSet{ 4 };
    double vsrMs = 0.0, vsrMaxMs = 0.0;
    uint64_t vsrN = 0;
    // the downscaling fit (2026-09-15): python's antialiased bicubic pair into the staging frame
    bool fitAa = false;         // the fit to (dw, dh) shrinks its source height (python's antialias rule)
    // Upscale to (2026-09-15, full native migration item 2): the app's --upscale H = the
    // INTERNAL render size (uw, uh), derived like _Fit.__init__ from the exe's own flag. The
    // model frame goes there first (RTX VSR when it enlarges, else bicubic, antialiased when
    // it shrinks), then the fit to (dw, dh) like any frame. 0 = off (also when it equals the
    // fit rect: one resize, not two, the python rule).
    int uw = 0, uh = 0;
    bool upAa = false;          // the first resize shrinks the height (uh < h)
    // live Restore (2026-09-15, full native migration item 3): the python handoff builds the
    // Real-ESRGAN TensorRT engine into the shared cache and hands its path over like the
    // IFNet's; the host runs it on every presented frame before the upscale, then folds the 4x
    // output to the first resize target (_Fit._load_restore's restore_target rule: back to the
    // model size when RTX VSR follows, else straight to the internal render size or the fit
    // rect). A mid-run failure drops the pass for the rest of the session with one line.
    std::string restorePath, rjitPath;
    bool restore = false;       // the handoff named an engine (the session asked for it)
    // no-engine mode (2026-09-15, full native migration item 4): the effects-only route (the
    // app's `echo` backend, no model ticked) runs here too. The handoff answers with the
    // geometry only (`engine=none`, plus the restore engine when asked for): no IFNet, no
    // encode, no tween; every group stores its one real frame through the same effects
    // chain, exactly live_server.py's Identity.process_shm. The cubin cache folder comes
    // from `NATIVE-PATH cache=` since no jit path names it.
    bool noEngine = false;
    std::string cachePath;
    // GMFSS (full native migration item 5, sub-steps 5a to 5c, 2026-09-15): the handoff answers
    // `engine=gmfss` with the five engines of live_server.Gmfss (feat_ext at the padded size,
    // the fused bidir GMFlow, metricnet, the GMFSS IFNet and fusionnet at the half) and their
    // jit caches; this host loads, warms and keeps the set resident (one model set at a time)
    // and since sub-step 5c runs the chain itself (nativeGmfssPair / nativeGmfssTween, the
    // glue kernels of 5b), so gmfss is a native backend like rife. (Sub-step 5d's Flow scale
    // below 100, gmflow at a /32 flow grid, was dropped with the control 2026-09-25.)
    bool gmfss = false;
    int hh = 0, hw = 0;                          // the half frame (the fusion grid)
    std::string gmPath[5], gmJit[5];
    nvinfer1::ICudaEngine* engGm[5] = {};
    nvinfer1::IRuntimeConfig* cfgGm[5] = {};
    nvinfer1::IExecutionContext* ctxGm[5] = {};
    // the chain's contract, read off the engines in nativeGmfssSetup (never from the handoff
    // line, the 5a rule), and the planar NCHW buffers it sizes from that. Levels: the half
    // (hh, hw), the quarter and the eighth, exactly GMFSS_infer_u's feature pyramid.
    int gmC[3] = {};                 // feat_ext channels at the half / quarter / eighth
    bool gmFeatHalf = true, gmMetricHalf = true, gmOutHalf = true;
    int gmCur = 0;                   // which dGmFeat set holds the CURRENT frame's features
    void* dGmFeat[2][3] = {};        // the two frames' feature sets (cur is the next pair's prev)
    float* dGmHalf = nullptr;        // (6, hh, hw): img0's half in planes 0..2, img1's in 3..5
    float* dGmFlow = nullptr;        // the gmflow output (2, 2, hh, hw) = flow01 then flow10
    void* dGmMetric = nullptr;       // (2, hh, hw) in the engine's dtype: m0 then m1
    float* dGmFlowP[2] = {};         // the flow pyramids at the quarter / eighth, pre-scaled
    float* dGmMetP[2] = {};          // the metric pyramids at the quarter / eighth (fp32)
    long long* dGmAcc = nullptr;     // the splat accumulator, max(C) + 1 planes of the half
    float* dGmFa = nullptr;          // fusionnet a (9, half): I1t, merged (ifnet writes it), I2t
    float* dGmFb = nullptr;          // fusionnet b (2 * gmC[0], half)
    float* dGmFc = nullptr;          // fusionnet c (2 * gmC[1], quarter)
    float* dGmFd = nullptr;          // fusionnet d (2 * gmC[2], eighth)
    float* dGmT = nullptr;           // the timestep, one device float (ifnet's 1x1x1x1 input)
    void* dGmOut = nullptr;          // the fusionnet output (3, ph, pw) in the engine's dtype
    float* dGmF = nullptr;           // that output clamped to fp32: what storeSlot consumes
    // SMV_LIVE_GMFSS_PROF=1: per-phase CUDA-event breakdown of the chain, printed every 32
    // groups (the native answer to the python route's [timing] line; nine spans, the tween ones
    // measured on the group's FIRST tween). Sub-step 5e split the old pair span into half |
    // flow | metric | pyr and added the CPU (QPC) enqueue spans beside the GPU ones: the pair
    // block, the gmflow enqueue inside it, and every tween. The exe runs a group ahead of the
    // GPU, so those CPU numbers are launch cost, not the GPU pace.
    bool gmProf = false, gmProfTween = false;
    cudaEvent_t gmEv[11] = {};
    double gmAcc[9] = {};
    uint32_t gmProfN = 0;
    double gmCpuPair = 0.0, gmCpuFlow = 0.0, gmCpuTween = 0.0;
    uint32_t gmCpuPairN = 0, gmCpuTweenN = 0;
    // NVIDIA Smooth Motion (fruc, 2026-09-21, memory priority 21): the handoff answers
    // `engine=fruc` with the /64 padded geometry and `NATIVE-PATH fruc=` (the folder holding the
    // shipped nvoffruc_bridge.dll beside the user-installed NvOFFRUC.dll + cudart64_110.dll);
    // this host drives that bridge's flat C API exactly as live_server.py's Fruc class does
    // through ctypes: the model planes packed to BGRA8 at pw x ph (k_packBgra, the VSR kernel),
    // one bridge call per tween, the BGRA8 result unpacked (k_unpackBgra). The planes go in as
    // stored, so FRUC sees true BGRA on live (python's SDR route swaps R and B into it).
    bool fruc = false;
    bool dlssg = false;              // offline DLSS 4.5 (priority 24 step 5): no IFNet, a child server
    std::string frucDir;             // the bridge folder the handoff named
    uint8_t* dFrSurf[3] = {};        // BGRA8 pw x ph: the packed frames, rotated per group
    uint8_t* dFrOutB = nullptr;      // BGRA8 pw x ph: the bridge's output
    float* dFrOut = nullptr;         // (3, ph, pw): the tween, the layout storeSlot reads
    int frPrev = -1, frLast = -1;    // surfaces: the previous frame, the last tweened pair's end
    int frA = -1, frB = -1;          // this group's pair
    // the feed-once bridge (nvoffruc_step): the surface whose frame FRUC was fed last (-1 = none,
    // or that surface was repacked since), and whether the pair's next tween is its first
    int frFed = -1;
    bool frFirst = false;
    bool frCreated = false;
    uint64_t frPrimed = 0, frRepeats = 0, frTweens = 0;
    // RIFE with DRBA timing (rifedrba, 2026-09-21, memory priority 21 (b) step 3): the RIFE
    // handoff plus `NATIVE-PATH block0=` (calc_flow's block0 as its own engine) and `engine=drba
    // lag=1`. live_server.RifeDrba natively: a four-frame history of padded frames and their
    // encodes, windows centred on a frame id (two kept, chained: a window's left flow is the
    // previous window's right one, reversed), the lag-1 group (every slot shows time (k-2) + f,
    // the real slot is frame k-1), one IFNet enqueue per tween with its own DRM timestep map
    // (the step 2 kernels k_drbaFlowSplat / FlowNorm / DrmSplat / DrmNorm, fp64-gated by
    // harness\drba\drba_equiv.py), the head fallback (plain pair RIFE on the lagged pair).
    bool drba = false;
    std::string block0Path, block0Jit;
    nvinfer1::ICudaEngine* engB0 = nullptr;
    nvinfer1::IRuntimeConfig* cfgB0 = nullptr;
    nvinfer1::IExecutionContext* ctxB0 = nullptr;
    uint32_t drFid = 0;              // frames pushed this session; frame id i sits in ring slot i & 3
    float* dDrI[4] = {};             // (3, ph, pw) the padded frames
    float* dDrF[4] = {};             // (16, ph, pw) their encodes, fp32
    float* dDrX[2] = {};             // (6, ph, pw) IFNet x: [0] = [k-1, k-2] (side -1), [1] = [k-2, k-1]
    uint32_t drXFor[2] = {};         // the newest frame id each x was built for (0 = none)
    struct DrWin { uint32_t c; float* f10; float* r; };   // r = (4, ph, pw): flow12 | flow21
    DrWin drWin[2] = {};             // c = 0: empty
    float* dDrFlow = nullptr;        // (4, ph, pw) block0's output
    float* dDrFlowN = nullptr;       // (4, ph, pw) a non-chained left flow's FlowNorm output
    long long* dDrAcc = nullptr;     // (6, ph, pw) the splat accumulator
    uint64_t drTweens = 0, drHeads = 0, drBlock0 = 0;
    // NVIDIA Optical Flow model (2026-09-21, memory priority 20 step 3): the handoff answers
    // `engine=nvof` with the geometry only (like echo, no TensorRT engine at all); this host runs
    // the Optical Flow Accelerator itself through the driver's nvofapi64.dll (grid 4, fast,
    // BOTH directions in one Execute, gray8 of the two model frames at the true w x h) and the
    // glue kernels k_nvofLuma / k_nvofUp / k_nvofMetric (fp64-gated by
    // harness\nvof\warp\nvof_equiv.py; k_splatNvof / k_splatNvofNorm in the kernel text are
    // the forward-splat tween the E2 tween below replaced, kept only because the text is
    // shared byte for byte with that harness).
    bool nvof = false;
    // plane order of the model planes: live capture packs (B, G, R) (BGRA textures), the OFFLINE
    // route's k_packInRaw8 / 16 pack the decoder's rgb24 / rgb48 as (R, G, B); k_nvofLuma needs
    // R, G, B, so it gets the R plane with a negative stride on live and plain planes offline
    // (found 2026-09-21: offline renders ran with red and blue weights swapped until then)
    bool planesRgb = false;
    NvOFHandle ofH = nullptr;
    NvOFGPUBufferHandle ofIn[2] = {}, ofOut[2] = {}, ofCost[2] = {};   // [0] forward, [1] backward
    CUdeviceptr ofInP[2] = {}, ofOutP[2] = {}, ofCostP[2] = {};
    uint32_t ofInPitch = 0, ofOutPitch[2] = {}, ofCostPitch[2] = {};
    int ofGw = 0, ofGh = 0;
    float* dNvFlow[2] = {};          // (2, h, w) px: [0] = F01 (frame 0 -> 1), [1] = F10
    float* dNvCost[2] = {};          // (h, w) the upsampled cost per direction
    float* dNvZ[2] = {};             // (h, w) the splat metric per direction
    long long* dNvAcc = nullptr;     // (5, h, w): RGB * e, e, coverage
    float* dNvOut = nullptr;         // (3, ph, pw): the tween, the layout storeSlot reads
    bool nvProf = false;             // SMV_LIVE_NVOF_PROF=1: per-pair GPU ms
    cudaEvent_t nvEv[2] = {};
    double nvPairMs = 0.0, nvTweenMs = 0.0;
    uint32_t nvPairN = 0, nvTweenN = 0;
    // the E2 tween (2026-09-21 fix round): the pull warp + confidence fallback. The velocity
    // at time t (n / d = the push-pull level 0 inputs), the two visibilities, the raw and the
    // blurred fallback mask, the blur's pass, and the push-pull pyramid levels 1.. (n, d, out)
    float* dNvN0 = nullptr;          // (2, h, w) velocity * confidence
    float* dNvD0 = nullptr;          // (h, w) confidence
    float* dNvVis = nullptr;         // (2, h, w) visibility in frame 0 / frame 1
    float* dNvMraw = nullptr;        // (h, w) the fallback mask before the blur
    float* dNvMask = nullptr;        // (h, w) the blurred mask (1 = warp, 0 = blend)
    float* dNvBlurTmp = nullptr;     // (h, w) the blur's horizontal pass
    float* dNvV = nullptr;           // (2, h, w) the filled velocity (push-pull level 0 out)
    struct NvLevel { int w, h; float* n; float* d; float* out; };
    std::vector<NvLevel> nvPyr;      // levels 1.. down to 1 px on the short side
    bool restHalfIn = true, restHalfOut = true;   // the engine's x / y dtypes (read off it)
    bool restFailed = false;
    int restTw = 0, restTh = 0; // the fold target (decided once VSR is known)
    nvinfer1::ICudaEngine* engRest = nullptr;
    nvinfer1::IRuntimeConfig* cfgRest = nullptr;
    nvinfer1::IExecutionContext* ctxRest = nullptr;
    void* dRestIn = nullptr;      // x: NCHW at the model size, fp16 (or fp32)
    void* dRestOut = nullptr;     // y: NCHW 4x, the engine's dtype
    float* dRestTmp = nullptr;    // the fold's horizontal pass, planar fp32 at (4h) x max target width
    float* dRestF = nullptr;      // the enlarging fold only: y as clamped fp32 planar
    float* dRest = nullptr;       // the fold back to the model size (VSR follows), planar fp32 w x h

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

    // ---- CUDA
    cudaStream_t stream = nullptr;
    cudaExternalMemory_t emCap = nullptr, emOut = nullptr;
    cudaExternalSemaphore_t semCap = nullptr;
    cudaMipmappedArray_t capMip = nullptr;
    cudaArray_t capArr = nullptr;
    uint8_t* dOutRing = nullptr;
    uint8_t* dCap = nullptr;
    float* dTmp = nullptr;
    float* dCapF = nullptr;     // HDR + image scale under 1: planar PQ at capture resolution
    uint8_t* dThdrIn = nullptr;   // WO-23: BGRA8 bridge input, pitch 4*cw
    uint32_t* dThdrOut = nullptr; // WO-23: packed 10:10:10:2 bridge output, B in the low bits
    float* dSrcG = nullptr;       // WO-23: the unquantised sRGB-encoded source, planar fp32
    // priority 24 step 2c: offline TrueHDR at the output size dw x dh (the three buffers above,
    // sized dw x dh there) and its per-frame statistics block, read back one frame late
    uint8_t* dThdrStats = nullptr;   // k_thdrOut's hist / misc / vSum (kThdrStatsBytes)
    uint8_t* hThdrStats = nullptr;   // its pinned copy, valid after the next stream sync
    bool thdrStatsPending = false;
    ThdrAcc* thdrAcc = nullptr;      // the render's accumulators (runOfflineSession owns them)
    uint32_t thdrRepeat = 0;         // held slots that reuse the pending frame's record (DLSS 5)
    // priority 24 step 2d: offline DLSS 5, the NR core on its own D3D12 device (dlssnr.exe's
    // startup + renderFrame), one RGBA16F frame at dw x dh through pinned host staging
    nr::Host* nrHost = nullptr;      // null = DLSS 5 off or unavailable for this render
    bool nrFailed = false;           // an evaluate failed: off for the rest of the render
    bool nrFirst = true;             // the stream's first frame evaluates with Reset
    bool nrResetEvery = false;       // SMV_NR_RESET_EVERY=1: every frame (the equivalence gate's lever)
    uint16_t* dNrIo = nullptr;       // device RGBA16F, both directions
    uint16_t* hNrIn = nullptr;       // pinned host copies handed to renderFrame
    uint16_t* hNrOut = nullptr;
    std::wstring nrDeltaPath;        // --nr-delta PATH: the pass's change map (the preview's mask)
    double nrMs = 0.0, nrMaxMs = 0.0;
    uint64_t nrN = 0;
    float* dPres = nullptr;       // sharpen / fitAa: the fitted frame, planar fp32 at dw x dh
    float* dFitTmp = nullptr;     // fitAa: the horizontal pass, planar fp32 at dw x (uh or h)
    float* dUp = nullptr;         // Upscale to: the internal render frame, planar fp32 at uw x uh
    float* dUpTmp = nullptr;      // upAa: the horizontal pass of the first resize, planar fp32 at uw x h
    uint8_t* dVsrIn = nullptr;    // VSR: tight BGRA8 at the model size w x h
    uint8_t* dVsrOut = nullptr;   // VSR: tight BGRA8 at the presented size dw x dh
    float* dX = nullptr;        // (1,6,ph,pw): prev in planes 0..2, cur in 3..5
    float* dF[2]{};             // f_prev / f_cur, (1,16,ph,pw)
    uint16_t* dEncHalf = nullptr;
    float* dT = nullptr;        // (B,1,ph,pw)
    float* dMerged = nullptr;   // (B,3,ph,pw)
    int fCur = 0;
    cudaEvent_t capEv = nullptr;
    std::vector<cudaEvent_t> slotEv;

    CUmodule cuMod = nullptr;
    CUfunction fPackInDirect = nullptr, fResizeH = nullptr, fResizeV = nullptr,
               fH2f = nullptr, fPackOut = nullptr;
    CUfunction fPackInDirectHdr = nullptr, fPqPlanar = nullptr, fResizeHf = nullptr,
               fPackOutHdr = nullptr;
    CUfunction fSdrEncode = nullptr, fThdrColor = nullptr;   // WO-23
    CUfunction fThdrIn = nullptr, fRcasThdrIn = nullptr, fThdrOut = nullptr, fPqLut = nullptr;   // offline TrueHDR
    CUfunction fPackBgra = nullptr, fUnpackBgra = nullptr, fFitPlanar = nullptr,   // sharpen / VSR
               fRcasOut = nullptr, fRcasOutHdr = nullptr, fRcasOutRaw = nullptr,
               fPackBgraRgb = nullptr, fUnpackBgraRgb = nullptr,   // offline (R, G, B) planes
               fUnpackRgba = nullptr;                              // offline DLSS 4.5 frames
    CUfunction fFitAaH = nullptr, fFitAaV = nullptr;                    // the downscaling fit
    CUfunction fRestIn = nullptr, fRestFoldH = nullptr, fRestFoldV = nullptr,   // live Restore
               fRestToF = nullptr, fClamp01 = nullptr;
    CUfunction fNrIn = nullptr, fNrOut = nullptr;                       // offline DLSS 5
    CUfunction fHalf = nullptr, fPyr = nullptr, fSplatSoft = nullptr,   // live GMFSS glue (5b)
               fSplatNorm = nullptr;
    CUfunction fPackInRaw16 = nullptr, fPackInRaw8 = nullptr,           // WO-32 offline
               fPackOutRaw16 = nullptr, fPackOutRaw8 = nullptr, fExpand8to16 = nullptr;
    CUfunction fPairDiff = nullptr;                                     // identical-pair test
    CUfunction fNvofLuma = nullptr, fNvofUp = nullptr, fNvofMetric = nullptr;   // the nvof model
    CUfunction fSplatVel = nullptr, fVelNorm = nullptr, fPpDown = nullptr,     // its E2 tween
               fPpTop = nullptr, fPpUp = nullptr, fBlur1 = nullptr, fNvofCompose = nullptr;
    CUfunction fDrFlowSplat = nullptr, fDrFlowNorm = nullptr,               // native DRBA
               fDrDrmSplat = nullptr, fDrDrmNorm = nullptr;
    int* dStaticFlag = nullptr;        // device flag k_pairDiff sets when the pair differs
    int* hStaticFlag = nullptr;        // pinned readback of it, one int per group
    uint64_t staticN = 0;              // identical pairs held this session

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
    int initState = 0;          // 0 pending, 1 ready, -1 failed
    uint32_t seq = 0;
    bool havePrev = false;

    // ---------------- lifetime ----------------
    void pushTok(uint32_t v)
    {
        { std::lock_guard<std::mutex> lk(mTok); toks.push_back(v); }
        cvTok.notify_one();
    }
    void die(const char* why)
    {
        LOG("native: %s\n", why);
        InterlockedExchange(&dead, 1);
        // pass through both locks before the wake: a waiter that has tested isDead but not
        // yet blocked would miss a notify sent in that gap
        { std::lock_guard<std::mutex> lk(mTok); }
        { std::lock_guard<std::mutex> lk(mMsg); }
        cvTok.notify_all();
        cvMsg.notify_all();
    }
    bool isDead() const { return InterlockedCompareExchange((volatile LONG*)&dead, 0, 0) != 0; }

    bool write(const void* p, DWORD n)
    {
        if (isDead()) return false;
        const uint8_t* b = (const uint8_t*)p;
        { std::lock_guard<std::mutex> lk(mMsg); msgs.emplace_back(b, b + n); }
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
            if (toks.empty()) return false;
            const uint32_t v = toks.front();
            toks.pop_front();
            lk.unlock();
            if (n < 4) return false;
            memcpy(d, &v, 4);
            d += 4;
            n -= 4;
        }
        return true;
    }
};
// ---- part 2: handoff, engine load, kernels, buffers, the compute thread -------------------

// What a --resident process keeps between sessions (memory priority 7): the TensorRT runtime,
// the two deserialized engines, the JIT kernel cache and the NVRTC kernel module, plus the
// last successful handoff's facts so the same window starts without a python process at all.
// Everything else in NativeRife (stream, execution contexts, buffers, imports, events) is per
// session. Sessions are strictly sequential (the previous compute thread is joined before
// the next early thread starts), so no lock guards this.
struct NativeResident
{
    std::string ifnetPath, encodePath, jitPath, restorePath;
    nvinfer1::IRuntime* rt = nullptr;
    nvinfer1::ICudaEngine* engIf = nullptr;
    nvinfer1::ICudaEngine* engEnc = nullptr;
    nvinfer1::ICudaEngine* engRest = nullptr;   // live Restore: kept beside the pair, per path
    std::string block0Path;                     // native DRBA: kept beside the pair, per path
    nvinfer1::ICudaEngine* engB0 = nullptr;
    std::string gmPath[5];                      // the GMFSS set (sub-step 5a), exclusive with the pair
    nvinfer1::ICudaEngine* engGm[5] = {};
    nvinfer1::IRuntimeCache* jit = nullptr;
    CUmodule cuMod = nullptr;
    bool encHalf = true;
    int dev = 0;
    // memoised handoff: the python command line minus gen (the fast path ignores gen and the
    // cold path builds the fixed 1to8 class regardless) and the facts it answered with
    std::wstring handoffKey;
    NativeRife facts;
    bool haveFacts = false;
};
static NativeResident g_res;

// drop the resident engine pair (a different pair is needed for a new window size). The
// kernel module and the memoised handoff are engine independent and stay: the session that
// calls this already adopted the module in nativeBuildKernels and stored the new handoff
// facts (unloading the module here broke the first kernel launch of session 3 in the
// 2026-09-12 harness run). The calling thread must have the device bound.
static void residentFreeEngines()
{
    if (!g_res.rt) return;
    if (g_res.jit) { delete g_res.jit; g_res.jit = nullptr; }
    if (g_res.engIf) { delete g_res.engIf; g_res.engIf = nullptr; }
    if (g_res.engEnc) { delete g_res.engEnc; g_res.engEnc = nullptr; }
    if (g_res.engRest) { delete g_res.engRest; g_res.engRest = nullptr; }
    if (g_res.engB0) { delete g_res.engB0; g_res.engB0 = nullptr; }
    g_res.block0Path.clear();
    for (int i = 0; i < 5; i++)
    {
        if (g_res.engGm[i]) { delete g_res.engGm[i]; g_res.engGm[i] = nullptr; }
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
    if (g_res.cuMod) { cuModuleUnload(g_res.cuMod); g_res.cuMod = nullptr; }
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
// fruc (Nvidia Smooth Motion) and rifedrba (RIFE with DRBA timing). Any other name is refused.
static bool nativeBackendOk(const std::wstring& backend)
{
    return backend == L"rife" || backend == L"blend" || backend == L"echo" || backend == L"gmfss"
        || backend == L"nvof" || backend == L"fruc" || backend == L"rifedrba";
}

// the GMFSS engine set, in handoff order (the NATIVE-PATH keys and the log names)
static const char* const kGmKey[5] = { "gfeat", "gflow", "gmetric", "gifnet", "gfusion" };
static const char* const kGmName[5] = { "feat_ext", "gmflow_bidir", "metricnet", "ifnet", "fusionnet" };

// Upscale to (2026-09-15, full native migration item 2): _Fit.__init__'s derivation of the
// internal render size from the exe's own --upscale H and the handoff geometry: the factor
// up_h / h clamped to 1/16..16 (render.py's UPSCALE_F clamp), even dims, dropped when it
// equals the fit rect (one resize, not two). Python rounds half to even and this rounds half
// away, but the & ~1 lands both on the same even number. Runs after (w, h, dw, dh) are known,
// on the resident facts too (uw is never a stored fact, the flag is per session).
static void nativeDeriveUpscale(NativeRife& nr)
{
    nr.uw = nr.uh = 0;
    nr.upAa = false;
    nr.fitAa = nr.dh < nr.h;
    if (g_upscaleH > 0 && g_upscaleH != nr.h && nr.h > 0)
    {
        double f = (double)g_upscaleH / (double)nr.h;
        f = f < 1.0 / 16 ? 1.0 / 16 : (f > 16.0 ? 16.0 : f);
        int uw = (int)lround(nr.w * f) & ~1, uh = (int)lround(nr.h * f) & ~1;
        if (uw < 2) uw = 2;
        if (uh < 2) uh = 2;
        if (uw != nr.dw || uh != nr.dh)
        {
            nr.uw = uw; nr.uh = uh;
            nr.upAa = uh < nr.h;
            nr.fitAa = nr.dh < uh;   // the second resize shrinks the internal frame's height
        }
    }
}

// ---- warm engine lookup ---------------------------------------------------------------------
// trt_lookup.py's naming and find_* checks in C++ (engine names are shared with the offline
// python renders, so the two must agree): when a previous session built AND warmed the engines
// for this window, the lookup answers the handoff as NATIVE-PATH lines plus one LIVE READY line,
// the format nativeHandoff parses. A miss returns false and nativeHandoff builds the engines.
struct LookupTags { std::string trt, w, r, rest; bool ok = false; };   // rest = realesr.weights_hash()

static std::string lkEnv(const char* name)
{
    char buf[512];
    size_t n = 0;
    if (getenv_s(&n, buf, sizeof(buf), name) || !n) return std::string();
    return std::string(buf);
}

// SMV_HANDOFF_DUMP=1: log every handoff line (python's and the host's own) as `handoff-line ...`
static const bool g_handoffDump = lkEnv("SMV_HANDOFF_DUMP") == "1";

// md5 of the concatenated file contents, first 10 hex digits (trt_lookup._md5_files)
static std::string lkMd5Files(const std::vector<std::wstring>& files)
{
    HCRYPTPROV prov = 0;
    HCRYPTHASH h = 0;
    std::string out;
    if (!CryptAcquireContextW(&prov, nullptr, nullptr, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT)) return out;
    if (CryptCreateHash(prov, CALG_MD5, 0, 0, &h))
    {
        bool ok = true;
        std::vector<BYTE> buf(1 << 20);
        for (const auto& f : files)
        {
            HANDLE fh = CreateFileW(f.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                    FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
            if (fh == INVALID_HANDLE_VALUE) { ok = false; break; }
            DWORD got = 0;
            while (ok && ReadFile(fh, buf.data(), (DWORD)buf.size(), &got, nullptr) && got)
                ok = CryptHashData(h, buf.data(), got, 0) != FALSE;
            CloseHandle(fh);
            if (!ok) break;
        }
        BYTE dig[16];
        DWORD dl = sizeof(dig);
        if (ok && CryptGetHashParam(h, HP_HASHVAL, dig, &dl, 0))
        {
            char hex[33];
            for (int i = 0; i < 16; i++) sprintf_s(hex + 2 * i, 3, "%02x", dig[i]);
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
    if (forDir == engDir) return t;
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
        while (!v.empty() && (v.back() == '\n' || v.back() == '\r' || v.back() == ' ')) v.pop_back();
        for (char& c : v) if (c == '.') c = '_';
        if (!v.empty()) t.trt = "trt" + v;
    }
    WIN32_FIND_DATAW fd{};
    HANDLE hf;
    std::vector<std::wstring> pkls;
    const std::wstring tl = engDir + L"\\GMFSS_Fortuna\\train_log";
    hf = FindFirstFileW((tl + L"\\*.pkl").c_str(), &fd);
    if (hf != INVALID_HANDLE_VALUE)
    {
        do { pkls.push_back(fd.cFileName); } while (FindNextFileW(hf, &fd));
        FindClose(hf);
    }
    // python: sorted(os.listdir(...)), code point order on the names
    std::sort(pkls.begin(), pkls.end());
    for (auto& p : pkls) p = tl + L"\\" + p;
    const std::string w = pkls.empty() ? std::string() : lkMd5Files(pkls);
    const std::string r = lkMd5Files({ engDir + L"\\rife\\flownet.pkl" });
    t.w = w.empty() ? std::string() : "w" + w;
    t.r = r.empty() ? std::string() : "r" + r;
    // the Restore engine's name part: md5[:8] of the bundled Real-ESRGAN weights (optional)
    const std::string rs = lkMd5Files({ engDir + L"\\realesr-animevideov3.pth" });
    t.rest = rs.size() >= 8 ? rs.substr(0, 8) : std::string();
    // a shipped tree carries no weight files (the ONNX hold the weights, priority 29): the tags
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
                while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
                const size_t sp = s.find(' ');
                if (sp == std::string::npos) continue;
                const std::string k = s.substr(0, sp), v = s.substr(sp + 1);
                if (k == "w" && t.w.empty()) t.w = v;
                else if (k == "r" && t.r.empty()) t.r = v;
                else if (k == "rest" && t.rest.empty()) t.rest = v;
            }
            fclose(tf);
        }
    }
    t.ok = !t.trt.empty() && !t.w.empty() && !t.r.empty();
    return t;
}

// python's round(): ties to even (the default FP rounding mode), then int()
static int lkRound(double x) { return (int)std::nearbyint(x); }

static std::string lkG(double v)   // f"{v:g}"
{
    char b[64];
    sprintf_s(b, "%g", v);
    return b;
}

struct LkShape { const char* name; std::vector<int64_t> dims; };

// MUST equal trt_lookup.ONNX_REV (a stale ONNX file is never used; rev 2 = the PRelu rewrite,
// priority 30 step 2). Engines built from an older graph never survive a bump: the app and the
// CLI empty the engine cache when its stamp (weights_tags.txt, which carries the rev, plus the
// TensorRT-RTX version) no longer matches (src/render/cache.ts)
static const int kOnnxRev = 2;

// engine_name(): <base>_<shape per input joined by x, inputs by _>_<trt>_<w>, a dynamic batch
// axis written lo"to"hi
static std::string lkEngineName(const std::string& base, const std::vector<LkShape>& set,
                                const char* dynInput, int dynLo, int dynHi, const LookupTags& t)
{
    std::string s = base;
    for (const auto& in : set)
    {
        s += '_';
        for (size_t i = 0; i < in.dims.size(); i++)
        {
            if (i) s += 'x';
            if (i == 0 && dynInput && !strcmp(in.name, dynInput))
                s += std::to_string(dynLo) + "to" + std::to_string(dynHi);
            else s += std::to_string(in.dims[i]);
        }
    }
    return s + "_" + t.trt + "_" + t.w;
}

static bool lkFile(const std::string& p) { return fileExistsA(p); }

static std::string lkJit(const std::string& enginePath)
{
    const std::string kind = lkEnv("SMV_TRT_CACHE_KIND");
    return enginePath + (kind.empty() ? "" : "." + kind) + ".jit";
}

static bool lkWarm(const std::string& jit, const std::string& key)
{
    if (!lkFile(jit)) return false;
    std::ifstream f(jit + ".warm");
    std::string ln;
    while (std::getline(f, ln))
    {
        while (!ln.empty() && (ln.back() == '\r' || ln.back() == ' ' || ln.back() == '\t')) ln.pop_back();
        size_t a = 0;
        while (a < ln.size() && (ln[a] == ' ' || ln[a] == '\t')) a++;
        if (ln.compare(a, std::string::npos, key) == 0) return true;
    }
    return false;
}

static const char* lkDtype(nvinfer1::DataType d)
{
    switch (d)
    {
    case nvinfer1::DataType::kFLOAT: return "fp32";
    case nvinfer1::DataType::kHALF: return "fp16";
    case nvinfer1::DataType::kBF16: return "bf16";
    case nvinfer1::DataType::kINT32: return "i32";
    case nvinfer1::DataType::kINT64: return "i64";
    case nvinfer1::DataType::kBOOL: return "bool";
    case nvinfer1::DataType::kUINT8: return "u8";
    default: return "?";
    }
}

struct LkIo { std::vector<std::pair<std::string, std::string>> ins, outs; };

// find_engine() + _inspect(): the file by name, deserialized without a context, every input
// admitting every shape set (static dims exact, dynamic ones inside profile 0)
static bool lkFind(nvinfer1::IRuntime* rt, const std::string& cacheDir, const std::string& base,
                   const std::vector<std::vector<LkShape>>& sets, const char* dynInput, int dynLo,
                   int dynHi, const LookupTags& t, std::string& path, LkIo& io)
{
    path = cacheDir + "\\" + lkEngineName(base, sets[0], dynInput, dynLo, dynHi, t) + ".engine";
    if (!lkFile(path)) return false;
    std::ifstream f(path, std::ios::binary);
    std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (blob.empty()) return false;
    std::unique_ptr<nvinfer1::ICudaEngine> eng(rt->deserializeCudaEngine(blob.data(), blob.size()));
    if (!eng) return false;
    for (const auto& set : sets)
        for (const auto& in : set)
        {
            const nvinfer1::Dims d = eng->getTensorShape(in.name);
            if (d.nbDims != (int)in.dims.size()) return false;
            bool dyn = false;
            for (int k = 0; k < d.nbDims; k++) dyn = dyn || d.d[k] < 0;
            if (!dyn)
            {
                for (int k = 0; k < d.nbDims; k++) if (d.d[k] != in.dims[k]) return false;
                continue;
            }
            const nvinfer1::Dims lo = eng->getProfileShape(in.name, 0, nvinfer1::OptProfileSelector::kMIN);
            const nvinfer1::Dims hi = eng->getProfileShape(in.name, 0, nvinfer1::OptProfileSelector::kMAX);
            if (lo.nbDims != d.nbDims || hi.nbDims != d.nbDims) return false;
            for (int k = 0; k < d.nbDims; k++)
                if (in.dims[k] < lo.d[k] || in.dims[k] > hi.d[k]) return false;
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

static bool lkAllFp32(const LkIo& io)
{
    for (auto& p : io.ins) if (p.second != "fp32") return false;
    for (auto& p : io.outs) if (p.second != "fp32") return false;
    return true;
}

static std::string lkDt(const LkIo& io, bool in, const char* name)
{
    for (auto& p : (in ? io.ins : io.outs)) if (p.first == name) return p.second;
    return "None";
}

static std::string lkF4(double v) { char b[64]; sprintf_s(b, "%.4f", v); return b; }

// One net of the GMFSS set: the handoff key, the engine base name, the ONNX file key, the
// build workspace multiplier and the input shapes.
struct LkGmNet { const char* key; const char* base; const char* onnx; int ws; std::vector<LkShape> set; };

// Everything the lookup and the build derive from a session's arguments, in ONE place: a name
// or a size that differed between the two would be an engine the build writes and the lookup
// never finds. Sizes follow trt_lookup / the engine classes in trt_runtime.py.
struct LkSession
{
    std::string backend, engDir, cacheDir;
    std::wstring engDirW;
    double imgScale = 1.0;               // the Image scale as the app passes it (%.2f)
    int cw = 0, ch = 0;                  // the capture
    int mw = 0, mh = 0;                  // the model frame: Image scale, even dims, 64 px floor
    int ph = 0, pw = 0;                  // the /64 pad (SMV_LIVE_SAFEPAD=1 on the RIFE family)
    std::string warmKey;                 // the line a warmed .jit.warm marker carries
    // gmfss: the half size, the five nets
    int hh = 0, hw = 0;
    std::vector<LkGmNet> gm;
    // rife / blend / rifedrba
    bool drba = false;
    double fs = 1.0;                     // SMV_RIFE_FLOW_SCALE snapped to a power-of-two rung
    std::string scaleTag;                // name part: the scale list
    std::vector<int64_t> x, f;           // the IFNet's frame pair and feature shapes
};

// Create a folder and its missing parents. Users delete engine\trt_cache_safe_to_delete whenever
// they like (the name invites it), and every engine, JIT and cubin write below assumes the folder
// exists: a missing one used to fail each save silently and rebuild every session.
static void ensureDirW(const std::wstring& dir)
{
    if (dir.empty() || dir == L".") return;
    const DWORD a = GetFileAttributesW(dir.c_str());
    if (a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY)) return;
    const size_t sl = dir.find_last_of(L"\\/");
    if (sl != std::wstring::npos && sl > 0 && dir[sl - 1] != L':') ensureDirW(dir.substr(0, sl));
    if (!CreateDirectoryW(dir.c_str(), nullptr) && GetLastError() != ERROR_ALREADY_EXISTS)
        LOG("cache: cannot create %ls (error %lu)\n", dir.c_str(), GetLastError());
}

static bool lkSession(const std::wstring& script, const std::wstring& backendW, uint32_t capW,
                      uint32_t capH, LkSession& s)
{
    s = LkSession();
    s.backend = wideToUtf8(backendW);
    const size_t sl = script.find_last_of(L"\\/");
    if (sl == std::wstring::npos) return false;
    s.engDirW = script.substr(0, sl);
    for (auto& c : s.engDirW) if (c == L'/') c = L'\\';   // the dev harnesses pass forward slashes
    s.engDir = wideToUtf8(s.engDirW);
    const std::string envCache = lkEnv("SMV_TRT_CACHE");
    s.cacheDir = envCache.empty() ? s.engDir + "\\trt_cache_safe_to_delete" : envCache;
    ensureDirW(utf8ToWide(s.cacheDir));
    char sb[32];
    sprintf_s(sb, "%.2f", g_flowScale);
    s.imgScale = (std::max)(0.01, (std::min)(1.0, strtod(sb, nullptr)));
    s.cw = (int)capW; s.ch = (int)capH;
    s.mw = s.cw; s.mh = s.ch;
    if (s.imgScale < 1.0)
    {
        s.mw = (std::max)(64, lkRound(s.cw * s.imgScale) & ~1);
        s.mh = (std::max)(64, lkRound(s.ch * s.imgScale) & ~1);
    }
    s.ph = (s.mh + 63) / 64 * 64; s.pw = (s.mw + 63) / 64 * 64;
    char key[64];
    if (s.backend == "gmfss")
    {
        const int hh = s.hh = s.ph / 2, hw = s.hw = s.pw / 2;
        // featurenet at the padded frame, the rest at the half, gmflow_bidir with the doubled
        // workspace ceiling; the IFNet's ONNX key carries its baked scale list
        s.gm = {
            { "gfeat", "featurenet", "featurenet", 1, { { "x", { 1, 3, s.ph, s.pw } } } },
            { "gflow", "gmflow_bidir", "gmflow_bidir", 2, { { "img0", { 1, 3, hh, hw } }, { "img1", { 1, 3, hh, hw } } } },
            { "gmetric", "metricnet", "metricnet", 1, { { "i0", { 1, 3, hh, hw } }, { "i1", { 1, 3, hh, hw } },
                                                        { "f01", { 1, 2, hh, hw } }, { "f10", { 1, 2, hh, hw } } } },
            { "gifnet", "ifnet", "ifnet_sl8-4-2-1", 1, { { "x", { 1, 6, hh, hw } }, { "timestep", { 1, 1, 1, 1 } } } },
            { "gfusion", "fusionnet", "fusionnet", 1, { { "a", { 1, 9, hh, hw } }, { "b", { 1, 128, hh, hw } },
                                                        { "c", { 1, 256, hh / 2, hw / 2 } }, { "d", { 1, 384, hh / 4, hw / 4 } } } },
        };
        // on the fusionnet jit; the `|0x0` tail was the flow grid of the dropped Flow scale,
        // kept so every warm marker written before 2026-09-25 still matches
        sprintf_s(key, "%dx%d|0x0", s.ph, s.pw);
        s.warmKey = key;
        return true;
    }
    s.drba = s.backend == "rifedrba";
    const std::string e = lkEnv("SMV_RIFE_FLOW_SCALE");
    if (!e.empty()) s.fs = strtod(e.c_str(), nullptr);
    s.fs = (std::min)(1.0, (std::max)(0.25, std::pow(2.0, std::nearbyint(std::log2((std::min)(1.0, (std::max)(0.25, s.fs)))))));
    if (lkEnv(g_offline ? "SMV_RIFE_SAFEPAD" : "SMV_LIVE_SAFEPAD") == "1") { s.pw = (std::max)(s.pw, 1152); s.ph = (std::max)(s.ph, 640); }
    s.scaleTag = lkG(16 / s.fs) + "-" + lkG(8 / s.fs) + "-" + lkG(4 / s.fs) + "-" + lkG(2 / s.fs) + "-" + lkG(1 / s.fs);
    s.x = { 1, 6, s.ph, s.pw };
    s.f = { 1, 16, s.ph, s.pw };
    sprintf_s(key, "%dx%d", s.ph, s.pw);
    s.warmKey = key;
    return true;
}

// the RIFE family's engine base names (they carry the weights tag)
static std::string lkIfnetBase(const LkSession& s, const LookupTags& t) { return "rife_ifnet_" + t.r + "_" + s.scaleTag; }
static std::string lkEncodeBase(const LkSession& s, const LookupTags& t) { (void)s; return "rife_encode_" + t.r; }
static std::string lkBlock0Base(const LkSession& s, const LookupTags& t) { return "rife_block0_" + t.r + "_" + lkG(16 / s.fs); }
// the IFNet input set at a timestep batch b
static std::vector<LkShape> lkIfnetSet(const LkSession& s, int b)
{
    return { { "x", s.x }, { "timestep", { b, 1, s.ph, s.pw } }, { "f0", s.f }, { "f1", s.f } };
}
static std::vector<LkShape> lkBlock0Set(const LkSession& s)
{
    const std::vector<int64_t> x = { 1, 3, s.ph, s.pw }, f = { 1, 16, s.ph, s.pw };
    return { { "img0", x }, { "img1", x }, { "f0", f }, { "f1", f } };
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
static bool lkBaseHandoff(const std::wstring& script, const std::wstring& backendW, uint32_t capW,
                          uint32_t capH, std::vector<std::string>& lines)
{
    if (g_offline) return false;   // the offline host gets its engine paths from render.py
    LkSession s;
    if (!lkSession(script, backendW, capW, capH, s)) return false;
    const std::string& backend = s.backend;
    const bool geoOnly = backend == "nvof" || backend == "fruc";   // no engine: geometry only
    if (!geoOnly && backend != "rife" && backend != "blend" && backend != "echo" && backend != "gmfss"
        && backend != "rifedrba")
        return false;
    const std::string& engDir = s.engDir;
    const std::string& cacheDir = s.cacheDir;
    const double imgScale = s.imgScale;
    const int cw = s.cw, ch = s.ch, mw = s.mw, mh = s.mh;
    const int outW = W ? (int)W : cw, outH = W ? (int)H : ch;
    auto fitRect = [&](int w, int h, int& dw, int& dh, int& x0, int& y0)
    {
        if (outW == w && outH == h) { dw = w; dh = h; x0 = 0; y0 = 0; return; }
        const double s = (std::min)((double)outW / w, (double)outH / h);
        dw = (std::max)(2, lkRound(w * s) & ~1);
        dh = (std::max)(2, lkRound(h * s) & ~1);
        x0 = (outW - dw) / 2; y0 = (outH - dh) / 2;   // python // on non-negative values
    };
    auto geo = [&](int ph, int pw, int w, int h, int dw, int dh, int x0, int y0) -> std::string
    {
        char b[512];
        sprintf_s(b, "ph=%d pw=%d w=%d h=%d cw=%d ch=%d outw=%d outh=%d dw=%d dh=%d x0=%d y0=%d",
                  ph, pw, w, h, cw, ch, outW, outH, dw, dh, x0, y0);
        return b;
    };
    if (backend == "echo")
    {
        int dw, dh, x0, y0;
        fitRect(cw, ch, dw, dh, x0, y0);
        lines.push_back("NATIVE-PATH cache=" + cacheDir);
        lines.push_back("LIVE READY native=1 " + geo(ch, cw, cw, ch, dw, dh, x0, y0)
                        + " scale=1.0000 batch=0 batchpad=0 effects=0 restore=0 engine=none fast=1");
        return true;
    }
    if (geoOnly)
    {
        const int gw = mw, gh = mh;
        int dw, dh, x0, y0;
        fitRect(gw, gh, dw, dh, x0, y0);
        const std::string tail = " scale=" + lkF4(imgScale) + " batch=0 batchpad=0 effects=0 restore=0 engine="
                                 + backend + " fast=1";
        if (backend == "nvof")
        {
            // no /64 pad: the Optical Flow engine takes any size from 32x32
            lines.push_back("NATIVE-PATH cache=" + cacheDir);
            lines.push_back("LIVE READY native=1 " + geo(gh, gw, gw, gh, dw, dh, x0, y0) + tail);
            return true;
        }
        // fruc: the bridge folder (SMV_NVOFFRUC_DIR as nvoffruc.py reads it) and its three DLLs
        std::string fdir = lkEnv("SMV_NVOFFRUC_DIR");
        if (fdir.empty()) fdir = engDir + "\\nvoffruc";
        for (const char* dll : { "nvoffruc_bridge.dll", "NvOFFRUC.dll", "cudart64_110.dll" })
            if (!lkFile(fdir + "\\" + dll))
            { LOG("native: Nvidia Smooth Motion needs %s in %s\n", dll, fdir.c_str()); return false; }
        const int ph = (gh + 63) / 64 * 64, pw = (gw + 63) / 64 * 64;   // never the safe pad
        lines.push_back("NATIVE-PATH cache=" + cacheDir);
        lines.push_back("NATIVE-PATH fruc=" + fdir);
        lines.push_back("LIVE READY native=1 " + geo(ph, pw, gw, gh, dw, dh, x0, y0) + tail);
        return true;
    }
    if (!nativeLoadDlls(script)) return false;
    const LookupTags& t = lkTags(s.engDirW);
    if (!t.ok) { LOG("native: host lookup could not read the engine name tags\n"); return false; }
    nvinfer1::IRuntime* rt = lkRuntime();
    if (!rt) return false;
    const int ph = s.ph, pw = s.pw;
    if (backend == "gmfss")
    {
        const int hh = s.hh, hw = s.hw;
        std::string path[5], jit[5];
        for (int i = 0; i < 5; i++)
        {
            LkIo io;
            if (!lkFind(rt, cacheDir, s.gm[i].base, { s.gm[i].set }, nullptr, 0, 0, t, path[i], io)) return false;
            jit[i] = lkJit(path[i]);
        }
        if (!lkWarm(jit[4], s.warmKey)) return false;
        for (int i = 0; i < 4; i++) if (!lkFile(jit[i])) return false;
        int dw, dh, x0, y0;
        fitRect(mw, mh, dw, dh, x0, y0);
        for (int i = 0; i < 5; i++)
        {
            lines.push_back(std::string("NATIVE-PATH ") + s.gm[i].key + "=" + path[i]);
            lines.push_back(std::string("NATIVE-PATH ") + s.gm[i].key + "jit=" + jit[i]);
        }
        lines.push_back("NATIVE-PATH cache=" + cacheDir);
        char tail[256];
        sprintf_s(tail, " batch=0 batchpad=0 hh=%d hw=%d effects=0 restore=0 engine=gmfss fast=1", hh, hw);
        lines.push_back("LIVE READY native=1 " + geo(ph, pw, mw, mh, dw, dh, x0, y0) + " scale="
                        + lkF4(imgScale) + tail);
        return true;
    }
    // rife / blend / rifedrba: the batched class (a timestep batch of 1 to 8, `_bd8`) unless a
    // build marked it "did not fit" at this size, then the unbatched one
    const bool drba = s.drba;
    int batch = 8;
    std::string base = lkIfnetBase(s, t) + "_bd8";
    std::vector<std::vector<LkShape>> sets = { lkIfnetSet(s, 8), lkIfnetSet(s, 1) };
    {
        const std::string p = cacheDir + "\\" + lkEngineName(base, sets[0], "timestep", 1, 8, t) + ".engine";
        if (!lkFile(p) && lkFile(p + ".nofit"))
        {
            LOG("native: batched engine marked \"did not fit\" at %dx%d, looking up the unbatched engine\n", pw, ph);
            batch = 0;
            base = lkIfnetBase(s, t);
            sets = { lkIfnetSet(s, 1) };
        }
    }
    std::string ipath, epath;
    LkIo iio, eio;
    if (!lkFind(rt, cacheDir, base, sets, batch ? "timestep" : nullptr, 1, batch, t, ipath, iio)) return false;
    if (!lkAllFp32(iio)) return false;
    const std::string ijit = lkJit(ipath);
    if (!lkWarm(ijit, s.warmKey)) return false;
    if (!lkFind(rt, cacheDir, lkEncodeBase(s, t), { { { "img", { 1, 3, ph, pw } } } }, nullptr, 0, 0, t, epath, eio))
        return false;
    std::string bpath, bjit;
    if (drba)
    {
        LkIo bio;
        if (!lkFind(rt, cacheDir, lkBlock0Base(s, t), { lkBlock0Set(s) }, nullptr, 0, 0, t, bpath, bio))
            return false;
        if (!lkAllFp32(bio)) return false;
        bjit = lkJit(bpath);
        if (!lkWarm(bjit, s.warmKey)) return false;
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
    char tail[512];
    sprintf_s(tail, " batch=%d batchpad=%d xdtype=%s tdtype=%s fdtype=%s outdtype=%s ename=%s einame=%s edtype=%s eidtype=%s effects=0 fast=1%s",
              batch, 0, lkDt(iio, true, "x").c_str(),
              lkDt(iio, true, "timestep").c_str(), lkDt(iio, true, "f0").c_str(),
              lkDt(iio, false, "merged").c_str(), en.c_str(), ein.c_str(), ed.c_str(), eid.c_str(),
              drba ? " engine=drba lag=1" : "");
    lines.push_back("LIVE READY native=1 " + geo(ph, pw, mw, mh, dw, dh, x0, y0) + " scale=" + lkF4(imgScale) + tail);
    return true;
}

// the Restore engine for a model size, as trt_runtime.RestoreEngine names it: restore_<md5[:8] of
// the weights>, input x (1,3,h,w) at the unpadded model size (the fit's w x h)
static std::string lkRestorePath(const std::string& cacheDir, const LookupTags& t, int w, int h)
{
    return cacheDir + "\\" + lkEngineName("restore_" + t.rest, { { "x", { 1, 3, h, w } } }, nullptr, 0, 0, t) + ".engine";
}

static int lkField(const std::string& line, const char* key)
{
    const std::string k = std::string(" ") + key + "=";
    const size_t p = line.find(k);
    return p == std::string::npos ? -1 : atoi(line.c_str() + p + k.size());
}

// The session's full handoff answer: the base answer plus, for live Restore (every backend:
// the host's storeSlot chain restores any model's output), the Restore engine and its jit once
// both exist (nativeLocalBuild creates them; the engine has no warm marker).
static bool nativeLocalHandoff(const std::wstring& script, const std::wstring& backendW, uint32_t capW,
                               uint32_t capH, std::vector<std::string>& lines)
{
    if (!lkBaseHandoff(script, backendW, capW, capH, lines)) return false;
    if (!g_restore) return true;
    LkSession s;
    const int w = lkField(lines.back(), "w"), h = lkField(lines.back(), "h");
    if (!lkSession(script, backendW, capW, capH, s) || w <= 0 || h <= 0) { lines.clear(); return false; }
    const LookupTags& t = lkTags(s.engDirW);
    if (!t.ok || t.rest.empty()) { lines.clear(); return false; }
    const std::string rpath = lkRestorePath(s.cacheDir, t, w, h);
    const std::string rjit = lkJit(rpath);
    if (!lkFile(rpath) || !lkFile(rjit)) { lines.clear(); return false; }
    const std::string ready = lines.back();
    lines.pop_back();
    lines.push_back("NATIVE-PATH restore=" + rpath);
    lines.push_back("NATIVE-PATH rjit=" + rjit);
    lines.push_back(ready);
    return true;
}

// ---- cold engine build inside the host (ONNX-in-exe step 3c-2, 2026-09-21) ------------------
// A window size with no warm engines used to need python for the model load, the torch export,
// the build and the warm-up. With the size-free ONNX in engine\onnx (step 3b) the host builds
// them itself with the settings of the python builder (removed in 27b) (strongly typed, workspace
// SMV_TRT_WORKSPACE_GB (8) x the per-graph multiplier, optimization level 5, one profile pinning
// every input to its shape, only a declared batch axis spanning), writes the engine file only
// after it deserializes, then warms every shape the session will run with a FRESH runtime cache
// (EAGER specialization, so the saved kernels are the specialized ones) and writes that cache as
// the engine's .jit and the warm key, exactly the files python's slow handoff leaves. Built from
// the same file, python and this builder give output-identical engines (bd8_triage.out). Not
// ported on purpose: python's eager-vs-TRT check of the encode engine (no torch here; the
// size-free build is proven bit-exact against the per-size engines, gate_3b). After the build
// the host lookup must hit, else python runs as before. RIFE / Frame Blend (3c-2a) first.
// (kOnnxRev sits above lkEngineName: the engine names carry it too)

class LkBuildLogger : public nvinfer1::ILogger
{
public:
    std::string errs;
    void log(Severity s, const char* msg) noexcept override
    {
        if (s <= Severity::kWARNING) LOG("[TRT] %s\n", msg);
        if (s <= Severity::kERROR) { errs += msg; errs += "\n"; }
    }
};

static bool lkOom(const std::string& s)
{
    std::string l = s;
    for (auto& c : l) c = (char)tolower((unsigned char)c);
    return l.find("out of memory") != std::string::npos || l.find("outofmemory") != std::string::npos
        || l.find("not enough gpu memory") != std::string::npos;
}

static bool lkWriteFile(const std::string& path, const void* data, size_t n)
{
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary);
        f.write(static_cast<const char*>(data), (std::streamsize)n);
        if (!f) return false;
    }
    return MoveFileExA(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING) != FALSE;
}

// one engine from a size-free ONNX; *oom = the failure was memory-shaped (python's .nofit rule:
// a null build or an out-of-memory message)
static bool lkBuild(nvinfer1::IRuntime* rt, const std::string& onnx, const std::string& out,
                    const std::vector<LkShape>& set, const char* dynInput, int dynLo, int dynHi,
                    int wsMult, bool* oom)
{
    *oom = false;
    LkBuildLogger lg;
    const auto t0 = std::chrono::steady_clock::now();
    std::unique_ptr<nvinfer1::IBuilder> b(nvinfer1::createInferBuilder(lg));
    if (!b) return false;
    const auto flags = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    std::unique_ptr<nvinfer1::INetworkDefinition> net(b->createNetworkV2(flags));
    if (!net) return false;
    std::unique_ptr<nvonnxparser::IParser> p(nvonnxparser::createParser(*net, lg));
    if (!p || !p->parseFromFile(onnx.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
    {
        if (p) for (int i = 0; i < p->getNbErrors(); i++) LOG("native: onnx parse: %s\n", p->getError(i)->desc());
        return false;
    }
    std::unique_ptr<nvinfer1::IBuilderConfig> cfg(b->createBuilderConfig());
    nvinfer1::IOptimizationProfile* prof = b->createOptimizationProfile();
    if (!cfg || !prof) return false;
    double ws = 8.0;
    const std::string wse = lkEnv("SMV_TRT_WORKSPACE_GB");
    if (!wse.empty()) { const double v = strtod(wse.c_str(), nullptr); if (v > 0) ws = v; }
    cfg->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, (size_t)(ws * (1ull << 30)) * wsMult);
    cfg->setBuilderOptimizationLevel(5);
    for (int i = 0; i < net->getNbInputs(); i++)
    {
        nvinfer1::ITensor* t = net->getInput(i);
        const LkShape* s = nullptr;
        for (const auto& in : set) if (!strcmp(in.name, t->getName())) s = &in;
        if (!s || (int)s->dims.size() != t->getDimensions().nbDims)
        { LOG("native: onnx input %s does not match the engine class\n", t->getName()); return false; }
        nvinfer1::Dims lo, hi;
        lo.nbDims = hi.nbDims = (int)s->dims.size();
        for (int k = 0; k < lo.nbDims; k++) lo.d[k] = hi.d[k] = s->dims[k];
        if (dynInput && !strcmp(s->name, dynInput)) { lo.d[0] = dynLo; hi.d[0] = dynHi; }
        if (!prof->setDimensions(t->getName(), nvinfer1::OptProfileSelector::kMIN, lo)
            || !prof->setDimensions(t->getName(), nvinfer1::OptProfileSelector::kOPT, hi)
            || !prof->setDimensions(t->getName(), nvinfer1::OptProfileSelector::kMAX, hi))
            return false;
    }
    cfg->addOptimizationProfile(prof);
    std::unique_ptr<nvinfer1::IHostMemory> ser(b->buildSerializedNetwork(*net, *cfg));
    if (!ser)
    {
        *oom = true;   // python: "returned None" counts as memory-shaped
        LOG("native: engine build returned nothing for %s\n", out.c_str());
        return false;
    }
    // deserialize BEFORE the file reaches the cache: a file that cannot load must not be left
    std::unique_ptr<nvinfer1::ICudaEngine> eng(rt->deserializeCudaEngine(ser->data(), ser->size()));
    if (!eng) { *oom = lkOom(lg.errs); return false; }
    eng.reset();
    if (!lkWriteFile(out, ser->data(), ser->size())) { LOG("native: cannot write %s\n", out.c_str()); return false; }
    LOG("native: built %s from %s in %.1fs\n", out.substr(out.find_last_of("\\/") + 1).c_str(),
        onnx.substr(onnx.find_last_of("\\/") + 1).c_str(),
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return true;
}

static bool lkReadFile(const std::string& path, std::vector<char>& out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return !out.empty();
}

static size_t lkDtypeBytes(nvinfer1::DataType d)
{
    switch (d)
    {
    case nvinfer1::DataType::kHALF: case nvinfer1::DataType::kBF16: return 2;
    case nvinfer1::DataType::kINT64: return 8;
    case nvinfer1::DataType::kBOOL: case nvinfer1::DataType::kUINT8: case nvinfer1::DataType::kINT8: return 1;
    default: return 4;
    }
}

// warm an engine for every input shape set with a fresh runtime cache (the existing .jit file
// merged first, as python's context does), then write the cache as <engine>[.kind].jit and add
// the warm key (none = the file only). Inputs are zeros: the kernels depend on shapes, not data.
static bool lkWarmEngine(nvinfer1::IRuntime* rt, const std::string& enginePath,
                         const std::vector<std::vector<LkShape>>& sets, const std::string& warmKey,
                         cudaStream_t st, bool* oom)
{
    *oom = false;
    std::vector<char> blob;
    if (!lkReadFile(enginePath, blob)) return false;
    std::unique_ptr<nvinfer1::ICudaEngine> eng(rt->deserializeCudaEngine(blob.data(), blob.size()));
    if (!eng) return false;
    std::unique_ptr<nvinfer1::IRuntimeConfig> cfg(eng->createRuntimeConfig());
    if (!cfg) return false;
    std::unique_ptr<nvinfer1::IRuntimeCache> cache(cfg->createRuntimeCache());
    if (!cache) return false;
    const std::string jit = lkJit(enginePath);
    std::vector<char> cb;
    if (lkReadFile(jit, cb) && !cache->deserialize(cb.data(), cb.size()))
    {
        // python's rule: a rejected cache starts fresh and nothing in it is warm any more
        DeleteFileA((jit + ".warm").c_str());
        cache.reset(cfg->createRuntimeCache());
        if (!cache) return false;
    }
    cfg->setRuntimeCache(*cache);
    cfg->setDynamicShapesKernelSpecializationStrategy(nvinfer1::DynamicShapesKernelSpecializationStrategy::kEAGER);
    std::unique_ptr<nvinfer1::IExecutionContext> ctx(eng->createExecutionContext(cfg.get()));
    if (!ctx) { *oom = true; LOG("native: warm-up context failed for %s\n", enginePath.c_str()); return false; }
    std::vector<void*> bufs;
    bool ok = true;
    for (const auto& set : sets)
    {
        for (const auto& in : set)
        {
            nvinfer1::Dims d;
            d.nbDims = (int)in.dims.size();
            for (int k = 0; k < d.nbDims; k++) d.d[k] = in.dims[k];
            if (!ctx->setInputShape(in.name, d)) { ok = false; break; }
        }
        for (int i = 0; ok && i < eng->getNbIOTensors(); i++)
        {
            const char* n = eng->getIOTensorName(i);
            const nvinfer1::Dims d = ctx->getTensorShape(n);
            size_t bytes = lkDtypeBytes(eng->getTensorDataType(n));
            for (int k = 0; k < d.nbDims; k++) bytes *= (size_t)(d.d[k] > 0 ? d.d[k] : 1);
            void* ptr = nullptr;
            if (cudaMalloc(&ptr, bytes) != cudaSuccess) { ok = false; *oom = true; break; }
            bufs.push_back(ptr);
            cudaMemsetAsync(ptr, 0, bytes, st);
            ctx->setTensorAddress(n, ptr);
        }
        ok = ok && ctx->enqueueV3(st) && cudaStreamSynchronize(st) == cudaSuccess;
        for (void* ptr : bufs) cudaFree(ptr);
        bufs.clear();
        if (!ok) break;
    }
    if (!ok) { LOG("native: warm-up enqueue failed for %s\n", enginePath.c_str()); return false; }
    ctx.reset();
    std::unique_ptr<nvinfer1::IHostMemory> ser(cache->serialize());
    if (!ser || !lkWriteFile(jit, ser->data(), ser->size())) { LOG("native: cannot write %s\n", jit.c_str()); return false; }
    if (!warmKey.empty())
    {
        // trt_lookup.mark_warm: the key set, sorted, one per line
        std::vector<std::string> keys;
        {
            std::ifstream f(jit + ".warm");
            std::string ln;
            while (std::getline(f, ln))
            {
                while (!ln.empty() && (ln.back() == '\r' || ln.back() == ' ')) ln.pop_back();
                if (!ln.empty()) keys.push_back(ln);
            }
        }
        keys.push_back(warmKey);
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
        std::string txt;
        for (auto& k : keys) txt += k + "\n";
        if (!lkWriteFile(jit + ".warm", txt.data(), txt.size())) return false;
    }
    return true;
}

static std::string lkOnnxPath(const std::string& engDir, const std::string& key, const LookupTags& t)
{
    const std::string env = lkEnv("SMV_ONNX_DIR");
    return (env.empty() ? engDir + "\\onnx" : env) + "\\" + key + "_" + t.w + "_x" + std::to_string(kOnnxRev) + ".onnx";
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
                     const std::string& onnxKey, const std::vector<LkShape>& set, const char* dynInput,
                     int dynLo, int dynHi, int wsMult, const std::vector<std::vector<LkShape>>& warmSets,
                     const std::string& warmKey, cudaStream_t st, bool* oom)
{
    *oom = false;
    if (!lkFile(path))
    {
        const std::string onnx = lkOnnxPath(s.engDir, onnxKey, t);
        if (!lkFile(onnx)) { LOG("native: %s is missing (node scripts/export-onnx.js writes it)\n", onnx.c_str()); return false; }
        if (!lkBuild(rt, onnx, path, set, dynInput, dynLo, dynHi, wsMult, oom)) return false;
    }
    return lkWarmEngine(rt, path, warmSets, warmKey, st, oom);
}

// the RIFE family: the IFNet (the batched class first, the unbatched one when it did not fit),
// the encoder, and DRBA's block0 flow engine
static bool lkBuildRife(nvinfer1::IRuntime* rt, const LkSession& s, const LookupTags& t, cudaStream_t st)
{
    auto path = [&](const std::string& base, const std::vector<LkShape>& set, bool dyn)
    {
        return s.cacheDir + "\\" + lkEngineName(base, set, dyn ? "timestep" : nullptr, dyn ? 1 : 0, dyn ? 8 : 0, t) + ".engine";
    };
    bool oom = false;
    const std::string ibase = lkIfnetBase(s, t);
    const std::vector<LkShape> bset = lkIfnetSet(s, 8), uset = lkIfnetSet(s, 1);
    const std::string bpath = path(ibase + "_bd8", bset, true);
    bool batched = !lkFile(bpath + ".nofit") || lkFile(bpath);
    if (batched)
    {
        std::vector<std::vector<LkShape>> sets;
        for (int b : { 8, 1, 2, 3, 4, 5, 6, 7 }) sets.push_back(lkIfnetSet(s, b));   // the warm order: B, then 1..B-1
        if (!lkEnsure(rt, s, t, bpath, ibase + "_bd8", bset, "timestep", 1, 8, 1, sets, s.warmKey, st, &oom))
        {
            if (!oom) return false;
            batched = false;
            char why[160];
            sprintf_s(why, "%dx%d: the batched engine did not fit (host build)\n", s.pw, s.ph);
            if (!lkWriteFile(bpath + ".nofit", why, strlen(why)))
                LOG("native: cannot write %s.nofit, the next start retries the batched build\n", bpath.c_str());
            LOG("native: batched engine did not fit at %dx%d, marked; building the unbatched engine\n", s.pw, s.ph);
        }
    }
    if (!batched && !lkEnsure(rt, s, t, path(ibase, uset, false), ibase, uset, nullptr, 0, 0, 1, { uset }, s.warmKey, st, &oom))
        return false;
    const std::vector<LkShape> eset = { { "img", { 1, 3, s.ph, s.pw } } };
    const std::string ebase = lkEncodeBase(s, t);
    if (!lkEnsure(rt, s, t, path(ebase, eset, false), ebase, eset, nullptr, 0, 0, 1, { eset }, std::string(), st, &oom))
        return false;
    if (s.drba)
    {
        const std::vector<LkShape> b0set = lkBlock0Set(s);
        const std::string b0base = lkBlock0Base(s, t);
        if (!lkEnsure(rt, s, t, path(b0base, b0set, false), b0base, b0set, nullptr, 0, 0, 1, { b0set }, s.warmKey, st, &oom))
            return false;
    }
    return true;
}

// build + warm what lkBaseHandoff needs for this session; true = worth a second lookup
static bool lkBuildBackend(const std::wstring& script, const std::wstring& backendW, uint32_t capW, uint32_t capH)
{
    if (g_offline) return false;
    LkSession s;
    if (!lkSession(script, backendW, capW, capH, s)) return false;
    const bool gmfss = s.backend == "gmfss";
    if (s.backend != "rife" && s.backend != "blend" && !gmfss && !s.drba) return false;
    if (!nativeLoadDlls(script) || !g_onnxParserOk) return false;
    const LookupTags& t = lkTags(s.engDirW);
    if (!t.ok) return false;
    const auto t0 = std::chrono::steady_clock::now();
    cudaStream_t st = nullptr;
    if (!lkBuildStream(st)) return false;
    nvinfer1::IRuntime* rt = lkRuntime();
    bool done = rt != nullptr;
    if (done && gmfss)
    {
        // the warm key sits on the last net's jit (fusionnet)
        for (size_t i = 0; done && i < s.gm.size(); i++)
        {
            bool oom = false;
            const LkGmNet& n = s.gm[i];
            const std::string path = s.cacheDir + "\\" + lkEngineName(n.base, n.set, nullptr, 0, 0, t) + ".engine";
            done = lkEnsure(rt, s, t, path, n.onnx, n.set, nullptr, 0, 0, n.ws, { n.set },
                            i + 1 == s.gm.size() ? s.warmKey : std::string(), st, &oom);
        }
    }
    else if (done) done = lkBuildRife(rt, s, t, st);
    cudaStreamDestroy(st);
    LOG("native: host %sengine build for %dx%d %s (%.1fs)\n", gmfss ? "GMFSS " : "", s.pw, s.ph,
        done ? "done" : "failed",
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return done;
}

// the cold-start build: the backend's engines when its warm answer misses (echo, nvof and fruc
// have none), then for live Restore the Real-ESRGAN engine from engine\onnx (restore_<hash>_dth:
// fp16 input) at the model size the answer names, warmed once so its .jit exists. A backend that
// is already warm is never rewarmed. true = worth a second lookup.
static bool lkEnsureRestore(const std::wstring& script, const LkSession& s, int w, int h,
                            std::string& rpath, std::string& rjit);
static bool nativeLocalBuild(const std::wstring& script, const std::wstring& backendW, uint32_t capW, uint32_t capH)
{
    if (g_offline) return false;
    std::vector<std::string> base;
    if (!lkBaseHandoff(script, backendW, capW, capH, base))
    {
        if (!lkBuildBackend(script, backendW, capW, capH)) return false;
        base.clear();
        if (!lkBaseHandoff(script, backendW, capW, capH, base)) return false;
    }
    if (!g_restore) return true;
    LkSession s;
    const int w = lkField(base.back(), "w"), h = lkField(base.back(), "h");
    if (!lkSession(script, backendW, capW, capH, s) || w <= 0 || h <= 0) return false;
    std::string rpath, rjit;
    return lkEnsureRestore(script, s, w, h, rpath, rjit);
}

// the Real-ESRGAN Restore engine at the model size w x h (restore_<hash>_dth: fp16 input) from
// engine\onnx when it is missing, warmed once so its .jit exists; the live cold build and the
// offline host (priority 24 step 2) share it. true = both files exist, paths in rpath / rjit.
static bool lkEnsureRestore(const std::wstring& script, const LkSession& s, int w, int h,
                            std::string& rpath, std::string& rjit)
{
    if (!nativeLoadDlls(script)) return false;
    const LookupTags& t = lkTags(s.engDirW);
    if (!t.ok || t.rest.empty()) return false;
    rpath = lkRestorePath(s.cacheDir, t, w, h);
    rjit = lkJit(rpath);
    if (lkFile(rpath) && lkFile(rjit)) return true;
    if (!g_onnxParserOk) { LOG("native: the ONNX parser DLL is missing, cannot build the Restore engine\n"); return false; }
    const auto t0 = std::chrono::steady_clock::now();
    cudaStream_t st = nullptr;
    if (!lkBuildStream(st)) return false;
    nvinfer1::IRuntime* rt = lkRuntime();
    bool oom = false;
    const std::vector<LkShape> rset = { { "x", { 1, 3, h, w } } };
    const bool done = rt && lkEnsure(rt, s, t, rpath, "restore_" + t.rest + "_dth", rset, nullptr, 0, 0, 1,
                                     { rset }, std::string(), st, &oom);
    cudaStreamDestroy(st);
    LOG("native: host Restore engine build for %dx%d %s (%.1fs)\n", w, h, done ? "done" : "failed",
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return done;
}

// ---- offline RIFE engines (priority 24 step 1, 2026-09-22) ----------------------------------
// The offline host finds, builds and warms its own IFNet + Head engines (render.py used to load
// the model, build and warm them with torch and hand the paths over). The classes are the ones
// trt_lookup._rife_ifnet_spec names for offline: x2 = the unbatched IFNet, xN = the FIXED batch
// class `_b{N-1}` (every group exactly full), pinned to the /64 pad of the source. The per-B
// ONNX never ships, so a fixed class is built from the shipped `_bd8` graph with its batch axis
// pinned to B (harness\offline\gate_b.py: bit-exact with python's own `_b{B}` build). A batched
// build that runs out of memory is marked `.nofit` and the unbatched engine serves, as live.
struct OfflineEngines { std::string ifnet, encode, jit; int ph = 0, pw = 0, batch = 1; };

static bool lkOfflineRife(const std::wstring& script, int w, int h, int multi, OfflineEngines& o)
{
    LkSession s;
    if (!lkSession(script, L"rife", (uint32_t)w, (uint32_t)h, s)) return false;
    if (!nativeLoadDlls(script)) return false;
    const LookupTags& t = lkTags(s.engDirW);
    if (!t.ok) { LOG("offline: could not read the engine name tags\n"); return false; }
    nvinfer1::IRuntime* rt = lkRuntime();
    if (!rt) return false;
    const auto t0 = std::chrono::steady_clock::now();
    auto enginePath = [&](const std::string& base, const std::vector<LkShape>& set)
    { return s.cacheDir + "\\" + lkEngineName(base, set, nullptr, 0, 0, t) + ".engine"; };
    // one class: its engine exists and is warm at this size, else build (when missing) and warm
    auto ensure = [&](const std::string& base, const std::string& onnxKey, const std::vector<LkShape>& set,
                      const std::string& warmKey, cudaStream_t& st, bool* oom) -> bool
    {
        *oom = false;
        const std::string p = enginePath(base, set);
        if (lkFile(p) && (warmKey.empty() ? lkFile(lkJit(p)) : lkWarm(lkJit(p), warmKey))) return true;
        if (!lkFile(p) && !g_onnxParserOk) { LOG("offline: the ONNX parser DLL is missing, cannot build %s\n", base.c_str()); return false; }
        if (!st && !lkBuildStream(st)) return false;
        return lkEnsure(rt, s, t, p, onnxKey, set, nullptr, 0, 0, 1, { set }, warmKey, st, oom);
    };
    cudaStream_t st = nullptr;
    const std::string ibase = lkIfnetBase(s, t);
    const int B = lkEnv("SMV_RIFE_BATCH") == "1" ? 1 : multi - 1;
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
            if (ok) { base = bbase; set = bset; }
            else if (oom)
            {
                char why[160];
                sprintf_s(why, "%dx%d: the batched engine did not fit (offline host build)\n", s.pw, s.ph);
                if (!lkWriteFile(bpath + ".nofit", why, strlen(why)))
                    LOG("offline: cannot write %s.nofit, the next render retries the batched build\n", bpath.c_str());
                LOG("offline: batched engine did not fit at %dx%d, marked; using the unbatched engine\n", s.pw, s.ph);
            }
            else { if (st) cudaStreamDestroy(st); return false; }
        }
        else LOG("offline: batched engine marked \"did not fit\" at %dx%d, using the unbatched engine\n", s.pw, s.ph);
    }
    if (base == ibase) ok = ensure(ibase, ibase, set, s.warmKey, st, &oom);
    const std::string ebase = lkEncodeBase(s, t);
    const std::vector<LkShape> eset = { { "img", { 1, 3, s.ph, s.pw } } };
    ok = ok && ensure(ebase, ebase, eset, std::string(), st, &oom);
    const bool built = st != nullptr;
    if (st) cudaStreamDestroy(st);
    if (!ok) { LOG("offline: engine build for %dx%d failed\n", s.pw, s.ph); return false; }
    LkIo iio, eio;
    if (!lkFind(rt, s.cacheDir, base, { set }, nullptr, 0, 0, t, o.ifnet, iio) || !lkAllFp32(iio)
        || !lkFind(rt, s.cacheDir, ebase, { eset }, nullptr, 0, 0, t, o.encode, eio))
    { LOG("offline: the engines at %dx%d do not match their class (delete them from the cache to rebuild)\n", s.pw, s.ph); return false; }
    o.jit = lkJit(o.ifnet);
    o.ph = s.ph; o.pw = s.pw;
    o.batch = base == ibase ? 1 : B;
    LOG("offline: engines for %dx%d %s%s (%.1fs)\n", s.pw, s.ph,
        o.batch > 1 ? ("batched B=" + std::to_string(o.batch)).c_str() : "single tween",
        built ? " built by the host" : " warm",
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return true;
}

// ---- offline GMFSS engines (priority 24 step 3b, 2026-09-23) --------------------------------
// The live host's five-engine set (lkSession "gmfss": featurenet at the /64 pad, the fused bidir
// gmflow, metricnet, the GMFSS IFNet and fusionnet at the half), found warm or built and warmed from engine\onnx exactly as
// lkBuildBackend does for live, then handed to nr like the live handoff's NATIVE-PATH lines.
static bool lkOfflineGmfss(const std::wstring& script, int w, int h, NativeRife& nr)
{
    LkSession s;
    if (!lkSession(script, L"gmfss", (uint32_t)w, (uint32_t)h, s)) return false;
    if (!nativeLoadDlls(script)) return false;
    const LookupTags& t = lkTags(s.engDirW);
    if (!t.ok) { LOG("offline: could not read the engine name tags\n"); return false; }
    nvinfer1::IRuntime* rt = lkRuntime();
    if (!rt) return false;
    const auto t0 = std::chrono::steady_clock::now();
    cudaStream_t st = nullptr;
    for (size_t i = 0; i < s.gm.size(); i++)
    {
        const LkGmNet& n = s.gm[i];
        const std::string p = s.cacheDir + "\\" + lkEngineName(n.base, n.set, nullptr, 0, 0, t) + ".engine";
        // the warm key sits on the last net's jit (fusionnet), the others need their .jit
        const std::string wk = i + 1 == s.gm.size() ? s.warmKey : std::string();
        if (lkFile(p) && (wk.empty() ? lkFile(lkJit(p)) : lkWarm(lkJit(p), wk))) continue;
        if (!lkFile(p) && !g_onnxParserOk) { LOG("offline: the ONNX parser DLL is missing, cannot build %s\n", n.base); return false; }
        bool oom = false;
        if ((!st && !lkBuildStream(st))
            || !lkEnsure(rt, s, t, p, n.onnx, n.set, nullptr, 0, 0, n.ws, { n.set }, wk, st, &oom))
        {
            if (st) cudaStreamDestroy(st);
            LOG("offline: GMFSS engine build for %dx%d failed%s\n", s.pw, s.ph, oom ? " (out of memory)" : "");
            return false;
        }
    }
    const bool built = st != nullptr;
    if (st) cudaStreamDestroy(st);
    for (int i = 0; i < 5; i++)
    {
        LkIo io;
        if (!lkFind(rt, s.cacheDir, s.gm[i].base, { s.gm[i].set }, nullptr, 0, 0, t, nr.gmPath[i], io))
        { LOG("offline: the GMFSS engines at %dx%d do not match their class (delete them from the cache to rebuild)\n", s.pw, s.ph); return false; }
        nr.gmJit[i] = lkJit(nr.gmPath[i]);
    }
    nr.gmfss = true;
    nr.ph = s.ph; nr.pw = s.pw;
    nr.hh = s.hh; nr.hw = s.hw;
    LOG("offline: GMFSS engines for %dx%d%s (%.1fs)\n", s.pw, s.ph,
        built ? ", built by the host" : ", warm",
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return true;
}

// ---- offline DRBA block0 engine (priority 24 step 3c, 2026-09-23) ---------------------------
// DRBA's calc_flow engine (lkBlock0Base / lkBlock0Set, the live names) at the /64 pad of the
// source, warm or built + warmed from engine\onnx; the IFNet and encode come from lkOfflineRife
// (x2 = the unbatched class: DRBA enqueues one tween at a time with its own DRM map).
static bool lkOfflineBlock0(const std::wstring& script, int w, int h, std::string& path, std::string& jit)
{
    LkSession s;
    if (!lkSession(script, L"rifedrba", (uint32_t)w, (uint32_t)h, s)) return false;
    if (!nativeLoadDlls(script)) return false;
    const LookupTags& t = lkTags(s.engDirW);
    if (!t.ok) { LOG("offline: could not read the engine name tags\n"); return false; }
    nvinfer1::IRuntime* rt = lkRuntime();
    if (!rt) return false;
    const std::string base = lkBlock0Base(s, t);
    const std::vector<LkShape> set = lkBlock0Set(s);
    const std::string p = s.cacheDir + "\\" + lkEngineName(base, set, nullptr, 0, 0, t) + ".engine";
    const bool warm = lkFile(p) && lkWarm(lkJit(p), s.warmKey);
    if (!warm)
    {
        if (!lkFile(p) && !g_onnxParserOk) { LOG("offline: the ONNX parser DLL is missing, cannot build %s\n", base.c_str()); return false; }
        cudaStream_t st = nullptr;
        bool oom = false;
        const bool ok = lkBuildStream(st) && lkEnsure(rt, s, t, p, base, set, nullptr, 0, 0, 1, { set }, s.warmKey, st, &oom);
        if (st) cudaStreamDestroy(st);
        if (!ok) { LOG("offline: DRBA block0 engine build for %dx%d failed%s\n", s.pw, s.ph, oom ? " (out of memory)" : ""); return false; }
    }
    LkIo io;
    if (!lkFind(rt, s.cacheDir, base, { set }, nullptr, 0, 0, t, path, io) || !lkAllFp32(io))
    { LOG("offline: the DRBA block0 engine at %dx%d does not match its class\n", s.pw, s.ph); return false; }
    jit = lkJit(path);
    LOG("offline: DRBA block0 engine for %dx%d %s\n", s.pw, s.ph, warm ? "warm" : "built by the host");
    return true;
}

// ---- offline Smooth Motion (priority 24 step 3d, 2026-09-24) ---------------------------------
// no engine: the bridge folder the live handoff names (SMV_NVOFFRUC_DIR as nvoffruc.py reads it,
// else engine\nvoffruc) with its three DLLs; nativeFrucSetup loads the bridge from it
static bool lkOfflineFruc(const std::wstring& script, std::string& dir)
{
    dir = lkEnv("SMV_NVOFFRUC_DIR");
    if (dir.empty())
    {
        std::wstring eng = script.substr(0, script.find_last_of(L"\\/"));
        for (auto& c : eng) if (c == L'/') c = L'\\';
        dir = wideToUtf8(eng) + "\\nvoffruc";
    }
    for (const char* dll : { "nvoffruc_bridge.dll", "NvOFFRUC.dll", "cudart64_110.dll" })
        if (!lkFile(dir + "\\" + dll))
        { LOG("offline: Nvidia Smooth Motion needs %s in %s\n", dll, dir.c_str()); return false; }
    return true;
}

// The engine handoff: the session's geometry and engine paths, from the host's own lookup
// (nativeLocalHandoff), after a cold build on a worker thread when the lookup misses.
// Blocking (the caller is srv.start() on the loader thread or the early handoff thread).
// A build whose session was ended meanwhile (the hotkey toggled off, or the target window
// resized during the load) keeps running while the resident host stays alive
// (residentMain's stay rule counts it): its engines land in the cache, so the next start at
// that size is warm (user: "we don't need to kill previous sizes buildup"). A host that
// exits anyway ends the build with it; engine files are written whole or not at all
// (lkWriteFile). One builder at a time: a new handoff waits for a running one first.
struct BgHandoff
{
    std::atomic<bool> pending{ false };
};
static BgHandoff g_bgHandoff;
static bool nativeBuildPending() { return g_bgHandoff.pending.load(); }
// the session is ending (stdin "stop", or the load-time resize); polled by the handoff wait
static bool nativeLoadAbort() { return g_stopReq.load() || g_resizeReq.load(); }

static bool nativeHandoff(const std::wstring& script, const std::wstring& backend, int gen,
                          uint32_t capW, uint32_t capH, NativeRife& nr)
{
    // resident host: the same window as the previous session (same overlay size, capture
    // size, image scale and HDR mode) gets the previous answer without a lookup, as long as
    // the engine files still exist (the user may empty the cache folder by hand);
    // live Restore rides in the key: its engine path is a fact of a restore session only
    wchar_t key[1024];
    swprintf_s(key, L"%s|%s|%.2f|%u|%u|%u|%u|%d|%d", script.c_str(), backend.c_str(),
               g_flowScale, W, H, capW, capH, g_hdr ? 1 : 0, g_restore ? 1 : 0);
    bool factsOk = g_resident && g_res.haveFacts && g_res.handoffKey == key;
    if (factsOk)
    {
        const NativeRife& f = g_res.facts;
        if (f.gmfss) { for (int i = 0; i < 5; i++) factsOk = factsOk && fileExistsA(f.gmPath[i]); }
        else factsOk = fileExistsA(f.ifnetPath) && fileExistsA(f.encodePath);
        factsOk = factsOk && (f.restorePath.empty() || fileExistsA(f.restorePath));
        factsOk = factsOk && (!f.drba || fileExistsA(f.block0Path));
    }
    if (factsOk)
    {
        const NativeRife& f = g_res.facts;
        nr.ifnetPath = f.ifnetPath; nr.encodePath = f.encodePath; nr.jitPath = f.jitPath;
        nr.ejitPath = f.ejitPath;
        nr.restorePath = f.restorePath; nr.rjitPath = f.rjitPath; nr.restore = f.restore;
        nr.noEngine = f.noEngine; nr.cachePath = f.cachePath; nr.nvof = f.nvof;
        nr.fruc = f.fruc; nr.frucDir = f.frucDir;
        nr.drba = f.drba; nr.block0Path = f.block0Path; nr.block0Jit = f.block0Jit;
        nr.gmfss = f.gmfss; nr.hh = f.hh; nr.hw = f.hw;
        for (int i = 0; i < 5; i++) { nr.gmPath[i] = f.gmPath[i]; nr.gmJit[i] = f.gmJit[i]; }
        nr.ph = f.ph; nr.pw = f.pw; nr.w = f.w; nr.h = f.h; nr.cw = f.cw; nr.ch = f.ch;
        nr.dw = f.dw; nr.dh = f.dh; nr.x0 = f.x0; nr.y0 = f.y0;
        nr.batchMax = f.batchMax; nr.identity = f.identity; nr.fitAa = f.fitAa;
        nr.hdr = g_hdr;
        nativeDeriveUpscale(nr);   // from this session's own --upscale, never a stored fact
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
            if (nativeLoadAbort()) { LOG("native: session ended while waiting for the earlier engine build\n"); return false; }
            Sleep(50);
        }
    }
    std::string all;
    bool ready = false;
    // one handoff line of the lookup's answer
    auto onLine = [&](const std::string& line)
    {
        if (line.rfind("NATIVE-PATH ifnet=", 0) == 0) nr.ifnetPath = line.substr(18);
        else if (line.rfind("NATIVE-PATH encode=", 0) == 0) nr.encodePath = line.substr(19);
        else if (line.rfind("NATIVE-PATH jit=", 0) == 0) nr.jitPath = line.substr(16);
        else if (line.rfind("NATIVE-PATH ejit=", 0) == 0) nr.ejitPath = line.substr(17);
        else if (line.rfind("NATIVE-PATH restore=", 0) == 0) nr.restorePath = line.substr(20);
        else if (line.rfind("NATIVE-PATH rjit=", 0) == 0) nr.rjitPath = line.substr(17);
        else if (line.rfind("NATIVE-PATH cache=", 0) == 0) nr.cachePath = line.substr(18);
        else if (line.rfind("NATIVE-PATH fruc=", 0) == 0) nr.frucDir = line.substr(17);
        else if (line.rfind("NATIVE-PATH block0=", 0) == 0) nr.block0Path = line.substr(19);
        else if (line.rfind("NATIVE-PATH block0jit=", 0) == 0) nr.block0Jit = line.substr(22);
        else if (line.rfind("LIVE READY", 0) == 0) { all = line; ready = true; }
        else if (line.rfind("NATIVE-PATH g", 0) == 0)
        {
            // the GMFSS set: `NATIVE-PATH <key>=` and `NATIVE-PATH <key>jit=` per engine
            for (int i = 0; i < 5; i++)
            {
                const std::string kp = std::string("NATIVE-PATH ") + kGmKey[i] + "=";
                const std::string kj = std::string("NATIVE-PATH ") + kGmKey[i] + "jit=";
                if (line.rfind(kp, 0) == 0) nr.gmPath[i] = line.substr(kp.size());
                else if (line.rfind(kj, 0) == 0) nr.gmJit[i] = line.substr(kj.size());
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
            auto state = std::make_shared<std::atomic<int>>(0);   // 0 running, 1 built, 2 not
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
            for (const auto& l : local) { onLine(l); if (g_handoffDump) LOG("handoff-line %s\n", l.c_str()); }
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
    auto num = [&](const char* key, int dflt) -> int
    {
        const std::string k = std::string(" ") + key + "=";
        const size_t p = all.find(k);
        return p == std::string::npos ? dflt : atoi(all.c_str() + p + k.size());
    };
    nr.ph = num("ph", 0);   nr.pw = num("pw", 0);
    nr.w = num("w", 0);     nr.h = num("h", 0);
    nr.cw = num("cw", 0);   nr.ch = num("ch", 0);
    nr.dw = num("dw", 0);   nr.dh = num("dh", 0);
    nr.x0 = num("x0", 0);   nr.y0 = num("y0", 0);
    nr.batchMax = num("batch", 0);
    if (nr.batchMax < 1) nr.batchMax = 1;
    // no-engine mode: the echo handoff answers `engine=none` (geometry only, no paths); the
    // flow factor and the engine paths are not facts of such a session
    nr.noEngine = all.find(" engine=none") != std::string::npos;
    if (nr.noEngine && nr.cachePath.empty())
    { LOG("native: no-engine handoff named no cache folder\n"); return false; }
    // the nvof model: the same geometry-only answer with `engine=nvof` (no TensorRT engine; the
    // cache folder holds the kernel cubin)
    nr.nvof = all.find(" engine=nvof") != std::string::npos;
    if (nr.nvof && nr.cachePath.empty())
    { LOG("native: nvof handoff named no cache folder\n"); return false; }
    // Smooth Motion: the same geometry-only shape, plus the bridge folder
    nr.fruc = all.find(" engine=fruc") != std::string::npos;
    if (nr.fruc && (nr.cachePath.empty() || nr.frucDir.empty()))
    { LOG("native: fruc handoff named no cache or bridge folder\n"); return false; }
    // GMFSS (sub-step 5a): five engine paths, the half-frame grid, the cache folder for the
    // kernel cubin (no RIFE jit path names it)
    nr.gmfss = all.find(" engine=gmfss") != std::string::npos;
    if (nr.gmfss)
    {
        nr.hh = num("hh", 0);
        nr.hw = num("hw", 0);
        for (int i = 0; i < 5; i++)
            if (nr.gmPath[i].empty())
            { LOG("native: gmfss handoff named no %s engine\n", kGmName[i]); return false; }
        if (nr.hh <= 0 || nr.hw <= 0 || nr.cachePath.empty())
        { LOG("native: gmfss handoff line incomplete (hh / hw / cache)\n"); return false; }
    }
    // native DRBA: the RIFE handoff plus the block0 engine; the lag is one capture
    nr.drba = all.find(" engine=drba") != std::string::npos;
    if (nr.drba && (nr.block0Path.empty() || num("lag", 0) != 1))
    { LOG("native: drba handoff named no block0 engine or no lag=1\n"); return false; }
    const int outW = num("outw", 0), outH = num("outh", 0);
    nr.identity = (outW == nr.w && outH == nr.h);
    // the downscaling fit (2026-09-15): python antialiases whenever the fit height shrinks
    // (_Fit._upscale: antialias = dh < h); decided here so the early helper knows to run
    // nativeRtxInit (its staging buffers) before any effect flag is looked at. Upscale to
    // (item 2) re-derives it against the internal render size in nativeDeriveUpscale.
    nr.fitAa = nr.dh < nr.h;
    nr.hdr = g_hdr;
    nativeDeriveUpscale(nr);
    if (nr.ph <= 0 || nr.pw <= 0 || nr.w <= 0 || nr.h <= 0 || nr.cw <= 0 || nr.ch <= 0
        || (!nr.noEngine && !nr.gmfss && !nr.nvof && !nr.fruc && (nr.ifnetPath.empty() || nr.encodePath.empty())))
    { LOG("native: handoff line incomplete\n"); return false; }
    if (num("effects", 0))
    { LOG("native: live effects are on, phase 1 has no sharpen or VSR path\n"); return false; }
    // live Restore (2026-09-15): the session asked for it, so the handoff must have named the
    // engine (an eager restore pass, SMV_LIVE_TRT=0, stays a python-route session)
    nr.restore = !nr.restorePath.empty();
    if (g_restore && !nr.restore)
    { LOG("native: Restore is on but the handoff named no restore engine\n"); return false; }
    // a downscaling fit runs natively since 2026-09-15 (k_fitAaH / k_fitAaV, nativeRtxInit)
    if ((uint32_t)nr.cw != capW || (uint32_t)nr.ch != capH)
    { LOG("native: handoff capture size %dx%d != %ux%u\n", nr.cw, nr.ch, capW, capH); return false; }
    if (g_resident)
    {
        NativeRife& f = g_res.facts;
        f.ifnetPath = nr.ifnetPath; f.encodePath = nr.encodePath; f.jitPath = nr.jitPath;
        f.ejitPath = nr.ejitPath;
        f.restorePath = nr.restorePath; f.rjitPath = nr.rjitPath; f.restore = nr.restore;
        f.noEngine = nr.noEngine; f.cachePath = nr.cachePath; f.nvof = nr.nvof;
        f.fruc = nr.fruc; f.frucDir = nr.frucDir;
        f.drba = nr.drba; f.block0Path = nr.block0Path; f.block0Jit = nr.block0Jit;
        f.gmfss = nr.gmfss; f.hh = nr.hh; f.hw = nr.hw;
        for (int i = 0; i < 5; i++) { f.gmPath[i] = nr.gmPath[i]; f.gmJit[i] = nr.gmJit[i]; }
        f.ph = nr.ph; f.pw = nr.pw; f.w = nr.w; f.h = nr.h; f.cw = nr.cw; f.ch = nr.ch;
        f.dw = nr.dw; f.dh = nr.dh; f.x0 = nr.x0; f.y0 = nr.y0;
        f.batchMax = nr.batchMax; f.identity = nr.identity; f.fitAa = nr.fitAa;
        g_res.handoffKey = key;
        g_res.haveFacts = true;
    }
    return true;
}

static bool nativeReadFile(const std::string& path, std::vector<char>& out)
{
    FILE* f = nullptr;
    if (fopen_s(&f, path.c_str(), "rb") || !f) return false;
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
// maximum shape, 2 GB for the live IFNet class, and is what the resident host drops between
// sessions; the engine itself stays)
static bool nativeMakeContext(NativeRife& nr, nvinfer1::ICudaEngine* eng,
                              nvinfer1::IRuntimeConfig** cfgOut,
                              nvinfer1::IExecutionContext** ctxOut)
{
    nvinfer1::IRuntimeConfig* cfg = eng->createRuntimeConfig();
    if (!cfg) { LOG("native: createRuntimeConfig failed\n"); return false; }
    if (!nr.jit)
    {
        nr.jit = cfg->createRuntimeCache();
        std::vector<char> cb;
        if (nr.jit && !nr.jitPath.empty() && nativeReadFile(nr.jitPath, cb))
            LOG("native: jit cache deserialize -> %d (%zu bytes)\n",
                (int)nr.jit->deserialize(cb.data(), cb.size()), cb.size());
        // python keeps one cache file per engine, so the encoder's kernels sit in its own
        // .jit; merged into this cache the encode context takes 7 ms instead of recompiling
        // them for 0.5 s on every first session of a pair (2026-09-14, harness\r9g)
        if (nr.jit && !nr.ejitPath.empty() && nativeReadFile(nr.ejitPath, cb))
            LOG("native: encode jit cache merged -> %d (%zu bytes)\n",
                (int)nr.jit->deserialize(cb.data(), cb.size()), cb.size());
        // the restore engine's kernels too (live Restore, 2026-09-15), same reason
        if (nr.jit && !nr.rjitPath.empty() && nativeReadFile(nr.rjitPath, cb))
            LOG("native: restore jit cache merged -> %d (%zu bytes)\n",
                (int)nr.jit->deserialize(cb.data(), cb.size()), cb.size());
        // native DRBA's block0 engine (priority 21 (b)), same reason
        if (nr.jit && !nr.block0Jit.empty() && nativeReadFile(nr.block0Jit, cb))
            LOG("native: block0 jit cache merged -> %d (%zu bytes)\n",
                (int)nr.jit->deserialize(cb.data(), cb.size()), cb.size());
        // the GMFSS set's five caches (sub-step 5a), one file per engine on the python side
        for (int i = 0; i < 5; i++)
            if (nr.jit && !nr.gmJit[i].empty() && nativeReadFile(nr.gmJit[i], cb))
                LOG("native: gmfss %s jit cache merged -> %d (%zu bytes)\n", kGmName[i],
                    (int)nr.jit->deserialize(cb.data(), cb.size()), cb.size());
    }
    if (nr.jit) cfg->setRuntimeCache(*nr.jit);
    // WO-32: offline keeps TRT-RTX's graph capture OFF by default (P2: flat above 480p, one
    // cudaErrorInvalidValue in ten runs, the unproven TDR link); SMV_OFFLINE_GRAPH=1 turns it on
    bool wantGraph = !g_offline || g_offlineGraph;
    // SMV_LIVE_GMFSS_GRAPH=0: the GMFSS session's contexts without the whole-graph strategy
    // (a sub-step 5c diagnostic; the python route runs these five engines with it off and
    // wraps GMFlow in a torch CUDA graph instead)
    if (nr.gmfss)
    {
        wchar_t v[8]{};
        if (GetEnvironmentVariableW(L"SMV_LIVE_GMFSS_GRAPH", v, 8) && !wcscmp(v, L"0")) wantGraph = false;
    }
    if (wantGraph)
        cfg->setCudaGraphStrategy(nvinfer1::CudaGraphStrategy::kWHOLE_GRAPH_CAPTURE);
    nvinfer1::IExecutionContext* ctx = eng->createExecutionContext(cfg);
    if (!ctx) { LOG("native: createExecutionContext failed\n"); delete cfg; return false; }
    *cfgOut = cfg;
    *ctxOut = ctx;
    return true;
}

// exact sequence from NATIVE-HOST-API-MAP.md (a): validity pre-check, deserialize, then the
// config + context through nativeMakeContext.
static nvinfer1::ICudaEngine* nativeLoadEngine(NativeRife& nr, const std::string& path,
                                              nvinfer1::IRuntimeConfig** cfgOut,
                                              nvinfer1::IExecutionContext** ctxOut)
{
    std::vector<char> blob;
    if (!nativeReadFile(path, blob)) { LOG("native: cannot read %s\n", path.c_str()); return nullptr; }
    uint64_t diag = 0;
    const auto v = nr.rt->getEngineValidity(blob.data(), blob.size(), &diag);
    if (v == nvinfer1::EngineValidity::kINVALID)
    { LOG("native: engine rejected by getEngineValidity (reasons 0x%llx): %s\n",
          (unsigned long long)diag, path.c_str()); return nullptr; }
    if (v == nvinfer1::EngineValidity::kSUBOPTIMAL)
        LOG("native: engine reports kSUBOPTIMAL (reasons 0x%llx), continuing\n",
            (unsigned long long)diag);
    nvinfer1::ICudaEngine* eng = nr.rt->deserializeCudaEngine(blob.data(), blob.size());
    if (!eng) { LOG("native: deserializeCudaEngine failed for %s\n", path.c_str()); return nullptr; }
    if (!nativeMakeContext(nr, eng, cfgOut, ctxOut)) { delete eng; return nullptr; }
    return eng;
}

// NVRTC once, then the cubin is cached on disk next to the TRT cache, keyed by source hash and
// device arch (no nvcc anywhere, per the WO).
static bool nativeBindKernels(NativeRife& nr)
{
    struct { CUfunction* fn; const char* nm; } fns[] = {
        { &nr.fPackInDirect, "k_packInDirect" }, { &nr.fResizeH, "k_resizeH" },
        { &nr.fResizeV, "k_resizeV" }, { &nr.fH2f, "k_h2f" }, { &nr.fPackOut, "k_packOut" },
        { &nr.fPackInDirectHdr, "k_packInDirectHdr" }, { &nr.fPqPlanar, "k_pqPlanar" },
        { &nr.fResizeHf, "k_resizeHf" }, { &nr.fPackOutHdr, "k_packOutHdr" },
        { &nr.fSdrEncode, "k_sdrEncode" }, { &nr.fThdrColor, "k_thdrColor" },
        { &nr.fThdrIn, "k_thdrIn" }, { &nr.fRcasThdrIn, "k_rcasThdrIn" },
        { &nr.fThdrOut, "k_thdrOut" }, { &nr.fPqLut, "k_pqLut" },
        { &nr.fPackBgra, "k_packBgra" }, { &nr.fUnpackBgra, "k_unpackBgra" },
        { &nr.fFitPlanar, "k_fitPlanar" }, { &nr.fRcasOut, "k_rcasOut" },
        { &nr.fRcasOutHdr, "k_rcasOutHdr" }, { &nr.fRcasOutRaw, "k_rcasOutRaw" },
        { &nr.fPackBgraRgb, "k_packBgraRgb" }, { &nr.fUnpackBgraRgb, "k_unpackBgraRgb" },
        { &nr.fUnpackRgba, "k_unpackRgba" },
        { &nr.fFitAaH, "k_fitAaH" }, { &nr.fFitAaV, "k_fitAaV" },
        { &nr.fRestIn, "k_restIn" }, { &nr.fRestFoldH, "k_restFoldH" }, { &nr.fRestFoldV, "k_restFoldV" },
        { &nr.fRestToF, "k_restToF" }, { &nr.fClamp01, "k_clamp01" },
        { &nr.fNrIn, "k_nrIn" }, { &nr.fNrOut, "k_nrOut" },
        { &nr.fHalf, "k_half" }, { &nr.fPyr, "k_pyr" },
        { &nr.fSplatSoft, "k_splatSoft" }, { &nr.fSplatNorm, "k_splatNorm" },
        { &nr.fPackInRaw16, "k_packInRaw16" }, { &nr.fPackInRaw8, "k_packInRaw8" },
        { &nr.fPackOutRaw16, "k_packOutRaw16" }, { &nr.fPackOutRaw8, "k_packOutRaw8" },
        { &nr.fExpand8to16, "k_expand8to16" }, { &nr.fPairDiff, "k_pairDiff" },
        { &nr.fNvofLuma, "k_nvofLuma" }, { &nr.fNvofUp, "k_nvofUp" },
        { &nr.fNvofMetric, "k_nvofMetric" },
        { &nr.fSplatVel, "k_splatVel" }, { &nr.fVelNorm, "k_velNorm" }, { &nr.fPpDown, "k_ppDown" },
        { &nr.fPpTop, "k_ppTop" }, { &nr.fPpUp, "k_ppUp" }, { &nr.fBlur1, "k_blur1" },
        { &nr.fNvofCompose, "k_nvofCompose" },
        { &nr.fDrFlowSplat, "k_drbaFlowSplat" }, { &nr.fDrFlowNorm, "k_drbaFlowNorm" },
        { &nr.fDrDrmSplat, "k_drbaDrmSplat" }, { &nr.fDrDrmNorm, "k_drbaDrmNorm" },
    };
    for (auto& e : fns)
        if (cuModuleGetFunction(e.fn, nr.cuMod, e.nm) != CUDA_SUCCESS)
        { LOG("native: kernel %s missing\n", e.nm); return false; }
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
    for (const char* p = kNativeKernels; *p; p++) { hash ^= (unsigned char)*p; hash *= 1099511628211ull; }
    wchar_t nm[128];
    swprintf_s(nm, L"\\smv_native_%016llx_sm%d%d.cubin",
               (unsigned long long)hash, prop.major, prop.minor);
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
            if (n > 0) { cubin.resize((size_t)n); haveCubin = fread(cubin.data(), 1, (size_t)n, f) == (size_t)n; }
            fclose(f);
        }
    }
    if (!haveCubin)
    {
        NvrtcApi nv;
        nv.mod = GetModuleHandleW(L"nvrtc64_130_0.dll");
        if (!nv.mod) { LOG("native: nvrtc not loaded\n"); return false; }
        nv.create = (PFN_nvrtcCreateProgram)GetProcAddress(nv.mod, "nvrtcCreateProgram");
        nv.compile = (PFN_nvrtcCompileProgram)GetProcAddress(nv.mod, "nvrtcCompileProgram");
        nv.cubinSize = (PFN_nvrtcGetCUBINSize)GetProcAddress(nv.mod, "nvrtcGetCUBINSize");
        nv.cubin = (PFN_nvrtcGetCUBIN)GetProcAddress(nv.mod, "nvrtcGetCUBIN");
        nv.logSize = (PFN_nvrtcGetProgramLogSize)GetProcAddress(nv.mod, "nvrtcGetProgramLogSize");
        nv.log = (PFN_nvrtcGetProgramLog)GetProcAddress(nv.mod, "nvrtcGetProgramLog");
        nv.destroy = (PFN_nvrtcDestroyProgram)GetProcAddress(nv.mod, "nvrtcDestroyProgram");
        if (!nv.ok()) { LOG("native: nvrtc entry points missing\n"); return false; }
        void* prog = nullptr;
        if (nv.create(&prog, kNativeKernels, "smv_native.cu", 0, nullptr, nullptr) != 0)
        { LOG("native: nvrtcCreateProgram failed\n"); return false; }
        char arch[64];
        sprintf_s(arch, "--gpu-architecture=sm_%d%d", prop.major, prop.minor);
        const char* opts[] = { arch, "--use_fast_math=false", "-default-device" };
        const int crc = nv.compile(prog, 1, opts);   // arch only; the other two are informational
        if (crc != 0)
        {
            size_t ls = 0;
            std::string lg;
            if (nv.logSize && !nv.logSize(prog, &ls) && ls > 1) { lg.resize(ls); nv.log(prog, lg.data()); }
            LOG("native: nvrtc compile failed (%d): %s\n", crc, lg.c_str());
            nv.destroy(&prog);
            return false;
        }
        size_t cs = 0;
        if (nv.cubinSize(prog, &cs) != 0 || !cs) { LOG("native: nvrtcGetCUBINSize failed\n"); nv.destroy(&prog); return false; }
        cubin.resize(cs);
        if (nv.cubin(prog, cubin.data()) != 0) { LOG("native: nvrtcGetCUBIN failed\n"); nv.destroy(&prog); return false; }
        nv.destroy(&prog);
        FILE* f = nullptr;
        const std::wstring tmp = cubinPath + L".tmp";
        if (!_wfopen_s(&f, tmp.c_str(), L"wb") && f)
        {
            fwrite(cubin.data(), 1, cubin.size(), f);
            fclose(f);
            _wunlink(cubinPath.c_str());
            _wrename(tmp.c_str(), cubinPath.c_str());
        }
        LOG("native: kernels compiled for sm_%d%d (%zu bytes cubin)\n", prop.major, prop.minor, cubin.size());
    }
    if (cuInit(0) != CUDA_SUCCESS) { LOG("native: cuInit failed\n"); return false; }
    if (cuModuleLoadData(&nr.cuMod, cubin.data()) != CUDA_SUCCESS)
    { LOG("native: cuModuleLoadData failed\n"); return false; }
    if (!nativeBindKernels(nr)) return false;
    if (g_resident) g_res.cuMod = nr.cuMod;
    return true;
}
// ---- part 3: CUDA init, the compute thread, shutdown --------------------------------------

// python inverts the two ICtCp matrices in float64 and casts the result to fp32
// (rtxvideo.py ~226), so the inverse is computed in double here and uploaded as fp32.
static bool inv3d(const double* m, float* out)
{
    const double d = m[0] * (m[4] * m[8] - m[5] * m[7])
                   - m[1] * (m[3] * m[8] - m[5] * m[6])
                   + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if (d == 0.0) return false;
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

// The native init is split in three (startup step 2, 2026-09-12) so that everything except
// the output ring can run BEFORE the source-rate measurement (the ring is sized from the
// measured slot count):
//  * nativeCudaDeviceInit: device, context, stream, capture texture + fence import. Depends on
//    the capture size only, so the early thread runs it in parallel with the python handoff
//    (it writes only dev / stream / emCap / capMip / capArr / semCap, the handoff writes the
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
static bool nativeCudaDeviceInit(NativeRife& nr, IDXGIAdapter1* adapter, HANDLE hTex, HANDLE hFence,
                                 uint32_t capW, uint32_t capH, bool hdr)
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
    // WO-8 capture format: BGRA8 in SDR, R16G16B16A16_FLOAT scRGB in HDR (8 bytes per pixel)
    const int capBpp = hdr ? 8 : 4;
    md.size = (size_t)capW * capH * capBpp;
    md.flags = cudaExternalMemoryDedicated;
    NCHK(cudaImportExternalMemory(&nr.emCap, &md), "import capture texture");
    cudaExternalMemoryMipmappedArrayDesc ad{};
    ad.offset = 0;
    // exactly what _CapTex asks cudart for: uchar4 in SDR, half4 with channel kind FLOAT in
    // HDR (PG p.143's sample table has no float16 four-channel case; the working half4 import is
    // the truth, do not "fix" it to match the table)
    ad.formatDesc = hdr ? cudaChannelFormatDesc{ 16, 16, 16, 16, cudaChannelFormatKindFloat }
                        : cudaChannelFormatDesc{ 8, 8, 8, 8, cudaChannelFormatKindUnsigned };
    ad.extent = cudaExtent{ (size_t)capW, (size_t)capH, 0 };
    ad.flags = 0;
    ad.numLevels = 1;
    if (cudaExternalMemoryGetMappedMipmappedArray(&nr.capMip, nr.emCap, &ad) != cudaSuccess)
    {
        ad.flags = cudaArrayColorAttachment;
        NCHK(cudaExternalMemoryGetMappedMipmappedArray(&nr.capMip, nr.emCap, &ad),
             "map capture mipmapped array");
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
    NCHK(cudaExternalMemoryGetMappedBuffer((void**)&nr.dOutRing, nr.emOut, &bd),
         "map output buffer");
    // letterbox bars and alpha, written once: opaque black is 0xFF000000 in BGRA8 and
    // 0xC0000000 in R10G10B10A2 with A = 3 (python's -1073741824 canvas fill)
    if (cuMemsetD32Async((CUdeviceptr)nr.dOutRing, nr.hdr ? 0xC0000000u : 0xFF000000u,
                         (size_t)(outBytes / 4), (CUstream)nr.stream) != CUDA_SUCCESS)
    { LOG("native: output ring clear failed\n"); return false; }
    nr.slotEv.resize(nr.slots);
    for (auto& e : nr.slotEv)
        NCHK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "create slot event");
    return true;
}

// WO-23 live TrueHDR setup: the bridge buffers, the ICtCp matrices, the NGX feature and its
// warm-up eval (~0.6 s). Needs the capture size, the stream and the kernel module only, so the
// early thread runs it on a helper beside the engine load (todo 9d, 2026-09-13); the non-early
// compute thread runs it after the engine load. The caller's thread must have bound the device.
static bool nativeRtxInit(NativeRife& nr)
{
    // live Sharpen and RTX VSR (2026-09-15): the fit is known now, so the python rule "VSR
    // only when the resize enlarges in both axes" (_Fit._setup_resize) decides here
    // Upscale to (item 2): the first resize target is the internal render size when set, else
    // the fit rect (_Fit._setup_resize); VSR and the aa rule read that target
    const int tw = nr.uw ? nr.uw : nr.dw, th = nr.uw ? nr.uh : nr.dh;
    if (nr.vsrWant)
    {
        if (tw > nr.w && th > nr.h) nr.vsr = true;
        else LOG("native: live RTX VSR skipped (upscales only; this resize does not enlarge), bicubic\n");
    }
    if (nr.uw)
    {
        NCHK(cudaMalloc((void**)&nr.dUp, (size_t)3 * nr.uw * nr.uh * sizeof(float)), "alloc internal render frame");
        if (nr.upAa)
            NCHK(cudaMalloc((void**)&nr.dUpTmp, (size_t)3 * nr.uw * nr.h * sizeof(float)), "alloc internal render pass");
        LOG("native: live upscale to: model %dx%d -> %dx%d first, then %s to %dx%d\n",
            nr.w, nr.h, nr.uw, nr.uh, nr.identity ? "1:1" : "fit", nr.dw, nr.dh);
    }
    // the downscaling fit (2026-09-15, nr.fitAa decided in the handoff parse): the horizontal
    // pass buffer and the staging frame, the same log line as python's `fit:` one; its source
    // is the internal render frame under Upscale to
    if (nr.fitAa)
    {
        NCHK(cudaMalloc((void**)&nr.dFitTmp, (size_t)3 * nr.dw * (nr.uw ? nr.uh : nr.h) * sizeof(float)), "alloc fit pass");
        LOG("native: fit: %dx%d -> %dx%d, antialiased bicubic (downscale)\n",
            nr.uw ? nr.uw : nr.w, nr.uw ? nr.uh : nr.h, nr.dw, nr.dh);
    }
    if (nr.sharpen > 0.0f || nr.fitAa || nr.restore)
        NCHK(cudaMalloc((void**)&nr.dPres, (size_t)3 * nr.dw * nr.dh * sizeof(float)), "alloc fit staging");
    if (nr.sharpen > 0.0f)
        LOG("native: live sharpen: FSR RCAS %.2f at the presented resolution\n", nr.sharpen);
    // live Restore (2026-09-15): the fold target is _Fit._load_restore's restore_target: back
    // to the model size when an RTX VSR instance follows (so VSR sees the restored frame),
    // else the first resize target directly (restore-as-upscaler, one resize)
    if (nr.restore)
    {
        nr.restTw = nr.vsr ? nr.w : tw;
        nr.restTh = nr.vsr ? nr.h : th;
        if (nr.restTh > 4 * nr.h)
            NCHK(cudaMalloc((void**)&nr.dRestF, (size_t)3 * 16 * nr.w * nr.h * sizeof(float)), "alloc restore fp32 output");
        LOG("native: live restore: Real-ESRGAN animevideov3 (TensorRT) at %dx%d -> %dx%d\n",
            nr.w, nr.h, nr.restTw, nr.restTh);
    }
    if (nr.vsr)
    {
        NCHK(cudaMalloc((void**)&nr.dVsrIn, (size_t)nr.w * nr.h * 4), "alloc VSR input");
        NCHK(cudaMalloc((void**)&nr.dVsrOut, (size_t)tw * th * 4), "alloc VSR output");
    }
    // live TrueHDR runs once per captured frame at the capture size, offline (priority 24
    // step 2c) on every output frame at the output size, last (rtxvideo.run_hdr)
    const int hw = g_offline ? nr.dw : nr.cw, hh = g_offline ? nr.dh : nr.ch;
    if (nr.rtxHdr)
    {
        NCHK(cudaMalloc((void**)&nr.dThdrIn, (size_t)hw * hh * 4), "alloc TrueHDR input");
        NCHK(cudaMalloc((void**)&nr.dThdrOut, (size_t)hw * hh * 4), "alloc TrueHDR output");
        NCHK(cudaMalloc((void**)&nr.dSrcG, (size_t)3 * hw * hh * sizeof(float)),
             "alloc source gamma planes");
        if (g_offline)
        {
            NCHK(cudaMalloc((void**)&nr.dThdrStats, kThdrStatsBytes), "alloc TrueHDR stats");
            NCHK(cudaHostAlloc((void**)&nr.hThdrStats, kThdrStatsBytes, cudaHostAllocDefault), "alloc TrueHDR stats copy");
            if (nr.thdrAcc)
            {
                // the _pq_lut of _accum_hp and its 100-nit code, from the kernels' own EOTF
                float* dLut = nullptr;
                NCHK(cudaMalloc((void**)&dLut, 1024 * sizeof(float)), "alloc PQ table");
                void* al[] = { &dLut };
                const bool lok = cuLaunchKernel(nr.fPqLut, 4, 1, 1, 256, 1, 1, 0, (CUstream)nr.stream, al, nullptr) == CUDA_SUCCESS
                    && cudaMemcpyAsync(nr.thdrAcc->lut, dLut, sizeof(nr.thdrAcc->lut), cudaMemcpyDeviceToHost, nr.stream) == cudaSuccess
                    && cudaStreamSynchronize(nr.stream) == cudaSuccess;
                cudaFree(dLut);
                if (!lok) { LOG("native: the PQ table failed\n"); return false; }
                nr.thdrAcc->brightCode = 0;
                while (nr.thdrAcc->brightCode < 1024 && !(nr.thdrAcc->lut[nr.thdrAcc->brightCode] > 0.01f))
                    nr.thdrAcc->brightCode++;
            }
        }
        // the two numerically inverted ICtCp matrices, into the kernel module's globals
        const double rgb2lms[9] = { 1688.0 / 4096.0, 2146.0 / 4096.0, 262.0 / 4096.0,
                                     683.0 / 4096.0, 2951.0 / 4096.0, 462.0 / 4096.0,
                                      99.0 / 4096.0,  309.0 / 4096.0, 3688.0 / 4096.0 };
        const double lms2ictcp[9] = { 2048.0 / 4096.0, 2048.0 / 4096.0, 0.0,
                                      6610.0 / 4096.0, -13613.0 / 4096.0, 7003.0 / 4096.0,
                                     17933.0 / 4096.0, -17390.0 / 4096.0, -543.0 / 4096.0 };
        float lms2rgb[9], ictcp2lms[9];
        if (!inv3d(rgb2lms, lms2rgb) || !inv3d(lms2ictcp, ictcp2lms))
        { LOG("native: ICtCp matrix inversion failed\n"); return false; }
        CUdeviceptr gp = 0;
        size_t gsz = 0;
        if (cuModuleGetGlobal(&gp, &gsz, nr.cuMod, "g_lms2rgb") != CUDA_SUCCESS
            || cuMemcpyHtoD(gp, lms2rgb, sizeof(lms2rgb)) != CUDA_SUCCESS
            || cuModuleGetGlobal(&gp, &gsz, nr.cuMod, "g_ictcp2lms") != CUDA_SUCCESS
            || cuMemcpyHtoD(gp, ictcp2lms, sizeof(ictcp2lms)) != CUDA_SUCCESS)
        { LOG("native: uploading the ICtCp matrices failed\n"); return false; }
    }
    if (nr.rtxHdr || nr.vsr)
    {
        // the bridge retains the PRIMARY context (cuContext = NULL), which is the context this
        // thread just bound with cudaSetDevice. cuStream is NULL exactly as rtxvideo.py passes
        // it: the bridge wraps every eval in host-synchronous cuMemcpy2D calls on the legacy
        // default stream, and the host's own stream is cudaStreamNonBlocking, so an NGX eval
        // placed on it would not be ordered against the bridge's own output copy. ONE create
        // per process (NGX is single-instance), so both features are asked for at once.
        if (g_rtxb.create(nullptr, nullptr, 0, nr.rtxHdr ? 1u : 0u, nr.vsr ? 1u : 0u) != 1u)
        { LOG("native: rtx_video_api_cuda_create failed\n"); return false; }
        g_rtxb.created = true;
        g_rtxUsed = true;   // resident host: NGX is single-instance per process, end it after this session
    }
    if (nr.vsr)
    {
        // warm-up eval on an opaque black frame, so the first presented frame pays nothing
        if (cuMemsetD32Async((CUdeviceptr)nr.dVsrIn, 0xFF000000u,
                             (size_t)nr.w * nr.h, (CUstream)nr.stream) != CUDA_SUCCESS)
        { LOG("native: RTX VSR warm-up clear failed\n"); return false; }
        NCHK(cudaStreamSynchronize(nr.stream), "VSR warm-up stream sync");
        const RtxRect ri{ 0, 0, (uint32_t)nr.w, (uint32_t)nr.h };
        const RtxRect ro{ 0, 0, (uint32_t)tw, (uint32_t)th };
        if (g_rtxb.evalVsr(nr.dVsrIn, nr.dVsrOut, ri, ro, &nr.vsrSet) != 1u)
        { LOG("native: the RTX VSR warm-up eval failed\n"); return false; }
        NCHK(cudaDeviceSynchronize(), "VSR warm-up eval sync");
        LOG("native: live upscale: RTX VSR %dx%d -> %dx%d\n", nr.w, nr.h, tw, th);
    }
    if (nr.rtxHdr)
    {
        // warm-up eval on a zero frame (opaque black BGRA), python live_server.py ~1859
        if (cuMemsetD32Async((CUdeviceptr)nr.dThdrIn, 0xFF000000u,
                             (size_t)hw * hh, (CUstream)nr.stream) != CUDA_SUCCESS)
        { LOG("native: TrueHDR warm-up clear failed\n"); return false; }
        NCHK(cudaStreamSynchronize(nr.stream), "warm-up stream sync");
        const RtxRect wr{ 0, 0, (uint32_t)hw, (uint32_t)hh };
        if (g_rtxb.evalThdr(nr.dThdrIn, nr.dThdrOut, wr, wr, &nr.thdr) != 1u)
        { LOG("native: the TrueHDR warm-up eval failed\n"); return false; }
        NCHK(cudaDeviceSynchronize(), "warm-up eval sync");
        if (g_offline)
            LOG("native: RTX TrueHDR on: %dx%d per output frame, colour %ls, contrast %u saturation %u, %u nits\n",
                hw, hh, g_hdrColor, nr.thdr.Contrast, nr.thdr.Saturation, nr.thdr.MaxLuminance);
        else
            LOG("native: live RTX TrueHDR on: %dx%d per real frame, colour %ls, SDR white %.0f nits,"
                " contrast %u saturation %u\n",
                nr.cw, nr.ch, g_hdrColor, nr.sdrScale * 80.0f, nr.thdr.Contrast, nr.thdr.Saturation);
    }
    return true;
}

// ---- the nvof model (2026-09-21, memory priority 20 step 3) --------------------------------
// The NVIDIA Optical Flow Accelerator through the driver's nvofapi64.dll (System32, opened by
// full path, never bundled; the process keeps it loaded once opened). One session per NativeRife
// at the true model size w x h: grid 4, preset fast, BOTH directions in one Execute, output cost
// on, temporal hints off (a paused or seeked source is not a continuous sequence). The defaults
// below are the step 2 sweep's pick (WO block "Step 2 results": the metric weights barely move
// the result, gray input was best on the in-range clip). Every failure here is a REFUSAL.
static NV_OF_CUDA_API_FUNCTION_LIST g_nvofApi{};
static bool g_nvofLoaded = false;
static const float kNvofA = 0.1f, kNvofB = 1.0f;
// the E2 tween's fallback: log of the mean landing weight at or above kNvofHi keeps the warp,
// at or below kNvofLo takes the plain blend (linear between), the mask blurred by kNvofSigma px
// (the fix round's E2 pick, WO block "Fix round", 2026-09-21)
static const float kNvofLo = -9.0f, kNvofHi = -5.0f, kNvofSigma = 6.0f;
static const int kNvofGrid = 4;

static bool nativeNvofLoad()
{
    if (g_nvofLoaded) return true;
    wchar_t sys[MAX_PATH]{};
    const UINT n = GetSystemDirectoryW(sys, MAX_PATH);
    if (!n || n >= MAX_PATH) { LOG("native: nvof: the system folder is unknown\n"); return false; }
    const std::wstring p = std::wstring(sys) + L"\\nvofapi64.dll";
    HMODULE m = LoadLibraryExW(p.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!m)
    { LOG("native: nvof: nvofapi64.dll not found in the system folder (err %lu); the NVIDIA display driver installs it\n", GetLastError()); return false; }
    typedef NV_OF_STATUS(NVOFAPI* PFN_CreateInstanceCuda)(uint32_t, NV_OF_CUDA_API_FUNCTION_LIST*);
    auto create = (PFN_CreateInstanceCuda)GetProcAddress(m, "NvOFAPICreateInstanceCuda");
    if (!create) { LOG("native: nvof: NvOFAPICreateInstanceCuda is not exported by nvofapi64.dll\n"); return false; }
    const NV_OF_STATUS s = create(NV_OF_API_VERSION, &g_nvofApi);
    if (s != NV_OF_SUCCESS)
    { LOG("native: nvof: NvOFAPICreateInstanceCuda(0x%x) refused (status %d), driver too old for the Optical Flow SDK 5 interface\n", (unsigned)NV_OF_API_VERSION, (int)s); return false; }
    g_nvofLoaded = true;
    return true;
}

static bool nativeNvofBuf(NativeRife& nr, uint32_t w, uint32_t h, NV_OF_BUFFER_USAGE usage,
                          NV_OF_BUFFER_FORMAT fmt, NvOFGPUBufferHandle& hb, CUdeviceptr& ptr,
                          uint32_t& pitch, const char* what)
{
    NV_OF_BUFFER_DESCRIPTOR d{};
    d.width = w;
    d.height = h;
    d.bufferUsage = usage;
    d.bufferFormat = fmt;
    NV_OF_STATUS s = g_nvofApi.nvOFCreateGPUBufferCuda(nr.ofH, &d, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &hb);
    if (s != NV_OF_SUCCESS) { LOG("native: nvof: %s buffer creation failed (status %d)\n", what, (int)s); return false; }
    ptr = g_nvofApi.nvOFGPUBufferGetCUdeviceptr(hb);
    NV_OF_CUDA_BUFFER_STRIDE_INFO si{};
    s = g_nvofApi.nvOFGPUBufferGetStrideInfo(hb, &si);
    if (s != NV_OF_SUCCESS || !ptr) { LOG("native: nvof: %s buffer stride query failed (status %d)\n", what, (int)s); return false; }
    pitch = si.strideInfo[0].strideXInBytes;
    return true;
}

// the session and the glue buffers; the calling thread has the device bound
static bool nativeNvofSetup(NativeRife& nr)
{
    if (!nativeNvofLoad()) return false;
    if (nr.w < 32 || nr.h < 32)
    { LOG("native: nvof: the model frame %dx%d is below the Optical Flow minimum 32x32\n", nr.w, nr.h); return false; }
    CUcontext ctx = nullptr;
    if (cuCtxGetCurrent(&ctx) != CUDA_SUCCESS || !ctx) { LOG("native: nvof: no current CUDA context\n"); return false; }
    NV_OF_STATUS s = g_nvofApi.nvCreateOpticalFlowCuda(ctx, &nr.ofH);
    if (s != NV_OF_SUCCESS) { nr.ofH = nullptr; LOG("native: nvof: nvCreateOpticalFlowCuda failed (status %d)\n", (int)s); return false; }
    NV_OF_INIT_PARAMS ip{};
    ip.width = (uint32_t)nr.w;
    ip.height = (uint32_t)nr.h;
    ip.outGridSize = (NV_OF_OUTPUT_VECTOR_GRID_SIZE)kNvofGrid;
    ip.hintGridSize = NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED;
    ip.mode = NV_OF_MODE_OPTICALFLOW;
    ip.perfLevel = NV_OF_PERF_LEVEL_FAST;
    ip.enableExternalHints = NV_OF_FALSE;
    ip.enableOutputCost = NV_OF_TRUE;
    ip.disparityRange = NV_OF_STEREO_DISPARITY_RANGE_UNDEFINED;
    ip.enableRoi = NV_OF_FALSE;
    ip.predDirection = NV_OF_PRED_DIRECTION_BOTH;
    ip.enableGlobalFlow = NV_OF_FALSE;
    ip.inputBufferFormat = NV_OF_BUFFER_FORMAT_GRAYSCALE8;
    s = g_nvofApi.nvOFInit(nr.ofH, &ip);
    if (s != NV_OF_SUCCESS) { LOG("native: nvof: nvOFInit %dx%d grid %d refused (status %d)\n", nr.w, nr.h, kNvofGrid, (int)s); return false; }
    // the output grid is ceil(size / grid): verified at 1914x1078 -> 479x270 (step 2)
    nr.ofGw = (nr.w + kNvofGrid - 1) / kNvofGrid;
    nr.ofGh = (nr.h + kNvofGrid - 1) / kNvofGrid;
    for (int k = 0; k < 2; k++)
    {
        uint32_t inPitch = 0;
        if (!nativeNvofBuf(nr, nr.w, nr.h, NV_OF_BUFFER_USAGE_INPUT, NV_OF_BUFFER_FORMAT_GRAYSCALE8,
                           nr.ofIn[k], nr.ofInP[k], inPitch, "input")
            || !nativeNvofBuf(nr, nr.ofGw, nr.ofGh, NV_OF_BUFFER_USAGE_OUTPUT, NV_OF_BUFFER_FORMAT_SHORT2,
                              nr.ofOut[k], nr.ofOutP[k], nr.ofOutPitch[k], "flow")
            || !nativeNvofBuf(nr, nr.ofGw, nr.ofGh, NV_OF_BUFFER_USAGE_COST, NV_OF_BUFFER_FORMAT_UINT8,
                              nr.ofCost[k], nr.ofCostP[k], nr.ofCostPitch[k], "cost"))
            return false;
        if (k == 0) nr.ofInPitch = inPitch;
        else if (inPitch != nr.ofInPitch) { LOG("native: nvof: the two input buffers differ in pitch\n"); return false; }
    }
    s = g_nvofApi.nvOFSetIOCudaStreams(nr.ofH, (CUstream)nr.stream, (CUstream)nr.stream);
    if (s != NV_OF_SUCCESS) { LOG("native: nvof: nvOFSetIOCudaStreams failed (status %d)\n", (int)s); return false; }
    const size_t mp = (size_t)nr.w * nr.h;
    for (int k = 0; k < 2; k++)
    {
        NCHK(cudaMalloc((void**)&nr.dNvFlow[k], 2 * mp * sizeof(float)), "alloc nvof flow");
        NCHK(cudaMalloc((void**)&nr.dNvCost[k], mp * sizeof(float)), "alloc nvof cost");
        NCHK(cudaMalloc((void**)&nr.dNvZ[k], mp * sizeof(float)), "alloc nvof metric");
    }
    NCHK(cudaMalloc((void**)&nr.dNvAcc, 5 * mp * sizeof(long long)), "alloc nvof accumulator");
    // the E2 tween's planes and the push-pull pyramid (levels halve, ceil, until the short
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
        NativeRife::NvLevel L{ lw, lh, nullptr, nullptr, nullptr };
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
    nr.nvProf = GetEnvironmentVariableW(L"SMV_LIVE_NVOF_PROF", nullptr, 0) != 0;   // the GMFSS lever's rule
    if (nr.nvProf)
    {
        NCHK(cudaEventCreate(&nr.nvEv[0]), "create nvof event");
        NCHK(cudaEventCreate(&nr.nvEv[1]), "create nvof event");
    }
    LOG("native: nvof session %dx%d, grid %d (%dx%d), fast, both directions, gray8\n",
        nr.w, nr.h, kNvofGrid, nr.ofGw, nr.ofGh);
    return true;
}

// the pair: both frames to gray8, one Execute for both fields, both upsampled, both metrics.
// dPrev / dCur are the model planes (B, G, R) at (ph * pw, pw) strides.
static bool nativeNvofPair(NativeRife& nr, const float* dPrev, const float* dCur)
{
    cudaStream_t st = nr.stream;
    const size_t plane = (size_t)nr.ph * nr.pw;
    if (nr.nvProf) cudaEventRecord(nr.nvEv[0], st);
    // BT.709 on (R, G, B): live planes are (B, G, R), so the R plane goes first with a negative
    // plane stride (k_nvofLuma's note); offline planes are already (R, G, B) (planesRgb)
    int nps = nr.planesRgb ? (int)plane : -(int)plane, pitch = (int)nr.ofInPitch;
    for (int k = 0; k < 2; k++)
    {
        const float* r = (k ? dCur : dPrev) + (nr.planesRgb ? 0 : 2 * plane);
        CUdeviceptr dst = nr.ofInP[k];
        void* a[] = { (void*)&r, &nps, &nr.pw, &nr.w, &nr.h, &dst, &pitch };
        if (cuLaunchKernel(nr.fNvofLuma, (nr.w + 15) / 16, (nr.h + 15) / 16, 1, 16, 16, 1, 0,
                           (CUstream)st, a, nullptr) != CUDA_SUCCESS)
        { nr.die("nvofLuma launch failed"); return false; }
    }
    NV_OF_EXECUTE_INPUT_PARAMS ei{};
    ei.inputFrame = nr.ofIn[0];       // frame 0 -> "inputFrame": the forward field is F01
    ei.referenceFrame = nr.ofIn[1];
    ei.disableTemporalHints = NV_OF_TRUE;
    NV_OF_EXECUTE_OUTPUT_PARAMS eo{};
    eo.outputBuffer = nr.ofOut[0];
    eo.outputCostBuffer = nr.ofCost[0];
    eo.bwdOutputBuffer = nr.ofOut[1];
    eo.bwdOutputCostBuffer = nr.ofCost[1];
    const NV_OF_STATUS s = g_nvofApi.nvOFExecute(nr.ofH, &ei, &eo);
    if (s != NV_OF_SUCCESS) { LOG("native: nvof: nvOFExecute failed (status %d)\n", (int)s); nr.die("nvof execute failed"); return false; }
    int grid = kNvofGrid;
    for (int k = 0; k < 2; k++)
    {
        CUdeviceptr v = nr.ofOutP[k], c = nr.ofCostP[k];
        int vp = (int)nr.ofOutPitch[k], cp = (int)nr.ofCostPitch[k];
        void* a[] = { &v, &vp, &c, &cp, &nr.ofGw, &nr.ofGh, &grid, &nr.w, &nr.h,
                      &nr.dNvFlow[k], &nr.dNvCost[k] };
        if (cuLaunchKernel(nr.fNvofUp, (nr.w + 15) / 16, (nr.h + 15) / 16, 1, 16, 16, 1, 0,
                           (CUstream)st, a, nullptr) != CUDA_SUCCESS)
        { nr.die("nvofUp launch failed"); return false; }
    }
    float ma = kNvofA, mb = kNvofB;
    for (int k = 0; k < 2; k++)
    {
        void* a[] = { &nr.dNvFlow[k], &nr.dNvCost[k], &nr.dNvFlow[k ^ 1], &nr.w, &nr.h, &ma, &mb, &nr.dNvZ[k] };
        if (cuLaunchKernel(nr.fNvofMetric, (nr.w + 15) / 16, (nr.h + 15) / 16, 1, 16, 16, 1, 0,
                           (CUstream)st, a, nullptr) != CUDA_SUCCESS)
        { nr.die("nvofMetric launch failed"); return false; }
    }
    if (nr.nvProf)
    {
        cudaEventRecord(nr.nvEv[1], st);
        float ms = 0.0f;
        if (cudaEventSynchronize(nr.nvEv[1]) == cudaSuccess && cudaEventElapsedTime(&ms, nr.nvEv[0], nr.nvEv[1]) == cudaSuccess)
        { nr.nvPairMs += ms; nr.nvPairN++; }
        if (nr.nvPairN && nr.nvPairN % 64 == 0)
            LOG("[nvof-prof] pair (luma + execute + upsample + metric) %.3f ms mean over %u pairs, tween %.3f ms mean\n",
                nr.nvPairMs / nr.nvPairN, nr.nvPairN, nr.nvTweenN ? nr.nvTweenMs / nr.nvTweenN : 0.0);
    }
    return true;
}

// one tween at t into dNvOut, the E2 chain (2026-09-21 fix round, gate nvof_equiv_e.py): the
// velocity of both frames splatted to time t, its gaps push-pull filled, both frames SAMPLED
// along it, and where the vectors are untrustworthy (the blurred fallback mask) the plain blend
// instead: failed vectors show as ghosting, never as speckle
static bool nativeNvofTween(NativeRife& nr, float t)
{
    cudaStream_t st = nr.stream;
    const size_t plane = (size_t)nr.ph * nr.pw, mp = (size_t)nr.w * nr.h;
    const unsigned gx = (unsigned)(nr.w + 15) / 16, gy = (unsigned)(nr.h + 15) / 16;
    auto launch = [&](CUfunction f, unsigned bx, unsigned by, void** a, const char* what) -> bool
    {
        if (cuLaunchKernel(f, bx, by, 1, 16, 16, 1, 0, (CUstream)st, a, nullptr) == CUDA_SUCCESS) return true;
        nr.die(what);
        return false;
    };
    if (nr.nvProf) cudaEventRecord(nr.nvEv[0], st);
    if (cudaMemsetAsync(nr.dNvAcc, 0, 5 * mp * sizeof(long long), st) != cudaSuccess)
    { nr.die("nvof accumulator clear failed"); return false; }
    // (1) the velocity (frame 0 -> 1 px) of both frames at time t, coverage per frame
    float s0 = t, s1 = 1.0f - t;
    float b0 = logf(1.0f - t), b1 = logf(t);
    float sg0 = 1.0f, sg1 = -1.0f;
    int cp0 = 3, cp1 = 4;
    for (int k = 0; k < 2; k++)
    {
        void* a[] = { &nr.dNvFlow[k], &nr.dNvZ[k], k ? &s1 : &s0, k ? &b1 : &b0, k ? &sg1 : &sg0,
                      k ? &cp1 : &cp0, &nr.w, &nr.h, &nr.dNvAcc };
        if (!launch(nr.fSplatVel, gx, gy, a, "splatVel launch failed")) return false;
    }
    // (2) the merged velocity, confidence, visibilities, the raw fallback mask
    float lo = kNvofLo, hi = kNvofHi;
    {
        void* a[] = { &nr.dNvAcc, &nr.w, &nr.h, &t, &lo, &hi, &nr.dNvN0, &nr.dNvD0, &nr.dNvVis, &nr.dNvMraw };
        if (!launch(nr.fVelNorm, gx, gy, a, "velNorm launch failed")) return false;
    }
    // (3) push-pull: down to 1 px, the top normalised, back up to level 0 (dNvV)
    float* pn = nr.dNvN0;
    float* pd = nr.dNvD0;
    int pw_ = nr.w, ph_ = nr.h;
    for (auto& L : nr.nvPyr)
    {
        void* a[] = { &pn, &pd, &pw_, &ph_, &L.n, &L.d, &L.w, &L.h };
        if (!launch(nr.fPpDown, (unsigned)(L.w + 15) / 16, (unsigned)(L.h + 15) / 16, a, "ppDown launch failed")) return false;
        pn = L.n; pd = L.d; pw_ = L.w; ph_ = L.h;
    }
    if (nr.nvPyr.empty()) { nr.die("nvof pyramid empty"); return false; }
    {
        NativeRife::NvLevel& T = nr.nvPyr.back();
        void* a[] = { &T.n, &T.d, &T.w, &T.h, &T.out };
        if (!launch(nr.fPpTop, (unsigned)(T.w + 15) / 16, (unsigned)(T.h + 15) / 16, a, "ppTop launch failed")) return false;
    }
    for (int i = (int)nr.nvPyr.size() - 2; i >= -1; i--)
    {
        NativeRife::NvLevel& C = nr.nvPyr[i + 1];              // the coarser level, already filled
        float* n = i >= 0 ? nr.nvPyr[i].n : nr.dNvN0;
        float* d = i >= 0 ? nr.nvPyr[i].d : nr.dNvD0;
        float* o = i >= 0 ? nr.nvPyr[i].out : nr.dNvV;
        int lw = i >= 0 ? nr.nvPyr[i].w : nr.w, lh = i >= 0 ? nr.nvPyr[i].h : nr.h;
        void* a[] = { &n, &d, &lw, &lh, &C.out, &C.w, &C.h, &o };
        if (!launch(nr.fPpUp, (unsigned)(lw + 15) / 16, (unsigned)(lh + 15) / 16, a, "ppUp launch failed")) return false;
    }
    // (4) the mask blurred (x then y, scipy's gaussian_filter)
    float sig = kNvofSigma;
    int d0 = 0, d1 = 1;
    {
        void* a[] = { &nr.dNvMraw, &nr.w, &nr.h, &sig, &d0, &nr.dNvBlurTmp };
        if (!launch(nr.fBlur1, gx, gy, a, "blur1 (x) launch failed")) return false;
        void* b[] = { &nr.dNvBlurTmp, &nr.w, &nr.h, &sig, &d1, &nr.dNvMask };
        if (!launch(nr.fBlur1, gx, gy, b, "blur1 (y) launch failed")) return false;
    }
    // (5) sample both frames along the velocity, mix with the plain blend by the mask
    int ps = (int)plane;
    const float* i0 = nr.dX;
    const float* i1 = nr.dX + 3 * plane;
    {
        void* a[] = { (void*)&i0, (void*)&i1, &ps, &nr.pw, &nr.dNvV, &nr.dNvVis, &nr.dNvMask,
                      &nr.w, &nr.h, &t, &nr.dNvOut };
        if (!launch(nr.fNvofCompose, gx, gy, a, "nvofCompose launch failed")) return false;
    }
    if (nr.nvProf)
    {
        cudaEventRecord(nr.nvEv[1], st);
        float ms = 0.0f;
        if (cudaEventSynchronize(nr.nvEv[1]) == cudaSuccess && cudaEventElapsedTime(&ms, nr.nvEv[0], nr.nvEv[1]) == cudaSuccess)
        { nr.nvTweenMs += ms; nr.nvTweenN++; }
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
            for (NvOFGPUBufferHandle* b : { &nr.ofIn[k], &nr.ofOut[k], &nr.ofCost[k] })
                if (*b) { g_nvofApi.nvOFDestroyGPUBufferCuda(*b); *b = nullptr; }
        if (nr.ofH) { g_nvofApi.nvOFDestroy(nr.ofH); nr.ofH = nullptr; }
    }
    for (int k = 0; k < 2; k++)
    {
        for (float** p : { &nr.dNvFlow[k], &nr.dNvCost[k], &nr.dNvZ[k] })
            if (*p) { cudaFree(*p); *p = nullptr; }
        nr.ofInP[k] = nr.ofOutP[k] = nr.ofCostP[k] = 0;
    }
    if (nr.dNvAcc) { cudaFree(nr.dNvAcc); nr.dNvAcc = nullptr; }
    if (nr.dNvOut) { cudaFree(nr.dNvOut); nr.dNvOut = nullptr; }
    for (float** p : { &nr.dNvN0, &nr.dNvD0, &nr.dNvVis, &nr.dNvMraw, &nr.dNvMask, &nr.dNvBlurTmp, &nr.dNvV })
        if (*p) { cudaFree(*p); *p = nullptr; }
    for (auto& L : nr.nvPyr)
        for (float* p : { L.n, L.d, L.out }) if (p) cudaFree(p);
    nr.nvPyr.clear();
    for (auto& e : nr.nvEv) if (e) { cudaEventDestroy(e); e = nullptr; }
}

// NVIDIA Smooth Motion (fruc, 2026-09-21, memory priority 21): the shipped nvoffruc_bridge.dll,
// loaded once per process by full path from the folder the handoff named (NvOFFRUC.dll and its
// cudart64_110.dll are user-installed beside it; the bridge loads NvOFFRUC.dll signature-checked
// itself). Its state is process-global (one FRUC instance), so a session creates it at setup
// and destroys it in nativeFree. The bridge runs FRUC in its own CUDA context and brackets
// every call with cuCtxSynchronize on the caller's context (its SYNC FENCE notes), so the
// packs queued on nr.stream have landed when it copies, and its output has landed on return.
struct FrucBridge
{
    HMODULE mod = nullptr;
    const char* (*lastError)() = nullptr;
    int (*create)(unsigned, unsigned) = nullptr;
    int (*interpolate)(void*, void*, void*, double, int*) = nullptr;
    // the feed-once call (bridge 2026-09-26): mode 0 = prime prev + feed cur, 1 = prev was fed
    // last, feed cur only, 2 = the same pair again; an older bridge lacks it (interpolate then)
    int (*step)(void*, void*, void*, double, int, int*) = nullptr;
    void (*destroy)() = nullptr;
};
static FrucBridge g_fruc;

static bool nativeFrucLoad(const std::string& dir)
{
    if (g_fruc.mod) return true;
    const std::wstring wdir = utf8ToWide(dir);
    const std::wstring p = wdir + L"\\nvoffruc_bridge.dll";
    HMODULE m = LoadLibraryExW(p.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!m) { LOG("native: fruc: nvoffruc_bridge.dll did not load from %s (err %lu)\n", dir.c_str(), GetLastError()); return false; }
    g_fruc.lastError = (const char* (*)())GetProcAddress(m, "nvoffruc_last_error");
    g_fruc.create = (int (*)(unsigned, unsigned))GetProcAddress(m, "nvoffruc_create");
    g_fruc.interpolate = (int (*)(void*, void*, void*, double, int*))GetProcAddress(m, "nvoffruc_interpolate");
    g_fruc.destroy = (void (*)())GetProcAddress(m, "nvoffruc_destroy");
    g_fruc.step = (int (*)(void*, void*, void*, double, int, int*))GetProcAddress(m, "nvoffruc_step");
    if (!g_fruc.lastError || !g_fruc.create || !g_fruc.interpolate || !g_fruc.destroy)
    { LOG("native: fruc: nvoffruc_bridge.dll lacks an expected export\n"); FreeLibrary(m); return false; }
    g_fruc.mod = m;
    return true;
}

// the FRUC instance at the padded model size (python sizes it to pw x ph the same way) and the
// BGRA8 surfaces; the calling thread has the device bound
static bool nativeFrucSetup(NativeRife& nr)
{
    if (!nativeFrucLoad(nr.frucDir)) return false;
    const int rc = g_fruc.create((unsigned)nr.pw, (unsigned)nr.ph);
    // the bridge points the DLL search at its folder while it loads NvOFFRUC.dll (its
    // secure_load); give the process its default search back
    SetDllDirectoryW(nullptr);
    if (rc != 0) { LOG("native: fruc: nvoffruc_create %dx%d failed: %s (rc %d)\n", nr.pw, nr.ph, g_fruc.lastError(), rc); return false; }
    nr.frCreated = true;
    const size_t bytes = (size_t)nr.pw * nr.ph * 4, plane = (size_t)nr.ph * nr.pw;
    for (auto& s : nr.dFrSurf) NCHK(cudaMalloc((void**)&s, bytes), "alloc fruc surface");
    NCHK(cudaMalloc((void**)&nr.dFrOutB, bytes), "alloc fruc output");
    NCHK(cudaMalloc((void**)&nr.dFrOut, 3 * plane * sizeof(float)), "alloc fruc tween");
    nr.frPrev = nr.frLast = nr.frA = nr.frB = nr.frFed = -1;
    nr.frFirst = false;
    LOG("native: fruc session %dx%d (model %dx%d), NvOFFRUC through nvoffruc_bridge.dll, 8-bit BGRA%s\n",
        nr.pw, nr.ph, nr.w, nr.h, g_fruc.step ? ", feed-once" : "");
    return true;
}

// the pair: the new frame packed into a surface that holds neither the previous frame nor the
// last tweened pair's end. python's Fruc._reuse, only when this group has tweens: a pair that
// does not continue the last interpolated one first gets one dropped warp at 0.5 of the
// skipped pair (last end, previous frame), so the OFA hints stay consecutive
static bool nativeFrucPair(NativeRife& nr, const float* dCur, uint32_t nTween)
{
    int n = 0;
    while (n == nr.frPrev || n == nr.frLast) n++;
    int ps = nr.ph * nr.pw;
    void* a[] = { (void*)&dCur, &ps, &nr.pw, &nr.pw, &nr.ph, &nr.dFrSurf[n] };
    // true BGRA from either plane order: live (B, G, R), offline (R, G, B) (nvoffruc.py _pack)
    if (cuLaunchKernel(nr.planesRgb ? nr.fPackBgraRgb : nr.fPackBgra, (nr.pw + 15) / 16, (nr.ph + 15) / 16, 1, 16, 16, 1, 0,
                       (CUstream)nr.stream, a, nullptr) != CUDA_SUCCESS)
    { nr.die("packBgra (fruc) launch failed"); return false; }
    if (n == nr.frFed) nr.frFed = -1;   // the frame FRUC was fed last is gone from its surface
    nr.frA = nr.frPrev;
    nr.frB = n;
    nr.frPrev = n;
    if (!nTween || nr.frA < 0) return true;
    if (g_fruc.step)
    {
        // feed-once bridge: no priming warp; the pair's first tween says whether FRUC was fed
        // frA last (mode 1) or must prime it (mode 0), the other tweens reuse the pair (mode 2)
        nr.frFirst = true;
        nr.frLast = nr.frB;
        return true;
    }
    if (nr.frLast >= 0 && nr.frLast != nr.frA)
    {
        int rep = 0;
        const int rc = g_fruc.interpolate(nr.dFrSurf[nr.frLast], nr.dFrSurf[nr.frA], nr.dFrOutB, 0.5, &rep);
        if (rc != 0) { LOG("native: fruc: priming warp failed: %s (rc %d)\n", g_fruc.lastError(), rc); nr.die("fruc priming failed"); return false; }
        nr.frPrimed++;
    }
    nr.frLast = nr.frB;
    return true;
}

// one tween at t into dFrOut (3, ph, pw), python's Fruc._infer + NvOFFRUC.interpolate (a
// double, the bridge's own type: offline --fps hands python's double fraction over unrounded)
static bool nativeFrucTween(NativeRife& nr, double t)
{
    int rep = 0;
    int rc;
    if (g_fruc.step)
    {
        const int mode = !nr.frFirst ? 2 : (nr.frFed >= 0 && nr.frFed == nr.frA) ? 1 : 0;
        // a pair that does not continue the last fed one (after the session's first tween: a
        // repack of the fed surface clears frFed, so frFed alone would miss those primes)
        if (mode == 0 && nr.frTweens > 0) nr.frPrimed++;
        rc = g_fruc.step(nr.dFrSurf[nr.frA], nr.dFrSurf[nr.frB], nr.dFrOutB, t, mode, &rep);
        if (rc == 0) { nr.frFirst = false; nr.frFed = nr.frB; }
    }
    else
        rc = g_fruc.interpolate(nr.dFrSurf[nr.frA], nr.dFrSurf[nr.frB], nr.dFrOutB, t, &rep);
    if (rc != 0) { LOG("native: fruc: interpolate failed: %s (rc %d)\n", g_fruc.lastError(), rc); nr.die("fruc interpolate failed"); return false; }
    if (rep) nr.frRepeats++;
    nr.frTweens++;
    void* a[] = { &nr.dFrOutB, &nr.pw, &nr.ph, &nr.dFrOut };
    if (cuLaunchKernel(nr.planesRgb ? nr.fUnpackBgraRgb : nr.fUnpackBgra, (nr.pw + 15) / 16, (nr.ph + 15) / 16, 1, 16, 16, 1, 0,
                       (CUstream)nr.stream, a, nullptr) != CUDA_SUCCESS)
    { nr.die("unpackBgra (fruc) launch failed"); return false; }
    return true;
}

static void nativeFrucFree(NativeRife& nr)
{
    if (nr.frCreated)
    {
        LOG("native: fruc session: %llu tweens, %llu primed pairs, %llu frame repeats\n",
            (unsigned long long)nr.frTweens, (unsigned long long)nr.frPrimed, (unsigned long long)nr.frRepeats);
        g_fruc.destroy();
        nr.frCreated = false;
    }
    for (auto& s : nr.dFrSurf) if (s) { cudaFree(s); s = nullptr; }
    if (nr.dFrOutB) { cudaFree(nr.dFrOutB); nr.dFrOutB = nullptr; }
    if (nr.dFrOut) { cudaFree(nr.dFrOut); nr.dFrOut = nullptr; }
}

static bool nativeCudaInitEarly(NativeRife& nr, const std::wstring& cacheDir)
{
    if (!nativeBuildKernels(nr, cacheDir)) return false;

    const int capBpp = nr.hdr ? 8 : 4;
    const size_t plane = (size_t)nr.ph * nr.pw;
    NCHK(cudaMalloc((void**)&nr.dCap, (size_t)nr.cw * nr.ch * capBpp), "alloc capture staging");
    NCHK(cudaMalloc((void**)&nr.dX, 6 * plane * sizeof(float)), "alloc x");
    // the identical-pair flag and its pinned readback (one int per group)
    NCHK(cudaMalloc((void**)&nr.dStaticFlag, sizeof(int)), "alloc static flag");
    NCHK(cudaHostAlloc((void**)&nr.hStaticFlag, sizeof(int), cudaHostAllocDefault), "alloc static flag readback");
    // no-engine mode: the packed frame (the cur half of x) is the whole model side; a GMFSS
    // session shares only dX (its chain's own buffers are sized off the engines in
    // nativeGmfssSetup), so neither needs the RIFE encode / timestep / merged buffers
    if (!nr.noEngine && !nr.gmfss && !nr.nvof && !nr.fruc && !nr.drba)
    {
        NCHK(cudaMalloc((void**)&nr.dF[0], 16 * plane * sizeof(float)), "alloc f0");
        NCHK(cudaMalloc((void**)&nr.dF[1], 16 * plane * sizeof(float)), "alloc f1");
        NCHK(cudaMalloc((void**)&nr.dEncHalf, 16 * plane * sizeof(uint16_t)), "alloc encode out");
        NCHK(cudaMalloc((void**)&nr.dT, (size_t)nr.batchMax * plane * sizeof(float)), "alloc timestep");
        NCHK(cudaMalloc((void**)&nr.dMerged, (size_t)nr.batchMax * 3 * plane * sizeof(float)), "alloc merged");
    }
    // native DRBA: one tween per IFNet enqueue (its own DRM map), so a batch-1 timestep and
    // merged; the history ring of padded frames and encodes replaces f0 / f1
    if (nr.drba)
    {
        NCHK(cudaMalloc((void**)&nr.dEncHalf, 16 * plane * sizeof(uint16_t)), "alloc encode out");
        NCHK(cudaMalloc((void**)&nr.dT, plane * sizeof(float)), "alloc timestep");
        NCHK(cudaMalloc((void**)&nr.dMerged, 3 * plane * sizeof(float)), "alloc merged");
        for (int i = 0; i < 4; i++)
        {
            NCHK(cudaMalloc((void**)&nr.dDrI[i], 3 * plane * sizeof(float)), "alloc drba frame ring");
            NCHK(cudaMalloc((void**)&nr.dDrF[i], 16 * plane * sizeof(float)), "alloc drba encode ring");
        }
        for (int i = 0; i < 2; i++)
        {
            NCHK(cudaMalloc((void**)&nr.dDrX[i], 6 * plane * sizeof(float)), "alloc drba x");
            NCHK(cudaMalloc((void**)&nr.drWin[i].f10, 2 * plane * sizeof(float)), "alloc drba window");
            NCHK(cudaMalloc((void**)&nr.drWin[i].r, 4 * plane * sizeof(float)), "alloc drba window");
        }
        NCHK(cudaMalloc((void**)&nr.dDrFlow, 4 * plane * sizeof(float)), "alloc drba flow");
        NCHK(cudaMalloc((void**)&nr.dDrFlowN, 4 * plane * sizeof(float)), "alloc drba flow");
        NCHK(cudaMalloc((void**)&nr.dDrAcc, 6 * plane * sizeof(long long)), "alloc drba accumulator");
    }
    // live Restore (2026-09-15): the engine's x / y (sized for fp32 so either dtype fits), the
    // fold's horizontal pass at the 4x height and the widest possible target, and the fold
    // back to the model size for the VSR-follows case; the enlarging fold's fp32 copy is
    // allocated in nativeRtxInit once the target is known (rare: Upscale to above 4x)
    if (nr.restore)
    {
        const size_t mp = (size_t)nr.w * nr.h;
        int maxTw = nr.w > nr.dw ? nr.w : nr.dw;
        if (nr.uw > maxTw) maxTw = nr.uw;
        NCHK(cudaMalloc(&nr.dRestIn, 3 * mp * sizeof(float)), "alloc restore input");
        NCHK(cudaMalloc(&nr.dRestOut, 3 * 16 * mp * sizeof(float)), "alloc restore output");
        NCHK(cudaMalloc((void**)&nr.dRestTmp, (size_t)3 * 4 * nr.h * maxTw * sizeof(float)), "alloc restore fold pass");
        NCHK(cudaMalloc((void**)&nr.dRest, 3 * mp * sizeof(float)), "alloc restore model-size frame");
    }
    // WO-23: with live TrueHDR the PQ frame always exists at capture resolution first, so
    // the resize pair runs at EVERY image scale (at 1.00 the triangle filter sits on identity
    // positions and is an exact copy) and both staging planes are needed even at scale 1.00.
    if (nr.w != nr.cw || nr.h != nr.ch || nr.rtxHdr)
    {
        NCHK(cudaMalloc((void**)&nr.dTmp, (size_t)3 * nr.ch * nr.w * sizeof(float)), "alloc resize temp");
        // HDR rescales ON PQ, so the conversion runs at capture resolution first
        if (nr.hdr)
            NCHK(cudaMalloc((void**)&nr.dCapF, (size_t)3 * nr.ch * nr.cw * sizeof(float)),
                 "alloc PQ capture plane");
    }
    NCHK(cudaEventCreateWithFlags(&nr.capEv, cudaEventDisableTiming), "create capture event");
    // the nvof model: the Optical Flow session and the glue buffers (the kernels are bound above)
    if (nr.nvof && !nativeNvofSetup(nr)) return false;
    // Smooth Motion: the bridge's FRUC instance and its BGRA8 surfaces
    if (nr.fruc && !nativeFrucSetup(nr)) return false;
    return true;
}

// the TRT cache folder = the folder of the jit cache the handoff named (kernels cubin lives there)
static std::wstring nativeCacheDir(const NativeRife& nr)
{
    if (!nr.cachePath.empty()) return std::wstring(nr.cachePath.begin(), nr.cachePath.end());
    if (nr.jitPath.empty()) return std::wstring();
    std::wstring w(nr.jitPath.begin(), nr.jitPath.end());
    const size_t sl = w.find_last_of(L"\\/");
    return sl == std::wstring::npos ? std::wstring(L".") : w.substr(0, sl);
}

// WO-23: live RTX TrueHDR runs in this process now. Everything comes from the exe's own
// globals, parsed from the app's RTX HDR controls, so no new flag exists. Runs after
// nativeLoadDlls (the bridge folder is derived from the runtime folder found there).
static bool nativeConfigHdr(NativeRife& nr)
{
    // live Sharpen and RTX VSR (2026-09-15) read the same globals the python route gets;
    // VSR follows the python rules: SDR only (WO-8: demoted to bicubic in HDR), the bridge
    // and nvngx_vsr.dll present (else bicubic, never a route change), and the fit must
    // enlarge (decided in nativeRtxInit once the handoff geometry is known)
    nr.sharpen = (float)(g_sharpen < 0.0 ? 0.0 : (g_sharpen > 1.0 ? 1.0 : g_sharpen));
    if (g_rtxVsr)
    {
        if (g_hdr)
            LOG("native: live RTX VSR demoted to bicubic in HDR mode (SDR-only feature)\n");
        else
        {
            std::wstring dir;
            wchar_t ov[MAX_PATH]{};
            if (GetEnvironmentVariableW(L"SMV_RTXVIDEO_DIR", ov, MAX_PATH) && ov[0]) dir = ov;
            else if (!g_nativeEngineDir.empty()) dir = g_nativeEngineDir + L"\\rtxvideo";
            const bool have = !dir.empty()
                && GetFileAttributesW((dir + L"\\rtxvideo_cuda.dll").c_str()) != INVALID_FILE_ATTRIBUTES
                && GetFileAttributesW((dir + L"\\nvngx_vsr.dll").c_str()) != INVALID_FILE_ATTRIBUTES;
            if (!have) LOG("native: live RTX VSR unavailable (no rtxvideo bridge or nvngx_vsr.dll), bicubic\n");
            else if (!rtxBridgeLoad()) return false;
            else nr.vsrWant = true;
        }
    }
    if (!(g_hdr && g_rtxHdr)) return true;
    if (!rtxBridgeLoad()) return false;
    nr.rtxHdr = true;
    nr.thdr.Contrast = (uint32_t)(g_hdrCon < 0 ? 0 : (g_hdrCon > 200 ? 200 : g_hdrCon));
    nr.thdr.Saturation = (uint32_t)(g_hdrSat < 0 ? 0 : (g_hdrSat > 200 ? 200 : g_hdrSat));
    nr.thdr.MiddleGray = 50;      // the live defaults, rtxvideo.py ~173
    nr.thdr.MaxLuminance = 1000;
    nr.sdrScale = (float)(g_sdrWhite / 80.0 < 0.5 ? 0.5 : g_sdrWhite / 80.0);
    nr.rtxMode = !wcscmp(g_hdrColor, L"rtx") ? 1 : (!wcscmp(g_hdrColor, L"raw") ? 2 : 0);
    nr.rtxVib = (float)(g_hdrVib < 0.0 ? 0.0 : (g_hdrVib > 1.0 ? 1.0 : g_hdrVib));
    nr.rtxSb = (float)(g_hdrSb < 0.0 ? 0.0 : (g_hdrSb > 1.0 ? 1.0 : g_hdrSb));
    return true;
}

// GMFSS sub-step 5a: one warm enqueue of an engine on zeroed scratch buffers sized from its own
// tensor shapes (every GMFSS engine is H/W pinned, so the shapes are static), which also logs
// the tensor contract the chain of sub-step 5c will bind to. Buffers are freed again here.
static const char* trtDtypeName(nvinfer1::DataType t)
{
    switch (t)
    {
    case nvinfer1::DataType::kFLOAT: return "fp32";
    case nvinfer1::DataType::kHALF: return "fp16";
    case nvinfer1::DataType::kBF16: return "bf16";
    case nvinfer1::DataType::kINT32: return "i32";
    case nvinfer1::DataType::kINT64: return "i64";
    default: return "other";
    }
}

static size_t trtElemSize(nvinfer1::DataType t)
{
    switch (t)
    {
    case nvinfer1::DataType::kFLOAT: case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kHALF: case nvinfer1::DataType::kBF16: return 2;
    case nvinfer1::DataType::kINT64: return 8;
    default: return 1;
    }
}

static bool nativeWarmEngine(NativeRife& nr, nvinfer1::ICudaEngine* eng,
                             nvinfer1::IExecutionContext* ctx, const char* tag,
                             const char* model = "gmfss")
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
            if (d.d[k] < 0) { LOG("native: %s %s tensor %s has a dynamic dim\n", model, tag, nm); ok = false; break; }
            vol *= (size_t)d.d[k];
            dims += (k ? "x" : "") + std::to_string(d.d[k]);
        }
        if (!ok) break;
        const bool in = eng->getTensorIOMode(nm) == nvinfer1::TensorIOMode::kINPUT;
        if (in && !ctx->setInputShape(nm, d))
        { LOG("native: %s %s setInputShape rejected on %s\n", model, tag, nm); ok = false; break; }
        void* p = nullptr;
        const size_t bytes = vol * trtElemSize(eng->getTensorDataType(nm));
        if (cudaMalloc(&p, bytes) != cudaSuccess)
        { LOG("native: %s %s warm-up alloc failed (%zu bytes)\n", model, tag, bytes); ok = false; break; }
        bufs.push_back(p);
        cudaMemsetAsync(p, 0, bytes, nr.stream);
        ctx->setTensorAddress(nm, p);
        contract += std::string(i ? ", " : "") + (in ? "" : "-> ") + nm + " " + trtDtypeName(eng->getTensorDataType(nm)) + " " + dims;
    }
    if (ok && !ctx->enqueueV3(nr.stream)) { LOG("native: %s %s warm-up enqueue failed\n", model, tag); ok = false; }
    if (ok && cudaStreamSynchronize(nr.stream) != cudaSuccess) { LOG("native: %s %s warm-up sync failed\n", model, tag); ok = false; }
    for (void* p : bufs) cudaFree(p);
    if (ok) LOG("native: %s %s: %s\n", model, tag, contract.c_str());
    return ok;
}

// GMFSS sub-step 5c: the chain's contract and its buffers. Every channel count and dtype is
// READ OFF THE ENGINES (the 5a rule: the handoff line never types them), then one pair's and
// one tween's planar buffers are allocated from it. Runs inside nativeTrtInit right after the
// five warm-ups, so a mismatch refuses the session before any group. The accumulator is one
// buffer for all eight splats (they are strictly sequential on the one stream): the widest is
// the half level's gmC[0] + 1 planes, and the quarter / eighth levels need less
// ((2 * gmC[1] + 1) / 4 and (2 * gmC[2] + 1) / 16 of a half plane at the stock 64/128/192).
static bool nativeGmfssSetup(NativeRife& nr)
{
    auto bad = [](const char* what) { LOG("native: gmfss chain: %s\n", what); return false; };
    if (nr.hh <= 0 || nr.hw <= 0 || (nr.hh & 3) || (nr.hw & 3)) return bad("the half frame is not a multiple of 4");
    if (nr.hh * 2 != nr.ph || nr.hw * 2 != nr.pw) return bad("the half frame is not half of the padded frame");
    // one tensor: 4 dims, the expected dtype (kFLOAT unless dtOut takes the answer) and the
    // expected shape (a negative expectation means "read it", n = the leading dim)
    int dims[4]{};
    auto tensor = [&](int e, const char* nm, int n, int c, int h, int w, bool* dtOut) -> bool
    {
        const nvinfer1::Dims s = nr.engGm[e]->getTensorShape(nm);
        if (s.nbDims != 4)
        { LOG("native: gmfss chain: %s %s is not a 4D tensor\n", kGmName[e], nm); return false; }
        for (int i = 0; i < 4; i++) dims[i] = (int)s.d[i];
        const nvinfer1::DataType t = nr.engGm[e]->getTensorDataType(nm);
        if (dtOut) *dtOut = t == nvinfer1::DataType::kHALF;
        if (!dtOut && t != nvinfer1::DataType::kFLOAT)
        { LOG("native: gmfss chain: %s %s is %s, the chain needs fp32\n", kGmName[e], nm, trtDtypeName(t)); return false; }
        if (dtOut && t != nvinfer1::DataType::kHALF && t != nvinfer1::DataType::kFLOAT)
        { LOG("native: gmfss chain: %s %s dtype %s unsupported\n", kGmName[e], nm, trtDtypeName(t)); return false; }
        if (dims[0] != n || (c >= 0 && dims[1] != c) || (h >= 0 && dims[2] != h) || (w >= 0 && dims[3] != w))
        {
            LOG("native: gmfss chain: %s %s is %dx%dx%dx%d, expected %dx%dx%dx%d\n", kGmName[e], nm,
                dims[0], dims[1], dims[2], dims[3], n, c, h, w);
            return false;
        }
        return true;
    };
    const int hh = nr.hh, hw = nr.hw;
    bool fh2 = false, fh3 = false, mh1 = false;
    // feat_ext: the padded frame in, the three feature levels out (64 / 128 / 192 at stock)
    if (!tensor(0, "x", 1, 3, nr.ph, nr.pw, nullptr)) return false;
    if (!tensor(0, "f1", 1, -1, hh, hw, &nr.gmFeatHalf)) return false;
    nr.gmC[0] = dims[1];
    if (!tensor(0, "f2", 1, -1, hh / 2, hw / 2, &fh2)) return false;
    nr.gmC[1] = dims[1];
    if (!tensor(0, "f3", 1, -1, hh / 4, hw / 4, &fh3)) return false;
    nr.gmC[2] = dims[1];
    if (fh2 != nr.gmFeatHalf || fh3 != nr.gmFeatHalf) return bad("the feature levels have mixed dtypes");
    // the fused bidir GMFlow: row 0 = flow01, row 1 = flow10 (so the pyramids take both at once)
    if (!tensor(1, "img0", 1, 3, hh, hw, nullptr) || !tensor(1, "img1", 1, 3, hh, hw, nullptr)) return false;
    if (!tensor(1, "flow", 2, 2, hh, hw, nullptr)) return false;
    if (!tensor(2, "i0", 1, 3, hh, hw, nullptr) || !tensor(2, "i1", 1, 3, hh, hw, nullptr)
        || !tensor(2, "f01", 1, 2, hh, hw, nullptr) || !tensor(2, "f10", 1, 2, hh, hw, nullptr)) return false;
    if (!tensor(2, "m0", 1, 1, hh, hw, &nr.gmMetricHalf)) return false;
    if (!tensor(2, "m1", 1, 1, hh, hw, &mh1)) return false;
    if (mh1 != nr.gmMetricHalf) return bad("the two metrics have different dtypes");
    // the GMFSS IFNet: the 6-plane half buffer and the scalar timestep in, `rife` out
    if (!tensor(3, "x", 1, 6, hh, hw, nullptr) || !tensor(3, "merged", 1, 3, hh, hw, nullptr)) return false;
    {
        const nvinfer1::Dims s = nr.engGm[3]->getTensorShape("timestep");
        if (s.nbDims != 4 || s.d[0] != 1 || s.d[1] != 1 || s.d[2] != 1 || s.d[3] != 1
            || nr.engGm[3]->getTensorDataType("timestep") != nvinfer1::DataType::kFLOAT)
            return bad("the ifnet timestep is not a 1x1x1x1 fp32 value");
    }
    // fusionnet: the concats are adjacent planes, so b / c / d are exactly 2x the feature levels
    if (!tensor(4, "a", 1, 9, hh, hw, nullptr) || !tensor(4, "b", 1, 2 * nr.gmC[0], hh, hw, nullptr)
        || !tensor(4, "c", 1, 2 * nr.gmC[1], hh / 2, hw / 2, nullptr)
        || !tensor(4, "d", 1, 2 * nr.gmC[2], hh / 4, hw / 4, nullptr)) return false;
    if (!tensor(4, "out", 1, 3, nr.ph, nr.pw, &nr.gmOutHalf)) return false;

    const size_t hp = (size_t)hh * hw, qp = hp / 4, ep = hp / 16;
    const size_t plane = (size_t)nr.ph * nr.pw;
    const size_t fe = nr.gmFeatHalf ? 2 : 4, me = nr.gmMetricHalf ? 2 : 4, oe = nr.gmOutHalf ? 2 : 4;
    size_t accPlanes = (size_t)nr.gmC[0] + 1;
    if (((size_t)nr.gmC[1] + 1) * qp > accPlanes * hp) accPlanes = (((size_t)nr.gmC[1] + 1) * qp + hp - 1) / hp;
    if (((size_t)nr.gmC[2] + 1) * ep > accPlanes * hp) accPlanes = (((size_t)nr.gmC[2] + 1) * ep + hp - 1) / hp;
    size_t freeB = 0, freeA = 0, totB = 0;
    cudaMemGetInfo(&freeB, &totB);
    for (int s = 0; s < 2; s++)
        for (int l = 0; l < 3; l++)
            NCHK(cudaMalloc(&nr.dGmFeat[s][l], (size_t)nr.gmC[l] * (l == 0 ? hp : (l == 1 ? qp : ep)) * fe),
                 "alloc gmfss features");
    NCHK(cudaMalloc((void**)&nr.dGmHalf, 6 * hp * sizeof(float)), "alloc gmfss halves");
    NCHK(cudaMalloc((void**)&nr.dGmFlow, 4 * hp * sizeof(float)), "alloc gmfss flow");
    NCHK(cudaMalloc(&nr.dGmMetric, 2 * hp * me), "alloc gmfss metric");
    NCHK(cudaMalloc((void**)&nr.dGmFlowP[0], 4 * qp * sizeof(float)), "alloc gmfss flow pyramid");
    NCHK(cudaMalloc((void**)&nr.dGmFlowP[1], 4 * ep * sizeof(float)), "alloc gmfss flow pyramid");
    NCHK(cudaMalloc((void**)&nr.dGmMetP[0], 2 * qp * sizeof(float)), "alloc gmfss metric pyramid");
    NCHK(cudaMalloc((void**)&nr.dGmMetP[1], 2 * ep * sizeof(float)), "alloc gmfss metric pyramid");
    NCHK(cudaMalloc((void**)&nr.dGmAcc, accPlanes * hp * sizeof(long long)), "alloc gmfss splat accumulator");
    NCHK(cudaMalloc((void**)&nr.dGmFa, 9 * hp * sizeof(float)), "alloc gmfss fusion a");
    NCHK(cudaMalloc((void**)&nr.dGmFb, (size_t)2 * nr.gmC[0] * hp * sizeof(float)), "alloc gmfss fusion b");
    NCHK(cudaMalloc((void**)&nr.dGmFc, (size_t)2 * nr.gmC[1] * qp * sizeof(float)), "alloc gmfss fusion c");
    NCHK(cudaMalloc((void**)&nr.dGmFd, (size_t)2 * nr.gmC[2] * ep * sizeof(float)), "alloc gmfss fusion d");
    NCHK(cudaMalloc((void**)&nr.dGmT, sizeof(float)), "alloc gmfss timestep");
    NCHK(cudaMalloc(&nr.dGmOut, 3 * plane * oe), "alloc gmfss fusion out");
    NCHK(cudaMalloc((void**)&nr.dGmF, 3 * plane * sizeof(float)), "alloc gmfss model frame");
    cudaMemGetInfo(&freeA, &totB);
    LOG("native: gmfss chain: half %dx%d, features %d/%d/%d %s, metric %s, out %s, accumulator"
        " %zu planes, buffers %.0f MB\n",
        hw, hh, nr.gmC[0], nr.gmC[1], nr.gmC[2], nr.gmFeatHalf ? "fp16" : "fp32",
        nr.gmMetricHalf ? "fp16" : "fp32", nr.gmOutHalf ? "fp16" : "fp32", accPlanes,
        ((double)freeB - (double)freeA) / 1048576.0);
    nr.gmProf = GetEnvironmentVariableW(L"SMV_LIVE_GMFSS_PROF", nullptr, 0) != 0;
    if (nr.gmProf)
        for (auto& e : nr.gmEv) NCHK(cudaEventCreate(&e), "create gmfss profile event");
    return true;
}

static bool nativeTrtInit(NativeRife& nr)
{
    // resident host: the same engine pair stays loaded across sessions, only the two
    // execution contexts are recreated (measured 2026-09-12: engine deserialize 0.10 s +
    // jit cache 1.0 to 3.5 s per start against 0.3 to 0.8 s for the contexts alone)
    // live Restore (2026-09-15): the Real-ESRGAN engine loads beside the pair when the handoff
    // named one, its dtypes read off the engine (x / y are fp16 as python exports the .half()
    // net; fp32 is accepted too), and one warm enqueue on zeros so the first presented frame
    // pays no kernel specialisation (the merged jit cache makes that cheap)
    auto dt = [](nvinfer1::ICudaEngine* e, const char* n) { return e->getTensorDataType(n); };
    auto restoreReady = [&]() -> bool
    {
        if (!nr.engRest) return true;
        const auto xd = dt(nr.engRest, "x"), yd = dt(nr.engRest, "y");
        if (xd == nvinfer1::DataType::kHALF) nr.restHalfIn = true;
        else if (xd == nvinfer1::DataType::kFLOAT) nr.restHalfIn = false;
        else { LOG("native: restore engine input dtype unsupported\n"); return false; }
        if (yd == nvinfer1::DataType::kHALF) nr.restHalfOut = true;
        else if (yd == nvinfer1::DataType::kFLOAT) nr.restHalfOut = false;
        else { LOG("native: restore engine output dtype unsupported\n"); return false; }
        if (!nr.dRestIn || !nr.dRestOut) { LOG("native: restore buffers missing\n"); return false; }
        NCHK(cudaMemsetAsync(nr.dRestIn, 0, (size_t)3 * nr.w * nr.h * 4, nr.stream), "restore warm-up clear");
        nvinfer1::Dims4 din{ 1, 3, nr.h, nr.w };
        if (!nr.ctxRest->setInputShape("x", din))
        { LOG("native: restore setInputShape rejected (engine built for another size)\n"); return false; }
        nr.ctxRest->setTensorAddress("x", nr.dRestIn);
        nr.ctxRest->setTensorAddress("y", nr.dRestOut);
        if (!nr.ctxRest->enqueueV3(nr.stream)) { LOG("native: restore warm-up enqueue failed\n"); return false; }
        NCHK(cudaStreamSynchronize(nr.stream), "restore warm-up sync");
        return true;
    };
    // native DRBA (priority 21 (b)): the block0 engine loads beside the pair like the restore
    // engine (resident per path; any session without DRBA drops it, nothing of another route
    // idles in VRAM), its fp32 contract read off the engine, one warm enqueue on zeros
    auto block0Ready = [&]() -> bool
    {
        if (!nr.drba)
        {
            if (g_res.engB0) { delete g_res.engB0; g_res.engB0 = nullptr; g_res.block0Path.clear(); }
            return true;
        }
        if (g_resident && g_res.engB0 && g_res.block0Path == nr.block0Path)
        {
            nr.engB0 = g_res.engB0;
            if (!nativeMakeContext(nr, nr.engB0, &nr.cfgB0, &nr.ctxB0)) return false;
        }
        else
        {
            if (g_res.engB0) { delete g_res.engB0; g_res.engB0 = nullptr; g_res.block0Path.clear(); }
            nr.engB0 = nativeLoadEngine(nr, nr.block0Path, &nr.cfgB0, &nr.ctxB0);
            if (!nr.engB0) return false;
            if (g_resident) { g_res.engB0 = nr.engB0; g_res.block0Path = nr.block0Path; }
        }
        for (const char* n : { "img0", "img1", "f0", "f1", "flow" })
            if (dt(nr.engB0, n) != nvinfer1::DataType::kFLOAT)
            { LOG("native: block0 tensor %s is not fp32\n", n); return false; }
        return nativeWarmEngine(nr, nr.engB0, nr.ctxB0, "block0", "drba");
    };
    // GMFSS (sub-steps 5a / 5c): the five-engine set, resident like the pair (the same paths =
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
        for (int i = 0; i < 5 && same; i++) same = g_res.engGm[i] && g_res.gmPath[i] == nr.gmPath[i];
        if (same)
        {
            nr.rt = g_res.rt;
            nr.jit = g_res.jit;
            for (int i = 0; i < 5; i++)
            {
                nr.engGm[i] = g_res.engGm[i];
                if (!nativeMakeContext(nr, nr.engGm[i], &nr.cfgGm[i], &nr.ctxGm[i])) return false;
            }
        }
        else
        {
            if (g_resident && g_res.rt) residentFreeEngines();   // the RIFE pair or another set
            nr.rt = nvinfer1::createInferRuntime(g_nativeTrtLogger);
            if (!nr.rt) { LOG("native: createInferRuntime failed\n"); return false; }
            for (int i = 0; i < 5; i++)
            {
                nr.engGm[i] = nativeLoadEngine(nr, nr.gmPath[i], &nr.cfgGm[i], &nr.ctxGm[i]);
                if (!nr.engGm[i]) return false;
            }
        }
        if (g_res.engRest) { delete g_res.engRest; g_res.engRest = nullptr; g_res.restorePath.clear(); }
        if (!nr.restorePath.empty())
        {
            nr.engRest = nativeLoadEngine(nr, nr.restorePath, &nr.cfgRest, &nr.ctxRest);
            if (!nr.engRest) return false;
        }
        for (int i = 0; i < 5; i++)
            if (!nativeWarmEngine(nr, nr.engGm[i], nr.ctxGm[i], kGmName[i])) return false;
        if (!restoreReady()) return false;
        cudaMemGetInfo(&freeA, &totB);
        LOG("native: gmfss engine set %s in %.2f s, VRAM %+.0f MB\n",
            same ? "reused (resident), contexts recreated" : "loaded",
            (nowQpc100() - t0) / 1e7, ((double)freeB - (double)freeA) / 1048576.0);
        if (!nativeGmfssSetup(nr)) return false;
        if (g_resident)
        {
            g_res.rt = nr.rt;
            g_res.jit = nr.jit;
            g_res.dev = nr.dev;
            for (int i = 0; i < 5; i++) { g_res.engGm[i] = nr.engGm[i]; g_res.gmPath[i] = nr.gmPath[i]; }
            g_res.ifnetPath.clear(); g_res.encodePath.clear(); g_res.jitPath.clear();
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
        // no-engine mode: the resident holds no pair (both paths empty on both sides), only
        // the runtime, the jit cache and, per path, the restore engine
        if (!nr.noEngine && !nr.nvof && !nr.fruc && !nr.dlssg)
        {
            if (!nativeMakeContext(nr, nr.engIf, &nr.cfgIf, &nr.ctxIf)) return false;
            if (!nativeMakeContext(nr, nr.engEnc, &nr.cfgEnc, &nr.ctxEnc)) return false;
        }
        // the restore engine follows its own path: kept when the same one is asked for again,
        // dropped for a session without Restore, loaded for one that brings a different path
        if (g_res.restorePath != nr.restorePath)
        {
            if (g_res.engRest) { delete g_res.engRest; g_res.engRest = nullptr; }
            g_res.restorePath.clear();
            if (!nr.restorePath.empty())
            {
                nr.engRest = nativeLoadEngine(nr, nr.restorePath, &nr.cfgRest, &nr.ctxRest);
                if (!nr.engRest) return false;
                g_res.engRest = nr.engRest;
                g_res.restorePath = nr.restorePath;
            }
        }
        else if (g_res.engRest)
        {
            nr.engRest = g_res.engRest;
            if (!nativeMakeContext(nr, nr.engRest, &nr.cfgRest, &nr.ctxRest)) return false;
        }
        if (!restoreReady() || !block0Ready()) return false;
        LOG("native: resident engines reused, contexts recreated in %.2f s\n", (nowQpc100() - t0) / 1e7);
        return true;
    }
    if (g_resident && g_res.rt) residentFreeEngines();   // a different pair (new window size)
    nr.rt = nvinfer1::createInferRuntime(g_nativeTrtLogger);
    if (!nr.rt) { LOG("native: createInferRuntime failed\n"); return false; }
    if (!nr.noEngine && !nr.nvof && !nr.fruc && !nr.dlssg)
    {
        nr.engIf = nativeLoadEngine(nr, nr.ifnetPath, &nr.cfgIf, &nr.ctxIf);
        if (!nr.engIf) return false;
        nr.engEnc = nativeLoadEngine(nr, nr.encodePath, &nr.cfgEnc, &nr.ctxEnc);
        if (!nr.engEnc) return false;
    }
    if (!nr.restorePath.empty())
    {
        nr.engRest = nativeLoadEngine(nr, nr.restorePath, &nr.cfgRest, &nr.ctxRest);
        if (!nr.engRest) return false;
    }
    if (!nr.noEngine && !nr.nvof && !nr.fruc && !nr.dlssg)
    {
        // dtype contract, read off the engines rather than trusted from the handoff line
        const char* need[] = { "x", "timestep", "f0", "f1", "merged" };
        for (const char* n : need)
            if (dt(nr.engIf, n) != nvinfer1::DataType::kFLOAT)
            { LOG("native: IFNet tensor %s is not fp32, phase 1 only handles fp32\n", n); return false; }
        const auto ed = dt(nr.engEnc, "feat");
        if (ed == nvinfer1::DataType::kHALF) nr.encHalf = true;
        else if (ed == nvinfer1::DataType::kFLOAT) nr.encHalf = false;
        else { LOG("native: encode output dtype unsupported\n"); return false; }
        if (dt(nr.engEnc, "img") != nvinfer1::DataType::kFLOAT)
        { LOG("native: encode input is not fp32\n"); return false; }
    }
    if (!restoreReady() || !block0Ready()) return false;
    if (g_resident)
    {
        g_res.rt = nr.rt;
        g_res.engIf = nr.engIf;
        g_res.engEnc = nr.engEnc;
        g_res.engRest = nr.engRest;
        g_res.jit = nr.jit;
        g_res.encHalf = nr.encHalf;
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
    if (nr.stream) cudaStreamSynchronize(nr.stream);
    nativeNvofFree(nr);   // the Optical Flow session and its buffers (nvof sessions only)
    nativeFrucFree(nr);   // the bridge's FRUC instance and its surfaces (fruc sessions only)
    // WO-23: release the TrueHDR feature on the compute thread, before the CUDA teardown
    if (nr.thdrN && !g_offline)   // offline logs its own line with the light levels
        LOG("native: TrueHDR eval %.2f ms mean, %.2f ms max, over %llu real frames\n",
            nr.thdrMs / (double)nr.thdrN, nr.thdrMaxMs, (unsigned long long)nr.thdrN);
    if (nr.vsrN)
        LOG("native: RTX VSR eval %.2f ms mean, %.2f ms max, over %llu presented frames\n",
            nr.vsrMs / (double)nr.vsrN, nr.vsrMaxMs, (unsigned long long)nr.vsrN);
    if (g_rtxb.created && g_rtxb.shutdown) { g_rtxb.shutdown(); g_rtxb.created = false; }
    if (nr.ctxIf) { delete nr.ctxIf; nr.ctxIf = nullptr; }
    if (nr.ctxEnc) { delete nr.ctxEnc; nr.ctxEnc = nullptr; }
    if (nr.ctxRest) { delete nr.ctxRest; nr.ctxRest = nullptr; }
    if (nr.cfgIf) { delete nr.cfgIf; nr.cfgIf = nullptr; }
    if (nr.cfgEnc) { delete nr.cfgEnc; nr.cfgEnc = nullptr; }
    if (nr.cfgRest) { delete nr.cfgRest; nr.cfgRest = nullptr; }
    if (nr.engRest && nr.engRest != g_res.engRest) delete nr.engRest;
    nr.engRest = nullptr;
    if (nr.ctxB0) { delete nr.ctxB0; nr.ctxB0 = nullptr; }
    if (nr.cfgB0) { delete nr.cfgB0; nr.cfgB0 = nullptr; }
    if (nr.engB0 && nr.engB0 != g_res.engB0) delete nr.engB0;
    nr.engB0 = nullptr;
    if (nr.drTweens || nr.drHeads)
        LOG("native: drba %llu tweens (%llu head pairs), %llu block0 runs\n",
            (unsigned long long)nr.drTweens, (unsigned long long)nr.drHeads,
            (unsigned long long)nr.drBlock0);
    for (void* p : { (void*)nr.dDrI[0], (void*)nr.dDrI[1], (void*)nr.dDrI[2], (void*)nr.dDrI[3],
                     (void*)nr.dDrF[0], (void*)nr.dDrF[1], (void*)nr.dDrF[2], (void*)nr.dDrF[3],
                     (void*)nr.dDrX[0], (void*)nr.dDrX[1], (void*)nr.drWin[0].f10, (void*)nr.drWin[0].r,
                     (void*)nr.drWin[1].f10, (void*)nr.drWin[1].r, (void*)nr.dDrFlow,
                     (void*)nr.dDrFlowN, (void*)nr.dDrAcc })
        if (p) cudaFree(p);
    for (int i = 0; i < 4; i++) nr.dDrI[i] = nr.dDrF[i] = nullptr;
    nr.dDrX[0] = nr.dDrX[1] = nullptr;
    for (auto& wn : nr.drWin) wn = NativeRife::DrWin{};
    nr.dDrFlow = nr.dDrFlowN = nullptr;
    nr.dDrAcc = nullptr;
    for (int i = 0; i < 5; i++)
    {
        if (nr.ctxGm[i]) { delete nr.ctxGm[i]; nr.ctxGm[i] = nullptr; }
        if (nr.cfgGm[i]) { delete nr.cfgGm[i]; nr.cfgGm[i] = nullptr; }
        if (nr.engGm[i] && nr.engGm[i] != g_res.engGm[i]) delete nr.engGm[i];
        nr.engGm[i] = nullptr;
    }
    // the resident cache owns these in --resident mode (residentFree drops them); the
    // pointer compare keeps the non-resident teardown byte for byte
    if (nr.jit && nr.jit != g_res.jit) delete nr.jit;
    nr.jit = nullptr;
    if (nr.engIf && nr.engIf != g_res.engIf) delete nr.engIf;
    nr.engIf = nullptr;
    if (nr.engEnc && nr.engEnc != g_res.engEnc) delete nr.engEnc;
    nr.engEnc = nullptr;
    if (nr.rt && nr.rt != g_res.rt) delete nr.rt;
    nr.rt = nullptr;
    for (auto& e : nr.slotEv) if (e) cudaEventDestroy(e);
    nr.slotEv.clear();
    for (auto& e : nr.gmEv) if (e) { cudaEventDestroy(e); e = nullptr; }
    if (nr.capEv) { cudaEventDestroy(nr.capEv); nr.capEv = nullptr; }
    // PG p.124: mappings must go before the external memory objects, and every outstanding
    // wait must have completed before the semaphore is destroyed (the stream sync above).
    if (nr.dOutRing) { cudaFree(nr.dOutRing); nr.dOutRing = nullptr; }
    if (nr.capMip) { cudaFreeMipmappedArray(nr.capMip); nr.capMip = nullptr; nr.capArr = nullptr; }
    if (nr.semCap) { cudaDestroyExternalSemaphore(nr.semCap); nr.semCap = nullptr; }
    if (nr.emCap) { cudaDestroyExternalMemory(nr.emCap); nr.emCap = nullptr; }
    if (nr.emOut) { cudaDestroyExternalMemory(nr.emOut); nr.emOut = nullptr; }
    for (void* p : { (void*)nr.dCap, (void*)nr.dX, (void*)nr.dF[0], (void*)nr.dF[1],
                     (void*)nr.dEncHalf, (void*)nr.dT, (void*)nr.dMerged, (void*)nr.dTmp,
                     (void*)nr.dCapF, (void*)nr.dThdrIn, (void*)nr.dThdrOut, (void*)nr.dSrcG,
                     (void*)nr.dPres, (void*)nr.dVsrIn, (void*)nr.dVsrOut, (void*)nr.dFitTmp,
                     (void*)nr.dUp, (void*)nr.dUpTmp, nr.dRestIn, nr.dRestOut,
                     (void*)nr.dRestTmp, (void*)nr.dRestF, (void*)nr.dRest,
                     (void*)nr.dStaticFlag, (void*)nr.dThdrStats })
        if (p) cudaFree(p);
    if (nr.hStaticFlag) { cudaFreeHost(nr.hStaticFlag); nr.hStaticFlag = nullptr; }
    if (nr.hThdrStats) { cudaFreeHost(nr.hThdrStats); nr.hThdrStats = nullptr; }
    nr.dStaticFlag = nullptr;
    nr.dThdrStats = nullptr;
    nr.thdrStatsPending = false;
    // the GMFSS chain's buffers (sub-step 5c): per session like every other model buffer, so a
    // backend switch on the resident host gives the VRAM back even though the engines stay
    for (void* p : { nr.dGmFeat[0][0], nr.dGmFeat[0][1], nr.dGmFeat[0][2],
                     nr.dGmFeat[1][0], nr.dGmFeat[1][1], nr.dGmFeat[1][2],
                     (void*)nr.dGmHalf, (void*)nr.dGmFlow, nr.dGmMetric,
                     (void*)nr.dGmFlowP[0], (void*)nr.dGmFlowP[1],
                     (void*)nr.dGmMetP[0], (void*)nr.dGmMetP[1], (void*)nr.dGmAcc,
                     (void*)nr.dGmFa, (void*)nr.dGmFb, (void*)nr.dGmFc, (void*)nr.dGmFd,
                     (void*)nr.dGmT, nr.dGmOut, (void*)nr.dGmF })
        if (p) cudaFree(p);
    for (int s = 0; s < 2; s++) for (int l = 0; l < 3; l++) nr.dGmFeat[s][l] = nullptr;
    nr.dGmHalf = nullptr; nr.dGmFlow = nullptr; nr.dGmMetric = nullptr;
    nr.dGmFlowP[0] = nr.dGmFlowP[1] = nullptr; nr.dGmMetP[0] = nr.dGmMetP[1] = nullptr;
    nr.dGmAcc = nullptr; nr.dGmFa = nullptr; nr.dGmFb = nullptr; nr.dGmFc = nullptr;
    nr.dGmFd = nullptr; nr.dGmT = nullptr; nr.dGmOut = nullptr; nr.dGmF = nullptr;
    nr.dCap = nullptr; nr.dX = nullptr; nr.dF[0] = nr.dF[1] = nullptr;
    nr.dEncHalf = nullptr; nr.dT = nullptr; nr.dMerged = nullptr; nr.dTmp = nullptr;
    nr.dCapF = nullptr; nr.dThdrIn = nullptr; nr.dThdrOut = nullptr; nr.dSrcG = nullptr;
    nr.dPres = nullptr; nr.dVsrIn = nullptr; nr.dVsrOut = nullptr; nr.dFitTmp = nullptr;
    nr.dUp = nullptr; nr.dUpTmp = nullptr;
    nr.dRestIn = nullptr; nr.dRestOut = nullptr; nr.dRestTmp = nullptr; nr.dRestF = nullptr; nr.dRest = nullptr;
    if (nr.cuMod && nr.cuMod != g_res.cuMod) cuModuleUnload(nr.cuMod);
    nr.cuMod = nullptr;
    if (nr.stream) { cudaStreamDestroy(nr.stream); nr.stream = nullptr; }
}

// ---- the GMFSS chain (2026-09-15, full native migration item 5 sub-step 5c) ----------------
// GMFSS_infer_u.Model.reuse() and .inference() as native launches on the group's stream: the
// five engines of sub-step 5a and the glue kernels of 5b (k_half / k_pyr / k_splatSoft /
// k_splatNorm, gated against fp64 by harness\eff\gmfss_equiv.py). Two differences from
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
    // sub-step 5e: the CPU cost of enqueueing this pair block (QPC, 100 ns units), taken only
    // on a pair that really runs the flow so the average is over comparable groups
    const bool profPair = nr.gmProf && needFlow;
    const int64_t cpu0 = profPair ? nowQpc100() : 0;
    if (profPair) cudaEventRecord(nr.gmEv[0], st);
    // feat_ext of the NEW frame, into the set the NEXT pair will read as feat0
    nr.gmCur ^= 1;
    void** fs = nr.dGmFeat[nr.gmCur];
    nvinfer1::IExecutionContext* ctx = nr.ctxGm[0];
    ctx->setTensorAddress("x", dCur);
    ctx->setTensorAddress("f1", fs[0]);
    ctx->setTensorAddress("f2", fs[1]);
    ctx->setTensorAddress("f3", fs[2]);
    if (!ctx->enqueueV3(st)) { nr.die("gmfss feat_ext enqueueV3 returned false"); return false; }
    if (!needFlow) return true;   // no tween in this group: python skips reuse() as well
    if (profPair) cudaEventRecord(nr.gmEv[1], st);
    // both halves into one (6, hh, hw) buffer: the ifnet's x as it is, gmflow's and
    // metricnet's inputs and the two image splats by pointer offset
    float* h0 = nr.dGmHalf;
    float* h1 = nr.dGmHalf + 3 * hp;
    void* a0[] = { (void*)&dPrev, &zero, &ps, &rs, &three, &nr.hw, &nr.hh, (void*)&h0, &zero };
    void* a1[] = { (void*)&dCur, &zero, &ps, &rs, &three, &nr.hw, &nr.hh, (void*)&h1, &zero };
    for (void** a : { a0, a1 })
        if (cuLaunchKernel(nr.fHalf, (nr.hw + 15) / 16, (nr.hh + 15) / 16, 1,
                           16, 16, 1, 0, (CUstream)st, a, nullptr) != CUDA_SUCCESS)
        { nr.die("gmfss half launch failed"); return false; }
    if (profPair) cudaEventRecord(nr.gmEv[2], st);
    // the fused bidir GMFlow (one call, both directions) and metricnet on the same halves
    ctx = nr.ctxGm[1];
    ctx->setTensorAddress("img0", h0);
    ctx->setTensorAddress("img1", h1);
    ctx->setTensorAddress("flow", nr.dGmFlow);
    const int64_t cpuF0 = profPair ? nowQpc100() : 0;
    if (!ctx->enqueueV3(st)) { nr.die("gmfss gmflow enqueueV3 returned false"); return false; }
    if (profPair) nr.gmCpuFlow += (double)(nowQpc100() - cpuF0) / 10000.0;
    if (profPair) cudaEventRecord(nr.gmEv[3], st);
    float* f10 = nr.dGmFlow + 2 * hp;
    void* m1 = (uint8_t*)nr.dGmMetric + hp * (nr.gmMetricHalf ? 2 : 4);
    ctx = nr.ctxGm[2];
    ctx->setTensorAddress("i0", h0);
    ctx->setTensorAddress("i1", h1);
    ctx->setTensorAddress("f01", nr.dGmFlow);
    ctx->setTensorAddress("f10", f10);
    ctx->setTensorAddress("m0", nr.dGmMetric);
    ctx->setTensorAddress("m1", m1);
    if (!ctx->enqueueV3(st)) { nr.die("gmfss metricnet enqueueV3 returned false"); return false; }
    if (profPair) cudaEventRecord(nr.gmEv[4], st);
    // the pyramids, once per pair: both flow directions in one launch (gmflow writes them as
    // one (2, 2, hh, hw) buffer) and both metrics in one (their planes are adjacent by the
    // addresses above). The flow carries the level's 0.5 / 0.25, a metric is only resampled.
    void* p0[] = { (void*)&nr.dGmFlow, &zero, &four, &nr.hw, &nr.hh, &lvl2, &s05,
                   (void*)&nr.dGmFlowP[0], &qw, &qh };
    void* p1[] = { (void*)&nr.dGmFlow, &zero, &four, &nr.hw, &nr.hh, &lvl4, &s025,
                   (void*)&nr.dGmFlowP[1], &ew, &eh };
    void* p2[] = { (void*)&nr.dGmMetric, &mHalf, &two, &nr.hw, &nr.hh, &lvl2, &s1,
                   (void*)&nr.dGmMetP[0], &qw, &qh };
    void* p3[] = { (void*)&nr.dGmMetric, &mHalf, &two, &nr.hw, &nr.hh, &lvl4, &s1,
                   (void*)&nr.dGmMetP[1], &ew, &eh };
    const int pw2[4] = { qw, ew, qw, ew }, ph2[4] = { qh, eh, qh, eh };
    void** pa[4] = { p0, p1, p2, p3 };
    for (int i = 0; i < 4; i++)
        if (cuLaunchKernel(nr.fPyr, (pw2[i] + 15) / 16, (ph2[i] + 15) / 16, 1,
                           16, 16, 1, 0, (CUstream)st, pa[i], nullptr) != CUDA_SUCCESS)
        { nr.die("gmfss pyramid launch failed"); return false; }
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
    auto splat = [&](void* in, int inHalf, int C, float* flow, void* metric, int metricHalf,
                     float s, int w, int h, float* dst) -> bool
    {
        int cc = C, ww = w, hh2 = h, ih = inHalf, mh = metricHalf, zero = 0;
        const int n = w * h;
        int ni = n;
        if (cudaMemsetAsync(nr.dGmAcc, 0, (size_t)(C + 1) * (size_t)n * sizeof(long long), st) != cudaSuccess)
        { nr.die("gmfss accumulator clear failed"); return false; }
        void* a[] = { &in, &ih, &cc, (void*)&flow, &metric, &mh, &s, &ww, &hh2, (void*)&nr.dGmAcc };
        if (cuLaunchKernel(nr.fSplatSoft, (w + 15) / 16, (h + 15) / 16, 1,
                           16, 16, 1, 0, (CUstream)st, a, nullptr) != CUDA_SUCCESS)
        { nr.die("gmfss splatSoft launch failed"); return false; }
        void* b[] = { (void*)&nr.dGmAcc, &cc, &ni, (void*)&dst, &zero };
        if (cuLaunchKernel(nr.fSplatNorm, (ni + 255) / 256, 1, 1, 256, 1, 1, 0,
                           (CUstream)st, b, nullptr) != CUDA_SUCCESS)
        { nr.die("gmfss splatNorm launch failed"); return false; }
        return true;
    };
    void** f0 = nr.dGmFeat[nr.gmCur ^ 1];    // the PREV frame's features (python's feat1x)
    void** f1 = nr.dGmFeat[nr.gmCur];        // the new frame's (feat2x)
    void* met0 = nr.dGmMetric;
    void* met1 = (uint8_t*)nr.dGmMetric + hp * me;
    float* flow0 = nr.dGmFlow;
    float* flow1 = nr.dGmFlow + 2 * hp;
    const bool prof = nr.gmProf && !nr.gmProfTween;   // the group's first tween carries the profile
    // sub-step 5e: the CPU enqueue span of EVERY tween (the ~20 launches below), the number the
    // per-tween CUDA graph decision rests on
    const int64_t cpu0 = nr.gmProf ? nowQpc100() : 0;
    if (prof) { nr.gmProfTween = true; cudaEventRecord(nr.gmEv[6], st); }
    // I1t and I2t into the fusionnet's a planes 0..2 and 6..8 (the ifnet fills 3..5 below)
    if (!splat(nr.dGmHalf, 0, 3, flow0, met0, mHalf, sA, nr.hw, nr.hh, nr.dGmFa)) return false;
    if (!splat(nr.dGmHalf + 3 * hp, 0, 3, flow1, met1, mHalf, sB, nr.hw, nr.hh,
               nr.dGmFa + 6 * hp)) return false;
    if (prof) cudaEventRecord(nr.gmEv[7], st);
    // the GMFSS IFNet, straight into a's middle three planes (python's cat([I1t, rife, I2t]))
    unsigned int bits;
    memcpy(&bits, &t, 4);
    if (cuMemsetD32Async((CUdeviceptr)nr.dGmT, bits, 1, (CUstream)st) != CUDA_SUCCESS)
    { nr.die("gmfss timestep fill failed"); return false; }
    nvinfer1::IExecutionContext* ctx = nr.ctxGm[3];
    ctx->setTensorAddress("x", nr.dGmHalf);
    ctx->setTensorAddress("timestep", nr.dGmT);
    ctx->setTensorAddress("merged", nr.dGmFa + 3 * hp);
    if (!ctx->enqueueV3(st)) { nr.die("gmfss ifnet enqueueV3 returned false"); return false; }
    if (prof) cudaEventRecord(nr.gmEv[8], st);
    // the three feature levels, both directions: the half from the engines' own flow and
    // metric, the quarter and the eighth from the pair's pyramids
    if (!splat(f0[0], fHalf, nr.gmC[0], flow0, met0, mHalf, sA, nr.hw, nr.hh, nr.dGmFb)) return false;
    if (!splat(f1[0], fHalf, nr.gmC[0], flow1, met1, mHalf, sB, nr.hw, nr.hh,
               nr.dGmFb + (size_t)nr.gmC[0] * hp)) return false;
    if (!splat(f0[1], fHalf, nr.gmC[1], nr.dGmFlowP[0], nr.dGmMetP[0], 0, sA,
               nr.hw / 2, nr.hh / 2, nr.dGmFc)) return false;
    if (!splat(f1[1], fHalf, nr.gmC[1], nr.dGmFlowP[0] + 2 * qp, nr.dGmMetP[0] + qp, 0, sB,
               nr.hw / 2, nr.hh / 2, nr.dGmFc + (size_t)nr.gmC[1] * qp)) return false;
    if (!splat(f0[2], fHalf, nr.gmC[2], nr.dGmFlowP[1], nr.dGmMetP[1], 0, sA,
               nr.hw / 4, nr.hh / 4, nr.dGmFd)) return false;
    if (!splat(f1[2], fHalf, nr.gmC[2], nr.dGmFlowP[1] + 2 * ep, nr.dGmMetP[1] + ep, 0, sB,
               nr.hw / 4, nr.hh / 4, nr.dGmFd + (size_t)nr.gmC[2] * ep)) return false;
    if (prof) cudaEventRecord(nr.gmEv[9], st);
    ctx = nr.ctxGm[4];
    ctx->setTensorAddress("a", nr.dGmFa);
    ctx->setTensorAddress("b", nr.dGmFb);
    ctx->setTensorAddress("c", nr.dGmFc);
    ctx->setTensorAddress("d", nr.dGmFd);
    ctx->setTensorAddress("out", nr.dGmOut);
    if (!ctx->enqueueV3(st)) { nr.die("gmfss fusionnet enqueueV3 returned false"); return false; }
    // python's torch.clamp(out, 0, 1) on the fp16 engine output: k_restToF is exactly that
    int n = (int)(3 * plane), half = nr.gmOutHalf ? 1 : 0;
    void* ac[] = { (void*)&nr.dGmOut, &half, &n, (void*)&nr.dGmF };
    if (cuLaunchKernel(nr.fRestToF, (n + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, ac, nullptr)
        != CUDA_SUCCESS)
    { nr.die("gmfss clamp launch failed"); return false; }
    if (prof) cudaEventRecord(nr.gmEv[10], st);
    if (nr.gmProf) { nr.gmCpuTween += (double)(nowQpc100() - cpu0) / 10000.0; nr.gmCpuTweenN++; }
    return true;
}

// SMV_LIVE_GMFSS_PROF=1: the nine spans of the group just finished (the stream is synced by the
// caller's final drain), averaged and printed every 32 groups, plus the CPU enqueue averages
// (sub-step 5e: the pair block, the gmflow enqueue inside it, and one tween)
static void nativeGmfssProfile(NativeRife& nr)
{
    static const int kSpan[9][2] = { { 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 4 }, { 4, 5 },
                                     { 6, 7 }, { 7, 8 }, { 8, 9 }, { 9, 10 } };
    for (int i = 0; i < 9; i++)
    {
        float ms = 0.0f;
        if (cudaEventElapsedTime(&ms, nr.gmEv[kSpan[i][0]], nr.gmEv[kSpan[i][1]]) == cudaSuccess)
            nr.gmAcc[i] += ms;
    }
    if (++nr.gmProfN % 32) return;
    const double n = 32.0;
    LOG("[gmfss] feat_ext %.2f | half %.2f | flow %.2f | metric %.2f | pyr %.2f || per tween: "
        "image splats %.2f | ifnet %.2f | feature splats %.2f | fusionnet+clamp %.2f ms "
        "(avg over %u groups)\n",
        nr.gmAcc[0] / n, nr.gmAcc[1] / n, nr.gmAcc[2] / n, nr.gmAcc[3] / n, nr.gmAcc[4] / n,
        nr.gmAcc[5] / n, nr.gmAcc[6] / n, nr.gmAcc[7] / n, nr.gmAcc[8] / n, nr.gmProfN);
    LOG("[gmfss-cpu] enqueue: pair block %.3f (gmflow %.3f) over %u pairs | tween %.3f ms over "
        "%u tweens\n",
        nr.gmCpuPair / (double)(nr.gmCpuPairN ? nr.gmCpuPairN : 1),
        nr.gmCpuFlow / (double)(nr.gmCpuPairN ? nr.gmCpuPairN : 1), nr.gmCpuPairN,
        nr.gmCpuTween / (double)(nr.gmCpuTweenN ? nr.gmCpuTweenN : 1), nr.gmCpuTweenN);
    for (double& a : nr.gmAcc) a = 0.0;
    nr.gmCpuPair = nr.gmCpuFlow = nr.gmCpuTween = 0.0;
    nr.gmCpuPairN = nr.gmCpuTweenN = 0;
}

// one group. Mirrors live_server.py's process_shm step for step: capture wait and read, the
// capture-release announcement, the pair encode, the tween chunks through the dynamic-batch
// engine, one pack-out per slot with its own event, then the bare end marker.
// ---- native DRBA (2026-09-21, memory priority 21 (b) step 3) ------------------------------
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
static float* drbaFrame(NativeRife& nr, uint32_t id) { return nr.dDrI[id & 3]; }
static float* drbaEnc(NativeRife& nr, uint32_t id) { return nr.dDrF[id & 3]; }

// the new packed frame into the ring, and its encode beside it
static bool nativeDrbaPush(NativeRife& nr, const float* dCur)
{
    cudaStream_t st = nr.stream;
    const size_t plane = (size_t)nr.ph * nr.pw;
    const uint32_t id = ++nr.drFid;
    float* dst = drbaFrame(nr, id);
    float* enc = drbaEnc(nr, id);
    if (cudaMemcpyAsync(dst, dCur, 3 * plane * sizeof(float), cudaMemcpyDeviceToDevice, st) != cudaSuccess)
    { nr.die("drba frame ring copy failed"); return false; }
    nvinfer1::Dims4 din{ 1, 3, nr.ph, nr.pw };
    if (!nr.ctxEnc->setInputShape("img", din))
    { nr.die("encode setInputShape rejected (shape outside the engine profile)"); return false; }
    nr.ctxEnc->setTensorAddress("img", dst);
    nr.ctxEnc->setTensorAddress("feat", nr.encHalf ? (void*)nr.dEncHalf : (void*)enc);
    if (!nr.ctxEnc->enqueueV3(st)) { nr.die("encode enqueueV3 returned false"); return false; }
    if (nr.encHalf)
    {
        int n = (int)(16 * plane);
        void* a[] = { &nr.dEncHalf, &enc, &n };
        if (cuLaunchKernel(nr.fH2f, (n + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, a, nullptr)
            != CUDA_SUCCESS)
        { nr.die("h2f launch failed"); return false; }
    }
    return true;
}

// calc_flow(frame a, frame b): block0 on the pair, then its tail (the step 2 kernels) into out
// (4 planes: flow05 * 2 | flow15 * 2)
static bool nativeDrbaFlow(NativeRife& nr, uint32_t a, uint32_t b, float* out)
{
    cudaStream_t st = nr.stream;
    const size_t plane = (size_t)nr.ph * nr.pw;
    nvinfer1::Dims4 di{ 1, 3, nr.ph, nr.pw }, df{ 1, 16, nr.ph, nr.pw };
    if (!nr.ctxB0->setInputShape("img0", di) || !nr.ctxB0->setInputShape("img1", di)
        || !nr.ctxB0->setInputShape("f0", df) || !nr.ctxB0->setInputShape("f1", df))
    { nr.die("block0 setInputShape rejected"); return false; }
    nr.ctxB0->setTensorAddress("img0", drbaFrame(nr, a));
    nr.ctxB0->setTensorAddress("img1", drbaFrame(nr, b));
    nr.ctxB0->setTensorAddress("f0", drbaEnc(nr, a));
    nr.ctxB0->setTensorAddress("f1", drbaEnc(nr, b));
    nr.ctxB0->setTensorAddress("flow", nr.dDrFlow);
    if (!nr.ctxB0->enqueueV3(st)) { nr.die("block0 enqueueV3 returned false"); return false; }
    if (cudaMemsetAsync(nr.dDrAcc, 0, 6 * plane * sizeof(long long), st) != cudaSuccess)
    { nr.die("drba accumulator clear failed"); return false; }
    void* as[] = { &nr.dDrFlow, &nr.pw, &nr.ph, &nr.dDrAcc };
    if (cuLaunchKernel(nr.fDrFlowSplat, (nr.pw + 15) / 16, (nr.ph + 15) / 16, 1, 16, 16, 1, 0,
                       (CUstream)st, as, nullptr) != CUDA_SUCCESS)
    { nr.die("drbaFlowSplat launch failed"); return false; }
    const int n = (int)plane;
    void* an[] = { &nr.dDrAcc, &nr.pw, &nr.ph, &out };
    if (cuLaunchKernel(nr.fDrFlowNorm, (n + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, an, nullptr)
        != CUDA_SUCCESS)
    { nr.die("drbaFlowNorm launch failed"); return false; }
    nr.drBlock0++;
    return true;
}

// RifeDrba._window(c): cached, or built (chained on a kept c - 1 window); nullptr = died
static NativeRife::DrWin* nativeDrbaWindow(NativeRife& nr, uint32_t c)
{
    for (auto& wn : nr.drWin) if (wn.c == c) return &wn;
    const size_t plane = (size_t)nr.ph * nr.pw;
    // keep the left neighbour (the chain), else a right one (python keeps every window >= c - 1;
    // the group builds k - 2 before k - 1 because the exe's fractions ascend, so both fit)
    const int s = (c > 1 && nr.drWin[0].c == c - 1) ? 1 : (c > 1 && nr.drWin[1].c == c - 1) ? 0
                : (nr.drWin[0].c == c + 1) ? 1 : 0;
    NativeRife::DrWin& wn = nr.drWin[s];
    const NativeRife::DrWin* left = (c > 1 && nr.drWin[1 - s].c == c - 1) ? &nr.drWin[1 - s] : nullptr;
    wn.c = 0;
    if (left)
    {
        if (cudaMemcpyAsync(wn.f10, left->r + 2 * plane, 2 * plane * sizeof(float),
                            cudaMemcpyDeviceToDevice, nr.stream) != cudaSuccess)
        { nr.die("drba window chain copy failed"); return nullptr; }
    }
    else
    {
        if (!nativeDrbaFlow(nr, c, c - 1, nr.dDrFlowN)) return nullptr;
        if (cudaMemcpyAsync(wn.f10, nr.dDrFlowN, 2 * plane * sizeof(float),
                            cudaMemcpyDeviceToDevice, nr.stream) != cudaSuccess)
        { nr.die("drba window left flow copy failed"); return nullptr; }
    }
    if (!nativeDrbaFlow(nr, c, c + 1, wn.r)) return nullptr;
    wn.c = c;
    return &wn;
}

// one tween of the group at fraction f into dMerged; held = python's None (the slot shows the
// lagged real frame). nHist = frames in the history (python's len(hist), at most 4). plain =
// plain pair RIFE on (k-2, k-1) at t = f whatever f is (the offline tail window, priority 24
// step 3c: render_loops.drba_loop's last window has no right frame).
static bool nativeDrbaTween(NativeRife& nr, float f, int nHist, bool& held, bool plain = false)
{
    cudaStream_t st = nr.stream;
    const size_t plane = (size_t)nr.ph * nr.pw;
    const uint32_t k = nr.drFid;
    held = false;
    NativeRife::DrWin* wn = nullptr;
    int side = +1;
    float tt = f;
    if (plain) {}
    else if (f >= 0.5f)
    {
        if (nHist < 3) { held = true; return true; }
        if (!(wn = nativeDrbaWindow(nr, k - 1))) return false;
        side = -1;
        tt = 1.0f - f;
    }
    else if (nHist >= 4)
    {
        if (!(wn = nativeDrbaWindow(nr, k - 2))) return false;
    }
    else if (nHist != 3) { held = true; return true; }
    // x = cat(I1, I0) on side -1, cat(I1, I2) on side +1 and for the head's plain pair
    const int xi = side < 0 ? 0 : 1;
    const uint32_t i1 = side < 0 ? k - 1 : k - 2, i0 = side < 0 ? k - 2 : k - 1;
    if (nr.drXFor[xi] != k)
    {
        if (cudaMemcpyAsync(nr.dDrX[xi], drbaFrame(nr, i1), 3 * plane * sizeof(float),
                            cudaMemcpyDeviceToDevice, st) != cudaSuccess
            || cudaMemcpyAsync(nr.dDrX[xi] + 3 * plane, drbaFrame(nr, i0), 3 * plane * sizeof(float),
                               cudaMemcpyDeviceToDevice, st) != cudaSuccess)
        { nr.die("drba x copy failed"); return false; }
        nr.drXFor[xi] = k;
    }
    if (!wn)
    {
        // the head: plain pair RIFE at t = f, a constant timestep map (python: base + float(t))
        unsigned int bits;
        memcpy(&bits, &f, 4);
        if (cuMemsetD32Async((CUdeviceptr)nr.dT, bits, plane, (CUstream)st) != CUDA_SUCCESS)
        { nr.die("timestep fill failed"); return false; }
        nr.drHeads++;
    }
    else
    {
        if (cudaMemsetAsync(nr.dDrAcc, 0, 2 * plane * sizeof(long long), st) != cudaSuccess)
        { nr.die("drba accumulator clear failed"); return false; }
        void* as[] = { &wn->f10, &wn->r, &side, &tt, &nr.pw, &nr.ph, &nr.dDrAcc };
        if (cuLaunchKernel(nr.fDrDrmSplat, (nr.pw + 15) / 16, (nr.ph + 15) / 16, 1, 16, 16, 1, 0,
                           (CUstream)st, as, nullptr) != CUDA_SUCCESS)
        { nr.die("drbaDrmSplat launch failed"); return false; }
        const int n = (int)plane;
        void* an[] = { &nr.dDrAcc, &wn->f10, &wn->r, &side, &tt, &nr.pw, &nr.ph, &nr.dT };
        if (cuLaunchKernel(nr.fDrDrmNorm, (n + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, an, nullptr)
            != CUDA_SUCCESS)
        { nr.die("drbaDrmNorm launch failed"); return false; }
    }
    nvinfer1::Dims4 dx{ 1, 6, nr.ph, nr.pw }, dts{ 1, 1, nr.ph, nr.pw }, df{ 1, 16, nr.ph, nr.pw };
    if (!nr.ctxIf->setInputShape("x", dx) || !nr.ctxIf->setInputShape("timestep", dts)
        || !nr.ctxIf->setInputShape("f0", df) || !nr.ctxIf->setInputShape("f1", df))
    { nr.die("IFNet setInputShape rejected (shape outside the engine profile)"); return false; }
    nr.ctxIf->setTensorAddress("x", nr.dDrX[xi]);
    nr.ctxIf->setTensorAddress("timestep", nr.dT);
    nr.ctxIf->setTensorAddress("f0", drbaEnc(nr, i1));
    nr.ctxIf->setTensorAddress("f1", drbaEnc(nr, i0));
    nr.ctxIf->setTensorAddress("merged", nr.dMerged);
    if (!nr.ctxIf->enqueueV3(st))
    { nr.die("IFNet enqueueV3 returned false (outputs would be garbage)"); return false; }
    nr.drTweens++;
    return true;
}

// Restore (live 2026-09-15, offline 2026-09-22): _Fit._restore / render_passes.restore on one
// model-size planar source, the result folded into dst (tw x th). false = a launch failed (die
// was called); an engine enqueue refusal drops the pass for the rest of the session instead
// (python's rule), the caller then continues with the unrestored source.
static bool nativeRestoreRun(NativeRife& nr, const float* s, int ps_, int rs_, float* dst, int tw, int th)
{
    cudaStream_t st = nr.stream;
    if (nr.restHalfIn)
    {
        void* a[] = { (void*)&s, &ps_, &rs_, &nr.w, &nr.h, &nr.dRestIn };
        if (cuLaunchKernel(nr.fRestIn, (nr.w + 15) / 16, (nr.h + 15) / 16, 1,
                           16, 16, 1, 0, (CUstream)st, a, nullptr) != CUDA_SUCCESS)
        { nr.die("restIn launch failed"); return false; }
    }
    else
    {
        for (int c = 0; c < 3; c++)
            if (cudaMemcpy2DAsync((float*)nr.dRestIn + (size_t)c * nr.w * nr.h, (size_t)nr.w * 4,
                                  s + (size_t)c * ps_, (size_t)rs_ * 4, (size_t)nr.w * 4, nr.h,
                                  cudaMemcpyDeviceToDevice, st) != cudaSuccess)
            { nr.die("restore input copy failed"); return false; }
    }
    nvinfer1::Dims4 din{ 1, 3, nr.h, nr.w };
    nr.ctxRest->setTensorAddress("x", nr.dRestIn);
    nr.ctxRest->setTensorAddress("y", nr.dRestOut);
    if (!nr.ctxRest->setInputShape("x", din) || !nr.ctxRest->enqueueV3(st))
    {
        LOG("native: restore failed, dropping the pass for the rest of the %s\n", g_offline ? "render" : "session");
        nr.restFailed = true;
        return true;
    }
    int half = nr.restHalfOut ? 1 : 0, w4 = 4 * nr.w, h4 = 4 * nr.h, ps4 = w4 * h4;
    if (th <= h4)
    {
        // realesr.fit: the antialiased pair when the target height shrinks (an exact copy
        // at 4x itself), with `out.clamp(0,1)` folded into the taps
        void* ah[] = { &nr.dRestOut, &half, &ps4, &w4, &w4, &h4, &nr.dRestTmp, &tw };
        if (cuLaunchKernel(nr.fRestFoldH, (tw + 15) / 16, (h4 + 15) / 16, 1,
                           16, 16, 1, 0, (CUstream)st, ah, nullptr) != CUDA_SUCCESS)
        { nr.die("restFoldH launch failed"); return false; }
        void* av[] = { &nr.dRestTmp, &tw, &h4, &dst, &th };
        if (cuLaunchKernel(nr.fRestFoldV, (tw + 15) / 16, (th + 15) / 16, 1,
                           16, 16, 1, 0, (CUstream)st, av, nullptr) != CUDA_SUCCESS)
        { nr.die("restFoldV launch failed"); return false; }
        return true;
    }
    // an enlarging target (above 4x): plain bicubic from the clamped fp32 copy
    int n = 3 * ps4;
    void* a0[] = { &nr.dRestOut, &half, &n, &nr.dRestF };
    if (cuLaunchKernel(nr.fRestToF, (n + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, a0, nullptr) != CUDA_SUCCESS)
    { nr.die("restToF launch failed"); return false; }
    void* a1[] = { &nr.dRestF, &ps4, &w4, &w4, &h4, &dst, &tw, &th };
    if (cuLaunchKernel(nr.fFitPlanar, (tw + 15) / 16, (th + 15) / 16, 1,
                       16, 16, 1, 0, (CUstream)st, a1, nullptr) != CUDA_SUCCESS)
    { nr.die("fitPlanar (restore) launch failed"); return false; }
    int n2 = 3 * tw * th;
    void* a2[] = { &dst, &n2 };
    if (cuLaunchKernel(nr.fClamp01, (n2 + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, a2, nullptr) != CUDA_SUCCESS)
    { nr.die("clamp01 launch failed"); return false; }
    return true;
}

// Offline TrueHDR (priority 24 step 2c): the previous frame's statistics block reaches the
// accumulators once the stream is known idle (the TrueHDR sync below, or the render's end).
static void nativeOfflineThdrDrain(NativeRife& nr)
{
    if (!nr.thdrStatsPending) return;
    nr.thdrStatsPending = false;
    if (nr.thdrAcc) nr.thdrAcc->add(nr.hThdrStats, (uint64_t)nr.dw * nr.dh);
    for (; nr.thdrRepeat; nr.thdrRepeat--)
        if (nr.thdrAcc) nr.thdrAcc->repeat();
}

// a held slot re-sent from the previous real frame's finished bytes: its statistics record
// is that frame's, still pending (read one frame late), so the repeat waits for the drain
static void nativeOfflineThdrRepeat(NativeRife& nr)
{
    if (nr.thdrStatsPending) nr.thdrRepeat++;
    else if (nr.thdrAcc) nr.thdrAcc->repeat();
}

static float halfToFloat(uint16_t h);   // main.cpp, after the parts

// --nr-delta PATH (priority 24 step 7, the preview's change mask, preview.py's _on_nr): the
// largest channel change of the DLSS 5 pass per pixel, |after - before| with `before` the pass's
// fp32 input planes (tight, tw x th each) and `after` its fp16 output clamped to 0..1 like
// k_nrOut, tw x th float32, rewritten per frame (the last frame's). A failed write costs only
// the mask.
static void nativeOfflineNrDelta(const NativeRife& nr, const std::vector<float>& before, int tw, int th)
{
    const size_t n = (size_t)tw * th;
    std::vector<float> d(n);
    for (size_t o = 0; o < n; o++)
    {
        float m = 0.0f;
        for (int c = 0; c < 3; c++)
        {
            float a = halfToFloat(nr.hNrOut[o * 4 + c]);
            a = a < 0.0f ? 0.0f : (a > 1.0f ? 1.0f : a);
            const float v = fabsf(a - before[c * n + o]);
            if (v > m) m = v;
        }
        d[o] = m > 1.0f ? 1.0f : m;
    }
    FILE* f = nullptr;
    if (_wfopen_s(&f, nr.nrDeltaPath.c_str(), L"wb") || !f || fwrite(d.data(), sizeof(float), d.size(), f) != d.size())
        LOG("offline: cannot write the DLSS 5 change map %s\n", wideToUtf8(nr.nrDeltaPath).c_str());
    if (f) fclose(f);
}

// Offline DLSS 5 (priority 24 step 2d): dlssnr.py process() on one output-size frame, planar
// (R, G, B) with its strides, into nr.dPres (tight, dw x dh). The NR core runs dlssnr.exe's
// own path in this process (startup on a private D3D12 device, renderFrame through its
// upload / readback staging, Reset on the first frame only: offline accumulates, live does
// not). An evaluate that fails turns the pass off for the rest of the render with a line and
// leaves dPres untouched (nr.nrFailed); python restarts its host once first, this process
// cannot (NGX has no teardown), so it goes straight to python's second-failure rule.
static bool nativeOfflineNr(NativeRife& nr, const float* src, int ps, int rs)
{
    cudaStream_t st = nr.stream;
    int tw = nr.dw, th = nr.dh;
    const size_t bytes = (size_t)tw * th * 8;
    void* a[] = { (void*)&src, &ps, &rs, &tw, &th, &nr.dNrIo };
    if (cuLaunchKernel(nr.fNrIn, (tw + 15) / 16, (th + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a, nullptr) != CUDA_SUCCESS
        || cudaMemcpyAsync(nr.hNrIn, nr.dNrIo, bytes, cudaMemcpyDeviceToHost, st) != cudaSuccess
        || cudaStreamSynchronize(st) != cudaSuccess)
    { nr.die("DLSS 5 input staging failed"); return false; }
    std::vector<float> before;   // --nr-delta: the pass's fp32 input (the stream is idle here)
    if (!nr.nrDeltaPath.empty())
    {
        before.resize((size_t)3 * tw * th);
        for (int c = 0; c < 3 && !before.empty(); c++)
            if (cudaMemcpy2D(before.data() + (size_t)c * tw * th, (size_t)tw * sizeof(float), src + (size_t)c * ps,
                             (size_t)rs * sizeof(float), (size_t)tw * sizeof(float), th, cudaMemcpyDeviceToHost) != cudaSuccess)
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
    if (ms > nr.nrMaxMs) nr.nrMaxMs = ms;
    nr.nrFirst = false;
    if (!before.empty()) nativeOfflineNrDelta(nr, before, tw, th);
    void* b[] = { &nr.dNrIo, &tw, &th, &nr.dPres };
    if (cudaMemcpyAsync(nr.dNrIo, nr.hNrOut, bytes, cudaMemcpyHostToDevice, st) != cudaSuccess
        || cuLaunchKernel(nr.fNrOut, (tw + 15) / 16, (th + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, b, nullptr) != CUDA_SUCCESS)
    { nr.die("DLSS 5 output staging failed"); return false; }
    return true;
}

// rtxvideo.run_hdr on the frame nativeOfflineEmit staged in dThdrIn / dSrcG: the bridge eval,
// then the colour mode, the x2rgb10le words into dO and the frame's statistics. A failed eval
// fails the render (python's run_hdr raises; the encoder is already on the HDR format).
static bool nativeOfflineThdr(NativeRife& nr, uint8_t* dO)
{
    cudaStream_t st = nr.stream;
    int tw = nr.dw, th = nr.dh;
    // the bridge reads dThdrIn with a synchronous cuMemcpy2D on the legacy default stream,
    // which is not ordered against this non-blocking stream: sync first (live's rule)
    if (cudaStreamSynchronize(st) != cudaSuccess) { nr.die("stream sync before the TrueHDR eval failed"); return false; }
    nativeOfflineThdrDrain(nr);
    const RtxRect rc{ 0, 0, (uint32_t)tw, (uint32_t)th };
    const int64_t t0 = nowQpc100();
    const unsigned int rv = g_rtxb.evalThdr(nr.dThdrIn, nr.dThdrOut, rc, rc, &nr.thdr);
    // MEASURED on live, do not remove: the bridge's output copy lands asynchronously on the
    // legacy default stream, so the eval returning proves nothing about dThdrOut
    if (cudaDeviceSynchronize() != cudaSuccess) { nr.die("TrueHDR eval sync failed"); return false; }
    const double ms = (double)(nowQpc100() - t0) / 10000.0;
    nr.thdrMs += ms;
    nr.thdrN++;
    if (ms > nr.thdrMaxMs) nr.thdrMaxMs = ms;
    if (rv != 1u)
    {
        LOG("[rtx] TrueHDR eval failed (rc %u)\n", rv);
        return false;
    }
    uint32_t* dW = (uint32_t*)dO;
    uint32_t* dHist = (uint32_t*)nr.dThdrStats;
    uint32_t* dMisc = dHist + 1024;
    double* dSum = (double*)(nr.dThdrStats + 1024 * 4 + 16);
    void* a[] = { &nr.dThdrOut, &nr.dSrcG, &tw, &th, &nr.rtxMode, &nr.rtxVib, &nr.rtxSb,
                  &dW, &dHist, &dMisc, &dSum };
    if (cudaMemsetAsync(nr.dThdrStats, 0, kThdrStatsBytes, st) != cudaSuccess
        || cuLaunchKernel(nr.fThdrOut, (tw + 15) / 16, (th + 15) / 16, 1,
                          16, 16, 1, 0, (CUstream)st, a, nullptr) != CUDA_SUCCESS
        || cudaMemcpyAsync(nr.hThdrStats, nr.dThdrStats, kThdrStatsBytes, cudaMemcpyDeviceToHost, st) != cudaSuccess)
    { nr.die("thdrOut launch failed"); return false; }
    nr.thdrStatsPending = true;
    return true;
}

// The offline pass chain (priority 24 step 2, 2026-09-22): render_passes.Passes.run on one
// planar model-size frame (ps / rs strides, nr.w x nr.h), then to_bytes' quantisation, into
// dO as tight rgb48le (out16) or rgb24 at the output size nr.dw x nr.dh. Order as python:
// Restore (back to the model size when RTX VSR follows, else folded straight to the output
// size), the resize (RTX VSR when it runs, else clamped bicubic; offline only enlarges, the
// downscale is folded into the decode), DLSS 5 (step 2d, nativeOfflineNr), RCAS last. With RTX HDR (step 2c) the SDR result goes
// through nativeOfflineThdr instead of the quantisation and dO holds x2rgb10le words. A pass
// that fails is dropped for the rest of the render with a line, as python does. false = a
// launch failed.
static bool nativeOfflineEmit(NativeRife& nr, const float* src, int ps, int rs, uint8_t* dO, bool out16)
{
    cudaStream_t st = nr.stream;
    const int tw = nr.dw, th = nr.dh;
    const bool resize = tw != nr.w || th != nr.h;
    const bool vsrNow = nr.vsr && !nr.vsrFailed && g_rtxb.created;
    bool staged = false;   // nr.dPres holds the output-size frame
    if (nr.restore && nr.ctxRest && !nr.restFailed)
    {
        if (vsrNow)
        {
            if (!nativeRestoreRun(nr, src, ps, rs, nr.dRest, nr.w, nr.h)) return false;
            if (!nr.restFailed) { src = nr.dRest; ps = nr.w * nr.h; rs = nr.w; }
        }
        else
        {
            if (!nativeRestoreRun(nr, src, ps, rs, nr.dPres, tw, th)) return false;
            staged = !nr.restFailed;
        }
    }
    if (!staged && resize)
    {
        bool haveVsr = false;
        if (vsrNow)
        {
            void* a[] = { (void*)&src, (void*)&ps, (void*)&rs, &nr.w, &nr.h, &nr.dVsrIn };
            if (cuLaunchKernel(nr.fPackBgraRgb, (nr.w + 15) / 16, (nr.h + 15) / 16, 1,
                               16, 16, 1, 0, (CUstream)st, a, nullptr) != CUDA_SUCCESS)
            { nr.die("packBgraRgb launch failed"); return false; }
            // the bridge copies ride the legacy default stream (rtxvideo.py run_vsr)
            if (cudaStreamSynchronize(st) != cudaSuccess) { nr.die("VSR input sync failed"); return false; }
            const RtxRect ri{ 0, 0, (uint32_t)nr.w, (uint32_t)nr.h };
            const RtxRect ro{ 0, 0, (uint32_t)tw, (uint32_t)th };
            const unsigned int rv = g_rtxb.evalVsr(nr.dVsrIn, nr.dVsrOut, ri, ro, &nr.vsrSet);
            if (cudaDeviceSynchronize() != cudaSuccess) { nr.die("VSR eval sync failed"); return false; }
            if (rv == 1u) haveVsr = true;
            else
            {
                LOG("[rtx] VSR run failed (rc %u), using bicubic for the rest of the render\n", rv);
                nr.vsrFailed = true;
            }
        }
        if (haveVsr)
        {
            void* a[] = { &nr.dVsrOut, (void*)&tw, (void*)&th, &nr.dPres };
            if (cuLaunchKernel(nr.fUnpackBgraRgb, (tw + 15) / 16, (th + 15) / 16, 1,
                               16, 16, 1, 0, (CUstream)st, a, nullptr) != CUDA_SUCCESS)
            { nr.die("unpackBgraRgb launch failed"); return false; }
        }
        else
        {
            void* a[] = { (void*)&src, (void*)&ps, (void*)&rs, &nr.w, &nr.h, &nr.dPres, (void*)&tw, (void*)&th };
            if (cuLaunchKernel(nr.fFitPlanar, (tw + 15) / 16, (th + 15) / 16, 1,
                               16, 16, 1, 0, (CUstream)st, a, nullptr) != CUDA_SUCCESS)
            { nr.die("fitPlanar launch failed"); return false; }
            int n = 3 * tw * th;
            void* a2[] = { &nr.dPres, &n };
            if (cuLaunchKernel(nr.fClamp01, (n + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, a2, nullptr) != CUDA_SUCCESS)
            { nr.die("clamp01 launch failed"); return false; }
        }
        staged = true;
    }
    if (nr.nrHost && !nr.nrFailed)
    {
        // DLSS 5 at the output size, after the resize and before RCAS (render_passes.run)
        const float* nin = staged ? nr.dPres : src;
        if (!nativeOfflineNr(nr, nin, staged ? tw * th : ps, staged ? tw : rs)) return false;
        if (!nr.nrFailed) staged = true;
    }
    if (nr.sharpen > 0.0f)
    {
        // RCAS reads a tight dw x dh frame: an unresized, unrestored source is copied in first
        if (!staged)
            for (int c = 0; c < 3; c++)
                if (cudaMemcpy2DAsync(nr.dPres + (size_t)c * tw * th, (size_t)tw * 4, src + (size_t)c * ps,
                                      (size_t)rs * 4, (size_t)tw * 4, th, cudaMemcpyDeviceToDevice, st) != cudaSuccess)
                { nr.die("sharpen input copy failed"); return false; }
        if (nr.rtxHdr)
        {
            // RCAS into the TrueHDR input (the BGRA8 bridge frame + the unquantised source)
            void* a[] = { &nr.dPres, (void*)&tw, (void*)&th, &nr.sharpen, &nr.dThdrIn, &nr.dSrcG };
            if (cuLaunchKernel(nr.fRcasThdrIn, (tw + 15) / 16, (th + 15) / 16, 1,
                               16, 16, 1, 0, (CUstream)st, a, nullptr) != CUDA_SUCCESS)
            { nr.die("rcasThdrIn launch failed"); return false; }
            return nativeOfflineThdr(nr, dO);
        }
        int o16 = out16 ? 1 : 0;
        void* a[] = { &nr.dPres, (void*)&tw, (void*)&th, &nr.sharpen, &dO, &o16 };
        if (cuLaunchKernel(nr.fRcasOutRaw, (tw + 15) / 16, (th + 15) / 16, 1,
                           16, 16, 1, 0, (CUstream)st, a, nullptr) != CUDA_SUCCESS)
        { nr.die("rcasOutRaw launch failed"); return false; }
        return true;
    }
    const float* fin = staged ? nr.dPres : src;
    int fps = staged ? tw * th : ps, frs = staged ? tw : rs;
    if (nr.rtxHdr)
    {
        void* a[] = { (void*)&fin, &fps, &frs, (void*)&tw, (void*)&th, &nr.dThdrIn, &nr.dSrcG };
        if (cuLaunchKernel(nr.fThdrIn, (tw + 15) / 16, (th + 15) / 16, 1,
                           16, 16, 1, 0, (CUstream)st, a, nullptr) != CUDA_SUCCESS)
        { nr.die("thdrIn launch failed"); return false; }
        return nativeOfflineThdr(nr, dO);
    }
    void* a[] = { (void*)&fin, &fps, &frs, (void*)&tw, (void*)&th, &dO };
    if (cuLaunchKernel(out16 ? nr.fPackOutRaw16 : nr.fPackOutRaw8, (tw + 15) / 16, (th + 15) / 16, 1,
                       16, 16, 1, 0, (CUstream)st, a, nullptr) != CUDA_SUCCESS)
    { nr.die("packOutRaw (passes) launch failed"); return false; }
    return true;
}

static bool nativeGroup(NativeRife& nr, const std::vector<uint8_t>& msg)
{
    if (msg.size() < 4) return false;
    uint32_t nfr = 0;
    memcpy(&nfr, msg.data(), 4);
    const float* fr = (const float*)(msg.data() + 4);
    if (msg.size() < 4 + 4ull * nfr) return false;
    if (nfr > nr.slots) nfr = nr.slots;
    // no-engine mode (Identity.process_shm): an empty group stays empty (the pair advances,
    // nothing presented), any other group is exactly ONE real frame through the effects
    // chain, whatever fractions the pair clock asked for
    static const float kReal = 1.0f;
    if (nr.noEngine && nfr) { nfr = 1; fr = &kReal; }

    ++nr.seq;
    const uint32_t set = nr.seq % 2;
    const size_t plane = (size_t)nr.ph * nr.pw;
    cudaStream_t st = nr.stream;

    // (1) the exe signalled the capture fence with this sequence number before writing us
    cudaExternalSemaphoreWaitParams wp{};
    wp.params.fence.value = nr.seq;
    if (cudaWaitExternalSemaphoresAsync(&nr.semCap, &wp, 1, st) != cudaSuccess)
    { nr.die("wait external semaphore failed"); return false; }
    // (2) read the shared capture texture; the event marks the exe's texture as free again
    const size_t capRow = (size_t)nr.cw * (nr.hdr ? 8 : 4);
    if (cudaMemcpy2DFromArrayAsync(nr.dCap, capRow, nr.capArr, 0, 0,
                                   capRow, nr.ch, cudaMemcpyDeviceToDevice, st) != cudaSuccess)
    { nr.die("capture memcpy2DFromArray failed"); return false; }
    cudaEventRecord(nr.capEv, st);
    bool capSent = false;

    // (3) the previous cur becomes prev, in place, then pack the new frame into the cur half
    // (native DRBA keeps its own history ring, so it skips the prev copy)
    if (nr.havePrev && !nr.noEngine && !nr.drba)
        cudaMemcpyAsync(nr.dX, nr.dX + 3 * plane, 3 * plane * sizeof(float),
                        cudaMemcpyDeviceToDevice, st);
    float* dCur = nr.dX + 3 * plane;
    // WO-23: live RTX TrueHDR runs ONCE PER REAL FRAME at capture resolution, mirroring
    // _cap_to_pq2020. capEv was recorded above, before this, so the exe's capture texture is
    // released exactly as early as on every other route even though the bridge's own
    // cuMemcpy2D calls are host synchronous.
    bool rtxThis = nr.rtxHdr && !nr.rtxFailed && g_rtxb.created;
    if (rtxThis)
    {
        void* ae[] = { &nr.dCap, &nr.cw, &nr.ch, &nr.sdrScale, &nr.dThdrIn, &nr.dSrcG };
        if (cuLaunchKernel(nr.fSdrEncode, (nr.cw + 15) / 16, (nr.ch + 15) / 16, 1,
                           16, 16, 1, 0, (CUstream)st, ae, nullptr) != CUDA_SUCCESS)
        { nr.die("sdrEncode launch failed"); return false; }
        // the bridge reads dThdrIn with a synchronous cuMemcpy2D on the legacy default
        // stream, which is NOT ordered against this non-blocking stream: sync first.
        if (cudaStreamSynchronize(st) != cudaSuccess)
        { nr.die("stream sync before the TrueHDR eval failed"); return false; }
        const RtxRect rc{ 0, 0, (uint32_t)nr.cw, (uint32_t)nr.ch };
        const int64_t t0 = nowQpc100();
        const unsigned int rv = g_rtxb.evalThdr(nr.dThdrIn, nr.dThdrOut, rc, rc, &nr.thdr);
        // MEASURED, do not remove: the bridge's array to device cuMemcpy2D is a DEVICE TO
        // DEVICE copy, which is asynchronous with respect to the host and lands on the legacy
        // default stream, so evalThdr returning proves nothing about dThdrOut. rtxvideo.py is
        // immune because _eval_thdr calls torch.cuda.synchronize() right after the eval; the
        // native host needs the same wait, and without it k_thdrColor consumed the PREVIOUS
        // frame's TrueHDR output (the source's colour at a stale frame's luminance).
        cudaDeviceSynchronize();
        const double ms = (double)(nowQpc100() - t0) / 10000.0;
        nr.thdrMs += ms;
        nr.thdrN++;
        if (ms > nr.thdrMaxMs) nr.thdrMaxMs = ms;
        if (rv != 1u)
        {
            // python doctrine (~101): log ONE line and run the faithful convert from here on
            LOG("native: live TrueHDR eval failed (%u), faithful PQ for the rest of the session\n", rv);
            nr.rtxFailed = true;
            rtxThis = false;
        }
        else
        {
            void* ac[] = { &nr.dThdrOut, &nr.dSrcG, &nr.cw, &nr.ch, &nr.dCapF,
                           &nr.rtxMode, &nr.rtxVib, &nr.rtxSb };
            if (cuLaunchKernel(nr.fThdrColor, (nr.cw + 15) / 16, (nr.ch + 15) / 16, 1,
                               16, 16, 1, 0, (CUstream)st, ac, nullptr) != CUDA_SUCCESS)
            { nr.die("thdrColor launch failed"); return false; }
        }
    }
    {
        const int ps = (int)plane;
        if (rtxThis)
        {
            // the PQ frame already exists at capture resolution, so resize ON PQ at every
            // image scale; at 1.00 the triangle filter sits on identity positions (single
            // tap, weight 1) and the pair is an exact copy plus the replicate pad.
            void* a1[] = { &nr.dCapF, &nr.cw, &nr.ch, &nr.dTmp, &nr.w };
            if (cuLaunchKernel(nr.fResizeHf, (nr.w + 15) / 16, (nr.ch + 15) / 16, 1,
                               16, 16, 1, 0, (CUstream)st, a1, nullptr) != CUDA_SUCCESS)
            { nr.die("resizeHf (TrueHDR) launch failed"); return false; }
            void* a2[] = { &nr.dTmp, &nr.w, &nr.ch, &dCur, &nr.h, &nr.ph, &nr.pw, (void*)&ps };
            if (cuLaunchKernel(nr.fResizeV, (nr.pw + 15) / 16, (nr.ph + 15) / 16, 1,
                               16, 16, 1, 0, (CUstream)st, a2, nullptr) != CUDA_SUCCESS)
            { nr.die("resizeV (TrueHDR) launch failed"); return false; }
        }
        else if (nr.w == nr.cw && nr.h == nr.ch)
        {
            void* a[] = { &nr.dCap, &nr.cw, &nr.ch, &dCur, &nr.ph, &nr.pw, (void*)&ps };
            if (cuLaunchKernel(nr.hdr ? nr.fPackInDirectHdr : nr.fPackInDirect,
                               (nr.pw + 15) / 16, (nr.ph + 15) / 16, 1,
                               16, 16, 1, 0, (CUstream)st, a, nullptr) != CUDA_SUCCESS)
            { nr.die("packInDirect launch failed"); return false; }
        }
        else
        {
            if (nr.hdr)
            {
                // convert the whole capture to PQ first, then resize ON PQ (the WO-8 fill rule
                // python follows: rescale-on-PQ, never rescale scRGB and convert after)
                void* a0[] = { &nr.dCap, &nr.cw, &nr.ch, &nr.dCapF };
                if (cuLaunchKernel(nr.fPqPlanar, (nr.cw + 15) / 16, (nr.ch + 15) / 16, 1,
                                   16, 16, 1, 0, (CUstream)st, a0, nullptr) != CUDA_SUCCESS)
                { nr.die("pqPlanar launch failed"); return false; }
            }
            void* a1[] = { nr.hdr ? (void*)&nr.dCapF : (void*)&nr.dCap,
                           &nr.cw, &nr.ch, &nr.dTmp, &nr.w };
            if (cuLaunchKernel(nr.hdr ? nr.fResizeHf : nr.fResizeH,
                               (nr.w + 15) / 16, (nr.ch + 15) / 16, 1,
                               16, 16, 1, 0, (CUstream)st, a1, nullptr) != CUDA_SUCCESS)
            { nr.die("resizeH launch failed"); return false; }
            void* a2[] = { &nr.dTmp, &nr.w, &nr.ch, &dCur, &nr.h, &nr.ph, &nr.pw, (void*)&ps };
            if (cuLaunchKernel(nr.fResizeV, (nr.pw + 15) / 16, (nr.ph + 15) / 16, 1,
                               16, 16, 1, 0, (CUstream)st, a2, nullptr) != CUDA_SUCCESS)
            { nr.die("resizeV launch failed"); return false; }
        }
    }

    // native DRBA: the new frame and its encode into the history ring; the group then shows
    // the LAGGED frame k - 1 as its real frame (the first capture: itself, one slot, exactly
    // RifeDrba's first group)
    int drHist = 0;
    float* drLag = dCur;
    if (nr.drba)
    {
        if (!nativeDrbaPush(nr, dCur)) return false;
        drHist = nr.drFid < 4 ? (int)nr.drFid : 4;
        if (drHist >= 2) drLag = drbaFrame(nr, nr.drFid - 1);
        else if (nfr > 1) nfr = 1;
    }

    // (4) the pair. How many tweens this group asks for, in fracs order, decides whether the
    // pair-level work is needed at all (python skips reuse() for a tween-less group).
    uint32_t nTween = 0;
    for (uint32_t i = 0; i < nfr; i++) if (fr[i] < 0.999f) nTween++;
    if (!nr.havePrev) nTween = 0;   // first pair: nothing to interpolate toward
    // IDENTICAL PAIR (2026-09-16): the two packed inputs are compared element by element on
    // the device and the flag is read back once. Equal = no motion exists in this pair, so the
    // tween work is dropped (exactly the tween-less group the adaptive ladder already
    // produces: the per-frame chain state, the encode or feat_ext, still runs for the next
    // pair) and every slot presents the real frame below. The readback sync costs only the
    // overlap between this group's pack and its own tweens: the compute thread already waits
    // for the whole group in drainReady(true) before it takes the next message.
    bool pairStatic = false;
    // native DRBA tests the LAGGED pair (k-2, k-1): every tween of its group lies inside it
    // (RifeDrba's static rule, three frames of history needed)
    if (nTween && g_staticHold && nr.fPairDiff && nr.dStaticFlag && nr.hStaticFlag
        && (!nr.drba || drHist >= 3))
    {
        const int n = (int)(3 * plane);
        float* dPrev = nr.drba ? drbaFrame(nr, nr.drFid - 2) : nr.dX;
        float* dNext = nr.drba ? drLag : dCur;
        if (cudaMemsetAsync(nr.dStaticFlag, 0, sizeof(int), st) != cudaSuccess)
        { nr.die("static flag clear failed"); return false; }
        void* ad[] = { &dPrev, &dNext, (void*)&n, &nr.dStaticFlag };
        if (cuLaunchKernel(nr.fPairDiff, (n + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, ad, nullptr)
            != CUDA_SUCCESS)
        { nr.die("pairDiff launch failed"); return false; }
        if (cudaMemcpyAsync(nr.hStaticFlag, nr.dStaticFlag, sizeof(int), cudaMemcpyDeviceToHost, st) != cudaSuccess
            || cudaStreamSynchronize(st) != cudaSuccess)
        { nr.die("static flag readback failed"); return false; }
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
        // GMFSS (sub-step 5c): feat_ext of the new frame, then the halves, the bidir flow,
        // the metrics and the two pyramid levels when this group interpolates
        if (!nativeGmfssPair(nr, nr.dX, dCur, nTween > 0)) return false;
    }
    else if (nr.nvof)
    {
        // the nvof model: no per-frame state (both frames go to gray8 per pair), so a
        // tween-less or identical pair costs nothing; the identical-pair gate above already
        // zeroed nTween, an identical pair never reaches the Optical Flow engine
        if (nTween && !nativeNvofPair(nr, nr.dX, dCur)) return false;
    }
    else if (nr.fruc)
    {
        // Smooth Motion: every frame is packed (the next pair's start), the bridge itself
        // runs per tween below; an identical pair got nTween = 0 above and never reaches FRUC
        if (!nativeFrucPair(nr, dCur, nTween)) return false;
    }
    else if (nr.drba)
    {
        // native DRBA: the encode already went into the ring (nativeDrbaPush), the windows
        // are built lazily by the tweens below
    }
    else if (!nr.noEngine)
    {
        nvinfer1::Dims4 din{ 1, 3, nr.ph, nr.pw };
        if (!nr.ctxEnc->setInputShape("img", din))
        { nr.die("encode setInputShape rejected (shape outside the engine profile)"); return false; }
        nr.ctxEnc->setTensorAddress("img", dCur);
        void* encOut = nr.encHalf ? (void*)nr.dEncHalf : (void*)nr.dF[nr.fCur];
        nr.ctxEnc->setTensorAddress("feat", encOut);
        if (!nr.ctxEnc->enqueueV3(st)) { nr.die("encode enqueueV3 returned false"); return false; }
        if (nr.encHalf)
        {
            int n = (int)(16 * plane);
            void* a[] = { &nr.dEncHalf, &nr.dF[nr.fCur], &n };
            if (cuLaunchKernel(nr.fH2f, (n + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, a, nullptr)
                != CUDA_SUCCESS)
            { nr.die("h2f launch failed"); return false; }
        }
    }

    // (5) slots. Tokens go out in slot-index order, exactly like _drain_ready.
    const int pitchI = (int)nr.pitch;
    uint32_t sent = 0;
    auto drainReady = [&](bool block)
    {
        if (!capSent && (block ? cudaEventSynchronize(nr.capEv) == cudaSuccess
                               : cudaEventQuery(nr.capEv) == cudaSuccess))
        { capSent = true; nr.pushTok(0x7FFFFFFFu); }
        while (sent < nfr && (block ? cudaEventSynchronize(nr.slotEv[sent]) == cudaSuccess
                                    : cudaEventQuery(nr.slotEv[sent]) == cudaSuccess))
        { nr.pushTok(sent + 1); sent++; }
    };

    uint32_t twDone = 0;      // tweens already computed
    uint32_t chunkBase = 0, chunkLen = 0;   // current chunk's [base, base+len) in tween order
    bool fail = false;

    // one presented frame from a model-size planar source into its slot. Without effects the
    // slot packer does the fit and the store in one kernel. With live effects (2026-09-15) it
    // mirrors live_server.py compose: RTX VSR (the bridge, model size -> presented size,
    // 8-bit in and out, host-synchronous like every bridge call) or the bicubic fit into the
    // planar staging frame, then RCAS in the slot store. A VSR eval failure demotes the rest
    // of the run to bicubic with one line, exactly like the python route's `self.vsr = None`.
    // A DOWNSCALING fit (2026-09-15) takes the staging frame too: the antialiased pair lands
    // there and the plain packer stores it 1:1 when no sharpen follows.
    auto packFrom = [&](const float* src, int ps, int rs, int sw, int sh, uint8_t* slot,
                        const char* what) -> bool
    {
        void* a[] = { (void*)&src, (void*)&ps, (void*)&rs, &sw, &sh, &slot,
                      (void*)&pitchI, &nr.x0, &nr.y0, &nr.dw, &nr.dh };
        if (cuLaunchKernel(nr.hdr ? nr.fPackOutHdr : nr.fPackOut,
                           (nr.dw + 15) / 16, (nr.dh + 15) / 16, 1,
                           16, 16, 1, 0, (CUstream)st, a, nullptr) != CUDA_SUCCESS)
        { nr.die(what); return false; }
        return true;
    };
    auto plainPack = [&](const float* src, uint8_t* slot, const char* what) -> bool
    { return packFrom(src, (int)plane, nr.pw, nr.w, nr.h, slot, what); };
    // one resize of a planar source (ps / rs / sw x sh) into a planar target tw x th: the
    // antialiased pair through tmp (3, sh, tw) when aa, else sampleOut's bicubic (an
    // enlarging or 1:1 resize, torch's antialias=False path)
    auto resizePlanar = [&](const float* src, int ps, int rs, int sw, int sh, bool aa,
                            float* tmp, float* dst, int tw, int th) -> bool
    {
        if (aa)
        {
            void* ah[] = { (void*)&src, &ps, &rs, &sw, &sh, &tmp, &tw };
            if (cuLaunchKernel(nr.fFitAaH, (tw + 15) / 16, (sh + 15) / 16, 1,
                               16, 16, 1, 0, (CUstream)st, ah, nullptr) != CUDA_SUCCESS)
            { nr.die("fitAaH launch failed"); return false; }
            void* av[] = { &tmp, &tw, &sh, &dst, &th };
            if (cuLaunchKernel(nr.fFitAaV, (tw + 15) / 16, (th + 15) / 16, 1,
                               16, 16, 1, 0, (CUstream)st, av, nullptr) != CUDA_SUCCESS)
            { nr.die("fitAaV launch failed"); return false; }
            return true;
        }
        void* a[] = { (void*)&src, &ps, &rs, &sw, &sh, &dst, &tw, &th };
        if (cuLaunchKernel(nr.fFitPlanar, (tw + 15) / 16, (th + 15) / 16, 1,
                           16, 16, 1, 0, (CUstream)st, a, nullptr) != CUDA_SUCCESS)
        { nr.die("fitPlanar launch failed"); return false; }
        return true;
    };
    // live Restore (2026-09-15, item 3): _Fit._restore on one model-size planar source, the
    // result folded into dst (tw x th). false = a launch failed (die was called); an engine
    // enqueue refusal drops the pass for the rest of the session instead (python's rule), the
    // caller then continues with the unrestored source.
    auto runRestore = [&](const float* s, int ps_, int rs_, float* dst, int tw, int th) -> bool
    { return nativeRestoreRun(nr, s, ps_, rs_, dst, tw, th); };
    auto storeSlot = [&](const float* src, uint8_t* slot, const char* what) -> bool
    {
        int ps = (int)plane, rs = nr.pw;
        const bool vsrNow = nr.vsr && !nr.vsrFailed && g_rtxb.created;
        const bool restNow = nr.restore && nr.ctxRest && !nr.restFailed;
        if (!vsrNow && nr.sharpen <= 0.0f && !nr.fitAa && !nr.uw && !restNow) return plainPack(src, slot, what);
        // Upscale to (item 2): the first resize lands in the internal render frame (uw x uh)
        // instead of the staging frame, then the fit takes it to (dw, dh) below
        const int tw = nr.uw ? nr.uw : nr.dw, th = nr.uw ? nr.uh : nr.dh;
        float* stage = nr.uw ? nr.dUp : nr.dPres;
        bool haveVsr = false;
        // live Restore FIRST (compose's order): back to the model size when VSR follows (the
        // source of the VSR pack below becomes the restored frame), else its fold IS the
        // first resize, straight into the stage
        bool staged = false;
        if (restNow)
        {
            if (vsrNow)
            {
                if (!runRestore(src, ps, rs, nr.dRest, nr.w, nr.h)) return false;
                if (!nr.restFailed) { src = nr.dRest; ps = nr.w * nr.h; rs = nr.w; }
            }
            else
            {
                if (!runRestore(src, ps, rs, stage, tw, th)) return false;
                staged = !nr.restFailed;
            }
        }
        if (vsrNow)
        {
            void* a[] = { (void*)&src, (void*)&ps, (void*)&rs, &nr.w, &nr.h, &nr.dVsrIn };
            if (cuLaunchKernel(nr.fPackBgra, (nr.w + 15) / 16, (nr.h + 15) / 16, 1,
                               16, 16, 1, 0, (CUstream)st, a, nullptr) != CUDA_SUCCESS)
            { nr.die("packBgra launch failed"); return false; }
            // the bridge copies ride the legacy default stream: finish the packing first,
            // and finish the eval before the stream reads its output (rtxvideo.py run_vsr)
            if (cudaStreamSynchronize(st) != cudaSuccess) { nr.die("VSR input sync failed"); return false; }
            const RtxRect ri{ 0, 0, (uint32_t)nr.w, (uint32_t)nr.h };
            const RtxRect ro{ 0, 0, (uint32_t)tw, (uint32_t)th };
            const int64_t t0 = nowQpc100();
            const unsigned int rv = g_rtxb.evalVsr(nr.dVsrIn, nr.dVsrOut, ri, ro, &nr.vsrSet);
            if (cudaDeviceSynchronize() != cudaSuccess) { nr.die("VSR eval sync failed"); return false; }
            const double ms = (nowQpc100() - t0) / 1e4;
            nr.vsrMs += ms;
            nr.vsrN++;
            if (ms > nr.vsrMaxMs) nr.vsrMaxMs = ms;
            if (rv == 1u) haveVsr = true;
            else
            {
                LOG("native: RTX VSR eval failed (rc %u), bicubic for the rest of the run\n", rv);
                nr.vsrFailed = true;
                if (nr.sharpen <= 0.0f && !nr.uw)   // VSR = enlarging fit (from the restored frame when on)
                    return packFrom(src, ps, rs, nr.w, nr.h, slot, what);
            }
        }
        if (haveVsr && nr.sharpen <= 0.0f && !nr.uw)
        {
            // the bridge output is the presented frame: into the content rect as it is
            if (cudaMemcpy2DAsync(slot + (size_t)nr.y0 * nr.pitch + (size_t)nr.x0 * 4, nr.pitch,
                                  nr.dVsrOut, (size_t)nr.dw * 4, (size_t)nr.dw * 4, nr.dh,
                                  cudaMemcpyDeviceToDevice, st) != cudaSuccess)
            { nr.die("VSR slot copy failed"); return false; }
            return true;
        }
        // the first resize target (the internal render frame, or the staging frame at the
        // presented size): the VSR output unpacked, or the bicubic fit
        if (haveVsr)
        {
            void* a[] = { &nr.dVsrOut, (void*)&tw, (void*)&th, &stage };
            if (cuLaunchKernel(nr.fUnpackBgra, (tw + 15) / 16, (th + 15) / 16, 1,
                               16, 16, 1, 0, (CUstream)st, a, nullptr) != CUDA_SUCCESS)
            { nr.die("unpackBgra launch failed"); return false; }
        }
        else if (!staged && !resizePlanar(src, ps, rs, nr.w, nr.h, nr.uw ? nr.upAa : nr.fitAa,
                                          nr.uw ? nr.dUpTmp : nr.dFitTmp, stage, tw, th))
            return false;
        if (nr.uw)
        {
            // the second resize, internal render frame -> the fit rect (_Fit._upscale's
            // second interpolate): the aa pair when it shrinks, else sampleOut's bicubic,
            // which the plain slot packer does in its own store when no sharpen follows
            if (!nr.fitAa && nr.sharpen <= 0.0f)
                return packFrom(nr.dUp, nr.uw * nr.uh, nr.uw, nr.uw, nr.uh, slot, what);
            if (!resizePlanar(nr.dUp, nr.uw * nr.uh, nr.uw, nr.uw, nr.uh, nr.fitAa,
                              nr.dFitTmp, nr.dPres, nr.dw, nr.dh))
                return false;
        }
        // no sharpen: the staging frame is the presented frame, stored 1:1 (sampleOut's
        // identity branch reads it as a dw x dh source with no pad)
        if (nr.sharpen <= 0.0f)
            return packFrom(nr.dPres, nr.dw * nr.dh, nr.dw, nr.dw, nr.dh, slot, what);
        void* a[] = { &nr.dPres, &nr.dw, &nr.dh, &nr.sharpen, &slot, (void*)&pitchI, &nr.x0, &nr.y0 };
        if (cuLaunchKernel(nr.hdr ? nr.fRcasOutHdr : nr.fRcasOut,
                           (nr.dw + 15) / 16, (nr.dh + 15) / 16, 1,
                           16, 16, 1, 0, (CUstream)st, a, nullptr) != CUDA_SUCCESS)
        { nr.die("rcasOut launch failed"); return false; }
        return true;
    };
    uint8_t* heldSlot = nullptr;   // a held pair's first real slot, copied into its other slots
    for (uint32_t i = 0; i < nfr && !fail; i++)
    {
        uint8_t* slot = nr.dOutRing + ((size_t)set * nr.slots + i) * nr.slotBytes;
        const bool real = !(fr[i] < 0.999f) || !nr.havePrev || pairStatic;
        if (real)
        {
            if (pairStatic && heldSlot)
            {
                // the held pair's remaining slots: the real frame's finished slot, copied as
                // it is, so the per-frame passes (VSR, Restore, the fit, RCAS) run ONCE
                if (cudaMemcpyAsync(slot, heldSlot, nr.slotBytes, cudaMemcpyDeviceToDevice, st)
                    != cudaSuccess)
                { nr.die("held slot copy failed"); fail = true; break; }
            }
            // the passthrough keys on identity AND no effect (the WO-35 / WO-37 lesson): with
            // sharpen, Upscale to or Restore on, real frames go through the store like every other frame
            // (never on DRBA: its real frame is the lagged k - 1, not this capture)
            else if (nr.identity && !nr.drba && !nr.hdr && nr.sharpen <= 0.0f && !nr.uw && !nr.restore)
            {
                // bit-exact passthrough, the python route's raw-frame path
                if (cudaMemcpy2DAsync(slot, nr.pitch, nr.dCap, (size_t)nr.cw * 4,
                                      (size_t)nr.cw * 4, nr.ch, cudaMemcpyDeviceToDevice, st)
                    != cudaSuccess)
                { nr.die("real slot copy failed"); fail = true; break; }
                heldSlot = slot;
            }
            else if (!storeSlot(nr.drba ? drLag : dCur, slot, "packOut (real) launch failed"))
            { fail = true; break; }
            else heldSlot = slot;
        }
        else if (nr.drba)
        {
            // native DRBA: one IFNet enqueue per tween with its own DRM timestep map, or the
            // lagged real frame where python's group holds (a head without enough history)
            bool held = false;
            if (!nativeDrbaTween(nr, fr[i], drHist, held)) { fail = true; break; }
            if (!storeSlot(held ? drLag : nr.dMerged, slot, "packOut (drba) launch failed")) { fail = true; break; }
            if (!held) twDone++;
        }
        else if (nr.gmfss)
        {
            // GMFSS: one tween at a time (no batch axis anywhere in its five engines), the
            // clamped fusionnet output then rides the shared effects and slot chain
            if (!nativeGmfssTween(nr, fr[i])) { fail = true; break; }
            if (!storeSlot(nr.dGmF, slot, "packOut (gmfss) launch failed")) { fail = true; break; }
            twDone++;
        }
        else if (nr.nvof)
        {
            // the nvof model: one splat pair per tween into the model layout, then the shared
            // effects and slot chain like every other model's output
            if (!nativeNvofTween(nr, fr[i])) { fail = true; break; }
            if (!storeSlot(nr.dNvOut, slot, "packOut (nvof) launch failed")) { fail = true; break; }
            twDone++;
        }
        else if (nr.fruc)
        {
            // Smooth Motion: one bridge call per tween (it syncs the context on entry, so the
            // previous tween's store has read dFrOut before the next unpack overwrites it)
            if (!nativeFrucTween(nr, fr[i])) { fail = true; break; }
            if (!storeSlot(nr.dFrOut, slot, "packOut (fruc) launch failed")) { fail = true; break; }
            twDone++;
        }
        else
        {
            if (twDone >= chunkBase + chunkLen)
            {
                // next chunk of tweens: no padding, exactly the WO-13 dynamic-batch contract
                chunkBase = twDone;
                chunkLen = nTween - chunkBase;
                if (chunkLen > (uint32_t)nr.batchMax) chunkLen = (uint32_t)nr.batchMax;
                // timestep planes are constant maps of t (python: base + float(t))
                // collect the chunk's t values in tween order
                float ts[64];
                if (chunkLen > 64) chunkLen = 64;
                uint32_t k = 0, seen = 0;
                for (uint32_t j = 0; j < nfr; j++)
                {
                    if (!(fr[j] < 0.999f)) continue;
                    if (seen >= chunkBase && k < chunkLen) ts[k++] = fr[j];
                    seen++;
                }
                for (uint32_t j = 0; j < chunkLen; j++)
                {
                    unsigned int bits;
                    memcpy(&bits, &ts[j], 4);
                    if (cuMemsetD32Async((CUdeviceptr)(nr.dT + (size_t)j * plane), bits,
                                         plane, (CUstream)st) != CUDA_SUCCESS)
                    { nr.die("timestep fill failed"); fail = true; break; }
                }
                if (fail) break;
                nvinfer1::Dims4 dx{ 1, 6, nr.ph, nr.pw };
                nvinfer1::Dims4 dtst{ (int)chunkLen, 1, nr.ph, nr.pw };
                nvinfer1::Dims4 df{ 1, 16, nr.ph, nr.pw };
                if (!nr.ctxIf->setInputShape("x", dx) || !nr.ctxIf->setInputShape("timestep", dtst)
                    || !nr.ctxIf->setInputShape("f0", df) || !nr.ctxIf->setInputShape("f1", df))
                { nr.die("IFNet setInputShape rejected (shape outside the engine profile)"); fail = true; break; }
                nr.ctxIf->setTensorAddress("x", nr.dX);
                nr.ctxIf->setTensorAddress("timestep", nr.dT);
                nr.ctxIf->setTensorAddress("f0", nr.dF[nr.fCur ^ 1]);
                nr.ctxIf->setTensorAddress("f1", nr.dF[nr.fCur]);
                nr.ctxIf->setTensorAddress("merged", nr.dMerged);
                if (!nr.ctxIf->enqueueV3(st))
                { nr.die("IFNet enqueueV3 returned false (outputs would be garbage)"); fail = true; break; }
            }
            const uint32_t off = twDone - chunkBase;
            const float* src = nr.dMerged + (size_t)off * 3 * plane;
            if (!storeSlot(src, slot, "packOut launch failed")) { fail = true; break; }
            twDone++;
        }
        cudaEventRecord(nr.slotEv[i], st);
        drainReady(false);
    }
    if (fail) return false;
    drainReady(true);          // sync the stragglers, exactly like _finish
    if (nr.gmProf && nr.gmProfTween) nativeGmfssProfile(nr);
    nr.pushTok(0x80000000u);   // bare end marker closes the group
    nr.havePrev = true;
    return true;
}
// ---- part 4: the compute thread and the public entry point --------------------------------

static void nativeThread(NativeRife* nrp, IDXGIAdapter1* adapter, HANDLE hTex, HANDLE hFence,
                         HANDLE hOutBuf, uint64_t outBytes, bool earlyDone)
{
    NativeRife& nr = *nrp;
    // earlyDone: PipeServer::beginNativeHandoff built everything but the output ring on its
    // own thread; this thread only binds the same device and finishes the ring
    bool ok = earlyDone ? nativeBindDevice(nr)
                        : nativeCudaDeviceInit(nr, adapter, hTex, hFence, (uint32_t)nr.cw, (uint32_t)nr.ch, nr.hdr)
                          && nativeCudaInitEarly(nr, nativeCacheDir(nr)) && nativeTrtInit(nr)
                          && nativeRtxInit(nr);
    ok = ok && nativeCudaInitLate(nr, hOutBuf, outBytes);
    {
        std::lock_guard<std::mutex> lk(nr.mInit);
        nr.initState = ok ? 1 : -1;
    }
    nr.cvInit.notify_all();
    if (!ok) { InterlockedExchange(&nr.dead, 1); nr.cvTok.notify_all(); nativeFree(nr); return; }
    LOG("native host ready: model %dx%d padded %dx%d, out %dx%d at (%d,%d), batch max %d%s%s%s%s%s%s%s%s%s%s%s\n",
        nr.w, nr.h, nr.pw, nr.ph, nr.dw, nr.dh, nr.x0, nr.y0, nr.batchMax,
        nr.noEngine ? ", engine=none (effects only)" : "",
        nr.gmfss ? ", engine=gmfss" : "", nr.nvof ? ", engine=nvof" : "", nr.fruc ? ", engine=fruc" : "",
        nr.drba ? ", engine=drba (lag 1)" : "",
        nr.rtxHdr ? ", rtxhdr=native" : "", nr.sharpen > 0.0f ? ", sharpen=native" : "",
        nr.vsr ? ", vsr=native" : "", nr.uw ? ", upscale=native" : "", nr.fitAa ? ", fit=native-aa" : "",
        nr.restore && nr.ctxRest ? ", restore=native" : "");
    for (;;)
    {
        std::vector<uint8_t> msg;
        {
            std::unique_lock<std::mutex> lk(nr.mMsg);
            nr.cvMsg.wait(lk, [&] { return !nr.msgs.empty() || nr.isDead(); });
            if (nr.msgs.empty()) break;
            msg = std::move(nr.msgs.front());
            nr.msgs.pop_front();
        }
        if (!nativeGroup(nr, msg)) break;
    }
    { std::lock_guard<std::mutex> lk(nr.mTok); InterlockedExchange(&nr.dead, 1); }
    nr.cvTok.notify_all();
    nativeFree(nr);
}

// Start the in-process host. Returns false on ANY problem (one log line says why) and the
// session ends.
static bool nativeStart(NativeRife& nr, const std::wstring& script,
                        const std::wstring& backend, int gen, uint32_t capW, uint32_t capH,
                        IDXGIAdapter1* adapter, HANDLE hTex, HANDLE hFence,
                        HANDLE hOutBuf, uint64_t outBytes,
                        uint32_t slots, uint32_t pitch, size_t slotBytes,
                        bool earlyDone)
{
    // earlyDone: PipeServer::beginNativeHandoff already ran the DLL load, the HDR config, the
    // engine handoff, the CUDA device init, the kernels, the buffers and the engine load on
    // its own thread; only the output ring (sized from the slot count) is left
    if (!nativeLoadDlls(script)) return false;
    if (!earlyDone)
    {
        if (!nativeConfigHdr(nr)) return false;
        if (!nativeHandoff(script, backend, gen, capW, capH, nr)) return false;
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
        if (nr.th.joinable()) nr.th.join();
        LOG("native host init failed\n");
        return false;
    }
    nr.started = true;
    return true;
}

static void nativeStop(NativeRife& nr)
{
    if (!nr.started) return;
    nr.die("stopping");
    if (nr.th.joinable()) nr.th.join();
    nr.started = false;
}

// a NativeRife the early thread initialised but no compute thread ever owned (a refusal in
// startNative, or a session that ends before it): free its CUDA and TRT state from the calling
// thread. Without a stream nothing CUDA was created (the stream is the first CUDA object), and
// the runtime DLL may not even be loaded, so no CUDA call is made at all then.
static void nativeDropEarly(NativeRife& nr)
{
    if (!nr.stream) return;
    nativeBindDevice(nr);
    nativeFree(nr);
}

