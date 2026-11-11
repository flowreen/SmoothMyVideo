// part of smv-live-native.inl (included there, in place); not a standalone translation unit
// ---- GMFSS's IFNet on the host's kernels --------------------------------------------------------------------------
// The GMFSS IFNet (RIFE 4.6's IFNet_HDv3: 4 IFBlocks at 8, 4, 2, 1 of the half, c = 192, 128, 96, 64, the warps between
// them, then the merge) as 47 launches of its own module smv_gmifnet_*.cubin (kFnHead's helpers + kGiA / kGiB), weights =
// engine\GMFSS_Fortuna\train_log\rife.pkl through pthStateDict (giNatLoad); no fallback (a missing /
// wrong pkl = `the GMFSS IFNet unavailable (...)`, the session refuses). It takes the 6 fp32 half planes and the fp32
// timestep and hands out merged (3 fp32 planes) straight into fusionnet's a planes 3..5. Its scratch (giArena) lives in
// the shared activation block (nativeGmShareMemory sizes it in; sound while every enqueue there runs serially on
// nr.stream), else dGiOwn.
static const char kGiA[] = R"GISRC(
// GMFSS's IFNet (RIFE 4.6's IFNet_HDv3: 4 IFBlocks at scales 8, 4, 2, 1 of the GMFSS half with c = 192, 128, 96, 64)
// on the host's own kernels; the RIFE IFNet module's design with GMFSS's inputs. The half is W x H (multiples of 32); block k (scale s = 8 >> k) works at L0 = W / s x H / s, L1 = L0 / 2, L2 = L0 / 4;
// its tmp stays at its L0 as 2 float4 quads a pixel ([quad][h][w][4]: quad 0 = the flow, quad 1 = the mask, the sixth
// channel and 2 zeros). The author's full-size tensors (the flow and the mask summed over the blocks, the warps) are
// read only where a reader needs them, by PyTorch's bilinear from those quads.
// k_giIn0     block 0's input cat(img0, img1, timestep) at L0: at an exact 1 / 8 each sample = PyTorch's bilinear mean of
//             the 2 x 2 source pixels at rows / columns 8 i + 3, 8 i + 4; channels-last fp16 [L0][16] (7 + 9 zeros)
// k_giIn<k>   block k's input (k = 1..3) at L0: cat(warp(img0), warp(img1), timestep, mask) resized + flow resized / s,
//             where the author resizes = the mean of the 4 full-size taps at rows / columns s i + s / 2 - 1, + 1 (s = 1 the
//             pixel); at each tap the flow = the earlier blocks' flow quads by bilinear x their scale summed in block
//             order, the mask = their mask planes summed the same way (fp32), the warps = bilinear taps of the fp32
//             frames at the pixel + the flow clamped into the frame (grid_sample's border rule); [L0][16] (12 + 4 zeros)
// k_giS2g16 / k_giS2g32   a 3 x 3 stride-2 conv + LeakyRelu (conv0[0] on the padded input, conv0[1]); k_giResg32 /
//             k_giResg64 the ResConvs {LeakyRelu(conv(x) * beta + x)}, the residual = the conv's own fp16 input at the
//             pixel: implicit GEMMs on the tensor cores (mma.sync m16n8k16, fp32 accumulators from the bias, a 4 x 32
//             output tile, a CG-channel group's halo and weights by cp.async)
// k_giLastQ32 / 64   ConvTranspose c -> 24 (4 x 4, stride 2, pad 1) + PixelShuffle(2): per output parity class a 2 x 2-
//             tap conv over L2 on the tensor cores, the 4 classes in one block, the L0 tile staged in shared memory and
//             stored as the 16-output group's quad (24 outputs padded to 2 groups)
// k_giOut     merged at the half: the 4 blocks' flows and masks summed at the pixel, sigmoid(mask), the two border warps
//             of the fp32 frames, w0 m + w1 (1 - m): 3 fp32 planes

__device__ __forceinline__ float giLr(float v)
{
    return v < 0.0f ? v * 0.2f : v;
}

// one axis of PyTorch's bilinear resize (align_corners False) from n source pixels to n x s at pixel x, s a power of 2
// (the source point (2 x + 1 - s) / 2 s is exact): the two source pixels and the second one's weight (the last pixel
// pairs with itself and keeps its weight, as PyTorch's h1p does)
__device__ __forceinline__ void giUpAxis(int x, int s, int n, int& i0, int& i1, float& l)
{
    const int num = 2 * x + 1 - s, den = 2 * s;
    i0 = 0;
    l = 0.0f;
    if (num > 0)
    {
        i0 = num / den;
        l = (float)(num - i0 * den) / (float)den;
    }
    i1 = i0 < n - 1 ? i0 + 1 : i0;
}

struct GiUp
{
    int o00, o01, o10, o11;
    float lx, ly;
};

// full-size pixel (x, y) of a level s times smaller (w x h)
__device__ __forceinline__ GiUp giUp(int x, int y, int s, int w, int h)
{
    int x0, x1, y0, y1;
    GiUp u;
    giUpAxis(x, s, w, x0, x1, u.lx);
    giUpAxis(y, s, h, y0, y1, u.ly);
    u.o00 = y0 * w + x0;
    u.o01 = y0 * w + x1;
    u.o10 = y1 * w + x0;
    u.o11 = y1 * w + x1;
    return u;
}

// PyTorch's order on each channel of a quad: h0 (w0 v00 + w1 v01) + h1 (w0 v10 + w1 v11)
__device__ __forceinline__ float4 giUpAt4(const float4* __restrict__ p, const GiUp& u)
{
    const float wx0 = 1.0f - u.lx, wy0 = 1.0f - u.ly;
    const float4 a = p[u.o00], b = p[u.o01], c = p[u.o10], d = p[u.o11];
    return make_float4(wy0 * (wx0 * a.x + u.lx * b.x) + u.ly * (wx0 * c.x + u.lx * d.x),
                       wy0 * (wx0 * a.y + u.lx * b.y) + u.ly * (wx0 * c.y + u.lx * d.y),
                       wy0 * (wx0 * a.z + u.lx * b.z) + u.ly * (wx0 * c.z + u.lx * d.z),
                       wy0 * (wx0 * a.w + u.lx * b.w) + u.ly * (wx0 * c.w + u.lx * d.w));
}

// the flow and the mask at full-size pixel (x, y) after blocks 0 .. NL - 1: tmp_j's flow quad by bilinear x 8 >> j and
// its mask plane by bilinear, each summed in block order (block 0's = its own, each later one added)
template <int NL>
__device__ __forceinline__ void giFlowMaskAt(const float* __restrict__ t0, const float* __restrict__ t1,
                                             const float* __restrict__ t2, const float* __restrict__ t3, int x, int y,
                                             int W, int H, float f[4], float& m)
{
#pragma unroll
    for (int j = 0; j < NL; ++j)
    {
        const float* t = j == 0 ? t0 : j == 1 ? t1 : j == 2 ? t2 : t3;
        const int s = 8 >> j, w = W / s, h = H / s;
        const GiUp u = giUp(x, y, s, w, h);
        const float4* q = reinterpret_cast<const float4*>(t);
        const float4 qf = giUpAt4(q, u), qm = giUpAt4(q + w * h, u);
        const float v4[4] = {qf.x, qf.y, qf.z, qf.w};
#pragma unroll
        for (int c = 0; c < 4; ++c)
        {
            const float v = v4[c] * (float)s;
            f[c] = j == 0 ? v : f[c] + v;
        }
        m = j == 0 ? qm.x : m + qm.x;
    }
}

// one axis of a warp tap at pixel x plus the offset u px, the point clamped into [0, n - 1] (grid_sample's border)
__device__ __forceinline__ void giAxis(int x, float u, int n, int& x0, int& x1, float& l)
{
    u = fminf(fmaxf(u, -(float)(n + 1)), (float)(n + 1));
    const float fl = floorf(u);
    x0 = x + (int)fl;
    l = u - fl;
    if (x0 < 0)
    {
        x0 = 0;
        l = 0.0f;
    }
    else if (x0 >= n - 1)
    {
        x0 = n - 1;
        l = 0.0f;
    }
    x1 = x0 < n - 1 ? x0 + 1 : x0;
}

__device__ __forceinline__ GiUp giTap(int x, int y, float ux, float uy, int W, int H)
{
    int x0, x1, y0, y1;
    GiUp t;
    giAxis(x, ux, W, x0, x1, t.lx);
    giAxis(y, uy, H, y0, y1, t.ly);
    t.o00 = y0 * W + x0;
    t.o01 = y0 * W + x1;
    t.o10 = y1 * W + x0;
    t.o11 = y1 * W + x1;
    return t;
}

__device__ __forceinline__ float giTapF(const float* __restrict__ p, const GiUp& t)
{
    const float wx0 = 1.0f - t.lx, wy0 = 1.0f - t.ly;
    return wy0 * (wx0 * p[t.o00] + t.lx * p[t.o01]) + t.ly * (wx0 * p[t.o10] + t.lx * p[t.o11]);
}

__device__ __forceinline__ void giStore16(unsigned short* __restrict__ out, const float* v, int n)
{
    uint4* d = reinterpret_cast<uint4*>(out);
#pragma unroll
    for (int q = 0; q < 2; ++q)
    {
        unsigned u[4];
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            const int c = q * 8 + 2 * k;
            u[k] = (c < n ? (unsigned)f2h(v[c]) : 0u) | ((c + 1 < n ? (unsigned)f2h(v[c + 1]) : 0u) << 16);
        }
        d[q] = make_uint4(u[0], u[1], u[2], u[3]);
    }
}

// block 0's input: x = 6 fp32 planes (img0, img1) at W x H, ts = the timestep (one fp32); out = [W / 8 x H / 8][16]
extern "C" __global__ void __launch_bounds__(256) k_giIn0(const float* __restrict__ x, const float* __restrict__ ts,
                                                           int W, int H, unsigned short* __restrict__ out)
{
    const int w8 = W >> 3, h8 = H >> 3;
    const int X = blockIdx.x * 32 + (threadIdx.x & 31), Y = blockIdx.y * 8 + (threadIdx.x >> 5);
    if (X >= w8 || Y >= h8)
        return;
    const int plane = W * H, s = (8 * Y + 3) * W + 8 * X + 3;
    float v[7];
#pragma unroll
    for (int c = 0; c < 6; ++c)
    {
        const float* src = x + c * plane;
        v[c] = 0.5f * (0.5f * src[s] + 0.5f * src[s + 1]) + 0.5f * (0.5f * src[s + W] + 0.5f * src[s + W + 1]);
    }
    v[6] = ts[0];
    giStore16(out + (size_t)(Y * w8 + X) * 16, v, 7);
}

// the 12 input channels of block K at full-size pixel (px, py)
template <int K>
__device__ __forceinline__ void giInTap(const float* __restrict__ x, float tsv, int W, int H,
                                        const float* __restrict__ t0, const float* __restrict__ t1,
                                        const float* __restrict__ t2, int px, int py, float v[12])
{
    const int plane = W * H;
    float f[4], m;
    giFlowMaskAt<K>(t0, t1, t2, nullptr, px, py, W, H, f, m);
    const GiUp T0 = giTap(px, py, f[0], f[1], W, H), T1 = giTap(px, py, f[2], f[3], W, H);
#pragma unroll
    for (int c = 0; c < 3; ++c)
    {
        v[c] = giTapF(x + c * plane, T0);
        v[3 + c] = giTapF(x + (3 + c) * plane, T1);
    }
    v[6] = tsv;
    v[7] = m;
#pragma unroll
    for (int c = 0; c < 4; ++c)
        v[8 + c] = f[c];
}

// block K's 12 input channels at L0 pixel (X, Y): the full-size taps the author's resize reads, averaged; the flow / s
template <int K>
__device__ __forceinline__ void giInAt(const float* __restrict__ x, float tsv, int W, int H,
                                       const float* __restrict__ t0, const float* __restrict__ t1,
                                       const float* __restrict__ t2, int X, int Y, float r[12])
{
    constexpr int S = 8 >> K;
    if (S == 1)
        giInTap<K>(x, tsv, W, H, t0, t1, t2, X, Y, r);
    else
    {
        constexpr int A = S / 2 - 1;
        float v0[12], v1[12], r0[12];
        giInTap<K>(x, tsv, W, H, t0, t1, t2, S * X + A, S * Y + A, v0);
        giInTap<K>(x, tsv, W, H, t0, t1, t2, S * X + A + 1, S * Y + A, v1);
#pragma unroll
        for (int c = 0; c < 12; ++c)
            r0[c] = 0.5f * v0[c] + 0.5f * v1[c];
        giInTap<K>(x, tsv, W, H, t0, t1, t2, S * X + A, S * Y + A + 1, v0);
        giInTap<K>(x, tsv, W, H, t0, t1, t2, S * X + A + 1, S * Y + A + 1, v1);
#pragma unroll
        for (int c = 0; c < 12; ++c)
            r[c] = 0.5f * r0[c] + 0.5f * (0.5f * v0[c] + 0.5f * v1[c]);
    }
#pragma unroll
    for (int c = 8; c < 12; ++c)
        r[c] = r[c] * (1.0f / (float)S);
}

template <int K>
__device__ __forceinline__ void giInBody(const float* __restrict__ x, const float* __restrict__ ts, int W, int H,
                                         const float* __restrict__ t0, const float* __restrict__ t1,
                                         const float* __restrict__ t2, unsigned short* __restrict__ out)
{
    constexpr int S = 8 >> K;
    const int wl = W / S, hl = H / S;
    const int X = blockIdx.x * 32 + (threadIdx.x & 31), Y = blockIdx.y * 8 + (threadIdx.x >> 5);
    if (X >= wl || Y >= hl)
        return;
    float r[12];
    giInAt<K>(x, ts[0], W, H, t0, t1, t2, X, Y, r);
    giStore16(out + (size_t)(Y * wl + X) * 16, r, 12);
}

#define GI_IN(K)                                                                                                       \
    extern "C" __global__ void __launch_bounds__(256)                                                                  \
        k_giIn##K(const float* __restrict__ x, const float* __restrict__ ts, int W, int H,                             \
                  const float* __restrict__ t0, const float* __restrict__ t1, const float* __restrict__ t2,            \
                  unsigned short* __restrict__ out)                                                                    \
    {                                                                                                                  \
        giInBody<K>(x, ts, W, H, t0, t1, t2, out);                                                                     \
    }
GI_IN(1)
GI_IN(2)
GI_IN(3)

)GISRC";
static const char kGiB[] = R"GISRC(
// the stored index of halo pixel sp and its chunk k's swizzled slot (each ldmatrix's 8 rows in 8 bank groups)
template <int NCH>
__device__ __forceinline__ unsigned giSlot(int sp, int k)
{
    const int sw = NCH == 8 ? (k ^ (sp & 7)) : NCH == 4 ? (k ^ ((sp >> 1) & 3)) : (k ^ ((sp >> 2) & 1));
    return (unsigned)((sp * NCH + sw) << 4);
}

__device__ __forceinline__ uint4 giLds(unsigned addr)
{
    uint4 v;
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr));
    return v;
}

// one conv, 128 threads: x = cin channels-last fp16 at W x H; y = cout channels-last fp16 at W / S x H / S; wf[z][step]
// [lane] (step = (group * 9 + tap) * CG / 16 + kc, the B registers of the z's two n-tiles); bias, beta fp32. The 4 x 32
// output tile's CG-channel halo (a stride-2 halo row keeps its even offsets before its odd ones) and the group's weights
// by cp.async into shared memory, one wait a group. RES: y = LeakyRelu(conv x beta + x), x at the output pixel = the
// residual; else y = LeakyRelu(conv)
template <int S, int CG, bool RES>
__device__ __forceinline__ void giConvT(const unsigned short* __restrict__ x, int cin, int H, int W,
                                        unsigned short* __restrict__ y, int cout, const uint4* __restrict__ wf,
                                        const float* __restrict__ bias, const float* __restrict__ beta)
{
    constexpr int TH = 4, NB = 2;
    constexpr int HR = S == 1 ? TH + 2 : 2 * TH + 1, HC = S == 1 ? 34 : 65;
    constexpr int NCH = CG / 8, KC = CG / 16;
    constexpr int HALO = HR * HC * NCH * 16, GQ = 9 * KC * 32;
    extern __shared__ __align__(128) unsigned char giSm[];
    const unsigned halo = (unsigned)__cvta_generic_to_shared(giSm), wsm = halo + HALO;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int tx0 = blockIdx.x * 32, ty0 = blockIdx.y * TH, z = blockIdx.z;
    const int OW = W / S, OH = H / S;
    wf += (size_t)z * (cin * 9 / 16) * 32;
    const int g = lane >> 2, t4 = lane & 3;
    float acc[2][NB][4];
#pragma unroll
    for (int nj = 0; nj < NB; ++nj)
    {
        const float b0 = bias[z * NB * 8 + nj * 8 + t4 * 2], b1 = bias[z * NB * 8 + nj * 8 + t4 * 2 + 1];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
        {
            acc[mi][nj][0] = acc[mi][nj][2] = b0;
            acc[mi][nj][1] = acc[mi][nj][3] = b1;
        }
    }
    const int arow = (lane & 7) + ((lane >> 3) & 1) * 8, akc = lane >> 4;
    const int groups = cin / CG;
#pragma unroll 1
    for (int gi = 0; gi < groups; ++gi)
    {
        if (gi)
            __syncthreads();
        for (int i = threadIdx.x; i < HR * HC * NCH; i += 128)
        {
            const int p = i / NCH, k = i - p * NCH;
            const int hr = p / HC, hc = p - hr * HC;
            const int gy = S * ty0 - 1 + hr, gx = S * tx0 - 1 + hc;
            const int sp = S == 1 ? p : hr * HC + ((hc & 1) ? 33 + (hc >> 1) : (hc >> 1));
            const bool in = gy >= 0 && gy < H && gx >= 0 && gx < W;
            const unsigned short* src = in ? x + ((size_t)gy * W + gx) * cin + gi * CG + k * 8 : x;
            rnCp16(halo + giSlot<NCH>(sp, k), src, in ? 16 : 0);
        }
        for (int i = threadIdx.x; i < GQ; i += 128)
            rnCp16(wsm + (unsigned)i * 16u, wf + (size_t)gi * GQ + i, 16);
        asm volatile(SMV_CP_COMMIT);
        asm volatile(SMV_CP_WAIT0);
        __syncthreads();
#pragma unroll 1
        for (int tap = 0; tap < 9; ++tap)
        {
            const int dy = tap / 3, dx = tap - dy * 3;
            int pa[2];
#pragma unroll
            for (int mi = 0; mi < 2; ++mi)
            {
                const int col = mi * 16 + arow;
                pa[mi] = S == 1 ? (warp + dy) * HC + col + dx : (2 * warp + dy) * HC + (dx == 1 ? 33 + col : col + (dx >> 1));
            }
#pragma unroll
            for (int kc = 0; kc < KC; ++kc)
            {
                const uint4 bw = giLds(wsm + (unsigned)((tap * KC + kc) * 32 + lane) * 16u);
#pragma unroll
                for (int mi = 0; mi < 2; ++mi)
                {
                    unsigned a[4];
                    rnLdm4(halo + giSlot<NCH>(pa[mi], kc * 2 + akc), a);
#pragma unroll
                    for (int q = 0; q < NB; ++q)
                    {
                        float* d = acc[mi][q];
                        const unsigned b0 = q ? bw.z : bw.x, b1 = q ? bw.w : bw.y;
                        asm volatile(SMV_MMA_F32 "{%0,%1,%2,%3}, "
                                     "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
                    }
                }
            }
        }
    }
#pragma unroll
    for (int nj = 0; nj < NB; ++nj)
    {
        const int n0 = z * NB * 8 + nj * 8 + t4 * 2;
        const float be0 = RES ? beta[n0] : 1.0f, be1 = RES ? beta[n0 + 1] : 1.0f;
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
        {
            const int oy = ty0 + warp;
            if (oy >= OH)
                continue;
#pragma unroll
            for (int h = 0; h < 2; ++h)
            {
                const int ox = tx0 + mi * 16 + g + h * 8;
                if (ox >= OW)
                    continue;
                float v0 = acc[mi][nj][2 * h], v1 = acc[mi][nj][2 * h + 1];
                const size_t pix = (size_t)oy * OW + ox;
                if (RES)
                {
                    const unsigned u = *reinterpret_cast<const unsigned*>(x + pix * cin + n0);
                    v0 = __fadd_rn(__fmul_rn(v0, be0), h2f((unsigned short)(u & 0xffffu)));
                    v1 = __fadd_rn(__fmul_rn(v1, be1), h2f((unsigned short)(u >> 16)));
                }
                v0 = giLr(v0);
                v1 = giLr(v1);
                *reinterpret_cast<unsigned*>(y + pix * cout + n0) = (unsigned)f2h(v0) | ((unsigned)f2h(v1) << 16);
            }
        }
    }
}

#define GI_CONV(NAME, S, CG, RES)                                                                                      \
    extern "C" __global__ void __launch_bounds__(128, 2)                                                               \
        NAME(const unsigned short* __restrict__ x, int cin, int H, int W, unsigned short* __restrict__ y, int cout,    \
             const uint4* __restrict__ wf, const float* __restrict__ bias, const float* __restrict__ beta)             \
    {                                                                                                                  \
        giConvT<S, CG, RES>(x, cin, H, W, y, cout, wf, bias, beta);                                                    \
    }
GI_CONV(k_giS2g16, 2, 16, false)
GI_CONV(k_giS2g32, 2, 32, false)
GI_CONV(k_giResg32, 1, 32, true)
GI_CONV(k_giResg64, 1, 64, true)

// ConvTranspose c -> 24 + PixelShuffle(2) on the tensor cores, the 4 output parity classes of a 4 x 32 L2 tile in one
// block: class (py, px) = the L1 pixel (2 yy + py, 2 xx + px) reads rows yy - 1 + py + ti, columns xx - 1 + px + tj
// (kernel taps ky = 3 - 2 (py + ti) + py, kx the same way), so each of the 9 halo positions' A fragments serves the
// classes that read it; fp32 accumulators from the bias; the input = the 8th ResConv's fp16 output. blockIdx.z = the
// 16-output group zc (24 outputs padded to 32, the pad's weights zero) = 4 shuffled planes = quad zc of the tmp; the
// group's L0 tile (16 x 128) is staged in shared memory, then stored as quads [zc][h0][w0] (h0 = 4 H, w0 = 4 W).
// wf[zc][group][class][tap][kc][lane]
template <int CG>
__device__ __forceinline__ void giLastQ(const unsigned short* __restrict__ x, int cin, int H, int W,
                                        float* __restrict__ out, const uint4* __restrict__ wf,
                                        const float* __restrict__ bias)
{
    constexpr int TH = 4, NB = 2, HR = TH + 2, HC = 34;
    constexpr int NCH = CG / 8, KC = CG / 16;
    constexpr int HALO = HR * HC * NCH * 16, GQ = 16 * KC * 32;
    extern __shared__ __align__(128) unsigned char giSm[];
    const unsigned halo = (unsigned)__cvta_generic_to_shared(giSm), wsm = halo + HALO;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int tx0 = blockIdx.x * 32, ty0 = blockIdx.y * TH, zc = blockIdx.z;
    wf += (size_t)zc * (cin / CG) * GQ;
    const int g = lane >> 2, t4 = lane & 3;
    float acc[4][2][NB][4];
#pragma unroll
    for (int nj = 0; nj < NB; ++nj)
    {
        const float b0 = bias[zc * 16 + nj * 8 + t4 * 2], b1 = bias[zc * 16 + nj * 8 + t4 * 2 + 1];
#pragma unroll
        for (int cl = 0; cl < 4; ++cl)
#pragma unroll
            for (int mi = 0; mi < 2; ++mi)
            {
                acc[cl][mi][nj][0] = acc[cl][mi][nj][2] = b0;
                acc[cl][mi][nj][1] = acc[cl][mi][nj][3] = b1;
            }
    }
    const int arow = (lane & 7) + ((lane >> 3) & 1) * 8, akc = lane >> 4;
    const int groups = cin / CG;
#pragma unroll 1
    for (int gi = 0; gi < groups; ++gi)
    {
        if (gi)
            __syncthreads();
        for (int i = threadIdx.x; i < HR * HC * NCH; i += 128)
        {
            const int p = i / NCH, k = i - p * NCH;
            const int hr = p / HC, hc = p - hr * HC;
            const int gy = ty0 - 1 + hr, gx = tx0 - 1 + hc;
            const bool in = gy >= 0 && gy < H && gx >= 0 && gx < W;
            const unsigned short* src = in ? x + ((size_t)gy * W + gx) * cin + gi * CG + k * 8 : x;
            rnCp16(halo + giSlot<NCH>(p, k), src, in ? 16 : 0);
        }
        for (int i = threadIdx.x; i < GQ; i += 128)
            rnCp16(wsm + (unsigned)i * 16u, wf + (size_t)gi * GQ + i, 16);
        asm volatile(SMV_CP_COMMIT);
        asm volatile(SMV_CP_WAIT0);
        __syncthreads();
#pragma unroll
        for (int pos = 0; pos < 9; ++pos)
        {
            const int dy = pos / 3, dx = pos % 3;
#pragma unroll
            for (int kc = 0; kc < KC; ++kc)
            {
                unsigned a[2][4];
#pragma unroll
                for (int mi = 0; mi < 2; ++mi)
                    rnLdm4(halo + giSlot<NCH>((warp + dy) * HC + mi * 16 + arow + dx, kc * 2 + akc), a[mi]);
#pragma unroll
                for (int cl = 0; cl < 4; ++cl)
                {
                    const int ti = dy - (cl >> 1), tj = dx - (cl & 1);
                    if (ti < 0 || ti > 1 || tj < 0 || tj > 1)
                        continue;
                    const uint4 bw = giLds(wsm + (unsigned)(((cl * 4 + ti * 2 + tj) * KC + kc) * 32 + lane) * 16u);
#pragma unroll
                    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
                        for (int q = 0; q < NB; ++q)
                        {
                            float* d = acc[cl][mi][q];
                            const unsigned b0 = q ? bw.z : bw.x, b1 = q ? bw.w : bw.y;
                            asm volatile(SMV_MMA_F32 "{%0,%1,%2,%3}, "
                                         "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                                         : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                                         : "r"(a[mi][0]), "r"(a[mi][1]), "r"(a[mi][2]), "r"(a[mi][3]), "r"(b0), "r"(b1));
                        }
                }
            }
        }
    }
    // the group's 4 planes of the L0 tile (16 rows x 128 columns) in shared memory (over the halo and weights)
    __syncthreads();
    float* st = reinterpret_cast<float*>(giSm);
#pragma unroll
    for (int cl = 0; cl < 4; ++cl)
#pragma unroll
        for (int nj = 0; nj < NB; ++nj)
#pragma unroll
            for (int e = 0; e < 2; ++e)
            {
                const int lo = nj * 8 + t4 * 2 + e, lp = lo >> 2, j = lo & 3;
                const int r = 2 * (2 * warp + (cl >> 1)) + (j >> 1);
#pragma unroll
                for (int mi = 0; mi < 2; ++mi)
#pragma unroll
                    for (int h = 0; h < 2; ++h)
                    {
                        const int cx = 2 * (2 * (mi * 16 + g + h * 8) + (cl & 1)) + (j & 1);
                        st[(lp * 16 + r) * 128 + cx] = acc[cl][mi][nj][2 * h + e];
                    }
            }
    __syncthreads();
    const int w0 = 4 * W, h0 = 4 * H, r0 = 4 * ty0, c0 = 4 * tx0;
    for (int i = threadIdx.x; i < 16 * 128; i += 128)
    {
        const int r = i >> 7, cx = i & 127, Y = r0 + r, X = c0 + cx;
        if (Y >= h0 || X >= w0)
            continue;
        reinterpret_cast<float4*>(out)[((size_t)zc * h0 + Y) * w0 + X] =
            make_float4(st[r * 128 + cx], st[(16 + r) * 128 + cx], st[(32 + r) * 128 + cx], st[(48 + r) * 128 + cx]);
    }
}

#define GI_LASTQ(CG)                                                                                                   \
    extern "C" __global__ void __launch_bounds__(128)                                                                  \
        k_giLastQ##CG(const unsigned short* __restrict__ x, int cin, int H, int W, float* __restrict__ out,            \
                      const uint4* __restrict__ wf, const float* __restrict__ bias)                                    \
    {                                                                                                                  \
        giLastQ<CG>(x, cin, H, W, out, wf, bias);                                                                      \
    }
GI_LASTQ(32)
GI_LASTQ(64)

// merged at the half pixel: the 4 blocks' flows and masks summed (block order), sigmoid(mask), the border warps of the
// fp32 frames x (img0 planes 0..2, img1 planes 3..5), w0 m + w1 (1 - m) into 3 fp32 planes
extern "C" __global__ void __launch_bounds__(256) k_giOut(const float* __restrict__ x, int W, int H,
                                                           const float* __restrict__ t0, const float* __restrict__ t1,
                                                           const float* __restrict__ t2, const float* __restrict__ t3,
                                                           float* __restrict__ merged)
{
    const int X = blockIdx.x * 32 + (threadIdx.x & 31), Y = blockIdx.y * 8 + (threadIdx.x >> 5);
    if (X >= W || Y >= H)
        return;
    const int plane = W * H, o = Y * W + X;
    float f[4], m;
    giFlowMaskAt<4>(t0, t1, t2, t3, X, Y, W, H, f, m);
    const float sg = 1.0f / (1.0f + expf(-m));
    const GiUp T0 = giTap(X, Y, f[0], f[1], W, H), T1 = giTap(X, Y, f[2], f[3], W, H);
#pragma unroll
    for (int c = 0; c < 3; ++c)
        merged[c * plane + o] = giTapF(x + c * plane, T0) * sg + giTapF(x + (3 + c) * plane, T1) * (1.0f - sg);
}
)GISRC";
static const char* const kGiKern[] = {"k_giIn0",     "k_giIn1",     "k_giIn2",    "k_giIn3",
                                      "k_giS2g16",   "k_giS2g32",   "k_giResg32", "k_giResg64",
                                      "k_giLastQ32", "k_giLastQ64", "k_giOut"};
static_assert(sizeof(kGiKern) / sizeof(kGiKern[0]) == sizeof(NativeRife::fGi) / sizeof(CUfunction),
              "one fGi slot a GMFSS IFNet kernel");
enum
{
    kGiIn0,
    kGiS2g16 = 4,
    kGiS2g32,
    kGiResg32,
    kGiResg64,
    kGiLq32,
    kGiLq64,
    kGiOut
};
static const int kGiC[4] = {192, 128, 96, 64}, kGiCin[4] = {7, 12, 12, 12};
static const int kGiPieces = 30;
// a conv's dynamic shared memory: its 4 x 32 tile's halo of a cg-channel group + the group's weights (stride s); the
// transposed conv's: the halo + its weights, at least the staged 16 x 128 x 4 fp32 tile
static constexpr int giSmConv(int s, int cg)
{
    return ((s == 1 ? 6 * 34 : 9 * 65) * (cg / 8) + 9 * (cg / 16) * 32) * 16;
}
static constexpr int giSmLq(int cg)
{
    return (6 * 34 * (cg / 8) + 16 * (cg / 16) * 32) * 16 > 16 * 128 * 4 * 4
               ? (6 * 34 * (cg / 8) + 16 * (cg / 16) * 32) * 16
               : 16 * 128 * 4 * 4;
}
static int giCg1(int c)
{
    return (c / 2) % 32 == 0 ? 32 : 16;
}
static int giCgRes(int c)
{
    return c % 64 == 0 ? 64 : 32;
}

// rife.pkl in the strict GMFSS IFNet form (block<k>.: conv0.0.0 [c / 2][cin][3][3], conv0.1.0 [c][c / 2][3][3],
// convblock.0 .. 7 {conv [c][c][3][3], beta [1][c][1][1]}, lastconv.0 [c][24][4][4], their biases; c = kGiC, cin =
// kGiCin; no block4) as one weight block of 256-byte aligned pieces, off[kGiPieces k + i]: 0 / 1 conv0[0]'s fragments
// (its inputs zero-padded to 16) and bias, 2 / 3 conv0[1]'s, 4 + 3 r .. 6 + 3 r ResConv r's fragments, bias and beta,
// 28 / 29 the transposed conv's fragments and its bias padded to 32; false = missing, or not this net (`why`)
static bool giNatLoad(std::vector<uint8_t>& blob, std::vector<size_t>& off, std::string& why)
{
    PthFile pf;
    if (!pthStateDict(g_nativeEngineDir + L"\\GMFSS_Fortuna\\train_log\\rife.pkl", pf, why))
        return false;
    auto& wts = pf.t;
    // the biases and the betas go to the kernels as fp32
    bool form = !pf.S.count("block4.conv0.0.0.weight");
    for (int k = 0; k < 4 && form; k++)
    {
        const std::string p = "block" + std::to_string(k) + ".";
        const int64_t c = kGiC[k], h = c / 2, ci = kGiCin[k];
        form = pf.is(p + "conv0.0.0.weight", {h, ci, 3, 3}) && pf.is32(p + "conv0.0.0.bias", {h}) &&
               pf.is(p + "conv0.1.0.weight", {c, h, 3, 3}) && pf.is32(p + "conv0.1.0.bias", {c}) &&
               pf.is(p + "lastconv.0.weight", {c, 24, 4, 4}) && pf.is32(p + "lastconv.0.bias", {24}) &&
               !pf.S.count(p + "convblock.8.beta");
        for (int r = 0; r < 8 && form; r++)
        {
            const std::string q = p + "convblock." + std::to_string(r) + ".";
            form = pf.is(q + "conv.weight", {c, c, 3, 3}) && pf.is32(q + "conv.bias", {c}) &&
                   pf.is32(q + "beta", {1, c, 1, 1});
        }
    }
    if (!form)
    {
        why = pf.notForm("GMFSS's IFNet");
        return false;
    }
    blob.clear();
    off.assign((size_t)4 * kGiPieces, 0);
    auto put = [&](const void* d, size_t n) {
        const size_t o = (blob.size() + 255) & ~(size_t)255;
        blob.resize(o + n);
        memcpy(blob.data() + o, d, n);
        return o;
    };
    for (int k = 0; k < 4; k++)
    {
        const std::string p = "block" + std::to_string(k) + ".";
        const int c = kGiC[k], h = c / 2, ci = kGiCin[k];
        size_t* o = off.data() + (size_t)k * kGiPieces;
        const PthW& w0 = wts[p + "conv0.0.0.weight"];
        std::vector<uint16_t> w0p((size_t)h * 16 * 9, 0);
        for (int n = 0; n < h; n++)
            for (int i = 0; i < ci; i++)
                for (int t = 0; t < 9; t++)
                    w0p[((size_t)n * 16 + i) * 9 + t] = w0.half(((size_t)n * ci + i) * 9 + t);
        const std::vector<uint16_t> f0 = fnFragOf(PthW(w0p), h, 16, 16, 2);
        o[0] = put(f0.data(), f0.size() * sizeof(uint16_t));
        o[1] = put(pf.f32(p + "conv0.0.0.bias").data(), (size_t)h * sizeof(float));
        const std::vector<uint16_t> f1 = fnFragOf(wts[p + "conv0.1.0.weight"], c, h, giCg1(c), 2);
        o[2] = put(f1.data(), f1.size() * sizeof(uint16_t));
        o[3] = put(pf.f32(p + "conv0.1.0.bias").data(), (size_t)c * sizeof(float));
        for (int r = 0; r < 8; r++)
        {
            const std::string q = p + "convblock." + std::to_string(r) + ".";
            const std::vector<uint16_t> fr = fnFragOf(wts[q + "conv.weight"], c, c, giCgRes(c), 2);
            o[4 + 3 * r] = put(fr.data(), fr.size() * sizeof(uint16_t));
            o[5 + 3 * r] = put(pf.f32(q + "conv.bias").data(), (size_t)c * sizeof(float));
            o[6 + 3 * r] = put(pf.f32(q + "beta").data(), (size_t)c * sizeof(float));
        }
        const std::vector<uint16_t> fl = ifnLastFrags(wts[p + "lastconv.0.weight"], c, 6, giCgRes(c), 24);
        o[28] = put(fl.data(), fl.size() * sizeof(uint16_t));
        const std::span<const float> lb = pf.f32(p + "lastconv.0.bias");
        std::vector<float> lbp(32, 0.0f);
        for (int i = 0; i < 24; i++)
            lbp[i] = lb[i];
        o[29] = put(lbp.data(), lbp.size() * sizeof(float));
    }
    return true;
}

// the GMFSS IFNet's scratch at the half w x h, 256-byte pieces: ONE input buffer ([L0][16] fp16, the largest), ONE
// conv0[0] output ([L1][c / 2] fp16, the largest), ONE ping-pong pair ([L2][c] fp16, the largest), the 4 tmps (2 fp32
// quads a pixel at each block's L0)
struct GiArena
{
    size_t in = 0, y0 = 0, ya = 0, yb = 0, tm[4] = {}, total = 0;
};
static GiArena giArena(int w, int h)
{
    GiArena a;
    size_t o = 0, inB = 0, y0B = 0, yB = 0;
    auto put = [&](size_t n) {
        const size_t at = o;
        o += (n + 255) & ~(size_t)255;
        return at;
    };
    for (int k = 0; k < 4; k++)
    {
        const int s = 8 >> k;
        const size_t n0 = (size_t)(w / s) * (size_t)(h / s);
        a.tm[k] = put(8 * n0 * sizeof(float));
        inB = (std::max)(inB, 16 * n0 * sizeof(uint16_t));
        y0B = (std::max)(y0B, (size_t)(kGiC[k] / 2) * (n0 / 4) * sizeof(uint16_t));
        yB = (std::max)(yB, (size_t)kGiC[k] * (n0 / 16) * sizeof(uint16_t));
    }
    a.in = put(inB);
    a.y0 = put(y0B);
    a.ya = put(yB);
    a.yb = put(yB);
    a.total = o;
    return a;
}

static size_t giNatBytes(const NativeRife& nr)
{
    return nr.gmfss && nr.hw > 0 && nr.hh > 0 ? giArena(nr.hw, nr.hh).total : 0;
}

// this session's GMFSS IFNet module: the weights read, its text compiled (or the cached cubin) and loaded, its kernels
// bound, the weight block uploaded; false = `why` (the session teardown unloads a module left in giMod)
static bool nativeGiModule(NativeRife& nr, std::string& why)
{
    if (nr.giMod)
        return true;
    std::vector<uint8_t> blob;
    if (!giNatLoad(blob, nr.giOff, why))
        return false;
    std::string text = kFnHead;
    text += kGiA;
    text += kGiB;
    std::vector<char> cubin;
    if (!nativeCubin(text.c_str(), "smv_gmifnet.cu", L"smv_gmifnet", "GMFSS IFNet kernels",
                     nr.kernCacheDir.empty() ? std::wstring(L".") : nr.kernCacheDir, cubin))
    {
        why = "its kernels did not compile";
        return false;
    }
    if (cuModuleLoadData(&nr.giMod, cubin.data()) != CUDA_SUCCESS)
    {
        nr.giMod = nullptr;
        why = "its kernel module did not load";
        return false;
    }
    for (int i = 0; i < (int)(sizeof(kGiKern) / sizeof(kGiKern[0])); i++)
        if (cuModuleGetFunction(&nr.fGi[i], nr.giMod, kGiKern[i]) != CUDA_SUCCESS)
        {
            why = std::string("its kernel ") + kGiKern[i] + " is missing";
            return false;
        }
    // the one kernel above the default 48 KB of dynamic shared memory: the 64-channel transposed conv
    if (cuFuncSetAttribute(nr.fGi[kGiLq64], CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, giSmLq(64)) !=
        CUDA_SUCCESS)
    {
        why = "its transposed conv's shared memory was refused";
        return false;
    }
    if (cudaMalloc(&nr.dGiW, blob.size()) != cudaSuccess ||
        cudaMemcpy(nr.dGiW, blob.data(), blob.size(), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        why = "its weights did not upload";
        return false;
    }
    return true;
}

// the GMFSS IFNet's 49 launches on st at the half nr.hw x nr.hh, its scratch at s (giArena's pieces). false = a launch
// failed
static bool nativeGiLaunch(NativeRife& nr, const float* x, const float* ts, float* merged, uint8_t* s, cudaStream_t st)
{
    int iw = nr.hw, ih = nr.hh;
    const GiArena ar = giArena(iw, ih);
    void* tm[4] = {s + ar.tm[0], s + ar.tm[1], s + ar.tm[2], s + ar.tm[3]};
    void* inb = s + ar.in;
    void* y0 = s + ar.y0;
    void* yab[2] = {s + ar.ya, s + ar.yb};
    const char* wb = (const char*)nr.dGiW;
    auto go = [&](int f, int gx, int gy, int gz, int threads, int sm, void** a) {
        return cuLaunchKernel(nr.fGi[f], (unsigned)gx, (unsigned)gy, (unsigned)gz, (unsigned)threads, 1, 1,
                              (unsigned)sm, (CUstream)st, a, nullptr) == CUDA_SUCCESS;
    };
    for (int k = 0; k < 4; k++)
    {
        const int sc = 8 >> k, c = kGiC[k], hc = c / 2, cg1 = giCg1(c), cgr = giCgRes(c);
        int w0 = iw / sc, h0 = ih / sc, w1 = w0 / 2, h1 = h0 / 2, w2 = w0 / 4, h2 = h0 / 4;
        const size_t* o = nr.giOff.data() + (size_t)k * kGiPieces;
        const void* c0w = wb + o[0];
        const void* c0b = wb + o[1];
        const void* c1w = wb + o[2];
        const void* c1b = wb + o[3];
        const void* lw = wb + o[28];
        const void* lb = wb + o[29];
        if (k == 0)
        {
            void* a[] = {(void*)&x, (void*)&ts, &iw, &ih, &inb};
            if (!go(kGiIn0, (w0 + 31) / 32, (h0 + 7) / 8, 1, 256, 0, a))
                return false;
        }
        else
        {
            void* a[] = {(void*)&x, (void*)&ts, &iw, &ih, &tm[0], &tm[1], &tm[2], &inb};
            if (!go(kGiIn0 + k, (w0 + 31) / 32, (h0 + 7) / 8, 1, 256, 0, a))
                return false;
        }
        {
            int cin = 16, cout = hc;
            void* a[] = {&inb, &cin, &h0, &w0, &y0, &cout, (void*)&c0w, (void*)&c0b, (void*)&c0b};
            if (!go(kGiS2g16, (w1 + 31) / 32, (h1 + 3) / 4, hc / 16, 128, giSmConv(2, 16), a))
                return false;
        }
        {
            int cin = hc, cout = c;
            void* a[] = {&y0, &cin, &h1, &w1, &yab[0], &cout, (void*)&c1w, (void*)&c1b, (void*)&c1b};
            if (!go(cg1 == 32 ? kGiS2g32 : kGiS2g16, (w2 + 31) / 32, (h2 + 3) / 4, c / 16, 128, giSmConv(2, cg1), a))
                return false;
        }
        for (int r = 0; r < 8; r++)
        {
            const void* rw = wb + o[4 + 3 * r];
            const void* rb = wb + o[5 + 3 * r];
            const void* rbe = wb + o[6 + 3 * r];
            int cin = c, cout = c;
            void* a[] = {&yab[r & 1], &cin, &h2, &w2, &yab[(r + 1) & 1], &cout, (void*)&rw, (void*)&rb, (void*)&rbe};
            if (!go(cgr == 64 ? kGiResg64 : kGiResg32, (w2 + 31) / 32, (h2 + 3) / 4, c / 16, 128, giSmConv(1, cgr), a))
                return false;
        }
        {
            int cin = c;
            void* a[] = {&yab[0], &cin, &h2, &w2, &tm[k], (void*)&lw, (void*)&lb};
            if (!go(cgr == 64 ? kGiLq64 : kGiLq32, (w2 + 31) / 32, (h2 + 3) / 4, 2, 128, giSmLq(cgr), a))
                return false;
        }
    }
    void* a[] = {(void*)&x, &iw, &ih, &tm[0], &tm[1], &tm[2], &tm[3], (void*)&merged};
    return go(kGiOut, (iw + 31) / 32, (ih + 7) / 8, 1, 256, 0, a);
}

// the GMFSS IFNet on one tween on st (nr.stream) at the half nr.hw x nr.hh: x = the 6 fp32 half planes, ts = the fp32
// timestep (one device float, read when the kernels run); merged = its output (3 fp32 planes); the launches go through
// nr.giGraphs (chainGraphRun). false = the scratch or a launch failed
static bool nativeGiRun(NativeRife& nr, const float* x, const float* ts, float* merged, cudaStream_t st)
{
    const GiArena ar = giArena(nr.hw, nr.hh);
    uint8_t* s = nr.dGmCtxMem && (size_t)nr.gmCtxMemB >= ar.total ? (uint8_t*)nr.dGmCtxMem : (uint8_t*)nr.dGiOwn;
    if (!s && cudaMalloc(&nr.dGiOwn, ar.total) == cudaSuccess)
        s = (uint8_t*)nr.dGiOwn;
    if (!s)
        return false;
    return chainGraphRun(nr.giGraphs, {s, x, ts, merged}, nr.hw, nr.hh, st, "gmfss ifnet",
                         [&] { return nativeGiLaunch(nr, x, ts, merged, s, st); });
}
