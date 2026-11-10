// part of smv-live-native.inl (included there, in place); not a standalone translation unit
// ---- RIFE 4.25's IFNet on the host's kernels ----------------------------------------------------------------------
// The whole IFNet (5 IFBlocks + the warps between them) as 59 launches of its own module smv_ifnet_*.cubin (kFnHead's
// helpers + kIfnA / B / C), weights = engine\rife\flownet.pkl's module.block0 .. 4 through pthStateDict (ifnNatLoad);
// no fallback (a missing / wrong pkl = `the IFNet unavailable (...)`, the session refuses). It
// takes the fp16 frame pair, the fp16 timestep plane and the fp16 encodes at the motion frame and hands out the fp16
// flow + mask RIFE's last step (nativeRifeBlend) reads. Its scratch (ifnArena) lives in the shared activation block
// (nativeModelShareMemory sizes it in; sound while every enqueue there runs serially on nr.stream), else dIfnOwn.
static const char kIfnA[] = R"IFNSRC(
// RIFE 4.25's IFNet (IFNet_HDv3: 5 IFBlocks at scales 16, 8, 4, 2, 1 with c = 192, 128, 96, 64, 32) on the host's own
// kernels. The frame is W x H (a multiple of 64). Block k (scale s = 16 >> k) works at L0 = W / s x H / s (its input
// and its output tmp), L1 = L0 / 2, L2 = L0 / 4. Every block's tmp stays at its L0 as 4 float4 quads a pixel
// ([quad][h][w][4]: quad 0 = the flow, quads 1..3 = the mask and the 8 features, channel 12's quad padded); the
// author's full-size tensors (the flow summed over the blocks, the mask, the features, the warps) are read only where a
// reader needs them, by PyTorch's bilinear from those quads. Nothing at the full size is written but the outputs.
// k_ifIn0h     block 0's input cat(img0, img1, f0, f1, timestep) at L0: at an exact 1 / 16 each sample = PyTorch's
//              bilinear mean of the 2 x 2 source pixels at rows / columns 16 i + 7, 16 i + 8; channels-last fp16 [L0][16]
// k_ifIn<k>h   block k's input (k = 1..3) at L0: cat(warp(img0), warp(img1), warp(f0), warp(f1), timestep, mask,
//              features, flow / s) where the author resizes the full-size cat to L0 = the mean of the 4 full-size taps at
//              rows / columns s i + s / 2 - 1, + 1; at each tap the flow = the blocks before summed as the author sums
//              them (block j's flow quad by bilinear x its scale, in block order, fp32), the warps = bilinear taps at the
//              pixel + the flow clamped into the frame (grid_sample's border rule), the mask / features = the previous
//              block's quads by bilinear; channels-last fp16 [L0][32] (28 channels + 4 zeros)
// k_ifInC0_4   block 4's conv0[0] with its input (the pixel itself, s = 1) built straight into its halo: never written
// k_ifS2g16 / k_ifS2g32   a 3 x 3 stride-2 conv + LeakyRelu (conv0[0] of blocks 0..3 on the padded fp16 input, conv0[1]
//              of every block); k_ifResg32 / k_ifResg64 the ResConvs {LeakyRelu(conv(x) * beta + x)}, the residual =
//              the conv's own fp16 input at the pixel: implicit GEMMs on the tensor cores (mma.sync m16n8k16, fp32
//              accumulators from the bias, a 4 x 32 output tile, a CG-channel group's halo and weights by cp.async)
// k_ifLastQ32 / 64   ConvTranspose c -> 52 (4 x 4, stride 2, pad 1) + PixelShuffle(2): per output parity class a 2 x 2-
//              tap conv over L2 on the tensor cores, the 4 classes in one block, the L0 tile staged in shared memory and
//              stored as the 16-output group's quad; k_ifLastQF32 (block 4, 20 outputs) = the IFNet's outputs: the flow
//              + the blocks before and sigmoid(mask), 4 + 1 fp16 planes

__device__ __forceinline__ float ifLr(float v)
{
    return v < 0.0f ? v * 0.2f : v;
}

// one axis of PyTorch's bilinear resize (align_corners False) from n source pixels to n x s at pixel x, s a power of 2
// (the source point (2 x + 1 - s) / 2 s is exact): the two source pixels and the second one's weight (the last pixel
// pairs with itself and keeps its weight, as PyTorch's h1p does)
__device__ __forceinline__ void ifUpAxis(int x, int s, int n, int& i0, int& i1, float& l)
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

struct IfUp
{
    int o00, o01, o10, o11;
    float lx, ly;
};

// full-size pixel (x, y) of a level s times smaller (w x h)
__device__ __forceinline__ IfUp ifUp(int x, int y, int s, int w, int h)
{
    int x0, x1, y0, y1;
    IfUp u;
    ifUpAxis(x, s, w, x0, x1, u.lx);
    ifUpAxis(y, s, h, y0, y1, u.ly);
    u.o00 = y0 * w + x0;
    u.o01 = y0 * w + x1;
    u.o10 = y1 * w + x0;
    u.o11 = y1 * w + x1;
    return u;
}

// PyTorch's order on each channel of a quad: h0 (w0 v00 + w1 v01) + h1 (w0 v10 + w1 v11)
__device__ __forceinline__ float4 ifUpAt4(const float4* __restrict__ p, const IfUp& u)
{
    const float wx0 = 1.0f - u.lx, wy0 = 1.0f - u.ly;
    const float4 a = p[u.o00], b = p[u.o01], c = p[u.o10], d = p[u.o11];
    return make_float4(wy0 * (wx0 * a.x + u.lx * b.x) + u.ly * (wx0 * c.x + u.lx * d.x),
                       wy0 * (wx0 * a.y + u.lx * b.y) + u.ly * (wx0 * c.y + u.lx * d.y),
                       wy0 * (wx0 * a.z + u.lx * b.z) + u.ly * (wx0 * c.z + u.lx * d.z),
                       wy0 * (wx0 * a.w + u.lx * b.w) + u.ly * (wx0 * c.w + u.lx * d.w));
}

// the flow at full-size pixel (x, y) after blocks 0 .. NL - 1: tmp_j's flow quad by bilinear x 16 >> j, summed in block
// order (block 0's flow = its own, each later one added)
template <int NL>
__device__ __forceinline__ void ifFlowAt(const float* __restrict__ t0, const float* __restrict__ t1,
                                         const float* __restrict__ t2, const float* __restrict__ t3, int x, int y,
                                         int W, int H, float f[4])
{
#pragma unroll
    for (int j = 0; j < NL; ++j)
    {
        const float* t = j == 0 ? t0 : j == 1 ? t1 : j == 2 ? t2 : t3;
        const int s = 16 >> j, w = W / s, h = H / s;
        const float4 q = ifUpAt4(reinterpret_cast<const float4*>(t), ifUp(x, y, s, w, h));
        const float v4[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
        for (int c = 0; c < 4; ++c)
        {
            const float v = v4[c] * (float)s;
            f[c] = j == 0 ? v : f[c] + v;
        }
    }
}

// one axis of a warp tap at pixel x plus the offset u px, the point clamped into [0, n - 1] (the host's nvAxis)
__device__ __forceinline__ void ifAxis(int x, float u, int n, int& x0, int& x1, float& l)
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

__device__ __forceinline__ IfUp ifTap(int x, int y, float ux, float uy, int W, int H)
{
    int x0, x1, y0, y1;
    IfUp t;
    ifAxis(x, ux, W, x0, x1, t.lx);
    ifAxis(y, uy, H, y0, y1, t.ly);
    t.o00 = y0 * W + x0;
    t.o01 = y0 * W + x1;
    t.o10 = y1 * W + x0;
    t.o11 = y1 * W + x1;
    return t;
}

__device__ __forceinline__ float ifTapH(const unsigned short* __restrict__ p, const IfUp& t)
{
    const float wx0 = 1.0f - t.lx, wy0 = 1.0f - t.ly;
    return wy0 * (wx0 * h2f(p[t.o00]) + t.lx * h2f(p[t.o01])) + t.ly * (wx0 * h2f(p[t.o10]) + t.lx * h2f(p[t.o11]));
}

// block 0's input: x = 6 fp16 planes (img0, img1), f0 / f1 = 4 fp16 planes, ts = 1 fp16 plane, all W x H; out =
// channels-last fp16 [W / 16 x H / 16][16] (the 15 channels + a zero)
extern "C" __global__ void __launch_bounds__(256) k_ifIn0h(const unsigned short* __restrict__ x,
                                                           const unsigned short* __restrict__ f0,
                                                           const unsigned short* __restrict__ f1,
                                                           const unsigned short* __restrict__ ts, int W, int H,
                                                           unsigned short* __restrict__ out)
{
    const int w16 = W >> 4, h16 = H >> 4;
    const int X = blockIdx.x * 32 + (threadIdx.x & 31), Y = blockIdx.y * 8 + (threadIdx.x >> 5);
    if (X >= w16 || Y >= h16)
        return;
    const int plane = W * H, o = Y * w16 + X;
    const int s = (16 * Y + 7) * W + 16 * X + 7;
    unsigned short v[16];
#pragma unroll
    for (int c = 0; c < 15; ++c)
    {
        const unsigned short* src = c < 6 ? x + c * plane : c < 10 ? f0 + (c - 6) * plane : c < 14 ? f1 + (c - 10) * plane : ts;
        const float a0 = h2f(src[s]), a1 = h2f(src[s + 1]), a2 = h2f(src[s + W]), a3 = h2f(src[s + W + 1]);
        v[c] = f2h(0.5f * (0.5f * a0 + 0.5f * a1) + 0.5f * (0.5f * a2 + 0.5f * a3));
    }
    v[15] = 0;
    uint4* d = reinterpret_cast<uint4*>(out + (size_t)o * 16);
#pragma unroll
    for (int q = 0; q < 2; ++q)
        d[q] = make_uint4((unsigned)v[q * 8] | ((unsigned)v[q * 8 + 1] << 16), (unsigned)v[q * 8 + 2] | ((unsigned)v[q * 8 + 3] << 16),
                          (unsigned)v[q * 8 + 4] | ((unsigned)v[q * 8 + 5] << 16), (unsigned)v[q * 8 + 6] | ((unsigned)v[q * 8 + 7] << 16));
}

// the 28 input channels of block K at full-size pixel (px, py)
template <int K>
__device__ __forceinline__ void ifInTap(const unsigned short* __restrict__ x, const unsigned short* __restrict__ f0,
                                        const unsigned short* __restrict__ f1, const unsigned short* __restrict__ ts,
                                        int W, int H, const float* __restrict__ t0, const float* __restrict__ t1,
                                        const float* __restrict__ t2, const float* __restrict__ t3, int px, int py,
                                        float v[28])
{
    const int plane = W * H;
    float f[4];
    ifFlowAt<K>(t0, t1, t2, t3, px, py, W, H, f);
    const IfUp T0 = ifTap(px, py, f[0], f[1], W, H), T1 = ifTap(px, py, f[2], f[3], W, H);
#pragma unroll
    for (int c = 0; c < 3; ++c)
    {
        v[c] = ifTapH(x + c * plane, T0);
        v[3 + c] = ifTapH(x + (3 + c) * plane, T1);
    }
#pragma unroll
    for (int c = 0; c < 4; ++c)
    {
        v[6 + c] = ifTapH(f0 + c * plane, T0);
        v[10 + c] = ifTapH(f1 + c * plane, T1);
    }
    v[14] = h2f(ts[py * W + px]);
    const float* tp = K == 1 ? t0 : K == 2 ? t1 : K == 3 ? t2 : t3;
    const int sp = 16 >> (K - 1), wp = W / sp, hp = H / sp, pl = wp * hp;
    const IfUp u = ifUp(px, py, sp, wp, hp);
    const float4* q4 = reinterpret_cast<const float4*>(tp);
    const float4 m1 = ifUpAt4(q4 + pl, u), m2 = ifUpAt4(q4 + 2 * pl, u), m3 = ifUpAt4(q4 + 3 * pl, u);
    v[15] = m1.x;
    v[16] = m1.y;
    v[17] = m1.z;
    v[18] = m1.w;
    v[19] = m2.x;
    v[20] = m2.y;
    v[21] = m2.z;
    v[22] = m2.w;
    v[23] = m3.x;
#pragma unroll
    for (int c = 0; c < 4; ++c)
        v[24 + c] = f[c];
}

// block K's 28 input channels at L0 pixel (X, Y): the full-size taps the author's resize reads, averaged
template <int K>
__device__ __forceinline__ void ifInAt(const unsigned short* __restrict__ x, const unsigned short* __restrict__ f0,
                                       const unsigned short* __restrict__ f1, const unsigned short* __restrict__ ts, int W,
                                       int H, const float* __restrict__ t0, const float* __restrict__ t1,
                                       const float* __restrict__ t2, const float* __restrict__ t3, int X, int Y,
                                       float r[28])
{
    constexpr int S = 16 >> K;
    if (S == 1)
        ifInTap<K>(x, f0, f1, ts, W, H, t0, t1, t2, t3, X, Y, r);
    else
    {
        constexpr int A = S / 2 - 1;
        float v0[28], v1[28];
        float r0[28];
        ifInTap<K>(x, f0, f1, ts, W, H, t0, t1, t2, t3, S * X + A, S * Y + A, v0);
        ifInTap<K>(x, f0, f1, ts, W, H, t0, t1, t2, t3, S * X + A + 1, S * Y + A, v1);
#pragma unroll
        for (int c = 0; c < 28; ++c)
            r0[c] = 0.5f * v0[c] + 0.5f * v1[c];
        ifInTap<K>(x, f0, f1, ts, W, H, t0, t1, t2, t3, S * X + A, S * Y + A + 1, v0);
        ifInTap<K>(x, f0, f1, ts, W, H, t0, t1, t2, t3, S * X + A + 1, S * Y + A + 1, v1);
#pragma unroll
        for (int c = 0; c < 28; ++c)
            r[c] = 0.5f * r0[c] + 0.5f * (0.5f * v0[c] + 0.5f * v1[c]);
    }
#pragma unroll
    for (int c = 24; c < 28; ++c)
        r[c] = r[c] * (1.0f / (float)S);
}

// block K's input at L0 into channels-last fp16 [L0][32] (the 28 channels + 4 zeros)
template <int K>
__device__ __forceinline__ void ifInBody(const unsigned short* __restrict__ x, const unsigned short* __restrict__ f0,
                                         const unsigned short* __restrict__ f1, const unsigned short* __restrict__ ts,
                                         int W, int H, const float* __restrict__ t0, const float* __restrict__ t1,
                                         const float* __restrict__ t2, const float* __restrict__ t3,
                                         unsigned short* __restrict__ out)
{
    constexpr int S = 16 >> K;
    const int wl = W / S, hl = H / S;
    const int X = blockIdx.x * 32 + (threadIdx.x & 31), Y = blockIdx.y * 8 + (threadIdx.x >> 5);
    if (X >= wl || Y >= hl)
        return;
    float r[28];
    ifInAt<K>(x, f0, f1, ts, W, H, t0, t1, t2, t3, X, Y, r);
    uint4* d = reinterpret_cast<uint4*>(out + (size_t)(Y * wl + X) * 32);
#pragma unroll
    for (int q = 0; q < 4; ++q)
    {
        unsigned u[4];
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            const int c = q * 8 + 2 * k;
            u[k] = (c < 28 ? (unsigned)f2h(r[c]) : 0u) | ((c + 1 < 28 ? (unsigned)f2h(r[c + 1]) : 0u) << 16);
        }
        d[q] = make_uint4(u[0], u[1], u[2], u[3]);
    }
}

#define IF_IN(K)                                                                                                       \
    extern "C" __global__ void __launch_bounds__(256)                                                                  \
        k_ifIn##K##h(const unsigned short* __restrict__ x, const unsigned short* __restrict__ f0,                      \
                     const unsigned short* __restrict__ f1, const unsigned short* __restrict__ ts, int W, int H,        \
                     const float* __restrict__ t0, const float* __restrict__ t1, const float* __restrict__ t2,         \
                     const float* __restrict__ t3, unsigned short* __restrict__ out)                                   \
    {                                                                                                                  \
        ifInBody<K>(x, f0, f1, ts, W, H, t0, t1, t2, t3, out);                                                         \
    }
IF_IN(1)
IF_IN(2)
IF_IN(3)

)IFNSRC";
static const char kIfnB[] = R"IFNSRC(
// the stored index of halo pixel sp and its chunk k's swizzled slot (each ldmatrix's 8 rows in 8 bank groups)
template <int NCH>
__device__ __forceinline__ unsigned ifSlot(int sp, int k)
{
    const int sw = NCH == 8 ? (k ^ (sp & 7)) : NCH == 4 ? (k ^ ((sp >> 1) & 3)) : (k ^ ((sp >> 2) & 1));
    return (unsigned)((sp * NCH + sw) << 4);
}

__device__ __forceinline__ uint4 ifLds(unsigned addr)
{
    uint4 v;
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr));
    return v;
}

// one conv, 128 threads: x = cin channels-last fp16 at W x H; y = cout channels-last fp16 at W / S x H / S; wf[z][step]
// [lane] (step = (group * 9 + tap) * CG / 16 + kc, the B registers of the z's two n-tiles); bias, beta fp32. The 4 x 32
// output tile's CG-channel halo (a stride-2 halo row keeps its even offsets before its odd ones, so a tap's 8 rows sit
// next to each other) and the group's weights by cp.async into shared memory, one wait a group. RES: y = LeakyRelu(conv
// x beta + x), x at the output pixel = the residual
template <int S, int CG, bool RES>
__device__ __forceinline__ void ifConvT(const unsigned short* __restrict__ x, int cin, int H, int W,
                                        unsigned short* __restrict__ y, int cout, const uint4* __restrict__ wf,
                                        const float* __restrict__ bias, const float* __restrict__ beta)
{
    constexpr int TH = 4, NB = 2;
    constexpr int HR = S == 1 ? TH + 2 : 2 * TH + 1, HC = S == 1 ? 34 : 65;
    constexpr int NCH = CG / 8, KC = CG / 16;
    constexpr int HALO = HR * HC * NCH * 16, GQ = 9 * KC * 32;
    extern __shared__ __align__(128) unsigned char ifSm[];
    const unsigned halo = (unsigned)__cvta_generic_to_shared(ifSm), wsm = halo + HALO;
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
            rnCp16(halo + ifSlot<NCH>(sp, k), src, in ? 16 : 0);
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
                const uint4 bw = ifLds(wsm + (unsigned)((tap * KC + kc) * 32 + lane) * 16u);
#pragma unroll
                for (int mi = 0; mi < 2; ++mi)
                {
                    unsigned a[4];
                    rnLdm4(halo + ifSlot<NCH>(pa[mi], kc * 2 + akc), a);
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
                v0 = ifLr(v0);
                v1 = ifLr(v1);
                *reinterpret_cast<unsigned*>(y + pix * cout + n0) = (unsigned)f2h(v0) | ((unsigned)f2h(v1) << 16);
            }
        }
    }
}

#define IF_CONV(NAME, S, CG, RES)                                                                                      \
    extern "C" __global__ void __launch_bounds__(128, 2)                                                               \
        NAME(const unsigned short* __restrict__ x, int cin, int H, int W, unsigned short* __restrict__ y, int cout,    \
             const uint4* __restrict__ wf, const float* __restrict__ bias, const float* __restrict__ beta)             \
    {                                                                                                                  \
        ifConvT<S, CG, RES>(x, cin, H, W, y, cout, wf, bias, beta);                                                    \
    }
IF_CONV(k_ifS2g16, 2, 16, false)
IF_CONV(k_ifS2g32, 2, 32, false)
IF_CONV(k_ifResg32, 1, 32, true)
IF_CONV(k_ifResg64, 1, 64, true)

)IFNSRC";
static const char kIfnC2[] = R"IFNSRC(
// ConvTranspose c -> 4 NP + PixelShuffle(2) on the tensor cores, the 4 output parity classes of a 4 x 32 L2 tile in one
// block: class (py, px) = the L1 pixel (2 yy + py, 2 xx + px) reads rows yy - 1 + py + ti, columns xx - 1 + px + tj
// (kernel taps ky = 3 - 2 (py + ti) + py, kx the same way), so each of the 9 halo positions' A fragments serves the
// classes that read it; fp32 accumulators from the bias; the input = the 8th ResConv's fp16 output. blockIdx.z = the
// 16-output group zc (4 NP padded to 16 nz, the pad's weights zero) = 4 shuffled planes = quad zc of the tmp; the
// group's L0 tile (16 x 128) is staged in shared memory, then stored as quads [zc][h0][w0] (h0 = 4 H, w0 = 4 W).
// FINAL (block 4, NP 5): group 0 (the flow) adds the blocks before (ifFlowAt<4>) and stores the fp16 flow planes, group
// 1's plane 0 stores sigmoid(mask) in fp16. wf[zc][group][class][tap][kc][lane]
template <int CG, bool FINAL>
__device__ __forceinline__ void ifLastQ(const unsigned short* __restrict__ x, int cin, int H, int W,
                                        float* __restrict__ out, const uint4* __restrict__ wf,
                                        const float* __restrict__ bias, const float* __restrict__ t0,
                                        const float* __restrict__ t1, const float* __restrict__ t2,
                                        const float* __restrict__ t3, unsigned short* __restrict__ flow,
                                        unsigned short* __restrict__ mask)
{
    constexpr int TH = 4, NB = 2, HR = TH + 2, HC = 34;
    constexpr int NCH = CG / 8, KC = CG / 16;
    constexpr int HALO = HR * HC * NCH * 16, GQ = 16 * KC * 32;
    extern __shared__ __align__(128) unsigned char ifSm[];
    const unsigned halo = (unsigned)__cvta_generic_to_shared(ifSm), wsm = halo + HALO;
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
            rnCp16(halo + ifSlot<NCH>(p, k), src, in ? 16 : 0);
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
                    rnLdm4(halo + ifSlot<NCH>((warp + dy) * HC + mi * 16 + arow + dx, kc * 2 + akc), a[mi]);
#pragma unroll
                for (int cl = 0; cl < 4; ++cl)
                {
                    const int ti = dy - (cl >> 1), tj = dx - (cl & 1);
                    if (ti < 0 || ti > 1 || tj < 0 || tj > 1)
                        continue;
                    const uint4 bw = ifLds(wsm + (unsigned)(((cl * 4 + ti * 2 + tj) * KC + kc) * 32 + lane) * 16u);
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
    float* st = reinterpret_cast<float*>(ifSm);
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
        if (FINAL)
        {
            const int o = Y * w0 + X, plane = w0 * h0;
            if (zc == 0)
            {
                float f[4];
                ifFlowAt<4>(t0, t1, t2, t3, X, Y, w0, h0, f);
#pragma unroll
                for (int c = 0; c < 4; ++c)
                    flow[c * plane + o] = f2h(f[c] + st[(c * 16 + r) * 128 + cx]);
            }
            else
                mask[o] = f2h(1.0f / (1.0f + expf(-st[r * 128 + cx])));
        }
        else
            reinterpret_cast<float4*>(out)[((size_t)zc * h0 + Y) * w0 + X] =
                make_float4(st[r * 128 + cx], st[(16 + r) * 128 + cx], st[(32 + r) * 128 + cx], st[(48 + r) * 128 + cx]);
    }
}

#define IF_LASTQ(CG)                                                                                                   \
    extern "C" __global__ void __launch_bounds__(128)                                                                  \
        k_ifLastQ##CG(const unsigned short* __restrict__ x, int cin, int H, int W, float* __restrict__ out,            \
                      const uint4* __restrict__ wf, const float* __restrict__ bias)                                    \
    {                                                                                                                  \
        ifLastQ<CG, false>(x, cin, H, W, out, wf, bias, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);         \
    }
IF_LASTQ(32)
IF_LASTQ(64)
extern "C" __global__ void __launch_bounds__(128)
    k_ifLastQF32(const unsigned short* __restrict__ x, int cin, int H, int W, const uint4* __restrict__ wf,
                 const float* __restrict__ bias, const float* __restrict__ t0, const float* __restrict__ t1,
                 const float* __restrict__ t2, const float* __restrict__ t3, unsigned short* __restrict__ flow,
                 unsigned short* __restrict__ mask)
{
    ifLastQ<32, true>(x, cin, H, W, nullptr, wf, bias, t0, t1, t2, t3, flow, mask);
}

// block 4's conv0[0] (32 padded input channels -> 16, 3 x 3 stride 2) + LeakyRelu with its input built in the halo: each
// halo pixel's 28 channels = ifInAt<4> (the pixel itself), rounded to fp16, 4 zeros; y = channels-last fp16 at L1
// [H / 2][W / 2][16]; wf[tap * 2 + kc][lane]
extern "C" __global__ void __launch_bounds__(128)
    k_ifInC0_4(const unsigned short* __restrict__ x, const unsigned short* __restrict__ f0,
               const unsigned short* __restrict__ f1, const unsigned short* __restrict__ ts, int W, int H,
               const float* __restrict__ t0, const float* __restrict__ t1, const float* __restrict__ t2,
               const float* __restrict__ t3, unsigned short* __restrict__ y, const uint4* __restrict__ wf,
               const float* __restrict__ bias)
{
    constexpr int TH = 4, NB = 2, HR = 2 * TH + 1, HC = 65, NCH = 4, KC = 2;
    constexpr int HALO = HR * HC * NCH * 16, GQ = 9 * KC * 32;
    extern __shared__ __align__(128) unsigned char ifSm[];
    const unsigned halo = (unsigned)__cvta_generic_to_shared(ifSm), wsm = halo + HALO;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int tx0 = blockIdx.x * 32, ty0 = blockIdx.y * TH;
    const int OW = W / 2, OH = H / 2;
    for (int i = threadIdx.x; i < GQ; i += 128)
        rnCp16(wsm + (unsigned)i * 16u, wf + i, 16);
    asm volatile(SMV_CP_COMMIT);
    for (int p = threadIdx.x; p < HR * HC; p += 128)
    {
        const int hr = p / HC, hc = p - hr * HC;
        const int gy = 2 * ty0 - 1 + hr, gx = 2 * tx0 - 1 + hc;
        const int sp = hr * HC + ((hc & 1) ? 33 + (hc >> 1) : (hc >> 1));
        float r[28];
        const bool in = gy >= 0 && gy < H && gx >= 0 && gx < W;
        if (in)
            ifInAt<4>(x, f0, f1, ts, W, H, t0, t1, t2, t3, gx, gy, r);
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            unsigned u[4];
#pragma unroll
            for (int q = 0; q < 4; ++q)
            {
                const int c = k * 8 + 2 * q;
                u[q] = in ? ((c < 28 ? (unsigned)f2h(r[c]) : 0u) | ((c + 1 < 28 ? (unsigned)f2h(r[c + 1]) : 0u) << 16))
                          : 0u;
            }
            *reinterpret_cast<uint4*>(ifSm + ifSlot<NCH>(sp, k)) = make_uint4(u[0], u[1], u[2], u[3]);
        }
    }
    asm volatile(SMV_CP_WAIT0);
    __syncthreads();
    const int g = lane >> 2, t4 = lane & 3;
    const int arow = (lane & 7) + ((lane >> 3) & 1) * 8, akc = lane >> 4;
    const int oy = ty0 + warp;
    float acc[2][NB][4];
#pragma unroll
    for (int nj = 0; nj < NB; ++nj)
    {
        const float b0 = bias[nj * 8 + t4 * 2], b1 = bias[nj * 8 + t4 * 2 + 1];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
        {
            acc[mi][nj][0] = acc[mi][nj][2] = b0;
            acc[mi][nj][1] = acc[mi][nj][3] = b1;
        }
    }
#pragma unroll 1
    for (int tap = 0; tap < 9; ++tap)
    {
        const int dy = tap / 3, dx = tap - dy * 3;
        int pa[2];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
        {
            const int col = mi * 16 + arow;
            pa[mi] = (2 * warp + dy) * HC + (dx == 1 ? 33 + col : col + (dx >> 1));
        }
#pragma unroll
        for (int kc = 0; kc < KC; ++kc)
        {
            const uint4 bw = ifLds(wsm + (unsigned)((tap * KC + kc) * 32 + lane) * 16u);
#pragma unroll
            for (int mi = 0; mi < 2; ++mi)
            {
                unsigned a[4];
                rnLdm4(halo + ifSlot<NCH>(pa[mi], kc * 2 + akc), a);
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
    if (oy >= OH)
        return;
#pragma unroll
    for (int nj = 0; nj < NB; ++nj)
    {
        const int n0 = nj * 8 + t4 * 2;
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int h = 0; h < 2; ++h)
            {
                const int ox = tx0 + mi * 16 + g + h * 8;
                if (ox >= OW)
                    continue;
                const float v0 = ifLr(acc[mi][nj][2 * h]), v1 = ifLr(acc[mi][nj][2 * h + 1]);
                *reinterpret_cast<unsigned*>(y + ((size_t)oy * OW + ox) * 16 + n0) =
                    (unsigned)f2h(v0) | ((unsigned)f2h(v1) << 16);
            }
    }
}
)IFNSRC";
static const char* const kIfnKern[] = {"k_ifIn0h",   "k_ifIn1h",    "k_ifIn2h",    "k_ifIn3h",
                                       "k_ifInC0_4", "k_ifS2g16",   "k_ifS2g32",   "k_ifResg32",
                                       "k_ifResg64", "k_ifLastQ32", "k_ifLastQ64", "k_ifLastQF32"};
enum
{
    kIfIn0,
    kIfIn1,
    kIfIn2,
    kIfIn3,
    kIfInC04,
    kIfS2g16,
    kIfS2g32,
    kIfRes32,
    kIfRes64,
    kIfLq32,
    kIfLq64,
    kIfLqF
};
static const int kIfnC[5] = {192, 128, 96, 64, 32}; // each block's width c
static const int kIfnCin[5] = {15, 28, 28, 28,
                               28}; // conv0[0]'s input channels, padded to kIfnCp (the pad's weights zero)
static const int kIfnCp[5] = {16, 32, 32, 32, 32};
static const int kIfnC1G[5] = {32, 32, 16, 32, 16}; // conv0[1]'s input-channel group (its input = c / 2)
static const int kIfnRsG[5] = {64, 64, 32, 64, 32}; // the ResConvs' and the transposed conv's group
static const int kIfnNp[5] = {13, 13, 13, 13, 5}; // the shuffled planes kept (the last block's 5 = the flow + the mask)
// a block's weight pieces: [0] conv0[0]'s fragments, [1] its bias, [2] / [3] conv0[1]'s, [4 + 3 r] / [5 + 3 r] / [6 + 3 r]
// ResConv r's fragments / bias / beta, [28] the transposed conv's fragments, [29] its bias padded to 16 nz
constexpr int kIfnPieces = 30;

// dynamic shared memory: a stride-2 conv's halo + one group's weights, a ResConv's, the transposed conv's (its staged
// L0 tile when larger), block 4's fused input + conv0[0]
static int ifnSmS2(int cg)
{
    return (9 * 65 * (cg / 8) + 9 * (cg / 16) * 32) * 16;
}
static int ifnSmRes(int cg)
{
    return (6 * 34 * (cg / 8) + 9 * (cg / 16) * 32) * 16;
}
static int ifnSmLq(int cg)
{
    return (std::max)((6 * 34 * (cg / 8) + 16 * (cg / 16) * 32) * 16, 4 * 16 * 128 * 4);
}
constexpr int kIfnSmInC0 = 9 * 65 * 4 * 16 + 9 * 2 * 32 * 16;

// the transposed conv's weights [c][nout][4][4] (PyTorch's layout; RIFE 4.25's nout = 52, GMFSS's 24) in ifLastQ's
// fragment order [zc][group][class][tap][kc][lane][8 halves] for the first 4 np outputs, padded to 16 nz with zeros,
// rounded to fp16: class (py, px), tap (ti, tj) = halo offset (py + ti, px + tj) = kernel tap (3 - 2 (py + ti) + py, 3 -
// 2 (px + tj) + px); output o = 16 zc + 8 hf + lane / 4, channel = group cg + 16 kc + 2 (lane % 4) (+ 1, + 8, + 9)
static std::vector<uint16_t> ifnLastFrags(std::span<const float> w, int c, int np, int cg, int nout = 52)
{
    static const int kOfs[4] = {0, 1, 8, 9};
    const int nz = (4 * np + 15) / 16, kcn = cg / 16, ng = c / cg;
    std::vector<uint16_t> out((size_t)nz * ng * 4 * 4 * kcn * 32 * 8);
    size_t i = 0;
    for (int zc = 0; zc < nz; zc++)
        for (int gi = 0; gi < ng; gi++)
            for (int cls = 0; cls < 4; cls++)
                for (int t = 0; t < 4; t++)
                {
                    const int py = cls >> 1, px = cls & 1, dy = py + (t >> 1), dx = px + (t & 1);
                    const int ky = 3 - 2 * dy + py, kx = 3 - 2 * dx + px;
                    for (int kc = 0; kc < kcn; kc++)
                        for (int l = 0; l < 32; l++)
                            for (int hf = 0; hf < 2; hf++)
                                for (int q = 0; q < 4; q++)
                                {
                                    const int o = zc * 16 + hf * 8 + (l >> 2);
                                    const int ch = gi * cg + kc * 16 + (l & 3) * 2 + kOfs[q];
                                    out[i++] = o < 4 * np ? f2hRne(w[(((size_t)ch * nout + o) * 4 + ky) * 4 + kx]) : 0;
                                }
                }
    return out;
}

// flownet.pkl's IFNet in the strict IFNet_HDv3 form (module.block<k>.: conv0.0.0 [c / 2][cin][3][3], conv0.1.0
// [c][c / 2][3][3], convblock.0 .. 7 {conv [c][c][3][3], beta [1][c][1][1]}, lastconv.0 [c][52][4][4], their biases;
// c = kIfnC, cin = kIfnCin) as one weight block of 256-byte aligned pieces, off[kIfnPieces k + i]; false = missing, or not
// this net (`why`)
static bool ifnNatLoad(std::vector<uint8_t>& blob, std::vector<size_t>& off, std::string& why)
{
    PthFile pf;
    if (!pthStateDict(g_nativeEngineDir + L"\\rife\\flownet.pkl", pf, why))
        return false;
    auto& wts = pf.t;
    const auto& S = pf.S;
    auto has = [&](const std::string& n, const std::vector<int64_t>& s) {
        const auto it = S.find(n);
        return it != S.end() && it->second == s;
    };
    bool form = true;
    for (int k = 0; k < 5 && form; k++)
    {
        const std::string p = "module.block" + std::to_string(k) + ".";
        const int64_t c = kIfnC[k], h = c / 2, ci = kIfnCin[k];
        form = has(p + "conv0.0.0.weight", {h, ci, 3, 3}) && has(p + "conv0.0.0.bias", {h}) &&
               has(p + "conv0.1.0.weight", {c, h, 3, 3}) && has(p + "conv0.1.0.bias", {c}) &&
               has(p + "lastconv.0.weight", {c, 52, 4, 4}) && has(p + "lastconv.0.bias", {52}) &&
               !S.count(p + "convblock.8.beta");
        for (int r = 0; r < 8 && form; r++)
        {
            const std::string q = p + "convblock." + std::to_string(r) + ".";
            form = has(q + "conv.weight", {c, c, 3, 3}) && has(q + "conv.bias", {c}) && has(q + "beta", {1, c, 1, 1});
        }
    }
    if (!form)
    {
        why = "its weights are not RIFE 4.25's IFNet";
        return false;
    }
    blob.clear();
    off.assign((size_t)5 * kIfnPieces, 0);
    auto put = [&](const void* d, size_t n) {
        const size_t o = (blob.size() + 255) & ~(size_t)255;
        blob.resize(o + n);
        memcpy(blob.data() + o, d, n);
        return o;
    };
    for (int k = 0; k < 5; k++)
    {
        const std::string p = "module.block" + std::to_string(k) + ".";
        const int c = kIfnC[k], h = c / 2, ci = kIfnCin[k], cp = kIfnCp[k];
        size_t* o = off.data() + (size_t)k * kIfnPieces;
        const std::span<const float> w0 = wts[p + "conv0.0.0.weight"];
        std::vector<float> w0p((size_t)h * cp * 9, 0.0f);
        for (int n = 0; n < h; n++)
            for (int i = 0; i < ci; i++)
                for (int t = 0; t < 9; t++)
                    w0p[((size_t)n * cp + i) * 9 + t] = w0[((size_t)n * ci + i) * 9 + t];
        const std::vector<uint16_t> f0 = fnFragOf(w0p, h, cp, cp, 2);
        o[0] = put(f0.data(), f0.size() * sizeof(uint16_t));
        o[1] = put(wts[p + "conv0.0.0.bias"].data(), (size_t)h * sizeof(float));
        const std::vector<uint16_t> f1 = fnFragOf(wts[p + "conv0.1.0.weight"], c, h, kIfnC1G[k], 2);
        o[2] = put(f1.data(), f1.size() * sizeof(uint16_t));
        o[3] = put(wts[p + "conv0.1.0.bias"].data(), (size_t)c * sizeof(float));
        for (int r = 0; r < 8; r++)
        {
            const std::string q = p + "convblock." + std::to_string(r) + ".";
            const std::vector<uint16_t> fr = fnFragOf(wts[q + "conv.weight"], c, c, kIfnRsG[k], 2);
            o[4 + 3 * r] = put(fr.data(), fr.size() * sizeof(uint16_t));
            o[5 + 3 * r] = put(wts[q + "conv.bias"].data(), (size_t)c * sizeof(float));
            o[6 + 3 * r] = put(wts[q + "beta"].data(), (size_t)c * sizeof(float));
        }
        const std::vector<uint16_t> fl = ifnLastFrags(wts[p + "lastconv.0.weight"], c, kIfnNp[k], kIfnRsG[k]);
        o[28] = put(fl.data(), fl.size() * sizeof(uint16_t));
        const std::span<const float> lb = wts[p + "lastconv.0.bias"];
        std::vector<float> lbp((size_t)16 * ((4 * kIfnNp[k] + 15) / 16), 0.0f);
        for (int i = 0; i < 4 * kIfnNp[k]; i++)
            lbp[i] = lb[i];
        o[29] = put(lbp.data(), lbp.size() * sizeof(float));
    }
    return true;
}

// the IFNet's scratch at the motion frame fw x fh (multiples of 64), 256-byte pieces: the 4 tmp quads (16 fp32 a pixel
// at blocks 0..3's L0), ONE input buffer (block 0's [L0][16] or blocks 1..3's [L0][32] fp16, the largest), ONE conv0[0]
// output ([L1][c / 2] fp16, the largest), ONE ping-pong pair ([L2][c] fp16, the largest)
struct IfnArena
{
    size_t tm[4] = {}, in = 0, y0 = 0, ya = 0, yb = 0, total = 0;
};
static IfnArena ifnArena(int fw, int fh)
{
    IfnArena a;
    size_t o = 0, inB = 0, y0B = 0, yB = 0;
    auto put = [&](size_t n) {
        const size_t at = o;
        o += (n + 255) & ~(size_t)255;
        return at;
    };
    for (int k = 0; k < 5; k++)
    {
        const int s = 16 >> k;
        const size_t n0 = (size_t)(fw / s) * (size_t)(fh / s);
        if (k < 4)
        {
            a.tm[k] = put(16 * n0 * sizeof(float));
            inB = (std::max)(inB, (size_t)kIfnCp[k] * n0 * sizeof(uint16_t));
        }
        y0B = (std::max)(y0B, (size_t)(kIfnC[k] / 2) * (n0 / 4) * sizeof(uint16_t));
        yB = (std::max)(yB, (size_t)kIfnC[k] * (n0 / 16) * sizeof(uint16_t));
    }
    a.in = put(inB);
    a.y0 = put(y0B);
    a.ya = put(yB);
    a.yb = put(yB);
    a.total = o;
    return a;
}

// the IFNet's frame: the motion frame (two domains), else the pictures' pad
static void ifnFrame(const NativeRife& nr, int& fw, int& fh)
{
    fw = nr.mph ? nr.mpw : nr.pw;
    fh = nr.mph ? nr.mph : nr.ph;
}

static size_t ifnNatBytes(const NativeRife& nr)
{
    if (!nr.ifnNat)
        return 0;
    int fw, fh;
    ifnFrame(nr, fw, fh);
    return ifnArena(fw, fh).total;
}

// this session's IFNet module: the weights read, its text compiled (or the cached cubin) and loaded, its kernels bound,
// the weight block uploaded; false = `why` (the session teardown unloads a module left in ifnMod)
static bool nativeIfnModule(NativeRife& nr, std::string& why)
{
    if (nr.ifnMod)
        return true;
    std::vector<uint8_t> blob;
    if (!ifnNatLoad(blob, nr.ifnOff, why))
        return false;
    std::string text = kFnHead;
    text += kIfnA;
    text += kIfnB;
    text += kIfnC2;
    std::vector<char> cubin;
    if (!nativeCubin(text.c_str(), "smv_ifnet.cu", L"smv_ifnet", "IFNet kernels",
                     nr.kernCacheDir.empty() ? std::wstring(L".") : nr.kernCacheDir, cubin))
    {
        why = "its kernels did not compile";
        return false;
    }
    if (cuModuleLoadData(&nr.ifnMod, cubin.data()) != CUDA_SUCCESS)
    {
        nr.ifnMod = nullptr;
        why = "its kernel module did not load";
        return false;
    }
    for (int i = 0; i < (int)(sizeof(kIfnKern) / sizeof(kIfnKern[0])); i++)
        if (cuModuleGetFunction(&nr.fIfn[i], nr.ifnMod, kIfnKern[i]) != CUDA_SUCCESS)
        {
            why = std::string("its kernel ") + kIfnKern[i] + " is missing";
            return false;
        }
    // the one kernel above the default 48 KB of dynamic shared memory: the 64-channel transposed conv
    if (cuFuncSetAttribute(nr.fIfn[kIfLq64], CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, ifnSmLq(64)) !=
        CUDA_SUCCESS)
    {
        why = "its transposed conv's shared memory was refused";
        return false;
    }
    if (cudaMalloc(&nr.dIfnW, blob.size()) != cudaSuccess ||
        cudaMemcpy(nr.dIfnW, blob.data(), blob.size(), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        why = "its weights did not upload";
        return false;
    }
    return true;
}

// the IFNet's 59 launches on st, its scratch at s (ifnArena's pieces). false = a launch failed
static bool nativeIfnLaunch(NativeRife& nr, const void* x, const void* ts, const void* f0, const void* f1, int fw,
                            int fh, void* flow, void* mask, uint8_t* s, cudaStream_t st)
{
    const IfnArena ar = ifnArena(fw, fh);
    void* tm[4] = {s + ar.tm[0], s + ar.tm[1], s + ar.tm[2], s + ar.tm[3]};
    void* inb = s + ar.in;
    void* y0 = s + ar.y0;
    void* yab[2] = {s + ar.ya, s + ar.yb};
    void* nul = nullptr;
    const char* wb = (const char*)nr.dIfnW;
    int iw = fw, ih = fh;
    auto go = [&](int f, int gx, int gy, int gz, int threads, int sm, void** a) {
        return cuLaunchKernel(nr.fIfn[f], (unsigned)gx, (unsigned)gy, (unsigned)gz, (unsigned)threads, 1, 1,
                              (unsigned)sm, (CUstream)st, a, nullptr) == CUDA_SUCCESS;
    };
    for (int k = 0; k < 5; k++)
    {
        const int sc = 16 >> k, c = kIfnC[k], hc = c / 2;
        int w0 = fw / sc, h0 = fh / sc, w1 = w0 / 2, h1 = h0 / 2, w2 = w0 / 4, h2 = h0 / 4;
        const size_t* o = nr.ifnOff.data() + (size_t)k * kIfnPieces;
        const void* c0w = wb + o[0];
        const void* c0b = wb + o[1];
        const void* c1w = wb + o[2];
        const void* c1b = wb + o[3];
        const void* lw = wb + o[28];
        const void* lb = wb + o[29];
        if (k == 0)
        {
            void* a[] = {(void*)&x, (void*)&f0, (void*)&f1, (void*)&ts, &iw, &ih, &inb};
            if (!go(kIfIn0, (w0 + 31) / 32, (h0 + 7) / 8, 1, 256, 0, a))
                return false;
        }
        else if (k < 4)
        {
            void* a[] = {(void*)&x, (void*)&f0, (void*)&f1, (void*)&ts, &iw, &ih, &tm[0], &tm[1], &tm[2], &tm[3], &inb};
            if (!go(kIfIn0 + k, (w0 + 31) / 32, (h0 + 7) / 8, 1, 256, 0, a))
                return false;
        }
        if (k < 4)
        {
            int cin = kIfnCp[k], cout = hc;
            void* a[] = {&inb, &cin, &h0, &w0, &y0, &cout, (void*)&c0w, (void*)&c0b, &nul};
            if (!go(cin == 16 ? kIfS2g16 : kIfS2g32, (w1 + 31) / 32, (h1 + 3) / 4, hc / 16, 128, ifnSmS2(cin), a))
                return false;
        }
        else
        {
            void* a[] = {(void*)&x, (void*)&f0, (void*)&f1, (void*)&ts, &iw,         &ih,        &tm[0],
                         &tm[1],    &tm[2],     &tm[3],     &y0,        (void*)&c0w, (void*)&c0b};
            if (!go(kIfInC04, (w1 + 31) / 32, (h1 + 3) / 4, 1, 128, kIfnSmInC0, a))
                return false;
        }
        {
            int cin = hc, cout = c;
            void* a[] = {&y0, &cin, &h1, &w1, &yab[0], &cout, (void*)&c1w, (void*)&c1b, &nul};
            if (!go(kIfnC1G[k] == 16 ? kIfS2g16 : kIfS2g32, (w2 + 31) / 32, (h2 + 3) / 4, c / 16, 128,
                    ifnSmS2(kIfnC1G[k]), a))
                return false;
        }
        for (int r = 0; r < 8; r++)
        {
            int cc = c;
            void* xi = yab[r & 1];
            void* yo = yab[(r & 1) ^ 1];
            const void* rw = wb + o[4 + 3 * r];
            const void* rb = wb + o[5 + 3 * r];
            const void* rbe = wb + o[6 + 3 * r];
            void* a[] = {&xi, &cc, &h2, &w2, &yo, &cc, (void*)&rw, (void*)&rb, (void*)&rbe};
            if (!go(kIfnRsG[k] == 64 ? kIfRes64 : kIfRes32, (w2 + 31) / 32, (h2 + 3) / 4, c / 16, 128,
                    ifnSmRes(kIfnRsG[k]), a))
                return false;
        }
        // the 8th ResConv wrote its output into yab[0]
        int cin = c;
        if (k < 4)
        {
            void* a[] = {&yab[0], &cin, &h2, &w2, &tm[k], (void*)&lw, (void*)&lb};
            if (!go(kIfnRsG[k] == 64 ? kIfLq64 : kIfLq32, (w2 + 31) / 32, (h2 + 3) / 4, 4, 128, ifnSmLq(kIfnRsG[k]), a))
                return false;
        }
        else
        {
            void* a[] = {&yab[0], &cin, &h2, &w2, (void*)&lw, (void*)&lb, &tm[0], &tm[1], &tm[2], &tm[3], &flow, &mask};
            if (!go(kIfLqF, (w2 + 31) / 32, (h2 + 3) / 4, 2, 128, ifnSmLq(32), a))
                return false;
        }
    }
    return true;
}

// the IFNet on one pair on st (nr.stream): x = the pair's 6 fp16 planes, ts = the timestep plane (fp16), f0 / f1 = the
// encodes (kRifeFeatC fp16 planes), all fw x fh; flow / mask = its outputs (4 + 1 fp16 planes); the launches go through
// nr.ifnGraphs (chainGraphRun). false = the scratch or a launch failed
static bool nativeIfnRun(NativeRife& nr, const void* x, const void* ts, const void* f0, const void* f1, int fw, int fh,
                         void* flow, void* mask, cudaStream_t st)
{
    const IfnArena ar = ifnArena(fw, fh);
    uint8_t* s = nr.dGmCtxMem && (size_t)nr.gmCtxMemB >= ar.total ? (uint8_t*)nr.dGmCtxMem : (uint8_t*)nr.dIfnOwn;
    if (!s && cudaMalloc(&nr.dIfnOwn, ar.total) == cudaSuccess)
        s = (uint8_t*)nr.dIfnOwn;
    if (!s)
        return false;
    return chainGraphRun(nr.ifnGraphs, {s, x, ts, f0, f1, flow, mask}, fw, fh, st, "rife ifnet",
                         [&] { return nativeIfnLaunch(nr, x, ts, f0, f1, fw, fh, flow, mask, s, st); });
}
