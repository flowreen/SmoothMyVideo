// part of smv-live-native.inl (included there, in place); not a standalone translation unit
// GMFSS's fusionnet (FusionNet_u.GridNet at the half: 4 heads, 9 ResidualBlocks, 4 DownsampleBlocks, 4 UpsampleBlocks,
// the tail up to its 64 -> 256 conv) as 45 launches (nativeFuRun) of its own module smv_fusion_*.cubin (kFnHead's
// helpers + kFuA / kFuB), weights = engine\GMFSS_Fortuna\train_log\fusionnet.pkl through pthStateDict (fuNatLoad); no
// fallback (a missing / wrong pkl = `fusionnet unavailable (...)`, the session refuses). Every
// activation channels-last fp16 at its level (0 = the half, 64 channels; 1 = the quarter, 128; 2 = the eighth, 192), fp16
// sums in every conv but head0's first. Its inputs: a = dGmFa's 9 fp32 planes (k_fuIn applies head0's first PReLU), b /
// c / d = the feature splats' channels-last stores (the other heads' first PReLU applied there: gmLr); its output = the
// 64 -> 256 conv before the pixel shuffle, channels-last fp16 into dGmAcc, which k_fuTailNhwc shuffles and runs
// conv_last on (dGmWl / dGmBl). Its activations (fuArena: the buffers alive at once per level, fuPlan) live in the
// shared activation block (sound while serial on nr.stream), else dFuOwn.
static const char kFuA[] = R"FUSRC(
// GMFSS's fusionnet (FusionNet_u.GridNet) on the host's own kernels at the GMFSS half W x H (multiples of 32): level 0
// = W x H (64 channels), level 1 = W / 2 x H / 2 (128), level 2 = W / 4 x H / 4 (192); every tensor channels-last fp16
// [h][w][c], the inputs b / c / d as the splat stores leave them (the heads' first PReLU already applied).
// k_fuIn      a (9 fp32 planes) through head0's first PReLU (fp32, one rounding) into [H][W][16] (7 zeros)
// k_fuC16     head0's first conv 16 -> 64 (the padded input): implicit GEMM on the tensor cores (mma.sync m16n8k16, fp32
//             accumulators from the bias, a 4 x 32 output tile, 16 outputs a block), the halo and weights by cp.async
// k_fuW64*    every other stride-1 conv (cin % 64 == 0): an 8 x 32 output tile (a warp = 2 rows = 4 m-tiles), 64 outputs
//             a block, the weights from global memory in fragment order a k-step ahead, the halo in shared memory, fp16
//             accumulators from the bias; P: the halo through PReLU(slope) in fp16 (the PReLU before the conv; zeros stay
//             zeros); 1 / 2: y = conv + add0 (+ add1) at the output pixel in fp32, one rounding to fp16
// k_fuW64P*4  the PReLU ones on a 4 x 32 output tile (a warp = 1 row = 2 m-tiles): a 6-row halo (26 KB), three CTAs an
//             SM; every output the same sum in the same order as its 8-row twin (the same bytes)
// k_fuW2PH    the stride-2 convs the same way with a 4 x 32 output tile (a warp = 1 row = 2 m-tiles), the halo 9 x 65 with
//             each row's even offsets before its odd ones, fp16 accumulators
// k_fuUp64H   ConvTranspose cin -> cout (4 x 4, stride 2, pad 1) from W x H to 2W x 2H through PReLU: per output parity
//             class (py, px) a 2 x 2-tap conv (input row yy - 1 + py + ti, kernel row 3 - py - 2 ti, the columns the
//             same way), the 4 classes of a 4 x 32 input tile in one block, fp16 accumulators from the bias

// the stored index of halo pixel sp and its chunk k's swizzled slot (each ldmatrix's 8 rows in 8 bank groups)
template <int NCH>
__device__ __forceinline__ unsigned fuSlot(int sp, int k)
{
    const int sw = NCH == 8 ? (k ^ (sp & 7)) : NCH == 4 ? (k ^ ((sp >> 1) & 3)) : (k ^ ((sp >> 2) & 1));
    return (unsigned)((sp * NCH + sw) << 4);
}

__device__ __forceinline__ uint4 fuLds(unsigned addr)
{
    uint4 v;
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr));
    return v;
}

// PReLU on two packed fp16 values (x >= 0 ? x : x * a rounded once)
__device__ __forceinline__ unsigned fuPrelu2(unsigned u, float a)
{
    const unsigned short lo = (unsigned short)(u & 0xffffu), hi = (unsigned short)(u >> 16);
    const float flo = h2f(lo), fhi = h2f(hi);
    const unsigned short rlo = flo >= 0.0f ? lo : f2h(flo * a), rhi = fhi >= 0.0f ? hi : f2h(fhi * a);
    return (unsigned)rlo | ((unsigned)rhi << 16);
}

// the halo (n uint4 chunks from smem byte address base) through PReLU(a), in place
__device__ __forceinline__ void fuPreluHalo(unsigned char* base, int n, float a)
{
    uint4* p = reinterpret_cast<uint4*>(base);
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        uint4 v = p[i];
        v.x = fuPrelu2(v.x, a);
        v.y = fuPrelu2(v.y, a);
        v.z = fuPrelu2(v.z, a);
        v.w = fuPrelu2(v.w, a);
        p[i] = v;
    }
}

// head0's input: a = 9 fp32 planes at W x H; PReLU(slope) in fp32, one rounding; out = [H][W][16] fp16
extern "C" __global__ void __launch_bounds__(256) k_fuIn(const float* __restrict__ a, int W, int H, float slope,
                                                         unsigned short* __restrict__ out)
{
    const int x = blockIdx.x * 32 + (threadIdx.x & 31), y = blockIdx.y * 8 + (threadIdx.x >> 5);
    if (x >= W || y >= H)
        return;
    const size_t plane = (size_t)W * H, pix = (size_t)y * W + x;
    unsigned v[8];
#pragma unroll
    for (int k = 0; k < 8; ++k)
    {
        unsigned short h[2];
#pragma unroll
        for (int e = 0; e < 2; ++e)
        {
            const int c = 2 * k + e;
            float f = c < 9 ? a[c * plane + pix] : 0.0f;
            f = f >= 0.0f ? f : f * slope;
            h[e] = f2h(f);
        }
        v[k] = (unsigned)h[0] | ((unsigned)h[1] << 16);
    }
    uint4* o = reinterpret_cast<uint4*>(out + pix * 16);
    o[0] = make_uint4(v[0], v[1], v[2], v[3]);
    o[1] = make_uint4(v[4], v[5], v[6], v[7]);
}

// head0's first conv, 128 threads: x = 16 channels channels-last fp16 at W x H; y = cout channels-last fp16; wf[z][step]
// [lane] (step = tap, the B registers of the z's two n-tiles); bias fp32
extern "C" __global__ void __launch_bounds__(128, 2)
    k_fuC16(const unsigned short* __restrict__ x, int H, int W, unsigned short* __restrict__ y, int cout,
            const uint4* __restrict__ wf, const float* __restrict__ bias)
{
    constexpr int TH = 4, NB = 2, CG = 16, HR = TH + 2, HC = 34;
    constexpr int NCH = CG / 8, KC = CG / 16;
    constexpr int HALO = HR * HC * NCH * 16, GQ = 9 * KC * 32;
    extern __shared__ __align__(128) unsigned char fuSm[];
    const unsigned halo = (unsigned)__cvta_generic_to_shared(fuSm), wsm = halo + HALO;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int tx0 = blockIdx.x * 32, ty0 = blockIdx.y * TH, z = blockIdx.z;
    wf += (size_t)z * 9 * 32;
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
    for (int i = threadIdx.x; i < HR * HC * NCH; i += 128)
    {
        const int p = i / NCH, k = i - p * NCH;
        const int hr = p / HC, hc = p - hr * HC;
        const int gy = ty0 - 1 + hr, gx = tx0 - 1 + hc;
        const bool in = gy >= 0 && gy < H && gx >= 0 && gx < W;
        const unsigned short* src = in ? x + ((size_t)gy * W + gx) * CG + k * 8 : x;
        rnCp16(halo + fuSlot<NCH>(p, k), src, in ? 16 : 0);
    }
    for (int i = threadIdx.x; i < GQ; i += 128)
        rnCp16(wsm + (unsigned)i * 16u, wf + i, 16);
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
            pa[mi] = (warp + dy) * HC + mi * 16 + arow + dx;
#pragma unroll
        for (int kc = 0; kc < KC; ++kc)
        {
            const uint4 bw = fuLds(wsm + (unsigned)((tap * KC + kc) * 32 + lane) * 16u);
#pragma unroll
            for (int mi = 0; mi < 2; ++mi)
            {
                unsigned a[4];
                rnLdm4(halo + fuSlot<NCH>(pa[mi], kc * 2 + akc), a);
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
#pragma unroll
    for (int nj = 0; nj < NB; ++nj)
    {
        const int n0 = z * NB * 8 + nj * 8 + t4 * 2;
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
        {
            const int oy = ty0 + warp;
            if (oy >= H)
                continue;
#pragma unroll
            for (int h = 0; h < 2; ++h)
            {
                const int ox = tx0 + mi * 16 + g + h * 8;
                if (ox >= W)
                    continue;
                const float v0 = acc[mi][nj][2 * h], v1 = acc[mi][nj][2 * h + 1];
                *reinterpret_cast<unsigned*>(y + ((size_t)oy * W + ox) * cout + n0) =
                    (unsigned)f2h(v0) | ((unsigned)f2h(v1) << 16);
            }
        }
    }
}

)FUSRC";
static const char kFuB[] = R"FUSRC(
// the wide conv at stride S, 128 threads: x = cin channels-last fp16 at W x H; y = cout channels-last fp16 at W / S x
// H / S; wf[z][step][q][lane] (step = (group * 9 + tap) * CG / 16 + kc, q = an n-tile pair); bias fp32; add0 / add1 =
// cout channels-last fp16 at the output size (NADD of them); A32: fp32 accumulators, else fp16 ones
// TH1 = the output rows of a stride-1 tile (8, or 4 = a warp = 1 row); the stride-2 tile is always 4 rows
template <int S, int CG, int NB, bool PRE, int NADD, bool A32, int TH1 = 8>
__device__ __forceinline__ void fuConvW(const unsigned short* __restrict__ x, int cin, int H, int W,
                                        unsigned short* __restrict__ y, int cout, const uint4* __restrict__ wf,
                                        const float* __restrict__ bias, float slope,
                                        const unsigned short* __restrict__ add0, const unsigned short* __restrict__ add1)
{
    constexpr int TH = S == 1 ? TH1 : 4, RW = TH / 4, MI = 2 * RW;
    constexpr int HR = S == 1 ? TH + 2 : 2 * TH + 1, HC = S == 1 ? 34 : 65;
    const int OW = W / S, OH = H / S;
    constexpr int NCH = CG / 8, KC = CG / 16, NQ = NB / 2;
    extern __shared__ __align__(128) unsigned char fuSm[];
    const unsigned halo = (unsigned)__cvta_generic_to_shared(fuSm);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int tx0 = blockIdx.x * 32, ty0 = blockIdx.y * TH, z = blockIdx.z;
    const int ksteps = cin * 9 / 16;
    wf += (size_t)z * ksteps * NQ * 32;
    const int g = lane >> 2, t4 = lane & 3;
    float accF[A32 ? MI : 1][NB][4];
    unsigned accH[A32 ? 1 : MI][NB][2];
#pragma unroll
    for (int nj = 0; nj < NB; ++nj)
    {
        const float b0 = bias[z * NB * 8 + nj * 8 + t4 * 2], b1 = bias[z * NB * 8 + nj * 8 + t4 * 2 + 1];
#pragma unroll
        for (int mi = 0; mi < MI; ++mi)
        {
            if constexpr (A32)
            {
                accF[mi][nj][0] = accF[mi][nj][2] = b0;
                accF[mi][nj][1] = accF[mi][nj][3] = b1;
            }
            else
                accH[mi][nj][0] = accH[mi][nj][1] = (unsigned)f2h(b0) | ((unsigned)f2h(b1) << 16);
        }
    }
    uint4 bc[NQ], bn[NQ];
#pragma unroll
    for (int q = 0; q < NQ; ++q)
        bc[q] = __ldg(wf + q * 32 + lane);
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
            rnCp16(halo + fuSlot<NCH>(sp, k), src, in ? 16 : 0);
        }
        asm volatile(SMV_CP_COMMIT);
        asm volatile(SMV_CP_WAIT0);
        __syncthreads();
        if (PRE)
        {
            fuPreluHalo(fuSm, HR * HC * NCH, slope);
            __syncthreads();
        }
#pragma unroll 1
        for (int tap = 0; tap < 9; ++tap)
        {
            const int dy = tap / 3, dx = tap - dy * 3;
            int pa[MI];
#pragma unroll
            for (int mi = 0; mi < MI; ++mi)
            {
                const int col = (mi & 1) * 16 + arow;
                pa[mi] = S == 1 ? (RW * warp + (mi >> 1) + dy) * HC + col + dx
                                : (2 * warp + dy) * HC + (dx == 1 ? 33 + col : col + (dx >> 1));
            }
#pragma unroll
            for (int kc = 0; kc < KC; ++kc)
            {
                const int step = (gi * 9 + tap) * KC + kc;
                if (step + 1 < ksteps)
#pragma unroll
                    for (int q = 0; q < NQ; ++q)
                        bn[q] = __ldg(wf + ((step + 1) * NQ + q) * 32 + lane);
#pragma unroll
                for (int mi = 0; mi < MI; ++mi)
                {
                    unsigned a[4];
                    rnLdm4(halo + fuSlot<NCH>(pa[mi], kc * 2 + akc), a);
#pragma unroll
                    for (int q = 0; q < NB; ++q)
                    {
                        const unsigned b0 = (q & 1) ? bc[q >> 1].z : bc[q >> 1].x;
                        const unsigned b1 = (q & 1) ? bc[q >> 1].w : bc[q >> 1].y;
                        if constexpr (A32)
                        {
                            float* d = accF[mi][q];
                            asm volatile(SMV_MMA_F32 "{%0,%1,%2,%3}, "
                                         "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                                         : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                                         : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
                        }
                        else
                            asm volatile(SMV_MMA_F16 "{%0,%1}, {%2,%3,%4,%5}, "
                                         "{%6,%7}, {%0,%1};\n"
                                         : "+r"(accH[mi][q][0]), "+r"(accH[mi][q][1])
                                         : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
                    }
                }
#pragma unroll
                for (int q = 0; q < NQ; ++q)
                    bc[q] = bn[q];
            }
        }
    }
#pragma unroll
    for (int nj = 0; nj < NB; ++nj)
    {
        const int n0 = z * NB * 8 + nj * 8 + t4 * 2;
#pragma unroll
        for (int mi = 0; mi < MI; ++mi)
        {
            const int oy = ty0 + RW * warp + (mi >> 1);
            if (oy >= OH)
                continue;
#pragma unroll
            for (int h = 0; h < 2; ++h)
            {
                const int ox = tx0 + (mi & 1) * 16 + g + h * 8;
                if (ox >= OW)
                    continue;
                float v0, v1;
                if constexpr (A32)
                {
                    v0 = accF[mi][nj][2 * h];
                    v1 = accF[mi][nj][2 * h + 1];
                }
                else
                {
                    v0 = h2f((unsigned short)(accH[mi][nj][h] & 0xffffu));
                    v1 = h2f((unsigned short)(accH[mi][nj][h] >> 16));
                }
                const size_t at = ((size_t)oy * OW + ox) * cout + n0;
                if (NADD > 0)
                {
                    const unsigned u = *reinterpret_cast<const unsigned*>(add0 + at);
                    v0 += h2f((unsigned short)(u & 0xffffu));
                    v1 += h2f((unsigned short)(u >> 16));
                }
                if (NADD > 1)
                {
                    const unsigned u = *reinterpret_cast<const unsigned*>(add1 + at);
                    v0 += h2f((unsigned short)(u & 0xffffu));
                    v1 += h2f((unsigned short)(u >> 16));
                }
                *reinterpret_cast<unsigned*>(y + at) = (unsigned)f2h(v0) | ((unsigned)f2h(v1) << 16);
            }
        }
    }
}

#define FU_CONVW(NAME, S, CG, PRE, NADD, A32)                                                                          \
    extern "C" __global__ void __launch_bounds__(128)                                                                  \
        NAME(const unsigned short* __restrict__ x, int cin, int H, int W, unsigned short* __restrict__ y, int cout,    \
             const uint4* __restrict__ wf, const float* __restrict__ bias, float slope,                               \
             const unsigned short* __restrict__ add0, const unsigned short* __restrict__ add1)                        \
    {                                                                                                                  \
        fuConvW<S, CG, 8, PRE, NADD, A32>(x, cin, H, W, y, cout, wf, bias, slope, add0, add1);                         \
    }
FU_CONVW(k_fuW64H, 1, 64, false, 0, false)
FU_CONVW(k_fuW64PH, 1, 64, true, 0, false)
FU_CONVW(k_fuW64P1H, 1, 64, true, 1, false)
FU_CONVW(k_fuW64P2H, 1, 64, true, 2, false)
FU_CONVW(k_fuW2PH, 2, 32, true, 0, false)
#define FU_CONVW4(NAME, PRE, NADD)                                                                                     \
    extern "C" __global__ void __launch_bounds__(128)                                                                  \
        NAME(const unsigned short* __restrict__ x, int cin, int H, int W, unsigned short* __restrict__ y, int cout,    \
             const uint4* __restrict__ wf, const float* __restrict__ bias, float slope,                               \
             const unsigned short* __restrict__ add0, const unsigned short* __restrict__ add1)                        \
    {                                                                                                                  \
        fuConvW<1, 64, 8, PRE, NADD, false, 4>(x, cin, H, W, y, cout, wf, bias, slope, add0, add1);                    \
    }
FU_CONVW4(k_fuW64PH4, true, 0)
FU_CONVW4(k_fuW64P1H4, true, 1)
FU_CONVW4(k_fuW64P2H4, true, 2)

// ConvTranspose cin -> cout (4 x 4, stride 2, pad 1) through PReLU(slope): x = cin channels-last fp16 at W x H, y = cout
// channels-last fp16 at 2W x 2H; blockIdx.z = the 16-output group; wf[z][group][class][tap][kc][lane]
extern "C" __global__ void __launch_bounds__(128)
    k_fuUp64H(const unsigned short* __restrict__ x, int cin, int H, int W, unsigned short* __restrict__ y, int cout,
              const uint4* __restrict__ wf, const float* __restrict__ bias, float slope)
{
    constexpr int TH = 4, NB = 2, CG = 64, HR = TH + 2, HC = 34;
    constexpr int NCH = CG / 8, KC = CG / 16;
    constexpr int HALO = HR * HC * NCH * 16, GQ = 16 * KC * 32;
    extern __shared__ __align__(128) unsigned char fuSm[];
    const unsigned halo = (unsigned)__cvta_generic_to_shared(fuSm), wsm = halo + HALO;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int tx0 = blockIdx.x * 32, ty0 = blockIdx.y * TH, zc = blockIdx.z;
    wf += (size_t)zc * (cin / CG) * GQ;
    const int g = lane >> 2, t4 = lane & 3;
    unsigned acc[4][2][NB][2];
#pragma unroll
    for (int nj = 0; nj < NB; ++nj)
    {
        const unsigned b = (unsigned)f2h(bias[zc * 16 + nj * 8 + t4 * 2]) |
                           ((unsigned)f2h(bias[zc * 16 + nj * 8 + t4 * 2 + 1]) << 16);
#pragma unroll
        for (int cl = 0; cl < 4; ++cl)
#pragma unroll
            for (int mi = 0; mi < 2; ++mi)
                acc[cl][mi][nj][0] = acc[cl][mi][nj][1] = b;
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
            rnCp16(halo + fuSlot<NCH>(p, k), src, in ? 16 : 0);
        }
        for (int i = threadIdx.x; i < GQ; i += 128)
            rnCp16(wsm + (unsigned)i * 16u, wf + (size_t)gi * GQ + i, 16);
        asm volatile(SMV_CP_COMMIT);
        asm volatile(SMV_CP_WAIT0);
        __syncthreads();
        fuPreluHalo(fuSm, HR * HC * NCH, slope);
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
                    rnLdm4(halo + fuSlot<NCH>((warp + dy) * HC + mi * 16 + arow + dx, kc * 2 + akc), a[mi]);
#pragma unroll
                for (int cl = 0; cl < 4; ++cl)
                {
                    const int ti = dy - (cl >> 1), tj = dx - (cl & 1);
                    if (ti < 0 || ti > 1 || tj < 0 || tj > 1)
                        continue;
                    const uint4 bw = fuLds(wsm + (unsigned)(((cl * 4 + ti * 2 + tj) * KC + kc) * 32 + lane) * 16u);
#pragma unroll
                    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
                        for (int q = 0; q < NB; ++q)
                        {
                            const unsigned b0 = q ? bw.z : bw.x, b1 = q ? bw.w : bw.y;
                            asm volatile(SMV_MMA_F16 "{%0,%1}, "
                                         "{%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                                         : "+r"(acc[cl][mi][q][0]), "+r"(acc[cl][mi][q][1])
                                         : "r"(a[mi][0]), "r"(a[mi][1]), "r"(a[mi][2]), "r"(a[mi][3]), "r"(b0),
                                           "r"(b1));
                        }
                }
            }
        }
    }
    const int OW = 2 * W, OH = 2 * H;
#pragma unroll
    for (int cl = 0; cl < 4; ++cl)
    {
        const int py = cl >> 1, px = cl & 1;
        const int oy = 2 * (ty0 + warp) + py;
        if (oy >= OH)
            continue;
#pragma unroll
        for (int nj = 0; nj < NB; ++nj)
        {
            const int n0 = zc * 16 + nj * 8 + t4 * 2;
#pragma unroll
            for (int mi = 0; mi < 2; ++mi)
#pragma unroll
                for (int h = 0; h < 2; ++h)
                {
                    const int ox = 2 * (tx0 + mi * 16 + g + h * 8) + px;
                    if (ox >= OW)
                        continue;
                    *reinterpret_cast<unsigned*>(y + ((size_t)oy * OW + ox) * cout + n0) = acc[cl][mi][nj][h];
                }
        }
    }
}
)FUSRC";
static const char* const kFuKern[] = {"k_fuIn",   "k_fuC16",   "k_fuW64H",   "k_fuW64PH",   "k_fuW64P1H", "k_fuW64P2H",
                                      "k_fuW2PH", "k_fuUp64H", "k_fuW64PH4", "k_fuW64P1H4", "k_fuW64P2H4"};
static_assert(sizeof(kFuKern) / sizeof(kFuKern[0]) == sizeof(NativeRife::fFu) / sizeof(CUfunction),
              "one fFu slot a fusionnet kernel");
enum
{
    kFuIn,
    kFuC16,
    kFuW64,
    kFuW64P,
    kFuW64P1,
    kFuW64P2,
    kFuW2,
    kFuUp,
    kFuW64PR4, // the 4-row twins of kFuW64P / kFuW64P1 / kFuW64P2, in that order
    kFuW64P1R4,
    kFuW64P2R4
};
// fu_host.cu's dynamic shared memory: k_fuC16's halo + weights (a 4 x 32 tile), the wide stride-1 convs' halo (8 x 32)
// and their 4-row twins' (4 x 32), the stride-2 ones' (9 x 65), the transposed conv's halo + one group's weights (above
// the default 48 KB)
constexpr int kFuSmC16 = (6 * 34 * 2 + 9 * 32) * 16, kFuSmW64 = 10 * 34 * 8 * 16, kFuSmW64R4 = 6 * 34 * 8 * 16,
              kFuSmW2 = 9 * 65 * 4 * 16, kFuSmUp = (6 * 34 * 8 + 16 * 4 * 32) * 16;
// a level's channels and the most buffers one level may hold
constexpr int kFuCh[3] = {64, 128, 192}, kFuMaxSlots = 8;
// fuPlan's buffer codes: the chain's inputs and output, then level l's buffer i = kFuRefSlot + l kFuMaxSlots + i
enum
{
    kFuRefA,
    kFuRefB,
    kFuRefC,
    kFuRefD,
    kFuRefOut,
    kFuRefSlot = 8
};

// one step of the chain: kind ('i' = k_fuIn, 'c' = a stride-1 conv, 's' = a stride-2 conv, 'u' = the transposed conv),
// buffers by code (add0 / add1 = -1: none), the input's level, the state-dict names of its conv and of the PReLU before
// it ("" = none), the channels in and out
struct FuStep
{
    char kind = 'c';
    int src = 0, dst = 0, add0 = -1, add1 = -1, lv = 0, cin = 0, cout = 0;
    std::string conv, prelu;
};
struct FuPlan
{
    std::vector<FuStep> steps;
    int slots[3] = {};
};

// the GridNet's forward (FusionNet_u.GridNet's order) with every intermediate in a level buffer: a step's output takes a free buffer of its level, its inputs free theirs after their
// last read, so no step writes a buffer it reads (4 / 4 / 3 buffers)
static const FuPlan& fuPlan()
{
    static const FuPlan plan = [] {
        struct Op
        {
            char kind;
            std::string src, dst, conv, prelu;
            std::vector<std::string> adds;
            int lv;
        };
        std::vector<Op> ops;
        const std::string h0 = "residual_model_head0.";
        ops.push_back({'i', "a", "A16", "", h0 + "0", {}, 0});
        ops.push_back({'c', "A16", "h0a", h0 + "1", "", {}, 0});
        ops.push_back({'c', "h0a", "h0", h0 + "3", h0 + "2", {}, 0});
        // a block {PReLU, conv (stride 1 / 2, or transposed), PReLU, conv}: the heads' first PReLU is the splats'
        auto block = [&](const char* name, const char* src, const char* mid, const char* dst, int lv,
                         std::vector<std::string> adds, char k0, bool pre0) {
            const std::string m = std::string(name) + ".";
            const int lv1 = k0 == 's' ? lv + 1 : k0 == 'u' ? lv - 1 : lv;
            ops.push_back({k0, src, mid, m + "1", pre0 ? m + "0" : "", {}, lv});
            ops.push_back({'c', mid, dst, m + "3", m + "2", std::move(adds), lv1});
        };
        block("residual_model_head1", "b", "h1a", "X00", 0, {"h0"}, 'c', false);
        block("residual_model_01", "X00", "r01a", "X01", 0, {"X00"}, 'c', true);
        block("downsample_model_10", "X00", "d10a", "d10", 0, {}, 's', true);
        block("residual_model_head2", "c", "h2a", "X10", 1, {"d10"}, 'c', false);
        block("downsample_model_20", "X10", "d20a", "d20", 1, {}, 's', true);
        block("residual_model_head3", "d", "h3a", "X20", 2, {"d20"}, 'c', false);
        block("residual_model_11", "X10", "r11a", "R11", 1, {"X10"}, 'c', true);
        block("downsample_model_11", "X01", "d11a", "X11", 0, {"R11"}, 's', true);
        block("residual_model_21", "X20", "r21a", "R21", 2, {"X20"}, 'c', true);
        block("downsample_model_21", "X11", "d21a", "X21", 1, {"R21"}, 's', true);
        block("residual_model_24", "X21", "r24a", "X24", 2, {"X21"}, 'c', true);
        block("residual_model_25", "X24", "r25a", "X25", 2, {"X24"}, 'c', true);
        block("upsample_model_14", "X24", "u14a", "U14", 2, {}, 'u', true);
        block("residual_model_14", "X11", "r14a", "X14", 1, {"X11", "U14"}, 'c', true);
        block("upsample_model_04", "X14", "u04a", "U04", 1, {}, 'u', true);
        block("residual_model_04", "X01", "r04a", "X04", 0, {"X01", "U04"}, 'c', true);
        block("upsample_model_15", "X25", "u15a", "U15", 2, {}, 'u', true);
        block("residual_model_15", "X14", "r15a", "X15", 1, {"X14", "U15"}, 'c', true);
        block("upsample_model_05", "X15", "u05a", "U05", 1, {}, 'u', true);
        block("residual_model_05", "X04", "r05a", "X05", 0, {"X04", "U05"}, 'c', true);
        const std::string t = "residual_model_tail.";
        ops.push_back({'c', "X05", "t1", t + "conv_before_upsample.0", "", {}, 0});
        ops.push_back({'c', "t1", "out", t + "upsample.0", t + "conv_before_upsample.1", {}, 0});
        const std::map<std::string, int> ext = {
            {"a", kFuRefA}, {"b", kFuRefB}, {"c", kFuRefC}, {"d", kFuRefD}, {"out", kFuRefOut}};
        const std::map<std::string, int> extC = {{"a", 9}, {"b", 128}, {"c", 256}, {"d", 384}};
        std::map<std::string, size_t> last;
        for (size_t i = 0; i < ops.size(); i++)
        {
            last[ops[i].src] = i;
            for (const std::string& a : ops[i].adds)
                last[a] = i;
        }
        FuPlan p;
        std::vector<int> freeSlot[3];
        std::map<std::string, int> code;
        for (const auto& e : ext)
            code[e.first] = e.second;
        for (size_t i = 0; i < ops.size(); i++)
        {
            const Op& o = ops[i];
            const int olv = o.kind == 's' ? o.lv + 1 : o.kind == 'u' ? o.lv - 1 : o.lv;
            if (!ext.count(o.dst))
            {
                int s = 0;
                if (!freeSlot[olv].empty())
                {
                    s = freeSlot[olv].back();
                    freeSlot[olv].pop_back();
                }
                else
                    s = p.slots[olv]++;
                code[o.dst] = kFuRefSlot + olv * kFuMaxSlots + s;
            }
            std::vector<std::string> reads = o.adds;
            reads.push_back(o.src);
            std::sort(reads.begin(), reads.end());
            reads.erase(std::unique(reads.begin(), reads.end()), reads.end());
            for (const std::string& r : reads)
                if (!ext.count(r) && last[r] == i)
                {
                    const int c = code[r] - kFuRefSlot;
                    freeSlot[c / kFuMaxSlots].push_back(c % kFuMaxSlots);
                }
            FuStep st;
            st.kind = o.kind;
            st.src = code[o.src];
            st.dst = code[o.dst];
            st.add0 = o.adds.size() > 0 ? code[o.adds[0]] : -1;
            st.add1 = o.adds.size() > 1 ? code[o.adds[1]] : -1;
            st.lv = o.lv;
            st.cin = extC.count(o.src) ? extC.at(o.src) : o.src == "A16" ? 9 : kFuCh[o.lv];
            st.cout = o.dst == "out" ? 256 : o.kind == 'i' ? 16 : kFuCh[olv];
            st.conv = o.conv;
            st.prelu = o.prelu;
            p.steps.push_back(st);
        }
        return p;
    }();
    return plan;
}

// fusionnet.pkl in the strict GridNet form (every step's conv [cout][cin][3][3] / the transposed ones [cin][cout][4][4]
// with its bias, every PReLU one scalar, conv_last [3][64][3][3] + [3], nothing else) as one weight block of 256-byte
// aligned pieces (ops[i] = step i's kernel, slope and pieces), conv_last in fp16 (wl / bl) and the heads' first PReLU
// slopes on b / c / d (lr); false = missing, or not this net (`why`)
static bool fuNatLoad(std::vector<uint8_t>& blob, std::vector<FuOp>& ops, std::vector<uint16_t>& wl,
                      std::vector<uint16_t>& bl, float* lr, std::string& why)
{
    PthFile pf;
    if (!pthStateDict(g_nativeEngineDir + L"\\GMFSS_Fortuna\\train_log\\fusionnet.pkl", pf, why))
        return false;
    auto& wts = pf.t;
    const FuPlan& pl = fuPlan();
    std::map<std::string, bool> want;
    auto has = [&](const std::string& n, const std::vector<int64_t>& s) {
        want[n] = true;
        return pf.is(n, s);
    };
    // the PReLU slopes and the convs' biases go to the kernels as fp32 (conv_last's are rounded to fp16)
    auto has32 = [&](const std::string& n, const std::vector<int64_t>& s) {
        want[n] = true;
        return pf.is32(n, s);
    };
    bool form = true;
    for (const FuStep& p : pl.steps)
    {
        if (!p.prelu.empty())
            form = form && has32(p.prelu + ".weight", {1});
        if (p.kind == 'i')
            continue;
        const std::vector<int64_t> ws =
            p.kind == 'u' ? std::vector<int64_t>{p.cin, p.cout, 4, 4} : std::vector<int64_t>{p.cout, p.cin, 3, 3};
        form = form && has(p.conv + ".weight", ws) && has32(p.conv + ".bias", {p.cout});
    }
    static const char* const kHeadLr[3] = {"residual_model_head1.0", "residual_model_head2.0",
                                           "residual_model_head3.0"};
    for (const char* h : kHeadLr)
        form = form && has32(std::string(h) + ".weight", {1});
    const std::string cl = "residual_model_tail.conv_last.";
    form = form && has(cl + "weight", {3, 64, 3, 3}) && has(cl + "bias", {3}) && pf.S.size() == want.size();
    if (!form)
    {
        why = pf.notForm("GMFSS's fusionnet");
        return false;
    }
    for (int i = 0; i < 3; i++)
    {
        lr[i] = pf.f32(std::string(kHeadLr[i]) + ".weight")[0];
        if (!(lr[i] > 0.0f && lr[i] <= 1.0f))
        {
            why = "its b / c / d PReLU slopes are not in (0, 1]";
            return false;
        }
    }
    blob.clear();
    ops.assign(pl.steps.size(), FuOp());
    auto put = [&](const void* d, size_t n) {
        const size_t o = (blob.size() + 255) & ~(size_t)255;
        blob.resize(o + n);
        memcpy(blob.data() + o, d, n);
        return o;
    };
    for (size_t i = 0; i < pl.steps.size(); i++)
    {
        const FuStep& p = pl.steps[i];
        FuOp& o = ops[i];
        o.cin = p.cin;
        o.cout = p.cout;
        o.slope = p.prelu.empty() ? 0.0f : pf.f32(p.prelu + ".weight")[0];
        if (p.kind == 'i')
        {
            o.k = kFuIn;
            continue;
        }
        const PthW& w = wts[p.conv + ".weight"];
        std::vector<uint16_t> fr;
        if (p.kind == 'u')
        {
            o.k = kFuUp;
            fr = ifnLastFrags(w, p.cin, p.cout / 4, 64, p.cout);
        }
        else if (p.kind == 's')
        {
            o.k = kFuW2;
            fr = fnFragOf(w, p.cout, p.cin, 32, 8);
        }
        else if (p.cin == 9)
        {
            // head0's first conv: its 9 inputs zero-padded to k_fuIn's 16
            o.k = kFuC16;
            std::vector<uint16_t> wp((size_t)p.cout * 16 * 9, 0);
            for (int n = 0; n < p.cout; n++)
                for (int c = 0; c < 9; c++)
                    for (int t = 0; t < 9; t++)
                        wp[((size_t)n * 16 + c) * 9 + t] = w.half(((size_t)n * 9 + c) * 9 + t);
            fr = fnFragOf(PthW(wp), p.cout, 16, 16, 2);
            o.cin = 16;
        }
        else
        {
            o.k = p.prelu.empty() ? kFuW64 : p.add1 >= 0 ? kFuW64P2 : p.add0 >= 0 ? kFuW64P1 : kFuW64P;
            fr = fnFragOf(w, p.cout, p.cin, 64, 8);
        }
        o.wOff = put(fr.data(), fr.size() * sizeof(uint16_t));
        o.bOff = put(pf.f32(p.conv + ".bias").data(), (size_t)p.cout * sizeof(float));
    }
    const PthW& cw = wts[cl + "weight"];
    const PthW& cb = wts[cl + "bias"];
    wl.resize(cw.n);
    bl.resize(cb.n);
    for (size_t i = 0; i < cw.n; i++)
        wl[i] = cw.half(i);
    for (size_t i = 0; i < cb.n; i++)
        bl[i] = cb.half(i);
    return true;
}

// fusionnet's activations at the half w x h: level l's fuPlan buffers of kFuCh[l] fp16 channels each, 256-byte pieces
struct FuArena
{
    size_t at[3][kFuMaxSlots] = {}, total = 0;
};
static FuArena fuArena(int w, int h)
{
    const FuPlan& pl = fuPlan();
    FuArena a;
    size_t o = 0;
    for (int l = 0; l < 3; l++)
        for (int i = 0; i < pl.slots[l] && i < kFuMaxSlots; i++)
        {
            a.at[l][i] = o;
            o += ((size_t)(w >> l) * (size_t)(h >> l) * kFuCh[l] * sizeof(uint16_t) + 255) & ~(size_t)255;
        }
    a.total = o;
    return a;
}

static size_t fuNatBytes(const NativeRife& nr)
{
    return nr.gmfss && nr.hw > 0 && nr.hh > 0 ? fuArena(nr.hw, nr.hh).total : 0;
}

static bool gaTuring();

// this session's fusionnet module: the weights read, its text compiled (or the cached cubin) and loaded, its kernels
// bound, the weight block and conv_last uploaded, the b / c / d slopes into gmLr; false = `why` (the session teardown
// unloads a module left in fuMod)
static bool nativeFuModule(NativeRife& nr, std::string& why)
{
    if (nr.fuMod)
        return true;
    for (int l = 0; l < 3; l++)
        if (fuPlan().slots[l] > kFuMaxSlots)
        {
            why = "its plan needs more buffers than a level holds";
            return false;
        }
    std::vector<uint8_t> blob;
    std::vector<uint16_t> wl, bl;
    float lr[3] = {};
    if (!fuNatLoad(blob, nr.fuOps, wl, bl, lr, why))
        return false;
    std::string text = kFnHead;
    text += kFuA;
    text += kFuB;
    std::vector<char> cubin;
    if (!nativeCubin(text.c_str(), "smv_fusion.cu", L"smv_fusion", "fusionnet kernels",
                     nr.kernCacheDir.empty() ? std::wstring(L".") : nr.kernCacheDir, cubin))
    {
        why = "its kernels did not compile";
        return false;
    }
    if (cuModuleLoadData(&nr.fuMod, cubin.data()) != CUDA_SUCCESS)
    {
        nr.fuMod = nullptr;
        why = "its kernel module did not load";
        return false;
    }
    for (int i = 0; i < (int)(sizeof(kFuKern) / sizeof(kFuKern[0])); i++)
        if (cuModuleGetFunction(&nr.fFu[i], nr.fuMod, kFuKern[i]) != CUDA_SUCCESS)
        {
            why = std::string("its kernel ") + kFuKern[i] + " is missing";
            return false;
        }
    if (cuFuncSetAttribute(nr.fFu[kFuUp], CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, kFuSmUp) != CUDA_SUCCESS)
    {
        why = "its transposed conv's shared memory was refused";
        return false;
    }
    // a PReLU wide conv below 256 outputs whose 8-row grid holds 8+ CTAs an SM runs as its 4-row twin (the same bytes,
    // three CTAs an SM where the 8-row halo allows two: 1.01x to 1.20x a launch on 82 SMs; a tie or slower on smaller
    // grids, at 256 outputs and on the head convs without PReLU); never on RTX 20 (unmeasured); SMV_FU_R4=0 = never
    {
        int dev = 0, sms = 0;
        nr.fuR4Min = !gaTuring() && lkEnv("SMV_FU_R4") != "0" && cudaGetDevice(&dev) == cudaSuccess &&
                             cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess
                         ? 8 * sms
                         : 0;
    }
    if (cudaMalloc(&nr.dFuW, blob.size()) != cudaSuccess ||
        cudaMemcpy(nr.dFuW, blob.data(), blob.size(), cudaMemcpyHostToDevice) != cudaSuccess ||
        (!nr.dGmWl && cudaMalloc(&nr.dGmWl, wl.size() * sizeof(uint16_t)) != cudaSuccess) ||
        (!nr.dGmBl && cudaMalloc(&nr.dGmBl, bl.size() * sizeof(uint16_t)) != cudaSuccess) ||
        cudaMemcpy(nr.dGmWl, wl.data(), wl.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(nr.dGmBl, bl.data(), bl.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        why = "its weights did not upload";
        return false;
    }
    memcpy(nr.gmLr, lr, sizeof(lr));
    return true;
}

// fusionnet's 45 launches on st at the half nr.hw x nr.hh, its activations at s (fuArena's pieces). false = a launch
// failed
static bool nativeFuLaunch(NativeRife& nr, uint8_t* s, cudaStream_t st)
{
    const FuArena ar = fuArena(nr.hw, nr.hh);
    const FuPlan& pl = fuPlan();
    const char* wb = (const char*)nr.dFuW;
    auto buf = [&](int code) -> void* {
        switch (code)
        {
        case -1:
            return nullptr;
        case kFuRefA:
            return nr.dGmFa;
        case kFuRefB:
            return nr.dGmFb;
        case kFuRefC:
            return nr.dGmFc;
        case kFuRefD:
            return nr.dGmFd;
        case kFuRefOut:
            return nr.dGmAcc;
        default:
            return s + ar.at[(code - kFuRefSlot) / kFuMaxSlots][(code - kFuRefSlot) % kFuMaxSlots];
        }
    };
    auto go = [&](int f, int gx, int gy, int gz, int threads, int sm, void** a) {
        return cuLaunchKernel(nr.fFu[f], (unsigned)gx, (unsigned)gy, (unsigned)gz, (unsigned)threads, 1, 1,
                              (unsigned)sm, (CUstream)st, a, nullptr) == CUDA_SUCCESS;
    };
    for (size_t i = 0; i < pl.steps.size(); i++)
    {
        const FuStep& p = pl.steps[i];
        const FuOp& o = nr.fuOps[i];
        int iw = nr.hw >> p.lv, ih = nr.hh >> p.lv, cin = o.cin, cout = o.cout;
        float sl = o.slope;
        void* x = buf(p.src);
        void* y = buf(p.dst);
        void* a0 = buf(p.add0);
        void* a1 = buf(p.add1);
        const void* wf = wb + o.wOff;
        const void* bias = wb + o.bOff;
        bool ok = false;
        if (o.k == kFuIn)
        {
            void* a[] = {&x, &iw, &ih, &sl, &y};
            ok = go(kFuIn, (iw + 31) / 32, (ih + 7) / 8, 1, 256, 0, a);
        }
        else if (o.k == kFuUp)
        {
            void* a[] = {&x, &cin, &ih, &iw, &y, &cout, (void*)&wf, (void*)&bias, &sl};
            ok = go(kFuUp, (iw + 31) / 32, (ih + 3) / 4, cout / 16, 128, kFuSmUp, a);
        }
        else if (o.k == kFuC16)
        {
            void* a[] = {&x, &ih, &iw, &y, &cout, (void*)&wf, (void*)&bias};
            ok = go(kFuC16, (iw + 31) / 32, (ih + 3) / 4, cout / 16, 128, kFuSmC16, a);
        }
        else
        {
            const bool s2 = o.k == kFuW2;
            const int ow = s2 ? iw / 2 : iw, oh = s2 ? ih / 2 : ih;
            const bool r4 = (o.k == kFuW64P || o.k == kFuW64P1 || o.k == kFuW64P2) && cout < 256 && nr.fuR4Min > 0 &&
                            ((ow + 31) / 32) * ((oh + 7) / 8) * (cout / 64) >= nr.fuR4Min;
            void* a[] = {&x, &cin, &ih, &iw, &y, &cout, (void*)&wf, (void*)&bias, &sl, &a0, &a1};
            if (r4)
                ok = go(kFuW64PR4 + (o.k - kFuW64P), (ow + 31) / 32, (oh + 3) / 4, cout / 64, 128, kFuSmW64R4, a);
            else
                ok = go(o.k, (ow + 31) / 32, s2 ? (oh + 3) / 4 : (oh + 7) / 8, cout / 64, 128, s2 ? kFuSmW2 : kFuSmW64,
                        a);
        }
        if (!ok)
            return false;
    }
    return true;
}

// fusionnet on one tween on st (nr.stream): dGmFa / dGmFb / dGmFc / dGmFd in, its out into dGmAcc; the launches go
// through nr.fuGraphs (chainGraphRun). false = the activations or a launch failed
static bool nativeFuRun(NativeRife& nr, cudaStream_t st)
{
    const FuArena ar = fuArena(nr.hw, nr.hh);
    uint8_t* s = nr.dGmCtxMem && (size_t)nr.gmCtxMemB >= ar.total ? (uint8_t*)nr.dGmCtxMem : (uint8_t*)nr.dFuOwn;
    if (!s && cudaMalloc(&nr.dFuOwn, ar.total) == cudaSuccess)
        s = (uint8_t*)nr.dFuOwn;
    if (!s)
        return false;
    return chainGraphRun(nr.fuGraphs, {s, nr.dGmFa, nr.dGmFb, nr.dGmFc, nr.dGmFd, nr.dGmAcc}, nr.hw, nr.hh, st,
                         "gmfss fusionnet", [&] { return nativeFuLaunch(nr, s, st); });
}
