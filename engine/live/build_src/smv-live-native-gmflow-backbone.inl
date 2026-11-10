// part of smv-live-native.inl (included there, in place); not a standalone translation unit
// gmflow's CNN backbone (backbone.CNNEncoder, two output scales) as 54 launches (nativeBbRun) of its own module
// smv_gmflowbb_*.cubin (kFnHead's helpers + kBbA / kBbB), weights = engine\GMFSS_Fortuna\train_log\flownet.pkl's
// backbone.* through pthStateDict (bbNatLoad); no fallback (a missing / wrong pkl = `gmflow's
// backbone unavailable (...)`, the session refuses). One frame a call: the half's 3 fp32 planes (0..1) in, map4 (the
// quarter) and map8 (the eighth) out, planar fp16 [128][h][w] into A's map batch slot. k_bbC7: normalize_img + conv1
// 7x7 s2 (fp32); every InstanceNorm = its statistics (k_bbStats: per-block Welford, k_bbStatsFin: a fixed-order merge,
// bit-stable) applied where the next pass reads it (the next 3x3 conv's halo, or k_bbRes*); the residual blocks' 3x3
// convs with fp16 sums (fusionnet's wide form), the downsamples and conv2 as 1 x 1 GEMMs (fp32 sums), the trident pair
// (one weight at strides 1 and 2). Its activations (bbArena) live in the shared activation block (sound while serial
// on nr.stream), else dBbOwn.
static const char kBbA[] = R"BBSRC(
// gmflow's CNN backbone on the host's kernels, after kFnHead: one frame at the GMFSS
// half, every activation channels-last fp16 [H][W][C]; a conv writes its raw output and the InstanceNorm after it is
// statistics (k_bbStats + k_bbStatsFin: per-block Welford partials merged in a fixed order, so every run is bit-stable)
// applied where the next pass reads it (the next conv's halo, or k_bbRes).

// the stored index of halo pixel sp and its chunk k's swizzled slot (each ldmatrix's 8 rows in 8 bank groups)
template <int NCH>
__device__ __forceinline__ unsigned bbSlot(int sp, int k)
{
    const int sw = NCH == 8 ? (k ^ (sp & 7)) : NCH == 4 ? (k ^ ((sp >> 1) & 3)) : (k ^ ((sp >> 2) & 1));
    return (unsigned)((sp * NCH + sw) << 4);
}

__device__ __forceinline__ void bbMma(float* d, const unsigned* a, unsigned b0, unsigned b1)
{
    asm volatile(SMV_MMA_F32 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void bbMmaH(unsigned* d, const unsigned* a, unsigned b0, unsigned b1)
{
    asm volatile(SMV_MMA_F16 "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(d[0]), "+r"(d[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// relu((v - m) inv) on two packed fp16 values in fp32, one rounding each
__device__ __forceinline__ unsigned bbNorm2(unsigned u, float m0, float i0, float m1, float i1)
{
    const float a = fmaxf((h2f((unsigned short)(u & 0xffffu)) - m0) * i0, 0.0f);
    const float b = fmaxf((h2f((unsigned short)(u >> 16)) - m1) * i1, 0.0f);
    return (unsigned)f2h(a) | ((unsigned)f2h(b) << 16);
}

// conv1: 7 x 7 stride 2 pad 3, 3 -> 64, no bias, on the frame's 3 fp32 planes [3][H][W] (0..1) normalized as
// normalize_img does ((x - mean) / std, in fp32, the padding zeros AFTER it), fp32 sums; out = the raw output
// channels-last fp16 [H / 2][W / 2][64]; wt = [64][3][7][7] fp32. A 128-thread block = a 4 x 32 output tile, 32 outputs
// a pass (two passes), its input tile (14 x 70 x 3) and that pass's weights in shared memory, tap-major (tw[t][n]: a
// tap's 32 weights in 8 float4 reads)
extern "C" __global__ void __launch_bounds__(128) k_bbC7(const float* __restrict__ img, int H, int W,
                                                          const float* __restrict__ wt, unsigned short* __restrict__ out)
{
    __shared__ float tin[3][14][70];
    __shared__ __align__(16) float tw[147][32];
    const int OW = W / 2, OH = H / 2;
    const int tx = threadIdx.x & 31, ty = threadIdx.x >> 5;
    const int ox0 = blockIdx.x * 32, oy0 = blockIdx.y * 4;
    const float mean[3] = {0.485f, 0.456f, 0.406f}, sd[3] = {0.229f, 0.224f, 0.225f};
    for (int i = threadIdx.x; i < 3 * 14 * 70; i += 128)
    {
        const int c = i / (14 * 70), r = (i / 70) % 14, q = i % 70;
        const int gy = 2 * oy0 - 3 + r, gx = 2 * ox0 - 3 + q;
        tin[c][r][q] = gy >= 0 && gy < H && gx >= 0 && gx < W ? (img[((size_t)c * H + gy) * W + gx] - mean[c]) / sd[c]
                                                              : 0.0f;
    }
    const int oy = oy0 + ty, ox = ox0 + tx;
    for (int half = 0; half < 2; ++half)
    {
        __syncthreads();
        for (int i = threadIdx.x; i < 32 * 147; i += 128)
            tw[i % 147][i / 147] = wt[(size_t)(half * 32 + i / 147) * 147 + i % 147];
        __syncthreads();
        float acc[32];
#pragma unroll
        for (int n = 0; n < 32; ++n)
            acc[n] = 0.0f;
        for (int c = 0; c < 3; ++c)
            for (int ky = 0; ky < 7; ++ky)
#pragma unroll
                for (int kx = 0; kx < 7; ++kx)
                {
                    const float v = tin[c][2 * ty + ky][2 * tx + kx];
                    const float4* wv = reinterpret_cast<const float4*>(tw[(c * 7 + ky) * 7 + kx]);
#pragma unroll
                    for (int n4 = 0; n4 < 8; ++n4)
                    {
                        const float4 q = wv[n4];
                        acc[4 * n4] = fmaf(v, q.x, acc[4 * n4]);
                        acc[4 * n4 + 1] = fmaf(v, q.y, acc[4 * n4 + 1]);
                        acc[4 * n4 + 2] = fmaf(v, q.z, acc[4 * n4 + 2]);
                        acc[4 * n4 + 3] = fmaf(v, q.w, acc[4 * n4 + 3]);
                    }
                }
        if (oy < OH && ox < OW)
        {
            unsigned short* o = out + ((size_t)oy * OW + ox) * 64 + half * 32;
#pragma unroll
            for (int n = 0; n < 32; n += 2)
                *reinterpret_cast<unsigned*>(o + n) = (unsigned)f2h(acc[n]) | ((unsigned)f2h(acc[n + 1]) << 16);
        }
    }
}

// InstanceNorm statistics, pass 1: y = [P][C] fp16 channels-last; block b (256 threads = 256 / C pixel lanes x C
// channels, C in 64 / 96 / 128 -> lanes 4 / 2 / 2 with 192 / 256 threads used) takes pixels [b PB, (b + 1) PB); each
// thread runs Welford over its pixels, the lanes merge in shared memory; part[b][c] = (n, mean, M2)
extern "C" __global__ void __launch_bounds__(256) k_bbStats(const unsigned short* __restrict__ y, int P, int C, int PB,
                                                             float* __restrict__ part)
{
    __shared__ float sn[256], sm[256], s2[256];
    const int lanes = C == 64 ? 4 : 2, c = threadIdx.x % C, l = threadIdx.x / C;
    const bool act = l < lanes;
    const int p0 = blockIdx.x * PB, p1 = min(P, p0 + PB);
    float n = 0.0f, mean = 0.0f, m2 = 0.0f;
    if (act)
        for (int p = p0 + l; p < p1; p += lanes)
        {
            const float v = h2f(y[(size_t)p * C + c]);
            n += 1.0f;
            const float d = v - mean;
            mean += d / n;
            m2 = fmaf(d, v - mean, m2);
        }
    sn[threadIdx.x] = n;
    sm[threadIdx.x] = mean;
    s2[threadIdx.x] = m2;
    __syncthreads();
    if (l == 0)
    {
        for (int k = 1; k < lanes; ++k)
        {
            const int j = k * C + c;
            const float nb = sn[j];
            if (nb == 0.0f)
                continue;
            const float nn = n + nb, d = sm[j] - mean;
            mean += d * (nb / nn);
            m2 += s2[j] + d * d * (n * nb / nn);
            n = nn;
        }
        float* o = part + ((size_t)blockIdx.x * C + c) * 3;
        o[0] = n;
        o[1] = mean;
        o[2] = m2;
    }
}

// Chan's merge of (nb, mb, qb) into (n, m, q)
__device__ __forceinline__ void bbMerge(float& n, float& m, float& q, float nb, float mb, float qb)
{
    if (nb == 0.0f)
        return;
    const float nn = n + nb, d = mb - m;
    m += d * (nb / nn);
    q += qb + d * d * (n * nb / nn);
    n = nn;
}

// pass 2: block c (128 threads) merges channel c's nb partials: thread t takes b = t, t + 128, ... in order, then a
// fixed pairwise tree over the threads (the same order every run); st[c] = mean, st[C + c] = 1 / sqrt(var + 1e-5) (the
// biased variance, InstanceNorm's)
extern "C" __global__ void __launch_bounds__(128) k_bbStatsFin(const float* __restrict__ part, int nb, int C,
                                                                float* __restrict__ st)
{
    __shared__ float sn[128], sm[128], sq[128];
    const int c = blockIdx.x;
    float n = 0.0f, mean = 0.0f, m2 = 0.0f;
    for (int b = threadIdx.x; b < nb; b += 128)
    {
        const float* q = part + ((size_t)b * C + c) * 3;
        bbMerge(n, mean, m2, q[0], q[1], q[2]);
    }
    sn[threadIdx.x] = n;
    sm[threadIdx.x] = mean;
    sq[threadIdx.x] = m2;
    __syncthreads();
    for (int s = 64; s; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            float a = sn[threadIdx.x], b = sm[threadIdx.x], d = sq[threadIdx.x];
            bbMerge(a, b, d, sn[threadIdx.x + s], sm[threadIdx.x + s], sq[threadIdx.x + s]);
            sn[threadIdx.x] = a;
            sm[threadIdx.x] = b;
            sq[threadIdx.x] = d;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        st[c] = sm[0];
        st[C + c] = 1.0f / sqrtf(sq[0] / sn[0] + 1e-5f);
    }
}

// the residual block's end: out = relu(skip + relu(IN(y))), y raw with st (mean | inv); skip by MODE: 0 = x as it is,
// 1 = relu(IN(x)) with sx (the first block of layer1: its input = conv1's raw output), 2 = IN(x) with sx (a downsample's
// raw output); 8 channels a thread, P pixels x C channels
template <int MODE>
__device__ __forceinline__ void bbResT(const unsigned short* __restrict__ y, const float* __restrict__ st,
                                       const unsigned short* __restrict__ x, const float* __restrict__ sx, int P, int C,
                                       unsigned short* __restrict__ out)
{
    const int i = blockIdx.x * 256 + threadIdx.x;
    const int nch = C / 8;
    if (i >= P * nch)
        return;
    const int c0 = (i % nch) * 8;
    const uint4 yv = *reinterpret_cast<const uint4*>(y + (size_t)i * 8);
    const uint4 xv = *reinterpret_cast<const uint4*>(x + (size_t)i * 8);
    const unsigned yu[4] = {yv.x, yv.y, yv.z, yv.w}, xu[4] = {xv.x, xv.y, xv.z, xv.w};
    unsigned o[4];
#pragma unroll
    for (int k = 0; k < 4; ++k)
    {
        float r[2];
#pragma unroll
        for (int e = 0; e < 2; ++e)
        {
            const int c = c0 + 2 * k + e;
            const float yy = h2f((unsigned short)(e ? yu[k] >> 16 : yu[k] & 0xffffu));
            float xx = h2f((unsigned short)(e ? xu[k] >> 16 : xu[k] & 0xffffu));
            if (MODE == 1)
                xx = fmaxf((xx - sx[c]) * sx[C + c], 0.0f);
            else if (MODE == 2)
                xx = (xx - sx[c]) * sx[C + c];
            r[e] = fmaxf(xx + fmaxf((yy - st[c]) * st[C + c], 0.0f), 0.0f);
        }
        o[k] = (unsigned)f2h(r[0]) | ((unsigned)f2h(r[1]) << 16);
    }
    *reinterpret_cast<uint4*>(out + (size_t)i * 8) = make_uint4(o[0], o[1], o[2], o[3]);
}
#define BB_RES(NAME, MODE)                                                                                             \
    extern "C" __global__ void __launch_bounds__(256)                                                                  \
        NAME(const unsigned short* __restrict__ y, const float* __restrict__ st, const unsigned short* __restrict__ x, \
             const float* __restrict__ sx, int P, int C, unsigned short* __restrict__ out)                             \
    {                                                                                                                  \
        bbResT<MODE>(y, st, x, sx, P, C, out);                                                                         \
    }
BB_RES(k_bbRes0, 0)
BB_RES(k_bbRes1, 1)
BB_RES(k_bbRes2, 2)

)BBSRC";
static const char kBbB[] = R"BBSRC(
// the 3 x 3 conv at stride S (fusionnet's wide form): x = cin channels-last fp16 at W x H; y = cout fp16 at W / S x
// H / S, channels-last or PLANAR [cout][H / S][W / S]; wf[z][step][q][lane] (step = (group 9 + tap) CG / 16 + kc);
// bias fp32 (zeros for a conv without one); NORM: the halo through relu(IN) with st = mean | inv over cin (cells off
// the frame stay the zero padding); A32: fp32 accumulators, else fp16
template <int S, int CG, int NB, bool NORM, bool A32, bool PLANAR>
__device__ __forceinline__ void bbConvW(const unsigned short* __restrict__ x, int cin, int H, int W,
                                        unsigned short* __restrict__ y, int cout, const uint4* __restrict__ wf,
                                        const float* __restrict__ bias, const float* __restrict__ st)
{
    constexpr int TH = S == 1 ? 8 : 4, RW = S == 1 ? 2 : 1, MI = 2 * RW;
    constexpr int HR = S == 1 ? TH + 2 : 2 * TH + 1, HC = S == 1 ? 34 : 65;
    const int OW = W / S, OH = H / S;
    constexpr int NCH = CG / 8, KC = CG / 16, NQ = NB / 2;
    extern __shared__ __align__(128) unsigned char bbSm[];
    const unsigned halo = (unsigned)__cvta_generic_to_shared(bbSm);
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
            rnCp16(halo + bbSlot<NCH>(sp, k), src, in ? 16 : 0);
        }
        asm volatile(SMV_CP_COMMIT);
        asm volatile(SMV_CP_WAIT0);
        __syncthreads();
        if (NORM)
        {
            for (int i = threadIdx.x; i < HR * HC * NCH; i += 128)
            {
                const int p = i / NCH, k = i - p * NCH;
                const int hr = p / HC, hc = p - hr * HC;
                const int gy = S * ty0 - 1 + hr, gx = S * tx0 - 1 + hc;
                if (gy < 0 || gy >= H || gx < 0 || gx >= W)
                    continue;
                const int sp = S == 1 ? p : hr * HC + ((hc & 1) ? 33 + (hc >> 1) : (hc >> 1));
                uint4* q = reinterpret_cast<uint4*>(bbSm + bbSlot<NCH>(sp, k));
                const int c = gi * CG + k * 8;
                uint4 v = *q;
                v.x = bbNorm2(v.x, st[c], st[cin + c], st[c + 1], st[cin + c + 1]);
                v.y = bbNorm2(v.y, st[c + 2], st[cin + c + 2], st[c + 3], st[cin + c + 3]);
                v.z = bbNorm2(v.z, st[c + 4], st[cin + c + 4], st[c + 5], st[cin + c + 5]);
                v.w = bbNorm2(v.w, st[c + 6], st[cin + c + 6], st[c + 7], st[cin + c + 7]);
                *q = v;
            }
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
                    rnLdm4(halo + bbSlot<NCH>(pa[mi], kc * 2 + akc), a);
#pragma unroll
                    for (int q = 0; q < NB; ++q)
                    {
                        const unsigned b0 = (q & 1) ? bc[q >> 1].z : bc[q >> 1].x;
                        const unsigned b1 = (q & 1) ? bc[q >> 1].w : bc[q >> 1].y;
                        if constexpr (A32)
                            bbMma(accF[mi][q], a, b0, b1);
                        else
                            bbMmaH(accH[mi][q], a, b0, b1);
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
                if (PLANAR)
                {
                    const size_t pl = (size_t)OH * OW, at = (size_t)oy * OW + ox;
                    y[(size_t)n0 * pl + at] = f2h(v0);
                    y[(size_t)(n0 + 1) * pl + at] = f2h(v1);
                }
                else
                    *reinterpret_cast<unsigned*>(y + ((size_t)oy * OW + ox) * cout + n0) =
                        (unsigned)f2h(v0) | ((unsigned)f2h(v1) << 16);
            }
        }
    }
}
#define BB_CONV(NAME, S, CG, NB, NORM, A32, PLANAR)                                                                    \
    extern "C" __global__ void __launch_bounds__(128)                                                                  \
        NAME(const unsigned short* __restrict__ x, int cin, int H, int W, unsigned short* __restrict__ y, int cout,    \
             const uint4* __restrict__ wf, const float* __restrict__ bias, const float* __restrict__ st)               \
    {                                                                                                                  \
        bbConvW<S, CG, NB, NORM, A32, PLANAR>(x, cin, H, W, y, cout, wf, bias, st);                                    \
    }
// layer1 (64), layer2's first conv (s2 64 -> 96) and its other convs (96), layer3's first (96 -> 128) and others
// (128), the trident (128, planar out); all with fp16 sums; N = the halo normalized on load
BB_CONV(k_bbC64H, 1, 64, 8, false, false, false)
BB_CONV(k_bbC64NH, 1, 64, 8, true, false, false)
BB_CONV(k_bbS64to96H, 2, 32, 4, false, false, false)
BB_CONV(k_bbC96H, 1, 32, 4, false, false, false)
BB_CONV(k_bbC96NH, 1, 32, 4, true, false, false)
BB_CONV(k_bbC96to128H, 1, 32, 8, false, false, false)
BB_CONV(k_bbTriH, 1, 64, 8, false, false, true)
BB_CONV(k_bbTri2H, 2, 32, 8, false, false, true)

// the 1 x 1 convs (the downsamples, conv2) on the tensor cores: Y[M][N] = A W^T + bias, fp32 sums, fp16 out; output
// pixel m = (oy, ox) of OW reads A's pixel (S oy, S ox) of W = S OW: K channels a row, K a multiple of 16 up to 128;
// 64 rows a block staged swizzled (a row = K / 8 chunks padded to a multiple of 8: 64 x 16 x 16 bytes of shared
// memory at most), NT n-tiles a block (blockIdx.y), W in fragment order wf[z][K / 16][NT / 2][lane]
template <int NT, int S>
__device__ __forceinline__ void bbGemmT(const unsigned short* __restrict__ A, int M, int OW, int K,
                                        const uint4* __restrict__ wf, const float* __restrict__ bias,
                                        unsigned short* __restrict__ Y, int N)
{
    extern __shared__ __align__(128) unsigned char bbSm[];
    const unsigned sm = (unsigned)__cvta_generic_to_shared(bbSm);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int m0 = blockIdx.x * 64, z = blockIdx.y;
    const int nch = K / 8, nrow = (nch + 7) & ~7;
    for (int i = threadIdx.x; i < 64 * nch; i += 128)
    {
        const int r = i / nch, k = i - r * nch, m = m0 + r;
        const bool in = m < M;
        const int oy = in ? m / OW : 0, ox = in ? m - oy * OW : 0;
        const size_t src = ((size_t)(S * oy) * (S * OW) + S * ox) * K + k * 8;
        rnCp16(sm + (unsigned)((r * nrow + (k ^ (r & 7))) << 4), in ? A + src : A, in ? 16 : 0);
    }
    asm volatile(SMV_CP_COMMIT);
    asm volatile(SMV_CP_WAIT0);
    __syncthreads();
    const int g = lane >> 2, t4 = lane & 3;
    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j)
    {
        const int n0 = z * NT * 8 + j * 8 + t4 * 2;
        acc[j][0] = acc[j][2] = bias[n0];
        acc[j][1] = acc[j][3] = bias[n0 + 1];
    }
    const int arow = warp * 16 + (lane & 7) + ((lane >> 3) & 1) * 8, akc = lane >> 4;
    const int ks = K / 16;
    wf += (size_t)z * ks * (NT / 2) * 32;
    for (int s = 0; s < ks; ++s)
    {
        unsigned a[4];
        const int kc = 2 * s + akc;
        rnLdm4(sm + (unsigned)((arow * nrow + (kc ^ (arow & 7))) << 4), a);
#pragma unroll
        for (int q = 0; q < NT / 2; ++q)
        {
            const uint4 bw = wf[((size_t)s * (NT / 2) + q) * 32 + lane];
            bbMma(acc[2 * q], a, bw.x, bw.y);
            bbMma(acc[2 * q + 1], a, bw.z, bw.w);
        }
    }
#pragma unroll
    for (int j = 0; j < NT; ++j)
    {
        const int n0 = z * NT * 8 + j * 8 + t4 * 2;
#pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            const int r = m0 + warp * 16 + g + h * 8;
            if (r < M)
                *reinterpret_cast<unsigned*>(Y + (size_t)r * N + n0) =
                    (unsigned)f2h(acc[j][2 * h]) | ((unsigned)f2h(acc[j][2 * h + 1]) << 16);
        }
    }
}
#define BB_GEMM(NAME, NT, S)                                                                                           \
    extern "C" __global__ void __launch_bounds__(128)                                                                  \
        NAME(const unsigned short* __restrict__ A, int M, int OW, int K, const uint4* __restrict__ wf,                 \
             const float* __restrict__ bias, unsigned short* __restrict__ Y, int N)                                    \
    {                                                                                                                  \
        bbGemmT<NT, S>(A, M, OW, K, wf, bias, Y, N);                                                                   \
    }
// layer2's downsample (s2, 64 -> 96: 2 z of 48), layer3's (96 -> 128) and conv2 (128 -> 128): 2 z of 64
BB_GEMM(k_bbDs2, 6, 2)
BB_GEMM(k_bbG1, 8, 1)
)BBSRC";
static const char* const kBbKern[] = {"k_bbC7",   "k_bbStats", "k_bbStatsFin", "k_bbRes0", "k_bbRes1",  "k_bbRes2",
                                      "k_bbC64H", "k_bbC64NH", "k_bbS64to96H", "k_bbC96H", "k_bbC96NH", "k_bbC96to128H",
                                      "k_bbTriH", "k_bbTri2H", "k_bbDs2",      "k_bbG1"};
static_assert(sizeof(kBbKern) / sizeof(kBbKern[0]) == sizeof(NativeRife::fBb) / sizeof(CUfunction),
              "one fBb slot a backbone kernel");
enum
{
    kBbC7,
    kBbStats,
    kBbFin,
    kBbRes0,
    kBbRes1,
    kBbRes2,
    kBbC64,
    kBbC64N,
    kBbS64to96,
    kBbC96,
    kBbC96N,
    kBbC96to128,
    kBbTri,
    kBbTri2,
    kBbDs2,
    kBbG1
};
// bbOff's pieces: conv1's fp32 weights, the 12 residual 3x3 convs (layer l, block b, conv c at 1 + 4 (l - 1) + 2 b + c -
// 1), the downsamples' and conv2's fragments + biases, the trident's two fragment orders, a zero bias
enum
{
    kBbW7,
    kBbW3 = 1,
    kBbWds2 = 13,
    kBbBds2,
    kBbWds3,
    kBbBds3,
    kBbWc2,
    kBbBc2,
    kBbWtri,
    kBbWtri2,
    kBbZero
};
// each residual 3x3 conv's (cin, cout, group, n-tiles) in the 12-conv order
static const int kBbConvShape[12][4] = {{64, 64, 64, 8},  {64, 64, 64, 8},   {64, 64, 64, 8},   {64, 64, 64, 8},
                                        {64, 96, 32, 4},  {96, 96, 32, 4},   {96, 96, 32, 4},   {96, 96, 32, 4},
                                        {96, 128, 32, 8}, {128, 128, 64, 8}, {128, 128, 64, 8}, {128, 128, 64, 8}};

// the statistics' pixels a block (about 256 blocks or more, 32 to 512, a multiple of 32: the kit's pb_of)
static int bbPb(int p)
{
    return (std::max)(32, (std::min)(512, (p / 256) / 32 * 32));
}

// flownet.pkl's backbone in its strict form (conv1 [64][3][7][7], layer1..3 x 2 ResidualBlocks' conv1 / conv2 (no bias),
// layer2.0 / layer3.0's downsample.0 [cout][cin][1][1] + bias, conv2 [128][128][1][1] + bias, trident_conv [128][128][3][3]
// (no bias), every InstanceNorm without parameters) as one weight block of 256-byte aligned pieces (off = bbOff, see
// kBbW7 ..); false = missing, or not this net (`why`)
static bool bbNatLoad(std::vector<uint8_t>& blob, size_t* off, std::string& why)
{
    PthFile pf;
    if (!pthStateDict(g_nativeEngineDir + L"\\GMFSS_Fortuna\\train_log\\flownet.pkl", pf, why))
        return false;
    auto& wts = pf.t;
    const auto& S = pf.S;
    auto has = [&](const std::string& n, const std::vector<int64_t>& s) {
        const auto it = S.find(n);
        return it != S.end() && it->second == s;
    };
    const std::string p = "backbone.";
    bool form =
        has(p + "conv1.weight", {64, 3, 7, 7}) && has(p + "conv2.weight", {128, 128, 1, 1}) &&
        has(p + "conv2.bias", {128}) && has(p + "trident_conv.weight", {128, 128, 3, 3}) &&
        has(p + "layer2.0.downsample.0.weight", {96, 64, 1, 1}) && has(p + "layer2.0.downsample.0.bias", {96}) &&
        has(p + "layer3.0.downsample.0.weight", {128, 96, 1, 1}) && has(p + "layer3.0.downsample.0.bias", {128}) &&
        !S.count(p + "layer1.0.downsample.0.weight") && !S.count(p + "norm1.weight");
    for (int i = 0; i < 12 && form; i++)
    {
        const std::string c = p + "layer" + std::to_string(1 + i / 4) + "." + std::to_string((i / 2) % 2) + ".conv" +
                              std::to_string(1 + i % 2) + ".weight";
        form = has(c, {kBbConvShape[i][1], kBbConvShape[i][0], 3, 3});
    }
    if (!form)
    {
        why = "its weights are not gmflow's backbone";
        return false;
    }
    blob.clear();
    auto put = [&](const void* d, size_t n) {
        const size_t o = (blob.size() + 255) & ~(size_t)255;
        blob.resize(o + n);
        memcpy(blob.data() + o, d, n);
        return o;
    };
    off[kBbW7] = put(wts[p + "conv1.weight"].data(), (size_t)64 * 147 * sizeof(float));
    for (int i = 0; i < 12; i++)
    {
        const std::string c = p + "layer" + std::to_string(1 + i / 4) + "." + std::to_string((i / 2) % 2) + ".conv" +
                              std::to_string(1 + i % 2) + ".weight";
        const int* sh = kBbConvShape[i];
        const std::vector<uint16_t> f = fnFragOf(wts[c], sh[1], sh[0], sh[2], sh[3]);
        off[kBbW3 + i] = put(f.data(), f.size() * sizeof(uint16_t));
    }
    const std::vector<uint16_t> d2 = gbFrag1(wts[p + "layer2.0.downsample.0.weight"], 96, 64, 6);
    off[kBbWds2] = put(d2.data(), d2.size() * sizeof(uint16_t));
    off[kBbBds2] = put(wts[p + "layer2.0.downsample.0.bias"].data(), (size_t)96 * sizeof(float));
    const std::vector<uint16_t> d3 = gbFrag1(wts[p + "layer3.0.downsample.0.weight"], 128, 96, 8);
    off[kBbWds3] = put(d3.data(), d3.size() * sizeof(uint16_t));
    off[kBbBds3] = put(wts[p + "layer3.0.downsample.0.bias"].data(), (size_t)128 * sizeof(float));
    const std::vector<uint16_t> c2 = gbFrag1(wts[p + "conv2.weight"], 128, 128, 8);
    off[kBbWc2] = put(c2.data(), c2.size() * sizeof(uint16_t));
    off[kBbBc2] = put(wts[p + "conv2.bias"].data(), (size_t)128 * sizeof(float));
    const std::vector<uint16_t> t1 = fnFragOf(wts[p + "trident_conv.weight"], 128, 128, 64, 8);
    off[kBbWtri] = put(t1.data(), t1.size() * sizeof(uint16_t));
    const std::vector<uint16_t> t2 = fnFragOf(wts[p + "trident_conv.weight"], 128, 128, 32, 8);
    off[kBbWtri2] = put(t2.data(), t2.size() * sizeof(uint16_t));
    const std::vector<float> z(128, 0.0f);
    off[kBbZero] = put(z.data(), z.size() * sizeof(float));
    return true;
}

// the backbone's activations at the half w x h (P1 = the half's half, P2 = its quarter), 256-byte pieces: conv1's raw
// output and layer1's four buffers ([P1][64] fp16), layer2's five ([P2][96]), layer3's five + conv2's ([P2][128]), the
// statistics' partials and the four mean | inv sets
struct BbArena
{
    size_t l1[5] = {}, l2[5] = {}, l3[6] = {}, part = 0, st[4] = {}, total = 0;
};
static BbArena bbArena(int w, int h)
{
    BbArena a;
    const size_t p1 = (size_t)(w / 2) * (h / 2), p2 = (size_t)(w / 4) * (h / 4);
    size_t o = 0;
    auto put = [&](size_t n) {
        const size_t at = o;
        o += (n + 255) & ~(size_t)255;
        return at;
    };
    for (size_t& x : a.l1)
        x = put(p1 * 64 * sizeof(uint16_t));
    for (size_t& x : a.l2)
        x = put(p2 * 96 * sizeof(uint16_t));
    for (size_t& x : a.l3)
        x = put(p2 * 128 * sizeof(uint16_t));
    a.part = put(((p1 + 31) / 32) * 128 * 3 * sizeof(float));
    for (size_t& x : a.st)
        x = put(256 * sizeof(float));
    a.total = o;
    return a;
}

static size_t bbNatBytes(const NativeRife& nr)
{
    return nr.gmfss && nr.hw > 0 && nr.hh > 0 ? bbArena(nr.hw, nr.hh).total : 0;
}

// this session's backbone module: the weights read, its text compiled (or the cached cubin) and loaded, its kernels
// bound, the weight block uploaded; false = `why` (the session teardown unloads a module left in bbMod)
static bool nativeBbModule(NativeRife& nr, std::string& why)
{
    if (nr.bbMod)
        return true;
    std::vector<uint8_t> blob;
    if (!bbNatLoad(blob, nr.bbOff, why))
        return false;
    std::string text = kFnHead;
    text += kBbA;
    text += kBbB;
    std::vector<char> cubin;
    if (!nativeCubin(text.c_str(), "smv_gmflowbb.cu", L"smv_gmflowbb", "gmflow backbone kernels",
                     nr.kernCacheDir.empty() ? std::wstring(L".") : nr.kernCacheDir, cubin))
    {
        why = "its kernels did not compile";
        return false;
    }
    if (cuModuleLoadData(&nr.bbMod, cubin.data()) != CUDA_SUCCESS)
    {
        nr.bbMod = nullptr;
        why = "its kernel module did not load";
        return false;
    }
    for (int i = 0; i < (int)(sizeof(kBbKern) / sizeof(kBbKern[0])); i++)
        if (cuModuleGetFunction(&nr.fBb[i], nr.bbMod, kBbKern[i]) != CUDA_SUCCESS)
        {
            why = std::string("its kernel ") + kBbKern[i] + " is missing";
            return false;
        }
    if (cudaMalloc(&nr.dBbW, blob.size()) != cudaSuccess ||
        cudaMemcpy(nr.dBbW, blob.data(), blob.size(), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        why = "its weights did not upload";
        return false;
    }
    return true;
}

// the backbone's 54 launches on st for one frame at the half nr.hw x nr.hh: x = its 3 fp32 planes, map4 / map8 = A's
// batch slot (planar fp16), the activations at s (bbArena's pieces). false = a launch failed
static bool nativeBbLaunch(NativeRife& nr, const float* x, void* map4, void* map8, uint8_t* s, cudaStream_t st)
{
    int hh = nr.hh, hw = nr.hw, h1 = hh / 2, w1 = hw / 2, h2 = hh / 4, w2 = hw / 4, p1 = h1 * w1, p2 = h2 * w2;
    const BbArena ar = bbArena(hw, hh);
    const char* wb = (const char*)nr.dBbW;
    auto wp = [&](int i) { return (const void*)(wb + nr.bbOff[i]); };
    void* l1[5];
    void* l2[5];
    void* l3[6];
    for (int i = 0; i < 5; i++)
    {
        l1[i] = s + ar.l1[i];
        l2[i] = s + ar.l2[i];
    }
    for (int i = 0; i < 6; i++)
        l3[i] = s + ar.l3[i];
    void* part = s + ar.part;
    float* sv[4];
    for (int i = 0; i < 4; i++)
        sv[i] = (float*)(s + ar.st[i]);
    auto go = [&](int f, int gx, int gy, int gz, int threads, int sm, void** a) {
        return cuLaunchKernel(nr.fBb[f], (unsigned)gx, (unsigned)gy, (unsigned)gz, (unsigned)threads, 1, 1,
                              (unsigned)sm, (CUstream)st, a, nullptr) == CUDA_SUCCESS;
    };
    auto stats = [&](void* y, int P, int C, float* so) {
        int pb = bbPb(P), nb = (P + pb - 1) / pb;
        void* a[] = {&y, &P, &C, &pb, &part};
        void* b[] = {&part, &nb, &C, (void*)&so};
        return go(kBbStats, nb, 1, 1, 256, 0, a) && go(kBbFin, C, 1, 1, 128, 0, b);
    };
    // a 3 x 3 conv: kernel f, stride sd, group cg, n-tiles nb; bias = the zero piece unless given
    auto conv = [&](int f, int sd, int cg, int nbt, void* xin, int cin, int ih, int iw, void* y, int cout, int wi,
                    const float* sn) {
        const void* w = wp(wi);
        const void* bz = wp(kBbZero);
        const float* sp = sn ? sn : sv[0];
        const int oh = ih / sd, ow = iw / sd, sm = (sd == 1 ? 10 * 34 : 9 * 65) * (cg / 8) * 16;
        void* a[] = {&xin, &cin, &ih, &iw, &y, &cout, (void*)&w, (void*)&bz, (void*)&sp};
        return go(f, (ow + 31) / 32, sd == 1 ? (oh + 7) / 8 : (oh + 3) / 4, cout / (8 * nbt), 128, sm, a);
    };
    auto res = [&](int f, void* y, float* sy, void* xin, float* sx, int P, int C, void* out) {
        void* a[] = {&y, (void*)&sy, &xin, (void*)&sx, &P, &C, &out};
        return go(f, (P * C / 8 + 255) / 256, 1, 1, 256, 0, a);
    };
    auto gemm = [&](int f, void* a0, int M, int OW, int K, int wi, int bi, void* y, int N, int nt) {
        const void* w = wp(wi);
        const void* b = wp(bi);
        const int nrow = ((K / 8) + 7) & ~7;
        void* a[] = {&a0, &M, &OW, &K, (void*)&w, (void*)&b, &y, &N};
        return go(f, (M + 63) / 64, N / (8 * nt), 1, 128, 64 * nrow * 16, a);
    };
    const void* w7 = wp(kBbW7);
    void* y0 = l1[0];
    {
        void* a[] = {(void*)&x, &hh, &hw, (void*)&w7, &y0};
        if (!go(kBbC7, (w1 + 31) / 32, (h1 + 3) / 4, 1, 128, 0, a) || !stats(y0, p1, 64, sv[0]))
            return false;
    }
    // layer1 (64): block 0 takes relu(IN(conv1)) (normalized on load, its skip by k_bbRes1), block 1 its output x1
    bool ok = conv(kBbC64N, 1, 64, 8, y0, 64, h1, w1, l1[1], 64, kBbW3 + 0, sv[0]) && stats(l1[1], p1, 64, sv[1]) &&
              conv(kBbC64N, 1, 64, 8, l1[1], 64, h1, w1, l1[2], 64, kBbW3 + 1, sv[1]) && stats(l1[2], p1, 64, sv[2]) &&
              res(kBbRes1, l1[2], sv[2], y0, sv[0], p1, 64, l1[3]) &&
              conv(kBbC64, 1, 64, 8, l1[3], 64, h1, w1, l1[1], 64, kBbW3 + 2, nullptr) && stats(l1[1], p1, 64, sv[1]) &&
              conv(kBbC64N, 1, 64, 8, l1[1], 64, h1, w1, l1[2], 64, kBbW3 + 3, sv[1]) && stats(l1[2], p1, 64, sv[2]) &&
              res(kBbRes0, l1[2], sv[2], l1[3], sv[2], p1, 64, l1[4]);
    // layer2 (stride 2, 64 -> 96, a 1 x 1 s2 downsample)
    ok = ok && conv(kBbS64to96, 2, 32, 4, l1[4], 64, h1, w1, l2[0], 96, kBbW3 + 4, nullptr) &&
         stats(l2[0], p2, 96, sv[1]) && conv(kBbC96N, 1, 32, 4, l2[0], 96, h2, w2, l2[1], 96, kBbW3 + 5, sv[1]) &&
         stats(l2[1], p2, 96, sv[2]) && gemm(kBbDs2, l1[4], p2, w2, 64, kBbWds2, kBbBds2, l2[2], 96, 6) &&
         stats(l2[2], p2, 96, sv[3]) && res(kBbRes2, l2[1], sv[2], l2[2], sv[3], p2, 96, l2[3]) &&
         conv(kBbC96, 1, 32, 4, l2[3], 96, h2, w2, l2[0], 96, kBbW3 + 6, nullptr) && stats(l2[0], p2, 96, sv[1]) &&
         conv(kBbC96N, 1, 32, 4, l2[0], 96, h2, w2, l2[1], 96, kBbW3 + 7, sv[1]) && stats(l2[1], p2, 96, sv[2]) &&
         res(kBbRes0, l2[1], sv[2], l2[3], sv[2], p2, 96, l2[4]);
    // layer3 (stride 1, 96 -> 128, a 1 x 1 downsample)
    ok = ok && conv(kBbC96to128, 1, 32, 8, l2[4], 96, h2, w2, l3[0], 128, kBbW3 + 8, nullptr) &&
         stats(l3[0], p2, 128, sv[1]) && conv(kBbC64N, 1, 64, 8, l3[0], 128, h2, w2, l3[1], 128, kBbW3 + 9, sv[1]) &&
         stats(l3[1], p2, 128, sv[2]) && gemm(kBbG1, l2[4], p2, w2, 96, kBbWds3, kBbBds3, l3[2], 128, 8) &&
         stats(l3[2], p2, 128, sv[3]) && res(kBbRes2, l3[1], sv[2], l3[2], sv[3], p2, 128, l3[3]) &&
         conv(kBbC64, 1, 64, 8, l3[3], 128, h2, w2, l3[0], 128, kBbW3 + 10, nullptr) && stats(l3[0], p2, 128, sv[1]) &&
         conv(kBbC64N, 1, 64, 8, l3[0], 128, h2, w2, l3[1], 128, kBbW3 + 11, sv[1]) && stats(l3[1], p2, 128, sv[2]) &&
         res(kBbRes0, l3[1], sv[2], l3[3], sv[2], p2, 128, l3[4]);
    // conv2 (1 x 1, bias), then the trident: map4 at stride 1, map8 at stride 2
    return ok && gemm(kBbG1, l3[4], p2, w2, 128, kBbWc2, kBbBc2, l3[5], 128, 8) &&
           conv(kBbTri, 1, 64, 8, l3[5], 128, h2, w2, map4, 128, kBbWtri, nullptr) &&
           conv(kBbTri2, 2, 32, 8, l3[5], 128, h2, w2, map8, 128, kBbWtri2, nullptr);
}

// the backbone on one frame on st (nr.stream): x = its 3 fp32 half planes, map4 / map8 = A's batch slot; the launches go
// through nr.bbGraphs (chainGraphRun). false = the activations or a launch failed
static bool nativeBbRun(NativeRife& nr, const float* x, void* map4, void* map8, cudaStream_t st)
{
    const BbArena ar = bbArena(nr.hw, nr.hh);
    uint8_t* s = nr.dGmCtxMem && (size_t)nr.gmCtxMemB >= ar.total ? (uint8_t*)nr.dGmCtxMem : (uint8_t*)nr.dBbOwn;
    if (!s && cudaMalloc(&nr.dBbOwn, ar.total) == cudaSuccess)
        s = (uint8_t*)nr.dBbOwn;
    if (!s)
        return false;
    return chainGraphRun(nr.bbGraphs, {s, x, map4, map8}, nr.hw, nr.hh, st, "gmfss gmflow backbone",
                         [&] { return nativeBbLaunch(nr, x, map4, map8, s, st); });
}
