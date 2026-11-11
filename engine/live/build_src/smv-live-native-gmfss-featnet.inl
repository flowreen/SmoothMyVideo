// part of smv-live-native.inl (included there, in place); not a standalone translation unit
// ---- GMFSS's feat_ext (FeatureNet) on the host's kernels -----------------------------------------------------------
// FeatureNet = 3 blocks {LeakyRelu, conv 3 x 3 stride 2, LeakyRelu, conv 3 x 3}: 3 -> 64 -> 64 (f1, at the half),
// -> 128 -> 128 (f2, the quarter), -> 192 -> 192 (f3, the eighth); its PReLUs are scalar. Weights from
// engine\GMFSS_Fortuna\train_log\feat.pkl (the host's torch zip reader). Its own NVRTC module
// (fnKernelText): k_fnConv0L = block 1's first conv on the fp32 frame planes, both LeakyRelus in fp32, its weights as
// float literals, one rounding to channels-last fp16; the five other convs on the tensor cores (fnConvT: an implicit
// GEMM, mma.sync m16n8k16 with fp16 sums started at the bias; M = 16-pixel row runs of a TH x 32 output tile, N = 8 NB
// outputs a block (blockIdx.z = the output slice), K = 9 taps x the input channels, CG at a time: stride 1 = the
// 64-channel halo of an 8 x 32 tile, stride 2 = the 32-channel halo of a 4 x 32 tile with each row's even offsets
// before its odd ones; 16-byte chunks XOR-swizzled so each ldmatrix reads 8 bank groups; the input's LeakyRelu on the
// A fragments (an fp16 multiply, then max: LeakyRelu for any slope up to 1), the output's in the epilogue; weights from
// global memory in fragment order, a k-step ahead). Activations channels-last fp16; f1 / f2 / f3 channels-last (the
// planar form, planes beside the channels-last copies the next convs read, has no caller since fusionnet runs on the
// host's kernels).
static const int kFnCh[4] = {3, 64, 128, 192};   // the channel widths
static const int kFnNb[5] = {8, 16, 16, 24, 12}; // n-tiles a block, conv 1..5
static const int kFnSmS1 = 10 * 34 * 8 * 16;     // a stride-1 block's halo bytes
static const int kFnSmS2 = 9 * 65 * 4 * 16;      // a stride-2 block's
static const char* const kFnKern[6] = {"k_fnConv0L",       "k_fnConvS1A16N8",  "k_fnConvS2A16N16",
                                       "k_fnConvS1A16N16", "k_fnConvS2A16N24", "k_fnConvS1A16N12"};

static const char kFnHead[] = R"FNSRC(
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
__device__ __forceinline__ float h2f(unsigned short h)
{
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}
__device__ __forceinline__ unsigned short f2h(float f)
{
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}
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
__device__ __forceinline__ unsigned rnPrelu(unsigned v, unsigned short a)
{
    const float y = h2f((unsigned short)v);
    return y < 0.0f ? (unsigned)f2h(h2f(a) * y) : v;
}
__device__ __forceinline__ float fnLr(float v, float a)
{
    return v < 0.0f ? v * a : v;
}
)FNSRC";

static const char kFnConv[] = R"FNSRC(
// the LeakyRelu of two fp16 values (slope s2 = the slope twice): max(v, s v), one fp16 rounding of the product
__device__ __forceinline__ unsigned fnLr2(unsigned v, unsigned s2)
{
    unsigned t;
    asm("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(t) : "r"(v), "r"(s2));
#if __CUDA_ARCH__ >= 800
    asm("max.f16x2 %0, %1, %2;\n" : "=r"(t) : "r"(v), "r"(t));
#else
    // sm_75 has no fp16 max: the same max in fp32 (exact: the result is one of the two fp16 values)
    t = (unsigned)f2h(fmaxf(h2f((unsigned short)v), h2f((unsigned short)t))) |
        ((unsigned)f2h(fmaxf(h2f((unsigned short)(v >> 16)), h2f((unsigned short)(t >> 16)))) << 16);
#endif
    return t;
}

// the stored index of halo pixel (hr, hc) and its chunk k's swizzled slot
template <int S, int NCH>
__device__ __forceinline__ unsigned fnSlot(int sp, int k)
{
    const int sw = NCH == 8 ? (k ^ (sp & 7)) : (k ^ ((sp >> 1) & 3));
    return (unsigned)((sp * NCH + sw) << 4);
}

// the CG-channel halo (channels c0 .. c0 + CG - 1) of the output tile at (tx0, ty0) into shared memory, zero outside
// the input frame W x H, cp.async; the caller waits
template <int S, int CG, int TH>
__device__ __forceinline__ void fnHalo(unsigned halo, const unsigned short* __restrict__ x, int cin, int c0, int tx0,
                                       int ty0, int H, int W)
{
    constexpr int NCH = CG / 8;
    constexpr int HR = S == 1 ? TH + 2 : 2 * TH + 1, HC = S == 1 ? 34 : 65;
    for (int i = threadIdx.x; i < HR * HC * NCH; i += 128)
    {
        const int p = i / NCH, k = i - p * NCH;
        const int hr = p / HC, hc = p - hr * HC;
        const int gy = S * ty0 - 1 + hr, gx = S * tx0 - 1 + hc;
        const int sp = S == 1 ? p : hr * HC + ((hc & 1) ? 33 + (hc >> 1) : (hc >> 1));
        const bool in = gy >= 0 && gy < H && gx >= 0 && gx < W;
        const unsigned short* src = in ? x + ((size_t)gy * W + gx) * cin + c0 + k * 8 : x;
        rnCp16(halo + fnSlot<S, NCH>(sp, k), src, in ? 16 : 0);
    }
    asm volatile(SMV_CP_COMMIT);
}

// one conv: x = cin channels-last fp16 at W x H, y = cout channels-last fp16 at W / S x H / S and / or ylin = the same
// values as cout fp16 planes (either may be null); wf[z][step][q][lane] (step = (group * 9 + tap) * CG / 16 + kc); bias
// fp32; aIn / aOut = the LeakyRelu slopes (1 = none)
template <int S, int CG, bool A32, int NB>
__device__ __forceinline__ void fnConvT(const unsigned short* __restrict__ x, int cin, int H, int W,
                                        unsigned short* __restrict__ y, int cout, const uint4* __restrict__ wf,
                                        const float* __restrict__ bias, float aIn, float aOut,
                                        unsigned short* __restrict__ ylin)
{
    constexpr int TH = S == 1 ? 8 : 4;
    constexpr int RW = TH / 4, MI = RW * 2;
    constexpr int HC = S == 1 ? 34 : 65;
    constexpr int NCH = CG / 8, KC = CG / 16;
    extern __shared__ __align__(128) unsigned char fnSm[];
    const unsigned halo = (unsigned)__cvta_generic_to_shared(fnSm);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int tx0 = blockIdx.x * 32, ty0 = blockIdx.y * TH, z = blockIdx.z;
    const int OW = W / S, OH = H / S;
    const int ksteps = cin * 9 / 16;
    constexpr int NQ = NB / 2;
    wf += (size_t)z * ksteps * NQ * 32;
    const int g = lane >> 2, t4 = lane & 3;
    const bool lrIn = aIn != 1.0f, lrOut = aOut != 1.0f;
    const unsigned s2 = (unsigned)f2h(aIn) * 0x10001u;
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
        fnHalo<S, CG, TH>(halo, x, cin, gi * CG, tx0, ty0, H, W);
        asm volatile(SMV_CP_WAIT0);
        __syncthreads();
#pragma unroll 1
        for (int tap = 0; tap < 9; ++tap)
        {
            const int dy = tap / 3, dx = tap - dy * 3;
            int pa[MI];
#pragma unroll
            for (int mi = 0; mi < MI; ++mi)
            {
                const int orow = RW * warp + (mi >> 1), col = (mi & 1) * 16 + arow;
                pa[mi] = S == 1 ? (orow + dy) * HC + col + dx
                                : (2 * orow + dy) * HC + (dx == 1 ? 33 + col : col + (dx >> 1));
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
                    rnLdm4(halo + fnSlot<S, NCH>(pa[mi], kc * 2 + akc), a);
                    if (lrIn)
#pragma unroll
                        for (int r = 0; r < 4; ++r)
                            a[r] = fnLr2(a[r], s2);
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
    const unsigned short so = f2h(aOut);
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
                unsigned v;
                if constexpr (A32)
                {
                    float v0 = accF[mi][nj][2 * h], v1 = accF[mi][nj][2 * h + 1];
                    if (lrOut)
                    {
                        v0 = fnLr(v0, aOut);
                        v1 = fnLr(v1, aOut);
                    }
                    v = (unsigned)f2h(v0) | ((unsigned)f2h(v1) << 16);
                }
                else
                {
                    v = accH[mi][nj][h];
                    if (lrOut)
                        v = rnPrelu(v & 0xffffu, so) | (rnPrelu(v >> 16, so) << 16);
                }
                const size_t pix = (size_t)oy * OW + ox;
                if (y)
                    *reinterpret_cast<unsigned*>(y + pix * cout + n0) = v;
                if (ylin)
                {
                    const size_t plane = (size_t)OW * OH;
                    ylin[(size_t)n0 * plane + pix] = (unsigned short)(v & 0xffffu);
                    ylin[(size_t)(n0 + 1) * plane + pix] = (unsigned short)(v >> 16);
                }
            }
        }
    }
}

#define FN_CONV(name, S, CG, A32, NB)                                                                                  \
    extern "C" __global__ void __launch_bounds__(128, 2)                                                               \
        name(const unsigned short* __restrict__ x, int cin, int H, int W, unsigned short* __restrict__ y, int cout,  \
             const uint4* __restrict__ wf, const float* __restrict__ bias, float aIn, float aOut,                      \
             unsigned short* __restrict__ ylin)                                                                        \
    {                                                                                                                  \
        fnConvT<S, CG, A32, NB>(x, cin, H, W, y, cout, wf, bias, aIn, aOut, ylin);                                     \
    }
FN_CONV(k_fnConvS1A16N8, 1, 64, false, 8)
FN_CONV(k_fnConvS2A16N16, 2, 32, false, 16)
FN_CONV(k_fnConvS1A16N16, 1, 64, false, 16)
FN_CONV(k_fnConvS2A16N24, 2, 32, false, 24)
FN_CONV(k_fnConvS1A16N12, 1, 64, false, 12)
)FNSRC";

// the module's whole text: kFnHead, kFnConv, then k_fnConv0L with block 1's first conv as literals (per output: the
// bias, then fmaf over channel, tap in PyTorch's order)
static void fnKernelText(std::string& s, const std::vector<float>& w0, const std::vector<float>& b0)
{
    s.reserve((size_t)1 << 18);
    s = kFnHead;
    s += kFnConv;
    s +=
        "extern \"C\" __global__ void __launch_bounds__(256) k_fnConv0L(const float* __restrict__ x, int ps, int rs,\n"
        "    int W, int H, unsigned short* __restrict__ y, float aIn, float aOut)\n{\n"
        "    const int OW = W >> 1, OH = H >> 1;\n"
        "    const int ox = blockIdx.x * 32 + (threadIdx.x & 31), oy = blockIdx.y * 8 + (threadIdx.x >> 5);\n"
        "    if (ox >= OW || oy >= OH)\n        return;\n    float v[27];\n"
        "#pragma unroll\n    for (int c = 0; c < 3; ++c)\n#pragma unroll\n        for (int t = 0; t < 9; ++t)\n        {\n"
        "            const int gy = 2 * oy + t / 3 - 1, gx = 2 * ox + t % 3 - 1;\n"
        "            v[c * 9 + t] = (gy >= 0 && gy < H && gx >= 0 && gx < W)\n"
        "                               ? fnLr(x[(size_t)c * ps + (size_t)gy * rs + gx], aIn)\n"
        "                               : 0.0f;\n        }\n"
        "    unsigned short* o = y + ((size_t)oy * OW + ox) * 64;\n    unsigned r[4];\n    float a;\n";
    char ln[96];
    auto lit = [&](float v) {
        uint32_t b = 0;
        memcpy(&b, &v, sizeof b);
        a4kLit(s, b);
    };
    for (int n = 0; n < 64; n++)
    {
        s += "    a = ";
        lit(b0[n]);
        s += ";";
        for (int k = 0; k < 27; k++)
        {
            s += " a = fmaf(";
            lit(w0[(size_t)n * 27 + k]);
            sprintf_s(ln, ", v[%d], a);", k);
            s += ln;
        }
        s += "\n";
        if (n % 2 == 0)
            sprintf_s(ln, "    r[%d] = (unsigned)f2h(fnLr(a, aOut));\n", (n % 8) / 2);
        else
            sprintf_s(ln, "    r[%d] |= (unsigned)f2h(fnLr(a, aOut)) << 16;\n", (n % 8) / 2);
        s += ln;
        if (n % 8 == 7)
        {
            sprintf_s(ln, "    *reinterpret_cast<uint4*>(o + %d) = make_uint4(r[0], r[1], r[2], r[3]);\n", n - 7);
            s += ln;
        }
    }
    s += "}\n";
}

// a conv's weights [co][ci][3][3] in fnConvT's fragment order [z][step][q][lane][8 halves] (8 nb outputs a z; lane's
// B registers for n-tiles 2q, 2q + 1: n = 8 (2q (+ 1)) + lane / 4, k = 2 (lane % 4) (+ 1, + 8, + 9)), rounded to fp16
static std::vector<uint16_t> fnFragOf(const PthW& w, int co, int ci, int cg, int nb)
{
    static const int kOfs[4] = {0, 1, 8, 9};
    const int kcn = cg / 16, nq = nb / 2;
    std::vector<uint16_t> out((size_t)(co / (8 * nb)) * (ci * 9 / 16) * nq * 32 * 8);
    size_t o = 0;
    for (int z = 0; z < co / (8 * nb); z++)
        for (int gi = 0; gi < ci / cg; gi++)
            for (int tap = 0; tap < 9; tap++)
                for (int kc = 0; kc < kcn; kc++)
                    for (int q = 0; q < nq; q++)
                        for (int lane = 0; lane < 32; lane++)
                            for (int hf = 0; hf < 2; hf++)
                                for (int j = 0; j < 4; j++)
                                {
                                    const int n = z * nb * 8 + (2 * q + hf) * 8 + (lane >> 2);
                                    const int c = gi * cg + kc * 16 + (lane & 3) * 2 + kOfs[j];
                                    out[o++] = w.half(((size_t)n * ci + c) * 9 + tap);
                                }
    return out;
}

// feat.pkl in the strict FeatureNet form (block{1,2,3}: .0 / .2 = scalar PReLU slopes up to 1, .1 = conv 3 x 3
// stride 2, .3 = conv 3 x 3; 64 / 128 / 192): conv 0's weights and bias (fp32, for the text), the six slopes, and per
// conv 1..5 (block1.3, block2.1, block2.3, block3.1, block3.3) its fragment-order weights and fp32 bias in one block
// (256-byte aligned, off[2 c] / off[2 c + 1]). false = missing, or not this net (`why`).
static bool fnNatLoad(std::vector<uint8_t>& blob, size_t* off, std::vector<float>& w0, std::vector<float>& b0,
                      float* slope, std::string& why)
{
    PthFile pf;
    if (!pthStateDict(g_nativeEngineDir + L"\\GMFSS_Fortuna\\train_log\\feat.pkl", pf, why))
        return false;
    auto& wts = pf.t;
    // the slopes, the biases and conv 0's weights go to the kernels as fp32
    bool form = wts.size() == 18 && pf.is32("block1.1.weight", {kFnCh[1], kFnCh[0], 3, 3});
    for (int b = 0; b < 3 && form; b++)
    {
        const std::string p = "block" + std::to_string(b + 1) + ".";
        const int64_t ci = kFnCh[b], co = kFnCh[b + 1];
        form = pf.is32(p + "0.weight", {1}) && pf.is32(p + "2.weight", {1}) && pf.is(p + "1.weight", {co, ci, 3, 3}) &&
               pf.is32(p + "1.bias", {co}) && pf.is(p + "3.weight", {co, co, 3, 3}) && pf.is32(p + "3.bias", {co});
        if (form)
        {
            slope[2 * b] = pf.f32(p + "0.weight")[0];
            slope[2 * b + 1] = pf.f32(p + "2.weight")[0];
            // the A fragments' LeakyRelu = max(x, a x), exact for a slope up to 1
            form = slope[2 * b] <= 1.0f && slope[2 * b + 1] <= 1.0f;
        }
    }
    if (!form)
    {
        why = pf.notForm("GMFSS's FeatureNet");
        return false;
    }
    const std::span<const float> c0w = pf.f32("block1.1.weight"), c0b = pf.f32("block1.1.bias");
    w0.assign(c0w.begin(), c0w.end());
    b0.assign(c0b.begin(), c0b.end());
    blob.clear();
    auto put = [&](const void* p, size_t n) {
        const size_t o = (blob.size() + 255) & ~(size_t)255;
        blob.resize(o + n);
        memcpy(blob.data() + o, p, n);
        return o;
    };
    static const char* const kConv[5] = {"block1.3", "block2.1", "block2.3", "block3.1", "block3.3"};
    for (int c = 0; c < 5; c++)
    {
        const int ci = kFnCh[c / 2 + 1], co = kFnCh[(c + 1) / 2 + 1];
        const std::vector<uint16_t> wf =
            fnFragOf(wts[std::string(kConv[c]) + ".weight"], co, ci, (c & 1) ? 32 : 64, kFnNb[c]);
        const std::span<const float> bs = pf.f32(std::string(kConv[c]) + ".bias");
        off[2 * c] = put(wf.data(), wf.size() * sizeof(uint16_t));
        off[2 * c + 1] = put(bs.data(), bs.size() * sizeof(float));
    }
    return true;
}

// feat_ext's scratch: t1 (64 channels at the half; t3 and t5 reuse it), and with planar levels the channels-last
// copies of f1 / f2 the next convs read (64 more)
static size_t fnNatBytes(const NativeRife& nr, bool hwc)
{
    return (size_t)(hwc ? 64 : 128) * (size_t)(nr.pw / 2) * (size_t)(nr.ph / 2) * sizeof(uint16_t);
}

// this session's feat_ext module: the weights read, its text compiled (or the cached cubin) and loaded, its six kernels
// bound, the weight block uploaded; false = `why` (the session teardown unloads a module left in fnMod)
static bool nativeFnModule(NativeRife& nr, std::string& why)
{
    if (nr.fnMod)
        return true;
    std::vector<uint8_t> blob;
    std::vector<float> w0, b0;
    if (!fnNatLoad(blob, nr.fnOff, w0, b0, nr.fnSlope, why))
        return false;
    std::string text;
    fnKernelText(text, w0, b0);
    std::vector<char> cubin;
    if (!nativeCubin(text.c_str(), "smv_featnet.cu", L"smv_featnet", "feat_ext kernels",
                     nr.kernCacheDir.empty() ? std::wstring(L".") : nr.kernCacheDir, cubin))
    {
        why = "its kernels did not compile";
        return false;
    }
    if (cuModuleLoadData(&nr.fnMod, cubin.data()) != CUDA_SUCCESS)
    {
        nr.fnMod = nullptr;
        why = "its kernel module did not load";
        return false;
    }
    for (int i = 0; i < 6; i++)
        if (cuModuleGetFunction(&nr.fFn[i], nr.fnMod, kFnKern[i]) != CUDA_SUCCESS)
        {
            why = std::string("its kernel ") + kFnKern[i] + " is missing";
            return false;
        }
    if (cudaMalloc(&nr.dFnW, blob.size()) != cudaSuccess ||
        cudaMemcpy(nr.dFnW, blob.data(), blob.size(), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        why = "its weights did not upload";
        return false;
    }
    return true;
}

// feat_ext of one frame on st: x = 3 fp32 planes (plane stride ps, row stride rs) at the padded frame -> f[0..2]
// (nr.gmFeatHwc: channels-last, else planes); false = a launch failed
static bool nativeFnRun(NativeRife& nr, const float* x, int ps, int rs, void** f, cudaStream_t st)
{
    int fw = nr.pw, fh = nr.ph;
    const int hw = fw / 2, hh = fh / 2;
    const size_t hp = (size_t)hw * hh;
    const bool lin = !nr.gmFeatHwc;
    unsigned short* t = (unsigned short*)nr.dFnScr; // t1, then t3, then t5
    unsigned short* cl = t + 64 * hp;               // planes only: f1's, then f2's channels-last copy
    unsigned short* const nul = nullptr;
    float a0 = nr.fnSlope[0], a1 = nr.fnSlope[1];
    const float* xp = x;
    void* c0[] = {(void*)&xp, &ps, &rs, &fw, &fh, (void*)&t, &a0, &a1};
    if (cuLaunchKernel(nr.fFn[0], (hw + 31) / 32, (hh + 7) / 8, 1, 256, 1, 1, 0, (CUstream)st, c0, nullptr) !=
        CUDA_SUCCESS)
        return false;
    unsigned short* f1 = (unsigned short*)f[0];
    unsigned short* f2 = (unsigned short*)f[1];
    unsigned short* f3 = (unsigned short*)f[2];
    struct Conv
    {
        unsigned short* in;
        int cin, h, w;
        unsigned short* y;
        int cout, s;
        float ai, ao;
        unsigned short* yl;
    } cv[5] = {{t, 64, hh, hw, lin ? cl : f1, 64, 1, 1.0f, 1.0f, lin ? f1 : nul},
               {lin ? cl : f1, 64, hh, hw, t, 128, 2, nr.fnSlope[2], nr.fnSlope[3], nul},
               {t, 128, hh / 2, hw / 2, lin ? cl : f2, 128, 1, 1.0f, 1.0f, lin ? f2 : nul},
               {lin ? cl : f2, 128, hh / 2, hw / 2, t, 192, 2, nr.fnSlope[4], nr.fnSlope[5], nul},
               {t, 192, hh / 4, hw / 4, lin ? nul : f3, 192, 1, 1.0f, 1.0f, lin ? f3 : nul}};
    for (int i = 0; i < 5; i++)
    {
        Conv& c = cv[i];
        const int oh = c.h / c.s, ow = c.w / c.s, th = c.s == 1 ? 8 : 4;
        const uint4* wf = (const uint4*)((const char*)nr.dFnW + nr.fnOff[2 * i]);
        const float* bias = (const float*)((const char*)nr.dFnW + nr.fnOff[2 * i + 1]);
        void* a[] = {(void*)&c.in, &c.cin,       &c.h,  &c.w,  (void*)&c.y, &c.cout,
                     (void*)&wf,   (void*)&bias, &c.ai, &c.ao, (void*)&c.yl};
        if (cuLaunchKernel(nr.fFn[1 + i], (ow + 31) / 32, (oh + th - 1) / th, c.cout / (8 * kFnNb[i]), 128, 1, 1,
                           c.s == 1 ? kFnSmS1 : kFnSmS2, (CUstream)st, a, nullptr) != CUDA_SUCCESS)
            return false;
    }
    return true;
}
