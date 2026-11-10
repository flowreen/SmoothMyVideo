// part of smv-live-native.inl (included there, in place); not a standalone translation unit
// ---- GMFSS's MetricNet on the host's kernels ----------------------------------------------------------------------
// MetricNet at the GMFSS half (the two backward warps' photometric means, gmflow's forward-backward check, conv_in, the
// three residual convs, conv_out, tanh x 10) as 6 launches of its own module smv_metric_*.cubin (kFnHead's helpers +
// kMnText), weights = engine\GMFSS_Fortuna\train_log\metric.pkl through pthStateDict (mnNatLoad); no fallback (a missing / wrong pkl =
// `metricnet unavailable (...)`, the session refuses). It takes the motion halves (3 fp32 planes each) and both flow
// directions (2 fp32 planes each) and hands out m0 then m1 as fp16 planes. Its scratch (mnArena) lives in the shared
// activation block (nativeGmShareMemory sizes it in; sound while every enqueue there runs serially on nr.stream), else
// dMnOwn.
static const char kMnText[] = R"MNSRC(
// GMFSS's MetricNet on the host's own kernels at the half size W x H (the host's GMFSS half, a multiple of 32).
// k_mnIn      per pixel: metric0 = mean_c |i0 - i1 at p + f01|, metric1 = mean_c |i1 - i0 at p + f10| (bilinear, zeros
//             outside: grid_sample's align_corners point of linspace + f / ((W - 1) / 2) is p + f), the forward-backward
//             check (|f01 + f10 at p + f01| > 0.01 (|f01| + |f10|) + 0.5, and the other way), the normalized flows; the
//             14-channel input cat(i0, i1, -metric0, -metric1, f01 n, f10 n, occ fwd, occ bwd) channels-last fp16
//             [H][W][16]
// k_mnConv16  conv_in 14 -> 64 on the zero-padded 16 channels; k_mnRes64 the ResConvs y = conv(PReLU(x)) + x, k_mnRes64P
//             the last one, its output stored through conv_out's PReLU (conv_out is its only reader); k_mnOut conv_out 64
//             -> 2 (padded to 16) then tanh x 10 into the fp16 planes m0, m1: implicit GEMMs on the tensor cores (mma.sync
//             m16n8k16, an 8 x 32 output tile, a warp = 2 rows = 4 m-tiles, 64 outputs a block (16 for conv_out), the
//             weights from global memory in fragment order a k-step ahead), the halo in shared memory (PReLU applied
//             to it once: fp16, the engine's LeakyRelu), fp16 accumulators from the bias (conv_out fp32), the residual =
//             x at the output pixel added in fp32, one rounding to fp16

struct MnTap
{
    int o[4];
    float w[4];
};

// PyTorch's grid_sampler_2d bilinear corners and weights at (px, py), zeros padding: a corner outside = weight 0
__device__ __forceinline__ MnTap mnTap(float px, float py, int W, int H)
{
    const float fx = floorf(px), fy = floorf(py);
    const int x0 = (int)fx, y0 = (int)fy, x1 = x0 + 1, y1 = y0 + 1;
    const float ex = fx + 1.0f, ey = fy + 1.0f;
    const float nw = (ex - px) * (ey - py), ne = (px - fx) * (ey - py), sw = (ex - px) * (py - fy),
                se = (px - fx) * (py - fy);
    MnTap t;
    const bool ix0 = x0 >= 0 && x0 < W, ix1 = x1 >= 0 && x1 < W, iy0 = y0 >= 0 && y0 < H, iy1 = y1 >= 0 && y1 < H;
    t.o[0] = ix0 && iy0 ? y0 * W + x0 : 0;
    t.w[0] = ix0 && iy0 ? nw : 0.0f;
    t.o[1] = ix1 && iy0 ? y0 * W + x1 : 0;
    t.w[1] = ix1 && iy0 ? ne : 0.0f;
    t.o[2] = ix0 && iy1 ? y1 * W + x0 : 0;
    t.w[2] = ix0 && iy1 ? sw : 0.0f;
    t.o[3] = ix1 && iy1 ? y1 * W + x1 : 0;
    t.w[3] = ix1 && iy1 ? se : 0.0f;
    return t;
}

__device__ __forceinline__ float mnAt(const float* __restrict__ p, const MnTap& t)
{
    float v = 0.0f;
    v += p[t.o[0]] * t.w[0];
    v += p[t.o[1]] * t.w[1];
    v += p[t.o[2]] * t.w[2];
    v += p[t.o[3]] * t.w[3];
    return v;
}

extern "C" __global__ void __launch_bounds__(128) k_mnIn(const float* __restrict__ i0, const float* __restrict__ i1,
                                                          const float* __restrict__ f01, const float* __restrict__ f10,
                                                          int W, int H, unsigned short* __restrict__ x)
{
    const int px = blockIdx.x * 128 + threadIdx.x, py = blockIdx.y;
    if (px >= W)
        return;
    const size_t hw = (size_t)W * H, p = (size_t)py * W + px;
    const float ax = f01[p], ay = f01[hw + p], bx = f10[p], by = f10[hw + p];
    const MnTap ta = mnTap((float)px + ax, (float)py + ay, W, H), tb = mnTap((float)px + bx, (float)py + by, W, H);
    float c[16];
    float m0 = 0.0f, m1 = 0.0f;
#pragma unroll
    for (int k = 0; k < 3; ++k)
    {
        const float a = i0[k * hw + p], b = i1[k * hw + p];
        c[k] = a;
        c[3 + k] = b;
        m0 += fabsf(a - mnAt(i1 + k * hw, ta));
        m1 += fabsf(b - mnAt(i0 + k * hw, tb));
    }
    c[6] = -__fdiv_rn(m0, 3.0f);
    c[7] = -__fdiv_rn(m1, 3.0f);
    const float nx = ((float)W - 1.0f) / 2.0f, ny = ((float)H - 1.0f) / 2.0f;
    c[8] = __fdiv_rn(ax, nx);
    c[9] = __fdiv_rn(ay, ny);
    c[10] = __fdiv_rn(bx, nx);
    c[11] = __fdiv_rn(by, ny);
    const float wbx = mnAt(f10, ta), wby = mnAt(f10 + hw, ta), wfx = mnAt(f01, tb), wfy = mnAt(f01 + hw, tb);
    const float mag = sqrtf(ax * ax + ay * ay) + sqrtf(bx * bx + by * by);
    const float thr = 0.01f * mag + 0.5f;
    const float dfx = ax + wbx, dfy = ay + wby, dbx = bx + wfx, dby = by + wfy;
    c[12] = sqrtf(dfx * dfx + dfy * dfy) > thr ? 1.0f : 0.0f;
    c[13] = sqrtf(dbx * dbx + dby * dby) > thr ? 1.0f : 0.0f;
    c[14] = c[15] = 0.0f;
    uint4 v[2];
    unsigned* u = reinterpret_cast<unsigned*>(v);
#pragma unroll
    for (int k = 0; k < 8; ++k)
        u[k] = (unsigned)f2h(c[2 * k]) | ((unsigned)f2h(c[2 * k + 1]) << 16);
    uint4* dst = reinterpret_cast<uint4*>(x + p * 16);
    dst[0] = v[0];
    dst[1] = v[1];
}

// the stored index of halo pixel sp and its chunk k's swizzled slot (each ldmatrix's 8 rows in 8 bank groups)
template <int NCH>
__device__ __forceinline__ unsigned mnSlot(int sp, int k)
{
    const int sw = NCH == 8 ? (k ^ (sp & 7)) : (k ^ ((sp >> 2) & 1));
    return (unsigned)((sp * NCH + sw) << 4);
}

__device__ __forceinline__ uint4 mnLds(unsigned addr)
{
    uint4 v;
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr));
    return v;
}

__device__ __forceinline__ void mnSts(unsigned addr, uint4 v)
{
    asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" ::"r"(addr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
}

// PReLU in fp16 (the engine's LeakyRelu on an fp16 tensor: x >= 0 ? x : x * slope rounded once)
__device__ __forceinline__ unsigned mnPrelu2(unsigned u, float a)
{
    const float lo = h2f((unsigned short)(u & 0xffffu)), hi = h2f((unsigned short)(u >> 16));
    const unsigned short l = lo < 0.0f ? f2h(lo * a) : (unsigned short)(u & 0xffffu);
    const unsigned short h = hi < 0.0f ? f2h(hi * a) : (unsigned short)(u >> 16);
    return (unsigned)l | ((unsigned)h << 16);
}

// one 3 x 3 stride-1 conv cin -> cout, 128 threads, x / y channels-last fp16 at W x H; wf[z][step][q][lane] (step =
// (group * 9 + tap) * CG / 16 + kc, q = an n-tile pair), bias fp32 (8 NB a slice). RES: the halo through PReLU(slope),
// y = conv + x at the output pixel; POUT: y stored through PReLU(slopeOut); FIN (conv_out): fp32 sums, outputs 0 / 1
// -> tanh x 10 -> the fp16 planes y and y1
template <int CG, int NB, bool RES, bool POUT, bool FIN>
__device__ __forceinline__ void mnConv(const unsigned short* __restrict__ x, int cin, int H, int W,
                                       unsigned short* __restrict__ y, int cout, const uint4* __restrict__ wf,
                                       const float* __restrict__ bias, float slope, float slopeOut,
                                       unsigned short* __restrict__ y1)
{
    constexpr bool A32 = FIN;
    constexpr int TH = 8, RW = 2, MI = 4, HR = TH + 2, HC = 34;
    constexpr int NCH = CG / 8, KC = CG / 16, NQ = NB / 2;
    extern __shared__ __align__(128) unsigned char mnSm[];
    const unsigned halo = (unsigned)__cvta_generic_to_shared(mnSm);
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
            const int gy = ty0 - 1 + hr, gx = tx0 - 1 + hc;
            const bool in = gy >= 0 && gy < H && gx >= 0 && gx < W;
            const unsigned short* src = in ? x + ((size_t)gy * W + gx) * cin + gi * CG + k * 8 : x;
            rnCp16(halo + mnSlot<NCH>(p, k), src, in ? 16 : 0);
        }
        asm volatile(SMV_CP_COMMIT);
        asm volatile(SMV_CP_WAIT0);
        __syncthreads();
        if (RES)
        {
            for (int i = threadIdx.x; i < HR * HC * NCH; i += 128)
            {
                uint4 v = mnLds(halo + (unsigned)i * 16u);
                v.x = mnPrelu2(v.x, slope);
                v.y = mnPrelu2(v.y, slope);
                v.z = mnPrelu2(v.z, slope);
                v.w = mnPrelu2(v.w, slope);
                mnSts(halo + (unsigned)i * 16u, v);
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
                pa[mi] = (RW * warp + (mi >> 1) + dy) * HC + (mi & 1) * 16 + arow + dx;
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
                    rnLdm4(halo + mnSlot<NCH>(pa[mi], kc * 2 + akc), a);
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
            if (oy >= H)
                continue;
#pragma unroll
            for (int h = 0; h < 2; ++h)
            {
                const int ox = tx0 + (mi & 1) * 16 + g + h * 8;
                if (ox >= W)
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
                const size_t pix = (size_t)oy * W + ox;
                if (FIN)
                {
                    if (n0 == 0)
                    {
                        y[pix] = f2h(tanhf(v0) * 10.0f);
                        y1[pix] = f2h(tanhf(v1) * 10.0f);
                    }
                    continue;
                }
                if (RES)
                {
                    const unsigned u = *reinterpret_cast<const unsigned*>(x + pix * cin + n0);
                    v0 = __fadd_rn(v0, h2f((unsigned short)(u & 0xffffu)));
                    v1 = __fadd_rn(v1, h2f((unsigned short)(u >> 16)));
                }
                unsigned o = (unsigned)f2h(v0) | ((unsigned)f2h(v1) << 16);
                if (POUT)
                    o = mnPrelu2(o, slopeOut);
                *reinterpret_cast<unsigned*>(y + pix * cout + n0) = o;
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(128, 2)
    k_mnConv16(const unsigned short* __restrict__ x, int H, int W, unsigned short* __restrict__ y,
               const uint4* __restrict__ wf, const float* __restrict__ bias)
{
    mnConv<16, 8, false, false, false>(x, 16, H, W, y, 64, wf, bias, 0.0f, 0.0f, nullptr);
}

extern "C" __global__ void __launch_bounds__(128, 2)
    k_mnRes64(const unsigned short* __restrict__ x, int H, int W, unsigned short* __restrict__ y,
              const uint4* __restrict__ wf, const float* __restrict__ bias, float slope)
{
    mnConv<64, 8, true, false, false>(x, 64, H, W, y, 64, wf, bias, slope, 0.0f, nullptr);
}

extern "C" __global__ void __launch_bounds__(128, 2)
    k_mnRes64P(const unsigned short* __restrict__ x, int H, int W, unsigned short* __restrict__ y,
               const uint4* __restrict__ wf, const float* __restrict__ bias, float slope, float slopeOut)
{
    mnConv<64, 8, true, true, false>(x, 64, H, W, y, 64, wf, bias, slope, slopeOut, nullptr);
}

extern "C" __global__ void __launch_bounds__(128, 2)
    k_mnOut(const unsigned short* __restrict__ x, int H, int W, const uint4* __restrict__ wf,
            const float* __restrict__ bias, unsigned short* __restrict__ m0, unsigned short* __restrict__ m1)
{
    mnConv<64, 2, false, false, true>(x, 64, H, W, m0, 16, wf, bias, 0.0f, 0.0f, m1);
}
)MNSRC";
static const char* const kMnKern[] = {"k_mnIn", "k_mnConv16", "k_mnRes64", "k_mnRes64P", "k_mnOut"};
static_assert(sizeof(kMnKern) / sizeof(kMnKern[0]) == sizeof(NativeRife::fMn) / sizeof(CUfunction),
              "one fMn slot a metricnet kernel");
enum
{
    kMnIn,
    kMnC16,
    kMnRes,
    kMnResP,
    kMnOut
};
// a conv kernel's dynamic shared memory: the 8 x 32 tile's halo of its cg input channels
static constexpr int mnSm(int cg)
{
    return 10 * 34 * (cg / 8) * 16;
}

// metric.pkl in the strict MetricNet form (metric_in [64][14][3][3]; metric_net1 .. 3 = {a scalar PReLU, a conv [64]
// [64][3][3]}; metric_out = {a scalar PReLU, a conv [2][64][3][3]}; their biases) as one weight block of 256-byte aligned
// pieces: off[0] / off[1] conv_in's fragments (its 14 inputs zero-padded to 16) and bias, off[2 + 2 r] / off[3 + 2 r]
// ResConv r's, off[8] / off[9] conv_out's (padded to 16 outputs); slope[0 .. 2] the ResConvs' PReLU slopes, slope[3]
// conv_out's. false = missing, or not this net (`why`)
static bool mnNatLoad(std::vector<uint8_t>& blob, size_t* off, float* slope, std::string& why)
{
    PthFile pf;
    if (!pthStateDict(g_nativeEngineDir + L"\\GMFSS_Fortuna\\train_log\\metric.pkl", pf, why))
        return false;
    auto& wts = pf.t;
    const auto& S = pf.S;
    auto has = [&](const std::string& n, const std::vector<int64_t>& s) {
        const auto it = S.find(n);
        return it != S.end() && it->second == s;
    };
    bool form = has("metric_in.weight", {64, 14, 3, 3}) && has("metric_in.bias", {64}) &&
                has("metric_out.0.weight", {1}) && has("metric_out.1.weight", {2, 64, 3, 3}) &&
                has("metric_out.1.bias", {2}) && !S.count("metric_net4.0.weight");
    for (int r = 1; r <= 3 && form; r++)
    {
        const std::string p = "metric_net" + std::to_string(r) + ".";
        form = has(p + "0.weight", {1}) && has(p + "1.weight", {64, 64, 3, 3}) && has(p + "1.bias", {64});
    }
    if (!form)
    {
        why = "its weights are not GMFSS's MetricNet";
        return false;
    }
    blob.clear();
    auto put = [&](const void* d, size_t n) {
        const size_t o = (blob.size() + 255) & ~(size_t)255;
        blob.resize(o + n);
        memcpy(blob.data() + o, d, n);
        return o;
    };
    const std::span<const float> w0 = wts["metric_in.weight"];
    std::vector<float> w0p((size_t)64 * 16 * 9, 0.0f);
    for (int n = 0; n < 64; n++)
        for (int i = 0; i < 14; i++)
            for (int t = 0; t < 9; t++)
                w0p[((size_t)n * 16 + i) * 9 + t] = w0[((size_t)n * 14 + i) * 9 + t];
    const std::vector<uint16_t> f0 = fnFragOf(w0p, 64, 16, 16, 8);
    off[0] = put(f0.data(), f0.size() * sizeof(uint16_t));
    off[1] = put(wts["metric_in.bias"].data(), 64 * sizeof(float));
    for (int r = 0; r < 3; r++)
    {
        const std::string p = "metric_net" + std::to_string(r + 1) + ".";
        const std::vector<uint16_t> fr = fnFragOf(wts[p + "1.weight"], 64, 64, 64, 8);
        off[2 + 2 * r] = put(fr.data(), fr.size() * sizeof(uint16_t));
        off[3 + 2 * r] = put(wts[p + "1.bias"].data(), 64 * sizeof(float));
        slope[r] = wts[p + "0.weight"][0];
    }
    const std::span<const float> wo = wts["metric_out.1.weight"], bo = wts["metric_out.1.bias"];
    std::vector<float> wop((size_t)16 * 64 * 9, 0.0f), bop(16, 0.0f);
    std::copy(wo.begin(), wo.end(), wop.begin());
    std::copy(bo.begin(), bo.end(), bop.begin());
    const std::vector<uint16_t> fo = fnFragOf(wop, 16, 64, 64, 2);
    off[8] = put(fo.data(), fo.size() * sizeof(uint16_t));
    off[9] = put(bop.data(), bop.size() * sizeof(float));
    slope[3] = wts["metric_out.0.weight"][0];
    return true;
}

// metricnet's scratch at the half W x H, 256-byte pieces: the [H][W][16] input, one [H][W][64] ping-pong pair (fp16)
struct MnArena
{
    size_t x = 0, fa = 0, fb = 0, total = 0;
};
static MnArena mnArena(int w, int h)
{
    MnArena a;
    const size_t n = (size_t)w * (size_t)h;
    size_t o = 0;
    auto put = [&](size_t b) {
        const size_t at = o;
        o += (b + 255) & ~(size_t)255;
        return at;
    };
    a.x = put(16 * n * sizeof(uint16_t));
    a.fa = put(64 * n * sizeof(uint16_t));
    a.fb = put(64 * n * sizeof(uint16_t));
    a.total = o;
    return a;
}

static size_t mnNatBytes(const NativeRife& nr)
{
    return nr.gmfss && nr.hw > 0 && nr.hh > 0 ? mnArena(nr.hw, nr.hh).total : 0;
}

// this session's metricnet module: the weights read, its text compiled (or the cached cubin) and loaded, its kernels
// bound, the weight block uploaded; false = `why` (the session teardown unloads a module left in mnMod)
static bool nativeMnModule(NativeRife& nr, std::string& why)
{
    if (nr.mnMod)
        return true;
    std::vector<uint8_t> blob;
    if (!mnNatLoad(blob, nr.mnOff, nr.mnSlope, why))
        return false;
    std::string text = kFnHead;
    text += kMnText;
    std::vector<char> cubin;
    if (!nativeCubin(text.c_str(), "smv_metric.cu", L"smv_metric", "metricnet kernels",
                     nr.kernCacheDir.empty() ? std::wstring(L".") : nr.kernCacheDir, cubin))
    {
        why = "its kernels did not compile";
        return false;
    }
    if (cuModuleLoadData(&nr.mnMod, cubin.data()) != CUDA_SUCCESS)
    {
        nr.mnMod = nullptr;
        why = "its kernel module did not load";
        return false;
    }
    for (int i = 0; i < (int)(sizeof(kMnKern) / sizeof(kMnKern[0])); i++)
        if (cuModuleGetFunction(&nr.fMn[i], nr.mnMod, kMnKern[i]) != CUDA_SUCCESS)
        {
            why = std::string("its kernel ") + kMnKern[i] + " is missing";
            return false;
        }
    if (cudaMalloc(&nr.dMnW, blob.size()) != cudaSuccess ||
        cudaMemcpy(nr.dMnW, blob.data(), blob.size(), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        why = "its weights did not upload";
        return false;
    }
    return true;
}

// metricnet on one pair on st (nr.stream) at the half nr.hw x nr.hh: i0 / i1 = the motion halves (3 fp32 planes),
// f01 / f10 = the flows (2 fp32 planes); m0 / m1 = its outputs (an fp16 plane each). false = the scratch or a launch
// failed
static bool nativeMnRun(NativeRife& nr, const float* i0, const float* i1, const float* f01, const float* f10, void* m0,
                        void* m1, cudaStream_t st)
{
    int w = nr.hw, h = nr.hh;
    const MnArena ar = mnArena(w, h);
    uint8_t* s = nr.dGmCtxMem && (size_t)nr.gmCtxMemB >= ar.total ? (uint8_t*)nr.dGmCtxMem : (uint8_t*)nr.dMnOwn;
    if (!s && cudaMalloc(&nr.dMnOwn, ar.total) == cudaSuccess)
        s = (uint8_t*)nr.dMnOwn;
    if (!s)
        return false;
    void* x = s + ar.x;
    void* fab[2] = {s + ar.fa, s + ar.fb};
    const char* wb = (const char*)nr.dMnW;
    auto go = [&](int f, int gx, int gy, int sm, void** a) {
        return cuLaunchKernel(nr.fMn[f], (unsigned)gx, (unsigned)gy, 1, 128, 1, 1, (unsigned)sm, (CUstream)st, a,
                              nullptr) == CUDA_SUCCESS;
    };
    {
        void* a[] = {(void*)&i0, (void*)&i1, (void*)&f01, (void*)&f10, &w, &h, &x};
        if (!go(kMnIn, (w + 127) / 128, h, 0, a))
            return false;
    }
    const int gx = (w + 31) / 32, gy = (h + 7) / 8;
    {
        const void* cw = wb + nr.mnOff[0];
        const void* cb = wb + nr.mnOff[1];
        void* a[] = {&x, &h, &w, &fab[0], (void*)&cw, (void*)&cb};
        if (!go(kMnC16, gx, gy, mnSm(16), a))
            return false;
    }
    for (int r = 0; r < 3; r++)
    {
        const void* cw = wb + nr.mnOff[2 + 2 * r];
        const void* cb = wb + nr.mnOff[3 + 2 * r];
        float sl = nr.mnSlope[r], so = nr.mnSlope[3];
        void* a[] = {&fab[r & 1], &h, &w, &fab[(r + 1) & 1], (void*)&cw, (void*)&cb, &sl, &so};
        if (!go(r < 2 ? kMnRes : kMnResP, gx, gy, mnSm(64), a))
            return false;
    }
    {
        const void* ow = wb + nr.mnOff[8];
        const void* ob = wb + nr.mnOff[9];
        void* a[] = {&fab[1], &h, &w, (void*)&ow, (void*)&ob, &m0, &m1};
        if (!go(kMnOut, gx, gy, mnSm(64), a))
            return false;
    }
    return true;
}
