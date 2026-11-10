// part of smv-live-native.inl (included there, in place); not a standalone translation unit
// ---- DRBA's block0 (calc_flow's first pyramid level) on the host's kernels -----------------------------------------
// RIFE 4.25's IFBlock(15, c = 192) at scale 16 (calc_flow's first level), weights from engine\rife\flownet.pkl
// (the host's torch zip reader). The motion frame is W x H (a multiple of 64); the levels: L0 =
// W / 16 x H / 16 (the block's input and its flow), L1 = W / 32 x H / 32, L2 = W / 64 x H / 64. Its own NVRTC module
// (kFnHead's helpers + kB0In + kB0Conv):
// k_b0In   cat(img0, img1, f0, f1, 0.5) at L0 by PyTorch's bilinear (align_corners = False): at an exact 1 / 16 each
//          sample is the mean of the 2 x 2 source pixels at rows / columns 16 i + 7, 16 i + 8; 15 fp32 planes
// k_b0C0   conv 15 -> 96, 3 x 3 stride 2, + LeakyRelu(0.2): fp32 FMAs, one rounding to channels-last fp16 at L1
// k_b0C1   conv 96 -> 192 stride 2 + LeakyRelu, and k_b0Res the 8 ResConvs {LeakyRelu(conv(x) * beta + x)} at L2: an
//          implicit GEMM on the tensor cores (mma.sync m16n8k16, fp32 accumulators started at the bias; M = 16-pixel
//          row runs of a 4 x 32 output tile, N = 16 outputs a block, K = 9 taps x the input channels CG at a time:
//          each group's halo (16-byte chunks XOR-swizzled so each ldmatrix reads 8 bank groups) and weights in shared
//          memory, one cp.async wait a group); the residual stream R in fp32 channels-last beside the fp16 copy the
//          next conv reads
// k_b0Last ConvTranspose 192 -> 52 (4 x 4, stride 2, pad 1) for the 16 channels PixelShuffle(2) turns into the 4
//          flow planes: fp32 FMAs on R, the 192 inputs split over a block's 8 warps, the partial sums added in a
//          fixed order; the 4 planes at L0. The flow at W x H = these planes by PyTorch's bilinear, times 16: the one
//          reader, DRBA's flow splat, samples it there (drbaFlowAt), so the full-size flow is never written.
static const int kB0SmC1 = 9 * 65 * 4 * 16 + 9 * 2 * 32 * 16;  // k_b0C1's halo + one group's weights, bytes
static const int kB0SmRes = 6 * 34 * 8 * 16 + 9 * 4 * 32 * 16; // k_b0Res's
static const char* const kB0Kern[5] = {"k_b0In", "k_b0C0", "k_b0C1", "k_b0Res", "k_b0Last"};

static const char kB0In[] = R"B0SRC(
__device__ __forceinline__ float b0Lr(float v)
{
    return v < 0.0f ? v * 0.2f : v;
}

__device__ __forceinline__ float b0Ld(const void* p, size_t i, int half)
{
    return half ? h2f(reinterpret_cast<const unsigned short*>(p)[i]) : reinterpret_cast<const float*>(p)[i];
}

// out = 15 fp32 planes at L0; img0 / img1 = 3 planes (fp16 when imgHalf), f0 / f1 = 4 planes (fp16 when featHalf)
extern "C" __global__ void __launch_bounds__(256) k_b0In(const void* __restrict__ img0, const void* __restrict__ img1,
                                                         int imgHalf, const void* __restrict__ f0,
                                                         const void* __restrict__ f1, int featHalf, int W, int H,
                                                         float* __restrict__ out)
{
    const int w16 = W >> 4, h16 = H >> 4;
    const int x = blockIdx.x * 32 + (threadIdx.x & 31), y = blockIdx.y * 8 + (threadIdx.x >> 5);
    if (x >= w16 || y >= h16)
        return;
    const size_t plane = (size_t)W * H, po = (size_t)w16 * h16, o = (size_t)y * w16 + x;
    const size_t s = (size_t)(16 * y + 7) * W + 16 * x + 7;
#pragma unroll
    for (int c = 0; c < 14; ++c)
    {
        const void* src = c < 3 ? img0 : c < 6 ? img1 : c < 10 ? f0 : f1;
        const int k = c < 3 ? c : c < 6 ? c - 3 : c < 10 ? c - 6 : c - 10, hf = c < 6 ? imgHalf : featHalf;
        const size_t b = (size_t)k * plane + s;
        const float a0 = b0Ld(src, b, hf), a1 = b0Ld(src, b + 1, hf);
        const float a2 = b0Ld(src, b + W, hf), a3 = b0Ld(src, b + W + 1, hf);
        out[c * po + o] = 0.5f * (0.5f * a0 + 0.5f * a1) + 0.5f * (0.5f * a2 + 0.5f * a3);
    }
    out[14 * po + o] = 0.5f;
}

// one L1 pixel and 24 outputs a thread (blockIdx.z = the 24), the z's weights staged in shared memory; in = the 15
// planes at L0 (w16 x h16); y = channels-last fp16 [h16 / 2][w16 / 2][96]; wb = w[96][15][9] then b[96], fp32
extern "C" __global__ void __launch_bounds__(256) k_b0C0(const float* __restrict__ in, int w16, int h16,
                                                         unsigned short* __restrict__ y, const float* __restrict__ wb)
{
    __shared__ float sw[24 * 135 + 24];
    const int z = blockIdx.z;
    for (int i = threadIdx.x; i < 24 * 135; i += 256)
        sw[i] = wb[z * 24 * 135 + i];
    if (threadIdx.x < 24)
        sw[24 * 135 + threadIdx.x] = wb[96 * 135 + z * 24 + threadIdx.x];
    __syncthreads();
    const int OW = w16 >> 1, OH = h16 >> 1;
    const int ox = blockIdx.x * 32 + (threadIdx.x & 31), oy = blockIdx.y * 8 + (threadIdx.x >> 5);
    if (ox >= OW || oy >= OH)
        return;
    const size_t po = (size_t)w16 * h16;
    float acc[24];
#pragma unroll
    for (int o = 0; o < 24; ++o)
        acc[o] = sw[24 * 135 + o];
#pragma unroll 1
    for (int c = 0; c < 15; ++c)
    {
        float v[9];
#pragma unroll
        for (int t = 0; t < 9; ++t)
        {
            const int gy = 2 * oy + t / 3 - 1, gx = 2 * ox + t % 3 - 1;
            v[t] = (gy >= 0 && gy < h16 && gx >= 0 && gx < w16) ? in[c * po + (size_t)gy * w16 + gx] : 0.0f;
        }
#pragma unroll
        for (int o = 0; o < 24; ++o)
#pragma unroll
            for (int t = 0; t < 9; ++t)
                acc[o] = fmaf(sw[o * 135 + c * 9 + t], v[t], acc[o]);
    }
    unsigned r[12];
#pragma unroll
    for (int o = 0; o < 24; o += 2)
        r[o / 2] = (unsigned)f2h(b0Lr(acc[o])) | ((unsigned)f2h(b0Lr(acc[o + 1])) << 16);
    uint4* d = reinterpret_cast<uint4*>(y + ((size_t)oy * OW + ox) * 96 + z * 24);
    d[0] = make_uint4(r[0], r[1], r[2], r[3]);
    d[1] = make_uint4(r[4], r[5], r[6], r[7]);
    d[2] = make_uint4(r[8], r[9], r[10], r[11]);
}
)B0SRC";

static const char kB0Conv[] = R"B0SRC(
// the stored index of halo pixel sp and its chunk k's swizzled slot
template <int NCH>
__device__ __forceinline__ unsigned b0Slot(int sp, int k)
{
    const int sw = NCH == 8 ? (k ^ (sp & 7)) : (k ^ ((sp >> 1) & 3));
    return (unsigned)((sp * NCH + sw) << 4);
}

__device__ __forceinline__ uint4 b0Lds(unsigned addr)
{
    uint4 v;
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr));
    return v;
}

// one conv, 128 threads: x = cin channels-last fp16 at W x H; y = cout channels-last fp16 at W / S x H / S (null =
// skipped); R = the same values in fp32 channels-last (RES: read, then overwritten in place: each value is read and
// written by the one thread that owns it); wf[z][step][lane] (step = (group * 9 + tap) * CG / 16 + kc, the B
// registers of the z's two n-tiles); bias, beta fp32. The 4 x 32 output tile's CG-channel halo (a stride-2 halo row
// keeps its even offsets before its odd ones, so a tap's 8 rows sit next to each other) and the group's weights are
// loaded by cp.async into shared memory, one wait a group
template <int S, int CG, bool RES>
__device__ __forceinline__ void b0ConvT(const unsigned short* __restrict__ x, int cin, int H, int W,
                                        unsigned short* __restrict__ y, int cout, const uint4* __restrict__ wf,
                                        const float* __restrict__ bias, const float* __restrict__ beta,
                                        float* __restrict__ R)
{
    constexpr int TH = 4, NB = 2;
    constexpr int HR = S == 1 ? TH + 2 : 2 * TH + 1, HC = S == 1 ? 34 : 65;
    constexpr int NCH = CG / 8, KC = CG / 16;
    constexpr int HALO = HR * HC * NCH * 16, GQ = 9 * KC * 32;
    extern __shared__ __align__(128) unsigned char b0Sm[];
    const unsigned halo = (unsigned)__cvta_generic_to_shared(b0Sm), wsm = halo + HALO;
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
            rnCp16(halo + b0Slot<NCH>(sp, k), src, in ? 16 : 0);
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
                const uint4 bw = b0Lds(wsm + (unsigned)((tap * KC + kc) * 32 + lane) * 16u);
#pragma unroll
                for (int mi = 0; mi < 2; ++mi)
                {
                    unsigned a[4];
                    rnLdm4(halo + b0Slot<NCH>(pa[mi], kc * 2 + akc), a);
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
                float2* rp = reinterpret_cast<float2*>(R + pix * cout + n0);
                if (RES)
                {
                    const float2 r = *rp;
                    v0 = __fadd_rn(__fmul_rn(v0, be0), r.x);
                    v1 = __fadd_rn(__fmul_rn(v1, be1), r.y);
                }
                v0 = b0Lr(v0);
                v1 = b0Lr(v1);
                *rp = make_float2(v0, v1);
                if (y)
                    *reinterpret_cast<unsigned*>(y + pix * cout + n0) =
                        (unsigned)f2h(v0) | ((unsigned)f2h(v1) << 16);
            }
        }
    }
}

// conv 96 -> 192 stride 2 + LeakyRelu (L1 -> L2) and one ResConv (L2 -> L2); grid (W / S / 32, H / S / 4, 12),
// shared memory kB0SmC1 / kB0SmRes
extern "C" __global__ void __launch_bounds__(128, 2)
    k_b0C1(const unsigned short* __restrict__ x, int cin, int H, int W, unsigned short* __restrict__ y, int cout,
           const uint4* __restrict__ wf, const float* __restrict__ bias, const float* __restrict__ beta,
           float* __restrict__ R)
{
    b0ConvT<2, 32, false>(x, cin, H, W, y, cout, wf, bias, beta, R);
}
extern "C" __global__ void __launch_bounds__(128, 2)
    k_b0Res(const unsigned short* __restrict__ x, int cin, int H, int W, unsigned short* __restrict__ y, int cout,
            const uint4* __restrict__ wf, const float* __restrict__ bias, const float* __restrict__ beta,
            float* __restrict__ R)
{
    b0ConvT<1, 64, true>(x, cin, H, W, y, cout, wf, bias, beta, R);
}

// One L1 pixel (Y, X) = (2 yy + py, 2 xx + px) and its 4 channels of flow plane c, blockIdx.z = (c, py, px): such a
// pixel takes kernel rows ky in {1, 3} (py 0, input rows yy, yy - 1) or {0, 2} (py 1, rows yy + 1, yy), the columns
// the same way; PixelShuffle(2) (CRD) puts channel 4c + 2 dy + dx at (Y, X) on plane c at (2 Y + dy, 2 X + dx). A
// block = one L2 row yy, 32 columns xx, the 192 inputs split over its 8 warps (24 each). R = the residual stream
// [h64][w64][192] fp32; out = 4 fp32 planes at L0; wl[z][192][2 rows][2 columns][4 channels] then b[16], fp32
extern "C" __global__ void __launch_bounds__(256) k_b0Last(const float* __restrict__ R, int w64, int h64,
                                                           float* __restrict__ out, const float* __restrict__ wl)
{
    __shared__ float4 sw[192 * 4];
    __shared__ float4 red[8][32];
    const int z = blockIdx.z, c = z >> 2, py = (z >> 1) & 1, px = z & 1;
    const float4* src = reinterpret_cast<const float4*>(wl + (size_t)z * 192 * 16);
    for (int i = threadIdx.x; i < 192 * 4; i += 256)
        sw[i] = src[i];
    __syncthreads();
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    const int xx = blockIdx.x * 32 + lane, yy = blockIdx.y;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (xx < w64)
#pragma unroll
        for (int ti = 0; ti < 2; ++ti)
        {
            const int iy = yy + (py ? 1 - ti : -ti);
            if (iy < 0 || iy >= h64)
                continue;
#pragma unroll
            for (int tj = 0; tj < 2; ++tj)
            {
                const int ix = xx + (px ? 1 - tj : -tj);
                if (ix < 0 || ix >= w64)
                    continue;
                const float4* r = reinterpret_cast<const float4*>(R + ((size_t)iy * w64 + ix) * 192) + wp * 6;
                const int tp = ti * 2 + tj;
                float4 v[6];
#pragma unroll
                for (int c4 = 0; c4 < 6; ++c4)
                    v[c4] = r[c4];
#pragma unroll
                for (int c4 = 0; c4 < 6; ++c4)
                {
                    const float vv[4] = {v[c4].x, v[c4].y, v[c4].z, v[c4].w};
#pragma unroll
                    for (int k = 0; k < 4; ++k)
                    {
                        const float4 w = sw[((wp * 6 + c4) * 4 + k) * 4 + tp];
                        acc[0] = fmaf(vv[k], w.x, acc[0]);
                        acc[1] = fmaf(vv[k], w.y, acc[1]);
                        acc[2] = fmaf(vv[k], w.z, acc[2]);
                        acc[3] = fmaf(vv[k], w.w, acc[3]);
                    }
                }
            }
        }
    red[wp][lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);
    __syncthreads();
    if (threadIdx.x >= 128)
        return;
    const int j = threadIdx.x >> 5, px2 = threadIdx.x & 31, x2 = blockIdx.x * 32 + px2;
    if (x2 >= w64)
        return;
    const float* bias = wl + (size_t)16 * 192 * 16;
    float s = reinterpret_cast<const float*>(&red[0][px2])[j];
#pragma unroll
    for (int k = 1; k < 8; ++k)
        s += reinterpret_cast<const float*>(&red[k][px2])[j];
    s += bias[4 * c + j];
    const int w16 = w64 * 4, Y = 2 * yy + py, X = 2 * x2 + px;
    out[(size_t)c * w16 * (h64 * 4) + (size_t)(2 * Y + (j >> 1)) * w16 + 2 * X + (j & 1)] = s;
}
)B0SRC";

// flownet.pkl's block0 in the strict IFBlock(15, c = 192) form (module.block0.: conv0.0.0 [96, 15, 3, 3], conv0.1.0
// [192, 96, 3, 3], convblock.0 .. 7 {conv [192, 192, 3, 3], beta [1, 192, 1, 1]}, lastconv.0 [192, 52, 4, 4], their
// biases) as one weight block of 256-byte aligned pieces: off[0] conv 0's weights then bias (fp32), off[1] / off[2]
// conv 1's fragment-order weights / bias, off[3 + 3 i] / [4 + 3 i] / [5 + 3 i] ResConv i's weights / bias / beta,
// off[27] the lastconv per (plane, row parity, column parity) as [192][2][2][4] (parity 0 = kernel taps 1, 3, parity
// 1 = 0, 2) then the 16 biases (fp32). false = missing, or not this net (`why`).
static bool b0NatLoad(std::vector<uint8_t>& blob, size_t* off, std::string& why)
{
    PthFile pf;
    if (!pthStateDict(g_nativeEngineDir + L"\\rife\\flownet.pkl", pf, why))
        return false;
    auto& wts = pf.t;
    const auto& S = pf.S;
    const std::string p = "module.block0.";
    auto has = [&](const std::string& n, const std::vector<int64_t>& s) {
        const auto it = S.find(p + n);
        return it != S.end() && it->second == s;
    };
    bool form = has("conv0.0.0.weight", {96, 15, 3, 3}) && has("conv0.0.0.bias", {96}) &&
                has("conv0.1.0.weight", {192, 96, 3, 3}) && has("conv0.1.0.bias", {192}) &&
                has("lastconv.0.weight", {192, 52, 4, 4}) && has("lastconv.0.bias", {52}) &&
                !S.count(p + "convblock.8.beta");
    for (int i = 0; i < 8 && form; i++)
    {
        const std::string c = "convblock." + std::to_string(i) + ".";
        form =
            has(c + "conv.weight", {192, 192, 3, 3}) && has(c + "conv.bias", {192}) && has(c + "beta", {1, 192, 1, 1});
    }
    if (!form)
    {
        why = "its weights are not RIFE 4.25's block0";
        return false;
    }
    blob.clear();
    auto put = [&](const void* d, size_t n) {
        const size_t o = (blob.size() + 255) & ~(size_t)255;
        blob.resize(o + n);
        memcpy(blob.data() + o, d, n);
        return o;
    };
    auto w = [&](const std::string& n) -> std::span<const float> { return wts[p + n]; };
    const std::span<const float> c0w = w("conv0.0.0.weight"), c0b = w("conv0.0.0.bias");
    std::vector<float> c0(c0w.begin(), c0w.end());
    c0.insert(c0.end(), c0b.begin(), c0b.end());
    off[0] = put(c0.data(), c0.size() * sizeof(float));
    const std::vector<uint16_t> f1 = fnFragOf(w("conv0.1.0.weight"), 192, 96, 32, 2);
    off[1] = put(f1.data(), f1.size() * sizeof(uint16_t));
    off[2] = put(w("conv0.1.0.bias").data(), 192 * sizeof(float));
    for (int i = 0; i < 8; i++)
    {
        const std::string c = "convblock." + std::to_string(i) + ".";
        const std::vector<uint16_t> fr = fnFragOf(w(c + "conv.weight"), 192, 192, 64, 2);
        off[3 + 3 * i] = put(fr.data(), fr.size() * sizeof(uint16_t));
        off[4 + 3 * i] = put(w(c + "conv.bias").data(), 192 * sizeof(float));
        off[5 + 3 * i] = put(w(c + "beta").data(), 192 * sizeof(float));
    }
    static const int kTap[2][2] = {{1, 3}, {0, 2}};
    const std::span<const float> lw = w("lastconv.0.weight");
    const std::span<const float> lb = w("lastconv.0.bias");
    std::vector<float> lp;
    lp.reserve((size_t)16 * 192 * 16 + 16);
    for (int z = 0; z < 16; z++)
    {
        const int c = z >> 2, py = (z >> 1) & 1, px = z & 1;
        for (int ci = 0; ci < 192; ci++)
            for (int ti = 0; ti < 2; ti++)
                for (int tj = 0; tj < 2; tj++)
                    for (int j = 0; j < 4; j++)
                        lp.push_back(lw[(((size_t)ci * 52 + 4 * c + j) * 4 + kTap[py][ti]) * 4 + kTap[px][tj]]);
    }
    lp.insert(lp.end(), lb.begin(), lb.begin() + 16);
    off[27] = put(lp.data(), lp.size() * sizeof(float));
    return true;
}

// block0's scratch at the motion frame (M x N): the 15 input planes and the 4 flow planes at L0 (fp32), conv 0's
// output at L1 (96 fp16), the ping-pong at L2 (2 x 192 fp16) and the residual stream (192 fp32), 256-byte pieces in
// that order; 0 without DRBA
static size_t b0NatPiece(size_t b)
{
    return (b + 255) & ~(size_t)255;
}
static size_t b0NatBytes(const NativeRife& nr)
{
    if (!nr.drba)
        return 0;
    const size_t w = (size_t)(nr.mph ? nr.mpw : nr.pw), h = (size_t)(nr.mph ? nr.mph : nr.ph);
    const size_t n16 = (w / 16) * (h / 16), n32 = (w / 32) * (h / 32), n64 = (w / 64) * (h / 64);
    return b0NatPiece(15 * n16 * 4) + b0NatPiece(4 * n16 * 4) + b0NatPiece(96 * n32 * 2) +
           2 * b0NatPiece(192 * n64 * 2) + b0NatPiece(192 * n64 * 4);
}

// this session's block0 module: the weights read, its text compiled (or the cached cubin) and loaded, its five kernels
// bound, the weight block uploaded; false = `why` (the session teardown unloads a module left in b0Mod)
static bool nativeB0Module(NativeRife& nr, std::string& why)
{
    if (nr.b0Mod)
        return true;
    std::vector<uint8_t> blob;
    if (!b0NatLoad(blob, nr.b0Off, why))
        return false;
    std::string text = kFnHead;
    text += kB0In;
    text += kB0Conv;
    std::vector<char> cubin;
    if (!nativeCubin(text.c_str(), "smv_block0.cu", L"smv_block0", "block0 kernels",
                     nr.kernCacheDir.empty() ? std::wstring(L".") : nr.kernCacheDir, cubin))
    {
        why = "its kernels did not compile";
        return false;
    }
    if (cuModuleLoadData(&nr.b0Mod, cubin.data()) != CUDA_SUCCESS)
    {
        nr.b0Mod = nullptr;
        why = "its kernel module did not load";
        return false;
    }
    for (int i = 0; i < 5; i++)
        if (cuModuleGetFunction(&nr.fB0[i], nr.b0Mod, kB0Kern[i]) != CUDA_SUCCESS)
        {
            why = std::string("its kernel ") + kB0Kern[i] + " is missing";
            return false;
        }
    if (cudaMalloc(&nr.dB0W, blob.size()) != cudaSuccess ||
        cudaMemcpy(nr.dB0W, blob.data(), blob.size(), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        why = "its weights did not upload";
        return false;
    }
    return true;
}

// block0 on one frame pair on st (nr.stream: its scratch shares the activation block with the contexts that enqueue
// there): img0 / img1 = 3 planes at the motion frame fw x fh (fp16 when imgHalf), f0 / f1 their encodes in the IFNet's
// feature dtype; *lo = its 4 flow planes at L0 in that scratch, for the splat that follows on the same stream. false =
// the scratch or a launch failed
static bool nativeB0Run(NativeRife& nr, const void* img0, const void* img1, int imgHalf, const void* f0, const void* f1,
                        int fw, int fh, float** lo, cudaStream_t st)
{
    const size_t need = b0NatBytes(nr);
    uint8_t* s = nr.dGmCtxMem && (size_t)nr.gmCtxMemB >= need ? (uint8_t*)nr.dGmCtxMem : (uint8_t*)nr.dB0Own;
    if (!s && cudaMalloc(&nr.dB0Own, need) == cudaSuccess)
        s = (uint8_t*)nr.dB0Own;
    if (!s)
        return false;
    int w16 = fw / 16, h16 = fh / 16, w32 = fw / 32, h32 = fh / 32, w64 = fw / 64, h64 = fh / 64;
    const size_t n16 = (size_t)w16 * h16, n32 = (size_t)w32 * h32, n64 = (size_t)w64 * h64;
    float* in = (float*)s;
    float* flo = (float*)(s + b0NatPiece(15 * n16 * 4));
    unsigned short* y0 = (unsigned short*)((uint8_t*)flo + b0NatPiece(4 * n16 * 4));
    unsigned short* ya = (unsigned short*)((uint8_t*)y0 + b0NatPiece(96 * n32 * 2));
    unsigned short* yb = (unsigned short*)((uint8_t*)ya + b0NatPiece(192 * n64 * 2));
    float* R = (float*)((uint8_t*)yb + b0NatPiece(192 * n64 * 2));
    const char* wb = (const char*)nr.dB0W;
    const float* c0w = (const float*)(wb + nr.b0Off[0]);
    const float* lw = (const float*)(wb + nr.b0Off[27]);
    int featHalf = nr.featHalf ? 1 : 0;
    void* p0[] = {(void*)&img0, (void*)&img1, &imgHalf, (void*)&f0, (void*)&f1, &featHalf, &fw, &fh, &in};
    void* p1[] = {&in, &w16, &h16, &y0, (void*)&c0w};
    if (cuLaunchKernel(nr.fB0[0], (unsigned)(w16 + 31) / 32, (unsigned)(h16 + 7) / 8, 1, 256, 1, 1, 0, (CUstream)st, p0,
                       nullptr) != CUDA_SUCCESS ||
        cuLaunchKernel(nr.fB0[1], (unsigned)(w32 + 31) / 32, (unsigned)(h32 + 7) / 8, 4, 256, 1, 1, 0, (CUstream)st, p1,
                       nullptr) != CUDA_SUCCESS)
        return false;
    const unsigned gx = (unsigned)(w64 + 31) / 32, gy = (unsigned)(h64 + 3) / 4;
    int c96 = 96, c192 = 192;
    float* const nulF = nullptr;
    unsigned short* const nulH = nullptr;
    {
        const uint4* wf = (const uint4*)(wb + nr.b0Off[1]);
        const float* bias = (const float*)(wb + nr.b0Off[2]);
        void* a[] = {&y0, &c96, &h32, &w32, &ya, &c192, (void*)&wf, (void*)&bias, (void*)&nulF, &R};
        if (cuLaunchKernel(nr.fB0[2], gx, gy, 12, 128, 1, 1, kB0SmC1, (CUstream)st, a, nullptr) != CUDA_SUCCESS)
            return false;
    }
    for (int i = 0; i < 8; i++)
    {
        unsigned short* x = (i & 1) ? yb : ya;
        unsigned short* y = i == 7 ? nulH : (i & 1) ? ya : yb;
        const uint4* wf = (const uint4*)(wb + nr.b0Off[3 + 3 * i]);
        const float* bias = (const float*)(wb + nr.b0Off[4 + 3 * i]);
        const float* beta = (const float*)(wb + nr.b0Off[5 + 3 * i]);
        void* a[] = {&x, &c192, &h64, &w64, &y, &c192, (void*)&wf, (void*)&bias, (void*)&beta, &R};
        if (cuLaunchKernel(nr.fB0[3], gx, gy, 12, 128, 1, 1, kB0SmRes, (CUstream)st, a, nullptr) != CUDA_SUCCESS)
            return false;
    }
    void* pl[] = {&R, &w64, &h64, &flo, (void*)&lw};
    if (cuLaunchKernel(nr.fB0[4], gx, (unsigned)h64, 16, 256, 1, 1, 0, (CUstream)st, pl, nullptr) != CUDA_SUCCESS)
        return false;
    *lo = flo;
    return true;
}
