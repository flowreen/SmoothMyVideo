// part of smv-live-native.inl (included there, in place); not a standalone translation unit
// ---- ArtCNN (`--artcnn`): C4F32_DN v1.6.2 (Artoriuz, MIT) in RTX VSR's resize, both routes ----
// kAcnA + kAcnB = the module text exactly as its kernel kit gated it (a splice script writes them and checks them after
// every clang-format; never edit the raw strings by hand): the luma net's four kernels (conv0, the 32 -> 32 body on
// the tensor cores with fp16 sums, conv6 with the DepthToSpace in its store) + k_acnY / k_acnMerge. The kit gates the
// text (acn_c4_gate.py: the net vs the author's ONNX in fp64, tiles vs whole frames byte for byte).
static const char kAcnA[] = R"ACNSRC(
// ArtCNN C4F32_DN v1.6.2 (Artoriuz, MIT) on the tensor cores, the host's own module text: the luma x2. conv0 1 -> 32,
// conv1 .. conv4 32 -> 32 + ReLU, conv5 32 -> 32 + conv0's output, conv6 32 -> 4 = DepthToSpace DCR 2, clipped 0..1;
// every conv 3 x 3, zero pad 1, + bias. Activations channels-last fp16 (unsigned short). Body conv = FP16 SUMS on
// mma.sync m16n8k16 (sm_75: its two k8 halves, SMV_MMA_F16): a block = 4 warps on a 4 x 32 pixel tile (a warp = one
// 32-pixel row = two m16 tiles), its 6 x 34 halo the only shared memory, each pixel at 80 bytes (64 + a 16-byte pad: the
// 8 ldmatrix rows on 8 bank groups); the weights from global memory in fragment order (wf[step][q][lane] = one uint4 =
// the B registers of n-tiles 2q, 2q + 1), one k-step ahead. EPI 2 = + skip (skip may BE y: each value is read and
// written by the same thread), 3 = ReLU. First conv (one fp32 plane -> 32) and last conv (32 -> 4, the DepthToSpace
// folded into the store, clipped) = one pixel a thread, fp32 sums; both take a TILE: the first reads w x h at src (row
// stride rs), zero outside the tile; the last writes only the core (cx0, cy0, cw x ch of the tile) at out. Colour =
// the author's Inferencer\utils.py (BT.709 full range): the luma from the slot's frame, the colour from its Lanczos
// enlargement (k_acnMerge).
#if __CUDA_ARCH__ >= 800
#define SMV_MMA_F16 "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
#define SMV_CP_COMMIT "cp.async.commit_group;\n"
#define SMV_CP_WAIT0 "cp.async.wait_group 0;\n"
#else
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
#define ACN_C 32
#define ACN_TW 32
#define ACN_NW 4
#define ACN_HH (ACN_NW + 2)
#define ACN_HW (ACN_TW + 2)

__device__ __forceinline__ float acnH2f(unsigned short h)
{
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}
__device__ __forceinline__ unsigned short acnF2h(float f)
{
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}
__device__ __forceinline__ unsigned acnPack(float a, float b)
{
    return (unsigned)acnF2h(a) | ((unsigned)acnF2h(b) << 16);
}
__device__ __forceinline__ void acnCp16(unsigned dst, const void* src, int bytes)
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
__device__ __forceinline__ void acnLdm4(unsigned addr, unsigned* r)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}
__device__ __forceinline__ float acnClip(float v)
{
    return fminf(fmaxf(v, 0.0f), 1.0f);
}

template <int EPI>
__device__ __forceinline__ void acnBody(const unsigned short* __restrict__ x, unsigned short* y,
                                        const unsigned short* skip, const uint4* __restrict__ wf,
                                        const float* __restrict__ bias, int H, int W)
{
    constexpr int C = ACN_C, CH8 = C / 8, PS = C * 2 + 16, KC = C / 16, NT = C / 8, NQ = C / 16;
    extern __shared__ __align__(128) unsigned char acnSm[];
    const unsigned halo = static_cast<unsigned>(__cvta_generic_to_shared(acnSm));
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int tx0 = blockIdx.x * ACN_TW, ty0 = blockIdx.y * ACN_NW;
    for (int i = tid; i < ACN_HH * ACN_HW * CH8; i += ACN_NW * 32)
    {
        const int p = i / CH8, k = i - p * CH8;
        const int hr = p / ACN_HW, hc = p - hr * ACN_HW;
        const int gy = ty0 - 1 + hr, gx = tx0 - 1 + hc;
        const bool in = gy >= 0 && gy < H && gx >= 0 && gx < W;
        const unsigned short* src = in ? x + ((size_t)gy * W + gx) * C + k * 8 : x;
        acnCp16(halo + p * PS + k * 16, src, in ? 16 : 0);
    }
    asm volatile(SMV_CP_COMMIT);
    uint4 bc[NQ], bn[NQ];
#pragma unroll
    for (int q = 0; q < NQ; ++q)
        bc[q] = __ldg(wf + q * 32 + lane);
    const int g = lane >> 2, t4 = lane & 3;
    unsigned acc[2][NT][2];
#pragma unroll
    for (int nj = 0; nj < NT; ++nj)
    {
        const unsigned b2 = acnPack(bias[nj * 8 + t4 * 2], bias[nj * 8 + t4 * 2 + 1]);
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
            acc[mi][nj][0] = acc[mi][nj][1] = b2;
    }
    asm volatile(SMV_CP_WAIT0);
    __syncthreads();
    const int arow = (lane & 7) + ((lane >> 3) & 1) * 8, akc = lane >> 4;
#pragma unroll 1
    for (int tap = 0; tap < 9; ++tap)
    {
        const int dy = tap / 3, dx = tap - dy * 3;
        int pa[2];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
            pa[mi] = (warp + dy) * ACN_HW + mi * 16 + arow + dx;
#pragma unroll
        for (int kc = 0; kc < KC; ++kc)
        {
            const int step = tap * KC + kc;
            if (step + 1 < 9 * KC)
#pragma unroll
                for (int q = 0; q < NQ; ++q)
                    bn[q] = __ldg(wf + ((step + 1) * NQ + q) * 32 + lane);
#pragma unroll
            for (int mi = 0; mi < 2; ++mi)
            {
                unsigned a[4];
                acnLdm4(halo + pa[mi] * PS + (kc * 2 + akc) * 16, a);
#pragma unroll
                for (int q = 0; q < 2 * NQ; ++q)
                {
                    const unsigned b0 = (q & 1) ? bc[q >> 1].z : bc[q >> 1].x;
                    const unsigned b1 = (q & 1) ? bc[q >> 1].w : bc[q >> 1].y;
                    asm volatile(SMV_MMA_F16 "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                                 : "+r"(acc[mi][q][0]), "+r"(acc[mi][q][1])
                                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
                }
            }
#pragma unroll
            for (int q = 0; q < NQ; ++q)
                bc[q] = bn[q];
        }
    }
    const int gy = ty0 + warp;
    if (gy >= H)
        return;
#pragma unroll
    for (int nj = 0; nj < NT; ++nj)
    {
        const int n0 = nj * 8 + t4 * 2;
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int h = 0; h < 2; ++h)
            {
                const int gx = tx0 + mi * 16 + g + h * 8;
                if (gx >= W)
                    continue;
                float v0 = acnH2f((unsigned short)(acc[mi][nj][h] & 0xffffu));
                float v1 = acnH2f((unsigned short)(acc[mi][nj][h] >> 16));
                const size_t o = ((size_t)gy * W + gx) * C + n0;
                if (EPI == 2)
                {
                    const unsigned s = *reinterpret_cast<const unsigned*>(skip + o);
                    v0 += acnH2f((unsigned short)(s & 0xffffu));
                    v1 += acnH2f((unsigned short)(s >> 16));
                }
                if (EPI == 3)
                {
                    v0 = fmaxf(v0, 0.0f);
                    v1 = fmaxf(v1, 0.0f);
                }
                *reinterpret_cast<unsigned*>(y + o) = acnPack(v0, v1);
            }
    }
}

)ACNSRC";
static const char kAcnB[] = R"ACNSRC(
extern "C" {
__global__ void __launch_bounds__(ACN_NW * 32) k_acnBodyR(const unsigned short* x, unsigned short* y,
                                                          const unsigned short* skip, const uint4* wf,
                                                          const float* bias, int H, int W)
{
    acnBody<3>(x, y, skip, wf, bias, H, W);
}
__global__ void __launch_bounds__(ACN_NW * 32) k_acnBodyA(const unsigned short* x, unsigned short* y,
                                                          const unsigned short* skip, const uint4* wf,
                                                          const float* bias, int H, int W)
{
    acnBody<2>(x, y, skip, wf, bias, H, W);
}

// the first conv of a tile: one fp32 plane at src (row stride rs; w x h read, zero outside) -> 32 channels-last fp16;
// wb = w[o][tap] then b[o]
__global__ void __launch_bounds__(256) k_acnFirst(const float* __restrict__ src, int rs, unsigned short* __restrict__ y,
                                                   const float* __restrict__ wb, int H, int W)
{
    __shared__ float sw[ACN_C * 9 + ACN_C];
    for (int i = threadIdx.x; i < ACN_C * 9 + ACN_C; i += 256)
        sw[i] = wb[i];
    __syncthreads();
    const int xx = blockIdx.x * 32 + (threadIdx.x & 31), yy = blockIdx.y * 8 + (threadIdx.x >> 5);
    if (xx >= W || yy >= H)
        return;
    float v[9];
#pragma unroll
    for (int t = 0; t < 9; ++t)
    {
        const int gy = yy + t / 3 - 1, gx = xx + t % 3 - 1;
        v[t] = (gy >= 0 && gy < H && gx >= 0 && gx < W) ? src[(size_t)gy * rs + gx] : 0.0f;
    }
    unsigned short* o = y + ((size_t)yy * W + xx) * ACN_C;
    for (int n = 0; n < ACN_C; n += 8)
    {
        unsigned r[4];
#pragma unroll
        for (int j = 0; j < 8; j += 2)
        {
            float a0 = sw[ACN_C * 9 + n + j], a1 = sw[ACN_C * 9 + n + j + 1];
#pragma unroll
            for (int k = 0; k < 9; ++k)
            {
                a0 = fmaf(sw[(n + j) * 9 + k], v[k], a0);
                a1 = fmaf(sw[(n + j + 1) * 9 + k], v[k], a1);
            }
            r[j / 2] = acnPack(a0, a1);
        }
        *reinterpret_cast<uint4*>(o + n) = make_uint4(r[0], r[1], r[2], r[3]);
    }
}

// the last conv of a tile (w x h activations) -> 4 outputs, fp32 sums, + bias, clipped, the DepthToSpace DCR 2 in the
// store (output o -> pixel (2y + (o >> 1), 2x + (o & 1))); only the core (cx0, cy0, cw x ch) is written, at out (the
// core's first output pixel, row stride ors)
__global__ void __launch_bounds__(256) k_acnLast(const unsigned short* __restrict__ x, float* __restrict__ out, int ors,
                                                  const float* __restrict__ wb, int H, int W, int cx0, int cy0, int cw,
                                                  int ch)
{
    __shared__ float sw[4 * ACN_C * 9 + 4];
    for (int i = threadIdx.x; i < 4 * ACN_C * 9 + 4; i += 256)
        sw[i] = wb[i];
    __syncthreads();
    const int xc = blockIdx.x * 32 + (threadIdx.x & 31), yc = blockIdx.y * 8 + (threadIdx.x >> 5);
    if (xc >= cw || yc >= ch)
        return;
    const int xx = cx0 + xc, yy = cy0 + yc;
    float acc[4];
#pragma unroll
    for (int o = 0; o < 4; ++o)
        acc[o] = sw[4 * ACN_C * 9 + o];
    for (int t = 0; t < 9; ++t)
    {
        const int gy = yy + t / 3 - 1, gx = xx + t % 3 - 1;
        if (gy < 0 || gy >= H || gx < 0 || gx >= W)
            continue;
        const uint4* px = reinterpret_cast<const uint4*>(x + ((size_t)gy * W + gx) * ACN_C);
        for (int c8 = 0; c8 < ACN_C / 8; ++c8)
        {
            const uint4 u = px[c8];
            const unsigned uu[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                const float f0 = acnH2f((unsigned short)(uu[e] & 0xffffu)), f1 = acnH2f((unsigned short)(uu[e] >> 16));
                const int c = c8 * 8 + e * 2;
#pragma unroll
                for (int o = 0; o < 4; ++o)
                {
                    acc[o] = fmaf(sw[(o * ACN_C + c) * 9 + t], f0, acc[o]);
                    acc[o] = fmaf(sw[(o * ACN_C + c + 1) * 9 + t], f1, acc[o]);
                }
            }
        }
    }
#pragma unroll
    for (int o = 0; o < 4; ++o)
        out[(size_t)(2 * yc + (o >> 1)) * ors + 2 * xc + (o & 1)] = acnClip(acc[o]);
}

// the slot's frame (planar, fp32 or fp16 by half; plane / row strides) -> the author's luma (BT.709 full range of the
// clipped RGB, clipped), one tight fp32 plane of w x h
__global__ void k_acnY(const void* __restrict__ src, int half, int ps, int rs, int w, int h, float* __restrict__ y)
{
    const int x = blockIdx.x * 16 + threadIdx.x, yy = blockIdx.y * 16 + threadIdx.y;
    if (x >= w || yy >= h)
        return;
    const size_t s = (size_t)yy * rs + x;
    float v[3];
    for (int c = 0; c < 3; ++c)
        v[c] = acnClip(half ? acnH2f(((const unsigned short*)src)[(size_t)c * ps + s])
                            : ((const float*)src)[(size_t)c * ps + s]);
    y[(size_t)yy * w + x] = acnClip(fmaf(0.0722f, v[2], fmaf(0.7152f, v[1], 0.2126f * v[0])));
}

// the net's luma y (2w x 2h) + the frame's Lanczos enlargement lz (three planes, the same size, plane stride n) -> RGB
// planes (tight, plane stride n): lz's colour in the author's YCbCr (clipped), the net's luma, the author's inverse,
// clipped
__global__ void k_acnMerge(const float* __restrict__ y, const float* __restrict__ lz, int n, float* __restrict__ rgb)
{
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n)
        return;
    const float r = acnClip(lz[i]), g = acnClip(lz[(size_t)n + i]), b = acnClip(lz[2 * (size_t)n + i]);
    const float cb = acnClip(fmaf(0.5f, b, fmaf(-0.385428f, g, -0.114572f * r)) + 0.5f) - 0.5f;
    const float cr = acnClip(fmaf(-0.045847f, b, fmaf(-0.454153f, g, 0.5f * r)) + 0.5f) - 0.5f;
    const float yy = acnClip(y[i]);
    rgb[i] = acnClip(fmaf(1.5748f, cr, yy));
    rgb[(size_t)n + i] = acnClip(fmaf(-0.4681f, cr, fmaf(-0.1873f, cb, yy)));
    rgb[2 * (size_t)n + i] = acnClip(fmaf(1.8556f, cb, yy));
}
}
)ACNSRC";
static uint64_t nativeVideoMemoryRoom();

// one 32 -> 32 conv's weights [o][c][3][3] (fp32 copies of fp16 values) -> the fragment order the body kernels load:
// [tap][kc 2][q 2][lane 32][8 halves] = n-tiles 2q and 2q + 1, each b0 (k 2t, 2t + 1) then b1 (k 2t + 8, 2t + 9)
static void acnFrag(const float* w, uint16_t* out)
{
    for (int tap = 0; tap < 9; tap++)
        for (int kc = 0; kc < 2; kc++)
            for (int q = 0; q < 2; q++)
                for (int lane = 0; lane < 32; lane++)
                {
                    const int g = lane >> 2, t = lane & 3, k0 = kc * 16 + 2 * t;
                    uint16_t* o = out + ((((size_t)tap * 2 + kc) * 2 + q) * 32 + lane) * 8;
                    for (int j = 0; j < 2; j++)
                    {
                        const size_t n = (size_t)(2 * q + j) * 8 + g;
                        auto at = [&](int c) { return w[(n * 32 + c) * 9 + tap]; };
                        o[4 * j + 0] = halfBitsRne(at(k0));
                        o[4 * j + 1] = halfBitsRne(at(k0 + 1));
                        o[4 * j + 2] = halfBitsRne(at(k0 + 8));
                        o[4 * j + 3] = halfBitsRne(at(k0 + 9));
                    }
                }
}

static bool nativeAcnModule(NativeRife& nr, std::string& why)
{
    if (nr.acnMod)
        return true;
    const std::string text = std::string(kAcnA) + kAcnB;
    std::vector<char> cubin;
    if (!nativeCubin(text.c_str(), "smv_artcnn.cu", L"smv_artcnn", "ArtCNN kernels",
                     nr.kernCacheDir.empty() ? std::wstring(L".") : nr.kernCacheDir, cubin))
    {
        why = "its kernels did not compile";
        return false;
    }
    if (cuModuleLoadData(&nr.acnMod, cubin.data()) != CUDA_SUCCESS)
    {
        nr.acnMod = nullptr;
        why = "its kernel module did not load";
        return false;
    }
    static const char* const kNames[6] = {"k_acnBodyR", "k_acnBodyA", "k_acnFirst",
                                          "k_acnLast",  "k_acnY",     "k_acnMerge"};
    for (int i = 0; i < 6; i++)
        if (cuModuleGetFunction(&nr.fAcn[i], nr.acnMod, kNames[i]) != CUDA_SUCCESS)
        {
            why = std::string("its kernel ") + kNames[i] + " is missing";
            return false;
        }
    return true;
}

// the weights from engine\artcnn\artcnn_c4f32_dn_v1.6.2_fp16.pkl (the author's v1.6.2 ONNX values with conv1 .. conv5
// stored as fp16): every key of C4's form with its shape and no other; conv1 .. conv5 into fragments, the
// rest fp32; false = `why`
static bool nativeAcnWeights(NativeRife& nr, std::string& why)
{
    if (nr.dAcnWf)
        return true;
    PthFile wf;
    if (!pthStateDict(g_nativeEngineDir + L"\\artcnn\\artcnn_c4f32_dn_v1.6.2_fp16.pkl", wf, why))
        return false;
    auto get = [&](const std::string& k, const std::vector<int64_t>& shape) -> const float* {
        const auto t = wf.t.find(k);
        const auto s = wf.S.find(k);
        if (t == wf.t.end() || s == wf.S.end() || s->second != shape)
        {
            why = "its weights are not ArtCNN C4's (" + k + ")";
            return nullptr;
        }
        return t->second.data();
    };
    std::vector<uint16_t> frags((size_t)5 * 9 * 32 * 32);
    std::vector<float> bias(5 * 32), w0(32 * 9 + 32), wz(4 * 32 * 9 + 4);
    for (int i = 0; i < 5; i++)
    {
        const std::string k = "conv" + std::to_string(i + 1);
        const float* w = get(k + ".weight", {32, 32, 3, 3});
        const float* b = w ? get(k + ".bias", {32}) : nullptr;
        if (!b)
            return false;
        acnFrag(w, frags.data() + (size_t)i * 9 * 32 * 32);
        memcpy(bias.data() + (size_t)i * 32, b, 32 * sizeof(float));
    }
    const float* fw = get("first.weight", {32, 1, 3, 3});
    const float* fb = fw ? get("first.bias", {32}) : nullptr;
    const float* lw = fb ? get("last.weight", {4, 32, 3, 3}) : nullptr;
    const float* lb = lw ? get("last.bias", {4}) : nullptr;
    if (!lb)
        return false;
    if (wf.t.size() != 14)
    {
        why = "its weights hold tensors ArtCNN C4 does not have";
        return false;
    }
    memcpy(w0.data(), fw, 32 * 9 * sizeof(float));
    memcpy(w0.data() + 32 * 9, fb, 32 * sizeof(float));
    memcpy(wz.data(), lw, 4 * 32 * 9 * sizeof(float));
    memcpy(wz.data() + 4 * 32 * 9, lb, 4 * sizeof(float));
    if (cudaMalloc(&nr.dAcnWf, frags.size() * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc((void**)&nr.dAcnB, bias.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void**)&nr.dAcnW0, w0.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void**)&nr.dAcnWz, wz.size() * sizeof(float)) != cudaSuccess ||
        cudaMemcpy(nr.dAcnWf, frags.data(), frags.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(nr.dAcnB, bias.data(), bias.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(nr.dAcnW0, w0.data(), w0.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(nr.dAcnWz, wz.data(), wz.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        cudaGetLastError();
        why = "its weights did not upload";
        return false;
    }
    return true;
}

// the frame buffers (nativeAcnSetup's), freed; the weights stay
static void nativeAcnBufFree(NativeRife& nr)
{
    for (void** p : {&nr.dAcnAct, (void**)&nr.dAcnY, (void**)&nr.dAcnY2, (void**)&nr.dAcnLz, (void**)&nr.dAcnOut,
                     (void**)&nr.dAcnTmp})
        if (*p)
        {
            cudaFree(*p);
            *p = nullptr;
        }
    nr.acnActN = nr.acnTmpN = 0;
    nr.acnCore = 0;
    nr.acnW = nr.acnH = nr.acnOw = 0;
    nr.acnN = 1;
}

// every ArtCNN buffer of the session (the module goes with the session's other modules, nativeFree)
static void nativeAcnRelease(NativeRife& nr)
{
    if (nr.stream && (nr.dAcnAct || nr.dAcnWf))
        cudaStreamSynchronize(nr.stream);
    nativeAcnBufFree(nr);
    for (void** p : {&nr.dAcnWf, (void**)&nr.dAcnB, (void**)&nr.dAcnW0, (void**)&nr.dAcnWz})
        if (*p)
        {
            cudaFree(*p);
            *p = nullptr;
        }
}

// the luma net on st: src (one fp32 plane, nw x nh, row stride rs) -> out (2nw x 2nh, row stride ors), the whole frame
// at once or tile by tile (nr.acnCore: each core read with 7 real pixels around it, the net's conv count, so a core holds
// the whole frame's bits); conv5 adds conv0's output in place (A), the body ping-pongs B / D
static bool nativeAcnNet(NativeRife& nr, const float* src, int rs, float* out, int ors, int nh, int nw, cudaStream_t st)
{
    const int core = nr.acnCore ? nr.acnCore : (std::max)(nh, nw), hal = nr.acnCore ? 7 : 0;
    unsigned short* A = (unsigned short*)nr.dAcnAct;
    unsigned short* B = A + nr.acnActN;
    unsigned short* D = B + nr.acnActN;
    const unsigned smem = 6 * 34 * 80;
    for (int y0 = 0; y0 < nh; y0 += core)
        for (int x0 = 0; x0 < nw; x0 += core)
        {
            const int y1 = (std::min)(y0 + core, nh), x1 = (std::min)(x0 + core, nw);
            const int ya = (std::max)(y0 - hal, 0), xa = (std::max)(x0 - hal, 0);
            int th = (std::min)(y1 + hal, nh) - ya, tw = (std::min)(x1 + hal, nw) - xa, trs = rs;
            const float* ts = src + (size_t)ya * rs + xa;
            void* af[] = {(void*)&ts, &trs, &A, (void*)&nr.dAcnW0, &th, &tw};
            if (cuLaunchKernel(nr.fAcn[2], (tw + 31) / 32, (th + 7) / 8, 1, 256, 1, 1, 0, (CUstream)st, af, nullptr) !=
                CUDA_SUCCESS)
                return false;
            auto body = [&](int fn, unsigned short* x, unsigned short* y, int i) {
                const void* wfi = (const char*)nr.dAcnWf + (size_t)i * 9 * 32 * 32 * sizeof(uint16_t);
                const float* bi = nr.dAcnB + (size_t)i * 32;
                unsigned short* skip = A;
                void* a[] = {&x, &y, &skip, (void*)&wfi, (void*)&bi, &th, &tw};
                return cuLaunchKernel(nr.fAcn[fn], (tw + 31) / 32, (th + 3) / 4, 1, 128, 1, 1, smem, (CUstream)st, a,
                                      nullptr) == CUDA_SUCCESS;
            };
            if (!body(0, A, B, 0) || !body(0, B, D, 1) || !body(0, D, B, 2) || !body(0, B, D, 3) || !body(1, D, A, 4))
                return false;
            int cx0 = x0 - xa, cy0 = y0 - ya, cw = x1 - x0, ch = y1 - y0, os = ors;
            float* o = out + (size_t)2 * y0 * ors + (size_t)2 * x0;
            void* al[] = {&A, &o, &os, (void*)&nr.dAcnWz, &th, &tw, &cx0, &cy0, &cw, &ch};
            if (cuLaunchKernel(nr.fAcn[3], (cw + 31) / 32, (ch + 7) / 8, 1, 256, 1, 1, 0, (CUstream)st, al, nullptr) !=
                CUDA_SUCCESS)
                return false;
        }
    return true;
}

// the Lanczos pair (k_fitAaH / V, the session's tap tables) from src (planar, fp32 or fp16 by half, plane / row
// strides) at sw x sh to dst (tight, three planes) at tw x th, through dAcnTmp
static bool nativeAcnFit(NativeRife& nr, const void* src, int half, int ps, int rs, int sw, int sh, float* dst, int tw,
                         int th, cudaStream_t st)
{
    if ((size_t)3 * sh * tw > nr.acnTmpN)
    {
        nr.die("ArtCNN resize pass larger than its buffer");
        return false;
    }
    const void* tabH = rsTabFor(nr, sw, tw, st);
    const void* tabV = rsTabFor(nr, sh, th, st);
    if (!tabH || !tabV)
    {
        nr.die("ArtCNN resize tap tables failed");
        return false;
    }
    void* ah[] = {(void*)&src, &half, &ps, &rs, &sw, &sh, &nr.dAcnTmp, &tw, (void*)&tabH};
    void* av[] = {&nr.dAcnTmp, &tw, &sh, &dst, &th, (void*)&tabV};
    if (cuLaunchKernel(nr.fFitAaH, (tw + 15) / 16, (sh + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, ah, nullptr) !=
            CUDA_SUCCESS ||
        cuLaunchKernel(nr.fFitAaV, (tw + 15) / 16, (th + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, av, nullptr) !=
            CUDA_SUCCESS)
    {
        nr.die("ArtCNN resize launch failed");
        return false;
    }
    return true;
}

// ArtCNN at the slot's input size w x h (-> ow x oh after the fit): its module and weights, the x2 passes the ratio
// needs (a4kPassesFor; pass k runs at 2^k x the input, the buffers hold the largest), the activations for whole frames
// when the video memory room holds them, else for the largest square tile that does (down to 128 px; one pass fewer
// when even that does not fit), then one run of the net on zeros, so the first frame pays no first launch. No module,
// no weights or no room for a single pass = Lanczos for the render / session (vsrFailed), with its line.
static bool nativeAcnSetup(NativeRife& nr, int w, int h, int ow, int oh, int upEnc)
{
    const auto t0 = std::chrono::steady_clock::now();
    const char* where = g_offline ? "render" : "session";
    std::string why;
    if (!nativeAcnModule(nr, why) || !nativeAcnWeights(nr, why))
    {
        LOG("native: ArtCNN unavailable (%s), Lanczos for the %s\n", why.c_str(), where);
        nr.vsrFailed = true;
        return true;
    }
    const uint64_t room0 = nativeVideoMemoryRoom();
    const uint64_t room = room0 ? room0 : UINT64_MAX;
    int n = a4kPassesFor(w, h, ow, oh);
    for (;; n--)
    {
        const size_t wm = (size_t)w << (n - 1), hm = (size_t)h << (n - 1), pin = wm * hm, pout = 4 * pin;
        const size_t tmpN = (std::max)(3 * hm * 2 * wm, 3 * 2 * hm * (std::max)((size_t)ow, 2 * wm));
        const size_t planesB = (pin + 7 * pout + tmpN) * sizeof(float);
        auto tileOf = [&](int c, size_t full) { return c ? (std::min)((size_t)c + 14, full) : full; };
        auto actB = [&](int c) { return tileOf(c, wm) * tileOf(c, hm) * 32 * sizeof(uint16_t) * 3; };
        int core = 0;
        if (planesB + actB(0) > room)
        {
            core = (int)(((std::max)(wm, hm) + 1) / 2);
            while (core >= 128 && planesB + actB(core) > room)
                core /= 2;
        }
        bool ok = core == 0 || core >= 128;
        if (ok)
        {
            nr.acnActN = tileOf(core, wm) * tileOf(core, hm) * 32;
            nr.acnTmpN = tmpN;
            ok = cudaMalloc(&nr.dAcnAct, nr.acnActN * 3 * sizeof(uint16_t)) == cudaSuccess &&
                 cudaMalloc((void**)&nr.dAcnY, pin * sizeof(float)) == cudaSuccess &&
                 cudaMalloc((void**)&nr.dAcnY2, pout * sizeof(float)) == cudaSuccess &&
                 cudaMalloc((void**)&nr.dAcnLz, 3 * pout * sizeof(float)) == cudaSuccess &&
                 cudaMalloc((void**)&nr.dAcnOut, 3 * pout * sizeof(float)) == cudaSuccess &&
                 cudaMalloc((void**)&nr.dAcnTmp, tmpN * sizeof(float)) == cudaSuccess;
        }
        if (ok)
        {
            nr.acnCore = core;
            break;
        }
        cudaGetLastError();
        nativeAcnBufFree(nr);
        if (n == 1)
        {
            LOG("native: ArtCNN ran out of video memory, Lanczos for the %s\n", where);
            nr.vsrFailed = true;
            return true;
        }
    }
    nr.acnW = w;
    nr.acnH = h;
    nr.acnN = n;
    nr.acnOw = ow;
    NCHK(cudaMemsetAsync(nr.dAcnY, 0, (size_t)w * h * sizeof(float), nr.stream), "ArtCNN warm-up clear");
    if (!nativeAcnNet(nr, nr.dAcnY, w, nr.dAcnY2, 2 * w, h, w, nr.stream))
    {
        LOG("native: ArtCNN warm-up launch failed\n");
        return false;
    }
    NCHK(cudaStreamSynchronize(nr.stream), "ArtCNN warm-up sync");
    char passes[48] = "", tiles[48] = "";
    if (n > 1)
        sprintf_s(passes, " in %d x2 passes", n);
    if (nr.acnCore)
        sprintf_s(tiles, ", %d-px tiles", nr.acnCore);
    LOG("native: %s upscale: ArtCNN C4F32_DN %dx%d -> %dx%d%s -> Lanczos %dx%d%s%s%s (CUDA passes, tensor-core layers "
        "with fp16 sums, %.1fs)\n",
        g_offline ? "offline" : "live", w, h, w << n, h << n, passes, ow, oh, tiles,
        nr.vsrPre ? ", the captured frame before the model" : "",
        nr.hdr ? ", on the HDR planes' SDR view" : (upEnc ? ", on the HDR video's SDR view" : ""),
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return true;
}

// one frame through ArtCNN: s (planar, fp32 or fp16 by sHalf, plane / row strides ps_ / rs_) at the slot's input
// size, pass by pass: its luma (k_acnY) through the net to 2w x 2h, the frame's Lanczos x2 for the colour, merged
// (k_acnMerge: the colour's YCbCr with the net's luma, the author's way back to RGB, clipped); a pass that is not the
// last writes dAcnOut, the next pass's input; the last one writes dst at an exact 2x target, else dAcnOut and then
// the Lanczos pair to tw x th into dst (planar fp32, tight, clipped to 0..1)
static bool nativeAcnRun(NativeRife& nr, const void* s, int sHalf, int ps_, int rs_, float* dst, int tw, int th)
{
    cudaStream_t st = nr.stream;
    if (tw > nr.acnOw)
    {
        nr.die("ArtCNN fit target wider than its buffer");
        return false;
    }
    const void* src = s;
    int half = sHalf, ps = ps_, rs = rs_;
    for (int k = 0; k < nr.acnN; k++)
    {
        int w = nr.acnW << k, h = nr.acnH << k, w2 = 2 * w, h2 = 2 * h, n2 = w2 * h2;
        void* ay[] = {(void*)&src, &half, &ps, &rs, &w, &h, &nr.dAcnY};
        if (cuLaunchKernel(nr.fAcn[4], (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, ay, nullptr) !=
                CUDA_SUCCESS ||
            !nativeAcnNet(nr, nr.dAcnY, w, nr.dAcnY2, w2, h, w, st))
        {
            nr.die("ArtCNN net launch failed");
            return false;
        }
        if (!nativeAcnFit(nr, src, half, ps, rs, w, h, nr.dAcnLz, w2, h2, st))
            return false;
        const bool last = k + 1 == nr.acnN, exact = last && tw == w2 && th == h2;
        float* o = exact ? dst : nr.dAcnOut;
        void* am[] = {&nr.dAcnY2, &nr.dAcnLz, &n2, &o};
        if (cuLaunchKernel(nr.fAcn[5], (n2 + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, am, nullptr) != CUDA_SUCCESS)
        {
            nr.die("ArtCNN merge launch failed");
            return false;
        }
        if (!last)
        {
            src = nr.dAcnOut;
            half = 0;
            ps = n2;
            rs = w2;
            continue;
        }
        if (!exact)
        {
            int n = 3 * tw * th;
            void* c[] = {&dst, &n};
            if (!nativeAcnFit(nr, nr.dAcnOut, 0, n2, w2, w2, h2, dst, tw, th, st))
                return false;
            if (cuLaunchKernel(nr.fClamp01, (n + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, c, nullptr) !=
                CUDA_SUCCESS)
            {
                nr.die("ArtCNN clip launch failed");
                return false;
            }
        }
    }
    return true;
}
