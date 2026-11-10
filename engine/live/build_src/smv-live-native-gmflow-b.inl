// part of smv-live-native.inl (included there, in place); not a standalone translation unit
// gmflow's second half B (gmflow's last scale after the local correlation: the window's softmax, the flow
// propagation, the convex upsampler) as 7 launches (nativeGbRun) of its own module smv_gmflowb_*.cubin (kFnHead's
// helpers + fusionnet's kFuA / kFuB for k_fuW64H + kGb), weights = engine\GMFSS_Fortuna\train_log\flownet.pkl's
// feature_flow_attn and upsampler through pthStateDict (gbNatLoad); no fallback (a missing / wrong
// pkl = `gmflow B unavailable (...)`, the session refuses). At the quarter of the half (nr.gmH x nr.gmW), both directions
// as the batch: corr (k_localCorr's raw sums, 81 taps) + A's qf / flow0 in, the half's flow (2, 2, hh, hw) out into
// dGmFlow. k_gbFlow: corr / sqrt(128), the off-frame taps masked, softmax, flow0 + sum(prob x offset); k_gbProj: q / k
// (a 1 x 1 mma GEMM, fp32 sums); k_gbProp: the 9 zero-padded neighbours' softmax over q . k / sqrt(128), the
// propagated flow, the upsampler's input u0 (feat | flow | zeros, 192 channels); k_fuW64H per direction: the 3 x 3 conv
// 130 -> 256 (fp16 sums); k_gbMask: the 1 x 1 conv on its ReLU (fp32 sums); k_gbUp: the softmax over the 9 taps per 4 x 4
// sub-pixel and the convex combination of 4 x the flow. Its activations (gbArena) live in the shared activation block
// (sound while serial on nr.stream), else dGbOwn.
static const char kGb[] = R"GBSRC(
// gmflow's second half B on the host's kernels, compiled after kFnHead and fusionnet's text (fuSlot, fuLds, fuConvW:
// k_fuW64H = the upsampler's 3 x 3 conv with fp16 sums). Every tensor at the quarter of the GMFSS half, both flow
// directions as the batch (B = 2), P = H W pixels an image.

__device__ __forceinline__ void gbMma(float* d, const unsigned* a, unsigned b0, unsigned b1)
{
    asm volatile(SMV_MMA_F32 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// ReLU on two packed fp16 values (a set sign bit = 0)
__device__ __forceinline__ unsigned gbRelu2(unsigned u)
{
    return ((u & 0x8000u) ? 0u : (u & 0xffffu)) | ((u & 0x80000000u) ? 0u : (u & 0xffff0000u));
}

// the local window's softmax: corr [B][P][81] = the host's raw fp32 sums (tap k = (dy + 4) 9 + dx + 4), / sqrt(128),
// the taps outside the frame masked, softmax over the 81, flow = flow0 + sum(prob x (dx, dy)); flow0 / flow planar
// fp32 [B][2][H][W]
extern "C" __global__ void __launch_bounds__(256) k_gbFlow(const float* __restrict__ corr,
                                                            const float* __restrict__ flow0, float* __restrict__ flow,
                                                            int H, int W, int B)
{
    const int P = H * W;
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= B * P)
        return;
    const int b = i / P, p = i - b * P, y = p / W, x = p - y * W;
    const float* c = corr + (size_t)i * 81;
    float s[81];
    float m = -3.402823466e38f;
#pragma unroll
    for (int k = 0; k < 81; ++k)
    {
        const int dx = k % 9 - 4, dy = k / 9 - 4;
        const bool v = x + dx >= 0 && x + dx < W && y + dy >= 0 && y + dy < H;
        s[k] = v ? c[k] / 11.313708498984761f : -1.7014117331926443e38f;
        m = fmaxf(m, s[k]);
    }
    float sum = 0.0f, fx = 0.0f, fy = 0.0f;
#pragma unroll
    for (int k = 0; k < 81; ++k)
    {
        const float e = expf(s[k] - m);
        sum += e;
        fx += e * (float)(k % 9 - 4);
        fy += e * (float)(k / 9 - 4);
    }
    const size_t o = (size_t)b * 2 * P + p;
    flow[o] = flow0[o] + fx / sum;
    flow[o + P] = flow0[o + P] + fy / sum;
}

// Y = act(A) W^T + bias on the tensor cores (mma m16n8k16, fp32 sums from the bias, fp16 out): A [M][K] fp16 rows (K a
// multiple of 16 up to 256) staged in shared memory with 16-byte chunks XOR-swizzled by row, W in fragment order
// wf[z][K / 16][NT / 2][lane] (blockIdx.y = z: NT n-tiles a block), bias fp32; Y [M][N] fp16; 128 threads = 4 warps x
// 16 rows; RELU = ReLU on A as it is read
template <int NT, bool RELU>
__device__ __forceinline__ void gbGemmT(const unsigned short* __restrict__ A, int M, int K,
                                        const uint4* __restrict__ wf, const float* __restrict__ bias,
                                        unsigned short* __restrict__ Y, int N)
{
    extern __shared__ __align__(128) unsigned char gbSm[];
    const unsigned sm = (unsigned)__cvta_generic_to_shared(gbSm);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int m0 = blockIdx.x * 64, z = blockIdx.y;
    const int nch = K / 8;
    for (int i = threadIdx.x; i < 64 * nch; i += 128)
    {
        const int r = i / nch, k = i - r * nch;
        const bool in = m0 + r < M;
        rnCp16(sm + (unsigned)((r * nch + (k ^ (r & 7))) << 4), in ? A + (size_t)(m0 + r) * K + k * 8 : A, in ? 16 : 0);
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
#pragma unroll 2
    for (int s = 0; s < ks; ++s)
    {
        unsigned a[4];
        const int kc = 2 * s + akc;
        rnLdm4(sm + (unsigned)((arow * nch + (kc ^ (arow & 7))) << 4), a);
        if (RELU)
        {
#pragma unroll
            for (int e = 0; e < 4; ++e)
                a[e] = gbRelu2(a[e]);
        }
#pragma unroll
        for (int q = 0; q < NT / 2; ++q)
        {
            const uint4 bw = wf[((size_t)s * (NT / 2) + q) * 32 + lane];
            gbMma(acc[2 * q], a, bw.x, bw.y);
            gbMma(acc[2 * q + 1], a, bw.z, bw.w);
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

// q / k projections: feat [B P][128] -> qk [B P][256] (q = 0..127, k = 128..255)
extern "C" __global__ void __launch_bounds__(128) k_gbProj(const unsigned short* __restrict__ A, int M,
                                                            const uint4* __restrict__ wf,
                                                            const float* __restrict__ bias,
                                                            unsigned short* __restrict__ Y)
{
    gbGemmT<8, false>(A, M, 128, wf, bias, Y, 256);
}

// the upsampler's 1 x 1 conv on ReLU(its 3 x 3 conv): y0 [B P][256] -> the mask logits [B P][144]
extern "C" __global__ void __launch_bounds__(128) k_gbMask(const unsigned short* __restrict__ A, int M,
                                                            const uint4* __restrict__ wf,
                                                            const float* __restrict__ bias,
                                                            unsigned short* __restrict__ Y)
{
    gbGemmT<6, true>(A, M, 256, wf, bias, Y, 144);
}

// the flow propagation (FeatureFlowAttention, radius 1), one warp a pixel: qk [B P][256] fp16, flow [B][2][H][W] fp32
// in; the 9 neighbours' scores q . k (fp32 sums of the fp16 values; a neighbour outside the frame = a zero key, score
// 0) / sqrt(128), softmax, the zero-padded flow summed -> prop [B][2][H][W] fp32; u0 [B P][192] fp16 = the
// upsampler's input: 0..127 = feat (qf's rows 0..B - 1), 128..129 = prop, 130..191 = 0
extern "C" __global__ void __launch_bounds__(256) k_gbProp(const unsigned short* __restrict__ qk,
                                                            const unsigned short* __restrict__ feat,
                                                            const float* __restrict__ flow, float* __restrict__ prop,
                                                            unsigned short* __restrict__ u0, int H, int W, int B)
{
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int P = H * W;
    if (i >= B * P)
        return;
    const int b = i / P, p = i - b * P, y = p / W, x = p - y * W;
    const uint2 qu = *reinterpret_cast<const uint2*>(qk + (size_t)i * 256 + lane * 4);
    const float q0 = h2f((unsigned short)(qu.x & 0xffffu)), q1 = h2f((unsigned short)(qu.x >> 16)),
                q2 = h2f((unsigned short)(qu.y & 0xffffu)), q3 = h2f((unsigned short)(qu.y >> 16));
    float s[9];
    float m = -3.402823466e38f;
#pragma unroll
    for (int n = 0; n < 9; ++n)
    {
        const int yy = y + n / 3 - 1, xx = x + n % 3 - 1;
        float d = 0.0f;
        if (yy >= 0 && yy < H && xx >= 0 && xx < W)
        {
            const uint2 ku =
                *reinterpret_cast<const uint2*>(qk + ((size_t)b * P + (size_t)yy * W + xx) * 256 + 128 + lane * 4);
            d = q0 * h2f((unsigned short)(ku.x & 0xffffu));
            d = fmaf(q1, h2f((unsigned short)(ku.x >> 16)), d);
            d = fmaf(q2, h2f((unsigned short)(ku.y & 0xffffu)), d);
            d = fmaf(q3, h2f((unsigned short)(ku.y >> 16)), d);
        }
#pragma unroll
        for (int o = 16; o; o >>= 1)
            d += __shfl_xor_sync(0xffffffffu, d, o);
        s[n] = d / 11.313708498984761f;
        m = fmaxf(m, s[n]);
    }
    const float* fl = flow + (size_t)b * 2 * P;
    float sum = 0.0f, fx = 0.0f, fy = 0.0f;
#pragma unroll
    for (int n = 0; n < 9; ++n)
    {
        const int yy = y + n / 3 - 1, xx = x + n % 3 - 1;
        const float e = expf(s[n] - m);
        sum += e;
        if (yy >= 0 && yy < H && xx >= 0 && xx < W)
        {
            fx = fmaf(e, fl[yy * W + xx], fx);
            fy = fmaf(e, fl[P + yy * W + xx], fy);
        }
    }
    const float px = fx / sum, py = fy / sum;
    if (lane == 0)
    {
        prop[(size_t)b * 2 * P + p] = px;
        prop[(size_t)b * 2 * P + P + p] = py;
    }
    if (lane < 24)
    {
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (lane < 16)
            v = *reinterpret_cast<const uint4*>(feat + (size_t)i * 128 + lane * 8);
        else if (lane == 16)
            v.x = (unsigned)f2h(px) | ((unsigned)f2h(py) << 16);
        *reinterpret_cast<uint4*>(u0 + (size_t)i * 192 + lane * 8) = v;
    }
}

// the convex upsampler's last step, one thread a quarter pixel's sub-pixel: lg [B P][144] fp16 = the mask logits
// (channel k 16 + 4 ky + kx: tap k of the 3 x 3, sub-pixel (ky, kx)); the softmax over the 9 taps (fp32), the zero-padded
// 3 x 3 of 4 x prop summed -> out [B][2][4H][4W] fp32 at (4 y + ky, 4 x + kx)
extern "C" __global__ void __launch_bounds__(256) k_gbUp(const unsigned short* __restrict__ lg,
                                                          const float* __restrict__ prop, float* __restrict__ out,
                                                          int H, int W, int B)
{
    const int t = blockIdx.x * 256 + threadIdx.x;
    const int P = H * W;
    if (t >= B * P * 16)
        return;
    const int sub = t & 15, i = t >> 4;
    const int b = i / P, p = i - b * P, y = p / W, x = p - y * W;
    const unsigned short* l = lg + (size_t)i * 144 + sub;
    float v[9];
    float m = -3.402823466e38f;
#pragma unroll
    for (int k = 0; k < 9; ++k)
    {
        v[k] = h2f(l[k * 16]);
        m = fmaxf(m, v[k]);
    }
    const float* f = prop + (size_t)b * 2 * P;
    float sum = 0.0f, ux = 0.0f, uy = 0.0f;
#pragma unroll
    for (int k = 0; k < 9; ++k)
    {
        const int yy = y + k / 3 - 1, xx = x + k % 3 - 1;
        const float e = expf(v[k] - m);
        sum += e;
        if (yy >= 0 && yy < H && xx >= 0 && xx < W)
        {
            ux = fmaf(e, 4.0f * f[yy * W + xx], ux);
            uy = fmaf(e, 4.0f * f[P + yy * W + xx], uy);
        }
    }
    const int OW = 4 * W, OH = 4 * H, oy = 4 * y + (sub >> 2), ox = 4 * x + (sub & 3);
    const size_t o = (size_t)b * 2 * OH * OW + (size_t)oy * OW + ox;
    out[o] = ux / sum;
    out[o + (size_t)OH * OW] = uy / sum;
}
)GBSRC";
static const char* const kGbKern[] = {"k_gbFlow", "k_gbProj", "k_gbProp", "k_fuW64H", "k_gbMask", "k_gbUp"};
static_assert(sizeof(kGbKern) / sizeof(kGbKern[0]) == sizeof(NativeRife::fGb) / sizeof(CUfunction),
              "one fGb slot a gmflow B kernel");
enum
{
    kGbFlow,
    kGbProj,
    kGbProp,
    kGbConv,
    kGbMask,
    kGbUp
};
// the upsampler input's channels, the GEMMs' shared memory (a 64-row A tile)
constexpr int kGbU0 = 192, kGbSmProj = 64 * 128 * 2, kGbSmMask = 64 * 256 * 2;

// a 1 x 1 weight [n][k] in gbGemmT's fragment order [z][k / 16][nt / 2][lane][8 halves] (n-tiles 2q, 2q + 1 of z's nt:
// row 8 (nt z + 2q (+ 1)) + lane / 4, column 16 s + 2 (lane % 4) (+ 1, + 8, + 9)), rounded to fp16
static std::vector<uint16_t> gbFrag1(std::span<const float> w, int n, int k, int nt)
{
    static const int kOfs[4] = {0, 1, 8, 9};
    std::vector<uint16_t> out((size_t)n * k);
    size_t o = 0;
    for (int z = 0; z < n / (8 * nt); z++)
        for (int s = 0; s < k / 16; s++)
            for (int q = 0; q < nt / 2; q++)
                for (int l = 0; l < 32; l++)
                    for (int hf = 0; hf < 2; hf++)
                        for (int j = 0; j < 4; j++)
                        {
                            const int r = z * 8 * nt + (2 * q + hf) * 8 + (l >> 2);
                            const int c = 16 * s + (l & 3) * 2 + kOfs[j];
                            out[o++] = f2hRne(w[(size_t)r * k + c]);
                        }
    return out;
}

// flownet.pkl's B in its strict form (feature_flow_attn.q_proj / k_proj [128][128] + bias, upsampler.0 [256][130][3][3]
// + bias, upsampler.2 [144][256][1][1] + bias) as one weight block of 256-byte aligned pieces: off[0] / [1] the q | k
// projection's fragments (nt 8) and fp32 bias, [2] / [3] the 3 x 3 conv's (its input channels in u0's order: feat 0..127,
// flow 128..129, zeros to 192) and bias, [4] / [5] the 1 x 1 conv's (nt 6) and bias; false = missing, or not this net
static bool gbNatLoad(std::vector<uint8_t>& blob, size_t* off, std::string& why)
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
    const std::string fa = "feature_flow_attn.", up = "upsampler.";
    if (!has(fa + "q_proj.weight", {128, 128}) || !has(fa + "q_proj.bias", {128}) ||
        !has(fa + "k_proj.weight", {128, 128}) || !has(fa + "k_proj.bias", {128}) ||
        !has(up + "0.weight", {256, 130, 3, 3}) || !has(up + "0.bias", {256}) ||
        !has(up + "2.weight", {144, 256, 1, 1}) || !has(up + "2.bias", {144}) || S.count(up + "3.weight"))
    {
        why = "its weights are not gmflow's";
        return false;
    }
    blob.clear();
    auto put = [&](const void* d, size_t n) {
        const size_t o = (blob.size() + 255) & ~(size_t)255;
        blob.resize(o + n);
        memcpy(blob.data() + o, d, n);
        return o;
    };
    std::vector<float> qk((size_t)256 * 128), qkb(256);
    const std::span<const float> qw = wts[fa + "q_proj.weight"], kw = wts[fa + "k_proj.weight"];
    const std::span<const float> qb = wts[fa + "q_proj.bias"], kb = wts[fa + "k_proj.bias"];
    for (size_t i = 0; i < (size_t)128 * 128; i++)
    {
        qk[i] = qw[i];
        qk[(size_t)128 * 128 + i] = kw[i];
    }
    for (int i = 0; i < 128; i++)
    {
        qkb[i] = qb[i];
        qkb[128 + i] = kb[i];
    }
    const std::vector<uint16_t> fp = gbFrag1(qk, 256, 128, 8);
    off[0] = put(fp.data(), fp.size() * sizeof(uint16_t));
    off[1] = put(qkb.data(), qkb.size() * sizeof(float));
    const std::span<const float> w0 = wts[up + "0.weight"];
    std::vector<float> w0p((size_t)256 * kGbU0 * 9, 0.0f);
    for (int n = 0; n < 256; n++)
        for (int c = 0; c < 130; c++)
        {
            const int u = c < 2 ? 128 + c : c - 2; // the model's cat(flow, feat) -> u0's feat | flow
            for (int t = 0; t < 9; t++)
                w0p[((size_t)n * kGbU0 + u) * 9 + t] = w0[((size_t)n * 130 + c) * 9 + t];
        }
    const std::vector<uint16_t> f0 = fnFragOf(w0p, 256, kGbU0, 64, 8);
    off[2] = put(f0.data(), f0.size() * sizeof(uint16_t));
    off[3] = put(wts[up + "0.bias"].data(), (size_t)256 * sizeof(float));
    const std::vector<uint16_t> fm = gbFrag1(wts[up + "2.weight"], 144, 256, 6);
    off[4] = put(fm.data(), fm.size() * sizeof(uint16_t));
    off[5] = put(wts[up + "2.bias"].data(), (size_t)144 * sizeof(float));
    return true;
}

// gmflow B's activations at the quarter w x h (P = w h, both directions), 256-byte pieces: the flow after the local
// softmax and the propagated one (fp32 [2][2][P] each), q | k ([2P][256] fp16), u0 ([2P][192] fp16), the 3 x 3 conv's
// output ([2P][256] fp16), the mask logits ([2P][144] fp16)
struct GbArena
{
    size_t flow = 0, qk = 0, prop = 0, u0 = 0, y0 = 0, lg = 0, total = 0;
};
static GbArena gbArena(int w, int h)
{
    GbArena a;
    const size_t P = (size_t)w * h;
    size_t o = 0;
    auto put = [&](size_t n) {
        const size_t at = o;
        o += (n + 255) & ~(size_t)255;
        return at;
    };
    a.flow = put(4 * P * sizeof(float));
    a.qk = put(2 * P * 256 * sizeof(uint16_t));
    a.prop = put(4 * P * sizeof(float));
    a.u0 = put(2 * P * kGbU0 * sizeof(uint16_t));
    a.y0 = put(2 * P * 256 * sizeof(uint16_t));
    a.lg = put(2 * P * 144 * sizeof(uint16_t));
    a.total = o;
    return a;
}

static size_t gbNatBytes(const NativeRife& nr)
{
    return nr.gmfss && nr.hw > 0 && nr.hh > 0 ? gbArena(nr.hw / 4, nr.hh / 4).total : 0;
}

// this session's gmflow B module: the weights read, its text compiled (or the cached cubin) and loaded, its kernels
// bound, the weight block uploaded; false = `why` (the session teardown unloads a module left in gbMod)
static bool nativeGbModule(NativeRife& nr, std::string& why)
{
    if (nr.gbMod)
        return true;
    std::vector<uint8_t> blob;
    if (!gbNatLoad(blob, nr.gbOff, why))
        return false;
    std::string text = kFnHead;
    text += kFuA;
    text += kFuB;
    text += kGb;
    std::vector<char> cubin;
    if (!nativeCubin(text.c_str(), "smv_gmflowb.cu", L"smv_gmflowb", "gmflow B kernels",
                     nr.kernCacheDir.empty() ? std::wstring(L".") : nr.kernCacheDir, cubin))
    {
        why = "its kernels did not compile";
        return false;
    }
    if (cuModuleLoadData(&nr.gbMod, cubin.data()) != CUDA_SUCCESS)
    {
        nr.gbMod = nullptr;
        why = "its kernel module did not load";
        return false;
    }
    for (int i = 0; i < (int)(sizeof(kGbKern) / sizeof(kGbKern[0])); i++)
        if (cuModuleGetFunction(&nr.fGb[i], nr.gbMod, kGbKern[i]) != CUDA_SUCCESS)
        {
            why = std::string("its kernel ") + kGbKern[i] + " is missing";
            return false;
        }
    if (cudaMalloc(&nr.dGbW, blob.size()) != cudaSuccess ||
        cudaMemcpy(nr.dGbW, blob.data(), blob.size(), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        why = "its weights did not upload";
        return false;
    }
    return true;
}

// gmflow B's 7 launches on st at the quarter nr.gmW x nr.gmH, its activations at s (gbArena's pieces): corr, A's qf
// (rows 0..1 = feat) and flow0 in, the half's flow out. false = a launch failed
static bool nativeGbLaunch(NativeRife& nr, const float* corr, const void* qf, const float* flow0, float* out,
                           uint8_t* s, cudaStream_t st)
{
    int iw = nr.gmW, ih = nr.gmH, nb = 2, m = 2 * iw * ih, cin = kGbU0, cout = 256;
    const GbArena ar = gbArena(iw, ih);
    void* flow = s + ar.flow;
    void* qk = s + ar.qk;
    void* prop = s + ar.prop;
    void* u0 = s + ar.u0;
    void* y0 = s + ar.y0;
    void* lg = s + ar.lg;
    const char* wb = (const char*)nr.dGbW;
    const void* wp = wb + nr.gbOff[0];
    const void* bp = wb + nr.gbOff[1];
    const void* w0 = wb + nr.gbOff[2];
    const void* b0 = wb + nr.gbOff[3];
    const void* wm = wb + nr.gbOff[4];
    const void* bm = wb + nr.gbOff[5];
    const void* nul = nullptr;
    float zero = 0.0f;
    auto go = [&](int f, int gx, int gy, int gz, int threads, int sm, void** a) {
        return cuLaunchKernel(nr.fGb[f], (unsigned)gx, (unsigned)gy, (unsigned)gz, (unsigned)threads, 1, 1,
                              (unsigned)sm, (CUstream)st, a, nullptr) == CUDA_SUCCESS;
    };
    {
        void* a[] = {(void*)&corr, (void*)&flow0, &flow, &ih, &iw, &nb};
        if (!go(kGbFlow, (m + 255) / 256, 1, 1, 256, 0, a))
            return false;
    }
    {
        void* a[] = {(void*)&qf, &m, (void*)&wp, (void*)&bp, &qk};
        if (!go(kGbProj, (m + 63) / 64, 4, 1, 128, kGbSmProj, a))
            return false;
    }
    {
        void* a[] = {&qk, (void*)&qf, &flow, &prop, &u0, &ih, &iw, &nb};
        if (!go(kGbProp, (m + 7) / 8, 1, 1, 256, 0, a))
            return false;
    }
    for (int d = 0; d < 2; d++)
    {
        void* x = (uint8_t*)u0 + (size_t)d * iw * ih * kGbU0 * sizeof(uint16_t);
        void* y = (uint8_t*)y0 + (size_t)d * iw * ih * 256 * sizeof(uint16_t);
        void* a[] = {&x, &cin, &ih, &iw, &y, &cout, (void*)&w0, (void*)&b0, &zero, (void*)&nul, (void*)&nul};
        if (!go(kGbConv, (iw + 31) / 32, (ih + 7) / 8, 4, 128, kFuSmW64, a))
            return false;
    }
    {
        void* a[] = {&y0, &m, (void*)&wm, (void*)&bm, &lg};
        if (!go(kGbMask, (m + 63) / 64, 3, 1, 128, kGbSmMask, a))
            return false;
    }
    void* a[] = {&lg, &prop, (void*)&out, &ih, &iw, &nb};
    return go(kGbUp, (m * 16 + 255) / 256, 1, 1, 256, 0, a);
}

// gmflow B on one pair on st (nr.stream): corr (dGmCorr) and A's two outputs in, the half's flow into dGmFlow; the
// launches go through nr.gbGraphs (chainGraphRun). false = the activations or a launch failed
static bool nativeGbRun(NativeRife& nr, cudaStream_t st)
{
    const GbArena ar = gbArena(nr.gmW, nr.gmH);
    uint8_t* s = nr.dGmCtxMem && (size_t)nr.gmCtxMemB >= ar.total ? (uint8_t*)nr.dGmCtxMem : (uint8_t*)nr.dGbOwn;
    if (!s && cudaMalloc(&nr.dGbOwn, ar.total) == cudaSuccess)
        s = (uint8_t*)nr.dGbOwn;
    if (!s)
        return false;
    const float* corr = nr.dGmCorr;
    const void* qf = nr.dGmA[0];
    const float* flow0 = (const float*)nr.dGmA[1];
    return chainGraphRun(nr.gbGraphs, {s, corr, qf, flow0, nr.dGmFlow}, nr.gmW, nr.gmH, st, "gmfss gmflow B",
                         [&] { return nativeGbLaunch(nr, corr, qf, flow0, nr.dGmFlow, s, st); });
}
