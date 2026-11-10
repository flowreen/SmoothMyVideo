// part of smv-live-native.inl (included there, in place); not a standalone translation unit
// gmflow's transformer at both of GMFSS's scales on the host's kernels, the module smv_gmflowa_*.cubin (kFnHead's
// helpers + kGaA / kGaB, the kernel text cut in two), weights = engine\GMFSS_Fortuna\train_log\flownet.pkl's transformer.* and
// feature_flow_attn's two projections through pthStateDict (gaNatLoad); no fallback (a missing /
// wrong pkl = `gmflow's transformer unavailable (...)`, the session refuses). A (scale 1, the quarter, 8 x 8 windows):
// k_gaFlowUp (prop -> flow0), k_gaWarpPos (map4's rows, feature1 warped by flow0, + the window position), 6 layers of
// k_gaProj (the folded q' / v' of both attentions), k_gaAttn self (+ norm1 + the residual), k_gaProj (the cross q'),
// k_gaAttn cross (+ norm1), k_gaMlp (+ norm2 + the residual) -> qf (dGmA[0]) and flow0 (dGmA[1]); `_m` (scale 0, the
// eighth, 2 x 2 windows, map8's pair, no warp) the same layers, then the propagation's projections -> mf / mq / mk
// (dGmM). fp16 operands, fp16 sums but in P v' (fp32), LayerNorm, softmax and the residual stream in fp32; an RTX 20
// card runs 32-key tiles and a 2-stage weight ring (64 KB of shared memory a block).
static const char kGaA[] = R"GASRC(
// gmflow A, GMFSS's scale 1: the prop flow upsampled, feature1 warped by it, the window position, then the 6-layer
// transformer in 8 x 8 window splits (shifted windows on the odd layers). Tokens [4][P][128], P = H * W: rows 0..1 =
// feature0 (img0, img1), rows 2..3 = feature1 (img1 warped by flow 0, img0 by flow 1); the cross attention of row b
// reads row b ^ 2. Every matmul takes fp16 operands with fp32 sums; LayerNorm, softmax and the residual stream stay
// fp32 (x32), with an fp16 copy of the stream for the matmuls.

__device__ __forceinline__ unsigned gaSa(const void* p)
{
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void gaCommit()
{
    asm volatile(SMV_CP_COMMIT);
}

template <int N> __device__ __forceinline__ void gaWait()
{
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

__device__ __forceinline__ void gaLdm4T(unsigned addr, unsigned* r)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

__device__ __forceinline__ void gaMma(float* c, const unsigned* a, unsigned b0, unsigned b1)
{
    asm volatile(SMV_MMA_F32 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ unsigned gaPack(float lo, float hi)
{
    return static_cast<unsigned>(f2h(lo)) | (static_cast<unsigned>(f2h(hi)) << 16);
}

// the same product on fp16 accumulators: c[0] = row g (two columns), c[1] = row g + 8
__device__ __forceinline__ void gaMmaH(unsigned* c, const unsigned* a, unsigned b0, unsigned b1)
{
    asm volatile(SMV_MMA_F16 "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(c[0]), "+r"(c[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float gaLo(unsigned v)
{
    return h2f(static_cast<unsigned short>(v & 0xffffu));
}

__device__ __forceinline__ float gaHi(unsigned v)
{
    return h2f(static_cast<unsigned short>(v >> 16));
}

// the 16-byte chunk c of row r in a tile of 256-byte rows, swizzled so the 8 rows of an ldmatrix hit 8 bank groups
__device__ __forceinline__ int gaSw(int r, int c)
{
    return r * 128 + ((c ^ (r & 7)) << 3);
}

// flow0 [2][2][H][W] fp32 = 2 x prop [2][h8 * w8][2] fp16 (x, y) upsampled 2x, bilinear with aligned corners
extern "C" __global__ void __launch_bounds__(256) k_gaFlowUp(const unsigned short* __restrict__ prop, int h8, int w8,
                                                             float* __restrict__ flow)
{
    const int H = 2 * h8, W = 2 * w8, P = H * W;
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= 2 * P)
        return;
    const int d = i / P, p = i - d * P, y = p / W, x = p - y * W;
    const float fy = (H > 1 ? static_cast<float>(h8 - 1) / static_cast<float>(H - 1) : 0.f) * y;
    const float fx = (W > 1 ? static_cast<float>(w8 - 1) / static_cast<float>(W - 1) : 0.f) * x;
    const int y0 = static_cast<int>(fy), x0 = static_cast<int>(fx);
    const int y1 = y0 + (y0 < h8 - 1 ? 1 : 0), x1 = x0 + (x0 < w8 - 1 ? 1 : 0);
    const float ly = fy - y0, lx = fx - x0;
    const unsigned short* s = prop + static_cast<size_t>(d) * h8 * w8 * 2;
#pragma unroll
    for (int c = 0; c < 2; ++c) {
        const float v00 = h2f(s[(y0 * w8 + x0) * 2 + c]), v01 = h2f(s[(y0 * w8 + x1) * 2 + c]);
        const float v10 = h2f(s[(y1 * w8 + x0) * 2 + c]), v11 = h2f(s[(y1 * w8 + x1) * 2 + c]);
        const float v = (1.f - ly) * ((1.f - lx) * v00 + lx * v01) + ly * ((1.f - lx) * v10 + lx * v11);
        flow[(static_cast<size_t>(d) * 2 + c) * P + p] = 2.f * v;
    }
}

// the tokens, each + the window's sine position (posY [wh][64] = channels 0..63, posX [ww][64] = 64..127), maps =
// [2][128][H][W] fp16. warp (A, map4): rows 0 / 1 = maps[0] / [1], row 2 = maps[1] warped by flow 0, row 3 = maps[0]
// warped by flow 1 (bilinear, zero outside); no warp (`_m`, map8): rows 0 / 1 = maps[0] / [1]. 32 tokens a block:
// channel-planar reads, then token rows through shared memory.
extern "C" __global__ void __launch_bounds__(256)
    k_gaWarpPos(const unsigned short* __restrict__ m4, const float* __restrict__ flow, const float* __restrict__ posY,
                const float* __restrict__ posX, int H, int W, int wh, int ww, int warp, float* __restrict__ x32,
                unsigned short* __restrict__ x16)
{
    __shared__ float tile[32][129];
    const int P = H * W, b = blockIdx.y, t0 = blockIdx.x * 32;
    const int tx = threadIdx.x & 31, cy = threadIdx.x >> 5;
    const int t = t0 + tx;
    const bool ok = t < P;
    const int y = ok ? t / W : 0, x = ok ? t - y * W : 0;
    const bool warped = warp && b >= 2;
    const unsigned short* plane = m4 + static_cast<size_t>(warp ? ((b == 0 || b == 3) ? 0 : 1) : b) * 128 * P;
    int i00 = t, i01 = t, i10 = t, i11 = t;
    float w00 = 0.f, w01 = 0.f, w10 = 0.f, w11 = 0.f;
    if (warped && ok) {
        const float* fl = flow + static_cast<size_t>(b - 2) * 2 * P;
        const float sx = fminf(fmaxf(x + fl[t], -2.f), W + 1.f), sy = fminf(fmaxf(y + fl[P + t], -2.f), H + 1.f);
        const float fx0 = floorf(sx), fy0 = floorf(sy);
        const int xa = static_cast<int>(fx0), ya = static_cast<int>(fy0);
        const float ax = sx - fx0, ay = sy - fy0;
        const bool vx0 = xa >= 0 && xa < W, vx1 = xa + 1 >= 0 && xa + 1 < W;
        const bool vy0 = ya >= 0 && ya < H, vy1 = ya + 1 >= 0 && ya + 1 < H;
        const int cx0 = min(max(xa, 0), W - 1), cx1 = min(max(xa + 1, 0), W - 1);
        const int cy0 = min(max(ya, 0), H - 1), cy1 = min(max(ya + 1, 0), H - 1);
        w00 = (vx0 && vy0) ? (1.f - ax) * (1.f - ay) : 0.f;
        w01 = (vx1 && vy0) ? ax * (1.f - ay) : 0.f;
        w10 = (vx0 && vy1) ? (1.f - ax) * ay : 0.f;
        w11 = (vx1 && vy1) ? ax * ay : 0.f;
        i00 = cy0 * W + cx0;
        i01 = cy0 * W + cx1;
        i10 = cy1 * W + cx0;
        i11 = cy1 * W + cx1;
    }
    const float* py = posY + (y % wh) * 64;
    const float* px = posX + (x % ww) * 64;
#pragma unroll 4
    for (int j = 0; j < 16; ++j) {
        const int c = cy + 8 * j;
        float v = 0.f;
        if (ok) {
            const unsigned short* pl = plane + static_cast<size_t>(c) * P;
            if (!warped)
                v = h2f(pl[t]);
            else
                v = w00 * h2f(pl[i00]) + w01 * h2f(pl[i01]) + w10 * h2f(pl[i10]) + w11 * h2f(pl[i11]);
            v += c < 64 ? py[c] : px[c - 64];
        }
        tile[tx][c] = v;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int idx = k * 256 + threadIdx.x, r = idx >> 5, q = idx & 31;
        if (t0 + r >= P)
            continue;
        const float4 v = make_float4(tile[r][q * 4], tile[r][q * 4 + 1], tile[r][q * 4 + 2], tile[r][q * 4 + 3]);
        const size_t row = (static_cast<size_t>(b) * P + t0 + r) * 128 + q * 4;
        *reinterpret_cast<float4*>(x32 + row) = v;
        *reinterpret_cast<uint2*>(x16 + row) = make_uint2(gaPack(v.x, v.y), gaPack(v.z, v.w));
    }
}

// out[n / 128][m][n % 128] = a[m][:] . w[n][:] (+ bias[n] in fp32 when bias is set; K = 128, fp16 out): 64 rows x 128
// outputs a CTA, 4 warps of 32 x 64, the two K halves loaded as two cp.async groups; a 1D grid of row tiles x nbc
// output blocks with a row tile's nbc CTAs adjacent, so the later ones read its 64 rows from L2 instead of DRAM
extern "C" __global__ void __launch_bounds__(128) k_gaProj(const unsigned short* __restrict__ a, int M,
                                                           const unsigned short* __restrict__ w,
                                                           unsigned short* __restrict__ out,
                                                           const float* __restrict__ bias, int nbc)
{
    extern __shared__ __align__(16) unsigned char sm[];
    unsigned short* sA = reinterpret_cast<unsigned short*>(sm);
    unsigned short* sW = sA + 64 * 128;
    const int m0 = static_cast<int>(blockIdx.x) / nbc * 64, nb = static_cast<int>(blockIdx.x) % nbc;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const unsigned short* wb = w + static_cast<size_t>(nb) * 128 * 128;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int idx = k * 128 + tid, r = idx >> 3, c = (idx & 7) + h * 8, gr = m0 + r;
            const bool ok = gr < M;
            rnCp16(gaSa(sA + gaSw(r, c)), a + static_cast<size_t>(ok ? gr : 0) * 128 + c * 8, ok ? 16 : 0);
        }
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int idx = k * 128 + tid, r = idx >> 3, c = (idx & 7) + h * 8;
            rnCp16(gaSa(sW + gaSw(r, c)), wb + static_cast<size_t>(r) * 128 + c * 8, 16);
        }
        gaCommit();
    }
    const int wm = (warp >> 1) * 32, wn = (warp & 1) * 64;
    unsigned acc[2][8][2];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 8; ++nt)
            acc[mt][nt][0] = acc[mt][nt][1] = 0u;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        if (h == 0)
            gaWait<1>();
        else
            gaWait<0>();
        __syncthreads();
#pragma unroll
        for (int ks = h * 4; ks < h * 4 + 4; ++ks) {
            unsigned af[2][4];
#pragma unroll
            for (int mt = 0; mt < 2; ++mt)
                rnLdm4(gaSa(sA + gaSw(wm + mt * 16 + (lane & 15), ks * 2 + (lane >> 4))), af[mt]);
#pragma unroll
            for (int np = 0; np < 4; ++np) {
                unsigned bf[4];
                rnLdm4(gaSa(sW + gaSw(wn + np * 16 + (lane & 7) + ((lane >> 4) << 3), ks * 2 + ((lane >> 3) & 1))),
                       bf);
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    gaMmaH(acc[mt][2 * np], af[mt], bf[0], bf[1]);
                    gaMmaH(acc[mt][2 * np + 1], af[mt], bf[2], bf[3]);
                }
            }
        }
    }
    unsigned short* o = out + static_cast<size_t>(nb) * M * 128;
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
        const int r0 = m0 + wm + mt * 16 + (lane >> 2), r1 = r0 + 8;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const int col = wn + nt * 8 + (lane & 3) * 2;
            unsigned v0 = acc[mt][nt][0], v1 = acc[mt][nt][1];
            if (bias) {
                const float b0 = bias[nb * 128 + col], b1 = bias[nb * 128 + col + 1];
                v0 = gaPack(gaLo(v0) + b0, gaHi(v0) + b1);
                v1 = gaPack(gaLo(v1) + b0, gaHi(v1) + b1);
            }
            if (r0 < M)
                *reinterpret_cast<unsigned*>(o + static_cast<size_t>(r0) * 128 + col) = v0;
            if (r1 < M)
                *reinterpret_cast<unsigned*>(o + static_cast<size_t>(r1) * 128 + col) = v1;
        }
    }
}

// LayerNorm over the 128 channels of a warp's rows g and g + 8, in place: o[nt][0..1] = row g, o[nt][2..3] = row g + 8,
// channel nt * 8 + 2 * t4 + e (a row's 128 values live in its quad)
__device__ __forceinline__ void gaLn(float (*o)[4], const float* __restrict__ gam, const float* __restrict__ bet,
                                     float eps, int t4)
{
    float mu[2] = {0.f, 0.f}, va[2] = {0.f, 0.f};
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
        mu[0] += o[nt][0] + o[nt][1];
        mu[1] += o[nt][2] + o[nt][3];
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        mu[r] += __shfl_xor_sync(0xffffffffu, mu[r], 1);
        mu[r] += __shfl_xor_sync(0xffffffffu, mu[r], 2);
        mu[r] *= 1.f / 128.f;
    }
#pragma unroll
    for (int nt = 0; nt < 16; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float d = o[nt][e] - mu[e >> 1];
            va[e >> 1] += d * d;
        }
    float rs[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        va[r] += __shfl_xor_sync(0xffffffffu, va[r], 1);
        va[r] += __shfl_xor_sync(0xffffffffu, va[r], 2);
        rs[r] = rsqrtf(va[r] * (1.f / 128.f) + eps);
    }
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
        const int c = nt * 8 + 2 * t4;
        const float g0 = gam[c], g1 = gam[c + 1], b0 = bet[c], b1 = bet[c + 1];
        o[nt][0] = (o[nt][0] - mu[0]) * rs[0] * g0 + b0;
        o[nt][1] = (o[nt][1] - mu[0]) * rs[0] * g1 + b1;
        o[nt][2] = (o[nt][2] - mu[1]) * rs[1] * g0 + b0;
        o[nt][3] = (o[nt][3] - mu[1]) * rs[1] * g1 + b1;
    }
}

// the rows' stores (row = element offset of the token's 128 channels, -1 = none): add = x32 += o, o16 = fp16(x32);
// else o16 = fp16(o)
__device__ __forceinline__ void gaStore(float (*o)[4], long long row0, long long row1, int t4, bool add,
                                        float* __restrict__ x32, unsigned short* __restrict__ o16)
{
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const long long row = r ? row1 : row0;
        if (row < 0)
            continue;
#pragma unroll
        for (int nt = 0; nt < 16; ++nt) {
            const long long i = row + nt * 8 + 2 * t4;
            float v0 = o[nt][2 * r], v1 = o[nt][2 * r + 1];
            if (add) {
                const float2 s = *reinterpret_cast<const float2*>(x32 + i);
                v0 += s.x;
                v1 += s.y;
                *reinterpret_cast<float2*>(x32 + i) = make_float2(v0, v1);
            }
            *reinterpret_cast<unsigned*>(o16 + i) = gaPack(v0, v1);
        }
    }
}

)GASRC";
static const char kGaB[] = R"GASRC(
// local index i of window (wy, wx) on the rolled grid -> its token y * W + x; lab = its shift region (0..8)
__device__ __forceinline__ int gaTok(int i, int wy, int wx, int wh, int ww, int H, int W, int sh, int sw, int& lab)
{
    const int ly = i / ww, lx = i - ly * ww;
    const int ry = wy * wh + ly, rx = wx * ww + lx;
    lab = ((ry >= H - wh) + (ry >= H - sh)) * 3 + (rx >= W - ww) + (rx >= W - sw);
    const int y = ry + sh < H ? ry + sh : ry + sh - H;
    const int x = rx + sw < W ? rx + sw : rx + sw - W;
    return y * W + x;
}

// the key tile: 64 keys double buffered (66 KB of shared memory); sm_75 (RTX 20: 64 KB a block) = 32 keys
#if __CUDA_ARCH__ >= 800
#define GA_KT 64
#else
#define GA_KT 32
#endif

// one window's attention for 128 queries a CTA (8 warps of 16): S = q' k^T / sqrt(128) with the shifted layers' region
// mask (pairs of different regions skipped), online softmax in fp32, O = P v' with P in fp16, then norm1. K / V tiles of
// KT keys, double buffered. The frame is split in S x S windows (blockIdx.y = the window); keys and values come from
// row b ^ part. mode 0 (self) = x32 += norm1(O), o16 = fp16(x32); mode 1 (cross) = o16 = fp16(norm1(O)). Two entry
// points: k_gaAttn (GA_KT-key tiles, one CTA an SM) and k_gaAttn2 (32-key tiles, two CTAs an SM in 33 KB, 128
// registers) for gmflow A's calls, where the second CTA hides the first one's waits (1.02-1.17x from 540p to 4K); the
// 32-key tiles split the online softmax differently (1-2 fp16 ulps of the output, as near the fp64 result as 64 keys).
template <int KT>
__device__ __forceinline__ void gaAttnBody(const unsigned short* __restrict__ q, const unsigned short* __restrict__ k,
                                           const unsigned short* __restrict__ v, int H, int W, int S, int shift,
                                           int part, const float* __restrict__ gam, const float* __restrict__ bet,
                                           float eps, int mode, float* __restrict__ x32, unsigned short* __restrict__ o16)
{
    constexpr int KN = KT / 8;
    extern __shared__ __align__(16) unsigned char sm[];
    unsigned short* sK = reinterpret_cast<unsigned short*>(sm);
    unsigned short* sV = sK + 2 * KT * 128;
    int* sLab = reinterpret_cast<int*>(sV + 2 * KT * 128);
    const int wh = H / S, ww = W / S;
    const int P = H * W, L = wh * ww, sh = shift ? wh / 2 : 0, sw = shift ? ww / 2 : 0;
    const int b = blockIdx.z, wy = blockIdx.y / S, wx = blockIdx.y - wy * S, bk = b ^ part;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;

    int qlab[2];
    long long qrow[2];
    unsigned qa[8][4];
    {
        const unsigned* qp[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int li = blockIdx.x * 128 + warp * 16 + g + 8 * r;
            int lab = 0, tok = 0;
            if (li < L)
                tok = gaTok(li, wy, wx, wh, ww, H, W, sh, sw, lab);
            qlab[r] = lab;
            qrow[r] = li < L ? (static_cast<long long>(b) * P + tok) * 128 : -1;
            qp[r] = reinterpret_cast<const unsigned*>(q + (li < L ? qrow[r] : 0));
        }
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            qa[ks][0] = qrow[0] >= 0 ? qp[0][ks * 8 + t4] : 0u;
            qa[ks][1] = qrow[1] >= 0 ? qp[1][ks * 8 + t4] : 0u;
            qa[ks][2] = qrow[0] >= 0 ? qp[0][ks * 8 + 4 + t4] : 0u;
            qa[ks][3] = qrow[1] >= 0 ? qp[1][ks * 8 + 4 + t4] : 0u;
        }
    }

    const int nkt = (L + KT - 1) / KT;
    auto load = [&](int kt, int buf) {
#pragma unroll
        for (int j = 0; j < KT / 16; ++j) {
            const int idx = j * 256 + tid, r = idx >> 4, c = idx & 15, li = kt * KT + r;
            int lab = 0, tok = 0;
            if (li < L)
                tok = gaTok(li, wy, wx, wh, ww, H, W, sh, sw, lab);
            const size_t off = (static_cast<size_t>(bk) * P + tok) * 128 + c * 8;
            const int dst = (buf * KT) * 128 + gaSw(r, c);
            rnCp16(gaSa(sK + dst), k + off, li < L ? 16 : 0);
            rnCp16(gaSa(sV + dst), v + off, li < L ? 16 : 0);
            if (c == 0)
                sLab[buf * KT + r] = li < L ? (shift ? lab : 0) : -1;
        }
        gaCommit();
    };

    const float sc = 1.4426950408889634f * 0.08838834764831845f;
    const float ninf = __int_as_float(0xff800000);
    float m[2] = {-1e30f, -1e30f}, l[2] = {0.f, 0.f};
    float o[16][4];
#pragma unroll
    for (int nt = 0; nt < 16; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e)
            o[nt][e] = 0.f;

    load(0, 0);
    for (int kt = 0; kt < nkt; ++kt) {
        const int buf = kt & 1;
        gaWait<0>();
        __syncthreads();
        if (kt + 1 < nkt)
            load(kt + 1, buf ^ 1);
        const unsigned short* Kb = sK + buf * KT * 128;
        const unsigned short* Vb = sV + buf * KT * 128;
        const int* lb = sLab + buf * KT;
        float s[KN][4];
        {
            unsigned sh[KN][2];
#pragma unroll
            for (int j = 0; j < KN; ++j)
                sh[j][0] = sh[j][1] = 0u;
#pragma unroll
            for (int ks = 0; ks < 8; ++ks)
#pragma unroll
                for (int np = 0; np < KN / 2; ++np) {
                    unsigned bf[4];
                    rnLdm4(gaSa(Kb + gaSw(np * 16 + (lane & 7) + ((lane >> 4) << 3), ks * 2 + ((lane >> 3) & 1))),
                           bf);
                    gaMmaH(sh[2 * np], qa[ks], bf[0], bf[1]);
                    gaMmaH(sh[2 * np + 1], qa[ks], bf[2], bf[3]);
                }
#pragma unroll
            for (int j = 0; j < KN; ++j) {
                s[j][0] = gaLo(sh[j][0]);
                s[j][1] = gaHi(sh[j][0]);
                s[j][2] = gaLo(sh[j][1]);
                s[j][3] = gaHi(sh[j][1]);
            }
        }
        float mx[2] = {m[0], m[1]};
#pragma unroll
        for (int j = 0; j < KN; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int kl = lb[j * 8 + 2 * t4 + (e & 1)], r = e >> 1;
                const float x = (kl < 0 || (shift && kl != qlab[r])) ? ninf : s[j][e] * sc;
                s[j][e] = x;
                mx[r] = fmaxf(mx[r], x);
            }
        float al[2], ps[2] = {0.f, 0.f};
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
            al[r] = exp2f(m[r] - mx[r]);
            m[r] = mx[r];
        }
#pragma unroll
        for (int j = 0; j < KN; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float p = exp2f(s[j][e] - m[e >> 1]);
                s[j][e] = p;
                ps[e >> 1] += p;
            }
#pragma unroll
        for (int r = 0; r < 2; ++r)
            l[r] = l[r] * al[r] + ps[r];
#pragma unroll
        for (int nt = 0; nt < 16; ++nt) {
            o[nt][0] *= al[0];
            o[nt][1] *= al[0];
            o[nt][2] *= al[1];
            o[nt][3] *= al[1];
        }
#pragma unroll
        for (int kk = 0; kk < KN / 2; ++kk) {
            const unsigned pa[4] = {gaPack(s[2 * kk][0], s[2 * kk][1]), gaPack(s[2 * kk][2], s[2 * kk][3]),
                                    gaPack(s[2 * kk + 1][0], s[2 * kk + 1][1]),
                                    gaPack(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
            for (int ntp = 0; ntp < 8; ++ntp) {
                unsigned bf[4];
                gaLdm4T(gaSa(Vb + gaSw(kk * 16 + (lane & 7) + (((lane >> 3) & 1) << 3), ntp * 2 + (lane >> 4))), bf);
                gaMma(o[2 * ntp], pa, bf[0], bf[1]);
                gaMma(o[2 * ntp + 1], pa, bf[2], bf[3]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        l[r] += __shfl_xor_sync(0xffffffffu, l[r], 1);
        l[r] += __shfl_xor_sync(0xffffffffu, l[r], 2);
        l[r] = 1.f / l[r];
    }
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
        o[nt][0] *= l[0];
        o[nt][1] *= l[0];
        o[nt][2] *= l[1];
        o[nt][3] *= l[1];
    }
    gaLn(o, gam, bet, eps, t4);
    gaStore(o, qrow[0], qrow[1], t4, mode == 0, x32, o16);
}

extern "C" __global__ void __launch_bounds__(256, 1)
    k_gaAttn(const unsigned short* __restrict__ q, const unsigned short* __restrict__ k,
             const unsigned short* __restrict__ v, int H, int W, int S, int shift, int part,
             const float* __restrict__ gam, const float* __restrict__ bet, float eps, int mode,
             float* __restrict__ x32, unsigned short* __restrict__ o16)
{
    gaAttnBody<GA_KT>(q, k, v, H, W, S, shift, part, gam, bet, eps, mode, x32, o16);
}

extern "C" __global__ void __launch_bounds__(256, 2)
    k_gaAttn2(const unsigned short* __restrict__ q, const unsigned short* __restrict__ k,
              const unsigned short* __restrict__ v, int H, int W, int S, int shift, int part,
              const float* __restrict__ gam, const float* __restrict__ bet, float eps, int mode,
              float* __restrict__ x32, unsigned short* __restrict__ o16)
{
    gaAttnBody<32>(q, k, v, H, W, S, shift, part, gam, bet, eps, mode, x32, o16);
}

#define GA_HC 32
#define GA_MST (GA_HC * 256 + 128 * GA_HC)
// the weight ring: 3 stages (72 KB of shared memory); sm_75 (RTX 20: 64 KB a block) = 2
#if __CUDA_ARCH__ >= 800
#define GA_MSTN 3
#else
#define GA_MSTN 2
#endif

// the cross layer's mlp on 128 tokens a CTA (8 warps of 16): mlp.0 (256 -> 1024, GELU) on cat(s16, m16) in 32-wide
// hidden chunks that stay in registers as mlp.2's (1024 -> 128) operand, then norm2 and the residual: x32 += norm2(..),
// x16 = fp16(x32). w0 = [1024][256]; w2c = mlp.2 in chunks [32][128][32]; weight chunks in an NST-stage cp.async
// ring. Two entry points, the same arithmetic in the same order (the same bytes): k_gaMlp (one CTA an SM, the GA_MSTN
// ring) and k_gaMlp2 (two CTAs an SM: a 2-stage ring of 48 KB, 128 registers) for grids of 3+ CTAs an SM, where the
// second CTA hides the first one's waits.
template <int NST>
__device__ __forceinline__ void gaMlpBody(const unsigned short* __restrict__ s16, const unsigned short* __restrict__ m16,
                                          int M, const unsigned short* __restrict__ w0,
                                          const unsigned short* __restrict__ w2c, const float* __restrict__ gam,
                                          const float* __restrict__ bet, float eps, float* __restrict__ x32,
                                          unsigned short* __restrict__ x16)
{
    extern __shared__ __align__(16) unsigned char sm[];
    unsigned short* st = reinterpret_cast<unsigned short*>(sm);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
    const int r0 = blockIdx.x * 128 + warp * 16 + g, r1 = r0 + 8;
    const bool ok0 = r0 < M, ok1 = r1 < M;

    auto load = [&](int hc, int s) {
        unsigned short* d0 = st + s * GA_MST;
        unsigned short* d2 = d0 + GA_HC * 256;
        const unsigned short* g0 = w0 + static_cast<size_t>(hc) * GA_HC * 256;
        const unsigned short* g2 = w2c + static_cast<size_t>(hc) * 128 * GA_HC;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int idx = j * 256 + tid, r = idx >> 5, c = idx & 31;
            rnCp16(gaSa(d0 + r * 256 + ((c ^ (r & 7)) << 3)), g0 + r * 256 + c * 8, 16);
        }
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int idx = j * 256 + tid, r = idx >> 2, c = idx & 3;
            rnCp16(gaSa(d2 + r * 32 + ((c ^ ((r >> 1) & 3)) << 3)), g2 + r * 32 + c * 8, 16);
        }
        gaCommit();
    };
#pragma unroll
    for (int s = 0; s < NST - 1; ++s)
        load(s, s);

    unsigned a[16][4];
    {
        const unsigned* p0 = reinterpret_cast<const unsigned*>(s16 + static_cast<size_t>(ok0 ? r0 : 0) * 128);
        const unsigned* p1 = reinterpret_cast<const unsigned*>(s16 + static_cast<size_t>(ok1 ? r1 : 0) * 128);
        const unsigned* n0 = reinterpret_cast<const unsigned*>(m16 + static_cast<size_t>(ok0 ? r0 : 0) * 128);
        const unsigned* n1 = reinterpret_cast<const unsigned*>(m16 + static_cast<size_t>(ok1 ? r1 : 0) * 128);
#pragma unroll
        for (int ks = 0; ks < 16; ++ks) {
            const unsigned* q0 = ks < 8 ? p0 : n0;
            const unsigned* q1 = ks < 8 ? p1 : n1;
            const int kc = (ks & 7) * 8;
            a[ks][0] = ok0 ? q0[kc + t4] : 0u;
            a[ks][1] = ok1 ? q1[kc + t4] : 0u;
            a[ks][2] = ok0 ? q0[kc + 4 + t4] : 0u;
            a[ks][3] = ok1 ? q1[kc + 4 + t4] : 0u;
        }
    }
    float o[16][4];
    unsigned oh[16][2];
#pragma unroll
    for (int nt = 0; nt < 16; ++nt)
        oh[nt][0] = oh[nt][1] = 0u;

    for (int hc = 0; hc < 1024 / GA_HC; ++hc) {
        gaWait<NST - 2>();
        __syncthreads();
        if (hc + NST - 1 < 1024 / GA_HC)
            load(hc + NST - 1, (hc + NST - 1) % NST);
        else
            gaCommit();
        const unsigned short* d0 = st + (hc % NST) * GA_MST;
        const unsigned short* d2 = d0 + GA_HC * 256;
        float h[4][4];
        {
            unsigned hh[4][2];
#pragma unroll
            for (int j = 0; j < 4; ++j)
                hh[j][0] = hh[j][1] = 0u;
#pragma unroll
            for (int ks = 0; ks < 16; ++ks)
#pragma unroll
                for (int np = 0; np < 2; ++np) {
                    const int n = np * 16 + (lane & 7) + ((lane >> 4) << 3), c = ks * 2 + ((lane >> 3) & 1);
                    unsigned bf[4];
                    rnLdm4(gaSa(d0 + n * 256 + ((c ^ (n & 7)) << 3)), bf);
                    gaMmaH(hh[2 * np], a[ks], bf[0], bf[1]);
                    gaMmaH(hh[2 * np + 1], a[ks], bf[2], bf[3]);
                }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                h[j][0] = gaLo(hh[j][0]);
                h[j][1] = gaHi(hh[j][0]);
                h[j][2] = gaLo(hh[j][1]);
                h[j][3] = gaHi(hh[j][1]);
            }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e)
                h[j][e] = 0.5f * h[j][e] * (1.f + erff(h[j][e] * 0.7071067811865476f));
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            const unsigned pa[4] = {gaPack(h[2 * kk][0], h[2 * kk][1]), gaPack(h[2 * kk][2], h[2 * kk][3]),
                                    gaPack(h[2 * kk + 1][0], h[2 * kk + 1][1]),
                                    gaPack(h[2 * kk + 1][2], h[2 * kk + 1][3])};
#pragma unroll
            for (int ntp = 0; ntp < 8; ++ntp) {
                const int n = ntp * 16 + (lane & 7) + ((lane >> 4) << 3), c = kk * 2 + ((lane >> 3) & 1);
                unsigned bf[4];
                rnLdm4(gaSa(d2 + n * 32 + ((c ^ ((n >> 1) & 3)) << 3)), bf);
                gaMmaH(oh[2 * ntp], pa, bf[0], bf[1]);
                gaMmaH(oh[2 * ntp + 1], pa, bf[2], bf[3]);
            }
        }
    }
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
        o[nt][0] = gaLo(oh[nt][0]);
        o[nt][1] = gaHi(oh[nt][0]);
        o[nt][2] = gaLo(oh[nt][1]);
        o[nt][3] = gaHi(oh[nt][1]);
    }
    gaLn(o, gam, bet, eps, t4);
    gaStore(o, ok0 ? static_cast<long long>(r0) * 128 : -1, ok1 ? static_cast<long long>(r1) * 128 : -1, t4, true, x32,
            x16);
}

extern "C" __global__ void __launch_bounds__(256, 1)
    k_gaMlp(const unsigned short* __restrict__ s16, const unsigned short* __restrict__ m16, int M,
            const unsigned short* __restrict__ w0, const unsigned short* __restrict__ w2c,
            const float* __restrict__ gam, const float* __restrict__ bet, float eps, float* __restrict__ x32,
            unsigned short* __restrict__ x16)
{
    gaMlpBody<GA_MSTN>(s16, m16, M, w0, w2c, gam, bet, eps, x32, x16);
}

extern "C" __global__ void __launch_bounds__(256, 2)
    k_gaMlp2(const unsigned short* __restrict__ s16, const unsigned short* __restrict__ m16, int M,
             const unsigned short* __restrict__ w0, const unsigned short* __restrict__ w2c,
             const float* __restrict__ gam, const float* __restrict__ bet, float eps, float* __restrict__ x32,
             unsigned short* __restrict__ x16)
{
    gaMlpBody<2>(s16, m16, M, w0, w2c, gam, bet, eps, x32, x16);
}
)GASRC";

static const char* const kGaKern[] = {"k_gaFlowUp", "k_gaWarpPos", "k_gaProj", "k_gaAttn",
                                      "k_gaMlp",    "k_gaMlp2",    "k_gaAttn2"};
enum
{
    kGaFlowUpK,
    kGaWarpPosK,
    kGaProjK,
    kGaAttnK,
    kGaMlpK,
    kGaMlp2K,
    kGaAttn2K,
    kGaKernN
};
// the weight block (gaNatLoad): layer l's pieces at 7 l: g1 = the self attention's q' and v' folds and the cross one's v'
// fold [384][128], g2 = the cross q' fold [128][128], mlp.0 [1024][256], mlp.2 in 32-wide hidden chunks [32][128][32]
// (fp16), then the three LayerNorms' gamma | beta (fp32 [2][128]: self norm1, cross norm1, norm2); 42..45 =
// feature_flow_attn's q_proj (fp16), its bias (fp32), k_proj, its bias
constexpr int kGaQw = 42, kGaQb = 43, kGaKw = 44, kGaKb = 45;
constexpr float kGaEps = 1e-5f;  // nn.LayerNorm's default (GMFlow's TransformerLayer)
constexpr int kGaSmProj = 49152; // k_gaProj: 64 rows + 128 weights of 256 bytes

static bool gaNatLoad(std::vector<uint8_t>& blob, size_t* off, std::string& why)
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
    const std::string fa = "feature_flow_attn.";
    bool form = has(fa + "q_proj.weight", {128, 128}) && has(fa + "q_proj.bias", {128}) &&
                has(fa + "k_proj.weight", {128, 128}) && has(fa + "k_proj.bias", {128}) &&
                !S.count("transformer.layers.6.self_attn.q_proj.weight");
    for (int l = 0; l < 6 && form; l++)
    {
        const std::string p = "transformer.layers." + std::to_string(l) + ".";
        for (const char* a : {"self_attn.", "cross_attn_ffn."})
        {
            for (const char* w : {"q_proj", "k_proj", "v_proj", "merge"})
                form = form && has(p + a + w + ".weight", {128, 128}) && !S.count(p + a + w + ".bias");
            form = form && has(p + a + "norm1.weight", {128}) && has(p + a + "norm1.bias", {128});
        }
        form = form && has(p + "cross_attn_ffn.mlp.0.weight", {1024, 256}) &&
               has(p + "cross_attn_ffn.mlp.2.weight", {128, 1024}) && has(p + "cross_attn_ffn.norm2.weight", {128}) &&
               has(p + "cross_attn_ffn.norm2.bias", {128}) && !S.count(p + "self_attn.mlp.0.weight");
    }
    if (!form)
    {
        why = "its weights are not gmflow's transformer";
        return false;
    }
    blob.clear();
    auto put = [&](const void* d, size_t n) {
        const size_t o = (blob.size() + 255) & ~(size_t)255;
        blob.resize(o + n);
        memcpy(blob.data() + o, d, n);
        return o;
    };
    auto half = [](const float* v, size_t n) {
        std::vector<uint16_t> h(n);
        for (size_t i = 0; i < n; i++)
            h[i] = f2hRne(v[i]);
        return h;
    };
    // the export's folds: q' = s (Wk^T Wq)^T, v' = t (Wm Wv)^T, in double, then fp32 (then fp16 by half)
    auto fold = [](std::span<const float> a, std::span<const float> b, bool kq, float* out) {
        for (int i = 0; i < 128; i++)
            for (int j = 0; j < 128; j++)
            {
                double s = 0.0;
                for (int r = 0; r < 128; r++)
                    s += kq ? (double)a[(size_t)r * 128 + i] * b[(size_t)r * 128 + j]
                            : (double)a[(size_t)i * 128 + r] * b[(size_t)r * 128 + j];
                out[i * 128 + j] = (float)s;
            }
    };
    std::vector<float> g1((size_t)384 * 128), g2((size_t)128 * 128), w2c((size_t)128 * 1024);
    for (int l = 0; l < 6; l++)
    {
        const std::string p = "transformer.layers." + std::to_string(l) + ".";
        const std::string sa = p + "self_attn.", ca = p + "cross_attn_ffn.";
        fold(wts[sa + "k_proj.weight"], wts[sa + "q_proj.weight"], true, g1.data());
        fold(wts[sa + "merge.weight"], wts[sa + "v_proj.weight"], false, g1.data() + 128 * 128);
        fold(wts[ca + "merge.weight"], wts[ca + "v_proj.weight"], false, g1.data() + 2 * 128 * 128);
        fold(wts[ca + "k_proj.weight"], wts[ca + "q_proj.weight"], true, g2.data());
        const std::vector<uint16_t> h1 = half(g1.data(), g1.size()), h2 = half(g2.data(), g2.size());
        off[7 * l] = put(h1.data(), h1.size() * sizeof(uint16_t));
        off[7 * l + 1] = put(h2.data(), h2.size() * sizeof(uint16_t));
        const std::span<const float> w0 = wts[ca + "mlp.0.weight"], w2 = wts[ca + "mlp.2.weight"];
        const std::vector<uint16_t> h0 = half(w0.data(), w0.size());
        off[7 * l + 2] = put(h0.data(), h0.size() * sizeof(uint16_t));
        for (int hc = 0; hc < 32; hc++)
            for (int n = 0; n < 128; n++)
                for (int j = 0; j < 32; j++)
                    w2c[((size_t)hc * 128 + n) * 32 + j] = w2[(size_t)n * 1024 + hc * 32 + j];
        const std::vector<uint16_t> hc2 = half(w2c.data(), w2c.size());
        off[7 * l + 3] = put(hc2.data(), hc2.size() * sizeof(uint16_t));
        const std::string ln[3] = {sa + "norm1.", ca + "norm1.", ca + "norm2."};
        for (int k = 0; k < 3; k++)
        {
            float gb[256];
            memcpy(gb, wts[ln[k] + "weight"].data(), 128 * sizeof(float));
            memcpy(gb + 128, wts[ln[k] + "bias"].data(), 128 * sizeof(float));
            off[7 * l + 4 + k] = put(gb, sizeof(gb));
        }
    }
    const std::vector<uint16_t> qw = half(wts[fa + "q_proj.weight"].data(), 128 * 128),
                                kw = half(wts[fa + "k_proj.weight"].data(), 128 * 128);
    off[kGaQw] = put(qw.data(), qw.size() * sizeof(uint16_t));
    off[kGaQb] = put(wts[fa + "q_proj.bias"].data(), 128 * sizeof(float));
    off[kGaKw] = put(kw.data(), kw.size() * sizeof(uint16_t));
    off[kGaKb] = put(wts[fa + "k_proj.bias"].data(), 128 * sizeof(float));
    return true;
}

// PositionEmbeddingSine(64, normalize) of one window's n rows (or columns), in double: out = [n][64]
static void gaPosTab(int n, float* out)
{
    for (int r = 0; r < n; r++)
    {
        const double a = (r + 1) / (n + 1e-6) * 2.0 * 3.14159265358979323846;
        for (int i = 0; i < 64; i++)
        {
            const double v = a / pow(10000.0, 2.0 * (i / 2) / 64.0);
            out[r * 64 + i] = (float)(i % 2 == 0 ? sin(v) : cos(v));
        }
    }
}

// the kernels' RTX 20 form (32-key tiles, a 2-stage weight ring) is the one NVRTC builds for sm_75 (or for
// SMV_NATIVE_ARCH below 80): the shared memory the host grants must match it
static bool gaTuring()
{
    if (g_nativeArch)
        return g_nativeArch < 80;
    int dev = 0, maj = 0;
    return cudaGetDevice(&dev) == cudaSuccess &&
           cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess && maj < 8;
}
static int gaSmAttn(bool two = false)
{
    const int kt = two || gaTuring() ? 32 : 64;
    return 4 * kt * 128 * 2 + 2 * kt * 4;
}
static int gaSmMlp(bool two = false)
{
    return (two || gaTuring() ? 2 : 3) * (32 * 256 + 128 * 32) * 2;
}

// the transformer's activations at A's scale (4 rows of the quarter's tokens; `_m`'s 2 rows of the eighth's fit
// inside), 256-byte pieces: the fp32 residual stream, the fp16 copy after the self attention, the three projections,
// the cross attention's normed message ([rows P][128] each)
struct GaArena
{
    size_t x32 = 0, xb = 0, pj = 0, nm = 0, total = 0;
};
static GaArena gaArena(int w4, int h4)
{
    GaArena a;
    const size_t m = (size_t)4 * w4 * h4;
    size_t o = 0;
    auto put = [&](size_t n) {
        const size_t at = o;
        o += (n + 255) & ~(size_t)255;
        return at;
    };
    a.x32 = put(m * 128 * sizeof(float));
    a.xb = put(m * 128 * sizeof(uint16_t));
    a.pj = put(3 * m * 128 * sizeof(uint16_t));
    a.nm = put(m * 128 * sizeof(uint16_t));
    a.total = o;
    return a;
}

static size_t gaNatBytes(const NativeRife& nr)
{
    return nr.gmfss && nr.gmW > 0 && nr.gmH > 0 ? gaArena(nr.gmW, nr.gmH).total : 0;
}

// this session's transformer module: the weights read and folded, its text compiled (or the cached cubin) and loaded,
// its kernels bound and granted their shared memory, the weight block uploaded, the window position tables of both
// scales uploaded (the quarter nr.gmW x nr.gmH); false = `why` (the session teardown unloads a module left in gaMod)
static bool nativeGaModule(NativeRife& nr, std::string& why)
{
    if (nr.gaMod)
        return true;
    std::vector<uint8_t> blob;
    if (!gaNatLoad(blob, nr.gaOff, why))
        return false;
    std::string text = kFnHead;
    text += kGaA;
    text += kGaB;
    std::vector<char> cubin;
    if (!nativeCubin(text.c_str(), "smv_gmflowa.cu", L"smv_gmflowa", "gmflow transformer kernels",
                     nr.kernCacheDir.empty() ? std::wstring(L".") : nr.kernCacheDir, cubin))
    {
        why = "its kernels did not compile";
        return false;
    }
    if (cuModuleLoadData(&nr.gaMod, cubin.data()) != CUDA_SUCCESS)
    {
        nr.gaMod = nullptr;
        why = "its kernel module did not load";
        return false;
    }
    static_assert(sizeof(kGaKern) / sizeof(kGaKern[0]) == kGaKernN && kGaKernN <= sizeof(nr.fGa) / sizeof(nr.fGa[0]),
                  "the gmflow kernel names, enum and handles agree");
    for (int i = 0; i < kGaKernN; i++)
        if (cuModuleGetFunction(&nr.fGa[i], nr.gaMod, kGaKern[i]) != CUDA_SUCCESS)
        {
            why = std::string("its kernel ") + kGaKern[i] + " is missing";
            return false;
        }
    if (cuFuncSetAttribute(nr.fGa[kGaAttnK], CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, gaSmAttn()) !=
            CUDA_SUCCESS ||
        cuFuncSetAttribute(nr.fGa[kGaMlpK], CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, gaSmMlp()) !=
            CUDA_SUCCESS ||
        cuFuncSetAttribute(nr.fGa[kGaMlp2K], CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, gaSmMlp(true)) !=
            CUDA_SUCCESS ||
        cuFuncSetAttribute(nr.fGa[kGaAttn2K], CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, gaSmAttn(true)) !=
            CUDA_SUCCESS)
    {
        why = "its kernels could not get their shared memory";
        return false;
    }
    // the mlp's two-CTA form (k_gaMlp2: the same bytes) from 3 CTAs an SM, where a second CTA hides the first one's
    // waits (1.04x at 256 CTAs on 82 SMs, 1.10x at 512 and 1020, 1.14x at 4080; 0.98x at 128); never on RTX 20 (two
    // 48 KB rings do not fit its 64 KB); SMV_GA_MLP2=0 = never
    {
        int dev = 0, sms = 0;
        nr.gaMlp2Min = !gaTuring() && lkEnv("SMV_GA_MLP2") != "0" && cudaGetDevice(&dev) == cudaSuccess &&
                               cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess
                           ? 3 * sms
                           : 0;
    }
    // gmflow A's attention in its two-CTA form (k_gaAttn2: 32-key tiles, 1.02-1.17x from 540p to 4K, as near gmflow's
    // fp64 output as k_gaAttn); `_m` keeps k_gaAttn (its random-map case fell 0.06-0.09 dB short there); never on RTX 20
    // (its k_gaAttn is the 32-key form already); SMV_GA_ATTN2=0 = never
    nr.gaAttn2 = !gaTuring() && lkEnv("SMV_GA_ATTN2") != "0";
    if (cudaMalloc(&nr.dGaW, blob.size()) != cudaSuccess ||
        cudaMemcpy(nr.dGaW, blob.data(), blob.size(), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        why = "its weights did not upload";
        return false;
    }
    const int ah = nr.gmH / 8, aw = nr.gmW / 8, mh = nr.gmH / 4, mw = nr.gmW / 4;
    std::vector<float> pos((size_t)(ah + aw + mh + mw) * 64);
    gaPosTab(ah, pos.data());
    gaPosTab(aw, pos.data() + (size_t)ah * 64);
    gaPosTab(mh, pos.data() + (size_t)(ah + aw) * 64);
    gaPosTab(mw, pos.data() + (size_t)(ah + aw + mh) * 64);
    if (cudaMalloc((void**)&nr.dGaPos, pos.size() * sizeof(float)) != cudaSuccess ||
        cudaMemcpy(nr.dGaPos, pos.data(), pos.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        why = "its position tables did not upload";
        return false;
    }
    return true;
}

// one transformer scale's launches on st: scale 1 = A (maps = map4 [2][128][H][W] fp16 at the quarter, prop [2][N8][2]
// fp16 -> xa = qf [4][P][128] fp16 + flow = flow0 [2][2][H][W] fp32), scale 0 = `_m` (maps = map8 at the eighth -> xa = mf
// [2][P][128] + mq / mk); the activations at s (gaArena's pieces). false = a launch failed
static bool nativeGaLaunch(NativeRife& nr, int scale, const void* maps, const void* prop, void* xa, void* flow,
                           void* mq, void* mk, uint8_t* s, cudaStream_t st)
{
    int th = scale ? nr.gmH : nr.gmH / 2, tw = scale ? nr.gmW : nr.gmW / 2, S = scale ? 8 : 2; // the token grid
    const int rows = scale ? 4 : 2, P = th * tw;
    int M = rows * P, wh = th / S, ww = tw / S, part = scale ? 2 : 1, warp = scale, zero = 0, one = 1, three = 3;
    const bool att2 = scale && nr.gaAttn2;
    const int kAtt = att2 ? kGaAttn2K : kGaAttnK;
    const unsigned att = (unsigned)((wh * ww + 127) / 128), smA = (unsigned)gaSmAttn(att2), smM = (unsigned)gaSmMlp();
    float eps = kGaEps;
    const GaArena ar = gaArena(nr.gmW, nr.gmH);
    void *x32 = s + ar.x32, *xb = s + ar.xb, *pj0 = s + ar.pj, *nm = s + ar.nm;
    void* pj1 = (uint8_t*)pj0 + (size_t)M * 128 * sizeof(uint16_t);
    void* pj2 = (uint8_t*)pj0 + (size_t)M * 256 * sizeof(uint16_t);
    const float* posY = nr.dGaPos + (scale ? 0 : (size_t)(nr.gmH / 8 + nr.gmW / 8) * 64);
    const float* posX = posY + (size_t)wh * 64;
    const uint8_t* wb = (const uint8_t*)nr.dGaW;
    void* nul = nullptr;
    auto go = [&](int k, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned sm, void** a) {
        return cuLaunchKernel(nr.fGa[k], gx, gy, gz, bx, 1, 1, sm, (CUstream)st, a, nullptr) == CUDA_SUCCESS;
    };
    if (scale)
    {
        int h8 = th / 2, w8 = tw / 2;
        void* a[] = {(void*)&prop, &h8, &w8, &flow};
        if (!go(kGaFlowUpK, (unsigned)((2 * P + 255) / 256), 1, 1, 256, 0, a))
            return false;
    }
    {
        void* a[] = {(void*)&maps, &flow, (void*)&posY, (void*)&posX, &th, &tw, &wh, &ww, &warp, &x32, &xa};
        if (!go(kGaWarpPosK, (unsigned)((P + 31) / 32), (unsigned)rows, 1, 256, 0, a))
            return false;
    }
    const unsigned gm = (unsigned)((M + 63) / 64), mlpCtas = (unsigned)((M + 127) / 128),
                   smM2 = (unsigned)gaSmMlp(true);
    const bool mlp2 = nr.gaMlp2Min > 0 && mlpCtas >= (unsigned)nr.gaMlp2Min;
    for (int l = 0; l < 6; l++)
    {
        int shift = l & 1;
        const void* g1 = wb + nr.gaOff[7 * l];
        const void* g2 = wb + nr.gaOff[7 * l + 1];
        const void* w0 = wb + nr.gaOff[7 * l + 2];
        const void* w2c = wb + nr.gaOff[7 * l + 3];
        const float* lnS = (const float*)(wb + nr.gaOff[7 * l + 4]);
        const float* lnC = (const float*)(wb + nr.gaOff[7 * l + 5]);
        const float* ln2 = (const float*)(wb + nr.gaOff[7 * l + 6]);
        const float *lnSb = lnS + 128, *lnCb = lnC + 128, *ln2b = ln2 + 128;
        void* a1[] = {&xa, &M, (void*)&g1, &pj0, &nul, &three};
        void* a2[] = {&pj0, &xa, &pj1, &th, &tw, &S, &shift, &zero, (void*)&lnS, (void*)&lnSb, &eps, &zero, &x32, &xb};
        void* a3[] = {&xb, &M, (void*)&g2, &pj0, &nul, &one};
        void* a4[] = {&pj0, &xa, &pj2, &th, &tw, &S, &shift, &part, (void*)&lnC, (void*)&lnCb, &eps, &one, &x32, &nm};
        void* a5[] = {&xb, &nm, &M, (void*)&w0, (void*)&w2c, (void*)&ln2, (void*)&ln2b, &eps, &x32, &xa};
        if (!go(kGaProjK, gm * 3, 1, 1, 128, kGaSmProj, a1) ||
            !go(kAtt, att, (unsigned)(S * S), (unsigned)rows, 256, smA, a2) ||
            !go(kGaProjK, gm, 1, 1, 128, kGaSmProj, a3) ||
            !go(kAtt, att, (unsigned)(S * S), (unsigned)rows, 256, smA, a4) ||
            !(mlp2 ? go(kGaMlp2K, mlpCtas, 1, 1, 256, smM2, a5) : go(kGaMlpK, mlpCtas, 1, 1, 256, smM, a5)))
            return false;
    }
    if (!scale)
    {
        const void *qw = wb + nr.gaOff[kGaQw], *kw = wb + nr.gaOff[kGaKw];
        const void *qb = wb + nr.gaOff[kGaQb], *kb = wb + nr.gaOff[kGaKb];
        void* a6[] = {&xa, &M, (void*)&qw, &mq, (void*)&qb, &one};
        void* a7[] = {&mq, &M, (void*)&kw, &mk, (void*)&kb, &one};
        if (!go(kGaProjK, gm, 1, 1, 128, kGaSmProj, a6) || !go(kGaProjK, gm, 1, 1, 128, kGaSmProj, a7))
            return false;
    }
    return true;
}

// one transformer scale on st (nr.stream) through nr.gaGraphs[scale] (chainGraphRun); the activations in the shared
// activation block when it holds them, else dGaOwn. false = the activations or a launch failed
static bool nativeGaRun(NativeRife& nr, int scale, const void* maps, const void* prop, void* xa, void* flow, void* mq,
                        void* mk, cudaStream_t st)
{
    const GaArena ar = gaArena(nr.gmW, nr.gmH);
    uint8_t* s = nr.dGmCtxMem && (size_t)nr.gmCtxMemB >= ar.total ? (uint8_t*)nr.dGmCtxMem : (uint8_t*)nr.dGaOwn;
    if (!s && cudaMalloc(&nr.dGaOwn, ar.total) == cudaSuccess)
        s = (uint8_t*)nr.dGaOwn;
    if (!s)
        return false;
    return chainGraphRun(nr.gaGraphs[scale], {s, maps, prop, xa, flow, mq, mk}, nr.gmW, nr.gmH, st,
                         scale ? "gmfss gmflow A" : "gmfss gmflow _m",
                         [&] { return nativeGaLaunch(nr, scale, maps, prop, xa, flow, mq, mk, s, st); });
}
