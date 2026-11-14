// One translation a frame as DLSS 4.5's motion vectors (dlssg2f and smv-live.exe include it after d3d12.h).
//
// SMV has no game vectors to give DLSS-G. With zero vectors DLSS-G's own flow has to find all motion, and on a fast
// camera pan (past ~60-90 px a frame) it loses track: the new frames melt or show people twice. The presenter measures
// how far the picture moved since the frame it presented before (luma thumbnails at 1/4, 1/16 and 1/32 of the size, a
// coarse-to-fine SAD search, a sub-pixel fit) and, when one translation explains the change, tags it as the vectors.
// Tiles a still picture explains better (subtitles, logos, the edge of a letterbox bar) keep zero vectors. Everything
// else keeps the exact zero vectors, so those frames stay the ones DLSS-G made without this.
#pragma once
#include <emmintrin.h>
#include <immintrin.h>
#include <intrin.h>

namespace panvec
{

// what the vector texture holds: off = all zero; on = (mx, my) pixels, current -> previous (a pixel sat at x + mx,
// y + my one frame earlier), except the `still` rectangles, which stay zero
struct Result
{
    bool on = false;
    float mx = 0, my = 0;
    std::vector<D3D12_RECT> still;

    bool same(const Result& o) const
    {
        if (on != o.on)
            return false;
        if (!on)
            return true;
        if (mx != o.mx || my != o.my || still.size() != o.still.size())
            return false;
        for (size_t i = 0; i < still.size(); i++)
            if (still[i].left != o.still[i].left || still[i].top != o.still[i].top ||
                still[i].right != o.still[i].right || still[i].bottom != o.still[i].bottom)
                return false;
        return true;
    }
};

// sum of |a[i] - b[i]| over n bytes
static inline uint64_t sadRow(const uint8_t* a, const uint8_t* b, int n)
{
    __m128i acc = _mm_setzero_si128();
    int i = 0;
    for (; i + 16 <= n; i += 16)
        acc = _mm_add_epi64(
            acc, _mm_sad_epu8(_mm_loadu_si128((const __m128i*)(a + i)), _mm_loadu_si128((const __m128i*)(b + i))));
    uint64_t s = (uint64_t)_mm_cvtsi128_si64(acc) + (uint64_t)_mm_cvtsi128_si64(_mm_unpackhi_epi64(acc, acc));
    for (; i < n; i++)
        s += (uint64_t)(a[i] > b[i] ? a[i] - b[i] : b[i] - a[i]);
    return s;
}

// the CPU runs AVX2 and the OS saves its registers (CPUID 7 EBX bit 5, OSXSAVE + AVX, XCR0 bits 1 and 2)
static inline bool hasAvx2()
{
    static const bool on = [] {
        int r[4];
        __cpuid(r, 0);
        if (r[0] < 7)
            return false;
        __cpuid(r, 1);
        if (!(r[2] & (1 << 27)) || !(r[2] & (1 << 28)) || (_xgetbv(0) & 6) != 6)
            return false;
        __cpuidex(r, 7, 0);
        return (r[1] & (1 << 5)) != 0;
    }();
    return on;
}

// sadRow 32 bytes a step (AVX2 CPUs only), then 16, then the bytes left
static inline uint64_t sadRow32(const uint8_t* a, const uint8_t* b, int n)
{
    __m256i acc = _mm256_setzero_si256();
    int i = 0;
    for (; i + 32 <= n; i += 32)
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(_mm256_loadu_si256((const __m256i*)(a + i)),
                                                    _mm256_loadu_si256((const __m256i*)(b + i))));
    __m128i h = _mm_add_epi64(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    if (i + 16 <= n)
    {
        h = _mm_add_epi64(
            h, _mm_sad_epu8(_mm_loadu_si128((const __m128i*)(a + i)), _mm_loadu_si128((const __m128i*)(b + i))));
        i += 16;
    }
    uint64_t s = (uint64_t)_mm_cvtsi128_si64(h) + (uint64_t)_mm_cvtsi128_si64(_mm_unpackhi_epi64(h, h));
    for (; i < n; i++)
        s += (uint64_t)(a[i] > b[i] ? a[i] - b[i] : b[i] - a[i]);
    return s;
}

// four 1/4 thumbnail pixels of an 8-bit RGBA frame: per 4 x 4 block (16 bytes of rows r1 and r2) the sum of
// R + 2 G + B over pixels 1 and 2 of both rows, >> 4 (the thumbnail loop's own integer sums)
static inline uint32_t thumbStep(const uint8_t* r1, const uint8_t* r2)
{
    const __m128i z = _mm_setzero_si128(), wt = _mm_setr_epi16(1, 2, 1, 0, 1, 2, 1, 0), one = _mm_set1_epi16(1);
    __m128i m[4];
    for (int k = 0; k < 4; k++)
    {
        // pixels 1 and 2 of the block as 16-bit channels, the two rows added: lanes R + 2 G, B for each pixel
        const __m128i a = _mm_srli_si128(_mm_loadu_si128((const __m128i*)(r1 + 16 * k)), 4);
        const __m128i b = _mm_srli_si128(_mm_loadu_si128((const __m128i*)(r2 + 16 * k)), 4);
        m[k] = _mm_madd_epi16(_mm_add_epi16(_mm_unpacklo_epi8(a, z), _mm_unpacklo_epi8(b, z)), wt);
    }
    const __m128i q01 = _mm_madd_epi16(_mm_packs_epi32(m[0], m[1]), one);
    const __m128i q23 = _mm_madd_epi16(_mm_packs_epi32(m[2], m[3]), one);
    const __m128i s = _mm_srli_epi32(_mm_madd_epi16(_mm_packs_epi32(q01, q23), one), 4);
    const __m128i p = _mm_packs_epi32(s, s);
    return (uint32_t)_mm_cvtsi128_si32(_mm_packus_epi16(p, p));
}

// the same for an R10G10B10A2 frame: (R + 2 G + B) of the 10-bit codes, >> 6
static inline uint32_t thumbStep10(const uint8_t* r1, const uint8_t* r2)
{
    const __m128i m10 = _mm_set1_epi32(1023), m10x2 = _mm_set1_epi32(2046), mid = _mm_setr_epi32(0, -1, -1, 0);
    auto luma = [&](__m128i v) {
        return _mm_add_epi32(_mm_add_epi32(_mm_and_si128(v, m10), _mm_and_si128(_mm_srli_epi32(v, 9), m10x2)),
                             _mm_and_si128(_mm_srli_epi32(v, 20), m10));
    };
    __m128i y[4];
    for (int k = 0; k < 4; k++)
        y[k] = _mm_and_si128(_mm_add_epi32(luma(_mm_loadu_si128((const __m128i*)(r1 + 16 * k))),
                                           luma(_mm_loadu_si128((const __m128i*)(r2 + 16 * k)))),
                             mid);
    // each block's four lanes summed: a 4 x 4 transpose, then the rows added
    const __m128i t0 = _mm_unpacklo_epi32(y[0], y[1]), t1 = _mm_unpackhi_epi32(y[0], y[1]);
    const __m128i t2 = _mm_unpacklo_epi32(y[2], y[3]), t3 = _mm_unpackhi_epi32(y[2], y[3]);
    const __m128i s = _mm_add_epi32(_mm_add_epi32(_mm_unpacklo_epi64(t0, t2), _mm_unpackhi_epi64(t0, t2)),
                                    _mm_add_epi32(_mm_unpacklo_epi64(t1, t3), _mm_unpackhi_epi64(t1, t3)));
    const __m128i p = _mm_packs_epi32(_mm_srli_epi32(s, 6), _mm_setzero_si128());
    return (uint32_t)_mm_cvtsi128_si32(_mm_packus_epi16(p, p));
}

// f() on a thread-pool thread beside g() on this one, both in this thread's float mode; f() runs here after g() when no
// pool thread takes it
template <class F, class G> static void beside(F f, G g)
{
    struct Job
    {
        F* f;
        unsigned csr;
        HANDLE done;
    } job{&f, _mm_getcsr(), CreateEventW(nullptr, FALSE, FALSE, nullptr)};
    const bool queued = job.done && TrySubmitThreadpoolCallback(
                                        [](PTP_CALLBACK_INSTANCE, void* ctx) {
                                            Job* j = (Job*)ctx;
                                            const unsigned csr = _mm_getcsr();
                                            _mm_setcsr(j->csr);
                                            (*j->f)();
                                            _mm_setcsr(csr);
                                            SetEvent(j->done);
                                        },
                                        &job, nullptr);
    g();
    if (queued)
        WaitForSingleObject(job.done, INFINITE);
    else
        f();
    if (job.done)
        CloseHandle(job.done);
}

class Estimator
{
  public:
    static constexpr double kMinMove = 1.0; // shorter moves keep zero vectors (frame pixels)

    // the counters the run's summary line reports (readMs: a caller's readback time for the measured frames)
    uint64_t frames = 0, tagged = 0;
    double sumLen = 0, sumStill = 0, sumMs = 0, maxMs = 0, readMs = 0;

    // frame: fw x fh pixels of 4 bytes, rows `pitch` bytes apart; hdr10 = R10G10B10A2, else 8 bits a channel with R
    // and B in either order
    Result run(const uint8_t* frame, uint32_t fw, uint32_t fh, size_t pitch, bool hdr10)
    {
        const int64_t t0 = qpc();
        Result r = measure(frame, fw, fh, pitch, hdr10);
        const double ms = (double)(qpc() - t0) * 1000.0 / (double)qpcFreq();
        frames++;
        sumMs += ms;
        maxMs = ms > maxMs ? ms : maxMs;
        if (r.on)
        {
            tagged++;
            sumLen += std::sqrt((double)r.mx * r.mx + (double)r.my * r.my);
            double area = 0;
            for (const D3D12_RECT& q : r.still)
                area += (double)(q.right - q.left) * (q.bottom - q.top);
            sumStill += area / ((double)fw * fh);
        }
        return r;
    }

    // run() from a ready 1/4 thumbnail (what thumbRows writes for the frame: fw / 4 bytes a row, rows `pitch` apart),
    // e.g. one a GPU made
    Result runThumb(const uint8_t* t4src, size_t pitch, uint32_t fw, uint32_t fh)
    {
        given4 = t4src;
        given4Pitch = pitch;
        const Result r = run(nullptr, fw, fh, 0, false);
        given4 = nullptr;
        return r;
    }

  private:
    const uint8_t* given4 = nullptr; // runThumb's thumbnail, thumbs() copies it instead of making one
    size_t given4Pitch = 0;
    static constexpr int kTile = 8;         // a tile = 8 x 8 thumbnail pixels = 32 x 32 frame pixels
    static constexpr int kRange32 = 12;     // the coarse search, in 1/32 pixels each way (384 frame pixels)
    static constexpr double kExplain = 0.6; // the translation's error must stay under this share of the still one's
    static constexpr double kCover = 0.5;   // the share of changed tiles the translation must explain

    std::vector<uint8_t> t4[2], t16[2], t32[2];
    int cur = 0;
    bool have = false;
    uint32_t w4 = 0, h4 = 0, w16 = 0, h16 = 0, w32 = 0, h32 = 0, fw0 = 0, fh0 = 0;

    static int64_t qpc()
    {
        LARGE_INTEGER c;
        QueryPerformanceCounter(&c);
        return c.QuadPart;
    }
    static int64_t qpcFreq()
    {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        return f.QuadPart;
    }

    // the 1/4 luma thumbnail from four samples of each 4 x 4 block (rows and columns 1 and 2), the 1/16 one as the
    // mean of each 4 x 4 block of that, the 1/32 one as the mean of each 2 x 2 block of the 1/16 one
    void thumbs(const uint8_t* frame, size_t pitch, bool hdr10, uint8_t* o4, uint8_t* o16, uint8_t* o32) const
    {
        // the 1/4 rows in two halves, the lower one on a pool thread (runThumb's own thumbnail copied)
        if (given4)
            for (uint32_t y = 0; y < h4; y++)
                memcpy(o4 + (size_t)y * w4, given4 + (size_t)y * given4Pitch, w4);
        else
            beside([&] { thumbRows(frame, pitch, hdr10, o4, h4 / 2, h4); },
                   [&] { thumbRows(frame, pitch, hdr10, o4, 0, h4 / 2); });
        for (uint32_t y = 0; y < h16; y++)
            for (uint32_t x = 0; x < w16; x++)
            {
                uint32_t s = 0;
                for (uint32_t j = 0; j < 4; j++)
                {
                    const uint8_t* p = o4 + (size_t)(y * 4 + j) * w4 + x * 4;
                    s += (uint32_t)p[0] + p[1] + p[2] + p[3];
                }
                o16[(size_t)y * w16 + x] = (uint8_t)(s >> 4);
            }
        for (uint32_t y = 0; y < h32; y++)
            for (uint32_t x = 0; x < w32; x++)
            {
                const uint8_t* p = o16 + (size_t)(y * 2) * w16 + x * 2;
                o32[(size_t)y * w32 + x] = (uint8_t)(((uint32_t)p[0] + p[1] + p[w16] + p[w16 + 1] + 2) >> 2);
            }
    }

    // rows y0 .. y1 - 1 of the 1/4 thumbnail
    void thumbRows(const uint8_t* frame, size_t pitch, bool hdr10, uint8_t* o4, uint32_t y0, uint32_t y1) const
    {
        for (uint32_t y = y0; y < y1; y++)
        {
            const uint8_t* r1 = frame + (size_t)(y * 4 + 1) * pitch;
            const uint8_t* r2 = r1 + pitch;
            uint8_t* o = o4 + (size_t)y * w4;
            const uint32_t x4 = w4 & ~3u; // four blocks a step, the loop below for the last few
            for (uint32_t x = 0; x < x4; x += 4)
            {
                const size_t b = (size_t)x * 16;
                const uint32_t v = hdr10 ? thumbStep10(r1 + b, r2 + b) : thumbStep(r1 + b, r2 + b);
                memcpy(o + x, &v, 4);
            }
            for (uint32_t x = x4; x < w4; x++)
            {
                const size_t b = (size_t)(x * 4 + 1) * 4;
                uint32_t s = 0;
                if (hdr10)
                {
                    for (const uint8_t* p : {r1 + b, r1 + b + 4, r2 + b, r2 + b + 4})
                    {
                        uint32_t v;
                        memcpy(&v, p, 4);
                        s += (v & 1023u) + 2u * ((v >> 10) & 1023u) + ((v >> 20) & 1023u);
                    }
                    o[x] = (uint8_t)(s >> 6);
                }
                else
                {
                    for (const uint8_t* p : {r1 + b, r1 + b + 4, r2 + b, r2 + b + 4})
                        s += (uint32_t)p[0] + 2u * p[1] + p[2];
                    o[x] = (uint8_t)(s >> 4);
                }
            }
        }
    }

    // mean |c(x, y) - p(x - dx, y - dy)| over the overlap of a w x h thumbnail; -1 when the overlap is under 40 %
    static double meanSad(const uint8_t* c, const uint8_t* p, int w, int h, int dx, int dy)
    {
        const int x0 = dx > 0 ? dx : 0, x1 = dx < 0 ? w + dx : w;
        const int y0 = dy > 0 ? dy : 0, y1 = dy < 0 ? h + dy : h;
        const int n = x1 - x0, m = y1 - y0;
        if (n <= 0 || m <= 0 || (double)n * m < 0.4 * w * h)
            return -1;
        uint64_t s = 0;
        if (hasAvx2())
            for (int y = y0; y < y1; y++)
                s += sadRow32(c + (size_t)y * w + x0, p + (size_t)(y - dy) * w + (x0 - dx), n);
        else
            for (int y = y0; y < y1; y++)
                s += sadRow(c + (size_t)y * w + x0, p + (size_t)(y - dy) * w + (x0 - dx), n);
        return (double)s / ((double)n * m);
    }

    // the sub-pixel offset of a SAD minimum from its two neighbours (the equiangular line fit), in [-0.5, 0.5]
    static double subPixel(double sm, double s0, double sp)
    {
        if (sm < 0 || sp < 0)
            return 0;
        const double d = (sm > sp ? sm : sp) - s0;
        if (d <= 0)
            return 0;
        const double o = (sm - sp) / (2 * d);
        return o < -0.5 ? -0.5 : o > 0.5 ? 0.5 : o;
    }

    Result measure(const uint8_t* frame, uint32_t fw, uint32_t fh, size_t pitch, bool hdr10)
    {
        Result r;
        if (fw != fw0 || fh != fh0)
        {
            fw0 = fw;
            fh0 = fh;
            w4 = fw / 4;
            h4 = fh / 4;
            w16 = w4 / 4;
            h16 = h4 / 4;
            w32 = w16 / 2;
            h32 = h16 / 2;
            for (int i = 0; i < 2; i++)
            {
                t4[i].assign((size_t)w4 * h4, 0);
                t16[i].assign((size_t)w16 * h16, 0);
                t32[i].assign((size_t)w32 * h32, 0);
            }
            have = false;
        }
        if (w32 < 16 || h32 < 8)
            return r;
        cur ^= 1;
        thumbs(frame, pitch, hdr10, t4[cur].data(), t16[cur].data(), t32[cur].data());
        if (!have)
        {
            have = true;
            return r;
        }
        const uint8_t *c4 = t4[cur].data(), *p4 = t4[cur ^ 1].data();
        const uint8_t *c16 = t16[cur].data(), *p16 = t16[cur ^ 1].data();
        const uint8_t *c32 = t32[cur].data(), *p32 = t32[cur ^ 1].data();
        const int W4 = (int)w4, H4 = (int)h4, W16 = (int)w16, H16 = (int)h16, W32 = (int)w32, H32 = (int)h32;

        // the lowest SAD among the offsets (cx +- r, cy +- r) of a level; a tie keeps the shorter move
        auto search = [](const uint8_t* c, const uint8_t* p, int w, int h, int cx, int cy, int rx, int ry, int& bx,
                         int& by) {
            double best = -1;
            for (int dy = cy - ry; dy <= cy + ry; dy++)
                for (int dx = cx - rx; dx <= cx + rx; dx++)
                {
                    const double s = meanSad(c, p, w, h, dx, dy);
                    if (s >= 0 &&
                        (best < 0 || s < best - 1e-9 || (s < best + 1e-9 && dx * dx + dy * dy < bx * bx + by * by)))
                    {
                        best = s;
                        bx = dx;
                        by = dy;
                    }
                }
        };
        // coarse: every offset within the range at 1/32, then +-2 around it at 1/16
        int bx = 0, by = 0;
        search(c32, p32, W32, H32, 0, 0, W32 / 3 < kRange32 ? W32 / 3 : kRange32,
               H32 / 3 < kRange32 ? H32 / 3 : kRange32, bx, by);
        int ax = 2 * bx, ay = 2 * by;
        search(c16, p16, W16, H16, 2 * bx, 2 * by, 2, 2, ax, ay);
        bx = ax;
        by = ay;

        // fine: +-4 around it at 1/4 (offsets 0 .. 40 on a pool thread, 41 .. 80 and the still one here), then the
        // sub-pixel fit
        double tab[9][9];
        auto fill = [&](int k0, int k1) {
            for (int k = k0; k < k1; k++)
                tab[k / 9][k % 9] = meanSad(c4, p4, W4, H4, 4 * bx + k % 9 - 4, 4 * by + k / 9 - 4);
        };
        double s0 = -1;
        beside([&] { fill(0, 41); },
               [&] {
                   fill(41, 81);
                   s0 = meanSad(c4, p4, W4, H4, 0, 0);
               });
        int fx = 4 * bx, fy = 4 * by;
        double fb = -1;
        for (int j = 0; j < 9; j++)
            for (int i = 0; i < 9; i++)
                if (tab[j][i] >= 0 && (fb < 0 || tab[j][i] < fb))
                {
                    fb = tab[j][i];
                    fx = 4 * bx + i - 4;
                    fy = 4 * by + j - 4;
                }
        if (fb < 0 || s0 < 0.5 || fb > kExplain * s0)
            return r;
        const int ti = fx - 4 * bx + 4, tj = fy - 4 * by + 4;
        const double ox = (ti > 0 && ti < 8) ? subPixel(tab[tj][ti - 1], fb, tab[tj][ti + 1]) : 0;
        const double oy = (tj > 0 && tj < 8) ? subPixel(tab[tj - 1][ti], fb, tab[tj + 1][ti]) : 0;
        const double dx = 4.0 * (fx + ox), dy = 4.0 * (fy + oy);
        if (dx * dx + dy * dy < kMinMove * kMinMove)
            return r;

        // tiles: the translation must explain most of the changed ones; the ones a still picture explains clearly
        // better keep zero. A tile row's sums row by row: the SAD against the same place two tiles a SSE2 step (each
        // 64-bit lane of _mm_sad_epu8 = one tile's 8 bytes), against the moved place a tile a step where the tile lies
        // inside the overlap, pixel by pixel at its edges
        static_assert(kTile == 8, "the tile sums take 8 bytes a lane");
        const int tw = W4 / kTile, th = H4 / kTile;
        std::vector<uint8_t> still((size_t)tw * th, 0);
        std::vector<uint64_t> a0s, avs;
        std::vector<int> nvs;
        int changed = 0, agree = 0;
        for (int ty = 0; ty < th; ty++)
        {
            a0s.assign((size_t)tw, 0);
            avs.assign((size_t)tw, 0);
            nvs.assign((size_t)tw, 0);
            for (int y = ty * kTile; y < (ty + 1) * kTile; y++)
            {
                const uint8_t *cr = c4 + (size_t)y * W4, *pr = p4 + (size_t)y * W4;
                int tx = 0;
                for (; tx + 2 <= tw; tx += 2)
                {
                    const __m128i s = _mm_sad_epu8(_mm_loadu_si128((const __m128i*)(cr + tx * kTile)),
                                                   _mm_loadu_si128((const __m128i*)(pr + tx * kTile)));
                    a0s[tx] += (uint64_t)_mm_cvtsi128_si64(s);
                    a0s[tx + 1] += (uint64_t)_mm_cvtsi128_si64(_mm_unpackhi_epi64(s, s));
                }
                if (tx < tw)
                    a0s[tx] +=
                        (uint64_t)_mm_cvtsi128_si64(_mm_sad_epu8(_mm_loadl_epi64((const __m128i*)(cr + tx * kTile)),
                                                                 _mm_loadl_epi64((const __m128i*)(pr + tx * kTile))));
                const int sy = y - fy;
                if (sy < 0 || sy >= H4)
                    continue;
                const uint8_t* ps = p4 + (size_t)sy * W4;
                for (tx = 0; tx < tw; tx++)
                {
                    const int x0 = tx * kTile, sx0 = x0 - fx;
                    if (sx0 >= 0 && sx0 + kTile <= W4)
                    {
                        avs[tx] += (uint64_t)_mm_cvtsi128_si64(_mm_sad_epu8(
                            _mm_loadl_epi64((const __m128i*)(cr + x0)), _mm_loadl_epi64((const __m128i*)(ps + sx0))));
                        nvs[tx] += kTile;
                        continue;
                    }
                    for (int x = x0; x < x0 + kTile; x++)
                    {
                        const int sx = x - fx;
                        if (sx < 0 || sx >= W4)
                            continue;
                        const uint8_t a = cr[x], b = ps[sx];
                        avs[tx] += (uint64_t)(a > b ? a - b : b - a);
                        nvs[tx]++;
                    }
                }
            }
            for (int tx = 0; tx < tw; tx++)
            {
                if (nvs[tx] < kTile * kTile / 2)
                    continue;
                const double m0 = (double)a0s[tx] / (kTile * kTile), mv = (double)avs[tx] / nvs[tx];
                if (m0 >= 2.0)
                {
                    changed++;
                    if (mv < 0.7 * m0 && m0 - mv >= 1.0)
                        agree++;
                }
                if (mv > m0 + 1.0 && mv > 1.5 * m0)
                    still[(size_t)ty * tw + tx] = 1;
            }
        }
        if (!changed || agree < kCover * changed)
            return r;

        // the still tiles as runs along each tile row, a run on the row below with the same ends joining it, in frame
        // pixels
        size_t open = 0; // r.still[open..] = the rectangles that reach the current row
        for (int ty = 0; ty < th; ty++)
        {
            const size_t rowStart = r.still.size();
            for (int tx = 0; tx < tw;)
            {
                if (!still[(size_t)ty * tw + tx])
                {
                    tx++;
                    continue;
                }
                int e = tx;
                while (e < tw && still[(size_t)ty * tw + e])
                    e++;
                const LONG l = tx * kTile * 4, rt = e * kTile * 4, top = ty * kTile * 4;
                bool joined = false;
                for (size_t i = open; i < rowStart && !joined; i++)
                    if (r.still[i].left == l && r.still[i].right == rt && r.still[i].bottom == top)
                    {
                        r.still[i].bottom = top + kTile * 4;
                        joined = true;
                    }
                if (!joined)
                    r.still.push_back({l, top, rt, top + kTile * 4});
                tx = e;
            }
            // the rectangles that did not reach this row are closed
            const LONG bottom = (ty + 1) * kTile * 4;
            std::vector<D3D12_RECT> closed, reach;
            for (size_t i = open; i < r.still.size(); i++)
                (r.still[i].bottom == bottom ? reach : closed).push_back(r.still[i]);
            r.still.resize(open);
            r.still.insert(r.still.end(), closed.begin(), closed.end());
            open = r.still.size();
            r.still.insert(r.still.end(), reach.begin(), reach.end());
        }
        r.on = true;
        r.mx = (float)-dx;
        r.my = (float)-dy;
        return r;
    }
};

// DLSS-G's depth and motion vector textures are the frame / kVecDiv per axis: the generated frames are byte for byte
// the full-size textures' at a quarter of the bytes (the still rects sit on 32 px tiles, so they scale exactly)
constexpr uint32_t kVecDiv = 2;
inline uint32_t vecSize(uint32_t n)
{
    return (n + kVecDiv - 1) / kVecDiv;
}

// the vector texture's contents, written by UAV clears on the present's command list (the texture needs
// D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS and rests in the tagged shader-resource state); w x h = the frame, the
// texture = vecSize(w) x vecSize(h), a still rect scaled outward to it
struct Writer
{
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> gpuHeap, cpuHeap;
    Result held; // what the texture holds (it starts at zero)
    float fw = 1, fh = 1;

    bool init(ID3D12Device* dev, ID3D12Resource* tex, uint32_t w, uint32_t h)
    {
        fw = (float)w;
        fh = (float)h;
        D3D12_DESCRIPTOR_HEAP_DESC hd{};
        hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        hd.NumDescriptors = 1;
        hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        if (FAILED(dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&gpuHeap))))
            return false;
        hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        if (FAILED(dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&cpuHeap))))
            return false;
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud{};
        ud.Format = DXGI_FORMAT_R16G16_FLOAT;
        ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        dev->CreateUnorderedAccessView(tex, nullptr, &ud, gpuHeap->GetCPUDescriptorHandleForHeapStart());
        dev->CreateUnorderedAccessView(tex, nullptr, &ud, cpuHeap->GetCPUDescriptorHandleForHeapStart());
        return true;
    }

    // records the clears that make `tex` hold r; nothing when it already does
    void record(ID3D12GraphicsCommandList* list, ID3D12Resource* tex, const Result& r)
    {
        if (!gpuHeap || held.same(r))
            return;
        const D3D12_RESOURCE_STATES srv =
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = tex;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b.Transition.StateBefore = srv;
        b.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        list->ResourceBarrier(1, &b);
        ID3D12DescriptorHeap* heaps[] = {gpuHeap.Get()};
        list->SetDescriptorHeaps(1, heaps);
        const D3D12_GPU_DESCRIPTOR_HANDLE g = gpuHeap->GetGPUDescriptorHandleForHeapStart();
        const D3D12_CPU_DESCRIPTOR_HANDLE c = cpuHeap->GetCPUDescriptorHandleForHeapStart();
        const float zero[4] = {0, 0, 0, 0};
        if (r.on)
        {
            const float v[4] = {r.mx / fw, r.my / fh, 0, 0};
            list->ClearUnorderedAccessViewFloat(g, c, tex, v, 0, nullptr);
            if (!r.still.empty())
            {
                D3D12_RESOURCE_BARRIER u{};
                u.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                u.UAV.pResource = tex;
                list->ResourceBarrier(1, &u);
                // one rectangle a call: a long list in one call stops NVIDIA's driver (0xC0000409 in nvwgf2umx.dll)
                constexpr LONG d = (LONG)kVecDiv;
                for (const D3D12_RECT& q : r.still)
                {
                    const D3D12_RECT s{q.left / d, q.top / d, (q.right + d - 1) / d, (q.bottom + d - 1) / d};
                    list->ClearUnorderedAccessViewFloat(g, c, tex, zero, 1, &s);
                }
            }
        }
        else
            list->ClearUnorderedAccessViewFloat(g, c, tex, zero, 0, nullptr);
        std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
        list->ResourceBarrier(1, &b);
        held = r;
    }
};

} // namespace panvec
