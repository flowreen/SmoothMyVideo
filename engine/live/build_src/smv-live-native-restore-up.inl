// part of smv-live-native.inl (included there, in place); not a standalone translation unit
// ---- Restore's nets as the upscaler (`--realesrgan`, `--animejanai-upscale FILE`) in RTX VSR's resize, both routes ----
// The upscaler runs after the model on every output frame (Restore runs before it, once a real frame, at the source
// size): the bundled Real-ESRGAN animevideov3 (x4) or AnimeJaNai HD (x2, the Restore panel's download; rgAj) on the
// model-size frame, folded to the target by Restore's Lanczos pair. Its own weights block and buffers, so a session can
// run Restore and this upscaler at once; never the shared activation block, which the model may still hold between a
// pair's tweens. A ratio past the net's scale runs equal passes of at most that scale each (the net, then the fold
// down to that pass's size), so every enlarge is the net's and Lanczos only shrinks. When the video memory room does not
// hold the net's activations for whole frames, the net runs tile by tile, each tile's core read with the net's halo
// around it, so the frames are the same bytes.

// the smallest tile core the fit tries; below it the halo's extra work outweighs the net's own
constexpr int kRgMinCore = 128;

// the net's scale: Real-ESRGAN 4, AnimeJaNai 2
static int rgScale(const NativeRife& nr)
{
    return nr.rgAj ? 2 : 4;
}

// the net's halo in input pixels: every 3 x 3 conv reads one pixel further out (Real-ESRGAN: 18 convs; AnimeJaNai: the
// first conv, three a block and the last, on the half grid = two input pixels each; its 1 x 1 convs read none), so a
// tile's core read with this many real pixels around it holds the whole frame's bits
static int rgHalo(const NativeRife& nr)
{
    return nr.rgAj ? 2 * (2 + 3 * nr.rgBlocks) : 18;
}

// a tile's side along an axis of full pixels for a core of c (0 = whole frames): the core and the halo on both sides,
// at most the frame
static int rgTileOf(int c, int hal, int full)
{
    return c ? (std::min)(c + 2 * hal, full) : full;
}

// the passes for w x h -> tw x th with a net of scale s: the fewest n with s^n >= the larger axis ratio (at most
// kRgMaxPasses), pass k's output = the input x ratio^((k + 1) / n) per axis, the last one the target; pw / ph[0] = the
// input, [n] = the target
static int rgUpPlan(int w, int h, int tw, int th, int s, int* pw, int* ph)
{
    const double rx = (double)tw / w, ry = (double)th / h, r = (std::max)(rx, ry);
    int n = 1;
    for (double c = s; c < r && n < kRgMaxPasses; c *= s)
        n++;
    pw[0] = w;
    ph[0] = h;
    for (int k = 1; k < n; k++)
    {
        pw[k] = (int)std::lround(w * std::pow(rx, (double)k / n));
        ph[k] = (int)std::lround(h * std::pow(ry, (double)k / n));
    }
    pw[n] = tw;
    ph[n] = th;
    return n;
}

// the buffers a plan needs with tile cores of core (0 = whole frames), each its largest pass's: the input planes
// (pixels), the net's activations (bytes, a tile's: Real-ESRGAN two 64-half frames at the input size, AnimeJaNai eight
// at the half grid), its 48 sub-pixel planes (halves: at the input size, AnimeJaNai's at the half grid), the fold's
// horizontal pass and an intermediate output (floats; no intermediate for one pass), and for a pass larger than one
// core a tile's input planes (pixels) and sub-pixel planes (halves)
struct RgNeeds
{
    size_t in = 0, act = 0, out = 0, tmp = 0, mid = 0, tin = 0, tout = 0;
    size_t bytes(size_t weights) const
    {
        return weights + (in + tin) * 3 * sizeof(uint16_t) + act + (out + tout) * sizeof(uint16_t) +
               (tmp + mid) * sizeof(float);
    }
};
static RgNeeds rgUpNeeds(bool aj, int n, const int* pw, const int* ph, int core, int hal)
{
    RgNeeds q;
    const size_t s = aj ? 2 : 4;
    auto halfOf = [](size_t w, size_t h) { return ((w + 1) / 2) * ((h + 1) / 2); };
    for (int k = 0; k < n; k++)
    {
        const size_t px = (size_t)pw[k] * ph[k];
        const size_t tw = rgTileOf(core, hal, pw[k]), th = rgTileOf(core, hal, ph[k]);
        q.in = (std::max)(q.in, px);
        q.act = (std::max)(q.act, aj ? 8 * halfOf(tw, th) * 128 : 2 * tw * th * 128);
        q.out = (std::max)(q.out, 48 * (aj ? halfOf(pw[k], ph[k]) : px));
        if (core && (pw[k] > core || ph[k] > core))
        {
            q.tin = (std::max)(q.tin, tw * th);
            q.tout = (std::max)(q.tout, 48 * (aj ? halfOf(tw, th) : tw * th));
        }
        q.tmp = (std::max)(q.tmp, 3 * s * ph[k] * pw[k + 1]);
        if (k + 1 < n)
            q.mid = (std::max)(q.mid, (size_t)3 * pw[k + 1] * ph[k + 1]);
    }
    return q;
}

// the frame buffers (nativeRgUpSetup's), freed; the weights stay
static void nativeRgUpBufFree(NativeRife& nr)
{
    for (void** p :
         {&nr.dRgIn, &nr.dRgAct, &nr.dRgOut, (void**)&nr.dRgMid, (void**)&nr.dRgTmp, &nr.dRgTin, &nr.dRgTout})
        if (*p)
        {
            cudaFree(*p);
            *p = nullptr;
        }
    nr.rgInN = nr.rgActB = nr.rgOutN = nr.rgTmpN = nr.rgMidN = nr.rgTinN = nr.rgToutN = 0;
    nr.rgW = nr.rgH = nr.rgCore = 0;
}

// every buffer of the upscaler, its weights too
static void nativeRgUpRelease(NativeRife& nr)
{
    if (nr.stream && (nr.dRgAct || nr.dRgW))
        cudaStreamSynchronize(nr.stream);
    nativeRgUpBufFree(nr);
    if (nr.dRgW)
    {
        cudaFree(nr.dRgW);
        nr.dRgW = nullptr;
    }
    nr.rgOff.clear();
    nr.rgBlocks = 0;
}

// the net on st at w x h: in (three fp16 planes) -> out (the sub-pixel planes); false = a launch failed
static bool rgNet(NativeRife& nr, cudaStream_t st, void* in, void* out, int w, int h)
{
    return nr.rgAj
               ? ajConvChain(nr, st, in, nr.dRgAct, nr.dRgW, nr.rgOff, nr.rgBlocks, out, w, h)
               : restConvChain(nr, st, in, nr.dRgAct, (char*)nr.dRgAct + nr.rgActB / 2, nr.dRgW, nr.rgOff, out, w, h);
}

// copies a bw x bh box of `planes` fp16 planes at (sx, sy) of src's sw x sh planes to (dx, dy) of dst's dw x dh planes,
// on st
static bool rgCopyBox(cudaStream_t st, void* src, int sw, int sh, int sx, int sy, void* dst, int dw, int dh, int dx,
                      int dy, int bw, int bh, int planes)
{
    // linear memory: x in bytes, a plane = ysize rows of pitch bytes
    cudaMemcpy3DParms p = {};
    p.srcPtr = {src, (size_t)sw * sizeof(uint16_t), (size_t)sw, (size_t)sh};
    p.srcPos = {(size_t)sx * sizeof(uint16_t), (size_t)sy, 0};
    p.dstPtr = {dst, (size_t)dw * sizeof(uint16_t), (size_t)dw, (size_t)dh};
    p.dstPos = {(size_t)dx * sizeof(uint16_t), (size_t)dy, 0};
    p.extent = {(size_t)bw * sizeof(uint16_t), (size_t)bh, (size_t)planes};
    p.kind = cudaMemcpyDeviceToDevice;
    return cudaMemcpy3DAsync(&p, st) == cudaSuccess;
}

// the net on st over a w x h frame, dRgIn -> dRgOut: at once, or tile by tile when the buffers hold tiles (rgCore, a
// pass larger than one core): a tile = a core and the halo around it (at the frame's edges the frame's own padding),
// its input planes copied out of dRgIn into dRgTin, the net into dRgTout, the core of its sub-pixel planes copied into
// dRgOut. AnimeJaNai's cores, halo and tile corners are even, so a tile's half grid is the frame's; false = a launch or
// copy failed
static bool rgNetFrame(NativeRife& nr, cudaStream_t st, int w, int h)
{
    const int core = nr.rgCore;
    if (!core || (w <= core && h <= core))
        return rgNet(nr, st, nr.dRgIn, nr.dRgOut, w, h);
    const int hal = rgHalo(nr), g = nr.rgAj ? 1 : 0; // g: the sub-pixel planes' grid = the input's >> g
    const int ow = (w + g) >> g, oh = (h + g) >> g;
    for (int y0 = 0; y0 < h; y0 += core)
        for (int x0 = 0; x0 < w; x0 += core)
        {
            const int x1 = (std::min)(x0 + core, w), y1 = (std::min)(y0 + core, h);
            const int xa = (std::max)(x0 - hal, 0), ya = (std::max)(y0 - hal, 0);
            const int tw = (std::min)(x1 + hal, w) - xa, th = (std::min)(y1 + hal, h) - ya;
            const int gtw = (tw + g) >> g, gth = (th + g) >> g, gx0 = x0 >> g, gy0 = y0 >> g;
            if (!rgCopyBox(st, nr.dRgIn, w, h, xa, ya, nr.dRgTin, tw, th, 0, 0, tw, th, 3) ||
                !rgNet(nr, st, nr.dRgTin, nr.dRgTout, tw, th) ||
                !rgCopyBox(st, nr.dRgTout, gtw, gth, gx0 - (xa >> g), gy0 - (ya >> g), nr.dRgOut, ow, oh, gx0, gy0,
                           ((x1 + g) >> g) - gx0, ((y1 + g) >> g) - gy0, 48))
                return false;
        }
    return true;
}

// the upscaler at the slot's input size w x h (-> ow x oh): the weights (Real-ESRGAN's file Restore reads, or
// AnimeJaNai's file of --animejanai-upscale), the plan's buffers for whole frames when the video memory room holds
// them, else for the largest tile core that fits (halved from half the plan's largest side down to kRgMinCore; even,
// for AnimeJaNai's half grid), one run of the net on zeros, so the first frame pays no first launch. No weights or no
// room even for the smallest tiles = Lanczos for the render / session (vsrFailed), with its line.
static bool nativeRgUpSetup(NativeRife& nr, int w, int h, int ow, int oh, int upEnc)
{
    const auto t0 = std::chrono::steady_clock::now();
    const char* where = g_offline ? "render" : "session";
    const char* name = nr.rgAj ? "AnimeJaNai HD" : "Real-ESRGAN";
    std::vector<uint8_t> blob;
    std::string why;
    if (nr.rgAj && g_ajUpModel.empty())
        why = "no AnimeJaNai file named";
    if (!why.empty() ||
        !(nr.rgAj ? ajNatLoad(g_ajUpModel, blob, nr.rgOff, nr.rgBlocks, why) : restNatLoad(blob, nr.rgOff, why)))
    {
        LOG("native: %s upscaling unavailable (%s), Lanczos for the %s\n", name, why.c_str(), where);
        nr.vsrFailed = true;
        return true;
    }
    int pw[kRgMaxPasses + 1], ph[kRgMaxPasses + 1];
    const int n = rgUpPlan(w, h, ow, oh, rgScale(nr), pw, ph);
    const int hal = rgHalo(nr);
    RgNeeds q = rgUpNeeds(nr.rgAj, n, pw, ph, 0, hal);
    size_t needB = q.bytes(blob.size());
    const uint64_t room = nativeVideoMemoryRoom();
    int core = 0;
    if (room && room < needB)
    {
        int big = 0;
        for (int k = 0; k < n; k++)
            big = (std::max)(big, (std::max)(pw[k], ph[k]));
        for (int c = ((big + 1) / 2 + 1) & ~1; c >= kRgMinCore; c = (c / 2 + 1) & ~1)
        {
            q = rgUpNeeds(nr.rgAj, n, pw, ph, c, hal);
            needB = q.bytes(blob.size());
            if (room >= needB)
            {
                core = c;
                break;
            }
        }
        if (!core)
        {
            LOG("native: %s upscaling needs %.0f MB of video memory (%.0f MB free), Lanczos for the %s\n", name,
                needB / 1048576.0, room / 1048576.0, where);
            nr.vsrFailed = true;
            return true;
        }
    }
    bool ok = (nr.dRgW || cudaMalloc(&nr.dRgW, blob.size()) == cudaSuccess) &&
              cudaMemcpy(nr.dRgW, blob.data(), blob.size(), cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMalloc(&nr.dRgIn, q.in * 3 * sizeof(uint16_t)) == cudaSuccess &&
              cudaMalloc(&nr.dRgAct, q.act) == cudaSuccess &&
              cudaMalloc(&nr.dRgOut, q.out * sizeof(uint16_t)) == cudaSuccess &&
              cudaMalloc((void**)&nr.dRgTmp, q.tmp * sizeof(float)) == cudaSuccess &&
              (!q.mid || cudaMalloc((void**)&nr.dRgMid, q.mid * sizeof(float)) == cudaSuccess) &&
              (!q.tin || cudaMalloc(&nr.dRgTin, q.tin * 3 * sizeof(uint16_t)) == cudaSuccess) &&
              (!q.tout || cudaMalloc(&nr.dRgTout, q.tout * sizeof(uint16_t)) == cudaSuccess);
    if (!ok)
    {
        cudaGetLastError();
        nativeRgUpRelease(nr);
        LOG("native: %s upscaling ran out of video memory, Lanczos for the %s\n", name, where);
        nr.vsrFailed = true;
        return true;
    }
    nr.rgInN = q.in;
    nr.rgActB = q.act;
    nr.rgOutN = q.out;
    nr.rgTmpN = q.tmp;
    nr.rgMidN = q.mid;
    nr.rgTinN = q.tin;
    nr.rgToutN = q.tout;
    nr.rgCore = core;
    nr.rgW = w;
    nr.rgH = h;
    NCHK(cudaMemsetAsync(nr.dRgIn, 0, (size_t)w * h * 3 * sizeof(uint16_t), nr.stream), "upscale net warm-up clear");
    if (!rgNetFrame(nr, nr.stream, w, h))
    {
        LOG("native: %s upscale warm-up launch failed\n", name);
        return false;
    }
    NCHK(cudaStreamSynchronize(nr.stream), "upscale net warm-up sync");
    char passes[64] = "", tiles[48] = "";
    if (n > 1)
        sprintf_s(passes, " in %d passes (%dx%d first)", n, pw[1], ph[1]);
    if (core)
        sprintf_s(tiles, ", %d-px tiles", core);
    LOG("native: %s upscale: %s %dx%d -> Lanczos %dx%d%s%s%s (the host's kernels, fp16 accumulate, %.0f MB, %.1fs)\n",
        g_offline ? "offline" : "live", nr.rgAj ? "AnimeJaNai HD x2" : "Real-ESRGAN animevideov3 x4", w, h, ow, oh,
        passes, tiles, nr.hdr ? ", on the HDR planes' SDR view" : (upEnc ? ", on the HDR video's SDR view" : ""),
        needB / 1048576.0, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return true;
}

// one frame through the upscaler: s (planar, fp32 or fp16 by sHalf, plane / row strides ps_ / rs_) at the slot's input
// size, pass by pass: fp16 planes (k_restIn), the net, the fold to the pass's size (Real-ESRGAN adds the nearest base,
// AnimeJaNai none; the last pass into dst, planar fp32, tight, clipped to 0..1; an earlier one into dRgMid, the next
// pass's input)
static bool nativeRgUpRun(NativeRife& nr, const void* s, int sHalf, int ps_, int rs_, float* dst, int tw, int th,
                          cudaStream_t stIn)
{
    cudaStream_t st = stIn ? stIn : nr.stream;
    const int sc = rgScale(nr);
    int pw[kRgMaxPasses + 1], ph[kRgMaxPasses + 1];
    const int n = rgUpPlan(nr.rgW, nr.rgH, tw, th, sc, pw, ph);
    const RgNeeds q = rgUpNeeds(nr.rgAj, n, pw, ph, nr.rgCore, rgHalo(nr));
    if (q.in > nr.rgInN || q.act > nr.rgActB || q.out > nr.rgOutN || q.tmp > nr.rgTmpN || q.mid > nr.rgMidN ||
        q.tin > nr.rgTinN || q.tout > nr.rgToutN)
    {
        nr.die("upscale net target larger than its buffers");
        return false;
    }
    const void* src = s;
    int half = sHalf, ps = ps_, rs = rs_;
    for (int k = 0; k < n; k++)
    {
        int w = pw[k], h = ph[k];
        const int ow = pw[k + 1], oh = ph[k + 1];
        void* a[] = {(void*)&src, &half, &ps, &rs, &w, &h, &nr.dRgIn};
        if (cuLaunchKernel(nr.fRestIn, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a, nullptr) !=
                CUDA_SUCCESS ||
            !rgNetFrame(nr, st, w, h))
        {
            nr.die("upscale net launch failed");
            return false;
        }
        float* o = k + 1 == n ? dst : nr.dRgMid;
        const int subW = nr.rgAj ? (w + 1) / 2 : w, subH = nr.rgAj ? (h + 1) / 2 : h;
        if (!restFold(nr, st, nr.dRgOut, 1, true, nr.rgAj ? nullptr : nr.dRgIn, 1, 4, subW, subH, sc * w, sc * h,
                      nr.dRgTmp, o, ow, oh))
            return false;
        src = nr.dRgMid;
        half = 0;
        ps = ow * oh;
        rs = ow;
    }
    return true;
}
