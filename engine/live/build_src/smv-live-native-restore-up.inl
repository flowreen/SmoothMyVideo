// part of smv-live-native.inl (included there, in place); not a standalone translation unit
// ---- Restore's nets as the upscaler (`--realesrgan`, `--animejanai-upscale FILE`) in RTX VSR's resize, both routes ----
// The upscaler runs after the model on every output frame (Restore runs before it, once a real frame, at the source
// size): the bundled Real-ESRGAN animevideov3 (x4) or AnimeJaNai HD (x2, the Restore panel's download; rgAj) on the
// model-size frame, folded to the target by Restore's Lanczos pair. Its own weights block and buffers, so a session can
// run Restore and this upscaler at once; never the shared activation block, which the model may still hold between a
// pair's tweens. A ratio past the net's scale runs equal passes of at most that scale each (the net, then the fold
// down to that pass's size), so every enlarge is the net's and Lanczos only shrinks.

// the net's scale: Real-ESRGAN 4, AnimeJaNai 2
static int rgScale(const NativeRife& nr)
{
    return nr.rgAj ? 2 : 4;
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

// the buffers a plan needs, each its largest pass's: the input planes (pixels), the net's activations (bytes:
// Real-ESRGAN two 64-half frames at the input size, AnimeJaNai eight at the half grid), its 48 sub-pixel planes
// (halves: at the input size, AnimeJaNai's at the half grid), the fold's horizontal pass and an intermediate output
// (floats; no intermediate for one pass)
struct RgNeeds
{
    size_t in = 0, act = 0, out = 0, tmp = 0, mid = 0;
};
static RgNeeds rgUpNeeds(bool aj, int n, const int* pw, const int* ph)
{
    RgNeeds q;
    const size_t s = aj ? 2 : 4;
    for (int k = 0; k < n; k++)
    {
        const size_t px = (size_t)pw[k] * ph[k], half = (size_t)((pw[k] + 1) / 2) * ((ph[k] + 1) / 2);
        q.in = (std::max)(q.in, px);
        q.act = (std::max)(q.act, aj ? 8 * half * 128 : 2 * px * 128);
        q.out = (std::max)(q.out, 48 * (aj ? half : px));
        q.tmp = (std::max)(q.tmp, 3 * s * ph[k] * pw[k + 1]);
        if (k + 1 < n)
            q.mid = (std::max)(q.mid, (size_t)3 * pw[k + 1] * ph[k + 1]);
    }
    return q;
}

// the frame buffers (nativeRgUpSetup's), freed; the weights stay
static void nativeRgUpBufFree(NativeRife& nr)
{
    for (void** p : {&nr.dRgIn, &nr.dRgAct, &nr.dRgOut, (void**)&nr.dRgMid, (void**)&nr.dRgTmp})
        if (*p)
        {
            cudaFree(*p);
            *p = nullptr;
        }
    nr.rgInN = nr.rgActB = nr.rgOutN = nr.rgTmpN = nr.rgMidN = 0;
    nr.rgW = nr.rgH = 0;
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

// the net on st at w x h: dRgIn -> dRgOut (the sub-pixel planes); false = a launch failed
static bool rgNet(NativeRife& nr, cudaStream_t st, int w, int h)
{
    return nr.rgAj ? ajConvChain(nr, st, nr.dRgIn, nr.dRgAct, nr.dRgW, nr.rgOff, nr.rgBlocks, nr.dRgOut, w, h)
                   : restConvChain(nr, st, nr.dRgIn, nr.dRgAct, (char*)nr.dRgAct + nr.rgActB / 2, nr.dRgW, nr.rgOff,
                                   nr.dRgOut, w, h);
}

// the upscaler at the slot's input size w x h (-> ow x oh): the weights (Real-ESRGAN's file Restore reads, or
// AnimeJaNai's file of --animejanai-upscale), the plan's buffers when the video memory room holds them, one run of the
// net on zeros, so the first frame pays no first launch. No weights or no room = Lanczos for the render / session
// (vsrFailed), with its line.
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
    const RgNeeds q = rgUpNeeds(nr.rgAj, n, pw, ph);
    const size_t needB = blob.size() + q.in * 3 * 2 + q.act + q.out * 2 + (q.tmp + q.mid) * 4;
    const uint64_t room = nativeVideoMemoryRoom();
    if (room && room < needB)
    {
        LOG("native: %s upscaling needs %.0f MB of video memory (%.0f MB free), Lanczos for the %s\n", name,
            needB / 1048576.0, room / 1048576.0, where);
        nr.vsrFailed = true;
        return true;
    }
    bool ok = (nr.dRgW || cudaMalloc(&nr.dRgW, blob.size()) == cudaSuccess) &&
              cudaMemcpy(nr.dRgW, blob.data(), blob.size(), cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMalloc(&nr.dRgIn, q.in * 3 * sizeof(uint16_t)) == cudaSuccess &&
              cudaMalloc(&nr.dRgAct, q.act) == cudaSuccess &&
              cudaMalloc(&nr.dRgOut, q.out * sizeof(uint16_t)) == cudaSuccess &&
              cudaMalloc((void**)&nr.dRgTmp, q.tmp * sizeof(float)) == cudaSuccess &&
              (!q.mid || cudaMalloc((void**)&nr.dRgMid, q.mid * sizeof(float)) == cudaSuccess);
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
    nr.rgW = w;
    nr.rgH = h;
    NCHK(cudaMemsetAsync(nr.dRgIn, 0, (size_t)w * h * 3 * sizeof(uint16_t), nr.stream), "upscale net warm-up clear");
    if (!rgNet(nr, nr.stream, w, h))
    {
        LOG("native: %s upscale warm-up launch failed\n", name);
        return false;
    }
    NCHK(cudaStreamSynchronize(nr.stream), "upscale net warm-up sync");
    char passes[64] = "";
    if (n > 1)
        sprintf_s(passes, " in %d passes (%dx%d first)", n, pw[1], ph[1]);
    LOG("native: %s upscale: %s %dx%d -> Lanczos %dx%d%s%s (the host's kernels, fp16 accumulate, %.0f MB, %.1fs)\n",
        g_offline ? "offline" : "live", nr.rgAj ? "AnimeJaNai HD x2" : "Real-ESRGAN animevideov3 x4", w, h, ow, oh,
        passes, nr.hdr ? ", on the HDR planes' SDR view" : (upEnc ? ", on the HDR video's SDR view" : ""),
        needB / 1048576.0, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return true;
}

// one frame through the upscaler: s (planar, fp32 or fp16 by sHalf, plane / row strides ps_ / rs_) at the slot's input
// size, pass by pass: fp16 planes (k_restIn), the net, the fold to the pass's size (Real-ESRGAN adds the nearest base,
// AnimeJaNai none; the last pass into dst, planar fp32, tight, clipped to 0..1; an earlier one into dRgMid, the next
// pass's input)
static bool nativeRgUpRun(NativeRife& nr, const void* s, int sHalf, int ps_, int rs_, float* dst, int tw, int th)
{
    cudaStream_t st = nr.stream;
    const int sc = rgScale(nr);
    int pw[kRgMaxPasses + 1], ph[kRgMaxPasses + 1];
    const int n = rgUpPlan(nr.rgW, nr.rgH, tw, th, sc, pw, ph);
    const RgNeeds q = rgUpNeeds(nr.rgAj, n, pw, ph);
    if (q.in > nr.rgInN || q.act > nr.rgActB || q.out > nr.rgOutN || q.tmp > nr.rgTmpN || q.mid > nr.rgMidN)
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
            !rgNet(nr, st, w, h))
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
