// part of smv-live-native.inl (included there, in place); not a standalone translation unit
// ---- Real-ESRGAN as the upscaler (`--realesrgan`): Restore's bundled net in RTX VSR's resize, both routes ----
// The upscaler runs after the model on every output frame (Restore runs before it, once a real frame, at the source
// size): the bundled animevideov3 (x4) on the model-size frame, folded to the target by Restore's Lanczos pair. Its own
// weights block and buffers, so a session can run Restore and this upscaler at once; never the shared activation
// block, which the model may still hold between a pair's tweens. A ratio past 4x runs equal passes of at most 4x each
// (the net's x4, then the fold down to that pass's size), so every enlarge is the net's and Lanczos only shrinks.

// the passes for w x h -> tw x th: the fewest n with 4^n >= the larger axis ratio (at most kRgMaxPasses), pass k's
// output = the input x ratio^((k + 1) / n) per axis, the last one the target; pw / ph[0] = the input, [n] = the target
static int rgUpPlan(int w, int h, int tw, int th, int* pw, int* ph)
{
    const double rx = (double)tw / w, ry = (double)th / h, r = (std::max)(rx, ry);
    int n = 1;
    for (double c = 4.0; c < r && n < kRgMaxPasses; c *= 4.0)
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

// the buffers a plan needs: the largest pass input (pixels: the net's input planes, its ping-pong, its sub-pixel
// planes), the fold's widest horizontal pass (floats), the largest intermediate output (floats; 0 for one pass)
static void rgUpNeeds(int n, const int* pw, const int* ph, size_t& inN, size_t& tmpN, size_t& midN)
{
    inN = tmpN = midN = 0;
    for (int k = 0; k < n; k++)
    {
        inN = (std::max)(inN, (size_t)pw[k] * ph[k]);
        tmpN = (std::max)(tmpN, (size_t)3 * 4 * ph[k] * pw[k + 1]);
        if (k + 1 < n)
            midN = (std::max)(midN, (size_t)3 * pw[k + 1] * ph[k + 1]);
    }
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
    nr.rgInN = nr.rgTmpN = nr.rgMidN = 0;
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
}

// Real-ESRGAN at the slot's input size w x h (-> ow x oh): the weights (restNatLoad, the file Restore reads), the plan's
// buffers when the video memory room holds them, one run of the net on zeros, so the first frame pays no first launch.
// No weights or no room = Lanczos for the render / session (vsrFailed), with its line.
static bool nativeRgUpSetup(NativeRife& nr, int w, int h, int ow, int oh, int upEnc)
{
    const auto t0 = std::chrono::steady_clock::now();
    const char* where = g_offline ? "render" : "session";
    std::vector<uint8_t> blob;
    std::string why;
    if (!restNatLoad(blob, nr.rgOff, why))
    {
        LOG("native: Real-ESRGAN upscaling unavailable (%s), Lanczos for the %s\n", why.c_str(), where);
        nr.vsrFailed = true;
        return true;
    }
    int pw[kRgMaxPasses + 1], ph[kRgMaxPasses + 1];
    const int n = rgUpPlan(w, h, ow, oh, pw, ph);
    size_t inN, tmpN, midN;
    rgUpNeeds(n, pw, ph, inN, tmpN, midN);
    // the net's input (3 fp16 planes), its ping-pong (two 64-half frames), its 48 fp16 sub-pixel planes
    const size_t actB = 2 * inN * 128, needB = blob.size() + inN * 3 * 2 + actB + inN * 48 * 2 + (tmpN + midN) * 4;
    const uint64_t room = nativeVideoMemoryRoom();
    if (room && room < needB)
    {
        LOG("native: Real-ESRGAN upscaling needs %.0f MB of video memory (%.0f MB free), Lanczos for the %s\n",
            needB / 1048576.0, room / 1048576.0, where);
        nr.vsrFailed = true;
        return true;
    }
    bool ok = (nr.dRgW || cudaMalloc(&nr.dRgW, blob.size()) == cudaSuccess) &&
              cudaMemcpy(nr.dRgW, blob.data(), blob.size(), cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMalloc(&nr.dRgIn, inN * 3 * sizeof(uint16_t)) == cudaSuccess &&
              cudaMalloc(&nr.dRgAct, actB) == cudaSuccess &&
              cudaMalloc(&nr.dRgOut, inN * 48 * sizeof(uint16_t)) == cudaSuccess &&
              cudaMalloc((void**)&nr.dRgTmp, tmpN * sizeof(float)) == cudaSuccess &&
              (!midN || cudaMalloc((void**)&nr.dRgMid, midN * sizeof(float)) == cudaSuccess);
    if (!ok)
    {
        cudaGetLastError();
        nativeRgUpRelease(nr);
        LOG("native: Real-ESRGAN upscaling ran out of video memory, Lanczos for the %s\n", where);
        nr.vsrFailed = true;
        return true;
    }
    nr.rgInN = inN;
    nr.rgTmpN = tmpN;
    nr.rgMidN = midN;
    nr.rgW = w;
    nr.rgH = h;
    NCHK(cudaMemsetAsync(nr.dRgIn, 0, (size_t)w * h * 3 * sizeof(uint16_t), nr.stream), "Real-ESRGAN warm-up clear");
    if (!restConvChain(nr, nr.stream, nr.dRgIn, nr.dRgAct, (char*)nr.dRgAct + actB / 2, nr.dRgW, nr.rgOff, nr.dRgOut, w,
                       h))
    {
        LOG("native: Real-ESRGAN warm-up launch failed\n");
        return false;
    }
    NCHK(cudaStreamSynchronize(nr.stream), "Real-ESRGAN warm-up sync");
    char passes[64] = "";
    if (n > 1)
        sprintf_s(passes, " in %d passes (%dx%d first)", n, pw[1], ph[1]);
    LOG("native: %s upscale: Real-ESRGAN animevideov3 x4 %dx%d -> Lanczos %dx%d%s%s (the host's kernels, fp16 "
        "accumulate, %.0f MB, %.1fs)\n",
        g_offline ? "offline" : "live", w, h, ow, oh, passes,
        nr.hdr ? ", on the HDR planes' SDR view" : (upEnc ? ", on the HDR video's SDR view" : ""), needB / 1048576.0,
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return true;
}

// one frame through Real-ESRGAN: s (planar, fp32 or fp16 by sHalf, plane / row strides ps_ / rs_) at the slot's input
// size, pass by pass: fp16 planes (k_restIn), the net, the fold to the pass's size (the last pass into dst, planar
// fp32, tight, clipped to 0..1; an earlier one into dRgMid, the next pass's input)
static bool nativeRgUpRun(NativeRife& nr, const void* s, int sHalf, int ps_, int rs_, float* dst, int tw, int th)
{
    cudaStream_t st = nr.stream;
    int pw[kRgMaxPasses + 1], ph[kRgMaxPasses + 1];
    const int n = rgUpPlan(nr.rgW, nr.rgH, tw, th, pw, ph);
    size_t inN, tmpN, midN;
    rgUpNeeds(n, pw, ph, inN, tmpN, midN);
    if (inN > nr.rgInN || tmpN > nr.rgTmpN || midN > nr.rgMidN)
    {
        nr.die("Real-ESRGAN upscale target larger than its buffers");
        return false;
    }
    void* actB = (char*)nr.dRgAct + nr.rgInN * 128;
    const void* src = s;
    int half = sHalf, ps = ps_, rs = rs_;
    for (int k = 0; k < n; k++)
    {
        int w = pw[k], h = ph[k];
        const int ow = pw[k + 1], oh = ph[k + 1];
        void* a[] = {(void*)&src, &half, &ps, &rs, &w, &h, &nr.dRgIn};
        if (cuLaunchKernel(nr.fRestIn, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a, nullptr) !=
                CUDA_SUCCESS ||
            !restConvChain(nr, st, nr.dRgIn, nr.dRgAct, actB, nr.dRgW, nr.rgOff, nr.dRgOut, w, h))
        {
            nr.die("Real-ESRGAN upscale launch failed");
            return false;
        }
        float* o = k + 1 == n ? dst : nr.dRgMid;
        if (!restFold(nr, st, nr.dRgOut, 1, true, nr.dRgIn, 1, 4, w, h, 4 * w, 4 * h, nr.dRgTmp, o, ow, oh))
            return false;
        src = nr.dRgMid;
        half = 0;
        ps = ow * oh;
        rs = ow;
    }
    return true;
}
