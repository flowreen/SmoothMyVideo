// part of smv-live-native.inl (included there, in place); not a standalone translation unit
// Restore = Anime4K's Restore CNN UL for this session (restNatKind 3): the Anime4K module, dRestIn (the three fp16
// planes every Restore input path fills), the four layer slots and the planar output at the source size, the
// statistics buffers, one run on zeros. false = the session cannot start, with its line (a Restore model that fails
// to load is treated the same).
static bool nativeA4kRestSetup(NativeRife& nr)
{
    std::string why;
    if (!nativeA4kModule(nr, why))
    {
        LOG("native: Restore unavailable (Anime4K: %s)\n", why.c_str());
        return false;
    }
    const int sw = srcW(nr), sh = srcH(nr);
    const size_t sp = (size_t)sw * sh;
    nr.restHalfIn = nr.restHalfOut = true; // the input path writes fp16 planes
    nr.restSub = false;
    nr.restScale = 1;
    nr.restSubBase = false;
    if ((!nr.dRestIn && cudaMalloc(&nr.dRestIn, 3 * sp * sizeof(uint16_t)) != cudaSuccess) ||
        cudaMalloc(&nr.dA4kRT, 4 * 3 * sp * 8) != cudaSuccess ||
        cudaMalloc((void**)&nr.dA4kRMid, 3 * sp * sizeof(float)) != cudaSuccess || !nativeA4kStatAlloc(nr, sw, sh))
    {
        cudaGetLastError();
        LOG("native: Restore unavailable (Anime4K: out of video memory at %dx%d)\n", sw, sh);
        return false;
    }
    NCHK(cudaMemsetAsync(nr.dRestIn, 0, 3 * sp * sizeof(uint16_t), nr.stream), "Anime4K restore warm-up clear");
    if (!nativeA4kStats(nr, nr.dRestIn, sw, sh, nr.stream) ||
        !nativeA4kPasses(nr, nr.stream, nr.dRestIn, sw, sh, nr.dA4kRT, nr.dA4kRMid, true) ||
        !nativeA4kClamp(nr, nr.dA4kRMid, sw, sh, nr.stream))
    {
        LOG("native: Anime4K restore warm-up launch failed\n");
        return false;
    }
    NCHK(cudaStreamSynchronize(nr.stream), "Anime4K restore warm-up sync");
    LOG("native: Restore = Anime4K Restore CNN UL on the host's kernels at %dx%d (CUDA passes, tensor-core layers with "
        "fp16 sums%s)\n",
        sw, sh, a4kClampOn() ? ", Clamp_Highlights" : "");
    return true;
}

// one Anime4K Restore on nr.stream: dRestIn (the source as three fp16 planes) -> the statistics -> the passes ->
// dA4kRMid (planar fp32, the net's values) -> the clamp + the clip to 0..1, unless the resize's Anime4K pass closes
// this frame's chain (a4kAfter: the statistics wait for it and the values go on unclipped, as the shader's textures
// do; clipAfter = the clip only, for HDR video, whose code restore reads a 0..1 view) -> dst at tw x th (a copy at the
// source size, else Restore's Lanczos pair)
static bool nativeA4kRestoreRun(NativeRife& nr, float* dst, int tw, int th, bool a4kAfter, bool clipAfter)
{
    cudaStream_t st = nr.stream;
    int sw = srcW(nr), sh = srcH(nr);
    const size_t sp = (size_t)sw * sh;
    if (!nativeA4kStats(nr, nr.dRestIn, sw, sh, st))
    {
        nr.die("Anime4K statistics launch failed");
        return false;
    }
    if (!nativeA4kPasses(nr, st, nr.dRestIn, sw, sh, nr.dA4kRT, nr.dA4kRMid, true))
    {
        nr.die("Anime4K restore pass launch failed");
        return false;
    }
    nr.a4kStatFresh = a4kAfter;
    if (!a4kAfter && !nativeA4kClamp(nr, nr.dA4kRMid, sw, sh, st))
    {
        nr.die("Anime4K clamp launch failed");
        return false;
    }
    if (a4kAfter && clipAfter)
    {
        int n = 3 * (int)sp;
        void* c[] = {&nr.dA4kRMid, &n};
        if (cuLaunchKernel(nr.fClamp01, (n + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, c, nullptr) != CUDA_SUCCESS)
        {
            nr.die("Anime4K restore clip launch failed");
            return false;
        }
    }
    if (tw == sw && th == sh)
    {
        if (cudaMemcpyAsync(dst, nr.dA4kRMid, 3 * sp * sizeof(float), cudaMemcpyDeviceToDevice, st) != cudaSuccess)
        {
            nr.die("Anime4K restore copy failed");
            return false;
        }
        return true;
    }
    if (nr.a4kRTmpW < tw)
    {
        // the fold's row buffer for this target width (a session's first frame at a width)
        cudaStreamSynchronize(st);
        if (nr.dA4kRTmp)
            cudaFree(nr.dA4kRTmp);
        nr.dA4kRTmp = nullptr;
        nr.a4kRTmpW = 0;
        if (cudaMalloc((void**)&nr.dA4kRTmp, (size_t)3 * tw * sh * sizeof(float)) != cudaSuccess)
        {
            cudaGetLastError();
            nr.die("Anime4K restore fold buffer failed");
            return false;
        }
        nr.a4kRTmpW = tw;
    }
    const void* tabH = rsTabFor(nr, sw, tw, st);
    const void* tabV = rsTabFor(nr, sh, th, st);
    if (!tabH || !tabV)
    {
        nr.die("Anime4K restore fold tap tables failed");
        return false;
    }
    int zero = 0, psi = (int)sp;
    const void* mid = nr.dA4kRMid;
    void* ah[] = {(void*)&mid, &zero, &psi, &sw, &sw, &sh, &nr.dA4kRTmp, &tw, (void*)&tabH};
    if (cuLaunchKernel(nr.fRestFoldH, (tw + 15) / 16, (sh + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, ah, nullptr) !=
        CUDA_SUCCESS)
    {
        nr.die("Anime4K restore fold (horizontal) launch failed");
        return false;
    }
    void* av[] = {&nr.dA4kRTmp, &tw, &sh, &dst, &th, (void*)&tabV};
    if (cuLaunchKernel(nr.fRestFoldV, (tw + 15) / 16, (th + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, av, nullptr) !=
        CUDA_SUCCESS)
    {
        nr.die("Anime4K restore fold (vertical) launch failed");
        return false;
    }
    return true;
}
