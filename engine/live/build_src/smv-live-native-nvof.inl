// part of smv-live-native.inl (included there, in place); not a standalone translation unit
// ---- the nvof model -------------------------------------------------------------------------
// The NVIDIA Optical Flow Accelerator through the driver's nvofapi64.dll (System32, opened by
// full path, never bundled; the process keeps it loaded once opened). One session per NativeRife
// at the true model size w x h: grid 4, preset fast, BOTH directions in one Execute, output cost
// on, temporal hints off (a paused or seeked source is not a continuous sequence). The defaults
// below are a sweep's pick (the metric weights barely move the result, gray input was best on
// the in-range clip). Every failure here is a REFUSAL.
static NV_OF_CUDA_API_FUNCTION_LIST g_nvofApi{};
static bool g_nvofLoaded = false;
static const float kNvofA = 0.1f, kNvofB = 1.0f;
// the pull-warp tween's fallback: log of the mean landing weight at or above kNvofHi keeps the warp,
// at or below kNvofLo takes the plain blend (linear between), the mask blurred by kNvofSigma px
// (picked by eye)
static const float kNvofLo = -9.0f, kNvofHi = -5.0f, kNvofSigma = 6.0f;
static const int kNvofGrid = 4;

// every Optical Flow session (the nvof model, DLSS 5's vectors, FSR's vectors) runs the engine's highest quality level,
// slow (the user's call: "we're not allowed to ship inferior quality"); SMV_OF_LEVEL=fast|medium|slow = A/B only
static NV_OF_PERF_LEVEL nativeOfLevel()
{
    char ev[16] = {};
    if (GetEnvironmentVariableA("SMV_OF_LEVEL", ev, sizeof(ev)) > 0)
    {
        if (!strcmp(ev, "fast"))
            return NV_OF_PERF_LEVEL_FAST;
        if (!strcmp(ev, "medium"))
            return NV_OF_PERF_LEVEL_MEDIUM;
    }
    return NV_OF_PERF_LEVEL_SLOW;
}

static const char* nativeOfLevelName(NV_OF_PERF_LEVEL l)
{
    return l == NV_OF_PERF_LEVEL_FAST ? "fast" : l == NV_OF_PERF_LEVEL_MEDIUM ? "medium" : "slow";
}

static bool nativeNvofLoad()
{
    if (g_nvofLoaded)
        return true;
    wchar_t sys[MAX_PATH]{};
    const UINT n = GetSystemDirectoryW(sys, MAX_PATH);
    if (!n || n >= MAX_PATH)
    {
        LOG("native: nvof: the system folder is unknown\n");
        return false;
    }
    const std::wstring p = std::wstring(sys) + L"\\nvofapi64.dll";
    HMODULE m = LoadLibraryExW(p.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!m)
    {
        LOG("native: nvof: nvofapi64.dll not found in the system folder (err %lu); the NVIDIA display driver installs it\n",
            GetLastError());
        return false;
    }
    typedef NV_OF_STATUS(NVOFAPI * PFN_CreateInstanceCuda)(uint32_t, NV_OF_CUDA_API_FUNCTION_LIST*);
    auto create = (PFN_CreateInstanceCuda)GetProcAddress(m, "NvOFAPICreateInstanceCuda");
    if (!create)
    {
        LOG("native: nvof: NvOFAPICreateInstanceCuda is not exported by nvofapi64.dll\n");
        return false;
    }
    const NV_OF_STATUS s = create(NV_OF_API_VERSION, &g_nvofApi);
    if (s != NV_OF_SUCCESS)
    {
        LOG("native: nvof: NvOFAPICreateInstanceCuda(0x%x) refused (status %d), driver too old for the Optical Flow SDK 5 interface\n",
            (unsigned)NV_OF_API_VERSION, (int)s);
        return false;
    }
    g_nvofLoaded = true;
    return true;
}

static bool nativeNvofBufOn(NvOFHandle ofh, uint32_t w, uint32_t h, NV_OF_BUFFER_USAGE usage, NV_OF_BUFFER_FORMAT fmt,
                            NvOFGPUBufferHandle& hb, CUdeviceptr& ptr, uint32_t& pitch, const char* what)
{
    NV_OF_BUFFER_DESCRIPTOR d{};
    d.width = w;
    d.height = h;
    d.bufferUsage = usage;
    d.bufferFormat = fmt;
    NV_OF_STATUS s = g_nvofApi.nvOFCreateGPUBufferCuda(ofh, &d, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &hb);
    if (s != NV_OF_SUCCESS)
    {
        LOG("native: nvof: %s buffer creation failed (status %d)\n", what, (int)s);
        return false;
    }
    ptr = g_nvofApi.nvOFGPUBufferGetCUdeviceptr(hb);
    NV_OF_CUDA_BUFFER_STRIDE_INFO si{};
    s = g_nvofApi.nvOFGPUBufferGetStrideInfo(hb, &si);
    if (s != NV_OF_SUCCESS || !ptr)
    {
        LOG("native: nvof: %s buffer stride query failed (status %d)\n", what, (int)s);
        return false;
    }
    pitch = si.strideInfo[0].strideXInBytes;
    return true;
}
static bool nativeNvofBuf(NativeRife& nr, uint32_t w, uint32_t h, NV_OF_BUFFER_USAGE usage, NV_OF_BUFFER_FORMAT fmt,
                          NvOFGPUBufferHandle& hb, CUdeviceptr& ptr, uint32_t& pitch, const char* what)
{
    return nativeNvofBufOn(nr.ofH, w, h, usage, fmt, hb, ptr, pitch, what);
}

// the session and the glue buffers; the calling thread has the device bound
static bool nativeNvofSetup(NativeRife& nr)
{
    if (!nativeNvofLoad())
        return false;
    if (nr.w < 32 || nr.h < 32)
    {
        LOG("native: nvof: the model frame %dx%d is below the Optical Flow minimum 32x32\n", nr.w, nr.h);
        return false;
    }
    CUcontext ctx = nullptr;
    if (cuCtxGetCurrent(&ctx) != CUDA_SUCCESS || !ctx)
    {
        LOG("native: nvof: no current CUDA context\n");
        return false;
    }
    NV_OF_STATUS s = g_nvofApi.nvCreateOpticalFlowCuda(ctx, &nr.ofH);
    if (s != NV_OF_SUCCESS)
    {
        nr.ofH = nullptr;
        LOG("native: nvof: nvCreateOpticalFlowCuda failed (status %d)\n", (int)s);
        return false;
    }
    NV_OF_INIT_PARAMS ip{};
    ip.width = (uint32_t)nr.w;
    ip.height = (uint32_t)nr.h;
    ip.outGridSize = (NV_OF_OUTPUT_VECTOR_GRID_SIZE)kNvofGrid;
    ip.hintGridSize = NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED;
    ip.mode = NV_OF_MODE_OPTICALFLOW;
    ip.perfLevel = nativeOfLevel();
    ip.enableExternalHints = NV_OF_FALSE;
    ip.enableOutputCost = NV_OF_TRUE;
    ip.disparityRange = NV_OF_STEREO_DISPARITY_RANGE_UNDEFINED;
    ip.enableRoi = NV_OF_FALSE;
    ip.predDirection = NV_OF_PRED_DIRECTION_BOTH;
    ip.enableGlobalFlow = NV_OF_FALSE;
    ip.inputBufferFormat = NV_OF_BUFFER_FORMAT_GRAYSCALE8;
    s = g_nvofApi.nvOFInit(nr.ofH, &ip);
    if (s != NV_OF_SUCCESS)
    {
        LOG("native: nvof: nvOFInit %dx%d grid %d refused (status %d)\n", nr.w, nr.h, kNvofGrid, (int)s);
        return false;
    }
    // the output grid is ceil(size / grid): verified at 1914x1078 -> 479x270
    nr.ofGw = (nr.w + kNvofGrid - 1) / kNvofGrid;
    nr.ofGh = (nr.h + kNvofGrid - 1) / kNvofGrid;
    for (int k = 0; k < 2; k++)
    {
        uint32_t inPitch = 0;
        if (!nativeNvofBuf(nr, nr.w, nr.h, NV_OF_BUFFER_USAGE_INPUT, NV_OF_BUFFER_FORMAT_GRAYSCALE8, nr.ofIn[k],
                           nr.ofInP[k], inPitch, "input") ||
            !nativeNvofBuf(nr, nr.ofGw, nr.ofGh, NV_OF_BUFFER_USAGE_OUTPUT, NV_OF_BUFFER_FORMAT_SHORT2, nr.ofOut[k],
                           nr.ofOutP[k], nr.ofOutPitch[k], "flow") ||
            !nativeNvofBuf(nr, nr.ofGw, nr.ofGh, NV_OF_BUFFER_USAGE_COST, NV_OF_BUFFER_FORMAT_UINT8, nr.ofCost[k],
                           nr.ofCostP[k], nr.ofCostPitch[k], "cost"))
            return false;
        if (k == 0)
            nr.ofInPitch = inPitch;
        else if (inPitch != nr.ofInPitch)
        {
            LOG("native: nvof: the two input buffers differ in pitch\n");
            return false;
        }
    }
    s = g_nvofApi.nvOFSetIOCudaStreams(nr.ofH, (CUstream)nr.stream, (CUstream)nr.stream);
    if (s != NV_OF_SUCCESS)
    {
        LOG("native: nvof: nvOFSetIOCudaStreams failed (status %d)\n", (int)s);
        return false;
    }
    const size_t mp = (size_t)nr.w * nr.h;
    for (int k = 0; k < 2; k++)
    {
        NCHK(cudaMalloc((void**)&nr.dNvFlow[k], 2 * mp * sizeof(float)), "alloc nvof flow");
        NCHK(cudaMalloc((void**)&nr.dNvCost[k], mp * sizeof(float)), "alloc nvof cost");
        NCHK(cudaMalloc((void**)&nr.dNvZ[k], mp * sizeof(float)), "alloc nvof metric");
    }
    NCHK(cudaMalloc((void**)&nr.dNvAcc, 5 * mp * sizeof(long long)), "alloc nvof accumulator");
    // the pull-warp tween's planes and the push-pull pyramid (levels halve, ceil, until the short
    // side is 1 px; level 0 is dNvN0 / dNvD0 in, dNvV out)
    NCHK(cudaMalloc((void**)&nr.dNvN0, 2 * mp * sizeof(float)), "alloc nvof velocity");
    NCHK(cudaMalloc((void**)&nr.dNvD0, mp * sizeof(float)), "alloc nvof confidence");
    NCHK(cudaMalloc((void**)&nr.dNvVis, 2 * mp * sizeof(float)), "alloc nvof visibility");
    NCHK(cudaMalloc((void**)&nr.dNvMraw, mp * sizeof(float)), "alloc nvof mask");
    NCHK(cudaMalloc((void**)&nr.dNvMask, mp * sizeof(float)), "alloc nvof blurred mask");
    NCHK(cudaMalloc((void**)&nr.dNvBlurTmp, mp * sizeof(float)), "alloc nvof blur pass");
    NCHK(cudaMalloc((void**)&nr.dNvV, 2 * mp * sizeof(float)), "alloc nvof filled velocity");
    for (int lw = nr.w, lh = nr.h; (lw < lh ? lw : lh) > 1;)
    {
        lw = (lw + 1) / 2;
        lh = (lh + 1) / 2;
        NativeRife::NvLevel L{lw, lh, nullptr, nullptr, nullptr};
        const size_t lp = (size_t)lw * lh;
        nr.nvPyr.push_back(L);
        NativeRife::NvLevel& B = nr.nvPyr.back();
        NCHK(cudaMalloc((void**)&B.n, 2 * lp * sizeof(float)), "alloc nvof pyramid");
        NCHK(cudaMalloc((void**)&B.d, lp * sizeof(float)), "alloc nvof pyramid");
        NCHK(cudaMalloc((void**)&B.out, 2 * lp * sizeof(float)), "alloc nvof pyramid");
    }
    // the tween in the model layout (3, ph, pw); the pad outside w x h is never written, zero it
    // once so no store path can ever read uninitialised memory there
    const size_t plane = (size_t)nr.ph * nr.pw;
    NCHK(cudaMalloc((void**)&nr.dNvOut, 3 * plane * sizeof(float)), "alloc nvof tween");
    NCHK(cudaMemsetAsync(nr.dNvOut, 0, 3 * plane * sizeof(float), nr.stream), "clear nvof tween");
    nr.nvProf = GetEnvironmentVariableW(L"SMV_LIVE_NVOF_PROF", nullptr, 0) != 0; // the GMFSS lever's rule
    if (nr.nvProf)
    {
        NCHK(cudaEventCreate(&nr.nvEv[0]), "create nvof event");
        NCHK(cudaEventCreate(&nr.nvEv[1]), "create nvof event");
    }
    LOG("native: nvof session %dx%d, grid %d (%dx%d), %s, both directions, gray8\n", nr.w, nr.h, kNvofGrid, nr.ofGw,
        nr.ofGh, nativeOfLevelName(ip.perfLevel));
    return true;
}

// the pair: both frames to gray8, one Execute for both fields, both upsampled, both metrics.
// dPrev / dCur are the model planes at (ph * pw, pw) strides.
static bool nativeNvofPair(NativeRife& nr, const float* dPrev, const float* dCur)
{
    cudaStream_t st = nr.stream;
    const size_t plane = (size_t)nr.ph * nr.pw;
    if (nr.nvProf)
        cudaEventRecord(nr.nvEv[0], st);
    int nps = (int)plane, pitch = (int)nr.ofInPitch;
    for (int k = 0; k < 2; k++)
    {
        const float* r = k ? dCur : dPrev;
        CUdeviceptr dst = nr.ofInP[k];
        void* a[] = {(void*)&r, &nps, &nr.pw, &nr.w, &nr.h, &dst, &pitch};
        if (cuLaunchKernel(nr.fNvofLuma, (nr.w + 15) / 16, (nr.h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a,
                           nullptr) != CUDA_SUCCESS)
        {
            nr.die("nvofLuma launch failed");
            return false;
        }
    }
    NV_OF_EXECUTE_INPUT_PARAMS ei{};
    ei.inputFrame = nr.ofIn[0]; // frame 0 -> "inputFrame": the forward field is F01
    ei.referenceFrame = nr.ofIn[1];
    ei.disableTemporalHints = NV_OF_TRUE;
    NV_OF_EXECUTE_OUTPUT_PARAMS eo{};
    eo.outputBuffer = nr.ofOut[0];
    eo.outputCostBuffer = nr.ofCost[0];
    eo.bwdOutputBuffer = nr.ofOut[1];
    eo.bwdOutputCostBuffer = nr.ofCost[1];
    const NV_OF_STATUS s = g_nvofApi.nvOFExecute(nr.ofH, &ei, &eo);
    if (s != NV_OF_SUCCESS)
    {
        LOG("native: nvof: nvOFExecute failed (status %d)\n", (int)s);
        nr.die("nvof execute failed");
        return false;
    }
    int grid = kNvofGrid;
    for (int k = 0; k < 2; k++)
    {
        CUdeviceptr v = nr.ofOutP[k], c = nr.ofCostP[k];
        int vp = (int)nr.ofOutPitch[k], cp = (int)nr.ofCostPitch[k];
        void* a[] = {&v, &vp, &c, &cp, &nr.ofGw, &nr.ofGh, &grid, &nr.w, &nr.h, &nr.dNvFlow[k], &nr.dNvCost[k]};
        if (cuLaunchKernel(nr.fNvofUp, (nr.w + 15) / 16, (nr.h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a, nullptr) !=
            CUDA_SUCCESS)
        {
            nr.die("nvofUp launch failed");
            return false;
        }
    }
    float ma = kNvofA, mb = kNvofB;
    for (int k = 0; k < 2; k++)
    {
        void* a[] = {&nr.dNvFlow[k], &nr.dNvCost[k], &nr.dNvFlow[k ^ 1], &nr.w, &nr.h, &ma, &mb, &nr.dNvZ[k]};
        if (cuLaunchKernel(nr.fNvofMetric, (nr.w + 15) / 16, (nr.h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a,
                           nullptr) != CUDA_SUCCESS)
        {
            nr.die("nvofMetric launch failed");
            return false;
        }
    }
    if (nr.nvProf)
    {
        cudaEventRecord(nr.nvEv[1], st);
        float ms = 0.0f;
        if (cudaEventSynchronize(nr.nvEv[1]) == cudaSuccess &&
            cudaEventElapsedTime(&ms, nr.nvEv[0], nr.nvEv[1]) == cudaSuccess)
        {
            nr.nvPairMs += ms;
            nr.nvPairN++;
        }
        if (nr.nvPairN && nr.nvPairN % 64 == 0)
            LOG("[nvof-prof] pair (luma + execute + upsample + metric) %.3f ms mean over %u pairs, tween %.3f ms mean\n",
                nr.nvPairMs / nr.nvPairN, nr.nvPairN, nr.nvTweenN ? nr.nvTweenMs / nr.nvTweenN : 0.0);
    }
    return true;
}

// one tween at t into dNvOut, the pull-warp chain: the
// velocity of both frames splatted to time t, its gaps push-pull filled, both frames SAMPLED
// along it, and where the vectors are untrustworthy (the blurred fallback mask) the plain blend
// instead: failed vectors show as ghosting, never as speckle
static bool nativeNvofTween(NativeRife& nr, float t)
{
    cudaStream_t st = nr.stream;
    const size_t plane = (size_t)nr.ph * nr.pw, mp = (size_t)nr.w * nr.h;
    const unsigned gx = (unsigned)(nr.w + 15) / 16, gy = (unsigned)(nr.h + 15) / 16;
    auto launch = [&](CUfunction f, unsigned bx, unsigned by, void** a, const char* what) -> bool {
        if (cuLaunchKernel(f, bx, by, 1, 16, 16, 1, 0, (CUstream)st, a, nullptr) == CUDA_SUCCESS)
            return true;
        nr.die(what);
        return false;
    };
    if (nr.nvProf)
        cudaEventRecord(nr.nvEv[0], st);
    if (cudaMemsetAsync(nr.dNvAcc, 0, 5 * mp * sizeof(long long), st) != cudaSuccess)
    {
        nr.die("nvof accumulator clear failed");
        return false;
    }
    // (1) the velocity (frame 0 -> 1 px) of both frames at time t, coverage per frame
    float s0 = t, s1 = 1.0f - t;
    float b0 = logf(1.0f - t), b1 = logf(t);
    float sg0 = 1.0f, sg1 = -1.0f;
    int cp0 = 3, cp1 = 4;
    for (int k = 0; k < 2; k++)
    {
        void* a[] = {&nr.dNvFlow[k],  &nr.dNvZ[k], k ? &s1 : &s0, k ? &b1 : &b0, k ? &sg1 : &sg0,
                     k ? &cp1 : &cp0, &nr.w,       &nr.h,         &nr.dNvAcc};
        if (!launch(nr.fSplatVel, gx, gy, a, "splatVel launch failed"))
            return false;
    }
    // (2) the merged velocity, confidence, visibilities, the raw fallback mask
    float lo = kNvofLo, hi = kNvofHi;
    {
        void* a[] = {&nr.dNvAcc, &nr.w, &nr.h, &t, &lo, &hi, &nr.dNvN0, &nr.dNvD0, &nr.dNvVis, &nr.dNvMraw};
        if (!launch(nr.fVelNorm, gx, gy, a, "velNorm launch failed"))
            return false;
    }
    // (3) push-pull: down to 1 px, the top normalised, back up to level 0 (dNvV)
    float* pn = nr.dNvN0;
    float* pd = nr.dNvD0;
    int pw_ = nr.w, ph_ = nr.h;
    for (auto& L : nr.nvPyr)
    {
        void* a[] = {&pn, &pd, &pw_, &ph_, &L.n, &L.d, &L.w, &L.h};
        if (!launch(nr.fPpDown, (unsigned)(L.w + 15) / 16, (unsigned)(L.h + 15) / 16, a, "ppDown launch failed"))
            return false;
        pn = L.n;
        pd = L.d;
        pw_ = L.w;
        ph_ = L.h;
    }
    if (nr.nvPyr.empty())
    {
        nr.die("nvof pyramid empty");
        return false;
    }
    {
        NativeRife::NvLevel& T = nr.nvPyr.back();
        void* a[] = {&T.n, &T.d, &T.w, &T.h, &T.out};
        if (!launch(nr.fPpTop, (unsigned)(T.w + 15) / 16, (unsigned)(T.h + 15) / 16, a, "ppTop launch failed"))
            return false;
    }
    for (int i = (int)nr.nvPyr.size() - 2; i >= -1; i--)
    {
        NativeRife::NvLevel& C = nr.nvPyr[i + 1]; // the coarser level, already filled
        float* n = i >= 0 ? nr.nvPyr[i].n : nr.dNvN0;
        float* d = i >= 0 ? nr.nvPyr[i].d : nr.dNvD0;
        float* o = i >= 0 ? nr.nvPyr[i].out : nr.dNvV;
        int lw = i >= 0 ? nr.nvPyr[i].w : nr.w, lh = i >= 0 ? nr.nvPyr[i].h : nr.h;
        void* a[] = {&n, &d, &lw, &lh, &C.out, &C.w, &C.h, &o};
        if (!launch(nr.fPpUp, (unsigned)(lw + 15) / 16, (unsigned)(lh + 15) / 16, a, "ppUp launch failed"))
            return false;
    }
    // (4) the mask blurred (x then y, scipy's gaussian_filter)
    float sig = kNvofSigma;
    int d0 = 0, d1 = 1;
    {
        void* a[] = {&nr.dNvMraw, &nr.w, &nr.h, &sig, &d0, &nr.dNvBlurTmp};
        if (!launch(nr.fBlur1, gx, gy, a, "blur1 (x) launch failed"))
            return false;
        void* b[] = {&nr.dNvBlurTmp, &nr.w, &nr.h, &sig, &d1, &nr.dNvMask};
        if (!launch(nr.fBlur1, gx, gy, b, "blur1 (y) launch failed"))
            return false;
    }
    // (5) sample both frames along the velocity, mix with the plain blend by the mask
    int ps = (int)plane;
    const float* i0 = nativeXPrev(nr);
    const float* i1 = nativeXCur(nr);
    {
        void* a[] = {(void*)&i0,  (void*)&i1, &ps,   &nr.pw, &nr.dNvV,  &nr.dNvVis,
                     &nr.dNvMask, &nr.w,      &nr.h, &t,     &nr.dNvOut};
        if (!launch(nr.fNvofCompose, gx, gy, a, "nvofCompose launch failed"))
            return false;
    }
    if (nr.nvProf)
    {
        cudaEventRecord(nr.nvEv[1], st);
        float ms = 0.0f;
        if (cudaEventSynchronize(nr.nvEv[1]) == cudaSuccess &&
            cudaEventElapsedTime(&ms, nr.nvEv[0], nr.nvEv[1]) == cudaSuccess)
        {
            nr.nvTweenMs += ms;
            nr.nvTweenN++;
        }
    }
    return true;
}

static void nativeNvofFree(NativeRife& nr)
{
    if (nr.nvProf && nr.nvPairN)
        LOG("[nvof-prof] session: pair %.3f ms mean over %u pairs, tween %.3f ms mean over %u tweens\n",
            nr.nvPairMs / nr.nvPairN, nr.nvPairN, nr.nvTweenN ? nr.nvTweenMs / nr.nvTweenN : 0.0, nr.nvTweenN);
    if (g_nvofLoaded)
    {
        for (int k = 0; k < 2; k++)
            for (NvOFGPUBufferHandle* b : {&nr.ofIn[k], &nr.ofOut[k], &nr.ofCost[k]})
                if (*b)
                {
                    g_nvofApi.nvOFDestroyGPUBufferCuda(*b);
                    *b = nullptr;
                }
        if (nr.ofH)
        {
            g_nvofApi.nvOFDestroy(nr.ofH);
            nr.ofH = nullptr;
        }
    }
    for (int k = 0; k < 2; k++)
    {
        for (float** p : {&nr.dNvFlow[k], &nr.dNvCost[k], &nr.dNvZ[k]})
            if (*p)
            {
                cudaFree(*p);
                *p = nullptr;
            }
        nr.ofInP[k] = nr.ofOutP[k] = nr.ofCostP[k] = 0;
    }
    if (nr.dNvAcc)
    {
        cudaFree(nr.dNvAcc);
        nr.dNvAcc = nullptr;
    }
    if (nr.dNvOut)
    {
        cudaFree(nr.dNvOut);
        nr.dNvOut = nullptr;
    }
    for (float** p : {&nr.dNvN0, &nr.dNvD0, &nr.dNvVis, &nr.dNvMraw, &nr.dNvMask, &nr.dNvBlurTmp, &nr.dNvV})
        if (*p)
        {
            cudaFree(*p);
            *p = nullptr;
        }
    for (auto& L : nr.nvPyr)
        for (float* p : {L.n, L.d, L.out})
            if (p)
                cudaFree(p);
    nr.nvPyr.clear();
    for (auto& e : nr.nvEv)
        if (e)
        {
            cudaEventDestroy(e);
            e = nullptr;
        }
}
