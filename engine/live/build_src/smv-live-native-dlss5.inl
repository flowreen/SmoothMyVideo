// part of smv-live-native.inl (included there, in place); not a standalone translation unit
static float halfToFloat(uint16_t h); // defined in smv-live.cpp, after its includes

// --nr-delta PATH (the preview's change mask): the
// largest channel change of the DLSS 5 pass per pixel, |after - before| with `before` the pass's
// fp32 input planes (tight, tw x th each) and `after` its fp16 output clamped to 0..1 like
// k_nrOut, or the given planes (HDR video: the frame after k_nrOutPq), tw x th float32, rewritten
// per frame (the last frame's). A failed write costs only the mask.
static void nativeOfflineNrDelta(const NativeRife& nr, const std::vector<float>& before, int tw, int th,
                                 const std::vector<float>* after = nullptr)
{
    const size_t n = (size_t)tw * th;
    std::vector<float> d(n);
    for (size_t o = 0; o < n; o++)
    {
        float m = 0.0f;
        for (int c = 0; c < 3; c++)
        {
            float a = after ? (*after)[c * n + o] : halfToFloat(nr.hNrOut[o * 4 + c]);
            if (!after)
                a = a < 0.0f ? 0.0f : (a > 1.0f ? 1.0f : a);
            const float v = fabsf(a - before[c * n + o]);
            if (v > m)
                m = v;
        }
        d[o] = m > 1.0f ? 1.0f : m;
    }
    FILE* f = nullptr;
    if (_wfopen_s(&f, nr.nrDeltaPath.c_str(), L"wb") || !f || fwrite(d.data(), sizeof(float), d.size(), f) != d.size())
        LOG("offline: cannot write the DLSS 5 change map %s\n", wideToUtf8(nr.nrDeltaPath).c_str());
    if (f)
        fclose(f);
}

// The zero-copy imports of the DLSS 5 handoff, released in the order CUDA asks for (the surface
// objects and mapped arrays first). The caller has drained the stream.
static void nativeNrReleaseImports(NativeRife& nr)
{
    for (unsigned long long* s : {&nr.sfNrIn, &nr.sfNrOut, &nr.sfNrMv})
        if (*s)
        {
            cudaDestroySurfaceObject(*s);
            *s = 0;
        }
    for (cudaMipmappedArray_t* m : {&nr.mmNrIn, &nr.mmNrOut, &nr.mmNrMv})
        if (*m)
        {
            cudaFreeMipmappedArray(*m);
            *m = nullptr;
        }
    if (nr.semNr)
    {
        cudaDestroyExternalSemaphore(nr.semNr);
        nr.semNr = nullptr;
    }
    for (cudaExternalMemory_t* e : {&nr.emNrIn, &nr.emNrOut, &nr.emNrMv})
        if (*e)
        {
            cudaDestroyExternalMemory(*e);
            *e = nullptr;
        }
}

// the current CUDA device's adapter LUID: the NR core creates its D3D12 device on that adapter, so the
// zero-copy handoff shares one GPU even when another GPU drives the main display
static bool nativeCudaLuid(LUID& out)
{
    int dev = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess)
        return false;
    memcpy(&out, prop.luid, sizeof(out));
    return true;
}

// one of the NR core's textures as a CUDA array (a dedicated import of its whole allocation, mapped
// with surface load / store) and its surface object
static bool nativeNrImportTex(HANDLE h, uint64_t bytes, bool rg, int w, int h2, cudaExternalMemory_t& em,
                              cudaMipmappedArray_t& mm, unsigned long long& sf)
{
    cudaExternalMemoryHandleDesc md{};
    md.type = cudaExternalMemoryHandleTypeD3D12Resource;
    md.size = bytes;
    md.flags = cudaExternalMemoryDedicated;
    md.handle.win32.handle = h;
    if (cudaImportExternalMemory(&em, &md) != cudaSuccess)
        return false;
    cudaExternalMemoryMipmappedArrayDesc ad{};
    ad.formatDesc = rg ? cudaCreateChannelDesc(16, 16, 0, 0, cudaChannelFormatKindFloat)
                       : cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindFloat);
    ad.extent = cudaExtent{(size_t)w, (size_t)h2, 0};
    ad.numLevels = 1;
    ad.flags = cudaArraySurfaceLoadStore;
    cudaArray_t arr = nullptr;
    if (cudaExternalMemoryGetMappedMipmappedArray(&mm, em, &ad) != cudaSuccess ||
        cudaGetMipmappedArrayLevel(&arr, mm, 0) != cudaSuccess)
        return false;
    cudaResourceDesc rd{};
    rd.resType = cudaResourceTypeArray;
    rd.res.array.array = arr;
    cudaSurfaceObject_t s = 0;
    if (cudaCreateSurfaceObject(&s, &rd) != cudaSuccess)
        return false;
    sf = (unsigned long long)s;
    return true;
}

// Zero-copy DLSS 5: import the NR core's Color / Output / MVec textures and fence
// (nr::Host::startShared) into this CUDA context. Refused when the core's D3D12 adapter is not the
// current CUDA device; the caller then keeps the CPU staging route.
static bool nativeNrImport(NativeRife& nr, nr::Host& host, int w, int h, std::string& err)
{
    int dev = 0;
    cudaDeviceProp prop{};
    const LUID luid = host.adapterLuid();
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess ||
        memcmp(prop.luid, &luid, sizeof(luid)) != 0)
    {
        err = "the NR device is not the CUDA device";
        return false;
    }
    if (!nativeNrImportTex(host.colorHandle(), host.colorBytes(), false, w, h, nr.emNrIn, nr.mmNrIn, nr.sfNrIn) ||
        !nativeNrImportTex(host.outputHandle(), host.outputBytes(), false, w, h, nr.emNrOut, nr.mmNrOut, nr.sfNrOut))
    {
        cudaGetLastError();
        err = "Color / Output texture import failed";
        return false;
    }
    cudaExternalSemaphoreHandleDesc sd{};
    sd.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
    sd.handle.win32.handle = host.sharedFenceHandle();
    if (cudaImportExternalSemaphore(&nr.semNr, &sd) != cudaSuccess)
    {
        err = "fence import failed";
        return false;
    }
    if (host.mvHandle() &&
        !nativeNrImportTex(host.mvHandle(), host.mvBytes(), true, w, h, nr.emNrMv, nr.mmNrMv, nr.sfNrMv))
    {
        cudaGetLastError();
        err = "MVec texture import failed";
        return false;
    }
    return true;
}

// a zero DLSSNR.MVec (k_nrMvClear) on the session's stream
static bool nativeNrMvClear(NativeRife& nr)
{
    int w = nr.nrW, h = nr.nrH;
    void* a[] = {&nr.sfNrMv, &w, &h};
    return cuLaunchKernel(nr.fNrMvClear, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)nr.stream, a,
                          nullptr) == CUDA_SUCCESS;
}

// DLSS 5 motion, set up once the core's textures are imported: an Optical Flow session at the NR
// size (forward only: the current frame is the input, the previous one the reference), its two
// luma slots, the field buffer and the upsampled field (no cost output: k_nrMv reads none). false = no
// motion for this render (the caller says why and frees what was made).
static bool nativeNrMotionSetup(NativeRife& nr, std::string& err)
{
    const int w = nr.nrW, h = nr.nrH;
    if (!nr.sfNrMv)
    {
        err = "the core shares no MVec texture";
        return false;
    }
    if (!nativeNvofLoad())
    {
        err = "the Optical Flow runtime is unavailable";
        return false;
    }
    if (w < 32 || h < 32)
    {
        err = "below the Optical Flow minimum 32x32";
        return false;
    }
    CUcontext ctx = nullptr;
    if (cuCtxGetCurrent(&ctx) != CUDA_SUCCESS || !ctx)
    {
        err = "no current CUDA context";
        return false;
    }
    if (g_nvofApi.nvCreateOpticalFlowCuda(ctx, &nr.nrOfH) != NV_OF_SUCCESS)
    {
        nr.nrOfH = nullptr;
        err = "nvCreateOpticalFlowCuda failed";
        return false;
    }
    NV_OF_INIT_PARAMS ip{};
    ip.width = (uint32_t)w;
    ip.height = (uint32_t)h;
    ip.outGridSize = (NV_OF_OUTPUT_VECTOR_GRID_SIZE)kNvofGrid;
    ip.hintGridSize = NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED;
    ip.mode = NV_OF_MODE_OPTICALFLOW;
    ip.perfLevel = nativeOfLevel();
    ip.enableExternalHints = NV_OF_FALSE;
    ip.enableOutputCost = NV_OF_FALSE;
    ip.disparityRange = NV_OF_STEREO_DISPARITY_RANGE_UNDEFINED;
    ip.enableRoi = NV_OF_FALSE;
    ip.predDirection = NV_OF_PRED_DIRECTION_FORWARD;
    ip.enableGlobalFlow = NV_OF_FALSE;
    ip.inputBufferFormat = NV_OF_BUFFER_FORMAT_GRAYSCALE8;
    if (g_nvofApi.nvOFInit(nr.nrOfH, &ip) != NV_OF_SUCCESS)
    {
        err = "nvOFInit refused the NR size";
        return false;
    }
    nr.nrOfGw = (w + kNvofGrid - 1) / kNvofGrid;
    nr.nrOfGh = (h + kNvofGrid - 1) / kNvofGrid;
    uint32_t p1 = 0;
    if (!nativeNvofBufOn(nr.nrOfH, w, h, NV_OF_BUFFER_USAGE_INPUT, NV_OF_BUFFER_FORMAT_GRAYSCALE8, nr.nrOfIn[0],
                         nr.nrOfInP[0], nr.nrOfInPitch, "dlss5 input") ||
        !nativeNvofBufOn(nr.nrOfH, w, h, NV_OF_BUFFER_USAGE_INPUT, NV_OF_BUFFER_FORMAT_GRAYSCALE8, nr.nrOfIn[1],
                         nr.nrOfInP[1], p1, "dlss5 input") ||
        !nativeNvofBufOn(nr.nrOfH, nr.nrOfGw, nr.nrOfGh, NV_OF_BUFFER_USAGE_OUTPUT, NV_OF_BUFFER_FORMAT_SHORT2,
                         nr.nrOfOut, nr.nrOfOutP, nr.nrOfOutPitch, "dlss5 flow"))
    {
        err = "Optical Flow buffer creation failed";
        return false;
    }
    if (p1 != nr.nrOfInPitch)
    {
        err = "the two luma slots differ in pitch";
        return false;
    }
    if (g_nvofApi.nvOFSetIOCudaStreams(nr.nrOfH, (CUstream)nr.stream, (CUstream)nr.stream) != NV_OF_SUCCESS)
    {
        err = "nvOFSetIOCudaStreams failed";
        return false;
    }
    if (cudaMalloc((void**)&nr.dNrFlow, 2 * (size_t)w * h * sizeof(float)) != cudaSuccess)
    {
        nr.dNrFlow = nullptr;
        err = "motion field allocation failed";
        return false;
    }
    nr.nrOfCur = 0;
    nr.nrMotion = true;
    return true;
}

// Everything nativeNrMotionSetup made; the caller has drained the stream.
static void nativeNrMotionFree(NativeRife& nr)
{
    for (NvOFGPUBufferHandle* b : {&nr.nrOfIn[0], &nr.nrOfIn[1], &nr.nrOfOut})
        if (*b)
        {
            g_nvofApi.nvOFDestroyGPUBufferCuda(*b);
            *b = nullptr;
        }
    if (nr.nrOfH)
    {
        g_nvofApi.nvOFDestroy(nr.nrOfH);
        nr.nrOfH = nullptr;
    }
    if (nr.dNrFlow)
    {
        cudaFree(nr.dNrFlow);
        nr.dNrFlow = nullptr;
    }
    nr.nrMotion = false;
}

// the motion field of one frame for DLSS 5 (zero-copy route, before the input fence): the frame's
// BT.709 luma into this frame's slot; from the second frame on NVOFA current -> previous, upsampled
// (k_nvofUp) and validated (k_nrMv) into the core's MVec texture; the stream's first frame
// (a Reset) gets a zero field. The slots alternate, so the previous frame's luma stays in place.
static const float kNrMvMargin = 25.0f; // one 8-bit level per pixel of the 5x5 window (k_nrMv)
static const float kNrMvReal = 0.5f;    // a window that changed by more than half a level a pixel ...
static const float kNrMvShare = 0.4f;   // ... keeps a vector that removes at least 40 % of that change (k_nrMv)
static const float kNrMvFlat = 1.0f;    // a window neither test decides keeps a vector of 1 px or more ...
static const float kNrMvCoh = 1.0f;     // ... when the field 8 px away agrees within 1 px (k_nrMv)
static bool nativeNrMotion(NativeRife& nr, const float* frame, int pw, int ps)
{
    cudaStream_t st = nr.stream;
    int w = nr.nrW, h = nr.nrH;
    const int cur = nr.nrOfCur, prev = cur ^ 1;
    nr.nrOfCur = prev;
    const float* r0 = frame;
    int nps = ps, lpitch = (int)nr.nrOfInPitch;
    CUdeviceptr lc = nr.nrOfInP[cur], lp = nr.nrOfInP[prev];
    void* a[] = {(void*)&r0, &nps, &pw, &w, &h, &lc, &lpitch};
    if (cuLaunchKernel(nr.fNvofLuma, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a, nullptr) !=
        CUDA_SUCCESS)
    {
        nr.die("DLSS 5 motion: luma launch failed");
        return false;
    }
    if (nr.nrFirst || nr.nrResetEvery)
    {
        if (!nativeNrMvClear(nr))
        {
            nr.die("DLSS 5 motion: clear failed");
            return false;
        }
        return true;
    }
    NV_OF_EXECUTE_INPUT_PARAMS ei{};
    ei.inputFrame = nr.nrOfIn[cur];      // the forward field of the current frame points into
    ei.referenceFrame = nr.nrOfIn[prev]; // the previous one: current -> previous, as DLSS 5 reads it
    ei.disableTemporalHints = NV_OF_TRUE;
    NV_OF_EXECUTE_OUTPUT_PARAMS eo{};
    eo.outputBuffer = nr.nrOfOut;
    const NV_OF_STATUS s = g_nvofApi.nvOFExecute(nr.nrOfH, &ei, &eo);
    if (s != NV_OF_SUCCESS)
    {
        LOG("native: DLSS 5 motion: nvOFExecute failed (status %d)\n", (int)s);
        nr.die("DLSS 5 motion: execute failed");
        return false;
    }
    int grid = kNvofGrid, vp = (int)nr.nrOfOutPitch, cp = 0;
    CUdeviceptr v = nr.nrOfOutP, c = 0;
    float* flow = nr.dNrFlow;
    float* cost = nullptr; // flow only
    // SMV_NR_MV_MARGIN=F: the validation's margin for an A/B (a large negative value keeps every Optical Flow vector)
    static const float marginSet = [] {
        const std::string m = lkEnv("SMV_NR_MV_MARGIN");
        return m.empty() ? kNrMvMargin : (float)atof(m.c_str());
    }();
    float margin = marginSet;
    // the relative keep (k_nrMv): a window that changed by more than kNrMvReal 8-bit levels a pixel keeps a vector that
    // removes at least kNrMvShare of that change; SMV_NR_MV_REAL=F / SMV_NR_MV_SHARE=F for an A/B (SHARE 0 = off: the
    // absolute test alone, the earlier field)
    static const float realSet = [] {
        const std::string m = lkEnv("SMV_NR_MV_REAL");
        return 25.0f * (m.empty() ? kNrMvReal : (float)atof(m.c_str()));
    }();
    static const float shareSet = [] {
        const std::string m = lkEnv("SMV_NR_MV_SHARE");
        return m.empty() ? kNrMvShare : (float)atof(m.c_str());
    }();
    // a window neither test decides keeps a vector of kNrMvFlat px or more (k_nrMv); SMV_NR_MV_FLAT=F for an A/B (0 =
    // off: such a window gets zero)
    static const float flatSet = [] {
        const std::string m = lkEnv("SMV_NR_MV_FLAT");
        return m.empty() ? kNrMvFlat : (float)atof(m.c_str());
    }();
    // ... when the field 8 px away agrees within kNrMvCoh px (k_nrMv); SMV_NR_MV_COH=F for an A/B (0 = no agreement test)
    static const float cohSet = [] {
        const std::string m = lkEnv("SMV_NR_MV_COH");
        return m.empty() ? kNrMvCoh : (float)atof(m.c_str());
    }();
    float real = realSet, share = shareSet, flat = flatSet, coh = cohSet;
    void* b[] = {&v, &vp, &c, &cp, &nr.nrOfGw, &nr.nrOfGh, &grid, &w, &h, &flow, &cost};
    void* m[] = {&flow, &lc, &lp, &lpitch, &w, &h, &margin, &real, &share, &flat, &coh, &nr.sfNrMv};
    if (cuLaunchKernel(nr.fNvofUp, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, b, nullptr) !=
            CUDA_SUCCESS ||
        cuLaunchKernel(nr.fNrMv, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, m, nullptr) !=
            CUDA_SUCCESS)
    {
        nr.die("DLSS 5 motion: field launch failed");
        return false;
    }
    return true;
}
