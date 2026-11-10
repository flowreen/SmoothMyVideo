// part of smv-live-native.inl (included there, in place); not a standalone translation unit
// ---- AMD FSR / Intel XeSS frame generation (fsrfg) ----------------------------------------------
// One C API, two bridges (fgKind): smv_fsrfg_bridge.dll (source engine\fsrfg\build_src, exports fsrfg_*; SMV_FSRFG_DIR,
// else engine\fsrfg) and smv_xefg_bridge.dll (source engine\xess\build_src, exports xefg_*; SMV_XESS_DIR, else
// engine\xess, Intel's libxess_fg.dll + libxell.dll beside it). Each loads once per process by full path from the folder
// the handoff named; up to 16 instances, each its own context: FSR's on vkd3d-proton's device, XeSS-FG's on the
// adapter's D3D12 device (with its own swap chain and parked window), both for the CUDA device's adapter
struct FsrfgShared
{
    HANDLE in, mv, out, fence;           // three textures and a fence: Vulkan opaque Win32 exports and a timeline
                                         // semaphore (FSR), D3D12 resources and a D3D12 fence (XeSS-FG)
    uint64_t inBytes, mvBytes, outBytes; // the textures' allocation sizes
    uint32_t rowPitch, mvRowPitch;       // the linear rows a caller keeps its frames / vectors in
};
struct FsrfgBridge
{
    HMODULE mod = nullptr;
    const char* (*lastError)() = nullptr;
    void (*setCacheDir)(const char*) = nullptr;
    const char* (*provider)() = nullptr;
    int (*createI)(int, uint32_t, uint32_t, uint32_t, int32_t, int, FsrfgShared*) = nullptr;
    int (*frameI)(int, uint64_t, uint64_t, int, double) = nullptr;
    void (*destroy)() = nullptr;
    int (*devices)() = nullptr; // 1 = an own vkd3d device for every instance (optional; an older bridge has none)
    // XeSS-FG's counters of instance i: calls, calls with no frame presented, presents with no call, frames out
    int (*stats)(int, uint64_t*, uint64_t*, uint64_t*, uint64_t*) = nullptr; // optional (the XeSS-FG bridge's)
};
static FsrfgBridge g_fsrfgK[2];
static const char* const kFgDll[2] = {"smv_fsrfg_bridge.dll", "smv_xefg_bridge.dll"};
static const char* const kFgPrefix[2] = {"fsrfg_", "xefg_"};

// the bridge of this kind; FSR's vkd3d-proton shader cache in cacheDir\fsr4_shaders (the model cache: users may delete
// it); FSR's instances share vkd3d's one device (SMV_FSRFG_DEVICE=own = an own device each, ~235 MiB + ~0.3 s an
// instance)
static bool nativeFsrfgLoad(const std::string& dir, const std::string& cacheDir, int kind)
{
    FsrfgBridge& b = g_fsrfgK[kind];
    if (b.mod)
        return true;
    const std::wstring p = utf8ToWide(dir) + L"\\" + utf8ToWide(kFgDll[kind]);
    HMODULE m = LoadLibraryExW(p.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!m)
    {
        LOG("native: fsrfg: %s did not load from %s (err %lu)\n", kFgDll[kind], dir.c_str(), GetLastError());
        return false;
    }
    const std::string pre = kFgPrefix[kind];
    auto fn = [&](const char* name) { return GetProcAddress(m, (pre + name).c_str()); };
    b.lastError = (const char* (*)())fn("last_error");
    b.setCacheDir = (void (*)(const char*))fn("set_cache_dir");
    b.provider = (const char* (*)())fn("provider");
    b.createI = (int (*)(int, uint32_t, uint32_t, uint32_t, int32_t, int, FsrfgShared*))fn("create_i");
    b.frameI = (int (*)(int, uint64_t, uint64_t, int, double))fn("frame_i");
    b.destroy = (void (*)())fn("destroy");
    b.devices = (int (*)())fn("devices");
    b.stats = (int (*)(int, uint64_t*, uint64_t*, uint64_t*, uint64_t*))fn("stats");
    if (!b.lastError || !b.setCacheDir || !b.provider || !b.createI || !b.frameI || !b.destroy)
    {
        LOG("native: fsrfg: %s lacks an expected export\n", kFgDll[kind]);
        FreeLibrary(m);
        b = FsrfgBridge{};
        return false;
    }
    if (!cacheDir.empty())
        b.setCacheDir((cacheDir + "\\fsr4_shaders").c_str());
    b.mod = m;
    return true;
}

// instance i at w x h (RGBA16F) with its shared textures (as CUDA arrays) and fence imported into this CUDA context
static bool nativeFsrfgCreate(NativeRife& nr, int i)
{
    NativeRife::FgInst& g = nr.fg[i];
    FsrfgShared sh{};
    const FsrfgBridge& br = g_fsrfgK[nr.fgKind];
    const int rc = br.createI(i, (uint32_t)nr.w, (uint32_t)nr.h, nr.fgLuid.LowPart, nr.fgLuid.HighPart, 1, &sh);
    if (rc != 0)
    {
        LOG("native: fsrfg: instance %d at %dx%d not created: %s (rc %d)\n", i, nr.w, nr.h, br.lastError(), rc);
        return false;
    }
    g.made = true;
    if (nr.fgPitch && ((int)sh.rowPitch != nr.fgPitch || (int)sh.mvRowPitch != nr.fgMvPitch))
    {
        LOG("native: fsrfg: instance %d's row pitches %u / %u differ from instance 0's %d / %d\n", i, sh.rowPitch,
            sh.mvRowPitch, nr.fgPitch, nr.fgMvPitch);
        return false;
    }
    nr.fgPitch = (int)sh.rowPitch;
    nr.fgMvPitch = (int)sh.mvRowPitch;
    const int w = nr.w, h = nr.h;
    const bool d3d = nr.fgKind == 1;
    auto imp = [w, h, d3d](HANDLE hd, uint64_t bytes, const cudaChannelFormatDesc& fd, cudaExternalMemory_t& em,
                           cudaMipmappedArray_t& mm, cudaArray_t& a) {
        cudaExternalMemoryHandleDesc md{};
        md.type = d3d ? cudaExternalMemoryHandleTypeD3D12Resource : cudaExternalMemoryHandleTypeOpaqueWin32;
        md.size = bytes;
        md.flags = cudaExternalMemoryDedicated;
        md.handle.win32.handle = hd;
        if (cudaImportExternalMemory(&em, &md) != cudaSuccess)
        {
            em = nullptr;
            return false;
        }
        cudaExternalMemoryMipmappedArrayDesc ad{};
        ad.formatDesc = fd;
        ad.extent.width = (size_t)w;
        ad.extent.height = (size_t)h;
        ad.numLevels = 1;
        return cudaExternalMemoryGetMappedMipmappedArray(&mm, em, &ad) == cudaSuccess &&
               cudaGetMipmappedArrayLevel(&a, mm, 0) == cudaSuccess;
    };
    const cudaChannelFormatDesc rgba = cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindFloat);
    const cudaChannelFormatDesc rg = cudaCreateChannelDesc(16, 16, 0, 0, cudaChannelFormatKindFloat);
    cudaExternalSemaphoreHandleDesc sd{};
    sd.type = d3d ? cudaExternalSemaphoreHandleTypeD3D12Fence : cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
    sd.handle.win32.handle = sh.fence;
    if (!imp(sh.in, sh.inBytes, rgba, g.emIn, g.mmIn, g.in) || !imp(sh.mv, sh.mvBytes, rg, g.emMv, g.mmMv, g.mv) ||
        !imp(sh.out, sh.outBytes, rgba, g.emOut, g.mmOut, g.out) ||
        cudaImportExternalSemaphore(&g.sem, &sd) != cudaSuccess)
    {
        LOG("native: fsrfg: instance %d's shared textures or fence did not import into CUDA\n", i);
        return false;
    }
    if (!nr.dFgMv && cudaMalloc((void**)&nr.dFgMv, (size_t)nr.fgMvPitch * nr.h) != cudaSuccess)
    {
        nr.dFgMv = nullptr;
        LOG("native: fsrfg: the vector buffer did not allocate\n");
        return false;
    }
    g.val = 0;
    nr.frLastKey[i] = 0;
    nr.frCalls[i] = 0;
    return true;
}

// an OfVec session at w x h (FSR's vectors: ABGR8, one direction, grid 1 = the measurement harness's vectors, or grid 4
// with k_nvofUp's bilinear taps where the engine refuses grid 1) at nativeOfLevel(); tag names
// it in the log lines; st = the stream it runs on (null = nr.stream)
static bool nativeOfVecSetup(NativeRife& nr, NativeRife::OfVec& o, int w, int h, const char* tag,
                             cudaStream_t st = nullptr)
{
    if (!nativeNvofLoad())
        return false;
    if (w < 32 || h < 32)
    {
        LOG("native: %s: the frame %dx%d is below the Optical Flow minimum 32x32\n", tag, w, h);
        return false;
    }
    CUcontext ctx = nullptr;
    if (cuCtxGetCurrent(&ctx) != CUDA_SUCCESS || !ctx)
    {
        LOG("native: %s: no current CUDA context\n", tag);
        return false;
    }
    o.w = w;
    o.h = h;
    o.st = st;
    {
        char ev[8] = {};
        o.both = GetEnvironmentVariableA("SMV_OFVEC_BOTH", ev, sizeof(ev)) > 0 && ev[0] == '1';
    }
    o.level = nativeOfLevel();
    for (int grid : {1, 4})
    {
        if (g_nvofApi.nvCreateOpticalFlowCuda(ctx, &o.of) != NV_OF_SUCCESS)
        {
            o.of = nullptr;
            LOG("native: %s: nvCreateOpticalFlowCuda failed\n", tag);
            return false;
        }
        NV_OF_INIT_PARAMS ip{};
        ip.width = (uint32_t)w;
        ip.height = (uint32_t)h;
        ip.outGridSize = (NV_OF_OUTPUT_VECTOR_GRID_SIZE)grid;
        ip.hintGridSize = NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED;
        ip.mode = NV_OF_MODE_OPTICALFLOW;
        ip.perfLevel = o.level;
        ip.enableExternalHints = NV_OF_FALSE;
        // FSR reads no cost: off
        ip.enableOutputCost = NV_OF_FALSE;
        ip.disparityRange = NV_OF_STEREO_DISPARITY_RANGE_UNDEFINED;
        ip.enableRoi = NV_OF_FALSE;
        // FSR reads one field, current -> previous: the forward field with the two frames swapped (half the work of
        // both directions); SMV_OFVEC_BOTH=1 = both directions, the backward one read
        ip.predDirection = o.both ? NV_OF_PRED_DIRECTION_BOTH : NV_OF_PRED_DIRECTION_FORWARD;
        ip.enableGlobalFlow = NV_OF_FALSE;
        ip.inputBufferFormat = NV_OF_BUFFER_FORMAT_ABGR8;
        if (g_nvofApi.nvOFInit(o.of, &ip) == NV_OF_SUCCESS)
        {
            o.grid = grid;
            break;
        }
        g_nvofApi.nvOFDestroy(o.of);
        o.of = nullptr;
    }
    if (!o.of)
    {
        LOG("native: %s: the Optical Flow engine refused %dx%d at grid 1 and 4\n", tag, w, h);
        return false;
    }
    if (o.grid != 1)
        LOG("native: %s: the Optical Flow engine refused grid 1, vectors at grid 4 (bilinear)\n", tag);
    o.gw = (w + o.grid - 1) / o.grid;
    o.gh = (h + o.grid - 1) / o.grid;
    for (int k = 0; k < 2; k++)
    {
        uint32_t inPitch = 0;
        if (!nativeNvofBufOn(o.of, w, h, NV_OF_BUFFER_USAGE_INPUT, NV_OF_BUFFER_FORMAT_ABGR8, o.in[k], o.inP[k],
                             inPitch, "FSR vector input") ||
            !nativeNvofBufOn(o.of, o.gw, o.gh, NV_OF_BUFFER_USAGE_OUTPUT, NV_OF_BUFFER_FORMAT_SHORT2, o.out[k],
                             o.outP[k], o.outPitch[k], "FSR vector flow"))
            return false;
        if (k == 0)
            o.inPitch = inPitch;
        else if (inPitch != o.inPitch)
        {
            LOG("native: %s: the two Optical Flow inputs differ in pitch\n", tag);
            return false;
        }
    }
    const CUstream ios = (CUstream)(st ? st : nr.stream);
    if (g_nvofApi.nvOFSetIOCudaStreams(o.of, ios, ios) != NV_OF_SUCCESS)
    {
        LOG("native: %s: nvOFSetIOCudaStreams failed\n", tag);
        return false;
    }
    if (o.grid != 1) // grid 1 writes the texels straight from the field (k_mvHalfG1)
        NCHK(cudaMalloc((void**)&o.dFlow, 2 * (size_t)w * h * sizeof(float)), "alloc FSR vectors");
    return true;
}

// the vectors between slots `prev` and `cur`, both holding their ABGR8 frames: the field cur -> prev in px (the backward
// field of the pair), into dst as R16G16_FLOAT texels, rows dstPitch bytes apart
static bool nativeOfVecRun(NativeRife& nr, NativeRife::OfVec& o, int prev, int cur, uint8_t* dst, int dstPitch,
                           const char* tag, cudaEvent_t evOf = nullptr, float gate = 0.0f)
{
    cudaStream_t st = o.st ? o.st : nr.stream;
    NV_OF_EXECUTE_INPUT_PARAMS ei{};
    NV_OF_EXECUTE_OUTPUT_PARAMS eo{};
    ei.disableTemporalHints = NV_OF_TRUE;
    if (o.both)
    {
        ei.inputFrame = o.in[prev];
        ei.referenceFrame = o.in[cur];
        eo.outputBuffer = o.out[0];
        eo.bwdOutputBuffer = o.out[1];
    }
    else
    {
        // the forward field from `cur` to `prev` = the backward one of (prev, cur), into the buffers it is read from
        ei.inputFrame = o.in[cur];
        ei.referenceFrame = o.in[prev];
        eo.outputBuffer = o.out[o.wr];
    }
    const NV_OF_STATUS s = g_nvofApi.nvOFExecute(o.of, &ei, &eo);
    if (s != NV_OF_SUCCESS)
    {
        LOG("native: %s: nvOFExecute failed (status %d)\n", tag, (int)s);
        nr.die("FSR optical flow failed");
        return false;
    }
    if (evOf) // a profiler's mark between the Optical Flow and the vector kernels (the session's stream)
        cudaEventRecord(evOf, st);
    const unsigned gx = (unsigned)(o.w + 15) / 16, gy = (unsigned)(o.h + 15) / 16;
    const int f = o.both ? 1 : o.wr;
    int grid = o.grid, vp = (int)o.outPitch[f], cp = 0, mp = dstPitch;
    CUdeviceptr v = o.outP[f], c = 0;
    if (grid == 1) // the field is the texels' grid: one pass, no float planes in between
    {
        void* g1[] = {&v, &vp, &o.w, &o.h, &dst, &mp, &gate};
        if (cuLaunchKernel(nr.fMvHalfG1, gx, gy, 1, 16, 16, 1, 0, (CUstream)st, g1, nullptr) != CUDA_SUCCESS)
        {
            nr.die("FSR vector launch failed");
            return false;
        }
        return true;
    }
    float* flow = o.dFlow;
    float* cost = nullptr; // k_nvofUp: the flow alone, the cost never read
    void* b[] = {&v, &vp, &c, &cp, &o.gw, &o.gh, &grid, &o.w, &o.h, &flow, &cost};
    void* m[] = {&flow, &o.w, &o.h, &dst, &mp, &gate};
    if (cuLaunchKernel(nr.fNvofUp, gx, gy, 1, 16, 16, 1, 0, (CUstream)st, b, nullptr) != CUDA_SUCCESS ||
        cuLaunchKernel(nr.fMvHalf, gx, gy, 1, 16, 16, 1, 0, (CUstream)st, m, nullptr) != CUDA_SUCCESS)
    {
        nr.die("FSR vector launch failed");
        return false;
    }
    return true;
}

static void nativeOfVecFree(NativeRife::OfVec& o)
{
    for (NvOFGPUBufferHandle* b : {&o.in[0], &o.in[1], &o.out[0], &o.out[1]})
        if (*b)
            g_nvofApi.nvOFDestroyGPUBufferCuda(*b);
    if (o.of)
        g_nvofApi.nvOFDestroy(o.of);
    if (o.dFlow)
        cudaFree(o.dFlow);
    o = NativeRife::OfVec{};
}

// prev and frame (RGBA16F) into OfVec o's two inputs as ABGR8, on o's stream; the slots they are in come back in
// slotPrev / slotCur. chain = the level-1 pass, whose prev is the last level-1 pass's frame: when a slot still holds
// that surface (keepSrc), only the new frame is converted, into the other slot
static bool nativeFsrfgOfIn(NativeRife& nr, NativeRife::OfVec& o, const uint8_t* prev, const uint8_t* frame, bool chain,
                            int& slotPrev, int& slotCur)
{
    const unsigned gx = (unsigned)(nr.w + 15) / 16, gy = (unsigned)(nr.h + 15) / 16;
    int sp = nr.fgPitch, ip = (int)o.inPitch;
    const bool kept = chain && o.keepSlot >= 0 && o.keepSrc == prev;
    slotPrev = kept ? o.keepSlot : 0;
    slotCur = 1 - slotPrev;
    for (int k = kept ? 1 : 0; k < 2; k++)
    {
        const uint8_t* s = k ? frame : prev;
        CUdeviceptr d = o.inP[k ? slotCur : slotPrev];
        void* a[] = {(void*)&s, &sp, &nr.w, &nr.h, &d, &ip};
        if (cuLaunchKernel(nr.fRgbaHAbgr, gx, gy, 1, 16, 16, 1, 0, (CUstream)(o.st ? o.st : nr.stream), a, nullptr) !=
            CUDA_SUCCESS)
        {
            nr.die("rgbaHAbgr (fsrfg) launch failed");
            return false;
        }
    }
    o.keepSlot = chain ? slotCur : -1;
    o.keepSrc = chain ? frame : nullptr;
    return true;
}

// AMD FSR 4's level-2+ calls take vectors derived from their pair's level-1 field (k_mvDerive) instead of an optical-flow
// pass each: one SLOW pass a call bounds FSR 4 on the optical-flow engine (72-86 % busy, the SMs 17-22 %), and on real
// 1000 fps footage the derived tweens score closer to the true frames. The field must be grid 1
// (an engine that refuses grid 1 keeps a pass a call). On by default, every instance on the one shared vkd3d device
// (calls of different instances back to back stay exact because our vkd3d does not offer the coopmat2 conversions);
// SMV_FSRFG_CHILDMV=0 = a pass a call
static bool fsrfgChildEnvOn()
{
    static const bool on = [] {
        char v[8] = "";
        return !(GetEnvironmentVariableA("SMV_FSRFG_CHILDMV", v, sizeof(v)) > 0 && v[0] == '0');
    }();
    return on;
}

static bool fsrfgDeriveOn(const NativeRife& nr)
{
    return fsrfgChildEnvOn() && nr.fgOf.grid == 1;
}

// a level-2+ call's derived vectors (its frames at t0 and t1 of the pair) into dst (rows fgMvPitch apart) on stream st,
// from fgOf's level-1 output: no level-2+ call runs fgOf once its vectors are derived, so the field holds for the pair
static bool nativeFsrfgChildVec(NativeRife& nr, uint8_t* dst, cudaStream_t st, float t0, float t1)
{
    CUdeviceptr v = nr.fgOf.outP[nr.fgOfRd];
    int vp = (int)nr.fgOf.outPitch[nr.fgOfRd], mp = nr.fgMvPitch;
    void* a[] = {&v, &vp, &nr.w, &nr.h, &t0, &t1, &dst, &mp};
    if (cuLaunchKernel(nr.fMvDerive, (unsigned)(nr.w + 15) / 16, (unsigned)(nr.h + 15) / 16, 1, 16, 16, 1, 0,
                       (CUstream)st, a, nullptr) != CUDA_SUCCESS)
    {
        nr.die("fsrfg derived vector launch failed");
        return false;
    }
    return true;
}

// one bridge call: instance i gets `frame` (RGBA16F rows fgPitch apart) with its vectors toward `prev` (the frame the
// instance was fed last; null = zero vectors: a reset, or a re-prime nobody reads) and returns into `out` (null =
// dropped) the frame halfway between them. The vectors as the harness made them: the field this frame -> the previous
// (px) = FSR's convention. ring >= 0: the vectors were made ahead into dFgMvR[ring] (nativeFsrfgAheadVec). l1 = the
// group's level-1 call: its FSR span goes into fgL1Ev for the x2 call price (nativeGroup). part 0 = the whole call on
// the session's stream; on fgL1St (the pipelined route's instance-0 calls, their vectors in dFgMvL1): 1 = the level-1
// Optical Flow on fgOfL1St, then the frame and its vectors into the instance's textures, 2 = after fgEvDeep the call
// and its copy out, 3 = the whole call (a reset or a re-prime, in a group that runs no deeper call)
static bool nativeFsrfgFeed(NativeRife& nr, int i, const uint8_t* frame, const uint8_t* prev, bool reset, uint8_t* out,
                            int ring = -1, bool l1 = false, int part = 0)
{
    NativeRife::FgInst& g = nr.fg[i];
    cudaStream_t st = part ? nr.fgL1St : nr.stream;
    uint8_t* mvBuf = part ? nr.dFgMvL1 : nr.dFgMv;
    const size_t row = (size_t)nr.w * 8;
    auto mark = [&nr, st](int e) {
        if (nr.fgProf)
            cudaEventRecord(nr.fgEv[e], st);
    };
    bool derived = false;
    if (part != 2)
    {
        // the pipelined level-1 pass (part 1) starts from the pack, not behind the instance's last call on fgL1St:
        // the frame goes into the instance's texture after the pass
        const bool pass1 = part == 1;
        mark(0);
        if (!pass1 && cudaMemcpy2DToArrayAsync(g.in, 0, 0, frame, nr.fgPitch, row, nr.h, cudaMemcpyDeviceToDevice,
                                               st) != cudaSuccess)
        {
            nr.die("fsrfg frame copy failed");
            return false;
        }
        mark(1);
        const uint8_t* mv = ring >= 0 ? nr.dFgMvR[ring] : mvBuf;
        if (ring >= 0)
        {
            if (cudaStreamWaitEvent(st, nr.fgEvVec[ring], 0) != cudaSuccess)
            {
                nr.die("fsrfg vector wait failed");
                return false;
            }
        }
        else if (prev && (i == 0 || !fsrfgDeriveOn(nr)))
        {
            // fgOf runs on fgOfL1St: from another stream the pass waits for that stream's work so far (the pipelined
            // pass: for the pack, nativeFsrfgPipeAdvance, and for fgL1St's last read of dFgMvL1), and it for the pass
            const bool hop = nr.fgOf.st && nr.fgOf.st != st;
            if (hop && (pass1 ? cudaStreamWaitEvent(nr.fgOf.st, nr.fgEvMv, 0) != cudaSuccess
                              : (cudaEventRecord(nr.fgEvS, st) != cudaSuccess ||
                                 cudaStreamWaitEvent(nr.fgOf.st, nr.fgEvS, 0) != cudaSuccess)))
            {
                nr.die("fsrfg vector event failed");
                return false;
            }
            // the level-1 pass's span for the pipelined route's price (nativeGroup)
            cudaStream_t os = nr.fgOf.st ? nr.fgOf.st : nr.stream;
            const int es = nr.fgEvSet;
            if (l1 && nr.fgOfEv[es][0])
                nr.grOfArmed[es] = cudaEventRecord(nr.fgOfEv[es][0], os) == cudaSuccess;
            int slotPrev = 0, slotCur = 1;
            if (!nativeFsrfgOfIn(nr, nr.fgOf, prev, frame, i == 0, slotPrev, slotCur))
                return false;
            mark(2);
            if (!nativeOfVecRun(nr, nr.fgOf, slotPrev, slotCur, mvBuf, nr.fgMvPitch, "fsrfg",
                                nr.fgProf ? nr.fgEv[3] : nullptr))
                return false;
            if (l1 && nr.grOfArmed[es])
                nr.grOfArmed[es] = cudaEventRecord(nr.fgOfEv[es][1], os) == cudaSuccess;
            if (hop && (cudaEventRecord(nr.fgEvOf, nr.fgOf.st) != cudaSuccess ||
                        cudaStreamWaitEvent(st, nr.fgEvOf, 0) != cudaSuccess))
            {
                nr.die("fsrfg vector event failed");
                return false;
            }
        }
        else if (prev)
        {
            derived = true;
            mark(2);
            if (!nativeFsrfgChildVec(nr, mvBuf, st, nr.fgChildT0, nr.fgChildT1))
                return false;
            mark(3);
        }
        else if (cudaMemset2DAsync(mvBuf, nr.fgMvPitch, 0, (size_t)nr.w * 4, nr.h, st) != cudaSuccess)
        {
            nr.die("fsrfg vector clear failed");
            return false;
        }
        else
        {
            mark(2);
            mark(3);
        }
        if (pass1 && cudaMemcpy2DToArrayAsync(g.in, 0, 0, frame, nr.fgPitch, row, nr.h, cudaMemcpyDeviceToDevice, st) !=
                         cudaSuccess)
        {
            nr.die("fsrfg frame copy failed");
            return false;
        }
        mark(4);
        // fgEvMv: dFgMvL1 is read, the next level-1 pass may write it
        if (cudaMemcpy2DToArrayAsync(g.mv, 0, 0, mv, nr.fgMvPitch, (size_t)nr.w * 4, nr.h, cudaMemcpyDeviceToDevice,
                                     st) != cudaSuccess ||
            (ring >= 0 && cudaEventRecord(nr.fgEvFree[ring], st) != cudaSuccess) ||
            (part && cudaEventRecord(nr.fgEvMv, st) != cudaSuccess))
        {
            nr.die("fsrfg vector copy failed");
            return false;
        }
        mark(5);
        if (part == 1)
            return true;
    }
    else if (cudaStreamWaitEvent(st, nr.fgEvDeep, 0) != cudaSuccess)
    {
        nr.die("fsrfg call wait failed");
        return false;
    }
    const int es = nr.fgEvSet;
    if (l1 && nr.fgL1Ev[es][0])
        nr.grL1Armed[es] = cudaEventRecord(nr.fgL1Ev[es][0], st) == cudaSuccess;
    cudaExternalSemaphoreSignalParams sp{};
    sp.params.fence.value = ++g.val;
    if (cudaSignalExternalSemaphoresAsync(&g.sem, &sp, 1, st) != cudaSuccess)
    {
        nr.die("fsrfg fence signal failed");
        return false;
    }
    const uint64_t ready = g.val, done = ++g.val;
    const int64_t c0 = nr.fgProf ? nowQpc100() : 0;
    // the frame time changes neither FSR 4's output (a 24 and a 60 fps run byte for byte the same) nor XeSS-FG's (33 and
    // 0 ms the same): one 60 Hz frame
    const FsrfgBridge& br = g_fsrfgK[nr.fgKind];
    const int rc = br.frameI(i, ready, done, reset ? 1 : 0, 1000.0 / 60.0);
    if (nr.fgProf)
        nr.fgCpuMs += (double)(nowQpc100() - c0) / 10000.0;
    if (rc != 0)
    {
        LOG("native: fsrfg: instance %d call failed: %s (rc %d)\n", i, br.lastError(), rc);
        nr.die("fsrfg call failed");
        return false;
    }
    cudaExternalSemaphoreWaitParams wp{};
    wp.params.fence.value = done;
    if (cudaWaitExternalSemaphoresAsync(&g.sem, &wp, 1, st) != cudaSuccess)
    {
        nr.die("fsrfg result failed");
        return false;
    }
    mark(6);
    if (l1 && nr.grL1Armed[es])
        nr.grL1Armed[es] = cudaEventRecord(nr.fgL1Ev[es][1], st) == cudaSuccess;
    if (out && cudaMemcpy2DFromArrayAsync(out, nr.fgPitch, g.out, 0, 0, row, nr.h, cudaMemcpyDeviceToDevice, st) !=
                   cudaSuccess)
    {
        nr.die("fsrfg result failed");
        return false;
    }
    mark(7);
    if (nr.fgProf && cudaEventSynchronize(nr.fgEv[7]) == cudaSuccess)
    {
        for (int s = 0; s < 7; s++)
        {
            float ms = 0.0f;
            if (cudaEventElapsedTime(&ms, nr.fgEv[s], nr.fgEv[s + 1]) == cudaSuccess)
                (derived && s >= 1 && s <= 3 ? nr.fgDeriveMs : nr.fgStageMs[s]) += ms;
        }
        nr.fgProfN++;
        nr.fgProfVecN += prev && !derived ? 1 : 0;
        nr.fgProfDerN += derived ? 1 : 0;
    }
    nr.frCalls[i]++;
    return true;
}

// the extra instances up to `want` (instance 0 is the session's own); a failed create keeps the ones made so far
static void nativeFsrfgGrow(NativeRife& nr, int want)
{
    want = (std::min)(want, kFrDepth);
    const int64_t t0 = nowQpc100();
    const int had = nr.frInst;
    for (int i = nr.frInst; i < want; i++)
    {
        if (!nativeFsrfgCreate(nr, i))
        {
            LOG("native: fsrfg: staying at %d instance%s\n", nr.frInst, nr.frInst > 1 ? "s" : "");
            nr.frInstMax = nr.frInst;
            break;
        }
        nr.frInst = i + 1;
    }
    if (nr.frInst > had)
    {
        nr.grPriceSkip = 2;
        LOG("native: fsrfg: %d instances (%d new in %.0f ms)\n", nr.frInst, nr.frInst - had,
            (double)(nowQpc100() - t0) / 10000.0);
    }
}

// node k / 2^L of this pair's midpoint tree (FRUC's nativeFrucNode): FSR on instance L - 1 fed the right parent
// after the left one. An instance fed another frame last (a pair of another depth, a new encoding) gets the left
// parent first, its output dropped: a reset only for its very first frame (after a reset FSR's next 10 tweens follow
// the vectors alone, ffx_frameinterpolation.h fFrameIndexFactor), else a plain call. ring >= 0: the node's vectors were
// made ahead into dFgMvR[ring] (nativeFsrfgLevel). The tree = the pair shown: this group's, or the previous one while
// pipelined (its nodes in dFrNode either way, nativeFsrfgPipeAdvance)
static FrTree nativeFsrfgShown(NativeRife& nr)
{
    return nr.fgPiped ? FrTree{nr.frLagSerial, nr.frLagA, nr.frLagB, nr.dFrNode, nr.frNodeOk} : nativeFrucCur(nr);
}

static uint8_t* nativeFsrfgNode(NativeRife& nr, int k, int L, int ring = -1)
{
    while (L > 0 && !(k & 1))
    {
        k >>= 1;
        L--;
    }
    const FrTree tr = nativeFsrfgShown(nr);
    if (L <= 0)
        return nr.dFrSurf[k ? tr.b : tr.a];
    const int p = k << (kFrDepth - L);
    if (nr.frNodeOk[p])
        return nr.dFrNode[p];
    uint8_t* left = nativeFsrfgNode(nr, k - 1, L);
    uint8_t* right = left ? nativeFsrfgNode(nr, k + 1, L) : nullptr;
    if (!right)
        return nullptr;
    if (!nr.dFrNode[p] && cudaMalloc((void**)&nr.dFrNode[p], (size_t)nr.fgPitch * nr.h) != cudaSuccess)
    {
        nr.dFrNode[p] = nullptr;
        nr.die("fsrfg midpoint buffer alloc failed");
        return nullptr;
    }
    const int i = L - 1;
    const uint64_t lk = nativeFrucKey(tr, k - 1, L);
    if (nr.frLastKey[i] != lk)
    {
        if (!nativeFsrfgFeed(nr, i, left, nullptr, nr.frLastKey[i] == 0, nullptr))
            return nullptr;
        if (nr.frLastKey[i] != 0)
            nr.frPrimed++;
        nr.frLastKey[i] = lk;
    }
    const int64_t c0 = nowQpc100();
    nr.fgChildT0 = (float)(k - 1) / (float)(1 << L);
    nr.fgChildT1 = (float)(k + 1) / (float)(1 << L);
    if (!nativeFsrfgFeed(nr, i, right, left, false, nr.dFrNode[p], ring, i == 0))
        return nullptr;
    nr.grFrCalls++;
    nr.grFrCallMs += (double)(nowQpc100() - c0) / 10000.0;
    nr.frLastKey[i] = nativeFrucKey(tr, k + 1, L);
    nr.frNodeCalls++;
    nr.frNodeOk[p] = true;
    return nr.dFrNode[p];
}

static void nativeFsrfgAheadFree(NativeRife& nr)
{
    if (nr.fgOfSt)
        cudaStreamSynchronize(nr.fgOfSt);
    nativeOfVecFree(nr.fgOf2);
    for (int s = 0; s < 2; s++)
    {
        if (nr.dFgMvR[s])
            cudaFree(nr.dFgMvR[s]);
        nr.dFgMvR[s] = nullptr;
        for (cudaEvent_t* e : {&nr.fgEvVec[s], &nr.fgEvFree[s]})
            if (*e)
            {
                cudaEventDestroy(*e);
                *e = nullptr;
            }
    }
    if (nr.fgEvLevel)
        cudaEventDestroy(nr.fgEvLevel);
    nr.fgEvLevel = nullptr;
    if (nr.fgOfSt)
        cudaStreamDestroy(nr.fgOfSt);
    nr.fgOfSt = nullptr;
    nr.fgAhead = 0;
}

// the side stream, its Optical Flow session and the two vector buffers, made at the session's first level with 2+
// calls; false = serial vectors (SMV_FSRFG_OVERLAP=0, the profiler, or a failed setup, which says so;
// SMV_FSRFG_OVERLAP=fail = a failed setup after a whole one, the fallback's trigger)
static bool nativeFsrfgAhead(NativeRife& nr)
{
    if (nr.fgAhead)
        return nr.fgAhead > 0;
    char ev[8] = {};
    if (nr.fgProf || (GetEnvironmentVariableA("SMV_FSRFG_OVERLAP", ev, sizeof(ev)) > 0 && ev[0] == '0'))
    {
        nr.fgAhead = -1;
        return false;
    }
    const size_t bytes = (size_t)nr.fgMvPitch * nr.h;
    bool ok = cudaStreamCreateWithFlags(&nr.fgOfSt, cudaStreamNonBlocking) == cudaSuccess &&
              cudaEventCreateWithFlags(&nr.fgEvLevel, cudaEventDisableTiming) == cudaSuccess;
    for (int s = 0; ok && s < 2; s++)
    {
        uint8_t* b = nullptr;
        ok = cudaEventCreateWithFlags(&nr.fgEvVec[s], cudaEventDisableTiming) == cudaSuccess &&
             cudaEventCreateWithFlags(&nr.fgEvFree[s], cudaEventDisableTiming) == cudaSuccess &&
             cudaMalloc((void**)&b, bytes) == cudaSuccess;
        nr.dFgMvR[s] = ok ? b : nullptr;
    }
    // derived level-2+ vectors need no side Optical Flow session (fsrfgDeriveOn)
    const bool derive = fsrfgDeriveOn(nr);
    if (!ok || (!derive && !nativeOfVecSetup(nr, nr.fgOf2, nr.w, nr.h, "fsrfg side vectors", nr.fgOfSt)) ||
        strcmp(ev, "fail") == 0)
    {
        LOG(derive ? "native: fsrfg: the side stream did not start, every call's vectors stay in its turn\n"
                   : "native: fsrfg: the side Optical Flow session did not start, every call's vectors stay in its "
                     "turn\n");
        nativeFsrfgAheadFree(nr);
        nr.fgAhead = -1;
        return false;
    }
    nr.fgAhead = 1;
    LOG(derive ? "native: fsrfg: tree levels with 2+ calls derive the next call's vectors during each FSR call (a side "
                 "stream)\n"
               : "native: fsrfg: tree levels with 2+ calls make the next call's vectors during each FSR call (a side "
                 "stream, its own Optical Flow session)\n");
    return true;
}

// a call's vectors (the field frame -> prev; its frames at t0 / t1 of the pair) on the side stream into dFgMvR[s] once
// that buffer's last copy is done, then fgEvVec[s]: derived (fsrfgDeriveOn) or the side session's optical flow. The
// copy's event is recorded before this is enqueued: a stream waits for the event's last record
static bool nativeFsrfgAheadVec(NativeRife& nr, int s, const uint8_t* prev, const uint8_t* frame, float t0, float t1)
{
    if (cudaStreamWaitEvent(nr.fgOfSt, nr.fgEvFree[s], 0) != cudaSuccess)
    {
        nr.die("fsrfg side vector wait failed");
        return false;
    }
    if (fsrfgDeriveOn(nr))
    {
        if (!nativeFsrfgChildVec(nr, nr.dFgMvR[s], nr.fgOfSt, t0, t1))
            return false;
    }
    else
    {
        int slotPrev = 0, slotCur = 1;
        if (!nativeFsrfgOfIn(nr, nr.fgOf2, prev, frame, false, slotPrev, slotCur) ||
            !nativeOfVecRun(nr, nr.fgOf2, slotPrev, slotCur, nr.dFgMvR[s], nr.fgMvPitch, "fsrfg side vectors"))
            return false;
    }
    if (cudaEventRecord(nr.fgEvVec[s], nr.fgOfSt) != cudaSuccess)
    {
        nr.die("fsrfg side vector event failed");
        return false;
    }
    return true;
}

// level l (2^(l-1) calls on instance l - 1) with its vectors made ahead: every call's two frames are coarser nodes,
// done before the level starts, so call j + 1's Optical Flow (the OFA engine) runs while call j's FSR (vkd3d's queue)
// does; each instance still gets nativeFsrfgNode's calls in the same order
static bool nativeFsrfgLevel(NativeRife& nr, int l)
{
    const int n = 1 << (l - 1);
    uint8_t* lf[kFrGrid / 2] = {};
    uint8_t* rt[kFrGrid / 2] = {};
    for (int j = 0; j < n; j++)
        if (!(lf[j] = nativeFsrfgNode(nr, 2 * j, l)) || !(rt[j] = nativeFsrfgNode(nr, 2 * j + 2, l)))
            return false;
    if (cudaEventRecord(nr.fgEvLevel, nr.stream) != cudaSuccess ||
        cudaStreamWaitEvent(nr.fgOfSt, nr.fgEvLevel, 0) != cudaSuccess)
    {
        nr.die("fsrfg level event failed");
        return false;
    }
    const float den = (float)(1 << l); // call j's frames = nodes 2 j and 2 j + 2 of 2^l
    if (!nativeFsrfgAheadVec(nr, 0, lf[0], rt[0], 0.0f, 2.0f / den))
        return false;
    for (int j = 0; j < n; j++)
    {
        if (j + 1 < n && !nativeFsrfgAheadVec(nr, (j + 1) & 1, lf[j + 1], rt[j + 1], (2.0f * j + 2.0f) / den,
                                              (2.0f * j + 4.0f) / den))
            return false;
        if (!nativeFsrfgNode(nr, 2 * j + 1, l, j & 1))
            return false;
    }
    // the stream waits for the side stream's last work too (a node made earlier leaves its vectors unread), so the
    // next pair's frames never overwrite a node the side stream still reads
    if (cudaEventRecord(nr.fgEvLevel, nr.fgOfSt) != cudaSuccess ||
        cudaStreamWaitEvent(nr.stream, nr.fgEvLevel, 0) != cudaSuccess)
    {
        nr.die("fsrfg level event failed");
        return false;
    }
    return true;
}

// one frame's planes into frame surface n as RGBA16F: the SDR view (sdr) or the planes as stored
static bool nativeFsrfgPack(NativeRife& nr, const float* src, int n, bool sdr)
{
    int ps = nr.ph * nr.pw, sdrI = sdr ? 1 : 0, pitch = nr.fgPitch;
    void* a[] = {(void*)&src, &ps, &nr.pw, &nr.w, &nr.h, &sdrI, &nr.encPost, &nr.encWhite, &nr.dFrSurf[n], &pitch};
    if (cuLaunchKernel(nr.fPackRgbaH, (nr.w + 15) / 16, (nr.h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)nr.stream, a,
                       nullptr) != CUDA_SUCCESS)
    {
        nr.die("packRgbaH (fsrfg) launch failed");
        return false;
    }
    nr.frSurfSdr[n] = sdr;
    if (nr.fgOf.keepSrc == nr.dFrSurf[n]) // its ABGR8 copy in the Optical Flow's input slot is stale now
    {
        nr.fgOf.keepSlot = -1;
        nr.fgOf.keepSrc = nullptr;
    }
    return true;
}

// the level-1 chain's two streams and their events (nativeFsrfgSetup makes them)
static void nativeFsrfgChainFree(NativeRife& nr)
{
    for (cudaEvent_t* e : {&nr.fgEvL1, &nr.fgEvS, &nr.fgEvDeep, &nr.fgEvOf, &nr.fgEvMv, &nr.fgEvSurf[0],
                           &nr.fgEvSurf[1], &nr.fgEvSurf[2]})
        if (*e)
        {
            cudaEventDestroy(*e);
            *e = nullptr;
        }
    for (cudaStream_t* s : {&nr.fgL1St, &nr.fgOfL1St})
        if (*s)
        {
            cudaStreamDestroy(*s);
            *s = nullptr;
        }
}

static bool nativeFsrfgSetup(NativeRife& nr)
{
    if (!nativeFsrfgLoad(nr.frucDir, nr.cachePath, nr.fgKind))
        return false;
    int dev = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess)
    {
        LOG("native: fsrfg: the CUDA device's properties are unavailable\n");
        return false;
    }
    memcpy(&nr.fgLuid, prop.luid, sizeof(nr.fgLuid));
    nr.fgPitch = nr.fgMvPitch = 0;
    for (NativeRife::FgInst& g : nr.fg)
        g = NativeRife::FgInst{};
    // fgOf on its own stream fgOfL1St, instance 0's pipelined calls on fgL1St (the pipelined route's level-1 chain);
    // the profiler keeps both on the session's
    nr.fgOfWr = nr.fgOfRd = 1;
    nr.fgPiped = nr.fgL1Pend = nr.fgL1Ok = nr.fgPipeLogged = false;
    {
        char ev[8] = {};
        if (!(GetEnvironmentVariableA("SMV_FSRFG_PROF", ev, sizeof(ev)) > 0 && ev[0] == '1'))
        {
            bool ok = cudaStreamCreateWithFlags(&nr.fgL1St, cudaStreamNonBlocking) == cudaSuccess &&
                      cudaStreamCreateWithFlags(&nr.fgOfL1St, cudaStreamNonBlocking) == cudaSuccess;
            for (cudaEvent_t* e : {&nr.fgEvL1, &nr.fgEvS, &nr.fgEvDeep, &nr.fgEvOf, &nr.fgEvMv, &nr.fgEvSurf[0],
                                   &nr.fgEvSurf[1], &nr.fgEvSurf[2]})
                ok = ok && cudaEventCreateWithFlags(e, cudaEventDisableTiming) == cudaSuccess;
            if (!ok)
            {
                LOG("native: fsrfg: the level-1 streams did not start: the Optical Flow on the session's stream, no "
                    "pipelined route\n");
                nativeFsrfgChainFree(nr);
            }
        }
    }
    // never the fast perf level here: at fast, grid 1, the engine's vectors can go wrong after ~320 calls of a 1080p
    // session with vkd3d-proton in the process (garbage tweens from then on, run to run different)
    if (!nativeFsrfgCreate(nr, 0) || !nativeOfVecSetup(nr, nr.fgOf, nr.w, nr.h, "fsrfg", nr.fgOfL1St))
        return false;
    if (nr.fgL1St)
        NCHK(cudaMalloc((void**)&nr.dFgMvL1, (size_t)nr.fgMvPitch * nr.h), "alloc fsrfg level-1 vectors");
    const size_t bytes = (size_t)nr.fgPitch * nr.h, plane = (size_t)nr.ph * nr.pw;
    for (auto& s : nr.dFrSurf)
        NCHK(cudaMalloc((void**)&s, bytes), "alloc fsrfg frame");
    // the tween in the model layout; the pad outside w x h is never written, zero it once
    NCHK(cudaMalloc((void**)&nr.dFrOut, 3 * plane * sizeof(float)), "alloc fsrfg tween");
    NCHK(cudaMemsetAsync(nr.dFrOut, 0, 3 * plane * sizeof(float), nr.stream), "clear fsrfg tween");
    {
        char ev[8] = {};
        nr.frSdrOn =
            nr.encPost && !(GetEnvironmentVariableA("SMV_FSRFG_SDR_PAIRS", ev, sizeof(ev)) > 0 && ev[0] == '0');
    }
    if (nr.frSdrOn)
    {
        NCHK(cudaMalloc((void**)&nr.dFrRange, sizeof(int)), "alloc fsrfg range flag");
        NCHK(cudaHostAlloc((void**)&nr.hFrRange, sizeof(int), cudaHostAllocDefault), "alloc fsrfg range readback");
    }
    {
        char ev[8] = {};
        nr.fgProf = GetEnvironmentVariableA("SMV_FSRFG_PROF", ev, sizeof(ev)) > 0 && ev[0] == '1';
        for (cudaEvent_t& e : nr.fgEv)
            if (nr.fgProf && !e && cudaEventCreate(&e) != cudaSuccess)
                nr.fgProf = false;
        for (double& s : nr.fgStageMs)
            s = 0.0;
        nr.fgCpuMs = nr.fgDeriveMs = 0.0;
        nr.fgProfN = nr.fgProfVecN = nr.fgProfDerN = 0;
    }
    for (int i = 0; i < 3; i++)
        nr.frIn[i] = nr.frSurfSdr[i] = false;
    nr.frPairSdr = false;
    nr.frModeLogs = 0;
    nr.frPrev = nr.frLast = nr.frA = nr.frB = -1;
    nr.frInst = 1;
    nr.frInstMax = kFrDepth;
    {
        // FRUC's lever: SMV_FRUC_INSTANCES caps the instances (5 = the tree to x32)
        char ev[8] = {};
        if (GetEnvironmentVariableA("SMV_FRUC_INSTANCES", ev, sizeof(ev)) > 0)
            nr.frInstMax = (std::max)(1, (std::min)(kFrDepth, atoi(ev)));
    }
    nr.frPlanN = nr.frPlanK = 0;
    nr.frMp = true;
    nr.frMpCap = g_offline ? 3 : 2; // FRUC's snap depths: offline x8's nodes, live x4's
    nr.frMpL = 0;
    nr.frMpBuilt = false;
    nr.frSerial = 0;
    nr.frNodeCalls = nr.frTweens = nr.frPrimed = nr.frRepeats = 0;
    for (bool& ok : nr.frNodeOk)
        ok = false;
    if (nr.fgKind == 1)
        LOG("native: fsrfg session %dx%d, Intel %s frame generation through smv_xefg_bridge.dll on D3D12, RGBA16F, NVIDIA optical flow vectors (grid %d, %s; level-2+ calls %s), recursive midpoints (depth cap %d)\n",
            nr.w, nr.h, g_fsrfgK[1].provider(), nr.fgOf.grid, nativeOfLevelName(nr.fgOf.level),
            fsrfgDeriveOn(nr) ? "derived from level 1" : "a pass a call", nr.frMpCap);
    else
        LOG("native: fsrfg session %dx%d, AMD FSR %s frame generation (ML) through smv_fsrfg_bridge.dll on vkd3d-proton, RGBA16F, NVIDIA optical flow vectors (grid %d, %s; level-2+ calls %s), recursive midpoints (depth cap %d)\n",
            nr.w, nr.h, g_fsrfgK[0].provider(), nr.fgOf.grid, nativeOfLevelName(nr.fgOf.level),
            fsrfgDeriveOn(nr) ? "derived from level 1" : "a pass a call", nr.frMpCap);
    if (nr.encPost)
        LOG("native: fsrfg: HDR planes: %s\n",
            nr.frSdrOn ? "frame pairs inside the SDR range go to FSR as their SDR view, any other pair as the HDR codes"
                       : "every pair as the HDR codes (SMV_FSRFG_SDR_PAIRS=0)");
    return true;
}

// the new frame packed into a frame surface (never the last one, nor `keep`) and the pair advanced to it (frA / frB /
// frSerial, its nodes cleared); repacked = the last frame was packed again in the other encoding
static bool nativeFsrfgPairStep(NativeRife& nr, const float* dCur, uint32_t nTween, int keep, bool& repacked)
{
    int n = 0;
    while (n == nr.frPrev || n == keep)
        n++;
    repacked = false;
    // HDR planes: FRUC's rule (nativeFrucPair): the SDR view only when both frames stay inside the SDR range
    bool in = false;
    if (nr.frSdrOn)
    {
        cudaStream_t st = nr.stream;
        int ps = nr.ph * nr.pw;
        float tol = 2e-3f;
        void* ar[] = {(void*)&dCur, &ps, &nr.pw, &nr.w, &nr.h, &nr.encPost, &nr.encWhite, &tol, (void*)&nr.dFrRange};
        if (cudaMemsetAsync(nr.dFrRange, 0, sizeof(int), st) != cudaSuccess ||
            cuLaunchKernel(nr.fHdrRange, (nr.w + 15) / 16, (nr.h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, ar,
                           nullptr) != CUDA_SUCCESS ||
            cudaMemcpyAsync(nr.hFrRange, nr.dFrRange, sizeof(int), cudaMemcpyDeviceToHost, st) != cudaSuccess ||
            cudaStreamSynchronize(st) != cudaSuccess)
        {
            nr.die("fsrfg range test failed");
            return false;
        }
        in = *nr.hFrRange == 0;
    }
    const bool sdr = in && (nr.frPrev < 0 || nr.frIn[nr.frPrev]);
    // pipelined, the level-1 pass and instance 0's copies read the surfaces off the session's stream: a pack waits for
    // its surface's last such read (fgEvSurf), never for instance 0's call
    auto surfFree = [&nr](int k) {
        if (nr.fgPiped && cudaStreamWaitEvent(nr.stream, nr.fgEvSurf[k], 0) != cudaSuccess)
        {
            nr.die("fsrfg frame wait failed");
            return false;
        }
        return true;
    };
    if (!surfFree(n) || !nativeFsrfgPack(nr, dCur, n, sdr))
        return false;
    nr.frIn[n] = in;
    if (nr.frPrev >= 0 && nr.frSurfSdr[nr.frPrev] != sdr)
    {
        // the previous frame in the other encoding: packed again, and every fed instance restarts there (a key that
        // names no frame: a plain call, never a reset)
        if (!surfFree(nr.frPrev) || !nativeFsrfgPack(nr, nativeXPrev(nr), nr.frPrev, sdr))
            return false;
        for (uint64_t& k : nr.frLastKey)
            if (k)
                k = ~0ull;
        repacked = true;
    }
    if (nr.frSdrOn && nTween && nr.frPrev >= 0 && nr.frModeLogs < 4 && (nr.frModeLogs == 0 || sdr != nr.frPairSdr))
    {
        LOG("native: fsrfg: %s\n", sdr ? "frame pairs inside the SDR range, their SDR view"
                                       : "a frame pair outside the SDR range, the HDR codes");
        nr.frModeLogs++;
    }
    nr.frPairSdr = sdr;
    nr.frA = nr.frPrev;
    nr.frB = n;
    nr.frPrev = n;
    nr.frSerial++;
    nr.frMpL = 0;
    nr.frMpBuilt = false;
    for (bool& ok : nr.frNodeOk)
        ok = false;
    return true;
}

static bool nativeFsrfgPair(NativeRife& nr, const float* dCur, uint32_t nTween)
{
    bool repacked = false;
    if (!nativeFsrfgPairStep(nr, dCur, nTween, -1, repacked))
        return false;
    if (nr.frA < 0)
    {
        // the stream's first frame: instance 0 starts there with a reset
        if (!nativeFsrfgFeed(nr, 0, nr.dFrSurf[nr.frB], nullptr, true, nullptr))
            return false;
        nr.frLastKey[0] = nativeFrucKey(nativeFrucCur(nr), 1, 0);
        return true;
    }
    // every real frame goes to instance 0 in order, tweens or not (a gap would cost a re-prime): that call is the
    // pair's midpoint
    return nativeFsrfgNode(nr, 1, 1) != nullptr;
}

// the pipelined route (SMV_FSRFG_PIPE, nativeGroup's route bit) is open: the derived level-2+ vectors (a pair's
// deeper calls read only its own level-1 field), one field a pair (not SMV_OFVEC_BOTH), fgOf on fgL1St (not the
// profiler)
static bool fsrfgPipeOk(const NativeRife& nr)
{
    return nr.fgL1St && nr.dFgMvL1 && fsrfgDeriveOn(nr) && !nr.fgOf.both && !nr.fgProf;
}

// SMV_FSRFG_PIPE=serial (an A/B lever): the pipelined route with the new pair's Optical Flow too after the shown pair's
// deeper calls (its bookkeeping without its overlap)
static bool fsrfgPipeSerial()
{
    static const bool on = [] {
        char v[16] = "";
        return GetEnvironmentVariableA("SMV_FSRFG_PIPE", v, sizeof(v)) > 0 && strcmp(v, "serial") == 0;
    }();
    return on;
}

// the new pair's level-1 Optical Flow and instance 0's copies (nativeFsrfgFeed part 1): the pair's two surfaces are
// read off the session's stream until fgL1St's copies are done (fgEvSurf, the next packs wait for it)
static bool nativeFsrfgL1Vec(NativeRife& nr)
{
    if (!nativeFsrfgFeed(nr, 0, nr.dFrSurf[nr.frB], nr.dFrSurf[nr.frA], false, nullptr, -1, true, 1))
        return false;
    if (cudaEventRecord(nr.fgEvSurf[nr.frA], nr.fgL1St) != cudaSuccess ||
        cudaEventRecord(nr.fgEvSurf[nr.frB], nr.fgL1St) != cudaSuccess)
    {
        nr.die("fsrfg level-1 event failed");
        return false;
    }
    return true;
}

// a pipelined group's pair step: the pair the last group made current becomes the one shown (its level-1 node, made
// on fgL1St then, swapped into dFrNode), the new frame is packed sparing the shown pair's first frame (each pack waits
// only for its surface's last read off the session's stream, nativeFsrfgPairStep), and the new pair's level-1 chain
// starts: its Optical Flow on fgOfL1St into the other field now, beside instance 0's last call, the call itself on
// fgL1St after the shown pair's deeper calls (nativeFsrfgL1Issue). The shown pair is held (its second frame in every
// slot) when it was identical, when the last group was not pipelined (it showed that pair itself: the route's entry
// costs one source period's hold) or when its second frame was packed again in the other encoding
static bool nativeFsrfgPipeAdvance(NativeRife& nr, const float* dCur, bool curStatic)
{
    nr.frLagSerial = nr.frSerial;
    nr.frLagA = nr.frA;
    nr.frLagB = nr.frB;
    nr.frLagSdr = nr.frPairSdr;
    nr.frLagStatic = nr.frCurStatic;
    const bool had = nr.fgPiped && nr.frLagA >= 0 && nr.fgL1Ok;
    nr.fgOfRd = nr.fgOfWr;
    bool repacked = false;
    if (!nativeFsrfgPairStep(nr, dCur, curStatic ? 0 : 1, nr.frLagA, repacked))
        return false;
    nr.frLagHave = had && !repacked;
    if (nr.frLagHave)
    {
        std::swap(nr.dFrNode[kFrGrid / 2], nr.dFgL1Node);
        nr.frNodeOk[kFrGrid / 2] = true;
    }
    nr.frLagL = 0;
    nr.fgL1Ok = false;
    nr.frCurStatic = curStatic;
    nr.frPipePrev = true;
    nr.frPipeGroups++;
    nr.fgPiped = true;
    if (!nr.fgPipeLogged)
    {
        LOG("native: fsrfg: pipelined across pairs: the new pair's level-1 Optical Flow and call on their own streams "
            "beside the shown pair's level-2+ calls\n");
        nr.fgPipeLogged = true;
    }
    if (!nr.dFgL1Node && cudaMalloc((void**)&nr.dFgL1Node, (size_t)nr.fgPitch * nr.h) != cudaSuccess)
    {
        nr.dFgL1Node = nullptr;
        nr.die("fsrfg level-1 node alloc failed");
        return false;
    }
    // the level-1 pass and instance 0's stream after the pack
    if (cudaEventRecord(nr.fgEvS, nr.stream) != cudaSuccess ||
        cudaStreamWaitEvent(nr.fgOfL1St, nr.fgEvS, 0) != cudaSuccess ||
        cudaStreamWaitEvent(nr.fgL1St, nr.fgEvS, 0) != cudaSuccess)
    {
        nr.die("fsrfg level-1 event failed");
        return false;
    }
    const FrTree cur = nativeFrucCur(nr);
    if (nr.frA < 0)
    {
        // the stream's first frame: instance 0 starts there with a reset
        if (!nativeFsrfgFeed(nr, 0, nr.dFrSurf[nr.frB], nullptr, true, nullptr, -1, false, 3) ||
            cudaEventRecord(nr.fgEvL1, nr.fgL1St) != cudaSuccess ||
            cudaEventRecord(nr.fgEvSurf[nr.frB], nr.fgL1St) != cudaSuccess)
            return false;
        nr.frLastKey[0] = nativeFrucKey(cur, 1, 0);
        return true;
    }
    const uint64_t lk = nativeFrucKey(cur, 0, 1);
    if (nr.frLastKey[0] != lk)
    {
        // a re-prime (after a repack): the shown pair is held, so no deeper call runs beside this whole call
        nr.frLagHave = false;
        if (!nativeFsrfgFeed(nr, 0, nr.dFrSurf[nr.frA], nullptr, nr.frLastKey[0] == 0, nullptr, -1, false, 3))
            return false;
        if (nr.frLastKey[0] != 0)
            nr.frPrimed++;
        nr.frLastKey[0] = lk;
    }
    nr.fgOfWr = nr.fgOf.wr = 1 - nr.fgOfRd;
    nr.fgL1Pend = true;
    return fsrfgPipeSerial() || nativeFsrfgL1Vec(nr);
}

// the pipelined route's instance-0 call of the new pair (nativeFsrfgPipeAdvance made its vectors) on fgL1St after
// fgEvDeep = all the session's stream queued so far (the shown pair's deeper calls), so FSR's calls keep the serial
// route's order on the device; its node into dFgL1Node, fgEvL1 after it
static bool nativeFsrfgL1Issue(NativeRife& nr)
{
    if (!nr.fgL1Pend)
        return true;
    nr.fgL1Pend = false;
    if (cudaEventRecord(nr.fgEvDeep, nr.stream) != cudaSuccess)
    {
        nr.die("fsrfg level-1 event failed");
        return false;
    }
    if (fsrfgPipeSerial() && (cudaStreamWaitEvent(nr.fgOfL1St, nr.fgEvDeep, 0) != cudaSuccess || !nativeFsrfgL1Vec(nr)))
    {
        nr.die("fsrfg level-1 call failed");
        return false;
    }
    if (!nativeFsrfgFeed(nr, 0, nr.dFrSurf[nr.frB], nr.dFrSurf[nr.frA], false, nr.dFgL1Node, -1, true, 2))
        return false;
    if (cudaEventRecord(nr.fgEvL1, nr.fgL1St) != cudaSuccess)
    {
        nr.die("fsrfg level-1 event failed");
        return false;
    }
    nr.frLastKey[0] = nativeFrucKey(nativeFrucCur(nr), 2, 1);
    nr.frNodeCalls++;
    nr.fgL1Ok = true;
    return true;
}

// a pipelined group's tween: the shown pair's levels 2 to its depth (level by level, as nativeFsrfgTween) once its
// level-1 node is done on fgL1St, then the new pair's level-1 call (nativeFsrfgL1Issue), then the nearest node
static bool nativeFsrfgLagTween(NativeRife& nr, double t)
{
    if (!nr.frMpL)
    {
        nr.frMpL = (std::min)(nativeFrucDepth(nr, &t, 1), nr.frInst);
        if (cudaStreamWaitEvent(nr.stream, nr.fgEvL1, 0) != cudaSuccess)
        {
            nr.die("fsrfg level-1 wait failed");
            return false;
        }
        // the deeper calls' price starts here (nativeGroup): the wait above belongs to the level-1 chain
        if (nr.grLagEv)
            nr.grLagArmed = cudaEventRecord(nr.grLagEv, nr.stream) == cudaSuccess;
        for (int l = 2; l <= nr.frMpL; l++)
            if (nativeFsrfgAhead(nr))
            {
                if (!nativeFsrfgLevel(nr, l))
                    return false;
            }
            else
                for (int k = 1; k < (1 << l); k += 2)
                    if (!nativeFsrfgNode(nr, k, l))
                        return false;
        if (!nativeFsrfgL1Issue(nr))
            return false;
        nr.frMpBuilt = true;
    }
    const int den = 1 << nr.frMpL;
    const int k = (std::max)(1, (std::min)(den - 1, (int)floor(t * den + 0.5)));
    uint8_t* node = nativeFsrfgNode(nr, k, nr.frMpL);
    if (!node)
        return false;
    nr.frTweens++;
    int pitch = nr.fgPitch, sdr = nr.frLagSdr ? 1 : 0, ps = nr.ph * nr.pw;
    void* a[] = {&node, &pitch, &nr.w, &nr.h, &sdr, &nr.encPost, &nr.encWhite, &nr.dFrOut, &ps, &nr.pw};
    if (cuLaunchKernel(nr.fUnpackRgbaH, (nr.w + 15) / 16, (nr.h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)nr.stream, a,
                       nullptr) != CUDA_SUCCESS)
    {
        nr.die("unpackRgbaH (fsrfg) launch failed");
        return false;
    }
    return true;
}

// the direct route after a pipelined group: the session's stream waits for fgL1St's last work (the level-1 call of
// the pair the last group made current, which this route never shows; it waited for fgOfL1St's last pass), one field
// again
static bool nativeFsrfgPipeExit(NativeRife& nr)
{
    if (!nr.fgPiped)
        return true;
    nr.fgPiped = nr.fgL1Ok = nr.fgL1Pend = false;
    nr.fgOfWr = nr.fgOfRd = nr.fgOf.wr = 1;
    if (cudaStreamWaitEvent(nr.stream, nr.fgEvL1, 0) != cudaSuccess)
    {
        nr.die("fsrfg level-1 wait failed");
        return false;
    }
    return true;
}

// one tween at t into dFrOut: the pair's whole tree to its depth first, level by level (FRUC's order: level L's
// instance sees the continuous stream A, 2 / 2^L, .. B), then the nearest node
static bool nativeFsrfgTween(NativeRife& nr, double t)
{
    if (!nr.frMpL)
        nr.frMpL = nativeFrucDepth(nr, &t, 1);
    if (nr.frInst < nr.frMpL && nr.frInst < nr.frInstMax)
        nativeFsrfgGrow(nr, (std::min)(nr.frMpL, nr.frInstMax));
    if (nr.frInst < nr.frMpL)
        nr.frMpL = nr.frInst;
    if (!nr.frMpBuilt)
    {
        for (int l = 1; l <= nr.frMpL; l++)
            if (l >= 2 && nativeFsrfgAhead(nr))
            {
                if (!nativeFsrfgLevel(nr, l))
                    return false;
            }
            else
                for (int k = 1; k < (1 << l); k += 2)
                    if (!nativeFsrfgNode(nr, k, l))
                        return false;
        nr.frMpBuilt = true;
    }
    const int den = 1 << nr.frMpL;
    const int k = (std::max)(1, (std::min)(den - 1, (int)floor(t * den + 0.5)));
    uint8_t* node = nativeFsrfgNode(nr, k, nr.frMpL);
    if (!node)
        return false;
    nr.frTweens++;
    int pitch = nr.fgPitch, sdr = nr.frPairSdr ? 1 : 0, ps = nr.ph * nr.pw;
    void* a[] = {&node, &pitch, &nr.w, &nr.h, &sdr, &nr.encPost, &nr.encWhite, &nr.dFrOut, &ps, &nr.pw};
    if (cuLaunchKernel(nr.fUnpackRgbaH, (nr.w + 15) / 16, (nr.h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)nr.stream, a,
                       nullptr) != CUDA_SUCCESS)
    {
        nr.die("unpackRgbaH (fsrfg) launch failed");
        return false;
    }
    return true;
}

// the instances' CUDA imports, then the bridge's instances, then the Optical Flow session (nativeFrucFree frees the
// frames after this)
static void nativeFsrfgRelease(NativeRife& nr)
{
    for (cudaStream_t s : {nr.fgOfL1St, nr.fgL1St})
        if (s)
            cudaStreamSynchronize(s);
    if (nr.stream)
        cudaStreamSynchronize(nr.stream);
    if (nr.fg[0].made)
        LOG("native: fsrfg session: %llu tweens, %llu midpoint calls, %llu re-primed streams%s, %d instance%s\n",
            (unsigned long long)nr.frTweens, (unsigned long long)nr.frNodeCalls, (unsigned long long)nr.frPrimed,
            nr.frPipeGroups ? (", " + std::to_string(nr.frPipeGroups) + " groups pipelined").c_str() : "", nr.frInst,
            nr.frInst > 1 ? "s" : "");
    // XeSS-FG's own counts per instance: a call with no frame presented keeps the previous output (none expected after
    // a sequence's first call)
    if (nr.fgKind == 1 && g_fsrfgK[1].stats)
        for (int i = 0; i < nr.frInst && i < (int)(sizeof(nr.fg) / sizeof(nr.fg[0])); i++)
        {
            uint64_t c = 0, nf = 0, ex = 0, cap = 0;
            if (nr.fg[i].made && g_fsrfgK[1].stats(i, &c, &nf, &ex, &cap) == 0)
                LOG("native: fsrfg: XeSS-FG instance %d: %llu calls, %llu frames out, %llu calls without a frame, %llu "
                    "presents without a call\n",
                    i, (unsigned long long)c, (unsigned long long)cap, (unsigned long long)nf, (unsigned long long)ex);
        }
    if (nr.fgProfN)
    {
        const double n = (double)nr.fgProfN, nv = nr.fgProfVecN ? (double)nr.fgProfVecN : 1.0,
                     nd = nr.fgProfDerN ? (double)nr.fgProfDerN : 1.0;
        double sum = nr.fgDeriveMs;
        for (double s : nr.fgStageMs)
            sum += s;
        LOG("[fsrfg-prof] %llu calls (%llu with an Optical Flow pass, %llu derived) %dx%d, ms a call: copy in %.3f, "
            "Optical Flow inputs %.3f, Optical Flow %.3f, vector kernels %.3f (the three a pass call), derived vectors "
            "%.3f (a derived call), vector copy %.3f, FSR on vkd3d %.3f, copy out %.3f, all %.3f; the bridge's CPU "
            "call %.3f\n",
            (unsigned long long)nr.fgProfN, (unsigned long long)nr.fgProfVecN, (unsigned long long)nr.fgProfDerN, nr.w,
            nr.h, nr.fgStageMs[0] / n, nr.fgStageMs[1] / nv, nr.fgStageMs[2] / nv, nr.fgStageMs[3] / nv,
            nr.fgDeriveMs / nd, nr.fgStageMs[4] / n, nr.fgStageMs[5] / n, nr.fgStageMs[6] / n, sum / n, nr.fgCpuMs / n);
    }
    for (cudaEvent_t& e : nr.fgEv)
        if (e)
        {
            cudaEventDestroy(e);
            e = nullptr;
        }
    nr.fgProfN = 0;
    bool any = false;
    for (NativeRife::FgInst& g : nr.fg)
    {
        for (cudaMipmappedArray_t* mm : {&g.mmIn, &g.mmMv, &g.mmOut})
            if (*mm)
            {
                cudaFreeMipmappedArray(*mm);
                *mm = nullptr;
            }
        if (g.sem)
            cudaDestroyExternalSemaphore(g.sem);
        for (cudaExternalMemory_t* em : {&g.emIn, &g.emMv, &g.emOut})
            if (*em)
            {
                cudaDestroyExternalMemory(*em);
                *em = nullptr;
            }
        any = any || g.made;
        g = NativeRife::FgInst{};
    }
    if (any && g_fsrfgK[nr.fgKind].destroy)
        g_fsrfgK[nr.fgKind].destroy(); // every instance
    if (nr.dFgMv)
    {
        cudaFree(nr.dFgMv);
        nr.dFgMv = nullptr;
    }
    nativeFsrfgAheadFree(nr);
    nativeOfVecFree(nr.fgOf);
    for (uint8_t** b : {&nr.dFgMvL1, &nr.dFgL1Node})
        if (*b)
        {
            cudaFree(*b);
            *b = nullptr;
        }
    nativeFsrfgChainFree(nr);
    nr.fgPiped = nr.fgL1Pend = nr.fgL1Ok = false;
    nr.frInst = 1;
}
