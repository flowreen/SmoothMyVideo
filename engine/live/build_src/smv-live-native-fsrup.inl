// part of smv-live-native.inl (included there, in place); not a standalone translation unit
// ---- AMD FSR / Intel XeSS-SR upscaling (fsrUp) ---------------------------------------------------
// One C API, two bridges (fuKind): smv_fsrup_bridge.dll (source engine\fsrup\build_src, exports fsrup_*; SMV_FSRUP_DIR,
// else engine\fsrup, AMD's signed loader + upscaler beside it) and smv_xess_bridge.dll (source engine\xess\build_src,
// exports xessup_*; SMV_XESS_DIR, else engine\xess, Intel's libxess.dll beside it). Each loads once per process by full
// path from its folder
struct FsrupShared
{
    HANDLE in, mv, out, fence;
    uint64_t inBytes, mvBytes, outBytes;
    uint32_t inPitch, mvPitch, outPitch;
    char version[32];
};
struct FsrupBridge
{
    HMODULE mod = nullptr;
    const char* (*lastError)() = nullptr;
    int (*create)(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, int32_t, const char*, int, FsrupShared*) = nullptr;
    int (*frame)(uint64_t, uint64_t, int, double) = nullptr;
    void (*destroy)() = nullptr;
    // instance k of the same (the chained passes past 3x, fuChainPlan); an older bridge lacks them
    int (*createI)(int, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, int32_t, const char*, int,
                   FsrupShared*) = nullptr;
    int (*frameI)(int, uint64_t, uint64_t, int, double) = nullptr;
    void (*destroyI)(int) = nullptr;
};
static FsrupBridge g_fsrupK[2];
static const char* const kFuDll[2] = {"smv_fsrup_bridge.dll", "smv_xess_bridge.dll"};
static const char* const kFuPrefix[2] = {"fsrup_", "xessup_"};
static const char* const kFuName[2] = {"AMD FSR", "Intel"};

static bool nativeFsrupLoad(const std::string& dir, int kind)
{
    FsrupBridge& b = g_fsrupK[kind];
    if (b.mod)
        return true;
    const std::wstring p = utf8ToWide(dir) + L"\\" + utf8ToWide(kFuDll[kind]);
    HMODULE m = LoadLibraryExW(p.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!m)
    {
        LOG("native: upscale bridge: %s did not load from %s (err %lu)\n", kFuDll[kind], dir.c_str(), GetLastError());
        return false;
    }
    const std::string pre = kFuPrefix[kind];
    b.lastError = (const char* (*)())GetProcAddress(m, (pre + "last_error").c_str());
    b.create = (int (*)(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, int32_t, const char*, int,
                        FsrupShared*))GetProcAddress(m, (pre + "create").c_str());
    b.frame = (int (*)(uint64_t, uint64_t, int, double))GetProcAddress(m, (pre + "frame").c_str());
    b.destroy = (void (*)())GetProcAddress(m, (pre + "destroy").c_str());
    if (!b.lastError || !b.create || !b.frame || !b.destroy)
    {
        LOG("native: upscale bridge: %s lacks an expected export\n", kFuDll[kind]);
        FreeLibrary(m);
        b = FsrupBridge{};
        return false;
    }
    b.createI = (int (*)(int, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, int32_t, const char*, int,
                         FsrupShared*))GetProcAddress(m, (pre + "create_i").c_str());
    b.frameI = (int (*)(int, uint64_t, uint64_t, int, double))GetProcAddress(m, (pre + "frame_i").c_str());
    b.destroyI = (void (*)(int))GetProcAddress(m, (pre + "destroy_i").c_str());
    b.mod = m;
    return true;
}

// the reason an early (handoff-time) setup failed, named by the pipe's "the early engine handoff failed" line (cleared
// when the early thread starts)
static char g_earlyWhy[240];
static void earlyWhy(const char* fmt, ...)
{
    va_list a;
    va_start(a, fmt);
    vsnprintf(g_earlyWhy, sizeof(g_earlyWhy), fmt, a);
    va_end(a);
}

// AMD FSR / Intel XeSS-SR past 3x a side (FSR's named presets end at 3.0x and past about 9x it leaves one dot an input
// pixel; XeSS-SR refuses past 3.0x): the fewest equal passes of at most 3x, the last landing on ow x oh, the sizes in
// between even (the upscaler leaves an odd last column / row untouched). pw / ph [0 .. n] = the sizes; returns n (1 =
// one call: within 3x, or a shrink), 0 = no chain within kFuMaxPass. SMV_FU_CHAIN=0: one call at any ratio (A/B)
static int fuChainPlan(int w, int h, int ow, int oh, int* pw, int* ph)
{
    pw[0] = w;
    ph[0] = h;
    pw[1] = ow;
    ph[1] = oh;
    static const bool off = [] {
        char v[8] = {};
        return GetEnvironmentVariableA("SMV_FU_CHAIN", v, sizeof(v)) > 0 && v[0] == '0';
    }();
    if (off || ow < w || oh < h || ((double)ow <= 3.0 * w && (double)oh <= 3.0 * h))
        return 1;
    const double rx = (double)ow / w, ry = (double)oh / h;
    const int n0 = (std::max)(2, (int)std::ceil(std::log((std::max)(rx, ry)) / std::log(3.0) - 1e-9));
    for (int n = n0; n <= NativeRife::kFuMaxPass; n++)
    {
        pw[n] = ow;
        ph[n] = oh;
        for (int k = 1; k < n; k++)
        {
            pw[k] = (std::min)(ow, ((int)std::ceil(w * std::pow(rx, (double)k / n)) + 1) & ~1);
            ph[k] = (std::min)(oh, ((int)std::ceil(h * std::pow(ry, (double)k / n)) + 1) & ~1);
        }
        for (int k = n - 1; k >= 1; k--) // the step into k + 1 at most 3x a side
        {
            pw[k] = (std::max)(pw[k], ((pw[k + 1] + 2) / 3 + 1) & ~1);
            ph[k] = (std::max)(ph[k], ((ph[k + 1] + 2) / 3 + 1) & ~1);
        }
        bool ok = true;
        for (int k = 1; k <= n; k++)
            ok = ok && pw[k] >= pw[k - 1] && ph[k] >= ph[k - 1] && pw[k] <= 3 * pw[k - 1] && ph[k] <= 3 * ph[k - 1];
        if (ok)
            return n;
    }
    return 0;
}

// the upscaler at RTX VSR's resize (w x h -> ow x oh, nativeRtxInit's slot; past 3x a side in fuChainPlan's passes),
// its shared buffers and fences imported, and the vectors' Optical Flow session at the render size
static bool nativeFsrupSetup(NativeRife& nr, int w, int h, int ow, int oh)
{
    if (!nativeFsrupLoad(nr.fsrUpDir, nr.fuKind))
        return false;
    FsrupBridge& br = g_fsrupK[nr.fuKind];
    const int n = fuChainPlan(w, h, ow, oh, nr.fuPw, nr.fuPh);
    if (!n || (n > 1 && (!br.createI || !br.frameI || !br.destroyI)))
    {
        earlyWhy("%s upscaling %dx%d -> %dx%d: %s", kFuName[nr.fuKind], w, h, ow, oh,
                 n ? "its bridge has no chained passes (an older one)" : "no chain of passes of at most 3x reaches it");
        LOG("native: fsr upscale: %s\n", g_earlyWhy);
        return false;
    }
    nr.fuPasses = n;
    int dev = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess)
    {
        LOG("native: fsr upscale: the CUDA device's properties are unavailable\n");
        return false;
    }
    LUID luid;
    memcpy(&luid, prop.luid, sizeof(luid));
    auto imp = [](HANDLE hd, uint64_t bytes, cudaExternalMemory_t& em, uint8_t*& p) {
        cudaExternalMemoryHandleDesc md{};
        md.type = cudaExternalMemoryHandleTypeD3D12Resource;
        md.size = bytes;
        md.flags = cudaExternalMemoryDedicated;
        md.handle.win32.handle = hd;
        if (cudaImportExternalMemory(&em, &md) != cudaSuccess)
        {
            em = nullptr;
            return false;
        }
        cudaExternalMemoryBufferDesc bd{};
        bd.size = bytes;
        return cudaExternalMemoryGetMappedBuffer((void**)&p, em, &bd) == cudaSuccess;
    };
    // pass k on bridge instance k (0 = the plain exports), its shared buffers and fence into the given fields
    auto make = [&](int k, cudaExternalMemory_t& emIn, uint8_t*& dIn, int& inP, cudaExternalMemory_t& emMv,
                    uint8_t*& dMv, int& mvP, cudaExternalMemory_t& emOut, uint8_t*& dOut, int& outP,
                    cudaExternalSemaphore_t& sem, bool& made) {
        const int pw = nr.fuPw[k], ph = nr.fuPh[k], qw = nr.fuPw[k + 1], qh = nr.fuPh[k + 1];
        FsrupShared sh{};
        // live HDR planes: the bridge's HDR mode (FP16 linear light in and out, k_fsrHdrIn / k_fsrHdrOut)
        const int rc = k ? br.createI(k, (uint32_t)pw, (uint32_t)ph, (uint32_t)qw, (uint32_t)qh, luid.LowPart,
                                      luid.HighPart, "4.", nr.fuHdr ? 1 : 0, &sh)
                         : br.create((uint32_t)pw, (uint32_t)ph, (uint32_t)qw, (uint32_t)qh, luid.LowPart,
                                     luid.HighPart, "4.", nr.fuHdr ? 1 : 0, &sh);
        if (rc != 0)
        {
            earlyWhy("%s upscaling %dx%d -> %dx%d not created: %s (rc %d)", kFuName[nr.fuKind], pw, ph, qw, qh,
                     br.lastError(), rc);
            LOG("native: fsr upscale: %dx%d -> %dx%d not created: %s (rc %d)\n", pw, ph, qw, qh, br.lastError(), rc);
            return false;
        }
        made = true;
        cudaExternalSemaphoreHandleDesc sd{};
        sd.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
        sd.handle.win32.handle = sh.fence;
        if (!imp(sh.in, sh.inBytes, emIn, dIn) || !imp(sh.mv, sh.mvBytes, emMv, dMv) ||
            !imp(sh.out, sh.outBytes, emOut, dOut) || cudaImportExternalSemaphore(&sem, &sd) != cudaSuccess)
        {
            LOG("native: fsr upscale: the shared buffers or fence did not import into CUDA\n");
            return false;
        }
        inP = (int)sh.inPitch;
        mvP = (int)sh.mvPitch;
        outP = (int)sh.outPitch;
        if (!k)
        {
            memcpy(nr.fuVersion, sh.version, sizeof(nr.fuVersion));
            nr.fuVersion[sizeof(nr.fuVersion) - 1] = 0;
        }
        return true;
    };
    const bool chain = n > 1;
    if (!make(0, nr.fuEmIn, nr.dFuIn, nr.fuInPitch, nr.fuEmMv, nr.dFuMv, nr.fuMvPitch, chain ? nr.fu0EmOut : nr.fuEmOut,
              chain ? nr.dFu0Out : nr.dFuOut, chain ? nr.fu0OutPitch : nr.fuOutPitch, nr.fuSem, nr.fuMade))
        return false;
    for (int k = 1; k < n; k++)
    {
        NativeRife::FuPass& p = nr.fuX[k - 1];
        const bool last = k == n - 1;
        p.val = 0;
        if (!make(k, p.emIn, p.dIn, p.inPitch, p.emMv, p.dMv, p.mvPitch, last ? nr.fuEmOut : p.emOut,
                  last ? nr.dFuOut : p.dOut, last ? nr.fuOutPitch : p.outPitch, p.sem, p.made))
            return false;
    }
    nr.fuVal = 0;
    nr.fuN = 0;
    nr.fuCur = 0;
    {
        char ev[8] = {};
        nr.fuProf = GetEnvironmentVariableA("SMV_FSRUP_PROF", ev, sizeof(ev)) > 0 && ev[0] == '1';
        for (cudaEvent_t& e : nr.fuEv)
            if (nr.fuProf && !e && cudaEventCreate(&e) != cudaSuccess)
                nr.fuProf = false;
        for (double& s : nr.fuStageMs)
            s = 0.0;
        nr.fuProfN = nr.fuProfVecN = 0;
    }
    if (!nativeOfVecSetup(nr, nr.fuOf, w, h, "fsr upscale"))
        return false;
    char via[160] = "";
    if (chain)
    {
        int o = snprintf(via, sizeof(via), " in %d passes (", n);
        for (int k = 1; k < n && o > 0 && o < (int)sizeof(via); k++)
            o += snprintf(via + o, sizeof(via) - o, "%s%dx%d", k > 1 ? ", " : "", nr.fuPw[k], nr.fuPh[k]);
        if (o > 0 && o < (int)sizeof(via))
            snprintf(via + o, sizeof(via) - o, " between)");
    }
    LOG("native: upscale: %s %s %dx%d -> %dx%d%s through %s, NVIDIA optical flow vectors (grid %d, %s)%s\n",
        kFuName[nr.fuKind], nr.fuVersion, w, h, ow, oh, via, kFuDll[nr.fuKind], nr.fuOf.grid,
        nativeOfLevelName(nr.fuOf.level), nr.fuHdr ? ", on the HDR planes' linear light (the bridge's HDR mode)" : "");
    return true;
}

// the tick the stall watchdog (PipeServer::wdogProc) waits for. The driver compiles FSR's shaders inside a session's first
// dispatch, per exe location and upscale ratio (47 s measured, a warm-up covers its own ratio only), and that call blocks this
// thread; a real hang there still ends at kFsrCompileMs, and the watchdog's 10 s start over when it returns
static volatile LONGLONG g_wdogHold = 0;
static constexpr int kFsrCompileMs = 180000;

// one frame of the slot's stream: nr.dVsrIn (tight BGRA8 ri) upscaled into nr.dVsrOut (tight BGRA8 ro), queued on st;
// the frame into the bridge's buffer and the vector session's slot, the vectors from the previous frame's slot (the
// stream's first frame: zero vectors and a reset), the bridge's queue waits on the stream and the stream on the bridge
static bool nativeFsrupEval(NativeRife& nr, const RtxRect& ri, const RtxRect& ro, cudaStream_t st, unsigned int& rv)
{
    int w = (int)ri.right, h = (int)ri.bottom, ow = (int)ro.right, oh = (int)ro.bottom;
    const int cur = nr.fuCur, prev = cur ^ 1;
    int fip = nr.fuInPitch, oip = (int)nr.fuOf.inPitch;
    CUdeviceptr ofIn = nr.fuOf.inP[cur];
    auto mark = [&nr, st](int e) {
        if (nr.fuProf)
            cudaEventRecord(nr.fuEv[e], st);
    };
    mark(0);
    // HDR planes (fuHdr): the callers packed the frame in linear light into dFuIn (k_fsrHdrIn) and take the result back
    // themselves (k_fsrHdrOut); dVsrIn (the frame's SDR view) feeds the vectors only
    uint8_t* in8 = nr.fuHdr ? nullptr : nr.dFuIn;
    void* a[] = {&nr.dVsrIn, &w, &h, &in8, &fip, &ofIn, &oip};
    if (cuLaunchKernel(nr.fBgraRgbaFsr, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a, nullptr) !=
        CUDA_SUCCESS)
    {
        nr.die("bgraRgbaFsr launch failed");
        return false;
    }
    mark(1);
    // The vectors between consecutive emitted frames are mostly near zero on still content (a tween step moves a
    // fraction of a pixel; the flow's noise is a few 1/32 px), and FSR resamples its history by any nonzero vector:
    // still detail then drifts and flips by whole output pixels while exact zeros hold it. So a vector shorter than
    // kFuVecGate px is written as exactly zero; real motion is far above it (a 1 px / frame pan tracks to 0.02 px),
    // and FSR follows a sub-gate tween step no better with the vector than without. SMV_FSRUP_TEST=gateN = N
    // thousandths of a px (gate0 = off, the A/B); =zerovec = zero vectors every frame (A/B only)
    static constexpr float kFuVecGate = 0.125f;
    static const float gate = [] {
        wchar_t v[16]{};
        if (!GetEnvironmentVariableW(L"SMV_FSRUP_TEST", v, 16) || wcsncmp(v, L"gate", 4))
            return kFuVecGate;
        return (float)_wtoi(v + 4) / 1000.0f;
    }();
    static const bool zeroVec = [] {
        wchar_t v[16]{};
        return GetEnvironmentVariableW(L"SMV_FSRUP_TEST", v, 16) && !wcscmp(v, L"zerovec");
    }();
    const bool first = nr.fuN == 0;
    if (first || zeroVec)
    {
        if (cudaMemset2DAsync(nr.dFuMv, nr.fuMvPitch, 0, (size_t)w * 4, h, st) != cudaSuccess)
        {
            nr.die("fsr upscale vector clear failed");
            return false;
        }
        mark(2);
    }
    else if (!nativeOfVecRun(nr, nr.fuOf, prev, cur, nr.dFuMv, nr.fuMvPitch, "fsr upscale",
                             nr.fuProf ? nr.fuEv[2] : nullptr, gate))
        return false;
    mark(3);
    cudaExternalSemaphoreSignalParams sp{};
    sp.params.fence.value = ++nr.fuVal;
    if (cudaSignalExternalSemaphoresAsync(&nr.fuSem, &sp, 1, st) != cudaSuccess)
    {
        nr.die("fsr upscale fence signal failed");
        return false;
    }
    const uint64_t ready = nr.fuVal, done = ++nr.fuVal;
    // SMV_FSRUP_TEST=hang1 / hang2: that frame's call blocks past the watchdog's bound, the watchdog's trigger
    static const int testHang = [] {
        wchar_t v[8]{};
        return GetEnvironmentVariableW(L"SMV_FSRUP_TEST", v, 8) && wcsncmp(v, L"hang", 4) == 0 ? _wtoi(v + 4) : 0;
    }();
    if (first)
        InterlockedExchange64(&g_wdogHold, (LONGLONG)GetTickCount64() + kFsrCompileMs);
    if (testHang && nr.fuN + 1 == (uint64_t)testHang)
        Sleep(kFsrCompileMs + 20000);
    // FSR 4.1.1 gives byte-identical frames at any frame time (measured, harness fsr4_int8_upscaler): one 60 Hz frame
    const int rc = g_fsrupK[nr.fuKind].frame(ready, done, first ? 1 : 0, 1000.0 / 60.0);
    if (first)
        InterlockedExchange64(&g_wdogHold, (LONGLONG)GetTickCount64() + 10000);
    if (rc != 0)
    {
        LOG("native: fsr upscale: frame %llu failed: %s (rc %d)\n", (unsigned long long)nr.fuN,
            g_fsrupK[nr.fuKind].lastError(), rc);
        nr.die("fsr upscale failed");
        return false;
    }
    cudaExternalSemaphoreWaitParams wp{};
    wp.params.fence.value = done;
    int op = nr.fuOutPitch, otp = ow * 4;
    void* b[] = {&nr.dFuOut, &op, &ow, &oh, &nr.dVsrOut, &otp};
    if (cudaWaitExternalSemaphoresAsync(&nr.fuSem, &wp, 1, st) != cudaSuccess)
    {
        nr.die("fsr upscale result failed");
        return false;
    }
    // past 3x (fuChainPlan): passes 1.. on their own instances, each on the previous pass's output with pass 0's
    // vectors scaled to its size (the stream's first frame: zero vectors and a reset, as pass 0)
    for (int k = 1; k < nr.fuPasses; k++)
    {
        NativeRife::FuPass& p = nr.fuX[k - 1];
        int kw = nr.fuPw[k], kh = nr.fuPh[k], mvSrcPitch = nr.fuMvPitch, dp = p.mvPitch;
        float kx = (float)kw / (float)w, ky = (float)kh / (float)h;
        const uint8_t* src = k == 1 ? nr.dFu0Out : nr.fuX[k - 2].dOut;
        const size_t srcPitch = (size_t)(k == 1 ? nr.fu0OutPitch : nr.fuX[k - 2].outPitch);
        void* v[] = {&nr.dFuMv, &mvSrcPitch, &w, &h, &p.dMv, &dp, &kw, &kh, &kx, &ky};
        const bool vecOk = first || zeroVec
                               ? cudaMemset2DAsync(p.dMv, p.mvPitch, 0, (size_t)kw * 4, kh, st) == cudaSuccess
                               : cuLaunchKernel(nr.fFuMvScale, (kw + 15) / 16, (kh + 15) / 16, 1, 16, 16, 1, 0,
                                                (CUstream)st, v, nullptr) == CUDA_SUCCESS;
        if (cudaMemcpy2DAsync(p.dIn, p.inPitch, src, srcPitch, (size_t)kw * (nr.fuHdr ? 8 : 4), kh,
                              cudaMemcpyDeviceToDevice, st) != cudaSuccess ||
            !vecOk)
        {
            nr.die("fsr upscale pass input failed");
            return false;
        }
        cudaExternalSemaphoreSignalParams ps{};
        ps.params.fence.value = ++p.val;
        if (cudaSignalExternalSemaphoresAsync(&p.sem, &ps, 1, st) != cudaSuccess)
        {
            nr.die("fsr upscale fence signal failed");
            return false;
        }
        const uint64_t pReady = p.val, pDone = ++p.val;
        if (first)
            InterlockedExchange64(&g_wdogHold, (LONGLONG)GetTickCount64() + kFsrCompileMs);
        const int prc = g_fsrupK[nr.fuKind].frameI(k, pReady, pDone, first ? 1 : 0, 1000.0 / 60.0);
        if (first)
            InterlockedExchange64(&g_wdogHold, (LONGLONG)GetTickCount64() + 10000);
        if (prc != 0)
        {
            LOG("native: fsr upscale: pass %d, frame %llu failed: %s (rc %d)\n", k + 1, (unsigned long long)nr.fuN,
                g_fsrupK[nr.fuKind].lastError(), prc);
            nr.die("fsr upscale failed");
            return false;
        }
        cudaExternalSemaphoreWaitParams pw{};
        pw.params.fence.value = pDone;
        if (cudaWaitExternalSemaphoresAsync(&p.sem, &pw, 1, st) != cudaSuccess)
        {
            nr.die("fsr upscale result failed");
            return false;
        }
    }
    mark(4);
    if (!nr.fuHdr)
    {
        if (cuLaunchKernel(nr.fRgbaBgra, (ow + 15) / 16, (oh + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, b, nullptr) !=
            CUDA_SUCCESS)
        {
            nr.die("fsr upscale result failed");
            return false;
        }
        void* e[] = {&nr.dVsrOut, &ow, &oh};
        if (((ow | oh) & 1) && cuLaunchKernel(nr.fOddEdge, (ow + 15) / 16, (oh + 15) / 16, 1, 16, 16, 1, 0,
                                              (CUstream)st, e, nullptr) != CUDA_SUCCESS)
        {
            nr.die("fsr upscale edge failed");
            return false;
        }
    }
    mark(5);
    if (nr.fuProf && cudaEventSynchronize(nr.fuEv[5]) == cudaSuccess)
    {
        for (int s = 0; s < 5; s++)
        {
            float ms = 0.0f;
            if (cudaEventElapsedTime(&ms, nr.fuEv[s], nr.fuEv[s + 1]) == cudaSuccess)
                nr.fuStageMs[s] += ms;
        }
        nr.fuProfN++;
        nr.fuProfVecN += first ? 0 : 1;
    }
    nr.fuCur = prev;
    nr.fuN++;
    rv = 1u;
    return true;
}

// nativeUpSwitch's warm-up (fuWarm): two black frames with zero vectors through every pass, the first a reset, on its
// own stream, as the app's fsrWarm session runs them. The native thread touches none of these fields while fuWarm is 1
// (every per-frame reader waits for nr.vsr, set by nativeUpWarmEnd); the session's first real frame resets again (fuN 0)
static DWORD WINAPI nativeFsrupWarmProc(void* p)
{
    NativeRife& nr = *(NativeRife*)p;
    FsrupBridge& br = g_fsrupK[nr.fuKind];
    const int bpp = nr.fuHdr ? 8 : 4;
    cudaStream_t st = nullptr;
    bool ok = nativeBindDevice(nr) && cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking) == cudaSuccess;
    InterlockedExchange64(&g_wdogHold, (LONGLONG)GetTickCount64() + kFsrCompileMs);
    for (int f = 0; f < 2 && ok; f++)
        for (int k = 0; k < nr.fuPasses && ok; k++)
        {
            NativeRife::FuPass* x = k ? &nr.fuX[k - 1] : nullptr;
            uint8_t* in = k ? x->dIn : nr.dFuIn;
            uint8_t* mv = k ? x->dMv : nr.dFuMv;
            const size_t inP = (size_t)(k ? x->inPitch : nr.fuInPitch), mvP = (size_t)(k ? x->mvPitch : nr.fuMvPitch);
            cudaExternalSemaphore_t sem = k ? x->sem : nr.fuSem;
            uint64_t& val = k ? x->val : nr.fuVal;
            const int kw = nr.fuPw[k], kh = nr.fuPh[k];
            cudaExternalSemaphoreSignalParams sp{};
            sp.params.fence.value = ++val;
            const uint64_t ready = val, done = ++val;
            cudaExternalSemaphoreWaitParams wp{};
            wp.params.fence.value = done;
            const int reset = f ? 0 : 1;
            ok = cudaMemset2DAsync(in, inP, 0, (size_t)kw * bpp, kh, st) == cudaSuccess &&
                 cudaMemset2DAsync(mv, mvP, 0, (size_t)kw * 4, kh, st) == cudaSuccess &&
                 cudaSignalExternalSemaphoresAsync(&sem, &sp, 1, st) == cudaSuccess &&
                 (k ? br.frameI(k, ready, done, reset, 1000.0 / 60.0) : br.frame(ready, done, reset, 1000.0 / 60.0)) ==
                     0 &&
                 cudaWaitExternalSemaphoresAsync(&sem, &wp, 1, st) == cudaSuccess;
            if (!ok)
                LOG("native: fsr upscale: warm-up frame %d, pass %d failed: %s\n", f + 1, k + 1, br.lastError());
        }
    ok = ok && cudaStreamSynchronize(st) == cudaSuccess;
    if (st)
        cudaStreamDestroy(st);
    InterlockedExchange64(&g_wdogHold, (LONGLONG)GetTickCount64() + 10000);
    InterlockedExchange(&nr.fuWarm, ok ? 2 : 3);
    return 0;
}

// the warm-up's end (nativeLiveSets, on the native thread): warm = the upscaler on, as nativeUpSwitch turns the other
// kinds on; an `up 0` meanwhile = built and off (an `up N` of its kind turns it on); failed = Lanczos stays, and the
// built bridge stays with the session (nativeUpSwitch refuses another start while fuMade)
static void nativeUpWarmEnd(NativeRife& nr)
{
    WaitForSingleObject(nr.fuWarmTh, INFINITE);
    CloseHandle(nr.fuWarmTh);
    nr.fuWarmTh = nullptr;
    const bool ok = InterlockedExchange(&nr.fuWarm, 0) == 2;
    const char* name = nr.fuKind ? "Intel XeSS upscaling" : "AMD FSR upscaling";
    const double ms = (double)(GetTickCount64() - nr.fuWarmT0);
    if (!ok)
    {
        LOG("native: live upscale: %s's first frames failed after %.0f ms, Lanczos stays\n", name, ms);
        return;
    }
    const int tw = nr.uw ? nr.uw : nr.dw, th = nr.uw ? nr.uh : nr.dh;
    nr.vsrWant = nr.fsrUp = nr.vsr = true;
    nr.vsrFailed = nr.vsrUserOff = !nr.fuWarmWant;
    LOG("native: live upscale: %s %dx%d -> %dx%d, started in place (its first frames on a worker: %.0f ms)%s\n", name,
        nr.w, nr.h, tw, th, ms, nr.fuWarmWant ? "" : ", off: unticked meanwhile");
}

static void nativeFsrupRelease(NativeRife& nr)
{
    if (nr.fuWarmTh)
    {
        // a start in place still warming (nativeUpSwitch): its frames end before the bridge goes; past the compile's
        // bound the worker is still inside the bridge, so its state is left to the process exit
        const DWORD w = WaitForSingleObject(nr.fuWarmTh, (DWORD)kFsrCompileMs + 10000);
        if (w != WAIT_OBJECT_0)
        {
            LOG("native: fsr upscale: its warm-up still runs after %d s, its state is left to the process exit\n",
                kFsrCompileMs / 1000 + 10);
            return;
        }
        CloseHandle(nr.fuWarmTh);
        nr.fuWarmTh = nullptr;
        InterlockedExchange(&nr.fuWarm, 0);
    }
    if (nr.stream)
        cudaStreamSynchronize(nr.stream);
    if (nr.fuMade)
        LOG("native: fsr upscale: %llu frames (%s %s)\n", (unsigned long long)nr.fuN, kFuName[nr.fuKind], nr.fuVersion);
    if (nr.fuProfN)
    {
        const double n = (double)nr.fuProfN, nv = nr.fuProfVecN ? (double)nr.fuProfVecN : 1.0;
        double sum = 0.0;
        for (double s : nr.fuStageMs)
            sum += s;
        LOG("[fsrup-prof] %llu frames (%llu with vectors), ms a frame: input pack %.3f, Optical Flow %.3f, vector "
            "kernels %.3f (the two a vector frame), FSR on D3D12 %.3f, output unpack %.3f, all %.3f\n",
            (unsigned long long)nr.fuProfN, (unsigned long long)nr.fuProfVecN, nr.fuStageMs[0] / n,
            nr.fuStageMs[1] / nv, nr.fuStageMs[2] / nv, nr.fuStageMs[3] / n, nr.fuStageMs[4] / n, sum / n);
    }
    for (cudaEvent_t& e : nr.fuEv)
        if (e)
        {
            cudaEventDestroy(e);
            e = nullptr;
        }
    nr.fuProfN = 0;
    for (int k = nr.fuPasses - 1; k >= 1; k--) // the chained passes past 3x (fuChainPlan), then pass 0's own output
    {
        NativeRife::FuPass& p = nr.fuX[k - 1];
        for (uint8_t** b : {&p.dIn, &p.dMv, &p.dOut})
            if (*b)
            {
                cudaFree(*b);
                *b = nullptr;
            }
        if (p.sem)
        {
            cudaDestroyExternalSemaphore(p.sem);
            p.sem = nullptr;
        }
        for (cudaExternalMemory_t* em : {&p.emIn, &p.emMv, &p.emOut})
            if (*em)
            {
                cudaDestroyExternalMemory(*em);
                *em = nullptr;
            }
        if (p.made && g_fsrupK[nr.fuKind].destroyI)
            g_fsrupK[nr.fuKind].destroyI(k);
        p.made = false;
    }
    nr.fuPasses = 1;
    if (nr.dFu0Out)
    {
        cudaFree(nr.dFu0Out);
        nr.dFu0Out = nullptr;
    }
    if (nr.fu0EmOut)
    {
        cudaDestroyExternalMemory(nr.fu0EmOut);
        nr.fu0EmOut = nullptr;
    }
    for (uint8_t** p : {&nr.dFuIn, &nr.dFuMv, &nr.dFuOut})
        if (*p)
        {
            cudaFree(*p);
            *p = nullptr;
        }
    if (nr.fuSem)
    {
        cudaDestroyExternalSemaphore(nr.fuSem);
        nr.fuSem = nullptr;
    }
    for (cudaExternalMemory_t* em : {&nr.fuEmIn, &nr.fuEmMv, &nr.fuEmOut})
        if (*em)
        {
            cudaDestroyExternalMemory(*em);
            *em = nullptr;
        }
    if (nr.fuMade && g_fsrupK[nr.fuKind].destroy)
        g_fsrupK[nr.fuKind].destroy();
    nr.fuMade = false;
    nativeOfVecFree(nr.fuOf);
}
