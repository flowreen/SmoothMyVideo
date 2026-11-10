// part of smv-live-native.inl (included there, in place); not a standalone translation unit
// AMD FSR frame generation (fsrfg) runs on the FRUC route below with its own bridge: these four take over FRUC's setup,
// pair, tween and free when nr.fsrfg (defined after nativeFrucFree)
static bool nativeFsrfgSetup(NativeRife& nr);
static bool nativeFsrfgPair(NativeRife& nr, const float* dCur, uint32_t nTween);
static bool nativeFsrfgTween(NativeRife& nr, double t);
static void nativeFsrfgRelease(NativeRife& nr);

// NVIDIA Smooth Motion (fruc): the shipped nvoffruc_bridge.dll,
// loaded once per process by full path from the folder the handoff named (NvOFFRUC.dll and its
// cudart64_110.dll are user-installed beside it; the bridge loads NvOFFRUC.dll signature-checked
// itself). Its state is process-global (up to 16 FRUC instances), so a
// session creates instance 0 at setup, the others at the first pair that needs them, and
// destroys them all in nativeFree. The bridge runs each FRUC instance in its own CUDA context and brackets
// every call with cuCtxSynchronize on the caller's context (its SYNC FENCE notes), so the
// packs queued on nr.stream have landed when it copies, and its output has landed on return.
struct FrucBridge
{
    HMODULE mod = nullptr;
    const char* (*lastError)() = nullptr;
    int (*create)(unsigned, unsigned) = nullptr;
    int (*interpolate)(void*, void*, void*, double, int*) = nullptr;
    // the feed-once call: mode 0 = prime prev + feed cur, 1 = prev was fed
    // last, feed cur only, 2 = the same pair again; an older bridge lacks it (interpolate then)
    int (*step)(void*, void*, void*, double, int, int*) = nullptr;
    void (*destroy)() = nullptr;
    // the instance calls: up to 16 FRUC instances in the one module, each
    // with its own context; absent = an older bridge = one instance
    int (*createI)(int, unsigned, unsigned) = nullptr;
    int (*stepI)(int, void*, void*, void*, double, int, int*) = nullptr;
};
static FrucBridge g_fruc;

// one worker per extra instance (1..kFrDepth - 1): the compute thread hands it one step at a time and
// collects the result in tween order. The worker binds the host's device first, so the bridge's
// entry fence (cuCtxSynchronize on the caller's context) covers the host's packs and unpacks.
// Instance 0's worker serves the pipelined route instead: a job of levels > 0 = the next pair's levels 1..levels
// (nativeFrucSpecLevels), its calls, ms and counters kept here until the compute thread's join adds them
struct FrucWorker
{
    std::thread th;
    std::mutex m;
    std::condition_variable cv;
    bool job = false, done = true, quit = false;
    void* a = nullptr;
    void* b = nullptr;
    void* out = nullptr;
    double t = 0.0;
    int mode = 0, rc = 0, rep = 0;
    std::string err;
    int levels = 0;
    NativeRife* nr = nullptr;
    uint32_t calls = 0;
    uint64_t nodeCalls = 0, repeats = 0, primed = 0;
    double ms = 0.0;
};
// never destroyed: no std::thread destructor runs at process exit (nativeFrucFree joins them)
static FrucWorker* const g_frW = new FrucWorker[kFrDepth];
static int nativeFrucSpecLevels(NativeRife& nr, FrucWorker& w);

// the pipelined route's level gate: the deepest level of the shown pair the compute thread has made (kFrDepth once it
// has no more to make), so instance 0's worker starts a level of the next pair only after the shown pair's same level
// (one instance a level: no instance fed by two threads or out of time order when a ladder deepens)
static std::mutex g_frLagM;
static std::condition_variable g_frLagCv;
static int g_frLagDone = kFrDepth;
// each instance's calls in flight (nativeFrucNode): a second thread entering one logs `entered by two threads` once a
// session; SMV_FRUC_PIPE_GATE=0 = the worker never waits on the gate (that line's trigger, a test lever)
static std::atomic<int> g_frInUse[kFrDepth];
static std::atomic<bool> g_frInUseLogged{false};
static bool frucPipeGateOn()
{
    static const bool on = [] {
        char v[8] = "";
        return !(GetEnvironmentVariableA("SMV_FRUC_PIPE_GATE", v, sizeof(v)) > 0 && v[0] == '0');
    }();
    return on;
}
static void frucLagDone(int l)
{
    {
        std::lock_guard<std::mutex> lk(g_frLagM);
        g_frLagDone = (std::max)(g_frLagDone, l);
    }
    g_frLagCv.notify_all();
}

// the levels instance 0's worker may take of a pipelined pair: SMV_FRUC_PIPE_K caps them (1 = the level-1 call alone)
static int frucPipeK()
{
    static const int k = [] {
        char v[8] = "";
        return GetEnvironmentVariableA("SMV_FRUC_PIPE_K", v, sizeof(v)) > 0 ? (std::max)(1, atoi(v)) : kFrDepth;
    }();
    return k;
}

static void frucWorkerLoop(int i, int dev)
{
    FrucWorker& w = g_frW[i];
    const bool bound = cudaSetDevice(dev) == cudaSuccess && cudaFree(nullptr) == cudaSuccess;
    std::unique_lock<std::mutex> lk(w.m);
    for (;;)
    {
        w.cv.wait(lk, [&] { return w.job || w.quit; });
        if (w.quit)
            return;
        w.job = false;
        void* a = w.a;
        void* b = w.b;
        void* out = w.out;
        const double t = w.t;
        const int mode = w.mode;
        const int levels = w.levels;
        NativeRife* nr = w.nr;
        lk.unlock();
        int rep = 0, rc = -100;
        std::string err = "worker CUDA bind failed";
        if (bound && levels > 0)
        {
            rc = nativeFrucSpecLevels(*nr, w); // its calls, counters and error text go to w
            err = rc ? w.err : std::string();
        }
        else if (bound)
        {
            rc = g_fruc.stepI(i, a, b, out, t, mode, &rep);
            err = rc ? g_fruc.lastError() : "";
        }
        lk.lock();
        w.rc = rc;
        w.rep = rep;
        w.err = std::move(err);
        w.done = true;
        w.cv.notify_all();
    }
}

static bool nativeFrucLoad(const std::string& dir)
{
    if (g_fruc.mod)
        return true;
    const std::wstring wdir = utf8ToWide(dir);
    const std::wstring p = wdir + L"\\nvoffruc_bridge.dll";
    HMODULE m = LoadLibraryExW(p.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!m)
    {
        LOG("native: fruc: nvoffruc_bridge.dll did not load from %s (err %lu)\n", dir.c_str(), GetLastError());
        return false;
    }
    g_fruc.lastError = (const char* (*)())GetProcAddress(m, "nvoffruc_last_error");
    g_fruc.create = (int (*)(unsigned, unsigned))GetProcAddress(m, "nvoffruc_create");
    g_fruc.interpolate = (int (*)(void*, void*, void*, double, int*))GetProcAddress(m, "nvoffruc_interpolate");
    g_fruc.destroy = (void (*)())GetProcAddress(m, "nvoffruc_destroy");
    g_fruc.step = (int (*)(void*, void*, void*, double, int, int*))GetProcAddress(m, "nvoffruc_step");
    g_fruc.createI = (int (*)(int, unsigned, unsigned))GetProcAddress(m, "nvoffruc_create_i");
    g_fruc.stepI = (int (*)(int, void*, void*, void*, double, int, int*))GetProcAddress(m, "nvoffruc_step_i");
    if (!g_fruc.step)
        g_fruc.createI = nullptr, g_fruc.stepI = nullptr;
    if (!g_fruc.lastError || !g_fruc.create || !g_fruc.interpolate || !g_fruc.destroy)
    {
        LOG("native: fruc: nvoffruc_bridge.dll lacks an expected export\n");
        FreeLibrary(m);
        return false;
    }
    g_fruc.mod = m;
    return true;
}

// the FRUC instance at the padded model size (pw x ph) and the
// BGRA8 surfaces; the calling thread has the device bound
static bool nativeFrucSetup(NativeRife& nr)
{
    if (nr.fsrfg)
        return nativeFsrfgSetup(nr);
    if (!nativeFrucLoad(nr.frucDir))
        return false;
    const int rc = g_fruc.create((unsigned)nr.pw, (unsigned)nr.ph);
    // the bridge points the DLL search at its folder while it loads NvOFFRUC.dll (its
    // secure_load); give the process its default search back
    SetDllDirectoryW(nullptr);
    if (rc != 0)
    {
        LOG("native: fruc: nvoffruc_create %dx%d failed: %s (rc %d)\n", nr.pw, nr.ph, g_fruc.lastError(), rc);
        return false;
    }
    nr.frCreated = true;
    const size_t bytes = (size_t)nr.pw * nr.ph * 4, plane = (size_t)nr.ph * nr.pw;
    for (auto& s : nr.dFrSurf)
        NCHK(cudaMalloc((void**)&s, bytes), "alloc fruc surface");
    NCHK(cudaMalloc((void**)&nr.dFrOutB, bytes), "alloc fruc output");
    NCHK(cudaMalloc((void**)&nr.dFrOut, 3 * plane * sizeof(float)), "alloc fruc tween");
    {
        char ev[8] = {};
        nr.frSdrOn = nr.encPost && !(GetEnvironmentVariableA("SMV_FRUC_SDR_PAIRS", ev, sizeof(ev)) > 0 && ev[0] == '0');
    }
    if (nr.frSdrOn)
    {
        NCHK(cudaMalloc((void**)&nr.dFrRange, sizeof(int)), "alloc fruc range flag");
        NCHK(cudaHostAlloc((void**)&nr.hFrRange, sizeof(int), cudaHostAllocDefault), "alloc fruc range readback");
    }
    for (int i = 0; i < 3; i++)
        nr.frIn[i] = nr.frSurfSdr[i] = false;
    nr.frPairSdr = false;
    nr.frModeLogs = 0;
    nr.frPrev = nr.frLast = nr.frA = nr.frB = -1;
    for (int i = 0; i < kFrDepth; i++)
    {
        nr.frFed[i] = -1;
        nr.frFirst[i] = false;
        nr.frCalls[i] = 0;
    }
    nr.dFrOutBI[0] = nr.dFrOutB;
    nr.frInst = 1;
    nr.frPlanN = nr.frPlanK = 0;
    // recursive midpoints are the default wherever the bridge has the instance calls, one
    // instance per tree level, up to kFrDepth on either route; SMV_FRUC_MIDPOINTS=0 keeps the direct-t
    // scheme (parallel instances up to 4 live, 1 offline: offline FRUC is paced by the encoder,
    // where they measured 0.99x / ~0.94x at x3 / x5 1080p), SMV_FRUC_DEPTH=1..9 the snap depth.
    // SMV_FRUC_INSTANCES caps the instances on either scheme (1 = one instance, 5 = the tree to x32); the extra ones
    // are created by the first pair that needs them (nativeFrucGrow)
    nr.frMp = false;
    nr.frInstMax = 1;
    if (g_fruc.stepI)
    {
        char ev[8] = {};
        nr.frMp = !(GetEnvironmentVariableA("SMV_FRUC_MIDPOINTS", ev, sizeof(ev)) > 0 && ev[0] == '0');
        nr.frMpCap = g_offline ? 3 : 2;
        if (GetEnvironmentVariableA("SMV_FRUC_DEPTH", ev, sizeof(ev)) > 0)
            nr.frMpCap = (std::max)(1, (std::min)(kFrDepth, atoi(ev)));
        nr.frInstMax = nr.frMp ? kFrDepth : g_offline ? 1 : 4;
        if (GetEnvironmentVariableA("SMV_FRUC_INSTANCES", ev, sizeof(ev)) > 0)
            nr.frInstMax = (std::max)(1, (std::min)(kFrDepth, atoi(ev)));
    }
    nr.frMpL = 0;
    nr.frSerial = 0;
    nr.frNodeCalls = 0;
    for (bool& ok : nr.frNodeOk)
        ok = false;
    for (uint64_t& k : nr.frLastKey)
        k = 0;
    nr.frPipePrev = nr.frLagHave = nr.frLagSdr = nr.frLagStatic = nr.frCurStatic = false;
    nr.frSpecRun = nr.frRepacked = false;
    nr.frLagA = nr.frLagB = nr.frKeep = -1;
    nr.frLagL = nr.frPipeDepth = 0;
    nr.frLagSerial = nr.frPipeGroups = 0;
    for (bool& ok : nr.frLagOk)
        ok = false;
    nr.grFrCalls = nr.grFrPreCalls = nr.grFrWkCalls = 0;
    nr.grPriceSkip = 0;
    nr.grFrCallMs = nr.grFrPreMs = nr.grFrChainMs = nr.gFrCallMs = nr.gFrStoreMs = nr.gFrBaseMs = nr.gFrL1Ratio =
        nr.gFrL1BaseMs = nr.grFrWkMs = nr.gFrPipeMainMs = nr.gFrPipeWkMs = nr.gFrPipeRm = nr.gFrPipeRw =
            nr.gFrCallEntryMs = nr.gFgPreMs = nr.gFgOfMs = nr.gFgL1Ms = 0.0;
    nr.grPipeLearn = 0;
    nr.grPiped = nr.grLagArmed = false;
    for (int k = 0; k < 2; k++)
        nr.grL1Armed[k] = nr.grOfArmed[k] = false;
    nr.gFrCallUs.store(0);
    nr.gFrStoreUs.store(0);
    nr.gFrBaseUs.store(0);
    nr.gFrPipeMainUs.store(0);
    nr.gFrPipeWkUs.store(0);
    nr.gFgPreUs.store(0);
    nr.gFgOfUs.store(0);
    nr.gFgL1Us.store(0);
    g_frInUseLogged = false;
    if (nr.frMp)
        LOG("native: fruc session %dx%d (model %dx%d), NvOFFRUC through nvoffruc_bridge.dll, 8-bit BGRA, feed-once, recursive midpoints (depth cap %d)\n",
            nr.pw, nr.ph, nr.w, nr.h, nr.frMpCap);
    else
        LOG("native: fruc session %dx%d (model %dx%d), NvOFFRUC through nvoffruc_bridge.dll, 8-bit BGRA%s, up to %d instance%s\n",
            nr.pw, nr.ph, nr.w, nr.h, g_fruc.step ? ", feed-once" : "", nr.frInstMax, nr.frInstMax > 1 ? "s" : "");
    if (nr.encPost)
        LOG("native: fruc: HDR planes: %s\n",
            nr.frSdrOn
                ? "frame pairs inside the SDR range go to FRUC as their SDR view, any other pair as the 8-bit HDR codes"
                : "every pair as the 8-bit HDR codes (SMV_FRUC_SDR_PAIRS=0)");
    return true;
}

// the extra FRUC instances up to `want` (instance 0 is the session's own), each with its output
// buffer and its worker; a failed create (VRAM, most likely) keeps the instances made so far
static void nativeFrucGrow(NativeRife& nr, int want)
{
    want = (std::min)(want, (int)(sizeof(nr.dFrOutBI) / sizeof(nr.dFrOutBI[0])));
    const int64_t t0 = nowQpc100();
    const int had = nr.frInst;
    const size_t bytes = (size_t)nr.pw * nr.ph * 4;
    // SMV_FRUC_INST_FAILAT=i (harness lever): instance i fails to create, the trigger of the
    // fallback below
    char ev[8] = {};
    const int failAt = GetEnvironmentVariableA("SMV_FRUC_INST_FAILAT", ev, sizeof(ev)) > 0 ? atoi(ev) : -1;
    for (int i = (std::max)(nr.frInst, 0); i < want; i++)
    {
        const int rc = i == failAt ? -99 : g_fruc.createI(i, (unsigned)nr.pw, (unsigned)nr.ph);
        if (rc != 0 || cudaMalloc((void**)&nr.dFrOutBI[i], bytes) != cudaSuccess)
        {
            LOG("native: fruc: instance %d not created (%s, rc %d), staying at %d\n", i,
                rc == -99 ? "SMV_FRUC_INST_FAILAT"
                : rc      ? g_fruc.lastError()
                          : "output alloc failed",
                rc, nr.frInst);
            if (nr.dFrOutBI[i])
            {
                cudaFree(nr.dFrOutBI[i]);
                nr.dFrOutBI[i] = nullptr;
            }
            nr.frInstMax = nr.frInst;
            break;
        }
        FrucWorker& w = g_frW[i];
        w.job = w.quit = false;
        w.done = true;
        w.th = std::thread(frucWorkerLoop, i, nr.dev);
        nr.frInst = i + 1;
    }
    if (nr.frInst > had)
    {
        nr.grPriceSkip = 2;
        LOG("native: fruc: %d instances (%d new in %.0f ms)\n", nr.frInst, nr.frInst - had,
            (double)(nowQpc100() - t0) / 10000.0);
    }
}

// wait until every worker is idle (a round the caller abandoned, a session's end); instance 0's worker exists once a
// pipelined group ran (the level gate opened first: a failed group never reached its join)
static void nativeFrucDrain(NativeRife& nr)
{
    frucLagDone(kFrDepth);
    for (int i = 0; i < nr.frInst; i++)
    {
        FrucWorker& w = g_frW[i];
        if (i == 0 && !w.th.joinable())
            continue;
        std::unique_lock<std::mutex> lk(w.m);
        w.cv.wait(lk, [&] { return w.done; });
    }
}

// a pipelined group's job on instance 0's worker (made by the first such group): the next pair's levels 1..levels
static void nativeFrucSpecStart(NativeRife& nr, int levels)
{
    FrucWorker& w = g_frW[0];
    if (!w.th.joinable())
    {
        w.job = w.quit = false;
        w.done = true;
        w.th = std::thread(frucWorkerLoop, 0, nr.dev);
    }
    std::lock_guard<std::mutex> lk(w.m);
    w.levels = levels;
    w.nr = &nr;
    w.calls = 0;
    w.nodeCalls = w.repeats = w.primed = 0;
    w.ms = 0.0;
    w.err.clear();
    w.done = false;
    w.job = true;
    w.cv.notify_all();
    nr.frSpecRun = true;
}

// its result: the gate opened (a level the shown pair no longer needs), the job waited for, its counters added and its
// calls kept for the pipelined price (its nodes and its instances' keys were set by nativeFrucNode)
static bool nativeFrucSpecJoin(NativeRife& nr)
{
    if (!nr.frSpecRun)
        return true;
    nr.frSpecRun = false;
    frucLagDone(kFrDepth);
    FrucWorker& w = g_frW[0];
    std::unique_lock<std::mutex> lk(w.m);
    w.cv.wait(lk, [&] { return w.done; });
    nr.frNodeCalls += w.nodeCalls;
    nr.frRepeats += w.repeats;
    nr.frPrimed += w.primed;
    nr.grFrWkCalls = w.calls;
    nr.grFrWkMs = w.ms;
    if (w.rc != 0)
    {
        LOG("native: fruc: the next pair's levels 1..%d on instance 0's worker (pipelined) failed: %s (rc %d)\n",
            w.levels, w.err.c_str(), w.rc);
        nr.die("fruc midpoint failed");
        return false;
    }
    return true;
}

// the depth of a pair's midpoint tree: the smallest L that holds every t as a node k / 2^L (x2 / x4 /
// .. / x512 are exact on both routes); a pair with a t that is no node takes frMpCap (its nearest node
// there), offline at least its deepest node's depth. Live never deepens such a pair: a grid phase that
// lands on a deep node by chance would cost the whole deep tree for the pair's few snapped tweens. A node is
// within 1e-6 (a ladder's k / 2^L is exact in float), so a phase near a deep node by chance is no node
static int nativeFrucDepth(const NativeRife& nr, const double* ts, uint32_t n)
{
    int L = 1;
    bool off = false;
    for (uint32_t i = 0; i < n; i++)
    {
        int d = 1;
        for (; d <= kFrDepth; d++)
        {
            const double x = std::ldexp(ts[i], d);
            if (fabs(x - floor(x + 0.5)) < 1e-6)
                break;
        }
        if (d <= kFrDepth)
            L = (std::max)(L, d);
        else
            off = true;
    }
    if (off)
        L = g_offline ? (std::max)(L, nr.frMpCap) : nr.frMpCap;
    return (std::min)(L, kFrDepth);
}

// the pair's tweens in the order the caller asks for them (every nativeFrucTween call then takes
// the next one); fewer than two, or one instance, = the one-instance path
static void nativeFrucPlan(NativeRife& nr, const double* ts, uint32_t n)
{
    if (nr.frMp)
    {
        // recursive midpoints: only the pair's depth; the first nativeFrucTween builds the tree
        nr.frMpL = nativeFrucDepth(nr, ts, n);
        return;
    }
    nativeFrucDrain(nr);
    nr.frPlanN = nr.frPlanK = 0;
    if (nr.frInstMax < 2 || n < 2 || n > 64)
        return;
    const int want = (std::min)((int)n, nr.frInstMax);
    if (want > nr.frInst)
        nativeFrucGrow(nr, want);
    if (nr.frInst < 2)
        return;
    memcpy(nr.frPlan, ts, n * sizeof(double));
    nr.frPlanN = n;
}

// the pair: the new frame packed into a surface that holds neither the previous frame nor the
// last tweened pair's end. Only when this group has tweens: a pair that
// does not continue the last interpolated one first gets one dropped warp at 0.5 of the
// skipped pair (last end, previous frame), so the OFA hints stay consecutive
// one frame's planes into surface n: the SDR view (sdr, k_packBgraSdr) or the planes as stored (true BGRA)
static bool nativeFrucPack(NativeRife& nr, const float* src, int n, bool sdr)
{
    int ps = nr.ph * nr.pw, f32 = 0;
    void* a[] = {(void*)&src, &f32, &ps, &nr.pw, &nr.pw, &nr.ph, &nr.dFrSurf[n]};
    void* b[] = {(void*)&src, &ps, &nr.pw, &nr.pw, &nr.ph, &nr.encPost, &nr.encWhite, &nr.dFrSurf[n]};
    const CUfunction f = sdr ? nr.fPackBgraSdr : nr.fPackBgraRgb;
    if (cuLaunchKernel(f, (nr.pw + 15) / 16, (nr.ph + 15) / 16, 1, 16, 16, 1, 0, (CUstream)nr.stream, sdr ? b : a,
                       nullptr) != CUDA_SUCCESS)
    {
        nr.die("packBgra (fruc) launch failed");
        return false;
    }
    nr.frSurfSdr[n] = sdr;
    for (int& fed : nr.frFed)
        if (fed == n)
            fed = -1; // the frame FRUC was fed last is gone from its surface
    return true;
}

static bool nativeFrucPair(NativeRife& nr, const float* dCur, uint32_t nTween)
{
    if (nr.fsrfg)
        return nativeFsrfgPair(nr, dCur, nTween);
    nativeFrucDrain(nr); // no worker still reads a surface this pair may repack
    // a pipelined group spares the shown pair's first frame (frKeep) where the others spare the last fed one
    int n = 0;
    while (n == nr.frPrev || n == (nr.frKeep >= 0 ? nr.frKeep : nr.frLast))
        n++;
    nr.frRepacked = false;
    // HDR planes: the new frame's range (one flag read back); the pair takes the SDR view only when both frames stay
    // inside the SDR range. A previous frame's surface in the other encoding is packed again from its planes (dX)
    // and every instance primes it again (its node key would still match)
    bool in = false;
    if (nr.frSdrOn)
    {
        cudaStream_t st = nr.stream;
        int ps = nr.ph * nr.pw;
        float tol = 2e-3f;
        void* ar[] = {(void*)&dCur, &ps, &nr.pw, &nr.pw, &nr.ph, &nr.encPost, &nr.encWhite, &tol, (void*)&nr.dFrRange};
        if (cudaMemsetAsync(nr.dFrRange, 0, sizeof(int), st) != cudaSuccess ||
            cuLaunchKernel(nr.fHdrRange, (nr.pw + 15) / 16, (nr.ph + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, ar,
                           nullptr) != CUDA_SUCCESS ||
            cudaMemcpyAsync(nr.hFrRange, nr.dFrRange, sizeof(int), cudaMemcpyDeviceToHost, st) != cudaSuccess ||
            cudaStreamSynchronize(st) != cudaSuccess)
        {
            nr.die("fruc range test failed");
            return false;
        }
        in = *nr.hFrRange == 0;
    }
    const bool sdr = in && (nr.frPrev < 0 || nr.frIn[nr.frPrev]);
    if (!nativeFrucPack(nr, dCur, n, sdr))
        return false;
    nr.frIn[n] = in;
    if (nr.frPrev >= 0 && nr.frSurfSdr[nr.frPrev] != sdr)
    {
        if (!nativeFrucPack(nr, nativeXPrev(nr), nr.frPrev, sdr))
            return false;
        nr.frRepacked = true;
        for (uint64_t& k : nr.frLastKey)
            k = 0;
    }
    if (nr.frSdrOn && nTween && nr.frPrev >= 0 && nr.frModeLogs < 4 && (nr.frModeLogs == 0 || sdr != nr.frPairSdr))
    {
        LOG("native: fruc: %s\n", sdr ? "frame pairs inside the SDR range, their SDR view"
                                      : "a frame pair outside the SDR range, the 8-bit HDR codes");
        nr.frModeLogs++;
    }
    nr.frPairSdr = sdr;
    nr.frA = nr.frPrev;
    nr.frB = n;
    nr.frPrev = n;
    nr.frPlanN = nr.frPlanK = 0; // the caller plans this pair's tweens after this (nativeFrucPlan)
    // recursive midpoints: this frame's serial names it in the node keys; the pair's tree starts empty
    nr.frSerial++;
    nr.frMpL = 0;
    nr.frMpBuilt = false;
    for (bool& ok : nr.frNodeOk)
        ok = false;
    if (!nTween || nr.frA < 0)
        return true;
    if (g_fruc.step)
    {
        // feed-once bridge: no priming warp; an instance's first tween of the pair says whether
        // it was fed frA last (mode 1) or must prime it (mode 0), its other tweens reuse the pair (mode 2)
        for (bool& f : nr.frFirst)
            f = true;
        nr.frLast = nr.frB;
        return true;
    }
    if (nr.frLast >= 0 && nr.frLast != nr.frA)
    {
        int rep = 0;
        const int rc = g_fruc.interpolate(nr.dFrSurf[nr.frLast], nr.dFrSurf[nr.frA], nr.dFrOutB, 0.5, &rep);
        if (rc != 0)
        {
            LOG("native: fruc: priming warp failed: %s (rc %d)\n", g_fruc.lastError(), rc);
            nr.die("fruc priming failed");
            return false;
        }
        nr.frPrimed++;
    }
    nr.frLast = nr.frB;
    return true;
}

// one tween at t into dFrOut (3, ph, pw) (t is a
// double, the bridge's own type: offline --fps hands its double fraction over unrounded)
// the feed-once mode of instance i's next call on this pair; a mode 0 after the instance's first
// call is a pair that does not continue the one it was fed last (a repack of the fed surface
// clears frFed, so frFed alone would miss those primes)
static int nativeFrucMode(NativeRife& nr, int i)
{
    const int mode = !nr.frFirst[i] ? 2 : (nr.frFed[i] >= 0 && nr.frFed[i] == nr.frA) ? 1 : 0;
    if (mode == 0 && nr.frCalls[i] > 0)
        nr.frPrimed++;
    return mode;
}

static void nativeFrucFedOk(NativeRife& nr, int i)
{
    nr.frFirst[i] = false;
    nr.frFed[i] = nr.frB;
    nr.frCalls[i]++;
}

// a pair's midpoint tree: this group's pair, or the previous one a pipelined group shows (frLag*)
struct FrTree
{
    uint64_t serial; // its second frame's serial
    int a, b;        // its frames' surfaces
    uint8_t** node;  // [kFrGrid]
    bool* ok;        // [kFrGrid]
};
static FrTree nativeFrucCur(NativeRife& nr)
{
    return {nr.frSerial, nr.frA, nr.frB, nr.dFrNode, nr.frNodeOk};
}
static FrTree nativeFrucLag(NativeRife& nr)
{
    return {nr.frLagSerial, nr.frLagA, nr.frLagB, nr.dFrLagNode, nr.frLagOk};
}

// recursive midpoints: the key of node k / 2^L of a pair (k = 0 / 2^L = the pair's first /
// second frame): a real frame = its serial, an interior node = the pair's first serial and its
// position on the kFrGrid grid; an instance fed the left parent of a node last feeds only the right one
static uint64_t nativeFrucKey(const FrTree& tr, int k, int L)
{
    while (L > 0 && !(k & 1))
    {
        k >>= 1;
        L--;
    }
    const uint64_t a = tr.serial - 1;
    if (L == 0)
        return (k ? tr.serial : a) * kFrGrid + 1;
    return a * kFrGrid + ((uint64_t)k << (kFrDepth - L)) + 1;
}

// node k / 2^L of a pair's midpoint tree, once per pair: FRUC at t = 0.5 between the node's two
// parents (computed first) on instance L - 1, mode 1 when that instance was fed the left parent
// last, else mode 0 (prime the left parent, feed the right one). On instance 0's worker (w) the calls, counters and a
// failure's text go to w (the compute thread runs the other instances meanwhile; its join adds them)
static uint8_t* nativeFrucNode(NativeRife& nr, const FrTree& tr, int k, int L, FrucWorker* w = nullptr)
{
    while (L > 0 && !(k & 1))
    {
        k >>= 1;
        L--;
    }
    if (L <= 0)
        return nr.dFrSurf[k ? tr.b : tr.a];
    const int p = k << (kFrDepth - L);
    if (tr.ok[p])
        return tr.node[p];
    uint8_t* left = nativeFrucNode(nr, tr, k - 1, L, w);
    uint8_t* right = left ? nativeFrucNode(nr, tr, k + 1, L, w) : nullptr;
    if (!right)
        return nullptr;
    if (!tr.node[p] && cudaMalloc((void**)&tr.node[p], (size_t)nr.pw * nr.ph * 4) != cudaSuccess)
    {
        tr.node[p] = nullptr;
        if (w)
            w->err = "fruc midpoint buffer alloc failed";
        else
            nr.die("fruc midpoint buffer alloc failed");
        return nullptr;
    }
    const int i = L - 1;
    const int mode = nr.frLastKey[i] == nativeFrucKey(tr, k - 1, L) ? 1 : 0;
    if (mode == 0 && nr.frCalls[i] > 0)
        (w ? w->primed : nr.frPrimed)++;
    int rep = 0;
    if (g_frInUse[i].fetch_add(1) && !g_frInUseLogged.exchange(true))
        LOG("native: fruc: instance %d entered by two threads (pipelined level %d)\n", i, L);
    const int64_t c0 = nowQpc100();
    const int rc = g_fruc.stepI(i, left, right, tr.node[p], 0.5, mode, &rep);
    const double ms = (double)(nowQpc100() - c0) / 10000.0;
    g_frInUse[i].fetch_sub(1);
    if (w)
    {
        w->calls++;
        w->ms += ms;
    }
    else
    {
        nr.grFrCalls++;
        nr.grFrCallMs += ms;
    }
    if (rc != 0)
    {
        if (w)
        {
            char e[256];
            snprintf(e, sizeof(e), "midpoint %d/%d on instance %d: %s (bridge rc %d)", k, 1 << L, i, g_fruc.lastError(),
                     rc);
            w->err = e;
            return nullptr;
        }
        LOG("native: fruc: midpoint %d/%d on instance %d failed: %s (rc %d)\n", k, 1 << L, i, g_fruc.lastError(), rc);
        nr.die("fruc midpoint failed");
        return nullptr;
    }
    nr.frLastKey[i] = nativeFrucKey(tr, k + 1, L);
    nr.frCalls[i]++;
    (w ? w->nodeCalls : nr.frNodeCalls)++;
    if (rep)
        (w ? w->repeats : nr.frRepeats)++;
    tr.ok[p] = true;
    return tr.node[p];
}

// the levels of the shown pair already made (each level's every node), the level gate's start
static int nativeFrucLagMade(const NativeRife& nr)
{
    int l = 0;
    for (int L = 1; L <= kFrDepth; L++)
    {
        for (int k = 1; k < (1 << L); k += 2)
            if (!nr.frLagOk[k << (kFrDepth - L)])
                return l;
        l = L;
    }
    return l;
}

// instance 0's worker on a pipelined group: the next pair's levels 1..w.levels (instances 0..levels - 1), level by
// level as nativeFrucTween builds a tree, each level after the level gate passed it (the shown pair's same level made)
static int nativeFrucSpecLevels(NativeRife& nr, FrucWorker& w)
{
    const FrTree tr = nativeFrucCur(nr);
    for (int l = 1; l <= w.levels; l++)
    {
        if (frucPipeGateOn())
        {
            std::unique_lock<std::mutex> lk(g_frLagM);
            g_frLagCv.wait(lk, [&] { return g_frLagDone >= l; });
        }
        for (int k = 1; k < (1 << l); k += 2)
            if (!nativeFrucNode(nr, tr, k, l, &w))
                return -1;
    }
    return 0;
}

// a FRUC output (true BGRA8) into dFrOut (3, ph, pw) as (R, G, B) planes; an SDR pair's tween back to the HDR
// codes (k_unpackBgraSdr)
static bool nativeFrucUnpack(NativeRife& nr, uint8_t* src, bool sdr)
{
    void* a[] = {&src, &nr.pw, &nr.ph, &nr.dFrOut};
    void* b[] = {&src, &nr.pw, &nr.ph, &nr.encPost, &nr.encWhite, &nr.dFrOut};
    const CUfunction f = sdr ? nr.fUnpackBgraSdr : nr.fUnpackBgraRgb;
    if (cuLaunchKernel(f, (nr.pw + 15) / 16, (nr.ph + 15) / 16, 1, 16, 16, 1, 0, (CUstream)nr.stream, sdr ? b : a,
                       nullptr) != CUDA_SUCCESS)
    {
        nr.die("unpackBgra (fruc) launch failed");
        return false;
    }
    return true;
}

static bool nativeFrucTween(NativeRife& nr, double t)
{
    if (nr.fsrfg)
        return nativeFsrfgTween(nr, t);
    if (nr.frMp)
    {
        // recursive midpoints: the pair's WHOLE tree to its depth first, level by level, left to
        // right, so level L's instance sees the continuous stream A, 2 / 2^L, 4 / 2^L .. B of new
        // frames. A node computed alone needs a prime whose optical flow spans the skipped nodes,
        // NvOFFRUC seeds the next flow from it, and that damaged every such node (up to 28 % of a
        // held frame). Then the nearest node (never an end frame); a failed instance
        // create caps the depth at the instances made
        if (!nr.frMpL)
            nr.frMpL = nativeFrucDepth(nr, &t, 1);
        // a failed create lowers frInstMax (nativeFrucGrow), so it is not retried every pair
        if (nr.frInst < nr.frMpL && nr.frInst < nr.frInstMax)
            nativeFrucGrow(nr, (std::min)(nr.frMpL, nr.frInstMax));
        if (nr.frInst < nr.frMpL)
            nr.frMpL = nr.frInst;
        const FrTree tr = nativeFrucCur(nr);
        if (!nr.frMpBuilt)
        {
            for (int l = 1; l <= nr.frMpL; l++)
                for (int k = 1; k < (1 << l); k += 2)
                    if (!nativeFrucNode(nr, tr, k, l))
                        return false;
            nr.frMpBuilt = true;
        }
        const int den = 1 << nr.frMpL;
        const int k = (std::max)(1, (std::min)(den - 1, (int)floor(t * den + 0.5)));
        uint8_t* node = nativeFrucNode(nr, tr, k, nr.frMpL);
        if (!node)
            return false;
        nr.frTweens++;
        if (!nativeFrucUnpack(nr, node, nr.frPairSdr))
            return false;
        return true;
    }
    int rep = 0;
    int rc;
    uint8_t* outB = nr.dFrOutB;
    if (nr.frPlanK < nr.frPlanN && nr.frPlan[nr.frPlanK] != t)
    {
        // the caller asked for a tween its plan did not name: finish what runs, then one instance
        LOG("native: fruc: tween %.4f is not the planned %.4f, one instance for the rest of the pair\n", t,
            nr.frPlan[nr.frPlanK]);
        nativeFrucDrain(nr);
        nr.frPlanN = nr.frPlanK = 0;
    }
    if (nr.frPlanK < nr.frPlanN)
    {
        // parallel instances: tween k runs on instance k % frInst; the first tween of a round
        // hands the round's others to their workers and runs its own here, the others collect
        const uint32_t k = nr.frPlanK++;
        const int inst = (int)(k % (uint32_t)nr.frInst);
        if (inst == 0)
        {
            for (uint32_t j = k + 1; j < nr.frPlanN && j < k + (uint32_t)nr.frInst; j++)
            {
                const int wi = (int)(j - k);
                FrucWorker& w = g_frW[wi];
                std::lock_guard<std::mutex> lk(w.m);
                w.a = nr.dFrSurf[nr.frA];
                w.b = nr.dFrSurf[nr.frB];
                w.out = nr.dFrOutBI[wi];
                w.t = nr.frPlan[j];
                w.mode = nativeFrucMode(nr, wi);
                w.done = false;
                w.job = true;
                w.cv.notify_all();
            }
            rc = g_fruc.stepI(0, nr.dFrSurf[nr.frA], nr.dFrSurf[nr.frB], nr.dFrOutB, t, nativeFrucMode(nr, 0), &rep);
            if (rc == 0)
                nativeFrucFedOk(nr, 0);
            else
                LOG("native: fruc: interpolate failed: %s (rc %d)\n", g_fruc.lastError(), rc);
        }
        else
        {
            FrucWorker& w = g_frW[inst];
            std::unique_lock<std::mutex> lk(w.m);
            w.cv.wait(lk, [&] { return w.done; });
            rc = w.rc;
            rep = w.rep;
            if (rc == 0)
                nativeFrucFedOk(nr, inst);
            else
                LOG("native: fruc: interpolate (instance %d) failed: %s (rc %d)\n", inst, w.err.c_str(), rc);
            outB = nr.dFrOutBI[inst];
        }
        if (rc != 0)
        {
            nativeFrucDrain(nr);
            nr.die("fruc interpolate failed");
            return false;
        }
    }
    else if (g_fruc.step)
    {
        rc = g_fruc.step(nr.dFrSurf[nr.frA], nr.dFrSurf[nr.frB], nr.dFrOutB, t, nativeFrucMode(nr, 0), &rep);
        if (rc == 0)
            nativeFrucFedOk(nr, 0);
    }
    else
        rc = g_fruc.interpolate(nr.dFrSurf[nr.frA], nr.dFrSurf[nr.frB], nr.dFrOutB, t, &rep);
    if (rc != 0)
    {
        LOG("native: fruc: interpolate failed: %s (rc %d)\n", g_fruc.lastError(), rc);
        nr.die("fruc interpolate failed");
        return false;
    }
    if (rep)
        nr.frRepeats++;
    nr.frTweens++;
    return nativeFrucUnpack(nr, outB, nr.frPairSdr);
}

// a pipelined group's pair step: the pair the last group made current becomes the shown one (its levels 1..k, made by
// instance 0's worker, are among its nodes), then the new frame is packed, sparing the shown pair's first
// frame. The shown pair is held (its second frame in every slot) when it was identical, when the last group was not
// pipelined (it showed that pair itself: the route's entry costs one source period's hold) or when its second frame
// was packed again in the other encoding
static bool nativeFrucPipeAdvance(NativeRife& nr, const float* dCur, bool curStatic)
{
    std::swap(nr.dFrNode, nr.dFrLagNode);
    std::swap(nr.frNodeOk, nr.frLagOk);
    nr.frLagSerial = nr.frSerial;
    nr.frLagA = nr.frA;
    nr.frLagB = nr.frB;
    nr.frLagSdr = nr.frPairSdr;
    nr.frLagStatic = nr.frCurStatic;
    nr.frLagHave = nr.frPipePrev && nr.frLagA >= 0 && nr.frLagOk[kFrGrid / 2];
    nr.frLagL = 0;
    nr.frKeep = nr.frLagA;
    const bool ok = nativeFrucPair(nr, dCur, curStatic ? 0 : 1);
    nr.frKeep = -1;
    if (nr.frRepacked)
        nr.frLagHave = false;
    nr.frCurStatic = curStatic;
    nr.frPipePrev = true;
    nr.frPipeGroups++;
    return ok;
}

// the new pair's levels 1..k on instance 0's worker (the next group shows the pair), unless the pair is identical or has
// no first frame yet: k = the depth of this group's tweens less one (x4: the level-1 call alone, x8 two levels, x32
// four), so the worker's 2^k - 1 calls run beside the compute thread's 2^L - 2^k of the shown pair (its deepest
// level); the worker's calls run slower beside them, so depth - 1 is the best split at x8 to x64. The level gate starts
// at the shown pair's levels already made (kFrDepth when it is held: the compute thread makes none)
static bool nativeFrucPipeSpec(NativeRife& nr)
{
    if (nr.frCurStatic || nr.frA < 0)
        return true;
    const int k = (std::max)(1, (std::min)({nr.frPipeDepth - 1, nr.frInst - 1, frucPipeK()}));
    {
        std::lock_guard<std::mutex> lk(g_frLagM);
        g_frLagDone = nr.frLagHave && !nr.frLagStatic ? nativeFrucLagMade(nr) : kFrDepth;
    }
    nativeFrucSpecStart(nr, k);
    return true;
}

// a pipelined group's tween: the shown pair's levels the worker did not make (level by level, as nativeFrucTween) on
// their instances beside instance 0's worker's levels of the new pair, each level opening the gate for its twin; the
// worker joined before the slots' stores; then the nearest node
static bool nativeFrucLagTween(NativeRife& nr, double t)
{
    const FrTree tr = nativeFrucLag(nr);
    if (!nr.frLagL)
    {
        nr.frLagL = (std::min)(nativeFrucDepth(nr, &t, 1), nr.frInst);
        const int64_t c0 = nowQpc100();
        for (int l = 2; l <= nr.frLagL; l++)
        {
            for (int k = 1; k < (1 << l); k += 2)
                if (!nativeFrucNode(nr, tr, k, l))
                    return false;
            frucLagDone(l);
        }
        if (!nativeFrucSpecJoin(nr))
            return false;
        nr.grFrChainMs = (double)(nowQpc100() - c0) / 10000.0;
    }
    const int den = 1 << nr.frLagL;
    const int k = (std::max)(1, (std::min)(den - 1, (int)floor(t * den + 0.5)));
    uint8_t* node = nativeFrucNode(nr, tr, k, nr.frLagL);
    if (!node)
        return false;
    nr.frTweens++;
    return nativeFrucUnpack(nr, node, nr.frLagSdr);
}

static void nativeFrucFree(NativeRife& nr)
{
    if (nr.fsrfg)
        nativeFsrfgRelease(nr); // the bridge's instances and the Optical Flow session; the frames below are shared
    nativeFrucDrain(nr);
    for (int i = 0; i < kFrDepth; i++)
    {
        FrucWorker& w = g_frW[i];
        if (!w.th.joinable())
            continue;
        {
            std::lock_guard<std::mutex> lk(w.m);
            w.quit = true;
            w.cv.notify_all();
        }
        w.th.join();
    }
    if (nr.frCreated)
    {
        char mp[96] = "";
        if (nr.frMp)
            snprintf(mp, sizeof(mp), ", %llu midpoint calls%s", (unsigned long long)nr.frNodeCalls,
                     nr.frPipeGroups ? (", " + std::to_string(nr.frPipeGroups) + " groups pipelined").c_str() : "");
        LOG("native: fruc session: %llu tweens, %llu primed pairs, %llu frame repeats%s, %d instance%s\n",
            (unsigned long long)nr.frTweens, (unsigned long long)nr.frPrimed, (unsigned long long)nr.frRepeats, mp,
            nr.frInst, nr.frInst > 1 ? "s" : "");
        g_fruc.destroy(); // every instance
        nr.frCreated = false;
    }
    for (int i = 1; i < kFrDepth; i++)
        if (nr.dFrOutBI[i])
        {
            cudaFree(nr.dFrOutBI[i]);
            nr.dFrOutBI[i] = nullptr;
        }
    nr.dFrOutBI[0] = nullptr;
    nr.frInst = 1;
    nr.frPlanN = nr.frPlanK = 0;
    for (auto& s : nr.dFrSurf)
        if (s)
        {
            cudaFree(s);
            s = nullptr;
        }
    for (auto& s : nr.dFrNode)
        if (s)
        {
            cudaFree(s);
            s = nullptr;
        }
    for (auto& s : nr.dFrLagNode)
        if (s)
        {
            cudaFree(s);
            s = nullptr;
        }
    for (bool& ok : nr.frNodeOk)
        ok = false;
    for (bool& ok : nr.frLagOk)
        ok = false;
    nr.frLagHave = nr.frPipePrev = false;
    if (nr.dFrOutB)
    {
        cudaFree(nr.dFrOutB);
        nr.dFrOutB = nullptr;
    }
    if (nr.dFrOut)
    {
        cudaFree(nr.dFrOut);
        nr.dFrOut = nullptr;
    }
    if (nr.dFrRange)
    {
        cudaFree(nr.dFrRange);
        nr.dFrRange = nullptr;
    }
    if (nr.hFrRange)
    {
        cudaFreeHost(nr.hFrRange);
        nr.hFrRange = nullptr;
    }
    nr.frSdrOn = nr.frPairSdr = false;
}
