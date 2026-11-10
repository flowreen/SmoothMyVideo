// part of smv-live-native.inl (included there, in place); not a standalone translation unit
// The session's full handoff answer: the base answer plus, for live Restore (every backend:
// the host restores the captured frame before any model reads it), the Restore model on the host's kernels.
static bool nativeLocalHandoff(const std::wstring& script, const std::wstring& backendW, uint32_t capW, uint32_t capH,
                               std::vector<std::string>& lines)
{
    if (!lkBaseHandoff(script, backendW, capW, capH, lines))
        return false;
    if (!g_restore)
        return true;
    LkSession s;
    const int w = lkField(lines.back(), "w"), h = lkField(lines.back(), "h");
    if (!lkSession(script, backendW, capW, capH, s) || w <= 0 || h <= 0)
    {
        lines.clear();
        return false;
    }
    if (g_restoreA4k && g_restoreModel.empty()) // Anime4K's Restore CNN UL: the Anime4K module, nothing to look up
    {
        const std::string ready = lines.back();
        lines.pop_back();
        lines.push_back(std::string("NATIVE-PATH restore=") + kRestNativeA4k);
        lines.push_back(ready);
        return true;
    }
    if (g_restoreModel.empty()) // the bundled Real-ESRGAN on the host's kernels
    {
        if (!restNatWanted())
        {
            lines.clear();
            return false;
        }
        const std::string ready = lines.back();
        lines.pop_back();
        lines.push_back(std::string("NATIVE-PATH restore=") + kRestNative);
        lines.push_back(ready);
        return true;
    }
    if (!ajNatWanted()) // a --restore-model in another form: no Restore (ajNatWanted's line says why)
    {
        lines.clear();
        return false;
    }
    const std::string ready = lines.back();
    lines.pop_back();
    lines.push_back(std::string("NATIVE-PATH restore=") + kRestNativeAj);
    lines.push_back(ready);
    return true;
}

static bool lkReadFile(const std::string& path, std::vector<char>& out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return false;
    out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return !out.empty();
}

// Restore on the host's kernels for a session: the bundled Real-ESRGAN = kRestNative (false when its weights are
// unusable), a --restore-model file = kRestNativeAj in AnimeJaNai's SPAN-F3 form, any other graph = false
// (ajNatWanted's line says why). The live cold start and the offline host share it. true = ready, the path in rpath.
static bool lkEnsureRestore(const std::wstring& script, std::string& rpath)
{
    if (!nativeLoadDlls(script))
        return false;
    if (g_restoreA4k && !g_restoreModel.empty())
        LOG("native: --restore-anime4k ignored: the --restore-model holds Restore\n");
    if (g_restoreA4k && g_restoreModel.empty())
    {
        rpath = kRestNativeA4k; // the Anime4K module builds at the session's setup (nativeA4kRestSetup)
        return true;
    }
    if (g_restoreModel.empty() ? !restNatWanted() : !ajNatWanted())
        return false;
    rpath = g_restoreModel.empty() ? kRestNative : kRestNativeAj;
    return true;
}

// the cold start: every backend's nets run on the host's kernels (echo, nvof and fruc have none) and so does live
// Restore, so nothing is built; true = the session can start (worth a second lookup)
static bool nativeLocalBuild(const std::wstring& script, const std::wstring& backendW, uint32_t capW, uint32_t capH)
{
    if (g_offline)
        return false;
    std::vector<std::string> base;
    if (!lkBaseHandoff(script, backendW, capW, capH, base))
        return false;
    std::string rpath;
    return !g_restore || lkEnsureRestore(script, rpath);
}

// ---- offline RIFE -------------------------------------------------------------------------------------------------
// The offline host's IFNet runs on the host's kernels (kIfnNative, one in-between frame a call at every multiplier):
// the session's pad of the source only (/64, the author's /128 for plain RIFE), nothing to find or build.
struct OfflineEngines
{
    std::string ifnet;
    int ph = 0, pw = 0;
};

static bool lkOfflineRife(const std::wstring& script, int w, int h, OfflineEngines& o, int padMult = 64)
{
    LkSession s;
    if (!lkSession(script, L"rife", (uint32_t)w, (uint32_t)h, s, padMult))
        return false;
    if (!nativeLoadDlls(script))
        return false;
    o.ifnet = kIfnNative;
    o.ph = s.ph;
    o.pw = s.pw;
    LOG("offline: RIFE's IFNet on the host's kernels at %dx%d\n", s.pw, s.ph);
    return true;
}

// ---- offline GMFSS ----------------------------------------------------------------------------
// lkSession "gmfss"'s sizes (the padded frame, the half) for nr: every GMFSS net runs on the host's kernels, so there
// is no engine to find or build
static bool lkOfflineGmfss(const std::wstring& script, int w, int h, NativeRife& nr)
{
    LkSession s;
    if (!lkSession(script, L"gmfss", (uint32_t)w, (uint32_t)h, s))
        return false;
    if (!nativeLoadDlls(script))
        return false;
    nr.gmfss = true;
    nr.ph = s.ph;
    nr.pw = s.pw;
    nr.hh = s.hh;
    nr.hw = s.hw;
    LOG("offline: GMFSS at %dx%d on the host's kernels (no engine)\n", s.pw, s.ph);
    return true;
}

// ---- offline Smooth Motion ------------------------------------------------------------------
// no engine: the bridge folder the live handoff names (SMV_NVOFFRUC_DIR, else engine\nvoffruc) with its three DLLs; nativeFrucSetup loads the bridge from it
static bool lkOfflineFruc(const std::wstring& script, std::string& dir)
{
    dir = lkEnv("SMV_NVOFFRUC_DIR");
    if (dir.empty())
    {
        std::wstring eng = script.substr(0, script.find_last_of(L"\\/"));
        for (auto& c : eng)
            if (c == L'/')
                c = L'\\';
        dir = wideToUtf8(eng) + "\\nvoffruc";
    }
    for (const char* dll : {"nvoffruc_bridge.dll", "NvOFFRUC.dll", "cudart64_110.dll"})
        if (!lkFile(dir + "\\" + dll))
        {
            LOG("offline: Nvidia Smooth Motion needs %s in %s\n", dll, dir.c_str());
            return false;
        }
    return true;
}

// Intel XeSS-FG's folder (SMV_XESS_DIR, else engDir\xess) holds the bridge and Intel's two DLLs (the app's XeSS setup
// installs them); tag names the route in the refusal line
static bool lkXefgDir(const std::string& engDir, std::string& dir, const char* tag)
{
    dir = lkEnv("SMV_XESS_DIR");
    if (dir.empty())
        dir = engDir + "\\xess";
    for (const char* f : {"smv_xefg_bridge.dll", "libxess_fg.dll", "libxell.dll"})
        if (!lkFile(dir + "\\" + f))
        {
            LOG("%s: Intel XeSS frame generation unavailable (smv_xefg_bridge.dll and Intel's libxess_fg.dll + "
                "libxell.dll needed in %s; the app's XeSS setup installs them)\n",
                tag, dir.c_str());
            return false;
        }
    return true;
}

// offline AMD FSR frame generation (kind 0): the bridge folder the live handoff names (SMV_FSRFG_DIR, else
// engine\fsrfg); Intel XeSS-FG (kind 1): lkXefgDir's
static bool lkOfflineFsrfg(const std::wstring& script, std::string& dir, int kind)
{
    std::wstring eng = script.substr(0, script.find_last_of(L"\\/"));
    for (auto& c : eng)
        if (c == L'/')
            c = L'\\';
    if (kind == 1)
        return lkXefgDir(wideToUtf8(eng), dir, "offline");
    dir = lkEnv("SMV_FSRFG_DIR");
    if (dir.empty())
        dir = wideToUtf8(eng) + "\\fsrfg";
    if (!lkFile(dir + "\\smv_fsrfg_bridge.dll"))
    {
        LOG("offline: AMD FSR frame generation needs smv_fsrfg_bridge.dll in %s\n", dir.c_str());
        return false;
    }
    return true;
}

// The engine handoff: the session's geometry and model paths, from the host's own lookup
// (nativeLocalHandoff), after the cold-start check (nativeLocalBuild) on a worker thread when the
// lookup misses. Blocking (the caller is srv.start() on the loader thread or the early handoff
// thread). One check at a time: a new handoff waits for a running one first.
struct BgHandoff
{
    std::atomic<bool> pending{false};
};
static BgHandoff g_bgHandoff;
static bool nativeBuildPending()
{
    return g_bgHandoff.pending.load();
}
// the session is ending (stdin "stop", or the load-time resize); polled by the handoff wait
static bool nativeLoadAbort()
{
    return g_stopReq.load() || g_resizeReq.load();
}

static bool nativeHandoff(const std::wstring& script, const std::wstring& backend, uint32_t capW, uint32_t capH,
                          NativeRife& nr)
{
    // resident host: the same window as the previous session (same overlay size, capture
    // size, DLSS mode or share and HDR mode) gets the previous answer without a lookup,
    // as long as the engine files still exist (the user may empty the cache folder by hand);
    // live Restore rides in the key: its engine path is a fact of a restore session only; DLSS 5 too: it raises a
    // small working size (liveWorkSizeFor); --xess-fg too: its bridge folder is a fact of an fsrfg session
    wchar_t key[1024];
    swprintf_s(key, L"%s|%s|%.2f|%d|%u|%u|%u|%u|%d|%d|%d|%d|%d", script.c_str(), backend.c_str(), g_flowScale,
               g_dlssMode, W, H, capW, capH, g_hdr ? 1 : 0, g_restore ? 1 : 0, g_restoreA4k ? 1 : 0, g_dlssnr ? 1 : 0,
               g_xessFg ? 1 : 0);
    bool factsOk = g_resident && g_res.haveFacts && g_res.handoffKey == key;
    if (factsOk)
    {
        const NativeRife& f = g_res.facts;
        if (!f.gmfss)
            factsOk = f.ifnetPath == kIfnNative;
        factsOk = factsOk && (f.restorePath.empty() || f.restorePath == kRestNative || f.restorePath == kRestNativeAj ||
                              f.restorePath == kRestNativeA4k);
    }
    if (factsOk)
    {
        const NativeRife& f = g_res.facts;
        nr.ifnetPath = f.ifnetPath;
        nr.restorePath = f.restorePath;
        nr.restore = f.restore;
        nr.noEngine = f.noEngine;
        nr.cachePath = f.cachePath;
        nr.nvof = f.nvof;
        nr.fruc = f.fruc;
        nr.fsrfg = f.fsrfg;
        nr.fgKind = f.fgKind;
        nr.frucDir = f.frucDir;
        nr.drba = f.drba;
        nr.gmfss = f.gmfss;
        nr.hh = f.hh;
        nr.hw = f.hw;
        nr.ph = f.ph;
        nr.pw = f.pw;
        nr.w = f.w;
        nr.h = f.h;
        nr.cw = f.cw;
        nr.ch = f.ch;
        nr.dw = f.dw;
        nr.dh = f.dh;
        nr.x0 = f.x0;
        nr.y0 = f.y0;
        nr.mph = f.mph;
        nr.mpw = f.mpw;
        nr.mw = f.mw;
        nr.mh = f.mh;
        nr.zeroPad = f.zeroPad;
        nr.batchMax = f.batchMax;
        nr.identity = f.identity;
        nr.fitAa = f.fitAa;
        nr.hdr = g_hdr;
        nativeDeriveUpscale(nr); // from this session's own --upscale, never a stored fact
        LOG("native: engine handoff skipped, same window as the resident session\n");
        return true;
    }
    // an earlier session's build is still running: wait for it (its engines may be the ones
    // this session needs), unless this session is ending too
    if (g_bgHandoff.pending.load())
    {
        LOG("native: an earlier engine build is still running, waiting for it\n");
        while (g_bgHandoff.pending.load())
        {
            if (nativeLoadAbort())
            {
                LOG("native: session ended while waiting for the earlier engine build\n");
                return false;
            }
            Sleep(50);
        }
    }
    std::string all;
    bool ready = false;
    // one handoff line of the lookup's answer
    auto onLine = [&](const std::string& line) {
        if (line.rfind("NATIVE-PATH ifnet=", 0) == 0)
            nr.ifnetPath = line.substr(18);
        else if (line.rfind("NATIVE-PATH restore=", 0) == 0)
            nr.restorePath = line.substr(20);
        else if (line.rfind("NATIVE-PATH cache=", 0) == 0)
            nr.cachePath = line.substr(18);
        else if (line.rfind("NATIVE-PATH fruc=", 0) == 0)
            nr.frucDir = line.substr(17);
        else if (line.rfind("NATIVE-PATH fsrfg=", 0) == 0)
            nr.frucDir = line.substr(18);
        else if (line.rfind("LIVE READY", 0) == 0)
        {
            all = line;
            ready = true;
        }
    };
    {
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<std::string> local;
        bool hit = nativeLocalHandoff(script, backend, capW, capH, local);
        if (!hit)
        {
            // the lookup missed: the cold-start check (nativeLocalBuild) on a worker thread; a session
            // ending meanwhile leaves it running under g_bgHandoff (see the comment above)
            auto state = std::make_shared<std::atomic<int>>(0); // 0 running, 1 built, 2 not
            g_bgHandoff.pending.store(true);
            std::thread([state, script, backend, capW, capH] {
                const bool ok = nativeLocalBuild(script, backend, capW, capH);
                state->store(ok ? 1 : 2);
                g_bgHandoff.pending.store(false);
            }).detach();
            while (state->load() == 0)
            {
                if (nativeLoadAbort())
                {
                    LOG("native: session ended during the host engine build, the build continues in the background\n");
                    return false;
                }
                Sleep(20);
            }
            local.clear();
            hit = state->load() == 1 && nativeLocalHandoff(script, backend, capW, capH, local);
        }
        if (hit)
        {
            for (const auto& l : local)
            {
                onLine(l);
                if (g_handoffDump)
                    LOG("handoff-line %s\n", l.c_str());
            }
            LOG("native: warm engines found by the host (%.2fs)\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
    }
    if (!ready)
    {
        // the lookup and the cold-start check cover every product case; what is left (a lookup
        // that failed: its own line says why)
        // ends the session here
        LOG("native: the host could not find or build the engines for this session (see the lines above; "
            "the engine folder's weights and engine\\gpu_runtime must be intact)\n");
        return false;
    }
    LOG("%s\n", all.c_str());
    auto num = [&](const char* key, int dflt) -> int {
        const std::string k = std::string(" ") + key + "=";
        const size_t p = all.find(k);
        return p == std::string::npos ? dflt : atoi(all.c_str() + p + k.size());
    };
    nr.ph = num("ph", 0);
    nr.pw = num("pw", 0);
    nr.w = num("w", 0);
    nr.h = num("h", 0);
    nr.cw = num("cw", 0);
    nr.ch = num("ch", 0);
    nr.dw = num("dw", 0);
    nr.dh = num("dh", 0);
    nr.x0 = num("x0", 0);
    nr.y0 = num("y0", 0);
    // RIFE's two domains: the motion frame's pad and content size (absent = the IFNet reads the pictures)
    nr.mph = num("mph", 0);
    nr.mpw = num("mpw", 0);
    nr.mw = num("motw", 0);
    nr.mh = num("moth", 0);
    nr.zeroPad = nr.mph && num("zpad", 0) == 1;
    if (nr.mph && (nr.mpw <= 0 || nr.mw <= 0 || nr.mh <= 0 || nr.mw > nr.mpw || nr.mh > nr.mph))
    {
        LOG("native: handoff line names an incomplete motion frame (mph / mpw / motw / moth)\n");
        return false;
    }
    nr.batchMax = num("batch", 0);
    if (nr.batchMax < 1)
        nr.batchMax = 1;
    // no-engine mode: the echo handoff answers `engine=none` (geometry only, no paths); the
    // flow factor and the engine paths are not facts of such a session
    nr.noEngine = all.find(" engine=none") != std::string::npos;
    if (nr.noEngine && nr.cachePath.empty())
    {
        LOG("native: no-engine handoff named no cache folder\n");
        return false;
    }
    // the nvof model: the same geometry-only answer with `engine=nvof` (the
    // cache folder holds the kernel cubin)
    nr.nvof = all.find(" engine=nvof") != std::string::npos;
    if (nr.nvof && nr.cachePath.empty())
    {
        LOG("native: nvof handoff named no cache folder\n");
        return false;
    }
    // Smooth Motion: the same geometry-only shape, plus the bridge folder
    nr.fruc = all.find(" engine=fruc") != std::string::npos;
    if (nr.fruc && (nr.cachePath.empty() || nr.frucDir.empty()))
    {
        LOG("native: fruc handoff named no cache or bridge folder\n");
        return false;
    }
    // AMD FSR frame generation: the same shape with its own bridge, on the FRUC route (NativeRife's fsrfg notes)
    nr.fsrfg = all.find(" engine=fsrfg") != std::string::npos;
    if (nr.fsrfg && (nr.cachePath.empty() || nr.frucDir.empty()))
    {
        LOG("native: fsrfg handoff named no cache or bridge folder\n");
        return false;
    }
    nr.fgKind = nr.fsrfg && g_xessFg ? 1 : 0; // the folder above is XeSS-FG's then (nativeLocalHandoff)
    nr.fruc = nr.fruc || nr.fsrfg;
    // GMFSS: the half-frame grid and the cache folder for the kernel cubins (no engine, no jit path)
    nr.gmfss = all.find(" engine=gmfss") != std::string::npos;
    if (nr.gmfss)
    {
        nr.hh = num("hh", 0);
        nr.hw = num("hw", 0);
        if (nr.hh <= 0 || nr.hw <= 0 || nr.cachePath.empty())
        {
            LOG("native: gmfss handoff line incomplete (hh / hw / cache)\n");
            return false;
        }
    }
    // native DRBA: the RIFE handoff (block0 runs on the host's kernels); the lag is one capture
    nr.drba = all.find(" engine=drba") != std::string::npos;
    if (nr.drba && num("lag", 0) != 1)
    {
        LOG("native: drba handoff named no lag=1\n");
        return false;
    }
    const int outW = num("outw", 0), outH = num("outh", 0);
    nr.identity = (outW == nr.w && outH == nr.h);
    // every fit that changes the size runs as the separable Lanczos pair (the sum of the two axes'
    // taps a pixel); decided here so the early helper knows to run nativeRtxInit
    // (its staging buffers) before any effect flag is looked at. Upscale to re-derives it
    // against the internal render size in nativeDeriveUpscale.
    nr.fitAa = nr.dw != nr.w || nr.dh != nr.h;
    nr.hdr = g_hdr;
    nativeDeriveUpscale(nr);
    if (nr.ph <= 0 || nr.pw <= 0 || nr.w <= 0 || nr.h <= 0 || nr.cw <= 0 || nr.ch <= 0 ||
        (!nr.noEngine && !nr.gmfss && !nr.nvof && !nr.fruc && nr.ifnetPath.empty()))
    {
        LOG("native: handoff line incomplete\n");
        return false;
    }
    if (num("effects", 0))
    {
        LOG("native: live effects are on, phase 1 has no sharpen or VSR path\n");
        return false;
    }
    // live Restore: the session asked for it, so the handoff must have named its model
    nr.restore = !nr.restorePath.empty();
    if (g_restore && !nr.restore)
    {
        LOG("native: Restore is on but the handoff named no Restore model\n");
        return false;
    }
    // a downscaling fit runs natively (k_fitAaH / k_fitAaV, nativeRtxInit)
    if ((uint32_t)nr.cw != capW || (uint32_t)nr.ch != capH)
    {
        LOG("native: handoff capture size %dx%d != %ux%u\n", nr.cw, nr.ch, capW, capH);
        return false;
    }
    if (g_resident)
    {
        NativeRife& f = g_res.facts;
        f.ifnetPath = nr.ifnetPath;
        f.restorePath = nr.restorePath;
        f.restore = nr.restore;
        f.noEngine = nr.noEngine;
        f.cachePath = nr.cachePath;
        f.nvof = nr.nvof;
        f.fruc = nr.fruc;
        f.fsrfg = nr.fsrfg;
        f.fgKind = nr.fgKind;
        f.frucDir = nr.frucDir;
        f.drba = nr.drba;
        f.gmfss = nr.gmfss;
        f.hh = nr.hh;
        f.hw = nr.hw;
        f.ph = nr.ph;
        f.pw = nr.pw;
        f.w = nr.w;
        f.h = nr.h;
        f.cw = nr.cw;
        f.ch = nr.ch;
        f.dw = nr.dw;
        f.dh = nr.dh;
        f.x0 = nr.x0;
        f.y0 = nr.y0;
        f.mph = nr.mph;
        f.mpw = nr.mpw;
        f.mw = nr.mw;
        f.mh = nr.mh;
        f.zeroPad = nr.zeroPad;
        f.batchMax = nr.batchMax;
        f.identity = nr.identity;
        f.fitAa = nr.fitAa;
        g_res.handoffKey = key;
        g_res.haveFacts = true;
    }
    return true;
}
