// part of smv-live-native.inl (included there, in place); not a standalone translation unit
// ---- part 2: handoff, engine load, kernels, buffers, the compute thread -------------------

// What a --resident process keeps between sessions: the NVRTC kernel module and the models' dtypes,
// plus the last successful handoff's facts so the same window starts again without a new handoff.
// Everything else in NativeRife (stream, buffers, imports, events) is per session. Sessions are
// strictly sequential (the previous compute thread is joined before the next early thread starts),
// so no lock guards this.
struct NativeResident
{
    std::string ifnetPath;
    bool ready = false; // a session's model state (the dtypes below) is held for the next one
    CUmodule cuMod = nullptr;
    bool featHalf = false;
    bool xHalf = false, outHalf = false, tHalf = false;
    int dev = 0;
    // memoised handoff: nativeHandoff's session key (gen left out: the fast path ignores gen and
    // the cold path builds the same engines whatever the gen) and the facts it answered with
    std::wstring handoffKey;
    NativeRife facts;
    bool haveFacts = false;
};
static NativeResident g_res;

// drop the resident engine pair (a different pair is needed for a new window size). The
// kernel module and the memoised handoff are engine independent and stay: the session that
// calls this already adopted the module in nativeBuildKernels and stored the new handoff
// facts (unloading the module here broke the first kernel launch of a later session). The
// calling thread must have the device bound.
static void residentFreeEngines()
{
    if (!g_res.ready)
        return;
    g_res.ready = false;
    g_res.ifnetPath.clear();
    LOG("native: resident engines released\n");
}

// everything, at host exit (after the last session's compute thread is gone)
static void residentFree()
{
    residentFreeEngines();
    if (g_res.cuMod)
    {
        cuModuleUnload(g_res.cuMod);
        g_res.cuMod = nullptr;
    }
    g_res.handoffKey.clear();
    g_res.haveFacts = false;
}

static bool fileExistsA(const std::string& p)
{
    const DWORD a = GetFileAttributesA(p.c_str());
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
}

static bool lkXefgDir(const std::string& engDir, std::string& dir, const char* tag); // Intel XeSS-FG's folder, below

// the server backends this host runs (PipeServer::nativeRefusal asks here): rife, echo (the
// effects-only route, the no-engine mode above), gmfss (its chain on the host's kernels), nvof (the NVIDIA
// Optical Flow model), fruc (Nvidia Smooth Motion), fsrfg (AMD FSR frame generation; Intel XeSS-FG with --xess-fg)
// and rifedrba (RIFE with DRBA timing). Any other name is refused.
static bool nativeBackendOk(const std::wstring& backend)
{
    return backend == L"rife" || backend == L"echo" || backend == L"gmfss" || backend == L"nvof" ||
           backend == L"fruc" || backend == L"fsrfg" || backend == L"rifedrba";
}

// the smallest GMFSS pad a side (lkSession): gmflow's quarter-scale windows are the pad / 64 a side and its shifted
// blocks roll by half a window, so a window needs 2 tokens a side (a 1-token window rolls by 0: an empty slice in the
// export's roll, transformer._roll_hw, untested); the coarse scale's per-region blocks (`_m`) need 2 at the pad / 32
static constexpr int kGmMinPad = 128;

// Upscale to: the
// internal render size from the exe's own --upscale H and the handoff geometry: the factor
// up_h / h clamped to 1/16..16, even dims, dropped when it
// equals the fit rect (one resize, not two). lround rounds half away from zero; the & ~1 lands a
// tie on the same even number as rounding half to even. Runs after (w, h, dw, dh) are known,
// on the resident facts too (uw is never a stored fact, the flag is per session).
static void nativeDeriveUpscale(NativeRife& nr)
{
    nr.uw = nr.uh = 0;
    nr.upAa = false;
    nr.fitAa = nr.dw != nr.w || nr.dh != nr.h;
    if (g_upscaleH > 0 && g_upscaleH != nr.h && nr.h > 0)
    {
        double f = (double)g_upscaleH / (double)nr.h;
        f = f < 1.0 / 16 ? 1.0 / 16 : (f > 16.0 ? 16.0 : f);
        int uw = (int)lround(nr.w * f) & ~1, uh = (int)lround(nr.h * f) & ~1;
        if (uw < 2)
            uw = 2;
        if (uh < 2)
            uh = 2;
        if (uw != nr.dw || uh != nr.dh)
        {
            nr.uw = uw;
            nr.uh = uh;
            nr.upAa = uw != nr.w || uh != nr.h;
            nr.fitAa = nr.dw != uw || nr.dh != uh; // the second resize, internal frame -> the fit rect
        }
    }
}

// ---- the session handoff --------------------------------------------------------------------
// Every net runs on the host's kernels, so the host answers the handoff itself from the session's
// arguments: NATIVE-PATH lines plus one LIVE READY line, the format nativeHandoff parses.

static std::string lkEnv(const char* name)
{
    char buf[512];
    size_t n = 0;
    if (getenv_s(&n, buf, sizeof(buf), name) || !n)
        return std::string();
    return std::string(buf);
}

// SMV_HANDOFF_DUMP=1: log every handoff line the host produces as `handoff-line ...`
static const bool g_handoffDump = lkEnv("SMV_HANDOFF_DUMP") == "1";

// round half to even (the default FP rounding mode)
static int lkRound(double x)
{
    return (int)std::nearbyint(x);
}

static bool lkFile(const std::string& p)
{
    return fileExistsA(p);
}

// RIFE's IFNet on the host's kernels (nativeIfnModule): the path every RIFE / DRBA lookup hands out, nothing to build
static const char* const kIfnNative = "native:rife-4.25-ifnet";

static std::string lkF4(double v)
{
    char b[64];
    sprintf_s(b, "%.4f", v);
    return b;
}

// Everything the lookup and the cold-start check derive from a session's arguments, in ONE
// place: both must see the same names and sizes.
struct LkSession
{
    std::string backend, engDir, cacheDir;
    std::wstring engDirW;
    double imgScale = 1.0;    // live: the share the working size came from (the DLSS mode's, or --scale's)
    int cw = 0, ch = 0;       // the capture
    int presW = 0, presH = 0; // live: the presented rect (the capture aspect-fit into the canvas)
    int mode = 0;             // live: the DLSS mode the size came from (1 + kDlssModeName index, 0 = a share)
    int mw = 0, mh = 0;       // the model frame = the working size (live: liveWorkSize)
    int ph = 0, pw = 0;       // the /64 pad (SMV_LIVE_SAFEPAD=1 on the RIFE family)
    // gmfss: the half size (every net on the host's kernels: no engine)
    int hh = 0, hw = 0;
    // rife / rifedrba
    bool drba = false;
    // live rife: RIFE's motion frame (mmw x mmh, padded mph x mpw), the frame the IFNet and
    // its encode run at (iph x ipw = the motion pad, else the pictures' pad)
    int mmw = 0, mmh = 0, mph = 0, mpw = 0, iph = 0, ipw = 0;
    bool zpad = false; // live rife: the motion frame padded as RIFE's author pads (NativeRife::zeroPad)
};

// Create a folder and its missing parents. Users delete model_cache_safe_to_delete whenever
// they like (the name invites it), and every engine, JIT and cubin write below assumes the folder
// exists: a missing one used to fail each save silently and rebuild every session.
static void ensureDirW(const std::wstring& dir)
{
    if (dir.empty() || dir == L".")
        return;
    const DWORD a = GetFileAttributesW(dir.c_str());
    if (a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY))
        return;
    const size_t sl = dir.find_last_of(L"\\/");
    if (sl != std::wstring::npos && sl > 0 && dir[sl - 1] != L':')
        ensureDirW(dir.substr(0, sl));
    if (!CreateDirectoryW(dir.c_str(), nullptr) && GetLastError() != ERROR_ALREADY_EXISTS)
        LOG("cache: cannot create %ls (error %lu)\n", dir.c_str(), GetLastError());
}

// the rect a w x h frame takes in the outW x outH canvas: aspect-fit, even, centred
static void lkFitRect(int w, int h, int outW, int outH, int& dw, int& dh, int& x0, int& y0)
{
    if (outW == w && outH == h)
    {
        dw = w;
        dh = h;
        x0 = 0;
        y0 = 0;
        return;
    }
    const double s = (std::min)((double)outW / w, (double)outH / h);
    dw = (std::max)(2, lkRound(w * s) & ~1);
    dh = (std::max)(2, lkRound(h * s) & ~1);
    x0 = (outW - dw) / 2;
    y0 = (outH - dh) / 2; // both non-negative: the division floors
}

// NVIDIA's DLSS modes as plan.ts has them (DLSS_MODES): the working size's share of the output per kDlssModeName entry
static const double kDlssModeShare[5] = {1.0, 1.0 / 1.5, 1.0 / 1.724, 1.0 / 2, 1.0 / 3};

// DLSS 5's GPU work never completes on a 64 px wide frame (the GPU resets; 102 and 128 px wide run, and so does a
// 64 px high one): with DLSS 5 on, a working size below the presented rect is at least this wide, and a frame that is
// itself smaller than kNrMinWorkW x kNrMinWorkH runs without DLSS 5 (both routes' NR init)
static const int kNrMinWorkW = 128, kNrMinWorkH = 64;

// Live DLSS 4.5 at the working size runs its frames at the working size itself, so the working size keeps the shorter
// side of the canvas at least this long (the helpers' floor, above DLSS-G's own: 160x90 refused, 192x108 and 128x128
// run)
static const int kDlssgMinWork = 128;

// Live's working size (Restore, the resize, DLSS 5, Sharpen, RTX HDR and the model run at it, the fit
// takes it to the canvas after the model): the DLSS mode's share of the PRESENTED rect (the capture aspect-fit into
// the canvas W x H: the window itself, or the Fill rect), even, at most the presented rect a side, the share raised
// (aspect kept) to the largest minimum of the session's pipeline, so the capture is shrunk once and never below what a
// model needs: the shorter side 64 px, with DLSS 5 the width kNrMinWorkW, with DLSS 4.5 at the working size the
// canvas's shorter side kDlssgMinWork (*raised: 1 = the 64 px floor, 2 = DLSS 5's width, 3 = DLSS 4.5's minimum);
// capped at 3840x2160 keeping the aspect (offline's workPlan). DLAA = the presented rect itself (in window
// mode the capture, odd sizes kept). liveWorkSizeFor = the size a mode gives the presented presW x presH (mode = 1 +
// the kDlssModeName index, 0 = --scale's share; fDg = DLSS 4.5's minimum share, 0 = none); liveWorkSize = the
// session's, and the mode it came from.
static void liveWorkSizeFor(int mode, int presW, int presH, int& mw, int& mh, int* raised = nullptr, double fDg = 0.0)
{
    double f = 1.0;
    if (mode)
        f = kDlssModeShare[mode - 1];
    else
    {
        char sb[32];
        sprintf_s(sb, "%.2f", g_flowScale);
        f = (std::max)(0.01, (std::min)(1.0, strtod(sb, nullptr)));
    }
    mw = presW;
    mh = presH;
    if (raised)
        *raised = 0;
    if (f < 1.0)
    {
        const double fFloor = (std::max)(64.0 / presW, 64.0 / presH);
        const double fNr = g_dlssnr ? (double)kNrMinWorkW / presW : 0.0;
        const double fModel = (std::max)(fFloor, fNr);
        const double k = (std::max)((std::max)(f, fDg), fModel);
        if (raised && k > f)
            *raised = fDg >= fModel ? 3 : fNr >= fFloor ? 2 : 1;
        mw = (std::min)(presW, (std::max)(g_dlssnr ? kNrMinWorkW : 64, lkRound(presW * k) & ~1));
        mh = (std::min)(presH, (std::max)(64, lkRound(presH * k) & ~1));
    }
    if ((double)mw * mh > 3840.0 * 2160)
    {
        const double k = std::sqrt(3840.0 * 2160 / ((double)mw * mh));
        mw = (int)std::floor(mw * k) & ~1;
        mh = (int)std::floor(mh * k) & ~1;
    }
}
static void liveWorkSize(int cw, int ch, int& presW, int& presH, int& mode, int& mw, int& mh, int* raised = nullptr)
{
    const int outW = g_presW ? (int)g_presW : W ? (int)W : cw, outH = g_presW ? (int)g_presH : W ? (int)H : ch;
    int x0, y0;
    lkFitRect(cw, ch, outW, outH, presW, presH, x0, y0);
    mode = g_dlssMode;
    liveWorkSizeFor(mode, presW, presH, mw, mh, raised,
                    g_dlssgWorkFloor ? (double)kDlssgMinWork / (std::min)(outW, outH) : 0.0);
}

// The model cache sits in the app's top folder: the repo root in a dev tree, the folder holding
// SmoothMyVideo.exe in an install (whose engine folder is resources\engine). src/render/cache.ts
// derives the same folder for its stamp check and the cubin folder.
static std::string lkModelCacheDir(const std::string& engDir)
{
    auto parentOf = [](const std::string& p) {
        const size_t sl = p.find_last_of('\\');
        return sl == std::string::npos ? std::string(".") : p.substr(0, sl);
    };
    // engDir can be relative or end in `..` (the default script is <exe dir>\..\live_server.py):
    // resolve it the way the file system does before taking its parent
    const std::wstring engW = utf8ToWide(engDir);
    wchar_t full[4096];
    const DWORD n = GetFullPathNameW(engW.c_str(), (DWORD)(sizeof(full) / sizeof(full[0])), full, nullptr);
    std::string resolved = n > 0 && n < sizeof(full) / sizeof(full[0]) ? wideToUtf8(full) : engDir;
    while (resolved.size() > 3 && resolved.back() == '\\')
        resolved.pop_back();
    std::string top = parentOf(resolved);
    const size_t sl = top.find_last_of('\\');
    std::string last = sl == std::string::npos ? top : top.substr(sl + 1);
    for (auto& c : last)
        c = (char)tolower((unsigned char)c);
    if (last == "resources")
        top = parentOf(top);
    return top + "\\model_cache_safe_to_delete";
}

static bool lkSession(const std::wstring& script, const std::wstring& backendW, uint32_t capW, uint32_t capH,
                      LkSession& s, int padMult = 64)
{
    s = LkSession();
    s.backend = wideToUtf8(backendW);
    const size_t sl = script.find_last_of(L"\\/");
    if (sl == std::wstring::npos)
        return false;
    s.engDirW = script.substr(0, sl);
    for (auto& c : s.engDirW)
        if (c == L'/')
            c = L'\\'; // the dev harnesses pass forward slashes
    s.engDir = wideToUtf8(s.engDirW);
    const std::string envCache = lkEnv("SMV_TRT_CACHE");
    s.cacheDir = envCache.empty() ? lkModelCacheDir(s.engDir) : envCache;
    ensureDirW(utf8ToWide(s.cacheDir));
    s.cw = (int)capW;
    s.ch = (int)capH;
    if (g_offline)
    {
        // the offline host names its model size as the capture
        s.mw = s.cw;
        s.mh = s.ch;
    }
    else
    {
        liveWorkSize(s.cw, s.ch, s.presW, s.presH, s.mode, s.mw, s.mh);
        char sb[32];
        sprintf_s(sb, "%.2f", g_flowScale);
        s.imgScale = s.mode ? kDlssModeShare[s.mode - 1] : (std::max)(0.01, (std::min)(1.0, strtod(sb, nullptr)));
    }
    s.ph = (s.mh + padMult - 1) / padMult * padMult;
    s.pw = (s.mw + padMult - 1) / padMult * padMult;
    if (s.backend == "gmfss")
    {
        // a 64 px pad side would give gmflow's quarter-scale windows 1 token a side and its shifted blocks a roll of 0
        // (kGmMinPad): such a side runs at kGmMinPad (the packers replicate the edge into the pad)
        s.ph = (std::max)(s.ph, kGmMinPad);
        s.pw = (std::max)(s.pw, kGmMinPad);
        // every net on the host's kernels (feat_ext at the padded frame; metricnet, the GMFSS IFNet, fusionnet and
        // gmflow (backbone, `_m` + A, the scale-0 matching and propagation between them, the local correlation, B) at
        // the half): no engine to look up, build or warm
        s.hh = s.ph / 2;
        s.hw = s.pw / 2;
        return true;
    }
    s.drba = s.backend == "rifedrba";
    const bool safePad = lkEnv(g_offline ? "SMV_RIFE_SAFEPAD" : "SMV_LIVE_SAFEPAD") == "1";
    if (safePad)
    {
        s.pw = (std::max)(s.pw, 1152);
        s.ph = (std::max)(s.ph, 640);
    }
    s.iph = s.ph;
    s.ipw = s.pw;
    // RIFE's two domains live (rife / rifedrba): the IFNet, its encode and DRBA's block0 find
    // the motion on the finished picture at the working size, or at the capture size when the working
    // size enlarges it (Fill at DLAA: an enlarged picture costs RIFE its motion); the warp + blend runs
    // on the pictures at the working size. SMV_RIFE_TWO_DOMAIN=0: the IFNet reads the pictures.
    if (!g_offline && (s.backend == "rife" || s.drba) && lkEnv("SMV_RIFE_TWO_DOMAIN") != "0")
    {
        const bool up = s.mw > s.cw || s.mh > s.ch;
        s.mmw = up ? s.cw : s.mw;
        s.mmh = up ? s.ch : s.mh;
        // plain RIFE pads as its author does (zeros to a multiple of 128, NativeRife::zeroPad); DRBA keeps 64
        s.zpad = s.backend == "rife" && lkEnv("SMV_RIFE_AUTHOR_PAD") != "0";
        const int pm = s.zpad ? 128 : 64;
        s.mph = (s.mmh + pm - 1) / pm * pm;
        s.mpw = (s.mmw + pm - 1) / pm * pm;
        if (safePad)
        {
            s.mpw = (std::max)(s.mpw, 1152);
            s.mph = (std::max)(s.mph, 640);
        }
        s.iph = s.mph;
        s.ipw = s.mpw;
    }
    return true;
}

// the session's handoff answer (the NATIVE-PATH lines and the LIVE READY line nativeHandoff
// parses), or false
static bool lkBaseHandoff(const std::wstring& script, const std::wstring& backendW, uint32_t capW, uint32_t capH,
                          std::vector<std::string>& lines)
{
    if (g_offline)
        return false; // the offline host has its own lookups (lkOffline*)
    LkSession s;
    if (!lkSession(script, backendW, capW, capH, s))
        return false;
    const std::string& backend = s.backend;
    const bool geoOnly = backend == "nvof" || backend == "fruc" || backend == "fsrfg"; // no engine: geometry only
    if (!geoOnly && backend != "rife" && backend != "echo" && backend != "gmfss" && backend != "rifedrba")
        return false;
    const std::string& engDir = s.engDir;
    const std::string& cacheDir = s.cacheDir;
    const double imgScale = s.imgScale;
    const int cw = s.cw, ch = s.ch, mw = s.mw, mh = s.mh;
    const int outW = W ? (int)W : cw, outH = W ? (int)H : ch;
    // the presented rect comes from the capture, never the working size: each of its axes rounds to even on its own,
    // so its aspect can miss the capture's and a fit from it would leave black rows or columns
    // (DLSS 4.5 at the working size, g_presW: the presented rect's fit scaled to the smaller output; a picture that
    // fills the presented rect fills it whole, since lkFitRect's even rounding there leaves a line DXGI then stretches)
    auto fitRect = [&](int& dw, int& dh, int& x0, int& y0) {
        if (!g_presW)
        {
            lkFitRect(cw, ch, outW, outH, dw, dh, x0, y0);
            return;
        }
        int pw = 0, ph = 0, px = 0, py = 0;
        lkFitRect(cw, ch, (int)g_presW, (int)g_presH, pw, ph, px, py);
        const bool whole = px <= 1 && py <= 1;
        dw = whole ? outW : (std::max)(2, lkRound((double)pw * outW / g_presW) & ~1);
        dh = whole ? outH : (std::max)(2, lkRound((double)ph * outH / g_presH) & ~1);
        x0 = (outW - dw) / 2;
        y0 = (outH - dh) / 2;
    };
    auto geo = [&](int ph, int pw, int w, int h, int dw, int dh, int x0, int y0) -> std::string {
        char b[512];
        sprintf_s(b, "ph=%d pw=%d w=%d h=%d cw=%d ch=%d outw=%d outh=%d dw=%d dh=%d x0=%d y0=%d", ph, pw, w, h, cw, ch,
                  outW, outH, dw, dh, x0, y0);
        return b;
    };
    if (backend == "echo")
    {
        // the effects run at the working size like the models' (no /64 pad: no engine reads the frame)
        int dw, dh, x0, y0;
        fitRect(dw, dh, x0, y0);
        lines.push_back("NATIVE-PATH cache=" + cacheDir);
        lines.push_back("LIVE READY native=1 " + geo(mh, mw, mw, mh, dw, dh, x0, y0) + " scale=" + lkF4(imgScale) +
                        " batch=0 batchpad=0 effects=0 restore=0 engine=none fast=1");
        return true;
    }
    if (geoOnly)
    {
        const int gw = mw, gh = mh;
        int dw, dh, x0, y0;
        fitRect(dw, dh, x0, y0);
        const std::string tail =
            " scale=" + lkF4(imgScale) + " batch=0 batchpad=0 effects=0 restore=0 engine=" + backend + " fast=1";
        if (backend == "nvof")
        {
            // no /64 pad: the Optical Flow engine takes any size from 32x32
            lines.push_back("NATIVE-PATH cache=" + cacheDir);
            lines.push_back("LIVE READY native=1 " + geo(gh, gw, gw, gh, dw, dh, x0, y0) + tail);
            return true;
        }
        if (backend == "fsrfg")
        {
            // AMD FSR frame generation: our bridge folder (SMV_FSRFG_DIR, else engine\fsrfg), no /64 pad (FSR and the
            // Optical Flow engine take any size from 32x32); --xess-fg: Intel XeSS-FG's folder (lkXefgDir)
            std::string gdir;
            if (g_xessFg)
            {
                if (!lkXefgDir(engDir, gdir, "native"))
                    return false;
            }
            else
            {
                gdir = lkEnv("SMV_FSRFG_DIR");
                if (gdir.empty())
                    gdir = engDir + "\\fsrfg";
                if (!lkFile(gdir + "\\smv_fsrfg_bridge.dll"))
                {
                    LOG("native: AMD FSR frame generation needs smv_fsrfg_bridge.dll in %s\n", gdir.c_str());
                    return false;
                }
            }
            lines.push_back("NATIVE-PATH cache=" + cacheDir);
            lines.push_back("NATIVE-PATH fsrfg=" + gdir);
            lines.push_back("LIVE READY native=1 " + geo(gh, gw, gw, gh, dw, dh, x0, y0) + tail);
            return true;
        }
        // fruc: the bridge folder (SMV_NVOFFRUC_DIR, else engine\nvoffruc) and its three DLLs
        std::string fdir = lkEnv("SMV_NVOFFRUC_DIR");
        if (fdir.empty())
            fdir = engDir + "\\nvoffruc";
        for (const char* dll : {"nvoffruc_bridge.dll", "NvOFFRUC.dll", "cudart64_110.dll"})
            if (!lkFile(fdir + "\\" + dll))
            {
                LOG("native: Nvidia Smooth Motion needs %s in %s\n", dll, fdir.c_str());
                return false;
            }
        const int ph = (gh + 63) / 64 * 64, pw = (gw + 63) / 64 * 64; // never the safe pad
        lines.push_back("NATIVE-PATH cache=" + cacheDir);
        lines.push_back("NATIVE-PATH fruc=" + fdir);
        lines.push_back("LIVE READY native=1 " + geo(ph, pw, gw, gh, dw, dh, x0, y0) + tail);
        return true;
    }
    if (!nativeLoadDlls(script))
        return false;
    const int ph = s.ph, pw = s.pw;
    // rife / rifedrba: the IFNet on the host's kernels (one in-between frame a call, offline's class too): nothing to
    // find, build or warm; its kernel cubin goes into the model cache
    if (backend == "rife" || backend == "rifedrba")
    {
        int dw, dh, x0, y0;
        fitRect(dw, dh, x0, y0);
        lines.push_back(std::string("NATIVE-PATH ifnet=") + kIfnNative);
        lines.push_back("NATIVE-PATH cache=" + cacheDir);
        char tail[512], motion[96] = "";
        if (s.mph)
            sprintf_s(motion, " mph=%d mpw=%d motw=%d moth=%d%s", s.mph, s.mpw, s.mmw, s.mmh, s.zpad ? " zpad=1" : "");
        sprintf_s(tail, " batch=0 batchpad=0 xdtype=fp16 tdtype=fp16 fdtype=fp16 outdtype=fp16 effects=0 fast=1%s%s",
                  motion, s.drba ? " engine=drba lag=1" : "");
        lines.push_back("LIVE READY native=1 " + geo(ph, pw, mw, mh, dw, dh, x0, y0) + " scale=" + lkF4(imgScale) +
                        tail);
        return true;
    }
    if (backend == "gmfss")
    {
        // every GMFSS net runs on the host's kernels: the session's half size and the kernel cache, no engine
        int dw, dh, x0, y0;
        fitRect(dw, dh, x0, y0);
        lines.push_back("NATIVE-PATH cache=" + cacheDir);
        char tail[256];
        sprintf_s(tail, " batch=0 batchpad=0 hh=%d hw=%d effects=0 restore=0 engine=gmfss fast=1", s.hh, s.hw);
        lines.push_back("LIVE READY native=1 " + geo(ph, pw, mw, mh, dw, dh, x0, y0) + " scale=" + lkF4(imgScale) +
                        tail);
        return true;
    }
    return false;
}

static bool lkReadFile(const std::string& path, std::vector<char>& out);

static int lkField(const std::string& line, const char* key)
{
    const std::string k = std::string(" ") + key + "=";
    const size_t p = line.find(k);
    return p == std::string::npos ? -1 : atoi(line.c_str() + p + k.size());
}
