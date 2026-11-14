// part of smv-live-native.inl (included there, in place); not a standalone translation unit
// ---- part 4: the compute thread and the public entry point --------------------------------

// the stdin `set` lines that arrived since the last frame group (g_liveSets), on the native thread that reads these
// values: the Sharpen strength (0 = off, on = its buffers first), RTX VSR's quality while RTX VSR is the upscaler, the RTX HDR
// sliders while RTX HDR runs, DLSS 5's Structure / Tone and its passes (more are built in place, memory permitting) while
// DLSS 5 runs (held for its start while it does not), DLSS 5 itself on / off (nativeNrSwitch), the HUD (the loop
// thread's), the upscaler / Restore / RTX HDR kind this session runs off and on again (their fallbacks); each read
// again at its next launch / evaluate; the HUD's latency segment (the loop thread's next paint). Any other key, or one
// whose effect this session does not run, waits for the next session's arguments (the app passes the new value there)
// Sharpen turned on in a running session: its tight result (and the HDR planes' SDR view) at the model frame's size,
// when the video memory room holds them; false = the next session takes it (its own fit gives the picked effects room)
static bool nativeSharpenOn(NativeRife& nr)
{
    const bool view = nr.encPost && nr.shHdrView;
    const size_t one = (size_t)3 * nr.w * nr.h * sizeof(float);
    const uint64_t need = (nr.dShOut ? 0 : one) + (view && !nr.dShIn ? one : 0);
    const uint64_t room = nativeVideoMemoryRoom();
    if (need && room && room < need)
        return false;
    if (!nr.dShOut && cudaMalloc((void**)&nr.dShOut, one) != cudaSuccess)
    {
        nr.dShOut = nullptr;
        return false;
    }
    if (view && !nr.dShIn && cudaMalloc((void**)&nr.dShIn, one) != cudaSuccess)
    {
        nr.dShIn = nullptr;
        return false;
    }
    return true;
}

// DLSS 5 switched in a running session: off = the pass skips (the core and its memory stay for the session); on again
// = it resumes from a Reset (the history stopped with it); on in a session that started without it = the core brought
// up as the session start does (nativeLiveNrInit, at the held Structure / Tone / passes / Style), when the video memory
// room holds its working memory. false = this session cannot (why says it); the next session takes it
static bool nativeNrSwitch(NativeRife& nr, bool on, std::string& why)
{
    if (!on)
    {
        nr.nrPaused = nr.liveNr;
        return true;
    }
    if (nr.liveNr && nr.nrHost && !nr.nrFailed)
    {
        if (nr.nrPaused)
        {
            nr.nrPaused = false;
            nr.nrFirst = true;
            nr.nrHaveLast = false;
        }
        return true;
    }
    if (nr.nrFailed)
    {
        why = "DLSS 5 stopped earlier in this session";
        return false;
    }
    if (g_backend != BK_SERVER)
    {
        why = "this route runs no native pass";
        return false;
    }
    const int p = g_nrPasses < 1 ? 1 : (g_nrPasses > nr::kMaxPasses ? nr::kMaxPasses : g_nrPasses);
    const uint64_t need = (uint64_t)((kNrLiveLateBase + nr::kFrameMp * nr.w * nr.h / 1e6 * p) * 1048576.0);
    const uint64_t room = nativeVideoMemoryRoom();
    if (room && room < need)
    {
        why = "not enough free video memory";
        return false;
    }
    g_liveNrCuda = g_nrNative = g_nrAttempted = true;
    if (!nativeLiveNrInit(nr))
    {
        why = "an allocation failed";
        return false;
    }
    if (!nr.liveNr)
    {
        why = "the core did not start (the line above)";
        return false;
    }
    return true;
}

// the running upscaler's own state released between two groups before another kind starts in place (nativeUpSwitch),
// the GPU idle first (the slot store's streams read its buffers): AMD FSR / Intel XeSS free their instance, imports and
// vector session (the DLL stays loaded; a bridge whose start failed too), Anime4K its resize buffers (the statistics
// and Restore's stay), ArtCNN its buffers and weights; RTX VSR keeps NGX (one create a process, its VSR feature reused
// by a switch back); the buffers every kind shares at this geometry (dVsrIn / dVsrOut, dPres, dVsrView / dVsrRem) stay
static void nativeUpRelease(NativeRife& nr)
{
    const char* name = !nr.vsr    ? nullptr
                       : nr.fsrUp ? (nr.fuKind ? "Intel XeSS upscaling" : "AMD FSR upscaling")
                       : nr.a4kUp ? "Anime4K"
                       : nr.acnUp ? "ArtCNN"
                                  : "RTX VSR";
    cudaDeviceSynchronize();
    if (nr.fuMade)
        nativeFsrupRelease(nr);
    if (nr.a4kUp)
        nativeA4kUpFree(nr);
    if (nr.acnUp)
        nativeAcnRelease(nr);
    nr.vsr = nr.vsrWant = nr.fsrUp = nr.fuHdr = nr.a4kUp = nr.acnUp = nr.vsrFailed = nr.vsrUserOff = false;
    if (name)
        LOG("native: live upscale: %s released in place\n", name);
}

// an upscaler ticked in a running session that started without one, or another kind than the one it runs (stdin `set
// up N`, N = the kind the renderer picked: 1 RTX VSR, 2 AMD FSR 4, 3 Anime4K, 4 ArtCNN, 5 Intel XeSS): the live case
// where the fit after the model enlarges (the upscaler's resize under nativeRtxInit's rule), on SDR or HDR planes, set
// up on the native thread between two groups the way the session start does (the DLL checks and HDR levers, the
// buffers the slot store reads, the bridge or module, the warm-up; FSR / XeSS's first frames on a worker), when the
// video memory room holds it; a running kind is released first (nativeUpRelease), after the checks that leave it
// running. false = this session cannot (why says it); the next session takes it. Left to the next session, as at the
// start: a resize that does not enlarge, RTX VSR once NGX was created without it (single-instance a process: TrueHDR
// holds it), another route; a setup that fails after the release = Lanczos
static bool nativeUpSwitch(NativeRife& nr, int want, std::string& why)
{
    if (g_offline || g_backend != BK_SERVER)
    {
        why = "this route runs no native pass";
        return false;
    }
    if (nr.hdr)
    {
        // HDR planes as the session start runs them (nativeConfigHdr's levers): RTX VSR, Anime4K and ArtCNN on the
        // frame's SDR view and back to the codes, AMD FSR / Intel XeSS in the bridge's HDR mode
        wchar_t ev[8]{};
        const bool vsrOff = want == 1 && GetEnvironmentVariableW(L"SMV_VSR_HDR", ev, 8) && ev[0] == L'0';
        const bool fuOff =
            (want == 2 || want == 5) && GetEnvironmentVariableW(L"SMV_FSRUP_HDR", ev, 8) && ev[0] == L'0';
        if (vsrOff || fuOff)
        {
            why = vsrOff ? "HDR planes (SMV_VSR_HDR=0)" : "HDR planes (SMV_FSRUP_HDR=0)";
            return false;
        }
    }
    const int tw = nr.uw ? nr.uw : nr.dw, th = nr.uw ? nr.uh : nr.dh;
    if (!(tw > nr.w && th > nr.h))
    {
        why = nr.w > nr.cw && nr.h > nr.ch ? "the resize before the model" : "this resize does not enlarge";
        return false;
    }
    const char* name = want == 1   ? "RTX VSR"
                       : want == 2 ? "AMD FSR upscaling"
                       : want == 3 ? "Anime4K"
                       : want == 4 ? "ArtCNN"
                       : want == 5 ? "Intel XeSS upscaling"
                                   : nullptr;
    if (!name)
    {
        why = "an unknown upscaler";
        return false;
    }
    std::string dir;
    if (want == 1)
    {
        if (g_rtxb.created && !g_rtxb.vsrMade)
        {
            why = "RTX Video is already created in this process without RTX VSR (RTX HDR holds it)";
            return false;
        }
        std::wstring d;
        wchar_t ov[MAX_PATH]{};
        if (GetEnvironmentVariableW(L"SMV_RTXVIDEO_DIR", ov, MAX_PATH) && ov[0])
            d = ov;
        else if (!g_nativeEngineDir.empty())
            d = g_nativeEngineDir + L"\\rtxvideo";
        if (d.empty() || GetFileAttributesW((d + L"\\rtxvideo_cuda.dll").c_str()) == INVALID_FILE_ATTRIBUTES ||
            GetFileAttributesW((d + L"\\nvngx_vsr.dll").c_str()) == INVALID_FILE_ATTRIBUTES || !rtxBridgeLoad())
        {
            why = "no rtxvideo bridge or nvngx_vsr.dll";
            return false;
        }
    }
    else if (want == 2 || want == 5)
    {
        dir = lkEnv(want == 2 ? "SMV_FSRUP_DIR" : "SMV_XESS_DIR");
        if (dir.empty() && !g_nativeEngineDir.empty())
            dir = wideToUtf8(g_nativeEngineDir) + (want == 2 ? "\\fsrup" : "\\xess");
        bool have = !dir.empty();
        if (want == 2)
            for (const char* f :
                 {"smv_fsrup_bridge.dll", "amd_fidelityfx_loader_dx12.dll", "amd_fidelityfx_upscaler_dx12.dll"})
                have = have && lkFile(dir + "\\" + f);
        else
            for (const char* f : {"smv_xess_bridge.dll", "libxess.dll"})
                have = have && lkFile(dir + "\\" + f);
        if (!have)
        {
            why = want == 2 ? "AMD's FidelityFX DLLs are missing" : "Intel's XeSS DLLs are missing";
            return false;
        }
    }
    if (nr.vsr || nr.fuMade)
        nativeUpRelease(nr); // the running kind (or a bridge whose start failed) first: its memory counts below
    // the buffers the slot store reads at this geometry (nativeRtxInit's): Anime4K / ArtCNN write planar floats into
    // the staging frame (under Upscale to the internal render frame, allocated), the bridges read and write BGRA
    // buffers; HDR planes add the staging frame for every kind (the way back to the codes lands there), the frame's SDR
    // view and the light the view never held at the target (dVsrView / dVsrRem)
    const bool planar = want == 3 || want == 4;
    const bool pres = (planar && !nr.uw) || nr.hdr;
    const uint64_t fl = sizeof(float);
    const uint64_t need = (pres && !nr.dPres ? 3 * fl * nr.dw * nr.dh : 0) +
                          (planar ? 0 : (uint64_t)nr.w * nr.h * 4 + (uint64_t)tw * th * 4) +
                          (nr.hdr && !nr.dVsrView ? 3 * fl * nr.w * nr.h : 0) +
                          (nr.hdr && !nr.dVsrRem ? 3 * fl * tw * th : 0);
    const uint64_t room = nativeVideoMemoryRoom();
    if (room && room < need)
    {
        why = "not enough free video memory";
        return false;
    }
    auto alloc = [](void** p, size_t bytes) {
        if (*p)
            return true;
        if (cudaMalloc(p, bytes) == cudaSuccess)
            return true;
        *p = nullptr;
        cudaGetLastError();
        return false;
    };
    if ((pres && !alloc((void**)&nr.dPres, (size_t)3 * nr.dw * nr.dh * sizeof(float))) ||
        (!planar &&
         (!alloc((void**)&nr.dVsrIn, (size_t)nr.w * nr.h * 4) || !alloc((void**)&nr.dVsrOut, (size_t)tw * th * 4))) ||
        (nr.hdr && (!alloc((void**)&nr.dVsrView, (size_t)3 * nr.w * nr.h * sizeof(float)) ||
                    !alloc((void**)&nr.dVsrRem, (size_t)3 * tw * th * sizeof(float)))))
    {
        why = "an allocation failed";
        return false;
    }
    nr.vsrWant = true;
    nr.fsrUp = want == 2 || want == 5;
    nr.fuHdr = nr.fsrUp && nr.hdr; // the start's rule (nativeRtxInit)
    nr.fuKind = want == 5 ? 1 : 0;
    if (nr.fsrUp)
        nr.fsrUpDir = dir;
    nr.a4kUp = want == 3;
    nr.acnUp = want == 4;
    nr.vsrFailed = nr.vsrUserOff = false;
    bool ok = true;
    if (want == 1 && g_rtxb.vsrMade)
        LOG("native: live upscale: RTX VSR runs the feature of this process's create (warmed then)\n");
    else if (want == 1)
    {
        if (g_rtxb.create(nullptr, nullptr, 0, 0u, 1u) != 1u)
        {
            why = "rtx_video_api_cuda_create failed";
            ok = false;
        }
        else
        {
            g_rtxb.created = g_rtxb.vsrMade = true;
            g_rtxUsed = true; // resident host: NGX is single-instance per process, end it after this session
            // the warm-up eval on an opaque black frame, as the session start runs it
            const RtxRect ri{0, 0, (uint32_t)nr.w, (uint32_t)nr.h};
            const RtxRect ro{0, 0, (uint32_t)tw, (uint32_t)th};
            if (cuMemsetD32Async((CUdeviceptr)nr.dVsrIn, 0xFF000000u, (size_t)nr.w * nr.h, (CUstream)nr.stream) !=
                    CUDA_SUCCESS ||
                cudaStreamSynchronize(nr.stream) != cudaSuccess ||
                g_rtxb.evalVsr(nr.dVsrIn, nr.dVsrOut, ri, ro, &nr.vsrSet) != 1u ||
                cudaDeviceSynchronize() != cudaSuccess)
            {
                why = "the RTX VSR warm-up eval failed";
                ok = false;
            }
        }
    }
    else if (nr.fsrUp)
        ok = nativeFsrupSetup(nr, nr.w, nr.h, tw, th);
    else
        ok = (nr.a4kUp ? nativeA4kSetup(nr, nr.w, nr.h, tw, th, 0) : nativeAcnSetup(nr, nr.w, nr.h, tw, th, 0)) &&
             !nr.vsrFailed;
    if (!ok)
    {
        // Lanczos stays: the flags off (the kind's buffers go with the session's teardown)
        nr.vsrWant = nr.fsrUp = nr.a4kUp = nr.acnUp = nr.vsrFailed = false;
        if (!nr.fuMade) // a bridge built before the failure is destroyed at the teardown by its kind
            nr.fuKind = 0;
        if (why.empty())
            why = "its setup failed (the line above)";
        return false;
    }
    if (nr.fsrUp)
    {
        // the first frames on a worker (nativeFsrupWarmProc); the session presents without the upscaler meanwhile,
        // nativeUpWarmEnd turns it on
        nr.vsrWant = nr.fsrUp = false;
        nr.fuWarmWant = true;
        nr.fuWarmT0 = GetTickCount64();
        InterlockedExchange(&nr.fuWarm, 1);
        nr.fuWarmTh = CreateThread(nullptr, 0, nativeFsrupWarmProc, &nr, 0, nullptr);
        if (!nr.fuWarmTh)
        {
            InterlockedExchange(&nr.fuWarm, 0);
            why = "its warm-up thread did not start";
            return false;
        }
        LOG("native: live upscale: %s %dx%d -> %dx%d, its first frames on a worker (the driver may compile its "
            "shaders), the picture goes on without it\n",
            name, nr.w, nr.h, tw, th);
        return true;
    }
    nr.vsr = true;
    LOG("native: live upscale: %s %dx%d -> %dx%d, started in place\n", name, nr.w, nr.h, tw, th);
    return true;
}

// Restore ticked in a running session that started without it (stdin `set restore N`: 1 Real-ESRGAN, 2 AnimeJaNai =
// the file the `restoremodel` key holds, 3 Anime4K's Restore CNN UL), set up on the native thread between two groups
// the way the session start does: the capture as Restore's source (restPre, srcW / srcH = the capture; every reader
// of those is Restore's own), the fold to the model size, the buffers the handoff allocates (dRestTmp, dRest, dCapF,
// dRestRem on HDR planes at a size change), then restNatSetup (weights, activations, warm-up), when the video memory
// room holds them. Left to the next session: another route, a start in place that failed earlier (its buffers stay
// for the teardown). false = why says it
static bool nativeRestoreSwitch(NativeRife& nr, int want, std::string& why)
{
    if (g_offline || g_backend != BK_SERVER)
    {
        why = "this route runs no native pass";
        return false;
    }
    if (want < 1 || want > 3)
    {
        why = "an unknown Restore kind";
        return false;
    }
    if (nr.dRest)
    {
        why = "its start in place failed earlier in this session";
        return false;
    }
    // the kind's model: Real-ESRGAN's weights readable, the held AnimeJaNai file in its form, the Anime4K module
    if (want == 1)
    {
        std::vector<uint8_t> blob;
        std::vector<size_t> off;
        if (!restNatLoad(blob, off, why))
            return false;
    }
    else if (want == 2 && g_restoreModel.empty())
    {
        why = "no AnimeJaNai file named (restoremodel)";
        return false;
    }
    else if (want == 2 && !ajNatWanted())
    {
        why = "the AnimeJaNai file is not in its SPAN-F3 form (the line above)";
        return false;
    }
    else if (want == 3 && !nativeA4kModule(nr, why))
        return false;
    const int cw = nr.cw, ch = nr.ch;
    const uint64_t fl = sizeof(float), cp = (uint64_t)cw * ch, mp = (uint64_t)nr.w * nr.h;
    const bool rem = (nr.encPre || nr.encPost) && (nr.w != cw || nr.h != ch);
    // the kind's buffers at the capture size (restNatSetup / nativeA4kRestSetup reads srcW / srcH): the input planes,
    // Real-ESRGAN / AnimeJaNai's sub-pixel output and activations (none where the shared activation block holds
    // them), Anime4K's layer slots, output and statistics; the weights (a few MB) sit inside the room's reserve
    nr.restNatKind = want;
    nr.restNat = true;
    nr.sw = cw;
    nr.sh = ch;
    auto undo = [&nr]() {
        nr.restNatKind = 0;
        nr.restNat = false;
        nr.sw = nr.sh = 0;
    };
    uint64_t kindB = nr.dRestIn ? 0 : 3 * cp * sizeof(uint16_t);
    if (want == 3)
        kindB += 4 * 3 * cp * 8 + 3 * cp * fl + (nr.a4kStatCap >= cp ? 0 : 2 * cp * sizeof(uint16_t));
    else
    {
        const uint64_t sub = want == 2 ? (uint64_t)((cw + 1) / 2) * ((ch + 1) / 2) : cp;
        const uint64_t pp = restNatBytes(nr);
        kindB += 48 * sub * sizeof(uint16_t) + (nr.dGmCtxMem && nr.gmCtxMemB >= (int64_t)pp ? 0 : pp);
    }
    const uint64_t need = 3 * 4 * (uint64_t)ch * nr.w * fl + 3 * mp * fl + (nr.dCapF ? 0 : 3 * cp * fl) +
                          (rem && !nr.dRestRem ? 3 * mp * fl : 0) + kindB;
    const uint64_t room = nativeVideoMemoryRoom();
    if (room && room < need)
    {
        undo();
        why = "not enough free video memory";
        return false;
    }
    auto alloc = [](void** p, size_t bytes) {
        if (*p)
            return true;
        if (cudaMalloc(p, bytes) == cudaSuccess)
            return true;
        *p = nullptr;
        cudaGetLastError();
        return false;
    };
    if (!alloc((void**)&nr.dRestTmp, (size_t)3 * 4 * ch * nr.w * sizeof(float)) ||
        !alloc((void**)&nr.dRest, (size_t)3 * mp * sizeof(float)) ||
        !alloc((void**)&nr.dCapF, (size_t)3 * cp * sizeof(float)) ||
        (rem && !alloc((void**)&nr.dRestRem, (size_t)3 * mp * sizeof(float))))
    {
        undo();
        why = "an allocation failed";
        return false;
    }
    nr.restTw = nr.w;
    nr.restTh = nr.h;
    nr.restFailed = nr.restUserOff = false;
    // the frames in flight first: Anime4K's statistics buffers, shared with its upscaler, may grow to the capture size
    cudaStreamSynchronize(nr.stream);
    if (!restNatSetup(nr))
    {
        cudaGetLastError();
        undo();
        why = "its setup failed (the line above)";
        return false;
    }
    nr.restore = nr.restPre = true;
    const std::string name = want == 1   ? std::string("Real-ESRGAN animevideov3")
                             : want == 3 ? std::string("Anime4K Restore CNN UL")
                                         : g_restoreModel.substr(g_restoreModel.find_last_of("\\/") + 1);
    LOG("native: live restore: %s at %dx%d -> %dx%d, started in place on the captured frame before the model\n",
        name.c_str(), cw, ch, nr.w, nr.h);
    return true;
}

// Anime4K's Mode A+A in a running session (stdin `set a4kaa 0|1`): off = the per-pass flag (the buffers stay for the
// session), on = Anime4K's resize buffers freed and set up again at the session's geometry with g_a4kAA set, the way
// the session start runs it (the room first: short = Mode A, nativeA4kSetup's line); Anime4K not set up in this
// session = held for its start in place (nativeUpSwitch's setup reads g_a4kAA). An Anime4K the user unticked stays
// off; a setup that fails = Lanczos for the session (vsrFailed, its buffers freed). The `live set` line's note, "" =
// applied
static std::string nativeA4kAaSwitch(NativeRife& nr, bool on)
{
    g_a4kAA = on;
    if (!nr.a4kUp || !nr.a4kW)
        return " (held for Anime4K's start)";
    if (!on || nr.a4kAA)
    {
        nr.a4kAA = on;
        return "";
    }
    if (nr.vsrFailed && !nr.vsrUserOff)
        return " (the next session takes it: Anime4K stopped earlier in this session)";
    const int w = nr.a4kW, h = nr.a4kH, ow = nr.a4kOw, oh = nr.a4kOh;
    const bool userOff = nr.vsrUserOff;
    cudaDeviceSynchronize();
    nativeA4kUpFree(nr);
    nr.vsrFailed = false;
    if (!nativeA4kSetup(nr, w, h, ow, oh, 0) || nr.vsrFailed)
    {
        cudaDeviceSynchronize();
        nativeA4kUpFree(nr);
        nr.vsrFailed = true;
        nr.vsrUserOff = false;
        return " (Anime4K stopped: its setup failed, the line above; Lanczos for the session)";
    }
    nr.vsrFailed = userOff;
    return nr.a4kAA ? "" : " (the next session takes it: not enough free video memory, Mode A runs)";
}

// the end of a DLSS 5 pass growth: the lines a growth in one go logs (the passes built, then the set), the run count
// set once to the passes asked (as many as were built); cut = why it ended before its last pass, nullptr = it ran out
static void nativeNrGrowEnd(NativeRife& nr, const char* cut)
{
    const int now = nr.nrHost ? nr.nrHost->passes() : nr.nrGrowFrom;
    if (now > nr.nrGrowFrom)
        LOG("live DLSS 5: %d more passes built in %.0f ms (%d now)%s%s\n", now - nr.nrGrowFrom, nr.nrGrowMs, now,
            nr.nrGrowWhy.empty() ? "" : ": ", nr.nrGrowWhy.c_str());
    std::string why = cut ? std::string(cut) : nr.nrGrowWhy;
    if (why.empty() && now < nr.nrGrowWant)
        why = "no video memory for more";
    const int run = nr.nrHost ? nr.nrHost->runPasses(nr.nrGrowWant) : 0;
    if (run < nr.nrGrowWant)
        LOG("live set: nrpasses %s (%d run: %s)\n", nr.nrGrowVal.c_str(), run, why.c_str());
    else
        LOG("live set: nrpasses %s\n", nr.nrGrowVal.c_str());
    nr.nrGrowWant = 0;
    nr.nrGrowVal.clear();
    nr.nrGrowWhy.clear();
}

// one step of a DLSS 5 pass growth, on the native thread before a frame group: ONE more pass (its feature and frame,
// nr::Host::growPasses) when the video memory room holds its working memory (nr::kFrameMp a megapixel), so the picture
// moves between the builds (a pass's build holds it ~100 ms at 640x360, 17 in one go ~1.9 s); the run count stays
// until the end (nativeNrGrowEnd), so the picture changes once, as a growth in one go does
static void nativeNrGrowStep(NativeRife& nr)
{
    if (!nr.nrGrowWant)
        return;
    if (!nr.liveNr || !nr.nrHost || nr.nrFailed)
    {
        nativeNrGrowEnd(nr, "DLSS 5 stopped");
        return;
    }
    const int had = nr.nrHost->passes();
    if (had < nr.nrGrowWant)
    {
        const uint64_t per = (uint64_t)(nr::kFrameMp * nr.nrW * nr.nrH / 1e6 * 1048576.0);
        const uint64_t room = nativeVideoMemoryRoom();
        if (!room || !per || room >= per)
        {
            const auto t0 = std::chrono::steady_clock::now();
            const int now = nr.nrHost->growPasses(had + 1, nr.nrGrowWhy);
            nr.nrGrowMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (now > had && now < nr.nrGrowWant && nr.nrGrowWhy.empty())
                return; // the next group builds the next pass
        }
    }
    nativeNrGrowEnd(nr, nullptr);
}

static void nativeLiveSets(NativeRife& nr)
{
    std::vector<std::pair<std::string, std::string>> q;
    if (nr.fuWarmTh && InterlockedCompareExchange(&nr.fuWarm, 0, 0) >= 2)
        nativeUpWarmEnd(nr); // an FSR / XeSS start in place: its first frames ended on the worker
    nativeNrGrowStep(nr);    // a DLSS 5 pass growth: one pass a frame group
    {
        std::lock_guard<std::mutex> lk(g_liveSets.m);
        if (g_liveSets.q.empty())
            return;
        q.swap(g_liveSets.q);
    }
    for (const auto& [key, val] : q)
    {
        const double v = atof(val.c_str());
        auto clampU = [](double x, double lo, double hi) { return (uint32_t)(x < lo ? lo : (x > hi ? hi : x)); };
        auto clampF = [](double x, double lo, double hi) { return (float)(x < lo ? lo : (x > hi ? hi : x)); };
        bool done = true;
        const bool nrOn = nr.liveNr && nr.nrHost && !nr.nrFailed;
        if (key == "sharpen" && v > 0.0 && (nr.sharpen > 0.0f || nativeSharpenOn(nr)))
            nr.sharpen = clampF(v, 0.0, 2.0);
        else if (key == "sharpen" && v <= 0.0)
            nr.sharpen = 0.0f;
        else if (key == "nrstruct" && nrOn)
            nr.nrHost->setStructure(clampF(v, 0.0, 2.0));
        else if (key == "nrtone" && nrOn)
            nr.nrHost->setTone(clampF(v, 0.0, 2.0));
        else if (key == "nrskin" && nrOn) // < 0 = follows the structure value
            nr.nrHost->setSkin(v < 0.0 ? -1.0f : clampF(v, 0.0, 2.0));
        else if (key == "nrautomask" && nrOn)
            nr.nrHost->setAutoMask(v != 0.0);
        else if (key == "nrstyle" && nrOn)
            nr.nrHost->setStyle((int)clampU(v, 0.0, 2.0));
        else if (key == "nrpasses" && nrOn)
        {
            // more passes than the session built: their features and frames one pass a frame group
            // (nativeNrGrowStep), as many as the video memory room holds; the growth's end logs this set's line. A new
            // value during a growth ends it first (its line names the cut)
            const int want = (int)clampU(v, 1.0, (double)nr::kMaxPasses);
            if (nr.nrGrowWant)
                nativeNrGrowEnd(nr, "another nrpasses arrived");
            if (want > nr.nrHost->passes())
            {
                nr.nrGrowWant = want;
                nr.nrGrowFrom = nr.nrHost->passes();
                nr.nrGrowMs = 0.0;
                nr.nrGrowVal = val;
                nr.nrGrowWhy.clear();
                nativeNrGrowStep(nr); // the first pass before this group
                continue;
            }
            nr.nrHost->runPasses(want);
        }
        else if ((key == "nrstruct" || key == "nrtone" || key == "nrskin" || key == "nrautomask" || key == "nrpasses" ||
                  key == "nrstyle") &&
                 !nrOn)
        {
            // DLSS 5 not running here: held for its start in place (key dlssnr)
            if (key == "nrstruct")
                g_nrStructure = clampF(v, 0.0, 2.0);
            else if (key == "nrtone")
                g_nrTone = clampF(v, 0.0, 2.0);
            else if (key == "nrskin")
                g_nrSkin = v < 0.0 ? -1.0 : (double)clampF(v, 0.0, 2.0);
            else if (key == "nrautomask")
                g_nrAutoMask = v != 0.0 ? 1 : 0;
            else if (key == "nrpasses")
                g_nrPasses = (int)clampU(v, 1.0, (double)nr::kMaxPasses);
            else
                g_nrStyle = (int)clampU(v, 0.0, 2.0);
            LOG("live set: %s %s (held for DLSS 5's start)\n", key.c_str(), val.c_str());
            continue;
        }
        else if (key == "dlssnr")
        {
            std::string why;
            if (!nativeNrSwitch(nr, v != 0.0, why))
            {
                LOG("live set: dlssnr %s (the next session takes it: %s)\n", val.c_str(), why.c_str());
                continue;
            }
        }
        else if (key == "hudlat" && !g_noHud)
            g_noHudLat.store(v == 0.0);
        else if (key == "hud")
            g_hudWant.store(v != 0.0 ? 1 : 0); // the loop thread owns the window (applyHudWant)
        else if (key == "restoremodel")
        {
            // AnimeJaNai's file (a path): held for Restore's start in place (`restore 2`); a running AnimeJaNai keeps
            // the file it loaded
            const bool ajRuns = nr.restore && nr.restNatKind == 2;
            if (ajRuns && val != g_restoreModel)
            {
                LOG("live set: restoremodel %s (the next session takes it: AnimeJaNai runs its file)\n", val.c_str());
                continue;
            }
            g_restoreModel = val;
            if (!ajRuns)
            {
                LOG("live set: restoremodel %s (held for Restore's start)\n", val.c_str());
                continue;
            }
        }
        else if (key == "a4kaa")
        {
            const std::string note = nativeA4kAaSwitch(nr, v != 0.0);
            if (!note.empty())
            {
                LOG("live set: a4kaa %s%s\n", val.c_str(), note.c_str());
                continue;
            }
        }
        else if (key == "up" || key == "restore" || key == "rtxhdr")
        {
            // the session's own kind (up: 1 RTX VSR, 2 AMD FSR 4, 3 Anime4K, 4 ArtCNN, 5 Intel XeSS; restore: 1
            // Real-ESRGAN, 2 AnimeJaNai, 3 Anime4K; rtxhdr: 1), 0 = it runs none
            const int kind = key == "up"        ? (!nr.vsr    ? 0
                                                   : nr.fsrUp ? (nr.fuKind ? 5 : 2)
                                                   : nr.a4kUp ? 3
                                                   : nr.acnUp ? 4
                                                              : 1)
                             : key == "restore" ? (nr.restore && nr.restNat ? nr.restNatKind : 0)
                                                : (nr.rtxHdr ? 1 : 0);
            bool& failed = key == "up" ? nr.vsrFailed : key == "restore" ? nr.restFailed : nr.rtxFailed;
            bool& userOff = key == "up" ? nr.vsrUserOff : key == "restore" ? nr.restUserOff : nr.rtxUserOff;
            const int want = (int)clampU(v, 0.0, 9.0);
            if (key == "up" && nr.fuWarmTh)
            {
                // an FSR / XeSS start in place still warming: the tick's last state decides at its end
                const int pend = nr.fuKind ? 5 : 2;
                if (want == 0 || want == pend)
                {
                    nr.fuWarmWant = want != 0;
                    LOG("live set: up %s (%s)\n", val.c_str(),
                        want ? "its start in place is under way" : "off once its start in place ends");
                }
                else
                    LOG("live set: up %s (the next session takes it: another kind is starting in place)\n",
                        val.c_str());
                continue;
            }
            const char* why = nullptr;
            if (want == 0 && kind)
            {
                if (!failed)
                    failed = userOff = true;
            }
            else if (want && want == kind)
            {
                if (userOff)
                    failed = userOff = false;
                else if (failed)
                    why = "it stopped earlier in this session";
            }
            else if (want != kind)
                why = kind ? "another kind than this session runs" : "this session started without it";
            // RTX HDR on stays a session start: its start sets the picture path (the HDR route, an upscaler before the
            // model, the model's colour domain, the one NGX create)
            if (why && key == "rtxhdr" && want && !kind)
                why = "this session started without it: RTX HDR sets the picture path at the session's start";
            // an upscaler the session started without, or another kind than it runs: its setup in place when this
            // session can run it (the running kind released first, nativeUpRelease)
            std::string whyUp;
            const bool other = key == "up" && want && want != kind;
            const bool restIn = key == "restore" && want && !kind; // Restore in a session that started without it
            if (why && other && nativeUpSwitch(nr, want, whyUp))
                why = nullptr;
            else if (why && restIn && nativeRestoreSwitch(nr, want, whyUp))
                why = nullptr;
            else if (why && (other || restIn))
            {
                whyUp = std::string(why) + ": " + whyUp;
                why = whyUp.c_str();
            }
            if (why)
            {
                LOG("live set: %s %s (the next session takes it: %s)\n", key.c_str(), val.c_str(), why);
                continue;
            }
        }
        else if (key == "vsrq" && !nr.fsrUp && !nr.a4kUp && !nr.acnUp)
        {
            // RTX VSR's level: read at every evaluate while it runs; held for its start in place (nativeUpSwitch)
            nr.vsrSet.QualityLevel = vsrLevelOk((int)v) ? (uint32_t)v : clampU(v, 1.0, 4.0);
            if (!nr.vsr)
            {
                LOG("live set: vsrq %s (held for RTX VSR's start)\n", val.c_str());
                continue;
            }
        }
        else if (key == "hdrsat" && nr.rtxHdr)
            nr.thdr.Saturation = clampU(v, 0.0, 200.0);
        else if (key == "hdrcon" && nr.rtxHdr)
            nr.thdr.Contrast = clampU(v, 0.0, 200.0);
        else if (key == "hdrvib" && nr.rtxHdr)
            nr.rtxVib = clampF(v, 0.0, 1.0);
        else if (key == "hdrsb" && nr.rtxHdr)
            nr.rtxSb = clampF(v, 0.0, 1.0);
        else
            done = false;
        LOG("live set: %s %s%s\n", key.c_str(), val.c_str(), done ? "" : " (the next session takes it)");
    }
}

static void nativeThread(NativeRife* nrp, IDXGIAdapter1* adapter, HANDLE hTex, HANDLE hFence, HANDLE hOutBuf,
                         uint64_t outBytes, bool earlyDone)
{
    NativeRife& nr = *nrp;
    // earlyDone: PipeServer::beginNativeHandoff built everything but the output ring on its
    // own thread; this thread only binds the same device and finishes the ring
    if (!earlyDone)
        nativeVideoMemoryMark();
    bool ok = earlyDone ? nativeBindDevice(nr)
                        : nativeCudaDeviceInit(nr, adapter, hTex, hFence, (uint32_t)nr.cw, (uint32_t)nr.ch, nr.hdr) &&
                              nativeCudaInitEarly(nr, nativeCacheDir(nr)) && nativeModelInit(nr) && nativeRtxInit(nr) &&
                              nativeLiveNrInit(nr);
    ok = ok && nativeCudaInitLate(nr, hOutBuf, outBytes);
    {
        std::lock_guard<std::mutex> lk(nr.mInit);
        nr.initState = ok ? 1 : -1;
    }
    nr.cvInit.notify_all();
    if (!ok)
    {
        InterlockedExchange(&nr.dead, 1);
        nr.cvTok.notify_all();
        nativeFree(nr);
        return;
    }
    LOG("native host ready: model %dx%d padded %dx%d, out %dx%d at (%d,%d), batch max %d%s%s%s%s%s%s%s%s%s%s%s\n", nr.w,
        nr.h, nr.pw, nr.ph, nr.dw, nr.dh, nr.x0, nr.y0, nr.batchMax, nr.noEngine ? ", engine=none (effects only)" : "",
        nr.gmfss ? ", engine=gmfss" : "", nr.nvof ? ", engine=nvof" : "",
        nr.fruc ? (nr.fsrfg ? ", engine=fsrfg"
                   : nr.vfg ? ", engine=nvvfg"
                            : ", engine=fruc")
                : "",
        nr.drba ? ", engine=drba (lag 1)" : "", nr.rtxHdr ? ", rtxhdr=native" : "",
        nr.sharpen > 0.0f ? ", sharpen=native" : "",
        nr.vsr ? (nr.fsrUp   ? (nr.fuKind ? ", xessup=native" : ", fsrup=native")
                  : nr.a4kUp ? ", anime4k=native"
                  : nr.acnUp ? ", artcnn=native"
                             : ", vsr=native")
               : "",
        nr.uw ? ", upscale=native" : "", nr.fitAa ? ", fit=native-aa" : "",
        nr.restore && nr.restNat ? (nr.restNatKind == 3 ? ", restore=anime4k" : ", restore=native") : "");
    for (;;)
    {
        std::vector<uint8_t> msg;
        {
            std::unique_lock<std::mutex> lk(nr.mMsg);
            nr.cvMsg.wait(lk, [&] { return !nr.msgs.empty() || nr.isDead(); });
            if (nr.msgs.empty())
                break;
            msg = std::move(nr.msgs.front());
            nr.msgs.pop_front();
        }
        nativeLiveSets(nr);
        if (!nativeGroup(nr, msg))
            break;
    }
    {
        std::lock_guard<std::mutex> lk(nr.mTok);
        InterlockedExchange(&nr.dead, 1);
    }
    nr.cvTok.notify_all();
    nativeFree(nr);
}

// Start the in-process host. Returns false on ANY problem (one log line says why) and the
// session ends.
static bool nativeStart(NativeRife& nr, const std::wstring& script, const std::wstring& backend, uint32_t capW,
                        uint32_t capH, IDXGIAdapter1* adapter, HANDLE hTex, HANDLE hFence, HANDLE hOutBuf,
                        uint64_t outBytes, uint32_t slots, uint32_t pitch, size_t slotBytes, bool earlyDone)
{
    // earlyDone: PipeServer::beginNativeHandoff already ran the DLL load, the HDR config, the
    // engine handoff, the CUDA device init, the kernels, the buffers and the engine load on
    // its own thread; only the output ring (sized from the slot count) is left
    if (!nativeLoadDlls(script))
        return false;
    if (!earlyDone)
    {
        if (!nativeConfigHdr(nr))
            return false;
        if (!nativeHandoff(script, backend, capW, capH, nr))
            return false;
    }
    nr.slots = slots;
    nr.pitch = pitch;
    nr.slotBytes = slotBytes;
    nr.th = std::thread(nativeThread, &nr, adapter, hTex, hFence, hOutBuf, outBytes, earlyDone);
    std::unique_lock<std::mutex> lk(nr.mInit);
    nr.cvInit.wait(lk, [&] { return nr.initState != 0; });
    const bool ok = nr.initState > 0;
    lk.unlock();
    if (!ok)
    {
        if (nr.th.joinable())
            nr.th.join();
        LOG("native host init failed\n");
        return false;
    }
    nr.started = true;
    return true;
}

static void nativeStop(NativeRife& nr)
{
    if (!nr.started)
        return;
    nr.die("stopping");
    if (nr.th.joinable())
        nr.th.join();
    nr.started = false;
}

// a NativeRife the early thread initialised but no compute thread ever owned (a refusal in
// startNative, or a session that ends before it): free its CUDA state from the calling
// thread. Without a stream nothing CUDA was created (the stream is the first CUDA object), and
// the runtime DLL may not even be loaded, so no CUDA call is made at all then.
static void nativeDropEarly(NativeRife& nr)
{
    if (!nr.stream)
        return;
    nativeBindDevice(nr);
    nativeFree(nr);
}
