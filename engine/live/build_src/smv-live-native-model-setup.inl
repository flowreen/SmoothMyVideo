// part of smv-live-native.inl (included there, in place); not a standalone translation unit
// native DRBA's buffers (both routes): the ring of pictures at the pictures' pad, everything the
// motion search uses at the model's frame (RIFE's two domains: the motion frame, where the ring also
// keeps the motion frames and the IFNet's x; else the pictures' pad, the encode's fp16 frame in dXh's
// cur half)
// DRBA as its author runs it (NativeRife::drAuthor) in two domains, the default; SMV_DRBA_AUTHOR=0 = the network's
// frames as SMV pads RIFE's (the edge replicated to a multiple of 64, R, G, B) and the blend on the pictures
static bool drbaAuthorWanted(const NativeRife& nr)
{
    return nr.drba && nr.mph && lkEnv("SMV_DRBA_AUTHOR") != "0";
}

static bool nativeDrbaAlloc(NativeRife& nr)
{
    const size_t plane = (size_t)nr.ph * nr.pw, mplane = nr.mph ? (size_t)nr.mph * nr.mpw : plane;
    nr.drAuthor = drbaAuthorWanted(nr);
    nr.drSFid[0] = nr.drSFid[1] = -1;
    if (nr.drAuthor)
        NCHK(cudaMalloc((void**)&nr.dDrS, 6 * mplane * sizeof(float)), "alloc drba stretched pair");
    if (!nr.mph)
        NCHK(cudaMalloc((void**)&nr.dXh, 6 * plane * sizeof(uint16_t)), "alloc x fp16");
    NCHK(cudaMalloc((void**)&nr.dDrIBase, 4 * 3 * plane * sizeof(float)), "alloc drba frame ring");
    if (nr.mph)
        NCHK(cudaMalloc((void**)&nr.dDrMBase, 5 * 3 * mplane * sizeof(uint16_t)), "alloc drba motion ring");
    for (int i = 0; i < 4; i++)
    {
        nr.dDrI[i] = nr.dDrIBase + (size_t)i * 3 * plane;
        if (nr.mph)
            nr.dDrM[i] = nr.dDrMBase + (size_t)(3 - i) * 3 * mplane;
    }
    for (int i = 0; i < 2; i++)
    {
        // two domains: the blend reads its pictures straight from the ring (nativeDrbaPics), the IFNet its
        // motion frames (nativeDrbaTween)
        if (!nr.mph)
            NCHK(cudaMalloc((void**)&nr.dDrX[i], 6 * plane * sizeof(float)), "alloc drba x");
        NCHK(cudaMalloc((void**)&nr.drWin[i].f10, 2 * mplane * sizeof(float)), "alloc drba window");
        NCHK(cudaMalloc((void**)&nr.drWin[i].r, 4 * mplane * sizeof(float)), "alloc drba window");
    }
    NCHK(cudaMalloc((void**)&nr.dDrAcc, 6 * mplane * sizeof(long long)), "alloc drba accumulator");
    // DRBA as its author: the accumulator is idle from a tween's DrmNorm (and block0's FlowNorm) to its next clear,
    // and every reader of these two runs on nr.stream before that clear (the blend's bilinear back, the pack of a real
    // frame): the blend at the network's frame, also a real frame's stretch on its way back, takes the accumulator's
    // last 12 B a pixel; the real frame stretched back (12 B a picture pixel) takes its first 36 B a motion pixel when
    // the picture fits there (the pictures sit at the working size, the motion frame at the decoded size), else its
    // own block
    if (nr.drAuthor)
    {
        nr.dDrB = (float*)nr.dDrAcc + 9 * mplane;
        if (plane <= 3 * mplane)
            nr.dDrRT = (float*)nr.dDrAcc;
        else
        {
            NCHK(cudaMalloc((void**)&nr.dDrRT, 3 * plane * sizeof(float)), "alloc drba real frame");
            nr.drRtOwn = true;
        }
        LOG("native: DRBA as its author runs it: the network's %dx%d frame stretched bilinearly as B, G, R, every "
            "frame back to %dx%d (SMV_DRBA_AUTHOR=0 = the padded frame); the blend%s in the accumulator\n",
            nr.mpw, nr.mph, nr.w, nr.h, nr.drRtOwn ? "" : " and the real frame");
    }
    return true;
}

static void nativeHdrModelSetup(NativeRife& nr, int pre, int post, float white);
static bool nativeCudaInitEarly(NativeRife& nr, const std::wstring& cacheDir)
{
    if (!nativeBuildKernels(nr, cacheDir))
        return false;
    // live: GMFSS and Restore read the HDR desktop's PQ planes as SDR sRGB over its SDR white, decided before
    // the engines and the RTX Video setup (a helper thread) size their buffers; offline sets it from the source
    if (!g_offline)
        nativeHdrModelSetup(nr, nr.hdr ? 1 : 0, nr.hdr ? 1 : 0, (float)(g_sdrWhite / 80.0));
    // the live working size the handoff sized the model frame with (liveWorkSize), and where it came from
    if (!g_offline)
    {
        int presW, presH, mode, mw, mh, raised;
        liveWorkSize(nr.cw, nr.ch, presW, presH, mode, mw, mh, &raised);
        const int shown = mode ? mode : (g_flowScale >= 0.995 ? 1 : 0); // no --scale = DLAA
        char how[64];
        if (shown)
            sprintf_s(how, "DLSS mode %ls", kDlssModeName[shown - 1]);
        else
            sprintf_s(how, "%.2f", g_flowScale);
        LOG("native: live working size %dx%d = %s of the %dx%d presented, capture %dx%d%s\n", nr.w, nr.h, how, presW,
            presH, nr.cw, nr.ch,
            raised == 3   ? " (raised to DLSS 4.5's minimum, 128 px a side, aspect kept)"
            : raised == 2 ? " (raised to DLSS 5's minimum width, 128 px, aspect kept)"
            : raised == 1 ? " (raised to the 64 px floor, aspect kept)"
                          : "");
    }
    // live: Restore reads the captured frame and folds it to the model size before the model runs, so srcW /
    // srcH name the capture from here on (Restore's setup in nativeModelInit and its target in nativeRtxInit read
    // them, possibly on two threads)
    if (nr.restore)
    {
        nr.restPre = true;
        nr.sw = nr.cw;
        nr.sh = nr.ch;
    }

    const int capBpp = nr.hdr ? 8 : 4;
    const size_t plane = (size_t)nr.ph * nr.pw;
    NCHK(cudaMalloc((void**)&nr.dCap, (size_t)nr.cw * nr.ch * capBpp), "alloc capture staging");
    if (!nr.drba) // DRBA builds each frame in its ring slot: no dX
        NCHK(cudaMalloc((void**)&nr.dX, 6 * plane * sizeof(float)), "alloc x");
    // the identical-pair flag and its pinned readback (one int per group)
    NCHK(cudaMalloc((void**)&nr.dStaticFlag, sizeof(int)), "alloc static flag");
    NCHK(cudaHostAlloc((void**)&nr.hStaticFlag, sizeof(int), cudaHostAllocDefault), "alloc static flag readback");
    // no-engine mode: the packed frame (the cur half of x) is the whole model side; a GMFSS
    // session shares only dX (its chain's own buffers are sized off the engines in
    // nativeGmfssSetup), so neither needs the RIFE encode buffers (the timestep and merged
    // buffers follow the engine's dtypes: nativeTweenBuffers)
    // RIFE's two domains: the features and the motion frames (instead of the fp16 copy of x) at the
    // motion frame, plus the shrink of an enlarged working-size picture back to it
    if (!nr.noEngine && !nr.gmfss && !nr.nvof && !nr.fruc && !nr.drba)
    {
        const size_t mplane = nr.mph ? (size_t)nr.mph * nr.mpw : plane;
        if (nr.mph)
            NCHK(cudaMalloc((void**)&nr.dM, 6 * mplane * sizeof(uint16_t)), "alloc motion frames");
        else
            NCHK(cudaMalloc((void**)&nr.dXh, 6 * plane * sizeof(uint16_t)), "alloc x fp16");
        if (nr.mph && (nr.mw != nr.w || nr.mh != nr.h))
            NCHK(cudaMalloc((void**)&nr.dMFit, (size_t)3 * nr.mw * nr.mh * sizeof(float)), "alloc motion fit");
    }
    // native DRBA: the history ring of padded frames and encodes replaces f0 / f1
    if (nr.drba)
    {
        if (!nativeDrbaAlloc(nr))
            return false;
        if (nr.mph && (nr.mw != nr.w || nr.mh != nr.h))
            NCHK(cudaMalloc((void**)&nr.dMFit, (size_t)3 * nr.mw * nr.mh * sizeof(float)), "alloc motion fit");
    }
    // live Restore: the fold's horizontal pass at the model's output height (4x, the largest scale) and the model-size
    // target, and the folded model-size frame (the model's input / output follow in nativeModelInit's restoreReady)
    if (nr.restore)
    {
        const size_t mp = (size_t)nr.w * nr.h;
        NCHK(cudaMalloc((void**)&nr.dRestTmp, (size_t)3 * 4 * srcH(nr) * nr.w * sizeof(float)),
             "alloc restore fold pass");
        NCHK(cudaMalloc((void**)&nr.dRest, 3 * mp * sizeof(float)), "alloc restore model-size frame");
    }
    // With live TrueHDR the PQ frame always exists at capture resolution first, so
    // the resize pair runs at EVERY image scale (at 1.00 the triangle filter sits on identity
    // positions and is an exact copy) and both staging planes are needed even at scale 1.00.
    if (nr.w != nr.cw || nr.h != nr.ch || nr.rtxHdr)
    {
        // also nativeFitPlanar's pass (3, source height, target width): the capture -> working size of Restore's
        // HDR reference and of the stage RTX VSR falls back from (ch x w), the motion frame's shrink (h x mw)
        nr.tmpN = (size_t)3 * (std::max)((size_t)nr.ch * nr.w, nr.dMFit ? (size_t)nr.h * nr.mw : (size_t)0);
        NCHK(cudaMalloc((void**)&nr.dTmp, nr.tmpN * sizeof(float)), "alloc resize temp");
        // HDR rescales ON PQ, so the conversion runs at capture resolution first
        if (nr.hdr)
            NCHK(cudaMalloc((void**)&nr.dCapF, (size_t)3 * nr.ch * nr.cw * sizeof(float)), "alloc PQ capture plane");
    }
    // Restore reads the captured frame as capture-size planes (PQ in HDR)
    if (nr.restPre && !nr.dCapF)
        NCHK(cudaMalloc((void**)&nr.dCapF, (size_t)3 * nr.ch * nr.cw * sizeof(float)), "alloc capture planes");
    NCHK(cudaEventCreateWithFlags(&nr.capEv, cudaEventDisableTiming), "create capture event");
    if (!g_offline)
    {
        for (cudaEvent_t& e : nr.pfEv)
            NCHK(cudaEventCreate(&e), "create frame timing event");
        for (cudaEvent_t& e : nr.grEv)
            NCHK(cudaEventCreate(&e), "create group timing event");
        for (auto& s : nr.fgL1Ev)
            for (cudaEvent_t& e : s)
                NCHK(cudaEventCreate(&e), "create fsrfg call timing event");
        for (auto& s : nr.fgOfEv)
            for (cudaEvent_t& e : s)
                NCHK(cudaEventCreate(&e), "create fsrfg Optical Flow timing event");
        NCHK(cudaEventCreate(&nr.grLagEv), "create fsrfg slot timing event");
    }
    // the nvof model: the Optical Flow session and the glue buffers (the kernels are bound above)
    if (nr.nvof && !nativeNvofSetup(nr))
        return false;
    // Smooth Motion: the bridge's FRUC instance and its BGRA8 surfaces
    if (nr.fruc && !nativeFrucSetup(nr))
        return false;
    return true;
}

// the model cache folder the handoff named (the kernels' cubins live there)
static std::wstring nativeCacheDir(const NativeRife& nr)
{
    return std::wstring(nr.cachePath.begin(), nr.cachePath.end());
}

// Live RTX TrueHDR runs in this process. Everything comes from the exe's own
// globals, parsed from the app's RTX HDR controls, so no new flag exists. Runs after
// nativeLoadDlls (the bridge folder is derived from the runtime folder found there).
// srcHdr = offline HDR video (1 PQ, 2 HLG): the upscalers' HDR levers apply to it as to live's HDR planes
static bool nativeConfigHdr(NativeRife& nr, int srcHdr = 0)
{
    // live Sharpen and RTX VSR read the exe's own globals; VSR needs the bridge and nvngx_vsr.dll present (else
    // Lanczos, never a route change), and the fit must enlarge (decided in nativeRtxInit once the handoff geometry is
    // known). It takes SDR, so on HDR planes it reads their SDR view (SMV_VSR_HDR=0 = Lanczos there)
    nr.sharpen = (float)(g_sharpen < 0.0 ? 0.0 : (g_sharpen > 2.0 ? 2.0 : g_sharpen));
    // RTX VSR's quality level (--vsr-quality, the app's slider): 1..4, 4 = Ultra by default; level 0 (bicubic) is left
    // out, since an unticked box already gives Lanczos. The network runs at the input size, so a level's price is the same
    // at any output size. 16..19 = the same levels of NVIDIA's high-bitrate models (the app's High quality source tick)
    nr.vsrSet.QualityLevel = (uint32_t)(vsrLevelOk(g_vsrQuality) ? g_vsrQuality : (g_vsrQuality < 1 ? 1 : 4));
    if (g_rtxVsr)
    {
        if (g_vsrQuality != (int)nr.vsrSet.QualityLevel)
            LOG("native: --vsr-quality %d is outside 1..4, RTX VSR runs quality %u\n", g_vsrQuality,
                nr.vsrSet.QualityLevel);
        else if (nr.vsrSet.QualityLevel > 4)
            LOG("native: RTX VSR for a high quality source: quality %u of 4 of NVIDIA's high-bitrate models, no removal of "
                "compression artifacts\n",
                nr.vsrSet.QualityLevel - 15);
        else if (nr.vsrSet.QualityLevel != 4)
            LOG("native: RTX VSR quality %u of 4\n", nr.vsrSet.QualityLevel);
        wchar_t vh[8]{};
        if ((g_hdr || srcHdr) && GetEnvironmentVariableW(L"SMV_VSR_HDR", vh, 8) && vh[0] == L'0')
            LOG(g_hdr ? "native: live RTX VSR off on HDR planes (SMV_VSR_HDR=0), Lanczos\n"
                      : "native: RTX VSR off on HDR video (SMV_VSR_HDR=0), Lanczos\n");
        else
        {
            std::wstring dir;
            wchar_t ov[MAX_PATH]{};
            if (GetEnvironmentVariableW(L"SMV_RTXVIDEO_DIR", ov, MAX_PATH) && ov[0])
                dir = ov;
            else if (!g_nativeEngineDir.empty())
                dir = g_nativeEngineDir + L"\\rtxvideo";
            const bool have = !dir.empty() &&
                              GetFileAttributesW((dir + L"\\rtxvideo_cuda.dll").c_str()) != INVALID_FILE_ATTRIBUTES &&
                              GetFileAttributesW((dir + L"\\nvngx_vsr.dll").c_str()) != INVALID_FILE_ATTRIBUTES;
            if (!have)
                LOG("native: live RTX VSR unavailable (no rtxvideo bridge or nvngx_vsr.dll), Lanczos\n");
            else if (!rtxBridgeLoad())
                return false;
            else
                nr.vsrWant = true;
        }
    }
    // AMD FSR upscaling (--fsr-upscale) takes RTX VSR's resize under its rules (enlarging only); the two exclude each
    // other (the app ticks one), RTX VSR wins a command line with both. On HDR planes it runs in the bridge's HDR mode
    // (linear light, nativeFsrupSetup; SMV_FSRUP_HDR=0 = Lanczos there)
    wchar_t fh[8]{};
    if (g_fsrUp && g_rtxVsr)
        LOG("native: --fsr-upscale ignored: RTX VSR holds the resize (the two exclude each other)\n");
    else if (g_fsrUp && (g_hdr || srcHdr) && GetEnvironmentVariableW(L"SMV_FSRUP_HDR", fh, 8) && fh[0] == L'0')
        LOG(g_hdr ? "native: AMD FSR upscaling off on HDR planes (SMV_FSRUP_HDR=0), Lanczos\n"
                  : "native: AMD FSR upscaling off on HDR video (SMV_FSRUP_HDR=0), Lanczos\n");
    else if (g_fsrUp)
    {
        std::string dir = lkEnv("SMV_FSRUP_DIR");
        if (dir.empty() && !g_nativeEngineDir.empty())
            dir = wideToUtf8(g_nativeEngineDir) + "\\fsrup";
        bool have = !dir.empty();
        for (const char* f :
             {"smv_fsrup_bridge.dll", "amd_fidelityfx_loader_dx12.dll", "amd_fidelityfx_upscaler_dx12.dll"})
            have = have && lkFile(dir + "\\" + f);
        if (!have)
            LOG("native: AMD FSR upscaling unavailable (smv_fsrup_bridge.dll and AMD's FidelityFX loader + upscaler "
                "needed in %s), Lanczos\n",
                dir.c_str());
        else
        {
            nr.vsrWant = true;
            nr.fsrUp = true;
            nr.fsrUpDir = dir;
        }
    }
    // Anime4K (--anime4k) takes RTX VSR's resize under the same rules; it excludes RTX VSR and AMD FSR upscaling the
    // way those two exclude each other (either of them wins a command line with both). Its kernels build at the
    // session's setup (nativeA4kSetup)
    if (g_a4kUp && (g_rtxVsr || g_fsrUp))
        LOG("native: --anime4k ignored: %s holds the resize (the upscalers exclude each other)\n",
            g_rtxVsr ? "RTX VSR" : "AMD FSR upscaling");
    else if (g_a4kUp)
    {
        nr.vsrWant = true;
        nr.a4kUp = true;
    }
    // ArtCNN (--artcnn) the same way: any of the other three on the command line wins over it
    if (g_acnUp && (g_rtxVsr || g_fsrUp || g_a4kUp))
        LOG("native: --artcnn ignored: %s holds the resize (the upscalers exclude each other)\n",
            g_rtxVsr  ? "RTX VSR"
            : g_fsrUp ? "AMD FSR upscaling"
                      : "Anime4K");
    else if (g_acnUp)
    {
        nr.vsrWant = true;
        nr.acnUp = true;
    }
    // Intel XeSS-SR (--xess-upscale) runs on AMD FSR upscaling's route through its own bridge (fuKind 1:
    // smv_xess_bridge.dll with Intel's libxess.dll, which the app's XeSS setup installs into engine\xess); every other
    // upscaler on the command line wins over it; SMV_FSRUP_HDR=0 turns the route off on HDR for both bridges
    if (g_xessUp && (g_rtxVsr || nr.fsrUp || nr.a4kUp || nr.acnUp))
        LOG("native: --xess-upscale ignored: %s holds the resize (the upscalers exclude each other)\n",
            g_rtxVsr   ? "RTX VSR"
            : nr.fsrUp ? "AMD FSR upscaling"
            : nr.a4kUp ? "Anime4K"
                       : "ArtCNN");
    else if (g_xessUp && (g_hdr || srcHdr) && GetEnvironmentVariableW(L"SMV_FSRUP_HDR", fh, 8) && fh[0] == L'0')
        LOG(g_hdr ? "native: Intel XeSS upscaling off on HDR planes (SMV_FSRUP_HDR=0), Lanczos\n"
                  : "native: Intel XeSS upscaling off on HDR video (SMV_FSRUP_HDR=0), Lanczos\n");
    else if (g_xessUp)
    {
        std::string dir = lkEnv("SMV_XESS_DIR");
        if (dir.empty() && !g_nativeEngineDir.empty())
            dir = wideToUtf8(g_nativeEngineDir) + "\\xess";
        bool have = !dir.empty();
        for (const char* f : {"smv_xess_bridge.dll", "libxess.dll"})
            have = have && lkFile(dir + "\\" + f);
        if (!have)
            LOG("native: Intel XeSS upscaling unavailable (smv_xess_bridge.dll and Intel's libxess.dll needed in %s; the "
                "app's XeSS setup installs them), Lanczos\n",
                dir.c_str());
        else
        {
            nr.vsrWant = true;
            nr.fsrUp = true;
            nr.fsrUpDir = dir;
            nr.fuKind = 1;
        }
    }
    if (!(g_hdr && g_rtxHdr))
        return true;
    if (!rtxBridgeLoad())
        return false;
    nr.rtxHdr = true;
    nr.thdr.Contrast = (uint32_t)(g_hdrCon < 0 ? 0 : (g_hdrCon > 200 ? 200 : g_hdrCon));
    nr.thdr.Saturation = (uint32_t)(g_hdrSat < 0 ? 0 : (g_hdrSat > 200 ? 200 : g_hdrSat));
    nr.thdr.MiddleGray = 50; // the live default
    nr.thdr.MaxLuminance = g_thdrPeak;
    nr.sdrScale = (float)(g_sdrWhite / 80.0 < 0.5 ? 0.5 : g_sdrWhite / 80.0);
    nr.rtxMode = !wcscmp(g_hdrColor, L"rtx") ? 1 : (!wcscmp(g_hdrColor, L"raw") ? 2 : 0);
    nr.rtxVib = (float)(g_hdrVib < 0.0 ? 0.0 : (g_hdrVib > 1.0 ? 1.0 : g_hdrVib));
    nr.rtxSb = (float)(g_hdrSb < 0.0 ? 0.0 : (g_hdrSb > 1.0 ? 1.0 : g_hdrSb));
    return true;
}

// gmflow split: the sizes of the contract between `_m`, the scale-0 matching and propagation (k_attn2), A, k_localCorr
// and gmflow B, all on the host's kernels: the backbone writes each frame's map4 / map8 as planar fp16 [128, H, W] (A /
// `_m` take both frames' as [2, 128, H, W]); A hands out qf [4, H, W, 128] fp16 (channels-last; q = rows 0..1, f = rows
// 2..3) and flow0 fp32 [2, 2, H, W] on the quarter (gmH x gmW) of the half frame; `_m` hands out mf / mq / mk fp16 [2, N,
// 128] on the eighth (N = gmN8 tokens, gmW8 wide), the propagation writes prop fp16 [2, N, 2]; k_localCorr computes the
// window of K = kGmCorrK samples itself and writes corr [2, HW, K] for B. A's 8 x 8 windows need a quarter in eighths.
static bool nativeGmSplitCheck(NativeRife& nr)
{
    if (nr.hh <= 0 || nr.hw <= 0 || nr.hh % 32 || nr.hw % 32)
    {
        LOG("native: gmfss split: the half %dx%d is not a multiple of 32\n", nr.hw, nr.hh);
        return false;
    }
    nr.gmH = nr.hh / 4;
    nr.gmW = nr.hw / 4;
    nr.gmN8 = (nr.hh / 8) * (nr.hw / 8);
    nr.gmW8 = nr.hw / 8;
    nr.gmPropHalf = true;
    nr.gmK = kGmCorrK;
    return true;
}

// GMFSS: the chain's contract and its buffers: one pair's and one tween's planar buffers at the
// nets' channel counts and dtypes. Runs inside nativeModelInit, so a mismatch refuses the session
// before any group. The accumulator is one
// buffer for all eight splats (they are strictly sequential on the one stream): the widest is
// the half level's gmC[0] + 1 planes, and the quarter / eighth levels need less
// ((2 * gmC[1] + 1) / 4 and (2 * gmC[2] + 1) / 16 of a half plane at the stock 64/128/192).
// GMFSS's eight contexts enqueue one after another on nr.stream (their aux streams join it at each enqueue's end), so
// they share ONE activation block, the largest net's, instead of one each: set before the warm-up, so a whole-graph
// capture records its address. SMV_GM_CTXSHARE=0 = each net its own.
// Restore on the host's kernels: its channels-last activations (Real-ESRGAN: two 64-half frames at the source size;
// AnimeJaNai: eight at the half grid: x0, the first block's output, two block ping-pong frames, c1, c2, c3, the concat
// conv's output), 0 = not this route
static size_t restNatBytes(const NativeRife& nr)
{
    if (nr.restNatKind == 3)
        return 0; // Anime4K's Restore: its own layer slots (nativeA4kRestSetup)
    if (nr.restNatKind == 2)
        return 8 * (size_t)((srcW(nr) + 1) / 2) * (size_t)((srcH(nr) + 1) / 2) * 128;
    return nr.restNat ? 2 * (size_t)srcW(nr) * (size_t)srcH(nr) * 128 : 0;
}

// AnimeJaNai on st: dRestIn -> k_ajFirst -> per block 3 x k_restConvMid + k_ajAtt -> k_ajCat -> k_restConvLast ->
// dRestOut (48 fp16 sub-pixel planes at the half grid); false = a launch failed
static bool ajNatRun(NativeRife& nr, cudaStream_t st)
{
    int sw = srcW(nr), sh = srcH(nr), hw = nr.restSubW, hh = nr.restSubH, n = hw * hh;
    const size_t nb = (size_t)n * 128;
    auto at = [&](size_t i) { return (void*)((char*)nr.dRestW + nr.restNatOff[i]); };
    auto buf = [&](int i) { return (void*)((char*)nr.dRestA + (size_t)i * nb); };
    void *x0 = buf(0), *out1 = buf(1), *tA = buf(2), *tB = buf(3), *c1 = buf(4), *c2 = buf(5), *c3 = buf(6),
         *cat = buf(7);
    const unsigned gx = (unsigned)(hw + 31) / 32, gy = (unsigned)(hh + 7) / 8, gp = (unsigned)(n + 127) / 128,
                   smem = 10 * 34 * 128;
    void *w0 = at(0), *b0 = at(1);
    void* a0[] = {&nr.dRestIn, &sw, &sh, &x0, &w0, &b0, &hw, &hh};
    if (cuLaunchKernel(nr.fAjFirst, gx, gy, 1, 256, 1, 1, 0, (CUstream)st, a0, nullptr) != CUDA_SUCCESS)
        return false;
    void* t = x0;
    for (int k = 0; k < nr.restNatBlocks; k++)
    {
        void* io[3][2] = {{t, c1}, {c1, c2}, {c2, c3}};
        for (int j = 0; j < 3; j++)
        {
            const size_t o = (size_t)(2 + 11 * k + 3 * j);
            void *wf = at(o), *bs = at(o + 1), *sl = at(o + 2);
            void* a[] = {&io[j][0], &io[j][1], &wf, &bs, &sl, &hh, &hw};
            if (cuLaunchKernel(nr.fRestConvMid, gx, gy, 1, 128, 1, 1, smem, (CUstream)st, a, nullptr) != CUDA_SUCCESS)
                return false;
        }
        void* dst = k == 0 ? out1 : (t == tA ? tB : tA);
        void *wa = at((size_t)(11 + 11 * k)), *ba = at((size_t)(12 + 11 * k));
        void* aa[] = {&c3, &t, &dst, &wa, &ba, &n};
        if (cuLaunchKernel(nr.fAjAtt, gp, 1, 1, 128, 1, 1, 0, (CUstream)st, aa, nullptr) != CUDA_SUCCESS)
            return false;
        t = dst;
    }
    const size_t tail = (size_t)(2 + 11 * nr.restNatBlocks);
    void *wc = at(tail), *bc = at(tail + 1), *wl = at(tail + 2), *bl = at(tail + 3);
    void* ac[] = {&x0, &t, &out1, &c1, &cat, &wc, &bc, &n};
    if (cuLaunchKernel(nr.fAjCat, gp, 1, 1, 128, 1, 1, 0, (CUstream)st, ac, nullptr) != CUDA_SUCCESS)
        return false;
    void* al[] = {&cat, &nr.dRestOut, &wl, &bl, &hh, &hw};
    return cuLaunchKernel(nr.fRestConvLast, gx, gy, 1, 128, 1, 1, smem, (CUstream)st, al, nullptr) == CUDA_SUCCESS;
}

// one Restore on the host's kernels on st: dRestIn (three fp16 planes) -> k_restConv0 -> 16 x k_restConvMid ->
// k_restConvLast -> dRestOut (48 fp16 sub-pixel planes); AnimeJaNai = ajNatRun; false = a launch failed
static bool restNatRun(NativeRife& nr, cudaStream_t st)
{
    if (nr.restNatKind == 2)
        return ajNatRun(nr, st);
    int w = srcW(nr), h = srcH(nr);
    auto at = [&](int i) { return (void*)((char*)nr.dRestW + nr.restNatOff[(size_t)i]); };
    const unsigned gx = (unsigned)(w + 31) / 32, gy = (unsigned)(h + 7) / 8, smem = 10 * 34 * 128;
    void *w0 = at(0), *b0 = at(1), *s0 = at(2);
    void* a0[] = {&nr.dRestIn, &nr.dRestA, &w0, &b0, &s0, &h, &w};
    if (cuLaunchKernel(nr.fRestConv0, gx, gy, 1, 256, 1, 1, 0, (CUstream)st, a0, nullptr) != CUDA_SUCCESS)
        return false;
    void *s = nr.dRestA, *d = nr.dRestB;
    for (int L = 0; L < 16; L++)
    {
        void *wf = at(3 + 3 * L), *bs = at(4 + 3 * L), *sl = at(5 + 3 * L);
        void* a[] = {&s, &d, &wf, &bs, &sl, &h, &w};
        if (cuLaunchKernel(nr.fRestConvMid, gx, gy, 1, 128, 1, 1, smem, (CUstream)st, a, nullptr) != CUDA_SUCCESS)
            return false;
        std::swap(s, d);
    }
    void *wl = at(51), *bl = at(52);
    void* al[] = {&s, &nr.dRestOut, &wl, &bl, &h, &w};
    return cuLaunchKernel(nr.fRestConvLast, gx, gy, 1, 128, 1, 1, smem, (CUstream)st, al, nullptr) == CUDA_SUCCESS;
}

// Restore on the host's kernels for this session: the weights uploaded, dRestIn / dRestOut, the ping-pong (inside the
// shared activation block when that holds it: every enqueue there runs one after another on nr.stream, Restore
// first), one run on zeros; the engine route's facts it reads (fp16 in and out, 48 sub-pixel planes at x4)
static bool restNatSetup(NativeRife& nr)
{
    if (nr.restNatKind == 3)
        return nativeA4kRestSetup(nr);
    std::vector<uint8_t> blob;
    std::string why;
    const bool aj = nr.restNatKind == 2;
    if (!(aj ? ajNatLoad(g_restoreModel, blob, nr.restNatOff, nr.restNatBlocks, why)
             : restNatLoad(blob, nr.restNatOff, why)))
    {
        LOG("native: Restore on the host's kernels failed (%s)\n", why.c_str());
        return false;
    }
    nr.restHalfIn = nr.restHalfOut = true;
    nr.restSub = true;
    nr.restScale = aj ? 2 : 4;
    nr.restSubS = 4;
    nr.restSubW = aj ? (srcW(nr) + 1) / 2 : srcW(nr);
    nr.restSubH = aj ? (srcH(nr) + 1) / 2 : srcH(nr);
    nr.restSubBase = !aj;
    const size_t sp = (size_t)srcW(nr) * srcH(nr), pp = restNatBytes(nr);
    if (!nr.dRestW)
        NCHK(cudaMalloc(&nr.dRestW, blob.size()), "alloc restore weights");
    NCHK(cudaMemcpy(nr.dRestW, blob.data(), blob.size(), cudaMemcpyHostToDevice), "restore weights upload");
    if (!nr.dRestIn)
        NCHK(cudaMalloc(&nr.dRestIn, 3 * sp * sizeof(uint16_t)), "alloc restore input");
    if (!nr.dRestOut)
        NCHK(cudaMalloc(&nr.dRestOut, 48 * (size_t)nr.restSubW * nr.restSubH * sizeof(uint16_t)),
             "alloc restore output");
    if (nr.dGmCtxMem && nr.gmCtxMemB >= (int64_t)pp)
    {
        nr.dRestA = nr.dGmCtxMem;
        nr.restNatOwn = false;
    }
    else if (!nr.dRestA)
    {
        nr.restNatOwn = true;
        NCHK(cudaMalloc(&nr.dRestA, pp), "alloc restore activations");
    }
    nr.dRestB = (char*)nr.dRestA + pp / 2; // Real-ESRGAN's pong
    NCHK(cudaMemsetAsync(nr.dRestIn, 0, 3 * sp * sizeof(uint16_t), nr.stream), "restore warm-up clear");
    if (!restNatRun(nr, nr.stream))
    {
        LOG("native: restore warm-up launch failed\n");
        return false;
    }
    NCHK(cudaStreamSynchronize(nr.stream), "restore warm-up sync");
    LOG("native: Restore = %s on the host's kernels at %dx%d (fp16 accumulate; its %.0f MB of activations %s)\n",
        aj ? "AnimeJaNai (SPAN-F3)" : "Real-ESRGAN", srcW(nr), srcH(nr), pp / 1048576.0,
        nr.restNatOwn ? "on their own" : "in the shared activation block");
    return true;
}
