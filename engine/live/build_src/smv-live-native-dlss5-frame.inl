// part of smv-live-native.inl (included there, in place); not a standalone translation unit
// DLSS 5 on one frame, in place, before anything reads it: the padded planar model-input frame
// (pw x ph, plane stride ps) whose top-left nrW x nrH is the picture; offline a decoded frame,
// live a captured one after Restore and the resize ((R, G, B) planes, PQ in
// HDR). Every model and every output frame then reads DLSS 5 output: DLSS 5 on
// the rendered frame and frame generation after it. Reset on the stream's first frame (live: and
// after a pause).
// Zero-copy: k_nrIn's frame goes into the core's shared input buffer at its row pitch, the stream
// signals an odd fence value, the core's queue waits for it, evaluates, writes the shared output
// buffer and signals the next even value, and the stream waits for that before k_nrOut reads the
// result: no CPU wait. CPU staging: the core's renderFrame through its upload / readback
// buffers. An evaluate that fails turns the pass off for the rest of the render with a line and
// leaves the frame untouched (nr.nrFailed); there is no retry: NGX has no teardown, so the NR core
// cannot restart in this process.
static bool nativeNrFrame(NativeRife& nr, float* frame, int pw, int ph, int ps)
{
    cudaStream_t st = nr.stream;
    int tw = nr.nrW, th = nr.nrH;
    const size_t bytes = (size_t)tw * th * 8;
    // live HDR planes are PQ, so the SDR-range math runs around the pass (k_nrInPq / k_nrOutPq), unless
    // RTX HDR converts after the pass (nrSdrIn: the planes are the capture's SDR range already); offline
    // HDR video (PQ or HLG codes) takes the same math
    const bool pq = (nr.liveNr && nr.hdr && !nr.nrSdrIn) || nr.nrSrcHdr != 0;
    int hlg = nr.nrSrcHdr == 2 ? 1 : 0;
    float* r0 = frame;
    const float* src = r0;
    int sps = ps;
    // the RGBA16F frame: zero-copy, the core's Color / Output textures (surface objects); staging, dNrIo tight
    uint16_t* io = nr.dNrIo;
    int iop = tw * 4;
    unsigned long long sIn = nr.nrZeroCopy ? nr.sfNrIn : 0, sOut = nr.nrZeroCopy ? nr.sfNrOut : 0;
    void* a[] = {(void*)&src, &sps, &pw, &tw, &th, &io, &iop, &sIn};
    void* b[] = {&io, &iop, &tw, &th, &r0, &ph, &pw, &sps, &sOut};
    void* aq[] = {(void*)&src, &sps, &pw, &tw, &th, &nr.nrSdrWhite, &io, &iop, &hlg, &sIn};
    void* bq[] = {&io, &iop, &tw, &th, &nr.nrSdrWhite, &r0, &ph, &pw, &sps, &hlg, &sOut};
    const unsigned gx = (unsigned)(pw + 15) / 16, gy = (unsigned)(ph + 15) / 16;
    if (nr.nrZeroCopy)
    {
        if (nr.nrMotion && !nativeNrMotion(nr, frame, pw, ps))
            return false;
        cudaExternalSemaphoreSignalParams sp{};
        sp.params.fence.value = ++nr.nrFenceV;
        if (cuLaunchKernel(pq ? nr.fNrInPq : nr.fNrIn, (tw + 15) / 16, (th + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st,
                           pq ? aq : a, nullptr) != CUDA_SUCCESS ||
            cudaSignalExternalSemaphoresAsync(&nr.semNr, &sp, 1, st) != cudaSuccess)
        {
            nr.die("DLSS 5 input handoff failed");
            return false;
        }
        const uint64_t ready = ++nr.nrFenceV;
        const int64_t t0 = nowQpc100();
        std::string err;
        if (!nr.nrHost->submitShared(nr.nrFirst || nr.nrResetEvery, sp.params.fence.value, ready, err))
        {
            LOG("[dlss5] %s; DLSS 5 disabled for the rest of this %s\n", err.c_str(), g_offline ? "render" : "session");
            nr.nrFailed = true;
            return true;
        }
        const double ms = (double)(nowQpc100() - t0) / 10000.0;
        nr.nrMs += ms;
        nr.nrN++;
        if (ms > nr.nrMaxMs)
            nr.nrMaxMs = ms;
        nr.nrFirst = false;
        cudaExternalSemaphoreWaitParams wp{};
        wp.params.fence.value = ready;
        if (cudaWaitExternalSemaphoresAsync(&nr.semNr, &wp, 1, st) != cudaSuccess ||
            cuLaunchKernel(pq ? nr.fNrOutPq : nr.fNrOut, gx, gy, 1, 16, 16, 1, 0, (CUstream)st, pq ? bq : b, nullptr) !=
                CUDA_SUCCESS)
        {
            nr.die("DLSS 5 output handoff failed");
            return false;
        }
        return true;
    }
    if (cuLaunchKernel(pq ? nr.fNrInPq : nr.fNrIn, (tw + 15) / 16, (th + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st,
                       pq ? aq : a, nullptr) != CUDA_SUCCESS ||
        cudaMemcpyAsync(nr.hNrIn, nr.dNrIo, bytes, cudaMemcpyDeviceToHost, st) != cudaSuccess ||
        cudaStreamSynchronize(st) != cudaSuccess)
    {
        nr.die("DLSS 5 input staging failed");
        return false;
    }
    std::vector<float> before; // --nr-delta: the pass's fp32 input (the stream is idle here)
    if (!nr.nrDeltaPath.empty())
    {
        before.resize((size_t)3 * tw * th);
        for (int c = 0; c < 3 && !before.empty(); c++)
            if (cudaMemcpy2D(before.data() + (size_t)c * tw * th, (size_t)tw * sizeof(float), src + (size_t)c * ps,
                             (size_t)pw * sizeof(float), (size_t)tw * sizeof(float), th,
                             cudaMemcpyDeviceToHost) != cudaSuccess)
            {
                LOG("offline: the DLSS 5 change map input download failed, no map\n");
                before.clear();
            }
    }
    const int64_t t0 = nowQpc100();
    std::string err;
    if (!nr.nrHost->renderFrame(nr.hNrIn, nr.hNrOut, nr.nrFirst || nr.nrResetEvery, err))
    {
        LOG("[dlss5] %s; DLSS 5 disabled for the rest of this render\n", err.c_str());
        nr.nrFailed = true;
        return true;
    }
    const double ms = (double)(nowQpc100() - t0) / 10000.0;
    nr.nrMs += ms;
    nr.nrN++;
    if (ms > nr.nrMaxMs)
        nr.nrMaxMs = ms;
    nr.nrFirst = false;
    if (!before.empty() && !pq)
        nativeOfflineNrDelta(nr, before, tw, th);
    if (cudaMemcpyAsync(nr.dNrIo, nr.hNrOut, bytes, cudaMemcpyHostToDevice, st) != cudaSuccess ||
        cuLaunchKernel(pq ? nr.fNrOutPq : nr.fNrOut, gx, gy, 1, 16, 16, 1, 0, (CUstream)st, pq ? bq : b, nullptr) !=
            CUDA_SUCCESS)
    {
        nr.die("DLSS 5 output staging failed");
        return false;
    }
    // HDR video: the change map compares the frame's own codes before and after the pass
    if (!before.empty() && pq)
    {
        std::vector<float> after((size_t)3 * tw * th);
        bool ok = cudaStreamSynchronize(st) == cudaSuccess;
        for (int c = 0; c < 3 && ok; c++)
            ok = cudaMemcpy2D(after.data() + (size_t)c * tw * th, (size_t)tw * sizeof(float), src + (size_t)c * ps,
                              (size_t)pw * sizeof(float), (size_t)tw * sizeof(float), th,
                              cudaMemcpyDeviceToHost) == cudaSuccess;
        if (ok)
            nativeOfflineNrDelta(nr, before, tw, th, &after);
        else
            LOG("offline: the DLSS 5 change map output download failed, no map\n");
    }
    return true;
}

// Live DLSS 5: the NR core brought up as offline does it (a private D3D12 device on
// the CUDA device's adapter) at the model size, with the zero-copy handoff and the CUDA Optical Flow
// motion; nativeGroup runs it once per captured frame on the model frame after Restore and the
// resize. A core that cannot start skips the pass for the session with a line and the session
// runs without it. false = an allocation failed.
static bool nativeLiveNrInit(NativeRife& nr)
{
    if (!g_liveNrCuda)
        return true;
    const int w = nr.w, h = nr.h;
    if ((uint64_t)w * h > 3840ull * 2160ull)
    {
        LOG("live DLSS 5 skipped for this session: the model frame %dx%d is above 3840x2160, the largest size the "
            "DLSS 5 host was probed at\n",
            w, h);
        return true;
    }
    if (w < kNrMinWorkW || h < kNrMinWorkH)
    {
        LOG("live DLSS 5 skipped for this session: the model frame %dx%d is below DLSS 5's minimum, %d px wide and %d "
            "high (a smaller frame resets the GPU)\n",
            w, h, kNrMinWorkW, kNrMinWorkH);
        return true;
    }
    wchar_t exePath[MAX_PATH]{};
    GetModuleFileNameW(nullptr, exePath, MAX_PATH);
    std::wstring nrDir(exePath);
    nrDir.resize(nrDir.find_last_of(L'\\'));
    nrDir += L"\\..\\dlssnr"; // shipped layout: engine\live\smv-live.exe beside engine\dlssnr
    wchar_t ov[MAX_PATH]{};
    if (GetEnvironmentVariableW(L"SMV_DLSSNR_DIR", ov, MAX_PATH) && ov[0])
        nrDir = ov;
    wchar_t full[MAX_PATH]{};
    if (GetFullPathNameW(nrDir.c_str(), MAX_PATH, full, nullptr))
        nrDir = full;
    nr::setModuleDir(nrDir.c_str());
    nr::Settings set;
    set.structure = (float)(g_nrStructure < 0.0 ? 0.0 : (g_nrStructure > 2.0 ? 2.0 : g_nrStructure));
    set.tone = (float)(g_nrTone < 0.0 ? 0.0 : (g_nrTone > 2.0 ? 2.0 : g_nrTone));
    set.skin = (float)(g_nrSkin < 0.0 ? -1.0 : (g_nrSkin > 2.0 ? 2.0 : g_nrSkin));
    set.automask = g_nrAutoMask ? 1 : 0;
    set.style = (g_nrStyle >= 0 && g_nrStyle <= 2) ? g_nrStyle : 0;
    set.passes = g_nrPasses < 1 ? 1 : (g_nrPasses > nr::kMaxPasses ? nr::kMaxPasses : g_nrPasses);
    // SMV_NR_MV=0 = no motion (a measurement lever); without motion every frame is a Reset, since kept
    // history with no motion slides; SMV_NR_AUTOMASK=0 = the mask off (read by the core)
    wchar_t mvv[8]{}, amv[8]{}, ru[8]{}, re[8]{};
    set.motion = !(GetEnvironmentVariableW(L"SMV_NR_MV", mvv, 8) && mvv[0] == L'0');
    const bool autoMaskEnvOff = GetEnvironmentVariableW(L"SMV_NR_AUTOMASK", amv, 8) && amv[0] == L'0';
    const bool autoMask = g_nrAutoMask && !autoMaskEnvOff;
    LUID luid{};
    const bool haveLuid = nativeCudaLuid(luid);
    std::string err;
    nr::Host* host = new nr::Host();
    const int nrc = host->startup((uint32_t)w, (uint32_t)h, set, err, true, haveLuid ? &luid : nullptr);
    if (nrc != 0)
    {
        LOG("live DLSS 5 skipped for this session: the NR core did not start: %s (exit %d)\n", err.c_str(), nrc);
        host->abandon(); // leaked on purpose, like every NGX object here
        return true;
    }
    if (!host->startShared(err) || !nativeNrImport(nr, *host, w, h, err))
    {
        nativeNrReleaseImports(nr);
        LOG("live DLSS 5 skipped for this session: the zero-copy handoff is unavailable (%s)\n", err.c_str());
        host->abandon();
        return true;
    }
    nr.nrHost = host;
    nr.nrW = w;
    nr.nrH = h;
    nr.nrZeroCopy = true;
    if (set.motion && !nativeNrMotionSetup(nr, err))
    {
        nativeNrMotionFree(nr);
        LOG("live DLSS 5: motion vectors unavailable (%s)\n", err.c_str());
    }
    NCHK(cudaMalloc((void**)&nr.dCapPrev, (size_t)nr.cw * nr.ch * (nr.hdr ? 8 : 4)), "alloc dlss5 reuse reference");
    nr.nrResetEvery = !nr.nrMotion || (GetEnvironmentVariableW(L"SMV_NR_RESET_EVERY", re, 8) && re[0] == L'1');
    nr.nrReuse = !(GetEnvironmentVariableW(L"SMV_NR_REUSE", ru, 8) && ru[0] == L'0');
    nr.nrSdrWhite = (float)(g_sdrWhite / 80.0);
    nr.nrFirst = true;
    nr.liveNr = true;
    char skinTxt[24] = ""; // the skin value when it is set apart from the structure
    if (set.skin >= 0.0f)
        sprintf_s(skinTxt, " skin %.2f", set.skin);
    LOG("live DLSS 5 native: on, %dx%d per captured frame on the model frame after Restore and the resize, "
        "zero-copy, structure %.2f tone %.2f%s style %d, passes %d, %s, %s, %s%s\n",
        w, h, set.structure, set.tone, skinTxt, set.style, host->passes(),
        nr.nrMotion ? "motion vectors (NVOFA grid 4), history kept"
                    : (set.motion ? "no motion vectors, every frame a Reset"
                                  : "no motion vectors (SMV_NR_MV=0), every frame a Reset"),
        autoMask ? "auto mask" : (autoMaskEnvOff ? "no auto mask (SMV_NR_AUTOMASK=0)" : "no auto mask"),
        nr.hdr ? "SDR range of the window (HDR highlights keep their light above white)" : "SDR window",
        nr.nrReuse ? ", identical frames reuse the last output" : "");
    if (host->passes() < set.passes)
        LOG("live DLSS 5 native: %d of %d passes (%s)\n", host->passes(), set.passes, host->passNote().c_str());
    if (nr.hdr)
        LOG("live DLSS 5 native: SDR reference white %.0f nits\n", g_sdrWhite);
    return true;
}
