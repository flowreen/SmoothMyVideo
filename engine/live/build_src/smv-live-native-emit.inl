// part of smv-live-native.inl (included there, in place); not a standalone translation unit
// After DLSS 5, once per real frame on the model frame dCur (pw x ph, planes dps apart, nr.w x nr.h
// used) before any model reads it: Adaptive Sharpen (post-processing at the working size), then RTX TrueHDR (SDR -> PQ
// BT.2020, after RTX VSR and every SDR pass), so the model interpolates the finished frames (frame generation last).
// thdr = TrueHDR runs this frame. A failed TrueHDR eval fails an offline render; live it turns the pass off for the
// session and this frame goes to PQ the faithful way. false = a launch failed (die was called).
static bool nativePreModelPost(NativeRife& nr, float* dCur, int dps, bool thdr)
{
    cudaStream_t st = nr.stream;
    int w = nr.w, h = nr.h, pw = nr.pw, ph = nr.ph;
    const bool sharpen = nr.sharpen > 0.0f;
    if (!sharpen && !thdr)
        return true;
    const size_t tight = (size_t)w * h;
    int fps = (int)tight;
    // HDR planes without RTX HDR: the sharpen reads their SDR view, so the SDR range is sharpened exactly as an SDR source's
    // (clamped at SDR white) and a frame of SDR content stays inside the SDR range; the light the view never held is
    // added back after it (k_hdrRestOut, the way Restore and DLSS 5 treat HDR planes)
    const bool hdrView = sharpen && !thdr && nr.encPost && nr.shHdrView;
    if (hdrView)
    {
        if (!nativeHdrEnc(nr, dCur, 0, dps, pw, w, h, nr.dShIn, 0, fps, w, nr.encPost, 1.0f, 0.0f))
        {
            nr.die("sharpen SDR view launch failed");
            return false;
        }
    }
    // the sharpen reads the HDR planes' tight SDR view, else the model frame itself with its pad strides
    float* shSrc = hdrView ? nr.dShIn : dCur;
    int shPs = hdrView ? fps : dps, shRs = hdrView ? w : pw;
    // the sharpen's tight result, padded back into dCur below (TrueHDR's colour step pads itself)
    float* out = nr.dShOut;
    if (thdr)
    {
        if (sharpen)
        {
            void* a[] = {&shSrc, &shPs, &shRs, &w, &h, &nr.sharpen, &nr.dThdrIn, &nr.dSrcG};
            if (cuLaunchKernel(nr.fSharpThdrIn, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a,
                               nullptr) != CUDA_SUCCESS)
            {
                nr.die("sharpThdrIn launch failed");
                return false;
            }
        }
        else
        {
            const void* src = dCur;
            int half = 0, rs = pw;
            void* a[] = {(void*)&src, &half, &dps, &rs, &w, &h, &nr.dThdrIn, &nr.dSrcG};
            if (cuLaunchKernel(nr.fThdrIn, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a, nullptr) !=
                CUDA_SUCCESS)
            {
                nr.die("thdrIn launch failed");
                return false;
            }
        }
        // the bridge reads dThdrIn with a synchronous cuMemcpy2D on the legacy default stream, which is
        // not ordered against this non-blocking stream: sync first
        if (cudaStreamSynchronize(st) != cudaSuccess)
        {
            nr.die("stream sync before the TrueHDR eval failed");
            return false;
        }
        const RtxRect rc{0, 0, (uint32_t)w, (uint32_t)h};
        const int64_t t0 = nowQpc100();
        const unsigned int rv = g_rtxb.evalThdr(nr.dThdrIn, nr.dThdrOut, rc, rc, &nr.thdr);
        // MEASURED, do not remove: the bridge's output copy is a device to device cuMemcpy2D on the
        // legacy default stream, asynchronous to the host, so the eval returning proves nothing about
        // dThdrOut; without this wait the colour step read the PREVIOUS frame's output
        if (cudaDeviceSynchronize() != cudaSuccess)
        {
            nr.die("TrueHDR eval sync failed");
            return false;
        }
        const double ms = (double)(nowQpc100() - t0) / 10000.0;
        nr.thdrMs += ms;
        nr.thdrN++;
        if (ms > nr.thdrMaxMs)
            nr.thdrMaxMs = ms;
        if (rv != 1u)
        {
            if (g_offline)
            {
                LOG("[rtx] TrueHDR eval failed (rc %u)\n", rv);
                nr.die("TrueHDR eval failed");
                return false;
            }
            LOG("native: live TrueHDR eval failed (%u), faithful PQ for the rest of the session\n", rv);
            nr.rtxFailed = true;
            if (sharpen)
            {
                void* a[] = {&shSrc, &shPs, &shRs, &w, &h, &nr.sharpen, &out};
                void* b[] = {&out, &fps, &w, &w, &h, &dCur, &pw, &ph, &dps};
                if (cuLaunchKernel(nr.fSharpPlanar, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a,
                                   nullptr) != CUDA_SUCCESS ||
                    cuLaunchKernel(nr.fPadPlanar, (pw + 15) / 16, (ph + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, b,
                                   nullptr) != CUDA_SUCCESS)
                {
                    nr.die("sharpPlanar launch failed");
                    return false;
                }
            }
            void* a[] = {&dCur, &pw, &ph, &dps, &nr.sdrScale};
            if (cuLaunchKernel(nr.fSdrPq, (pw + 15) / 16, (ph + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a, nullptr) !=
                CUDA_SUCCESS)
            {
                nr.die("sdrPq launch failed");
                return false;
            }
            return true;
        }
        // the colour step writes the padded model frame itself, its pad included
        void* a[] = {&nr.dThdrOut, &nr.dSrcG, &w, &h, &dCur, &pw, &ph, &dps, &nr.rtxMode, &nr.rtxVib, &nr.rtxSb};
        if (cuLaunchKernel(nr.fThdrColor, (pw + 15) / 16, (ph + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a, nullptr) !=
            CUDA_SUCCESS)
        {
            nr.die("thdrColor launch failed");
            return false;
        }
        return true;
    }
    else
    {
        void* a[] = {&shSrc, &shPs, &shRs, &w, &h, &nr.sharpen, &out};
        if (cuLaunchKernel(nr.fSharpPlanar, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a, nullptr) !=
            CUDA_SUCCESS)
        {
            nr.die("sharpPlanar launch failed");
            return false;
        }
        if (hdrView)
        {
            int half = 0;
            void* r[] = {&out, &fps, &w,   (void*)&dCur, &half, &dps,        &pw,
                         &w,   &h,   &out, &fps,         &w,    &nr.encPost, &nr.encWhite};
            if (cuLaunchKernel(nr.fHdrRestOut, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, r,
                               nullptr) != CUDA_SUCCESS)
            {
                nr.die("sharpen HDR codes launch failed");
                return false;
            }
        }
    }
    void* b[] = {&out, &fps, &w, &w, &h, &dCur, &pw, &ph, &dps};
    if (cuLaunchKernel(nr.fPadPlanar, (pw + 15) / 16, (ph + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, b, nullptr) !=
        CUDA_SUCCESS)
    {
        nr.die("padPlanar (after DLSS 5) launch failed");
        return false;
    }
    return true;
}

// the offline RTX HDR emit: the PQ frame (TrueHDR ran before the model) -> the x2rgb10le words in dO and the frame's
// statistics. The sync first makes the stream idle, so the previous frame's statistics copy is complete and reaches
// the accumulators before this frame's copy reuses the host block. false = a launch failed.
static bool nativeOfflinePqOut(NativeRife& nr, const void* src, int half, int ps, int rs, uint8_t* dO)
{
    cudaStream_t st = nr.stream;
    int tw = nr.dw, th = nr.dh;
    if (cudaStreamSynchronize(st) != cudaSuccess)
    {
        nr.die("stream sync before the HDR statistics failed");
        return false;
    }
    nativeOfflineThdrDrain(nr);
    uint32_t* dW = (uint32_t*)dO;
    uint32_t* dHist = (uint32_t*)nr.dThdrStats;
    uint32_t* dMisc = dHist + 1024;
    double* dSum = (double*)(nr.dThdrStats + 1024 * 4 + 16);
    void* a[] = {(void*)&src, &half, &ps, &rs, &tw, &th, &nr.rtxMode, &dW, &dHist, &dMisc, &dSum};
    if (cudaMemsetAsync(nr.dThdrStats, 0, kThdrStatsBytes, st) != cudaSuccess ||
        cuLaunchKernel(nr.fPqOut, (tw + 15) / 16, (th + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a, nullptr) !=
            CUDA_SUCCESS ||
        cudaMemcpyAsync(nr.hThdrStats, nr.dThdrStats, kThdrStatsBytes, cudaMemcpyDeviceToHost, st) != cudaSuccess)
    {
        nr.die("pqOut launch failed");
        return false;
    }
    nr.thdrStatsPending = true;
    return true;
}

// One offline resize stage on a planar frame of sw x sh (ps / rs strides) to tw x th: Restore
// first when withRestore (back to the source size when RTX VSR follows, else folded straight to
// the target), then the resize (RTX VSR when this stage owns it, else clamped Lanczos; offline
// only enlarges, a downscale is folded into the decode). The stages: without a working size the
// model output to the output size; with one the decoded frame to the working size before
// the model, and the final resize (no Restore) from the working size to the output after it.
// staged = nr.dPres holds the tw x th frame; src / ps / rs / srcHalf follow a Restore that hands on
// its fp32 frame; enc = the frame's HDR encoding for Restore (nativeRestoreRun). false = a launch failed.
// one RTX VSR eval of nr.dVsrIn into nr.dVsrOut for the work queued on st: through the bridge's event-ordered export
// when it has one (the thread keeps enqueuing while VSR runs), else (an older bridge, SMV_VSR_SYNC=1) the blocking
// export between a drain of st and a device-wide sync, because its copies ride the legacy default stream. The bridge
// takes and returns the R, G, B, A bytes both buffers hold
static bool nativeVsrEval(NativeRife& nr, const RtxRect& ri, const RtxRect& ro, cudaStream_t st, unsigned int& rv)
{
    if (nr.fsrUp)
        return nativeFsrupEval(nr, ri, ro, st, rv); // AMD FSR upscaling holds RTX VSR's resize, on B, G, R, A buffers
    static const bool sync = [] {
        wchar_t v[8]{};
        return GetEnvironmentVariableW(L"SMV_VSR_SYNC", v, 8) && v[0] != L'0';
    }();
    if (g_rtxb.evalVsrAsync && !sync)
    {
        rv = g_rtxb.evalVsrAsync(nr.dVsrIn, nr.dVsrOut, ri, ro, &nr.vsrSet, (void*)st);
        nr.vsrN++;
        nr.vsrAsync = true;
        return true;
    }
    if (cudaStreamSynchronize(st) != cudaSuccess)
    {
        nr.die("VSR input sync failed");
        return false;
    }
    const int64_t t0 = nowQpc100();
    rv = g_rtxb.evalVsr(nr.dVsrIn, nr.dVsrOut, ri, ro, &nr.vsrSet);
    if (cudaDeviceSynchronize() != cudaSuccess)
    {
        nr.die("VSR eval sync failed");
        return false;
    }
    const double ms = (nowQpc100() - t0) / 1e4;
    nr.vsrMs += ms;
    nr.vsrN++;
    if (ms > nr.vsrMaxMs)
        nr.vsrMaxMs = ms;
    return true;
}

// Restore and the resize of one frame into nr.dPres (staged) or in place. enc = the frame's encoding (0 SDR planes; 1 PQ,
// 2 HLG = HDR video or RTX HDR's PQ, src holds the codes). inRange = the caller's range test of src (1 inside the SDR
// range, 0 outside, -1 none).
static bool nativeOfflineStage(NativeRife& nr, const void*& src, int& ps, int& rs, int& srcHalf, bool& staged, int sw,
                               int sh, int tw, int th, bool withRestore, bool vsrHere, int enc = 0,
                               bool viewFit = false, int inRange = -1)
{
    cudaStream_t st = nr.stream;
    const bool resize = tw != sw || th != sh;
    const bool vsrNow = vsrHere && nr.vsr && !nr.vsrFailed && (g_rtxb.created || nr.fsrUp || nr.planarUp());
    staged = false;
    if (withRestore && nr.restore && nr.restNat && !nr.restFailed)
    {
        if (vsrNow)
        {
            if (!nativeRestoreRun(nr, src, ps, rs, nr.dRest, sw, sh, srcHalf, enc, viewFit, nr.a4kUp && resize))
                return false;
            if (!nr.restFailed)
            {
                src = nr.dRest;
                ps = sw * sh;
                rs = sw;
                srcHalf = 0;
            }
        }
        else
        {
            if (!nativeRestoreRun(nr, src, ps, rs, nr.dPres, tw, th, srcHalf, enc, viewFit))
                return false;
            staged = !nr.restFailed;
        }
    }
    if (!staged && resize)
    {
        bool haveVsr = false;
        const int hlg = enc == 2 ? 1 : 0;
        if (vsrNow && nr.planarUp())
        {
            // Anime4K / ArtCNN / Real-ESRGAN: the planes (HDR video: their SDR view, as RTX VSR reads it) through its
            // net and the Lanczos pair into nr.dPres; HDR video comes back to the codes in RTX VSR's block below
            const void* a4kSrc = src;
            int aHalf = srcHalf, aps = ps, ars = rs;
            if (enc)
            {
                if (!nativeHdrEnc(nr, src, srcHalf, ps, rs, sw, sh, nr.dVsrView, 0, sw * sh, sw, enc, 1.0f, 0.0f))
                {
                    nr.die((std::string(planarUpName(nr)) + " SDR view launch failed").c_str());
                    return false;
                }
                a4kSrc = nr.dVsrView;
                aHalf = 0;
                aps = sw * sh;
                ars = sw;
            }
            if (!nativePlanarUpRun(nr, a4kSrc, aHalf, aps, ars, nr.dPres, tw, th))
                return false;
            haveVsr = true;
        }
        else if (vsrNow)
        {
            // HDR video (enc: src holds the codes, restored or not): RTX VSR takes SDR, so it reads the frame's SDR
            // view (knee 1, head 0) and comes back to the codes below; AMD FSR upscaling takes the codes in linear
            // light (k_fsrHdrIn) and the view feeds only its vectors, as live's planes and view do
            // (HDR video through AMD FSR upscaling: the view and the linear frame in ONE pass over the codes, below)
            const bool onePass = enc && nr.fuHdr;
            const void* vsrSrc = src;
            int vHalf = srcHalf, vps = ps, vrs = rs;
            if (enc && !onePass)
            {
                if (!nativeHdrEnc(nr, src, srcHalf, ps, rs, sw, sh, nr.dVsrView, 0, sw * sh, sw, enc, 1.0f, 0.0f))
                {
                    nr.die("VSR SDR view launch failed");
                    return false;
                }
                vsrSrc = nr.dVsrView;
                vHalf = 0;
                vps = sw * sh;
                vrs = sw;
            }
            void* a[] = {(void*)&vsrSrc, &vHalf, &vps, &vrs, &sw, &sh, &nr.dVsrIn};
            if (!onePass && cuLaunchKernel(nr.fsrUp ? nr.fPackBgraRgb : nr.fPackBgra, (sw + 15) / 16, (sh + 15) / 16, 1,
                                           16, 16, 1, 0, (CUstream)st, a, nullptr) != CUDA_SUCCESS)
            {
                nr.die("packBgraRgb launch failed");
                return false;
            }
            if (onePass)
            {
                // the SDR view packed for the vectors (k_hdrEnc + k_packBgraRgb's bytes) and the codes in linear
                // light for the bridge (k_fsrHdrIn's halves) from one read of the frame (k_fsrHdrPack)
                const float scale = 125.0f / nr.encWhite;
                void* f[] = {(void*)&src,  &srcHalf,      &ps,         &rs,        &sw,       &sh,          (void*)&enc,
                             &nr.encWhite, (void*)&scale, (void*)&hlg, &nr.dVsrIn, &nr.dFuIn, &nr.fuInPitch};
                if (cuLaunchKernel(nr.fFsrHdrPack, (sw + 15) / 16, (sh + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, f,
                                   nullptr) != CUDA_SUCCESS)
                {
                    nr.die("FSR HDR pack launch failed");
                    return false;
                }
            }
            const RtxRect ri{0, 0, (uint32_t)sw, (uint32_t)sh};
            const RtxRect ro{0, 0, (uint32_t)tw, (uint32_t)th};
            unsigned int rv = 0;
            if (!nativeVsrEval(nr, ri, ro, st, rv))
                return false;
            if (rv == 1u)
                haveVsr = true;
            else
            {
                LOG("[rtx] VSR run failed (rc %u), using Lanczos for the rest of the render\n", rv);
                nr.vsrFailed = true;
            }
        }
        if (haveVsr && nr.fuHdr)
        {
            // AMD FSR upscaling on HDR planes: the bridge's linear-light result back to the codes (k_fsrHdrOut)
            const float inv = nr.encWhite / 125.0f;
            void* a[] = {&nr.dFuOut, &nr.fuOutPitch, (void*)&tw, (void*)&th, (void*)&inv, (void*)&hlg, &nr.dPres};
            if (cuLaunchKernel(nr.fFsrHdrOut, (tw + 15) / 16, (th + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a,
                               nullptr) != CUDA_SUCCESS)
            {
                nr.die("FSR HDR output launch failed");
                return false;
            }
        }
        else if (haveVsr)
        {
            void* a[] = {&nr.dVsrOut, (void*)&tw, (void*)&th, &nr.dPres};
            if (!nr.planarUp() && // Anime4K / ArtCNN / Real-ESRGAN wrote nr.dPres itself
                cuLaunchKernel(nr.fsrUp ? nr.fUnpackBgraRgb : nr.fUnpackRgba, (tw + 15) / 16, (th + 15) / 16, 1, 16, 16,
                               1, 0, (CUstream)st, a, nullptr) != CUDA_SUCCESS)
            {
                nr.die("unpackBgraRgb launch failed");
                return false;
            }
            if (enc)
            {
                // HDR video, back to the codes: a frame inside the SDR range takes the view's way back (the caller's
                // range test, else one here on fp32 codes: an fp16 frame widened into dVsrView, free once the view is
                // packed), any other gets the light the view never held from the frame's Lanczos fit (k_hdrRestOut)
                int in = inRange;
                if (in < 0 && nr.dRsRange)
                {
                    const float* f32 = (const float*)src;
                    int fps = ps, frs = rs;
                    if (srcHalf)
                    {
                        void* wa[] = {(void*)&src, &srcHalf, &ps, &rs, &sw, &sh, &nr.dVsrView, &sw, &sh};
                        if (cuLaunchKernel(nr.fFitPlanar, (sw + 15) / 16, (sh + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st,
                                           wa, nullptr) != CUDA_SUCCESS)
                        {
                            nr.die("VSR range widen failed");
                            return false;
                        }
                        f32 = nr.dVsrView;
                        fps = sw * sh;
                        frs = sw;
                    }
                    if ((in = nativeRsInRange(nr, f32, fps, frs, sw, sh, enc)) < 0)
                    {
                        nr.die("VSR range test failed");
                        return false;
                    }
                }
                const int tps = tw * th;
                if (in == 1 ? !nativeHdrFromSdr(nr, nr.dPres, 0, tps, tw, tw, th, nr.dPres, tps, tw, enc)
                            : !nativeFitPlanar(nr, src, srcHalf, ps, rs, sw, sh, nr.dVsrRem, tw, th, st) ||
                                  !nativeHdrRestOut(nr, nr.dPres, tw, th, nr.dVsrRem, 0, tps, tw, tw, th, enc))
                {
                    nr.die("VSR HDR codes launch failed");
                    return false;
                }
            }
        }
        else if (viewFit && enc)
        {
            // an HDR frame inside the SDR range: resized as its SDR view, as the SDR route resizes
            if (!nativeViewFit(nr, src, srcHalf, ps, rs, sw, sh, nr.dPres, tw, th, enc))
            {
                nr.die("SDR view fit launch failed");
                return false;
            }
        }
        else
        {
            if (!nativeFitPlanar(nr, src, srcHalf, ps, rs, sw, sh, nr.dPres, tw, th, st))
            {
                nr.die("fitPlanar launch failed");
                return false;
            }
            int n = 3 * tw * th;
            void* a2[] = {&nr.dPres, &n};
            if (cuLaunchKernel(nr.fClamp01, (n + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, a2, nullptr) !=
                CUDA_SUCCESS)
            {
                nr.die("clamp01 launch failed");
                return false;
            }
        }
        staged = true;
    }
    return true;
}

// Before DLSS 5 and the model: the decoded frame in nr.dSrcPl (srcW x srcH)
// through Restore and the Lanczos resize to the working size, which is the model size (the picked upscaler runs
// after the model), then into the model frame dCur (pw x ph, planes dps apart) with the packers' replicate pad
static bool nativeOfflinePreModel(NativeRife& nr, float* dCur, int dps)
{
    const void* src = nr.dSrcPl;
    int ps = srcW(nr) * srcH(nr), rs = srcW(nr), half = 0;
    bool staged = false;
    // HDR planes: the resize and Restore's reference take the decoded frame's SDR view when it stays inside the range
    int in = 0;
    if (nr.dRsView && (in = nativeRsInRange(nr, nr.dSrcPl, ps, rs, srcW(nr), srcH(nr), nr.encPre)) < 0)
    {
        nr.die("resize range test failed");
        return false;
    }
    if (!nativeOfflineStage(nr, src, ps, rs, half, staged, srcW(nr), srcH(nr), nr.w, nr.h, true, false, nr.encPre,
                            in == 1, nr.dRsView ? in : -1))
        return false;
    // unstaged = no resize and Restore dropped: the source already has the model size
    const float* from = staged ? nr.dPres : (const float*)src;
    int fps = staged ? nr.w * nr.h : ps, frs = staged ? nr.w : rs;
    void* a[] = {&from, &fps, &frs, &nr.w, &nr.h, &dCur, &nr.pw, &nr.ph, &dps};
    if (cuLaunchKernel(nr.fPadPlanar, (nr.pw + 15) / 16, (nr.ph + 15) / 16, 1, 16, 16, 1, 0, (CUstream)nr.stream, a,
                       nullptr) != CUDA_SUCCESS)
    {
        nr.die("padPlanar launch failed");
        return false;
    }
    return true;
}

// The offline pass chain on one planar model-size frame (ps / rs strides, nr.w x nr.h), then
// the quantisation, into dO as tight rgb48le (out16) or rgb24 at the output size nr.dw x nr.dh:
// nativeOfflineStage (Restore and the resize; with a working size Restore ran before the model and
// only the final resize from the working size is left). DLSS 5, Sharpen and RTX TrueHDR already ran on
// the decoded frame before the interpolation (nativeNrFrame, nativePreModelPost): with RTX HDR the
// frame is PQ, the picked upscaler reads it as it reads HDR video (RTX VSR the SDR view, the light
// above white back from the Lanczos fit) and nativeOfflinePqOut writes the x2rgb10le words and the
// frame's statistics. A pass that fails is dropped for the rest of the render
// with a line. false = a launch failed.
static bool nativeOfflineEmit(NativeRife& nr, const void* src, int ps, int rs, uint8_t* dO, bool out16, int srcHalf = 0)
{
    cudaStream_t st = nr.stream;
    const int tw = nr.dw, th = nr.dh;
    bool staged = false; // nr.dPres holds the output-size frame
    if (!nr.nvPre)
    {
        if (!nativeOfflineStage(nr, src, ps, rs, srcHalf, staged, srcW(nr), srcH(nr), tw, th, true, true, nr.encPost))
            return false;
    }
    else if ((nr.w != tw || nr.h != th) &&
             !nativeOfflineStage(nr, src, ps, rs, srcHalf, staged, nr.w, nr.h, tw, th, false, nr.vsrPost, nr.encPost))
        return false;
    const void* fin = staged ? (const void*)nr.dPres : src;
    int fps = staged ? tw * th : ps, frs = staged ? tw : rs, fhalf = staged ? 0 : srcHalf;
    if (nr.rtxHdr)
        return nativeOfflinePqOut(nr, fin, fhalf, fps, frs, dO);
    void* a[] = {(void*)&fin, &fhalf, &fps, &frs, (void*)&tw, (void*)&th, &dO};
    if (cuLaunchKernel(out16 ? nr.fPackOutRaw16 : nr.fPackOutRaw8, (tw + 15) / 16, (th + 15) / 16, 1, 16, 16, 1, 0,
                       (CUstream)st, a, nullptr) != CUDA_SUCCESS)
    {
        nr.die("packOutRaw (passes) launch failed");
        return false;
    }
    return true;
}

// THE UPSCALER AFTER LIVE DLSS 4.5 (the present's stage): DLSS 4.5's stages make their frames at the working size and
// the present thread shows them (Host::presentRing); with an upscaler picked every shown frame goes through it to the
// presented size here, the one enlarge after the model that every other model's frames get in the slot store. The
// present copies the frame (BGRA8, w x h) into hIn's buffer and signals hFenceIn; nativePresentUpRun waits for it on
// puSt, turns the bytes into R, G, B, A order, runs the upscaler (kind, nativeUpSwitch's numbering: 1 RTX VSR into
// dVsrOut; 3 Anime4K, 4 ArtCNN, 6 Real-ESRGAN, 7 AnimeJaNai HD = the planar net from the frame's planes into dPuPres,
// on puSt) to uw x uh (Upscale to's size, else the presented tw x th; Lanczos to tw x th after it, Live's second
// resize), writes BGRA8 into hOut's buffer and signals hFenceOut, which the present waits on before its copy: the GPU
// orders the three, no thread waits. HDR (hdr: the frames are R10G10B10A2 PQ words): the upscaler reads the frame's SDR
// view and its result comes back to the codes as on Live's own route (storeSlot), the way picked by the last four
// captures' range bits (hdrInBits) instead of a readback; R10G10B10A2 out. The caller's thread binds the device; false =
// it did not start (the present keeps the Lanczos pair)
// (the present thread's rsTabFor calls go to the stage's own tables while it runs: t_puTabs)
struct PuTabsScope
{
    PuTabsScope()
    {
        t_puTabs = true;
    }
    ~PuTabsScope()
    {
        t_puTabs = false;
    }
};
static const char* puName(int kind)
{
    return kind == 1   ? "RTX VSR"
           : kind == 3 ? "Anime4K"
           : kind == 4 ? "ArtCNN"
           : kind == 6 ? "Real-ESRGAN"
           : kind == 7 ? "AnimeJaNai HD"
                       : "an unknown upscaler";
}
// the stage's geometry part: its stream, the two imported frame buffers and fences, the buffers every kind uses at this
// size (the input, HDR's, Upscale to's); then the upscaler itself (nativePresentUpKind)
static bool nativePresentUpKind(NativeRife& nr, int kind);
static bool nativePresentUpInit(NativeRife& nr, HANDLE hIn, uint64_t inBytes, int inPitch, HANDLE hOut,
                                uint64_t outBytes, int outPitch, HANDLE hFenceIn, HANDLE hFenceOut, int w, int h,
                                int tw, int th, int uw, int uh, int kind, bool hdr)
{
    const PuTabsScope tabsHere;
    // uw x uh = the upscaler's own size (Upscale to), 0 = the presented tw x th
    if (uw <= 0 || uh <= 0)
    {
        uw = tw;
        uh = th;
    }
    const bool fit2 = uw != tw || uh != th;
    if (!nativeBindDevice(nr))
        return false;
    NCHK(cudaStreamCreateWithFlags(&nr.puSt, cudaStreamNonBlocking), "the upscaler after DLSS 4.5: stream");
    const HANDLE hb[2] = {hIn, hOut};
    const uint64_t nb[2] = {inBytes, outBytes};
    cudaExternalMemory_t* em[2] = {&nr.puEmIn, &nr.puEmOut};
    uint8_t** dp[2] = {&nr.dPuIn, &nr.dPuOut};
    for (int i = 0; i < 2; i++)
    {
        cudaExternalMemoryHandleDesc od{};
        od.type = cudaExternalMemoryHandleTypeD3D12Resource;
        od.handle.win32.handle = hb[i];
        od.size = nb[i];
        od.flags = cudaExternalMemoryDedicated;
        NCHK(cudaImportExternalMemory(em[i], &od), "RTX VSR after DLSS 4.5: import a frame buffer");
        cudaExternalMemoryBufferDesc bd{};
        bd.size = nb[i];
        NCHK(cudaExternalMemoryGetMappedBuffer((void**)dp[i], *em[i], &bd),
             "RTX VSR after DLSS 4.5: map a frame buffer");
    }
    const HANDLE hf[2] = {hFenceIn, hFenceOut};
    cudaExternalSemaphore_t* sm[2] = {&nr.puSemIn, &nr.puSemOut};
    for (int i = 0; i < 2; i++)
    {
        cudaExternalSemaphoreHandleDesc sd{};
        sd.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
        sd.handle.win32.handle = hf[i];
        NCHK(cudaImportExternalSemaphore(sm[i], &sd), "RTX VSR after DLSS 4.5: import a fence");
    }
    // the frame as tight R, G, B, A bytes: RTX VSR's input, a planar net's way to its planes
    if (!nr.dVsrIn)
        NCHK(cudaMalloc((void**)&nr.dVsrIn, (size_t)w * h * 4), "the upscaler after DLSS 4.5: input");
    // the Lanczos passes' horizontal buffer: the codes' fit (HDR, w x h -> uw x uh) and Upscale to's (uw x uh -> tw x th)
    const size_t tmpCodes = hdr ? (size_t)3 * h * uw : 0, tmpUpTo = fit2 ? (size_t)3 * uh * tw : 0;
    nr.puTmpN = tmpCodes > tmpUpTo ? tmpCodes : tmpUpTo;
    if (nr.puTmpN)
        NCHK(cudaMalloc((void**)&nr.dPuTmp, nr.puTmpN * sizeof(float)), "the upscaler after DLSS 4.5: the fit's pass");
    if (hdr)
    {
        NCHK(cudaMalloc((void**)&nr.dPuView, (size_t)3 * w * h * sizeof(float)),
             "the upscaler after DLSS 4.5: the SDR view");
        NCHK(cudaMalloc((void**)&nr.dPuRem, (size_t)3 * uw * uh * sizeof(float)),
             "the upscaler after DLSS 4.5: the codes' fit");
    }
    if (fit2)
        NCHK(cudaMalloc((void**)&nr.dPuFin, (size_t)3 * tw * th * sizeof(float)),
             "the upscaler after DLSS 4.5: Upscale to's resize");
    nr.puHdr = hdr;
    nr.puW = w;
    nr.puH = h;
    nr.puUw = uw;
    nr.puUh = uh;
    nr.puTw = tw;
    nr.puTh = th;
    nr.puInPitch = inPitch;
    nr.puOutPitch = outPitch;
    return nativePresentUpKind(nr, kind);
}

// the stage's upscaler at its geometry (nativePresentUpInit's; a live switch: nativePresentUpSwitch): RTX VSR's NGX
// feature (one create a process) and its warm-up eval, or a planar net's setup (its own warm-up); the planar buffers a
// planar net, HDR or Upscale to reads; false = it did not start
static bool nativePresentUpKind(NativeRife& nr, int kind)
{
    const PuTabsScope tabsHere;
    const char* name = puName(kind);
    const int w = nr.puW, h = nr.puH, uw = nr.puUw, uh = nr.puUh, tw = nr.puTw, th = nr.puTh;
    const bool hdr = nr.puHdr, fit2 = uw != tw || uh != th;
    if (kind != 1 && kind != 3 && kind != 4 && kind != 6 && kind != 7)
        return false;
    if (kind == 1)
    {
        if (!rtxBridgeLoad())
            return false;
        if (g_rtxb.created && !g_rtxb.vsrMade)
        {
            LOG("native: RTX VSR after DLSS 4.5: this process made its RTX Video feature without RTX VSR\n");
            return false;
        }
        if (!g_rtxb.created)
        {
            // one create per process (NGX is single-instance), the primary context, as nativeRtxInit makes it
            if (g_rtxb.create(nullptr, nullptr, 0, 0u, 1u) != 1u)
            {
                LOG("native: RTX VSR after DLSS 4.5: rtx_video_api_cuda_create failed\n");
                return false;
            }
            g_rtxb.created = true;
            g_rtxb.vsrMade = true;
            g_rtxUsed = true; // resident host: NGX is single-instance per process, end it after this session
        }
        if (!nr.dVsrOut)
            NCHK(cudaMalloc((void**)&nr.dVsrOut, (size_t)uw * uh * 4), "RTX VSR after DLSS 4.5: output");
        // the warm-up eval on an opaque black frame, so the first shown frame pays nothing
        NCHK(cudaMemsetAsync(nr.dVsrIn, 0, (size_t)w * h * 4, nr.puSt), "RTX VSR after DLSS 4.5: warm-up clear");
        NCHK(cudaStreamSynchronize(nr.puSt), "RTX VSR after DLSS 4.5: warm-up clear sync");
        const RtxRect ri{0, 0, (uint32_t)w, (uint32_t)h};
        const RtxRect ro{0, 0, (uint32_t)uw, (uint32_t)uh};
        if (g_rtxb.evalVsr(nr.dVsrIn, nr.dVsrOut, ri, ro, &nr.vsrSet) != 1u)
        {
            LOG("native: RTX VSR after DLSS 4.5: the warm-up eval failed\n");
            return false;
        }
        NCHK(cudaDeviceSynchronize(), "RTX VSR after DLSS 4.5: warm-up eval sync");
    }
    else
    {
        // the planar net as the session's own upscaler (its setup reads these kinds; the echo route stays at the
        // working size, where no resize asks for an upscaler), its own warm-up inside the setup
        nr.a4kUp = kind == 3;
        nr.acnUp = kind == 4;
        nr.rgUp = kind == 6 || kind == 7;
        nr.rgAj = kind == 7;
        nr.vsrFailed = false;
        if (!nativePlanarUpSetup(nr, w, h, uw, uh, 0) || nr.vsrFailed)
        {
            LOG("native: %s after DLSS 4.5: its setup failed\n", name);
            nr.a4kUp = nr.acnUp = nr.rgUp = nr.rgAj = false;
            return false;
        }
    }
    // the planar buffers: a net's planes in and result out; on HDR or with Upscale to every upscaler's (the codes, their
    // view, the way back, the second resize); kept across a live switch
    if ((kind != 1 || hdr || fit2) && !nr.dPuPl)
    {
        NCHK(cudaMalloc((void**)&nr.dPuPl, (size_t)3 * w * h * sizeof(float)), "the upscaler after DLSS 4.5: planes");
        NCHK(cudaMalloc((void**)&nr.dPuPres, (size_t)3 * uw * uh * sizeof(float)),
             "the upscaler after DLSS 4.5: its result");
    }
    nr.puKind = kind;
    nr.puFailed = false;
    nr.puOn = true;
    char then[64] = "";
    if (fit2)
        sprintf_s(then, ", then Lanczos to %dx%d (Upscale to)", tw, th);
    char vsrQ[24] = "";
    if (kind == 1)
        sprintf_s(vsrQ, " (quality %u)", nr.vsrSet.QualityLevel);
    LOG("native: live upscale: %s %dx%d -> %dx%d after DLSS 4.5%s, on every shown frame%s%s\n", name, w, h, uw, uh,
        then, vsrQ, hdr ? ", on the HDR frames' SDR view" : "");
    return true;
}

// a live `set up N` on a session whose DLSS 4.5 frames the stage enlarges (the loop thread, between two shown frames,
// after nativePresentUpInit started the stage): the stage's queued frames finish, the running planar net's own state
// goes (nativeUpRelease: its buffers, Anime4K's resize ones only; RTX VSR keeps NGX for a switch back), then kind's
// setup; kind 0 = only the release. false = the new kind did not start (the stage stays off)
static void nativeUpRelease(NativeRife& nr);
static bool nativePresentUpSwitch(NativeRife& nr, int kind)
{
    const PuTabsScope tabsHere;
    if (nr.puSt)
        cudaStreamSynchronize(nr.puSt);
    nr.puOn = false;
    if (nr.puKind > 1)
        nativeUpRelease(nr);
    nr.puKind = 0;
    return kind == 0 || nativePresentUpKind(nr, kind);
}

// one shown frame on puSt, its bytes already waited for in dPuIn (nativePresentUpRun): the frame -> the upscaler's input
// (SDR: k_rgbaBgra into tight RGBA8 in dVsrIn; HDR: the words tight in dVsrIn -> the code planes, k_unpackR10 -> their
// SDR view in dPuView, knee 1, head 0) -> the upscaler to puUw x puUh (RTX VSR: dVsrIn -> dVsrOut, k_packBgra of the
// view first on HDR, k_unpackRgba into dPuPres when planes follow; a planar net from the planes into dPuPres) -> HDR:
// back to the codes (Live's storeSlot rule) -> Upscale to: Lanczos into dPuFin at puTw x puTh -> the store into dPuOut
// (BGRA8, R10G10B10A2 on HDR). false = a call failed (an upscaler's own failure says its line here, puFailed set)
static bool nativePresentUpFrame(NativeRife& nr)
{
    cudaStream_t st = nr.puSt;
    int w = nr.puW, h = nr.puH, uw = nr.puUw, uh = nr.puUh, tw = nr.puTw, th = nr.puTh;
    int ps = w * h, ups = uw * uh, tps = tw * th, zero = 0, mode = 1, tight = w * 4;
    const bool hdr = nr.puHdr, fit2 = uw != tw || uh != th;
    if (hdr)
    {
        void* u[] = {&nr.dVsrIn, &w, &h, &zero, &nr.dPuPl};
        if (cudaMemcpy2DAsync(nr.dVsrIn, (size_t)w * 4, nr.dPuIn, (size_t)nr.puInPitch, (size_t)w * 4, h,
                              cudaMemcpyDeviceToDevice, st) != cudaSuccess ||
            cuLaunchKernel(nr.fUnpackR10, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, u, nullptr) !=
                CUDA_SUCCESS ||
            !nativeHdrEnc(nr, nr.dPuPl, 0, ps, w, w, h, nr.dPuView, 0, ps, w, mode, 1.0f, 0.0f, st))
            return false;
    }
    else
    {
        void* a[] = {&nr.dPuIn, &nr.puInPitch, &w, &h, &nr.dVsrIn, &tight};
        if (cuLaunchKernel(nr.fRgbaBgra, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a, nullptr) !=
            CUDA_SUCCESS)
            return false;
    }
    if (nr.puKind == 1)
    {
        void* a[] = {&nr.dPuView, &zero, &ps, &w, &w, &h, &nr.dVsrIn};
        if (hdr && cuLaunchKernel(nr.fPackBgra, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a,
                                  nullptr) != CUDA_SUCCESS)
            return false;
        const RtxRect ri{0, 0, (uint32_t)w, (uint32_t)h};
        const RtxRect ro{0, 0, (uint32_t)uw, (uint32_t)uh};
        unsigned int rv = 0;
        if (g_rtxb.evalVsrAsync)
            rv = g_rtxb.evalVsrAsync(nr.dVsrIn, nr.dVsrOut, ri, ro, &nr.vsrSet, (void*)st);
        else
        {
            // an older bridge: its blocking export between a drain of puSt and a device-wide sync (nativeVsrEval's rule)
            if (cudaStreamSynchronize(st) != cudaSuccess)
                return false;
            rv = g_rtxb.evalVsr(nr.dVsrIn, nr.dVsrOut, ri, ro, &nr.vsrSet);
            if (cudaDeviceSynchronize() != cudaSuccess)
                return false;
        }
        if (rv != 1u)
        {
            LOG("native: RTX VSR after DLSS 4.5 failed (rc %u), Lanczos for the rest of the session\n", rv);
            nr.puFailed = true;
            return false;
        }
        if (!hdr && !fit2)
        {
            // the bridge's bytes straight into the store (no planes: SDR at the presented size)
            int outTight = uw * 4;
            void* b[] = {&nr.dVsrOut, &outTight, &uw, &uh, &nr.dPuOut, &nr.puOutPitch};
            return cuLaunchKernel(nr.fRgbaBgra, (uw + 15) / 16, (uh + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, b,
                                  nullptr) == CUDA_SUCCESS;
        }
        void* b[] = {&nr.dVsrOut, &uw, &uh, &nr.dPuPres};
        if (cuLaunchKernel(nr.fUnpackRgba, (uw + 15) / 16, (uh + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, b, nullptr) !=
            CUDA_SUCCESS)
            return false;
    }
    else
    {
        // a planar net: SDR reads the frame's planes, HDR their SDR view
        void* u[] = {&nr.dVsrIn, &w, &h, &nr.dPuPl};
        if (!hdr && cuLaunchKernel(nr.fUnpackRgba, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, u,
                                   nullptr) != CUDA_SUCCESS)
            return false;
        if (!nativePlanarUpRun(nr, hdr ? nr.dPuView : nr.dPuPl, 0, ps, w, nr.dPuPres, uw, uh, st))
        {
            LOG("native: %s after DLSS 4.5 failed, Lanczos for the rest of the session\n", puName(nr.puKind));
            nr.puFailed = true;
            return false;
        }
    }
    // back to the codes (Live's storeSlot rule): the view's way when the last four captures stayed inside the SDR
    // range (a shown frame comes from the newest few), else the light the view never held from the codes' Lanczos fit
    if (hdr)
    {
        const unsigned bits = std::atomic_ref<unsigned>(nr.hdrInBits).load(std::memory_order_acquire);
        if ((bits & 15u) == 15u
                ? !nativeHdrFromSdr(nr, nr.dPuPres, 0, ups, uw, uw, uh, nr.dPuPres, ups, uw, mode, st)
                : !nativeFitPlanar(nr, nr.dPuPl, 0, ps, w, w, h, nr.dPuRem, uw, uh, st, nr.dPuTmp, nr.puTmpN) ||
                      !nativeHdrRestOut(nr, nr.dPuPres, uw, uh, nr.dPuRem, 0, ups, uw, uw, uh, mode, false, st))
            return false;
    }
    // Upscale to: the upscaler's size -> the presented one (Live's second resize), the store clamps
    float* out = nr.dPuPres;
    int ops = ups, ors = uw;
    if (fit2)
    {
        if (!nativeFitPlanar(nr, nr.dPuPres, 0, ups, uw, uw, uh, nr.dPuFin, tw, th, st, nr.dPuTmp, nr.puTmpN))
            return false;
        out = nr.dPuFin;
        ops = tps;
        ors = tw;
    }
    void* p[] = {&out, &zero, &ops, &ors, &tw, &th, &nr.dPuOut, &nr.puOutPitch, &zero, &zero, &tw, &th};
    return cuLaunchKernel(hdr ? nr.fPackOutHdr : nr.fPackOut, (tw + 15) / 16, (th + 15) / 16, 1, 16, 16, 1, 0,
                          (CUstream)st, p, nullptr) == CUDA_SUCCESS;
}

// one shown frame through the present's stage (nativePresentUpInit): queued on puSt behind hFenceIn's value v, ends by
// setting hFenceOut to v. Runs on the present thread (its first call binds the device there). false = nothing queued
// past the wait, the caller shows the frame its own way
static bool nativePresentUpRun(NativeRife& nr, uint64_t v)
{
    if (!nr.puOn || nr.puFailed)
        return false;
    const PuTabsScope tabsHere;
    static thread_local bool bound = false;
    if (!bound)
    {
        if (!nativeBindDevice(nr))
            return false;
        bound = true;
    }
    cudaExternalSemaphoreWaitParams wp{};
    wp.params.fence.value = v;
    if (cudaWaitExternalSemaphoresAsync(&nr.puSemIn, &wp, 1, nr.puSt) != cudaSuccess)
        return false;
    if (!nativePresentUpFrame(nr))
    {
        if (!nr.puFailed)
            LOG("native: %s after DLSS 4.5: a step failed, Lanczos for the rest of the session\n", puName(nr.puKind));
        nr.puFailed = true;
        return false;
    }
    cudaExternalSemaphoreSignalParams sp{};
    sp.params.fence.value = v;
    return cudaSignalExternalSemaphoresAsync(&nr.puSemOut, &sp, 1, nr.puSt) == cudaSuccess;
}
