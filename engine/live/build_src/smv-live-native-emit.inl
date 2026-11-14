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
    const bool vsrNow = vsrHere && nr.vsr && !nr.vsrFailed && (g_rtxb.created || nr.fsrUp || nr.a4kUp || nr.acnUp);
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
        if (vsrNow && (nr.a4kUp || nr.acnUp))
        {
            // Anime4K / ArtCNN: the planes (HDR video: their SDR view, as RTX VSR reads it) through its x2 net and the
            // Lanczos pair into nr.dPres; HDR video comes back to the codes in RTX VSR's block below
            const void* a4kSrc = src;
            int aHalf = srcHalf, aps = ps, ars = rs;
            if (enc)
            {
                if (!nativeHdrEnc(nr, src, srcHalf, ps, rs, sw, sh, nr.dVsrView, 0, sw * sh, sw, enc, 1.0f, 0.0f))
                {
                    nr.die(nr.acnUp ? "ArtCNN SDR view launch failed" : "Anime4K SDR view launch failed");
                    return false;
                }
                a4kSrc = nr.dVsrView;
                aHalf = 0;
                aps = sw * sh;
                ars = sw;
            }
            if (!(nr.acnUp ? nativeAcnRun(nr, a4kSrc, aHalf, aps, ars, nr.dPres, tw, th)
                           : nativeA4kRun(nr, a4kSrc, aHalf, aps, ars, nr.dPres, tw, th)))
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
            if (!nr.a4kUp && !nr.acnUp && // Anime4K / ArtCNN wrote nr.dPres itself
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
