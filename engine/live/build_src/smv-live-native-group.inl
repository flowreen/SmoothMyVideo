// part of smv-live-native.inl (included there, in place); not a standalone translation unit
// One group: the capture wait and read, the capture-release announcement, the pair encode, the tweens, one
// pack-out per slot with its own event, then the bare end marker.
static bool nativeGroup(NativeRife& nr, const std::vector<uint8_t>& msg)
{
    if (msg.size() < 4)
        return false;
    uint32_t nfr = 0;
    memcpy(&nfr, msg.data(), 4);
    // the exe's route bit (sendPairStream): this Smooth Motion / AMD FSR 4 group shows the previous pair, pipelined
    const bool pipeAsk = (nfr & 0x80000000u) != 0;
    nfr &= 0x7fffffffu;
    const float* fr = (const float*)(msg.data() + 4);
    if (msg.size() < 4 + 4ull * nfr)
        return false;
    if (nfr > nr.slots)
        nfr = nr.slots;
    const bool pipe =
        pipeAsk && nr.fruc && nr.frMp && nr.havePrev && (nr.fsrfg ? fsrfgPipeOk(nr) : g_fruc.stepI != nullptr);
    // no-engine mode: an empty group stays empty (the pair advances,
    // nothing presented), any other group is exactly ONE real frame through the effects
    // chain, whatever fractions the pair clock asked for
    static const float kReal = 1.0f;
    if (nr.noEngine && nfr)
    {
        nfr = 1;
        fr = &kReal;
    }

    ++nr.seq;
    const uint32_t set = nr.seq % 2;
    const size_t plane = (size_t)nr.ph * nr.pw;
    cudaStream_t st = nr.stream;

    // the previous group's own GPU time (the group ended in drainReady(true), so its events are done)
    if (nr.grArmed && cudaEventQuery(nr.grEv[1]) == cudaSuccess)
    {
        float pre = 0.0f, slots = 0.0f;
        if (cudaEventElapsedTime(&pre, nr.pfEv[0], nr.grEv[0]) == cudaSuccess &&
            cudaEventElapsedTime(&slots, nr.grEv[0], nr.grEv[1]) == cudaSuccess)
        {
            auto ema = [](double& v, double x) { v = v > 0.0 ? v * 0.8 + x * 0.2 : x; };
            if (nr.grTween)
            {
                // a GMFSS catch-up (a skipped frame's feat_ext) is no part of a tweening group's steady price
                float cu = 0.0f;
                if (nr.gmCatchUp && cudaEventElapsedTime(&cu, nr.gmCatchEv[0], nr.gmCatchEv[1]) == cudaSuccess)
                    pre = (std::max)(0.0f, pre - cu);
                ema(nr.gIntMs, pre);
                ema(nr.gTweenMs, slots / nr.grTween);
                nr.gIntUs.store((uint32_t)(nr.gIntMs * 1000.0));
                nr.gTweenUs.store((uint32_t)(nr.gTweenMs * 1000.0));
                // the midpoint tree (Smooth Motion, AMD FSR 4): a call = the compute thread's calls' mean; a slot = the
                // slots' time less the calls' chain in them (a direct group: its calls in the slots; a pipelined one:
                // its levels and the worker's join); a group that made instances is skipped (grPriceSkip)
                if (nr.fruc && nr.frMp && nr.grPriceSkip > 0)
                    nr.grPriceSkip--;
                else if (nr.fruc && nr.frMp && nr.fsrfg && nr.grSlots)
                {
                    // the level-1 chain's parts, on either route (the pipelined route's price, see fgOfEv): its
                    // Optical Flow, its call's FSR span and the session stream's pre outside them. A pipelined group's
                    // level-1 call may still run (the next pack does not wait for it): the spans come from the newest
                    // set whose call is done (the last group's, else the one before), read once, never waited for
                    float l1 = 0.0f, of = 0.0f;
                    int ls = -1;
                    for (int k : {(int)((nr.seq - 1) % 2), (int)(nr.seq % 2)})
                        if (ls < 0 && nr.grL1Armed[k] && cudaEventQuery(nr.fgL1Ev[k][1]) == cudaSuccess)
                            ls = k;
                    const bool l1Ok = ls >= 0 &&
                                      cudaEventElapsedTime(&l1, nr.fgL1Ev[ls][0], nr.fgL1Ev[ls][1]) == cudaSuccess &&
                                      l1 > 0.0f;
                    if (l1Ok && nr.grOfArmed[ls] &&
                        cudaEventElapsedTime(&of, nr.fgOfEv[ls][0], nr.fgOfEv[ls][1]) == cudaSuccess && of > 0.0f)
                    {
                        ema(nr.gFgOfMs, of);
                        ema(nr.gFgL1Ms, l1);
                        ema(nr.gFgPreMs, nr.grPiped ? (double)pre : (std::max)(0.0, (double)pre - of - l1));
                        nr.gFgOfUs.store((uint32_t)(nr.gFgOfMs * 1000.0));
                        nr.gFgL1Us.store((uint32_t)(nr.gFgL1Ms * 1000.0));
                        nr.gFgPreUs.store((uint32_t)(nr.gFgPreMs * 1000.0));
                    }
                    if (ls >= 0)
                        nr.grL1Armed[ls] = nr.grOfArmed[ls] = false;
                    const uint32_t in = nr.grFrCalls - nr.grFrPreCalls;
                    if (nr.grPiped)
                    {
                        // pipelined: the slots = the wait for the shown pair's level-1 node (the level-1 chain's
                        // time), the pair's deeper calls + the stores, so a call = the slots from grLagEv less the
                        // stores, per call; an x2 group (no deeper call) prices the call by its level-1 span, as a
                        // direct x2 group does. The direct price takes it too (its base and store stay the direct x2
                        // groups'): the route's exit reads the call as the GPU runs it now
                        float lag = 0.0f;
                        const double sl =
                            nr.grLagArmed && cudaEventElapsedTime(&lag, nr.grLagEv, nr.grEv[1]) == cudaSuccess ? lag
                                                                                                               : slots;
                        if (in && nr.gFrStoreMs > 0.0)
                        {
                            ema(nr.gFrCallMs, (std::max)(0.0, sl - nr.grSlots * nr.gFrStoreMs) / in);
                            nr.gFrCallUs.store((uint32_t)(nr.gFrCallMs * 1000.0));
                        }
                        else if (!in && l1Ok && nr.gFrL1Ratio > 0.0)
                        {
                            ema(nr.gFrCallMs, nr.gFrL1Ratio * l1);
                            nr.gFrCallUs.store((uint32_t)(nr.gFrCallMs * 1000.0));
                        }
                    }
                    else
                    {
                        // AMD FSR 4 queues its calls on the stream (its CPU call returns at once) and the stream waits
                        // for a node where it is first read, so a call's GPU time lands before or inside the slots: the
                        // prices come off a group's WHOLE time. A group with no call in its slots (x2: level 1 is in
                        // pre) gives the base and a slot's store; a group with calls in them gives a call = its whole
                        // time less the base and its extra stores, per call in the slots. The level-1 call's FSR span
                        // (fgL1Ev) is timed in every group: one with calls in its slots learns the call price over that
                        // span (gFrL1Ratio), an x2 group then prices its call by it, so a load that slowed FSR as it
                        // slowed the rest leaves no stale price once it passed. The base and the store are not
                        // re-measured at x4 and deeper, so the ratio takes them scaled by the span's load (this span
                        // over the x2 groups' span): that load's share of them stays out of the ratio. The call keeps
                        // the share: a held x4 is priced at its cost
                        const double whole = (double)pre + slots;
                        if (!in)
                        {
                            ema(nr.gFrStoreMs, (double)slots / nr.grSlots);
                            ema(nr.gFrBaseMs, whole);
                            if (l1Ok)
                            {
                                ema(nr.gFrL1BaseMs, l1);
                                if (nr.gFrL1Ratio > 0.0)
                                    ema(nr.gFrCallMs, nr.gFrL1Ratio * l1);
                            }
                        }
                        else if (nr.gFrBaseMs > 0.0)
                        {
                            const double rest = nr.gFrBaseMs + (nr.grSlots - 2.0) * nr.gFrStoreMs;
                            ema(nr.gFrCallMs, (std::max)(0.0, whole - rest) / in);
                            if (l1Ok && nr.gFrL1BaseMs > 0.0)
                            {
                                const double own = (whole - rest * l1 / nr.gFrL1BaseMs) / in;
                                if (own > 0.0)
                                    ema(nr.gFrL1Ratio, own / l1);
                            }
                        }
                        nr.gFrCallUs.store((uint32_t)(nr.gFrCallMs * 1000.0));
                        nr.gFrStoreUs.store((uint32_t)(nr.gFrStoreMs * 1000.0));
                        nr.gFrBaseUs.store((uint32_t)(nr.gFrBaseMs * 1000.0));
                    }
                }
                else if (nr.fruc && nr.frMp && nr.grFrCalls && nr.grSlots)
                {
                    // a call = the compute thread's calls' mean in every group (on a pipelined one its calls beside the
                    // worker's read a little dearer: the route's exit stays on the safe side). The pipelined route's
                    // two prices (the compute thread's chain a call incl. the join's wait, a worker call) are measured
                    // while it runs and, while the direct tree runs, follow the call by their ratios to the call before
                    // the last pipelined spell (gFrPipeRm / gFrPipeRw, its first kFrPipeLearn groups): a load that passed
                    // while the direct tree ran leaves no stale pipelined price
                    if (nr.grPiped && nr.grPipeLearn == kFrPipeLearn)
                        nr.gFrCallEntryMs = nr.gFrCallMs; // the call before this pipelined spell
                    ema(nr.gFrCallMs, nr.grFrCallMs / nr.grFrCalls);
                    if (!nr.grPiped)
                    {
                        nr.grPipeLearn = kFrPipeLearn;
                        if (nr.gFrPipeRm > 0.0)
                        {
                            nr.gFrPipeMainMs = nr.gFrCallMs * nr.gFrPipeRm;
                            nr.gFrPipeWkMs = nr.gFrCallMs * nr.gFrPipeRw;
                        }
                    }
                    else
                    {
                        const double main = nr.grFrChainMs / nr.grFrCalls;
                        const double wk = nr.grFrWkCalls ? nr.grFrWkMs / nr.grFrWkCalls : 0.0;
                        ema(nr.gFrPipeMainMs, main);
                        if (wk > 0.0)
                            ema(nr.gFrPipeWkMs, wk);
                        if (nr.grPipeLearn > 0 && nr.gFrCallEntryMs > 0.0 && wk > 0.0)
                        {
                            ema(nr.gFrPipeRm, main / nr.gFrCallEntryMs);
                            ema(nr.gFrPipeRw, wk / nr.gFrCallEntryMs);
                            nr.grPipeLearn--;
                        }
                    }
                    ema(nr.gFrStoreMs, (std::max)(0.0, ((double)slots - nr.grFrChainMs) / nr.grSlots));
                    nr.gFrCallUs.store((uint32_t)(nr.gFrCallMs * 1000.0));
                    nr.gFrStoreUs.store((uint32_t)(nr.gFrStoreMs * 1000.0));
                    nr.gFrPipeMainUs.store((uint32_t)(nr.gFrPipeMainMs * 1000.0));
                    nr.gFrPipeWkUs.store((uint32_t)(nr.gFrPipeWkMs * 1000.0));
                }
            }
        }
    }
    // the level-1 spans stay armed until read (the alternate set is the next group's); this group records into its own
    nr.grArmed = nr.grLagArmed = false;
    nr.fgEvSet = (int)(nr.seq % 2);
    nr.grFrCalls = nr.grFrPreCalls = nr.grFrWkCalls = 0;
    nr.grFrCallMs = nr.grFrPreMs = nr.grFrChainMs = nr.grFrWkMs = 0.0;

    // (1) the exe signalled the capture fence with this sequence number before writing us
    cudaExternalSemaphoreWaitParams wp{};
    wp.params.fence.value = nr.seq;
    if (cudaWaitExternalSemaphoresAsync(&nr.semCap, &wp, 1, st) != cudaSuccess)
    {
        nr.die("wait external semaphore failed");
        return false;
    }
    if (nr.pfEv[0])
        cudaEventRecord(nr.pfEv[0], st);
    // (2) read the shared capture texture; the event marks the exe's texture as free again. Live DLSS 5's reuse
    // test keeps the previous capture: the two buffers trade places and the new one lands in the older
    if (nr.dCapPrev)
        std::swap(nr.dCap, nr.dCapPrev);
    const size_t capRow = (size_t)nr.cw * (nr.hdr ? 8 : 4);
    if (cudaMemcpy2DFromArrayAsync(nr.dCap, capRow, nr.capArr, 0, 0, capRow, nr.ch, cudaMemcpyDeviceToDevice, st) !=
        cudaSuccess)
    {
        nr.die("capture memcpy2DFromArray failed");
        return false;
    }
    cudaEventRecord(nr.capEv, st);
    bool capSent = false;

    // (3) the previous cur becomes prev, then pack the new frame into the cur half
    // (native DRBA keeps its own history ring, so it skips this). dX's halves trade roles (xSwap) unless an IFNet reads
    // dX itself as its x, one tensor with prev in planes 0..2: then the cur half is copied into the prev half
    const size_t mplane = nr.mph ? (size_t)nr.mph * nr.mpw : plane;
    bool halvesTraded = false; // the cur half now holds the frame before the previous one
    if (nr.havePrev && !nr.noEngine && !nr.drba)
    {
        if (!nr.gmfss && !nr.nvof && !nr.fruc && !nr.mph && !nr.xHalf)
            cudaMemcpyAsync(nr.dX, nr.dX + 3 * plane, 3 * plane * sizeof(float), cudaMemcpyDeviceToDevice, st);
        else
        {
            nr.xSwap = !nr.xSwap;
            halvesTraded = true;
        }
        // the frames the IFNet reads keep the same two frames: the motion frames, or the fp16 copy an
        // fp16-x IFNet reads
        if (nr.mph)
            cudaMemcpyAsync(nr.dM, nr.dM + 3 * mplane, 3 * mplane * sizeof(uint16_t), cudaMemcpyDeviceToDevice, st);
        else if (nr.xHalf)
            cudaMemcpyAsync(nr.dXh, nr.dXh + 3 * plane, 3 * plane * sizeof(uint16_t), cudaMemcpyDeviceToDevice, st);
    }
    // DRBA: the ring slot of the frame this group pushes (the previous group's tweens, on st before it, are done with it)
    float* dCur = nr.drba ? drbaFrame(nr, nr.drFid + 1) : nativeXCur(nr);
    // live DLSS 5's identical-frame reuse: a capture byte-identical to the previous one takes the
    // last DLSS 5 output and skips TrueHDR, Restore, the resize and the evaluate (the history stays
    // untouched, so a paused picture stays exactly still)
    bool nrSame = false;
    if (nr.liveNr && !nr.nrPaused && !nr.nrFailed)
    {
        if (g_liveNrReset.exchange(false))
            nr.nrFirst = true; // the overlay was hidden: a new stream
        const long long nb = (long long)nr.cw * nr.ch * (nr.hdr ? 8 : 4);
        if (nr.nrReuse && nr.nrHaveLast && !nr.nrFirst)
        {
            void* ad[] = {&nr.dCapPrev, &nr.dCap, (void*)&nb, &nr.dStaticFlag};
            const unsigned blocks = (unsigned)((nb + 16LL * 256 - 1) / (16LL * 256));
            if (cudaMemsetAsync(nr.dStaticFlag, 0, sizeof(int), st) != cudaSuccess ||
                cuLaunchKernel(nr.fRawDiff, blocks, 1, 1, 256, 1, 1, 0, (CUstream)st, ad, nullptr) != CUDA_SUCCESS ||
                cudaMemcpyAsync(nr.hStaticFlag, nr.dStaticFlag, sizeof(int), cudaMemcpyDeviceToHost, st) !=
                    cudaSuccess ||
                cudaStreamSynchronize(st) != cudaSuccess)
            {
                nr.die("DLSS 5 reuse test failed");
                return false;
            }
            nrSame = *nr.hStaticFlag == 0;
        }
        if (nrSame)
        {
            // the last output is the previous frame's cur half, read only since: it is still the cur half where
            // nothing moved it (the copy shift, echo) and the prev half where the halves traded places; DRBA's is
            // the ring slot it was built in. The motion frame of an identical capture is the previous one (the shift
            // put it in the prev half; DRBA's push copies it inside its ring)
            if ((halvesTraded && cudaMemcpyAsync(dCur, nativeXPrev(nr), 3 * plane * sizeof(float),
                                                 cudaMemcpyDeviceToDevice, st) != cudaSuccess) ||
                (nr.drba && cudaMemcpyAsync(dCur, drbaFrame(nr, nr.drFid), 3 * plane * sizeof(float),
                                            cudaMemcpyDeviceToDevice, st) != cudaSuccess) ||
                (nr.mph && nr.havePrev && !nr.drba &&
                 cudaMemcpyAsync(nr.dM + 3 * mplane, nr.dM, 3 * mplane * sizeof(uint16_t), cudaMemcpyDeviceToDevice,
                                 st) != cudaSuccess))
            {
                nr.die("DLSS 5 reuse copy failed");
                return false;
            }
            nr.nrReused++;
        }
    }
    // live with RTX HDR: Restore, the resize and DLSS 5 work on the SDR picture and RTX TrueHDR converts it
    // after them (nativePreModelPost). The capture's SDR range as planes (R, G, B, sRGB-encoded
    // against the SDR reference white) in dSrcG; without RTX HDR, or once it failed, the chain works on PQ from
    // the capture on.
    const bool sdrPre = !nrSame && nr.rtxHdr && !nr.rtxFailed && g_rtxb.created;
    nr.nrSdrIn = sdrPre;
    if (sdrPre)
    {
        void* ae[] = {&nr.dCap, &nr.cw, &nr.ch, &nr.sdrScale, &nr.dSrcG};
        if (cuLaunchKernel(nr.fSdrEncode, (nr.cw + 15) / 16, (nr.ch + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, ae,
                           nullptr) != CUDA_SUCCESS)
        {
            nr.die("sdrEncode launch failed");
            return false;
        }
    }
    // RTX VSR before the model (the working size above the capture while the fit after the model does not
    // enlarge): the capture's planes through Restore (folded back to the capture size) and VSR to the working
    // size, offline's pre-model stage. It replaces Restore and the resize below; after a VSR failure those
    // run instead. VSR takes SDR: RTX HDR's SDR planes as they are (TrueHDR follows, NVIDIA's order), other HDR
    // planes as their SDR view, back to the codes after it with the light the view never held.
    bool restored = false;
    bool hdrIn = false; // this capture stayed inside the SDR range (the range test ran and said so)
    if (!nrSame && nr.vsrPre && !nr.vsrFailed)
    {
        const int cps = nr.cw * nr.ch;
        const bool viewHere = nr.hdr && !sdrPre;
        if (viewHere)
        {
            void* a0[] = {&nr.dCap, &nr.cw, &nr.ch, &nr.dCapF};
            if (cuLaunchKernel(nr.fPqPlanar, (nr.cw + 15) / 16, (nr.ch + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a0,
                               nullptr) != CUDA_SUCCESS ||
                !nativeHdrEnc(nr, nr.dCapF, 0, cps, nr.cw, nr.cw, nr.ch, nr.dVsrView, 0, cps, nr.cw, 1, 1.0f, 0.0f))
            {
                nr.die("capture SDR view (RTX VSR) launch failed");
                return false;
            }
            int in = 0;
            if (nr.dRsView && (in = nativeRsInRange(nr, nr.dCapF, cps, nr.cw, nr.cw, nr.ch, 1)) < 0)
            {
                nr.die("resize range test failed");
                return false;
            }
            hdrIn = in == 1;
        }
        else if (!sdrPre)
        {
            void* ap[] = {&nr.dCap, &nr.cw, &nr.ch, &nr.dCapF, &nr.ch, &nr.cw, (void*)&cps};
            if (cuLaunchKernel(nr.fPackInDirect, (nr.cw + 15) / 16, (nr.ch + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st,
                               ap, nullptr) != CUDA_SUCCESS)
            {
                nr.die("capture planes (RTX VSR) launch failed");
                return false;
            }
        }
        const void* src = viewHere ? nr.dVsrView : (sdrPre ? nr.dSrcG : nr.dCapF);
        int ps = cps, rs = nr.cw, half = 0;
        bool staged = false;
        if (!nativeOfflineStage(nr, src, ps, rs, half, staged, nr.cw, nr.ch, nr.w, nr.h, true, true, 0, false, nr.dCapF,
                                0, cps, nr.cw))
            return false;
        if (staged)
        {
            const int wps = nr.w * nr.h;
            // (AMD FSR upscaling on HDR planes staged the codes itself: k_fsrHdrOut; after a failure the view's fit)
            if (viewHere && !(nr.fuHdr && !nr.vsrFailed) &&
                (hdrIn ? !nativeHdrFromSdr(nr, nr.dPres, 0, wps, nr.w, nr.w, nr.h, nr.dPres, wps, nr.w, 1)
                       : !nativeFitPlanar(nr, nr.dCapF, 0, cps, nr.cw, nr.cw, nr.ch, nr.dVsrRem, nr.w, nr.h, st) ||
                             !nativeHdrRestOut(nr, nr.dPres, nr.w, nr.h, nr.dVsrRem, 0, wps, nr.w, nr.w, nr.h, 1)))
            {
                nr.die("capture HDR codes (RTX VSR) launch failed");
                return false;
            }
            const float* from = nr.dPres;
            int fps = wps, dps = (int)plane;
            void* a[] = {&from, &fps, &nr.w, &nr.w, &nr.h, &dCur, &nr.pw, &nr.ph, &dps};
            if (cuLaunchKernel(nr.fPadPlanar, (nr.pw + 15) / 16, (nr.ph + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a,
                               nullptr) != CUDA_SUCCESS)
            {
                nr.die("padPlanar (RTX VSR) launch failed");
                return false;
            }
            restored = true;
        }
    }
    // Restore: once per captured frame, on the capture itself (its SDR planes when
    // RTX HDR converts later), folded straight to the model size and into the model frame with the packers'
    // replicate pad, so every model reads restored frames; it replaces the resize below. When the
    // engine refuses, the pass is dropped for the session and the resize below runs instead.
    if (!restored && !nrSame && nr.restPre && nr.restNat && !nr.restFailed)
    {
        const int cps = nr.cw * nr.ch;
        if (!sdrPre)
        {
            // the capture as capture-size planes in the chain's colour space (PQ in HDR)
            void* ap[] = {&nr.dCap, &nr.cw, &nr.ch, &nr.dCapF, &nr.ch, &nr.cw, (void*)&cps};
            void* aq[] = {&nr.dCap, &nr.cw, &nr.ch, &nr.dCapF};
            if (cuLaunchKernel(nr.hdr ? nr.fPqPlanar : nr.fPackInDirect, (nr.cw + 15) / 16, (nr.ch + 15) / 16, 1, 16,
                               16, 1, 0, (CUstream)st, nr.hdr ? aq : ap, nullptr) != CUDA_SUCCESS)
            {
                nr.die("capture planes launch failed");
                return false;
            }
        }
        float* capPl = sdrPre ? nr.dSrcG : nr.dCapF;
        // HDR planes: Restore's reference at the working size takes the capture's SDR view when it stays in range
        int in = 0;
        if (!sdrPre && nr.dRsView && (in = nativeRsInRange(nr, nr.dCapF, cps, nr.cw, nr.cw, nr.ch, nr.encPost)) < 0)
        {
            nr.die("resize range test failed");
            return false;
        }
        hdrIn = in == 1;
        if (!nativeRestoreRun(nr, capPl, cps, nr.cw, nr.dRest, nr.w, nr.h, 0, sdrPre ? 0 : nr.encPost, in == 1,
                              nr.a4kUp && nr.vsr && !nr.vsrFailed))
            return false;
        if (!nr.restFailed)
        {
            int fps = nr.w * nr.h, dps = (int)plane;
            const float* from = nr.dRest;
            void* a[] = {&from, &fps, &nr.w, &nr.w, &nr.h, &dCur, &nr.pw, &nr.ph, &dps};
            if (cuLaunchKernel(nr.fPadPlanar, (nr.pw + 15) / 16, (nr.ph + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a,
                               nullptr) != CUDA_SUCCESS)
            {
                nr.die("padPlanar (restore) launch failed");
                return false;
            }
            restored = true;
        }
    }
    if (!restored && !nrSame)
    {
        const int ps = (int)plane;
        if (sdrPre)
        {
            // the SDR planes already exist at capture resolution, so resize them at every
            // image scale; at 1.00 the Lanczos taps sit on identity positions (weight 1 on the
            // pixel itself, 0 elsewhere) and the pair is an exact copy plus the replicate pad.
            const void* tabH = rsTabFor(nr, nr.cw, nr.w, st);
            const void* tabV = rsTabFor(nr, nr.ch, nr.h, st);
            if (!tabH || !tabV)
            {
                nr.die("resize tap tables (RTX HDR) failed");
                return false;
            }
            void* a1[] = {&nr.dSrcG, &nr.cw, &nr.ch, &nr.dTmp, &nr.w, (void*)&tabH};
            if (cuLaunchKernel(nr.fResizeHf, (nr.w + 15) / 16, (nr.ch + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a1,
                               nullptr) != CUDA_SUCCESS)
            {
                nr.die("resizeHf (RTX HDR) launch failed");
                return false;
            }
            void* a2[] = {&nr.dTmp, &nr.w, &nr.ch, &dCur, &nr.h, &nr.ph, &nr.pw, (void*)&ps, (void*)&tabV};
            if (cuLaunchKernel(nr.fResizeV, (nr.pw + 15) / 16, (nr.ph + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a2,
                               nullptr) != CUDA_SUCCESS)
            {
                nr.die("resizeV (RTX HDR) launch failed");
                return false;
            }
        }
        else if (nr.w == nr.cw && nr.h == nr.ch)
        {
            void* a[] = {&nr.dCap, &nr.cw, &nr.ch, &dCur, &nr.ph, &nr.pw, (void*)&ps};
            if (cuLaunchKernel(nr.hdr ? nr.fPackInDirectHdr : nr.fPackInDirect, (nr.pw + 15) / 16, (nr.ph + 15) / 16, 1,
                               16, 16, 1, 0, (CUstream)st, a, nullptr) != CUDA_SUCCESS)
            {
                nr.die("packInDirect launch failed");
                return false;
            }
        }
        else
        {
            const void* rsrc = nr.hdr ? (const void*)nr.dCapF : (const void*)nr.dCap;
            bool view = false;
            if (nr.hdr)
            {
                // convert the whole capture to PQ first, then resize ON PQ (the fill rule:
                // rescale on PQ, never rescale scRGB and convert after)
                void* a0[] = {&nr.dCap, &nr.cw, &nr.ch, &nr.dCapF};
                if (cuLaunchKernel(nr.fPqPlanar, (nr.cw + 15) / 16, (nr.ch + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st,
                                   a0, nullptr) != CUDA_SUCCESS)
                {
                    nr.die("pqPlanar launch failed");
                    return false;
                }
                // a capture inside the SDR range is resized as its SDR view (the SDR route's resize, clamped at SDR
                // white) and comes back to PQ after it, so its ringing never crosses white
                const int cps = nr.cw * nr.ch;
                int in = 0;
                if (nr.dRsView && (in = nativeRsInRange(nr, nr.dCapF, cps, nr.cw, nr.cw, nr.ch, nr.encPost)) < 0)
                {
                    nr.die("resize range test failed");
                    return false;
                }
                view = in == 1;
                hdrIn = view;
                if (view && !nativeHdrEnc(nr, nr.dCapF, 0, cps, nr.cw, nr.cw, nr.ch, nr.dRsView, 0, cps, nr.cw,
                                          nr.encPost, 1.0f, 0.0f))
                {
                    nr.die("resize SDR view launch failed");
                    return false;
                }
                if (view)
                    rsrc = nr.dRsView;
            }
            const void* tabH = rsTabFor(nr, nr.cw, nr.w, st);
            const void* tabV = rsTabFor(nr, nr.ch, nr.h, st);
            if (!tabH || !tabV)
            {
                nr.die("resize tap tables failed");
                return false;
            }
            void* a1[] = {(void*)&rsrc, &nr.cw, &nr.ch, &nr.dTmp, &nr.w, (void*)&tabH};
            if (cuLaunchKernel(nr.hdr ? nr.fResizeHf : nr.fResizeH, (nr.w + 15) / 16, (nr.ch + 15) / 16, 1, 16, 16, 1,
                               0, (CUstream)st, a1, nullptr) != CUDA_SUCCESS)
            {
                nr.die("resizeH launch failed");
                return false;
            }
            void* a2[] = {&nr.dTmp, &nr.w, &nr.ch, &dCur, &nr.h, &nr.ph, &nr.pw, (void*)&ps, (void*)&tabV};
            if (cuLaunchKernel(nr.fResizeV, (nr.pw + 15) / 16, (nr.ph + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a2,
                               nullptr) != CUDA_SUCCESS)
            {
                nr.die("resizeV launch failed");
                return false;
            }
            if (view && !nativeHdrFromSdr(nr, dCur, 0, ps, nr.pw, nr.pw, nr.ph, dCur, ps, nr.pw, nr.encPost))
            {
                nr.die("resize HDR codes launch failed");
                return false;
            }
        }
    }
    if (!nrSame)
        nr.hdrInBits = (nr.hdrInBits << 1) | (hdrIn ? 1u : 0u);
    // DLSS 5: once per captured frame on the model frame, after Restore and the
    // resize and before any model reads it; then Sharpen and RTX TrueHDR (nativePreModelPost), and the
    // frame the model reads is kept for the reuse above
    if (nr.liveNr && !nr.nrPaused && !nr.nrFailed && !nrSame && !nativeNrFrame(nr, dCur, nr.pw, nr.ph, (int)plane))
        return false;
    // RIFE's motion frame (two domains): the finished picture as the IFNet would have read it, minus
    // what costs RIFE its motion: the HDR desktop's PQ encoding (read as SDR sRGB over the desktop's
    // SDR white) and an enlarge (Lanczos back to the capture size), padded to the motion frame. Taken
    // after Sharpen, or before the post with RTX HDR (the picture is still SDR there, Sharpen is fused
    // into TrueHDR's input). An identical capture copied it above.
    auto motionFrame = [&]() -> bool {
        const float* msrc = dCur;
        int sps = (int)plane, srs = nr.pw, sw = nr.w, sh = nr.h;
        if (nr.dMFit)
        {
            int n = 3 * nr.mw * nr.mh;
            void* b[] = {&nr.dMFit, &n};
            if (!nativeFitPlanar(nr, dCur, 0, sps, nr.pw, nr.w, nr.h, nr.dMFit, nr.mw, nr.mh, st) ||
                cuLaunchKernel(nr.fClamp01, (n + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, b, nullptr) !=
                    CUDA_SUCCESS)
                return false;
            msrc = nr.dMFit;
            sps = nr.mw * nr.mh;
            srs = nr.mw;
            sw = nr.mw;
            sh = nr.mh;
        }
        int dps = (int)mplane, mode = nr.hdr && !sdrPre ? 1 : 0;
        float white = (float)(g_sdrWhite / 80.0), knee = kMotionPqKnee, cap = kMotionPqCap;
        uint16_t* mdst = nr.drba ? drbaMotionNext(nr) : nr.dM + 3 * mplane;
        int zp = nr.zeroPad ? 1 : 0, stb = nr.drAuthor ? 1 : 0;
        void* a[] = {(void*)&msrc, &sps,    &srs,  &sw,    &sh,   &mdst, &dps, &nr.mpw,
                     &nr.mpw,      &nr.mph, &mode, &white, &knee, &cap,  &zp,  &stb};
        return cuLaunchKernel(nr.fMotionIn, (nr.mpw + 15) / 16, (nr.mph + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a,
                              nullptr) == CUDA_SUCCESS;
    };
    const bool motionHere = nr.mph && !nrSame;
    if (motionHere && sdrPre && !motionFrame())
    {
        nr.die("motion frame launch failed");
        return false;
    }
    if (!nrSame && !nativePreModelPost(nr, dCur, (int)plane, sdrPre))
        return false;
    if (motionHere && !sdrPre && !motionFrame())
    {
        nr.die("motion frame launch failed");
        return false;
    }
    if (nr.liveNr && !nr.nrPaused && !nr.nrFailed && !nrSame)
        nr.nrHaveLast = true;

    // native DRBA: the new frame and its encode into the history ring; the group then shows
    // the LAGGED frame k - 1 as its real frame (the first capture: itself, one slot:
    // DRBA's first group)
    int drHist = 0;
    float* drLag = dCur;
    if (nr.drba)
    {
        if (!nativeDrbaPush(nr, dCur, nrSame))
            return false;
        drHist = nr.drFid < 4 ? (int)nr.drFid : 4;
        if (drHist >= 2)
            drLag = drbaFrame(nr, nr.drFid - 1);
        else if (nfr > 1)
            nfr = 1;
    }

    // (4) the pair. How many tweens this group asks for, in fracs order, decides whether the
    // pair-level work is needed at all.
    uint32_t nTween = 0;
    for (uint32_t i = 0; i < nfr; i++)
        if (fr[i] < 0.999f)
            nTween++;
    if (!nr.havePrev)
        nTween = 0; // first pair: nothing to interpolate toward
    // IDENTICAL PAIR: the two packed inputs are compared element by element on
    // the device and the flag is read back once. Equal = no motion exists in this pair, so the
    // tween work is dropped (exactly the tween-less group the adaptive ladder already
    // produces: the per-frame chain state, the encode or feat_ext, still runs for the next
    // pair) and every slot presents the real frame below. The readback sync costs only the
    // overlap between this group's pack and its own tweens: the compute thread already waits
    // for the whole group in drainReady(true) before it takes the next message.
    bool pairStatic = false;
    // native DRBA tests the LAGGED pair (k-2, k-1): every tween of its group lies inside it
    // (DRBA's static rule, three frames of history needed)
    if (nTween && g_staticHold && nr.fPairDiff && nr.dStaticFlag && nr.hStaticFlag && (!nr.drba || drHist >= 3))
    {
        const int n = (int)(3 * plane);
        float* dPrev = nr.drba ? drbaFrame(nr, nr.drFid - 2) : nativeXPrev(nr);
        float* dNext = nr.drba ? drLag : dCur;
        if (cudaMemsetAsync(nr.dStaticFlag, 0, sizeof(int), st) != cudaSuccess)
        {
            nr.die("static flag clear failed");
            return false;
        }
        void* ad[] = {&dPrev, &dNext, (void*)&n, &nr.dStaticFlag};
        if (cuLaunchKernel(nr.fPairDiff, (n + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, ad, nullptr) !=
            CUDA_SUCCESS)
        {
            nr.die("pairDiff launch failed");
            return false;
        }
        if (cudaMemcpyAsync(nr.hStaticFlag, nr.dStaticFlag, sizeof(int), cudaMemcpyDeviceToHost, st) != cudaSuccess ||
            cudaStreamSynchronize(st) != cudaSuccess)
        {
            nr.die("static flag readback failed");
            return false;
        }
        if (*nr.hStaticFlag == 0)
        {
            pairStatic = true;
            nTween = 0;
            g_staticHeld = ++nr.staticN;
        }
    }
    // RIFE: the pair encode. The previous frame's encode is REUSED (encode is a pure function
    // of the frame, so this is exact and halves the Head work per pair).
    nr.fCur ^= 1;
    if (nr.gmfss)
    {
        // GMFSS: feat_ext of the new frame, then the halves, the bidir flow,
        // the metrics and the two pyramid levels when this group interpolates
        if (!nativeGmfssPair(nr, nativeXPrev(nr), dCur, nTween > 0))
            return false;
    }
    else if (nr.nvof)
    {
        // the nvof model: no per-frame state (both frames go to gray8 per pair), so a
        // tween-less or identical pair costs nothing; the identical-pair gate above already
        // zeroed nTween, an identical pair never reaches the Optical Flow engine
        if (nTween && !nativeNvofPair(nr, nativeXPrev(nr), dCur))
            return false;
    }
    else if (nr.fruc && pipe)
    {
        // Smooth Motion / AMD FSR 4 pipelined: the static test above judged the NEW pair, which the next group shows;
        // this group shows the previous pair (nativeFrucPipeAdvance / nativeFsrfgPipeAdvance), held when it was
        // identical or never split. The levels' instances are made here, before instance 0's worker / call runs
        const bool curStatic = pairStatic;
        if (!(nr.fsrfg ? nativeFsrfgPipeAdvance(nr, dCur, curStatic) : nativeFrucPipeAdvance(nr, dCur, curStatic)))
            return false;
        pairStatic = !nr.frLagHave || nr.frLagStatic;
        nTween = 0;
        if (!pairStatic)
        {
            double tw[64];
            for (uint32_t i = 0; i < nfr && nTween < 64; i++)
                if (fr[i] < 0.999f)
                    tw[nTween++] = fr[i];
            const int want = nTween ? (std::min)(nativeFrucDepth(nr, tw, nTween), nr.frInstMax) : 0;
            if (nr.frInst < want)
            {
                if (nr.fsrfg)
                    nativeFsrfgGrow(nr, want);
                else
                    nativeFrucGrow(nr, want);
            }
            if (want)
                nr.frPipeDepth = want;
        }
    }
    else if (nr.fruc)
    {
        // Smooth Motion: every frame is packed (the next pair's start), the bridge itself
        // runs per tween below; an identical pair got nTween = 0 above and never reaches FRUC
        if (nr.fsrfg && !nativeFsrfgPipeExit(nr))
            return false;
        nr.frPipePrev = false;
        nr.frCurStatic = pairStatic;
        if (!nativeFrucPair(nr, dCur, nTween))
            return false;
        if (nTween > 1)
        {
            // the tweens in slot order, for the parallel instances
            double tw[64];
            uint32_t n = 0;
            for (uint32_t i = 0; i < nfr && n < 64; i++)
                if (fr[i] < 0.999f)
                    tw[n++] = fr[i];
            nativeFrucPlan(nr, tw, n);
        }
    }
    else if (nr.drba)
    {
        // native DRBA: the encode already went into the ring (nativeDrbaPush), the windows
        // are built lazily by the tweens below
    }
    else if (!nr.noEngine)
    {
        // fp16 frames: the new frame into the fp16 copy's cur half for an fp16-x IFNet (every new frame, static pairs
        // too); RIFE's two domains: the motion frame
        if (!nr.mph && nr.xHalf && !nativeF2h(nr, dCur, nr.dXh + 3 * plane, 3 * plane, st))
        {
            nr.die("f2h launch failed");
            return false;
        }
        // the encode reads the motion frame (fp16), else the new frame itself (fp32), into dF in the IFNet's dtype
        if (!(nr.mph ? nativeEncodeRun(nr, nr.dM + 3 * mplane, 1, nr.mpw, nr.mph, nr.dF[nr.fCur], st)
                     : nativeEncodeRun(nr, dCur, 0, nr.pw, nr.ph, nr.dF[nr.fCur], st)))
            return false;
    }

    // (5) slots. Tokens go out in slot-index order, exactly like _drain_ready.
    if (nr.grEv[0])
        cudaEventRecord(nr.grEv[0], st);
    nr.grFrPreMs = nr.grFrCallMs;
    nr.grFrPreCalls = nr.grFrCalls;
    // a pipelined group: the new pair's levels 1..k start on instance 0's worker beside this group's slots (joined in
    // nativeFrucLagTween, or below when the shown pair is held)
    if (pipe && !nr.fsrfg && !nativeFrucPipeSpec(nr))
        return false;
    const int pitchI = (int)nr.pitch;
    uint32_t sent = 0;
    auto drainReady = [&](bool block) {
        if (!capSent &&
            (block ? cudaEventSynchronize(nr.capEv) == cudaSuccess : cudaEventQuery(nr.capEv) == cudaSuccess))
        {
            capSent = true;
            nr.pushTok(0x7FFFFFFFu);
        }
        while (sent < nfr && (block ? cudaEventSynchronize(nr.slotEv[sent]) == cudaSuccess
                                    : cudaEventQuery(nr.slotEv[sent]) == cudaSuccess))
        {
            nr.pushTok(sent + 1);
            sent++;
        }
    };

    uint32_t twDone = 0;                  // tweens already computed
    uint32_t chunkBase = 0, chunkLen = 0; // current chunk's [base, base+len) in tween order
    bool fail = false;

    // one presented frame from a model-size planar source into its slot. Without effects a 1:1
    // slot takes the plain packer, and a fit that changes the size runs the Lanczos pair's
    // horizontal pass into dFitTmp, then its vertical pass fused with the slot store
    // (k_packOutV / k_packOutHdrV). With live effects: RTX VSR (the bridge, model size ->
    // presented size, 8-bit in and out, host-synchronous like every bridge call) or the Lanczos
    // pair into the planar staging frame, then the sharpen in the slot store. A VSR eval failure demotes
    // the rest of the run to Lanczos with one line.
    // (half != 0 on any of these = an fp16 source: a tween as the IFNet wrote it)
    auto packFrom = [&](const void* src, int ps, int rs, int sw, int sh, uint8_t* slot, const char* what,
                        int half = 0) -> bool {
        void* a[] = {(void*)&src, &half,          (void*)&ps, (void*)&rs, &sw,    &sh,
                     &slot,       (void*)&pitchI, &nr.x0,     &nr.y0,     &nr.dw, &nr.dh};
        if (cuLaunchKernel(nr.hdr ? nr.fPackOutHdr : nr.fPackOut, (nr.dw + 15) / 16, (nr.dh + 15) / 16, 1, 16, 16, 1, 0,
                           (CUstream)st, a, nullptr) != CUDA_SUCCESS)
        {
            nr.die(what);
            return false;
        }
        return true;
    };
    auto plainPack = [&](const void* src, uint8_t* slot, const char* what, int half = 0) -> bool {
        return packFrom(src, (int)plane, nr.pw, nr.w, nr.h, slot, what, half);
    };
    // a fit with no effect after it: the pair's horizontal pass into dFitTmp (3, sh, dw), then the
    // vertical pass fused with the slot store
    auto fitPack = [&](const void* src, int ps, int rs, int sw, int sh, uint8_t* slot, const char* what,
                       int half = 0) -> bool {
        const void* tabH = rsTabFor(nr, sw, nr.dw, st);
        const void* tabV = rsTabFor(nr, sh, nr.dh, st);
        if (!tabH || !tabV)
        {
            nr.die(what);
            return false;
        }
        void* ah[] = {(void*)&src, &half, &ps, &rs, &sw, &sh, &nr.dFitTmp, &nr.dw, (void*)&tabH};
        if (cuLaunchKernel(nr.fFitAaH, (nr.dw + 15) / 16, (sh + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, ah, nullptr) !=
            CUDA_SUCCESS)
        {
            nr.die(what);
            return false;
        }
        void* av[] = {&nr.dFitTmp, &nr.dw, &sh, &slot, (void*)&pitchI, &nr.x0, &nr.y0, &nr.dh, (void*)&tabV};
        if (cuLaunchKernel(nr.hdr ? nr.fPackOutHdrV : nr.fPackOutV, (nr.dw + 15) / 16, (nr.dh + 15) / 16, 1, 16, 16, 1,
                           0, (CUstream)st, av, nullptr) != CUDA_SUCCESS)
        {
            nr.die(what);
            return false;
        }
        return true;
    };
    // one resize of a planar source (ps / rs / sw x sh) into a planar target tw x th: the Lanczos pair through tmp
    // (3, sh, tw) when aa, else k_fitPlanar's 1:1 crop copy
    auto resizePlanar = [&](const void* src, int ps, int rs, int sw, int sh, bool aa, float* tmp, float* dst, int tw,
                            int th, int half = 0) -> bool {
        if (aa)
        {
            const void* tabH = rsTabFor(nr, sw, tw, st);
            const void* tabV = rsTabFor(nr, sh, th, st);
            if (!tabH || !tabV)
            {
                nr.die("fitAa tap tables failed");
                return false;
            }
            void* ah[] = {(void*)&src, &half, &ps, &rs, &sw, &sh, &tmp, &tw, (void*)&tabH};
            if (cuLaunchKernel(nr.fFitAaH, (tw + 15) / 16, (sh + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, ah,
                               nullptr) != CUDA_SUCCESS)
            {
                nr.die("fitAaH launch failed");
                return false;
            }
            void* av[] = {&tmp, &tw, &sh, &dst, &th, (void*)&tabV};
            if (cuLaunchKernel(nr.fFitAaV, (tw + 15) / 16, (th + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, av,
                               nullptr) != CUDA_SUCCESS)
            {
                nr.die("fitAaV launch failed");
                return false;
            }
            return true;
        }
        void* a[] = {(void*)&src, &half, &ps, &rs, &sw, &sh, &dst, &tw, &th};
        if (cuLaunchKernel(nr.fFitPlanar, (tw + 15) / 16, (th + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a, nullptr) !=
            CUDA_SUCCESS)
        {
            nr.die("fitPlanar launch failed");
            return false;
        }
        return true;
    };
    // one output slot's chain after the model (Restore, DLSS 5, Sharpen and RTX HDR already ran on the
    // captured frame): RTX VSR or the Lanczos resize, Upscale to
    auto storeSlot = [&](const void* src, uint8_t* slot, const char* what, int half = 0) -> bool {
        if (!src)
            return false; // a source that failed before (its line is logged)
        int ps = (int)plane, rs = nr.pw;
        const bool vsrNow =
            nr.vsr && !nr.vsrPre && !nr.vsrFailed && (g_rtxb.created || nr.fsrUp || nr.a4kUp || nr.acnUp);
        if (!vsrNow && !nr.fitAa && !nr.uw)
            return plainPack(src, slot, what, half);
        if (!vsrNow && nr.fitAa && !nr.uw)
            return fitPack(src, ps, rs, nr.w, nr.h, slot, what, half);
        // Upscale to: the first resize lands in the internal render frame (uw x uh)
        // instead of the staging frame, then the fit takes it to (dw, dh) below
        const int tw = nr.uw ? nr.uw : nr.dw, th = nr.uw ? nr.uh : nr.dh;
        float* stage = nr.uw ? nr.dUp : nr.dPres;
        bool haveVsr = false;
        if (vsrNow && (nr.a4kUp || nr.acnUp))
        {
            // Anime4K / ArtCNN: the model frame (HDR planes: its SDR view, as RTX VSR reads it) through its x2 net and
            // the Lanczos pair into the stage; HDR planes come back to the codes in RTX VSR's block below
            const void* a4kSrc = src;
            int aHalf = half, aps = ps, ars = rs;
            if (nr.hdr)
            {
                aps = nr.w * nr.h;
                ars = nr.w;
                if (!nativeHdrEnc(nr, src, half, ps, rs, nr.w, nr.h, nr.dVsrView, 0, aps, ars, 1, 1.0f, 0.0f))
                {
                    nr.die(nr.acnUp ? "ArtCNN SDR view launch failed" : "Anime4K SDR view launch failed");
                    return false;
                }
                a4kSrc = nr.dVsrView;
                aHalf = 0;
            }
            if (!(nr.acnUp ? nativeAcnRun(nr, a4kSrc, aHalf, aps, ars, stage, tw, th)
                           : nativeA4kRun(nr, a4kSrc, aHalf, aps, ars, stage, tw, th)))
                return false;
            haveVsr = true;
        }
        else if (vsrNow)
        {
            // HDR planes: RTX VSR takes SDR, so it reads the frame's SDR view; the way back to the codes is below
            // (HDR planes through AMD FSR upscaling: the view and the linear frame in ONE pass over the planes, below)
            const bool onePass = nr.hdr && nr.fuHdr;
            const void* vsrSrc = nr.hdr ? (const void*)nr.dVsrView : src;
            int vHalf = nr.hdr ? 0 : half, vps = nr.hdr ? nr.w * nr.h : ps, vrs = nr.hdr ? nr.w : rs;
            if (nr.hdr && !onePass &&
                !nativeHdrEnc(nr, src, half, ps, rs, nr.w, nr.h, nr.dVsrView, 0, vps, vrs, 1, 1.0f, 0.0f))
            {
                nr.die("VSR SDR view launch failed");
                return false;
            }
            void* a[] = {(void*)&vsrSrc, &vHalf, (void*)&vps, (void*)&vrs, &nr.w, &nr.h, &nr.dVsrIn};
            if (!onePass && cuLaunchKernel(nr.fsrUp ? nr.fPackBgraRgb : nr.fPackBgra, (nr.w + 15) / 16,
                                           (nr.h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a, nullptr) != CUDA_SUCCESS)
            {
                nr.die("packBgra launch failed");
                return false;
            }
            if (onePass)
            {
                // the SDR view packed for the vectors (k_hdrEnc + k_packBgraRgb's bytes) and the planes in linear
                // light for the bridge (k_fsrHdrIn's halves) from one read of the frame (k_fsrHdrPack); the way back
                // is k_fsrHdrOut below
                const float scale = 125.0f / nr.encWhite;
                int mode = 1, hlg = nr.encPost == 2 ? 1 : 0;
                void* f[] = {(void*)&src,  &half,         (void*)&ps, (void*)&rs, &nr.w,     &nr.h,        &mode,
                             &nr.encWhite, (void*)&scale, &hlg,       &nr.dVsrIn, &nr.dFuIn, &nr.fuInPitch};
                if (cuLaunchKernel(nr.fFsrHdrPack, (nr.w + 15) / 16, (nr.h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, f,
                                   nullptr) != CUDA_SUCCESS)
                {
                    nr.die("FSR HDR pack launch failed");
                    return false;
                }
            }
            else if (nr.fuHdr)
            {
                // AMD FSR upscaling on HDR planes: the frame in linear light into the bridge's buffer (the view above
                // feeds the vectors only); the way back is k_fsrHdrOut below
                const float scale = 125.0f / nr.encWhite;
                int hlg = nr.encPost == 2 ? 1 : 0;
                void* f[] = {(void*)&src, &half,         (void*)&ps, (void*)&rs, &nr.w,
                             &nr.h,       (void*)&scale, &hlg,       &nr.dFuIn,  &nr.fuInPitch};
                if (cuLaunchKernel(nr.fFsrHdrIn, (nr.w + 15) / 16, (nr.h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, f,
                                   nullptr) != CUDA_SUCCESS)
                {
                    nr.die("FSR HDR input launch failed");
                    return false;
                }
            }
            const RtxRect ri{0, 0, (uint32_t)nr.w, (uint32_t)nr.h};
            const RtxRect ro{0, 0, (uint32_t)tw, (uint32_t)th};
            unsigned int rv = 0;
            if (!nativeVsrEval(nr, ri, ro, st, rv))
                return false;
            if (rv == 1u)
                haveVsr = true;
            else
            {
                LOG("native: RTX VSR eval failed (rc %u), Lanczos for the rest of the run\n", rv);
                nr.vsrFailed = true;
                if (!nr.uw) // VSR = enlarging fit
                    return fitPack(src, ps, rs, nr.w, nr.h, slot, what, half);
            }
        }
        if (haveVsr && !nr.uw && !nr.hdr && !nr.a4kUp && !nr.acnUp)
        {
            // the bridge output is the presented frame: into the content rect, RTX VSR's R, G, B, A as the slot's
            // B, G, R, A in one pass; AMD FSR upscaling's buffer already holds the slot's order
            uint8_t* to = slot + (size_t)nr.y0 * nr.pitch + (size_t)nr.x0 * 4;
            int srcPitch = nr.dw * 4;
            void* a[] = {&nr.dVsrOut, &srcPitch, &nr.dw, &nr.dh, &to, (void*)&pitchI};
            const bool ok = nr.fsrUp ? cudaMemcpy2DAsync(to, nr.pitch, nr.dVsrOut, (size_t)nr.dw * 4, (size_t)nr.dw * 4,
                                                         nr.dh, cudaMemcpyDeviceToDevice, st) == cudaSuccess
                                     : cuLaunchKernel(nr.fRgbaBgra, (nr.dw + 15) / 16, (nr.dh + 15) / 16, 1, 16, 16, 1,
                                                      0, (CUstream)st, a, nullptr) == CUDA_SUCCESS;
            if (!ok)
            {
                nr.die("VSR slot copy failed");
                return false;
            }
            return true;
        }
        // the first resize target (the internal render frame, or the staging frame at the
        // presented size): the VSR output unpacked, or the Lanczos fit
        if (haveVsr && nr.fuHdr)
        {
            // AMD FSR upscaling on HDR planes: the bridge's linear-light result back to the codes (k_fsrHdrOut), no
            // SDR view to come back from
            const float inv = nr.encWhite / 125.0f;
            int hlg = nr.encPost == 2 ? 1 : 0;
            void* a[] = {&nr.dFuOut, &nr.fuOutPitch, (void*)&tw, (void*)&th, (void*)&inv, &hlg, &stage};
            if (cuLaunchKernel(nr.fFsrHdrOut, (tw + 15) / 16, (th + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a,
                               nullptr) != CUDA_SUCCESS)
            {
                nr.die("FSR HDR output launch failed");
                return false;
            }
        }
        else if (haveVsr)
        {
            void* a[] = {&nr.dVsrOut, (void*)&tw, (void*)&th, &stage};
            if (!nr.a4kUp && !nr.acnUp && // Anime4K / ArtCNN wrote the stage itself
                cuLaunchKernel(nr.fsrUp ? nr.fUnpackBgraRgb : nr.fUnpackRgba, (tw + 15) / 16, (th + 15) / 16, 1, 16, 16,
                               1, 0, (CUstream)st, a, nullptr) != CUDA_SUCCESS)
            {
                nr.die("unpackBgra launch failed");
                return false;
            }
            // HDR planes: back to the codes. The frames this one came from all inside the SDR range take the view's way
            // back; any other gets the light the view never held from the frame's Lanczos fit (k_hdrRestOut). A DRBA
            // frame is 3 captures old, so the last 4 count
            const int fps = tw * th;
            if (nr.hdr && ((nr.hdrInBits & 15u) == 15u
                               ? !nativeHdrFromSdr(nr, stage, 0, fps, tw, tw, th, stage, fps, tw, 1)
                               : !resizePlanar(src, ps, rs, nr.w, nr.h, true, nr.uw ? nr.dUpTmp : nr.dFitTmp,
                                               nr.dVsrRem, tw, th, half) ||
                                     !nativeHdrRestOut(nr, stage, tw, th, nr.dVsrRem, 0, fps, tw, tw, th, 1)))
            {
                nr.die("VSR HDR codes launch failed");
                return false;
            }
        }
        else if (!resizePlanar(src, ps, rs, nr.w, nr.h, nr.uw ? nr.upAa : nr.fitAa, nr.uw ? nr.dUpTmp : nr.dFitTmp,
                               stage, tw, th, half))
            return false;
        // the second resize, internal render frame -> the fit rect: the pair with the fused store (the
        // plain packer's 1:1 copy when the size stays)
        if (nr.uw)
            return nr.fitAa ? fitPack(nr.dUp, nr.uw * nr.uh, nr.uw, nr.uw, nr.uh, slot, what)
                            : packFrom(nr.dUp, nr.uw * nr.uh, nr.uw, nr.uw, nr.uh, slot, what);
        // the staging frame is the presented frame, stored 1:1 (sampleOut reads it as a dw x dh source with no pad)
        return packFrom(nr.dPres, nr.dw * nr.dh, nr.dw, nr.dw, nr.dh, slot, what);
    };
    uint8_t* heldSlot = nullptr; // a held pair's first real slot, copied into its other slots
    for (uint32_t i = 0; i < nfr && !fail; i++)
    {
        uint8_t* slot = nr.dOutRing + ((size_t)set * nr.slots + i) * nr.slotBytes;
        const bool real = !(fr[i] < 0.999f) || !nr.havePrev || pairStatic;
        if (real)
        {
            if (pairStatic && heldSlot)
            {
                // the held pair's remaining slots: the real frame's finished slot, copied as
                // it is, so the per-frame passes (VSR, the fit, the sharpen) run ONCE
                if (cudaMemcpyAsync(slot, heldSlot, nr.slotBytes, cudaMemcpyDeviceToDevice, st) != cudaSuccess)
                {
                    nr.die("held slot copy failed");
                    fail = true;
                    break;
                }
            }
            else if (!storeSlot(nr.drba ? nativeDrbaReal(nr, drLag)
                                : pipe  ? nativeXPrev(nr)
                                        : dCur,
                                slot, "packOut (real) launch failed"))
            {
                fail = true;
                break;
            }
            else
                heldSlot = slot;
        }
        else if (nr.drba)
        {
            // native DRBA: one IFNet enqueue per tween with its own DRM timestep map, or the
            // lagged real frame where the group holds (a head without enough history)
            bool held = false;
            if (!nativeDrbaTween(nr, fr[i], drHist, held, false, nr.blendOut ? slot : nullptr))
            {
                fail = true;
                break;
            }
            if ((held || !nr.blendOut) &&
                !storeSlot(held ? (const void*)nativeDrbaReal(nr, drLag) : mergedAt(nr, 0, plane), slot,
                           "packOut (drba) launch failed", held ? 0 : (int)nr.outHalf))
            {
                fail = true;
                break;
            }
            if (!held)
                twDone++;
        }
        else if (nr.gmfss)
        {
            // GMFSS: one tween at a time (no batch axis anywhere in its four engines), the
            // clamped fusionnet output then rides the shared effects and slot chain
            if (!nativeGmfssTween(nr, fr[i]))
            {
                fail = true;
                break;
            }
            if (!storeSlot(nr.dGmF, slot, "packOut (gmfss) launch failed"))
            {
                fail = true;
                break;
            }
            twDone++;
        }
        else if (nr.nvof)
        {
            // the nvof model: one splat pair per tween into the model layout, then the shared
            // effects and slot chain like every other model's output
            if (!nativeNvofTween(nr, fr[i]))
            {
                fail = true;
                break;
            }
            if (!storeSlot(nr.dNvOut, slot, "packOut (nvof) launch failed"))
            {
                fail = true;
                break;
            }
            twDone++;
        }
        else if (nr.fruc)
        {
            // Smooth Motion: one bridge call per tween (it syncs the context on entry, so the
            // previous tween's store has read dFrOut before the next unpack overwrites it)
            if (!(pipe ? (nr.fsrfg ? nativeFsrfgLagTween(nr, fr[i]) : nativeFrucLagTween(nr, fr[i]))
                       : nativeFrucTween(nr, fr[i])))
            {
                fail = true;
                break;
            }
            if (!storeSlot(nr.dFrOut, slot, "packOut (fruc) launch failed"))
            {
                fail = true;
                break;
            }
            twDone++;
        }
        else
        {
            if (twDone >= chunkBase + chunkLen)
            {
                // next chunk of tweens, up to the engine's batch (1: the unbatched class)
                chunkBase = twDone;
                chunkLen = nTween - chunkBase;
                if (chunkLen > (uint32_t)nr.batchMax)
                    chunkLen = (uint32_t)nr.batchMax;
                // timestep planes are constant maps of t
                // collect the chunk's t values in tween order
                float ts[64];
                if (chunkLen > 64)
                    chunkLen = 64;
                uint32_t k = 0, seen = 0;
                for (uint32_t j = 0; j < nfr; j++)
                {
                    if (!(fr[j] < 0.999f))
                        continue;
                    if (seen >= chunkBase && k < chunkLen)
                        ts[k++] = fr[j];
                    seen++;
                }
                for (uint32_t j = 0; j < chunkLen; j++)
                {
                    if (!nativeFillT(nr, j, ts[j], mplane, st))
                    {
                        nr.die("timestep fill failed");
                        fail = true;
                        break;
                    }
                }
                if (fail)
                    break;
                // the IFNet's frame: the motion frame (two domains), else the pictures' pad
                const int iph = nr.mph ? nr.mph : nr.ph, ipw = nr.mph ? nr.mpw : nr.pw;
                if (!nativeIfnRun(nr, nr.mph ? (const void*)nr.dM : (const void*)nr.dXh, nr.dT, nr.dF[nr.fCur ^ 1],
                                  nr.dF[nr.fCur], ipw, iph, nr.dFlow, nr.dMask, st))
                {
                    nr.die("IFNet launch failed");
                    fail = true;
                    break;
                }
                // the chunk's tweens: RIFE's last step on the frames this enqueue read, or on the fp32
                // pictures at the working size (two domains); blendOut makes each one in its slot below
                const bool picHalf = !nr.mph && nr.xHalf;
                if (!nr.blendOut && !nativeRifeBlend(nr, picHalf ? (const void*)nr.dXh : (const void*)nr.dX,
                                                     (int)picHalf, (int)chunkLen, st, !picHalf && nr.xSwap))
                {
                    nr.die("rifeBlend launch failed");
                    fail = true;
                    break;
                }
            }
            const uint32_t off = twDone - chunkBase;
            if (nr.blendOut)
            {
                const bool picHalf = !nr.mph && nr.xHalf;
                if (!nativeRifeBlendOut(nr, picHalf ? (const void*)nr.dXh : (const void*)nr.dX, (int)picHalf, (int)off,
                                        slot, st, !picHalf && nr.xSwap))
                {
                    nr.die("rifeBlendOut launch failed");
                    fail = true;
                    break;
                }
            }
            else if (!storeSlot(mergedAt(nr, off, plane), slot, "packOut launch failed", (int)nr.outHalf))
            {
                fail = true;
                break;
            }
            twDone++;
        }
        cudaEventRecord(nr.slotEv[i], st);
        drainReady(false);
    }
    // a pipelined group's last step: Smooth Motion's worker joined, AMD FSR 4's level-1 call issued (when no tween did)
    if (fail || (pipe && !(nr.fsrfg ? nativeFsrfgL1Issue(nr) : nativeFrucSpecJoin(nr))))
        return false;
    if (!pipe)
        nr.grFrChainMs = nr.grFrCallMs - nr.grFrPreMs;
    // the first group has no pair and a group with no slot stores nothing: their times say nothing about either kind
    if (nr.grEv[1] && nr.havePrev && nfr)
    {
        cudaEventRecord(nr.grEv[1], st);
        nr.grArmed = true;
        nr.grPiped = pipe;
        nr.grTween = twDone;
        nr.grSlots = nfr;
    }
    drainReady(true); // sync the stragglers, exactly like _finish
    if (nr.gmProf && nr.gmProfTween)
        nativeGmfssProfile(nr);
    nr.pushTok(0x80000000u); // bare end marker closes the group
    nr.havePrev = true;
    nativeVideoMemoryNote(); // once a session, after its first group
    return true;
}
