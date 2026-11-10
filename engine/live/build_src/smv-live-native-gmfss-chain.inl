// part of smv-live-native.inl (included there, in place); not a standalone translation unit
// ---- the GMFSS chain ------------------------------------------------------------------------
// GMFSS_infer_u.Model.reuse() and .inference() as native launches on the group's stream: the
// four engines, the native nets and the glue kernels (k_half / k_pyr / k_splatSoft / k_splatNorm, gated against a
// float64 reference). Two differences from
// the author's code, both exact:
//   * feat_ext runs ONCE per group (on the new frame) and the previous group's output is this
//     pair's feat0: the same reuse the native RIFE route does with the encode (feat_ext is a
//     pure function of the frame); the author's code recomputes both sets per pair.
//   * the flow and metric pyramids are built once per PAIR and the timestep is applied by the
//     splat: the author's interp(t * flow) * 0.5 equals t * (interp(flow) * 0.5).
// Everything else follows the author's op order literally, including which splat takes s = t and
// which takes s = 1 - t. Shapes come from the engines (nativeGmfssSetup),
// and the warm-up there left every static input shape set on these contexts, so only addresses
// are set here.
static bool nativeGmfssPair(NativeRife& nr, float* dPrev, float* dCur, bool needFlow)
{
    cudaStream_t st = nr.stream;
    const size_t plane = (size_t)nr.ph * nr.pw, hp = (size_t)nr.hh * nr.hw;
    int ps = (int)plane, rs = nr.pw, zero = 0, three = 3, four = 4, two = 2, lvl2 = 2, lvl4 = 4;
    int qw = nr.hw / 2, qh = nr.hh / 2, ew = nr.hw / 4, eh = nr.hh / 4;
    int mHalf = nr.gmMetricHalf ? 1 : 0;
    float s05 = 0.5f, s025 = 0.25f, s1 = 1.0f;
    nr.gmProfTween = false;
    // the CPU cost of enqueueing this pair block (QPC, 100 ns units), taken only
    // on a pair that really runs the flow so the average is over comparable groups
    const bool profPair = nr.gmProf && needFlow;
    const int64_t cpu0 = profPair ? nowQpc100() : 0;
    // the backbone maps in batch slot 1 belong to this call's previous frame only when the last call was a pair
    const bool mapsPrev = nr.gmMapsNext;
    nr.gmMapsNext = false;
    if (profPair)
        cudaEventRecord(nr.gmEv[0], st);
    // feat_ext of the NEW frame, into the set the NEXT pair will read as feat0. HDR planes (the kernels' comment): a
    // pair whose two frames stay inside the SDR range (k_hdrRange, read back here) runs every net on the SDR view
    // (hdrToModel, knee 1, head 0: an SDR source's values) through one scratch frame, its tweens come back through
    // k_hdrFromSdr; any other pair runs on the pictures with gmflow and metricnet on the motion view (kMotionHead's
    // roll-off). A previous frame's features taken in the other mode are taken again in this pair's.
    nr.gmCur ^= 1;
    void** fs = nr.dGmFeat[nr.gmCur];
    bool sdr = false;
    if (nr.encPost && nr.gmSdrPairs)
    {
        float tol = 2e-3f;
        void* ar[] = {(void*)&dCur, &ps, &rs, &nr.pw, &nr.ph, &nr.encPost, &nr.encWhite, &tol, (void*)&nr.dGmRange};
        if (cudaMemsetAsync(nr.dGmRange, 0, sizeof(int), st) != cudaSuccess ||
            cuLaunchKernel(nr.fHdrRange, (nr.pw + 15) / 16, (nr.ph + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, ar,
                           nullptr) != CUDA_SUCCESS ||
            cudaMemcpyAsync(nr.hGmRange, nr.dGmRange, sizeof(int), cudaMemcpyDeviceToHost, st) != cudaSuccess ||
            cudaStreamSynchronize(st) != cudaSuccess)
        {
            nr.die("gmfss range test failed");
            return false;
        }
        nr.gmFrameSdr[nr.gmCur] = *nr.hGmRange == 0;
        sdr = nr.gmFrameSdr[nr.gmCur] && nr.gmFrameSdr[nr.gmCur ^ 1];
        if (needFlow && nr.gmModeLogs < 4 && (nr.gmModeLogs == 0 || sdr != nr.gmPairSdr))
        {
            LOG("native: gmfss: %s\n", sdr ? "frame pairs inside the SDR range, every net on their SDR view"
                                           : "a frame pair outside the SDR range, the motion nets on the motion view");
            nr.gmModeLogs++;
        }
    }
    nr.gmPairSdr = sdr;
    // one frame's SDR view into the scratch (head 0 for an SDR pair, kMotionHead for the motion nets)
    auto view = [&](float* src, float head) {
        return nativeHdrEnc(nr, src, 0, ps, rs, nr.pw, nr.ph, nr.dGmEnc, 0, ps, rs, nr.encPost, kMotionKnee, head);
    };
    auto featExt = [&](float* x, void** f) { return nativeFnRun(nr, x, ps, rs, f, st); };
    float* xCur = sdr ? nr.dGmEnc : dCur;
    if (!needFlow && nr.gmFeatLazy)
    {
        // no tween in this group: its frame's features wait for the next pair that makes tweens, which takes them
        // from dPrev (the same bytes) in its own mode; gmFeatSdr keeps the mode the eager route would have used
        nr.gmFeatOk[nr.gmCur] = false;
        nr.gmFeatSdr[nr.gmCur] = sdr;
        nr.gmFeatSkips++;
        return true;
    }
    if (sdr && !view(dCur, 0.0f))
    {
        nr.die("gmfss SDR view launch failed");
        return false;
    }
    if (!featExt(xCur, fs))
    {
        nr.die("gmfss feat_ext launch failed");
        return false;
    }
    nr.gmFeatSdr[nr.gmCur] = sdr;
    nr.gmFeatOk[nr.gmCur] = true;
    if (!needFlow)
        return true; // no tween in this group: the pair-level work is skipped
    if (profPair)
        cudaEventRecord(nr.gmEv[1], st);
    // both halves into one (6, hh, hw) buffer: the ifnet's x as it is, gmflow's and
    // metricnet's inputs and the two image splats by pointer offset (an SDR pair: the new frame's view first, then
    // the previous frame's takes the scratch for its half and, when they came from the pictures, its features)
    float* h0 = nr.dGmHalf;
    float* h1 = nr.dGmHalf + 3 * hp;
    float* x0 = sdr ? nr.dGmEnc : dPrev;
    void* a0[] = {(void*)&x0, &zero, &ps, &rs, &three, &nr.hw, &nr.hh, (void*)&h0, &zero};
    void* a1[] = {(void*)&xCur, &zero, &ps, &rs, &three, &nr.hw, &nr.hh, (void*)&h1, &zero};
    auto halfOf = [&](void** a) {
        return cuLaunchKernel(nr.fHalf, (nr.hw + 15) / 16, (nr.hh + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a,
                              nullptr) == CUDA_SUCCESS;
    };
    // the previous frame's features again: taken in the other mode, or skipped by its call (a catch-up, timed apart
    // live when the eager route would not have taken them again)
    const bool stale = !nr.gmFeatOk[nr.gmCur ^ 1];
    const bool modeAgain = nr.gmFeatSdr[nr.gmCur ^ 1] != sdr;
    const bool refeat = stale || modeAgain;
    const bool catchUp = stale && !modeAgain && nr.gmCatchEv[0];
    if (!halfOf(a1) || (sdr && !view(dPrev, 0.0f)) || !halfOf(a0) ||
        (catchUp && cudaEventRecord(nr.gmCatchEv[0], st) != cudaSuccess) ||
        (refeat && !featExt(x0, nr.dGmFeat[nr.gmCur ^ 1])) ||
        (catchUp && cudaEventRecord(nr.gmCatchEv[1], st) != cudaSuccess))
    {
        nr.die("gmfss half launch failed");
        return false;
    }
    nr.gmFeatSdr[nr.gmCur ^ 1] = sdr;
    nr.gmFeatOk[nr.gmCur ^ 1] = true;
    nr.gmCatchUp = catchUp;
    if (stale)
        nr.gmCatchUps++;
    // the motion nets: an SDR pair's halves, else (HDR planes) the motion view's
    const bool mview = nr.encPost && !sdr;
    float* mv0 = mview ? nr.dGmHalfM : h0;
    float* mv1 = mview ? nr.dGmHalfM + 3 * hp : h1;
    float* e = nr.dGmEnc;
    void* am0[] = {(void*)&e, &zero, &ps, &rs, &three, &nr.hw, &nr.hh, (void*)&mv0, &zero};
    void* am1[] = {(void*)&e, &zero, &ps, &rs, &three, &nr.hw, &nr.hh, (void*)&mv1, &zero};
    if (mview && (!view(dCur, kMotionHead) || !halfOf(am1) || !view(dPrev, kMotionHead) || !halfOf(am0)))
    {
        nr.die("gmfss motion halves launch failed");
        return false;
    }
    if (profPair)
        cudaEventRecord(nr.gmEv[2], st);
    // gmflow's backbone once a frame: the previous frame's maps are the last pair's new ones (batch slot 1 -> 0) when
    // that pair took them on the same view of it (the pictures, the SDR view or the motion view), else taken again;
    // the new frame's into slot 1
    const int kind = mview ? 2 : (sdr ? 1 : 0);
    auto backbone = [&](float* x, int slot) {
        return nativeBbRun(nr, x, (uint8_t*)nr.dGmMap[0] + slot * nr.gmMapSlot[0],
                           (uint8_t*)nr.dGmMap[1] + slot * nr.gmMapSlot[1], st);
    };
    bool mapsOk = true;
    if (mapsPrev && nr.gmMapsKind == kind)
        for (int i = 0; i < 2 && mapsOk; i++)
            mapsOk = cudaMemcpyAsync(nr.dGmMap[i], (uint8_t*)nr.dGmMap[i] + nr.gmMapSlot[i], nr.gmMapSlot[i],
                                     cudaMemcpyDeviceToDevice, st) == cudaSuccess;
    else
        mapsOk = backbone(mv0, 0);
    if (!mapsOk || !backbone(mv1, 1))
    {
        nr.die("gmfss gmflow backbone's kernels (or the maps' copy) failed to launch");
        return false;
    }
    nr.gmMapsKind = kind;
    // the fused bidir GMFlow (both directions: `_m` on both frames' maps, the scale-0 global matching and its propagation
    // (k_attn2 twice: the matching with the pixel grid as its value writes the flow, the propagation reads it), A, the
    // local correlation (k_localCorr: one warp a pixel, 4 a block) and B) and metricnet on the same halves
    const int64_t cpuF0 = profPair ? nowQpc100() : 0;
    if (!nativeGaRun(nr, 0, nr.dGmMap[1], nullptr, nr.dGmM[0], nullptr, nr.dGmM[1], nr.dGmM[2], st))
    {
        nr.die("gmfss gmflow _m's kernels failed to launch");
        return false;
    }
    {
        float scale2 = 1.44269504f / 11.3137085f; // log2 e / sqrt(128): the softmax in base 2
        int n8 = nr.gmN8, kswapMatch = 1, kswapProp = 0, modeMatch = 3, modeProp = nr.gmPropHalf ? 4 : 0;
        const float* noV = nullptr;
        void* am[] = {(void*)&nr.dGmM[0],
                      (void*)&nr.dGmM[0],
                      (void*)&noV,
                      (void*)&nr.dGmFlow8,
                      &n8,
                      &kswapMatch,
                      &scale2,
                      &modeMatch,
                      &nr.gmW8};
        void* ap[] = {(void*)&nr.dGmM[1],
                      (void*)&nr.dGmM[2],
                      (void*)&nr.dGmFlow8,
                      (void*)&nr.dGmProp,
                      &n8,
                      &kswapProp,
                      &scale2,
                      &modeProp,
                      &nr.gmW8};
        const unsigned gx = (unsigned)((n8 + 63) / 64);
        const CUfunction fa = nr.fAttn2;
        if (cuLaunchKernel(fa, gx, 2, 1, 128, 1, 1, 0, (CUstream)st, am, nullptr) != CUDA_SUCCESS ||
            cuLaunchKernel(fa, gx, 2, 1, 128, 1, 1, 0, (CUstream)st, ap, nullptr) != CUDA_SUCCESS)
        {
            nr.die("gmfss global matching launch failed");
            return false;
        }
    }
    if (!nativeGaRun(nr, 1, nr.dGmMap[0], nr.dGmProp, nr.dGmA[0], nr.dGmA[1], nullptr, nullptr, st))
    {
        nr.die("gmfss gmflow A's kernels failed to launch");
        return false;
    }
    int nb = 2, hw4 = nr.gmH * nr.gmW;
    void* ac[] = {(void*)&nr.dGmA[0], (void*)&nr.dGmCorr, &nb, &nr.gmH, &nr.gmW, &hw4, &nr.gmK};
    if (cuLaunchKernel(nr.fLocalCorr, (unsigned)((2LL * hw4 * 32 + 127) / 128), 1, 1, 128, 1, 1, 0, (CUstream)st, ac,
                       nullptr) != CUDA_SUCCESS)
    {
        nr.die("gmfss local correlation launch failed");
        return false;
    }
    if (!nativeGbRun(nr, st))
    {
        nr.die("gmfss gmflow B's kernels failed to launch");
        return false;
    }
    nr.gmMapsNext = true; // slot 1 = the new frame's maps = the next pair's previous frame
    if (profPair)
        nr.gmCpuFlow += (double)(nowQpc100() - cpuF0) / 10000.0;
    if (profPair)
        cudaEventRecord(nr.gmEv[3], st);
    float* f10 = nr.dGmFlow + 2 * hp;
    void* m1 = (uint8_t*)nr.dGmMetric + hp * (nr.gmMetricHalf ? 2 : 4);
    if (!nativeMnRun(nr, mv0, mv1, nr.dGmFlow, f10, nr.dGmMetric, m1, st))
    {
        nr.die("gmfss metricnet's kernels failed to launch");
        return false;
    }
    if (profPair)
        cudaEventRecord(nr.gmEv[4], st);
    // the pyramids, once per pair: both flow directions in one launch (gmflow writes them as
    // one (2, 2, hh, hw) buffer) and both metrics in one (their planes are adjacent by the
    // addresses above). The flow carries the level's 0.5 / 0.25, a metric is only resampled.
    void* p0[] = {(void*)&nr.dGmFlow, &zero, &four, &nr.hw, &nr.hh, &lvl2, &s05, (void*)&nr.dGmFlowP[0], &qw, &qh};
    void* p1[] = {(void*)&nr.dGmFlow, &zero, &four, &nr.hw, &nr.hh, &lvl4, &s025, (void*)&nr.dGmFlowP[1], &ew, &eh};
    void* p2[] = {(void*)&nr.dGmMetric, &mHalf, &two, &nr.hw, &nr.hh, &lvl2, &s1, (void*)&nr.dGmMetP[0], &qw, &qh};
    void* p3[] = {(void*)&nr.dGmMetric, &mHalf, &two, &nr.hw, &nr.hh, &lvl4, &s1, (void*)&nr.dGmMetP[1], &ew, &eh};
    const int pw2[4] = {qw, ew, qw, ew}, ph2[4] = {qh, eh, qh, eh};
    void** pa[4] = {p0, p1, p2, p3};
    for (int i = 0; i < 4; i++)
        if (cuLaunchKernel(nr.fPyr, (pw2[i] + 15) / 16, (ph2[i] + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, pa[i],
                           nullptr) != CUDA_SUCCESS)
        {
            nr.die("gmfss pyramid launch failed");
            return false;
        }
    if (profPair)
    {
        cudaEventRecord(nr.gmEv[5], st);
        nr.gmCpuPair += (double)(nowQpc100() - cpu0) / 10000.0;
        nr.gmCpuPairN++;
    }
    return true;
}

// one tween at timestep t, into nr.dGmF (the clamped fp32 model frame storeSlot consumes).
// The author's inference(): I1t, I2t, the ifnet, the six feature splats, fusionnet, clamp.
static bool nativeGmfssTween(NativeRife& nr, float t)
{
    cudaStream_t st = nr.stream;
    const size_t plane = (size_t)nr.ph * nr.pw, hp = (size_t)nr.hh * nr.hw, qp = hp / 4, ep = hp / 16;
    const size_t me = nr.gmMetricHalf ? 2 : 4;
    // the engines' metric keeps their dtype, a pyramid level is always fp32 (k_pyr's output)
    const int mHalf = nr.gmMetricHalf ? 1 : 0, fHalf = nr.gmFeatHalf ? 1 : 0;
    float sA = t, sB = 1.0f - t;
    // one softsplat into channels co .. co + C of the fusionnet input `dst` (dc channels; a channel concat = this splat's
    // own channel range, so no copy follows); hwc = `in` and `dst` channels-last fp16 (nr.gmFeatHwc), else planar.
    // Gather form (nr.gmGather): list set ls built for (flow, metric, s) when `fresh` (k_gsCount, the three scans,
    // k_gsFill), then k_gsGather / k_gsGatherNhwc splats `in` through it; a level's image and feature splats share their
    // direction's set. Scatter form: zero the accumulator, scatter every source pixel, normalize.
    auto splat = [&](int ls, bool fresh, void* in, int inHalf, int C, float* flow, void* metric, int metricHalf,
                     float s, int w, int h, void* dst, int dstHalf, int dc, int co, bool hwc, float lr) -> bool {
        int cc = C, ww = w, hh2 = h, ih = inHalf, mh = metricHalf, dh = dstHalf, dcc = dc, dco = co;
        float slope = lr; // the target's LeakyRelu (fusionnet's b / c / d without theirs), -1 = none
        const int n = w * h;
        int ni = n;
        // planar: channel c of cell i at c * n + i of the range's start; channels-last: c + i * dc after co
        void* dstAt = (uint8_t*)dst + (size_t)co * (hwc ? 1 : (size_t)n) * (dstHalf ? 2 : 4);
        int isc = hwc ? 1 : n, isp = hwc ? C : 1, dsc = hwc ? 1 : n, dsp = hwc ? dc : 1;
        if (nr.gmGather)
        {
            int nb = (n + 2047) / 2048;
            const unsigned gx = (unsigned)(w + 31) / 32, gy = (unsigned)(h + 7) / 8;
            void* k0[] = {(void*)&flow, &metric, &mh, &s, &ww, &hh2, (void*)&nr.dGsCnt[ls], (void*)&nr.dGsSrc[ls]};
            void* k1[] = {(void*)&nr.dGsCnt[ls], &ni, (void*)&nr.dGsOff[ls], (void*)&nr.dGsBlk[ls]};
            void* k2[] = {(void*)&nr.dGsBlk[ls], &nb, (void*)&nr.dGsOff[ls], &ni};
            void* k3[] = {(void*)&nr.dGsOff[ls], (void*)&nr.dGsBlk[ls], &ni, (void*)&nr.dGsCnt[ls]};
            void* k4[] = {(void*)&flow, &s, &ww, &hh2, (void*)&nr.dGsCnt[ls], (void*)&nr.dGsList[ls]};
            void* k5[] = {
                &in,    &ih, &cc,   (void*)&nr.dGsSrc[ls], &ww, &hh2, (void*)&nr.dGsOff[ls], (void*)&nr.dGsList[ls],
                &dstAt, &dh, &slope};
            void* k5n[] = {&in,
                           &cc,
                           (void*)&nr.dGsSrc[ls],
                           &ww,
                           &hh2,
                           (void*)&nr.dGsOff[ls],
                           (void*)&nr.dGsList[ls],
                           &dst,
                           &dcc,
                           &dco,
                           &slope};
            const CUstream cs = (CUstream)st;
            // k_gsGatherNhwc: a cell's C / 8 channel groups on consecutive lanes, 256 / (C / 8) cells a block
            const unsigned ng = (unsigned)C / 8, nc = 256 / ng;
            const bool built =
                !fresh ||
                (cudaMemsetAsync(nr.dGsCnt[ls], 0, (size_t)n * sizeof(int), st) == cudaSuccess &&
                 cuLaunchKernel(nr.fGsCount, gx, gy, 1, 32, 8, 1, 0, cs, k0, nullptr) == CUDA_SUCCESS &&
                 cuLaunchKernel(nr.fGsScanA, (unsigned)nb, 1, 1, 1024, 1, 1, 0, cs, k1, nullptr) == CUDA_SUCCESS &&
                 cuLaunchKernel(nr.fGsScanB, 1, 1, 1, 1024, 1, 1, 0, cs, k2, nullptr) == CUDA_SUCCESS &&
                 cuLaunchKernel(nr.fGsScanC, (unsigned)(n + 255) / 256, 1, 1, 256, 1, 1, 0, cs, k3, nullptr) ==
                     CUDA_SUCCESS &&
                 cuLaunchKernel(nr.fGsFill, gx, gy, 1, 32, 8, 1, 0, cs, k4, nullptr) == CUDA_SUCCESS);
            if (!built || (hwc ? cuLaunchKernel(nr.fGsGatherNhwc, ((unsigned)n + nc - 1) / nc, 1, 1, ng, nc, 1, 0, cs,
                                                k5n, nullptr)
                               : cuLaunchKernel(nr.fGsGather, (unsigned)(n + 255) / 256, (unsigned)(C + 7) / 8, 1, 256,
                                                1, 1, 0, cs, k5, nullptr)) != CUDA_SUCCESS)
            {
                nr.die("gmfss gather splat launch failed");
                return false;
            }
            return true;
        }
        if (cudaMemsetAsync(nr.dGmAcc, 0, (size_t)(C + 1) * (size_t)n * sizeof(long long), st) != cudaSuccess)
        {
            nr.die("gmfss accumulator clear failed");
            return false;
        }
        void* a[] = {&in, &ih, &cc, &isc, &isp, (void*)&flow, &metric, &mh, &s, &ww, &hh2, (void*)&nr.dGmAcc};
        // 32 x 8: one warp = 32 consecutive x of a row, which k_splatSoft's neighbour combine
        // pairs (any shape of <= 256 threads, a multiple of 32, gives the same accumulators)
        if (cuLaunchKernel(nr.fSplatSoft, (w + 31) / 32, (h + 7) / 8, 1, 32, 8, 1, 0, (CUstream)st, a, nullptr) !=
            CUDA_SUCCESS)
        {
            nr.die("gmfss splatSoft launch failed");
            return false;
        }
        void* b[] = {(void*)&nr.dGmAcc, &cc, &ni, &dstAt, &dh, &dsc, &dsp, &slope};
        if (cuLaunchKernel(nr.fSplatNorm, (ni + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, b, nullptr) !=
            CUDA_SUCCESS)
        {
            nr.die("gmfss splatNorm launch failed");
            return false;
        }
        return true;
    };
    void** f0 = nr.dGmFeat[nr.gmCur ^ 1]; // the PREV frame's features (the author's feat1x)
    void** f1 = nr.dGmFeat[nr.gmCur];     // the new frame's (feat2x)
    void* met0 = nr.dGmMetric;
    void* met1 = (uint8_t*)nr.dGmMetric + hp * me;
    float* flow0 = nr.dGmFlow;
    float* flow1 = nr.dGmFlow + 2 * hp;
    const bool prof = nr.gmProf && !nr.gmProfTween; // the group's first tween carries the profile
    // the CPU enqueue span of EVERY tween (the ~20 launches below), the number the
    // per-tween CUDA graph decision rests on
    const int64_t cpu0 = nr.gmProf ? nowQpc100() : 0;
    if (prof)
    {
        nr.gmProfTween = true;
        cudaEventRecord(nr.gmEv[6], st);
    }
    // I1t and I2t into the fusionnet's a planes 0..2 and 6..8 (the ifnet fills 3..5 below)
    if (!splat(0, true, nr.dGmHalf, 0, 3, flow0, met0, mHalf, sA, nr.hw, nr.hh, nr.dGmFa, 0, 9, 0, false, -1.0f))
        return false;
    if (!splat(1, true, nr.dGmHalf + 3 * hp, 0, 3, flow1, met1, mHalf, sB, nr.hw, nr.hh, nr.dGmFa, 0, 9, 6, false,
               -1.0f))
        return false;
    if (prof)
        cudaEventRecord(nr.gmEv[7], st);
    // the GMFSS IFNet, straight into a's middle three planes (the author's cat([I1t, rife, I2t]))
    unsigned int bits;
    memcpy(&bits, &t, 4);
    if (cuMemsetD32Async((CUdeviceptr)nr.dGmT, bits, 1, (CUstream)st) != CUDA_SUCCESS)
    {
        nr.die("gmfss timestep fill failed");
        return false;
    }
    if (!nativeGiRun(nr, nr.dGmHalf, nr.dGmT, nr.dGmFa + 3 * hp, st))
    {
        nr.die("gmfss ifnet's kernels failed to launch");
        return false;
    }
    if (prof)
        cudaEventRecord(nr.gmEv[8], st);
    // the three feature levels, both directions: the half from the engines' own flow and
    // metric, the quarter and the eighth from the pair's pyramids
    // b / c / d channels-last fp16 (fusionnet's layout): the second direction starts gmC channels in
    const int bHalf = nr.gmFuseHalf ? 1 : 0;
    const bool fx = nr.gmFeatHwc;
    const int c0 = nr.gmC[0], c1 = nr.gmC[1], c2 = nr.gmC[2];
    // the half reuses the image splats' list sets (the same flow, metric and s); the quarter and the eighth build theirs
    const float* lr = nr.gmLr;
    if (!splat(0, false, f0[0], fHalf, c0, flow0, met0, mHalf, sA, nr.hw, nr.hh, nr.dGmFb, bHalf, 2 * c0, 0, fx, lr[0]))
        return false;
    if (!splat(1, false, f1[0], fHalf, c0, flow1, met1, mHalf, sB, nr.hw, nr.hh, nr.dGmFb, bHalf, 2 * c0, c0, fx,
               lr[0]))
        return false;
    if (!splat(0, true, f0[1], fHalf, c1, nr.dGmFlowP[0], nr.dGmMetP[0], 0, sA, nr.hw / 2, nr.hh / 2, nr.dGmFc, bHalf,
               2 * c1, 0, fx, lr[1]))
        return false;
    if (!splat(1, true, f1[1], fHalf, c1, nr.dGmFlowP[0] + 2 * qp, nr.dGmMetP[0] + qp, 0, sB, nr.hw / 2, nr.hh / 2,
               nr.dGmFc, bHalf, 2 * c1, c1, fx, lr[1]))
        return false;
    if (!splat(0, true, f0[2], fHalf, c2, nr.dGmFlowP[1], nr.dGmMetP[1], 0, sA, nr.hw / 4, nr.hh / 4, nr.dGmFd, bHalf,
               2 * c2, 0, fx, lr[2]))
        return false;
    if (!splat(1, true, f1[2], fHalf, c2, nr.dGmFlowP[1] + 2 * ep, nr.dGmMetP[1] + ep, 0, sB, nr.hw / 4, nr.hh / 4,
               nr.dGmFd, bHalf, 2 * c2, c2, fx, lr[2]))
        return false;
    if (prof)
        cudaEventRecord(nr.gmEv[9], st);
    if (!nativeFuRun(nr, st))
    {
        nr.die("gmfss fusionnet's kernels failed to launch");
        return false;
    }
    const int ps = (int)plane;
    if (nr.gmTail)
    {
        // the cut graph's out (in the splat accumulator) -> k_fuTail into dGmF: the frame clamped (the author's
        // torch.clamp(out, 0, 1)), or for an SDR pair unclamped, which k_hdrFromSdr then clamps and takes back to the
        // HDR codes in place (each pixel's three planes are read before it writes them)
        const void* pre = nr.dGmAcc;
        void* dst = nr.dGmF;
        int th = nr.hh, tw = nr.hw, cl = nr.gmPairSdr ? 0 : 1;
        void* at[] = {(void*)&pre, (void*)&nr.dGmWl, (void*)&nr.dGmBl, (void*)&dst, &th, &tw, &cl};
        // k_fuTailNhwc's shared tile is sized for these 16 x 8 threads
        if (cuLaunchKernel(nr.gmTailHwc ? nr.fFuTailNhwc : nr.fFuTail, (unsigned)((tw / 2 + 15) / 16),
                           (unsigned)((th / 2 + 7) / 8), 1, 16, 8, 1, 0, (CUstream)st, at, nullptr) != CUDA_SUCCESS ||
            (nr.gmPairSdr &&
             !nativeHdrFromSdr(nr, nr.dGmF, 0, ps, nr.pw, nr.pw, nr.ph, nr.dGmF, ps, nr.pw, nr.encPost)))
        {
            nr.die("gmfss fusion tail launch failed");
            return false;
        }
    }
    else
    {
        // the author's torch.clamp(out, 0, 1) on the fp16 output: k_restToF is exactly that; an SDR pair's tween
        // (the SDR view) goes back to the HDR codes in the same step (k_hdrFromSdr)
        int n = (int)(3 * plane), half = nr.gmOutHalf ? 1 : 0;
        void* ac[] = {(void*)&nr.dGmOut, &half, &n, (void*)&nr.dGmF};
        if (nr.gmPairSdr
                ? !nativeHdrFromSdr(nr, nr.dGmOut, half, ps, nr.pw, nr.pw, nr.ph, nr.dGmF, ps, nr.pw, nr.encPost)
                : cuLaunchKernel(nr.fRestToF, (n + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, ac, nullptr) !=
                      CUDA_SUCCESS)
        {
            nr.die("gmfss clamp launch failed");
            return false;
        }
    }
    if (prof)
        cudaEventRecord(nr.gmEv[10], st);
    if (nr.gmProf)
    {
        nr.gmCpuTween += (double)(nowQpc100() - cpu0) / 10000.0;
        nr.gmCpuTweenN++;
    }
    return true;
}

// SMV_LIVE_GMFSS_PROF=1: the nine spans of the group just finished (the stream is synced by the
// caller's final drain), averaged and printed every 32 groups, plus the CPU enqueue averages
// (the pair block, the gmflow enqueue inside it, and one tween)
static void nativeGmfssProfile(NativeRife& nr)
{
    static const int kSpan[9][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {6, 7}, {7, 8}, {8, 9}, {9, 10}};
    for (int i = 0; i < 9; i++)
    {
        float ms = 0.0f;
        if (cudaEventElapsedTime(&ms, nr.gmEv[kSpan[i][0]], nr.gmEv[kSpan[i][1]]) == cudaSuccess)
            nr.gmAcc[i] += ms;
    }
    if (++nr.gmProfN % 32)
        return;
    const double n = 32.0;
    LOG("[gmfss] feat_ext %.2f | half %.2f | flow %.2f | metric %.2f | pyr %.2f || per tween: "
        "image splats %.2f | ifnet %.2f | feature splats %.2f | fusionnet+clamp %.2f ms "
        "(avg over %u groups)\n",
        nr.gmAcc[0] / n, nr.gmAcc[1] / n, nr.gmAcc[2] / n, nr.gmAcc[3] / n, nr.gmAcc[4] / n, nr.gmAcc[5] / n,
        nr.gmAcc[6] / n, nr.gmAcc[7] / n, nr.gmAcc[8] / n, nr.gmProfN);
    LOG("[gmfss-cpu] enqueue: pair block %.3f (gmflow %.3f) over %u pairs | tween %.3f ms over "
        "%u tweens\n",
        nr.gmCpuPair / (double)(nr.gmCpuPairN ? nr.gmCpuPairN : 1),
        nr.gmCpuFlow / (double)(nr.gmCpuPairN ? nr.gmCpuPairN : 1), nr.gmCpuPairN,
        nr.gmCpuTween / (double)(nr.gmCpuTweenN ? nr.gmCpuTweenN : 1), nr.gmCpuTweenN);
    for (double& a : nr.gmAcc)
        a = 0.0;
    nr.gmCpuPair = nr.gmCpuFlow = nr.gmCpuTween = 0.0;
    nr.gmCpuPairN = nr.gmCpuTweenN = 0;
}
