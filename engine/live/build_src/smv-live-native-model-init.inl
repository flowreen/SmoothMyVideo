// part of smv-live-native.inl (included there, in place); not a standalone translation unit
static bool nativeGmfssSetup(NativeRife& nr)
{
    auto bad = [](const char* what) {
        LOG("native: gmfss chain: %s\n", what);
        return false;
    };
    if (nr.hh <= 0 || nr.hw <= 0 || (nr.hh & 3) || (nr.hw & 3))
        return bad("the half frame is not a multiple of 4");
    if (nr.hh * 2 != nr.ph || nr.hw * 2 != nr.pw)
        return bad("the half frame is not half of the padded frame");
    if (nr.ph < kGmMinPad || nr.pw < kGmMinPad)
        return bad("the padded frame is below 128 a side (gmflow's quarter-scale windows need 2 tokens a side)");
    const int hh = nr.hh, hw = nr.hw;
    // feat_ext on the host's kernels: the padded frame in, fp16 levels of 64 / 128 / 192 channels out
    if (nr.ph % 64 || nr.pw % 64)
        return bad("the padded frame is not a multiple of 64 (feat_ext's three stride-2 convs and the tiles)");
    nr.gmC[0] = kFnCh[1];
    nr.gmC[1] = kFnCh[2];
    nr.gmC[2] = kFnCh[3];
    nr.gmFeatHalf = true;
    // the fused bidir GMFlow at the half, all on the host's kernels; B hands the flow out: row 0 = flow01, row 1 = flow10
    // (so the pyramids take both at once)
    if (!nativeGmSplitCheck(nr))
        return false;
    // metricnet on the host's kernels: the motion halves and both flows in, m0 / m1 fp16 out
    nr.gmMetricHalf = true;
    // the GMFSS IFNet on the host's kernels (the half = the /64 pad / 2, so its 1 / 32 level is whole): the 6-plane half
    // buffer and the fp32 timestep in, `rife` (3 fp32 planes) out; its weights and module are loaded below
    // fusionnet on the host's kernels (its module and weights are loaded below): the concats are adjacent planes, so b /
    // c / d are exactly 2x the feature levels, written channels-last fp16 by the feature splats (k_gsGatherNhwc, the
    // heads' first PReLU in their stores) beside feat_ext's channels-last levels; its out = the 64 -> 256 conv before
    // the pixel shuffle, channels-last fp16, which k_fuTailNhwc shuffles and runs conv_last on (fp32 sums, the frame in
    // fp32, straight into dGmF)
    nr.gmFuseHalf = true;
    nr.gmFeatHwc = true;
    nr.gmTail = true;
    nr.gmTailHwc = true;
    nr.gmOutHalf = false;

    const size_t hp = (size_t)hh * hw, qp = hp / 4, ep = hp / 16;
    const size_t plane = (size_t)nr.ph * nr.pw;
    const size_t fe = nr.gmFeatHalf ? 2 : 4, me = nr.gmMetricHalf ? 2 : 4, oe = nr.gmOutHalf ? 2 : 4,
                 be = nr.gmFuseHalf ? 2 : 4;
    size_t accPlanes = (size_t)nr.gmC[0] + 1;
    if (((size_t)nr.gmC[1] + 1) * qp > accPlanes * hp)
        accPlanes = (((size_t)nr.gmC[1] + 1) * qp + hp - 1) / hp;
    if (((size_t)nr.gmC[2] + 1) * ep > accPlanes * hp)
        accPlanes = (((size_t)nr.gmC[2] + 1) * ep + hp - 1) / hp;
    nr.gmGather = lkEnv("SMV_GM_SPLAT") != "old" && hp <= (size_t)2048 * 4096;
    // the gather splats never touch the accumulator: it then holds only the cut fusionnet's out (256 fp16 a half
    // pixel = 64 planes of 8 bytes), and nothing with the full graph
    if (nr.gmGather)
        accPlanes = nr.gmTail ? 256 * 2 / sizeof(long long) : 0;
    if (nr.gmTail && accPlanes * sizeof(long long) < 256 * 2)
        return bad("the splat accumulator cannot hold the cut fusionnet's out");
    size_t freeB = 0, freeA = 0, totB = 0;
    cudaMemGetInfo(&freeB, &totB);
    for (int s = 0; s < 2; s++)
        for (int l = 0; l < 3; l++)
            NCHK(cudaMalloc(&nr.dGmFeat[s][l], (size_t)nr.gmC[l] * (l == 0 ? hp : (l == 1 ? qp : ep)) * fe),
                 "alloc gmfss features");
    // feat_ext's module and weights, its scratch in the shared activation block when that holds it
    {
        std::string why;
        if (!nativeFnModule(nr, why))
        {
            LOG("native: gmfss chain: feat_ext unavailable (%s)\n", why.c_str());
            return false;
        }
        const size_t fb = fnNatBytes(nr, nr.gmFeatHwc);
        if (nr.dGmCtxMem && (size_t)nr.gmCtxMemB >= fb)
            nr.dFnScr = nr.dGmCtxMem;
        else
        {
            NCHK(cudaMalloc(&nr.dFnOwn, fb), "alloc gmfss feat_ext scratch");
            nr.dFnScr = nr.dFnOwn;
        }
        LOG("native: gmfss feat_ext on the host's kernels (fp16 sums, %s levels; its %.0f MB of activations %s)\n",
            nr.gmFeatHwc ? "channels-last" : "planar", fb / 1048576.0,
            nr.dFnOwn ? "on their own" : "in the shared activation block");
    }
    // metricnet's module and weights, its scratch in the shared activation block when that holds it
    {
        std::string why;
        if (!nativeMnModule(nr, why))
        {
            LOG("native: gmfss chain: metricnet unavailable (%s)\n", why.c_str());
            return false;
        }
        const size_t mb = mnNatBytes(nr);
        if (!(nr.dGmCtxMem && (size_t)nr.gmCtxMemB >= mb))
            NCHK(cudaMalloc(&nr.dMnOwn, mb), "alloc gmfss metricnet scratch");
        LOG("native: gmfss metricnet on the host's kernels (fp16 sums; its %.0f MB of activations %s)\n",
            mb / 1048576.0, nr.dMnOwn ? "on their own" : "in the shared activation block");
    }
    // the GMFSS IFNet's module and weights, its scratch in the shared activation block when that holds it
    {
        std::string why;
        if (!nativeGiModule(nr, why))
        {
            LOG("native: gmfss chain: the GMFSS IFNet unavailable (%s)\n", why.c_str());
            return false;
        }
        const size_t gb = giNatBytes(nr);
        if (!(nr.dGmCtxMem && (size_t)nr.gmCtxMemB >= gb))
            NCHK(cudaMalloc(&nr.dGiOwn, gb), "alloc gmfss ifnet scratch");
        LOG("native: gmfss ifnet on the host's kernels (fp32 sums; its %.0f MB of activations %s)\n", gb / 1048576.0,
            nr.dGiOwn ? "on their own" : "in the shared activation block");
        nr.giGraphs.mode = chainGraphMode("SMV_GI_GRAPH");
    }
    // fusionnet's module, weights and conv_last, its activations in the shared activation block when that holds them
    {
        std::string why;
        if (!nativeFuModule(nr, why))
        {
            LOG("native: gmfss chain: fusionnet unavailable (%s)\n", why.c_str());
            return false;
        }
        const size_t ub = fuNatBytes(nr);
        if (!(nr.dGmCtxMem && (size_t)nr.gmCtxMemB >= ub))
            NCHK(cudaMalloc(&nr.dFuOwn, ub), "alloc gmfss fusionnet activations");
        LOG("native: gmfss fusionnet on the host's kernels (fp16 sums, head0's first conv fp32; its %.0f MB of "
            "activations %s)\n",
            ub / 1048576.0, nr.dFuOwn ? "on their own" : "in the shared activation block");
        nr.fuGraphs.mode = chainGraphMode("SMV_FU_GRAPH");
    }
    // gmflow B's module and weights, its activations in the shared activation block when that holds them
    {
        std::string why;
        if (!nativeGbModule(nr, why))
        {
            LOG("native: gmfss chain: gmflow B unavailable (%s)\n", why.c_str());
            return false;
        }
        const size_t bb = gbNatBytes(nr);
        if (!(nr.dGmCtxMem && (size_t)nr.gmCtxMemB >= bb))
            NCHK(cudaMalloc(&nr.dGbOwn, bb), "alloc gmfss gmflow B activations");
        LOG("native: gmfss gmflow B on the host's kernels (fp16 sums in its 3 x 3 conv; its %.0f MB of activations %s)\n",
            bb / 1048576.0, nr.dGbOwn ? "on their own" : "in the shared activation block");
        nr.gbGraphs.mode = chainGraphMode("SMV_GB_GRAPH");
    }
    // gmflow backbone's module and weights, its activations in the shared activation block when that holds them
    {
        std::string why;
        if (!nativeBbModule(nr, why))
        {
            LOG("native: gmfss chain: gmflow's backbone unavailable (%s)\n", why.c_str());
            return false;
        }
        const size_t kb = bbNatBytes(nr);
        if (!(nr.dGmCtxMem && (size_t)nr.gmCtxMemB >= kb))
            NCHK(cudaMalloc(&nr.dBbOwn, kb), "alloc gmfss gmflow backbone activations");
        LOG("native: gmfss gmflow backbone on the host's kernels (fp16 sums in its 3 x 3 convs; its %.0f MB of "
            "activations %s)\n",
            kb / 1048576.0, nr.dBbOwn ? "on their own" : "in the shared activation block");
        nr.bbGraphs.mode = chainGraphMode("SMV_BB_GRAPH");
    }
    // gmflow's transformer (A and `_m`): its module, weights and window tables; its activations in the shared activation
    // block when that holds them; A takes the propagation's output in fp16
    {
        std::string why;
        if (!nr.gmPropHalf)
            why = "the propagated flow is not fp16";
        if (!nr.gmPropHalf || !nativeGaModule(nr, why))
        {
            LOG("native: gmfss chain: gmflow's transformer unavailable (%s)\n", why.c_str());
            return false;
        }
        const size_t ab = gaNatBytes(nr);
        if (!(nr.dGmCtxMem && (size_t)nr.gmCtxMemB >= ab))
            NCHK(cudaMalloc(&nr.dGaOwn, ab), "alloc gmfss gmflow transformer activations");
        LOG("native: gmfss gmflow A and _m on the host's kernels (fp16 sums but in P v'; %s; their %.0f MB of activations "
            "%s)\n",
            gaTuring()   ? "32-key tiles, a 2-stage weight ring"
            : nr.gaAttn2 ? "64-key tiles (A's attention 32-key, two CTAs an SM), a 3-stage weight ring"
                         : "64-key tiles, a 3-stage weight ring",
            ab / 1048576.0, nr.dGaOwn ? "on their own" : "in the shared activation block");
        nr.gaGraphs[0].mode = nr.gaGraphs[1].mode = chainGraphMode("SMV_GA_GRAPH");
    }
    NCHK(cudaMalloc((void**)&nr.dGmHalf, 6 * hp * sizeof(float)), "alloc gmfss halves");
    NCHK(cudaMalloc((void**)&nr.dGmFlow, 4 * hp * sizeof(float)), "alloc gmfss flow");
    NCHK(cudaMalloc(&nr.dGmMetric, 2 * hp * me), "alloc gmfss metric");
    NCHK(cudaMalloc((void**)&nr.dGmFlowP[0], 4 * qp * sizeof(float)), "alloc gmfss flow pyramid");
    NCHK(cudaMalloc((void**)&nr.dGmFlowP[1], 4 * ep * sizeof(float)), "alloc gmfss flow pyramid");
    NCHK(cudaMalloc((void**)&nr.dGmMetP[0], 2 * qp * sizeof(float)), "alloc gmfss metric pyramid");
    NCHK(cudaMalloc((void**)&nr.dGmMetP[1], 2 * ep * sizeof(float)), "alloc gmfss metric pyramid");
    if (accPlanes)
        NCHK(cudaMalloc((void**)&nr.dGmAcc, accPlanes * hp * sizeof(long long)), "alloc gmfss splat accumulator");
    if (nr.gmGather)
        for (int s = 0; s < 2; s++)
        {
            NCHK(cudaMalloc((void**)&nr.dGsCnt[s], hp * sizeof(int)), "alloc gmfss gather counts");
            NCHK(cudaMalloc((void**)&nr.dGsOff[s], (hp + 1) * sizeof(int)), "alloc gmfss gather offsets");
            NCHK(cudaMalloc((void**)&nr.dGsList[s], 4 * hp * sizeof(int)), "alloc gmfss gather lists");
            NCHK(cudaMalloc((void**)&nr.dGsSrc[s], 3 * hp * sizeof(float)), "alloc gmfss gather sources");
            NCHK(cudaMalloc((void**)&nr.dGsBlk[s], 4096 * sizeof(int)), "alloc gmfss gather block sums");
        }
    LOG("native: gmfss splats by %s\n", nr.gmGather ? "gather (lists)" : "scatter (atomics)");
    NCHK(cudaMalloc((void**)&nr.dGmFa, 9 * hp * sizeof(float)), "alloc gmfss fusion a");
    NCHK(cudaMalloc(&nr.dGmFb, (size_t)2 * nr.gmC[0] * hp * be), "alloc gmfss fusion b");
    NCHK(cudaMalloc(&nr.dGmFc, (size_t)2 * nr.gmC[1] * qp * be), "alloc gmfss fusion c");
    NCHK(cudaMalloc(&nr.dGmFd, (size_t)2 * nr.gmC[2] * ep * be), "alloc gmfss fusion d");
    NCHK(cudaMalloc((void**)&nr.dGmT, sizeof(float)), "alloc gmfss timestep");
    if (!nr.gmTail)
        NCHK(cudaMalloc(&nr.dGmOut, 3 * plane * oe), "alloc gmfss fusion out");
    NCHK(cudaMalloc((void**)&nr.dGmF, 3 * plane * sizeof(float)), "alloc gmfss model frame");
    // the pair's scratch (A's qf / flow0, the local correlation, the matching operands and flow, the propagated flow;
    // with encPost the SDR view and the motion halves) is written and read inside nativeGmfssPair only, and the
    // accumulator under the gather splats inside nativeGmfssTween only (the cut out, written before it is read), both
    // on nr.stream: so the scratch is carved from the accumulator when it fits (256-B aligned), else it gets its own
    // buffers (SMV_GM_SCRATCH=own)
    struct Scratch
    {
        void** p;
        size_t bytes;
        const char* what;
    };
    const size_t q4 = (size_t)nr.gmH * nr.gmW;
    std::vector<Scratch> scr;
    scr.push_back({&nr.dGmA[0], 4 * q4 * 128 * sizeof(uint16_t), "alloc gmfss split tensors"});
    scr.push_back({&nr.dGmA[1], 2 * 2 * q4 * sizeof(float), "alloc gmfss split tensors"});
    scr.push_back(
        {(void**)&nr.dGmCorr, (size_t)2 * nr.gmH * nr.gmW * nr.gmK * sizeof(float), "alloc gmfss local correlation"});
    for (int i = 0; i < 3; i++)
        scr.push_back({&nr.dGmM[i], (size_t)2 * nr.gmN8 * 128 * sizeof(uint16_t), "alloc gmfss matching operands"});
    scr.push_back({(void**)&nr.dGmFlow8, (size_t)2 * nr.gmN8 * 2 * sizeof(float), "alloc gmfss matching flow"});
    scr.push_back({&nr.dGmProp, (size_t)2 * nr.gmN8 * 2 * sizeof(uint16_t), "alloc gmfss propagated flow"});
    if (nr.encPost)
    {
        scr.push_back({(void**)&nr.dGmEnc, 3 * plane * sizeof(float), "alloc gmfss SDR view"});
        scr.push_back({(void**)&nr.dGmHalfM, 6 * hp * sizeof(float), "alloc gmfss motion halves"});
    }
    size_t scrB = 0;
    for (const Scratch& s : scr)
        scrB += (s.bytes + 255) & ~(size_t)255;
    nr.gmScratchInAcc =
        nr.gmTail && nr.gmGather && lkEnv("SMV_GM_SCRATCH") != "own" && scrB <= accPlanes * hp * sizeof(long long);
    size_t scrOff = 0;
    for (const Scratch& s : scr)
        if (nr.gmScratchInAcc)
        {
            *s.p = (uint8_t*)nr.dGmAcc + scrOff;
            scrOff += (s.bytes + 255) & ~(size_t)255;
        }
        else
            NCHK(cudaMalloc(s.p, s.bytes), s.what);
    LOG("native: gmfss chain: pair scratch %.0f MB %s\n", (double)scrB / 1048576.0,
        nr.gmScratchInAcc ? "inside the splat accumulator" : "in its own buffers");
    for (int i = 0; i < 2; i++)
    {
        // map4 / map8: both frames' [2, 128, H, W] fp16 at the quarter / the eighth
        const size_t bytes = (size_t)2 * 128 * (i ? q4 / 4 : q4) * sizeof(uint16_t);
        NCHK(cudaMalloc(&nr.dGmMap[i], bytes), "alloc gmfss backbone maps");
        nr.gmMapSlot[i] = bytes / 2;
    }
    nr.gmMapsNext = false;
    if (nr.encPost)
    {
        NCHK(cudaMalloc((void**)&nr.dGmRange, sizeof(int)), "alloc gmfss range flag");
        NCHK(cudaHostAlloc((void**)&nr.hGmRange, sizeof(int), cudaHostAllocDefault), "alloc gmfss range readback");
        nr.gmFrameSdr[0] = nr.gmFrameSdr[1] = nr.gmFeatSdr[0] = nr.gmFeatSdr[1] = nr.gmPairSdr = false;
        nr.gmModeLogs = 0;
    }
    cudaMemGetInfo(&freeA, &totB);
    LOG("native: gmfss chain: half %dx%d, features %d/%d/%d %s, metric %s, fusion features %s, out %s, accumulator"
        " %zu planes, buffers %.0f MB\n",
        hw, hh, nr.gmC[0], nr.gmC[1], nr.gmC[2], nr.gmFeatHalf ? "fp16" : "fp32", nr.gmMetricHalf ? "fp16" : "fp32",
        nr.gmFuseHalf ? "fp16" : "fp32", nr.gmOutHalf ? "fp16" : "fp32", accPlanes,
        ((double)freeB - (double)freeA) / 1048576.0);
    LOG("native: gmfss chain: fusion tail %s\n",
        !nr.gmTail     ? "in the engine"
        : nr.gmTailHwc ? "on the host (k_fuTailNhwc: pixel shuffle + conv_last, fp32; out channels-last)"
                       : "on the host (k_fuTail: pixel shuffle + conv_last, fp32)");
    LOG("native: gmfss chain: feature splats %s\n",
        nr.gmFeatHwc ? "channels-last (k_gsGatherNhwc: feat_ext's levels and fusionnet's b / c / d NHWC)" : "planar");
    LOG("native: gmfss chain: fusionnet's input LeakyRelu in the feature splats' stores (slopes %.9g / %.9g / %.9g)\n",
        nr.gmLr[0], nr.gmLr[1], nr.gmLr[2]);
    LOG("native: gmfss chain: gmflow split, its local correlation in one kernel (%dx%d, %d samples a pixel)\n", nr.gmW,
        nr.gmH, nr.gmK);
    LOG("native: gmfss chain: the global matching by k_attn2 (%d tokens a frame)\n", nr.gmN8);
    nr.gmProf = GetEnvironmentVariableW(L"SMV_LIVE_GMFSS_PROF", nullptr, 0) != 0;
    if (nr.gmProf)
        for (auto& e : nr.gmEv)
            NCHK(cudaEventCreate(&e), "create gmfss profile event");
    nr.gmFeatLazy = lkEnv("SMV_GM_FEAT_LAZY") != "0";
    nr.gmFeatOk[0] = nr.gmFeatOk[1] = nr.gmCatchUp = false;
    nr.gmFeatSkips = nr.gmCatchUps = 0;
    if (!g_offline && nr.gmFeatLazy)
        for (auto& e : nr.gmCatchEv)
            NCHK(cudaEventCreate(&e), "create gmfss catch-up event");
    LOG("native: gmfss chain: %s\n", nr.gmFeatLazy
                                         ? "feat_ext only for pairs that make tweens (the next such pair takes a "
                                           "skipped frame's)"
                                         : "feat_ext on every new frame (SMV_GM_FEAT_LAZY=0)");
    return true;
}

// fp32 -> fp16 bits rounded to nearest even, the value the graph's own Cast (and f2h's cvt.rn) gives:
// the constant timestep fills of an fp16 timestep (subnormal halves and the overflow to infinity included)
static uint16_t halfBitsRne(float f)
{
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t s = (x >> 16) & 0x8000u, a = x & 0x7fffffffu;
    if (a > 0x7f800000u)
        return (uint16_t)(s | 0x7e00u); // NaN
    if (a >= 0x477ff000u)
        return (uint16_t)(s | 0x7c00u); // 65520 and up round to infinity
    if (a < 0x38800000u)
    {
        // below 2^-14: a subnormal half, in units of 2^-24; up to 2^-25 rounds to zero (the tie to
        // the even zero)
        if (a <= 0x33000000u)
            return (uint16_t)s;
        const uint32_t m = 0x800000u | (a & 0x7fffffu), sh = 126u - (a >> 23);
        const uint32_t rem = m & ((1u << sh) - 1u), halfway = 1u << (sh - 1u);
        uint32_t h = m >> sh;
        if (rem > halfway || (rem == halfway && (h & 1u)))
            h++;
        return (uint16_t)(s | h);
    }
    // a normal half: the exponent rebiased (127 -> 15), the 13 dropped bits rounded to nearest even
    // (a carry out of the mantissa steps the exponent, which is the right result)
    const uint32_t r = a - 0x38000000u;
    return (uint16_t)(s | ((r + 0x0fffu + ((r >> 13) & 1u)) >> 13));
}

// timestep plane j of the IFNet's batch: a constant map of t in the engine's timestep dtype; a plane that already
// holds this t is left as it is (x2 asks for 0.5 every group)
static bool nativeFillT(NativeRife& nr, size_t j, float t, size_t plane, cudaStream_t st)
{
    unsigned int bits;
    memcpy(&bits, &t, 4);
    const bool track = j < 64;
    const uint64_t bit = track ? 1ull << j : 0;
    if (track && (nr.tHeldMask & bit) && nr.tHeld[j] == bits)
        return true;
    const bool ok = nr.tHalf ? cuMemsetD16Async((CUdeviceptr)((uint16_t*)nr.dT + j * plane), halfBitsRne(t), plane,
                                                (CUstream)st) == CUDA_SUCCESS
                             : cuMemsetD32Async((CUdeviceptr)((float*)nr.dT + j * plane), bits, plane, (CUstream)st) ==
                                   CUDA_SUCCESS;
    if (ok && track)
    {
        nr.tHeld[j] = bits;
        nr.tHeldMask |= bit;
    }
    else
        nr.tHeldMask &= ~bit;
    return ok;
}

// the IFNet's timestep, flow and mask at the model's frame (the timestep sized by the dtype read
// off the engine) and the tweens RIFE's last step writes at the pictures' layout (fp32, fp16 when outHalf): batchMax
// entries each
// (DRBA: one, every tween is its own enqueue with its own map)
static bool nativeTweenBuffers(NativeRife& nr)
{
    const size_t plane = (size_t)nr.ph * nr.pw, b = nr.drba ? 1 : (size_t)(nr.batchMax < 1 ? 1 : nr.batchMax);
    const size_t mplane = nr.mph ? (size_t)nr.mph * nr.mpw : plane;
    // the features in the IFNet's own dtype (f0 / f1, DRBA's encode ring), the encode's fp16 output only where it is
    // widened for an fp32 IFNet
    const size_t fb = nr.featHalf ? sizeof(uint16_t) : sizeof(float);
    if (nr.drba)
        for (int i = 0; i < 4; i++)
            NCHK(cudaMalloc((void**)&nr.dDrF[i], kRifeFeatC * mplane * fb), "alloc drba encode ring");
    else
    {
        NCHK(cudaMalloc((void**)&nr.dF[0], kRifeFeatC * mplane * fb), "alloc f0");
        NCHK(cudaMalloc((void**)&nr.dF[1], kRifeFeatC * mplane * fb), "alloc f1");
    }
    NCHK(cudaMalloc(&nr.dT, b * mplane * (nr.tHalf ? sizeof(uint16_t) : sizeof(float))), "alloc timestep");
    nr.tHeldMask = 0;
    NCHK(cudaMalloc((void**)&nr.dFlow, b * 4 * mplane * sizeof(uint16_t)), "alloc flow");
    NCHK(cudaMalloc((void**)&nr.dMask, b * mplane * sizeof(uint16_t)), "alloc mask");
    nr.blendOut = !g_offline && !nr.gmfss && nr.w == nr.dw && nr.h == nr.dh && !nr.uw && !drbaAuthorWanted(nr);
    if (nr.blendOut)
        LOG("native: RIFE's last step stores each tween straight into its slot (the %dx%d model frame is the "
            "presented one)\n",
            nr.w, nr.h);
    else
    {
        NCHK(cudaMalloc((void**)&nr.dMerged, b * 3 * plane * (nr.outHalf ? sizeof(uint16_t) : sizeof(float))),
             "alloc tweens");
        if (nr.outHalf)
            LOG("native: RIFE's tweens are stored in fp16 (SMV_TWEEN_FP16=1)\n");
    }
    return true;
}

// k_motionIn's HDR curve, in units of SDR white: up to the knee the light passes exactly (what an SDR
// source hands RIFE; the knee sits k_hdrRange's tolerance above white, where the decode of SDR white's
// 16-bit PQ code can land), above it PQ's spacing with 10000 nits at kMotionPqCap above the knee's
// sRGB, so every stop of highlight keeps a similar share of contrast while the values stay near the
// range RIFE was trained on (measured on exact pans of highlight-heavy pictures: a smaller cap squeezes
// the highlights' motion, a larger one takes RIFE out of its range; 0.75 measured best)
constexpr float kMotionPqKnee = 1.002f, kMotionPqCap = 0.75f;
// GMFSS's motion view on a frame pair outside the SDR range (hdrToModel): the SDR range exactly, the
// light above white rolled off toward 5 x white
constexpr float kMotionKnee = 1.0f, kMotionHead = 4.0f;

// GMFSS and Restore on HDR planes (hdrToModel, the kernels' comment): pre = the decoded frame's encoding, post = the
// model frames' (1 PQ, 2 HLG, 0 SDR planes), white = SDR white in scRGB units. SMV_HDR_MODEL_ENC=0 = both read the HDR
// codes as they are; SMV_GMFSS_SDR_PAIRS=0 = GMFSS treats every pair as one outside the SDR range.
static void nativeHdrModelSetup(NativeRife& nr, int pre, int post, float white)
{
    wchar_t v[8]{};
    const bool off = GetEnvironmentVariableW(L"SMV_HDR_MODEL_ENC", v, 8) && v[0] == L'0';
    nr.encPre = off ? 0 : pre;
    nr.encPost = off ? 0 : post;
    nr.encWhite = white;
    nr.gmSdrPairs = !(GetEnvironmentVariableW(L"SMV_GMFSS_SDR_PAIRS", v, 8) && v[0] == L'0');
    nr.shHdrView = !(GetEnvironmentVariableW(L"SMV_HDR_SHARPEN_VIEW", v, 8) && v[0] == L'0');
    nr.rsHdrView = !(GetEnvironmentVariableW(L"SMV_HDR_RESIZE_VIEW", v, 8) && v[0] == L'0');
    if (nr.encPre || nr.encPost)
        LOG("native: HDR planes: Restore reads their SDR view over %.0f nits and gets back the light outside it; %s\n",
            white * 80.0,
            nr.gmSdrPairs ? "GMFSS reads it on frame pairs inside the SDR range, its motion nets on any other pair"
                          : "GMFSS's motion nets read it on every pair (SMV_GMFSS_SDR_PAIRS=0)");
}

// one planar frame (w x h, strides in elements) into its SDR view (fp32, or fp16 when dstHalf; hdrToModel's knee and
// head given), and a GMFSS tween of an SDR pair back to the HDR codes, on the render stream; false = the launch failed
static bool nativeHdrEnc(NativeRife& nr, const void* src, int half, int sps, int srs, int w, int h, void* dst,
                         int dstHalf, int dps, int drs, int mode, float knee, float head)
{
    void* a[] = {(void*)&src, &half, &sps, &srs, &w, &h, &dst, &dstHalf, &dps, &drs, &mode, &nr.encWhite, &knee, &head};
    return cuLaunchKernel(nr.fHdrEnc, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)nr.stream, a, nullptr) ==
           CUDA_SUCCESS;
}
static bool nativeHdrFromSdr(NativeRife& nr, const void* src, int half, int sps, int srs, int w, int h, float* dst,
                             int dps, int drs, int mode)
{
    void* a[] = {(void*)&src, &half, &sps, &srs, &w, &h, (void*)&dst, &dps, &drs, &mode, &nr.encWhite};
    return cuLaunchKernel(nr.fHdrFromSdr, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)nr.stream, a,
                          nullptr) == CUDA_SUCCESS;
}
// a resize's source frame (HDR planes, fp32, w x h, planes ps apart, rows rs apart): 1 inside the SDR range (k_hdrRange,
// read back), 0 outside, -1 when a launch or the readback failed; the mode is logged on the first frame and on up to 3
// switches
static int nativeRsInRange(NativeRife& nr, const float* src, int ps, int rs, int w, int h, int mode)
{
    cudaStream_t st = nr.stream;
    float tol = 2e-3f;
    void* a[] = {(void*)&src, &ps, &rs, &w, &h, &mode, &nr.encWhite, &tol, (void*)&nr.dRsRange};
    if (cudaMemsetAsync(nr.dRsRange, 0, sizeof(int), st) != cudaSuccess ||
        cuLaunchKernel(nr.fHdrRange, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a, nullptr) !=
            CUDA_SUCCESS ||
        cudaMemcpyAsync(nr.hRsRange, nr.dRsRange, sizeof(int), cudaMemcpyDeviceToHost, st) != cudaSuccess ||
        cudaStreamSynchronize(st) != cudaSuccess)
        return -1;
    const bool in = *nr.hRsRange == 0;
    if (nr.rsModeLogs < 4 && (nr.rsModeLogs == 0 || in != nr.rsLastView))
    {
        LOG("native: resize: %s\n", in ? "frames inside the SDR range, resized as their SDR view"
                                       : "a frame outside the SDR range, resized on "
                                         "the codes");
        nr.rsModeLogs++;
    }
    nr.rsLastView = in;
    return in ? 1 : 0;
}
// one planar Lanczos resize: the separable pair, k_fitAaH into dTmp (3, sh, tw) then k_fitAaV (a size kept on
// both axes = an exact copy); dTmp holds the session's largest such pass (tmpN floats: live nativeCudaInitEarly,
// offline runOfflineSession). false = dTmp cannot hold the pass or a launch failed.
static bool nativeFitPlanar(NativeRife& nr, const void* src, int half, int ps, int rs, int sw, int sh, float* dst,
                            int tw, int th, cudaStream_t st)
{
    if (!nr.dTmp || (size_t)3 * sh * tw > nr.tmpN)
    {
        LOG("native: the resize pass buffer cannot hold %dx%d -> %dx%d\n", sw, sh, tw, th);
        return false;
    }
    const void* tabH = rsTabFor(nr, sw, tw, st);
    const void* tabV = rsTabFor(nr, sh, th, st);
    if (!tabH || !tabV)
    {
        LOG("native: the resize tap tables for %dx%d -> %dx%d failed\n", sw, sh, tw, th);
        return false;
    }
    void* ah[] = {(void*)&src, &half, &ps, &rs, &sw, &sh, &nr.dTmp, &tw, (void*)&tabH};
    void* av[] = {&nr.dTmp, &tw, &sh, &dst, &th, (void*)&tabV};
    return cuLaunchKernel(nr.fFitAaH, (tw + 15) / 16, (sh + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, ah, nullptr) ==
               CUDA_SUCCESS &&
           cuLaunchKernel(nr.fFitAaV, (tw + 15) / 16, (th + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, av, nullptr) ==
               CUDA_SUCCESS;
}

// the resize of an SDR-range frame: its SDR view (knee 1, head 0) into dRsView, Lanczos (nativeFitPlanar) to tw x th
// into dst clamped at SDR white as on the SDR route, back to the HDR codes in place; false = a launch failed
static bool nativeViewFit(NativeRife& nr, const void* src, int half, int ps, int rs, int sw, int sh, float* dst, int tw,
                          int th, int mode)
{
    cudaStream_t st = nr.stream;
    int vps = sw * sh, n = 3 * tw * th;
    void* b[] = {(void*)&dst, &n};
    return nativeHdrEnc(nr, src, half, ps, rs, sw, sh, nr.dRsView, 0, vps, sw, mode, 1.0f, 0.0f) &&
           nativeFitPlanar(nr, nr.dRsView, 0, vps, sw, sw, sh, dst, tw, th, st) &&
           cuLaunchKernel(nr.fClamp01, (n + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, b, nullptr) == CUDA_SUCCESS &&
           nativeHdrFromSdr(nr, dst, 0, tw * th, tw, tw, th, dst, tw * th, tw, mode);
}

// RIFE's last step (k_rifeBlendF, k_rifeBlend when outHalf) for the n tweens of the last IFNet enqueue, into dMerged:
// pic = frame 0 in planes 0..2 and frame 1 in 3..5 at the pictures' layout (ph x pw), fp32 or fp16 by
// half, or the reverse (sw). The model's frame is the pictures' (the engine's own x: the engine's warp clamps at its
// padded frame) unless mph is set (RIFE's two domains: the flow is read at the picture size and a
// tap clamps at the picture's content). zeroPad: a tap in the picture's pad reads 0 (the author's zero pad).
static bool nativeRifeBlend(NativeRife& nr, const void* pic, int half, int n, cudaStream_t st, int sw = 0)
{
    int pps = nr.ph * nr.pw, prs = nr.pw, w = nr.w, h = nr.h;
    int mps = nr.mph ? nr.mph * nr.mpw : pps, mrs = nr.mph ? nr.mpw : nr.pw;
    int fw = nr.mph ? nr.mw : w, fh = nr.mph ? nr.mh : h;
    int cw = (fw == w && fh == h) ? nr.pw : w, ch = (fw == w && fh == h) ? nr.ph : h;
    int zp = nr.zeroPad ? 1 : 0;
    void* a[] = {(void*)&pic, &sw,       &half, &pps, &prs, &w,  &h,          &cw, &ch,
                 &nr.dFlow,   &nr.dMask, &mps,  &mrs, &fw,  &fh, &nr.dMerged, &zp};
    return cuLaunchKernel(nr.outHalf ? nr.fRifeBlend : nr.fRifeBlendF, (w + 15) / 16, (h + 15) / 16, (unsigned)n, 16,
                          16, 1, 0, (CUstream)st, a, nullptr) == CUDA_SUCCESS;
}

// blendOut: RIFE's last step for tween k of the last IFNet enqueue straight into its output slot
// (k_rifeBlendOut, the content rect at (x0, y0) as k_packOut / k_packOutHdr store it)
static bool nativeRifeBlendOut(NativeRife& nr, const void* pic, int half, int k, uint8_t* slot, cudaStream_t st,
                               int sw = 0)
{
    int pps = nr.ph * nr.pw, prs = nr.pw, w = nr.w, h = nr.h;
    int mps = nr.mph ? nr.mph * nr.mpw : pps, mrs = nr.mph ? nr.mpw : nr.pw;
    int fw = nr.mph ? nr.mw : w, fh = nr.mph ? nr.mh : h;
    int cw = (fw == w && fh == h) ? nr.pw : w, ch = (fw == w && fh == h) ? nr.ph : h;
    const uint16_t* flow = nr.dFlow + (size_t)k * 4 * mps;
    const uint16_t* mask = nr.dMask + (size_t)k * mps;
    int hdr = nr.hdr ? 1 : 0, pitch = (int)nr.pitch, zp = nr.zeroPad ? 1 : 0;
    void* a[] = {(void*)&pic, &sw,  &half, &pps, &prs, &w,    &h,     &cw,    &ch,    (void*)&flow, (void*)&mask,
                 &mps,        &mrs, &fw,   &fh,  &hdr, &slot, &pitch, &nr.x0, &nr.y0, &zp};
    void* a32[] = {(void*)&pic, &sw,  &pps, &prs, &w,   &h,           &cw,    &ch,    (void*)&flow, (void*)&mask,
                   &mps,        &mrs, &fw,  &fh,  &hdr, (void*)&slot, &pitch, &nr.x0, &nr.y0,       &zp};
    return cuLaunchKernel(half ? nr.fRifeBlendOut : nr.fRifeBlendOut32, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0,
                          (CUstream)st, half ? a : a32, nullptr) == CUDA_SUCCESS;
}

static bool nativeModelInit(NativeRife& nr)
{
    // every net runs on the host's kernels: per session their modules and weights (the ready steps below), the tween
    // buffers and the shared activation block; the resident host keeps the dtypes across sessions
    // Restore on the host's kernels: the bundled Real-ESRGAN or AnimeJaNai's form (restNatSetup)
    nr.restNatKind = nr.restorePath == kRestNative      ? 1
                     : nr.restorePath == kRestNativeAj  ? 2
                     : nr.restorePath == kRestNativeA4k ? 3
                                                        : 0;
    nr.restNat = nr.restNatKind != 0;
    // RIFE's IFNet on the host's kernels: no engine either
    nr.ifnNat = nr.ifnetPath == kIfnNative;
    auto restoreReady = [&]() -> bool {
        if (nr.restNat)
            return restNatSetup(nr);
        if (nr.restorePath.empty())
            return true;
        LOG("native: the Restore model %s does not run on the host's kernels\n", nr.restorePath.c_str());
        return false;
    };
    // native DRBA: block0's module and weights (nativeB0Module), per session; it reads the IFNet's encode ring in the
    // IFNet's feature dtype
    auto block0Ready = [&]() -> bool {
        if (!nr.drba)
            return true;
        std::string why;
        if (!nativeB0Module(nr, why))
        {
            LOG("native: drba: block0 unavailable (%s)\n", why.c_str());
            return false;
        }
        LOG("native: drba: block0 on the host's kernels\n");
        return true;
    };
    // RIFE's IFNet: its module and weights (nativeIfnModule), per session
    auto ifnReady = [&]() -> bool {
        if (!nr.ifnNat)
            return true;
        std::string why;
        if (!nativeIfnModule(nr, why))
        {
            LOG("native: rife: the IFNet unavailable (%s)\n", why.c_str());
            return false;
        }
        LOG("native: rife: the IFNet on the host's kernels\n");
        nr.ifnGraphs.mode = chainGraphMode("SMV_IFN_GRAPH");
        return true;
    };
    // the RIFE family's nets on one activation block, set once they all exist
    const bool modelShare = lkEnv("SMV_CTXSHARE") != "0";
    auto shareReady = [&]() -> bool { return !modelShare || nativeModelShareMemory(nr); };
    // GMFSS: every net on the host's kernels. nativeGmfssSetup allocates the chain's buffers (per session, freed in
    // nativeFree).
    if (nr.gmfss)
    {
        const int64_t t0 = nowQpc100();
        size_t freeB = 0, freeA = 0, totB = 0;
        cudaMemGetInfo(&freeB, &totB);
        const bool gmShare = lkEnv("SMV_GM_CTXSHARE") != "0";
        if (gmShare && !nativeGmShareMemory(nr))
            return false;
        if (!restoreReady())
            return false;
        cudaMemGetInfo(&freeA, &totB);
        LOG("native: gmfss ready in %.2f s, VRAM %+.0f MB\n", (nowQpc100() - t0) / 1e7,
            ((double)freeB - (double)freeA) / 1048576.0);
        if (!nativeGmfssSetup(nr))
            return false;
        if (g_resident)
        {
            g_res.ready = true; // the host stays resident after a GMFSS session too (no IFNet to reuse)
            g_res.dev = nr.dev;
            g_res.ifnetPath.clear();
        }
        return true;
    }
    if (g_resident && g_res.ready && g_res.ifnetPath == nr.ifnetPath)
    {
        const int64_t t0 = nowQpc100();
        nr.featHalf = g_res.featHalf;
        nr.xHalf = g_res.xHalf;
        nr.outHalf = g_res.outHalf;
        nr.tHalf = g_res.tHalf;
        if (!nr.noEngine && !nr.nvof && !nr.fruc && !nr.dlssg && !nativeTweenBuffers(nr))
            return false;
        if (!block0Ready() || !ifnReady() || !shareReady() || !restoreReady())
            return false;
        LOG("native: resident engines reused, contexts recreated in %.2f s\n", (nowQpc100() - t0) / 1e7);
        return true;
    }
    if (g_resident && g_res.ready)
        residentFreeEngines(); // a different pair (new window size)
    if (!nr.noEngine && !nr.nvof && !nr.fruc && !nr.dlssg)
    {
        if (!nr.ifnNat)
        {
            LOG("native: rife: the handoff named no IFNet on the host's kernels (%s)\n", nr.ifnetPath.c_str());
            return false;
        }
        // the native IFNet's dtypes: the frame pair x, the timestep (the constant fills and DRBA's map are written in
        // fp16) and the encodes f0 / f1 (nativeEncodeRun writes that dtype) in fp16; flow / mask out in fp16, the tween
        // made by RIFE's last step (nativeRifeBlend)
        nr.tHalf = nr.xHalf = nr.featHalf = true;
        // the tweens in fp32, each reader rounds once; SMV_TWEEN_FP16=1 = the fp16 store (a second rounding)
        nr.outHalf = lkEnv("SMV_TWEEN_FP16") == "1";
        if (!nativeTweenBuffers(nr))
            return false;
    }
    if (!g_offline && nr.mph)
        LOG("native: RIFE finds the motion on the picture at %dx%d%s, its warp + blend runs on the %dx%d pictures\n",
            nr.mw, nr.mh,
            nr.hdr ? (nr.rtxHdr ? " (the SDR picture before RTX HDR)" : " (the HDR desktop's PQ read as SDR sRGB)")
                   : "",
            nr.w, nr.h);
    if (!block0Ready() || !ifnReady() || !shareReady() || !restoreReady())
        return false;
    if (g_resident)
    {
        g_res.ready = true;
        g_res.featHalf = nr.featHalf;
        g_res.xHalf = nr.xHalf;
        g_res.outHalf = nr.outHalf;
        g_res.tHalf = nr.tHalf;
        g_res.dev = nr.dev;
        g_res.ifnetPath = nr.ifnetPath;
    }
    return true;
}

static void nativeFree(NativeRife& nr)
{
    if (nr.stream)
        cudaStreamSynchronize(nr.stream);
    nativeNvofFree(nr);     // the Optical Flow session and its buffers (nvof sessions only)
    nativeFrucFree(nr);     // the bridge's FRUC instance and its surfaces (fruc sessions only)
    nativeFsrupRelease(nr); // the FSR upscaler and its vector session (fsrUp sessions only)
    nativeA4kRelease(nr);   // Anime4K's buffers: the resize's, Restore's, the statistics (Anime4K sessions only)
    nativeAcnRelease(nr);   // ArtCNN's weights and frame buffers (ArtCNN sessions only)
    // release the TrueHDR feature on the compute thread, before the CUDA teardown
    if (nr.thdrN && !g_offline) // offline logs its own line with the light levels
        LOG("native: TrueHDR eval %.2f ms mean, %.2f ms max, over %llu real frames\n", nr.thdrMs / (double)nr.thdrN,
            nr.thdrMaxMs, (unsigned long long)nr.thdrN);
    if (nr.vsrN && nr.vsrAsync)
        LOG("native: RTX VSR eval ordered by events over %llu %s frames (SMV_VSR_SYNC=1 times each call)\n",
            (unsigned long long)nr.vsrN, g_offline ? "output" : "presented");
    else if (nr.vsrN)
        LOG("native: RTX VSR eval %.2f ms mean, %.2f ms max, over %llu %s frames\n", nr.vsrMs / (double)nr.vsrN,
            nr.vsrMaxMs, (unsigned long long)nr.vsrN, g_offline ? "output" : "presented");
    if (nr.liveNr)
        LOG("native: live DLSS 5 %.2f ms mean, %.2f ms max over %llu captured frames (host submit, zero-copy), "
            "its last output reused on %llu identical captures\n",
            nr.nrN ? nr.nrMs / (double)nr.nrN : 0.0, nr.nrMaxMs, (unsigned long long)nr.nrN,
            (unsigned long long)nr.nrReused);
    if (g_rtxb.created && g_rtxb.shutdown)
    {
        g_rtxb.shutdown();
        g_rtxb.created = g_rtxb.vsrMade = false;
    }
    if (nr.drTweens || nr.drHeads)
        LOG("native: drba %llu tweens (%llu head pairs), %llu block0 runs\n", (unsigned long long)nr.drTweens,
            (unsigned long long)nr.drHeads, (unsigned long long)nr.drBlock0);
    for (void* p : {(void*)nr.dDrIBase, (void*)nr.dDrF[0], (void*)nr.dDrF[1], (void*)nr.dDrF[2], (void*)nr.dDrF[3],
                    (void*)nr.dDrX[0], (void*)nr.dDrX[1], (void*)nr.drWin[0].f10, (void*)nr.drWin[0].r,
                    (void*)nr.drWin[1].f10, (void*)nr.drWin[1].r, (void*)nr.dDrAcc, (void*)nr.dDrMBase, (void*)nr.dDrS,
                    (void*)(nr.drRtOwn ? nr.dDrRT : nullptr)})
        if (p)
            cudaFree(p);
    nr.dDrIBase = nullptr;
    nr.dDrMBase = nullptr;
    nr.dDrS = nr.dDrB = nr.dDrRT = nullptr;
    nr.drRtOwn = false;
    nr.drAuthor = false;
    nr.drSFid[0] = nr.drSFid[1] = -1;
    for (int i = 0; i < 4; i++)
    {
        nr.dDrI[i] = nr.dDrF[i] = nullptr;
        nr.dDrM[i] = nullptr;
    }
    nr.dDrX[0] = nr.dDrX[1] = nullptr;
    nr.dDrMX[0] = nr.dDrMX[1] = nullptr;
    for (auto& wn : nr.drWin)
        wn = NativeRife::DrWin{};
    nr.dDrAcc = nullptr;
    if (nr.dGmCtxMem) // after the contexts that ran on it
    {
        cudaFree(nr.dGmCtxMem);
        nr.dGmCtxMem = nullptr;
        nr.gmCtxMemB = 0;
    }
    for (auto& e : nr.slotEv)
        if (e)
            cudaEventDestroy(e);
    nr.slotEv.clear();
    for (auto& e : nr.gmEv)
        if (e)
        {
            cudaEventDestroy(e);
            e = nullptr;
        }
    for (auto& e : nr.gmCatchEv)
        if (e)
        {
            cudaEventDestroy(e);
            e = nullptr;
        }
    if (nr.gmFeatSkips)
        LOG("native: gmfss: feat_ext skipped on %u frames without tweens, %u taken by a later pair\n", nr.gmFeatSkips,
            nr.gmCatchUps);
    nr.gmFeatSkips = nr.gmCatchUps = 0;
    nr.gmCatchUp = false;
    if (nr.capEv)
    {
        cudaEventDestroy(nr.capEv);
        nr.capEv = nullptr;
    }
    for (cudaEvent_t& e : nr.pfEv)
        if (e)
        {
            cudaEventDestroy(e);
            e = nullptr;
        }
    for (cudaEvent_t& e : nr.grEv)
        if (e)
        {
            cudaEventDestroy(e);
            e = nullptr;
        }
    for (auto* s : {&nr.fgL1Ev, &nr.fgOfEv})
        for (auto& set : *s)
            for (cudaEvent_t& e : set)
                if (e)
                {
                    cudaEventDestroy(e);
                    e = nullptr;
                }
    if (nr.grLagEv)
    {
        cudaEventDestroy(nr.grLagEv);
        nr.grLagEv = nullptr;
    }
    nr.grArmed = nr.grLagArmed = false;
    for (int k = 0; k < 2; k++)
        nr.grL1Armed[k] = nr.grOfArmed[k] = false;
    // PG p.124: mappings must go before the external memory objects, and every outstanding
    // wait must have completed before the semaphore is destroyed (the stream sync above).
    if (nr.dOutRing)
    {
        cudaFree(nr.dOutRing);
        nr.dOutRing = nullptr;
    }
    if (nr.capMip)
    {
        cudaFreeMipmappedArray(nr.capMip);
        nr.capMip = nullptr;
        nr.capArr = nullptr;
    }
    if (nr.semCap)
    {
        cudaDestroyExternalSemaphore(nr.semCap);
        nr.semCap = nullptr;
    }
    if (nr.emCap)
    {
        cudaDestroyExternalMemory(nr.emCap);
        nr.emCap = nullptr;
    }
    if (nr.emOut)
    {
        cudaDestroyExternalMemory(nr.emOut);
        nr.emOut = nullptr;
    }
    for (void* p : {(void*)nr.dFlow, (void*)nr.dMask, (void*)nr.dM, (void*)nr.dMFit, (void*)nr.dRestRem,
                    (void*)nr.dRsView, (void*)nr.dRsRange})
        if (p)
            cudaFree(p);
    if (nr.hRsRange)
        cudaFreeHost(nr.hRsRange);
    nr.dFlow = nr.dMask = nr.dM = nullptr;
    nr.dMFit = nr.dRestRem = nr.dRsView = nullptr;
    nr.dRsRange = nr.hRsRange = nullptr;
    nr.rsModeLogs = 0;
    for (const NativeRife::RsTab& e : nr.rsTabs)
        cudaFree(e.d);
    nr.rsTabs.clear();
    for (void* p :
         {(void*)nr.dCap,    (void*)nr.dX,          (void*)nr.dXh,        (void*)nr.dF[0],   (void*)nr.dF[1],
          (void*)nr.dEncOwn, (void*)nr.dT,          (void*)nr.dMerged,    (void*)nr.dTmp,    (void*)nr.dCapF,
          (void*)nr.dThdrIn, (void*)nr.dThdrOut,    (void*)nr.dSrcG,      (void*)nr.dPres,   (void*)nr.dSrcPl,
          (void*)nr.dVsrIn,  (void*)nr.dVsrOut,     (void*)nr.dVsrView,   (void*)nr.dVsrRem, (void*)nr.dFitTmp,
          (void*)nr.dUp,     (void*)nr.dUpTmp,      nr.dRestIn,           nr.dRestOut,       (void*)nr.dRestTmp,
          (void*)nr.dRest,   (void*)nr.dStaticFlag, (void*)nr.dThdrStats, (void*)nr.dShIn,   (void*)nr.dShOut})
        if (p)
            cudaFree(p);
    if (nr.restNatOwn && nr.dRestA) // else the activations sit in the shared activation block
        cudaFree(nr.dRestA);
    if (nr.dRestW)
        cudaFree(nr.dRestW);
    nr.dRestA = nr.dRestB = nr.dRestW = nullptr;
    nr.restNat = nr.restNatOwn = false;
    nr.restNatKind = 0;
    if (nr.hStaticFlag)
    {
        cudaFreeHost(nr.hStaticFlag);
        nr.hStaticFlag = nullptr;
    }
    if (nr.hThdrStats)
    {
        cudaFreeHost(nr.hThdrStats);
        nr.hThdrStats = nullptr;
    }
    nr.dStaticFlag = nullptr;
    nr.dThdrStats = nullptr;
    nr.thdrStatsPending = false;
    // the GMFSS chain's buffers: per session like every other model buffer, so a
    // backend switch on the resident host gives the VRAM back even though the engines stay
    if (nr.gmScratchInAcc)
    {
        // the pair scratch lives inside dGmAcc: the accumulator's free returns it
        for (auto& p : nr.dGmA)
            p = nullptr;
        for (auto& p : nr.dGmM)
            p = nullptr;
        nr.dGmCorr = nr.dGmFlow8 = nr.dGmEnc = nr.dGmHalfM = nullptr;
        nr.dGmProp = nullptr;
        nr.gmScratchInAcc = false;
    }
    for (void* p : {nr.dGmFeat[0][0],
                    nr.dGmFeat[0][1],
                    nr.dGmFeat[0][2],
                    nr.dGmFeat[1][0],
                    nr.dGmFeat[1][1],
                    nr.dGmFeat[1][2],
                    (void*)nr.dGmHalf,
                    (void*)nr.dGmFlow,
                    nr.dGmMetric,
                    (void*)nr.dGmFlowP[0],
                    (void*)nr.dGmFlowP[1],
                    (void*)nr.dGmMetP[0],
                    (void*)nr.dGmMetP[1],
                    (void*)nr.dGmAcc,
                    (void*)nr.dGmFa,
                    nr.dGmFb,
                    nr.dGmFc,
                    nr.dGmFd,
                    (void*)nr.dGmT,
                    nr.dGmOut,
                    nr.dGmWl,
                    nr.dGmBl,
                    (void*)nr.dGmF,
                    (void*)nr.dGmEnc,
                    (void*)nr.dGmHalfM,
                    (void*)nr.dGmRange,
                    nr.dGmA[0],
                    nr.dGmA[1],
                    (void*)nr.dGmCorr,
                    nr.dGmM[0],
                    nr.dGmM[1],
                    nr.dGmM[2],
                    (void*)nr.dGmFlow8,
                    nr.dGmProp,
                    nr.dGmMap[0],
                    nr.dGmMap[1],
                    (void*)nr.dGsCnt[0],
                    (void*)nr.dGsCnt[1],
                    (void*)nr.dGsOff[0],
                    (void*)nr.dGsOff[1],
                    (void*)nr.dGsList[0],
                    (void*)nr.dGsList[1],
                    (void*)nr.dGsSrc[0],
                    (void*)nr.dGsSrc[1],
                    (void*)nr.dGsBlk[0],
                    (void*)nr.dGsBlk[1],
                    nr.dFnW,
                    nr.dFnOwn,
                    nr.dMnW,
                    nr.dMnOwn,
                    nr.dGiW,
                    nr.dGiOwn,
                    nr.dFuW,
                    nr.dFuOwn,
                    nr.dGbW,
                    nr.dGbOwn,
                    nr.dBbW,
                    nr.dBbOwn,
                    nr.dGaW,
                    nr.dGaOwn,
                    (void*)nr.dGaPos,
                    nr.dB0W,
                    nr.dB0Own,
                    nr.dIfnW,
                    nr.dIfnOwn})
        if (p)
            cudaFree(p);
    nr.dFnW = nr.dFnOwn = nr.dFnScr = nullptr;
    nr.dMnW = nr.dMnOwn = nullptr;
    nr.dGiW = nr.dGiOwn = nullptr;
    nr.dFuW = nr.dFuOwn = nullptr;
    nr.fuOps.clear();
    nr.dGbW = nr.dGbOwn = nullptr;
    nr.dBbW = nr.dBbOwn = nullptr;
    nr.dGaW = nr.dGaOwn = nullptr;
    nr.dGaPos = nullptr;
    nr.dB0W = nr.dB0Own = nullptr;
    nr.dIfnW = nr.dIfnOwn = nullptr;
    for (int s = 0; s < 2; s++)
    {
        nr.dGsCnt[s] = nr.dGsOff[s] = nr.dGsList[s] = nr.dGsBlk[s] = nullptr;
        nr.dGsSrc[s] = nullptr;
    }
    nr.gmGather = false;
    for (auto& p : nr.dGmA)
        p = nullptr;
    for (auto& p : nr.dGmM)
        p = nullptr;
    nr.dGmFlow8 = nullptr;
    nr.dGmProp = nullptr;
    nr.dGmCorr = nullptr;
    nr.dGmMap[0] = nr.dGmMap[1] = nullptr;
    nr.gmMapsNext = false;
    if (nr.hGmRange)
        cudaFreeHost(nr.hGmRange);
    nr.hGmRange = nullptr;
    nr.dGmRange = nullptr;
    for (int s = 0; s < 2; s++)
        for (int l = 0; l < 3; l++)
            nr.dGmFeat[s][l] = nullptr;
    nr.dGmHalf = nullptr;
    nr.dGmFlow = nullptr;
    nr.dGmMetric = nullptr;
    nr.dGmFlowP[0] = nr.dGmFlowP[1] = nullptr;
    nr.dGmMetP[0] = nr.dGmMetP[1] = nullptr;
    nr.dGmAcc = nullptr;
    nr.dGmFa = nullptr;
    nr.dGmFb = nullptr;
    nr.dGmFc = nullptr;
    nr.dGmFd = nullptr;
    nr.dGmT = nullptr;
    nr.dGmOut = nullptr;
    nr.dGmWl = nr.dGmBl = nullptr;
    nr.gmLr[0] = nr.gmLr[1] = nr.gmLr[2] = -1.0f;
    nr.gmTail = nr.gmTailHwc = nr.gmFeatHwc = false;
    nr.dGmF = nullptr;
    nr.dGmEnc = nullptr;
    nr.dGmHalfM = nullptr;
    nr.dCap = nullptr;
    nr.dX = nullptr;
    nr.dXh = nullptr;
    nr.dF[0] = nr.dF[1] = nullptr;
    nr.dEncOwn = nullptr;
    nr.dT = nullptr;
    nr.tHeldMask = 0;
    nr.dMerged = nullptr;
    nr.dTmp = nullptr;
    nr.tmpN = 0;
    nr.dCapF = nullptr;
    nr.dThdrIn = nullptr;
    nr.dThdrOut = nullptr;
    nr.dSrcG = nullptr;
    nr.dShIn = nullptr;
    nr.dShOut = nullptr;
    nr.dPres = nullptr;
    nr.dSrcPl = nullptr;
    nr.dVsrIn = nullptr;
    nr.dVsrOut = nullptr;
    nr.dVsrView = nullptr;
    nr.dVsrRem = nullptr;
    nr.dFitTmp = nullptr;
    nr.dUp = nullptr;
    nr.dUpTmp = nullptr;
    nr.dRestIn = nullptr;
    nr.dRestOut = nullptr;
    nr.dRestTmp = nullptr;
    nr.dRest = nullptr;
    if (nr.cuMod && nr.cuMod != g_res.cuMod)
        cuModuleUnload(nr.cuMod);
    nr.cuMod = nullptr;
    if (nr.a4kMod)
        cuModuleUnload(nr.a4kMod);
    nr.a4kMod = nullptr;
    for (CUfunction& f : nr.fA4k)
        f = nullptr;
    if (nr.acnMod)
        cuModuleUnload(nr.acnMod);
    nr.acnMod = nullptr;
    for (CUfunction& f : nr.fAcn)
        f = nullptr;
    if (nr.fnMod)
        cuModuleUnload(nr.fnMod);
    nr.fnMod = nullptr;
    for (CUfunction& f : nr.fFn)
        f = nullptr;
    if (nr.mnMod)
        cuModuleUnload(nr.mnMod);
    nr.mnMod = nullptr;
    for (CUfunction& f : nr.fMn)
        f = nullptr;
    chainGraphsFree(nr.giGraphs);
    if (nr.giMod)
        cuModuleUnload(nr.giMod);
    nr.giMod = nullptr;
    for (CUfunction& f : nr.fGi)
        f = nullptr;
    chainGraphsFree(nr.fuGraphs);
    if (nr.fuMod)
        cuModuleUnload(nr.fuMod);
    nr.fuMod = nullptr;
    for (CUfunction& f : nr.fFu)
        f = nullptr;
    chainGraphsFree(nr.gbGraphs);
    if (nr.gbMod)
        cuModuleUnload(nr.gbMod);
    nr.gbMod = nullptr;
    for (CUfunction& f : nr.fGb)
        f = nullptr;
    chainGraphsFree(nr.bbGraphs);
    if (nr.bbMod)
        cuModuleUnload(nr.bbMod);
    nr.bbMod = nullptr;
    for (CUfunction& f : nr.fBb)
        f = nullptr;
    for (ChainGraphs& g : nr.gaGraphs)
        chainGraphsFree(g);
    if (nr.gaMod)
        cuModuleUnload(nr.gaMod);
    nr.gaMod = nullptr;
    for (CUfunction& f : nr.fGa)
        f = nullptr;
    if (nr.b0Mod)
        cuModuleUnload(nr.b0Mod);
    nr.b0Mod = nullptr;
    for (CUfunction& f : nr.fB0)
        f = nullptr;
    chainGraphsFree(nr.ifnGraphs);
    if (nr.ifnMod)
        cuModuleUnload(nr.ifnMod);
    nr.ifnMod = nullptr;
    for (CUfunction& f : nr.fIfn)
        f = nullptr;
    nr.ifnNat = false;
    if (nr.stream)
    {
        cudaStreamDestroy(nr.stream);
        nr.stream = nullptr;
    }
}
