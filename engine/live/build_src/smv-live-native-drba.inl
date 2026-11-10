// part of smv-live-native.inl (included there, in place); not a standalone translation unit
// ---- native DRBA ----------------------------------------------------------------------------
// DRBA as native launches on the group's stream. Frame ids count from 1 per
// session (drFid = the newest, k); id i sits in ring slot i & 3, four kept.
// Every frame is encoded once when pushed (encode is a pure function of the
// frame). A window centred on c holds flow10 (2 planes) and
// r = calc_flow(I1 = c, I2 = c + 1)'s flow12 | flow21 (4 planes); two slots, the one holding
// c - 1 is kept when c is built, and a kept left
// neighbour chains: this window's flow10 = its flow21, else block0 on
// (c, c - 1) computes it. The group presents time (k - 2) + f: f >= 0.5 from window k - 1 side
// -1 (x = [k-1, k-2], tt = 1 - f), f < 0.5 from window k - 2 side +1 (x = [k-2, k-1], tt = f);
// the head (three frames) runs plain pair RIFE on (k-2, k-1), fewer frames hold the lagged real
// frame k - 1.
// fp32 -> fp16 of n elements on the host's stream (the frames of the RIFE engines that take fp16)
static bool nativeF2h(NativeRife& nr, const float* src, uint16_t* dst, size_t n, cudaStream_t st)
{
    int k = (int)n;
    void* a[] = {&src, &dst, &k};
    return cuLaunchKernel(nr.fF2h, (k + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, a, nullptr) == CUDA_SUCCESS;
}

static float* drbaFrame(NativeRife& nr, uint32_t id)
{
    return nr.dDrI[id & 3];
}
static float* drbaEnc(NativeRife& nr, uint32_t id)
{
    return nr.dDrF[id & 3];
}
static uint16_t* drbaMotion(NativeRife& nr, uint32_t id)
{
    return nr.dDrM[id & 3];
}
// two domains: where k_motionIn writes the motion frame of the frame the next push takes
static uint16_t* drbaMotionNext(NativeRife& nr)
{
    return drbaMotion(nr, nr.drFid + 1);
}
// the model's frame: RIFE's motion frame in two domains, else the pictures' pad
static int drbaMh(const NativeRife& nr)
{
    return nr.mph ? nr.mph : nr.ph;
}
static int drbaMw(const NativeRife& nr)
{
    return nr.mph ? nr.mpw : nr.pw;
}

// RIFE's encode on the host's kernels, k_encC01 -> k_encC2Up on st (nr.stream: the planes share the activation block
// with the contexts that enqueue there): img = a planar frame at the encode's frame w x h (even, the /64 pad), fp16
// (imgHalf) or fp32; out = its kRifeFeatC planes in the IFNet's feature dtype (featHalf). cnn1's half-size planes live
// in the shared block, else in dEncOwn (SMV_CTXSHARE=0)
static bool nativeEncodeRun(NativeRife& nr, const void* img, int imgHalf, int w, int h, void* out, cudaStream_t st)
{
    const size_t need = encNatBytes(nr);
    float* a = nr.dGmCtxMem && (size_t)nr.gmCtxMemB >= need ? (float*)nr.dGmCtxMem : nr.dEncOwn;
    if (!a && cudaMalloc((void**)&nr.dEncOwn, need) == cudaSuccess)
        a = nr.dEncOwn;
    if (!a)
    {
        nr.die("encode planes alloc failed");
        return false;
    }
    int w2 = w / 2, h2 = h / 2, oh = nr.featHalf ? 1 : 0;
    const unsigned gx = (unsigned)(w2 + 31) / 32, gy = (unsigned)(h2 + 7) / 8;
    void* p0[] = {(void*)&img, &imgHalf, &w, &h, &a};
    void* p1[] = {&a, &w2, &h2, &out, &oh};
    if (cuLaunchKernel(nr.fEncC01, gx, gy, 1, 32, 8, 1, 0, (CUstream)st, p0, nullptr) != CUDA_SUCCESS ||
        cuLaunchKernel(nr.fEncC2Up, gx, gy, 1, 32, 8, 1, 0, (CUstream)st, p1, nullptr) != CUDA_SUCCESS)
    {
        nr.die("encode launch failed");
        return false;
    }
    return true;
}

// the new packed frame into the ring, and its encode beside it. Two domains: its motion frame sits in
// the ring already (drbaMotionNext), the encode reads that; same = a frame that reuses the previous
// frame's picture takes the previous motion frame too.
static bool nativeDrbaPush(NativeRife& nr, const float* dCur, bool same = false)
{
    cudaStream_t st = nr.stream;
    const size_t plane = (size_t)nr.ph * nr.pw, mplane = (size_t)drbaMh(nr) * drbaMw(nr);
    const uint32_t id = ++nr.drFid;
    float* dst = drbaFrame(nr, id);
    float* enc = drbaEnc(nr, id);
    // a frame built straight in its ring slot (dCur == dst) needs no copy
    if ((dCur != dst &&
         cudaMemcpyAsync(dst, dCur, 3 * plane * sizeof(float), cudaMemcpyDeviceToDevice, st) != cudaSuccess) ||
        (nr.mph && same && id > 1 &&
         cudaMemcpyAsync(drbaMotion(nr, id), drbaMotion(nr, id - 1), 3 * mplane * sizeof(uint16_t),
                         cudaMemcpyDeviceToDevice, st) != cudaSuccess))
    {
        nr.die("drba frame ring copy failed");
        return false;
    }
    // the encode reads the motion frame (fp16), else the new frame itself (fp32), and writes the ring slot in the
    // IFNet's feature dtype (read by block0 / IFNet)
    return nr.mph ? nativeEncodeRun(nr, drbaMotion(nr, id), 1, drbaMw(nr), drbaMh(nr), enc, st)
                  : nativeEncodeRun(nr, dst, 0, drbaMw(nr), drbaMh(nr), enc, st);
}

// calc_flow(frame a, frame b): block0 on the pair (nativeB0Run, its flow at 1/16), then its tail (the DRBA glue
// kernels) into out (4 planes: flow05 * 2 | flow15 * 2): the splat samples block0's planes at the motion frame into the
// accumulator, FlowNorm writes every value of out. Two domains: block0 reads the two motion frames (fp16).
static bool nativeDrbaFlow(NativeRife& nr, uint32_t a, uint32_t b, float* out)
{
    cudaStream_t st = nr.stream;
    int mw = drbaMw(nr), mh = drbaMh(nr);
    const size_t plane = (size_t)mh * mw;
    const void* img0 = nr.mph ? (const void*)drbaMotion(nr, a) : (const void*)drbaFrame(nr, a);
    const void* img1 = nr.mph ? (const void*)drbaMotion(nr, b) : (const void*)drbaFrame(nr, b);
    float* lo = nullptr;
    if (!nativeB0Run(nr, img0, img1, nr.mph ? 1 : 0, drbaEnc(nr, a), drbaEnc(nr, b), mw, mh, &lo, st))
    {
        nr.die("block0 launch failed");
        return false;
    }
    if (cudaMemsetAsync(nr.dDrAcc, 0, 6 * plane * sizeof(long long), st) != cudaSuccess)
    {
        nr.die("drba accumulator clear failed");
        return false;
    }
    void* as[] = {&lo, &mw, &mh, &nr.dDrAcc};
    if (cuLaunchKernel(nr.fDrFlowSplat, (mw + 15) / 16, (mh + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, as, nullptr) !=
        CUDA_SUCCESS)
    {
        nr.die("drbaFlowSplat launch failed");
        return false;
    }
    const int n = (int)plane;
    void* an[] = {&nr.dDrAcc, &mw, &mh, &out};
    if (cuLaunchKernel(nr.fDrFlowNorm, (n + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, an, nullptr) != CUDA_SUCCESS)
    {
        nr.die("drbaFlowNorm launch failed");
        return false;
    }
    nr.drBlock0++;
    return true;
}

// the window centred on c: cached, or built (chained on a kept c - 1 window); nullptr = died
static NativeRife::DrWin* nativeDrbaWindow(NativeRife& nr, uint32_t c)
{
    for (auto& wn : nr.drWin)
        if (wn.c == c)
            return &wn;
    const size_t plane = (size_t)drbaMh(nr) * drbaMw(nr);
    // keep the left neighbour (the chain), else a right one (every window >= c - 1 is needed;
    // the group builds k - 2 before k - 1 because the exe's fractions ascend, so both fit)
    const int s = (c > 1 && nr.drWin[0].c == c - 1)   ? 1
                  : (c > 1 && nr.drWin[1].c == c - 1) ? 0
                  : (nr.drWin[0].c == c + 1)          ? 1
                                                      : 0;
    NativeRife::DrWin& wn = nr.drWin[s];
    const NativeRife::DrWin* left = (c > 1 && nr.drWin[1 - s].c == c - 1) ? &nr.drWin[1 - s] : nullptr;
    wn.c = 0;
    if (left)
    {
        if (cudaMemcpyAsync(wn.f10, left->r + 2 * plane, 2 * plane * sizeof(float), cudaMemcpyDeviceToDevice,
                            nr.stream) != cudaSuccess)
        {
            nr.die("drba window chain copy failed");
            return nullptr;
        }
    }
    else
    {
        // the left flow lands in wn.r, which the right flow below overwrites (wn.c = 0: nothing reads it between)
        if (!nativeDrbaFlow(nr, c, c - 1, wn.r))
            return nullptr;
        if (cudaMemcpyAsync(wn.f10, wn.r, 2 * plane * sizeof(float), cudaMemcpyDeviceToDevice, nr.stream) !=
            cudaSuccess)
        {
            nr.die("drba window left flow copy failed");
            return nullptr;
        }
    }
    if (!nativeDrbaFlow(nr, c, c + 1, wn.r))
        return nullptr;
    wn.c = c;
    return &wn;
}

// one tween of the group at fraction f into dMerged; held = no tween (the slot shows the
// lagged real frame). nHist = frames in the history (at most 4). plain =
// plain pair RIFE on (k-2, k-1) at t = f whatever f is (the offline tail window:
// the last window has no right frame). slot = blendOut's output slot: the tween
// goes straight into it instead of dMerged.
// two domains: the blend's frame pair [k-2, k-1] read straight from the ring, whose slots are one block: slots
// (k-2) & 3 and (k-1) & 3 are neighbours unless (k-2) & 3 == 3; then frame k - 1 is copied once per k into slot
// (k+1) & 3, the slot of frame k - 3, which no picture reader uses after push(k) (the emits and the live lag read k - 1
// and k - 2), giving [k-1, k-2]. Frame k + 1 is then built in that same slot (both loops build each frame in its ring
// slot), so its decode / pack must be enqueued after group k's tweens on the same stream. pic = the pair's first slot, sw = the blend's frame order (1: frame 0 is the second
// slot); the IFNet's frame 0 is k - 1 on side -1, k - 2 on side +1 and the head's plain pair.
static bool nativeDrbaPics(NativeRife& nr, uint32_t k, int side, const float*& pic, int& sw)
{
    const size_t plane = (size_t)nr.ph * nr.pw;
    if (((k - 2) & 3) != 3)
    {
        pic = drbaFrame(nr, k - 2);
        sw = side < 0 ? 1 : 0;
        return true;
    }
    float* fix = nr.dDrI[(k + 1) & 3];
    if (nr.drPicFix != k)
    {
        if (cudaMemcpyAsync(fix, drbaFrame(nr, k - 1), 3 * plane * sizeof(float), cudaMemcpyDeviceToDevice,
                            nr.stream) != cudaSuccess)
        {
            nr.die("drba wrap pair copy failed");
            return false;
        }
        nr.drPicFix = k;
    }
    pic = fix;
    sw = side < 0 ? 0 : 1;
    return true;
}

// k_bilin: three planes (sw x sh at sps / srs) to dw x dh (at dps / drs), fp16 out when dHalf
static bool nativeBilin(NativeRife& nr, const float* src, int sps, int srs, int sw, int sh, void* dst, int dps, int drs,
                        int dw, int dh, int dHalf, cudaStream_t st)
{
    void* a[] = {(void*)&src, &sps, &srs, &sw, &sh, &dst, &dps, &drs, &dw, &dh, &dHalf};
    return cuLaunchKernel(nr.fBilin, (dw + 15) / 16, (dh + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a, nullptr) ==
           CUDA_SUCCESS;
}

// DRBA as its author (drAuthor): the pair's pictures (pic, frame 0 / frame 1 swapped by sw as k_rifeBlend reads them)
// stretched to the network's frame once per FRAME (to_inp; dDrS slot i & 1 holds frame i: both sides of a group and
// the next group's left frame find theirs there), RIFE's last step there with the IFNet's flow and mask as they are,
// the tween back to the pictures' size into dMerged (to_out)
static bool nativeDrbaAuthorBlend(NativeRife& nr, const float* pic, int sw, uint32_t k, int side, cudaStream_t st)
{
    int mw = nr.mpw, mh = nr.mph, mps = mw * mh, pps = nr.ph * nr.pw, half = 0, zp = 0;
    // frame k - 2 sits at pic's slot k2 (nativeDrbaPics: sw = 1 puts the IFNet's frame 0 second; frame 0 is k - 1 on
    // side -1, k - 2 on side +1), frame k - 1 at the other
    const int k2 = side < 0 ? 1 - sw : sw;
    for (int f = 0; f < 2; f++)
    {
        const int64_t fid = (int64_t)k - 2 + f;
        const int slot = (int)(fid & 1), src = f ? 1 - k2 : k2;
        if (nr.drSFid[slot] == fid)
            continue;
        if (!nativeBilin(nr, pic + (size_t)src * 3 * pps, pps, nr.pw, nr.w, nr.h, nr.dDrS + (size_t)slot * 3 * mps, mps,
                         mw, mw, mh, 0, st))
            return false;
        nr.drSFid[slot] = fid;
    }
    // the blend's frame 0 (k - 1 on side -1, k - 2 on side +1) is second in dDrS when its id is odd
    int ssw = (int)(((int64_t)k - (side < 0 ? 1 : 2)) & 1);
    const float* sp = nr.dDrS;
    void* a[] = {(void*)&sp, &ssw,      &half, &mps, &mw, &mw, &mh,      &mw, &mh,
                 &nr.dFlow,  &nr.dMask, &mps,  &mw,  &mw, &mh, &nr.dDrB, &zp};
    return cuLaunchKernel(nr.fRifeBlendF, (mw + 15) / 16, (mh + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a, nullptr) ==
               CUDA_SUCCESS &&
           nativeBilin(nr, nr.dDrB, mps, mw, mw, mh, nr.dMerged, pps, nr.pw, nr.w, nr.h, nr.outHalf ? 1 : 0, st);
}

// a real frame of a DRBA session as its author outputs it (drAuthor: stretched to the network's frame and back, as
// infer.py's to_out does to every frame it writes), else the picture itself
static const float* nativeDrbaReal(NativeRife& nr, const float* pic)
{
    if (!nr.drAuthor)
        return pic;
    const int mps = nr.mph * nr.mpw, pps = nr.ph * nr.pw;
    if (!nativeBilin(nr, pic, pps, nr.pw, nr.w, nr.h, nr.dDrB, mps, nr.mpw, nr.mpw, nr.mph, 0, nr.stream) ||
        !nativeBilin(nr, nr.dDrB, mps, nr.mpw, nr.mpw, nr.mph, nr.dDrRT, pps, nr.pw, nr.w, nr.h, 0, nr.stream))
    {
        nr.die("drba real frame stretch failed");
        return nullptr;
    }
    return nr.dDrRT;
}

static bool nativeDrbaTween(NativeRife& nr, float f, int nHist, bool& held, bool plain = false, uint8_t* slot = nullptr)
{
    cudaStream_t st = nr.stream;
    int mw = drbaMw(nr), mh = drbaMh(nr);
    const size_t plane = (size_t)nr.ph * nr.pw, mplane = (size_t)mh * mw;
    const uint32_t k = nr.drFid;
    held = false;
    NativeRife::DrWin* wn = nullptr;
    int side = +1;
    float tt = f;
    if (plain)
    {
    }
    else if (f >= 0.5f)
    {
        if (nHist < 3)
        {
            held = true;
            return true;
        }
        if (!(wn = nativeDrbaWindow(nr, k - 1)))
            return false;
        side = -1;
        tt = 1.0f - f;
    }
    else if (nHist >= 4)
    {
        if (!(wn = nativeDrbaWindow(nr, k - 2)))
            return false;
    }
    else if (nHist != 3)
    {
        held = true;
        return true;
    }
    // x = cat(I1, I0) on side -1, cat(I1, I2) on side +1 and for the head's plain pair
    const int xi = side < 0 ? 0 : 1;
    const uint32_t i1 = side < 0 ? k - 1 : k - 2, i0 = side < 0 ? k - 2 : k - 1;
    if (nr.drXFor[xi] != k)
    {
        if (nr.mph)
        {
            // two domains: the IFNet's x is the motion ring itself from frame i1's slot on (its pictures:
            // nativeDrbaPics). The ring descends, so on side -1 frame k - 2 already follows frame k - 1 unless
            // frame k - 1 holds the last slot (then frame k - 2 is copied into the spare). Side +1 and the head's
            // plain pair copy frame k - 1 into the slot after frame k - 2: the spare, or frame k - 3's, whose last
            // reader is window k - 2's block0, built above before this copy (the offline tail's plain tweens come
            // after the group's windowed ones). Frame k + 1's motion frame lands in that slot next, so its pack
            // must be enqueued after group k's tweens on the same stream, like its picture.
            uint16_t* x = drbaMotion(nr, i1);
            if (x + 3 * mplane != drbaMotion(nr, i0) &&
                cudaMemcpyAsync(x + 3 * mplane, drbaMotion(nr, i0), 3 * mplane * sizeof(uint16_t),
                                cudaMemcpyDeviceToDevice, st) != cudaSuccess)
            {
                nr.die("drba x copy failed");
                return false;
            }
            nr.dDrMX[xi] = x;
        }
        else if (nr.xHalf)
        {
            // the IFNet takes x in fp16: the two ring frames narrowed into dDrX (fp32-sized)
            uint16_t* x16 = (uint16_t*)nr.dDrX[xi];
            if (!nativeF2h(nr, drbaFrame(nr, i1), x16, 3 * plane, st) ||
                !nativeF2h(nr, drbaFrame(nr, i0), x16 + 3 * plane, 3 * plane, st))
            {
                nr.die("drba x f2h failed");
                return false;
            }
        }
        else if (cudaMemcpyAsync(nr.dDrX[xi], drbaFrame(nr, i1), 3 * plane * sizeof(float), cudaMemcpyDeviceToDevice,
                                 st) != cudaSuccess ||
                 cudaMemcpyAsync(nr.dDrX[xi] + 3 * plane, drbaFrame(nr, i0), 3 * plane * sizeof(float),
                                 cudaMemcpyDeviceToDevice, st) != cudaSuccess)
        {
            nr.die("drba x copy failed");
            return false;
        }
        nr.drXFor[xi] = k;
    }
    if (!wn)
    {
        // the head: plain pair RIFE at t = f, a constant timestep map
        if (!nativeFillT(nr, 0, f, mplane, st))
        {
            nr.die("timestep fill failed");
            return false;
        }
        nr.drHeads++;
    }
    else
    {
        if (cudaMemsetAsync(nr.dDrAcc, 0, 2 * mplane * sizeof(long long), st) != cudaSuccess)
        {
            nr.die("drba accumulator clear failed");
            return false;
        }
        void* as[] = {&wn->f10, &wn->r, &side, &tt, &mw, &mh, &nr.dDrAcc};
        if (cuLaunchKernel(nr.fDrDrmSplat, (mw + 15) / 16, (mh + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, as,
                           nullptr) != CUDA_SUCCESS)
        {
            nr.die("drbaDrmSplat launch failed");
            return false;
        }
        const int n = (int)mplane, th = nr.tHalf ? 1 : 0;
        nr.tHeldMask = 0; // the DRM map overwrites the timestep plane
        void* an[] = {&nr.dDrAcc, &wn->f10, &wn->r, &side, &tt, &mw, &mh, &nr.dT, (void*)&th};
        if (cuLaunchKernel(nr.fDrDrmNorm, (n + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, an, nullptr) !=
            CUDA_SUCCESS)
        {
            nr.die("drbaDrmNorm launch failed");
            return false;
        }
    }
    if (!nativeIfnRun(nr, nr.mph ? (const void*)nr.dDrMX[xi] : (const void*)nr.dDrX[xi], nr.dT, drbaEnc(nr, i1),
                      drbaEnc(nr, i0), mw, mh, nr.dFlow, nr.dMask, st))
    {
        nr.die("IFNet launch failed");
        return false;
    }
    // the tween: RIFE's last step on the frames this enqueue read, in two domains on their pictures
    const float* pic = nr.dDrX[xi];
    int sw = 0;
    if (nr.mph && !nativeDrbaPics(nr, k, side, pic, sw))
        return false;
    if (nr.drAuthor)
    {
        if (!nativeDrbaAuthorBlend(nr, pic, sw, k, side, st))
        {
            nr.die("drba stretched blend failed");
            return false;
        }
    }
    else if (slot ? !nativeRifeBlendOut(nr, pic, nr.mph ? 0 : (int)nr.xHalf, 0, slot, st, sw)
                  : !nativeRifeBlend(nr, pic, nr.mph ? 0 : (int)nr.xHalf, 1, st, sw))
    {
        nr.die(slot ? "rifeBlendOut launch failed" : "rifeBlend launch failed");
        return false;
    }
    nr.drTweens++;
    return true;
}
