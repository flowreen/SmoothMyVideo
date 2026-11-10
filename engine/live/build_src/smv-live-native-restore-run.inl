// part of smv-live-native.inl (included there, in place); not a standalone translation unit
// tween k of the last batch (RIFE's last step's, 3 planes of `plane` elements each, fp32, fp16 when outHalf)
static const void* mergedAt(const NativeRife& nr, size_t k, size_t plane)
{
    const size_t n = k * 3 * plane;
    return nr.outHalf ? (const void*)((const uint16_t*)nr.dMerged + n) : (const void*)(nr.dMerged + n);
}

// Restore (live and offline) on one
// source-size planar frame (srcW x srcH: live the captured frame, offline the decoded one with
// a working size, else the model frame; fp16 when sHalf: a tween as the IFNet wrote it), the
// result folded into dst (tw x th).
// Restore's way back on HDR planes (k_hdrRestOut, in place on dst, tw x th): the remainder comes from the source
// (sw x sh, planes ps_ apart, rows rs_ apart, fp16 when sHalf), fitted to the target first when the sizes differ
static bool nativeHdrRestOut(NativeRife& nr, float* dst, int tw, int th, const void* s, int sHalf, int ps_, int rs_,
                             int sw, int sh, int mode, bool viewRef = false)
{
    cudaStream_t st = nr.stream;
    const void* ref = s;
    int half = sHalf, fps = ps_, frs = rs_, rps = tw * th;
    if (tw != sw || th != sh)
    {
        if (!nr.dRestRem)
        {
            nr.die("restore HDR fit buffer missing");
            return false;
        }
        // a source inside the SDR range: the reference is its SDR view's fit, so it carries no ringing above white
        if (viewRef ? !nativeViewFit(nr, s, sHalf, ps_, rs_, sw, sh, nr.dRestRem, tw, th, mode)
                    : !nativeFitPlanar(nr, s, sHalf, ps_, rs_, sw, sh, nr.dRestRem, tw, th, st))
        {
            nr.die("restore HDR fit launch failed");
            return false;
        }
        ref = nr.dRestRem;
        half = 0;
        fps = rps;
        frs = tw;
    }
    void* a[] = {(void*)&dst, &rps, &tw,         (void*)&ref, &half, &fps,  &frs,
                 &tw,         &th,  (void*)&dst, &rps,        &tw,   &mode, &nr.encWhite};
    if (cuLaunchKernel(nr.fHdrRestOut, (tw + 15) / 16, (th + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a, nullptr) !=
        CUDA_SUCCESS)
    {
        nr.die("restore HDR codes launch failed");
        return false;
    }
    return true;
}

// false = a launch failed (die was called); an engine enqueue refusal drops the pass for the rest
// of the session instead, the caller then continues with the unrestored source.
// HDR planes (enc 1 PQ, 2 HLG): the engine reads the frame's SDR view (hdrToModel, knee 1, head 0, the sign
// clipped), the fold is the SDR one, and k_hdrRestOut adds back the light the view never held.
// Restore = Anime4K (restNatKind 3): its passes on the same input, then its de-ringing clamp, unless the resize's Anime4K
// pass closes this frame's chain (a4kAfter: the clamp and its statistics move there).
static bool nativeRestoreRun(NativeRife& nr, const void* s, int ps_, int rs_, float* dst, int tw, int th, int sHalf = 0,
                             int enc = 0, bool viewRef = false, bool a4kAfter = false)
{
    cudaStream_t st = nr.stream;
    int sw = srcW(nr), sh = srcH(nr);
    if (enc)
    {
        if (!nativeHdrEnc(nr, s, sHalf, ps_, rs_, sw, sh, nr.dRestIn, nr.restHalfIn ? 1 : 0, sw * sh, sw, enc, 1.0f,
                          0.0f))
        {
            nr.die("restore model values launch failed");
            return false;
        }
    }
    else if (nr.restHalfIn)
    {
        void* a[] = {(void*)&s, &sHalf, &ps_, &rs_, &sw, &sh, &nr.dRestIn};
        if (cuLaunchKernel(nr.fRestIn, (sw + 15) / 16, (sh + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a, nullptr) !=
            CUDA_SUCCESS)
        {
            nr.die("restIn launch failed");
            return false;
        }
    }
    else if (sHalf)
    {
        // an fp32 engine input from an fp16 source: the 1:1 fit is the widening crop copy
        void* a[] = {(void*)&s, &sHalf, &ps_, &rs_, &sw, &sh, &nr.dRestIn, &sw, &sh};
        if (cuLaunchKernel(nr.fFitPlanar, (sw + 15) / 16, (sh + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a, nullptr) !=
            CUDA_SUCCESS)
        {
            nr.die("restore input widen failed");
            return false;
        }
    }
    else
    {
        for (int c = 0; c < 3; c++)
            if (cudaMemcpy2DAsync((float*)nr.dRestIn + (size_t)c * sw * sh, (size_t)sw * 4,
                                  (const float*)s + (ptrdiff_t)c * ps_, (size_t)rs_ * 4, (size_t)sw * 4, sh,
                                  cudaMemcpyDeviceToDevice, st) != cudaSuccess)
            {
                nr.die("restore input copy failed");
                return false;
            }
    }
    if (nr.restNatKind == 3)
    {
        if (!nativeA4kRestoreRun(nr, dst, tw, th, a4kAfter, enc != 0))
            return false;
        return !enc || nativeHdrRestOut(nr, dst, tw, th, s, sHalf, ps_, rs_, sw, sh, enc, viewRef);
    }
    if (!restNatRun(nr, st))
    {
        nr.die("restore kernel launch failed");
        return false;
    }
    // w4 / h4 = the model's output size (restScale x the input: Real-ESRGAN 4, AnimeJaNai 2)
    int half = nr.restHalfOut ? 1 : 0, w4 = nr.restScale * sw, h4 = nr.restScale * sh, ps4 = w4 * h4;
    // realesr.fit: the Lanczos pair at every ratio (an exact copy at the model's own scale), with `out.clamp(0,1)`
    // folded into the taps and the final clamp in the vertical pass
    const void* tabH = rsTabFor(nr, w4, tw, st);
    const void* tabV = rsTabFor(nr, h4, th, st);
    if (!tabH || !tabV)
    {
        nr.die("restore fold tap tables failed");
        return false;
    }
    int inHalf = nr.restHalfIn ? 1 : 0;
    const void* base = nr.restSubBase ? nr.dRestIn : nullptr; // Real-ESRGAN adds the nearest base, AnimeJaNai none
    void* ah[] = {&nr.dRestOut, &half, &ps4, &w4, &w4, &h4, &nr.dRestTmp, &tw, (void*)&tabH};
    void* ahs[] = {&nr.dRestOut, &half, (void*)&base, &inHalf,      &nr.restSubS, &nr.restSubW,
                   &nr.restSubH, &w4,   &h4,          &nr.dRestTmp, &tw,          (void*)&tabH};
    if (cuLaunchKernel(nr.restSub ? nr.fRestFoldHSub : nr.fRestFoldH, (tw + 15) / 16, (h4 + 15) / 16, 1, 16, 16, 1, 0,
                       (CUstream)st, nr.restSub ? ahs : ah, nullptr) != CUDA_SUCCESS)
    {
        nr.die("restFoldH launch failed");
        return false;
    }
    void* av[] = {&nr.dRestTmp, &tw, &h4, &dst, &th, (void*)&tabV};
    if (cuLaunchKernel(nr.fRestFoldV, (tw + 15) / 16, (th + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, av, nullptr) !=
        CUDA_SUCCESS)
    {
        nr.die("restFoldV launch failed");
        return false;
    }
    return !enc || nativeHdrRestOut(nr, dst, tw, th, s, sHalf, ps_, rs_, sw, sh, enc, viewRef);
}

// Offline TrueHDR: the previous frame's statistics block reaches the
// accumulators once the stream is known idle (the TrueHDR sync below, or the render's end).
static void nativeOfflineThdrDrain(NativeRife& nr)
{
    if (!nr.thdrStatsPending)
        return;
    nr.thdrStatsPending = false;
    if (nr.thdrAcc)
        nr.thdrAcc->add(nr.hThdrStats, (uint64_t)nr.dw * nr.dh);
    for (; nr.thdrRepeat; nr.thdrRepeat--)
        if (nr.thdrAcc)
            nr.thdrAcc->repeat();
}

// a held slot re-sent from the previous real frame's finished bytes: its statistics record
// is that frame's, still pending (read one frame late), so the repeat waits for the drain
static void nativeOfflineThdrRepeat(NativeRife& nr)
{
    if (nr.thdrStatsPending)
        nr.thdrRepeat++;
    else if (nr.thdrAcc)
        nr.thdrAcc->repeat();
}
