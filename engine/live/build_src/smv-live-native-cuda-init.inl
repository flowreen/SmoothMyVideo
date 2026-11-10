// part of smv-live-native.inl (included there, in place); not a standalone translation unit
// ---- part 3: CUDA init, the compute thread, shutdown --------------------------------------

// the two ICtCp matrices are inverted in double and uploaded as fp32
static bool inv3d(const double* m, float* out)
{
    const double d =
        m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if (d == 0.0)
        return false;
    const double id = 1.0 / d;
    out[0] = (float)((m[4] * m[8] - m[5] * m[7]) * id);
    out[1] = (float)((m[2] * m[7] - m[1] * m[8]) * id);
    out[2] = (float)((m[1] * m[5] - m[2] * m[4]) * id);
    out[3] = (float)((m[5] * m[6] - m[3] * m[8]) * id);
    out[4] = (float)((m[0] * m[8] - m[2] * m[6]) * id);
    out[5] = (float)((m[2] * m[3] - m[0] * m[5]) * id);
    out[6] = (float)((m[3] * m[7] - m[4] * m[6]) * id);
    out[7] = (float)((m[1] * m[6] - m[0] * m[7]) * id);
    out[8] = (float)((m[0] * m[4] - m[1] * m[3]) * id);
    return true;
}

// The native init is split in three so that everything except
// the output ring can run BEFORE the source-rate measurement (the ring is sized from the
// measured slot count):
//  * nativeCudaDeviceInit: device, context, stream, capture texture + fence import. Depends on
//    the capture size only, so the early thread runs it in parallel with the engine handoff
//    (it writes only dev / stream / emCap / capMip / capArr / semCap, the handoff writes the
//    geometry fields; distinct members, no race).
//  * nativeCudaInitEarly: kernels and the model buffers (needs the handoff geometry).
//  * nativeRtxInit: the live TrueHDR / RTX VSR bridge, the sharpen staging and the warm-ups
//    (needs the kernel module and the handoff geometry), on a helper thread beside the engine
//    load in the early thread.
//  * nativeCudaInitLate: the output ring and the slot events (needs the slot count), always
//    on the compute thread.
// The CUDA runtime binds the device's PRIMARY context on every thread that calls
// cudaSetDevice, so streams, allocations and the kernel modules created on
// the early thread are valid on the compute thread once it binds the same device.
static bool nativeCudaDeviceInit(NativeRife& nr, IDXGIAdapter1* adapter, HANDLE hTex, HANDLE hFence, uint32_t capW,
                                 uint32_t capH, bool hdr)
{
    int dev = 0;
    if (cudaD3D11GetDevice(&dev, adapter) != cudaSuccess)
    {
        LOG("native: cudaD3D11GetDevice failed, using device 0\n");
        dev = 0;
    }
    nr.dev = dev;
    NCHK(cudaSetDevice(dev), "cudaSetDevice");
    NCHK(cudaFree(nullptr), "cuda context init");
    NCHK(cudaStreamCreateWithFlags(&nr.stream, cudaStreamNonBlocking), "cudaStreamCreate");

    // capture texture: D3D11 resource, DEDICATED (mandatory for this handle type), read as a
    // mipmapped array.
    cudaExternalMemoryHandleDesc md{};
    md.type = cudaExternalMemoryHandleTypeD3D11Resource;
    md.handle.win32.handle = hTex;
    // Capture format: BGRA8 in SDR, R16G16B16A16_FLOAT scRGB in HDR (8 bytes per pixel)
    const int capBpp = hdr ? 8 : 4;
    md.size = (size_t)capW * capH * capBpp;
    md.flags = cudaExternalMemoryDedicated;
    NCHK(cudaImportExternalMemory(&nr.emCap, &md), "import capture texture");
    cudaExternalMemoryMipmappedArrayDesc ad{};
    ad.offset = 0;
    // the capture's format: uchar4 in SDR, half4 with channel kind FLOAT in
    // HDR (PG p.143's sample table has no float16 four-channel case; the working half4 import is
    // the truth, do not "fix" it to match the table)
    ad.formatDesc = hdr ? cudaChannelFormatDesc{16, 16, 16, 16, cudaChannelFormatKindFloat}
                        : cudaChannelFormatDesc{8, 8, 8, 8, cudaChannelFormatKindUnsigned};
    ad.extent = cudaExtent{(size_t)capW, (size_t)capH, 0};
    ad.flags = 0;
    ad.numLevels = 1;
    if (cudaExternalMemoryGetMappedMipmappedArray(&nr.capMip, nr.emCap, &ad) != cudaSuccess)
    {
        ad.flags = cudaArrayColorAttachment;
        NCHK(cudaExternalMemoryGetMappedMipmappedArray(&nr.capMip, nr.emCap, &ad), "map capture mipmapped array");
    }
    NCHK(cudaGetMipmappedArrayLevel(&nr.capArr, nr.capMip, 0), "capture array level");

    cudaExternalSemaphoreHandleDesc sd{};
    sd.type = cudaExternalSemaphoreHandleTypeD3D11Fence;
    sd.handle.win32.handle = hFence;
    NCHK(cudaImportExternalSemaphore(&nr.semCap, &sd), "import capture fence");
    return true;
}

// bind the device's primary context on the CALLING thread: cudaSetDevice alone only records
// the device, the context becomes current at the first runtime call that needs one, and the
// driver API (cuModuleLoadData, cuLaunchKernel) never triggers that, so force it here
static bool nativeBindDevice(const NativeRife& nr)
{
    NCHK(cudaSetDevice(nr.dev), "cudaSetDevice (bind)");
    NCHK(cudaFree(nullptr), "cuda context bind");
    return true;
}

static bool nativeCudaInitLate(NativeRife& nr, HANDLE hOutBuf, uint64_t outBytes)
{
    // output ring: the exe's own shared D3D12 committed buffer, imported as a linear buffer
    cudaExternalMemoryHandleDesc od{};
    od.type = cudaExternalMemoryHandleTypeD3D12Resource;
    od.handle.win32.handle = hOutBuf;
    od.size = outBytes;
    od.flags = cudaExternalMemoryDedicated;
    NCHK(cudaImportExternalMemory(&nr.emOut, &od), "import output buffer");
    cudaExternalMemoryBufferDesc bd{};
    bd.offset = 0;
    bd.size = outBytes;
    bd.flags = 0;
    NCHK(cudaExternalMemoryGetMappedBuffer((void**)&nr.dOutRing, nr.emOut, &bd), "map output buffer");
    // letterbox bars and alpha, written once: opaque black is 0xFF000000 in BGRA8 and
    // 0xC0000000 in R10G10B10A2 with A = 3
    if (cuMemsetD32Async((CUdeviceptr)nr.dOutRing, nr.hdr ? 0xC0000000u : 0xFF000000u, (size_t)(outBytes / 4),
                         (CUstream)nr.stream) != CUDA_SUCCESS)
    {
        LOG("native: output ring clear failed\n");
        return false;
    }
    nr.slotEv.resize(nr.slots);
    for (auto& e : nr.slotEv)
        NCHK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "create slot event");
    return true;
}

// Live TrueHDR setup: the bridge buffers, the ICtCp matrices, the NGX feature and its
// warm-up eval (~0.6 s). Needs the capture size, the stream and the kernel module only, so the
// early thread runs it on a helper beside the engine load; the non-early
// compute thread runs it after the engine load. The caller's thread must have bound the device.
// the frame Restore and the first resize read: the model frame, or with a working size the decoded one
static int srcW(const NativeRife& nr)
{
    return nr.sw ? nr.sw : nr.w;
}
static int srcH(const NativeRife& nr)
{
    return nr.sh ? nr.sh : nr.h;
}

// AMD FSR upscaling at RTX VSR's resize (NativeRife's fsrUp notes; defined after nativeFsrfgRelease)
static bool nativeFsrupSetup(NativeRife& nr, int w, int h, int ow, int oh);
static bool nativeFsrupEval(NativeRife& nr, const RtxRect& ri, const RtxRect& ro, cudaStream_t st, unsigned int& rv);
static void nativeFsrupRelease(NativeRife& nr);
static uint64_t nativeVideoMemoryRoom();
static bool nativeA4kSetup(NativeRife& nr, int w, int h, int ow, int oh, int upEnc);
static bool nativeA4kRun(NativeRife& nr, const void* s, int sHalf, int ps_, int rs_, float* dst, int tw, int th);
static void nativeA4kRelease(NativeRife& nr);
static bool nativeAcnSetup(NativeRife& nr, int w, int h, int ow, int oh, int upEnc);
static bool nativeAcnRun(NativeRife& nr, const void* s, int sHalf, int ps_, int rs_, float* dst, int tw, int th);
static void nativeAcnRelease(NativeRife& nr);

static bool nativeRtxInit(NativeRife& nr)
{
    // live Sharpen and RTX VSR: the fit is known now, so the rule "VSR
    // only when the resize enlarges in both axes" decides here
    // Upscale to: the first resize target is the internal render size when set, else
    // the fit rect; VSR and the aa rule read that target
    const int tw = nr.uw ? nr.uw : nr.dw, th = nr.uw ? nr.uh : nr.dh;
    const int sw = srcW(nr), sh = srcH(nr);
    // the resize RTX VSR runs (vw x vh -> vtw x vth): the first one; offline with a working size there
    // are two, the pre-model one (the decoded frame to the working size) and the final one (the
    // working size to the output): VSR is ONE bridge instance, so it takes the final one when that
    // enlarges, else the pre-model one, and the other is Lanczos. Live has the same two: the capture
    // to the working size before the model, and the fit after it
    int vw = g_offline ? sw : nr.w, vh = g_offline ? sh : nr.h, vtw = tw, vth = th;
    nr.vsrPre = false;
    if (!g_offline && nr.vsrWant && !(tw > nr.w && th > nr.h) && nr.w > nr.cw && nr.h > nr.ch)
    {
        nr.vsrPre = true;
        vw = nr.cw;
        vh = nr.ch;
        vtw = nr.w;
        vth = nr.h;
    }
    if (g_offline && nr.nvPre)
    {
        // with RTX HDR the frames after the model are PQ and RTX VSR takes SDR only (NVIDIA: TrueHDR after
        // VSR), so VSR can only take the pre-model resize
        nr.vsrPost = nr.dw > nr.w && nr.dh > nr.h && !nr.rtxHdr;
        if (nr.vsrWant && nr.rtxHdr && nr.dw > nr.w && nr.dh > nr.h)
            LOG("native: %s takes SDR only and RTX HDR runs before the model: the resize after it is Lanczos\n",
                nr.fsrUp   ? (nr.fuKind ? "Intel XeSS upscaling" : "AMD FSR upscaling")
                : nr.a4kUp ? "Anime4K"
                : nr.acnUp ? "ArtCNN"
                           : "RTX VSR");
        if (nr.vsrPost)
        {
            vw = nr.w;
            vh = nr.h;
        }
        else
        {
            vtw = nr.w;
            vth = nr.h;
        }
    }
    if (nr.vsrWant)
    {
        const char* up = nr.fsrUp   ? (nr.fuKind ? "Intel XeSS upscaling" : "AMD FSR upscaling")
                         : nr.a4kUp ? "Anime4K"
                         : nr.acnUp ? "ArtCNN"
                                    : "RTX VSR";
        if (g_offline && !nr.nvPre && nr.rtxHdr)
            LOG("native: %s skipped (it takes SDR only, and the one resize follows RTX HDR and the model), Lanczos\n",
                up);
        else if (vtw > vw && vth > vh)
            nr.vsr = true;
        else
            LOG("native: live %s skipped (upscales only; this resize does not enlarge), Lanczos\n", up);
    }
    nr.fsrUp = nr.fsrUp && nr.vsr;
    nr.a4kUp = nr.a4kUp && nr.vsr;
    nr.acnUp = nr.acnUp && nr.vsr;
    nr.vsrPre = nr.vsrPre && nr.vsr;
    // offline HDR video (--src-hdr): the upscaler's frames are the source's PQ / HLG codes at either stage (RTX HDR,
    // which would make them PQ after the model, keeps the upscaler off that resize), so RTX VSR reads their SDR view
    // and AMD FSR upscaling runs in its HDR mode, as live does on HDR planes
    const int upEnc = g_offline ? nr.encPre : 0;
    if (nr.uw)
    {
        NCHK(cudaMalloc((void**)&nr.dUp, (size_t)3 * nr.uw * nr.uh * sizeof(float)), "alloc internal render frame");
        if (nr.upAa)
            NCHK(cudaMalloc((void**)&nr.dUpTmp, (size_t)3 * nr.uw * nr.h * sizeof(float)),
                 "alloc internal render pass");
        LOG("native: live upscale to: model %dx%d -> %dx%d first, then %s to %dx%d\n", nr.w, nr.h, nr.uw, nr.uh,
            nr.identity ? "1:1" : "fit", nr.dw, nr.dh);
    }
    // the fit as the separable pair (nr.fitAa decided in the handoff parse): the horizontal
    // pass buffer and the staging frame; its source is the internal render frame under Upscale to
    if (nr.fitAa)
    {
        const int fh = nr.uw ? nr.uh : nr.h;
        NCHK(cudaMalloc((void**)&nr.dFitTmp, (size_t)3 * nr.dw * fh * sizeof(float)), "alloc fit pass");
        LOG("native: fit: %dx%d -> %dx%d, Lanczos (%s)\n", nr.uw ? nr.uw : nr.w, fh, nr.dw, nr.dh,
            nr.dh < fh ? "downscale" : "upscale");
    }
    // a resize stage's staging frame (nativeOfflineStage): Restore after the model, or RTX VSR before it (the
    // working-size frame, from the capture's planes) or after it on HDR planes (VSR's way back to the codes), and live
    // Anime4K / ArtCNN after the model without Upscale to (their nets write planes, the slot's store reads them back);
    // live's other stores never stage (RTX VSR straight into the slot, the Lanczos fit fused into the store, Upscale
    // to through dUp), offline's emit allocates its own
    if ((nr.restore && !nr.restPre) || nr.vsrPre || (nr.vsr && nr.hdr) ||
        (!g_offline && nr.vsr && (nr.a4kUp || nr.acnUp) && !nr.uw))
        NCHK(cudaMalloc((void**)&nr.dPres,
                        (size_t)3 * (std::max)(nr.dw * nr.dh, nr.vsrPre ? nr.w * nr.h : 0) * sizeof(float)),
             "alloc fit staging");
    if (nr.vsrPre && !nr.dCapF)
        NCHK(cudaMalloc((void**)&nr.dCapF, (size_t)3 * nr.ch * nr.cw * sizeof(float)), "alloc capture planes");
    if (nr.sharpen > 0.0f)
        NCHK(cudaMalloc((void**)&nr.dShOut, (size_t)3 * nr.w * nr.h * sizeof(float)), "alloc after-DLSS 5 output");
    if (nr.sharpen > 0.0f && nr.encPost && nr.shHdrView)
        NCHK(cudaMalloc((void**)&nr.dShIn, (size_t)3 * nr.w * nr.h * sizeof(float)), "alloc sharpen SDR view");
    if (nr.sharpen > 0.0f)
    {
        LOG("native: sharpen: Adaptive Sharpen %.2f at %dx%d, after DLSS 5 and before RTX HDR and the model\n",
            nr.sharpen, nr.w, nr.h);
        if (nr.encPost && !nr.rtxHdr)
            LOG("native: sharpen: HDR planes: %s\n",
                nr.shHdrView
                    ? "the sharpen edits their SDR view as an SDR source's, clamped at SDR white, and the light "
                      "outside the view is added back"
                    : "the sharpen on the PQ / HLG codes (SMV_HDR_SHARPEN_VIEW=0)");
    }
    // Restore's fold target. Offline with a
    // working size, and live, Restore runs before the model: the target is the working / model size,
    // or the source (the capture live) when VSR runs the pre-model resize right after it. Offline without a
    // working size restores the model output: back to its size when VSR follows (so VSR sees the
    // restored frame), else the first resize target directly (restore-as-upscaler, one resize)
    if (nr.restore)
    {
        const bool vsrAfter = nr.vsr && (g_offline ? !nr.vsrPost : nr.vsrPre);
        const bool toModel = (g_offline && nr.nvPre) || nr.restPre;
        nr.restTw = vsrAfter ? sw : (toModel ? nr.w : tw);
        nr.restTh = vsrAfter ? sh : (toModel ? nr.h : th);
        // HDR planes: the source fitted to the target, the remainder Restore's way back adds (nativeHdrRestOut)
        if ((nr.encPre || nr.encPost) && (nr.restTw != sw || nr.restTh != sh))
            NCHK(cudaMalloc((void**)&nr.dRestRem, (size_t)3 * nr.restTw * nr.restTh * sizeof(float)),
                 "alloc restore HDR fit");
        const std::string rm = !g_restoreModel.empty() ? g_restoreModel.substr(g_restoreModel.find_last_of("\\/") + 1)
                               : g_restoreA4k          ? std::string("Anime4K Restore CNN UL")
                                                       : std::string("Real-ESRGAN animevideov3");
        LOG("native: live restore: %s (the host's kernels) at %dx%d -> %dx%d%s\n", rm.c_str(), sw, sh, nr.restTw,
            nr.restTh, nr.restPre ? ", on the captured frame before the model" : "");
    }
    // a resize of HDR planes before the model: live the capture to the working size (with Restore, its reference),
    // offline the pre-model stage (the resize, or Restore at a size change)
    const int rsw = g_offline ? sw : nr.cw, rsh = g_offline ? sh : nr.ch;
    const bool rsHere = g_offline
                            ? nr.encPre && nr.nvPre &&
                                  (nr.w != sw || nr.h != sh || (nr.restore && (nr.restTw != sw || nr.restTh != sh)))
                            : nr.encPost && (nr.w != nr.cw || nr.h != nr.ch);
    if (rsHere && nr.rsHdrView)
    {
        NCHK(cudaMalloc((void**)&nr.dRsView, (size_t)3 * rsw * rsh * sizeof(float)), "alloc resize SDR view");
        NCHK(cudaMalloc((void**)&nr.dRsRange, sizeof(int)), "alloc resize range flag");
        NCHK(cudaHostAlloc((void**)&nr.hRsRange, sizeof(int), cudaHostAllocDefault), "alloc resize range readback");
        LOG("native: resize: HDR planes: a frame inside the SDR range is resized as its SDR view (the SDR route's "
            "Lanczos, clamped at SDR white), any other on the codes\n");
    }
    else if (rsHere)
        LOG("native: resize: HDR planes: resized on the codes (SMV_HDR_RESIZE_VIEW=0)\n");
    if (upEnc && nr.vsr && !nr.fsrUp && !nr.dRsRange)
    {
        // RTX VSR's way back to the codes tests each frame's range itself at a stage with no resize view (the final one)
        NCHK(cudaMalloc((void**)&nr.dRsRange, sizeof(int)), "alloc resize range flag");
        NCHK(cudaHostAlloc((void**)&nr.hRsRange, sizeof(int), cudaHostAllocDefault), "alloc resize range readback");
    }
    if (nr.vsr)
    {
        if (!nr.a4kUp && !nr.acnUp) // Anime4K / ArtCNN read and write planar floats, never the bridge's BGRA buffers
        {
            NCHK(cudaMalloc((void**)&nr.dVsrIn, (size_t)vw * vh * 4), "alloc VSR input");
            NCHK(cudaMalloc((void**)&nr.dVsrOut, (size_t)vtw * vth * 4), "alloc VSR output");
        }
        if (nr.hdr || upEnc)
        {
            NCHK(cudaMalloc((void**)&nr.dVsrView, (size_t)3 * vw * vh * sizeof(float)), "alloc VSR SDR view");
            NCHK(cudaMalloc((void**)&nr.dVsrRem, (size_t)3 * vtw * vth * sizeof(float)), "alloc VSR HDR fit");
        }
    }
    // RTX TrueHDR runs once per real frame at the working size, after Restore, DLSS 5 and Sharpen and before the
    // model (nativePreModelPost): its input and output (and the bridge's arrays, which the warm-up makes) are
    // working-size, dSrcG too offline; live dSrcG first carries the capture's SDR planes (k_sdrEncode), so it takes
    // the capture or the working size, whichever is bigger
    const bool workBig = (size_t)nr.w * nr.h > (size_t)nr.cw * nr.ch;
    const int gw = g_offline || workBig ? nr.w : nr.cw, gh = g_offline || workBig ? nr.h : nr.ch;
    if (nr.rtxHdr)
    {
        NCHK(cudaMalloc((void**)&nr.dThdrIn, (size_t)nr.w * nr.h * 4), "alloc TrueHDR input");
        NCHK(cudaMalloc((void**)&nr.dThdrOut, (size_t)nr.w * nr.h * 4), "alloc TrueHDR output");
        NCHK(cudaMalloc((void**)&nr.dSrcG, (size_t)3 * gw * gh * sizeof(float)), "alloc source gamma planes");
        if (g_offline)
        {
            NCHK(cudaMalloc((void**)&nr.dThdrStats, kThdrStatsBytes), "alloc TrueHDR stats");
            NCHK(cudaHostAlloc((void**)&nr.hThdrStats, kThdrStatsBytes, cudaHostAllocDefault),
                 "alloc TrueHDR stats copy");
            if (nr.thdrAcc)
            {
                // the _pq_lut of _accum_hp and its 100-nit code, from the kernels' own EOTF
                float* dLut = nullptr;
                NCHK(cudaMalloc((void**)&dLut, 1024 * sizeof(float)), "alloc PQ table");
                void* al[] = {&dLut};
                const bool lok = cuLaunchKernel(nr.fPqLut, 4, 1, 1, 256, 1, 1, 0, (CUstream)nr.stream, al, nullptr) ==
                                     CUDA_SUCCESS &&
                                 cudaMemcpyAsync(nr.thdrAcc->lut, dLut, sizeof(nr.thdrAcc->lut), cudaMemcpyDeviceToHost,
                                                 nr.stream) == cudaSuccess &&
                                 cudaStreamSynchronize(nr.stream) == cudaSuccess;
                cudaFree(dLut);
                if (!lok)
                {
                    LOG("native: the PQ table failed\n");
                    return false;
                }
                nr.thdrAcc->brightCode = 0;
                while (nr.thdrAcc->brightCode < 1024 && !(nr.thdrAcc->lut[nr.thdrAcc->brightCode] > 0.01f))
                    nr.thdrAcc->brightCode++;
            }
        }
        // the two numerically inverted ICtCp matrices, into the kernel module's globals
        const double rgb2lms[9] = {1688.0 / 4096.0, 2146.0 / 4096.0, 262.0 / 4096.0, 683.0 / 4096.0, 2951.0 / 4096.0,
                                   462.0 / 4096.0,  99.0 / 4096.0,   309.0 / 4096.0, 3688.0 / 4096.0};
        const double lms2ictcp[9] = {2048.0 / 4096.0,  2048.0 / 4096.0,   0.0,
                                     6610.0 / 4096.0,  -13613.0 / 4096.0, 7003.0 / 4096.0,
                                     17933.0 / 4096.0, -17390.0 / 4096.0, -543.0 / 4096.0};
        float lms2rgb[9], ictcp2lms[9];
        if (!inv3d(rgb2lms, lms2rgb) || !inv3d(lms2ictcp, ictcp2lms))
        {
            LOG("native: ICtCp matrix inversion failed\n");
            return false;
        }
        CUdeviceptr gp = 0;
        size_t gsz = 0;
        if (cuModuleGetGlobal(&gp, &gsz, nr.cuMod, "g_lms2rgb") != CUDA_SUCCESS ||
            cuMemcpyHtoD(gp, lms2rgb, sizeof(lms2rgb)) != CUDA_SUCCESS ||
            cuModuleGetGlobal(&gp, &gsz, nr.cuMod, "g_ictcp2lms") != CUDA_SUCCESS ||
            cuMemcpyHtoD(gp, ictcp2lms, sizeof(ictcp2lms)) != CUDA_SUCCESS)
        {
            LOG("native: uploading the ICtCp matrices failed\n");
            return false;
        }
    }
    const bool rtxVsrNow = nr.vsr && !nr.fsrUp && !nr.a4kUp && !nr.acnUp; // NVIDIA's RTX VSR holds the slot
    if (nr.rtxHdr || rtxVsrNow)
    {
        // the bridge retains the PRIMARY context (cuContext = NULL), which is the context this
        // thread just bound with cudaSetDevice. cuStream is NULL:
        // the bridge wraps every eval in host-synchronous cuMemcpy2D calls on the legacy
        // default stream, and the host's own stream is cudaStreamNonBlocking, so an NGX eval
        // placed on it would not be ordered against the bridge's own output copy. ONE create
        // per process (NGX is single-instance), so both features are asked for at once.
        if (g_rtxb.create(nullptr, nullptr, 0, nr.rtxHdr ? 1u : 0u, rtxVsrNow ? 1u : 0u) != 1u)
        {
            LOG("native: rtx_video_api_cuda_create failed\n");
            return false;
        }
        g_rtxb.created = true;
        g_rtxb.vsrMade = rtxVsrNow;
        g_rtxUsed = true; // resident host: NGX is single-instance per process, end it after this session
    }
    // AMD FSR upscaling on HDR planes runs in the bridge's HDR mode (linear light); with RTX HDR on, the frame before
    // the model is the SDR planes (NVIDIA's order: the upscale, then TrueHDR), the SDR mode
    nr.fuHdr = nr.fsrUp && (g_offline ? upEnc != 0 : nr.hdr && !(nr.vsrPre && nr.rtxHdr));
    if (nr.fsrUp && !nativeFsrupSetup(nr, vw, vh, vtw, vth))
        return false;
    if (nr.a4kUp && !nativeA4kSetup(nr, vw, vh, vtw, vth, upEnc))
        return false;
    if (nr.acnUp && !nativeAcnSetup(nr, vw, vh, vtw, vth, upEnc))
        return false;
    if (rtxVsrNow)
    {
        // warm-up eval on an opaque black frame, so the first presented frame pays nothing
        if (cuMemsetD32Async((CUdeviceptr)nr.dVsrIn, 0xFF000000u, (size_t)vw * vh, (CUstream)nr.stream) != CUDA_SUCCESS)
        {
            LOG("native: RTX VSR warm-up clear failed\n");
            return false;
        }
        NCHK(cudaStreamSynchronize(nr.stream), "VSR warm-up stream sync");
        const RtxRect ri{0, 0, (uint32_t)vw, (uint32_t)vh};
        const RtxRect ro{0, 0, (uint32_t)vtw, (uint32_t)vth};
        if (g_rtxb.evalVsr(nr.dVsrIn, nr.dVsrOut, ri, ro, &nr.vsrSet) != 1u)
        {
            LOG("native: the RTX VSR warm-up eval failed\n");
            return false;
        }
        NCHK(cudaDeviceSynchronize(), "VSR warm-up eval sync");
        LOG("native: live upscale: RTX VSR %dx%d -> %dx%d%s%s\n", vw, vh, vtw, vth,
            nr.vsrPre ? ", the captured frame before the model" : "",
            nr.hdr ? ", on the HDR planes' SDR view" : (upEnc ? ", on the HDR video's SDR view" : ""));
    }
    if (nr.rtxHdr)
    {
        // warm-up eval on a zero frame (opaque black RGBA)
        if (cuMemsetD32Async((CUdeviceptr)nr.dThdrIn, 0xFF000000u, (size_t)nr.w * nr.h, (CUstream)nr.stream) !=
            CUDA_SUCCESS)
        {
            LOG("native: TrueHDR warm-up clear failed\n");
            return false;
        }
        NCHK(cudaStreamSynchronize(nr.stream), "warm-up stream sync");
        const RtxRect wr{0, 0, (uint32_t)nr.w, (uint32_t)nr.h};
        if (g_rtxb.evalThdr(nr.dThdrIn, nr.dThdrOut, wr, wr, &nr.thdr) != 1u)
        {
            LOG("native: the TrueHDR warm-up eval failed\n");
            return false;
        }
        NCHK(cudaDeviceSynchronize(), "warm-up eval sync");
        if (g_offline)
            LOG("native: RTX TrueHDR on: %dx%d per decoded frame, after Restore, DLSS 5 and Sharpen and before the "
                "model, colour %ls, contrast %u saturation %u, %u nits\n",
                nr.w, nr.h, g_hdrColor, nr.thdr.Contrast, nr.thdr.Saturation, nr.thdr.MaxLuminance);
        else
            LOG("native: live RTX TrueHDR on: %dx%d per real frame, after Restore, DLSS 5 and Sharpen and before the "
                "model, colour %ls, SDR white %.0f nits, contrast %u saturation %u, %u nits\n",
                nr.w, nr.h, g_hdrColor, nr.sdrScale * 80.0f, nr.thdr.Contrast, nr.thdr.Saturation,
                nr.thdr.MaxLuminance);
    }
    return true;
}
