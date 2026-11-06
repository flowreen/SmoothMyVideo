// part of smv-live.cpp (unity include, see the file map there); not a standalone translation unit
// ---------------------------------------------------------------- WGC capture

// The DLSS-G route in HDR mode. SL's DLSS-G mandates a UINT10/RGB10 backbuffer
// in HDR10/BT.2100 PQ and explicitly rejects FP16 scRGB (ProgrammingGuideDLSS_G.md 11.0), which
// is exactly the server route's present format - but the DLSS-G route has no model host to
// convert, so the exe does it: this compute shader is the same math as the native kernels'
// scrgb_to_pq2020 + R10G10B10A2 pack (709->2020 matrix FIRST, clamp AFTER it - negative scRGB is
// valid wide gamut - scRGB 1.0 = 80 nits over PQ 10000, A = 3). Output is R32_UINT (manual bit
// pack, typed-UAV support guaranteed) whose bit pattern IS R10G10B10A2_UNORM. src is read at `off`: the client crop
// inside the WGC frame read in place, 0 for a copied crop. SLOT = the GPU handoff's variant: the codes stored into the
// R10G10B10A2_UNORM slot itself (code / 1023 and alpha / 3: exact) and the estimate's 1/4 thumbnail written beside it
// (u1) from the same codes: per 4 x 4 block R + 2 G + B over rows and columns 1 and 2, >> 6 (panvec thumbStep10's
// integer sums), through the group's shared sums (a group = 2 x 2 blocks), so no pass reads the slot back
static const char kHdrPackCS[] =
    "Texture2D<float4> src : register(t0);\n"
    "#ifdef SLOT\n"
    "RWTexture2D<unorm float4> dst : register(u0);\n"
    "RWTexture2D<uint> thumb : register(u1);\n"
    "groupshared uint lum[8][8];\n"
    "#else\n"
    "RWTexture2D<uint> dst : register(u0);\n"
    "#endif\n"
    "cbuffer crop : register(b0) { uint2 off; uint2 unused; };\n"
    "uint3 codes(uint2 px)\n"
    "{\n"
    "    float3 c = src[px + off].rgb;\n"
    "    float3 v = float3(dot(float3(0.6274, 0.3293, 0.0433), c),\n"
    "                      dot(float3(0.0691, 0.9195, 0.0114), c),\n"
    "                      dot(float3(0.0164, 0.0880, 0.8956), c));\n"
    "    v = max(v, 0.0) * (80.0 / 10000.0);\n"
    "    float3 lm = pow(v, 0.1593017578125);\n"
    "    float3 p = pow((0.8359375 + 18.8515625 * lm) / (1.0 + 18.6875 * lm), 78.84375);\n"
    "    return (uint3)(saturate(p) * 1023.0 + 0.5);\n"
    "}\n"
    "[numthreads(8,8,1)]\n"
    "void main(uint3 id : SV_DispatchThreadID, uint3 t : SV_GroupThreadID)\n"
    "{\n"
    "    uint w, h;\n"
    "    dst.GetDimensions(w, h);\n"
    "#ifdef SLOT\n"
    "    uint3 q = 0;\n"
    "    if (id.x < w && id.y < h)\n"
    "    {\n"
    "        q = codes(id.xy);\n"
    "        dst[id.xy] = float4(q, 3) / float4(1023.0, 1023.0, 1023.0, 3.0);\n"
    "    }\n"
    "    lum[t.y][t.x] = q.x + 2 * q.y + q.z;\n"
    "    GroupMemoryBarrierWithGroupSync();\n"
    "    if ((t.x & 3) == 0 && (t.y & 3) == 0 && (id.x >> 2) < (w >> 2) && (id.y >> 2) < (h >> 2))\n"
    "        thumb[id.xy >> 2] = (lum[t.y + 1][t.x + 1] + lum[t.y + 1][t.x + 2] + lum[t.y + 2][t.x + 1] +\n"
    "                             lum[t.y + 2][t.x + 2]) >> 6;\n"
    "#else\n"
    "    if (id.x >= w || id.y >= h) return;\n"
    "    uint3 q = codes(id.xy);\n"
    "    dst[id.xy] = q.x | (q.y << 10) | (q.z << 20) | (3u << 30);\n"
    "#endif\n"
    "}\n";

// The direct DLSS-G pass's GPU handoff (Capture::latestFrameShared): the client crop (B8G8R8A8 read as float: exact, one
// value per byte) stored into an R8G8B8A8 slot of the DLSS-G device, the bytes in its RGBA order (the CPU route's swizzle)
static const char kDgSwizzleCS[] = "Texture2D<float4> src : register(t0);\n"
                                   "RWTexture2D<unorm float4> dst : register(u0);\n"
                                   "[numthreads(8,8,1)]\n"
                                   "void main(uint3 id : SV_DispatchThreadID)\n"
                                   "{\n"
                                   "    uint w, h;\n"
                                   "    dst.GetDimensions(w, h);\n"
                                   "    if (id.x >= w || id.y >= h) return;\n"
                                   "    dst[id.xy] = src[id.xy];\n"
                                   "}\n";

// ... and the estimate's 1/4 thumbnail of the SDR crop (panvec thumbRows' integer sums: per 4 x 4 block R + 2 G + B over
// rows and columns 1 and 2, >> 4; R and B either order), read back instead of the frame. The HDR10 route's thumbnail
// comes out of the pack itself (kHdrPackCS with SLOT, from its 10-bit codes)
static const char kDgThumbCS[] =
    "Texture2D<float4> src : register(t0);\n"
    "RWTexture2D<uint> dst : register(u0);\n"
    "[numthreads(8,8,1)]\n"
    "void main(uint3 id : SV_DispatchThreadID)\n"
    "{\n"
    "    uint w, h;\n"
    "    dst.GetDimensions(w, h);\n"
    "    if (id.x >= w || id.y >= h) return;\n"
    "    uint s = 0;\n"
    "    for (uint j = 1; j <= 2; j++)\n"
    "        for (uint i = 1; i <= 2; i++)\n"
    "        {\n"
    "            uint3 c = (uint3)(src[uint2(id.x * 4 + i, id.y * 4 + j)].rgb * 255.0 + 0.5);\n"
    "            s += c.r + 2 * c.g + c.b;\n"
    "        }\n"
    "    dst[id.xy] = s >> 4;\n"
    "}\n";

// Whole-screen capture (Capture::samePicture): any bit that differs between two pictures sets the flag. a = the
// arrival (the WGC frame itself, read at the client crop's offset offA; the copy route: its crop, offA = 0), b = the
// reference (the last kept WGC frame itself, read at the crop's offset offB; a copied reference: its crop, offB = 0),
// size = the crop's. The formats read as float4 (B8G8R8A8 unorm: exact, one value per byte; FP16: exact), compared as
// their bits. A group covers kSameRows rows (four stacked 16 x 16 tiles, a thread four pixels) and a thread that sees
// the flag already set skips its reads: the verdict is decided, so a changed picture stops costing a full read of both.
// The read is a plain one: within a dispatch the flag only goes 0 -> 1 (the clear is ordered before it), so a stale read
// is a 0 and that thread just compares.
constexpr UINT kSameRows = 64;
static const char kSamePictureCS[] =
    "Texture2D<float4> a : register(t0);\n"
    "Texture2D<float4> b : register(t1);\n"
    "RWByteAddressBuffer flag : register(u0);\n"
    "cbuffer crop : register(b0) { uint2 offA; uint2 offB; uint2 size; uint2 unused; };\n"
    "[numthreads(16,16,1)]\n"
    "void main(uint3 g : SV_GroupID, uint3 t : SV_GroupThreadID)\n"
    "{\n"
    "    if (flag.Load(0)) return;\n"
    "    uint x = g.x * 16 + t.x;\n"
    "    if (x >= size.x) return;\n"
    "    bool d = false;\n"
    "    [unroll] for (uint k = 0; k < 4; k++)\n"
    "    {\n"
    "        uint y = g.y * 64 + k * 16 + t.y;\n"
    "        if (y < size.y && any(asuint(a[uint2(x, y) + offA]) != asuint(b[uint2(x, y) + offB]))) d = true;\n"
    "    }\n"
    "    if (d) flag.Store(0, 1u);\n"
    "}\n";

struct Capture
{
    ComPtr<ID3D11Device> dev11;
    ComPtr<ID3D11DeviceContext> ctx11;
    ComPtr<ID3D11Texture2D> staging;
    wgc::GraphicsCaptureItem item{nullptr};
    wgc::Direct3D11CaptureFramePool pool{nullptr};
    wgc::GraphicsCaptureSession session{nullptr};
    HANDLE evt{};              // signaled by FrameArrived
    volatile LONG arrived = 0; // set by FrameArrived, cleared by whoever drains next
    volatile LONG closed = 0;
    uint32_t cw = 0, ch = 0; // CLIENT size = staging/sharedTex size = what the pipeline sees
    // The WGC frame pool captures the WHOLE window (frame bounds), including the title bar
    // and borders. fullW/fullH are that captured size; cropX/cropY + cw/ch select the CLIENT area
    // out of it, so the non-client chrome never reaches the model or the overlay. clientScreenX/Y
    // is the client-area origin in screen coords (where the window-mode overlay must sit).
    uint32_t fullW = 0, fullH = 0; // frame pool size (full window)
    int cropX = 0, cropY = 0;      // client-area offset inside the captured frame
    int clientScreenX = 0, clientScreenY = 0;
    bool swizzle = false; // DLSS-G route only: convert BGRA -> RGBA on readback
    // dlssg HDR: CPU-route conversion chain, FP16 scRGB -> packed R10A2 PQ on
    // the capture GPU before readback (kHdrPackCS above). Set BEFORE init(); readback stays
    // 4 B/px so presentFrame and every buffer size are unchanged. swizzle is ignored on this
    // path (the pack writes the R10A2 bit layout directly).
    bool hdrPack = false;
    ComPtr<ID3D11ComputeShader> convCS;
    ComPtr<ID3D11Texture2D> convSrc, convDst, convStaging;
    ComPtr<ID3D11ShaderResourceView> convSrcView;
    ComPtr<ID3D11UnorderedAccessView> convUav;
    // the pack's read offset (kHdrPackCS's cbuffer crop)
    ComPtr<ID3D11Buffer> convCb;
    // the pack reads the WGC frame itself at the crop's offset on both routes (inPlaceView); SMV_LIVE_INPLACE=0 = the
    // crop copied into convSrc first, =fail = the view's failure (its note, then that route)
    bool hdrInPlace = false, hdrInPlaceFail = false;
    int64_t lastFrameTs = 0; // capture timestamp of the newest drained frame (WGC
                             // SystemRelativeTime, 100ns units in the QPC time domain)
    int newestIn = 0;        // where that frame's client pixels sit: 0 nowhere, 1 staging (latestFrame), 2 sharedTex
                             // (latestFrameGpu); newestToShared hands it to the host
    uint64_t dropped = 0;    // frames superseded in drainNewest (the SATURATION signal:
                             // captured + dropped per window = the true source cadence)
    double emaArrMs = 0;     // smoothed ARRIVAL interval (ms) over every delivered frame,
                             // superseded ones included = the true source cadence. Pacing
                             // must read THIS, not processed-frame intervals: a loop paced
                             // by its own processing rate feeds back into itself and locks
                             // capture into subharmonic plateaus (measured).
    int64_t lastArrTs = 0;   // SystemRelativeTime of the previously delivered frame
    int64_t arrGap = 0;      // 100 ns between the two newest arrivals (the source's own cadence, however slow the
                             // pipeline runs: the static-source hold keys on it)
    // The pool has two buffers and WGC discards, uncounted, every frame that arrives while both
    // are held. So FrameArrived takes each frame off the pool at once (takeFrames) and keeps only
    // the newest here until drainNewest hands it out: a host slower than the source then sees
    // every arrival, `dropped` counts the superseded ones and emaArrMs reads the source's cadence
    // (with the pool alone, a host at 3.5 pairs a second read a 24 fps source as 3.5 fps and 1
    // drop a second, and its throttle never fired).
    SRWLOCK lk = SRWLOCK_INIT;
    wgc::Direct3D11CaptureFrame held{nullptr};
    // the loop's frame (drainNewest hands it out, release() takes it back). A frame goes back to the pool only once it
    // is none of held, the loop's and the unchanged-picture test's reference (refFrame below): closeFree, under lk
    wgc::Direct3D11CaptureFrame loopFrame{nullptr};
    bool stopping = false; // set under lk by stop(): a late FrameArrived leaves the pool alone
    // An arrival whose picture equals the last new one is no source frame. Whole screen: the overlay is excluded from
    // the picture, but each of its presents still composes the monitor and WGC hands that composition over as a frame,
    // so without the test the loop takes its own presents for the source (245 arrivals a second on a static screen at
    // the 1 ms interval, ~45 at Windows' default). Window and Fill: a display-sync player (mpv's display-resample)
    // presents the same picture every refresh, so a 24 fps video reads as 60 to 124 and each new picture's motion lands
    // in one refresh. samePicture tests every arrival's client picture on FrameArrived's thread (the cadence keeps
    // counting arrivals, not the loop's drains) and takeFrames drops an unchanged one uncounted. An unchanged arrival
    // still counts once kSameKeepalive has passed since the last kept one: the static hold's refresh repaint brings
    // the current picture after a model load or a pause, and DLSS-G's input-gap reset (700 ms) never fires on a
    // source that keeps presenting. The test reads the arrival in place (a view of the WGC frame at the crop's offset)
    // against the reference = the last kept WGC frame itself, held open (refFrame, its view refSrv) until the next new
    // picture's test has read it: a new picture costs no reference copy and no reference texture, an unchanged arrival
    // the compare alone. A frame no view can read, and the copy route (SMV_WGC_SAME=copy), keep the reference as a
    // copy of the crop instead (sameRefTex, made on first use). The immediate context is multithread-protected while
    // this runs (`mt`; its compute-shader sequences hold mt->Enter()).
    static constexpr int64_t kSameKeepalive = 5000000; // 100 ns units: 500 ms
    bool allowSame = true; // a caller that wants every arrival (the probe) clears it before init()
    bool sameSkip = false;
    ComPtr<ID3D11Multithread> mt;
    ComPtr<ID3D11ComputeShader> sameCS;
    wgc::Direct3D11CaptureFrame refFrame{nullptr};  // the reference when it is a held frame (read at the crop's offset)
    ComPtr<ID3D11ShaderResourceView> refSrv;        // its view
    ComPtr<ID3D11Texture2D> sameRefTex, sameTmpTex; // the reference as a copy of the crop; the copy route's arrival
    ComPtr<ID3D11ShaderResourceView> sameRefSrv, sameTmpSrv;
    DXGI_FORMAT sameFmt = DXGI_FORMAT_UNKNOWN;            // the pool's format (the two textures, the frame check)
    ComPtr<ID3D11Buffer> sameFlag, sameFlagStage, sameCb; // sameCb = the shader's offsets and size
    ComPtr<ID3D11UnorderedAccessView> sameUav;
    bool sameCopy =
        false; // the copy route: the arrival's crop copied into sameTmpTex, then compared (SMV_WGC_SAME=copy;
               // a frame no view can read takes it too)
    bool sameViewFail = false; // a frame no view could read (logged once a session)
    bool sameRef = false;      // a reference exists: refFrame, else sameRefTex
    uint64_t sameN = 0;        // arrivals dropped as unchanged
    uint64_t sameKept = 0;     // unchanged arrivals kept by kSameKeepalive
    // the HUD meter's new pictures: each kept arrival's capture stamp (100 ns) unless its test found it unchanged (a
    // keepalive arrival is tested too, for this alone: it goes through whatever it shows), the newest kArrN, under lk
    static constexpr uint32_t kArrN = 1024;
    int64_t arrRing[kArrN]{};
    uint64_t arrCount = 0;
    // the HUD meter's presents: every arrival's stamp, an unchanged picture included (a window capture hands over each
    // present of the target), but for the repaints our refresh requests draw (notePresent), the newest kArrN, under lk
    int64_t presRing[kArrN]{};
    uint64_t presCount = 0;
    int64_t refreshAskTs = 0; // the newest requestRefresh's time (QPC 100 ns), under lk
    int64_t askRing[64]{};    // the newest requests' times (SMV_LAT_TRACE's Q rows), under lk
    uint64_t askCount = 0;
    bool arrNew = true; // samePicture's verdict on the arrival it just tested
    double sameMsSum = 0, sameMsMax = 0;
    uint64_t sameTests = 0;
    SRWLOCK sameLk = SRWLOCK_INIT; // one takeFrames at a time while sameSkip (FrameArrived's), before lk
    // Zero-copy capture interop (server route): frames are GPU-copied into a SHARED texture
    // the server imports as CUDA external memory, with a shared D3D11 fence for
    // ordering. Capture never touches the CPU: no staging Map, no shm memcpy, no H2D upload.
    ComPtr<ID3D11Texture2D> sharedTex;
    ComPtr<ID3D11Fence> sharedFence;
    ComPtr<ID3D11DeviceContext4> ctx4;
    HANDLE hTex = nullptr, hFence = nullptr; // INHERITABLE NT handles (same values in the child)
    bool interop = false;
    HWND targetWnd = nullptr; // capture target (monitor mode: the seeding window), for requestRefresh
    // THE GPU HANDOFF (the direct DLSS-G pass; the loop sets it up, SMV_LIVE_DGPU=0 = off, =fail = its fallback): a
    // frame reaches the DLSS-G device on the GPU, stored into one of two shared slots in the route's format (SDR: the
    // crop as RGBA, kDgSwizzleCS; HDR10: the pack itself, kHdrPackCS with SLOT), instead of through the CPU and an upload
    // copy; the estimate reads back its 1/4 thumbnail (kDgThumbCS of the crop; HDR10: the pack writes it beside the
    // slot from its own codes, ~1/64 of the frame) instead of the frame. dgWrote = this device's fence (a slot written),
    // dgCopied = the DLSS-G device's (a slot copied): a
    // slot is rewritten only once its last frame's copy ran (dgFree[slot], set by the loop from the present). HDR10's
    // pack reads the WGC frame in place (hdrInPlace, inPlaceView)
    ComPtr<ID3D11ComputeShader> dgCS, dgThumbCS;
    ComPtr<ID3D11Texture2D> dgSrc, dgTex[2], dgThumb, dgThumbStage;
    ComPtr<ID3D11ShaderResourceView> dgSrcView;
    ComPtr<ID3D11UnorderedAccessView> dgUav[2], dgThumbUav;
    ComPtr<ID3D11Fence> dgWrote, dgCopied;
    HANDLE dgTexH[2]{}, dgWroteH{};
    uint64_t dgWroteVal = 0, dgFree[2]{};
    int dgSlot = 0;

    bool dgOn() const
    {
        return dgCopied != nullptr;
    }

    // this device's half of the handoff (the slots, their handles, dgWrote); false = unavailable, released again
    bool dgInit()
    {
        dgRelease();
        ComPtr<ID3D11Device5> dev5;
        ComPtr<ID3DBlob> cs, ts, err;
        const char* sw = hdrPack ? kHdrPackCS : kDgSwizzleCS;
        const D3D_SHADER_MACRO slot[] = {{"SLOT", "1"}, {nullptr, nullptr}};
        // SDR's thumbnail pass; HDR10's thumbnail is the pack's own write
        if (cw < 4 || ch < 4 || FAILED(dev11.As(&dev5)) || (!ctx4 && FAILED(ctx11.As(&ctx4))) ||
            FAILED(D3DCompile(sw, strlen(sw), nullptr, hdrPack ? slot : nullptr, nullptr, "main", "cs_5_0", 0, 0, &cs,
                              &err)) ||
            FAILED(dev11->CreateComputeShader(cs->GetBufferPointer(), cs->GetBufferSize(), nullptr, &dgCS)) ||
            (!hdrPack &&
             (FAILED(D3DCompile(kDgThumbCS, strlen(kDgThumbCS), nullptr, nullptr, nullptr, "main", "cs_5_0", 0, 0, &ts,
                                &err)) ||
              FAILED(dev11->CreateComputeShader(ts->GetBufferPointer(), ts->GetBufferSize(), nullptr, &dgThumbCS)))))
            return dgFail();
        D3D11_TEXTURE2D_DESC td{};
        td.Width = cw / 4;
        td.Height = ch / 4;
        td.MipLevels = 1;
        td.ArraySize = 1;
        td.SampleDesc = {1, 0};
        td.Format = DXGI_FORMAT_R8_UINT;
        td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
        if (FAILED(dev11->CreateTexture2D(&td, nullptr, &dgThumb)) ||
            FAILED(dev11->CreateUnorderedAccessView(dgThumb.Get(), nullptr, &dgThumbUav)))
            return dgFail();
        td.Usage = D3D11_USAGE_STAGING;
        td.BindFlags = 0;
        td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        if (FAILED(dev11->CreateTexture2D(&td, nullptr, &dgThumbStage)))
            return dgFail();
        td = D3D11_TEXTURE2D_DESC{};
        td.Width = cw;
        td.Height = ch;
        td.MipLevels = 1;
        td.ArraySize = 1;
        td.SampleDesc = {1, 0};
        td.Usage = D3D11_USAGE_DEFAULT;
        td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        if (!hdrPack && (FAILED(dev11->CreateTexture2D(&td, nullptr, &dgSrc)) ||
                         FAILED(dev11->CreateShaderResourceView(dgSrc.Get(), nullptr, &dgSrcView))))
            return dgFail(); // HDR10 packs from the frame (or convSrc) into the slot instead
        td.Format = hdrPack ? DXGI_FORMAT_R10G10B10A2_UNORM : DXGI_FORMAT_R8G8B8A8_UNORM;
        td.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
        td.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
        for (int i = 0; i < 2; i++)
        {
            ComPtr<IDXGIResource1> r1;
            if (FAILED(dev11->CreateTexture2D(&td, nullptr, &dgTex[i])) ||
                FAILED(dev11->CreateUnorderedAccessView(dgTex[i].Get(), nullptr, &dgUav[i])) ||
                FAILED(dgTex[i].As(&r1)) || FAILED(r1->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &dgTexH[i])))
                return dgFail();
        }
        if (FAILED(dev5->CreateFence(0, D3D11_FENCE_FLAG_SHARED, IID_PPV_ARGS(&dgWrote))) ||
            FAILED(dgWrote->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &dgWroteH)))
            return dgFail();
        return true;
    }

    // the DLSS-G device's dgCopied, from its handle: the handoff is on once this opens
    bool dgOpenCopied(HANDLE h)
    {
        ComPtr<ID3D11Device5> dev5;
        if (FAILED(dev11.As(&dev5)) || FAILED(dev5->OpenSharedFence(h, IID_PPV_ARGS(&dgCopied))))
            return dgFail();
        return true;
    }

    bool dgFail()
    {
        dgRelease();
        return false;
    }

    void dgRelease()
    {
        dgCS.Reset();
        dgThumbCS.Reset();
        dgThumb.Reset();
        dgThumbStage.Reset();
        dgThumbUav.Reset();
        dgSrc.Reset();
        dgSrcView.Reset();
        dgWrote.Reset();
        dgCopied.Reset();
        for (int i = 0; i < 2; i++)
        {
            dgTex[i].Reset();
            dgUav[i].Reset();
            if (dgTexH[i])
                CloseHandle(dgTexH[i]);
            dgTexH[i] = nullptr;
            dgFree[i] = 0;
        }
        if (dgWroteH)
            CloseHandle(dgWroteH);
        dgWroteH = nullptr;
        dgWroteVal = 0;
        dgSlot = 0;
    }

    void initInterop()
    {
        ComPtr<ID3D11Device5> dev5;
        if (FAILED(dev11.As(&dev5)) || FAILED(ctx11.As(&ctx4)))
        {
            LOG("capture interop: D3D11.4 unavailable\n");
            return;
        }
        D3D11_TEXTURE2D_DESC td{};
        td.Width = cw;
        td.Height = ch;
        td.MipLevels = 1;
        td.ArraySize = 1;
        td.Format = g_hdr ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_B8G8R8A8_UNORM;
        td.SampleDesc = {1, 0};
        td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
        td.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
        if (FAILED(dev11->CreateTexture2D(&td, nullptr, &sharedTex)))
        {
            LOG("capture interop: shared texture creation failed\n");
            return;
        }
        ComPtr<IDXGIResource1> res1;
        if (FAILED(sharedTex.As(&res1)))
            return;
        SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE}; // inheritable: value survives into the child
        if (FAILED(res1->CreateSharedHandle(&sa, GENERIC_ALL, nullptr, &hTex)))
        {
            LOG("capture interop: CreateSharedHandle(texture) failed\n");
            return;
        }
        if (FAILED(dev5->CreateFence(0, D3D11_FENCE_FLAG_SHARED, IID_PPV_ARGS(&sharedFence))) ||
            FAILED(sharedFence->CreateSharedHandle(&sa, GENERIC_ALL, nullptr, &hFence)))
        {
            LOG("capture interop: shared fence creation failed\n");
            return;
        }
        interop = true;
        LOG("capture interop: shared texture + fence ready\n");
    }

    // dlssg HDR: compile the pack shader and create the conversion chain (cw x ch known by
    // now). Failure is graceful: caller clears g_hdr and the session runs SDR with the clip
    // notice.
    bool initHdrPack()
    {
        ComPtr<ID3DBlob> cs, err;
        if (FAILED(D3DCompile(kHdrPackCS, sizeof(kHdrPackCS) - 1, nullptr, nullptr, nullptr, "main", "cs_5_0", 0, 0,
                              &cs, &err)))
        {
            LOG("HDR pack shader compile failed: %s\n", err ? (const char*)err->GetBufferPointer() : "?");
            return false;
        }
        if (FAILED(dev11->CreateComputeShader(cs->GetBufferPointer(), cs->GetBufferSize(), nullptr, &convCS)))
            return false;
        D3D11_TEXTURE2D_DESC td{};
        td.Width = cw;
        td.Height = ch;
        td.MipLevels = 1;
        td.ArraySize = 1;
        td.SampleDesc = {1, 0};
        td.Usage = D3D11_USAGE_DEFAULT;
        td.Format = DXGI_FORMAT_R32_UINT;
        td.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
        if (FAILED(dev11->CreateTexture2D(&td, nullptr, &convDst)))
            return false;
        td.BindFlags = 0;
        td.Usage = D3D11_USAGE_STAGING;
        td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        if (FAILED(dev11->CreateTexture2D(&td, nullptr, &convStaging)))
            return false;
        if (FAILED(dev11->CreateUnorderedAccessView(convDst.Get(), nullptr, &convUav)))
            return false;
        D3D11_BUFFER_DESC bd{};
        bd.ByteWidth = 16;
        bd.Usage = D3D11_USAGE_DEFAULT;
        bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        const UINT zero[4]{};
        const D3D11_SUBRESOURCE_DATA sd{zero, 0, 0};
        if (FAILED(dev11->CreateBuffer(&bd, &sd, &convCb)))
            return false;
        LOG("dlssg HDR: exe-side scRGB -> PQ R10A2 pack ready (%ux%u)\n", cw, ch);
        return true;
    }

    // convSrc, the crop's FP16 copy the HDR pack reads when it cannot read the frame in place (inPlaceView): made by
    // the first frame that copies, so a session reading in place never holds it; false = it cannot be made
    bool convSrcReady()
    {
        if (convSrc)
            return true;
        D3D11_TEXTURE2D_DESC td{};
        td.Width = cw;
        td.Height = ch;
        td.MipLevels = 1;
        td.ArraySize = 1;
        td.SampleDesc = {1, 0};
        td.Usage = D3D11_USAGE_DEFAULT;
        td.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        if (FAILED(dev11->CreateTexture2D(&td, nullptr, &convSrc)) ||
            FAILED(dev11->CreateShaderResourceView(convSrc.Get(), nullptr, &convSrcView)))
        {
            convSrc.Reset();
            convSrcView.Reset();
            LOG("HDR pack: the crop's copy texture (%ux%u FP16) cannot be made\n", cw, ch);
            return false;
        }
        return true;
    }

    // a view of the WGC frame for the HDR pack to read at the crop's offset; null = the crop is copied into convSrc
    // first: in place off, a frame smaller than the crop, or the view's failure (its note once, then every frame)
    ComPtr<ID3D11ShaderResourceView> inPlaceView(ID3D11Texture2D* tex)
    {
        ComPtr<ID3D11ShaderResourceView> v;
        if (!hdrInPlace)
            return v;
        D3D11_TEXTURE2D_DESC fd{};
        tex->GetDesc(&fd);
        if (fd.Width < (UINT)cropX + cw || fd.Height < (UINT)cropY + ch)
            return v;
        if (hdrInPlaceFail || FAILED(dev11->CreateShaderResourceView(tex, nullptr, &v)))
        {
            LOG("note: the frame cannot be read in place (format %d, bind 0x%x): each frame's crop is copied "
                "before the HDR pack\n",
                (int)fd.Format, fd.BindFlags);
            hdrInPlace = false;
            v = nullptr;
        }
        return v;
    }

    // a crop-sized texture in the pool's format and its view (the copied reference, the copy route's arrival), made on
    // first use; false = it cannot be made (both left null)
    bool sameTexMake(ComPtr<ID3D11Texture2D>& t, ComPtr<ID3D11ShaderResourceView>& v)
    {
        D3D11_TEXTURE2D_DESC td{};
        td.Width = cw;
        td.Height = ch;
        td.MipLevels = 1;
        td.ArraySize = 1;
        td.Format = sameFmt;
        td.SampleDesc = {1, 0};
        td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        if (SUCCEEDED(dev11->CreateTexture2D(&td, nullptr, &t)) &&
            SUCCEEDED(dev11->CreateShaderResourceView(t.Get(), nullptr, &v)))
            return true;
        t.Reset();
        v.Reset();
        return false;
    }

    // the unchanged-picture test (samePicture) at the client crop's size and the pool's format; false = not available
    // (a whole-screen caller then keeps Windows' update interval). SMV_WGC_SAME=fail = this failure's trigger,
    // SMV_WGC_SAME=copy = the copy route on every arrival
    bool initSame(DXGI_FORMAT fmt)
    {
        wchar_t lv[8]{};
        if (GetEnvironmentVariableW(L"SMV_WGC_SAME", lv, 8) && wcscmp(lv, L"fail") == 0)
            return false;
        sameCopy = wcscmp(lv, L"copy") == 0;
        sameFmt = fmt;
        ComPtr<ID3DBlob> cs, err;
        if (FAILED(D3DCompile(kSamePictureCS, sizeof(kSamePictureCS) - 1, nullptr, nullptr, nullptr, "main", "cs_5_0",
                              0, 0, &cs, &err)) ||
            FAILED(dev11->CreateComputeShader(cs->GetBufferPointer(), cs->GetBufferSize(), nullptr, &sameCS)) ||
            (sameCopy && !sameTexMake(sameTmpTex, sameTmpSrv)))
            return false;
        D3D11_BUFFER_DESC bd{};
        bd.ByteWidth = 32;
        bd.Usage = D3D11_USAGE_DEFAULT;
        bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        if (FAILED(dev11->CreateBuffer(&bd, nullptr, &sameCb)))
            return false;
        bd = {};
        bd.ByteWidth = 16;
        bd.Usage = D3D11_USAGE_DEFAULT;
        bd.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
        bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
        if (FAILED(dev11->CreateBuffer(&bd, nullptr, &sameFlag)))
            return false;
        D3D11_UNORDERED_ACCESS_VIEW_DESC ud{};
        ud.Format = DXGI_FORMAT_R32_TYPELESS;
        ud.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        ud.Buffer.NumElements = 4;
        ud.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
        if (FAILED(dev11->CreateUnorderedAccessView(sameFlag.Get(), &ud, &sameUav)))
            return false;
        bd.Usage = D3D11_USAGE_STAGING;
        bd.BindFlags = 0;
        bd.MiscFlags = 0;
        bd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        if (FAILED(dev11->CreateBuffer(&bd, nullptr, &sameFlagStage)) || FAILED(ctx11.As(&mt)))
            return false;
        mt->SetMultithreadProtected(TRUE);
        return true;
    }

    // true = this arrival's client picture equals the reference; the flag's readback waits for the test's reads, so an
    // unchanged frame may be closed right after. A new picture becomes the reference: `view` = its in-place view, and
    // the caller holds the frame open (refFrame) until the next new picture's test has read it; no view = its crop was
    // copied into sameRefTex, queued after the test and not waited for (the copy route, or a frame no view can read):
    // takeFrames closes such a frame only once a later arrival's test (whose readback waits for everything queued
    // before it) has run, or after the loop's own copy of it. A resize or a caption-sized frame drops the reference:
    // the next picture starts over
    bool samePicture(wgc::Direct3D11CaptureFrame const& f, ComPtr<ID3D11ShaderResourceView>& view)
    {
        view = nullptr;
        arrNew = true;
        winrt::com_ptr<ID3D11Texture2D> tex;
        int64_t ts = 0;
        try
        {
            const auto csz = f.ContentSize();
            ts = f.SystemRelativeTime().count();
            auto access = f.Surface().as<Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>();
            if ((uint32_t)csz.Width != fullW || (uint32_t)csz.Height != fullH ||
                FAILED(access->GetInterface(winrt::guid_of<ID3D11Texture2D>(), tex.put_void())))
                tex = nullptr;
        }
        catch (...)
        {
            tex = nullptr;
        }
        D3D11_TEXTURE2D_DESC fd{};
        if (tex)
            tex->GetDesc(&fd);
        if (!tex || fd.Width < (UINT)cropX + cw || fd.Height < (UINT)cropY + ch || fd.Format != sameFmt)
        {
            sameRef = false; // a resize or a caption-sized frame: the next picture starts over
            return false;
        }
        const bool keepalive = sameRef && lastArrTs && ts - lastArrTs >= kSameKeepalive;
        const int64_t t0 = nowQpc100();
        bool same = false;
        mt->Enter();
        const D3D11_BOX box{(UINT)cropX, (UINT)cropY, 0, (UINT)cropX + cw, (UINT)cropY + ch, 1};
        // the arrival as the test reads it, and as the reference once it is new: a view of the frame itself at the
        // crop's offset, else (the copy route) its crop copied first; no texture for that = no test, the arrival counts
        // as new. The reference: the held frame at the crop's offset, else its copy at 0
        ComPtr<ID3D11ShaderResourceView> arr;
        const bool refHeld = refFrame != nullptr;
        UINT off[8]{(UINT)cropX, (UINT)cropY, refHeld ? (UINT)cropX : 0u, refHeld ? (UINT)cropY : 0u, cw, ch, 0, 0};
        const bool test = sameRef;
        if (!sameCopy && FAILED(dev11->CreateShaderResourceView(tex.get(), nullptr, &arr)) && !sameViewFail)
        {
            sameViewFail = true;
            LOG("capture: the frame cannot be read in place (format %d, bind 0x%x): each arrival is copied before its "
                "unchanged-picture test\n",
                (int)fd.Format, fd.BindFlags);
        }
        if (test && !arr && (sameTmpTex || sameTexMake(sameTmpTex, sameTmpSrv)))
        {
            ctx11->CopySubresourceRegion(sameTmpTex.Get(), 0, 0, 0, 0, tex.get(), 0, &box);
            off[0] = off[1] = 0;
        }
        if (test && (arr || sameTmpTex))
        {
            ctx11->UpdateSubresource(sameCb.Get(), 0, nullptr, off, 0, 0);
            const UINT zero[4]{};
            ctx11->ClearUnorderedAccessViewUint(sameUav.Get(), zero);
            ctx11->CSSetShader(sameCS.Get(), nullptr, 0);
            ID3D11ShaderResourceView* srv[2]{arr ? arr.Get() : sameTmpSrv.Get(),
                                             refHeld ? refSrv.Get() : sameRefSrv.Get()};
            ctx11->CSSetShaderResources(0, 2, srv);
            ID3D11UnorderedAccessView* uav = sameUav.Get();
            ctx11->CSSetUnorderedAccessViews(0, 1, &uav, nullptr);
            ID3D11Buffer* cb = sameCb.Get();
            ctx11->CSSetConstantBuffers(0, 1, &cb);
            ctx11->Dispatch((cw + 15) / 16, (ch + kSameRows - 1) / kSameRows, 1);
            ID3D11ShaderResourceView* none[2]{};
            ctx11->CSSetShaderResources(0, 2, none);
            uav = nullptr;
            ctx11->CSSetUnorderedAccessViews(0, 1, &uav, nullptr);
            cb = nullptr;
            ctx11->CSSetConstantBuffers(0, 1, &cb);
            ctx11->CopyResource(sameFlagStage.Get(), sameFlag.Get());
            // the Map waits with the context locked, so the loop's next capture copy can queue behind one test; an
            // unlocked wait on a fence event was slower (1.7 ms mean, 11.5 max vs 0.87 / 4.5 on a 270/s display-resample
            // source) and WGC discarded real pictures meanwhile (22.6 instead of 24 a second taken)
            D3D11_MAPPED_SUBRESOURCE m{};
            if (SUCCEEDED(ctx11->Map(sameFlagStage.Get(), 0, D3D11_MAP_READ, 0, &m)))
            {
                same = *(const uint32_t*)m.pData == 0;
                ctx11->Unmap(sameFlagStage.Get(), 0);
            }
            const double ms = (nowQpc100() - t0) / 10000.0;
            sameMsSum += ms;
            sameMsMax = (std::max)(sameMsMax, ms);
            sameTests++;
        }
        arrNew = !same;
        if (keepalive)
        {
            sameKept++;
            same = false; // the static hold's arrival goes through whatever it shows
        }
        if (!same)
        {
            if (arr)
                view = arr;
            else if (sameRefTex || sameTexMake(sameRefTex, sameRefSrv))
                ctx11->CopySubresourceRegion(sameRefTex.Get(), 0, 0, 0, 0, tex.get(), 0, &box);
        }
        mt->Leave();
        if (!same)
            sameRef = view != nullptr || sameRefTex != nullptr;
        return same;
    }

    int init(HWND target, IDXGIAdapter1* adapter)
    {
        // resident host: this object is static and re-entered once per session, so every
        // flag the previous session left behind is cleared here (a stale `closed` ended the
        // next session on its first loop iteration, a stale `interop` handed out dead handles)
        closed = 0;
        arrived = 0;
        dropped = 0;
        arrCount = 0;
        presCount = 0;
        refreshAskTs = 0;
        askCount = 0;
        emaArrMs = 0;
        lastArrTs = 0;
        arrGap = 0;
        lastFrameTs = 0;
        newestIn = 0;
        held = nullptr;
        loopFrame = nullptr;
        refFrame = nullptr;
        refSrv.Reset();
        dgRelease();
        stopping = false;
        interop = false;
        sharedTex.Reset();
        sharedFence.Reset();
        ctx4.Reset();
        staging.Reset();
        convCS.Reset();
        convSrc.Reset();
        convDst.Reset();
        convStaging.Reset();
        convSrcView.Reset();
        convUav.Reset();
        convCb.Reset();
        sameSkip = false;
        sameRef = false;
        sameFmt = DXGI_FORMAT_UNKNOWN;
        sameCopy = sameViewFail = false;
        sameN = sameTests = sameKept = 0;
        sameMsSum = sameMsMax = 0;
        mt.Reset();
        sameCS.Reset();
        sameFlag.Reset();
        sameFlagStage.Reset();
        sameUav.Reset();
        sameCb.Reset();
        sameRefTex.Reset();
        sameRefSrv.Reset();
        sameTmpTex.Reset();
        sameTmpSrv.Reset();
        targetWnd = target;
        evt = CreateEventW(nullptr, FALSE, FALSE, nullptr);

        // create the capture device via the REAL d3d11.dll: linking sl.interposer.lib makes the
        // import D3D11CreateDevice resolve to SL's proxy, which "automatically assigns" this
        // device to Streamline and breaks DLSS-G's pacer (presents stop generating; verified:
        // numFramesActuallyPresented stayed 1 until this bypass)
        HMODULE d3d11 = LoadLibraryExW(L"C:\\Windows\\System32\\d3d11.dll", nullptr, 0);
        if (!d3d11)
        {
            LOG("LoadLibrary d3d11.dll failed\n");
            return 1;
        }
        auto realCreate = (PFN_D3D11_CREATE_DEVICE)GetProcAddress(d3d11, "D3D11CreateDevice");
        if (!realCreate)
        {
            LOG("GetProcAddress D3D11CreateDevice failed\n");
            return 1;
        }
        CHECK_HR(realCreate(adapter, D3D_DRIVER_TYPE_UNKNOWN, nullptr, D3D11_CREATE_DEVICE_BGRA_SUPPORT, nullptr, 0,
                            D3D11_SDK_VERSION, &dev11, nullptr, &ctx11));
        ComPtr<IDXGIDevice> dxgiDev;
        CHECK_HR(dev11.As(&dxgiDev));
        winrt::com_ptr<::IInspectable> insp;
        CHECK_HR(CreateDirect3D11DeviceFromDXGIDevice(dxgiDev.Get(), insp.put()));
        auto rtDev = insp.as<wgdx::Direct3D11::IDirect3DDevice>();

        auto itemInterop = winrt::get_activation_factory<wgc::GraphicsCaptureItem>().as<IGraphicsCaptureItemInterop>();
        HRESULT hr =
            g_monitor ? itemInterop->CreateForMonitor(MonitorFromWindow(target, MONITOR_DEFAULTTONEAREST),
                                                      winrt::guid_of<wgc::GraphicsCaptureItem>(), winrt::put_abi(item))
                      : itemInterop->CreateForWindow(target, winrt::guid_of<wgc::GraphicsCaptureItem>(),
                                                     winrt::put_abi(item));
        if (FAILED(hr))
        {
            LOG("%s failed hr=0x%08lx\n", g_monitor ? "CreateForMonitor" : "CreateForWindow", hr);
            return 1;
        }

        auto sz = item.Size();
        fullW = (uint32_t)sz.Width;
        fullH = (uint32_t)sz.Height;
        cw = fullW;
        ch = fullH;
        cropX = cropY = 0;
        clientScreenX = clientScreenY = 0;
        // For a window target, crop the capture to the client area (drop the title bar and
        // borders). The captured frame's (0,0) is the DWM extended frame-bounds origin; the client
        // area starts below the caption. The process is per-monitor-DPI-aware (set at startup), so
        // every rect below is in physical pixels and no scaling is needed. Monitor capture has no
        // client concept, so it is left full-size.
        if (!g_monitor)
        {
            RECT cr{}, fbr{};
            POINT tl{0, 0};
            DwmGetWindowAttribute(target, DWMWA_EXTENDED_FRAME_BOUNDS, &fbr, sizeof(fbr));
            if (GetClientRect(target, &cr) && ClientToScreen(target, &tl))
            {
                int cx = tl.x - fbr.left;
                int cy = tl.y - fbr.top;
                int clw = cr.right - cr.left;
                int clh = cr.bottom - cr.top;
                if (cx < 0)
                    cx = 0;
                if (cy < 0)
                    cy = 0;
                if (clw > (int)fullW - cx)
                    clw = (int)fullW - cx; // maximized: client can exceed bounds
                if (clh > (int)fullH - cy)
                    clh = (int)fullH - cy;
                if (clw > 0 && clh > 0)
                {
                    cropX = cx;
                    cropY = cy;
                    cw = (uint32_t)clw;
                    ch = (uint32_t)clh;
                    clientScreenX = tl.x;
                    clientScreenY = tl.y;
                    if (cropX || cropY || cw != fullW || ch != fullH)
                        LOG("client crop: window %ux%u -> client %ux%u at offset (%d,%d)\n", fullW, fullH, cw, ch,
                            cropX, cropY);
                }
            }
        }

        // dlssg HDR converts in the exe. Must happen before the pool creation
        // below (its format reads g_hdr) so a failed setup can drop the session to SDR.
        if (hdrPack && !initHdrPack())
        {
            LOG("dlssg HDR conversion unavailable, falling back to SDR\n");
            hdrPack = false;
            g_hdr = false;
        }
        {
            wchar_t iv[8]{};
            const bool ivSet = GetEnvironmentVariableW(L"SMV_LIVE_INPLACE", iv, 8) > 0;
            hdrInPlace = hdrPack && !(ivSet && wcscmp(iv, L"0") == 0);
            hdrInPlaceFail = ivSet && wcscmp(iv, L"fail") == 0;
        }

        // the unchanged-picture test (sameSkip), every fit; SMV_WGC_SAME=0 = off (A/B)
        if (allowSame)
        {
            wchar_t lv[8]{};
            if (!(GetEnvironmentVariableW(L"SMV_WGC_SAME", lv, 8) && lv[0] == L'0'))
            {
                sameSkip = initSame(g_hdr ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_B8G8R8A8_UNORM);
                if (sameSkip)
                    LOG("capture: arrivals whose picture is unchanged are dropped (%s)%s\n",
                        g_monitor ? "the overlay's own presents" : "a player's repeated presents",
                        sameCopy ? "; each arrival copied before its test (SMV_WGC_SAME=copy)" : "");
                else
                    LOG(g_monitor ? "note: the whole screen's unchanged-picture test is unavailable: the capture keeps "
                                    "Windows' update interval (60 fps at most)\n"
                                  : "note: the unchanged-picture test is unavailable: a player's repeated presents "
                                    "count as source frames\n");
            }
        }

        // HDR mode captures FP16 scRGB (linear, 709 primaries, 1.0 = 80 nits); the server
        // re-encodes to BT.2020 PQ. SDR stays 8-bit BGRA. The unchanged-picture test holds a frame while it runs:
        // a third buffer keeps WGC from discarding the next arrival meanwhile (its reference is a held frame too, the
        // one the loop's release would free)
        pool = wgc::Direct3D11CaptureFramePool::CreateFreeThreaded(
            rtDev,
            g_hdr ? wgdx::DirectXPixelFormat::R16G16B16A16Float : wgdx::DirectXPixelFormat::B8G8R8A8UIntNormalized,
            sameSkip ? 3 : 2, sz);
        session = pool.CreateCaptureSession(item);
        try
        {
            session.IsCursorCaptureEnabled(false);
        }
        catch (...)
        {
            LOG("note: cursor exclusion unavailable, cursor will be baked into capture\n");
        }
        try
        {
            // yellow-border suppression (Win11); best effort
            wgc::GraphicsCaptureAccess::RequestAccessAsync(wgc::GraphicsCaptureAccessKind::Borderless).get();
            session.IsBorderRequired(false);
        }
        catch (...)
        {
            LOG("note: capture border suppression unavailable (pre-Win11?)\n");
        }
        try
        {
            // Windows holds a capture at 60 frames a second by default (the interval reads 16 ms) whatever
            // the source presents; 1 ms passes every composed frame. A whole-screen capture without the
            // unchanged-picture test keeps the default: at 1 ms each of the overlay's own presents arrives as a
            // frame (sameSkip above). SMV_WGC_INTERVAL=0 keeps Windows' default (A/B), =fail takes the
            // fallback below (its trigger)
            wchar_t iv[8]{};
            const bool lever = GetEnvironmentVariableW(L"SMV_WGC_INTERVAL", iv, 8) > 0;
            if (lever && wcscmp(iv, L"fail") == 0)
                throw winrt::hresult_no_interface();
            if (!(lever && iv[0] == L'0') && (!g_monitor || sameSkip))
                session.MinUpdateInterval(winrt::Windows::Foundation::TimeSpan{10000});
        }
        catch (...)
        {
            LOG("note: capture update interval unavailable on this Windows: a window captures at 60 fps at most\n");
        }

        D3D11_TEXTURE2D_DESC sd{};
        sd.Width = cw;
        sd.Height = ch;
        sd.MipLevels = 1;
        sd.ArraySize = 1;
        sd.Format = g_hdr ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_B8G8R8A8_UNORM;
        sd.SampleDesc = {1, 0};
        sd.Usage = D3D11_USAGE_STAGING;
        sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        CHECK_HR(dev11->CreateTexture2D(&sd, nullptr, &staging));

        item.Closed([this](auto&&, auto&&) {
            InterlockedExchange(&closed, 1);
            SetEvent(evt);
        });
        pool.FrameArrived([this](auto&&, auto&&) {
            if (!takeFrames() && sameSkip)
                return; // an unchanged whole-screen picture: nothing for the loop
            InterlockedExchange(&arrived, 1);
            SetEvent(evt);
        });
        session.StartCapture();
        return 0;
    }

    // the source's own cadence, however slow the pipeline runs: its two newest frames under 300 ms apart, and
    // nothing from it for 300 ms (the static-source hold; false before its first frame)
    bool sourceCadence() const
    {
        return arrGap > 0 && arrGap < 3000000;
    }
    bool sourceQuiet() const
    {
        return lastArrTs && nowQpc100() - lastArrTs > 3000000;
    }

    // back to the pool, unless the frame still holds a role (held, the loop's, the test's reference); under lk
    void closeFree(wgc::Direct3D11CaptureFrame const& f)
    {
        if (f && f != held && f != loopFrame && f != refFrame)
            f.Close();
    }

    // the loop's frame back (to the pool, unless it is the test's reference); every drained frame ends here
    void release(wgc::Direct3D11CaptureFrame& frame)
    {
        AcquireSRWLockExclusive(&lk);
        if (frame == loopFrame)
            loopFrame = nullptr;
        closeFree(frame);
        ReleaseSRWLockExclusive(&lk);
        frame = nullptr;
    }

    // every frame on the pool into `held`, the newest kept (FrameArrived's thread and drainNewest; with sameSkip
    // FrameArrived's alone, so the arrivals are tested in their order). The test runs outside lk: drainNewest never
    // waits for the GPU behind it. A kept frame takes the roles (held; with a view, the test's reference) and the
    // frames that lost every role go back to the pool. True = a frame was kept
    bool takeFrames()
    {
        bool kept = false;
        if (sameSkip)
            AcquireSRWLockExclusive(&sameLk);
        AcquireSRWLockExclusive(&lk);
        while (!stopping)
        {
            auto f = pool.TryGetNextFrame();
            if (!f)
                break;
            ComPtr<ID3D11ShaderResourceView> view;
            if (sameSkip)
            {
                ReleaseSRWLockExclusive(&lk);
                const bool same = samePicture(f, view);
                AcquireSRWLockExclusive(&lk);
                if (!stopping)
                    notePresent(f.SystemRelativeTime().count(), !arrNew);
                if (same || stopping)
                {
                    sameN += same;
                    f.Close();
                    continue;
                }
            }
            else
                notePresent(f.SystemRelativeTime().count(), true);
            const int64_t ts = f.SystemRelativeTime().count();
            if (lastArrTs && ts > lastArrTs)
            {
                const double d = (ts - lastArrTs) / 10000.0;
                // >300ms = idle/static gap (1 Hz hold refresh), not cadence; skip like the pacer
                if (d < 300.0)
                    emaArrMs = emaArrMs > 0 ? emaArrMs * 0.8 + d * 0.2 : d;
            }
            arrGap = lastArrTs && ts > lastArrTs ? ts - lastArrTs : 0;
            lastArrTs = ts;
            if (!sameSkip || arrNew)
                arrRing[arrCount++ % kArrN] = ts;
            const auto oldHeld = held, oldRef = refFrame;
            held = f;
            if (sameSkip)
            {
                refSrv = view;
                refFrame = nullptr;
                if (view != nullptr)
                    refFrame = f;
            }
            if (oldHeld)
                dropped++; // superseded = the pipeline fell behind
            closeFree(oldHeld);
            if (oldRef != oldHeld)
                closeFree(oldRef);
            kept = true;
        }
        ReleaseSRWLockExclusive(&lk);
        if (sameSkip)
            ReleaseSRWLockExclusive(&sameLk);
        return kept;
    }

    // one present of the source for the HUD meter (lk held); `unchanged` = its picture repeats the last one (or no test
    // ran): such arrivals within 250 ms of our refresh request to a source idle for 300 ms are its repaints for us (a
    // player can answer one request with several), not its own presents; a new picture there is a real frame (a
    // slideshow's next picture can land in that window)
    void notePresent(int64_t ts, bool unchanged)
    {
        const int64_t prev = presCount ? presRing[(presCount - 1) % kArrN] : 0;
        if (unchanged && refreshAskTs && ts >= refreshAskTs && ts - refreshAskTs <= 2500000 &&
            (!prev || refreshAskTs - prev >= 3000000))
            return;
        presRing[presCount++ % kArrN] = ts;
    }

    // SMV_LAT_TRACE's capture rows, oldest first: `A <stamp>` the presents, `N <stamp>` the new pictures, `Q <time>` our
    // refresh requests (QPC 100 ns)
    void dumpStamps(FILE* f)
    {
        AcquireSRWLockExclusive(&lk);
        for (uint64_t i = presCount > kArrN ? presCount - kArrN : 0; i < presCount; ++i)
            fprintf(f, "A %lld\n", (long long)presRing[i % kArrN]);
        for (uint64_t i = arrCount > kArrN ? arrCount - kArrN : 0; i < arrCount; ++i)
            fprintf(f, "N %lld\n", (long long)arrRing[i % kArrN]);
        for (uint64_t i = askCount > 64 ? askCount - 64 : 0; i < askCount; ++i)
            fprintf(f, "Q %lld\n", (long long)askRing[i % 64]);
        ReleaseSRWLockExclusive(&lk);
    }

    // the HUD meter: the newest gap wider than `gap` between two consecutive presents (true) / new pictures (false)
    // stamped at or after `from` (`before` / `after` = its two frames); false = none
    bool latestGap(bool presents, int64_t from, int64_t gap, int64_t& before, int64_t& after)
    {
        AcquireSRWLockExclusive(&lk);
        const int64_t* ring = presents ? presRing : arrRing;
        const uint64_t count = presents ? presCount : arrCount;
        const uint64_t lo = count > kArrN ? count - kArrN : 0;
        bool found = false;
        for (uint64_t i = count; i > lo + 1 && !found; --i)
        {
            const int64_t t = ring[(i - 1) % kArrN], t0 = ring[(i - 2) % kArrN];
            if (t0 < from)
                break;
            if (t - t0 > gap)
            {
                before = t0;
                after = t;
                found = true;
            }
        }
        ReleaseSRWLockExclusive(&lk);
        return found;
    }

    // the HUD meter: the presents (true) or the new pictures (false) stamped from `from` to `to` (their count, the
    // oldest and the newest of them), and the newest two stamps of all (0 = none)
    int framesSince(bool presents, int64_t from, int64_t to, int64_t& first, int64_t& lastIn, int64_t& last,
                    int64_t& prev)
    {
        AcquireSRWLockExclusive(&lk);
        const int64_t* ring = presents ? presRing : arrRing;
        const uint64_t count = presents ? presCount : arrCount;
        int n = 0;
        first = lastIn = 0;
        last = count ? ring[(count - 1) % kArrN] : 0;
        prev = count > 1 ? ring[(count - 2) % kArrN] : 0;
        const uint64_t lo = count > kArrN ? count - kArrN : 0;
        for (uint64_t i = count; i > lo; --i)
        {
            const int64_t t = ring[(i - 1) % kArrN];
            if (t > to)
                continue;
            if (t < from)
                break;
            if (!n)
                lastIn = t;
            first = t;
            n++;
        }
        ReleaseSRWLockExclusive(&lk);
        return n;
    }

    // hand out the newest frame; -3 = none pending, -2 = resized, -1 = failure,
    // else 0 and `tex` holds the frame texture (the caller hands `frame` back through release())
    int drainNewest(wgc::Direct3D11CaptureFrame& frame, winrt::com_ptr<ID3D11Texture2D>& tex)
    {
        if (!sameSkip)
            takeFrames();
        AcquireSRWLockExclusive(&lk);
        if (held)
        {
            if (frame)
            {
                if (frame == loopFrame)
                    loopFrame = nullptr;
                closeFree(frame);
                dropped++;
            }
            frame = held;
            held = nullptr;
            loopFrame = frame;
        }
        ReleaseSRWLockExclusive(&lk);
        if (!frame)
            return -3;

        auto csz = frame.ContentSize();
        // The pool still captures the whole window, so compare against the full size, not
        // the cropped client size. A real resize triggers the exit-4 restart, which recomputes
        // the crop from fresh rects.
        if ((uint32_t)csz.Width != fullW || (uint32_t)csz.Height != fullH)
        {
            // >2px change = a real resize (1px wobble happens during DWM animations)
            if (abs(csz.Width - (int)fullW) > 2 || abs(csz.Height - (int)fullH) > 2)
            {
                // Not a resize: a minimized player hands out a caption-sized frame (183x34 on this
                // desktop) and the first frame after the restore can still carry it, while the
                // window keeps its size. Dropped, so a restore resumes the session (the pause
                // rule) instead of restarting it.
                RECT wb{};
                if (!g_monitor && targetWnd &&
                    (IsIconic(targetWnd) ||
                     (SUCCEEDED(DwmGetWindowAttribute(targetWnd, DWMWA_EXTENDED_FRAME_BOUNDS, &wb, sizeof(wb))) &&
                      abs((int)(wb.right - wb.left) - (int)fullW) <= 2 &&
                      abs((int)(wb.bottom - wb.top) - (int)fullH) <= 2)))
                {
                    release(frame);
                    return -3;
                }
                LOG("target window resized (%dx%d -> %ux%u)\n", csz.Width, csz.Height, fullW, fullH);
                release(frame);
                return -2;
            }
        }
        auto access = frame.Surface().as<Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>();
        if (FAILED(access->GetInterface(winrt::guid_of<ID3D11Texture2D>(), tex.put_void())))
        {
            release(frame);
            return -1;
        }
        lastFrameTs = frame.SystemRelativeTime().count(); // capture->present latency anchor
        return 0;
    }

    // WGC emits frames only when the source PRESENTS, so a static window (paused video) goes
    // silent forever - measured on a paused mpv: no initial frame, and neither pool
    // Recreate() nor a full capture-session restart delivers anything (both tried; --probe
    // hangs frameless on such a window too). The only working lever is making the TARGET
    // present once: InvalidateRect makes it repaint on its next message-loop pass, and that
    // Present (even of identical pixels) is what WGC forwards as a frame. This is the lever
    // behind the paused-source workflow: one refresh after the model load so warmup gets a
    // frame, then 1 Hz while idle so the picture stays current (seek/frame-step while paused
    // reaches the screen) and the HUD's captured-fps line reads the true ~1 fps. Apps that
    // ignore WM_PAINT are almost always continuously presenting anyway.
    void requestRefresh()
    {
        if (targetWnd)
        {
            AcquireSRWLockExclusive(&lk);
            refreshAskTs = nowQpc100();
            askRing[askCount++ % 64] = refreshAskTs;
            ReleaseSRWLockExclusive(&lk);
            InvalidateRect(targetWnd, nullptr, FALSE);
        }
    }

    // Copy the client sub-region of the full-window capture into dst (a cw x ch resource).
    // No crop (borderless / fullscreen / monitor) falls back to the plain full copy.
    void copyCropped(ID3D11Resource* dst, ID3D11Texture2D* src)
    {
        if (cropX || cropY || cw != fullW || ch != fullH)
        {
            D3D11_BOX box{(UINT)cropX, (UINT)cropY, 0, (UINT)cropX + cw, (UINT)cropY + ch, 1};
            ctx11->CopySubresourceRegion(dst, 0, 0, 0, 0, src, 0, &box);
        }
        else
        {
            ctx11->CopyResource(dst, src);
        }
    }

    // interop path: newest frame -> sharedTex, GPU copy only (the server reads it via CUDA after
    // the fence signal). Same return codes as latestFrame.
    int latestFrameGpu()
    {
        wgc::Direct3D11CaptureFrame frame{nullptr};
        winrt::com_ptr<ID3D11Texture2D> tex;
        int rc = drainNewest(frame, tex);
        if (rc == -3)
            return 0;
        if (rc)
            return rc;
        copyCropped(sharedTex.Get(), tex.get());
        release(frame);
        newestIn = 2;
        return 1;
    }

    // the newest drained frame into sharedTex (the server route's warm-up on a static source: the rate measurement
    // and the load-loop passthrough drained every frame it sent); staging and sharedTex share the size and format.
    // Stamped now: nothing has arrived since, so it is the current picture, and the pacing clock and the latency
    // anchor read the stamp (lastArrTs keeps the source's own arrival, so the static hold still arms)
    bool newestToShared()
    {
        if (!interop || !newestIn)
            return false;
        if (newestIn == 1)
            ctx11->CopyResource(sharedTex.Get(), staging.Get());
        newestIn = 2;
        lastFrameTs = nowQpc100();
        return true;
    }

    // interop path: order the shared-texture copy against the server's CUDA reads
    void signalFence(uint64_t v)
    {
        ctx4->Signal(sharedFence.Get(), v);
        ctx11->Flush(); // the immediate context may defer submission; the server is waiting
    }

    // CPU path: drain + staging readback into out (tight cw*4 rows). The bytes stay BGRA on the
    // server/identity routes (the native packers read them as (R, G, B) planes and the slot
    // stores write BGRA back for the B8G8R8A8 present textures); only the DLSS-G route sets
    // `swizzle` to keep its verified RGBA pipeline byte-identical. Returns
    // 1 = got a frame, 0 = none pending, -1 = failure, -2 = content resized.
    int latestFrame(uint8_t* out)
    {
        return latestFrameMapped([&](const uint8_t* src, size_t pitch, bool swz) {
            copyRows(out, (size_t)cw * 4, src, pitch, cw, ch, swz);
        });
    }

    // latestFrame without the copy: use(rows, rowPitch, swz) reads the mapped readback before it is unmapped (swz = BGRA
    // rows for the RGBA route to swap; the HDR pack's rows are its R10G10B10A2 words). Same returns.
    template <class F> int latestFrameMapped(F&& use)
    {
        wgc::Direct3D11CaptureFrame frame{nullptr};
        winrt::com_ptr<ID3D11Texture2D> tex;
        int rc = drainNewest(frame, tex);
        if (rc == -3)
            return 0;
        if (rc)
            return rc;

        // dlssg HDR: pack scRGB FP16 -> R10A2 PQ on the GPU, read back 4 B/px.
        // The R32_UINT staging bytes ARE the R10G10B10A2_UNORM bit pattern presentFrame uploads.
        if (hdrPack)
        {
            // the pack reads the frame at the crop's offset, the frame back to the pool after the Map (it waits for
            // the pack); else the crop is copied first and read at 0
            const ComPtr<ID3D11ShaderResourceView> arr = inPlaceView(tex.get());
            if (!arr)
            {
                if (!convSrcReady())
                {
                    release(frame);
                    return -1;
                }
                copyCropped(convSrc.Get(), tex.get());
                release(frame);
            }
            newestIn = 0;
            if (mt)
                mt->Enter(); // samePicture binds its own compute shader from FrameArrived's thread
            ctx11->CSSetShader(convCS.Get(), nullptr, 0);
            ID3D11ShaderResourceView* srv = arr ? arr.Get() : convSrcView.Get();
            ctx11->CSSetShaderResources(0, 1, &srv);
            ID3D11UnorderedAccessView* uav = convUav.Get();
            ctx11->CSSetUnorderedAccessViews(0, 1, &uav, nullptr);
            const UINT off[4]{arr ? (UINT)cropX : 0u, arr ? (UINT)cropY : 0u, 0, 0};
            ctx11->UpdateSubresource(convCb.Get(), 0, nullptr, off, 0, 0);
            ID3D11Buffer* cb = convCb.Get();
            ctx11->CSSetConstantBuffers(0, 1, &cb);
            ctx11->Dispatch((cw + 7) / 8, (ch + 7) / 8, 1);
            srv = nullptr;
            uav = nullptr; // unbind so next frame's copy into convSrc is hazard-free
            cb = nullptr;
            ctx11->CSSetShaderResources(0, 1, &srv);
            ctx11->CSSetUnorderedAccessViews(0, 1, &uav, nullptr);
            ctx11->CSSetConstantBuffers(0, 1, &cb);
            if (mt)
                mt->Leave();
            ctx11->CopyResource(convStaging.Get(), convDst.Get());
            D3D11_MAPPED_SUBRESOURCE pm{};
            const HRESULT mapped = ctx11->Map(convStaging.Get(), 0, D3D11_MAP_READ, 0, &pm);
            if (arr)
                release(frame); // the pack has run: the Map waited for it
            if (FAILED(mapped))
                return -1;
            use((const uint8_t*)pm.pData, (size_t)pm.RowPitch, false);
            ctx11->Unmap(convStaging.Get(), 0);
            return 1;
        }

        copyCropped(staging.Get(), tex.get());
        release(frame);
        newestIn = 1;

        D3D11_MAPPED_SUBRESOURCE map{};
        if (FAILED(ctx11->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &map)))
            return -1;
        use((const uint8_t*)map.pData, (size_t)map.RowPitch, swizzle);
        ctx11->Unmap(staging.Get(), 0);
        return 1;
    }

    // the frame through the GPU handoff: its crop into dgTex[slot] for the DLSS-G device, its thumbnail read back.
    // use(thumb, pitch, slot, wrote) runs the estimate on the thumbnail and presents the slot once dgWrote reaches
    // `wrote`; the caller sets dgFree[slot] to the dgCopied value of that present. latestFrame's returns
    template <class F> int latestFrameShared(F&& use)
    {
        wgc::Direct3D11CaptureFrame frame{nullptr};
        winrt::com_ptr<ID3D11Texture2D> tex;
        int rc = drainNewest(frame, tex);
        if (rc == -3)
            return 0;
        if (rc)
            return rc;
        const int s = dgSlot;
        dgSlot ^= 1;
        // HDR10: the pack reads the frame itself at the crop's offset, the frame stays out of the pool until the
        // thumbnail's Map below (the thumbnail reads the pack's slot); else the crop is copied first
        const ComPtr<ID3D11ShaderResourceView> arr = inPlaceView(tex.get());
        if (!arr)
        {
            if (hdrPack && !convSrcReady())
            {
                release(frame);
                return -1;
            }
            copyCropped(hdrPack ? convSrc.Get() : dgSrc.Get(), tex.get());
            release(frame);
        }
        newestIn = 0;
        if (mt)
            mt->Enter(); // samePicture binds its own compute shader from FrameArrived's thread
        // one compute pass: t0 = srv, u0 = dst (and u1 = dst1 when given)
        auto pass = [&](ID3D11ComputeShader* cs, ID3D11ShaderResourceView* srv, ID3D11UnorderedAccessView* dst, UINT w,
                        UINT h, ID3D11UnorderedAccessView* dst1 = nullptr) {
            ctx11->CSSetShader(cs, nullptr, 0);
            ctx11->CSSetShaderResources(0, 1, &srv);
            ID3D11UnorderedAccessView* uavs[2]{dst, dst1};
            const UINT n = dst1 ? 2u : 1u;
            ctx11->CSSetUnorderedAccessViews(0, n, uavs, nullptr);
            ctx11->Dispatch((w + 7) / 8, (h + 7) / 8, 1);
            srv = nullptr;
            uavs[0] = uavs[1] = nullptr;
            ctx11->CSSetShaderResources(0, 1, &srv);
            ctx11->CSSetUnorderedAccessViews(0, n, uavs, nullptr);
        };
        if (hdrPack)
        {
            // dlssg HDR: the pack stores the slot itself once its last frame was copied out, and the thumbnail beside
            // it (u1) from the codes it made: no pass reads the slot back
            if (dgFree[s])
                ctx4->Wait(dgCopied.Get(), dgFree[s]);
            const UINT off[4]{arr ? (UINT)cropX : 0u, arr ? (UINT)cropY : 0u, 0, 0};
            ctx11->UpdateSubresource(convCb.Get(), 0, nullptr, off, 0, 0);
            ID3D11Buffer* cb = convCb.Get();
            ctx11->CSSetConstantBuffers(0, 1, &cb);
            pass(dgCS.Get(), arr ? arr.Get() : convSrcView.Get(), dgUav[s].Get(), cw, ch, dgThumbUav.Get());
            cb = nullptr;
            ctx11->CSSetConstantBuffers(0, 1, &cb);
            ctx4->Signal(dgWrote.Get(), ++dgWroteVal);
            ctx11->CopyResource(dgThumbStage.Get(), dgThumb.Get());
        }
        else
        {
            // the thumbnail first: its Map below waits for it alone, not for the slot's wait
            pass(dgThumbCS.Get(), dgSrcView.Get(), dgThumbUav.Get(), cw / 4, ch / 4);
            ctx11->CopyResource(dgThumbStage.Get(), dgThumb.Get());
            if (dgFree[s])
                ctx4->Wait(dgCopied.Get(), dgFree[s]); // the slot's last frame was copied out
            pass(dgCS.Get(), dgSrcView.Get(), dgUav[s].Get(), cw, ch);
            ctx4->Signal(dgWrote.Get(), ++dgWroteVal);
        }
        if (mt)
            mt->Leave();
        ctx11->Flush(); // the DLSS-G device waits for the signal
        D3D11_MAPPED_SUBRESOURCE map{};
        const HRESULT mapped = ctx11->Map(dgThumbStage.Get(), 0, D3D11_MAP_READ, 0, &map);
        if (arr)
            release(frame); // the pack has run (the thumbnail is its write): the Map waited for it
        if (FAILED(mapped))
            return -1;
        use((const uint8_t*)map.pData, (size_t)map.RowPitch, s, dgWroteVal);
        ctx11->Unmap(dgThumbStage.Get(), 0);
        return 1;
    }

    void stop()
    {
        if (sameSkip)
            AcquireSRWLockExclusive(&sameLk); // no test in flight reads the reference while it goes
        AcquireSRWLockExclusive(&lk);
        stopping = true;
        const auto h = held, r = refFrame;
        held = nullptr;
        refFrame = nullptr;
        refSrv.Reset();
        closeFree(h);
        if (r != h)
            closeFree(r);
        ReleaseSRWLockExclusive(&lk);
        if (sameSkip)
            ReleaseSRWLockExclusive(&sameLk);
        if (sameSkip)
            LOG("capture: %llu unchanged arrivals dropped, %llu kept after %d ms; the test %.3f ms mean, %.3f ms max "
                "over %llu\n",
                (unsigned long long)sameN, (unsigned long long)sameKept, (int)(kSameKeepalive / 10000),
                sameTests ? sameMsSum / (double)sameTests : 0.0, sameMsMax, (unsigned long long)sameTests);
        // session.Close() calls Windows' capture service in another process and can block indefinitely (a host's main
        // thread was found inside it, waiting on that call, after its target window closed): both Closes run on a
        // helper thread, waited for at most kCloseWaitMs; past that the helper keeps the two objects and the teardown
        // goes on. SMV_WGC_CLOSE=hang = the trigger (the helper waits 4 s before its Closes)
        if (session || pool)
        {
            constexpr DWORD kCloseWaitMs = 2000;
            wchar_t cv[8] = {};
            const bool hangTest = GetEnvironmentVariableW(L"SMV_WGC_CLOSE", cv, 8) && wcscmp(cv, L"hang") == 0;
            std::shared_ptr<void> done(CreateEventW(nullptr, TRUE, FALSE, nullptr), CloseHandle);
            std::thread([s = std::move(session), p = std::move(pool), done, hangTest]() mutable {
                if (hangTest)
                    Sleep(4000);
                if (s)
                    s.Close();
                if (p)
                    p.Close();
                s = nullptr;
                p = nullptr;
                SetEvent(done.get());
            }).detach();
            session = nullptr;
            pool = nullptr;
            if (WaitForSingleObject(done.get(), kCloseWaitMs) != WAIT_OBJECT_0)
                LOG("capture: closing the capture session did not return in %lu ms (Windows' capture service): left "
                    "to a helper thread\n",
                    kCloseWaitMs);
        }
        if (evt)
            CloseHandle(evt);
        evt = nullptr;
        // the shared NT handles of the interop texture and fence (the native host released
        // its imports already, in nativeFree)
        if (hTex)
        {
            CloseHandle(hTex);
            hTex = nullptr;
        }
        if (hFence)
        {
            CloseHandle(hFence);
            hFence = nullptr;
        }
        interop = false;
    }
};
