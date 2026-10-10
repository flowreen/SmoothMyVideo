// part of smv-live.cpp (unity include, see the file map there); not a standalone translation unit
// ---------------------------------------------------------------- WGC capture

// The DLSS-G route in HDR mode. SL's DLSS-G mandates a UINT10/RGB10 backbuffer
// in HDR10/BT.2100 PQ and explicitly rejects FP16 scRGB (ProgrammingGuideDLSS_G.md 11.0), which
// is exactly the server route's present format - but the DLSS-G route has no model host to
// convert, so the exe does it: this compute shader is the same math as the native kernels'
// scrgb_to_pq2020 + R10G10B10A2 pack (709->2020 matrix FIRST, clamp AFTER it - negative scRGB is
// valid wide gamut - scRGB 1.0 = 80 nits over PQ 10000, A = 3). Output is R32_UINT (manual bit
// pack, typed-UAV support guaranteed) whose bit pattern IS R10G10B10A2_UNORM.
static const char kHdrPackCS[] = "Texture2D<float4> src : register(t0);\n"
                                 "RWTexture2D<uint> dst : register(u0);\n"
                                 "[numthreads(8,8,1)]\n"
                                 "void main(uint3 id : SV_DispatchThreadID)\n"
                                 "{\n"
                                 "    uint w, h;\n"
                                 "    dst.GetDimensions(w, h);\n"
                                 "    if (id.x >= w || id.y >= h) return;\n"
                                 "    float3 c = src[id.xy].rgb;\n"
                                 "    float3 v = float3(dot(float3(0.6274, 0.3293, 0.0433), c),\n"
                                 "                      dot(float3(0.0691, 0.9195, 0.0114), c),\n"
                                 "                      dot(float3(0.0164, 0.0880, 0.8956), c));\n"
                                 "    v = max(v, 0.0) * (80.0 / 10000.0);\n"
                                 "    float3 lm = pow(v, 0.1593017578125);\n"
                                 "    float3 p = pow((0.8359375 + 18.8515625 * lm) / (1.0 + 18.6875 * lm), 78.84375);\n"
                                 "    uint3 q = (uint3)(saturate(p) * 1023.0 + 0.5);\n"
                                 "    dst[id.xy] = q.x | (q.y << 10) | (q.z << 20) | (3u << 30);\n"
                                 "}\n";

// Whole-screen capture (Capture::samePicture): any bit that differs between two pictures sets the flag. The
// formats read as float4 (B8G8R8A8 unorm: exact, one value per byte; FP16: exact), compared as their bits.
static const char kSamePictureCS[] = "Texture2D<float4> a : register(t0);\n"
                                     "Texture2D<float4> b : register(t1);\n"
                                     "RWByteAddressBuffer flag : register(u0);\n"
                                     "[numthreads(16,16,1)]\n"
                                     "void main(uint3 id : SV_DispatchThreadID)\n"
                                     "{\n"
                                     "    uint w, h;\n"
                                     "    a.GetDimensions(w, h);\n"
                                     "    if (id.x >= w || id.y >= h) return;\n"
                                     "    if (any(asuint(a[id.xy]) != asuint(b[id.xy])))\n"
                                     "        flag.Store(0, 1u);\n"
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
    // source that keeps presenting. sameTex[sameCur] = the arrival under test, sameTex[1 - sameCur] = the last kept
    // picture. The immediate context is multithread-protected while this runs (`mt`; its compute-shader sequences hold
    // mt->Enter()).
    static constexpr int64_t kSameKeepalive = 5000000; // 100 ns units: 500 ms
    bool allowSame = true; // a caller that wants every arrival (the probe) clears it before init()
    bool sameSkip = false;
    ComPtr<ID3D11Multithread> mt;
    ComPtr<ID3D11ComputeShader> sameCS;
    ComPtr<ID3D11Texture2D> sameTex[2];
    ComPtr<ID3D11ShaderResourceView> sameSrv[2];
    ComPtr<ID3D11Buffer> sameFlag, sameFlagStage;
    ComPtr<ID3D11UnorderedAccessView> sameUav;
    int sameCur = 0;
    bool sameRef = false;  // sameTex[1 - sameCur] holds a picture
    uint64_t sameN = 0;    // arrivals dropped as unchanged
    uint64_t sameKept = 0; // unchanged arrivals kept by kSameKeepalive
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
        td.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        if (FAILED(dev11->CreateTexture2D(&td, nullptr, &convSrc)))
            return false;
        td.Format = DXGI_FORMAT_R32_UINT;
        td.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
        if (FAILED(dev11->CreateTexture2D(&td, nullptr, &convDst)))
            return false;
        td.BindFlags = 0;
        td.Usage = D3D11_USAGE_STAGING;
        td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        if (FAILED(dev11->CreateTexture2D(&td, nullptr, &convStaging)))
            return false;
        if (FAILED(dev11->CreateShaderResourceView(convSrc.Get(), nullptr, &convSrcView)))
            return false;
        if (FAILED(dev11->CreateUnorderedAccessView(convDst.Get(), nullptr, &convUav)))
            return false;
        LOG("dlssg HDR: exe-side scRGB -> PQ R10A2 pack ready (%ux%u)\n", cw, ch);
        return true;
    }

    // the unchanged-picture test (samePicture) at the client crop's size and the pool's format; false = not available
    // (a whole-screen caller then keeps Windows' update interval). SMV_WGC_SAME=fail = this failure's trigger
    bool initSame(DXGI_FORMAT fmt)
    {
        wchar_t lv[8]{};
        if (GetEnvironmentVariableW(L"SMV_WGC_SAME", lv, 8) && wcscmp(lv, L"fail") == 0)
            return false;
        ComPtr<ID3DBlob> cs, err;
        if (FAILED(D3DCompile(kSamePictureCS, sizeof(kSamePictureCS) - 1, nullptr, nullptr, nullptr, "main", "cs_5_0",
                              0, 0, &cs, &err)) ||
            FAILED(dev11->CreateComputeShader(cs->GetBufferPointer(), cs->GetBufferSize(), nullptr, &sameCS)))
            return false;
        D3D11_TEXTURE2D_DESC td{};
        td.Width = cw;
        td.Height = ch;
        td.MipLevels = 1;
        td.ArraySize = 1;
        td.Format = fmt;
        td.SampleDesc = {1, 0};
        td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        for (int i = 0; i < 2; i++)
            if (FAILED(dev11->CreateTexture2D(&td, nullptr, &sameTex[i])) ||
                FAILED(dev11->CreateShaderResourceView(sameTex[i].Get(), nullptr, &sameSrv[i])))
                return false;
        D3D11_BUFFER_DESC bd{};
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

    // true = this arrival's client picture equals the last kept one; a new picture becomes the reference. The flag's
    // readback waits for the copy too, so the frame may be closed right after
    bool samePicture(wgc::Direct3D11CaptureFrame const& f)
    {
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
        D3D11_TEXTURE2D_DESC fd{}, rd{};
        if (tex)
        {
            tex->GetDesc(&fd);
            sameTex[0]->GetDesc(&rd);
        }
        if (!tex || fd.Width < (UINT)cropX + cw || fd.Height < (UINT)cropY + ch || fd.Format != rd.Format)
        {
            sameRef = false; // a resize or a caption-sized frame: the next picture starts over
            return false;
        }
        const bool keepalive = sameRef && lastArrTs && ts - lastArrTs >= kSameKeepalive;
        const int64_t t0 = nowQpc100();
        bool same = false;
        mt->Enter();
        const D3D11_BOX box{(UINT)cropX, (UINT)cropY, 0, (UINT)cropX + cw, (UINT)cropY + ch, 1};
        ctx11->CopySubresourceRegion(sameTex[sameCur].Get(), 0, 0, 0, 0, tex.get(), 0, &box);
        if (keepalive)
            sameKept++;
        else if (sameRef)
        {
            const UINT zero[4]{};
            ctx11->ClearUnorderedAccessViewUint(sameUav.Get(), zero);
            ctx11->CSSetShader(sameCS.Get(), nullptr, 0);
            ID3D11ShaderResourceView* srv[2]{sameSrv[sameCur].Get(), sameSrv[1 - sameCur].Get()};
            ctx11->CSSetShaderResources(0, 2, srv);
            ID3D11UnorderedAccessView* uav = sameUav.Get();
            ctx11->CSSetUnorderedAccessViews(0, 1, &uav, nullptr);
            ctx11->Dispatch((cw + 15) / 16, (ch + 15) / 16, 1);
            ID3D11ShaderResourceView* none[2]{};
            ctx11->CSSetShaderResources(0, 2, none);
            uav = nullptr;
            ctx11->CSSetUnorderedAccessViews(0, 1, &uav, nullptr);
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
        mt->Leave();
        if (!same)
        {
            sameCur = 1 - sameCur;
            sameRef = true;
        }
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
        emaArrMs = 0;
        lastArrTs = 0;
        arrGap = 0;
        lastFrameTs = 0;
        newestIn = 0;
        held = nullptr;
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
        sameSkip = false;
        sameRef = false;
        sameCur = 0;
        sameN = sameTests = sameKept = 0;
        sameMsSum = sameMsMax = 0;
        mt.Reset();
        sameCS.Reset();
        sameFlag.Reset();
        sameFlagStage.Reset();
        sameUav.Reset();
        for (int i = 0; i < 2; i++)
        {
            sameTex[i].Reset();
            sameSrv[i].Reset();
        }
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

        // the unchanged-picture test (sameSkip), every fit; SMV_WGC_SAME=0 = off (A/B)
        if (allowSame)
        {
            wchar_t lv[8]{};
            if (!(GetEnvironmentVariableW(L"SMV_WGC_SAME", lv, 8) && lv[0] == L'0'))
            {
                sameSkip = initSame(g_hdr ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_B8G8R8A8_UNORM);
                if (sameSkip)
                    LOG("capture: arrivals whose picture is unchanged are dropped (%s)\n",
                        g_monitor ? "the overlay's own presents" : "a player's repeated presents");
                else
                    LOG(g_monitor ? "note: the whole screen's unchanged-picture test is unavailable: the capture keeps "
                                    "Windows' update interval (60 fps at most)\n"
                                  : "note: the unchanged-picture test is unavailable: a player's repeated presents "
                                    "count as source frames\n");
            }
        }

        // HDR mode captures FP16 scRGB (linear, 709 primaries, 1.0 = 80 nits); the server
        // re-encodes to BT.2020 PQ. SDR stays 8-bit BGRA. The unchanged-picture test holds a frame while it runs:
        // a third buffer keeps WGC from discarding the next arrival meanwhile
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

    // every frame on the pool into `held`, the newest kept (FrameArrived's thread and drainNewest; with sameSkip
    // FrameArrived's alone, so the arrivals are tested in their order). The test runs outside lk: drainNewest never
    // waits for the GPU behind it. True = a frame was kept
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
            if (sameSkip)
            {
                ReleaseSRWLockExclusive(&lk);
                const bool same = samePicture(f);
                AcquireSRWLockExclusive(&lk);
                if (same || stopping)
                {
                    sameN += same;
                    f.Close();
                    continue;
                }
            }
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
            if (held)
            {
                held.Close();
                dropped++;
            } // superseded = the pipeline fell behind
            held = f;
            kept = true;
        }
        ReleaseSRWLockExclusive(&lk);
        if (sameSkip)
            ReleaseSRWLockExclusive(&sameLk);
        return kept;
    }

    // hand out the newest frame; -3 = none pending, -2 = resized, -1 = failure,
    // else 0 and `tex` holds the frame texture (caller must Close `frame`)
    int drainNewest(wgc::Direct3D11CaptureFrame& frame, winrt::com_ptr<ID3D11Texture2D>& tex)
    {
        if (!sameSkip)
            takeFrames();
        AcquireSRWLockExclusive(&lk);
        if (held)
        {
            if (frame)
            {
                frame.Close();
                dropped++;
            }
            frame = held;
            held = nullptr;
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
                    frame.Close();
                    frame = nullptr;
                    return -3;
                }
                LOG("target window resized (%dx%d -> %ux%u)\n", csz.Width, csz.Height, fullW, fullH);
                frame.Close();
                frame = nullptr;
                return -2;
            }
        }
        auto access = frame.Surface().as<Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>();
        if (FAILED(access->GetInterface(winrt::guid_of<ID3D11Texture2D>(), tex.put_void())))
        {
            frame.Close();
            frame = nullptr;
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
            InvalidateRect(targetWnd, nullptr, FALSE);
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
        frame.Close();
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
            copyCropped(convSrc.Get(), tex.get());
            frame.Close();
            newestIn = 0;
            if (mt)
                mt->Enter(); // samePicture binds its own compute shader from FrameArrived's thread
            ctx11->CSSetShader(convCS.Get(), nullptr, 0);
            ID3D11ShaderResourceView* srv = convSrcView.Get();
            ctx11->CSSetShaderResources(0, 1, &srv);
            ID3D11UnorderedAccessView* uav = convUav.Get();
            ctx11->CSSetUnorderedAccessViews(0, 1, &uav, nullptr);
            ctx11->Dispatch((cw + 7) / 8, (ch + 7) / 8, 1);
            srv = nullptr;
            uav = nullptr; // unbind so next frame's copy into convSrc is hazard-free
            ctx11->CSSetShaderResources(0, 1, &srv);
            ctx11->CSSetUnorderedAccessViews(0, 1, &uav, nullptr);
            if (mt)
                mt->Leave();
            ctx11->CopyResource(convStaging.Get(), convDst.Get());
            D3D11_MAPPED_SUBRESOURCE pm{};
            if (FAILED(ctx11->Map(convStaging.Get(), 0, D3D11_MAP_READ, 0, &pm)))
                return -1;
            for (uint32_t y = 0; y < ch; y++)
                memcpy(out + (size_t)y * cw * 4, (const uint8_t*)pm.pData + (size_t)y * pm.RowPitch, (size_t)cw * 4);
            ctx11->Unmap(convStaging.Get(), 0);
            return 1;
        }

        copyCropped(staging.Get(), tex.get());
        frame.Close();
        newestIn = 1;

        D3D11_MAPPED_SUBRESOURCE map{};
        if (FAILED(ctx11->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &map)))
            return -1;
        for (uint32_t y = 0; y < ch; y++)
        {
            const uint8_t* s = (const uint8_t*)map.pData + (size_t)y * map.RowPitch;
            if (!swizzle)
            {
                memcpy(out + (size_t)y * cw * 4, s, (size_t)cw * 4);
                continue;
            }
            uint32_t* d = (uint32_t*)(out + (size_t)y * cw * 4);
            for (uint32_t x = 0; x < cw; x++)
            {
                uint32_t v = ((const uint32_t*)s)[x]; // BGRA in memory = 0xAARRGGBB little-endian
                d[x] = (v & 0xFF00FF00u) | ((v & 0x00FF0000u) >> 16) | ((v & 0x000000FFu) << 16);
            }
        }
        ctx11->Unmap(staging.Get(), 0);
        return 1;
    }

    void stop()
    {
        AcquireSRWLockExclusive(&lk);
        stopping = true;
        if (held)
            held.Close();
        held = nullptr;
        ReleaseSRWLockExclusive(&lk);
        if (sameSkip)
            LOG("capture: %llu unchanged arrivals dropped, %llu kept after %d ms; the test %.3f ms mean, %.3f ms max "
                "over %llu\n",
                (unsigned long long)sameN, (unsigned long long)sameKept, (int)(kSameKeepalive / 10000),
                sameTests ? sameMsSum / (double)sameTests : 0.0, sameMsMax, (unsigned long long)sameTests);
        if (session)
            session.Close();
        if (pool)
            pool.Close();
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
