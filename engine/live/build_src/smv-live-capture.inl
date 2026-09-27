// part of smv-live.cpp (unity include, see the file map there); not a standalone translation unit
// ---------------------------------------------------------------- WGC capture

// The DLSS-G route in HDR mode. SL's DLSS-G mandates a UINT10/RGB10 backbuffer
// in HDR10/BT.2100 PQ and explicitly rejects FP16 scRGB (ProgrammingGuideDLSS_G.md 11.0), which
// is exactly the server route's present format - but dlssg has no python server to convert, so
// the exe does it: this compute shader is the verbatim math of live_server.py's
// _scrgb_to_pq2020 + _pack_r10a2 (709->2020 matrix FIRST, clamp AFTER it - negative scRGB is
// valid wide gamut - scRGB 1.0 = 80 nits over PQ 10000, A = 3). Output is R32_UINT (manual bit
// pack, typed-UAV support guaranteed) whose bit pattern IS R10G10B10A2_UNORM.
static const char kHdrPackCS[] =
"Texture2D<float4> src : register(t0);\n"
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

// DLSS 5 NR hooks into the capture (defined after LiveNr below): the NR pass reads and rewrites the
// shared capture texture between the D3D11 copy and the fence signal.
struct Capture;
static void liveNrPreCopy(Capture& cap);
static bool liveNrSignal(Capture& cap, uint64_t v);

struct Capture
{
    ComPtr<ID3D11Device> dev11;
    ComPtr<ID3D11DeviceContext> ctx11;
    ComPtr<ID3D11Texture2D> staging;
    wgc::GraphicsCaptureItem item{ nullptr };
    wgc::Direct3D11CaptureFramePool pool{ nullptr };
    wgc::GraphicsCaptureSession session{ nullptr };
    HANDLE evt{};            // signaled by FrameArrived
    volatile LONG arrived = 0;   // set by FrameArrived, cleared by whoever drains next
    volatile LONG closed = 0;
    uint32_t cw = 0, ch = 0; // CLIENT size = staging/sharedTex size = what the pipeline sees
    // The WGC frame pool captures the WHOLE window (frame bounds), including the title bar
    // and borders. fullW/fullH are that captured size; cropX/cropY + cw/ch select the CLIENT area
    // out of it, so the non-client chrome never reaches the model or the overlay. clientScreenX/Y
    // is the client-area origin in screen coords (where the window-mode overlay must sit).
    uint32_t fullW = 0, fullH = 0;  // frame pool size (full window)
    int cropX = 0, cropY = 0;       // client-area offset inside the captured frame
    int clientScreenX = 0, clientScreenY = 0;
    bool swizzle = false;    // DLSS-G route only: convert BGRA -> RGBA on readback
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
    uint64_t dropped = 0;    // frames superseded in drainNewest (the SATURATION signal:
                             // captured + dropped per window = the true source cadence)
    double emaArrMs = 0;     // smoothed ARRIVAL interval (ms) over every delivered frame,
                             // superseded ones included = the true source cadence. Pacing
                             // must read THIS, not processed-frame intervals: a loop paced
                             // by its own processing rate feeds back into itself and locks
                             // capture into subharmonic plateaus (measured).
    int64_t lastArrTs = 0;   // SystemRelativeTime of the previously delivered frame
    // Zero-copy capture interop (server route): frames are GPU-copied into a SHARED texture
    // the python server imports as CUDA external memory, with a shared D3D11 fence for
    // ordering. Capture never touches the CPU: no staging Map, no shm memcpy, no H2D upload.
    ComPtr<ID3D11Texture2D> sharedTex;
    ComPtr<ID3D11Fence> sharedFence;
    ComPtr<ID3D11DeviceContext4> ctx4;
    HANDLE hTex = nullptr, hFence = nullptr;   // INHERITABLE NT handles (same values in the child)
    bool interop = false;
    HWND targetWnd = nullptr;   // capture target (monitor mode: the seeding window), for requestRefresh

    void initInterop()
    {
        ComPtr<ID3D11Device5> dev5;
        if (FAILED(dev11.As(&dev5)) || FAILED(ctx11.As(&ctx4)))
        { LOG("capture interop: D3D11.4 unavailable\n"); return; }
        D3D11_TEXTURE2D_DESC td{};
        td.Width = cw; td.Height = ch;
        td.MipLevels = 1; td.ArraySize = 1;
        td.Format = g_hdr ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_B8G8R8A8_UNORM;
        td.SampleDesc = { 1, 0 };
        td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
        td.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
        if (FAILED(dev11->CreateTexture2D(&td, nullptr, &sharedTex)))
        { LOG("capture interop: shared texture creation failed\n"); return; }
        ComPtr<IDXGIResource1> res1;
        if (FAILED(sharedTex.As(&res1))) return;
        SECURITY_ATTRIBUTES sa{ sizeof(sa), nullptr, TRUE };   // inheritable: value survives into the child
        if (FAILED(res1->CreateSharedHandle(&sa, GENERIC_ALL, nullptr, &hTex)))
        { LOG("capture interop: CreateSharedHandle(texture) failed\n"); return; }
        if (FAILED(dev5->CreateFence(0, D3D11_FENCE_FLAG_SHARED, IID_PPV_ARGS(&sharedFence))) ||
            FAILED(sharedFence->CreateSharedHandle(&sa, GENERIC_ALL, nullptr, &hFence)))
        { LOG("capture interop: shared fence creation failed\n"); return; }
        interop = true;
        LOG("capture interop: shared texture + fence ready\n");
    }

    // dlssg HDR: compile the pack shader and create the conversion chain (cw x ch known by
    // now). Failure is graceful: caller clears g_hdr and the session runs SDR with the clip
    // notice.
    bool initHdrPack()
    {
        ComPtr<ID3DBlob> cs, err;
        if (FAILED(D3DCompile(kHdrPackCS, sizeof(kHdrPackCS) - 1, nullptr, nullptr, nullptr,
                              "main", "cs_5_0", 0, 0, &cs, &err)))
        {
            LOG("HDR pack shader compile failed: %s\n", err ? (const char*)err->GetBufferPointer() : "?");
            return false;
        }
        if (FAILED(dev11->CreateComputeShader(cs->GetBufferPointer(), cs->GetBufferSize(), nullptr, &convCS)))
            return false;
        D3D11_TEXTURE2D_DESC td{};
        td.Width = cw; td.Height = ch;
        td.MipLevels = 1; td.ArraySize = 1;
        td.SampleDesc = { 1, 0 };
        td.Usage = D3D11_USAGE_DEFAULT;
        td.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        if (FAILED(dev11->CreateTexture2D(&td, nullptr, &convSrc))) return false;
        td.Format = DXGI_FORMAT_R32_UINT;
        td.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
        if (FAILED(dev11->CreateTexture2D(&td, nullptr, &convDst))) return false;
        td.BindFlags = 0;
        td.Usage = D3D11_USAGE_STAGING;
        td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        if (FAILED(dev11->CreateTexture2D(&td, nullptr, &convStaging))) return false;
        if (FAILED(dev11->CreateShaderResourceView(convSrc.Get(), nullptr, &convSrcView))) return false;
        if (FAILED(dev11->CreateUnorderedAccessView(convDst.Get(), nullptr, &convUav))) return false;
        LOG("dlssg HDR: exe-side scRGB -> PQ R10A2 pack ready (%ux%u)\n", cw, ch);
        return true;
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
        lastFrameTs = 0;
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
        targetWnd = target;
        evt = CreateEventW(nullptr, FALSE, FALSE, nullptr);

        // create the capture device via the REAL d3d11.dll: linking sl.interposer.lib makes the
        // import D3D11CreateDevice resolve to SL's proxy, which "automatically assigns" this
        // device to Streamline and breaks DLSS-G's pacer (presents stop generating; verified:
        // numFramesActuallyPresented stayed 1 until this bypass)
        HMODULE d3d11 = LoadLibraryExW(L"C:\\Windows\\System32\\d3d11.dll", nullptr, 0);
        if (!d3d11) { LOG("LoadLibrary d3d11.dll failed\n"); return 1; }
        auto realCreate = (PFN_D3D11_CREATE_DEVICE)GetProcAddress(d3d11, "D3D11CreateDevice");
        if (!realCreate) { LOG("GetProcAddress D3D11CreateDevice failed\n"); return 1; }
        CHECK_HR(realCreate(adapter, D3D_DRIVER_TYPE_UNKNOWN, nullptr,
                            D3D11_CREATE_DEVICE_BGRA_SUPPORT, nullptr, 0,
                            D3D11_SDK_VERSION, &dev11, nullptr, &ctx11));
        ComPtr<IDXGIDevice> dxgiDev;
        CHECK_HR(dev11.As(&dxgiDev));
        winrt::com_ptr<::IInspectable> insp;
        CHECK_HR(CreateDirect3D11DeviceFromDXGIDevice(dxgiDev.Get(), insp.put()));
        auto rtDev = insp.as<wgdx::Direct3D11::IDirect3DDevice>();

        auto interop = winrt::get_activation_factory<wgc::GraphicsCaptureItem>().as<IGraphicsCaptureItemInterop>();
        HRESULT hr = g_monitor
            ? interop->CreateForMonitor(MonitorFromWindow(target, MONITOR_DEFAULTTONEAREST),
                                        winrt::guid_of<wgc::GraphicsCaptureItem>(), winrt::put_abi(item))
            : interop->CreateForWindow(target, winrt::guid_of<wgc::GraphicsCaptureItem>(), winrt::put_abi(item));
        if (FAILED(hr)) { LOG("%s failed hr=0x%08lx\n", g_monitor ? "CreateForMonitor" : "CreateForWindow", hr); return 1; }

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
            POINT tl{ 0, 0 };
            DwmGetWindowAttribute(target, DWMWA_EXTENDED_FRAME_BOUNDS, &fbr, sizeof(fbr));
            if (GetClientRect(target, &cr) && ClientToScreen(target, &tl))
            {
                int cx = tl.x - fbr.left;
                int cy = tl.y - fbr.top;
                int clw = cr.right - cr.left;
                int clh = cr.bottom - cr.top;
                if (cx < 0) cx = 0;
                if (cy < 0) cy = 0;
                if (clw > (int)fullW - cx) clw = (int)fullW - cx;   // maximized: client can exceed bounds
                if (clh > (int)fullH - cy) clh = (int)fullH - cy;
                if (clw > 0 && clh > 0)
                {
                    cropX = cx; cropY = cy;
                    cw = (uint32_t)clw; ch = (uint32_t)clh;
                    clientScreenX = tl.x; clientScreenY = tl.y;
                    if (cropX || cropY || cw != fullW || ch != fullH)
                        LOG("client crop: window %ux%u -> client %ux%u at offset (%d,%d)\n",
                            fullW, fullH, cw, ch, cropX, cropY);
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

        // HDR mode captures FP16 scRGB (linear, 709 primaries, 1.0 = 80 nits); the server
        // re-encodes to BT.2020 PQ. SDR stays 8-bit BGRA.
        pool = wgc::Direct3D11CaptureFramePool::CreateFreeThreaded(
            rtDev, g_hdr ? wgdx::DirectXPixelFormat::R16G16B16A16Float
                         : wgdx::DirectXPixelFormat::B8G8R8A8UIntNormalized, 2, sz);
        session = pool.CreateCaptureSession(item);
        try { session.IsCursorCaptureEnabled(false); }
        catch (...) { LOG("note: cursor exclusion unavailable, cursor will be baked into capture\n"); }
        try
        {
            // yellow-border suppression (Win11); best effort
            wgc::GraphicsCaptureAccess::RequestAccessAsync(wgc::GraphicsCaptureAccessKind::Borderless).get();
            session.IsBorderRequired(false);
        }
        catch (...) { LOG("note: capture border suppression unavailable (pre-Win11?)\n"); }

        D3D11_TEXTURE2D_DESC sd{};
        sd.Width = cw; sd.Height = ch;
        sd.MipLevels = 1; sd.ArraySize = 1;
        sd.Format = g_hdr ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_B8G8R8A8_UNORM;
        sd.SampleDesc = { 1, 0 };
        sd.Usage = D3D11_USAGE_STAGING;
        sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        CHECK_HR(dev11->CreateTexture2D(&sd, nullptr, &staging));

        item.Closed([this](auto&&, auto&&) { InterlockedExchange(&closed, 1); SetEvent(evt); });
        pool.FrameArrived([this](auto&&, auto&&) { InterlockedExchange(&arrived, 1); SetEvent(evt); });
        session.StartCapture();
        return 0;
    }

    // drain the pool to the newest frame; -3 = none pending, -2 = resized, -1 = failure,
    // else 0 and `tex` holds the frame texture (caller must Close `frame`)
    int drainNewest(wgc::Direct3D11CaptureFrame& frame, winrt::com_ptr<ID3D11Texture2D>& tex)
    {
        for (;;)
        {
            auto f = pool.TryGetNextFrame();
            if (!f) break;
            const int64_t ts = f.SystemRelativeTime().count();
            if (lastArrTs && ts > lastArrTs)
            {
                const double d = (ts - lastArrTs) / 10000.0;
                // >300ms = idle/static gap (1 Hz hold refresh), not cadence; skip like the pacer
                if (d < 300.0) emaArrMs = emaArrMs > 0 ? emaArrMs * 0.8 + d * 0.2 : d;
            }
            lastArrTs = ts;
            if (frame) { frame.Close(); dropped++; }   // superseded = the pipeline fell behind
            frame = f;
        }
        if (!frame) return -3;

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
                if (!g_monitor && targetWnd
                    && (IsIconic(targetWnd)
                        || (SUCCEEDED(DwmGetWindowAttribute(targetWnd, DWMWA_EXTENDED_FRAME_BOUNDS, &wb, sizeof(wb)))
                            && abs((int)(wb.right - wb.left) - (int)fullW) <= 2
                            && abs((int)(wb.bottom - wb.top) - (int)fullH) <= 2)))
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
        lastFrameTs = frame.SystemRelativeTime().count();   // capture->present latency anchor
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
        if (targetWnd) InvalidateRect(targetWnd, nullptr, FALSE);
    }

    // Copy the client sub-region of the full-window capture into dst (a cw x ch resource).
    // No crop (borderless / fullscreen / monitor) falls back to the plain full copy.
    void copyCropped(ID3D11Resource* dst, ID3D11Texture2D* src)
    {
        if (cropX || cropY || cw != fullW || ch != fullH)
        {
            D3D11_BOX box{ (UINT)cropX, (UINT)cropY, 0, (UINT)cropX + cw, (UINT)cropY + ch, 1 };
            ctx11->CopySubresourceRegion(dst, 0, 0, 0, 0, src, 0, &box);
        }
        else
        {
            ctx11->CopyResource(dst, src);
        }
    }

    // interop path: newest frame -> sharedTex, GPU copy only (python reads it via CUDA after
    // the fence signal). Same return codes as latestFrame.
    int latestFrameGpu()
    {
        wgc::Direct3D11CaptureFrame frame{ nullptr };
        winrt::com_ptr<ID3D11Texture2D> tex;
        int rc = drainNewest(frame, tex);
        if (rc == -3) return 0;
        if (rc) return rc;
        liveNrPreCopy(*this);   // the previous frame's NR list must be done with the texture
        copyCropped(sharedTex.Get(), tex.get());
        frame.Close();
        return 1;
    }

    // interop path: order the shared-texture copy against python's CUDA reads
    void signalFence(uint64_t v)
    {
        // With the NR pass on, the NR queue signals this value after rewriting the texture
        if (liveNrSignal(*this, v)) return;
        ctx4->Signal(sharedFence.Get(), v);
        ctx11->Flush();   // the immediate context may defer submission; python is waiting
    }

    // CPU path: drain + staging readback into out (tight cw*4 rows). The frame stays BGRA
    // end-to-end on the server/identity routes (interpolation is channel-agnostic and the
    // present textures are B8G8R8A8, so nothing ever swizzles); only the DLSS-G route sets
    // `swizzle` to keep its verified RGBA pipeline byte-identical. Returns
    // 1 = got a frame, 0 = none pending, -1 = failure, -2 = content resized.
    int latestFrame(uint8_t* out)
    {
        wgc::Direct3D11CaptureFrame frame{ nullptr };
        winrt::com_ptr<ID3D11Texture2D> tex;
        int rc = drainNewest(frame, tex);
        if (rc == -3) return 0;
        if (rc) return rc;

        // dlssg HDR: pack scRGB FP16 -> R10A2 PQ on the GPU, read back 4 B/px.
        // The R32_UINT staging bytes ARE the R10G10B10A2_UNORM bit pattern presentFrame uploads.
        if (hdrPack)
        {
            copyCropped(convSrc.Get(), tex.get());
            frame.Close();
            ctx11->CSSetShader(convCS.Get(), nullptr, 0);
            ID3D11ShaderResourceView* srv = convSrcView.Get();
            ctx11->CSSetShaderResources(0, 1, &srv);
            ID3D11UnorderedAccessView* uav = convUav.Get();
            ctx11->CSSetUnorderedAccessViews(0, 1, &uav, nullptr);
            ctx11->Dispatch((cw + 7) / 8, (ch + 7) / 8, 1);
            srv = nullptr; uav = nullptr;   // unbind so next frame's copy into convSrc is hazard-free
            ctx11->CSSetShaderResources(0, 1, &srv);
            ctx11->CSSetUnorderedAccessViews(0, 1, &uav, nullptr);
            ctx11->CopyResource(convStaging.Get(), convDst.Get());
            D3D11_MAPPED_SUBRESOURCE pm{};
            if (FAILED(ctx11->Map(convStaging.Get(), 0, D3D11_MAP_READ, 0, &pm))) return -1;
            for (uint32_t y = 0; y < ch; y++)
                memcpy(out + (size_t)y * cw * 4,
                       (const uint8_t*)pm.pData + (size_t)y * pm.RowPitch, (size_t)cw * 4);
            ctx11->Unmap(convStaging.Get(), 0);
            return 1;
        }

        copyCropped(staging.Get(), tex.get());
        frame.Close();

        D3D11_MAPPED_SUBRESOURCE map{};
        if (FAILED(ctx11->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &map))) return -1;
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
                uint32_t v = ((const uint32_t*)s)[x];   // BGRA in memory = 0xAARRGGBB little-endian
                d[x] = (v & 0xFF00FF00u) | ((v & 0x00FF0000u) >> 16) | ((v & 0x000000FFu) << 16);
            }
        }
        ctx11->Unmap(staging.Get(), 0);
        return 1;
    }

    void stop()
    {
        if (session) session.Close();
        if (pool) pool.Close();
        if (evt) CloseHandle(evt);
        evt = nullptr;
        // the shared NT handles of the interop texture and fence (every importer released
        // them already: the native host in nativeFree, the python child at its exit)
        if (hTex) { CloseHandle(hTex); hTex = nullptr; }
        if (hFence) { CloseHandle(hFence); hFence = nullptr; }
        interop = false;
    }
};

// ---------------------------------------------------------------- DLSS 5 NR in the exe
//
// The Neural Rendering pass (NGX feature 18, engine\dlssnr\build_src\nr_host.cpp) runs INSIDE
// this process on the shared capture texture, between the D3D11 capture copy and the fence
// signal the native host waits on (its semaphore), so every backend inherits it once per
// captured frame with no protocol change. A route over pipes measured 34 to 36 ms per frame
// at 1080p; the NGX evaluate itself is about 10 ms. Ordering is GPU-only:
// D3D11 signals inFence with the frame's seq after its copy, the NR queue waits on it,
// records the three steps and signals the capture fence with the same seq; D3D11 waits on
// that value before the next copy. Colour: SDR captures (B8G8R8A8) hand the sRGB-encoded
// values to the model unchanged (the offline host's contract); HDR captures (FP16 scRGB) use
// the verbatim math of live_server.py _nr_scrgb (normalise by the SDR reference white,
// inverse sRGB EOTF, model, sRGB EOTF, pixels with any channel above SDR white keep their
// original values). The shared texture is COMMON at every list boundary (the cross-API rule);
// NGX clobbers the list's heaps, root signature and PSO, so they are re-bound after the
// evaluate. Coexistence with the native TrueHDR bridge (rtxvideo NGX over CUDA) in one
// process is measured (both orders, every eval Success); the DLSS-G route stays excluded (the
// NR host starves DLSS-G).
static const char kLiveNrCS[] =
"cbuffer C : register(b0) { float sdrWhite; uint hdr; uint w; uint h; };\n"
"Texture2D<float4> src : register(t0);\n"
"Texture2D<float4> nrout : register(t1);\n"
"RWTexture2D<float4> dst : register(u0);\n"
"float3 oetf(float3 c) { return c <= 0.0031308 ? c * 12.92 : 1.055 * pow(max(c, 1e-6), 1.0 / 2.4) - 0.055; }\n"
"float3 eotf(float3 g) { return g <= 0.04045 ? g / 12.92 : pow((g + 0.055) / 1.055, 2.4); }\n"
"[numthreads(8,8,1)]\n"
"void csIn(uint3 id : SV_DispatchThreadID)\n"
"{\n"
"    if (id.x >= w || id.y >= h) return;\n"
"    float3 c = src[id.xy].rgb;\n"
"    if (hdr) c = oetf(saturate(c / sdrWhite));\n"
"    dst[id.xy] = float4(c, 1.0);\n"
"}\n"
"[numthreads(8,8,1)]\n"
"void csOut(uint3 id : SV_DispatchThreadID)\n"
"{\n"
"    if (id.x >= w || id.y >= h) return;\n"
"    float4 o = src[id.xy];\n"
"    float3 g = saturate(nrout[id.xy].rgb);\n"
"    if (hdr)\n"
"    {\n"
"        float3 lin = o.rgb / sdrWhite;\n"
"        bool over = any(lin > 1.0);\n"
"        dst[id.xy] = float4((over ? lin : eotf(g)) * sdrWhite, o.a);\n"
"    }\n"
"    else dst[id.xy] = float4(g, o.a);\n"
"}\n";

// Identical-frame reuse: the capture against the last evaluated one (prev), every channel exactly
// (UNORM and FP16 loads map one to one onto their codes); any difference stores this frame's tag,
// so the flag never needs clearing and a stale value can never read as "differs".
static const char kLiveNrCmpCS[] =
"cbuffer C : register(b0) { uint tag; uint pad; uint w; uint h; };\n"
"Texture2D<float4> cur : register(t0);\n"
"Texture2D<float4> prev : register(t1);\n"
"RWByteAddressBuffer flag : register(u0);\n"
"[numthreads(8,8,1)]\n"
"void csCmp(uint3 id : SV_DispatchThreadID)\n"
"{\n"
"    if (id.x >= w || id.y >= h) return;\n"
"    if (any(cur[id.xy] != prev[id.xy])) flag.Store(0, tag);\n"
"}\n";

// Live DLSS 5 motion (DLSSNR.MVec), the live form of the offline k_nvofLuma / k_nvofUp / k_nrMv.
// csLuma: the BT.709 luma (8-bit codes) of the colour csIn hands DLSS 5, into an R8 texture that
// is copied into this frame's Optical Flow input slot. csMv, after NVOFA (current -> previous,
// grid 4): the grid field bilinear to every pixel (integer-ratio taps, exact at grid 4, raw / 32
// = px), then a vector is kept only where it explains its 5x5 luma window better than no motion
// by more than margin (the previous frame bilinear at the moved position, the tap = the pixel
// plus the floor and the exact fraction of the vector, border clamped); zero elsewhere, what a
// still pixel gets from a game engine. zero = the whole field zero (a stream's first frame).
static const char kLiveNrLumaCS[] =
"cbuffer C : register(b0) { float sdrWhite; uint hdr; uint w; uint h; };\n"
"Texture2D<float4> src : register(t0);\n"
"RWTexture2D<float> luma : register(u0);\n"
"float3 oetf(float3 c) { return c <= 0.0031308 ? c * 12.92 : 1.055 * pow(max(c, 1e-6), 1.0 / 2.4) - 0.055; }\n"
"[numthreads(8,8,1)]\n"
"void csLuma(uint3 id : SV_DispatchThreadID)\n"
"{\n"
"    if (id.x >= w || id.y >= h) return;\n"
"    float3 c = src[id.xy].rgb;\n"
"    if (hdr) c = oetf(saturate(c / sdrWhite));\n"
"    luma[id.xy] = round(saturate(dot(float3(0.2126, 0.7152, 0.0722), c)) * 255.0) / 255.0;\n"
"}\n";

static const char kLiveNrMvCS[] =
"cbuffer C : register(b0) { uint w; uint h; uint gw; uint gh; uint grid; uint zero; float margin; uint pad; };\n"
"Texture2D<int2> flow : register(t0);\n"
"Texture2D<float> lumCur : register(t1);\n"
"Texture2D<float> lumPrev : register(t2);\n"
"RWTexture2D<float2> mv : register(u0);\n"
"void upTap(int o, int g, out int i0, out int i1, out float l)\n"
"{\n"
"    const int num = 2 * o + 1 - (int)grid, den = 2 * (int)grid;\n"
"    int i = num > 0 ? num / den : 0;\n"
"    if (i > g - 1) i = g - 1;\n"
"    const int rem = num > 0 ? num - i * den : 0;\n"
"    l = rem >= den ? 1.0 : (float)rem / (float)den;\n"
"    i0 = i;\n"
"    i1 = i < g - 1 ? i + 1 : i;\n"
"}\n"
"float code(Texture2D<float> t, int x, int y) { return floor(t.Load(int3(x, y, 0)) * 255.0 + 0.5); }\n"
"[numthreads(8,8,1)]\n"
"void csMv(uint3 id : SV_DispatchThreadID)\n"
"{\n"
"    if (id.x >= w || id.y >= h) return;\n"
"    const int x = (int)id.x, y = (int)id.y, W = (int)w, H = (int)h;\n"
"    float2 o = float2(0.0, 0.0);\n"
"    if (!zero)\n"
"    {\n"
"        int x0, x1, y0, y1;\n"
"        float lx, ly;\n"
"        upTap(x, (int)gw, x0, x1, lx);\n"
"        upTap(y, (int)gh, y0, y1, ly);\n"
"        const float2 a = (1.0 - lx) * (float2)flow.Load(int3(x0, y0, 0)) + lx * (float2)flow.Load(int3(x1, y0, 0));\n"
"        const float2 b = (1.0 - lx) * (float2)flow.Load(int3(x0, y1, 0)) + lx * (float2)flow.Load(int3(x1, y1, 0));\n"
"        const float2 f = ((1.0 - ly) * a + ly * b) * (1.0 / 32.0);\n"
"        if (f.x != 0.0 || f.y != 0.0)\n"
"        {\n"
"            const float fu = floor(f.x), fv = floor(f.y);\n"
"            const int iu = (int)fu, iv = (int)fv;\n"
"            const float kx = f.x - fu, ky = f.y - fv;\n"
"            float e0 = 0.0, e1 = 0.0;\n"
"            for (int dy = -2; dy <= 2; dy++)\n"
"            {\n"
"                const int yy = clamp(y + dy, 0, H - 1);\n"
"                const int r0 = clamp(yy + iv, 0, H - 1), r1 = clamp(yy + iv + 1, 0, H - 1);\n"
"                for (int dx = -2; dx <= 2; dx++)\n"
"                {\n"
"                    const int xx = clamp(x + dx, 0, W - 1);\n"
"                    const int c0 = clamp(xx + iu, 0, W - 1), c1 = clamp(xx + iu + 1, 0, W - 1);\n"
"                    const float c = code(lumCur, xx, yy);\n"
"                    const float pa = (1.0 - kx) * code(lumPrev, c0, r0) + kx * code(lumPrev, c1, r0);\n"
"                    const float pb = (1.0 - kx) * code(lumPrev, c0, r1) + kx * code(lumPrev, c1, r1);\n"
"                    e0 += abs(c - code(lumPrev, xx, yy));\n"
"                    e1 += abs(c - ((1.0 - ky) * pa + ky * pb));\n"
"                }\n"
"            }\n"
"            if (e1 + margin < e0) o = f;\n"
"        }\n"
"    }\n"
"    mv[id.xy] = o;\n"
"}\n";

// the D3D12 Optical Flow entry points, loaded once per process from the driver's nvofapi64.dll
static NV_OF_D3D12_API_FUNCTION_LIST g_nrOf{};
static bool nrOfLoad(std::string& why)
{
    static bool loaded = false;
    if (loaded) return true;
    wchar_t sys[MAX_PATH]{};
    const UINT n = GetSystemDirectoryW(sys, MAX_PATH);
    if (!n || n >= MAX_PATH) { why = "the system folder is unknown"; return false; }
    HMODULE m = LoadLibraryExW((std::wstring(sys) + L"\\nvofapi64.dll").c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!m) { why = "nvofapi64.dll is not in the system folder"; return false; }
    typedef NV_OF_STATUS(NVOFAPI* PFN_Create)(uint32_t, NV_OF_D3D12_API_FUNCTION_LIST*);
    auto create = (PFN_Create)GetProcAddress(m, "NvOFAPICreateInstanceD3D12");
    if (!create) { why = "nvofapi64.dll exports no D3D12 interface"; return false; }
    if (create(NV_OF_API_VERSION, &g_nrOf) != NV_OF_SUCCESS || !g_nrOf.nvCreateOpticalFlowD3D12 || !g_nrOf.nvOFInit
        || !g_nrOf.nvOFRegisterResourceD3D12 || !g_nrOf.nvOFUnregisterResourceD3D12 || !g_nrOf.nvOFExecuteD3D12
        || !g_nrOf.nvOFDestroy)
    { why = "the driver refused the Optical Flow SDK 5 D3D12 interface"; return false; }
    loaded = true;
    return true;
}

// kNrMvMargin of the offline route (native.inl): one 8-bit level per pixel of the 5x5 window
static const float kLiveNrMvMargin = 25.0f;

struct LiveNr
{
    bool active = false;
    uint32_t w = 0, h = 0;
    bool hdr = false;
    float sdrWhite = 3.0f;                       // scRGB scale of the SDR reference white (nits / 80)
    nr::Host host;
    ComPtr<ID3D12Device> dev;
    ComPtr<ID3D12CommandQueue> queue;            // own DIRECT queue, never the present queue
    ComPtr<ID3D12CommandAllocator> allocs[2];
    uint64_t allocFence[2]{};
    UINT allocIdx = 0;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> done;                    // completion fence, gates allocator reuse
    uint64_t doneValue = 0;
    HANDLE doneEvent = nullptr;
    ComPtr<ID3D12Fence> capFence12;              // the capture's shared fence, opened here
    ComPtr<ID3D11Fence> inFence11;               // D3D11 copy -> NR queue ordering
    HANDLE hInFence = nullptr;
    ComPtr<ID3D12Fence> inFence12;
    ComPtr<ID3D12Resource> shared12;             // the capture's shared texture, opened here
    ComPtr<ID3D12Resource> outTex;               // capture format, UAV: csOut target, copied back
    ComPtr<ID3D12RootSignature> rs;
    ComPtr<ID3D12PipelineState> psoIn, psoOut;
    ComPtr<ID3D12DescriptorHeap> heap;           // 6 descriptors: set A (csIn) at 0, set B (csOut) at 3
    UINT descSize = 0;
    uint64_t lastSeq = 0;                        // last seq the NR queue signalled
    uint64_t frames = 0;
    double recMs = 0;                            // CPU ms spent recording and submitting
    // DLSS 5 motion vectors (DLSSNR.MVec): an Optical Flow session on this device, current ->
    // previous for every frame; with them DLSS 5 keeps its history from a stream's second frame on
    bool motion = false;
    bool fresh = true;                           // the next frame starts a stream: a Reset, a zero field
    std::string mvNote;                          // the ready line's motion part
    NvOFHandle of = nullptr;
    NvOFGPUBufferHandle ofBuf[3]{};              // the two luma slots and the flow grid, registered
    ComPtr<ID3D12Resource> lumaTex;              // R8 UAV: csLuma's target, copied into a slot
    ComPtr<ID3D12Resource> slot[2];              // R8 Optical Flow inputs, COMMON between uses
    ComPtr<ID3D12Resource> flowTex;              // R16G16_SINT grid field, COMMON between uses
    ComPtr<ID3D12Fence> ofIn, ofOut;             // NR queue -> Optical Flow -> NR queue
    uint64_t ofInValue = 0, ofOutValue = 0;
    UINT cur = 0;                                // the slot this frame's luma goes into
    uint32_t gw = 0, gh = 0;
    ComPtr<ID3D12GraphicsCommandList> list2;     // the evaluate list when a luma list runs first
    ComPtr<ID3D12RootSignature> rsMv;
    ComPtr<ID3D12PipelineState> psoLuma, psoMv;
    // SMV_LIVE_NR_MVDUMP=<path prefix> (diagnostics): the field and the luma of frames 30..37 as raw
    // files, <prefix>_f<n>_<w>x<h>_mv.f16 (R16G16_FLOAT px, current -> previous) and _luma.u8
    std::wstring dumpPrefix;
    ComPtr<ID3D12Resource> mvRb, lumaRb;
    uint64_t mvRbPitch = 0, lumaRbPitch = 0;
    // Identical frames (a paused or held picture): the capture is compared on this queue with the
    // last evaluated one and the answer read back; equal = the last output goes back into the
    // shared texture with no evaluate, so the picture stays exactly still (the kept history would
    // re-shade it a code or two every refresh). SMV_NR_REUSE=0 = every frame evaluated
    bool reuse = false;
    bool havePrev = false;                       // prevTex holds this stream's last evaluated capture
    ComPtr<ID3D12Resource> prevTex;              // capture format, COMMON between uses
    ComPtr<ID3D12Resource> cmpBuf, cmpRb;        // the difference tag (raw UAV) and its readback
    ComPtr<ID3D12PipelineState> psoCmp;
    uint32_t cmpTag = 0;
    uint64_t reused = 0;

    static D3D12_RESOURCE_BARRIER tr(ID3D12Resource* r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b)
    {
        D3D12_RESOURCE_BARRIER x{};
        x.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        x.Transition.pResource = r;
        x.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        x.Transition.StateBefore = a;
        x.Transition.StateAfter = b;
        return x;
    }

    bool init(Host& h12, Capture& cap, const std::wstring& nrDir, std::string& err)
    {
        w = cap.cw; h = cap.ch; hdr = g_hdr;
        sdrWhite = (float)(g_sdrWhite / 80.0);
        if (sdrWhite < 0.5f) sdrWhite = 0.5f;
        dev = h12.device;
        const DXGI_FORMAT capFmt = hdr ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_B8G8R8A8_UNORM;
        // csOut stores into a texture of the capture format: FP16 typed UAV stores are always
        // supported, B8G8R8A8_UNORM ones are optional in D3D12
        D3D12_FEATURE_DATA_FORMAT_SUPPORT fs{ capFmt };
        if (FAILED(dev->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &fs, sizeof(fs)))
            || !(fs.Support1 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW))
        { err = "no typed UAV support for the capture format"; return false; }
        if (FAILED(dev->OpenSharedHandle(cap.hTex, IID_PPV_ARGS(&shared12))))
        { err = "OpenSharedHandle(capture texture) failed"; return false; }
        if (FAILED(dev->OpenSharedHandle(cap.hFence, IID_PPV_ARGS(&capFence12))))
        { err = "OpenSharedHandle(capture fence) failed"; return false; }
        ComPtr<ID3D11Device5> dev5;
        if (FAILED(cap.dev11.As(&dev5))) { err = "D3D11.4 device unavailable"; return false; }
        if (FAILED(dev5->CreateFence(0, D3D11_FENCE_FLAG_SHARED, IID_PPV_ARGS(&inFence11)))
            || FAILED(inFence11->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &hInFence))
            || FAILED(dev->OpenSharedHandle(hInFence, IID_PPV_ARGS(&inFence12))))
        { err = "input fence creation failed"; return false; }
        D3D12_COMMAND_QUEUE_DESC qd{};
        qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        if (FAILED(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)))) { err = "CreateCommandQueue failed"; return false; }
        for (auto& a : allocs)
            if (FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&a))))
            { err = "CreateCommandAllocator failed"; return false; }
        if (FAILED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocs[0].Get(), nullptr, IID_PPV_ARGS(&list))))
        { err = "CreateCommandList failed"; return false; }
        list->Close();
        if (FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&done)))) { err = "CreateFence failed"; return false; }
        doneEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        {
            D3D12_HEAP_PROPERTIES hp{ D3D12_HEAP_TYPE_DEFAULT };
            D3D12_RESOURCE_DESC rd{};
            rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            rd.Width = w; rd.Height = h;
            rd.DepthOrArraySize = 1; rd.MipLevels = 1;
            rd.Format = capFmt;
            rd.SampleDesc = { 1, 0 };
            rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            if (FAILED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                                    nullptr, IID_PPV_ARGS(&outTex))))
            { err = "write-back texture creation failed"; return false; }
        }
        ComPtr<ID3DBlob> csIn, csOut, e;
        if (FAILED(D3DCompile(kLiveNrCS, sizeof(kLiveNrCS) - 1, nullptr, nullptr, nullptr, "csIn", "cs_5_0", 0, 0, &csIn, &e))
            || FAILED(D3DCompile(kLiveNrCS, sizeof(kLiveNrCS) - 1, nullptr, nullptr, nullptr, "csOut", "cs_5_0", 0, 0, &csOut, &e)))
        { err = std::string("shader compile failed: ") + (e ? (const char*)e->GetBufferPointer() : "?"); return false; }
        // root signature: b0 = 4 root constants, one table = t0..t1 + u0
        if (!rootSig(4, 2, rs)) { err = "root signature creation failed"; return false; }
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
        pd.pRootSignature = rs.Get();
        pd.CS = { csIn->GetBufferPointer(), csIn->GetBufferSize() };
        if (FAILED(dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&psoIn)))) { err = "csIn pipeline state failed"; return false; }
        pd.CS = { csOut->GetBufferPointer(), csOut->GetBufferSize() };
        if (FAILED(dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&psoOut)))) { err = "csOut pipeline state failed"; return false; }
        // the motion field, set up BEFORE the feature (it binds MVec at create); without it the pass
        // runs as it always did, every frame a Reset
        {
            std::string why;
            motion = motionInit(why);
            if (!motion)
            {
                motionFree();
                mvNote = "no motion vectors (" + why + "), every frame a Reset";
            }
            else mvNote = "motion vectors (NVOFA grid 4), history kept";
        }
        // the feature on this device (Color and Output are owned by the host); quiet NGX log
        // (the core still writes engine\dlssnr\nvngx.log), the two "dlssnr:" lines stay
        nr::setModuleDir(nrDir.c_str());
        nr::Settings set;
        set.structure = (float)g_nrStructure;
        set.tone = (float)g_nrTone;
        set.style = (g_nrStyle >= 0 && g_nrStyle <= 2) ? g_nrStyle : 1;
        set.passes = g_nrPasses < 1 ? 1 : (g_nrPasses > nr::kMaxPasses ? nr::kMaxPasses : g_nrPasses);
        set.motion = motion;
        set.motionUav = motion;   // csMv writes MVec in place
        nr::Variant var;
        if (host.startupOn(dev.Get(), queue.Get(), w, h, set, var, true, err) != 0) return false;
        // descriptors: set A (csIn) = shared SRV, null SRV, Color UAV; set B (csOut) = shared
        // SRV, Output SRV, outTex UAV; with motion, set L (csLuma, rs) at 6 = shared SRV, null SRV,
        // luma UAV, and sets M0 / M1 (csMv, rsMv) at 9 / 13 = flow SRV, this frame's slot SRV, the
        // other slot SRV, MVec UAV (M0 when slot 0 holds this frame, M1 when slot 1 does); with
        // reuse, set C (csCmp, rs) at 17 = shared SRV, prevTex SRV, the tag buffer's raw UAV
        D3D12_DESCRIPTOR_HEAP_DESC hd{};
        hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        hd.NumDescriptors = 20;
        hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        if (FAILED(dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)))) { err = "descriptor heap creation failed"; return false; }
        descSize = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        auto at = [&](UINT i)
        {
            D3D12_CPU_DESCRIPTOR_HANDLE c = heap->GetCPUDescriptorHandleForHeapStart();
            c.ptr += (SIZE_T)i * descSize;
            return c;
        };
        D3D12_SHADER_RESOURCE_VIEW_DESC nullSrv{};
        nullSrv.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        nullSrv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        nullSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        nullSrv.Texture2D.MipLevels = 1;
        dev->CreateShaderResourceView(shared12.Get(), nullptr, at(0));
        dev->CreateShaderResourceView(nullptr, &nullSrv, at(1));
        dev->CreateUnorderedAccessView(host.color(), nullptr, nullptr, at(2));
        dev->CreateShaderResourceView(shared12.Get(), nullptr, at(3));
        dev->CreateShaderResourceView(host.output(), nullptr, at(4));
        dev->CreateUnorderedAccessView(outTex.Get(), nullptr, nullptr, at(5));
        if (motion)
        {
            dev->CreateShaderResourceView(shared12.Get(), nullptr, at(6));
            dev->CreateShaderResourceView(nullptr, &nullSrv, at(7));
            dev->CreateUnorderedAccessView(lumaTex.Get(), nullptr, nullptr, at(8));
            for (UINT k = 0; k < 2; k++)
            {
                const UINT base = 9 + 4 * k;
                dev->CreateShaderResourceView(flowTex.Get(), nullptr, at(base));
                dev->CreateShaderResourceView(slot[k].Get(), nullptr, at(base + 1));
                dev->CreateShaderResourceView(slot[k ^ 1].Get(), nullptr, at(base + 2));
                dev->CreateUnorderedAccessView(host.motion(), nullptr, nullptr, at(base + 3));
            }
            dumpInit();
        }
        {
            wchar_t rv[8]{};
            havePrev = false;
            cmpTag = 0;
            reused = 0;
            reuse = !(GetEnvironmentVariableW(L"SMV_NR_REUSE", rv, 8) && rv[0] == L'0');
            if (reuse && !reuseInit(capFmt, at(17)))
            {
                reuse = false;
                LOG("live DLSS 5 native: identical-frame reuse unavailable, every frame evaluated\n");
            }
        }
        active = true;
        return true;
    }

    // the reuse resources: prevTex, the tag buffer and its readback, csCmp; the descriptors from d
    bool reuseInit(DXGI_FORMAT capFmt, D3D12_CPU_DESCRIPTOR_HANDLE d)
    {
        D3D12_HEAP_PROPERTIES hp{ D3D12_HEAP_TYPE_DEFAULT };
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rd.Width = w; rd.Height = h;
        rd.DepthOrArraySize = 1; rd.MipLevels = 1;
        rd.Format = capFmt;
        rd.SampleDesc = { 1, 0 };
        if (FAILED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON,
                                                nullptr, IID_PPV_ARGS(&prevTex))))
            return false;
        D3D12_RESOURCE_DESC bd{};
        bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bd.Width = 256; bd.Height = 1;
        bd.DepthOrArraySize = 1; bd.MipLevels = 1;
        bd.Format = DXGI_FORMAT_UNKNOWN;
        bd.SampleDesc = { 1, 0 };
        bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        bd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        if (FAILED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COMMON,
                                                nullptr, IID_PPV_ARGS(&cmpBuf))))
            return false;
        D3D12_HEAP_PROPERTIES rp{ D3D12_HEAP_TYPE_READBACK };
        bd.Flags = D3D12_RESOURCE_FLAG_NONE;
        if (FAILED(dev->CreateCommittedResource(&rp, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST,
                                                nullptr, IID_PPV_ARGS(&cmpRb))))
            return false;
        ComPtr<ID3DBlob> cs, e;
        if (FAILED(D3DCompile(kLiveNrCmpCS, sizeof(kLiveNrCmpCS) - 1, nullptr, nullptr, nullptr, "csCmp", "cs_5_0", 0, 0, &cs, &e)))
            return false;
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
        pd.pRootSignature = rs.Get();
        pd.CS = { cs->GetBufferPointer(), cs->GetBufferSize() };
        if (FAILED(dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&psoCmp)))) return false;
        dev->CreateShaderResourceView(shared12.Get(), nullptr, d);
        d.ptr += descSize;
        dev->CreateShaderResourceView(prevTex.Get(), nullptr, d);
        d.ptr += descSize;
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud{};
        ud.Format = DXGI_FORMAT_R32_TYPELESS;
        ud.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        ud.Buffer.FirstElement = 0;
        ud.Buffer.NumElements = 64;
        ud.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        dev->CreateUnorderedAccessView(cmpBuf.Get(), nullptr, &ud, d);
        return true;
    }

    // a compute root signature: b0 = nConst root constants, one table = nSrv SRVs from t0 + one UAV
    // at u0. Serialized through d3d12.dll by name: the exe links no d3d12.lib (D3D12CreateDevice
    // must keep coming from sl.interposer.lib).
    bool rootSig(UINT nConst, UINT nSrv, ComPtr<ID3D12RootSignature>& out)
    {
        D3D12_DESCRIPTOR_RANGE ranges[2]{};
        ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        ranges[0].NumDescriptors = nSrv;
        ranges[0].OffsetInDescriptorsFromTableStart = 0;
        ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        ranges[1].NumDescriptors = 1;
        ranges[1].OffsetInDescriptorsFromTableStart = nSrv;
        D3D12_ROOT_PARAMETER params[2]{};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants.Num32BitValues = nConst;
        params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[1].DescriptorTable.NumDescriptorRanges = 2;
        params[1].DescriptorTable.pDescriptorRanges = ranges;
        params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        D3D12_ROOT_SIGNATURE_DESC rsd{};
        rsd.NumParameters = 2;
        rsd.pParameters = params;
        typedef HRESULT (WINAPI* PFN_Serialize)(const D3D12_ROOT_SIGNATURE_DESC*, D3D_ROOT_SIGNATURE_VERSION, ID3DBlob**, ID3DBlob**);
        HMODULE d3d12 = GetModuleHandleW(L"d3d12.dll");
        if (!d3d12) d3d12 = LoadLibraryW(L"d3d12.dll");
        auto serialize = d3d12 ? (PFN_Serialize)GetProcAddress(d3d12, "D3D12SerializeRootSignature") : nullptr;
        ComPtr<ID3DBlob> sig, e;
        return serialize && SUCCEEDED(serialize(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &e))
            && SUCCEEDED(dev->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(), IID_PPV_ARGS(&out)));
    }

    // The Optical Flow session for DLSS 5's motion vectors: FORWARD only (the current frame is the
    // input, the previous one the reference), grid 4, FAST, no cost, 8-bit luma, like the offline
    // route's; its two luma slots, the grid texture, the shaders and the second list. false = no
    // motion for this session (why says it; the caller frees what was made).
    bool motionInit(std::string& why)
    {
        fresh = true;
        cur = 0;
        ofInValue = ofOutValue = 0;
        wchar_t ev[8]{};
        if (GetEnvironmentVariableW(L"SMV_NR_MV", ev, 8) && ev[0] == L'0') { why = "SMV_NR_MV=0"; return false; }
        if (GetEnvironmentVariableW(L"SMV_NR_RESET_EVERY", ev, 8) && ev[0] == L'1') { why = "SMV_NR_RESET_EVERY=1"; return false; }
        if (w < 32 || h < 32) { why = "below the Optical Flow minimum 32x32"; return false; }
        if (!nrOfLoad(why)) return false;
        if (g_nrOf.nvCreateOpticalFlowD3D12(dev.Get(), &of) != NV_OF_SUCCESS) { of = nullptr; why = "nvCreateOpticalFlowD3D12 failed"; return false; }
        NV_OF_INIT_PARAMS ip{};
        ip.width = w;
        ip.height = h;
        ip.outGridSize = NV_OF_OUTPUT_VECTOR_GRID_SIZE_4;
        ip.hintGridSize = NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED;
        ip.mode = NV_OF_MODE_OPTICALFLOW;
        ip.perfLevel = NV_OF_PERF_LEVEL_FAST;
        ip.enableExternalHints = NV_OF_FALSE;
        ip.enableOutputCost = NV_OF_FALSE;
        ip.disparityRange = NV_OF_STEREO_DISPARITY_RANGE_UNDEFINED;
        ip.enableRoi = NV_OF_FALSE;
        ip.predDirection = NV_OF_PRED_DIRECTION_FORWARD;
        ip.enableGlobalFlow = NV_OF_FALSE;
        ip.inputBufferFormat = NV_OF_BUFFER_FORMAT_GRAYSCALE8;
        if (g_nrOf.nvOFInit(of, &ip) != NV_OF_SUCCESS) { why = "nvOFInit refused " + std::to_string(w) + "x" + std::to_string(h); return false; }
        gw = (w + 3) / 4;
        gh = (h + 3) / 4;
        auto tex = [&](uint32_t tw, uint32_t th, DXGI_FORMAT f, bool uav, D3D12_RESOURCE_STATES st, ComPtr<ID3D12Resource>& out)
        {
            D3D12_HEAP_PROPERTIES hp{ D3D12_HEAP_TYPE_DEFAULT };
            D3D12_RESOURCE_DESC rd{};
            rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            rd.Width = tw; rd.Height = th;
            rd.DepthOrArraySize = 1; rd.MipLevels = 1;
            rd.Format = f;
            rd.SampleDesc = { 1, 0 };
            rd.Flags = uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
            return SUCCEEDED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, st, nullptr, IID_PPV_ARGS(&out)));
        };
        if (!tex(w, h, DXGI_FORMAT_R8_UNORM, true, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, lumaTex)
            || !tex(w, h, DXGI_FORMAT_R8_UNORM, false, D3D12_RESOURCE_STATE_COMMON, slot[0])
            || !tex(w, h, DXGI_FORMAT_R8_UNORM, false, D3D12_RESOURCE_STATE_COMMON, slot[1])
            || !tex(gw, gh, DXGI_FORMAT_R16G16_SINT, true, D3D12_RESOURCE_STATE_COMMON, flowTex))
        { why = "motion texture creation failed"; return false; }
        if (FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&ofIn)))
            || FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&ofOut))))
        { why = "motion fence creation failed"; return false; }
        ID3D12Resource* regs[3] = { slot[0].Get(), slot[1].Get(), flowTex.Get() };
        for (int i = 0; i < 3; i++)
        {
            NV_OF_REGISTER_RESOURCE_PARAMS_D3D12 rp{};
            rp.resource = regs[i];
            rp.inputFencePoint = { ofIn.Get(), ofInValue };
            rp.hOFGpuBuffer = &ofBuf[i];
            rp.outputFencePoint = { ofOut.Get(), ++ofOutValue };
            if (g_nrOf.nvOFRegisterResourceD3D12(of, &rp) != NV_OF_SUCCESS)
            { ofBuf[i] = nullptr; why = "Optical Flow buffer registration failed"; return false; }
            if (ofOut->GetCompletedValue() < ofOutValue)
            {
                ofOut->SetEventOnCompletion(ofOutValue, doneEvent);
                if (WaitForSingleObject(doneEvent, 5000) != WAIT_OBJECT_0)
                { why = "Optical Flow buffer registration did not complete within 5 s"; return false; }
            }
        }
        ComPtr<ID3DBlob> csL, csM, e;
        if (FAILED(D3DCompile(kLiveNrLumaCS, sizeof(kLiveNrLumaCS) - 1, nullptr, nullptr, nullptr, "csLuma", "cs_5_0", 0, 0, &csL, &e))
            || FAILED(D3DCompile(kLiveNrMvCS, sizeof(kLiveNrMvCS) - 1, nullptr, nullptr, nullptr, "csMv", "cs_5_0", 0, 0, &csM, &e)))
        { why = std::string("motion shader compile failed: ") + (e ? (const char*)e->GetBufferPointer() : "?"); return false; }
        // csMv: b0 = 8 root constants, one table = t0..t2 + u0
        if (!rootSig(8, 3, rsMv)) { why = "motion root signature creation failed"; return false; }
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
        pd.pRootSignature = rs.Get();
        pd.CS = { csL->GetBufferPointer(), csL->GetBufferSize() };
        if (FAILED(dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&psoLuma)))) { why = "csLuma pipeline state failed"; return false; }
        pd.pRootSignature = rsMv.Get();
        pd.CS = { csM->GetBufferPointer(), csM->GetBufferSize() };
        if (FAILED(dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&psoMv)))) { why = "csMv pipeline state failed"; return false; }
        if (FAILED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocs[0].Get(), nullptr, IID_PPV_ARGS(&list2))))
        { why = "CreateCommandList failed"; return false; }
        list2->Close();
        return true;
    }

    // everything motionInit and dumpInit made; the queue is drained (or never ran)
    void motionFree()
    {
        for (NvOFGPUBufferHandle& hb : ofBuf)
            if (hb)
            {
                NV_OF_UNREGISTER_RESOURCE_PARAMS_D3D12 up{};
                up.hOFGpuBuffer = hb;
                g_nrOf.nvOFUnregisterResourceD3D12(&up);
                hb = nullptr;
            }
        if (of) { g_nrOf.nvOFDestroy(of); of = nullptr; }
        lumaTex.Reset(); slot[0].Reset(); slot[1].Reset(); flowTex.Reset();
        ofIn.Reset(); ofOut.Reset();
        list2.Reset(); rsMv.Reset(); psoLuma.Reset(); psoMv.Reset();
        mvRb.Reset(); lumaRb.Reset();
        dumpPrefix.clear();
        motion = false;
    }

    // SMV_LIVE_NR_MVDUMP (diagnostics): readback buffers for the field and the luma
    void dumpInit()
    {
        wchar_t p[MAX_PATH]{};
        dumpPrefix.clear();
        if (!GetEnvironmentVariableW(L"SMV_LIVE_NR_MVDUMP", p, MAX_PATH) || !p[0]) return;
        auto rb = [&](DXGI_FORMAT f, uint64_t& pitch, ComPtr<ID3D12Resource>& out)
        {
            D3D12_RESOURCE_DESC td{};
            td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            td.Width = w; td.Height = h;
            td.DepthOrArraySize = 1; td.MipLevels = 1;
            td.Format = f;
            td.SampleDesc = { 1, 0 };
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
            UINT64 total = 0;
            dev->GetCopyableFootprints(&td, 0, 1, 0, &fp, nullptr, nullptr, &total);
            pitch = fp.Footprint.RowPitch;
            D3D12_HEAP_PROPERTIES hp{ D3D12_HEAP_TYPE_READBACK };
            D3D12_RESOURCE_DESC bd{};
            bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            bd.Width = total; bd.Height = 1;
            bd.DepthOrArraySize = 1; bd.MipLevels = 1;
            bd.Format = DXGI_FORMAT_UNKNOWN;
            bd.SampleDesc = { 1, 0 };
            bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            return SUCCEEDED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST,
                                                          nullptr, IID_PPV_ARGS(&out)));
        };
        if (rb(DXGI_FORMAT_R16G16_FLOAT, mvRbPitch, mvRb) && rb(DXGI_FORMAT_R8_UNORM, lumaRbPitch, lumaRb)) dumpPrefix = p;
        else LOG("live DLSS 5 native: SMV_LIVE_NR_MVDUMP readback buffers failed, no dump\n");
    }

    void dumpCopy(ID3D12GraphicsCommandList* l, ID3D12Resource* src, ID3D12Resource* rbuf, DXGI_FORMAT f, uint64_t pitch)
    {
        D3D12_TEXTURE_COPY_LOCATION s{}, d{};
        s.pResource = src;
        s.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        d.pResource = rbuf;
        d.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        d.PlacedFootprint.Footprint.Format = f;
        d.PlacedFootprint.Footprint.Width = w;
        d.PlacedFootprint.Footprint.Height = h;
        d.PlacedFootprint.Footprint.Depth = 1;
        d.PlacedFootprint.Footprint.RowPitch = (UINT)pitch;
        l->CopyTextureRegion(&d, 0, 0, 0, &s, nullptr);
    }

    // after the frame's lists were submitted: wait for them, write the two files (tight rows)
    void dumpWrite(uint64_t n)
    {
        if (done->GetCompletedValue() < doneValue)
        {
            done->SetEventOnCompletion(doneValue, doneEvent);
            WaitForSingleObject(doneEvent, 5000);
        }
        auto put = [&](ID3D12Resource* rbuf, uint64_t pitch, uint32_t bpp, const wchar_t* tag)
        {
            void* p = nullptr;
            if (FAILED(rbuf->Map(0, nullptr, &p))) return;
            const std::wstring f = dumpPrefix + L"_f" + std::to_wstring(n) + L"_" + std::to_wstring(w) + L"x"
                                   + std::to_wstring(h) + tag;
            FILE* fp = _wfopen(f.c_str(), L"wb");
            if (fp)
            {
                for (uint32_t y = 0; y < h; y++) fwrite((const uint8_t*)p + y * pitch, bpp, w, fp);
                fclose(fp);
            }
            D3D12_RANGE none{ 0, 0 };
            rbuf->Unmap(0, &none);
        };
        put(mvRb.Get(), mvRbPitch, 4, L"_mv.f16");
        put(lumaRb.Get(), lumaRbPitch, 1, L"_luma.u8");
    }

    // Before the D3D11 copy into the shared texture: the previous frame's NR list reads and
    // rewrites that texture, so the copy waits for its signal (GPU-side, immediate context).
    void preCopy(Capture& cap)
    {
        if (active && lastSeq) cap.ctx4->Wait(cap.sharedFence.Get(), lastSeq);
    }

    bool drop(const char* why)
    {
        LOG("live DLSS 5 native: %s; the pass is off for the rest of this session\n", why);
        active = false;
        return false;   // the caller signals the capture fence from D3D11 as before
    }

    // Instead of the plain D3D11 fence signal: order the copy, run the pass on the NR queue,
    // signal the capture fence with the same seq from there. false = pass off (never started
    // or just dropped itself): the caller signals from D3D11.
    bool signal(Capture& cap, uint64_t seq)
    {
        if (!active) return false;
        const int64_t t0 = nowQpc100();
        cap.ctx4->Signal(inFence11.Get(), seq);
        cap.ctx11->Flush();
        if (done->GetCompletedValue() < allocFence[allocIdx])
        {
            done->SetEventOnCompletion(allocFence[allocIdx], doneEvent);
            if (WaitForSingleObject(doneEvent, 5000) != WAIT_OBJECT_0)
                return drop("the previous NR list did not complete within 5 s");
        }
        ID3D12CommandAllocator* alloc = allocs[allocIdx].Get();
        if (FAILED(alloc->Reset()) || FAILED(list->Reset(alloc, nullptr)))
            return drop("command list Reset failed");
        struct { float sdrWhite; uint32_t hdr, w, h; } consts{ sdrWhite, hdr ? 1u : 0u, w, h };
        ID3D12DescriptorHeap* heaps[] = { heap.Get() };
        auto bind = [&](ID3D12GraphicsCommandList* l, ID3D12RootSignature* r, ID3D12PipelineState* pso,
                        UINT first, UINT nConst, const void* c)
        {
            D3D12_GPU_DESCRIPTOR_HANDLE g = heap->GetGPUDescriptorHandleForHeapStart();
            g.ptr += (UINT64)first * descSize;
            l->SetDescriptorHeaps(1, heaps);
            l->SetComputeRootSignature(r);
            l->SetPipelineState(pso);
            l->SetComputeRoot32BitConstants(0, nConst, c, 0);
            l->SetComputeRootDescriptorTable(1, g);
        };
        D3D12_RESOURCE_BARRIER b[4];
        // every list reads the captured frame: the queue waits for the D3D11 copy first
        queue->Wait(inFence12.Get(), seq);
        // identical to the last evaluated capture: the last output goes back, no evaluate
        if (reuse && havePrev && !fresh)
        {
            if (!++cmpTag) ++cmpTag;   // never 0: the buffers start zeroed
            struct { uint32_t tag, pad, w, h; } cc{ cmpTag, 0u, w, h };
            b[0] = tr(shared12.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            b[1] = tr(prevTex.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            b[2] = tr(cmpBuf.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            list->ResourceBarrier(3, b);
            bind(list.Get(), rs.Get(), psoCmp.Get(), 17, 4, &cc);
            list->Dispatch((w + 7) / 8, (h + 7) / 8, 1);
            b[0] = tr(shared12.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
            b[1] = tr(prevTex.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
            b[2] = tr(cmpBuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(3, b);
            list->CopyBufferRegion(cmpRb.Get(), 0, cmpBuf.Get(), 0, 4);
            b[0] = tr(cmpBuf.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
            list->ResourceBarrier(1, b);
            if (FAILED(list->Close())) return drop("command list Close failed");
            ID3D12CommandList* cmp[] = { list.Get() };
            queue->ExecuteCommandLists(1, cmp);
            queue->Signal(done.Get(), ++doneValue);
            done->SetEventOnCompletion(doneValue, doneEvent);
            if (WaitForSingleObject(doneEvent, 5000) != WAIT_OBJECT_0)
                return drop("the identical-frame test did not complete within 5 s");
            uint32_t got = 0;
            void* p = nullptr;
            D3D12_RANGE rr{ 0, 4 };
            if (FAILED(cmpRb->Map(0, &rr, &p))) return drop("the identical-frame readback failed");
            memcpy(&got, p, 4);
            D3D12_RANGE none{ 0, 0 };
            cmpRb->Unmap(0, &none);
            if (FAILED(list->Reset(alloc, nullptr))) return drop("command list Reset failed");
            if (got != cmpTag)
            {
                b[0] = tr(outTex.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
                b[1] = tr(shared12.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
                list->ResourceBarrier(2, b);
                list->CopyResource(shared12.Get(), outTex.Get());
                b[0] = tr(outTex.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                b[1] = tr(shared12.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
                list->ResourceBarrier(2, b);
                if (FAILED(list->Close())) return drop("command list Close failed");
                ID3D12CommandList* back[] = { list.Get() };
                queue->ExecuteCommandLists(1, back);
                queue->Signal(capFence12.Get(), seq);
                allocFence[allocIdx] = ++doneValue;
                queue->Signal(done.Get(), doneValue);
                allocIdx ^= 1;
                lastSeq = seq;
                frames++;
                reused++;
                recMs += (nowQpc100() - t0) / 1e4;
                return true;
            }
        }
        // this capture is the next frame's reference
        if (reuse)
        {
            b[0] = tr(shared12.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
            b[1] = tr(prevTex.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
            list->ResourceBarrier(2, b);
            list->CopyResource(prevTex.Get(), shared12.Get());
            b[0] = tr(shared12.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
            b[1] = tr(prevTex.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
            list->ResourceBarrier(2, b);
            havePrev = true;
        }
        // The motion field (DLSSNR.MVec): a first list writes this frame's luma into its Optical
        // Flow slot; from a stream's second frame on NVOFA (current -> previous) runs between the
        // two lists and the second turns its grid into the field before the evaluate. History is
        // kept only with that field (a Reset on a stream's first frame; a stream starts at the
        // session start and after a pause). Without motion vectors every evaluate is a Reset:
        // kept history had nothing valid to reproject by and a frozen source came back out
        // different frame to frame (DEVELOPMENT.md, "LIVE RUNS THE PASS NON-TEMPORALLY").
        bool flowOk = false;
        const bool dumpNow = motion && !dumpPrefix.empty() && frames >= 30 && frames < 38;
        ID3D12GraphicsCommandList* ev = list.Get();
        if (motion)
        {
            b[0] = tr(shared12.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            list->ResourceBarrier(1, b);
            bind(list.Get(), rs.Get(), psoLuma.Get(), 6, 4, &consts);
            list->Dispatch((w + 7) / 8, (h + 7) / 8, 1);
            b[0] = tr(lumaTex.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
            b[1] = tr(slot[cur].Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
            b[2] = tr(shared12.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
            list->ResourceBarrier(3, b);
            list->CopyResource(slot[cur].Get(), lumaTex.Get());
            b[0] = tr(lumaTex.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            b[1] = tr(slot[cur].Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
            list->ResourceBarrier(2, b);
            if (FAILED(list->Close())) return drop("command list Close failed");
            ID3D12CommandList* first[] = { list.Get() };
            queue->ExecuteCommandLists(1, first);
            if (!fresh)
            {
                queue->Signal(ofIn.Get(), ++ofInValue);
                NV_OF_FENCE_POINT ready{ ofIn.Get(), ofInValue }, flowDone{ ofOut.Get(), ofOutValue + 1 };
                NV_OF_EXECUTE_INPUT_PARAMS_D3D12 ei{};
                ei.inputFrame = ofBuf[cur];          // the forward field of the current frame points
                ei.referenceFrame = ofBuf[cur ^ 1];  // into the previous one: current -> previous
                ei.disableTemporalHints = NV_OF_TRUE;
                ei.numFencePoints = 1;
                ei.fencePoint = &ready;
                NV_OF_EXECUTE_OUTPUT_PARAMS_D3D12 eo{};
                eo.outputBuffer = ofBuf[2];
                eo.fencePoint = &flowDone;
                const NV_OF_STATUS s = g_nrOf.nvOFExecuteD3D12(of, &ei, &eo);
                if (s == NV_OF_SUCCESS)
                {
                    queue->Wait(ofOut.Get(), ++ofOutValue);
                    flowOk = true;
                }
                else
                {
                    // this frame's field goes to zero below and the pass runs as without motion
                    LOG("live DLSS 5 native: nvOFExecute failed (status %d), no motion vectors for the rest of this session, every frame a Reset\n", (int)s);
                    motion = false;
                }
            }
            if (FAILED(list2->Reset(alloc, nullptr))) return drop("command list Reset failed");
            ev = list2.Get();
            // the field: NVOFA's grid validated, or zero (a stream's first frame, the motion just off)
            struct { uint32_t w, h, gw, gh, grid, zero; float margin; uint32_t pad; }
                mc{ w, h, gw, gh, 4u, flowOk ? 0u : 1u, kLiveNrMvMargin, 0u };
            b[0] = tr(flowTex.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            b[1] = tr(slot[0].Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            b[2] = tr(slot[1].Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            b[3] = tr(host.motion(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            ev->ResourceBarrier(4, b);
            bind(ev, rsMv.Get(), psoMv.Get(), cur ? 13 : 9, 8, &mc);
            ev->Dispatch((w + 7) / 8, (h + 7) / 8, 1);
            b[0] = tr(flowTex.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
            b[1] = tr(slot[0].Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
            b[2] = tr(slot[1].Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
            b[3] = tr(host.motion(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            ev->ResourceBarrier(4, b);
            if (dumpNow)
            {
                b[0] = tr(host.motion(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE);
                b[1] = tr(slot[cur].Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
                ev->ResourceBarrier(2, b);
                dumpCopy(ev, host.motion(), mvRb.Get(), DXGI_FORMAT_R16G16_FLOAT, mvRbPitch);
                dumpCopy(ev, slot[cur].Get(), lumaRb.Get(), DXGI_FORMAT_R8_UNORM, lumaRbPitch);
                b[0] = tr(host.motion(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                b[1] = tr(slot[cur].Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
                ev->ResourceBarrier(2, b);
            }
        }
        const bool reset = !flowOk;
        b[0] = tr(shared12.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        b[1] = tr(host.color(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        ev->ResourceBarrier(2, b);
        bind(ev, rs.Get(), psoIn.Get(), 0, 4, &consts);
        ev->Dispatch((w + 7) / 8, (h + 7) / 8, 1);
        b[0] = tr(host.color(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        ev->ResourceBarrier(1, b);
        std::string err;
        if (!host.evaluateOn(ev, reset, err))
        {
            ev->Close();
            return drop(err.c_str());
        }
        b[0] = tr(host.output(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        ev->ResourceBarrier(1, b);
        bind(ev, rs.Get(), psoOut.Get(), 3, 4, &consts);
        ev->Dispatch((w + 7) / 8, (h + 7) / 8, 1);
        b[0] = tr(outTex.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        b[1] = tr(shared12.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
        ev->ResourceBarrier(2, b);
        ev->CopyResource(shared12.Get(), outTex.Get());
        b[0] = tr(outTex.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        b[1] = tr(shared12.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
        b[2] = tr(host.output(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        ev->ResourceBarrier(3, b);
        if (FAILED(ev->Close())) return drop("command list Close failed");
        ID3D12CommandList* lists[] = { ev };
        queue->ExecuteCommandLists(1, lists);
        queue->Signal(capFence12.Get(), seq);
        allocFence[allocIdx] = ++doneValue;
        queue->Signal(done.Get(), doneValue);
        if (dumpNow) dumpWrite(frames);
        allocIdx ^= 1;
        lastSeq = seq;
        cur ^= 1;       // this frame's slot is the next frame's reference
        fresh = false;
        frames++;
        recMs += (nowQpc100() - t0) / 1e4;
        return true;
    }

    void shutdown()
    {
        if (queue && done)
        {
            queue->Signal(done.Get(), ++doneValue);
            if (done->GetCompletedValue() < doneValue)
            {
                done->SetEventOnCompletion(doneValue, doneEvent);
                WaitForSingleObject(doneEvent, 5000);
            }
        }
        motionFree();
        if (frames)
            LOG("live DLSS 5 native: %llu captured frames, %.2f ms CPU per frame to record and submit\n",
                (unsigned long long)frames, recMs / (double)frames);
        if (reused)
            LOG("live DLSS 5 native: %llu captured frames identical to the previous one reused the last output\n",
                (unsigned long long)reused);
        // no NGX release chain: it faults (measured), the process leaves through ExitProcess
        // (run() below) and the OS reclaims the session, exactly like dlssnr.exe
        host.abandon();
        if (doneEvent) { CloseHandle(doneEvent); doneEvent = nullptr; }
        if (hInFence) { CloseHandle(hInFence); hInFence = nullptr; }
        active = false;
    }
};
static LiveNr* g_liveNr = nullptr;
static void liveNrPreCopy(Capture& cap) { if (g_liveNr) g_liveNr->preCopy(cap); }
static bool liveNrSignal(Capture& cap, uint64_t v) { return g_liveNr && g_liveNr->signal(cap, v); }

