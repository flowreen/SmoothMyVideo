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
    uint64_t dropped = 0;    // frames superseded in drainNewest (the SATURATION signal:
                             // captured + dropped per window = the true source cadence)
    double emaArrMs = 0;     // smoothed ARRIVAL interval (ms) over every delivered frame,
                             // superseded ones included = the true source cadence. Pacing
                             // must read THIS, not processed-frame intervals: a loop paced
                             // by its own processing rate feeds back into itself and locks
                             // capture into subharmonic plateaus (measured).
    int64_t lastArrTs = 0;   // SystemRelativeTime of the previously delivered frame
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

        // HDR mode captures FP16 scRGB (linear, 709 primaries, 1.0 = 80 nits); the server
        // re-encodes to BT.2020 PQ. SDR stays 8-bit BGRA.
        pool = wgc::Direct3D11CaptureFramePool::CreateFreeThreaded(
            rtDev,
            g_hdr ? wgdx::DirectXPixelFormat::R16G16B16A16Float : wgdx::DirectXPixelFormat::B8G8R8A8UIntNormalized, 2,
            sz);
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
            InterlockedExchange(&arrived, 1);
            SetEvent(evt);
        });
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
            if (!f)
                break;
            const int64_t ts = f.SystemRelativeTime().count();
            if (lastArrTs && ts > lastArrTs)
            {
                const double d = (ts - lastArrTs) / 10000.0;
                // >300ms = idle/static gap (1 Hz hold refresh), not cadence; skip like the pacer
                if (d < 300.0)
                    emaArrMs = emaArrMs > 0 ? emaArrMs * 0.8 + d * 0.2 : d;
            }
            lastArrTs = ts;
            if (frame)
            {
                frame.Close();
                dropped++;
            } // superseded = the pipeline fell behind
            frame = f;
        }
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
        return 1;
    }

    // interop path: order the shared-texture copy against the server's CUDA reads
    void signalFence(uint64_t v)
    {
        ctx4->Signal(sharedFence.Get(), v);
        ctx11->Flush(); // the immediate context may defer submission; the server is waiting
    }

    // CPU path: drain + staging readback into out (tight cw*4 rows). The frame stays BGRA
    // end-to-end on the server/identity routes (interpolation is channel-agnostic and the
    // present textures are B8G8R8A8, so nothing ever swizzles); only the DLSS-G route sets
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
