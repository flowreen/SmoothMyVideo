// part of smv-live.cpp (unity include, see the file map there); not a standalone translation unit
// ---------------------------------------------------------------- WIC / GDI helpers

static ComPtr<IWICImagingFactory> g_wic;

static bool savePng(const wchar_t* path, const uint8_t* px, uint32_t w, uint32_t h, bool bgra)
{
    ComPtr<IWICStream> stream;
    if (FAILED(g_wic->CreateStream(&stream)))
        return false;
    if (FAILED(stream->InitializeFromFilename(path, GENERIC_WRITE)))
        return false;
    ComPtr<IWICBitmapEncoder> enc;
    if (FAILED(g_wic->CreateEncoder(GUID_ContainerFormatPng, nullptr, &enc)))
        return false;
    if (FAILED(enc->Initialize(stream.Get(), WICBitmapEncoderNoCache)))
        return false;
    ComPtr<IWICBitmapFrameEncode> frame;
    if (FAILED(enc->CreateNewFrame(&frame, nullptr)))
        return false;
    if (FAILED(frame->Initialize(nullptr)))
        return false;
    if (FAILED(frame->SetSize(w, h)))
        return false;
    WICPixelFormatGUID fmt = bgra ? GUID_WICPixelFormat32bppBGRA : GUID_WICPixelFormat32bppRGBA;
    if (FAILED(frame->SetPixelFormat(&fmt)))
        return false;
    if (FAILED(frame->WritePixels(h, w * 4, w * h * 4, (BYTE*)px)))
        return false;
    if (FAILED(frame->Commit()))
        return false;
    return SUCCEEDED(enc->Commit());
}

// GDI screen capture of a rect (DWM-composited) - the verification path: what the USER sees.
static bool captureScreen(int x, int y, uint32_t w, uint32_t h, const wchar_t* path)
{
    HDC sdc = GetDC(nullptr);
    HDC mdc = CreateCompatibleDC(sdc);
    HBITMAP bmp = CreateCompatibleBitmap(sdc, w, h);
    HGDIOBJ old = SelectObject(mdc, bmp);
    BitBlt(mdc, 0, 0, w, h, sdc, x, y, SRCCOPY | CAPTUREBLT);
    SelectObject(mdc, old);

    BITMAPINFO bi{};
    bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bi.bmiHeader.biWidth = (LONG)w;
    bi.bmiHeader.biHeight = -(LONG)h;
    bi.bmiHeader.biPlanes = 1;
    bi.bmiHeader.biBitCount = 32;
    bi.bmiHeader.biCompression = BI_RGB;
    std::vector<uint8_t> px((size_t)w * h * 4);
    int got = GetDIBits(mdc, bmp, 0, h, px.data(), &bi, DIB_RGB_COLORS);
    DeleteObject(bmp);
    DeleteDC(mdc);
    ReleaseDC(nullptr, sdc);
    if (got == 0)
        return false;
    return savePng(path, px.data(), w, h, true);
}

// ---------------------------------------------------------------- D3D helpers

static LRESULT CALLBACK wndProc(HWND h, UINT m, WPARAM w, LPARAM l)
{
    if (m == WM_DESTROY)
    {
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(h, m, w, l);
}

#ifndef WDA_EXCLUDEFROMCAPTURE
#define WDA_EXCLUDEFROMCAPTURE 0x00000011 // Win10 2004+ SDK constant (older headers lack it)
#endif

static void pumpMessages()
{
    MSG msg;
    while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE))
    {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
}

// h rows of w 4-byte pixels from src to dst in two halves, the lower one on a pool thread; swz = swap bytes 0 and 2 of
// every pixel (BGRA <-> RGBA)
static void copyRows(uint8_t* dst, size_t dstPitch, const uint8_t* src, size_t srcPitch, uint32_t w, uint32_t h,
                     bool swz)
{
    auto rows = [&](uint32_t y0, uint32_t y1) {
        for (uint32_t y = y0; y < y1; y++)
        {
            const uint8_t* s = src + (size_t)y * srcPitch;
            if (!swz)
            {
                memcpy(dst + (size_t)y * dstPitch, s, (size_t)w * 4);
                continue;
            }
            uint32_t* d = (uint32_t*)(dst + (size_t)y * dstPitch);
            for (uint32_t x = 0; x < w; x++)
            {
                uint32_t v = ((const uint32_t*)s)[x]; // BGRA in memory = 0xAARRGGBB little-endian
                d[x] = (v & 0xFF00FF00u) | ((v & 0x00FF0000u) >> 16) | ((v & 0x000000FFu) << 16);
            }
        }
    };
    panvec::beside([&] { rows(h / 2, h); }, [&] { rows(0, h / 2); });
}

// "now" in the WGC SystemRelativeTime domain (100ns since boot, QPC-based). Integer split
// instead of double: at ~day-scale uptimes a double loses enough mantissa to skew by ~1ms.
static int64_t qpcFreq()
{
    static LARGE_INTEGER f = [] {
        LARGE_INTEGER x;
        QueryPerformanceFrequency(&x);
        return x;
    }();
    return f.QuadPart;
}

static int64_t nowQpc100()
{
    const int64_t f = qpcFreq();
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return (c.QuadPart / f) * 10000000LL + (c.QuadPart % f) * 10000000LL / f;
}

// row-major perspective (v * M convention, D3D clip space)
static void fillConstants(sl::Constants& c)
{
    const float fovY = 60.0f * 3.14159265f / 180.0f;
    const float aspect = (float)W / (float)H;
    const float zn = 0.1f, zf = 100.0f;
    const float f = 1.0f / tanf(fovY * 0.5f);
    const float a = f / aspect, b = f;
    const float cc = zf / (zf - zn);
    const float d = -zn * zf / (zf - zn);

    sl::float4x4 ident{};
    ident.setRow(0, {1, 0, 0, 0});
    ident.setRow(1, {0, 1, 0, 0});
    ident.setRow(2, {0, 0, 1, 0});
    ident.setRow(3, {0, 0, 0, 1});

    sl::float4x4 proj{};
    proj.setRow(0, {a, 0, 0, 0});
    proj.setRow(1, {0, b, 0, 0});
    proj.setRow(2, {0, 0, cc, 1});
    proj.setRow(3, {0, 0, d, 0});

    sl::float4x4 projInv{};
    projInv.setRow(0, {1.0f / a, 0, 0, 0});
    projInv.setRow(1, {0, 1.0f / b, 0, 0});
    projInv.setRow(2, {0, 0, 0, 1.0f / d});
    projInv.setRow(3, {0, 0, 1, -cc / d});

    c.cameraViewToClip = proj;
    c.clipToCameraView = projInv;
    c.clipToPrevClip = ident; // static camera
    c.prevClipToClip = ident;
    c.jitterOffset = {0, 0};
    c.mvecScale = {1, 1}; // MVs already in [-1,1] (they are all zero)
    c.cameraPinholeOffset = {0, 0};
    c.cameraPos = {0, 0, 0};
    c.cameraUp = {0, 1, 0};
    c.cameraRight = {1, 0, 0};
    c.cameraFwd = {0, 0, 1};
    c.cameraNear = zn;
    c.cameraFar = zf;
    c.cameraFOV = fovY;
    c.cameraAspectRatio = aspect;
    c.depthInverted = sl::Boolean::eFalse;
    c.cameraMotionIncluded = sl::Boolean::eTrue;
    c.motionVectors3D = sl::Boolean::eFalse;
    c.reset = sl::Boolean::eFalse;
    c.orthographicProjection = sl::Boolean::eFalse;
    c.motionVectorsDilated = sl::Boolean::eFalse;
    c.motionVectorsJittered = sl::Boolean::eFalse;
}

// ---------------------------------------------------------------- host (fork of dlssg2f Host)

struct Host
{
    HWND hwnd{};
    ComPtr<IDXGIFactory2> factory;
    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<IDXGISwapChain3> sc;       // SL proxy
    ComPtr<IDXGISwapChain3> scNative; // real swap chain behind the proxy
    UINT nativeBufCount = 0;
    ComPtr<ID3D12Resource> backbuffers[3];
    ComPtr<ID3D12Resource> texFrame;
    ComPtr<ID3D12Resource> texDepth;
    ComPtr<ID3D12Resource> texMvec;
    bool panOn = false;       // the pan as motion vectors (pan_vectors.h), set once its writer is up on the SL route
    panvec::Estimator pan;    // measures each CPU frame against the one presented before
    panvec::Writer panTex;    // writes texMvec
    panvec::Result panNext;   // the next SL present's vectors (zero unless presentFrame measured its frame)
    HANDLE panDone = nullptr; // set by the pool thread presentFrame hands its estimate to
    struct
    {
        const uint8_t* src;
        UINT pitch;
        unsigned csr; // the presenting thread's float mode, so the pool thread's Result is the inline one
    } panJob{};
    ComPtr<ID3D12Resource> uploadBuf;     // the upload buffer presentFrame filled last (the init uploads use it too)
    ComPtr<ID3D12Resource> uploadRing[2]; // the SL route's two: one fills while the GPU still copies from the other
    uint64_t uploadFence[2]{};            // the fence value of the present that last copied from each
    UINT uploadIdx = 0;
    // the GPU handoff (Capture::dgInit): the capture device's two shared slots and its dgWrote opened here; dgCopied =
    // this queue's fence the capture device waits on before rewriting a slot (dgCopiedH = its handle there)
    ComPtr<ID3D12Resource> dgTex[2];
    ComPtr<ID3D12Fence> dgWrote, dgCopied;
    HANDLE dgCopiedH{};
    uint64_t dgCopiedVal = 0;
    int dgPend = -1;          // the slot the next present copies into the back buffer (presentShared), -1 = none
    int dgLast = -1;          // the slot presented last (presentLast presents it again), -1 = a CPU frame was
    uint64_t dgPendWrote = 0; // the dgWrote value that present waits for (still dgLast's when presentLast runs)
    UINT rowPitch = 0;
    // Ring of 3 allocators so the slot-buffer presents can be PIPELINED (recorded while earlier
    // presents still execute); allocFence[i] gates reuse of allocs[i]. alloc stays an alias
    // of allocs[0] for the fully-synchronous init-time uploads.
    ComPtr<ID3D12CommandAllocator> allocs[3];
    uint64_t allocFence[3]{};
    UINT allocIdx = 0;
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    uint64_t fenceValue = 0;
    // Fence value signalled by the LAST streaming present out of each double-buffer
    // half. The cross-group present queue replaces the per-group waitQueue barrier with a
    // non-blocking completed-value check against this (the server may not rewrite a half while
    // presents from it are still executing).
    uint64_t halfFence[2]{};
    HANDLE fenceEvent{};
    sl::ViewportHandle vp{0u};
    sl::Constants consts{};
    uint32_t frameIndex = 1;
    int genFrames = 1;
    UINT syncInterval = 0;
    bool tearPresent = true; // pass DXGI_PRESENT_ALLOW_TEARING on sync-0 presents (server route)
    uint32_t maxGen = 0;
    uint32_t minWH = 0;     // DLSS-G minimum width/height once known
    int posX = 0, posY = 0; // overlay position (target window frame bounds)
    bool clickthrough = true;
    bool park = false; // diagnostic: park the overlay offscreen like dlssg2f
    bool useSL = true; // false = no Streamline anywhere (identity/server backends)
    // Frame/swapchain pixel format. Server/identity routes run BGRA end-to-end (WGC's native
    // order, no CPU swizzle anywhere); the DLSS-G route keeps its verified RGBA pipeline.
    DXGI_FORMAT texFmt = DXGI_FORMAT_R8G8B8A8_UNORM;
    // Direct-present path: a SHARED committed buffer in VRAM the server imports as CUDA external
    // memory and writes output slots into, so a presented frame never leaves the GPU.
    ComPtr<ID3D12Resource> outBuf;
    HANDLE hOutBuf = nullptr; // inheritable NT handle value (rides the cmdline)
    // Per-present cost instrumentation (read + reset by run()'s 2s stats tick): wall ms
    // inside presentTail, split into the allocator-fence backpressure wait and the DXGI
    // Present call itself. Isolates the HDR high-gen capture starve (presents saturating
    // the capture/drain loop) - which component eats the loop decides the fix.
    double prMsTot = 0, prMsWait = 0, prMsFlip = 0;
    uint32_t prN = 0;
    // Vsync-alignment instrumentation (the 360Hz panel cross-check): DWM samples a
    // windowed sync-interval-0 flip chain once per refresh, so two Presents landing in
    // one refresh slot mean only the later one reaches the panel. prBunch counts
    // Presents whose spacing to the previous one is under half the measured refresh
    // period; refreshQpc100 comes from DXGI frame statistics deltas each stats tick
    // (SyncRefreshCount over SyncQPCTime = the rate the panel actually runs, not the
    // mode the OS claims).
    int64_t prLastQpc100 = 0;
    uint32_t prBunch = 0;
    // PRESENT-SPACING HISTOGRAM (the 1%-low statistic): every present's interval to the
    // previous one, bucketed at 0.02ms up to 40ms (anything longer clamps into the last
    // bucket, which only inflates the low percentile - never flatters it). prHist is the
    // 2s stats window, prHistAll the whole run. Frame-time percentiles are the honest way
    // to describe smoothness at four-figure output rates: an average alone hides the
    // group-boundary stalls that a viewer actually sees.
    static const int kHistN = 2000;
    static constexpr double kHistMs = 0.02;
    uint32_t prHist[kHistN]{}, prHistAll[kHistN]{};
    uint64_t prHistNAll = 0;
    void recordSpacing(double ms)
    {
        int b = (int)(ms / kHistMs);
        if (b < 0)
            b = 0;
        if (b >= kHistN)
            b = kHistN - 1;
        prHist[b]++;
        prHistAll[b]++;
        prHistNAll++;
    }
    // avg fps over the histogram, the 1% low (mean frame time of the slowest 1% of
    // presents, expressed as fps) and the 0.1% low. n = sample count in `h`.
    static void histStats(const uint32_t* h, uint64_t n, double& avgFps, double& low1, double& low01)
    {
        avgFps = low1 = low01 = 0.0;
        if (!n)
            return;
        double sum = 0;
        for (int i = 0; i < kHistN; i++)
            sum += (double)h[i] * ((i + 0.5) * kHistMs);
        avgFps = sum > 0 ? 1000.0 * (double)n / sum : 0.0;
        auto tail = [&](double frac) {
            uint64_t want = (uint64_t)((double)n * frac);
            if (want < 1)
                want = 1;
            uint64_t got = 0;
            double s = 0;
            for (int i = kHistN - 1; i >= 0 && got < want; i--)
            {
                uint64_t take = h[i];
                if (got + take > want)
                    take = want - got;
                s += (double)take * ((i + 0.5) * kHistMs);
                got += take;
            }
            return got && s > 0 ? 1000.0 * (double)got / s : 0.0;
        };
        low1 = tail(0.01);
        low01 = tail(0.001);
    }
    int64_t refreshQpc100 = 0;
    DXGI_FRAME_STATISTICS fsPrev{};
    bool fsValid = false;
    // DWM's own vblank counter (DwmGetCompositionTimingInfo): the desktop-wide refresh
    // cadence, independent of this swapchain's stats - separates "panel dropped below
    // its mode (VRR)" from "our swapchain misses refreshes".
    uint64_t dwmPrevRefresh = 0;
    uint64_t dwmPrevQpc = 0;

    bool createOutBuf(uint64_t bytes)
    {
        D3D12_HEAP_PROPERTIES hp{};
        hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC bd{};
        bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bd.Width = bytes;
        bd.Height = 1;
        bd.DepthOrArraySize = 1;
        bd.MipLevels = 1;
        bd.SampleDesc = {1, 0};
        bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        // shared resources must start in COMMON; buffers promote to COPY_SOURCE implicitly
        if (FAILED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED, &bd, D3D12_RESOURCE_STATE_COMMON,
                                                   nullptr, IID_PPV_ARGS(&outBuf))))
            return false;
        SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE}; // inheritable, like the captex handles
        if (FAILED(device->CreateSharedHandle(outBuf.Get(), &sa, GENERIC_ALL, nullptr, &hOutBuf)))
        {
            outBuf.Reset();
            return false;
        }
        return true;
    }

    bool waitQueue()
    {
        queue->Signal(fence.Get(), ++fenceValue);
        if (fence->GetCompletedValue() < fenceValue)
        {
            fence->SetEventOnCompletion(fenceValue, fenceEvent);
            if (WaitForSingleObject(fenceEvent, 5000) != WAIT_OBJECT_0)
                return false;
        }
        return true;
    }

    bool createSwapchain(bool logIt)
    {
        DXGI_SWAP_CHAIN_DESC1 scd{};
        scd.Width = W;
        scd.Height = H;
        scd.Format = texFmt;
        scd.SampleDesc = {1, 0};
        scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        scd.BufferCount = 3;
        scd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        // SL's pacer calls SetMaximumFrameLatency and presents with DXGI_PRESENT_ALLOW_TEARING
        scd.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT | DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;
        ComPtr<IDXGISwapChain1> sc1;
        if (FAILED(factory->CreateSwapChainForHwnd(queue.Get(), hwnd, &scd, nullptr, nullptr, &sc1)))
            return false;
        if (FAILED(sc1.As(&sc)))
            return false;
        for (UINT i = 0; i < 3; i++)
            if (FAILED(sc->GetBuffer(i, IID_PPV_ARGS(&backbuffers[i]))))
                return false;

        if (useSL)
        {
            ComPtr<IUnknown> nativeUnk;
            if (slGetNativeInterface(sc.Get(), (void**)nativeUnk.GetAddressOf()) != sl::Result::eOk)
                return false;
            if (FAILED(nativeUnk.As(&scNative)))
                return false;
        }
        else
        {
            scNative = sc; // no SL proxy in the way: the swap chain IS the native one
        }
        // Declare the color space explicitly rather than letting DXGI infer it. SDR = G22/P709
        // (correct intent for the 8-bit present, removes the HDR-desktop ambiguity); HDR = G2084/P2020
        // (PQ) so DWM composites our R10G10B10A2 buffer as HDR. On the SL route set it on
        // the PROXY swapchain too - the interposer only learns the colorspace from calls it can
        // hook, and DLSS-G with an R10A2 backbuffer but no declared HDR10 colorspace silently
        // passes through (measured: status=0 yet actuallyPresented stuck at 1 until this call).
        const DXGI_COLOR_SPACE_TYPE cs =
            g_hdr ? DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020 : DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
        UINT csSupport = 0;
        if (SUCCEEDED(scNative->CheckColorSpaceSupport(cs, &csSupport)) &&
            (csSupport & DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT))
        {
            scNative->SetColorSpace1(cs);
            if (useSL)
                sc->SetColorSpace1(cs);
        }
        else if (g_hdr)
            LOG("WARNING: PQ color space not accepted by the swapchain; HDR present may look wrong\n");

        DXGI_SWAP_CHAIN_DESC1 nd{};
        scNative->GetDesc1(&nd);
        nativeBufCount = nd.BufferCount;
        if (logIt)
            LOG("native swap chain: %ux%u fmt=%d buffers=%u (colorspace: %s)\n", nd.Width, nd.Height, (int)nd.Format,
                nativeBufCount, g_hdr ? "HDR PQ BT.2020" : "SDR sRGB");
        return true;
    }

    int init()
    {
        // HDR presents R10G10B10A2 (PQ, still 4 bytes/px) on a G2084 swapchain, BOTH
        // routes: DLSS-G mandates exactly RGB10 + BT.2100 PQ for HDR (SL guide 11.0, FP16
        // scRGB explicitly unsupported), so the DLSS-G route rides the same format as the server route.
        // DLSS-G over the echo route presents the server's BGRA slots (DLSS-G takes any UNORM colour format)
        texFmt = g_hdr                  ? DXGI_FORMAT_R10G10B10A2_UNORM
                 : (useSL && !g_fgOver) ? DXGI_FORMAT_R8G8B8A8_UNORM
                                        : DXGI_FORMAT_B8G8R8A8_UNORM;
        WNDCLASSW wc{};
        wc.lpfnWndProc = wndProc;
        wc.hInstance = GetModuleHandleW(nullptr);
        wc.lpszClassName = L"smvlive";
        wc.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));
        RegisterClassW(&wc);
        // click-through: WS_EX_TRANSPARENT + WS_EX_LAYERED at full alpha lets input fall
        // through to the real window underneath while DWM still composits our presents.
        // The overlay NEVER activates (WS_EX_NOACTIVATE, shown with SW_SHOWNOACTIVATE), so a
        // live start or resume never takes the user's focus. Streamline pauses DLSS-G while its
        // window is not the foreground; the focus shim (sl_focus_shim.h, installed after the
        // "DLSS-G ready" line) answers that test with this window, and SMV_DLSSG_FOCUS_SHIM=0
        // leaves the pacer in passthrough (numFramesActuallyPresented=1) whenever another window
        // has focus. TOOLWINDOW keeps the overlay out of alt-tab and the taskbar: the user
        // alt-tabs between real apps, and the pause below handles the rest.
        DWORD ex = WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE;
        if (clickthrough)
            ex |= WS_EX_LAYERED | WS_EX_TRANSPARENT;
        wchar_t exOverride[16]{};
        if (GetEnvironmentVariableW(L"SMV_EXSTYLE", exOverride, 16))
            ex = (DWORD)wcstoul(exOverride, nullptr, 16); // diagnostic bisect knob
        int px = park ? -32000 : posX, py = park ? -32000 : posY;
        hwnd = CreateWindowExW(ex, L"smvlive", L"SMV Live", WS_POPUP, px, py, W, H, nullptr, nullptr, wc.hInstance,
                               nullptr);
        if (!hwnd)
        {
            LOG("CreateWindow failed\n");
            return 1;
        }
        ShowWindow(hwnd, SW_SHOWNOACTIVATE);
        if (clickthrough)
            SetLayeredWindowAttributes(hwnd, 0, 255, LWA_ALPHA);
        // monitor-capture mode: the overlay sits ON the captured monitor, so it must be
        // excluded from its own capture or every frame feeds back into the next (echo trails)
        if (g_monitor && !SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE))
        {
            LOG("SetWindowDisplayAffinity(EXCLUDEFROMCAPTURE) failed (Win10 2004+ required)\n");
            return 1;
        }
        pumpMessages();

        CHECK_HR(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)));
        DXGI_ADAPTER_DESC1 adesc{};
        for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; i++)
        {
            adapter->GetDesc1(&adesc);
            if (adesc.VendorId == 0x10DE)
                break;
            adapter.Reset();
        }
        if (!adapter)
        {
            LOG("no NVIDIA adapter found\n");
            return 1;
        }
        LOG("adapter: %ls\n", adesc.Description);

        CHECK_HR(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)));
        if (useSL)
        {
            CHECK_SL(slSetD3DDevice(device.Get()));

            sl::AdapterInfo ai{};
            ai.deviceLUID = (uint8_t*)&adesc.AdapterLuid;
            ai.deviceLUIDSizeInBytes = sizeof(LUID);
            sl::Result sup = slIsFeatureSupported(sl::kFeatureDLSS_G, ai);
            if (sup != sl::Result::eOk)
            {
                LOG("DLSS-G not supported on this system (sl::Result=%d)\n", (int)sup);
                return 2;
            }

            sl::FeatureVersion ver{};
            if (slGetFeatureVersion(sl::kFeatureDLSS_G, ver) == sl::Result::eOk)
                LOG("DLSS-G ready: SL %u.%u.%u, NGX model %u.%u.%u\n", ver.versionSL.major, ver.versionSL.minor,
                    ver.versionSL.build, ver.versionNGX.major, ver.versionNGX.minor, ver.versionNGX.build);
            installFocusShim(hwnd);
        }

        D3D12_COMMAND_QUEUE_DESC qd{};
        qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        CHECK_HR(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)));
        if (!createSwapchain(true))
            return 1;

        if (useSL)
        {
            sl::ReflexOptions ro{};
            ro.mode = sl::ReflexMode::eLowLatency;
            CHECK_SL(slReflexSetOptions(ro));

            sl::DLSSGOptions go{};
            go.mode = sl::DLSSGMode::eOn;
            go.numFramesToGenerate = (uint32_t)genFrames;
            go.mvecDepthWidth = panvec::vecSize(W);
            go.mvecDepthHeight = panvec::vecSize(H);
            CHECK_SL(slDLSSGSetOptions(vp, go));

            sl::DLSSGState st{};
            if (slDLSSGGetState(vp, st, nullptr) == sl::Result::eOk)
            {
                maxGen = st.numFramesToGenerateMax;
                minWH = st.minWidthOrHeight;
            }
            if (maxGen && (uint32_t)genFrames > maxGen)
            {
                LOG("multi-frame generation beyond %ux is not supported on this GPU (requested %ux)\n", maxGen + 1,
                    genFrames + 1);
                return 3;
            }
        }

        for (int i = 0; i < 3; i++)
            CHECK_HR(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocs[i])));
        alloc = allocs[0];
        CHECK_HR(
            device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr, IID_PPV_ARGS(&list)));
        CHECK_HR(list->Close());
        CHECK_HR(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
        fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);

        rowPitch = (W * 4 + D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1) & ~(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1);

        auto makeTex = [&](DXGI_FORMAT fmt, ComPtr<ID3D12Resource>& tex,
                           D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE, uint32_t tw = W,
                           uint32_t th = H) -> bool {
            D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_DEFAULT};
            D3D12_RESOURCE_DESC rd{};
            rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            rd.Width = tw;
            rd.Height = th;
            rd.DepthOrArraySize = 1;
            rd.MipLevels = 1;
            rd.Format = fmt;
            rd.SampleDesc = {1, 0};
            rd.Flags = flags;
            return SUCCEEDED(device->CreateCommittedResource(
                &hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&tex)));
        };
        auto makeBuf = [&](D3D12_HEAP_TYPE heap, UINT64 size, D3D12_RESOURCE_STATES state,
                           ComPtr<ID3D12Resource>& buf) -> bool {
            D3D12_HEAP_PROPERTIES hp{heap};
            D3D12_RESOURCE_DESC bd{};
            bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            bd.Width = size;
            bd.Height = 1;
            bd.DepthOrArraySize = 1;
            bd.MipLevels = 1;
            bd.SampleDesc = {1, 0};
            bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            return SUCCEEDED(
                device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd, state, nullptr, IID_PPV_ARGS(&buf)));
        };
        // texFrame = the SL route's staging copy (the server route copies the footprint straight into the back buffer)
        const uint32_t vw = panvec::vecSize(W), vh = panvec::vecSize(H);
        if ((useSL &&
             (!makeTex(texFmt, texFrame) ||
              !makeTex(DXGI_FORMAT_R32_FLOAT, texDepth, D3D12_RESOURCE_FLAG_NONE, vw, vh) ||
              !makeTex(DXGI_FORMAT_R16G16_FLOAT, texMvec, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, vw, vh))) ||
            !makeBuf(D3D12_HEAP_TYPE_UPLOAD, (UINT64)rowPitch * H, D3D12_RESOURCE_STATE_GENERIC_READ, uploadRing[0]) ||
            (useSL &&
             !makeBuf(D3D12_HEAP_TYPE_UPLOAD, (UINT64)rowPitch * H, D3D12_RESOURCE_STATE_GENERIC_READ, uploadRing[1])))
        {
            LOG("resource creation failed\n");
            return 1;
        }
        uploadBuf = uploadRing[0];

        // fill depth (flat 0.5) + mvec (zero) once, via the shared upload buffer (SL tag inputs)
        if (useSL)
        {
            std::vector<float> depthData((size_t)W * H, 0.5f);
            uint8_t* dst = nullptr;
            CHECK_HR(uploadBuf->Map(0, nullptr, (void**)&dst));
            for (uint32_t y = 0; y < H; y++)
                memcpy(dst + (size_t)y * rowPitch, depthData.data() + (size_t)y * W, W * 4);
            uploadBuf->Unmap(0, nullptr);

            alloc->Reset();
            list->Reset(alloc.Get(), nullptr);
            D3D12_TEXTURE_COPY_LOCATION dl{texDepth.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            D3D12_TEXTURE_COPY_LOCATION sl_{uploadBuf.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
            sl_.PlacedFootprint.Footprint = {DXGI_FORMAT_R32_FLOAT, vw, vh, 1, rowPitch};
            list->CopyTextureRegion(&dl, 0, 0, 0, &sl_, nullptr);
            list->Close();
            ID3D12CommandList* ls[] = {list.Get()};
            queue->ExecuteCommandLists(1, ls);
            if (!waitQueue())
                return 1;

            std::vector<uint8_t> zero((size_t)W * H * 4, 0);
            CHECK_HR(uploadBuf->Map(0, nullptr, (void**)&dst));
            for (uint32_t y = 0; y < H; y++)
                memcpy(dst + (size_t)y * rowPitch, zero.data() + (size_t)y * W * 4, W * 4);
            uploadBuf->Unmap(0, nullptr);
            alloc->Reset();
            list->Reset(alloc.Get(), nullptr);
            dl.pResource = texMvec.Get();
            sl_.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R16G16_FLOAT;
            list->CopyTextureRegion(&dl, 0, 0, 0, &sl_, nullptr);
            D3D12_RESOURCE_BARRIER bs[2]{};
            for (int i = 0; i < 2; i++)
            {
                bs[i].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                bs[i].Transition.pResource = i ? texDepth.Get() : texMvec.Get();
                bs[i].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                bs[i].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
                bs[i].Transition.StateAfter =
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            }
            list->ResourceBarrier(2, bs);
            list->Close();
            queue->ExecuteCommandLists(1, ls);
            if (!waitQueue())
                return 1;
        }
        if (useSL)
        {
            panOn = panTex.init(device.Get(), texMvec.Get(), W, H);
            if (!panOn)
                LOG("pan vectors: the descriptor heaps did not allocate, zero vectors\n");
        }

        fillConstants(consts);
        return 0;
    }

    // the session's pan vector summary
    void logPan() const
    {
        if (!panOn || !pan.frames)
            return;
        char rb[64] = "";
        if (pan.readMs > 0) // the effects route's slots: the readback and the measure together
            snprintf(rb, sizeof(rb), ", %.2f ms a frame with the slot readback", pan.readMs / (double)pan.frames);
        LOG("pan vectors: %llu of %llu frames moved (mean %.1f px, %.1f %% of their area still), %.2f ms a frame "
            "(max %.2f)%s\n",
            (unsigned long long)pan.tagged, (unsigned long long)pan.frames,
            pan.tagged ? pan.sumLen / (double)pan.tagged : 0.0,
            pan.tagged ? 100.0 * pan.sumStill / (double)pan.tagged : 0.0, pan.sumMs / (double)pan.frames, pan.maxMs,
            rb);
    }

    // CPU path: copy a frame into the upload buffer, then run the present tail.
    // srcPitch = source row stride in bytes (0 = tight W*4; shm slots are 256-aligned); swz = BGRA rows (the mapped
    // capture), swapped to the route's RGBA on the way in.
    bool presentFrame(const uint8_t* rgba, UINT srcPitch = 0, bool swz = false)
    {
        if (!srcPitch)
            srcPitch = W * 4;
        // the estimate (R and B in either order) reads the frame on a pool thread while this one copies it into the
        // upload buffer
        const bool panAsync = panOn && startPan(rgba, srcPitch);
        if (panOn && !panAsync)
            panNext = pan.run(rgba, W, H, srcPitch, texFmt == DXGI_FORMAT_R10G10B10A2_UNORM);
        // the SL route's presents do not wait for the GPU: fill the ring's other buffer once its last copy ran
        bool ringFree = true;
        if (useSL)
        {
            uploadIdx ^= 1;
            uploadBuf = uploadRing[uploadIdx];
            if (fence->GetCompletedValue() < uploadFence[uploadIdx])
            {
                fence->SetEventOnCompletion(uploadFence[uploadIdx], fenceEvent);
                ringFree = WaitForSingleObject(fenceEvent, 5000) == WAIT_OBJECT_0;
            }
        }
        uint8_t* dst = nullptr;
        const bool mapped = ringFree && SUCCEEDED(uploadBuf->Map(0, nullptr, (void**)&dst));
        if (mapped)
        {
            copyRows(dst, rowPitch, rgba, srcPitch, W, H, swz);
            uploadBuf->Unmap(0, nullptr);
        }
        if (panAsync)
            WaitForSingleObject(panDone, INFINITE); // the caller may reuse rgba once this returns
        if (!mapped)
            return false;
        return presentTail(uploadBuf.Get(), 0, rowPitch);
    }

    // the capture device's half of the GPU handoff (its slots' and dgWrote's handles); false = unavailable, closed again
    bool dgOpen(HANDLE t0, HANDLE t1, HANDLE wrote)
    {
        dgClose();
        const HANDLE th[2] = {t0, t1};
        bool ok = true;
        for (int i = 0; i < 2 && ok; i++)
            ok = SUCCEEDED(device->OpenSharedHandle(th[i], IID_PPV_ARGS(&dgTex[i])));
        ok = ok && SUCCEEDED(device->OpenSharedHandle(wrote, IID_PPV_ARGS(&dgWrote))) &&
             SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&dgCopied))) &&
             SUCCEEDED(device->CreateSharedHandle(dgCopied.Get(), nullptr, GENERIC_ALL, nullptr, &dgCopiedH));
        if (!ok)
            dgClose();
        return ok;
    }

    void dgClose()
    {
        for (auto& t : dgTex)
            t.Reset();
        dgWrote.Reset();
        dgCopied.Reset();
        if (dgCopiedH)
            CloseHandle(dgCopiedH);
        dgCopiedH = nullptr;
        dgCopiedVal = 0;
        dgPend = dgLast = -1;
    }

    // the GPU handoff's present: the estimate on the frame's 1/4 thumbnail (rows `pitch` apart), then slot `slot`
    // copied into the back buffer once the capture device's dgWrote reaches `wrote`. Returns the dgCopied value that
    // frees the slot, 0 = the present failed
    uint64_t presentShared(const uint8_t* thumb, UINT pitch, int slot, uint64_t wrote)
    {
        if (panOn)
            panNext = pan.runThumb(thumb, pitch, W, H);
        dgPend = slot;
        dgPendWrote = wrote;
        return presentTail(nullptr, 0, 0) ? dgCopiedVal : 0;
    }

    // queues pan.run on the thread pool; false = no event or no pool thread, the caller measures inline
    bool startPan(const uint8_t* rgba, UINT srcPitch)
    {
        if (!panDone && !(panDone = CreateEventW(nullptr, FALSE, FALSE, nullptr)))
            return false;
        panJob = {rgba, srcPitch, _mm_getcsr()};
        return TrySubmitThreadpoolCallback(
                   [](PTP_CALLBACK_INSTANCE, void* ctx) {
                       Host* h = (Host*)ctx;
                       const unsigned csr = _mm_getcsr();
                       _mm_setcsr(h->panJob.csr);
                       h->panNext =
                           h->pan.run(h->panJob.src, W, H, h->panJob.pitch, h->texFmt == DXGI_FORMAT_R10G10B10A2_UNORM);
                       _mm_setcsr(csr);
                       SetEvent(h->panDone);
                   },
                   this, nullptr) != FALSE;
    }

    bool presentTail(ID3D12Resource* src, UINT64 srcOffset, UINT srcPitch)
    {
        const int64_t t0 = nowQpc100();
        const bool ok = presentTailInner(src, srcOffset, srcPitch);
        prMsTot += (nowQpc100() - t0) / 1e4;
        prN++;
        return ok;
    }

    // src null on the SL route = the GPU handoff's slot dgPend (presentShared), else its slot presented last again
    // (presentLast; the caller moves that slot's dgFree to dgCopiedVal), else texFrame as it is (the last CPU frame)
    bool presentTailInner(ID3D12Resource* src, UINT64 srcOffset, UINT srcPitch)
    {
        pumpMessages();
        const int pend = src ? -1 : dgPend >= 0 ? dgPend : dgLast;
        dgPend = -1;

        // Pipelining: the shared VRAM slot buffer (stable until the next group) and, on the SL route, a CPU frame
        // (its upload buffer = one of uploadRing's two, see the end). Any other source stays fully synchronous.
        const bool pipelined = !useSL && outBuf && src == outBuf.Get();
        const UINT ai = allocIdx;
        allocIdx = (allocIdx + 1) % 3;
        if (allocFence[ai] && fence->GetCompletedValue() < allocFence[ai])
        {
            const int64_t tw = nowQpc100();
            fence->SetEventOnCompletion(allocFence[ai], fenceEvent);
            if (WaitForSingleObject(fenceEvent, 5000) != WAIT_OBJECT_0)
                return false;
            prMsWait += (nowQpc100() - tw) / 1e4;
        }

        sl::FrameToken* tok{};
        if (useSL)
        {
            uint32_t fi = frameIndex++;
            if (slGetNewFrameToken(tok, &fi) != sl::Result::eOk)
                return false;

            slReflexSleep(*tok);
            slPCLSetMarker(sl::PCLMarker::eSimulationStart, *tok);
            if (slSetConstants(consts, *tok, vp) != sl::Result::eOk)
                return false;
            slPCLSetMarker(sl::PCLMarker::eSimulationEnd, *tok);
        }

        UINT bb = sc->GetCurrentBackBufferIndex();

        allocs[ai]->Reset();
        list->Reset(allocs[ai].Get(), nullptr);
        if (panOn)
        {
            // a frame presentFrame measured carries its vectors; a slot of the shared VRAM ring keeps zero
            panTex.record(list.Get(), texMvec.Get(), panNext);
            panNext = panvec::Result{};
        }
        D3D12_TEXTURE_COPY_LOCATION sl_{src, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        sl_.PlacedFootprint.Offset = srcOffset;
        sl_.PlacedFootprint.Footprint = {texFmt, W, H, 1, srcPitch};

        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        if (useSL && pend >= 0)
        {
            // the capture device's slot (resting in COMMON) straight into the back buffer
            D3D12_RESOURCE_BARRIER bs[2]{b, b};
            bs[0].Transition.pResource = dgTex[pend].Get();
            bs[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
            bs[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
            bs[1].Transition.pResource = backbuffers[bb].Get();
            bs[1].Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
            bs[1].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
            list->ResourceBarrier(2, bs);
            list->CopyResource(backbuffers[bb].Get(), dgTex[pend].Get());
            for (D3D12_RESOURCE_BARRIER& x : bs)
                std::swap(x.Transition.StateBefore, x.Transition.StateAfter);
            list->ResourceBarrier(2, bs);
            dgLast = pend;
        }
        else if (useSL)
        {
            // SL route: stage through texFrame (unchanged, verified pipeline)
            D3D12_TEXTURE_COPY_LOCATION dl{texFrame.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            if (src)
            {
                list->CopyTextureRegion(&dl, 0, 0, 0, &sl_, nullptr);
                dgLast = -1;
            }
            b.Transition.pResource = texFrame.Get();
            b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
            b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
            list->ResourceBarrier(1, &b);
            b.Transition.pResource = backbuffers[bb].Get();
            b.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
            b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
            list->ResourceBarrier(1, &b);
            list->CopyResource(backbuffers[bb].Get(), texFrame.Get());
            b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
            b.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
            list->ResourceBarrier(1, &b);
            b.Transition.pResource = texFrame.Get();
            b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
            b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
            list->ResourceBarrier(1, &b);
        }
        else
        {
            // server/identity route: footprint-copy straight into the backbuffer, no
            // texFrame hop (one less full-frame GPU copy per present)
            b.Transition.pResource = backbuffers[bb].Get();
            b.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
            b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
            list->ResourceBarrier(1, &b);
            D3D12_TEXTURE_COPY_LOCATION dl{backbuffers[bb].Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            list->CopyTextureRegion(&dl, 0, 0, 0, &sl_, nullptr);
            b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
            b.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
            list->ResourceBarrier(1, &b);
        }
        list->Close();

        if (useSL)
        {
            sl::Extent vecExtent{0, 0, panvec::vecSize(W), panvec::vecSize(H)};
            sl::Resource depthRes(sl::ResourceType::eTex2d, texDepth.Get(),
                                  (uint32_t)(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE |
                                             D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE));
            depthRes.width = vecExtent.width;
            depthRes.height = vecExtent.height;
            depthRes.nativeFormat = DXGI_FORMAT_R32_FLOAT;
            sl::Resource mvecRes(sl::ResourceType::eTex2d, texMvec.Get(),
                                 (uint32_t)(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE |
                                            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE));
            mvecRes.width = vecExtent.width;
            mvecRes.height = vecExtent.height;
            mvecRes.nativeFormat = DXGI_FORMAT_R16G16_FLOAT;
            // NOTE: do NOT tag kBufferTypeBackbuffer (see dlssg-two-frame: it flips sl.dlss_g
            // into the subrect present path)
            sl::ResourceTag tags[] = {
                sl::ResourceTag(&depthRes, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilPresent, &vecExtent),
                sl::ResourceTag(&mvecRes, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eValidUntilPresent,
                                &vecExtent),
            };
            if (slSetTagForFrame(*tok, vp, tags, _countof(tags), nullptr) != sl::Result::eOk)
                return false;

            slPCLSetMarker(sl::PCLMarker::eRenderSubmitStart, *tok);
            ID3D12CommandList* ls[] = {list.Get()};
            if (pend >= 0)
                queue->Wait(dgWrote.Get(), dgPendWrote); // the capture device wrote the slot
            queue->ExecuteCommandLists(1, ls);
            if (pend >= 0)
                queue->Signal(dgCopied.Get(), ++dgCopiedVal); // the slot may be rewritten
            slPCLSetMarker(sl::PCLMarker::eRenderSubmitEnd, *tok);

            slPCLSetMarker(sl::PCLMarker::ePresentStart, *tok);
            HRESULT hr = sc->Present(syncInterval, 0);
            slPCLSetMarker(sl::PCLMarker::ePresentEnd, *tok);
            if (FAILED(hr))
            {
                LOG("Present failed hr=0x%08lx\n", hr);
                return false;
            }
        }
        else
        {
            ID3D12CommandList* ls[] = {list.Get()};
            queue->ExecuteCommandLists(1, ls);
            const int64_t tp = nowQpc100();
            // ALLOW_TEARING (sync interval 0 only): without it DXGI throttles the flip queue
            // to the panel's refresh, so an above-refresh output target (1000fps on a 360Hz
            // panel) cannot get its Presents out. The swapchain always carries the tearing
            // FLAG, but if the runtime ever rejects the PRESENT flag, drop it permanently
            // rather than failing the session.
            HRESULT hr = sc->Present(syncInterval, (syncInterval == 0 && tearPresent) ? DXGI_PRESENT_ALLOW_TEARING : 0);
            if (hr == DXGI_ERROR_INVALID_CALL && syncInterval == 0 && tearPresent)
            {
                tearPresent = false;
                LOG("present: ALLOW_TEARING rejected, falling back to plain sync-0 presents\n");
                hr = sc->Present(0, 0);
            }
            const int64_t tp2 = nowQpc100();
            prMsFlip += (tp2 - tp) / 1e4;
            if (prLastQpc100 && refreshQpc100 && tp2 - prLastQpc100 < refreshQpc100 / 2)
                prBunch++;
            if (prLastQpc100)
                recordSpacing((double)(tp2 - prLastQpc100) / 1e4);
            prLastQpc100 = tp2;
            if (FAILED(hr))
            {
                LOG("Present failed hr=0x%08lx\n", hr);
                return false;
            }
        }

        queue->Signal(fence.Get(), ++fenceValue);
        allocFence[ai] = fenceValue;
        if (pipelined)
            return true;
        if (useSL && !src)
            return true; // the slot (or texFrame) is the GPU's alone (the capture device waits on dgCopied)
        if (useSL && src == uploadBuf.Get())
        {
            // a CPU frame on the SL route: the next one fills the ring's other buffer, this one is free once the fence
            // passes (the queue keeps the copies in order; the allocator ring bounds the frames in flight)
            uploadFence[uploadIdx] = fenceValue;
            return true;
        }
        if (fence->GetCompletedValue() < fenceValue)
        {
            fence->SetEventOnCompletion(fenceValue, fenceEvent);
            if (WaitForSingleObject(fenceEvent, 5000) != WAIT_OBJECT_0)
                return false;
        }
        return true;
    }

    // The live cascade's stage 2: present a texture another device fills (a stage-1
    // ring slot resting in COMMON, the same format and size as the back buffer) once `wf` reaches `wv` on this queue;
    // presentTailInner's SL sequence with the slot copied straight into the back buffer
    bool presentRing(ID3D12Resource* tex, ID3D12Fence* wf, uint64_t wv)
    {
        const int64_t t0 = nowQpc100();
        pumpMessages();
        const UINT ai = allocIdx;
        allocIdx = (allocIdx + 1) % 3;
        if (allocFence[ai] && fence->GetCompletedValue() < allocFence[ai])
        {
            fence->SetEventOnCompletion(allocFence[ai], fenceEvent);
            if (WaitForSingleObject(fenceEvent, 5000) != WAIT_OBJECT_0)
                return false;
        }
        sl::FrameToken* tok{};
        uint32_t fi = frameIndex++;
        if (slGetNewFrameToken(tok, &fi) != sl::Result::eOk)
            return false;
        slReflexSleep(*tok);
        slPCLSetMarker(sl::PCLMarker::eSimulationStart, *tok);
        if (slSetConstants(consts, *tok, vp) != sl::Result::eOk)
            return false;
        slPCLSetMarker(sl::PCLMarker::eSimulationEnd, *tok);
        const UINT bb = sc->GetCurrentBackBufferIndex();
        allocs[ai]->Reset();
        list->Reset(allocs[ai].Get(), nullptr);
        if (panOn) // stage 1's frames are a fraction of a source frame apart: zero vectors
            panTex.record(list.Get(), texMvec.Get(), panvec::Result{});
        D3D12_RESOURCE_BARRIER b[2]{};
        for (auto& x : b)
        {
            x.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            x.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        }
        b[0].Transition.pResource = tex;
        b[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
        b[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        b[1].Transition.pResource = backbuffers[bb].Get();
        b[1].Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
        b[1].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
        list->ResourceBarrier(2, b);
        list->CopyResource(backbuffers[bb].Get(), tex);
        for (auto& x : b)
            std::swap(x.Transition.StateBefore, x.Transition.StateAfter);
        list->ResourceBarrier(2, b);
        list->Close();
        sl::Extent vecExtent{0, 0, panvec::vecSize(W), panvec::vecSize(H)};
        sl::Resource depthRes(
            sl::ResourceType::eTex2d, texDepth.Get(),
            (uint32_t)(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE));
        depthRes.width = vecExtent.width;
        depthRes.height = vecExtent.height;
        depthRes.nativeFormat = DXGI_FORMAT_R32_FLOAT;
        sl::Resource mvecRes(
            sl::ResourceType::eTex2d, texMvec.Get(),
            (uint32_t)(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE));
        mvecRes.width = vecExtent.width;
        mvecRes.height = vecExtent.height;
        mvecRes.nativeFormat = DXGI_FORMAT_R16G16_FLOAT;
        sl::ResourceTag tags[] = {
            sl::ResourceTag(&depthRes, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilPresent, &vecExtent),
            sl::ResourceTag(&mvecRes, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eValidUntilPresent,
                            &vecExtent),
        };
        if (slSetTagForFrame(*tok, vp, tags, _countof(tags), nullptr) != sl::Result::eOk)
            return false;
        queue->Wait(wf, wv); // the slot's copy on the other device is complete
        slPCLSetMarker(sl::PCLMarker::eRenderSubmitStart, *tok);
        ID3D12CommandList* ls[] = {list.Get()};
        queue->ExecuteCommandLists(1, ls);
        slPCLSetMarker(sl::PCLMarker::eRenderSubmitEnd, *tok);
        slPCLSetMarker(sl::PCLMarker::ePresentStart, *tok);
        const HRESULT hr = sc->Present(syncInterval, 0);
        slPCLSetMarker(sl::PCLMarker::ePresentEnd, *tok);
        if (FAILED(hr))
        {
            LOG("Present failed hr=0x%08lx\n", hr);
            return false;
        }
        queue->Signal(fence.Get(), ++fenceValue);
        allocFence[ai] = fenceValue;
        if (fence->GetCompletedValue() < fenceValue)
        {
            fence->SetEventOnCompletion(fenceValue, fenceEvent);
            if (WaitForSingleObject(fenceEvent, 5000) != WAIT_OBJECT_0)
                return false;
        }
        prMsTot += (nowQpc100() - t0) / 1e4;
        prN++;
        return true;
    }

    // The effects route's cascade feed: `bytes` of the shared slot buffer from `srcOffset` copied into a READBACK
    // buffer on this queue, then `use` reads them while mapped. The fence value it signals orders it like a present
    // (the caller's halfFence).
    ComPtr<ID3D12Resource> readBuf;
    template <class F> bool readSlot(ID3D12Resource* src, UINT64 srcOffset, UINT64 bytes, F&& use)
    {
        if (!readBuf || readBuf->GetDesc().Width < bytes)
        {
            readBuf.Reset();
            D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_READBACK};
            D3D12_RESOURCE_DESC rd{};
            rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            rd.Width = bytes;
            rd.Height = 1;
            rd.DepthOrArraySize = 1;
            rd.MipLevels = 1;
            rd.SampleDesc.Count = 1;
            rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            if (FAILED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST,
                                                       nullptr, IID_PPV_ARGS(&readBuf))))
                return false;
        }
        const UINT ai = allocIdx;
        allocIdx = (allocIdx + 1) % 3;
        if (allocFence[ai] && fence->GetCompletedValue() < allocFence[ai])
        {
            fence->SetEventOnCompletion(allocFence[ai], fenceEvent);
            if (WaitForSingleObject(fenceEvent, 5000) != WAIT_OBJECT_0)
                return false;
        }
        allocs[ai]->Reset();
        list->Reset(allocs[ai].Get(), nullptr);
        list->CopyBufferRegion(readBuf.Get(), 0, src, srcOffset, bytes);
        list->Close();
        ID3D12CommandList* ls[] = {list.Get()};
        queue->ExecuteCommandLists(1, ls);
        queue->Signal(fence.Get(), ++fenceValue);
        allocFence[ai] = fenceValue;
        if (fence->GetCompletedValue() < fenceValue)
        {
            fence->SetEventOnCompletion(fenceValue, fenceEvent);
            if (WaitForSingleObject(fenceEvent, 5000) != WAIT_OBJECT_0)
                return false;
        }
        const D3D12_RANGE rr{0, (SIZE_T)bytes}, none{0, 0};
        void* p = nullptr;
        if (FAILED(readBuf->Map(0, &rr, &p)))
            return false;
        use((const uint8_t*)p);
        readBuf->Unmap(0, &none);
        return true;
    }

    void shutdown()
    {
        if (useSL)
        {
            sl::DLSSGOptions off{};
            off.mode = sl::DLSSGMode::eOff;
            slDLSSGSetOptions(vp, off);
        }
        waitQueue();
        dgClose();
        if (fenceEvent)
            CloseHandle(fenceEvent);
        if (panDone)
        {
            CloseHandle(panDone);
            panDone = nullptr;
        }
        // the output ring's shared NT handle keeps the ring's video memory alive however the
        // resource and the CUDA import are released (the server's stop dropped the import already)
        if (hOutBuf)
        {
            CloseHandle(hOutBuf);
            hOutBuf = nullptr;
        }
        if (hwnd)
            DestroyWindow(hwnd);
        pumpMessages();
    }
};
