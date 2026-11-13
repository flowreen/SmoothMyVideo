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

// h rows of w 4-byte pixels from src to dst in two halves, the lower one on a pool thread
static void copyRows(uint8_t* dst, size_t dstPitch, const uint8_t* src, size_t srcPitch, uint32_t w, uint32_t h)
{
    auto rows = [&](uint32_t y0, uint32_t y1) {
        for (uint32_t y = y0; y < y1; y++)
            memcpy(dst + (size_t)y * dstPitch, src + (size_t)y * srcPitch, (size_t)w * 4);
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

// The part of a session's start that needs no window and no size: the NVIDIA adapter and the D3D12 device.
// Host::init runs it, or adopts what a prewarmed host (--prewarm) ran before its session's arguments arrived.
struct DeviceWarm
{
    ComPtr<IDXGIFactory2> factory;
    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D12Device> device;
    bool ok = false;
};
static DeviceWarm g_devWarm;

// 0 = ok, else Host::init's exit code
static int deviceInit(DeviceWarm& w)
{
    CHECK_HR(CreateDXGIFactory2(0, IID_PPV_ARGS(&w.factory)));
    DXGI_ADAPTER_DESC1 adesc{};
    for (UINT i = 0; w.factory->EnumAdapters1(i, &w.adapter) != DXGI_ERROR_NOT_FOUND; i++)
    {
        w.adapter->GetDesc1(&adesc);
        if (adesc.VendorId == 0x10DE)
            break;
        w.adapter.Reset();
    }
    if (!w.adapter)
    {
        LOG("no NVIDIA adapter found\n");
        return 1;
    }
    LOG("adapter: %ls\n", adesc.Description);

    CHECK_HR(D3D12CreateDevice(w.adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&w.device)));
    w.ok = true;
    return 0;
}

// ---------------------------------------------------------------- host

struct Host
{
    HWND hwnd{};
    ComPtr<IDXGIFactory2> factory;
    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<IDXGISwapChain3> sc;
    UINT nativeBufCount = 0;
    // the swap chain's buffers: 3, or a flip queue (flipQueue, set before init) for the display-timed presents
    // (dtSet): a source frame's whole set of frames and the frames that wait ahead of it (dtSlack). DXGI's most: at
    // one refresh a frame a set of six behind a wait of five fills a queue of 8, and its presents then block
    static const UINT kFlipQueue = 16;
    bool flipQueue = false;
    UINT bufCount = 3;
    ComPtr<ID3D12Resource> backbuffers[kFlipQueue];
    ComPtr<ID3D12Resource> texFrame;  // the final enlarge's source for a frame that comes as a buffer (lzOn)
    ComPtr<ID3D12Resource> uploadBuf; // the upload buffer presentFrame fills
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
    UINT syncInterval = 0;
    bool tearPresent = true;     // pass DXGI_PRESENT_ALLOW_TEARING on sync-0 presents
    uint32_t maxGen = 0;         // DLSS frame generation's most frames a pair on this GPU, once NGX said (0 = not yet)
    uint32_t winW = 0, winH = 0; // the window's size when the frames are smaller (0 = W x H)
    // THE FINAL ENLARGE (lzOn, DLSS 4.5 at the working size: loop.inl sets it with winW x winH): the swap chain at
    // winW x winH, every W x H frame the session presents (an effects slot, a stage's ring slot) enlarged into the
    // back buffer by the native fit's Lanczos: k_fitAaH / V's taps (the native k_rsTab tables, lzTabFn), a horizontal
    // pass into lzTmp (winW x H, fp32), a vertical pass into the back buffer with k_packOutV's / k_packOutHdrV's
    // rounding (lzRecord)
    bool lzOn = false;
    // one axis's tap table (in -> out): the RsTap array (fs x out entries of {float w, int i}); 1 = the native k_rsTab,
    // 2 = its CPU port (the native pipeline not up), 0 = none
    std::function<int(int in, int out, std::vector<uint8_t>& tab)> lzTabFn;
    ComPtr<ID3D12RootSignature> lzRs;
    ComPtr<ID3D12PipelineState> lzPsoH, lzPsoV;
    ComPtr<ID3D12DescriptorHeap> lzSrv, lzRtv; // lzSrv: 8 source views (a ring: <= 3 lists in flight), lzTmp's;
                                               // lzRtv: lzTmp's, the 3 back buffers'
    ComPtr<ID3D12Resource> lzTmp, lzTabH, lzTabV;
    UINT lzSrvInc = 0, lzRtvInc = 0, lzSrvNext = 0, lzFsH = 0, lzFsV = 0;
    bool lzReady = false;
    std::string lzDumpDir; // SMV_DLSSG_LZ_DUMP=DIR: the enlarge SMV_DLSSG_LZ_DUMP_AT=N (default 120): its source,
                           // result and tables written there
    uint64_t lzCount = 0, lzDumpAt = 120;
    ComPtr<ID3D12Resource> lzRbSrc, lzRbOut; // that dump's readback buffers (lzDumpArm .. lzDumpWrite)
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT lzFpSrc{}, lzFpOut{};
    bool lzDumpPend = false;
    // a ring frame's present does not wait for the GPU (the allocator ring bounds the lists in flight at 3, lzSrv's 8
    // views cover them; a 12-slot helper ring is overwritten under a read only when the session lags it by 9+ frames,
    // where nextFrame skips anyway); SMV_DLSSG_LZ_PIPE=0 = the wait after every present
    bool lzPipe = true;
    uint32_t scW() const
    {
        return lzOn ? winW : W;
    }
    uint32_t scH() const
    {
        return lzOn ? winH : H;
    }
    int posX = 0, posY = 0; // overlay position (target window frame bounds)
    bool clickthrough = true;
    bool park = false; // diagnostic: park the overlay offscreen
    // Frame/swapchain pixel format: BGRA end-to-end (WGC's native order, no CPU swizzle anywhere), R10G10B10A2 in HDR
    DXGI_FORMAT texFmt = DXGI_FORMAT_B8G8R8A8_UNORM;
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
    uint32_t fsPrevRepeats = 0; // dtRepeats at fsPrev
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
        scd.Width = scW();
        scd.Height = scH();
        scd.Format = texFmt;
        scd.SampleDesc = {1, 0};
        scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        scd.BufferCount = bufCount = flipQueue ? kFlipQueue : 3;
        scd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        // the flip queue sets its frame latency (SetMaximumFrameLatency), sync-0 presents pass ALLOW_TEARING
        scd.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT | DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;
        ComPtr<IDXGISwapChain1> sc1;
        if (FAILED(factory->CreateSwapChainForHwnd(queue.Get(), hwnd, &scd, nullptr, nullptr, &sc1)))
            return false;
        if (FAILED(sc1.As(&sc)))
            return false;
        // a waitable swap chain starts at a frame latency of 1: the flip queue takes a source frame's whole set
        if (flipQueue && FAILED(sc->SetMaximumFrameLatency(kFlipQueue)))
            return false;
        for (UINT i = 0; i < bufCount; i++)
            if (FAILED(sc->GetBuffer(i, IID_PPV_ARGS(&backbuffers[i]))))
                return false;

        // Declare the color space explicitly rather than letting DXGI infer it. SDR = G22/P709
        // (correct intent for the 8-bit present, removes the HDR-desktop ambiguity); HDR = G2084/P2020
        // (PQ) so DWM composites our R10G10B10A2 buffer as HDR.
        const DXGI_COLOR_SPACE_TYPE cs =
            g_hdr ? DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020 : DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
        UINT csSupport = 0;
        if (SUCCEEDED(sc->CheckColorSpaceSupport(cs, &csSupport)) &&
            (csSupport & DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT))
            sc->SetColorSpace1(cs);
        else if (g_hdr)
            LOG("WARNING: PQ color space not accepted by the swapchain; HDR present may look wrong\n");

        DXGI_SWAP_CHAIN_DESC1 nd{};
        sc->GetDesc1(&nd);
        nativeBufCount = nd.BufferCount;
        if (logIt)
            LOG("native swap chain: %ux%u fmt=%d buffers=%u (colorspace: %s)\n", nd.Width, nd.Height, (int)nd.Format,
                nativeBufCount, g_hdr ? "HDR PQ BT.2020" : "SDR sRGB");
        return true;
    }

    int init()
    {
        // HDR presents R10G10B10A2 (PQ, still 4 bytes/px) on a G2084 swapchain on every route: DLSS frame
        // generation takes exactly RGB10 + BT.2100 PQ for HDR (FP16 scRGB is not supported there)
        texFmt = g_hdr ? DXGI_FORMAT_R10G10B10A2_UNORM : DXGI_FORMAT_B8G8R8A8_UNORM;
        WNDCLASSW wc{};
        wc.lpfnWndProc = wndProc;
        wc.hInstance = GetModuleHandleW(nullptr);
        wc.lpszClassName = L"smvlive";
        wc.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));
        RegisterClassW(&wc);
        // click-through: WS_EX_TRANSPARENT + WS_EX_LAYERED at full alpha lets input fall
        // through to the real window underneath while DWM still composits our presents.
        // The overlay NEVER activates (WS_EX_NOACTIVATE, shown with SW_SHOWNOACTIVATE), so a
        // live start or resume never takes the user's focus. TOOLWINDOW keeps the overlay out of
        // alt-tab and the taskbar: the user alt-tabs between real apps, and the pause below
        // handles the rest.
        DWORD ex = WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE;
        if (clickthrough)
            ex |= WS_EX_LAYERED | WS_EX_TRANSPARENT;
        wchar_t exOverride[16]{};
        if (GetEnvironmentVariableW(L"SMV_EXSTYLE", exOverride, 16))
            ex = (DWORD)wcstoul(exOverride, nullptr, 16); // diagnostic bisect knob
        int px = park ? -32000 : posX, py = park ? -32000 : posY;
        hwnd = CreateWindowExW(ex, L"smvlive", L"SMV Live", WS_POPUP, px, py, winW ? winW : W, winH ? winH : H, nullptr,
                               nullptr, wc.hInstance, nullptr);
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

        // a prewarmed host made the device before this session existed
        if (!g_devWarm.ok)
        {
            g_devWarm = DeviceWarm{};
            if (const int rc = deviceInit(g_devWarm))
                return rc;
        }
        factory = g_devWarm.factory;
        adapter = g_devWarm.adapter;
        device = g_devWarm.device;
        g_devWarm = DeviceWarm{};

        D3D12_COMMAND_QUEUE_DESC qd{};
        qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        CHECK_HR(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)));
        if (!createSwapchain(true))
            return 1;

        for (int i = 0; i < 3; i++)
            CHECK_HR(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocs[i])));
        alloc = allocs[0];
        CHECK_HR(
            device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr, IID_PPV_ARGS(&list)));
        CHECK_HR(list->Close());
        CHECK_HR(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
        fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);

        rowPitch = (W * 4 + D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1) & ~(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1);

        auto makeTex = [&](DXGI_FORMAT fmt, ComPtr<ID3D12Resource>& tex) -> bool {
            D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_DEFAULT};
            D3D12_RESOURCE_DESC rd{};
            rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            rd.Width = W;
            rd.Height = H;
            rd.DepthOrArraySize = 1;
            rd.MipLevels = 1;
            rd.Format = fmt;
            rd.SampleDesc = {1, 0};
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
        // texFrame = the final enlarge's staging copy of a frame that comes as a buffer (every other frame's footprint
        // is copied straight into the back buffer)
        if ((lzOn && !makeTex(texFmt, texFrame)) ||
            !makeBuf(D3D12_HEAP_TYPE_UPLOAD, (UINT64)rowPitch * H, D3D12_RESOURCE_STATE_GENERIC_READ, uploadBuf))
        {
            LOG("resource creation failed\n");
            return 1;
        }
        if (lzOn && !lzInit())
        {
            LOG("DLSS 4.5's final enlarge: its passes did not build\n");
            return 1;
        }
        return 0;
    }

    // ---- the final enlarge (lzOn) ----
    // vs: one triangle over the target; psH / psV: k_fitAaH / aaVSum's sums (a tap = its weight times the source,
    // fused into the running sum in tap order, as NVRTC's default fmad compiles them: so no IEEE strictness, which
    // marks every mad precise = unfused), psV's store = k_packOutV (rintf(v 255)) / k_packOutHdrV (v 1023 + 0.5
    // truncated) on the clamped sum; the root signature rides in the bytecode
    static constexpr const char* kLzShader = R"lz(
#define RS "DescriptorTable(SRV(t0), visibility = SHADER_VISIBILITY_PIXEL), SRV(t1, visibility = SHADER_VISIBILITY_PIXEL), RootConstants(num32BitConstants = 4, b0, visibility = SHADER_VISIBILITY_PIXEL)"
Texture2D<float4> src : register(t0);
StructuredBuffer<uint2> tab : register(t1);
cbuffer C : register(b0) { uint outN; uint fs; uint hdr; uint pad; };
[RootSignature(RS)]
float4 vs(uint id : SV_VertexID) : SV_Position
{
    const float2 p = float2((id << 1) & 2, id & 2);
    return float4(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0, 0.0, 1.0);
}
[RootSignature(RS)]
float4 psH(float4 pos : SV_Position) : SV_Target
{
    const uint ox = (uint)pos.x, y = (uint)pos.y;
    float3 a = float3(0.0, 0.0, 0.0);
    for (uint k = 0; k < fs; k++)
    {
        const uint2 t = tab[k * outN + ox];
        a = mad(asfloat(t.x).xxx, src.Load(int3(t.y, y, 0)).rgb, a);
    }
    return float4(a, 1.0);
}
[RootSignature(RS)]
float4 psV(float4 pos : SV_Position) : SV_Target
{
    const uint x = (uint)pos.x, oy = (uint)pos.y;
    float3 a = float3(0.0, 0.0, 0.0);
    for (uint k = 0; k < fs; k++)
    {
        const uint2 t = tab[k * outN + oy];
        a = mad(asfloat(t.x).xxx, src.Load(int3(x, t.y, 0)).rgb, a);
    }
    a = saturate(a);
    const float3 q = hdr ? floor(mad(a, 1023.0.xxx, 0.5.xxx)) / 1023.0 : round(a * 255.0) / 255.0;
    return float4(q, 1.0);
}
)lz";

    // the passes' shaders, root signature, pipelines, views and lzTmp; the tables wait for the first enlarge (lzTables)
    bool lzInit()
    {
        ComPtr<ID3DBlob> vsb, hb, vb, err;
        const size_t n = strlen(kLzShader);
        auto compile = [&](const char* entry, const char* target, ComPtr<ID3DBlob>& out) {
            err.Reset();
            if (FAILED(D3DCompile(kLzShader, n, nullptr, nullptr, nullptr, entry, target,
                                  D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &out, &err)))
            {
                LOG("DLSS 4.5's final enlarge: %s did not compile: %s\n", entry,
                    err ? (const char*)err->GetBufferPointer() : "?");
                return false;
            }
            return true;
        };
        if (!compile("vs", "vs_5_1", vsb) || !compile("psH", "ps_5_1", hb) || !compile("psV", "ps_5_1", vb) ||
            FAILED(device->CreateRootSignature(0, vsb->GetBufferPointer(), vsb->GetBufferSize(), IID_PPV_ARGS(&lzRs))))
            return false;
        D3D12_GRAPHICS_PIPELINE_STATE_DESC pd{};
        pd.pRootSignature = lzRs.Get();
        pd.VS = {vsb->GetBufferPointer(), vsb->GetBufferSize()};
        pd.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        pd.SampleMask = UINT_MAX;
        pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        pd.RasterizerState.DepthClipEnable = TRUE;
        pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pd.NumRenderTargets = 1;
        pd.SampleDesc = {1, 0};
        pd.PS = {hb->GetBufferPointer(), hb->GetBufferSize()};
        pd.RTVFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
        if (FAILED(device->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&lzPsoH))))
            return false;
        pd.PS = {vb->GetBufferPointer(), vb->GetBufferSize()};
        pd.RTVFormats[0] = texFmt;
        if (FAILED(device->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&lzPsoV))))
            return false;
        D3D12_DESCRIPTOR_HEAP_DESC hd{};
        hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        hd.NumDescriptors = 9;
        hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        if (FAILED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&lzSrv))))
            return false;
        hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        hd.NumDescriptors = 1 + bufCount;
        hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        if (FAILED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&lzRtv))))
            return false;
        lzSrvInc = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        lzRtvInc = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rd.Width = winW;
        rd.Height = H;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        rd.SampleDesc = {1, 0};
        rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        if (FAILED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                                                   D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, nullptr,
                                                   IID_PPV_ARGS(&lzTmp))))
            return false;
        D3D12_CPU_DESCRIPTOR_HANDLE r = lzRtv->GetCPUDescriptorHandleForHeapStart();
        device->CreateRenderTargetView(lzTmp.Get(), nullptr, r);
        for (UINT i = 0; i < bufCount; i++)
        {
            r.ptr += lzRtvInc;
            device->CreateRenderTargetView(backbuffers[i].Get(), nullptr, r);
        }
        D3D12_SHADER_RESOURCE_VIEW_DESC sd{};
        sd.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        sd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        sd.Texture2D.MipLevels = 1;
        D3D12_CPU_DESCRIPTOR_HANDLE s = lzSrv->GetCPUDescriptorHandleForHeapStart();
        s.ptr += (SIZE_T)8 * lzSrvInc;
        device->CreateShaderResourceView(lzTmp.Get(), &sd, s);
        wchar_t dd[MAX_PATH]{};
        if (GetEnvironmentVariableW(L"SMV_DLSSG_LZ_DUMP", dd, MAX_PATH) && dd[0])
            lzDumpDir = wideToUtf8(dd);
        if (GetEnvironmentVariableW(L"SMV_DLSSG_LZ_DUMP_AT", dd, MAX_PATH) && _wtoi64(dd) > 0)
            lzDumpAt = (uint64_t)_wtoi64(dd);
        lzPipe = !(GetEnvironmentVariableW(L"SMV_DLSSG_LZ_PIPE", dd, MAX_PATH) && dd[0] == L'0');
        return true;
    }

    // both axes' tap tables in default-heap buffers, once (the first enlarge: the native k_rsTab needs the native
    // pipeline up); false = no table
    bool lzTables()
    {
        if (lzReady)
            return true;
        std::vector<uint8_t> th, tv;
        const int kh = lzTabFn ? lzTabFn((int)W, (int)winW, th) : 0;
        const int kv = kh ? lzTabFn((int)H, (int)winH, tv) : 0;
        if (!kh || !kv || th.empty() || tv.empty())
        {
            LOG("DLSS 4.5's final enlarge: no tap table\n");
            return false;
        }
        lzFsH = (UINT)(th.size() / 8 / winW);
        lzFsV = (UINT)(tv.size() / 8 / winH);
        D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC bd{};
        bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bd.Height = 1;
        bd.DepthOrArraySize = 1;
        bd.MipLevels = 1;
        bd.SampleDesc = {1, 0};
        bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        bd.Width = th.size();
        bool ok = SUCCEEDED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COMMON,
                                                            nullptr, IID_PPV_ARGS(&lzTabH)));
        bd.Width = tv.size();
        ok = ok && SUCCEEDED(device->CreateCommittedResource(
                       &hp, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&lzTabV)));
        ComPtr<ID3D12Resource> up;
        D3D12_HEAP_PROPERTIES uh{D3D12_HEAP_TYPE_UPLOAD};
        bd.Width = th.size() + tv.size();
        ok = ok && SUCCEEDED(device->CreateCommittedResource(
                       &uh, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&up)));
        uint8_t* p = nullptr;
        ok = ok && SUCCEEDED(up->Map(0, nullptr, (void**)&p));
        if (!ok)
        {
            LOG("DLSS 4.5's final enlarge: the tap tables did not allocate\n");
            return false;
        }
        memcpy(p, th.data(), th.size());
        memcpy(p + th.size(), tv.data(), tv.size());
        up->Unmap(0, nullptr);
        ComPtr<ID3D12CommandAllocator> a;
        ComPtr<ID3D12GraphicsCommandList> l;
        if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&a))) ||
            FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, a.Get(), nullptr, IID_PPV_ARGS(&l))))
            return false;
        l->CopyBufferRegion(lzTabH.Get(), 0, up.Get(), 0, th.size());
        l->CopyBufferRegion(lzTabV.Get(), 0, up.Get(), th.size(), tv.size());
        D3D12_RESOURCE_BARRIER b[2]{};
        for (int i = 0; i < 2; i++)
        {
            b[i].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            b[i].Transition.pResource = i ? lzTabV.Get() : lzTabH.Get();
            b[i].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            b[i].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
            b[i].Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        }
        l->ResourceBarrier(2, b);
        l->Close();
        ID3D12CommandList* ls[] = {l.Get()};
        queue->ExecuteCommandLists(1, ls);
        if (!waitQueue())
            return false;
        LOG("DLSS 4.5's final enlarge: %ux%u -> %ux%u by the native fit's Lanczos, %u / %u taps (%s)\n", W, H, winW,
            winH, lzFsH, lzFsV, kh == 1 && kv == 1 ? "the native k_rsTab tables" : "the tables' CPU port");
        lzReady = true;
        return true;
    }

    // records the enlarge of `srcTex` (W x H, resting in PIXEL_SHADER_RESOURCE) into back buffer bb (PRESENT before
    // and after)
    void lzRecord(ID3D12GraphicsCommandList* l, ID3D12Resource* srcTex, UINT bb)
    {
        const UINT s = lzSrvNext;
        lzSrvNext = (lzSrvNext + 1) % 8;
        D3D12_CPU_DESCRIPTOR_HANDLE c = lzSrv->GetCPUDescriptorHandleForHeapStart();
        c.ptr += (SIZE_T)s * lzSrvInc;
        D3D12_SHADER_RESOURCE_VIEW_DESC sd{};
        sd.Format = srcTex->GetDesc().Format;
        sd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        sd.Texture2D.MipLevels = 1;
        device->CreateShaderResourceView(srcTex, &sd, c);
        D3D12_GPU_DESCRIPTOR_HANDLE g = lzSrv->GetGPUDescriptorHandleForHeapStart(), gt = g;
        g.ptr += (UINT64)s * lzSrvInc;
        gt.ptr += (UINT64)8 * lzSrvInc;
        D3D12_CPU_DESCRIPTOR_HANDLE rt = lzRtv->GetCPUDescriptorHandleForHeapStart(), rb = rt;
        rb.ptr += (SIZE_T)(1 + bb) * lzRtvInc;
        const UINT hdr = texFmt == DXGI_FORMAT_R10G10B10A2_UNORM ? 1u : 0u;
        const bool dump = !lzDumpDir.empty() && ++lzCount == lzDumpAt && lzDumpArm(srcTex);

        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        auto to = [&](ID3D12Resource* r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES z) {
            b.Transition.pResource = r;
            b.Transition.StateBefore = a;
            b.Transition.StateAfter = z;
            l->ResourceBarrier(1, &b);
        };
        if (dump)
        {
            to(srcTex, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE);
            D3D12_TEXTURE_COPY_LOCATION d{lzRbSrc.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT},
                sr{srcTex, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            d.PlacedFootprint = lzFpSrc;
            l->CopyTextureRegion(&d, 0, 0, 0, &sr, nullptr);
            to(srcTex, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        }
        l->SetGraphicsRootSignature(lzRs.Get());
        ID3D12DescriptorHeap* hs[] = {lzSrv.Get()};
        l->SetDescriptorHeaps(1, hs);
        l->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        // horizontal: W x H -> lzTmp (winW x H)
        to(lzTmp.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
        D3D12_VIEWPORT v{0, 0, (float)winW, (float)H, 0, 1};
        D3D12_RECT sc_{0, 0, (LONG)winW, (LONG)H};
        l->RSSetViewports(1, &v);
        l->RSSetScissorRects(1, &sc_);
        l->OMSetRenderTargets(1, &rt, FALSE, nullptr);
        l->SetPipelineState(lzPsoH.Get());
        l->SetGraphicsRootDescriptorTable(0, g);
        l->SetGraphicsRootShaderResourceView(1, lzTabH->GetGPUVirtualAddress());
        const UINT ch[4] = {winW, lzFsH, hdr, 0};
        l->SetGraphicsRoot32BitConstants(2, 4, ch, 0);
        l->DrawInstanced(3, 1, 0, 0);
        to(lzTmp.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        // vertical: lzTmp -> the back buffer (winW x winH)
        to(backbuffers[bb].Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
        v.Height = (float)winH;
        sc_.bottom = (LONG)winH;
        l->RSSetViewports(1, &v);
        l->RSSetScissorRects(1, &sc_);
        l->OMSetRenderTargets(1, &rb, FALSE, nullptr);
        l->SetPipelineState(lzPsoV.Get());
        l->SetGraphicsRootDescriptorTable(0, gt);
        l->SetGraphicsRootShaderResourceView(1, lzTabV->GetGPUVirtualAddress());
        const UINT cv[4] = {winH, lzFsV, hdr, 0};
        l->SetGraphicsRoot32BitConstants(2, 4, cv, 0);
        l->DrawInstanced(3, 1, 0, 0);
        if (dump)
        {
            to(backbuffers[bb].Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
            D3D12_TEXTURE_COPY_LOCATION d{lzRbOut.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT},
                sr{backbuffers[bb].Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            d.PlacedFootprint = lzFpOut;
            l->CopyTextureRegion(&d, 0, 0, 0, &sr, nullptr);
            to(backbuffers[bb].Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PRESENT);
            lzDumpPend = true;
        }
        else
            to(backbuffers[bb].Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
    }

    // SMV_DLSSG_LZ_DUMP: the readback buffers for one enlarge's source and result
    bool lzDumpArm(ID3D12Resource* srcTex)
    {
        auto rbFor = [&](ID3D12Resource* t, D3D12_PLACED_SUBRESOURCE_FOOTPRINT& fp, ComPtr<ID3D12Resource>& rb) {
            const D3D12_RESOURCE_DESC td = t->GetDesc();
            UINT64 total = 0;
            device->GetCopyableFootprints(&td, 0, 1, 0, &fp, nullptr, nullptr, &total);
            D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_READBACK};
            D3D12_RESOURCE_DESC bd{};
            bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            bd.Width = total;
            bd.Height = 1;
            bd.DepthOrArraySize = 1;
            bd.MipLevels = 1;
            bd.SampleDesc = {1, 0};
            bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            return SUCCEEDED(device->CreateCommittedResource(
                &hp, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&rb)));
        };
        return rbFor(srcTex, lzFpSrc, lzRbSrc) && rbFor(backbuffers[0].Get(), lzFpOut, lzRbOut);
    }

    // after the dumped enlarge's present ran: its source, result and both tables as raw files (lz_src.bin W x H x 4,
    // lz_out.bin winW x winH x 4, lz_tabh.bin / lz_tabv.bin the RsTap arrays) + lz_meta.txt
    void lzDumpWrite()
    {
        if (!lzDumpPend)
            return;
        lzDumpPend = false;
        if (!waitQueue())
            return;
        auto open = [&](const char* name, const char* mode) {
            FILE* f = nullptr;
            return fopen_s(&f, (lzDumpDir + "\\" + name).c_str(), mode) == 0 ? f : nullptr;
        };
        auto rows = [&](ID3D12Resource* rb, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& fp, uint32_t w, uint32_t h,
                        const char* name) {
            uint8_t* p = nullptr;
            if (FAILED(rb->Map(0, nullptr, (void**)&p)))
                return;
            if (FILE* f = open(name, "wb"))
            {
                for (uint32_t y = 0; y < h; y++)
                    fwrite(p + fp.Offset + (size_t)y * fp.Footprint.RowPitch, 1, (size_t)w * 4, f);
                fclose(f);
            }
            D3D12_RANGE none{0, 0};
            rb->Unmap(0, &none);
        };
        rows(lzRbSrc.Get(), lzFpSrc, W, H, "lz_src.bin");
        rows(lzRbOut.Get(), lzFpOut, winW, winH, "lz_out.bin");
        std::vector<uint8_t> th, tv;
        const int kh = lzTabFn ? lzTabFn((int)W, (int)winW, th) : 0;
        const int kv = lzTabFn ? lzTabFn((int)H, (int)winH, tv) : 0;
        for (int i = 0; i < 2; i++)
        {
            const std::vector<uint8_t>& t = i ? tv : th;
            if (FILE* f = open(i ? "lz_tabv.bin" : "lz_tabh.bin", "wb"))
            {
                fwrite(t.data(), 1, t.size(), f);
                fclose(f);
            }
        }
        if (FILE* f = open("lz_meta.txt", "w"))
        {
            fprintf(f, "W %u\nH %u\nwinW %u\nwinH %u\nfmt %d\nfsH %u\nfsV %u\ntabs %d %d\n", W, H, winW, winH,
                    (int)texFmt, lzFsH, lzFsV, kh, kv);
            fclose(f);
        }
        LOG("DLSS 4.5's final enlarge: enlarge %llu dumped to %s\n", (unsigned long long)lzCount, lzDumpDir.c_str());
    }

    // CPU path: copy a frame into the upload buffer, then run the present tail.
    // srcPitch = source row stride in bytes (0 = tight W*4; shm slots are 256-aligned)
    bool presentFrame(const uint8_t* px, UINT srcPitch = 0)
    {
        if (!srcPitch)
            srcPitch = W * 4;
        uint8_t* dst = nullptr;
        if (FAILED(uploadBuf->Map(0, nullptr, (void**)&dst)))
            return false;
        copyRows(dst, rowPitch, px, srcPitch, W, H);
        uploadBuf->Unmap(0, nullptr);
        return presentTail(uploadBuf.Get(), 0, rowPitch);
    }

    bool presentTail(ID3D12Resource* src, UINT64 srcOffset, UINT srcPitch)
    {
        const int64_t t0 = nowQpc100();
        const bool ok = presentTailInner(src, srcOffset, srcPitch);
        prMsTot += (nowQpc100() - t0) / 1e4;
        prN++;
        lzDumpWrite();
        return ok;
    }

    bool presentTailInner(ID3D12Resource* src, UINT64 srcOffset, UINT srcPitch)
    {
        pumpMessages();

        // Pipelining: the shared VRAM slot buffer (stable until the next group). Any other source stays fully
        // synchronous.
        const bool pipelined = outBuf && src == outBuf.Get();
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

        UINT bb = sc->GetCurrentBackBufferIndex();
        if (lzOn && !lzTables())
            return false;

        allocs[ai]->Reset();
        list->Reset(allocs[ai].Get(), nullptr);
        D3D12_TEXTURE_COPY_LOCATION sl_{src, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        sl_.PlacedFootprint.Offset = srcOffset;
        sl_.PlacedFootprint.Footprint = {texFmt, W, H, 1, srcPitch};

        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        if (lzOn)
        {
            // the final enlarge: the frame (W x H) through texFrame into the back buffer
            D3D12_TEXTURE_COPY_LOCATION dl{texFrame.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            list->CopyTextureRegion(&dl, 0, 0, 0, &sl_, nullptr);
            b.Transition.pResource = texFrame.Get();
            b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
            b.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            list->ResourceBarrier(1, &b);
            lzRecord(list.Get(), texFrame.Get(), bb);
            std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
            list->ResourceBarrier(1, &b);
        }
        else
        {
            // the footprint copied straight into the back buffer
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

        ID3D12CommandList* ls[] = {list.Get()};
        queue->ExecuteCommandLists(1, ls);
        const int64_t tp = nowQpc100();
        HRESULT hr = presentPlain();
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

        queue->Signal(fence.Get(), ++fenceValue);
        allocFence[ai] = fenceValue;
        if (pipelined)
            return true;
        if (fence->GetCompletedValue() < fenceValue)
        {
            fence->SetEventOnCompletion(fenceValue, fenceEvent);
            if (WaitForSingleObject(fenceEvent, 5000) != WAIT_OBJECT_0)
                return false;
        }
        return true;
    }

    // a present on the swap chain. ALLOW_TEARING (sync interval 0 only): without it DXGI throttles the flip queue to
    // the panel's refresh, so an above-refresh output target (1000fps on a 360Hz panel) cannot get its Presents out.
    // The swapchain always carries the tearing FLAG, but if the runtime ever rejects the PRESENT flag, drop it
    // permanently rather than failing the session.
    HRESULT presentPlain()
    {
        HRESULT hr = sc->Present(syncInterval, (syncInterval == 0 && tearPresent) ? DXGI_PRESENT_ALLOW_TEARING : 0);
        if (hr == DXGI_ERROR_INVALID_CALL && syncInterval == 0 && tearPresent)
        {
            tearPresent = false;
            LOG("present: ALLOW_TEARING rejected, falling back to plain sync-0 presents\n");
            hr = sc->Present(0, 0);
        }
        return hr;
    }

    // DISPLAY-TIMED PRESENTS (a DLSS 4.5 session presenting a stage's ring at a rate the panel can show: one
    // to kDtStay refreshes a frame). The presented stage shares a source frame's frames together and each is queued
    // here with a sync interval, the whole refreshes since the frame before it (a frame that stays more than DXGI's
    // four is queued again for the rest, presentRing): the display gives every frame its time,
    // exact to the refresh and without tearing, and the GPU is woken once a source frame instead of once a presented
    // frame (a present timed on the CPU, sync interval 0, wakes it for every frame and holds its clock up). dtRatio =
    // the refreshes a frame; a rate that is no whole divisor of the refresh gets the two whole intervals around it,
    // mixed through dtAcc (2.4 = 2, 2, 3, 2, 3). The queue must not grow, or every later frame shows late: at a source
    // frame's first frame dtShown (the refresh the frame queued last shows at, counted as the swap chain's statistics
    // count) says how many whole refreshes the new frame would wait behind the queue: its wait. A source's frames do
    // not come evenly (a player's own presents land some refreshes early or late), so the wait moves with every
    // source frame and what it needs is the source's: a frame that comes after the queue ran empty shows at the next
    // refresh (the frame before it stayed longer, the source was late) and the queue is that much ahead from then on,
    // which covers the next frame that late. The wait is given back only when it was not needed: when the least wait
    // of the last two windows (four seconds of source frames) left more than one refresh spare, the spare whole
    // refreshes are taken out of dtAcc, so the next frames that have more than one refresh give them back, one
    // refresh each; only at a rate near the refresh itself, where every frame has one, a frame is left out instead.
    // A wait above dtSlack (a source that came far too early, the queue's room) is cut at once the same way.
    bool dtOn = false;
    double dtRatio = 0, dtAcc = 0;
    double dtPeriod100 = 0;         // a refresh of the display mode, 100 ns
    static const UINT kDtStay = 16; // the most refreshes a frame stays: four presents of it
    double dtSlack = 2;             // the most whole refreshes a source frame's first frame may wait: two, or 24 ms
    uint32_t dtRepeats = 0;         // presents that queued a frame again (no frames: the rates leave them out)
    double dtMinPrev = 1e9, dtMinCur = 1e9; // the least wait of the window before and of this one
    double dtWait = 0;                      // the last first frame's wait, refreshes (the refresh trace's figure)
    double dtShown = 0;                     // valid once dtHave
    bool dtHave = false;
    // a vertical blank of the statistics: its count from the session's first one (dtRef0), its time
    UINT dtRef0 = 0;
    bool dtRefHave = false;
    double dtSyncCount = 0;
    int64_t dtSyncQpc = 0;
    // the two-second window: where it began, the frames since, the first frames of a source frame's set that got a
    // shorter interval, were left out, came late
    double dtWin0 = 0;
    double dtMeasPrev = 0; // the refreshes a frame as the window before this one's frames came (0 = none yet)
    uint64_t dtWinFrames = 0;
    uint32_t dtSets = 0, dtCuts = 0, dtLeft = 0, dtLate = 0;

    // the plan's refreshes a frame (0 = not known) on a display mode of hz, plan = what tells one plan from another
    // (its multiple and keep); true = the ring's frames are presented display-timed
    double dtNomLast = 0, dtHz = 0;
    int dtNomOff = 0, dtPlan = -1;
    bool dtSet(double nominal, double hz, int plan)
    {
        const bool can = flipQueue && hz > 1.0 && nominal >= 0.95 && nominal <= kDtStay + 0.02;
        if (can)
        {
            // a rate within 3 % of a whole divisor of the refresh is that divisor (23.976 for 24, a rate measured
            // over two seconds); the window's own measure refines the rest
            const double whole = floor(nominal + 0.5);
            if (fabs(nominal - whole) < 0.03 * whole)
                nominal = whole;
            if (nominal < 1.0)
                nominal = 1.0;
        }
        if (dtOn && plan == dtPlan && hz == dtHz && nominal > 0)
        {
            // the same plan on the same display mode: the nominal figure moves only with the source's rate as one
            // stats window read it, and a window that got its frames late, then at once, reads 5 % and more off.
            // That is no new rate: the pattern keeps the figure it measures itself (dtNext). Three such windows in
            // a row are another rate
            if (nominal != dtNomLast)
            {
                dtNomLast = nominal;
                dtNomOff = !can || fabs(nominal - dtRatio) > 0.05 * dtRatio ? dtNomOff + 1 : 0;
            }
            if (dtNomOff < 3)
                return true;
        }
        dtNomLast = nominal;
        dtNomOff = 0;
        dtPlan = plan;
        dtHz = hz;
        if (!can)
        {
            if (dtOn)
                LOG("display-timed presents off at %.2f refreshes a frame: each frame presented at its time\n",
                    nominal);
            dtOn = false;
            return false;
        }
        if (!dtOn || fabs(nominal - dtRatio) > 0.05 * dtRatio)
        {
            dtRatio = nominal;
            dtPeriod100 = 1e7 / hz;
            dtSlack = (std::max)(2.0, ceil(24e4 / dtPeriod100));
            dtAcc = 0;
            dtHave = false;
            dtMeasPrev = 0;
            dtMinPrev = dtMinCur = 1e9;
            LOG("display-timed presents: %.3f refreshes a frame at %.0f Hz, a flip queue of %u, a wait of %.0f "
                "refreshes at most\n",
                dtRatio, hz, kFlipQueue, dtSlack);
            dtOn = true;
        }
        return true;
    }

    // the refresh count now, fractional; fresh = read the statistics (once a source frame), else from the last read
    bool dtRefresh(bool fresh, double& cur)
    {
        if (fresh || !dtSyncQpc)
        {
            DXGI_FRAME_STATISTICS fs{};
            if (FAILED(sc->GetFrameStatistics(&fs)) || !fs.SyncQPCTime.QuadPart)
            {
                dtSyncQpc = 0;
                return false;
            }
            if (!dtRefHave)
            {
                dtRef0 = fs.SyncRefreshCount;
                dtRefHave = true;
            }
            dtSyncCount = (double)(UINT)(fs.SyncRefreshCount - dtRef0);
            dtSyncQpc = fs.SyncQPCTime.QuadPart;
        }
        LARGE_INTEGER c;
        QueryPerformanceCounter(&c);
        cur = dtSyncCount + (double)(c.QuadPart - dtSyncQpc) * 1e7 / (double)qpcFreq() / dtPeriod100;
        return true;
    }

    // the next display-timed present's sync interval n; first = the first frame of a source frame's set; false = the
    // frame is left out
    bool dtNext(bool first, UINT& n)
    {
        dtAcc += dtRatio;
        // the stay the pattern gives this frame: the whole refreshes in dtAcc, one at least, kDtStay at most
        auto interval = [&] { return dtAcc < 1.0 ? 1u : dtAcc >= (double)kDtStay ? kDtStay : (UINT)dtAcc; };
        double cur = 0;
        const bool counted = dtRefresh(first, cur);
        const double next = ceil(cur);
        if (!counted || !dtHave)
        {
            // no count of refreshes (yet): the pattern alone
            n = interval();
            dtAcc -= n;
            dtShown = next;
            dtHave = counted;
            dtWin0 = cur;
            dtWinFrames = 1;
            dtSets = dtCuts = dtLeft = dtLate = 0;
            return true;
        }
        if (first)
        {
            // the refreshes a frame as the frames came, over two seconds: the plan's figure is a rate measured over
            // a short window. A figure 5 % off is no measure of this rate (a pause, a new plan)
            const double span = cur - dtWin0;
            double spare = 0;
            if (span * dtPeriod100 >= 2e7 && dtWinFrames > 8)
            {
                const double m = span / (double)dtWinFrames;
                if (fabs(m - dtRatio) < 0.05 * dtRatio)
                {
                    const double whole = floor(m + 0.5);
                    dtRatio = fabs(m - whole) < 0.004 * whole ? whole : 0.5 * (dtRatio + m);
                    if (dtRatio < 1.0)
                        dtRatio = 1.0;
                }
                else if (dtMeasPrev > 0 && fabs(m - dtMeasPrev) < 0.01 * m && m >= 0.95 && m <= kDtStay + 0.02)
                {
                    // two windows in a row came at the same rate, 5 % and more off the plan's figure: that is the
                    // rate. The plan's figure is the source's rate over one stats window, a count of some ten
                    // frames from a source under 10 fps, while this one counts every presented frame
                    const double whole = floor(m + 0.5);
                    dtRatio = fabs(m - whole) < 0.004 * whole ? whole : m;
                    if (dtRatio < 1.0)
                        dtRatio = 1.0;
                    dtMinPrev = dtMinCur = 1e9;
                }
                dtMeasPrev = m;
                // the wait no source frame of the last two windows needed, less one refresh kept; at a rate near the
                // refresh, where giving a refresh back leaves a frame out, four are kept
                const double least = (std::min)(dtMinPrev, dtMinCur), kept = dtRatio < 1.25 ? 4.0 : 1.0;
                if (least < 1e8 && least >= kept + 1.0)
                    spare = floor(least - kept);
                LOG("display-timed: %.3f refreshes a frame (%.3f as they came), %u source frames: %u shortened, %u "
                    "left out, %u late, least wait %.0f\n",
                    dtRatio, m, dtSets, dtCuts, dtLeft, dtLate, dtMinCur < 1e8 ? dtMinCur : 0.0);
                dtMinPrev = dtMinCur - spare;
                dtMinCur = 1e9;
                dtWin0 = cur;
                dtWinFrames = 0;
                dtSets = dtCuts = dtLeft = dtLate = 0;
            }
            dtSets++;
            const double wait = dtShown + (double)interval() - next;
            dtWait = wait;
            if (wait > 1000.0)
            {
                // far more than a queue holds: the count of refreshes started again (another display mode)
                dtShown = next;
                dtAcc = dtRatio;
                dtMinPrev = dtMinCur = 1e9;
            }
            else if (wait >= dtSlack)
            {
                const double cut = floor(wait - dtSlack) + 1.0;
                dtAcc -= cut;
                dtCuts++;
                dtMinCur = (std::min)(dtMinCur, wait - cut);
            }
            else if (wait < -0.5)
            {
                // the frame before this one stayed -wait refreshes longer: that much of what dtAcc still owes is paid
                dtLate++;
                const double owed = dtRatio - dtAcc;
                if (owed > 0)
                    dtAcc += (std::min)(owed, -wait);
                dtMinCur = (std::min)(dtMinCur, 0.0);
            }
            else
            {
                if (spare >= 1.0)
                {
                    dtAcc -= spare;
                    dtCuts++;
                }
                dtMinCur = (std::min)(dtMinCur, wait - spare);
            }
        }
        if (dtAcc < 0.5 && dtRatio < 1.25)
        {
            // every frame has one refresh and the queue is ahead: this frame's refresh is the one given back
            dtLeft++;
            dtWinFrames++;
            return false;
        }
        n = interval();
        dtAcc -= n;
        dtShown = (std::max)(dtShown + n, next);
        dtWinFrames++;
        return true;
    }

    // A plan's presented stage on screen: present a texture another queue fills (a ring slot resting in COMMON, the
    // same format and size as the back buffer, or W x H under the final enlarge) once `wf` reaches `wv` on this
    // queue: the slot copied straight into the back buffer and a plain present, or a display-timed one (first = the
    // first frame of a source frame's set)
    bool presentRing(ID3D12Resource* tex, ID3D12Fence* wf, uint64_t wv, bool first = false)
    {
        const int64_t t0 = nowQpc100();
        pumpMessages();
        UINT dtN = 0;
        const bool left = dtOn && !dtNext(first, dtN);
        if (rtRun && rtPres.size() < rtPres.capacity())
            rtPres.push_back({t0, rtPubQpc, rtCapQpc, first ? dtWait : 0.0, dtAcc, dtN, first, left});
        if (left)
            return true; // left out: the queue is ahead of the display
        // a sync interval ends at four refreshes: a frame that stays longer is queued again for the rest, in parts as
        // equal as whole refreshes allow (6 = 3 + 3), so the display still times all of its stay
        const UINT parts = dtN > 4 ? (dtN + 3) / 4 : 1;
        for (UINT i = 0; i < parts; i++)
            if (!presentRingOnce(tex, wf, wv, parts == 1 ? dtN : dtN / parts + (i < dtN % parts ? 1 : 0)))
                return false;
        dtRepeats += parts - 1;
        prMsTot += (nowQpc100() - t0) / 1e4;
        prN++;
        lzDumpWrite();
        return true;
    }

    // one present of presentRing: dtN = its sync interval on the display-timed route, 0 = a plain one
    bool presentRingOnce(ID3D12Resource* tex, ID3D12Fence* wf, uint64_t wv, UINT dtN)
    {
        const UINT ai = allocIdx;
        allocIdx = (allocIdx + 1) % 3;
        if (allocFence[ai] && fence->GetCompletedValue() < allocFence[ai])
        {
            fence->SetEventOnCompletion(allocFence[ai], fenceEvent);
            if (WaitForSingleObject(fenceEvent, 5000) != WAIT_OBJECT_0)
                return false;
        }
        const UINT bb = sc->GetCurrentBackBufferIndex();
        if (lzOn && !lzTables())
            return false;
        allocs[ai]->Reset();
        list->Reset(allocs[ai].Get(), nullptr);
        D3D12_RESOURCE_BARRIER b[2]{};
        for (auto& x : b)
        {
            x.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            x.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        }
        if (lzOn)
        {
            // the final enlarge: the slot (W x H) into the back buffer
            b[0].Transition.pResource = tex;
            b[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
            b[0].Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            list->ResourceBarrier(1, b);
            lzRecord(list.Get(), tex, bb);
            std::swap(b[0].Transition.StateBefore, b[0].Transition.StateAfter);
            list->ResourceBarrier(1, b);
        }
        else
        {
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
        }
        list->Close();
        queue->Wait(wf, wv); // the slot's copy on the other queue is complete
        ID3D12CommandList* ls[] = {list.Get()};
        queue->ExecuteCommandLists(1, ls);
        const HRESULT hr = dtN ? sc->Present(dtN, 0) : presentPlain();
        if (FAILED(hr))
        {
            LOG("Present failed hr=0x%08lx\n", hr);
            return false;
        }
        queue->Signal(fence.Get(), ++fenceValue);
        allocFence[ai] = fenceValue;
        if (!(lzOn && lzPipe) && fence->GetCompletedValue() < fenceValue)
        {
            fence->SetEventOnCompletion(fenceValue, fenceEvent);
            if (WaitForSingleObject(fenceEvent, 5000) != WAIT_OBJECT_0)
                return false;
        }
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

    // SMV_LIVE_REFRESH_TRACE=<file> (a measurement aid, every Live route): a thread reads the swap chain's frame
    // statistics every millisecond and keeps each reading that differs from the one before it: `<the reading's QPC>
    // <the present on screen> <the refresh it came on> <the refresh count of the sync time> <the sync time> <the last
    // present queued> <the time since the reading before>` (times in 100 ns), written at the session's end. Two rows
    // in a row say how many refreshes a present stayed on screen, a present count that moves by more than one names
    // presents that held no whole refresh, and queued less on screen is the flip queue's fill. The figure is the
    // same whoever times the frames: a sync interval or a present timed on the CPU. Those rows
    // begin with S. A ring frame's present (presentRing) adds a P row: `<the call's QPC> <when its stage shared the
    // frame> <the newest capture's stamp> <a set's first frame: its wait in refreshes x 1000> <dtAcc x 1000> <the
    // sync interval, 0 = timed on the CPU> <first of its set> <left out>`.
    struct RefreshRow
    {
        int64_t qpc, syncQpc;
        UINT shown, refresh, syncRefresh, queued;
        uint32_t gap;
    };
    struct PresentRow
    {
        int64_t qpc, pub, cap;
        double wait, acc;
        UINT n;
        bool first, left;
    };
    std::vector<RefreshRow> rtRows;
    std::vector<PresentRow> rtPres;
    int64_t rtPubQpc = 0, rtCapQpc = 0; // the loop's stamps of the ring frame presentRing takes next
    std::thread rtThread;
    std::atomic<bool> rtRun{false};
    wchar_t rtPath[MAX_PATH] = L"";

    void rtStart()
    {
        if (!GetEnvironmentVariableW(L"SMV_LIVE_REFRESH_TRACE", rtPath, MAX_PATH) || !sc)
            return;
        rtRows.reserve(1 << 18);
        rtPres.reserve(1 << 18);
        rtRun = true;
        rtThread = std::thread([this] {
            HANDLE t =
                CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
            LARGE_INTEGER due;
            due.QuadPart = -10000;
            DXGI_FRAME_STATISTICS last{};
            int64_t prev = nowQpc100();
            while (rtRun)
            {
                if (t && SetWaitableTimer(t, &due, 0, nullptr, nullptr, FALSE))
                    WaitForSingleObject(t, 100);
                else
                    Sleep(1);
                const int64_t now = nowQpc100();
                DXGI_FRAME_STATISTICS fs{};
                if (SUCCEEDED(sc->GetFrameStatistics(&fs)) && fs.SyncQPCTime.QuadPart &&
                    (fs.PresentCount != last.PresentCount || fs.PresentRefreshCount != last.PresentRefreshCount ||
                     fs.SyncRefreshCount != last.SyncRefreshCount) &&
                    rtRows.size() < rtRows.capacity())
                {
                    UINT queued = 0;
                    sc->GetLastPresentCount(&queued);
                    rtRows.push_back({now, (int64_t)((double)fs.SyncQPCTime.QuadPart * 1e7 / (double)qpcFreq()),
                                      fs.PresentCount, fs.PresentRefreshCount, fs.SyncRefreshCount, queued,
                                      (uint32_t)(now - prev)});
                    last = fs;
                }
                prev = now;
            }
            if (t)
                CloseHandle(t);
        });
    }

    void rtStop()
    {
        if (!rtThread.joinable())
            return;
        rtRun = false;
        rtThread.join();
        FILE* f = nullptr;
        if (_wfopen_s(&f, rtPath, L"w") == 0 && f)
        {
            for (const RefreshRow& r : rtRows)
                fprintf(f, "S %lld %u %u %u %lld %u %u\n", (long long)r.qpc, r.shown, r.refresh, r.syncRefresh,
                        (long long)r.syncQpc, r.queued, r.gap);
            for (const PresentRow& r : rtPres)
                fprintf(f, "P %lld %lld %lld %.0f %.0f %u %d %d\n", (long long)r.qpc, (long long)r.pub,
                        (long long)r.cap, r.wait * 1000.0, r.acc * 1000.0, r.n, r.first ? 1 : 0, r.left ? 1 : 0);
            fclose(f);
        }
        LOG("refresh trace: %zu statistics rows, %zu ring presents\n", rtRows.size(), rtPres.size());
    }

    // a session that ends before its loop's end leaves no trace thread reading the swap chain
    ~Host()
    {
        rtRun = false;
        if (rtThread.joinable())
            rtThread.join();
    }

    void shutdown()
    {
        waitQueue();
        if (fenceEvent)
            CloseHandle(fenceEvent);
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
