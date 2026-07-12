// dlssg2f - bare-bones "game engine" host that feeds video frames through
// DLSS Frame Generation (Streamline sl.dlss_g) and returns the generated frames.
//
// The app is the minimum DLSS-G contract: a real D3D12 device + flip-model swap
// chain created through sl.interposer, per-frame common constants, depth + motion
// vector tags (flat depth, zero MVs - video frames have neither), Reflex + PCL
// markers, and DLSSGMode::eOn with eShowOnlyInterpolatedFrame so only the AI
// generated in-between frame reaches the (native) swap chain, where we read it
// back from the last-presented buffer.
//
// Modes:
//   dlssg2f.exe frameA.png frameB.png out.png [--gdi]
//       single pair -> interpolated PNG (native readback; --gdi = legacy screen
//       capture used to validate the readback path, needs the window on-screen)
//   dlssg2f.exe --server W H [--gen N] [--wait ms] [--onscreen]
//       streaming: raw RGBA8 frames (W*H*4) on stdin; for every frame after the
//       first, the N (default 1, max 5) DLSS-G frames generated between it and
//       its predecessor - evenly spaced, in temporal order - are written to
//       stdout as raw RGBA8 (multi-frame generation: N=1 is 2x, N=5 is 6x).
//       All logging goes to stderr; stdout is binary-only. EOF on stdin exits.
//       This is the SmoothMyVideo backend.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <dxgidebug.h>
#include <wincodec.h>
#include <wrl/client.h>
#include <io.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <string>
#include <vector>

#include <sl.h>
#include <sl_consts.h>
#include <sl_dlss_g.h>
#include <sl_reflex.h>
#include <sl_pcl.h>

using Microsoft::WRL::ComPtr;

static uint32_t W = 1920;
static uint32_t H = 1080;

#define LOG(...) do { fprintf(stderr, __VA_ARGS__); fflush(stderr); } while (0)
#define CHECK_HR(x) do { HRESULT _hr = (x); if (FAILED(_hr)) { LOG("dlssg2f FAIL hr=0x%08lx at %s:%d\n", _hr, __FILE__, __LINE__); return 1; } } while (0)
#define CHECK_SL(x) do { sl::Result _r = (x); if (_r != sl::Result::eOk) { LOG("dlssg2f FAIL sl::Result=%d at %s:%d\n", (int)_r, __FILE__, __LINE__); return 1; } } while (0)

static bool g_verbose = false;

static void slLog(sl::LogType type, const char* msg)
{
    if (g_verbose) { LOG("[SL%d] %s", (int)type, msg); return; }
    if (type == sl::LogType::eInfo) return;
    // Known-benign SDK chatter this host provokes by design; keep it out of the user-visible
    // engine log. Real warnings and all errors still pass through.
    static const char* benign[] = {
        "Invalid backbuffer resource extent",      // belt-and-braces: we do tag the full extent
        "reseting frame timer",                    // game pacer notices our slow present cadence
        "Frame rate over",
        "duplicated unique id",                    // exe dir scanned twice (pathsToPlugins + cwd)
        "invoked before slInit",                   // process-startup probe order, harmless here
        "must be synchronized with the present thread", // we are single-threaded by construction
    };
    for (const char* b : benign)
        if (strstr(msg, b)) return;
    LOG("[SL%d] %s", (int)type, msg);
}

// ---------------------------------------------------------------- WIC helpers

static ComPtr<IWICImagingFactory> g_wic;

static bool loadPngRGBA(const wchar_t* path, std::vector<uint8_t>& out)
{
    ComPtr<IWICBitmapDecoder> dec;
    if (FAILED(g_wic->CreateDecoderFromFilename(path, nullptr, GENERIC_READ, WICDecodeMetadataCacheOnDemand, &dec))) return false;
    ComPtr<IWICBitmapFrameDecode> frame;
    if (FAILED(dec->GetFrame(0, &frame))) return false;
    UINT fw = 0, fh = 0;
    frame->GetSize(&fw, &fh);
    if (fw != W || fh != H) { LOG("frame is %ux%u, expected %ux%u\n", fw, fh, W, H); return false; }
    ComPtr<IWICBitmapSource> conv;
    if (FAILED(WICConvertBitmapSource(GUID_WICPixelFormat32bppRGBA, frame.Get(), &conv))) return false;
    out.resize((size_t)W * H * 4);
    return SUCCEEDED(conv->CopyPixels(nullptr, W * 4, (UINT)out.size(), out.data()));
}

static bool savePng(const wchar_t* path, const uint8_t* px, uint32_t w, uint32_t h, bool bgra)
{
    ComPtr<IWICStream> stream;
    if (FAILED(g_wic->CreateStream(&stream))) return false;
    if (FAILED(stream->InitializeFromFilename(path, GENERIC_WRITE))) return false;
    ComPtr<IWICBitmapEncoder> enc;
    if (FAILED(g_wic->CreateEncoder(GUID_ContainerFormatPng, nullptr, &enc))) return false;
    if (FAILED(enc->Initialize(stream.Get(), WICBitmapEncoderNoCache))) return false;
    ComPtr<IWICBitmapFrameEncode> frame;
    if (FAILED(enc->CreateNewFrame(&frame, nullptr))) return false;
    if (FAILED(frame->Initialize(nullptr))) return false;
    if (FAILED(frame->SetSize(w, h))) return false;
    WICPixelFormatGUID fmt = bgra ? GUID_WICPixelFormat32bppBGRA : GUID_WICPixelFormat32bppRGBA;
    if (FAILED(frame->SetPixelFormat(&fmt))) return false;
    if (FAILED(frame->WritePixels(h, w * 4, w * h * 4, (BYTE*)px))) return false;
    if (FAILED(frame->Commit())) return false;
    return SUCCEEDED(enc->Commit());
}

// GDI screen capture of a rect (DWM-composited). Validation-only path.
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
    if (got == 0) return false;
    return savePng(path, px.data(), w, h, true);
}

// ---------------------------------------------------------------- D3D helpers

static LRESULT CALLBACK wndProc(HWND h, UINT m, WPARAM w, LPARAM l)
{
    if (m == WM_DESTROY) { PostQuitMessage(0); return 0; }
    return DefWindowProcW(h, m, w, l);
}

static void pumpMessages()
{
    MSG msg;
    while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE))
    {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
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
    ident.setRow(0, { 1, 0, 0, 0 });
    ident.setRow(1, { 0, 1, 0, 0 });
    ident.setRow(2, { 0, 0, 1, 0 });
    ident.setRow(3, { 0, 0, 0, 1 });

    sl::float4x4 proj{};
    proj.setRow(0, { a, 0, 0, 0 });
    proj.setRow(1, { 0, b, 0, 0 });
    proj.setRow(2, { 0, 0, cc, 1 });
    proj.setRow(3, { 0, 0, d, 0 });

    sl::float4x4 projInv{};
    projInv.setRow(0, { 1.0f / a, 0, 0, 0 });
    projInv.setRow(1, { 0, 1.0f / b, 0, 0 });
    projInv.setRow(2, { 0, 0, 0, 1.0f / d });
    projInv.setRow(3, { 0, 0, 1, -cc / d });

    c.cameraViewToClip = proj;
    c.clipToCameraView = projInv;
    c.clipToPrevClip = ident;   // static camera
    c.prevClipToClip = ident;
    c.jitterOffset = { 0, 0 };
    c.mvecScale = { 1, 1 };     // MVs already in [-1,1] (they are all zero)
    c.cameraPinholeOffset = { 0, 0 };
    c.cameraPos = { 0, 0, 0 };
    c.cameraUp = { 0, 1, 0 };
    c.cameraRight = { 1, 0, 0 };
    c.cameraFwd = { 0, 0, 1 };
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

// ---------------------------------------------------------------- host

struct Host
{
    HWND hwnd{};
    ComPtr<IDXGIFactory2> factory;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<IDXGISwapChain3> sc;        // SL proxy
    ComPtr<IDXGISwapChain3> scNative;  // real swap chain behind the proxy
    UINT nativeBufCount = 0;
    ComPtr<ID3D12Resource> backbuffers[3];
    ComPtr<ID3D12Resource> texFrame;   // upload target, copied to backbuffer each present
    ComPtr<ID3D12Resource> texDepth;
    ComPtr<ID3D12Resource> texMvec;
    ComPtr<ID3D12Resource> uploadBuf;
    ComPtr<ID3D12Resource> readbackBuf;
    UINT rowPitch = 0;                 // 256-aligned pitch shared by upload + readback
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    uint64_t fenceValue = 0;
    HANDLE fenceEvent{};
    sl::ViewportHandle vp{ 0u };
    sl::Constants consts{};
    uint32_t frameIndex = 1;
    int waitMs = 5;                    // settle after a native present is observed
    int genFrames = 1;                 // generated frames per source frame (numFramesToGenerate)
    UINT syncInterval = 0;             // Present sync interval (1 = vsync-paced FG)
    uint32_t maxGen = 0;               // device limit reported by the SDK (0 = not yet known)
    UINT presentSnapshot = 0;          // native present count taken just before each proxy Present

    bool waitQueue()
    {
        queue->Signal(fence.Get(), ++fenceValue);
        if (fence->GetCompletedValue() < fenceValue)
        {
            fence->SetEventOnCompletion(fenceValue, fenceEvent);
            if (WaitForSingleObject(fenceEvent, 5000) != WAIT_OBJECT_0) return false;
        }
        return true;
    }

    // create (or re-create) the swap chain through SL's proxy factory and resolve the native
    // chain behind it - the buffers DLSS-G presents into and we read generated frames from
    bool createSwapchain(bool logIt)
    {
        DXGI_SWAP_CHAIN_DESC1 scd{};
        scd.Width = W;
        scd.Height = H;
        scd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        scd.SampleDesc = { 1, 0 };
        scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        scd.BufferCount = 3;
        scd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        // SL's pacer calls SetMaximumFrameLatency and presents with DXGI_PRESENT_ALLOW_TEARING
        scd.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT | DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;
        ComPtr<IDXGISwapChain1> sc1;
        if (FAILED(factory->CreateSwapChainForHwnd(queue.Get(), hwnd, &scd, nullptr, nullptr, &sc1))) return false;
        if (FAILED(sc1.As(&sc))) return false;
        for (UINT i = 0; i < 3; i++)
            if (FAILED(sc->GetBuffer(i, IID_PPV_ARGS(&backbuffers[i])))) return false;

        ComPtr<IUnknown> nativeUnk;
        if (slGetNativeInterface(sc.Get(), (void**)nativeUnk.GetAddressOf()) != sl::Result::eOk) return false;
        if (FAILED(nativeUnk.As(&scNative))) return false;
        DXGI_SWAP_CHAIN_DESC1 nd{};
        scNative->GetDesc1(&nd);
        nativeBufCount = nd.BufferCount;
        if (logIt)
            LOG("native swap chain: %ux%u fmt=%d buffers=%u\n", nd.Width, nd.Height, (int)nd.Format, nativeBufCount);
        return true;
    }


    int init(bool onscreen)
    {
        // window: swap chain needs an HWND, but with native readback it never
        // has to be visible on the desktop, so park it far offscreen by default
        WNDCLASSW wc{};
        wc.lpfnWndProc = wndProc;
        wc.hInstance = GetModuleHandleW(nullptr);
        wc.lpszClassName = L"dlssg2f";
        wc.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));
        RegisterClassW(&wc);
        int px = onscreen ? 0 : -32000, py = onscreen ? 0 : -32000;
        hwnd = CreateWindowExW(onscreen ? WS_EX_TOPMOST : WS_EX_TOOLWINDOW, L"dlssg2f", L"dlssg2f",
                               WS_POPUP | WS_VISIBLE, px, py, W, H, nullptr, nullptr, wc.hInstance, nullptr);
        if (!hwnd) { LOG("CreateWindow failed\n"); return 1; }
        pumpMessages();

        CHECK_HR(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)));
        ComPtr<IDXGIAdapter1> adapter;
        DXGI_ADAPTER_DESC1 adesc{};
        for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; i++)
        {
            adapter->GetDesc1(&adesc);
            if (adesc.VendorId == 0x10DE) break;
            adapter.Reset();
        }
        if (!adapter) { LOG("no NVIDIA adapter found\n"); return 1; }
        LOG("adapter: %ls\n", adesc.Description);

        CHECK_HR(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)));
        CHECK_SL(slSetD3DDevice(device.Get()));

        sl::AdapterInfo ai{};
        ai.deviceLUID = (uint8_t*)&adesc.AdapterLuid;
        ai.deviceLUIDSizeInBytes = sizeof(LUID);
        sl::Result sup = slIsFeatureSupported(sl::kFeatureDLSS_G, ai);
        if (sup != sl::Result::eOk) { LOG("DLSS-G not supported on this system (sl::Result=%d)\n", (int)sup); return 2; }

        sl::FeatureVersion ver{};
        if (slGetFeatureVersion(sl::kFeatureDLSS_G, ver) == sl::Result::eOk)
            LOG("DLSS-G ready: SL %u.%u.%u, NGX model %u.%u.%u\n",
                ver.versionSL.major, ver.versionSL.minor, ver.versionSL.build,
                ver.versionNGX.major, ver.versionNGX.minor, ver.versionNGX.build);

        D3D12_COMMAND_QUEUE_DESC qd{};
        qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        CHECK_HR(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)));
        if (!createSwapchain(true)) return 1;

        sl::ReflexOptions ro{};
        ro.mode = sl::ReflexMode::eLowLatency;
        CHECK_SL(slReflexSetOptions(ro));

        // NOTE: no eShowOnlyInterpolatedFrame - that debug flag was the v1 capture mechanism, but
        // the production present path ([generated frames..., real frame] per app present) is what
        // games actually exercise, and the caller selects generated frames by CONTENT anyway
        // (real-frame presents are byte-copies of the inputs and get filtered out).
        sl::DLSSGOptions go{};
        go.mode = sl::DLSSGMode::eOn;
        go.numFramesToGenerate = (uint32_t)genFrames;
        CHECK_SL(slDLSSGSetOptions(vp, go));

        // multi-frame support is a device/driver capability; refuse early when the requested
        // multiplier exceeds it (0 = not reported yet; re-checked after warmup in that case)
        sl::DLSSGState st{};
        if (slDLSSGGetState(vp, st, nullptr) == sl::Result::eOk)
            maxGen = st.numFramesToGenerateMax;
        if (maxGen && (uint32_t)genFrames > maxGen)
        {
            LOG("multi-frame generation beyond %ux is not supported on this GPU (requested %ux)\n",
                maxGen + 1, genFrames + 1);
            return 3;
        }

        // static resources
        CHECK_HR(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc)));
        CHECK_HR(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr, IID_PPV_ARGS(&list)));
        CHECK_HR(list->Close());
        CHECK_HR(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
        fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);

        rowPitch = (W * 4 + D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1) & ~(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1);

        auto makeTex = [&](DXGI_FORMAT fmt, ComPtr<ID3D12Resource>& tex) -> bool
        {
            D3D12_HEAP_PROPERTIES hp{ D3D12_HEAP_TYPE_DEFAULT };
            D3D12_RESOURCE_DESC rd{};
            rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            rd.Width = W; rd.Height = H;
            rd.DepthOrArraySize = 1; rd.MipLevels = 1;
            rd.Format = fmt;
            rd.SampleDesc = { 1, 0 };
            return SUCCEEDED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&tex)));
        };
        auto makeBuf = [&](D3D12_HEAP_TYPE heap, UINT64 size, D3D12_RESOURCE_STATES state, ComPtr<ID3D12Resource>& buf) -> bool
        {
            D3D12_HEAP_PROPERTIES hp{ heap };
            D3D12_RESOURCE_DESC bd{};
            bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            bd.Width = size; bd.Height = 1; bd.DepthOrArraySize = 1; bd.MipLevels = 1;
            bd.SampleDesc = { 1, 0 };
            bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            return SUCCEEDED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd, state, nullptr, IID_PPV_ARGS(&buf)));
        };
        if (!makeTex(DXGI_FORMAT_R8G8B8A8_UNORM, texFrame) ||
            !makeTex(DXGI_FORMAT_R32_FLOAT, texDepth) ||
            !makeTex(DXGI_FORMAT_R16G16_FLOAT, texMvec) ||
            !makeBuf(D3D12_HEAP_TYPE_UPLOAD, (UINT64)rowPitch * H, D3D12_RESOURCE_STATE_GENERIC_READ, uploadBuf) ||
            !makeBuf(D3D12_HEAP_TYPE_READBACK, (UINT64)rowPitch * H, D3D12_RESOURCE_STATE_COPY_DEST, readbackBuf))
        {
            LOG("resource creation failed\n");
            return 1;
        }

        // fill depth (flat 0.5) + mvec (zero) once, via the shared upload buffer
        {
            std::vector<float> depthData((size_t)W * H, 0.5f);
            uint8_t* dst = nullptr;
            CHECK_HR(uploadBuf->Map(0, nullptr, (void**)&dst));
            for (uint32_t y = 0; y < H; y++)
                memcpy(dst + (size_t)y * rowPitch, depthData.data() + (size_t)y * W, W * 4);
            uploadBuf->Unmap(0, nullptr);

            alloc->Reset();
            list->Reset(alloc.Get(), nullptr);
            D3D12_TEXTURE_COPY_LOCATION dl{ texDepth.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX };
            D3D12_TEXTURE_COPY_LOCATION sl_{ uploadBuf.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT };
            sl_.PlacedFootprint.Footprint = { DXGI_FORMAT_R32_FLOAT, W, H, 1, rowPitch };
            list->CopyTextureRegion(&dl, 0, 0, 0, &sl_, nullptr);
            list->Close();
            ID3D12CommandList* ls[] = { list.Get() };
            queue->ExecuteCommandLists(1, ls);
            if (!waitQueue()) return 1;

            // mvec: all zeros. R16G16_FLOAT zero bits = 0.0
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
            // both static inputs move to their tagged state once
            D3D12_RESOURCE_BARRIER bs[2]{};
            for (int i = 0; i < 2; i++)
            {
                bs[i].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                bs[i].Transition.pResource = i ? texDepth.Get() : texMvec.Get();
                bs[i].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                bs[i].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
                bs[i].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            }
            list->ResourceBarrier(2, bs);
            list->Close();
            queue->ExecuteCommandLists(1, ls);
            if (!waitQueue()) return 1;
        }

        fillConstants(consts);
        return 0;
    }

    // upload an RGBA8 frame and present it through DLSS-G
    bool presentFrame(const uint8_t* rgba)
    {
        pumpMessages();

        uint8_t* dst = nullptr;
        if (FAILED(uploadBuf->Map(0, nullptr, (void**)&dst))) return false;
        for (uint32_t y = 0; y < H; y++)
            memcpy(dst + (size_t)y * rowPitch, rgba + (size_t)y * W * 4, W * 4);
        uploadBuf->Unmap(0, nullptr);

        uint32_t fi = frameIndex++;
        sl::FrameToken* tok{};
        if (slGetNewFrameToken(tok, &fi) != sl::Result::eOk) return false;

        slReflexSleep(*tok);
        slPCLSetMarker(sl::PCLMarker::eSimulationStart, *tok);
        if (slSetConstants(consts, *tok, vp) != sl::Result::eOk) return false;
        slPCLSetMarker(sl::PCLMarker::eSimulationEnd, *tok);

        UINT bb = sc->GetCurrentBackBufferIndex();

        alloc->Reset();
        list->Reset(alloc.Get(), nullptr);
        D3D12_TEXTURE_COPY_LOCATION dl{ texFrame.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX };
        D3D12_TEXTURE_COPY_LOCATION sl_{ uploadBuf.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT };
        sl_.PlacedFootprint.Footprint = { DXGI_FORMAT_R8G8B8A8_UNORM, W, H, 1, rowPitch };
        list->CopyTextureRegion(&dl, 0, 0, 0, &sl_, nullptr);

        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
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
        list->Close();

        sl::Extent fullExtent{ 0, 0, W, H };
        sl::Resource depthRes(sl::ResourceType::eTex2d, texDepth.Get(),
                              (uint32_t)(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE));
        depthRes.width = W; depthRes.height = H; depthRes.nativeFormat = DXGI_FORMAT_R32_FLOAT;
        sl::Resource mvecRes(sl::ResourceType::eTex2d, texMvec.Get(),
                             (uint32_t)(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE));
        mvecRes.width = W; mvecRes.height = H; mvecRes.nativeFormat = DXGI_FORMAT_R16G16_FLOAT;
        // NOTE: do NOT tag kBufferTypeBackbuffer here, even with a null resource "just to silence
        // the extent warning": the tag flips sl.dlss_g into its subrect present path and the real
        // frame ends up in the native swap chain right behind the generated one, so the readback
        // returns the REAL frame (tween == right endpoint, verified 2026-07-12). The extent
        // warning it would silence is benign and filtered in slLog instead.
        sl::ResourceTag tags[] = {
            sl::ResourceTag(&depthRes, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilPresent, &fullExtent),
            sl::ResourceTag(&mvecRes, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eValidUntilPresent, &fullExtent),
        };
        if (slSetTagForFrame(*tok, vp, tags, _countof(tags), nullptr) != sl::Result::eOk) return false;

        slPCLSetMarker(sl::PCLMarker::eRenderSubmitStart, *tok);
        ID3D12CommandList* ls[] = { list.Get() };
        queue->ExecuteCommandLists(1, ls);
        slPCLSetMarker(sl::PCLMarker::eRenderSubmitEnd, *tok);

        scNative->GetLastPresentCount(&presentSnapshot);
        slPCLSetMarker(sl::PCLMarker::ePresentStart, *tok);
        HRESULT hr = sc->Present(syncInterval, 0);
        slPCLSetMarker(sl::PCLMarker::ePresentEnd, *tok);
        if (FAILED(hr)) { LOG("Present failed hr=0x%08lx\n", hr); return false; }

        return waitQueue();
    }

    // copy one native swap-chain buffer into rgbaOut via the shared readback buffer
    bool copyNativeBuffer(UINT idx, uint8_t* rgbaOut)
    {
        ComPtr<ID3D12Resource> buf;
        if (FAILED(scNative->GetBuffer(idx, IID_PPV_ARGS(&buf)))) return false;

        alloc->Reset();
        list->Reset(alloc.Get(), nullptr);
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = buf.Get();
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
        b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        list->ResourceBarrier(1, &b);
        D3D12_TEXTURE_COPY_LOCATION src{ buf.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX };
        D3D12_TEXTURE_COPY_LOCATION dst{ readbackBuf.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT };
        dst.PlacedFootprint.Footprint = { DXGI_FORMAT_R8G8B8A8_UNORM, W, H, 1, rowPitch };
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
        b.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
        list->ResourceBarrier(1, &b);
        list->Close();
        ID3D12CommandList* ls[] = { list.Get() };
        queue->ExecuteCommandLists(1, ls);
        if (!waitQueue()) return false;

        uint8_t* mapped = nullptr;
        D3D12_RANGE rr{ 0, (SIZE_T)rowPitch * H };
        if (FAILED(readbackBuf->Map(0, &rr, (void**)&mapped))) return false;
        for (uint32_t y = 0; y < H; y++)
            memcpy(rgbaOut + (size_t)y * W * 4, mapped + (size_t)y * rowPitch, W * 4);
        D3D12_RANGE wr{ 0, 0 };
        readbackBuf->Unmap(0, &wr);
        return true;
    }

    // incremental present capture: append the pixels of every native present that landed since
    // the last poll (the ledger maps present k back to buffer (cur - 1 - (count - k)) mod N; 6
    // native buffers vs at most 5 generated frames means nothing is overwritten before we drain
    // it). The caller decides WHICH captured presents are generated frames by content - present
    // order cannot be trusted: a real-frame present sometimes slips into the stream despite
    // eShowOnlyInterpolatedFrame, and on some runs it even leads the generated one.
    UINT pollSeen = 0;   // reset to presentSnapshot after each proxy Present
    int pollNewPresents(std::vector<std::vector<uint8_t>>& raw)
    {
        UINT c = 0;
        scNative->GetLastPresentCount(&c);
        if (c == pollSeen) return 0;
        Sleep(waitMs);                       // settle: let the newest present's GPU write finish
        scNative->GetLastPresentCount(&c);   // re-snapshot after the settle (more may have landed)
        UINT cur = scNative->GetCurrentBackBufferIndex();
        int added = 0;
        for (UINT p = pollSeen + 1; p <= c; p++)
        {
            UINT idx = (cur + nativeBufCount - 1 - (c - p)) % nativeBufCount;
            raw.emplace_back((size_t)W * H * 4);
            if (!copyNativeBuffer(idx, raw.back().data())) { raw.pop_back(); return -1; }
            added++;
        }
        pollSeen = c;
        return added;
    }

    void shutdown()
    {
        sl::DLSSGOptions off{};
        off.mode = sl::DLSSGMode::eOff;
        slDLSSGSetOptions(vp, off);
        waitQueue();
        if (fenceEvent) CloseHandle(fenceEvent);
        if (hwnd) DestroyWindow(hwnd);
        pumpMessages();
    }
};

// ---------------------------------------------------------------- entry

static int slInitCommon()
{
    wchar_t exePath[MAX_PATH];
    GetModuleFileNameW(nullptr, exePath, MAX_PATH);
    static std::wstring exeDir(exePath);
    exeDir.resize(exeDir.find_last_of(L'\\'));
    static const wchar_t* pluginPaths[] = { exeDir.c_str() };
    static sl::Feature features[] = { sl::kFeatureDLSS_G, sl::kFeatureReflex, sl::kFeaturePCL };

    g_verbose = GetEnvironmentVariableW(L"DLSSG_VERBOSE", nullptr, 0) != 0;
    sl::Preferences pref{};
    pref.showConsole = false;
    pref.logLevel = g_verbose ? sl::LogLevel::eVerbose : sl::LogLevel::eDefault;
    pref.pathsToPlugins = pluginPaths;
    pref.numPathsToPlugins = 1;
    pref.pathToLogsAndData = nullptr;   // no log file; warnings/errors go to stderr via callback
    pref.logMessageCallback = slLog;
    pref.featuresToLoad = features;
    pref.numFeaturesToLoad = _countof(features);
    pref.applicationId = 231313132; // Streamline sample app id
    pref.flags |= sl::PreferenceFlags::eUseFrameBasedResourceTagging;
    // SL's defense against injected overlays (NVIDIA App overlay, Parsec, ...): they hook the
    // same DXGI v-tables and can silently break the interposer's present path
    pref.flags |= sl::PreferenceFlags::eUseDXGIFactoryProxy;
    pref.renderAPI = sl::RenderAPI::eD3D12;
    CHECK_SL(slInit(pref, sl::kSDKVersion));
    return 0;
}

static int runServer(int waitMs, int genFrames, bool vsync, bool onscreen)
{
    Host host;
    host.waitMs = waitMs;
    host.genFrames = genFrames;
    host.syncInterval = vsync ? 1 : 0;
    int rc = host.init(onscreen);
    if (rc) return rc;

    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);

    // handshake so the driving process can distinguish "up" from "unsupported"
    fprintf(stdout, "DLSSG READY gen=%d max=%u\n", genFrames, host.maxGen);
    fflush(stdout);

    const size_t frameBytes = (size_t)W * H * 4;
    std::vector<uint8_t> inFrame(frameBytes), prevFrame(frameBytes);
    std::vector<std::vector<uint8_t>> outFrames((size_t)genFrames, std::vector<uint8_t>(frameBytes));
    std::vector<uint8_t*> outPtrs;
    for (auto& f : outFrames) outPtrs.push_back(f.data());

    // wait until the pacer's native presents go quiet (used after warmup and after resyncs, so
    // leftover generated presents are never misattributed to the next pair's ledger)
    auto drainQuiet = [&]()
    {
        UINT stable = 0;
        int quietMs = 0;
        while (quietMs < 100)
        {
            UINT c = 0;
            host.scNative->GetLastPresentCount(&c);
            if (c == stable) { Sleep(5); quietMs += 5; }
            else { stable = c; quietMs = 0; }
        }
    };

    auto readFull = [&](uint8_t* p, size_t n) -> bool
    {
        size_t got = 0;
        while (got < n)
        {
            size_t r = fread(p + got, 1, n - got, stdin);
            if (r == 0) return false;
            got += r;
        }
        return true;
    };

    // sampled RGB equality (alpha skipped): true when two frames carry identical image content.
    // Used to detect the pacer presenting the REAL frame instead of a generated one.
    auto diffSamples = [&](const uint8_t* a, const uint8_t* b) -> int
    {
        const size_t px = (size_t)W * H;
        int n = 0;
        for (size_t i = 0; i < px; i += 397)
        {
            const size_t o = i * 4;
            if (a[o] != b[o] || a[o + 1] != b[o + 1] || a[o + 2] != b[o + 2]) n++;
        }
        return n;
    };
    auto sameImage = [&](const uint8_t* a, const uint8_t* b) -> bool
    {
        return diffSamples(a, b) == 0;
    };

    // DLSS-G reset: off (applied by a sacrificial present - options only take effect AT a
    // present), back on, re-warm on the previous frame, drain. This recovers the pacer after an
    // input gap (pause). It does NOT cure the machine-level "poisoned window" state in which the
    // pacer stops delivering generated frames entirely - nothing in-process does (mode toggles,
    // full swap-chain re-creation and even fresh processes were all tried and failed while a
    // window was active; they pass again once it lifts on its own after a few minutes).
    auto resetFG = [&]() -> bool
    {
        sl::DLSSGOptions off{};
        off.mode = sl::DLSSGMode::eOff;
        off.flags = sl::DLSSGFlags::eRetainResourcesWhenOff;
        if (slDLSSGSetOptions(host.vp, off) != sl::Result::eOk) return false;
        if (!host.presentFrame(prevFrame.data())) return false;
        sl::DLSSGOptions on{};
        on.mode = sl::DLSSGMode::eOn;
        on.numFramesToGenerate = (uint32_t)genFrames;
        if (slDLSSGSetOptions(host.vp, on) != sl::Result::eOk) return false;
        for (int i = 0; i < 2; i++)
            if (!host.presentFrame(prevFrame.data())) return false;
        drainQuiet();
        return true;
    };

    uint64_t emitted = 0;
    uint64_t resets = 0;
    bool first = true;
    bool metered = false;   // sticky: hardware flip metering detected, capture by buffer sweep
    std::vector<std::vector<uint8_t>> preSig(host.nativeBufCount);
    std::vector<uint8_t> sweepScratch(frameBytes);
    ULONGLONG lastPresentTick = 0;
    while (readFull(inFrame.data(), frameBytes))
    {
        if (first)
        {
            // warm-up: FG feature builds on the first present; interp(f0,f0)=f0
            for (int i = 0; i < 3; i++)
                if (!host.presentFrame(inFrame.data())) { LOG("warmup present failed\n"); return 1; }
            sl::DLSSGState st{};
            if (slDLSSGGetState(host.vp, st, nullptr) == sl::Result::eOk)
            {
                if (st.status != sl::DLSSGStatus::eOk)
                {
                    LOG("DLSS-G runtime status=%d, aborting\n", (int)st.status);
                    return 1;
                }
                // definitive multi-frame limit (may have been unknown before the first present)
                if (st.numFramesToGenerateMax && (uint32_t)genFrames > st.numFramesToGenerateMax)
                {
                    LOG("multi-frame generation beyond %ux is not supported on this GPU (requested %ux)\n",
                        st.numFramesToGenerateMax + 1, genFrames + 1);
                    return 3;
                }
            }
            // drain: the warmup presents spawn generated presents of their own (interp(f0,f0));
            // wait until the native present count goes quiet so pair 0's ledger starts clean
            drainQuiet();
            prevFrame.swap(inFrame);
            lastPresentTick = GetTickCount64();
            first = false;
            continue;
        }

        // pair = (prevFrame, inFrame). A long input gap (GUI pause, encoder stall) is a
        // near-certain pacer poisoning, so reset proactively instead of wasting an attempt.
        if (GetTickCount64() - lastPresentTick > 500)
        {
            LOG("input gap detected, resetting DLSS-G\n");
            resets++;
            if (!resetFG()) { LOG("DLSS-G reset failed\n"); return 1; }
        }

        // present + content-selected capture, two tiers:
        //  1. FAST: poll counted DXGI presents and keep the first genFrames that are not
        //     byte-copies of either real frame (real presents slip into the stream, order is
        //     untrustworthy).
        //  2. SWEEP: when the driver switches to HARDWARE FLIP METERING (observed e.g. while
        //     NVIDIA RTX Video processes a browser video, and it can persist after), generated
        //     frames still land in the native swap-chain buffers but their presents are NOT
        //     counted by DXGI - so sweep all buffers and select by content, using per-buffer
        //     signatures from before the present to reject stale frames, and distance-to-A
        //     ordering for multi-frame generation (it grows monotonically with the timestep).
        const bool pairStatic = sameImage(prevFrame.data(), inFrame.data());
        std::vector<std::vector<uint8_t>> raw;
        bool good = false;

        auto takeSig = [&](const uint8_t* f, std::vector<uint8_t>& sig)
        {
            sig.clear();
            const size_t px = (size_t)W * H;
            for (size_t i = 0; i < px; i += 397)
            {
                const size_t o = i * 4;
                sig.push_back(f[o]); sig.push_back(f[o + 1]); sig.push_back(f[o + 2]);
            }
        };
        if (metered)
        {
            // pre-present snapshot of every buffer so the sweep can tell fresh from stale
            for (UINT i = 0; i < host.nativeBufCount; i++)
                if (host.copyNativeBuffer(i, sweepScratch.data()))
                    takeSig(sweepScratch.data(), preSig[i]);
        }
        // bounded backoff: temporary system conditions can make the pacer stop delivering
        // generated frames, and no in-process reset cures an active "poisoned window" - short
        // windows are ridden out (~30s total), longer ones fail loudly (silently emitting the
        // real-frame copies the pacer falls back to would corrupt the render with holds).
        static const int backoffMs[] = { 0, 500, 1000, 2000, 4000, 8000, 15000 };
        for (int attempt = 0; attempt < (int)_countof(backoffMs) && !good; attempt++)
        {
            if (attempt > 0)
            {
                LOG("generated frames missing, resetting DLSS-G and retrying in %d ms (attempt %d)\n",
                    backoffMs[attempt], attempt);
                resets++;
                Sleep(backoffMs[attempt]);
                if (!resetFG()) { LOG("DLSS-G reset failed\n"); return 1; }
            }
            if (!host.presentFrame(inFrame.data())) { LOG("presentFrame failed\n"); return 1; }
            host.pollSeen = host.presentSnapshot;
            raw.clear();
            size_t scanned = 0;
            int filled = 0;
            const ULONGLONG t0 = GetTickCount64();
            ULONGLONG nextSweep = t0 + (metered ? 50 : 350);
            while (filled < genFrames && GetTickCount64() < t0 + 1000)
            {
                int added = host.pollNewPresents(raw);
                if (added < 0) { LOG("present readback failed\n"); return 1; }
                for (; scanned < raw.size() && filled < genFrames; scanned++)
                {
                    const uint8_t* f = raw[scanned].data();
                    if (!pairStatic && (sameImage(f, inFrame.data()) || sameImage(f, prevFrame.data())))
                        continue;   // a real-frame present, not a generated frame
                    memcpy(outPtrs[filled++], f, frameBytes);
                }
                if (filled >= genFrames) break;
                if (GetTickCount64() >= nextSweep)
                {
                    nextSweep = GetTickCount64() + 50;
                    // sweep every native buffer; fresh non-copies are this pair's generated frames
                    std::vector<std::vector<uint8_t>> cand;
                    std::vector<uint8_t> sig;
                    for (UINT i = 0; i < host.nativeBufCount; i++)
                    {
                        if (!host.copyNativeBuffer(i, sweepScratch.data())) continue;
                        const uint8_t* f = sweepScratch.data();
                        if (!pairStatic && (sameImage(f, inFrame.data()) || sameImage(f, prevFrame.data())))
                            continue;
                        if (metered && !preSig[i].empty())
                        {
                            takeSig(f, sig);
                            if (sig == preSig[i]) continue;   // unchanged since before the present: stale
                        }
                        cand.push_back(sweepScratch);
                    }
                    if ((int)cand.size() >= genFrames)
                    {
                        // multi-frame order: distance to the left real frame grows with the timestep
                        std::sort(cand.begin(), cand.end(),
                                  [&](const std::vector<uint8_t>& a, const std::vector<uint8_t>& b)
                                  { return diffSamples(a.data(), prevFrame.data()) < diffSamples(b.data(), prevFrame.data()); });
                        for (filled = 0; filled < genFrames; filled++)
                            memcpy(outPtrs[filled], cand[filled].data(), frameBytes);
                        if (!metered)
                        {
                            LOG("hardware flip metering detected (generated presents are not counted); "
                                "switching to buffer-sweep capture\n");
                            metered = true;
                        }
                        break;
                    }
                }
                if (added == 0) Sleep(1);
            }
            good = (filled == genFrames);
            if (!good)
            {
                LOG("captured %d generated frame(s) of %d (plus %zu real presents) within 1s\n",
                    filled, genFrames, raw.size() - filled);
                // post-mortem: what is in each native buffer right now?
                std::vector<uint8_t> sig;
                for (UINT i = 0; i < host.nativeBufCount; i++)
                {
                    if (!host.copyNativeBuffer(i, sweepScratch.data())) { LOG("  buf%u: copy failed\n", i); continue; }
                    const uint8_t* f = sweepScratch.data();
                    int dA = diffSamples(f, prevFrame.data());
                    int dB = diffSamples(f, inFrame.data());
                    bool stale = false;
                    if (metered && !preSig[i].empty()) { takeSig(f, sig); stale = (sig == preSig[i]); }
                    LOG("  buf%u: dA=%d dB=%d%s\n", i, dA, dB, stale ? " (unchanged since pre-present)" : "");
                }
            }
        }
        if (!good)
        {
            LOG("DLSS-FG stopped producing generated frames and did not recover within ~30s of "
                "retries. Root cause on the reference machine: NVIDIA RTX Video enhancement "
                "(Super Resolution / RTX Video HDR) processing a video playing in a browser "
                "preempts frame generation machine-wide. Pause the video (or disable RTX Video "
                "enhancements in the NVIDIA App) and start the render again, or use GMFSS.\n");
            return 1;
        }
        lastPresentTick = GetTickCount64();

        for (int i = 0; i < genFrames; i++)
            fwrite(outPtrs[i], 1, frameBytes, stdout);
        fflush(stdout);
        emitted += genFrames;
        prevFrame.swap(inFrame);
    }
    if (resets)
        LOG("note: %llu DLSS-G pacer resets during this run\n", (unsigned long long)resets);
    LOG("server done, %llu interpolated frames\n", (unsigned long long)emitted);
    host.shutdown();
    slShutdown();
    return 0;
}

static int runSingleShot(const wchar_t* pathA, const wchar_t* pathB, const wchar_t* pathOut, bool gdi)
{
    CHECK_HR(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&g_wic)));

    Host host;
    int rc = host.init(gdi); // GDI validation needs the window on-screen
    if (rc) return rc;

    std::vector<uint8_t> imgA, imgB;
    if (!loadPngRGBA(pathA, imgA) || !loadPngRGBA(pathB, imgB)) { LOG("png load failed\n"); return 1; }

    for (int i = 0; i < 3; i++)
        if (!host.presentFrame(imgA.data())) { LOG("warmup present failed\n"); return 1; }

    sl::DLSSGState st{};
    if (slDLSSGGetState(host.vp, st, nullptr) == sl::Result::eOk)
        LOG("DLSS-G status=%d minWH=%u maxGen=%u\n", (int)st.status, st.minWidthOrHeight, st.numFramesToGenerateMax);
    if (st.status != sl::DLSSGStatus::eOk) { LOG("DLSS-G runtime status NOT ok\n"); return 1; }

    if (!host.presentFrame(imgB.data())) { LOG("B present failed\n"); return 1; }

    if (GetEnvironmentVariableW(L"DLSSG_DUMPBUFS", nullptr, 0))
    {
        // diagnostic: dump every native buffer after the pair present, to learn where generated
        // frames live when the driver's hardware flip queue bypasses countable DXGI presents
        Sleep(200);
        std::vector<uint8_t> buf((size_t)W * H * 4);
        for (UINT i = 0; i < host.nativeBufCount; i++)
        {
            if (!host.copyNativeBuffer(i, buf.data())) { LOG("buffer %u copy failed\n", i); continue; }
            for (size_t p = 0; p < buf.size(); p += 4) std::swap(buf[p], buf[p + 2]);
            wchar_t name[64];
            swprintf_s(name, L"out_buf%u.png", i);
            savePng(name, buf.data(), W, H, true);
        }
        LOG("dumped %u native buffers\n", host.nativeBufCount);
    }

    bool ok = false;
    if (gdi)
    {
        Sleep(250);
        RECT wr{};
        GetWindowRect(host.hwnd, &wr);
        ok = captureScreen(wr.left, wr.top, W, H, pathOut);
    }
    else
    {
        auto sameImage = [&](const uint8_t* a, const uint8_t* b) -> bool
        {
            for (size_t i = 0; i < (size_t)W * H; i += 397)
            {
                const size_t o = i * 4;
                if (a[o] != b[o] || a[o + 1] != b[o + 1] || a[o + 2] != b[o + 2]) return false;
            }
            return true;
        };
        std::vector<uint8_t> out((size_t)W * H * 4);
        std::vector<std::vector<uint8_t>> raw;
        host.pollSeen = host.presentSnapshot;
        size_t scanned = 0;
        const ULONGLONG deadline = GetTickCount64() + 1000;
        while (!ok && GetTickCount64() < deadline)
        {
            int added = host.pollNewPresents(raw);
            if (added < 0) break;
            for (; scanned < raw.size() && !ok; scanned++)
            {
                const uint8_t* f = raw[scanned].data();
                if (sameImage(f, imgA.data()) || sameImage(f, imgB.data())) continue;
                memcpy(out.data(), f, out.size());
                ok = true;
            }
            if (added == 0) Sleep(1);
        }
        // WIC's PNG encoder negotiates to BGRA regardless of the requested format, so swizzle
        for (size_t i = 0; ok && i < out.size(); i += 4) std::swap(out[i], out[i + 2]);
        if (ok) ok = savePng(pathOut, out.data(), W, H, true);
    }
    if (!ok) { LOG("capture failed\n"); return 1; }
    LOG("wrote %ls (%s)\n", pathOut, gdi ? "gdi" : "native readback");

    host.shutdown();
    slShutdown();
    return 0;
}

int wmain(int argc, wchar_t** argv)
{
    setvbuf(stderr, nullptr, _IONBF, 0);
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    CHECK_HR(CoInitializeEx(nullptr, COINIT_MULTITHREADED));

    if (argc >= 4 && wcscmp(argv[1], L"--server") == 0)
    {
        W = (uint32_t)_wtoi(argv[2]);
        H = (uint32_t)_wtoi(argv[3]);
        if (W < 128 || H < 128 || W > 8192 || H > 8192) { LOG("bad size %ux%u\n", W, H); return 1; }
        int waitMs = 5;
        int genFrames = 1;
        bool vsync = true;   // vsync pacing measured far more reliable for the capture (see BUILD.md)
        bool onscreen = false;
        for (int i = 4; i < argc; i++)
        {
            if (wcscmp(argv[i], L"--wait") == 0 && i + 1 < argc) waitMs = _wtoi(argv[++i]);
            else if (wcscmp(argv[i], L"--gen") == 0 && i + 1 < argc) genFrames = _wtoi(argv[++i]);
            else if (wcscmp(argv[i], L"--vsync") == 0) vsync = true;
            else if (wcscmp(argv[i], L"--no-vsync") == 0) vsync = false;
            else if (wcscmp(argv[i], L"--onscreen") == 0) onscreen = true;
        }
        if (genFrames < 1 || genFrames > 5) { LOG("--gen must be 1..5 (got %d)\n", genFrames); return 1; }
        if (slInitCommon()) return 1;
        return runServer(waitMs, genFrames, vsync, onscreen);
    }

    if (argc >= 4)
    {
        bool gdi = (argc >= 5 && wcscmp(argv[4], L"--gdi") == 0);
        if (slInitCommon()) return 1;
        return runSingleShot(argv[1], argv[2], argv[3], gdi);
    }

    LOG("usage: dlssg2f.exe frameA.png frameB.png out.png [--gdi]\n"
        "       dlssg2f.exe --server W H [--gen N] [--wait ms] [--onscreen]\n");
    return 1;
}
