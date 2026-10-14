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
// Usage:
//   dlssg2f.exe --server W H [--gen N] [--hdr10]
//       streaming: raw RGBA8 frames (W*H*4) on stdin; for every frame after the
//       first, the N (default 1, max 5) DLSS-G frames generated between it and
//       its predecessor - evenly spaced, in temporal order - are written to
//       stdout as raw RGBA8 (multi-frame generation: N=1 is 2x, N=5 is 6x).
//       --hdr10: HDR10 frames both ways instead (PQ BT.2020 as R10G10B10A2 words,
//       still W*H*4 bytes), presented the way the DLSS-G guide asks for HDR
//       (section 11.0): an R10G10B10A2 swap chain with the G2084 / P2020 colour
//       space. The handshake then ends in " hdr10=1"; a swap chain that refuses
//       that colour space exits 5 before the handshake.
//       All logging goes to stderr; stdout is binary-only. EOF on stdin exits.
//       This is the SmoothMyVideo backend.
//   dlssg2f.exe --stage1 W H [--gen N] [--hdr10 | --bgra] --share TAG [--from TAG]
//       a stage of a live cascade before the host's own: --from = another stage's ring as its input (on the GPU,
//       in present order; stdin then only ends it), else the same frames on stdin (--bgra: SDR frames in B8G8R8A8 order, on a
//       B8G8R8A8 swap chain and ring, the order of Live's effects route), each presented the moment it arrives
//       (the cadence of a game, so DLSS-G's pacer spreads its frames over the source's own interval); every
//       native present, generated and real alike in present order, is copied on the GPU into a ring of
//       textures shared by name with the process that reads them (smv_dg1_TAG_ring0.., a shared fence
//       smv_dg1_TAG_fence, the header Stage1Hdr in the file mapping smv_dg1_TAG_hdr). stdout carries only
//       the handshake line.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <dxgidebug.h>
#include <Psapi.h>
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
#include <atomic>
#include <thread>

#include <sl.h>
#include <sl_consts.h>
#include <sl_dlss_g.h>
#include <sl_reflex.h>
#include <sl_pcl.h>

using Microsoft::WRL::ComPtr;

static uint32_t W = 1920;
static uint32_t H = 1080;

#define LOG(...)                                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        fprintf(stderr, __VA_ARGS__);                                                                                  \
        fflush(stderr);                                                                                                \
    } while (0)
#define CHECK_HR(x)                                                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        HRESULT _hr = (x);                                                                                             \
        if (FAILED(_hr))                                                                                               \
        {                                                                                                              \
            LOG("dlssg2f FAIL hr=0x%08lx at %s:%d\n", _hr, __FILE__, __LINE__);                                        \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)
#define CHECK_SL(x)                                                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        sl::Result _r = (x);                                                                                           \
        if (_r != sl::Result::eOk)                                                                                     \
        {                                                                                                              \
            LOG("dlssg2f FAIL sl::Result=%d at %s:%d\n", (int)_r, __FILE__, __LINE__);                                 \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

static bool g_verbose = false;

static void slLog(sl::LogType type, const char* msg)
{
    if (g_verbose)
    {
        LOG("[SL%d] %s", (int)type, msg);
        return;
    }
    if (type == sl::LogType::eInfo)
        return;
    // Known-benign SDK chatter this host provokes by design; keep it out of the user-visible
    // engine log. Real warnings and all errors still pass through.
    static const char* benign[] = {
        "Invalid backbuffer resource extent", // belt-and-braces: we do tag the full extent
        "reseting frame timer",               // game pacer notices our slow present cadence
        "Frame rate over",
        "duplicated unique id",                         // exe dir scanned twice (pathsToPlugins + cwd)
        "invoked before slInit",                        // process-startup probe order, harmless here
        "must be synchronized with the present thread", // we are single-threaded by construction
    };
    for (const char* b : benign)
        if (strstr(msg, b))
            return;
    LOG("[SL%d] %s", (int)type, msg);
}

#include "sl_focus_shim.h"

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

// ---------------------------------------------------------------- host

struct Host
{
    HWND hwnd{};
    ComPtr<IDXGIFactory2> factory;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<IDXGISwapChain3> sc;       // SL proxy
    ComPtr<IDXGISwapChain3> scNative; // real swap chain behind the proxy
    UINT nativeBufCount = 0;
    ComPtr<ID3D12Resource> backbuffers[3];
    ComPtr<ID3D12Resource> texFrame; // upload target, copied to backbuffer each present
    ComPtr<ID3D12Resource> texDepth;
    ComPtr<ID3D12Resource> texMvec;
    ComPtr<ID3D12Resource> uploadBuf;
    ComPtr<ID3D12Resource> readbackBuf;
    UINT rowPitch = 0; // 256-aligned pitch shared by upload + readback
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    uint64_t fenceValue = 0;
    HANDLE fenceEvent{};
    sl::ViewportHandle vp{0u};
    sl::Constants consts{};
    uint32_t frameIndex = 1;
    int waitMs = 5;           // settle after a native present is observed
    int genFrames = 1;        // generated frames per source frame (numFramesToGenerate)
    UINT syncInterval = 1;    // Present sync interval: vsync-paced FG, measured far more reliable for the capture
    uint32_t maxGen = 0;      // device limit reported by the SDK (0 = not yet known)
    UINT presentSnapshot = 0; // native present count taken just before each proxy Present
    bool hdr10 = false;       // --hdr10: R10G10B10A2 frames on a G2084 / P2020 swap chain
    bool bgra = false;        // --bgra (--stage1, SDR): B8G8R8A8 frames
    bool hdr10Refused = false;

    DXGI_FORMAT frameFormat() const
    {
        return hdr10 ? DXGI_FORMAT_R10G10B10A2_UNORM : bgra ? DXGI_FORMAT_B8G8R8A8_UNORM : DXGI_FORMAT_R8G8B8A8_UNORM;
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

    // create (or re-create) the swap chain through SL's proxy factory and resolve the native
    // chain behind it - the buffers DLSS-G presents into and we read generated frames from
    bool createSwapchain(bool logIt)
    {
        DXGI_SWAP_CHAIN_DESC1 scd{};
        scd.Width = W;
        scd.Height = H;
        scd.Format = frameFormat();
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

        ComPtr<IUnknown> nativeUnk;
        if (slGetNativeInterface(sc.Get(), (void**)nativeUnk.GetAddressOf()) != sl::Result::eOk)
            return false;
        if (FAILED(nativeUnk.As(&scNative)))
            return false;
        if (hdr10)
        {
            // declared on the proxy chain too: the interposer only learns the colour space from a call it hooks, and
            // an R10G10B10A2 chain without the HDR10 colour space passes through ungenerated (measured on the live host)
            const DXGI_COLOR_SPACE_TYPE cs = DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020;
            UINT support = 0;
            if (GetEnvironmentVariableW(L"SMV_DLSSG_REFUSE_HDR10", nullptr, 0) ||
                FAILED(scNative->CheckColorSpaceSupport(cs, &support)) ||
                !(support & DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT) || FAILED(scNative->SetColorSpace1(cs)) ||
                FAILED(sc->SetColorSpace1(cs)))
            {
                hdr10Refused = true;
                return false;
            }
        }
        DXGI_SWAP_CHAIN_DESC1 nd{};
        scNative->GetDesc1(&nd);
        nativeBufCount = nd.BufferCount;
        if (logIt)
            LOG("native swap chain: %ux%u fmt=%d buffers=%u%s\n", nd.Width, nd.Height, (int)nd.Format, nativeBufCount,
                hdr10 ? ", HDR10 (G2084 / P2020)" : "");
        return true;
    }

    int init()
    {
        // window: swap chain needs an HWND, but with native readback it never
        // has to be visible on the desktop, so park it far offscreen
        WNDCLASSW wc{};
        wc.lpfnWndProc = wndProc;
        wc.hInstance = GetModuleHandleW(nullptr);
        wc.lpszClassName = L"dlssg2f";
        wc.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));
        RegisterClassW(&wc);
        hwnd = CreateWindowExW(WS_EX_TOOLWINDOW, L"dlssg2f", L"dlssg2f", WS_POPUP | WS_VISIBLE, -32000, -32000, W, H,
                               nullptr, nullptr, wc.hInstance, nullptr);
        if (!hwnd)
        {
            LOG("CreateWindow failed\n");
            return 1;
        }
        pumpMessages();

        CHECK_HR(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)));
        ComPtr<IDXGIAdapter1> adapter;
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

        D3D12_COMMAND_QUEUE_DESC qd{};
        qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        CHECK_HR(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)));
        if (!createSwapchain(true))
        {
            if (!hdr10Refused)
                return 1;
            LOG("the swap chain refused the HDR10 colour space (G2084 / P2020)\n");
            return 5;
        }

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
            LOG("multi-frame generation beyond %ux is not supported on this GPU (requested %ux)\n", maxGen + 1,
                genFrames + 1);
            return 3;
        }

        // static resources
        CHECK_HR(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc)));
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
        if (!makeTex(frameFormat(), texFrame) || !makeTex(DXGI_FORMAT_R32_FLOAT, texDepth) ||
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
            D3D12_TEXTURE_COPY_LOCATION dl{texDepth.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            D3D12_TEXTURE_COPY_LOCATION sl_{uploadBuf.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
            sl_.PlacedFootprint.Footprint = {DXGI_FORMAT_R32_FLOAT, W, H, 1, rowPitch};
            list->CopyTextureRegion(&dl, 0, 0, 0, &sl_, nullptr);
            list->Close();
            ID3D12CommandList* ls[] = {list.Get()};
            queue->ExecuteCommandLists(1, ls);
            if (!waitQueue())
                return 1;

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
                bs[i].Transition.StateAfter =
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            }
            list->ResourceBarrier(2, bs);
            list->Close();
            queue->ExecuteCommandLists(1, ls);
            if (!waitQueue())
                return 1;
        }

        fillConstants(consts);
        return 0;
    }

    // upload a frame (RGBA8, or R10G10B10A2 with --hdr10) and present it through DLSS-G; rgba null = a texture of
    // this device in frameFormat() instead (src, after srcFence reaches srcValue: an upstream stage's ring slot,
    // resting in COMMON), or with no src the frame presented last again
    bool presentFrame(const uint8_t* rgba, ID3D12Resource* src = nullptr, ID3D12Fence* srcFence = nullptr,
                      uint64_t srcValue = 0)
    {
        pumpMessages();

        if (rgba)
        {
            uint8_t* dst = nullptr;
            if (FAILED(uploadBuf->Map(0, nullptr, (void**)&dst)))
                return false;
            for (uint32_t y = 0; y < H; y++)
                memcpy(dst + (size_t)y * rowPitch, rgba + (size_t)y * W * 4, W * 4);
            uploadBuf->Unmap(0, nullptr);
        }

        uint32_t fi = frameIndex++;
        sl::FrameToken* tok{};
        if (slGetNewFrameToken(tok, &fi) != sl::Result::eOk)
            return false;

        slReflexSleep(*tok);
        slPCLSetMarker(sl::PCLMarker::eSimulationStart, *tok);
        if (slSetConstants(consts, *tok, vp) != sl::Result::eOk)
            return false;
        slPCLSetMarker(sl::PCLMarker::eSimulationEnd, *tok);

        UINT bb = sc->GetCurrentBackBufferIndex();

        alloc->Reset();
        list->Reset(alloc.Get(), nullptr);
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        if (rgba)
        {
            D3D12_TEXTURE_COPY_LOCATION dl{texFrame.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            D3D12_TEXTURE_COPY_LOCATION sl_{uploadBuf.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
            sl_.PlacedFootprint.Footprint = {frameFormat(), W, H, 1, rowPitch};
            list->CopyTextureRegion(&dl, 0, 0, 0, &sl_, nullptr);
        }
        else if (src)
        {
            b.Transition.pResource = src;
            b.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
            b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
            list->ResourceBarrier(1, &b);
            list->CopyResource(texFrame.Get(), src);
            std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
            list->ResourceBarrier(1, &b);
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
        list->Close();

        sl::Extent fullExtent{0, 0, W, H};
        sl::Resource depthRes(
            sl::ResourceType::eTex2d, texDepth.Get(),
            (uint32_t)(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE));
        depthRes.width = W;
        depthRes.height = H;
        depthRes.nativeFormat = DXGI_FORMAT_R32_FLOAT;
        sl::Resource mvecRes(
            sl::ResourceType::eTex2d, texMvec.Get(),
            (uint32_t)(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE));
        mvecRes.width = W;
        mvecRes.height = H;
        mvecRes.nativeFormat = DXGI_FORMAT_R16G16_FLOAT;
        // NOTE: do NOT tag kBufferTypeBackbuffer here, even with a null resource "just to silence
        // the extent warning": the tag flips sl.dlss_g into its subrect present path and the real
        // frame ends up in the native swap chain right behind the generated one, so the readback
        // returns the REAL frame (tween == right endpoint, verified). The extent
        // warning it would silence is benign and filtered in slLog instead.
        sl::ResourceTag tags[] = {
            sl::ResourceTag(&depthRes, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilPresent, &fullExtent),
            sl::ResourceTag(&mvecRes, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eValidUntilPresent,
                            &fullExtent),
        };
        if (slSetTagForFrame(*tok, vp, tags, _countof(tags), nullptr) != sl::Result::eOk)
            return false;

        if (src && srcFence)
            queue->Wait(srcFence, srcValue); // the upstream slot's copy on its own device is complete
        slPCLSetMarker(sl::PCLMarker::eRenderSubmitStart, *tok);
        ID3D12CommandList* ls[] = {list.Get()};
        queue->ExecuteCommandLists(1, ls);
        slPCLSetMarker(sl::PCLMarker::eRenderSubmitEnd, *tok);

        scNative->GetLastPresentCount(&presentSnapshot);
        slPCLSetMarker(sl::PCLMarker::ePresentStart, *tok);
        HRESULT hr = sc->Present(syncInterval, 0);
        slPCLSetMarker(sl::PCLMarker::ePresentEnd, *tok);
        if (FAILED(hr))
        {
            LOG("Present failed hr=0x%08lx\n", hr);
            return false;
        }

        return waitQueue();
    }

    // copy one native swap-chain buffer into rgbaOut via the shared readback buffer
    bool copyNativeBuffer(UINT idx, uint8_t* rgbaOut)
    {
        ComPtr<ID3D12Resource> buf;
        if (FAILED(scNative->GetBuffer(idx, IID_PPV_ARGS(&buf))))
            return false;

        alloc->Reset();
        list->Reset(alloc.Get(), nullptr);
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = buf.Get();
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
        b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        list->ResourceBarrier(1, &b);
        D3D12_TEXTURE_COPY_LOCATION src{buf.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        D3D12_TEXTURE_COPY_LOCATION dst{readbackBuf.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        dst.PlacedFootprint.Footprint = {frameFormat(), W, H, 1, rowPitch};
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
        b.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
        list->ResourceBarrier(1, &b);
        list->Close();
        ID3D12CommandList* ls[] = {list.Get()};
        queue->ExecuteCommandLists(1, ls);
        if (!waitQueue())
            return false;

        uint8_t* mapped = nullptr;
        D3D12_RANGE rr{0, (SIZE_T)rowPitch * H};
        if (FAILED(readbackBuf->Map(0, &rr, (void**)&mapped)))
            return false;
        for (uint32_t y = 0; y < H; y++)
            memcpy(rgbaOut + (size_t)y * W * 4, mapped + (size_t)y * rowPitch, W * 4);
        D3D12_RANGE wr{0, 0};
        readbackBuf->Unmap(0, &wr);
        return true;
    }

    // incremental present capture: append the pixels of every native present that landed since
    // the last poll (the ledger maps present k back to buffer (cur - 1 - (count - k)) mod N; 6
    // native buffers vs at most 5 generated frames means nothing is overwritten before we drain
    // it). The caller decides WHICH captured presents are generated frames by content - present
    // order cannot be trusted: a real-frame present sometimes slips into the stream despite
    // eShowOnlyInterpolatedFrame, and on some runs it even leads the generated one.
    UINT pollSeen = 0; // reset to presentSnapshot after each proxy Present
    int pollNewPresents(std::vector<std::vector<uint8_t>>& raw)
    {
        UINT c = 0;
        scNative->GetLastPresentCount(&c);
        if (c == pollSeen)
            return 0;
        Sleep(waitMs);                     // settle: let the newest present's GPU write finish
        scNative->GetLastPresentCount(&c); // re-snapshot after the settle (more may have landed)
        UINT cur = scNative->GetCurrentBackBufferIndex();
        int added = 0;
        for (UINT p = pollSeen + 1; p <= c; p++)
        {
            UINT idx = (cur + nativeBufCount - 1 - (c - p)) % nativeBufCount;
            raw.emplace_back((size_t)W * H * 4);
            if (!copyNativeBuffer(idx, raw.back().data()))
            {
                raw.pop_back();
                return -1;
            }
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
        if (fenceEvent)
            CloseHandle(fenceEvent);
        if (hwnd)
            DestroyWindow(hwnd);
        pumpMessages();
    }
};

// ---------------------------------------------------------------- --stage1: the shared present ring

// The header a reader maps (smv_dg1_TAG_hdr). A slot's copy is complete once the shared fence reaches its slotFence;
// slotSeq = the present's number since the ring started (a reader that finds a newer number lost that slot)
struct Stage1Hdr
{
    uint32_t magic;   // 'SMV1'
    uint32_t version; // 1
    volatile uint64_t frames;
    uint32_t w, h, ring, fmt;
    uint64_t allocSize;
    volatile uint32_t status;    // 0 starting, 1 running, 2 ended, 3 failed
    volatile uint32_t lastInput; // the source frame presented last (frame 0 = the warm-up)
    volatile uint64_t slotFence[16];
    volatile uint64_t slotSeq[16];
    volatile int64_t slotQpc[16];    // QueryPerformanceCounter when the capture saw the present
    volatile uint32_t slotInput[16]; // lastInput at that moment
};

// Every native present copied into a ring of shared textures on the swap chain's own queue: that queue runs the copy
// after the present it follows, so the copy needs no settle; the rings rest in COMMON, the state another device's
// queue may assume
struct Stage1Ring
{
    static const int kRing = 12, kAllocs = 16;
    Host* h = nullptr;
    Stage1Hdr* hdr = nullptr;
    ComPtr<ID3D12Resource> ring[kRing];
    ComPtr<ID3D12CommandAllocator> alloc[kAllocs];
    ComPtr<ID3D12GraphicsCommandList> list[kAllocs];
    uint64_t allocUse[kAllocs] = {};
    ComPtr<ID3D12Fence> fence;
    HANDLE ev{};
    uint64_t fv = 0;
    std::vector<ComPtr<ID3D12Resource>> native;
    std::vector<HANDLE> handles;
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> frames{0};
    std::thread th;

    bool init(Host& host, const std::wstring& tag)
    {
        h = &host;
        D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rd.Width = W;
        rd.Height = H;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.Format = host.frameFormat();
        rd.SampleDesc = {1, 0};
        for (int i = 0; i < kRing; i++)
        {
            HANDLE sh = nullptr;
            const std::wstring nm = L"smv_dg1_" + tag + L"_ring" + std::to_wstring(i);
            if (FAILED(host.device->CreateCommittedResource(
                    &hp, D3D12_HEAP_FLAG_SHARED, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&ring[i]))) ||
                FAILED(host.device->CreateSharedHandle(ring[i].Get(), nullptr, GENERIC_ALL, nm.c_str(), &sh)))
            {
                LOG("stage1: ring texture %d not shared\n", i);
                return false;
            }
            handles.push_back(sh);
        }
        for (int i = 0; i < kAllocs; i++)
        {
            if (FAILED(host.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc[i]))) ||
                FAILED(host.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc[i].Get(), nullptr,
                                                      IID_PPV_ARGS(&list[i]))))
                return false;
            list[i]->Close();
        }
        HANDLE fh = nullptr;
        const std::wstring fn = L"smv_dg1_" + tag + L"_fence";
        if (FAILED(host.device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence))) ||
            FAILED(host.device->CreateSharedHandle(fence.Get(), nullptr, GENERIC_ALL, fn.c_str(), &fh)))
        {
            LOG("stage1: fence not shared\n");
            return false;
        }
        handles.push_back(fh);
        ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        const std::wstring mn = L"smv_dg1_" + tag + L"_hdr";
        HANDLE mh = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, sizeof(Stage1Hdr), mn.c_str());
        if (!mh)
            return false;
        handles.push_back(mh);
        hdr = (Stage1Hdr*)MapViewOfFile(mh, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(Stage1Hdr));
        if (!hdr)
            return false;
        memset((void*)hdr, 0, sizeof(Stage1Hdr));
        hdr->version = 1;
        hdr->w = W;
        hdr->h = H;
        hdr->ring = kRing;
        hdr->fmt = (uint32_t)host.frameFormat();
        hdr->allocSize = host.device->GetResourceAllocationInfo(0, 1, &rd).SizeInBytes;
        native.resize(host.nativeBufCount);
        for (UINT i = 0; i < host.nativeBufCount; i++)
            if (FAILED(host.scNative->GetBuffer(i, IID_PPV_ARGS(&native[i]))))
                return false;
        MemoryBarrier();
        hdr->magic = 0x31564D53; // 'SMV1', last: a reader waits for it
        return true;
    }

    static void barrier(ID3D12GraphicsCommandList* l, ID3D12Resource* r, D3D12_RESOURCE_STATES a,
                        D3D12_RESOURCE_STATES b)
    {
        D3D12_RESOURCE_BARRIER br{};
        br.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        br.Transition.pResource = r;
        br.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        br.Transition.StateBefore = a;
        br.Transition.StateAfter = b;
        l->ResourceBarrier(1, &br);
    }

    void run()
    {
        UINT seen = 0;
        h->scNative->GetLastPresentCount(&seen);
        uint64_t n = 0;
        while (!stop.load())
        {
            UINT c = 0, c2 = 0;
            h->scNative->GetLastPresentCount(&c);
            if (c == seen)
            {
                SwitchToThread();
                continue;
            }
            const UINT cur = h->scNative->GetCurrentBackBufferIndex();
            h->scNative->GetLastPresentCount(&c2);
            if (c2 != c)
                continue; // a present landed between the two reads: the buffer index may be stale
            LARGE_INTEGER q;
            QueryPerformanceCounter(&q);
            const uint32_t input = hdr->lastInput;
            for (UINT p = seen + 1; p <= c; p++)
            {
                const UINT idx = (cur + h->nativeBufCount - 1 - (c - p)) % h->nativeBufCount;
                const int a = (int)(n % kAllocs);
                if (fence->GetCompletedValue() < allocUse[a])
                {
                    fence->SetEventOnCompletion(allocUse[a], ev);
                    WaitForSingleObject(ev, 2000);
                }
                alloc[a]->Reset();
                ID3D12GraphicsCommandList* l = list[a].Get();
                l->Reset(alloc[a].Get(), nullptr);
                ID3D12Resource* src = native[idx].Get();
                ID3D12Resource* dst = ring[n % kRing].Get();
                barrier(l, src, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_SOURCE);
                barrier(l, dst, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
                l->CopyResource(dst, src);
                barrier(l, src, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PRESENT);
                barrier(l, dst, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
                l->Close();
                ID3D12CommandList* ls[] = {l};
                h->queue->ExecuteCommandLists(1, ls);
                h->queue->Signal(fence.Get(), ++fv);
                allocUse[a] = fv;
                const int s = (int)(n % kRing);
                hdr->slotFence[s] = fv;
                hdr->slotSeq[s] = n;
                hdr->slotQpc[s] = q.QuadPart;
                hdr->slotInput[s] = input;
                MemoryBarrier();
                hdr->frames = ++n;
                frames++;
            }
            seen = c;
        }
    }

    void start()
    {
        hdr->status = 1;
        th = std::thread([this] { run(); });
    }
    void finish(uint32_t status)
    {
        stop = true;
        if (th.joinable())
            th.join();
        if (hdr)
            hdr->status = status;
    }
    ~Stage1Ring()
    {
        finish(hdr && hdr->status == 1 ? 2 : (hdr ? hdr->status : 0));
        for (HANDLE x : handles)
            CloseHandle(x);
    }
};

// --stage1 --from TAG: another stage's ring as this stage's input, opened by name on this device (the live host's
// cascade probe does the same) and read in present order: a slot already reused is skipped to the newest
struct UpstreamRing
{
    HANDLE map = nullptr;
    const Stage1Hdr* hdr = nullptr;
    ComPtr<ID3D12Resource> ring[16];
    ComPtr<ID3D12Fence> fence;
    uint32_t n = 0;
    uint64_t nextSeq = 0, skipped = 0;

    bool open(ID3D12Device* dev, const std::wstring& tag, DXGI_FORMAT fmt)
    {
        const std::wstring mn = L"smv_dg1_" + tag + L"_hdr";
        map = OpenFileMappingW(FILE_MAP_READ, FALSE, mn.c_str());
        hdr = map ? (const Stage1Hdr*)MapViewOfFile(map, FILE_MAP_READ, 0, 0, sizeof(Stage1Hdr)) : nullptr;
        if (!hdr || hdr->magic != 0x31564D53 || hdr->version != 1 || hdr->ring > 16)
        {
            LOG("--from %ls: no stage header\n", tag.c_str());
            return false;
        }
        if (hdr->w != W || hdr->h != H || hdr->fmt != (uint32_t)fmt)
        {
            LOG("--from %ls: shares %ux%u format %u, %ux%u format %u wanted\n", tag.c_str(), hdr->w, hdr->h, hdr->fmt,
                W, H, (uint32_t)fmt);
            return false;
        }
        n = hdr->ring;
        for (uint32_t i = 0; i <= n; i++)
        {
            HANDLE sh = nullptr;
            const std::wstring nm = L"smv_dg1_" + tag + (i < n ? L"_ring" + std::to_wstring(i) : L"_fence");
            const bool ok = SUCCEEDED(dev->OpenSharedHandleByName(nm.c_str(), GENERIC_ALL, &sh)) &&
                            (i < n ? SUCCEEDED(dev->OpenSharedHandle(sh, IID_PPV_ARGS(&ring[i])))
                                   : SUCCEEDED(dev->OpenSharedHandle(sh, IID_PPV_ARGS(&fence))));
            if (sh)
                CloseHandle(sh);
            if (!ok)
            {
                LOG("--from %ls: %ls not opened\n", tag.c_str(), nm.c_str());
                return false;
            }
        }
        return true;
    }

    // the next shared present in order: its slot and fence value (false = none yet)
    bool next(uint32_t& s, uint64_t& fv)
    {
        while (hdr->frames > nextSeq)
        {
            s = (uint32_t)(nextSeq % n);
            const uint64_t seq = hdr->slotSeq[s];
            if (seq != nextSeq)
            {
                skipped++;
                nextSeq = seq > nextSeq ? seq : nextSeq + 1;
                continue;
            }
            fv = hdr->slotFence[s];
            nextSeq++;
            return true;
        }
        return false;
    }

    ~UpstreamRing()
    {
        if (hdr)
            UnmapViewOfFile(hdr);
        if (map)
            CloseHandle(map);
    }
};

// ---------------------------------------------------------------- entry

static int slInitCommon()
{
    wchar_t exePath[MAX_PATH];
    GetModuleFileNameW(nullptr, exePath, MAX_PATH);
    static std::wstring exeDir(exePath);
    exeDir.resize(exeDir.find_last_of(L'\\'));
    static const wchar_t* pluginPaths[] = {exeDir.c_str()};
    static sl::Feature features[] = {sl::kFeatureDLSS_G, sl::kFeatureReflex, sl::kFeaturePCL};

    g_verbose = GetEnvironmentVariableW(L"DLSSG_VERBOSE", nullptr, 0) != 0;
    sl::Preferences pref{};
    pref.showConsole = false;
    pref.logLevel = g_verbose ? sl::LogLevel::eVerbose : sl::LogLevel::eDefault;
    pref.pathsToPlugins = pluginPaths;
    pref.numPathsToPlugins = 1;
    pref.pathToLogsAndData = nullptr; // no log file; warnings/errors go to stderr via callback
    pref.logMessageCallback = slLog;
    pref.featuresToLoad = features;
    pref.numFeaturesToLoad = _countof(features);
    pref.applicationId = 231313132; // Streamline sample app id
    pref.flags |= sl::PreferenceFlags::eUseFrameBasedResourceTagging;
    // SL's defense against injected overlays (NVIDIA App overlay, Parsec, ...): they hook the
    // same DXGI v-tables and can silently break the interposer's present path
    pref.flags |= sl::PreferenceFlags::eUseDXGIFactoryProxy;
    // the plugins shipped beside the exe, never a newer one Streamline downloads into ProgramData: a downloaded
    // copy changes behaviour untested, and its hashed file name escapes the focus shim (it patches sl.* modules)
    pref.flags &= ~(sl::PreferenceFlags::eAllowOTA | sl::PreferenceFlags::eLoadDownloadedPlugins);
    pref.renderAPI = sl::RenderAPI::eD3D12;
    CHECK_SL(slInit(pref, sl::kSDKVersion));
    return 0;
}

static int runServer(int genFrames, bool hdr10)
{
    Host host;
    host.genFrames = genFrames;
    host.hdr10 = hdr10;
    int rc = host.init();
    if (rc)
        return rc;

    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);

    // handshake so the driving process can distinguish "up" from "unsupported"
    fprintf(stdout, "DLSSG READY gen=%d max=%u%s\n", genFrames, host.maxGen, host.hdr10 ? " hdr10=1" : "");
    fflush(stdout);

    const size_t frameBytes = (size_t)W * H * 4;
    std::vector<uint8_t> inFrame(frameBytes), prevFrame(frameBytes);
    std::vector<std::vector<uint8_t>> outFrames((size_t)genFrames, std::vector<uint8_t>(frameBytes));
    std::vector<uint8_t*> outPtrs;
    for (auto& f : outFrames)
        outPtrs.push_back(f.data());

    // wait until the pacer's native presents go quiet (used after warmup and after resyncs, so
    // leftover generated presents are never misattributed to the next pair's ledger)
    auto drainQuiet = [&]() {
        UINT stable = 0;
        int quietMs = 0;
        while (quietMs < 100)
        {
            UINT c = 0;
            host.scNative->GetLastPresentCount(&c);
            if (c == stable)
            {
                Sleep(5);
                quietMs += 5;
            }
            else
            {
                stable = c;
                quietMs = 0;
            }
        }
    };

    auto readFull = [&](uint8_t* p, size_t n) -> bool {
        size_t got = 0;
        while (got < n)
        {
            size_t r = fread(p + got, 1, n - got, stdin);
            if (r == 0)
                return false;
            got += r;
        }
        return true;
    };

    // a sampled pixel's colour bits, the alpha bits masked off: RGBA8's three bytes or R10G10B10A2's three fields
    const uint32_t rgbMask = host.hdr10 ? 0x3FFFFFFFu : 0x00FFFFFFu;
    auto rgbAt = [&](const uint8_t* f, size_t i) -> uint32_t {
        uint32_t v;
        memcpy(&v, f + i * 4, 4);
        return v & rgbMask;
    };
    // sampled RGB equality (alpha skipped): true when two frames carry identical image content.
    // Used to detect the pacer presenting the REAL frame instead of a generated one.
    auto diffSamples = [&](const uint8_t* a, const uint8_t* b) -> int {
        const size_t px = (size_t)W * H;
        int n = 0;
        for (size_t i = 0; i < px; i += 397)
            if (rgbAt(a, i) != rgbAt(b, i))
                n++;
        return n;
    };
    auto sameImage = [&](const uint8_t* a, const uint8_t* b) -> bool { return diffSamples(a, b) == 0; };
    // sampled flat-colour test: a cleared swap chain buffer reads uniform on every sample
    auto uniformImage = [&](const uint8_t* f) -> bool {
        const size_t px = (size_t)W * H;
        for (size_t i = 0; i < px; i += 397)
            if (rgbAt(f, i) != rgbAt(f, 0))
                return false;
        return true;
    };

    // DLSS-G reset: off (applied by a sacrificial present - options only take effect AT a
    // present), back on, re-warm on the previous frame, drain. This recovers the pacer after an
    // input gap (pause). It does NOT cure the machine-level "poisoned window" state in which the
    // pacer stops delivering generated frames entirely - nothing in-process does (mode toggles,
    // full swap-chain re-creation and even fresh processes were all tried and failed while a
    // window was active; they pass again once it lifts on its own after a few minutes).
    auto resetFG = [&]() -> bool {
        sl::DLSSGOptions off{};
        off.mode = sl::DLSSGMode::eOff;
        off.flags = sl::DLSSGFlags::eRetainResourcesWhenOff;
        if (slDLSSGSetOptions(host.vp, off) != sl::Result::eOk)
            return false;
        if (!host.presentFrame(prevFrame.data()))
            return false;
        sl::DLSSGOptions on{};
        on.mode = sl::DLSSGMode::eOn;
        on.numFramesToGenerate = (uint32_t)genFrames;
        if (slDLSSGSetOptions(host.vp, on) != sl::Result::eOk)
            return false;
        for (int i = 0; i < 2; i++)
            if (!host.presentFrame(prevFrame.data()))
                return false;
        drainQuiet();
        return true;
    };

    uint64_t emitted = 0;
    uint64_t resets = 0;
    bool first = true;
    bool metered = false; // sticky: hardware flip metering detected, capture by buffer sweep
    // SMV_DLSSG_SWEEP=1 latches the sweep tier at startup (hidden knob for measurement work: the
    // tier otherwise latches only when the fast tier misses its window, which no test can force).
    {
        char buf[8] = {0};
        DWORD n = GetEnvironmentVariableA("SMV_DLSSG_SWEEP", buf, (DWORD)sizeof(buf));
        if (n == 1 && buf[0] == '1')
        {
            metered = true;
            LOG("SMV_DLSSG_SWEEP=1: buffer-sweep capture latched from the start\n");
        }
    }
    std::vector<std::vector<uint32_t>> preSig(host.nativeBufCount);
    std::vector<uint8_t> sweepScratch(frameBytes);
    ULONGLONG lastPresentTick = 0;
    while (readFull(inFrame.data(), frameBytes))
    {
        if (first)
        {
            // warm-up: FG feature builds on the first present; interp(f0,f0)=f0
            for (int i = 0; i < 3; i++)
                if (!host.presentFrame(inFrame.data()))
                {
                    LOG("warmup present failed\n");
                    return 1;
                }
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
            if (!resetFG())
            {
                LOG("DLSS-G reset failed\n");
                return 1;
            }
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
        const bool inputUniform = uniformImage(inFrame.data());
        std::vector<std::vector<uint8_t>> raw;
        bool good = false;

        auto takeSig = [&](const uint8_t* f, std::vector<uint32_t>& sig) {
            sig.clear();
            const size_t px = (size_t)W * H;
            for (size_t i = 0; i < px; i += 397)
                sig.push_back(rgbAt(f, i));
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
        static const int backoffMs[] = {0, 500, 1000, 2000, 4000, 8000, 15000};
        for (int attempt = 0; attempt < (int)_countof(backoffMs) && !good; attempt++)
        {
            if (attempt > 0)
            {
                LOG("generated frames missing, resetting DLSS-G and retrying in %d ms (attempt %d)\n",
                    backoffMs[attempt], attempt);
                resets++;
                Sleep(backoffMs[attempt]);
                if (!resetFG())
                {
                    LOG("DLSS-G reset failed\n");
                    return 1;
                }
            }
            if (!host.presentFrame(inFrame.data()))
            {
                LOG("presentFrame failed\n");
                return 1;
            }
            host.pollSeen = host.presentSnapshot;
            raw.clear();
            size_t scanned = 0;
            int filled = 0;
            const ULONGLONG t0 = GetTickCount64();
            ULONGLONG nextSweep = t0 + (metered ? 50 : 350);
            while (filled < genFrames && GetTickCount64() < t0 + 1000)
            {
                int added = host.pollNewPresents(raw);
                if (added < 0)
                {
                    LOG("present readback failed\n");
                    return 1;
                }
                for (; scanned < raw.size() && filled < genFrames; scanned++)
                {
                    const uint8_t* f = raw[scanned].data();
                    if (!pairStatic && (sameImage(f, inFrame.data()) || sameImage(f, prevFrame.data())))
                        continue; // a real-frame present, not a generated frame
                    memcpy(outPtrs[filled++], f, frameBytes);
                }
                if (filled >= genFrames)
                    break;
                if (GetTickCount64() >= nextSweep)
                {
                    nextSweep = GetTickCount64() + 50;
                    // sweep every native buffer; fresh non-copies are this pair's generated frames
                    std::vector<std::vector<uint8_t>> cand;
                    std::vector<uint32_t> sig;
                    for (UINT i = 0; i < host.nativeBufCount; i++)
                    {
                        if (!host.copyNativeBuffer(i, sweepScratch.data()))
                            continue;
                        const uint8_t* f = sweepScratch.data();
                        // a flat buffer is a cleared one, never a generated frame of a picture that is not flat itself: a restarted host has no pre present signature to reject it with and once shipped two black tweens
                        if (!inputUniform && uniformImage(f))
                            continue;
                        if (pairStatic)
                        {
                            // static pair: the generated frames equal the input (measured) and a buffer that already held the input can never look fresh by signature, so any buffer equal to the input is a correct frame and a buffer holding the previous pair's tween is not
                            if (!sameImage(f, inFrame.data()))
                                continue;
                        }
                        else
                        {
                            if (sameImage(f, inFrame.data()) || sameImage(f, prevFrame.data()))
                                continue;
                            if (metered && !preSig[i].empty())
                            {
                                takeSig(f, sig);
                                if (sig == preSig[i])
                                    continue; // unchanged since before the present: stale
                            }
                        }
                        cand.push_back(sweepScratch);
                    }
                    if ((int)cand.size() >= genFrames)
                    {
                        // multi-frame order: distance to the left real frame grows with the timestep
                        std::sort(cand.begin(), cand.end(),
                                  [&](const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
                                      return diffSamples(a.data(), prevFrame.data()) <
                                             diffSamples(b.data(), prevFrame.data());
                                  });
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
                if (added == 0)
                    Sleep(1);
            }
            good = (filled == genFrames);
            if (!good)
            {
                LOG("captured %d generated frame(s) of %d (plus %zu real presents) within 1s\n", filled, genFrames,
                    raw.size() - filled);
                // post-mortem: what is in each native buffer right now?
                std::vector<uint32_t> sig;
                for (UINT i = 0; i < host.nativeBufCount; i++)
                {
                    if (!host.copyNativeBuffer(i, sweepScratch.data()))
                    {
                        LOG("  buf%u: copy failed\n", i);
                        continue;
                    }
                    const uint8_t* f = sweepScratch.data();
                    int dA = diffSamples(f, prevFrame.data());
                    int dB = diffSamples(f, inFrame.data());
                    bool stale = false;
                    if (metered && !preSig[i].empty())
                    {
                        takeSig(f, sig);
                        stale = (sig == preSig[i]);
                    }
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

// --stage1 (see the usage at the top): present every input as it arrives, share every native present
static int runStage1(int genFrames, bool hdr10, bool bgra, const std::wstring& tag, const std::wstring& from)
{
    Host host;
    host.genFrames = genFrames;
    host.hdr10 = hdr10;
    host.bgra = bgra && !hdr10;
    int rc = host.init();
    if (rc)
        return rc;
    Stage1Ring ring;
    if (!ring.init(host, tag))
        return 1;
    UpstreamRing up;
    if (!from.empty() && !up.open(host.device.Get(), from, host.frameFormat()))
        return 1;
    _setmode(_fileno(stdin), _O_BINARY);
    fprintf(stdout, "DLSSG STAGE1 READY gen=%d max=%u share=%ls%s\n", genFrames, host.maxGen, tag.c_str(),
            host.hdr10  ? " hdr10=1"
            : host.bgra ? " bgra=1"
                        : "");
    fflush(stdout);

    const size_t frameBytes = (size_t)W * H * 4;
    std::vector<uint8_t> inFrame(from.empty() ? frameBytes : 0), prevFrame(from.empty() ? frameBytes : 0);
    auto readFull = [&](uint8_t* p, size_t n) -> bool {
        size_t got = 0;
        while (got < n)
        {
            const size_t r = fread(p + got, 1, n - got, stdin);
            if (r == 0)
                return false;
            got += r;
        }
        return true;
    };
    // --from: the frames come from the upstream ring, stdin only carries the end (the host closes it); the warm-up
    // and a reset present the input presented last again (texFrame still holds it)
    static std::atomic<bool> inEnd{false}; // static: the detached reader outlives this function
    if (!from.empty())
        std::thread([] {
            char c;
            while (fread(&c, 1, 1, stdin) == 1)
            {
            }
            inEnd = true;
        }).detach();
    uint32_t upSlot = 0;
    uint64_t upValue = 0;
    auto nextInput = [&]() -> bool {
        if (from.empty())
            return readFull(inFrame.data(), frameBytes);
        while (!up.next(upSlot, upValue))
        {
            if (inEnd || up.hdr->status >= 2)
                return false;
            SwitchToThread();
        }
        return true;
    };
    auto presentInput = [&]() -> bool {
        return from.empty() ? host.presentFrame(inFrame.data())
                            : host.presentFrame(nullptr, up.ring[upSlot].Get(), up.fence.Get(), upValue);
    };
    auto presentAgain = [&]() -> bool {
        return from.empty() ? host.presentFrame(prevFrame.data()) : host.presentFrame(nullptr);
    };
    // the pacer's presents gone quiet for 100 ms (after the warm-up and a reset)
    auto drainQuiet = [&]() {
        UINT stable = 0;
        int quietMs = 0;
        while (quietMs < 100)
        {
            UINT c = 0;
            host.scNative->GetLastPresentCount(&c);
            if (c == stable)
            {
                Sleep(5);
                quietMs += 5;
            }
            else
            {
                stable = c;
                quietMs = 0;
            }
        }
    };
    // the same reset as --server's (off at a present, back on, two re-warm presents)
    auto resetFG = [&]() -> bool {
        sl::DLSSGOptions off{};
        off.mode = sl::DLSSGMode::eOff;
        off.flags = sl::DLSSGFlags::eRetainResourcesWhenOff;
        if (slDLSSGSetOptions(host.vp, off) != sl::Result::eOk || !presentAgain())
            return false;
        sl::DLSSGOptions on{};
        on.mode = sl::DLSSGMode::eOn;
        on.numFramesToGenerate = (uint32_t)genFrames;
        if (slDLSSGSetOptions(host.vp, on) != sl::Result::eOk)
            return false;
        for (int i = 0; i < 2; i++)
            if (!presentAgain())
                return false;
        drainQuiet();
        return true;
    };

    if (!nextInput())
        return 0;
    for (int i = 0; i < 3; i++)
        if (!(i && !from.empty() ? host.presentFrame(nullptr) : presentInput()))
        {
            LOG("warmup present failed\n");
            return 1;
        }
    sl::DLSSGState st{};
    if (slDLSSGGetState(host.vp, st, nullptr) == sl::Result::eOk)
    {
        if (st.status != sl::DLSSGStatus::eOk)
        {
            LOG("DLSS-G runtime status=%d, aborting\n", (int)st.status);
            return 1;
        }
        if (st.numFramesToGenerateMax && (uint32_t)genFrames > st.numFramesToGenerateMax)
        {
            LOG("multi-frame generation beyond %ux is not supported on this GPU (requested %ux)\n",
                st.numFramesToGenerateMax + 1, genFrames + 1);
            return 3;
        }
    }
    drainQuiet();
    prevFrame.swap(inFrame);
    ring.start();

    uint64_t inputs = 1, resets = 0, statInputs = 0, statFrames = ring.frames;
    ULONGLONG lastPresentTick = GetTickCount64(), statTick = lastPresentTick;
    while (nextInput())
    {
        // an input gap past 700 ms poisons the pacer: reset before presenting (the live host's own value; the capture
        // repeats a static picture every 500 ms, which must not reset stage 1 each time; --server keeps 500)
        if (GetTickCount64() - lastPresentTick > 700)
        {
            LOG("input gap detected, resetting DLSS-G\n");
            resets++;
            if (!resetFG())
            {
                LOG("DLSS-G reset failed\n");
                ring.finish(3);
                return 1;
            }
        }
        if (!presentInput())
        {
            LOG("presentFrame failed\n");
            ring.finish(3);
            return 1;
        }
        lastPresentTick = GetTickCount64();
        ring.hdr->lastInput = (uint32_t)inputs++;
        prevFrame.swap(inFrame);
        if (lastPresentTick - statTick >= 5000)
        {
            const double s = (lastPresentTick - statTick) / 1000.0;
            const uint64_t f = ring.frames, in = inputs - statInputs;
            LOG("stage1: %.1f inputs/s, %.1f shared presents/s (%.2f an input)%s\n", in / s, (f - statFrames) / s,
                in ? (double)(f - statFrames) / in : 0.0,
                from.empty() ? "" : (", " + std::to_string(up.skipped) + " upstream frames skipped").c_str());
            statTick = lastPresentTick;
            statInputs = inputs;
            statFrames = f;
        }
    }
    ring.finish(2);
    if (resets)
        LOG("note: %llu DLSS-G pacer resets during this run\n", (unsigned long long)resets);
    LOG("stage1 done, %llu inputs, %llu shared presents\n", (unsigned long long)inputs,
        (unsigned long long)ring.frames.load());
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
        if (W < 128 || H < 128 || W > 8192 || H > 8192)
        {
            LOG("bad size %ux%u\n", W, H);
            return 1;
        }
        int genFrames = 1;
        bool hdr10 = false;
        for (int i = 4; i < argc; i++)
        {
            if (wcscmp(argv[i], L"--gen") == 0 && i + 1 < argc)
                genFrames = _wtoi(argv[++i]);
            else if (wcscmp(argv[i], L"--hdr10") == 0)
                hdr10 = true;
        }
        if (genFrames < 1 || genFrames > 5)
        {
            LOG("--gen must be 1..5 (got %d)\n", genFrames);
            return 1;
        }
        if (slInitCommon())
            return 1;
        return runServer(genFrames, hdr10);
    }
    if (argc >= 4 && wcscmp(argv[1], L"--stage1") == 0)
    {
        W = (uint32_t)_wtoi(argv[2]);
        H = (uint32_t)_wtoi(argv[3]);
        if (W < 128 || H < 128 || W > 8192 || H > 8192)
        {
            LOG("bad size %ux%u\n", W, H);
            return 1;
        }
        int genFrames = 1;
        bool hdr10 = false, bgra = false;
        std::wstring tag, from;
        for (int i = 4; i < argc; i++)
        {
            if (wcscmp(argv[i], L"--gen") == 0 && i + 1 < argc)
                genFrames = _wtoi(argv[++i]);
            else if (wcscmp(argv[i], L"--hdr10") == 0)
                hdr10 = true;
            else if (wcscmp(argv[i], L"--bgra") == 0)
                bgra = true;
            else if (wcscmp(argv[i], L"--share") == 0 && i + 1 < argc)
                tag = argv[++i];
            else if (wcscmp(argv[i], L"--from") == 0 && i + 1 < argc)
                from = argv[++i];
        }
        if (genFrames < 1 || genFrames > 5 || tag.empty() || tag.size() > 40 || from.size() > 40 || from == tag)
        {
            LOG("--stage1 needs --gen 1..5 (got %d), --share TAG (1..40 chars) and an optional --from TAG of another "
                "stage\n",
                genFrames);
            return 1;
        }
        if (slInitCommon())
            return 1;
        return runStage1(genFrames, hdr10, bgra, tag, from);
    }

    LOG("usage: dlssg2f.exe --server W H [--gen N] [--hdr10]\n"
        "       dlssg2f.exe --stage1 W H [--gen N] [--hdr10 | --bgra] --share TAG [--from TAG]\n");
    return 1;
}
