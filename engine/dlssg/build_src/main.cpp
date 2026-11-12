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
//   dlssg2f.exe --server W H [--gen N] [--hdr10] [--cut-event HANDLE]
//       streaming: raw RGBA8 frames (W*H*4) on stdin; for every frame after the
//       first, the N (default 1, max 5) DLSS-G frames generated between it and
//       its predecessor - evenly spaced, in temporal order - are written to
//       stdout as raw RGBA8 (multi-frame generation: N=1 is 2x, N=5 is 6x).
//       --cut-event: an inherited auto-reset event (its handle value in decimal). Set
//       before a frame is sent, it makes that frame the first of a new clip: no
//       frames come back for it and the tweens after it are a fresh server's. The
//       handshake then ends in " cut=1".
//       --hdr10: HDR10 frames both ways instead (PQ BT.2020 as R10G10B10A2 words,
//       still W*H*4 bytes), presented the way the DLSS-G guide asks for HDR
//       (section 11.0): an R10G10B10A2 swap chain with the G2084 / P2020 colour
//       space. The handshake then ends in " hdr10=1"; a swap chain that refuses
//       that colour space exits 5 before the handshake.
//       All logging goes to stderr; stdout is binary-only. EOF on stdin exits.
//       This is the SmoothMyVideo backend.
//   dlssg2f.exe --stage1 W H [--gen N] [--hdr10 | --bgra] --share TAG [--from TAG]
//       a stage of a live cascade before the host's own: --from = another stage's ring as its input (on the GPU,
//       in order; stdin then only ends it), else the same frames on stdin (--bgra: SDR frames in B8G8R8A8 order, the
//       order of Live's effects route). An input's generated frames and then the input itself go into a ring of
//       textures shared by name with the process that reads them (smv_dg1_TAG_ring0.., a shared fence
//       smv_dg1_TAG_fence, the header Stage1Hdr in the file mapping smv_dg1_TAG_hdr), spread over the source's own
//       interval (the cadence DLSS-G's pacer gives a game). NGX's own calls make them (runStage1Direct, the
//       handshake ends in " ngx=1"); on Streamline's route (runStage1) each input is presented the moment it
//       arrives on a swap chain in the frames' order, DLSS-G's pacer spreads the presents, and every native present
//       is copied into the ring. stdout carries only the handshake line.

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
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
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
#include "pan_vectors.h"
#include "ngx_direct.h"

// milliseconds on the performance counter (GetTickCount64 steps in ~16 ms)
static double qpcMs()
{
    static LARGE_INTEGER f{};
    if (!f.QuadPart)
        QueryPerformanceFrequency(&f);
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)f.QuadPart;
}

// The native swap chain's Present / Present1 wrapped (its vtable, this process only): when a call returns, its present
// is queued, so a readback queued after it reads that present's frame. g_presentsQueued = the present count the last
// returned call reached; the capture copies presents up to it.
static std::atomic<UINT> g_presentsQueued{0};
static std::atomic<int> g_presentsInFlight{0};
using PresentFn = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain*, UINT, UINT);
using Present1Fn = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain1*, UINT, UINT, const DXGI_PRESENT_PARAMETERS*);
static PresentFn g_present = nullptr;
static Present1Fn g_present1 = nullptr;

// after a returned call: the count it reached, kept only when no other Present call was in flight (another thread's
// present may be counted but not yet queued)
static void notePresentQueued(IDXGISwapChain* sc)
{
    UINT c = 0;
    if (FAILED(sc->GetLastPresentCount(&c)) || g_presentsInFlight.load() != 1)
        return;
    UINT o = g_presentsQueued.load();
    while (c > o && !g_presentsQueued.compare_exchange_weak(o, c))
    {
    }
}

static HRESULT STDMETHODCALLTYPE presentQueued(IDXGISwapChain* sc, UINT sync, UINT flags)
{
    g_presentsInFlight++;
    const HRESULT hr = g_present(sc, sync, flags);
    notePresentQueued(sc);
    g_presentsInFlight--;
    return hr;
}

static HRESULT STDMETHODCALLTYPE present1Queued(IDXGISwapChain1* sc, UINT sync, UINT flags,
                                                const DXGI_PRESENT_PARAMETERS* p)
{
    g_presentsInFlight++;
    const HRESULT hr = g_present1(sc, sync, flags, p);
    notePresentQueued(sc);
    g_presentsInFlight--;
    return hr;
}

// The real factory's CreateSwapChain / CreateSwapChainForHwnd wrapped (its vtable, this process only): Streamline creates
// the native swap chain on a queue of its own, and only a copy on THAT queue runs behind a present.
// g_nativeScQueue = the device / queue argument of the last real swap chain creation.
static std::atomic<IUnknown*> g_nativeScQueue{nullptr};
using CreateScFn = HRESULT(STDMETHODCALLTYPE*)(IDXGIFactory*, IUnknown*, DXGI_SWAP_CHAIN_DESC*, IDXGISwapChain**);
using CreateScHwndFn = HRESULT(STDMETHODCALLTYPE*)(IDXGIFactory2*, IUnknown*, HWND, const DXGI_SWAP_CHAIN_DESC1*,
                                                   const DXGI_SWAP_CHAIN_FULLSCREEN_DESC*, IDXGIOutput*,
                                                   IDXGISwapChain1**);
static CreateScFn g_createSc = nullptr;
static CreateScHwndFn g_createScHwnd = nullptr;

static HRESULT STDMETHODCALLTYPE createScSeen(IDXGIFactory* f, IUnknown* dev, DXGI_SWAP_CHAIN_DESC* d,
                                              IDXGISwapChain** out)
{
    g_nativeScQueue = dev;
    return g_createSc(f, dev, d, out);
}

static HRESULT STDMETHODCALLTYPE createScHwndSeen(IDXGIFactory2* f, IUnknown* dev, HWND w,
                                                  const DXGI_SWAP_CHAIN_DESC1* d,
                                                  const DXGI_SWAP_CHAIN_FULLSCREEN_DESC* fd, IDXGIOutput* o,
                                                  IDXGISwapChain1** out)
{
    g_nativeScQueue = dev;
    return g_createScHwnd(f, dev, w, d, fd, o, out);
}

// vtable slots: IDXGIFactory::CreateSwapChain = 10, IDXGIFactory2::CreateSwapChainForHwnd = 15
static bool wrapRealFactory(IDXGIFactory2* proxy)
{
    ComPtr<IUnknown> real;
    if (slGetNativeInterface(proxy, (void**)real.GetAddressOf()) != sl::Result::eOk || !real)
        return false;
    void** vt = *(void***)real.Get();
    DWORD old = 0;
    if (!VirtualProtect(&vt[10], sizeof(void*) * 6, PAGE_READWRITE, &old))
        return false;
    if (vt[10] != (void*)&createScSeen)
    {
        g_createSc = (CreateScFn)vt[10];
        vt[10] = (void*)&createScSeen;
    }
    if (vt[15] != (void*)&createScHwndSeen)
    {
        g_createScHwnd = (CreateScHwndFn)vt[15];
        vt[15] = (void*)&createScHwndSeen;
    }
    VirtualProtect(&vt[10], sizeof(void*) * 6, old, &old);
    FlushInstructionCache(GetCurrentProcess(), nullptr, 0);
    return true;
}

// vtable slots: IDXGISwapChain::Present = 8, IDXGISwapChain1::Present1 = 22
static bool wrapNativePresent(IDXGISwapChain1* sc)
{
    void** vt = *(void***)sc;
    DWORD old = 0;
    if (!VirtualProtect(&vt[8], sizeof(void*) * 15, PAGE_READWRITE, &old))
        return false;
    if (vt[8] != (void*)&presentQueued)
    {
        g_present = (PresentFn)vt[8];
        vt[8] = (void*)&presentQueued;
    }
    if (vt[22] != (void*)&present1Queued)
    {
        g_present1 = (Present1Fn)vt[22];
        vt[22] = (void*)&present1Queued;
    }
    VirtualProtect(&vt[8], sizeof(void*) * 15, old, &old);
    FlushInstructionCache(GetCurrentProcess(), nullptr, 0);
    return true;
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
    bool panOn = false;    // the pan as motion vectors (pan_vectors.h), set once its writer is up
    panvec::Estimator pan; // measures each uploaded frame against the one presented before
    panvec::Writer panTex; // writes texMvec
    ComPtr<ID3D12Resource> uploadBuf;
    ComPtr<ID3D12Resource> readbackBuf;
    UINT rowPitch = 0; // 256-aligned pitch shared by upload + readback
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12CommandQueue> rbQueue; // the readback's queue: the one the native swap chain presents on
    ComPtr<ID3D12CommandAllocator> rbAlloc;
    ComPtr<ID3D12GraphicsCommandList> rbList;
    ComPtr<ID3D12Fence> fence;
    uint64_t fenceValue = 0;
    HANDLE fenceEvent{};
    sl::ViewportHandle vp{0u};
    sl::Constants consts{};
    uint32_t frameIndex = 1;
    int waitMs = 5;           // the longest the newest native present waits for a later one before its readback
    int genFrames = 1;        // generated frames per source frame (numFramesToGenerate)
    UINT syncInterval = 1;    // Present sync interval: vsync-paced FG, measured far more reliable for the capture
                              // (the server route); a live stage presents free-running at 0 (runStage1)
    bool tearPresent = true;  // sync-0 presents pass DXGI_PRESENT_ALLOW_TEARING: without it DXGI throttles the flip
                              // queue to the refresh; cleared when DXGI rejects the flag
    uint32_t maxGen = 0;      // device limit reported by the SDK (0 = not yet known)
    UINT presentSnapshot = 0; // native present count taken just before each proxy Present
    bool hdr10 = false;       // --hdr10: R10G10B10A2 frames on a G2084 / P2020 swap chain
    bool bgra = false;        // --bgra (--stage1, SDR): B8G8R8A8 frames
    bool hdr10Refused = false;

    DXGI_FORMAT frameFormat() const
    {
        return hdr10 ? DXGI_FORMAT_R10G10B10A2_UNORM : bgra ? DXGI_FORMAT_B8G8R8A8_UNORM : DXGI_FORMAT_R8G8B8A8_UNORM;
    }

    bool waitQueue(ID3D12CommandQueue* q = nullptr)
    {
        (q ? q : queue.Get())->Signal(fence.Get(), ++fenceValue);
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
        g_nativeScQueue = nullptr;
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
        if (nativeCapture)
        {
            // the readback runs on the queue the native chain presents on (a copy there queues behind each present);
            // SMV_DLSSG_RB_QUEUE=own, or no such queue seen = this host's queue with the settle before each readback
            ComPtr<ID3D12CommandQueue> nq;
            IUnknown* seen = g_nativeScQueue.exchange(nullptr);
            wchar_t rq[8] = {};
            GetEnvironmentVariableW(L"SMV_DLSSG_RB_QUEUE", rq, 8);
            rbOwn = _wcsicmp(rq, L"own") == 0 || !seen || FAILED(seen->QueryInterface(IID_PPV_ARGS(&nq)));
            if (rbOwn)
                nq = queue;
            if (nq != rbQueue)
            {
                const D3D12_COMMAND_LIST_TYPE t = nq->GetDesc().Type;
                rbAlloc.Reset();
                rbList.Reset();
                if (FAILED(device->CreateCommandAllocator(t, IID_PPV_ARGS(&rbAlloc))) ||
                    FAILED(device->CreateCommandList(0, t, rbAlloc.Get(), nullptr, IID_PPV_ARGS(&rbList))) ||
                    FAILED(rbList->Close()))
                    return false;
                rbQueue = nq;
            }
            if (logIt)
                LOG("readback queue: %s\n",
                    rbOwn ? "this host's, a settle before each readback" : "the native swap chain's");
        }
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
        // SMV_DLSSG_PRESENT_WRAP=0: no wrap (the capture then waits for a later present, or waitMs on the newest)
        if (nativeCapture && !rbOwn)
        {
            wchar_t pw[4] = {};
            GetEnvironmentVariableW(L"SMV_DLSSG_PRESENT_WRAP", pw, 4);
            presentWrapped = wcscmp(pw, L"0") != 0 && wrapNativePresent(scNative.Get());
            if (logIt)
                LOG("native Present wrapped: %s\n", presentWrapped ? "yes" : "no");
        }
        return true;
    }

    int init()
    {
        // window: swap chain needs an HWND, but with native readback it never has to be visible on the desktop, so
        // park it far offscreen. Created without WS_VISIBLE and shown SW_SHOWNOACTIVATE (plus WS_EX_NOACTIVATE), as the
        // live overlay is: a popup created visible activates itself, and from a render the app started that pulls
        // the foreground off the app or the game; Streamline's focus test is answered by the focus shim
        WNDCLASSW wc{};
        wc.lpfnWndProc = wndProc;
        wc.hInstance = GetModuleHandleW(nullptr);
        wc.lpszClassName = L"dlssg2f";
        wc.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));
        RegisterClassW(&wc);
        hwnd = CreateWindowExW(WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE, L"dlssg2f", L"dlssg2f", WS_POPUP, -32000, -32000, W,
                               H, nullptr, nullptr, wc.hInstance, nullptr);
        if (!hwnd)
        {
            LOG("CreateWindow failed\n");
            return 1;
        }
        ShowWindow(hwnd, SW_SHOWNOACTIVATE);
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
        if (nativeCapture && !wrapRealFactory(factory.Get()))
            LOG("the real DXGI factory could not be wrapped: reading frames back on this host's queue\n");
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
        go.mvecDepthWidth = panvec::vecSize(W);
        go.mvecDepthHeight = panvec::vecSize(H);
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
        const uint32_t vw = panvec::vecSize(W), vh = panvec::vecSize(H);
        if (!makeTex(frameFormat(), texFrame) ||
            !makeTex(DXGI_FORMAT_R32_FLOAT, texDepth, D3D12_RESOURCE_FLAG_NONE, vw, vh) ||
            !makeTex(DXGI_FORMAT_R16G16_FLOAT, texMvec, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, vw, vh) ||
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
            sl_.PlacedFootprint.Footprint = {DXGI_FORMAT_R32_FLOAT, vw, vh, 1, rowPitch};
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
        panOn = panTex.init(device.Get(), texMvec.Get(), W, H);
        if (!panOn)
            LOG("pan vectors: the descriptor heaps did not allocate, zero vectors\n");

        fillConstants(consts);
        return 0;
    }

    // the run's pan vector summary
    void logPan() const
    {
        if (!panOn || !pan.frames)
            return;
        LOG("pan vectors: %llu of %llu frames moved (mean %.1f px, %.1f %% of their area still), %.2f ms a frame "
            "(max %.2f)\n",
            (unsigned long long)pan.tagged, (unsigned long long)pan.frames,
            pan.tagged ? pan.sumLen / (double)pan.tagged : 0.0,
            pan.tagged ? 100.0 * pan.sumStill / (double)pan.tagged : 0.0, pan.sumMs / (double)pan.frames, pan.maxMs);
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
        // an uploaded frame is measured; a ring texture or the last frame again keeps zero vectors
        const panvec::Result panR = panOn && rgba ? pan.run(rgba, W, H, (size_t)W * 4, hdr10) : panvec::Result{};

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
        if (panOn)
            panTex.record(list.Get(), texMvec.Get(), panR);
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
        // NOTE: do NOT tag kBufferTypeBackbuffer here, even with a null resource "just to silence
        // the extent warning": the tag flips sl.dlss_g into its subrect present path and the real
        // frame ends up in the native swap chain right behind the generated one, so the readback
        // returns the REAL frame (tween == right endpoint, verified). The extent
        // warning it would silence is benign and filtered in slLog instead.
        sl::ResourceTag tags[] = {
            sl::ResourceTag(&depthRes, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilPresent, &vecExtent),
            sl::ResourceTag(&mvecRes, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eValidUntilPresent,
                            &vecExtent),
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
        HRESULT hr = sc->Present(syncInterval, syncInterval == 0 && tearPresent ? DXGI_PRESENT_ALLOW_TEARING : 0);
        if (hr == DXGI_ERROR_INVALID_CALL && syncInterval == 0 && tearPresent)
        {
            tearPresent = false;
            LOG("present: ALLOW_TEARING rejected, plain sync-0 presents\n");
            hr = sc->Present(0, 0);
        }
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
        if (!rbList || FAILED(scNative->GetBuffer(idx, IID_PPV_ARGS(&buf))))
            return false;

        rbAlloc->Reset();
        rbList->Reset(rbAlloc.Get(), nullptr);
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = buf.Get();
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
        b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        rbList->ResourceBarrier(1, &b);
        D3D12_TEXTURE_COPY_LOCATION src{buf.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        D3D12_TEXTURE_COPY_LOCATION dst{readbackBuf.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        dst.PlacedFootprint.Footprint = {frameFormat(), W, H, 1, rowPitch};
        rbList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
        b.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
        rbList->ResourceBarrier(1, &b);
        rbList->Close();
        ID3D12CommandList* ls[] = {rbList.Get()};
        rbQueue->ExecuteCommandLists(1, ls);
        if (!waitQueue(rbQueue.Get()))
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
    // the last poll (6 native buffers vs at most 5 generated frames means nothing is overwritten
    // before we drain it). The caller decides WHICH captured presents are generated frames by
    // content - present order cannot be trusted: a real-frame present sometimes slips into the
    // stream despite eShowOnlyInterpolatedFrame, and on some runs it even leads the generated one.
    // The copies run on rbQueue, the queue the native chain presents on (Streamline's own, not this
    // host's: a copy on another queue raced the presents and read buffers' previous frames). A copy
    // there runs behind a present once that present is queued: its Present call returned
    // (g_presentsQueued, the wrapped native Present), else a later present shows (the pacer presents
    // in order from one thread) or, for the newest, waitMs passed (a pair's last present has no later one).
    // On this host's queue (rbOwn) no rule orders a copy: a waitMs settle before each poll's copies.
    bool nativeCapture = false;  // --server / --stage1: the readback queue, the wraps and the capture rules above
    bool rbOwn = true;           // the readback runs on this host's queue
    UINT pollSeen = 0;           // reset to presentSnapshot after each proxy Present
    UINT ringOff = 0;            // present p sits in native buffer (p + ringOff) % nativeBufCount (flip model)
    UINT newest = 0;             // the newest present count seen
    double newestMs = 0;         // when it was first seen
    bool presentWrapped = false; // the native Present / Present1 wrapped (wrapNativePresent)
    bool queuedKnown = false;    // the wrap sees every native present (checked at each quiet point)

    // copy presents pollSeen + 1 .. last into raw[n..], present p from native buffer bufOf(p)
    template <class BufOf> int copyPresents(std::vector<std::vector<uint8_t>>& raw, size_t& n, UINT last, BufOf bufOf)
    {
        int added = 0;
        for (UINT p = pollSeen + 1; p <= last; p++)
        {
            if (raw.size() <= n)
                raw.emplace_back((size_t)W * H * 4);
            if (!copyNativeBuffer(bufOf(p), raw[n].data()))
                return -1;
            n++;
            added++;
        }
        pollSeen = last;
        return added;
    }

    // the buffer ring's offset, read while no present is in flight (count and index then agree); a wrap that
    // reached the same count sees the presents
    void syncRing()
    {
        UINT c = 0;
        scNative->GetLastPresentCount(&c);
        const UINT cur = scNative->GetCurrentBackBufferIndex();
        ringOff = (cur + 2 * nativeBufCount - 1 - c % nativeBufCount) % nativeBufCount;
        const bool known = presentWrapped && g_presentsQueued.load() == c;
        if (known != queuedKnown)
            LOG("present capture: %s\n", known ? "copies each present once its Present call returned"
                                               : "the wrapped Present misses presents, waits for a later present");
        queuedKnown = known;
    }

    // raw keeps its buffers from pair to pair; n = how many of them hold the presents captured since the caller reset it
    int pollNewPresents(std::vector<std::vector<uint8_t>>& raw, size_t& n)
    {
        UINT c = 0;
        scNative->GetLastPresentCount(&c);
        if (c == pollSeen)
            return 0;
        if (rbOwn)
        {
            Sleep(waitMs);                     // settle: let the newest present's GPU write finish
            scNative->GetLastPresentCount(&c); // re-snapshot after the settle (more may have landed)
            const UINT cur = scNative->GetCurrentBackBufferIndex();
            return copyPresents(raw, n, c,
                                [&](UINT p) { return (cur + nativeBufCount - 1 - (c - p)) % nativeBufCount; });
        }
        // the ring must still agree with the count: a present that turns the buffers without being counted (hardware
        // flip metering) breaks the mapping, and the buffer sweep takes over
        const UINT cur = scNative->GetCurrentBackBufferIndex();
        UINT c2 = 0;
        scNative->GetLastPresentCount(&c2);
        if (c2 != c || (cur + 2 * nativeBufCount - 1 - c % nativeBufCount) % nativeBufCount != ringOff)
            return 0;
        const double now = qpcMs();
        if (c != newest)
        {
            newest = c;
            newestMs = now;
        }
        UINT last = now - newestMs >= waitMs ? c : c - 1;
        if (queuedKnown)
            last = (std::max)(last, (std::min)(c, g_presentsQueued.load()));
        if (last <= pollSeen)
            return 0;
        return copyPresents(raw, n, last, [&](UINT p) { return (p + ringOff) % nativeBufCount; });
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
// slotSeq = the present's number since the ring started (a reader that finds a newer number lost that slot). Version 2
// adds wantGen / gen: the live host asks a running stage for another multiple, applied in place at its next present;
// version 3 adds the ask kGenOff = 1x (DLSS-G off, each input passed through as it is; gen 0)
static const uint32_t kGenOff = 0xFFFFFFFFu;
struct Stage1Hdr
{
    uint32_t magic;   // 'SMV1'
    uint32_t version; // 3
    volatile uint64_t frames;
    uint32_t w, h, ring, fmt;
    uint64_t allocSize;
    volatile uint32_t status;    // 0 starting, 1 running, 2 ended, 3 failed
    volatile uint32_t lastInput; // the source frame presented last (frame 0 = the warm-up)
    volatile uint64_t slotFence[16];
    volatile uint64_t slotSeq[16];
    volatile int64_t slotQpc[16];    // QueryPerformanceCounter when the capture saw the present
    volatile uint32_t slotInput[16]; // lastInput at that moment
    volatile uint32_t wantGen;       // the host's ask: numFramesToGenerate from the next present (0 = none)
    volatile uint32_t gen;           // the stage's numFramesToGenerate now
};

// Every native present copied into a ring of shared textures on rbQueue, the queue the native chain presents on
// (Streamline's own): a copy there runs behind present p once p is queued (its Present call returned, the wrap, or a
// later present shows), so it needs no settle; on this host's queue (rbOwn) nothing orders it and each present is
// copied as soon as it is counted. The rings rest in COMMON, the state another device's queue may assume
struct Stage1Ring
{
    static const int kRing = 12, kAllocs = 16;
    Host* h = nullptr;
    Stage1Hdr* hdr = nullptr;
    int slots = kRing; // the ring's textures (the header's arrays hold 16)
    ComPtr<ID3D12Resource> ring[16];
    ComPtr<ID3D12CommandAllocator> alloc[kAllocs];
    ComPtr<ID3D12GraphicsCommandList> list[kAllocs];
    uint64_t allocUse[kAllocs] = {};
    ComPtr<ID3D12Fence> fence;
    HANDLE ev{};
    uint64_t fv = 0;
    uint64_t count = 0; // frames shared so far: the next one's number
    std::vector<ComPtr<ID3D12Resource>> native;
    std::vector<HANDLE> handles;
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> frames{0};
    std::thread th;
    HANDLE wake = nullptr; // the named event a publish sets

    // the ring of `ringSlots` textures, its fence and the header on dev, each shared by name (smv_dg1_TAG_...);
    // ready() then lets the readers in
    bool share(ID3D12Device* dev, DXGI_FORMAT fmt, int gen, int ringSlots, const std::wstring& tag)
    {
        slots = ringSlots;
        D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rd.Width = W;
        rd.Height = H;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.Format = fmt;
        rd.SampleDesc = {1, 0};
        for (int i = 0; i < slots; i++)
        {
            HANDLE sh = nullptr;
            const std::wstring nm = L"smv_dg1_" + tag + L"_ring" + std::to_wstring(i);
            if (FAILED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED, &rd, D3D12_RESOURCE_STATE_COMMON,
                                                    nullptr, IID_PPV_ARGS(&ring[i]))) ||
                FAILED(dev->CreateSharedHandle(ring[i].Get(), nullptr, GENERIC_ALL, nm.c_str(), &sh)))
            {
                LOG("stage1: ring texture %d not shared\n", i);
                return false;
            }
            handles.push_back(sh);
        }
        HANDLE fh = nullptr;
        const std::wstring fn = L"smv_dg1_" + tag + L"_fence";
        if (FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence))) ||
            FAILED(dev->CreateSharedHandle(fence.Get(), nullptr, GENERIC_ALL, fn.c_str(), &fh)))
        {
            LOG("stage1: fence not shared\n");
            return false;
        }
        handles.push_back(fh);
        ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        // set at every publish: a stage that takes this ring as its input waits on it (smv_dg1_TAG_wake)
        wake = CreateEventW(nullptr, FALSE, FALSE, (L"smv_dg1_" + tag + L"_wake").c_str());
        if (wake)
            handles.push_back(wake);
        const std::wstring mn = L"smv_dg1_" + tag + L"_hdr";
        HANDLE mh = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, sizeof(Stage1Hdr), mn.c_str());
        if (!mh)
            return false;
        handles.push_back(mh);
        hdr = (Stage1Hdr*)MapViewOfFile(mh, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(Stage1Hdr));
        if (!hdr)
            return false;
        memset((void*)hdr, 0, sizeof(Stage1Hdr));
        hdr->version = 3;
        hdr->gen = (uint32_t)gen;
        hdr->w = W;
        hdr->h = H;
        hdr->ring = (uint32_t)slots;
        hdr->fmt = (uint32_t)fmt;
        hdr->allocSize = dev->GetResourceAllocationInfo(0, 1, &rd).SizeInBytes;
        return true;
    }

    // the header's magic, written last: a reader waits for it
    void ready()
    {
        MemoryBarrier();
        hdr->magic = 0x31564D53; // 'SMV1'
    }

    // the ring's next frame is in its slot (number % slots) once the fence reaches value; qpc = when it was seen
    // (Streamline's stage) or shared (NGX's), input = the source frame it belongs to
    void publish(uint64_t value, int64_t qpc, uint32_t input)
    {
        const int s = (int)(count % slots);
        hdr->slotFence[s] = value;
        hdr->slotSeq[s] = count;
        hdr->slotQpc[s] = qpc;
        hdr->slotInput[s] = input;
        MemoryBarrier();
        hdr->frames = ++count;
        frames++;
        if (wake)
            SetEvent(wake);
    }

    bool init(Host& host, const std::wstring& tag)
    {
        h = &host;
        if (!share(host.device.Get(), host.frameFormat(), host.genFrames, kRing, tag))
            return false;
        const D3D12_COMMAND_LIST_TYPE t = host.rbQueue->GetDesc().Type;
        for (int i = 0; i < kAllocs; i++)
        {
            if (FAILED(host.device->CreateCommandAllocator(t, IID_PPV_ARGS(&alloc[i]))) ||
                FAILED(host.device->CreateCommandList(0, t, alloc[i].Get(), nullptr, IID_PPV_ARGS(&list[i]))))
                return false;
            list[i]->Close();
        }
        native.resize(host.nativeBufCount);
        for (UINT i = 0; i < host.nativeBufCount; i++)
            if (FAILED(host.scNative->GetBuffer(i, IID_PPV_ARGS(&native[i]))))
                return false;
        ready();
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
        UINT copied = seen;
        int64_t seenQpc[kRing] = {};    // when the capture first counted present p (slot p % kRing)
        uint32_t seenInput[kRing] = {}; // lastInput at that moment
        ID3D12CommandQueue* q = h->rbQueue.Get();
        while (!stop.load())
        {
            UINT c = 0, c2 = 0;
            h->scNative->GetLastPresentCount(&c);
            if (c != seen)
            {
                LARGE_INTEGER t;
                QueryPerformanceCounter(&t);
                const uint32_t input = hdr->lastInput;
                for (UINT p = seen + 1; p <= c; p++)
                {
                    seenQpc[p % kRing] = t.QuadPart;
                    seenInput[p % kRing] = input;
                }
                seen = c;
            }
            UINT last = c;
            if (!h->rbOwn)
            {
                last = c ? c - 1 : 0;
                if (h->presentWrapped)
                    last = (std::max)(last, (std::min)(c, g_presentsQueued.load()));
            }
            if (last <= copied)
            {
                SwitchToThread();
                continue;
            }
            const UINT cur = h->scNative->GetCurrentBackBufferIndex();
            h->scNative->GetLastPresentCount(&c2);
            if (c2 != c)
                continue; // a present landed between the two reads: the buffer index may be stale
            for (UINT p = copied + 1; p <= last; p++)
            {
                const UINT idx = (cur + h->nativeBufCount - 1 - (c - p)) % h->nativeBufCount;
                const int a = (int)(count % kAllocs);
                if (fence->GetCompletedValue() < allocUse[a])
                {
                    fence->SetEventOnCompletion(allocUse[a], ev);
                    WaitForSingleObject(ev, 2000);
                }
                alloc[a]->Reset();
                ID3D12GraphicsCommandList* l = list[a].Get();
                l->Reset(alloc[a].Get(), nullptr);
                ID3D12Resource* src = native[idx].Get();
                ID3D12Resource* dst = ring[count % slots].Get();
                barrier(l, src, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_SOURCE);
                barrier(l, dst, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
                l->CopyResource(dst, src);
                barrier(l, src, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PRESENT);
                barrier(l, dst, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
                l->Close();
                ID3D12CommandList* ls[] = {l};
                q->ExecuteCommandLists(1, ls);
                q->Signal(fence.Get(), ++fv);
                allocUse[a] = fv;
                publish(fv, seenQpc[p % kRing], seenInput[p % kRing]);
            }
            copied = last;
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
    HANDLE wake = nullptr; // the upstream stage's publish event; null = it has none, its ring is polled

    bool open(ID3D12Device* dev, const std::wstring& tag, DXGI_FORMAT fmt)
    {
        // the upstream stage may still be starting (the live host starts every stage at once): its header's magic,
        // written last by Stage1Ring::init, is waited for up to 15 s
        const std::wstring mn = L"smv_dg1_" + tag + L"_hdr";
        const ULONGLONG until = GetTickCount64() + 15000;
        for (;;)
        {
            map = OpenFileMappingW(FILE_MAP_READ, FALSE, mn.c_str());
            hdr = map ? (const Stage1Hdr*)MapViewOfFile(map, FILE_MAP_READ, 0, 0, sizeof(Stage1Hdr)) : nullptr;
            if ((hdr && hdr->magic == 0x31564D53) || GetTickCount64() > until)
                break;
            if (hdr)
                UnmapViewOfFile(hdr);
            if (map)
                CloseHandle(map);
            hdr = nullptr;
            map = nullptr;
            Sleep(20);
        }
        if (!hdr || hdr->magic != 0x31564D53 || hdr->version < 1 || hdr->version > 3 || hdr->ring > 16)
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
        wake = OpenEventW(SYNCHRONIZE, FALSE, (L"smv_dg1_" + tag + L"_wake").c_str());
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
        if (wake)
            CloseHandle(wake);
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

// --server's frame buffers: the server thread's set (raw = the captured presents, outFrames = the sweep tier's copies)
// and the pair writer's set; the two trade places at every hand-over (a vector swap keeps every frame's address)
struct ServerFrames
{
    std::vector<std::vector<uint8_t>> raw, rawW, outFrames, outFramesW;
};

// --server's pair writer: one thread streams a pair's generated frames to stdout while the server thread reads the
// next input frame and presents it, so the pipe transfer leaves the present-to-present interval DLSS-G paces over.
// post() hands over the frame pointers of a buffer set the server thread no longer touches; it waits for the previous
// pair to be written first, so the frames leave in order. A failed write is logged once (the host's read of the pair
// fails and restarts the server, as before). The thread holds the writer and the frames (keep), so a writer an early
// return leaves blocked on a host that stopped reading never touches freed memory before the process exits.
struct PairWriter : std::enable_shared_from_this<PairWriter>
{
    std::mutex m;
    std::condition_variable cv;
    std::vector<const uint8_t*> src;
    size_t bytes = 0;
    bool busy = false, quit = false, failed = false;
    std::shared_ptr<ServerFrames> keep;
    std::thread th;

    void start()
    {
        th = std::thread([self = shared_from_this()] { self->run(); });
    }
    void run()
    {
        for (;;)
        {
            std::unique_lock<std::mutex> lk(m);
            cv.wait(lk, [&] { return busy || quit; });
            if (!busy)
                return;
            const std::vector<const uint8_t*> s = src;
            const size_t n = bytes;
            lk.unlock();
            bool ok = true;
            for (const uint8_t* p : s)
                ok = ok && fwrite(p, 1, n, stdout) == n;
            ok = fflush(stdout) == 0 && ok;
            lk.lock();
            if (!ok && !failed)
            {
                failed = true;
                LOG("the output pipe write failed (the host stopped reading)\n");
            }
            busy = false;
            cv.notify_all();
        }
    }
    void waitIdle()
    {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&] { return !busy; });
    }
    void post(const std::vector<const uint8_t*>& s, size_t n)
    {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&] { return !busy; });
        src = s;
        bytes = n;
        busy = true;
        cv.notify_all();
    }
    // a normal end has waited for the last pair; an early return still lets the pair in flight reach the host (it
    // reads that pair in either order, and a host that stopped reading fails the write), so a failure regenerates
    // only the failing pair; a writer still blocked after 10 s (the host's watchdog) is left to the process exit
    void finish()
    {
        bool idle;
        {
            std::unique_lock<std::mutex> lk(m);
            idle = cv.wait_for(lk, std::chrono::seconds(10), [&] { return !busy; });
            quit = true;
        }
        cv.notify_all();
        if (idle)
            th.join();
        else
            th.detach();
    }
};
struct PairWriterGuard
{
    std::shared_ptr<PairWriter> w;
    ~PairWriterGuard()
    {
        if (w && w->th.joinable())
            w->finish();
    }
};

// SMV_DLSSG_NGX is set to value: 0 = Streamline's route, fail = NGX's own route treated as unavailable after its full
// start (the trigger test of the fallback to Streamline's)
static bool ngxLever(const wchar_t* value)
{
    wchar_t v[8] = {0};
    return GetEnvironmentVariableW(L"SMV_DLSSG_NGX", v, _countof(v)) == wcslen(value) && wcscmp(v, value) == 0;
}

// --server on NGX's own calls (ngx_direct.h): runServer's protocol and frames without Streamline, its swap chain and
// its pacer. false = this route did not come up and nothing went to stdout: the caller starts Streamline's route.
static bool runServerDirect(int genFrames, bool hdr10, HANDLE cutEv, int& rc)
{
    ngxfg::Direct fg;
    if (!fg.init(genFrames, hdr10))
        return false;
    if (ngxLever(L"fail"))
    {
        LOG("SMV_DLSSG_NGX=fail: NGX's own route treated as unavailable\n");
        fg.shutdown();
        return false;
    }
    rc = 1;
    if (fg.maxGen && (uint32_t)genFrames > fg.maxGen)
    {
        LOG("multi-frame generation beyond %ux is not supported on this GPU (requested %ux)\n", fg.maxGen + 1,
            genFrames + 1);
        rc = 3;
        return true;
    }
    LOG("DLSS-G ready: NGX's own calls, no Streamline\n");

    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
    // runServer's handshake, plus ngx=1 = this route
    fprintf(stdout, "DLSSG READY gen=%d max=%u%s wq=1%s ngx=1\n", genFrames, fg.maxGen, hdr10 ? " hdr10=1" : "",
            cutEv ? " cut=1" : "");
    fflush(stdout);

    const size_t frameBytes = (size_t)W * H * 4;
    std::vector<uint8_t> inFrame(frameBytes);
    const auto frames = std::make_shared<ServerFrames>();
    auto& outFrames = frames->outFrames;
    auto& outFramesW = frames->outFramesW;
    outFrames.assign((size_t)genFrames, std::vector<uint8_t>(frameBytes));
    outFramesW.assign((size_t)genFrames, std::vector<uint8_t>(frameBytes));
    std::vector<uint8_t*> outPtrs, outPtrsW;
    for (auto& f : outFrames)
        outPtrs.push_back(f.data());
    for (auto& f : outFramesW)
        outPtrsW.push_back(f.data());
    const auto writerP = std::make_shared<PairWriter>();
    writerP->keep = frames;
    writerP->start();
    PairWriterGuard writerGuard{writerP};
    PairWriter& writer = *writerP;

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

    uint64_t emitted = 0;
    bool first = true;
    while (readFull(inFrame.data(), frameBytes))
    {
        // a new clip (the host set the cut event before it sent this frame): nothing of the clip before may reach
        // its tweens, so the feature is made again and this frame takes a first frame's warm-up
        if (!first && cutEv && WaitForSingleObject(cutEv, 0) == WAIT_OBJECT_0)
        {
            LOG("new clip: DLSS-G's feature made again, this frame warms it up; the clip before it:\n");
            fg.logPan();
            if (!fg.newClip())
            {
                LOG("new clip: the DLSS-G feature did not come back\n");
                return true;
            }
            first = true;
        }
        if (first)
        {
            // the first frame three times, as the three warm-up presents of Streamline's route: the frames after it
            // depend on how many evaluates came before
            for (int i = 0; i < 3; i++)
                if (!fg.frame(inFrame.data(), nullptr))
                {
                    LOG("warmup evaluate failed\n");
                    return true;
                }
            first = false;
            continue;
        }
        if (!fg.frame(inFrame.data(), outPtrs.data()))
            return true;
        // the pair to the writer (after the previous one is written), then this thread takes the other buffer set
        const std::vector<const uint8_t*> outSrc(outPtrs.begin(), outPtrs.end());
        writer.post(outSrc, frameBytes);
        outFrames.swap(outFramesW);
        outPtrs.swap(outPtrsW);
        emitted += genFrames;
    }
    writer.waitIdle();
    fg.logPan();
    LOG("server done, %llu interpolated frames\n", (unsigned long long)emitted);
    fg.shutdown();
    rc = 0;
    return true;
}

static int runServer(int genFrames, bool hdr10, HANDLE cutEv)
{
    Host host;
    host.nativeCapture = true;
    host.genFrames = genFrames;
    host.hdr10 = hdr10;
    int rc = host.init();
    if (rc)
        return rc;

    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);

    // handshake so the driving process can distinguish "up" from "unsupported"; wq=1 = the pair writer: the host may
    // send the next frame before it reads this pair; cut=1 = the cut event is watched (a new clip on this server)
    fprintf(stdout, "DLSSG READY gen=%d max=%u%s wq=1%s\n", genFrames, host.maxGen, host.hdr10 ? " hdr10=1" : "",
            cutEv ? " cut=1" : "");
    fflush(stdout);

    const size_t frameBytes = (size_t)W * H * 4;
    std::vector<uint8_t> inFrame(frameBytes), prevFrame(frameBytes);
    const auto frames = std::make_shared<ServerFrames>();
    auto& raw = frames->raw; // the captured presents, buffers reused from pair to pair
    auto& rawW = frames->rawW;
    auto& outFrames = frames->outFrames;
    auto& outFramesW = frames->outFramesW;
    outFrames.assign((size_t)genFrames, std::vector<uint8_t>(frameBytes));
    outFramesW.assign((size_t)genFrames, std::vector<uint8_t>(frameBytes));
    std::vector<uint8_t*> outPtrs, outPtrsW;
    for (auto& f : outFrames)
        outPtrs.push_back(f.data());
    for (auto& f : outFramesW)
        outPtrsW.push_back(f.data());
    // the frames written: a captured present in raw (the counted tier) or an outFrames copy (the sweep tier)
    std::vector<const uint8_t*> outSrc((size_t)genFrames);
    size_t rawN = 0;
    const auto writerP = std::make_shared<PairWriter>();
    writerP->keep = frames;
    writerP->start();
    PairWriterGuard writerGuard{writerP};
    PairWriter& writer = *writerP;

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
        host.syncRing();
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
    // every pixel's RGB equal (alpha skipped), 8 bytes at a time
    const uint64_t rgbMask2 = (uint64_t)rgbMask | ((uint64_t)rgbMask << 32);
    auto sameFull = [&](const uint8_t* a, const uint8_t* b) -> bool {
        const size_t px = (size_t)W * H, words = px / 2;
        uint64_t acc = 0;
        for (size_t i = 0; i < words; i++)
        {
            uint64_t x, y;
            memcpy(&x, a + i * 8, 8);
            memcpy(&y, b + i * 8, 8);
            acc |= (x ^ y) & rgbMask2;
        }
        if (px & 1)
            acc |= rgbAt(a, px - 1) ^ rgbAt(b, px - 1);
        return acc == 0;
    };
    // the pixels whose RGB differs, all of them (the sweep's multi-frame order on pairs that barely differ)
    auto diffFull = [&](const uint8_t* a, const uint8_t* b) -> size_t {
        const size_t px = (size_t)W * H;
        size_t n = 0;
        for (size_t i = 0; i < px; i++)
            n += rgbAt(a, i) != rgbAt(b, i);
        return n;
    };
    // identical image content: the samples first (most frames differ there), then every pixel, since a generated
    // frame of a pair that barely moves (a later cascade pass's input) can differ from a real frame off every sample
    // (350 of 2.1 M pixels, none on one), and a sampled match alone took it for a real-frame copy
    auto sameImage = [&](const uint8_t* a, const uint8_t* b) -> bool {
        const size_t px = (size_t)W * H;
        for (size_t i = 0; i < px; i += 397)
            if (rgbAt(a, i) != rgbAt(b, i))
                return false;
        return sameFull(a, b);
    };
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
        on.mvecDepthWidth = panvec::vecSize(W);
        on.mvecDepthHeight = panvec::vecSize(H);
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
    // the previous pair's real present did not come among its own window's presents: DLSS-G presents every real frame
    // once, so it can still arrive first in the next pair's stream
    bool realPending = false;
    bool metered = false;  // sticky: hardware flip metering detected, capture by buffer sweep
    bool gapReset = false; // SMV_DLSSG_GAP_RESET=1: a reset before every pair whose input came 500 ms after the last
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
        // SMV_DLSSG_WAITMS=N: the newest present's wait (an A/B lever)
        n = GetEnvironmentVariableA("SMV_DLSSG_WAITMS", buf, (DWORD)sizeof(buf));
        if (n > 0 && n < sizeof(buf))
        {
            host.waitMs = atoi(buf);
            LOG("SMV_DLSSG_WAITMS=%d: the newest present waits up to %d ms for a later one\n", host.waitMs,
                host.waitMs);
        }
        n = GetEnvironmentVariableA("SMV_DLSSG_GAP_RESET", buf, (DWORD)sizeof(buf));
        if (n == 1 && buf[0] == '1')
        {
            gapReset = true;
            LOG("SMV_DLSSG_GAP_RESET=1: a pair whose input came over 500 ms after the last present resets DLSS-G first\n");
        }
    }
    std::vector<std::vector<uint32_t>> preSig(host.nativeBufCount);
    std::vector<uint8_t> sweepScratch(frameBytes);
    ULONGLONG lastPresentTick = 0;
    while (readFull(inFrame.data(), frameBytes))
    {
        // A new clip (the host set the cut event before it sent this frame): this frame makes no pair with the clip
        // before, and nothing of that clip may reach the new one's tweens. DLSS-G keeps more than the last frame once
        // a pan was tagged, so it goes off (its resources kept; a present of the old clip's last frame applies it) and
        // on, the pan estimator starts from nothing and this frame takes the first frame's warm-up below: the tweens
        // after it are those of a server that just started
        if (!first && cutEv && WaitForSingleObject(cutEv, 0) == WAIT_OBJECT_0)
        {
            sl::DLSSGOptions off{};
            off.mode = sl::DLSSGMode::eOff;
            off.flags = sl::DLSSGFlags::eRetainResourcesWhenOff;
            if (slDLSSGSetOptions(host.vp, off) != sl::Result::eOk || !host.presentFrame(prevFrame.data()))
            {
                LOG("new clip: DLSS-G did not go off\n");
                return 1;
            }
            sl::DLSSGOptions on{};
            on.mode = sl::DLSSGMode::eOn;
            on.numFramesToGenerate = (uint32_t)genFrames;
            on.mvecDepthWidth = panvec::vecSize(W);
            on.mvecDepthHeight = panvec::vecSize(H);
            if (slDLSSGSetOptions(host.vp, on) != sl::Result::eOk)
            {
                LOG("new clip: DLSS-G did not come back on\n");
                return 1;
            }
            LOG("new clip: DLSS-G off and on, this frame warms it up; the clip before it:\n");
            host.logPan();
            host.pan = panvec::Estimator();
            realPending = false;
            first = true;
        }
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

        // pair = (prevFrame, inFrame). No reset on the clock: the wait for an input also counts a later cascade pass
        // still starting (its full pipe stops this host's writer, then its feed), so a clock reset lands at a pair that
        // depends on timing and changes the tweens after it; a pair whose generated frames do not come is reset and
        // retried by the attempts below
        if (gapReset && GetTickCount64() - lastPresentTick > 500)
        {
            LOG("input gap detected, resetting DLSS-G\n");
            resets++;
            if (!resetFG())
            {
                LOG("DLSS-G reset failed\n");
                return 1;
            }
            realPending = false;
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
        // a pair whose frames differ in at most 1 pixel of 2000 (a later cascade pass's input on footage that barely
        // moves: 18 of 2.1 M pixels measured) counts as static for the capture: its generated frames can come out equal
        // to a real frame, so presents are taken in order, not told apart by content (a real-frame copy there differs
        // from the generated frame in those few pixels at most); the full count runs only when few samples differ
        const int pairSampled = diffSamples(prevFrame.data(), inFrame.data());
        const size_t pairDiff = pairSampled > 8 ? (size_t)W * H : diffFull(prevFrame.data(), inFrame.data());
        const bool pairStatic = pairDiff == 0;
        const bool pairNear = pairDiff * 2000 <= (size_t)W * H;
        const bool inputUniform = uniformImage(inFrame.data());
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
                realPending = false; // the reset's drain took every present before it
            }
            if (!host.presentFrame(inFrame.data()))
            {
                LOG("presentFrame failed\n");
                return 1;
            }
            host.pollSeen = host.presentSnapshot;
            rawN = 0;
            size_t scanned = 0;
            int filled = 0;
            // a near-static pair takes its presents in order: while the previous pair's real present is owed, the first
            // exact copy of A is that present arriving late, set aside; it fills the pair only if the pair comes up short
            // (a generated frame with A's pixels, the earliest tween)
            bool skipA = pairNear && !pairStatic && realPending;
            const uint8_t* asideA = nullptr;
            const ULONGLONG t0 = GetTickCount64();
            ULONGLONG nextSweep = t0 + (metered ? 50 : 350);
            while (filled < genFrames && GetTickCount64() < t0 + 1000)
            {
                int added = host.pollNewPresents(raw, rawN);
                if (added < 0)
                {
                    LOG("present readback failed\n");
                    return 1;
                }
                for (; scanned < rawN && filled < genFrames; scanned++)
                {
                    const uint8_t* f = raw[scanned].data();
                    if (!pairNear && (sameImage(f, inFrame.data()) || sameImage(f, prevFrame.data())))
                        continue; // a real-frame present, not a generated frame
                    if (skipA && sameImage(f, prevFrame.data()))
                    {
                        skipA = false;
                        asideA = f;
                        continue;
                    }
                    outSrc[filled++] = f;
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
                        // multi-frame order: distance to the left real frame grows with the timestep (every pixel:
                        // the samples tie on a pair that barely moves)
                        std::vector<size_t> dist(cand.size()), ord(cand.size());
                        for (size_t c = 0; c < cand.size(); c++)
                        {
                            dist[c] = diffFull(cand[c].data(), prevFrame.data());
                            ord[c] = c;
                        }
                        std::stable_sort(ord.begin(), ord.end(), [&](size_t a, size_t b) { return dist[a] < dist[b]; });
                        for (filled = 0; filled < genFrames; filled++)
                        {
                            memcpy(outPtrs[filled], cand[ord[filled]].data(), frameBytes);
                            outSrc[filled] = outPtrs[filled];
                        }
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
            if (filled < genFrames && asideA)
            {
                for (int k = filled; k > 0; k--)
                    outSrc[k] = outSrc[k - 1];
                outSrc[0] = asideA;
                filled++;
            }
            good = (filled == genFrames);
            if (!good)
            {
                LOG("captured %d generated frame(s) of %d (plus %zu real presents) within 1s\n", filled, genFrames,
                    rawN - filled);
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
        // this pair's real present owed = no exact copy of B among the presents its window received (the sweep tier
        // reads buffers, not the present stream: nothing owed)
        realPending = !metered;
        for (size_t i = 0; i < rawN && realPending; i++)
            if (sameImage(raw[i].data(), inFrame.data()))
                realPending = false;
        lastPresentTick = GetTickCount64();

        // the pair to the writer (after the previous one is written), then this thread takes the other buffer set
        writer.post(outSrc, frameBytes);
        raw.swap(rawW);
        outFrames.swap(outFramesW);
        outPtrs.swap(outPtrsW);
        emitted += genFrames;
        prevFrame.swap(inFrame);
    }
    writer.waitIdle();
    if (resets)
        LOG("note: %llu DLSS-G pacer resets during this run\n", (unsigned long long)resets);
    host.logPan();
    LOG("server done, %llu interpolated frames\n", (unsigned long long)emitted);
    host.shutdown();
    slShutdown();
    return 0;
}

// --stage1 --from: stdin carries no frame, only the end (the host closes it). The first call starts a reader that
// outlives its caller and sets the flag returned
static std::atomic<bool>& stdinEnd()
{
    static std::atomic<bool> end{false};
    std::thread([] {
        char c;
        while (fread(&c, 1, 1, stdin) == 1)
        {
        }
        end = true;
    }).detach();
    return end;
}

// --stage1's frames on stdin, read by a thread of their own: NGX's stage shares a pair's frames at their times while
// the next input arrives. One frame is handed over at a time: the reader waits until the stage took the one before
// (which holds the feeding pipe back, as a blocking read does) and never fills the buffer the stage still works from
struct StdinFrames
{
    std::mutex m;
    std::condition_variable cv;
    std::vector<uint8_t> buf[2];
    int have = -1; // the buffer holding a frame the stage has not taken
    int held = -1; // the buffer the stage works from
    double haveMs = 0;
    bool end = false;
    HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr); // set at every hand-over and at the end

    // the reader holds the frames (self): it outlives its caller, blocked in a read
    static void start(const std::shared_ptr<StdinFrames>& self, size_t bytes)
    {
        self->buf[0].resize(bytes);
        self->buf[1].resize(bytes);
        std::thread([self, bytes] {
            int w = 0;
            for (;;)
            {
                {
                    std::unique_lock<std::mutex> lk(self->m);
                    self->cv.wait(lk, [&] { return self->held != w; });
                }
                size_t got = 0;
                while (got < bytes)
                {
                    const size_t r = fread(self->buf[w].data() + got, 1, bytes - got, stdin);
                    if (r == 0)
                        break;
                    got += r;
                }
                const double t = qpcMs();
                std::unique_lock<std::mutex> lk(self->m);
                if (got < bytes)
                {
                    self->end = true;
                    SetEvent(self->ev);
                    return;
                }
                self->cv.wait(lk, [&] { return self->have < 0; });
                self->have = w;
                self->haveMs = t;
                w ^= 1;
                SetEvent(self->ev);
            }
        }).detach();
    }

    // 1 = a frame (its bytes and when it arrived; done() gives its buffer back), 0 = none yet, -1 = stdin ended
    int take(const uint8_t*& p, double& at)
    {
        std::lock_guard<std::mutex> lk(m);
        if (have < 0)
            return end ? -1 : 0;
        held = have;
        have = -1;
        p = buf[held].data();
        at = haveMs;
        cv.notify_all();
        return 1;
    }
    void done()
    {
        {
            std::lock_guard<std::mutex> lk(m);
            held = -1;
        }
        cv.notify_all();
    }
};

// The source's frame interval, which a pair's frames are spaced over: the mean of the last four gaps between inputs
// (a 24 fps video on a 60 Hz desktop arrives 33 and 50 ms apart), at most 1.25 x the last gap (a source that sped up
// is followed at once). One gap far off the mean stays out of it (a dropped frame, a stall); a second one like it is
// the source's new rate.
struct SourceClock
{
    double win[4] = {};
    int n = 0, at = 0;
    double held = 0; // a gap far off the mean, waiting for a second like it

    double mean() const
    {
        double s = 0;
        for (int i = 0; i < n; i++)
            s += win[i];
        return n ? s / n : 0;
    }
    void push(double gap)
    {
        win[at] = gap;
        at = (at + 1) % 4;
        n = (std::min)(n + 1, 4);
    }
    // the interval for the pair an input closes, gap = the time since the input before it
    double add(double gap)
    {
        const double m = mean();
        if (n && (gap > 1.6 * m || gap < 0.6 * m))
        {
            if (held > 0 && fabs(gap - held) <= 0.3 * held)
            {
                n = at = 0;
                push(held);
                push(gap);
                held = 0;
            }
            else
                held = gap;
        }
        else
        {
            push(gap);
            held = 0;
        }
        return (std::min)(mean(), 1.25 * gap);
    }
};

// --stage1 on NGX's own calls (ngx_direct.h): runStage1's ring, header and frames without Streamline, its swap chain,
// its window and its pacer. An input is evaluated the moment it arrives and its frames are shared one by one at their
// places in the source's interval (SourceClock), the first as soon as it is made: the cadence Streamline's pacer gives
// a game. No clock runs inside the frames, so an input gap needs no reset. false = this route did not come up and
// nothing was shared or written: the caller starts Streamline's stage.
static bool runStage1Direct(int genFrames, bool hdr10, bool bgra, const std::wstring& tag, const std::wstring& from,
                            int& rc)
{
    ngxfg::Direct fg;
    if (!fg.init(genFrames, hdr10, bgra, false))
        return false;
    if (ngxLever(L"fail"))
    {
        LOG("SMV_DLSSG_NGX=fail: NGX's own route treated as unavailable\n");
        fg.shutdown();
        return false;
    }
    rc = 1;
    if (fg.maxGen && (uint32_t)genFrames > fg.maxGen)
    {
        LOG("multi-frame generation beyond %ux is not supported on this GPU (requested %ux)\n", fg.maxGen + 1,
            genFrames + 1);
        rc = 3;
        return true;
    }
    LOG("DLSS-G ready: NGX's own calls, no Streamline\n");
    // 16 slots: an input's frames are made ahead of their times, so up to six slots a reader has not been given yet
    // are already written
    Stage1Ring ring;
    UpstreamRing up;
    if (!ring.share(fg.device.Get(), fg.frameFormat(), genFrames, 16, tag) || !fg.stageInit(ring.fence.Get()) ||
        (!from.empty() && !up.open(fg.device.Get(), from, fg.frameFormat())))
        return true;
    ring.ready();
    _setmode(_fileno(stdin), _O_BINARY);
    // runStage1's handshake, plus ngx=1 = this route
    fprintf(stdout, "DLSSG STAGE1 READY gen=%d max=%u share=%ls%s ngx=1\n", genFrames, fg.maxGen, tag.c_str(),
            fg.hdr10  ? " hdr10=1"
            : fg.bgra ? " bgra=1"
                      : "");
    fflush(stdout);

    const auto in = std::make_shared<StdinFrames>();
    const std::atomic<bool>* inEnd = nullptr;
    if (from.empty())
        StdinFrames::start(in, (size_t)W * H * 4);
    else
        inEnd = &stdinEnd();
    const uint8_t* cpu = nullptr;
    uint32_t upSlot = 0;
    uint64_t upValue = 0;
    // the next input if one is there: at = when it arrived; ended = none will come
    auto takeInput = [&](double& at, bool& ended) -> bool {
        ended = false;
        if (from.empty())
        {
            const int r = in->take(cpu, at);
            ended = r < 0;
            return r > 0;
        }
        if (up.next(upSlot, upValue))
        {
            at = qpcMs();
            return true;
        }
        ended = *inEnd || up.hdr->status >= 2;
        return false;
    };
    auto input = [&]() -> bool {
        return from.empty() ? fg.stageInput(cpu)
                            : fg.stageInput(nullptr, up.ring[upSlot].Get(), up.fence.Get(), upValue);
    };
    // a clip's first input three times, each with its evaluates (the three warm-up presents of Streamline's stage;
    // --server's rule): the frame texture is left for stageEnd
    auto warm = [&]() -> bool {
        for (int i = 0; i < 3; i++)
        {
            if (!(i && !from.empty() ? fg.stageInput(nullptr) : input()))
                return false;
            for (int k = 1; k <= fg.gen; k++)
                if (!fg.stageTween(k, nullptr))
                    return false;
            if (i < 2 && !fg.stageEnd(nullptr))
                return false;
        }
        return true;
    };

    // set when an input arrives: the stdin reader's hand-over, or the upstream stage's publish (null = an upstream
    // without that event: its ring is polled)
    const HANDLE inputEv = from.empty() ? in->ev : up.wake;
    double at = 0;
    bool ended = false;
    while (!takeInput(at, ended))
    {
        if (ended)
        {
            rc = 0;
            return true;
        }
        if (inputEv)
            WaitForSingleObject(inputEv, 250);
        else
            SwitchToThread();
    }
    if (!warm() || !fg.stageEnd(nullptr))
    {
        LOG("warmup evaluate failed\n");
        return true;
    }
    if (from.empty())
        in->done();
    ring.hdr->status = 1;

    // a frame made and not shared yet: complete at the fence's `value`; the first of its input goes out as soon as it
    // is complete, the others `after` ms behind that one
    struct Made
    {
        uint64_t value;
        double after;
        bool first;
        uint32_t input;
    };
    std::deque<Made> made;
    uint64_t slotNo = 0; // frames written into the ring, shared or not
    auto slot = [&]() { return ring.ring[slotNo++ % ring.slots].Get(); };
    auto shareIt = [&](const Made& f) {
        LARGE_INTEGER t;
        QueryPerformanceCounter(&t);
        ring.publish(f.value, t.QuadPart, f.input);
    };
    SourceClock clock;
    double lastAt = at, base = 0;
    bool rewarm = false; // DLSS-G saw none of the inputs passed through at 1x: the next one starts a clip
    uint64_t inputs = 1, statInputs = 0, statFrames = 0;
    ULONGLONG statTick = GetTickCount64();
    // an input: its generated frames and itself into the ring, each with its place in the source's interval
    auto take = [&]() -> bool {
        const double interval = clock.add(at - lastAt);
        lastAt = at;
        // the host's ask for another multiple (a plan stepping up or down): the count is an argument of every
        // evaluate, so it changes in place; kGenOff = 1x, the inputs shared as they are
        const uint32_t ask = ring.hdr->wantGen;
        const int askGen = ask == kGenOff ? 0 : (int)ask;
        if (ask && askGen != fg.gen)
        {
            if ((uint32_t)askGen <= (fg.maxGen ? fg.maxGen : 5u))
            {
                LOG("stage1: %dx from input %llu on (asked)\n", askGen + 1, (unsigned long long)inputs);
                rewarm = fg.gen == 0;
                fg.gen = askGen;
                ring.hdr->gen = (uint32_t)askGen;
            }
            else
            {
                LOG("stage1: %dx asked, refused: stays at %dx\n", askGen + 1, fg.gen + 1);
                ring.hdr->wantGen = 0;
            }
        }
        const uint32_t no = (uint32_t)inputs++;
        ring.hdr->lastInput = no;
        uint64_t v = 0;
        if (fg.gen > 0 && rewarm)
        {
            fg.resetNext = true;
            if (!warm() || !(v = fg.stageEnd(slot())))
                return false;
            rewarm = false;
            made.push_back({v, 0, true, no});
        }
        else
        {
            if (!input())
                return false;
            const double step = interval / (fg.gen + 1);
            for (int k = 1; k <= fg.gen; k++)
            {
                if (!(v = fg.stageTween(k, slot())))
                    return false;
                made.push_back({v, (k - 1) * step, k == 1, no});
            }
            if (!(v = fg.stageEnd(slot())))
                return false;
            made.push_back({v, fg.gen * step, fg.gen == 0, no});
        }
        if (from.empty())
            in->done();
        return true;
    };

    // CREATE_WAITABLE_TIMER_HIGH_RESOLUTION: a wait ends within about half a millisecond of its time
    const HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, 0x00000002, TIMER_ALL_ACCESS);
    const HANDLE gpuEv = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    for (;;)
    {
        // share the frames whose time has come, in order
        double wait = -1; // ms until the first waiting frame's time; -1 = none waits for a time
        bool gpuWait = false;
        while (!made.empty())
        {
            const Made& f = made.front();
            if (ring.fence->GetCompletedValue() < f.value)
            {
                gpuWait = true;
                break;
            }
            const double now = qpcMs();
            if (f.first)
                base = now;
            else if (now < base + f.after)
            {
                wait = base + f.after - now;
                break;
            }
            shareIt(f);
            made.pop_front();
        }
        if (takeInput(at, ended))
        {
            // frames still waiting: the source came sooner than its interval said, they go out now
            for (const Made& f : made)
                shareIt(f);
            made.clear();
            if (!take())
            {
                LOG("stage1: an input's frames were not made\n");
                ring.finish(3);
                return true;
            }
            const ULONGLONG tick = GetTickCount64();
            if (tick - statTick >= 5000)
            {
                const double s = (tick - statTick) / 1000.0;
                const uint64_t f = ring.frames, n = inputs - statInputs;
                LOG("stage1: %.1f inputs/s, %.1f shared presents/s (%.2f an input)%s\n", n / s, (f - statFrames) / s,
                    n ? (double)(f - statFrames) / n : 0.0,
                    from.empty() ? "" : (", " + std::to_string(up.skipped) + " upstream frames skipped").c_str());
                statTick = tick;
                statInputs = inputs;
                statFrames = f;
            }
            continue;
        }
        if (ended)
            break;
        // until an input, the GPU or the next frame's time (the end of a stage fed by a ring shows at a wait's
        // timeout); the last half millisecond before a frame's time is spun
        if (!inputEv)
            SwitchToThread();
        else if (gpuWait)
        {
            ring.fence->SetEventOnCompletion(made.front().value, gpuEv);
            const HANDLE hs[2] = {inputEv, gpuEv};
            WaitForMultipleObjects(2, hs, FALSE, 100);
        }
        else if (wait < 0)
            WaitForSingleObject(inputEv, 250);
        else if (wait > 0.6 && timer)
        {
            LARGE_INTEGER due;
            due.QuadPart = -(LONGLONG)((wait - 0.4) * 10000.0);
            SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE);
            const HANDLE hs[2] = {inputEv, timer};
            WaitForMultipleObjects(2, hs, FALSE, 250);
        }
        else
            YieldProcessor();
    }
    for (const Made& f : made)
        shareIt(f);
    // every shared frame's fence value is reached before the stage ends: a reader waits on it
    fg.stageWait(fg.sValue);
    ring.finish(2);
    fg.logPan();
    LOG("stage1 done, %llu inputs, %llu shared presents\n", (unsigned long long)inputs,
        (unsigned long long)ring.frames.load());
    fg.shutdown();
    rc = 0;
    return true;
}

// --stage1 (see the usage at the top): present every input as it arrives, share every native present
static int runStage1(int genFrames, bool hdr10, bool bgra, const std::wstring& tag, const std::wstring& from)
{
    Host host;
    host.nativeCapture = true;
    host.genFrames = genFrames;
    host.hdr10 = hdr10;
    host.bgra = bgra && !hdr10;
    // free-running: each input is presented on arrival, so no vsync pacing (a sync-1 present caps the stage at the
    // panel's refresh: 360 frames a second on a 360 Hz panel, 60 on a 60 Hz one); SMV_DLSSG_STAGE1_VSYNC=1 = paced (A/B)
    wchar_t sv[8]{};
    host.syncInterval = GetEnvironmentVariableW(L"SMV_DLSSG_STAGE1_VSYNC", sv, 8) && sv[0] == L'1' ? 1u : 0u;
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
    const std::atomic<bool>* inEnd = from.empty() ? nullptr : &stdinEnd();
    uint32_t upSlot = 0;
    uint64_t upValue = 0;
    auto nextInput = [&]() -> bool {
        if (from.empty())
            return readFull(inFrame.data(), frameBytes);
        while (!up.next(upSlot, upValue))
        {
            if (*inEnd || up.hdr->status >= 2)
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
    // the same reset as --server's (off at a present, back on, two re-warm presents); at 1x (genFrames 0) DLSS-G stays off
    auto resetFG = [&]() -> bool {
        sl::DLSSGOptions off{};
        off.mode = sl::DLSSGMode::eOff;
        off.flags = sl::DLSSGFlags::eRetainResourcesWhenOff;
        if (slDLSSGSetOptions(host.vp, off) != sl::Result::eOk || !presentAgain())
            return false;
        if (genFrames <= 0)
            return true;
        sl::DLSSGOptions on{};
        on.mode = sl::DLSSGMode::eOn;
        on.numFramesToGenerate = (uint32_t)genFrames;
        on.mvecDepthWidth = panvec::vecSize(W);
        on.mvecDepthHeight = panvec::vecSize(H);
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
        // the host's ask for another multiple (a plan stepping up or down): DLSS-G takes a new numFramesToGenerate at the
        // next present, in place (the frames from it on are byte-identical to a fresh run at that count: harness
        // dlssg_gen_inplace); kGenOff = 1x: DLSS-G off, the inputs presented as they are
        const uint32_t ask = ring.hdr->wantGen;
        const int askGen = ask == kGenOff ? 0 : (int)ask;
        if (ask && askGen != genFrames)
        {
            sl::DLSSGOptions opt{};
            opt.mode = askGen ? sl::DLSSGMode::eOn : sl::DLSSGMode::eOff;
            if (askGen)
            {
                opt.numFramesToGenerate = (uint32_t)askGen;
                opt.mvecDepthWidth = panvec::vecSize(W);
                opt.mvecDepthHeight = panvec::vecSize(H);
            }
            else
                opt.flags = sl::DLSSGFlags::eRetainResourcesWhenOff;
            if ((uint32_t)askGen <= (host.maxGen ? host.maxGen : 5u) &&
                slDLSSGSetOptions(host.vp, opt) == sl::Result::eOk)
            {
                LOG("stage1: %dx from input %llu on (asked)\n", askGen + 1, (unsigned long long)inputs);
                genFrames = askGen;
                host.genFrames = askGen;
                ring.hdr->gen = (uint32_t)askGen;
            }
            else
            {
                LOG("stage1: %dx asked, refused: stays at %dx\n", askGen + 1, genFrames + 1);
                ring.hdr->wantGen = 0;
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
    host.logPan();
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
        HANDLE cutEv = nullptr;
        for (int i = 4; i < argc; i++)
        {
            if (wcscmp(argv[i], L"--gen") == 0 && i + 1 < argc)
                genFrames = _wtoi(argv[++i]);
            else if (wcscmp(argv[i], L"--hdr10") == 0)
                hdr10 = true;
            else if (wcscmp(argv[i], L"--cut-event") == 0 && i + 1 < argc)
                cutEv = (HANDLE)(uintptr_t)_wcstoui64(argv[++i], nullptr, 10);
        }
        if (genFrames < 1 || genFrames > 5)
        {
            LOG("--gen must be 1..5 (got %d)\n", genFrames);
            return 1;
        }
        // NGX's own calls first; Streamline's route when they do not come up here, with SMV_DLSSG_NGX=0, and for the
        // HDR10 refusal's trigger (a swap chain's answer)
        const bool refuseHdr = hdr10 && GetEnvironmentVariableW(L"SMV_DLSSG_REFUSE_HDR10", nullptr, 0) != 0;
        int rc = 1;
        if (!ngxLever(L"0") && !refuseHdr && runServerDirect(genFrames, hdr10, cutEv, rc))
            return rc;
        if (slInitCommon())
            return 1;
        return runServer(genFrames, hdr10, cutEv);
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
        // NGX's own calls first, as --server; Streamline's stage when they do not come up here and with
        // SMV_DLSSG_NGX=0
        int rc = 1;
        if (!ngxLever(L"0") && runStage1Direct(genFrames, hdr10, bgra, tag, from, rc))
            return rc;
        if (slInitCommon())
            return 1;
        return runStage1(genFrames, hdr10, bgra, tag, from);
    }

    LOG("usage: dlssg2f.exe --server W H [--gen N] [--hdr10]\n"
        "       dlssg2f.exe --stage1 W H [--gen N] [--hdr10 | --bgra] --share TAG [--from TAG]\n");
    return 1;
}
