// SmoothMyVideo's Intel XeSS-FG bridge: Intel's libxess_fg.dll + libxell.dll (XeSS SDK 3.0.2, unmodified, beside this
// DLL in engine\xess; the app's XeSS setup downloads them) behind the same small C API as engine\fsrfg's bridge (exports
// xefg_*), so the native host drives XeSS frame generation on AMD FSR frame generation's route.
// XeSS-FG is a swap chain: the caller presents real frames to its proxy swap chain and XeSS-FG presents the generated
// frames to a real swap chain it creates itself, through the DXGI factory it is given. The bridge hands it OUR factory
// (FactoryWrap, every call forwarded), whose swap chain (SwapWrap) copies each presented back buffer into the shared
// output texture on the queue XeSS-FG presents with, then signals the caller's fence, then presents. XeSS-FG runs with
// its debug feature SHOW_ONLY_INTERPOLATION (only generated frames are presented; a failed interpolation presents the
// real frame instead), so every call after the first presents exactly one frame. The window each real swap chain needs
// is a no-activate popup parked on the primary monitor's last pixel, excluded from screen capture, on its own message
// thread. Intel's DLLs stay unmodified and are driven through their public API only.
// Frames go in and out through three D3D12 textures shared with CUDA (NT handles the host imports as D3D12 resources and
// maps as CUDA arrays) and one shared D3D12 fence, so the GPU waits on the GPU: the host writes the frame and its motion
// vectors, signals `wait`; the bridge's queue waits for it, copies the frame into the proxy's back buffer, tags the
// vectors and presents; the generated frame halfway between the previous call's frame and this one lands in the output
// texture and `signal` follows. One XeSS-FG context, swap chain and window per instance; every call feeds the next frame.
// Formats: fmt 0 = RGBA8 UNORM (sRGB-encoded bytes), fmt 1 = RGBA16F (sRGB-encoded values 0..1); motion vectors = RG16F
// in pixels, current -> previous (XeSS's convention); depth = a constant (no depth in video); identity matrices. XeSS-FG
// takes no FP16 back buffer (its guide: RGBA8, or R10G10B10A2), so the swap chains are R10G10B10A2 (SMV_XEFG_BB=8 =
// RGBA8; 10 bits a channel scored a little above 8 on SDR video, the same cost): one compute pass (compiled once by the
// system's d3dcompiler_47.dll) converts the frame to the swap chains' format on the way in (the UNORM store's rounding)
// and the generated frame back to the shared format on the way out.
#include <windows.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>

#include "xess_fg/xefg_swapchain.h"
#include "xess_fg/xefg_swapchain_d3d12.h"
#include "xess_fg/xefg_swapchain_debug.h"
#include "xell/xell.h"
#include "xell/xell_d3d12.h"

#define XEFG_API extern "C" __declspec(dllexport)

struct XefgShared
{
    HANDLE in;                           // the real frame: a w x h texture in fmt
    HANDLE mv;                           // the motion vectors: a w x h RG16F texture
    HANDLE out;                          // the generated frame: a w x h texture in fmt
    HANDLE fence;                        // the shared D3D12 fence both sides wait on / signal
    uint64_t inBytes, mvBytes, outBytes; // the textures' allocation sizes (the external-memory import size)
    uint32_t rowPitch, mvRowPitch; // a linear copy's row pitches (D3D12's copy footprint), for the caller's buffers
};

namespace
{
constexpr int kLists = 2;
constexpr int kWrapLists = 3;
constexpr int kMaxInstances = 16;
constexpr uint32_t kMaxBuffers = 8;

std::string g_err;
std::mutex g_logLk;
std::string g_lastXefg; // XeSS-FG's / XeLL's newest warning or error message, appended to a failure's text
std::string g_provider;

template <class T> void release(T*& p)
{
    if (p)
    {
        p->Release();
        p = nullptr;
    }
}

void xefgLog(const char* msg, xefg_swapchain_logging_level_t, void*)
{
    std::lock_guard<std::mutex> g(g_logLk);
    g_lastXefg = msg ? msg : "";
}

void xellLog(const char* msg, xell_logging_level_t)
{
    std::lock_guard<std::mutex> g(g_logLk);
    g_lastXefg = msg ? msg : "";
}

bool fail(const char* what, HRESULT hr = S_OK)
{
    char b[512];
    if (hr != S_OK)
        snprintf(b, sizeof b, "%s (hr 0x%08lx)", what, (unsigned long)hr);
    else
        snprintf(b, sizeof b, "%s", what);
    g_err = b;
    return false;
}

bool failX(const char* what, int r)
{
    char b[512];
    std::lock_guard<std::mutex> g(g_logLk);
    snprintf(b, sizeof b, "%s (%d)%s%s", what, r, g_lastXefg.empty() ? "" : ": ", g_lastXefg.c_str());
    g_err = b;
    return false;
}

D3D12_RESOURCE_BARRIER transition(ID3D12Resource* r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b)
{
    D3D12_RESOURCE_BARRIER x = {};
    x.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    x.Transition.pResource = r;
    x.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    x.Transition.StateBefore = a;
    x.Transition.StateAfter = b;
    return x;
}

ID3D12Resource* texture(ID3D12Device* d, DXGI_FORMAT f, uint32_t w, uint32_t h, bool shared, D3D12_RESOURCE_STATES st,
                        bool uav = false)
{
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = w;
    rd.Height = h;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = f;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    rd.Flags = uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
    ID3D12Resource* r = nullptr;
    if (FAILED(d->CreateCommittedResource(&hp, shared ? D3D12_HEAP_FLAG_SHARED : D3D12_HEAP_FLAG_NONE, &rd, st, nullptr,
                                          IID_PPV_ARGS(&r))))
        return nullptr;
    return r;
}

// the conversion pass: dst = src, clamped to 0..1 (the typed UAV store converts to dst's format: UNORM8 rounds)
const char kConvHlsl[] = "Texture2D<float4> src : register(t0);\n"
                         "RWTexture2D<float4> dst : register(u0);\n"
                         "[numthreads(8, 8, 1)] void main(uint3 id : SV_DispatchThreadID)\n"
                         "{\n"
                         "    dst[id.xy] = saturate(src[id.xy]);\n"
                         "}\n";
std::mutex g_csLk;
ID3DBlob* g_cs = nullptr; // the pass's bytecode, compiled once a process

bool compileConv()
{
    std::lock_guard<std::mutex> g(g_csLk);
    if (g_cs)
        return true;
    HMODULE m = LoadLibraryExW(L"d3dcompiler_47.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    pD3DCompile compile = m ? (pD3DCompile)GetProcAddress(m, "D3DCompile") : nullptr;
    if (!compile)
        return fail("the system's d3dcompiler_47.dll did not load");
    ID3DBlob* err = nullptr;
    const HRESULT hr = compile(kConvHlsl, sizeof(kConvHlsl) - 1, "smv_xefg_conv", nullptr, nullptr, "main", "cs_5_0",
                               D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &g_cs, &err);
    release(err);
    if (FAILED(hr) || !g_cs)
    {
        release(g_cs);
        return fail("the conversion pass did not compile", hr);
    }
    return true;
}

ID3D12Resource* buffer(ID3D12Device* d, D3D12_HEAP_TYPE t, uint64_t bytes, D3D12_RESOURCE_STATES st)
{
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = t;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = bytes;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource* r = nullptr;
    if (FAILED(d->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, st, nullptr, IID_PPV_ARGS(&r))))
        return nullptr;
    return r;
}

struct Instance;

// ---- the real swap chain XeSS-FG presents to: each present's back buffer goes to the instance's output first --------
class SwapWrap : public IDXGISwapChain4
{
  public:
    SwapWrap(IDXGISwapChain4* real, ID3D12CommandQueue* q, IUnknown* factory, Instance* inst);
    virtual ~SwapWrap();

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** pp) override
    {
        if (!pp)
            return E_POINTER;
        if (riid == __uuidof(IUnknown) || riid == __uuidof(IDXGIObject) || riid == __uuidof(IDXGIDeviceSubObject) ||
            riid == __uuidof(IDXGISwapChain) || riid == __uuidof(IDXGISwapChain1) ||
            riid == __uuidof(IDXGISwapChain2) || riid == __uuidof(IDXGISwapChain3) || riid == __uuidof(IDXGISwapChain4))
        {
            *pp = static_cast<IDXGISwapChain4*>(this);
            AddRef();
            return S_OK;
        }
        return m_real->QueryInterface(riid, pp);
    }
    ULONG STDMETHODCALLTYPE AddRef() override
    {
        return ++m_ref;
    }
    ULONG STDMETHODCALLTYPE Release() override
    {
        const ULONG r = --m_ref;
        if (!r)
            delete this;
        return r;
    }
    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID n, UINT s, const void* d) override
    {
        return m_real->SetPrivateData(n, s, d);
    }
    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID n, const IUnknown* u) override
    {
        return m_real->SetPrivateDataInterface(n, u);
    }
    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID n, UINT* s, void* d) override
    {
        return m_real->GetPrivateData(n, s, d);
    }
    HRESULT STDMETHODCALLTYPE GetParent(REFIID riid, void** pp) override
    {
        if (m_fac)
            return m_fac->QueryInterface(riid, pp);
        return m_real->GetParent(riid, pp);
    }
    HRESULT STDMETHODCALLTYPE GetDevice(REFIID riid, void** pp) override
    {
        return m_real->GetDevice(riid, pp);
    }
    HRESULT STDMETHODCALLTYPE Present(UINT sync, UINT flags) override
    {
        if (!(flags & DXGI_PRESENT_TEST))
            capture();
        return m_real->Present(sync, flags);
    }
    HRESULT STDMETHODCALLTYPE GetBuffer(UINT b, REFIID riid, void** pp) override
    {
        return m_real->GetBuffer(b, riid, pp);
    }
    HRESULT STDMETHODCALLTYPE SetFullscreenState(BOOL f, IDXGIOutput* t) override
    {
        return m_real->SetFullscreenState(f, t);
    }
    HRESULT STDMETHODCALLTYPE GetFullscreenState(BOOL* f, IDXGIOutput** t) override
    {
        return m_real->GetFullscreenState(f, t);
    }
    HRESULT STDMETHODCALLTYPE GetDesc(DXGI_SWAP_CHAIN_DESC* d) override
    {
        return m_real->GetDesc(d);
    }
    HRESULT STDMETHODCALLTYPE ResizeBuffers(UINT c, UINT w, UINT h, DXGI_FORMAT f, UINT fl) override
    {
        return m_real->ResizeBuffers(c, w, h, f, fl);
    }
    HRESULT STDMETHODCALLTYPE ResizeTarget(const DXGI_MODE_DESC* p) override
    {
        return m_real->ResizeTarget(p);
    }
    HRESULT STDMETHODCALLTYPE GetContainingOutput(IDXGIOutput** o) override
    {
        return m_real->GetContainingOutput(o);
    }
    HRESULT STDMETHODCALLTYPE GetFrameStatistics(DXGI_FRAME_STATISTICS* s) override
    {
        return m_real->GetFrameStatistics(s);
    }
    HRESULT STDMETHODCALLTYPE GetLastPresentCount(UINT* c) override
    {
        return m_real->GetLastPresentCount(c);
    }
    HRESULT STDMETHODCALLTYPE GetDesc1(DXGI_SWAP_CHAIN_DESC1* d) override
    {
        return m_real->GetDesc1(d);
    }
    HRESULT STDMETHODCALLTYPE GetFullscreenDesc(DXGI_SWAP_CHAIN_FULLSCREEN_DESC* d) override
    {
        return m_real->GetFullscreenDesc(d);
    }
    HRESULT STDMETHODCALLTYPE GetHwnd(HWND* h) override
    {
        return m_real->GetHwnd(h);
    }
    HRESULT STDMETHODCALLTYPE GetCoreWindow(REFIID riid, void** pp) override
    {
        return m_real->GetCoreWindow(riid, pp);
    }
    HRESULT STDMETHODCALLTYPE Present1(UINT sync, UINT flags, const DXGI_PRESENT_PARAMETERS* p) override
    {
        if (!(flags & DXGI_PRESENT_TEST))
            capture();
        return m_real->Present1(sync, flags, p);
    }
    BOOL STDMETHODCALLTYPE IsTemporaryMonoSupported() override
    {
        return m_real->IsTemporaryMonoSupported();
    }
    HRESULT STDMETHODCALLTYPE GetRestrictToOutput(IDXGIOutput** o) override
    {
        return m_real->GetRestrictToOutput(o);
    }
    HRESULT STDMETHODCALLTYPE SetBackgroundColor(const DXGI_RGBA* c) override
    {
        return m_real->SetBackgroundColor(c);
    }
    HRESULT STDMETHODCALLTYPE GetBackgroundColor(DXGI_RGBA* c) override
    {
        return m_real->GetBackgroundColor(c);
    }
    HRESULT STDMETHODCALLTYPE SetRotation(DXGI_MODE_ROTATION r) override
    {
        return m_real->SetRotation(r);
    }
    HRESULT STDMETHODCALLTYPE GetRotation(DXGI_MODE_ROTATION* r) override
    {
        return m_real->GetRotation(r);
    }
    HRESULT STDMETHODCALLTYPE SetSourceSize(UINT w, UINT h) override
    {
        return m_real->SetSourceSize(w, h);
    }
    HRESULT STDMETHODCALLTYPE GetSourceSize(UINT* w, UINT* h) override
    {
        return m_real->GetSourceSize(w, h);
    }
    HRESULT STDMETHODCALLTYPE SetMaximumFrameLatency(UINT m) override
    {
        return m_real->SetMaximumFrameLatency(m);
    }
    HRESULT STDMETHODCALLTYPE GetMaximumFrameLatency(UINT* m) override
    {
        return m_real->GetMaximumFrameLatency(m);
    }
    HANDLE STDMETHODCALLTYPE GetFrameLatencyWaitableObject() override
    {
        return m_real->GetFrameLatencyWaitableObject();
    }
    HRESULT STDMETHODCALLTYPE SetMatrixTransform(const DXGI_MATRIX_3X2_F* m) override
    {
        return m_real->SetMatrixTransform(m);
    }
    HRESULT STDMETHODCALLTYPE GetMatrixTransform(DXGI_MATRIX_3X2_F* m) override
    {
        return m_real->GetMatrixTransform(m);
    }
    UINT STDMETHODCALLTYPE GetCurrentBackBufferIndex() override
    {
        return m_real->GetCurrentBackBufferIndex();
    }
    HRESULT STDMETHODCALLTYPE CheckColorSpaceSupport(DXGI_COLOR_SPACE_TYPE c, UINT* s) override
    {
        return m_real->CheckColorSpaceSupport(c, s);
    }
    HRESULT STDMETHODCALLTYPE SetColorSpace1(DXGI_COLOR_SPACE_TYPE c) override
    {
        return m_real->SetColorSpace1(c);
    }
    HRESULT STDMETHODCALLTYPE ResizeBuffers1(UINT c, UINT w, UINT h, DXGI_FORMAT f, UINT fl, const UINT* nm,
                                             IUnknown* const* q) override
    {
        return m_real->ResizeBuffers1(c, w, h, f, fl, nm, q);
    }
    HRESULT STDMETHODCALLTYPE SetHDRMetaData(DXGI_HDR_METADATA_TYPE t, UINT s, void* d) override
    {
        return m_real->SetHDRMetaData(t, s, d);
    }

  private:
    void capture();

    std::atomic<ULONG> m_ref{1};
    IDXGISwapChain4* m_real;
    ID3D12CommandQueue* m_q;
    IUnknown* m_fac;
    Instance* m_inst;
    ID3D12Device* m_dev = nullptr;
    ID3D12Fence* m_priv = nullptr; // this wrapper's own list ring
    HANDLE m_ev = nullptr;
    ID3D12CommandAllocator* m_alloc[kWrapLists] = {};
    ID3D12GraphicsCommandList* m_list[kWrapLists] = {};
    uint64_t m_done[kWrapLists] = {};
    uint64_t m_next = 0;
};

// ---- the DXGI factory XeSS-FG creates its real swap chain with --------------------------------------------------
class FactoryWrap : public IDXGIFactory7
{
  public:
    FactoryWrap(IDXGIFactory7* real, Instance* inst) : m_real(real), m_inst(inst)
    {
    }
    virtual ~FactoryWrap()
    {
        m_real->Release();
    }
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** pp) override
    {
        if (!pp)
            return E_POINTER;
        if (riid == __uuidof(IUnknown) || riid == __uuidof(IDXGIObject) || riid == __uuidof(IDXGIFactory) ||
            riid == __uuidof(IDXGIFactory1) || riid == __uuidof(IDXGIFactory2) || riid == __uuidof(IDXGIFactory3) ||
            riid == __uuidof(IDXGIFactory4) || riid == __uuidof(IDXGIFactory5) || riid == __uuidof(IDXGIFactory6) ||
            riid == __uuidof(IDXGIFactory7))
        {
            *pp = static_cast<IDXGIFactory7*>(this);
            AddRef();
            return S_OK;
        }
        return m_real->QueryInterface(riid, pp);
    }
    ULONG STDMETHODCALLTYPE AddRef() override
    {
        return ++m_ref;
    }
    ULONG STDMETHODCALLTYPE Release() override
    {
        const ULONG r = --m_ref;
        if (!r)
            delete this;
        return r;
    }
    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID n, UINT s, const void* d) override
    {
        return m_real->SetPrivateData(n, s, d);
    }
    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID n, const IUnknown* u) override
    {
        return m_real->SetPrivateDataInterface(n, u);
    }
    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID n, UINT* s, void* d) override
    {
        return m_real->GetPrivateData(n, s, d);
    }
    HRESULT STDMETHODCALLTYPE GetParent(REFIID riid, void** pp) override
    {
        return m_real->GetParent(riid, pp);
    }
    HRESULT STDMETHODCALLTYPE EnumAdapters(UINT a, IDXGIAdapter** pp) override
    {
        return m_real->EnumAdapters(a, pp);
    }
    HRESULT STDMETHODCALLTYPE MakeWindowAssociation(HWND h, UINT f) override
    {
        return m_real->MakeWindowAssociation(h, f);
    }
    HRESULT STDMETHODCALLTYPE GetWindowAssociation(HWND* h) override
    {
        return m_real->GetWindowAssociation(h);
    }
    HRESULT STDMETHODCALLTYPE CreateSwapChain(IUnknown* d, DXGI_SWAP_CHAIN_DESC* desc, IDXGISwapChain** pp) override
    {
        return m_real->CreateSwapChain(d, desc, pp);
    }
    HRESULT STDMETHODCALLTYPE CreateSoftwareAdapter(HMODULE m, IDXGIAdapter** pp) override
    {
        return m_real->CreateSoftwareAdapter(m, pp);
    }
    HRESULT STDMETHODCALLTYPE EnumAdapters1(UINT a, IDXGIAdapter1** pp) override
    {
        return m_real->EnumAdapters1(a, pp);
    }
    BOOL STDMETHODCALLTYPE IsCurrent() override
    {
        return m_real->IsCurrent();
    }
    BOOL STDMETHODCALLTYPE IsWindowedStereoEnabled() override
    {
        return m_real->IsWindowedStereoEnabled();
    }
    HRESULT STDMETHODCALLTYPE CreateSwapChainForHwnd(IUnknown* dev, HWND h, const DXGI_SWAP_CHAIN_DESC1* d,
                                                     const DXGI_SWAP_CHAIN_FULLSCREEN_DESC* fs, IDXGIOutput* o,
                                                     IDXGISwapChain1** pp) override;
    HRESULT STDMETHODCALLTYPE CreateSwapChainForCoreWindow(IUnknown* d, IUnknown* w, const DXGI_SWAP_CHAIN_DESC1* desc,
                                                           IDXGIOutput* o, IDXGISwapChain1** pp) override
    {
        return m_real->CreateSwapChainForCoreWindow(d, w, desc, o, pp);
    }
    HRESULT STDMETHODCALLTYPE GetSharedResourceAdapterLuid(HANDLE h, LUID* l) override
    {
        return m_real->GetSharedResourceAdapterLuid(h, l);
    }
    HRESULT STDMETHODCALLTYPE RegisterStereoStatusWindow(HWND h, UINT m, DWORD* c) override
    {
        return m_real->RegisterStereoStatusWindow(h, m, c);
    }
    HRESULT STDMETHODCALLTYPE RegisterStereoStatusEvent(HANDLE e, DWORD* c) override
    {
        return m_real->RegisterStereoStatusEvent(e, c);
    }
    void STDMETHODCALLTYPE UnregisterStereoStatus(DWORD c) override
    {
        m_real->UnregisterStereoStatus(c);
    }
    HRESULT STDMETHODCALLTYPE RegisterOcclusionStatusWindow(HWND h, UINT m, DWORD* c) override
    {
        return m_real->RegisterOcclusionStatusWindow(h, m, c);
    }
    HRESULT STDMETHODCALLTYPE RegisterOcclusionStatusEvent(HANDLE e, DWORD* c) override
    {
        return m_real->RegisterOcclusionStatusEvent(e, c);
    }
    void STDMETHODCALLTYPE UnregisterOcclusionStatus(DWORD c) override
    {
        m_real->UnregisterOcclusionStatus(c);
    }
    HRESULT STDMETHODCALLTYPE CreateSwapChainForComposition(IUnknown* d, const DXGI_SWAP_CHAIN_DESC1* desc,
                                                            IDXGIOutput* o, IDXGISwapChain1** pp) override
    {
        return m_real->CreateSwapChainForComposition(d, desc, o, pp);
    }
    UINT STDMETHODCALLTYPE GetCreationFlags() override
    {
        return m_real->GetCreationFlags();
    }
    HRESULT STDMETHODCALLTYPE EnumAdapterByLuid(LUID l, REFIID riid, void** pp) override
    {
        return m_real->EnumAdapterByLuid(l, riid, pp);
    }
    HRESULT STDMETHODCALLTYPE EnumWarpAdapter(REFIID riid, void** pp) override
    {
        return m_real->EnumWarpAdapter(riid, pp);
    }
    HRESULT STDMETHODCALLTYPE CheckFeatureSupport(DXGI_FEATURE f, void* d, UINT s) override
    {
        return m_real->CheckFeatureSupport(f, d, s);
    }
    HRESULT STDMETHODCALLTYPE EnumAdapterByGpuPreference(UINT a, DXGI_GPU_PREFERENCE p, REFIID riid, void** pp) override
    {
        return m_real->EnumAdapterByGpuPreference(a, p, riid, pp);
    }
    HRESULT STDMETHODCALLTYPE RegisterAdaptersChangedEvent(HANDLE e, DWORD* c) override
    {
        return m_real->RegisterAdaptersChangedEvent(e, c);
    }
    HRESULT STDMETHODCALLTYPE UnregisterAdaptersChangedEvent(DWORD c) override
    {
        return m_real->UnregisterAdaptersChangedEvent(c);
    }

  private:
    std::atomic<ULONG> m_ref{1};
    IDXGIFactory7* m_real;
    Instance* m_inst;
};

struct Instance
{
    bool live = false;
    uint32_t w = 0, h = 0;
    DXGI_FORMAT fmt = DXGI_FORMAT_R16G16B16A16_FLOAT;
    ID3D12Device* dev = nullptr;
    ID3D12CommandQueue* q = nullptr;
    ID3D12CommandAllocator* alloc[kLists] = {};
    ID3D12GraphicsCommandList* list[kLists] = {};
    uint64_t done[kLists] = {};
    int next = 0;
    ID3D12Resource* depth = nullptr; // the constant depth
    ID3D12Resource* shIn = nullptr;  // the textures shared with CUDA
    ID3D12Resource* shMv = nullptr;
    ID3D12Resource* shOut = nullptr;
    ID3D12Fence* fence = nullptr; // shared with the caller: its values are the caller's
    ID3D12Fence* priv = nullptr;  // the bridge's own setup work and list ring
    uint64_t privVal = 0;
    HANDLE ev = nullptr;
    // the swap chains' format (R10G10B10A2; SMV_XEFG_BB=8 = RGBA8) and, when the shared textures' format differs, the
    // conversion pass (descriptors 0 / 1 = shIn -> in8, 2 / 3 = out8 -> shOut; in8 / out8 in the swap chains' format)
    DXGI_FORMAT bbFmt = DXGI_FORMAT_R8G8B8A8_UNORM;
    bool conv = false;
    ID3D12RootSignature* rs = nullptr;
    ID3D12PipelineState* pso = nullptr;
    ID3D12DescriptorHeap* heap = nullptr;
    UINT incr = 0;
    ID3D12Resource* in8 = nullptr;  // the RGBA8 frame the back buffer gets (rests in UNORDERED_ACCESS)
    ID3D12Resource* out8 = nullptr; // the RGBA8 generated frame's copy (rests in NON_PIXEL_SHADER_RESOURCE)
    // the window
    HWND hwnd = nullptr;
    std::thread winThread;
    // XeLL + XeSS-FG
    xell_context_handle_t xell = nullptr;
    xefg_swapchain_handle_t fg = nullptr;
    IDXGISwapChain4* proxy = nullptr;
    ID3D12Resource* proxyBuf[kMaxBuffers] = {};
    uint32_t proxyN = 0;
    UINT presentFlags = 0;
    uint32_t id = 0; // XeSS-FG / XeLL frame id of the last call (from 1)
    // the caller values the wrapped swap chain signals once the next present's frame is in shOut
    std::mutex lk;
    std::deque<uint64_t> pending;
    // counters: calls, calls XeSS-FG presented no frame for, presents with no call waiting
    std::atomic<uint64_t> calls{0}, noFrame{0}, extra{0}, captured{0};
    XefgShared sh = {};
};

Instance g_inst[kMaxInstances];

// the conversion pass from descriptor pair `first` (an SRV, then a UAV) over the instance's w x h on list l
void convDispatch(Instance& s, ID3D12GraphicsCommandList* l, int first)
{
    ID3D12DescriptorHeap* heaps[] = {s.heap};
    l->SetDescriptorHeaps(1, heaps);
    l->SetComputeRootSignature(s.rs);
    l->SetPipelineState(s.pso);
    D3D12_GPU_DESCRIPTOR_HANDLE gh = s.heap->GetGPUDescriptorHandleForHeapStart();
    gh.ptr += (UINT64)first * s.incr;
    l->SetComputeRootDescriptorTable(0, gh);
    l->Dispatch((s.w + 7) / 8, (s.h + 7) / 8, 1);
}

// the conversion pass's root signature, pipeline, descriptors and RGBA8 textures
bool convSetup(Instance& s)
{
    if (!compileConv())
        return false;
    D3D12_DESCRIPTOR_RANGE r[2] = {};
    r[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    r[0].NumDescriptors = 1;
    r[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    r[1].NumDescriptors = 1;
    r[1].OffsetInDescriptorsFromTableStart = 1;
    D3D12_ROOT_PARAMETER p = {};
    p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    p.DescriptorTable.NumDescriptorRanges = 2;
    p.DescriptorTable.pDescriptorRanges = r;
    p.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rd = {};
    rd.NumParameters = 1;
    rd.pParameters = &p;
    ID3DBlob* blob = nullptr;
    ID3DBlob* err = nullptr;
    HRESULT hr = D3D12SerializeRootSignature(&rd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err);
    release(err);
    if (FAILED(hr) || FAILED(hr = s.dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                                             IID_PPV_ARGS(&s.rs))))
        return release(blob), fail("the conversion pass's root signature failed", hr);
    release(blob);
    D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
    pd.pRootSignature = s.rs;
    pd.CS.pShaderBytecode = g_cs->GetBufferPointer();
    pd.CS.BytecodeLength = g_cs->GetBufferSize();
    if (FAILED(hr = s.dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&s.pso))))
        return fail("the conversion pass's pipeline failed", hr);
    D3D12_DESCRIPTOR_HEAP_DESC hd = {};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.NumDescriptors = 4;
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(hr = s.dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&s.heap))))
        return fail("the conversion pass's descriptor heap failed", hr);
    s.incr = s.dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    s.in8 = texture(s.dev, s.bbFmt, s.w, s.h, false, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true);
    s.out8 = texture(s.dev, s.bbFmt, s.w, s.h, false, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (!s.in8 || !s.out8)
        return fail("the conversion pass's textures failed");
    D3D12_CPU_DESCRIPTOR_HANDLE c = s.heap->GetCPUDescriptorHandleForHeapStart();
    auto srv = [&](ID3D12Resource* t, DXGI_FORMAT f) {
        D3D12_SHADER_RESOURCE_VIEW_DESC v = {};
        v.Format = f;
        v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        v.Texture2D.MipLevels = 1;
        s.dev->CreateShaderResourceView(t, &v, c);
        c.ptr += s.incr;
    };
    auto uav = [&](ID3D12Resource* t, DXGI_FORMAT f) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC v = {};
        v.Format = f;
        v.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        s.dev->CreateUnorderedAccessView(t, nullptr, &v, c);
        c.ptr += s.incr;
    };
    srv(s.shIn, s.fmt);
    uav(s.in8, s.bbFmt);
    srv(s.out8, s.bbFmt);
    uav(s.shOut, s.fmt);
    s.conv = true;
    return true;
}

SwapWrap::SwapWrap(IDXGISwapChain4* real, ID3D12CommandQueue* q, IUnknown* factory, Instance* inst)
    : m_real(real), m_q(q), m_fac(factory), m_inst(inst)
{
    m_q->AddRef();
    m_q->GetDevice(IID_PPV_ARGS(&m_dev));
    m_dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_priv));
    m_ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    for (int k = 0; k < kWrapLists; ++k)
        if (SUCCEEDED(m_dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_alloc[k]))) &&
            SUCCEEDED(m_dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_alloc[k], nullptr,
                                               IID_PPV_ARGS(&m_list[k]))))
            m_list[k]->Close();
    if (m_fac)
        m_fac->AddRef();
}

SwapWrap::~SwapWrap()
{
    if (m_priv && m_ev && m_priv->GetCompletedValue() < m_next)
    {
        m_priv->SetEventOnCompletion(m_next, m_ev);
        WaitForSingleObject(m_ev, 10000);
    }
    for (int k = 0; k < kWrapLists; ++k)
    {
        release(m_list[k]);
        release(m_alloc[k]);
    }
    m_real->Release();
    m_q->Release();
    release(m_priv);
    release(m_dev);
    if (m_ev)
        CloseHandle(m_ev);
    if (m_fac)
        m_fac->Release();
}

// the present's back buffer -> the instance's output texture, then the oldest waiting caller value; a present with no
// call waiting (none expected: SHOW_ONLY_INTERPOLATION presents once a call) is only counted
void SwapWrap::capture()
{
    Instance& s = *m_inst;
    uint64_t v = 0;
    {
        std::lock_guard<std::mutex> g(s.lk);
        if (s.pending.empty())
        {
            ++s.extra;
            return;
        }
        v = s.pending.front();
        s.pending.pop_front();
    }
    const int k = (int)(m_next % kWrapLists);
    if (m_priv->GetCompletedValue() < m_done[k])
    {
        m_priv->SetEventOnCompletion(m_done[k], m_ev);
        WaitForSingleObject(m_ev, 10000);
    }
    ID3D12Resource* bb = nullptr;
    if (!m_list[k] || FAILED(m_real->GetBuffer(m_real->GetCurrentBackBufferIndex(), IID_PPV_ARGS(&bb))) ||
        FAILED(m_alloc[k]->Reset()) || FAILED(m_list[k]->Reset(m_alloc[k], nullptr)))
    {
        release(bb);
        m_q->Signal(s.fence, v); // the caller never hangs: its value comes, the output keeps the previous frame
        return;
    }
    ID3D12GraphicsCommandList* l = m_list[k];
    D3D12_RESOURCE_BARRIER b[2];
    if (s.conv)
    {
        // RGBA8 back buffer -> out8 -> the conversion pass -> RGBA16F shOut
        b[0] = transition(bb, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_SOURCE);
        b[1] = transition(s.out8, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
        l->ResourceBarrier(2, b);
        l->CopyResource(s.out8, bb);
        b[0] = transition(bb, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PRESENT);
        b[1] = transition(s.out8, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        l->ResourceBarrier(2, b);
        b[0] = transition(s.shOut, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        l->ResourceBarrier(1, b);
        convDispatch(s, l, 2);
        b[0] = transition(s.shOut, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        l->ResourceBarrier(1, b);
    }
    else
    {
        b[0] = transition(bb, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_SOURCE);
        b[1] = transition(s.shOut, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        l->ResourceBarrier(2, b);
        l->CopyResource(s.shOut, bb);
        b[0] = transition(bb, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PRESENT);
        b[1] = transition(s.shOut, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
        l->ResourceBarrier(2, b);
    }
    l->Close();
    ID3D12CommandList* lists[] = {l};
    m_q->ExecuteCommandLists(1, lists);
    m_q->Signal(s.fence, v);
    m_done[k] = ++m_next;
    m_q->Signal(m_priv, m_done[k]);
    bb->Release();
    ++s.captured;
}

HRESULT FactoryWrap::CreateSwapChainForHwnd(IUnknown* dev, HWND h, const DXGI_SWAP_CHAIN_DESC1* d,
                                            const DXGI_SWAP_CHAIN_FULLSCREEN_DESC* fs, IDXGIOutput* o,
                                            IDXGISwapChain1** pp)
{
    IDXGISwapChain1* sc1 = nullptr;
    HRESULT hr = m_real->CreateSwapChainForHwnd(dev, h, d, fs, o, &sc1);
    if (FAILED(hr))
        return hr;
    IDXGISwapChain4* sc4 = nullptr;
    ID3D12CommandQueue* q = nullptr;
    if (FAILED(sc1->QueryInterface(IID_PPV_ARGS(&sc4))) || FAILED(dev->QueryInterface(IID_PPV_ARGS(&q))))
    {
        // not a D3D12 swap chain: handed back unwrapped (no frame reaches the output; the counters show it)
        release(sc4);
        *pp = sc1;
        return hr;
    }
    sc1->Release();
    SwapWrap* w = new SwapWrap(sc4, q, static_cast<IDXGIFactory7*>(this), m_inst);
    q->Release();
    *pp = w;
    return hr;
}

// ---- the parked window, on its own message thread ---------------------------------------------------------------
LRESULT CALLBACK wndProc(HWND h, UINT m, WPARAM w, LPARAM l)
{
    if (m == WM_MOUSEACTIVATE)
        return MA_NOACTIVATE;
    if (m == WM_DESTROY)
    {
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(h, m, w, l);
}

bool openWindow(Instance& s)
{
    HANDLE ready = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!ready)
        return fail("CreateEvent failed"), false;
    HWND made = nullptr;
    const int w = (int)s.w, h = (int)s.h;
    s.winThread = std::thread([&made, ready, w, h] {
        static std::once_flag once;
        const HINSTANCE hi = GetModuleHandleW(nullptr);
        std::call_once(once, [hi] {
            WNDCLASSW wc = {};
            wc.lpfnWndProc = wndProc;
            wc.hInstance = hi;
            wc.lpszClassName = L"smv_xefg_bridge";
            RegisterClassW(&wc);
        });
        const int parkX = GetSystemMetrics(SM_CXSCREEN) - 1, parkY = GetSystemMetrics(SM_CYSCREEN) - 1;
        HWND hw = CreateWindowExW(WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW, L"smv_xefg_bridge", L"SmoothMyVideo XeSS-FG",
                                  WS_POPUP, parkX, parkY, w, h, nullptr, nullptr, hi, nullptr);
        if (hw)
        {
            SetWindowDisplayAffinity(hw, WDA_EXCLUDEFROMCAPTURE);
            ShowWindow(hw, SW_SHOWNOACTIVATE);
        }
        made = hw;
        SetEvent(ready);
        if (!hw)
            return;
        MSG m;
        while (GetMessageW(&m, nullptr, 0, 0) > 0)
            DispatchMessageW(&m);
    });
    const bool ok = WaitForSingleObject(ready, 10000) == WAIT_OBJECT_0;
    CloseHandle(ready);
    s.hwnd = ok ? made : nullptr;
    if (!s.hwnd)
    {
        if (s.winThread.joinable())
            s.winThread.join();
        return fail("the parked window was not created"), false;
    }
    return true;
}

void closeWindow(Instance& s)
{
    if (s.hwnd)
        PostMessageW(s.hwnd, WM_CLOSE, 0, 0);
    if (s.winThread.joinable())
        s.winThread.join();
    s.hwnd = nullptr;
}

// the queue's work so far finished (10 s at most)
void drain(Instance& s)
{
    if (!s.q || !s.priv || !s.ev)
        return;
    const uint64_t v = ++s.privVal;
    if (SUCCEEDED(s.q->Signal(s.priv, v)) && SUCCEEDED(s.priv->SetEventOnCompletion(v, s.ev)))
        WaitForSingleObject(s.ev, 10000);
}

void destroy(Instance& s)
{
    if (s.fg)
        xefgSwapChainSetEnabled(s.fg, 0);
    {
        // a caller value still waiting for its present comes now (a GPU wait on it never hangs)
        std::lock_guard<std::mutex> g(s.lk);
        if (!s.pending.empty() && s.q && s.fence)
            s.q->Signal(s.fence, s.pending.back());
        s.pending.clear();
    }
    drain(s);
    for (uint32_t k = 0; k < kMaxBuffers; ++k)
        release(s.proxyBuf[k]);
    s.proxyN = 0;
    release(s.proxy);
    if (s.fg)
    {
        xefgSwapChainDestroy(s.fg);
        s.fg = nullptr;
    }
    if (s.xell)
    {
        xellDestroyContext(s.xell);
        s.xell = nullptr;
    }
    closeWindow(s);
    for (HANDLE* hnd : {&s.sh.in, &s.sh.mv, &s.sh.out, &s.sh.fence})
        if (*hnd)
        {
            CloseHandle(*hnd);
            *hnd = nullptr;
        }
    for (int k = 0; k < kLists; ++k)
    {
        release(s.list[k]);
        release(s.alloc[k]);
        s.done[k] = 0;
    }
    release(s.depth);
    release(s.shIn);
    release(s.shMv);
    release(s.shOut);
    release(s.in8);
    release(s.out8);
    release(s.heap);
    release(s.pso);
    release(s.rs);
    s.conv = false;
    release(s.fence);
    release(s.priv);
    release(s.q);
    release(s.dev);
    if (s.ev)
    {
        CloseHandle(s.ev);
        s.ev = nullptr;
    }
    {
        std::lock_guard<std::mutex> g(s.lk);
        s.pending.clear();
    }
    s.privVal = 0;
    s.id = 0;
    s.calls = s.noFrame = s.extra = s.captured = 0;
    s.live = false;
}

// a list from the ring, its previous use finished
ID3D12GraphicsCommandList* openList(Instance& s, int& k)
{
    k = s.next;
    s.next = (k + 1) % kLists;
    if (s.priv->GetCompletedValue() < s.done[k])
    {
        if (FAILED(s.priv->SetEventOnCompletion(s.done[k], s.ev)) || WaitForSingleObject(s.ev, 10000) != WAIT_OBJECT_0)
        {
            fail("an earlier XeSS-FG call did not finish within 10 s");
            return nullptr;
        }
    }
    if (FAILED(s.alloc[k]->Reset()) || FAILED(s.list[k]->Reset(s.alloc[k], nullptr)))
    {
        fail("command list Reset failed");
        return nullptr;
    }
    return s.list[k];
}
} // namespace

XEFG_API const char* xefg_last_error()
{
    return g_err.c_str();
}

// XeSS-FG keeps no shader cache file: the folder is unused (the FSR bridge's API)
XEFG_API void xefg_set_cache_dir(const char*)
{
}

// the XeSS-FG swap chain and XeLL versions ("XeSS-FG 1.3.1, XeLL 1.3.2"), empty before the first xefg_create_i
XEFG_API const char* xefg_provider()
{
    return g_provider.c_str();
}

// 0: every instance runs on the adapter's one D3D12 device (the host keeps their calls in turn)
XEFG_API int xefg_devices()
{
    return 0;
}

// instance i's counters since its creation: calls, calls XeSS-FG presented no frame for (the output kept the previous
// frame), presents with no call waiting, frames copied out. 0 = ok
XEFG_API int xefg_stats(int i, uint64_t* calls, uint64_t* noFrame, uint64_t* extra, uint64_t* captured)
{
    if (i < 0 || i >= kMaxInstances || !g_inst[i].live)
        return -1;
    const Instance& s = g_inst[i];
    if (calls)
        *calls = s.calls;
    if (noFrame)
        *noFrame = s.noFrame;
    if (extra)
        *extra = s.extra;
    if (captured)
        *captured = s.captured;
    return 0;
}

// instance i (0..kMaxInstances - 1) at w x h on the adapter with this LUID; fmt 0 = RGBA8, 1 = RGBA16F. Fills *sh (the
// handles stay owned by the bridge until xefg_destroy_i). 0 = ok, < 0 = xefg_last_error.
XEFG_API int xefg_create_i(int i, uint32_t w, uint32_t h, uint32_t luidLow, int32_t luidHigh, int fmt, XefgShared* sh)
{
    if (i < 0 || i >= kMaxInstances || !sh || !w || !h)
        return fail("bad arguments"), -1;
    Instance& s = g_inst[i];
    if (s.live)
        destroy(s);
    s.w = w;
    s.h = h;
    s.fmt = fmt ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_R8G8B8A8_UNORM;
    IDXGIFactory7* fac = nullptr;
    HRESULT hr = CreateDXGIFactory2(0, IID_PPV_ARGS(&fac));
    if (FAILED(hr))
        return fail("CreateDXGIFactory2 failed", hr), -2;
    LUID luid;
    luid.LowPart = luidLow;
    luid.HighPart = luidHigh;
    IDXGIAdapter1* ad = nullptr;
    hr = fac->EnumAdapterByLuid(luid, IID_PPV_ARGS(&ad));
    if (FAILED(hr))
        return release(fac), fail("no DXGI adapter with the CUDA device's LUID", hr), -3;
    hr = D3D12CreateDevice(ad, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&s.dev));
    release(ad);
    if (FAILED(hr))
        return release(fac), fail("D3D12CreateDevice failed", hr), -4;
    BOOL tearing = FALSE;
    if (FAILED(fac->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &tearing, sizeof tearing)))
        tearing = FALSE;
    s.presentFlags = tearing ? DXGI_PRESENT_ALLOW_TEARING : 0;
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(hr = s.dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&s.q))))
        return release(fac), destroy(s), fail("CreateCommandQueue failed", hr), -5;
    for (int k = 0; k < kLists; ++k)
    {
        if (FAILED(hr = s.dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&s.alloc[k]))) ||
            FAILED(hr = s.dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, s.alloc[k], nullptr,
                                                 IID_PPV_ARGS(&s.list[k]))))
            return release(fac), destroy(s), fail("command list creation failed", hr), -6;
        s.list[k]->Close();
    }
    s.ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (FAILED(hr = s.dev->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&s.fence))) ||
        FAILED(hr = s.dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&s.priv))))
        return release(fac), destroy(s), fail("fence creation failed", hr), -7;

    // the shared textures rest in COMMON between calls, the state CUDA reads and writes them in; at fmt 1 the
    // conversion pass writes shOut (a UAV)
    {
        char bv[8] = {};
        s.bbFmt = GetEnvironmentVariableA("SMV_XEFG_BB", bv, sizeof bv) > 0 && !strcmp(bv, "8")
                      ? DXGI_FORMAT_R8G8B8A8_UNORM
                      : DXGI_FORMAT_R10G10B10A2_UNORM;
    }
    const bool conv = s.fmt != s.bbFmt;
    s.depth = texture(s.dev, DXGI_FORMAT_R32_FLOAT, w, h, false, D3D12_RESOURCE_STATE_COPY_DEST);
    s.shIn = texture(s.dev, s.fmt, w, h, true, D3D12_RESOURCE_STATE_COMMON);
    s.shMv = texture(s.dev, DXGI_FORMAT_R16G16_FLOAT, w, h, true, D3D12_RESOURCE_STATE_COMMON);
    s.shOut = texture(s.dev, s.fmt, w, h, true, D3D12_RESOURCE_STATE_COMMON, conv);
    if (!s.depth || !s.shIn || !s.shMv || !s.shOut)
        return release(fac), destroy(s), fail("texture creation failed"), -8;
    if (conv && !convSetup(s))
        return release(fac), destroy(s), -8;
    if (FAILED(hr = s.dev->CreateSharedHandle(s.shIn, nullptr, GENERIC_ALL, nullptr, &s.sh.in)) ||
        FAILED(hr = s.dev->CreateSharedHandle(s.shMv, nullptr, GENERIC_ALL, nullptr, &s.sh.mv)) ||
        FAILED(hr = s.dev->CreateSharedHandle(s.shOut, nullptr, GENERIC_ALL, nullptr, &s.sh.out)) ||
        FAILED(hr = s.dev->CreateSharedHandle(s.fence, nullptr, GENERIC_ALL, nullptr, &s.sh.fence)))
        return release(fac), destroy(s), fail("a shared handle's creation failed", hr), -9;
    const D3D12_RESOURCE_DESC cd = s.shIn->GetDesc(), md = s.shMv->GetDesc(), od = s.shOut->GetDesc();
    s.sh.inBytes = s.dev->GetResourceAllocationInfo(0, 1, &cd).SizeInBytes;
    s.sh.outBytes = s.dev->GetResourceAllocationInfo(0, 1, &od).SizeInBytes;
    s.sh.mvBytes = s.dev->GetResourceAllocationInfo(0, 1, &md).SizeInBytes;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fpColor = {}, fpMv = {};
    s.dev->GetCopyableFootprints(&cd, 0, 1, 0, &fpColor, nullptr, nullptr, nullptr);
    s.dev->GetCopyableFootprints(&md, 0, 1, 0, &fpMv, nullptr, nullptr, nullptr);
    s.sh.rowPitch = fpColor.Footprint.RowPitch;
    s.sh.mvRowPitch = fpMv.Footprint.RowPitch;

    // the constant depth (0.5, as the FSR bridge's), uploaded once
    {
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp = {};
        UINT64 bytes = 0;
        const D3D12_RESOURCE_DESC dd = s.depth->GetDesc();
        s.dev->GetCopyableFootprints(&dd, 0, 1, 0, &fp, nullptr, nullptr, &bytes);
        ID3D12Resource* up = buffer(s.dev, D3D12_HEAP_TYPE_UPLOAD, bytes, D3D12_RESOURCE_STATE_GENERIC_READ);
        uint8_t* p = nullptr;
        if (!up || FAILED(up->Map(0, nullptr, (void**)&p)))
            return release(up), release(fac), destroy(s), fail("depth upload failed"), -10;
        for (uint32_t y = 0; y < h; ++y)
        {
            float* row = (float*)(p + fp.Offset + (size_t)y * fp.Footprint.RowPitch);
            for (uint32_t x = 0; x < w; ++x)
                row[x] = 0.5f;
        }
        up->Unmap(0, nullptr);
        ID3D12GraphicsCommandList* l = s.list[0];
        if (FAILED(s.alloc[0]->Reset()) || FAILED(l->Reset(s.alloc[0], nullptr)))
            return release(up), release(fac), destroy(s), fail("command list Reset failed"), -10;
        D3D12_TEXTURE_COPY_LOCATION dst = {}, src = {};
        dst.pResource = s.depth;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.pResource = up;
        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src.PlacedFootprint = fp;
        l->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        D3D12_RESOURCE_BARRIER b =
            transition(s.depth, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        l->ResourceBarrier(1, &b);
        l->Close();
        ID3D12CommandList* lists[] = {l};
        s.q->ExecuteCommandLists(1, lists);
        drain(s);
        release(up);
    }

    if (!openWindow(s))
        return release(fac), destroy(s), -11;

    // XeLL (XeSS-FG requires it), then XeSS-FG on our factory
    xell_result_t lr = xellD3D12CreateContext(s.dev, &s.xell);
    if (lr != XELL_RESULT_SUCCESS)
        return release(fac), destroy(s), failX("xellD3D12CreateContext failed", (int)lr), -12;
    xellSetLoggingCallback(s.xell, XELL_LOGGING_LEVEL_WARNING, xellLog);
    xell_sleep_params_t sp = {};
    sp.bLowLatencyMode = 1;
    if ((lr = xellSetSleepMode(s.xell, &sp)) != XELL_RESULT_SUCCESS)
        return release(fac), destroy(s), failX("xellSetSleepMode failed", (int)lr), -12;
    xefg_swapchain_result_t fr = xefgSwapChainD3D12CreateContext(s.dev, &s.fg);
    if (fr != XEFG_SWAPCHAIN_RESULT_SUCCESS)
        return release(fac), destroy(s), failX("xefgSwapChainD3D12CreateContext failed", (int)fr), -13;
    xefgSwapChainSetLoggingCallback(s.fg, XEFG_SWAPCHAIN_LOGGING_LEVEL_WARNING, xefgLog, nullptr);
    if ((fr = xefgSwapChainSetLatencyReduction(s.fg, s.xell)) != XEFG_SWAPCHAIN_RESULT_SUCCESS)
        return release(fac), destroy(s), failX("xefgSwapChainSetLatencyReduction failed", (int)fr), -14;
    DXGI_SWAP_CHAIN_DESC1 sd = {};
    sd.Width = w;
    sd.Height = h;
    sd.Format = s.bbFmt;
    sd.SampleDesc.Count = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.BufferCount = 3;
    sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    sd.Flags = tearing ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;
    FactoryWrap* fw = new FactoryWrap(fac, &s); // takes the factory reference
    xefg_swapchain_d3d12_init_params_t ip = {};
    ip.maxInterpolatedFrames = 1;
    ip.uiMode = XEFG_SWAPCHAIN_UI_MODE_NONE;
    fr = xefgSwapChainD3D12InitFromSwapChainDesc(s.fg, s.hwnd, &sd, nullptr, s.q, fw, &ip);
    fw->Release(); // XeSS-FG keeps its own references
    if (fr != XEFG_SWAPCHAIN_RESULT_SUCCESS)
        return destroy(s), failX("xefgSwapChainD3D12InitFromSwapChainDesc failed", (int)fr), -15;
    if ((fr = xefgSwapChainD3D12GetSwapChainPtr(s.fg, IID_PPV_ARGS(&s.proxy))) != XEFG_SWAPCHAIN_RESULT_SUCCESS ||
        !s.proxy)
        return destroy(s), failX("xefgSwapChainD3D12GetSwapChainPtr failed", (int)fr), -16;
    DXGI_SWAP_CHAIN_DESC1 pd = {};
    s.proxy->GetDesc1(&pd);
    s.proxyN = pd.BufferCount < kMaxBuffers ? pd.BufferCount : kMaxBuffers;
    for (uint32_t k = 0; k < s.proxyN; ++k)
        if (FAILED(hr = s.proxy->GetBuffer(k, IID_PPV_ARGS(&s.proxyBuf[k]))))
            return destroy(s), fail("the proxy swap chain's GetBuffer failed", hr), -17;
    fr = xefgSwapChainEnableDebugFeature(s.fg, XEFG_SWAPCHAIN_DEBUG_FEATURE_SHOW_ONLY_INTERPOLATION, 1, nullptr);
    if (fr != XEFG_SWAPCHAIN_RESULT_SUCCESS)
        return destroy(s), failX("XeSS-FG's only-interpolated-frames mode was refused", (int)fr), -18;
    if ((fr = xefgSwapChainSetEnabled(s.fg, 1)) != XEFG_SWAPCHAIN_RESULT_SUCCESS)
        return destroy(s), failX("xefgSwapChainSetEnabled failed", (int)fr), -19;
    xefg_swapchain_properties_t props = {};
    if (xefgSwapChainGetProperties(s.fg, &props) == XEFG_SWAPCHAIN_RESULT_SUCCESS &&
        props.maxSupportedInterpolations < 1)
        return destroy(s), fail("XeSS-FG offers no interpolated frame on this GPU"), -20;
    if (g_provider.empty())
    {
        xefg_swapchain_version_t fv = {};
        xell_version_t lv = {};
        xefgSwapChainGetVersion(&fv);
        xellGetVersion(&lv);
        char b[96];
        snprintf(b, sizeof b, "XeSS-FG %u.%u.%u, XeLL %u.%u.%u", fv.major, fv.minor, fv.patch, lv.major, lv.minor,
                 lv.patch);
        g_provider = b;
    }
    s.live = true;
    *sh = s.sh;
    return 0;
}

// instance i: the queue waits for `waitValue` (the caller wrote the frame into sh.in and its vectors into sh.mv), copies
// the frame into the proxy's back buffer, tags the vectors and presents it; the frame XeSS-FG generates halfway between
// the previous call's frame and this one lands in sh.out and `signalValue` follows. reset = the first frame of a sequence
// (XeSS-FG's history reset; no previous frame: sh.out then holds no tween). Non-blocking but for the list ring and
// XeSS-FG's own present queue. 0 = ok.
XEFG_API int xefg_frame_i(int i, uint64_t waitValue, uint64_t signalValue, int reset, double frameTimeMs)
{
    if (i < 0 || i >= kMaxInstances || !g_inst[i].live)
        return fail("instance not created"), -1;
    Instance& s = g_inst[i];
    const uint32_t id = ++s.id;
    ++s.calls;
    xellAddMarkerData(s.xell, id, XELL_SIMULATION_START);
    xellAddMarkerData(s.xell, id, XELL_SIMULATION_END);
    xellAddMarkerData(s.xell, id, XELL_RENDERSUBMIT_START);
    if (FAILED(s.q->Wait(s.fence, waitValue)))
        return fail("queue Wait failed"), -2;
    int k = 0;
    ID3D12GraphicsCommandList* l = openList(s, k);
    if (!l)
        return -3;
    ID3D12Resource* bb = s.proxyBuf[s.proxy->GetCurrentBackBufferIndex() % (s.proxyN ? s.proxyN : 1)];
    D3D12_RESOURCE_BARRIER b[3];
    if (s.conv)
    {
        // RGBA16F shIn -> the conversion pass -> in8 -> the RGBA8 back buffer
        b[0] = transition(s.shIn, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        b[1] = transition(s.shMv, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        l->ResourceBarrier(2, b);
        convDispatch(s, l, 0);
        b[0] = transition(s.in8, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        b[1] = transition(bb, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST);
        b[2] = transition(s.shIn, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
        l->ResourceBarrier(3, b);
        l->CopyResource(bb, s.in8);
        b[0] = transition(s.in8, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        b[1] = transition(bb, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT);
        l->ResourceBarrier(2, b);
    }
    else
    {
        b[0] = transition(bb, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST);
        b[1] = transition(s.shIn, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
        b[2] = transition(s.shMv, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        l->ResourceBarrier(3, b);
        l->CopyResource(bb, s.shIn);
        b[0] = transition(bb, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT);
        b[1] = transition(s.shIn, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        l->ResourceBarrier(2, b);
    }
    // the vectors are copied by XeSS-FG now (the caller rewrites them for the next call), the depth stays
    xefg_swapchain_d3d12_resource_data_t rd = {};
    rd.type = XEFG_SWAPCHAIN_RES_MOTION_VECTOR;
    rd.validity = XEFG_SWAPCHAIN_RV_ONLY_NOW;
    rd.resourceSize = {s.w, s.h};
    rd.pResource = s.shMv;
    rd.incomingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    xefg_swapchain_result_t fr = xefgSwapChainD3D12TagFrameResource(s.fg, l, id, &rd);
    if (fr != XEFG_SWAPCHAIN_RESULT_SUCCESS)
        return l->Close(), failX("tagging the motion vectors failed", (int)fr), -4;
    rd.type = XEFG_SWAPCHAIN_RES_DEPTH;
    rd.validity = XEFG_SWAPCHAIN_RV_UNTIL_NEXT_PRESENT;
    rd.pResource = s.depth;
    if ((fr = xefgSwapChainD3D12TagFrameResource(s.fg, l, id, &rd)) != XEFG_SWAPCHAIN_RESULT_SUCCESS)
        return l->Close(), failX("tagging the depth failed", (int)fr), -5;
    b[0] = transition(s.shMv, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
    l->ResourceBarrier(1, b);
    if (FAILED(l->Close()))
        return fail("command list Close failed"), -6;
    ID3D12CommandList* lists[] = {l};
    s.q->ExecuteCommandLists(1, lists);
    s.done[k] = ++s.privVal;
    s.q->Signal(s.priv, s.done[k]);

    xefg_swapchain_frame_constant_data_t fc = {};
    for (int j = 0; j < 4; ++j)
        fc.viewMatrix[j * 5] = fc.projectionMatrix[j * 5] = 1.0f;
    fc.motionVectorScaleX = fc.motionVectorScaleY = 1.0f;
    fc.resetHistory = reset != 0 ? 1u : 0u;
    fc.frameRenderTime = (float)frameTimeMs;
    if ((fr = xefgSwapChainTagFrameConstants(s.fg, id, &fc)) != XEFG_SWAPCHAIN_RESULT_SUCCESS)
        return failX("tagging the frame constants failed", (int)fr), -7;
    xellAddMarkerData(s.xell, id, XELL_RENDERSUBMIT_END);
    xellAddMarkerData(s.xell, id, XELL_PRESENT_START);
    {
        std::lock_guard<std::mutex> g(s.lk);
        s.pending.push_back(signalValue);
    }
    xefgSwapChainSetPresentId(s.fg, id);
    const HRESULT hr = s.proxy->Present(0, s.presentFlags);
    xellAddMarkerData(s.xell, id, XELL_PRESENT_END);
    xefg_swapchain_present_status_t ps = {};
    const bool status = xefgSwapChainGetLastPresentStatus(s.fg, &ps) == XEFG_SWAPCHAIN_RESULT_SUCCESS;
    if (FAILED(hr) || (status && ps.framesPresented == 0))
    {
        // no frame comes for this call (the first of a sequence, or a failed present): its value from here, sh.out
        // unchanged
        bool mine = false;
        {
            std::lock_guard<std::mutex> g(s.lk);
            if (!s.pending.empty() && s.pending.back() == signalValue)
            {
                s.pending.pop_back();
                mine = true;
            }
        }
        if (mine)
        {
            if (!reset)
                ++s.noFrame;
            s.q->Signal(s.fence, signalValue);
        }
        if (FAILED(hr))
            return fail("the proxy swap chain's Present failed", hr), -8;
    }
    return 0;
}

XEFG_API void xefg_destroy_i(int i)
{
    if (i >= 0 && i < kMaxInstances)
        destroy(g_inst[i]);
}

XEFG_API void xefg_destroy()
{
    for (Instance& s : g_inst)
        destroy(s);
}
