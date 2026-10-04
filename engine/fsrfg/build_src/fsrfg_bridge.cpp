// SmoothMyVideo's AMD FSR frame generation bridge: AMD's signed FidelityFX SDK v2.3.0 frame generation DLL
// (amd_fidelityfx_framegeneration_dx12.dll 4.0.1, unmodified), its ML provider (FSR 4) only, behind a small C API for
// the native host. FSR 4's shaders use AMD's WMMA FP8 driver intrinsics, so its D3D12 device comes from vkd3d-proton
// (D3D12 on Vulkan, built from source with ..\source's patches: FP8 matrices emulated on FP16 cooperative matrices),
// and amdxc64.dll beside this file (amdxc64_shim.cpp) answers the DLL's driver-extension queries. vkd3d-proton's two
// DLLs are renamed smv_vkd3d_*.dll so they never share a module name with the system D3D12 in the host's process.
// Frames go in and out through three D3D12 textures shared with CUDA (vkd3d-proton shares textures, not buffers: NT
// handles the host imports as external memory and maps as CUDA arrays) and one shared fence (vkd3d-proton exports it
// as a Vulkan timeline semaphore, which CUDA imports), so the GPU waits on the GPU: the host copies the frame and its
// motion vectors in, signals `wait`; the bridge's queue waits for it, runs configure -> prepare V2 -> generate on the
// shared textures and signals `signal`; the host copies the generated frame out. One context per instance; every call
// feeds the next real frame (frameID + 1), the generated frame lies halfway between the previous call's frame and this.
// Formats: fmt 0 = RGBA8 UNORM (sRGB-encoded bytes), fmt 1 = RGBA16F (sRGB-encoded values 0..1); motion vectors =
// RG16F in pixels, current -> previous (FSR's convention); depth = a constant (no depth in video).
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include "ffx_api.h"
#include "ffx_api_loader.h"
#include "dx12/ffx_api_dx12.h"
#include "ffx_framegeneration.h"

#define FSRFG_API extern "C" __declspec(dllexport)

struct FsrfgShared
{
    HANDLE in;                           // the real frame: a w x h texture in fmt
    HANDLE mv;                           // the motion vectors: a w x h RG16F texture
    HANDLE out;                          // the generated frame: a w x h texture in fmt
    HANDLE fence;                        // the shared fence (a Vulkan timeline semaphore) both sides wait on / signal
    uint64_t inBytes, mvBytes, outBytes; // the textures' allocation sizes (the external-memory import size)
    uint32_t rowPitch, mvRowPitch; // a linear copy's row pitches (D3D12's copy footprint), for the caller's buffers
};

namespace
{
constexpr int kLists = 2;
constexpr int kMaxInstances = 4;

std::string g_err;
std::wstring g_cacheRoot; // fsrfg_set_cache_dir, else shader_cache beside this DLL
bool g_loaded = false;
PFN_D3D12_CREATE_DEVICE g_createDevice = nullptr;
ffxFunctions g_ffx = {};
uint64_t g_versionId = 0; // the FSR 4 version id the DLL offers, found at the first create
std::string g_provider;

template <class T> void release(T*& p)
{
    if (p)
    {
        p->Release();
        p = nullptr;
    }
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

struct Instance
{
    bool live = false;
    uint32_t w = 0, h = 0;
    DXGI_FORMAT fmt = DXGI_FORMAT_R8G8B8A8_UNORM;
    ID3D12Device* dev = nullptr;
    ID3D12CommandQueue* q = nullptr;
    ID3D12CommandAllocator* alloc[kLists] = {};
    ID3D12GraphicsCommandList* list[kLists] = {};
    uint64_t done[kLists] = {};
    int next = 0;
    ID3D12Resource* color = nullptr; // FSR's own textures
    ID3D12Resource* motion = nullptr;
    ID3D12Resource* depth = nullptr;
    ID3D12Resource* out = nullptr;
    ID3D12Resource* shIn = nullptr; // the textures shared with CUDA, copied from / into FSR's own
    ID3D12Resource* shMv = nullptr;
    ID3D12Resource* shOut = nullptr;
    ID3D12Fence* fence = nullptr; // shared with the caller: its values are the caller's
    ID3D12Fence* priv = nullptr;  // the bridge's own setup work
    HANDLE ev = nullptr;
    ffxContext ctx = nullptr;
    uint64_t frameID = 0;
    FsrfgShared sh = {};
};

Instance g_inst[kMaxInstances];

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

std::wstring moduleDir()
{
    HMODULE self = nullptr;
    wchar_t path[MAX_PATH] = {};
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       (LPCWSTR)&moduleDir, &self);
    GetModuleFileNameW(self, path, MAX_PATH);
    std::wstring d(path);
    return d.substr(0, d.find_last_of(L"\\/") + 1);
}

void makeDirs(const std::wstring& p)
{
    for (size_t i = p.find_first_of(L"\\/", 3); i != std::wstring::npos; i = p.find_first_of(L"\\/", i + 1))
        CreateDirectoryW(p.substr(0, i).c_str(), nullptr);
    CreateDirectoryW(p.c_str(), nullptr);
}

void setEnv(const char* name, const char* value, bool keepCallers)
{
    char b[4];
    if (keepCallers && GetEnvironmentVariableA(name, b, sizeof(b)) > 0)
        return;
    SetEnvironmentVariableA(name, value);
}

HMODULE loadBeside(const std::wstring& dir, const wchar_t* name)
{
    return LoadLibraryExW((dir + name).c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
}

// once per process: the switches of the patched vkd3d-proton, the shim, vkd3d-proton and AMD's DLLs
bool loadOnce()
{
    if (g_loaded)
        return true;
    const std::wstring dir = moduleDir();
    // vkd3d-proton's pipeline cache key does not cover DXIL_SPIRV_CONFIG, so this route keeps a cache of its own, in a
    // folder named after the vkd3d-proton build (its core DLL's size and write time): a cache of another build or
    // configuration once replayed broken pipelines
    WIN32_FILE_ATTRIBUTE_DATA fa = {};
    if (!GetFileAttributesExW((dir + L"smv_vkd3d_d3d12core.dll").c_str(), GetFileExInfoStandard, &fa))
        return fail("smv_vkd3d_d3d12core.dll is missing beside the bridge");
    wchar_t key[64];
    swprintf(key, 64, L"vkd3d_%lx_%08lx%08lx", (unsigned long)fa.nFileSizeLow,
             (unsigned long)fa.ftLastWriteTime.dwHighDateTime, (unsigned long)fa.ftLastWriteTime.dwLowDateTime);
    const std::wstring cache = (g_cacheRoot.empty() ? dir + L"shader_cache" : g_cacheRoot) + L"\\" + key;
    makeDirs(cache);
    SetEnvironmentVariableW(L"VKD3D_SHADER_CACHE_PATH", cache.c_str());
    setEnv("VKD3D_FP8_EMULATION", "1", false);
    setEnv("DXIL_SPIRV_CONFIG", "wmma_fp8_staging", false);
    setEnv("VKD3D_DEBUG", "none", true);
    setEnv("VKD3D_SHADER_DEBUG", "none", true);

    // AMD's DLL loads amdxc64.dll by name: the shim, loaded first, is the module it finds
    HMODULE shim = loadBeside(dir, L"amdxc64.dll");
    if (!shim)
        return fail("amdxc64.dll (the driver-extension shim) did not load");
    if (GetModuleHandleW(L"amdxc64.dll") != shim)
        return fail("another amdxc64.dll (an AMD driver's) is loaded in this process");
    // the core first: the wrapper loads it by name
    if (!loadBeside(dir, L"smv_vkd3d_d3d12core.dll"))
        return fail("smv_vkd3d_d3d12core.dll did not load");
    HMODULE d3d = loadBeside(dir, L"smv_vkd3d_d3d12.dll");
    g_createDevice = d3d ? (PFN_D3D12_CREATE_DEVICE)GetProcAddress(d3d, "D3D12CreateDevice") : nullptr;
    if (!g_createDevice)
        return fail("smv_vkd3d_d3d12.dll did not load or lacks D3D12CreateDevice");
    // AMD's DLL serializes its root signatures through the loaded module named D3D12.dll (found by name; without one
    // every pipeline fails to create and each dispatch is dropped): the system's, whose blobs vkd3d-proton reads
    if (!LoadLibraryExW(L"d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32))
        return fail("the system's d3d12.dll did not load");
    // the frame generation DLL first: AMD's loader loads its providers by name
    if (!loadBeside(dir, L"amd_fidelityfx_framegeneration_dx12.dll"))
        return fail("amd_fidelityfx_framegeneration_dx12.dll did not load");
    HMODULE loader = loadBeside(dir, L"amd_fidelityfx_loader_dx12.dll");
    if (!loader)
        return fail("amd_fidelityfx_loader_dx12.dll did not load");
    ffxLoadFunctions(&g_ffx, loader);
    if (!g_ffx.CreateContext || !g_ffx.DestroyContext || !g_ffx.Configure || !g_ffx.Query || !g_ffx.Dispatch)
        return fail("amd_fidelityfx_loader_dx12.dll lacks an ffx entry point");
    g_loaded = true;
    return true;
}

// the FSR 4 version among those the DLL offers on this device (a name starting "4."); 0 = none
bool findVersion(ID3D12Device* dev)
{
    if (g_versionId)
        return true;
    uint64_t count = 0;
    ffxQueryDescGetVersions gv = {};
    gv.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    gv.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
    gv.device = dev;
    gv.outputCount = &count;
    if (g_ffx.Query(nullptr, &gv.header) != FFX_API_RETURN_OK || !count || count > 16)
        return fail("AMD's frame generation DLL lists no versions");
    uint64_t ids[16] = {};
    const char* names[16] = {};
    gv.versionIds = ids;
    gv.versionNames = names;
    if (g_ffx.Query(nullptr, &gv.header) != FFX_API_RETURN_OK)
        return fail("AMD's frame generation DLL's version query failed");
    std::string offered;
    for (uint64_t k = 0; k < count; ++k)
    {
        const char* n = names[k] ? names[k] : "?";
        offered += (k ? ", " : "") + std::string(n);
        if (!g_versionId && n[0] == '4' && n[1] == '.')
        {
            g_versionId = ids[k];
            g_provider = n;
        }
    }
    if (!g_versionId)
        return fail(("AMD's frame generation DLL does not offer FSR 4 on this GPU (offered: " + offered + ")").c_str());
    return true;
}

// a w x h texture; shared = one CUDA imports too: simultaneous access keeps it in a layout both APIs read and write
ID3D12Resource* texture(ID3D12Device* d, DXGI_FORMAT f, uint32_t w, uint32_t h, bool uav, bool shared,
                        D3D12_RESOURCE_STATES st)
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
    rd.Flags = (uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE) |
               (shared ? D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS : D3D12_RESOURCE_FLAG_NONE);
    ID3D12Resource* r = nullptr;
    if (FAILED(d->CreateCommittedResource(&hp, shared ? D3D12_HEAP_FLAG_SHARED : D3D12_HEAP_FLAG_NONE, &rd, st, nullptr,
                                          IID_PPV_ARGS(&r))))
        return nullptr;
    return r;
}

ID3D12Resource* buffer(ID3D12Device* d, D3D12_HEAP_TYPE t, D3D12_HEAP_FLAGS flags, uint64_t bytes,
                       D3D12_RESOURCE_STATES st)
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
    if (FAILED(d->CreateCommittedResource(&hp, flags, &rd, st, nullptr, IID_PPV_ARGS(&r))))
        return nullptr;
    return r;
}

void destroy(Instance& s)
{
    if (s.q && s.priv && s.ev)
    {
        // drain the queue before anything it uses goes away (a queue still waiting for a caller value that never comes
        // is given up on after 10 s)
        const uint64_t v = s.priv->GetCompletedValue() + 1;
        if (SUCCEEDED(s.q->Signal(s.priv, v)) && SUCCEEDED(s.priv->SetEventOnCompletion(v, s.ev)))
            WaitForSingleObject(s.ev, 10000);
    }
    if (s.ctx)
    {
        ffxConfigureDescFrameGeneration cfg = {};
        cfg.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
        cfg.frameGenerationEnabled = false;
        cfg.flags = FFX_FRAMEGENERATION_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY;
        cfg.frameID = s.frameID;
        g_ffx.Configure(&s.ctx, &cfg.header);
        g_ffx.DestroyContext(&s.ctx, nullptr);
        s.ctx = nullptr;
    }
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
    release(s.color);
    release(s.motion);
    release(s.depth);
    release(s.out);
    release(s.shIn);
    release(s.shMv);
    release(s.shOut);
    release(s.fence);
    release(s.priv);
    release(s.q);
    release(s.dev);
    if (s.ev)
    {
        CloseHandle(s.ev);
        s.ev = nullptr;
    }
    s.live = false;
    s.frameID = 0;
}

// a list from the ring, its previous use finished (the GPU waits out only GPU work: the caller signals every value the
// queue waits for before it asks for the next frame)
ID3D12GraphicsCommandList* openList(Instance& s, int& k)
{
    k = s.next;
    s.next = (k + 1) % kLists;
    if (s.fence->GetCompletedValue() < s.done[k])
    {
        if (FAILED(s.fence->SetEventOnCompletion(s.done[k], s.ev)) || WaitForSingleObject(s.ev, 10000) != WAIT_OBJECT_0)
        {
            fail("an earlier FSR frame generation call did not finish within 10 s");
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

D3D12_TEXTURE_COPY_LOCATION subresource(ID3D12Resource* r)
{
    D3D12_TEXTURE_COPY_LOCATION l = {};
    l.pResource = r;
    l.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    l.SubresourceIndex = 0;
    return l;
}

D3D12_TEXTURE_COPY_LOCATION footprint(ID3D12Resource* r, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& fp)
{
    D3D12_TEXTURE_COPY_LOCATION l = {};
    l.pResource = r;
    l.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    l.PlacedFootprint = fp;
    return l;
}
} // namespace

FSRFG_API const char* fsrfg_last_error()
{
    return g_err.c_str();
}

// the folder vkd3d-proton's shader cache goes into (UTF-8), before the first fsrfg_create_i; unset = shader_cache beside
// this DLL
FSRFG_API void fsrfg_set_cache_dir(const char* dir)
{
    g_cacheRoot.clear();
    if (dir && *dir)
    {
        const int n = MultiByteToWideChar(CP_UTF8, 0, dir, -1, nullptr, 0);
        if (n > 1)
        {
            g_cacheRoot.resize((size_t)n - 1);
            MultiByteToWideChar(CP_UTF8, 0, dir, -1, &g_cacheRoot[0], n);
        }
    }
}

// the FSR version the contexts run ("4.0.1"), empty before the first fsrfg_create_i
FSRFG_API const char* fsrfg_provider()
{
    return g_provider.c_str();
}

// instance i (0..3) at w x h on the adapter with this LUID; fmt 0 = RGBA8, 1 = RGBA16F. Fills *sh (the handles stay
// owned by the bridge until fsrfg_destroy_i). 0 = ok, < 0 = fsrfg_last_error.
FSRFG_API int fsrfg_create_i(int i, uint32_t w, uint32_t h, uint32_t luidLow, int32_t luidHigh, int fmt,
                             FsrfgShared* sh)
{
    if (i < 0 || i >= kMaxInstances || !sh || !w || !h)
        return fail("bad arguments"), -1;
    if (!loadOnce())
        return -14;
    Instance& s = g_inst[i];
    if (s.live)
        destroy(s);
    s.w = w;
    s.h = h;
    s.fmt = fmt ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_R8G8B8A8_UNORM;
    IDXGIFactory4* fac = nullptr;
    HRESULT hr = CreateDXGIFactory2(0, IID_PPV_ARGS(&fac));
    if (FAILED(hr))
        return fail("CreateDXGIFactory2 failed", hr), -2;
    LUID luid;
    luid.LowPart = luidLow;
    luid.HighPart = luidHigh;
    IDXGIAdapter1* ad = nullptr;
    hr = fac->EnumAdapterByLuid(luid, IID_PPV_ARGS(&ad));
    release(fac);
    if (FAILED(hr))
        return fail("no DXGI adapter with the CUDA device's LUID", hr), -3;
    hr = g_createDevice(ad, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&s.dev));
    release(ad);
    if (FAILED(hr))
        return fail("vkd3d-proton's D3D12CreateDevice failed", hr), -4;
    if (!findVersion(s.dev))
        return destroy(s), -15;
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(hr = s.dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&s.q))))
        return destroy(s), fail("CreateCommandQueue failed", hr), -5;
    for (int k = 0; k < kLists; ++k)
    {
        if (FAILED(hr = s.dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&s.alloc[k]))) ||
            FAILED(hr = s.dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, s.alloc[k], nullptr,
                                                 IID_PPV_ARGS(&s.list[k]))))
            return destroy(s), fail("command list creation failed", hr), -6;
        s.list[k]->Close();
    }
    s.ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (FAILED(hr = s.dev->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&s.fence))) ||
        FAILED(hr = s.dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&s.priv))))
        return destroy(s), fail("fence creation failed", hr), -7;

    // FSR's own textures in the states its calls name; the shared ones rest in COMMON between calls, the state CUDA
    // reads and writes them in
    s.color = texture(s.dev, s.fmt, w, h, false, false, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    s.motion =
        texture(s.dev, DXGI_FORMAT_R16G16_FLOAT, w, h, false, false, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    s.depth = texture(s.dev, DXGI_FORMAT_R32_FLOAT, w, h, false, false, D3D12_RESOURCE_STATE_COPY_DEST);
    s.out = texture(s.dev, s.fmt, w, h, true, false, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    s.shIn = texture(s.dev, s.fmt, w, h, false, true, D3D12_RESOURCE_STATE_COMMON);
    s.shMv = texture(s.dev, DXGI_FORMAT_R16G16_FLOAT, w, h, false, true, D3D12_RESOURCE_STATE_COMMON);
    s.shOut = texture(s.dev, s.fmt, w, h, false, true, D3D12_RESOURCE_STATE_COMMON);
    if (!s.color || !s.motion || !s.depth || !s.out || !s.shIn || !s.shMv || !s.shOut)
        return destroy(s), fail("texture creation failed"), -8;
    if (FAILED(hr = s.dev->CreateSharedHandle(s.shIn, nullptr, GENERIC_ALL, nullptr, &s.sh.in)) ||
        FAILED(hr = s.dev->CreateSharedHandle(s.shMv, nullptr, GENERIC_ALL, nullptr, &s.sh.mv)) ||
        FAILED(hr = s.dev->CreateSharedHandle(s.shOut, nullptr, GENERIC_ALL, nullptr, &s.sh.out)))
        return destroy(s), fail("a shared texture's handle creation failed", hr), -10;
    if (FAILED(hr = s.dev->CreateSharedHandle(s.fence, nullptr, GENERIC_ALL, nullptr, &s.sh.fence)))
        return destroy(s), fail("the shared fence's handle creation failed", hr), -10;
    const D3D12_RESOURCE_DESC cd = s.color->GetDesc(), md = s.motion->GetDesc();
    s.sh.inBytes = s.sh.outBytes = s.dev->GetResourceAllocationInfo(0, 1, &cd).SizeInBytes;
    s.sh.mvBytes = s.dev->GetResourceAllocationInfo(0, 1, &md).SizeInBytes;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fpColor = {}, fpMv = {};
    s.dev->GetCopyableFootprints(&cd, 0, 1, 0, &fpColor, nullptr, nullptr, nullptr);
    s.dev->GetCopyableFootprints(&md, 0, 1, 0, &fpMv, nullptr, nullptr, nullptr);
    s.sh.rowPitch = fpColor.Footprint.RowPitch;
    s.sh.mvRowPitch = fpMv.Footprint.RowPitch;

    // the constant depth (0.5, as the harness's flat depth), uploaded once
    {
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp = {};
        UINT64 bytes = 0;
        const D3D12_RESOURCE_DESC dd = s.depth->GetDesc();
        s.dev->GetCopyableFootprints(&dd, 0, 1, 0, &fp, nullptr, nullptr, &bytes);
        ID3D12Resource* up =
            buffer(s.dev, D3D12_HEAP_TYPE_UPLOAD, D3D12_HEAP_FLAG_NONE, bytes, D3D12_RESOURCE_STATE_GENERIC_READ);
        uint8_t* p = nullptr;
        if (!up || FAILED(up->Map(0, nullptr, (void**)&p)))
            return release(up), destroy(s), fail("depth upload failed"), -11;
        for (uint32_t y = 0; y < h; ++y)
        {
            float* row = (float*)(p + fp.Offset + (size_t)y * fp.Footprint.RowPitch);
            for (uint32_t x = 0; x < w; ++x)
                row[x] = 0.5f;
        }
        up->Unmap(0, nullptr);
        ID3D12GraphicsCommandList* l = s.list[0];
        if (FAILED(s.alloc[0]->Reset()) || FAILED(l->Reset(s.alloc[0], nullptr)))
            return release(up), destroy(s), fail("command list Reset failed"), -12;
        D3D12_TEXTURE_COPY_LOCATION dst = subresource(s.depth), src = footprint(up, fp);
        l->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        D3D12_RESOURCE_BARRIER b =
            transition(s.depth, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        l->ResourceBarrier(1, &b);
        l->Close();
        ID3D12CommandList* lists[] = {l};
        s.q->ExecuteCommandLists(1, lists);
        if (FAILED(s.q->Signal(s.priv, 1)) || FAILED(s.priv->SetEventOnCompletion(1, s.ev)) ||
            WaitForSingleObject(s.ev, 10000) != WAIT_OBJECT_0)
            return release(up), destroy(s), fail("depth upload did not finish"), -12;
        release(up);
    }

    ffxCreateBackendDX12Desc backend = {};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
    backend.device = s.dev;
    ffxOverrideVersion over = {};
    over.header.type = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
    over.header.pNext = &backend.header;
    over.versionId = g_versionId;
    ffxCreateContextDescFrameGeneration create = {};
    create.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
    create.header.pNext = &over.header;
    create.flags = 0;
    create.displaySize = {w, h};
    create.maxRenderSize = {w, h};
    create.backBufferFormat = fmt ? FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT : FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM;
    if (g_ffx.CreateContext(&s.ctx, &create.header, nullptr) != FFX_API_RETURN_OK)
        return destroy(s), fail("ffxCreateContext(frame generation) failed"), -13;
    s.frameID = 0;
    s.live = true;
    *sh = s.sh;
    return 0;
}

// instance i: the queue waits for `waitValue` (the caller wrote the frame into sh.in and its vectors into sh.mv), then
// generates the frame halfway between the previous call's frame and this one into sh.out and signals `signalValue`.
// reset = the first frame of a sequence (no previous frame: sh.out then holds no tween). Non-blocking but for the
// command-list ring. 0 = ok.
FSRFG_API int fsrfg_frame_i(int i, uint64_t waitValue, uint64_t signalValue, int reset, double frameTimeMs)
{
    if (i < 0 || i >= kMaxInstances || !g_inst[i].live)
        return fail("instance not created"), -1;
    Instance& s = g_inst[i];
    if (FAILED(s.q->Wait(s.fence, waitValue)))
        return fail("queue Wait failed"), -2;
    int k = 0;
    ID3D12GraphicsCommandList* l = openList(s, k);
    if (!l)
        return -3;
    // the shared frame and vectors into FSR's own textures
    D3D12_RESOURCE_BARRIER b[4];
    b[0] = transition(s.color, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    b[1] = transition(s.motion, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    b[2] = transition(s.shIn, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    b[3] = transition(s.shMv, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    l->ResourceBarrier(4, b);
    l->CopyResource(s.color, s.shIn);
    l->CopyResource(s.motion, s.shMv);
    b[0] = transition(s.color, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    b[1] = transition(s.motion, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    b[2] = transition(s.shIn, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    b[3] = transition(s.shMv, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    l->ResourceBarrier(4, b);

    const FfxApiRect2D rect{0, 0, (int32_t)s.w, (int32_t)s.h};
    ffxConfigureDescFrameGeneration cfg = {};
    cfg.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
    cfg.swapChain = nullptr;
    cfg.frameGenerationEnabled = true;
    cfg.allowAsyncWorkloads = false;
    cfg.flags = FFX_FRAMEGENERATION_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY;
    cfg.generationRect = rect;
    cfg.frameID = s.frameID;
    if (g_ffx.Configure(&s.ctx, &cfg.header) != FFX_API_RETURN_OK)
        return l->Close(), fail("ffxConfigure(frame generation) failed"), -4;

    ffxDispatchDescFrameGenerationPrepareV2 prep = {};
    prep.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE_V2;
    prep.frameID = s.frameID;
    prep.flags = FFX_FRAMEGENERATION_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY;
    prep.commandList = l;
    prep.renderSize = {s.w, s.h};
    prep.jitterOffset = {0.0f, 0.0f};
    prep.motionVectorScale = {1.0f, 1.0f};
    prep.frameTimeDelta = (float)frameTimeMs;
    prep.reset = reset != 0;
    prep.cameraNear = 0.1f;
    prep.cameraFar = 1000.0f;
    prep.cameraFovAngleVertical = 1.0f;
    prep.viewSpaceToMetersFactor = 1.0f;
    prep.depth = ffxApiGetResourceDX12(s.depth, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    prep.motionVectors = ffxApiGetResourceDX12(s.motion, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    const float up[3] = {0, 1, 0}, right[3] = {1, 0, 0}, forward[3] = {0, 0, 1};
    memcpy(prep.cameraUp, up, sizeof(up));
    memcpy(prep.cameraRight, right, sizeof(right));
    memcpy(prep.cameraForward, forward, sizeof(forward));
    if (g_ffx.Dispatch(&s.ctx, &prep.header) != FFX_API_RETURN_OK)
        return l->Close(), fail("ffxDispatch(prepare V2) failed"), -5;

    ffxDispatchDescFrameGeneration fg = {};
    fg.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION;
    fg.commandList = l;
    fg.presentColor = ffxApiGetResourceDX12(s.color, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    fg.outputs[0] = ffxApiGetResourceDX12(s.out, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
    fg.numGeneratedFrames = 1;
    fg.reset = reset != 0;
    fg.backbufferTransferFunction = FFX_API_BACKBUFFER_TRANSFER_FUNCTION_SRGB;
    fg.minMaxLuminance[0] = 0.0f;
    fg.minMaxLuminance[1] = 80.0f;
    fg.generationRect = rect;
    fg.frameID = s.frameID;
    if (g_ffx.Dispatch(&s.ctx, &fg.header) != FFX_API_RETURN_OK)
        return l->Close(), fail("ffxDispatch(frame generation) failed"), -6;

    // the generated frame into the shared texture
    b[0] = transition(s.out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    b[1] = transition(s.shOut, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    l->ResourceBarrier(2, b);
    l->CopyResource(s.shOut, s.out);
    b[0] = transition(s.out, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    b[1] = transition(s.shOut, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    l->ResourceBarrier(2, b);
    if (FAILED(l->Close()))
        return fail("command list Close failed"), -7;
    ID3D12CommandList* lists[] = {l};
    s.q->ExecuteCommandLists(1, lists);
    if (FAILED(s.q->Signal(s.fence, signalValue)))
        return fail("queue Signal failed"), -8;
    s.done[k] = signalValue;
    ++s.frameID;
    return 0;
}

FSRFG_API void fsrfg_destroy_i(int i)
{
    if (i >= 0 && i < kMaxInstances)
        destroy(g_inst[i]);
}

FSRFG_API void fsrfg_destroy()
{
    for (Instance& s : g_inst)
        destroy(s);
}
