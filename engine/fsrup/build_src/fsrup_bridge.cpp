// SmoothMyVideo's AMD FSR upscaling bridge: AMD's signed FidelityFX loader + upscaler DLLs (FSR SDK v2.3.0, MIT, beside
// this DLL) behind a small C API for the native host, the upscaler version picked by name (FSR 3.1.x on NVIDIA). Frames
// go in and out through D3D12 buffers shared with CUDA (NT handles the host imports as external memory) and one shared
// fence (an external semaphore), so the GPU waits on the GPU: the host writes the frame and its motion vectors, signals
// `wait`; the bridge's queue waits for it, copies the buffers into textures, runs the upscale, copies the result out and
// signals `signal`. One context; every call upscales the next frame of one continuous stream.
// Formats: frames RGBA8 UNORM (sRGB-encoded); motion vectors RG16F in render-size pixels, current -> previous; depth = a
// constant (no depth in video); no jitter.
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ffx_api.h"
#include "ffx_api_loader.h"
#include "dx12/ffx_api_dx12.h"
#include "ffx_upscale.h"

#define FSRUP_API extern "C" __declspec(dllexport)

struct FsrupShared
{
    HANDLE in;    // the render-size frame, RGBA8, inPitch bytes a row
    HANDLE mv;    // RG16F motion vectors at the render size, mvPitch bytes a row
    HANDLE out;   // the upscaled frame, RGBA8, outPitch bytes a row
    HANDLE fence; // the shared fence both sides wait on / signal
    uint64_t inBytes, mvBytes, outBytes;
    uint32_t inPitch, mvPitch, outPitch;
    char version[32]; // the upscaler version the context runs
};

namespace
{
constexpr int kLists = 2;

std::string g_err;

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

struct State
{
    bool live = false;
    uint32_t w = 0, h = 0, ow = 0, oh = 0;
    HMODULE loader = nullptr;
    ffxFunctions ffx = {};
    ID3D12Device* dev = nullptr;
    ID3D12CommandQueue* q = nullptr;
    ID3D12CommandAllocator* alloc[kLists] = {};
    ID3D12GraphicsCommandList* list[kLists] = {};
    uint64_t done[kLists] = {};
    int next = 0;
    ID3D12Resource* color = nullptr;
    ID3D12Resource* motion = nullptr;
    ID3D12Resource* depth = nullptr;
    ID3D12Resource* out = nullptr;
    ID3D12Resource* shIn = nullptr;
    ID3D12Resource* shMv = nullptr;
    ID3D12Resource* shOut = nullptr;
    ID3D12Fence* fence = nullptr; // shared with the caller: its values are the caller's
    ID3D12Fence* priv = nullptr;  // the bridge's own setup work
    HANDLE ev = nullptr;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fpIn = {}, fpMv = {}, fpOut = {};
    ffxContext ctx = nullptr;
    FsrupShared sh = {};
};

State g_s;

bool fail(const char* what, HRESULT hr = S_OK)
{
    char b[256];
    if (hr != S_OK)
        snprintf(b, sizeof b, "%s (hr 0x%08lx)", what, (unsigned long)hr);
    else
        snprintf(b, sizeof b, "%s", what);
    g_err = b;
    return false;
}

ID3D12Resource* texture(ID3D12Device* d, DXGI_FORMAT f, uint32_t w, uint32_t h, bool uav, D3D12_RESOURCE_STATES st)
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
    if (FAILED(d->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, st, nullptr, IID_PPV_ARGS(&r))))
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

void destroy(State& s)
{
    if (s.q && s.priv && s.ev)
    {
        // drain the queue before anything it uses goes away (a queue still waiting for a caller value that never comes
        // is given up on after 10 s)
        const uint64_t v = s.priv->GetCompletedValue() + 1;
        if (SUCCEEDED(s.q->Signal(s.priv, v)) && SUCCEEDED(s.priv->SetEventOnCompletion(v, s.ev)))
            WaitForSingleObject(s.ev, 10000);
    }
    if (s.ctx && s.ffx.DestroyContext)
    {
        s.ffx.DestroyContext(&s.ctx, nullptr);
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
}

// a list from the ring, its previous use finished (the GPU waits out only GPU work: the caller signals every value the
// queue waits for before it asks for the next frame)
ID3D12GraphicsCommandList* openList(State& s, int& k)
{
    k = s.next;
    s.next = (k + 1) % kLists;
    if (s.fence->GetCompletedValue() < s.done[k])
    {
        if (FAILED(s.fence->SetEventOnCompletion(s.done[k], s.ev)) || WaitForSingleObject(s.ev, 10000) != WAIT_OBJECT_0)
        {
            fail("an earlier FSR upscale did not finish within 10 s");
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

// AMD's loader from this DLL's own folder (it finds the upscaler DLL beside it)
bool loadLoader(State& s)
{
    if (s.loader)
        return true;
    HMODULE self = nullptr;
    wchar_t path[MAX_PATH] = {};
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            (LPCWSTR)&loadLoader, &self) ||
        !GetModuleFileNameW(self, path, MAX_PATH))
        return fail("this DLL's folder is unknown");
    std::wstring dir(path);
    dir = dir.substr(0, dir.find_last_of(L"\\/") + 1);
    // the loader opens the upscaler DLL by name, which the process's search path (the exe's folder) does not reach:
    // loaded first by full path, the by-name load finds the module already in memory
    if (!LoadLibraryExW((dir + L"amd_fidelityfx_upscaler_dx12.dll").c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH))
        return fail("amd_fidelityfx_upscaler_dx12.dll did not load from this DLL's folder",
                    HRESULT_FROM_WIN32(GetLastError()));
    s.loader =
        LoadLibraryExW((dir + L"amd_fidelityfx_loader_dx12.dll").c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!s.loader)
        return fail("amd_fidelityfx_loader_dx12.dll did not load from this DLL's folder",
                    HRESULT_FROM_WIN32(GetLastError()));
    ffxLoadFunctions(&s.ffx, s.loader);
    if (!s.ffx.CreateContext || !s.ffx.DestroyContext || !s.ffx.Query || !s.ffx.Dispatch)
        return fail("the FidelityFX loader lacks an entry point");
    return true;
}
} // namespace

FSRUP_API const char* fsrup_last_error()
{
    return g_err.c_str();
}

// the upscaler context w x h -> ow x oh on the adapter with this LUID, the first upscaler version whose name starts with
// `want` (e.g. "3.1"). Fills *sh (the handles stay owned by the bridge until fsrup_destroy). 0 = ok, < 0 =
// fsrup_last_error.
FSRUP_API int fsrup_create(uint32_t w, uint32_t h, uint32_t ow, uint32_t oh, uint32_t luidLow, int32_t luidHigh,
                           const char* want, FsrupShared* sh)
{
    if (!sh || !w || !h || !ow || !oh || !want)
        return fail("bad arguments"), -1;
    State& s = g_s;
    if (s.live)
        destroy(s);
    if (!loadLoader(s))
        return -2;
    s.w = w;
    s.h = h;
    s.ow = ow;
    s.oh = oh;
    IDXGIFactory4* fac = nullptr;
    HRESULT hr = CreateDXGIFactory2(0, IID_PPV_ARGS(&fac));
    if (FAILED(hr))
        return fail("CreateDXGIFactory2 failed", hr), -3;
    LUID luid;
    luid.LowPart = luidLow;
    luid.HighPart = luidHigh;
    IDXGIAdapter1* ad = nullptr;
    hr = fac->EnumAdapterByLuid(luid, IID_PPV_ARGS(&ad));
    release(fac);
    if (FAILED(hr))
        return fail("no DXGI adapter with the CUDA device's LUID", hr), -4;
    hr = D3D12CreateDevice(ad, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&s.dev));
    release(ad);
    if (FAILED(hr))
        return fail("D3D12CreateDevice failed", hr), -5;
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(hr = s.dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&s.q))))
        return destroy(s), fail("CreateCommandQueue failed", hr), -6;
    for (int k = 0; k < kLists; ++k)
    {
        if (FAILED(hr = s.dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&s.alloc[k]))) ||
            FAILED(hr = s.dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, s.alloc[k], nullptr,
                                                 IID_PPV_ARGS(&s.list[k]))))
            return destroy(s), fail("command list creation failed", hr), -7;
        s.list[k]->Close();
    }
    s.ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (FAILED(hr = s.dev->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&s.fence))) ||
        FAILED(hr = s.dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&s.priv))))
        return destroy(s), fail("fence creation failed", hr), -8;

    s.color = texture(s.dev, DXGI_FORMAT_R8G8B8A8_UNORM, w, h, false, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    s.motion = texture(s.dev, DXGI_FORMAT_R16G16_FLOAT, w, h, false, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    s.depth = texture(s.dev, DXGI_FORMAT_R32_FLOAT, w, h, false, D3D12_RESOURCE_STATE_COPY_DEST);
    s.out = texture(s.dev, DXGI_FORMAT_R8G8B8A8_UNORM, ow, oh, true, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (!s.color || !s.motion || !s.depth || !s.out)
        return destroy(s), fail("texture creation failed"), -9;
    UINT64 inBytes = 0, mvBytes = 0, outBytes = 0;
    const D3D12_RESOURCE_DESC cd = s.color->GetDesc(), md = s.motion->GetDesc(), od = s.out->GetDesc();
    s.dev->GetCopyableFootprints(&cd, 0, 1, 0, &s.fpIn, nullptr, nullptr, &inBytes);
    s.dev->GetCopyableFootprints(&md, 0, 1, 0, &s.fpMv, nullptr, nullptr, &mvBytes);
    s.dev->GetCopyableFootprints(&od, 0, 1, 0, &s.fpOut, nullptr, nullptr, &outBytes);
    // shared buffers start in COMMON: a buffer promotes to COPY_SOURCE / COPY_DEST implicitly and decays back when its
    // list completes, the state CUDA reads and writes it in
    s.shIn = buffer(s.dev, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_SHARED, inBytes, D3D12_RESOURCE_STATE_COMMON);
    s.shMv = buffer(s.dev, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_SHARED, mvBytes, D3D12_RESOURCE_STATE_COMMON);
    s.shOut = buffer(s.dev, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_SHARED, outBytes, D3D12_RESOURCE_STATE_COMMON);
    if (!s.shIn || !s.shMv || !s.shOut)
        return destroy(s), fail("shared buffer creation failed"), -10;
    if (FAILED(hr = s.dev->CreateSharedHandle(s.shIn, nullptr, GENERIC_ALL, nullptr, &s.sh.in)) ||
        FAILED(hr = s.dev->CreateSharedHandle(s.shMv, nullptr, GENERIC_ALL, nullptr, &s.sh.mv)) ||
        FAILED(hr = s.dev->CreateSharedHandle(s.shOut, nullptr, GENERIC_ALL, nullptr, &s.sh.out)) ||
        FAILED(hr = s.dev->CreateSharedHandle(s.fence, nullptr, GENERIC_ALL, nullptr, &s.sh.fence)))
        return destroy(s), fail("shared handle creation failed", hr), -11;
    s.sh.inBytes = inBytes;
    s.sh.mvBytes = mvBytes;
    s.sh.outBytes = outBytes;
    s.sh.inPitch = s.fpIn.Footprint.RowPitch;
    s.sh.mvPitch = s.fpMv.Footprint.RowPitch;
    s.sh.outPitch = s.fpOut.Footprint.RowPitch;

    // the constant depth (0.5, as the measurement harness's flat depth), uploaded once
    {
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp = {};
        UINT64 bytes = 0;
        const D3D12_RESOURCE_DESC dd = s.depth->GetDesc();
        s.dev->GetCopyableFootprints(&dd, 0, 1, 0, &fp, nullptr, nullptr, &bytes);
        ID3D12Resource* up =
            buffer(s.dev, D3D12_HEAP_TYPE_UPLOAD, D3D12_HEAP_FLAG_NONE, bytes, D3D12_RESOURCE_STATE_GENERIC_READ);
        uint8_t* p = nullptr;
        if (!up || FAILED(up->Map(0, nullptr, (void**)&p)))
            return release(up), destroy(s), fail("depth upload failed"), -12;
        for (uint32_t y = 0; y < h; ++y)
        {
            float* row = (float*)(p + fp.Offset + (size_t)y * fp.Footprint.RowPitch);
            for (uint32_t x = 0; x < w; ++x)
                row[x] = 0.5f;
        }
        up->Unmap(0, nullptr);
        ID3D12GraphicsCommandList* l = s.list[0];
        if (FAILED(s.alloc[0]->Reset()) || FAILED(l->Reset(s.alloc[0], nullptr)))
            return release(up), destroy(s), fail("command list Reset failed"), -13;
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
            return release(up), destroy(s), fail("depth upload did not finish"), -13;
        release(up);
    }

    // the upscaler version: the first one the loader lists whose name starts with `want`
    uint64_t pick = 0;
    {
        uint64_t count = 0;
        ffxQueryDescGetVersions gv = {};
        gv.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
        gv.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
        gv.device = s.dev;
        gv.outputCount = &count;
        if (s.ffx.Query(nullptr, &gv.header) != FFX_API_RETURN_OK || !count)
            return destroy(s), fail("the upscaler versions query failed"), -14;
        std::vector<uint64_t> ids((size_t)count);
        std::vector<const char*> names((size_t)count);
        gv.versionIds = ids.data();
        gv.versionNames = names.data();
        if (s.ffx.Query(nullptr, &gv.header) != FFX_API_RETURN_OK)
            return destroy(s), fail("the upscaler versions query failed"), -14;
        std::string list;
        for (size_t k = 0; k < ids.size(); ++k)
        {
            const char* n = names[k] ? names[k] : "?";
            list += std::string(list.empty() ? "" : ", ") + n;
            if (!pick && strncmp(n, want, strlen(want)) == 0)
            {
                pick = ids[k];
                snprintf(s.sh.version, sizeof s.sh.version, "%s", n);
            }
        }
        if (!pick)
            return destroy(s),
                   fail(("no upscaler version starts with " + std::string(want) + " (offered: " + list + ")").c_str()),
                   -15;
    }
    ffxCreateBackendDX12Desc backend = {};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
    backend.device = s.dev;
    ffxOverrideVersion over = {};
    over.header.type = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
    over.header.pNext = &backend.header;
    over.versionId = pick;
    ffxCreateContextDescUpscaleVersion ver = {};
    ver.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION;
    ver.header.pNext = &over.header;
    ver.version = FFX_UPSCALER_VERSION;
    ffxCreateContextDescUpscale create = {};
    create.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    create.header.pNext = &ver.header;
    create.flags = FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE;
    create.maxRenderSize = {w, h};
    create.maxUpscaleSize = {ow, oh};
    if (s.ffx.CreateContext(&s.ctx, &create.header, nullptr) != FFX_API_RETURN_OK)
        return destroy(s), fail("ffxCreateContext(upscale) failed"), -16;
    s.live = true;
    *sh = s.sh;
    return 0;
}

// the queue waits for `waitValue` (the caller wrote the frame into sh.in and its vectors into sh.mv), upscales it into
// sh.out and signals `signalValue`. reset = the first frame of a stream. Non-blocking but for the command-list ring. 0 =
// ok.
FSRUP_API int fsrup_frame(uint64_t waitValue, uint64_t signalValue, int reset, double frameTimeMs)
{
    State& s = g_s;
    if (!s.live)
        return fail("context not created"), -1;
    if (FAILED(s.q->Wait(s.fence, waitValue)))
        return fail("queue Wait failed"), -2;
    int k = 0;
    ID3D12GraphicsCommandList* l = openList(s, k);
    if (!l)
        return -3;
    D3D12_RESOURCE_BARRIER b[2];
    b[0] = transition(s.color, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    b[1] = transition(s.motion, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    l->ResourceBarrier(2, b);
    D3D12_TEXTURE_COPY_LOCATION dc = subresource(s.color), sc = footprint(s.shIn, s.fpIn);
    l->CopyTextureRegion(&dc, 0, 0, 0, &sc, nullptr);
    D3D12_TEXTURE_COPY_LOCATION dm = subresource(s.motion), sm = footprint(s.shMv, s.fpMv);
    l->CopyTextureRegion(&dm, 0, 0, 0, &sm, nullptr);
    b[0] = transition(s.color, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    b[1] = transition(s.motion, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    l->ResourceBarrier(2, b);

    ffxDispatchDescUpscale up = {};
    up.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    up.commandList = l;
    up.color = ffxApiGetResourceDX12(s.color, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    up.depth = ffxApiGetResourceDX12(s.depth, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    up.motionVectors = ffxApiGetResourceDX12(s.motion, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    up.output = ffxApiGetResourceDX12(s.out, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
    up.jitterOffset = {0.0f, 0.0f};
    up.motionVectorScale = {1.0f, 1.0f};
    up.renderSize = {s.w, s.h};
    up.upscaleSize = {s.ow, s.oh};
    up.enableSharpening = false;
    up.frameTimeDelta = (float)frameTimeMs;
    up.preExposure = 1.0f;
    up.reset = reset != 0;
    up.cameraNear = 0.1f;
    up.cameraFar = 1000.0f;
    up.cameraFovAngleVertical = 1.0f;
    up.viewSpaceToMetersFactor = 1.0f;
    up.flags = FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_SRGB;
    if (s.ffx.Dispatch(&s.ctx, &up.header) != FFX_API_RETURN_OK)
        return l->Close(), fail("ffxDispatch(upscale) failed"), -4;

    D3D12_RESOURCE_BARRIER ob =
        transition(s.out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    l->ResourceBarrier(1, &ob);
    D3D12_TEXTURE_COPY_LOCATION so = subresource(s.out), dout = footprint(s.shOut, s.fpOut);
    l->CopyTextureRegion(&dout, 0, 0, 0, &so, nullptr);
    ob = transition(s.out, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    l->ResourceBarrier(1, &ob);
    if (FAILED(l->Close()))
        return fail("command list Close failed"), -5;
    ID3D12CommandList* lists[] = {l};
    s.q->ExecuteCommandLists(1, lists);
    if (FAILED(s.q->Signal(s.fence, signalValue)))
        return fail("queue Signal failed"), -6;
    s.done[k] = signalValue;
    return 0;
}

FSRUP_API void fsrup_destroy()
{
    destroy(g_s);
}
