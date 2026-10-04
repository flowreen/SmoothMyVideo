// SmoothMyVideo's AMD FSR 3 frame generation bridge: the FidelityFX SDK v2.3.0 FSR 3.1.6 frame generation (MIT, built
// from source with the optical flow's scene-change reset removed: build.py beside this file) behind a small C API
// for the native host. Frames go in and out through D3D12 buffers shared with CUDA (NT handles the host imports as
// external memory) and one shared fence (an external semaphore), so the GPU waits on the GPU: the host writes the
// frame and its motion vectors, signals `wait`; the bridge's queue waits for it, copies the buffers into textures,
// runs configure -> prepare V2 -> generate, copies the generated frame out and signals `signal`. One context per
// instance; every call feeds the next real frame (frameID + 1), the generated frame lies halfway between the previous
// call's frame and this one.
// Formats: fmt 0 = RGBA8 UNORM (sRGB-encoded bytes), fmt 1 = RGBA16F (sRGB-encoded values 0..1); motion vectors =
// RG16F in pixels, current -> previous (FSR's convention); depth = a constant (no depth in video).
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "Kits/FidelityFX/api/include/ffx_api.h"
#include "Kits/FidelityFX/api/include/dx12/ffx_api_dx12.h"
#include "Kits/FidelityFX/framegeneration/include/ffx_framegeneration.h"
#include "Kits/FidelityFX/framegeneration/include/dx12/ffx_api_framegeneration_dx12.h"

#define FSRFG_API extern "C" __declspec(dllexport)

struct FsrfgShared
{
    HANDLE in;    // the real frame, fmt's layout, rowPitch bytes a row
    HANDLE mv;    // RG16F motion vectors, mvRowPitch bytes a row
    HANDLE out;   // the generated frame, fmt's layout, rowPitch bytes a row
    HANDLE fence; // the shared fence both sides wait on / signal
    uint64_t inBytes, mvBytes, outBytes;
    uint32_t rowPitch, mvRowPitch;
};

namespace
{
constexpr int kLists = 2;
constexpr int kMaxInstances = 4;

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
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fpColor = {}, fpMv = {};
    ffxContext ctx = nullptr;
    uint64_t frameID = 0;
    FsrfgShared sh = {};
};

Instance g_inst[kMaxInstances];

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
        ffxConfigure(&s.ctx, &cfg.header);
        ffxDestroyContext(&s.ctx, nullptr);
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

// instance i (0..3) at w x h on the adapter with this LUID; fmt 0 = RGBA8, 1 = RGBA16F. Fills *sh (the handles stay
// owned by the bridge until fsrfg_destroy_i). 0 = ok, < 0 = fsrfg_last_error.
FSRFG_API int fsrfg_create_i(int i, uint32_t w, uint32_t h, uint32_t luidLow, int32_t luidHigh, int fmt,
                             FsrfgShared* sh)
{
    if (i < 0 || i >= kMaxInstances || !sh || !w || !h)
        return fail("bad arguments"), -1;
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
    hr = D3D12CreateDevice(ad, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&s.dev));
    release(ad);
    if (FAILED(hr))
        return fail("D3D12CreateDevice failed", hr), -4;
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

    s.color = texture(s.dev, s.fmt, w, h, false, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    s.motion = texture(s.dev, DXGI_FORMAT_R16G16_FLOAT, w, h, false, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    s.depth = texture(s.dev, DXGI_FORMAT_R32_FLOAT, w, h, false, D3D12_RESOURCE_STATE_COPY_DEST);
    s.out = texture(s.dev, s.fmt, w, h, true, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (!s.color || !s.motion || !s.depth || !s.out)
        return destroy(s), fail("texture creation failed"), -8;
    UINT64 inBytes = 0, mvBytes = 0;
    const D3D12_RESOURCE_DESC cd = s.color->GetDesc(), md = s.motion->GetDesc();
    s.dev->GetCopyableFootprints(&cd, 0, 1, 0, &s.fpColor, nullptr, nullptr, &inBytes);
    s.dev->GetCopyableFootprints(&md, 0, 1, 0, &s.fpMv, nullptr, nullptr, &mvBytes);
    // shared buffers start in COMMON: a buffer promotes to COPY_SOURCE / COPY_DEST implicitly and decays back when its
    // list completes, the state CUDA reads and writes it in
    s.shIn = buffer(s.dev, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_SHARED, inBytes, D3D12_RESOURCE_STATE_COMMON);
    s.shMv = buffer(s.dev, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_SHARED, mvBytes, D3D12_RESOURCE_STATE_COMMON);
    s.shOut = buffer(s.dev, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_SHARED, inBytes, D3D12_RESOURCE_STATE_COMMON);
    if (!s.shIn || !s.shMv || !s.shOut)
        return destroy(s), fail("shared buffer creation failed"), -9;
    if (FAILED(hr = s.dev->CreateSharedHandle(s.shIn, nullptr, GENERIC_ALL, nullptr, &s.sh.in)) ||
        FAILED(hr = s.dev->CreateSharedHandle(s.shMv, nullptr, GENERIC_ALL, nullptr, &s.sh.mv)) ||
        FAILED(hr = s.dev->CreateSharedHandle(s.shOut, nullptr, GENERIC_ALL, nullptr, &s.sh.out)) ||
        FAILED(hr = s.dev->CreateSharedHandle(s.fence, nullptr, GENERIC_ALL, nullptr, &s.sh.fence)))
        return destroy(s), fail("shared handle creation failed", hr), -10;
    s.sh.inBytes = s.sh.outBytes = inBytes;
    s.sh.mvBytes = mvBytes;
    s.sh.rowPitch = s.fpColor.Footprint.RowPitch;
    s.sh.mvRowPitch = s.fpMv.Footprint.RowPitch;

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
    ffxCreateContextDescFrameGeneration create = {};
    create.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
    create.header.pNext = &backend.header;
    create.flags = 0;
    create.displaySize = {w, h};
    create.maxRenderSize = {w, h};
    create.backBufferFormat = fmt ? FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT : FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM;
    if (ffxCreateContext(&s.ctx, &create.header, nullptr) != FFX_API_RETURN_OK)
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
    D3D12_RESOURCE_BARRIER b[2];
    b[0] = transition(s.color, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    b[1] = transition(s.motion, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    l->ResourceBarrier(2, b);
    D3D12_TEXTURE_COPY_LOCATION dc = subresource(s.color), sc = footprint(s.shIn, s.fpColor);
    l->CopyTextureRegion(&dc, 0, 0, 0, &sc, nullptr);
    D3D12_TEXTURE_COPY_LOCATION dm = subresource(s.motion), sm = footprint(s.shMv, s.fpMv);
    l->CopyTextureRegion(&dm, 0, 0, 0, &sm, nullptr);
    b[0] = transition(s.color, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    b[1] = transition(s.motion, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    l->ResourceBarrier(2, b);

    const FfxApiRect2D rect{0, 0, (int32_t)s.w, (int32_t)s.h};
    ffxConfigureDescFrameGeneration cfg = {};
    cfg.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
    cfg.swapChain = nullptr;
    cfg.frameGenerationEnabled = true;
    cfg.allowAsyncWorkloads = false;
    cfg.flags = FFX_FRAMEGENERATION_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY;
    cfg.generationRect = rect;
    cfg.frameID = s.frameID;
    if (ffxConfigure(&s.ctx, &cfg.header) != FFX_API_RETURN_OK)
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
    if (ffxDispatch(&s.ctx, &prep.header) != FFX_API_RETURN_OK)
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
    if (ffxDispatch(&s.ctx, &fg.header) != FFX_API_RETURN_OK)
        return l->Close(), fail("ffxDispatch(frame generation) failed"), -6;

    D3D12_RESOURCE_BARRIER ob =
        transition(s.out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    l->ResourceBarrier(1, &ob);
    D3D12_TEXTURE_COPY_LOCATION so = subresource(s.out), dout = footprint(s.shOut, s.fpColor);
    l->CopyTextureRegion(&dout, 0, 0, 0, &so, nullptr);
    ob = transition(s.out, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    l->ResourceBarrier(1, &ob);
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
