// SmoothMyVideo's Intel XeSS-SR bridge: Intel's libxess.dll (XeSS SDK 3.0.2, unmodified, beside this DLL in engine\xess;
// the app's XeSS setup downloads it) behind the same small C API as engine\fsrup's bridge (exports xessup_*), so the
// native host drives it on AMD FSR upscaling's route. Frames go in and out through D3D12 buffers shared with CUDA (NT
// handles the host imports as external memory) and one shared fence (an external semaphore), so the GPU waits on the
// GPU: the host writes the frame and its motion vectors, signals `wait`; the bridge's queue waits for it, copies the
// buffers into textures, runs XeSS-SR, copies the result out and signals `signal`. One context; every call upscales the
// next frame of one continuous stream.
// Formats: frames RGBA8 UNORM (sRGB-encoded, XESS_INIT_FLAG_LDR_INPUT_COLOR), or in the HDR mode (hdr) FP16 RGBA linear
// light with SDR white at 1.0; motion vectors RG16F in render-size pixels, current -> previous (XeSS's convention); depth
// = a constant (no depth in video; XeSS dilates render-size vectors by it); no jitter. The quality preset = the one
// whose input range holds the render size with its optimal ratio nearest to the call's.
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>

#include "xess/xess_d3d12.h"

#define XESSUP_API extern "C" __declspec(dllexport)

struct XessupShared
{
    HANDLE in;    // the render-size frame, RGBA8 (FP16 RGBA in the HDR mode), inPitch bytes a row
    HANDLE mv;    // RG16F motion vectors at the render size, mvPitch bytes a row
    HANDLE out;   // the upscaled frame, RGBA8 (FP16 RGBA in the HDR mode), outPitch bytes a row
    HANDLE fence; // the shared fence both sides wait on / signal
    uint64_t inBytes, mvBytes, outBytes;
    uint32_t inPitch, mvPitch, outPitch;
    char version[32]; // the XeSS-SR version and the preset the context runs
};

namespace
{
constexpr int kLists = 2;

std::string g_err;
std::mutex g_logLk;
std::string g_lastXess; // XeSS's newest warning or error message, appended to a failure's text

template <class T> void release(T*& p)
{
    if (p)
    {
        p->Release();
        p = nullptr;
    }
}

void xessLog(const char* msg, xess_logging_level_t)
{
    std::lock_guard<std::mutex> g(g_logLk);
    g_lastXess = msg ? msg : "";
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
    bool hdr = false;
    uint32_t w = 0, h = 0, ow = 0, oh = 0;
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
    xess_context_handle_t ctx = nullptr;
    XessupShared sh = {};
};

State g_s;

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

bool failX(const char* what, xess_result_t r)
{
    char b[512];
    std::lock_guard<std::mutex> g(g_logLk);
    snprintf(b, sizeof b, "%s (xess %d)%s%s", what, (int)r, g_lastXess.empty() ? "" : ": ", g_lastXess.c_str());
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
    if (s.ctx)
    {
        xessDestroyContext(s.ctx);
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
            fail("an earlier XeSS upscale did not finish within 10 s");
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

const char* qualityName(int q)
{
    switch (q)
    {
    case XESS_QUALITY_SETTING_ULTRA_PERFORMANCE:
        return "Ultra Performance";
    case XESS_QUALITY_SETTING_PERFORMANCE:
        return "Performance";
    case XESS_QUALITY_SETTING_BALANCED:
        return "Balanced";
    case XESS_QUALITY_SETTING_QUALITY:
        return "Quality";
    case XESS_QUALITY_SETTING_ULTRA_QUALITY:
        return "Ultra Quality";
    case XESS_QUALITY_SETTING_ULTRA_QUALITY_PLUS:
        return "Ultra Quality Plus";
    case XESS_QUALITY_SETTING_AA:
        return "Native AA";
    }
    return "?";
}
} // namespace

XESSUP_API const char* xessup_last_error()
{
    return g_err.c_str();
}

// the XeSS-SR context w x h -> ow x oh on the adapter with this LUID (`want` is unused: the FSR bridge's version pick);
// hdr = the HDR mode (FP16 linear-light frames in and out), else RGBA8 sRGB. Fills *sh (the handles stay owned by the
// bridge until xessup_destroy). 0 = ok, < 0 = xessup_last_error.
XESSUP_API int xessup_create(uint32_t w, uint32_t h, uint32_t ow, uint32_t oh, uint32_t luidLow, int32_t luidHigh,
                             const char* want, int hdr, XessupShared* sh)
{
    (void)want;
    if (!sh || !w || !h || !ow || !oh)
        return fail("bad arguments"), -1;
    State& s = g_s;
    if (s.live)
        destroy(s);
    s.hdr = hdr != 0;
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

    const DXGI_FORMAT cf = s.hdr ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_R8G8B8A8_UNORM;
    s.color = texture(s.dev, cf, w, h, false, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    s.motion = texture(s.dev, DXGI_FORMAT_R16G16_FLOAT, w, h, false, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    s.depth = texture(s.dev, DXGI_FORMAT_R32_FLOAT, w, h, false, D3D12_RESOURCE_STATE_COPY_DEST);
    s.out = texture(s.dev, cf, ow, oh, true, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
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

    // the constant depth (0.5, as the FSR bridge's), uploaded once
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

    xess_result_t xr = xessD3D12CreateContext(s.dev, &s.ctx);
    if (xr != XESS_RESULT_SUCCESS)
        return destroy(s), failX("xessD3D12CreateContext failed", xr), -14;
    xessSetLoggingCallback(s.ctx, XESS_LOGGING_LEVEL_WARNING, xessLog);

    // the preset: the one whose input range holds w x h with its optimal ratio nearest to ow / w
    const xess_2d_t outRes = {ow, oh};
    const double ratio = (double)ow / (double)w;
    int pick = 0;
    double best = 1e9;
    for (int q = XESS_QUALITY_SETTING_ULTRA_PERFORMANCE; q <= XESS_QUALITY_SETTING_AA; ++q)
    {
        xess_2d_t opt = {}, mn = {}, mx = {};
        if (xessGetOptimalInputResolution(s.ctx, &outRes, (xess_quality_settings_t)q, &opt, &mn, &mx) !=
            XESS_RESULT_SUCCESS)
            continue;
        const bool holds = w >= mn.x && w <= mx.x && h >= mn.y && h <= mx.y;
        const double d = std::fabs((double)ow / (double)(opt.x ? opt.x : 1) - ratio);
        if (holds && d < best)
        {
            best = d;
            pick = q;
        }
    }
    if (!pick)
        return destroy(s), fail("no XeSS-SR preset takes this input size for this output size"), -15;
    xess_d3d12_init_params_t ip = {};
    ip.outputResolution = outRes;
    ip.qualitySetting = (xess_quality_settings_t)pick;
    ip.initFlags = s.hdr ? XESS_INIT_FLAG_NONE : XESS_INIT_FLAG_LDR_INPUT_COLOR;
    xr = xessD3D12Init(s.ctx, &ip);
    if (xr != XESS_RESULT_SUCCESS)
        return destroy(s), failX("xessD3D12Init failed", xr), -16;
    xess_version_t v = {};
    xessGetVersion(&v);
    snprintf(s.sh.version, sizeof s.sh.version, "XeSS %u.%u.%u %s", v.major, v.minor, v.patch, qualityName(pick));
    s.live = true;
    *sh = s.sh;
    return 0;
}

// the queue waits for `waitValue` (the caller wrote the frame into sh.in and its vectors into sh.mv), upscales it into
// sh.out and signals `signalValue`. reset = the first frame of a stream (XeSS's history reset). Non-blocking but for
// the command-list ring. 0 = ok.
XESSUP_API int xessup_frame(uint64_t waitValue, uint64_t signalValue, int reset, double frameTimeMs)
{
    (void)frameTimeMs;
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

    xess_d3d12_execute_params_t ep = {};
    ep.pColorTexture = s.color;
    ep.pVelocityTexture = s.motion;
    ep.pDepthTexture = s.depth;
    ep.pOutputTexture = s.out;
    ep.jitterOffsetX = 0.0f;
    ep.jitterOffsetY = 0.0f;
    ep.exposureScale = 1.0f;
    ep.resetHistory = reset != 0 ? 1u : 0u;
    ep.inputWidth = s.w;
    ep.inputHeight = s.h;
    const xess_result_t xr = xessD3D12Execute(s.ctx, l, &ep);
    if (xr != XESS_RESULT_SUCCESS)
        return l->Close(), failX("xessD3D12Execute failed", xr), -4;

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

XESSUP_API void xessup_destroy()
{
    destroy(g_s);
}
