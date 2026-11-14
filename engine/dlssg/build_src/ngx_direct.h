// DLSS frame generation through NGX's own D3D12 calls (the DLSS SDK's nvsdk_ngx_helpers_dlssg_d3d.h): colour, depth
// and motion vectors in, each generated frame into a texture of this device. No swap chain, no window and no pacer,
// so a pair costs its evaluates alone and nothing on the desktop (a focus change, another app's presents) can stop
// the frames. The calls every route's frames are pinned to: the vector scale is the vector texture's size, and a
// clip's first frame is evaluated three times, the first with the reset flag.
// Included after pan_vectors.h and a LOG macro; --server uses frame(), a live stage (ngx_stage.h) the stage* calls.
#pragma once
#include <mutex>
#include <nvsdk_ngx.h>
#include <nvsdk_ngx_helpers_dlssg_d3d.h>

namespace ngxfg
{

static void NVSDK_CONV ngxLog(const char* msg, NVSDK_NGX_Logging_Level, NVSDK_NGX_Feature)
{
    LOG("[NGX] %s", msg);
}

// NGX starts once a process, on the one D3D12 device a process has on the adapter, and every Direct in the process
// uses that start: the live host runs several stages at once, each on a thread of its own. It stops with the last
// Direct, unless `hold` keeps it up for the process's life (a process with other NGX users on the same device, which
// NGX's shutdown would reach too). NGX's calls are made one at a time (call()).
struct Core
{
    std::mutex m;
    int users = 0;
    bool up = false, hold = false;
    ComPtr<ID3D12Device> device; // the device NGX started on, kept while it is up
};
static Core& core()
{
    static Core c;
    return c;
}
static std::mutex& callLock()
{
    static std::mutex m;
    return m;
}

// NGX started for the process on `device`, once (Core): each call that returns true is one use, given back by
// coreStop(). nvngx_dlssg.dll beside this exe; no NGX log file, its lines go to stderr with DLSSG_VERBOSE=1. false =
// NGX did not start, or runs on another device (the reason is logged)
static bool coreStart(ID3D12Device* device)
{
    wchar_t exePath[MAX_PATH];
    GetModuleFileNameW(nullptr, exePath, MAX_PATH);
    std::wstring exeDir(exePath);
    exeDir.resize(exeDir.find_last_of(L'\\'));
    wchar_t tempDir[MAX_PATH];
    if (!GetTempPathW(MAX_PATH, tempDir))
        wcscpy_s(tempDir, exeDir.c_str());
    const wchar_t* paths[] = {exeDir.c_str()};
    const bool verbose = GetEnvironmentVariableW(L"DLSSG_VERBOSE", nullptr, 0) != 0;
    NVSDK_NGX_FeatureCommonInfo fci{};
    fci.PathListInfo.Path = paths;
    fci.PathListInfo.Length = 1;
    fci.LoggingInfo.LoggingCallback = ngxLog;
    fci.LoggingInfo.MinimumLoggingLevel = verbose ? NVSDK_NGX_LOGGING_LEVEL_ON : NVSDK_NGX_LOGGING_LEVEL_OFF;
    Core& c = core();
    std::lock_guard<std::mutex> lk(c.m);
    if (!c.up)
    {
        std::lock_guard<std::mutex> cl(callLock());
        // NVIDIA's sample application id
        const NVSDK_NGX_Result r = NVSDK_NGX_D3D12_Init(231313132ull, tempDir, device, &fci);
        if (NVSDK_NGX_FAILED(r))
        {
            LOG("NGX: init failed (0x%08x)\n", (unsigned)r);
            return false;
        }
        c.up = true;
        c.device = device;
    }
    else if (c.device.Get() != device)
    {
        LOG("NGX: started on another device\n");
        return false;
    }
    c.users++;
    return true;
}

// one use of NGX given back: it stops with the last one, unless the process holds it up
static void coreStop()
{
    Core& c = core();
    std::lock_guard<std::mutex> lk(c.m);
    if (--c.users == 0 && !c.hold && c.up)
    {
        {
            std::lock_guard<std::mutex> cl(callLock());
            NVSDK_NGX_D3D12_Shutdown1(c.device.Get());
        }
        c.up = false;
        c.device.Reset();
    }
}

// The most frames DLSS frame generation makes a pair on this GPU and driver (DLSSG.MultiFrameCountMax: 5 on an RTX
// 50, 1 on an RTX 40), asked of NGX without a feature: what a plan of stages is made with before any stage runs.
// 0 = not reported, or NGX did not start
static uint32_t maxGenOf(ID3D12Device* device)
{
    if (!coreStart(device))
        return 0;
    uint32_t most = 0;
    {
        std::lock_guard<std::mutex> lk(callLock());
        NVSDK_NGX_Parameter* p = nullptr;
        if (NVSDK_NGX_SUCCEED(NVSDK_NGX_D3D12_GetCapabilityParameters(&p)) && p)
        {
            p->Get(NVSDK_NGX_DLSSG_Parameter_MultiFrameCountMax, &most);
            NVSDK_NGX_D3D12_DestroyParameters(p);
        }
    }
    coreStop();
    return most;
}

struct Direct
{
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    HANDLE fenceEvent = nullptr;
    uint64_t fenceValue = 0;
    ComPtr<ID3D12Resource> texFrame, texDepth, texMvec, texOut, uploadBuf, readbackBuf;
    uint8_t* upload = nullptr;         // uploadBuf, mapped for the process's life
    const uint8_t* readback = nullptr; // readbackBuf, the same
    uint32_t rowPitch = 0;
    NVSDK_NGX_Parameter* params = nullptr;
    NVSDK_NGX_Handle* feature = nullptr;
    NVSDK_NGX_DLSSG_Opt_Eval_Params opt{};
    panvec::Estimator pan;  // measures each frame against the one before it
    panvec::Writer panTex;  // writes texMvec
    panvec::Result lastPan; // a live stage: what the last CPU input measured against the one before it
    bool panOn = false, ngxUp = false, hdr10 = false, bgra = false;
    bool resetNext = true; // the next evaluate starts a clip
    uint32_t W = 0, H = 0; // the frames' size
    int gen = 1;
    uint32_t maxGen = 0; // DLSSG.MultiFrameCountMax, 0 = not reported

    // a live stage: every input and frame is queued on its own list and never waited for; a list is used again once
    // the GPU has passed its last run
    static const int kStageLists = 16;
    ComPtr<ID3D12CommandAllocator> sAlloc[kStageLists];
    ComPtr<ID3D12GraphicsCommandList> sList[kStageLists];
    uint64_t sUse[kStageLists] = {}; // sFence's value behind each list's last run
    uint64_t sLists = 0;             // lists handed out
    int sCur = 0;                    // the list handed out last
    ID3D12Fence* sFence = nullptr;   // the stage's fence (the ring's, shared with its readers)
    uint64_t sValue = 0;
    uint64_t sUpload = 0; // sFence's value behind the last copy out of uploadBuf

    // one owner: a Direct gives its NGX use back when it goes (shutdown() before that is fine, a second one does nothing)
    Direct() = default;
    Direct(const Direct&) = delete;
    Direct& operator=(const Direct&) = delete;
    ~Direct()
    {
        shutdown();
    }

    // both shader-resource bits: the state the colour, depth and vector textures are evaluated in
    static constexpr D3D12_RESOURCE_STATES kSrv =
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;

    DXGI_FORMAT frameFormat() const
    {
        return hdr10 ? DXGI_FORMAT_R10G10B10A2_UNORM : bgra ? DXGI_FORMAT_B8G8R8A8_UNORM : DXGI_FORMAT_R8G8B8A8_UNORM;
    }

    bool submit()
    {
        if (FAILED(list->Close()))
            return false;
        ID3D12CommandList* ls[] = {list.Get()};
        queue->ExecuteCommandLists(1, ls);
        if (FAILED(queue->Signal(fence.Get(), ++fenceValue)))
            return false;
        if (fence->GetCompletedValue() < fenceValue)
        {
            if (FAILED(fence->SetEventOnCompletion(fenceValue, fenceEvent)))
                return false;
            WaitForSingleObject(fenceEvent, INFINITE);
        }
        return SUCCEEDED(alloc->Reset()) && SUCCEEDED(list->Reset(alloc.Get(), nullptr));
    }

    static void transition(ID3D12GraphicsCommandList* l, ID3D12Resource* r, D3D12_RESOURCE_STATES from,
                           D3D12_RESOURCE_STATES to)
    {
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = r;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b.Transition.StateBefore = from;
        b.Transition.StateAfter = to;
        l->ResourceBarrier(1, &b);
    }
    void transition(ID3D12Resource* r, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to)
    {
        transition(list.Get(), r, from, to);
    }

    bool makeTex(DXGI_FORMAT fmt, uint32_t w, uint32_t h, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state,
                 ComPtr<ID3D12Resource>& tex)
    {
        D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rd.Width = w;
        rd.Height = h;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.Format = fmt;
        rd.SampleDesc = {1, 0};
        rd.Flags = flags;
        return SUCCEEDED(
            device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, state, nullptr, IID_PPV_ARGS(&tex)));
    }

    bool makeBuf(D3D12_HEAP_TYPE heap, UINT64 size, D3D12_RESOURCE_STATES state, ComPtr<ID3D12Resource>& buf)
    {
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
    }

    bool createFeature()
    {
        NVSDK_NGX_DLSSG_Create_Params cp{};
        cp.Width = W;
        cp.Height = H;
        cp.NativeBackbufferFormat = (unsigned int)frameFormat();
        cp.RenderWidth = panvec::vecSize(W);
        cp.RenderHeight = panvec::vecSize(H);
        cp.DynamicResolutionScaling = false;
        NVSDK_NGX_Result r;
        {
            std::lock_guard<std::mutex> lk(callLock());
            r = NGX_D3D12_CREATE_DLSSG(list.Get(), 1, 1, &feature, params, &cp);
        }
        if (NVSDK_NGX_FAILED(r) || !submit())
        {
            LOG("NGX: the frame generation feature was refused (0x%08x)\n", (unsigned)r);
            feature = nullptr;
            return false;
        }
        resetNext = true;
        return true;
    }

    void releaseFeature()
    {
        if (feature)
        {
            std::lock_guard<std::mutex> lk(callLock());
            NVSDK_NGX_D3D12_ReleaseFeature(feature);
        }
        feature = nullptr;
    }

    // the camera a video frame is given (a static pinhole camera) as NGX's optional evaluate parameters
    void fillOptions()
    {
        const float fovY = 60.0f * 3.14159265f / 180.0f;
        const float aspect = (float)W / (float)H;
        const float zn = 0.1f, zf = 100.0f;
        const float f = 1.0f / tanf(fovY * 0.5f);
        const float a = f / aspect, b = f;
        const float cc = zf / (zf - zn);
        const float d = -zn * zf / (zf - zn);
        const float ident[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
        const float proj[4][4] = {{a, 0, 0, 0}, {0, b, 0, 0}, {0, 0, cc, 1}, {0, 0, d, 0}};
        const float projInv[4][4] = {{1.0f / a, 0, 0, 0}, {0, 1.0f / b, 0, 0}, {0, 0, 0, 1.0f / d}, {0, 0, 1, -cc / d}};
        memcpy(opt.cameraViewToClip, proj, sizeof(proj));
        memcpy(opt.clipToCameraView, projInv, sizeof(projInv));
        memcpy(opt.clipToLensClip, ident, sizeof(ident));
        memcpy(opt.clipToPrevClip, ident, sizeof(ident));
        memcpy(opt.prevClipToClip, ident, sizeof(ident));
        const uint32_t vw = panvec::vecSize(W), vh = panvec::vecSize(H);
        // the texture holds a pixel's motion over the frame's size (pan_vectors.h); NGX takes it times this scale
        opt.mvecScale[0] = (float)vw;
        opt.mvecScale[1] = (float)vh;
        opt.cameraUp[1] = 1.0f;
        opt.cameraRight[0] = 1.0f;
        opt.cameraFwd[2] = 1.0f;
        opt.cameraNear = zn;
        opt.cameraFar = zf;
        opt.cameraFOV = fovY;
        opt.cameraAspectRatio = aspect;
        opt.colorBuffersHDR = hdr10;
        opt.cameraMotionIncluded = true;
        opt.motionVectorsInvalidValue = 3.40282346638528859811704183484516925440e38f;
        opt.mvecsSubrectSize = {vw, vh};
        opt.depthSubrectSize = {vw, vh};
    }

    // true = ready for frames of w x h. false = frame generation is not available here (the reason is logged,
    // everything it made is released). bgraFrames: SDR frames in B8G8R8A8 order; readBack false = no frame comes back
    // to the CPU (a live stage)
    bool init(uint32_t w, uint32_t h, int genFrames, bool hdr, bool bgraFrames = false, bool readBack = true)
    {
        W = w;
        H = h;
        gen = genFrames;
        hdr10 = hdr;
        bgra = bgraFrames && !hdr;
        ComPtr<IDXGIFactory1> factory;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))))
        {
            LOG("NGX: no DXGI factory\n");
            return false;
        }
        ComPtr<IDXGIAdapter1> adapter;
        for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; i++)
        {
            DXGI_ADAPTER_DESC1 adesc;
            adapter->GetDesc1(&adesc);
            if (adesc.VendorId == 0x10DE)
                break;
            adapter.Reset();
        }
        D3D12_COMMAND_QUEUE_DESC qd{};
        qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        if (!adapter || FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device))) ||
            FAILED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue))) ||
            FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc))) ||
            FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr,
                                             IID_PPV_ARGS(&list))) ||
            FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))))
        {
            LOG("NGX: no D3D12 device on an NVIDIA adapter\n");
            shutdown();
            return false;
        }
        fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);

        ngxUp = coreStart(device.Get());
        if (!ngxUp)
        {
            shutdown();
            return false;
        }
        bool capOk;
        {
            std::lock_guard<std::mutex> lk(callLock());
            capOk = NVSDK_NGX_SUCCEED(NVSDK_NGX_D3D12_GetCapabilityParameters(&params)) && params;
        }
        if (!capOk)
        {
            LOG("NGX: no capability parameters\n");
            shutdown();
            return false;
        }
        params->Get(NVSDK_NGX_DLSSG_Parameter_MultiFrameCountMax, &maxGen);
        if (!createFeature())
        {
            shutdown();
            return false;
        }

        const uint32_t vw = panvec::vecSize(W), vh = panvec::vecSize(H);
        rowPitch = (W * 4 + D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1) & ~(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1);
        const UINT64 bufBytes = (UINT64)rowPitch * H;
        if (!makeTex(frameFormat(), W, H, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, texFrame) ||
            !makeTex(DXGI_FORMAT_R32_FLOAT, vw, vh, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST,
                     texDepth) ||
            !makeTex(DXGI_FORMAT_R16G16_FLOAT, vw, vh, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                     D3D12_RESOURCE_STATE_COPY_DEST, texMvec) ||
            !makeTex(frameFormat(), W, H,
                     D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                     D3D12_RESOURCE_STATE_UNORDERED_ACCESS, texOut) ||
            !makeBuf(D3D12_HEAP_TYPE_UPLOAD, bufBytes, D3D12_RESOURCE_STATE_GENERIC_READ, uploadBuf) ||
            (readBack && !makeBuf(D3D12_HEAP_TYPE_READBACK, bufBytes, D3D12_RESOURCE_STATE_COPY_DEST, readbackBuf)) ||
            FAILED(uploadBuf->Map(0, nullptr, (void**)&upload)) ||
            (readBack && FAILED(readbackBuf->Map(0, nullptr, (void**)&readback))))
        {
            LOG("NGX: resource creation failed\n");
            shutdown();
            return false;
        }
        // depth = a flat 0.5 and vectors = zero, once: a video has neither
        {
            D3D12_TEXTURE_COPY_LOCATION dst{texDepth.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            D3D12_TEXTURE_COPY_LOCATION src{uploadBuf.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
            for (uint32_t y = 0; y < vh; y++)
            {
                float* row = (float*)(upload + (size_t)y * rowPitch);
                for (uint32_t x = 0; x < vw; x++)
                    row[x] = 0.5f;
            }
            src.PlacedFootprint.Footprint = {DXGI_FORMAT_R32_FLOAT, vw, vh, 1, rowPitch};
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            transition(texDepth.Get(), D3D12_RESOURCE_STATE_COPY_DEST, kSrv);
            bool ok = submit();
            memset(upload, 0, (size_t)rowPitch * vh);
            dst.pResource = texMvec.Get();
            src.PlacedFootprint.Footprint = {DXGI_FORMAT_R16G16_FLOAT, vw, vh, 1, rowPitch};
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            transition(texMvec.Get(), D3D12_RESOURCE_STATE_COPY_DEST, kSrv);
            ok = ok && submit();
            if (!ok)
            {
                LOG("NGX: the depth and vector textures did not fill\n");
                shutdown();
                return false;
            }
        }
        panOn = panTex.init(device.Get(), texMvec.Get(), W, H);
        if (!panOn)
            LOG("pan vectors: the descriptor heaps did not allocate, zero vectors\n");
        fillOptions();
        return true;
    }

    // one frame of the clip: uploaded, measured for the pan, then one evaluate per generated frame of the pair it
    // closes. outs = `gen` buffers of W * H * 4 bytes in temporal order; null = nothing is read back (the warm-up)
    bool frame(const uint8_t* rgba, uint8_t* const* outs)
    {
        for (uint32_t y = 0; y < H; y++)
            memcpy(upload + (size_t)y * rowPitch, rgba + (size_t)y * W * 4, (size_t)W * 4);
        const panvec::Result panR = panOn ? pan.run(rgba, W, H, (size_t)W * 4, hdr10) : panvec::Result{};
        const DXGI_FORMAT fmt = frameFormat();
        for (int k = 1; k <= gen; k++)
        {
            if (k == 1)
            {
                if (panOn)
                    panTex.record(list.Get(), texMvec.Get(), panR);
                D3D12_TEXTURE_COPY_LOCATION dst{texFrame.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
                D3D12_TEXTURE_COPY_LOCATION src{uploadBuf.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
                src.PlacedFootprint.Footprint = {fmt, W, H, 1, rowPitch};
                list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
                transition(texFrame.Get(), D3D12_RESOURCE_STATE_COPY_DEST, kSrv);
            }
            NVSDK_NGX_D3D12_DLSSG_Eval_Params ep{};
            ep.pBackbuffer = texFrame.Get();
            ep.pDepth = texDepth.Get();
            ep.pMVecs = texMvec.Get();
            ep.pOutputInterpFrame = texOut.Get();
            opt.multiFrameCount = (unsigned int)gen;
            opt.multiFrameIndex = (unsigned int)k;
            opt.reset = resetNext;
            NVSDK_NGX_Result r;
            {
                std::lock_guard<std::mutex> lk(callLock());
                r = NGX_D3D12_EVALUATE_DLSSG(list.Get(), feature, params, &ep, &opt);
            }
            if (NVSDK_NGX_FAILED(r))
            {
                LOG("NGX: evaluate failed (0x%08x)\n", (unsigned)r);
                return false;
            }
            if (k == gen)
            {
                resetNext = false;
                transition(texFrame.Get(), kSrv, D3D12_RESOURCE_STATE_COPY_DEST);
            }
            if (outs)
            {
                transition(texOut.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
                D3D12_TEXTURE_COPY_LOCATION dst{readbackBuf.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
                dst.PlacedFootprint.Footprint = {fmt, W, H, 1, rowPitch};
                D3D12_TEXTURE_COPY_LOCATION src{texOut.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
                list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
                transition(texOut.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            }
            if (!submit())
            {
                LOG("NGX: the GPU did not finish an evaluate (device removed: 0x%08lx)\n",
                    device->GetDeviceRemovedReason());
                return false;
            }
            if (outs)
                for (uint32_t y = 0; y < H; y++)
                    memcpy(outs[k - 1] + (size_t)y * W * 4, readback + (size_t)y * rowPitch, (size_t)W * 4);
        }
        return true;
    }

    // ---- a live stage: the same calls as frame(), each queued on a list of its own and none waited for. A frame is
    // complete once sFence reaches the value its call returned (0 = the call failed)

    bool stageInit(ID3D12Fence* stageFence)
    {
        sFence = stageFence;
        for (int i = 0; i < kStageLists; i++)
            if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&sAlloc[i]))) ||
                FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, sAlloc[i].Get(), nullptr,
                                                 IID_PPV_ARGS(&sList[i]))) ||
                FAILED(sList[i]->Close()))
                return false;
        return true;
    }

    bool stageWait(uint64_t value)
    {
        if (sFence->GetCompletedValue() >= value)
            return true;
        return SUCCEEDED(sFence->SetEventOnCompletion(value, fenceEvent)) &&
               WaitForSingleObject(fenceEvent, 5000) == WAIT_OBJECT_0;
    }

    ID3D12GraphicsCommandList* stageList()
    {
        const int i = (int)(sLists++ % kStageLists);
        if (!stageWait(sUse[i]) || FAILED(sAlloc[i]->Reset()) || FAILED(sList[i]->Reset(sAlloc[i].Get(), nullptr)))
            return nullptr;
        sCur = i;
        return sList[i].Get();
    }

    uint64_t stageRun(ID3D12GraphicsCommandList* l)
    {
        if (FAILED(l->Close()))
            return 0;
        ID3D12CommandList* ls[] = {l};
        queue->ExecuteCommandLists(1, ls);
        if (FAILED(queue->Signal(sFence, ++sValue)))
            return 0;
        sUse[sCur] = sValue;
        return sValue;
    }

    // an input into the frame texture: CPU bytes (measured for the pan, the result kept in lastPan), or a texture of
    // this device resting in COMMON once srcFence reaches srcValue (an upstream stage's ring slot: the vectors `given`,
    // which the stage derives from the slots' motion, else zero), or neither = the input before it again (zero)
    bool stageInput(const uint8_t* rgba, ID3D12Resource* src = nullptr, ID3D12Fence* srcFence = nullptr,
                    uint64_t srcValue = 0, const panvec::Result* given = nullptr)
    {
        panvec::Result panR{};
        if (rgba)
        {
            // the GPU may still copy the input before out of the upload buffer
            if (!stageWait(sUpload))
                return false;
            for (uint32_t y = 0; y < H; y++)
                memcpy(upload + (size_t)y * rowPitch, rgba + (size_t)y * W * 4, (size_t)W * 4);
            if (panOn)
                panR = pan.run(rgba, W, H, (size_t)W * 4, hdr10);
            lastPan = panR;
        }
        else if (src && given)
            panR = *given;
        ID3D12GraphicsCommandList* l = stageList();
        if (!l)
            return false;
        if (panOn)
            panTex.record(l, texMvec.Get(), panR);
        if (rgba)
        {
            D3D12_TEXTURE_COPY_LOCATION dst{texFrame.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            D3D12_TEXTURE_COPY_LOCATION from{uploadBuf.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
            from.PlacedFootprint.Footprint = {frameFormat(), W, H, 1, rowPitch};
            l->CopyTextureRegion(&dst, 0, 0, 0, &from, nullptr);
        }
        else if (src)
        {
            transition(l, src, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
            l->CopyResource(texFrame.Get(), src);
            transition(l, src, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        }
        transition(l, texFrame.Get(), D3D12_RESOURCE_STATE_COPY_DEST, kSrv);
        if (src && srcFence)
            queue->Wait(srcFence, srcValue); // the slot's copy on its own device is complete
        const uint64_t v = stageRun(l);
        if (rgba)
            sUpload = v;
        return v != 0;
    }

    // generated frame k of `gen` between the last two inputs, copied into slot (a texture resting in COMMON; null =
    // made and left where it is, the warm-up)
    uint64_t stageTween(int k, ID3D12Resource* slot)
    {
        ID3D12GraphicsCommandList* l = stageList();
        if (!l)
            return 0;
        NVSDK_NGX_D3D12_DLSSG_Eval_Params ep{};
        ep.pBackbuffer = texFrame.Get();
        ep.pDepth = texDepth.Get();
        ep.pMVecs = texMvec.Get();
        ep.pOutputInterpFrame = texOut.Get();
        opt.multiFrameCount = (unsigned int)gen;
        opt.multiFrameIndex = (unsigned int)k;
        opt.reset = resetNext;
        NVSDK_NGX_Result r;
        {
            std::lock_guard<std::mutex> lk(callLock());
            r = NGX_D3D12_EVALUATE_DLSSG(l, feature, params, &ep, &opt);
        }
        if (NVSDK_NGX_FAILED(r))
        {
            LOG("NGX: evaluate failed (0x%08x)\n", (unsigned)r);
            return 0;
        }
        if (k == gen)
            resetNext = false;
        if (slot)
        {
            transition(l, texOut.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
            transition(l, slot, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
            l->CopyResource(slot, texOut.Get());
            transition(l, slot, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
            transition(l, texOut.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        return stageRun(l);
    }

    // the input itself copied into slot (null = not copied), and the frame texture ready for the next input
    uint64_t stageEnd(ID3D12Resource* slot)
    {
        ID3D12GraphicsCommandList* l = stageList();
        if (!l)
            return 0;
        if (slot)
        {
            transition(l, texFrame.Get(), kSrv, D3D12_RESOURCE_STATE_COPY_SOURCE);
            transition(l, slot, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
            l->CopyResource(slot, texFrame.Get());
            transition(l, slot, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
            transition(l, texFrame.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
        }
        else
            transition(l, texFrame.Get(), kSrv, D3D12_RESOURCE_STATE_COPY_DEST);
        return stageRun(l);
    }

    // a new clip on this server: what DLSS-G keeps between frames is the feature's, so the feature is made again and
    // the pan estimator starts from nothing; the next frame is a clip's first
    bool newClip()
    {
        releaseFeature();
        pan = panvec::Estimator();
        lastPan = panvec::Result();
        return createFeature();
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

    void shutdown()
    {
        releaseFeature();
        if (params)
        {
            std::lock_guard<std::mutex> lk(callLock());
            NVSDK_NGX_D3D12_DestroyParameters(params);
        }
        params = nullptr;
        if (ngxUp)
            coreStop();
        ngxUp = false;
        panTex = panvec::Writer();
        texFrame.Reset();
        texDepth.Reset();
        texMvec.Reset();
        texOut.Reset();
        uploadBuf.Reset();
        readbackBuf.Reset();
        upload = nullptr;
        readback = nullptr;
        for (int i = 0; i < kStageLists; i++)
        {
            sList[i].Reset();
            sAlloc[i].Reset();
        }
        sFence = nullptr;
        list.Reset();
        alloc.Reset();
        fence.Reset();
        queue.Reset();
        device.Reset();
        if (fenceEvent)
            CloseHandle(fenceEvent);
        fenceEvent = nullptr;
    }
};

} // namespace ngxfg
