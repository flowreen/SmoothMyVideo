// nr_host.h - reusable DLSS 5 Neural Rendering core for SmoothMyVideo.
//
// The core creates its own D3D12 device: smv-live.exe hands the frames over through
// shared buffers (startShared) or through CPU staging (renderFrame).
// Everything DLSS 5 specific lives here.
//
// Nothing here is copied from any third party host. The NGX call order, the
// parameter key strings and feature id 18 were established by probing the runtime.
#pragma once

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include <cstdint>
#include <string>

#include <nvsdk_ngx.h>
#include <nvsdk_ngx_params.h>

namespace nr
{

// NGX feature id of DLSS 5 Neural Rendering. Not in the public SDK headers.
const int kFeatureId = 18;

// The project id the NR snippet accepts. A plain Init_Ext session gets
// CreateFeature(18) refused.
extern const char* const kProjectId;

// Fixed internally, not exposed in the GUI: only structure and tone are user facing.
struct Settings
{
    float structure = 1.0f; // DLSSNR.LocalStructureStrength, 0..2
    float tone = 1.0f;      // DLSSNR.LocalToneStrength, 0..2
    float intensity = 1.0f; // DLSSNR.Intensity, 0..2
    int style = 1;          // DLSSNR.Style, 0 default / 1 natural / 2 cinematic
    int preset = 3;         // DLSSNR.Hint.Render.Preset, 0..3
    int automask = 1;       // DLSSNR.UseAutoMask, as every reference host sets it (SMV_NR_AUTOMASK=0 = off)
    // DLSSNR.MVec: a per-frame R16G16_FLOAT motion field (current -> previous, px) that the caller
    // writes into the shared motion buffer (startShared); without it the runtime gets no motion
    bool motion = false;
    // the pass run 1..kMaxPasses times in a chain on every frame: pass k's output is pass k + 1's
    // input, each pass its OWN feature and history (one feature called twice a frame would see two
    // evaluations with no motion between them), all fed the same motion field
    int passes = 1;
};

const int kMaxPasses = 10;

// What a chain needs beyond the creation of its passes, in MiB: the caller's handoff and motion
// buffers and the first frames' one-time share (kAfterBase + kAfterMp a megapixel), and each pass's
// working memory (kFrameMp a megapixel). Measured on nvngx_dlssnr 310.8 at 854x480, 1920x1080 and
// 3840x2160 with 1 and 4 passes (123 / 253 / 745 and 7 / 34 / 128 MiB), a quarter on top.
const double kAfterBase = 128.0, kAfterMp = 96.0, kFrameMp = 20.0;

// What each pass takes at its creation, in MiB: kPassBase + kPassMp a megapixel (186 / 360 / 496 / 933
// MiB at 854x480 / 1920x1080 / 2560x1440 / 3840x2160 on nvngx_dlssnr 310.8). A caller that sizes its
// own memory before the chain exists prices the chain with these and the shares above.
const double kPassBase = 147.0, kPassMp = 95.0;

class Host
{
  public:
    ~Host();

    // Full bring-up: resolve the driver core and the NR snippet, create a D3D12
    // device, init NGX, create feature 18 and the FP16 color/output textures.
    // Returns 0 on success, 2 when the runtime is missing or the GPU/driver does
    // not support it, 3 when CreateFeature(18) is refused. quietLog = no NGX log callback output
    // on stderr (smv-live.exe: the NGX chatter stays out of the render log; the driver core still
    // writes its own nvngx.log in the module folder). adapter = the LUID of the adapter to create
    // the device on (the caller's CUDA device, so the zero-copy handoff shares one GPU), else the
    // first hardware adapter, which is not the NVIDIA one when another GPU drives the main display.
    int startup(uint32_t w, uint32_t h, const Settings& s, std::string& err, bool quietLog = false,
                const LUID* adapter = nullptr);

    // One frame in, one frame out. src and dst are w*h*8 bytes of RGBA16F.
    // reset must be true for the first frame of a stream and after any
    // discontinuity; the runtime keeps temporal history otherwise.
    bool renderFrame(const void* src, void* dst, bool reset, std::string& err);

    // Zero-copy handoff for a caller that renders with CUDA on the same GPU, after startup():
    // two shared DEFAULT-heap buffers carry the frame in and out in the copy footprint layout
    // (rowPitch() bytes per row, sharedBytes() in all) and a shared fence orders the caller's
    // work and the evaluate on the GPU, so no frame crosses the CPU. false = a resource or a
    // handle could not be made (err says which); renderFrame stays usable either way.
    bool startShared(std::string& err);
    HANDLE sharedInHandle() const
    {
        return m_shInH;
    }
    HANDLE sharedOutHandle() const
    {
        return m_shOutH;
    }
    HANDLE sharedFenceHandle() const
    {
        return m_shFenceH;
    }
    uint64_t sharedBytes() const
    {
        return m_shBytes;
    }
    uint64_t rowPitch() const
    {
        return m_rowPitch;
    }
    // Settings::motion: a third shared buffer, the motion field in the MVec texture's copy
    // footprint (mvRowPitch() bytes per row, R16G16_FLOAT), copied into MVec before each evaluate
    HANDLE sharedMvHandle() const
    {
        return m_shMvH;
    }
    uint64_t sharedMvBytes() const
    {
        return m_shMvBytes;
    }
    uint64_t mvRowPitch() const
    {
        return m_mvPitch;
    }
    LUID adapterLuid() const;

    // One frame through the shared buffers: the queue waits until the fence reaches waitValue
    // (the caller signals it once the input buffer holds the frame), copies it into Color,
    // evaluates, copies Output into the output buffer and signals signalValue. Returns without
    // waiting for the GPU; a command allocator still in flight is waited for, so at most
    // kSharedLists frames are queued.
    bool submitShared(bool reset, uint64_t waitValue, uint64_t signalValue, std::string& err);

    // The passes actually built (Settings::passes, fewer when a later feature could not be created:
    // passNote() says why).
    int passes() const
    {
        return m_passes;
    }
    const std::string& passNote() const
    {
        return m_passNote;
    }

    void shutdown();
    void ngxShutdown(); // the NGX-only teardown, called under SEH by shutdown()

    // Drop every reference WITHOUT any NGX call, for a host that leaves through ExitProcess:
    // the NR snippet's release and shutdown chain faults (measured in smv-live.exe,
    // "NGX teardown faulted" and then 0xC0000005 on the way out). The OS reclaims
    // the session. A later shutdown() or the destructor is a no-op.
    void abandon();

  private:
    bool resolveModules(std::string& err);
    bool createDevice(std::string& err);
    bool createCommandObjects(std::string& err);
    bool createResources(std::string& err);
    int initNgx(std::string& err);
    bool runCommandList(std::string& err);
    // Record the evaluate on an OPEN command list (renderFrame's and submitShared's). Contract at
    // the call: m_color in NON_PIXEL_SHADER_RESOURCE holding the frame, m_output in
    // UNORDERED_ACCESS; the result lands in m_output in that state. The list's descriptor heaps,
    // root signature and pipeline state are clobbered by NGX.
    bool evaluateOn(ID3D12GraphicsCommandList* list, bool reset, std::string& err);

    bool m_wantAdapter = false; // startup's adapter LUID was given
    LUID m_wantLuid = {};
    bool m_quiet = false; // no NGX log callback output

    template <class T> using CP = Microsoft::WRL::ComPtr<T>;

    uint32_t m_w = 0, m_h = 0;
    Settings m_set;

    HMODULE m_core = nullptr;    // _nvngx.dll, the driver core
    HMODULE m_snippet = nullptr; // nvngx_dlssnr.dll, the NR layer
    HMODULE m_shim = nullptr;    // nvngx.dll, our caller shim
    std::wstring m_corePath, m_snippetPath;

    CP<ID3D12Device> m_dev;
    CP<ID3D12CommandQueue> m_queue;
    CP<ID3D12CommandAllocator> m_alloc;
    CP<ID3D12GraphicsCommandList> m_list;
    CP<ID3D12Fence> m_fence;
    HANDLE m_fenceEvent = nullptr;
    uint64_t m_fenceValue = 0;

    CP<ID3D12Resource> m_color;    // RGBA16F, NON_PIXEL_SHADER_RESOURCE
    CP<ID3D12Resource> m_output;   // RGBA16F, UNORDERED_ACCESS
    CP<ID3D12Resource> m_upload;   // CPU write, linear
    CP<ID3D12Resource> m_readback; // CPU read, linear
    uint64_t m_rowPitch = 0;       // aligned row pitch of the staging buffers
    CP<ID3D12Resource> m_mv;       // Settings::motion: R16G16_FLOAT, NON_PIXEL_SHADER_RESOURCE (DLSSNR.MVec)
    uint64_t m_mvPitch = 0;        // its copy footprint row pitch

    static const int kSharedLists = 2;
    CP<ID3D12Resource> m_shIn;  // shared input buffer, COMMON between lists
    CP<ID3D12Resource> m_shOut; // shared output buffer, COMMON between lists
    CP<ID3D12Fence> m_shFence;  // shared with the caller's CUDA stream
    CP<ID3D12Resource> m_shMv;  // Settings::motion: the shared motion buffer, COMMON between lists
    HANDLE m_shInH = nullptr, m_shOutH = nullptr, m_shFenceH = nullptr, m_shMvH = nullptr;
    uint64_t m_shBytes = 0, m_shMvBytes = 0;
    CP<ID3D12CommandAllocator> m_shAlloc[kSharedLists];
    CP<ID3D12GraphicsCommandList> m_shList[kSharedLists];
    uint64_t m_shDone[kSharedLists] = {}; // fence value that frees each allocator
    int m_shNext = 0;
    void closeSharedHandles();

    NVSDK_NGX_Parameter* m_params = nullptr; // pass 0 (m_pparams[0])
    NVSDK_NGX_Handle* m_feature = nullptr;   // pass 0 (m_pfeature[0])
    NVSDK_NGX_Parameter* m_pparams[kMaxPasses] = {};
    NVSDK_NGX_Handle* m_pfeature[kMaxPasses] = {};
    CP<ID3D12Resource> m_passOut[kMaxPasses - 1]; // pass k's output = pass k + 1's input (UAV between frames)
    int m_passes = 1;
    std::string m_passNote;
    NVSDK_NGX_Result m_last = NVSDK_NGX_Result_Success;
    bool m_ngxUp = false;
    void setPassParams(int k, int last); // the create-time keys of pass k (of 0..last) on m_pparams[k]
};

// Folder holding nvngx_dlssnr.dll, the caller shim nvngx.dll and the NGX log (the data path).
// Default = the folder of the running exe; smv-live.exe (in engine\live) points this at
// engine\dlssnr before startup. Empty or null restores the default.
void setModuleDir(const wchar_t* dir);

// Human readable NGX result, including the codes the public header names.
std::string resultString(NVSDK_NGX_Result r);

} // namespace nr
