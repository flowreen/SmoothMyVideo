// nr_host.h - reusable DLSS 5 Neural Rendering core for SmoothMyVideo.
//
// Device-agnostic in the sense that the caller may hand in its own D3D12 device
// (smv-live.exe) or let the core create a private one (dlssnr.exe, the offline
// pipe server). Everything DLSS 5 specific lives here; main.cpp is only the pipe
// server around it.
//
// Nothing here is copied from any third party host. The NGX call order, the
// parameter key strings and feature id 18 were established by probing the runtime.
#pragma once

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <wrl/client.h>

#include <cstdint>
#include <string>

#include <nvsdk_ngx.h>
#include <nvsdk_ngx_params.h>

namespace nr {

// NGX feature id of DLSS 5 Neural Rendering. Not in the public SDK headers.
const int kFeatureId = 18;

// The project id the NR snippet accepts. A plain Init_Ext session gets
// CreateFeature(18) refused.
extern const char* const kProjectId;

// Fixed internally, not exposed in the GUI: only structure and tone are user facing.
struct Settings
{
    float structure = 1.0f;   // DLSSNR.LocalStructureStrength, 0..2
    float tone      = 1.0f;   // DLSSNR.LocalToneStrength, 0..2
    float intensity = 1.0f;   // DLSSNR.Intensity, 0..2
    int   style     = 1;      // DLSSNR.Style, 0 default / 1 natural / 2 cinematic
    int   preset    = 3;      // DLSSNR.Hint.Render.Preset, 0..3
    int   automask  = 1;      // DLSSNR.UseAutoMask, as every reference host sets it (SMV_NR_AUTOMASK=0 = off)
    // DLSSNR.MVec: a per-frame R16G16_FLOAT motion field (current -> previous, px) that the caller
    // writes into the shared motion buffer (startShared); without it the runtime gets no motion
    bool  motion    = false;
    // the pass run 1..kMaxPasses times in a chain on every frame: pass k's output is pass k + 1's
    // input, each pass its OWN feature and history (one feature called twice a frame would see two
    // evaluations with no motion between them), all fed the same motion field
    int   passes    = 1;
};

const int kMaxPasses = 10;

// Probe knobs. The snippet validates its caller and the exact rule is
// unknown, so every plausible route is reachable without a rebuild.
struct Variant
{
    // Route the NGX entry points through the caller shim (a DLL named nvngx.dll)
    // so the return address lands inside it. false = call the driver core direct.
    bool useShim = true;
    // Where the shim is loaded from: 0 = beside the exe, 1 = <exe>\caller\nvngx.dll,
    // 2 = beside the snippet DLL.
    int shimLocation = 0;
    // true = Init_ProjectID (the documented working route), false = Init_Ext.
    bool initProjectId = true;
    // Argument order of the Init_ProjectID export, unverifiable from the public
    // headers: 0 = (.., device, featureInfo, version), 1 = (.., device, version, featureInfo).
    int initArgOrder = 1;     // 1 is the order that survives Init_ProjectID on 616.56 (order 0 faults inside the core)
    // true = CreateFeature / EvaluateFeature / ReleaseFeature resolved from the NR
    // snippet's OWN exports (it exports the whole NVSDK_NGX_D3D12_* API) after its
    // own Init_Ext; false = through the driver core, whose feature table on 616.56
    // refuses id 18 before touching any snippet (per the NGX log).
    bool viaSnippet = true;
};

class Host
{
public:
    ~Host();

    // Full bring-up: resolve the driver core and the NR snippet, create a D3D12
    // device, init NGX, create feature 18 and the FP16 color/output textures.
    // Returns 0 on success, 2 when the runtime is missing or the GPU/driver does
    // not support it, 3 when CreateFeature(18) is refused. quietLog as in startupOn
    // (smv-live.exe's offline host: the NGX chatter stays out of
    // the render log, dlssnr.py drops it the same way).
    int startup(uint32_t w, uint32_t h, const Settings& s, const Variant& v, std::string& err,
                bool quietLog = false);

    // Hosted in smv-live.exe: the same bring-up on the CALLER's device and
    // queue. The host owns the Color / Output textures and a private command list
    // for CreateFeature; the caller records the per-frame work (see evaluateOn) on
    // its own lists and never touches CPU staging. quietLog = no NGX log callback
    // output on stderr (the driver core still writes its own nvngx.log in the
    // module folder). Same return codes as startup.
    int startupOn(ID3D12Device* device, ID3D12CommandQueue* queue, uint32_t w, uint32_t h,
                  const Settings& s, const Variant& v, bool quietLog, std::string& err);

    // Record the evaluate on the caller's OPEN command list. Contract at the call:
    // color() in NON_PIXEL_SHADER_RESOURCE holding the frame, output() in
    // UNORDERED_ACCESS; the result lands in output() in that state. The list's
    // descriptor heaps, root signature and pipeline state are clobbered by NGX:
    // re-bind them afterwards.
    bool evaluateOn(ID3D12GraphicsCommandList* list, bool reset, std::string& err);
    ID3D12Resource* color() const { return m_color.Get(); }
    ID3D12Resource* output() const { return m_output.Get(); }

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
    HANDLE sharedInHandle() const { return m_shInH; }
    HANDLE sharedOutHandle() const { return m_shOutH; }
    HANDLE sharedFenceHandle() const { return m_shFenceH; }
    uint64_t sharedBytes() const { return m_shBytes; }
    uint64_t rowPitch() const { return m_rowPitch; }
    // Settings::motion: a third shared buffer, the motion field in the MVec texture's copy
    // footprint (mvRowPitch() bytes per row, R16G16_FLOAT), copied into MVec before each evaluate
    HANDLE sharedMvHandle() const { return m_shMvH; }
    uint64_t sharedMvBytes() const { return m_shMvBytes; }
    uint64_t mvRowPitch() const { return m_mvPitch; }
    LUID adapterLuid() const;

    // One frame through the shared buffers: the queue waits until the fence reaches waitValue
    // (the caller signals it once the input buffer holds the frame), copies it into Color,
    // evaluates, copies Output into the output buffer and signals signalValue. Returns without
    // waiting for the GPU; a command allocator still in flight is waited for, so at most
    // kSharedLists frames are queued.
    bool submitShared(bool reset, uint64_t waitValue, uint64_t signalValue, std::string& err);

    // The passes actually built (Settings::passes, fewer when a later feature could not be created:
    // passNote() says why).
    int passes() const { return m_passes; }
    const std::string& passNote() const { return m_passNote; }

    // Last NGX result seen, for the probe table.
    NVSDK_NGX_Result lastResult() const { return m_last; }
    const std::wstring& corePath() const { return m_corePath; }
    const std::wstring& snippetPath() const { return m_snippetPath; }
    const std::wstring& shimPath() const { return m_shimPath; }

    void shutdown();
    void ngxShutdown();  // the NGX-only teardown, called under SEH by shutdown()

    // Drop every reference WITHOUT any NGX call, for a host that leaves through ExitProcess:
    // the NR snippet's release and shutdown chain faults (measured in smv-live.exe,
    // "NGX teardown faulted" and then 0xC0000005 on the way out), and dlssnr.exe never calls it
    // either. The OS reclaims the session. A later shutdown() or the destructor is a no-op.
    void abandon();

private:
    bool resolveModules(const Variant& v, std::string& err);
    bool createDevice(std::string& err);
    bool createCommandObjects(std::string& err);
    bool createResources(bool staging, std::string& err);
    int  initNgx(std::string& err);
    bool runCommandList(std::string& err);

    bool m_external = false;   // startupOn: device and queue belong to the caller
    bool m_quiet = false;      // no NGX log callback output

    template <class T> using CP = Microsoft::WRL::ComPtr<T>;

    uint32_t m_w = 0, m_h = 0;
    Settings m_set;
    Variant  m_var;

    HMODULE m_core = nullptr;     // _nvngx.dll, the driver core
    HMODULE m_snippet = nullptr;  // nvngx_dlssnr.dll, the NR layer
    HMODULE m_shim = nullptr;     // nvngx.dll, our caller shim
    std::wstring m_corePath, m_snippetPath, m_shimPath;

    CP<ID3D12Device>              m_dev;
    CP<ID3D12CommandQueue>        m_queue;
    CP<ID3D12CommandAllocator>    m_alloc;
    CP<ID3D12GraphicsCommandList> m_list;
    CP<ID3D12Fence>               m_fence;
    HANDLE   m_fenceEvent = nullptr;
    uint64_t m_fenceValue = 0;

    CP<ID3D12Resource> m_color;    // RGBA16F, NON_PIXEL_SHADER_RESOURCE
    CP<ID3D12Resource> m_output;   // RGBA16F, UNORDERED_ACCESS
    CP<ID3D12Resource> m_upload;   // CPU write, linear
    CP<ID3D12Resource> m_readback; // CPU read, linear
    uint64_t m_rowPitch = 0;       // aligned row pitch of the staging buffers
    CP<ID3D12Resource> m_mv;       // Settings::motion: R16G16_FLOAT, NON_PIXEL_SHADER_RESOURCE (DLSSNR.MVec)
    uint64_t m_mvPitch = 0;        // its copy footprint row pitch

    static const int kSharedLists = 2;
    CP<ID3D12Resource> m_shIn;     // shared input buffer, COMMON between lists
    CP<ID3D12Resource> m_shOut;    // shared output buffer, COMMON between lists
    CP<ID3D12Fence>    m_shFence;  // shared with the caller's CUDA stream
    CP<ID3D12Resource> m_shMv;     // Settings::motion: the shared motion buffer, COMMON between lists
    HANDLE   m_shInH = nullptr, m_shOutH = nullptr, m_shFenceH = nullptr, m_shMvH = nullptr;
    uint64_t m_shBytes = 0, m_shMvBytes = 0;
    CP<ID3D12CommandAllocator>    m_shAlloc[kSharedLists];
    CP<ID3D12GraphicsCommandList> m_shList[kSharedLists];
    uint64_t m_shDone[kSharedLists] = {};   // fence value that frees each allocator
    int      m_shNext = 0;
    void closeSharedHandles();

    NVSDK_NGX_Parameter* m_params = nullptr;   // pass 0 (m_pparams[0])
    NVSDK_NGX_Handle*    m_feature = nullptr;  // pass 0 (m_pfeature[0])
    NVSDK_NGX_Parameter* m_pparams[kMaxPasses] = {};
    NVSDK_NGX_Handle*    m_pfeature[kMaxPasses] = {};
    CP<ID3D12Resource>   m_passOut[kMaxPasses - 1];   // pass k's output = pass k + 1's input (UAV between frames)
    int m_passes = 1;
    std::string m_passNote;
    NVSDK_NGX_Result     m_last = NVSDK_NGX_Result_Success;
    bool m_ngxUp = false;
    void setPassParams(int k, int last);   // the create-time keys of pass k (of 0..last) on m_pparams[k]
};

// Folder holding nvngx_dlssnr.dll, the caller shim nvngx.dll and the NGX log (the data path).
// Default = the folder of the running exe (dlssnr.exe lives in engine\dlssnr). A host that
// lives elsewhere (smv-live.exe in engine\live, phase 2) points this at engine\dlssnr before
// startup. Empty or null restores the default.
void setModuleDir(const wchar_t* dir);

// Human readable NGX result, including the codes the public header names.
std::string resultString(NVSDK_NGX_Result r);

} // namespace nr
