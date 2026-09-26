// nr_host.cpp - see nr_host.h. Reimplemented from the recorded facts, nothing
// copied from any third party host.
#include "nr_host.h"
#include "shim_abi.h"

#include <dxgi1_6.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cstdlib>

#pragma comment(lib, "advapi32.lib")

using Microsoft::WRL::ComPtr;

namespace nr {

// ---------------------------------------------------------------------------
// Caller-validation shim for the NR snippet.
//
// The leaked NR runtime (nvngx_dlssnr.dll 310.8.0) answers unknown callers with
// 0xBAD00002. The RenoDX add-on shipped in the same pack passes by patching the
// snippet's own GetModuleFileNameW import (its log strings: "signed feature has
// no GetModuleFileNameW import", "failed to make signed-feature IAT writable"),
// i.e. the snippet asks Windows for its caller module's file name and checks the
// basename. We do the same to our OWN process, loading a DLL the user supplied:
// replace GetModuleFileName{W,A} in the snippet's import table with a thunk that,
// for the snippet's queries, reports an allowed basename (SMV_NR_SPOOF, default
// nvngx.dll = our shim). SMV_NR_NOHOOK disables it, SMV_NR_HOOKLOG traces queries.
// ---------------------------------------------------------------------------
typedef DWORD (WINAPI* PFN_GMFNW)(HMODULE, LPWSTR, DWORD);
typedef DWORD (WINAPI* PFN_GMFNA)(HMODULE, LPSTR, DWORD);
static PFN_GMFNW s_realGMFNW = nullptr;
static PFN_GMFNA s_realGMFNA = nullptr;

static const char* spoofName() { const char* s = getenv("SMV_NR_SPOOF"); return (s && *s) ? s : "nvngx.dll"; }
static bool hookLog() { return getenv("SMV_NR_HOOKLOG") != nullptr; }

static void replaceBasenameW(wchar_t* buf, DWORD size, const char* base)
{
    wchar_t* slash = wcsrchr(buf, L'\\');
    size_t keep = slash ? (size_t)(slash + 1 - buf) : 0;
    wchar_t wbase[MAX_PATH];
    mbstowcs(wbase, base, MAX_PATH - 1); wbase[MAX_PATH - 1] = 0;
    if (keep + wcslen(wbase) + 1 > size) return;
    wcscpy(buf + keep, wbase);
}

static DWORD WINAPI hookGMFNW(HMODULE h, LPWSTR buf, DWORD size)
{
    DWORD n = s_realGMFNW ? s_realGMFNW(h, buf, size) : 0;
    if (hookLog()) { fprintf(stderr, "dlssnr: [hook] GetModuleFileNameW(%p) -> \"%ls\"\n", (void*)h, buf); fflush(stderr); }
    replaceBasenameW(buf, size, spoofName());
    return (DWORD)wcslen(buf);
}

static DWORD WINAPI hookGMFNA(HMODULE h, LPSTR buf, DWORD size)
{
    DWORD n = s_realGMFNA ? s_realGMFNA(h, buf, size) : 0;
    if (hookLog()) { fprintf(stderr, "dlssnr: [hook] GetModuleFileNameA(%p) -> \"%s\"\n", (void*)h, buf); fflush(stderr); }
    char* slash = strrchr(buf, '\\');
    size_t keep = slash ? (size_t)(slash + 1 - buf) : 0;
    if (keep + strlen(spoofName()) + 1 <= size) strcpy(buf + keep, spoofName());
    return (DWORD)strlen(buf);
}

// Redirect one named kernel32 import in a module's IAT to hook; caches the real
// pointer. Returns entries changed.
static int patchImport(HMODULE mod, const char* fn, void* hook, void** realOut)
{
    auto base = (BYTE*)mod;
    auto dos = (IMAGE_DOS_HEADER*)base;
    auto nt = (IMAGE_NT_HEADERS*)(base + dos->e_lfanew);
    auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (!dir.VirtualAddress) return 0;
    int changed = 0;
    for (auto desc = (IMAGE_IMPORT_DESCRIPTOR*)(base + dir.VirtualAddress); desc->Name; ++desc)
    {
        const char* dll = (const char*)(base + desc->Name);
        if (_stricmp(dll, "kernel32.dll") != 0 && _strnicmp(dll, "api-ms-win-core", 15) != 0) continue;
        if (!desc->OriginalFirstThunk) continue;
        auto thunk = (IMAGE_THUNK_DATA*)(base + desc->OriginalFirstThunk);
        auto iat = (IMAGE_THUNK_DATA*)(base + desc->FirstThunk);
        for (; thunk->u1.AddressOfData; ++thunk, ++iat)
        {
            if (thunk->u1.Ordinal & IMAGE_ORDINAL_FLAG) continue;
            auto ibn = (IMAGE_IMPORT_BY_NAME*)(base + thunk->u1.AddressOfData);
            if (strcmp(ibn->Name, fn) != 0) continue;
            DWORD old;
            if (!VirtualProtect(&iat->u1.Function, sizeof(void*), PAGE_READWRITE, &old)) continue;
            if (realOut && !*realOut) *realOut = (void*)iat->u1.Function;
            iat->u1.Function = (ULONG_PTR)hook;
            VirtualProtect(&iat->u1.Function, sizeof(void*), old, &old);
            ++changed;
        }
    }
    return changed;
}

static void installCallerShim(HMODULE snippet)
{
    if (getenv("SMV_NR_NOHOOK")) { fprintf(stderr, "dlssnr: caller shim disabled (SMV_NR_NOHOOK)\n"); return; }
    int w = patchImport(snippet, "GetModuleFileNameW", (void*)hookGMFNW, (void**)&s_realGMFNW);
    int a = patchImport(snippet, "GetModuleFileNameA", (void*)hookGMFNA, (void**)&s_realGMFNA);
    fprintf(stderr, "dlssnr: caller shim: GetModuleFileNameW x%d GetModuleFileNameA x%d, spoof=%s\n",
            w, a, spoofName());
    fflush(stderr);
}

const char* const kProjectId = "53f803cc-a12f-4d69-90d5-19b7599cad19";

static const DXGI_FORMAT kFmt = DXGI_FORMAT_R16G16B16A16_FLOAT;

// Direct (no shim) route: the call originates inside dlssnr.exe. Kept so the
// phase 0 probe can show whether the caller validation is real.
static NVSDK_NGX_Result directInit(void* fn, int argOrder, const char* projectId,
                                   NVSDK_NGX_EngineType engineType, const char* engineVersion,
                                   const wchar_t* dataPath, ID3D12Device* device,
                                   const NVSDK_NGX_FeatureCommonInfo* featureInfo,
                                   NVSDK_NGX_Version sdkVersion)
{
    if (argOrder == 0)
        return ((PFN_NGX_InitProjectID_A)fn)(projectId, engineType, engineVersion, dataPath, device,
                                             featureInfo, sdkVersion);
    return ((PFN_NGX_InitProjectID_B)fn)(projectId, engineType, engineVersion, dataPath, device,
                                         sdkVersion, featureInfo);
}

std::string resultString(NVSDK_NGX_Result r)
{
    char buf[64];
    const char* name = "Unknown";
    switch (r)
    {
    case NVSDK_NGX_Result_Success:                       name = "Success"; break;
    case NVSDK_NGX_Result_Fail:                          name = "Fail"; break;
    case NVSDK_NGX_Result_FAIL_FeatureNotSupported:      name = "FeatureNotSupported"; break;
    case NVSDK_NGX_Result_FAIL_PlatformError:            name = "PlatformError"; break;
    case NVSDK_NGX_Result_FAIL_FeatureAlreadyExists:     name = "FeatureAlreadyExists"; break;
    case NVSDK_NGX_Result_FAIL_FeatureNotFound:          name = "FeatureNotFound"; break;
    case NVSDK_NGX_Result_FAIL_InvalidParameter:         name = "InvalidParameter"; break;
    case NVSDK_NGX_Result_FAIL_ScratchBufferTooSmall:    name = "ScratchBufferTooSmall"; break;
    case NVSDK_NGX_Result_FAIL_NotInitialized:           name = "NotInitialized"; break;
    case NVSDK_NGX_Result_FAIL_UnsupportedInputFormat:   name = "UnsupportedInputFormat"; break;
    case NVSDK_NGX_Result_FAIL_RWFlagMissing:            name = "RWFlagMissing"; break;
    case NVSDK_NGX_Result_FAIL_MissingInput:             name = "MissingInput"; break;
    case NVSDK_NGX_Result_FAIL_UnableToInitializeFeature:name = "UnableToInitializeFeature"; break;
    case NVSDK_NGX_Result_FAIL_OutOfDate:                name = "OutOfDate"; break;
    case NVSDK_NGX_Result_FAIL_OutOfGPUMemory:           name = "OutOfGPUMemory"; break;
    case NVSDK_NGX_Result_FAIL_UnsupportedFormat:        name = "UnsupportedFormat"; break;
    case NVSDK_NGX_Result_FAIL_UnableToWriteToAppDataPath: name = "UnableToWriteToAppDataPath"; break;
    case NVSDK_NGX_Result_FAIL_UnsupportedParameter:     name = "UnsupportedParameter"; break;
    case NVSDK_NGX_Result_FAIL_Denied:                   name = "Denied"; break;
    case NVSDK_NGX_Result_FAIL_NotImplemented:           name = "NotImplemented"; break;
    default: break;
    }
    snprintf(buf, sizeof(buf), "0x%08X %s", (unsigned)r, name);
    return std::string(buf);
}

static std::wstring exeDir()
{
    wchar_t path[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, path, MAX_PATH);
    std::wstring s(path);
    size_t cut = s.find_last_of(L"\\/");
    return cut == std::wstring::npos ? std::wstring(L".") : s.substr(0, cut);
}

static std::wstring s_moduleDir;   // see setModuleDir in nr_host.h

void setModuleDir(const wchar_t* dir) { s_moduleDir = (dir && *dir) ? dir : L""; }

// The folder the snippet, the shim and the NGX data path resolve from.
static std::wstring moduleDir() { return s_moduleDir.empty() ? exeDir() : s_moduleDir; }

static bool fileExists(const std::wstring& p)
{
    DWORD a = GetFileAttributesW(p.c_str());
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
}

// The driver core and, on a future official driver, the NR snippet live under
// C:\Windows\System32\DriverStore\FileRepository\nv*.inf_amd64_*. The reference
// host only searched nv_dispi.inf_*; on this machine the core is under
// nvami.inf_amd64_*, so every nv*.inf_amd64_* directory is scanned.
static bool findInDriverStore(const wchar_t* fileName, std::wstring& out)
{
    wchar_t sysdir[MAX_PATH] = {};
    GetSystemDirectoryW(sysdir, MAX_PATH);
    std::wstring root = std::wstring(sysdir) + L"\\DriverStore\\FileRepository\\";
    WIN32_FIND_DATAW fd = {};
    HANDLE h = FindFirstFileW((root + L"nv*.inf_amd64_*").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return false;
    bool found = false;
    do
    {
        if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) continue;
        std::wstring cand = root + fd.cFileName + L"\\" + fileName;
        if (fileExists(cand)) { out = cand; found = true; break; }
    } while (FindNextFileW(h, &fd));
    FindClose(h);
    return found;
}

// The ACTIVE driver package. NGX itself resolves the core through the kernel driver
// (QueryAdapterInfo); the installer mirrors that package path into this registry value.
// Several nv*.inf_amd64_* packages can sit in the DriverStore at once (old drivers are
// not purged on update), and a plain name-order scan picked a stale one (the 616.56 core
// loaded under the 616.64 kernel driver), so this comes first.
static bool ngxCoreFromRegistry(const wchar_t* fileName, std::wstring& out)
{
    wchar_t buf[MAX_PATH * 2] = {};
    DWORD cb = sizeof(buf);
    if (RegGetValueW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\NVIDIA Corporation\\Global\\NGXCore",
                     L"FullPath", RRF_RT_REG_SZ, nullptr, buf, &cb) != ERROR_SUCCESS || !buf[0])
        return false;
    std::wstring cand = std::wstring(buf) + L"\\" + fileName;
    if (!fileExists(cand)) return false;
    out = cand;
    return true;
}

Host::~Host() { shutdown(); }

bool Host::resolveModules(const Variant& v, std::string& err)
{
    const std::wstring dir = moduleDir();

    // 1. the NR snippet: the user drop beside the exe wins, a DriverStore copy
    //    is the fallback for the day an official driver ships one.
    std::wstring snippet = dir + L"\\nvngx_dlssnr.dll";
    if (!fileExists(snippet) && !ngxCoreFromRegistry(L"nvngx_dlssnr.dll", snippet)
        && !findInDriverStore(L"nvngx_dlssnr.dll", snippet))
    {
        err = "nvngx_dlssnr.dll not found beside the host or in the DriverStore";
        return false;
    }
    m_snippetPath = snippet;

    // 2. the driver core: the active package first, the store scan as the fallback.
    std::wstring core;
    if (!ngxCoreFromRegistry(L"_nvngx.dll", core) && !findInDriverStore(L"_nvngx.dll", core))
    {
        wchar_t sysdir[MAX_PATH] = {};
        GetSystemDirectoryW(sysdir, MAX_PATH);
        core = std::wstring(sysdir) + L"\\_nvngx.dll";
        if (!fileExists(core)) { err = "_nvngx.dll (NVIDIA driver core) not found"; return false; }
    }
    m_corePath = core;

    // The snippet is loaded first so the core sees the NR layer.
    m_snippet = LoadLibraryW(m_snippetPath.c_str());
    if (!m_snippet)
    {
        char b[128]; snprintf(b, sizeof(b), "LoadLibrary nvngx_dlssnr.dll failed, GetLastError=%lu", GetLastError());
        err = b; return false;
    }
    // Pass the snippet's caller-name check before any NGX call reaches it.
    installCallerShim(m_snippet);
    m_core = LoadLibraryW(m_corePath.c_str());
    if (!m_core)
    {
        char b[128]; snprintf(b, sizeof(b), "LoadLibrary _nvngx.dll failed, GetLastError=%lu", GetLastError());
        err = b; return false;
    }

    // 3. the caller shim, only when this variant routes through it.
    if (v.useShim)
    {
        std::wstring shim;
        if (v.shimLocation == 1) shim = dir + L"\\caller\\nvngx.dll";
        else if (v.shimLocation == 2)
        {
            std::wstring sd = m_snippetPath.substr(0, m_snippetPath.find_last_of(L"\\/"));
            shim = sd + L"\\nvngx.dll";
        }
        else shim = dir + L"\\nvngx.dll";
        if (!fileExists(shim)) { err = "caller shim nvngx.dll not found for this variant"; return false; }
        m_shimPath = shim;
        m_shim = LoadLibraryW(shim.c_str());
        if (!m_shim)
        {
            char b[128]; snprintf(b, sizeof(b), "LoadLibrary nvngx.dll (shim) failed, GetLastError=%lu", GetLastError());
            err = b; return false;
        }
    }
    return true;
}

bool Host::createDevice(std::string& err)
{
    // Both entry points from the System32 DLLs by name: linked into smv-live.exe (its offline
    // host) the plain imports resolve to Streamline's sl.interposer, which
    // that route never initialises (the WGC capture device follows the same rule).
    wchar_t sys[MAX_PATH]{};
    GetSystemDirectoryW(sys, MAX_PATH);
    HMODULE dxgiDll = LoadLibraryW((std::wstring(sys) + L"\\dxgi.dll").c_str());
    HMODULE d3d12Dll = LoadLibraryW((std::wstring(sys) + L"\\d3d12.dll").c_str());
    typedef HRESULT (WINAPI* PFN_Factory2)(UINT, REFIID, void**);
    auto pFactory = dxgiDll ? (PFN_Factory2)GetProcAddress(dxgiDll, "CreateDXGIFactory2") : nullptr;
    auto pCreate = d3d12Dll ? (PFN_D3D12_CREATE_DEVICE)GetProcAddress(d3d12Dll, "D3D12CreateDevice") : nullptr;
    if (!pFactory || !pCreate) { err = "System32 dxgi.dll / d3d12.dll entry points unavailable"; return false; }
    ComPtr<IDXGIFactory4> factory;
    if (FAILED(pFactory(0, IID_PPV_ARGS(&factory)))) { err = "CreateDXGIFactory2 failed"; return false; }

    ComPtr<IDXGIAdapter1> adapter;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i)
    {
        DXGI_ADAPTER_DESC1 d = {};
        adapter->GetDesc1(&d);
        if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) { adapter.Reset(); continue; }
        if (SUCCEEDED(pCreate(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&m_dev)))) break;
        adapter.Reset();
    }
    if (!m_dev) { err = "no D3D12 capable adapter"; return false; }

    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(m_dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&m_queue)))) { err = "CreateCommandQueue failed"; return false; }
    return true;
}

bool Host::createCommandObjects(std::string& err)
{
    if (FAILED(m_dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_alloc)))) { err = "CreateCommandAllocator failed"; return false; }
    if (FAILED(m_dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_alloc.Get(), nullptr, IID_PPV_ARGS(&m_list)))) { err = "CreateCommandList failed"; return false; }
    m_list->Close();
    if (FAILED(m_dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)))) { err = "CreateFence failed"; return false; }
    m_fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!m_fenceEvent) { err = "CreateEvent failed"; return false; }
    return true;
}

static D3D12_RESOURCE_BARRIER transition(ID3D12Resource* r, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to)
{
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = r;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = from;
    b.Transition.StateAfter = to;
    return b;
}

bool Host::createResources(bool staging, std::string& err)
{
    D3D12_HEAP_PROPERTIES def = {}; def.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_HEAP_PROPERTIES up  = {}; up.Type  = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_HEAP_PROPERTIES rb  = {}; rb.Type  = D3D12_HEAP_TYPE_READBACK;

    D3D12_RESOURCE_DESC td = {};
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width = m_w; td.Height = m_h;
    td.DepthOrArraySize = 1; td.MipLevels = 1;
    td.Format = kFmt;
    td.SampleDesc.Count = 1;
    td.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    td.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    if (FAILED(m_dev->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &td,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, nullptr, IID_PPV_ARGS(&m_color))))
    { err = "CreateCommittedResource(color) failed"; return false; }

    if (FAILED(m_dev->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &td,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&m_output))))
    { err = "CreateCommittedResource(output) failed"; return false; }

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp = {};
    UINT64 total = 0;
    m_dev->GetCopyableFootprints(&td, 0, 1, 0, &fp, nullptr, nullptr, &total);
    m_rowPitch = fp.Footprint.RowPitch;
    if (!staging) return true;   // the caller feeds the textures on the GPU (startupOn)

    D3D12_RESOURCE_DESC bd = {};
    bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bd.Width = total; bd.Height = 1;
    bd.DepthOrArraySize = 1; bd.MipLevels = 1;
    bd.Format = DXGI_FORMAT_UNKNOWN;
    bd.SampleDesc.Count = 1;
    bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    if (FAILED(m_dev->CreateCommittedResource(&up, D3D12_HEAP_FLAG_NONE, &bd,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&m_upload))))
    { err = "CreateCommittedResource(upload) failed"; return false; }
    if (FAILED(m_dev->CreateCommittedResource(&rb, D3D12_HEAP_FLAG_NONE, &bd,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&m_readback))))
    { err = "CreateCommittedResource(readback) failed"; return false; }
    return true;
}

bool Host::runCommandList(std::string& err)
{
    if (FAILED(m_list->Close())) { err = "command list Close failed"; return false; }
    ID3D12CommandList* lists[] = { m_list.Get() };
    m_queue->ExecuteCommandLists(1, lists);
    const uint64_t v = ++m_fenceValue;
    if (FAILED(m_queue->Signal(m_fence.Get(), v))) { err = "queue Signal failed"; return false; }
    if (m_fence->GetCompletedValue() < v)
    {
        m_fence->SetEventOnCompletion(v, m_fenceEvent);
        WaitForSingleObject(m_fenceEvent, INFINITE);
    }
    return true;
}

int Host::startup(uint32_t w, uint32_t h, const Settings& s, const Variant& v, std::string& err,
                  bool quietLog)
{
    m_w = w; m_h = h; m_set = s; m_var = v;
    m_external = false;
    m_quiet = quietLog;

    if (!resolveModules(v, err)) return 2;
    if (!createDevice(err)) return 2;
    if (!createCommandObjects(err)) return 2;
    if (!createResources(true, err)) return 2;
    return initNgx(err);
}

int Host::startupOn(ID3D12Device* device, ID3D12CommandQueue* queue, uint32_t w, uint32_t h,
                    const Settings& s, const Variant& v, bool quietLog, std::string& err)
{
    m_w = w; m_h = h; m_set = s; m_var = v;
    m_external = true;
    m_quiet = quietLog;
    if (!device || !queue) { err = "startupOn needs a device and a queue"; return 2; }

    if (!resolveModules(v, err)) return 2;
    m_dev = device;
    m_queue = queue;
    if (!createCommandObjects(err)) return 2;
    if (!createResources(false, err)) return 2;
    return initNgx(err);
}

int Host::initNgx(std::string& err)
{
    const Variant& v = m_var;

    // Driver core entry points.
    void* pInit    = (void*)GetProcAddress(m_core, v.initProjectId ? "NVSDK_NGX_D3D12_Init_ProjectID"
                                                                   : "NVSDK_NGX_D3D12_Init_Ext");
    void* pAlloc   = (void*)GetProcAddress(m_core, "NVSDK_NGX_D3D12_AllocateParameters");
    // Feature entry points: from the NR snippet itself (default, it exports the whole
    // NVSDK_NGX_D3D12_* API and the driver core's feature table does not know id 18)
    // or from the driver core (--via-core, the original phase 0 route).
    HMODULE feat = v.viaSnippet ? m_snippet : m_core;
    void* pCreate  = (void*)GetProcAddress(feat, "NVSDK_NGX_D3D12_CreateFeature");
    void* pEval    = (void*)GetProcAddress(feat, "NVSDK_NGX_D3D12_EvaluateFeature_C");
    if (!pEval) pEval = (void*)GetProcAddress(feat, "NVSDK_NGX_D3D12_EvaluateFeature");
    void* pRelease = (void*)GetProcAddress(feat, "NVSDK_NGX_D3D12_ReleaseFeature");
    void* pShut    = (void*)GetProcAddress(m_core, "NVSDK_NGX_D3D12_Shutdown1");
    if (!pInit || !pAlloc || !pCreate || !pEval || !pRelease)
    { err = std::string(v.viaSnippet ? "nvngx_dlssnr.dll" : "_nvngx.dll") + " is missing one of the D3D12 NGX exports"; return 2; }

    // Shim entry points, when this variant routes through nvngx.dll.
    PFN_ShimInit        sInit  = nullptr;
    PFN_ShimInitExt     sInitE = nullptr;
    PFN_ShimAllocParams sAlloc = nullptr;
    if (m_shim)
    {
        sInit  = (PFN_ShimInit)GetProcAddress(m_shim, "DLSSNR_CallInit");
        sInitE = (PFN_ShimInitExt)GetProcAddress(m_shim, "DLSSNR_CallInitExt");
        sAlloc = (PFN_ShimAllocParams)GetProcAddress(m_shim, "DLSSNR_CallAllocParams");
        if (!sInit || !sInitE || !sAlloc) { err = "caller shim is missing an export"; return 2; }
    }

    // The snippet's own folder is offered to NGX as an extra search path, the
    // same reason rtxvideo.py calls os.add_dll_directory.
    std::wstring snipDir = m_snippetPath.substr(0, m_snippetPath.find_last_of(L"\\/"));
    const wchar_t* paths[1] = { snipDir.c_str() };
    NVSDK_NGX_FeatureCommonInfo fci = {};
    fci.PathListInfo.Path = paths;
    fci.PathListInfo.Length = 1;
    // NGX verbose logging to stderr: the driver core and the snippet say WHY a
    // feature is refused (probe diagnostics; the callback must be thread safe).
    fci.LoggingInfo.LoggingCallback = [](const char* msg, NVSDK_NGX_Logging_Level, NVSDK_NGX_Feature src)
    {
        fprintf(stderr, "[ngx %d] %s", (int)src, msg);
        if (!msg || !*msg || msg[strlen(msg) - 1] != '\n') fputc('\n', stderr);
        fflush(stderr);
    };
    fci.LoggingInfo.MinimumLoggingLevel = m_quiet ? NVSDK_NGX_LOGGING_LEVEL_OFF
                                                  : NVSDK_NGX_LOGGING_LEVEL_VERBOSE;
    fci.LoggingInfo.DisableOtherLoggingSinks = false;

    std::wstring dataPath = moduleDir();

    if (v.initProjectId)
    {
        m_last = m_shim
            ? sInit(pInit, v.initArgOrder, kProjectId, NVSDK_NGX_ENGINE_TYPE_CUSTOM, "0.1",
                    dataPath.c_str(), m_dev.Get(), &fci, NVSDK_NGX_Version_API)
            : directInit(pInit, v.initArgOrder, kProjectId, NVSDK_NGX_ENGINE_TYPE_CUSTOM, "0.1",
                          dataPath.c_str(), m_dev.Get(), &fci, NVSDK_NGX_Version_API);
    }
    else
    {
        m_last = m_shim
            ? sInitE(pInit, 0x53f803ccull, dataPath.c_str(), m_dev.Get(), NVSDK_NGX_Version_API)
            : ((PFN_NGX_InitExt)pInit)(0x53f803ccull, dataPath.c_str(), m_dev.Get(), NVSDK_NGX_Version_API, nullptr);
    }
    if (m_last != NVSDK_NGX_Result_Success)
    { err = "NGX init failed: " + resultString(m_last); return 2; }
    m_ngxUp = true;

    if (v.viaSnippet)
    {
        // The snippet keeps its own device and data path: initialise it through its
        // own Init_Ext export before creating the feature on it (the core session
        // above still provides AllocateParameters and the caller module context).
        // App id = the cms id the core mapped for the project id (0x0876232C, log line
        // "MapProjectId: Found cms id 876232c"), version 0x15, and the FeatureCommonInfo
        // as the fifth argument, exactly the reference sequence; its result is logged
        // only, the reference does not gate creation on it either.
        void* pSnipInit = (void*)GetProcAddress(m_snippet, "NVSDK_NGX_D3D12_Init_Ext");
        if (!pSnipInit) { err = "nvngx_dlssnr.dll has no NVSDK_NGX_D3D12_Init_Ext"; return 2; }
        const unsigned long long kCmsAppId = 141959980ull;
        typedef NVSDK_NGX_Result (*PFN_SnipInitExt)(unsigned long long, const wchar_t*, ID3D12Device*,
                                                    unsigned int, const NVSDK_NGX_FeatureCommonInfo*);
        typedef NVSDK_NGX_Result (*PFN_ShimInitExtF)(void*, unsigned long long, const wchar_t*, ID3D12Device*,
                                                     unsigned int, const NVSDK_NGX_FeatureCommonInfo*);
        NVSDK_NGX_Result rs = m_shim
            ? ((PFN_ShimInitExtF)GetProcAddress(m_shim, "DLSSNR_CallInitExtF"))(pSnipInit, kCmsAppId, dataPath.c_str(), m_dev.Get(), 0x15, &fci)
            : ((PFN_SnipInitExt)pSnipInit)(kCmsAppId, dataPath.c_str(), m_dev.Get(), 0x15, &fci);
        fprintf(stderr, "dlssnr: snippet Init_Ext -> %s\n", resultString(rs).c_str());
        fflush(stderr);
    }

    m_last = m_shim ? sAlloc(pAlloc, &m_params) : ((PFN_NGX_AllocParams)pAlloc)(&m_params);
    if (m_last != NVSDK_NGX_Result_Success || !m_params)
    { err = "AllocateParameters failed: " + resultString(m_last); return 2; }

    // Minimum resource and parameter set for feature 18 (DLAA style, ratio 1.0).
    m_params->Set("DLSSNR.ScalingRatio", 1.0f);
    m_params->Set("DLSSNR.Width",  (unsigned int)m_w);
    m_params->Set("DLSSNR.Height", (unsigned int)m_h);
    m_params->Set("DLSSNR.Color",      m_color.Get());
    m_params->Set("DLSSNR.Output",     m_output.Get());
    m_params->Set("DLSSNR.Backbuffer", m_output.Get());
    m_params->Set("DLSSNR.ColorSubrectBaseX", (unsigned int)0);
    m_params->Set("DLSSNR.ColorSubrectBaseY", (unsigned int)0);
    m_params->Set("DLSSNR.ColorSubrectWidth", (unsigned int)m_w);
    m_params->Set("DLSSNR.ColorSubrectHeight", (unsigned int)m_h);
    m_params->Set("DLSSNR.OutputSubrectBaseX", (unsigned int)0);
    m_params->Set("DLSSNR.OutputSubrectBaseY", (unsigned int)0);
    m_params->Set("DLSSNR.OutputSubrectWidth", (unsigned int)m_w);
    m_params->Set("DLSSNR.OutputSubrectHeight", (unsigned int)m_h);
    m_params->Set("DLSSNR.Reset", 1);
    m_params->Set("DLSSNR.Enabled", 1);
    m_params->Set("DLSSNR.UICorrection", 0);
    m_params->Set("DLSSNR.DepthInverted", 1);
    m_params->Set("DLSSNR.UseAutoMask", (unsigned int)0);
    m_params->Set("DLSSNR.MVecScaleX", 1.0f);
    m_params->Set("DLSSNR.MVecScaleY", 1.0f);
    m_params->Set("DLSSNR.Style", m_set.style);
    m_params->Set("DLSSNR.Hint.Render.Preset", m_set.preset);
    m_params->Set("DLSSNR.Intensity", m_set.intensity);
    m_params->Set("DLSSNR.LocalStructureStrength", m_set.structure);
    m_params->Set("DLSSNR.LocalToneStrength", m_set.tone);
    m_params->Set("DLSSNR.SkinStructureStrength", -1.0f);
    // MVec, Depth and ControlMask are deliberately not set.

    if (FAILED(m_alloc->Reset()) || FAILED(m_list->Reset(m_alloc.Get(), nullptr)))
    { err = "command list Reset failed"; return 2; }

    if (m_shim)
    {
        PFN_ShimCreate sCreate = (PFN_ShimCreate)GetProcAddress(m_shim, "DLSSNR_CallCreate");
        if (!sCreate) { err = "caller shim is missing DLSSNR_CallCreate"; return 2; }
        m_last = sCreate(pCreate, m_list.Get(), kFeatureId, m_params, &m_feature);
    }
    else
    {
        m_last = ((PFN_NGX_CreateFeature)pCreate)(m_list.Get(), (NVSDK_NGX_Feature)kFeatureId, m_params, &m_feature);
    }
    if (m_last != NVSDK_NGX_Result_Success || !m_feature)
    {
        m_list->Close();
        err = "CreateFeature(18) refused: " + resultString(m_last);
        return 3;
    }
    if (!runCommandList(err)) return 2;
    return 0;
}

bool Host::evaluateOn(ID3D12GraphicsCommandList* list, bool reset, std::string& err)
{
    HMODULE feat = m_var.viaSnippet ? m_snippet : m_core;
    void* pEval = (void*)GetProcAddress(feat, "NVSDK_NGX_D3D12_EvaluateFeature_C");
    if (!pEval) pEval = (void*)GetProcAddress(feat, "NVSDK_NGX_D3D12_EvaluateFeature");
    PFN_ShimEvaluate sEval = m_shim ? (PFN_ShimEvaluate)GetProcAddress(m_shim, "DLSSNR_CallEvaluate") : nullptr;
    if (!pEval || (m_shim && !sEval)) { err = "EvaluateFeature entry point missing"; return false; }
    if (!m_feature || !m_params) { err = "feature not created"; return false; }

    m_params->Set("DLSSNR.Reset", reset ? 1 : 0);
    m_params->Set("DLSSNR.Intensity", m_set.intensity);
    m_params->Set("DLSSNR.LocalStructureStrength", m_set.structure);
    m_params->Set("DLSSNR.LocalToneStrength", m_set.tone);

    m_last = m_shim ? sEval(pEval, list, m_feature, m_params)
                    : ((PFN_NGX_EvaluateFeature)pEval)(list, m_feature, m_params, nullptr);
    if (m_last != NVSDK_NGX_Result_Success)
    {
        err = "EvaluateFeature failed: " + resultString(m_last);
        return false;
    }
    return true;
}

bool Host::renderFrame(const void* src, void* dst, bool reset, std::string& err)
{
    if (m_external) { err = "renderFrame is the staging route, not available on a caller device"; return false; }

    const size_t srcRow = (size_t)m_w * 8;

    // Upload the frame into the staging buffer, row by row (the copy footprint
    // row pitch is 256 byte aligned, the caller's frame is tightly packed).
    void* mapped = nullptr;
    D3D12_RANGE none = { 0, 0 };
    if (FAILED(m_upload->Map(0, &none, &mapped))) { err = "upload Map failed"; return false; }
    for (uint32_t y = 0; y < m_h; ++y)
        memcpy((uint8_t*)mapped + (size_t)y * m_rowPitch, (const uint8_t*)src + y * srcRow, srcRow);
    m_upload->Unmap(0, nullptr);

    if (FAILED(m_alloc->Reset()) || FAILED(m_list->Reset(m_alloc.Get(), nullptr)))
    { err = "command list Reset failed"; return false; }

    D3D12_TEXTURE_COPY_LOCATION dstLoc = {};
    dstLoc.pResource = m_color.Get();
    dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dstLoc.SubresourceIndex = 0;
    D3D12_TEXTURE_COPY_LOCATION srcLoc = {};
    srcLoc.pResource = m_upload.Get();
    srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    srcLoc.PlacedFootprint.Offset = 0;
    srcLoc.PlacedFootprint.Footprint.Format = kFmt;
    srcLoc.PlacedFootprint.Footprint.Width = m_w;
    srcLoc.PlacedFootprint.Footprint.Height = m_h;
    srcLoc.PlacedFootprint.Footprint.Depth = 1;
    srcLoc.PlacedFootprint.Footprint.RowPitch = (UINT)m_rowPitch;

    D3D12_RESOURCE_BARRIER b = transition(m_color.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                          D3D12_RESOURCE_STATE_COPY_DEST);
    m_list->ResourceBarrier(1, &b);
    m_list->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
    b = transition(m_color.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    m_list->ResourceBarrier(1, &b);

    if (!evaluateOn(m_list.Get(), reset, err))
    {
        m_list->Close();
        return false;
    }

    b = transition(m_output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    m_list->ResourceBarrier(1, &b);
    D3D12_TEXTURE_COPY_LOCATION outSrc = {};
    outSrc.pResource = m_output.Get();
    outSrc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    outSrc.SubresourceIndex = 0;
    D3D12_TEXTURE_COPY_LOCATION outDst = srcLoc;
    outDst.pResource = m_readback.Get();
    m_list->CopyTextureRegion(&outDst, 0, 0, 0, &outSrc, nullptr);
    b = transition(m_output.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    m_list->ResourceBarrier(1, &b);

    if (!runCommandList(err)) return false;

    // A copyable footprint pads every row to 256 bytes EXCEPT the last, so the buffer is
    // RowPitch * (h - 1) + w * 8 bytes: mapping RowPitch * h reaches past its end and Map fails
    // for every width that is not a multiple of 32 (the render then ran without DLSS 5; the
    // same finding and fix as DLSS5-NeuralScreen's readback).
    D3D12_RANGE all = { 0, (SIZE_T)((size_t)m_rowPitch * (m_h - 1) + srcRow) };
    void* rd = nullptr;
    if (FAILED(m_readback->Map(0, &all, &rd))) { err = "readback Map failed"; return false; }
    for (uint32_t y = 0; y < m_h; ++y)
        memcpy((uint8_t*)dst + y * srcRow, (const uint8_t*)rd + (size_t)y * m_rowPitch, srcRow);
    m_readback->Unmap(0, &none);
    return true;
}

// NGX teardown on a HALF-initialized session (snippet Init_Ext failed after the
// core session came up) faults inside the driver. The process is exiting anyway,
// so guard the NGX calls with SEH: on a fault, skip to releasing the D3D objects
// rather than crashing the exit path with 0xC0000005.
static void guardedNgxShutdown(Host* self);

void Host::shutdown()
{
    guardedNgxShutdown(this);
    m_readback.Reset(); m_upload.Reset(); m_output.Reset(); m_color.Reset();
    // on a caller device the queue and device references are just dropped
    m_list.Reset(); m_alloc.Reset(); m_queue.Reset(); m_fence.Reset(); m_dev.Reset();
    if (m_fenceEvent) { CloseHandle(m_fenceEvent); m_fenceEvent = nullptr; }
    // The snippet and the driver core stay loaded for the life of the process on
    // purpose: NGX does not support a clean unload and reload in place.
}

void Host::abandon()
{
    m_feature = nullptr;
    m_params = nullptr;
    m_ngxUp = false;
    m_readback.Reset(); m_upload.Reset(); m_output.Reset(); m_color.Reset();
    m_list.Reset(); m_alloc.Reset(); m_queue.Reset(); m_fence.Reset(); m_dev.Reset();
    if (m_fenceEvent) { CloseHandle(m_fenceEvent); m_fenceEvent = nullptr; }
}

void Host::ngxShutdown()
{
    if (m_core && m_feature)
    {
        void* pRelease = (void*)GetProcAddress(m_var.viaSnippet ? m_snippet : m_core, "NVSDK_NGX_D3D12_ReleaseFeature");
        PFN_ShimRelease sRel = m_shim ? (PFN_ShimRelease)GetProcAddress(m_shim, "DLSSNR_CallRelease") : nullptr;
        if (pRelease)
        {
            if (sRel) sRel(pRelease, m_feature);
            else ((PFN_NGX_ReleaseFeature)pRelease)(m_feature);
        }
        m_feature = nullptr;
    }
    if (m_core && m_params)
    {
        PFN_NGX_DestroyParams destroy = (PFN_NGX_DestroyParams)GetProcAddress(m_core, "NVSDK_NGX_D3D12_DestroyParameters");
        if (destroy) destroy(m_params);
        m_params = nullptr;
    }
    if (m_core && m_ngxUp)
    {
        void* pShut = (void*)GetProcAddress(m_core, "NVSDK_NGX_D3D12_Shutdown1");
        PFN_ShimShutdown sShut = m_shim ? (PFN_ShimShutdown)GetProcAddress(m_shim, "DLSSNR_CallShutdown") : nullptr;
        if (pShut)
        {
            if (sShut) sShut(pShut, m_dev.Get());
            else ((PFN_NGX_Shutdown1)pShut)(m_dev.Get());
        }
        m_ngxUp = false;
    }
}

static void guardedNgxShutdown(Host* self)
{
    __try { self->ngxShutdown(); }
    __except (EXCEPTION_EXECUTE_HANDLER)
    { fprintf(stderr, "dlssnr: NGX teardown faulted, skipped\n"); fflush(stderr); }
}

} // namespace nr
