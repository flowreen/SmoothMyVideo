// nr_host.cpp - see nr_host.h. Reimplemented from the recorded facts, nothing
// copied from any third party host.
#include "nr_host.h"
#include "shim_abi.h"

#include <dxgi1_6.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cstdlib>
#include <cerrno>

#pragma comment(lib, "advapi32.lib")

using Microsoft::WRL::ComPtr;

namespace nr
{

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
typedef DWORD(WINAPI* PFN_GMFNW)(HMODULE, LPWSTR, DWORD);
typedef DWORD(WINAPI* PFN_GMFNA)(HMODULE, LPSTR, DWORD);
static PFN_GMFNW s_realGMFNW = nullptr;
static PFN_GMFNA s_realGMFNA = nullptr;

// The environment through getenv_s: a copy into a local buffer, never a pointer into the CRT's block.
static bool envSet(const char* name)
{
    size_t n = 0;
    getenv_s(&n, nullptr, 0, name);
    return n > 0;
}
static std::string envValue(const char* name)
{
    char buf[MAX_PATH];
    size_t n = 0;
    return (getenv_s(&n, buf, sizeof(buf), name) == 0 && n > 0) ? std::string(buf) : std::string();
}

static std::string spoofName()
{
    const std::string s = envValue("SMV_NR_SPOOF");
    return s.empty() ? std::string("nvngx.dll") : s;
}
static bool hookLog()
{
    return envSet("SMV_NR_HOOKLOG");
}

static void replaceBasenameW(wchar_t* buf, DWORD size, const char* base)
{
    wchar_t* slash = wcsrchr(buf, L'\\');
    size_t keep = slash ? (size_t)(slash + 1 - buf) : 0;
    wchar_t wbase[MAX_PATH];
    size_t conv = 0;
    const errno_t e = mbstowcs_s(&conv, wbase, MAX_PATH, base, _TRUNCATE);
    if ((e != 0 && e != STRUNCATE) || keep + wcslen(wbase) + 1 > size)
        return;
    wcscpy_s(buf + keep, size - keep, wbase);
}

static DWORD WINAPI hookGMFNW(HMODULE h, LPWSTR buf, DWORD size)
{
    if (s_realGMFNW)
        s_realGMFNW(h, buf, size);
    if (hookLog())
    {
        fprintf(stderr, "dlssnr: [hook] GetModuleFileNameW(%p) -> \"%ls\"\n", (void*)h, buf);
        fflush(stderr);
    }
    replaceBasenameW(buf, size, spoofName().c_str());
    return (DWORD)wcslen(buf);
}

static DWORD WINAPI hookGMFNA(HMODULE h, LPSTR buf, DWORD size)
{
    if (s_realGMFNA)
        s_realGMFNA(h, buf, size);
    if (hookLog())
    {
        fprintf(stderr, "dlssnr: [hook] GetModuleFileNameA(%p) -> \"%s\"\n", (void*)h, buf);
        fflush(stderr);
    }
    char* slash = strrchr(buf, '\\');
    size_t keep = slash ? (size_t)(slash + 1 - buf) : 0;
    const std::string spoof = spoofName();
    if (keep + spoof.size() + 1 <= size)
        strcpy_s(buf + keep, size - keep, spoof.c_str());
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
    if (!dir.VirtualAddress)
        return 0;
    int changed = 0;
    for (auto desc = (IMAGE_IMPORT_DESCRIPTOR*)(base + dir.VirtualAddress); desc->Name; ++desc)
    {
        const char* dll = (const char*)(base + desc->Name);
        if (_stricmp(dll, "kernel32.dll") != 0 && _strnicmp(dll, "api-ms-win-core", 15) != 0)
            continue;
        if (!desc->OriginalFirstThunk)
            continue;
        auto thunk = (IMAGE_THUNK_DATA*)(base + desc->OriginalFirstThunk);
        auto iat = (IMAGE_THUNK_DATA*)(base + desc->FirstThunk);
        for (; thunk->u1.AddressOfData; ++thunk, ++iat)
        {
            if (thunk->u1.Ordinal & IMAGE_ORDINAL_FLAG)
                continue;
            auto ibn = (IMAGE_IMPORT_BY_NAME*)(base + thunk->u1.AddressOfData);
            if (strcmp(ibn->Name, fn) != 0)
                continue;
            DWORD old;
            if (!VirtualProtect(&iat->u1.Function, sizeof(void*), PAGE_READWRITE, &old))
                continue;
            if (realOut && !*realOut)
                *realOut = (void*)iat->u1.Function;
            iat->u1.Function = (ULONG_PTR)hook;
            VirtualProtect(&iat->u1.Function, sizeof(void*), old, &old);
            ++changed;
        }
    }
    return changed;
}

static void installCallerShim(HMODULE snippet)
{
    if (envSet("SMV_NR_NOHOOK"))
    {
        fprintf(stderr, "dlssnr: caller shim disabled (SMV_NR_NOHOOK)\n");
        return;
    }
    int w = patchImport(snippet, "GetModuleFileNameW", (void*)hookGMFNW, (void**)&s_realGMFNW);
    int a = patchImport(snippet, "GetModuleFileNameA", (void*)hookGMFNA, (void**)&s_realGMFNA);
    fprintf(stderr, "dlssnr: caller shim: GetModuleFileNameW x%d GetModuleFileNameA x%d, spoof=%s\n", w, a,
            spoofName().c_str());
    fflush(stderr);
}

const char* const kProjectId = "53f803cc-a12f-4d69-90d5-19b7599cad19";

static const DXGI_FORMAT kFmt = DXGI_FORMAT_R16G16B16A16_FLOAT;

// Argument order of the driver core's Init_ProjectID export, which the public headers do not
// pin down: 1 = (.., device, version, featureInfo), the order that survives on 616.56 (order 0,
// (.., device, featureInfo, version), faults inside the core).
static const int kInitArgOrder = 1;

std::string resultString(NVSDK_NGX_Result r)
{
    char buf[64];
    const char* name = "Unknown";
    switch (r)
    {
    case NVSDK_NGX_Result_Success:
        name = "Success";
        break;
    case NVSDK_NGX_Result_Fail:
        name = "Fail";
        break;
    case NVSDK_NGX_Result_FAIL_FeatureNotSupported:
        name = "FeatureNotSupported";
        break;
    case NVSDK_NGX_Result_FAIL_PlatformError:
        name = "PlatformError";
        break;
    case NVSDK_NGX_Result_FAIL_FeatureAlreadyExists:
        name = "FeatureAlreadyExists";
        break;
    case NVSDK_NGX_Result_FAIL_FeatureNotFound:
        name = "FeatureNotFound";
        break;
    case NVSDK_NGX_Result_FAIL_InvalidParameter:
        name = "InvalidParameter";
        break;
    case NVSDK_NGX_Result_FAIL_ScratchBufferTooSmall:
        name = "ScratchBufferTooSmall";
        break;
    case NVSDK_NGX_Result_FAIL_NotInitialized:
        name = "NotInitialized";
        break;
    case NVSDK_NGX_Result_FAIL_UnsupportedInputFormat:
        name = "UnsupportedInputFormat";
        break;
    case NVSDK_NGX_Result_FAIL_RWFlagMissing:
        name = "RWFlagMissing";
        break;
    case NVSDK_NGX_Result_FAIL_MissingInput:
        name = "MissingInput";
        break;
    case NVSDK_NGX_Result_FAIL_UnableToInitializeFeature:
        name = "UnableToInitializeFeature";
        break;
    case NVSDK_NGX_Result_FAIL_OutOfDate:
        name = "OutOfDate";
        break;
    case NVSDK_NGX_Result_FAIL_OutOfGPUMemory:
        name = "OutOfGPUMemory";
        break;
    case NVSDK_NGX_Result_FAIL_UnsupportedFormat:
        name = "UnsupportedFormat";
        break;
    case NVSDK_NGX_Result_FAIL_UnableToWriteToAppDataPath:
        name = "UnableToWriteToAppDataPath";
        break;
    case NVSDK_NGX_Result_FAIL_UnsupportedParameter:
        name = "UnsupportedParameter";
        break;
    case NVSDK_NGX_Result_FAIL_Denied:
        name = "Denied";
        break;
    case NVSDK_NGX_Result_FAIL_NotImplemented:
        name = "NotImplemented";
        break;
    default:
        break;
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

static std::wstring s_moduleDir; // see setModuleDir in nr_host.h

void setModuleDir(const wchar_t* dir)
{
    s_moduleDir = (dir && *dir) ? dir : L"";
}

// The folder the snippet, the shim and the NGX data path resolve from.
static std::wstring moduleDir()
{
    return s_moduleDir.empty() ? exeDir() : s_moduleDir;
}

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
    if (h == INVALID_HANDLE_VALUE)
        return false;
    bool found = false;
    do
    {
        if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
            continue;
        std::wstring cand = root + fd.cFileName + L"\\" + fileName;
        if (fileExists(cand))
        {
            out = cand;
            found = true;
            break;
        }
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
    if (RegGetValueW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\NVIDIA Corporation\\Global\\NGXCore", L"FullPath", RRF_RT_REG_SZ,
                     nullptr, buf, &cb) != ERROR_SUCCESS ||
        !buf[0])
        return false;
    std::wstring cand = std::wstring(buf) + L"\\" + fileName;
    if (!fileExists(cand))
        return false;
    out = cand;
    return true;
}

Host::~Host()
{
    shutdown();
}

bool Host::resolveModules(std::string& err)
{
    const std::wstring dir = moduleDir();

    // 1. the NR snippet: the user drop beside the exe wins, a DriverStore copy
    //    is the fallback for the day an official driver ships one.
    std::wstring snippet = dir + L"\\nvngx_dlssnr.dll";
    if (!fileExists(snippet) && !ngxCoreFromRegistry(L"nvngx_dlssnr.dll", snippet) &&
        !findInDriverStore(L"nvngx_dlssnr.dll", snippet))
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
        if (!fileExists(core))
        {
            err = "_nvngx.dll (NVIDIA driver core) not found";
            return false;
        }
    }
    m_corePath = core;

    // The snippet is loaded first so the core sees the NR layer.
    m_snippet = LoadLibraryW(m_snippetPath.c_str());
    if (!m_snippet)
    {
        char b[128];
        snprintf(b, sizeof(b), "LoadLibrary nvngx_dlssnr.dll failed, GetLastError=%lu", GetLastError());
        err = b;
        return false;
    }
    // Pass the snippet's caller-name check before any NGX call reaches it.
    installCallerShim(m_snippet);
    m_core = LoadLibraryW(m_corePath.c_str());
    if (!m_core)
    {
        char b[128];
        snprintf(b, sizeof(b), "LoadLibrary _nvngx.dll failed, GetLastError=%lu", GetLastError());
        err = b;
        return false;
    }

    // 3. the caller shim beside the exe (a DLL named nvngx.dll): every NGX entry point is called
    // through it, so the return address the runtime validates lands inside it.
    const std::wstring shim = dir + L"\\nvngx.dll";
    if (!fileExists(shim))
    {
        err = "caller shim nvngx.dll not found beside the host";
        return false;
    }
    m_shim = LoadLibraryW(shim.c_str());
    if (!m_shim)
    {
        char b[128];
        snprintf(b, sizeof(b), "LoadLibrary nvngx.dll (shim) failed, GetLastError=%lu", GetLastError());
        err = b;
        return false;
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
    typedef HRESULT(WINAPI * PFN_Factory2)(UINT, REFIID, void**);
    auto pFactory = dxgiDll ? (PFN_Factory2)GetProcAddress(dxgiDll, "CreateDXGIFactory2") : nullptr;
    auto pCreate = d3d12Dll ? (PFN_D3D12_CREATE_DEVICE)GetProcAddress(d3d12Dll, "D3D12CreateDevice") : nullptr;
    if (!pFactory || !pCreate)
    {
        err = "System32 dxgi.dll / d3d12.dll entry points unavailable";
        return false;
    }
    ComPtr<IDXGIFactory4> factory;
    if (FAILED(pFactory(0, IID_PPV_ARGS(&factory))))
    {
        err = "CreateDXGIFactory2 failed";
        return false;
    }

    ComPtr<IDXGIAdapter1> adapter;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i)
    {
        DXGI_ADAPTER_DESC1 d = {};
        adapter->GetDesc1(&d);
        if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
        {
            adapter.Reset();
            continue;
        }
        if (m_wantAdapter &&
            (d.AdapterLuid.LowPart != m_wantLuid.LowPart || d.AdapterLuid.HighPart != m_wantLuid.HighPart))
        {
            adapter.Reset();
            continue;
        }
        if (SUCCEEDED(pCreate(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&m_dev))))
            break;
        adapter.Reset();
    }
    if (!m_dev)
    {
        err = m_wantAdapter ? "no D3D12 adapter matches the CUDA device" : "no D3D12 capable adapter";
        return false;
    }
    m_adapter = adapter;

    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(m_dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&m_queue))))
    {
        err = "CreateCommandQueue failed";
        return false;
    }
    return true;
}

bool Host::createCommandObjects(std::string& err)
{
    if (FAILED(m_dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_alloc))))
    {
        err = "CreateCommandAllocator failed";
        return false;
    }
    if (FAILED(
            m_dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_alloc.Get(), nullptr, IID_PPV_ARGS(&m_list))))
    {
        err = "CreateCommandList failed";
        return false;
    }
    m_list->Close();
    if (FAILED(m_dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence))))
    {
        err = "CreateFence failed";
        return false;
    }
    m_fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!m_fenceEvent)
    {
        err = "CreateEvent failed";
        return false;
    }
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

bool Host::createResources(std::string& err)
{
    D3D12_HEAP_PROPERTIES def = {};
    def.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_HEAP_PROPERTIES up = {};
    up.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_HEAP_PROPERTIES rb = {};
    rb.Type = D3D12_HEAP_TYPE_READBACK;

    D3D12_RESOURCE_DESC td = {};
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width = m_w;
    td.Height = m_h;
    td.DepthOrArraySize = 1;
    td.MipLevels = 1;
    td.Format = kFmt;
    td.SampleDesc.Count = 1;
    td.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    td.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    if (FAILED(m_dev->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &td,
                                              D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, nullptr,
                                              IID_PPV_ARGS(&m_color))))
    {
        err = "CreateCommittedResource(color) failed";
        return false;
    }

    if (FAILED(m_dev->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &td, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                              nullptr, IID_PPV_ARGS(&m_output))))
    {
        err = "CreateCommittedResource(output) failed";
        return false;
    }
    // the chain's intermediate frames: pass k writes m_passOut[k], pass k + 1 reads it
    const int wanted = m_set.passes < 1 ? 1 : (m_set.passes > kMaxPasses ? kMaxPasses : m_set.passes);
    for (int k = 0; k + 1 < wanted; ++k)
        if (FAILED(m_dev->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &td,
                                                  D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                  IID_PPV_ARGS(&m_passOut[k]))))
        {
            err = "CreateCommittedResource(pass output) failed";
            return false;
        }

    if (m_set.motion)
    {
        // DLSSNR.MVec, bound at create; a committed DEFAULT-heap texture starts zeroed (no motion)
        D3D12_RESOURCE_DESC md = td;
        md.Format = DXGI_FORMAT_R16G16_FLOAT;
        md.Flags = D3D12_RESOURCE_FLAG_NONE;
        if (FAILED(m_dev->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &md,
                                                  D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, nullptr,
                                                  IID_PPV_ARGS(&m_mv))))
        {
            err = "CreateCommittedResource(motion) failed";
            return false;
        }
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT mfp = {};
        m_dev->GetCopyableFootprints(&md, 0, 1, 0, &mfp, nullptr, nullptr, nullptr);
        m_mvPitch = mfp.Footprint.RowPitch;
    }

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp = {};
    UINT64 total = 0;
    m_dev->GetCopyableFootprints(&td, 0, 1, 0, &fp, nullptr, nullptr, &total);
    m_rowPitch = fp.Footprint.RowPitch;

    D3D12_RESOURCE_DESC bd = {};
    bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bd.Width = total;
    bd.Height = 1;
    bd.DepthOrArraySize = 1;
    bd.MipLevels = 1;
    bd.Format = DXGI_FORMAT_UNKNOWN;
    bd.SampleDesc.Count = 1;
    bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    if (FAILED(m_dev->CreateCommittedResource(&up, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_GENERIC_READ,
                                              nullptr, IID_PPV_ARGS(&m_upload))))
    {
        err = "CreateCommittedResource(upload) failed";
        return false;
    }
    if (FAILED(m_dev->CreateCommittedResource(&rb, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                              IID_PPV_ARGS(&m_readback))))
    {
        err = "CreateCommittedResource(readback) failed";
        return false;
    }
    return true;
}

bool Host::runCommandList(std::string& err)
{
    if (FAILED(m_list->Close()))
    {
        err = "command list Close failed";
        return false;
    }
    ID3D12CommandList* lists[] = {m_list.Get()};
    m_queue->ExecuteCommandLists(1, lists);
    const uint64_t v = ++m_fenceValue;
    if (FAILED(m_queue->Signal(m_fence.Get(), v)))
    {
        err = "queue Signal failed";
        return false;
    }
    if (m_fence->GetCompletedValue() < v)
    {
        m_fence->SetEventOnCompletion(v, m_fenceEvent);
        WaitForSingleObject(m_fenceEvent, INFINITE);
    }
    return true;
}

int Host::startup(uint32_t w, uint32_t h, const Settings& s, std::string& err, bool quietLog, const LUID* adapter)
{
    m_w = w;
    m_h = h;
    m_set = s;
    m_quiet = quietLog;
    m_wantAdapter = adapter != nullptr;
    if (adapter)
        m_wantLuid = *adapter;

    if (!resolveModules(err))
        return 2;
    if (!createDevice(err))
        return 2;
    if (!createCommandObjects(err))
        return 2;
    if (!createResources(err))
        return 2;
    return initNgx(err);
}

// The NVIDIA driver version as the user sees it (617.14): the D3D user-mode driver's version (32.0.16.1714) carries
// it in its last five digits.
static std::string driverVersionText(IDXGIAdapter* a)
{
    LARGE_INTEGER umd{};
    if (!a || FAILED(a->CheckInterfaceSupport(__uuidof(IDXGIDevice), &umd)))
        return "version unknown";
    const unsigned v = (HIWORD(umd.LowPart) % 10) * 10000u + LOWORD(umd.LowPart);
    char b[32];
    snprintf(b, sizeof(b), "%u.%02u", v / 100, v % 100);
    return b;
}

// The create under a fault guard: on a driver too old for DLSS 5, NVIDIA's runtime has faulted inside the create
// instead of refusing it (NeuralScreen #145, 576.x drivers); a fault here becomes a refusal line, never the end of the
// process. testFault = the trigger test's fault (SMV_NR_TEST=fault). No C++ object lives in this frame (__try).
static NVSDK_NGX_Result guardedCreate(PFN_ShimCreate sCreate, void* pCreate, ID3D12GraphicsCommandList* list, int id,
                                      NVSDK_NGX_Parameter* p, NVSDK_NGX_Handle** f, bool testFault, bool& faulted)
{
    faulted = false;
    __try
    {
        if (testFault)
            RaiseException(EXCEPTION_ACCESS_VIOLATION, 0, 0, nullptr);
        return sCreate(pCreate, list, id, p, f);
    }
    // any fault of the create is a refusal on purpose: the session runs without DLSS 5
#pragma warning(suppress : 6320)
    __except (EXCEPTION_EXECUTE_HANDLER)
    {
        faulted = true;
        return NVSDK_NGX_Result_Fail;
    }
}

int Host::initNgx(std::string& err)
{
    // Driver core entry points.
    void* pInit = (void*)GetProcAddress(m_core, "NVSDK_NGX_D3D12_Init_ProjectID");
    void* pAlloc = (void*)GetProcAddress(m_core, "NVSDK_NGX_D3D12_AllocateParameters");
    // Feature entry points: from the NR snippet itself (it exports the whole NVSDK_NGX_D3D12_* API
    // and the driver core's feature table refuses id 18 before touching any snippet).
    void* pCreate = (void*)GetProcAddress(m_snippet, "NVSDK_NGX_D3D12_CreateFeature");
    void* pEval = (void*)GetProcAddress(m_snippet, "NVSDK_NGX_D3D12_EvaluateFeature_C");
    if (!pEval)
        pEval = (void*)GetProcAddress(m_snippet, "NVSDK_NGX_D3D12_EvaluateFeature");
    void* pRelease = (void*)GetProcAddress(m_snippet, "NVSDK_NGX_D3D12_ReleaseFeature");
    if (!pInit || !pAlloc || !pCreate || !pEval || !pRelease)
    {
        err = "nvngx_dlssnr.dll or _nvngx.dll is missing one of the D3D12 NGX exports";
        return 2;
    }

    // Shim entry points (nvngx.dll).
    typedef NVSDK_NGX_Result (*PFN_ShimInitExtF)(void*, unsigned long long, const wchar_t*, ID3D12Device*, unsigned int,
                                                 const NVSDK_NGX_FeatureCommonInfo*);
    auto sInit = (PFN_ShimInit)GetProcAddress(m_shim, "DLSSNR_CallInit");
    auto sAlloc = (PFN_ShimAllocParams)GetProcAddress(m_shim, "DLSSNR_CallAllocParams");
    auto sInitF = (PFN_ShimInitExtF)GetProcAddress(m_shim, "DLSSNR_CallInitExtF");
    auto sCreate = (PFN_ShimCreate)GetProcAddress(m_shim, "DLSSNR_CallCreate");
    if (!sInit || !sAlloc || !sInitF || !sCreate)
    {
        err = "caller shim is missing an export";
        return 2;
    }

    // The snippet's own folder is offered to NGX as an extra search path, the
    // same reason rtxvideo.py calls os.add_dll_directory.
    std::wstring snipDir = m_snippetPath.substr(0, m_snippetPath.find_last_of(L"\\/"));
    const wchar_t* paths[1] = {snipDir.c_str()};
    NVSDK_NGX_FeatureCommonInfo fci = {};
    fci.PathListInfo.Path = paths;
    fci.PathListInfo.Length = 1;
    // NGX verbose logging to stderr: the driver core and the snippet say WHY a
    // feature is refused (probe diagnostics; the callback must be thread safe).
    fci.LoggingInfo.LoggingCallback = [](const char* msg, NVSDK_NGX_Logging_Level, NVSDK_NGX_Feature src) {
        fprintf(stderr, "[ngx %d] %s", (int)src, msg);
        if (!msg || !*msg || msg[strlen(msg) - 1] != '\n')
            fputc('\n', stderr);
        fflush(stderr);
    };
    fci.LoggingInfo.MinimumLoggingLevel = m_quiet ? NVSDK_NGX_LOGGING_LEVEL_OFF : NVSDK_NGX_LOGGING_LEVEL_VERBOSE;
    fci.LoggingInfo.DisableOtherLoggingSinks = false;

    std::wstring dataPath = moduleDir();
    const unsigned long long kCmsAppId = 141959980ull;
    // SMV_NR_TEST=outofdate / =fault: the trigger tests of the two refusals below (never a product setting)
    wchar_t testv[16]{};
    GetEnvironmentVariableW(L"SMV_NR_TEST", testv, 16);
    const bool testOutOfDate = wcscmp(testv, L"outofdate") == 0, testFault = wcscmp(testv, L"fault") == 0;
    const std::string driver = driverVersionText(m_adapter.Get());

    // The feature's requirements first (NGX needs no Init for it): a driver too old for DLSS 5 answers OutOfDate or
    // sets the driver bit here, and on such drivers the create has faulted instead of refusing. Only those two
    // answers refuse; any other goes to the log and the create decides.
    typedef NVSDK_NGX_Result (*PFN_Requirements)(IDXGIAdapter*, const NVSDK_NGX_FeatureDiscoveryInfo*,
                                                 NVSDK_NGX_FeatureRequirement*);
    if (auto pReq = (PFN_Requirements)GetProcAddress(m_snippet, "NVSDK_NGX_D3D12_GetFeatureRequirements"))
    {
        NVSDK_NGX_FeatureDiscoveryInfo di = {};
        di.SDKVersion = NVSDK_NGX_Version_API;
        di.FeatureID = (NVSDK_NGX_Feature)kFeatureId;
        di.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Application_Id;
        di.Identifier.v.ApplicationId = kCmsAppId;
        di.ApplicationDataPath = dataPath.c_str();
        di.FeatureInfo = &fci;
        NVSDK_NGX_FeatureRequirement req = {};
        const NVSDK_NGX_Result rr = testOutOfDate ? NVSDK_NGX_Result_FAIL_OutOfDate : pReq(m_adapter.Get(), &di, &req);
        fprintf(stderr, "dlssnr: feature requirements -> %s, unsupported bits 0x%x, driver %s\n",
                resultString(rr).c_str(), (unsigned)req.FeatureSupported, driver.c_str());
        fflush(stderr);
        if (rr == NVSDK_NGX_Result_FAIL_OutOfDate ||
            (rr == NVSDK_NGX_Result_Success &&
             (req.FeatureSupported & NVSDK_NGX_FeatureSupportResult_DriverVersionUnsupported)))
        {
            err =
                "the NVIDIA driver " + driver + " is too old for DLSS 5 (" + resultString(rr) + "): update the driver";
            return 3;
        }
    }

    m_last = sInit(pInit, kInitArgOrder, kProjectId, NVSDK_NGX_ENGINE_TYPE_CUSTOM, "0.1", dataPath.c_str(), m_dev.Get(),
                   &fci, NVSDK_NGX_Version_API);
    if (m_last != NVSDK_NGX_Result_Success)
    {
        err = "NGX init failed: " + resultString(m_last);
        return 2;
    }
    m_ngxUp = true;

    // The snippet keeps its own device and data path: initialise it through its
    // own Init_Ext export before creating the feature on it (the core session
    // above still provides AllocateParameters and the caller module context).
    // App id = the cms id the core mapped for the project id (0x0876232C, log line
    // "MapProjectId: Found cms id 876232c"), version 0x15, and the FeatureCommonInfo
    // as the fifth argument, exactly the reference sequence; its result is logged
    // only, the reference does not gate creation on it either.
    void* pSnipInit = (void*)GetProcAddress(m_snippet, "NVSDK_NGX_D3D12_Init_Ext");
    if (!pSnipInit)
    {
        err = "nvngx_dlssnr.dll has no NVSDK_NGX_D3D12_Init_Ext";
        return 2;
    }
    const NVSDK_NGX_Result rs = sInitF(pSnipInit, kCmsAppId, dataPath.c_str(), m_dev.Get(), 0x15, &fci);
    fprintf(stderr, "dlssnr: snippet Init_Ext -> %s\n", resultString(rs).c_str());
    fflush(stderr);

    m_last = sAlloc(pAlloc, &m_params);
    if (m_last != NVSDK_NGX_Result_Success || !m_params)
    {
        err = "AllocateParameters failed: " + resultString(m_last);
        return 2;
    }

    const int wanted = m_set.passes < 1 ? 1 : (m_set.passes > kMaxPasses ? kMaxPasses : m_set.passes);
    m_pparams[0] = m_params;
    setPassParams(0, wanted - 1);

    if (FAILED(m_alloc->Reset()) || FAILED(m_list->Reset(m_alloc.Get(), nullptr)))
    {
        err = "command list Reset failed";
        return 2;
    }

    bool faulted = false;
    auto create = [&](NVSDK_NGX_Parameter* p, NVSDK_NGX_Handle** f) {
        return guardedCreate(sCreate, pCreate, m_list.Get(), kFeatureId, p, f, testFault, faulted);
    };
    m_last = create(m_params, &m_feature);
    if (faulted)
    {
        m_list->Close();
        err = "DLSS 5's create faulted inside NVIDIA's runtime (driver " + driver + "): update the driver";
        return 3;
    }
    if (m_last != NVSDK_NGX_Result_Success || !m_feature)
    {
        m_list->Close();
        err = "CreateFeature(18) refused: " + resultString(m_last);
        return 3;
    }
    m_pfeature[0] = m_feature;
    // passes 2..wanted: their own parameter block and feature; one that cannot be made ends the
    // chain at the passes built so far (the rest of the render runs them, passNote says why)
    m_passes = 1;
    m_passNote.clear();
    for (int k = 1; k < wanted; ++k)
    {
        NVSDK_NGX_Parameter* p = nullptr;
        NVSDK_NGX_Result r = sAlloc(pAlloc, &p);
        if (r != NVSDK_NGX_Result_Success || !p)
        {
            m_passNote = "pass " + std::to_string(k + 1) + " AllocateParameters " + resultString(r);
            break;
        }
        m_pparams[k] = p;
        setPassParams(k, wanted - 1);
        NVSDK_NGX_Handle* f = nullptr;
        r = create(p, &f);
        if (faulted || r != NVSDK_NGX_Result_Success || !f)
        {
            m_passNote = "pass " + std::to_string(k + 1) + " CreateFeature " + (faulted ? "faulted" : resultString(r));
            break;
        }
        m_pfeature[k] = f;
        m_passes = k + 1;
    }
    // a chain cut short: the frames between the passes that were not built go (the last built pass
    // keeps the one it was created with)
    for (int k = m_passes; k + 1 < kMaxPasses; ++k)
        m_passOut[k].Reset();
    if (!runCommandList(err))
        return 2;
    return 0;
}

// Minimum resource and parameter set for feature 18 (DLAA style, ratio 1.0), pass k of the chain
// 0..last: pass 0 reads Color, pass k reads pass k - 1's output, the last one writes Output
// (evaluateOn re-binds both per frame, so a chain cut short still ends in Output).
void Host::setPassParams(int k, int last)
{
    NVSDK_NGX_Parameter* p = m_pparams[k];
    ID3D12Resource* in = k == 0 ? m_color.Get() : m_passOut[k - 1].Get();
    ID3D12Resource* out = k == last ? m_output.Get() : m_passOut[k].Get();
    p->Set("DLSSNR.ScalingRatio", 1.0f);
    p->Set("DLSSNR.Width", (unsigned int)m_w);
    p->Set("DLSSNR.Height", (unsigned int)m_h);
    p->Set("DLSSNR.Color", in);
    p->Set("DLSSNR.Output", out);
    p->Set("DLSSNR.Backbuffer", out);
    p->Set("DLSSNR.ColorSubrectBaseX", (unsigned int)0);
    p->Set("DLSSNR.ColorSubrectBaseY", (unsigned int)0);
    p->Set("DLSSNR.ColorSubrectWidth", (unsigned int)m_w);
    p->Set("DLSSNR.ColorSubrectHeight", (unsigned int)m_h);
    p->Set("DLSSNR.OutputSubrectBaseX", (unsigned int)0);
    p->Set("DLSSNR.OutputSubrectBaseY", (unsigned int)0);
    p->Set("DLSSNR.OutputSubrectWidth", (unsigned int)m_w);
    p->Set("DLSSNR.OutputSubrectHeight", (unsigned int)m_h);
    p->Set("DLSSNR.Reset", 1);
    p->Set("DLSSNR.Enabled", 1);
    p->Set("DLSSNR.UICorrection", 0);
    p->Set("DLSSNR.DepthInverted", 1);
    const std::string am = envValue("SMV_NR_AUTOMASK"); // measurement lever: 0 = the mask off
    p->Set("DLSSNR.UseAutoMask", (unsigned int)((!am.empty() && am[0] == '0') ? 0 : m_set.automask));
    p->Set("DLSSNR.MVecScaleX", 1.0f);
    p->Set("DLSSNR.MVecScaleY", 1.0f);
    if (m_mv)
    {
        // pixels, current -> previous, the full frame (MVecScale 1 at DLAA); every pass follows the
        // same picture motion
        p->Set("DLSSNR.MVec", m_mv.Get());
        p->Set("DLSSNR.MVecSubrectBaseX", (unsigned int)0);
        p->Set("DLSSNR.MVecSubrectBaseY", (unsigned int)0);
        p->Set("DLSSNR.MVecSubrectWidth", (unsigned int)m_w);
        p->Set("DLSSNR.MVecSubrectHeight", (unsigned int)m_h);
    }
    p->Set("DLSSNR.Style", m_set.style);
    p->Set("DLSSNR.Hint.Render.Preset", m_set.preset);
    p->Set("DLSSNR.Intensity", m_set.intensity);
    p->Set("DLSSNR.LocalStructureStrength", m_set.structure);
    p->Set("DLSSNR.LocalToneStrength", m_set.tone);
    p->Set("DLSSNR.SkinStructureStrength", -1.0f);
    // Depth and ControlMask are deliberately not set (video has no depth); MVec only with Settings::motion.
}

bool Host::evaluateOn(ID3D12GraphicsCommandList* list, bool reset, std::string& err)
{
    void* pEval = (void*)GetProcAddress(m_snippet, "NVSDK_NGX_D3D12_EvaluateFeature_C");
    if (!pEval)
        pEval = (void*)GetProcAddress(m_snippet, "NVSDK_NGX_D3D12_EvaluateFeature");
    auto sEval = m_shim ? (PFN_ShimEvaluate)GetProcAddress(m_shim, "DLSSNR_CallEvaluate") : nullptr;
    if (!pEval || !sEval)
    {
        err = "EvaluateFeature entry point missing";
        return false;
    }
    if (!m_feature || !m_params)
    {
        err = "feature not created";
        return false;
    }

    // the chain: pass k reads pass k - 1's output (UAV while written, a shader resource while read,
    // back to UAV for the next frame's write), the last pass writes Output
    for (int k = 0; k < m_passes; ++k)
    {
        NVSDK_NGX_Parameter* p = m_pparams[k];
        ID3D12Resource* in = k == 0 ? m_color.Get() : m_passOut[k - 1].Get();
        ID3D12Resource* out = k == m_passes - 1 ? m_output.Get() : m_passOut[k].Get();
        if (k > 0)
        {
            const D3D12_RESOURCE_BARRIER b =
                transition(in, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            list->ResourceBarrier(1, &b);
        }
        p->Set("DLSSNR.Color", in);
        p->Set("DLSSNR.Output", out);
        p->Set("DLSSNR.Backbuffer", out);
        p->Set("DLSSNR.Reset", reset ? 1 : 0);
        p->Set("DLSSNR.Intensity", m_set.intensity);
        p->Set("DLSSNR.LocalStructureStrength", m_set.structure);
        p->Set("DLSSNR.LocalToneStrength", m_set.tone);
        m_last = sEval(pEval, list, m_pfeature[k], p);
        if (m_last != NVSDK_NGX_Result_Success)
        {
            err = std::string("EvaluateFeature failed") +
                  (m_passes > 1 ? " (pass " + std::to_string(k + 1) + ")" : std::string()) + ": " +
                  resultString(m_last);
            return false;
        }
        if (k > 0)
        {
            const D3D12_RESOURCE_BARRIER b =
                transition(in, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            list->ResourceBarrier(1, &b);
        }
    }
    return true;
}

bool Host::renderFrame(const void* src, void* dst, bool reset, std::string& err)
{
    const size_t srcRow = (size_t)m_w * 8;

    // Upload the frame into the staging buffer, row by row (the copy footprint
    // row pitch is 256 byte aligned, the caller's frame is tightly packed).
    void* mapped = nullptr;
    D3D12_RANGE none = {0, 0};
    if (FAILED(m_upload->Map(0, &none, &mapped)))
    {
        err = "upload Map failed";
        return false;
    }
    for (uint32_t y = 0; y < m_h; ++y)
        memcpy((uint8_t*)mapped + (size_t)y * m_rowPitch, (const uint8_t*)src + y * srcRow, srcRow);
    m_upload->Unmap(0, nullptr);

    if (FAILED(m_alloc->Reset()) || FAILED(m_list->Reset(m_alloc.Get(), nullptr)))
    {
        err = "command list Reset failed";
        return false;
    }

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

    D3D12_RESOURCE_BARRIER b =
        transition(m_color.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    m_list->ResourceBarrier(1, &b);
    m_list->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
    b = transition(m_color.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
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

    if (!runCommandList(err))
        return false;

    // A copyable footprint pads every row to 256 bytes EXCEPT the last, so the buffer is
    // RowPitch * (h - 1) + w * 8 bytes: mapping RowPitch * h reaches past its end and Map fails
    // for every width that is not a multiple of 32 (the render then ran without DLSS 5; the
    // same finding and fix as DLSS5-NeuralScreen's readback).
    D3D12_RANGE all = {0, (SIZE_T)((size_t)m_rowPitch * (m_h - 1) + srcRow)};
    void* rd = nullptr;
    if (FAILED(m_readback->Map(0, &all, &rd)))
    {
        err = "readback Map failed";
        return false;
    }
    for (uint32_t y = 0; y < m_h; ++y)
        memcpy((uint8_t*)dst + y * srcRow, (const uint8_t*)rd + (size_t)y * m_rowPitch, srcRow);
    m_readback->Unmap(0, &none);
    return true;
}

LUID Host::adapterLuid() const
{
    LUID l = {};
    if (m_dev)
        l = m_dev->GetAdapterLuid();
    return l;
}

void Host::closeSharedHandles()
{
    for (HANDLE* h : {&m_shInH, &m_shOutH, &m_shFenceH, &m_shMvH})
        if (*h)
        {
            CloseHandle(*h);
            *h = nullptr;
        }
}

bool Host::startShared(std::string& err)
{
    if (!m_dev || !m_color)
    {
        err = "the shared handoff needs a started host";
        return false;
    }
    const D3D12_RESOURCE_DESC td = m_color->GetDesc();
    UINT64 total = 0;
    m_dev->GetCopyableFootprints(&td, 0, 1, 0, nullptr, nullptr, nullptr, &total);

    D3D12_HEAP_PROPERTIES def = {};
    def.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC bd = {};
    bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bd.Width = total;
    bd.Height = 1;
    bd.DepthOrArraySize = 1;
    bd.MipLevels = 1;
    bd.Format = DXGI_FORMAT_UNKNOWN;
    bd.SampleDesc.Count = 1;
    bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    // Shared resources start in COMMON: a buffer promotes to COPY_SOURCE / COPY_DEST implicitly
    // and decays back to COMMON when its list completes, the state CUDA reads and writes it in.
    if (FAILED(m_dev->CreateCommittedResource(&def, D3D12_HEAP_FLAG_SHARED, &bd, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                              IID_PPV_ARGS(&m_shIn))) ||
        FAILED(m_dev->CreateCommittedResource(&def, D3D12_HEAP_FLAG_SHARED, &bd, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                              IID_PPV_ARGS(&m_shOut))))
    {
        err = "shared buffer creation failed";
        return false;
    }
    if (FAILED(m_dev->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&m_shFence))))
    {
        err = "shared fence creation failed";
        return false;
    }
    if (FAILED(m_dev->CreateSharedHandle(m_shIn.Get(), nullptr, GENERIC_ALL, nullptr, &m_shInH)) ||
        FAILED(m_dev->CreateSharedHandle(m_shOut.Get(), nullptr, GENERIC_ALL, nullptr, &m_shOutH)) ||
        FAILED(m_dev->CreateSharedHandle(m_shFence.Get(), nullptr, GENERIC_ALL, nullptr, &m_shFenceH)))
    {
        closeSharedHandles();
        err = "shared handle creation failed";
        return false;
    }
    if (m_mv)
    {
        const D3D12_RESOURCE_DESC md = m_mv->GetDesc();
        UINT64 mtotal = 0;
        m_dev->GetCopyableFootprints(&md, 0, 1, 0, nullptr, nullptr, nullptr, &mtotal);
        D3D12_RESOURCE_DESC mb = bd;
        mb.Width = mtotal;
        if (FAILED(m_dev->CreateCommittedResource(&def, D3D12_HEAP_FLAG_SHARED, &mb, D3D12_RESOURCE_STATE_COMMON,
                                                  nullptr, IID_PPV_ARGS(&m_shMv))) ||
            FAILED(m_dev->CreateSharedHandle(m_shMv.Get(), nullptr, GENERIC_ALL, nullptr, &m_shMvH)))
        {
            closeSharedHandles();
            err = "shared motion buffer creation failed";
            return false;
        }
        m_shMvBytes = mtotal;
    }
    for (int k = 0; k < kSharedLists; ++k)
    {
        if (FAILED(m_dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_shAlloc[k]))) ||
            FAILED(m_dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_shAlloc[k].Get(), nullptr,
                                            IID_PPV_ARGS(&m_shList[k]))))
        {
            closeSharedHandles();
            err = "shared command list creation failed";
            return false;
        }
        m_shList[k]->Close();
        m_shDone[k] = 0;
    }
    m_shNext = 0;
    m_shBytes = total;
    return true;
}

bool Host::submitShared(bool reset, uint64_t waitValue, uint64_t signalValue, std::string& err)
{
    if (!m_shBytes)
    {
        err = "the shared handoff is not started";
        return false;
    }
    const int k = m_shNext;
    m_shNext = (k + 1) % kSharedLists;
    // This allocator's previous list must have run. The caller's stream signals every value the
    // queue waits for before it asks for the next frame, so this only waits out GPU work.
    if (m_shFence->GetCompletedValue() < m_shDone[k])
    {
        if (FAILED(m_shFence->SetEventOnCompletion(m_shDone[k], m_fenceEvent)) ||
            WaitForSingleObject(m_fenceEvent, 10000) != WAIT_OBJECT_0)
        {
            err = "an earlier DLSS 5 frame did not finish within 10 s";
            return false;
        }
    }
    ID3D12CommandAllocator* alloc = m_shAlloc[k].Get();
    ID3D12GraphicsCommandList* list = m_shList[k].Get();
    if (FAILED(alloc->Reset()) || FAILED(list->Reset(alloc, nullptr)))
    {
        err = "command list Reset failed";
        return false;
    }

    D3D12_TEXTURE_COPY_LOCATION tex = {};
    tex.pResource = m_color.Get();
    tex.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    tex.SubresourceIndex = 0;
    D3D12_TEXTURE_COPY_LOCATION buf = {};
    buf.pResource = m_shIn.Get();
    buf.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    buf.PlacedFootprint.Offset = 0;
    buf.PlacedFootprint.Footprint.Format = kFmt;
    buf.PlacedFootprint.Footprint.Width = m_w;
    buf.PlacedFootprint.Footprint.Height = m_h;
    buf.PlacedFootprint.Footprint.Depth = 1;
    buf.PlacedFootprint.Footprint.RowPitch = (UINT)m_rowPitch;

    D3D12_RESOURCE_BARRIER b =
        transition(m_color.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    list->ResourceBarrier(1, &b);
    list->CopyTextureRegion(&tex, 0, 0, 0, &buf, nullptr);
    b = transition(m_color.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    list->ResourceBarrier(1, &b);
    if (m_shMv)
    {
        // this frame's motion field, written by the caller before the fence value the queue waits for
        D3D12_TEXTURE_COPY_LOCATION mTex = tex;
        mTex.pResource = m_mv.Get();
        D3D12_TEXTURE_COPY_LOCATION mBuf = buf;
        mBuf.pResource = m_shMv.Get();
        mBuf.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R16G16_FLOAT;
        mBuf.PlacedFootprint.Footprint.RowPitch = (UINT)m_mvPitch;
        b = transition(m_mv.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
        list->ResourceBarrier(1, &b);
        list->CopyTextureRegion(&mTex, 0, 0, 0, &mBuf, nullptr);
        b = transition(m_mv.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        list->ResourceBarrier(1, &b);
    }

    if (!evaluateOn(list, reset, err))
    {
        list->Close();
        return false;
    }

    b = transition(m_output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &b);
    D3D12_TEXTURE_COPY_LOCATION outTex = tex;
    outTex.pResource = m_output.Get();
    D3D12_TEXTURE_COPY_LOCATION outBuf = buf;
    outBuf.pResource = m_shOut.Get();
    list->CopyTextureRegion(&outBuf, 0, 0, 0, &outTex, nullptr);
    b = transition(m_output.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    list->ResourceBarrier(1, &b);
    if (FAILED(list->Close()))
    {
        err = "command list Close failed";
        return false;
    }

    if (FAILED(m_queue->Wait(m_shFence.Get(), waitValue)))
    {
        err = "queue Wait failed";
        return false;
    }
    ID3D12CommandList* lists[] = {list};
    m_queue->ExecuteCommandLists(1, lists);
    if (FAILED(m_queue->Signal(m_shFence.Get(), signalValue)))
    {
        err = "queue Signal failed";
        return false;
    }
    m_shDone[k] = signalValue;
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
    for (int k = 0; k + 1 < kMaxPasses; ++k)
        m_passOut[k].Reset();
    for (int k = 0; k < kSharedLists; ++k)
    {
        m_shList[k].Reset();
        m_shAlloc[k].Reset();
    }
    m_shIn.Reset();
    m_shOut.Reset();
    m_shFence.Reset();
    m_shMv.Reset();
    m_shBytes = 0;
    m_shMvBytes = 0;
    closeSharedHandles();
    m_readback.Reset();
    m_upload.Reset();
    m_output.Reset();
    m_color.Reset();
    m_mv.Reset();
    // on a caller device the queue and device references are just dropped
    m_list.Reset();
    m_alloc.Reset();
    m_queue.Reset();
    m_fence.Reset();
    m_dev.Reset();
    if (m_fenceEvent)
    {
        CloseHandle(m_fenceEvent);
        m_fenceEvent = nullptr;
    }
    // The snippet and the driver core stay loaded for the life of the process on
    // purpose: NGX does not support a clean unload and reload in place.
}

void Host::abandon()
{
    m_feature = nullptr;
    m_params = nullptr;
    for (int k = 0; k < kMaxPasses; ++k)
    {
        m_pfeature[k] = nullptr;
        m_pparams[k] = nullptr;
    }
    for (int k = 0; k + 1 < kMaxPasses; ++k)
        m_passOut[k].Reset();
    m_passes = 1;
    m_ngxUp = false;
    for (int k = 0; k < kSharedLists; ++k)
    {
        m_shList[k].Reset();
        m_shAlloc[k].Reset();
    }
    m_shIn.Reset();
    m_shOut.Reset();
    m_shFence.Reset();
    m_shMv.Reset();
    m_shBytes = 0;
    m_shMvBytes = 0;
    closeSharedHandles();
    m_readback.Reset();
    m_upload.Reset();
    m_output.Reset();
    m_color.Reset();
    m_mv.Reset();
    m_list.Reset();
    m_alloc.Reset();
    m_queue.Reset();
    m_fence.Reset();
    m_dev.Reset();
    if (m_fenceEvent)
    {
        CloseHandle(m_fenceEvent);
        m_fenceEvent = nullptr;
    }
}

void Host::ngxShutdown()
{
    // Every feature and the NGX session exist only after the shim loaded (startup refuses without it).
    void* pRelease = m_snippet ? (void*)GetProcAddress(m_snippet, "NVSDK_NGX_D3D12_ReleaseFeature") : nullptr;
    auto sRel = m_shim ? (PFN_ShimRelease)GetProcAddress(m_shim, "DLSSNR_CallRelease") : nullptr;
    // passes 2..: their features and parameter blocks first (pass 0 = m_feature / m_params below)
    for (int k = 1; k < kMaxPasses && m_core; ++k)
    {
        if (m_pfeature[k])
        {
            if (pRelease && sRel)
                sRel(pRelease, m_pfeature[k]);
            m_pfeature[k] = nullptr;
        }
        if (m_pparams[k])
        {
            PFN_NGX_DestroyParams destroy =
                (PFN_NGX_DestroyParams)GetProcAddress(m_core, "NVSDK_NGX_D3D12_DestroyParameters");
            if (destroy)
                destroy(m_pparams[k]);
            m_pparams[k] = nullptr;
        }
    }
    m_pfeature[0] = nullptr;
    m_pparams[0] = nullptr;
    if (m_core && m_feature)
    {
        if (pRelease && sRel)
            sRel(pRelease, m_feature);
        m_feature = nullptr;
    }
    if (m_core && m_params)
    {
        PFN_NGX_DestroyParams destroy =
            (PFN_NGX_DestroyParams)GetProcAddress(m_core, "NVSDK_NGX_D3D12_DestroyParameters");
        if (destroy)
            destroy(m_params);
        m_params = nullptr;
    }
    if (m_core && m_ngxUp)
    {
        void* pShut = (void*)GetProcAddress(m_core, "NVSDK_NGX_D3D12_Shutdown1");
        auto sShut = m_shim ? (PFN_ShimShutdown)GetProcAddress(m_shim, "DLSSNR_CallShutdown") : nullptr;
        if (pShut && sShut)
            sShut(pShut, m_dev.Get());
        m_ngxUp = false;
    }
}

static void guardedNgxShutdown(Host* self)
{
    __try
    {
        self->ngxShutdown();
    }
    // every fault of the NGX teardown is skipped on purpose: the chain faults on the way out
#pragma warning(suppress : 6320)
    __except (EXCEPTION_EXECUTE_HANDLER)
    {
        fprintf(stderr, "dlssnr: NGX teardown faulted, skipped\n");
        fflush(stderr);
    }
}

} // namespace nr
