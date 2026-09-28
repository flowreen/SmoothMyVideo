// nvoffruc_bridge.cpp
//
// Bridge between SmoothMyVideo's Python engine and NVIDIA's NvOFFRUC.dll (the Frame Rate Up
// Conversion library in the NVIDIA Optical Flow SDK). This is the "Nvidia Smooth Motion" model:
// hardware optical-flow interpolation on the OFA engine, deliberately offered as the inferior /
// faster alternative to GMFSS.
//
// This software contains source code provided by NVIDIA Corporation.
//
// Derivative of the NvOFFRUCSample, reduced to the CUDA path and exposed as a flat cdecl C API for
// ctypes, mirroring engine/rtxvideo/build_src. It loads NvOFFRUC.dll from its own folder only after
// an Authenticode check that NVIDIA signed it (signed_by_nvidia; never a plain LoadLibrary).
//
// NO CUDA TOOLKIT NEEDED TO BUILD. The bridge uses only the CUDA *driver* API from nvcuda.dll (the
// always-present driver). Like the SDK sample, it creates its OWN CUDA context (cuCtxCreate) and runs
// FRUC there, with three DEDICATED cuMemAlloc surfaces registered with FRUC; the engine's torch
// frames (in torch's primary context) are copied in/out with cuMemcpyDtoD, which works across
// contexts via CUDA unified addressing. (Historical note: the earlier "access violation" in Process
// was NOT a context issue - it was a pointer-indirection bug. FRUC's pFrame and registered resources
// must be CUdeviceptr* i.e. the HOST address of the variable holding the device pointer, exactly as
// NvOFFRUCSample passes &m_pRenderFrameCudaMemPtr[i], NOT the device-pointer value; the library
// dereferences pFrame host-side at Process, so a raw device value AV'd. The dedicated context is kept
// only because it mirrors the sample and is harmless.) FRUC carries its own cudart64_110.dll.
//
// INSTANCES: up to kMaxInst FRUC instances live in this one module, each with its own CUDA
// context, surfaces and NvOFFRUC handle, so callers on different threads run them concurrently
// (the host runs each level of a pair's midpoint tree, or each tween index of a group, on its own
// instance). The classic exports drive
// instance 0; the _i exports name the instance. One instance is only ever called from one thread
// at a time; the last error is per thread.
//
// Runtime layout (mirrors engine/rtxvideo/): this bridge dll, NvOFFRUC.dll and cudart64_110.dll all
// sit together in engine/nvoffruc/; NvOFFRUC.dll + cudart are user-installed, not redistributed.
//
// Build: see DEVELOPMENT.md, "Building the native bridges".

#include <windows.h>
#include <wincrypt.h>
#include <wintrust.h>
#include <softpub.h>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdint>

#include "NvOFFRUC.h" // SDK: NvOFFRUC/Interface/NvOFFRUC.h (EULA: never commit it)

// ---- NvOFFRUC.dll entry points ---------------------------------------------------------------
static HINSTANCE g_hDLL = nullptr;
static PtrToFuncNvOFFRUCCreate pCreate = nullptr;
static PtrToFuncNvOFFRUCRegisterResource pRegister = nullptr;
static PtrToFuncNvOFFRUCUnregisterResource pUnregister = nullptr;
static PtrToFuncNvOFFRUCProcess pProcess = nullptr;
static PtrToFuncNvOFFRUCDestroy pDestroy = nullptr;

// ---- CUDA driver API (nvcuda.dll) ------------------------------------------------------------
typedef int (*PFN_cuInit)(unsigned int);
typedef int (*PFN_cuDeviceGet)(int*, int);
typedef int (*PFN_cuCtxCreate)(void**, unsigned int, int);
typedef int (*PFN_cuCtxDestroy)(void*);
typedef int (*PFN_cuCtxGetCurrent)(void**);
typedef int (*PFN_cuCtxSetCurrent)(void*);
typedef int (*PFN_cuMemAlloc)(unsigned long long*, size_t);
typedef int (*PFN_cuMemcpyDtoD)(unsigned long long, unsigned long long, size_t);
typedef int (*PFN_cuMemFree)(unsigned long long);
typedef int (*PFN_cuCtxSynchronize)();
static PFN_cuInit cuInitFn = nullptr;
static PFN_cuDeviceGet cuDeviceGetFn = nullptr;
static PFN_cuCtxCreate cuCtxCreateFn = nullptr;
static PFN_cuCtxDestroy cuCtxDestroyFn = nullptr;
static PFN_cuCtxGetCurrent cuCtxGetCurrentFn = nullptr;
static PFN_cuCtxSetCurrent cuCtxSetCurrentFn = nullptr;
static PFN_cuMemAlloc cuMemAllocFn = nullptr;
static PFN_cuMemcpyDtoD cuMemcpyDtoDFn = nullptr;
static PFN_cuMemFree cuMemFreeFn = nullptr;
static PFN_cuCtxSynchronize cuCtxSynchronizeFn = nullptr;

// ---- one FRUC instance ------------------------------------------------------------------------
static const int kMaxInst = 4;
struct Inst
{
    void* ctx = nullptr; // FRUC's dedicated context
    NvOFFRUCHandle fruc = nullptr;
    unsigned long long prev = 0, cur = 0, out = 0;
    unsigned w = 0, h = 0;
    size_t bytes = 0;
    double ts = 0.0;
    // nvoffruc_step's feed state: which input surface (0 = prev, 1 = cur) holds the frame FRUC was
    // fed last (-1 = none since create / reset), its timestamp and the one fed before it. The two
    // input surfaces alternate, so the previous frame is never overwritten while FRUC can still read it.
    int last = -1;
    double lastTs = 0.0, prevTs = 0.0;
    unsigned long long* in_surf(int i)
    {
        return i ? &cur : &prev;
    }
};
static Inst g_inst[kMaxInst];
static Inst* inst_at(int i)
{
    return (i >= 0 && i < kMaxInst) ? &g_inst[i] : nullptr;
}

static thread_local char g_err[512] = {0};
static void set_err(const char* m)
{
    strncpy_s(g_err, sizeof(g_err), m ? m : "", _TRUNCATE);
}
static void set_errf(const char* m, long code)
{
    _snprintf_s(g_err, sizeof(g_err), _TRUNCATE, "%s (code %ld)", m, code);
}

static std::wstring self_dir()
{
    HMODULE h = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       reinterpret_cast<LPCWSTR>(&self_dir), &h);
    wchar_t path[MAX_PATH] = {0};
    GetModuleFileNameW(h, path, MAX_PATH);
    std::wstring s(path);
    size_t slash = s.find_last_of(L"\\/");
    return slash == std::wstring::npos ? std::wstring(L".") : s.substr(0, slash);
}

// true when the file carries a valid Authenticode signature (WinVerifyTrust) whose signer
// certificate names "NVIDIA Corporation" (the SDK's SecureLibraryLoader.h does the same check;
// this bridge does not need that header)
static bool signed_by_nvidia(const std::wstring& path)
{
    WINTRUST_FILE_INFO fi;
    memset(&fi, 0, sizeof(fi));
    fi.cbStruct = sizeof(fi);
    fi.pcwszFilePath = path.c_str();
    WINTRUST_DATA wd;
    memset(&wd, 0, sizeof(wd));
    wd.cbStruct = sizeof(wd);
    wd.dwUIChoice = WTD_UI_NONE;
    wd.fdwRevocationChecks = WTD_REVOKE_NONE;
    wd.dwUnionChoice = WTD_CHOICE_FILE;
    wd.pFile = &fi;
    wd.dwStateAction = WTD_STATEACTION_VERIFY;
    GUID action = WINTRUST_ACTION_GENERIC_VERIFY_V2;
    const LONG trust = WinVerifyTrust(nullptr, &action, &wd);
    wd.dwStateAction = WTD_STATEACTION_CLOSE;
    WinVerifyTrust(nullptr, &action, &wd);
    if (trust != ERROR_SUCCESS)
        return false;

    HCERTSTORE store = nullptr;
    HCRYPTMSG msg = nullptr;
    if (!CryptQueryObject(CERT_QUERY_OBJECT_FILE, path.c_str(), CERT_QUERY_CONTENT_FLAG_PKCS7_SIGNED_EMBED,
                          CERT_QUERY_FORMAT_FLAG_BINARY, 0, nullptr, nullptr, nullptr, &store, &msg, nullptr))
        return false;
    bool ok = false;
    DWORD n = 0;
    if (CryptMsgGetParam(msg, CMSG_SIGNER_INFO_PARAM, 0, nullptr, &n) && n)
    {
        std::string buf(n, '\0');
        auto* si = reinterpret_cast<CMSG_SIGNER_INFO*>(&buf[0]);
        if (CryptMsgGetParam(msg, CMSG_SIGNER_INFO_PARAM, 0, si, &n))
        {
            CERT_INFO ci;
            memset(&ci, 0, sizeof(ci));
            ci.Issuer = si->Issuer;
            ci.SerialNumber = si->SerialNumber;
            PCCERT_CONTEXT cert = CertFindCertificateInStore(store, X509_ASN_ENCODING | PKCS_7_ASN_ENCODING, 0,
                                                             CERT_FIND_SUBJECT_CERT, &ci, nullptr);
            if (cert)
            {
                wchar_t name[256] = {0};
                CertGetNameStringW(cert, CERT_NAME_SIMPLE_DISPLAY_TYPE, 0, nullptr, name, 256);
                ok = wcscmp(name, L"NVIDIA Corporation") == 0;
                CertFreeCertificateContext(cert);
            }
        }
    }
    CryptMsgClose(msg);
    CertCloseStore(store, 0);
    return ok;
}

// NvOFFRUC.dll from our own folder only, by full path, after the signature check; the altered
// search path lets it find its cudart64_110.dll beside it
static void secure_load(HINSTANCE* out)
{
    *out = nullptr;
    const std::wstring path = self_dir() + L"\\NvOFFRUC.dll";
    if (GetFileAttributesW(path.c_str()) == INVALID_FILE_ATTRIBUTES || !signed_by_nvidia(path))
        return;
    *out = LoadLibraryExW(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
}

static bool load_dll()
{
    if (g_hDLL)
        return true;
    secure_load(&g_hDLL);
    if (!g_hDLL)
    {
        set_err("SecureLoadLibrary failed (NvOFFRUC.dll missing or not NVIDIA-signed)");
        return false;
    }
    pCreate = (PtrToFuncNvOFFRUCCreate)GetProcAddress(g_hDLL, CreateProcName);
    pRegister = (PtrToFuncNvOFFRUCRegisterResource)GetProcAddress(g_hDLL, RegisterResourceProcName);
    pUnregister = (PtrToFuncNvOFFRUCUnregisterResource)GetProcAddress(g_hDLL, UnregisterResourceProcName);
    pProcess = (PtrToFuncNvOFFRUCProcess)GetProcAddress(g_hDLL, ProcessProcName);
    pDestroy = (PtrToFuncNvOFFRUCDestroy)GetProcAddress(g_hDLL, DestroyProcName);
    if (!pCreate || !pRegister || !pUnregister || !pProcess || !pDestroy)
    {
        set_err("NvOFFRUC.dll is missing expected exports");
        return false;
    }
    return true;
}

static bool load_driver()
{
    if (cuMemAllocFn)
        return true;
    HMODULE cu = GetModuleHandleW(L"nvcuda.dll");
    if (!cu)
        cu = LoadLibraryW(L"nvcuda.dll");
    if (!cu)
    {
        set_err("nvcuda.dll (CUDA driver) not found");
        return false;
    }
    cuInitFn = (PFN_cuInit)GetProcAddress(cu, "cuInit");
    cuDeviceGetFn = (PFN_cuDeviceGet)GetProcAddress(cu, "cuDeviceGet");
    cuCtxCreateFn = (PFN_cuCtxCreate)GetProcAddress(cu, "cuCtxCreate_v2");
    cuCtxDestroyFn = (PFN_cuCtxDestroy)GetProcAddress(cu, "cuCtxDestroy_v2");
    cuCtxGetCurrentFn = (PFN_cuCtxGetCurrent)GetProcAddress(cu, "cuCtxGetCurrent");
    cuCtxSetCurrentFn = (PFN_cuCtxSetCurrent)GetProcAddress(cu, "cuCtxSetCurrent");
    cuMemAllocFn = (PFN_cuMemAlloc)GetProcAddress(cu, "cuMemAlloc_v2");
    cuMemcpyDtoDFn = (PFN_cuMemcpyDtoD)GetProcAddress(cu, "cuMemcpyDtoD_v2");
    cuMemFreeFn = (PFN_cuMemFree)GetProcAddress(cu, "cuMemFree_v2");
    cuCtxSynchronizeFn = (PFN_cuCtxSynchronize)GetProcAddress(cu, "cuCtxSynchronize");
    if (!cuInitFn || !cuDeviceGetFn || !cuCtxCreateFn || !cuCtxDestroyFn || !cuCtxGetCurrentFn || !cuCtxSetCurrentFn ||
        !cuMemAllocFn || !cuMemcpyDtoDFn || !cuMemFreeFn || !cuCtxSynchronizeFn)
    {
        set_err("CUDA driver entry points missing");
        return false;
    }
    return true;
}

// FRUC's context current for a call; returns the caller's context (torch's / the host's) to restore
static void* enter_ctx(Inst& s)
{
    void* p = nullptr;
    cuCtxGetCurrentFn(&p);
    cuCtxSetCurrentFn(s.ctx);
    return p;
}
static void leave_ctx(void* p)
{
    cuCtxSetCurrentFn(p);
}

static int safe_process(Inst& s, const NvOFFRUC_PROCESS_IN_PARAMS* in, const NvOFFRUC_PROCESS_OUT_PARAMS* out,
                        NvOFFRUC_STATUS* st)
{
    __try
    {
        *st = pProcess(s.fruc, in, out);
        return 0;
    }
    __except (EXCEPTION_EXECUTE_HANDLER)
    {
        return (int)GetExceptionCode();
    }
}

static int create_inst(Inst& s, unsigned width, unsigned height)
{
    if (s.fruc)
        return 0;
    if (!load_dll())
        return -1;
    if (!load_driver())
        return -6;

    cuInitFn(0);
    int dev = 0;
    if (cuDeviceGetFn(&dev, 0) != 0)
    {
        set_err("cuDeviceGet(0) failed");
        return -8;
    }
    void* caller = nullptr;
    cuCtxGetCurrentFn(&caller); // remember torch's context
    if (cuCtxCreateFn(&s.ctx, 0, dev) != 0)
    {
        set_err("cuCtxCreate failed");
        return -9;
    }
    // s.ctx is now current.

    s.w = width;
    s.h = height;
    s.bytes = size_t(width) * height * 4;
    int ok = (cuMemAllocFn(&s.prev, s.bytes) == 0) && (cuMemAllocFn(&s.cur, s.bytes) == 0) &&
             (cuMemAllocFn(&s.out, s.bytes) == 0);
    if (!ok)
    {
        set_err("cuMemAlloc for FRUC surfaces failed");
        cuCtxSetCurrentFn(caller);
        return -7;
    }

    NvOFFRUC_CREATE_PARAM cp;
    memset(&cp, 0, sizeof(cp));
    cp.uiWidth = width;
    cp.uiHeight = height;
    cp.pDevice = nullptr;
    cp.eResourceType = CudaResource;
    cp.eSurfaceFormat = ARGBSurface;
    cp.eCUDAResourceType = CudaResourceCuDevicePtr;
    NvOFFRUC_STATUS st = pCreate(&cp, &s.fruc);
    if (st != NvOFFRUC_SUCCESS)
    {
        s.fruc = nullptr;
        set_errf("NvOFFRUCCreate failed", (long)st);
        cuCtxSetCurrentFn(caller);
        return -4;
    }

    NvOFFRUC_REGISTER_RESOURCE_PARAM rp;
    memset(&rp, 0, sizeof(rp));
    // Resources are CUdeviceptr* (host addresses of the device-pointer variables), matching the pFrame
    // form used in Process below. Passing the device-pointer values here (and there) was the "AV
    // reading a device pointer" bug; FRUC dereferences these host-side at Process time.
    rp.pArrResource[0] = (void*)&s.prev;
    rp.pArrResource[1] = (void*)&s.cur;
    rp.pArrResource[2] = (void*)&s.out;
    rp.uiCount = 3;
    rp.pD3D11FenceObj = nullptr;
    st = pRegister(s.fruc, &rp);
    if (st != NvOFFRUC_SUCCESS)
    {
        set_errf("NvOFFRUCRegisterResource failed", (long)st);
        cuCtxSetCurrentFn(caller);
        return -5;
    }

    cuCtxSetCurrentFn(caller); // restore torch's context
    s.ts = 0.0;
    s.last = -1;
    return 0;
}

// one Process call on FRUC's context (current): frame surface `f` at timestamp `ts`, output at `tOut`
static int process_one(Inst& s, unsigned long long* f, double ts, bool skipWarp, double tOut, bool* repeated)
{
    NvOFFRUC_PROCESS_IN_PARAMS in;
    memset(&in, 0, sizeof(in));
    NvOFFRUC_PROCESS_OUT_PARAMS out;
    memset(&out, 0, sizeof(out));
    in.stFrameDataInput.pFrame = (void*)f;
    in.stFrameDataInput.nTimeStamp = ts;
    in.stFrameDataInput.nCuSurfacePitch = size_t(s.w) * 4;
    in.bSkipWarp = skipWarp ? 1 : 0;
    out.stFrameDataOutput.pFrame = (void*)&s.out;
    out.stFrameDataOutput.nTimeStamp = tOut;
    out.stFrameDataOutput.nCuSurfacePitch = size_t(s.w) * 4;
    out.stFrameDataOutput.bHasFrameRepetitionOccurred = (void*)repeated;
    NvOFFRUC_STATUS st = NvOFFRUC_SUCCESS;
    const int ex = safe_process(s, &in, &out, &st);
    if (ex)
    {
        set_errf("NvOFFRUCProcess crashed - access violation", ex);
        return -30;
    }
    if (st != NvOFFRUC_SUCCESS)
    {
        set_errf("NvOFFRUCProcess failed", (long)st);
        return -3;
    }
    return 0;
}

// FEED-ONCE tweens: nvoffruc_interpolate re-feeds BOTH
// frames per tween (a bSkipWarp prime costs a full optical flow, as much as the warp: 15.1 vs 14.6
// ms at 2560x1472), so k tweens on a pair cost 2k Process calls. Here the caller says what FRUC
// already has:
//   mode 0: prev is not the frame fed last: prime prev (bSkipWarp), then feed cur (2 calls)
//   mode 1: prev IS the frame fed last (a continuous stream): feed cur only (1 call)
//   mode 2: another tween of the pair fed last: cur again at its SAME timestamp, only the output
//           timestamp moves (1 call, no copy in; its tweens match the re-feed way within
//           NvOFFRUC's run-to-run noise)
// t in (0, 1) between prev and cur. The caller's prevPtr / curPtr must hold the frames it names.
static int step_inst(Inst& s, void* prevPtr, void* curPtr, void* outPtr, double t, int mode, int* frameRepeated)
{
    if (!s.fruc)
    {
        set_err("nvoffruc_step before nvoffruc_create");
        return -1;
    }
    if (mode < 0 || mode > 2)
    {
        set_err("nvoffruc_step: unknown mode");
        return -1;
    }
    if (mode != 0 && s.last < 0)
    {
        set_err("nvoffruc_step: mode 1 / 2 with nothing fed yet");
        return -1;
    }
    cuCtxSynchronizeFn(); // SYNC FENCE (in), as in nvoffruc_interpolate
    void* caller = enter_ctx(s);
    bool repeated = false;
    int rc = 0;
    if (mode == 0)
    {
        const int a = 0, b = 1;
        const double ts = s.ts;
        if (cuMemcpyDtoDFn(*s.in_surf(a), (unsigned long long)(uintptr_t)prevPtr, s.bytes) != 0 ||
            cuMemcpyDtoDFn(*s.in_surf(b), (unsigned long long)(uintptr_t)curPtr, s.bytes) != 0)
        {
            set_err("cuMemcpyDtoD of input frames failed");
            leave_ctx(caller);
            return -2;
        }
        rc = process_one(s, s.in_surf(a), ts, true, ts, &repeated);
        if (!rc)
            rc = process_one(s, s.in_surf(b), ts + 1.0, false, ts + t, &repeated);
        s.prevTs = ts;
        s.lastTs = ts + 1.0;
        s.last = b;
    }
    else if (mode == 1)
    {
        const int b = 1 - s.last;
        if (cuMemcpyDtoDFn(*s.in_surf(b), (unsigned long long)(uintptr_t)curPtr, s.bytes) != 0)
        {
            set_err("cuMemcpyDtoD of the input frame failed");
            leave_ctx(caller);
            return -2;
        }
        rc = process_one(s, s.in_surf(b), s.lastTs + 1.0, false, s.lastTs + t, &repeated);
        s.prevTs = s.lastTs;
        s.lastTs += 1.0;
        s.last = b;
    }
    else
    {
        rc = process_one(s, s.in_surf(s.last), s.lastTs, false, s.prevTs + t, &repeated);
    }
    if (rc)
    {
        leave_ctx(caller);
        return rc;
    }
    s.ts = s.lastTs + 1.0;
    if (cuMemcpyDtoDFn((unsigned long long)(uintptr_t)outPtr, s.out, s.bytes) != 0)
    {
        set_err("cuMemcpyDtoD of interpolated frame out failed");
        leave_ctx(caller);
        return -5;
    }
    if (frameRepeated)
        *frameRepeated = repeated ? 1 : 0;
    cuCtxSynchronizeFn(); // SYNC FENCE (out)
    leave_ctx(caller);
    return 0;
}

static void destroy_inst(Inst& s)
{
    void* caller = nullptr;
    if (cuCtxGetCurrentFn)
        cuCtxGetCurrentFn(&caller);
    if (s.ctx)
        cuCtxSetCurrentFn(s.ctx);
    if (s.fruc)
    {
        NvOFFRUC_UNREGISTER_RESOURCE_PARAM up;
        memset(&up, 0, sizeof(up));
        up.pArrResource[0] = (void*)&s.prev;
        up.pArrResource[1] = (void*)&s.cur;
        up.pArrResource[2] = (void*)&s.out;
        up.uiCount = 3;
        if (pUnregister)
            pUnregister(s.fruc, &up);
        if (pDestroy)
            pDestroy(s.fruc);
        s.fruc = nullptr;
    }
    if (cuMemFreeFn)
    {
        if (s.prev)
            cuMemFreeFn(s.prev);
        if (s.cur)
            cuMemFreeFn(s.cur);
        if (s.out)
            cuMemFreeFn(s.out);
    }
    s.prev = s.cur = s.out = 0;
    if (s.ctx)
    {
        cuCtxSetCurrentFn(caller != s.ctx ? caller : nullptr);
        cuCtxDestroyFn(s.ctx);
        s.ctx = nullptr;
    }
    s.last = -1;
}

extern "C"
{

    __declspec(dllexport) const char* nvoffruc_last_error()
    {
        return g_err;
    }

    __declspec(dllexport) int nvoffruc_probe()
    {
        if (g_hDLL)
            return 1;
        HINSTANCE h = nullptr;
        secure_load(&h);
        if (!h)
        {
            set_err("NvOFFRUC.dll not found beside the bridge or not NVIDIA-signed");
            return 0;
        }
        FreeLibrary(h);
        return 1;
    }

    __declspec(dllexport) int nvoffruc_create(unsigned width, unsigned height)
    {
        return create_inst(g_inst[0], width, height);
    }

    // instance i (0 .. kMaxInst - 1) at width x height; each instance is independent of the others
    __declspec(dllexport) int nvoffruc_create_i(int i, unsigned width, unsigned height)
    {
        Inst* s = inst_at(i);
        if (!s)
        {
            set_err("nvoffruc_create_i: instance index out of range");
            return -10;
        }
        return create_inst(*s, width, height);
    }

    __declspec(dllexport) int nvoffruc_step(void* prevPtr, void* curPtr, void* outPtr, double t, int mode,
                                            int* frameRepeated)
    {
        return step_inst(g_inst[0], prevPtr, curPtr, outPtr, t, mode, frameRepeated);
    }

    __declspec(dllexport) int nvoffruc_step_i(int i, void* prevPtr, void* curPtr, void* outPtr, double t, int mode,
                                              int* frameRepeated)
    {
        Inst* s = inst_at(i);
        if (!s)
        {
            set_err("nvoffruc_step_i: instance index out of range");
            return -10;
        }
        return step_inst(*s, prevPtr, curPtr, outPtr, t, mode, frameRepeated);
    }

    __declspec(dllexport) int nvoffruc_interpolate(void* prevPtr, void* curPtr, void* outPtr, double t,
                                                   int* frameRepeated)
    {
        Inst& s = g_inst[0];
        if (!s.fruc)
        {
            set_err("nvoffruc_interpolate before nvoffruc_create");
            return -1;
        }
        // SYNC FENCE (in): CUDA does not order this context's null-stream copies against the CALLER's
        // context/streams (torch packs the input surfaces with kernels of its own). Drain the caller's
        // context while it is still current, or the DtoD below can copy half-written surfaces - every
        // tween came out sliced at horizontal seams under torch 2.13 (2.12 won the race by luck).
        cuCtxSynchronizeFn();
        void* caller = enter_ctx(s); // FRUC's context current for the duration

        // cuMemcpyDtoD works across contexts via unified addressing (source is torch's pointer).
        if (cuMemcpyDtoDFn(s.prev, (unsigned long long)(uintptr_t)prevPtr, s.bytes) != 0 ||
            cuMemcpyDtoDFn(s.cur, (unsigned long long)(uintptr_t)curPtr, s.bytes) != 0)
        {
            set_err("cuMemcpyDtoD of input frames failed");
            leave_ctx(caller);
            return -2;
        }

        bool repeated = false;
        const double tsPrev = s.ts, tsCur = s.ts + 1.0;

        NvOFFRUC_PROCESS_IN_PARAMS inPrev;
        memset(&inPrev, 0, sizeof(inPrev));
        NvOFFRUC_PROCESS_OUT_PARAMS outPrev;
        memset(&outPrev, 0, sizeof(outPrev));
        inPrev.stFrameDataInput.pFrame = (void*)&s.prev;
        inPrev.stFrameDataInput.nTimeStamp = tsPrev;
        inPrev.stFrameDataInput.nCuSurfacePitch = size_t(s.w) * 4;
        // Prime with I0 as a STATE-ONLY feed (bSkipWarp=1): update FRUC's previous-frame cache WITHOUT
        // running the warp. Without it, the priming warps I0 against a stale/empty previous - a bogus flow
        // that pollutes the temporal state and drives FRUC's "repeat a source frame" collapse (verified:
        // removing it made per-pair tweens/brackets collapse onto an endpoint). It is load-bearing for the
        // isolated per-pair interpolation the engine drives (reset before each pair, then this clean prime),
        // which is how NvOFFRUC is made to behave like GMFSS's stateless per-pair model. RESTORED after a
        // session stripped it. philipl's vf_nvoffruc omits it and would collapse the same way on hard motion.
        inPrev.bSkipWarp = 1;
        outPrev.stFrameDataOutput.pFrame = (void*)&s.out;
        outPrev.stFrameDataOutput.nTimeStamp = tsPrev;
        outPrev.stFrameDataOutput.nCuSurfacePitch = size_t(s.w) * 4;
        outPrev.stFrameDataOutput.bHasFrameRepetitionOccurred = &repeated;
        NvOFFRUC_STATUS st = NvOFFRUC_SUCCESS;
        int ex = safe_process(s, &inPrev, &outPrev, &st);
        if (ex)
        {
            set_errf("NvOFFRUCProcess(prev) crashed - access violation", ex);
            leave_ctx(caller);
            return -30;
        }
        if (st != NvOFFRUC_SUCCESS)
        {
            set_errf("NvOFFRUCProcess(prev) failed", (long)st);
            leave_ctx(caller);
            return -3;
        }

        NvOFFRUC_PROCESS_IN_PARAMS inCur;
        memset(&inCur, 0, sizeof(inCur));
        NvOFFRUC_PROCESS_OUT_PARAMS outCur;
        memset(&outCur, 0, sizeof(outCur));
        inCur.stFrameDataInput.pFrame = (void*)&s.cur;
        inCur.stFrameDataInput.nTimeStamp = tsCur;
        inCur.stFrameDataInput.nCuSurfacePitch = size_t(s.w) * 4;
        outCur.stFrameDataOutput.pFrame = (void*)&s.out;
        outCur.stFrameDataOutput.nTimeStamp = tsPrev + t;
        outCur.stFrameDataOutput.nCuSurfacePitch = size_t(s.w) * 4;
        outCur.stFrameDataOutput.bHasFrameRepetitionOccurred = &repeated;
        ex = safe_process(s, &inCur, &outCur, &st);
        if (ex)
        {
            set_errf("NvOFFRUCProcess(cur) crashed - access violation", ex);
            leave_ctx(caller);
            return -40;
        }
        if (st != NvOFFRUC_SUCCESS)
        {
            set_errf("NvOFFRUCProcess(cur) failed", (long)st);
            leave_ctx(caller);
            return -4;
        }

        if (cuMemcpyDtoDFn((unsigned long long)(uintptr_t)outPtr, s.out, s.bytes) != 0)
        {
            set_err("cuMemcpyDtoD of interpolated frame out failed");
            leave_ctx(caller);
            return -5;
        }

        if (frameRepeated)
            *frameRepeated = repeated ? 1 : 0;
        s.ts = tsCur + 1.0;
        s.prevTs = tsPrev;
        s.lastTs = tsCur;
        s.last = 1; // s.cur was fed last (nvoffruc_step's state)
        // SYNC FENCE (out): Process + the out-copy above are queued on THIS context's null stream and are
        // not ordered against the caller reading outPtr from its own context. Block until they land, so
        // the interpolated frame is fully materialized when we return (same seam failure mode as above).
        cuCtxSynchronizeFn();
        leave_ctx(caller);
        return 0;
    }

    // Recreate ONLY instance 0's FRUC handle (keep the CUDA context and the registered surfaces), clearing
    // the library's internal OFA temporal-hint state. NvOFFRUC keeps no public knob to disable those hints
    // (NvOFFRUC.h has only bSkipWarp + reserved fields); it seeds each OFA Execute from the previous
    // call's flow. That is correct for a continuous real-frame stream, but when the caller drives
    // bisection (feeding GENERATED in-between frames as pairs at different motion scales) the stale hint
    // is grossly wrong and the OFA diverges into a corrupt flow -> rainbow-band garbage on the tween. A
    // fresh handle makes the next Process a clean first-flow (no hint). Reusing the context + the
    // cuMemAlloc'd surfaces avoids CUDA-context churn, so this is far cheaper than a full destroy+create.
    __declspec(dllexport) int nvoffruc_reset()
    {
        Inst& s = g_inst[0];
        if (!s.fruc)
            return 0;                // nothing yet; nvoffruc_create() will build a fresh one
        void* caller = enter_ctx(s); // FRUC's context current for destroy+create+register
        {
            NvOFFRUC_UNREGISTER_RESOURCE_PARAM up;
            memset(&up, 0, sizeof(up));
            up.pArrResource[0] = (void*)&s.prev;
            up.pArrResource[1] = (void*)&s.cur;
            up.pArrResource[2] = (void*)&s.out;
            up.uiCount = 3;
            if (pUnregister)
                pUnregister(s.fruc, &up);
            if (pDestroy)
                pDestroy(s.fruc);
            s.fruc = nullptr;
        }
        NvOFFRUC_CREATE_PARAM cp;
        memset(&cp, 0, sizeof(cp));
        cp.uiWidth = s.w;
        cp.uiHeight = s.h;
        cp.pDevice = nullptr;
        cp.eResourceType = CudaResource;
        cp.eSurfaceFormat = ARGBSurface;
        cp.eCUDAResourceType = CudaResourceCuDevicePtr;
        NvOFFRUC_STATUS st = pCreate(&cp, &s.fruc);
        if (st != NvOFFRUC_SUCCESS)
        {
            s.fruc = nullptr;
            set_errf("nvoffruc_reset: NvOFFRUCCreate failed", (long)st);
            leave_ctx(caller);
            return -1;
        }
        NvOFFRUC_REGISTER_RESOURCE_PARAM rp;
        memset(&rp, 0, sizeof(rp));
        rp.pArrResource[0] = (void*)&s.prev;
        rp.pArrResource[1] = (void*)&s.cur;
        rp.pArrResource[2] = (void*)&s.out;
        rp.uiCount = 3;
        rp.pD3D11FenceObj = nullptr;
        st = pRegister(s.fruc, &rp);
        if (st != NvOFFRUC_SUCCESS)
        {
            set_errf("nvoffruc_reset: NvOFFRUCRegisterResource failed", (long)st);
            leave_ctx(caller);
            return -2;
        }
        s.ts = 0.0;
        s.last = -1;
        leave_ctx(caller);
        return 0;
    }

    // every instance (a session's end)
    __declspec(dllexport) void nvoffruc_destroy()
    {
        for (Inst& s : g_inst)
            destroy_inst(s);
    }

    __declspec(dllexport) void nvoffruc_destroy_i(int i)
    {
        if (Inst* s = inst_at(i))
            destroy_inst(*s);
    }

} // extern "C"
