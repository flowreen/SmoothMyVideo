// amdxc64.dll for SmoothMyVideo's AMD FSR frame generation bridge (engine\fsrfg): a stand-in for AMD's driver
// extension DLL, so AMD's frame generation DLL offers its FSR 4 (ML) provider on a GPU without AMD's driver. That DLL asks
// AmdExtD3DCreateInterface for IAmdExtD3DFactory, the factory for IAmdExtD3DDevice8 (GetWaveMatrixProperties: the
// wave matrix types, here FP8 x FP8 -> FP32 at 16 x 16 x 16) and IAmdExtD3DShaderIntrinsics (every intrinsic reported
// supported: the ML shaders' WMMA FP8 operations, which the bridge's vkd3d-proton translates to Vulkan cooperative
// matrices). Interface ids and vtable order as OptiScaler documents them (Amdxc64_Hooks.h); no OptiScaler or AMD code.
// SMV_FSRFG_SHIM_LOG=1 logs every call to stderr.
#include <windows.h>
#include <unknwn.h>
#include <cstdint>
#include <cstdio>

static const GUID kIidFactory = {0x014937EC, 0x9288, 0x446F, {0xA9, 0xAC, 0xD7, 0x5A, 0x8E, 0x3A, 0x98, 0x4F}};
static const GUID kIidDevice8 = {0xF714E11A, 0xB54E, 0x4E0F, {0xAB, 0xC5, 0xDF, 0x58, 0xB1, 0x81, 0x33, 0xD1}};
static const GUID kIidIntrinsics = {0xBA019D53, 0xCCAB, 0x4CBD, {0xB5, 0x6A, 0x72, 0x30, 0xED, 0x43, 0x30, 0xAD}};

enum WaveMatrixType : int32_t
{
    kFloat32 = 0x1,
    kFp8 = 0xB,
};

struct alignas(8) WaveMatrixProperties
{
    uint64_t mSize, nSize, kSize;
    WaveMatrixType aType, bType, cType, resultType;
    bool saturatingAccumulation;
};

static bool logOn()
{
    static int on = -1;
    if (on < 0)
    {
        char b[4] = {};
        on = GetEnvironmentVariableA("SMV_FSRFG_SHIM_LOG", b, sizeof(b)) == 1 && b[0] == '1';
    }
    return on == 1;
}

static void logGuid(const char* what, REFIID g)
{
    if (logOn())
        fprintf(stderr, "[amdxc64 shim] %s {%08lX-%04X-%04X-%02X%02X-%02X%02X%02X%02X%02X%02X}\n", what, g.Data1,
                g.Data2, g.Data3, g.Data4[0], g.Data4[1], g.Data4[2], g.Data4[3], g.Data4[4], g.Data4[5], g.Data4[6],
                g.Data4[7]);
}

static void logLine(const char* s)
{
    if (logOn())
        fprintf(stderr, "[amdxc64 shim] %s\n", s);
}

#define SHIM_STUB(n)                                                                                                   \
    virtual HRESULT STDMETHODCALLTYPE unknown##n()                                                                     \
    {                                                                                                                  \
        logLine("IAmdExtD3DDevice8 slot " #n " + 2 called");                                                           \
        return S_OK;                                                                                                   \
    }

// IUnknown + 13 slots the frame generation DLL is not known to call + GetWaveMatrixProperties (vtable slot 16)
struct Device8 : IUnknown
{
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppv) override
    {
        logGuid("IAmdExtD3DDevice8::QueryInterface", riid);
        *ppv = nullptr;
        return E_NOINTERFACE;
    }
    ULONG STDMETHODCALLTYPE AddRef() override
    {
        return 1;
    }
    ULONG STDMETHODCALLTYPE Release() override
    {
        return 1;
    }
    SHIM_STUB(1)
    SHIM_STUB(2)
    SHIM_STUB(3)
    SHIM_STUB(4)
    SHIM_STUB(5)
    SHIM_STUB(6)
    SHIM_STUB(7)
    SHIM_STUB(8)
    SHIM_STUB(9)
    SHIM_STUB(10)
    SHIM_STUB(11)
    SHIM_STUB(12)
    SHIM_STUB(13)
    virtual HRESULT STDMETHODCALLTYPE GetWaveMatrixProperties(uint64_t* count, WaveMatrixProperties* p)
    {
        logLine("GetWaveMatrixProperties: one entry, FP8 x FP8 -> FP32");
        if (p && count && *count >= 1)
        {
            p[0].mSize = p[0].nSize = p[0].kSize = 16;
            p[0].aType = p[0].bType = kFp8;
            p[0].cType = p[0].resultType = kFloat32;
            p[0].saturatingAccumulation = false;
        }
        if (count)
            *count = 1;
        return S_OK;
    }
};

// IAmdExtD3DShaderIntrinsics: GetInfo, CheckSupport, Enable
struct Intrinsics : IUnknown
{
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppv) override
    {
        logGuid("IAmdExtD3DShaderIntrinsics::QueryInterface", riid);
        *ppv = nullptr;
        return E_NOINTERFACE;
    }
    ULONG STDMETHODCALLTYPE AddRef() override
    {
        return 1;
    }
    ULONG STDMETHODCALLTYPE Release() override
    {
        return 1;
    }
    virtual HRESULT STDMETHODCALLTYPE GetInfo(void*)
    {
        logLine("ShaderIntrinsics::GetInfo");
        return E_NOTIMPL;
    }
    virtual HRESULT STDMETHODCALLTYPE CheckSupport(int32_t intrinsic)
    {
        if (logOn())
            fprintf(stderr, "[amdxc64 shim] ShaderIntrinsics::CheckSupport(%d) -> supported\n", intrinsic);
        return S_OK;
    }
    virtual HRESULT STDMETHODCALLTYPE Enable()
    {
        logLine("ShaderIntrinsics::Enable -> ok");
        return S_OK;
    }
};

static Device8 g_device8;
static Intrinsics g_intrinsics;

struct Factory : IUnknown
{
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppv) override
    {
        logGuid("IAmdExtD3DFactory::QueryInterface", riid);
        *ppv = nullptr;
        return E_NOINTERFACE;
    }
    ULONG STDMETHODCALLTYPE AddRef() override
    {
        return 1;
    }
    ULONG STDMETHODCALLTYPE Release() override
    {
        return 1;
    }
    virtual HRESULT STDMETHODCALLTYPE CreateInterface(IUnknown*, REFIID riid, void** ppv)
    {
        if (riid == kIidDevice8)
        {
            logLine("CreateInterface IAmdExtD3DDevice8 -> shim");
            *ppv = &g_device8;
            return S_OK;
        }
        if (riid == kIidIntrinsics)
        {
            logLine("CreateInterface IAmdExtD3DShaderIntrinsics -> shim");
            *ppv = &g_intrinsics;
            return S_OK;
        }
        logGuid("CreateInterface, not provided:", riid);
        *ppv = nullptr;
        return E_NOINTERFACE;
    }
};

static Factory g_factory;

extern "C" __declspec(dllexport) HRESULT __cdecl AmdExtD3DCreateInterface(IUnknown*, REFIID riid, void** ppv)
{
    if (riid == kIidFactory)
    {
        logLine("AmdExtD3DCreateInterface IAmdExtD3DFactory -> shim");
        *ppv = &g_factory;
        return S_OK;
    }
    logGuid("AmdExtD3DCreateInterface, not provided:", riid);
    *ppv = nullptr;
    return E_NOINTERFACE;
}
