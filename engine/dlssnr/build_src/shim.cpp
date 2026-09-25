// shim.cpp - the caller shim, built as engine\dlssnr\nvngx.dll.
//
// The NR snippet validates its caller (observed fact; the exact rule, module
// name or address range or export table, is unknown). Every NGX entry point the
// host uses is therefore invoked from inside this module, so the return address
// the snippet sees belongs to a DLL named nvngx.dll.
//
// The thunks are deliberately feature agnostic: the host resolves the driver
// core's exports and hands each one in, so nothing DLSS 5 specific lives here.
#define SMV_SHIM_BUILD
#include "shim_abi.h"

BOOL APIENTRY DllMain(HMODULE, DWORD, LPVOID)
{
    return TRUE;
}

NVSDK_NGX_Result DLSSNR_CallInit(void* fn, int argOrder, const char* projectId,
                                 NVSDK_NGX_EngineType engineType, const char* engineVersion,
                                 const wchar_t* dataPath, ID3D12Device* device,
                                 const NVSDK_NGX_FeatureCommonInfo* featureInfo,
                                 NVSDK_NGX_Version sdkVersion)
{
    if (!fn) return NVSDK_NGX_Result_FAIL_InvalidParameter;
    if (argOrder == 0)
        return ((PFN_NGX_InitProjectID_A)fn)(projectId, engineType, engineVersion, dataPath, device,
                                             featureInfo, sdkVersion);
    return ((PFN_NGX_InitProjectID_B)fn)(projectId, engineType, engineVersion, dataPath, device,
                                         sdkVersion, featureInfo);
}

NVSDK_NGX_Result DLSSNR_CallInitExt(void* fn, unsigned long long appId, const wchar_t* dataPath,
                                    ID3D12Device* device, NVSDK_NGX_Version sdkVersion)
{
    if (!fn) return NVSDK_NGX_Result_FAIL_InvalidParameter;
    return ((PFN_NGX_InitExt)fn)(appId, dataPath, device, sdkVersion, nullptr);
}

// Snippet-side Init_Ext: (appId, dataPath, device, version, FeatureCommonInfo*), the
// reference sequence for nvngx_dlssnr.dll's own export (version passed raw, 0x15).
typedef NVSDK_NGX_Result (*PFN_SnipInitExt)(unsigned long long, const wchar_t*, ID3D12Device*,
                                            unsigned int, const NVSDK_NGX_FeatureCommonInfo*);
extern "C" __declspec(dllexport)
NVSDK_NGX_Result DLSSNR_CallInitExtF(void* fn, unsigned long long appId, const wchar_t* dataPath,
                                     ID3D12Device* device, unsigned int version,
                                     const NVSDK_NGX_FeatureCommonInfo* featureInfo)
{
    if (!fn) return NVSDK_NGX_Result_FAIL_InvalidParameter;
    return ((PFN_SnipInitExt)fn)(appId, dataPath, device, version, featureInfo);
}

NVSDK_NGX_Result DLSSNR_CallAllocParams(void* fn, NVSDK_NGX_Parameter** outParams)
{
    if (!fn) return NVSDK_NGX_Result_FAIL_InvalidParameter;
    return ((PFN_NGX_AllocParams)fn)(outParams);
}

NVSDK_NGX_Result DLSSNR_CallCreate(void* fn, ID3D12GraphicsCommandList* list, int featureId,
                                   NVSDK_NGX_Parameter* params, NVSDK_NGX_Handle** outHandle)
{
    if (!fn) return NVSDK_NGX_Result_FAIL_InvalidParameter;
    return ((PFN_NGX_CreateFeature)fn)(list, (NVSDK_NGX_Feature)featureId, params, outHandle);
}

NVSDK_NGX_Result DLSSNR_CallEvaluate(void* fn, ID3D12GraphicsCommandList* list,
                                     const NVSDK_NGX_Handle* handle, const NVSDK_NGX_Parameter* params)
{
    if (!fn) return NVSDK_NGX_Result_FAIL_InvalidParameter;
    return ((PFN_NGX_EvaluateFeature)fn)(list, handle, params, nullptr);
}

NVSDK_NGX_Result DLSSNR_CallRelease(void* fn, NVSDK_NGX_Handle* handle)
{
    if (!fn) return NVSDK_NGX_Result_FAIL_InvalidParameter;
    return ((PFN_NGX_ReleaseFeature)fn)(handle);
}

NVSDK_NGX_Result DLSSNR_CallShutdown(void* fn, ID3D12Device* device)
{
    if (!fn) return NVSDK_NGX_Result_FAIL_InvalidParameter;
    return ((PFN_NGX_Shutdown1)fn)(device);
}
