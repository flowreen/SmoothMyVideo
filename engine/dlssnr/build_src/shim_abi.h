// shim_abi.h - the ABI between the host and the caller shim (nvngx.dll).
//
// The NR snippet validates its caller: the module that calls its entry points
// must be a DLL named nvngx.dll, so the return address lands inside that module.
// The host resolves the driver core's exports itself and passes each one in as
// the first argument, so the shim stays a set of thin, feature agnostic thunks.
#pragma once

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>

#include <nvsdk_ngx.h>
#include <nvsdk_ngx_params.h>

// Driver core export prototypes, resolved by the host with GetProcAddress on
// _nvngx.dll. The public NGX headers only declare the SDK side wrappers, so the
// exported Init_ProjectID argument order cannot be checked at compile time; both
// plausible orders are declared and nr_host.cpp passes the one that works (kInitArgOrder).
typedef NVSDK_NGX_Result (*PFN_NGX_InitProjectID_A)(const char* projectId, NVSDK_NGX_EngineType engineType,
                                                    const char* engineVersion, const wchar_t* dataPath,
                                                    ID3D12Device* device,
                                                    const NVSDK_NGX_FeatureCommonInfo* featureInfo,
                                                    NVSDK_NGX_Version sdkVersion);
typedef NVSDK_NGX_Result (*PFN_NGX_InitProjectID_B)(const char* projectId, NVSDK_NGX_EngineType engineType,
                                                    const char* engineVersion, const wchar_t* dataPath,
                                                    ID3D12Device* device, NVSDK_NGX_Version sdkVersion,
                                                    const NVSDK_NGX_FeatureCommonInfo* featureInfo);
typedef NVSDK_NGX_Result (*PFN_NGX_InitExt)(unsigned long long appId, const wchar_t* dataPath, ID3D12Device* device,
                                            NVSDK_NGX_Version sdkVersion, const NVSDK_NGX_Parameter* params);
typedef NVSDK_NGX_Result (*PFN_NGX_AllocParams)(NVSDK_NGX_Parameter** outParams);
typedef NVSDK_NGX_Result (*PFN_NGX_DestroyParams)(NVSDK_NGX_Parameter* params);
typedef NVSDK_NGX_Result (*PFN_NGX_CapParams)(NVSDK_NGX_Parameter** outParams);
typedef NVSDK_NGX_Result (*PFN_NGX_CreateFeature)(ID3D12GraphicsCommandList* list, NVSDK_NGX_Feature featureId,
                                                  NVSDK_NGX_Parameter* params, NVSDK_NGX_Handle** outHandle);
typedef NVSDK_NGX_Result (*PFN_NGX_EvaluateFeature)(ID3D12GraphicsCommandList* list, const NVSDK_NGX_Handle* handle,
                                                    const NVSDK_NGX_Parameter* params,
                                                    PFN_NVSDK_NGX_ProgressCallback_C callback);
typedef NVSDK_NGX_Result (*PFN_NGX_ReleaseFeature)(NVSDK_NGX_Handle* handle);
typedef NVSDK_NGX_Result (*PFN_NGX_Shutdown1)(ID3D12Device* device);

// Shim exports. Each takes the driver core entry point as its first argument.
#ifdef SMV_SHIM_BUILD
#define SMV_SHIM_API extern "C" __declspec(dllexport)
#else
#define SMV_SHIM_API extern "C"
#endif

SMV_SHIM_API NVSDK_NGX_Result DLSSNR_CallInit(void* fn, int argOrder, const char* projectId,
                                              NVSDK_NGX_EngineType engineType, const char* engineVersion,
                                              const wchar_t* dataPath, ID3D12Device* device,
                                              const NVSDK_NGX_FeatureCommonInfo* featureInfo,
                                              NVSDK_NGX_Version sdkVersion);
SMV_SHIM_API NVSDK_NGX_Result DLSSNR_CallInitExt(void* fn, unsigned long long appId, const wchar_t* dataPath,
                                                 ID3D12Device* device, NVSDK_NGX_Version sdkVersion);
SMV_SHIM_API NVSDK_NGX_Result DLSSNR_CallAllocParams(void* fn, NVSDK_NGX_Parameter** outParams);
SMV_SHIM_API NVSDK_NGX_Result DLSSNR_CallCreate(void* fn, ID3D12GraphicsCommandList* list, int featureId,
                                                NVSDK_NGX_Parameter* params, NVSDK_NGX_Handle** outHandle);
SMV_SHIM_API NVSDK_NGX_Result DLSSNR_CallEvaluate(void* fn, ID3D12GraphicsCommandList* list,
                                                  const NVSDK_NGX_Handle* handle, const NVSDK_NGX_Parameter* params);
SMV_SHIM_API NVSDK_NGX_Result DLSSNR_CallRelease(void* fn, NVSDK_NGX_Handle* handle);
SMV_SHIM_API NVSDK_NGX_Result DLSSNR_CallShutdown(void* fn, ID3D12Device* device);

// Signatures of the shim exports, for GetProcAddress on the host side.
typedef NVSDK_NGX_Result (*PFN_ShimInit)(void*, int, const char*, NVSDK_NGX_EngineType, const char*, const wchar_t*,
                                         ID3D12Device*, const NVSDK_NGX_FeatureCommonInfo*, NVSDK_NGX_Version);
typedef NVSDK_NGX_Result (*PFN_ShimAllocParams)(void*, NVSDK_NGX_Parameter**);
typedef NVSDK_NGX_Result (*PFN_ShimCreate)(void*, ID3D12GraphicsCommandList*, int, NVSDK_NGX_Parameter*,
                                           NVSDK_NGX_Handle**);
typedef NVSDK_NGX_Result (*PFN_ShimEvaluate)(void*, ID3D12GraphicsCommandList*, const NVSDK_NGX_Handle*,
                                             const NVSDK_NGX_Parameter*);
typedef NVSDK_NGX_Result (*PFN_ShimRelease)(void*, NVSDK_NGX_Handle*);
typedef NVSDK_NGX_Result (*PFN_ShimShutdown)(void*, ID3D12Device*);
