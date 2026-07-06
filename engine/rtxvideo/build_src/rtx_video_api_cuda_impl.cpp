//----------------------------------------------------------------------------------
// File:        rtx_video_api_cuda_impl.cpp
// SDK Version: 1.1.0
//
// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.
//
//----------------------------------------------------------------------------------

/**
*  This sample application demonstrates use of RTX Video SDK
*  by providing an api taking CUDA input and output.
*  Inputs for CUDA are ARGB/ABGR10.
*  Output from VSR is in ARGB or if input is 10 bit, ABGR10.
*  Output from THDR is in 10 bit ABGR10.
*  If both are enabled then VSR -> THDR.
*/

#include "rtx_video_api.h"

#include <nvsdk_ngx_defs.h>
#include <nvsdk_ngx_helpers_truehdr.h>
#include <nvsdk_ngx_helpers_vsr.h>

// link with nvsdk_ngx_<X>.lib

#include <cuda.h>
#include <iostream>
#include <cstring>

#include "utils.h"

#define CUDA_VERSION_INT_101010_2_DEFINED           12080       // this is based on cuda.h 

#if (CUDA_VERSION < CUDA_VERSION_INT_101010_2_DEFINED)
  #define CU_AD_FORMAT_UNORM_INT_101010_2 ((CUarray_format)0x50)
#endif

#if defined(_WIN32)
#define NGX_BREAKPOINT() __debugbreak()
#else
#include <signal.h>
#define NGX_BREAKPOINT() raise(SIGTRAP)
#endif

#define CHECK_CUDA(func)                                                                            \
{                                                                                                   \
    cudaError_t status = (func);                                                                    \
    if(status != cudaError_t(CUDA_SUCCESS)){                                                        \
        std::cerr << "CUDA Error At Line : " << __FILE__ << ":" << __LINE__ << " : "                \
        << cudaGetErrorString(status);                                                              \
        getchar();                                                                                  \
        NGX_BREAKPOINT();                                                                           \
        exit(EXIT_FAILURE);                                                                         \
    }                                                                                               \
}

#define CHECK_NGX(func)                                                                             \
{                                                                                                   \
    NVSDK_NGX_Result status = (func);                                                               \
    if(status != NVSDK_NGX_Result_Success) {                                                        \
        std::cerr << "NGX error at : " << __FILE__ << ":" << __LINE__ << " : "                      \
        << status << std::endl;                                                                     \
        getchar();                                                                                  \
        exit(EXIT_FAILURE);                                                                         \
        }                                                                                           \
}

#define CUDADRV_CHECK(x)                                                                            \
{                                                                                                   \
    CUresult rval;                                                                                  \
    if ((rval = (x)) != CUDA_SUCCESS)                                                               \
    {                                                                                               \
        const char *error_str;                                                                      \
        cuGetErrorString(rval, &error_str);                                                         \
        std::printf("%s():%i: CUDA driver API error: \"%s\"\n", __FUNCTION__, __LINE__, error_str); \
        exit(0);                                                                                    \
    }                                                                                               \
}


class cuda_api_impl
{
    NVSDK_NGX_Parameter*        m_ngxParameters             = nullptr;
    NVSDK_NGX_Handle*           m_TrueHDRFeature            = nullptr;
    NVSDK_NGX_Handle*           m_VSRFeature                = nullptr;

    CUdevice                    m_cuDevice                  = 0;
    CUcontext                   m_cuContext                 = NULL;

    bool                        m_bNeedMiddle               = false;

    CUarray                     m_cuArrayMid                = nullptr;
    CUtexObject                 m_cuTexObjectMid            = 0;
    CUsurfObject                m_cuSurfObjectMid           = 0;
    size_t                      m_uMiddleWidth              = 0;
    size_t                      m_uMiddleHeight             = 0;

    CUarray                     m_cuArraySrc                = nullptr;
    CUtexObject                 m_cuTexObjectSrc            = 0;
    size_t                      m_uSrcArrayWidth            = 0;
    size_t                      m_uSrcArrayHeight           = 0;

    CUarray                     m_cuArrayDst                = nullptr;
    CUsurfObject                m_cuSurfObjectDst           = 0;
    size_t                      m_uDstArrayWidth            = 0;
    size_t                      m_uDstArrayHeight           = 0;

    // Dedicated, persistent arrays for the split single-feature deviceptr paths (VSR-only and
    // TrueHDR-only). Keeping them separate from the combined m_cuArray* above means that running
    // VSR then (a Python-side RCAS sharpen) then TrueHDR as two evals does not thrash one shared
    // array between the two stages' differing dims/formats; each stays at a stable size.
    CUarray                     m_vsrSrcArr                 = nullptr;
    CUtexObject                 m_vsrSrcTex                 = 0;
    size_t                      m_vsrSrcW                   = 0, m_vsrSrcH = 0;
    CUarray                     m_vsrDstArr                 = nullptr;
    CUsurfObject                m_vsrDstSurf                = 0;
    size_t                      m_vsrDstW                   = 0, m_vsrDstH = 0;
    CUarray                     m_thdrSrcArr                = nullptr;
    CUtexObject                 m_thdrSrcTex                = 0;
    size_t                      m_thdrSrcW                  = 0, m_thdrSrcH = 0;
    CUarray                     m_thdrDstArr                = nullptr;
    CUsurfObject                m_thdrDstSurf               = 0;
    size_t                      m_thdrDstW                  = 0, m_thdrDstH = 0;

    void ensure_src_tex(CUarray& arr, CUtexObject& tex, size_t& cw, size_t& ch, size_t nw, size_t nh);
    void ensure_dst_surf(CUarray& arr, CUsurfObject& surf, size_t& cw, size_t& ch, size_t nw, size_t nh,
                         CUarray_format fmt);


public:
    API_BOOL create(void* cuContext, void* cuStream, int iGpu, API_BOOL THDREnable, API_BOOL VSREnable);
    API_BOOL evaluate(uint64_t cuTexObject_Input, uint64_t cuSurfObject_Output, API_RECT inputRect, API_RECT outputRect, API_VSR_Setting* pVSRSetting, API_THDR_Setting* pTHDRSetting, bool runVSR, bool runTHDR);
    API_BOOL evaluate_deviceptr(void* cuDeviceptr_Input, void* cuDeviceptr_Output, API_RECT inputRect, API_RECT outputRect, API_VSR_Setting* pVSRSetting, API_THDR_Setting* pTHDRSetting);
    API_BOOL evaluate_hostptr(void* hostptr_Input, void* hostptr_Output, API_RECT inputRect, API_RECT outputRect, API_VSR_Setting* pVSRSetting, API_THDR_Setting* pTHDRSetting);
    // Single-feature deviceptr paths so a Python-side RCAS sharpen can run between VSR and TrueHDR
    // at the output resolution. VSR: 8-bit RGBA in (inputRect) -> 8-bit RGBA out (outputRect).
    // TrueHDR: 8-bit RGBA in -> packed 10:10:10:2 out, both at the same (output) rect.
    API_BOOL evaluate_vsr_deviceptr(void* cuDeviceptr_Input, void* cuDeviceptr_Output, API_RECT inputRect, API_RECT outputRect, API_VSR_Setting* pVSRSetting);
    API_BOOL evaluate_thdr_deviceptr(void* cuDeviceptr_Input, void* cuDeviceptr_Output, API_RECT inputRect, API_RECT outputRect, API_THDR_Setting* pTHDRSetting);
    void shutdown();
};

API_BOOL cuda_api_impl::create(void* cuContext, void* cuStream, int iGpu, API_BOOL THDREnable, API_BOOL VSREnable)
{
    if (!cuContext)
    {
        CUDADRV_CHECK(cuDeviceGet(&m_cuDevice, iGpu));
        CUDADRV_CHECK(cuDevicePrimaryCtxRetain(&m_cuContext, m_cuDevice));
        cuContext = m_cuContext;
    }

    CHECK_NGX(NVSDK_NGX_CUDA_Init(APP_ID, APP_PATH));

    CHECK_NGX(NVSDK_NGX_CUDA_GetCapabilityParameters(&m_ngxParameters));

    m_bNeedMiddle = (THDREnable && VSREnable);

    if (THDREnable)
    {
        int TrueHDRAvailable = 0;
        CHECK_NGX(m_ngxParameters->Get(NVSDK_NGX_Parameter_TrueHDR_Available, &TrueHDRAvailable));
        if (!TrueHDRAvailable) return API_BOOL_FAIL;

        // check ScratchBufferSize - truehdr is not expected to request any
        size_t byteSize = 0;
        CHECK_NGX(NVSDK_NGX_CUDA_GetScratchBufferSize(NVSDK_NGX_Feature_TrueHDR, m_ngxParameters, &byteSize));
        if (byteSize != 0) return API_BOOL_FAIL;

        NVSDK_NGX_CUDA_TRUEHDR_Create_Params TrueHDRCreateParams = {};
        TrueHDRCreateParams.InCUContext = cuContext;
        TrueHDRCreateParams.InCUStream = cuStream;
        CHECK_NGX(NGX_CUDA_CREATE_TRUEHDR(&m_TrueHDRFeature, m_ngxParameters, &TrueHDRCreateParams));
    }
    if (VSREnable)
    {
        int VSRAvailable = 0;
        CHECK_NGX(m_ngxParameters->Get(NVSDK_NGX_Parameter_VSR_Available, &VSRAvailable));
        if (!VSRAvailable) return API_BOOL_FAIL;

        // check ScratchBufferSize - vsr is not expected to request any
        size_t byteSize = 0;
        CHECK_NGX(NVSDK_NGX_CUDA_GetScratchBufferSize(NVSDK_NGX_Feature_VSR, m_ngxParameters, &byteSize));
        if (byteSize != 0) return API_BOOL_FAIL;

        NVSDK_NGX_CUDA_VSR_Create_Params VSRCreateParams = {};
        VSRCreateParams.InCUContext = cuContext;
        VSRCreateParams.InCUStream = cuStream;
        CHECK_NGX(NGX_CUDA_CREATE_VSR(&m_VSRFeature, m_ngxParameters, &VSRCreateParams));
    }

    return API_BOOL_SUCCESS;
}

API_BOOL cuda_api_impl::evaluate(uint64_t cuTexObject_Input, uint64_t cuSurfObject_Output, API_RECT inputRect, API_RECT outputRect, API_VSR_Setting* pVSRSetting, API_THDR_Setting* pTHDRSetting, bool runVSR, bool runTHDR)
{
    // Run only the requested feature(s). The middle buffer (VSR output feeding TrueHDR input) is only
    // needed when BOTH run in this single call; the split single-feature paths run one feature each, so
    // needMiddle is false for them and VSR/TrueHDR read/write the caller's surfaces directly.
    const bool needMiddle = runVSR && runTHDR && m_VSRFeature && m_TrueHDRFeature;
    if (needMiddle && (!m_cuArrayMid || outputRect.right != m_uMiddleWidth || outputRect.bottom != m_uMiddleHeight))
    {
        if (m_cuArrayMid)
        {
            CUDADRV_CHECK(cuArrayDestroy(m_cuArrayMid));
            CUDADRV_CHECK(cuTexObjectDestroy(m_cuTexObjectMid));
            CUDADRV_CHECK(cuSurfObjectDestroy(m_cuSurfObjectMid));
        }

        m_uMiddleWidth     = outputRect.right;
        m_uMiddleHeight    = outputRect.bottom;

        CUarray_format cuArrayFormat = CU_AD_FORMAT_UNSIGNED_INT8;
        CUDA_ARRAY_DESCRIPTOR cuArrayOutputDesc{
            static_cast<size_t>(outputRect.right),
            static_cast<size_t>(outputRect.bottom),
            cuArrayFormat,
            4
        };
        CUDADRV_CHECK(cuArrayCreate(&m_cuArrayMid, &cuArrayOutputDesc));
        {
            CUDA_RESOURCE_DESC resDescOutput;
            memset(&resDescOutput, 0, sizeof(CUDA_RESOURCE_DESC));
            resDescOutput.resType = CU_RESOURCE_TYPE_ARRAY;
            resDescOutput.res.array.hArray = m_cuArrayMid;

            CUDA_TEXTURE_DESC texDescOutput;
            memset(&texDescOutput, 0, sizeof(CUDA_TEXTURE_DESC));
            texDescOutput.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
            texDescOutput.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
            texDescOutput.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
            texDescOutput.filterMode = CU_TR_FILTER_MODE_LINEAR;
            texDescOutput.flags = CU_TRSF_NORMALIZED_COORDINATES;

            CUDADRV_CHECK(cuTexObjectCreate(&m_cuTexObjectMid, &resDescOutput, &texDescOutput, nullptr));
            CUDADRV_CHECK(cuSurfObjectCreate(&m_cuSurfObjectMid, &resDescOutput));
        }
    }


    if (m_VSRFeature && runVSR)
    {
        NVSDK_NGX_CUDA_VSR_Eval_Params CUDAVsrEvalParams = {};
        CUDAVsrEvalParams.pInput                   = &cuTexObject_Input;
        CUDAVsrEvalParams.pOutput                  = needMiddle ? &m_cuTexObjectMid : (CUsurfObject*)&cuSurfObject_Output;
        CUDAVsrEvalParams.InputSubrectBase.X       = inputRect.left;
        CUDAVsrEvalParams.InputSubrectBase.Y       = inputRect.top;
        CUDAVsrEvalParams.InputSubrectSize.Width   = inputRect.right - inputRect.left;
        CUDAVsrEvalParams.InputSubrectSize.Height  = inputRect.bottom - inputRect.top;
        CUDAVsrEvalParams.OutputSubrectBase.X      = outputRect.left;
        CUDAVsrEvalParams.OutputSubrectBase.Y      = outputRect.top;
        CUDAVsrEvalParams.OutputSubrectSize.Width  = outputRect.right - outputRect.left;
        CUDAVsrEvalParams.OutputSubrectSize.Height = outputRect.bottom - outputRect.top;
        CUDAVsrEvalParams.QualityLevel             = (NVSDK_NGX_VSR_QualityLevel)pVSRSetting->QualityLevel;

        CHECK_NGX(NGX_CUDA_EVALUATE_VSR(m_VSRFeature, m_ngxParameters, &CUDAVsrEvalParams));
    }
    if (m_TrueHDRFeature && runTHDR)
    {
        NVSDK_NGX_CUDA_TRUEHDR_Eval_Params CUDATrueHDREvalParams = {};
        CUDATrueHDREvalParams.pInput                   = needMiddle ? &m_cuTexObjectMid : (CUtexObject*)&cuTexObject_Input;
        CUDATrueHDREvalParams.pOutput                  = &cuSurfObject_Output;
        CUDATrueHDREvalParams.InputSubrectTL.X         = needMiddle ? outputRect.left   : inputRect.left;
        CUDATrueHDREvalParams.InputSubrectTL.Y         = needMiddle ? outputRect.top    : inputRect.top;
        CUDATrueHDREvalParams.InputSubrectBR.Width     = needMiddle ? outputRect.right  : inputRect.right;
        CUDATrueHDREvalParams.InputSubrectBR.Height    = needMiddle ? outputRect.bottom : inputRect.bottom;
        CUDATrueHDREvalParams.OutputSubrectTL.X        = outputRect.left;
        CUDATrueHDREvalParams.OutputSubrectTL.Y        = outputRect.top;
        CUDATrueHDREvalParams.OutputSubrectBR.Width    = outputRect.right;
        CUDATrueHDREvalParams.OutputSubrectBR.Height   = outputRect.bottom;
        CUDATrueHDREvalParams.Contrast                 = pTHDRSetting->Contrast;
        CUDATrueHDREvalParams.Saturation               = pTHDRSetting->Saturation;
        CUDATrueHDREvalParams.MiddleGray               = pTHDRSetting->MiddleGray;
        CUDATrueHDREvalParams.MaxLuminance             = pTHDRSetting->MaxLuminance;

        CHECK_NGX(NGX_CUDA_EVALUATE_TRUEHDR(m_TrueHDRFeature, m_ngxParameters, &CUDATrueHDREvalParams));
    }

    return API_BOOL_SUCCESS;
}


API_BOOL cuda_api_impl::evaluate_deviceptr(void* cuDeviceptr_Input, void* cuDeviceptr_Output, API_RECT inputRect, API_RECT outputRect, API_VSR_Setting* pVSRSetting, API_THDR_Setting* pTHDRSetting)
{
    if (!m_cuArraySrc || inputRect.right != m_uSrcArrayWidth || inputRect.bottom != m_uSrcArrayHeight)
    {
        if (m_cuArraySrc)
        {
            CUDADRV_CHECK(cuArrayDestroy(m_cuArraySrc));
            CUDADRV_CHECK(cuTexObjectDestroy(m_cuTexObjectSrc));
        }

        m_uSrcArrayWidth     = inputRect.right;
        m_uSrcArrayHeight    = inputRect.bottom;

        CUarray_format cuArrayFormat = CU_AD_FORMAT_UNSIGNED_INT8;
        CUDA_ARRAY_DESCRIPTOR cuArrayOutputDesc{
            m_uSrcArrayWidth,
            m_uSrcArrayHeight,
            cuArrayFormat,
            4
        };
        CUDADRV_CHECK(cuArrayCreate(&m_cuArraySrc, &cuArrayOutputDesc));
        {
            CUDA_RESOURCE_DESC resDescOutput;
            memset(&resDescOutput, 0, sizeof(CUDA_RESOURCE_DESC));
            resDescOutput.resType = CU_RESOURCE_TYPE_ARRAY;
            resDescOutput.res.array.hArray = m_cuArraySrc;

            CUDA_TEXTURE_DESC texDescOutput;
            memset(&texDescOutput, 0, sizeof(CUDA_TEXTURE_DESC));
            texDescOutput.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
            texDescOutput.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
            texDescOutput.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
            texDescOutput.filterMode = CU_TR_FILTER_MODE_LINEAR;
            texDescOutput.flags = CU_TRSF_NORMALIZED_COORDINATES;

            CUDADRV_CHECK(cuTexObjectCreate(&m_cuTexObjectSrc, &resDescOutput, &texDescOutput, nullptr));
        }
    }
    if (!m_cuArrayDst || outputRect.right != m_uDstArrayWidth || outputRect.bottom != m_uDstArrayHeight)
    {
        if (m_cuArrayDst)
        {
            CUDADRV_CHECK(cuArrayDestroy(m_cuArrayDst));
            CUDADRV_CHECK(cuTexObjectDestroy(m_cuSurfObjectDst));
        }

        m_uDstArrayWidth     = outputRect.right;
        m_uDstArrayHeight    = outputRect.bottom;

        CUarray_format cuArrayFormat = m_TrueHDRFeature ? CU_AD_FORMAT_UNORM_INT_101010_2 : CU_AD_FORMAT_UNSIGNED_INT8;
        CUDA_ARRAY_DESCRIPTOR cuArrayOutputDesc{
            m_uDstArrayWidth,
            m_uDstArrayHeight,
            cuArrayFormat,
            4
        };
        CUDADRV_CHECK(cuArrayCreate(&m_cuArrayDst, &cuArrayOutputDesc));
        {
            CUDA_RESOURCE_DESC resDescOutput;
            memset(&resDescOutput, 0, sizeof(CUDA_RESOURCE_DESC));
            resDescOutput.resType = CU_RESOURCE_TYPE_ARRAY;
            resDescOutput.res.array.hArray = m_cuArrayDst;

            CUDADRV_CHECK(cuSurfObjectCreate(&m_cuSurfObjectDst, &resDescOutput));
        }
    }
    {
        // copy input from device to array
        CUDA_MEMCPY2D copyParam     = {};
        copyParam.dstMemoryType     = CU_MEMORYTYPE_ARRAY;
        copyParam.dstArray          = m_cuArraySrc;
        copyParam.srcMemoryType     = CU_MEMORYTYPE_DEVICE;
        copyParam.srcDevice         = (CUdeviceptr)cuDeviceptr_Input;
        copyParam.srcPitch          = m_uSrcArrayWidth * 4;
        copyParam.WidthInBytes      = m_uSrcArrayWidth * 4;
        copyParam.Height            = m_uSrcArrayHeight;
        CUDADRV_CHECK(cuMemcpy2D(&copyParam));
    }


    API_BOOL res = evaluate(m_cuTexObjectSrc, m_cuSurfObjectDst, inputRect, outputRect, pVSRSetting, pTHDRSetting, true, true);
;
    if (res == API_BOOL_SUCCESS)
    {
        // copy output from array to device
        CUDA_MEMCPY2D copyParam     = {};
        copyParam.dstMemoryType     = CU_MEMORYTYPE_DEVICE;
        copyParam.dstDevice         = (CUdeviceptr)cuDeviceptr_Output;
        copyParam.dstPitch          = m_uDstArrayWidth * 4;
        copyParam.srcMemoryType     = CU_MEMORYTYPE_ARRAY;
        copyParam.srcArray          = m_cuArrayDst;
        copyParam.WidthInBytes      = m_uDstArrayWidth * 4;
        copyParam.Height            = m_uDstArrayHeight;
        CUDADRV_CHECK(cuMemcpy2D(&copyParam));
    }

    return res;
}


API_BOOL cuda_api_impl::evaluate_hostptr(void* hostptr_Input, void* hostptr_Output, API_RECT inputRect, API_RECT outputRect, API_VSR_Setting* pVSRSetting, API_THDR_Setting* pTHDRSetting)
{
    if (!m_cuArraySrc || inputRect.right != m_uSrcArrayWidth || inputRect.bottom != m_uSrcArrayHeight)
    {
        if (m_cuArraySrc)
        {
            CUDADRV_CHECK(cuArrayDestroy(m_cuArraySrc));
            CUDADRV_CHECK(cuTexObjectDestroy(m_cuTexObjectSrc));
        }

        m_uSrcArrayWidth     = inputRect.right;
        m_uSrcArrayHeight    = inputRect.bottom;

        CUarray_format cuArrayFormat = CU_AD_FORMAT_UNSIGNED_INT8;
        CUDA_ARRAY_DESCRIPTOR cuArrayOutputDesc{
            m_uSrcArrayWidth,
            m_uSrcArrayHeight,
            cuArrayFormat,
            4
        };
        CUDADRV_CHECK(cuArrayCreate(&m_cuArraySrc, &cuArrayOutputDesc));
        {
            CUDA_RESOURCE_DESC resDescOutput;
            memset(&resDescOutput, 0, sizeof(CUDA_RESOURCE_DESC));
            resDescOutput.resType = CU_RESOURCE_TYPE_ARRAY;
            resDescOutput.res.array.hArray = m_cuArraySrc;

            CUDA_TEXTURE_DESC texDescOutput;
            memset(&texDescOutput, 0, sizeof(CUDA_TEXTURE_DESC));
            texDescOutput.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
            texDescOutput.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
            texDescOutput.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
            texDescOutput.filterMode = CU_TR_FILTER_MODE_LINEAR;
            texDescOutput.flags = CU_TRSF_NORMALIZED_COORDINATES;

            CUDADRV_CHECK(cuTexObjectCreate(&m_cuTexObjectSrc, &resDescOutput, &texDescOutput, nullptr));
        }
    }
    if (!m_cuArrayDst || outputRect.right != m_uDstArrayWidth || outputRect.bottom != m_uDstArrayHeight)
    {
        if (m_cuArrayDst)
        {
            CUDADRV_CHECK(cuArrayDestroy(m_cuArrayDst));
            CUDADRV_CHECK(cuTexObjectDestroy(m_cuSurfObjectDst));
        }

        m_uDstArrayWidth     = outputRect.right;
        m_uDstArrayHeight    = outputRect.bottom;

        CUarray_format cuArrayFormat = m_TrueHDRFeature ? CU_AD_FORMAT_UNORM_INT_101010_2 : CU_AD_FORMAT_UNSIGNED_INT8;
        CUDA_ARRAY_DESCRIPTOR cuArrayOutputDesc{
            m_uDstArrayWidth,
            m_uDstArrayHeight,
            cuArrayFormat,
            4
        };
        CUDADRV_CHECK(cuArrayCreate(&m_cuArrayDst, &cuArrayOutputDesc));
        {
            CUDA_RESOURCE_DESC resDescOutput;
            memset(&resDescOutput, 0, sizeof(CUDA_RESOURCE_DESC));
            resDescOutput.resType = CU_RESOURCE_TYPE_ARRAY;
            resDescOutput.res.array.hArray = m_cuArrayDst;

            CUDADRV_CHECK(cuSurfObjectCreate(&m_cuSurfObjectDst, &resDescOutput));
        }
    }

    {
        // copy input from host to array
        CUDA_MEMCPY2D copyParam     = {};
        copyParam.dstMemoryType     = CU_MEMORYTYPE_ARRAY;
        copyParam.dstArray          = m_cuArraySrc;
        copyParam.srcMemoryType     = CU_MEMORYTYPE_HOST;
        copyParam.srcHost           = hostptr_Input;
        copyParam.srcPitch          = m_uSrcArrayWidth * 4;
        copyParam.WidthInBytes      = m_uSrcArrayWidth * 4;
        copyParam.Height            = m_uSrcArrayHeight;
        CUDADRV_CHECK(cuMemcpy2D(&copyParam));
    }


    API_BOOL res = evaluate(m_cuTexObjectSrc, m_cuSurfObjectDst, inputRect, outputRect, pVSRSetting, pTHDRSetting, true, true);
;
    if (res == API_BOOL_SUCCESS)
    {
        // copy output from array to host
        CUDA_MEMCPY2D copyParam     = {};
        copyParam.dstMemoryType     = CU_MEMORYTYPE_HOST;
        copyParam.dstHost           = hostptr_Output;
        copyParam.dstPitch          = m_uDstArrayWidth * 4;
        copyParam.srcMemoryType     = CU_MEMORYTYPE_ARRAY;
        copyParam.srcArray          = m_cuArrayDst;
        copyParam.WidthInBytes      = m_uDstArrayWidth * 4;
        copyParam.Height            = m_uDstArrayHeight;
        CUDADRV_CHECK(cuMemcpy2D(&copyParam));
    }

    return res;
}



// Lazily (re)create an 8-bit RGBA CUDA array + texture object sized (nw x nh), reused across frames.
void cuda_api_impl::ensure_src_tex(CUarray& arr, CUtexObject& tex, size_t& cw, size_t& ch, size_t nw, size_t nh)
{
    if (arr && nw == cw && nh == ch) return;
    if (arr)
    {
        CUDADRV_CHECK(cuArrayDestroy(arr));
        CUDADRV_CHECK(cuTexObjectDestroy(tex));
    }
    cw = nw; ch = nh;
    CUDA_ARRAY_DESCRIPTOR desc{ cw, ch, CU_AD_FORMAT_UNSIGNED_INT8, 4 };
    CUDADRV_CHECK(cuArrayCreate(&arr, &desc));
    CUDA_RESOURCE_DESC resDesc; memset(&resDesc, 0, sizeof(CUDA_RESOURCE_DESC));
    resDesc.resType = CU_RESOURCE_TYPE_ARRAY;
    resDesc.res.array.hArray = arr;
    CUDA_TEXTURE_DESC texDesc; memset(&texDesc, 0, sizeof(CUDA_TEXTURE_DESC));
    texDesc.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
    texDesc.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
    texDesc.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
    texDesc.filterMode = CU_TR_FILTER_MODE_LINEAR;
    texDesc.flags = CU_TRSF_NORMALIZED_COORDINATES;
    CUDADRV_CHECK(cuTexObjectCreate(&tex, &resDesc, &texDesc, nullptr));
}

// Lazily (re)create a CUDA array + surface object sized (nw x nh) in the given format, reused across
// frames (fmt is UNSIGNED_INT8 for VSR output, UNORM_INT_101010_2 for TrueHDR output).
void cuda_api_impl::ensure_dst_surf(CUarray& arr, CUsurfObject& surf, size_t& cw, size_t& ch, size_t nw, size_t nh,
                                    CUarray_format fmt)
{
    if (arr && nw == cw && nh == ch) return;
    if (arr)
    {
        CUDADRV_CHECK(cuArrayDestroy(arr));
        CUDADRV_CHECK(cuSurfObjectDestroy(surf));
    }
    cw = nw; ch = nh;
    CUDA_ARRAY_DESCRIPTOR desc{ cw, ch, fmt, 4 };
    CUDADRV_CHECK(cuArrayCreate(&arr, &desc));
    CUDA_RESOURCE_DESC resDesc; memset(&resDesc, 0, sizeof(CUDA_RESOURCE_DESC));
    resDesc.resType = CU_RESOURCE_TYPE_ARRAY;
    resDesc.res.array.hArray = arr;
    CUDADRV_CHECK(cuSurfObjectCreate(&surf, &resDesc));
}

// VSR-only deviceptr eval: 8-bit RGBA in (inputRect) -> 8-bit RGBA out (outputRect). Uses its own
// persistent src/dst arrays so it can interleave with TrueHDR-only calls without reallocating.
API_BOOL cuda_api_impl::evaluate_vsr_deviceptr(void* cuDeviceptr_Input, void* cuDeviceptr_Output, API_RECT inputRect, API_RECT outputRect, API_VSR_Setting* pVSRSetting)
{
    if (!m_VSRFeature) return API_BOOL_FAIL;
    ensure_src_tex(m_vsrSrcArr, m_vsrSrcTex, m_vsrSrcW, m_vsrSrcH, inputRect.right, inputRect.bottom);
    ensure_dst_surf(m_vsrDstArr, m_vsrDstSurf, m_vsrDstW, m_vsrDstH, outputRect.right, outputRect.bottom,
                    CU_AD_FORMAT_UNSIGNED_INT8);
    {
        CUDA_MEMCPY2D copyParam = {};
        copyParam.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        copyParam.dstArray      = m_vsrSrcArr;
        copyParam.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        copyParam.srcDevice     = (CUdeviceptr)cuDeviceptr_Input;
        copyParam.srcPitch      = m_vsrSrcW * 4;
        copyParam.WidthInBytes  = m_vsrSrcW * 4;
        copyParam.Height        = m_vsrSrcH;
        CUDADRV_CHECK(cuMemcpy2D(&copyParam));
    }
    API_THDR_Setting dummyTHDR = { 100, 100, 50, 1000 };
    API_BOOL res = evaluate((uint64_t)m_vsrSrcTex, (uint64_t)m_vsrDstSurf, inputRect, outputRect,
                            pVSRSetting, &dummyTHDR, true, false);
    if (res == API_BOOL_SUCCESS)
    {
        CUDA_MEMCPY2D copyParam = {};
        copyParam.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        copyParam.dstDevice     = (CUdeviceptr)cuDeviceptr_Output;
        copyParam.dstPitch      = m_vsrDstW * 4;
        copyParam.srcMemoryType = CU_MEMORYTYPE_ARRAY;
        copyParam.srcArray      = m_vsrDstArr;
        copyParam.WidthInBytes  = m_vsrDstW * 4;
        copyParam.Height        = m_vsrDstH;
        CUDADRV_CHECK(cuMemcpy2D(&copyParam));
    }
    return res;
}

// TrueHDR-only deviceptr eval: 8-bit RGBA in -> packed 10:10:10:2 out, both at (outputRect == inputRect).
API_BOOL cuda_api_impl::evaluate_thdr_deviceptr(void* cuDeviceptr_Input, void* cuDeviceptr_Output, API_RECT inputRect, API_RECT outputRect, API_THDR_Setting* pTHDRSetting)
{
    if (!m_TrueHDRFeature) return API_BOOL_FAIL;
    ensure_src_tex(m_thdrSrcArr, m_thdrSrcTex, m_thdrSrcW, m_thdrSrcH, inputRect.right, inputRect.bottom);
    ensure_dst_surf(m_thdrDstArr, m_thdrDstSurf, m_thdrDstW, m_thdrDstH, outputRect.right, outputRect.bottom,
                    CU_AD_FORMAT_UNORM_INT_101010_2);
    {
        CUDA_MEMCPY2D copyParam = {};
        copyParam.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        copyParam.dstArray      = m_thdrSrcArr;
        copyParam.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        copyParam.srcDevice     = (CUdeviceptr)cuDeviceptr_Input;
        copyParam.srcPitch      = m_thdrSrcW * 4;
        copyParam.WidthInBytes  = m_thdrSrcW * 4;
        copyParam.Height        = m_thdrSrcH;
        CUDADRV_CHECK(cuMemcpy2D(&copyParam));
    }
    API_VSR_Setting dummyVSR = { 4 };
    API_BOOL res = evaluate((uint64_t)m_thdrSrcTex, (uint64_t)m_thdrDstSurf, inputRect, outputRect,
                            &dummyVSR, pTHDRSetting, false, true);
    if (res == API_BOOL_SUCCESS)
    {
        CUDA_MEMCPY2D copyParam = {};
        copyParam.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        copyParam.dstDevice     = (CUdeviceptr)cuDeviceptr_Output;
        copyParam.dstPitch      = m_thdrDstW * 4;
        copyParam.srcMemoryType = CU_MEMORYTYPE_ARRAY;
        copyParam.srcArray      = m_thdrDstArr;
        copyParam.WidthInBytes  = m_thdrDstW * 4;
        copyParam.Height        = m_thdrDstH;
        CUDADRV_CHECK(cuMemcpy2D(&copyParam));
    }
    return res;
}


void cuda_api_impl::shutdown()
{
    if (m_TrueHDRFeature)
    {
        CHECK_NGX(NVSDK_NGX_CUDA_ReleaseFeature(m_TrueHDRFeature));
    }
    if (m_VSRFeature)
    {
        CHECK_NGX(NVSDK_NGX_CUDA_ReleaseFeature(m_VSRFeature));
    }
    if (m_cuContext)
    {
        CUDADRV_CHECK(cuDevicePrimaryCtxRelease(m_cuDevice));
    }
    if (m_cuArrayMid)
    {
        CUDADRV_CHECK(cuArrayDestroy(m_cuArrayMid));
        CUDADRV_CHECK(cuTexObjectDestroy(m_cuTexObjectMid));
        CUDADRV_CHECK(cuSurfObjectDestroy(m_cuSurfObjectMid));
    }
    if (m_cuArraySrc)
    {
        CUDADRV_CHECK(cuArrayDestroy(m_cuArraySrc));
        CUDADRV_CHECK(cuTexObjectDestroy(m_cuTexObjectSrc));
    }
    if (m_cuArrayDst)
    {
        CUDADRV_CHECK(cuArrayDestroy(m_cuArrayDst));
        CUDADRV_CHECK(cuTexObjectDestroy(m_cuSurfObjectDst));
    }
    if (m_vsrSrcArr)
    {
        CUDADRV_CHECK(cuArrayDestroy(m_vsrSrcArr));
        CUDADRV_CHECK(cuTexObjectDestroy(m_vsrSrcTex));
    }
    if (m_vsrDstArr)
    {
        CUDADRV_CHECK(cuArrayDestroy(m_vsrDstArr));
        CUDADRV_CHECK(cuSurfObjectDestroy(m_vsrDstSurf));
    }
    if (m_thdrSrcArr)
    {
        CUDADRV_CHECK(cuArrayDestroy(m_thdrSrcArr));
        CUDADRV_CHECK(cuTexObjectDestroy(m_thdrSrcTex));
    }
    if (m_thdrDstArr)
    {
        CUDADRV_CHECK(cuArrayDestroy(m_thdrDstArr));
        CUDADRV_CHECK(cuSurfObjectDestroy(m_thdrDstSurf));
    }

    NVSDK_NGX_CUDA_DestroyParameters(m_ngxParameters);
    CHECK_NGX(NVSDK_NGX_CUDA_Shutdown());
}


////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

cuda_api_impl* p_cuda_api_impl = nullptr;

#if !defined(_WIN32)
__attribute__ ((visibility("default")))
#endif
API_BOOL  rtx_video_api_cuda_create(void* cuContext, void* cuStream, int iGpu, API_BOOL THDREnable, API_BOOL VSREnable)
{
    if (!p_cuda_api_impl)
    {
        p_cuda_api_impl = new cuda_api_impl;
    }
    if (!p_cuda_api_impl) return API_BOOL_FAIL;
    return p_cuda_api_impl->create(cuContext, cuStream, iGpu, THDREnable, VSREnable);
}

#if !defined(_WIN32)
__attribute__((visibility("default")))
#endif
API_BOOL rtx_video_api_cuda_evaluate(uint64_t cuTexObject_Input, uint64_t cuSurfObject_Output, API_RECT inputRect, API_RECT outputRect, API_VSR_Setting* pVSRSetting, API_THDR_Setting* pTHDRSetting)
{
    if (!p_cuda_api_impl) return API_BOOL_FAIL;
    return p_cuda_api_impl->evaluate(cuTexObject_Input, cuSurfObject_Output, inputRect, outputRect, pVSRSetting, pTHDRSetting, true, true);
}

#if !defined(_WIN32)
__attribute__((visibility("default")))
#endif
API_BOOL rtx_video_api_cuda_evaluate_deviceptr(void* cuDeviceptr_Input, void* cuDeviceptr_Output, API_RECT inputRect, API_RECT outputRect, API_VSR_Setting* pVSRSetting, API_THDR_Setting* pTHDRSetting)
{
    if (!p_cuda_api_impl) return API_BOOL_FAIL;
    return p_cuda_api_impl->evaluate_deviceptr(cuDeviceptr_Input, cuDeviceptr_Output, inputRect, outputRect, pVSRSetting, pTHDRSetting);
}

#if !defined(_WIN32)
__attribute__((visibility("default")))
#endif
extern "C" API_BOOL rtx_video_api_cuda_evaluate_vsr_deviceptr(void* cuDeviceptr_Input, void* cuDeviceptr_Output, API_RECT inputRect, API_RECT outputRect, API_VSR_Setting* pVSRSetting)
{
    if (!p_cuda_api_impl) return API_BOOL_FAIL;
    return p_cuda_api_impl->evaluate_vsr_deviceptr(cuDeviceptr_Input, cuDeviceptr_Output, inputRect, outputRect, pVSRSetting);
}

#if !defined(_WIN32)
__attribute__((visibility("default")))
#endif
extern "C" API_BOOL rtx_video_api_cuda_evaluate_thdr_deviceptr(void* cuDeviceptr_Input, void* cuDeviceptr_Output, API_RECT inputRect, API_RECT outputRect, API_THDR_Setting* pTHDRSetting)
{
    if (!p_cuda_api_impl) return API_BOOL_FAIL;
    return p_cuda_api_impl->evaluate_thdr_deviceptr(cuDeviceptr_Input, cuDeviceptr_Output, inputRect, outputRect, pTHDRSetting);
}

#if !defined(_WIN32)
__attribute__((visibility("default")))
#endif
API_BOOL rtx_video_api_cuda_evaluate_hostptr(void* hostptr_Input, void* hostptr_Output, API_RECT inputRect, API_RECT outputRect, API_VSR_Setting* pVSRSetting, API_THDR_Setting* pTHDRSetting)
{
    if (!p_cuda_api_impl) return API_BOOL_FAIL;
    return p_cuda_api_impl->evaluate_hostptr(hostptr_Input, hostptr_Output, inputRect, outputRect, pVSRSetting, pTHDRSetting);
}

#if !defined(_WIN32)
__attribute__((visibility("default")))
#endif
void rtx_video_api_cuda_shutdown()
{
    if (p_cuda_api_impl)
    {
        p_cuda_api_impl->shutdown();
        delete p_cuda_api_impl;
        p_cuda_api_impl = nullptr;
    }
}
