// Holds the NGX model/feature-dll path used by the SDK impl's APP_PATH.
// rtxv_set_model_path() must be called before rtx_video_api_cuda_create().
#include <string>

extern "C" const wchar_t* g_rtxv_model_path = L".";
static std::wstring g_store;

extern "C" void rtxv_set_model_path(const wchar_t* p)
{
    if (p && *p) { g_store.assign(p); g_rtxv_model_path = g_store.c_str(); }
    else         { g_rtxv_model_path = L"."; }
}
