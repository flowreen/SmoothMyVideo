// Custom utils.h for the SmoothMyVideo RTX Video bridge.
// Same as the SDK sample's utils.h, except APP_PATH is a settable global (an
// absolute path to the folder holding nvngx_vsr.dll / nvngx_truehdr.dll) instead
// of the hardcoded L".", so NGX does not depend on the process working directory.
#pragma once

#if defined(_WIN32)
#pragma comment( lib, "user32" )
#pragma comment( lib, "shell32" )
#pragma comment( lib, "Advapi32" )
#endif

#define APP_ID 0

extern "C" const wchar_t* g_rtxv_model_path;   // defined in rtxvideo_pathshim.cpp
#define APP_PATH g_rtxv_model_path

template <class T> inline void SafeRelease(T*& pT)
{
    if (pT != NULL)
    {
        pT->Release();
        pT = NULL;
    }
}
