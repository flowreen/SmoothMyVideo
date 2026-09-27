// Redirects GetForegroundWindow in the loaded Streamline modules (sl.*.dll) to the host's own
// window, so DLSS-G keeps generating frames while that window is not in the foreground.
// Shared by dlssg2f and smv-live. Include after the host defines LOG(fmt, ...).
#pragma once

#include <windows.h>
#include <Psapi.h>
#pragma comment(lib, "Psapi.lib")

inline HWND g_focusHwnd = nullptr;

inline HWND WINAPI focusStub(void) { return g_focusHwnd; }

inline int applyFocusRedirect()
{
    int count = 0;
    HMODULE snap[512];
    DWORD needed = 0;
    if (!EnumProcessModules(GetCurrentProcess(), snap, sizeof(snap), &needed))
        return 0;
    DWORD nMods = needed / sizeof(HMODULE);

    for (DWORD i = 0; i < nMods; i++)
    {
        wchar_t name[MAX_PATH];
        if (!GetModuleFileNameW(snap[i], name, MAX_PATH)) continue;
        const wchar_t* slash = wcsrchr(name, L'\\');
        const wchar_t* base = slash ? slash + 1 : name;
        if (_wcsnicmp(base, L"sl.", 3) != 0) continue;

        auto dos = (PIMAGE_DOS_HEADER)snap[i];
        auto nt  = (PIMAGE_NT_HEADERS)((BYTE*)snap[i] + dos->e_lfanew);
        auto& impDir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
        if (!impDir.VirtualAddress) continue;

        auto desc = (PIMAGE_IMPORT_DESCRIPTOR)((BYTE*)snap[i] + impDir.VirtualAddress);
        for (; desc->Name; desc++)
        {
            auto dllName = (const char*)((BYTE*)snap[i] + desc->Name);
            if (_stricmp(dllName, "user32.dll") != 0 && _stricmp(dllName, "USER32.dll") != 0
                && _stricmp(dllName, "USER32.DLL") != 0)
            {
                if (_strnicmp(dllName, "user32", 6) != 0) continue;
            }

            auto thunk = (PIMAGE_THUNK_DATA)((BYTE*)snap[i] + desc->FirstThunk);
            auto orig  = desc->OriginalFirstThunk
                       ? (PIMAGE_THUNK_DATA)((BYTE*)snap[i] + desc->OriginalFirstThunk)
                       : thunk;
            for (; orig->u1.AddressOfData; orig++, thunk++)
            {
                if (IMAGE_SNAP_BY_ORDINAL(orig->u1.Ordinal)) continue;
                auto hint = (PIMAGE_IMPORT_BY_NAME)((BYTE*)snap[i] + orig->u1.AddressOfData);
                if (strcmp(hint->Name, "GetForegroundWindow") != 0) continue;

                DWORD oldProt;
                if (VirtualProtect(&thunk->u1.Function, sizeof(thunk->u1.Function),
                                   PAGE_READWRITE, &oldProt))
                {
                    thunk->u1.Function = (ULONG_PTR)&focusStub;
                    VirtualProtect(&thunk->u1.Function, sizeof(thunk->u1.Function),
                                   oldProt, &oldProt);
                    count++;
                    LOG("  focus shim: patched %ls\n", base);
                }
            }
        }
    }
    return count;
}

// Call once after slInit with the host window. SMV_DLSSG_FOCUS_SHIM=0 turns it off.
inline void installFocusShim(HWND hwnd)
{
    g_focusHwnd = hwnd;
    wchar_t off[8]{};
    if (GetEnvironmentVariableW(L"SMV_DLSSG_FOCUS_SHIM", off, 8) && off[0] == L'0')
        LOG("focus shim off (SMV_DLSSG_FOCUS_SHIM=0)\n");
    else
        LOG("focus shim: %d import(s) redirected\n", applyFocusRedirect());
}
