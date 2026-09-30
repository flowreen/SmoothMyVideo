// smv-live - real-time frame generation overlay (Lossless-Scaling-style) built on the
// dlssg-two-frame host: capture a target window with Windows.Graphics.Capture, feed every
// captured frame through DLSS Frame Generation (Streamline sl.dlss_g) as if it were a game
// render, and let SL's own pacer present real + generated frames into a borderless topmost
// overlay window placed exactly over the target. The overlay is click-through, so the user
// keeps interacting with the real window while looking at the smoothed one.
//
// Modes:
//   smv-live.exe --live "title substring" [--gen N] [--vsync] [--no-clickthrough] [--diag S]
//       capture the first visible top-level window whose title contains the substring
//       (case-insensitive) and run the FG overlay over it. --gen N = generated frames per
//       captured frame (server backends 1..15 -> 2x..16x; dlssg/identity 1..5). Esc exits.
//       --diag S = GDI-dump the overlay region
//       every 40 ms for S seconds to bin\live_diag_NNN.png (verification tooling).
//   smv-live.exe --testsrc
//       verification source: a 960x540 window ("SMV Live TestSrc") whose red square jumps in
//       discrete 120 px steps at 10 fps. Interpolated frames land at ~60 px midpoints, which
//       cannot occur in the source - the smoking gun that FG frames reached the screen.
//
// Inherited from dlssg-two-frame (see BUILD.md there): the swap chain MUST carry
// FRAME_LATENCY_WAITABLE_OBJECT | ALLOW_TEARING or SL's pacer kills the first Present; a
// >500 ms input gap poisons the pacer and needs an off/on re-warm; HAGS must be ON.
//
// File map (ONE translation unit; this file holds the includes, the globals and LOG, then
// #includes the parts below in this order, then keeps the test source, the HDR probe, the
// offline render, the argument parsing, wmain and the resident hosts):
//   smv-live-host.inl      WIC / GDI and D3D helpers, Host (swap chain, overlay, present ring)
//   smv-live-capture.inl   WGC Capture (shared texture + fence, HDR pack shader)
//   smv-live-native.inl    the native RIFE host: NVRTC kernels, TrueHDR math, the RTX bridge,
//                          NativeRife (handoff, engine load, compute thread, resident state)
//   smv-live-pipe.inl      PipeServer (the present loop's message interface to the native host)
//   smv-live-loop.inl      window finding, the HUD, the diag thread, runLive (the present loops)

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <unknwn.h> // must precede winrt/base.h for classic-COM interop (.as<>)
#include <d3d12.h>
#include <d3d11_4.h> // ID3D11Device5/Fence/DeviceContext4 (shared-fence capture interop)
#include <dxgi1_4.h>
#include <dxgi1_6.h>     // IDXGIOutput6 / DXGI_OUTPUT_DESC1 for HDR display detection
#include <d3dcompiler.h> // D3DCompile for the dlssg HDR pack shader
#include "nr_host.h"     // the DLSS 5 Neural Rendering core, run inside this process
#include <dwmapi.h>
#include <timeapi.h>
#include <shlwapi.h>
#include <wincodec.h>
#include <wrl/client.h>
#include <io.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdarg>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <fstream>
#include <memory>
#include <wincrypt.h> // MD5 of the weights for the engine names (the host-side lookup)

#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif

#include <winrt/base.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Graphics.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>
#include <windows.graphics.capture.interop.h>
#include <windows.graphics.directx.direct3d11.interop.h>

#include <sl.h>
#include <sl_consts.h>
#include <sl_dlss_g.h>
#include <sl_reflex.h>
#include <sl_pcl.h>
#include <nvapi.h> // the GPU's free video memory as its driver counts it (nativeVideoMemory)

using Microsoft::WRL::ComPtr;
namespace wgc = winrt::Windows::Graphics::Capture;
namespace wgdx = winrt::Windows::Graphics::DirectX;

static uint32_t W = 1920;
static uint32_t H = 1080;

// Enhancement backend. BK_DLSSG = Streamline DLSS Frame Generation (SL interposes the swap
// chain, its pacer presents generated frames, needs RTX 40/50 + foreground). BK_IDENTITY =
// no Streamline at all: captured frames are presented directly (the SL-free base of the
// backend-agnostic server route). BK_SERVER = a model backend (rife, blend, gmfss, nvof, fruc,
// rifedrba, echo) run by the in-process native host (smv-live-native.inl); it returns N output
// frames per input and this host paces their presentation across the measured capture
// interval. Comments that name functions of engine/live_server.py (the python server this host
// replaced) record where the math was ported from.
enum Backend
{
    BK_DLSSG,
    BK_IDENTITY,
    BK_SERVER
};
static Backend g_backend = BK_DLSSG;
static std::wstring g_serverBackend; // BK_SERVER: the backend name (--backend)
static std::wstring g_modelLabel;    // user-facing model name for the loading message/HUD
                                     // (the GUI's effective-model label, e.g. "RIFE (DRBA)";
                                     // falls back to the raw backend name when absent)
static std::wstring g_modelNote;     // substitution note appended to the loading log line
                                     // (e.g. Smooth Motion having no live mode runs GMFSS)
static bool g_capRel = false;        // request the capture-release token (xq route)
static bool g_offline = false;       // --offline render mode (stdin frames in, stdout out)
static bool g_offlineGraph = false;  // SMV_OFFLINE_GRAPH=1
// IDENTICAL-PAIR PASSTHROUGH: two BYTE IDENTICAL frames in a row have no motion
// between them, so the model is skipped for that pair and every tween slot presents the real
// frame itself (a paused source or a static scene then stops shimmering). Exact equality only,
// never a tolerance and never a near-identical gate. SMV_NO_STATIC_HOLD=1 turns it off for
// measurement; there is no user-facing setting.
static bool g_staticHold = true;  // the passthrough is on (SMV_NO_STATIC_HOLD=1 clears it)
static uint64_t g_staticHeld = 0; // pairs held this session, reported as `static=N`
static std::wstring g_script;     // --script: any path inside the engine folder (see engineScript)

// The host finds the runtime DLLs, the engine cache, engine\onnx and the weights relative to
// the engine folder, which it takes as the folder of this path. The file name is never opened:
// callers (src/render, harnesses) pass engine/live_server.py, the old live server's path.
// Shipped layout: engine\live\smv-live.exe, so the default is one level up.
static std::wstring engineScript(const std::wstring& exeDir)
{
    return g_script.empty() ? exeDir + L"\\..\\live_server.py" : g_script;
}
static double g_flowScale = 1.0; // --scale F: live, the working size as this share of the
                                 // presented size (tests, the app's Custom)
static int g_dlssMode = 0;       // --scale MODE: NVIDIA's DLSS mode instead, 1 + its index in
                                 // kDlssModeName (0 = none: g_flowScale)
static int g_liveAutoFit = 0;    // live Auto: the mode the free video memory fits, picked once a session
                                 // before anything loads (liveAutoFitMemory); 0 = Auto's own pick
static int g_liveAutoFloor = 0;  // --auto-floor MODE: live Auto runs this mode or a smaller one (the app passes
                                 // the mode an exit 8 named, the GPU-time step); 0 = none
static const wchar_t* const kDlssModeName[6] = {L"auto", L"dlaa", L"quality", L"balanced", L"performance", L"ultra"};
// --scale's value: a DLSS mode name, or a share 0.01..1; false = neither
static bool parseLiveScale(const wchar_t* v)
{
    for (int m = 0; m < 6; m++)
        if (_wcsicmp(v, kDlssModeName[m]) == 0)
        {
            g_dlssMode = m + 1;
            g_flowScale = 1.0;
            return true;
        }
    g_dlssMode = 0;
    g_flowScale = _wtof(v);
    return g_flowScale >= 0.01 && g_flowScale <= 1.0;
}
static bool g_noHud = false;                   // --no-hud: suppress the on-screen fps/latency readout
static bool g_noHudLat = false;                // --no-hud-latency: keep the readout, drop the "~X ms behind" part
static bool g_noFillMouse = false;             // --no-fill-mouse: Fill leaves the mouse alone (FillMouse)
static double g_sharpen = 0.0;                 // --sharpen S: forwarded to the server (live RCAS)
static bool g_rtxVsr = false;                  // --rtx-vsr: forwarded to the server (VSR fill upscaler)
static int g_upscaleH = 0;                     // --upscale H: forwarded to the server (the app's "Upscale to"
                                               // height as the internal render size before the fit)
static bool g_restore = false;                 // --restore: forwarded to the server (Real-ESRGAN first on
                                               // every composed frame, the app's Restore checkbox)
static bool g_dlssnr = false;                  // --dlssnr: forwarded to the server (DLSS 5 Neural Rendering
                                               // once per captured frame, the app's NVIDIA DLSS 5 checkbox)
static double g_nrStructure = 1.0;             // --nr-structure F (0..2, the DLSS 5 Structure Intensity)
static double g_nrTone = 1.0;                  // --nr-tone F (0..2, the DLSS 5 Tone Intensity)
static int g_nrStyle = 1;                      // --nr-style N (DLSSNR.Style: 0 Default, 1 Natural, 2 Cinematic)
static int g_nrPasses = 1;                     // --nr-passes N (1..nr::kMaxPasses: DLSS 5 chained N times per frame)
static bool g_nrNative = false;                // DLSS 5 NR runs inside this exe (the native host's pass); the
                                               // server is then NOT told --dlssnr
static bool g_liveNrCuda = false;              // live DLSS 5: the native host runs it on the
                                               // model frame after Restore and the resize (nativeLiveNrInit)
static std::atomic<bool> g_liveNrReset{false}; // the overlay was hidden: the next evaluate starts a new stream
static bool g_nrAttempted = false;             // NGX was initialised in this process (even if it then
                                               // refused): leave through ExitProcess, see the end of run()
// Live TrueHDR (SDR window -> HDR out). Applied only when the HDR live mode is on (server
// backends; not DLSS-G).
static bool g_rtxHdr = false;            // --rtx-hdr
static wchar_t g_hdrColor[8] = L"vivid"; // --hdr-color vivid|rtx|raw
static int g_hdrSat = 0;                 // --hdr-saturation (SDK 0..200; drives the rtx mode)
static int g_hdrCon = 100;               // --hdr-contrast (SDK 0..200, 100 = neutral)
static double g_hdrVib = 0.0;            // --hdr-vibrance 0..1 (Dynamic Vibrance intensity)
static double g_hdrSb = 0.0;             // --hdr-satboost 0..1 (Dynamic Vibrance saturation)
static double g_sdrWhite = 240.0;        // SDR reference white (nits) of the target's monitor
// Server-backend generated-frame ceiling. gen+1 slots per pair IS the ceiling on how far
// above the source rate the adaptive grid can go (a pair that wants more grid points than
// it has slots degrades to the uniform ladder), so the slot count must follow the TARGET,
// never a fixed number: it is derived at runtime as ceil(target / measured source fps)
// plus a margin (serverGenForTarget below). There is deliberately NO fps constant anywhere
// in this file - a 10000fps target must fail on physics (present + inference cost), not on
// a literal. The only ceilings are MEMORY ones: each slot costs pitch*H bytes twice over
// (the output ring holds two sets of slots, the host writes one while the loop presents the
// other), so the slot count is capped here by a fraction of the adapter's dedicated VRAM,
// computed from the real slot size (slotBudget() returns that cap in SLOTS), and again by
// the free video memory when the host creates the ring (nativeRingFit).
static uint64_t g_vramBytes = 0; // adapter DedicatedVideoMemory, filled at adapter pick
static int slotBudget(size_t slotBytes)
{
    // a quarter of dedicated VRAM for the double-buffered output ring. Floor of 2 slots so a
    // tiny/unknown adapter still runs.
    const uint64_t budget = (g_vramBytes ? g_vramBytes : (2ull << 30)) / 4;
    if (!slotBytes)
        return 2;
    uint64_t n = budget / (2ull * slotBytes);
    if (n < 2)
        n = 2;
    if (n > 4096)
        n = 4096; // protocol sanity, not an fps limit (4096 slots at 24fps = 98k fps)
    return (int)n;
}
static bool g_noAdapt = false;     // --no-adapt: disable adaptive smoothness (benchmarks)
static bool g_noGpuFit = false;    // --no-gpu-fit (the GUI's "Fit to the GPU" off): no video memory fit on either route
                                   // (nativeVideoMemoryRoom), live holds its target (no throttle, no Auto step)
static bool g_genExplicit = false; // --gen given on the CLI: never derive it from the target
static int g_targetFps = 0;        // --target N: adaptive output target; 0 = the overlay
                                   // monitor's refresh rate
static bool g_monitor = false;     // --fit monitor: capture the target's WHOLE MONITOR and
                                   // present 1:1 (whole-screen smoothing; capture == overlay
                                   // size, so even DLSS-G works; the overlay is excluded from
                                   // its own capture via WDA_EXCLUDEFROMCAPTURE)
static bool g_fill = false;        // --fit fill: fullscreen overlay on the target's monitor,
                                   // content aspect-fit upscaled by the server (server backends
                                   // only; DLSS-G owns its pipeline and cannot be resized yet)
// HDR live mode. When the target's display has Windows HDR on and we run a server
// backend, capture FP16 scRGB, the server converts to BT.2020 PQ [0,1], and everything downstream
// (canvas, shm, present) stays 4 bytes/px as R10G10B10A2 on a PQ (G2084) swapchain. SDR is
// unaffected. SMV_LIVE_HDR overrides detection: 1 = force on, 0 = force off (test lever).
static bool g_hdr = false;
// Resident live host: --resident keeps this process alive
// between sessions with the TensorRT runtime, the two RIFE engines, the JIT cache and the
// kernel module loaded (about 300 MB of VRAM idle) and recreates only the per-session state
// (capture, overlay, execution contexts, buffers). Sessions start and stop over stdin lines
// ("start<TAB>arg<TAB>arg...", "stop", "quit"; residentMain at the end of the file). A session
// that touched Streamline, NGX (native DLSS 5) or the TrueHDR bridge still ends the process.
static bool g_resident = false;
static std::atomic<bool> g_stopReq{false}; // stdin "stop": the session loops exit cleanly
static std::atomic<bool> g_resizeReq{
    false};                          // the target resized during the model load: end with exit 4 (the app revives)
static bool g_sessionClean = false;  // runLive left through its full teardown block
static bool g_rtxUsed = false;       // the TrueHDR bridge (single-instance NGX) ran here
static bool g_teardownTrace = false; // SMV_LIVE_TEARDOWN_TRACE=1: one line per teardown stage

// Every log line goes to stderr; the resident offline host also
// writes it to the control pipe of the render it is serving (g_logPipe set for the item), so
// PROGRESS / OUTFRAMES and the rest reach the GUI exactly as they did through the one-shot exe's
// stderr. One mutex: the reader, writer and compute threads all log.
static HANDLE g_logPipe = nullptr;
static std::mutex g_logMutex;
static void logWrite(const char* fmt, ...)
{
    char stackBuf[4096];
    std::vector<char> heapBuf;
    char* buf = stackBuf;
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(stackBuf, sizeof stackBuf, fmt, ap);
    va_end(ap);
    if (n < 0)
        return;
    if (n >= (int)sizeof stackBuf)
    {
        heapBuf.resize((size_t)n + 1);
        buf = heapBuf.data();
        va_start(ap, fmt);
        vsnprintf(buf, (size_t)n + 1, fmt, ap);
        va_end(ap);
    }
    std::lock_guard<std::mutex> lk(g_logMutex);
    fwrite(buf, 1, (size_t)n, stderr);
    fflush(stderr);
    if (g_logPipe)
    {
        DWORD wr = 0;
        WriteFile(g_logPipe, buf, (DWORD)n, &wr, nullptr); // the client may be gone: ignored
    }
}
#define LOG(...) logWrite(__VA_ARGS__)
#define CHECK_HR(x)                                                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        HRESULT _hr = (x);                                                                                             \
        if (FAILED(_hr))                                                                                               \
        {                                                                                                              \
            LOG("smv-live FAIL hr=0x%08lx at %s:%d\n", _hr, __FILE__, __LINE__);                                       \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)
#define CHECK_SL(x)                                                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        sl::Result _r = (x);                                                                                           \
        if (_r != sl::Result::eOk)                                                                                     \
        {                                                                                                              \
            LOG("smv-live FAIL sl::Result=%d at %s:%d\n", (int)_r, __FILE__, __LINE__);                                \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

// Every window title and file path takes this road into a log line. The CRT's "%ls" converts
// through the C locale (this exe never calls setlocale), so vsnprintf returns -1 and logWrite
// above DROPS THE WHOLE LINE as soon as one wide character has no representation there: a
// window title carrying an en dash or a CJK character silently cost us the "target window:"
// line, and with it the app's hotkey-mode revive after a resize. So convert to
// UTF-8 and print "%s"; never "%ls" for text the user can influence.
static std::string wideToUtf8(const std::wstring& s)
{
    std::string o;
    const int n = WideCharToMultiByte(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0, nullptr, nullptr);
    if (n > 0)
    {
        o.resize((size_t)n);
        WideCharToMultiByte(CP_UTF8, 0, s.c_str(), (int)s.size(), o.data(), n, nullptr, nullptr);
    }
    return o;
}

static std::wstring utf8ToWide(const std::string& s)
{
    std::wstring o;
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
    if (n > 0)
    {
        o.resize((size_t)n);
        MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), o.data(), n);
    }
    return o;
}

// the window the running session captures, resolved by runLive for --live / --hwnd / --fg.
// The resident "live session ended" line carries it so the app can revive an exit 4 / 6
// session even when it never learned the hwnd from a "target window:" line.
static HWND g_targetHwnd = nullptr;

static bool g_verbose = false;

static void slLog(sl::LogType type, const char* msg)
{
    if (g_verbose)
    {
        LOG("[SL%d] %s", (int)type, msg);
        return;
    }
    if (type == sl::LogType::eInfo)
        return;
    static const char* benign[] = {
        "Invalid backbuffer resource extent",
        "reseting frame timer",
        "Frame rate over",
        "duplicated unique id",
        "invoked before slInit",
        "must be synchronized with the present thread",
    };
    for (const char* b : benign)
        if (strstr(msg, b))
            return;
    LOG("[SL%d] %s", (int)type, msg);
}

#include "sl_focus_shim.h"
#include "smv-live-host.inl"
#include "smv-live-capture.inl"
#include "smv-live-native.inl"
#include "smv-live-pipe.inl"
#include "smv-live-loop.inl"
// ---------------------------------------------------------------- test source

static int g_tsStep = 0;
static UINT g_tsInterval = 100; // ms per step (100 = 10 fps)
// --testsrc cycle: ramp the animation rate 10 -> 30 -> 60 -> 30 -> 10 fps, 12s per stage,
// looping - the adaptive-smoothness scenario source (base rises then falls; the target-grid
// resampler must hold the output pinned while the multiplier drifts 6x -> 2x -> 1x -> back).
static const UINT g_tsCycle[] = {100, 33, 16, 33};
static bool g_tsCycling = false;
static int g_tsCycleIdx = 0;
static ULONGLONG g_tsCycleTick = 0;

// --pan: a grey value-noise texture (8 px lattice, bilinear) moving right by 4 px per tick, a
// known-motion source: every pixel's motion current -> previous is exactly (-4, 0) px per tick
static bool g_tsPan = false;
static uint32_t g_tsTick = 0;
static uint32_t tsHash(int x, int y)
{
    uint32_t h = (uint32_t)x * 374761393u + (uint32_t)y * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return h ^ (h >> 16);
}
static void tsPanFill(uint32_t* px, int w, int h, int shift)
{
    for (int y = 0; y < h; y++)
    {
        const int cy = y >> 3, fy = y & 7;
        for (int x = 0; x < w; x++)
        {
            const int u = x - shift; // C++20: >> and & on a negative int = floor and remainder
            const int cx = u >> 3, fx = u & 7;
            const int a = (int)(tsHash(cx, cy) & 255) * (8 - fx) + (int)(tsHash(cx + 1, cy) & 255) * fx;
            const int b = (int)(tsHash(cx, cy + 1) & 255) * (8 - fx) + (int)(tsHash(cx + 1, cy + 1) & 255) * fx;
            const uint32_t g = 40u + (uint32_t)(((a * (8 - fy) + b * fy) * 175) / (64 * 255));
            px[(size_t)y * w + x] = g * 0x010101u;
        }
    }
}

// one test frame into dc: the dark background and the 80 px red square jumping in 120 px steps
// (interpolation midpoints, ~60 px offsets, cannot occur in the source), or the --pan texture
static void tsPaint(HDC dc, const RECT& cr)
{
    if (g_tsPan)
    {
        BITMAPINFO bi{};
        bi.bmiHeader.biSize = sizeof(bi.bmiHeader);
        bi.bmiHeader.biWidth = cr.right;
        bi.bmiHeader.biHeight = -cr.bottom; // top-down rows
        bi.bmiHeader.biPlanes = 1;
        bi.bmiHeader.biBitCount = 32;
        bi.bmiHeader.biCompression = BI_RGB;
        std::vector<uint32_t> px((size_t)cr.right * cr.bottom);
        tsPanFill(px.data(), cr.right, cr.bottom, (int)(g_tsTick * 4));
        SetDIBitsToDevice(dc, 0, 0, cr.right, cr.bottom, 0, 0, 0, cr.bottom, px.data(), &bi, DIB_RGB_COLORS);
        return;
    }
    HBRUSH bg = CreateSolidBrush(RGB(24, 24, 24));
    FillRect(dc, &cr, bg);
    DeleteObject(bg);
    RECT sq{40 + g_tsStep * 120, (cr.bottom - 80) / 2, 0, 0};
    sq.right = sq.left + 80;
    sq.bottom = sq.top + 80;
    HBRUSH red = CreateSolidBrush(RGB(230, 40, 40));
    FillRect(dc, &sq, red);
    DeleteObject(red);
}

// The parked test source is a layered popup that hands DWM its whole frame on every tick: GDI
// painting is clipped to the part of a window that lies on the desktop, so a parked painted window
// showed one pixel and the capture got a white frame. The frame follows the client size, so a
// harness may resize the window.
static bool g_tsLayered = false;
static void tsUpdateLayered(HWND h)
{
    RECT cr;
    GetClientRect(h, &cr);
    if (cr.right <= 0 || cr.bottom <= 0)
        return;
    HDC screen = GetDC(nullptr);
    HDC mdc = CreateCompatibleDC(screen);
    HBITMAP bmp = CreateCompatibleBitmap(screen, cr.right, cr.bottom);
    HGDIOBJ old = SelectObject(mdc, bmp);
    tsPaint(mdc, cr);
    POINT src{0, 0};
    SIZE sz{cr.right, cr.bottom};
    UpdateLayeredWindow(h, screen, nullptr, &sz, mdc, &src, 0, nullptr, ULW_OPAQUE);
    SelectObject(mdc, old);
    DeleteObject(bmp);
    DeleteDC(mdc);
    ReleaseDC(nullptr, screen);
}

static LRESULT CALLBACK testWndProc(HWND h, UINT m, WPARAM w, LPARAM l)
{
    switch (m)
    {
    case WM_TIMER:
        g_tsStep = (g_tsStep + 1) % 7;
        g_tsTick++;
        if (g_tsLayered)
            tsUpdateLayered(h);
        else
            InvalidateRect(h, nullptr, FALSE);
        if (g_tsCycling && GetTickCount64() - g_tsCycleTick > 12000)
        {
            g_tsCycleTick = GetTickCount64();
            g_tsCycleIdx = (g_tsCycleIdx + 1) % _countof(g_tsCycle);
            SetTimer(h, 1, g_tsCycle[g_tsCycleIdx], nullptr);
        }
        return 0;
    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC dc = BeginPaint(h, &ps);
        RECT cr;
        GetClientRect(h, &cr);
        HDC mdc = CreateCompatibleDC(dc);
        HBITMAP bmp = CreateCompatibleBitmap(dc, cr.right, cr.bottom);
        HGDIOBJ old = SelectObject(mdc, bmp);
        tsPaint(mdc, cr);
        BitBlt(dc, 0, 0, cr.right, cr.bottom, mdc, 0, 0, SRCCOPY);
        SelectObject(mdc, old);
        DeleteObject(bmp);
        DeleteDC(mdc);
        EndPaint(h, &ps);
        return 0;
    }
    case WM_ERASEBKGND:
        return 1;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(h, m, w, l);
}

static int runTestSrc(bool onScreen)
{
    WNDCLASSW wc{};
    wc.lpfnWndProc = testWndProc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"smvlivetestsrc";
    wc.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));
    RegisterClassW(&wc);
    RECT r{0, 0, 960, 540};
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
    // parked by default so a test run shows nothing on the user's screen: a borderless layered
    // popup with ONE pixel on the primary monitor's bottom-right corner (a window with no pixel on
    // the desktop is not composed, and a capture of it gets no frames), under the taskbar, never
    // activated (a new window took the keyboard focus from the user, p53) and so without a taskbar
    // button a click could minimize. --onscreen = the old framed window at (80, 80)
    HWND h = nullptr;
    if (onScreen)
        h = CreateWindowExW(0, L"smvlivetestsrc", L"SMV Live TestSrc", WS_OVERLAPPEDWINDOW | WS_VISIBLE, 80, 80,
                            r.right - r.left, r.bottom - r.top, nullptr, nullptr, wc.hInstance, nullptr);
    else
    {
        g_tsLayered = true;
        h = CreateWindowExW(WS_EX_LAYERED | WS_EX_NOACTIVATE, L"smvlivetestsrc", L"SMV Live TestSrc", WS_POPUP,
                            GetSystemMetrics(SM_CXSCREEN) - 1, GetSystemMetrics(SM_CYSCREEN) - 1, 960, 540, nullptr,
                            nullptr, wc.hInstance, nullptr);
        if (h)
        {
            tsUpdateLayered(h); // a layered window shows nothing before its first update
            ShowWindow(h, SW_SHOWNOACTIVATE);
        }
    }
    if (!h)
    {
        LOG("CreateWindow failed\n");
        return 1;
    }
    SetTimer(h, 1, g_tsInterval, nullptr);
    LOG("TESTSRC READY interval=%ums\n", g_tsInterval);
    // Drop the console window: it is a second visible window that confuses tests - it can win
    // the --fg foreground race (observed: an overlay targeted the console instead of this
    // window) and it pollutes --list. Redirected stderr keeps working; a console stderr goes
    // quiet, but READY above was the last line anyway.
    FreeConsole();
    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0))
    {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    return 0;
}

// ---------------------------------------------------------------- entry

// ---------------------------------------------------------------- HDR probe (diagnostic)

// IEEE half -> float. Needed because the probe reads a FP16 scRGB capture on the CPU.
static float halfToFloat(uint16_t h)
{
    uint32_t s = (h >> 15) & 1u, e = (h >> 10) & 0x1Fu, m = h & 0x3FFu, f;
    if (e == 0)
    {
        if (m == 0)
            f = s << 31;
        else
        {
            uint32_t ee = 127 - 15 + 1;
            while (!(m & 0x400u))
            {
                m <<= 1;
                ee--;
            }
            f = (s << 31) | (ee << 23) | ((m & 0x3FFu) << 13);
        }
    }
    else if (e == 31)
        f = (s << 31) | 0x7F800000u | (m << 13);
    else
        f = (s << 31) | ((e - 15 + 127) << 23) | (m << 13);
    float out;
    memcpy(&out, &f, 4);
    return out;
}

// HDR diagnostic: capture ANY window as FP16 scRGB and report what it actually contains, per
// channel, plus sample pixels. GDI dumps and screenshots cannot read HDR (they clip to 8-bit SDR),
// so this is the only way to see the real values - including by pointing it at OUR OWN overlay to
// check whether the HDR present is correct instead of guessing from how it looks.
static int runProbe(HWND target, int frames, const wchar_t* dumpPath)
{
    if (!IsWindow(target))
    {
        LOG("probe: hwnd 0x%p is not a window\n", (void*)target);
        return 1;
    }
    FILE* dump = nullptr;
    if (dumpPath && _wfopen_s(&dump, dumpPath, L"wb"))
    {
        LOG("probe: cannot open dump file\n");
        return 1;
    }
    g_hdr = true; // force the FP16 capture path regardless of display state
    ComPtr<IDXGIFactory2> fac;
    CHECK_HR(CreateDXGIFactory2(0, IID_PPV_ARGS(&fac)));
    ComPtr<IDXGIAdapter1> a;
    DXGI_ADAPTER_DESC1 ad{};
    for (UINT i = 0; fac->EnumAdapters1(i, &a) != DXGI_ERROR_NOT_FOUND; i++)
    {
        a->GetDesc1(&ad);
        if (ad.VendorId == 0x10DE)
            break;
        a.Reset();
    }
    if (!a)
    {
        LOG("probe: no NVIDIA adapter found\n");
        return 1;
    }

    static Capture cap;
    if (cap.init(target, a.Get()))
        return 1;
    wchar_t title[256]{};
    GetWindowTextW(target, title, 256);
    LOG("probe: hwnd=0x%p \"%s\" %ux%u FP16 scRGB (1.0 = 80 nits)\n", (void*)target, wideToUtf8(title).c_str(), cap.cw,
        cap.ch);

    // a target that stops presenting (a live session that ended, a closed window) ends the probe
    // instead of a wait that never returns
    const ULONGLONG kProbeIdleMs = 10000;
    ULONGLONG lastFrame = GetTickCount64();
    for (int n = 0; n < frames;)
    {
        WaitForSingleObject(cap.evt, 300);
        wgc::Direct3D11CaptureFrame frame{nullptr};
        winrt::com_ptr<ID3D11Texture2D> tex;
        int rc = cap.drainNewest(frame, tex);
        if (rc == -3) // nothing pending yet
        {
            const bool gone = !IsWindow(target);
            if (gone || GetTickCount64() - lastFrame > kProbeIdleMs)
            {
                LOG("probe: %s, stopping at frame %d of %d\n", gone ? "the window is gone" : "no frame for 10 s", n,
                    frames);
                if (dump)
                    fclose(dump);
                cap.stop();
                return 1;
            }
            continue;
        }
        lastFrame = GetTickCount64();
        if (rc)
        {
            LOG("probe: capture failed rc=%d\n", rc);
            cap.stop();
            return 1;
        }
        cap.copyCropped(cap.staging.Get(), tex.get());
        frame.Close();

        D3D11_MAPPED_SUBRESOURCE map{};
        if (FAILED(cap.ctx11->Map(cap.staging.Get(), 0, D3D11_MAP_READ, 0, &map)))
        {
            LOG("probe: Map failed\n");
            cap.stop();
            return 1;
        }
        double mn[4] = {1e30, 1e30, 1e30, 1e30}, mx[4] = {-1e30, -1e30, -1e30, -1e30}, sum[4] = {0, 0, 0, 0};
        for (uint32_t y = 0; y < cap.ch; y++)
        {
            const uint16_t* row = (const uint16_t*)((const uint8_t*)map.pData + (size_t)y * map.RowPitch);
            for (uint32_t x = 0; x < cap.cw; x++)
                for (int c = 0; c < 4; c++)
                {
                    float v = halfToFloat(row[x * 4 + c]);
                    if (v < mn[c])
                        mn[c] = v;
                    if (v > mx[c])
                        mx[c] = v;
                    sum[c] += v;
                }
        }
        const double npx = (double)cap.cw * cap.ch;
        if (!dump || n % 30 == 0)
            LOG("probe[%d] R[min %.4f avg %.4f max %.4f]  G[min %.4f avg %.4f max %.4f]  "
                "B[min %.4f avg %.4f max %.4f]  A[avg %.3f]\n",
                n, mn[0], sum[0] / npx, mx[0], mn[1], sum[1] / npx, mx[1], mn[2], sum[2] / npx, mx[2], sum[3] / npx);
        if (dump)
        {
            // tightly packed rgbaf16le rows, same layout as an ffmpeg rawvideo dump
            for (uint32_t y = 0; y < cap.ch; y++)
                fwrite((const uint8_t*)map.pData + (size_t)y * map.RowPitch, 1, (size_t)cap.cw * 8, dump);
        }
        else
        {
            // sample pixels: centre and the four quarter points, to expose channel swaps / constants
            const uint32_t px[5] = {cap.cw / 2, cap.cw / 4, cap.cw * 3 / 4, cap.cw / 4, cap.cw * 3 / 4};
            const uint32_t py[5] = {cap.ch / 2, cap.ch / 4, cap.ch / 4, cap.ch * 3 / 4, cap.ch * 3 / 4};
            for (int s = 0; s < 5; s++)
            {
                const uint16_t* row = (const uint16_t*)((const uint8_t*)map.pData + (size_t)py[s] * map.RowPitch);
                LOG("   px(%u,%u) = R %.4f  G %.4f  B %.4f  A %.3f\n", px[s], py[s], halfToFloat(row[px[s] * 4 + 0]),
                    halfToFloat(row[px[s] * 4 + 1]), halfToFloat(row[px[s] * 4 + 2]), halfToFloat(row[px[s] * 4 + 3]));
            }
        }
        cap.ctx11->Unmap(cap.staging.Get(), 0);
        n++;
    }
    if (dump)
        fclose(dump);
    cap.stop();
    return 0;
}

// ---- offline render through this host --------------------------------------------------------
// smv-live.exe --offline --w W --h H --multi N [--frames T] [--pixfmt rgb48le|rgb24]
//               [--python p] [--script s] [--progress-every K] [--pause-file P]
//               [--ifnet E --encode E --jit J --ph N --pw N --batch B]
//               [--nvof | --no-interp | --gmfss | --drba | --fruc] [--cache DIR] (--no-interp = no model, --multi ignored)
//               [--fps-ratio R] (--fps mode: R output frames per source frame, --multi ignored)
//               (no --ifnet = the host finds or builds the engines itself, lkOfflineRife)
//               the per-frame passes: [--out-w W --out-h H] [--restore] [--rtx-vsr]
//               [--dlssnr --nr-structure F --nr-tone F --nr-style N [--nr-delta PATH]] [--sharpen S] [--rtx-hdr ...]
// Raw frames in on stdin (W*H*bpp each, the decoder pipe the render already runs), raw frames out on
// stdout in the same pixfmt (the encoder pipe). Real frames go out from the host buffer they came
// in (bit exact); the N-1 tweens of every pair run pack-in, Head encode, batched IFNet, pack-out and
// a pinned download on the compute stream. stderr carries the log plus the PROGRESS k/total and
// OUTFRAMES n lines src/render forwards to the GUI.
struct OfflineIo
{
    size_t frameBytes = 0, frameBytesOut = 0;
    int nIn = 0, nOut = 0;
    std::vector<uint8_t*> hIn, hOut;
    std::vector<cudaEvent_t> h2dEv, outEv;
    std::vector<int> inState;    // 0 free, 1 filled by the reader, 2 consumed by compute
    std::vector<char> inWritten; // the real frame of that slot has been written out
    std::deque<int> outFree;
    struct Item
    {
        int kind;
        int idx;
    }; // 0 real frame from hIn[idx], 1 tween from hOut[idx], 2 end
    std::deque<Item> items;
    std::mutex m;
    std::condition_variable cv;
    bool readerEof = false, fail = false;
    uint64_t framesIn = 0, framesOut = 0, totalFrames = 0, lastPairs = 0;
    int progressEvery = 10;
    int multi = 2;
    bool frameUnits = false; // --no-interp: PROGRESS counts frames over --frames (render.py's NB)
    double fpsRatio = 0.0;   // --fps mode: pairs close on render.py's _pair_fracs grid, not per multi
    uint64_t fpsPairs = 0;
    uint64_t outBase = 0; // --resume-out: the banked output frames, so PROGRESS / OUTFRAMES count the whole render
    uint64_t skipIn = 0;  // --skip-in: input frames the reader discards before the first real one
    HANDLE hStdin = nullptr, hStdout = nullptr;
    // --thumb PATH: the GUI's render progress thumbnail, render.py's _live_preview.
    // About once a second the writer drops a small copy of the frame it is writing (the output
    // pixfmt kept, the TS side tonemaps and encodes it) and logs THUMB; --thumb-off FILE present =
    // the GUI's Hide, nothing is made. outFmt: 0 rgb24, 1 rgb48le, 2 x2rgb10le
    std::wstring thumbPath, thumbOff;
    int outW = 0, outH = 0, outFmt = 0;
    int64_t thumbLast = 0;

    void setFail(const char* why)
    {
        LOG("offline: %s\n", why);
        {
            std::lock_guard<std::mutex> lk(m);
            fail = true;
        }
        cv.notify_all();
    }
};

static bool offlineReadFrame(HANDLE h, uint8_t* dst, size_t n)
{
    size_t got = 0;
    while (got < n)
    {
        DWORD chunk = (DWORD)((n - got) > (1u << 24) ? (1u << 24) : (n - got));
        DWORD rd = 0;
        if (!ReadFile(h, dst + got, chunk, &rd, nullptr) || rd == 0)
            return false;
        got += rd;
    }
    return true;
}
static bool offlineWriteAll(HANDLE h, const uint8_t* src, size_t n)
{
    size_t done = 0;
    while (done < n)
    {
        DWORD chunk = (DWORD)((n - done) > (1u << 24) ? (1u << 24) : (n - done));
        DWORD wr = 0;
        if (!WriteFile(h, src + done, chunk, &wr, nullptr) || wr == 0)
            return false;
        done += wr;
    }
    return true;
}

static void offlineReader(OfflineIo* io)
{
    // a resumed VFR source: the decoder could not pre-skip (render_encode.decode_filters), so the
    // banked source frames arrive and are read into slot 0 and dropped (nothing uses it yet)
    for (uint64_t k = 0; k < io->skipIn; k++)
        if (!offlineReadFrame(io->hStdin, io->hIn[0], io->frameBytes))
        {
            io->setFail(
                "the source ended before the resume point (source changed?); delete the .part files next to the output and render fresh");
            return;
        }
    for (uint64_t i = 0;; i++)
    {
        const int slot = (int)(i % io->nIn);
        {
            std::unique_lock<std::mutex> lk(io->m);
            io->cv.wait(lk, [&] { return io->inWritten[slot] || io->fail; });
            if (io->fail)
                return;
        }
        // the slot's previous frame is written out; its H2D copy must have landed too
        if (cudaEventSynchronize(io->h2dEv[slot]) != cudaSuccess)
        {
            io->setFail("reader: H2D event sync failed");
            return;
        }
        const bool ok = offlineReadFrame(io->hStdin, io->hIn[slot], io->frameBytes);
        {
            std::lock_guard<std::mutex> lk(io->m);
            if (!ok)
            {
                io->readerEof = true;
            }
            else
            {
                io->inWritten[slot] = 0;
                io->inState[slot] = 1;
                io->framesIn++;
            }
        }
        io->cv.notify_all();
        if (!ok)
            return;
    }
}

// The thumbnail of one output frame (host bytes in the output pixfmt): at most 480 rows, each
// sample the mean of 2x2 taps spread over its k x k block (cheap on the writer thread: ~1.6M reads
// at 4K, where the full box mean would cost the encoder-bound render a few percent), written as
// "SMVT" + uint32 w, h, fmt + the samples, tmp then replace, then the THUMB line.
static void offlineThumb(OfflineIo* io, const uint8_t* src)
{
    if (io->thumbPath.empty())
        return;
    const int64_t now = nowQpc100();
    if (io->thumbLast && now - io->thumbLast < 10000000)
        return; // 1 s in 100 ns units
    io->thumbLast = now;
    if (!io->thumbOff.empty() && GetFileAttributesW(io->thumbOff.c_str()) != INVALID_FILE_ATTRIBUTES)
        return;
    const int ow = io->outW, oh = io->outH, fmt = io->outFmt;
    const int k = (oh + 479) / 480, tw = ow / k, th = oh / k;
    if (tw < 1 || th < 1)
        return;
    const int t = k < 2 ? 1 : 2;
    int off[2] = {k < 2 ? 0 : k / 4, k < 2 ? 0 : (3 * k) / 4};
    const size_t bpp = fmt == 0 ? 3 : (fmt == 1 ? 6 : 4);
    std::vector<uint8_t> buf(16 + (size_t)tw * th * bpp);
    const uint32_t head[3] = {(uint32_t)tw, (uint32_t)th, (uint32_t)fmt};
    memcpy(buf.data(), "SMVT", 4);
    memcpy(buf.data() + 4, head, sizeof(head));
    uint8_t* d = buf.data() + 16;
    const uint32_t n = (uint32_t)(t * t);
    for (int y = 0; y < th; y++)
        for (int x = 0; x < tw; x++)
        {
            uint32_t s0 = 0, s1 = 0, s2 = 0;
            for (int a = 0; a < t; a++)
                for (int b = 0; b < t; b++)
                {
                    const size_t p = (size_t)(y * k + off[a]) * ow + (size_t)(x * k + off[b]);
                    if (fmt == 0)
                    {
                        const uint8_t* q = src + 3 * p;
                        s0 += q[0];
                        s1 += q[1];
                        s2 += q[2];
                    }
                    else if (fmt == 1)
                    {
                        uint16_t q[3];
                        memcpy(q, src + 6 * p, 6);
                        s0 += q[0];
                        s1 += q[1];
                        s2 += q[2];
                    }
                    else
                    {
                        uint32_t u;
                        memcpy(&u, src + 4 * p, 4);
                        s0 += (u >> 20) & 1023;
                        s1 += (u >> 10) & 1023;
                        s2 += u & 1023;
                    }
                }
            s0 = (s0 + n / 2) / n;
            s1 = (s1 + n / 2) / n;
            s2 = (s2 + n / 2) / n;
            if (fmt == 0)
            {
                d[0] = (uint8_t)s0;
                d[1] = (uint8_t)s1;
                d[2] = (uint8_t)s2;
                d += 3;
            }
            else if (fmt == 1)
            {
                const uint16_t q[3] = {(uint16_t)s0, (uint16_t)s1, (uint16_t)s2};
                memcpy(d, q, 6);
                d += 6;
            }
            else
            {
                const uint32_t u = (s0 << 20) | (s1 << 10) | s2;
                memcpy(d, &u, 4);
                d += 4;
            }
        }
    const std::wstring tmp = io->thumbPath + L".tmp";
    HANDLE f = CreateFileW(tmp.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE)
        return; // a thumbnail never breaks the render
    DWORD wr = 0;
    const bool ok = WriteFile(f, buf.data(), (DWORD)buf.size(), &wr, nullptr) && wr == buf.size();
    CloseHandle(f);
    if (ok && MoveFileExW(tmp.c_str(), io->thumbPath.c_str(), MOVEFILE_REPLACE_EXISTING))
        LOG("THUMB\n");
}

static void offlineWriter(OfflineIo* io)
{
    uint64_t reals = 0;
    const uint64_t total = io->frameUnits ? io->totalFrames : (io->totalFrames ? io->totalFrames - 1 : 0);
    for (;;)
    {
        OfflineIo::Item it{};
        {
            std::unique_lock<std::mutex> lk(io->m);
            io->cv.wait(lk, [&] { return !io->items.empty() || io->fail; });
            if (io->fail)
                return;
            it = io->items.front();
            io->items.pop_front();
        }
        if (it.kind == 2)
            break;
        if (it.kind == 0)
        {
            if (!offlineWriteAll(io->hStdout, io->hIn[it.idx], io->frameBytes))
            {
                io->setFail("writer: real frame write failed");
                return;
            }
            offlineThumb(io, io->hIn[it.idx]); // kind 0 = same pixfmt, no passes: the output frame as is
            {
                std::lock_guard<std::mutex> lk(io->m);
                io->inWritten[it.idx] = 1;
                io->framesOut++;
            }
            io->cv.notify_all();
            reals++;
        }
        else
        {
            if (cudaEventSynchronize(io->outEv[it.idx]) != cudaSuccess)
            {
                io->setFail("writer: tween event sync failed");
                return;
            }
            if (!offlineWriteAll(io->hStdout, io->hOut[it.idx], io->frameBytesOut))
            {
                io->setFail("writer: tween write failed");
                return;
            }
            offlineThumb(io, io->hOut[it.idx]);
            {
                std::lock_guard<std::mutex> lk(io->m);
                io->outFree.push_back(it.idx);
                io->framesOut++;
            }
            io->cv.notify_all();
        }
        // PROGRESS in pairs: every (multi) output frames close one pair, whichever kind they are
        // (--no-interp: in frames, python's frames_loop unit; --fps: pair p closes once
        // ceil((p + 1) * ratio - 0.5) frames are out, and a jump over a multiple still reports)
        {
            const uint64_t outN = io->outBase + io->framesOut;
            uint64_t pairs = io->frameUnits ? outN : (outN ? (outN - 1) / (uint64_t)io->multi : 0);
            bool due = pairs % (uint64_t)(io->progressEvery > 0 ? io->progressEvery : 1) == 0;
            if (io->fpsRatio > 0.0)
            {
                while ((!total || io->fpsPairs < total) &&
                       (uint64_t)std::ceil((double)(io->fpsPairs + 1) * io->fpsRatio - 0.5) <= outN)
                    io->fpsPairs++;
                pairs = io->fpsPairs;
                due = io->progressEvery > 0 &&
                      pairs / (uint64_t)io->progressEvery != io->lastPairs / (uint64_t)io->progressEvery;
            }
            if (pairs && pairs != io->lastPairs && io->progressEvery > 0 && due)
                LOG("PROGRESS %llu/%llu\n", (unsigned long long)pairs, (unsigned long long)(total ? total : pairs));
            io->lastPairs = pairs;
        }
    }
    LOG("OUTFRAMES %llu\n", (unsigned long long)(io->outBase + io->framesOut));
}

// ---- DLSS 4.5 = engine\dlssg\dlssg2f.exe --server as this process's child ---------------------
// dlssg.py's DLSSG class in C++: raw RGBA8 frames at the padded size on the child's stdin (hdr10:
// R10G10B10A2 words of PQ BT.2020, --hdr10), after the first one the gen generated frames of every
// pair on its stdout; its stderr lines are forwarded to the log. A read that blocks over 10 s (or
// over 1 s while the GUI holds Pause) kills the child (the watchdog = python's _pause_watch), and a
// failed pair restarts it and re-primes with the pair's left frame, up to 4 times (python's
// _MAX_RESTARTS, 3 s apart).
struct DgChild
{
    std::wstring dir, pauseFile;
    int w = 0, h = 0, gen = 1;
    unsigned maxGen = 0;
    bool hdr10 = false; // every spawn asks for --hdr10 and its handshake must confirm it
    HANDLE hProc = nullptr, hIn = nullptr, hOut = nullptr, hErr = nullptr;
    std::thread errPump, watch;
    std::mutex m;                       // guards hProc against the watchdog
    std::atomic<uint64_t> recvSince{0}; // GetTickCount64 at a blocked read's start, 0 = none
    std::atomic<bool> stalled{false}, closing{false};
};

static const int kDgMaxRestarts = 4;

static void dgEnd(DgChild& c, bool kill)
{
    if (c.hIn)
    {
        CloseHandle(c.hIn);
        c.hIn = nullptr;
    } // EOF: the server exits on its own
    HANDLE p = nullptr;
    {
        std::lock_guard<std::mutex> lk(c.m);
        p = c.hProc;
    }
    if (p)
    {
        if (kill || WaitForSingleObject(p, 5000) != WAIT_OBJECT_0)
            TerminateProcess(p, 1);
        WaitForSingleObject(p, 5000);
    }
    if (c.errPump.joinable())
        c.errPump.join(); // ends at the child's stderr EOF
    if (c.hOut)
    {
        CloseHandle(c.hOut);
        c.hOut = nullptr;
    }
    if (c.hErr)
    {
        CloseHandle(c.hErr);
        c.hErr = nullptr;
    }
    {
        std::lock_guard<std::mutex> lk(c.m);
        if (c.hProc)
            CloseHandle(c.hProc);
        c.hProc = nullptr;
    }
}

// start the server and read its handshake; 0 = up, else the child's exit code (-1 = no process,
// -2 = up without the HDR10 mode it was asked for, ended)
static int dgSpawn(DgChild& c)
{
    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    HANDLE inR = nullptr, inW = nullptr, outR = nullptr, outW = nullptr, errR = nullptr, errW = nullptr;
    if (!CreatePipe(&inR, &inW, &sa, 1 << 20) || !CreatePipe(&outR, &outW, &sa, 1 << 20) ||
        !CreatePipe(&errR, &errW, &sa, 0))
    {
        LOG("dlss: pipe creation failed (%lu)\n", GetLastError());
        return -1;
    }
    SetHandleInformation(inW, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(outR, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(errR, HANDLE_FLAG_INHERIT, 0);
    // the child inherits exactly its three pipe ends (never a frame pipe of the resident host)
    HANDLE inherit[3] = {inR, outW, errW};
    SIZE_T attrSize = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &attrSize);
    std::vector<uint8_t> attrBuf(attrSize);
    auto attrs = (LPPROC_THREAD_ATTRIBUTE_LIST)attrBuf.data();
    STARTUPINFOEXW si{};
    si.StartupInfo.cb = sizeof(si);
    si.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    si.StartupInfo.hStdInput = inR;
    si.StartupInfo.hStdOutput = outW;
    si.StartupInfo.hStdError = errW;
    bool ok = InitializeProcThreadAttributeList(attrs, 1, 0, &attrSize) &&
              UpdateProcThreadAttribute(attrs, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherit, sizeof(inherit), nullptr,
                                        nullptr);
    si.lpAttributeList = attrs;
    const std::wstring exe = c.dir + L"\\dlssg2f.exe";
    std::wstring cmd = L"\"" + exe + L"\" --server " + std::to_wstring(c.w) + L" " + std::to_wstring(c.h) + L" --gen " +
                       std::to_wstring(c.gen) + (c.hdr10 ? L" --hdr10" : L"");
    PROCESS_INFORMATION pi{};
    ok = ok &&
         CreateProcessW(exe.c_str(), cmd.data(), nullptr, nullptr, TRUE,
                        EXTENDED_STARTUPINFO_PRESENT | CREATE_NO_WINDOW, nullptr, c.dir.c_str(), &si.StartupInfo, &pi);
    const DWORD err = GetLastError();
    DeleteProcThreadAttributeList(attrs);
    CloseHandle(inR);
    CloseHandle(outW);
    CloseHandle(errW);
    if (!ok)
    {
        CloseHandle(inW);
        CloseHandle(outR);
        CloseHandle(errR);
        LOG("dlss: cannot start %s (%lu)\n", wideToUtf8(exe).c_str(), err);
        return -1;
    }
    CloseHandle(pi.hThread);
    {
        std::lock_guard<std::mutex> lk(c.m);
        c.hProc = pi.hProcess;
    }
    c.hIn = inW;
    c.hOut = outR;
    c.hErr = errR;
    c.errPump = std::thread([h = errR]() {
        // the server's own log (the "DLSS-G ready: SL x.y.z" line, resets, errors), line by line
        std::string acc;
        char buf[1024];
        DWORD rd = 0;
        while (ReadFile(h, buf, sizeof(buf), &rd, nullptr) && rd)
        {
            acc.append(buf, rd);
            size_t nl;
            while ((nl = acc.find('\n')) != std::string::npos)
            {
                std::string ln = acc.substr(0, nl);
                if (!ln.empty() && ln.back() == '\r')
                    ln.pop_back();
                LOG("%s\n", ln.c_str());
                acc.erase(0, nl + 1);
            }
        }
        if (!acc.empty())
            LOG("%s\n", acc.c_str());
    });
    // handshake: one text line, printed only once the SDK is fully up
    std::string line;
    char ch = 0;
    DWORD rd = 0;
    while (line.size() < 256 && ReadFile(c.hOut, &ch, 1, &rd, nullptr) && rd == 1 && ch != '\n')
        line += ch;
    if (line.rfind("DLSSG READY", 0) == 0)
    {
        const size_t mp = line.find("max=");
        c.maxGen = mp == std::string::npos ? 0u : (unsigned)strtoul(line.c_str() + mp + 4, nullptr, 10);
        if (!c.hdr10 || line.find(" hdr10=1") != std::string::npos)
            return 0;
        dgEnd(c, true); // an older dlssg2f.exe ignores --hdr10 and would read the words as RGBA8
        return -2;
    }
    DWORD rc = 1;
    WaitForSingleObject(pi.hProcess, 5000);
    GetExitCodeProcess(pi.hProcess, &rc);
    dgEnd(c, true);
    return rc == STILL_ACTIVE ? 1 : (int)rc;
}

// every pipe transfer is watched: a stalled server also stops reading its stdin
static bool dgRead(DgChild& c, uint8_t* p, size_t n)
{
    c.recvSince = GetTickCount64();
    const bool ok = offlineReadFrame(c.hOut, p, n);
    c.recvSince = 0;
    return ok;
}
static bool dgWrite(DgChild& c, const uint8_t* p, size_t n)
{
    c.recvSince = GetTickCount64();
    const bool ok = offlineWriteAll(c.hIn, p, n);
    c.recvSince = 0;
    return ok;
}

// the watchdog (python's _pause_watch): a stuck read gets its server killed, so the pair's
// restart loop takes over
static void dgWatch(DgChild* c)
{
    while (!c->closing)
    {
        Sleep(300);
        const uint64_t t = c->recvSince;
        if (!t)
            continue;
        const uint64_t blocked = GetTickCount64() - t;
        const bool paused =
            !c->pauseFile.empty() && GetFileAttributesW(c->pauseFile.c_str()) != INVALID_FILE_ATTRIBUTES;
        if (blocked > 10000 || (paused && blocked > 1000))
        {
            c->stalled = blocked > 10000;
            std::lock_guard<std::mutex> lk(c->m);
            if (c->hProc)
                TerminateProcess(c->hProc, 1);
        }
    }
}

// one pair through the server (dlssg.py interpolate): prev once when the server has not seen it
// (primed = it has), then cur, then the gen frames into out. 0 = ok, 3 = the multiplier is beyond
// this GPU, 4 = the server stayed lost through its restarts (python's DLSSHostLost)
static int dgPair(DgChild& c, const uint8_t* prev, const uint8_t* cur, uint8_t* out, bool& primed)
{
    const size_t fb = (size_t)c.w * c.h * 4;
    for (int attempt = 0;; attempt++)
    {
        bool ok = true;
        if (!primed)
            ok = dgWrite(c, prev, fb);
        ok = ok && dgWrite(c, cur, fb);
        primed = ok;
        for (int j = 0; ok && j < c.gen; j++)
            ok = dgRead(c, out + (size_t)j * fb, fb);
        if (ok)
            return 0;
        primed = false;
        DWORD rc = STILL_ACTIVE;
        {
            std::lock_guard<std::mutex> lk(c.m);
            if (c.hProc && WaitForSingleObject(c.hProc, 2000) == WAIT_OBJECT_0)
                GetExitCodeProcess(c.hProc, &rc);
        }
        if (rc == 3)
        {
            LOG("this GPU does not support %dx DLSS multi-frame generation (RTX 50 series does up to 6x, "
                "RTX 40 series only 2x); pick a lower multiplier or GMFSS\n",
                c.gen + 1);
            return 3;
        }
        if (attempt >= kDgMaxRestarts)
            return 4;
        LOG("[dlss] frame-generation host stopped (%s); restarting it (attempt %d/%d). If a video with "
            "NVIDIA RTX Video enhancement is playing, close it now and the render will recover on its own.\n",
            c.stalled ? "stalled >10s, killed" : "exit or broken pipe", attempt + 1, kDgMaxRestarts);
        c.stalled = false;
        // honor a GUI Pause (retries stop until Resume), then let the GPU / driver settle
        while (!c.pauseFile.empty() && GetFileAttributesW(c.pauseFile.c_str()) != INVALID_FILE_ATTRIBUTES)
            Sleep(200);
        Sleep(3000);
        dgEnd(c, true);
        if (dgSpawn(c) != 0)
            return 4; // a server that will not even respawn is lost too
    }
}

// one --offline command line (the resident host parses one per item from its "start" line)
struct OfflineArgs
{
    int w = 0, h = 0, multi = 2;
    uint64_t frames = 0;
    bool fmt16 = true; // input pipe format (render.py's DEC_FMT)
    int out16 = -1;    // output pipe format, -1 = same as input
    int progressEvery = 10;
    std::wstring ifnetW, encodeW, jitW, pauseFile;
    int phArg = 0, pwArg = 0, batchArg = 0;
    bool resident = false;             // --resident: stay alive between renders (offlineResidentMain)
    std::wstring pipeName;             // --pipe NAME: the resident host's control pipe, \\.\pipe\NAME
    bool nvof = false;                 // --nvof: the NVIDIA Optical Flow model (no engines, no handoff)
    std::wstring cacheW;               // --cache DIR: the kernel cubin folder when no jit path names one
    int outW = 0, outH = 0;            // --out-w / --out-h: the output size (render.py's OUT_W x OUT_H), 0 = w x h
    int workW = 0, workH = 0;          // --work-w / --work-h: the working size (the DLSS mode x the output), 0 = none
    bool outX2 = false;                // --out-pixfmt x2rgb10le: the RTX HDR encode pipe (needs --rtx-hdr)
    std::wstring hdrStatsW;            // --hdr-stats PATH: the light statistics JSON render.py reads at the finalize
    bool hdrDv = false, hdrHp = false; // --hdr-dv / --hdr-hp: add the DV L1 / HDR10+ per-frame records
    bool echo = false;                 // --no-interp: no model, every frame a real frame (live's echo, noEngine)
    bool gmfss = false;                // --gmfss: the GMFSS model (live's five-engine chain, lkOfflineGmfss)
    bool drba = false;                 // --drba: RIFE with DRBA timing (live's lag-1 windows, drba_loop's grid)
    bool fruc = false;                 // --fruc: Nvidia Smooth Motion (live's nvoffruc bridge path, no engine)
    double fpsRatio = 0.0;             // --fps-ratio R: --fps mode, render.py's ratio (repr, so the same double)
    bool dlssg = false;                // --dlssg: DLSS 4.5, the dlssg2f.exe --server child (DgChild), 2x..6x
    // resume: render.py's _try_resume mapping. The decoder already skipped
    // to source frame P, so this run's pair i is the render's pair P + i (the --fps grid, the
    // closing slot); its first D outputs are banked already and are dropped BEFORE any pass (no
    // DLSS 5 history, no statistics record), exactly python's resume_active / skip / pend rule
    bool resumed = false;    // --resume-pair given
    uint64_t resumePair = 0; // --resume-pair P
    uint64_t resumeOut = 0;  // --resume-out C: output frames banked (PROGRESS / OUTFRAMES count the whole render)
    int resumeDrop = 0;      // --resume-drop D
    uint64_t skipIn = 0; // --skip-in N: input frames to discard first (a resumed VFR source, render.py's _PIPE_DISCARD)
    std::wstring
        hdrFramesW; // --hdr-frames PATH: one statistics line per output frame, appended + flushed (the resume prefix)
    std::wstring nrDeltaW;          // --nr-delta PATH: DLSS 5's per-pixel change, float32 (the preview's mask)
    int srcHdr = 0;                 // --src-hdr pq|hlg: the source is HDR video (1 PQ, 2 HLG), 0 = SDR
    std::wstring thumbW, thumbOffW; // --thumb PATH / --thumb-off FILE: the GUI's progress thumbnail (OfflineIo)
    std::wstring fitWork;           // --fit-work LIST: offline Auto's candidates, priced and answered (offlineFitWork)
};

// argv[first..] are the flags after --offline; 0 = parsed, else the exit code
static int parseOfflineArgs(int argc, wchar_t** argv, int first, OfflineArgs& oa)
{
    for (int i = first; i < argc; i++)
    {
        if (wcscmp(argv[i], L"--w") == 0 && i + 1 < argc)
            oa.w = _wtoi(argv[++i]);
        else if (wcscmp(argv[i], L"--h") == 0 && i + 1 < argc)
            oa.h = _wtoi(argv[++i]);
        else if (wcscmp(argv[i], L"--multi") == 0 && i + 1 < argc)
            oa.multi = _wtoi(argv[++i]);
        else if (wcscmp(argv[i], L"--frames") == 0 && i + 1 < argc)
            oa.frames = (uint64_t)_wtoi64(argv[++i]);
        else if (wcscmp(argv[i], L"--progress-every") == 0 && i + 1 < argc)
            oa.progressEvery = _wtoi(argv[++i]);
        else if (wcscmp(argv[i], L"--pixfmt") == 0 && i + 1 < argc)
        {
            ++i;
            if (wcscmp(argv[i], L"rgb48le") == 0)
                oa.fmt16 = true;
            else if (wcscmp(argv[i], L"rgb24") == 0)
                oa.fmt16 = false;
            else
            {
                LOG("offline: --pixfmt must be rgb48le or rgb24\n");
                return 1;
            }
        }
        else if (wcscmp(argv[i], L"--out-pixfmt") == 0 && i + 1 < argc)
        {
            ++i;
            if (wcscmp(argv[i], L"rgb48le") == 0)
                oa.out16 = 1;
            else if (wcscmp(argv[i], L"rgb24") == 0)
                oa.out16 = 0;
            else if (wcscmp(argv[i], L"x2rgb10le") == 0)
                oa.outX2 = true;
            else
            {
                LOG("offline: --out-pixfmt must be rgb48le, rgb24 or x2rgb10le\n");
                return 1;
            }
        }
        else if (wcscmp(argv[i], L"--python") == 0 && i + 1 < argc)
            ++i; // accepted and ignored: no python process is ever started
        else if (wcscmp(argv[i], L"--script") == 0 && i + 1 < argc)
            g_script = argv[++i];
        else if (wcscmp(argv[i], L"--scale") == 0 && i + 1 < argc)
            g_flowScale = _wtof(argv[++i]);
        // engine paths handed over (the harnesses; render.py lets the host find or build them)
        else if (wcscmp(argv[i], L"--ifnet") == 0 && i + 1 < argc)
            oa.ifnetW = argv[++i];
        else if (wcscmp(argv[i], L"--encode") == 0 && i + 1 < argc)
            oa.encodeW = argv[++i];
        else if (wcscmp(argv[i], L"--jit") == 0 && i + 1 < argc)
            oa.jitW = argv[++i];
        else if (wcscmp(argv[i], L"--ph") == 0 && i + 1 < argc)
            oa.phArg = _wtoi(argv[++i]);
        else if (wcscmp(argv[i], L"--pw") == 0 && i + 1 < argc)
            oa.pwArg = _wtoi(argv[++i]);
        else if (wcscmp(argv[i], L"--batch") == 0 && i + 1 < argc)
            oa.batchArg = _wtoi(argv[++i]);
        else if (wcscmp(argv[i], L"--pause-file") == 0 && i + 1 < argc)
            oa.pauseFile = argv[++i];
        else if (wcscmp(argv[i], L"--resident") == 0)
            oa.resident = true;
        else if (wcscmp(argv[i], L"--pipe") == 0 && i + 1 < argc)
            oa.pipeName = argv[++i];
        else if (wcscmp(argv[i], L"--nvof") == 0)
            oa.nvof = true;
        else if (wcscmp(argv[i], L"--no-interp") == 0)
            oa.echo = true;
        else if (wcscmp(argv[i], L"--gmfss") == 0)
            oa.gmfss = true;
        else if (wcscmp(argv[i], L"--drba") == 0)
            oa.drba = true;
        else if (wcscmp(argv[i], L"--fruc") == 0)
            oa.fruc = true;
        else if (wcscmp(argv[i], L"--dlssg") == 0)
            oa.dlssg = true;
        else if (wcscmp(argv[i], L"--fps-ratio") == 0 && i + 1 < argc)
            oa.fpsRatio = wcstod(argv[++i], nullptr);
        else if (wcscmp(argv[i], L"--cache") == 0 && i + 1 < argc)
            oa.cacheW = argv[++i];
        else if (wcscmp(argv[i], L"--resume-pair") == 0 && i + 1 < argc)
        {
            oa.resumePair = (uint64_t)_wtoi64(argv[++i]);
            oa.resumed = true;
        }
        else if (wcscmp(argv[i], L"--resume-out") == 0 && i + 1 < argc)
            oa.resumeOut = (uint64_t)_wtoi64(argv[++i]);
        else if (wcscmp(argv[i], L"--resume-drop") == 0 && i + 1 < argc)
            oa.resumeDrop = _wtoi(argv[++i]);
        else if (wcscmp(argv[i], L"--skip-in") == 0 && i + 1 < argc)
            oa.skipIn = (uint64_t)_wtoi64(argv[++i]);
        else if (wcscmp(argv[i], L"--hdr-frames") == 0 && i + 1 < argc)
            oa.hdrFramesW = argv[++i];
        // the passes in order: Restore, the resize, DLSS 5, FSR, RTX HDR, the model, the final resize
        else if (wcscmp(argv[i], L"--out-w") == 0 && i + 1 < argc)
            oa.outW = _wtoi(argv[++i]);
        else if (wcscmp(argv[i], L"--out-h") == 0 && i + 1 < argc)
            oa.outH = _wtoi(argv[++i]);
        else if (wcscmp(argv[i], L"--work-w") == 0 && i + 1 < argc)
            oa.workW = _wtoi(argv[++i]);
        else if (wcscmp(argv[i], L"--work-h") == 0 && i + 1 < argc)
            oa.workH = _wtoi(argv[++i]);
        else if (wcscmp(argv[i], L"--fit-work") == 0 && i + 1 < argc)
            oa.fitWork = argv[++i];
        else if (wcscmp(argv[i], L"--no-gpu-fit") == 0)
            g_noGpuFit = true;
        else if (wcscmp(argv[i], L"--sharpen") == 0 && i + 1 < argc)
            g_sharpen = _wtof(argv[++i]);
        else if (wcscmp(argv[i], L"--rtx-vsr") == 0)
            g_rtxVsr = true;
        else if (wcscmp(argv[i], L"--restore") == 0)
            g_restore = true;
        // RTX HDR: TrueHDR once per decoded frame after DLSS 5 and FSR, before the model; x2rgb10le out
        else if (wcscmp(argv[i], L"--rtx-hdr") == 0)
            g_rtxHdr = true;
        else if (wcscmp(argv[i], L"--hdr-color") == 0 && i + 1 < argc)
            wcsncpy_s(g_hdrColor, argv[++i], _TRUNCATE);
        else if (wcscmp(argv[i], L"--hdr-saturation") == 0 && i + 1 < argc)
            g_hdrSat = _wtoi(argv[++i]);
        else if (wcscmp(argv[i], L"--hdr-contrast") == 0 && i + 1 < argc)
            g_hdrCon = _wtoi(argv[++i]);
        else if (wcscmp(argv[i], L"--hdr-vibrance") == 0 && i + 1 < argc)
            g_hdrVib = _wtof(argv[++i]);
        else if (wcscmp(argv[i], L"--hdr-satboost") == 0 && i + 1 < argc)
            g_hdrSb = _wtof(argv[++i]);
        else if (wcscmp(argv[i], L"--hdr-stats") == 0 && i + 1 < argc)
            oa.hdrStatsW = argv[++i];
        else if (wcscmp(argv[i], L"--hdr-dv") == 0)
            oa.hdrDv = true;
        else if (wcscmp(argv[i], L"--hdr-hp") == 0)
            oa.hdrHp = true;
        // DLSS 5: once per decoded frame after Restore and the resize, before FSR and RTX HDR
        else if (wcscmp(argv[i], L"--dlssnr") == 0)
            g_dlssnr = true;
        else if (wcscmp(argv[i], L"--nr-structure") == 0 && i + 1 < argc)
            g_nrStructure = _wtof(argv[++i]);
        else if (wcscmp(argv[i], L"--nr-tone") == 0 && i + 1 < argc)
            g_nrTone = _wtof(argv[++i]);
        else if (wcscmp(argv[i], L"--nr-style") == 0 && i + 1 < argc)
            g_nrStyle = _wtoi(argv[++i]);
        else if (wcscmp(argv[i], L"--nr-passes") == 0 && i + 1 < argc)
            g_nrPasses = _wtoi(argv[++i]);
        else if (wcscmp(argv[i], L"--nr-delta") == 0 && i + 1 < argc)
            oa.nrDeltaW = argv[++i];
        // the source's HDR transfer from the probe: DLSS 5 then sees its SDR range, as live's HDR does
        else if (wcscmp(argv[i], L"--src-hdr") == 0 && i + 1 < argc)
        {
            const wchar_t* v = argv[++i];
            oa.srcHdr = wcscmp(v, L"pq") == 0 ? 1 : (wcscmp(v, L"hlg") == 0 ? 2 : -1);
            if (oa.srcHdr < 0)
            {
                LOG("offline: --src-hdr takes pq or hlg\n");
                return 1;
            }
        }
        else if (wcscmp(argv[i], L"--thumb") == 0 && i + 1 < argc)
            oa.thumbW = argv[++i];
        else if (wcscmp(argv[i], L"--thumb-off") == 0 && i + 1 < argc)
            oa.thumbOffW = argv[++i];
        else
        {
            LOG("offline: unknown argument %s\n", wideToUtf8(argv[i]).c_str());
            return 1;
        }
    }
    return 0;
}

static bool offlineArgsValid(const OfflineArgs& oa)
{
    if (oa.w < 16 || oa.h < 16 || (!oa.echo && oa.fpsRatio == 0.0 && (oa.multi < 2 || oa.multi > 65)))
    {
        LOG("offline: need --w --h (>= 16) and --multi 2..65 (or --no-interp or --fps-ratio)\n");
        return false;
    }
    if ((int)oa.echo + (int)oa.nvof + (int)oa.gmfss + (int)oa.drba + (int)oa.fruc + (int)oa.dlssg > 1)
    {
        LOG("offline: --no-interp, --nvof, --gmfss, --drba, --fruc and --dlssg exclude each other\n");
        return false;
    }
    if (oa.fpsRatio != 0.0 && (oa.echo || oa.nvof || oa.dlssg || !(oa.fpsRatio > 0.0 && oa.fpsRatio <= 1000.0)))
    {
        LOG("offline: --fps-ratio needs a model other than --nvof or --dlssg and 0 < R <= 1000\n");
        return false;
    }
    if (oa.dlssg && (oa.multi < 2 || oa.multi > 6))
    {
        LOG("offline: --dlssg generates 1..5 frames per pair: --multi 2..6\n");
        return false;
    }
    return true;
}

static int offlineResidentMain(const OfflineArgs& base);
static int runOfflineSession(const OfflineArgs& oa, HANDLE hIn, HANDLE hOut, bool namedPipes);

// --fit-work WW:WH:DW:DH,...: offline Auto's working size by the free video memory, asked before the render plans its
// decode (a working size below the source is folded into the decode). The candidates come in order, Auto's pick
// first, each a working size and the decode it needs. The price per model, in MiB (the process's own memory over x2
// renders at 854x480 and 1920x1080 with the working size at the decode, a quarter on top): RIFE and Frame Blend
// nativeOfflineRifeMiB (the IFNet at the decode), DRBA 310 + 1028 a padded megapixel, GMFSS 462 + 2446, Smooth Motion
// 461 + 182, NVIDIA Optical Flow 233 + 179, no interpolation 225 + 41; the output's buffers beyond the working size, 25
// a megapixel (1920x1080 and 3840x2160 outputs of a 960x540 render); the passes around the model
// (nativeOfflineEffectsMiB). Prints `OFFLINE FIT i ROOM NEED0 NEEDi` in MiB: the first candidate that fits
// nativeVideoMemoryRoom(), the last when none does, 0 without a room figure. No render and no CUDA.
static int offlineFitWork(const OfflineArgs& oa)
{
    struct Cand
    {
        int ww, wh, dw, dh;
    };
    std::vector<Cand> cands;
    for (const wchar_t* p = oa.fitWork.c_str(); p && *p;)
    {
        Cand k{};
        if (swscanf_s(p, L"%d:%d:%d:%d", &k.ww, &k.wh, &k.dw, &k.dh) != 4 || k.ww < 16 || k.wh < 16 || k.dw < 16 ||
            k.dh < 16)
        {
            LOG("offline: --fit-work takes WW:WH:DW:DH,... (each candidate's working size and decode)\n");
            return 1;
        }
        cands.push_back(k);
        p = wcschr(p, L',');
        if (p)
            ++p;
    }
    if (cands.empty())
    {
        LOG("offline: --fit-work names no candidate\n");
        return 1;
    }
    const bool two = lkEnv("SMV_RIFE_TWO_DOMAIN") != "0";
    auto needMiB = [&](const Cand& k) {
        const double work = padMp64(k.ww, k.wh);
        double model;
        if (oa.gmfss)
            model = 462.0 + 2446.0 * work;
        else if (oa.fruc)
            model = 461.0 + 182.0 * work;
        else if (oa.nvof)
            model = 233.0 + 179.0 * work;
        else if (oa.echo)
            model = 225.0 + 41.0 * work;
        else if (oa.drba)
            model = 310.0 + 1028.0 * (std::max)(work, two ? padMp64(k.dw, k.dh) : work);
        else
            model = nativeOfflineRifeMiB(two ? k.dw : k.ww, two ? k.dh : k.wh, k.ww, k.wh);
        const int ow = oa.outW ? oa.outW : k.ww, oh = oa.outH ? oa.outH : k.wh;
        const double outMore = (std::max)(0.0, ((double)ow * oh - (double)k.ww * k.wh) / 1e6);
        return 1.25 * (model + 25.0 * outMore) + nativeOfflineEffectsMiB(k.ww, k.wh, k.dw, k.dh);
    };
    g_vramNoCuda = true;
    const uint64_t roomB = nativeVideoMemoryRoom();
    const double room = roomB > 1 ? roomB / 1048576.0 : 0.0;
    size_t pick = 0;
    if (roomB)
    {
        pick = cands.size() - 1;
        for (size_t i = 0; i < cands.size(); i++)
            if (needMiB(cands[i]) <= room)
            {
                pick = i;
                break;
            }
    }
    printf("OFFLINE FIT %zu %.0f %.0f %.0f\n", pick, room, needMiB(cands[0]), needMiB(cands[pick]));
    fflush(stdout);
    return 0;
}

static int runOffline(int argc, wchar_t** argv)
{
    OfflineArgs oa;
    const int prc = parseOfflineArgs(argc, argv, 2, oa);
    if (prc)
        return prc;
    if (!oa.fitWork.empty())
        return offlineFitWork(oa);
    if (oa.resident)
    {
        if (oa.pipeName.empty())
        {
            LOG("offline: --resident needs --pipe NAME\n");
            return 1;
        }
        return offlineResidentMain(oa);
    }
    if (!offlineArgsValid(oa))
        return 1;
    return runOfflineSession(oa, GetStdHandle(STD_INPUT_HANDLE), GetStdHandle(STD_OUTPUT_HANDLE), false);
}

// one render: frames in on hIn, out on hOut (the process stdio, or the resident host's frame
// pipes; namedPipes = flush hOut before closing it, a named pipe discards unread data otherwise)
static int runOfflineSession(const OfflineArgs& oa, HANDLE hIn, HANDLE hOut, bool namedPipes)
{
    int w = oa.w, h = oa.h; // not const: kernel argument arrays take their addresses as void*
    // The passes at the working size (--work-w / --work-h, the DLSS mode x the output): the
    // decoded frame (srcW x srcH) goes through Restore and the resize to the working size first, the
    // model and DLSS 5 run at it (w x h from here on), the emit's final resize takes it to the output.
    // Without the flags the decoded size is the working size and Restore runs in the emit.
    int srcW = w, srcH = h;
    bool nvPre = false;
    if (oa.workW > 0 && oa.workH > 0)
    {
        const int ow = oa.outW > 0 ? oa.outW : w, oh = oa.outH > 0 ? oa.outH : h;
        if ((long long)oa.workW * oa.workH > 3840LL * 2160)
        {
            LOG("offline: the working size %dx%d is above 3840x2160, the interpolation's reach\n", oa.workW, oa.workH);
            return 1;
        }
        if (oa.workW < w || oa.workH < h || oa.workW > ow || oa.workH > oh)
        {
            LOG("offline: the working size %dx%d must lie between the decoded %dx%d and the output %dx%d (a "
                "downscale folds into the decode)\n",
                oa.workW, oa.workH, w, h, ow, oh);
            return 1;
        }
        if (g_restore || oa.workW != w || oa.workH != h)
        {
            nvPre = true;
            w = oa.workW;
            h = oa.workH;
            LOG("offline: Restore and the resize on the decoded %dx%d first, DLSS 5 and the "
                "interpolation at %dx%d%s\n",
                srcW, srcH, w, h, (w != ow || h != oh) ? ", then the final resize" : "");
        }
    }
    // --fps mode: render_loops.fps_loop / drba_loop at render.py's ratio,
    // every output an interior slot of _pair_fracs, no real frame passes through; one tween
    // per enqueue (python's per-tween inference), so the RIFE class is the unbatched x2
    const bool fpsMode = oa.fpsRatio > 0.0;
    const double ratio = oa.fpsRatio;
    const int multi = oa.echo ? 1 : (fpsMode ? 2 : oa.multi);
    const uint64_t frames = oa.frames;
    const bool fmt16 = oa.fmt16;
    const int out16 = oa.out16;
    const int progressEvery = oa.progressEvery;
    const std::wstring& ifnetW = oa.ifnetW;
    const std::wstring& encodeW = oa.encodeW;
    const std::wstring& jitW = oa.jitW;
    const std::wstring& pauseFile = oa.pauseFile;
    const int phArg = oa.phArg, pwArg = oa.pwArg, batchArg = oa.batchArg;
    g_offline = true;
    g_hdr = false;
    W = (uint32_t)w;
    H = (uint32_t)h;
    if (g_flowScale < 0.01 || g_flowScale > 1.0)
        g_flowScale = 1.0;

    wchar_t exePath[MAX_PATH];
    GetModuleFileNameW(nullptr, exePath, MAX_PATH);
    std::wstring exeDir(exePath);
    exeDir.resize(exeDir.find_last_of(L'\\'));
    const std::wstring script = engineScript(exeDir);

    NativeRife nr;
    if (!nativeLoadDlls(script))
    {
        LOG("offline: runtime DLLs unavailable\n");
        return 2;
    }
    // RIFE: the IFNet + Head engines at the true /64 pad of the source size (out = capture =
    // source, no fit, no effects), found or built by the host (lkOfflineRife) unless a harness
    // hands paths over. The nvof model needs neither: the source size IS the model
    // size (the Optical Flow engine takes any size from 32x32, no pad), one tween at a time.
    OfflineEngines oe;
    if (oa.gmfss)
    {
        // GMFSS: the five engines at the /64 pad of the source, found or built by
        // the host; one tween at a time like nvof (live's chain has no timestep batch)
        if (!lkOfflineGmfss(script, w, h, nr))
            return 2;
        nr.w = w;
        nr.h = h;
        nr.cw = w;
        nr.ch = h;
        nr.dw = w;
        nr.dh = h;
        nr.batchMax = 1;
    }
    else if (oa.fruc)
    {
        // Smooth Motion: live's bridge path at the /64 pad of the source (render.py
        // sizes NvOFFRUC to pw x ph the same way), true BGRA from the (R, G, B) planes
        if (!lkOfflineFruc(script, nr.frucDir))
            return 2;
        nr.fruc = true;
        nr.planesRgb = true;
        nr.ph = (h + 63) / 64 * 64;
        nr.pw = (w + 63) / 64 * 64;
        nr.w = w;
        nr.h = h;
        nr.cw = w;
        nr.ch = h;
        nr.dw = w;
        nr.dh = h;
        nr.batchMax = 1;
    }
    else if (oa.dlssg)
    {
        // DLSS 4.5: the server sized to the /64 pad of the source (render.py's pw x ph,
        // dlssg.py sends the padded frame), RGBA8 from the (R, G, B) planes, no engine
        nr.dlssg = true;
        nr.planesRgb = true;
        nr.ph = (h + 63) / 64 * 64;
        nr.pw = (w + 63) / 64 * 64;
        nr.w = w;
        nr.h = h;
        nr.cw = w;
        nr.ch = h;
        nr.dw = w;
        nr.dh = h;
        nr.batchMax = 1;
    }
    else if (!oa.nvof && !oa.echo && ifnetW.empty())
    {
        // RIFE's two domains: the IFNet finds the motion on the finished picture in SDR sRGB at the
        // decoded size (the PQ encoding and an enlarge cost RIFE its motion, measured on an exact pan)
        // and its last step, the warp + blend, runs on the pictures at the working size.
        // SMV_RIFE_TWO_DOMAIN=0: the IFNet reads the pictures. DRBA: the unbatched IFNet (one tween per
        // enqueue, its own DRM timestep map) plus the block0 flow engine, at the same frame
        const bool two = lkEnv("SMV_RIFE_TWO_DOMAIN") != "0";
        if (!lkOfflineRife(script, two ? srcW : w, two ? srcH : h, oa.drba ? 2 : multi, oe,
                           nativeOfflineNeed(two ? srcW : w, two ? srcH : h, w, h, srcW, srcH)))
            return 2;
        if (oa.drba)
        {
            if (!lkOfflineBlock0(script, two ? srcW : w, two ? srcH : h, nr.block0Path, nr.block0Jit))
                return 2;
            nr.drba = true;
        }
        nr.ifnetPath = oe.ifnet;
        nr.encodePath = oe.encode;
        nr.jitPath = oe.jit;
        nr.ph = oe.ph;
        nr.pw = oe.pw;
        if (two)
        {
            nr.mph = oe.ph;
            nr.mpw = oe.pw;
            nr.mw = srcW;
            nr.mh = srcH;
            if (w != srcW || h != srcH)
            {
                nr.ph = (h + 63) / 64 * 64; // the pictures at the working size
                nr.pw = (w + 63) / 64 * 64;
            }
        }
        nr.w = w;
        nr.h = h;
        nr.cw = w;
        nr.ch = h;
        nr.dw = w;
        nr.dh = h;
        nr.batchMax = oe.batch;
    }
    else if (oa.nvof || oa.echo)
    {
        // --no-interp = live's echo: the no-engine mode, the same geometry
        nr.nvof = oa.nvof;
        nr.noEngine = oa.echo;
        nr.planesRgb = true; // k_packInRaw8 / 16 pack the decoder's rgb as (R, G, B)
        nr.ph = h;
        nr.pw = w;
        nr.w = w;
        nr.h = h;
        nr.cw = w;
        nr.ch = h;
        nr.dw = w;
        nr.dh = h;
        nr.batchMax = 1;
    }
    else
    {
        nr.ifnetPath = wideToUtf8(ifnetW);
        nr.encodePath = wideToUtf8(encodeW);
        nr.jitPath = wideToUtf8(jitW);
        nr.ph = phArg;
        nr.pw = pwArg;
        nr.w = w;
        nr.h = h;
        nr.cw = w;
        nr.ch = h;
        nr.dw = w;
        nr.dh = h;
        nr.batchMax = batchArg < 1 ? 1 : batchArg;
        if (nr.ph < 16 || nr.pw < 16 || nr.encodePath.empty())
        {
            LOG("offline: --ifnet needs --encode, --ph and --pw\n");
            return 1;
        }
    }
    nr.sw = srcW;
    nr.sh = srcH; // Restore and the first resize read the decoded frame's size
    nr.nvPre = nvPre;
    // the per-frame passes: render.py's output size (only an enlarge: a
    // downscale folds into the decode), Sharpen, RTX VSR and Restore, on every output frame
    const int outW = oa.outW > 0 ? oa.outW : w, outH = oa.outH > 0 ? oa.outH : h;
    if (outW < w || outH < h)
    {
        LOG("offline: --out-w / --out-h below the working size (a downscale folds into the decode)\n");
        return 1;
    }
    nr.dw = outW;
    nr.dh = outH;
    // RTX HDR needs the x2rgb10le encode pipe and the other way round
    if (g_rtxHdr != oa.outX2)
    {
        LOG("offline: --rtx-hdr and --out-pixfmt x2rgb10le go together\n");
        return 1;
    }
    // (every frame goes through the passes; the raw-copy real frames would be source-size)
    const bool passes = g_sharpen > 0.0 || g_restore || outW != w || outH != h || g_rtxHdr || g_dlssnr || nvPre;
    if (passes && !nativeConfigHdr(nr))
        return 2; // the sharpen strength and RTX VSR's availability (g_hdr is off)
    ThdrAcc thdrAcc;
    if (g_rtxHdr)
    {
        // render.py's RTXVideo settings: MaxLuminance HDR_NITS 1000, MiddleGray 50, the
        // SDK Contrast / Saturation 0..200, the colour mode and Dynamic Vibrance 0..1
        if (!rtxBridgeLoad())
            return 2;
        nr.rtxHdr = true;
        nr.thdr.Contrast = (uint32_t)(g_hdrCon < 0 ? 0 : (g_hdrCon > 200 ? 200 : g_hdrCon));
        nr.thdr.Saturation = (uint32_t)(g_hdrSat < 0 ? 0 : (g_hdrSat > 200 ? 200 : g_hdrSat));
        nr.thdr.MiddleGray = 50;
        nr.thdr.MaxLuminance = 1000;
        nr.rtxMode = !wcscmp(g_hdrColor, L"rtx") ? 1 : (!wcscmp(g_hdrColor, L"raw") ? 2 : 0);
        nr.rtxVib = (float)(g_hdrVib < 0.0 ? 0.0 : (g_hdrVib > 1.0 ? 1.0 : g_hdrVib));
        nr.rtxSb = (float)(g_hdrSb < 0.0 ? 0.0 : (g_hdrSb > 1.0 ? 1.0 : g_hdrSb));
        thdrAcc.raw = nr.rtxMode == 2;
        thdrAcc.wantL1 = oa.hdrDv;
        thdrAcc.wantHp = oa.hdrHp;
        nr.thdrAcc = &thdrAcc;
        // appended: a resumed render's file already holds the banked prefix's lines (render.py
        // rewrites it to exactly those before the start); a failed open only costs the resume
        if (!oa.hdrFramesW.empty() && (_wfopen_s(&thdrAcc.lines, oa.hdrFramesW.c_str(), L"ab") || !thdrAcc.lines))
        {
            thdrAcc.lines = nullptr;
            LOG("offline: cannot open the per-frame statistics %s (a resume will skip DV / HDR10+)\n",
                wideToUtf8(oa.hdrFramesW).c_str());
        }
    }
    if (g_restore)
    {
        // python's rule: a Restore that cannot load drops the pass with a notice, never the render
        LkSession ls;
        if (lkSession(script, L"rife", (uint32_t)srcW, (uint32_t)srcH, ls) &&
            lkEnsureRestore(script, ls, srcW, srcH, nr.restorePath, nr.rjitPath))
            nr.restore = true;
        else
        {
            LOG("[restore] unavailable, skipping\n");
            nr.restorePath.clear();
            nr.rjitPath.clear();
        }
    }
    std::wstring cacheDir = oa.cacheW.empty() ? std::wstring(L".") : oa.cacheW;
    if (!nr.jitPath.empty())
    {
        std::wstring jp(nr.jitPath.begin(), nr.jitPath.end());
        const size_t sl = jp.find_last_of(L"\\/");
        if (sl != std::wstring::npos)
            cacheDir = jp.substr(0, sl);
    }
    const int64_t tStart = nowQpc100();

    // CUDA: device 0, one non-blocking compute stream, the kernels, the model buffers
    nativeVideoMemoryMark();
    if (cudaSetDevice(0) != cudaSuccess || cudaFree(nullptr) != cudaSuccess)
    {
        LOG("offline: CUDA init failed\n");
        return 2;
    }
    if (cudaStreamCreateWithFlags(&nr.stream, cudaStreamNonBlocking) != cudaSuccess)
    {
        LOG("offline: stream create failed\n");
        return 2;
    }
    if (!nativeBuildKernels(nr, cacheDir))
        return 2;
    const size_t plane = (size_t)nr.ph * nr.pw;
    const int batchMax = nr.batchMax < 1 ? 1 : nr.batchMax;
    const bool outIs16 = out16 < 0 ? fmt16 : (out16 == 1);
    const bool sameFmt = (outIs16 == fmt16);
    const size_t frameBytes = (size_t)srcW * srcH * (fmt16 ? 6 : 3);                       // input frame
    const size_t frameBytesOut = (size_t)outW * outH * (oa.outX2 ? 4 : (outIs16 ? 6 : 3)); // output frame
    uint8_t* dRaw[2]{};
    uint8_t* dOutRaw = nullptr;
    auto cm = [&](void** p, size_t n, const char* what) -> bool {
        if (cudaMalloc(p, n) != cudaSuccess)
        {
            LOG("offline: alloc %s failed\n", what);
            return false;
        }
        return true;
    };
    // the IFNet's timestep and tween buffers follow once the engine's dtypes are known
    // (nativeTrtInit's nativeTweenBuffers)
    // the model's frame: the pictures' own (ph x pw), or RIFE's motion frame at the decoded size (mph x mpw,
    // never larger: the working size is at least the decoded one), which the IFNet reads instead of dX / dXh
    const size_t mplane = nr.mph ? (size_t)nr.mph * nr.mpw : plane;
    if (!cm((void**)&nr.dX, 6 * plane * sizeof(float), "x") ||
        (!nr.nvof && !nr.noEngine && !nr.gmfss && !nr.fruc && !oa.dlssg && !nr.drba &&
         (!cm((void**)&nr.dF[0], 16 * mplane * sizeof(float), "f0") ||
          !cm((void**)&nr.dF[1], 16 * mplane * sizeof(float), "f1") ||
          !cm((void**)&nr.dEncHalf, 16 * mplane * sizeof(uint16_t), "enc") ||
          (nr.mph ? !cm((void**)&nr.dM, 6 * mplane * sizeof(uint16_t), "motion frames")
                  : !cm((void**)&nr.dXh, 6 * plane * sizeof(uint16_t), "x fp16")))) ||
        !cm((void**)&dRaw[0], frameBytes, "raw0") || !cm((void**)&dRaw[1], frameBytes, "raw1") ||
        !cm((void**)&dOutRaw, (size_t)(batchMax + 1) * frameBytesOut, "rawout") ||
        !cm((void**)&nr.dStaticFlag, sizeof(int), "static flag"))
        return 2;
    if (cudaHostAlloc((void**)&nr.hStaticFlag, sizeof(int), cudaHostAllocDefault) != cudaSuccess)
    {
        LOG("offline: pinned static flag alloc failed\n");
        return 2;
    }
    // live's DRBA history: four padded frames and their encodes, the two x buffers, two windows
    // (flow10 | flow12 + flow21), the block0 flow and its splat accumulator
    if (nr.drba && !nativeDrbaAlloc(nr))
        return 2;
    if (nr.restore)
    {
        // the Restore engine's x / y (sized for fp32, either dtype fits), the fold's horizontal
        // pass at the 4x height and the widest target, the fold back to the source size (VSR follows)
        const size_t mp = (size_t)srcW * srcH;
        const int maxTw = outW > srcW ? outW : srcW;
        if (!cm(&nr.dRestIn, 3 * mp * sizeof(float), "restore input") ||
            !cm(&nr.dRestOut, 3 * 16 * mp * sizeof(float), "restore output") ||
            !cm((void**)&nr.dRestTmp, (size_t)3 * 4 * srcH * maxTw * sizeof(float), "restore fold pass") ||
            !cm((void**)&nr.dRest, 3 * mp * sizeof(float), "restore source-size frame"))
            return 2;
    }
    // GMFSS and Restore on HDR video read it as SDR sRGB around BT.2408's reference white: Restore the decoded
    // frames, GMFSS the model frames (also PQ after RTX HDR's TrueHDR on an SDR source)
    nativeHdrModelSetup(nr, oa.srcHdr, oa.srcHdr ? oa.srcHdr : (nr.rtxHdr ? 1 : 0), 203.0f / 80.0f);
    // nvof: the Optical Flow session instead of the engines (every failure is a refusal line);
    // the TensorRT side still loads for Restore
    if (nr.nvof && !nativeNvofSetup(nr))
        return 2;
    if (nr.fruc && !nativeFrucSetup(nr))
        return 2; // the bridge's FRUC instance and surfaces
    if ((!(nr.nvof || nr.noEngine || nr.fruc || oa.dlssg) || nr.restore) && !nativeTrtInit(nr))
        return 2;
    if (nr.mph)
    {
        if (!nr.xHalf || !nr.imgHalf)
        {
            LOG("offline: RIFE's motion frames are fp16 but the engines take fp32 frames (an ONNX before rev 4)\n");
            return 2;
        }
        LOG("offline: RIFE finds the motion on the picture at the decoded %dx%d%s, its warp + blend runs on the "
            "%dx%d pictures\n",
            srcW, srcH, oa.srcHdr == 1 ? " (PQ read as SDR sRGB)" : (oa.srcHdr == 2 ? " (HLG read as SDR sRGB)" : ""),
            w, h);
    }
    if (passes)
    {
        // the staging frame, the VSR bridge and its warm-up, the Restore fold target (the live
        // helper; offline sets no Upscale to and no downscaling fit)
        if (!nativeRtxInit(nr))
            return 2;
        if (!nr.dPres && !cm((void**)&nr.dPres, (size_t)3 * outW * outH * sizeof(float), "pass staging"))
            return 2;
    }
    if (nvPre && !cm((void**)&nr.dSrcPl, (size_t)3 * srcW * srcH * sizeof(float), "decoded source frame"))
        return 2;
    if (g_dlssnr)
    {
        // DLSS 5: the NR core with its own bring-up (a private D3D12 device) at the
        // decoded picture's size w x h: it runs once per decoded frame, before the interpolation
        // (nativeNrFrame from the frame loop). A runtime that
        // cannot start drops the pass with a notice, never the render. NGX prints to the
        // process stdout, which is the encode pipe on a one-shot run: keep a private handle
        // for the frames and point stdout at stderr before any NGX module loads.
        // NGX has no teardown, so this process leaves through ExitProcess and
        // a resident host ends after the item (g_nrAttempted).
        if (!namedPipes)
        {
            HANDLE dup = nullptr;
            if (!DuplicateHandle(GetCurrentProcess(), hOut, GetCurrentProcess(), &dup, 0, FALSE, DUPLICATE_SAME_ACCESS))
            {
                LOG("offline: stdout duplicate failed (%lu)\n", GetLastError());
                return 2;
            }
            hOut = dup;
            fflush(stdout);
            SetStdHandle(STD_OUTPUT_HANDLE, GetStdHandle(STD_ERROR_HANDLE));
            if (_dup2(_fileno(stderr), _fileno(stdout)) != 0)
            {
                LOG("offline: stdout redirect failed (errno %d)\n", errno);
                return 2;
            }
        }
        std::wstring nrDir = exeDir + L"\\..\\dlssnr"; // shipped layout, as live
        wchar_t ov[MAX_PATH]{};
        if (GetEnvironmentVariableW(L"SMV_DLSSNR_DIR", ov, MAX_PATH) && ov[0])
            nrDir = ov;
        wchar_t full[MAX_PATH]{};
        if (GetFullPathNameW(nrDir.c_str(), MAX_PATH, full, nullptr))
            nrDir = full;
        nr::setModuleDir(nrDir.c_str());
        nr::Settings set;
        set.structure = (float)(g_nrStructure < 0.0 ? 0.0 : (g_nrStructure > 2.0 ? 2.0 : g_nrStructure));
        set.tone = (float)(g_nrTone < 0.0 ? 0.0 : (g_nrTone > 2.0 ? 2.0 : g_nrTone));
        set.style = (g_nrStyle >= 0 && g_nrStyle <= 2) ? g_nrStyle : 1;
        set.passes = g_nrPasses < 1 ? 1 : (g_nrPasses > nr::kMaxPasses ? nr::kMaxPasses : g_nrPasses);
        // Zero-copy unless the preview's change map needs the host copies or SMV_NR_STAGED=1
        // asks for CPU staging (the route A/B and the fallback's trigger test). DLSS 5 gets motion
        // vectors on the zero-copy route only (the core binds DLSSNR.MVec at create);
        // SMV_NR_MV=0 = none (a measurement lever), SMV_NR_AUTOMASK=0 = the mask off (read by the core)
        wchar_t sg[8]{}, mvv[8]{}, amv[8]{};
        const bool staged = !oa.nrDeltaW.empty() || (GetEnvironmentVariableW(L"SMV_NR_STAGED", sg, 8) && sg[0] == L'1');
        set.motion = !staged && !(GetEnvironmentVariableW(L"SMV_NR_MV", mvv, 8) && mvv[0] == L'0');
        const bool autoMask = !(GetEnvironmentVariableW(L"SMV_NR_AUTOMASK", amv, 8) && amv[0] == L'0');
        nr::Variant var;
        std::string err;
        g_nrAttempted = true;
        nr::Host* host = new nr::Host();
        LUID cudaLuid{}; // the core's device on the CUDA GPU's adapter, whatever GPU drives the display
        const bool haveLuid = nativeCudaLuid(cudaLuid);
        const int nrc = host->startup((uint32_t)w, (uint32_t)h, set, var, err, true, haveLuid ? &cudaLuid : nullptr);
        if (nrc != 0)
        {
            LOG("[dlss5] unavailable, skipping: %s (exit %d)\n", err.c_str(), nrc);
            host->abandon(); // leaked on purpose, like every NGX object here
        }
        else if (!cm((void**)&nr.dNrIo, (size_t)w * h * 8, "dlss5 frame") ||
                 !cm((void**)&nr.dRawPrev, 3 * plane * sizeof(float), "dlss5 pair reference"))
        {
            LOG("offline: DLSS 5 buffers failed\n");
            return 2;
        }
        else
        {
            nr.nrHost = host;
            nr.nrW = w;
            nr.nrH = h;
            nr.nrDeltaPath = oa.nrDeltaW;
            // HDR video: the pass sees the SDR range around BT.2408's reference white (203 nits, 75 %
            // HLG) through live's HDR math, and a brighter pixel keeps its light above white on top of
            // the pass's result
            nr.nrSrcHdr = oa.srcHdr;
            if (oa.srcHdr)
                nr.nrSdrWhite = 203.0f / 80.0f;
            // measurement lever, never a product setting: Reset on every frame makes the pass a
            // pure function of its input, so a route gate can compare it frame by frame
            wchar_t re[8]{};
            nr.nrResetEvery = GetEnvironmentVariableW(L"SMV_NR_RESET_EVERY", re, 8) && re[0] == L'1';
            // a decoded frame byte-identical to the previous one takes the previous frame's output
            // instead of an evaluate, so a paused or held picture stays exactly still (the kept
            // history would re-shade it a code or two every frame); SMV_NR_REUSE=0 = every frame
            // evaluated
            wchar_t ru[8]{};
            nr.nrReuse = !(GetEnvironmentVariableW(L"SMV_NR_REUSE", ru, 8) && ru[0] == L'0');
            // A handoff that cannot start falls back to staging with a line; the pass still runs, so
            // the line carries no [dlss5] tag (the preview reads that tag as "DLSS 5 not applied").
            // Motion that cannot start leaves the shared field at zero (no motion) with a line.
            if (!staged)
            {
                std::string zerr;
                nr.nrZeroCopy = host->startShared(zerr) && nativeNrImport(nr, *host, zerr);
                if (!nr.nrZeroCopy)
                {
                    nativeNrReleaseImports(nr);
                    LOG("offline: DLSS 5 zero-copy handoff unavailable (%s), CPU staging\n", zerr.c_str());
                }
                else if (set.motion && !nativeNrMotionSetup(nr, zerr))
                {
                    nativeNrMotionFree(nr);
                    if (nr.dNrShMv)
                        cudaMemset2DAsync(nr.dNrShMv, nr.nrMvPitch, 0, (size_t)w * 4, h, nr.stream);
                    LOG("offline: DLSS 5 motion vectors unavailable (%s), none\n", zerr.c_str());
                }
            }
            if (!nr.nrZeroCopy &&
                (cudaHostAlloc((void**)&nr.hNrIn, (size_t)w * h * 8, cudaHostAllocDefault) != cudaSuccess ||
                 cudaHostAlloc((void**)&nr.hNrOut, (size_t)w * h * 8, cudaHostAllocDefault) != cudaSuccess))
            {
                LOG("offline: DLSS 5 buffers failed\n");
                return 2;
            }
            LOG("DLSS 5 Neural Rendering ready (DLAA, structure %.2f, tone %.2f, style %d, %s, before the "
                "interpolation%s%s, passes %d%s%s%s) @ %dx%d\n",
                set.structure, set.tone, set.style, nr.nrZeroCopy ? "zero-copy" : "CPU staging",
                nr.nrMotion ? ", motion vectors" : ", no motion vectors", autoMask ? ", auto mask" : "", host->passes(),
                nr.nrResetEvery ? ", SMV_NR_RESET_EVERY" : "",
                nr.nrReuse ? ", identical frames reuse the last output" : "",
                oa.srcHdr == 2   ? ", HLG video: its SDR range through the pass, white 203 nits"
                : oa.srcHdr == 1 ? ", PQ video: its SDR range through the pass, white 203 nits"
                                 : "",
                w, h);
            if (host->passes() < set.passes)
                LOG("offline: DLSS 5 runs %d of %d passes (%s)\n", host->passes(), set.passes,
                    host->passNote().c_str());
        }
    }

    OfflineIo io;
    io.frameBytes = frameBytes;
    io.frameBytesOut = frameBytesOut;
    io.nIn = 4;
    io.nOut = (batchMax < 8 ? batchMax : 8) + 2;
    io.totalFrames = frames;
    io.progressEvery = progressEvery;
    io.multi = multi;
    io.frameUnits = oa.echo;
    io.fpsRatio = fpsMode ? ratio : 0.0;
    io.outBase = oa.resumeOut;
    io.skipIn = oa.skipIn;
    io.thumbPath = oa.thumbW;
    io.thumbOff = oa.thumbOffW;
    io.outW = outW;
    io.outH = outH;
    io.outFmt = oa.outX2 ? 2 : (outIs16 ? 1 : 0);
    io.hStdin = hIn;
    io.hStdout = hOut;
    io.hIn.resize(io.nIn);
    io.h2dEv.resize(io.nIn);
    io.inState.assign(io.nIn, 0);
    io.inWritten.assign(io.nIn, 1);
    io.hOut.resize(io.nOut);
    io.outEv.resize(io.nOut);
    for (int i = 0; i < io.nIn; i++)
    {
        if (cudaHostAlloc((void**)&io.hIn[i], frameBytes, cudaHostAllocDefault) != cudaSuccess)
        {
            LOG("offline: pinned input alloc failed\n");
            return 2;
        }
        cudaEventCreateWithFlags(&io.h2dEv[i], cudaEventDisableTiming);
    }
    for (int i = 0; i < io.nOut; i++)
    {
        if (cudaHostAlloc((void**)&io.hOut[i], frameBytesOut, cudaHostAllocDefault) != cudaSuccess)
        {
            LOG("offline: pinned output alloc failed\n");
            return 2;
        }
        cudaEventCreateWithFlags(&io.outEv[i], cudaEventDisableTiming);
        io.outFree.push_back(i);
    }
    // DLSS 4.5: the RGBA8 staging (device + pinned: the two latest frames and the pair's
    // generated frames), then the server itself; its handshake decides "up" vs "unsupported"
    DgChild dgc;
    uint8_t* dDgRgba = nullptr;
    float* dDgF = nullptr;
    uint8_t* hDgIn[2]{};
    uint8_t* hDgOut = nullptr;
    const size_t dgBytes = (size_t)nr.pw * nr.ph * 4;
    int dgHlg = oa.srcHdr == 2 ? 1 : 0; // HDR10 frames: the HLG picture goes as PQ and comes back as HLG
    if (oa.dlssg)
    {
        if (!cm((void**)&dDgRgba, dgBytes, "dlss frame") ||
            !cm((void**)&dDgF, 3 * plane * sizeof(float), "dlss tween") ||
            cudaHostAlloc((void**)&hDgIn[0], dgBytes, cudaHostAllocDefault) != cudaSuccess ||
            cudaHostAlloc((void**)&hDgIn[1], dgBytes, cudaHostAllocDefault) != cudaSuccess ||
            cudaHostAlloc((void**)&hDgOut, dgBytes * (size_t)(multi - 1), cudaHostAllocDefault) != cudaSuccess)
        {
            LOG("offline: DLSS 4.5 buffers failed\n");
            return 2;
        }
        dgc.dir = exeDir + L"\\..\\dlssg"; // shipped layout (engine\live -> engine\dlssg), as dlssg.py
        wchar_t ov[MAX_PATH]{};
        if (GetEnvironmentVariableW(L"SMV_DLSSG_DIR", ov, MAX_PATH) && ov[0])
            dgc.dir = ov;
        wchar_t full[MAX_PATH]{};
        if (GetFullPathNameW(dgc.dir.c_str(), MAX_PATH, full, nullptr))
            dgc.dir = full;
        dgc.w = nr.pw;
        dgc.h = nr.ph;
        dgc.gen = multi - 1;
        dgc.pauseFile = pauseFile;
        // HDR planes (PQ or HLG video, RTX HDR's PQ) go as HDR10, the DLSS-G guide's HDR input; SMV_DLSSG_HDR10=0 =
        // RGBA8 of the codes
        dgc.hdr10 = (oa.srcHdr || nr.rtxHdr) && lkEnv("SMV_DLSSG_HDR10") != "0";
        int spawnRc = dgSpawn(dgc);
        if (dgc.hdr10 && (spawnRc == 5 || spawnRc == -2))
        {
            LOG("DLSS 4.5: %s, 8-bit frames\n",
                spawnRc == 5 ? "the swap chain refused the HDR10 colour space" : "this dlssg2f.exe has no HDR10 mode");
            dgc.hdr10 = false;
            spawnRc = dgSpawn(dgc);
        }
        if (spawnRc == 3 || (spawnRc == 0 && dgc.maxGen && (unsigned)dgc.gen > dgc.maxGen))
        {
            if (spawnRc == 0)
                dgEnd(dgc, true);
            LOG("DLSS Frame Generation unavailable: this GPU does not support %dx DLSS multi-frame generation "
                "(RTX 50 series does up to 6x, RTX 40 series only 2x); pick a lower multiplier or GMFSS\n",
                multi);
            return 3;
        }
        if (spawnRc != 0)
        {
            LOG("DLSS Frame Generation unavailable: DLSS Frame Generation host failed to start%s; it needs an "
                "RTX 40/50 GPU, a recent driver and Windows HW-accelerated GPU scheduling ON\n",
                spawnRc == 2 ? " (GPU/driver/OS unsupported)" : "");
            return 2;
        }
        dgc.watch = std::thread(dgWatch, &dgc);
        LOG("DLSS Frame Generation ready (%dx%d, %dx, server max %u%s)\n", nr.pw, nr.ph, multi, dgc.maxGen,
            dgc.hdr10 ? (dgHlg ? ", HDR10 frames, the HLG picture as PQ" : ", HDR10 frames") : "");
    }
    if (g_noGpuFit)
        LOG("offline: GPU fit off (--no-gpu-fit): no video memory fit\n");
    LOG("offline host ready:%dx%d %s in, %dx%d %s out, x%d, padded %dx%d, batch max %d, graph %s%s%s%s%s%s%s, %.2f s to ready\n",
        w, h, fmt16 ? "rgb48le" : "rgb24", outW, outH, oa.outX2 ? "x2rgb10le" : (outIs16 ? "rgb48le" : "rgb24"), multi,
        nr.pw, nr.ph, batchMax, g_offlineGraph ? "on" : "off",
        nr.nvof ? ", model nvof"
                : (nr.noEngine
                       ? ", engine=none (effects only)"
                       : (nr.gmfss ? ", model gmfss"
                                   : (nr.drba ? ", model rife drba"
                                              : (nr.fruc ? ", model fruc" : (oa.dlssg ? ", model dlss 4.5" : ""))))),
        nr.restore ? ", restore" : "", nr.vsr ? ", rtx vsr" : ((outW != w || outH != h) ? ", lanczos3 upscale" : ""),
        nr.nrHost ? ", dlss 5" : "", nr.sharpen > 0.0f ? ", sharpen" : "", nr.rtxHdr ? ", rtx hdr" : "",
        (double)(nowQpc100() - tStart) / 1e7);
    if (fpsMode)
        LOG("offline: --fps mode, %.17g output frames per source frame\n", ratio);

    std::thread reader(offlineReader, &io);
    std::thread writer(offlineWriter, &io);
    cudaStream_t st = nr.stream;
    const int ps = (int)plane, rs = nr.pw;
    bool havePrev = false;
    uint64_t pairs = 0;
    uint64_t heldPairs = 0; // identical pairs passed through instead of interpolated
    bool failed = false;
    int failRc = 1;        // the exit code of a failed render (DLSS 4.5: 3 multiplier, 4 host lost)
    bool dgPrimed = false; // the DLSS 4.5 server has seen the pair's left frame
    const int64_t tLoop = nowQpc100();
    bool paused = false;
    // --fps mode: render.py's _pair_fracs(p) in the same double arithmetic (the output times
    // (j + 0.5) / ratio inside [p, p + 1), zero or more per pair), and closing_slot's count for
    // the last source frame's own interval (n = source frames, python's i = n - 1)
    // Both take this run's local index; a resumed run adds --resume-pair, the render's own
    // source index of its first decoded frame (python's i / RESUME_SKIP_SRC).
    std::vector<double> fr;
    bool realOutFresh = false; // the real-frame output slot holds the pair's left frame through the chain
    const uint64_t pairBase = oa.resumePair;
    auto pairFracs = [&](uint64_t p) {
        fr.clear();
        const double pd = (double)(pairBase + p);
        const int64_t lo = (int64_t)std::ceil(pd * ratio - 0.5), hi = (int64_t)std::ceil((pd + 1.0) * ratio - 0.5);
        for (int64_t j = lo; j < hi; j++)
            fr.push_back(((double)j + 0.5) / ratio - pd);
    };
    auto closingCount = [&](uint64_t n) -> int64_t {
        if (!fpsMode)
            return multi;
        n += pairBase;
        return (int64_t)std::ceil((double)n * ratio - 0.5) - (int64_t)std::ceil((double)(n - 1) * ratio - 0.5);
    };
    // resume: the first --resume-drop outputs are banked already. Every emission point asks
    // before it runs any pass, so a dropped frame never reaches DLSS 5's history, the TrueHDR
    // statistics or the encoder (python never generates them: resume_active, skip, pend)
    int dropLeft = oa.resumeDrop > 0 ? oa.resumeDrop : 0;
    auto dropOne = [&]() -> bool {
        if (!dropLeft)
            return false;
        dropLeft--;
        return true;
    };
    if (oa.resumed)
        LOG("offline: resumed at source frame %llu, %llu output frames banked, dropping the first %d\n",
            (unsigned long long)pairBase, (unsigned long long)oa.resumeOut, dropLeft);
    uint64_t nFrames = 0;
    // ---- DRBA: render_loops.drba_loop on live's lag-1 windows --------------------------------
    // Every output sits on the uniform offset grid f = (j + 0.5) / multi of its pair; no real
    // frame passes through. Pair (k-2, k-1) goes out once frame k is in the ring (live's group
    // order: f < 0.5 from window k-2 side +1, f >= 0.5 from window k-1 side -1, the first pair's
    // f < 0.5 plain pair RIFE = python's head window). An identical pair holds its frame (one
    // processing per pair side, python's held_out per window). At EOF the last pair's f >= 0.5
    // is plain pair RIFE (python's tail window), then the closing slot repeats the last output
    // multi times; a lone frame goes out once.
    uint8_t* const dHeldOut = dOutRaw + (size_t)batchMax * frameBytesOut;
    const uint8_t* drLastOut = nullptr;
    bool drLastStatic = false;
    auto drSend = [&](const uint8_t* dO, bool repeat) -> bool {
        int o = -1;
        {
            std::unique_lock<std::mutex> lk(io.m);
            io.cv.wait(lk, [&] { return !io.outFree.empty() || io.fail; });
            if (io.fail)
                return false;
            o = io.outFree.front();
            io.outFree.pop_front();
        }
        if (repeat && nr.rtxHdr)
            nativeOfflineThdrRepeat(nr);
        if (cudaMemcpyAsync(io.hOut[o], dO, frameBytesOut, cudaMemcpyDeviceToHost, nr.stream) != cudaSuccess)
        {
            io.setFail("D2H copy (drba) failed");
            return false;
        }
        cudaEventRecord(io.outEv[o], nr.stream);
        {
            std::lock_guard<std::mutex> lk(io.m);
            io.items.push_back({1, o});
        }
        io.cv.notify_all();
        drLastOut = dO;
        return true;
    };
    // half != 0: an fp16 source (a tween as the IFNet wrote it)
    auto drEmit = [&](const void* src, uint8_t* dO, int half = 0) -> bool {
        if (passes)
            return nativeOfflineEmit(nr, src, ps, rs, dO, outIs16, half);
        void* a[] = {(void*)&src, &half, (void*)&ps, (void*)&rs, &w, &h, &dO};
        return cuLaunchKernel(outIs16 ? nr.fPackOutRaw16 : nr.fPackOutRaw8, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1,
                              0, (CUstream)nr.stream, a, nullptr) == CUDA_SUCCESS;
    };
    // pair (drFid - 2, drFid - 1); tailPlain = its f >= 0.5 as plain pair RIFE (the EOF window).
    // --fps: the pair's _pair_fracs (0-based pair drFid - 3); a slot at f = 0 is python's t == 1
    // (inference_ts_drba, or the head window's inference at t - 1 = 0): the pair's left frame
    auto drPair = [&](bool held, bool tailPlain) -> bool {
        const int nHist = nr.drFid >= 4 ? 4 : (int)nr.drFid;
        int heldSide = -1; // the side whose held frame sits in dHeldOut
        if (fpsMode)
            pairFracs((uint64_t)nr.drFid - 3);
        const int nF = fpsMode ? (int)fr.size() : multi;
        for (int j = 0; j < nF; j++)
        {
            if (dropOne())
                continue; // resume: drba_loop's pend, the pair's banked slots in order
            float f = fpsMode ? (float)fr[j] : ((float)j + 0.5f) / (float)multi;
            const int side = (fpsMode ? fr[j] < 0.5 : f < 0.5f) ? 0 : 1;
            if (side == 0 && f >= 0.5f)
                f = nextafterf(0.5f, 0.0f); // python splits on the double
            bool h = held;
            if (!h && fpsMode && side == 0 && 1.0 + fr[j] == 1.0)
            {
                if (!drEmit(drbaFrame(nr, nr.drFid - 2), dOutRaw))
                {
                    io.setFail("drba left frame emit failed");
                    return false;
                }
                if (!drSend(dOutRaw, false))
                    return false;
                continue;
            }
            if (!h)
            {
                if (!nativeDrbaTween(nr, f, nHist, h, tailPlain && side == 1))
                {
                    io.setFail("drba tween failed");
                    return false;
                }
                if (!h)
                {
                    if (!drEmit(mergedAt(nr, 0, plane), dOutRaw, (int)nr.outHalf))
                    {
                        io.setFail("drba emit failed");
                        return false;
                    }
                    if (!drSend(dOutRaw, false))
                        return false;
                    continue;
                }
            }
            // held: the lagged real frame k - 1 (both frames of an identical pair are the same)
            const bool again = heldSide == side;
            if (!again)
            {
                if (!drEmit(drbaFrame(nr, nr.drFid - 1), dHeldOut))
                {
                    io.setFail("drba held emit failed");
                    return false;
                }
                heldSide = side;
            }
            if (!drSend(dHeldOut, again))
                return false;
        }
        pairs++;
        if (held)
            heldPairs++;
        return true;
    };
    for (uint64_t i = 0; !failed; i++)
    {
        if (i == 3)
            nativeVideoMemoryNote(); // the first frames are through every pass
        // cooperative pause, render.py's contract: the GUI creates the file to pause and
        // deletes it to resume; checked at the pair boundary, queued output keeps draining
        if (!pauseFile.empty())
        {
            while (GetFileAttributesW(pauseFile.c_str()) != INVALID_FILE_ATTRIBUTES)
            {
                if (!paused)
                {
                    paused = true;
                    LOG("PAUSED\n");
                }
                Sleep(200);
            }
            if (paused)
            {
                paused = false;
                LOG("RESUMED\n");
            }
        }
        const int slot = (int)(i % io.nIn);
        {
            std::unique_lock<std::mutex> lk(io.m);
            io.cv.wait(lk, [&] { return io.inState[slot] == 1 || io.readerEof || io.fail; });
            if (io.fail)
            {
                failed = true;
                break;
            }
            if (io.inState[slot] != 1)
                break; // EOF: no more frames
        }
        uint8_t* dR = dRaw[i & 1];
        if (cudaMemcpyAsync(dR, io.hIn[slot], frameBytes, cudaMemcpyHostToDevice, st) != cudaSuccess)
        {
            io.setFail("H2D copy failed");
            failed = true;
            break;
        }
        cudaEventRecord(io.h2dEv[slot], st);
        {
            std::lock_guard<std::mutex> lk(io.m);
            io.inState[slot] = 2;
        }
        nFrames++;
        if (havePrev)
        {
            cudaMemcpyAsync(nr.dX, nr.dX + 3 * plane, 3 * plane * sizeof(float), cudaMemcpyDeviceToDevice, st);
            // the frames the IFNet reads keep the same two frames: the motion frames, or the fp16
            // copy an fp16-x IFNet reads (DRBA builds its own x from its ring)
            if (nr.mph && !nr.drba)
                cudaMemcpyAsync(nr.dM, nr.dM + 3 * mplane, 3 * mplane * sizeof(uint16_t), cudaMemcpyDeviceToDevice, st);
            else if (nr.xHalf && !nr.drba)
                cudaMemcpyAsync(nr.dXh, nr.dXh + 3 * plane, 3 * plane * sizeof(uint16_t), cudaMemcpyDeviceToDevice, st);
        }
        float* dCur = nr.dX + 3 * plane;
        // DLSS 5 reuse: the decoded bytes against the previous frame's (dRaw alternates, so the other
        // buffer still holds them). Identical = this frame takes the previous frame's DLSS 5 output
        // and skips the pack, the pre-model stage and the evaluate: a pair route's shift above left
        // that output in dX; --no-interp forms no pairs and never shifts, so it is still in dCur
        const bool nrOn = nr.nrHost && !nr.nrFailed;
        bool nrSame = false, nrTested = false;
        if (nrOn && nr.nrReuse && i > 0 && !nr.nrFirst)
        {
            const long long nb = (long long)frameBytes;
            const uint8_t* dPrevRaw = dRaw[(i - 1) & 1];
            void* ad[] = {(void*)&dPrevRaw, &dR, (void*)&nb, &nr.dStaticFlag};
            const unsigned blocks = (unsigned)((nb + 16LL * 256 - 1) / (16LL * 256));
            if (cudaMemsetAsync(nr.dStaticFlag, 0, sizeof(int), st) != cudaSuccess ||
                cuLaunchKernel(nr.fRawDiff, blocks, 1, 1, 256, 1, 1, 0, (CUstream)st, ad, nullptr) != CUDA_SUCCESS ||
                cudaMemcpyAsync(nr.hStaticFlag, nr.dStaticFlag, sizeof(int), cudaMemcpyDeviceToHost, st) !=
                    cudaSuccess ||
                cudaStreamSynchronize(st) != cudaSuccess)
            {
                io.setFail("DLSS 5 reuse test failed");
                failed = true;
                break;
            }
            nrSame = *nr.hStaticFlag == 0;
            nrTested = true;
            if (nrSame && havePrev &&
                (cudaMemcpyAsync(dCur, nr.dX, 3 * plane * sizeof(float), cudaMemcpyDeviceToDevice, st) != cudaSuccess ||
                 (nr.mph && !nr.drba &&
                  cudaMemcpyAsync(nr.dM + 3 * mplane, nr.dM, 3 * mplane * sizeof(uint16_t), cudaMemcpyDeviceToDevice,
                                  st) != cudaSuccess)))
            {
                io.setFail("DLSS 5 reuse copy failed");
                failed = true;
                break;
            }
        }
        if (nrSame)
            nr.nrReused++;
        else if (nvPre)
        {
            // the decoded frame at its own size, then Restore / RTX VSR / Lanczos3
            // to the model (output) size, into dCur with the pad
            int sps = srcW * srcH;
            void* a[] = {&dR, &srcW, &srcH, &nr.dSrcPl, &srcH, &srcW, &sps};
            if (cuLaunchKernel(fmt16 ? nr.fPackInRaw16 : nr.fPackInRaw8, (srcW + 15) / 16, (srcH + 15) / 16, 1, 16, 16,
                               1, 0, (CUstream)st, a, nullptr) != CUDA_SUCCESS)
            {
                io.setFail("packInRaw launch failed");
                failed = true;
                break;
            }
            if (!nativeOfflinePreModel(nr, dCur, ps))
            {
                io.setFail("pre-model passes failed");
                failed = true;
                break;
            }
        }
        else
        {
            void* a[] = {&dR, &w, &h, &dCur, &nr.ph, &nr.pw, (void*)&ps};
            if (cuLaunchKernel(fmt16 ? nr.fPackInRaw16 : nr.fPackInRaw8, (nr.pw + 15) / 16, (nr.ph + 15) / 16, 1, 16,
                               16, 1, 0, (CUstream)st, a, nullptr) != CUDA_SUCCESS)
            {
                io.setFail("packInRaw launch failed");
                failed = true;
                break;
            }
        }
        // IDENTICAL PAIR: two packed inputs compared element by element on the device, the flag
        // read back once per pair (used below)
        auto identicalPair = [&](float* dPrev, bool& same) -> bool {
            const int nCmp = (int)(3 * plane);
            void* ad[] = {&dPrev, &dCur, (void*)&nCmp, &nr.dStaticFlag};
            if (cudaMemsetAsync(nr.dStaticFlag, 0, sizeof(int), st) != cudaSuccess ||
                cuLaunchKernel(nr.fPairDiff, (nCmp + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, ad, nullptr) !=
                    CUDA_SUCCESS ||
                cudaMemcpyAsync(nr.hStaticFlag, nr.dStaticFlag, sizeof(int), cudaMemcpyDeviceToHost, st) !=
                    cudaSuccess ||
                cudaStreamSynchronize(st) != cudaSuccess)
                return false;
            same = (*nr.hStaticFlag == 0);
            return true;
        };
        const bool testPair = havePrev && g_staticHold && nr.fPairDiff && !oa.dlssg; // DLSS 4.5 never holds (hold_ok)
        bool pairStatic = false, pairTested = false;
        // the reuse test above compared the decoded frames already: it decides the pair too
        if (nrTested && testPair)
        {
            pairStatic = nrSame;
            pairTested = true;
        }
        // DLSS 5 on the decoded frame, in place, before anything reads it (the Head encode, the
        // pair's model, the emit): frame generation after DLSS 5. The pair test
        // compares the decoded frames first: the pass keeps history, so two identical frames can
        // come back a fraction of a level apart.
        if (nrOn && !nrSame)
        {
            if (testPair && !pairTested)
            {
                if (!identicalPair(nr.dRawPrev, pairStatic))
                {
                    io.setFail("identical-pair test failed");
                    failed = true;
                    break;
                }
                pairTested = true;
            }
            if (cudaMemcpyAsync(nr.dRawPrev, dCur, 3 * plane * sizeof(float), cudaMemcpyDeviceToDevice, st) !=
                cudaSuccess)
            {
                io.setFail("DLSS 5 pair reference copy failed");
                failed = true;
                break;
            }
            if (!nativeNrFrame(nr, dCur, nr.pw, nr.ph, ps))
            {
                io.setFail("DLSS 5 failed");
                failed = true;
                break;
            }
        }
        // RIFE's motion frame: the finished picture as the IFNet would have read it, minus what costs
        // RIFE its motion (measured on an exact pan): the PQ encoding (read as SDR sRGB) and an enlarge
        // (Lanczos3 back to the decoded size, into dSrcPl, which the pre-model stage no longer needs),
        // padded to the model's frame. Taken after FSR, or before the post with RTX HDR (FSR is fused
        // into TrueHDR's input there). A reused frame copied it above.
        auto motionFrame = [&]() -> bool {
            const float* msrc = dCur;
            int sps = (int)plane, srs = nr.pw, sw = w, sh = h;
            if (w != srcW || h != srcH)
            {
                int zero = 0, tps = srcW * srcH, n = 3 * srcW * srcH;
                void* a[] = {(void*)&dCur, &zero, (void*)&ps, &nr.pw, &w, &h, &nr.dSrcPl, &srcW, &srcH};
                void* b[] = {&nr.dSrcPl, &n};
                if (cuLaunchKernel(nr.fFitPlanar, (srcW + 15) / 16, (srcH + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st, a,
                                   nullptr) != CUDA_SUCCESS ||
                    cuLaunchKernel(nr.fClamp01, (n + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, b, nullptr) !=
                        CUDA_SUCCESS)
                    return false;
                msrc = nr.dSrcPl;
                sps = tps;
                srs = srcW;
                sw = srcW;
                sh = srcH;
            }
            int dps = (int)mplane, mode = oa.srcHdr;
            float white = 203.0f / 80.0f; // BT.2408's reference white in scRGB units
            float knee = kMotionKnee, head = kMotionHead;
            uint16_t* mdst = nr.drba ? drbaMotionNext(nr) : nr.dM + 3 * mplane;
            void* a[] = {(void*)&msrc, &sps,    &srs,    &sw,   &sh,    &mdst, &dps,
                         &nr.mpw,      &nr.mpw, &nr.mph, &mode, &white, &knee, &head};
            return cuLaunchKernel(nr.fMotionIn, (nr.mpw + 15) / 16, (nr.mph + 15) / 16, 1, 16, 16, 1, 0, (CUstream)st,
                                  a, nullptr) == CUDA_SUCCESS;
        };
        const bool motionHere = nr.mph && !nrSame;
        if (motionHere && nr.rtxHdr && !motionFrame())
        {
            io.setFail("motion frame launch failed");
            failed = true;
            break;
        }
        // FSR and RTX TrueHDR after DLSS 5, once per decoded frame before anything reads it (NVIDIA's
        // order: post-processing and the HDR conversion before frame generation)
        if (!nrSame && !nativePreModelPost(nr, dCur, ps, nr.rtxHdr))
        {
            io.setFail("FSR / RTX HDR before the model failed");
            failed = true;
            break;
        }
        if (motionHere && !nr.rtxHdr && !motionFrame())
        {
            io.setFail("motion frame launch failed");
            failed = true;
            break;
        }
        // Head encode of the new frame; the previous frame's encode is reused (exact). The nvof
        // model has no per-frame state: both frames go to gray8 per pair below.
        nr.fCur ^= 1;
        if (!nr.nvof && !nr.noEngine && !nr.gmfss && !nr.drba && !nr.fruc && !oa.dlssg)
        {
            nvinfer1::Dims4 din{1, 3, nr.mph ? nr.mph : nr.ph, nr.mph ? nr.mpw : nr.pw};
            if (!nr.ctxEnc->setInputShape("img", din))
            {
                io.setFail("encode setInputShape rejected");
                failed = true;
                break;
            }
            // fp16 frames (ONNX rev 4): the new frame into the fp16 copy's cur half, read by an
            // fp16-x IFNet and / or an fp16-img encode; RIFE's two domains: the motion frame
            uint16_t* cur16 = nr.mph ? nr.dM + 3 * mplane : nr.dXh + 3 * plane;
            if (!nr.mph && (nr.xHalf || nr.imgHalf) && !nativeF2h(nr, dCur, cur16, 3 * plane, st))
            {
                io.setFail("f2h launch failed");
                failed = true;
                break;
            }
            nr.ctxEnc->setTensorAddress("img", nr.imgHalf ? (void*)cur16 : (void*)dCur);
            // fp16 features (ONNX rev 3): the encode writes dF directly, no widen pass
            const bool widen = nr.encHalf && !nr.featHalf;
            void* encOut = widen ? (void*)nr.dEncHalf : (void*)nr.dF[nr.fCur];
            nr.ctxEnc->setTensorAddress("feat", encOut);
            if (!nr.ctxEnc->enqueueV3(st))
            {
                io.setFail("encode enqueueV3 returned false");
                failed = true;
                break;
            }
            if (widen)
            {
                int n = (int)(16 * mplane);
                void* a[] = {&nr.dEncHalf, &nr.dF[nr.fCur], &n};
                if (cuLaunchKernel(nr.fH2f, (n + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, a, nullptr) !=
                    CUDA_SUCCESS)
                {
                    io.setFail("h2f launch failed");
                    failed = true;
                    break;
                }
            }
        }
        // IDENTICAL PAIR (identicalPair above). Equal = there is no motion in this pair, so the
        // engine is skipped and every tween slot carries the real frame's own output bytes (the
        // Head encode above still ran, so the next pair's chain state is unchanged). Exact
        // equality only; SMV_NO_STATIC_HOLD=1 turns it off.
        if (testPair && !pairTested && !identicalPair(nr.dX, pairStatic))
        {
            io.setFail("identical-pair test failed");
            failed = true;
            break;
        }
        // the pair's slots: the N-1 on-grid tweens, or --fps mode's _pair_fracs (maybe none)
        int nT = multi - 1;
        if (fpsMode)
        {
            if (havePrev)
                pairFracs(i - 1);
            else
                fr.clear();
            nT = (int)fr.size();
        }
        const bool makesTweens = havePrev && !pairStatic && nT > 0;
        // GMFSS: feat_ext of every new frame (the next pair's feat0), the flow, metric and
        // pyramids only for a pair that makes tweens (live's nativeGroup; python skips reuse)
        if (nr.gmfss && !nativeGmfssPair(nr, nr.dX, dCur, makesTweens))
        {
            io.setFail("gmfss pair failed");
            failed = true;
            break;
        }
        if (nr.fruc)
        {
            // Smooth Motion: every new frame into a BGRA surface; the pair is interpolated only
            // when it makes tweens (python's hold skips NvOFFRUC, and so does an --fps pair with
            // no slot). The bridge primes I0 itself on every call, so python never warps a
            // skipped pair: marking the end of a pair without tweens as the last tweened end
            // skips live's priming warp and the bridge sees python's call sequence (no reset, ever)
            if (!nativeFrucPair(nr, dCur, makesTweens ? (uint32_t)nT : 0u))
            {
                io.setFail("fruc pair failed");
                failed = true;
                break;
            }
            if (havePrev && !makesTweens && nr.frLast >= 0)
                nr.frLast = nr.frB;
        }
        if (oa.dlssg)
        {
            // DLSS 4.5: every frame to RGBA8 (hdr10: R10G10B10A2) at the padded size (dlssg.py _send), then
            // for a pair the server's generated frames (dlssg.py interpolate; the sync also retires the last
            // pair's uploads from hDgOut before it is refilled)
            uint8_t* hCur = hDgIn[i & 1];
            int f32 = 0;
            void* a[] = {&dCur, &f32, (void*)&ps, (void*)&rs, &nr.pw, &nr.ph, &dDgRgba};
            void* a10[] = {&dCur, &f32, (void*)&ps, (void*)&rs, &nr.pw, &nr.ph, &dgHlg, &dDgRgba};
            if (cuLaunchKernel(dgc.hdr10 ? nr.fPackR10 : nr.fPackBgra, (nr.pw + 15) / 16, (nr.ph + 15) / 16, 1, 16, 16,
                               1, 0, (CUstream)st, dgc.hdr10 ? a10 : a, nullptr) != CUDA_SUCCESS ||
                cudaMemcpyAsync(hCur, dDgRgba, dgBytes, cudaMemcpyDeviceToHost, st) != cudaSuccess ||
                cudaStreamSynchronize(st) != cudaSuccess)
            {
                io.setFail("dlss frame staging failed");
                failed = true;
                break;
            }
            if (havePrev)
            {
                const int dr = dgPair(dgc, hDgIn[(i - 1) & 1], hCur, hDgOut, dgPrimed);
                if (dr)
                {
                    failRc = dr;
                    io.setFail(dr == 4 ? "the DLSS 4.5 host stayed lost through its restarts"
                                       : "the DLSS 4.5 host refused the multiplier");
                    failed = true;
                    break;
                }
            }
        }
        if (nr.drba)
        {
            // the frame and its encode into the ring, then the lagged pair (its identical-pair
            // test is the previous iteration's); no real frame goes out, so the input slot is
            // released here (the reader syncs its H2D event before reusing it)
            if (!nativeDrbaPush(nr, dCur, nrSame))
            {
                io.setFail("drba push failed");
                failed = true;
                break;
            }
            if (nr.drFid >= 3 && !drPair(drLastStatic, false))
            {
                failed = true;
                break;
            }
            drLastStatic = pairStatic;
            {
                std::lock_guard<std::mutex> lk(io.m);
                io.inWritten[slot] = 1;
            }
            io.cv.notify_all();
            havePrev = true;
            continue;
        }
        if (fpsMode && havePrev && pairStatic)
        {
            // --fps held pair (fps_loop): every slot of the pair is the frame itself, through
            // the passes once (to_bytes(I0)), repeated
            bool emitted = false;
            for (int j = 0; j < nT; j++)
            {
                if (dropOne())
                    continue; // resume: fps_loop's fracs[skip:]
                if (!emitted && !drEmit(dCur, dHeldOut))
                {
                    io.setFail("held emit (fps) failed");
                    failed = true;
                    break;
                }
                if (!drSend(dHeldOut, emitted))
                {
                    failed = true;
                    break;
                }
                emitted = true;
            }
            pairs++;
            if (nT)
                heldPairs++;
        }
        else if (havePrev && pairStatic)
        {
            // the held pair: the real frame's bytes on every tween slot, the same bytes the
            // real frame below is written from (rgb24 in and out: the staged input itself;
            // rgb24 in, rgb48le out: its x * 257 expansion, exactly the real frame's path)
            for (int j = 0; j < multi - 1 && !failed; j++)
            {
                if (dropOne())
                    continue; // resume: ongrid_loop's mtw[skip:]
                int o = -1;
                {
                    std::unique_lock<std::mutex> lk(io.m);
                    io.cv.wait(lk, [&] { return !io.outFree.empty() || io.fail; });
                    if (io.fail)
                    {
                        failed = true;
                        break;
                    }
                    o = io.outFree.front();
                    io.outFree.pop_front();
                }
                const uint8_t* srcOut = dR;
                if (passes && nr.nrHost)
                {
                    // DLSS 5 is temporal: python's held slots carry the real frame's finished
                    // bytes (render_loops out_cur) with no second NR call, so the history sees
                    // each real frame once. That output still sits in the real-frame slot,
                    // unless a resume dropped that real frame: python's out_cur is None then
                    // and its to_bytes runs the chain on the frame once, here
                    uint8_t* dO = dOutRaw + (size_t)batchMax * frameBytesOut;
                    srcOut = dO;
                    if (!realOutFresh)
                    {
                        if (!nativeOfflineEmit(nr, dCur, ps, rs, dO, outIs16))
                        {
                            io.setFail("passes (held, resumed) failed");
                            failed = true;
                            break;
                        }
                        realOutFresh = true;
                    }
                    else if (nr.rtxHdr)
                        nativeOfflineThdrRepeat(nr);
                }
                else if (passes)
                {
                    // with passes the held slot is the real frame through the chain, as python
                    uint8_t* dO = dOutRaw + (size_t)batchMax * frameBytesOut;
                    if (!nativeOfflineEmit(nr, dCur, ps, rs, dO, outIs16))
                    {
                        io.setFail("passes (held) failed");
                        failed = true;
                        break;
                    }
                    srcOut = dO;
                }
                else if (!sameFmt)
                {
                    uint8_t* dO = dOutRaw + (size_t)batchMax * frameBytesOut;
                    int n = (int)((size_t)w * h * 3);
                    void* a[] = {&dR, &n, &dO};
                    if (cuLaunchKernel(nr.fExpand8to16, (n + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, a,
                                       nullptr) != CUDA_SUCCESS)
                    {
                        io.setFail("expand8to16 (held) launch failed");
                        failed = true;
                        break;
                    }
                    srcOut = dO;
                }
                if (cudaMemcpyAsync(io.hOut[o], srcOut, frameBytesOut, cudaMemcpyDeviceToHost, st) != cudaSuccess)
                {
                    io.setFail("D2H copy (held) failed");
                    failed = true;
                    break;
                }
                cudaEventRecord(io.outEv[o], st);
                {
                    std::lock_guard<std::mutex> lk(io.m);
                    io.items.push_back({1, o});
                }
                io.cv.notify_all();
            }
            pairs++;
            heldPairs++;
        }
        else if (havePrev && (nr.nvof || nr.gmfss || nr.fruc || oa.dlssg))
        {
            // the nvof model: one Optical Flow pair, then the N-1 interior tweens at k/N one
            // splat at a time, each packed like a RIFE tween (dOutRaw slot 0 is reused in
            // stream order: the D2H of the previous tween is enqueued before the next pack).
            // GMFSS: the pair ran above, then one chain per tween into dGmF (clamped fp32).
            // fruc: the pair ran above, then one bridge warp per tween into dFrOut (x / 255).
            if (nr.nvof && !nativeNvofPair(nr, nr.dX, dCur))
            {
                io.setFail("nvof pair failed");
                failed = true;
                break;
            }
            if (nr.fruc && nT - (std::min)(dropLeft, nT) > 1)
            {
                // the tweens the loop below generates (resume drops the first dropLeft), in its
                // order and with its exact t, for the parallel FRUC instances
                double tw[64];
                uint32_t n = 0;
                for (int j = (std::min)(dropLeft, nT); j < nT && n < 64; j++)
                    tw[n++] = fpsMode ? fr[j] : (double)((float)(j + 1) / (float)multi);
                nativeFrucPlan(nr, tw, n);
            }
            for (int j = 0; j < nT && !failed; j++)
            {
                // resume: the banked slots are never generated (python's fracs[skip:]); DLSS 4.5
                // still got the whole pair above, its temporal stream stays intact (tws[-take:])
                if (dropOne())
                    continue;
                const float t = fpsMode ? (float)fr[j] : (float)(j + 1) / (float)multi;
                if (oa.dlssg)
                {
                    // DLSS 4.5: the pair's j-th generated frame, RGBA8 (hdr10: R10G10B10A2) back to the planes
                    void* au[] = {&dDgRgba, &nr.pw, &nr.ph, &dDgF};
                    void* au10[] = {&dDgRgba, &nr.pw, &nr.ph, &dgHlg, &dDgF};
                    if (cudaMemcpyAsync(dDgRgba, hDgOut + (size_t)j * dgBytes, dgBytes, cudaMemcpyHostToDevice, st) !=
                            cudaSuccess ||
                        cuLaunchKernel(dgc.hdr10 ? nr.fUnpackR10 : nr.fUnpackRgba, (nr.pw + 15) / 16, (nr.ph + 15) / 16,
                                       1, 16, 16, 1, 0, (CUstream)st, dgc.hdr10 ? au10 : au, nullptr) != CUDA_SUCCESS)
                    {
                        io.setFail("dlss tween upload failed");
                        failed = true;
                        break;
                    }
                }
                else if (nr.gmfss
                             ? !nativeGmfssTween(nr, t)
                             : (nr.fruc ? !nativeFrucTween(nr, fpsMode ? fr[j] : (double)t) : !nativeNvofTween(nr, t)))
                {
                    io.setFail(nr.gmfss ? "gmfss tween failed" : (nr.fruc ? "fruc tween failed" : "nvof tween failed"));
                    failed = true;
                    break;
                }
                int o = -1;
                {
                    std::unique_lock<std::mutex> lk(io.m);
                    io.cv.wait(lk, [&] { return !io.outFree.empty() || io.fail; });
                    if (io.fail)
                    {
                        failed = true;
                        break;
                    }
                    o = io.outFree.front();
                    io.outFree.pop_front();
                }
                const float* src = oa.dlssg ? dDgF : (nr.gmfss ? nr.dGmF : (nr.fruc ? nr.dFrOut : nr.dNvOut));
                uint8_t* dO = dOutRaw;
                if (passes)
                {
                    if (!nativeOfflineEmit(nr, src, ps, rs, dO, outIs16))
                    {
                        io.setFail("passes (nvof / gmfss) failed");
                        failed = true;
                        break;
                    }
                }
                else
                {
                    int f32 = 0;
                    void* a[] = {(void*)&src, &f32, (void*)&ps, (void*)&rs, &w, &h, &dO};
                    if (cuLaunchKernel(outIs16 ? nr.fPackOutRaw16 : nr.fPackOutRaw8, (w + 15) / 16, (h + 15) / 16, 1,
                                       16, 16, 1, 0, (CUstream)st, a, nullptr) != CUDA_SUCCESS)
                    {
                        io.setFail("packOutRaw (nvof) launch failed");
                        failed = true;
                        break;
                    }
                }
                if (cudaMemcpyAsync(io.hOut[o], dO, frameBytesOut, cudaMemcpyDeviceToHost, st) != cudaSuccess)
                {
                    io.setFail("D2H copy (nvof) failed");
                    failed = true;
                    break;
                }
                cudaEventRecord(io.outEv[o], st);
                drLastOut = dO; // --fps: the closing slot repeats the last output
                {
                    std::lock_guard<std::mutex> lk(io.m);
                    io.items.push_back({1, o});
                }
                io.cv.notify_all();
            }
            pairs++;
        }
        else if (havePrev)
        {
            // the N-1 interior tweens at k/N, in chunks of at most batchMax through the batched
            // engine; --fps: the pair's slots one per enqueue, and a slot at t <= 0 is the left
            // frame itself with no engine call (rife_backend.inference returns a)
            const int bm = fpsMode ? 1 : batchMax;
            for (int base = 0; base < nT && !failed; base += bm)
            {
                const int len = (nT - base) < bm ? (nT - base) : bm;
                const bool left = fpsMode && fr[base] <= 0.0;
                for (int j = 0; j < len && !left; j++)
                {
                    const float t = fpsMode ? (float)fr[base + j] : (float)(base + j + 1) / (float)multi;
                    if (!nativeFillT(nr, (size_t)j, t, mplane, st))
                    {
                        io.setFail("timestep fill failed");
                        failed = true;
                        break;
                    }
                }
                if (failed)
                    break;
                if (!left)
                {
                    const int modelH = nr.mph ? nr.mph : nr.ph, modelW = nr.mph ? nr.mpw : nr.pw;
                    nvinfer1::Dims4 dx{1, 6, modelH, modelW};
                    nvinfer1::Dims4 dtst{len, 1, modelH, modelW};
                    nvinfer1::Dims4 df{1, 16, modelH, modelW};
                    if (!nr.ctxIf->setInputShape("x", dx) || !nr.ctxIf->setInputShape("timestep", dtst) ||
                        !nr.ctxIf->setInputShape("f0", df) || !nr.ctxIf->setInputShape("f1", df))
                    {
                        io.setFail("IFNet setInputShape rejected");
                        failed = true;
                        break;
                    }
                    const void* xin =
                        nr.mph ? (const void*)nr.dM : (nr.xHalf ? (const void*)nr.dXh : (const void*)nr.dX);
                    nr.ctxIf->setTensorAddress("x", const_cast<void*>(xin));
                    nr.ctxIf->setTensorAddress("timestep", nr.dT);
                    nr.ctxIf->setTensorAddress("f0", nr.dF[nr.fCur ^ 1]);
                    nr.ctxIf->setTensorAddress("f1", nr.dF[nr.fCur]);
                    nr.ctxIf->setTensorAddress("flow", nr.dFlow);
                    nr.ctxIf->setTensorAddress("mask", nr.dMask);
                    if (!nr.ctxIf->enqueueV3(st))
                    {
                        io.setFail("IFNet enqueueV3 returned false");
                        failed = true;
                        break;
                    }
                    // the tweens: RIFE's last step on the pictures (RIFE's two domains: dX at the working
                    // size, whatever the passes before made of them), else on the frames the IFNet read
                    const bool picHalf = !nr.mph && nr.xHalf;
                    if (!nativeRifeBlend(nr, picHalf ? (const void*)nr.dXh : (const void*)nr.dX, (int)picHalf, len, st))
                    {
                        io.setFail("rifeBlend launch failed");
                        failed = true;
                        break;
                    }
                }
                for (int j = 0; j < len && !failed; j++)
                {
                    // resume: a banked slot of the first pair is not emitted (the batch still
                    // ran whole, so the kept slots are the uninterrupted render's own bytes)
                    if (dropOne())
                        continue;
                    int o = -1;
                    {
                        std::unique_lock<std::mutex> lk(io.m);
                        io.cv.wait(lk, [&] { return !io.outFree.empty() || io.fail; });
                        if (io.fail)
                        {
                            failed = true;
                            break;
                        }
                        o = io.outFree.front();
                        io.outFree.pop_front();
                    }
                    const void* src = left ? (const void*)nr.dX : mergedAt(nr, j, plane);
                    int srcHalf = left ? 0 : (int)nr.outHalf;
                    uint8_t* dO = dOutRaw + (size_t)j * frameBytesOut;
                    if (passes)
                    {
                        if (!nativeOfflineEmit(nr, src, ps, rs, dO, outIs16, srcHalf))
                        {
                            io.setFail("passes failed");
                            failed = true;
                            break;
                        }
                    }
                    else
                    {
                        void* a[] = {(void*)&src, &srcHalf, (void*)&ps, (void*)&rs, &w, &h, &dO};
                        if (cuLaunchKernel(outIs16 ? nr.fPackOutRaw16 : nr.fPackOutRaw8, (w + 15) / 16, (h + 15) / 16,
                                           1, 16, 16, 1, 0, (CUstream)st, a, nullptr) != CUDA_SUCCESS)
                        {
                            io.setFail("packOutRaw launch failed");
                            failed = true;
                            break;
                        }
                    }
                    if (cudaMemcpyAsync(io.hOut[o], dO, frameBytesOut, cudaMemcpyDeviceToHost, st) != cudaSuccess)
                    {
                        io.setFail("D2H copy failed");
                        failed = true;
                        break;
                    }
                    cudaEventRecord(io.outEv[o], st);
                    {
                        std::lock_guard<std::mutex> lk(io.m);
                        io.items.push_back({1, o});
                    }
                    io.cv.notify_all();
                    drLastOut = dO; // --fps: the closing slot repeats the last output
                }
            }
            pairs++;
        }
        if (failed)
            break;
        if (fpsMode)
        {
            // --fps: no real frame passes through (every output is an interior slot), so the
            // input slot is released here (the reader syncs its H2D event before reusing it)
            {
                std::lock_guard<std::mutex> lk(io.m);
                io.inWritten[slot] = 1;
            }
            io.cv.notify_all();
            havePrev = true;
            continue;
        }
        if (dropOne())
        {
            // resume: this real frame is banked (ongrid_loop's resume_active head); nothing
            // goes out, so the input slot is released here
            {
                std::lock_guard<std::mutex> lk(io.m);
                io.inWritten[slot] = 1;
            }
            io.cv.notify_all();
            realOutFresh = false;
            havePrev = !nr.noEngine;
            continue;
        }
        if (sameFmt && !passes)
        {
            {
                std::lock_guard<std::mutex> lk(io.m);
                io.items.push_back({0, slot});
            }
        }
        else
        {
            // rgb24 in, rgb48le out: render.py's to_tensor / to_bytes round trip of a real frame
            // is exactly x * 257 per sample; done on the GPU from the staged input, then written
            // like a tween. The input slot needs no writer release then.
            int o = -1;
            {
                std::unique_lock<std::mutex> lk(io.m);
                io.cv.wait(lk, [&] { return !io.outFree.empty() || io.fail; });
                if (io.fail)
                {
                    failed = true;
                    break;
                }
                o = io.outFree.front();
                io.outFree.pop_front();
            }
            uint8_t* dO = dOutRaw + (size_t)batchMax * frameBytesOut;
            if (passes)
            {
                // the real frame through the pass chain (python's to_bytes on every frame)
                if (!nativeOfflineEmit(nr, dCur, ps, rs, dO, outIs16))
                {
                    io.setFail("passes (real) failed");
                    failed = true;
                    break;
                }
                realOutFresh = true;
            }
            else
            {
                int n = (int)((size_t)w * h * 3);
                void* a[] = {&dR, &n, &dO};
                if (cuLaunchKernel(nr.fExpand8to16, (n + 255) / 256, 1, 1, 256, 1, 1, 0, (CUstream)st, a, nullptr) !=
                    CUDA_SUCCESS)
                {
                    io.setFail("expand8to16 launch failed");
                    failed = true;
                    break;
                }
            }
            if (cudaMemcpyAsync(io.hOut[o], dO, frameBytesOut, cudaMemcpyDeviceToHost, st) != cudaSuccess)
            {
                io.setFail("D2H copy (real) failed");
                failed = true;
                break;
            }
            cudaEventRecord(io.outEv[o], st);
            {
                std::lock_guard<std::mutex> lk(io.m);
                io.items.push_back({1, o});
                io.inWritten[slot] = 1;
            }
        }
        io.cv.notify_all();
        havePrev = !nr.noEngine; // --no-interp: never a pair, every frame is a real frame
    }
    if (!failed && !io.fail && nr.drba && nr.drFid >= 1)
    {
        // EOF (drba_loop's tail): the last pair with a virtual next frame id (never read), its
        // f < 0.5 from window N-1 side +1 (plain when N-1 is the head), its f >= 0.5 plain
        // pair RIFE; then the closing slot, or the lone frame of a one-frame source
        if (nr.drFid >= 2)
        {
            nr.drFid++;
            const bool ok = drPair(drLastStatic, true);
            nr.drFid--;
            if (ok && !drLastOut)
            {
                // --fps: no pair had a slot (closing_slot's last_out None): the last frame once,
                // never on a resumed run (that frame is banked already)
                if (!oa.resumed && (!drEmit(drbaFrame(nr, nr.drFid), dHeldOut) || !drSend(dHeldOut, false)))
                    failed = true;
            }
            else
            {
                const int64_t nClose = closingCount(nr.drFid);
                for (int64_t j = 0; ok && j < nClose; j++)
                    if (!drSend(drLastOut, true))
                        break;
            }
            if (!ok)
                failed = true;
        }
        else if (!oa.resumed && (!drEmit(drbaFrame(nr, 1), dHeldOut) || !drSend(dHeldOut, false)))
            failed = true;
    }
    if (!failed && !io.fail && fpsMode && !nr.drba && nFrames >= 1)
    {
        // --fps EOF (fps_loop + closing_slot): the last source frame's interval holds the last
        // output; a source whose pairs made no slot (or a lone frame) sends its last frame once
        if (drLastOut)
        {
            const int64_t nClose = closingCount(nFrames);
            for (int64_t j = 0; j < nClose; j++)
                if (!drSend(drLastOut, true))
                {
                    failed = true;
                    break;
                }
        }
        else if (!oa.resumed && (!drEmit(nr.dX + 3 * plane, dHeldOut) || !drSend(dHeldOut, false)))
            failed = true;
    }
    const double loopS = (double)(nowQpc100() - tLoop) / 1e7;
    if (!failed && !io.fail)
    {
        std::lock_guard<std::mutex> lk(io.m);
        io.items.push_back({2, 0});
    }
    io.cv.notify_all();
    writer.join();
    {
        std::lock_guard<std::mutex> lk(io.m);
        if (!io.readerEof)
            io.fail = true;
    }
    io.cv.notify_all();
    reader.join();
    if (oa.dlssg)
    {
        // EOF to the server (it logs "server done" and exits), then the watchdog
        dgEnd(dgc, false);
        dgc.closing = true;
        if (dgc.watch.joinable())
            dgc.watch.join();
    }
    cudaStreamSynchronize(st);
    if (namedPipes)
        FlushFileBuffers(io.hStdout); // wait until the encoder has read everything
    CloseHandle(io.hStdout);          // EOF to the encoder
    bool ok = !failed && !io.fail;
    if (nr.rtxHdr)
    {
        // the last frame's statistics, then the file the finalize reads (HDR10 clli, DV, HDR10+)
        nativeOfflineThdrDrain(nr);
        LOG("offline: TrueHDR %.2f ms mean, %.2f ms max over %llu frames, MaxCLL %.1f MaxFALL %.1f nits\n",
            nr.thdrN ? nr.thdrMs / (double)nr.thdrN : 0.0, nr.thdrMaxMs, (unsigned long long)nr.thdrN, thdrAcc.cll,
            thdrAcc.fall);
        if (ok && !oa.hdrStatsW.empty() && !thdrAcc.writeJson(oa.hdrStatsW))
        {
            LOG("offline: cannot write the HDR statistics %s\n", wideToUtf8(oa.hdrStatsW).c_str());
            ok = false;
        }
    }
    nativeVideoMemoryNote(); // a render of three frames or fewer
    if (nr.nrHost)
    {
        LOG("offline: DLSS 5 %.2f ms mean, %.2f ms max over %llu frames (%s)\n",
            nr.nrN ? nr.nrMs / (double)nr.nrN : 0.0, nr.nrMaxMs, (unsigned long long)nr.nrN,
            nr.nrZeroCopy ? "host submit, zero-copy" : "render and staging");
        if (nr.nrReused)
            LOG("offline: DLSS 5 reused its last output on %llu frames identical to the previous one\n",
                (unsigned long long)nr.nrReused);
        cudaDeviceSynchronize(); // the last fence wait has run before its semaphore goes
        nativeNrMotionFree(nr);
        nativeNrReleaseImports(nr);
        nr.nrHost->abandon(); // no NGX release chain (it faults); the process exits after this item
        cudaFree(nr.dNrIo);
        cudaFree(nr.dRawPrev);
        cudaFreeHost(nr.hNrIn);
        cudaFreeHost(nr.hNrOut);
    }
    LOG("offline: %llu frames in, %llu out, %llu pairs, %.2f s, %.2f ms per pair%s\n", (unsigned long long)io.framesIn,
        (unsigned long long)io.framesOut, (unsigned long long)pairs, loopS,
        pairs ? loopS * 1000.0 / (double)pairs : 0.0, ok ? "" : " (FAILED)");
    if (heldPairs)
        LOG("static pairs held: %llu\n", (unsigned long long)heldPairs);
    for (int i = 0; i < io.nIn; i++)
    {
        cudaEventDestroy(io.h2dEv[i]);
        cudaFreeHost(io.hIn[i]);
    }
    for (int i = 0; i < io.nOut; i++)
    {
        cudaEventDestroy(io.outEv[i]);
        cudaFreeHost(io.hOut[i]);
    }
    cudaFree(dRaw[0]);
    cudaFree(dRaw[1]);
    cudaFree(dOutRaw);
    if (oa.dlssg)
    {
        cudaFree(dDgRgba);
        cudaFree(dDgF);
        cudaFreeHost(hDgIn[0]);
        cudaFreeHost(hDgIn[1]);
        cudaFreeHost(hDgOut);
    }
    nativeFree(nr);
    return ok ? 0 : failRc;
}

// a live session's parsed command line (the flags that are not globals)
struct LiveArgs
{
    const wchar_t* needle = nullptr; // --live "title substring"
    HWND targetOverride = nullptr;   // --hwnd, or the foreground pick of --fg
    HWND excludeHwnd = nullptr;      // --fg --exclude: the caller's own window
    int genFrames = 1;
    int diagSecs = 0;
    bool vsync = false;
    bool clickthrough = true;
    bool park = false;
    bool byFg = false;
    bool byHwnd = false;
};
static int parseLiveArgs(int argc, wchar_t** argv, LiveArgs& la);
static int runLiveArgs(LiveArgs& la);
static int residentMain(LiveArgs first);

int wmain(int argc, wchar_t** argv)
{
    setvbuf(stderr, nullptr, _IONBF, 0);
    g_verbose = GetEnvironmentVariableW(L"DLSSG_VERBOSE", nullptr, 0) != 0; // server backends skip slInitCommon
    {
        wchar_t tv[8]{};
        g_teardownTrace = GetEnvironmentVariableW(L"SMV_LIVE_TEARDOWN_TRACE", tv, 8) && tv[0] == L'1';
        wchar_t sv[8]{};
        g_staticHold = !(GetEnvironmentVariableW(L"SMV_NO_STATIC_HOLD", sv, 8) && sv[0] == L'1');
    }
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    CHECK_HR(CoInitializeEx(nullptr, COINIT_MULTITHREADED));

    // the app runs this after it killed a live host (or saw it crash) while Fill held the mouse
    if (argc >= 2 && wcscmp(argv[1], L"--restore-mouse") == 0)
    {
        if (!fillMouseRestore())
            LOG("restore-mouse: nothing to restore\n");
        return 0;
    }

    if (argc >= 2 && wcscmp(argv[1], L"--testsrc") == 0)
    {
        bool onScreen = false;
        for (int i = 2; i < argc; i++)
        {
            if (wcscmp(argv[i], L"--onscreen") == 0)
                onScreen = true;
            else if (wcscmp(argv[i], L"--pan") == 0)
                g_tsPan = true;
            else if (wcscmp(argv[i], L"cycle") == 0)
            {
                g_tsCycling = true;
                g_tsCycleTick = GetTickCount64();
                g_tsInterval = g_tsCycle[0];
            }
            else
                g_tsInterval = (UINT)_wtoi(argv[i]);
        }
        if (g_tsInterval < 5 || g_tsInterval > 2000)
            g_tsInterval = 100;
        return runTestSrc(onScreen);
    }

    if (argc >= 2 && wcscmp(argv[1], L"--synth") == 0)
    {
        int genFrames = (argc >= 3) ? _wtoi(argv[2]) : 1;
        if (genFrames < 1 || genFrames > 5)
            genFrames = 1;
        bool vsync = (argc >= 4 && wcscmp(argv[3], L"--vsync") == 0);
        if (slInitCommon())
            return 1;
        return runSynth(genFrames, vsync);
    }

    if (argc >= 2 && wcscmp(argv[1], L"--list") == 0)
        return runList();

    // --lookup-probe script backend outW outH capW capH imgScale: print the handoff lines the
    // host's own warm engine lookup answers, or `MISS`; stdout only. A debug entry point: a
    // change to the lookup's naming or geometry is gated by dumping this over a config matrix
    // before and after
    if (argc >= 9 && wcscmp(argv[1], L"--lookup-probe") == 0)
    {
        W = (uint32_t)_wtoi(argv[4]);
        H = (uint32_t)_wtoi(argv[5]);
        if (!parseLiveScale(argv[8]))
            g_flowScale = 1.0;
        std::vector<std::string> lines;
        const bool ok = nativeLocalHandoff(argv[2], argv[3], (uint32_t)_wtoi(argv[6]), (uint32_t)_wtoi(argv[7]), lines);
        for (const auto& l : lines)
            printf("%s\n", l.c_str());
        if (!ok)
            printf("MISS\n");
        return 0;
    }

    // --probe 0xHWND [frames] [dump.raw]: read a window's real FP16 scRGB content (HDR
    // diagnostic); with a dump path, append every frame as tightly packed rgbaf16le raws
    if (argc >= 3 && wcscmp(argv[1], L"--probe") == 0)
    {
        HWND h = (HWND)(uintptr_t)wcstoull(argv[2], nullptr, 16);
        int frames = (argc >= 4) ? _wtoi(argv[3]) : 3;
        const wchar_t* dumpPath = (argc >= 5) ? argv[4] : nullptr;
        const int cap = dumpPath ? 2000 : 60;
        return runProbe(h, (frames < 1 || frames > cap) ? 3 : frames, dumpPath);
    }

    if (argc >= 2 && wcscmp(argv[1], L"--offline") == 0)
    {
        wchar_t gv[8]{};
        g_offlineGraph = GetEnvironmentVariableW(L"SMV_OFFLINE_GRAPH", gv, 8) && gv[0] == L'1';
        const int orc = runOffline(argc, argv);
        // DLSS 5 ran here: NGX's process-detach teardown faults after main returns,
        // so leave the way the live host does
        if (g_nrAttempted)
        {
            fflush(stderr);
            ExitProcess((UINT)orc);
        }
        return orc;
    }
    const bool byLive = argc >= 3 && wcscmp(argv[1], L"--live") == 0;
    const bool byHwnd = argc >= 3 && wcscmp(argv[1], L"--hwnd") == 0;
    const bool byFg = argc >= 2 && wcscmp(argv[1], L"--fg") == 0;
    if (byLive || byHwnd || byFg)
    {
        LiveArgs la;
        const int prc = parseLiveArgs(argc, argv, la);
        if (prc)
            return prc;
        if (!g_resident)
            return runLiveArgs(la);
        return residentMain(la);
    }

    LOG("usage: smv-live.exe --live \"title substring\" [--gen N] [--vsync] [--no-clickthrough]\n"
        "                          [--no-hud | --no-hud-latency] [--diag S] [--resident] [--no-fill-mouse]\n"
        "       smv-live.exe --hwnd 0xHWND [same flags]\n"
        "       smv-live.exe --fg [--exclude 0xHWND] [same flags]   (overlay the foreground window)\n"
        "       smv-live.exe --list\n"
        "       smv-live.exe --restore-mouse   (the pointer speed and clip a killed Fill session held)\n"
        "       smv-live.exe --testsrc [ms|cycle] [--pan] [--onscreen]   (parked with 1 px on the desktop unless\n"
        "                    --onscreen; --pan = a texture moving 4 px right per tick instead of the square)\n");
    return 1;
}

// ---------------------------------------------------------------- live session arguments

// every global a live session's command line or run can set, back to the defaults the
// process started with (the resident host parses a fresh command line per session)
static void resetSessionGlobals()
{
    W = 1920;
    H = 1080;
    g_backend = BK_DLSSG;
    g_serverBackend.clear();
    g_modelLabel.clear();
    g_modelNote.clear();
    g_capRel = false;
    g_script.clear();
    g_flowScale = 1.0;
    g_dlssMode = 0;
    g_liveAutoFit = 0;
    g_liveAutoFloor = 0;
    g_noHud = g_noHudLat = false;
    g_noFillMouse = false;
    g_sharpen = 0.0;
    g_rtxVsr = false;
    g_upscaleH = 0;
    g_restore = false;
    g_dlssnr = false;
    g_nrStructure = 1.0;
    g_nrTone = 1.0;
    g_nrStyle = 1;
    g_nrPasses = 1;
    g_nrNative = false;
    g_liveNrCuda = false;
    g_liveNrReset.store(false);
    g_rtxHdr = false;
    wcscpy_s(g_hdrColor, L"vivid");
    g_hdrSat = 0;
    g_hdrCon = 100;
    g_hdrVib = 0.0;
    g_hdrSb = 0.0;
    g_sdrWhite = 240.0;
    g_vramBytes = 0;
    g_noAdapt = false;
    g_noGpuFit = false;
    g_genExplicit = false;
    g_targetFps = 0;
    g_monitor = false;
    g_fill = false;
    g_hdr = false;
    g_stopReq.store(false);
    g_sessionClean = false;
    g_targetHwnd = nullptr; // runLive resolves it again; a session that never gets there reports none
    g_staticHeld = 0;       // per session; g_staticHold is the process-wide env lever, kept
}

// argv[1] is --live / --hwnd / --fg, the rest are the session flags (main.ts builds them,
// the resident "start" command carries the same tokens). 0 = parsed, else the exit code.
static int parseLiveArgs(int argc, wchar_t** argv, LiveArgs& la)
{
    const bool byLive = wcscmp(argv[1], L"--live") == 0;
    la.byHwnd = wcscmp(argv[1], L"--hwnd") == 0;
    la.byFg = wcscmp(argv[1], L"--fg") == 0;
    if ((byLive || la.byHwnd) && argc < 3)
    {
        LOG("%ls needs a value\n", argv[1]);
        return 1;
    }
    la.needle = byLive ? argv[2] : nullptr;
    la.targetOverride = la.byHwnd ? (HWND)(uintptr_t)wcstoull(argv[2], nullptr, 16) : nullptr;
    la.excludeHwnd = nullptr;
    la.genFrames = 1;
    la.vsync = false; // pacer handles timing; tearing allowed for VRR
    la.clickthrough = true;
    la.park = false;
    la.diagSecs = 0;
    int& genFrames = la.genFrames;
    for (int i = la.byFg ? 2 : 3; i < argc; i++)
    {
        if (wcscmp(argv[i], L"--gen") == 0 && i + 1 < argc)
        {
            genFrames = _wtoi(argv[++i]);
            g_genExplicit = true;
        }
        else if (wcscmp(argv[i], L"--vsync") == 0)
            la.vsync = true;
        else if (wcscmp(argv[i], L"--no-clickthrough") == 0)
            la.clickthrough = false;
        else if (wcscmp(argv[i], L"--park") == 0)
            la.park = true;
        else if (wcscmp(argv[i], L"--native") == 0)
        {
        } // accepted and ignored: the native host is the only route
        else if (wcscmp(argv[i], L"--resident") == 0)
            g_resident = true;
        else if (wcscmp(argv[i], L"--no-hud") == 0)
            g_noHud = true;
        else if (wcscmp(argv[i], L"--no-fill-mouse") == 0)
            g_noFillMouse = true;
        else if (wcscmp(argv[i], L"--no-hud-latency") == 0)
            g_noHudLat = true;
        else if (wcscmp(argv[i], L"--no-adapt") == 0)
            g_noAdapt = true;
        else if (wcscmp(argv[i], L"--no-gpu-fit") == 0)
            g_noGpuFit = true;
        else if (wcscmp(argv[i], L"--target") == 0 && i + 1 < argc)
            g_targetFps = _wtoi(argv[++i]);
        else if (wcscmp(argv[i], L"--diag") == 0 && i + 1 < argc)
            la.diagSecs = _wtoi(argv[++i]);
        else if (wcscmp(argv[i], L"--exclude") == 0 && i + 1 < argc)
            la.excludeHwnd = (HWND)(uintptr_t)wcstoull(argv[++i], nullptr, 16);
        else if (wcscmp(argv[i], L"--backend") == 0 && i + 1 < argc)
        {
            ++i;
            if (wcscmp(argv[i], L"dlssg") == 0)
                g_backend = BK_DLSSG;
            else if (wcscmp(argv[i], L"identity") == 0)
                g_backend = BK_IDENTITY;
            else
            {
                g_backend = BK_SERVER;
                g_serverBackend = argv[i];
            } // resolved by live_server.py
        }
        else if (wcscmp(argv[i], L"--python") == 0 && i + 1 < argc)
            ++i; // accepted and ignored: no python process is ever started
        else if (wcscmp(argv[i], L"--script") == 0 && i + 1 < argc)
            g_script = argv[++i];
        else if (wcscmp(argv[i], L"--label") == 0 && i + 1 < argc)
            g_modelLabel = argv[++i];
        else if (wcscmp(argv[i], L"--note") == 0 && i + 1 < argc)
            g_modelNote = argv[++i];
        else if (wcscmp(argv[i], L"--sharpen") == 0 && i + 1 < argc)
            g_sharpen = _wtof(argv[++i]);
        else if (wcscmp(argv[i], L"--rtx-vsr") == 0)
            g_rtxVsr = true;
        else if (wcscmp(argv[i], L"--upscale") == 0 && i + 1 < argc)
            g_upscaleH = _wtoi(argv[++i]);
        else if (wcscmp(argv[i], L"--restore") == 0)
            g_restore = true;
        else if (wcscmp(argv[i], L"--dlssnr") == 0)
            g_dlssnr = true;
        else if (wcscmp(argv[i], L"--nr-structure") == 0 && i + 1 < argc)
            g_nrStructure = _wtof(argv[++i]);
        else if (wcscmp(argv[i], L"--nr-tone") == 0 && i + 1 < argc)
            g_nrTone = _wtof(argv[++i]);
        else if (wcscmp(argv[i], L"--nr-style") == 0 && i + 1 < argc)
            g_nrStyle = _wtoi(argv[++i]);
        else if (wcscmp(argv[i], L"--nr-passes") == 0 && i + 1 < argc)
            g_nrPasses = _wtoi(argv[++i]);
        else if (wcscmp(argv[i], L"--rtx-hdr") == 0)
            g_rtxHdr = true;
        else if (wcscmp(argv[i], L"--hdr-color") == 0 && i + 1 < argc)
            wcsncpy_s(g_hdrColor, argv[++i], _TRUNCATE);
        else if (wcscmp(argv[i], L"--hdr-saturation") == 0 && i + 1 < argc)
            g_hdrSat = _wtoi(argv[++i]);
        else if (wcscmp(argv[i], L"--hdr-contrast") == 0 && i + 1 < argc)
            g_hdrCon = _wtoi(argv[++i]);
        else if (wcscmp(argv[i], L"--hdr-vibrance") == 0 && i + 1 < argc)
            g_hdrVib = _wtof(argv[++i]);
        else if (wcscmp(argv[i], L"--hdr-satboost") == 0 && i + 1 < argc)
            g_hdrSb = _wtof(argv[++i]);
        else if (wcscmp(argv[i], L"--scale") == 0 && i + 1 < argc)
        {
            if (!parseLiveScale(argv[++i]))
            {
                LOG("--scale must be 0.01..1.0 or a DLSS mode (auto, dlaa, quality, balanced, performance, ultra)\n");
                return 1;
            }
        }
        else if (wcscmp(argv[i], L"--auto-floor") == 0 && i + 1 < argc)
        {
            ++i;
            for (int m = 2; m <= 6; m++)
                if (_wcsicmp(argv[i], kDlssModeName[m - 1]) == 0)
                    g_liveAutoFloor = m;
            if (!g_liveAutoFloor)
            {
                LOG("--auto-floor must be a DLSS mode (dlaa, quality, balanced, performance, ultra)\n");
                return 1;
            }
        }
        else if (wcscmp(argv[i], L"--fit") == 0 && i + 1 < argc)
        {
            ++i;
            if (wcscmp(argv[i], L"fill") == 0)
                g_fill = true;
            else if (wcscmp(argv[i], L"monitor") == 0)
                g_monitor = true;
            else if (wcscmp(argv[i], L"window") == 0)
                g_fill = false;
            else
            {
                LOG("unknown --fit \"%s\"\n", wideToUtf8(argv[i]).c_str());
                return 1;
            }
        }
    }
    if (g_fill && g_backend != BK_SERVER)
    {
        LOG("--fit fill needs a server backend (e.g. --backend rife); DLSS-G cannot be resized\n");
        return 1;
    }
    // Server backends stream per-slot and size shm/outbuf from gen, so their only limit is
    // the MEMORY budget - which needs the presented resolution and so cannot be known here;
    // runLive clamps an oversized --gen once the slot size is known. DLSS-G is model-capped
    // at 5 generated frames (numFramesToGenerateMax); identity has no reason to go higher.
    const int genCap = (g_backend == BK_SERVER) ? INT_MAX : 5;
    if (genFrames < 1 || genFrames > genCap)
    {
        LOG("--gen must be 1..%d for this backend (got %d)\n", genCap, genFrames);
        return 1;
    }
    if (la.byHwnd && !la.targetOverride)
    {
        LOG("bad --hwnd value \"%s\"\n", wideToUtf8(argv[2]).c_str());
        return 1;
    }
    return 0;
}

// one session from parsed arguments: the foreground pick, the Streamline init, runLive
static int runLiveArgs(LiveArgs& la)
{
    if (la.byFg)
    {
        // hotkey mode: the target is whatever window the user is in right now. The exe
        // starts ~100ms after the keypress, before our overlay exists, so the foreground
        // window is still the user's. --exclude carries the caller's own window (the app),
        // so the hotkey pressed inside SMV itself politely refuses (exit 5).
        HWND fg = GetAncestor(GetForegroundWindow(), GA_ROOT);
        wchar_t t[512];
        if (!fg || fg == la.excludeHwnd || !candidateWindow(fg, GetCurrentProcessId(), t, 512))
        {
            LOG("the foreground window cannot be captured\n");
            return 5;
        }
        la.targetOverride = fg;
    }
    if (g_backend == BK_DLSSG && slInitCommon())
        return 1;
    return runLive(la.needle, la.targetOverride, la.genFrames, la.vsync, la.clickthrough, la.diagSecs, la.park);
}

// ---------------------------------------------------------------- resident host
//
// --resident: after a session ends the process stays alive with the engines loaded (g_res)
// and reads one command per line from stdin:
//   start<TAB>--hwnd<TAB>0x...<TAB>--backend<TAB>rife ...   the next session's argv, tab separated
//   stop                                                     end the running session (exit 0)
//   quit                                                     release everything and exit
// Every session prints "live session ended: exit N, host resident" when the process stays
// (main.ts reads that line the way it read the exit code before); a session the host cannot
// stay after (an error exit, DLSS-G, native DLSS 5, the TrueHDR bridge, no engines kept)
// exits the process with the session's code exactly as a non-resident run does. stdin EOF
// (the app is gone) and the idle limit (SMV_LIVE_RESIDENT_IDLE_S, default 600) exit too.
struct ResidentIo
{
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::string> lines; // every command but "stop"
    bool eof = false;
    void start()
    {
        std::thread([this] {
            HANDLE h = GetStdHandle(STD_INPUT_HANDLE);
            std::string line;
            char buf[512];
            for (;;)
            {
                DWORD got = 0;
                if (!ReadFile(h, buf, sizeof(buf), &got, nullptr) || !got)
                    break;
                for (DWORD i = 0; i < got; i++)
                {
                    const char c = buf[i];
                    if (c == '\r')
                        continue;
                    if (c != '\n')
                    {
                        line += c;
                        continue;
                    }
                    if (line == "stop")
                        g_stopReq.store(true); // reaches the running loop directly
                    else if (!line.empty())
                    {
                        std::lock_guard<std::mutex> lk(m);
                        lines.push_back(line);
                        cv.notify_all();
                    }
                    line.clear();
                }
            }
            {
                std::lock_guard<std::mutex> lk(m);
                eof = true;
            }
            cv.notify_all();
        }).detach(); // blocked in ReadFile for the life of the process, never joined
    }
    // false = stdin closed or idle limit reached
    bool next(std::string& out, int idleSecs)
    {
        std::unique_lock<std::mutex> lk(m);
        if (!cv.wait_for(lk, std::chrono::seconds(idleSecs), [&] { return !lines.empty() || eof; }))
        {
            LOG("live host: idle for %d s, exiting\n", idleSecs);
            return false;
        }
        if (lines.empty())
        {
            LOG("live host: stdin closed, exiting\n");
            return false;
        }
        out = std::move(lines.front());
        lines.pop_front();
        return true;
    }
};
static ResidentIo* g_rio = nullptr; // heap, never freed: its reader thread outlives main

static int residentMain(LiveArgs first)
{
    g_rio = new ResidentIo();
    g_rio->start();
    int idleSecs = 600;
    {
        wchar_t v[16]{};
        if (GetEnvironmentVariableW(L"SMV_LIVE_RESIDENT_IDLE_S", v, 16) && _wtoi(v) > 0)
            idleSecs = _wtoi(v);
    }
    auto freeAll = [&] {
        if (!g_res.rt && !g_res.cuMod)
            return;
        cudaSetDevice(g_res.dev);
        cudaFree(nullptr); // bind the device on this (main) thread before the TRT deletes
        residentFree();
    };
    LiveArgs la = first;
    std::vector<std::wstring> toks; // the current session's argv storage (la points into it)
    for (;;)
    {
        g_stopReq.store(false);
        g_resizeReq.store(false);
        g_sessionClean = false;
        const int rc = runLiveArgs(la);
        // exit 7 = the captured window closed, a clean end like a stop; exit 8 = Auto's GPU-time step (the app
        // starts the session again at the mode it names, like a resize's exit 4). Exit 5 (hotkey pressed inside
        // the app itself) and an exit 7 before the teardown (the window was gone at the start) never
        // started anything, stay for them too. What the host holds: the engines (g_res), or an engine
        // build a session left running in the background.
        const bool held = g_res.rt || nativeBuildPending();
        const bool stay = (g_sessionClean && (rc == 0 || rc == 4 || rc == 7 || rc == 8) && held && !g_nrAttempted &&
                           !g_rtxUsed && g_backend != BK_DLSSG) ||
                          ((rc == 5 || (rc == 7 && !g_sessionClean)) && held);
        if (!stay)
        {
            freeAll();
            return rc;
        }
        // the session's target rides along: hotkey (--fg) mode is the only mode where the app
        // does not already know the hwnd, and its exit 4 (resize) / exit 6 (stall) revive needs
        // it. The "target window:" line above carries it too, this one is the belt: a session
        // that ends without that line still revives.
        char tgt[40] = "";
        if (g_targetHwnd)
            snprintf(tgt, sizeof tgt, " target=0x%p", (void*)g_targetHwnd);
        LOG("live session ended: exit %d, host resident (%s%s kept, idle limit %d s)%s\n", rc,
            g_res.rt ? "engines" : "",
            nativeBuildPending() ? (g_res.rt ? " and a running engine build" : "a running engine build") : "", idleSecs,
            tgt);
        bool haveNext = false;
        while (!haveNext)
        {
            std::string line;
            if (!g_rio->next(line, idleSecs))
            {
                freeAll();
                return 0;
            }
            if (line == "quit")
            {
                LOG("live host: quit\n");
                freeAll();
                return 0;
            }
            if (line.rfind("start\t", 0) != 0)
            {
                LOG("live host: unknown command \"%s\"\n", line.c_str());
                continue;
            }
            toks.clear();
            toks.push_back(L"smv-live.exe");
            size_t p = 6;
            while (p <= line.size())
            {
                size_t q = line.find('\t', p);
                if (q == std::string::npos)
                    q = line.size();
                toks.push_back(utf8ToWide(line.substr(p, q - p)));
                p = q + 1;
            }
            std::vector<wchar_t*> av;
            for (auto& t : toks)
                av.push_back(&t[0]);
            resetSessionGlobals();
            g_resident = true; // resetSessionGlobals does not touch it, the flag is process-wide
            LiveArgs nla;
            const bool byLive = av.size() >= 2 && wcscmp(av[1], L"--live") == 0;
            const bool byHwnd = av.size() >= 2 && wcscmp(av[1], L"--hwnd") == 0;
            const bool byFg = av.size() >= 2 && wcscmp(av[1], L"--fg") == 0;
            const int prc = (byLive || byHwnd || byFg) ? parseLiveArgs((int)av.size(), av.data(), nla) : 1;
            if (prc)
            {
                // main.ts treats this line as the session's end (no session ran)
                LOG("live session ended: exit %d, host resident (bad start command)\n", prc);
                continue;
            }
            la = nla;
            haveNext = true;
        }
    }
}

// ---------------------------------------------------------------- resident offline host
//
// smv-live.exe --offline --resident --pipe NAME [--script s --python p]: the offline render host
// stays alive between queue items with the TensorRT runtime, the engines, the JIT cache and the
// kernel module loaded (g_res, the live resident host's cache), so a later render of the same
// size and multiplier pays only the execution contexts instead of the process start plus the
// engine load. render.py spawns it detached (it outlives the render) and finds it again through
// the control pipe \\.\pipe\NAME, one client at a time, one line per command:
//   start<TAB>--w<TAB>1920<TAB>...   one item's flags (the --offline command line minus --offline)
//   <- "offline pipes ready"         the host created \\.\pipe\NAME-in and \\.\pipe\NAME-out
//   go                               the client opened both and started its ffmpegs on them
//   <- every log line of the item    (PROGRESS, OUTFRAMES, ... exactly the one-shot exe's stderr)
//   <- "offline done: exit N"        the item's exit code; the host disconnects and accepts again
//   quit                             release everything and exit
// A client that vanishes mid item ends that item through the pipe errors, the host stays. Idle
// for SMV_OFFLINE_RESIDENT_IDLE_S seconds (default 600) = a timer thread connects to the host's
// own pipe and writes "idle", so the blocking accept needs no cancel.
static bool pipeReadLine(HANDLE h, std::string& line)
{
    line.clear();
    for (;;)
    {
        char c = 0;
        DWORD got = 0;
        if (!ReadFile(h, &c, 1, &got, nullptr) || !got)
            return false;
        if (c == '\r')
            continue;
        if (c == '\n')
            return true;
        line += c;
        if (line.size() > 65536)
            return false;
    }
}

static int offlineResidentMain(const OfflineArgs& base)
{
    wchar_t exePath[MAX_PATH];
    GetModuleFileNameW(nullptr, exePath, MAX_PATH);
    std::wstring exeDir(exePath);
    exeDir.resize(exeDir.find_last_of(L'\\'));
    const std::wstring script = engineScript(exeDir);
    if (!nativeLoadDlls(script))
    {
        LOG("offline host: runtime DLLs unavailable\n");
        return 2;
    }
    g_resident = true; // nativeTrtInit / nativeBuildKernels / nativeFree keep g_res across items
    g_offline = true;
    int idleSecs = 600;
    {
        wchar_t v[16]{};
        if (GetEnvironmentVariableW(L"SMV_OFFLINE_RESIDENT_IDLE_S", v, 16) && _wtoi(v) > 0)
            idleSecs = _wtoi(v);
    }
    const std::wstring ctlName = L"\\\\.\\pipe\\" + base.pipeName;
    const std::wstring inName = ctlName + L"-in", outName = ctlName + L"-out";
    HANDLE ctl = CreateNamedPipeW(ctlName.c_str(), PIPE_ACCESS_DUPLEX, PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
                                  1, 1 << 16, 1 << 16, 0, nullptr);
    if (ctl == INVALID_HANDLE_VALUE)
    {
        LOG("offline host: cannot create pipe %s (%lu)\n", wideToUtf8(ctlName).c_str(), GetLastError());
        return 2;
    }
    LOG("offline host: resident on %s, idle limit %d s\n", wideToUtf8(ctlName).c_str(), idleSecs);

    std::atomic<int64_t> lastAct{nowQpc100()};
    std::atomic<bool> busy{false}, stopTimer{false};
    std::thread timer([&] {
        for (;;)
        {
            for (int i = 0; i < 10 && !stopTimer.load(); i++)
                Sleep(100);
            if (stopTimer.load())
                return;
            if (busy.load() || (nowQpc100() - lastAct.load()) < (int64_t)idleSecs * 10000000)
                continue;
            HANDLE h =
                CreateFileW(ctlName.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
            if (h == INVALID_HANDLE_VALUE)
                continue; // a client got in first: not idle after all
            DWORD wr = 0;
            WriteFile(h, "idle\n", 5, &wr, nullptr);
            CloseHandle(h);
            return;
        }
    });

    int rc = 0;
    std::vector<std::wstring> toks;
    for (;;)
    {
        if (!ConnectNamedPipe(ctl, nullptr) && GetLastError() != ERROR_PIPE_CONNECTED)
        {
            LOG("offline host: accept failed (%lu)\n", GetLastError());
            rc = 2;
            break;
        }
        busy.store(true);
        std::string line;
        bool quit = false;
        if (!pipeReadLine(ctl, line))
            LOG("offline host: client left before a command\n");
        else if (line == "quit")
        {
            LOG("offline host: quit\n");
            quit = true;
        }
        else if (line == "idle")
        {
            LOG("offline host: idle for %d s, exiting\n", idleSecs);
            quit = true;
        }
        else if (line.rfind("start\t", 0) != 0)
            LOG("offline host: unknown command \"%s\"\n", line.c_str());
        else
        {
            toks.clear();
            toks.push_back(L"smv-live.exe");
            toks.push_back(L"--offline");
            size_t p = 6;
            while (p <= line.size())
            {
                size_t q = line.find('\t', p);
                if (q == std::string::npos)
                    q = line.size();
                toks.push_back(utf8ToWide(line.substr(p, q - p)));
                p = q + 1;
            }
            std::vector<wchar_t*> av;
            for (auto& t : toks)
                av.push_back(&t[0]);
            g_flowScale = 1.0; // the globals an item's flags may set, which must not carry over
            g_dlssMode = 0;
            g_sharpen = 0.0;
            g_rtxVsr = false;
            g_restore = false;
            g_rtxHdr = false;
            wcscpy_s(g_hdrColor, L"vivid");
            g_hdrSat = 0;
            g_hdrCon = 100;
            g_hdrVib = 0.0;
            g_hdrSb = 0.0;
            g_dlssnr = false;
            g_nrStructure = 1.0;
            g_nrTone = 1.0;
            g_nrStyle = 1;
            g_nrPasses = 1;
            g_noGpuFit = false;
            g_logPipe = ctl; // from here every log line reaches the client too
            OfflineArgs oa;
            int irc = parseOfflineArgs((int)av.size(), av.data(), 2, oa);
            if (!irc && !offlineArgsValid(oa))
                irc = 1;
            if (!irc)
            {
                HANDLE hIn =
                    CreateNamedPipeW(inName.c_str(), PIPE_ACCESS_INBOUND,
                                     PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT, 1, 0, 4 << 20, 0, nullptr);
                HANDLE hOut =
                    CreateNamedPipeW(outName.c_str(), PIPE_ACCESS_OUTBOUND,
                                     PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT, 1, 4 << 20, 0, 0, nullptr);
                if (hIn == INVALID_HANDLE_VALUE || hOut == INVALID_HANDLE_VALUE)
                {
                    LOG("offline host: cannot create the frame pipes (%lu)\n", GetLastError());
                    irc = 2;
                }
                else
                {
                    LOG("offline pipes ready\n");
                    std::string go;
                    if (!pipeReadLine(ctl, go) || go != "go")
                    {
                        LOG("offline host: no go from the client, item dropped\n");
                        irc = 1;
                    }
                    else
                    {
                        // the client opened both ends before "go", so these return at once
                        // (ERROR_PIPE_CONNECTED = connected before the call, a good connection)
                        const bool okIn = ConnectNamedPipe(hIn, nullptr) || GetLastError() == ERROR_PIPE_CONNECTED;
                        const bool okOut = ConnectNamedPipe(hOut, nullptr) || GetLastError() == ERROR_PIPE_CONNECTED;
                        if (!okIn || !okOut)
                        {
                            LOG("offline host: frame pipe connect failed (%lu)\n", GetLastError());
                            irc = 2;
                        }
                        else
                        {
                            irc = runOfflineSession(oa, hIn, hOut, true); // closes hOut itself
                            hOut = INVALID_HANDLE_VALUE;
                        }
                    }
                }
                if (hIn && hIn != INVALID_HANDLE_VALUE)
                    CloseHandle(hIn);
                if (hOut && hOut != INVALID_HANDLE_VALUE)
                    CloseHandle(hOut);
            }
            LOG("offline done: exit %d\n", irc);
            g_logPipe = nullptr;
            // the RTX Video bridge (VSR) is single-instance NGX and cannot come back in this
            // process once shut down: the next render spawns a fresh host (the live rule)
            if (g_rtxUsed)
            {
                LOG("offline host: the RTX Video bridge ran here, exiting after this item\n");
                quit = true;
            }
            // DLSS 5: NGX has no teardown and a second feature in this process is
            // unproven, so the host ends and leaves through ExitProcess (wmain)
            if (g_nrAttempted)
            {
                LOG("offline host: DLSS 5 ran here, exiting after this item\n");
                quit = true;
            }
        }
        FlushFileBuffers(ctl);
        DisconnectNamedPipe(ctl);
        busy.store(false);
        lastAct.store(nowQpc100());
        if (quit)
            break;
    }
    stopTimer.store(true);
    timer.join();
    if (g_res.rt || g_res.cuMod)
    {
        cudaSetDevice(g_res.dev);
        cudaFree(nullptr); // bind the device on this thread before the TRT deletes
        residentFree();
    }
    CloseHandle(ctl);
    return rc;
}
