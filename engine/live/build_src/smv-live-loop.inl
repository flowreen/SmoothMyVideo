// part of smv-live.cpp (unity include, see the file map there); not a standalone translation unit
// ---------------------------------------------------------------- window finding

// shared capture-candidate filter: visible, titled, not ours, not cloaked, not a toolwindow
static bool candidateWindow(HWND h, DWORD ownPid, wchar_t* title, int cch)
{
    if (!IsWindowVisible(h))
        return false;
    DWORD pid = 0;
    GetWindowThreadProcessId(h, &pid);
    if (pid == ownPid)
        return false;
    BOOL cloaked = FALSE;
    DwmGetWindowAttribute(h, DWMWA_CLOAKED, &cloaked, sizeof(cloaked));
    if (cloaked)
        return false;
    if (GetWindowLongW(h, GWL_EXSTYLE) & WS_EX_TOOLWINDOW)
        return false;
    if (!GetWindowTextW(h, title, cch) || !title[0])
        return false;
    return true;
}

struct FindCtx
{
    const wchar_t* needle;
    DWORD ownPid;
    HWND found;
};

static BOOL CALLBACK enumProc(HWND h, LPARAM lp)
{
    FindCtx* c = (FindCtx*)lp;
    wchar_t title[512];
    if (!candidateWindow(h, c->ownPid, title, 512))
        return TRUE;
    if (!StrStrIW(title, c->needle))
        return TRUE;
    c->found = h;
    // UTF-8, never "%ls": the app parses the hwnd out of this line (hotkey mode has no other
    // source for it) and a title the C locale cannot convert would drop the whole line
    LOG("target window: hwnd=0x%p \"%s\"\n", (void*)h, wideToUtf8(title).c_str());
    return FALSE;
}

static HWND findTargetWindow(const wchar_t* needle)
{
    FindCtx c{needle, GetCurrentProcessId(), nullptr};
    EnumWindows(enumProc, (LPARAM)&c);
    return c.found;
}

static BOOL CALLBACK enumListProc(HWND h, LPARAM lp)
{
    wchar_t title[512];
    if (candidateWindow(h, (DWORD)lp, title, 512))
        wprintf(L"0x%llx\t%s\n", (unsigned long long)(uintptr_t)h, title);
    return TRUE;
}

// picker support: print "0xHWND<TAB>title" per capturable window to stdout (UTF-8)
static int runList()
{
    if (_setmode(_fileno(stdout), _O_U8TEXT) == -1)
    {
        LOG("list: stdout mode change failed\n");
        return 1;
    }
    EnumWindows(enumListProc, (LPARAM)GetCurrentProcessId());
    fflush(stdout);
    return 0;
}

static bool frameBounds(HWND h, RECT& r)
{
    return SUCCEEDED(DwmGetWindowAttribute(h, DWMWA_EXTENDED_FRAME_BOUNDS, &r, sizeof(r)));
}

// The live pause: the overlay sits topmost over the player, so it hides (and the smoothing pauses)
// when the player is minimized or on another virtual desktop, or when the window in front covers
// part of it; a window in front elsewhere (another monitor, beside the player) leaves the
// smoothing running. A frame that cannot be read pauses, as a foreground change always did.
static bool livePauseWanted(HWND target, HWND overlay)
{
    if (IsIconic(target))
        return true;
    BOOL cloaked = FALSE;
    DwmGetWindowAttribute(target, DWMWA_CLOAKED, &cloaked, sizeof(cloaked));
    if (cloaked)
        return true;
    const HWND fg = GetAncestor(GetForegroundWindow(), GA_ROOT);
    if (!fg || fg == target || fg == overlay)
        return false;
    if (!IsWindowVisible(fg) || IsIconic(fg))
        return false;
    RECT a{}, b{}, c{};
    if (!frameBounds(target, a) || !frameBounds(fg, b))
        return true;
    return IntersectRect(&c, &a, &b) != 0;
}

// Is Windows HDR ("Use HDR") active on the display the target window sits on? The
// output reports G2084 (PQ) as its current color space when HDR is on. Used to warn that today's
// 8-bit SDR capture clips HDR content, and later to switch on the HDR live pipeline.
static bool monitorIsHDR(HWND target)
{
    HMONITOR hmon = MonitorFromWindow(target, MONITOR_DEFAULTTONEAREST);
    ComPtr<IDXGIFactory1> f;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&f))))
        return false;
    ComPtr<IDXGIAdapter1> a;
    for (UINT i = 0; f->EnumAdapters1(i, &a) != DXGI_ERROR_NOT_FOUND; i++, a.Reset())
    {
        ComPtr<IDXGIOutput> o;
        for (UINT j = 0; a->EnumOutputs(j, &o) != DXGI_ERROR_NOT_FOUND; j++, o.Reset())
        {
            DXGI_OUTPUT_DESC od{};
            if (FAILED(o->GetDesc(&od)) || od.Monitor != hmon)
                continue;
            ComPtr<IDXGIOutput6> o6;
            DXGI_OUTPUT_DESC1 od1{};
            if (SUCCEEDED(o.As(&o6)) && SUCCEEDED(o6->GetDesc1(&od1)))
                return od1.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020;
            return false;
        }
    }
    return false;
}

// The SDR reference white (nits) of the display the target sits on - the level DWM
// composes SDR window content at inside an HDR desktop's scRGB FP16 capture ("SDR content
// brightness" slider; SDRWhiteLevel is fixed point, nits = level / 1000 * 80). The server divides
// the capture by it so TrueHDR sees a proper [0,1] SDR frame. Falls back to 240 (this dev box's
// measured desktop white) when the query fails.
static double sdrWhiteNits(HWND target)
{
    HMONITOR hmon = MonitorFromWindow(target, MONITOR_DEFAULTTONEAREST);
    MONITORINFOEXW mi{};
    mi.cbSize = sizeof(mi);
    if (!GetMonitorInfoW(hmon, &mi))
        return 240.0;
    UINT32 nPath = 0, nMode = 0;
    if (GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &nPath, &nMode) != ERROR_SUCCESS)
        return 240.0;
    std::vector<DISPLAYCONFIG_PATH_INFO> paths(nPath);
    std::vector<DISPLAYCONFIG_MODE_INFO> modes(nMode);
    if (QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, &nPath, paths.data(), &nMode, modes.data(), nullptr) != ERROR_SUCCESS)
        return 240.0;
    for (UINT32 i = 0; i < nPath; i++)
    {
        DISPLAYCONFIG_SOURCE_DEVICE_NAME src{};
        src.header = {DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME, sizeof(src), paths[i].sourceInfo.adapterId,
                      paths[i].sourceInfo.id};
        if (DisplayConfigGetDeviceInfo(&src.header) != ERROR_SUCCESS)
            continue;
        if (wcscmp(src.viewGdiDeviceName, mi.szDevice) != 0)
            continue;
        DISPLAYCONFIG_SDR_WHITE_LEVEL wl{};
        wl.header = {DISPLAYCONFIG_DEVICE_INFO_GET_SDR_WHITE_LEVEL, sizeof(wl), paths[i].targetInfo.adapterId,
                     paths[i].targetInfo.id};
        if (DisplayConfigGetDeviceInfo(&wl.header) == ERROR_SUCCESS && wl.SDRWhiteLevel > 0)
            return wl.SDRWhiteLevel * 80.0 / 1000.0;
    }
    return 240.0;
}

// ---------------------------------------------------------------- perf HUD

// Small always-on-top readout at the overlay's corner: base fps -> interpolated fps plus the
// estimated capture->screen delay, in white. A SEPARATE window on purpose:
// pixels stamped into the frames would get interpolated (DLSS-G smears moving text), and the
// HUD must NEVER take activation (WS_EX_NOACTIVATE) or DLSS-G's focus gate would drop. Black
// colorkey = floating text, click-through, excluded from capture so monitor mode never sees
// itself (this also keeps it out of the --diag dumps).
struct Hud
{
    HWND hwnd = nullptr;
    HFONT font = nullptr;
    wchar_t text[128]{};

    void create(int x, int y)
    {
        WNDCLASSW wc{};
        wc.lpfnWndProc = DefWindowProcW;
        wc.hInstance = GetModuleHandleW(nullptr);
        wc.lpszClassName = L"smvlivehud";
        RegisterClassW(&wc);
        hwnd = CreateWindowExW(WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE | WS_EX_LAYERED | WS_EX_TRANSPARENT,
                               L"smvlivehud", L"", WS_POPUP, x, y, 620, 36, nullptr, nullptr, wc.hInstance, nullptr);
        if (!hwnd)
        {
            LOG("HUD window creation failed (continuing without)\n");
            return;
        }
        SetLayeredWindowAttributes(hwnd, RGB(0, 0, 0), 0, LWA_COLORKEY);
        SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE); // best effort
        font = CreateFontW(-24, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS,
                           CLIP_DEFAULT_PRECIS, ANTIALIASED_QUALITY, DEFAULT_PITCH, L"Segoe UI");
        wcscpy_s(text, L"\u2026"); // ellipsis until the first stats window lands (escaped: see update())
        ShowWindow(hwnd, SW_SHOWNOACTIVATE);
        paint();
    }

    void update(double inFps, double outFps, double latMs)
    {
        if (!hwnd)
            return;
        // \u2192 = the arrow; ESCAPED on purpose: a raw UTF-8 literal in this file compiles
        // as ANSI mojibake unless the build adds /utf-8 (it showed as "weird letters")
        // --no-hud-latency hides the latency segment only; the fps segment is governed by
        // --no-hud (whole readout). DLSS-G never reaches here with a latency to show (SL
        // paces internally and the caller passes its handoff figure), so that path is
        // unchanged either way.
        if (latMs > 0.5 && !g_noHudLat)
            swprintf_s(text, L"%.0f fps \u2192 %.0f fps   ~%.0f ms behind", inFps, outFps, latMs);
        else
            swprintf_s(text, L"%.0f fps \u2192 %.0f fps", inFps, outFps);
        // re-assert the top of the TOPMOST band: the overlay's HWND_TOPMOST moves raise it above us
        SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
        paint();
    }

    void paint()
    {
        if (!hwnd)
            return;
        HDC dc = GetDC(hwnd);
        RECT r;
        GetClientRect(hwnd, &r);
        FillRect(dc, &r, (HBRUSH)GetStockObject(BLACK_BRUSH));
        HGDIOBJ of = SelectObject(dc, font);
        SetBkMode(dc, TRANSPARENT);
        SetTextColor(dc, RGB(255, 255, 255)); // full white: easier to read than green
        r.left += 4;
        DrawTextW(dc, text, -1, &r, DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_NOPREFIX);
        SelectObject(dc, of);
        ReleaseDC(hwnd, dc);
    }

    void show(bool visible)
    {
        if (hwnd)
            ShowWindow(hwnd, visible ? SW_SHOWNOACTIVATE : SW_HIDE);
    }
    void move(int x, int y)
    {
        if (hwnd)
            SetWindowPos(hwnd, HWND_TOPMOST, x, y, 0, 0, SWP_NOSIZE | SWP_NOACTIVATE);
    }

    void destroy()
    {
        if (hwnd)
            DestroyWindow(hwnd);
        hwnd = nullptr;
        if (font)
            DeleteObject(font);
        font = nullptr;
    }
};

// ---------------------------------------------------------------- Fill: the mouse on the stretched picture

// Fill stretches a smaller window over its monitor through a click-through overlay, so a click lands on whatever sits
// under that screen point, not on the matching point of the window. The model Magpie and Lossless Scaling use: while
// the cursor is on the picture, the REAL cursor sits at the matching point of the window's client area (clicks,
// drags, hover and raw input all reach the window natively), the OS cursor is hidden, a copy is drawn at the picture
// point in a separate click-through window (nothing is stamped into the frames, the HUD's rule) and the pointer speed
// is divided by the stretch so the copy moves at the usual speed. A window above the overlay at the picture point,
// the target covered at its point, a hidden overlay (pause, resize), leaving through a monitor edge with a neighbour
// and the session end hand the real cursor back at the picture point with the speed restored. The original speed and
// the clip are kept in a marker file until they are restored: the next session, or `--restore-mouse` (the app runs it
// after killing the host), restores them after a crash or a kill.
static std::wstring fillMouseMarker()
{
    wchar_t tmp[MAX_PATH]{};
    GetTempPathW(MAX_PATH, tmp);
    return std::wstring(tmp) + L"smv-live-mouse.txt";
}

using ShowSystemCursorFn = BOOL(WINAPI*)(BOOL);
static void showSystemCursor(bool show)
{
    static ShowSystemCursorFn fn = [] {
        ShowSystemCursorFn f = nullptr;
        if (HMODULE user = GetModuleHandleW(L"user32.dll"))
            f = (ShowSystemCursorFn)(void*)GetProcAddress(user, "ShowSystemCursor");
        if (!f)
            if (HMODULE mag = LoadLibraryW(L"Magnification.dll"))
                f = (ShowSystemCursorFn)(void*)GetProcAddress(mag, "MagShowSystemCursor");
        return f;
    }();
    if (fn)
        fn(show ? TRUE : FALSE);
}

// The restore after a session that could not run its own: the speed from the marker, the clip when it is still ours,
// the OS cursor shown (a no-op when the hidden state ended with the process). true = a marker was found.
static bool fillMouseRestore()
{
    const std::wstring path = fillMouseMarker();
    FILE* f = nullptr;
    if (_wfopen_s(&f, path.c_str(), L"r") || !f)
        return false;
    int speed = 0;
    RECT clip{};
    const int n =
        fscanf_s(f, "speed %d clip %ld %ld %ld %ld", &speed, &clip.left, &clip.top, &clip.right, &clip.bottom);
    fclose(f);
    if (n >= 1 && speed >= 1 && speed <= 20)
        SystemParametersInfoW(SPI_SETMOUSESPEED, 0, (void*)(intptr_t)speed, 0);
    RECT cur{};
    // the marker's clip, or the 1x1 clip of a teleport the kill landed in
    if (GetClipCursor(&cur) &&
        ((n == 5 && EqualRect(&cur, &clip)) || (cur.right - cur.left == 1 && cur.bottom - cur.top == 1)))
        ClipCursor(nullptr);
    showSystemCursor(true);
    DeleteFileW(path.c_str());
    LOG("mouse restored after a session that did not end cleanly (pointer speed %d%s)\n", speed,
        n == 5 ? ", its clip released when still set" : "");
    return true;
}

struct FillMouse
{
    HWND target = nullptr, overlay = nullptr;
    RECT mon{}, pic{}; // the Fill monitor and the picture rect on it (screen, physical pixels)
    std::thread th;
    std::atomic<bool> quit{false};

    // cursor thread state
    bool in = false;
    HWND wnd = nullptr; // the drawn copy
    HCURSOR shape = nullptr;
    int hotX = 0, hotY = 0;
    bool shown = false;
    int origSpeed = 0;
    RECT clip{};
    RECT client{}; // the target's client rect at the last poll (screen)
    POINT lastP{-1, -1};
    int scaleSpeed = 0;

    void start(HWND t, HWND ov, const RECT& m, const RECT& p)
    {
        target = t;
        overlay = ov;
        mon = m;
        pic = p;
        th = std::thread([this] { run(); });
    }
    void stop()
    {
        if (!th.joinable())
            return;
        quit.store(true);
        th.join();
    }
    ~FillMouse()
    {
        stop();
    }

    // Magpie's mapping: first pixel to first pixel, last to last
    static int mapAxis(int v, int a0, int aLen, int b0, int bLen)
    {
        if (aLen <= 1)
            return b0;
        return b0 + (int)std::lround((double)(v - a0) * (bLen - 1) / (aLen - 1));
    }
    POINT toWindow(POINT p) const
    {
        return {mapAxis(p.x, pic.left, pic.right - pic.left, client.left, client.right - client.left),
                mapAxis(p.y, pic.top, pic.bottom - pic.top, client.top, client.bottom - client.top)};
    }
    // a point outside the client rect keeps its offset unscaled past the picture's edge (leaving the picture)
    POINT toPicture(POINT q) const
    {
        POINT c{std::clamp(q.x, client.left, client.right - 1), std::clamp(q.y, client.top, client.bottom - 1)};
        POINT p{mapAxis(c.x, client.left, client.right - client.left, pic.left, pic.right - pic.left),
                mapAxis(c.y, client.top, client.bottom - client.top, pic.top, pic.bottom - pic.top)};
        p.x += q.x - c.x;
        p.y += q.y - c.y;
        return p;
    }

    bool clientRect(RECT& r) const
    {
        RECT c{};
        POINT o{0, 0};
        if (!GetClientRect(target, &c) || !ClientToScreen(target, &o) || c.right <= 0 || c.bottom <= 0)
            return false;
        r = {o.x, o.y, o.x + c.right, o.y + c.bottom};
        return true;
    }

    // the target is on top at q: its own popups (menus, tooltips) count as the target
    bool targetAt(POINT q) const
    {
        const HWND h = topAt(q, nullptr);
        if (h == target)
            return true;
        DWORD a = 0, b = 0;
        GetWindowThreadProcessId(h, &a);
        GetWindowThreadProcessId(target, &b);
        return h && a == b;
    }

    // the first window at a screen point from the top of the z-order, skipping hidden, cloaked and click-through
    // windows and this process's own (the overlay is click-through too, so it is named: stopAt)
    static HWND topAt(POINT p, HWND stopAt)
    {
        const DWORD own = GetCurrentProcessId();
        for (HWND h = GetTopWindow(nullptr); h; h = GetWindow(h, GW_HWNDNEXT))
        {
            if (h == stopAt)
                return h;
            if (!IsWindowVisible(h) || IsIconic(h))
                continue;
            DWORD pid = 0;
            GetWindowThreadProcessId(h, &pid);
            if (pid == own || (GetWindowLongPtrW(h, GWL_EXSTYLE) & WS_EX_TRANSPARENT))
                continue;
            BOOL cloaked = FALSE;
            DwmGetWindowAttribute(h, DWMWA_CLOAKED, &cloaked, sizeof(cloaked));
            if (cloaked)
                continue;
            RECT r{};
            if (!frameBounds(h, r) && !GetWindowRect(h, &r))
                continue;
            if (PtInRect(&r, p))
                return h;
        }
        return nullptr;
    }

    // a neighbouring monitor past one edge of the Fill monitor (three probes along the edge)
    bool neighbour(int side) const
    {
        const LONG xs[3] = {mon.left, (mon.left + mon.right) / 2, mon.right - 1};
        const LONG ys[3] = {mon.top, (mon.top + mon.bottom) / 2, mon.bottom - 1};
        for (int i = 0; i < 3; i++)
        {
            const POINT q = side == 0   ? POINT{mon.left - 1, ys[i]}
                            : side == 1 ? POINT{mon.right, ys[i]}
                            : side == 2 ? POINT{xs[i], mon.top - 1}
                                        : POINT{xs[i], mon.bottom};
            if (MonitorFromPoint(q, MONITOR_DEFAULTTONULL))
                return true;
        }
        return false;
    }

    // the clip while the cursor is in: the client rect, open towards a neighbouring monitor
    RECT captureClip() const
    {
        RECT r = client;
        const int vx = GetSystemMetrics(SM_XVIRTUALSCREEN), vy = GetSystemMetrics(SM_YVIRTUALSCREEN);
        const int vw = GetSystemMetrics(SM_CXVIRTUALSCREEN), vh = GetSystemMetrics(SM_CYVIRTUALSCREEN);
        if (neighbour(0))
            r.left = vx;
        if (neighbour(1))
            r.right = vx + vw;
        if (neighbour(2))
            r.top = vy;
        if (neighbour(3))
            r.bottom = vy + vh;
        return r;
    }

    void writeMarker() const
    {
        FILE* f = nullptr;
        if (_wfopen_s(&f, fillMouseMarker().c_str(), L"w") || !f)
            return;
        fprintf(f, "speed %d clip %ld %ld %ld %ld\n", origSpeed, clip.left, clip.top, clip.right, clip.bottom);
        fclose(f);
    }

    void setClip(const RECT& r)
    {
        RECT cur{};
        if (GetClipCursor(&cur) && EqualRect(&cur, &r) && EqualRect(&clip, &r))
            return; // each ClipCursor sends the foreground window a WM_MOUSEMOVE: only on a change
        clip = r;
        ClipCursor(&r);
        writeMarker();
    }

    // SetCursorPos through a 1x1 clip first, as Magpie does: queued hardware input can otherwise undo the move;
    // `after` = the clip left in place (nullptr = none)
    static void moveCursor(POINT p, const RECT* after)
    {
        const RECT one{p.x, p.y, p.x + 1, p.y + 1};
        ClipCursor(&one);
        SetCursorPos(p.x, p.y);
        Sleep(8);
        ClipCursor(after);
    }

    // the pointer speed that keeps the copy at the usual speed (Magpie's formula: the speed over the stretch with
    // "Enhance pointer precision" on, else the nearest step of Windows' 20-step multiplier table)
    int scaledSpeed(int orig) const
    {
        const double s = ((double)(pic.right - pic.left) / (client.right - client.left) +
                          (double)(pic.bottom - pic.top) / (client.bottom - client.top)) /
                         2;
        int accel[3]{};
        SystemParametersInfoW(SPI_GETMOUSE, 0, accel, 0);
        if (accel[2])
            return std::clamp((int)std::lround(orig / s), 1, 20);
        static const double kMul[20] = {0.03125, 0.0625, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0,
                                        1.25,    1.5,    1.75,  2.0,  2.25,  2.5, 2.75,  3.0,  3.25,  3.5};
        const double want = kMul[std::clamp(orig, 1, 20) - 1] / s;
        int best = 1;
        for (int i = 2; i <= 20; i++)
            if (std::fabs(kMul[i - 1] - want) < std::fabs(kMul[best - 1] - want))
                best = i;
        return best;
    }

    void enter(POINT p)
    {
        const POINT q = toWindow(p);
        SystemParametersInfoW(SPI_GETMOUSESPEED, 0, &origSpeed, 0);
        clip = RECT{};
        writeMarker(); // the original speed is on disk before it changes
        scaleSpeed = scaledSpeed(origSpeed);
        SystemParametersInfoW(SPI_SETMOUSESPEED, 0, (void*)(intptr_t)scaleSpeed, 0);
        showSystemCursor(false);
        const RECT c = captureClip();
        moveCursor(q, &c);
        clip = c;
        writeMarker();
        in = true;
        lastP = {-1, -1};
    }

    // the real cursor goes to p when a monitor is there (the clip is always ours while in: setClip reasserts it)
    void leave(POINT p)
    {
        in = false;
        if (MonitorFromPoint(p, MONITOR_DEFAULTTONULL))
            moveCursor(p, nullptr);
        else
            ClipCursor(nullptr);
        clip = RECT{};
        SystemParametersInfoW(SPI_SETMOUSESPEED, 0, (void*)(intptr_t)origSpeed, 0);
        showSystemCursor(true);
        DeleteFileW(fillMouseMarker().c_str());
        if (wnd)
            ShowWindow(wnd, SW_HIDE);
        shown = false;
    }

    // the copy's image: DrawIconEx on black and on white gives the alpha (255 - (white - black)) and the
    // premultiplied colour (the black draw); an inverting pixel (white draw darker) is drawn opaque white
    void setShape(HCURSOR c)
    {
        shape = c;
        ICONINFO ii{};
        if (!GetIconInfo(c, &ii))
            return;
        BITMAP bm{};
        GetObjectW(ii.hbmColor ? ii.hbmColor : ii.hbmMask, sizeof(bm), &bm);
        const int w = bm.bmWidth, h = ii.hbmColor ? bm.bmHeight : bm.bmHeight / 2;
        hotX = (int)ii.xHotspot;
        hotY = (int)ii.yHotspot;
        if (ii.hbmColor)
            DeleteObject(ii.hbmColor);
        if (ii.hbmMask)
            DeleteObject(ii.hbmMask);
        if (w <= 0 || h <= 0)
            return;
        BITMAPINFO bi{};
        bi.bmiHeader = {sizeof(BITMAPINFOHEADER), w, -h, 1, 32, BI_RGB, (DWORD)w * (DWORD)h * 4};
        const size_t n = bi.bmiHeader.biSizeImage / 4;
        HDC screen = GetDC(nullptr);
        HDC dcs[3]{};
        HBITMAP bmps[3]{};
        uint32_t* px[3]{};
        bool ok = screen != nullptr;
        for (int i = 0; i < 3 && ok; i++)
        {
            dcs[i] = CreateCompatibleDC(screen);
            bmps[i] = CreateDIBSection(screen, &bi, DIB_RGB_COLORS, (void**)&px[i], nullptr, 0);
            ok = dcs[i] && bmps[i] && px[i];
            if (ok)
                SelectObject(dcs[i], bmps[i]);
        }
        if (ok)
        {
            for (size_t i = 0; i < n; i++)
            {
                px[0][i] = 0x00000000;
                px[1][i] = 0x00FFFFFF;
            }
            DrawIconEx(dcs[0], 0, 0, c, w, h, 0, nullptr, DI_NORMAL);
            DrawIconEx(dcs[1], 0, 0, c, w, h, 0, nullptr, DI_NORMAL);
            GdiFlush();
            for (size_t i = 0; i < n; i++)
            {
                const uint32_t b = px[0][i] & 0xFFFFFF, wt = px[1][i] & 0xFFFFFF;
                const int bg = (int)((b >> 8) & 0xFF), wg = (int)((wt >> 8) & 0xFF);
                const int a = 255 - (wg - bg);
                px[2][i] = a > 255 ? 0xFFFFFFFFu : ((uint32_t)std::clamp(a, 0, 255) << 24) | b;
            }
            POINT src{0, 0};
            SIZE sz{w, h};
            BLENDFUNCTION bf{AC_SRC_OVER, 0, 255, AC_SRC_ALPHA};
            UpdateLayeredWindow(wnd, screen, nullptr, &sz, dcs[2], &src, 0, &bf, ULW_ALPHA);
        }
        for (int i = 0; i < 3; i++)
        {
            if (dcs[i])
                DeleteDC(dcs[i]);
            if (bmps[i])
                DeleteObject(bmps[i]);
        }
        if (screen)
            ReleaseDC(nullptr, screen);
    }

    void draw(POINT p, const CURSORINFO& ci)
    {
        const bool visible = (ci.flags & CURSOR_SHOWING) && ci.hCursor;
        if (!visible)
        {
            if (shown)
                ShowWindow(wnd, SW_HIDE);
            shown = false;
            return;
        }
        if (ci.hCursor != shape)
        {
            setShape(ci.hCursor);
            lastP = {-1, -1};
        }
        if (p.x != lastP.x || p.y != lastP.y || !shown)
        {
            SetWindowPos(wnd, HWND_TOPMOST, p.x - hotX, p.y - hotY, 0, 0,
                         SWP_NOSIZE | SWP_NOACTIVATE | (shown ? 0 : SWP_SHOWWINDOW));
            lastP = p;
            shown = true;
        }
    }

    void run()
    {
        WNDCLASSW wc{};
        wc.lpfnWndProc = DefWindowProcW;
        wc.hInstance = GetModuleHandleW(nullptr);
        wc.lpszClassName = L"smvlivecursor";
        RegisterClassW(&wc);
        wnd = CreateWindowExW(WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE | WS_EX_LAYERED | WS_EX_TRANSPARENT,
                              L"smvlivecursor", L"", WS_POPUP, 0, 0, 32, 32, nullptr, nullptr, wc.hInstance, nullptr);
        if (!wnd)
        {
            LOG("Fill mouse: cursor window creation failed, clicks stay unmapped\n");
            return;
        }
        LOG("Fill mouse: clicks on the picture reach the window (the cursor is drawn, pointer speed follows the "
            "stretch)\n");
        POINT p{};
        ULONGLONG zTick = 0; // the z-order checks run every 16 ms, the cursor every 1 ms
        while (!quit.load())
        {
            MSG m;
            while (PeekMessageW(&m, nullptr, 0, 0, PM_REMOVE))
                DispatchMessageW(&m);
            Sleep(1);
            CURSORINFO ci{sizeof(ci)};
            if (!GetCursorInfo(&ci))
                continue;
            const bool alive = IsWindow(target) && IsWindowVisible(overlay) && clientRect(client);
            const bool held =
                (GetAsyncKeyState(VK_LBUTTON) | GetAsyncKeyState(VK_RBUTTON) | GetAsyncKeyState(VK_MBUTTON)) < 0;
            const ULONGLONG now = GetTickCount64();
            const bool zCheck = now - zTick >= 16;
            if (zCheck)
                zTick = now;
            if (!in)
            {
                p = ci.ptScreenPos;
                if (alive && !held && zCheck && PtInRect(&pic, p) && topAt(p, overlay) == overlay &&
                    targetAt(toWindow(p)))
                    enter(p);
                continue;
            }
            const POINT q = ci.ptScreenPos;
            p = toPicture(q);
            if (!alive)
            {
                leave(p);
                continue;
            }
            if (!PtInRect(&pic, p))
            {
                // out through an open side: onto a black bar or a neighbouring monitor; else (the clip was down for a
                // moment) back inside
                if (!held && MonitorFromPoint(p, MONITOR_DEFAULTTONULL))
                {
                    leave(p);
                    continue;
                }
                const POINT c{std::clamp(q.x, client.left, client.right - 1),
                              std::clamp(q.y, client.top, client.bottom - 1)};
                moveCursor(c, &clip);
                p = toPicture(c);
            }
            else if (!held && zCheck && (topAt(p, overlay) != overlay || !targetAt(q)))
            {
                leave(p); // a window above the overlay at p (the Start menu, a toast), or the target covered at q
                continue;
            }
            setClip(captureClip()); // the OS drops the clip at every foreground change
            draw(p, ci);
            if (zCheck && shown)
                SetWindowPos(wnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
        }
        if (in)
            leave(p);
        DestroyWindow(wnd);
        wnd = nullptr;
    }
};

// ---------------------------------------------------------------- diag dump thread

struct DiagCtx
{
    RECT rect;
    int seconds;
    volatile LONG* running;
};

static DWORD WINAPI diagThread(LPVOID p)
{
    DiagCtx* d = (DiagCtx*)p;
    const HRESULT coHr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const uint32_t w = d->rect.right - d->rect.left;
    const uint32_t h = d->rect.bottom - d->rect.top;
    ULONGLONG end = GetTickCount64() + (ULONGLONG)d->seconds * 1000;
    int i = 0;
    while (*d->running && GetTickCount64() < end && i < 250)
    {
        wchar_t name[64];
        swprintf_s(name, L"live_diag_%03d.png", i++); // cwd-relative (bin/ in dev, engine/live shipped)
        captureScreen(d->rect.left, d->rect.top, w, h, name);
        Sleep(40);
    }
    LOG("diag: wrote %d overlay dumps\n", i);
    if (SUCCEEDED(coHr))
        CoUninitialize();
    return 0;
}

// ---------------------------------------------------------------- live mode

static int slInitCommon()
{
    wchar_t exePath[MAX_PATH];
    GetModuleFileNameW(nullptr, exePath, MAX_PATH);
    static std::wstring exeDir(exePath);
    exeDir.resize(exeDir.find_last_of(L'\\'));
    static const wchar_t* pluginPaths[] = {exeDir.c_str()};
    static sl::Feature features[] = {sl::kFeatureDLSS_G, sl::kFeatureReflex, sl::kFeaturePCL};

    g_verbose = GetEnvironmentVariableW(L"DLSSG_VERBOSE", nullptr, 0) != 0;
    sl::Preferences pref{};
    pref.showConsole = false;
    pref.logLevel = g_verbose ? sl::LogLevel::eVerbose : sl::LogLevel::eDefault;
    pref.pathsToPlugins = pluginPaths;
    pref.numPathsToPlugins = 1;
    pref.pathToLogsAndData = nullptr;
    pref.logMessageCallback = slLog;
    pref.featuresToLoad = features;
    pref.numFeaturesToLoad = _countof(features);
    pref.applicationId = 231313132; // Streamline sample app id
    pref.flags |= sl::PreferenceFlags::eUseFrameBasedResourceTagging;
    pref.flags |= sl::PreferenceFlags::eUseDXGIFactoryProxy;
    pref.renderAPI = sl::RenderAPI::eD3D12;
    CHECK_SL(slInit(pref, sl::kSDKVersion));
    return 0;
}

// diagnostic: no WGC at all, present synthetic moving-square frames at ~30fps through the
// identical host/present path. Isolates whether in-process WGC breaks the DLSS-G pacer.
static int runSynth(int genFrames, bool vsync)
{
    W = 960;
    H = 540;
    Host host;
    host.genFrames = genFrames;
    host.syncInterval = vsync ? 1 : 0;
    host.park = true;
    int rc = host.init();
    if (rc)
        return rc;

    const size_t frameBytes = (size_t)W * H * 4;
    std::vector<uint8_t> buf(frameBytes);
    auto paint = [&](int step) {
        memset(buf.data(), 24, frameBytes);
        int x0 = 40 + step * 120;
        for (uint32_t y = 230; y < 310; y++)
            for (int x = x0; x < x0 + 80; x++)
            {
                uint8_t* p = buf.data() + ((size_t)y * W + x) * 4;
                p[0] = 230;
                p[1] = 40;
                p[2] = 40;
                p[3] = 255;
            }
    };

    paint(0);
    for (int i = 0; i < 3; i++)
        if (!host.presentFrame(buf.data()))
        {
            LOG("warmup present failed\n");
            return 1;
        }
    sl::DLSSGState st{};
    slDLSSGGetState(host.vp, st, nullptr);
    LOG("synth warmup: status=%d\n", (int)st.status);

    UINT base = 0;
    host.scNative->GetLastPresentCount(&base);
    ULONGLONG t0 = GetTickCount64();
    int presented = 0;
    for (int i = 1; GetTickCount64() - t0 < 12000; i++)
    {
        paint(i % 7);
        if (!host.presentFrame(buf.data()))
        {
            LOG("presentFrame failed\n");
            return 1;
        }
        presented++;
        if (presented % 60 == 0)
        {
            UINT c = 0;
            host.scNative->GetLastPresentCount(&c);
            slDLSSGGetState(host.vp, st, nullptr);
            LOG("synth: %d app presents -> %u native presents (ratio %.2f) actuallyPresented=%u\n", presented, c - base,
                (double)(c - base) / presented, st.numFramesActuallyPresented);
        }
        Sleep(33);
    }
    UINT c = 0;
    host.scNative->GetLastPresentCount(&c);
    LOG("synth done: %d app presents -> %u native presents (ratio %.2f)\n", presented, c - base,
        (double)(c - base) / presented);
    host.shutdown();
    slShutdown();
    return 0;
}

// RESIZE DEBOUNCE: ending the session (exit 4, the app revives it) on the FIRST >2 px size
// change would turn a drag resize into a chain of engine builds at whatever intermediate size
// the window had 400 ms later, each blocking the host for up to a minute. So the stale overlay
// hides at once, the HUD shows the loading note at once, and the session ends only after the client
// size has held for SMV_LIVE_RESIZE_SETTLE_MS (default 1000). Returns early when the window
// closes or the session is ending anyway (stdin "stop"). g_monitor sessions never get here
// (monitor capture has no window size).
// What the loading note names: the GUI's --label lists the ticked model and live effects ("GMFSS + DLSS 5",
// "DLSS 5 + Sharpen") and is shown as is; a run without a label names the backend.
static std::wstring loadingWhat()
{
    if (!g_modelLabel.empty())
        return g_modelLabel;
    return (g_serverBackend.empty() ? std::wstring(L"DLSS 4.5") : g_serverBackend) + L" model";
}
// Waits until the target's client size holds still. true = it came back to the captured size (a
// transient: the restore animation after a minimize hands out a frame of another size), the
// session continues with the overlay shown again; false = a real resize, the caller restarts.
static bool resizeSettle(HWND target, Host& host, const Capture& cap, Hud& hud, bool hidden)
{
    static DWORD settleMs = 0;
    if (!settleMs)
    {
        wchar_t sv[16]{};
        settleMs = GetEnvironmentVariableW(L"SMV_LIVE_RESIZE_SETTLE_MS", sv, 16) ? (DWORD)_wtoi(sv) : 0;
        if (settleMs < 100 || settleMs > 10000)
            settleMs = 1000;
    }
    ShowWindow(host.hwnd, SW_HIDE); // the old-size frames no longer cover the window
    if (hud.hwnd)
    {
        _snwprintf_s(hud.text, _TRUNCATE, L"SMV Live: loading %ls...", loadingWhat().c_str());
        hud.show(!hidden);
        SetWindowPos(hud.hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
        hud.paint();
    }
    RECT last{};
    GetClientRect(target, &last);
    const ULONGLONG t0 = GetTickCount64();
    ULONGLONG stable = t0;
    for (;;)
    {
        pumpMessages();
        Sleep(50);
        if (!IsWindow(target) || g_stopReq.load())
            break;
        RECT r{};
        GetClientRect(target, &r);
        if (r.right != last.right || r.bottom != last.bottom)
        {
            last = r;
            stable = GetTickCount64();
        }
        else if (GetTickCount64() - stable >= settleMs)
            break;
        RECT fb{}; // the note follows the window while it is dragged
        if (frameBounds(target, fb))
            hud.move(fb.left + cap.cropX + 16, fb.top + cap.cropY + 16);
    }
    LOG("target window size settled at %ldx%ld after %llu ms\n", last.right, last.bottom,
        (unsigned long long)(GetTickCount64() - t0));
    if (IsWindow(target) && !g_stopReq.load() && last.right == (LONG)cap.cw && last.bottom == (LONG)cap.ch)
    {
        LOG("target window back at its captured size, the session continues\n");
        if (!hidden)
            ShowWindow(host.hwnd, SW_SHOWNA);
        if (g_fill)
            hud.move(host.posX + 16, host.posY + 16); // back on the screen-wide overlay
        return true;
    }
    return false;
}

// A resize that did not settle back ends the session: exit 4 (the app revives it at the new size), or exit 7
// when the window closed during the settle (a player that shrinks while it closes).
static int endAfterResize(HWND target)
{
    if (!g_monitor && !IsWindow(target))
    {
        LOG("target window closed\n");
        return 7;
    }
    LOG("restart smv-live after resizing the target window\n");
    return 4;
}

static int runLive(const wchar_t* needle, HWND targetOverride, int genFrames, bool vsync, bool clickthrough,
                   int diagSecs, bool park)
{
    CHECK_HR(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&g_wic)));
    timeBeginPeriod(1);
    fillMouseRestore(); // a Fill session that was killed or crashed left the pointer speed changed

    HWND target = targetOverride ? targetOverride : findTargetWindow(needle);
    if (targetOverride)
    {
        wchar_t title[512]{};
        // exit 7 = the captured window is gone: the app's revive and settings restart reuse the handle,
        // and a player that shrinks while it closes (mpv leaving fullscreen) ends its last session with
        // a resize (exit 4) just before its window goes
        if (!IsWindow(target) || !IsWindowVisible(target))
        {
            LOG("target window closed (hwnd 0x%p %s)\n", (void*)target,
                IsWindow(target) ? "is hidden" : "no longer exists");
            return 7;
        }
        GetWindowTextW(target, title, 512);
        // UTF-8 for the same reason as in enumProc above (the app parses this line)
        LOG("target window: hwnd=0x%p \"%s\"\n", (void*)target, wideToUtf8(title).c_str());
    }
    if (!target)
    {
        LOG("no visible window matching \"%s\"\n", wideToUtf8(needle ? needle : L"").c_str());
        return 1;
    }
    g_targetHwnd = target; // the resident "live session ended" line reports it (the app's revive)
    // Resolve HDR live mode. Env SMV_LIVE_HDR overrides detection (1 = force on, 0 = force
    // off); otherwise HDR runs when the display has Windows HDR on AND the mode supports it.
    // The native identity echo stays SDR. Re-evaluated per session (the exit-4 restart re-enters
    // here). When HDR is on but this mode can't use it, fall back to the
    // SDR-clips-HDR notice.
    {
        const bool hdrDisplay = monitorIsHDR(target);
        wchar_t ov[8]{};
        // rife/gmfss run on PQ and compose R10A2, so HDR reaches them (plus the
        // echo identity path). Fill is HDR-capable too: the server rescales on the PQ
        // tensors and letterboxes into the R10A2 canvas (RTX VSR demoted to Lanczos3 there,
        // with a log line). SMV_LIVE_HDR forces the mode for testing. "blend" (the LSFG
        // comparison baseline) composes the exact same PQ R10A2 path as rife/gmfss and exists
        // precisely for HDR A-B comparisons, so it is HDR-capable too.
        // DLSS-G is HDR-capable through the exe-side pack (kHdrPackCS): SL mandates RGB10 + PQ
        // for HDR, the capture converts before present, SL interpolates on the PQ backbuffers.
        // "fruc" (Smooth Motion): the server feeds the NvOFFRUC bridge the PQ-encoded frames like rife, the bridge
        // quantises them to 8-bit BGRA for the flow and the warp, so only the TWEENS carry 8-bit
        // PQ precision (the real frames stay full precision); far better than the SDR capture
        // of an HDR-presented window, which is 2-3x over-bright and clipped (below).
        const bool hdrCapable =
            g_backend == BK_DLSSG ||
            (g_backend == BK_SERVER &&
             (g_serverBackend == L"echo" || g_serverBackend == L"rife" || g_serverBackend == L"gmfss" ||
              g_serverBackend == L"blend" || g_serverBackend == L"nvof" || g_serverBackend == L"rifedrba" ||
              g_serverBackend == L"fruc"));
        // AUTO: HDR runs whenever the display has Windows HDR on and the mode supports it. A
        // "purple screen" at the start was the model-load passthrough presenting raw FP16 capture
        // bytes into the R10A2 swap chain (a startup transient, suppressed below); the
        // steady-state present itself probes value-exact.
        // Without HDR, the 8-bit capture of an HDR-presented source re-lifts to desktop SDR white
        // (measured 2-3x over-bright + 240-nit clip), so auto-on is the safe default. SMV_LIVE_HDR
        // still overrides both ways: 1 forces on (testing), 0 forces off (escape hatch).
        g_hdr = hdrDisplay && hdrCapable;
        if (GetEnvironmentVariableW(L"SMV_LIVE_HDR", ov, 8))
            g_hdr = (ov[0] == L'1') && hdrCapable;
        if (g_hdr)
            LOG("HDR live mode ON: FP16 scRGB capture -> BT.2020 PQ, R10G10B10A2 present (G2084)\n");
        // TrueHDR needs the HDR live pipeline (server backends only; not the exe-side DLSS-G
        // pack). Resolve the target monitor's
        // SDR white here, while the target is known.
        if (g_rtxHdr && g_hdr && g_backend == BK_SERVER)
        {
            g_sdrWhite = sdrWhiteNits(target);
            LOG("RTX HDR live: TrueHDR expansion on (colour %ls, SDR white %.0f nits)\n", g_hdrColor, g_sdrWhite);
        }
        else if (g_rtxHdr)
            LOG("RTX HDR live skipped (needs an HDR display and a server model: RIFE/GMFSS/Frame Blend)\n");
        else if (g_hdr && g_backend == BK_SERVER)
            g_sdrWhite = sdrWhiteNits(target); // the SDR range DLSS 5 and RIFE's motion frame work on
        if (!g_hdr && hdrDisplay)
            LOG("NOTICE: Windows HDR is ON for this display. This model captures 8-bit SDR for now, "
                "so HDR highlights will look over-bright/clipped. HDR live support is in progress for "
                "this model; until then, turn off Windows HDR on this display (Win+Alt+B) or let the "
                "player tonemap to SDR.\n");
    }
    RECT fb{};
    if (!frameBounds(target, fb))
    {
        LOG("DwmGetWindowAttribute failed\n");
        return 1;
    }

    // capture first: the frame pool size defines the swap chain size
    Host host;
    host.useSL = (g_backend == BK_DLSSG);
    host.genFrames = genFrames;
    host.syncInterval = vsync ? 1 : 0;
    host.clickthrough = clickthrough;
    host.park = park;
    host.posX = fb.left;
    host.posY = fb.top;

    // Capture needs the adapter; Host owns adapter discovery, so init capture in two steps:
    // a bare factory/adapter probe here, full host init after W/H are known.
    {
        ComPtr<IDXGIFactory2> f;
        CHECK_HR(CreateDXGIFactory2(0, IID_PPV_ARGS(&f)));
        ComPtr<IDXGIAdapter1> a;
        DXGI_ADAPTER_DESC1 ad{};
        for (UINT i = 0; f->EnumAdapters1(i, &a) != DXGI_ERROR_NOT_FOUND; i++)
        {
            a->GetDesc1(&ad);
            if (ad.VendorId == 0x10DE)
                break;
            a.Reset();
        }
        if (!a)
        {
            LOG("no NVIDIA adapter found\n");
            return 1;
        }
        g_vramBytes = ad.DedicatedVideoMemory; // the slot-count memory budget (slotBudget)

        static Capture cap;       // static: outlives this scope, single instance per process
        cap.swizzle = host.useSL; // DLSS-G keeps RGBA; server/identity routes stay BGRA
        // dlssg in HDR packs scRGB -> R10A2 PQ in the exe; swizzle is bypassed there. Must be set before init() (it sizes the chain
        // and can clear g_hdr on setup failure, before anything else reads it).
        cap.hdrPack = g_hdr && host.useSL;
        int rc = cap.init(target, a.Get());
        if (rc && !g_monitor && !IsWindow(target))
        {
            LOG("target window closed\n");
            return 7;
        }
        if (rc)
            return rc;
        if (g_backend == BK_SERVER)
            cap.initInterop(); // zero-copy capture for the server
        const uint32_t capW = cap.cw, capH = cap.ch;
        RECT mon{};
        if (g_monitor)
        {
            // whole-screen mode: capture IS the monitor, overlay covers it 1:1 (capture size ==
            // presented size, so every backend works, including DLSS-G)
            HMONITOR hm = MonitorFromWindow(target, MONITOR_DEFAULTTONEAREST);
            MONITORINFO mi{sizeof(mi)};
            GetMonitorInfoW(hm, &mi);
            mon = mi.rcMonitor;
            W = capW;
            H = capH;
            host.posX = mon.left;
            host.posY = mon.top;
            LOG("whole screen: capturing the monitor %ux%u at (%ld,%ld), gen=%d (%dx)\n", W, H, mon.left, mon.top,
                genFrames, genFrames + 1);
        }
        else if (g_fill)
        {
            // fullscreen on the monitor the target window currently occupies: the overlay takes
            // the monitor's size and the SERVER upscales content into it (aspect-fit letterbox)
            HMONITOR hm = MonitorFromWindow(target, MONITOR_DEFAULTTONEAREST);
            MONITORINFO mi{sizeof(mi)};
            GetMonitorInfoW(hm, &mi);
            mon = mi.rcMonitor;
            W = (uint32_t)(mon.right - mon.left);
            H = (uint32_t)(mon.bottom - mon.top);
            host.posX = mon.left;
            host.posY = mon.top;
            LOG("fill screen: capture %ux%u -> monitor %ux%u at (%ld,%ld), gen=%d (%dx)\n", capW, capH, W, H, mon.left,
                mon.top, genFrames, genFrames + 1);
        }
        else
        {
            W = capW;
            H = capH;
            // The overlay covers the CLIENT area (cap cropped the title bar out of the capture),
            // so the real title bar stays visible above it.
            host.posX = cap.clientScreenX;
            host.posY = cap.clientScreenY;
            LOG("capturing %ux%u at (%ld,%ld), gen=%d (%dx)\n", W, H, (long)cap.clientScreenX, (long)cap.clientScreenY,
                genFrames, genFrames + 1);
        }

        rc = host.init();
        if (rc)
            return rc;
        if (host.minWH && (W < host.minWH || H < host.minWH))
            LOG("WARNING: %ux%u is below the DLSS-G minimum extent %u, FG may refuse\n", W, H, host.minWH);

        // The DLSS 5 pass inside this process. Server backends only (never DLSS-G: the NR host
        // starves it). The native host runs it on the model frame after Restore and
        // the resize (nativeLiveNrInit logs its line once the model size is known, or why the
        // session runs without it).
        if (g_dlssnr && g_backend == BK_SERVER)
        {
            wchar_t nv[8]{};
            if (GetEnvironmentVariableW(L"SMV_LIVE_NR_NATIVE", nv, 8) && nv[0] == L'0')
            {
                LOG("live DLSS 5 skipped for this session: SMV_LIVE_NR_NATIVE=0 turns the pass off\n");
                g_dlssnr = false;
            }
            else
            {
                g_liveNrCuda = true;
                g_nrNative = true;
                g_nrAttempted = true;
            }
        }

        // HUD created up front (not after warmup) so server backends can show a "loading" note
        // during the cold model build: the passthrough overlay looks identical to the source
        // until smoothing starts, so without a visible cue the long first build still reads as
        // "nothing happened". WS_EX_NOACTIVATE keeps it clear of DLSS-G's foreground gate.
        // Fill presents nothing until the first served group, so its note starts at the window's corner and
        // moves to the screen's once the overlay runs.
        Hud hud;
        FillMouse fillMouse; // started once the overlay runs (Fill only), stopped before the teardown
        if (!g_noHud && !park)
            hud.create((g_fill ? cap.clientScreenX : host.posX) + 16, (g_fill ? cap.clientScreenY : host.posY) + 16);

        PipeServer srv;
        if (g_backend == BK_SERVER)
        {
            wchar_t exePath[MAX_PATH];
            GetModuleFileNameW(nullptr, exePath, MAX_PATH);
            std::wstring exeDir(exePath);
            exeDir.resize(exeDir.find_last_of(L'\\'));
            const std::wstring script = engineScript(exeDir);
            // the native host is the only route for every server backend (a session it cannot
            // run ends on its reason line). Its DLL load and engine handoff overlap the
            // source-rate measurement below (they depend on the capture size only)
            srv.beginNativeHandoff(script, g_serverBackend, capW, capH, cap.hTex, cap.hFence, &host);
            // SOURCE-RATE MEASUREMENT, before the server spawns. Two consumers:
            //  * fixed mode (--no-adapt with a --target and no --gen): seeds the whole
            //    multiplier from the fps target. Drift
            //    tracking re-derives it in the stats tick; each group's ladder carries it.
            //  * adaptive mode: sizes the SLOT CEILING. A pair can only reach the target
            //    if it carries ceil(target/source) slots, so the slot count is derived
            //    from the target here instead of being a fixed number - which is what
            //    lets four-figure targets work at all. Memory is the only cap.
            // Bounded (~2s / 9 intervals); a static or paused source falls back to a
            // nominal ~24 fps. Passthrough presents keep the overlay visibly live
            // meanwhile (SDR window mode only), exactly like the load loop below.
            const uint32_t mPitch = (W * 4 + 255) & ~255u;
            const size_t mSlotBytes = ((size_t)mPitch * H + 511) & ~(size_t)511;
            const int slotCap = slotBudget(mSlotBytes);
            if (genFrames > slotCap - 1)
            {
                LOG("--gen %d needs %.1f GB of slots, clamping to %d (memory budget)\n", genFrames,
                    2.0 * (genFrames + 1) * mSlotBytes / 1073741824.0, slotCap - 1);
                genFrames = slotCap - 1;
            }
            const bool derive = g_targetFps > 0 && !g_genExplicit;
            int ceilGen = genFrames; // slots-1 the server graph is built at
            if (derive)
            {
                double base = 0, measSec = 0;
                {
                    std::vector<uint8_t> mbuf((size_t)capW * capH * 4);
                    // HDR capture is FP16 and latestFrame is BGRA8-only, so time the
                    // GPU-side drain there instead (same cadence, no pixels needed)
                    const bool cpuRead = !g_hdr;
                    const bool canPass = cpuRead && (W == capW && H == capH);
                    auto pull = [&] { return cpuRead ? cap.latestFrame(mbuf.data()) : cap.latestFrameGpu(); };
                    // drain WGC's queued initial burst first: buffered frames arrive back to
                    // back and would bias the measured rate high (26 read for a true 21)
                    const double tD = nowQpc100() / 1e4;
                    while (nowQpc100() / 1e4 - tD < 250.0)
                    {
                        pumpMessages();
                        if (pull() <= 0)
                            Sleep(5);
                    }
                    const double t0 = nowQpc100() / 1e4;
                    double tPrev = 0, tFirst = 0, tLast = 0;
                    int intervals = 0;
                    while (nowQpc100() / 1e4 - t0 < 2000.0 && intervals < 9)
                    {
                        pumpMessages();
                        WaitForSingleObject(cap.evt, 50);
                        if (pull() > 0)
                        {
                            const double tn = nowQpc100() / 1e4;
                            if (tPrev && tn - tPrev < 300.0)
                            {
                                if (!tFirst)
                                    tFirst = tPrev;
                                tLast = tn;
                                intervals++;
                            }
                            tPrev = tn;
                            if (canPass)
                                host.presentFrame(mbuf.data());
                        }
                    }
                    if (intervals >= 4 && tLast > tFirst)
                    {
                        base = 1000.0 * intervals / (tLast - tFirst);
                        measSec = (tLast - tFirst) / 1000.0;
                    }
                }
                if (base <= 0)
                {
                    base = 24.0;
                    LOG("source rate unmeasurable (static/paused?), assuming ~24 fps\n");
                }
                // +1 slot of headroom: the pair clock jitters, so a pair occasionally holds
                // one more grid point than the nominal ratio. Clamped ONLY by memory.
                // The slot count is the HARD ceiling on the presented rate (a pair can
                // hold at most `slots` grid points, so the session tops out at base * slots).
                // The old ceil(target/base)+1 sat right on that ceiling, and `base` is a 2 s
                // one-shot measurement: reading a 23.976 fps source as 25.8 gave 40 slots and
                // hard capped a target-1000 session at 959 fps for its whole life. Adaptive
                // mode now sizes with 15% headroom on the measurement plus 2 spare slots.
                // Fixed mode keeps the classic count (its multiplier is drift-tracked).
                const int slotsFixed = (int)ceil((double)g_targetFps / base) + 1;
                int slots = g_noAdapt ? slotsFixed : (int)ceil((double)g_targetFps / base * 1.15) + 2;
                if (slots < 2)
                    slots = 2;
                bool capped = false;
                if (slots > slotCap)
                {
                    slots = slotCap;
                    capped = true;
                }
                ceilGen = slots - 1;
                if (g_noAdapt)
                {
                    int want = (int)((double)g_targetFps / base + 0.5) - 1;
                    genFrames = want < 1 ? 1 : want > ceilGen ? ceilGen : want;
                    LOG("measured ~%.1f fps source, target %d fps: fixed %dx (drift-tracked)\n", base, g_targetFps,
                        genFrames + 1);
                }
                else
                {
                    genFrames = ceilGen;
                    LOG("slots %d (target %d, source %.1f fps measured over %.2f s)%s "
                        "(memory cap %d slots at %.1f MB/slot)\n",
                        slots, g_targetFps, base, measSec, capped ? " (MEMORY-CAPPED, target unreachable)" : "",
                        slotCap, mSlotBytes / 1048576.0);
                }
            }
            // A derived fixed multiplier is drift-tracked (re-derived in the stats tick),
            // so size the server at the slot CEILING and let each group's ladder carry the
            // current value. An explicit --gen (harness/perf use) keeps the exact classic spawn.
            const bool fixedDerived = g_noAdapt && g_targetFps > 0 && !g_genExplicit;
            const int spawnGen = fixedDerived ? ceilGen : genFrames;
            // The backend's model + TRT engines can take tens of seconds to build the FIRST time
            // at a new resolution (a fullscreen capture is often a never-before-built size:
            // measured ~6s RIFE, ~35s GMFSS). srv.start() blocks for that whole cold start, and
            // server backends do no warmup pre-present, so the overlay used to show nothing until
            // it finished - which reads as "it never turned on" (while DLSS-G, no server, feels
            // instant). Load on a BACKGROUND thread and present the raw captured frames meanwhile,
            // so the overlay is visibly LIVE from the first second; it just isn't smoothed yet.
            // The cross-group queue is the only consumer of the capture-release
            // token. Decide here, before the spawn: the route's other conditions (shm,
            // streaming) are answered by the same handshake, and an unused --caprel simply
            // comes back as caprel=0.
            g_capRel = g_backend == BK_SERVER && !host.useSL;
            std::atomic<int> srvRc{-1}; // -1 loading, 0 ready, 1 failed
            std::thread loader([&] {
                srvRc.store(srv.start(script, g_serverBackend, spawnGen, capW, capH, cap.hTex, cap.hFence, &host) ? 1
                                                                                                                  : 0);
            });
            // 1:1 passthrough is only meaningful when capture and presented sizes match (window /
            // whole-screen modes); fill mode upscales inside the server, so there is nothing to
            // present yet and it simply waits (same as before, minus the black overlay).
            // HDR: the CPU passthrough is BGRA8-only. Under HDR the capture pool is FP16 (8 B/px)
            // and the swap chain is R10A2 PQ, so latestFrame's 4 B/px row copy plus a raw
            // presentFrame upload scrambles both = the purple screen seen for the whole model-load
            // window. Skip it; the overlay stays blank until the server's converted frames arrive
            // and the HUD loading note covers the gap.
            const bool canPassthrough = (W == capW && H == capH) && !g_hdr;
            std::vector<uint8_t> loadBuf((size_t)capW * capH * 4);
            const ULONGLONG loadStart = GetTickCount64();
            bool announced = false;
            ULONGLONG loadPosTick = 0;
            ULONGLONG loadSizeTick = 0;
            bool loadGone = false; // the window closed during that resize: exit 7, never revived
            while (srvRc.load(std::memory_order_acquire) < 0)
            {
                pumpMessages();
                // window tracking during the cold model build: the main loop's 250ms tracker
                // only starts after the server is up, so dragging the target here left the
                // passthrough overlay and the "loading ... model" note stuck at the old spot
                if (!host.park && !g_monitor && GetTickCount64() - loadPosTick > 250)
                {
                    loadPosTick = GetTickCount64();
                    RECT r{};
                    if (frameBounds(target, r) && (r.left != fb.left || r.top != fb.top))
                    {
                        fb = r;
                        const int ox = r.left + cap.cropX;
                        const int oy = r.top + cap.cropY;
                        if (!g_fill) // Fill's overlay covers the monitor: only its note follows the window
                            SetWindowPos(host.hwnd, HWND_TOPMOST, ox, oy, 0, 0, SWP_NOSIZE | SWP_NOACTIVATE);
                        hud.move(ox + 16, oy + 16);
                    }
                }
                // a resize during the model load: the loader cannot see it (the
                // engine handoff only polls the flag), so watch the client size here; once it
                // settles at a new size the session ends with exit 4 like a mid-session resize, the
                // app revives it at the new size and the build in flight finishes in the background
                if (!g_monitor && !g_resizeReq.load() && GetTickCount64() - loadSizeTick > 250)
                {
                    loadSizeTick = GetTickCount64();
                    RECT cr{};
                    if (IsWindow(target) && GetClientRect(target, &cr) &&
                        (abs(cr.right - (int)capW) > 2 || abs(cr.bottom - (int)capH) > 2))
                    {
                        LOG("target window resized during the model load (%ux%u -> %ldx%ld)\n", capW, capH, cr.right,
                            cr.bottom);
                        if (resizeSettle(target, host, cap, hud, false))
                        {
                            // back at the captured size (a transient): the load goes on, and Fill's
                            // loading note returns to the window (the settle put it on the overlay)
                            RECT r{};
                            if (g_fill && !host.park && frameBounds(target, r))
                                hud.move(r.left + cap.cropX + 16, r.top + cap.cropY + 16);
                        }
                        else
                        {
                            loadGone = !IsWindow(target);
                            g_resizeReq.store(true);
                        }
                    }
                }
                if (canPassthrough && !g_resizeReq.load() && cap.latestFrame(loadBuf.data()) > 0)
                    host.presentFrame(loadBuf.data());
                else
                    Sleep(8);
                if (!announced && GetTickCount64() - loadStart > 400)
                {
                    // user-facing name (the GUI's effective-model label + substitution note)
                    // over the raw backend id, so "Smooth Motion -> GMFSS live" style mappings
                    // read as intended instead of as the wrong model
                    const std::wstring what = loadingWhat();
                    const wchar_t* nm = what.c_str();
                    if (g_modelNote.empty())
                        LOG("loading %ls (first build at this resolution can take up to a minute)...\n", nm);
                    else
                        LOG("loading %ls (%ls; first build at this resolution can take up to a minute)...\n", nm,
                            g_modelNote.c_str());
                    if (hud.hwnd)
                        _snwprintf_s(hud.text, _TRUNCATE, L"SMV Live: loading %ls...", nm);
                    announced = true;
                }
                // keep the loading note on top of the passthrough presents (each Present leaves
                // the overlay above the HUD in the topmost band) and repaint it
                if (announced && hud.hwnd)
                {
                    SetWindowPos(hud.hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
                    hud.paint();
                }
            }
            loader.join();
            if (g_stopReq.load() || g_resizeReq.load())
            {
                // the session ended during the model load (hotkey off, or the resize above):
                // leave through the same teardown order as the loop's exit path so the resident
                // host stays and an engine build left running in the background survives
                const int rcLoad = loadGone ? 7 : (g_resizeReq.load() ? 4 : 0);
                LOG(rcLoad == 7   ? "target window closed during the model load\n"
                    : rcLoad == 4 ? "target window resized during the model load, restarting\n"
                                  : "stop requested during the model load, ending the session\n");
                hud.destroy();
                srv.stop();
                cap.stop();
                host.shutdown();
                if (host.useSL)
                    slShutdown();
                timeEndPeriod(1);
                if (g_nrAttempted)
                {
                    fflush(stderr);
                    ExitProcess((UINT)rcLoad);
                }
                g_sessionClean = true;
                return rcLoad;
            }
            if (srvRc.load() != 0)
            {
                LOG("live server failed to start\n");
                return 1;
            }
            if (srv.outbufAck && host.outBuf)
                LOG("present path: direct from VRAM (shared output buffer)\n");
            // a started host always holds the zero-copy capture import (nativeRefusal)
            LOG("capture path: zero-copy (shared texture, no CPU readback)\n");
            // the load-loop passthrough drained WGC's single initial frame, and a static source
            // (the start-on-paused workflow) never sends another: without this refresh, warmup
            // below would exit 1 after 5s ("no frames captured") on any paused/idle target
            cap.requestRefresh();
        }

        const size_t capBytes = (size_t)capW * capH * 4; // capture-size frames (server input)
        std::vector<uint8_t> buf(capBytes), lastBuf(capBytes);
        uint32_t shmSeq = 0;   // shm-mode: frame token sequence
        double statLatSum = 0; // capture->present latency accumulation (ms)
        uint64_t statLatN = 0;
        // ADAPTIVE SMOOTHNESS (server route, streaming only), Lossless-Scaling-style TARGET
        // mode: output frames are generated AT THE TARGET GRID's timestamps (the overlay
        // monitor's refresh rate, or --target N) by FRACTIONAL resampling - each source pair
        // gets tweens at whatever grid times fall inside it (base 25 -> ~2.4 tweens per pair
        // on a 60 grid), so the presented rate sits pinned at the target no matter where the
        // base fps drifts, exactly like the offline --fps mode. The Smoothness setting is the
        // slot CEILING per pair (it sizes the shm sets): a base below target/(gen+1) tops out
        // at the ceiling instead. Fixed mode (--no-adapt) keeps uniform fractions + the real
        // frame, byte-identical to the classic behavior.
        double adaptTarget = 0; // resolved target output fps (0 = fixed mode)
        double gridStep = 0;    // target grid period in 100ns units
        int64_t nextGrid = 0;   // next un-emitted grid timestamp (capture domain)
        int64_t prevSentTs = 0; // capture ts of the previously sent frame
        if (g_backend == BK_SERVER && !g_noAdapt)
        {
            adaptTarget = g_targetFps;
            if (adaptTarget <= 0)
            {
                MONITORINFOEXW mi{};
                mi.cbSize = sizeof(mi);
                GetMonitorInfoW(MonitorFromWindow(host.hwnd, MONITOR_DEFAULTTONEAREST), &mi);
                DEVMODEW dm{};
                dm.dmSize = sizeof(dm);
                if (EnumDisplaySettingsExW(mi.szDevice, ENUM_CURRENT_SETTINGS, &dm, 0) && dm.dmDisplayFrequency > 1)
                    adaptTarget = (double)dm.dmDisplayFrequency;
            }
            if (adaptTarget > 0)
            {
                gridStep = 1e7 / adaptTarget;
                LOG("adaptive smoothness: target %.0f fps (fractional resample, up to %dx per pair)\n", adaptTarget,
                    genFrames + 1);
            }
        }
        // DYNAMIC OUTPUT THROTTLE (the "consume every source frame" loop). The slot ceiling
        // alone lets the pipeline ask for more tweens per pair than it can actually produce
        // inside one pair interval; the loop then falls behind, WGC frames get superseded
        // (Capture.dropped climbs) and latency parks at whatever the group overrun is - the
        // 24fps source read as 9.6 consumed with ~190ms latency at 42 slots. So the grid
        // marches on an EFFECTIVE target that floats below the user's: step DOWN fast when
        // we are the bottleneck, back UP slowly when there is sustained headroom. The user's
        // target is what is displayed and the ceiling we recover toward; the effective rate
        // is what the hardware can sustain, exactly like in-game frame generation.
        // Signals (both per group, no extra measurement cost):
        //   * cap.dropped advanced over the group = a source frame was superseded, hard
        //     evidence that we did not keep up;
        //   * group wall time > the pair interval = we are over budget even if the drop has
        //     not landed yet (faster to react, and it catches the first overrun).
        // Multiplicative decrease / small additive-ish increase with a clean-time gate is the
        // standard anti-oscillation shape: a single bad group costs 10%, and winning it back
        // takes a full clean window, so the loop settles instead of pumping.
        double effTarget = adaptTarget; // throttled output target (<= adaptTarget)
        uint64_t thrDropBase = 0;       // cap.dropped at the previous evaluation
        double thrCleanMs = 0;          // accumulated clean-group time toward a step up
        double thrLastLog = 0;
        double thrWinT0 = 0; // control-window start (ms); 0 = not started
        uint32_t thrGroups = 0, thrOverruns = 0;
        const double kThrWinMs = 500; // control window: long enough to measure a drop RATE
        const double kThrDown = 0.95; // step down per bad window (every 500ms at worst)
        const double kThrUp = 1.05;   // step up per clean window (every 3s at best)
        double emaDt = 0;             // smoothed capture interval (ms), paces the group
        ULONGLONG lastArrival = 0;
        int idleSlotIdx = -1;     // last presented slot, re-presentable while the
        uint32_t idleSlotSet = 0; // source is static (the server only rewrites a half
                                  // when a NEW group lands, so the content is stable)
        auto slotOffset = [&](uint32_t set, uint32_t i) -> size_t {
            return srv.shmInBytes + ((size_t)set * srv.shmSlots + i) * srv.shmSlot;
        };
        // present one output slot GPU-direct from the shared VRAM ring (slots never touch host
        // memory)
        auto presentSlotFrom = [&](uint32_t set, uint32_t i) -> bool {
            return host.presentTail(host.outBuf.Get(), slotOffset(set, i) - srv.shmInBytes, srv.shmPitch);
        };

        auto drainQuiet = [&]() {
            UINT stable = 0;
            int quietMs = 0;
            while (quietMs < 100)
            {
                UINT c = 0;
                host.scNative->GetLastPresentCount(&c);
                if (c == stable)
                {
                    Sleep(5);
                    quietMs += 5;
                }
                else
                {
                    stable = c;
                    quietMs = 0;
                }
            }
        };
        auto resetFG = [&]() -> bool {
            sl::DLSSGOptions off{};
            off.mode = sl::DLSSGMode::eOff;
            off.flags = sl::DLSSGFlags::eRetainResourcesWhenOff;
            if (slDLSSGSetOptions(host.vp, off) != sl::Result::eOk)
                return false;
            if (!host.presentFrame(lastBuf.data()))
                return false;
            sl::DLSSGOptions on{};
            on.mode = sl::DLSSGMode::eOn;
            on.numFramesToGenerate = (uint32_t)genFrames;
            if (slDLSSGSetOptions(host.vp, on) != sl::Result::eOk)
                return false;
            for (int i = 0; i < 2; i++)
                if (!host.presentFrame(lastBuf.data()))
                    return false;
            drainQuiet();
            return true;
        };

        // warmup: block for the first frame, present it 3x so the FG feature builds
        {
            ULONGLONG deadline = GetTickCount64() + 5000;
            int got = 0;
            while (!got && GetTickCount64() < deadline)
            {
                WaitForSingleObject(cap.evt, 100);
                got = cap.latestFrame(buf.data());
                if (got < 0)
                {
                    LOG("capture failed during warmup\n");
                    return 1;
                }
            }
            if (!got)
            {
                LOG("no frames captured in 5s (window occluded by exclusive fullscreen?)\n");
                return 1;
            }
            if (host.useSL)
            {
                for (int i = 0; i < 3; i++)
                    if (!host.presentFrame(buf.data()))
                    {
                        LOG("warmup present failed\n");
                        return 1;
                    }
                sl::DLSSGState st{};
                if (slDLSSGGetState(host.vp, st, nullptr) == sl::Result::eOk)
                {
                    if (st.status != sl::DLSSGStatus::eOk)
                    {
                        LOG("DLSS-G runtime status=%d, aborting\n", (int)st.status);
                        return 1;
                    }
                    if (st.numFramesToGenerateMax && (uint32_t)genFrames > st.numFramesToGenerateMax)
                    {
                        LOG("multi-frame generation beyond %ux is not supported on this GPU (requested %ux)\n",
                            st.numFramesToGenerateMax + 1, genFrames + 1);
                        return 3;
                    }
                }
                drainQuiet();
            }
            else if (g_backend == BK_IDENTITY)
            {
                if (!host.presentFrame(buf.data()))
                {
                    LOG("first present failed\n");
                    return 1;
                }
            }
            // server backends: no pre-present (capture and output sizes differ under --fit fill;
            // the first served group arrives within one round trip anyway)
            memcpy(lastBuf.data(), buf.data(), capBytes);
            LOG("live overlay running (Esc to stop)\n");
            if (g_fill)
                hud.move(host.posX + 16, host.posY + 16);
            if (g_fill && !park && !g_noFillMouse)
            {
                int dw = 0, dh = 0, x0 = 0, y0 = 0;
                lkFitRect((int)capW, (int)capH, (int)W, (int)H, dw, dh, x0, y0);
                fillMouse.start(target, host.hwnd, mon,
                                RECT{mon.left + x0, mon.top + y0, mon.left + x0 + dw, mon.top + y0 + dh});
            }
        }

        volatile LONG running = 1;
        HANDLE diagH = nullptr;
        DiagCtx dctx{};
        if (diagSecs > 0)
        {
            RECT wr{};
            GetWindowRect(host.hwnd, &wr);
            dctx = {wr, diagSecs, &running};
            diagH = CreateThread(nullptr, 0, diagThread, &dctx, 0, nullptr);
        }

        // main loop
        ULONGLONG lastPresentTick = GetTickCount64();
        ULONGLONG lastStatTick = GetTickCount64();
        ULONGLONG lastPosTick = 0;
        UINT statPresentBase = 0;
        host.scNative->GetLastPresentCount(&statPresentBase);
        uint64_t statDropBase = 0; // cap.dropped at the last stats tick
        uint64_t statCaptured = 0;
        bool ratioWarned = false;
        int fgWantPrev = -1;  // DLSS-G target derivation: last derived gen (hysteresis, see stats)
        int fixWantPrev = -1; // fixed-ladder server derivation: same two-window hysteresis
        bool hidden = false;
        int rc2 = 0;
        // STATIC-SOURCE HOLD (server route): a paused video emits no WGC frames, so the session
        // used to sit at 0 fps (and die at warmup if started paused). Instead, once no frame has
        // arrived for >300ms, the exe re-presents the last output slot at the cadence the mode
        // implies (adaptive: the target grid; fixed: gen+1 per refresh second) and asks WGC for
        // one real frame per second via cap.requestRefresh() (an InvalidateRect on the target,
        // the one lever that makes a static window present; pool Recreate delivers nothing);
        // once armed, the hold keeps its beat through those refresh frames (only real cadence
        // ends it). The HUD then honestly shows e.g.
        // "1.0 fps -> 30.0 fps" on a paused frame - that visible line IS the ready signal for
        // the start-paused workflow - at near-zero GPU cost (presents only, no inference), and
        // a seek/frame-step while paused reaches the screen within a second.
        // QPC-based ms clock for the hold cadence (GetTickCount64 ticks at ~15.6ms)
        auto nowMs = [] { return nowQpc100() / 10000.0; };
        double nextIdleTick = 0;                      // next idle re-present deadline, nowMs domain (0 = live)
        ULONGLONG lastRefreshTick = GetTickCount64(); // last target-refresh request
        // hold rate = the mode's cadence CAPPED AT 10 FPS (Lossless Scaling parity on static
        // frames). The cap also keeps the beat well above Windows'
        // ~15.6ms timer quantization, which ate a 33ms cadence down to 22 of 30 fps (Win11
        // ignores timeBeginPeriod for windowless/occluded processes, so precise sub-50ms
        // sleeps are not reliably available here).
        double idleStepMs = adaptTarget > 0 ? gridStep / 10000.0 : 1000.0 / (genFrames + 1);
        if (idleStepMs < 100.0)
            idleStepMs = 100.0;
        // PRESENT-PACING SMOOTHING (the 360Hz vsync cross-check follow-up): the old pacing
        // computed targets in the GetTickCount64 domain (~15.6ms ticks) and presented
        // back-to-back after any sleep overshoot, so about half of all presents landed
        // under half a refresh apart and DWM coalesced them (presented ~320/s collapsed
        // to ~230 distinct frames on screen). Targets are now QPC-anchored and every
        // present is floored to one panel MODE refresh interval after the previous one -
        // the minimum spacing a frame can be displayed at. The measured VRR cadence
        // (host.refreshQpc100) is deliberately NOT the floor: it follows our own flip
        // cadence and would feed back into a plateau, the same lock-in class the
        // arrival-cadence emaDt fix removed. Waits run on a high-resolution waitable
        // timer with a short spin tail (Sleep granularity is what caused the overshoot).
        double minSpaceMs = 0;
        double panelHzMode = 0; // panel refresh from the display MODE (not the measured one)
        // the floor follows the rate we actually sustain; recomputed on every throttle step
        auto refloor = [&] {
            if (panelHzMode <= 1.0)
                return;
            // The floor is 0.95x the LONGER of the panel refresh period and the effective
            // output period. Dropping the panel term once the user's target is above the
            // refresh rate was tried and is unstable: a lower effective rate then means a
            // LONGER floor, which stretches groups, which trips the throttle again - measured
            // as a spiral down to ~160 fps. The panel period bounds that feedback.
            const double hz = effTarget > panelHzMode ? effTarget : panelHzMode;
            minSpaceMs = 1000.0 / hz * 0.95;
        };
        {
            MONITORINFOEXW mi{};
            mi.cbSize = sizeof(mi);
            GetMonitorInfoW(MonitorFromWindow(host.hwnd, MONITOR_DEFAULTTONEAREST), &mi);
            DEVMODEW dm{};
            dm.dmSize = sizeof(dm);
            if (EnumDisplaySettingsExW(mi.szDevice, ENUM_CURRENT_SETTINGS, &dm, 0) && dm.dmDisplayFrequency > 1)
            {
                // 0.95x: a full-refresh floor leaves ZERO slack when the output target equals
                // the panel rate (16 slots x 2.78ms = 44.5ms vs a 41.7ms pair at 24->360,
                // measured: capture starved at 22.8fps). 5% under-spacing keeps
                // the no-two-per-refresh intent in practice while giving the chain headroom.
                // ABOVE-REFRESH TARGETS: the no-two-per-refresh intent is unreachable by
                // definition once the requested output rate exceeds the panel mode (a 1000fps
                // target on a 360Hz panel), and keeping the panel-derived floor would simply
                // clamp the output at ~379fps. The floor is therefore 0.95x whichever period
                // is LONGER, the panel refresh or the output slot - which reduces exactly to
                // the old panel floor whenever the target sits under the refresh rate.
                // It tracks the EFFECTIVE target (see the throttle), not the user's: slots
                // that land later than their deadline present at the floor, so a floor sized
                // for an unreachable target fires the whole group back to back and then idles
                // until the next pair (measured: bunch 280/s, 1% low 28 fps at effective 430
                // with a 1000fps floor). Sizing it from the rate we actually sustain spreads
                // the group across the pair instead.
                panelHzMode = (double)dm.dmDisplayFrequency;
                refloor();
                LOG("present pacing: floor %.3fms (panel mode %u Hz, target %.0f fps%s)\n", minSpaceMs,
                    dm.dmDisplayFrequency, adaptTarget,
                    adaptTarget > panelHzMode ? ", above panel: presents run tear-allowed past refresh" : "");
            }
        }
        HANDLE paceTimer =
            CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
        auto waitUntilMs = [&](double tMs) {
            for (;;)
            {
                const double rem = tMs - nowMs();
                if (rem <= 0.03)
                    return;
                if (paceTimer && rem > 0.6)
                {
                    LARGE_INTEGER due;
                    due.QuadPart = -(LONGLONG)((rem - 0.3) * 10000.0); // relative, 100ns units
                    if (SetWaitableTimer(paceTimer, &due, 0, nullptr, nullptr, FALSE))
                    {
                        WaitForSingleObject(paceTimer, (DWORD)rem + 2);
                        continue;
                    }
                }
                YieldProcessor(); // sub-0.6ms tail: spin
            }
        };
        // SMV_LIVE_TIMING=1: per-group exe-side breakdown (streaming route), printed with
        // the 2s stats tick.
        const bool tmOn = GetEnvironmentVariableW(L"SMV_LIVE_TIMING", nullptr, 0) != 0;
        double tmEma = 0, tmNfr = 0, tmInPair = 0;
        uint32_t tmN = 0, tmDegrade = 0;
        double lastPresSched = 0; // previous present's SCHEDULED time (the floor chain anchor)
        double hitchPrev = 0;     // hitch trace (SMV_LIVE_HITCH=1, see the present queue)
        const bool hitchOn = GetEnvironmentVariableW(L"SMV_LIVE_HITCH", nullptr, 0) != 0;
        int64_t smoothTs = 0; // EMA-smoothed pair clock (see sendPairStream)
        // one streaming pair handoff: bump seq, publish the frame (fence or shm copy), build
        // the fraction list, write the message. Returns nfr, or UINT32_MAX on a write failure.
        // fraction message: one f32 per slot the server was actually sized for (shmSlots),
        // so the drift tracker can raise the ladder up to the ceiling without overrunning it
        std::vector<uint8_t> msgBuf(4 + 4 * (size_t)(srv.shmSlots ? srv.shmSlots : (uint32_t)genFrames + 1));
        auto sendPairStream = [&](int64_t sentTs) -> uint32_t {
            // SMOOTHED PAIR CLOCK for the adaptive grid: WGC delivery times quantize to the
            // panel/VRR refresh cadence (measured +-4ms), which makes pair spans bimodal
            // (~37/46ms on a 41.7ms source) - the long half then overflows the slot ceiling
            // and degrades to the ladder, skipping its grid share and capping output at
            // ~330 of a 360 target (measured). The grid marches on an EMA clock
            // instead (the source's true cadence is uniform); the /8 pull bounds drift and
            // a 1.5-step error snaps on real cadence changes (drops, seeks, pauses).
            if (adaptTarget > 0 && emaDt > 0)
            {
                const int64_t step100 = (int64_t)(emaDt * 10000.0);
                if (!smoothTs || llabs(sentTs - (smoothTs + step100)) > step100 * 3 / 2)
                    smoothTs = sentTs;
                else
                {
                    smoothTs += step100;
                    smoothTs += (sentTs - smoothTs) / 8;
                }
                sentTs = smoothTs;
            }
            ++shmSeq;
            cap.signalFence(shmSeq);
            // one frac per slot (gen tweens + the real frame); sized from the runtime gen,
            // so no fixed message capacity caps the target
            uint8_t* msg = msgBuf.data();
            float* fr = (float*)(msg + 4);
            uint32_t nfr = 0;
            if (shmSeq == 1)
            {
                fr[nfr++] = 1.0f; // very first frame: nothing to interpolate yet
            }
            else if (nextIdleTick)
            {
                // static-source hold refresh: only the real frame is needed
                // (tweens of an identical pair are identical; the idle cadence
                // supplies the output rate). nextGrid re-seeds on the next
                // real pair - an invisible phase reset on identical content.
                fr[nfr++] = 1.0f;
                nextGrid = 0;
            }
            else if (adaptTarget > 0)
            {
                const int64_t span = sentTs - prevSentTs;
                if (!nextGrid)
                    nextGrid = prevSentTs + (int64_t)gridStep;
                const uint32_t inPair =
                    span > 0 && nextGrid <= sentTs ? (uint32_t)((sentTs - nextGrid) / (int64_t)gridStep) + 1 : 0;
                tmInPair += inPair;
                if (inPair > srv.shmSlots)
                {
                    tmDegrade++;
                    // slot ceiling reached (base far below target): the first-N
                    // grid times would bunch at the pair's start, so this pair
                    // degrades to the evenly-spread classic ladder instead
                    for (int k = 1; k <= genFrames; k++)
                        fr[nfr++] = (float)k / (float)(genFrames + 1);
                    fr[nfr++] = 1.0f;
                    while (nextGrid <= sentTs)
                        nextGrid += (int64_t)gridStep;
                }
                else
                {
                    while (span > 0 && nextGrid <= sentTs)
                    {
                        fr[nfr++] = (float)((double)(nextGrid - prevSentTs) / (double)span);
                        nextGrid += (int64_t)gridStep;
                    }
                }
            }
            else
            {
                for (int k = 1; k <= genFrames; k++)
                    fr[nfr++] = (float)k / (float)(genFrames + 1);
                fr[nfr++] = 1.0f; // the real frame, bit-exact passthrough
            }
            memcpy(msg, &nfr, 4);
            if (!srv.writeFull(msg, 4 + 4 * nfr))
                return UINT32_MAX;
            prevSentTs = sentTs;
            return nfr;
        };
        // The 2s stats line, shared by both loops below. scripts/smoke.py and the GUI parse
        // "live: A captured fps -> B presented fps (ratio R, ...) latency ~Nms"; DLSS-G
        // reports its SL handoff instead of a latency (SL paces and presents internally
        // after it) and the GUI omits that. target = output frames per source frame the
        // route is asked for: the fixed ladder, or the (throttled) adaptive target over the
        // measured capture rate. The DLSS-G state fields are appended only where they exist.
        // panel = the refresh rate DWM actually ran this window (frame-statistics deltas;
        // 0.0 until two valid samples or when stats go disjoint, e.g. right after a mode
        // change), it feeds refreshQpc100 for prBunch; disp = presents DWM displayed per
        // second (PresentCount only advances for those), the vsync cross-check.
        auto logLiveStats = [&](double capFps, double outFps, double latAvg, double secs, const sl::DLSSGState* st) {
            const double ratio = capFps > 0 ? outFps / capFps : 0.0;
            const double tgt = g_backend == BK_IDENTITY           ? 1.0
                               : (host.useSL || adaptTarget <= 0) ? (double)(genFrames + 1)
                                                                  : (capFps > 0 ? effTarget / capFps : 0.0);
            const double prTot = host.prN ? host.prMsTot / host.prN : 0.0;
            const double prWait = host.prN ? host.prMsWait / host.prN : 0.0;
            const double prFlip = host.prN ? host.prMsFlip / host.prN : 0.0;
            double panelHz = 0.0, dispFps = 0.0;
            DXGI_FRAME_STATISTICS fs{};
            if (SUCCEEDED(host.scNative->GetFrameStatistics(&fs)) && fs.SyncQPCTime.QuadPart)
            {
                if (host.fsValid && fs.SyncRefreshCount > host.fsPrev.SyncRefreshCount &&
                    fs.SyncQPCTime.QuadPart > host.fsPrev.SyncQPCTime.QuadPart)
                {
                    const double dq = (double)(fs.SyncQPCTime.QuadPart - host.fsPrev.SyncQPCTime.QuadPart);
                    panelHz = (double)(fs.SyncRefreshCount - host.fsPrev.SyncRefreshCount) * qpcFreq() / dq;
                    dispFps = (double)(fs.PresentCount - host.fsPrev.PresentCount) * qpcFreq() / dq;
                    if (panelHz > 1.0)
                        host.refreshQpc100 = (int64_t)(1e7 / panelHz);
                }
                host.fsPrev = fs;
                host.fsValid = true;
            }
            double dwmHz = 0.0;
            DWM_TIMING_INFO ti{};
            ti.cbSize = sizeof(ti);
            if (SUCCEEDED(DwmGetCompositionTimingInfo(nullptr, &ti)) && ti.qpcVBlank)
            {
                if (host.dwmPrevQpc && ti.qpcVBlank > host.dwmPrevQpc)
                    dwmHz = (double)(ti.cRefresh - host.dwmPrevRefresh) * qpcFreq() /
                            (double)(ti.qpcVBlank - host.dwmPrevQpc);
                host.dwmPrevRefresh = ti.cRefresh;
                host.dwmPrevQpc = ti.qpcVBlank;
            }
            char slf[64] = "";
            if (st)
                sprintf_s(slf, " status=%d actuallyPresented=%u", (int)st->status, st->numFramesActuallyPresented);
            // identical-pair passthrough: the native host's held count, APPENDED (the GUI and
            // scripts/smoke.py read the fields before it).
            char stc[32] = "";
            if (g_staticHold)
                sprintf_s(stc, " static=%llu", (unsigned long long)g_staticHeld);
            LOG("live: %.1f captured fps -> %.1f presented fps (ratio %.2f, target %.1f) "
                "%s ~%.0fms%s present %.2fms (wait %.2f flip %.2f) drop %.1f "
                "panel %.1fHz dwm %.1fHz disp %.1f/s bunch %.1f/s%s\n",
                capFps, outFps, ratio, tgt, host.useSL ? "handoff" : "latency", latAvg, slf, prTot, prWait, prFlip,
                (cap.dropped - statDropBase) / secs, panelHz, dwmHz, dispFps, host.prBunch / secs, stc);
            return ratio;
        };
        // ================= CROSS-GROUP PRESENT QUEUE (every streaming server) =============
        // A group-serial loop (group k's tokens read in a nested loop, ALL of its presents
        // drained with waitQueue(), only then group k+1) presents an early-arriving pair's
        // first slot 20-28ms late, which was 100% of the measured 35-40ms hitches. The
        // per-group loop below serves only the direct routes (DLSS-G, identity).
        // This path keeps the protocol, the pacing formula and the floor rule and replaces
        // the structure: a reader thread owns the token stream, tokens land in ONE FIFO of
        // presentable slots that spans groups, and the barrier becomes a non-blocking
        // completed-value check on host.halfFence (GATE A) plus the capture-texture rule
        // that a new frame may only be drained once the last sent group has been OPENED by
        // the host (GATE B: its first token or end marker came back, so the host has acquired
        // the previous capture).
        const bool xqRoute = g_backend == BK_SERVER; // every model server runs the queue
        LOG("present queue: xq=%d (%s)\n", xqRoute ? 1 : 0, xqRoute ? "cross-group queue" : "direct present");
        if (xqRoute)
        {
            struct XqGroup
            {
                uint32_t seq, set, expect, nfr;
                double arrMs, spanMs;
                int64_t sentTs;
                ULONGLONG sentTick;
                bool opened; // first slot token or end marker came back
                bool capRel; // the server released the shared capture texture
            };
            struct XqSlot
            {
                uint32_t set, idx, nfr, groupSeq;
                double target, arrMs;
                int64_t sentTs;
                bool lastOfGroup;
            };
            std::deque<XqGroup> pend; // sent, not yet closed by an end marker
            std::deque<XqSlot> fifo;  // presentable slots, spanning groups
            std::deque<uint32_t> rq;  // raw tokens from the reader thread
            // The capture-release token: not a valid slot index, high bit clear so it is
            // never mistaken for an end marker.
            const uint32_t kCapRelTok = 0x7FFFFFFFu;
            std::mutex rqM;
            std::atomic<bool> rDead{false};
            HANDLE tokEvt = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            const double kXqDropRate = 3.0;   // drops/s tolerated before throttling
            const double kXqCleanMs = 1000.0; // clean time needed before a step up
            uint64_t halfDefer = 0, gateCDefer = 0, halfPend = 0;
            // Phase instrumentation (SMV_LIVE_XQPHASE=1, off by default). Per-iteration
            // QPC accumulators for every named phase of this loop, printed on the 2s tick as
            // microseconds per PRESENT. Covers the whole loop body: pump, token drain, the
            // present scheduler, housekeeping, the gate checks, the capture probe call, the
            // send, the static hold, the stats tick and the wait (kernel vs spin tail).
            bool phOn = false;
            {
                wchar_t pv[8]{};
                if (GetEnvironmentVariableW(L"SMV_LIVE_XQPHASE", pv, 8) && pv[0] == L'1')
                    phOn = true;
            }
            // Cut escapes, one env lever each so every cut can be A/B'd on its own.
            // SMV_LIVE_XQ_CAPEV: drive the capture probe from the frame pool's FrameArrived
            //   flag instead of calling TryGetNextFrame on every iteration. The loop also WAKES
            //   on the arrival event, so a frame is drained as promptly as it was before (more
            //   promptly while slots are pending, see XQ_WAITCAP), and a hard 8ms safety net
            //   probes anyway so a lost event can never strand the session. Newest-wins drain
            //   semantics are untouched: the flag only decides WHETHER to call drainNewest.
            // SMV_LIVE_XQ_WAITCAP: cap the wait at 1ms even while the FIFO holds slots, so the
            //   capture probe keeps running during a group's present span.
            auto envOn = [](const wchar_t* nm, bool dflt) {
                wchar_t v[8]{};
                if (!GetEnvironmentVariableW(nm, v, 8))
                    return dflt;
                return v[0] != L'0';
            };
            const bool capEv = envOn(L"SMV_LIVE_XQ_CAPEV", false);
            const bool waitCap = envOn(L"SMV_LIVE_XQ_WAITCAP", false);
            double lastProbeMs = 0;
            double phPump = 0, phDrain = 0, phPres = 0, phHouse = 0, phGate = 0, phProbe = 0, phSend = 0, phHold = 0,
                   phStats = 0, phWaitK = 0, phWaitS = 0;
            uint64_t phIter = 0, phPresN = 0, phProbeN = 0;
            int64_t phT = 0;
            auto phEnd = [&](double& acc) {
                if (!phOn)
                    return;
                const int64_t t = nowQpc100();
                acc += (t - phT) / 1e4;
                phT = t;
            };
            // SMV_LIVE_XQLATE=1: diagnostic escape to a LATENESS over-budget signal (how far
            // a group's last slot presented past its unfloored deadline). Measured:
            // it over-fires, because with the present floor in play the floored chain pushes
            // the last slot past its unfloored deadline on nearly every group, so the throttle
            // ratchets down to ~320 of 1000 (383 fps avg vs 810-932 with the classic signal).
            // DEFAULT = the classic group-share signal; gate C is what bounds the queue.
            bool xqLate = false;
            {
                wchar_t lv[8]{};
                if (GetEnvironmentVariableW(L"SMV_LIVE_XQLATE", lv, 8) && lv[0] == L'1')
                    xqLate = true;
            }
            // The group-close instant is ALWAYS a present time (the group's last
            // slot). The end-marker branch used to fall back to nowMs() whenever the group had
            // no slot left in the FIFO, mixing token-arrival times into a series compared
            // against emaDt; these carry the present clock into that branch.
            double xqLastPresMs = 0, xqLastLateMs = 0;
            uint32_t lastNfr = 0;     // slots the most recently sent group asked for
            double xqPrevGroupMs = 0; // last present time of the previous group (throttle)
            double hitchArrPrev = 0;
            std::thread rdTh([&] {
                for (;;)
                {
                    uint32_t v = 0;
                    if (!srv.readFullRaw(&v, 4))
                    {
                        rDead.store(true);
                        SetEvent(tokEvt);
                        return;
                    }
                    // Teardown sets rDead and aborts the host's waits; check it after
                    // every read so a token that lands during teardown cannot restart the wait
                    if (rDead.load())
                        return;
                    {
                        std::lock_guard<std::mutex> lk(rqM);
                        rq.push_back(v);
                    }
                    SetEvent(tokEvt);
                }
            });
            // wait until tMs, but wake early on a token (waitUntilMs's timer + spin tail)
            auto xqWait = [&](double tMs) {
                const int64_t w0 = phOn ? nowQpc100() : 0;
                double kern = 0;
                // the spin tail is measured as (total in here) minus (time inside the kernel
                // wait), so the YieldProcessor loop itself is not instrumented per spin
                auto fin = [&] {
                    if (!phOn)
                        return;
                    phWaitK += kern;
                    phWaitS += (nowQpc100() - w0) / 1e4 - kern;
                };
                for (;;)
                {
                    const double rem = tMs - nowMs();
                    if (rem <= 0.03)
                    {
                        fin();
                        return;
                    }
                    if (paceTimer && rem > 0.6)
                    {
                        LARGE_INTEGER due;
                        due.QuadPart = -(LONGLONG)((rem - 0.3) * 10000.0);
                        if (SetWaitableTimer(paceTimer, &due, 0, nullptr, nullptr, FALSE))
                        {
                            HANDLE hs[3] = {paceTimer, tokEvt, cap.evt};
                            const DWORD nh = capEv ? 3 : 2;
                            const int64_t k0 = phOn ? nowQpc100() : 0;
                            const DWORD w = WaitForMultipleObjects(nh, hs, FALSE, (DWORD)rem + 2);
                            if (phOn)
                                kern += (nowQpc100() - k0) / 1e4;
                            if (w == WAIT_OBJECT_0 + 1 || w == WAIT_OBJECT_0 + 2)
                            {
                                CancelWaitableTimer(paceTimer);
                                fin();
                                return;
                            }
                            continue;
                        }
                    }
                    YieldProcessor();
                }
            };
            // THROTTLE STEP, one evaluation per group as its last slot presents. The rules
            // are the classic ones (group share of the timeline > emaDt * 1.15 = over budget,
            // plus the drop-rate rule), and GATE C is what keeps the queue bounded at
            // saturation: the earlier "queue random-walks into a permanent backlog" finding
            // (latency 149 to 1863ms over 60s at target 1000) was measured BEFORE gate C
            // existed. The LATENESS signal (how far the group's last slot presented past
            // its unfloored deadline) survives only as the SMV_LIVE_XQLATE=1 diagnostic
            // escape above, because it over-fires under the present floor.
            auto xqThrottle = [&](double tn, double lateMs) {
                if (!(adaptTarget > 0 && !hidden && !nextIdleTick && emaDt > 0))
                {
                    // An alt-tab pause or a static hold invalidates the whole control
                    // window, not just its start time. Leaving thrWinT0 and the counters alone
                    // made the first group after a 15s resume close a 15s "window" and step the
                    // target on evidence gathered before the pause.
                    xqPrevGroupMs = tn;
                    thrWinT0 = 0;
                    thrGroups = thrOverruns = 0;
                    thrCleanMs = 0;
                    thrDropBase = cap.dropped;
                    return;
                }
                const double groupMs = xqPrevGroupMs > 0 ? tn - xqPrevGroupMs : 0;
                xqPrevGroupMs = tn;
                if (groupMs <= 0)
                    return;
                thrGroups++;
                if (xqLate ? (lateMs > emaDt * 0.15) : (groupMs > emaDt * 1.15))
                    thrOverruns++;
                if (!thrWinT0)
                {
                    thrWinT0 = tn;
                    thrDropBase = cap.dropped;
                    return;
                }
                if (tn - thrWinT0 < kThrWinMs)
                    return;
                const double winS = (tn - thrWinT0) / 1000.0;
                const double dropRate = (double)(cap.dropped - thrDropBase) / winS;
                // Queue path tuning. With gate C
                // bounding the queue, a superseded capture frame is normal at-capacity
                // behaviour here (the exe DEFERRED a send on purpose), so the classic
                // 1 drop/s rule fires on noise; and with the 3s recovery a single early down
                // step stranded effTarget at 340 for longer than a whole run. Drop rule at
                // 3/s, recovery after 1 clean second. Down factor, floor and the
                // majority-over-budget rule are unchanged.
                const bool bad = dropRate > kXqDropRate || thrOverruns * 2 > thrGroups;
                double next = effTarget;
                if (bad)
                {
                    thrCleanMs = 0;
                    next = effTarget * kThrDown;
                    const double floorT = 2000.0 / emaDt;
                    if (next < floorT)
                        next = floorT;
                }
                else
                {
                    thrCleanMs += tn - thrWinT0;
                    if (thrCleanMs >= kXqCleanMs)
                    {
                        thrCleanMs = 0;
                        next = effTarget * kThrUp;
                    }
                }
                if (next > adaptTarget)
                    next = adaptTarget;
                if (next < 1.0)
                    next = 1.0;
                if (next != effTarget)
                {
                    effTarget = next;
                    gridStep = 1e7 / effTarget;
                    refloor();
                    if (tn - thrLastLog > 2000.0)
                    {
                        thrLastLog = tn;
                        LOG("throttle: effective target %.0f fps of %.0f "
                            "(drops %.1f/s, %u/%u groups over the %.1fms pair)\n",
                            effTarget, adaptTarget, dropRate, thrOverruns, thrGroups, emaDt);
                    }
                }
                thrWinT0 = tn;
                thrDropBase = cap.dropped;
                thrGroups = thrOverruns = 0;
            };
            for (;;)
            {
                if (phOn)
                {
                    phIter++;
                    phT = nowQpc100();
                }
                pumpMessages();
                if (!hidden && (GetAsyncKeyState(VK_ESCAPE) & 0x8000))
                {
                    LOG("Esc pressed, exiting\n");
                    break;
                }
                if (g_stopReq.load())
                {
                    LOG("stop requested, ending the session\n");
                    break;
                }
                if (cap.closed || (!g_monitor && !IsWindow(target)))
                {
                    LOG(g_monitor ? "capture closed\n" : "target window closed\n");
                    if (!g_monitor)
                        rc2 = 7;
                    break;
                }
                phEnd(phPump);

                // ---- (i) tokens -> group FIFO
                for (;;)
                {
                    uint32_t v = 0;
                    bool have = false;
                    {
                        std::lock_guard<std::mutex> lk(rqM);
                        if (!rq.empty())
                        {
                            v = rq.front();
                            rq.pop_front();
                            have = true;
                        }
                    }
                    if (!have)
                        break;
                    if (pend.empty())
                    {
                        LOG("live server protocol error (token with no open group)\n");
                        rc2 = 1;
                        break;
                    }
                    if (v == kCapRelTok)
                    {
                        // Capture released. The server is single threaded and strictly
                        // group ordered, so this belongs to the oldest group that has not
                        // been released yet. It is NOT a slot and does not open the group.
                        for (auto& gg : pend)
                            if (!gg.capRel)
                            {
                                gg.capRel = true;
                                break;
                            }
                        continue;
                    }
                    XqGroup& g = pend.front();
                    g.opened = true;
                    const bool last = (v & 0x80000000u) != 0;
                    const uint32_t raw = v & 0x7FFFFFFFu;
                    if (raw == 0)
                    {
                        // bare end marker: closes the group (an empty group opens and closes here)
                        const uint32_t seq = g.seq;
                        pend.pop_front();
                        bool tagged = false;
                        for (auto it = fifo.rbegin(); it != fifo.rend(); ++it)
                            if (it->groupSeq == seq)
                            {
                                it->lastOfGroup = true;
                                tagged = true;
                                break;
                            }
                        if (!tagged)
                            xqThrottle(xqLastPresMs > 0 ? xqLastPresMs : nowMs(), xqLastLateMs);
                        continue;
                    }
                    const uint32_t idx = raw - 1;
                    if (idx >= g.nfr)
                    {
                        LOG("live server protocol error (slot index)\n");
                        rc2 = 1;
                        break;
                    }
                    XqSlot s{g.set,   idx,      g.nfr, g.seq, g.arrMs + g.spanMs * (idx + 1) / (g.expect + 1),
                             g.arrMs, g.sentTs, false};
                    if (last)
                    {
                        s.lastOfGroup = true;
                        pend.pop_front();
                    }
                    fifo.push_back(s);
                }
                if (rc2)
                    break;
                bool rqEmpty = false;
                {
                    std::lock_guard<std::mutex> lk(rqM);
                    rqEmpty = rq.empty();
                }
                if (rqEmpty && rDead.load())
                {
                    LOG(srv.stalled ? "exiting for safe-mode restart\n" : "live server protocol error\n");
                    rc2 = srv.stalled ? 6 : 1;
                    break;
                }
                // Stall watchdog on this path (see readFullRaw): ioSince always equals the
                // send time of the OLDEST outstanding group, or 0 when none is outstanding.
                InterlockedExchange64(&srv.ioSince, pend.empty() ? 0 : (LONGLONG)pend.front().sentTick);
                phEnd(phDrain);

                // ---- (ii) present every slot that is due
                double nextDue = 0;
                while (!fifo.empty() && !rc2)
                {
                    XqSlot& s = fifo.front();
                    double dueMs = s.target;
                    if (minSpaceMs > 0)
                    {
                        const double prev =
                            lastPresSched > 0 ? lastPresSched : (host.prLastQpc100 ? host.prLastQpc100 / 10000.0 : 0);
                        const double floorMs = prev + minSpaceMs;
                        if (prev > 0 && floorMs > dueMs)
                            dueMs = floorMs;
                    }
                    const double tNow = nowMs();
                    if (tNow < dueMs)
                    {
                        nextDue = dueMs;
                        break;
                    }
                    lastPresSched = (tNow - dueMs <= 0.5) ? dueMs : tNow;
                    if (!presentSlotFrom(s.set, s.idx))
                    {
                        LOG("presentFrame failed\n");
                        rc2 = 1;
                        break;
                    }
                    phPresN++;
                    host.halfFence[s.set] = host.fenceValue; // GATE A reference for this half
                    idleSlotSet = s.set;
                    idleSlotIdx = (int)s.idx;
                    const double pNow = nowMs();
                    if (s.sentTs)
                    {
                        statLatSum += (nowQpc100() - s.sentTs) / 10000.0 + srv.contentLag * emaDt;
                        statLatN++;
                    }
                    if (hitchOn)
                    {
                        if (hitchPrev > 0 && pNow - hitchPrev > 8.0)
                            LOG("[hitch] gap %.1fms | slot %u/%u | sinceArr %.1f | "
                                "late %.2f | slotWait %.1f | arrGap %.1f | ema %.1f | eff %.0f\n",
                                pNow - hitchPrev, s.idx + 1, s.nfr, pNow - s.arrMs, pNow - dueMs, 0.0,
                                s.arrMs - hitchArrPrev, emaDt, effTarget);
                        hitchPrev = pNow;
                    }
                    const bool wasLast = s.lastOfGroup;
                    const double lateMs = tNow - s.target; // vs the UNFLOORED deadline
                    xqLastPresMs = pNow;                   // F5: the one group-close clock
                    xqLastLateMs = lateMs;
                    fifo.pop_front();
                    if (wasLast)
                        xqThrottle(pNow, lateMs);
                }
                if (rc2)
                    break;
                phEnd(phPres);

                // ---- (iii) housekeeping (nothing is due right now)
                ULONGLONG now = GetTickCount64();
                if (!host.park && !g_monitor && now - lastPosTick > 250)
                {
                    lastPosTick = now;
                    const bool wantHidden = livePauseWanted(target, host.hwnd);
                    if (wantHidden != hidden)
                    {
                        hidden = wantHidden;
                        ShowWindow(host.hwnd, hidden ? SW_HIDE : SW_SHOWNA);
                        hud.show(!hidden);
                        LOG(hidden ? "paused (the player is covered or minimized)\n" : "resumed\n");
                        xqPrevGroupMs = 0; // the throttle reference does not survive a pause
                        if (g_liveNrCuda)
                            g_liveNrReset.store(true); // DLSS 5: a new stream after the gap
                    }
                    if (!hidden)
                    {
                        if (g_fill)
                        {
                            HMONITOR hm = MonitorFromWindow(target, MONITOR_DEFAULTTONEAREST);
                            MONITORINFO mi{sizeof(mi)};
                            GetMonitorInfoW(hm, &mi);
                            if (mi.rcMonitor.left != mon.left || mi.rcMonitor.top != mon.top ||
                                mi.rcMonitor.right != mon.right || mi.rcMonitor.bottom != mon.bottom)
                            {
                                LOG("target moved to another monitor\n");
                                rc2 = 4;
                                break;
                            }
                        }
                        else
                        {
                            RECT r{};
                            if (frameBounds(target, r) && (r.left != fb.left || r.top != fb.top))
                            {
                                fb = r;
                                const int ox = r.left + cap.cropX;
                                const int oy = r.top + cap.cropY;
                                SetWindowPos(host.hwnd, HWND_TOPMOST, ox, oy, 0, 0, SWP_NOSIZE | SWP_NOACTIVATE);
                                hud.move(ox + 16, oy + 16);
                            }
                        }
                    }
                }

                phEnd(phHouse);

                // ---- (iv) capture poll behind GATE A (half safety) and GATE B (capture texture)
                const bool gpuCap = cap.interop && srv.captexAck;
                const uint32_t nextSet = (uint32_t)((shmSeq + 1) % 2);
                // GATE A, part 1: the last present out of the half we are about to hand to
                // the server has completed on the GPU.
                const bool gateA = host.fence->GetCompletedValue() >= host.halfFence[nextSet];
                // GATE A, part 2: that half must also hold no slot the FIFO has not
                // presented yet. Parts 1 and C alone leave a hole - a half whose presents all
                // completed can still own queued, unpresented slots, and the server would
                // overwrite them mid-queue.
                bool halfBusy = false;
                for (const auto& fs : fifo)
                    if (fs.set == nextSet)
                    {
                        halfBusy = true;
                        break;
                    }
                // GATE B: the server has released the shared capture texture for the last group we
                // sent. The capture-release token says that directly; without it (a server that
                // did not ack caprel) we fall back to the old proxy, the group being OPENED.
                const bool gateB = pend.empty() || (srv.capRelAck ? pend.back().capRel : pend.back().opened);
                // GATE C: never queue more than one full group of unpresented slots. The
                // per-group barrier used to be the backpressure; without a cap the FIFO
                // keeps whatever backlog it accumulates and latency grows without bound.
                const bool gateC = lastNfr == 0 || fifo.size() <= lastNfr;
                if (!gateA)
                    halfDefer++;
                if (halfBusy)
                    halfPend++;
                if (!gateC)
                    gateCDefer++;
                // XQ_CAPEV: skip the TryGetNextFrame call unless the pool signalled an arrival
                // (or the 8ms safety net is due). The flag is cleared only when the probe
                // actually runs, so an arrival that lands while the gates are shut is drained
                // as soon as they open, exactly as before.
                bool wantProbe = true;
                if (capEv)
                {
                    const double tp = nowMs();
                    wantProbe = InterlockedExchange(&cap.arrived, 0) != 0 || lastProbeMs == 0 || tp - lastProbeMs > 8.0;
                    if (wantProbe)
                        lastProbeMs = tp;
                }
                phEnd(phGate);
                if (gateA && !halfBusy && gateB && gateC && wantProbe)
                {
                    const int g2 = gpuCap ? cap.latestFrameGpu() : cap.latestFrame(buf.data());
                    phProbeN++;
                    phEnd(phProbe);
                    if (g2 == -1)
                    {
                        LOG("capture readback failed\n");
                        rc2 = 1;
                        break;
                    }
                    if (g2 == -2 && (g_monitor || !resizeSettle(target, host, cap, hud, hidden)))
                    {
                        rc2 = endAfterResize(target);
                        break;
                    }
                    if (g2 == 1)
                    {
                        if (cap.emaArrMs > 0)
                            emaDt = cap.emaArrMs;
                        else if (lastArrival && GetTickCount64() > lastArrival)
                        {
                            const double d = (double)(GetTickCount64() - lastArrival);
                            if (d < 300.0)
                                emaDt = emaDt > 0 ? emaDt * 0.8 + d * 0.2 : d;
                        }
                        lastArrival = GetTickCount64();
                        // hidden (alt-tab): frames are drained but NOT sent, exactly as today
                        if (!hidden)
                        {
                            const double arrMs = nowMs();
                            const int64_t sentTs = cap.lastFrameTs;
                            const uint32_t nfr = sendPairStream(sentTs);
                            if (nfr == UINT32_MAX)
                            {
                                LOG("live server write failed\n");
                                rc2 = 1;
                                break;
                            }
                            const ULONGLONG sendTick = GetTickCount64();
                            if (pend.empty())
                                InterlockedExchange64(&srv.ioSince, (LONGLONG)sendTick);
                            pend.push_back(XqGroup{shmSeq, (uint32_t)(shmSeq % 2), nfr ? nfr : 1, nfr, arrMs, emaDt,
                                                   sentTs, sendTick, false, false});
                            hitchArrPrev = arrMs;
                            lastNfr = nfr ? nfr : 1;
                            if (tmOn)
                            {
                                tmEma += emaDt;
                                tmNfr += nfr;
                                tmN++;
                            }
                            const ULONGLONG prevProcTick = lastPresentTick;
                            lastPresentTick = GetTickCount64();
                            statCaptured++;
                            if (lastPresentTick - prevProcTick < 300)
                                nextIdleTick = 0;
                            else if (nextIdleTick > 0)
                                nextIdleTick = nowMs() + idleStepMs;
                        }
                    }
                    phEnd(phSend);
                }

                // ---- (v) static-source hold (see the block comment above the loop). Once
                // armed (nextIdleTick > 0) it keeps beating through its own 1 Hz refresh
                // frames: gating it only on 300ms of silence left a 300ms hole after every
                // refresh, 1 real + 7 held = 8 of the 10 fps cap ("1/8 instead of 1/10").
                // Real cadence (<300ms apart) clears nextIdleTick at the send.
                if (!hidden && (nextIdleTick > 0 || GetTickCount64() - lastPresentTick > 300))
                {
                    const ULONGLONG now3 = GetTickCount64();
                    if (now3 - lastRefreshTick >= 1000)
                    {
                        lastRefreshTick = now3;
                        cap.requestRefresh();
                        if (nextIdleTick > 0)
                            nextIdleTick = nowMs() + idleStepMs;
                    }
                    if (idleSlotIdx >= 0)
                    {
                        const double nowQ = nowMs();
                        if (nextIdleTick <= 0)
                        {
                            nextIdleTick = nowQ + idleStepMs;
                            xqPrevGroupMs = 0;
                        }
                        else if (nowQ >= nextIdleTick)
                        {
                            if (!presentSlotFrom(idleSlotSet, (uint32_t)idleSlotIdx))
                            {
                                LOG("presentFrame failed\n");
                                rc2 = 1;
                                break;
                            }
                            host.halfFence[idleSlotSet] = host.fenceValue;
                            nextIdleTick += idleStepMs;
                            if (nextIdleTick <= nowQ)
                                nextIdleTick = nowQ + idleStepMs;
                        }
                    }
                }

                phEnd(phHold);

                // ---- (vi) 2s stats / HUD tick (server route fields only)
                if (now - lastStatTick >= 2000)
                {
                    UINT c = 0;
                    host.scNative->GetLastPresentCount(&c);
                    const double secs = (now - lastStatTick) / 1000.0;
                    const double dropWin = (double)(cap.dropped - statDropBase);
                    const double capFps = (statCaptured + dropWin) / secs;
                    const double outFps = (c - statPresentBase) / secs;
                    const double latAvg = statLatN ? statLatSum / statLatN : 0.0;
                    logLiveStats(capFps, outFps, latAvg, secs, nullptr);
                    {
                        uint64_t hn = 0;
                        for (int i = 0; i < Host::kHistN; i++)
                            hn += host.prHist[i];
                        if (hn > 20)
                        {
                            double avg = 0, l1 = 0, l01 = 0;
                            Host::histStats(host.prHist, hn, avg, l1, l01);
                            LOG("live pacing: present spacing %.0f fps avg | 1%% low %.0f | 0.1%% low %.0f "
                                "(%llu presents, effective target %.0f of %.0f) xq=1 halfdefer %llu "
                                "halfpend %llu gatec %llu caprel %d\n",
                                avg, l1, l01, (unsigned long long)hn, effTarget, adaptTarget,
                                (unsigned long long)halfDefer, (unsigned long long)halfPend,
                                (unsigned long long)gateCDefer, srv.capRelAck ? 1 : 0);
                        }
                        memset(host.prHist, 0, sizeof(host.prHist));
                    }
                    if (phOn && phPresN)
                    {
                        const double d = (double)phPresN;
                        LOG("[xq-phase] us/present over %llu presents (%llu iters, %llu probes): "
                            "pump %.0f drain %.0f present %.0f house %.0f gate %.0f probe %.0f "
                            "send %.0f hold %.0f stats %.0f waitK %.0f waitSpin %.0f | sum %.0f\n",
                            (unsigned long long)phPresN, (unsigned long long)phIter, (unsigned long long)phProbeN,
                            phPump * 1000 / d, phDrain * 1000 / d, phPres * 1000 / d, phHouse * 1000 / d,
                            phGate * 1000 / d, phProbe * 1000 / d, phSend * 1000 / d, phHold * 1000 / d,
                            phStats * 1000 / d, phWaitK * 1000 / d, phWaitS * 1000 / d,
                            (phPump + phDrain + phPres + phHouse + phGate + phProbe + phSend + phHold + phStats +
                             phWaitK + phWaitS) *
                                1000 / d);
                        phPump = phDrain = phPres = phHouse = phGate = phProbe = phSend = phHold = phStats = phWaitK =
                            phWaitS = 0;
                        phIter = phPresN = phProbeN = 0;
                    }
                    if (tmOn && tmN)
                    {
                        // xq path: the per-phase token/pace/present/drain split does not exist
                        // any more (no nested token loop, no per-group drain); the remaining
                        // fields are the pair clock, the requested slots and the queue state.
                        LOG("[exe-timing] xq: emaDt %.1fms | nfr %.2f | inPair %.2f | degrades %u "
                            "| shmSlots %u | pend %u fifo %u | halfdefer %llu (avg over %u groups)\n",
                            tmEma / tmN, tmNfr / tmN, tmInPair / tmN, tmDegrade, srv.shmSlots, (unsigned)pend.size(),
                            (unsigned)fifo.size(), (unsigned long long)halfDefer, tmN);
                        tmEma = tmNfr = tmInPair = 0;
                        tmN = 0;
                        tmDegrade = 0;
                    }
                    statDropBase = cap.dropped;
                    host.prMsTot = host.prMsWait = host.prMsFlip = 0;
                    host.prN = 0;
                    host.prBunch = 0;
                    hud.update(capFps, outFps, latAvg);
                    statLatSum = 0;
                    statLatN = 0;
                    if (g_noAdapt && g_targetFps > 0 && !g_genExplicit && capFps > 3.0)
                    {
                        int want = (int)((double)g_targetFps / capFps + 0.5) - 1;
                        if (want < 1)
                            want = 1;
                        const int wantCap = srv.shmSlots ? (int)srv.shmSlots - 1 : genFrames;
                        if (want > wantCap)
                            want = wantCap;
                        if (want != genFrames && want == fixWantPrev)
                        {
                            LOG("target %d fps at ~%.1f captured: switching fixed multiplier to %dx\n", g_targetFps,
                                capFps, want + 1);
                            genFrames = want;
                            idleStepMs = 1000.0 / (genFrames + 1);
                            if (idleStepMs < 100.0)
                                idleStepMs = 100.0;
                        }
                        fixWantPrev = want;
                    }
                    lastStatTick = now;
                    statPresentBase = c;
                    statCaptured = 0;
                }
                phEnd(phStats);

                // ---- (vii) wait: the next present deadline, a token, or 1ms of capture polling
                double dl = nowMs() + 1.0;
                if (nextDue > 0 && nextDue < dl)
                    dl = nextDue;
                if (!waitCap && !fifo.empty() && nextDue > 0)
                    dl = nextDue;
                xqWait(dl);
            }
            // Reader teardown. The reader waits on the host's token condition, never on I/O:
            // nativeAbort marks the host dead under that condition's lock and wakes it, so the
            // read fails whether the reader was already waiting or had not entered the read yet
            // (a stop on the first loop iteration). The thread captures this stack frame by
            // reference, so detaching is never an option: it must be joined.
            if (g_teardownTrace)
                LOG("teardown: loop left (rc2=%d), stopping the token reader\n", rc2);
            rDead.store(true);
            srv.nativeAbort();
            if (WaitForSingleObject(rdTh.native_handle(), 3000) != WAIT_OBJECT_0)
                LOG("note: token reader did not exit within 3 s, joining\n");
            if (rdTh.joinable())
                rdTh.join();
            CloseHandle(tokEvt);
            if (g_teardownTrace)
                LOG("teardown: token reader joined, draining the present queue\n");
            host.waitQueue(); // one final drain before the shared slots go away
            if (g_teardownTrace)
                LOG("teardown: present queue drained\n");
        }
        else
            for (;;)
            {
                DWORD waitMs = 50;
                if (nextIdleTick > 0) // idle cadence can be finer than 50ms (e.g. 33ms at target 30)
                {
                    const double d = nextIdleTick - nowMs();
                    waitMs = d > 1.0 ? (DWORD)d : 1;
                    if (waitMs > 50)
                        waitMs = 50;
                }
                WaitForSingleObject(cap.evt, waitMs);
                pumpMessages();

                // Esc is read globally (GetAsyncKeyState): only honor it while engaged, else typing
                // Esc in an unrelated app would kill a paused session
                if (!hidden && (GetAsyncKeyState(VK_ESCAPE) & 0x8000))
                {
                    LOG("Esc pressed, exiting\n");
                    break;
                }
                if (g_stopReq.load())
                {
                    LOG("stop requested, ending the session\n");
                    break;
                }
                // monitor mode: the target window only seeded the monitor choice; its lifetime and
                // focus are irrelevant (whole-screen smoothing keeps running until Esc/hotkey/Stop)
                if (cap.closed || (!g_monitor && !IsWindow(target)))
                {
                    LOG(g_monitor ? "capture closed\n" : "target window closed\n");
                    if (!g_monitor)
                        rc2 = 7;
                    break;
                }

                ULONGLONG now = GetTickCount64();
                if (!host.park && !g_monitor && now - lastPosTick > 250) // monitor mode: no pause/tracking
                {
                    lastPosTick = now;
                    // The pause (livePauseWanted): the player minimized or covered by the window in
                    // front hides the overlay and stops processing; a window in front elsewhere does
                    // not. A resume shows the overlay without activating it (SW_SHOWNA): the focus
                    // stays wherever the user put it.
                    const bool wantHidden = livePauseWanted(target, host.hwnd);
                    if (wantHidden != hidden)
                    {
                        hidden = wantHidden;
                        ShowWindow(host.hwnd, hidden ? SW_HIDE : SW_SHOWNA);
                        hud.show(!hidden);
                        LOG(hidden ? "paused (the player is covered or minimized)\n" : "resumed\n");
                    }
                    if (!hidden)
                    {
                        if (g_fill)
                        {
                            // fullscreen overlays don't track the window; they track its MONITOR
                            HMONITOR hm = MonitorFromWindow(target, MONITOR_DEFAULTTONEAREST);
                            MONITORINFO mi{sizeof(mi)};
                            GetMonitorInfoW(hm, &mi);
                            if (mi.rcMonitor.left != mon.left || mi.rcMonitor.top != mon.top ||
                                mi.rcMonitor.right != mon.right || mi.rcMonitor.bottom != mon.bottom)
                            {
                                LOG("target moved to another monitor\n");
                                rc2 = 4; // the app restarts the overlay onto the new monitor
                                break;
                            }
                        }
                        else
                        {
                            RECT r{};
                            if (frameBounds(target, r) && (r.left != fb.left || r.top != fb.top))
                            {
                                fb = r;
                                // The overlay tracks the CLIENT origin (frame origin + constant crop
                                // offset), so it keeps covering the client area as the window moves.
                                const int ox = r.left + cap.cropX;
                                const int oy = r.top + cap.cropY;
                                SetWindowPos(host.hwnd, HWND_TOPMOST, ox, oy, 0, 0, SWP_NOSIZE | SWP_NOACTIVATE);
                                hud.move(ox + 16, oy + 16);
                            }
                        }
                    }
                }

                if (g_verbose)
                    LOG("[loop] top hidden=%d\n", (int)hidden);
                const bool gpuCap = cap.interop && srv.captexAck; // zero-copy capture active
                int got = gpuCap ? cap.latestFrameGpu() : cap.latestFrame(buf.data());
                if (g_verbose)
                    LOG("[loop] latest=%d\n", got);
                if (got == -1)
                {
                    LOG("capture readback failed\n");
                    rc2 = 1;
                    break;
                }
                if (got == -2 && (g_monitor || !resizeSettle(target, host, cap, hud, hidden)))
                {
                    rc2 = endAfterResize(target);
                    break;
                }
                // paused (alt-tab): frames are drained but not processed (the overlay is hidden)
                if (got == 1 && !hidden)
                {
                    // a long input gap (paused video, idle desktop) poisons the DLSS-G pacer: re-warm
                    if (host.useSL && now - lastPresentTick > 700)
                    {
                        LOG("input gap %llums, resetting DLSS-G\n", (unsigned long long)(now - lastPresentTick));
                        if (!resetFG())
                        {
                            LOG("DLSS-G reset failed\n");
                            rc2 = 1;
                            break;
                        }
                    }
                    if (!host.presentFrame(buf.data()))
                    {
                        LOG("presentFrame failed\n");
                        rc2 = 1;
                        break;
                    }
                    // direct routes (dlssg/identity): latency of the frame just presented
                    statLatSum += (nowQpc100() - cap.lastFrameTs) / 10000.0;
                    statLatN++;
                    if (!gpuCap)
                        memcpy(lastBuf.data(), buf.data(), capBytes); // lastBuf = SL-reset fodder only
                    const ULONGLONG prevProcTick = lastPresentTick;
                    lastPresentTick = GetTickCount64();
                    statCaptured++;
                    // leave the static-source hold only when REAL cadence resumed (<300ms between
                    // frames); the hold's own 1 Hz refresh frames keep the idle cadence running,
                    // else every refresh costs a 300ms re-arm gap (~measured 25 of 30 fps held).
                    // A refresh frame's real-slot present consumes the current beat (without this
                    // the hold ran one present/s hot: 3.0 instead of 2.0 at 2x).
                    if (lastPresentTick - prevProcTick < 300)
                        nextIdleTick = 0;
                    else if (nextIdleTick > 0)
                        nextIdleTick = nowMs() + idleStepMs;
                }

                if (now - lastStatTick >= 2000)
                {
                    UINT c = 0;
                    host.scNative->GetLastPresentCount(&c);
                    double secs = (now - lastStatTick) / 1000.0;
                    // SOURCE CADENCE, not consumed cadence: cap.dropped counts frames WGC
                    // delivered that the loop superseded before it could use them, so
                    // captured + dropped is the rate the target app actually produces. The
                    // consumed count alone sags with our own group cost (a 24fps source read
                    // 9.6 at gen 47 while mpv kept presenting 24), which made the "in" number
                    // look like the source had slowed down. This one field feeds the stats
                    // line, the on-screen HUD and the GUI's lvstat, so they all agree.
                    const double dropWin = (double)(cap.dropped - statDropBase);
                    double capFps = (statCaptured + dropWin) / secs;
                    double outFps = (c - statPresentBase) / secs;
                    sl::DLSSGState st{};
                    if (host.useSL)
                        slDLSSGGetState(host.vp, st, nullptr);
                    // server/identity: true capture->present latency (we present every slot).
                    // DLSS-G: only the capture->SL-handoff is visible to us.
                    const double latAvg = statLatN ? statLatSum / statLatN : 0.0;
                    const double ratio = logLiveStats(capFps, outFps, latAvg, secs, host.useSL ? &st : nullptr);
                    {
                        uint64_t hn = 0;
                        for (int i = 0; i < Host::kHistN; i++)
                            hn += host.prHist[i];
                        if (hn > 20)
                        {
                            double avg = 0, l1 = 0, l01 = 0;
                            Host::histStats(host.prHist, hn, avg, l1, l01);
                            // NOT "live: " - scripts/smoke.py reads the LAST line with that exact
                            // prefix as the captured/presented stats line
                            LOG("live pacing: present spacing %.0f fps avg | 1%% low %.0f | 0.1%% low %.0f "
                                "(%llu presents, effective target %.0f of %.0f)\n",
                                avg, l1, l01, (unsigned long long)hn, effTarget, adaptTarget);
                        }
                        memset(host.prHist, 0, sizeof(host.prHist));
                    }
                    statDropBase = cap.dropped;
                    host.prMsTot = host.prMsWait = host.prMsFlip = 0;
                    host.prN = 0;
                    host.prBunch = 0;
                    // HUD estimate: server routes show the measured number; DLSS-G shows the
                    // handoff plus one capture interval (the inherent interpolation delay SL
                    // adds while it waits for the next real frame to interpolate toward).
                    hud.update(capFps, outFps, host.useSL && capFps > 0.5 ? latAvg + 1000.0 / capFps : latAvg);
                    statLatSum = 0;
                    statLatN = 0;
                    // (adaptive smoothness needs no controller here: the target grid IS the
                    // policy - every stats window simply reflects however many grid frames the
                    // source pairs produced. cap.dropped remains available for diagnostics.)
                    // Multiplier-free targeting for DLSS-G: with no multiplier knob left, derive the
                    // generated-frame count from the fps target and the measured capture rate, capped
                    // by the model (numFramesToGenerateMax, 5 on current NGX). Applied through the
                    // existing off/on re-warm, and only after the same value is derived twice in a row
                    // (2s stats windows), so a momentary rate wobble cannot thrash the FG pipeline.
                    if (host.useSL && g_targetFps > 0 && !g_genExplicit && capFps > 3.0)
                    {
                        const int capMax = host.maxGen ? (int)(host.maxGen < 5 ? host.maxGen : 5) : 5;
                        int want = (int)((double)g_targetFps / capFps + 0.5) - 1;
                        if (want < 1)
                            want = 1;
                        if (want > capMax)
                            want = capMax;
                        if (want != genFrames && want == fgWantPrev)
                        {
                            LOG("target %d fps at ~%.1f captured: switching DLSS-G to %dx\n", g_targetFps, capFps,
                                want + 1);
                            genFrames = want;
                            if (!resetFG())
                            {
                                LOG("DLSS-G reset failed\n");
                                rc2 = 1;
                                break;
                            }
                        }
                        fgWantPrev = want;
                    }
                    // The same drift tracking for the fixed-ladder server route.
                    // The server is sized at the 16x ceiling and each group's ladder carries the
                    // current multiplier, so retargeting is just resizing the ladder. Streaming only (the batch fallback has no per-group
                    // ladder); capFps > 3 skips static holds, whose ~0 rate would derive the cap.
                    if (g_backend == BK_SERVER && g_noAdapt && g_targetFps > 0 && !g_genExplicit && srv.streamAck &&
                        capFps > 3.0)
                    {
                        int want = (int)((double)g_targetFps / capFps + 0.5) - 1;
                        if (want < 1)
                            want = 1;
                        // ceiling = the slots the server was actually sized for
                        const int wantCap = srv.shmSlots ? (int)srv.shmSlots - 1 : genFrames;
                        if (want > wantCap)
                            want = wantCap;
                        if (want != genFrames && want == fixWantPrev)
                        {
                            LOG("target %d fps at ~%.1f captured: switching fixed multiplier to %dx\n", g_targetFps,
                                capFps, want + 1);
                            genFrames = want;
                            idleStepMs = 1000.0 / (genFrames + 1);
                            if (idleStepMs < 100.0)
                                idleStepMs = 100.0;
                        }
                        fixWantPrev = want;
                    }
                    if (host.useSL && !ratioWarned && capFps > 3.0 && ratio < 1.2)
                    {
                        ratioWarned = true;
                        LOG("NOTE: presented/captured ratio is ~1x. Either the driver is using hardware "
                            "flip metering (presents not counted, FG may still be fine - check visually) "
                            "or FG is preempted (RTX Video enhancement on a playing browser video is a "
                            "known machine-wide preemptor on this hardware).\n");
                    }
                    lastStatTick = now;
                    statPresentBase = c;
                    statCaptured = 0;
                }
            }

        running = 0;
        fillMouse.stop(); // hands the cursor and the pointer speed back before the overlay goes
        if (host.prHistNAll > 20)
        {
            double avg = 0, l1 = 0, l01 = 0;
            Host::histStats(host.prHistAll, host.prHistNAll, avg, l1, l01);
            LOG("live SUMMARY: %.1f fps avg presented | 1%% low %.1f | 0.1%% low %.1f "
                "over %llu presents\n",
                avg, l1, l01, (unsigned long long)host.prHistNAll);
        }
        if (paceTimer)
            CloseHandle(paceTimer);
        hud.destroy();
        if (diagH)
        {
            WaitForSingleObject(diagH, 12000);
            CloseHandle(diagH);
        }
        if (g_teardownTrace)
            LOG("teardown: hud gone, stopping the server\n");
        if (g_backend == BK_SERVER)
            srv.stop();
        if (g_teardownTrace)
            LOG("teardown: server stopped, stopping the capture\n");
        cap.stop();
        if (g_teardownTrace)
            LOG("teardown: capture stopped, shutting the host down\n");
        host.shutdown();
        if (g_teardownTrace)
            LOG("teardown: host down\n");
        if (host.useSL)
            slShutdown();
        timeEndPeriod(1);
        // With the NR snippet loaded, the process-exit teardown after main returns
        // faults (0xC0000005 on a clean "target window closed" exit;
        // the offline host leaves through ExitProcess for the same reason), which would
        // also replace the exit codes the app acts on (4 resize restart, 6 stall revive).
        if (g_nrAttempted)
        {
            fflush(stderr);
            ExitProcess((UINT)rc2);
        }
        g_sessionClean = true; // resident host: only this exit path may keep the process
        return rc2;
    }
}
