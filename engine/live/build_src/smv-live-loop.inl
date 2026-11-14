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

// Live DLSS 4.5's rungs: every product of one, two or three passes of 2..lim (lim = numFramesToGenerateMax + 1, the
// one-pass limit; the cascade below: stage 1 = dlssg2f --stage1, a second helper for three passes, the last pass = the
// session's own DLSS-G), the passes ASCENDING: a generated midpoint is the same frame whatever the pass count, and each
// later pass works on pairs closer together, where DLSS-G's flow is better (measured against 1000 fps truth on the file
// route: 2x then 3x beats one 6x pass by 0.8 dB on 30 ms pairs). lim 6: one pass 2..6; two passes 4, 6, 8, 9, 10, 12,
// 15, 16, 18, 20, 24, 25, 30, 36 (a x b, a <= b: 15x = 3x then 5x = 360 fps from 24); three a x b x c up to 216.
// Sorted by the multiple; the same multiple with fewer passes first, then the smaller first pass (12x = 2x then 6x).
struct DlssgRung
{
    int m, a, b, c; // m = a * b * c; b, c = 1 for fewer passes
};
static std::vector<DlssgRung> dlssgRungs(int lim, int passes)
{
    std::vector<DlssgRung> v;
    for (int a = 2; a <= lim; a++)
    {
        v.push_back({a, a, 1, 1});
        if (passes >= 2)
            for (int b = a; b <= lim; b++)
            {
                v.push_back({a * b, a, b, 1});
                if (passes >= 3)
                    for (int c = b; c <= lim; c++)
                        v.push_back({a * b * c, a, b, c});
            }
    }
    std::sort(v.begin(), v.end(), [](const DlssgRung& x, const DlssgRung& y) {
        if (x.m != y.m)
            return x.m < y.m;
        const int px = (x.b > 1) + (x.c > 1), py = (y.b > 1) + (y.c > 1);
        if (px != py)
            return px < py;
        return x.a != y.a ? x.a < y.a : x.b < y.b;
    });
    return v;
}

// Live DLSS 4.5's multiple for w = the target / the source rate: the first rung (dlssgRungs, two passes, with m3 three)
// that reaches w within 3 % (a 23.976 fps source at 48 stays 2x), the largest when none does, x2 below 2. m1, m2 = the
// passes in order (m2 = 1 for one pass), *m3 = the third (1 for one or two passes); *tie = false (the snap table's
// format)
static int dlssgSnap(double w, int lim, int& m1, int& m2, bool* tie = nullptr, int* m3 = nullptr)
{
    const auto rungs = dlssgRungs(lim, m3 ? 3 : 2);
    const DlssgRung* pick = &rungs.back();
    for (const auto& r : rungs)
        if (r.m * 1.03 >= w)
        {
            pick = &r;
            break;
        }
    m1 = pick->a;
    m2 = pick->b;
    if (m3)
        *m3 = pick->c;
    if (tie)
        *tie = false;
    return pick->m;
}

// the largest rung (with m3 the three-pass ones too) at most cap, its passes in the same order (the same multiple: the
// fewer passes, the smaller first pass); x2 below 2 (a plan that fell short steps down to what the GPU held)
static int dlssgAtMost(int cap, int lim, int& m1, int& m2, int* m3 = nullptr)
{
    const auto rungs = dlssgRungs(lim, m3 ? 3 : 2);
    const DlssgRung* pick = &rungs.front();
    for (const auto& r : rungs)
        if (r.m <= cap && r.m > pick->m)
            pick = &r;
    m1 = pick->a;
    m2 = pick->b;
    if (m3)
        *m3 = pick->c;
    return pick->m;
}

// The one pass's generated frames for w (the first whole multiple reaching it within 3 %, as dlssgSnap), 1..capMax
static int dlssgOnePassGen(double w, int capMax)
{
    int g = 1;
    while (g < capMax && (g + 1) * 1.03 < w)
        g++;
    return g;
}

// A non-whole Live multiple the file route's way (dlssgFpsPlan): a rung R of one or two helper passes (dlssgRungs) and
// a keep k of 2..6 with R / k within 3 % of w; the helper(s) generate R and the session shows every k-th frame of their
// ring, evenly spaced. The fewest passes first, then the smallest R (4.5x = 3x then 3x every 2nd, 7.5x = 3x then 5x
// every 2nd); 0 = none fits.
static int dlssgKeepPlan(double w, int lim, int& m1, int& m2, int& keep)
{
    const auto rungs = dlssgRungs(lim, 2);
    for (int passes = 1; passes <= 2; passes++)
        for (const auto& r : rungs)
        {
            if ((r.b > 1 ? 2 : 1) != passes)
                continue;
            for (int k = 2; k <= 6; k++)
                if (fabs((double)r.m / k - w) <= 0.03 * w)
                {
                    keep = k;
                    m1 = r.a;
                    m2 = r.b;
                    return r.m;
                }
        }
    return 0;
}

// "2nd" .. "6th": the keep's ordinal in the log lines
static const char* dlssgNth(int k)
{
    return k == 2 ? "2nd" : k == 3 ? "3rd" : k == 4 ? "4th" : k == 5 ? "5th" : "6th";
}

// The Live DLSS 4.5 cascade's first stage (SMV_LIVE_CASCADE=probe on the direct DLSS-G route: consume-only):
// dlssg2f.exe --stage1 in a job object that ends it with this process, fed every captured
// frame by a writer thread (a frame arriving while the last one is still being written is dropped, counted), and every
// present it shares read back on THIS process's D3D12 device: its queue waits the slot's fence value and copies the
// slot into a scratch texture. Nothing it reads is presented: the overlay keeps today's direct route.
struct Stage1HdrView // dlssg2f's Stage1Hdr (engine\dlssg\build_src\main.cpp), version 1 to 3
{
    uint32_t magic, version;
    volatile uint64_t frames;
    uint32_t w, h, ring, fmt;
    uint64_t allocSize;
    volatile uint32_t status, lastInput;
    volatile uint64_t slotFence[16];
    volatile uint64_t slotSeq[16];
    volatile int64_t slotQpc[16];
    volatile uint32_t slotInput[16];
    volatile uint32_t wantGen, gen; // version 2: the multiple asked of a running stage, and the one it runs
    // after them, on a stage that makes only the frames a keep shows (keep >= 1; 0 on any other stage: a mapping's
    // page past an older, shorter header reads zero): the keep asked, and the one it runs
    volatile uint32_t wantKeep, keep;
    // after them, on a stage that takes the ask (0 on any other): 1 = it shares a frame as soon as it is made (the
    // session's display times the frames), and what it does now
    volatile uint32_t wantAtOnce, atOnce;
    // after them: the slots' motion block (ngx_stage.h's Stage1Hdr from motionMagic on), which only the stage reading
    // this ring uses; the session reads none of it
};
static const uint32_t kStageOff = 0xFFFFFFFFu; // version 3's ask for 1x (dlssg2f's kGenOff): DLSS-G off, gen 0

struct CascadeProbe
{
    HANDLE job = nullptr, hProc = nullptr, hIn = nullptr, hOut = nullptr, hErr = nullptr, hMap = nullptr;
    const Stage1HdrView* hdr = nullptr;
    Stage1HdrView* hdrW = nullptr; // the same view, written only for an ask (askGen)
    bool inPlace = false;          // a version 2 stage: its multiple changes in place (askGen)
    bool passOk = false;           // a version 3 stage: it also takes 1x, its input passed through as it is
    uint32_t asked = 0;            // the numFramesToGenerate asked last (0 = none, kStageOff = 1x)
    std::thread errPump, writer, reader;
    std::mutex m;
    std::condition_variable cv;
    std::vector<uint8_t> mail;
    bool mailFull = false, quit = false;
    std::atomic<bool> readerQuit{false};
    // abortStart: the session ended while start() waits for READY; writerFailed: stage 1's stdin broke
    std::atomic<bool> abortStart{false}, writerFailed{false}, writerDone{false}, pumpDone{false};
    std::atomic<uint64_t> fed{0}, dropped{0}, consumed{0}, overrun{0}, skipped{0};
    uint64_t wdFed = 0, wdFrames = 0; // stalled()'s last window
    uint64_t repFed = 0, repDropped = 0, repConsumed = 0, repFrames = 0;
    ComPtr<ID3D12Resource> ring[16];
    ComPtr<ID3D12Fence> shared, own;
    ComPtr<ID3D12CommandQueue> q;
    ComPtr<ID3D12CommandAllocator> al;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Resource> scratch;
    HANDLE ev = nullptr;
    uint32_t n = 0;
    uint32_t fw = 0, fh = 0; // the frame size stage 1 takes
    // present mode (SMV_LIVE_CASCADE=present): no reader thread; the session's loop takes the slots in order
    // (nextFrame) and presents them through its own DLSS-G (stage 2)
    bool present = false;
    uint64_t nextSeq = 0;
    uint32_t keep = 1; // a non-whole multiple: every keep-th shared frame is presented, the rest left out
    std::atomic<uint64_t> unshown{0}; // the shared frames the keep left out
    // a stage that takes the keep itself (its header's keep; SMV_LIVE_DLSSG_KEEPDIRECT=0 = never): it makes and shares
    // only every stageKeep-th of its frames, and every frame it shares is presented (keep stays 1). keepWant = the
    // plan's keep, asked before the stage's first input
    bool keepOk = false;
    uint32_t stageKeep = 1, keepWant = 1;
    // the share tag (live<pid>; a third stage's middle probe live<pid>b) and, for that probe, the probe it takes its
    // frames from (dlssg2f --from) instead of the session's feed
    std::wstring tag;
    const CascadeProbe* up = nullptr;
    // the stage on a thread of this process (SMV_LIVE_DLSSG_LOCAL=0 = a dlssg2f --stage1 process): ngx_stage.h's
    // ngxStageMain, the code dlssg2f runs, sharing its ring and header by the same names, its inputs taken from mail
    bool local = lkEnv("SMV_LIVE_DLSSG_LOCAL") != "0";
    std::thread localTh;
    std::atomic<int> localUp{0};        // 1 = its ring is ready (localLine = its READY line), -1 = it ended first
    std::atomic<bool> localEnd{false};  // its inputs end (the session's end, a fallback)
    std::atomic<bool> localDone{false}; // its thread returned
    int localRc = 0;                    // the stage's result once done (2 = NGX's route did not come up)
    std::string localLine;
    std::vector<uint8_t> held; // the frame the stage works from
    double mailAt = 0;         // when mail's frame came (qpcMs)
    // SMV_LIVE_DLSSG_LOCAL_END=S: the stage's inputs end S seconds after its ring is ready (qpcMs; 0 = never), the
    // trigger test of a stage that ends while its frames are presented
    double localEndAt = 0;
    struct LocalFeed : StageFeed
    {
        CascadeProbe* p = nullptr;
        int take(const uint8_t*& f, double& at) override
        {
            return p->localTake(f, at);
        }
        void done() override
        {
        }
    } lfeed;

    // the next shared present in order: its slot and fence value (false = none yet); a slot already reused = skipped
    bool nextFrame(uint32_t& s, uint64_t& fv)
    {
        while (hdr->frames > nextSeq)
        {
            s = (uint32_t)(nextSeq % n);
            const uint64_t seq = hdr->slotSeq[s];
            if (seq != nextSeq)
            {
                skipped++;
                nextSeq = seq > nextSeq ? seq : nextSeq + 1;
                continue;
            }
            fv = hdr->slotFence[s];
            nextSeq++;
            consumed++;
            if (keep > 1 && seq % keep)
            {
                unshown++;
                continue;
            }
            return true;
        }
        return false;
    }

    // the plan's keep k: asked of a stage that takes it (applied at its next input, with a multiple asked in the same
    // window), else every k-th frame the stage shares is presented here
    void setKeep(uint32_t k)
    {
        if (keepOk && hdrW)
        {
            hdrW->wantKeep = k;
            stageKeep = k;
            keep = 1;
        }
        else
            keep = k;
    }

    // display-timed presents (Host::dtSet): the stage is asked to share each frame as soon as it is made, the display
    // gives each its time; off = the stage shares each at its time again
    bool atOnce = false;
    void setAtOnce(bool on)
    {
        if (hdrW && on != atOnce)
        {
            hdrW->wantAtOnce = on ? 1u : 0u;
            atOnce = on;
        }
    }

    // ask a running version 2 stage for another multiple mult (2..6; 1 on version 3): it applies it at its next
    // present; false = it cannot
    bool askGen(int mult)
    {
        if (!inPlace || !hdrW || mult < 1 || (mult == 1 && !passOk))
            return false;
        asked = mult == 1 ? kStageOff : (uint32_t)(mult - 1);
        hdrW->wantGen = asked;
        return true;
    }
    // the stage runs what was asked last (or nothing was asked); a refused ask clears wantGen
    uint32_t askedGen() const
    {
        return asked == kStageOff ? 0 : asked;
    }
    bool genHeld() const
    {
        return !asked || hdr->gen == askedGen();
    }
    bool askRefused() const
    {
        return asked && hdr->wantGen == 0 && hdr->gen != askedGen();
    }

    // the probe's place in the chain before any start: its upstream and its share tag (a probe on another probe's ring:
    // b, one on a b probe's ring, the final enlarge's third helper: c), so stages started at once name each other
    void prepare(const CascadeProbe* upstream)
    {
        up = upstream;
        tag = L"live" + std::to_wstring(GetCurrentProcessId()) + (up ? (up->up ? L"c" : L"b") : L"");
    }

    // stage stageNo as dlssg2f --stage1, a child in a job object that ends it with this process; true = its READY line
    // came (in `line`)
    bool startChild(uint32_t w, uint32_t h, int gen, bool hdr10, bool bgra, int stageNo, std::string& line)
    {
        wchar_t exePath[MAX_PATH]{};
        GetModuleFileNameW(nullptr, exePath, MAX_PATH);
        std::wstring dir(exePath);
        dir.resize(dir.find_last_of(L'\\'));
        dir += L"\\..\\dlssg";
        wchar_t ov[MAX_PATH]{};
        if (GetEnvironmentVariableW(L"SMV_DLSSG_DIR", ov, MAX_PATH) && ov[0])
            dir = ov;
        wchar_t full[MAX_PATH]{};
        if (GetFullPathNameW(dir.c_str(), MAX_PATH, full, nullptr))
            dir = full;
        job = CreateJobObjectW(nullptr, nullptr);
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION li{};
        li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if (!job || !SetInformationJobObject(job, JobObjectExtendedLimitInformation, &li, sizeof(li)))
        {
            LOG("cascade probe: no job object (%lu)\n", GetLastError());
            return false;
        }
        SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
        HANDLE inR = nullptr, outW = nullptr, errW = nullptr;
        if (!CreatePipe(&inR, &hIn, &sa, 1 << 20) || !CreatePipe(&hOut, &outW, &sa, 4096) ||
            !CreatePipe(&hErr, &errW, &sa, 0))
            return false;
        SetHandleInformation(hIn, HANDLE_FLAG_INHERIT, 0);
        SetHandleInformation(hOut, HANDLE_FLAG_INHERIT, 0);
        SetHandleInformation(hErr, HANDLE_FLAG_INHERIT, 0);
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
                  UpdateProcThreadAttribute(attrs, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherit, sizeof(inherit),
                                            nullptr, nullptr);
        si.lpAttributeList = attrs;
        const std::wstring exe = dir + L"\\dlssg2f.exe";
        std::wstring cmd = L"\"" + exe + L"\" --stage1 " + std::to_wstring(w) + L" " + std::to_wstring(h) + L" --gen " +
                           std::to_wstring(gen) +
                           (hdr10  ? L" --hdr10"
                            : bgra ? L" --bgra"
                                   : L"") +
                           L" --share " + tag + (up ? L" --from " + up->tag : L"");
        PROCESS_INFORMATION pi{};
        // suspended until the job holds it: a child that dies with this process can never outlive it
        ok = ok && CreateProcessW(exe.c_str(), cmd.data(), nullptr, nullptr, TRUE,
                                  EXTENDED_STARTUPINFO_PRESENT | CREATE_NO_WINDOW | CREATE_SUSPENDED, nullptr,
                                  dir.c_str(), &si.StartupInfo, &pi);
        DeleteProcThreadAttributeList(attrs);
        CloseHandle(inR);
        CloseHandle(outW);
        CloseHandle(errW);
        if (!ok)
        {
            LOG("cascade probe: cannot start %s (%lu)\n", wideToUtf8(exe).c_str(), GetLastError());
            return false;
        }
        hProc = pi.hProcess;
        if (!AssignProcessToJobObject(job, hProc))
        {
            LOG("cascade probe: the child is not in the job (%lu)\n", GetLastError());
            TerminateProcess(hProc, 1);
            CloseHandle(pi.hThread);
            return false;
        }
        ResumeThread(pi.hThread);
        CloseHandle(pi.hThread);
        errPump = std::thread([this, e = hErr, stageNo]() {
            std::string acc;
            char b[1024];
            DWORD rd = 0;
            while (ReadFile(e, b, sizeof(b), &rd, nullptr) && rd)
            {
                acc.append(b, rd);
                size_t nl;
                while ((nl = acc.find('\n')) != std::string::npos)
                {
                    std::string ln = acc.substr(0, nl);
                    if (!ln.empty() && ln.back() == '\r')
                        ln.pop_back();
                    LOG("cascade stage %d: %s\n", stageNo, ln.c_str());
                    acc.erase(0, nl + 1);
                }
            }
            pumpDone = true;
        });
        // READY within 20 s (NGX's start + the DLSS-G feature), polled: a child that hangs or dies, or a session
        // that ends meanwhile (abortStart), never holds the caller
        bool eol = false;
        const ULONGLONG until = GetTickCount64() + 20000;
        while (!eol && !abortStart.load() && GetTickCount64() < until)
        {
            DWORD avail = 0;
            if (!PeekNamedPipe(hOut, nullptr, 0, nullptr, &avail, nullptr))
                break; // stdout closed: the child ended
            if (!avail)
            {
                Sleep(5);
                continue;
            }
            char ch = 0;
            DWORD rd = 0;
            while (avail-- && ReadFile(hOut, &ch, 1, &rd, nullptr) && rd == 1)
            {
                if (ch == '\n')
                {
                    eol = true;
                    break;
                }
                if (ch != '\r' && line.size() < 256)
                    line += ch;
            }
        }
        if (line.rfind("DLSSG STAGE1 READY", 0) != 0)
        {
            DWORD rc = 1;
            const bool alive = WaitForSingleObject(hProc, 0) != WAIT_OBJECT_0;
            if (alive)
                TerminateJobObject(job, 1);
            WaitForSingleObject(hProc, 5000);
            GetExitCodeProcess(hProc, &rc);
            LOG("cascade probe: stage %d did not start (rc %lu%s)\n", stageNo, rc,
                !alive              ? ""
                : abortStart.load() ? ", the session ended"
                                    : ", no READY in 20 s");
            return false;
        }
        return true;
    }

    // the same stage on a thread of this process; true = its ring is ready (its READY line in `line`, ending in
    // local=1)
    bool startLocal(uint32_t w, uint32_t h, int gen, bool hdr10, bool bgra, int stageNo, std::string& line)
    {
        lfeed.p = this;
        lfeed.ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        const std::wstring from = up ? up->tag : std::wstring();
        localTh = std::thread([this, w, h, gen, hdr10, bgra, stageNo, from] {
            static const char* const tags[] = {"cascade stage 1: ", "cascade stage 2: ", "cascade stage 3: "};
            g_logTag = tags[(std::min)(stageNo, 3) - 1];
            int rc = 1;
            const bool ok = ngxStageMain(
                w, h, gen, hdr10, bgra, tag, from, &lfeed, &localEnd, false,
                [this](const std::string& ln) {
                    localLine = ln + " local=1";
                    localUp = 1;
                },
                rc);
            localRc = ok ? rc : 2;
            if (localUp.load() != 1)
                localUp = -1;
            localDone = true;
            g_logTag = nullptr;
        });
        // the ring ready within 20 s (NGX's start + the DLSS-G feature), or the session's end
        const ULONGLONG until = GetTickCount64() + 20000;
        while (localUp.load() == 0 && !abortStart.load() && GetTickCount64() < until)
            Sleep(5);
        if (localUp.load() == 1)
        {
            line = localLine;
            const std::string le = lkEnv("SMV_LIVE_DLSSG_LOCAL_END");
            if (!le.empty() && atof(le.c_str()) > 0)
            {
                std::lock_guard<std::mutex> lk(m);
                localEndAt = qpcMs() + atof(le.c_str()) * 1000.0;
            }
            return true;
        }
        const bool ended = localUp.load() < 0;
        endLocal();
        LOG("cascade probe: stage %d did not start (rc %d%s)\n", stageNo, localRc,
            ended               ? ""
            : abortStart.load() ? ", the session ended"
                                : ", no READY in 20 s");
        return false;
    }

    // the in-process stage's end: its inputs end and its thread is joined (an evaluate in flight and the frames made
    // are shared first)
    void endLocal()
    {
        localEnd = true;
        if (lfeed.ev)
            SetEvent(lfeed.ev);
        if (localTh.joinable())
            localTh.join();
    }

    // the in-process stage's input (its StageFeed): the newest fed frame; 0 = none yet, -1 = the inputs ended
    int localTake(const uint8_t*& f, double& at)
    {
        std::lock_guard<std::mutex> lk(m);
        if (quit || localEnd.load() || (localEndAt > 0 && qpcMs() >= localEndAt))
            return -1;
        if (!mailFull)
            return 0;
        held.swap(mail);
        mailFull = false;
        f = held.data();
        at = mailAt;
        fed++;
        return 1;
    }

    // bgra: SDR frames in B8G8R8A8 order (the effects route's slots and back buffer), else R8G8B8A8; upstream = a
    // running probe whose ring this one takes as its input (its stage number 2, never fed by the session)
    bool start(ID3D12Device* dev, uint32_t w, uint32_t h, int gen, bool hdr10, bool presentMode, bool bgra,
               const CascadeProbe* upstream = nullptr)
    {
        present = presentMode;
        if (tag.empty())
            prepare(upstream);
        fw = w;
        fh = h;
        const int stageNo = up ? (up->up ? 3 : 2) : 1;
        std::string line;
        if (local ? !startLocal(w, h, gen, hdr10, bgra, stageNo, line)
                  : !startChild(w, h, gen, hdr10, bgra, stageNo, line))
            return false;
        const std::wstring mn = L"smv_dg1_" + tag + L"_hdr";
        hMap = OpenFileMappingW(FILE_MAP_READ | FILE_MAP_WRITE, FALSE, mn.c_str());
        // the whole mapping (0): a version 1 stage's header is shorter than this view
        hdrW = hMap ? (Stage1HdrView*)MapViewOfFile(hMap, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, 0) : nullptr;
        hdr = hdrW;
        if (!hdr || hdr->magic != 0x31564D53 || hdr->version < 1 || hdr->version > 3 || hdr->ring > 16)
        {
            LOG("cascade probe: no stage-1 header\n");
            return false;
        }
        inPlace = hdr->version >= 2;
        passOk = hdr->version >= 3;
        keepOk = passOk && hdr->keep >= 1 && lkEnv("SMV_LIVE_DLSSG_KEEPDIRECT") != "0";
        setKeep(keepWant);
        // stage 2 copies a ring frame straight into its back buffer: the same size and format or no cascade (a
        // dlssg2f that does not know --bgra shares R8G8B8A8)
        const uint32_t wantFmt = (uint32_t)(hdr10  ? DXGI_FORMAT_R10G10B10A2_UNORM
                                            : bgra ? DXGI_FORMAT_B8G8R8A8_UNORM
                                                   : DXGI_FORMAT_R8G8B8A8_UNORM);
        if (hdr->w != w || hdr->h != h || hdr->fmt != wantFmt)
        {
            LOG("cascade probe: stage 1 shares %ux%u format %u, %ux%u format %u wanted\n", hdr->w, hdr->h, hdr->fmt, w,
                h, wantFmt);
            return false;
        }
        n = hdr->ring;
        for (uint32_t i = 0; i < n; i++)
        {
            HANDLE sh = nullptr;
            const std::wstring nm = L"smv_dg1_" + tag + L"_ring" + std::to_wstring(i);
            if (FAILED(dev->OpenSharedHandleByName(nm.c_str(), GENERIC_ALL, &sh)) ||
                FAILED(dev->OpenSharedHandle(sh, IID_PPV_ARGS(&ring[i]))))
            {
                LOG("cascade probe: ring texture %u not opened\n", i);
                return false;
            }
            CloseHandle(sh);
        }
        {
            HANDLE sh = nullptr;
            const std::wstring fn = L"smv_dg1_" + tag + L"_fence";
            if (FAILED(dev->OpenSharedHandleByName(fn.c_str(), GENERIC_ALL, &sh)) ||
                FAILED(dev->OpenSharedHandle(sh, IID_PPV_ARGS(&shared))))
            {
                LOG("cascade probe: stage-1 fence not opened\n");
                return false;
            }
            CloseHandle(sh);
        }
        D3D12_COMMAND_QUEUE_DESC qd{};
        qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC rd2 = ring[0]->GetDesc();
        if (FAILED(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&q))) ||
            FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&al))) ||
            FAILED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, al.Get(), nullptr, IID_PPV_ARGS(&list))) ||
            FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&own))) ||
            FAILED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd2, D3D12_RESOURCE_STATE_COPY_DEST,
                                                nullptr, IID_PPV_ARGS(&scratch))))
        {
            LOG("cascade probe: reader resources failed\n");
            return false;
        }
        list->Close();
        ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        mail.resize((size_t)w * h * 4);
        held.resize(mail.size());
        if (!local)
            writer = std::thread([this] { writeLoop(); });
        nextSeq = hdr->frames;
        if (!present)
            reader = std::thread([this] { readLoop(); });
        LOG("cascade probe: stage %d up (%s), %ux%u x%d, its ring of %u opened on this device%s\n", stageNo,
            line.c_str(), w, h, gen + 1, n,
            up        ? (up->up ? ", from stage 2's ring" : ", from stage 1's ring")
            : present ? ", stage 2 presents it"
                      : "");
        return true;
    }

    // the newest captured frame for the child; one in flight at most
    void feed(const uint8_t* p, size_t bytes)
    {
        std::lock_guard<std::mutex> lk(m);
        if (mailFull || bytes != mail.size())
        {
            dropped++;
            return;
        }
        memcpy(mail.data(), p, bytes);
        mailFull = true;
        mailAt = qpcMs();
        cv.notify_one();
        if (lfeed.ev)
            SetEvent(lfeed.ev);
    }

    // the same for a frame whose rows sit `pitch` bytes apart (the effects route's output slots)
    void feedRows(const uint8_t* p, size_t pitch)
    {
        std::lock_guard<std::mutex> lk(m);
        const size_t row = (size_t)fw * 4;
        if (mailFull || row * fh != mail.size())
        {
            dropped++;
            return;
        }
        if (pitch == row)
            memcpy(mail.data(), p, mail.size());
        else
            for (uint32_t y = 0; y < fh; y++)
                memcpy(mail.data() + y * row, p + y * pitch, row);
        mailFull = true;
        mailAt = qpcMs();
        cv.notify_one();
        if (lfeed.ev)
            SetEvent(lfeed.ev);
    }

    void writeLoop()
    {
        std::vector<uint8_t> out(mail.size());
        for (;;)
        {
            {
                std::unique_lock<std::mutex> lk(m);
                cv.wait(lk, [this] { return mailFull || quit; });
                if (quit)
                {
                    writerDone = true;
                    return;
                }
                out.swap(mail);
                mailFull = false;
            }
            size_t off = 0;
            while (off < out.size())
            {
                DWORD wr = 0;
                if (!WriteFile(hIn, out.data() + off, (DWORD)(std::min)(out.size() - off, (size_t)1 << 20), &wr,
                               nullptr) ||
                    !wr)
                {
                    writerFailed = true;
                    writerDone = true;
                    return;
                }
                off += wr;
            }
            fed++;
        }
    }

    // stage 1 is gone: its process or thread ended, its header says ended / failed, or its stdin broke
    bool dead() const
    {
        return (hProc && WaitForSingleObject(hProc, 0) == WAIT_OBJECT_0) || (hdr && hdr->status >= 2) ||
               writerFailed.load() || localDone.load();
    }

    // one call a 2 s window: stage 1 took inputs but shared nothing (a hung child; a static source still feeds it the
    // capture's 500 ms keepalive, each of which it presents); a middle stage's inputs = its upstream's shared frames
    bool stalled()
    {
        const uint64_t f = up ? (up->hdr ? up->hdr->frames : 0) : fed.load(), fr = hdr ? hdr->frames : 0;
        const bool s = f - wdFed >= 2 && fr == wdFrames;
        wdFed = f;
        wdFrames = fr;
        return s;
    }

    // stage 2 starts at the newest shared present (a resume after a pause shows no stale tail)
    void skipToNewest()
    {
        if (hdr)
            nextSeq = hdr->frames;
    }

    // a fallback's first step: end the child at once (its job) and release any stage-2 queue wait on its fence. An
    // in-process stage only has its inputs ended: its GPU work completes and signals the fence itself, and a value
    // signalled here would end its last wait early, so its textures and feature would go while that work still runs
    // (stop() joins it, then signals)
    void kill()
    {
        if (local)
        {
            localEnd = true;
            if (lfeed.ev)
                SetEvent(lfeed.ev);
            return;
        }
        if (job)
            TerminateJobObject(job, 1);
        if (shared)
            shared->Signal(UINT64_MAX);
    }

    void readLoop()
    {
        uint64_t next = hdr->frames, ov = 0;
        while (!readerQuit.load())
        {
            if (hdr->frames <= next)
            {
                SwitchToThread();
                continue;
            }
            const uint32_t s = (uint32_t)(next % n);
            const uint64_t seq = hdr->slotSeq[s], fv = hdr->slotFence[s];
            if (seq != next)
            {
                if (seq > next)
                    skipped++;
                next = seq > next ? seq : next + 1;
                continue;
            }
            q->Wait(shared.Get(), fv);
            al->Reset();
            list->Reset(al.Get(), nullptr);
            D3D12_RESOURCE_BARRIER b{};
            b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            b.Transition.pResource = ring[s].Get();
            b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            b.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
            b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
            list->ResourceBarrier(1, &b);
            list->CopyResource(scratch.Get(), ring[s].Get());
            b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
            b.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
            list->ResourceBarrier(1, &b);
            list->Close();
            ID3D12CommandList* ls[] = {list.Get()};
            q->ExecuteCommandLists(1, ls);
            q->Signal(own.Get(), ++ov);
            if (own->GetCompletedValue() < ov)
            {
                own->SetEventOnCompletion(ov, ev);
                WaitForSingleObject(ev, 2000);
            }
            if (hdr->slotSeq[s] != next)
                overrun++;
            else
                consumed++;
            next++;
        }
    }

    // one line a stats window (the direct loop's 2 s block)
    void report(double secs)
    {
        const uint64_t f = fed, d = dropped, c = consumed, fr = hdr ? hdr->frames : 0;
        char ks[64] = "";
        if (keepOk && stageKeep > 1)
            snprintf(ks, sizeof ks, "; every %s shown, the others not made", dlssgNth((int)stageKeep));
        else if (keep > 1)
            snprintf(ks, sizeof ks, "; every %s shown, %llu left out", dlssgNth((int)keep),
                     (unsigned long long)unshown.load());
        if (up)
            LOG("cascade probe 2: stage 2 shared %.1f/s, read %.1f/s; skipped %llu%s\n", (fr - repFrames) / secs,
                (c - repConsumed) / secs, (unsigned long long)skipped.load(), ks);
        else
            LOG("cascade probe: fed %.1f/s (dropped %llu), stage 1 shared %.1f/s, read %.1f/s; overrun %llu, skipped "
                "%llu%s\n",
                (f - repFed) / secs, (unsigned long long)(d - repDropped), (fr - repFrames) / secs,
                (c - repConsumed) / secs, (unsigned long long)overrun.load(), (unsigned long long)skipped.load(), ks);
        repFed = f;
        repDropped = d;
        repConsumed = c;
        repFrames = fr;
    }

    void stop()
    {
        abortStart = true;
        {
            std::lock_guard<std::mutex> lk(m);
            quit = true;
        }
        cv.notify_all();
        if (writer.joinable())
        {
            // a writer blocked in WriteFile (a child that stopped reading) is cancelled; past 500 ms the child is ended,
            // which breaks its pipe
            const ULONGLONG until = GetTickCount64() + 500;
            while (!writerDone.load())
            {
                CancelSynchronousIo(writer.native_handle());
                if (GetTickCount64() > until && job)
                    TerminateJobObject(job, 1);
                Sleep(1);
            }
            writer.join();
        }
        if (hIn)
        {
            CloseHandle(hIn); // EOF: stage 1 ends on its own
            hIn = nullptr;
        }
        const int stageNo = up ? 2 : 1;
        if (hProc && WaitForSingleObject(hProc, 3000) != WAIT_OBJECT_0)
        {
            LOG("cascade probe: stage %d still running 3 s after its input closed: ended\n", stageNo);
            TerminateProcess(hProc, 1);
            if (WaitForSingleObject(hProc, 2000) != WAIT_OBJECT_0)
                LOG("cascade probe: stage %d not gone 2 s after its end\n", stageNo);
        }
        endLocal();
        // a session queue wait on a value the child never signals now (it ended between a share and its copy)
        if (shared)
            shared->Signal(UINT64_MAX);
        readerQuit = true;
        if (reader.joinable())
            reader.join();
        if (errPump.joinable())
        {
            // the pump ends at the child's stderr EOF; a process the child started that inherited the pipe holds it open
            // past the child, so a read still blocked 2 s on is cancelled
            const ULONGLONG until = GetTickCount64() + 2000;
            bool told = false;
            while (!pumpDone.load())
            {
                if (GetTickCount64() > until)
                {
                    if (!told)
                        LOG("cascade probe: stage %d's stderr still open 2 s on (a process it started holds it): "
                            "read cancelled\n",
                            stageNo);
                    told = true;
                    CancelSynchronousIo(errPump.native_handle());
                }
                Sleep(1);
            }
            errPump.join();
        }
        LOG("cascade probe%s: ended (fed %llu, dropped %llu, read %llu, overrun %llu, skipped %llu)\n", up ? " 2" : "",
            (unsigned long long)fed.load(), (unsigned long long)dropped.load(), (unsigned long long)consumed.load(),
            (unsigned long long)overrun.load(), (unsigned long long)skipped.load());
        if (hdr)
            UnmapViewOfFile((LPCVOID)hdr);
        hdr = nullptr;
        hdrW = nullptr;
        for (HANDLE x : {hProc, hOut, hErr, hMap, ev, job, lfeed.ev})
            if (x)
                CloseHandle(x);
        hProc = hOut = hErr = hMap = ev = job = lfeed.ev = nullptr;
    }
};

// The DXGI description of the adapter output that shows hmon
static bool monitorDesc1(HMONITOR hmon, DXGI_OUTPUT_DESC1& od1)
{
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
            return SUCCEEDED(o.As(&o6)) && SUCCEEDED(o6->GetDesc1(&od1));
        }
    }
    return false;
}

// Is Windows HDR ("Use HDR") active on the display the target window sits on? The
// output reports G2084 (PQ) as its current color space when HDR is on. Used to warn that today's
// 8-bit SDR capture clips HDR content, and later to switch on the HDR live pipeline.
static bool monitorIsHDR(HWND target)
{
    DXGI_OUTPUT_DESC1 od1{};
    return monitorDesc1(MonitorFromWindow(target, MONITOR_DEFAULTTONEAREST), od1) &&
           od1.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020;
}

// The peak (nits) of a monitor in HDR mode as Windows reports it (DXGI MaxLuminance), the level NVIDIA's own RTX
// Video HDR expands to; 0 when the monitor is not in HDR mode
static int monitorPeakNits(HMONITOR hmon)
{
    DXGI_OUTPUT_DESC1 od1{};
    if (!monitorDesc1(hmon, od1) || od1.ColorSpace != DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020)
        return 0;
    return (int)(od1.MaxLuminance + 0.5f);
}

// TrueHDR's MaxLuminance: --hdr-nits, else the monitor's peak, else 1000 (an SDR monitor: the usual HDR10 master),
// inside the SDK's 400..2000
static uint32_t thdrPeakNits(HMONITOR hmon)
{
    int n = g_hdrNits > 0 ? g_hdrNits : monitorPeakNits(hmon);
    if (n <= 0)
        n = 1000;
    return (uint32_t)(n < 400 ? 400 : (n > 2000 ? 2000 : n));
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

// Small always-on-top readout at the overlay's corner: base fps -> interpolated fps, their ratio, plus the
// estimated capture->screen delay, in white. A SEPARATE window on purpose:
// pixels stamped into the frames would get interpolated (DLSS-G smears moving text), and the
// HUD must NEVER take activation (WS_EX_NOACTIVATE) or DLSS-G's focus gate would drop. Black
// colorkey = floating text, click-through, excluded from capture so monitor mode never sees
// itself (this also keeps it out of the --diag dumps).
struct Hud
{
    HWND hwnd = nullptr;
    HFONT font = nullptr;
    wchar_t text[256]{};
    wchar_t sized[256]{};       // the text the window was last sized for
    int textH = 0;              // that text's height as drawn, one line or more
    bool readoutLogged = false; // the first readout of the session went to the log (its form)

    void create(int x, int y, const wchar_t* first)
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
        // SMV_HUD_CAPTURE=1 (a measurement aid): the HUD stays capturable, so a harness can read it off the screen
        if (GetEnvironmentVariableW(L"SMV_HUD_CAPTURE", nullptr, 0) == 0)
            SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE); // best effort
        font = CreateFontW(-24, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS,
                           CLIP_DEFAULT_PRECIS, ANTIALIASED_QUALITY, DEFAULT_PITCH, L"Segoe UI");
        _snwprintf_s(text, _TRUNCATE, L"%ls", first); // the opening note until the first stats window lands
        LOG("HUD opens with: %s\n", wideToUtf8(text).c_str());
        ShowWindow(hwnd, SW_SHOWNOACTIVATE);
        paint();
    }

    // sourceOnly = passthrough and effects only (the echo route, no frame generated): one figure, the source's frame
    // rate (its new pictures a second, those the pipeline superseded included: what an overlay inside the source app
    // shows for a game); the presents' count would add the held frame's re-presents
    void update(double inFps, double outFps, double latMs, bool sourceOnly = false)
    {
        if (!hwnd)
            return;
        // \u2192 = the arrow, \u00D7 = the times sign; ESCAPED on purpose: a raw UTF-8 literal in this file compiles
        // as ANSI mojibake unless the build adds /utf-8 (it showed as "weird letters")
        // --no-hud-latency hides the latency segment only; the fps segment is governed by
        // --no-hud (whole readout).
        // the in rate in whole numbers like the in-game overlays (a decimal only showed the capture stamps' jitter: 23.9 /
        // 24 / 24.1 on a 23.976 video), and the measured multiplier, out over in (the live: line's ratio): a shown 0
        // never carries one
        wchar_t inTxt[16];
        swprintf_s(inTxt, L"%.0f", inFps);
        wchar_t mul[32] = L"";
        if (wcscmp(inTxt, L"0"))
            swprintf_s(mul, L" (%.2f\u00D7)", outFps / inFps);
        if (sourceOnly && latMs > 0.5 && !g_noHudLat)
            swprintf_s(text, L"%ls fps   ~%.0f ms behind", inTxt, latMs);
        else if (sourceOnly)
            swprintf_s(text, L"%ls fps", inTxt);
        else if (latMs > 0.5 && !g_noHudLat)
            swprintf_s(text, L"%ls fps \u2192 %.0f fps%ls   ~%.0f ms behind", inTxt, outFps, mul, latMs);
        else
            swprintf_s(text, L"%ls fps \u2192 %.0f fps%ls", inTxt, outFps, mul);
        if (!readoutLogged)
        {
            LOG("HUD readout: %s (%s)\n", wideToUtf8(text).c_str(),
                sourceOnly ? "the source's frame rate" : "captured -> presented");
            readoutLogged = true;
        }
        // re-assert the top of the TOPMOST band: the overlay's HWND_TOPMOST moves raise it above us
        SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
        paint();
    }

    // The window follows its text: one line as wide as the text, up to the right edge of the monitor's work area, and
    // more lines past it, so a loading note that names several effects shows whole
    void fit()
    {
        if (!wcscmp(text, sized))
            return;
        wcscpy_s(sized, text);
        RECT w{};
        GetWindowRect(hwnd, &w);
        MONITORINFO mi{sizeof(mi)};
        GetMonitorInfoW(MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST), &mi);
        const int room = (std::max)(200, (int)(mi.rcWork.right - w.left) - 16);
        RECT r{0, 0, room - 8 - 2 * kEdge, 0};
        HDC dc = GetDC(hwnd);
        HGDIOBJ of = SelectObject(dc, font);
        DrawTextW(dc, text, -1, &r, drawFlags() | DT_CALCRECT);
        SelectObject(dc, of);
        ReleaseDC(hwnd, dc);
        textH = (int)(r.bottom - r.top);
        SetWindowPos(hwnd, nullptr, 0, 0, (int)(r.right - r.left) + 8 + 2 * kEdge,
                     (std::max)(36, textH + 4 + 2 * kEdge), SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
    }

    // the edge around the white text, in px: a near-black copy of the text drawn within it (pure black is the colour
    // key and would show through)
    static constexpr int kEdge = 2;

    // right-to-left text (Arabic, Hebrew) takes DrawText's right-to-left reading order, right-aligned
    UINT drawFlags() const
    {
        bool rtl = false;
        for (const wchar_t* s = text; *s && !rtl; ++s)
            rtl = (*s >= 0x0590 && *s <= 0x08FF) || (*s >= 0xFB1D && *s <= 0xFDFF) || (*s >= 0xFE70 && *s <= 0xFEFF);
        return DT_WORDBREAK | DT_NOPREFIX | (rtl ? DT_RIGHT | DT_RTLREADING : DT_LEFT);
    }

    void paint()
    {
        if (!hwnd)
            return;
        fit();
        HDC dc = GetDC(hwnd);
        RECT c;
        GetClientRect(hwnd, &c);
        // drawn off screen and copied in one BitBlt: the black fill (the colour key) and the edge's copies drawn on the
        // window itself let DWM compose a blank or edge-only HUD between them (the model-load wait repaints every pass)
        HDC mem = CreateCompatibleDC(dc);
        HBITMAP bmp = CreateCompatibleBitmap(dc, c.right, c.bottom);
        HGDIOBJ ob = SelectObject(mem, bmp);
        FillRect(mem, &c, (HBRUSH)GetStockObject(BLACK_BRUSH));
        HGDIOBJ of = SelectObject(mem, font);
        SetBkMode(mem, TRANSPARENT);
        RECT r = c;
        r.left += 4 + kEdge;
        r.right -= 4 + kEdge;
        r.top = (r.bottom - textH) / 2; // centred: one line in the window's 36 px, more lines in their own height
        const UINT fl = drawFlags();
        SetTextColor(mem, RGB(8, 8, 8)); // the edge: readable on a white picture
        for (int dy = -kEdge; dy <= kEdge; ++dy)
            for (int dx = -kEdge; dx <= kEdge; ++dx)
                if ((dx || dy) && dx * dx + dy * dy <= kEdge * kEdge + 1)
                {
                    RECT e = r;
                    OffsetRect(&e, dx, dy);
                    DrawTextW(mem, text, -1, &e, fl);
                }
        SetTextColor(mem, RGB(255, 255, 255)); // full white: easier to read than green
        DrawTextW(mem, text, -1, &r, fl);
        SelectObject(mem, of);
        BitBlt(dc, 0, 0, c.right, c.bottom, mem, 0, 0, SRCCOPY);
        SelectObject(mem, ob);
        DeleteObject(bmp);
        DeleteDC(mem);
        ReleaseDC(hwnd, dc);
    }

    void show(bool visible)
    {
        if (hwnd)
            ShowWindow(hwnd, visible && !g_noHud ? SW_SHOWNOACTIVATE : SW_HIDE);
    }
    void move(int x, int y)
    {
        if (!hwnd)
            return;
        SetWindowPos(hwnd, HWND_TOPMOST, x, y, 0, 0, SWP_NOSIZE | SWP_NOACTIVATE);
        sized[0] = 0; // the room to the monitor's edge changed: the next paint sizes the window again
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

// The windowed overlay follows its player on every pass of the live loops and of the model-load wait, apart from the
// pause test's 250 ms tick: DWM's frame query costs ~1 us, so a dragged player is trailed by one pass at most. The
// overlay (not Fill's, which covers the monitor) and the HUD move only when the frame's origin changed; the overlay
// keeps covering the client area (frame origin + the constant crop offset). True = it moved.
static bool followTarget(HWND target, HWND overlay, RECT& fb, int cropX, int cropY, Hud& hud, bool moveOverlay)
{
    RECT r{};
    if (!frameBounds(target, r) || (r.left == fb.left && r.top == fb.top))
        return false;
    fb = r;
    const int ox = r.left + cropX;
    const int oy = r.top + cropY;
    if (moveOverlay)
        SetWindowPos(overlay, HWND_TOPMOST, ox, oy, 0, 0, SWP_NOSIZE | SWP_NOACTIVATE);
    hud.move(ox + 16, oy + 16);
    return true;
}

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

// RESIZE DEBOUNCE: ending the session (exit 4, the app revives it) on the FIRST >2 px size
// change would turn a drag resize into a chain of engine builds at whatever intermediate size
// the window had 400 ms later, each blocking the host for up to a minute. So the stale overlay
// hides at once, the HUD shows the loading note at once, and the session ends only after the client
// size has held for 1 s. Returns early when the window
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
// The HUD's loading note: the GUI's --hud-note (its language) as is, else the English note around loadingWhat()
static std::wstring loadingNote()
{
    if (!g_hudNote.empty())
        return g_hudNote;
    return L"SMV Live: Loading " + loadingWhat() + L"...";
}
// The HUD's figures every 0.5 s (the `live:` line keeps its 2 s window), counted the way the in-game overlays count
// frames: the source's rate = its PRESENTS where the capture sees each one (a window: every present arrives, an unchanged
// picture too), else its new pictures (the whole screen carries every other update on it; a player that repeats its
// picture on every refresh shows what it plays, not the panel's rate); (n - 1) over the first-to-last span (one frame
// in the second = n over it). SMV's own figures (the presented rate = the swap chain's present count between two
// readings about a second apart; the latency = the mean of 0.5 s of samples, each present call - capture stamp + lag +
// one panel refresh, see stampLagMs) follow the new pictures: a still picture makes no new frame. Every figure covers
// the second up to the PREVIOUS reading (0.5 s behind), so a stream's last frames before it stops (a player's repaint
// when it pauses or ends) never reach one. A stream that stops (no frame for 2.5 of its mean intervals, at least 250
// ms: a paused player, a menu that stops rendering, a still shot on the whole screen) keeps its figures; moving again for
// 0.5 s at least, they count only what came since
struct HudMeter
{
    static constexpr int64_t kWin = 10000000, kEvery = 5000000, kStopMin = 2500000; // 1 s, 0.5 s, 250 ms in 100 ns
    // one frame stream's stop and resume
    struct Track
    {
        bool stopped = false;
        bool resumed = false; // this read is the first after a stop
        int64_t stopLast = 0; // the newest frame when it stopped
        int64_t resumeAt = 0; // the first frame after the stop, until the figures' second is past it (0 = none)
        int64_t seen = 0;     // the newest frame the previous step saw
        // false = the stream stopped (hold its figures)
        bool step(int64_t now, Capture& cap, bool presents)
        {
            resumed = false;
            int64_t first = 0, lastIn = 0, last = 0, prev = 0;
            const int n = cap.framesSince(presents, now - kWin, now, first, lastIn, last, prev);
            const int64_t iv = n >= 2 && lastIn > first ? (lastIn - first) / (n - 1) : (prev ? last - prev : 0);
            const int64_t stopGap = (std::max)(kStopMin, iv * 5 / 2);
            if (!last || now - last > stopGap)
            {
                if (!stopped)
                    stopLast = last;
                stopped = true;
                seen = last;
                return false;
            }
            // a stop between two readings (a pause, then a short burst before this one): a gap wider than the stop
            // threshold among the frames since the previous step
            int64_t before = 0, after = 0;
            if (!stopped && seen && cap.latestGap(presents, seen, stopGap, before, after))
            {
                stopLast = before;
                stopped = true;
            }
            seen = last;
            if (stopped)
            {
                stopped = false;
                resumed = true;
                cap.framesSince(presents, stopLast + 1, now, first, lastIn, last, prev);
                resumeAt = first ? first : last;
            }
            if (resumeAt && now - resumeAt >= kWin + kEvery)
            {
                // one stray picture is no resume (a player's on-screen note while paused, a frame step): stopped again
                const bool one = cap.framesSince(presents, resumeAt, now, first, lastIn, last, prev) < 2;
                resumeAt = 0;
                if (one)
                {
                    stopped = true;
                    stopLast = last;
                    return false;
                }
            }
            return true;
        }
        // its figures may move up to `to`: no resume, or one that lasted (three frames over 0.5 s at least: a player's
        // on-screen note or a frame step while paused is a short burst, playing again is not)
        bool settled(int64_t to, Capture& cap, bool presents) const
        {
            int64_t first = 0, lastIn = 0, last = 0, prev = 0;
            return !resumeAt || (cap.framesSince(presents, resumeAt, to, first, lastIn, last, prev) >= 3 &&
                                 lastIn - resumeAt >= kEvery);
        }
    };
    int64_t lastTick = 0;
    std::deque<std::pair<int64_t, UINT>> pcs; // (QPC 100 ns, the swap chain's present count), the newest four readings
    std::deque<std::pair<int64_t, double>> lat;
    Track src, pic;                     // the source's rate stream; the new pictures (SMV's own figures)
    bool started = false;               // three readings done (a presented rate up to the previous one exists)
    bool picSeen = false;               // the pictures moved at least once
    double inF = 0, outF = 0, latF = 0; // the figures shown
    void addLat(int64_t t, double ms)
    {
        lat.emplace_back(t, ms);
        while (!lat.empty() && t - lat.front().first > kWin)
            lat.pop_front();
    }
    bool due(int64_t now) const
    {
        return now - lastTick >= kEvery;
    }
    // false = no figure yet (the first second): the HUD keeps its loading note. panelHz = the overlay monitor's refresh
    // rate (0 = unknown)
    bool read(int64_t now, Capture& cap, UINT pc, double panelHz, double& inFps, double& outFps, double& latMs)
    {
        const int64_t upTo = lastTick; // the previous reading: the figures' second ends there
        lastTick = now;
        pcs.emplace_back(now, pc);
        while (pcs.size() > 4)
            pcs.pop_front();
        while (!lat.empty() && now - lat.front().first > kWin)
            lat.pop_front();
        started = started || pcs.size() >= 3;
        int64_t f = 0, li = 0, l = 0, p = 0;
        const int nP = g_monitor ? 0 : cap.framesSince(true, now - kWin, now, f, li, l, p);
        const int nC = cap.framesSince(false, now - kWin, now, f, li, l, p);
        const bool presents = !g_monitor && !(panelHz > 0 && nP >= 0.8 * panelHz && 2 * nC <= nP);
        if (src.step(now, cap, presents) && upTo && src.settled(upTo, cap, presents))
        {
            int64_t first = 0, lastIn = 0, last = 0, prev = 0;
            const int n =
                cap.framesSince(presents, (std::max)(upTo - kWin, src.resumeAt), upTo, first, lastIn, last, prev);
            inF = n >= 2 && lastIn > first ? (n - 1) * 1e7 / (double)(lastIn - first) : n * 1e7 / (double)kWin;
        }
        const bool picMoving = pic.step(now, cap, false);
        picSeen = picSeen || picMoving;
        if (picMoving && pic.resumed)
        {
            // SMV's output since the resume only: the stop's re-presents and old samples stay out
            pcs.clear();
            pcs.emplace_back(now, pc);
            lat.clear();
        }
        else if (((picMoving && pic.settled(upTo, cap, false)) || !picSeen) && pcs.size() >= 3)
        {
            // (a picture that never moved yet: the plain present count, nothing to hold)
            const auto& a = pcs.front();
            const auto& b = pcs[pcs.size() - 2];
            if (b.first > a.first)
                outF = (double)(b.second - a.second) * 1e7 / (double)(b.first - a.first);
            double s = 0;
            int k = 0;
            for (const auto& x : lat)
                if (x.first <= upTo && upTo - x.first <= kEvery)
                {
                    s += x.second;
                    k++;
                }
            if (k)
                latF = s / k;
        }
        inFps = inF;
        outFps = outF;
        latMs = latF;
        return started;
    }
};
// Waits until the target's client size holds still. true = it came back to the captured size (a
// transient: the restore animation after a minimize hands out a frame of another size), the
// session continues with the overlay shown again; false = a real resize, the caller restarts.
static bool resizeSettle(HWND target, Host& host, const Capture& cap, Hud& hud, bool hidden)
{
    const DWORD settleMs = 1000;    // how long the client size must hold still
    ShowWindow(host.hwnd, SW_HIDE); // the old-size frames no longer cover the window
    if (hud.hwnd)
    {
        _snwprintf_s(hud.text, _TRUNCATE, L"%ls", loadingNote().c_str());
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
    // the count as asked: DLSS 4.5 with effects zeroes genFrames (its echo route holds one slot), and a fixed
    // multiple's plan still needs it
    const int genAsked = genFrames;

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
    // a minimized window captures as its caption (183x34 here), the load-phase resize watch then ends the session with
    // exit 4 and the app revives it at that size: wait for the restore first, like the pause rule mid-session (Whole
    // screen captures the monitor whatever the seeding window's state)
    if (!g_monitor && IsIconic(target))
    {
        LOG("target window minimized: waiting for it to be restored\n");
        for (;;)
        {
            if (!IsWindow(target) || !IsWindowVisible(target))
            {
                LOG("target window closed (hwnd 0x%p %s)\n", (void*)target,
                    IsWindow(target) ? "is hidden" : "no longer exists");
                timeEndPeriod(1);
                return 7;
            }
            if (!IsIconic(target))
                break;
            if (g_stopReq.load())
            {
                LOG("stop requested while the target window was minimized\n");
                timeEndPeriod(1);
                g_sessionClean = true;
                return 0;
            }
            pumpMessages();
            Sleep(100);
        }
        LOG("target window restored\n");
    }
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
        // tensors and letterboxes into the R10A2 canvas (RTX VSR reads the SDR view there).
        // SMV_LIVE_HDR forces the mode for testing.
        // DLSS-G is HDR-capable through the exe-side pack (kHdrPackCS): it takes RGB10 + PQ
        // for HDR, the capture converts before the stages, which interpolate on the PQ frames.
        // "fruc" (Smooth Motion): the server feeds the NvOFFRUC bridge the PQ-encoded frames like rife, the bridge
        // quantises them to 8-bit BGRA for the flow and the warp, so only the TWEENS carry 8-bit
        // PQ precision (the real frames stay full precision); far better than the SDR capture
        // of an HDR-presented window, which is 2-3x over-bright and clipped (below). "fsrfg" (AMD FSR frame
        // generation) takes FRUC's HDR rule at fp16 (nativeFsrfgPair). "nvvfg" (NVIDIA Video Frame Generation) gets
        // every pair as its PQ codes in R10G10B10A2 words, so its tweens carry 10 bits, the swap chain's own depth.
        const bool hdrCapable =
            g_backend == BK_DLSSG ||
            (g_backend == BK_SERVER &&
             (g_serverBackend == L"echo" || g_serverBackend == L"rife" || g_serverBackend == L"gmfss" ||
              g_serverBackend == L"nvof" || g_serverBackend == L"rifedrba" || g_serverBackend == L"fruc" ||
              g_serverBackend == L"fsrfg" || g_serverBackend == L"nvvfg"));
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
        // SDR white and peak here, while the target is known.
        if (g_rtxHdr && g_hdr && g_backend == BK_SERVER)
        {
            g_sdrWhite = sdrWhiteNits(target);
            g_thdrPeak = thdrPeakNits(MonitorFromWindow(target, MONITOR_DEFAULTTONEAREST));
            LOG("RTX HDR live: TrueHDR expansion on (colour %ls, SDR white %.0f nits, peak %u nits)\n", g_hdrColor,
                g_sdrWhite, g_thdrPeak);
        }
        else if (g_rtxHdr)
            LOG("RTX HDR live skipped (needs an HDR display and a server model: RIFE/GMFSS)\n");
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
    // A Live DLSS 4.5 session (backend dlssg, or DLSS-G with effects): every pass of its plan is a stage in this
    // process and the session presents the last stage's frames; until the plan's stages are up the captured frames
    // (with effects: the echo route's frames) are shown as they are
    const bool dgSess = g_backend == BK_DLSSG || g_fgOver;
    host.flipQueue = dgSess && lkEnv("SMV_LIVE_DLSSG_TIMED") != "0";
    if (dgSess)
        LOG("DLSS 4.5: every pass a stage in this process, the session presents their frames\n");
    if (g_fgOver)
        LOG("DLSS-G with effects: the echo route runs them on every captured frame, DLSS-G generates from its "
            "frames\n");
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

        static Capture cap; // static: outlives this scope, single instance per process
        // dlssg in HDR packs scRGB -> R10A2 PQ in the exe. Must be set before init() (it sizes the chain and can
        // clear g_hdr on setup failure, before anything else reads it).
        cap.hdrPack = g_hdr && g_backend == BK_DLSSG;
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

        // DLSS 4.5 AT THE WORKING SIZE (SMV_DLSSG_AT_WORK=0 = off): below DLAA with no upscaler picked (RTX VSR, AMD FSR
        // 4, Anime4K and Upscale to make their frames at the presented size), DLSS-G runs at the working size itself
        // (never more pixels than the share picked), which liveWorkSize then keeps at least kDlssgMinWork on the
        // canvas's shorter side, so the capture is shrunk once, straight to it; every DLSS-G pass of a plan runs at
        // that size and the session enlarges every frame, real or generated, to the presented size with the native
        // fit's Lanczos (Host::lzOn): the order RIFE's tweens take. The frames shown while no plan presents are
        // enlarged the same way
        g_presW = g_presH = 0;
        g_dlssgWorkFloor = false;
        if (g_fgOver && !g_rtxVsr && !g_fsrUp && !g_a4kUp && !g_acnUp && !g_xessUp && g_upscaleH <= 0 &&
            lkEnv("SMV_DLSSG_AT_WORK") != "0")
        {
            g_dlssgWorkFloor = true;
            int pw = 0, ph = 0, md = 0, mw = 0, mh = 0, raised = 0;
            liveWorkSize((int)capW, (int)capH, pw, ph, md, mw, mh, &raised);
            const double k = pw > 0 ? (double)mw / pw : 1.0;
            // the frames = the working size when the picture fills the canvas, else the canvas at the working size's
            // scale (the nearest even sides) with the picture letterboxed inside
            const bool fills = pw == (int)W && ph == (int)H;
            const uint32_t sw = fills ? (uint32_t)mw : 2u * (uint32_t)lround(W * k / 2.0),
                           sh = fills ? (uint32_t)mh : 2u * (uint32_t)lround(H * k / 2.0);
            if (!(mw > 0 && k < 1.0 && sw < W && sh < H))
                g_dlssgWorkFloor = false;
            else
            {
                // SMV_DLSSG_STRETCH=1: DXGI's bilinear stretch of a swap chain at the frames' size instead (no enlarge
                // here: the look A/B)
                host.lzOn = lkEnv("SMV_DLSSG_STRETCH") != "1";
                LOG("DLSS 4.5 at the working size: its frames %ux%u (the %dx%d working size%s), %s the %ux%u "
                    "presented%s\n",
                    sw, sh, mw, mh, raised == 3 ? ", raised to DLSS 4.5's minimum, 128 px a side" : "",
                    host.lzOn ? "enlarged to" : "stretched over", W, H,
                    host.lzOn ? " by the native fit's Lanczos" : " (SMV_DLSSG_STRETCH=1)");
                g_presW = W;
                g_presH = H;
                host.winW = W;
                host.winH = H;
                W = sw;
                H = sh;
            }
        }
        rc = host.init();
        if (rc)
            return rc;
        g_fgDevice = dgSess ? host.device.Get() : nullptr;
        host.rtStart();

        // The DLSS 5 pass inside this process. Server backends only (DLSS-G with effects runs it on the echo
        // route). The native host runs it on the model frame after Restore and
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

        // HUD created up front (not after warmup), opening with the loading note that names the picked models and
        // effects (the GUI's --label) until the first stats window: the passthrough overlay looks identical to the
        // source until smoothing starts, so without a visible cue the start reads as "nothing happened".
        // WS_EX_NOACTIVATE keeps it clear of DLSS-G's foreground gate.
        // Fill presents nothing until the first served group, so its note starts at the window's corner and
        // moves to the screen's once the overlay runs.
        Hud hud;
        HudMeter meter;
        FillMouse fillMouse; // started once the overlay runs (Fill only), stopped before the teardown
        if (!g_noHud && !park)
            hud.create((g_fill ? cap.clientScreenX : host.posX) + 16, (g_fill ? cap.clientScreenY : host.posY) + 16,
                       loadingNote().c_str());

        PipeServer srv;
        // the final enlarge's tap tables: the native k_rsTab once the native pipeline is up (SMV_DLSSG_LZ_TAB=cpu =
        // the CPU port, the A/B of the two)
        if (host.lzOn)
            host.lzTabFn = [&srv, cpu = lkEnv("SMV_DLSSG_LZ_TAB") == "cpu"](int in, int out,
                                                                            std::vector<uint8_t>& tab) {
                return nativeRsTabHost(cpu ? nullptr : srv.nr, in, out, tab);
            };
        double srcMeasFps = 0; // the source rate measured before the server spawns (0 = not measured)
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
                srcMeasFps = base;
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
                // Smooth Motion / AMD FSR 4 run the powers of two only: room for the first that reaches the target (the
                // node above it, lockedMultiple's step-up); the GPU fit decides whether it runs, Fit to the GPU off
                // runs it
                if (!g_noAdapt && g_backend == BK_SERVER && (g_serverBackend == L"fruc" || g_serverBackend == L"fsrfg"))
                {
                    int p2 = 2;
                    while (p2 < (1 << kFrDepth) && p2 * base < g_targetFps)
                        p2 *= 2;
                    slots = (std::max)(slots, p2);
                }
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
                    if (!g_fgOver) // DLSS-G over echo logs its own multiple below
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
            // DLSS-G with effects: the plan's stages make the in-between frames, the echo route holds one slot
            if (g_fgOver)
                genFrames = ceilGen = 0;
            // A derived fixed multiplier is drift-tracked (re-derived in the stats tick),
            // so size the server at the slot CEILING and let each group's ladder carry the
            // current value. An explicit --gen (harness/perf use) keeps the exact classic spawn.
            const bool fixedDerived = g_noAdapt && g_targetFps > 0 && !g_genExplicit && !g_fgOver;
            const int spawnGen = fixedDerived ? ceilGen : genFrames;
            // The backend's start can take seconds (its kernels are compiled on a first run, then
            // the weights and the buffers). srv.start() blocks for that whole cold start, and
            // server backends do no warmup pre-present, so the overlay would show nothing until
            // it finished - which reads as "it never turned on" (while DLSS-G, no server, feels
            // instant). Load on a BACKGROUND thread and present the raw captured frames meanwhile,
            // so the overlay is visibly LIVE from the first second; it just isn't smoothed yet.
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
            ULONGLONG loadSizeTick = 0;
            bool loadGone = false; // the window closed during that resize: exit 7, never revived
            while (srvRc.load(std::memory_order_acquire) < 0)
            {
                pumpMessages();
                // window tracking during the cold model build: the main loop's tracker only starts
                // after the server is up, so the passthrough overlay and the "loading ... model" note
                // follow a dragged target here (Fill: only the note, its overlay covers the monitor)
                if (!host.park && !g_monitor)
                    followTarget(target, host.hwnd, fb, cap.cropX, cap.cropY, hud, !g_fill);
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
                        LOG("Loading %ls (first build at this resolution can take up to a minute)...\n", nm);
                    else
                        LOG("Loading %ls (%ls; first build at this resolution can take up to a minute)...\n", nm,
                            g_modelNote.c_str());
                    if (hud.hwnd)
                        _snwprintf_s(hud.text, _TRUNCATE, L"%ls", loadingNote().c_str());
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
            // the host may hold fewer slots than asked (its ring fits the free video memory): the ladders and the
            // drift tracker stay inside them
            if (srv.shmSlots && genFrames > (int)srv.shmSlots - 1)
                genFrames = (int)srv.shmSlots - 1;
            if (srv.outbufAck && host.outBuf)
                LOG("present path: direct from VRAM (shared output buffer)\n");
            // a started host always holds the zero-copy capture import (nativeRefusal)
            LOG("capture path: zero-copy (shared texture, no CPU readback)\n");
            // the rate measurement and the load-loop passthrough drained WGC's frames, and a static
            // source (the start-on-paused workflow) sends no more: this refresh asks a window that
            // repaints on WM_PAINT for a current one; the warm-up below starts from the newest
            // drained frame when none comes
            cap.requestRefresh();
        }

        const size_t capBytes = (size_t)capW * capH * 4; // capture-size frames (server input)
        std::vector<uint8_t> buf(capBytes);
        uint32_t shmSeq = 0;   // shm-mode: frame token sequence
        double statLatSum = 0; // capture->present latency accumulation (ms)
        uint64_t statLatN = 0;
        // The latency figure (the HUD's "~N ms behind", the `live:` line) = SMV's present call - the picture's capture
        // stamp + the route's lag + ONE PANEL REFRESH. WGC stamps a capture one refresh after the picture's own flip
        // (PresentMon beside a visible source: the stamps sit one refresh period after the source's displays, a
        // DWM-composed source the same), so present - stamp alone read a passthrough as ~0 ms behind a picture it
        // followed by a whole refresh, negative at times. The refresh = the overlay monitor's mode (steady; the frame
        // statistics' measured rate wobbles between stats ticks), the measured period only when the mode gives none.
        // Not in the figure: SMV's own present -> flip (tear-allowed presents, well under a refresh).
        double modeRefreshMs = 0;
        {
            MONITORINFOEXW mi{};
            mi.cbSize = sizeof(mi);
            GetMonitorInfoW(MonitorFromWindow(host.hwnd, MONITOR_DEFAULTTONEAREST), &mi);
            DEVMODEW dm{};
            dm.dmSize = sizeof(dm);
            if (EnumDisplaySettingsExW(mi.szDevice, ENUM_CURRENT_SETTINGS, &dm, 0) && dm.dmDisplayFrequency > 1)
                modeRefreshMs = 1000.0 / dm.dmDisplayFrequency;
        }
        auto stampLagMs = [&]() {
            return modeRefreshMs > 0 ? modeRefreshMs : host.refreshQpc100 > 0 ? host.refreshQpc100 / 1e4 : 0.0;
        };
        LOG("latency figure: present - capture stamp + lag + one refresh (%.2f ms)\n", stampLagMs());
        // SMV_LAT_TRACE=<file> (a measurement aid, the server route): every served present as `P <present count> <its
        // QPC> <the frame's capture stamp> <the lag term ms>` and each new frame statistics reading as `S <displayed
        // present count> <its sync QPC> <the refresh period> <PresentRefreshCount> <SyncRefreshCount>` (QPC in 100 ns
        // units, the capture stamp's timeline), written at the end
        // and every HUD refresh as `H 0 <its reading's QPC> 0 <the source fps> <the presented fps> <the latency ms>`, then the
        // capture's A / N / Q rows (Capture::dumpStamps)
        struct LatRow
        {
            char kind;
            uint64_t a;
            int64_t b, c;
            double d, e = 0, f = 0;
        };
        std::vector<LatRow> latRows;
        wchar_t latPath[MAX_PATH] = L"";
        const bool latOn = GetEnvironmentVariableW(L"SMV_LAT_TRACE", latPath, MAX_PATH) > 0;
        UINT latFsPrev = 0;
        if (latOn)
            latRows.reserve(1 << 16);
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
            if (g_noGpuFit)
                LOG("GPU fit off (--no-gpu-fit): the target holds, no video memory fit\n");
        }
        // THE THROTTLE ("Fit to the GPU"): the grid marches on an EFFECTIVE target below the user's, computed from the
        // host's own GPU time per group so every source frame is taken (xqThrottle); at its floor, the source's rate,
        // the pairs carry their real frame alone.
        double effTarget = adaptTarget; // throttled output target (<= adaptTarget)
        uint64_t thrDropBase = 0;       // cap.dropped at the window start
        double thrCleanMs = 0;          // drop-free time toward a budget step up
        double thrLastLog = 0;
        double thrWinT0 = 0;     // control-window start (ms); 0 = not started
        double thrBudget = 0.95; // the share of the source's interval a group's GPU work may take
        double thrProbeGapMs = 4000.0, thrProbeLast = 0; // at the floor: the next tween re-measure (4 .. 60 s)
        bool thrProbing = false;
        // a re-measure adds one tween after each pair's real frame, computed and stored but never presented (the hidden
        // slot of sendPairStream / XqGroup::show): the host re-times the tween under the probe's load while the picture
        // keeps the floor's real frames, pacing and latency; the source frame the extra work costs is the same either way
        // a re-measure leaves the floor only once kThrProbeWins consecutive windows afford a tween: on a power-capped
        // GPU the first half second of load after seconds at the floor runs at a boosted clock and reads "fits" for
        // work that does not fit sustained (a 55 W cap: 1.1-1.7 tweens a pair in the first window, the floor in the
        // next, so the gap never grew and a re-measure stuttered every 5 s); one floor answer ends the re-measure
        uint32_t thrProbeWins = 0; // windows the running re-measure has answered "fits"
        const uint32_t kThrProbeWins = 2;
        bool thrFloor = false; // the throttle's floor: every pair carries its real frame alone (sendPairStream)
        double thrDropEma = 0; // drops a second, smoothed over the 500 ms windows (~2 s)
        double thrMoveT = 0;   // when the target last moved: drops within 2 s of it are the move's, not the budget's
        // Smooth Motion's midpoint tree priced per route from the host's call and per-slot store times (gFrCallUs /
        // gFrStoreUs): the tweens a pair the direct tree affords ((M - 1) calls + M stores) and the pipelined one (M
        // stores + the longer of the compute thread's calls and instance 0's worker's, fruc-pipe); 0 = not priced (the
        // plain tween time rules)
        double thrMd = 0, thrMp = 0;
        // the pipelined tree's calls at the pipelined groups' own prices (the host's gFrPipeMainUs / gFrPipeWkUs); until
        // they are timed, the direct call times these (measured at 640x360 / 1080p / 1440p: the compute thread's calls
        // 1.04-1.17 x the direct call, a worker call beside them 1.33-1.66 x)
        const double kFrPipeMainK = 1.17, kFrPipeWkK = 1.66;
        auto frucPipePrices = [&](double fc, double& pm, double& pw) {
            pm = srv.nr ? srv.nr->gFrPipeMainUs.load() / 1000.0 : 0.0;
            pw = srv.nr ? srv.nr->gFrPipeWkUs.load() / 1000.0 : 0.0;
            if (pm <= 0.0)
                pm = fc * kFrPipeMainK;
            if (pw <= 0.0)
                pw = fc * kFrPipeWkK;
        };
        // an M ladder's pipelined calls (M a node, x4 or deeper): instance 0's worker runs the next pair's levels 1..k
        // (2^k - 1 calls, k = the depth less one, capped by SMV_FRUC_PIPE_K), the compute thread the shown pair's
        // deeper levels (M - 2^k calls), side by side
        auto frucPipeCalls = [](double m, double pm, double pw) {
            const double half = (std::min)(m / 2.0, std::ldexp(1.0, frucPipeK()));
            return (std::max)((m - half) * pm, (half - 1.0) * pw);
        };
        // the tweens a pair the pipelined tree affords in B ms with M stores of fs: frucPipeCalls solved for M,
        // continuous between the nodes
        auto frucPipeAfford = [](double B, double fs, double pm, double pw) {
            const double kCap = std::ldexp(1.0, frucPipeK());
            // up to the cap (2^k = M / 2): M fs + max(M / 2 pm, (M / 2 - 1) pw) <= B
            double m = (std::min)(B / (fs + pm / 2.0), (B + pw) / (fs + pw / 2.0));
            if (m / 2.0 > kCap) // past it: M fs + max((M - 2^k) pm, (2^k - 1) pw) <= B
                m = (std::min)((B + kCap * pm) / (fs + pm), (B - (kCap - 1.0) * pw) / fs);
            return m - 1.0;
        };
        const double kThrWinMs = 500; // control window: long enough to measure a drop RATE
        double emaDt = 0;             // smoothed capture interval (ms), paces the group
        ULONGLONG lastArrival = 0;
        int idleSlotIdx = -1;     // last presented slot, re-presentable while the
        uint32_t idleSlotSet = 0; // source is static (the server only rewrites a half
                                  // when a NEW group lands, so the content is stable)
        auto slotOffset = [&](uint32_t set, uint32_t i) -> size_t {
            return srv.shmInBytes + ((size_t)set * srv.shmSlots + i) * srv.shmSlot;
        };
        // present one output slot GPU-direct from the shared VRAM ring (slots never touch host memory)
        auto presentSlotFrom = [&](uint32_t set, uint32_t i) -> bool {
            return host.presentTail(host.outBuf.Get(), slotOffset(set, i) - srv.shmInBytes, srv.shmPitch);
        };

        // warmup: block for the first frame. The server route takes the
        // newest frame the rate measurement / the load loop drained when nothing newer arrives in 250 ms (a static
        // source: a paused video, an idle desktop, a Chromium or layered window, which InvalidateRect never makes
        // present) and the queue loop sends it as the first group
        bool sendNewest = false;
        {
            const ULONGLONG warmStart = GetTickCount64();
            ULONGLONG deadline = warmStart + 5000;
            // SMV_LIVE_TEST=noframe: the warm-up sees no frame, the trigger of the no-frame exit's teardown (every
            // window, an undrawn, cloaked or minimized one too, hands WGC an initial frame)
            char testEv[16] = {};
            const bool testNoFrame =
                GetEnvironmentVariableA("SMV_LIVE_TEST", testEv, sizeof(testEv)) > 0 && strcmp(testEv, "noframe") == 0;
            const bool newestOk = !testNoFrame && g_backend == BK_SERVER && cap.interop && srv.captexAck;
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
                if (testNoFrame)
                    got = 0;
                if (!got && newestOk && GetTickCount64() - warmStart >= 250 && cap.newestToShared())
                {
                    LOG("static source: starting from the newest captured frame\n");
                    sendNewest = true;
                    got = 1;
                }
            }
            if (!got)
            {
                LOG("no frames captured in 5s (the target drew nothing: off-screen, minimized or in exclusive "
                    "fullscreen?)\n");
                return 1;
            }
            // the direct routes show the first frame at once; DLSS-G with effects presents the server's frames only
            if (g_backend == BK_IDENTITY || g_backend == BK_DLSSG)
            {
                if (!host.presentFrame(buf.data()))
                {
                    LOG("first present failed\n");
                    return 1;
                }
            }
            // server backends: no pre-present (capture and output sizes differ under --fit fill;
            // the first served group arrives within one round trip anyway)
            LOG("live overlay running\n");
            if (g_fill)
                hud.move(host.posX + 16, host.posY + 16);
            if (g_fill && !park && !g_noFillMouse)
            {
                int dw = 0, dh = 0, x0 = 0, y0 = 0;
                lkFitRect((int)capW, (int)capH, (int)(g_presW ? g_presW : W), (int)(g_presH ? g_presH : H), dw, dh, x0,
                          y0);
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
        // RTSS (RivaTuner Statistics Server) injects its hook into Direct3D processes and limits their presents inside
        // Present: a frame limit set there caps this overlay with nothing in the throttle or pacing lines saying why,
        // so each session names the hook once (SMV_RTSS_MODULE = another module name: the trigger test's lever)
        wchar_t rtssModule[64] = L"RTSSHooks64.dll";
        GetEnvironmentVariableW(L"SMV_RTSS_MODULE", rtssModule, 64);
        bool rtssNamed = false;
        UINT statPresentBase = 0;
        host.sc->GetLastPresentCount(&statPresentBase);
        // cap.dropped at the last stats tick; it starts at the drops of the model load, which are not this window's
        uint64_t statDropBase = cap.dropped;
        uint64_t statCaptured = 0;
        int fixWantPrev = -1; // fixed-ladder server derivation: last derived gen (two-window hysteresis)
        // Live DLSS 4.5's plan (the cascade): the session's ONE plan, made at the 2 s derive when the same multiple
        // comes out of dlssgSnap twice in a row (cascWantPrev; cascPlan), M = cascM1 x cascM2 x cascM3 in up to three
        // stages (cascM3 1 = two, cascM2 1 = one); cascWarm = the stages run and are fed until the presented one
        // shares a frame; cascShown = the multiple the stats line reports while the session presents the plan's
        // frames (0 = no plan presents); cascLow = 2 s windows in a row a watched plan (three stages, or two above
        // the one-pass limit) ran short of its multiple; cascMaxM = such a plan fell short: the next plan's multiple
        // at most (what the GPU held, 0 = none)
        int cascM = 0, cascM1 = 0, cascM2 = 0, cascM3 = 1, cascWantPrev = -1, cascShown = 0, cascLow = 0, cascMaxM = 0;
        bool cascTwoTopShort = false; // the two-pass top (the limit x the limit) fell short this session
        bool cascFitOffSaid = false;  // Fit to the GPU off: this plan's shortfall line was logged
        ULONGLONG cascStepTick = 0;   // an in-place step's ask (cascSwitch), until every helper runs it
        // THE CLIMB (SMV_LIVE_DLSSG_CLIMB=0 = off: the target's rung first): a plan for a target past the panel's refresh
        // whose rung (cascTopM) lies past the first rung above one pass starts its helpers at the target's multiples
        // (their memory for the top), asks
        // them down to that first rung before its takeover and climbs in place, a rung of at least 1.2 x at a time,
        // while each holds its multiple (cascHold, capture drops at most 3 a second); a rung that does not is the ceiling
        // (cascCeilM): the plan steps back to the rung that held (cascHeld = M, m1, m2, m3) and the next try waits
        // cascClimbWait (15 s, doubling to 240 s), so a ceiling check is one in-place switch, never a restart
        const bool cascClimbOn = lkEnv("SMV_LIVE_DLSSG_CLIMB") != "0";
        // A rung HOLDS when the session presents this share of its multiple. A rung in good order presents 95 % and
        // more (DLSS-G's own pacing takes the rest); one past what the session can show presents under 90 % at the
        // same rate as the rung below it, its other frames made and never shown. SMV_LIVE_DLSSG_HOLD = another share,
        // 0.50 .. 0.99 (the A/B)
        double cascHold = 0.90;
        {
            const double h = atof(lkEnv("SMV_LIVE_DLSSG_HOLD").c_str());
            if (h >= 0.5 && h <= 0.99)
                cascHold = h;
        }
        int cascTopM = 0, cascCeilM = 0, cascHoldWins = 0, cascClimbWins = 0;
        int cascHeld[4] = {0, 0, 0, 0}, cascFirst[4] = {0, 0, 0, 0}; // cascFirst = the first rung, asked at takeover
        bool cascClimbing = false; // the rung on show is a climb's try, judged on its second full window
        ULONGLONG cascClimbTick = 0, cascClimbWait = 0;
        // a non-whole multiple (dlssgKeepPlan): the helpers make cascM, the session shows every cascKeep-th ring frame
        // and makes none itself; cascShown / cascKeep = the multiple presented
        int cascKeep = 1, cascKeepPrev = 1;
        // display-timed presents (Host::dtSet): the source's rate over the last stats window, and the source frame the
        // ring's frame presented last belongs to (another one = the first frame of a source frame's set)
        double dtCapFps = 0;
        uint32_t dtInput = 0xFFFFFFFFu;
        bool cascTried = false, cascWarm = false;
        // the session has no pass of its own to fall back to: after a plan's end the next stats
        // window plans again (cascNow: without the second matching window). cascLost = the plans in a row that ended
        // because a stage ended or stopped presenting: three are made again, the count starts over once a plan has
        // presented for a minute (cascUpTick = its takeover)
        bool cascNow = false;
        int cascLost = 0;
        ULONGLONG cascUpTick = 0;
        bool hidden = false;
        // the HUD switched by a stdin set (nativeLiveSets flags it, this thread owns the window): off hides it, on shows
        // it, made here when the session started without one; read at the top of both loops
        auto applyHudWant = [&] {
            const int hw = g_hudWant.exchange(-1);
            if (hw < 0)
                return;
            g_noHud = hw == 0;
            if (hw && !hud.hwnd && !park)
                hud.create((g_fill ? cap.clientScreenX : host.posX) + 16, (g_fill ? cap.clientScreenY : host.posY) + 16,
                           L"");
            hud.show(!hidden);
            LOG("live HUD %s%s\n", hw ? "on" : "off", hw && park ? " (parked: no window)" : "");
        };
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
        // (DLSS-G over the echo route: the floor)
        double idleStepMs = g_fgOver ? 100.0 : adaptTarget > 0 ? gridStep / 10000.0 : 1000.0 / (genFrames + 1);
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
        double nodeHz = 0;      // a locked node above the target (lockedMultiple's step-up): its own present rate
        // the floor follows the rate we actually sustain; recomputed on every throttle step
        auto refloor = [&] {
            if (panelHzMode <= 1.0)
                return;
            // The floor is 0.95x the LONGER of the panel refresh period and the effective
            // output period. Dropping the panel term once the user's target is above the
            // refresh rate was tried and is unstable: a lower effective rate then means a
            // LONGER floor, which stretches groups, which trips the throttle again - measured
            // as a spiral down to ~160 fps. The panel period bounds that feedback.
            // A node above the target presents faster than both: the floor follows it, else
            // its slots outrun the floor, each group overruns its source period and source
            // frames drop (x64 past a 400 target: ~3 drops a second, the GPU price swinging
            // with them and the lock falling to x32 and back every ~20 s).
            const double hz = (std::max)((std::max)(effTarget, panelHzMode), nodeHz);
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
                // until the throttle has timed a group the target starts at one tween a pair (twice the source rate
                // measured before the spawn), never above the panel's rate or the user's target: a start the GPU
                // cannot afford fills the queue (the panel's rate = ~15 tweens a pair of 1440p RIFE, ~240 ms a group),
                // and the drops while that backlog drains outlast the move's 2 s window and cut the budget for about a
                // minute; 10000 asked GMFSS for the whole slot ladder a pair (213 tweens, seconds a group)
                const double startT = srcMeasFps > 0 ? (std::min)(panelHzMode, 2.0 * srcMeasFps) : panelHzMode;
                if (adaptTarget > startT && !g_noGpuFit)
                {
                    effTarget = startT;
                    gridStep = 1e7 / effTarget;
                    LOG("throttle: starting at %.0f fps until a group is timed\n", effTarget);
                }
                refloor();
                LOG("present pacing: floor %.3fms (panel mode %u Hz, target %.0f fps%s)\n", minSpaceMs,
                    dm.dmDisplayFrequency, adaptTarget,
                    adaptTarget > panelHzMode ? ", above panel: presents run tear-allowed past refresh" : "");
            }
        }
        HANDLE paceTimer =
            CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
        // the direct loop's follow poll (followTarget): its wait ends on this timer while the target moves
        HANDLE followTimer =
            CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
        ULONGLONG followUntil = 0;
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
        int64_t smoothTs = 0;           // EMA-smoothed pair clock (see sendPairStream)
        int64_t prevPaceTs = 0;         // the previous frame's place on that clock
        std::vector<double> sendGridMs; // each slot of the last sent group on that clock (ms), its present time - delay
        // fruc-pipe: Smooth Motion's tree pipelined across pairs (nativeFrucPipeAdvance): a group of x4 or deeper shows
        // the previous pair, so its deepest level runs beside the next pair's levels 1..k on instance 0's worker (k =
        // the depth less one, nativeFrucPipeSpec), one source period later. A
        // ladder goes pipelined when the direct tree cannot hold it (M - 1 > 1.03 x the direct tree's tweens a pair)
        // and the pipelined one affords it, back to direct only with 10 % to spare (M - 1 <= 0.9 x them: a direct tree
        // at its edge flaps its lock, and every switch shows, entering holds one source period, leaving skips one
        // pair), never sooner than 2 s after the last switch. SMV_FRUC_PIPE=0 = never, =force =
        // every x4 or deeper ladder, the fixed multipliers too (a gate's lever: the same exe both ways on one source)
        wchar_t fpv[8]{};
        const bool fpSet = GetEnvironmentVariableW(L"SMV_FRUC_PIPE", fpv, 8) != 0;
        const bool frucPipeOn = !(fpSet && fpv[0] == L'0');
        const bool frucPipeForce = fpSet && !wcscmp(fpv, L"force");
        // AMD FSR 4's tree pipelined the same way (nativeFsrfgPipeAdvance: the next pair's level-1 Optical Flow and call
        // on their own streams beside this pair's deeper calls), by the same rule at its own price (fsrfgPipeCost; the
        // route needs fsrfgPipeOk), from x2 up (its Optical Flow runs beside the shown pair's level-1 call).
        // SMV_FSRFG_PIPE=0 = never, =force = every ladder and fixed multiplier (=serial the same route without its
        // overlap, an A/B lever)
        wchar_t fgpv[8]{};
        const bool fgpSet = GetEnvironmentVariableW(L"SMV_FSRFG_PIPE", fgpv, 8) != 0;
        const bool fsrfgPipeOn = !(fgpSet && fgpv[0] == L'0');
        const bool fsrfgPipeForce = fgpSet && (!wcscmp(fgpv, L"force") || !wcscmp(fgpv, L"serial"));
        bool frucPiped = false;
        double frucPipeT = -1e9; // nowMs of the last route switch
        // the whole multiple M an adaptive target runs as M's ladder (0 = the grid). On the grid nearly every output
        // frame is a tween, so r times the source's rate costs ~r model calls a pair; M's ladder costs M - 1 and shows
        // the real frame as it is. So with r = the throttle's target over the source's rate (the tweens it affords a
        // pair, or the user's target), M = ceil(0.97 r) gives more frames for no more work whenever it stays inside
        // the user's target and the slot ceiling; the grid is left only to a target between two multiples that
        // binds. A lock holds while its M - 1 tweens fit; the ladder above it is admitted at its price (upK below).
        // The rate is a slow average of the pair clock's period (emaDt swings ~ +-2 % with WGC's delivery
        // quantization). SMV_LIVE_GRID_LOCK=0 = the grid at every target.
        wchar_t glv[8]{};
        const bool gridLock = !(GetEnvironmentVariableW(L"SMV_LIVE_GRID_LOCK", glv, 8) && glv[0] == L'0');
        // SMV_FRUC_STEPUP=0 = the midpoint tree never runs a node above the user's target (the largest that fits)
        wchar_t fsu[8]{};
        const bool frStepUp = !(GetEnvironmentVariableW(L"SMV_FRUC_STEPUP", fsu, 8) && fsu[0] == L'0');
        uint32_t lockLogged = 0;
        double lockT = 0; // when lockLogged last changed (the step-up's dwell)
        double lockDtMs = 0;
        // the ladder above a lock is admitted at a price, not a band: M + 1 costs this lock's readings pre + M tw
        // times upK, the excess the ladder above showed over such a prediction the last time one ran (every stage
        // runs slower once the GPU has no idle gaps: GMFSS x3 over x2's readings ~+4 % at a 175 W ceiling, ~+8 %
        // at 160 W under the power cap, so a half-tween band would price x3 ~5 ms over its cost). kUpKDefault is
        // also upK's floor: a measured excess is never read below it, so a step-up always keeps that margin inside
        // the budget (a prediction admitted at the budget's edge ran ~10 % over it). One attempt per regime: a
        // step-up records its prediction (upPred for upM); the new lock's settled readings (kUpSettleMs) or its fall
        // set upK to what it cost, kept per ladder in upKAt (a lower lock's own price never re-admits a ladder above
        // at less than that ladder showed), so a failed step is retried only once the readings improve that much,
        // or as a probe after a back-off that doubles per failure (kUpBackoff0 to kUpBackoffMax: a price learned
        // through a transient, a CPU burst lowering the GPU's ceiling, would otherwise hold the session below a
        // ladder it affords; at a real cap the probe costs ~3 s of the ladder above). A probe prices the ladder
        // halfway from the excess it showed to the default, and a refused probe keeps that halfway price and doubles
        // the back-off: a transient's excess relaxes over a few probes, a marginal ladder (its cost drifting around
        // the period with the GPU's clocks) is looked at ever more rarely instead of flapping with the lock below.
        // The back-off ends only once the ladder it guards has held kUpResetMs, so a ladder that keeps falling
        // keeps its back-off growing. Enter
        // with margin, hold with less: a lock holds while its cost stays under the budget times kUpHold (a ladder
        // at the edge of the budget would otherwise leave and re-enter every few seconds) and under kUpPeriod of
        // the source period: a ladder that costs more than the period keeps up only by losing source frames (the
        // capture's dropped arrivals, about one a second at 3 to 5 % over, under the throttle's own drop cut), so the
        // band never reaches the period. It falls only once it has been over for kUpOverMs (the group times are a
        // 5-group EMA) and never within kUpSettleMs of its lock; the back-off ends once the ladder it guards has
        // held kUpResetMs
        const double kUpKDefault = 1.05, kUpSettleMs = 2000.0, kUpBackoff0 = 4000.0, kUpBackoffMax = 300000.0;
        const double kUpHold = 1.10, kUpPeriod = 0.98, kUpOverMs = 1000.0, kUpResetMs = 120000.0;
        double upK = kUpKDefault, upPred = 0, upFailT = 0, upBackoffMs = kUpBackoff0, upOverT = 0;
        uint32_t upM = 0;
        // A DEEP TRY: a Smooth Motion step-up to a deeper pipelined ladder is priced at the running ladder's pipelined
        // calls (or the ratios learned there), yet a call costs less the more bridge instances run side by side
        // (2560x1440, 21.5 fps: x4's prices put x8 at ~1.23 x what x8 then measured running, ~1.3 x the budget with
        // kUpKDefault for an x8 whose own price fits). Such a ladder is admitted up to kDeepTry x the budget as a timed
        // run, entering the pipelined route at once (deepTry = that ladder until it settles); its own prices then hold
        // it in the band or let it fall. A tried ladder keeps a back-off of its own (deepBackoff: kDeepBackoff0
        // doubling, kUpBackoffMax once its own price came out at or above the price it was tried at: no overprice to
        // find), so a failed try never holds back the step-ups the prices admit (the shared back-off above); deepLock
        // = the tried ladder while it is the lock, until it has held kUpResetMs. Only one node up from x4 or below
        // (kDeepFromMax): the compute thread's call price holds its wait for the worker's join, spread over 2 calls at
        // x4 and over 32 at x64, so the overprice is a shallow ladder's (x2 -> x4 0.91-0.94 x, x4 -> x8 0.81-0.85 x
        // their own prices; x16 -> x32 at 854x480 and x64 -> x128 1.03-1.05 x: none, a try there only falls)
        const double kDeepTry = 1.4, kDeepBackoff0 = 15000.0;
        const uint32_t kDeepFromMax = 4;
        uint32_t deepTry = 0, deepLock = 0;
        double deepPred = 0;
        bool deepSettled = false;
        std::map<uint32_t, double> deepFailAt, deepBackoff;
        // the tried ladder falls (c = its price now, 0 = not read): its back-off in seconds
        auto deepFall = [&](double c) {
            double& b = deepBackoff[deepLock];
            b = c > 0.0 && c >= deepPred ? kUpBackoffMax : b > 0.0 ? (std::min)(b * 2.0, kUpBackoffMax) : kDeepBackoff0;
            deepFailAt[deepLock] = nowMs();
            deepLock = deepTry = 0;
            return b / 1000.0;
        };
        std::map<uint32_t, double> upKAt; // the excess each stepped-up ladder showed over its prediction, by ladder
        bool upSettled = false;           // the stepped-up lock has logged its settled price
        // an M ladder's route (fruc-pipe), before the switch dwell: the sender runs it, lockedMultiple prices it. The
        // direct / pipelined trees' tweens a pair (frucAffD / frucAffP) come off the throttle's tree prices over the
        // ladder's own budget (lockedMultiple's: the faster of the two period averages), never over emaDt alone: that
        // swings with the source's delivery (+-9 % on the parked testsrc) wider than the band between the two tests
        double frucAffD = 0, frucAffP = 0;
        // AMD FSR 4 pipelined: the session stream runs its pre outside the level-1 chain (pp, the pack), the shown
        // pair's M - 2 deeper calls once the last level-1 call is done, then its M stores; fgOfL1St runs the new pair's
        // Optical Flow (po) from the pack on, beside the last level-1 call, and fgL1St the new pair's call (p1) after
        // the deeper calls. An M ladder costs the longest of the Optical Flow chain (po a pair), the FSR + session
        // chain (the deeper calls + the later of p1 and the stores + the next pack) and half the loop through both
        // (pack, Optical Flow, call, the next pair's deeper calls and stores: two pairs' worth) (the host's gFgPreUs /
        // gFgOfUs / gFgL1Us, timed on either route; false = not priced yet)
        auto fsrfgPipeParts = [&](double& pp, double& po, double& p1) {
            pp = srv.nr->gFgPreUs.load() / 1000.0;
            po = srv.nr->gFgOfUs.load() / 1000.0;
            p1 = srv.nr->gFgL1Us.load() / 1000.0;
            return po > 0.0 && p1 > 0.0;
        };
        // AMD FSR 4's call price for the route sums: before an x4 or deeper group timed a call (gFrCallUs 0) the level-1
        // call's FSR span stands in (a call of the same kind), so a session whose direct x2 does not fit still prices
        // the pipelined x2 and the ladder above it; the ladder stays untimed (one node a step) until a call is timed
        auto fsrfgCall = [&](double fc) { return fc > 0.0 ? fc : srv.nr->gFgL1Us.load() / 1000.0; };
        auto fsrfgPipeCost = [&](double m, double fc, double fs) {
            double pp, po, p1;
            fsrfgPipeParts(pp, po, p1);
            const double deep = (m - 2.0) * fc;
            return (std::max)({po, deep + (std::max)(p1, pp + m * fs), (pp + po + p1 + deep + m * fs) / 2.0});
        };
        // the tweens a pair the pipelined route affords in B ms: fsrfgPipeCost solved for M, continuous between the
        // nodes (0 = not priced, or x2 alone takes more than B)
        auto fsrfgPipeAfford = [&](double B, double fc, double fs) {
            double pp, po, p1;
            if (!fsrfgPipeParts(pp, po, p1) || fc <= 0.0 || po > B || (std::max)(p1, pp + 2.0 * fs) > B ||
                pp + po + p1 + 2.0 * fs > 2.0 * B)
                return 0.0;
            const double s = fc + fs;
            return (std::min)({(B - p1) / fc + 2.0, (B - pp + 2.0 * fc) / s, (2.0 * B - pp - po - p1 + 2.0 * fc) / s}) -
                   1.0;
        };
        auto frucPipeWant = [&](uint32_t m) {
            const double fc = srv.nr->gFrCallUs.load() / 1000.0, fs = srv.nr->gFrStoreUs.load() / 1000.0;
            if (srv.nr->fsrfg)
            {
                // AMD FSR 4: over the whole budget (its x2 group's whole time holds the pre)
                const double B = (std::min)(lockDtMs, emaDt) * thrBudget, c = fsrfgCall(fc);
                frucAffD = 1.0 + (B - srv.nr->gFrBaseUs.load() / 1000.0) / (c + fs);
                frucAffP = fsrfgPipeAfford(B, c, fs);
            }
            else
            {
                const double B = (std::min)(lockDtMs, emaDt) * thrBudget - srv.nr->gIntUs.load() / 1000.0;
                double pm, pw;
                frucPipePrices(fc, pm, pw);
                frucAffD = (B - fs) / (fc + fs);
                frucAffP = frucPipeAfford(B, fs, pm, pw);
            }
            const double need = (double)(m - 1);
            if (!frucPiped)
                return need > frucAffD * 1.03 && need <= frucAffP * 1.03;
            return need > frucAffD * 0.9;
        };
        // Fit to the GPU off runs no throttle, whose tree prices (thrMd / thrMp) open the route choice: there the
        // host's own timings do, once a call is timed (AMD FSR 4: its x2 group too)
        auto noFitPriced = [&]() {
            if (!g_noGpuFit || !srv.nr)
                return false;
            const double fc = srv.nr->gFrCallUs.load() / 1000.0;
            return srv.nr->fsrfg ? fsrfgCall(fc) > 0.0 && srv.nr->gFrBaseUs.load() > 0 : fc > 0.0;
        };
        auto lockedMultiple = [&](double target) -> uint32_t {
            lockDtMs = lockDtMs > 0 ? lockDtMs + (emaDt - lockDtMs) / 64.0 : emaDt;
            const double r = target * lockDtMs / 1000.0;
            const double userCap = adaptTarget * lockDtMs / 1000.0;
            const double ceiling = srv.shmSlots ? srv.shmSlots : genFrames + 1;
            // FRUC's midpoint tree computes exact phases only at its nodes k / 2^L (L <= kFrDepth, one bridge instance
            // a level): any other ladder or grid phase snaps to the nearest node and costs the whole tree (3 calls a
            // pair at depth 2), so a tween's cost and the throttle's answer swing with the mode. Its ladders are the
            // powers of two up to x512 (fewer when an instance could not be made). A user's target between two nodes
            // (above the lower one by more than 3 %) runs the node ABOVE it (`above`), the first that reaches the
            // target however far past it or the panel that lands (24 fps at 1000: x64), whenever its price fits and the
            // present floor (refloor) can space its slots: no node ladder makes x3, NvOFFRUC is clean only at t = 0.5.
            // Every tree ladder is priced, held and backed off like any other ladder, at the throttle's tree prices
            const bool tree = srv.nr && srv.nr->fruc && srv.nr->frMp;
            const uint32_t top = tree ? 1u << (std::min)(kFrDepth, srv.nr->frInstMax) : 0;
            uint32_t above = 0;
            if (tree && gridLock && frStepUp && panelHzMode > 1.0)
            {
                uint32_t n = 2;
                while (n <= top && n <= userCap * 1.03)
                    n *= 2;
                if (n <= top && userCap > n / 2 * 1.03)
                    above = n;
            }
            // Fit to the GPU off: the user's target is reached whatever it costs (Live may then skip source frames),
            // so the tree runs the first node that reaches it, the node above or the target's own, within the slots
            // the session got: no price, step, hold band or back-off (those are the GPU fit's)
            if (tree && g_noGpuFit)
            {
                uint32_t m = 0;
                for (uint32_t n = 2; n <= top && n <= userCap * 1.03; n *= 2)
                    m = n;
                if (above)
                    m = above;
                while (m > ceiling)
                    m /= 2;
                return gridLock && m >= 2 ? m : 0;
            }
            // a ladder within the slots (the tree: its nodes) and the user's target, or the node above it
            auto slotOk = [&](uint32_t m) {
                return m >= 2 && m <= ceiling && (m == above || m <= userCap * 1.03) &&
                       (!tree || ((m & (m - 1)) == 0 && m <= top));
            };
            auto fits = [&](uint32_t m) { return slotOk(m) && (m == above || m - 1 <= r * 1.03); };
            // the ladder above, priced at this lock's readings times the learned excess
            const double pre = srv.nr ? srv.nr->gIntUs.load() / 1000.0 : 0.0;
            const double tw = srv.nr ? srv.nr->gTweenUs.load() / 1000.0 : 0.0;
            // the tree's prices (xqThrottle's): a call, a slot's store, AMD FSR 4's x2 group
            const double fc = tree ? srv.nr->gFrCallUs.load() / 1000.0 : 0.0;
            const double fs = tree ? srv.nr->gFrStoreUs.load() / 1000.0 : 0.0;
            const double fb = tree ? srv.nr->gFrBaseUs.load() / 1000.0 : 0.0;
            const bool fsr = tree && srv.nr->fsrfg;
            const bool timed = tree ? fc > 0.0 && (fsr ? fb > 0.0 : pre > 0.0) : tw > 0.0;
            // the faster of the two period averages: a source-rate rise tightens the budget within a few pairs
            const double period = (std::min)(lockDtMs, emaDt);
            const double budget = period * thrBudget;
            // Smooth Motion on the pipelined route: M stores + the pipelined calls
            auto frucPipedCost = [&](uint32_t m) {
                double pm, pw;
                frucPipePrices(fc, pm, pw);
                return pre + m * fs + frucPipeCalls(m, pm, pw);
            };
            auto cost = [&](uint32_t m) {
                if (!tree)
                    return pre + (m - 1) * tw;
                // AMD FSR 4: an x2 group's whole time + (M - 2) calls and stores; Smooth Motion: (M - 1) calls + M
                // stores, or M stores + the pipelined calls; either pipelined (FSR 4 at any ladder, Smooth Motion x4
                // and deeper) on the route the sender pipelines
                if (fsr)
                {
                    double pp, po, p1;
                    const bool piped = fsrfgPipeOk(*srv.nr) && fsrfgPipeParts(pp, po, p1) &&
                                       (fsrfgPipeForce || (fsrfgPipeOn && frucPipeWant(m)));
                    return piped ? fsrfgPipeCost(m, fc, fs) : fb + (m - 2.0) * (fc + fs);
                }
                const bool piped = m >= 4 && (frucPipeForce || (frucPipeOn && thrMd > 0.0 && frucPipeWant(m)));
                return piped ? frucPipedCost(m) : pre + m * fs + (m - 1.0) * fc;
            };
            // a lock holds within slotOk and the hold band (kUpHold, under kUpPeriod of the period), never within
            // kUpSettleMs of its lock; before a group is timed the throttle's r stands in for the price (the node
            // above holds unpriced)
            auto holds = [&](uint32_t m) {
                if (!slotOk(m))
                    return false;
                if (!timed)
                    return m == above || m - 1 <= r * 1.03;
                if (cost(m) <= (std::min)(budget * kUpHold, period * kUpPeriod) || nowMs() - lockT < kUpSettleMs)
                {
                    upOverT = 0;
                    return true;
                }
                if (upOverT <= 0)
                    upOverT = nowMs();
                return nowMs() - upOverT < kUpOverMs; // over the band: the fall waits for it to last
            };
            // the learned price admits a ladder, or the default price once the back-off since a failed step is over;
            // a fallen ladder (and any above it) waits for that back-off whatever the price says
            const bool probe = upFailT > 0 && nowMs() - upFailT >= upBackoffMs;
            // the price a ladder is admitted at: the excess it showed itself when it last ran as a step-up, at least
            // the current one
            auto kFor = [&](uint32_t m) {
                const auto it = upKAt.find(m);
                return it == upKAt.end() ? upK : (std::max)(upK, it->second);
            };
            // a probe's price: halfway from the ladder's learned excess to the default
            auto kProbe = [&](uint32_t m) { return (std::max)(kUpKDefault, (kUpKDefault + kFor(m)) * 0.5); };
            auto afford = [&](uint32_t m) {
                if (upFailT > 0 && !probe && m >= upM)
                    return false;
                // an untimed tree (AMD FSR 4 learns its call price once an x4 or deeper group is priced): the
                // throttle's r within the target, the node above as one attempt
                if (!timed)
                    return tree && (m == above || m - 1 <= r * 0.97);
                return cost(m) * kFor(m) <= budget || (probe && cost(m) * kProbe(m) <= budget);
            };
            // a deep try's admission (kDeepTry): a pipelined Smooth Motion ladder within the slots and the target (the
            // throttle's r stands aside: it comes off the same shallower prices), past the shared back-off like afford
            // and past its own ladder's
            auto deepOk = [&](uint32_t m) {
                if (!tree || fsr || !timed || !frucPipeOn || thrMd <= 0.0 || m < 4 || !slotOk(m) ||
                    (upFailT > 0 && !probe && m >= upM))
                    return false;
                const auto f = deepFailAt.find(m);
                if (f != deepFailAt.end() && nowMs() - f->second < deepBackoff[m])
                    return false;
                return frucPipedCost(m) * kFor(m) <= budget * kDeepTry;
            };
            // the stepped-up lock's live price: what it costs now over the prediction that admitted it (re-read on
            // every settled pair and at its fall, so a sag raises it and its end lowers it again), never under the
            // default, kept for its ladder
            auto learn = [&]() {
                if (upPred > 0 && upM == lockLogged && timed)
                    upKAt[lockLogged] = upK = (std::max)(kUpKDefault, cost(lockLogged) / upPred);
            };
            if (lockLogged && holds(lockLogged))
            {
                // a step up waits for settled readings (kUpSettleMs since the lock: the group times are an EMA, and a
                // lock's first groups still carry the ladder before it)
                const bool settled = nowMs() - lockT >= kUpSettleMs;
                if (settled)
                {
                    learn();
                    if (deepLock == lockLogged && !deepSettled)
                    {
                        // the try's own prices hold it: the route rule takes over
                        deepSettled = true;
                        deepTry = 0;
                        LOG("ladder price: x%u costs %.1f ms, %.2f x the %.1f ms it was tried at\n", lockLogged,
                            cost(lockLogged), cost(lockLogged) / deepPred, deepPred);
                    }
                    if (!upSettled && upPred > 0 && upM == lockLogged)
                    {
                        upSettled = true;
                        LOG("ladder price: x%u costs %.1f ms, %.2f x the %.1f ms its step-up priced\n", lockLogged,
                            cost(lockLogged), cost(lockLogged) / upPred, upPred);
                    }
                }
                if (upFailT > 0 && lockLogged == upM && nowMs() - lockT >= kUpResetMs)
                {
                    upFailT = 0; // the ladder the back-off guards has held: the next failure starts the ladder over
                    upBackoffMs = kUpBackoff0;
                }
                if (deepLock == lockLogged && nowMs() - lockT >= kUpResetMs)
                {
                    deepBackoff.erase(deepLock); // the tried ladder has held: a later fall is any lock's
                    deepFailAt.erase(deepLock);
                    deepLock = 0;
                }
                const uint32_t next = tree ? lockLogged * 2 : lockLogged + 1;
                if (!(settled && fits(next) && afford(next)))
                {
                    // a deep try: the next node, from x4 or below
                    const uint32_t dt = settled && tree && lockLogged <= kDeepFromMax ? next : 0;
                    if (dt && deepOk(dt))
                    {
                        deepPred = frucPipedCost(dt);
                        deepTry = deepLock = dt;
                        deepSettled = false;
                        LOG("ladder price: x%u tried at %.1f ms, %.2f x the budget at x%u's pipelined prices\n", dt,
                            deepPred, deepPred / budget, lockLogged);
                        return dt;
                    }
                    // a refused probe of the ladder that fell: its excess relaxes to the probe's price, the next look
                    // waits for the doubled back-off
                    if (probe && timed && next == upM && upKAt.count(upM))
                    {
                        upKAt[upM] = kProbe(upM);
                        upBackoffMs = (std::min)(upBackoffMs * 2.0, kUpBackoffMax);
                        upFailT = nowMs();
                    }
                    return lockLogged;
                }
                // a step up: the largest ladder the price admits up to ceil(0.97 r) (the tree: up to its top node), the
                // prediction it ran on kept (an untimed tree's: one node, unpriced)
                uint32_t up = tree ? top : (std::max)(next, (uint32_t)ceil(r * 0.97));
                while (up > next && !(timed && fits(up) && afford(up)))
                    up = tree ? up / 2 : up - 1;
                upPred = timed ? cost(up) : 0.0;
                upM = up;
                upSettled = false;
                deepTry = deepLock = 0;
                return up;
            }
            if (lockLogged && lockLogged == deepLock)
            {
                // the tried ladder falls: its own back-off, the step-ups' left as they were
                upOverT = 0;
                const double c = cost(lockLogged), pred = deepPred, s = deepFall(c);
                LOG("ladder price: x%u falls at %.1f ms (%.2f x the price it was tried at); tried again in %.0f s\n",
                    lockLogged, c, c / pred, s);
            }
            else if (lockLogged)
            {
                learn(); // the lock falls: what it costs now is its price
                upOverT = 0;
                deepTry = 0;
                if (lockLogged == upM)
                {
                    // a stepped-up ladder falls: admitted again only after the back-off
                    upBackoffMs = upFailT > 0 ? (std::min)(upBackoffMs * 2.0, kUpBackoffMax) : kUpBackoff0;
                    upFailT = nowMs();
                    if (upPred > 0)
                        LOG("ladder price: x%u falls at %.1f ms (%.2f x its step-up's price); re-tried in %.0f s\n",
                            lockLogged, cost(lockLogged), cost(lockLogged) / upPred, upBackoffMs / 1000.0);
                    else
                        LOG("ladder price: x%u falls (an unpriced step-up); re-tried in %.0f s\n", lockLogged,
                            upBackoffMs / 1000.0);
                }
            }
            if (tree)
            {
                // the largest node the throttle's r affords within the target; with none, the node above (x2) at its
                // price, kUpSettleMs after the last lock
                uint32_t m = 0;
                for (uint32_t n = 2; n <= top; n *= 2)
                    if (n != above && fits(n) && n - 1 <= r * 0.97)
                        m = n;
                if (!m && above == 2 && nowMs() - lockT >= kUpSettleMs && afford(2))
                {
                    upPred = timed ? cost(2) : 0.0;
                    upM = 2;
                    upSettled = false;
                    deepTry = deepLock = 0;
                    return 2;
                }
                return gridLock ? m : 0;
            }
            const uint32_t m = (uint32_t)ceil(r * 0.97);
            return gridLock && fits(m) ? m : 0;
        };
        int sendLag = 0;       // the last sent group's content lag in source periods (its pend entry takes it)
        uint32_t sendHide = 0; // the last sent group's hidden slots (a floor re-measure's tween: 1), never presented
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
            // a 1.5-step error snaps on real cadence changes (drops, seeks, pauses). The same
            // clock paces the presents in every mode (sendGridMs).
            sendHide = 0;
            const int64_t step100 = (int64_t)(emaDt * 10000.0);
            int64_t paceTs = sentTs;
            if (emaDt > 0)
            {
                if (!smoothTs || llabs(sentTs - (smoothTs + step100)) > step100 * 3 / 2)
                    smoothTs = sentTs;
                else
                {
                    smoothTs += step100;
                    smoothTs += (sentTs - smoothTs) / 8;
                }
                paceTs = smoothTs;
                if (adaptTarget > 0)
                    sentTs = smoothTs;
            }
            ++shmSeq;
            cap.signalFence(shmSeq);
            // one frac per slot (gen tweens + the real frame); sized from the runtime gen,
            // so no fixed message capacity caps the target
            uint8_t* msg = msgBuf.data();
            float* fr = (float*)(msg + 4);
            uint32_t nfr = 0;
            bool pipeThis = false;
            if (shmSeq == 1 || g_fgOver)
            {
                fr[nfr++] = 1.0f; // very first frame: nothing to interpolate yet (DLSS-G over echo: every frame)
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
            else if (adaptTarget > 0 && emaDt > 0 && (thrFloor || effTarget * emaDt <= 1050.0))
            {
                // the throttle's floor, or a target at the source's rate: the real frame alone (the grid would put
                // one tween per pair at a fixed phase in its place, a model call for no added frame). A lock ends
                // here, not at the floor's re-time seconds later on readings frozen since (a stepped-up ladder that
                // drove the throttle to the floor fell: its back-off starts now)
                if (lockLogged)
                {
                    if (lockLogged == deepLock)
                    {
                        const double s = deepFall(0.0);
                        LOG("ladder price: x%u falls at the floor; tried again in %.0f s\n", lockLogged, s);
                    }
                    else if (lockLogged == upM && upPred > 0)
                    {
                        upBackoffMs = upFailT > 0 ? (std::min)(upBackoffMs * 2.0, kUpBackoffMax) : kUpBackoff0;
                        upFailT = nowMs();
                        LOG("ladder price: x%u falls at the floor; re-tried in %.0f s\n", lockLogged,
                            upBackoffMs / 1000.0);
                    }
                    upOverT = 0;
                    deepTry = 0;
                    LOG("present grid: unlocked (target %.1f, source %.3f fps)\n", effTarget, 1000.0 / lockDtMs);
                    lockLogged = 0;
                }
                if (nodeHz > 0)
                {
                    nodeHz = 0;
                    refloor();
                }
                fr[nfr++] = 1.0f;
                nextGrid = 0;
                if (thrProbing && nfr < srv.shmSlots)
                {
                    // the floor re-measure's tween: the host computes and stores it after the real frame, for its GPU
                    // time alone; its slot is never presented (XqGroup::show)
                    fr[nfr++] = 0.5f;
                    sendHide = 1;
                }
            }
            else if (const uint32_t m = adaptTarget > 0 && emaDt > 0 ? lockedMultiple(effTarget) : 0)
            {
                // M's ladder (k / M + the real frame as it is, see lockedMultiple); on FRUC's midpoint tree M is a
                // power of two, so every k / M is an exact node
                if (m != lockLogged)
                {
                    LOG("present grid: locked to x%u (target %.1f, source %.3f fps)%s\n", m, effTarget,
                        1000.0 / lockDtMs,
                        m * 1000.0 / lockDtMs > adaptTarget * 1.03 ? ", the node above the target" : "");
                    lockLogged = m;
                    lockT = nowMs();
                }
                {
                    // the node's rate at the faster of the two source clocks, so the floor never outruns its slots
                    const double hz = m * 1000.0 / (std::min)(emaDt, lockDtMs);
                    const double want = hz > adaptTarget * 1.03 ? hz : 0.0;
                    if (fabs(want - nodeHz) > nodeHz * 0.01 + 0.5)
                    {
                        nodeHz = want;
                        refloor();
                    }
                }
                for (uint32_t k = 1; k < m; k++)
                    fr[nfr++] = (float)k / (float)m;
                fr[nfr++] = 1.0f;
                tmInPair += m;
                nextGrid = 0;
                // fruc-pipe's route for this ladder (see frucPipeOn; AMD FSR 4's from x2 up)
                const uint32_t pipeMin = srv.nr && srv.nr->fsrfg ? 2 : 4;
                if (srv.nr && srv.nr->fruc && srv.nr->frMp &&
                    (srv.nr->fsrfg ? fsrfgPipeForce && fsrfgPipeOk(*srv.nr) : frucPipeForce) && m >= pipeMin)
                    pipeThis = true;
                else if (srv.nr && srv.nr->fruc && srv.nr->frMp && m >= pipeMin &&
                         (srv.nr->fsrfg
                              ? (thrMd > 0 || thrMp > 0 || noFitPriced()) && fsrfgPipeOn && fsrfgPipeOk(*srv.nr)
                              : (thrMd > 0 || noFitPriced()) && frucPipeOn))
                {
                    // a deep try runs pipelined from its first group, the switch dwell aside (it holds a ladder's
                    // route; this is a new ladder)
                    bool want = frucPipeWant(m) || m == deepTry;
                    // Fit to the GPU off: a node neither route holds runs on the one that affords more of it
                    if (g_noGpuFit && !want && m - 1 > frucAffD * 1.03 && frucAffP > frucAffD)
                        want = true;
                    pipeThis = want != frucPiped && nowMs() - frucPipeT < 2000.0 && m != deepTry ? frucPiped : want;
                }
            }
            else if (adaptTarget > 0)
            {
                if (lockLogged)
                {
                    LOG("present grid: unlocked (target %.1f, source %.3f fps)\n", effTarget, 1000.0 / lockDtMs);
                    lockLogged = 0;
                }
                if (nodeHz > 0)
                {
                    nodeHz = 0;
                    refloor();
                }
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
                fr[nfr++] = 1.0f; // the real frame
                pipeThis = srv.nr && srv.nr->fruc && srv.nr->frMp &&
                           (srv.nr->fsrfg ? genFrames >= 1 && fsrfgPipeForce && fsrfgPipeOk(*srv.nr)
                                          : genFrames >= 3 && frucPipeForce);
            }
            // each slot's place on the pacing clock: its fraction of the pair (on the adaptive route
            // exactly its grid point); a span the clock just snapped over falls back to the source's period
            const int64_t paceSpan =
                prevPaceTs > 0 && paceTs > prevPaceTs && (step100 <= 0 || paceTs - prevPaceTs < 3 * step100)
                    ? paceTs - prevPaceTs
                    : step100;
            sendGridMs.resize(nfr);
            for (uint32_t k = 0; k < nfr; ++k)
                sendGridMs[k] = ((double)paceTs - (1.0 - (double)fr[k]) * (double)paceSpan) / 10000.0;
            prevPaceTs = paceTs;
            if (pipeThis != frucPiped)
            {
                LOG("present grid: x%u %s (the direct tree affords %.1f tweens, the pipelined one %.1f)\n", nfr,
                    pipeThis ? "pipelined across pairs, the tweens one source period later" : "direct", frucAffD,
                    frucAffP);
                frucPiped = pipeThis;
                frucPipeT = nowMs();
            }
            sendLag = pipeThis ? 1 : 0;
            const uint32_t hdr = nfr | (pipeThis ? 0x80000000u : 0u); // the host's route bit (nativeGroup)
            memcpy(msg, &hdr, 4);
            if (!srv.writeFull(msg, 4 + 4 * nfr))
                return UINT32_MAX;
            prevSentTs = sentTs;
            return nfr;
        };
        // The 2s stats line, shared by both loops below. The GUI and test scripts parse
        // "live: A captured fps -> B presented fps (ratio R, ...) latency ~Nms". target = output frames
        // per source frame the route is asked for: a DLSS 4.5 plan's multiple while its frames are
        // presented (1 before), the fixed ladder, or the (throttled) adaptive target over the
        // measured capture rate.
        // panel = the refresh rate DWM actually ran this window (frame-statistics deltas;
        // 0.0 until two valid samples or when stats go disjoint, e.g. right after a mode
        // change), it feeds refreshQpc100 for prBunch; disp = presents DWM displayed per
        // second (PresentCount only advances for those), the vsync cross-check.
        auto logLiveStats = [&](double capFps, double outFps, double latAvg, double secs) {
            const double ratio = capFps > 0 ? outFps / capFps : 0.0;
            const double tgt = g_backend == BK_IDENTITY ? 1.0
                               : dgSess                 ? (cascShown ? (double)cascShown / cascKeep : 1.0)
                               : adaptTarget <= 0       ? (double)(genFrames + 1)
                                                        : (capFps > 0 ? effTarget / capFps : 0.0);
            const double prTot = host.prN ? host.prMsTot / host.prN : 0.0;
            const double prWait = host.prN ? host.prMsWait / host.prN : 0.0;
            const double prFlip = host.prN ? host.prMsFlip / host.prN : 0.0;
            double panelHz = 0.0, dispFps = 0.0;
            DXGI_FRAME_STATISTICS fs{};
            if (SUCCEEDED(host.sc->GetFrameStatistics(&fs)) && fs.SyncQPCTime.QuadPart)
            {
                if (host.fsValid && fs.SyncRefreshCount > host.fsPrev.SyncRefreshCount &&
                    fs.SyncQPCTime.QuadPart > host.fsPrev.SyncQPCTime.QuadPart)
                {
                    const double dq = (double)(fs.SyncQPCTime.QuadPart - host.fsPrev.SyncQPCTime.QuadPart);
                    panelHz = (double)(fs.SyncRefreshCount - host.fsPrev.SyncRefreshCount) * qpcFreq() / dq;
                    // presents the display took, less those that queued a frame again for the rest of its stay
                    dispFps =
                        (double)(fs.PresentCount - host.fsPrev.PresentCount - (host.dtRepeats - host.fsPrevRepeats)) *
                        qpcFreq() / dq;
                    if (panelHz > 1.0)
                        host.refreshQpc100 = (int64_t)(1e7 / panelHz);
                }
                host.fsPrev = fs;
                host.fsPrevRepeats = host.dtRepeats;
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
            // identical-pair passthrough: the native host's held count, APPENDED (the GUI and
            // test scripts read the fields before it).
            char stc[32] = "";
            if (g_staticHold)
                sprintf_s(stc, " static=%llu", (unsigned long long)g_staticHeld);
            // the host's GPU time of its latest groups with tweens (the throttle's pre / tw, timed in every live
            // session), APPENDED: a fixed-work A/B (--no-adapt, --no-gpu-fit) reads its same-work cost here
            char gpu[64] = "";
            if (srv.nr && srv.nr->gTweenUs.load())
                sprintf_s(gpu, " gpu pre %.2fms tween %.2fms", srv.nr->gIntUs.load() / 1000.0,
                          srv.nr->gTweenUs.load() / 1000.0);
            if (!rtssNamed && GetModuleHandleW(rtssModule))
            {
                rtssNamed = true;
                LOG("RTSS (RivaTuner Statistics Server) has hooked this process (%ls): a frame limit set in RTSS caps the "
                    "presented rate; if it stays below the target, set smv-live.exe's Application detection level to "
                    "None in RTSS\n",
                    rtssModule);
            }
            nativeVideoMemoryLate();
            LOG("live: %.1f captured fps -> %.1f presented fps (ratio %.2f, target %.1f) "
                "latency ~%.0fms present %.2fms (wait %.2f flip %.2f) drop %.1f "
                "panel %.1fHz dwm %.1fHz disp %.1f/s bunch %.1f/s%s%s\n",
                capFps, outFps, ratio, tgt, latAvg, prTot, prWait, prFlip, (cap.dropped - statDropBase) / secs, panelHz,
                dwmHz, dispFps, host.prBunch / secs, stc, gpu);
            return ratio;
        };
        // ================= CROSS-GROUP PRESENT QUEUE (every streaming server) =============
        // A group-serial loop (group k's tokens read in a nested loop, ALL of its presents
        // drained with waitQueue(), only then group k+1) presents an early-arriving pair's
        // first slot 20-28ms late, which was 100% of the measured 35-40ms hitches. The
        // per-group loop below serves only the direct routes (DLSS 4.5 without effects, identity).
        // This path keeps the protocol, the pacing formula and the floor rule and replaces
        // the structure: a reader thread owns the token stream, tokens land in ONE FIFO of
        // presentable slots that spans groups, and the barrier becomes a non-blocking
        // completed-value check on host.halfFence (GATE A) plus the capture-texture rule
        // that a new frame may only be drained once the last sent group has been OPENED by
        // the host (GATE B: its first token or end marker came back, so the host has acquired
        // the previous capture).
        const bool xqRoute = g_backend == BK_SERVER; // every model server runs the queue
        LOG("present queue: xq=%d (%s)\n", xqRoute ? 1 : 0, xqRoute ? "cross-group queue" : "direct present");
        std::unique_ptr<CascadeProbe> casc;
        const std::string cascMode = lkEnv("SMV_LIVE_CASCADE");
        // SMV_LIVE_DLSSG_KEEP=0: whole multiples only (no kept ring: the rule before the keep plan, an A/B)
        const bool cascKeepOn = lkEnv("SMV_LIVE_DLSSG_KEEP") != "0";
        // SMV_LIVE_DLSSG_SPLIT=0: an even multiple at or under the one-pass limit stays one pass (no 2x split, the A/B)
        const bool cascSplitOn = lkEnv("SMV_LIVE_DLSSG_SPLIT") != "0";
        // the plan: DLSS 4.5 on the direct route at the capture size or on the effects route at the presented size
        // (stage 1 takes the effects' output slots); a stage takes frames of 128 px a side up
        const uint32_t cascW = g_fgOver ? W : capW, cascH = g_fgOver ? H : capH;
        // SMV_LIVE_CASCADE=off: one pass only = a plan of one stage (there is no pass made by the session itself)
        const bool cascOneOnly = cascMode == "off";
        // a fixed multiple (--gen, or no --target): a plan of one stage at it, made at the first stats window
        const bool cascFixed = dgSess && (g_genExplicit || g_targetFps <= 0);
        const int cascFixedGen = g_fgOver ? genAsked : genFrames;
        if (cascFixed)
            cascNow = true;
        const bool cascAuto = (cascMode.empty() || cascOneOnly) && dgSess && cascW >= 128 && cascH >= 128;
        std::unique_ptr<CascadeProbe> cascPend; // the presented stage starting on cascTh (cascRes -1 = running, 1 = up,
                                                // 0 = failed)
        // every pass of a plan is a stage: casc = the presented one (the last), cascUp = stage 1 of two or three, fed
        // by the session, cascMid = stage 2 of three; a later stage reads the ring of the one before it
        std::unique_ptr<CascadeProbe> cascUp, cascPendUp, cascMid, cascPendMid;
        // the plan running has three stages (the shortfall rule's three-stage case). A kept plan never has: it is two
        // stages at most
        auto cascThree = [&]() { return cascMid != nullptr; };
        std::thread cascTh;
        std::atomic<int> cascRes{-1};
        // the one-pass limit: NGX's most frames a pair + 1 (6 on RTX 50, 2 on RTX 40), 6 while NGX has not said;
        // SMV_DLSSG_LIMIT=N (2..6) = another GPU's limit on this one (RTX 40's x4 = 2 x 2)
        auto liveDlssgLimit = [&]() {
            const std::string lv = lkEnv("SMV_DLSSG_LIMIT");
            if (!lv.empty())
                return (std::max)(2, (std::min)(6, atoi(lv.c_str())));
            return (host.maxGen ? (int)(std::min)(host.maxGen, 5u) : 5) + 1;
        };
        // The session reads that limit from NGX itself (ngxfg::maxGenOf), on a thread of
        // its own: NGX's start takes a second or two, and a stage that starts later finds it done. Asked at the
        // first plan, a stats window into the session (every other start on the device is over by then); no plan
        // starts before the answer
        struct NgxMax
        {
            std::thread th;
            std::atomic<int> most{-1};
            bool done = false;
            ~NgxMax()
            {
                if (th.joinable())
                    th.join();
            }
        } ngxMax;
        auto ngxMaxKnown = [&]() {
            if (ngxMax.done)
                return true;
            if (g_ngxMaxPre >= 0) // a spare host asked before this session came
            {
                ngxMax.done = true;
                if (g_ngxMaxPre > 0)
                {
                    host.maxGen = (uint32_t)g_ngxMaxPre;
                    LOG("DLSS-G: up to %ux in one pass on this GPU\n", host.maxGen + 1);
                }
                return true;
            }
            if (!ngxMax.th.joinable())
            {
                ngxMax.th =
                    std::thread([&ngxMax, dev = host.device.Get()] { ngxMax.most = (int)ngxfg::maxGenOf(dev); });
                return false;
            }
            if (ngxMax.most.load() < 0)
                return false;
            ngxMax.th.join();
            ngxMax.done = true;
            if (ngxMax.most.load() > 0)
            {
                host.maxGen = (uint32_t)ngxMax.most.load();
                LOG("DLSS-G: up to %ux in one pass on this GPU\n", host.maxGen + 1);
            }
            return true;
        };
        // the video memory a stage takes: ~250 MiB a megapixel + 60 for its frame generation, 19 frames (its ring
        // and its working copies), its depth and vectors (panvec::vecSize: a quarter of a frame each)
        auto cascPrice = [&]() -> uint64_t {
            const uint64_t frame = (uint64_t)cascW * cascH * 4;
            return ((uint64_t)60 << 20) + frame / 4 * 250 + 19 * frame +
                   2 * (uint64_t)panvec::vecSize(cascW) * panvec::vecSize(cascH) * 4;
        };
        // the plan lost, failed or not wanted any more: its stages end, and the captured frames are shown as they are
        // while the plan is made again, at the next stats window. lost = a stage ended or stopped presenting (not a
        // plan the session itself replaces): a plan lost four times in a row is not made again
        auto cascFallback = [&](const char* why, bool lost = false) {
            if (lost)
                cascTried = ++cascLost > 3;
            cascNow = !cascTried;
            LOG("DLSS 4.5's pass%s ended (%s): the captured frames are shown as they are %s\n", cascUp ? "es" : "", why,
                cascNow ? "until the plan is made again"
                        : "from here on (the plan ended this way four times in a row)");
            cascStepTick = 0;
            casc->kill();
            if (cascMid)
                cascMid->kill();
            if (cascUp)
                cascUp->kill();
            casc->stop();
            casc.reset();
            if (cascMid)
            {
                cascMid->stop();
                cascMid.reset();
            }
            if (cascUp)
            {
                cascUp->stop();
                cascUp.reset();
            }
            cascWarm = false;
            cascShown = 0;
            cascKeep = 1;
            cascLow = 0;
        };
        // the effects route's cascade feed: one output slot read back and handed to stage 1, so every frame it
        // generates from carries the effects
        auto cascFeedSlot = [&](uint32_t set, uint32_t i) -> bool {
            CascadeProbe* to = cascUp ? cascUp.get() : casc.get();
            return host.readSlot(host.outBuf.Get(), slotOffset(set, i) - srv.shmInBytes, (UINT64)srv.shmPitch * H,
                                 [&](const uint8_t* p) { to->feedRows(p, srv.shmPitch); });
        };
        // the presented stage's ring on screen: every frame it shared since the last pass, in order. On a flip queue
        // the frames are display-timed (Host::dtSet with the refreshes a presented frame at the plan's rate; the stage
        // then shares a source frame's frames together, and a frame of another source frame is a set's first).
        // false = a present failed
        auto cascPresentShared = [&]() -> bool {
            if (host.flipQueue)
                casc->setAtOnce(host.dtSet(cascShown && dtCapFps > 3.0 && panelHzMode > 1.0
                                               ? panelHzMode * cascKeep / (cascShown * dtCapFps)
                                               : 0.0,
                                           panelHzMode, cascShown * 16 + cascKeep));
            uint32_t s = 0;
            uint64_t fv = 0;
            while (casc->nextFrame(s, fv))
            {
                const uint32_t in = casc->hdr->slotInput[s];
                const bool first = in != dtInput;
                dtInput = in;
                if (host.rtRun)
                {
                    host.rtPubQpc = (int64_t)((double)casc->hdr->slotQpc[s] * 1e7 / (double)qpcFreq());
                    host.rtCapQpc = cap.lastFrameTs;
                }
                if (!host.presentRing(casc->ring[s].Get(), casc->shared.Get(), fv, first))
                {
                    LOG("presentRing failed\n");
                    return false;
                }
            }
            return true;
        };
        // every check covers every probe of a plan (the session feeds cascUp, presents casc)
        auto cascDead = [&]() { return casc->dead() || (cascUp && cascUp->dead()) || (cascMid && cascMid->dead()); };
        auto cascStalled = [&]() {
            const bool a = casc->stalled(), b = cascUp && cascUp->stalled(), c = cascMid && cascMid->stalled();
            return a || b || c;
        };
        auto cascReport = [&](double secs) {
            if (cascUp)
                cascUp->report(secs);
            if (cascMid)
                cascMid->report(secs);
            casc->report(secs);
        };
        // the plan the rules give a captured rate (cascPlan starts it, a shortfall steps a running plan to it in place):
        // the rung for target / captured (capped by cascMaxM after a shortfall), its passes m1, m2, m3 and its keep.
        // A rung whose rate lands on the panel's refresh runs like any other: on screen it presents the refresh in full
        // (12x of 30 fps and 15x of 24 fps on a 360 Hz panel: 360 of 360)
        auto cascChoose = [&](double capFps, int& m1, int& m2, int& m3, int& keep) -> int {
            const int lim = liveDlssgLimit();
            m1 = 0;
            m2 = 0;
            m3 = 1;
            keep = 1;
            if (cascFixed)
            {
                m1 = (std::max)(1, (std::min)(cascFixedGen + 1, lim));
                m2 = 1;
                return m1;
            }
            const double want = (double)g_targetFps / capFps;
            int M0 = dlssgSnap(want, lim, m1, m2, nullptr, &m3);
            if (cascMaxM && M0 > cascMaxM)
                M0 = dlssgAtMost(cascMaxM, lim, m1, m2, &m3);
            // no whole rung within 3 % of the want: a rung and a keep (dlssgKeepPlan), unless the keep plan is off, the
            // rung is above what a plan that fell short held (cascMaxM), or it takes two passes on a source above 100
            // fps (two passes cost more than such a source leaves time for: 5x then 5x of 119 fps held 2.5x to 2.9x of
            // its 4.17x)
            if (cascKeepOn && fabs((double)M0 - want) > 0.03 * want)
            {
                int k1 = 0, k2 = 0, kk = 1;
                const int R = dlssgKeepPlan(want, lim, k1, k2, kk);
                if (R && !(cascMaxM && R > cascMaxM) && !(k2 > 1 && capFps > 100.0))
                {
                    M0 = R;
                    m1 = k1;
                    m2 = k2;
                    m3 = 1;
                    keep = kk;
                }
            }
            // an even rung at or under the one-pass limit splits: 2x in a first stage, then M / 2 in a second (the
            // file route's order: against 1000 fps truth 2x then 3x beat one 6x pass by 0.8 dB on 30 ms pairs, 2x then
            // 2x one 4x pass by 0.6 on 28 ms pairs, while one 4x pass was better on 4 ms pairs: a source above 100 fps
            // keeps one pass)
            if (cascSplitOn && keep == 1 && m2 == 1 && m3 == 1 && M0 >= 4 && M0 <= lim && M0 % 2 == 0 &&
                capFps <= 100.0)
            {
                m1 = 2;
                m2 = M0 / 2;
            }
            // one pass only (SMV_LIVE_CASCADE=off): one stage, the target's whole multiple up to the one-pass limit
            if (cascOneOnly)
            {
                M0 = m1 = dlssgOnePassGen(want, lim - 1) + 1;
                m2 = m3 = keep = 1;
            }
            return M0;
        };
        // a plan's multiples applied to the RUNNING stages when each takes an ask (Stage1Hdr version 2): each stage
        // switches its multiple at its next input (DLSS-G's frame count changes in place, the frames byte-identical
        // to a fresh run at that count), so the session keeps presenting instead of the captured frames for the time
        // new stages take to start. A plan of fewer passes than run gives the later ones 1x (version 3: DLSS-G off
        // there, the frames passed through). SMV_LIVE_DLSSG_INPLACE=0 = the restart. False = not applicable (the
        // caller restarts)
        const bool cascInPlaceOn = lkEnv("SMV_LIVE_DLSSG_INPLACE") != "0";
        // the stages running: stage 1 cascUp (two or three), stage 2 cascMid (three), the presented casc
        auto cascRunning = [&]() { return cascMid ? 3 : cascUp ? 2 : 1; };
        auto cascSwitch = [&](int M, int m1, int m2, int m3, int keep, const char* why) -> bool {
            const auto inPl = [](const std::unique_ptr<CascadeProbe>& p) { return !p || p->inPlace; };
            const auto pass = [](const std::unique_ptr<CascadeProbe>& p) { return !p || p->passOk; };
            if (!cascInPlaceOn || !casc || !inPl(casc) || !inPl(cascUp) || !inPl(cascMid))
                return false;
            const int passes = m3 > 1 ? 3 : m2 > 1 ? 2 : 1;
            const int running = cascRunning();
            if (passes > running || (passes < running && !(pass(casc) && pass(cascUp) && pass(cascMid))) || M == cascM)
                return false;
            // the running stages' multiples: the plan's on the LAST stages, 1x before them (only the source frames
            // take the extra hop)
            int r[3] = {1, 1, 1};
            const int pm[3] = {m1, m2, m3};
            for (int i = 0; i < passes; i++)
                r[running - passes + i] = pm[i];
            const bool ok = running == 3   ? cascUp->askGen(r[0]) && cascMid->askGen(r[1]) && casc->askGen(r[2])
                            : running == 2 ? cascUp->askGen(r[0]) && casc->askGen(r[1])
                                           : casc->askGen(r[0]);
            if (!ok)
                return false;
            char passesTxt[64];
            if (running == 3)
                snprintf(passesTxt, sizeof passesTxt, "%dx, %dx, %dx", r[0], r[1], r[2]);
            else if (running == 2)
                snprintf(passesTxt, sizeof passesTxt, "%dx, %dx", r[0], r[1]);
            else
                snprintf(passesTxt, sizeof passesTxt, "%dx", r[0]);
            LOG("DLSS 4.5 steps to %.2fx in place (%s in the running helpers; %s): no restart\n", (double)M / keep,
                passesTxt, why);
            cascM = M;
            cascM1 = r[0];
            cascM2 = r[1];
            cascM3 = r[2];
            cascKeep = keep;
            casc->setKeep((uint32_t)keep);
            cascShown = M;
            cascLow = 0;
            cascStepTick = GetTickCount64();
            return true;
        };
        // a shortfall's next plan (cascChoose), when it is smaller, in place
        auto cascStepInPlace = [&](double capFps, const char* why) -> bool {
            int m1 = 0, m2 = 0, m3 = 1, keep = 1;
            const int M = cascChoose(capFps, m1, m2, m3, keep);
            return M < cascM && cascSwitch(M, m1, m2, m3, keep, why);
        };
        // the climb's next rung above `from` within the running passes: the first of at least 1.2 x from, else the
        // largest above it; at most the top, under the ceiling; 0 = none
        auto cascRungUp = [&](int from, int& m1, int& m2, int& m3) -> int {
            const int topM = cascCeilM ? (std::min)(cascTopM, cascCeilM - 1) : cascTopM;
            const auto rs = dlssgRungs(liveDlssgLimit(), cascRunning());
            const DlssgRung *step = nullptr, *largest = nullptr;
            for (const auto& r : rs)
            {
                if (r.m <= from || r.m > topM || (largest && largest->m == r.m))
                    continue;
                largest = &r;
                if (!step && r.m >= 1.2 * from)
                    step = &r;
            }
            const DlssgRung* pick = step ? step : largest;
            if (!pick)
                return 0;
            m1 = pick->a;
            m2 = pick->b;
            m3 = pick->c;
            return pick->m;
        };
        // the largest rung at most maxM within the running passes (the fewest passes for the same multiple); 0 = none
        auto cascRungAtMost = [&](int maxM, int& m1, int& m2, int& m3) -> int {
            const auto rs = dlssgRungs(liveDlssgLimit(), cascRunning());
            const DlssgRung* pick = nullptr;
            for (const auto& r : rs)
                if (r.m <= maxM && (!pick || r.m > pick->m))
                    pick = &r;
            if (!pick)
                return 0;
            m1 = pick->a;
            m2 = pick->b;
            m3 = pick->c;
            return pick->m;
        };
        // the climb's wait after a rung fell short: 15 s, doubling to 240 s
        auto cascClimbBackOff = [&](int fell) {
            cascCeilM = fell;
            cascClimbWait = cascClimbWait ? (std::min)(cascClimbWait * 2, (ULONGLONG)240000) : 15000;
            cascClimbTick = GetTickCount64() + cascClimbWait;
        };
        // three stages, or two above the one-pass limit (no keep: a keep is never capped by its cost), below cascHold of
        // their multiple three 2 s windows in a row: what the GPU affords is less, so they end and the next plan takes
        // the largest multiple at most what they held, with a tenth to spare (cascMaxM). Three stages that held less
        // than the two-pass top (the limit x the limit) leave that rung its own try: the third stage's cost leaves
        // with it (1 % of 2560x1440 from 21.5 fps: three passes held 28.5x, the next plan's 30x presented 634, 6x then
        // 6x 723)
        auto cascShortCheck = [&](double capFps, double outFps, double dropRate) {
            const int lim = liveDlssgLimit();
            // an in-place step every helper took within 3 s, or the restart (a refused ask, or one never applied)
            if (cascStepTick)
            {
                const auto held = [](const std::unique_ptr<CascadeProbe>& p) { return !p || p->genHeld(); };
                const auto refused = [](const std::unique_ptr<CascadeProbe>& p) { return p && p->askRefused(); };
                if (refused(casc) || refused(cascUp) || refused(cascMid) ||
                    (GetTickCount64() - cascStepTick > 3000 && !(held(casc) && held(cascUp) && held(cascMid))))
                {
                    cascStepTick = 0;
                    cascTried = false;
                    cascWantPrev = -1;
                    cascFallback("a helper kept its multiple");
                    return;
                }
                if (held(casc) && held(cascUp) && held(cascMid))
                    cascStepTick = 0;
            }
            const bool watched = cascUp || (cascKeep == 1 && cascM > lim);
            if (!watched || !cascShown || hidden || capFps < 0.5)
            {
                cascLow = 0;
                return;
            }
            // the climb (cascTopM): a rung holds when it presents cascHold of its multiple (the shortfall rule's own
            // bar) with at most 3 capture drops a second; a climb's try is judged on its second full window (the
            // first straddles the switch)
            if (cascTopM && cascKeep == 1 && !cascStepTick)
            {
                const bool good = outFps >= cascHold * cascShown * capFps && dropRate <= 3.0;
                if (cascClimbing)
                {
                    if (++cascClimbWins < 2)
                        return;
                    cascClimbing = false;
                    if (good)
                    {
                        cascHeld[0] = cascShown;
                        cascHeld[1] = cascM1;
                        cascHeld[2] = cascM2;
                        cascHeld[3] = cascM3;
                        cascClimbWait = 0;
                        cascHoldWins = 1; // the next window that holds climbs again
                        return;
                    }
                    cascClimbBackOff(cascShown);
                    char why[128];
                    snprintf(why, sizeof why,
                             "%.2fx of %dx, %.1f capture drops a second: the GPU's ceiling, the next try in %llu s",
                             outFps / capFps, cascShown, dropRate, (unsigned long long)(cascClimbWait / 1000));
                    if (cascHeld[0] && cascSwitch(cascHeld[0], cascHeld[1], cascHeld[2], cascHeld[3], 1, why))
                        return;
                    cascTopM = 0;
                    cascTried = false;
                    cascWantPrev = -1;
                    cascFallback(why);
                    return;
                }
                cascHoldWins = good ? cascHoldWins + 1 : 0;
                if (good && cascHoldWins >= 2)
                {
                    cascHeld[0] = cascShown;
                    cascHeld[1] = cascM1;
                    cascHeld[2] = cascM2;
                    cascHeld[3] = cascM3;
                    if (cascShown < cascTopM && GetTickCount64() >= cascClimbTick)
                    {
                        int m1 = 0, m2 = 0, m3 = 1;
                        int up = cascRungUp(cascShown, m1, m2, m3);
                        if (!up && cascCeilM)
                        {
                            cascCeilM = 0; // the wait is over: the ceiling is tried again
                            up = cascRungUp(cascShown, m1, m2, m3);
                        }
                        char why[96];
                        snprintf(why, sizeof why, "%.2fx held, the climb toward %dx", outFps / capFps, cascTopM);
                        if (up && cascSwitch(up, m1, m2, m3, 1, why))
                        {
                            cascClimbing = true;
                            cascClimbWins = 0;
                            cascHoldWins = 0;
                            return;
                        }
                    }
                }
            }
            // Fit to the GPU off: the plan keeps the target's rung whatever share of it the GPU shows (the target or
            // above; lowering it is the GPU fit's), one line a plan when it falls short
            if (g_noGpuFit)
            {
                if (!cascFitOffSaid && outFps < cascHold * cascShown / cascKeep * capFps)
                {
                    cascFitOffSaid = true;
                    LOG("DLSS 4.5: %.2fx of %.2fx shown, Fit to the GPU is off: the plan keeps its multiple\n",
                        outFps / capFps, (double)cascShown / cascKeep);
                }
                cascLow = 0;
                return;
            }
            cascLow = outFps < cascHold * cascShown / cascKeep * capFps ? cascLow + 1 : 0;
            if (cascLow < 3)
                return;
            char why[96];
            snprintf(why, sizeof(why), "%.2fx of %.2fx, what the GPU affords is less", outFps / capFps,
                     (double)cascShown / cascKeep);
            cascMaxM = (std::max)(2, (std::min)(cascShown - 1, (int)(outFps / capFps * cascKeep / 0.9)));
            if (cascTopM && cascKeep == 1) // the climb: the largest rung under what held, in place; the climb waits
            {
                int m1 = 0, m2 = 0, m3 = 1;
                const int fell = cascShown;
                const int down = cascRungAtMost(cascMaxM, m1, m2, m3);
                if (down && cascSwitch(down, m1, m2, m3, 1, why))
                {
                    cascClimbBackOff(fell);
                    cascHeld[0] = 0;
                    cascHoldWins = 0;
                    cascClimbing = false;
                    return;
                }
            }
            else
            {
                if (cascThree() && cascMaxM < lim * lim && !cascTwoTopShort)
                    cascMaxM = lim * lim;
                if (!cascThree() && cascShown == lim * lim)
                    cascTwoTopShort = true; // the two-pass top fell short too: it is not tried again
            }
            if (cascStepInPlace(capFps, why))
                return;
            cascTried = false;
            cascWantPrev = -1;
            cascFallback(why);
        };
        // a 2 s window's checks of a presenting plan: a stalled stage, then the shortfall rule
        auto cascWatch = [&](double capFps, double outFps, double dropRate) {
            if (!casc || !casc->present || hidden)
                return;
            if (cascLost && GetTickCount64() - cascUpTick > 60000)
                cascLost = 0; // this plan has presented for a minute: a later loss counts from one again
            if (cascStalled())
                cascFallback("it stopped presenting", true);
            else
                cascShortCheck(capFps, outFps, dropRate);
        };
        // a plan's start done (cascRes >= 0): its probes join the session warming (fed, not shown yet), or end
        auto cascTakePend = [&]() {
            cascTh.join();
            if (cascRes.load() == 1)
            {
                casc = std::move(cascPend);
                cascUp = std::move(cascPendUp);
                cascMid = std::move(cascPendMid);
                casc->present = false;
                cascWarm = true;
                // the climb's first rung, asked of the stages that started at the target's multiples
                if (cascTopM && cascFirst[0])
                {
                    const int a = cascFirst[1], b = cascFirst[2], c = cascFirst[3];
                    const bool ok = cascMid  ? cascUp->askGen(a) && cascMid->askGen(b) && casc->askGen(c)
                                    : cascUp ? cascUp->askGen(a) && casc->askGen(b)
                                             : casc->askGen(a);
                    if (ok)
                    {
                        LOG("DLSS 4.5 climbs from %dx: the GPU's ceiling is found in place, up to %dx\n", cascFirst[0],
                            cascTopM);
                        cascM = cascFirst[0];
                        cascM1 = a;
                        cascM2 = b;
                        cascM3 = c;
                        cascStepTick = GetTickCount64();
                    }
                    else
                        cascTopM = 0;
                }
                return;
            }
            cascPend->stop();
            cascPend.reset();
            if (cascPendMid)
            {
                cascPendMid->stop();
                cascPendMid.reset();
            }
            if (cascPendUp)
            {
                cascPendUp->stop();
                cascPendUp.reset();
            }
            // a plan that did not start counts as one lost (cascFallback)
            cascTried = ++cascLost > 3;
            cascNow = !cascTried;
            LOG("DLSS 4.5's pass did not start: the captured frames are shown as they are %s\n",
                cascNow ? "until the plan is made again"
                        : "from here on (the plan ended this way four times in a row)");
        };
        // the presented stage shared its first frame: the session presents its frames from now on
        auto cascTakeover = [&]() {
            cascWarm = false;
            cascUpTick = GetTickCount64();
            casc->present = true;
            casc->setKeep((uint32_t)cascKeep);
            casc->skipToNewest();
            cascStalled();
            cascShown = cascM;
            cascLow = 0;
            char more[64] = "";
            if (cascM3 > 1)
                snprintf(more, sizeof more, ", %dx in a third, %dx in a fourth", cascM2, cascM3);
            else if (cascM2 > 1)
                snprintf(more, sizeof more, ", %dx in a third", cascM2);
            char kept[48] = "every frame";
            if (cascKeep > 1)
                snprintf(kept, sizeof kept, "every %s of its frames", dlssgNth(cascKeep));
            LOG("DLSS 4.5 at %.2fx: %dx in a second stage%s, %s %s here, none made here\n", (double)cascM / cascKeep,
                cascM1, more, kept, host.lzOn ? "enlarged" : "presented");
        };
        // the plan, once a session, when the same multiple is derived twice in a row (either loop's 2 s window); the
        // captured frames are shown as they are meanwhile. A stage takes cascW x cascH frames in the back buffer's
        // format: B8G8R8A8, or R10G10B10A2 in HDR (the capture's pack on the direct route, the output slots on the
        // effects route). Past lim x lim three stages when the room holds them (else the best two); after three fell
        // short of their multiple, at most what they held (cascMaxM)
        auto cascPlan = [&](double capFps) {
            const bool limKnown = ngxMaxKnown(); // before the limit is read
            const int lim = liveDlssgLimit();
            int m1 = 0, m2 = 0, m3 = 1, keep = 1;
            const double want = (double)g_targetFps / capFps;
            const int M0 = cascChoose(capFps, m1, m2, m3, keep);
            int M = M0;
            if (limKnown && (M >= 2 || keep > 1) && ((M == cascWantPrev && keep == cascKeepPrev) || cascNow))
            {
                cascTried = true;
                cascNow = false;
                const uint64_t room = nativeVideoMemoryRoomOf(false);
                // the stages: stage 1 at m1, m2 > 1 a second on its ring, m3 > 1 a third on the second's
                bool two = m2 > 1, three = m3 > 1;
                uint64_t need = cascPrice() * (1 + (two ? 1 : 0) + (three ? 1 : 0));
                if (three && room && need > room)
                {
                    LOG("DLSS 4.5: %dx (%dx, %dx, then %dx) needs ~%.0f MiB of video memory for its second and third "
                        "passes, %.0f MiB are free; two passes at most\n",
                        M, m1, m2, m3, need / 1048576.0, room / 1048576.0);
                    M = cascMaxM ? dlssgAtMost((std::min)(cascMaxM, lim * lim), lim, m1, m2)
                                 : dlssgSnap(want, lim, m1, m2);
                    m3 = 1;
                    two = m2 > 1;
                    three = false;
                    need = cascPrice() * (two ? 2 : 1);
                }
                // the largest plan the room holds is then one stage, the target's whole multiple up to the one-pass
                // limit
                if (room && need > room && two)
                {
                    const int one = dlssgOnePassGen(want, lim - 1) + 1;
                    LOG("DLSS 4.5: %.2fx needs ~%.0f MiB of video memory for its passes, %.0f MiB are free; one pass "
                        "at %dx\n",
                        (double)M / keep, need / 1048576.0, room / 1048576.0, one);
                    M = m1 = one;
                    m2 = m3 = keep = 1;
                    two = three = false;
                    need = cascPrice();
                }
                if (room && need > room)
                {
                    LOG("DLSS 4.5: %.2fx needs ~%.0f MiB of video memory, %.0f MiB are free: the captured frames are "
                        "shown as they are\n",
                        (double)M / keep, need / 1048576.0, room / 1048576.0);
                }
                else
                {
                    cascM = M;
                    cascM1 = m1;
                    cascM2 = m2;
                    cascM3 = m3;
                    cascKeep = keep;
                    // the climb: the first rung above one pass with the plan's passes, when the target lies past it and
                    // past the panel's refresh (a target at or under the refresh is a rate the user expects: its plan
                    // runs as it is, a shortfall steps down in place; a target past it asks for what the GPU affords).
                    // Finding what the GPU affords is the GPU fit's: with Fit to the GPU off the target's rung runs at
                    // once
                    cascTopM = cascCeilM = cascHoldWins = cascClimbWins = 0;
                    cascHeld[0] = cascFirst[0] = 0;
                    cascClimbing = false;
                    cascClimbWait = cascClimbTick = 0;
                    cascFitOffSaid = false;
                    const int P = m3 > 1 ? 3 : m2 > 1 ? 2 : 1;
                    const bool pastPanel = panelHzMode > 1.0 ? g_targetFps > 1.02 * panelHzMode : P == 3;
                    if (cascClimbOn && cascInPlaceOn && keep == 1 && pastPanel && !g_noGpuFit)
                    {
                        const auto rs = dlssgRungs(lim, P);
                        for (const auto& r : rs)
                            if ((r.c > 1 ? 3 : r.b > 1 ? 2 : 1) == P && r.m > lim)
                            {
                                if (r.m < M)
                                {
                                    cascTopM = M;
                                    cascFirst[0] = r.m;
                                    cascFirst[1] = r.a;
                                    cascFirst[2] = r.b;
                                    cascFirst[3] = r.c;
                                }
                                break;
                            }
                    }
                    char passes[64], kept[40] = "";
                    if (three)
                        snprintf(passes, sizeof passes, "%dx, then %dx, then %dx", m1, m2, m3);
                    else if (two)
                        snprintf(passes, sizeof passes, "%dx, then %dx", m1, m2);
                    else
                        snprintf(passes, sizeof passes, "%dx", m1);
                    if (keep > 1)
                        snprintf(kept, sizeof kept, ", every %s frame shown", dlssgNth(keep));
                    LOG("DLSS 4.5 plan: %.2fx for target %d fps at ~%.1f captured = %s in helpers%s, %s here "
                        "(~%.0f MiB more video memory); starting %d helper%s\n",
                        (double)M / keep, g_targetFps, capFps, passes, kept, host.lzOn ? "enlarged" : "presented",
                        need / 1048576.0, 1 + (two ? 1 : 0) + (three ? 1 : 0), two ? "s" : "");
                    cascPend = std::make_unique<CascadeProbe>();
                    cascPend->keepWant = (uint32_t)keep; // the presented stage: asked before its first input
                    if (two)
                        cascPendUp = std::make_unique<CascadeProbe>();
                    if (three)
                        cascPendMid = std::make_unique<CascadeProbe>();
                    cascRes = -1;
                    CascadeProbe *p = cascPend.get(), *u = cascPendUp.get(), *mid = cascPendMid.get();
                    ID3D12Device* dev = host.device.Get();
                    // BGRA frames (the effects route's slots, the captured frames, the back buffer), R10G10B10A2 in HDR
                    const bool hdr10 = g_fgOver ? g_hdr : cap.hdrPack, bgra = !g_hdr;
                    // the presented probe's pass: the last of the plan's stages
                    const int mLast = three ? m3 : two ? m2 : m1;
                    // every stage starts at once (each one's DLSS-G init side by side instead of one after another):
                    // the chain's tags first, so a later stage's --from names its upstream, whose ring it waits for;
                    // SMV_LIVE_DLSSG_SERIAL_START=1 = one after another
                    const bool serial = lkEnv("SMV_LIVE_DLSSG_SERIAL_START") == "1";
                    cascTh = std::thread(
                        [p, u, mid, dev, w = cascW, h = cascH, m1, m2, mLast, hdr10, bgra, serial, &cascRes] {
                            if (u)
                                u->prepare(nullptr);
                            if (mid)
                                mid->prepare(u);
                            p->prepare(mid ? mid : u);
                            bool okU = true, okM = true, okP = true;
                            auto runU = [&] { okU = !u || u->start(dev, w, h, m1 - 1, hdr10, true, bgra); };
                            auto runM = [&] { okM = !mid || mid->start(dev, w, h, m2 - 1, hdr10, true, bgra, u); };
                            auto runP = [&] { okP = p->start(dev, w, h, mLast - 1, hdr10, true, bgra, mid ? mid : u); };
                            if (serial)
                            {
                                runU();
                                if (okU)
                                    runM();
                                if (okU && okM)
                                    runP();
                            }
                            else
                            {
                                std::thread tu(runU), tm(runM);
                                runP();
                                tu.join();
                                tm.join();
                            }
                            cascRes = okU && okM && okP ? 1 : 0;
                        });
                }
            }
            cascWantPrev = M0;
            cascKeepPrev = keep;
        };
        if (xqRoute)
        {
            struct XqGroup
            {
                uint32_t seq, set, expect, nfr;
                double arrMs, spanMs;
                int64_t sentTs;
                ULONGLONG sentTick;
                bool opened;                // first slot token or end marker came back
                bool capRel;                // the server released the shared capture texture
                std::vector<double> gridMs; // each slot's place on the pacing clock (sendGridMs)
                int lag;                    // its content's lag in source periods (a pipelined group: 1)
                uint32_t show;              // slots presented: every slot but a floor re-measure's hidden tween
            };
            struct XqSlot
            {
                uint32_t set, idx, nfr, groupSeq;
                double target, arrMs;
                int64_t sentTs;
                bool lastOfGroup;
                int lag;
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
            const double kXqDropRate = 3.0; // drops/s tolerated before the budget shrinks
            // GRID PACING: a slot is presented at its own place on the pacing clock plus ONE delay for the
            // whole stream, the time the slots take to come back (raised at once when a slot would be late,
            // lowered by kXqDelayDecay of the slot interval a slot: each gap 0.5 % short, ~5 ms a second at
            // any rate), capped below two source periods so a half is always free again before its next
            // group (GATE A). SMV_LIVE_GRID_PACE=0 = the old targets: the slots spread at (idx + 1) /
            // (nfr + 1) of the period after the drain, a late one at once.
            wchar_t gpv[8]{};
            const bool gridPace = !(GetEnvironmentVariableW(L"SMV_LIVE_GRID_PACE", gpv, 8) && gpv[0] == L'0');
            const double kXqReadyMarginMs = 1.0, kXqDelayDecay = 0.005;
            double xqDelay = 0;
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
            // The capture probe runs on the frame pool's FrameArrived flag instead of calling TryGetNextFrame on every
            // iteration. The loop also WAKES on the arrival event, so a frame is drained as promptly as a probe on every
            // iteration would, and a hard 8ms safety net probes anyway so a lost event can never strand the session.
            // Newest-wins drain semantics are untouched: the flag only decides WHETHER to call drainNewest.
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
            // The group-close instant is ALWAYS a present time (the group's last slot); the end-marker branch carries
            // the present clock when the group had no slot left in the FIFO.
            double xqLastPresMs = 0;
            uint32_t lastNfr = 0; // slots the most recently sent group asked for
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
                            const int64_t k0 = phOn ? nowQpc100() : 0;
                            const DWORD w = WaitForMultipleObjects(3, hs, FALSE, (DWORD)rem + 2);
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
            // THE THROTTLE, one evaluation per 500 ms window as a group's last slot presents. The host times every
            // group on the GPU (gIntUs = the work before the slots of a group with tweens: the frame's own passes and
            // the pair's model work; gTweenUs = one tween with its store). A group must finish within thrBudget of the
            // source's interval for the next source frame to be taken, so the pair affords m = (budget - that work) /
            // one tween; the grid makes every presented frame a tween, so the target is m per source frame (capped by
            // the user's target and the ring's slots), and under one tween a pair it is the source's own rate: the
            // real frames alone (thrFloor, read by sendPairStream). The drops keep the budget honest
            // (other work on the GPU, a bridge's own context outside the timed stream): drops above kXqDropRate a
            // second, smoothed over ~2 s, take a tenth off it (not at the floor), 5 clean seconds give 2 points back. At the floor a
            // tween is re-timed after 4 s: each pair computes one after its real frame, never presented (sendPairStream's
            // hidden slot), over kThrProbeWins consecutive windows (the boosted first window of a power-capped GPU,
            // above), the gap doubling to 60 s while any window answers "the floor"; the target moves only once the
            // windows afford a tween. A pause or a static hold starts the window over; before any tween is timed the
            // start target stands.
            auto xqThrottle = [&](double tn) {
                if (g_noGpuFit || !(adaptTarget > 0 && !hidden && !nextIdleTick && emaDt > 0))
                {
                    thrWinT0 = 0;
                    thrCleanMs = 0;
                    thrDropBase = cap.dropped;
                    return;
                }
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
                thrDropEma = thrDropEma * 0.75 + dropRate * 0.25;
                if (thrProbing || tn - thrMoveT < 2000.0)
                    thrDropEma = 0;
                if (thrDropEma > kXqDropRate)
                {
                    thrCleanMs = 0;
                    thrDropEma = 0; // one cut per spell: the next needs ~2 s of fresh drops
                    // at the floor the pairs carry their real frame alone, so a smaller tween budget removes no work:
                    // drops there (a foreign load) leave the budget where the floor found it, so the ladder comes back
                    // at that share once they end instead of climbing 2 points every 5 clean seconds from 30 %
                    if (!thrFloor)
                        thrBudget = (std::max)(0.3, thrBudget * 0.9);
                }
                else if (dropRate > kXqDropRate)
                    thrCleanMs = 0;
                else if ((thrCleanMs += tn - thrWinT0) >= 5000.0)
                {
                    thrCleanMs = 0;
                    thrBudget = (std::min)(0.95, thrBudget + 0.02);
                }
                const double srcFps = 1000.0 / emaDt;
                const double pre = srv.nr ? srv.nr->gIntUs.load() / 1000.0 : 0.0;
                const double tw = srv.nr ? srv.nr->gTweenUs.load() / 1000.0 : 0.0;
                double next = effTarget, m = 0.0;
                // the midpoint tree (Smooth Motion, AMD FSR 4): a call (c) and a slot's store (s) apart, once the host
                // timed them. One mean tween time cannot price it: FSR 4 makes every real frame's level 1 before the
                // slots (in pre), so an x2 group's tween reads ~0.1 ms and an x4 group's ~9 ms at 1440p: one mean
                // would step the ladder up at x2 and back down at x4, over and over
                const double fc = srv.nr ? srv.nr->gFrCallUs.load() / 1000.0 : 0.0;
                const double fs = srv.nr ? srv.nr->gFrStoreUs.load() / 1000.0 : 0.0;
                const double fb = srv.nr ? srv.nr->gFrBaseUs.load() / 1000.0 : 0.0;
                const bool frPriced = srv.nr && srv.nr->fruc && srv.nr->frMp && fc > 0.0;
                // FSR 4 is priced from its first x2 groups: an untimed call takes the level-1 call's span (fsrfgCall)
                const double fcF = srv.nr && srv.nr->fsrfg ? fsrfgCall(fc) : fc;
                const bool frFsr = srv.nr && srv.nr->fruc && srv.nr->frMp && srv.nr->fsrfg && fcF > 0.0 && fb > 0.0;
                thrMd = thrMp = 0.0;
                if (pre > 0.0 && tw > 0.0)
                {
                    m = (emaDt * thrBudget - pre) / tw;
                    if (frFsr)
                    {
                        // FSR 4: an x2 pair's whole time (fb) + (M - 2) calls and stores more direct, its own cost
                        // pipelined (fsrfgPipeCost)
                        thrMd = 1.0 + (emaDt * thrBudget - fb) / (fcF + fs);
                        thrMp = fsrfgPipeOn && fsrfgPipeOk(*srv.nr) ? fsrfgPipeAfford(emaDt * thrBudget, fcF, fs) : 0.0;
                        m = thrMp >= 1.0 ? (std::max)(thrMd, thrMp) : thrMd;
                    }
                    else if (frPriced && !srv.nr->fsrfg)
                    {
                        // Smooth Motion: (M - 1) calls + M stores direct; M stores + the pipelined calls pipelined
                        const double B = emaDt * thrBudget - pre;
                        double pm, pw;
                        frucPipePrices(fc, pm, pw);
                        thrMd = (B - fs) / (fc + fs);
                        thrMp = frucPipeAfford(B, fs, pm, pw);
                        m = frucPipeOn && thrMp >= 3.0 ? (std::max)(thrMd, thrMp) : thrMd;
                    }
                    next = m >= 1.0 ? m * srcFps : srcFps;
                    const bool floor = m < 1.0;
                    thrFloor = floor;
                    if (thrProbing && !floor && ++thrProbeWins < kThrProbeWins)
                    {
                        // the window timed fresh tweens that fit: one more window of hidden tweens before the floor is
                        // left (the boosted first window, above); the target holds
                        next = effTarget;
                    }
                    else if (thrProbing)
                    {
                        thrProbing = false; // the windows timed fresh tweens: the model above read them
                        thrProbeGapMs = floor ? (std::min)(60000.0, thrProbeGapMs * 2.0) : 4000.0;
                        thrProbeLast = tn;
                    }
                    else if (!floor)
                    {
                        thrProbeGapMs = 4000.0;
                        thrProbeLast = 0;
                    }
                    else if (!thrProbeLast)
                        thrProbeLast = tn;
                    else if (tn - thrProbeLast >= thrProbeGapMs)
                    {
                        // the re-measure: the pairs carry a hidden tween each (sendPairStream) while the target and the
                        // floor's presentation hold
                        thrProbing = true;
                        thrProbeWins = 0;
                        LOG("throttle: re-timing a tween at the floor (computed after the real frame, not shown)\n");
                    }
                }
                const double ceilT = (srv.shmSlots > 1 ? srv.shmSlots - 1 : 1) * srcFps;
                if (next > ceilT)
                    next = ceilT;
                if (next > adaptTarget)
                    next = adaptTarget;
                if (next < 1.0)
                    next = 1.0;
                if (fabs(next - effTarget) > effTarget * 0.03)
                {
                    if (!thrFloor || next > effTarget)
                        thrMoveT = tn;
                    effTarget = next;
                    gridStep = 1e7 / effTarget;
                    refloor();
                    if (tn - thrLastLog > 2000.0)
                    {
                        thrLastLog = tn;
                        char frp[128] = "";
                        if (frFsr)
                            sprintf_s(
                                frp,
                                "; an AMD FSR 4 call %.1f ms%s + %.1f ms a slot over an x2 pair's %.1f ms, direct %.1f / "
                                "pipelined %.1f",
                                fcF, fc > 0.0 ? "" : " (its level-1 span, untimed)", fs, fb, thrMd, thrMp);
                        else if (frPriced && !srv.nr->fsrfg)
                            sprintf_s(frp,
                                      "; a Smooth Motion call %.1f ms + %.1f ms a slot, direct %.1f / pipelined %.1f",
                                      fc, fs, thrMd, thrMp);
                        LOG("throttle: target %.0f fps of %.0f (%.1f ms before the slots + %.1f ms a tween of the %.1f "
                            "ms between source frames at %.0f %%: %.1f tweens a pair%s; drops %.1f/s%s)\n",
                            effTarget, adaptTarget, pre, tw, emaDt, thrBudget * 100.0, m,
                            thrProbing ? ", re-timing a tween" : "", dropRate, frp);
                    }
                }
                thrWinT0 = tn;
                thrDropBase = cap.dropped;
            };
            for (;;)
            {
                if (phOn)
                {
                    phIter++;
                    phT = nowQpc100();
                }
                pumpMessages();
                applyHudWant();
                if (g_stopReq.load())
                {
                    LOG("stop requested, ending the session\n");
                    break;
                }
                // Whole screen: the captured monitor left the desktop (unplugged, the layout changed): exit 4, the app
                // restarts the session on the monitor its target window sits on now
                if (cap.closed || (!g_monitor && !IsWindow(target)))
                {
                    LOG(g_monitor ? "capture closed (the display layout changed), restarting on the current monitor\n"
                                  : "target window closed\n");
                    rc2 = g_monitor ? 4 : 7;
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
                    if (raw && raw - 1 >= g.nfr)
                    {
                        LOG("live server protocol error (slot index)\n");
                        rc2 = 1;
                        break;
                    }
                    // a floor re-measure's tween: it ran for its GPU time, its slot is never presented
                    const bool hiddenSlot = raw && raw - 1 >= g.show;
                    if (raw == 0 || (hiddenSlot && last))
                    {
                        // bare end marker (or a hidden slot's last token): closes the group (an empty group opens and
                        // closes here)
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
                            xqThrottle(xqLastPresMs > 0 ? xqLastPresMs : nowMs());
                        continue;
                    }
                    if (hiddenSlot)
                        continue;
                    const uint32_t idx = raw - 1;
                    double due = g.arrMs + g.spanMs * (idx + 1) / (g.expect + 1);
                    if (gridPace && idx < g.gridMs.size())
                    {
                        const double need = nowMs() - g.gridMs[idx] + kXqReadyMarginMs;
                        const double delayCap = 1.75 * (emaDt > 0 ? emaDt : g.spanMs);
                        const double decay = kXqDelayDecay * 1000.0 / (effTarget > 0 ? effTarget : 60.0);
                        // the cap holds on the way down too: a delay raised while the source ran slow (a paused or
                        // slowed player) would outlive the resume, keep ~200 ms of slots queued and the capture
                        // gates shut
                        xqDelay = (std::min)(delayCap, need > xqDelay ? need : (std::max)(need, xqDelay - decay));
                        due = g.gridMs[idx] + xqDelay;
                    }
                    XqSlot s{g.set, idx, g.nfr, g.seq, due, g.arrMs, g.sentTs, false, g.lag};
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
                    // DLSS-G over echo: a real frame goes to DLSS-G the moment it is ready, as on the direct route
                    // (DLSS-G paces its own output)
                    double dueMs = g_fgOver ? 0.0 : s.target;
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
                    // the effects route's cascade: stage 1 takes the slot (its frames are presented below); while it
                    // warms up it is fed beside the one-pass present
                    if (!(casc && casc->present) && !presentSlotFrom(s.set, s.idx))
                    {
                        LOG("presentFrame failed\n");
                        rc2 = 1;
                        break;
                    }
                    if (casc && !cascFeedSlot(s.set, s.idx))
                    {
                        LOG("slot readback failed\n");
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
                        const double latMs =
                            (nowQpc100() - s.sentTs) / 10000.0 + (srv.contentLag + s.lag) * emaDt + stampLagMs();
                        statLatSum += latMs;
                        statLatN++;
                        meter.addLat(nowQpc100(), latMs);
                    }
                    if (latOn && latRows.size() < latRows.capacity())
                    {
                        UINT pc = 0;
                        host.sc->GetLastPresentCount(&pc);
                        latRows.push_back({'P', pc, nowQpc100(), s.sentTs, (srv.contentLag + s.lag) * emaDt});
                        DXGI_FRAME_STATISTICS lfs{};
                        if (SUCCEEDED(host.sc->GetFrameStatistics(&lfs)) && lfs.SyncQPCTime.QuadPart &&
                            lfs.PresentCount != latFsPrev)
                        {
                            latFsPrev = lfs.PresentCount;
                            latRows.push_back({'S', lfs.PresentCount,
                                               (int64_t)((double)lfs.SyncQPCTime.QuadPart * 1e7 / qpcFreq()),
                                               (int64_t)host.refreshQpc100, (double)lfs.PresentRefreshCount,
                                               (double)lfs.SyncRefreshCount});
                        }
                    }
                    if (hitchOn)
                    {
                        if (hitchPrev > 0 && pNow - hitchPrev > 8.0)
                            LOG("[hitch] gap %.1fms | slot %u/%u | sinceArr %.1f | "
                                "late %.2f | slotWait %.1f | arrGap %.1f | ema %.1f | eff %.0f | delay %.1f\n",
                                pNow - hitchPrev, s.idx + 1, s.nfr, pNow - s.arrMs, pNow - dueMs, 0.0,
                                s.arrMs - hitchArrPrev, emaDt, effTarget, xqDelay);
                        hitchPrev = pNow;
                    }
                    const bool wasLast = s.lastOfGroup;
                    xqLastPresMs = pNow; // F5: the one group-close clock
                    fifo.pop_front();
                    if (wasLast)
                        xqThrottle(pNow);
                }
                if (rc2)
                    break;
                // the effects route's plan: its stages came up (fed until the presented one shares a frame) or did
                // not; a dead one = the plan made again; the presented stage's first shared frame = the takeover
                if (cascPend && cascRes.load() >= 0)
                    cascTakePend();
                if (casc && (casc->present || cascWarm) && cascDead())
                    cascFallback(casc->local ? "its stage ended" : "its process ended", true);
                else if (cascWarm && casc->hdr && casc->hdr->frames > 0 && idleSlotIdx >= 0)
                    cascTakeover();
                // the presented stage: every frame it shared since the last pass, in order
                if (casc && casc->present && !hidden && !cascPresentShared())
                {
                    rc2 = 1;
                    break;
                }
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
                        if (g_liveNrCuda)
                            g_liveNrReset.store(true); // DLSS 5: a new stream after the gap
                        if (!hidden && casc && casc->present)
                            casc->skipToNewest(); // stage 1 idled with the session: no stale pre-pause tail
                    }
                    if (!hidden && g_fill)
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
                }
                if (!host.park && !g_monitor && !hidden && !g_fill)
                    followTarget(target, host.hwnd, fb, cap.cropX, cap.cropY, hud, true);

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
                // skip the TryGetNextFrame call unless the pool signalled an arrival (or the 8ms
                // safety net is due). The flag is read and cleared only while the gates are open, so
                // an arrival that lands while they are shut is drained as soon as they open, exactly
                // as with a probe on every iteration.
                bool wantProbe = gateA && !halfBusy && gateB && gateC;
                if (wantProbe)
                {
                    const double tp = nowMs();
                    wantProbe = InterlockedExchange(&cap.arrived, 0) != 0 || lastProbeMs == 0 || tp - lastProbeMs > 8.0;
                    if (wantProbe)
                        lastProbeMs = tp;
                }
                phEnd(phGate);
                if (wantProbe)
                {
                    // the warm-up's static-source frame already sits in sharedTex: the first group
                    const int g2 = sendNewest ? 1 : gpuCap ? cap.latestFrameGpu() : cap.latestFrame(buf.data());
                    sendNewest = false;
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
                            const uint32_t show = nfr - sendHide; // a re-measure's hidden tween is never presented
                            pend.push_back(XqGroup{shmSeq, (uint32_t)(shmSeq % 2), show ? show : 1, nfr, arrMs, emaDt,
                                                   sentTs, sendTick, false, false, sendGridMs, sendLag, show});
                            hitchArrPrev = arrMs;
                            lastNfr = show ? show : 1;
                            if (tmOn)
                            {
                                tmEma += emaDt;
                                tmNfr += show;
                                tmN++;
                            }
                            const ULONGLONG prevProcTick = lastPresentTick;
                            lastPresentTick = GetTickCount64();
                            statCaptured++;
                            // the SOURCE's cadence, not our sends: a pipeline slower than 300 ms a group
                            // read a playing video as paused, and the hold then kept the throttle off
                            if (cap.lastArrTs ? cap.sourceCadence() : lastPresentTick - prevProcTick < 300)
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
                if (!hidden && (nextIdleTick > 0 ||
                                (cap.lastArrTs ? cap.sourceQuiet() : GetTickCount64() - lastPresentTick > 300)))
                {
                    const ULONGLONG now3 = GetTickCount64();
                    if (now3 - lastRefreshTick >= 1000)
                    {
                        lastRefreshTick = now3;
                        cap.requestRefresh();
                        if (nextIdleTick > 0)
                            nextIdleTick = nowMs() + idleStepMs;
                    }
                    // while the cascade presents, stage 1 keeps going on the capture's 500 ms keepalive, as on the direct
                    // route: no re-present here
                    if (idleSlotIdx >= 0 && !(casc && casc->present))
                    {
                        const double nowQ = nowMs();
                        if (nextIdleTick <= 0)
                        {
                            nextIdleTick = nowQ + idleStepMs;
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

                // the HUD: every 0.5 s over the last second (HudMeter)
                if (hud.hwnd && meter.due(nowQpc100()))
                {
                    UINT pc = 0;
                    host.sc->GetLastPresentCount(&pc);
                    pc -= host.dtRepeats;
                    double inF = 0, outF = 0, latM = 0;
                    // DLSS-G over echo: the figure plus one capture interval, as on the direct DLSS-G route
                    if (meter.read(nowQpc100(), cap, pc, stampLagMs() > 0 ? 1000.0 / stampLagMs() : 0.0, inF, outF,
                                   latM))
                    {
                        const double latShown = g_fgOver && inF > 0.5 ? latM + 1000.0 / inF : latM;
                        hud.update(inF, outF, latShown, g_serverBackend == L"echo" && !g_fgOver);
                        if (latOn && latRows.size() < latRows.capacity())
                            latRows.push_back({'H', 0, meter.lastTick, 0, inF, outF, latShown});
                    }
                }

                // ---- (vi) 2s stats tick (server route fields only)
                if (now - lastStatTick >= 2000)
                {
                    UINT c = 0;
                    host.sc->GetLastPresentCount(&c);
                    c -= host.dtRepeats; // a frame queued again for the rest of its stay is no frame more
                    const double secs = (now - lastStatTick) / 1000.0;
                    const double dropWin = (double)(cap.dropped - statDropBase);
                    const double capFps = (statCaptured + dropWin) / secs;
                    const double outFps = (c - statPresentBase) / secs;
                    dtCapFps = capFps;
                    const double latAvg = statLatN ? statLatSum / statLatN : 0.0;
                    logLiveStats(capFps, outFps, latAvg, secs);
                    if (casc)
                        cascReport(secs);
                    cascWatch(capFps, outFps, dropWin / secs);
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
                    statLatSum = 0;
                    statLatN = 0;
                    // DLSS-G with effects: the plan follows the target (two equal 2 s derivations in a row)
                    if (g_fgOver && capFps > 3.0 && idleSlotIdx >= 0 && !casc && !cascPend && cascAuto && !cascTried)
                        cascPlan(capFps);
                    if (g_noAdapt && !g_fgOver && g_targetFps > 0 && !g_genExplicit && capFps > 3.0)
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
                if (!fifo.empty() && nextDue > 0)
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
                // the follow poll (windowed): the loop wakes within 16 ms to see a move, and every 2 ms on the
                // high-resolution timer while the target moves (a move in the last 500 ms); the wait's own timeout
                // keeps the ~15.6 ms timer quantum here
                const bool following = !host.park && !g_monitor && !hidden && !g_fill;
                if (following && waitMs > 16)
                    waitMs = 16;
                if (casc && casc->present)
                    waitMs = 1; // stage 1's frames arrive at its rate (~7 ms apart at x6 of 24 fps)
                if (following && followTimer && GetTickCount64() < followUntil)
                {
                    LARGE_INTEGER due{};
                    due.QuadPart = -20000; // 2 ms, relative, in 100 ns units
                    SetWaitableTimer(followTimer, &due, 0, nullptr, nullptr, FALSE);
                    const HANDLE waits[2] = {cap.evt, followTimer};
                    WaitForMultipleObjects(2, waits, FALSE, waitMs);
                }
                else
                    WaitForSingleObject(cap.evt, waitMs);
                pumpMessages();
                applyHudWant();

                if (g_stopReq.load())
                {
                    LOG("stop requested, ending the session\n");
                    break;
                }
                // monitor mode: the target window only seeded the monitor choice; its lifetime and
                // focus are irrelevant (whole-screen smoothing keeps running until the hotkey / Stop);
                // the captured monitor leaving the desktop = exit 4, the app restarts on the current one
                if (cap.closed || (!g_monitor && !IsWindow(target)))
                {
                    LOG(g_monitor ? "capture closed (the display layout changed), restarting on the current monitor\n"
                                  : "target window closed\n");
                    rc2 = g_monitor ? 4 : 7;
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
                        if (!hidden && casc && casc->present)
                            casc->skipToNewest(); // stage 1 idled with the session: no stale pre-pause tail
                    }
                    if (!hidden && g_fill)
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
                }
                if (!host.park && !g_monitor && !hidden && !g_fill &&
                    followTarget(target, host.hwnd, fb, cap.cropX, cap.cropY, hud, true))
                    followUntil = GetTickCount64() + 500;

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
                    if (casc && casc->present)
                        (cascUp ? cascUp : casc)->feed(buf.data(), capBytes); // stage 1 takes it; presented below
                    else
                    {
                        if (!host.presentFrame(buf.data()))
                        {
                            LOG("presentFrame failed\n");
                            rc2 = 1;
                            break;
                        }
                        if (casc)
                            (cascUp ? cascUp : casc)->feed(buf.data(), capBytes);
                    }
                    // direct routes (dlssg/identity): latency of the frame just presented
                    const double latDirect = (nowQpc100() - cap.lastFrameTs) / 10000.0 + stampLagMs();
                    statLatSum += latDirect;
                    statLatN++;
                    meter.addLat(nowQpc100(), latDirect);
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
                // the plan's stages came up (fed until the presented one shares a frame) or did not
                if (cascPend && cascRes.load() >= 0)
                    cascTakePend();
                if (casc && (casc->present || cascWarm) && cascDead())
                    cascFallback(casc->local ? "its stage ended" : "its process ended", true);
                else if (cascWarm && casc->hdr && casc->hdr->frames > 0)
                    cascTakeover();
                // the presented stage: every frame it shared since the last pass, in order
                if (casc && casc->present && !hidden && !cascPresentShared())
                {
                    rc2 = 1;
                    break;
                }

                // the HUD every 0.5 s over the last second (HudMeter); a plan's frames show the figure plus one
                // capture interval (the interpolation's own delay: a stage waits for the next real frame)
                if (hud.hwnd && meter.due(nowQpc100()))
                {
                    UINT pc = 0;
                    host.sc->GetLastPresentCount(&pc);
                    pc -= host.dtRepeats;
                    double inF = 0, outF = 0, latM = 0;
                    if (meter.read(nowQpc100(), cap, pc, stampLagMs() > 0 ? 1000.0 / stampLagMs() : 0.0, inF, outF,
                                   latM))
                    {
                        const bool between = casc && casc->present;
                        const double latShown = between && inF > 0.5 ? latM + 1000.0 / inF : latM;
                        hud.update(inF, outF, latShown);
                        if (latOn && latRows.size() < latRows.capacity())
                            latRows.push_back({'H', 0, meter.lastTick, 0, inF, outF, latShown});
                    }
                }

                if (now - lastStatTick >= 2000)
                {
                    UINT c = 0;
                    host.sc->GetLastPresentCount(&c);
                    c -= host.dtRepeats; // a frame queued again for the rest of its stay is no frame more
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
                    dtCapFps = capFps;
                    // the capture->present latency of the frames this loop presents itself
                    const double latAvg = statLatN ? statLatSum / statLatN : 0.0;
                    logLiveStats(capFps, outFps, latAvg, secs);
                    if (casc)
                        cascReport(secs);
                    cascWatch(capFps, outFps, dropWin / secs);
                    {
                        uint64_t hn = 0;
                        for (int i = 0; i < Host::kHistN; i++)
                            hn += host.prHist[i];
                        if (hn > 20)
                        {
                            double avg = 0, l1 = 0, l01 = 0;
                            Host::histStats(host.prHist, hn, avg, l1, l01);
                            // NOT "live: " - test scripts read the LAST line with that exact
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
                    statLatSum = 0;
                    statLatN = 0;
                    // DLSS 4.5: the plan follows the fps target and the measured capture rate, made once the same
                    // multiple is derived twice in a row (2 s stats windows), so a momentary rate wobble starts none
                    if (dgSess && capFps > 3.0 && !casc && !cascPend && cascAuto && !cascTried)
                        cascPlan(capFps);
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
                    lastStatTick = now;
                    statPresentBase = c;
                    statCaptured = 0;
                }
            }

        running = 0;
        fillMouse.stop(); // hands the cursor and the pointer speed back before the overlay goes
        if (latOn && !latRows.empty())
        {
            FILE* lf = nullptr;
            if (_wfopen_s(&lf, latPath, L"w") == 0 && lf)
            {
                for (const LatRow& r : latRows)
                    fprintf(lf, "%c %llu %lld %lld %.4f %.4f %.4f\n", r.kind, (unsigned long long)r.a, (long long)r.b,
                            (long long)r.c, r.d, r.e, r.f);
                cap.dumpStamps(lf);
                fclose(lf);
            }
            LOG("latency trace: %zu rows\n", latRows.size());
        }
        host.rtStop();
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
        if (followTimer)
            CloseHandle(followTimer);
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
        if (cascPend) // a plan still starting: its READY wait ends at once
        {
            cascPend->abortStart = true;
            if (cascPendUp)
                cascPendUp->abortStart = true;
            if (cascPendMid)
                cascPendMid->abortStart = true;
            if (cascTh.joinable())
                cascTh.join();
            cascPend->stop();
            if (cascPendMid)
                cascPendMid->stop();
            if (cascPendUp)
                cascPendUp->stop();
        }
        if (casc)
            casc->stop(); // its reader queue lives on the host's device
        if (cascMid)
            cascMid->stop();
        if (cascUp)
            cascUp->stop();
        if (g_teardownTrace)
            LOG("teardown: capture stopped, shutting the host down\n");
        host.shutdown();
        if (g_teardownTrace)
            LOG("teardown: host down\n");
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
