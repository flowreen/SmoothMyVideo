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

// Live DLSS 4.5's multiple for w = the target / the source rate: one pass up to `lim` (numFramesToGenerateMax + 1),
// above it the full lim first and then the rest (the cascade below: stage 1 = dlssg2f --stage1, the last pass = the
// session's own DLSS-G; more frames made from the real ones), so the reachable multiples are 2..lim and lim x 2..lim
// (lim 6: 12, 18, .. 36), with m3 also lim x lim x 2..lim (72, 108, .. 216); the first of them that reaches w (within 3 %:
// a 23.976 fps source at 48 stays 2x), the largest when none does, x2 below 2. m1, m2 = the passes in order (m2 = 1 for one
// pass), *m3 = the third (1 for one or two passes); *tie = false (the snap table's format)
static int dlssgSnap(double w, int lim, int& m1, int& m2, bool* tie = nullptr, int* m3 = nullptr)
{
    const int passes = m3 ? 3 : 2;
    int full = 1, r = lim, k = passes;
    for (int n = 1; n <= passes; n++, full *= lim)
    {
        int f = 2;
        while (f < lim && full * f * 1.03 < w)
            f++;
        if (full * f * 1.03 >= w || n == passes)
        {
            k = n;
            r = f;
            break;
        }
    }
    m1 = k == 1 ? r : lim;
    m2 = k == 1 ? 1 : k == 2 ? r : lim;
    if (m3)
        *m3 = k == 3 ? r : 1;
    if (tie)
        *tie = false;
    return full * r;
}

// the largest of dlssgSnap's multiples (with m3 the three-pass ones too) at most cap, its passes in the same order; x2
// below 2 (a plan that fell short steps down to what the GPU held)
static int dlssgAtMost(int cap, int lim, int& m1, int& m2, int* m3 = nullptr)
{
    int best = 2, k = 1, r = 2;
    for (int n = 1, full = 1; n <= (m3 ? 3 : 2); n++, full *= lim)
        for (int f = 2; f <= lim; f++)
            if (full * f <= cap && full * f >= best)
            {
                best = full * f;
                k = n;
                r = f;
            }
    m1 = k == 1 ? r : lim;
    m2 = k == 1 ? 1 : k == 2 ? r : lim;
    if (m3)
        *m3 = k == 3 ? r : 1;
    return best;
}

// The one pass's generated frames for w (the first whole multiple reaching it within 3 %, as dlssgSnap), 1..capMax
static int dlssgOnePassGen(double w, int capMax)
{
    int g = 1;
    while (g < capMax && (g + 1) * 1.03 < w)
        g++;
    return g;
}

// The Live DLSS 4.5 cascade's first stage (SMV_LIVE_CASCADE=probe on the direct DLSS-G route: consume-only):
// dlssg2f.exe --stage1 in a job object that ends it with this process, fed every captured
// frame by a writer thread (a frame arriving while the last one is still being written is dropped, counted), and every
// present it shares read back on THIS process's D3D12 device: its queue waits the slot's fence value and copies the
// slot into a scratch texture. Nothing it reads is presented: the overlay keeps today's direct route.
struct Stage1HdrView // dlssg2f's Stage1Hdr (engine\dlssg\build_src\main.cpp), version 1
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
};

struct CascadeProbe
{
    HANDLE job = nullptr, hProc = nullptr, hIn = nullptr, hOut = nullptr, hErr = nullptr, hMap = nullptr;
    const Stage1HdrView* hdr = nullptr;
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
    // the share tag (live<pid>; a third stage's middle probe live<pid>b) and, for that probe, the probe it takes its
    // frames from (dlssg2f --from) instead of the session's feed
    std::wstring tag;
    const CascadeProbe* up = nullptr;

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
            return true;
        }
        return false;
    }

    // bgra: SDR frames in B8G8R8A8 order (the effects route's slots and back buffer), else R8G8B8A8; upstream = a
    // running probe whose ring this one takes as its input (its stage number 2, never fed by the session)
    bool start(ID3D12Device* dev, uint32_t w, uint32_t h, int gen, bool hdr10, bool presentMode, bool bgra,
               const CascadeProbe* upstream = nullptr)
    {
        present = presentMode;
        up = upstream;
        fw = w;
        fh = h;
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
        tag = L"live" + std::to_wstring(GetCurrentProcessId()) + (up ? L"b" : L"");
        const int stageNo = up ? 2 : 1;
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
        // READY within 20 s (Streamline's init + the DLSS-G feature), polled: a child that hangs or dies, or a session
        // that ends meanwhile (abortStart), never holds the caller
        std::string line;
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
        const std::wstring mn = L"smv_dg1_" + tag + L"_hdr";
        hMap = OpenFileMappingW(FILE_MAP_READ, FALSE, mn.c_str());
        hdr = hMap ? (const Stage1HdrView*)MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, sizeof(Stage1HdrView)) : nullptr;
        if (!hdr || hdr->magic != 0x31564D53 || hdr->version != 1 || hdr->ring > 16)
        {
            LOG("cascade probe: no stage-1 header\n");
            return false;
        }
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
        writer = std::thread([this] { writeLoop(); });
        nextSeq = hdr->frames;
        if (!present)
            reader = std::thread([this] { readLoop(); });
        LOG("cascade probe: stage %d up (%s), %ux%u x%d, its ring of %u opened on this device%s\n", stageNo,
            line.c_str(), w, h, gen + 1, n,
            up        ? ", from stage 1's ring, the session presents it"
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
        cv.notify_one();
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
        cv.notify_one();
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

    // stage 1 is gone: its process ended, its header says ended / failed, or its stdin broke
    bool dead() const
    {
        return (hProc && WaitForSingleObject(hProc, 0) == WAIT_OBJECT_0) || (hdr && hdr->status >= 2) ||
               writerFailed.load();
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

    // a fallback's first step: end the child at once (its job) and release any stage-2 queue wait on its fence
    void kill()
    {
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
        if (up)
            LOG("cascade probe 2: stage 2 shared %.1f/s, read %.1f/s; skipped %llu\n", (fr - repFrames) / secs,
                (c - repConsumed) / secs, (unsigned long long)skipped.load());
        else
            LOG("cascade probe: fed %.1f/s (dropped %llu), stage 1 shared %.1f/s, read %.1f/s; overrun %llu, skipped "
                "%llu\n",
                (f - repFed) / secs, (unsigned long long)(d - repDropped), (fr - repFrames) / secs,
                (c - repConsumed) / secs, (unsigned long long)overrun.load(), (unsigned long long)skipped.load());
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
        for (HANDLE x : {hProc, hOut, hErr, hMap, ev, job})
            if (x)
                CloseHandle(x);
        hProc = hOut = hErr = hMap = ev = job = nullptr;
    }
};

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
    wchar_t sized[256]{}; // the text the window was last sized for
    int textH = 0;        // that text's height as drawn, one line or more

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
        // \u2192 = the arrow, \u00D7 = the times sign; ESCAPED on purpose: a raw UTF-8 literal in this file compiles
        // as ANSI mojibake unless the build adds /utf-8 (it showed as "weird letters")
        // --no-hud-latency hides the latency segment only; the fps segment is governed by
        // --no-hud (whole readout). DLSS-G never reaches here with a latency to show (SL
        // paces internally and the caller passes its handoff figure), so that path is
        // unchanged either way.
        // the in rate with one decimal, a trailing .0 dropped (a 2 s window with one arrival = 0.5 fps), and the
        // measured multiplier, out over in (the live: line's ratio): a shown 0 never carries one
        wchar_t inTxt[16];
        swprintf_s(inTxt, L"%.1f", inFps);
        const size_t inLen = wcslen(inTxt);
        if (inLen > 2 && !wcscmp(inTxt + inLen - 2, L".0"))
            inTxt[inLen - 2] = 0;
        wchar_t mul[32] = L"";
        if (wcscmp(inTxt, L"0"))
            swprintf_s(mul, L" (%.2f\u00D7)", outFps / inFps);
        if (latMs > 0.5 && !g_noHudLat)
            swprintf_s(text, L"%ls fps \u2192 %.0f fps%ls   ~%.0f ms behind", inTxt, outFps, mul, latMs);
        else
            swprintf_s(text, L"%ls fps \u2192 %.0f fps%ls", inTxt, outFps, mul);
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
        RECT r{0, 0, room - 8, 0};
        HDC dc = GetDC(hwnd);
        HGDIOBJ of = SelectObject(dc, font);
        DrawTextW(dc, text, -1, &r, DT_LEFT | DT_WORDBREAK | DT_NOPREFIX | DT_CALCRECT);
        SelectObject(dc, of);
        ReleaseDC(hwnd, dc);
        textH = (int)(r.bottom - r.top);
        SetWindowPos(hwnd, nullptr, 0, 0, (int)(r.right - r.left) + 8, (std::max)(36, textH + 4),
                     SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
    }

    void paint()
    {
        if (!hwnd)
            return;
        fit();
        HDC dc = GetDC(hwnd);
        RECT r;
        GetClientRect(hwnd, &r);
        FillRect(dc, &r, (HBRUSH)GetStockObject(BLACK_BRUSH));
        HGDIOBJ of = SelectObject(dc, font);
        SetBkMode(dc, TRANSPARENT);
        SetTextColor(dc, RGB(255, 255, 255)); // full white: easier to read than green
        r.left += 4;
        r.top = (r.bottom - textH) / 2; // centred: one line in the window's 36 px, more lines in their own height
        DrawTextW(dc, text, -1, &r, DT_LEFT | DT_WORDBREAK | DT_NOPREFIX);
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
    // the plugins shipped beside the exe, never a newer one Streamline downloads into ProgramData: a downloaded
    // copy changes behaviour untested, and its hashed file name escapes the focus shim (it patches sl.* modules)
    pref.flags &= ~(sl::PreferenceFlags::eAllowOTA | sl::PreferenceFlags::eLoadDownloadedPlugins);
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
// Waits until the target's client size holds still. true = it came back to the captured size (a
// transient: the restore animation after a minimize hands out a frame of another size), the
// session continues with the overlay shown again; false = a real resize, the caller restarts.
static bool resizeSettle(HWND target, Host& host, const Capture& cap, Hud& hud, bool hidden)
{
    const DWORD settleMs = 1000;    // how long the client size must hold still
    ShowWindow(host.hwnd, SW_HIDE); // the old-size frames no longer cover the window
    if (hud.hwnd)
    {
        _snwprintf_s(hud.text, _TRUNCATE, L"SMV Live: Loading %ls...", loadingWhat().c_str());
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
    lkGmUpgradeStop();  // before this session's lookups

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
        // tensors and letterboxes into the R10A2 canvas (RTX VSR demoted to Lanczos there,
        // with a log line). SMV_LIVE_HDR forces the mode for testing.
        // DLSS-G is HDR-capable through the exe-side pack (kHdrPackCS): SL mandates RGB10 + PQ
        // for HDR, the capture converts before present, SL interpolates on the PQ backbuffers.
        // "fruc" (Smooth Motion): the server feeds the NvOFFRUC bridge the PQ-encoded frames like rife, the bridge
        // quantises them to 8-bit BGRA for the flow and the warp, so only the TWEENS carry 8-bit
        // PQ precision (the real frames stay full precision); far better than the SDR capture
        // of an HDR-presented window, which is 2-3x over-bright and clipped (below). "fsrfg" (AMD FSR frame
        // generation) takes FRUC's HDR rule at fp16 (nativeFsrfgPair).
        const bool hdrCapable = g_backend == BK_DLSSG ||
                                (g_backend == BK_SERVER && (g_serverBackend == L"echo" || g_serverBackend == L"rife" ||
                                                            g_serverBackend == L"gmfss" || g_serverBackend == L"nvof" ||
                                                            g_serverBackend == L"rifedrba" ||
                                                            g_serverBackend == L"fruc" || g_serverBackend == L"fsrfg"));
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
    host.useSL = (g_backend == BK_DLSSG || g_fgOver);
    if (g_fgOver)
        LOG("DLSS-G with effects: the echo route runs them on every captured frame, DLSS-G generates from its "
            "frames\n");
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

        static Capture cap; // static: outlives this scope, single instance per process
        // DLSS-G keeps RGBA; server/identity routes stay BGRA, and so does DLSS-G over the echo route (it presents
        // the server's slots)
        cap.swizzle = host.useSL && !g_fgOver;
        // dlssg in HDR packs scRGB -> R10A2 PQ in the exe; swizzle is bypassed there. Must be set before init() (it sizes the chain
        // and can clear g_hdr on setup failure, before anything else reads it).
        cap.hdrPack = g_hdr && host.useSL && !g_fgOver;
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
                // node above it, lockedMultiple's step-up); the GPU fit decides whether it runs
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
            if (g_fgOver)
            {
                // DLSS-G makes the in-between frames: the first whole multiple of the measured rate that reaches the
                // target (the model caps it), set before the first served frame; the echo route holds one slot
                if (derive)
                {
                    const int capMax = host.maxGen ? (int)(host.maxGen < 5 ? host.maxGen : 5) : 5;
                    const int want = dlssgOnePassGen((double)g_targetFps / srcMeasFps, capMax);
                    sl::DLSSGOptions on{};
                    on.mode = sl::DLSSGMode::eOn;
                    on.numFramesToGenerate = (uint32_t)want;
                    if (slDLSSGSetOptions(host.vp, on) == sl::Result::eOk)
                        host.genFrames = want;
                    LOG("measured ~%.1f fps source, target %d fps: DLSS-G %dx\n", srcMeasFps, g_targetFps,
                        host.genFrames + 1);
                }
                genFrames = ceilGen = 0;
            }
            // A derived fixed multiplier is drift-tracked (re-derived in the stats tick),
            // so size the server at the slot CEILING and let each group's ladder carry the
            // current value. An explicit --gen (harness/perf use) keeps the exact classic spawn.
            const bool fixedDerived = g_noAdapt && g_targetFps > 0 && !g_genExplicit && !g_fgOver;
            const int spawnGen = fixedDerived ? ceilGen : genFrames;
            // The backend's model + TRT engines can take tens of seconds to build the FIRST time
            // at a new resolution (a fullscreen capture is often a never-before-built size:
            // measured ~6s RIFE, ~35s GMFSS). srv.start() blocks for that whole cold start, and
            // server backends do no warmup pre-present, so the overlay used to show nothing until
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
                        _snwprintf_s(hud.text, _TRUNCATE, L"SMV Live: Loading %ls...", nm);
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
        // DLSS-G over the echo route: off for one passthrough present of this slot, then on at `gen` generated frames
        // (a new count, or the re-warm after an input gap: a long gap poisons the DLSS-G pacer, see resetFG)
        ULONGLONG fgLastPresent = 0;
        auto fgRewarm = [&](uint32_t set, uint32_t i, int gen) -> bool {
            sl::DLSSGOptions off{};
            off.mode = sl::DLSSGMode::eOff;
            off.flags = sl::DLSSGFlags::eRetainResourcesWhenOff;
            if (slDLSSGSetOptions(host.vp, off) != sl::Result::eOk ||
                !host.presentTail(host.outBuf.Get(), slotOffset(set, i) - srv.shmInBytes, srv.shmPitch))
                return false;
            sl::DLSSGOptions on{};
            on.mode = sl::DLSSGMode::eOn;
            on.numFramesToGenerate = (uint32_t)gen;
            if (slDLSSGSetOptions(host.vp, on) != sl::Result::eOk)
                return false;
            host.genFrames = gen;
            return true;
        };
        // present one output slot GPU-direct from the shared VRAM ring (slots never touch host
        // memory)
        auto presentSlotFrom = [&](uint32_t set, uint32_t i) -> bool {
            if (g_fgOver)
            {
                const ULONGLONG t = GetTickCount64();
                if (fgLastPresent && t - fgLastPresent > 700)
                {
                    LOG("input gap %llums, resetting DLSS-G\n", (unsigned long long)(t - fgLastPresent));
                    if (!fgRewarm(set, i, host.genFrames))
                        return false;
                }
                fgLastPresent = t;
            }
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

        // warmup: block for the first frame, present it 3x so the FG feature builds. The server route takes the
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
            if (host.useSL && !g_fgOver) // DLSS-G over echo presents the server's frames only (FP16 under HDR)
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
                    // the GPU's one-pass limit once frame generation ran (init can read 0): the cascade's plan
                    if (st.numFramesToGenerateMax)
                    {
                        host.maxGen = st.numFramesToGenerateMax;
                        LOG("DLSS-G: up to %ux in one pass on this GPU\n", host.maxGen + 1);
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
            LOG("live overlay running\n");
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
        // RTSS (RivaTuner Statistics Server) injects its hook into Direct3D processes and limits their presents inside
        // Present: a frame limit set there caps this overlay with nothing in the throttle or pacing lines saying why,
        // so each session names the hook once (SMV_RTSS_MODULE = another module name: the trigger test's lever)
        wchar_t rtssModule[64] = L"RTSSHooks64.dll";
        GetEnvironmentVariableW(L"SMV_RTSS_MODULE", rtssModule, 64);
        bool rtssNamed = false;
        UINT statPresentBase = 0;
        host.scNative->GetLastPresentCount(&statPresentBase);
        // cap.dropped at the last stats tick; it starts at the drops of the model load, which are not this window's
        uint64_t statDropBase = cap.dropped;
        uint64_t statCaptured = 0;
        bool ratioWarned = false;
        int fgWantPrev = -1;  // DLSS-G target derivation: last derived gen (hysteresis, see stats)
        int fixWantPrev = -1; // fixed-ladder server derivation: same two-window hysteresis
        // Live DLSS 4.5 above one pass (the cascade): the session's ONE plan, made at the
        // 2 s derive when dlssgSnap's multiple exceeds the GPU's one-pass limit twice in a row (cascWantPrev), M =
        // cascM1 x cascM2; cascWarm = stage 1 runs and is fed beside the direct pass until it shares a frame; cascShown =
        // the multiple the stats line reports while stage 2 presents its frames (0 = one pass). Past lim x lim three:
        // cascM1 x cascM2 x cascM3 (cascM3 1 = two); cascLow = 2 s windows in a row three stages ran short of theirs;
        // cascMaxM = three stages fell short: the next plan's multiple at most (what the GPU held, 0 = none)
        int cascM = 0, cascM1 = 0, cascM2 = 0, cascM3 = 1, cascWantPrev = -1, cascShown = 0, cascLow = 0, cascMaxM = 0;
        bool cascTried = false, cascWarm = false;
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
        // (DLSS-G over the echo route: the floor, so a held frame reaches DLSS-G well inside its 700 ms gap)
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
        // at 160 W under the power cap, so a half-tween band would price x3 ~5 ms over its cost). One attempt per
        // regime: a step-up records its prediction (upPred for upM); the new lock's settled readings (kUpSettleMs)
        // or its fall set upK to what it cost, so a failed step is retried only once the readings improve that much,
        // or as a probe at the default price after a back-off that doubles per failure (kUpBackoff0 to
        // kUpBackoffMax: a price learned through a transient, a CPU burst lowering the GPU's ceiling, would otherwise
        // hold the session below a ladder it affords; at a real cap the probe costs ~2 s of the ladder above). Enter
        // with margin, hold with less: a lock holds while its cost stays under the budget times kUpHold (a ladder
        // at the edge of the budget would otherwise leave and re-enter every few seconds; up to ~5 % over the
        // source period the pacing absorbs a sag as latency, 0 to 0.5 drops/s, and the throttle's own drop cut
        // lowers the budget past that), falls only once it has been over for kUpOverMs (the group times are a
        // 5-group EMA) and never within kUpSettleMs of its lock; the back-off ends once the ladder it guards has
        // held kUpResetMs
        const double kUpKDefault = 1.05, kUpSettleMs = 2000.0, kUpBackoff0 = 4000.0, kUpBackoffMax = 300000.0;
        const double kUpHold = 1.10, kUpOverMs = 1000.0, kUpResetMs = 30000.0;
        double upK = kUpKDefault, upPred = 0, upFailT = 0, upBackoffMs = kUpBackoff0, upOverT = 0;
        uint32_t upM = 0;
        bool upSettled = false; // the stepped-up lock has logged its settled price
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
            // the GPU fit's node: Fit to the GPU off holds the user's target (and leaves FSR 4 untimed)
            if (tree && gridLock && frStepUp && !g_noGpuFit && panelHzMode > 1.0)
            {
                uint32_t n = 2;
                while (n <= top && n <= userCap * 1.03)
                    n *= 2;
                if (n <= top && userCap > n / 2 * 1.03)
                    above = n;
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
            const double budget = (std::min)(lockDtMs, emaDt) * thrBudget;
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
                if (!piped)
                    return pre + m * fs + (m - 1.0) * fc;
                double pm, pw;
                frucPipePrices(fc, pm, pw);
                return pre + m * fs + frucPipeCalls(m, pm, pw);
            };
            // a lock holds within slotOk and the hold band (kUpHold), never within kUpSettleMs of its lock; before a
            // group is timed the throttle's r stands in for the price (the node above holds unpriced)
            auto holds = [&](uint32_t m) {
                if (!slotOk(m))
                    return false;
                if (!timed)
                    return m == above || m - 1 <= r * 1.03;
                if (cost(m) <= budget * kUpHold || nowMs() - lockT < kUpSettleMs)
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
            auto afford = [&](uint32_t m) {
                if (upFailT > 0 && !probe && m >= upM)
                    return false;
                // an untimed tree (AMD FSR 4 learns its call price once an x4 or deeper group is priced): the
                // throttle's r within the target, the node above as one attempt
                if (!timed)
                    return tree && (m == above || m - 1 <= r * 0.97);
                return cost(m) * upK <= budget || (probe && cost(m) * kUpKDefault <= budget);
            };
            // the stepped-up lock's live price: what it costs now over the prediction that admitted it (re-read on
            // every settled pair and at its fall, so a sag raises it and its end lowers it again)
            auto learn = [&]() {
                if (upPred > 0 && upM == lockLogged && timed)
                    upK = (std::max)(1.0, cost(lockLogged) / upPred);
            };
            if (lockLogged && holds(lockLogged))
            {
                // a step up waits for settled readings (kUpSettleMs since the lock: the group times are an EMA, and a
                // lock's first groups still carry the ladder before it)
                const bool settled = nowMs() - lockT >= kUpSettleMs;
                if (settled)
                {
                    learn();
                    if (!upSettled && upPred > 0 && upM == lockLogged)
                    {
                        upSettled = true;
                        LOG("ladder price: x%u costs %.1f ms, %.2f x the %.1f ms its step-up priced\n", lockLogged,
                            cost(lockLogged), upK, upPred);
                    }
                }
                if (upFailT > 0 && lockLogged == upM && nowMs() - lockT >= kUpResetMs)
                {
                    upFailT = 0; // the ladder the back-off guards has held: the next failure starts the ladder over
                    upBackoffMs = kUpBackoff0;
                }
                const uint32_t next = tree ? lockLogged * 2 : lockLogged + 1;
                if (!(settled && fits(next) && afford(next)))
                    return lockLogged;
                // a step up: the largest ladder the price admits up to ceil(0.97 r) (the tree: up to its top node), the
                // prediction it ran on kept (an untimed tree's: one node, unpriced)
                uint32_t up = tree ? top : (std::max)(next, (uint32_t)ceil(r * 0.97));
                while (up > next && !(timed && fits(up) && afford(up)))
                    up = tree ? up / 2 : up - 1;
                upPred = timed ? cost(up) : 0.0;
                upM = up;
                upSettled = false;
                return up;
            }
            if (lockLogged)
            {
                learn(); // the lock falls: what it costs now is its price
                upOverT = 0;
                if (lockLogged == upM)
                {
                    // a stepped-up ladder falls: admitted again only after the back-off
                    upBackoffMs = upFailT > 0 ? (std::min)(upBackoffMs * 2.0, kUpBackoffMax) : kUpBackoff0;
                    upFailT = nowMs();
                    if (upPred > 0)
                        LOG("ladder price: x%u falls at %.1f ms (%.2f x its step-up's price); re-tried in %.0f s\n",
                            lockLogged, cost(lockLogged), upK, upBackoffMs / 1000.0);
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
                    return 2;
                }
                return gridLock ? m : 0;
            }
            const uint32_t m = (uint32_t)ceil(r * 0.97);
            return gridLock && fits(m) ? m : 0;
        };
        int sendLag = 0; // the last sent group's content lag in source periods (its pend entry takes it)
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
                    if (lockLogged == upM && upPred > 0)
                    {
                        upBackoffMs = upFailT > 0 ? (std::min)(upBackoffMs * 2.0, kUpBackoffMax) : kUpBackoff0;
                        upFailT = nowMs();
                        LOG("ladder price: x%u falls at the floor; re-tried in %.0f s\n", lockLogged,
                            upBackoffMs / 1000.0);
                    }
                    upOverT = 0;
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
                         (srv.nr->fsrfg ? (thrMd > 0 || thrMp > 0) && fsrfgPipeOn && fsrfgPipeOk(*srv.nr)
                                        : thrMd > 0 && frucPipeOn))
                {
                    const bool want = frucPipeWant(m);
                    pipeThis = want != frucPiped && nowMs() - frucPipeT < 2000.0 ? frucPiped : want;
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
                               : g_fgOver                         ? (double)(cascShown ? cascShown : host.genFrames + 1)
                               : (host.useSL || adaptTarget <= 0) ? (double)(cascShown ? cascShown : genFrames + 1)
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
            LOG("live: %.1f captured fps -> %.1f presented fps (ratio %.2f, target %.1f) "
                "%s ~%.0fms%s present %.2fms (wait %.2f flip %.2f) drop %.1f "
                "panel %.1fHz dwm %.1fHz disp %.1f/s bunch %.1f/s%s%s\n",
                capFps, outFps, ratio, tgt, host.useSL ? "handoff" : "latency", latAvg, slf, prTot, prWait, prFlip,
                (cap.dropped - statDropBase) / secs, panelHz, dwmHz, dispFps, host.prBunch / secs, stc, gpu);
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
        // SMV_LIVE_CASCADE=probe (the direct DLSS-G route only): the cascade's stage 1 beside the session, consume-only
        // (CascadeProbe); =present: this session's DLSS-G (stage 2, its --gen) presents stage 1's frames instead of the
        // captured ones; SMV_LIVE_CASCADE_GEN = stage 1's generated frames a source frame (1..5, default 5)
        std::unique_ptr<CascadeProbe> casc;
        const std::string cascMode = lkEnv("SMV_LIVE_CASCADE");
        if (!xqRoute && host.useSL && !g_fgOver && (cascMode == "probe" || cascMode == "present"))
        {
            const std::string cg = lkEnv("SMV_LIVE_CASCADE_GEN");
            const int g = cg.empty() ? 5 : (std::max)(1, (std::min)(5, atoi(cg.c_str())));
            casc = std::make_unique<CascadeProbe>();
            if (!casc->start(host.device.Get(), capW, capH, g, cap.hdrPack, cascMode == "present", false))
            {
                casc->stop();
                casc.reset();
            }
        }
        // the product's plan (no SMV_LIVE_CASCADE, =off = one pass only): DLSS-G as the app starts it, with --target (an
        // explicit --gen = fixed work, no plan), on the direct route at the capture size or on the effects route at the
        // presented size (stage 1 takes the effects' output slots); stage 1 = dlssg2f, which takes frames of 128 px a side
        // up
        const uint32_t cascW = g_fgOver ? W : capW, cascH = g_fgOver ? H : capH;
        const bool cascAuto = cascMode.empty() && host.useSL && (g_fgOver ? xqRoute : !xqRoute) && !g_genExplicit &&
                              g_targetFps > 0 && cascW >= 128 && cascH >= 128;
        std::unique_ptr<CascadeProbe> cascPend; // stage 1 starting on cascTh (cascRes -1 = running, 1 = up, 0 = failed)
        // three stages: cascUp = stage 1, fed by the session; casc = stage 2 on its ring (dlssg2f --from), presented
        std::unique_ptr<CascadeProbe> cascUp, cascPendUp;
        std::thread cascTh;
        std::atomic<int> cascRes{-1};
        // the one-pass limit: numFramesToGenerateMax + 1 (6 on RTX 50, 2 on RTX 40), 6 while the driver has not said;
        // SMV_DLSSG_LIMIT=N (2..6) = another GPU's limit on this one (RTX 40's x4 = 2 x 2)
        auto liveDlssgLimit = [&]() {
            const std::string lv = lkEnv("SMV_DLSSG_LIMIT");
            if (!lv.empty())
                return (std::max)(2, (std::min)(6, atoi(lv.c_str())));
            return (host.maxGen ? (int)(std::min)(host.maxGen, 5u) : 5) + 1;
        };
        // the video memory stage 1 takes: Streamline's own estimate for its DLSS-G (6 back buffers, m1 - 1 generated
        // frames; without one, S0 (c)'s figures: ~250 MiB a megapixel + 60) + its ring of 12 shared frames, its 3 tagged
        // inputs and its 6 back buffers
        auto cascPrice = [&](int m1) -> uint64_t {
            const uint64_t frame = (uint64_t)cascW * cascH * 4;
            sl::DLSSGOptions eo{};
            eo.mode = sl::DLSSGMode::eOn;
            eo.numFramesToGenerate = (uint32_t)(m1 - 1);
            eo.flags = sl::DLSSGFlags::eRequestVRAMEstimate;
            eo.numBackBuffers = 6;
            eo.colorWidth = eo.mvecDepthWidth = cascW;
            eo.colorHeight = eo.mvecDepthHeight = cascH;
            eo.colorBufferFormat = (uint32_t)host.backbuffers[0]->GetDesc().Format;
            eo.mvecBufferFormat = DXGI_FORMAT_R16G16_FLOAT;
            eo.depthBufferFormat = DXGI_FORMAT_R32_FLOAT;
            sl::DLSSGState es{};
            uint64_t est = slDLSSGGetState(host.vp, es, &eo) == sl::Result::eOk ? es.estimatedVRAMUsageInBytes : 0;
            if (!est)
                est = ((uint64_t)60 << 20) + frame / 4 * 250;
            return est + 21 * frame;
        };
        // stage 1 lost, failed or not wanted any more: end it, release stage 2's waits on its fence, one pass again
        auto cascFallback = [&](const char* why) -> bool {
            const bool presenting = casc->present;
            LOG("DLSS 4.5's second %spass%s ended (%s): back to one pass at %dx\n", cascUp ? "and third " : "",
                cascUp ? "es" : "", why, liveDlssgLimit());
            casc->kill();
            if (cascUp)
                cascUp->kill();
            casc->stop();
            casc.reset();
            if (cascUp)
            {
                cascUp->stop();
                cascUp.reset();
            }
            cascWarm = false;
            cascShown = 0;
            cascLow = 0;
            if (g_fgOver) // the effects route: re-warmed on its newest output slot
                return !presenting || idleSlotIdx < 0 ||
                       fgRewarm(idleSlotSet, (uint32_t)idleSlotIdx, liveDlssgLimit() - 1);
            if (!g_genExplicit)
                genFrames = liveDlssgLimit() - 1;
            return !presenting || resetFG();
        };
        // the effects route's cascade feed: one output slot read back and handed to stage 1, so every frame it
        // generates from carries the effects
        auto cascFeedSlot = [&](uint32_t set, uint32_t i) -> bool {
            CascadeProbe* to = cascUp ? cascUp.get() : casc.get();
            return host.readSlot(host.outBuf.Get(), slotOffset(set, i) - srv.shmInBytes, (UINT64)srv.shmPitch * H,
                                 [&](const uint8_t* p) { to->feedRows(p, srv.shmPitch); });
        };
        // every check covers both probes of three stages (the session feeds cascUp, presents casc)
        auto cascDead = [&]() { return casc->dead() || (cascUp && cascUp->dead()); };
        auto cascStalled = [&]() {
            const bool a = casc->stalled(), b = cascUp && cascUp->stalled();
            return a || b;
        };
        auto cascReport = [&](double secs) {
            if (cascUp)
                cascUp->report(secs);
            casc->report(secs);
        };
        // three stages below 85 % of their multiple three 2 s windows in a row: what the GPU affords is less, so they
        // end and the next plan takes the largest multiple at most what they held, with a tenth to spare (cascMaxM);
        // false = that fallback's reset failed
        auto cascShortCheck = [&](double capFps, double outFps) -> bool {
            if (!cascUp || !cascShown || hidden || capFps < 0.5)
            {
                cascLow = 0;
                return true;
            }
            cascLow = outFps < 0.85 * cascShown * capFps ? cascLow + 1 : 0;
            if (cascLow < 3)
                return true;
            char why[96];
            snprintf(why, sizeof(why), "%.1fx of %dx, what the GPU affords is less", outFps / capFps, cascShown);
            cascMaxM = (std::max)(2, (std::min)(cascShown - 1, (int)(outFps / capFps / 0.9)));
            cascTried = false;
            cascWantPrev = -1;
            return cascFallback(why);
        };
        // a plan's start done (cascRes >= 0): its probes join the session warming (fed, not shown yet), or end
        auto cascTakePend = [&]() {
            cascTh.join();
            if (cascRes.load() == 1)
            {
                casc = std::move(cascPend);
                cascUp = std::move(cascPendUp);
                casc->present = false;
                cascWarm = true;
                return;
            }
            cascPend->stop();
            cascPend.reset();
            if (cascPendUp)
            {
                cascPendUp->stop();
                cascPendUp.reset();
            }
            LOG("DLSS 4.5 stays at %dx: its second pass did not start\n", liveDlssgLimit());
        };
        // the presented stage shared its first frame: the session presents its frames from now on; returns the
        // session's own pass (its multiple between those frames)
        auto cascTakeover = [&]() -> int {
            cascWarm = false;
            casc->present = true;
            casc->skipToNewest();
            cascStalled();
            cascShown = cascM;
            cascLow = 0;
            if (cascM3 > 1)
                LOG("DLSS 4.5 at %dx: %dx in a second process, %dx in a third, then %dx here between those frames\n",
                    cascM, cascM1, cascM2, cascM3);
            else
                LOG("DLSS 4.5 at %dx: %dx in a second process, then %dx here between those frames\n", cascM, cascM1,
                    cascM2);
            return cascM3 > 1 ? cascM3 : cascM2;
        };
        // above one pass: the plan, once a session, when the same multiple is derived twice in a row (either loop's 2 s
        // window); one pass runs at the limit meanwhile. Stage 1 takes cascW x cascH frames in the back buffer's format:
        // the capture's R8G8B8A8 / R10G10B10A2 packing on the direct route, the output slots' B8G8R8A8 / R10G10B10A2
        // on the effects route. Past lim x lim three passes when the room holds both processes (else the best two); after
        // three fell short of their multiple, at most what they held (cascMaxM)
        auto cascPlan = [&](double capFps) {
            const int lim = liveDlssgLimit();
            int m1 = 0, m2 = 0, m3 = 1;
            const double want = (double)g_targetFps / capFps;
            int M0 = dlssgSnap(want, lim, m1, m2, nullptr, &m3);
            if (cascMaxM && M0 > cascMaxM)
                M0 = dlssgAtMost(cascMaxM, lim, m1, m2, &m3);
            int M = M0;
            if (M > lim && M == cascWantPrev)
            {
                cascTried = true;
                const uint64_t room = nativeVideoMemoryRoom();
                uint64_t need = cascPrice(m1) + (m3 > 1 ? cascPrice(m2) : 0);
                if (m3 > 1 && room && need > room)
                {
                    LOG("DLSS 4.5: %dx (%dx, %dx, then %dx) needs ~%.0f MiB of video memory for its second and third "
                        "passes, %.0f MiB are free; two passes at most\n",
                        M, m1, m2, m3, need / 1048576.0, room / 1048576.0);
                    M = cascMaxM ? dlssgAtMost((std::min)(cascMaxM, lim * lim), lim, m1, m2)
                                 : dlssgSnap(want, lim, m1, m2);
                    m3 = 1;
                    need = cascPrice(m1);
                }
                if (room && need > room)
                    LOG("DLSS 4.5 stays at %dx: %dx (%dx, then %dx) needs ~%.0f MiB of video memory for its second "
                        "pass, %.0f MiB are free\n",
                        lim, M, m1, m2, need / 1048576.0, room / 1048576.0);
                else
                {
                    cascM = M;
                    cascM1 = m1;
                    cascM2 = m2;
                    cascM3 = m3;
                    if (m3 > 1)
                        LOG("DLSS 4.5 plan: %dx for target %d fps at ~%.1f captured = %dx, then %dx, then %dx between "
                            "those (~%.0f MiB more video memory); starting the second and third passes\n",
                            M, g_targetFps, capFps, m1, m2, m3, need / 1048576.0);
                    else
                        LOG("DLSS 4.5 plan: %dx for target %d fps at ~%.1f captured = %dx, then %dx between those "
                            "(~%.0f MiB more video memory); starting the second pass\n",
                            M, g_targetFps, capFps, m1, m2, need / 1048576.0);
                    cascPend = std::make_unique<CascadeProbe>();
                    if (m3 > 1)
                        cascPendUp = std::make_unique<CascadeProbe>();
                    cascRes = -1;
                    CascadeProbe *p = cascPend.get(), *u = cascPendUp.get();
                    ID3D12Device* dev = host.device.Get();
                    const bool hdr10 = g_fgOver ? g_hdr : cap.hdrPack, bgra = g_fgOver && !g_hdr;
                    // three: stage 1 first (the session feeds it), then stage 2 on its ring (the one presented)
                    cascTh = std::thread([p, u, dev, w = cascW, h = cascH, m1, m2, hdr10, bgra, &cascRes] {
                        cascRes = (u ? u->start(dev, w, h, m1 - 1, hdr10, true, bgra) &&
                                           p->start(dev, w, h, m2 - 1, hdr10, true, bgra, u)
                                     : p->start(dev, w, h, m1 - 1, hdr10, true, bgra))
                                      ? 1
                                      : 0;
                    });
                }
            }
            cascWantPrev = M0;
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
            // second, smoothed over ~2 s, take a tenth off it (not at the floor), 5 clean seconds give 2 points back. At the floor the
            // tween is re-timed by one window up after 4 s, doubling to 60 s while it keeps answering "the floor". A
            // pause or a static hold starts the window over; before any tween is timed the start target stands.
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
                    if (thrProbing)
                    {
                        thrProbing = false; // the window just timed fresh tweens: the model above read them
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
                        next = srcFps * 1.1;
                        thrProbing = true;
                        thrFloor = false;
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
                            xqThrottle(xqLastPresMs > 0 ? xqLastPresMs : nowMs());
                        continue;
                    }
                    const uint32_t idx = raw - 1;
                    if (idx >= g.nfr)
                    {
                        LOG("live server protocol error (slot index)\n");
                        rc2 = 1;
                        break;
                    }
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
                        statLatSum += (nowQpc100() - s.sentTs) / 10000.0 + (srv.contentLag + s.lag) * emaDt;
                        statLatN++;
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
                // the effects route's cascade: its stage 1 came up (fed beside the one pass until it shares a frame)
                // or did not; a dead one = one pass again; its first shared frame = stage 2 takes over at m2
                if (cascPend && cascRes.load() >= 0)
                    cascTakePend();
                if (casc && (casc->present || cascWarm) && cascDead())
                {
                    if (!cascFallback("its process ended"))
                    {
                        LOG("DLSS-G reset failed\n");
                        rc2 = 1;
                        break;
                    }
                }
                else if (cascWarm && casc->hdr && casc->hdr->frames > 0 && idleSlotIdx >= 0)
                {
                    if (!fgRewarm(idleSlotSet, (uint32_t)idleSlotIdx, cascTakeover() - 1))
                    {
                        LOG("DLSS-G reset failed\n");
                        rc2 = 1;
                        break;
                    }
                    fgLastPresent = GetTickCount64();
                }
                // stage 2: every frame stage 1 shared since the last pass, in order (a long gap re-warms DLSS-G on the
                // newest output slot first, as presentSlotFrom does)
                if (casc && casc->present && !hidden)
                {
                    uint32_t rs = 0;
                    uint64_t fv = 0;
                    while (!rc2 && casc->nextFrame(rs, fv))
                    {
                        const ULONGLONG t = GetTickCount64();
                        if (fgLastPresent && t - fgLastPresent > 700 && idleSlotIdx >= 0)
                        {
                            LOG("input gap %llums, resetting DLSS-G\n", (unsigned long long)(t - fgLastPresent));
                            if (!fgRewarm(idleSlotSet, (uint32_t)idleSlotIdx, host.genFrames))
                            {
                                LOG("DLSS-G reset failed\n");
                                rc2 = 1;
                                break;
                            }
                        }
                        fgLastPresent = t;
                        if (!host.presentRing(casc->ring[rs].Get(), casc->shared.Get(), fv))
                        {
                            LOG("presentRing failed\n");
                            rc2 = 1;
                        }
                    }
                    if (rc2)
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
                            pend.push_back(XqGroup{shmSeq, (uint32_t)(shmSeq % 2), nfr ? nfr : 1, nfr, arrMs, emaDt,
                                                   sentTs, sendTick, false, false, sendGridMs, sendLag});
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
                    sl::DLSSGState fgSt{};
                    if (g_fgOver)
                        slDLSSGGetState(host.vp, fgSt, nullptr);
                    logLiveStats(capFps, outFps, latAvg, secs, g_fgOver ? &fgSt : nullptr);
                    if (casc)
                        cascReport(secs);
                    if (casc && casc->present && !hidden &&
                        ((cascStalled() && !cascFallback("it stopped presenting")) || !cascShortCheck(capFps, outFps)))
                    {
                        LOG("DLSS-G reset failed\n");
                        rc2 = 1;
                        break;
                    }
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
                    // DLSS-G over echo: the handoff plus one capture interval, as on the direct DLSS-G route
                    hud.update(capFps, outFps, g_fgOver && capFps > 0.5 ? latAvg + 1000.0 / capFps : latAvg);
                    statLatSum = 0;
                    statLatN = 0;
                    // DLSS-G over echo: its generated-frame count follows the target the way the direct route's does
                    // (two equal 2 s derivations in a row before a switch)
                    if (g_fgOver && g_targetFps > 0 && !g_genExplicit && capFps > 3.0 && idleSlotIdx >= 0 && !casc &&
                        !cascPend)
                    {
                        if (cascAuto && !cascTried)
                            cascPlan(capFps); // one pass runs at the limit meanwhile (want below)
                        const int capMax = liveDlssgLimit() - 1;
                        const int want = dlssgOnePassGen((double)g_targetFps / capFps, capMax);
                        if (want != host.genFrames && want == fixWantPrev)
                        {
                            LOG("target %d fps at ~%.1f captured: switching DLSS-G to %dx\n", g_targetFps, capFps,
                                want + 1);
                            if (!fgRewarm(idleSlotSet, (uint32_t)idleSlotIdx, want))
                            {
                                LOG("DLSS-G reset failed\n");
                                rc2 = 1;
                                break;
                            }
                        }
                        fixWantPrev = want;
                    }
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
                // the plan's stage 1 came up (fed beside the direct pass until it shares a frame) or did not
                if (cascPend && cascRes.load() >= 0)
                    cascTakePend();
                if (casc && (casc->present || cascWarm) && cascDead())
                {
                    if (!cascFallback("its process ended"))
                    {
                        LOG("DLSS-G reset failed\n");
                        rc2 = 1;
                        break;
                    }
                }
                else if (cascWarm && casc->hdr && casc->hdr->frames > 0)
                {
                    genFrames = cascTakeover() - 1;
                    if (!resetFG())
                    {
                        LOG("DLSS-G reset failed\n");
                        rc2 = 1;
                        break;
                    }
                }
                // the cascade's stage 2: every frame stage 1 shared since the last pass, in order
                if (casc && casc->present && !hidden)
                {
                    uint32_t s = 0;
                    uint64_t fv = 0;
                    while (casc->nextFrame(s, fv))
                        if (!host.presentRing(casc->ring[s].Get(), casc->shared.Get(), fv))
                        {
                            LOG("presentRing failed\n");
                            rc2 = 1;
                            break;
                        }
                    if (rc2)
                        break;
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
                    if (casc)
                        cascReport(secs);
                    if (casc && casc->present && !hidden &&
                        ((cascStalled() && !cascFallback("it stopped presenting")) || !cascShortCheck(capFps, outFps)))
                    {
                        LOG("DLSS-G reset failed\n");
                        rc2 = 1;
                        break;
                    }
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
                    if (host.useSL && g_targetFps > 0 && !g_genExplicit && capFps > 3.0 && !casc && !cascPend)
                    {
                        const int capMax = liveDlssgLimit() - 1;
                        if (cascAuto && !cascTried)
                            cascPlan(capFps); // the direct pass runs at the limit meanwhile (want below)
                        const int want = dlssgOnePassGen((double)g_targetFps / capFps, capMax);
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
            if (cascTh.joinable())
                cascTh.join();
            cascPend->stop();
            if (cascPendUp)
                cascPendUp->stop();
        }
        if (casc)
            casc->stop(); // its reader queue lives on the host's device
        if (cascUp)
            cascUp->stop();
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
