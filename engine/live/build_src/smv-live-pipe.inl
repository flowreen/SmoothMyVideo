// part of smv-live.cpp (unity include, see the file map there); not a standalone translation unit
// ---------------------------------------------------------------- model server (the native host)

// The present loop's view of the model backend. The backend is the in-process native host
// (NativeRife, smv-live-native.inl): the loop writes one group message per captured frame and
// reads per-slot tokens back, both through nr->write / nr->read; frames never cross a pipe, the
// capture texture and the output ring are shared GPU memory. The struct keeps the message
// interface of the python live server it replaced, which is why the loop talks to a "server".
struct PipeServer
{
    // output ring geometry (the present loop's slotOffset math reads these)
    size_t shmInBytes = 0;   // one captured frame, rounded up to 512
    size_t shmSlot = 0;      // one output slot size (pitch * H, rounded up to 512)
    uint32_t shmSlots = 0;   // slots per set (gen + 1)
    uint32_t shmPitch = 0;   // slot row stride, 256-aligned (D3D12 texture-copy requirement)
    // what the started host provides, read by the present loop; all true once start() succeeded
    bool captexAck = false;  // zero-copy capture import
    bool outbufAck = false;  // shared-VRAM output ring
    bool streamAck = false;  // per-slot streaming tokens
    bool capRelAck = false;  // the capture-release token
    int contentLag = 0;      // backend pipeline lag in CAPTURE INTERVALS (DRBA trails by one
                             // capture); added to the displayed latency for honesty
    HANDLE wdog = nullptr;   // stall watchdog (armed after the start, so the model load is exempt)
    HANDLE wdogEv = nullptr; // stop() signals it: the watchdog wakes at once instead of after
                             // the rest of its second
    volatile LONGLONG ioSince = 0;        // tick since a group is outstanding (armed by the present
                                          // loop), 0 = idle
    volatile LONG wdogStop = 0;
    volatile LONG stalled = 0;            // the watchdog ended a stalled host (TRT shape-bug class)
    NativeRife* nr = nullptr;
    void nativeAbort() { if (nr) nr->die("shutting down"); }
    // Startup steps 1b and 2 (2026-09-12): everything the native host needs except the output
    // ring depends on the capture size only, never on the measured source rate, so runLive
    // starts it here BEFORE its source-rate measurement (0.5 to 0.7 s) instead of after it:
    // the DLL load, the engine handoff (with the CUDA device init and the capture import in
    // parallel on a helper thread), then the kernels, the model buffers, the engine
    // deserialize, the jit cache and the execution contexts. startNative joins the thread and
    // continues with the ring. Both ask nativeRefusal, so a session this never starts for is
    // one startNative refuses (and logs). Measured: warm native start 3.1 s to 2.7 s.
    std::thread earlyTh;
    std::atomic<int> earlyRc{ -1 };   // -1 not started or running, 1 early init ok, 0 failed

    // why the host cannot run this session, or nullptr
    static const char* nativeRefusal(const std::wstring& backend, HANDLE capTex, HANDLE capFence, Host* h12)
    {
        if (g_backend != BK_SERVER || !nativeBackendOk(backend))
            return "the native host runs rife, blend, gmfss, nvof, fruc, rifedrba and echo only";
        if (g_dlssnr && !g_nrNative) return "DLSS 5 is on without its native pass";
        if (!h12 || !capTex || !capFence) return "capture interop or the D3D12 host is unavailable";
        return nullptr;
    }

    void beginNativeHandoff(const std::wstring& script, const std::wstring& backend, int gen,
                            uint32_t capW, uint32_t capH, HANDLE capTex, HANDLE capFence, Host* h12)
    {
        if (nativeRefusal(backend, capTex, capFence, h12)) return;
        nr = new NativeRife();
        NativeRife* early = nr;
        IDXGIAdapter1* adapter = h12->adapter.Get();   // owned by the Host, which outlives the server
        earlyTh = std::thread([this, early, script, backend, gen, capW, capH, adapter, capTex, capFence] {
            const int64_t t0 = nowQpc100();
            // the cudart import library is a lazy loader stub that opens the DLL by name on the
            // first call, so every CUDA call sits behind nativeLoadDlls
            bool ok = nativeLoadDlls(script) && nativeConfigHdr(*early);
            bool devOk = false;
            std::thread devTh;
            if (ok) devTh = std::thread([&] {
                devOk = nativeCudaDeviceInit(*early, adapter, capTex, capFence, capW, capH, g_hdr);
            });
            ok = ok && nativeHandoff(script, backend, gen, capW, capH, *early);
            const int64_t t1 = nowQpc100();
            if (devTh.joinable()) devTh.join();
            // the helper thread bound the context on ITSELF; this thread needs its own bind
            ok = ok && devOk && nativeBindDevice(*early)
                 && nativeCudaInitEarly(*early, nativeCacheDir(*early));
            // the TrueHDR setup overlaps the engine load on a helper thread (it needs the
            // kernel module, the stream and the capture size, all ready here; the bridge
            // retains the primary context, which the helper binds on itself)
            bool thdrOk = true;
            std::thread thdrTh;
            if (ok && (early->rtxHdr || early->vsrWant || early->sharpen > 0.0f || early->fitAa
                       || early->uw > 0 || early->restore))
                thdrTh = std::thread([&] { thdrOk = nativeBindDevice(*early) && nativeRtxInit(*early); });
            ok = ok && nativeTrtInit(*early);
            if (thdrTh.joinable()) thdrTh.join();
            ok = ok && thdrOk;
            const int64_t t2 = nowQpc100();
            if (ok)
                LOG("native: early init done: handoff %.2f s, kernels + engines + contexts%s %.2f s\n",
                    (t1 - t0) / 1e7, early->rtxHdr ? " + TrueHDR" : "", (t2 - t1) / 1e7);
            else
                nativeDropEarly(*early);
            earlyRc.store(ok ? 1 : 0);
        });
        LOG("native: engine handoff started during the source-rate measurement\n");
    }
    ~PipeServer() { if (earlyTh.joinable()) earlyTh.join(); }

    // A TRT-RTX engine at an unlucky shape can hang the host PERMANENTLY mid-stream (a GPU sync
    // that never completes). The token read then blocks forever and the overlay freezes on its
    // last frame. The watchdog turns that failure mode into a detectable one: the host's waits
    // are aborted, the read fails, the exe exits with code 6 and the app revives the session.
    static DWORD WINAPI wdogProc(LPVOID p)
    {
        PipeServer* s = (PipeServer*)p;
        while (!s->wdogStop)
        {
            // one second, or a stop; without the event (creation failed) the plain sleep
            if (s->wdogEv ? WaitForSingleObject(s->wdogEv, 1000) == WAIT_OBJECT_0 : (Sleep(1000), false))
                break;
            const LONGLONG since = s->ioSince;
            if (since && (LONGLONG)GetTickCount64() - since > 10000)
            {
                LOG("live server stalled >10s, ending the session (engine shape bug? the app revives it)\n");
                InterlockedExchange(&s->stalled, 1);
                s->nativeAbort();   // wakes the loop's blocked read
                break;
            }
        }
        return 0;
    }

    void armWatchdog()
    {
        wdogEv = CreateEventW(nullptr, TRUE, FALSE, nullptr);   // manual reset, signalled by stop()
        wdog = CreateThread(nullptr, 0, wdogProc, this, 0, nullptr);
    }

    // The token reader thread blocks here essentially all the time (also while the session is
    // alt-tab paused and sends nothing), so the reads and writes do not drive ioSince: the
    // present loop arms it while a group is OUTSTANDING (sent, not yet closed), the condition
    // the watchdog was written for.
    bool readFullRaw(void* p, DWORD n) { return nr && nr->read(p, n); }
    bool writeFull(const void* p, DWORD n) { return nr && nr->write(p, n); }

    // The output ring is the exe's own D3D12 buffer, imported straight back into CUDA.
    // Returns 0 on success; any other return ends the session (the reason is logged).
    int startNative(const std::wstring& script, const std::wstring& backend, int gen, uint32_t capW,
                    uint32_t capH, HANDLE capTex, HANDLE capFence, Host* h12)
    {
        // an early handoff thread owns nr until joined here; every refusal drops that object
        const bool early = earlyTh.joinable();
        if (early) earlyTh.join();
        auto dropEarly = [&] { if (early && nr) { nativeDropEarly(*nr); delete nr; nr = nullptr; } };
        // decided before the handoff, which does not carry the effect flags, so the model is
        // never loaded just to refuse
        if (const char* why = nativeRefusal(backend, capTex, capFence, h12))
        { dropEarly(); LOG("native host: %s\n", why); return 1; }
        if (early && earlyRc.load() != 1)
        { dropEarly(); LOG("native host: the early engine handoff failed\n"); return 1; }
        shmInBytes = ((size_t)capW * capH * 4 + 511) & ~(size_t)511;
        shmPitch = (W * 4 + 255) & ~255u;
        shmSlot = ((size_t)shmPitch * H + 511) & ~(size_t)511;
        shmSlots = (uint32_t)gen + 1;
        const uint64_t obBytes = (2ull * shmSlots * shmSlot + 0xFFFFull) & ~0xFFFFull;
        if (!h12->createOutBuf(obBytes))
        { dropEarly(); LOG("native host: output ring creation failed\n"); return 1; }
        if (!early) nr = new NativeRife();
        if (!nativeStart(*nr, script, backend, gen, capW, capH, h12->adapter.Get(),
                         capTex, capFence, h12->hOutBuf, obBytes, shmSlots, shmPitch, shmSlot,
                         early))
        {
            delete nr;
            nr = nullptr;
            if (h12->hOutBuf) { CloseHandle(h12->hOutBuf); h12->hOutBuf = nullptr; }
            h12->outBuf.Reset();
            h12->outBufBytes = 0;
            shmInBytes = shmSlot = 0;
            shmSlots = shmPitch = 0;
            return 1;
        }
        // zero-copy capture, zero-copy output, per-slot streaming tokens and the
        // capture-release token are all the host's by construction
        captexAck = outbufAck = streamAck = capRelAck = true;
        contentLag = nr->drba ? 1 : 0;   // DRBA trails by one capture
        armWatchdog();
        return 0;
    }
    int start(const std::wstring& script, const std::wstring& backend, int gen, uint32_t capW,
              uint32_t capH, HANDLE capTex = nullptr, HANDLE capFence = nullptr, Host* h12 = nullptr)
    {
        if (startNative(script, backend, gen, capW, capH, capTex, capFence, h12) == 0) return 0;
        // the session ended during the engine load (hotkey off, load-time resize): the caller
        // ends the session cleanly
        if (nativeLoadAbort()) { LOG("session ended during the engine load\n"); return 1; }
        // the host is the only live route: a session it cannot start ends here
        LOG("native host: this session cannot start in the native host (the line above names the reason)\n");
        return 1;
    }
    void stop()
    {
        wdogStop = 1;
        if (wdogEv) SetEvent(wdogEv);
        // a session that ends before startNative ran: the early thread may still own nr
        if (earlyTh.joinable()) earlyTh.join();
        if (nr)
        {
            if (nr->started) nativeStop(*nr); else nativeDropEarly(*nr);
            delete nr;
            nr = nullptr;
        }
        if (wdog) { WaitForSingleObject(wdog, 2000); CloseHandle(wdog); wdog = nullptr; }
        if (wdogEv) { CloseHandle(wdogEv); wdogEv = nullptr; }
    }
};

