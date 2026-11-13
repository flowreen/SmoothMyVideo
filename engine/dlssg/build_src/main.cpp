// dlssg2f - feeds video frames through DLSS Frame Generation on NGX's own calls (ngx_direct.h) and returns the
// generated frames: a D3D12 device, flat depth and the pan's motion vectors (video frames have neither), no swap
// chain and no window.
//
// Usage:
//   dlssg2f.exe --server W H [--gen N] [--hdr10] [--cut-event HANDLE]
//       streaming: raw RGBA8 frames (W*H*4) on stdin; for every frame after the
//       first, the N (default 1, max 5) DLSS-G frames generated between it and
//       its predecessor - evenly spaced, in temporal order - are written to
//       stdout as raw RGBA8 (multi-frame generation: N=1 is 2x, N=5 is 6x).
//       --cut-event: an inherited auto-reset event (its handle value in decimal). Set
//       before a frame is sent, it makes that frame the first of a new clip: no
//       frames come back for it and the tweens after it are a fresh server's. The
//       handshake then ends in " cut=1".
//       --hdr10: HDR10 frames both ways instead (PQ BT.2020 as R10G10B10A2 words,
//       still W*H*4 bytes). The handshake then ends in " hdr10=1".
//       All logging goes to stderr; stdout is binary-only. EOF on stdin exits.
//       This is the SmoothMyVideo backend.
//   dlssg2f.exe --stage1 W H [--gen N] [--hdr10 | --bgra] --share TAG [--from TAG]
//       a stage of a live plan as a process: --from = another stage's ring as its input (on the GPU, in order; stdin
//       then only ends it), else the same frames on stdin (--bgra: SDR frames in B8G8R8A8 order, the order of Live's
//       frames). An input's generated frames and then the input itself go into a ring of textures shared by name
//       with the process that reads them (smv_dg1_TAG_ring0.., a shared fence smv_dg1_TAG_fence, the header
//       Stage1Hdr in the file mapping smv_dg1_TAG_hdr), spread over the source's own interval (ngx_stage.h). stdout
//       carries only the handshake line.
// Exit 2 = DLSS frame generation did not start on this system, 3 = --gen is more than this GPU makes in one pass.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <io.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <string>
#include <vector>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

using Microsoft::WRL::ComPtr;

static uint32_t W = 1920;
static uint32_t H = 1080;

#define LOG(...)                                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        fprintf(stderr, __VA_ARGS__);                                                                                  \
        fflush(stderr);                                                                                                \
    } while (0)
#define CHECK_HR(x)                                                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        HRESULT _hr = (x);                                                                                             \
        if (FAILED(_hr))                                                                                               \
        {                                                                                                              \
            LOG("dlssg2f FAIL hr=0x%08lx at %s:%d\n", _hr, __FILE__, __LINE__);                                        \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

#include "pan_vectors.h"
#include "ngx_direct.h"
#include "ngx_stage.h"

// ---------------------------------------------------------------- entry

struct ServerFrames
{
    std::vector<std::vector<uint8_t>> outFrames, outFramesW;
};

// --server's pair writer: one thread streams a pair's generated frames to stdout while the server thread reads the
// next input frame and evaluates it. post() hands over the frame pointers of a buffer set the server thread no longer
// touches; it waits for the previous pair to be written first, so the frames leave in order. A failed write is logged
// once (the host's read of the pair fails and restarts the server). The thread holds the writer and the frames
// (keep), so a writer an early return leaves blocked on a host that stopped reading never touches freed memory before
// the process exits.
struct PairWriter : std::enable_shared_from_this<PairWriter>
{
    std::mutex m;
    std::condition_variable cv;
    std::vector<const uint8_t*> src;
    size_t bytes = 0;
    bool busy = false, quit = false, failed = false;
    std::shared_ptr<ServerFrames> keep;
    std::thread th;

    void start()
    {
        th = std::thread([self = shared_from_this()] { self->run(); });
    }
    void run()
    {
        for (;;)
        {
            std::unique_lock<std::mutex> lk(m);
            cv.wait(lk, [&] { return busy || quit; });
            if (!busy)
                return;
            const std::vector<const uint8_t*> s = src;
            const size_t n = bytes;
            lk.unlock();
            bool ok = true;
            for (const uint8_t* p : s)
                ok = ok && fwrite(p, 1, n, stdout) == n;
            ok = fflush(stdout) == 0 && ok;
            lk.lock();
            if (!ok && !failed)
            {
                failed = true;
                LOG("the output pipe write failed (the host stopped reading)\n");
            }
            busy = false;
            cv.notify_all();
        }
    }
    void waitIdle()
    {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&] { return !busy; });
    }
    void post(const std::vector<const uint8_t*>& s, size_t n)
    {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&] { return !busy; });
        src = s;
        bytes = n;
        busy = true;
        cv.notify_all();
    }
    // a normal end has waited for the last pair; an early return still lets the pair in flight reach the host (it
    // reads that pair in either order, and a host that stopped reading fails the write), so a failure regenerates
    // only the failing pair; a writer still blocked after 10 s (the host's watchdog) is left to the process exit
    void finish()
    {
        bool idle;
        {
            std::unique_lock<std::mutex> lk(m);
            idle = cv.wait_for(lk, std::chrono::seconds(10), [&] { return !busy; });
            quit = true;
        }
        cv.notify_all();
        if (idle)
            th.join();
        else
            th.detach();
    }
};
struct PairWriterGuard
{
    std::shared_ptr<PairWriter> w;
    ~PairWriterGuard()
    {
        if (w && w->th.joinable())
            w->finish();
    }
};

// SMV_DLSSG_NGX=fail: NGX's frame generation treated as unavailable after its full start (the trigger test of the
// refusal)
static bool ngxFailLever()
{
    wchar_t v[8] = {0};
    return GetEnvironmentVariableW(L"SMV_DLSSG_NGX", v, _countof(v)) == 4 && wcscmp(v, L"fail") == 0;
}

// --server (see the usage at the top). false = frame generation did not come up and nothing went to stdout
static bool runServer(int genFrames, bool hdr10, HANDLE cutEv, int& rc)
{
    ngxfg::Direct fg;
    if (!fg.init(W, H, genFrames, hdr10))
        return false;
    if (ngxFailLever())
    {
        LOG("SMV_DLSSG_NGX=fail: NGX's frame generation treated as unavailable\n");
        fg.shutdown();
        return false;
    }
    rc = 1;
    if (fg.maxGen && (uint32_t)genFrames > fg.maxGen)
    {
        LOG("multi-frame generation beyond %ux is not supported on this GPU (requested %ux)\n", fg.maxGen + 1,
            genFrames + 1);
        rc = 3;
        return true;
    }
    LOG("DLSS-G ready: NGX's own calls\n");

    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
    fprintf(stdout, "DLSSG READY gen=%d max=%u%s wq=1%s ngx=1\n", genFrames, fg.maxGen, hdr10 ? " hdr10=1" : "",
            cutEv ? " cut=1" : "");
    fflush(stdout);

    const size_t frameBytes = (size_t)W * H * 4;
    std::vector<uint8_t> inFrame(frameBytes);
    const auto frames = std::make_shared<ServerFrames>();
    auto& outFrames = frames->outFrames;
    auto& outFramesW = frames->outFramesW;
    outFrames.assign((size_t)genFrames, std::vector<uint8_t>(frameBytes));
    outFramesW.assign((size_t)genFrames, std::vector<uint8_t>(frameBytes));
    std::vector<uint8_t*> outPtrs, outPtrsW;
    for (auto& f : outFrames)
        outPtrs.push_back(f.data());
    for (auto& f : outFramesW)
        outPtrsW.push_back(f.data());
    const auto writerP = std::make_shared<PairWriter>();
    writerP->keep = frames;
    writerP->start();
    PairWriterGuard writerGuard{writerP};
    PairWriter& writer = *writerP;

    auto readFull = [&](uint8_t* p, size_t n) -> bool {
        size_t got = 0;
        while (got < n)
        {
            size_t r = fread(p + got, 1, n - got, stdin);
            if (r == 0)
                return false;
            got += r;
        }
        return true;
    };

    uint64_t emitted = 0;
    bool first = true;
    while (readFull(inFrame.data(), frameBytes))
    {
        // a new clip (the host set the cut event before it sent this frame): nothing of the clip before may reach
        // its tweens, so the feature is made again and this frame takes a first frame's warm-up
        if (!first && cutEv && WaitForSingleObject(cutEv, 0) == WAIT_OBJECT_0)
        {
            LOG("new clip: DLSS-G's feature made again, this frame warms it up; the clip before it:\n");
            fg.logPan();
            if (!fg.newClip())
            {
                LOG("new clip: the DLSS-G feature did not come back\n");
                return true;
            }
            first = true;
        }
        if (first)
        {
            // a clip's first frame is evaluated three times: the frames after it depend on how many evaluates came
            // before, and three is the count every route's frames are pinned to
            for (int i = 0; i < 3; i++)
                if (!fg.frame(inFrame.data(), nullptr))
                {
                    LOG("warmup evaluate failed\n");
                    return true;
                }
            first = false;
            continue;
        }
        if (!fg.frame(inFrame.data(), outPtrs.data()))
            return true;
        // the pair to the writer (after the previous one is written), then this thread takes the other buffer set
        const std::vector<const uint8_t*> outSrc(outPtrs.begin(), outPtrs.end());
        writer.post(outSrc, frameBytes);
        outFrames.swap(outFramesW);
        outPtrs.swap(outPtrsW);
        emitted += genFrames;
    }
    writer.waitIdle();
    fg.logPan();
    LOG("server done, %llu interpolated frames\n", (unsigned long long)emitted);
    fg.shutdown();
    rc = 0;
    return true;
}

static int refused()
{
    LOG("DLSS frame generation did not start on this system (NGX refused it: see the lines above)\n");
    return 2;
}

int wmain(int argc, wchar_t** argv)
{
    setvbuf(stderr, nullptr, _IONBF, 0);
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    CHECK_HR(CoInitializeEx(nullptr, COINIT_MULTITHREADED));

    if (argc >= 4 && wcscmp(argv[1], L"--server") == 0)
    {
        W = (uint32_t)_wtoi(argv[2]);
        H = (uint32_t)_wtoi(argv[3]);
        if (W < 128 || H < 128 || W > 8192 || H > 8192)
        {
            LOG("bad size %ux%u\n", W, H);
            return 1;
        }
        int genFrames = 1;
        bool hdr10 = false;
        HANDLE cutEv = nullptr;
        for (int i = 4; i < argc; i++)
        {
            if (wcscmp(argv[i], L"--gen") == 0 && i + 1 < argc)
                genFrames = _wtoi(argv[++i]);
            else if (wcscmp(argv[i], L"--hdr10") == 0)
                hdr10 = true;
            else if (wcscmp(argv[i], L"--cut-event") == 0 && i + 1 < argc)
                cutEv = (HANDLE)(uintptr_t)_wcstoui64(argv[++i], nullptr, 10);
        }
        if (genFrames < 1 || genFrames > 5)
        {
            LOG("--gen must be 1..5 (got %d)\n", genFrames);
            return 1;
        }
        int rc = 1;
        return runServer(genFrames, hdr10, cutEv, rc) ? rc : refused();
    }
    if (argc >= 4 && wcscmp(argv[1], L"--stage1") == 0)
    {
        W = (uint32_t)_wtoi(argv[2]);
        H = (uint32_t)_wtoi(argv[3]);
        if (W < 128 || H < 128 || W > 8192 || H > 8192)
        {
            LOG("bad size %ux%u\n", W, H);
            return 1;
        }
        int genFrames = 1;
        bool hdr10 = false, bgra = false;
        std::wstring tag, from;
        for (int i = 4; i < argc; i++)
        {
            if (wcscmp(argv[i], L"--gen") == 0 && i + 1 < argc)
                genFrames = _wtoi(argv[++i]);
            else if (wcscmp(argv[i], L"--hdr10") == 0)
                hdr10 = true;
            else if (wcscmp(argv[i], L"--bgra") == 0)
                bgra = true;
            else if (wcscmp(argv[i], L"--share") == 0 && i + 1 < argc)
                tag = argv[++i];
            else if (wcscmp(argv[i], L"--from") == 0 && i + 1 < argc)
                from = argv[++i];
        }
        if (genFrames < 1 || genFrames > 5 || tag.empty() || tag.size() > 40 || from.size() > 40 || from == tag)
        {
            LOG("--stage1 needs --gen 1..5 (got %d), --share TAG (1..40 chars) and an optional --from TAG of another "
                "stage\n",
                genFrames);
            return 1;
        }
        int rc = 1;
        return ngxStage1Stdin(W, H, genFrames, hdr10, bgra, tag, from, ngxFailLever(), rc) ? rc : refused();
    }

    LOG("usage: dlssg2f.exe --server W H [--gen N] [--hdr10] [--cut-event HANDLE]\n"
        "       dlssg2f.exe --stage1 W H [--gen N] [--hdr10 | --bgra] --share TAG [--from TAG]\n");
    return 1;
}
