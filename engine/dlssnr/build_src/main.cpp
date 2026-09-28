// dlssnr - the DLSS 5 Neural Rendering host for SmoothMyVideo.
//
// A pipe server around the reusable core in nr_host.cpp, shaped exactly like
// engine\dlssg\dlssg2f.exe so engine\dlssnr.py can drive it the way
// engine\dlssg.py drives that one.
//
// Modes:
//   dlssnr.exe --server W H [--structure F] [--tone F] [variant flags]
//       streaming: raw RGBA16F frames (W*H*8 bytes) on stdin, one processed
//       frame of the same size per input frame on stdout, in order. Prints
//       "DLSSNR READY w=W h=H" on stdout once the feature is up. All logging
//       goes to stderr, stdout is binary only. EOF on stdin exits.
//   dlssnr.exe --probe [W H] [--frames N] [variant flags]
//       phase 0: bring the feature up on a synthetic frame, evaluate N frames
//       and print one PROBE line with the NGX result and the per frame ms. One
//       variant per process on purpose: a refused NGX init can leave the driver
//       core in a state the next attempt would inherit, so the variant table is
//       driven by running this exe once per row.
//
// Variant flags (phase 0 only, the shipped default is the first of each):
//   --shim / --no-shim          route the NGX entry points through nvngx.dll
//   --shim-loc N                0 beside the exe, 1 <exe>\caller, 2 beside the snippet
//   --init-order N              0 (device, featureInfo, version), 1 (device, version, featureInfo)
//   --no-projectid              use Init_Ext instead of Init_ProjectID
//
// Exit codes: 0 ok, 1 usage or IO error, 2 runtime missing or GPU/driver
// unsupported, 3 CreateFeature(18) refused.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <io.h>
#include <fcntl.h>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "nr_host.h"

#define LOG(...)                                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        fprintf(stderr, __VA_ARGS__);                                                                                  \
        fflush(stderr);                                                                                                \
    } while (0)

// stdout carries the binary frame protocol, but the NGX core prints status lines of its
// own to the process stdout (e.g. "[ngx::util::openLogFileInPath] Logging to
// requested file ... enabled successfully" ahead of the READY line, which corrupts the
// stream). So the ORIGINAL stdout is kept as a private handle for the protocol and the
// process stdout (the Win32 std handle AND CRT fd 1) is pointed at stderr before any NGX
// module loads: every log line from any module then lands on stderr, where the engine
// already streams it.
static HANDLE g_out = INVALID_HANDLE_VALUE;
static void detachStdout()
{
    DuplicateHandle(GetCurrentProcess(), GetStdHandle(STD_OUTPUT_HANDLE), GetCurrentProcess(), &g_out, 0, FALSE,
                    DUPLICATE_SAME_ACCESS);
    fflush(stdout);
    SetStdHandle(STD_OUTPUT_HANDLE, GetStdHandle(STD_ERROR_HANDLE));
    _dup2(_fileno(stderr), _fileno(stdout));
}
static bool outWrite(const void* p, size_t n)
{
    const uint8_t* b = (const uint8_t*)p;
    while (n)
    {
        DWORD wrote = 0;
        const DWORD chunk = (DWORD)(n > (1u << 30) ? (1u << 30) : n);
        if (!WriteFile(g_out, b, chunk, &wrote, nullptr) || wrote == 0)
            return false;
        b += wrote;
        n -= wrote;
    }
    return true;
}
static bool outPrintf(const char* fmt, ...)
{
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    const int n = vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    return n > 0 && outWrite(buf, (size_t)n);
}

static uint16_t floatToHalf(float f)
{
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0)
        return (uint16_t)sign;
    if (exp >= 31)
        return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

static void usage()
{
    LOG("usage: dlssnr.exe --server W H [--structure F] [--tone F] [--reset-every]\n");
    LOG("       dlssnr.exe --probe [W H] [--frames N]\n");
    LOG("       variant flags: --shim | --no-shim, --shim-loc N, --init-order N, --no-projectid\n");
}

int main(int argc, char** argv)
{
    bool server = false, probe = false;
    // --reset-every: void the feature's temporal history on EVERY evaluate, making the pass a
    // pure function of the frame handed in. The live routes ask for it because this host binds
    // no motion vectors, depth or jitter (see nr_host.cpp), so kept history has nothing valid
    // to reproject by and identical frames come back out different.
    bool resetEvery = false;
    uint32_t w = 1920, h = 1080;
    int frames = 10;
    nr::Settings set;
    nr::Variant var;

    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        auto needNum = [&](float& dst) {
            if (i + 1 < argc)
                dst = (float)atof(argv[++i]);
        };
        if (a == "--server" && i + 2 < argc)
        {
            server = true;
            w = (uint32_t)strtoul(argv[++i], nullptr, 10);
            h = (uint32_t)strtoul(argv[++i], nullptr, 10);
        }
        else if (a == "--probe")
        {
            probe = true;
            if (i + 2 < argc && argv[i + 1][0] != '-' && argv[i + 2][0] != '-')
            {
                w = (uint32_t)strtoul(argv[++i], nullptr, 10);
                h = (uint32_t)strtoul(argv[++i], nullptr, 10);
            }
        }
        else if (a == "--frames" && i + 1 < argc)
            frames = atoi(argv[++i]);
        else if (a == "--reset-every")
            resetEvery = true;
        else if (a == "--structure")
            needNum(set.structure);
        else if (a == "--tone")
            needNum(set.tone);
        else if (a == "--intensity")
            needNum(set.intensity);
        else if (a == "--style" && i + 1 < argc)
            set.style = atoi(argv[++i]);
        else if (a == "--preset" && i + 1 < argc)
            set.preset = atoi(argv[++i]);
        else if (a == "--shim")
            var.useShim = true;
        else if (a == "--no-shim")
            var.useShim = false;
        else if (a == "--shim-loc" && i + 1 < argc)
            var.shimLocation = atoi(argv[++i]);
        else if (a == "--init-order" && i + 1 < argc)
            var.initArgOrder = atoi(argv[++i]);
        else if (a == "--no-projectid")
            var.initProjectId = false;
        else if (a == "--via-core")
            var.viaSnippet = false;
        else
        {
            LOG("dlssnr: unknown argument %s\n", a.c_str());
            usage();
            return 1;
        }
    }
    if (!server && !probe)
    {
        usage();
        return 1;
    }
    if (w == 0 || h == 0 || frames <= 0)
    {
        usage();
        return 1;
    }
    detachStdout(); // before the first NGX module loads, see g_out

    LOG("dlssnr: %ux%u structure=%.2f tone=%.2f style=%d preset=%d shim=%d shimLoc=%d initOrder=%d projectId=%d resetEvery=%d\n",
        w, h, set.structure, set.tone, set.style, set.preset, (int)var.useShim, var.shimLocation, var.initArgOrder,
        (int)var.initProjectId, (int)resetEvery);

    nr::Host host;
    std::string err;
    const int rc = host.startup(w, h, set, var, err);
    LOG("dlssnr: core=%ls\n", host.corePath().empty() ? L"(none)" : host.corePath().c_str());
    LOG("dlssnr: snippet=%ls\n", host.snippetPath().empty() ? L"(none)" : host.snippetPath().c_str());
    LOG("dlssnr: shim=%ls\n", host.shimPath().empty() ? L"(none)" : host.shimPath().c_str());
    if (rc != 0)
    {
        LOG("dlssnr: startup failed (%s)\n", err.c_str());
        if (probe)
            outPrintf("PROBE shim=%d shimLoc=%d initOrder=%d projectId=%d ngx=%s exit=%d reason=%s\n", (int)var.useShim,
                      var.shimLocation, var.initArgOrder, (int)var.initProjectId,
                      nr::resultString(host.lastResult()).c_str(), rc, err.c_str());
        return rc;
    }

    const size_t frameBytes = (size_t)w * h * 8;
    std::vector<uint8_t> in(frameBytes), out(frameBytes);

    if (probe)
    {
        // Synthetic frame: a diagonal gradient with a little structure, enough
        // for the runtime to have something to work on.
        uint16_t* px = (uint16_t*)in.data();
        for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < w; ++x)
            {
                const float u = (float)x / (float)w, vv = (float)y / (float)h;
                const float checker = (((x >> 4) + (y >> 4)) & 1) ? 0.15f : 0.0f;
                uint16_t* p = px + ((size_t)y * w + x) * 4;
                p[0] = floatToHalf(u + checker);
                p[1] = floatToHalf(vv + checker);
                p[2] = floatToHalf(0.5f * (u + vv));
                p[3] = floatToHalf(1.0f);
            }

        // One warm-up frame, then the timed run.
        if (!host.renderFrame(in.data(), out.data(), true, err))
        {
            LOG("dlssnr: warm-up evaluate failed (%s)\n", err.c_str());
            outPrintf("PROBE shim=%d shimLoc=%d initOrder=%d projectId=%d result=%s exit=2\n", (int)var.useShim,
                      var.shimLocation, var.initArgOrder, (int)var.initProjectId,
                      nr::resultString(host.lastResult()).c_str());
            fflush(stdout);
            ExitProcess(2);
        }
        LARGE_INTEGER freq, t0, t1;
        QueryPerformanceFrequency(&freq);
        QueryPerformanceCounter(&t0);
        for (int i = 0; i < frames; ++i)
            if (!host.renderFrame(in.data(), out.data(), false, err))
            {
                LOG("dlssnr: evaluate failed at frame %d (%s)\n", i, err.c_str());
                outPrintf("PROBE shim=%d shimLoc=%d initOrder=%d projectId=%d result=%s exit=2\n", (int)var.useShim,
                          var.shimLocation, var.initArgOrder, (int)var.initProjectId,
                          nr::resultString(host.lastResult()).c_str());
                fflush(stdout);
                ExitProcess(2);
            }
        QueryPerformanceCounter(&t1);
        const double ms = 1000.0 * (double)(t1.QuadPart - t0.QuadPart) / (double)freq.QuadPart / frames;

        // Did the runtime actually change the frame?
        double diff = 0.0;
        for (size_t i = 0; i < frameBytes; ++i)
            diff += abs((int)out[i] - (int)in[i]);
        outPrintf("PROBE shim=%d shimLoc=%d initOrder=%d projectId=%d result=Success exit=0 "
                  "size=%ux%u ms_per_eval=%.3f byte_diff_mean=%.4f\n",
                  (int)var.useShim, var.shimLocation, var.initArgOrder, (int)var.initProjectId, w, h, ms,
                  diff / (double)frameBytes);
        fflush(stdout);
        // NGX has no clean unload; its own process-detach teardown faults after
        // main returns (misleading exit 127). The probe is done, so exit now.
        ExitProcess(0);
    }

    // Streaming server. stdin is binary; the frames go out on the private g_out handle
    // (the original stdout, see detachStdout), never through the CRT stdout.
    _setmode(_fileno(stdin), _O_BINARY);
    if (!outPrintf("DLSSNR READY w=%u h=%u\n", w, h))
    {
        LOG("dlssnr: stdout write failed\n");
        return 1;
    }

    bool first = true;
    for (;;)
    {
        size_t got = 0;
        while (got < frameBytes)
        {
            const size_t n = fread(in.data() + got, 1, frameBytes - got, stdin);
            if (n == 0)
                break;
            got += n;
        }
        if (got == 0)
            break; // clean EOF between frames
        if (got < frameBytes)
        {
            LOG("dlssnr: short frame (%zu of %zu bytes)\n", got, frameBytes);
            return 1;
        }

        if (!host.renderFrame(in.data(), out.data(), first || resetEvery, err))
        {
            LOG("dlssnr: evaluate failed (%s)\n", err.c_str());
            return 2;
        }
        first = false;
        if (!outWrite(out.data(), frameBytes))
        {
            LOG("dlssnr: stdout write failed\n");
            return 1;
        }
    }
    LOG("dlssnr: stdin EOF, exiting\n");
    fflush(stdout);
    ExitProcess(0); // skip NGX process-detach teardown fault
    return 0;
}
