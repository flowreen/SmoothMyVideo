// part of smv-live-native.inl (included there, in place); not a standalone translation unit
// ---- the RTX Video SDK CUDA bridge (engine\rtxvideo\rtxvideo_cuda.dll) --------------------
// The TrueHDR bridge is NOT an NGX D3D11 bridge: it is a plain C ABI over CUDA device
// pointers. The NGX feature DLLs
// (nvngx_truehdr.dll, nvngx_vsr.dll) resolve relative to the LOADING module's directory, so
// the bridge is loaded by full path out of its own folder and nothing is copied or split.
struct RtxRect
{
    uint32_t left, top, right, bottom;
};
struct RtxThdrSetting
{
    uint32_t Contrast, Saturation, MiddleGray, MaxLuminance;
};
struct RtxVsrSetting
{
    uint32_t QualityLevel;
}; // 0 bicubic .. 4 Ultra; the host asks --vsr-quality's level, 1..4 (vsrSet, default 4)
typedef void (*PFN_rtxvSetModelPath)(const wchar_t*);
typedef unsigned int (*PFN_rtxCreate)(void*, void*, int, unsigned int, unsigned int);
typedef unsigned int (*PFN_rtxEvalThdr)(void*, void*, RtxRect, RtxRect, RtxThdrSetting*);
typedef unsigned int (*PFN_rtxEvalVsr)(void*, void*, RtxRect, RtxRect, RtxVsrSetting*);
typedef unsigned int (*PFN_rtxEvalVsrAsync)(void*, void*, RtxRect, RtxRect, RtxVsrSetting*, void*);
typedef void (*PFN_rtxShutdown)();

struct RtxBridge
{
    HMODULE mod = nullptr;
    PFN_rtxvSetModelPath setModelPath = nullptr;
    PFN_rtxCreate create = nullptr;
    PFN_rtxEvalThdr evalThdr = nullptr;
    PFN_rtxEvalVsr evalVsr = nullptr; // live RTX VSR: 8-bit BGRA in and out
    // the same eval ordered against the caller's stream by events instead of blocking the thread (optional: an
    // older bridge lacks it and every caller keeps evalVsr)
    PFN_rtxEvalVsrAsync evalVsrAsync = nullptr;
    PFN_rtxShutdown shutdown = nullptr;
    bool created = false;
    bool vsrMade = false; // the one create held RTX VSR's feature (an upscaler switch back to RTX VSR reuses it)
    std::wstring dir;
};
static RtxBridge g_rtxb;

// Resolve the bridge folder (SMV_RTXVIDEO_DIR, else
// <engine>\rtxvideo) and bind the four entry points. Any failure logs one line naming the
// file.
static bool rtxBridgeLoad()
{
    if (g_rtxb.mod)
        return true;
    std::wstring dir;
    wchar_t ov[MAX_PATH]{};
    if (GetEnvironmentVariableW(L"SMV_RTXVIDEO_DIR", ov, MAX_PATH) && ov[0])
        dir = ov;
    else if (!g_nativeEngineDir.empty())
        dir = g_nativeEngineDir + L"\\rtxvideo";
    if (dir.empty())
    {
        LOG("native: cannot locate the rtxvideo folder\n");
        return false;
    }
    const std::wstring dll = dir + L"\\rtxvideo_cuda.dll";
    HMODULE m = LoadLibraryExW(dll.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!m)
    {
        LOG("native: %s not loadable (err %lu)\n", wideToUtf8(dll).c_str(), GetLastError());
        return false;
    }
    g_rtxb.setModelPath = (PFN_rtxvSetModelPath)GetProcAddress(m, "rtxv_set_model_path");
    g_rtxb.create = (PFN_rtxCreate)GetProcAddress(m, "rtx_video_api_cuda_create");
    g_rtxb.evalThdr = (PFN_rtxEvalThdr)GetProcAddress(m, "rtx_video_api_cuda_evaluate_thdr_deviceptr");
    g_rtxb.evalVsr = (PFN_rtxEvalVsr)GetProcAddress(m, "rtx_video_api_cuda_evaluate_vsr_deviceptr");
    g_rtxb.shutdown = (PFN_rtxShutdown)GetProcAddress(m, "rtx_video_api_cuda_shutdown");
    if (!g_rtxb.setModelPath || !g_rtxb.create || !g_rtxb.evalThdr || !g_rtxb.evalVsr || !g_rtxb.shutdown)
    {
        LOG("native: %s is missing an entry point\n", wideToUtf8(dll).c_str());
        FreeLibrary(m);
        g_rtxb.setModelPath = nullptr;
        g_rtxb.create = nullptr;
        g_rtxb.evalThdr = nullptr;
        g_rtxb.evalVsr = nullptr;
        g_rtxb.shutdown = nullptr;
        return false;
    }
    g_rtxb.evalVsrAsync = (PFN_rtxEvalVsrAsync)GetProcAddress(m, "rtx_video_api_cuda_evaluate_vsr_deviceptr_async");
    g_rtxb.mod = m;
    g_rtxb.dir = dir;
    g_rtxb.setModelPath(dir.c_str()); // NGX APP_PATH, must precede create
    LOG("native: RTX Video bridge %s\n", wideToUtf8(dll).c_str());
    return true;
}

// Offline TrueHDR's light statistics (MaxCLL, MaxFALL, the Dolby Vision L1 triples, the HDR10+
// records), accumulated per output
// frame from k_pqOut's block (hist[1024] u32, misc[4] u32, vSum int64 2^24 fixed point) and written as one
// JSON file the render reads at the finalize: maxcll / maxfall always, the Dolby Vision L1
// triples and the HDR10+ records when asked for (--hdr-dv / --hdr-hp).
static const size_t kThdrStatsBytes = 1024 * 4 + 4 * 4 + 8;
struct ThdrL1
{
    int v[3];
};
struct ThdrHp
{
    int avg;
    int maxscl[3];
    int dist[9];
};
struct ThdrAcc
{
    float lut[1024]{};  // the light of every PQ code (k_pqLut, the kernel's own PQ EOTF)
    int brightCode = 0; // HDR10+: the first code brighter than 100 nits
    double cll = 0.0, fall = 0.0;
    bool raw = false; // --hdr-color raw: the stats read nits, not linear
    bool wantL1 = false, wantHp = false;
    std::vector<ThdrL1> l1;
    std::vector<ThdrHp> hp;
    // --hdr-frames: one line per output frame, appended and flushed as the
    // record lands, "<frame MaxCLL> <frame MaxFALL>[ L <l1 x3>][ H <avg> <maxscl x3> <dist x9>]":
    // a killed render keeps every record of the frames it wrote, and a resumed render reads the
    // banked prefix back from it instead of decoding the banked video
    FILE* lines = nullptr;
    double lastCll = 0.0, lastFall = 0.0;
    bool haveLast = false;
    ThdrAcc() = default;
    ThdrAcc(const ThdrAcc&) = delete;
    ThdrAcc& operator=(const ThdrAcc&) = delete;
    ~ThdrAcc()
    {
        if (lines)
            fclose(lines);
    }

    void writeLine(double c, double f)
    {
        lastCll = c;
        lastFall = f;
        haveLast = true;
        if (!lines)
            return;
        fprintf(lines, "%.17g %.17g", c, f);
        if (wantL1 && !l1.empty())
            fprintf(lines, " L %d %d %d", l1.back().v[0], l1.back().v[1], l1.back().v[2]);
        if (wantHp && !hp.empty())
        {
            const ThdrHp& r = hp.back();
            fprintf(lines, " H %d %d %d %d %d %d %d %d %d %d %d %d %d", r.avg, r.maxscl[0], r.maxscl[1], r.maxscl[2],
                    r.dist[0], r.dist[1], r.dist[2], r.dist[3], r.dist[4], r.dist[5], r.dist[6], r.dist[7], r.dist[8]);
        }
        fprintf(lines, "\n");
        fflush(lines);
    }

    void add(const uint8_t* s, uint64_t n)
    {
        const uint32_t* hist = (const uint32_t*)s;
        const uint32_t* misc = hist + 1024;
        long long vFix = 0;
        memcpy(&vFix, s + 1024 * 4 + 16, 8);
        const double vSum = (double)vFix * (1.0 / 16777216.0);
        float vMax = 0.0f;
        memcpy(&vMax, &misc[3], 4);
        // the frame's largest value and its mean, x 10000 to nits (the mean rounded to fp32
        // first); --hdr-color raw reads nits already
        const double sc = raw ? 1.0 : 10000.0;
        const double fc = (double)vMax * sc;
        if (fc > cll)
            cll = fc;
        const double fa = (double)(float)(vSum / (double)n) * sc;
        if (fa > fall)
            fall = fa;
        if (wantL1)
        {
            // _accum_l1: [amin, mean, amax] of the maxRGB code, * 4095 / 1023, round, clamp
            int mn = -1, mx = 0;
            double sum = 0.0;
            for (int c = 0; c < 1024; c++)
                if (hist[c])
                {
                    if (mn < 0)
                        mn = c;
                    mx = c;
                    sum += (double)hist[c] * c;
                }
            if (mn < 0)
                mn = 0;
            const float k = (float)(4095.0 / 1023.0);
            const float st[3] = {(float)mn * k, (float)(sum / (double)n) * k, (float)mx * k};
            ThdrL1 r{};
            for (int i = 0; i < 3; i++)
            {
                float v = rintf(st[i]);
                r.v[i] = (int)(v < 0.0f ? 0.0f : (v > 4095.0f ? 4095.0f : v));
            }
            l1.push_back(r);
        }
        if (wantHp)
        {
            // _accum_hp: fp32 cumsum, searchsorted (the smallest code with cum >= q * n)
            static const float qs[8] = {0.01f, 0.9998f, 0.25f, 0.50f, 0.75f, 0.90f, 0.95f, 0.99f};
            float cum[1024];
            float acc = 0.0f;
            double lsum = 0.0;
            for (int c = 0; c < 1024; c++)
            {
                acc += (float)hist[c];
                cum[c] = acc;
                lsum += (double)((float)hist[c] * lut[c]);
            }
            const float nf = (float)n;
            int pv[8];
            for (int i = 0; i < 8; i++)
            {
                const float t = qs[i] * nf;
                int c = 0;
                while (c < 1024 && cum[c] < t)
                    c++;
                if (c > 1023)
                    c = 1023;
                pv[i] = (int)rintf(lut[c] * 100000.0f);
            }
            ThdrHp r{};
            for (int c = 0; c < 3; c++)
                r.maxscl[c] = (int)rintf(lut[misc[c] > 1023u ? 1023u : misc[c]] * 100000.0f);
            r.avg = (int)nearbyint((double)(float)lsum / (double)n * 100000.0);
            const double bright = (double)n - (brightCode > 0 ? (double)cum[brightCode - 1] : 0.0);
            const int d[9] = {pv[0], pv[1], (int)nearbyint(1000.0 * bright / (double)n), pv[2], pv[3], pv[4], pv[5],
                              pv[6], pv[7]};
            memcpy(r.dist, d, sizeof(d));
            hp.push_back(r);
        }
        writeLine(fc, fa);
    }

    // a held slot that carries the previous frame's finished bytes (DLSS 5 on): the
    // same picture, so the same record (MaxCLL / MaxFALL are unchanged by a repeat)
    void repeat()
    {
        if (wantL1 && !l1.empty())
            l1.push_back(l1.back());
        if (wantHp && !hp.empty())
            hp.push_back(hp.back());
        if (haveLast)
            writeLine(lastCll, lastFall);
    }

    bool writeJson(const std::wstring& path) const
    {
        FILE* f = nullptr;
        if (_wfopen_s(&f, path.c_str(), L"wb") || !f)
            return false;
        const int mc = cll > 65535.0 ? 65535 : (int)ceil(cll);
        const int mf = fall > 65535.0 ? 65535 : (int)ceil(fall);
        fprintf(f, "{\"maxcll\": %d, \"maxfall\": %d, \"frames\": %zu", mc, mf, wantL1 ? l1.size() : hp.size());
        if (wantL1)
        {
            fprintf(f, ", \"l1\": [");
            for (size_t i = 0; i < l1.size(); i++)
                fprintf(f, "%s[%d, %d, %d]", i ? ", " : "", l1[i].v[0], l1[i].v[1], l1[i].v[2]);
            fprintf(f, "]");
        }
        if (wantHp)
        {
            fprintf(f, ", \"hp\": [");
            for (size_t i = 0; i < hp.size(); i++)
            {
                const ThdrHp& r = hp[i];
                fprintf(f, "%s{\"avg\": %d, \"maxscl\": [%d, %d, %d], \"dist\": [%d, %d, %d, %d, %d, %d, %d, %d, %d]}",
                        i ? ", " : "", r.avg, r.maxscl[0], r.maxscl[1], r.maxscl[2], r.dist[0], r.dist[1], r.dist[2],
                        r.dist[3], r.dist[4], r.dist[5], r.dist[6], r.dist[7], r.dist[8]);
            }
            fprintf(f, "]");
        }
        fprintf(f, "}\n");
        const bool ok = !ferror(f);
        fclose(f);
        return ok;
    }
};
