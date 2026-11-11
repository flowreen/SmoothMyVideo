// part of smv-live-native.inl (included there, in place); not a standalone translation unit
// NVIDIA Video Frame Generation (vfg): NVIDIA's two-frame video interpolator, the "VideoFrameGeneration" effect of its
// Video Effects runtime, on the FRUC route: nativeFrucSetup's surfaces and nativeFrucPair's packs feed it, and the
// functions below take over the setup, the tween and the release when nr.vfg. The runtime is four files of NVIDIA's,
// user-installed in one folder (the session's frucDir): NVVideoEffects.dll (the C API), NVCVImage.dll (its image
// descriptor), nvVFXVideoFrameGeneration.dll (the model) and the cudart64_12.dll they import. They are opened there by
// full path and called through GetProcAddress, so the host builds without NVIDIA's SDK. The effect runs on the
// session's stream in the host's context: it reads two device surfaces and writes a third, one call a tween at
// any time between the pair, and keeps nothing from one pair to the next. It takes every surface's bytes as R G B A,
// whatever pixel format the descriptor names (B G R A bytes give the model red and blue swapped: another picture),
// so the route packs and unpacks a vfg session's surfaces in that order. Its other image family is RGB10A2, one
// 32-bit word a pixel with red in the low ten bits, then green, blue and two alpha bits; all three surfaces of a
// session are of one family. A session on HDR planes uses that one (nr.vfR10): the planes' codes go in at 10 bits as
// they are stored, PQ or HLG, and the tween comes back at 10 bits.
//
// Pinned: the model's highest quality mode, and NVIDIA's automatic shot-change answer off (with it on, a pair it
// takes for a cut gets a copy of a real frame in place of a tween).

// NVCVImage.dll's image descriptor (64 bytes on x64), field for field
struct VfgImage
{
    unsigned width, height;
    int pitch;
    int pixelFormat, componentType;
    unsigned char pixelBytes, componentBytes, numComponents, planar, gpuMem, colorspace, reserved[2];
    void* pixels;
    void* deletePtr;
    void (*deleteProc)(void*);
    unsigned long long bufferBytes;
};
static_assert(sizeof(VfgImage) == 64 && offsetof(VfgImage, pixels) == 32, "NVCVImage.dll's image descriptor");
// the descriptor's values for an interleaved image in CUDA device memory: RGBA8, or RGB10A2 in its one packed 32-bit
// component (4 bytes a pixel either way)
constexpr int kVfgRgba = 6, kVfgU8 = 1, kVfgRgb10a2 = 13, kVfgP32 = 11;
constexpr unsigned kVfgInterleaved = 0, kVfgCuda = 1;
constexpr unsigned kVfgModeHigh = 2;

struct VfgRuntime
{
    HMODULE image = nullptr, effects = nullptr, model = nullptr;
    int (*imageInit)(VfgImage*, unsigned, unsigned, int, void*, int, int, unsigned, unsigned) = nullptr;
    const char* (*errorString)(int) = nullptr;
    int (*create)(const char*, void**) = nullptr;
    void (*destroy)(void*) = nullptr;
    int (*setU32)(void*, const char*, unsigned) = nullptr;
    int (*setF32)(void*, const char*, float) = nullptr;
    int (*setImage)(void*, const char*, VfgImage*) = nullptr;
    int (*setStream)(void*, const char*, CUstream) = nullptr;
    int (*load)(void*) = nullptr;
    int (*run)(void*, int) = nullptr;
    // the tween's two frames and its result, described again before every run (the surfaces rotate)
    VfgImage img[3] = {};
};
static VfgRuntime g_vfg;

// the runtime's three DLLs from the session's folder, once a process; each brings its imports from that folder
static bool nativeVfgLoad(const std::string& dir)
{
    if (g_vfg.effects)
        return true;
    const std::wstring wdir = utf8ToWide(dir);
    HMODULE mods[3] = {};
    const wchar_t* names[3] = {L"NVCVImage.dll", L"NVVideoEffects.dll", L"nvVFXVideoFrameGeneration.dll"};
    for (int i = 0; i < 3; i++)
    {
        const std::wstring p = wdir + L"\\" + names[i];
        mods[i] = LoadLibraryExW(p.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (!mods[i])
        {
            LOG("native: vfg: %ls did not load from %s (err %lu)\n", names[i], dir.c_str(), GetLastError());
            for (int k = i - 1; k >= 0; k--)
                FreeLibrary(mods[k]);
            return false;
        }
    }
    VfgRuntime r;
    r.imageInit = (decltype(r.imageInit))GetProcAddress(mods[0], "NvCVImage_Init");
    r.errorString = (decltype(r.errorString))GetProcAddress(mods[0], "NvCV_GetErrorStringFromCode");
    r.create = (decltype(r.create))GetProcAddress(mods[1], "NvVFX_CreateEffect");
    r.destroy = (decltype(r.destroy))GetProcAddress(mods[1], "NvVFX_DestroyEffect");
    r.setU32 = (decltype(r.setU32))GetProcAddress(mods[1], "NvVFX_SetU32");
    r.setF32 = (decltype(r.setF32))GetProcAddress(mods[1], "NvVFX_SetF32");
    r.setImage = (decltype(r.setImage))GetProcAddress(mods[1], "NvVFX_SetImage");
    r.setStream = (decltype(r.setStream))GetProcAddress(mods[1], "NvVFX_SetCudaStream");
    r.load = (decltype(r.load))GetProcAddress(mods[1], "NvVFX_Load");
    r.run = (decltype(r.run))GetProcAddress(mods[1], "NvVFX_Run");
    if (!r.imageInit || !r.errorString || !r.create || !r.destroy || !r.setU32 || !r.setF32 || !r.setImage ||
        !r.setStream || !r.load || !r.run)
    {
        LOG("native: vfg: NVIDIA's Video Effects runtime in %s lacks an expected export\n", dir.c_str());
        for (HMODULE m : mods)
            FreeLibrary(m);
        return false;
    }
    r.image = mods[0];
    r.effects = mods[1];
    r.model = mods[2];
    g_vfg = r;
    return true;
}

// one runtime call's status: 0 passes, anything else is logged with NVIDIA's own text
static bool nativeVfgOk(int rc, const char* what)
{
    if (rc == 0)
        return true;
    LOG("native: vfg: %s failed: %s (status %d)\n", what, g_vfg.errorString(rc), rc);
    return false;
}

// the effect at the session's size on the session's stream; the calling thread has the device bound. Every failure
// here is a refusal (a GPU the model does not run on answers at the create or the load).
static bool nativeVfgSetup(NativeRife& nr)
{
    if (!nativeVfgLoad(nr.frucDir))
        return false;
    const int64_t t0 = nowQpc100();
    if (!nativeVfgOk(g_vfg.create("VideoFrameGeneration", &nr.vfEff), "the effect's create"))
    {
        nr.vfEff = nullptr;
        return false;
    }
    if (!nativeVfgOk(g_vfg.setStream(nr.vfEff, "CudaStream", (CUstream)nr.stream), "the stream") ||
        !nativeVfgOk(g_vfg.setU32(nr.vfEff, "InputWidth", (unsigned)nr.pw), "the width") ||
        !nativeVfgOk(g_vfg.setU32(nr.vfEff, "InputHeight", (unsigned)nr.ph), "the height") ||
        !nativeVfgOk(g_vfg.setU32(nr.vfEff, "Mode", kVfgModeHigh), "the quality mode") ||
        !nativeVfgOk(g_vfg.setU32(nr.vfEff, "AutomaticShotChangeDetectionEnabled", 0), "the shot-change switch") ||
        !nativeVfgOk(g_vfg.load(nr.vfEff), "the model's load") ||
        !nativeVfgOk(g_vfg.setU32(nr.vfEff, "FrameMultiplier", 0), "the free-time mode"))
        return false;
    LOG("native: vfg session %dx%d, NVIDIA Video Frame Generation (highest quality mode, every pair interpolated), "
        "%s, loaded in %.0f ms\n",
        nr.pw, nr.ph, nr.vfR10 ? "10-bit R10G10B10A2" : "8-bit RGBA", (double)(nowQpc100() - t0) / 1e4);
    return true;
}

// one tween at t (strictly between the pair) of surfaces frA and frB into dFrOutB, then dFrOut (nativeFrucUnpack)
static bool nativeVfgTween(NativeRife& nr, double t)
{
    void* px[3] = {nr.dFrSurf[nr.frA], nr.dFrSurf[nr.frB], nr.dFrOutB};
    const char* slot[3] = {"SrcImage0", "SrcImage1", "DstImage0"};
    const int fmt = nr.vfR10 ? kVfgRgb10a2 : kVfgRgba, comp = nr.vfR10 ? kVfgP32 : kVfgU8;
    for (int i = 0; i < 3; i++)
        if (!nativeVfgOk(g_vfg.imageInit(&g_vfg.img[i], (unsigned)nr.pw, (unsigned)nr.ph, nr.pw * 4, px[i], fmt, comp,
                                         kVfgInterleaved, kVfgCuda),
                         "an image descriptor") ||
            !nativeVfgOk(g_vfg.setImage(nr.vfEff, slot[i], &g_vfg.img[i]), slot[i]))
        {
            nr.die("vfg image binding failed");
            return false;
        }
    if (!nativeVfgOk(g_vfg.setU32(nr.vfEff, "ShotChange", 0), "the shot-change flag") ||
        !nativeVfgOk(g_vfg.setF32(nr.vfEff, "Timestep", (float)t), "the tween's time") ||
        !nativeVfgOk(g_vfg.run(nr.vfEff, 0), "the run"))
    {
        nr.die("vfg run failed");
        return false;
    }
    nr.frTweens++;
    return nativeFrucUnpack(nr, nr.dFrOutB, nr.frPairSdr);
}

// the session's effect; the runtime's DLLs stay loaded for the process (a resident host's next session reuses them)
static void nativeVfgRelease(NativeRife& nr)
{
    if (!nr.vfEff)
        return;
    LOG("native: vfg session: %llu tweens\n", (unsigned long long)nr.frTweens);
    g_vfg.destroy(nr.vfEff);
    nr.vfEff = nullptr;
}
