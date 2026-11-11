// part of smv-live-native.inl (included there, in place); not a standalone translation unit
// SMV_VRAM_MIB: the GPU's memory as the app's memory checks see it, in MiB (a smaller card's trigger test on a
// card everything fits in); 0 = the GPU's own
static uint64_t nativeVideoCardLever()
{
    const std::string lever = lkEnv("SMV_VRAM_MIB");
    const long long mib = lever.empty() ? 0 : atoll(lever.c_str());
    return mib > 0 ? (uint64_t)mib << 20 : 0;
}

// The GPU's video memory as its driver counts it, every process included (NvAPI_GPU_GetMemoryInfoEx, the figures
// nvidia-smi shows; cudaMemGetInfo and the DXGI budget describe this process alone): cardB = its size, freeB = what
// is free right now, moved = how many allocations the driver has moved out of video memory to make room since the
// system started. The GPU is the CUDA device's (matched by its adapter LUID), the first NVIDIA one when that cannot
// be told. SMV_VRAM_MIB shrinks the card and the free memory with it. false = the driver does not say.
static bool nativeVideoMemory(uint64_t& freeB, uint64_t& cardB, uint64_t* moved = nullptr)
{
    static const NvPhysicalGpuHandle gpu = [] {
        NvPhysicalGpuHandle all[NVAPI_MAX_PHYSICAL_GPUS] = {};
        NvU32 n = 0;
        if (NvAPI_Initialize() != NVAPI_OK || NvAPI_EnumPhysicalGPUs(all, &n) != NVAPI_OK || !n)
            return (NvPhysicalGpuHandle) nullptr;
        LUID want{};
        if (n > 1 && nativeCudaLuid(want))
            for (NvU32 i = 0; i < n; i++)
            {
                NvLogicalGpuHandle lg = nullptr;
                LUID id{};
                NV_LOGICAL_GPU_DATA d = {};
                d.version = NV_LOGICAL_GPU_DATA_VER;
                d.pOSAdapterId = &id;
                if (NvAPI_GetLogicalGPUFromPhysicalGPU(all[i], &lg) == NVAPI_OK &&
                    NvAPI_GPU_GetLogicalGpuInfo(lg, &d) == NVAPI_OK && memcmp(&id, &want, sizeof(id)) == 0)
                    return all[i];
            }
        return all[0];
    }();
    NV_GPU_MEMORY_INFO_EX m = {};
    m.version = NV_GPU_MEMORY_INFO_EX_VER;
    if (!gpu || NvAPI_GPU_GetMemoryInfoEx(gpu, &m) != NVAPI_OK || !m.dedicatedVideoMemory)
        return false;
    cardB = m.dedicatedVideoMemory;
    freeB = m.curAvailableDedicatedVideoMemory;
    const uint64_t lever = nativeVideoCardLever();
    if (lever && lever < cardB)
    {
        freeB = freeB > cardB - lever ? freeB - (cardB - lever) : 0;
        cardB = lever;
    }
    if (moved)
        *moved = m.dedicatedVideoMemoryEvictionCount;
    return true;
}

// The same figures once this process's own last allocations count in them. The driver's free figure trails an
// allocation by a few milliseconds: read right after a session's buffers were made, it still misses part of them
// (170 to 718 MiB of a session's own buffers in the read that followed its init; the figure then fell in steps for 1.2
// to 8.0 ms, pausing up to 3.1 ms on the way, and stayed flat from there on), and a fit sized on that read hands out
// memory the session already holds. So the figure is read every 3 ms until it has not moved by 1 MiB over four reads
// in a row (12 ms), for 100 ms at most (another app allocating meanwhile, or a coarse system timer): the lowest
// figure read is the answer then.
static bool nativeVideoMemorySettled(uint64_t& freeB, uint64_t& cardB)
{
    if (!nativeVideoMemory(freeB, cardB))
        return false;
    uint64_t low = freeB;
    int same = 0;
    for (const int64_t until = nowQpc100() + 1000000; same < 4 && nowQpc100() < until;)
    {
        Sleep(3);
        uint64_t f = 0, c = 0;
        if (!nativeVideoMemory(f, c))
            break;
        same = (f > freeB ? f - freeB : freeB - f) < ((uint64_t)1 << 20) ? same + 1 : 0;
        freeB = f;
        low = (std::min)(low, f);
    }
    if (same < 4)
        freeB = low;
    return true;
}

// The video memory a session may still take, in bytes: what is free less a reserve of 4 % of the card (512 MiB at
// least) for what the desktop and other apps take next. Memory past it would not fail: Windows moves what does not
// fit into system memory, which slows the session and every other app on the GPU. 0 = no limit (--no-gpu-fit,
// SMV_VRAM_CAP=0, or the driver does not report its memory); 1 = nothing is free. settled = the figure once this
// process's last allocations count in it (every fit: it follows the session's own buffers); false = the figure as it
// reads right now, for the present loop, which must not wait and follows no allocation of its own.
static uint64_t nativeVideoMemoryRoomOf(bool settled)
{
    uint64_t freeB = 0, cardB = 0;
    if (g_noGpuFit || lkEnv("SMV_VRAM_CAP") == "0" ||
        !(settled ? nativeVideoMemorySettled(freeB, cardB) : nativeVideoMemory(freeB, cardB)))
        return 0;
    const uint64_t reserve = (std::max)((uint64_t)512 << 20, cardB / 25);
    return freeB > reserve ? freeB - reserve : 1;
}
static uint64_t nativeVideoMemoryRoom()
{
    return nativeVideoMemoryRoomOf(true);
}

// What a live session still takes after its output ring, in bytes. The ring is its last allocation (the engines,
// contexts, RTX Video and DLSS 5's passes exist by then), so only what the first frames create comes after it
// (NVAPI's free figure from the ring's creation to the steady session, a quarter on top of the measured): DLSS 5's
// first frames, 120 MiB + 16.6 a megapixel a pass (157 / 226 / 465 MiB at 1920x1080 with 1 / 3 / 10 passes, 139 / 190
// at 854x480 with 3 / 10: its per-size buffers exist before the ring on this route, so they are not added), and
// FRUC's second midpoint instance with NvOFFRUC's buffers for it, 297 / 353 / 511 / 716 / 1222 MiB at 854x480 /
// 1280x720 / 1920x1080 / 2560x1440 / 3840x2160 (x4, the same in both sessions of a size, from the settled figure at
// the ring's fit; x2 runs one instance and adds 8 to 72 MiB), a quarter on top, as a line by the padded megapixel of
// the model frame that covers every size, 341 + 147, and AMD FSR 4 frame generation's second midpoint instance with its node
// frames and derived level-2+ vectors (fsrfgDeriveOn: the side stream's two vector buffers), 1249 MiB at any size (the
// whole GPU's rise, x4 against x2, at five sizes from 854x480 to 3840x2160: 538 to 999 MiB, 161 to 230 at 2560x1440,
// a quarter on top; the card's figure, not the process's own memory counter, which counts a D3D12 buffer CUDA imports
// twice and reads up to 1.7x the card's rise: a session whose own counter passes the free memory while the card's
// figure fits moves nothing out), or without derived vectors its side Optical Flow session instead of the vector
// buffers (nativeFsrfgAhead, x4 and up, 2 + 98 a megapixel with a quarter less the vectors' 10: 1251 + 88); Intel
// XeSS-FG's, fgKind 1, by the card's figure at the same five sizes: 247 + 265 a megapixel (without derived vectors
// 249 + 353),
// plus, while the bridge gives every instance its own vkd3d device (fsrfg_devices), 300 MiB an instance past the first
// (x4 parked, shared vs own devices: +243 / +49 MiB at 854x480 / 1708x960, x8 at 1280x720 +471 for two: the largest,
// flat, a quarter on top); NVIDIA Video Frame Generation, whose model builds its working memory at its first frames
// and not at its load: 764 / 923 / 1007 / 1102 / 1356 MiB at 854x480 / 1280x720 / 1920x1080 / 2560x1440 / 3840x2160
// (the larger of two sessions a size, the same at x2 and x4: nothing grows with the multiple; from the settled figure
// at the ring's fit), a quarter on top, as a line by the megapixel of the model frame that covers every size, 1085 +
// 84; the other routes add nothing the reserve does not cover (NVIDIA Optical Flow's first run: 40 MiB at 3840x2160).
// The x8 /
// x16 / x32 ladders' third, fourth and fifth tree levels (an instance and twice the node frames a level) come on top
// when the session's slots reach them (FRUC: the whole GPU's rise over x4 at five sizes, 854x480 to 3840x2160 with the
// fit off, a quarter on top, less the ring's slots past x32's and the pipelined second tree priced below, as a line by
// the padded megapixel that covers every size and never falls with depth: x8 884 + 224, x16 985 + 378, x32 1207 + 784,
// x64 1678 + 879, x128 1873 + 1363, x256 1873 + 2234 a megapixel). x512 (level 9): x128 to x256's step twice over,
// raised to cover its measures (FRUC 854x480 / 1280x720: 1873 + 3977). AMD FSR 4 frame generation's levels
// come from the whole GPU's rise over x2 at five sizes, 854x480 to 3840x2160 (x2 varies by up to ~330 MiB run to run,
// so a size takes the largest of its runs; 3840x2160 to x64 with the fit off): a level's term = that rise, a quarter on
// top, less the late term and the ring's slots past x32's, as a line that covers every size and never falls with depth
// (x8 155 + 86, x16 441 + 260, x32 441 + 626, x64 636 + 1023, x128 996 + 1712, x256 1134 + 3245 a megapixel; x512 by
// the step rule above, raised to cover its 854x480 measure: 1555 + 6309). Intel XeSS-FG's levels by the same measure
// and rule from its own rows (3840x2160 to x64, 2560x1440 to x128, with the fit off): 362 + 103, 362 + 511, 362 + 928,
// 362 + 1506, 362 + 2589, 596 + 4063, x512 1275 + 7011 a megapixel.
constexpr double kNrLiveLateBase = 150.0, kFrucLateBase = 341.0, kFrucLateMp = 147.0;
constexpr double kVfgLateBase = 1085.0, kVfgLateMp = 84.0;
constexpr double kFsrfgLateBase = 1251.0, kFsrfgLateMp = 88.0, kFsrfgLateDerBase = 1249.0, kFsrfgLateDerMp = 0.0,
                 kFsrfgOwnDevice = 300.0;
constexpr double kXefgLateBase = 249.0, kXefgLateMp = 353.0, kXefgLateDerBase = 247.0, kXefgLateDerMp = 265.0;
constexpr double kXefgDeepBase[] = {362.0, 362.0, 362.0, 362.0, 362.0, 596.0, 1275.0},
                 kXefgDeepMp[] = {103.0, 511.0, 928.0, 1506.0, 2589.0, 4063.0, 7011.0};
constexpr double kFrucDeepBase[] = {884.0, 985.0, 1207.0, 1678.0, 1873.0, 1873.0, 1873.0},
                 kFrucDeepMp[] = {224.0, 378.0, 784.0, 879.0, 1363.0, 2234.0, 3977.0};
constexpr double kFsrfgDeepBase[] = {155.0, 441.0, 441.0, 636.0, 996.0, 1134.0, 1555.0},
                 kFsrfgDeepMp[] = {86.0, 260.0, 626.0, 1023.0, 1712.0, 3245.0, 6309.0};
static_assert(sizeof(kFrucDeepBase) == sizeof(double) * (kFrDepth - 2) &&
                  sizeof(kFrucDeepMp) == sizeof(kFrucDeepBase) && sizeof(kFsrfgDeepBase) == sizeof(kFrucDeepBase) &&
                  sizeof(kFsrfgDeepMp) == sizeof(kFrucDeepBase) && sizeof(kXefgDeepBase) == sizeof(kFrucDeepBase) &&
                  sizeof(kXefgDeepMp) == sizeof(kFrucDeepBase),
              "one memory term a tree level from x8 to the deepest");

// the deepest midpoint-tree level `slots` output frames a source frame reach (x2^L needs 2^L slots; 2 at least, the
// session's instances at most); 0 = no tree
static int nativeTreeLevels(const NativeRife& nr, uint32_t slots)
{
    if (!nr.fruc || !nr.frMp)
        return 0;
    int lv = 2;
    while (lv < kFrDepth && slots >= (2u << lv))
        lv++;
    return (std::min)(nr.frInstMax, lv);
}

static uint64_t nativeLiveLateNeed(const NativeRife& nr, uint32_t slots)
{
    const double mib = 1048576.0;
    double need = 0.0;
    if (nr.liveNr && nr.nrHost)
    {
        const double mp = (double)nr.nrW * nr.nrH / 1e6;
        const int p = g_nrPasses < 1 ? 1 : (g_nrPasses > nr::kMaxPasses ? nr::kMaxPasses : g_nrPasses);
        need += kNrLiveLateBase + nr::kFrameMp * mp * p;
    }
    if (nr.fruc)
    {
        const double mp = (double)nr.pw * nr.ph / 1e6;
        const bool xe = nr.fsrfg && nr.fgKind == 1;
        if (nr.vfg)
            need += kVfgLateBase + kVfgLateMp * mp;
        else if (!nr.fsrfg)
            need += kFrucLateBase + kFrucLateMp * mp;
        else if (fsrfgDeriveOn(nr))
            need += xe ? kXefgLateDerBase + kXefgLateDerMp * mp : kFsrfgLateDerBase + kFsrfgLateDerMp * mp;
        else
            need += xe ? kXefgLateBase + kXefgLateMp * mp : kFsrfgLateBase + kFsrfgLateMp * mp;
        // the deepest ladder the slots and the instances reach: x8 = 3 levels, x16 = 4, .. x512 = 9
        const int levels = nativeTreeLevels(nr, slots);
        if (nr.frMp && levels > 2)
            need += !nr.fsrfg ? kFrucDeepBase[levels - 3] + kFrucDeepMp[levels - 3] * mp
                    : xe      ? kXefgDeepBase[levels - 3] + kXefgDeepMp[levels - 3] * mp
                              : kFsrfgDeepBase[levels - 3] + kFsrfgDeepMp[levels - 3] * mp;
        // an own vkd3d device for every FSR 4 instance past the first (x4 = 2 instances, x2^L = L)
        if (nr.fsrfg && g_fsrfgK[nr.fgKind].devices && g_fsrfgK[nr.fgKind].devices() == 1)
            need += ((std::max)(levels, (std::min)(2, nr.frInstMax)) - 1) * kFsrfgOwnDevice;
        // a pipelined group's second tree: the shown pair's nodes beside the new pair's (BGRA8, 4 bytes a pixel)
        if (nr.frMp && !nr.fsrfg)
            need += ((1 << levels) - 1) * 4.0 * mp * 1e6 / mib;
    }
    return (uint64_t)(need * mib);
}

// The output ring's slots the free video memory holds: the room less what the session takes after the ring, over the
// ring's two sets of slotBytes; 2 at least (the session then runs anyway, and says `video memory ran out` when the
// driver moves memory out). ringB = the bytes the ring may take. 0 = no limit (SMV_VRAM_CAP=0, or the driver does not
// report its memory). slots = the ring's asked slots (the deepest midpoint tree they reach is priced).
static uint32_t nativeRingFit(const NativeRife& nr, size_t slotBytes, uint32_t slots, uint64_t& ringB)
{
    ringB = 0;
    const uint64_t room = nativeVideoMemoryRoom();
    if (!room || !slotBytes)
        return 0;
    // a midpoint tree's level the room cannot hold beside the ring's slots for it is given up, one level at a time:
    // the slots then stop below the next power of two, so the ladder never reaches a level that was not priced
    uint32_t s = slots;
    for (;;)
    {
        const uint64_t late = nativeLiveLateNeed(nr, s);
        ringB = room > late ? room - late : 0;
        const uint64_t n = (std::min)((uint64_t)s, ringB / (2ull * slotBytes));
        const int lv = nativeTreeLevels(nr, s);
        if (lv <= 2 || n >= (1ull << lv))
        {
            if (s < slots)
            {
                uint32_t top = 2;
                while ((uint64_t)top * 2 <= n)
                    top *= 2;
                const uint32_t up = top * 2;
                LOG("native: video memory: the midpoint tree stops at x%u (x%u with its ring slots needs about %.0f "
                    "MiB, %.0f MiB are free beside the session)\n",
                    top, up, (double)(nativeLiveLateNeed(nr, up) + 2ull * up * slotBytes) / 1048576.0,
                    (double)room / 1048576.0);
            }
            return (uint32_t)(n < 2 ? 2 : (n > 4096 ? 4096 : n));
        }
        s = (1u << lv) - 1;
    }
}

// A session's video memory check, for every route: nativeVideoMemoryMark at its start (before the engines and
// buffers), nativeVideoMemoryNote once its first frames are out. One line when the driver moved allocations out of
// video memory meanwhile: the session did not fit in what was free, so part of it, or of another app, now lives in
// system memory and runs slower there (the pixels are the same). A live session keeps allocating after its first
// frames (AMD FSR 4 frame generation's second instance and deeper tree levels, a ladder the throttle raises), so when
// the first check found nothing, nativeVideoMemoryLate repeats it at every 2 s stat line until the line is said once.
static uint64_t g_vramMoved = 0, g_vramFree = 0;
static bool g_vramMarked = false, g_vramLate = false;
static void nativeVideoMemoryMark()
{
    uint64_t cardB = 0;
    g_vramLate = false;
    g_vramMarked = nativeVideoMemory(g_vramFree, cardB, &g_vramMoved);
}
// true = the driver moved memory out since the mark (the line said)
static bool nativeVideoMemoryCheck()
{
    uint64_t freeB = 0, cardB = 0, moved = 0;
    if (!nativeVideoMemory(freeB, cardB, &moved) || !(moved > g_vramMoved || (nativeVideoCardLever() && !freeB)))
        return false;
    const double gb = 1024.0 * 1024.0 * 1024.0;
    LOG("%s: video memory ran out: %.1f GB of the GPU's %.1f GB were free when this %s started and it needs "
        "more, so the driver moved memory out to system memory, which slows it and other apps down (the pixels "
        "stay the same); a smaller picture, a lower DLSS mode, fewer DLSS 5 passes or closing other apps leave "
        "more room\n",
        g_offline ? "offline" : "native", (double)g_vramFree / gb, (double)cardB / gb,
        g_offline ? "render" : "session");
    return true;
}
static void nativeVideoMemoryNote()
{
    if (!g_vramMarked)
        return;
    g_vramMarked = false;
    g_vramLate = !nativeVideoMemoryCheck() && !g_offline;
}
static void nativeVideoMemoryLate()
{
    if (g_vramLate && nativeVideoMemoryCheck())
        g_vramLate = false;
}
