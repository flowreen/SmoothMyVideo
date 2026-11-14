// A live DLSS frame generation stage on NGX's own calls (ngx_direct.h): the header its readers map, the ring of shared
// textures it writes, another stage's ring as its input, and the loop that turns inputs into frames shared at their
// times. dlssg2f --stage1 runs it as a process fed on stdin (ngxStage1Stdin), the live host on a thread of its own
// (ngxStageMain). Included after ngx_direct.h (so after pan_vectors.h and a LOG macro).
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <fcntl.h>
#include <functional>
#include <io.h>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// milliseconds on the performance counter (GetTickCount64 steps in ~16 ms)
static double qpcMs()
{
    static LARGE_INTEGER f{};
    if (!f.QuadPart)
        QueryPerformanceFrequency(&f);
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)f.QuadPart;
}

// The header a reader maps (smv_dg1_TAG_hdr). A slot's copy is complete once the shared fence reaches its slotFence;
// slotSeq = the present's number since the ring started (a reader that finds a newer number lost that slot). Version 2
// adds wantGen / gen: the live host asks a running stage for another multiple, applied in place at its next present;
// version 3 adds the ask kGenOff = 1x (DLSS-G off, each input passed through as it is; gen 0). wantKeep / keep come
// after them and the version stays 3 (a host that knows version 3 reads a stage with them; a stage without them
// leaves keep 0): a host that shows only every K-th frame of this stage (a non-whole multiple) asks wantKeep = K, and
// a stage on NGX's own calls then makes and shares only those frames
static const uint32_t kGenOff = 0xFFFFFFFFu;
static const uint32_t kSlotStills = 96;          // still rectangles a slot's motion holds
static const uint32_t kMotionMagic = 0x4D564D53; // 'SMVM': the header's motion block is written
struct Stage1Hdr
{
    uint32_t magic;   // 'SMV1'
    uint32_t version; // 3
    volatile uint64_t frames;
    uint32_t w, h, ring, fmt;
    uint64_t allocSize;
    volatile uint32_t status;    // 0 starting, 1 running, 2 ended, 3 failed
    volatile uint32_t lastInput; // the source frame presented last (frame 0 = the warm-up)
    volatile uint64_t slotFence[16];
    volatile uint64_t slotSeq[16];
    volatile int64_t slotQpc[16];    // QueryPerformanceCounter when the capture saw the present
    volatile uint32_t slotInput[16]; // lastInput at that moment
    volatile uint32_t wantGen;       // the host's ask: numFramesToGenerate from the next present (0 = none)
    volatile uint32_t gen;           // the stage's numFramesToGenerate now
    volatile uint32_t wantKeep;      // the host's ask: only every wantKeep-th frame is shown (0 = none asked)
    volatile uint32_t keep;          // the keep the stage runs, 1 = every frame; 0 = a stage that takes no keep
    // after them, the version still 3: a host whose display times the frames it presents (a sync interval a frame)
    // asks a stage on NGX's own calls for each frame as soon as it is made
    volatile uint32_t wantAtOnce; // the host's ask: 1 = a frame is shared as soon as it is made, 0 = at its time
    volatile uint32_t atOnce;     // what the stage does now; stays 0 on a stage that takes no such ask
    // after them, the version still 3: each slot's MOTION, for a stage that takes this ring as its input (its own
    // vectors: pairMotion). pos = the frame's place on the first stage's input timeline (that stage's input n = n,
    // its tween k of g + 1 between inputs n - 1 and n = n - 1 + k / (g + 1)); iv = the input interval the frame lies
    // in (pos in (iv - 1, iv]) and that interval's pan (pan_vectors.h's Result, one input's motion, current ->
    // previous) with its still rectangles; slotStillN past kSlotStills = more than fit: that slot's vectors stay zero.
    // motionMagic = the block is written (a reader of a shorter header, or one without it, takes zero vectors)
    volatile uint32_t motionMagic;
    volatile uint32_t slotIv[16];
    volatile double slotPos[16];
    volatile float slotMx[16], slotMy[16];
    volatile uint32_t slotPanOn[16];
    volatile uint32_t slotStillN[16];
    D3D12_RECT slotStill[16][kSlotStills];
};

// One ring frame's motion as the stage after it reads it (Stage1Hdr's motion block)
struct SlotMotion
{
    double pos = 0;
    uint32_t iv = 0;
    panvec::Result pan; // the pan of input interval iv
};

// The vectors of a stage whose input pair is the ring frames a -> b (the newer): each input interval's pan times the
// share of it between them (an interval a skipped slot hid takes b's pan); an interval in that span without a pan, or
// a move under the estimator's own minimum (*under set), = zero vectors, as the first stage would record; the still
// rectangles = b's interval's
static panvec::Result pairMotion(const SlotMotion& a, const SlotMotion& b, bool* under = nullptr)
{
    panvec::Result r;
    if (under)
        *under = false;
    if (!(b.pos > a.pos) || b.iv < a.iv)
        return r;
    double mx, my;
    if (b.iv == a.iv)
    {
        if (!b.pan.on)
            return r;
        mx = b.pan.mx * (b.pos - a.pos);
        my = b.pan.my * (b.pos - a.pos);
    }
    else
    {
        if (!a.pan.on || !b.pan.on)
            return r;
        const double fa = (double)a.iv - a.pos, fb = b.pos - (double)a.iv;
        mx = a.pan.mx * fa + b.pan.mx * fb;
        my = a.pan.my * fa + b.pan.my * fb;
    }
    if (mx * mx + my * my < panvec::Estimator::kMinMove * panvec::Estimator::kMinMove)
    {
        if (under)
            *under = true;
        return r;
    }
    r.on = true;
    r.mx = (float)mx;
    r.my = (float)my;
    r.still = b.pan.still;
    return r;
}

// What a stage shares with its readers, each by name (smv_dg1_TAG_...): a ring of textures resting in COMMON (the
// state another device's queue may assume), the fence their copies complete on, the header, and the event a publish
// sets
struct StageShare
{
    Stage1Hdr* hdr = nullptr;
    int slots = 12; // the ring's textures (the header's arrays hold 16)
    ComPtr<ID3D12Resource> ring[16];
    ComPtr<ID3D12Fence> fence;
    HANDLE ev{};
    uint64_t count = 0; // frames shared so far: the next one's number
    std::vector<HANDLE> handles;
    std::atomic<uint64_t> frames{0};
    HANDLE wake = nullptr; // the named event a publish sets

    // the ring of `ringSlots` textures of w x h, its fence and the header on dev; ready() then lets the readers in
    bool share(ID3D12Device* dev, DXGI_FORMAT fmt, uint32_t w, uint32_t h, int gen, int ringSlots,
               const std::wstring& tag)
    {
        slots = ringSlots;
        D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rd.Width = w;
        rd.Height = h;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.Format = fmt;
        rd.SampleDesc = {1, 0};
        for (int i = 0; i < slots; i++)
        {
            HANDLE sh = nullptr;
            const std::wstring nm = L"smv_dg1_" + tag + L"_ring" + std::to_wstring(i);
            if (FAILED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED, &rd, D3D12_RESOURCE_STATE_COMMON,
                                                    nullptr, IID_PPV_ARGS(&ring[i]))) ||
                FAILED(dev->CreateSharedHandle(ring[i].Get(), nullptr, GENERIC_ALL, nm.c_str(), &sh)))
            {
                LOG("stage1: ring texture %d not shared\n", i);
                return false;
            }
            handles.push_back(sh);
        }
        HANDLE fh = nullptr;
        const std::wstring fn = L"smv_dg1_" + tag + L"_fence";
        if (FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence))) ||
            FAILED(dev->CreateSharedHandle(fence.Get(), nullptr, GENERIC_ALL, fn.c_str(), &fh)))
        {
            LOG("stage1: fence not shared\n");
            return false;
        }
        handles.push_back(fh);
        ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        // set at every publish: a stage that takes this ring as its input waits on it (smv_dg1_TAG_wake)
        wake = CreateEventW(nullptr, FALSE, FALSE, (L"smv_dg1_" + tag + L"_wake").c_str());
        if (wake)
            handles.push_back(wake);
        const std::wstring mn = L"smv_dg1_" + tag + L"_hdr";
        HANDLE mh = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, sizeof(Stage1Hdr), mn.c_str());
        if (!mh)
            return false;
        handles.push_back(mh);
        hdr = (Stage1Hdr*)MapViewOfFile(mh, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(Stage1Hdr));
        if (!hdr)
            return false;
        memset((void*)hdr, 0, sizeof(Stage1Hdr));
        hdr->version = 3;
        hdr->gen = (uint32_t)gen;
        hdr->w = w;
        hdr->h = h;
        hdr->ring = (uint32_t)slots;
        hdr->fmt = (uint32_t)fmt;
        hdr->allocSize = dev->GetResourceAllocationInfo(0, 1, &rd).SizeInBytes;
        hdr->motionMagic = kMotionMagic;
        return true;
    }

    // the header's magic, written last: a reader waits for it
    void ready()
    {
        MemoryBarrier();
        hdr->magic = 0x31564D53; // 'SMV1'
    }

    // the ring's next frame is in its slot (number % slots) once the fence reaches value; qpc = when it was shared,
    // input = the source frame it belongs to, mot = its motion for the stage after this one
    void publish(uint64_t value, int64_t qpc, uint32_t input, const SlotMotion& mot)
    {
        const int s = (int)(count % slots);
        hdr->slotFence[s] = value;
        hdr->slotSeq[s] = count;
        hdr->slotQpc[s] = qpc;
        hdr->slotInput[s] = input;
        hdr->slotPos[s] = mot.pos;
        hdr->slotIv[s] = mot.iv;
        hdr->slotPanOn[s] = mot.pan.on ? 1u : 0u;
        hdr->slotMx[s] = mot.pan.mx;
        hdr->slotMy[s] = mot.pan.my;
        const size_t ns = mot.pan.still.size();
        hdr->slotStillN[s] = ns <= kSlotStills ? (uint32_t)ns : kSlotStills + 1;
        for (size_t i = 0; i < ns && i < kSlotStills; i++)
            hdr->slotStill[s][i] = mot.pan.still[i];
        MemoryBarrier();
        hdr->frames = ++count;
        frames++;
        if (wake)
            SetEvent(wake);
    }

    void finish(uint32_t status)
    {
        if (hdr)
            hdr->status = status;
    }

    // a stage that goes away while running has ended: its readers see it in the header they still map
    ~StageShare()
    {
        if (hdr && hdr->status == 1)
            hdr->status = 2;
        for (HANDLE x : handles)
            CloseHandle(x);
        if (ev)
            CloseHandle(ev);
        if (hdr)
            UnmapViewOfFile(hdr);
    }
};

// Another stage's ring as a stage's input (--stage1 --from TAG), opened by name on this device (the live host's
// cascade probe does the same) and read in present order: a slot already reused is skipped to the newest
struct UpstreamRing
{
    HANDLE map = nullptr;
    const Stage1Hdr* hdr = nullptr;
    ComPtr<ID3D12Resource> ring[16];
    ComPtr<ID3D12Fence> fence;
    uint32_t n = 0;
    uint64_t nextSeq = 0, skipped = 0;
    HANDLE wake = nullptr; // the upstream stage's publish event; null = it has none, its ring is polled
    bool motion = false;   // the upstream header carries the slots' motion (Stage1Hdr's motion block)

    bool open(ID3D12Device* dev, const std::wstring& tag, DXGI_FORMAT fmt, uint32_t w, uint32_t h)
    {
        // the upstream stage may still be starting (the live host starts every stage at once): its header's magic,
        // written last, is waited for up to 15 s. The whole mapping is viewed: a stage built before the motion block
        // shares a shorter header
        const std::wstring mn = L"smv_dg1_" + tag + L"_hdr";
        const ULONGLONG until = GetTickCount64() + 15000;
        for (;;)
        {
            map = OpenFileMappingW(FILE_MAP_READ, FALSE, mn.c_str());
            hdr = map ? (const Stage1Hdr*)MapViewOfFile(map, FILE_MAP_READ, 0, 0, 0) : nullptr;
            if ((hdr && hdr->magic == 0x31564D53) || GetTickCount64() > until)
                break;
            if (hdr)
                UnmapViewOfFile(hdr);
            if (map)
                CloseHandle(map);
            hdr = nullptr;
            map = nullptr;
            Sleep(20);
        }
        if (!hdr || hdr->magic != 0x31564D53 || hdr->version < 1 || hdr->version > 3 || hdr->ring > 16)
        {
            LOG("--from %ls: no stage header\n", tag.c_str());
            return false;
        }
        if (hdr->w != w || hdr->h != h || hdr->fmt != (uint32_t)fmt)
        {
            LOG("--from %ls: shares %ux%u format %u, %ux%u format %u wanted\n", tag.c_str(), hdr->w, hdr->h, hdr->fmt,
                w, h, (uint32_t)fmt);
            return false;
        }
        n = hdr->ring;
        MEMORY_BASIC_INFORMATION mbi{};
        motion = VirtualQuery(hdr, &mbi, sizeof(mbi)) && mbi.RegionSize >= sizeof(Stage1Hdr) &&
                 hdr->motionMagic == kMotionMagic;
        if (!motion)
            LOG("--from %ls: its header carries no slot motion: zero vectors\n", tag.c_str());
        wake = OpenEventW(SYNCHRONIZE, FALSE, (L"smv_dg1_" + tag + L"_wake").c_str());
        for (uint32_t i = 0; i <= n; i++)
        {
            HANDLE sh = nullptr;
            const std::wstring nm = L"smv_dg1_" + tag + (i < n ? L"_ring" + std::to_wstring(i) : L"_fence");
            const bool ok = SUCCEEDED(dev->OpenSharedHandleByName(nm.c_str(), GENERIC_ALL, &sh)) &&
                            (i < n ? SUCCEEDED(dev->OpenSharedHandle(sh, IID_PPV_ARGS(&ring[i])))
                                   : SUCCEEDED(dev->OpenSharedHandle(sh, IID_PPV_ARGS(&fence))));
            if (sh)
                CloseHandle(sh);
            if (!ok)
            {
                LOG("--from %ls: %ls not opened\n", tag.c_str(), nm.c_str());
                return false;
            }
        }
        return true;
    }

    // the next shared present in order: its slot, fence value and motion (false = none yet; no motion block = the
    // motion stays zero); a slot the writer took again while it was read counts as skipped
    bool next(uint32_t& s, uint64_t& fv, SlotMotion& mot)
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
            mot = SlotMotion();
            if (motion)
            {
                mot.pos = hdr->slotPos[s];
                mot.iv = hdr->slotIv[s];
                const uint32_t ns = hdr->slotStillN[s];
                mot.pan.on = hdr->slotPanOn[s] != 0 && ns <= kSlotStills;
                mot.pan.mx = hdr->slotMx[s];
                mot.pan.my = hdr->slotMy[s];
                if (mot.pan.on)
                    mot.pan.still.assign(&hdr->slotStill[s][0], &hdr->slotStill[s][0] + ns);
                MemoryBarrier();
                if (hdr->slotSeq[s] != seq)
                {
                    skipped++;
                    nextSeq = seq + 1;
                    continue;
                }
            }
            nextSeq++;
            return true;
        }
        return false;
    }

    ~UpstreamRing()
    {
        if (wake)
            CloseHandle(wake);
        if (hdr)
            UnmapViewOfFile(hdr);
        if (map)
            CloseHandle(map);
    }
};

// The source's frame interval, which a pair's frames are spaced over: the mean of the last four gaps between inputs
// (a 24 fps video on a 60 Hz desktop arrives 33 and 50 ms apart), at most 1.25 x the last gap (a source that sped up
// is followed at once). One gap far off the mean stays out of it (a dropped frame, a stall); a second one like it is
// the source's new rate.
struct SourceClock
{
    double win[4] = {};
    int n = 0, at = 0;
    double held = 0; // a gap far off the mean, waiting for a second like it

    double mean() const
    {
        double s = 0;
        for (int i = 0; i < n; i++)
            s += win[i];
        return n ? s / n : 0;
    }
    void push(double gap)
    {
        win[at] = gap;
        at = (at + 1) % 4;
        n = (std::min)(n + 1, 4);
    }
    // the interval for the pair an input closes, gap = the time since the input before it
    double add(double gap)
    {
        const double m = mean();
        if (n && (gap > 1.6 * m || gap < 0.6 * m))
        {
            if (held > 0 && fabs(gap - held) <= 0.3 * held)
            {
                n = at = 0;
                push(held);
                push(gap);
                held = 0;
            }
            else
                held = gap;
        }
        else
        {
            push(gap);
            held = 0;
        }
        return (std::min)(mean(), 1.25 * gap);
    }
};

// A stage's own inputs as CPU frames of the stage's size and format, one handed over at a time
struct StageFeed
{
    HANDLE ev = nullptr; // set at every hand-over and at the end
    // 1 = a frame (its bytes, the stage's until done(), and when it arrived on qpcMs's clock), 0 = none yet, -1 = no
    // more will come
    virtual int take(const uint8_t*& p, double& at) = 0;
    virtual void done() = 0;
    virtual ~StageFeed() = default;
};

// The stage's loop on a Direct that is up (stageInit done) and a ring that is shared and ready. Its inputs come from
// `feed`, or, with feed null, from the ring `up` of another stage until *upEnd is set or that stage ends. An input is
// evaluated the moment it arrives and its frames are shared one by one at their places in the source's interval
// (SourceClock), the first as soon as it is made: the cadence a frame generation pacer gives a game. No clock runs
// inside the frames, so an input gap needs no reset. Returns 0 when the inputs ended (the ring's status 2, unless no
// input ever came), 1 when an input's frames could not be made (status 3 once the stage ran).
static int ngxStageRun(ngxfg::Direct& fg, StageShare& ring, StageFeed* feed, UpstreamRing* up,
                       const std::atomic<bool>* upEnd)
{
    const uint8_t* cpu = nullptr;
    uint32_t upSlot = 0;
    uint64_t upValue = 0;
    // THE MOTION a frame carries to the next stage (Stage1Hdr's motion block): the input taken last (taken), the
    // input evaluated now (cur) and the one before it (prev). Fed CPU frames, this stage is the first: its input n
    // sits at n, its interval's pan = what it measured (fg.lastPan); fed a ring, each input's motion is its slot's,
    // and the vectors of the pair prev -> cur come from them (pairMotion)
    SlotMotion taken, cur, prev;
    bool haveCur = false, havePrev = false;
    uint64_t fedNo = 0; // CPU inputs taken (the warm-up's = 0)
    // the stat window's inputs: evaluated with vectors on, all of them, those whose derived move stayed under the
    // estimator's minimum (a stage fed a ring), and the vectors' summed length (pixels)
    uint64_t statVec = 0, statVecInputs = 0, statUnder = 0;
    double statVecLen = 0;
    // the next input if one is there: at = when it arrived; ended = none will come
    auto takeInput = [&](double& at, bool& ended) -> bool {
        ended = false;
        if (feed)
        {
            const int r = feed->take(cpu, at);
            ended = r < 0;
            if (r > 0)
            {
                taken = SlotMotion();
                taken.pos = (double)fedNo;
                taken.iv = (uint32_t)fedNo++;
            }
            return r > 0;
        }
        if (up->next(upSlot, upValue, taken))
        {
            if (!up->motion) // an upstream without slot motion: one input interval a frame, no pan
            {
                taken.pos = haveCur ? cur.pos + 1.0 : 0.0;
                taken.iv = (uint32_t)taken.pos;
            }
            at = qpcMs();
            return true;
        }
        ended = (upEnd && *upEnd) || up->hdr->status >= 2;
        return false;
    };
    // the input taken last into the frame texture with its vectors; a CPU frame evaluated again (the warm-up) stays
    // the same input
    auto input = [&]() -> bool {
        const bool fresh = !haveCur || taken.pos != cur.pos;
        panvec::Result r;
        bool under = false;
        if (feed)
        {
            if (!fg.stageInput(cpu))
                return false;
            if (fresh)
                taken.pan = fg.lastPan;
            r = fg.lastPan;
        }
        else
        {
            if (haveCur && fresh)
                r = pairMotion(cur, taken, &under);
            if (!fg.stageInput(nullptr, up->ring[upSlot].Get(), up->fence.Get(), upValue, &r))
                return false;
        }
        if (fresh)
        {
            prev = cur;
            havePrev = haveCur;
            cur = taken;
            haveCur = true;
            statVecInputs++;
            statVec += r.on ? 1 : 0;
            statUnder += under ? 1 : 0;
            statVecLen += r.on ? std::sqrt((double)r.mx * r.mx + (double)r.my * r.my) : 0.0;
        }
        return true;
    };
    // the motion of this stage's frame at `pos` between prev and cur: the interval it lies in, with that interval's pan
    auto motionAt = [&](double pos) -> SlotMotion {
        SlotMotion m = havePrev && cur.iv > prev.iv && pos <= (double)prev.iv ? prev : cur;
        m.pos = pos;
        return m;
    };
    // a clip's first input three times, each with its evaluates (--server's rule): the frame texture is left for
    // stageEnd
    auto warm = [&]() -> bool {
        for (int i = 0; i < 3; i++)
        {
            if (!(i && !feed ? fg.stageInput(nullptr) : input()))
                return false;
            for (int k = 1; k <= fg.gen; k++)
                if (!fg.stageTween(k, nullptr))
                    return false;
            if (i < 2 && !fg.stageEnd(nullptr))
                return false;
        }
        return true;
    };

    // set when an input arrives: the feed's hand-over, or the upstream stage's publish (null = an upstream without
    // that event: its ring is polled)
    const HANDLE inputEv = feed ? feed->ev : up->wake;
    double at = 0;
    bool ended = false;
    while (!takeInput(at, ended))
    {
        if (ended)
            return 0;
        if (inputEv)
            WaitForSingleObject(inputEv, 250);
        else
            SwitchToThread();
    }
    if (!warm() || !fg.stageEnd(nullptr))
    {
        LOG("warmup evaluate failed\n");
        return 1;
    }
    if (feed)
        feed->done();
    ring.hdr->status = 1;

    // an evaluate queued and not handled yet: complete at the fence's `value`. An input's first one sets the input's
    // time base when it completes; a frame to share goes out `after` ms behind that base (the input's first tween:
    // 0, as soon as it is complete). share = false: an evaluate whose frame the keep leaves out (the pair's last
    // index, which DLSS-G needs)
    struct Made
    {
        uint64_t value;
        double after;
        bool first;
        uint32_t input;
        bool share;
        SlotMotion mot; // its motion for the stage after this one
    };
    std::deque<Made> made;
    // THE KEEP (the host's ask, wantKeep): the host shows only every keep-th frame of this stage, so only those are
    // made and shared. wouldNo numbers the frames a stage without a keep shares (an input's tweens, then the input):
    // frame n is kept when n % keep == 0, the frames a host that filters by itself shows. A pair's tweens need not
    // all be evaluated: DLSS-G returns the same frame for an index whichever others were asked, as long as the
    // pair's last index (index == count) is evaluated, and evaluated last: its history moves on at that call
    uint64_t wouldNo = 0;
    int keep = 1;
    bool atOnce = false; // the host's ask wantAtOnce: its display times the frames, each goes out as soon as it is made
    uint64_t slotNo = 0; // frames written into the ring, shared or not
    auto slot = [&]() { return ring.ring[slotNo++ % ring.slots].Get(); };
    auto shareIt = [&](const Made& f) {
        LARGE_INTEGER t;
        QueryPerformanceCounter(&t);
        ring.publish(f.value, t.QuadPart, f.input, f.mot);
    };
    SourceClock clock;
    double lastAt = at, base = 0;
    bool rewarm = false; // DLSS-G saw none of the inputs passed through at 1x: the next one starts a clip
    uint64_t inputs = 1, statInputs = 0, statFrames = 0;
    ULONGLONG statTick = GetTickCount64();
    // an input: its generated frames and itself into the ring, each with its place in the source's interval
    auto take = [&]() -> bool {
        const double interval = clock.add(at - lastAt);
        lastAt = at;
        // the host's ask for another multiple (a plan stepping up or down): the count is an argument of every
        // evaluate, so it changes in place; kGenOff = 1x, the inputs shared as they are
        const uint32_t ask = ring.hdr->wantGen;
        const int askGen = ask == kGenOff ? 0 : (int)ask;
        if (ask && askGen != fg.gen)
        {
            if ((uint32_t)askGen <= (fg.maxGen ? fg.maxGen : 5u))
            {
                LOG("stage1: %dx from input %llu on (asked)\n", askGen + 1, (unsigned long long)inputs);
                rewarm = fg.gen == 0;
                fg.gen = askGen;
                ring.hdr->gen = (uint32_t)askGen;
            }
            else
            {
                LOG("stage1: %dx asked, refused: stays at %dx\n", askGen + 1, fg.gen + 1);
                ring.hdr->wantGen = 0;
            }
        }
        const uint32_t wk = ring.hdr->wantKeep;
        if (wk >= 1 && wk <= 6 && (int)wk != keep)
        {
            static const char* const which[] = {"",
                                                "every frame",
                                                "every 2nd frame",
                                                "every 3rd frame",
                                                "every 4th frame",
                                                "every 5th frame",
                                                "every 6th frame"};
            LOG("stage1: %s made and shared from input %llu on (asked)\n", which[wk], (unsigned long long)inputs);
            keep = (int)wk;
            ring.hdr->keep = wk;
        }
        const bool wa = ring.hdr->wantAtOnce != 0;
        if (wa != atOnce)
        {
            LOG("stage1: frames shared %s from input %llu on (asked)\n", wa ? "as soon as made" : "at their times",
                (unsigned long long)inputs);
            atOnce = wa;
            ring.hdr->atOnce = wa ? 1u : 0u;
        }
        const uint32_t no = (uint32_t)inputs++;
        ring.hdr->lastInput = no;
        uint64_t v = 0;
        // the input's frame j (1 .. gen: its tweens, gen + 1: the input itself) is one the keep shows
        auto kept = [&](int j) { return (wouldNo + (uint64_t)(j - 1)) % (uint64_t)keep == 0; };
        if (fg.gen > 0 && rewarm)
        {
            fg.resetNext = true;
            const bool sh = kept(1);
            if (!warm() || !(v = fg.stageEnd(sh ? slot() : nullptr)))
                return false;
            rewarm = false;
            wouldNo++;
            made.push_back({v, 0, true, no, sh, cur});
        }
        else
        {
            if (!input())
                return false;
            const double step = atOnce ? 0.0 : interval / (fg.gen + 1);
            bool first = true;
            for (int k = 1; k <= fg.gen; k++)
            {
                const bool sh = kept(k);
                if (!sh && k != fg.gen)
                    continue;
                if (!(v = fg.stageTween(k, sh ? slot() : nullptr)))
                    return false;
                const double pos = havePrev ? prev.pos + (cur.pos - prev.pos) * k / (fg.gen + 1) : cur.pos;
                made.push_back({v, (k - 1) * step, first, no, sh, motionAt(pos)});
                first = false;
            }
            const bool sh = kept(fg.gen + 1);
            if (!(v = fg.stageEnd(sh ? slot() : nullptr)))
                return false;
            made.push_back({v, fg.gen * step, first, no, sh, cur});
            wouldNo += (uint64_t)fg.gen + 1;
        }
        if (feed)
            feed->done();
        return true;
    };

    // CREATE_WAITABLE_TIMER_HIGH_RESOLUTION: a wait ends within about half a millisecond of its time
    const HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, 0x00000002, TIMER_ALL_ACCESS);
    const HANDLE gpuEv = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    int rc = 0;
    for (;;)
    {
        // share the frames whose time has come, in order
        double wait = -1; // ms until the first waiting frame's time; -1 = none waits for a time
        bool gpuWait = false;
        while (!made.empty())
        {
            Made& f = made.front();
            if (ring.fence->GetCompletedValue() < f.value)
            {
                gpuWait = true;
                break;
            }
            const double now = qpcMs();
            if (f.first)
            {
                base = now;
                f.first = false;
            }
            if (f.share && now < base + f.after)
            {
                wait = base + f.after - now;
                break;
            }
            if (f.share)
                shareIt(f);
            made.pop_front();
        }
        if (takeInput(at, ended))
        {
            // frames still waiting: the source came sooner than its interval said, they go out now
            for (const Made& f : made)
                if (f.share)
                    shareIt(f);
            made.clear();
            if (!take())
            {
                LOG("stage1: an input's frames were not made\n");
                rc = 1;
                break;
            }
            const ULONGLONG tick = GetTickCount64();
            if (tick - statTick >= 5000)
            {
                const double s = (tick - statTick) / 1000.0;
                const uint64_t f = ring.frames, n = inputs - statInputs;
                char under[48] = "";
                if (up)
                    sprintf_s(under, ", %llu under 1 px", (unsigned long long)statUnder);
                LOG("stage1: %.1f inputs/s, %.1f shared presents/s (%.2f an input)%s, pan vectors on %llu of %llu "
                    "inputs (mean %.1f px%s)\n",
                    n / s, (f - statFrames) / s, n ? (double)(f - statFrames) / n : 0.0,
                    up ? (", " + std::to_string(up->skipped) + " upstream frames skipped").c_str() : "",
                    (unsigned long long)statVec, (unsigned long long)statVecInputs,
                    statVec ? statVecLen / (double)statVec : 0.0, under);
                statTick = tick;
                statInputs = inputs;
                statFrames = f;
                statVec = statVecInputs = statUnder = 0;
                statVecLen = 0;
            }
            continue;
        }
        if (ended)
            break;
        // until an input, the GPU or the next frame's time (the end of a stage fed by a ring shows at a wait's
        // timeout); the last half millisecond before a frame's time is spun
        if (!inputEv)
            SwitchToThread();
        else if (gpuWait)
        {
            ring.fence->SetEventOnCompletion(made.front().value, gpuEv);
            const HANDLE hs[2] = {inputEv, gpuEv};
            WaitForMultipleObjects(2, hs, FALSE, 100);
        }
        else if (wait < 0)
            WaitForSingleObject(inputEv, 250);
        else if (wait > 0.6 && timer)
        {
            LARGE_INTEGER due;
            due.QuadPart = -(LONGLONG)((wait - 0.4) * 10000.0);
            SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE);
            const HANDLE hs[2] = {inputEv, timer};
            WaitForMultipleObjects(2, hs, FALSE, 250);
        }
        else
            YieldProcessor();
    }
    if (timer)
        CloseHandle(timer);
    if (gpuEv)
        CloseHandle(gpuEv);
    if (rc)
    {
        ring.finish(3);
        return rc;
    }
    for (const Made& f : made)
        if (f.share)
            shareIt(f);
    // every shared frame's fence value is reached before the stage ends: a reader waits on it
    fg.stageWait(fg.sValue);
    ring.finish(2);
    fg.logPan();
    LOG("stage1 done, %llu inputs, %llu shared presents\n", (unsigned long long)inputs,
        (unsigned long long)ring.frames.load());
    return 0;
}

// A whole stage: NGX's frame generation up for frames of w x h making `gen` frames a pair, its ring of 16 shared as
// `tag`, its inputs from `feed`, or with `from` set from that stage's ring until *upEnd is set; ready(line) once the
// ring is shared (line = the READY handshake dlssg2f prints), then the loop until the inputs end. false = NGX's route
// did not come up and nothing was shared (the reason is logged; failAfterInit treats a route that came up as one that
// did not, the fallback's trigger), true = rc holds the stage's result (0 ended, 1 failed, 3 = a multiple past this
// GPU's limit)
static bool ngxStageMain(uint32_t w, uint32_t h, int gen, bool hdr10, bool bgra, const std::wstring& tag,
                         const std::wstring& from, StageFeed* feed, const std::atomic<bool>* upEnd, bool failAfterInit,
                         const std::function<void(const std::string&)>& ready, int& rc)
{
    ngxfg::Direct fg;
    if (!fg.init(w, h, gen, hdr10, bgra, false))
        return false;
    if (failAfterInit)
    {
        LOG("SMV_DLSSG_NGX=fail: NGX's frame generation treated as unavailable\n");
        fg.shutdown();
        return false;
    }
    rc = 1;
    if (fg.maxGen && (uint32_t)gen > fg.maxGen)
    {
        LOG("multi-frame generation beyond %ux is not supported on this GPU (requested %ux)\n", fg.maxGen + 1, gen + 1);
        rc = 3;
        return true;
    }
    LOG("DLSS-G ready: NGX's own calls\n");
    // 16 slots: an input's frames are made ahead of their times, so up to six slots a reader has not been given yet
    // are already written
    StageShare ring;
    UpstreamRing up;
    if (!ring.share(fg.device.Get(), fg.frameFormat(), w, h, gen, 16, tag) || !fg.stageInit(ring.fence.Get()) ||
        (!from.empty() && !up.open(fg.device.Get(), from, fg.frameFormat(), w, h)))
        return true;
    ring.hdr->keep = 1; // this stage takes a keep (wantKeep)
    ring.ready();
    char line[256];
    snprintf(line, sizeof line, "DLSSG STAGE1 READY gen=%d max=%u share=%ls%s ngx=1", gen, fg.maxGen, tag.c_str(),
             fg.hdr10  ? " hdr10=1"
             : fg.bgra ? " bgra=1"
                       : "");
    ready(line);
    rc = ngxStageRun(fg, ring, from.empty() ? feed : nullptr, from.empty() ? nullptr : &up, upEnd);
    fg.shutdown();
    return true;
}

// stdin read to its end by a thread of its own, started by stdinEnd(): a stage on another stage's ring reads no frames
// there, its input closing is its end
static std::atomic<bool>& stdinEndFlag()
{
    static std::atomic<bool> end{false};
    return end;
}
static std::atomic<bool>& stdinEnd()
{
    std::thread([] {
        char c;
        while (fread(&c, 1, 1, stdin) == 1)
        {
        }
        stdinEndFlag() = true;
    }).detach();
    return stdinEndFlag();
}

// --stage1's frames on stdin, read by a thread of their own: NGX's stage shares a pair's frames at their times while
// the next input arrives. One frame is handed over at a time: the reader waits until the stage took the one before
// (which holds the feeding pipe back, as a blocking read does) and never fills the buffer the stage still works from
struct StdinFrames : StageFeed
{
    std::mutex m;
    std::condition_variable cv;
    std::vector<uint8_t> buf[2];
    int have = -1; // the buffer holding a frame the stage has not taken
    int held = -1; // the buffer the stage works from
    double haveMs = 0;
    bool end = false;

    StdinFrames()
    {
        ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    }

    // the reader holds the frames (self): it outlives its caller, blocked in a read
    static void start(const std::shared_ptr<StdinFrames>& self, size_t bytes)
    {
        self->buf[0].resize(bytes);
        self->buf[1].resize(bytes);
        std::thread([self, bytes] {
            int w = 0;
            for (;;)
            {
                {
                    std::unique_lock<std::mutex> lk(self->m);
                    self->cv.wait(lk, [&] { return self->held != w; });
                }
                size_t got = 0;
                while (got < bytes)
                {
                    const size_t r = fread(self->buf[w].data() + got, 1, bytes - got, stdin);
                    if (r == 0)
                        break;
                    got += r;
                }
                const double t = qpcMs();
                std::unique_lock<std::mutex> lk(self->m);
                if (got < bytes)
                {
                    self->end = true;
                    SetEvent(self->ev);
                    return;
                }
                self->cv.wait(lk, [&] { return self->have < 0; });
                self->have = w;
                self->haveMs = t;
                w ^= 1;
                SetEvent(self->ev);
            }
        }).detach();
    }

    // 1 = a frame (its bytes and when it arrived; done() gives its buffer back), 0 = none yet, -1 = stdin ended
    int take(const uint8_t*& p, double& at) override
    {
        std::lock_guard<std::mutex> lk(m);
        if (have < 0)
            return end ? -1 : 0;
        held = have;
        have = -1;
        p = buf[held].data();
        at = haveMs;
        cv.notify_all();
        return 1;
    }
    void done() override
    {
        {
            std::lock_guard<std::mutex> lk(m);
            held = -1;
        }
        cv.notify_all();
    }
};

// --stage1 as a process (dlssg2f's; the live host's for the identity gates): ngxStageMain fed on stdin, its READY line
// on stdout. false = NGX's route did not come up and stdin was not read
static bool ngxStage1Stdin(uint32_t w, uint32_t h, int gen, bool hdr10, bool bgra, const std::wstring& tag,
                           const std::wstring& from, bool failAfterInit, int& rc)
{
    const auto in = std::make_shared<StdinFrames>();
    auto ready = [&](const std::string& line) {
        if (_setmode(_fileno(stdin), _O_BINARY) == -1)
            LOG("stage1: stdin stays in text mode, its frames would be read wrong\n");
        fprintf(stdout, "%s\n", line.c_str());
        fflush(stdout);
        if (from.empty())
            StdinFrames::start(in, (size_t)w * h * 4);
        else
            stdinEnd();
    };
    return ngxStageMain(w, h, gen, hdr10, bgra, tag, from, in.get(), from.empty() ? nullptr : &stdinEndFlag(),
                        failAfterInit, ready, rc);
}
