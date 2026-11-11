// part of smv-live-native.inl (included there, in place); not a standalone translation unit
#define NCHK(call, what)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        const cudaError_t _e = (call);                                                                                 \
        if (_e != cudaSuccess)                                                                                         \
        {                                                                                                              \
            LOG("native: %s failed (cuda %d)\n", what, (int)_e);                                                       \
            return false;                                                                                              \
        }                                                                                                              \
    } while (0)

// gmflow A's outputs (the transformer on the host's kernels, nativeGaRun): qf (the transformer's fp16 stream as it is:
// q and feat = rows 0..1, f = rows 2..3) feeds k_localCorr (which computes the window positions itself) and B, flow0
// goes to B; B takes both beside the correlation
constexpr int kGmAn = 2;
// the quarter-scale local correlation's window (gmflow's radius 4: 9 x 9 taps), k_localCorr's K and gmflow B's
constexpr int kGmCorrK = 81;

// the midpoint tree of Smooth Motion and AMD FSR 4 (one bridge instance a level, the bridges hold 16): its deepest
// level (x512) and its node grid, kFrGrid steps a pair. The ring's memory fit prices the level the slots reach, so a
// GPU without the room for a level never reaches it
constexpr int kFrDepth = 9, kFrGrid = 1 << kFrDepth;
static_assert(kFrDepth <= 9, "a slot's fraction below 0.999f is a tween: x1024's last node (1023 / 1024) is not");
// the groups at a pipelined spell's start whose prices over the direct call before it give the pipelined route's ratios
constexpr int kFrPipeLearn = 8;

// RIFE's feature planes: the encode's output and the IFNet / block0 inputs f0 / f1 (RIFE 4.25: 4)
constexpr int kRifeFeatC = 4;

// a kernel chain on the host's kernels replayed as CUDA graphs (chainGraphRun): up to kN executable graphs, each
// captured on the first call with its key (the chain's buffers, then its w and h), replaced oldest first; mode 0 =
// direct launches, 1 = graphs, 2 = every capture thrown away (the fallback's trigger); logged = the graph line written;
// captures past kMaxCaptures (a chain whose buffers keep changing) = direct launches
struct ChainGraphs
{
    static constexpr int kN = 16, kKey = 10, kMaxCaptures = 4 * kN;
    struct Entry
    {
        const void* key[kKey] = {};
        cudaGraphExec_t exec = nullptr;
    } e[kN];
    int next = 0, mode = 0, captures = 0;
    bool logged = false;
};

// one launch of GMFSS's fusionnet on the host's kernels (fuPlan's step at the same index): its kernel (kFuKern), the
// channels, the PReLU slope before the conv (0 = none), the offsets of its fragment-order weights and fp32 bias in the
// weight block (fuNatLoad)
struct FuOp
{
    int k = 0, cin = 0, cout = 0;
    float slope = 0.0f;
    size_t wOff = 0, bOff = 0;
};

struct NativeRife
{
    // ---- handoff facts (nativeHandoff)
    std::string ifnetPath;
    int ph = 0, pw = 0, w = 0, h = 0, cw = 0, ch = 0;
    int dw = 0, dh = 0, x0 = 0, y0 = 0;
    int batchMax = 1;
    bool identity = false;
    bool hdr = false; // FP16 scRGB capture in, PQ R10A2 ring out
    int dev = 0;      // the CUDA device of the capture adapter (bound per thread)
    // live RTX TrueHDR through the CUDA bridge, once per real frame at capture size
    bool rtxHdr = false;    // the session asked for it and the bridge loaded
    bool rtxFailed = false; // a mid-run eval failed: faithful PQ for the rest of the run
    RtxThdrSetting thdr{};  // Contrast, Saturation, MiddleGray 50, MaxLuminance (thdrPeakNits)
    float sdrScale = 3.0f;  // SDR reference white as an scRGB scale (nits / 80)
    int rtxMode = 0;        // 0 vivid, 1 rtx, 2 raw
    float rtxVib = 0.0f, rtxSb = 0.0f;
    double thdrMs = 0.0, thdrMaxMs = 0.0; // host-side eval cost, reported at teardown
    uint64_t thdrN = 0;
    // live Sharpen and RTX VSR, the app's --sharpen / --rtx-vsr on this route
    float sharpen = 0.0f;   // Adaptive Sharpen strength 0..2 (curve_height) at the presented size (0 = off)
    bool vsrWant = false;   // --rtx-vsr on an SDR session with the bridge and its DLL present
    bool vsr = false;       // ...and the fit enlarges in both axes (decided after the handoff)
    bool vsrFailed = false; // a mid-run eval failed: Lanczos for the rest of the run
    // the live early start runs nativeRtxInit beside the model's init, while the free video memory still misses the
    // model's buffers: Anime4K's and ArtCNN's setups size their buffers to that memory, so they are held there
    // (upHold; their arguments in upHeld) and run once the model's init is done (nativeUpHeldSetup)
    bool upHold = false, upHeldSet = false;
    int upHeld[5] = {};
    RtxVsrSetting vsrSet{4};
    double vsrMs = 0.0, vsrMaxMs = 0.0;
    bool vsrAsync = false; // the evals ran through the bridge's event-ordered export (vsrMs not timed)
    uint64_t vsrN = 0;
    // the shrinking fit: the Lanczos pair (k_fitAaH / V) into the staging frame
    bool fitAa = false; // the fit to (dw, dh) changes the size: the separable Lanczos pair
    // Upscale to: the app's --upscale H = the
    // INTERNAL render size (uw, uh), derived from the exe's own flag. The
    // model frame goes there first (RTX VSR when it enlarges, else Lanczos), then the fit to
    // (dw, dh) like any frame. 0 = off (also when it equals the
    // fit rect: one resize, not two).
    int uw = 0, uh = 0;
    bool upAa = false; // the first resize shrinks the height (uh < h)
    // live Restore: the handoff names the bundled Real-ESRGAN or a --restore-model in AnimeJaNai's
    // SPAN-F3 form, both on the host's kernels; the host runs it on every presented frame before the
    // upscale, then folds the s x output to the first resize target (the
    // rule: back to the model size when RTX VSR follows, else straight to the internal
    // render size or the fit rect). A mid-run failure drops the pass for the rest of the session with
    // one line.
    std::string restorePath;
    bool restore = false; // the handoff named a Restore model (the session asked for it)
    // no-engine mode: the effects-only route (the
    // app's `echo` backend, no model ticked) runs here too. The handoff answers with the
    // geometry only (`engine=none`, plus the restore engine when asked for): no IFNet, no
    // encode, no tween; every group stores its one real frame through the same effects
    // chain. The cubin cache folder comes
    // from `NATIVE-PATH cache=` since no jit path names it.
    bool noEngine = false;
    std::string cachePath;
    // GMFSS: the handoff answers `engine=gmfss` with the half size and the kernel cache; this host runs the whole chain
    // on its own kernels (feat_ext, metricnet, the GMFSS IFNet, fusionnet, gmflow's backbone, A, `_m` and B,
    // nativeGmfssPair / nativeGmfssTween with the glue kernels above): no engine, so gmfss is a native backend like rife.
    bool gmfss = false;
    int hh = 0, hw = 0; // the half frame (the fusion grid)
    // the one shared activation block, gmCtxMemB bytes: feat_ext's, metricnet's, the GMFSS IFNet's, fusionnet's and
    // gmflow's scratch + Restore's (nativeGmShareMemory) or
    // the IFNet's, the encode's, Restore's and DRBA block0's scratch (nativeModelShareMemory)
    void* dGmCtxMem = nullptr;
    int64_t gmCtxMemB = 0;
    // between gmflow's halves: A's outputs (kGmA order), the correlation k_localCorr writes for B, and the shapes it
    // runs at (the quarter grid gmH x gmW, gmK window samples a pixel)
    void* dGmA[kGmAn] = {};
    // A's inputs map4 / map8 [2, 128, ...] in the backbone's dtype: batch slot 0 = the previous frame, 1 = the new one;
    // gmMapSlot = one frame's bytes of each; gmMapsNext = slot 1 holds the frame the next call reads as its previous
    // one, taken on view gmMapsKind (0 the pictures, 1 the SDR view, 2 the motion view)
    void* dGmMap[2] = {};
    size_t gmMapSlot[2] = {};
    bool gmMapsNext = false;
    int gmMapsKind = 0;
    float* dGmCorr = nullptr;
    int gmH = 0, gmW = 0, gmK = 0;
    // between `_m` and A: its outputs mf / mq / mk (fp16 [2, N, 128] each), the matching's flow (fp32 [2, N, 2]) and
    // A's prop input; N = gmN8 tokens on the eighth grid gmW8 wide, gmPropHalf = prop is fp16
    void* dGmM[3] = {};
    float* dGmFlow8 = nullptr;
    void* dGmProp = nullptr;
    int gmN8 = 0, gmW8 = 0;
    bool gmPropHalf = false;
    // the chain's contract, read off the engines in nativeGmfssSetup (never from the handoff
    // line, the 5a rule), and the planar NCHW buffers it sizes from that. Levels: the half
    // (hh, hw), the quarter and the eighth, exactly GMFSS_infer_u's feature pyramid.
    int gmC[3] = {}; // feat_ext channels at the half / quarter / eighth
    bool gmFeatHalf = true, gmMetricHalf = true, gmOutHalf = true, gmFuseHalf = true;
    int gmCur = 0;             // which dGmFeat set holds the CURRENT frame's features
    void* dGmFeat[2][3] = {};  // the two frames' feature sets (cur is the next pair's prev)
    float* dGmHalf = nullptr;  // (6, hh, hw): img0's half in planes 0..2, img1's in 3..5
    float* dGmFlow = nullptr;  // the gmflow output (2, 2, hh, hw) = flow01 then flow10
    void* dGmMetric = nullptr; // (2, hh, hw) in the engine's dtype: m0 then m1
    float* dGmFlowP[2] = {};   // the flow pyramids at the quarter / eighth, pre-scaled
    float* dGmMetP[2] = {};    // the metric pyramids at the quarter / eighth (fp32)
    // the scatter splats' accumulator, max(C) + 1 planes of the half; with gmTail it also holds the cut fusionnet's
    // out (256 fp16 planes, free from the splats' end to the next clear); the gather splats never use it: then only
    // that out (64 planes), or null
    long long* dGmAcc = nullptr;
    // the pair scratch (dGmA, dGmCorr, dGmM, dGmFlow8, dGmProp, dGmEnc, dGmHalfM) carved from dGmAcc, not allocated
    bool gmScratchInAcc = false;
    // the gather splats (k_gs*): two list sets (a level's two directions), each the counts / fill cursor, the list
    // offsets, the entries and the sources' e / dx / dy planes, sized for the half; gmGather false (SMV_GM_SPLAT=old,
    // or a half above the scan's 2048 x 4096 cells) = k_splatSoft + k_splatNorm on dGmAcc
    int* dGsCnt[2] = {};
    int* dGsOff[2] = {};
    int* dGsList[2] = {};
    float* dGsSrc[2] = {};
    int* dGsBlk[2] = {};
    bool gmGather = false;
    bool gmTail = false;    // the cut fusionnet (out = the pre-shuffle tensor): k_fuTail does shuffle + conv_last
    bool gmTailHwc = false; // gmTail with out channels-last (the engine's HWC8 / HWC16): k_fuTailNhwc reads it
    bool gmFeatHwc = false; // feat_ext's levels and fusionnet's b / c / d channels-last: the feature splats' layout
    // feat_ext on the host's kernels (nativeFnModule): its module and six kernels, the weight block (fnOff = per conv
    // 1..5 its fragment-order weights, then its fp32 bias), the six LeakyRelu slopes, the scratch (in the shared
    // activation block, else dFnOwn)
    CUmodule fnMod = nullptr;
    CUfunction fFn[6] = {};
    void* dFnW = nullptr;
    size_t fnOff[10] = {};
    float fnSlope[6] = {};
    void* dFnScr = nullptr;
    void* dFnOwn = nullptr;
    // metricnet on the host's kernels (nativeMnModule): its module and five kernels, the weight block (mnOff, see
    // mnNatLoad), the four PReLU slopes, the scratch (in the shared activation block, else dMnOwn)
    CUmodule mnMod = nullptr;
    CUfunction fMn[5] = {};
    void* dMnW = nullptr;
    size_t mnOff[10] = {};
    float mnSlope[4] = {};
    void* dMnOwn = nullptr;
    // the GMFSS IFNet on the host's kernels (nativeGiModule): its module and kernels, the weight block (giOff, see
    // giNatLoad), the scratch (in the shared activation block, else dGiOwn)
    CUmodule giMod = nullptr;
    CUfunction fGi[11] = {};
    void* dGiW = nullptr;
    std::vector<size_t> giOff;
    void* dGiOwn = nullptr;
    ChainGraphs giGraphs; // its launches as CUDA graphs (nativeGiRun; SMV_GI_GRAPH)
    // fusionnet on the host's kernels (nativeFuModule): its module and kernels, one FuOp a launch, the weight block, the
    // activations (fuArena; in the shared activation block, else dFuOwn)
    CUmodule fuMod = nullptr;
    CUfunction fFu[11] = {};
    int fuR4Min = 0; // the 8-row grid's CTAs from which a PReLU wide conv runs as its 4-row twin (8 an SM; 0 = never)
    void* dFuW = nullptr;
    std::vector<FuOp> fuOps;
    void* dFuOwn = nullptr;
    ChainGraphs fuGraphs; // its launches as CUDA graphs (nativeFuRun; SMV_FU_GRAPH)
    // gmflow's second half B on the host's kernels (nativeGbModule): its module and kernels, the weight block (gbOff, see
    // gbNatLoad), the activations (gbArena; in the shared activation block, else dGbOwn)
    CUmodule gbMod = nullptr;
    CUfunction fGb[6] = {};
    void* dGbW = nullptr;
    size_t gbOff[6] = {};
    void* dGbOwn = nullptr;
    ChainGraphs gbGraphs; // its launches as CUDA graphs (nativeGbRun; SMV_GB_GRAPH)
    // gmflow's backbone on the host's kernels (nativeBbModule): its module and kernels, the weight block (bbOff, see
    // bbNatLoad), the activations (bbArena; in the shared activation block, else dBbOwn)
    CUmodule bbMod = nullptr;
    CUfunction fBb[16] = {};
    void* dBbW = nullptr;
    size_t bbOff[22] = {};
    void* dBbOwn = nullptr;
    ChainGraphs bbGraphs; // its launches as CUDA graphs (nativeBbRun; SMV_BB_GRAPH)
    // gmflow's transformer at both scales on the host's kernels (nativeGaModule): its module and kernels, the weight
    // block (gaOff, see gaNatLoad), the window position tables (dGaPos: A's posY | posX, then `_m`'s), the activations
    // (gaArena; in the shared activation block, else dGaOwn)
    CUmodule gaMod = nullptr;
    CUfunction fGa[7] = {};
    int gaMlp2Min = 0;    // the CTAs from which the mlp runs as k_gaMlp2 (3 an SM; 0 = never: RTX 20, SMV_GA_MLP2=0)
    bool gaAttn2 = false; // gmflow A's attention as k_gaAttn2 (never on RTX 20; SMV_GA_ATTN2=0 = never)
    void* dGaW = nullptr;
    size_t gaOff[46] = {};
    float* dGaPos = nullptr;
    void* dGaOwn = nullptr;
    ChainGraphs gaGraphs[2]; // `_m` (0) and A (1) as CUDA graphs (nativeGaRun; SMV_GA_GRAPH)
    void* dGmWl = nullptr;   // conv_last's weight (3, 64, 3, 3) fp16 (fusionnet.pkl, k_fuTailNhwc reads it)
    void* dGmBl = nullptr;   // its bias (3) fp16
    float gmLr[3] = {-1.0f, -1.0f,
                     -1.0f}; // the heads' first PReLU slopes on b / c / d: the feature splats' stores apply them
    float* dGmFa = nullptr;  // fusionnet a (9, half): I1t, merged (ifnet writes it), I2t
    void* dGmFb = nullptr;   // fusionnet b (2 * gmC[0], half), channels-last fp16
    void* dGmFc = nullptr;   // fusionnet c (2 * gmC[1], quarter), the same
    void* dGmFd = nullptr;   // fusionnet d (2 * gmC[2], eighth), the same
    float* dGmT = nullptr;   // the timestep, one device float (ifnet's 1x1x1x1 input)
    void* dGmOut = nullptr;  // the full graph's fusionnet output (3, ph, pw) in the engine's dtype (gmTail: null, the
                             // tail writes dGmF)
    float* dGmF = nullptr;   // that output clamped to fp32: what storeSlot consumes
    // HDR planes (nativeGmfssPair): the SDR view of one padded frame (hdrToModel, the scratch), the motion view's
    // halves gmflow and metricnet read on a pair outside the SDR range, the range flag (device, pinned host copy);
    // per feature slot: its frame stays inside the SDR range, its features came from the SDR view; the pair's mode
    float* dGmEnc = nullptr;
    float* dGmHalfM = nullptr;
    int* dGmRange = nullptr;
    int* hGmRange = nullptr;
    bool gmSdrPairs = true; // SMV_GMFSS_SDR_PAIRS=0: every pair the way outside the SDR range
    bool gmFrameSdr[2] = {}, gmFeatSdr[2] = {}, gmPairSdr = false;
    // gmFeatOk[s] = dGmFeat[s] holds its frame's features: a call without tweens skips feat_ext (gmFeatLazy;
    // SMV_GM_FEAT_LAZY=0 = never) and the next pair that makes tweens takes the previous frame's again (the refeat
    // branch). gmCatchUp = the last pair ran such a catch-up the eager route would not have (its span between
    // gmCatchEv[0] and [1], live only), which the throttle's price takes out of that group's pre
    bool gmFeatOk[2] = {}, gmFeatLazy = true, gmCatchUp = false;
    cudaEvent_t gmCatchEv[2] = {};
    uint32_t gmFeatSkips = 0, gmCatchUps = 0;
    int gmModeLogs = 0; // the mode lines this session printed (the first pair's and up to 3 switches)
    // SMV_LIVE_GMFSS_PROF=1: per-phase CUDA-event breakdown of the chain, printed every 32
    // groups (nine spans, the tween ones
    // measured on the group's FIRST tween). The pair span splits into half | flow | metric |
    // pyr, with the CPU (QPC) enqueue spans beside the GPU ones: the pair
    // block, the gmflow enqueue inside it, and every tween. The exe runs a group ahead of the
    // GPU, so those CPU numbers are launch cost, not the GPU pace.
    bool gmProf = false, gmProfTween = false;
    cudaEvent_t gmEv[11] = {};
    double gmAcc[9] = {};
    uint32_t gmProfN = 0;
    double gmCpuPair = 0.0, gmCpuFlow = 0.0, gmCpuTween = 0.0;
    uint32_t gmCpuPairN = 0, gmCpuTweenN = 0;
    // NVIDIA Smooth Motion (fruc): the handoff answers
    // `engine=fruc` with the /64 padded geometry and `NATIVE-PATH fruc=` (the folder holding the
    // shipped nvoffruc_bridge.dll beside the user-installed NvOFFRUC.dll + cudart64_110.dll);
    // this host drives that bridge's flat C API:
    // the (R, G, B) model planes packed to true BGRA8 at pw x ph (k_packBgraRgb),
    // one bridge call per tween, the BGRA8 result unpacked (k_unpackBgraRgb).
    bool fruc = false;
    bool dlssg = false;           // offline DLSS 4.5: no IFNet, a child server
    std::string frucDir;          // the bridge folder the handoff named
    uint8_t* dFrSurf[3] = {};     // BGRA8 pw x ph: the packed frames, rotated per group
    uint8_t* dFrOutB = nullptr;   // BGRA8 pw x ph: the bridge's output
    float* dFrOut = nullptr;      // (3, ph, pw): the tween, the layout storeSlot reads
    int frPrev = -1, frLast = -1; // surfaces: the previous frame, the last tweened pair's end
    int frA = -1, frB = -1;       // this group's pair
    // the feed-once bridge (nvoffruc_step), per FRUC instance: the surface whose frame it was fed
    // last (-1 = none, or that surface was repacked since), whether the pair's next tween on it is
    // its first, and how many calls it has served
    int frFed[kFrDepth] = {}; // -1 each from nativeFrucSetup on
    bool frFirst[kFrDepth] = {};
    uint64_t frCalls[kFrDepth] = {};
    bool frCreated = false;
    uint64_t frPrimed = 0, frRepeats = 0, frTweens = 0;
    // parallel instances (the direct-t scheme, bridge nvoffruc_step_i): tween k of a pair's plan
    // runs on instance k % frInst, a round of frInst tweens at a time; instance 0 on the compute
    // thread, the others on their own workers (g_frW), each into its own BGRA8 output
    int frInst = 1, frInstMax = 1;
    uint8_t* dFrOutBI[kFrDepth] = {}; // [0] = dFrOutB
    double frPlan[64] = {};
    uint32_t frPlanN = 0, frPlanK = 0;
    // recursive midpoints (the default): NvOFFRUC damages content that does not move at every
    // t != 0.5, even called exactly as NVIDIA documents it, so every tween is node k / 2^L of the
    // pair's midpoint tree:
    // FRUC at t = 0.5 between the node's two parents, level L on instance L - 1, which then sees
    // one continuous stream of new frames (NVIDIA's one call per new frame). A t that is no node
    // takes the nearest node of depth frMpCap. SMV_FRUC_MIDPOINTS=0 = the direct-t scheme above.
    bool frMp = false;
    int frMpCap = 2;                // off-node snap depth: offline 3, live 2 (cost); SMV_FRUC_DEPTH=1..9 sets both
    int frMpL = 0;                  // this pair's depth, 0 = not chosen yet
    bool frMpBuilt = false;         // this pair's whole tree to frMpL is computed
    uint64_t frSerial = 0;          // frames packed this session: the node keys
    uint8_t* dFrNode[kFrGrid] = {}; // BGRA8 pw x ph: node j / kFrGrid of this pair (j = 1..kFrGrid - 1)
    bool frNodeOk[kFrGrid] = {};
    uint64_t frLastKey[kFrDepth] = {}; // per instance: the key of the frame it was fed last, 0 = none
    uint64_t frNodeCalls = 0;
    // pipelined across pairs (live Smooth Motion, the exe's route bit on a group of x4 or deeper): a group shows the
    // PREVIOUS pair, whose levels 1..k instance 0 made during the previous group on its own worker (g_frW[0], k = the
    // depth less one: nativeFrucPipeSpec) while the compute thread ran the deeper levels of the pair before; every
    // instance still sees its frames in time order. frLag* = the pair shown (its serial, surfaces, SDR view, nodes,
    // held = identical or never split); frKeep = the surface the next pack spares (the shown pair's first frame);
    // frPipeDepth = the depth the last pipelined group's tweens asked (the worker's k)
    bool frPipePrev = false, frLagHave = false, frLagSdr = false, frLagStatic = false, frCurStatic = false;
    bool frSpecRun = false, frRepacked = false;
    int frLagA = -1, frLagB = -1, frKeep = -1, frLagL = 0, frPipeDepth = 0;
    uint64_t frLagSerial = 0, frPipeGroups = 0;
    uint8_t* dFrLagNode[kFrGrid] = {};
    bool frLagOk[kFrGrid] = {};
    // this group's midpoint-tree calls on the compute thread (Smooth Motion's bridge or AMD FSR 4's: calls, their ms,
    // the calls and ms of those made before the slots = FSR 4's level 1, and the chain inside the slots incl. a
    // pipelined group's worker join) for the exe's price (gFrCallUs / gFrStoreUs); a pipelined group's route
    // (grPiped) and instance 0's worker's calls + ms (grFrWk*) for the pipelined prices (gFrPipeMainUs / gFrPipeWkUs)
    uint32_t grFrCalls = 0, grFrPreCalls = 0, grFrWkCalls = 0;
    double grFrCallMs = 0.0, grFrPreMs = 0.0, grFrChainMs = 0.0, gFrCallMs = 0.0, gFrStoreMs = 0.0, grFrWkMs = 0.0;
    double gFrPipeMainMs = 0.0, gFrPipeWkMs = 0.0, gFrPipeRm = 0.0, gFrPipeRw = 0.0, gFrCallEntryMs = 0.0;
    int grPipeLearn = 0; // the groups left of a pipelined spell's start that learn gFrPipeRm / gFrPipeRw
    bool grPiped = false;
    // groups whose midpoint-tree price is not sampled: one that made instances (their creation and first, cold calls
    // land in its time: a first x4 group billed AMD FSR 4's call at 20.3 ms at 854x480) and the next
    int grPriceSkip = 0;
    // HDR planes (encPost): a pair whose two frames stay inside the SDR range goes to FRUC as their SDR view
    // (k_packBgraSdr) and its tweens come back through k_unpackBgraSdr; any other pair as the 8-bit HDR codes. Per
    // surface: its frame inside the range, whether it holds the SDR view. SMV_FRUC_SDR_PAIRS=0 = every pair as the codes
    bool frSdrOn = false, frPairSdr = false;
    bool frIn[3] = {}, frSurfSdr[3] = {};
    int* dFrRange = nullptr;
    int* hFrRange = nullptr;
    int frModeLogs = 0;
    // NVIDIA Video Frame Generation (vfg, `--nvvfg`): the FRUC route above (fruc is set too: its surfaces and their
    // rotation) with the "VideoFrameGeneration" effect of NVIDIA's Video Effects runtime in place of NvOFFRUC (frucDir
    // = the folder holding the runtime's four files). The surfaces hold R G B A bytes (the effect's order); on HDR
    // planes they hold one R10G10B10A2 word a pixel instead (vfR10: the planes' codes at 10 bits, as stored;
    // SMV_VFG_HDR10=0 = their 8-bit codes). One call a tween at any time between the pair: no midpoint tree, one
    // instance, the unpadded geometry. Live: the handoff answers `engine=nvvfg` with that geometry and `NATIVE-PATH
    // nvvfg=` (the runtime's folder); with no tree, a session's ladders and prices are the any-time models' (a tween's
    // own time).
    bool vfg = false, vfR10 = false;
    void* vfEff = nullptr; // this session's effect
    // AMD FSR frame generation (fsrfg): the FRUC route above (fruc is set too: its ladders, its midpoint tree, its
    // surfaces) with our smv_fsrfg_bridge.dll (AMD's FSR 4 ML frame generation on vkd3d-proton) in place of NvOFFRUC.
    // The handoff answers `engine=fsrfg` with the unpadded geometry and `NATIVE-PATH fsrfg=` (the bridge folder). Every
    // frame (real or a node) is RGBA16F at the bridge's row pitch in dFrSurf / dFrNode; a call feeds instance i one frame
    // and its vectors (NVIDIA optical flow, frame -> the instance's last frame, px) and returns the frame halfway between
    // them. The bridge's three textures (frame, vectors, generated frame) are Vulkan exports CUDA maps as arrays, its fence
    // a Vulkan timeline semaphore: the stream copies in, signals an odd value, the bridge's queue waits, generates,
    // signals the next even value, the stream waits for it and copies out. Instance 0 is fed every real frame in order
    // (FSR is built for one continuous stream); level L of a pair's tree on instance L - 1, as FRUC's. frLastKey 0 =
    // never fed (its first call resets). Intel XeSS-FG (--xess-fg) runs on this route through its own bridge (fgKind 1:
    // smv_xefg_bridge.dll with Intel's libxess_fg.dll + libxell.dll in engine\xess): D3D12 textures and a D3D12 fence
    // instead of the Vulkan exports, the same calls.
    bool fsrfg = false;
    int fgKind =
        0; // the bridge on this route: 0 = AMD FSR 4 (smv_fsrfg_bridge.dll), 1 = Intel XeSS-FG (smv_xefg_bridge.dll)
    struct FgInst
    {
        cudaExternalMemory_t emIn = nullptr, emMv = nullptr, emOut = nullptr;
        cudaMipmappedArray_t mmIn = nullptr, mmMv = nullptr, mmOut = nullptr;
        cudaArray_t in = nullptr, mv = nullptr, out = nullptr; // the three textures' level 0
        cudaExternalSemaphore_t sem = nullptr;
        uint64_t val = 0; // the fence's last value
        bool made = false;
    };
    FgInst fg[kFrDepth];
    LUID fgLuid = {};
    int fgPitch = 0, fgMvPitch = 0; // the bridge's row pitches (bytes), the same for every instance
    uint8_t* dFgMv = nullptr;       // a call's vectors (RG16F rows fgMvPitch apart) before their copy into its texture
    // SMV_FSRFG_PROF=1: CUDA events between a call's stages (copy in, the Optical Flow inputs, the Optical Flow, the
    // vector kernels, the vector copy, FSR on vkd3d, the copy out), means at the session's end; every call then waits
    // for its last event, so the stages do not overlap the next call's. A derived call's k_mvDerive counts apart
    // (fgDeriveMs over fgProfDerN), the three Optical Flow stages over the calls that ran a pass (fgProfVecN)
    bool fgProf = false;
    cudaEvent_t fgEv[8] = {};
    double fgStageMs[7] = {}, fgCpuMs = 0.0, fgDeriveMs = 0.0;
    uint64_t fgProfN = 0, fgProfVecN = 0, fgProfDerN = 0;
    // FSR's motion vectors (frame generation and upscaling): an Optical Flow session at w x h, ABGR8 (R, G, B, A
    // bytes), one direction, fast, its output grid: the field from slot `cur` to slot `prev` = the current frame -> its
    // predecessor (FSR's convention), in px: two planes (FSR reads no cost: no cost buffers, k_nvofUp writes none)
    struct OfVec
    {
        NvOFHandle of = nullptr;
        NvOFGPUBufferHandle in[2] = {}, out[2] = {};
        CUdeviceptr inP[2] = {}, outP[2] = {};
        uint32_t inPitch = 0, outPitch[2] = {};
        int w = 0, h = 0, grid = 1, gw = 0, gh = 0;
        float* dFlow = nullptr;
        bool both = false; // SMV_OFVEC_BOTH=1: both directions, the backward field read (else forward, frames swapped)
        NV_OF_PERF_LEVEL level = NV_OF_PERF_LEVEL_SLOW; // nativeOfLevel()
        cudaStream_t st = nullptr; // the stream its Optical Flow and kernels run on, null = the session's
        int wr = 1; // one direction: the field goes to out[wr] (FSR 4's pipelined route alternates out[0] / out[1])
        // FSR 4's level-1 chain (nativeFsrfgOfIn): the input slot still holding frame surface keepSrc as the last pass
        // converted it, -1 = none; a pack into that surface (nativeFsrfgPack) and every other pass clear it
        int keepSlot = -1;
        const uint8_t* keepSrc = nullptr;
    };
    OfVec fgOf;
    // a tree level with 2+ calls makes its vectors ahead (nativeFsrfgLevel): call j + 1's Optical Flow on fgOfSt (its own
    // session fgOf2, into dFgMvR[(j + 1) & 1]) while call j's FSR runs. fgAhead: 0 = not made yet, 1 = on, -1 = off
    // (SMV_FSRFG_OVERLAP=0, the profiler, or a failed setup: serial vectors)
    int fgAhead = 0;
    cudaStream_t fgOfSt = nullptr;
    OfVec fgOf2;
    uint8_t* dFgMvR[2] = {};
    float fgChildT0 = 0.0f, fgChildT1 = 0.0f; // a serial level-2+ call's two frames in its pair (k_mvDerive's t0, t1)
    cudaEvent_t fgEvLevel = nullptr, fgEvVec[2] = {}, fgEvFree[2] = {};
    // the level-1 chain's two streams: fgOfL1St runs fgOf's Optical Flow and its kernels (the session's stream waits
    // for them on the direct route), fgL1St on the pipelined route (SMV_FSRFG_PIPE, nativeFsrfgPipeAdvance) instance
    // 0's copies and whole call of the new pair, beside the shown pair's level-2+ calls on the session's stream, so the
    // next pair's Optical Flow runs beside instance 0's call; FSR's calls keep the serial route's order on the device
    // (instance 0's waits for fgEvDeep, the shown pair's last deeper call). fgOfWr / fgOfRd = the field fgOf writes /
    // the derived vectors read (out[0] / out[1] alternate by pair while pipelined, else 1); dFgMvL1 = the calls'
    // vectors on fgL1St, dFgL1Node = the new pair's level-1 node (swapped into dFrNode when that pair is shown);
    // fgEvL1 = fgL1St's last work, fgEvS = the session stream's point the two streams wait for, fgEvOf = fgOfL1St's
    // last pass, fgEvMv = fgL1St's last read of dFgMvL1 (the next pass writes it after), fgEvSurf[n] = frame surface
    // n's last read off the session's stream (a pack into it waits for it); fgPiped = the last group ran pipelined,
    // fgL1Pend = the new pair's call still to issue (nativeFsrfgL1Issue), fgL1Ok = its node is made
    cudaStream_t fgL1St = nullptr, fgOfL1St = nullptr;
    cudaEvent_t fgEvL1 = nullptr, fgEvS = nullptr, fgEvDeep = nullptr, fgEvOf = nullptr, fgEvMv = nullptr;
    cudaEvent_t fgEvSurf[3] = {};
    int fgOfWr = 1, fgOfRd = 1;
    uint8_t* dFgMvL1 = nullptr;
    uint8_t* dFgL1Node = nullptr;
    bool fgPiped = false, fgL1Pend = false, fgL1Ok = false, fgPipeLogged = false;
    // AMD FSR upscaling (fsrUp, `--fsr-upscale`): RTX VSR's resize (vsr is set too, so the slot rules, the staging and
    // nativeVsrEval's callers stay one path; nativeVsrEval hands dVsrIn to smv_fsrup_bridge.dll instead of NGX). The
    // bridge runs AMD's signed FidelityFX upscaler (FSR 4.1.1's INT8 model) on its own D3D12 device, frames in and out
    // through shared buffers + one fence like FSR FG's; every frame of the slot's stream is fed in order (the first
    // resets), its vectors from the previous one (fuOf, the two input slots alternate)
    bool fsrUp = false;
    bool fuHdr = false; // live HDR planes: the bridge in its HDR mode (linear light; off with RTX HDR before the model)
    std::string fsrUpDir;
    int fuKind =
        0; // the bridge on this route: 0 = AMD FSR (smv_fsrup_bridge.dll), 1 = Intel XeSS-SR (smv_xess_bridge.dll)
    cudaExternalMemory_t fuEmIn = nullptr, fuEmMv = nullptr, fuEmOut = nullptr;
    uint8_t* dFuIn = nullptr;
    uint8_t* dFuMv = nullptr;
    uint8_t* dFuOut = nullptr;
    cudaExternalSemaphore_t fuSem = nullptr;
    uint64_t fuVal = 0;
    int fuInPitch = 0, fuMvPitch = 0, fuOutPitch = 0;
    bool fuMade = false;
    int fuCur = 0;    // the vector session's slot that holds this frame
    uint64_t fuN = 0; // frames upscaled
    // past 3x a side (fuChainPlan): pass k = fuPw[k] x fuPh[k] -> fuPw[k + 1] x fuPh[k + 1] on bridge instance k; pass 0
    // = the fields above (its output dFu0Out when chained), passes 1 .. fuPasses - 1 = fuX; dFuOut = the last pass's
    // output, what the callers read; a pass's vectors = pass 0's scaled to its size (k_fuMvScale)
    static constexpr int kFuMaxPass = 6;
    struct FuPass
    {
        cudaExternalMemory_t emIn = nullptr, emMv = nullptr, emOut = nullptr;
        uint8_t* dIn = nullptr;
        uint8_t* dMv = nullptr;
        uint8_t* dOut = nullptr; // null on the last pass (its output = dFuOut)
        cudaExternalSemaphore_t sem = nullptr;
        uint64_t val = 0;
        int inPitch = 0, mvPitch = 0, outPitch = 0;
        bool made = false;
    };
    int fuPasses = 1;
    int fuPw[kFuMaxPass + 1] = {}, fuPh[kFuMaxPass + 1] = {};
    FuPass fuX[kFuMaxPass - 1];
    cudaExternalMemory_t fu0EmOut = nullptr;
    uint8_t* dFu0Out = nullptr;
    int fu0OutPitch = 0;
    // SMV_FSRUP_PROF=1: CUDA events between a frame's stages (the input pack, the Optical Flow, the vector kernels, FSR
    // on D3D12, the output unpack), means at the session's end; every frame then waits for its last event
    bool fuProf = false;
    cudaEvent_t fuEv[6] = {};
    double fuStageMs[5] = {};
    uint64_t fuProfN = 0, fuProfVecN = 0;
    char fuVersion[32] = {};
    OfVec fuOf;
    // an FSR / XeSS upscaler ticked mid-session (nativeUpSwitch): its first two frames run on a worker thread
    // (nativeFsrupWarmProc), so the driver's shader compile inside them (up to kFsrCompileMs on an exe location and
    // upscale ratio it has not seen) blocks the worker, not the native thread, while the session presents without the
    // upscaler. fuWarm: 1 = warming, 2 = warm, 3 = failed; nativeUpWarmEnd turns it on at the next group when the tick
    // still stands (fuWarmWant: an `up 0` meanwhile clears it)
    volatile LONG fuWarm = 0;
    HANDLE fuWarmTh = nullptr;
    bool fuWarmWant = false;
    ULONGLONG fuWarmT0 = 0;
    // ArtCNN (acnUp, `--artcnn`): RTX VSR's resize too (vsr is set), on planar floats like Anime4K: C4F32_DN v1.6.2's
    // luma x2 in its own module (acnMod, kAcnA + kAcnB), the weights from engine\artcnn\artcnn_c4f32_dn_v1.6.2_fp16.pkl,
    // the colour from the frame's Lanczos enlargement (nativeAcnRun); x2 passes as Anime4K's (a4kPassesFor)
    bool acnUp = false;
    CUmodule acnMod = nullptr;
    CUfunction fAcn[6] = {}; // k_acnBodyR, k_acnBodyA, k_acnFirst, k_acnLast, k_acnY, k_acnMerge
    void* dAcnWf = nullptr;  // conv1 .. conv5's fp16 B fragments, conv after conv (9 x 32 x 32 halves each)
    float* dAcnB = nullptr;  // conv1 .. conv5's fp32 biases, 32 each
    float* dAcnW0 = nullptr; // conv0's weights [32][9], then its 32 biases (fp32)
    float* dAcnWz = nullptr; // conv6's weights [4][32][9], then its 4 biases (fp32)
    void* dAcnAct = nullptr; // three activation buffers (channels-last fp16) of acnActN halves each
    size_t acnActN = 0;
    int acnCore = 0;                  // the tile core (0 = whole frames)
    int acnW = 0, acnH = 0, acnN = 1; // the first pass's input size, the x2 passes a frame
    int acnOw = 0;                    // the widest fit target dAcnTmp holds
    float* dAcnY = nullptr;           // the luma of a pass's input (the largest pass's size)
    float* dAcnY2 = nullptr;          // the net's luma (the largest pass's output)
    float* dAcnLz = nullptr;          // the pass's input Lanczos x2: three planes (the largest pass's output)
    float* dAcnOut = nullptr;         // a pass's RGB, three planes = the next pass's input
    float* dAcnTmp = nullptr;         // the Lanczos pairs' horizontal pass
    size_t acnTmpN = 0;               // floats dAcnTmp holds
    // Anime4K (a4kUp, `--anime4k`): RTX VSR's resize too (vsr is set), on planar floats instead of the bridge's BGRA:
    // Upscale CNN x2 UL's conv passes in CUDA (its own module a4kMod, a4kKernelText: layers 1..6 on the tensor cores,
    // layer 0 inside layer 1's kernel and the last 1 x 1 inside layer 6's, both with literal weights) at the slot's
    // input size, then the fold (k_a4kFoldHSub: the shader's Depth-to-Space pass, and the Lanczos pair from 2w x 2h to
    // the slot's target); HDR frames go in as their SDR view and come back through RTX VSR's way to the codes
    bool a4kUp = false;
    CUmodule a4kMod = nullptr;
    // the module build and the statistics buffers are shared by Restore (nativeModelInit) and the resize
    // (nativeRtxInit), which the live early init runs on two threads at once: one builds, the other finds it
    std::mutex mA4k;
    // k_a4kConvM16First, k_a4kConvM16, k_a4kConvM16Last (the Upscale net), k_a4kRestFirst, k_a4kRestLast (the Restore
    // net), k_a4kStatH, k_a4kStatV, k_a4kClamp (Clamp_Highlights)
    CUfunction fA4k[8] = {};
    int a4kW = 0, a4kH = 0;   // the first pass's input size (the slot's)
    int a4kN = 1;             // x2 passes a frame: the fewest whose 2^n reaches the slot's ratio, memory permitting
    bool a4kAA = false;       // Mode A+A (--anime4k-aa, memory permitting): Restore on the first pass's 2x frame
    int a4kOw = 0, a4kOh = 0; // the fold target dA4kTmp holds (the widest; a live A+A switch sets up again at it)
    void* dA4kIn = nullptr;   // NCHW fp16, the largest pass's input size
    void* dA4kT = nullptr;    // four layer slots, 3 textures x 4 fp16 a pixel each (nativeA4kRun), the largest pass's
    void* dA4kOut = nullptr;  // the three last textures as 12 fp16 planes, the largest pass's
    float* dA4kMid = nullptr; // a pass's exact 2x frame = the next pass's input, planar fp32 (a4kN > 1 only)
    float* dA4kTmp = nullptr; // the fold's horizontal pass: planar fp32, target width x 2 x the last pass's height
    // Clamp_Highlights, the shader set's de-ringing (part of every Anime4K mode): the 5 x 5 maximum of the luma, taken
    // before a frame's first Anime4K pass (Restore's input when Restore is Anime4K, else the resize's) and kept fp16 at
    // that size; the chain's last output is lowered where its luma exceeds the bilinear sample of that maximum
    unsigned short* dA4kStat = nullptr;    // the maximum, a4kStatW x a4kStatH
    unsigned short* dA4kStatTmp = nullptr; // its horizontal pass
    size_t a4kStatCap = 0;                 // pixels the two hold
    int a4kStatW = 0, a4kStatH = 0;
    bool a4kStatFresh = false; // this frame's Restore (Anime4K) took the maximum and leaves the clamp to the resize
    // Restore = Anime4K's Restore CNN UL (restNatKind 3, `--restore-anime4k`): the same kind of passes at the source
    // size (weights 6..12 of the module, four layer slots of its own), out as planar fp32 (the shader adds the source
    // back; clamped 0..1 like every Restore), then Restore's Lanczos pair when the target size differs
    void* dA4kRT = nullptr;    // four layer slots at the source size
    float* dA4kRMid = nullptr; // the restored frame, planar fp32 at the source size
    float* dA4kRTmp = nullptr; // the fold's horizontal pass (planar fp32, a4kRTmpW x the source height)
    int a4kRTmpW = 0;
    // RIFE with DRBA timing (rifedrba): the RIFE
    // handoff plus `engine=drba lag=1` (calc_flow's block0 runs on the host's kernels,
    // nativeB0Module): a four-frame history of padded frames and their
    // encodes, windows centred on a frame id (two kept, chained: a window's left flow is the
    // previous window's right one, reversed), the lag-1 group (every slot shows time (k-2) + f,
    // the real slot is frame k-1), one IFNet enqueue per tween with its own DRM timestep map
    // (the kernels k_drbaFlowSplat / FlowNorm / DrmSplat / DrmNorm above, gated against a float64
    // reference), the head fallback (plain pair RIFE on the lagged pair).
    // RIFE's two domains (mph set): the ring keeps each frame's motion frame beside the picture, and
    // the encodes, block0 and its flows, the windows, the DRM map and the IFNet's x live at the
    // motion frame (M x N = mph x mpw, else the pictures' ph x pw); k_rifeBlend warps the pictures.
    bool drba = false;
    // DRBA's block0 on the host's kernels (nativeB0Module): its module and five kernels, the weight block (b0Off, see
    // b0NatLoad), the scratch (in the shared activation block, else dB0Own)
    CUmodule b0Mod = nullptr;
    CUfunction fB0[5] = {};
    void* dB0W = nullptr;
    size_t b0Off[28] = {};
    void* dB0Own = nullptr;
    // RIFE's IFNet on the host's kernels (ifnetPath = kIfnNative, nativeIfnModule): its module and kernels, the weight
    // block (ifnOff, see ifnNatLoad), the scratch (in the shared activation block, else dIfnOwn)
    bool ifnNat = false;
    CUmodule ifnMod = nullptr;
    CUfunction fIfn[12] = {};
    void* dIfnW = nullptr;
    std::vector<size_t> ifnOff;
    void* dIfnOwn = nullptr;
    ChainGraphs ifnGraphs;        // its launches as CUDA graphs (nativeIfnRun; SMV_IFN_GRAPH)
    uint32_t drFid = 0;           // frames pushed this session; frame id i sits in ring slot i & 3
    float* dDrIBase = nullptr;    // the four ring slots in ONE block, slot i at i * (3, ph, pw): two ring neighbours
                                  // are one (6, ph, pw) frame pair
    float* dDrI[4] = {};          // (3, ph, pw) the padded frames (the pictures), slot i of dDrIBase
    uint32_t drPicFix = 0;        // two domains: the k whose wrap pair copied frame k - 1 into slot (k + 1) & 3
    uint16_t* dDrMBase = nullptr; // two domains: the motion ring, five (3, M, N) fp16 slots in ONE block: frame i
                                  // in slot 3 - (i & 3) (descending, so frame i - 1 follows frame i), slot 4 a spare
    uint16_t* dDrM[4] = {};       // two domains: their motion frames, ring slot i & 3 = block slot 3 - (i & 3)
    float* dDrF[4] = {};          // (kRifeFeatC, M, N) their encodes in the IFNet's feature dtype (nativeTweenBuffers)
    float* dDrX[2] = {};          // one domain: (6, ph, pw) the frame pair: [0] = [k-1, k-2] (side -1), [1] = [k-2,
                                  // k-1]; the IFNet's x (fp16 in place when xHalf)
    uint16_t* dDrMX[2] = {};      // two domains: the IFNet's x, two neighbouring slots of dDrMBase (not owned)
    uint32_t drXFor[2] = {};      // the newest frame id each x was built for (0 = none)
    struct DrWin
    {
        uint32_t c;
        float* f10;
        float* r;
    }; // r = (4, M, N): flow12 | flow21
    DrWin drWin[2] = {};         // c = 0: empty
    long long* dDrAcc = nullptr; // (6, M, N) the splat accumulator
    uint64_t drTweens = 0, drHeads = 0, drBlock0 = 0;
    // DRBA as its author runs it (routineLife1/DRBA infer.py; drbaAuthorWanted): the network's frames stretched
    // bilinearly over its whole /64 frame as B, G, R (k_motionIn st), RIFE's last step at that frame on the pair's
    // stretched pictures, every output frame (tweens and real frames alike) bilinearly back to the pictures' size
    bool drAuthor = false;
    float* dDrS = nullptr; // (6, M, N) the pair's pictures stretched (nativeDrbaAuthorBlend): two (3, M, N) slots,
                           // frame i in slot i & 1, each frame stretched once (both sides of a group and the next
                           // group's left frame read it)
    int64_t drSFid[2] = {-1, -1}; // the frame id in each slot of dDrS (-1 = none)
    float* dDrB = nullptr;  // (3, M, N) the blend at the network's frame; a real frame's stretch on its way back: the
                            // accumulator's last 12 B a pixel (nativeDrbaAlloc, not owned)
    float* dDrRT = nullptr; // (3, ph, pw) a real frame stretched and back (nativeDrbaReal): the accumulator's first
                            // 36 B a motion pixel when the picture fits there, else its own block (drRtOwn)
    bool drRtOwn = false;
    // NVIDIA Optical Flow model: the handoff answers
    // `engine=nvof` with the geometry only (like echo); this host runs
    // the Optical Flow Accelerator itself through the driver's nvofapi64.dll (grid 4, fast,
    // BOTH directions in one Execute, gray8 of the two model frames at the true w x h) and the
    // glue kernels k_nvofLuma / k_nvofUp / k_nvofMetric (checked against fp64 references;
    // k_splatNvof / k_splatNvofNorm in the kernel text are a forward-splat tween with no host
    // launch, kept because the kernel text is shared byte for byte with that check).
    bool nvof = false;
    NvOFHandle ofH = nullptr;
    NvOFGPUBufferHandle ofIn[2] = {}, ofOut[2] = {}, ofCost[2] = {}; // [0] forward, [1] backward
    CUdeviceptr ofInP[2] = {}, ofOutP[2] = {}, ofCostP[2] = {};
    uint32_t ofInPitch = 0, ofOutPitch[2] = {}, ofCostPitch[2] = {};
    int ofGw = 0, ofGh = 0;
    float* dNvFlow[2] = {};      // (2, h, w) px: [0] = F01 (frame 0 -> 1), [1] = F10
    float* dNvCost[2] = {};      // (h, w) the upsampled cost per direction
    float* dNvZ[2] = {};         // (h, w) the splat metric per direction
    long long* dNvAcc = nullptr; // (5, h, w): RGB * e, e, coverage
    float* dNvOut = nullptr;     // (3, ph, pw): the tween, the layout storeSlot reads
    bool nvProf = false;         // SMV_LIVE_NVOF_PROF=1: per-pair GPU ms
    cudaEvent_t nvEv[2] = {};
    double nvPairMs = 0.0, nvTweenMs = 0.0;
    uint32_t nvPairN = 0, nvTweenN = 0;
    // the tween: the pull warp + confidence fallback. The velocity
    // at time t (n / d = the push-pull level 0 inputs), the two visibilities, the raw and the
    // blurred fallback mask, the blur's pass, and the push-pull pyramid levels 1.. (n, d, out)
    float* dNvN0 = nullptr;      // (2, h, w) velocity * confidence
    float* dNvD0 = nullptr;      // (h, w) confidence
    float* dNvVis = nullptr;     // (2, h, w) visibility in frame 0 / frame 1
    float* dNvMraw = nullptr;    // (h, w) the fallback mask before the blur
    float* dNvMask = nullptr;    // (h, w) the blurred mask (1 = warp, 0 = blend)
    float* dNvBlurTmp = nullptr; // (h, w) the blur's horizontal pass
    float* dNvV = nullptr;       // (2, h, w) the filled velocity (push-pull level 0 out)
    struct NvLevel
    {
        int w, h;
        float* n;
        float* d;
        float* out;
    };
    std::vector<NvLevel> nvPyr;                 // levels 1.. down to 1 px on the short side
    bool restHalfIn = true, restHalfOut = true; // Restore's input / output dtypes (fp16 on the host's kernels)
    int restScale = 4;                          // the model's scale: Real-ESRGAN 4, AnimeJaNai 2
    bool restSub = false; // the output = 3 s^2 sub-pixel planes at the input size (Real-ESRGAN): k_restFoldHSub
    bool restFailed = false;
    int restTw = 0, restTh = 0; // the fold target (decided once VSR is known)
    // Real-ESRGAN on the host's kernels (restorePath = kRestNative, no engine): dRestA / dRestB = the channels-last fp16
    // ping-pong (64 halves a pixel each; inside the shared activation block when it holds them, else restNatOwn),
    // dRestW = the weights (restNatOff offsets)
    bool restNat = false, restNatOwn = false;
    int restNatKind = 0;   // 1 = the bundled Real-ESRGAN, 2 = AnimeJaNai (SPAN-F3 64, the --restore-model file),
                           // 3 = Anime4K's Restore CNN UL (--restore-anime4k: the Anime4K module's passes)
    int restNatBlocks = 0; // AnimeJaNai's block count
    void *dRestA = nullptr, *dRestB = nullptr, *dRestW = nullptr; // dRestA = the whole activation region
    std::vector<size_t> restNatOff;
    // the fold's sub-pixel input (restSub): s, the plane grid w x h, the nearest-base add (Real-ESRGAN) or none
    int restSubS = 4, restSubW = 0, restSubH = 0;
    bool restSubBase = true;
    void* dRestIn = nullptr;   // the input: NCHW at the source size (srcW x srcH), fp16
    void* dRestOut = nullptr;  // the output: the model's fp16 sub-pixel planes (restSub)
    float* dRestTmp = nullptr; // the fold's horizontal pass, planar fp32 at (4h, the largest scale) x max target width
    float* dRest = nullptr;    // the fold back to the model size (VSR follows), planar fp32 w x h
    float* dRestRem = nullptr; // HDR planes: the source fitted to Restore's target (k_hdrRestOut's remainder)

    // ---- ring geometry (mirrors the exe's own slot layout)
    uint32_t slots = 0, pitch = 0;
    size_t slotBytes = 0;

    // ---- the models' dtypes
    bool featHalf = false; // IFNet / block0 take f0 / f1 in fp16: the encode (nativeEncodeRun) writes that dtype
    // the IFNet's x in fp16 (the native IFNet takes the fp16 copy of the pair, dXh, or the motion frames)
    bool xHalf = false;
    // the tweens (merged) are fp16 (k_rifeBlend, SMV_TWEEN_FP16=1), else fp32 (k_rifeBlendF): every reader of a
    // tween gets this as its half flag (mergedAt)
    bool outHalf = false;
    // the IFNet takes its timestep in fp16: the constant fills and DRBA's map are written in fp16
    bool tHalf = false;

    // ---- CUDA
    cudaStream_t stream = nullptr;
    cudaExternalMemory_t emCap = nullptr, emOut = nullptr;
    cudaExternalSemaphore_t semCap = nullptr;
    cudaMipmappedArray_t capMip = nullptr;
    cudaArray_t capArr = nullptr;
    uint8_t* dOutRing = nullptr;
    uint8_t* dCap = nullptr;
    float* dTmp = nullptr; // the capture resize's horizontal pass, and nativeFitPlanar's (tmpN floats)
    size_t tmpN = 0;
    // the Lanczos tap tables, one per resize axis in -> out (rsTabFor), freed by nativeFree
    struct RsTab
    {
        int in, out;
        void* d;
    };
    std::vector<RsTab> rsTabs;
    float* dCapF = nullptr;       // HDR + image scale under 1: planar PQ at capture resolution
    uint8_t* dThdrIn = nullptr;   // RGBA8 bridge input, pitch 4*cw
    uint32_t* dThdrOut = nullptr; // packed 10:10:10:2 bridge output, B in the low bits
    float* dSrcG = nullptr;       // the unquantised sRGB-encoded source, planar fp32
    float* dShIn = nullptr;       // the Sharpen's tight SDR view of HDR planes (w x h)
    float* dShOut = nullptr;      // and the sharpen's tight result padded back into the model frame (Sharpen on only)
    // offline: the TrueHDR buffers above are sized dw x dh (TrueHDR runs at the working size before the
    // model), and the emit's per-frame statistics block (nativeOfflinePqOut), read back one frame late
    uint8_t* dThdrStats = nullptr; // k_pqOut's hist / misc / vSum (kThdrStatsBytes)
    uint8_t* hThdrStats = nullptr; // its pinned copy, valid after the next stream sync
    bool thdrStatsPending = false;
    ThdrAcc* thdrAcc = nullptr; // the render's accumulators (runOfflineSession owns them)
    uint32_t thdrRepeat = 0;    // held slots that reuse the pending frame's record (DLSS 5)
    // offline DLSS 5: the NR core on its own D3D12 device, one RGBA16F frame at nrW x nrH, the
    // decoded frame before the interpolation (nativeNrFrame). Zero-copy by default: the frame
    // moves through the core's two shared D3D12 buffers, which the stream writes and reads
    // directly, and the core's shared fence orders both sides on the GPU. CPU staging
    // (SMV_NR_STAGED=1, the preview's change map, or a handoff that cannot start): the core's
    // renderFrame through pinned host copies.
    nr::Host* nrHost = nullptr; // null = DLSS 5 off or unavailable for this render
    bool nrFailed = false;      // an evaluate failed: off for the rest of the render
    bool nrFirst = true;        // the stream's first frame evaluates with Reset
    bool nrResetEvery = false;  // SMV_NR_RESET_EVERY=1: every frame (the equivalence gate's lever)
    bool nrReuse = false;       // a decoded frame identical to the previous one takes its DLSS 5 output
    uint64_t nrReused = 0;      // frames that did, this render
    bool nrZeroCopy = false;    // the shared-buffer route is up
    int nrW = 0, nrH = 0;       // the decoded picture's size, the size DLSS 5 runs at
    float* dRawPrev = nullptr;  // the previous decoded frame, packed: the identical-pair test's
                                // reference once DLSS 5 has rewritten the frames in dX (SMV_NR_REUSE=0 only)
    uint16_t* dNrIo = nullptr;  // CPU staging: device RGBA16F, both directions (zero-copy uses the core's textures)
    uint16_t* hNrIn = nullptr;  // pinned host copies handed to renderFrame (CPU staging)
    uint16_t* hNrOut = nullptr;
    // zero-copy: the core's own Color / Output / MVec textures as CUDA arrays (dedicated imports, mapped
    // arrays, surface objects the kernels store into and load from)
    cudaExternalMemory_t emNrIn = nullptr, emNrOut = nullptr, emNrMv = nullptr;
    cudaMipmappedArray_t mmNrIn = nullptr, mmNrOut = nullptr, mmNrMv = nullptr;
    unsigned long long sfNrIn = 0, sfNrOut = 0, sfNrMv = 0;
    cudaExternalSemaphore_t semNr = nullptr; // the core's shared fence
    uint64_t nrFenceV = 0;                   // last value put on it: odd = the stream's, even = the queue's
    // DLSS 5 motion (zero-copy route, nativeNrMotion): its own Optical Flow session at nrW x nrH on
    // the NR input frames, current -> previous, into the core's MVec texture (DLSSNR.MVec).
    // Off = SMV_NR_MV=0 or CPU staging: the pass then gets no motion
    bool nrMotion = false;
    NvOFHandle nrOfH = nullptr;
    NvOFGPUBufferHandle nrOfIn[2] = {}, nrOfOut = nullptr;
    CUdeviceptr nrOfInP[2] = {}, nrOfOutP = 0;
    uint32_t nrOfInPitch = 0, nrOfOutPitch = 0;
    int nrOfGw = 0, nrOfGh = 0;
    int nrOfCur = 0;          // the input slot this frame's luma goes into (they alternate)
    float* dNrFlow = nullptr; // (2, nrH, nrW): the field in px (k_nrMv reads no cost)
    std::wstring nrDeltaPath; // --nr-delta PATH: the pass's change map (the preview's mask)
    double nrMs = 0.0, nrMaxMs = 0.0;
    uint64_t nrN = 0;
    // live DLSS 5: the same core, handoff and motion once per captured frame on the
    // model frame after Restore / the resize (nativeLiveNrInit, nativeGroup); a capture byte-identical
    // to the previous one keeps the last output in the cur half instead of an evaluate
    bool liveNr = false;
    bool nrHaveLast = false;     // the frame before holds an evaluated output
    uint8_t* dCapPrev = nullptr; // the previous raw capture, the reuse test's reference; trades places with dCap
    float nrSdrWhite = 3.0f;     // HDR: the SDR reference white in scRGB units (nits / 80)
    int nrSrcHdr = 0;            // offline: HDR video (1 PQ, 2 HLG), the pass sees its SDR range
    // GMFSS and Restore on HDR planes (hdrToModel): the decoded frame's encoding (offline Restore reads it; 1 PQ,
    // 2 HLG, 0 SDR planes or off) and the model frames' (after RTX HDR: GMFSS, offline Restore after the model, live
    // Restore on PQ), the SDR white in scRGB units
    int encPre = 0, encPost = 0;
    float encWhite = 203.0f / 80.0f;
    // Sharpen on HDR planes without RTX HDR: the sharpen edits their SDR view (knee 1 head 0) as an SDR source's, clamped
    // at SDR white, and the light outside the view is added back; false = SMV_HDR_SHARPEN_VIEW=0, the sharpen on the codes
    bool shHdrView = true;
    // a resize of HDR planes before the model (live the capture to the working size, offline the decoded frame to it,
    // Restore's reference at a size change): a source frame inside the SDR range is resized as its SDR view (the SDR
    // route's Lanczos, clamped at SDR white) and comes back through k_hdrFromSdr, any other on the codes; false =
    // SMV_HDR_RESIZE_VIEW=0. dRsView = the source's SDR view (planar fp32 at the source size), set only when used.
    bool rsHdrView = true;
    float* dRsView = nullptr;
    int* dRsRange = nullptr;
    int* hRsRange = nullptr;
    int rsModeLogs = 0;
    bool rsLastView = false;
    bool nrSdrIn = false;  // live: DLSS 5's input is SDR this frame (RTX HDR converts after it)
    bool nrPaused = false; // live DLSS 5 switched off in a running session (stdin set): the core stays, the pass skips
    // a DLSS 5 pass growth in progress (stdin nrpasses above the passes built, nativeNrGrowStep): nrGrowWant = the run
    // count asked (0 = none), nrGrowFrom = the passes before it, the build time so far, the set's value and the note
    int nrGrowWant = 0, nrGrowFrom = 0;
    double nrGrowMs = 0.0;
    std::string nrGrowVal, nrGrowWhy;
    // an effect switched off in a running session (stdin set) through its own fallback (vsrFailed = Lanczos, restFailed =
    // no Restore, rtxFailed = faithful PQ): set when the switch set that flag, so on again clears only a switch, never a
    // real failure
    bool vsrUserOff = false, restUserOff = false, rtxUserOff = false;
    // offline: Restore and the resize run on the decoded frame (the pre-model
    // SOURCE, sw x sh, in dSrcPl) before DLSS 5 and the model, which then run at the working
    // size w x h, and the emit's final resize takes it to dw x dh; sw / sh = 0 = the source is the
    // model frame (no working size, and live)
    int sw = 0, sh = 0;
    bool nvPre = false;
    bool vsrPost = false;       // RTX VSR runs the final resize (it enlarges), not the first
    bool vsrPre = false;        // live: RTX VSR runs the capture -> working-size resize before the model, not the fit
    bool restPre = false;       // live: Restore on the captured frame, folded to the model size, before the model
    float* dSrcPl = nullptr;    // the decoded frame, planar fp32 at sw x sh
    float* dPres = nullptr;     // a resize stage's staging frame, planar fp32 at dw x dh (nativeOfflineStage)
    float* dFitTmp = nullptr;   // fitAa: the horizontal pass, planar fp32 at dw x (uh or h)
    float* dUp = nullptr;       // Upscale to: the internal render frame, planar fp32 at uw x uh
    float* dUpTmp = nullptr;    // upAa: the horizontal pass of the first resize, planar fp32 at uw x h
    uint8_t* dVsrIn = nullptr;  // VSR: tight RGBA8 (FSR upscaling: BGRA8) at the source size (srcW x srcH)
    uint8_t* dVsrOut = nullptr; // VSR: tight RGBA8 (FSR upscaling: BGRA8) at the presented size dw x dh
    float* dVsrView = nullptr;  // live HDR planes: the SDR view RTX VSR reads, planar fp32 at its input size
    float* dVsrRem = nullptr;   // live HDR planes: the Lanczos fit at VSR's target (k_hdrRestOut's remainder)
    unsigned hdrInBits = 0;     // live HDR planes: one bit a capture, newest lowest: it stayed inside the SDR range
    float* dX = nullptr;        // (1,6,ph,pw): prev in planes 0..2, cur in 3..5 (xSwap: the reverse)
    bool xSwap = false;         // live: dX's halves trade roles instead of the cur -> prev copy (nativeXPrev / Cur)
    uint16_t* dXh = nullptr;    // dX in fp16 for an IFNet that takes fp16 frames (xHalf)
    float* dF[2]{};             // f_prev / f_cur, (1,kRifeFeatC,ph,pw) in the IFNet's feature dtype (featHalf: fp16)
    float* dEncOwn = nullptr;   // the encode's half-size planes when no shared activation block holds them
    // the IFNet's features, timestep, flow and mask and the tweens, allocated once the engine's dtypes are
    // known (nativeTweenBuffers)
    void* dT = nullptr;        // (B,1,mph,mpw), fp32 or fp16 by tHalf
    uint32_t tHeld[64] = {};   // the t bits timestep plane j holds when bit j of tHeldMask is set: nativeFillT skips
    uint64_t tHeldMask = 0;    // a refill of the same t; cleared whenever anything else writes dT
    uint16_t* dFlow = nullptr; // (B,4,mph,mpw) fp16: the IFNet's final flow 0->t | 1->t
    uint16_t* dMask = nullptr; // (B,1,mph,mpw) fp16: its blend mask
    float* dMerged = nullptr;  // (B,3,ph,pw) fp32 or fp16 by outHalf: the tweens RIFE's last step makes from them
    // live with the model frame = the presented rect (no resize after the model): k_rifeBlendOut stores each
    // tween straight into its slot and dMerged is never allocated
    bool blendOut = false;
    // RIFE's two domains (offline and live RIFE): the IFNet and its
    // encode run on the motion frames dM, the finished picture in SDR sRGB at the decoded size
    // (offline) or at the working size, the capture size when the working size enlarges it (live)
    // (mw x mh, padded to mph x mpw), and k_rifeBlend applies their flow and mask to the pictures dX
    // at the working size. mph = 0: the model's frame is the picture's (ph x pw), the IFNet reads dX / dXh.
    int mph = 0, mpw = 0, mw = 0, mh = 0;
    // plain RIFE (two domains, both routes) as its author pads: zeros to a multiple of 128 (Practical-RIFE at scale 1)
    // in the motion frames and under the blend's taps, instead of the edge replicated to a multiple of 64 (DRBA's pad);
    // live reads it off the handoff's zpad=1; SMV_RIFE_AUTHOR_PAD=0 = the old pad
    bool zeroPad = false;
    uint16_t* dM = nullptr; // (1,6,mph,mpw) fp16: prev in planes 0..2, cur in 3..5
    float* dMFit = nullptr; // live: the working-size picture shrunk to mw x mh (3 fp32 planes)
    int fCur = 0;
    cudaEvent_t capEv = nullptr;
    std::vector<cudaEvent_t> slotEv;
    // live: the capture read's GPU time stamp, where a group's own GPU time starts (grEv below)
    cudaEvent_t pfEv[1] = {};
    // live: a group's own GPU time for the present loop's throttle, from pfEv[0] (the capture read): grEv[0] where the
    // slots start (after the pair's model work), grEv[1] after the last slot. Smoothed per kind of group, in
    // microseconds: gIntUs = the work before the slots of a group with tweens, gTweenUs = one tween with its store
    cudaEvent_t grEv[2] = {};
    bool grArmed = false;
    uint32_t grTween = 0, grSlots = 0;
    double gIntMs = 0.0, gTweenMs = 0.0;
    std::atomic<uint32_t> gIntUs{0}, gTweenUs{0};
    // Smooth Motion's midpoint tree: one bridge call, and one slot's unpack + store (the slots' time less the calls'),
    // so the exe prices the direct tree ((M - 1) calls + M stores) and the pipelined one (M stores + the longer of
    // the compute thread's calls and instance 0's worker's, each at the pipelined groups' own price: gFrPipeMainUs =
    // the compute thread's chain a call incl. the join, gFrPipeWkUs = a worker call) apart; AMD FSR 4's: one call, one
    // slot, and an x2 group's whole time (gFrBaseUs); 0 = not timed yet
    std::atomic<uint32_t> gFrCallUs{0}, gFrStoreUs{0}, gFrBaseUs{0}, gFrPipeMainUs{0}, gFrPipeWkUs{0};
    double gFrBaseMs = 0.0;
    // AMD FSR 4's level-1 call (made before the slots at every ladder): its FSR span on the stream (fgL1Ev, from the
    // fence signal to the result's wait), that span in the x2 groups that set the base (gFrL1BaseMs) and the call
    // price over it as x4 and deeper groups show it (gFrL1Ratio), so an x2 group keeps the call price current; priced
    // by x4 and deeper groups alone, a load that passed while x4 fell would keep pricing x4 out for the session. Two
    // sets by group (fgEvSet = the one this group records): a pipelined group's call may still run when the next
    // group starts (its pack does not wait for it), so its spans are read once done, without a wait
    cudaEvent_t fgL1Ev[2][2] = {};
    bool grL1Armed[2] = {};
    int fgEvSet = 0;
    double gFrL1Ratio = 0.0, gFrL1BaseMs = 0.0;
    // AMD FSR 4's pipelined route (SMV_FSRFG_PIPE) is priced off three parts every group times on either route: the
    // level-1 Optical Flow on fgOfL1St (fgOfEv: the direct route hops there too), the level-1 call's FSR span
    // (fgL1Ev) and the session stream's pre outside that chain (the pipelined pre, or a direct pre less the two); the
    // exe's pipelined cost = the longest of the Optical Flow chain, the session stream's chain (the deeper calls + the
    // later of the stores + pre and the next level-1 call) and half the loop through both. 0 = not timed yet
    cudaEvent_t fgOfEv[2][2] = {};
    bool grOfArmed[2] = {};
    // a pipelined group's slots start with the wait for the shown pair's level-1 node (grLagEv after it, armed in
    // nativeFsrfgLagTween): its deeper calls are priced from there
    cudaEvent_t grLagEv = nullptr;
    bool grLagArmed = false;
    double gFgPreMs = 0.0, gFgOfMs = 0.0, gFgL1Ms = 0.0;
    std::atomic<uint32_t> gFgPreUs{0}, gFgOfUs{0}, gFgL1Us{0};

    CUmodule cuMod = nullptr;
    std::wstring kernCacheDir; // the cubin folder nativeBuildKernels was given (Anime4K's module caches there too)
    CUfunction fPackInDirect = nullptr, fResizeH = nullptr, fResizeV = nullptr, fH2f = nullptr, fF2h = nullptr,
               fPackOut = nullptr;
    CUfunction fPackInDirectHdr = nullptr, fPqPlanar = nullptr, fResizeHf = nullptr, fPackOutHdr = nullptr;
    CUfunction fSdrEncode = nullptr, fThdrColor = nullptr;                                    // live TrueHDR
    CUfunction fThdrIn = nullptr, fSharpThdrIn = nullptr, fPqOut = nullptr, fPqLut = nullptr, // TrueHDR in, the
        fSdrPq = nullptr;                                                                     // offline PQ emit
    CUfunction fFitPlanar = nullptr, fSharpPlanar = nullptr,                                  // sharpen
        fPackBgraRgb = nullptr, fUnpackBgraRgb = nullptr,                   // FSR upscaling / FRUC: (R, G, B) <-> BGRA8
        fPackRgbaH = nullptr, fRgbaHAbgr = nullptr, fUnpackRgbaH = nullptr, // FSR frame generation: RGBA16F frames,
        fMvHalf = nullptr, fMvHalfG1 = nullptr, fMvDerive = nullptr,        // the Optical Flow input, the vectors
        fBgraRgbaFsr = nullptr, fRgbaBgra = nullptr, fOddEdge = nullptr, // FSR upscaling: BGRA8 <-> RGBA8; VSR's slot
        fFuMvScale = nullptr,                                            // its chained passes' vectors
        fPackBgra = nullptr, fUnpackRgba = nullptr, fPackR10 = nullptr,  // offline DLSS 4.5 frames, RTX VSR
        fUnpackR10 = nullptr;
    CUfunction fFitAaH = nullptr, fFitAaV = nullptr, fRsTab = nullptr;           // the separable fit, its tap tables
    CUfunction fFsrHdrIn = nullptr, fFsrHdrOut = nullptr, fFsrHdrPack = nullptr; // AMD FSR upscaling on HDR planes
    CUfunction fPackOutV = nullptr, fPackOutHdrV = nullptr;                      // its vertical pass + the slot store
    CUfunction fRestIn = nullptr, fRestFoldH = nullptr, fRestFoldV = nullptr,    // live Restore
        fRestFoldHSub = nullptr, fA4kFoldHSub = nullptr, fA4kFold2x = nullptr, fRestToF = nullptr, fClamp01 = nullptr;
    CUfunction fRestConv0 = nullptr, fRestConvMid = nullptr, fRestConvLast = nullptr; // Restore on the host's kernels
    CUfunction fAjFirst = nullptr, fAjAtt = nullptr, fAjCat = nullptr;                // AnimeJaNai's own three
    CUfunction fPadPlanar = nullptr; // the decoded frame into the model frame
    CUfunction fNrIn = nullptr, fNrOut = nullptr, fNrMv = nullptr, fNrMvClear = nullptr; // DLSS 5
    CUfunction fNrInPq = nullptr, fNrOutPq = nullptr;                 // live DLSS 5 on HDR (PQ) planes
    CUfunction fHalf = nullptr, fPyr = nullptr, fSplatSoft = nullptr, // live GMFSS glue (5b)
        fSplatNorm = nullptr;
    CUfunction fGsCount = nullptr, fGsScanA = nullptr, fGsScanB = nullptr, fGsScanC = nullptr, fGsFill = nullptr,
               fGsGather = nullptr;                           // the gather splats
    CUfunction fGsGatherNhwc = nullptr;                       // the feature splats on channels-last features
    CUfunction fPackInRaw16 = nullptr, fPackInRaw8 = nullptr, // offline
        fPackOutRaw16 = nullptr, fPackOutRaw8 = nullptr, fExpand8to16 = nullptr;
    CUfunction fPairDiff = nullptr, fRawDiff = nullptr;                       // identical-pair test, DLSS 5 reuse test
    CUfunction fNvofLuma = nullptr, fNvofUp = nullptr, fNvofMetric = nullptr; // the nvof model
    CUfunction fSplatVel = nullptr, fVelNorm = nullptr, fPpDown = nullptr,    // its pull-warp tween
        fPpTop = nullptr, fPpUp = nullptr, fBlur1 = nullptr, fNvofCompose = nullptr;
    CUfunction fDrFlowSplat = nullptr, fDrFlowNorm = nullptr, // native DRBA
        fDrDrmSplat = nullptr, fDrDrmNorm = nullptr;
    CUfunction fMotionIn = nullptr, fRifeBlend = nullptr, // RIFE's motion frame, its last step (the tween: fp16
        fRifeBlendF = nullptr;                            // store, fp32 store)
    CUfunction fRifeBlendOut = nullptr;                   // its last step fused with a 1:1 slot store (blendOut)
    CUfunction fRifeBlendOut32 = nullptr;                 // the same for fp32 pictures
    CUfunction fBilin = nullptr;                          // PyTorch's bilinear resize (DRBA as its author)
    CUfunction fHdrEnc = nullptr, fHdrFromSdr = nullptr, fHdrRange = nullptr; // GMFSS / Restore on HDR planes
    CUfunction fPackBgraSdr = nullptr, fUnpackBgraSdr = nullptr;              // FRUC's SDR pairs on HDR planes
    CUfunction fHdrRestOut = nullptr;                                         // Restore's way back on HDR planes
    CUfunction fLocalCorr = nullptr;                                          // gmflow's local correlation, fused
    CUfunction fAttn2 = nullptr;      // gmflow's global matching and propagation (k_attn2)
    CUfunction fFuTail = nullptr;     // GMFSS fusion's pixel shuffle + conv_last on the cut graph's out (k_fuTail)
    CUfunction fFuTailNhwc = nullptr; // the same on a channels-last out (k_fuTailNhwc)
    CUfunction fEncC01 = nullptr, fEncC2Up = nullptr; // RIFE's encode (nativeEncodeRun)
    int* dStaticFlag = nullptr;                       // device flag k_pairDiff sets when the pair differs
    int* hStaticFlag = nullptr;                       // pinned readback of it, one int per group
    uint64_t staticN = 0;                             // identical pairs held this session

    // ---- protocol plumbing
    std::mutex mMsg, mTok;
    std::condition_variable cvMsg, cvTok;
    std::deque<std::vector<uint8_t>> msgs;
    std::deque<uint32_t> toks;
    std::thread th;
    volatile LONG dead = 0;
    bool started = false;
    std::mutex mInit;
    std::condition_variable cvInit;
    int initState = 0; // 0 pending, 1 ready, -1 failed
    uint32_t seq = 0;
    bool havePrev = false;

    // ---------------- lifetime ----------------
    void pushTok(uint32_t v)
    {
        {
            std::lock_guard<std::mutex> lk(mTok);
            toks.push_back(v);
        }
        cvTok.notify_one();
    }
    void die(const char* why)
    {
        LOG("native: %s\n", why);
        InterlockedExchange(&dead, 1);
        // pass through both locks before the wake: a waiter that has tested isDead but not
        // yet blocked would miss a notify sent in that gap
        {
            std::lock_guard<std::mutex> lk(mTok);
        }
        {
            std::lock_guard<std::mutex> lk(mMsg);
        }
        cvTok.notify_all();
        cvMsg.notify_all();
    }
    bool isDead() const
    {
        return InterlockedCompareExchange((volatile LONG*)&dead, 0, 0) != 0;
    }

    bool write(const void* p, DWORD n)
    {
        if (isDead())
            return false;
        const uint8_t* b = (const uint8_t*)p;
        {
            std::lock_guard<std::mutex> lk(mMsg);
            msgs.emplace_back(b, b + n);
        }
        cvMsg.notify_one();
        return true;
    }
    bool read(void* p, DWORD n)
    {
        uint8_t* d = (uint8_t*)p;
        while (n)
        {
            std::unique_lock<std::mutex> lk(mTok);
            cvTok.wait(lk, [&] { return !toks.empty() || isDead(); });
            if (toks.empty())
                return false;
            const uint32_t v = toks.front();
            toks.pop_front();
            lk.unlock();
            if (n < 4)
                return false;
            memcpy(d, &v, 4);
            d += 4;
            n -= 4;
        }
        return true;
    }
};
// the previous and the current frame's planes in dX (offline never swaps: prev in planes 0..2)
static float* nativeXPrev(const NativeRife& nr)
{
    return nr.dX + (nr.xSwap ? 3 * (size_t)nr.ph * nr.pw : 0);
}
static float* nativeXCur(const NativeRife& nr)
{
    return nr.dX + (nr.xSwap ? 0 : 3 * (size_t)nr.ph * nr.pw);
}

// the reason an early (handoff-time) setup failed, named by the pipe's "the early engine handoff failed" line (cleared
// when the early thread starts): the one line of a failed Live start the app's panel keeps
static char g_earlyWhy[240];
static void earlyWhy(const char* fmt, ...)
{
    va_list a;
    va_start(a, fmt);
    vsnprintf(g_earlyWhy, sizeof(g_earlyWhy), fmt, a);
    va_end(a);
}
