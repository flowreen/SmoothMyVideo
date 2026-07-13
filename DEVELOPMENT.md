# Smooth My Video: Technical & Developer Guide

Build instructions, architecture, the engine CLI, and design rationale. For the product overview see
[README.md](README.md).

## Status

Works end to end. The packaged build is fully self-contained: a recipient extracts the zip and runs
`SmoothMyVideo.exe`, no Python, no pip, no ffmpeg, only the NVIDIA driver. Built and tested on an
RTX 5090 Laptop (Blackwell, sm_120); the CUDA 13 stack (torch 2.13.0+cu130, cupy-cuda13x, TensorRT
cu13) is validated across eager, TensorRT, RTX VSR/HDR and all three codecs.

## Architecture

* **`src/main.ts`**, Electron main: window, open/save dialogs, ffprobe (`-of json`), spawns the engine,
  streams progress, tracks the child so **Cancel** can `taskkill /T /F` it. IPC for the monitor refresh
  rate (match-screen), screen size, and the single-frame preview. Resolves the interpreter as
  `engine/runtime/python.exe` and ffprobe as `engine/bin/ffprobe.exe` (both fall back to PATH); sets
  `PYTHONUTF8` and a writable `SMV_TRT_CACHE`.
* **`renderer/index.html`**, the UI: select/drag a video, a target-fps control, an **FSR** sharpen
  toggle, **Restore**, **Upscale**, a **Codec** selector, an opt-in **NVIDIA RTX** panel (VSR + HDR), a
  **Dolby Vision** panel and an **HDR10+** panel (each a one-tool install), output path, progress + ETA,
  a batch queue (crash-resumable, keeps going past failed files), a live thumbnail, a before/after
  preview pane, and a launch-time new-release notice. Electron
  `require` with `nodeIntegration`; most settings persist in `localStorage` (Restore and RTX Dynamic
  Vibrance deliberately don't, per-session opt-ins).
* **`engine/render.py`**, the render engine and model orchestrator: ffmpeg decode → the chosen
  interpolation model (GMFSS default, RIFE/DRBA, FRUC, DLSS-FG or SVP) → per-frame passes → ffmpeg encode.
  TensorRT backend by default for GMFSS (per-subnet eager fallback; `--no-trt`), NVENC with a CPU
  SVT-AV1 fallback, always fp16, always visually lossless, 10-bit by default. Owns the shared
  plumbing every model uses: probe, track passthrough, pause, crash-resume, progress
  (`PROGRESS k/total` on stderr), live thumbnail and the encoder selection.
* **`engine/trt_runtime.py`**, optional TensorRT backend. Swaps the five GMFSS sub-nets for strongly-typed
  fp16 engines; softsplat + the interpolate glue stay eager. Also engines the RIFE IFNet forward
  (`rife_trtify`) and the `--restore` Real-ESRGAN pass, each keyed by its own weights hash. Engines
  are cached per `(net, shapes, gpu, trt version, weights hash)`; the weights-hash in each filename
  makes the cache self-invalidating on a weight swap (stale engines deleted at next start).
* **`engine/rife_backend.py`** + **`engine/rife/`**, the RIFE model backend: vendored
  Practical-RIFE 4.26 heavy (MIT, weights bundled) exposing the same pair interface as GMFSS,
  plus the DRBA triple interface (DistanceRatioMap timing) the engine's DRBA window loop drives.
* **`engine/svp_backend.py`**, the SVP model backend: generates the standalone VapourSynth host
  process that runs svpflow (SVP 4's plugin DLLs) at a max-quality offline profile and streams y4m
  back to the engine; also documents the profile derivation and the block-size caution.
* **`engine/nvoffruc.py`** + **`engine/nvoffruc/`**, the "NVIDIA Smooth Motion" ctypes bridge to
  NVIDIA's NvOFFRUC library (user-installed DLLs, same pattern as the RTX folder).
* **`engine/dlssg.py`** + **`engine/dlssg/`**, the DLSS frame-generation bridge.
* **`engine/rcas.py`**, the FSR RCAS sharpen kernel (shared by render and preview).
* **`engine/rtxvideo.py`** + **`engine/rtxvideo/`**, the RTX Video bridge (VSR + TrueHDR) over a small
  compiled CUDA DLL (`rtxvideo_cuda.dll`, sources in `build_src/`). The non-redistributable NGX feature
  DLLs are user-installed via the in-app NVIDIA RTX panel; the whole folder is gitignored / excluded from
  the zip, so RTX stays a local feature.
* **`engine/realesr.py`**, the `--restore` Real-ESRGAN detail pass (vendored SRVGGNetCompact, BSD-3).
* **`engine/hdr10_meta.py`**, pure-stdlib ISOBMFF injector for HDR10 static metadata (`mdcv`/`clli`) and
  the Dolby Vision configuration box (`dvvC`, via `inject_dv_config`); shared box-insertion surgery.
* **`engine/preview.py`**, single-frame before/after preview (same passes, same order as a render).
* **`engine/runtime/`**, bundled relocatable Python 3.14 (python-build-standalone) with the CUDA 13 GPU
  stack. Gitignored (see Setup).
* **`engine/bin/`**, bundled shared-build `ffmpeg.exe` + `ffprobe.exe` and their DLLs. Fetched, not committed.
* **`engine/GMFSS_Fortuna/`** (model + `train_log/` weights) and **`engine/realesr-animevideov3.pth`**,
  committed to the repo.

## Setup (fresh clone)

Weights ship in git. Two pieces are fetched/copied: `engine/bin` (ffmpeg, ~137 MB) and the ~5.7 GB
`engine/runtime` (Python). Both must be present before `npm run dist`.

**1. Deps + ffmpeg**
```
npm install
node scripts/fetch-ffmpeg.js
```
`fetch-ffmpeg.js` downloads the BtbN win64 **LGPL shared** ffmpeg into `engine/bin` (idempotent; skips if
present; the app falls back to PATH ffmpeg otherwise, or the GUI "Choose .zip" button). If the Electron
binary didn't download: `node node_modules/electron/install.js`.

**2. Python runtime → `engine/runtime`** (the only gitignored piece)
* *Easy:* copy `resources/engine/runtime` out of any packaged build (extract a release zip), it's the
  ready-to-run interpreter, nothing else to do.
* *From scratch:* unpack a
  [python-build-standalone](https://github.com/astral-sh/python-build-standalone/releases) CPython 3.14
  `install_only` win64 build to `engine/runtime`, then:
```
engine\runtime\python.exe -m pip install torch==2.13.0 torchvision --index-url https://download.pytorch.org/whl/cu130
engine\runtime\python.exe -m pip install -r engine\requirements.txt
```
`requirements.txt` pulls cupy-cuda13x, the **unsuffixed** `nvidia-cuda-nvrtc` / `nvidia-cuda-runtime` cu13
wheels (the `-cu13` names are deprecated placeholders that fail to build), `tensorrt` (cu13), and
onnx/onnxscript. A `python -m venv` is **not** usable, a Windows venv isn't relocatable and breaks the
portable bundle.

### Refreshing bundled binaries
* **ffmpeg:** delete `engine/bin` and re-run `node scripts/fetch-ffmpeg.js`. To pin an exact build, drop a
  matched `ffmpeg.exe` + `ffprobe.exe` + `*.dll` set in by hand, never mix DLLs across builds (the exe
  links specific SONAME majors like `avcodec-63`).
* **Weights:** the GMFSS `train_log` pkls (from the GMFSS_Fortuna release) and `realesr-animevideov3.pth`
  (Real-ESRGAN v0.2.5.0). Both committed; this is only for updating them.

## Scripts
* `npm start`, build (`tsc`) and launch.
* `npm run dist`, the build command: wipes `release/`, compiles with `tsc`, runs electron-builder, and
  zips the result into `release/SmoothMyVideo-<version>-win.zip` (~4 GB with TensorRT bundled). The
  multi-GB staging folder is deleted once the zip passes a size sanity check, so the zip is the only
  artifact left; to inspect the unpacked app, extract the zip. Recipients extract and run
  `SmoothMyVideo.exe`; nothing required on the target but the NVIDIA driver. (A zip, not an NSIS
  installer, `makensis` can't memory-map an archive this large.)
* `npm run lint`, one command that does everything: Prettier formats `src/**/*.ts` (writes), then ESLint
  lints `src`, then pyright lints `engine`. Stops at the first failure. See below.
* `engine\runtime\python.exe scripts\smoke.py [--full] [--trt]`, the render smoke tests: real engine runs
  on `samples/test.mp4` asserting frame counts, VFR duration preservation, `.part` promotion and (with
  `--full`, when their runtimes are installed) the HDR10 boxes, DV configuration record and HDR10+ SEI.
  Run it after every engine change; eager renders are not bit-deterministic, so the checks are
  structural, never checksums.

## Engine CLI (used by the GUI, also runnable directly)
```
engine\runtime\python.exe engine\render.py <input> <multi> [output] [--scale 1.0] [--fps TARGET] [--no-trt] [--sharpen S] [--restore] [--no-interp] [--rife] [--rife-drba] [--fruc] [--svp] [--svp-nvof] [--no-passthrough] [--upscale F] [--codec hevc|av1|vvc] [--out-bits 8|10] [--rtx-vsr] [--rtx-hdr] [--dv] [--hdr10plus] [--hdr-nits N] [--hdr-color vivid|rtx|raw] [--hdr-vibrance B] [--hdr-satboost S] [--hdr-mastering-prim display-p3|dci-p3|bt2020|bt709]
```
* `<multi>` integer multiplier, or `--fps TARGET` to resample to any output fps (the model interpolates at
  arbitrary fractional timesteps; `<multi>` is required positionally but ignored when `--fps` is given).
* `--scale F` optical-flow resolution factor (GMFlow already runs at half the source; this scales it
  further). **Auto by default**: 1.0 below 4K, 0.5 for 4K+ sources. GMFlow's global attention grows
  super-linearly with area and dominates the interpolation wall, so quarter-resolution flow at UHD
  (still 1080p-class motion detail) makes a 4K render cost barely more than a 1080p one; verified
  equal-or-slightly-better against ground truth (dropped-frame reconstruction: mean tween PSNR 28.3 vs
  27.9 dB, same worst frame). Pass an explicit value to override.
* `--sharpen S` (0..1) FSR-style RCAS on every output frame (bare `--sharpen` = 0.8; off unless given).
* `--no-interp` re-encodes at source fps with sharpen only (no model/TRT loaded).
* `--fruc` "NVIDIA Smooth Motion": interpolates on the OFA hardware via NVIDIA's NvOFFRUC library
  instead of GMFSS (lower quality; ghosts on fast/large motion, inherent to the optical-flow model).
  Needs `NvOFFRUC.dll` + `cudart64_110.dll` user-installed into `engine/nvoffruc` from the Optical
  Flow SDK .zip (the GUI's Smooth Motion checkbox offers a one-time installer); GMFSS stays the
  default and the quality path. `--fruc-native` is a debug variant (real frames + FRUC tweens).
* `--rife` "RIFE": interpolates with Practical-RIFE 4.26 heavy (vendored under `engine/rife`,
  MIT, weights bundled) instead of GMFSS - the strongest open general-purpose model, the
  recommendation for live action (GMFSS stays the anime specialist). Same on-grid/--fps timing,
  passes, pause and crash-resume as GMFSS; TensorRT-accelerated like GMFSS (the IFNet forward is
  engined per resolution, `--no-trt` forces eager fp16).
* `--rife-drba` (the GUI's RIFE model with "Preserve anime pacing (DRBA)" on): RIFE tweens with
  DistanceRatioMap timing (routineLife1/DRBA) - pans smooth fully while character motion keeps
  closer to its original cadence, which also avoids forced-midpoint warping artifacts. Renders
  on the uniform offset grid for integer `--multi` too (the adjusted timing is the feature); the
  first window after a start or resume seam falls back to plain pair RIFE (deterministic).
* `--svp` (the GUI's SVP model with its "NVIDIA Optical Flow" sub-option off): interpolates with
  the svpflow engine (block-matching vectors + GPU rendering) whose two plugin DLLs are
  borrowed from a local SVP 4 installation
  (`SMV_SVP_DIR` overrides the default `C:\Program Files (x86)\SVP 4`; SVPManager need not run,
  and SVP's optional mpv component is NOT needed). The engine spawns a sibling python on the
  bundled runtime that hosts svpflow in the runtime's own VapourSynth wheel (`vapoursynth==77`,
  PINNED - svpflow is a deprecated-API3 plugin, re-verify a render before any bump): bundled
  ffmpeg decodes the source, a sliding-window frame cache feeds the filter (no VS source plugin),
  and the already-interpolated stream leaves as y4m; the bundled ffmpeg converts it to the raw
  frames the engine expects, and the engine's per-frame passes + encode run unchanged (1:1 loop).
  The three svpflow parameter strings are a max-quality offline profile derived from SVP's own
  generate.js mappings: uniform interpolation (every pair, no adaptive holding), blend at scene
  cuts instead of repeating, shader 13 "Standard", no artifact masking, finest 8px vector grid
  with the largest search radius, strongest wide search and a refine pass (see
  `_svp_host_script` in the engine, including the block:{w:32} caution); the GPU device id is
  read from the user's own SVP settings (frc.cfg), and `SMV_SVP_ALGO` (env) overrides the
  shader for comparisons. Pause and crash-resume work like the other models (a resumed run trims
  the host to the banked output-frame count). SmoothFps renders at 16-bit 4:2:0 precision for
  every source depth (only the vector-search clips drop to 8-bit, mirroring SVP's own scripts)
  and the SVP output is always 10-bit, `--out-bits 8` included, so tween blends never band.
  Known v1 limits: chroma rides at 4:2:0 through svpflow, and the clip length is the
  container's frame count (a tail frame can repeat or drop on containers with wrong metadata).
* `--svp-nvof` (the GUI's SVP model with "NVIDIA Optical Flow" on, the GUI default): same SVP
  pipeline, but the motion vectors come from the NVIDIA Optical Flow hardware (svpflow's
  `SmoothFps_NVOF`, fed a dense 4px-grid P8 vector clip exactly as SVP's own generator builds
  it) instead of SVP's block-matching search. Needs SVP 4 plus a Turing-or-newer NVIDIA GPU;
  same parameters and limits as `--svp` (except shaders >=21, which `SmoothFps_NVOF` ignores).
* `--restore` runs the Real-ESRGAN detail pass per output frame, before the upscale (works with `--no-interp`).
* `--upscale F` spatial upscale just before encode (bare = 1.5, clamp 16.0; above 8192 px auto-switches to
  a CPU AV1/VVC encoder). `--rtx-vsr` uses RTX Video Super Resolution, else bicubic.
* `--rtx-hdr` SDR→HDR10 (BT.2020 PQ) via TrueHDR; `--hdr-nits` mastering peak (400..2000, default 1000);
  `--hdr-color` {`vivid` (default: source hue+chroma), `rtx` (SDK saturation, hue-corrected), `raw` (debug)};
  `--hdr-mastering-prim` sets the `mdcv` gamut by name.
* `--dv` additionally exports a **Dolby Vision Profile 8.1** MP4 (needs `--rtx-hdr`, HEVC, MP4 out, and
  user-installed `dovi_tool` in `engine/dvtools`). See the Dolby Vision section below; GPAC-free.
* `--hdr10plus` additionally embeds **HDR10+** (SMPTE ST 2094-40) dynamic metadata (needs `--rtx-hdr`,
  HEVC, MP4 out, and user-installed `hdr10plus_tool` in `engine/hptools`); combinable with `--dv`. See
  the HDR10+ section below.
* `--out-bits` {`10` default, `8` legacy}; `--codec` {`hevc` default, `av1`, `vvc`}; `--no-passthrough` first-audio-only.
* Per-frame order: (restore →) upscale → RCAS sharpen → TrueHDR.

## How it works (key behaviour)

**Interpolation.** On the default `--multi` path every real source frame passes through at its integer
timestamp at full quality, and M-1 AI-generated tweens are inserted between each pair (`--fps` resamples to
an arbitrary rate off the source grid). Keeping the real frames on-grid is deliberate (2026-07-05): a
bracket blend `inference(f[k-1], f[k+1], 0.5)` would skip the real frame's pose, and interpolating two
already-generated tweens would double-fade it, so the frames we already have are kept at max quality.
Duplicate / near-duplicate frames are interpolated the same as real motion (no held cels), so the source's
own frame timings are preserved exactly.

**Output.** Always visually lossless (no quality knob). HEVC by default (`hevc_nvenc`; CPU `libsvtav1`
fallback with no NVENC), AV1 (`av1_nvenc`) or H.266/VVC (`libvvenc`, CPU) selectable. 10-bit by default
whatever the source depth, so the float-precision interpolated frames never band gradients (a dark-gradient
clip carries 281 luma levels at 10-bit vs 70 at 8-bit). Source colour signalling (matrix/transfer/
primaries/range) is carried through with `setparams`. Every audio, subtitle, chapter and font track is
copied (output auto-switches to `.mkv` when the tracks need it); HDR-into-MKV keeps the full HDR10 metadata
via a two-stage finalize.

**Encode quality.** Quality-first: professional-grade fidelity regardless of size, one standard at every
frame rate (2026-07-10; the earlier high-fps CQ relief was removed under this policy). NVENC runs
constant-quality VBR at max effort: **preset p7 + full-resolution multipass + rc-lookahead 1** (AQ + a
small chroma-QP boost), with CQ values verified against a lossless 8K master: HEVC **CQ 17** (VMAF 99.78 /
57.0 dB / SSIM 0.9986), AV1 **CQ 22**. The effort ladder is what moves quality now, CQ is saturated at max
effort (HEVC CQ 14 to 21 encode byte-identically): measured on the 1080p sample, the ladder lifts HEVC from
50.8 to 53.5 dB (worst frame 49.3→52.3) and AV1 from 51.2 to 53.1 dB at modest size cost. The gain needs
multipass and lookahead *together* (either alone measures ≈0), and shallow beats deep: depths 1/2/4 encode
byte-identically and measure ~1 dB *better* than 8 to 32 (deep queues enable B-frame restructuring that
trades fidelity for size, the wrong trade here) while using a single lookahead slot of VRAM. Above 120 fps
output the queue is dropped entirely (on tween-dense streams any lookahead measured −1 dB, so high-fps
renders get better fidelity and zero lookahead VRAM at once). `tune uhq` was evaluated and rejected: its
temporal filtering rewrites frame content.
VVC runs **QP 17** with perceptual QP adaptation always off (QPA trades fidelity in "unnoticed" regions for
size, and inverts outright on 120+ fps tween streams); the SVT-AV1 fallback runs **CRF 17 preset 6**. The
`SMV_CQ` env var overrides any CQ for measurement work.

**Size projection + disk check.** During a render the engine emits `SIZE cur projected` beside each
PROGRESS heartbeat (bytes written so far, linearly extrapolated), so the GUI shows the expected final size
next to the ETA within the first minutes of a long render. A one-time warning fires early when the
projection (doubled for the HDR-into-MKV two-stage, whose temp and final coexist) exceeds the free space
on the output drive.

**Upscale + RTX.** `--upscale` to any resolution up to 16K (RTX VSR, or bicubic fallback). Past 8192 px
NVENC/HEVC can't encode, so the engine probes CPU encoders at the output size and auto-switches
(SVT-AV1 → VVC), plus a **fail-closed RAM preflight** (true 16K needs ~54 GB free; the CPU encoders keep
dozens of large frames in flight). RTX HDR is a real HDR10 master: 10-bit BT.2020 PQ + injected
mastering-display / content-light metadata, with source-faithful (cyan-free) colour rebuilt in ICtCp
(TrueHDR itself rotates hues even at Saturation 0, so its chroma is dropped and the source's is transplanted).

**Dolby Vision (`--dv`).** Layers a **Profile 8.1** RPU on top of the HDR10 render, HDR10-compatible, so
non-DV players fall back to HDR10 (mdcv/clli) and DV displays read the dynamic metadata. **GPAC-free**: the
one external tool is `dovi_tool` (open source, user-installed in `engine/dvtools` via the UI's "Dolby Vision"
panel); the RPU is muxed by the bundled ffmpeg and the DV configuration box (`dvvC`) is written in-engine by
`hdr10_meta.inject_dv_config` (same ISOBMFF surgery as the HDR10 boxes, the LGPL ffmpeg can't emit `dvvC`
itself). Flow (`_dv_export` in render.py): during the HDR render `rtxvideo.run_hdr` accumulates **per-frame
L1** (min/avg/max PQ brightness), near-free, reusing the MaxCLL reduction, and only when `--dv` is set, then
after encode: extract HEVC → `dovi_tool generate` (one L1 shot/frame) + `inject-rpu` → ffmpeg mux with the
audio → inject `dvvC` + HDR10 fallback boxes. **B-frames are disabled (`-bf 0`) for DV renders**: dovi_tool
needs an Annex-B elementary stream, and a raw-HEVC→MP4 copy assigns non-monotonic DTS with a reorder buffer
and silently drops the tail frames; no B-frames ⇒ coding order = display order ⇒ exact remux + 1:1 RPU
alignment. Requires HEVC + MP4 out; skipped with a notice otherwise. Best-effort, any failure keeps the
HDR10 file. Legal note: `dvvC` is our own code writing a documented public box format (no Dolby/GPAC source,
no patented processing), so the exposure is only the "Dolby Vision" **trademark**, the UI labels it
"Profile 8.1 (experimental)", not "certified", credits dovi_tool, and carries a non-affiliation
disclaimer (also in the README). Dolby and Dolby Vision are trademarks of Dolby Laboratories; this
project is independent, not affiliated with or endorsed by Dolby, and bundles no Dolby software. The before/after preview does **not** change with `--dv`: the base pixels are
identical HDR10; the DV difference is display-side tone-mapping we can't (and shouldn't) simulate.

**HDR10+ (`--hdr10plus`).** Embeds **SMPTE ST 2094-40** dynamic metadata into the HDR10 render, measured
per frame during `rtxvideo.run_hdr` (per-channel MaxScl, average maxRGB and a maxRGB percentile
distribution, computed from a 1024-bin histogram of the 10-bit PQ codes, near-free like the DV L1 pass;
`collect_hp`). After encode, `_hp_export` extracts the HEVC, writes the metadata JSON in the layout
`hdr10plus_tool` itself extracts from real masters (Profile A, per-frame SceneInfo; luminance in 0.1-nit
units; DistributionValues = [p1, p99.98, bright-pixel fraction, p25..p99]), injects the SEI with the
user-installed **hdr10plus_tool** (`engine/hptools`, same one-tool install flow as dovi_tool) and remuxes
with the audio. The SEI rides inside the samples (no container box needed), so HDR10+ runs FIRST and a
following DV export passes it through, which is why `--dv --hdr10plus` can coexist on one file. Same
`-bf 0` rule as DV (exact raw-ES remux), MP4 + HEVC only, best-effort with HDR10 fallback. Trademark note:
the app never claims certification; metadata is produced by the third-party open-source tool.

**Failure-safe output (`.part`).** Every render writes to `<name>.part.<ext>` and promotes it with an
atomic rename only at success, so a cancelled, crashed or failed render can never leave a silently
truncated file at the final path or destroy an existing good file it was about to replace. The GUI
deletes the `.part` remnant after a Cancel.

**VFR sources.** When a source's average frame rate disagrees with its container rate (`avg_frame_rate`
vs `r_frame_rate`, >0.5%), the stream is variable-frame-rate (phone clips, screen recordings) and the
container rate is usually the useless max instantaneous rate; the engine then decodes at a constant
average rate (`-fps_mode cfr`) and derives all output timing from it, so duration and audio sync are
preserved (a notice is logged).

**Restore.** `--restore` runs Real-ESRGAN's anime-video model per output frame to clean compression noise
and redraw linework (a generative repaint, it targets cel-style anime and can flatten fine texture).
~+50% wall at 2× 1080p; runs through the same per-resolution TensorRT cache as the GMFSS sub-nets.

**FSR sharpen.** AMD FidelityFX **RCAS** at the output resolution crisps the softer generated tweens; on by
default at 1.0. It limits its lobe to the neighbour min/max (no overshoot/ringing), eases off in noisy
regions, and applies one scalar per pixel to all channels (so it can't decorrelate them into colour speckle).

**Deterministic renders.** The GMFSS path is bit-deterministic: the same command produces a
byte-identical output file, run after run, on both the TensorRT and eager paths (verified by md5;
`scripts/smoke.py --full` asserts it). Two changes made this true (2026-07-09): softsplat's forward
splat accumulates in **int64 fixed point** (integer addition is associative, so thread scheduling
can't reorder a float sum; this was the single nondeterministic stage, isolated by a per-stage
bit-exactness probe), and `cudnn.benchmark` is off with deterministic algorithm selection (benchmark
mode could pick different conv algorithms per process). Output differs from the old float-atomics
kernel by ~1e-4 max (82 dB, the old kernel's own run-to-run jitter level). Cost: the int64
accumulator doubles the splat's memory traffic: invisible at 2x, roughly +15% inference time at
very high multipliers. The RTX passes (VSR/TrueHDR) are NVIDIA NGX black boxes with no determinism
contract, so HDR/VSR renders are outside the byte-identical guarantee.

**Preview / batch / live.** A before/after pane runs the same passes on a single frame (click for 1:1
pixels); a batch queue renders picked/dropped files back to back; a ~1/s live thumbnail shows the graded
output frame during a render (near-zero render cost, a producer/worker split keeps the render thread only
snapshotting).

## Performance
fp16 + a cupy softsplat kernel is the base (~2.2× over the original fp32 path). The **TensorRT** backend
(the five sub-nets as strongly-typed fp16 engines, built and cached per resolution on first use) adds
~2.2× end to end, numerically matching. A 2026-06 code audit found the pipeline **GPU-compute-bound** at
its practical limit on this hardware: I/O and host-sync changes are perf-neutral, batching the per-timestep
nets doesn't help (FusionNet saturates at batch 1), and fp8 fails a quality gate (GMFlow's flow range
overflows e4m3 → 61 px outliers). `torch.compile` and dynamic-shape engines were both ruled out (shipping a
JIT compiler breaks the no-deps promise; `grid_sample` has no dynamic-ONNX path). Two non-regressing
cleanups were kept: GPU-side transposes and a single shared CUDA stream with no per-call TRT sync.
A 2026-07 re-audit confirmed the compute-bound conclusion and closed calibrated FP8 with data
(FusionNet: no speedup; GMFlow: 1.3x but with disqualifying flow outliers). The two wins that DID
land: flow is estimated at an automatic resolution-appropriate scale (4K renders near 1080p cost,
see `--scale`), and the engine warns at startup when the GPU's power limit sits below its board
default (a laptop Silent profile can cost 2-3x wall time; the pipeline is power-bound before it is
anything else).

The **RIFE** backend is engined the same way: its whole IFNet forward (including the interleaved
`grid_sample` warps) exports to one strongly-typed fp16 engine per resolution, while the cheap
feature-head calls stay eager. Measured 1.54× end-to-end on a 60s 1080p 2× render (59s vs 90s
eager), numerically matching the eager path (output-vs-output PSNR ~60 dB / SSIM 0.999). Like GMFSS
it pays a one-time per-resolution build (~100s at 1080p), so `--no-trt` can win for a single one-off.

## Constraints
* **CUDA 13 (Blackwell, sm_120):** torch is the cu130 build; cupy-cuda13x finds the runtime via
  `cuda-pathfinder`, so the old `_add_cuda_dll_dirs` nvrtc shim is no longer load-bearing.
* **RTX bridge:** keep the **cu12**-built `rtxvideo_cuda.dll` and ship `cudart64_12.dll` beside it, NGX's
  static import lib is CUDA-12-ABI, so a bridge relinked against CUDA 13 crashes in `create()`
  (see "Building the native bridges" below).
* **Runtime:** keep `engine/runtime` a relocatable python-build-standalone install, never a `venv`.
* **Renderer:** uses `require('electron')` with `nodeIntegration`, so it can't run in a plain browser,
  launch via `npm start`, the shortcut, or the vbs.

## Releasing

The zip is ~4.4 GB, which exceeds GitHub's 2 GiB release-asset cap, so binaries live on SourceForge
and GitHub carries the release page (notes + `.sha256`). The ceremony:

1. Bump `version` in package.json, commit and push.
2. `npm run dist` → `release/SmoothMyVideo-<v>-win.zip` + `.sha256` (staging folder auto-cleans),
   then `git archive --format=zip -o release/SmoothMyVideo-<v>-src.zip HEAD` for the source snapshot.
3. Upload BOTH zips to SourceForge (web UI caps at 500 MB, use SFTP; create the `<v>` folder on the
   Files tab first, scp does not mkdir):
   `scp release/SmoothMyVideo-<v>-win.zip flowreen@frs.sourceforge.net:/home/frs/project/smoothmyvideo/<v>/`
   then on the Files tab mark the new win.zip as the default Windows download (the ⓘ icon).
   The src.zip is not optional: SF hosting is "solely for Open Source software development" and a
   binary-only project on a fresh account was removed without notice (2026-07-11, the original
   1.0.0 project vanished 1-2 days after creation; recreated with MIT license category, GitHub
   homepage, full description AND the source zip alongside the binary).
4. Tag `v<v>` on the release commit and push the tag (Sourcetree: right-click → Tag → "Push tag").
5. Create the GitHub release for the tag: paste the notes, attach the `.sha256`.

The tag is what installed copies compare against (`checkForUpdate` in main.ts), so publishing the
GitHub release is what lights up the in-app "new version" notice for existing users. **Never re-upload
different code under an existing version**: the version string is baked into the zip, the checksum
stops matching, and same-version installs never see the update notice. Fix-ups ship as a patch version.

**Retention policy: keep the latest and one previous zip on SourceForge; delete older ones.** The
previous build is the rollback/diagnostic escape hatch when a new release misbehaves on some setup;
anything older is disk hygiene. New users always get the newest build (the README button and
SourceForge's default download resolve to it), and GitHub release pages (notes + checksums, a few KB)
are kept forever so the changelog and hashes survive even for pruned binaries.

## Linting & formatting
All dev-only, all in `node_modules` (never shipped, `dist` bundles only `dist/`, `renderer/`, and the
`engine` extraResources, not `node_modules`). Deliberately a **light touch**: the engine Python and the
renderer's inline JS are intentionally dense (long lines, `x; y` one-liners, load-bearing comments), so
nothing reflows them, only `src/*.ts` is auto-formatted.
* **Prettier** (`.prettierrc.json`) formats `src/**/*.ts` only. `.prettierignore` guards `engine/`,
  `renderer/`, and build dirs so a stray `prettier .` can't reflow the hand-tuned files. Config matches the
  existing style (single quotes, semicolons, 2-space, printWidth 120).
* **ESLint** (`eslint.config.js`, flat config, `typescript-eslint` recommended) lints `src/**/*.ts` for real
  bugs, scoped to `src`, engine/renderer excluded. CommonJS config on purpose (no `"type":"module"`).
* **pyright** (`pyrightconfig.json`) lints `engine/*.py` as a **linter, not a type checker**:
  `typeCheckingMode: "off"` so the dynamic torch/numpy/cupy code isn't buried in type noise, only
  high-signal checks stay on (undefined names → error, unused imports/vars → warning). Vendored
  `runtime/`, `GMFSS_Fortuna/`, `trt_cache/`, and the RTX bridge are excluded from checking (`extraPaths`
  still resolves the GMFSS model + local engine modules). No Python-side install needed, pyright runs from
  npm. Ruff is a fine stronger alternative if you later install it (`uv tool install ruff`), but pyright
  keeps everything in the one `npm install`.

Note: `npm install <pkg>` rewrites `package.json` and re-expands its inline arrays (e.g. the `build.filter`
list) to one-per-line; a plain `npm install` / `npm ci` leaves formatting alone. Re-inline by hand if it
bothers you.

Dev dependencies are pinned to caret majors (not `latest`) because `npm run setup` deletes the lockfile:
with `latest`, a fresh setup after a major release would silently pull a breaking toolchain. In particular
**TypeScript stays on `^6` until 7.1**: TS 7 (the Go-native compiler) changed the programmatic API and
typescript-eslint support is slated for 7.1; the compile-speed win is irrelevant at this project's size.

## Dev toolchain
The dev machine has VS 2019 Build Tools (MSVC `cl.exe` 19.29) and VS 2026 Community (`cl.exe` 19.51) +
the Windows 10 SDK, enough to build both native bridges below; the NGX SDK's entry points live in a
static import lib (`nvsdk_ngx_s.lib`), so ctypes alone can't reach them (recipe in "Building the native
bridges" below). Nothing that needs MSVC is bundled; a recipient still needs only the NVIDIA driver.

## Building the native bridges

The engine reaches both NVIDIA runtimes through small locally-built cdecl DLLs driven by ctypes. Their
sources live in `engine/rtxvideo/build_src/` and `engine/nvoffruc/build_src/`; this section is the
build documentation for both.

(The DLSS 4.5 model's host, `engine/dlssg/dlssg2f.exe`, is a standalone exe rather than a ctypes
bridge and has its own build doc at `engine/dlssg/build_src/BUILD.md`. Unlike the bridges here, both
its exe and its Streamline runtime DLLs are redistributable and ship committed/bundled.)

### RTX Video bridge (`rtxvideo_cuda.dll`)

A small CUDA bridge that lets `engine/rtxvideo.py` drive NVIDIA's RTX Video SDK (RTX VSR + TrueHDR).
It statically links the SDK's `nvsdk_ngx_s.lib`, so the built DLL is **not redistributable**; only the
sources are committed. You only need to rebuild it after updating the RTX Video SDK or moving to a
different CUDA runtime.

**Sources** (`engine/rtxvideo/build_src/`):
* `rtx_video_api_cuda_impl.cpp` - the SDK's CUDA convenience layer (`samples/RTX_Video_API/`), copied
  so its `#include "utils.h"` picks up our override below.
* `utils.h` - overrides the SDK's hardcoded `APP_PATH` with an extern global so the model path can be
  set at runtime (`g_rtxv_model_path`).
* `rtxvideo_pathshim.cpp` - defines that global and exports `rtxv_set_model_path(const wchar_t*)`.
* `rtxvideo.def` - the exported C symbols (extern "C", undecorated on x64).

**Toolchain** (verified on this machine): MSVC v142 (VS2019 Build Tools, `cl.exe` 19.29) via
`VC\Auxiliary\Build\vcvars64.bat`; RTX Video SDK at `D:\AIStuff\RTX_Video_SDK` (headers in `include/`,
`nvsdk_ngx_s.lib` in `lib\Windows\x64`, feature DLLs in `bin\Windows\x64\rel`); CUDA headers/libs come
from the bundled torch runtime wheel (no separate CUDA Toolkit needed):
`engine\runtime\Lib\site-packages\nvidia\cuda_runtime\{include, lib\x64}`.

**Recipe** - from an `x64 Native Tools` prompt (or after vcvars64.bat), in `engine/rtxvideo/build_src/`:

```
set SDK=D:\AIStuff\RTX_Video_SDK
set RT=..\..\runtime\Lib\site-packages\nvidia\cuda_runtime
cl /nologo /LD /EHsc /MT /DNDEBUG ^
   /I"%SDK%\include" /I"%SDK%\samples\RTX_Video_API" /I"%RT%\include" ^
   rtx_video_api_cuda_impl.cpp rtxvideo_pathshim.cpp ^
   /Fe:rtxvideo_cuda.dll ^
   /link /DEF:rtxvideo.def /LIBPATH:"%SDK%\lib\Windows\x64" /LIBPATH:"%RT%\lib\x64" ^
   nvsdk_ngx_s.lib cuda.lib cudart.lib user32.lib shell32.lib advapi32.lib
```

Then copy `rtxvideo_cuda.dll` up into `engine/rtxvideo/` next to `nvngx_vsr.dll` + `nvngx_truehdr.dll`
(NGX resolves the feature DLLs relative to the loading module, so co-location is what matters).

Gotchas:
* `/MT` (static CRT) is required - `nvsdk_ngx_s.lib` uses the static CRT; `/MD` gives LNK4098 +
  unresolved CRT symbols.
* `cudart.lib` is required (the NGX static lib references `cudaGetDevice`/`cudaGetDeviceProperties`
  to map the CUDA device to an adapter LUID); `user32`/`shell32`/`advapi32` are also needed.
* The feature DLLs (`nvngx_vsr.dll`, `nvngx_truehdr.dll`) are obtained from the RTX Video SDK and
  placed in `engine/rtxvideo/` - in the app, the in-GUI "Install runtime" button does this.

**CUDA 13 runtimes - do NOT rebuild this bridge against CUDA 13.** NVIDIA's `nvsdk_ngx_s.lib` is built
for the **CUDA 12** runtime ABI: internally it calls `cudaGetDeviceProperties` with a CUDA 12-sized
`cudaDeviceProp`. Link a CUDA 13 `cudart` and the runtime writes the larger CUDA 13 struct into that
smaller buffer, overrunning the stack - the process dies with `0xC0000409` (STATUS_STACK_BUFFER_OVERRUN)
inside NGX `create()`. Verified: a bridge relinked against `cudart64_13` (whether via the wheel's static
`cudart.lib` or a synthesized dynamic import lib) loads fine but crashes in `create()`. We cannot
recompile NVIDIA's lib. So to run under a **CUDA 13** runtime (torch `cu130` etc.), keep the **cu12
bridge exactly as built** and just drop `cudart64_12.dll` next to it in `engine/rtxvideo/` (a cu13
runtime ships only `cudart64_13.dll`, so the bridge's `cudart64_12` import would otherwise be
unresolved). The bridge uses cu12 `cudart` for its read-only device-property / LUID query (matching
NGX's ABI) while torch uses cu13 `cudart` separately; the CUDA **driver** context is shared (driver
API, version agnostic), so VSR and TrueHDR run correctly. Validated end to end on torch 2.12.1+cu130 +
cupy-cuda13x.

### NVIDIA Smooth Motion bridge (`nvoffruc_bridge.dll`)

The bridge between the Python engine and NVIDIA's `NvOFFRUC.dll` (Optical Flow SDK FRUC), i.e. the
"NVIDIA Smooth Motion" interpolation model. Same idea as the RTX bridge: our source compiles to a small
cdecl DLL that ctypes drives; the NVIDIA runtime DLLs stay user-installed.

> This bridge contains source code provided by NVIDIA Corporation (it `#include`s the SDK's
> `NvOFFRUC.h` and `SecureLibraryLoader.h` and follows the NvOFFRUCSample sequence). Ship the built
> DLL, not the SDK.

**Prerequisites:**
* Visual Studio with the Desktop C++ workload. **No CUDA Toolkit is needed** - the bridge uses only the
  CUDA *driver* API from `nvcuda.dll`, so it links only the Win32 crypto libs.
* The **NVIDIA Optical Flow SDK** extracted somewhere (e.g. the `Optical_Flow_SDK_5.0.7` folder;
  EULA-gated download from NVIDIA). We need its headers on the include path:
  `<SDK>/NvOFFRUC/Interface` (`NvOFFRUC.h`) and `<SDK>/NvOFFRUC/NvOFFRUCSample/inc`
  (`SecureLibraryLoader.h`).

**Build** (x64 Native Tools Command Prompt, in `engine/nvoffruc/build_src/`):

```bat
set SDK=C:\path\to\Optical_Flow_SDK_5.0.7

cl /LD /O2 /EHsc /std:c++17 nvoffruc_bridge.cpp ^
   /I "%SDK%\NvOFFRUC\Interface" ^
   /I "%SDK%\NvOFFRUC\NvOFFRUCSample\inc" ^
   /Fe:nvoffruc_bridge.dll ^
   /link crypt32.lib wintrust.lib
```

Verified to compile clean with VS2019 BuildTools (cl 19.29) and VS 2026 Community (cl 19.51).
`SecureLibraryLoader.h` already `#pragma comment`s `crypt32`/`wintrust`; they are listed above too so
the command is copy-pasteable. No CUDA include or link is needed.

**Install** (runtime layout in `engine/nvoffruc/`) - three files sit together:
* `nvoffruc_bridge.dll` - built above. It is ours, so it is **committed and shipped** in the package.
* `NvOFFRUC.dll` - from `<SDK>/NvOFFRUC/NvOFFRUCSample/bin/win64/` (NVIDIA proprietary, user-installed
  via the GUI "Choose .zip" step, not redistributed, gitignored).
* `cudart64_110.dll` - from the same `bin/win64/` folder (NvOFFRUC.dll depends on it; also gitignored).

`SecureLoadLibrary` resolves the bare name `NvOFFRUC.dll` against BOTH the working directory (its
signature / `WinVerifyTrust` calls) and the DLL search path, so the bridge temporarily sets its own
folder as the current directory (plus `SetDllDirectory`) around the load, then restores it. That is
what makes the signed load and its `cudart64_110.dll` dependency resolve regardless of the process
working directory. (Passing a full path instead does not work: the loader hardcodes the bare name.)

**Hardware note:** the Optical Flow hardware (OFA) exists on Turing through **Blackwell**
(RTX 20/30/40/50). Per the SDK's `Deprecation_Notices.pdf` (Jan 2026) the OFA is being removed on GPUs
*after* Blackwell, where this bridge will not function. This is the inferior/faster model on purpose;
GMFSS stays the default.

**Sync fences (2026-07-11):** `nvoffruc_interpolate` calls `cuCtxSynchronize` twice: once at entry
while the CALLER's context is still current (drains e.g. torch's kernels so the input surfaces are
fully written before the bridge's `cuMemcpyDtoD` reads them) and once on its own context before
returning (so the warp + out-copy have landed before the caller reads the output buffer). CUDA does not
order one context's null stream against another context's streams, and without the fences every tween
rendered under torch 2.13 was sliced at horizontal seams (torch 2.12 won the race by timing).
Verified: 5 renders x 24 tweens all seam-free.

**Debugging history** (kept because both cost real time): the original "access violation reading a
device pointer" on the first `Process` was a POINTER-INDIRECTION bug - FRUC's `pFrame` and every
registered `pArrResource` entry must be a `CUdeviceptr*` (the HOST address of the variable holding the
device pointer, exactly as `NvOFFRUCSample` passes `&m_pRenderFrameCudaMemPtr[i]`), and
`nCuSurfacePitch` must be `width*4`. And the priming `Process` call sets `bSkipWarp = 1` (a state-only
feed of I0): without it the prime warps against stale state, polluting the temporal hints that FRUC's
"bad quality → repeat a source frame" fallback depends on. FRUC's tearing on fast anime motion is
inherent to the optical-flow model - it was verified against NVIDIA's own `NvOFFRUCSample` output; the
RTX 5090 (Blackwell) runs the Feb-2023 `NvOFFRUC.dll` correctly once driven this way.
