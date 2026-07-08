# Smooth My Video: Technical & Developer Guide

Build instructions, architecture, the engine CLI, and design rationale. For the product overview see
[README.md](README.md).

## Status

Works end to end. The packaged build is fully self-contained: a recipient extracts the zip and runs
`SmoothMyVideo.exe`, no Python, no pip, no ffmpeg, only the NVIDIA driver. Built and tested on an
RTX 5090 Laptop (Blackwell, sm_120); the CUDA 13 stack (torch 2.12.1+cu130, cupy-cuda13x, TensorRT
cu13) is validated across eager, TensorRT, RTX VSR/HDR and all three codecs.

## Architecture

- **`src/main.ts`**, Electron main: window, open/save dialogs, ffprobe (`-of json`), spawns the engine,
  streams progress, tracks the child so **Cancel** can `taskkill /T /F` it. IPC for the monitor refresh
  rate (match-screen), screen size, and the single-frame preview. Resolves the interpreter as
  `engine/runtime/python.exe` and ffprobe as `engine/bin/ffprobe.exe` (both fall back to PATH); sets
  `PYTHONUTF8` and a writable `SMV_TRT_CACHE`.
- **`renderer/index.html`**, the UI: select/drag a video, a target-fps control, an **FSR** sharpen
  toggle, **Restore**, **Upscale**, a **Codec** selector, an opt-in **NVIDIA RTX** panel (VSR + HDR), a
  **Dolby Vision** panel and an **HDR10+** panel (each a one-tool install), output path, progress + ETA,
  a batch queue (crash-resumable, keeps going past failed files), a live thumbnail, a before/after
  preview pane, and a launch-time new-release notice. Electron
  `require` with `nodeIntegration`; most settings persist in `localStorage` (Restore and RTX Dynamic
  Vibrance deliberately don't, per-session opt-ins).
- **`engine/gmfss_interp.py`**, the GMFSS pipe engine: ffmpeg decode → GMFSS → ffmpeg encode. TensorRT
  backend by default (per-subnet eager fallback; `--no-trt`), NVENC with a CPU SVT-AV1 fallback, always
  fp16, always visually lossless, 10-bit by default. Prints `PROGRESS k/total` to stderr.
- **`engine/trt_runtime.py`**, optional TensorRT backend. Swaps the five GMFSS sub-nets for strongly-typed
  fp16 engines; softsplat + the interpolate glue stay eager. Engines are cached per
  `(net, shapes, gpu, trt version, weights hash)`; the weights-hash in each filename makes the cache
  self-invalidating on a weight swap (stale engines deleted at next start).
- **`engine/rtxvideo.py`** + **`engine/rtxvideo/`**, the RTX Video bridge (VSR + TrueHDR) over a small
  compiled CUDA DLL (`rtxvideo_cuda.dll`, sources in `build_src/`). The non-redistributable NGX feature
  DLLs are user-installed via the in-app NVIDIA RTX panel; the whole folder is gitignored / excluded from
  the zip, so RTX stays a local feature.
- **`engine/realesr.py`**, the `--restore` Real-ESRGAN detail pass (vendored SRVGGNetCompact, BSD-3).
- **`engine/hdr10_meta.py`**, pure-stdlib ISOBMFF injector for HDR10 static metadata (`mdcv`/`clli`) and
  the Dolby Vision configuration box (`dvvC`, via `inject_dv_config`); shared box-insertion surgery.
- **`engine/preview.py`**, single-frame before/after preview (same passes, same order as a render).
- **`engine/runtime/`**, bundled relocatable Python 3.14 (python-build-standalone) with the CUDA 13 GPU
  stack. Gitignored (see Setup).
- **`engine/bin/`**, bundled shared-build `ffmpeg.exe` + `ffprobe.exe` and their DLLs. Fetched, not committed.
- **`engine/GMFSS_Fortuna/`** (model + `train_log/` weights) and **`engine/realesr-animevideov3.pth`**,
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
- *Easy:* copy `resources/engine/runtime` out of any packaged build (local `release/win-unpacked/` or a
  release zip), it's the ready-to-run interpreter, nothing else to do.
- *From scratch:* unpack a
  [python-build-standalone](https://github.com/astral-sh/python-build-standalone/releases) CPython 3.14
  `install_only` win64 build to `engine/runtime`, then:
```
engine\runtime\python.exe -m pip install torch==2.12.1 torchvision --index-url https://download.pytorch.org/whl/cu130
engine\runtime\python.exe -m pip install -r engine\requirements.txt
```
`requirements.txt` pulls cupy-cuda13x, the **unsuffixed** `nvidia-cuda-nvrtc` / `nvidia-cuda-runtime` cu13
wheels (the `-cu13` names are deprecated placeholders that fail to build), `tensorrt` (cu13), and
onnx/onnxscript. A `python -m venv` is **not** usable, a Windows venv isn't relocatable and breaks the
portable bundle.

### Refreshing bundled binaries
- **ffmpeg:** delete `engine/bin` and re-run `node scripts/fetch-ffmpeg.js`. To pin an exact build, drop a
  matched `ffmpeg.exe` + `ffprobe.exe` + `*.dll` set in by hand, never mix DLLs across builds (the exe
  links specific SONAME majors like `avcodec-63`).
- **Weights:** the GMFSS `train_log` pkls (from the GMFSS_Fortuna release) and `realesr-animevideov3.pth`
  (Real-ESRGAN v0.2.5.0). Both committed; this is only for updating them.

## Scripts
- `npm start`, build (`tsc`) and launch.
- `npm run dist`, the build command: wipes `release/`, compiles with `tsc`, runs electron-builder (zip
  target → both `release/win-unpacked/` and `SmoothMyVideo-<version>-win.zip`, ~4 GB with TensorRT
  bundled). Recipients extract and run `SmoothMyVideo.exe`; nothing required on the target but the NVIDIA
  driver. (A zip, not an NSIS installer, `makensis` can't memory-map an archive this large.)
- `npm run lint`, one command that does everything: Prettier formats `src/**/*.ts` (writes), then ESLint
  lints `src`, then pyright lints `engine`. Stops at the first failure. See below.
- `engine\runtime\python.exe scripts\smoke.py [--full] [--trt]`, the render smoke tests: real engine runs
  on `samples/test.mp4` asserting frame counts, VFR duration preservation, `.part` promotion and (with
  `--full`, when their runtimes are installed) the HDR10 boxes, DV configuration record and HDR10+ SEI.
  Run it after every engine change; eager renders are not bit-deterministic, so the checks are
  structural, never checksums.

## Engine CLI (used by the GUI, also runnable directly)
```
engine\runtime\python.exe engine\gmfss_interp.py <input> <multi> [output] [--scale 1.0] [--fps TARGET] [--no-trt] [--sharpen S] [--restore] [--no-interp] [--no-passthrough] [--upscale F] [--codec hevc|av1|vvc] [--out-bits 8|10] [--rtx-vsr] [--rtx-hdr] [--dv] [--hdr10plus] [--hdr-nits N] [--hdr-color vivid|rtx|raw] [--hdr-vibrance B] [--hdr-satboost S] [--hdr-mastering-prim display-p3|dci-p3|bt2020|bt709]
```
- `<multi>` integer multiplier, or `--fps TARGET` to resample to any output fps (the model interpolates at
  arbitrary fractional timesteps; `<multi>` is required positionally but ignored when `--fps` is given).
- `--scale F` optical-flow resolution factor (GMFlow already runs at half the source; this scales it
  further). **Auto by default**: 1.0 below 4K, 0.5 for 4K+ sources. GMFlow's global attention grows
  super-linearly with area and dominates the interpolation wall, so quarter-resolution flow at UHD
  (still 1080p-class motion detail) makes a 4K render cost barely more than a 1080p one; verified
  equal-or-slightly-better against ground truth (dropped-frame reconstruction: mean tween PSNR 28.3 vs
  27.9 dB, same worst frame). Pass an explicit value to override.
- `--sharpen S` (0..1) FSR-style RCAS on every output frame (bare `--sharpen` = 0.8; off unless given).
- `--no-interp` re-encodes at source fps with sharpen only (no model/TRT loaded).
- `--restore` runs the Real-ESRGAN detail pass per output frame, before the upscale (works with `--no-interp`).
- `--upscale F` spatial upscale just before encode (bare = 1.5, clamp 16.0; above 8192 px auto-switches to
  a CPU AV1/VVC encoder). `--rtx-vsr` uses RTX Video Super Resolution, else bicubic.
- `--rtx-hdr` SDR→HDR10 (BT.2020 PQ) via TrueHDR; `--hdr-nits` mastering peak (400..2000, default 1000);
  `--hdr-color` {`vivid` (default: source hue+chroma), `rtx` (SDK saturation, hue-corrected), `raw` (debug)};
  `--hdr-mastering-prim` sets the `mdcv` gamut by name.
- `--dv` additionally exports a **Dolby Vision Profile 8.1** MP4 (needs `--rtx-hdr`, HEVC, MP4 out, and
  user-installed `dovi_tool` in `engine/dvtools`). See the Dolby Vision section below; GPAC-free.
- `--hdr10plus` additionally embeds **HDR10+** (SMPTE ST 2094-40) dynamic metadata (needs `--rtx-hdr`,
  HEVC, MP4 out, and user-installed `hdr10plus_tool` in `engine/hptools`); combinable with `--dv`. See
  the HDR10+ section below.
- `--out-bits` {`10` default, `8` legacy}; `--codec` {`hevc` default, `av1`, `vvc`}; `--no-passthrough` first-audio-only.
- Per-frame order: (restore →) upscale → RCAS sharpen → TrueHDR.

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

**Encode quality.** NVENC constant-quality VBR (AQ + a small chroma-QP boost), tuned and verified against a
lossless 8K master: HEVC **CQ 17** (VMAF 99.78 / 57.0 dB / SSIM 0.9986), AV1 **CQ 22**, VVC **QP 20**, all
past the visually-lossless bar on mean and worst frame. vvenc's perceptual QP adaptation is switched off
above 120 fps output (it inverts on wall-to-wall tween streams and bloats the file).

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
itself). Flow (`_dv_export` in gmfss_interp): during the HDR render `rtxvideo.run_hdr` accumulates **per-frame
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
can't reorder a float sum - this was the single nondeterministic stage, isolated by a per-stage
bit-exactness probe), and `cudnn.benchmark` is off with deterministic algorithm selection (benchmark
mode could pick different conv algorithms per process). Output differs from the old float-atomics
kernel by ~1e-4 max (82 dB, the old kernel's own run-to-run jitter level). Cost: the int64
accumulator doubles the splat's memory traffic - invisible at 2x, roughly +15% inference time at
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

## Constraints
- **CUDA 13 (Blackwell, sm_120):** torch is the cu130 build; cupy-cuda13x finds the runtime via
  `cuda-pathfinder`, so the old `_add_cuda_dll_dirs` nvrtc shim is no longer load-bearing.
- **RTX bridge:** keep the **cu12**-built `rtxvideo_cuda.dll` and ship `cudart64_12.dll` beside it, NGX's
  static import lib is CUDA-12-ABI, so a bridge relinked against CUDA 13 crashes in `create()`
  (see `engine/rtxvideo/build_src/BUILD.md`).
- **Runtime:** keep `engine/runtime` a relocatable python-build-standalone install, never a `venv`.
- **Renderer:** uses `require('electron')` with `nodeIntegration`, so it can't run in a plain browser,
  launch via `npm start`, the shortcut, or the vbs.

## Linting & formatting
All dev-only, all in `node_modules` (never shipped, `dist` bundles only `dist/`, `renderer/`, and the
`engine` extraResources, not `node_modules`). Deliberately a **light touch**: the engine Python and the
renderer's inline JS are intentionally dense (long lines, `x; y` one-liners, load-bearing comments), so
nothing reflows them, only `src/*.ts` is auto-formatted.
- **Prettier** (`.prettierrc.json`) formats `src/**/*.ts` only. `.prettierignore` guards `engine/`,
  `renderer/`, and build dirs so a stray `prettier .` can't reflow the hand-tuned files. Config matches the
  existing style (single quotes, semicolons, 2-space, printWidth 120).
- **ESLint** (`eslint.config.js`, flat config, `typescript-eslint` recommended) lints `src/**/*.ts` for real
  bugs, scoped to `src`, engine/renderer excluded. CommonJS config on purpose (no `"type":"module"`).
- **pyright** (`pyrightconfig.json`) lints `engine/*.py` as a **linter, not a type checker**:
  `typeCheckingMode: "off"` so the dynamic torch/numpy/cupy code isn't buried in type noise, only
  high-signal checks stay on (undefined names → error, unused imports/vars → warning). Vendored
  `runtime/`, `GMFSS_Fortuna/`, `trt_cache/`, and the RTX bridge are excluded from checking (`extraPaths`
  still resolves the GMFSS model + local engine modules). No Python-side install needed, pyright runs from
  npm. Ruff is a fine stronger alternative if you later install it (`uv tool install ruff`), but pyright
  keeps everything in the one `npm install`.

Note: `npm install <pkg>` rewrites `package.json` and re-expands its inline arrays (e.g. the `build.filter`
list) to one-per-line; a plain `npm install` / `npm ci` leaves formatting alone. Re-inline by hand if it
bothers you.

## Dev toolchain
The dev machine has VS 2019 Build Tools (MSVC `cl.exe` 19.29) + the Windows 10 SDK, enough to build the
RTX Video bridge, the NGX SDK's entry points live in a static import lib (`nvsdk_ngx_s.lib`), so ctypes
alone can't reach them (recipe in `engine/rtxvideo/build_src/BUILD.md`). Nothing that needs MSVC is
bundled; a recipient still needs only the NVIDIA driver.
