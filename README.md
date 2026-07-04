# SmoothMyVideo

Desktop app for offline AI video frame interpolation on NVIDIA GPUs. You pick a
video, choose how many frames to insert (2x to 16x), and it renders a smoother high
frame rate copy beside the original, with audio preserved. The UI is Electron plus
TypeScript; the interpolation runs in a Python GMFSS engine spawned as a subprocess.
Built and tested on an RTX 5090 Laptop (Blackwell, sm_120).

The model is GMFSS_Fortuna, a "union" interpolator: gmflow optical flow, an IFNet /
RIFE refiner, plus MetricNet, FeatureNet, FusionNet and softsplat warping. It produces
clean interpolated frames where the older NVIDIA optical flow path (FRUC) tore at high
multipliers.

## Status (2026-06-25)

Working end to end and verified on real clips. The packaged build is now fully self
contained: a recipient extracts the zip and runs `SmoothMyVideo.exe` with no Python, no
pip, and no ffmpeg installed (only the NVIDIA driver is assumed).

**Recent (2026-06-25):**
- **Production-grade HDR10 mastering.** RTX HDR writes HDR10 static metadata (mastering-display plus
  measured MaxCLL/MaxFALL), so one PQ file tone-maps on any display with no per-display nits input.
  TrueHDR Saturation defaults to a faithful **0** (its SDK "neutral" 100 over-saturates versus the
  source); Contrast/MiddleGray stay at the neutral 100/50. The per-display nits slider was removed in
  favour of a fixed 1000-nit master. See HDR mastering.
- **Faithful & vivid HDR colour (cyan/teal fixed).** The blue/teal cast in RTX HDR output was
  root-caused to the **TrueHDR model itself** rotating hues (it greens the blues even at Saturation 0);
  SMV's decode, interpolation and encode were each proven colour-faithful. The default `vivid` mode keeps
  TrueHDR's luminance (the HDR expansion) but rebuilds colour from the SDR source's hue AND chroma in
  ICtCp (BT.2100): faithful colour, cyan removed, and the SDK Saturation knob is inert (the model's
  chroma is dropped entirely; measured A/B, the model adds no real colourfulness at Saturation 0, it
  mostly rotates hue). Colour pop comes from the `rtx` mode and `--hdr-vibrance` instead. `raw` restores
  the unmodified model colour. See HDR mastering.
- **HDR10 mastering primaries default to Display P3.** The injected `mdcv` now carries P3 / D65 (the
  real grading gamut, and a faithful bound for SDR-sourced HDR), so a player reports real chromaticities
  like other HDR masters instead of collapsing to the nominal BT.2020 name. Selected by colorspace
  name, `--hdr-mastering-prim {display-p3,dci-p3,bt2020,bt709}` (display-p3 and dci-p3 share the P3
  gamut and differ only in white point: D65 vs DCI theatrical). The mastering black is declared 0
  cd/m² (perfect black), the reference an OLED-graded master carries. This is metadata only; the
  stream stays BT.2020 PQ.
- **CUDA 13 migration.** The bundled runtime moved to torch 2.12.1+cu130 + cupy-cuda13x + TensorRT
  cu13, validated end to end (eager, RTX HDR, TensorRT), and the distributable zip was rebuilt on cu13
  (`npm run dist`). See Setup.

**Recent (2026-06-28):**
- **Output codec selector (HEVC / AV1 / H.266).** A **Codec** dropdown picks the output family:
  HEVC (`hevc_nvenc`, the default), AV1 (`av1_nvenc`, hardware encode on RTX 40/50), or H.266/VVC
  (CPU `libvvenc`, the best compression of the three, slow, limited player support, always 10-bit).
  Hardware picks degrade gracefully (no NVENC session falls back to CPU `libsvtav1`; a missing
  `libvvenc` falls back to HEVC). RTX HDR works with all three, and the injected HDR10 static
  metadata is codec-agnostic (verified `mdcv`/`clli` inside an `av01` sample entry).
- **HDR sources handled properly.** Dropping an already-HDR file (PQ or HLG transfer) now: keeps
  interpolation/sharpening/upscaling working as before with the source HDR signalling carried
  through untouched; disables the RTX HDR toggle (TrueHDR is an SDR-to-HDR model, and the engine
  also guards the CLI flag, skipping the conversion with a notice); and tonemaps the preview pane
  for display, PQ shown raw used to read flat and washed out.
- **Single-instance app.** Launching SmoothMyVideo while it is already running no longer opens an
  empty second window. Both instances shared one Chromium profile, and the disk/GPU cache and Local
  Storage locks held by the first left the second renderer blank ("Unable to move the cache: Access
  is denied", "Gpu Cache Creation failed", captured from a live repro); a second instance would also
  fight the first over the preview PNGs and the TRT cache. A second launch now just focuses the
  running window (`app.requestSingleInstanceLock`).
- **Cleanup.** The clean-machine second-PC test was retired as an open item and the old cu12
  rollback runtime (`engine/runtime_cu12_bak`) was deleted; the cu13 stack has been validated end
  to end on the dev box across eager, TensorRT, RTX VSR/HDR, all three codecs and the packaged zip.

**Software stack updates still pending:**
- **Optional torch bump.** Stable torch 2.12.1+cu130 is the current pick; 2.13/2.14 nightly cu130
  exist but there is no reason to move yet.

- GUI: select a video (or drag one onto the window), view its info (resolution, source
  fps, duration, codec), choose a multiplier, type a target fps, or tick **match screen
  refresh rate** to target your monitor's Hz (rounded up), then click **Smooth It!**. An
  **FSR** toggle (FSR-style RCAS sharpening, on at full strength by default, with a
  strength slider) crisps the output. An **Upscale to** selector resizes the output to a chosen
  resolution (Off / Match screen / 1080p / 1440p / 4K / 8K / 16K / a custom height), keeping the source
  aspect ratio; the **RTX Video Super Resolution** toggle (opt-in, see NVIDIA RTX) does that
  upscale with NVIDIA AI, otherwise it is a bicubic resize. The **Interpolate** toggle (on by
  default) is the master switch for frame generation: untick it to *only* sharpen / upscale the
  video, keeping the source frame rate (the multiplier / fps / match-screen controls grey out and
  the engine skips the GMFSS model entirely). A **Codec** selector picks the output codec: HEVC
  (default, GPU), AV1 (GPU) or H.266/VVC (CPU). An opt-in **NVIDIA RTX** panel adds real RTX Video
  Super Resolution and **RTX HDR** (SDR to HDR10) with App-style **Contrast** and **Saturation**
  sliders, plus an **RTX Dynamic Vibrance** filter (**Saturation boost** / **Intensity**); all off
  by default and unlocked once the RTX Video runtime is installed (a one-click in-app installer,
  see NVIDIA RTX). A **Preview** pane above Smooth It shows original versus processed on real
  frames (random-frame stepper, click an image for 1:1 pixels) and follows every spatial setting
  live. **Cancel** kills the running job; **Open folder** reveals the result. The last used folder,
  multiplier, sharpen, codec, upscale resolution, interpolate, match-screen, RTX and HDR settings
  are remembered between sessions, and are wiped once automatically when a new build changes their
  storage format (a stored settings-schema version guards against stale-key conflicts).
- Progress: a bar that starts at the source frame count and fills to the post process
  total, plus a live frame counter and an ETA.
- Output: written beside the source as `<name>_<fps>fps.mp4` (or a custom path chosen
  with **Change...**). Always visually lossless, no quality knob. The output is always HEVC
  (`hevc_nvenc`) when the device has a usable NVENC session, since HEVC at the same quality is
  far smaller than H.264 and an interpolated clip is a new artifact (matching an H.264 source
  would only bloat it). The output is 10 bit (`p010le` / `main10`) whatever the source depth, so
  the float-precision interpolated frames never get re-quantised to 8 bit levels (which is what
  bands gradients like skies and glows); the source colour signalling is carried through, and
  every audio track, subtitle track (translations), chapter and font attachment is copied too
  (the output becomes .mkv automatically when the tracks need it; see Passthrough quality). With
  no usable NVENC it falls back automatically to CPU SVT-AV1, still visually lossless (see
  Passthrough quality).
- Batch: pick or drop several files at once and they queue, rendering back to back with the
  same settings and default output names (status shows `File k/N`; Cancel clears the queue).
  After a render, **Play video** opens the result in the system default player and
  **Open folder** reveals it.
- Uniform look (no popping): every output frame is interpolated, the first and last included.
  No source frame is passed through and none lands on a source timestamp, because the sample
  grid is shifted half a step so each frame is an interior blend. This removes the sharp original
  then soft tween alternation that otherwise makes fine detail pop in and out on every Nth frame.
  The whole clip is uniformly a touch softer than the source in exchange for that consistency.
  See Uniform look below.
- Engine: GMFSS at fp16 with a cupy softsplat kernel, about 2.2x faster than the original
  fp32 path. The TensorRT backend is the default when available (about another 2.2x on top,
  built and cached per resolution on first run) and falls back automatically to the eager
  pipeline when TensorRT is unavailable. See Performance below.
- Quality and throughput refinements: generated frames are kept sharp (the model's multiple
  of 64 size requirement is met by padding, not resizing, and the output is rounded to nearest
  rather than truncated), byte exact duplicate source frames are held instead of interpolated
  (anime is drawn on twos), and the ffmpeg decode and encode run on background reader and
  writer threads so pipe I/O overlaps GPU work.
- Bundled: a relocatable Python 3.14 runtime (torch cu130 / CUDA 13 + cupy) at `engine/runtime`,
  and a shared-build ffmpeg (NVENC plus a CPU SVT-AV1 fallback) at `engine/bin`. Both ship inside
  the zip; the app uses neither system Python nor system ffmpeg.
- Launch: Desktop and Start menu shortcuts, a no console `SmoothMyVideo.vbs`, and a
  custom icon.
- A sample 24fps clip ships in `samples/test.mp4` for quick testing.

## Run it
- Double click the **SmoothMyVideo** shortcut (Desktop / Start menu), or `SmoothMyVideo.vbs`.
- Or from a terminal: `npm start` (builds, then launches).
- Select a video (or drag one onto the window), choose a multiplier (2x / 4x / 8x / 16x)
  or type a target fps, then click **Smooth It!**. The result is written next to the
  source as `<name>_<fps>fps.mp4`, or wherever you point it with **Change...**.

## Architecture
- `src/main.ts` - Electron main: window, open and save dialogs, ffprobe (`-of json`),
  spawns the engine, streams progress, tracks the running child so **Cancel** can
  `taskkill /T /F` it. A `refresh-rate` IPC returns the rounded-up refresh rate of the
  monitor the window is on (`screen.getDisplayMatching`) for the match-screen option.
  Resolves the interpreter as `engine/runtime/python.exe`
  (`RUNTIME_PY`, falls back to `python` on PATH) and ffprobe as `engine/bin/ffprobe.exe`
  (`FFPROBE`, falls back to `ffprobe` on PATH). It always sets `PYTHONUTF8` plus a writable
  `SMV_TRT_CACHE` (under userData) for the engine cache, since the engine runs the TensorRT
  backend by default.
- `renderer/index.html` - the UI (select or drag in a video, an **Interpolate** master
  toggle, multiplier / fps / **match screen refresh rate**, an **FSR** sharpen toggle with
  strength slider, a **Restore** AI-detail toggle (see Sharper generated frames), output path
  with **Change...**, progress bar with frame counter and ETA,
  Cancel, Open folder, Play video, log). Uses `require('electron')`; dropped files are resolved
  to paths with `webUtils.getPathForFile` (several picked or dropped files queue as a batch and
  render back to back), and the folder, multiplier, sharpen, interpolate and
  match-screen settings are saved in `localStorage` (the Restore and RTX Dynamic Vibrance
  toggles deliberately are NOT: both are per-session opt-ins that always start off).
- `engine/gmfss_interp.py` - GMFSS pipe engine: ffmpeg decode into GMFSS into ffmpeg
  encode (audio copied), always encoding HEVC at 10 bit with the colour tags matched to the
  probed source and always targeting visually lossless (see Passthrough quality). Runs the
  TensorRT backend by default (per-subnet eager fallback; `--no-trt` forces eager) and NVENC
  with an automatic CPU SVT-AV1 fallback. Always fp16; takes an integer `<multi>` or
  `--fps TARGET` for an arbitrary resampled output fps. Every emitted frame is a GMFSS render on a
  grid offset by half a step, so no source frame is passed through and none lands on a source
  timestamp and the whole clip shares one look (see Uniform look). To keep generated frames sharp it pads
  each frame to the multiple of 64 the model needs (rather than resizing, which resamples every
  pixel) and rounds the output to nearest instead of truncating; it also holds byte exact
  duplicate source frames instead of interpolating them, and runs the ffmpeg decode and encode
  on background reader and writer threads so pipe I/O overlaps GPU work. Prints `PROGRESS k/total` to
  stderr. Resolves `ffmpeg`/`ffprobe` from `engine/bin` first and falls back to PATH
  (`_tool()`). `_add_cuda_dll_dirs()` puts the nvidia wheel bin dirs on the Windows DLL
  search before the model imports so cupy can JIT its kernel.
- `engine/trt_runtime.py` - optional TensorRT backend. `trtify(model)` swaps the five GMFSS
  sub nets for engines exported under autocast and built strongly typed (fp16); softsplat
  and the interpolate glue stay in eager. Engines are cached per `(net, shapes, gpu, trt
  version, weights hash)` under `SMV_TRT_CACHE`, built on first use, with eager fallback on any
  failure. The weights hash (an md5 of the `train_log` .pkl contents, in every engine filename
  since 2026-07-02) makes the cache self-invalidating: swapping the model weights is a cache
  miss plus automatic deletion of the stale engines at the next engine start, instead of
  silently serving engines compiled from the old model (`build_or_load` finds engines by name
  and never re-reads the .pkl files). Pre-fingerprint caches were adopted by rename, not
  rebuilt, since only one weight set has ever shipped.
- `engine/runtime/` - bundled relocatable Python (python-build-standalone CPython 3.14)
  with the full GPU stack installed (torch cu130 / CUDA 13, cupy-cuda13x, nvidia cu13 wheels). Gitignored, see Setup.
- `engine/bin/` - bundled shared-build `ffmpeg.exe` + `ffprobe.exe` (built with `hevc_nvenc`)
  plus the FFmpeg DLLs both exes load (one copy of `avcodec` and friends instead of two
  static embeddings) and their license. Gitignored, see Setup.
- `engine/GMFSS_Fortuna/` - GMFSS model code (inference chain only) plus `train_log/`
  weights (gitignored, see Setup).
- `engine/realesr.py` - the `--restore` detail-restoration pass: Real-ESRGAN's SRVGGNetCompact
  (vendored verbatim, BSD-3, license alongside) plus a loader for the bundled
  `realesr-animevideov3.pth` weights (2.4 MB, gitignored, see Setup).
- `engine/benchmark.py` - speed benchmark; appends a dated entry to `BENCHMARKS.md`.

## Setup (fresh clone)
A fresh clone is missing four gitignored pieces: `engine/runtime`, `engine/bin`,
`engine/GMFSS_Fortuna/train_log` and `engine/realesr-animevideov3.pth`. The app needs them
to run, and `npm run dist` needs them present in order to bundle them.

**1. GUI deps**
```
npm install
```
If the Electron binary did not download (its postinstall is sometimes skipped):
```
node node_modules/electron/install.js
```

**2. Python runtime into `engine/runtime`**
The bundled interpreter is a relocatable
[python-build-standalone](https://github.com/astral-sh/python-build-standalone/releases)
CPython 3.14 (the `install_only` win64 build). Download it, unpack the tarball, and move
its inner `python/` folder to `engine/runtime`. Then install the Blackwell **CUDA 13** GPU
stack into it (this is the exact environment that gets bundled):
```
engine\runtime\python.exe -m pip install torch==2.12.1 torchvision --index-url https://download.pytorch.org/whl/cu130
engine\runtime\python.exe -m pip install -r engine\requirements.txt
```
`requirements.txt` includes `cupy-cuda13x` plus the `nvidia-cuda-nvrtc` and
`nvidia-cuda-runtime` wheels (the CUDA 13 wheels dropped the `-cuXX` suffix; the
`nvidia-cuda-*-cu13` packages are deprecated placeholders that fail to build, so use the
unsuffixed names). cupy JITs its softsplat kernel and uses `cuda-pathfinder` to locate the
CUDA 13 runtime, so the old `nvrtc-builtins` DLL-search shim is no longer load-bearing. It also
pulls `tensorrt` (the cu13 build, ~2 GB of CUDA 13.3 libraries, matching torch's cu130) plus
`onnx`/`onnxscript` for the default TensorRT backend. A standard `python -m venv` is **not** usable here: a
Windows venv keeps its standard library in the base Python install, so it is not
relocatable and breaks on a machine that lacks that exact Python. python-build-standalone
is self contained, which is what makes the bundle portable.

**3. ffmpeg into `engine/bin`**
Download a shared Windows ffmpeg that includes `hevc_nvenc` (e.g.
[BtbN FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds/releases),
`ffmpeg-master-latest-win64-lgpl-shared.zip`) and copy `bin\ffmpeg.exe`, `bin\ffprobe.exe`
and all the `bin\*.dll` files into `engine\bin` (skip `ffplay.exe`). A static build's two
exes also work but nearly triple the size. The app prefers these and only falls back to
ffmpeg/ffprobe on PATH, so for local dev you can skip this if you already have ffmpeg
installed; it must be present for a portable `npm run dist`.

**4. GMFSS weights into `engine/GMFSS_Fortuna/train_log`**
The weights (feat, flownet, fusionnet, metric, rife pkl files) are gitignored because
they are large. Restore them from the original GMFSS_Fortuna release.

**5. Restore weights into `engine/`**
Download `realesr-animevideov3.pth` (2.4 MB) from the Real-ESRGAN v0.2.5.0 GitHub release
(https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth)
into `engine\`. Without it the `--restore` pass just logs a notice and is skipped.

## Dev toolchain
The development machine has Visual Studio 2019 Build Tools v142 (MSVC `cl.exe` 19.29) and the
Windows 10 SDK (10.0.19041) installed, so native code that links the NVIDIA SDK import
libraries can be built locally. This is what enables the RTX Video bridge: the SDK is NGX based
and its entry points live in a static import lib `nvsdk_ngx_s.lib`, so they cannot be reached by
ctypes alone and need a compile (see `engine/rtxvideo/build_src/BUILD.md` for the recipe). The
DX11 and DX12 paths build with only MSVC and the Windows SDK; the CUDA and
Vulkan sample paths would each need the CUDA Toolkit or the Vulkan SDK first, neither of which
is installed. This does not change the shipping promise: nothing that needs MSVC is bundled,
and a recipient still needs only the NVIDIA driver.

## Scripts
- `npm start` - build (`tsc`) and launch.
- `npm run starti` - wipe `node_modules` and `package-lock.json`, fresh install, then start.
- `npm run dist` - the single build command. It wipes `release/` (inlined `fs.rmSync`), compiles with
  `tsc`, then runs `electron-builder`, whose zip target produces **both** `release/win-unpacked/` and
  `release/SmoothMyVideo-<version>-win.zip` (about 4 GB with the TensorRT backend bundled). Recipients
  extract the zip and run `SmoothMyVideo.exe`; no install step, and nothing is required on the target
  machine but the NVIDIA driver. `release/` is wiped first, so no stale artifacts are ever bundled.

## Performance
Precision is **fp16 only** (fp32 was removed: fp16 is visually lossless versus fp32,
PSNR about 51 dB, and just slower). The fast path is **fp16 plus cupy softsplat**.

Core inference at 1080p (warmup plus cuda.synchronize, excludes model load and ffmpeg I/O):
- original fp32: about 357 ms per frame
- fp16, pure torch softsplat: about 276 ms per frame
- fp16 plus cupy softsplat: about 160 ms per frame, roughly 2.2x over the fp32 baseline

A 360 frame clip at 16x lands around 17 minutes. Measure with `engine/benchmark.py`,
which logs dated entries to `BENCHMARKS.md` so progress is tracked over time.

**TensorRT (default; `--no-trt` forces eager).** Each of the five sub networks
(FeatureNet, GMFlow, MetricNet, IFNet, FusionNet) is exported to ONNX under autocast and
built into a strongly typed fp16 TensorRT engine; softsplat (cupy) and the interpolate
glue stay in eager. Measured on the RTX 5090: GMFlow (the heavy net, run twice per pair)
2.33x, FeatureNet 1.60x, and about **2.2x end to end** over the cupy fp16 path, numerically
matching (interpolated frame mean diff about 1%). Engines build on first use for each input
resolution (a few minutes) and are cached, so later runs at that resolution start instantly.
They are specific to the GPU, the resolution, and the TensorRT version, and rebuild on a
different machine. Any engine that fails to build falls back to eager, so the app never breaks.

A code-side performance audit (2026-06-26) probed or attempted six further optimizations and found the
pipeline is **at its practical compute limit** on this hardware and model: I/O and host-sync changes are
perf-neutral (it is GPU-compute-bound, not launch or sync bound), batching the per-timestep nets does not
help (FusionNet is already saturated at batch 1), and fp8 fails a quality gate (precision, 61px flow
outliers). Two non-regressing cleanups were kept: GPU-side transposes, and the whole inference now runs on
one CUDA stream with no per-call TensorRT sync (which also keeps the TRT default-stream warning fixed). See
**Performance headroom** under What can be done next for the per-item results.

`torch.compile` was tried and dropped. Its inductor backend needs Triton plus MSVC;
Triton is a pip wheel that would bundle fine, but the path is deliberately not bundled,
for two reasons. First, inductor compiles just in time, on the first call on whatever
machine runs it, so MSVC would have to ship and work on every recipient's PC, breaking the
"only the NVIDIA driver is assumed" promise (MSVC is also not cleanly redistributable and
would add multiple GB to an already large zip). Second, it would only duplicate the
TensorRT backend, which already fuses these same nets at about 2.2x and serializes to a
portable cached engine, so there is no clear win for this model class. Compiling from
Dynamo/FX would sidestep the grid_sample ONNX issue below and avoid per resolution builds,
but neither is worth shipping a compiler to every machine.

**Dynamic shape** TensorRT engines (one engine covering a range of resolutions via a
`min`/`opt`/`max` profile) were also tried. They fail because GMFlow and IFNet warp with
`grid_sample`, which the dynamo ONNX exporter routes through `cudnn_grid_sampler` with no
ONNX translation on the dynamic path (the static export handles it fine). Dynamic engines
are also documented as somewhat slower with more VRAM, so static per resolution engines,
built on first use and cached, are the right call.

## Benchmark vs other GMFSS engines (2026-06-19)
Compared SmoothMyVideo against the three repos it draws from, to find per frame speedups
that keep quality. All numbers are one session, one unified harness, GMFSS compute only at
1080p fp16 on the RTX 5090 (warmup plus cuda.synchronize, excludes decode and encode):
`reuse` is flow / feat / metric done once per source pair, `inference` is one interpolated
frame, `pair@2x` is reuse plus one inference.

| Engine | inference ms/frame | reuse ms | pair@2x ms | VRAM |
| --- | --- | --- | --- | --- |
| SMV (Fortuna union, fp16 eager) | 177 | 580 | 757 | 8.3 GB |
| GMFSS_Fortuna (union, fp16) | 182 | 588 | 770 | 8.3 GB |
| GMFSS_Fortuna (base, fp16) | 168 | 588 | 756 | 8.2 GB |
| GMFSS_union (old arch, fp16) | 147 | 566 | 713 | 5.0 GB |
| SMV (Fortuna union, TensorRT) | 101 | 270 | 370 | 1.3 GB |

Recommendations, with the repo each came from:

- **TensorRT is the per frame win, and it is the default backend in SMV.** This run measured
  about 1.75x per interpolated frame and 2.05x per pair over fp16 eager, plus a new finding:
  VRAM drops from 8.3 to 1.3 GB (about 6.5x), same union weights so no quality change.
  Source: SMV `engine/trt_runtime.py`; the idea of running GMFSS on TensorRT comes from
  **enhancr**, whose GMFSS path lowers the model with `torch_tensorrt`.
- **Attack GMFlow next for 2x.** At 2x, reuse is most of the TensorRT time (about 270 of
  370 ms) and reuse runs GMFlow twice (forward and backward flow), so GMFlow is the hotspot.
  Source: observed here; matches the note above that GMFlow is the heavy net.
- **Try a single combined engine, channels_last, and multi stream.** enhancr lowers the whole
  GMFSS module to one `torch_tensorrt` engine, runs it in `channels_last`, and overlaps several
  frames across CUDA streams (`num_streams`). SMV instead builds five separate subnet engines
  and keeps softsplat eager (the cupy softsplat blocks a single ONNX export). Two cheap
  experiments fall out: run the TensorRT engine I/O and eager glue in `channels_last`, and give
  `TRTModule` a dedicated non default CUDA stream (TensorRT warns that the default stream forces
  an extra sync every call). This is separate from the overlapped decode item below, which is
  about ffmpeg I/O, not GPU parallelism. Source: **enhancr** `arch/vsgmfss_fortuna/__init__.py`.
- **Lighter models are faster but cost quality (so out under no quality loss).** GMFSS base
  drops the RIFE branch for about 8 percent faster inference (source: **GMFSS_Fortuna**,
  `GMFSS_infer_b`), and the older **GMFSS_union** arch is about 17 percent faster at half the
  VRAM. The union Fortuna model SMV ships is the quality pick; these are the speed floor for
  reference.
- **Do not chase an enhancr install.** enhancr is no longer distributable (its payload CDN is
  dead; only a CPU torch lite build with no TensorRT survives on the Internet Archive) and
  enhancr 0.9.9 is torch 2.1 plus TensorRT 8.6.1, which predate Blackwell `sm_120` and will not
  run on this GPU. Its one speed advantage, TensorRT, is already covered by SMV's modern stack
  (torch 2.12 `cu130`, TensorRT 11.1).

## Uniform look (no detail popping)
A naive interpolator keeps the original frames and inserts generated tweens between them, so the
output alternates byte exact source frames (sharp, full real detail) with softer model frames.
At the interpolation rate that reads as fine detail popping in and out (sharp original, then soft
tween, then sharp original), which breaks immersion. SmoothMyVideo instead generates *every*
output frame: it samples on a grid shifted by half an output step, so no frame coincides with a
source timestamp and every frame is an interior blend. For an integer `multi` M the timesteps
inside each source pair are `1/2M, 3/2M, ... (2M-1)/2M` (symmetric around 0.5, spacing `1/M`);
`--fps` mode uses the same half step offset. The first and last output frames are generated too.
This is the "generate every displayed frame, never pass a real one through" idea (as requested,
modelled on Lossless Scaling's fully generated output).

A tempting cheaper fix, keeping the original timing but regenerating the frames that sit on a
source timestamp through GMFSS at timestep 0, does **not** work: at `t=0` the model reconstructs
the frame about as sharply as the original, so the pop survives. Measured on the sample clip
(sharpness = variance of the Laplacian, mean over frames):

| Output | on grid vs tween sharpness | even/odd ratio | mean (source 42.5) |
| --- | --- | --- | --- |
| regenerate on grid at `t=0` | 43.1 vs 37.1 | 1.16 (visible pop) | 40.1 |
| half step grid (shipped) | 38.1 vs 36.5 | 1.04 (uniform) | 37.3 |

The cost: the true pixels that sat on the source grid are dropped, so the clip is uniformly a
little softer than a passthrough render, and that uniformity is the goal. The output frame count
is `multi*frames` (true doubling for 2x, and so on): every source frame gets `multi` output
frames, and the last source frame's own time slot, which has no frame after it to interpolate
toward, is filled by holding the last generated frame. This matches the count you intuitively
expect, the source duration, and the GUI's own frame total, and it is the same target fps
approach Topaz uses. Held cels (byte exact duplicate source frames) are still detected and
rendered once, so a static shot stays as crisp as its source without per frame shimmer.

## Passthrough quality
The encode keeps the source's chroma and colour and never drops below its bit depth, so the
deliberate changes are interpolation, the codec (always HEVC) and the default 10-bit output.
From the ffprobe of the input the engine sets:

- **Codec.** Always HEVC (`hevc_nvenc`), whatever the source codec was. HEVC at the same visually
  lossless CQ is far smaller than H.264, and the interpolated clip is a brand new artifact, so
  echoing the source codec would only bloat it: a 3 Mbps H.264 source produced a 100 Mbps H.264
  output under the old match the source rule, and HEVC brings that to about a quarter of the size
  at the same quality. Other interpolation tools agree (enhancr, Topaz and Flowframes all offer a
  codec menu centred on HEVC and AV1; GMFSS_Fortuna's own script only dumps `mp4v`). The encoder
  is preflighted on a tiny frame; if the device has no usable HEVC NVENC session (no NVIDIA GPU, a
  GPU too old, or no driver) the engine falls back to CPU `libsvtav1` for the encode.
- **Bit depth.** Decode follows the source (8 bit via `rgb24` byte for byte; 10 bit and up via
  `rgb48le` so nothing truncates before the model), but the encode is **10 bit by default for
  every source** (`p010le` / HEVC `main10`, `yuv444p16le` + `rext` for 4:4:4 sources, 10-bit AV1;
  `--out-bits 8` restores the legacy 8-bit output for maximum device compatibility, honoured for
  8-bit sources only). Every emitted frame is a floating point blend (the pipe is fp16, which
  holds 10 bit precision), so quantising back to 8 bit would throw away real sub-8-bit precision
  the interpolation just created and band smooth gradients (skies, glows); measured on a dark
  gradient clip, the 8-bit render keeps the source's 70 distinct luma levels while the 10-bit
  render carries 281. GMFSS_Fortuna and GMFSS_union are 8 bit only, and enhancr runs GMFSS
  inference in float but writes 8 bit (`YUV422P8`), so writing the float precision into a 10-bit
  file is one step past all three references.
- **Colour.** The source matrix, transfer, primaries and range are stamped on with a
  `setparams` filter (NVENC drops transfer and primaries from the bare `-color_*` output
  flags), so bt2020 / PQ / HLG HDR signalling survives the round trip.
- **Quality.** Always visually lossless, no knob. On NVENC: constant quality VBR with AQ on and
  a small chroma QP boost; the values were **verified and tuned against a lossless 8K master
  (2026-07-03)** with frame-aligned VMAF/PSNR/SSIM. HEVC **CQ 17** is the quality reference:
  VMAF 99.78 / 57.0 dB / SSIM 0.9986 (worst frame 55.1 dB). AV1's CQ scale is not HEVC's - the
  old CQ 20 measured *higher* fidelity than the reference at larger size, so AV1 now runs
  **CQ 22**: still above the reference on every metric (VMAF 99.84 / 58.2 dB / SSIM 0.9988) at
  ~13% smaller files (CQ 23 dips just below the reference, so 22 is the sweet spot). VVC had
  the thinnest margin, so it moved from QP 21 to **QP 20**: VMAF 99.82 / 51.3 dB / SSIM 0.9966
  at ~5% more bits, still under a third of the HEVC size. All three sit far past the ~95 VMAF
  transparency bar and the SSIM >= 0.995 visually lossless target on both mean and worst frame.
  One conditional: vvenc's perceptual QP adaptation (QPA) INVERTS on high-fps interpolated
  streams - it saves ~27% on normal content but bloated a 360fps render enough to make VVC
  *larger* than HEVC at 8K - so the engine switches QPA off above 120 fps output (measured 46%
  smaller there, both variants above the standards; a stderr note says when). On the
  `libsvtav1` fallback: CRF 20 at preset 8, SVT-AV1's visually lossless range. Audio is
  always copied.
- **Tracks (translations passthrough, 2026-07-03).** Every audio track, subtitle track, the
  chapter list and font attachments are copied into the output (the old behaviour - first audio
  only, everything else dropped - is `--no-passthrough`). Container rule: mp4 cannot hold styled
  ASS/PGS subtitles or some audio codecs, so the default output name switches to **.mkv**
  whenever the source has subtitles or mp4-incompatible audio (anything outside
  aac/ac3/eac3/mp3/alac/opus/flac); an explicitly chosen `.mp4` path keeps mp4 and drops the
  incompatible tracks with a notice. `mov_text` subtitles (mp4-native) are converted to SRT for
  mkv since they cannot be stream-copied. Verified on a worst-case fixture (jpn+eng audio,
  styled ASS sub, font attachment, chapters): all tracks and language tags survive into the
  interpolated .mkv; plain single-audio mp4 sources behave exactly as before. HDR renders that
  land in .mkv keep the full HDR10 static metadata too, via a two-stage finalize: the video is
  encoded to a temp .mp4, the `mdcv`/`clli` boxes are injected there (the injector is
  ISOBMFF-only), and a stream-copy remux merges it with the source's tracks into the final .mkv -
  ffmpeg maps the boxes onto Matroska's NATIVE `MasteringMetadata`/`MaxCLL` elements (verified
  value-exact: MaxCLL/MaxFALL, P3 primaries and the 1000-nit peak all survive). For comparison,
  Topaz Video AI drops subtitles entirely; Flowframes copies audio+subs like this does.

The bundled ffmpeg is the BtbN **lgpl** build, which has no `libx264`/`libx265`, so the
software fallback uses `libsvtav1` (SVT-AV1: true CRF, clean 8 and 10 bit) rather than an
x264/x265 `-crf` path. True lossless of the
*source* is not possible for an interpolator, and here every output frame is generated rather
than copied from the source (see Uniform look), so "passthrough" refers only to the codec
family, bit depth and colour signalling. It means adding no visible encode loss and not
downgrading the source format, not preserving the original pixels frame for frame.

## HDR mastering (production-grade HDR10)
The **RTX HDR** path (`--rtx-hdr`) produces a real HDR10 master, not just a PQ-tagged file. The point
to understand: HDR does not make the file look different per screen. The picture is graded once
against an absolute reference (PQ encodes absolute nits, not relative brightness), and each display
tone-maps that reference down to its own peak. Metadata is what makes that adaptation accurate. The
engine does the four things a production HDR10 deliverable is made of:

1. **Encode PQ / BT.2020, 10-bit.** TrueHDR outputs BT.2020 PQ (SMPTE 2084); the encode is `main10` /
   `p010le` with the colour signalling stamped via `setparams` (NVENC drops the bare `-color_*`
   transfer / primaries otherwise).
2. **Keep the source colour (hue and chroma locked).** The default `vivid` mode rebuilds colour in
   ICtCp (BT.2100): it keeps TrueHDR's luminance (the HDR expansion) but takes the **hue and chroma**
   from the colorimetric SDR source (`RTXVideo._ictcp_correct`), so the picture keeps exactly the
   source's colours at HDR brightness. Why not the SDK's own knob: even at Saturation 0 the model
   rotates hues (greens the blues, so skies and snow read cyan/teal) *and* adds no real colourfulness,
   so its chroma is dropped entirely - which also makes the SDK `--hdr-saturation` inert in `vivid`.
   Pop, when wanted, comes from `--hdr-color rtx` (NVIDIA's saturation, hue-corrected) and
   `--hdr-vibrance`. `raw` emits TrueHDR's colour unmodified. Accurate hue at full HDR brightness, on
   by default.
3. **Master to a fixed peak, set once.** `--hdr-nits` (default 1000) is the mastering peak the PQ
   values are shaped to, not a per-monitor knob. 1000 is the consumer standard; 4000 is premium.
4. **Write the mastering metadata.** `mdcv` (mastering-display: Display P3 / D65 by default, the
   1000-nit peak and a 0 cd/m² perfect-black floor; `--hdr-mastering-prim {display-p3,dci-p3,bt2020,bt709}`
   selects the gamut by colorspace name, so a player shows real chromaticities like other HDR
   masters) and `clli` (MaxCLL / MaxFALL measured from the frames) are injected into the mp4 by
   `engine/hdr10_meta.py`. A
   400-nit panel reads these and rolls the highlights down; a 1000-nit panel shows it near-native. One
   file, no per-display input.

That is the same mechanism a Dolby Vision or HDR10 master uses to look right on a 4000-nit reference
monitor, a 1000-nit OLED and a 400-nit TV. The tiers beyond HDR10, for whoever wants to go further:
- **HDR10+** (royalty-free to author): adds *dynamic*, per-scene metadata for better low-nit roll-off.
  Author it with `hdr10plus_tool` and feed x265 via `dhdr10-info`; needs an x265 built with HDR10+
  (the bundled LGPL ffmpeg has neither libx265 nor HDR10+).
- **Dolby Vision** (licensed): per-shot colorist trims for specific target nits, with HDR10 fallback
  (profile 8.1). Creation and distribution need a Dolby licence and the content-mapping analysis is
  gated behind DaVinci Resolve or Dolby's tools; `dovi_tool` only extracts / edits / muxes the RPU.
  Out of scope for a self-contained, offline app.

## What can be done next
For whoever picks this up.

- **SDR to HDR colour / cyan (root-caused and fixed).** The blue/teal cast was the **TrueHDR model**
  rotating hues, not a pipeline bug. Measured tonemap-free in linear BT.2020, blue-dominant pixels gain
  about +0.048 green chromaticity (roughly 18% relative) even at Saturation 0, while SMV's decode, GMFSS
  interpolation and encode each round-trip colour-faithful: a colorimetric `zscale` SDR to BT.2020-PQ
  conversion of the same source matched it exactly, an encode-path patch test stayed neutral, and the
  shift survived a no-interpolation render, isolating it to the model. TrueHDR exposes no hue / white
  balance knob, so the fix keeps its per-pixel luminance and **transplants the source chromaticity** in
  ICtCp (BT.2100, `RTXVideo._ictcp_correct`): about 78% of the cast removed (+0.048 to +0.011), the
  residual being 4:2:0 chroma subsampling on the p010 encode. This is the default `vivid` mode (the
  model adds no real chroma at Saturation 0, so nothing of value is lost by dropping the SDK Saturation
  knob along with the model's hue-rotated chroma). `raw` keeps the old model colour. Note: an earlier "viewing transform, not the pipeline" reading was
  **wrong** - it judged the neutral white point normalized by green, which hides a green shift; the
  shift is hue-dependent and shows in saturated blues, which is why the mountains looked teal.

- **Clean machine test (waived 2026-06-28).** The packaged zip is self contained and was
  verified locally with system Python and ffmpeg stripped from PATH; a run on a *separate*
  PC was waived as a requirement, so no portability items remain open. If it is ever wanted
  anyway: copy `release/SmoothMyVideo-<version>-win.zip` to a machine with no Python and no
  ffmpeg (just an NVIDIA driver), extract, run `SmoothMyVideo.exe`, and test both a normal
  render and one with the **TensorRT** toggle on (the first TRT render at a given resolution
  spends a few minutes building engines, then caches them).
- **Scene change detection (done 2026-07-02; OPT-IN via `--scene-detect` since 2026-07-04).**
  With the flag, the engine does not morph one shot into the
  next at a hard cut: cut pairs are detected and the boundary frames are held instead (slots
  before the mid point hold shot A, the rest hold shot B, so the cut lands sharp between two
  output frames). The held frames are rendered the way held duplicate cels already are (GMFSS
  on the `(I, I)` pair at t=0.5), keeping the clip's uniform soft look instead of popping a
  sharp source frame into a run of tweens. Detection reuses the flows `model.reuse()` already
  computes, per the plan: not a raw pixel difference (a fast pan would false-positive; enhancr's
  VapourSynth `misc.SCDetect` has exactly that flaw) but two flow checks that must BOTH fire - the
  forward/backward consistency failure fraction (`occ` > 0.5) and the bidirectional warp
  reconstruction error (`photo` > 0.08). Measured: within-shot anime pairs sit at occ
  0.042..0.063 / photo 0.034..0.047, a 100 px/frame whip pan at occ 0.000 (GMFlow tracks it,
  confirming the design premise), and a true cut saturates at occ 1.000 / photo 0.260 - roughly
  8x margin against false positives. A same-shot crop-zoom reframe (occ 0.196) deliberately
  still interpolates: the flow matches it, so the tween is a coherent zoom, not a smear. At the
  boundary the ghost is measurably gone (output frames match their own shot at MAD ~0.006 vs
  0.10..0.21 blends before). Cuts are logged to stderr and counted in the done line;
  `SMV_SCENE_DEBUG=1` prints both metrics per pair. **Why it became opt-in (2026-07-04):**
  field data from a real action clip flagged 8 "cuts" in ~70 pairs at occ 0.51-0.84 /
  photo 0.09-0.35 - inside the gap between the within-shot and true-cut calibration bands,
  with the consecutive-pair signature of 1-frame flashes and impact frames rather than real
  cuts. Every false hold pauses the smoothing for a source-frame interval (a visible stutter
  mid-action), which defeats the app's purpose, while a missed real cut merely morphs for one
  source-frame interval - brief at high output fps. So the default is now to always
  interpolate; pass `--scene-detect` for cut-heavy content where held boundaries matter more.
- **Uniform interpolation, no duplicate holding or dejudder (2026-07-04).** Earlier builds
  detected repeated cels (byte exact, plus near duplicates that differ only by compression
  noise) and held them, and retimed short duplicate runs into even motion (dejudder). Both are
  removed. Every source pair is now interpolated the same way on the `_pair_fracs` slot grid, so
  the gap between any two consecutive source frames, whether real motion, a held cel or an exact
  repeat, gets the same even smoothing, and the source's own frame timings are preserved by
  construction (frame count, duration and A/V sync unchanged). The reason: on grainy or textured
  stills the per block duplicate metric straddled its threshold. Film grain on high contrast art
  measured right around the old 0.012 bound, so a genuinely held card fragmented unpredictably,
  some frames held, some retimed, some interpolated, instead of smoothing cleanly. Interpolating
  every pair removes that inconsistency and maximises smoothness, at the cost of the compute the
  duplicate hold used to save (roughly 2x more inferences on twos cadence content). Hard cuts are
  interpolated by default; `--scene-detect` still holds their boundaries (see the scene change
  bullet).
- **Sharper generated frames (free half done).** The free half is fixed: `to_tensor` and
  `to_bytes` now pad to the next multiple of 64 the model needs and crop the padding back,
  instead of resizing the whole frame up a fraction and back with bilinear. No real pixel is
  resampled any more, so the frames are strictly sharper at no cost (replicate padding keeps the
  flow net from tracking a hard edge). All three GMFSS scripts (this one, Fortuna's
  `inference_video.py`, enhancr) resized, so this was a known better technique none of them wired
  up. The blur that remains is motion ghosting at fast motion and occlusions, a flow accuracy
  problem. The AnimeRun
  fine tune idea is RESOLVED (2026-07-02): the shipped `train_log` weights already ARE the
  AnimeRun anime optical flow fine tune - verified by downloading both upstream Model Zoo zips
  (the bundled GMFSS_Fortuna README's own Drive links) and hashing: the "new union model using
  anime optical flow data fine-tune" is byte-identical to our five .pkl files, while the
  original union model differs in four of five (only rife.pkl shared). So there is no better
  drop-in weight set to swap to; the scene detection and dedup items (both now done) stop the
  model interpolating where the flow is meaningless. The heavy option - chaining a restoration
  model after interpolation, as enhancr does - is **done (2026-07-04)** as the opt-in
  `--restore` flag / GUI **Restore** checkbox: Real-ESRGAN's official anime-video model
  (`realesr-animevideov3`, 2.4 MB, bundled; vendored SRVGGNetCompact in `engine/realesr.py`,
  BSD-3) cleans compression noise and generatively redraws the linework on every output frame
  (not a mere filter - and not a texture builder either: on anime it tends to SMOOTH fine
  texture, see the trade-offs below). It runs FIRST in the
  per-frame chain (restore -> upscale -> RCAS -> HDR), so RTX VSR receives a restored,
  in-distribution frame - and without RTX VSR the model's own 4x output directly feeds the
  upscale (an `--upscale 2 --restore` render resizes the 4x reconstruction once instead of
  restoring, downscaling and re-upscaling). Measured on the sample at 2x: variance-of-Laplacian
  sharpness 2.2-2.4x the plain render's, with soft tween linework visibly rebuilt into clean
  outlines. Honest trade-offs: the model REPAINTS the frame (local value changes of a few 8-bit
  levels as it cleans haze and flattens some painterly texture - it targets cel-style anime),
  and every output frame pays a second model pass. That pass is real work (~2.6 TFLOP per 1080p
  frame), so it runs through the same per-resolution TensorRT engine cache as the GMFSS sub
  nets (~40 s one-time build, eager fp16 fallback): roughly +50% wall on a 2x 1080p render,
  proportionally more at higher multipliers since the cost scales with output frames, not
  pairs. The before/after preview pane mirrors it exactly (shared `realesr.py`, eager - one
  frame needs no engine). The `scale` flag is
  already at its sharp maximum (1.0); lowering it is what blurs.
- **FSR-style sharpening with RCAS (done; on by default in the GUI).** The uniform look (every output
  frame generated on a half step grid, see Uniform look) trades a little global sharpness for
  consistency, so the whole clip is a touch softer than the source. The counter is AMD FidelityFX
  **RCAS** (Robust Contrast-Adaptive Sharpening), the exact sharpen AMD FSR and Lossless Scaling's FSR
  mode use, implemented on the GPU in the engine (`_rcas()` in `gmfss_interp.py`, applied to every
  output frame in `to_bytes`). RCAS limits its sharpening lobe to the four-neighbour min/max (no
  overshoot or ringing) and eases off in noisy/textured regions (its denoise term), so it crisps real
  edges and recovers detail without amplifying fine texture into grain or mush. RCAS computes one
  scalar lobe per pixel and applies it to all three channels, so it cannot decorrelate them into
  colour speckle. The GUI **FSR** toggle plus a 0..1 strength slider (default **on at
  1.0**; RCAS self-limits, so 1.0 keeps texture) drives it; the on/off state and value
  persist between sessions. At the engine CLI the flag is off unless given (`--sharpen S`; a bare
  `--sharpen` uses 0.8; `0`/omitted leaves frames untouched). Verified on the sample (extracted frames,
  side-by-side crop): RCAS@1.0 crisps the dragon's edges and recovers mountain detail with the texture
  intact, at about 1.55 MB/frame versus the clean 1.42 MB. The sharpen runs at the
  **output** resolution: the per-frame order in `to_bytes` is upscale (RTX VSR or bicubic), then RCAS,
  then RTX TrueHDR. So RCAS crisps the final-resolution image (a source-resolution sharpen would just be
  scaled up and softened by the upscaler), the AI upscaler still receives an unsharpened, in-distribution
  frame, and the sharpen stays in SDR where its luma weighting is valid (ahead of the HDR expansion).
- **Upscale to any resolution + RTX VSR / RTX HDR (done).** The GUI **Upscale to** selector picks a
  target (Off / Match screen / 1080p / 1440p / 4K / 8K / 16K / custom height); the engine's `--upscale F`
  resizes each output frame just before encode by an arbitrary factor (aspect preserved, even
  dimensions), with decode and GMFSS interpolation staying at the source resolution. The AI backend is
  **NVIDIA RTX Video Super Resolution** (opt-in `--rtx-vsr`), with a bicubic resize as the fallback.
  RTX VSR places no integer-scale restriction on the output (probed clean and crash-free to 16K on a
  24 GB GPU - it is memory-bound, not model-bound), so any exact target up to the 16K option is allowed.
  **16K support (2026-07-02):** past 8192 px in either dimension NVENC cannot encode (probed: 8192
  passes, 8704 fails, both codecs) and HEVC as a format ends there too, so the engine probes the CPU
  encoders AT the output size and switches automatically - SVT-AV1 (AV1) up to ~12K (12288x6912
  passed, 14336x8064 refused on this build), then H.266/VVC (`libvvenc`, verified at 15360x8640,
  ~4 s/frame to encode) as the last encoder standing; a stderr notice names the pick, the GUI warns
  "CPU encode at this size", and the factor cap is 16x. Verified end to end from 1080p: 4x stays
  NVENC HEVC, 5x (9600x5400) auto-picks SVT-AV1, 8x (15360x8640) auto-picks VVC, with and without
  RTX VSR (VSR runs the full 16K natively). **RAM safety (added 2026-07-03 after a real DPC-watchdog
  BSOD, bugcheck 0x133):** the CPU encoders keep dozens of frames in flight at these sizes - the
  encode-side ffmpeg alone was measured at ~42 GB RSS during a 16K render (vvenc's ~30 GB working
  set plus ffmpeg's fixed inter-stage frame queues; GOP/thread knobs barely shrink it,
  `maxparallelframes=2` saves ~4 GB at no wall cost and is applied automatically) - and exhausting
  physical RAM with a small pagefile ends in commit exhaustion, stalled kernel drivers, and a
  hard reboot. The engine therefore runs a **fail-closed RAM preflight** for >8192 px outputs
  (~0.36 GB per megapixel for the VVC path, ~0.55 for SVT-AV1, calibrated on a monitored render
  that was watchdog-killed at 87% with ~47 GB consumed): true 16K needs ~54 GB of free RAM and is
  refused with an actionable message otherwise; ~10K (9600x5400) needs ~35 GB and passes on a
  64 GB machine. RTX HDR is separately capped at 8192 px (TrueHDR's colour math at 16K would
  oversubscribe VRAM, the other half of the crash risk). A 16K rgb48le pipe frame is ~0.8 GB (the
  writer queue is byte-bounded and ffmpeg's input queue/threads are capped at ultra sizes), and
  16K playback support is thin everywhere - treat it as an archival/stills-adjacent format;
  the old 2x/3x/4x limit was a quirk of Maxine SuperRes, which has been **removed** (DLSS never applied
  to video - it needs a game engine's motion vectors/depth/jitter, which a finished frame lacks).
  **RTX HDR (SDR to HDR10)** is also done (`--rtx-hdr`): the TrueHDR pass outputs 10-bit BT.2020 PQ
  (`x2rgb10le`), tagged HDR10 (p010 / main10 / smpte2084). VSR and TrueHDR run as **two separate bridge
  passes** (`run_vsr` then `run_hdr`, via the bridge's `evaluate_vsr_deviceptr` / `evaluate_thdr_deviceptr`
  entries), not the SDK's fused VSR->THDR eval, so the RCAS sharpen lands between them at the output
  resolution (see the FSR bullet for why that order is the quality-correct one).
  The HDR **peak brightness** (TrueHDR `MaxLuminance`) defaults to 1000 nits (the SDK allows
  400..2000, on the GUI slider / `--hdr-nits`). This is the **mastering peak**: a target the picture
  is graded to *once*, not a per-viewer setting. The default `vivid` mode keeps the source's own
  saturation (the model's hue-rotated chroma is dropped, so the SDK Saturation is inert there); colour
  pop comes from `--hdr-color rtx` (where the SDK `--hdr-saturation`, 0..200, ranges in
  `nvsdk_ngx_defs_truehdr.h`, drives NVIDIA's saturation hue-corrected) and from `--hdr-vibrance`.
  Contrast / MiddleGray stay at the SDK neutral 100 / 50.
  **HDR10 static metadata is written** (`engine/hdr10_meta.py`): the mastering-display colour volume
  (BT.2020 primaries, D65 white, the chosen peak) and the content light level (MaxCLL / MaxFALL
  **measured** from the actual frames) are stamped into the mp4 as the `mdcv` / `clli` boxes after the
  encode. The bundled LGPL ffmpeg cannot write these on the `hevc_nvenc` path (`-master_display` is a
  libx265/GPL option, the `hevc_metadata` bitstream filter has no such field, and `hevc_nvenc` exposes
  no mastering / CLL option - all verified against the bundled binary), so the boxes are injected
  directly at the container level (idempotent; `stco` / `co64` chunk offsets patched for any layout).
  With them, this one PQ file tone-maps correctly on both a 1000-nit and a 400-nit display with no
  per-display nits input, the same way an HDR10 / Dolby Vision master adapts (minus the per-shot
  dynamic metadata; see HDR mastering above). Both features run through a small compiled CUDA bridge
  `engine/rtxvideo/rtxvideo_cuda.dll` (built from the SDK's CUDA convenience layer plus a path shim;
  sources + build recipe in `engine/rtxvideo/build_src/`), which feeds each frame into NGX by GPU
  pointer the same zero-copy way TensorRT is driven; `engine/rtxvideo.py` wraps it. **Shipping:** the
  feature DLLs (`nvngx_vsr.dll`, `nvngx_truehdr.dll`) are non-redistributable and the bridge is a local
  build, so all of `engine/rtxvideo/` is gitignored and excluded from the packaged zip - RTX stays a
  local feature. Setup is one click in the **NVIDIA RTX** panel: it auto-detects a downloaded RTX Video
  SDK (Downloads / Desktop, a folder or a `.zip`) and **Install runtime** copies the two feature DLLs
  into `engine/rtxvideo` (zips are read with Windows' `bsdtar`, extracting only those two members);
  **Get from NVIDIA** opens the SDK page in the browser and **Choose folder / .zip** are manual
  fallbacks. The RTX toggles unlock only once the runtime is present (`rtx-ready` gates them). For
  viewing on this PC, mpv's live RTX VSR at playback stays the zero-storage alternative.
- **Overlapped decode, inference and encode (done).** Decode and encode now run on background
  reader and writer threads fed by bounded FIFO queues, so ffmpeg input and output pipe I/O
  overlap the next frame's GPU work instead of stalling a single thread (one reader and one
  writer preserve frame order, so the output is unchanged; the bounds apply backpressure rather
  than growing memory, and a failed encode write is surfaced as a nonzero exit). This matches the
  reader plus writer pattern in GMFSS_Fortuna's own inference_video.py. The host to device upload is
  non blocking too now: `to_tensor` stages each frame in pinned (page locked) memory and copies it with
  `non_blocking=True`, so the main thread no longer stalls on the copy and races ahead to queue the next
  frame's GPU work (a pageable `.to()` is synchronous; only a pinned source copies async). That closes
  the last synchronous stall in the overlap design.
- **Batch queue (done 2026-07-02).** Pick several files in the dialog (multi-select) or drop
  several onto the window and they queue: each renders back to back with the same settings and
  a default output name beside its source (a custom **Change...** path applies only to the file
  it was set on). The status line shows `File k/N`, the queue advances automatically on success,
  and it stops on a failure, an unreadable file, or Cancel (which also clears the queue). While
  the queue auto-advances, the probe alert is routed to the log (no modal to block an unattended
  run) and the before/after preview refresh is skipped. Renderer + main process only (the file
  dialog gained `multiSelections`), no engine change. Folder picking is not a separate mode:
  multi-select inside the folder (Ctrl+A) covers it.
- **AV1 and H.266/VVC output codecs (done 2026-06-28).** `--codec {hevc,av1,vvc}` plus the GUI
  **Codec** dropdown (labels + a per-choice hint spell out the trade-offs since 2026-07-03).
  `av1` uses the Blackwell hardware encoder (`av1_nvenc`, CQ 22 since the 2026-07-03 tuning; CPU
  `libsvtav1` fallback) - at the verified visually lossless settings it now measures slightly
  SMALLER than HEVC on the sample (2.04 vs 2.35 MB) while still exceeding HEVC's fidelity, and
  its other pitch is being royalty-free and decodable in every modern browser; `vvc` measured
  ~3.2x smaller (0.73 MB) and uses CPU `libvvenc`
  (QP 20 preset fast, always 10-bit `yuv420p10le` - its only input format - muxed with
  `-strict experimental`, falling back to HEVC when libvvenc is absent). The HDR10 `mdcv`/`clli`
  box injection is codec agnostic. Validated: AV1 SDR bt709, AV1 HDR PQ with boxes, VVC SDR 10-bit.
  HEVC stays the default for player compatibility.
- **Live preview of the frame being written (done 2026-07-02).** During a render the GUI shows a
  thumbnail of the frame most recently produced, under the progress bar, refreshing about once a
  second. The engine drops a 480p JPEG (`SMV_LIVE_PREVIEW`, set by the GUI to a userData path)
  from inside `to_bytes` - time-gated to ~1/s, and split producer/worker so the render thread
  only snapshots (a 480p GPU downscale + download, measured 0.25 ms per tick at 1080p and 4K,
  i.e. ~0.03% of the render) while a background thread does all decoding/tonemapping/JPEG work
  (~5 ms SDR, ~100 ms HDR per tick on an otherwise idle core; the pipeline is GPU-bound, see the
  performance audit). A busy worker skips ticks instead of queueing, a bounded flush at render
  end lets the final thumbnail land, and the cost is exactly zero for CLI runs without the env
  var; written tmp-then-rename so the poller never reads a torn file. The thumbnail shows what the OUTPUT will look like: an HDR
  render's frame is the graded TrueHDR result (the packed PQ bytes are unpacked, downscaled and
  tonemapped with the SAME source-anchored tonemap the before/after pane uses, so colour mode,
  vibrance and contrast all show - verified pixel-close to the pane's rendition, MAD 0.008, and a
  satboost 1.0 grade visibly lifts the thumb's saturation 83 -> 134), and an HDR source carried
  through gets the pane's self-anchored PQ display map instead of reading flat and washed out.
  The renderer polls the file's mtime once a
  second, anchors the watermark at run start so a previous run's leftover never shows, and keeps
  the final frame visible after Done. (Encoding itself was already done: always HEVC, 10-bit
  output by default with preserved colour, always visually lossless, with an automatic CPU
  SVT-AV1 fallback when NVENC is unavailable; see Passthrough quality.)
- **Performance headroom (ranked easiest to hardest, 2026-06-26 code audit).** The pipeline is already
  fp16 + cupy softsplat, TensorRT on all five sub-nets, pinned async upload, threaded decode/encode
  overlap and byte exact duplicate skip (`torch.compile` is ruled out for this model class: it would only
  duplicate the TensorRT backend with no win, and shipping a JIT compiler breaks the no-deps promise; MSVC
  is present locally but Triton is not, see Dev toolchain).
  Remaining code side wins, in implementation order:
  1. **GPU side transposes.** `to_tensor` / `to_bytes` do the HWC/CHW transpose on the CPU (a host copy
     per frame); upload and download in HWC and permute on the GPU instead. Low risk, bit identical.
     *Done 2026-06-26: correct and kept, but perf-neutral here (the CPU transpose was never the bottleneck).*
  2. **One shared CUDA stream, drop the per call TRT sync.** Every TRT sub-net call host synchronizes
     (`trt_runtime.py`), dozens of full GPU drains per pair at high multipliers. softsplat already runs
     on `torch.cuda.current_stream()`, so run the whole inference inside one stream (not the default
     one, so the stream warning stays fixed), enqueue TRT on it, and drop the per call sync. Medium risk.
     *Done 2026-06-26: replaces the per-engine-stream warning fix, keeps the warning gone and removes the
     redundant per-call syncs. Measured perf-neutral on a 16x 1080p render on the 5090 (removed ~840 host
     syncs per run with no wall-clock change), which shows the pipeline is **GPU-compute-bound** here, not
     sync or launch bound. That is the key finding: items 1 to 3 (I/O and syncs) cannot move a
     compute-bound wall clock; the real levers are items 4 and 5.*
  3. **Async pinned D2H.** `to_bytes` ends in a blocking pageable `.cpu()` each frame; pin a reusable
     host buffer, copy non blocking with an event, and let the writer thread resolve it so the download
     overlaps the next frame's compute. Low to medium risk.
     *Reassessed 2026-06-26: targets the same I/O path items 1 to 2 showed is not the bottleneck, so
     expected perf-neutral on a compute-bound render; deferred as cleanup only, not a speedup.*
  4. **Batch the per timestep inference at high multipliers.** `inference()` repeats per output frame
     with only `timestep` changing; batch the timesteps (batch dim = M - 1) to collapse 8 warps + IFNet
     + FusionNet per frame into one batched pass. Biggest win at 8x to 16x. Medium risk.
     *Probed 2026-06-26 (eager batch-scaling of the per-timestep nets): NOT worth it. FusionNet DOMINATES
     (~52ms/elem at 544x960) and does NOT benefit from batching - confirmed over 5 stable trials with
     `cudnn.benchmark=False`: per-elem 51.7ms (b1) -> 55.1ms (b2) -> 55.7ms (b4), i.e. ~6-8% net NEGATIVE,
     the flat-then-rising signature of a compute-saturated net. (A first pass showed ~2.8x worse, but that
     was a `cudnn.benchmark=True` algo-selection artifact: huge variance, it even mis-picked batch 1 at
     150ms vs 53ms run to run. Ruled out by the deterministic retest.) IFNet (the minor ~7ms net) batches
     better, but cannot offset the saturated bottleneck, so the overall ceiling is tiny. Confirms the
     pipeline is at its per-timestep compute limit. Skipped.*
  5. **fp8 / NVFP4 engines for GMFlow (the bottleneck).** Blackwell fp8 tensor cores are about 2x fp16;
     build GMFlow (and maybe Feature / Fusion) fp8, keep softmax / normalize fp16. Accuracy sensitive
     for a flow model, so gate on a PSNR check vs fp16. Was the standing "fp8 / NVFP4 not attempted" item.
     *Attempted 2026-06-26 (modelopt fp8 PTQ, quality-gated): FAILED the gate, reverted. GMFlow flow vs
     fp16 was fine on average (mean|d| 0.31px, rmse 0.67px, near the fp16-TRT baseline) but had
     catastrophic outliers (maxdiff 61px): fp8's narrow e4m3 range cannot hold GMFlow's large flow
     magnitudes (up to ~121px at half res), so a few pixels splat to the wrong place = visible glitches.
     modelopt's fp8 CUDA extension cannot build here (the CUDA Toolkit / nvcc is not installed; MSVC `cl.exe`
     19.29 IS present, see Dev toolchain), so calibration ran on the pure-torch fp8 sim, which is accurate
     (just slower) - so the 61px is real fp8 precision behaviour, not a tooling artefact. modelopt
     uninstalled, core stack verified intact. Selective per-layer fp8 (keep the wide-range correlation /
     attention layers in fp16, quantize only the safe convs) might salvage it, but that is deeper research.*
  6. **CUDA graph capture of `inference()`.** Fixed shapes; capture the ~20 op subgraph to kill launch
     overhead. Highest effort and risk, needs persistent I/O buffers. Only helps a launch-bound pipeline,
     which this is not (see below), so expected neutral. Not attempted.

  **Conclusion (2026-06-26 audit, all six probed or attempted):** items 1 to 3 are perf-neutral (the render
  is GPU-compute-bound, not sync/launch/IO-bound); 4 is skipped (the dominant per-timestep net FusionNet is
  already GPU-saturated and does not batch); 5 fails the quality gate (fp8 cannot hold GMFlow's flow range,
  61px outliers); 6 is irrelevant (not launch-bound). Net: the pipeline is at its practical compute limit on
  this hardware + model. Items 1 (transposes) and 2 (one shared stream, no per-call TRT sync) were kept as
  correct, non-regressing cleanups (2 also keeps the TRT default-stream warning fixed). Further speed would
  need a smaller/faster model, lower precision with accepted quality loss (fails for this flow model), or
  newer hardware - not a code change.
- **Smaller ffmpeg (done 2026-07-03).** `engine/bin` now bundles the BtbN LGPL **shared** build
  (small `ffmpeg.exe`/`ffprobe.exe` plus one set of FFmpeg DLLs next to them; `ffplay.exe` left
  out): 137 MB versus the old two ~174 MB static exes' 332 MB, a 195 MB saving that came
  entirely from no longer embedding two copies of libavcodec and friends. Verified end to end
  from the bundled location: all four encoders probe clean (`hevc_nvenc`, `av1_nvenc`,
  `libsvtav1`, `libvvenc`), the sample renders 2x in HEVC / AV1 / VVC (Main 10, 50 frames,
  structure identical to the static-build renders), and the worst-case HDR-into-MKV two-stage
  finalize carries every track and the value-exact HDR10 side data. One engine fix rode along:
  newer FFmpeg removed the deprecated underscore NVENC aliases, so the encode args now use
  `-spatial-aq`/`-temporal-aq` (dash forms, accepted for years, so older PATH ffmpegs keep
  working).
- **HDR colour controls in the GUI (final layout 2026-07-02).** Two features, mirroring the NVIDIA
  App's two separate filters, each with NVIDIA's own control names, App display scales, and a small
  red tick marking the default position on every slider:
  **RTX HDR** - inline right of its checkbox: **Contrast** and **Saturation**, both -100..100 with 0 =
  NVIDIA's default (the TrueHDR SDK scales 0..200 internally). Saturation is hue-corrected end to
  end: above -100 it drives NVIDIA's own TrueHDR saturation via the engine's `rtx` colour mode
  (chroma magnitude from the model, floored at the source, hue locked to the source so the cyan/teal
  cast is gone); at exactly -100 it routes to the source-chroma path (`vivid`), bit-exact true to
  source (the rtx floor would otherwise leave ~0.33/255 mean pixel difference). Default 0 = NVIDIA's
  out-of-box HDR look, hue-corrected; verified chroma tracks the SDK knob (0.044 to 0.056 across the
  range) with blue hue locked at the source 217 degrees (raw model sits at ~199).
  **RTX Dynamic Vibrance** - a checkbox with **Saturation boost** and **Intensity** to its right
  (both 0..100, defaults 50/50 as read off a fresh filter in the NVIDIA App 2026-07-02), greyed out
  until the feature is on, and inert at 0/0 (NVIDIA convention: the filter at zero strength changes
  nothing). NVIDIA's actual Dynamic Vibrance is a live game filter, not part of the RTX Video SDK,
  so this is SMV's hue-safe ICtCp analog stacking on top of whatever RTX HDR produced: Saturation
  boost (`--hdr-satboost`, 0..1 = +0..100%) is a uniform chroma gain, independent of RTX HDR's
  Saturation exactly like NVIDIA's two separate filters (verified uniform: low- and high-chroma
  pixels gain 1.498x/1.499x at 0.5, hue drift under 0.02 degrees); Intensity (`--hdr-vibrance`)
  weights the boost toward muted colours (1.79x low-chroma versus 1.40x high-chroma at full, skin
  protected). All persisted, passed to renders, picked up live by the preview pane, and RTX VSR
  always runs at the SDK's maximum quality (level 4, Ultra). The engine's `raw` mode (the unmodified
  model output, cyan cast and all) stays CLI-only (`--hdr-color raw`) as a debug/reference.
- **Before and after preview pane in the GUI (v2 done).** Shipped 2026-06-27, reworked 2026-06-28: a
  fast single-frame engine path (`engine/preview.py`), a `main.ts` `preview` IPC handler (returns the
  two PNG paths plus frame index and count), and a Preview panel above Smooth It in
  `renderer/index.html`, left the original frame, a large green right-arrow, right the processed frame,
  with back / random-frame buttons stepping through random positions across the clip. The processed
  side applies the SAME passes in the SAME order as a render: the FSR RCAS kernel itself (extracted to
  `engine/rcas.py`, shared by `gmfss_interp.py` and the preview), then TrueHDR when RTX HDR is on. The HDR tonemap is anchored to
  the source (median-luminance match plus a Reinhard highlight shoulder), replacing a p99 auto-exposure
  that made previews dim and washed out; midtones now match the source and expanded highlights bloom
  toward white. Clicking either image toggles 1:1 pixels with mirrored scrolling (sharpening is nearly
  invisible when a 1080p frame is shrunk to pane width). The pane auto-loads as soon as a video is
  selected (no button), shows a spinner over the processed image while a render is in flight, labels
  the result "Unchanged" when neither FSR nor RTX HDR is enabled (that case copies the frame straight
  through without importing torch, about 0.4 s), and serializes renders so rapid slider changes
  coalesce instead of overlapping. The pane follows every HDR colour control live (2026-06-28) and,
  since 2026-07-02, the Upscale / RTX VSR setting too: the processed side runs upscale (RTX VSR, or
  bicubic when the runtime is absent), then RCAS at the output resolution, then HDR, exactly the
  render's order, so the 1:1 zoom shows real output pixels; the 2026-07-04 Restore pass is
  mirrored too (shared `realesr.py`, eager for the single frame). Nothing remains outstanding
  for the pane.

History (already done): the build was made portable by bundling a relocatable
python-build-standalone runtime (replacing a non relocatable venv) and a bundled ffmpeg
(replacing the bare `ffmpeg`/`ffprobe` PATH dependency; static at first, the shared build
since 2026-07-03). The distributable is a **zip, not
an NSIS installer**: `makensis` cannot memory map an app archive this large (about 2.4 GB),
so the installer target was dropped.

## Constraints to keep in mind
- RTX 50 (Blackwell, sm_120): torch is the **cu130** build (CUDA 13). cupy-cuda13x locates the
  runtime via `cuda-pathfinder`, so `_add_cuda_dll_dirs` is no longer load-bearing for nvrtc.
- RTX Video bridge on CUDA 13: keep the cu12-built `rtxvideo_cuda.dll` and ship `cudart64_12.dll`
  next to it in `engine/rtxvideo/`. NVIDIA's NGX static lib is CUDA-12-ABI, so a bridge relinked
  against CUDA 13 crashes in `create()` (see `engine/rtxvideo/build_src/BUILD.md`).
- Keep `engine/runtime` a relocatable python-build-standalone install. Do not replace it
  with a `python -m venv` venv, which is not self contained and breaks the portable bundle.
- The renderer uses `require('electron')` with nodeIntegration on, so it cannot run in a
  plain browser. Launch via `npm start`, the shortcut, or the vbs.

## Engine CLI (used by the GUI, also runnable directly)
```
engine\runtime\python.exe engine\gmfss_interp.py <input> <multi> [output] [--scale 1.0] [--fps TARGET] [--no-trt] [--sharpen S] [--restore] [--no-interp] [--scene-detect] [--no-passthrough] [--upscale F] [--codec hevc|av1|vvc] [--out-bits 8|10] [--rtx-vsr] [--rtx-hdr] [--hdr-nits N] [--hdr-color vivid|rtx|raw] [--hdr-vibrance B] [--hdr-satboost S] [--hdr-mastering-prim display-p3|dci-p3|bt2020|bt709]
```

`--fps TARGET` overrides `<multi>` and resamples the timeline to any output fps (the model
interpolates at arbitrary fractional timesteps). `<multi>` stays required as a positional
but is ignored when `--fps` is given. The encode always targets visually lossless and the
backend is TensorRT by default (engines built and cached per resolution on first use, with
eager fallback on any failure); `--no-trt` forces the eager pipeline. `--sharpen S` applies
Contrast Adaptive Sharpening (`strength` 0..1) to every output frame to offset the uniform
look softness; it is off unless given (a bare `--sharpen` uses 0.8). `--no-interp` skips
interpolation entirely: the clip is only re-encoded at its source fps with `--sharpen`
applied (one output frame per source frame, no GMFSS model or TRT loaded), for when you
just want the sharpening and not the smoothing. `--out-bits` sets the output bit depth:
`10` (the default) encodes 10-bit even from an 8-bit source so the float-precision frames
never band, `8` restores the legacy 8-bit output for maximum device compatibility (8-bit
sources only; the output never drops below the source depth, and HDR and VVC are always
10-bit). `--scene-detect` enables hard-cut detection, holding detected cuts instead of interpolating
them (OFF by default: real action content lands in the detector's gray zone and a false hold
stutters the smoothing; see Scene change detection under What can be done next for the
calibration numbers and the field data behind the flip). Duplicate and near-duplicate frames
are no longer held or retimed: every source pair is interpolated uniformly on the same slot
grid (see Uniform interpolation under What can be done next), so the source timings are kept and
near-identical drawings smooth the same way as real motion.
`--restore` (off by default) runs Real-ESRGAN's anime-video model on every output frame to
clean compression noise and redraw linework (a generative repaint; fine texture can flatten),
before the upscale (without RTX VSR its 4x output directly
feeds the upscale); works with `--no-interp` too (restoration without smoothing). See Sharper
generated frames for the measurements and trade-offs.
`--no-passthrough` disables track passthrough (on by default: all audio/subtitle tracks,
chapters and fonts are copied, switching the default output to .mkv when the tracks need it;
see the Tracks bullet under Passthrough quality).
`--upscale F` spatially upscales every
output frame by an arbitrary factor `F` (e.g. `2.0`, or `1.5`, ...) just before encode,
leaving decode and interpolation at the source resolution; it is off at `1.0` unless given
(a bare `--upscale` uses `1.5`, clamped to 16.0; above 8192 px the encoder switches to CPU
AV1/VVC automatically, see the upscale bullet). With `--rtx-vsr` the upscale uses NVIDIA RTX
Video Super Resolution (real AI SR, any target resolution; needs the `engine/rtxvideo`
runtime), otherwise a bicubic resize. `--rtx-hdr` converts the output to HDR10 (BT.2020 PQ)
via the RTX Video TrueHDR model, and `--hdr-nits N` sets the mastering peak luminance (400..2000,
default 1000); the output also gets HDR10 static metadata (mastering-display plus measured
MaxCLL/MaxFALL, written by `engine/hdr10_meta.py`) so one file tone-maps to any display.
`--hdr-color` picks colour handling (default `vivid`: keep TrueHDR's luminance, take the SDR source's
hue AND chroma in ICtCp - faithful colour, and the SDK `--hdr-saturation` is inert here; `rtx`: drive
saturation with the SDK `--hdr-saturation` like real RTX TrueHDR but hue-corrected (TrueHDR's chroma
magnitude, source hue, floored at source) so the familiar NVIDIA slider works without the cyan cast;
`raw` emits TrueHDR's colour unmodified, a debug/reference mode with the cyan cast, CLI only and
deliberately not offered in the GUI), and
`--hdr-mastering-prim {display-p3,dci-p3,bt2020,bt709}` sets the `mdcv` mastering gamut by colorspace
name (default display-p3, metadata only).
The per-frame order is upscale (RTX VSR or bicubic), then RCAS sharpen at the
output resolution, then TrueHDR; VSR and TrueHDR run as two separate RTX bridge passes so the
sharpen can sit between them. See What can be done next for the RTX runtime / installer details.
