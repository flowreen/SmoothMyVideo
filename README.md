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
- **SDR to HDR colour validated.** A real SDR/HDR reference pair confirmed the conversion is
  colour-faithful and the cyan/cool look is the HDR to SDR viewing transform, not the pipeline. See
  What can be done next.
- **CUDA 13 migration.** The bundled runtime moved to torch 2.12.1+cu130 + cupy-cuda13x + TensorRT
  cu13, validated end to end (eager, RTX HDR, TensorRT). See Setup.

**Software stack updates still pending:**
- **Re-bundle on CUDA 13.** `npm run dist` has not been re-run since the migration, so the shipped zip
  still carries the old cu128 runtime. Rebuild it before distributing.
- **Clean-machine test on CUDA 13.** The cu13 runtime (and the RTX `cudart64_12.dll` bundling) were
  validated on the dev box but not yet on a separate PC (the long-standing open item).
- **Optional torch bump.** Stable torch 2.12.1+cu130 is the current pick; 2.13/2.14 nightly cu130
  exist but there is no reason to move yet.

- GUI: select a video (or drag one onto the window), view its info (resolution, source
  fps, duration, codec), choose a multiplier, type a target fps, or tick **match screen
  refresh rate** to target your monitor's Hz (rounded up), then click **Smooth It!**. An
  **FSR** toggle (FSR-style RCAS sharpening, on at full strength by default, with a
  strength slider) crisps the output. An **Upscale to** selector resizes the output to a chosen
  resolution (Off / Match screen / 1080p / 1440p / 4K / 8K / a custom height), keeping the source
  aspect ratio; the **RTX Video Super Resolution** toggle (opt-in, see NVIDIA RTX) does that
  upscale with NVIDIA AI, otherwise it is a bicubic resize. The **Interpolate** toggle (on by
  default) is the master switch for frame generation: untick it to *only* sharpen / upscale the
  video, keeping the source frame rate (the multiplier / fps / match-screen controls grey out and
  the engine skips the GMFSS model entirely). An opt-in **NVIDIA RTX** panel adds real RTX Video
  Super Resolution and **RTX HDR** (SDR to HDR10, with a peak-brightness slider); both are off by
  default and unlock once the RTX Video runtime is installed (a one-click in-app installer, see
  NVIDIA RTX). **Cancel** kills the running job; **Open folder** reveals the result. The last used
  folder, multiplier, sharpen, upscale resolution, interpolate, match-screen, RTX and HDR settings
  are remembered between sessions.
- Progress: a bar that starts at the source frame count and fills to the post process
  total, plus a live frame counter and an ETA.
- Output: written beside the source as `<name>_<fps>fps.mp4` (or a custom path chosen
  with **Change...**). Always visually lossless, no quality knob. The output is always HEVC
  (`hevc_nvenc`) when the device has a usable NVENC session, since HEVC at the same quality is
  far smaller than H.264 and an interpolated clip is a new artifact (matching an H.264 source
  would only bloat it). The source bit depth is preserved (8 bit, or 10 bit and HDR via
  `p010le`), the source colour signalling is carried through, and original audio is copied. With
  no usable NVENC it falls back automatically to CPU SVT-AV1, still visually lossless (see
  Passthrough quality).
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
  and a static ffmpeg (NVENC plus a CPU SVT-AV1 fallback) at `engine/bin`. Both ship inside
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
  strength slider, output path with **Change...**, progress bar with frame counter and ETA,
  Cancel, Open folder, log). Uses `require('electron')`; a dropped file is resolved to a path
  with `webUtils.getPathForFile`, and the folder, multiplier, sharpen, interpolate and
  match-screen settings are saved in `localStorage`.
- `engine/gmfss_interp.py` - GMFSS pipe engine: ffmpeg decode into GMFSS into ffmpeg
  encode (audio copied), always encoding HEVC with the bit depth and colour tags matched to the
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
  version)` under `SMV_TRT_CACHE`, built on first use, with eager fallback on any failure.
- `engine/runtime/` - bundled relocatable Python (python-build-standalone CPython 3.14)
  with the full GPU stack installed (torch cu130 / CUDA 13, cupy-cuda13x, nvidia cu13 wheels). Gitignored, see Setup.
- `engine/bin/` - bundled static `ffmpeg.exe` + `ffprobe.exe` (built with `hevc_nvenc`)
  plus their license. Gitignored, see Setup.
- `engine/GMFSS_Fortuna/` - GMFSS model code (inference chain only) plus `train_log/`
  weights (gitignored, see Setup).
- `engine/benchmark.py` - speed benchmark; appends a dated entry to `BENCHMARKS.md`.

## Setup (fresh clone)
A fresh clone is missing three large, gitignored pieces: `engine/runtime`, `engine/bin`,
and `engine/GMFSS_Fortuna/train_log`. The app needs all three to run, and `npm run dist`
needs them present in order to bundle them.

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
Download a static Windows ffmpeg that includes `hevc_nvenc` (e.g.
[BtbN FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds/releases),
`ffmpeg-master-latest-win64-lgpl.zip`) and copy `bin\ffmpeg.exe` and `bin\ffprobe.exe`
into `engine\bin`. The app prefers these and only falls back to ffmpeg/ffprobe on PATH,
so for local dev you can skip this if you already have ffmpeg installed; it must be
present for a portable `npm run dist`.

**4. GMFSS weights into `engine/GMFSS_Fortuna/train_log`**
The weights (feat, flownet, fusionnet, metric, rife pkl files) are gitignored because
they are large. Restore them from the original GMFSS_Fortuna release.

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
The encode keeps the source's bit depth, chroma and colour, so the only deliberate changes are
interpolation and the codec (always HEVC). From the ffprobe of the input the engine sets:

- **Codec.** Always HEVC (`hevc_nvenc`), whatever the source codec was. HEVC at the same visually
  lossless CQ is far smaller than H.264, and the interpolated clip is a brand new artifact, so
  echoing the source codec would only bloat it: a 3 Mbps H.264 source produced a 100 Mbps H.264
  output under the old match the source rule, and HEVC brings that to about a quarter of the size
  at the same quality. Other interpolation tools agree (enhancr, Topaz and Flowframes all offer a
  codec menu centred on HEVC and AV1; GMFSS_Fortuna's own script only dumps `mp4v`). The encoder
  is preflighted on a tiny frame; if the device has no usable HEVC NVENC session (no NVIDIA GPU, a
  GPU too old, or no driver) the engine falls back to CPU `libsvtav1` for the encode.
- **Bit depth.** 8 bit decodes via `rgb24`; 10 bit and up decode via `rgb48le` and encode to
  `p010le` (HEVC `main10`). The pipe is already fp16, which holds 10 bit precision, so this is
  free. GMFSS_Fortuna and GMFSS_union are 8 bit only, and enhancr runs GMFSS inference in
  float but writes 8 bit (`YUV422P8`), so carrying 10 bit all the way to the file is one step
  past all three references.
- **Colour.** The source matrix, transfer, primaries and range are stamped on with a
  `setparams` filter (NVENC drops transfer and primaries from the bare `-color_*` output
  flags), so bt2020 / PQ / HLG HDR signalling survives the round trip.
- **Quality.** Always visually lossless, no knob. On NVENC: constant quality VBR around CQ 17
  (CQ 20 for AV1), the visually lossless point from the linked H.264 guide, with AQ on and a
  small chroma QP boost. On the `libsvtav1` fallback: CRF 20 at preset 8, SVT-AV1's visually
  lossless range. Audio is always copied.

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
2. **Do not add saturation.** TrueHDR Saturation defaults to **0** (faithful), not the SDK's own
   "neutral" 100: measured against the source, 100 pushes mean saturation about a third high (the SDR
   to HDR model adds vibrance of its own), while 0 lands on the original's colour and is not washed
   out. Contrast and MiddleGray stay at the SDK neutral 100 / 50. `--hdr-saturation` (0..200) tunes
   it; 100 restores the vivid look.
3. **Master to a fixed peak, set once.** `--hdr-nits` (default 1000) is the mastering peak the PQ
   values are shaped to, not a per-monitor knob. 1000 is the consumer standard; 4000 is premium.
4. **Write the mastering metadata.** `mdcv` (mastering-display: BT.2020 / D65 + the peak) and `clli`
   (MaxCLL / MaxFALL measured from the frames) are injected into the mp4 by `engine/hdr10_meta.py`. A
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

- **SDR to HDR colour / cyan (investigated, resolved).** Using a real reference pair (a Costa Rica 4K
  HDR clip and its SDR version), SMV's SDR to HDR was confirmed colour-faithful: tonemapping SMV's HDR
  back to SDR matches the source, and the *real* graded HDR cools even more under the identical
  tonemap, so the cyan/cool cast is the HDR to SDR viewing transform (a screenshot or a player's
  tonemap), not the conversion. A Contrast/MiddleGray sweep confirmed those knobs change tone, not
  colour (Saturation is the only colour control, defaulted to a faithful 0). The lone residual is a
  slight brightness lift / cool shift intrinsic to the HDR expansion that saturation cannot remove;
  TrueHDR exposes no white-balance control, so nudging it would need a custom shader or a post trim.
  Low priority, since on a real HDR display the file is correct.

- **Clean machine test (the one open item).** The packaged zip is self contained and was
  verified locally with system Python and ffmpeg stripped from PATH, but it has not yet
  been run on a *separate* PC. Copy `release/SmoothMyVideo-<version>-win.zip` to a machine
  with no Python and no ffmpeg (just an NVIDIA driver), extract, and run `SmoothMyVideo.exe`.
  Test both paths: a normal render, and one with the **TensorRT** toggle on (the first TRT
  render at a given resolution spends a few minutes building engines, then caches them).
- **Scene change detection (skip interpolation across true cuts).** The engine interpolates
  every pair, so at a hard cut it morphs one shot into the next and emits a smeared ghost
  frame. The fix is to detect the cut and emit a held frame instead of a tween. Do not use a
  raw pixel difference threshold for this: a fast camera pan or hard action also produces a
  large pixel difference and would be falsely flagged as a cut, killing interpolation exactly
  where it is most wanted. Use the optical flow GMFSS already computes (gmflow returns flow01
  and flow10 per pair): on a real cut the forward and backward flow disagree and the warp
  residual is large, while a fast pan has large but consistent flow with a small warp residual.
  A forward/backward flow consistency or warp error check separates the two cleanly and reuses
  work already done. enhancr leans on VapourSynth misc.SCDetect, a plain frame difference
  detector that has exactly this pan false positive, so this is a place to do better than it.
- **Duplicate frame handling for anime (exact case done).** GMFSS targets anime, which is drawn
  on twos and threes, so many consecutive source frames are identical. The engine now detects a
  byte exact duplicate pair and holds the frame instead of interpolating, skipping the wasted
  compute and the shimmer GMFSS can add on identical input (measured: 1 of 24 pairs in the sample
  clip, with the next nearest pair an order of magnitude away in mean difference, so exact match
  has no false positives). Still open: catch near duplicates too with a small downscaled frame
  difference threshold rather than exact equality, which also feeds the cut signal above (very
  high means cut, near zero means duplicate, in between means interpolate). Even so this cheap
  skip is not a full de judder: truly smoothing on twos motion needs duplicate removal plus
  retiming so the real motion is spread evenly to the target fps, a larger feature worth its own pass.
- **Sharper generated frames (free half done).** The free half is fixed: `to_tensor` and
  `to_bytes` now pad to the next multiple of 64 the model needs and crop the padding back,
  instead of resizing the whole frame up a fraction and back with bilinear. No real pixel is
  resampled any more, so the frames are strictly sharper at no cost (replicate padding keeps the
  flow net from tracking a hard edge). All three GMFSS scripts (this one, Fortuna's
  `inference_video.py`, enhancr) resized, so this was a known better technique none of them wired
  up. The blur that remains is motion ghosting at fast motion and occlusions, a flow accuracy
  problem: try the
  AnimeRun anime optical flow fine tune of the Fortuna weights (enhancr exposes it as GMFSS
  Fortuna Union, model 1) as a drop in weight swap for less ghosting on anime, and let the scene
  detection and dedup items stop the model interpolating where the flow is meaningless. The heavy
  option, used in enhancr, is to chain a restoration or upscaling model (RealESRGAN, SCUNet) after
  interpolation to re sharpen, at the cost of a second model per frame. The `scale` flag is
  already at its sharp maximum (1.0); lowering it is what blurs.
- **FSR-style sharpening with RCAS (done; on by default in the GUI).** The uniform look (every output
  frame generated on a half step grid, see Uniform look) trades a little global sharpness for
  consistency, so the whole clip is a touch softer than the source. The counter is AMD FidelityFX
  **RCAS** (Robust Contrast-Adaptive Sharpening), the exact sharpen AMD FSR and Lossless Scaling's FSR
  mode use, implemented on the GPU in the engine (`_rcas()` in `gmfss_interp.py`, applied to every
  output frame in `to_bytes`). RCAS limits its sharpening lobe to the four-neighbour min/max (no
  overshoot or ringing) and eases off in noisy/textured regions (its denoise term), so it crisps real
  edges and recovers detail without amplifying fine texture into grain or mush. It replaced an earlier
  attempt with ffmpeg's plain `cas` filter, which has neither limiter: in RGB it decorrelated channel
  noise into red/blue speckle, and even luma-only at high strength it ground fine mountain texture into
  grey grain (measured as 2.7-3.3x the high-frequency content of the clean frame, versus ~1.1x for
  RCAS). RCAS computes one scalar lobe per pixel and applies it to all three channels, so it cannot
  recreate the colour speckle. The GUI **FSR** toggle plus a 0..1 strength slider (default **on at
  1.0**; RCAS self-limits, so 1.0 keeps texture, unlike CAS) drives it; the on/off state and value
  persist between sessions. At the engine CLI the flag is off unless given (`--sharpen S`; a bare
  `--sharpen` uses 0.8; `0`/omitted leaves frames untouched). Verified on the sample (extracted frames,
  side-by-side crop): RCAS@1.0 crisps the dragon's edges and recovers mountain detail with the texture
  intact, at about 1.55 MB/frame versus the clean 1.42 MB (CAS was 2.7-3.3 MB). The sharpen runs at the
  **output** resolution: the per-frame order in `to_bytes` is upscale (RTX VSR or bicubic), then RCAS,
  then RTX TrueHDR. So RCAS crisps the final-resolution image (a source-resolution sharpen would just be
  scaled up and softened by the upscaler), the AI upscaler still receives an unsharpened, in-distribution
  frame, and the sharpen stays in SDR where its luma weighting is valid (ahead of the HDR expansion).
- **Upscale to any resolution + RTX VSR / RTX HDR (done).** The GUI **Upscale to** selector picks a
  target (Off / Match screen / 1080p / 1440p / 4K / 8K / custom height); the engine's `--upscale F`
  resizes each output frame just before encode by an arbitrary factor (aspect preserved, even
  dimensions), with decode and GMFSS interpolation staying at the source resolution. The AI backend is
  **NVIDIA RTX Video Super Resolution** (opt-in `--rtx-vsr`), with a bicubic resize as the fallback.
  RTX VSR places no integer-scale restriction on the output (probed clean and crash-free to 16K on a
  24 GB GPU - it is memory-bound, not model-bound), so any exact target up to the 8K option is allowed;
  the old 2x/3x/4x limit was a quirk of Maxine SuperRes, which has been **removed** (DLSS never applied
  to video - it needs a game engine's motion vectors/depth/jitter, which a finished frame lacks).
  **RTX HDR (SDR to HDR10)** is also done (`--rtx-hdr`): the TrueHDR pass outputs 10-bit BT.2020 PQ
  (`x2rgb10le`), tagged HDR10 (p010 / main10 / smpte2084). VSR and TrueHDR run as **two separate bridge
  passes** (`run_vsr` then `run_hdr`, via the bridge's `evaluate_vsr_deviceptr` / `evaluate_thdr_deviceptr`
  entries), not the SDK's fused VSR->THDR eval, so the RCAS sharpen lands between them at the output
  resolution (see the FSR bullet for why that order is the quality-correct one).
  The HDR **peak brightness** (TrueHDR `MaxLuminance`) defaults to 1000 nits (the SDK allows
  400..2000, on the GUI slider / `--hdr-nits`). This is the **mastering peak**: a target the picture
  is graded to *once*, not a per-viewer setting. TrueHDR Saturation defaults to **0** (faithful): the
  SDK's own "neutral" 100 measurably oversaturates versus the SDR source (about a third high), and 0
  lands on the original's colour without washing it out. Contrast / MiddleGray stay at the SDK neutral
  100 / 50; `--hdr-saturation` (0..200, ranges in `nvsdk_ngx_defs_truehdr.h`) tunes it, 100 = vivid.
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
- **Batch queue.** Process several files (or a folder) unattended, one after another. Pure UI
  work in the renderer and main process, no engine change.
- **Optional smaller items.** Add a live preview of the frame being written. (Encoding is now
  done: always HEVC, preserved bit depth and colour, always visually lossless, with an automatic
  CPU SVT-AV1 fallback when NVENC is unavailable; see Passthrough quality.)
- **TensorRT fp8 / NVFP4.** The native TensorRT backend is done (about 2.2x, see
  Performance). The remaining headroom is fp8 or NVFP4 on the Blackwell tensor cores,
  which needs calibration and is accuracy sensitive for a flow model; not attempted.
- **Smaller ffmpeg.** `engine/bin` uses static builds (about 174 MB each). A shared ffmpeg
  build would shrink the bundle by a couple hundred MB at the cost of carrying its DLLs.

History (already done): the build was made portable by bundling a relocatable
python-build-standalone runtime (replacing a non relocatable venv) and a static ffmpeg
(replacing the bare `ffmpeg`/`ffprobe` PATH dependency). The distributable is a **zip, not
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
engine\runtime\python.exe engine\gmfss_interp.py <input> <multi> [output] [--scale 1.0] [--fps TARGET] [--no-trt] [--sharpen S] [--no-interp] [--upscale F] [--rtx-vsr] [--rtx-hdr] [--hdr-nits N]
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
just want the sharpening and not the smoothing. `--upscale F` spatially upscales every
output frame by an arbitrary factor `F` (e.g. `2.0`, or `1.5`, ...) just before encode,
leaving decode and interpolation at the source resolution; it is off at `1.0` unless given
(a bare `--upscale` uses `1.5`, clamped to 8.0). With `--rtx-vsr` the upscale uses NVIDIA RTX
Video Super Resolution (real AI SR, any target resolution; needs the `engine/rtxvideo`
runtime), otherwise a bicubic resize. `--rtx-hdr` converts the output to HDR10 (BT.2020 PQ)
via the RTX Video TrueHDR model, and `--hdr-nits N` sets the mastering peak luminance (400..2000,
default 1000); the output also gets HDR10 static metadata (mastering-display plus measured
MaxCLL/MaxFALL, written by `engine/hdr10_meta.py`) so one file tone-maps to any display.
The per-frame order is upscale (RTX VSR or bicubic), then RCAS sharpen at the
output resolution, then TrueHDR; VSR and TrueHDR run as two separate RTX bridge passes so the
sharpen can sit between them. See What can be done next for the RTX runtime / installer details.
