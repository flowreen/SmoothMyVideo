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

## Status (2026-06-19)

Working end to end and verified on real clips. The packaged build is now fully self
contained: a recipient extracts the zip and runs `SmoothMyVideo.exe` with no Python, no
pip, and no ffmpeg installed (only the NVIDIA driver is assumed).

- GUI: select a video (or drag one onto the window), view its info (resolution, source
  fps, duration, codec), choose a multiplier or type a target fps, click **Smooth It!**.
  **Cancel** kills the running job; **Open folder** reveals the result. The last used
  folder and multiplier are remembered between sessions.
- Progress: a bar that starts at the source frame count and fills to the post process
  total, plus a live frame counter and an ETA.
- Output: written beside the source as `<name>_<fps>fps.mp4` (or a custom path chosen
  with **Change...**). Always visually lossless, no quality knob. Passthrough encoding: the
  encoder family is matched to the source codec (h264/hevc/av1) using NVENC when the device
  has a usable encode session, the source bit depth is preserved (8 bit, or 10 bit and HDR
  via `p010le`), the source colour signalling is carried through, and original audio is
  copied. With no usable NVENC it falls back automatically to CPU SVT-AV1, still visually
  lossless (see Passthrough quality).
- Engine: GMFSS at fp16 with a cupy softsplat kernel, about 2.2x faster than the original
  fp32 path. The TensorRT backend is the default when available (about another 2.2x on top,
  built and cached per resolution on first run) and falls back automatically to the eager
  pipeline when TensorRT is unavailable. See Performance below.
- Bundled: a relocatable Python 3.14 runtime (torch cu128 + cupy) at `engine/runtime`,
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
  `taskkill /T /F` it. Resolves the interpreter as `engine/runtime/python.exe`
  (`RUNTIME_PY`, falls back to `python` on PATH) and ffprobe as `engine/bin/ffprobe.exe`
  (`FFPROBE`, falls back to `ffprobe` on PATH). It always sets `PYTHONUTF8` plus a writable
  `SMV_TRT_CACHE` (under userData) for the engine cache, since the engine runs the TensorRT
  backend by default.
- `renderer/index.html` - the UI (select or drag in a video, multiplier or fps, output path
  with **Change...**, progress bar with frame counter and ETA, Cancel, Open folder, log).
  Uses `require('electron')`; a dropped file is resolved to a path with
  `webUtils.getPathForFile`, and the last folder and multiplier are saved in `localStorage`.
- `engine/gmfss_interp.py` - GMFSS pipe engine: ffmpeg decode into GMFSS into ffmpeg
  encode (audio copied), with the encoder family, bit depth and colour tags matched to the
  probed source and always targeting visually lossless (see Passthrough quality). Runs the
  TensorRT backend by default (per-subnet eager fallback; `--no-trt` forces eager) and NVENC
  with an automatic CPU SVT-AV1 fallback. Always fp16; takes an integer `<multi>` or
  `--fps TARGET` for an arbitrary resampled output fps. Prints `PROGRESS k/total` to
  stderr. Resolves `ffmpeg`/`ffprobe` from `engine/bin` first and falls back to PATH
  (`_tool()`). `_add_cuda_dll_dirs()` puts the nvidia wheel bin dirs on the Windows DLL
  search before the model imports so cupy can JIT its kernel.
- `engine/trt_runtime.py` - optional TensorRT backend. `trtify(model)` swaps the five GMFSS
  sub nets for engines exported under autocast and built strongly typed (fp16); softsplat
  and the interpolate glue stay in eager. Engines are cached per `(net, shapes, gpu, trt
  version)` under `SMV_TRT_CACHE`, built on first use, with eager fallback on any failure.
- `engine/runtime/` - bundled relocatable Python (python-build-standalone CPython 3.14)
  with the full GPU stack installed (torch cu128, cupy, nvidia wheels). Gitignored, see Setup.
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
its inner `python/` folder to `engine/runtime`. Then install the Blackwell cu128 GPU
stack into it (this is the exact environment that gets bundled):
```
engine\runtime\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
engine\runtime\python.exe -m pip install -r engine\requirements.txt
```
`requirements.txt` includes `cupy-cuda12x` plus the `nvidia-cuda-nvrtc-cu12` and
`nvidia-cuda-runtime-cu12` wheels. cupy needs those to JIT its softsplat kernel; the
engine adds their bin dirs to the Windows DLL search at startup (`_add_cuda_dll_dirs`)
so `nvrtc-builtins*.dll` is found. It also pulls `tensorrt` plus `onnx`/`onnxscript` for
the default TensorRT backend; `tensorrt` brings about 2 GB of cu13 libraries that coexist
with torch's cu128. A standard `python -m venv` is **not** usable here: a
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

## Scripts
- `npm start` - build (`tsc`) and launch.
- `npm run starti` - wipe `node_modules` and `package-lock.json`, fresh install, then start.
- `npm run pack` - build an unpacked app into `release/win-unpacked/` with the engine
  (including its bundled `runtime` and `bin`) shipped as `extraResources`. For local testing.
- `npm run dist` - build the distributable `release/SmoothMyVideo-<version>-win.zip`
  (about 5 GB with the TensorRT backend bundled, or 3.2 GB without it). Recipients extract
  it and run `SmoothMyVideo.exe`; no install step, and nothing is required on the target
  machine but the NVIDIA driver.

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

Tried and not viable on this machine: `torch.compile` (its inductor backend needs
Triton plus MSVC, neither installed); and **dynamic shape** TensorRT engines (one engine
covering a range of resolutions via a `min`/`opt`/`max` profile). The latter fails because
GMFlow and IFNet warp with `grid_sample`, which the dynamo ONNX exporter routes through
`cudnn_grid_sampler` with no ONNX translation on the dynamic path (the static export
handles it fine). Dynamic engines are also documented as somewhat slower with more VRAM,
so static per resolution engines, built on first use and cached, are the right call.

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
  (torch 2.11 `cu128`, TensorRT 11.1).

## Passthrough quality
The encode is matched to the source so interpolation is the only thing that changes, not the
format. From the ffprobe of the input the engine picks:

- **Codec family.** h264 -> `h264_nvenc`, hevc -> `hevc_nvenc`, av1 -> `av1_nvenc`, anything
  else -> `hevc_nvenc`. A 10 bit clip that arrived as h264 is promoted to HEVC because NVENC
  H.264 is 8 bit only. The chosen NVENC encoder is preflighted on a tiny frame; if the device
  has no usable NVENC session (no NVIDIA GPU, a GPU too old for that codec, or no driver) the
  engine falls back to CPU `libsvtav1` for the encode.
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
*source* is not possible for an interpolator (frames are invented and the originals are round
tripped through RGB), so "passthrough" here means adding no visible encode loss and not
downgrading the source format.

## What can be done next
For whoever picks this up.

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
- **Duplicate frame handling for anime.** GMFSS targets anime, which is drawn on twos and
  threes, so many consecutive source frames are identical. Running inference on an identical
  pair is wasted compute and can make the flow model shimmer on degenerate input; detecting
  the duplicate and repeating the frame avoids both. A cheap downscaled frame difference covers
  this and the cut signal above with one metric (very high means cut, near zero means duplicate,
  in between means interpolate). This cheap skip is not a full de judder: truly smoothing on
  twos motion needs duplicate removal plus retiming so the real motion is spread evenly to the
  target fps, which is a larger feature worth its own pass.
- **Sharper generated frames.** The blur in interpolated frames has two parts with different
  fixes. The free one: the engine makes dimensions divisible by 64 by resizing the whole frame
  up a fraction and back with bilinear (`to_tensor` and `to_bytes`), which resamples every pixel.
  Padding to the next multiple of 64 with `F.pad`, running the model, then cropping back resamples
  none of the real content and is strictly sharper. All three GMFSS scripts (this one, Fortuna's
  `inference_video.py`, enhancr) resize, so this is a known better technique none of them wired
  up; Fortuna even ships an unused `pad_image` function that hints at the intent. The effect is
  subtle (about a one percent stretch at 1080p), free, and never worse. The blur people actually
  notice is motion ghosting at fast motion and occlusions, a flow accuracy problem: try the
  AnimeRun anime optical flow fine tune of the Fortuna weights (enhancr exposes it as GMFSS
  Fortuna Union, model 1) as a drop in weight swap for less ghosting on anime, and let the scene
  detection and dedup items stop the model interpolating where the flow is meaningless. The heavy
  option, used in enhancr, is to chain a restoration or upscaling model (RealESRGAN, SCUNet) after
  interpolation to re sharpen, at the cost of a second model per frame. The `scale` flag is
  already at its sharp maximum (1.0); lowering it is what blurs.
- **Overlapped decode, inference and encode.** The loop reads one frame, interpolates, then
  writes, all on one thread, so the GPU stalls during pipe I/O. GMFSS_Fortuna's own
  inference_video.py runs a reader thread and a writer thread feeding bounded queues; porting
  that pattern plus non_blocking host to device copies overlaps ffmpeg I/O with GPU work for a
  throughput gain on top of fp16, cupy and TensorRT.
- **Batch queue.** Process several files (or a folder) unattended, one after another. Pure UI
  work in the renderer and main process, no engine change.
- **Optional smaller items.** Add a live preview of the frame being written. (Encoding is now
  done: source matched codec, preserved bit depth and colour, always visually lossless, with
  an automatic CPU SVT-AV1 fallback when NVENC is unavailable; see Passthrough quality.)
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
- RTX 50 (Blackwell, sm_120): torch must be the cu128 build, and do not break
  `_add_cuda_dll_dirs` or cupy will fail to find `nvrtc-builtins`.
- Keep `engine/runtime` a relocatable python-build-standalone install. Do not replace it
  with a `python -m venv` venv, which is not self contained and breaks the portable bundle.
- The renderer uses `require('electron')` with nodeIntegration on, so it cannot run in a
  plain browser. Launch via `npm start`, the shortcut, or the vbs.

## Engine CLI (used by the GUI, also runnable directly)
```
engine\runtime\python.exe engine\gmfss_interp.py <input> <multi> [output] [--scale 1.0] [--fps TARGET] [--no-trt]
```

`--fps TARGET` overrides `<multi>` and resamples the timeline to any output fps (the model
interpolates at arbitrary fractional timesteps). `<multi>` stays required as a positional
but is ignored when `--fps` is given. The encode always targets visually lossless and the
backend is TensorRT by default (engines built and cached per resolution on first use, with
eager fallback on any failure); `--no-trt` forces the eager pipeline.
