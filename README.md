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

## Status (2026-06-18)

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
  with **Change...**), encoded with `hevc_nvenc`, original audio copied through.
- Engine: GMFSS at fp16 with a cupy softsplat kernel, about 2.2x faster than the
  original fp32 path. See Performance below.
- Bundled: a relocatable Python 3.14 runtime (torch cu128 + cupy) at `engine/runtime`,
  and a static ffmpeg with `hevc_nvenc` at `engine/bin`. Both ship inside the zip; the
  app uses neither system Python nor system ffmpeg.
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
  (`FFPROBE`, falls back to `ffprobe` on PATH).
- `renderer/index.html` - the UI (select or drag in a video, multiplier or fps, output
  path with **Change...**, progress bar with frame counter and ETA, Cancel, Open folder, log).
  Uses `require('electron')`; a dropped file is resolved to a path with
  `webUtils.getPathForFile`, and the last folder and multiplier are saved in `localStorage`.
- `engine/gmfss_interp.py` - GMFSS pipe engine: ffmpeg decode (rgb24) into GMFSS into
  ffmpeg encode (`hevc_nvenc`, audio copied). Always fp16; takes an integer `<multi>` or
  `--fps TARGET` for an arbitrary resampled output fps. Prints `PROGRESS k/total` to
  stderr. Resolves `ffmpeg`/`ffprobe` from `engine/bin` first and falls back to PATH
  (`_tool()`). `_add_cuda_dll_dirs()` puts the nvidia wheel bin dirs on the Windows DLL
  search before the model imports so cupy can JIT its kernel.
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
so `nvrtc-builtins*.dll` is found. A standard `python -m venv` is **not** usable here: a
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
  (about 3.2 GB). Recipients extract it and run `SmoothMyVideo.exe`; no install step, and
  nothing is required on the target machine but the NVIDIA driver.

## Performance
Precision is **fp16 only** (fp32 was removed: fp16 is visually lossless versus fp32,
PSNR about 51 dB, and just slower). The fast path is **fp16 plus cupy softsplat**.

Core inference at 1080p (warmup plus cuda.synchronize, excludes model load and ffmpeg I/O):
- original fp32: about 357 ms per frame
- fp16, pure torch softsplat: about 276 ms per frame
- fp16 plus cupy softsplat: about 160 ms per frame, roughly 2.2x over the fp32 baseline

A 360 frame clip at 16x lands around 17 minutes. Measure with `engine/benchmark.py`,
which logs dated entries to `BENCHMARKS.md` so progress is tracked over time.

Tried and not viable on this machine: `torch.compile` (its inductor backend needs
Triton plus MSVC, neither installed).

## What can be done next
For whoever picks this up.

- **Clean machine test (the one open item).** The packaged zip is self contained and was
  verified locally with system Python and ffmpeg stripped from PATH, but it has not yet
  been run on a *separate* PC. Copy `release/SmoothMyVideo-<version>-win.zip` to a machine
  with no Python and no ffmpeg (just an NVIDIA driver), extract, and run `SmoothMyVideo.exe`.
- **TensorRT speed path.** The headroom beyond the cupy ceiling (fp8 or NVFP4, more kernel
  fusion). A much larger undertaking for a conv heavy model like GMFSS; not attempted.
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
engine\runtime\python.exe engine\gmfss_interp.py <input> <multi> [output] [--scale 1.0] [--fps TARGET]
```

`--fps TARGET` overrides `<multi>` and resamples the timeline to any output fps (the model
interpolates at arbitrary fractional timesteps). `<multi>` stays required as a positional
but is ignored when `--fps` is given.
