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

Working end to end and verified by the user on real clips. Nothing is half finished.

- GUI: select a video (or drag one onto the window), view its info (resolution, source
  fps, duration, codec), choose a multiplier, click **Smooth It!**. **Cancel** kills the
  running job; **Open folder** reveals the result.
- Progress: a bar that starts at the source frame count and fills to the post process
  total, plus a live frame counter and an ETA.
- Output: written beside the source as `<name>_<fps>fps.mp4`, encoded with
  `hevc_nvenc`, original audio copied through.
- Engine: GMFSS at fp16 with a cupy softsplat kernel, about 2.2x faster than the
  original fp32 path. See Performance below.
- Launch: Desktop and Start menu shortcuts, a no console `SmoothMyVideo.vbs`, and a
  custom icon.
- A sample 24fps clip ships in `samples/gtest.mp4` for quick testing.

## Run it
- Double click the **SmoothMyVideo** shortcut (Desktop / Start menu), or `SmoothMyVideo.vbs`.
- Or from a terminal: `npm start` (builds, then launches).
- Pick a video, choose a multiplier (2x / 4x / 8x / 16x), click **Smooth It!**. The
  result is written next to the source as `<name>_<fps>fps.mp4`.

## Architecture
- `src/main.ts` - Electron main: window, file dialog, ffprobe (`-of json`), spawns the
  engine, streams progress, tracks the running child so **Cancel** can `taskkill /T /F` it.
- `renderer/index.html` - the UI (select or drag in a video, multiplier, progress bar
  with frame counter and ETA, Cancel, Open folder, log). Uses `require('electron')`;
  a dropped file is resolved to a path with `webUtils.getPathForFile`.
- `engine/gmfss_interp.py` - GMFSS pipe engine: ffmpeg decode (rgb24) into GMFSS into
  ffmpeg encode (`hevc_nvenc`, audio copied). Always fp16. Prints `PROGRESS k/total` to
  stderr. `_add_cuda_dll_dirs()` puts the nvidia wheel bin dirs on the Windows DLL search
  before the model imports so cupy can JIT its kernel.
- `engine/GMFSS_Fortuna/` - GMFSS model code (inference chain only) plus `train_log/`
  weights (gitignored, see Setup).
- `engine/benchmark.py` - speed benchmark; appends a dated entry to `BENCHMARKS.md`.

## Setup
GUI deps:
```
npm install
```
If the Electron binary did not download (its postinstall is sometimes skipped):
```
node node_modules/electron/install.js
```
Engine (Python 3.12). torch must be the Blackwell cu128 build for RTX 50:
```
python -m venv engine\.venv
engine\.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
engine\.venv\Scripts\python -m pip install -r engine\requirements.txt
```
`requirements.txt` includes `cupy-cuda12x` plus the `nvidia-cuda-nvrtc-cu12` and
`nvidia-cuda-runtime-cu12` wheels. cupy needs those to JIT its softsplat kernel; the
engine adds their bin dirs to the Windows DLL search at startup (`_add_cuda_dll_dirs`)
so `nvrtc-builtins*.dll` is found.

The GMFSS weights live in `engine/GMFSS_Fortuna/train_log/` (feat, flownet, fusionnet,
metric, rife pkl files). They are gitignored because they are large, and must be present
for the engine to run. Restore them from the original GMFSS_Fortuna release if a fresh
clone is missing them.

## Scripts
- `npm start` - build and launch.
- `npm run starti` - wipe `node_modules` and `package-lock.json`, fresh install, then start.

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
For whoever picks this up. Ordered roughly small to large. The recurring small step work
is essentially exhausted; the real remaining items are the two larger ones.

Small UX:
- Arbitrary target fps. The original intent allowed any output fps (editable, up to 1000).
  The UI currently exposes only fixed multipliers (2x, 4x, 8x, 16x). Add a free numeric
  fps field. The engine takes a multiplier today and could take a target fps instead.
- Remember the last used folder for the file dialog and the last chosen multiplier.
- Let the user pick a custom output location instead of always writing beside the source.

Larger:
- Package into a distributable installer with electron-builder. The catch is size: it has
  to ship the Python venv with torch and cupy, which is multiple GB. Decide between
  bundling the venv or a first run setup step that builds it.
- TensorRT speed path. This is the headroom beyond the cupy ceiling (fp8 or NVFP4, more
  kernel fusion). It is a much larger undertaking for a conv heavy model like GMFSS and
  was not attempted.

Constraints to keep in mind:
- RTX 50 (Blackwell, sm_120): torch must be the cu128 build, and do not break
  `_add_cuda_dll_dirs` or cupy will fail to find `nvrtc-builtins`.
- The renderer uses `require('electron')` with nodeIntegration on, so it cannot run in a
  plain browser. Launch via `npm start`, the shortcut, or the vbs.

## Engine CLI (used by the GUI, also runnable directly)
```
engine\.venv\Scripts\python engine\gmfss_interp.py <input> <multi> [output] [--scale 1.0]
```
