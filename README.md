# SmoothMyVideo

Desktop app for offline AI video frame interpolation on NVIDIA GPUs. The UI is
Electron + TypeScript; the interpolation runs in a Python GMFSS engine spawned as a
subprocess. Built and tested on an RTX 5090 Laptop (Blackwell, sm_120).

## Run it
- Double click the **SmoothMyVideo** shortcut (Desktop / Start menu), or `SmoothMyVideo.vbs`.
- Or from a terminal: `npm start` (builds, then launches).
- Pick a video, choose a multiplier (2x / 4x / 8x / 16x), click **Smooth It!**. The
  result is written next to the source as `<name>_<fps>fps.mp4`, with audio preserved.

## Architecture
- `src/main.ts` - Electron main: window, file dialog, ffprobe, spawns the engine, streams progress.
- `renderer/index.html` - the UI (select video, multiplier, frame-scaled progress bar, log).
- `engine/gmfss_interp.py` - GMFSS pipe engine: ffmpeg decode (rgb24) -> GMFSS -> ffmpeg encode (hevc_nvenc, audio copied). Always fp16. Prints `PROGRESS k/total`.
- `engine/GMFSS_Fortuna/` - GMFSS model code (inference chain only) + `train_log/` weights (anime union model).
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

## Scripts
- `npm start` - build and launch.
- `npm run starti` - wipe `node_modules` + `package-lock.json`, fresh install, then start.

## Speed
Precision is **fp16 only** (fp32 was removed: it is visually lossless vs fp32,
PSNR ~51 dB, and just slower). The fast path is **fp16 + cupy softsplat**, about
**2.2x** over the original fp32 / pure-torch baseline. Measure with
`engine/benchmark.py` (logs dated entries to `BENCHMARKS.md`).

Tried and not viable on this machine: `torch.compile` (its inductor backend needs
Triton + MSVC, neither installed). Remaining headroom (fp8 / NVFP4, more fusion)
would need a **TensorRT** path, a much larger undertaking.

## Engine CLI (used by the GUI, also runnable directly)
```
engine\.venv\Scripts\python engine\gmfss_interp.py <input> <multi> [output] [--scale 1.0]
```
