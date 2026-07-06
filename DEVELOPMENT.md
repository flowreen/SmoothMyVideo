# SmoothMyVideo — Technical & Developer Guide

Build instructions, architecture, the engine CLI, and design rationale. For the product overview see
[README.md](README.md).

## Status

Works end to end. The packaged build is fully self-contained: a recipient extracts the zip and runs
`SmoothMyVideo.exe` — no Python, no pip, no ffmpeg, only the NVIDIA driver. Built and tested on an
RTX 5090 Laptop (Blackwell, sm_120); the CUDA 13 stack (torch 2.12.1+cu130, cupy-cuda13x, TensorRT
cu13) is validated across eager, TensorRT, RTX VSR/HDR and all three codecs.

## Architecture

- **`src/main.ts`** — Electron main: window, open/save dialogs, ffprobe (`-of json`), spawns the engine,
  streams progress, tracks the child so **Cancel** can `taskkill /T /F` it. IPC for the monitor refresh
  rate (match-screen), screen size, and the single-frame preview. Resolves the interpreter as
  `engine/runtime/python.exe` and ffprobe as `engine/bin/ffprobe.exe` (both fall back to PATH); sets
  `PYTHONUTF8` and a writable `SMV_TRT_CACHE`.
- **`renderer/index.html`** — the UI: select/drag a video, a target-fps control, an **FSR** sharpen
  toggle, **Restore**, **Upscale**, a **Codec** selector, an opt-in **NVIDIA RTX** panel (VSR + HDR),
  output path, progress + ETA, a batch queue, a live thumbnail, and a before/after preview pane. Electron
  `require` with `nodeIntegration`; most settings persist in `localStorage` (Restore and RTX Dynamic
  Vibrance deliberately don't — per-session opt-ins).
- **`engine/gmfss_interp.py`** — the GMFSS pipe engine: ffmpeg decode → GMFSS → ffmpeg encode. TensorRT
  backend by default (per-subnet eager fallback; `--no-trt`), NVENC with a CPU SVT-AV1 fallback, always
  fp16, always visually lossless, 10-bit by default. Prints `PROGRESS k/total` to stderr.
- **`engine/trt_runtime.py`** — optional TensorRT backend. Swaps the five GMFSS sub-nets for strongly-typed
  fp16 engines; softsplat + the interpolate glue stay eager. Engines are cached per
  `(net, shapes, gpu, trt version, weights hash)`; the weights-hash in each filename makes the cache
  self-invalidating on a weight swap (stale engines deleted at next start).
- **`engine/rtxvideo.py`** + **`engine/rtxvideo/`** — the RTX Video bridge (VSR + TrueHDR) over a small
  compiled CUDA DLL (`rtxvideo_cuda.dll`, sources in `build_src/`). The non-redistributable NGX feature
  DLLs are user-installed via the in-app NVIDIA RTX panel; the whole folder is gitignored / excluded from
  the zip, so RTX stays a local feature.
- **`engine/realesr.py`** — the `--restore` Real-ESRGAN detail pass (vendored SRVGGNetCompact, BSD-3).
- **`engine/hdr10_meta.py`** — pure-stdlib ISOBMFF injector for HDR10 static metadata (`mdcv`/`clli`).
- **`engine/preview.py`** — single-frame before/after preview (same passes, same order as a render).
- **`engine/runtime/`** — bundled relocatable Python 3.14 (python-build-standalone) with the CUDA 13 GPU
  stack. Gitignored (see Setup).
- **`engine/bin/`** — bundled shared-build `ffmpeg.exe` + `ffprobe.exe` and their DLLs. Fetched, not committed.
- **`engine/GMFSS_Fortuna/`** (model + `train_log/` weights) and **`engine/realesr-animevideov3.pth`** —
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
  release zip) — it's the ready-to-run interpreter, nothing else to do.
- *From scratch:* unpack a
  [python-build-standalone](https://github.com/astral-sh/python-build-standalone/releases) CPython 3.14
  `install_only` win64 build to `engine/runtime`, then:
```
engine\runtime\python.exe -m pip install torch==2.12.1 torchvision --index-url https://download.pytorch.org/whl/cu130
engine\runtime\python.exe -m pip install -r engine\requirements.txt
```
`requirements.txt` pulls cupy-cuda13x, the **unsuffixed** `nvidia-cuda-nvrtc` / `nvidia-cuda-runtime` cu13
wheels (the `-cu13` names are deprecated placeholders that fail to build), `tensorrt` (cu13), and
onnx/onnxscript. A `python -m venv` is **not** usable — a Windows venv isn't relocatable and breaks the
portable bundle.

### Refreshing bundled binaries
- **ffmpeg:** delete `engine/bin` and re-run `node scripts/fetch-ffmpeg.js`. To pin an exact build, drop a
  matched `ffmpeg.exe` + `ffprobe.exe` + `*.dll` set in by hand — never mix DLLs across builds (the exe
  links specific SONAME majors like `avcodec-63`).
- **Weights:** the GMFSS `train_log` pkls (from the GMFSS_Fortuna release) and `realesr-animevideov3.pth`
  (Real-ESRGAN v0.2.5.0). Both committed; this is only for updating them.

## Scripts
- `npm start` — build (`tsc`) and launch.
- `npm run dist` — the build command: wipes `release/`, compiles with `tsc`, runs electron-builder (zip
  target → both `release/win-unpacked/` and `SmoothMyVideo-<version>-win.zip`, ~4 GB with TensorRT
  bundled). Recipients extract and run `SmoothMyVideo.exe`; nothing required on the target but the NVIDIA
  driver. (A zip, not an NSIS installer — `makensis` can't memory-map an archive this large.)

## Engine CLI (used by the GUI, also runnable directly)
```
engine\runtime\python.exe engine\gmfss_interp.py <input> <multi> [output] [--scale 1.0] [--fps TARGET] [--no-trt] [--sharpen S] [--restore] [--no-interp] [--no-passthrough] [--upscale F] [--codec hevc|av1|vvc] [--out-bits 8|10] [--rtx-vsr] [--rtx-hdr] [--hdr-nits N] [--hdr-color vivid|rtx|raw] [--hdr-vibrance B] [--hdr-satboost S] [--hdr-mastering-prim display-p3|dci-p3|bt2020|bt709]
```
- `<multi>` integer multiplier, or `--fps TARGET` to resample to any output fps (the model interpolates at
  arbitrary fractional timesteps; `<multi>` is required positionally but ignored when `--fps` is given).
- `--sharpen S` (0..1) FSR-style RCAS on every output frame (bare `--sharpen` = 0.8; off unless given).
- `--no-interp` re-encodes at source fps with sharpen only (no model/TRT loaded).
- `--restore` runs the Real-ESRGAN detail pass per output frame, before the upscale (works with `--no-interp`).
- `--upscale F` spatial upscale just before encode (bare = 1.5, clamp 16.0; above 8192 px auto-switches to
  a CPU AV1/VVC encoder). `--rtx-vsr` uses RTX Video Super Resolution, else bicubic.
- `--rtx-hdr` SDR→HDR10 (BT.2020 PQ) via TrueHDR; `--hdr-nits` mastering peak (400..2000, default 1000);
  `--hdr-color` {`vivid` (default: source hue+chroma), `rtx` (SDK saturation, hue-corrected), `raw` (debug)};
  `--hdr-mastering-prim` sets the `mdcv` gamut by name.
- `--out-bits` {`10` default, `8` legacy}; `--codec` {`hevc` default, `av1`, `vvc`}; `--no-passthrough` first-audio-only.
- Per-frame order: (restore →) upscale → RCAS sharpen → TrueHDR.

## How it works (key behaviour)

**Interpolation.** On the default `--multi` path every real source frame passes through at its integer
timestamp at full quality, and M-1 AI-generated tweens are inserted between each pair (`--fps` resamples to
an arbitrary rate off the source grid). Keeping the real frames on-grid is deliberate (2026-07-05): a
bracket blend `inference(f[k-1], f[k+1], 0.5)` would skip the real frame's pose, and interpolating two
already-generated tweens would double-fade it — so the frames we already have are kept at max quality.
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
lossless 8K master: HEVC **CQ 17** (VMAF 99.78 / 57.0 dB / SSIM 0.9986), AV1 **CQ 22**, VVC **QP 20** — all
past the visually-lossless bar on mean and worst frame. vvenc's perceptual QP adaptation is switched off
above 120 fps output (it inverts on wall-to-wall tween streams and bloats the file).

**Upscale + RTX.** `--upscale` to any resolution up to 16K (RTX VSR, or bicubic fallback). Past 8192 px
NVENC/HEVC can't encode, so the engine probes CPU encoders at the output size and auto-switches
(SVT-AV1 → VVC), plus a **fail-closed RAM preflight** (true 16K needs ~54 GB free; the CPU encoders keep
dozens of large frames in flight). RTX HDR is a real HDR10 master: 10-bit BT.2020 PQ + injected
mastering-display / content-light metadata, with source-faithful (cyan-free) colour rebuilt in ICtCp
(TrueHDR itself rotates hues even at Saturation 0, so its chroma is dropped and the source's is transplanted).

**Restore.** `--restore` runs Real-ESRGAN's anime-video model per output frame to clean compression noise
and redraw linework (a generative repaint — it targets cel-style anime and can flatten fine texture).
~+50% wall at 2× 1080p; runs through the same per-resolution TensorRT cache as the GMFSS sub-nets.

**FSR sharpen.** AMD FidelityFX **RCAS** at the output resolution crisps the softer generated tweens; on by
default at 1.0. It limits its lobe to the neighbour min/max (no overshoot/ringing), eases off in noisy
regions, and applies one scalar per pixel to all channels (so it can't decorrelate them into colour speckle).

**Preview / batch / live.** A before/after pane runs the same passes on a single frame (click for 1:1
pixels); a batch queue renders picked/dropped files back to back; a ~1/s live thumbnail shows the graded
output frame during a render (near-zero render cost — a producer/worker split keeps the render thread only
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

## Constraints
- **CUDA 13 (Blackwell, sm_120):** torch is the cu130 build; cupy-cuda13x finds the runtime via
  `cuda-pathfinder`, so the old `_add_cuda_dll_dirs` nvrtc shim is no longer load-bearing.
- **RTX bridge:** keep the **cu12**-built `rtxvideo_cuda.dll` and ship `cudart64_12.dll` beside it — NGX's
  static import lib is CUDA-12-ABI, so a bridge relinked against CUDA 13 crashes in `create()`
  (see `engine/rtxvideo/build_src/BUILD.md`).
- **Runtime:** keep `engine/runtime` a relocatable python-build-standalone install, never a `venv`.
- **Renderer:** uses `require('electron')` with `nodeIntegration`, so it can't run in a plain browser —
  launch via `npm start`, the shortcut, or the vbs.

## Dev toolchain
The dev machine has VS 2019 Build Tools (MSVC `cl.exe` 19.29) + the Windows 10 SDK, enough to build the
RTX Video bridge — the NGX SDK's entry points live in a static import lib (`nvsdk_ngx_s.lib`), so ctypes
alone can't reach them (recipe in `engine/rtxvideo/build_src/BUILD.md`). Nothing that needs MSVC is
bundled; a recipient still needs only the NVIDIA driver.
