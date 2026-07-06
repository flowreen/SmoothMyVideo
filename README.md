<p align="center">
  <img src="icon.png" alt="SmoothMyVideo logo" width="128">
</p>

<h1 align="center">SmoothMyVideo</h1>

<p align="center">
  <b>Offline AI frame interpolation for NVIDIA GPUs.</b><br>
  Drop in a video, pick a target frame rate, and get a buttery-smooth high-FPS copy —
  no cloud, no subscription, no install.
</p>

---

## What it does

Pick a video (or drag it in), choose a **target frame rate** (double it, 4×, 8× and up, or **match
your monitor's refresh**), and click **Smooth It!**. SmoothMyVideo generates the in-between frames
with a GMFSS AI model on your GPU and writes a smoother, high-frame-rate copy right next to the
original. In the same render it can also **upscale** (up to 16K), **sharpen**, **restore detail**, and
convert **SDR → real HDR10** — while carrying over every audio track, subtitle, chapter and font.

Built and tested on an RTX 5090 Laptop; runs on any recent NVIDIA GPU with a current driver.

## Why choose it

- 🎞️ **Smooth *and* sharp.** Your real frames pass through at full quality with AI-generated frames woven
  in between — you get the higher frame rate without softening or reprocessing the original footage.
- 🧊 **10-bit output by default.** Float-precision interpolated frames are written at 10-bit, so smooth
  gradients (skies, glows) never band into visible steps.
- 🎨 **Production-grade HDR10.** Real SDR→HDR10 conversion with proper mastering metadata
  (mastering-display + measured MaxCLL/MaxFALL) and faithful, cyan-free colour — not just a PQ tag.
- 🔍 **AI upscaling to 16K + detail restoration.** NVIDIA RTX Video Super Resolution plus a Real-ESRGAN
  restore pass, layered with the interpolation in a single render.
- 💬 **Keeps every track.** All audio, subtitles/translations, chapters and font attachments are preserved
  (auto-switches to `.mkv` when needed) — nothing silently dropped.
- 🗜️ **Visually lossless, small files.** HEVC / AV1 / H.266 encodes tuned against a lossless 8K master
  (VMAF ~99.8, SSIM ≥ 0.995) — no fiddly quality knob to guess at.
- 📦 **100% offline & self-contained.** Extract the zip and run — no Python, no pip, no ffmpeg to install,
  no account, no cloud upload. Only the NVIDIA driver is assumed. Free.
- ⚡ **Fast.** fp16 with a TensorRT backend, built and cached per resolution — about 2.2× over the eager path.
- 🎚️ **Plus the essentials:** FSR-style sharpening, a batch queue, a live before/after preview, and every
  setting remembered between runs.

## Get started

- **Download & run:** grab the latest `SmoothMyVideo-<version>-win.zip`, extract it anywhere, and run
  **`SmoothMyVideo.exe`** (or the Desktop / Start-menu shortcut). No install, no dependencies — just a
  current NVIDIA driver. A sample clip ships in `samples/test.mp4`.
- **From source:** `npm start`. See **[DEVELOPMENT.md](DEVELOPMENT.md)** for the full setup.

## Under the hood

The UI is Electron + TypeScript; the interpolation runs in a Python **GMFSS_Fortuna** engine spawned as a
subprocess. GMFSS_Fortuna is a "union" interpolator — GMFlow optical flow, an IFNet/RIFE refiner, plus
MetricNet, FeatureNet, FusionNet and softsplat warping — producing clean frames even at high multipliers.
It runs fp16 with a cupy softsplat kernel and an optional TensorRT backend.

📖 **Build it, hack on it, or read the design rationale → [DEVELOPMENT.md](DEVELOPMENT.md)**
