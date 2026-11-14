<p align="center">
  <img src="assets/icon.png" alt="Smooth My Video logo" width="128">
</p>

<h1 align="center">Smooth My Video</h1>

<p align="center">
  <b>Offline AI frame interpolation for NVIDIA GPUs.</b><br>
  Drop in a video, pick a target frame rate, and get a buttery-smooth high-FPS copy.<br>
  Built natively on NVIDIA CUDA: its AI runs on your GPU's Tensor Cores.<br>
  No cloud, no subscription, no install.
</p>

<p align="center">
  <a href="https://github.com/flowreen/SmoothMyVideo/releases/latest"><img src="https://img.shields.io/badge/download-windows%20x64-2ea44f" alt="Download for Windows"></a>
  <a href="https://github.com/flowreen/SmoothMyVideo/releases"><img src="https://img.shields.io/github/downloads/flowreen/SmoothMyVideo/total" alt="Downloads"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT license"></a>
</p>

<p align="center">
  <a href="#get-started">Get started</a> ·
  <a href="#models">Models</a> ·
  <a href="#picture">Picture</a> ·
  <a href="#file-renders">File renders</a> ·
  <a href="#live-mode">Live mode</a> ·
  <a href="#under-the-hood">Under the hood</a>
</p>

<p align="center">
  <img src="docs/demo-fight.webp" alt="Side by side: a fast 1080p anime fight and a backflip at their original 24 fps and interpolated to 60 fps" width="880"><br>
  <sub><b>The same scene, before and after.</b> About 8 seconds from episode 1 of <a href="https://sh-anime.shochiku.co.jp/jukishi-anime/"><i>The Exiled Heavy Knight Knows How to Game the System</i></a>, downscaled. Left: the source frames at 24 fps, no interpolation. Right: Smooth My Video's 60 fps render with the default GMFSS model, drawing the frames in between through sparks, blades, camera moves and a backflip.</sub><br>
  <sub><i>&copy; Nekoko, BroccoLee, Jaian, KODANSHA/"The Exiled Heavy Knight Knows How to Game the System" Production Committee (&copy;猫子・武六甲理衣・じゃいあん・講談社／「追放された転生重騎士はゲーム知識で無双する」製作委員会)</i></sub>
</p>

<p align="center">
  <img src="docs/demo-spiderman.webp" alt="Side by side: a fast live-action fight scene at its original 24 fps and interpolated to 60 fps" width="880"><br>
  <sub><b>Live action, too.</b> 14 seconds from <a href="https://en.wikipedia.org/wiki/Spider-Man:_Brand_New_Day"><i>Spider-Man: Brand New Day</i></a> (2026), downscaled. Left: the source frames at 24 fps, no interpolation. Right: Smooth My Video's 60 fps render with the default GMFSS model, drawing the frames in between through motion blur, flying debris and web lines.</sub><br>
  <sub><i>&copy; 2026 Columbia Pictures (Sony Pictures) and Marvel Studios</i></sub>
</p>

<p align="center">
  <img src="docs/demo-pan.webp" alt="Side by side: the same anime-style panning shot at 10 fps and interpolated to 60 fps" width="880"><br>
  <sub><b>Slow pans, too.</b> Left: an anime-style pan at 10 fps (drawn procedurally for this README), the cadence anime pans are actually drawn at. Right: interpolated 6&times; to 60 fps. Every original frame passes through untouched; the AI draws only the frames in between.</sub>
</p>

**Why it exists:** NVIDIA's RTX Video (AI upscaling + RTX HDR) is gorgeous but playback-only: it enhances what you
watch, then throws it away. Smooth My Video applies those same RTX passes, plus AI frame interpolation, straight to
your file, so the result is saved, not just streamed.

---

## Two modes

A **Video / Live** switch at the top of the window picks the mode. Each mode shows only its own controls, plus the
settings both share. The panels are numbered in the order the passes run.

<table>
  <tr>
    <td align="center" width="50%"><b>Video</b>: render a smoother copy of a file</td>
    <td align="center" width="50%"><b>Live</b>: smooth any window in real time</td>
  </tr>
  <tr>
    <td><img src="docs/ui.png" alt="The Smooth My Video window in Video mode: the Video / Live switch, Select video, then the numbered panels in processing order (Restore, DLSS 5, Sharpen, HDR with Dolby Vision and HDR10+, Interpolate with the Speed target, Upscale with the output Size, Output)" width="100%"></td>
    <td><img src="docs/ui-live.png" alt="The same window in Live mode: the Video / Live switch, the Smooth It Live button with its display, FPS meter and hotkey options, and the shared numbered panels (Restore, DLSS 5, Sharpen, RTX HDR, Interpolate with the Speed target, Upscale with the DLSS mode)" width="100%"></td>
  </tr>
  <tr>
    <td valign="top">Pick a video (or several, or drag them in), set a <b>target frame rate</b> and click
    <b>Smooth It!</b>. The smooth copy is written next to the original, with its audio tracks, subtitles and
    chapters carried over.</td>
    <td valign="top">Smooths any window on your screen (a video player, a browser, a windowed game) or the whole
    screen, with no file to render, Lossless-Scaling-style.</td>
  </tr>
</table>

**Speed** works in both modes: type a target fps, or pick **Screen refresh rate** to match your monitor. Video renders
keep the exact rate (59.94 stays 59.94), up to 100&times; the source. A whole multiple of the source rate is
"on-grid" (every original frame is kept), and the **&asymp;** button snaps to the nearest one. Live uses whole fps, up
to 10000.

## Get started

1. Download `SmoothMyVideo-<version>-win.7z` from the
   [latest release](https://github.com/flowreen/SmoothMyVideo/releases/latest) (the release notes carry its
   SHA-256).
2. Extract it anywhere. Windows 11 23H2 and later open `.7z` in File Explorer; on older Windows use
   [7-Zip](https://www.7-zip.org/).
3. Run **`SmoothMyVideo.exe`**. A second launch just brings the running window forward.

**Requirements:** an NVIDIA RTX GPU with a current driver. No Python, pip or ffmpeg to install, no account, no cloud
upload, and it is free. Built and tested on an RTX 5090 Laptop.

**Optional extras.** A few features need a file the app can't include. Tick the feature and the app shows a short
setup box. Most install with one click. NVIDIA's Optical Flow SDK sits behind a free NVIDIA developer login, so you
download that one yourself.

| Feature | What to get | How |
|---|---|---|
| RTX Video Super Resolution, RTX HDR | NVIDIA's RTX Video runtime | One click: **Download RTX Video** (NVIDIA's own package from NVIDIA's server, about 416 MB, checked and installed). Without it, upscaling uses a standard resize and RTX HDR is off. |
| NVIDIA DLSS 5 | The Neural Rendering runtime (`nvngx_dlssnr.dll`) | One click: **Download the DLSS 5 runtime** ([details](#nvidia-dlss-5)). |
| NVIDIA Video Frame Generation | NVIDIA's Video Frame Generation runtime | One click: **Download Video Frame Generation** (the same NVIDIA package as RTX Video, about 416 MB, checked and installed). Needs an RTX 40 or 50. |
| NVIDIA Smooth Motion | `NvOFFRUC.dll` from NVIDIA's Optical Flow SDK | **Get from NVIDIA** (free developer login), then **Choose .zip&hellip;**. GMFSS is used until it is installed. |
| Dolby Vision, HDR10+ | The open-source [dovi_tool](https://github.com/quietvoid/dovi_tool) / [hdr10plus_tool](https://github.com/quietvoid/hdr10plus_tool) | One click: **Download dovi_tool** / **Download hdr10plus_tool** (a fixed release, checked and installed). |

**From source:** install [Node.js](https://nodejs.org) 24 or later, then double-click `SmoothMyVideo.cmd`. The first
run sets the folder up by itself (the app's packages, ffmpeg and NVIDIA's CUDA runtime, under 1 GB) and starts the
app; later runs start it at once. A start that cannot build the app opens a window that shows why. The
`Smooth My Video` shortcut that appears beside it shows the app icon. See
**[DEVELOPMENT.md](docs/DEVELOPMENT.md)** for the rest.

## Models

One model at a time; every model works in videos and in Live. **NVIDIA DLSS 4.5** is the default on RTX 40 and 50 GPUs
(with Windows' hardware-accelerated GPU scheduling on), **GMFSS** on other GPUs. Untick every model to only enhance the
picture, without adding frames. Hover a row in the app to see its exact version.

| Model | Best for | Needs | Notes |
|---|---|---|---|
| **NVIDIA DLSS 4.5** (DLSS Frame Generation; default on RTX 40/50) | Most videos, high frame rates, Live | RTX 40/50, a recent driver, Windows hardware-accelerated GPU scheduling | NVIDIA's game frame generation, on video, and the fastest model. Makes whole multiples of the source rate, and reaches other targets with evenly spaced frames of a higher multiple (24 to 60 fps = every 2nd frame of 5&times;). An RTX 50 makes up to 6&times; in one pass, an RTX 40 2&times;; higher multipliers take several passes. |
| **GMFSS** (default on other GPUs) | Anime | Any RTX GPU | The cleanest frames, even at high multipliers; the heaviest on the GPU. |
| **RIFE** (Practical-RIFE 4.25) | Live action, filmed content | Any RTX GPU | A strong all-round model. Its **DRBA** option (for anime) keeps characters moving at their drawn pace while camera pans get fully smooth; in Live it adds one frame of delay. |
| **NVIDIA Optical Flow** | Slower PCs, fast renders, moderate motion | Any RTX GPU | Runs on a dedicated part of the GPU, so it is fast. Very fast motion turns into a soft double image instead of breaking up. Videos need a whole multiplier (2&times;, 3&times; ...). |
| **AMD FSR 4** frame generation | A different AI look | Any RTX GPU | AMD's AI frame generation, running on NVIDIA GPUs. Fine edges can break up on fast anime motion, where GMFSS stays clean. Each pass doubles the frame rate (2&times;, 4&times;, 8&times; ...). |
| **Intel XeSS-FG** frame generation | Filmed content, a different AI look | Any RTX GPU (Intel's files are downloaded on request) | Intel's AI frame generation from games, running on NVIDIA GPUs. In SMV's test on real footage its new frames came out as close to the real ones as RIFE's, closer than AMD FSR 4's and DLSS 4.5's. Each pass doubles the frame rate (2&times;, 4&times;, 8&times; ...). |
| **NVIDIA Video Frame Generation** | Any target frame rate on RTX 40 and 50 | RTX 40 or 50 (NVIDIA's files are downloaded on request) | NVIDIA's AI frame generation made for video rather than games. It makes a new frame at any point between two frames, so any target frame rate works. HDR and 10-bit video go through it at 10-bit precision. |
| **NVIDIA Smooth Motion** | Legacy: a non-AI reference | Any RTX GPU | The same technology as the NVIDIA driver's Smooth Motion. The lowest quality of the models. Each pass doubles the frame rate (2&times;, 4&times;, 8&times; ...). |

**Compare to**, in the preview, puts the model you ticked next to another one on the same moment of your video: the
real frames before and after on the left, each model's new frame between them on the right. Step through the new
frames with **&lsaquo; &rsaquo;**, click to zoom all four pictures together, and hold **Space** to swap the two
models. **Save image** keeps what you see; **Save clip** saves the next second as a lossless video of the four
pictures, every frame, to step through in a player.

<img src="docs/compare.webp" alt="Compare: the real frames 317 and 318 of a fight scene on the left, GMFSS's and NVIDIA DLSS 4.5's new frame halfway between them on the right. GMFSS draws her face once and clean; DLSS 4.5 doubles it but keeps the sparks and light beams crisp" width="100%">

## Picture

These run before the new frames are made, in videos and in Live:

* 🔍 **Restore**: cleans up compression noise and blur, with Real-ESRGAN (it can flatten fine texture), AnimeJaNai
  HD (downloaded on request) or Anime4K's line restore (light, made for anime, keeps the frame's size). Off by
  default.
* 🧠 **NVIDIA DLSS 5** (optional): NVIDIA's Neural Rendering, an AI picture enhancement on every original frame.
  [Details](#nvidia-dlss-5).
* ✏️ **Sharpen**: bacondither's Adaptive Sharpen, strength 0 to 2 (on at 1.00 by default).
* 🎨 **RTX HDR**: turns normal (SDR) video into HDR with the same colours as NVIDIA's RTX Video HDR in a player, with
  **Contrast** and **Saturation** sliders, plus optional **RTX Dynamic Vibrance** (richer colours, skin tones kept).
  Its brightest white matches your HDR display; type a **Peak** in nits to make a file for another screen. HDR sources
  stay as they are.

The new frames are made from the video at its own size. Then every frame, original and new, is enlarged:

* 📐 **Upscale**: makes the picture bigger and sharper with NVIDIA RTX Video Super Resolution (on by default; its Disable artifact
  removal tick is for high-bitrate or lossless video, whose clean detail it leaves as it is), AMD
  FSR 4.1, Intel XeSS (Intel's files are downloaded on request), Anime4K or ArtCNN (both made for anime; each upscale
  pass doubles the picture, so 720p to 4K takes two passes; with Restore set to Anime4K too, Anime4K makes its Mode
  A, and its A+A box runs Anime4K's Restore again on the enlarged picture: crisper lines, slower, and it can add
  ringing); with none ticked, a standard resize. They only run when the picture is enlarged. In Live with DLSS 4.5
  the upscaler stays off and DLSS 4.5's frames get the standard resize.

In the preview, **Show changed pixels** turns the processed side into a heat map of where DLSS 5 changed the picture.

## File renders

* 🎯 **On-grid keeps the real frames.** At a whole multiple of the source rate (2&times;, 3&times; ...), every original
  frame is kept and the AI only draws the frames in between. At other targets every output frame is a new one. A
  still picture stays perfectly still.
* 📏 **One Size control.** Pick the output **Size**: Off, your screen's resolution, 720p up to 8640p, or a custom
  height (240p to 16K). The AI work runs at the video's own size (or the Size, when that is smaller; at most 4K), and
  the result is enlarged at the end.
* 🌈 **Proper HDR10**, with real brightness data and accurate colour. Optional **Dolby Vision** and **HDR10+**,
  together or apart. Dolby Vision comes two ways: Profile 8.1 keeps the HDR10 video (other TVs play HDR10), Profile
  8.4 writes the video as HLG (other HDR TVs play HDR, SDR TVs show it close to right; not combined with HDR10+). All
  need RTX HDR, HEVC and an MP4 file; the app tells you when it has to skip them.
* 🗜️ **Visually lossless, 10-bit.** HEVC (default), AV1 or H.266 (VVC), at top quality with no knob to guess at.
  10-bit output keeps skies and glows free of banding.
* 💬 **Keeps your tracks.** All audio, subtitles / translations and chapters are copied, and the output switches to
  `.mkv` when MP4 cannot carry a track. Font attachments are kept in `.mkv` outputs. Above 1000 fps the output stays
  `.mp4` (MKV cannot time that many frames a second) and the subtitles are saved as separate files beside it, which
  players load by themselves.
* 📱 **Handles real-world files.** Phone clips and screen recordings with a changing frame rate stay in sync, HDR
  sources stay HDR, and DVDs and other stretched formats keep their shape.
* 🔁 **Reproducible.** The same file with the same settings renders byte-for-byte identically, every time, except
  with the NVIDIA Smooth Motion model and AMD FSR 4.1 upscaling.
* ⏯️ **Pause, preview, resume.** **Pause** holds the render; **Play preview** opens the part done so far, with sound
  and subtitles. A render that was closed or crashed continues where it stopped (with the same settings and an
  unchanged source). **Cancel** deletes the partial file. A render never leaves a cut-off file behind.
* ⚡ **Long renders in two halves at once.** A RIFE render of 600 frames or more renders its two halves side by side
  when your GPU has the memory for both (about 1.2&times; faster at 1080p, 1.6&times; at 8K) and joins them into one
  file, the same as a single run frame for frame. Pause, preview and resume work the same way; without the memory for
  two, it renders as one.
* 🧺 **Set-and-forget batches.** Pick or drop several files: they render one after another with the same settings, a
  file that fails is noted and the rest keep going, and a batch interrupted by a crash or restart is queued again on
  the next launch.
* 📊 **Always in view.** Frames, render fps, an ETA with the finish time and the projected file size (with a warning
  when the drive is too small), a thumbnail of the frame being written, taskbar progress, and a notification when a
  render or batch finishes while the window is in the background. Your PC will not sleep mid-render.

<details>
<summary><b>More: preview, output names, very large outputs, first runs</b></summary>

* **Before / after preview:** a still frame with your Restore, resize, DLSS 5, Sharpen and HDR settings over the
  original, with a divider you drag to compare the two. It updates as you change settings; step to a random frame,
  and click the picture for a 1:1 pixel view.
* **Output name:** `<name>_<fps>fps` (the fps rounded, so 59.94 gives `_60fps`; plus `_<height>p` when resized) next
  to the source, `.mp4` or `.mkv` as needed. With no model ticked it is `_<height>p`, or `_restored` / `_dlss5` /
  `_sharpened` / `_hdr` after the pass that runs. Edit the **Saves to** path in place or with **Change&hellip;**;
  clearing it brings the automatic name back. After a render, **Open folder** or **Play video**.
* **Outputs above 8192 px** encode on the CPU (AV1 or VVC); the app checks your free memory first (a 16K render
  needs about 54 GB), and RTX HDR stops at 8192 px.
* **Colour** stays accurate: conversions use exact math, and a file with no colour tags gets the usual standard for
  its size.
* **First run at a new size** prepares the AI models for it once (kept in `model_cache_safe_to_delete` in the app's
  folder, safe to delete); later runs start fast.
* **Laptops:** the app tells you when a "Silent" power profile is slowing the GPU.
* **The engine log** is `%TEMP%\smv-engine.log`; an error line in the app points at it.

</details>

## Live mode

Real-time frame generation on any window: the Lossless-Scaling-style feature, built in and free.

* **Start:** press the **hotkey** (` by default, changeable with the **Hotkey** button) in any app, or click
  **Smooth It Live!** and then click the window you want within 5 seconds.
* **Stop:** press the hotkey again, or click **Stop**. The hotkey is reserved system-wide while the app runs.
* **Display:** **Whole screen** (the default) smooths the whole monitor the active window is on, everything on it, and
  works with every model including DLSS 4.5. **Original size** smooths only the window. **Fill screen** fills its
  monitor with the window (aspect kept), upscaled with RTX VSR, AMD FSR 4.1, Intel XeSS, Anime4K or ArtCNN when
  ticked.
* **Adaptive:** with GMFSS, RIFE or NVIDIA Optical Flow the output holds your fps target whatever the source does (a
  game dropping from 30 to 10 fps still plays at the same smooth rate). AMD FSR 4, Intel XeSS-FG and Smooth Motion
  double the frame rate per pass, so they pick the doubling that reaches your target (60 fps from 24 shows 96). DLSS 4.5 matches
  your target, your screen's refresh rate included: a whole multiple when one fits, else evenly spaced frames of a
  higher one (60 fps from 24 = every 2nd frame of 5&times;).
* **Fit to the GPU** (on by default) lowers the frame rate to what your GPU can keep up with, never below the source,
  with every effect kept. Unticked, a target the GPU can't reach makes it skip frames.
* **Readout:** a small green corner readout shows the source fps, the smoothed fps, and roughly how far the picture
  runs behind reality: about a source frame, made for watching, not for competitive play. **Show FPS meter** turns it
  off; **Show latency** hides its latency part.
* **The first session at a new window size** takes a while to prepare (about 30 s for GMFSS, 45 s for RIFE, 55 s
  with DRBA); later ones start in seconds.
* **Windowed or borderless windows only** (exclusive fullscreen can't be captured).

<details>
<summary><b>More: DLSS mode, effects, HDR displays, what pauses and restarts it</b></summary>

* **DLSS mode** trades sharpness for speed: the work runs at a smaller size, then the picture is enlarged. DLAA (the
  default) is the full size; Quality 67%, Balanced 58%, Performance 50%, Ultra Performance 33%, or any size down to 1%
  with the slider. Lower is faster and softer.
* **Effects** (Restore, DLSS 5, Sharpen) work with every model, DLSS 4.5 included; the upscaler works with every model
  but DLSS 4.5. Untick every model to get just the effects, at the window's own frame rate.
* **On an HDR display** every model runs in HDR, and RTX HDR can turn SDR windows into HDR.
* **Pauses by itself** when the window is minimized, on another virtual desktop, or covered by the window in front,
  and resumes when it is back.
* **Restarts by itself**, on the same window, when you change the Speed target, the DLSS mode or Fit to the GPU, when
  the window is resized or moved to another monitor, and if the engine stalls. A closed window stops Live.
* **Fill screen and the mouse:** clicks on the stretched picture reach the matching spot of your window, and the
  pointer speed follows the stretch; the normal pointer comes back when the cursor leaves the picture or Live stops.
* **Fast restarts:** the loaded model stays ready after Stop, so the next session starts faster (not after DLSS 5 or
  RTX HDR; DLSS 4.5 keeps its own engine ready instead); it is freed after 10 idle minutes or when you pick another
  model.
* **RivaTuner (RTSS):** if it hooks the overlay, the engine log (`%TEMP%\smv-engine.log`) names it, with the fix.

</details>

## NVIDIA DLSS 5

One checkbox runs NVIDIA's DLSS 5 Neural Rendering on every original frame, in videos and in Live.

<details>
<summary><b>Where it runs, its controls, and the one-click download</b></summary>

* **Where:** after Restore, at the video's own size (or a smaller Size; at most 4K), before the new frames and the
  upscale (in Live, at the DLSS mode's size).
* **How:** fed the way games feed it, so it follows the motion, and a paused picture stays perfectly still.
* **Controls:** NVIDIA's three looks as a **Style** selector (Default, Natural, Cinematic; Default is preselected),
  **Structure Intensity** and **Tone Intensity** sliders, and **Passes** (1 to 20): DLSS 5 again on its own result,
  for a stronger effect at that many times the time and video memory. **Auto mask** (on by default) is NVIDIA's
  character mask: DLSS 5 finds the characters and treats their skin on its own, as much as the **Skin Structure**
  slider says (at 1.00, the default, skin follows Structure Intensity).
* **The download:** NVIDIA's Neural Rendering file (`nvngx_dlssnr.dll`) isn't included, and NVIDIA offers no download for it.
  The one-click **Download the DLSS 5 runtime** button fetches the community RTX 40 + 50 build
  ([rhi-repo](https://github.com/RankFTW/rhi-repo/releases) on GitHub), checks it and installs it.
* **In Live:** on every captured frame of SDR content.

</details>

## Everyday comforts

* Most settings are remembered between runs (Restore and RTX Dynamic Vibrance start off each launch), and so is the
  window size.
* Zoom the interface with Ctrl+= / Ctrl+- / Ctrl+0, or Ctrl + mouse wheel.
* The source panel shows the resolution, the frame rate (the exact fraction on hover), the duration and the codec,
  plus the track counts when there is more than one audio track or any subtitle.
* A newer release downloads in the background and installs itself at the next start, or when you press the green
  Update button that appears once it is ready; a quiet one-line notice shows the progress. Untick automatic updates
  on that line to keep the notice and its link only. The update replaces only the app's own files: videos or anything
  else you keep in the SmoothMyVideo folder stay. In a folder Windows does not let the app change (Program Files),
  the notice says so and links the new build instead. Pressing Update during a render pauses it: open the same video
  with the same settings after the restart and the button offers to resume it. A Live session stops.

## Under the hood

The UI is Electron + TypeScript. Every render and Live session runs in one native host (`smv-live.exe`, C++ on
CUDA: GMFSS, RIFE, Restore, Anime4K and ArtCNN run on its own CUDA kernels, their heavy layers on the Tensor Cores),
driven by a TypeScript orchestrator that handles probing, encoding (ffmpeg) and resume;
DLSS 4.5 file renders run NVIDIA's frame generation in a small separate helper. The default model on
GPUs without DLSS 4.5, **GMFSS_Fortuna**, is a "union" interpolator: GMFlow optical flow, an IFNet/RIFE refiner, plus MetricNet, FeatureNet,
FusionNet and softsplat warping.

File renders also run without the window: `node dist\render\cli.js <input> <multiplier> [output] [flags]` takes the
same settings the GUI builds, and the native host can start Live on a window by its title.

📖 **Every flag, the build and the design rationale: [DEVELOPMENT.md](docs/DEVELOPMENT.md)**

## Contributing

Smooth My Video was built end to end by AI coding agents ([Claude Code](https://claude.com/claude-code)), and it's
meant to keep growing that way. If there's a feature you want, open an issue and describe it, or send a pull request.
Contributions are welcome whether you write the code yourself or hand the idea to an agent, exactly how the rest of
this app was built.

---

<sub>Dolby and Dolby Vision are trademarks of Dolby Laboratories. Smooth My Video is an independent project and
is not affiliated with, endorsed by, sponsored by, or certified by Dolby Laboratories. Dolby Vision Profile
8.1 and 8.4 metadata is produced by the separately-installed, third-party open-source
[dovi_tool](https://github.com/quietvoid/dovi_tool); no Dolby software is bundled or redistributed.
HDR10+ is a trademark of HDR10+ Technologies, LLC; Smooth My Video is likewise not affiliated with, endorsed
by, or certified by HDR10+ Technologies. HDR10+ metadata is injected by the separately-installed, third-party
open-source [hdr10plus_tool](https://github.com/quietvoid/hdr10plus_tool); no HDR10+ LLC software is bundled
or redistributed.</sub>

<sub>The anime and film demo clips above (`docs/demo-fight.webp`, `docs/demo-spiderman.webp`) are short excerpts used
only to show what Smooth My Video does. The footage belongs to its copyright holders, credited under each clip, and is
not covered by this project's MIT license; Smooth My Video is not affiliated with or endorsed by them. Rights holders
who want a clip removed can [open an issue](https://github.com/flowreen/SmoothMyVideo/issues) and it will be taken
down promptly.</sub>
