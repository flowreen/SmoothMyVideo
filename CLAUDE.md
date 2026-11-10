# SmoothMyVideo: map for coding agents

Electron + TypeScript GUI over one native host, `engine/live/smv-live.exe`, which runs both file renders and Live
mode: AI frame interpolation, upscaling and HDR for NVIDIA GPUs, fully offline. Users read `README.md`, developers
read `docs/DEVELOPMENT.md` (build recipes, CLI reference, every `SMV_*` lever); this file says where things are and
which rules a change must keep. These three are the only docs: add to them, never create another.

## Commands

| Command | What it does |
|---|---|
| `npm start` | `tsc`, the renderer's type check, the renderer build (`app.ts` and its parts -> `app.js`), then the app |
| `npm run lint` | prettier (writes) on `src/**/*.ts`, `scripts/*.ts`, `renderer/*.ts`, `renderer/i18n/*.json`; `tsc --noEmit`; `tsc -p scripts`; `tsc -p renderer`; the lever check; clang-format (writes) on every `engine/*/build_src` |
| `node scripts/check-levers.ts` | every `SMV_*` environment lever a source reads has its row in DEVELOPMENT.md's "Environment variables", and no row names a lever that is gone |
| `node scripts/scan-index.ts` | after ANY edit of `renderer/index.html` or `renderer/app*.ts`: builds `app.js`, checks duplicate ids and `$()` lookups |
| `node scripts/build-renderer.ts --where N` | the part and line that line N of `renderer/app.js` comes from (an error's line number) |
| `node scripts/cpp-check.ts analyze` (or `w4`) | static checks of the native host; needs what `build.bat` needs (Visual Studio, the SDK paths in DEVELOPMENT.md) |
| `SmoothMyVideo.cmd` | the double-click entrance; sets a fresh folder up first |

Never run unasked: `npm run setup` (it deletes `node_modules` and `engine/bin`), `npm run dist` (the release build),
a version bump in `package.json`.

## Where things are

| Path | What |
|---|---|
| `src/main.ts` | the Electron main process: windows, dialogs, setup downloads, the update, spawning and talking to the live host |
| `src/render/*.ts` | file renders and the CLI: `cli.ts` (entry), `native.ts` (drives the host), `plan.ts`, `probe.ts`, `encode.ts`, `resume.ts`, `preview.ts`, `dvhp.ts` + `hdr10meta.ts` (Dolby Vision, HDR10+), `cache.ts` |
| `renderer/index.html` | the window's markup and CSS |
| `renderer/app.ts` + `renderer/app-<name>.ts` | the window's logic: one classic script in parts (table below). `renderer/app.js` is its build: never edit or read it |
| `renderer/i18n/<code>.json` | every UI text, eleven languages (`en` is the source; `ar` is right-to-left) |
| `scripts/*.ts` | build and setup scripts; Node runs them directly, no compile step |
| `engine/live/build_src/` | the native host's source; `build.bat` there builds `engine/live/smv-live.exe` |
| `engine/<name>/build_src/` | one bridge each, built into the DLL or exe beside it: `dlssg` (DLSS frame generation helper), `dlssnr` (DLSS 5), `fsrfg` + `fsrup` (AMD FSR), `xess` (Intel XeSS), `rtxvideo` (RTX VSR, RTX HDR), `nvoffruc` (Smooth Motion), `update` (the updater) |
| `engine/rife`, `engine/GMFSS_Fortuna`, `engine/realesrgan`, `engine/artcnn` | the models' weights as their authors publish them, read in place |

### The native host is one translation unit

`smv-live.cpp` includes, in this order, `smv-live-host.inl` (window, D3D12, swap chain, Streamline),
`smv-live-capture.inl` (window capture), `smv-live-native.inl` (every model and pass, CUDA), `smv-live-pipe.inl`,
`smv-live-loop.inl` (the live loop, pacing, the HUD). A name defined in an earlier file is visible in the later ones.

`smv-live-native.inl` (about 5,200 lines) keeps `kNativeKernels`, the shared CUDA kernel text (packs, resizes,
splats, blends, sharpen, HDR, optical flow; find a kernel with a search, then read around it), the CUDA runtime's
entry points and loader (`nativeLoadDlls`), the glue between parts (`nativeGmShareMemory`, `nativeModelShareMemory`,
the CUDA graphs `chainGraphRun`), and an `#include` line for each part below. All other native code lives in a part,
`smv-live-native-<name>.inl`, one concern each and small enough to read whole, except `anime4k` (weight tables). The
table is the include order, which is the source order and matters: a later part uses names an earlier one defines.

| Part `<name>` | What |
|---|---|
| `rtxvideo` | the RTX Video SDK bridge (RTX VSR, TrueHDR) and TrueHDR's statistics |
| `state` | `NativeRife` (a session's whole state) and the constants it is sized by |
| `handoff` | the resident host's kept state, the session lookup (`lkSession`), the live working size |
| `weights` | the weight readers: `.pkl` / `.pth` (`pthStateDict`), ONNX, AnimeJaNai's weights |
| `engine-load` | the full handoff (`nativeHandoff`), the offline lookups, the background build |
| `kernel-build` | RIFE's encode weights and kernel text, the kernel bindings, the resize tap tables, compiling and caching the cubins (`nativeCubin`, `nativeBuildKernels`) |
| `cuda-init` | the CUDA device init, the RTX bridge's session init (`nativeRtxInit`) |
| `nvof` | the nvof model (NVIDIA Optical Flow) |
| `fruc` | Smooth Motion (FRUC): its bridge, workers and tween tree |
| `fsrfg` | AMD FSR 4 / Intel XeSS frame generation |
| `fsrup` | AMD FSR 4 / Intel XeSS-SR upscaling |
| `anime4k`, `artcnn`, `anime4k-restore` | Anime4K: weight tables, kernel-text builder, upscale passes; ArtCNN (`kAcnA`); Anime4K's Restore pass |
| `model-setup` | DRBA's buffers, the early CUDA init, the HDR config, Restore's own nets (`restNatSetup`) |
| `gmfss-featnet`, `drba-block0`, `gmfss-metricnet` | GMFSS's feature net (`kFnConv`), DRBA's block 0 (`kB0Conv`), GMFSS's metric net (`kMnText`) |
| `rife-ifnet` | RIFE's IFNet (`kIfnA`) |
| `gmfss-ifnet`, `gmfss-fusionnet`, `gmflow-b`, `gmflow-backbone`, `gmflow-a` | GMFSS's IFNet (`kGiA`), fusion net (`kFuA`), GMFlow's second half (`kGb`), backbone (`kBbA`), transformer (`kGaA`) |
| `model-init` | GMFSS's setup, the tween buffers, the HDR model steps, the fits and blends, `nativeModelInit`, `nativeFree` |
| `gmfss-chain` | the GMFSS pair and tween |
| `drba` | DRBA's frame ring, windows and tween |
| `restore-run` | the Restore run, TrueHDR's offline statistics |
| `dlss5` | DLSS 5's change mask (`--nr-delta`), texture imports and motion field |
| `video-memory` | the free video memory figure and the live ring's fit |
| `dlss5-frame` | DLSS 5 on one frame (`nativeNrFrame`), its live init |
| `emit` | after DLSS 5: Sharpen and RTX HDR, RTX VSR, the offline stage and emit |
| `group` | `nativeGroup`: one group of tweens |
| `thread` | the live switches (`nativeLiveSets`), the compute thread, `nativeStart` / `nativeStop` |

CUDA kernels are C++ raw strings compiled at run time; the compiled result is cached by a hash of the text, so a
kernel text that does not change keeps its cache entry.

### The window's logic is one script in parts

`scripts/build-renderer.ts` strips the types from `renderer/app.ts` and its parts and joins them, in the order of the
`files` list in `renderer/tsconfig.json`, into the one `renderer/app.js` that `index.html` loads. They share one
scope: no imports, a name declared in one part is visible in all. The table is that order, and it matters: a
top-level `const` or `let` exists only once its line has run. A new part is a new line in that `files` list.

| File | What |
|---|---|
| `app.ts` | start-up, UI zoom, the shared state (the loaded video, the batch), status and ETA helpers, `$` |
| `app-i18n.ts` | UI texts: loading a language, `tr` / `txt`, `applyI18n`, the language picker |
| `app-video.ts` | the Video / Live switch, speed and target fps, the output name, the resume check, loading and queueing videos |
| `app-effects.ts` | Sharpen, Restore (AnimeJaNai setup), the panel order, the screen rate, Upscale to, the DLSS mode, the codec |
| `app-upscalers.ts` | RTX Video setup, the upscaler ticks (FSR 4, Anime4K, ArtCNN, XeSS), RTX VSR quality |
| `app-dlss5.ts` | DLSS 5: its controls and setup |
| `app-models.ts` | the Interpolate models, Smooth Motion and XeSS-FG setup |
| `app-live.ts` | Live mode: its controls, start and stop, the host's lines |
| `app-hdr-tools.ts` | the Dolby Vision and HDR10+ export tools |
| `app-preview.ts` | the before / after preview, the RTX HDR colour controls |
| `app-compare.ts` | the model comparison view |
| `app-run.ts` | drag and drop, starting a render, the engine's progress lines |
| `app-update.ts` | the update notice and the automatic update |
| `app-done.ts` | a render's end, the batch's next video |

## Never read or search these

`node_modules/`, `dist/`, `release/`, `renderer/app.js`, `engine/runtime/` (git-ignored local tooling, thousands
of files), `engine/bin/`, `engine/gpu_runtime/`, `engine/live/build_src/cuda/`, `model_cache_safe_to_delete/`, any `.exe`, `.dll`, `.pkl`, `.pth`, and the
vendored trees `engine/GMFSS_Fortuna/`, `engine/rife/`, `engine/live/build_src/nvofa/`. Scope every search to the
source paths in the table above; never search from the repo root.

## Rules a change must keep

* Edit sources, never builds: `.ts` not `dist/*.js`, `app*.ts` not `app.js`.
* A new UI text is a key in ALL eleven `renderer/i18n` files. A line written at run time goes through
  `txt(id, build)` (`app-i18n.ts`), never a plain string. CSS uses logical properties only (`margin-inline-start`, not
  `margin-left`), because Arabic mirrors the window.
* A change to native source is complete only with its binary rebuilt by the `build.bat` beside it and committed
  together; the build prints no warnings, and a new one is fixed, not suppressed.
* Native C++ holds no raw non-ASCII character: write `\u` escapes.
* A new, renamed or removed `SMV_*` environment lever changes its row in DEVELOPMENT.md's "Environment variables"
  table in the same change: that table is the one list of the levers, and the lint fails when it is not complete.
* CUDA kernels use no 64-bit floats.
* A speed change must not change the picture. A model is called the way its author calls it. Nothing that changes
  the look runs unless the user ticked it.
* Comments and these docs say what the code does and why. No history: no dates, no task names, no "used to".
* Prettier never touches `engine/`, `renderer/index.html` or `renderer/app.js` (`.prettierignore`).
* Git: the maintainer commits and pushes. An agent stages only work that passed its checks and never commits,
  pushes or reverts.
