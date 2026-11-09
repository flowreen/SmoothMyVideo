# SmoothMyVideo: map for coding agents

Electron + TypeScript GUI over one native host, `engine/live/smv-live.exe`, which runs both file renders and Live
mode: AI frame interpolation, upscaling and HDR for NVIDIA GPUs, fully offline. Users read `README.md`, developers
read `docs/DEVELOPMENT.md` (build recipes, CLI reference, every `SMV_*` lever); this file says where things are and
which rules a change must keep. These three are the only docs: add to them, never create another.

## Commands

| Command | What it does |
|---|---|
| `npm start` | `tsc`, the renderer's type check, the renderer build (`app.ts` -> `app.js`), then the app |
| `npm run lint` | prettier (writes) on `src/**/*.ts`, `scripts/*.ts`, `renderer/app.ts`, `renderer/i18n/*.json`; `tsc --noEmit`; `tsc -p scripts`; `tsc -p renderer`; clang-format (writes) on every `engine/*/build_src` |
| `node scripts/scan-index.ts` | after ANY edit of `renderer/index.html` or `renderer/app.ts`: builds `app.js`, checks duplicate ids and `$()` lookups |
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
| `renderer/app.ts` | the window's logic (progress, IPC, settings). `renderer/app.js` is its build: never edit or read it |
| `renderer/i18n/<code>.json` | every UI text, eleven languages (`en` is the source; `ar` is right-to-left) |
| `scripts/*.ts` | build and setup scripts; Node runs them directly, no compile step |
| `engine/live/build_src/` | the native host's source; `build.bat` there builds `engine/live/smv-live.exe` |
| `engine/<name>/build_src/` | one bridge each, built into the DLL or exe beside it: `dlssg` (DLSS frame generation helper), `dlssnr` (DLSS 5), `fsrfg` + `fsrup` (AMD FSR), `xess` (Intel XeSS), `rtxvideo` (RTX VSR, RTX HDR), `nvoffruc` (Smooth Motion), `update` (the updater) |
| `engine/rife`, `engine/GMFSS_Fortuna`, `engine/realesrgan`, `engine/artcnn` | the models' weights as their authors publish them, read in place |

### The native host is one translation unit

`smv-live.cpp` includes, in this order, `smv-live-host.inl` (window, D3D12, swap chain, Streamline),
`smv-live-capture.inl` (window capture), `smv-live-native.inl` (every model and pass, CUDA), `smv-live-pipe.inl`,
`smv-live-loop.inl` (the live loop, pacing, the HUD). A name defined in an earlier file is visible in the later ones.

`smv-live-native.inl` is far too long to read whole (over 30,000 lines). Find a symbol with a search, then read around
it. Its layout, top to bottom:

1. `kNativeKernels`: the shared CUDA kernel text (packs, resizes, splats, blends, sharpen, HDR, optical flow).
2. Host code: resize tables, the vendor bridges, the weight readers, kernel compilation and its cache.
3. Anime4K: its weight tables and kernel-text builder.
4. One run per network, each with its kernel text, loader and launch code: ArtCNN (`kAcnA`), GMFSS's feature net
   (`kFnConv`), DRBA's block 0 (`kB0Conv`), GMFSS's metric net (`kMnText`), RIFE's IFNet (`kIfnA`), GMFSS's IFNet
   (`kGiA`), GMFSS's fusion net (`kFuA`), GMFlow's second half (`kGb`), its backbone (`kBbA`) and its transformer
   (`kGaA`).
5. Session code: the tween and emit paths, `nativeGroup`, `nativeThread`.

CUDA kernels are C++ raw strings compiled at run time; the compiled result is cached by a hash of the text, so a
kernel text that does not change keeps its cache entry.

## Never read or search these

`node_modules/`, `dist/`, `release/`, `renderer/app.js`, `engine/runtime/` (git-ignored local tooling, thousands
of files), `engine/bin/`, `engine/gpu_runtime/`, `engine/live/build_src/cuda/`, `model_cache_safe_to_delete/`, any `.exe`, `.dll`, `.pkl`, `.pth`, and the
vendored trees `engine/GMFSS_Fortuna/`, `engine/rife/`, `engine/live/build_src/nvofa/`. Scope every search to the
source paths in the table above; never search from the repo root.

## Rules a change must keep

* Edit sources, never builds: `.ts` not `dist/*.js`, `app.ts` not `app.js`.
* A new UI text is a key in ALL eleven `renderer/i18n` files. A line written at run time goes through
  `txt(id, build)` in `app.ts`, never a plain string. CSS uses logical properties only (`margin-inline-start`, not
  `margin-left`), because Arabic mirrors the window.
* A change to native source is complete only with its binary rebuilt by the `build.bat` beside it and committed
  together; the build prints no warnings, and a new one is fixed, not suppressed.
* Native C++ holds no raw non-ASCII character: write `\u` escapes.
* CUDA kernels use no 64-bit floats.
* A speed change must not change the picture. A model is called the way its author calls it. Nothing that changes
  the look runs unless the user ticked it.
* Comments and these docs say what the code does and why. No history: no dates, no task names, no "used to".
* Prettier never touches `engine/`, `renderer/index.html` or `renderer/app.js` (`.prettierignore`).
* Git: the maintainer commits and pushes. An agent stages only work that passed its checks, and never commits,
  pushes, stashes or reverts.
