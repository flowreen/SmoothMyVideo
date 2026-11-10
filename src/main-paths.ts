// Main process: where the app's files are (the engine folder, the render and preview CLIs, ffprobe).
import { app } from 'electron';
import * as path from 'path';
import * as fs from 'fs';

export const ROOT = path.join(__dirname, '..');
// When packaged, the engine ships as an unpacked extraResource (the host exe, its DLLs and
// the model weights must be real files on disk, not inside app.asar). The renderer and icon stay
// under ROOT (Electron reads those from the asar fine).
export const ENGINE = app.isPackaged ? path.join(process.resourcesPath, 'engine') : path.join(ROOT, 'engine');
// The render orchestrator and the before/after preview, compiled next to this file.
export const RENDER_CLI = path.join(__dirname, 'render', 'cli.js');
export const PREVIEW_CLI = path.join(__dirname, 'render', 'preview.js');
// Prefer ffprobe bundled at engine/bin (portable build); fall back to PATH for dev.
export const FFPROBE = fs.existsSync(path.join(ENGINE, 'bin', 'ffprobe.exe'))
  ? path.join(ENGINE, 'bin', 'ffprobe.exe')
  : 'ffprobe';
// The real cold-start when a video is selected is the before/after PREVIEW: any enabled pass starts the
// native host, which creates a CUDA context. We deliberately do NOT warm that at launch: on hybrid-GPU
// laptops the CUDA-context creation saturates the RTX and stalls Chromium's GPU compositor for several
// seconds, so a user who selects a file during that window sees the (instantly-probed, ~100ms) file
// info fail to paint until the warmup finishes - the exact "video info loads slowly" symptom. The first
// preview instead pays its own cold start behind its own spinner, which never blocks the file info. The
// preview spawn is also fenced behind a composited frame in the renderer (see loadVideo) so the info is
// always on screen before the host starts.
