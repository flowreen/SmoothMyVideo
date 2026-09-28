import {
  app,
  BrowserWindow,
  ipcMain,
  dialog,
  screen,
  shell,
  powerSaveBlocker,
  Notification,
  globalShortcut,
} from 'electron';
import { spawn, execFile, execFileSync, ChildProcess } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import * as crypto from 'crypto';
import { Readable } from 'stream';
import { pipeline } from 'stream/promises';
import { checkEngineCache } from './render/cache';

const ROOT = path.join(__dirname, '..');
// When packaged, the engine ships as an unpacked extraResource (the host exe, its DLLs and
// the ONNX must be real files on disk, not inside app.asar). The renderer and icon stay
// under ROOT (Electron reads those from the asar fine).
const ENGINE = app.isPackaged ? path.join(process.resourcesPath, 'engine') : path.join(ROOT, 'engine');
// The render orchestrator and the before/after preview, compiled next to this file.
const RENDER_CLI = path.join(__dirname, 'render', 'cli.js');
const PREVIEW_CLI = path.join(__dirname, 'render', 'preview.js');
// Prefer ffprobe bundled at engine/bin (portable build); fall back to PATH for dev.
const FFPROBE = fs.existsSync(path.join(ENGINE, 'bin', 'ffprobe.exe'))
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

let win: BrowserWindow | null = null;

// Remember the window size between sessions (size only, NOT position, so a disconnected monitor can't
// leave it opening off-screen). getNormalBounds() ignores a maximized/minimized state, so we store a
// real restorable size. Falls back to a 900x1000 default (portrait: the UI is one tall column).
const windowStateFile = () => path.join(app.getPath('userData'), 'window-state.json');
function loadWindowSize(): { width: number; height: number; maximized: boolean } {
  try {
    const s = JSON.parse(fs.readFileSync(windowStateFile(), 'utf8'));
    if (typeof s.width === 'number' && typeof s.height === 'number')
      return { width: s.width, height: s.height, maximized: !!s.maximized };
  } catch {
    /* no saved state yet */
  }
  return { width: 900, height: 1000, maximized: false };
}
function saveWindowSize(): void {
  if (!win) return;
  try {
    const b = win.getNormalBounds();
    fs.writeFileSync(
      windowStateFile(),
      JSON.stringify({ width: b.width, height: b.height, maximized: win.isMaximized() }),
    );
  } catch {
    /* best effort */
  }
}

function createWindow() {
  const st = loadWindowSize();
  win = new BrowserWindow({
    width: st.width,
    height: st.height,
    minWidth: 680,
    minHeight: 640,
    title: 'Smooth My Video',
    backgroundColor: '#1b1b1b',
    icon: path.join(ROOT, 'icon.ico'),
    webPreferences: { nodeIntegration: true, contextIsolation: false },
  });
  if (st.maximized) win.maximize();
  win.setMenuBarVisibility(false);
  win.on('close', saveWindowSize); // persist size on close so the next launch reopens at the same size
  win.loadFile(path.join(ROOT, 'renderer', 'index.html'));
  // Guard against an accidental reload (Ctrl/Cmd+R, F5) WHILE the engine is running/paused: a reload
  // wipes the renderer's job state but leaves the engine alive in this process, and the next run spawns
  // a rival engine (both stream PROGRESS -> the bar ping-pongs 1%->15%->1%). Swallow the reload keys
  // while a job is live so the render is preserved; the 'renderer-ready' handler is the fallback for
  // any reload that arrives by another route (menu, devtools, programmatic).
  win.webContents.on('before-input-event', (event, input) => {
    if (input.type !== 'keyDown') return;
    const key = (input.key || '').toLowerCase();
    if (current && ((key === 'r' && (input.control || input.meta)) || key === 'f5')) event.preventDefault();
  });
}

// Windows keys the taskbar icon, grouping and notification identity to the AppUserModelID, NOT the window
// icon. Without this the app inherits electron.exe's identity and shows the Electron logo in the taskbar
// (and notifications read "electron.app..."). Match the packaged appId so dev and zip builds agree.
app.setAppUserModelId('com.smoothmyvideo.app');

// Single instance only. A second launch shares this profile dir, and Chromium's disk/GPU cache and
// Local Storage locks (held by the first instance) make the second window come up empty ("Unable to
// move the cache: Access is denied", "Gpu Cache Creation failed"); it would also fight the first
// instance over the preview PNGs and the TRT cache. So a second launch focuses the running window.
if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (win) {
      if (win.isMinimized()) win.restore();
      win.show();
      win.focus();
    }
  });
  app.whenReady().then(() => {
    // a new TensorRT-RTX or ONNX graph revision empties the engine cache once, before any live
    // session or render can load an engine built from the old graph (src/render/cache.ts)
    console.log(checkEngineCache(ENGINE));
    createWindow();
    checkForUpdate();
  });
}
app.on('window-all-closed', () => {
  stopLive(true); // never leave a headless FG overlay (or an idle resident host) running after the GUI is gone
  offlineHostQuit();
  if (process.platform !== 'darwin') app.quit();
});
app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});

// --- Update check (best-effort, silent unless a newer release exists) ------------------------------
// The app ships as a plain zip with no installer or auto-update channel, so users have no signal a
// new build exists. Poke the GitHub releases API once per launch (deferred a few seconds so it never
// competes with the preview-engine warmup) and tell the renderer when a newer tag is out - it shows a
// one-line link, nothing more. Silent on ANY failure (offline, repo private, no releases yet, rate
// limit); nothing is ever downloaded.
const UPDATE_REPO = 'flowreen/SmoothMyVideo';
function newerVersion(a: string, b: string): boolean {
  // true when dotted-numeric a > b (non-numeric parts count as 0; length-agnostic)
  const pa = a.split('.').map((x) => parseInt(x, 10) || 0);
  const pb = b.split('.').map((x) => parseInt(x, 10) || 0);
  for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
    if ((pa[i] || 0) !== (pb[i] || 0)) return (pa[i] || 0) > (pb[i] || 0);
  }
  return false;
}
function checkForUpdate(): void {
  setTimeout(async () => {
    try {
      const res = await fetch(`https://api.github.com/repos/${UPDATE_REPO}/releases/latest`, {
        headers: { accept: 'application/vnd.github+json', 'user-agent': 'SmoothMyVideo' },
      });
      if (!res.ok) return;
      const rel = (await res.json()) as { tag_name?: string; html_url?: string };
      const latest = String(rel.tag_name || '').replace(/^v/i, '');
      if (!latest || !newerVersion(latest, app.getVersion())) return;
      const url = rel.html_url || `https://github.com/${UPDATE_REPO}/releases/`;
      try {
        if (win && !win.webContents.isDestroyed()) win.webContents.send('update-available', { version: latest, url });
      } catch {
        /* window gone */
      }
    } catch {
      /* offline / API unreachable: stay silent */
    }
  }, 5000);
}

ipcMain.handle('pick-video', async (_e, defaultPath?: string) => {
  // Multi-select: several files become a batch queue in the renderer (processed back to back
  // with the same settings); a single selection behaves as before.
  const r = await dialog.showOpenDialog(win!, {
    defaultPath,
    properties: ['openFile', 'multiSelections'],
    filters: [{ name: 'Video', extensions: ['mp4', 'mkv', 'mov', 'avi', 'webm', 'm4v', 'wmv', 'ts'] }],
  });
  return r.canceled ? null : r.filePaths;
});

ipcMain.handle('pick-output', async (_e, defaultPath?: string) => {
  // One combined filter so the dialog keeps whichever extension the default name carries
  // (.mkv when the passthrough tracks need it, .mp4 otherwise).
  const r = await dialog.showSaveDialog(win!, {
    defaultPath,
    filters: [{ name: 'Video', extensions: ['mp4', 'mkv'] }],
  });
  return r.canceled ? null : r.filePath || null;
});

ipcMain.handle('probe', async (_e, file: string) => {
  // All streams (not just v:0): the renderer needs the audio/subtitle track list to decide
  // whether the passthrough output must be .mkv (subtitles / mp4-incompatible audio).
  return new Promise((resolve) => {
    execFile(
      FFPROBE,
      [
        '-v',
        'error',
        '-show_entries',
        'stream=codec_type,width,height,r_frame_rate,codec_name,nb_frames,color_transfer',
        '-show_entries',
        'format=duration',
        '-of',
        'json',
        file,
      ],
      (err, stdout) => {
        if (err) {
          resolve({ error: String(err) });
          return;
        }
        try {
          resolve(JSON.parse(stdout));
        } catch {
          resolve({ error: 'probe parse failed' });
        }
      },
    );
  });
});

// TRUE refresh rate of the monitor the app window is on (fallback: primary), kept FRACTIONAL so a
// 59.94 / 359.98 Hz panel feeds the renderer its exact rate - drives "match screen" and its decimal
// precision (see decimalsOf/targetDecimals in the renderer). Electron's display.displayFrequency is
// integer-only on Windows, so the exact rational comes from QueryDisplayConfig via the bundled
// engine/exact_hz.ps1 helper (one "numerator denominator" line per active display path), matched to
// the Electron display by nearest integer. Cached: the helper costs a PowerShell spawn (~0.5 s) and
// rates only change on display mode switches.
let exactHz: number[] = [];
let exactHzAt = 0;
function queryExactRates(): Promise<number[]> {
  return new Promise((resolve) => {
    execFile(
      'powershell.exe',
      ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', path.join(ENGINE, 'exact_hz.ps1')],
      { timeout: 8000 },
      (err, stdout) => {
        if (err) return resolve([]);
        resolve(
          String(stdout)
            .split(/\r?\n/)
            .map((l) => {
              const m = l.trim().match(/^(\d+) (\d+)$/);
              return m ? Number(m[1]) / Number(m[2]) : 0;
            })
            .filter((v) => v > 1),
        );
      },
    );
  });
}
ipcMain.handle('refresh-rate', async () => {
  try {
    const d = win ? screen.getDisplayMatching(win.getBounds()) : screen.getPrimaryDisplay();
    const hz = d.displayFrequency || screen.getPrimaryDisplay().displayFrequency || 60;
    if (Date.now() - exactHzAt > 60000) {
      exactHz = await queryExactRates();
      exactHzAt = Date.now();
    }
    const exact = exactHz.find((r) => Math.abs(r - hz) < 1);
    return Math.round((exact || hz) * 1000) / 1000;
  } catch {
    return 60;
  }
});

ipcMain.handle('screen-size', () => {
  // Physical pixel resolution of the monitor the window is on (fallback: primary). display.size is
  // in logical DIPs, so multiply by the scale factor to get the real panel resolution. Feeds the
  // renderer's "RTX Video Super Resolution: upscale to screen" target.
  try {
    const d = win ? screen.getDisplayMatching(win.getBounds()) : screen.getPrimaryDisplay();
    const f = d.scaleFactor || 1;
    return { width: Math.round(d.size.width * f), height: Math.round(d.size.height * f) };
  } catch {
    return { width: 0, height: 0 };
  }
});

// --- RTX Video runtime: readiness + one-click install --------------------------------------------
// The opt-in RTX features (VSR / TrueHDR) run through the compiled CUDA bridge (rtxvideo_cuda.dll,
// which ships with the app) plus NVIDIA's two RTX Video feature DLLs. Those feature DLLs are NVIDIA
// proprietary and NON-redistributable, so they are never bundled; the user downloads NVIDIA's RTX
// Video SDK (a deliberate, EULA-gated action) and this app drops the two DLLs into engine/rtxvideo
// for them. A feature is "ready" only when the bridge AND its feature DLL are present there.
const RTX_DIR = path.join(ENGINE, 'rtxvideo');
const RTX_FEATURE_DLLS = ['nvngx_vsr.dll', 'nvngx_truehdr.dll'];
const RTX_SDK_URL = 'https://developer.nvidia.com/rtx-video-sdk/getting-started';
// NvOFFRUC ("NVIDIA Smooth Motion"): our bridge (nvoffruc_bridge.dll) is locally built and ships,
// but NvOFFRUC.dll + cudart64_110.dll are NVIDIA proprietary and user-installed from the Optical
// Flow SDK .zip - same EULA-gated, non-redistributable pattern as the RTX feature DLLs.
const NVOFFRUC_DIR = path.join(ENGINE, 'nvoffruc');
const NVOFFRUC_DLLS = ['NvOFFRUC.dll', 'cudart64_110.dll'];
const OF_SDK_URL = 'https://developer.nvidia.com/opticalflow/download';
// "DLSS 4.5" (DLSS Frame Generation): fully bundled, no installer - the offline host exe
// (engine/dlssg/dlssg2f.exe, built from build_src) plus NVIDIA's redistributable Streamline
// runtime (~10 MB, licenses alongside). Readiness = all files present (False means a broken
// install); actual GPU/driver support surfaces as a clear engine error at render time.
const DLSSG_DIR = path.join(ENGINE, 'dlssg');
const DLSSG_FILES = [
  'dlssg2f.exe',
  'sl.interposer.dll',
  'sl.common.dll',
  'sl.dlss_g.dll',
  'sl.pcl.dll',
  'sl.reflex.dll',
  'nvngx_dlssg.dll',
];
// "NVIDIA DLSS 5" (Neural Rendering): the host (engine/dlssnr/dlssnr.exe + its nvngx.dll caller
// shim, built from build_src) ships, but the NR runtime nvngx_dlssnr.dll is NOT shipped: NVIDIA
// publishes no download (no SDK, no driver copy, no NVIDIA App override as of 2026-09), the only
// NVIDIA copy sits inside NBA 2K27. So the app offers a one-click download of the community build
// every DLSS 5 tool pulls from (RankFTW/rhi-repo release assets, the RenoDX author's RTX 40 + 50
// rebuild), pinned to one asset and verified twice (zip SHA256 from the GitHub API digest, then the
// DLL inside), or the user drops a copy in, like the RTX Video and NvOFFRUC DLLs. nvngx_dlss.dll is
// the DLSS SR runtime; the NGX core only warns when it is absent (probe-verified), so it
// is copied when found beside a dropped runtime, reported, never required.
const DLSSNR_DIR = path.join(ENGINE, 'dlssnr');
const DLSSNR_HOST = ['dlssnr.exe', 'nvngx.dll'];
const DLSSNR_RUNTIME = 'nvngx_dlssnr.dll';
const DLSSNR_SR = 'nvngx_dlss.dll';
// The one-click source: the most-downloaded rhi-repo asset, and the ONE hash this app holds: the
// SHA256 of the nvngx_dlssnr.dll inside it (probe-verified on the host). The download refuses to install any other DLL; a dropped file with another hash still
// installs, the UI just says it is unverified. Bump tag, asset and hash together when moving to a
// newer build (the zip's own digest is at https://api.github.com/repos/RankFTW/rhi-repo/releases).
const DLSSNR_DL = {
  url: 'https://github.com/RankFTW/rhi-repo/releases/download/dlssnr-310.8.SF-v2/nvngx_dlssnr_310.8.SF-v2.zip',
  page: 'https://github.com/RankFTW/rhi-repo/releases/tag/dlssnr-310.8.SF-v2',
  dllSha256: '6eb209e764f39872625debd6abaf45e2bb6322f6f270f781f70c059ae30b3927',
  label: '310.8.SF-v2 (rhi-repo, RTX 40 + 50)',
  zipBytes: 116693212,
};
function sha256File(p: string): string {
  const h = crypto.createHash('sha256');
  const fd = fs.openSync(p, 'r');
  try {
    const buf = Buffer.alloc(1 << 20);
    for (;;) {
      const n = fs.readSync(fd, buf, 0, buf.length, null);
      if (n <= 0) break;
      h.update(buf.subarray(0, n));
    }
  } finally {
    fs.closeSync(fd);
  }
  return h.digest('hex');
}
const SYS_TAR = path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'tar.exe');
const fileExists = (p: string) => {
  try {
    return fs.existsSync(p);
  } catch {
    return false;
  }
};

// A directory "has the runtime" when both feature DLLs sit in it. The SDK keeps them under
// bin/Windows/x64/rel, so probe the dir itself, that subpath, and one level of child dirs.
function findFeatureDllDir(root: string): string | null {
  if (!root) return null;
  const rel = path.join('bin', 'Windows', 'x64', 'rel');
  const hasBoth = (d: string) => RTX_FEATURE_DLLS.every((n) => fileExists(path.join(d, n)));
  const seeds = [root, path.join(root, rel)];
  try {
    for (const e of fs.readdirSync(root, { withFileTypes: true }))
      if (e.isDirectory()) seeds.push(path.join(root, e.name), path.join(root, e.name, rel));
  } catch {
    /* unreadable root */
  }
  return seeds.find(hasBoth) || null;
}

// Look in the usual download spots for an extracted SDK folder or a recognizable SDK .zip.
function scanForSdk(): { folder: string | null; zip: string | null } {
  const roots: string[] = [];
  for (const k of ['downloads', 'desktop', 'home'] as const) {
    try {
      roots.push(app.getPath(k));
    } catch {
      /* none */
    }
  }
  let folder: string | null = null;
  for (const r of roots) {
    folder = findFeatureDllDir(r);
    if (folder) break;
  }
  let zip: string | null = null;
  for (const r of roots) {
    try {
      const hit = fs
        .readdirSync(r, { withFileTypes: true })
        .find((e) => e.isFile() && /\.zip$/i.test(e.name) && /rtx.*video.*sdk/i.test(e.name));
      if (hit) {
        zip = path.join(r, hit.name);
        break;
      }
    } catch {
      /* unreadable root */
    }
  }
  return { folder, zip };
}

type InstallResult = { ok: boolean; error?: string; copied: string[] };

// Every file under root whose name matches one of `names` (case-insensitive; a release zip can
// nest its payload, and the __MACOSX ._ copies fall out because their basename differs).
function findFilesNamed(root: string, names: string[]): string[] {
  const want = new Set(names.map((n) => n.toLowerCase()));
  const hits: string[] = [];
  const stack = [root];
  while (stack.length) {
    const d = stack.pop()!;
    let ents: fs.Dirent[];
    try {
      ents = fs.readdirSync(d, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const e of ents) {
      const p = path.join(d, e.name);
      if (e.isDirectory()) stack.push(p);
      else if (want.has(e.name.toLowerCase())) hits.push(p);
    }
  }
  return hits;
}

// The one zip-extract-and-copy flow behind every user-installed runtime (the RTX feature DLLs,
// the NvOFFRUC pair, dovi_tool / hdr10plus_tool, the DLSS 5 runtime). `source` is a .zip (only
// the wanted members are extracted, with Windows' bundled bsdtar, into a temp dir that is removed
// again), an extracted folder, or a picked file (its folder is searched). `prefer` breaks ties
// when an archive ships several copies of a name (the RTX SDK's arm64 / x64 dev / rel builds, the
// Optical Flow SDK's win32 / win64). Every wanted name must be found, else nothing is copied.
function installFiles(
  source: string,
  names: string[],
  destDir: string,
  opts: { what: string; where: string; prefer?: RegExp },
): InstallResult {
  try {
    fs.mkdirSync(destDir, { recursive: true });
  } catch {
    /* exists */
  }
  let tmp: string | null = null;
  const srcFiles: string[] = [];
  try {
    let searchDir = source;
    if (/\.zip$/i.test(source)) {
      tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'smv-install-'));
      try {
        execFileSync(SYS_TAR, ['-xf', source, '-C', tmp, ...names.map((n) => '*' + n)]);
      } catch {
        // bsdtar's member patterns are case-sensitive and fail when one matches nothing (a
        // DOVI_TOOL.EXE member, a renamed SDK layout): take the whole archive and search it
        execFileSync(SYS_TAR, ['-xf', source, '-C', tmp]);
      }
      searchDir = tmp;
    } else if (!fs.statSync(source).isDirectory()) {
      searchDir = path.dirname(source);
    }
    const found = findFilesNamed(searchDir, names);
    for (const n of names) {
      const all = found.filter((f) => path.basename(f).toLowerCase() === n.toLowerCase());
      const hit = (opts.prefer && all.find((f) => opts.prefer!.test(f))) || all[0];
      if (hit) srcFiles.push(hit);
    }
    if (srcFiles.length < names.length)
      return { ok: false, error: names.join(' / ') + ' not found in ' + opts.what, copied: [] };
    const copied: string[] = [];
    try {
      for (const f of srcFiles) {
        const dest = path.join(destDir, path.basename(f));
        // Overwrite any existing copy (a newer release replaces the old files), even one a previous
        // extraction left read-only; clearing the flag first avoids an EPERM on copy.
        try {
          if (fileExists(dest)) fs.chmodSync(dest, 0o666);
        } catch {
          /* best effort */
        }
        fs.copyFileSync(f, dest);
        copied.push(path.basename(f));
      }
    } catch (e) {
      return {
        ok: false,
        error: 'Could not write ' + opts.where + ' (' + String(e) + '). If a render is running, stop it and try again.',
        copied,
      };
    }
    return { ok: true, copied };
  } catch (e) {
    return { ok: false, error: String(e), copied: [] };
  } finally {
    if (tmp)
      try {
        fs.rmSync(tmp, { recursive: true, force: true });
      } catch {
        /* temp cleanup best effort */
      }
  }
}

// The two RTX Video feature DLLs into engine/rtxvideo (the SDK ships arm64 + x64 dev/rel copies;
// take the x64 release build).
const installRtx = (source: string): InstallResult =>
  installFiles(source, RTX_FEATURE_DLLS, RTX_DIR, {
    what: 'the selected RTX Video SDK',
    where: 'the RTX DLLs in engine/rtxvideo',
    prefer: /x64[\\/]+rel/i,
  });

// NvOFFRUC.dll + cudart64_110.dll into engine/nvoffruc, beside the locally built bridge (the SDK
// ships win32 + win64 copies; take the x64 build).
const installFruc = (source: string): InstallResult =>
  installFiles(source, NVOFFRUC_DLLS, NVOFFRUC_DIR, {
    what: 'the selected Optical Flow SDK',
    where: 'the FRUC DLLs in engine/nvoffruc',
    prefer: /win64/i,
  });

ipcMain.handle('rtx-ready', () => {
  const bridge = fileExists(path.join(RTX_DIR, 'rtxvideo_cuda.dll'));
  return {
    vsr: bridge && fileExists(path.join(RTX_DIR, 'nvngx_vsr.dll')),
    hdr: bridge && fileExists(path.join(RTX_DIR, 'nvngx_truehdr.dll')),
    bridge,
    dir: RTX_DIR,
  };
});

ipcMain.handle('rtx-open-download', () => {
  shell.openExternal(RTX_SDK_URL);
  return true;
});

// Install from a given source (the picked .zip), or auto-detect one when none is passed.
ipcMain.handle('rtx-install', (_e, source?: string) => {
  let src = source;
  if (!src) {
    const s = scanForSdk();
    src = s.folder || s.zip || undefined;
  }
  if (!src)
    return {
      ok: false,
      error: 'No RTX Video SDK found in Downloads/Desktop. Use "Get from NVIDIA", then "Choose..."',
      copied: [],
    };
  return installRtx(src);
});

// Manual picker fallback: a folder (extracted SDK) or a .zip.
ipcMain.handle('rtx-choose', async (_e, mode: 'dir' | 'zip') => {
  const r = await dialog.showOpenDialog(
    win!,
    mode === 'dir'
      ? { title: 'Select the extracted RTX Video SDK folder', properties: ['openDirectory'] }
      : {
          title: 'Select the RTX Video SDK .zip',
          properties: ['openFile'],
          filters: [{ name: 'Zip', extensions: ['zip'] }],
        },
  );
  return r.canceled ? null : r.filePaths[0] || null;
});

// "NVIDIA Smooth Motion" (NvOFFRUC) runtime: ready only when our bridge AND NvOFFRUC.dll are present.
// "DLSS 4.5" (DLSS Frame Generation): bundled, so this only guards against a broken install.
ipcMain.handle('dlssg-ready', () => {
  const missing = DLSSG_FILES.filter((f) => !fileExists(path.join(DLSSG_DIR, f)));
  return { ready: missing.length === 0, missing, dir: DLSSG_DIR };
});

// "NVIDIA DLSS 5" (Neural Rendering): ready when the shipped host files AND the user-supplied
// runtime are present in engine/dlssnr (see the DLSSNR_* constants).
ipcMain.handle('dlssnr-ready', () => {
  const host = DLSSNR_HOST.every((f) => fileExists(path.join(DLSSNR_DIR, f)));
  const runtime = fileExists(path.join(DLSSNR_DIR, DLSSNR_RUNTIME));
  const sr = fileExists(path.join(DLSSNR_DIR, DLSSNR_SR));
  return {
    ready: host && runtime,
    host,
    runtime,
    sr,
    dir: DLSSNR_DIR,
    file: DLSSNR_RUNTIME,
    srFile: DLSSNR_SR,
    download: { page: DLSSNR_DL.page, mb: Math.round(DLSSNR_DL.zipBytes / 1048576) },
  };
});

// Install the NR runtime from a picked/dropped nvngx_dlssnr.dll, a folder, or a .zip holding it
// (installBin searches the selection by name); nvngx_dlss.dll is copied too when it sits in the
// same selection, and its absence is not an error. The installed file is hashed so the UI can say
// whether it is the verified build.
ipcMain.handle('dlssnr-install', (_e, source: string) => {
  const r: { ok: boolean; error?: string; copied: string[]; sha256?: string; known?: string } = installBin(
    source,
    DLSSNR_RUNTIME,
    DLSSNR_DIR,
  );
  if (r.ok) {
    const sr = installBin(source, DLSSNR_SR, DLSSNR_DIR);
    if (sr.ok) r.copied.push(...sr.copied);
    try {
      r.sha256 = sha256File(path.join(DLSSNR_DIR, DLSSNR_RUNTIME));
      r.known = r.sha256 === DLSSNR_DL.dllSha256 ? DLSSNR_DL.label : undefined;
    } catch {
      /* hash is informational */
    }
  }
  return r;
});

// One-click: download the pinned rhi-repo zip into a temp dir (progress to the renderer as
// 'dlssnr-progress'), extract with bsdtar, verify the DLL hash, copy it into engine/dlssnr. A
// mismatch aborts before the copy, so a replaced release asset can never install anything.
ipcMain.handle('dlssnr-download', async () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'smv-dlssnr-'));
  const zip = path.join(tmp, 'runtime.zip');
  const progress = (received: number, total: number, stage: string) => {
    try {
      if (win && !win.webContents.isDestroyed()) win.webContents.send('dlssnr-progress', { received, total, stage });
    } catch {
      /* window gone */
    }
  };
  try {
    const res = await fetch(DLSSNR_DL.url, { headers: { 'user-agent': 'SmoothMyVideo' } });
    if (!res.ok || !res.body) return { ok: false, error: `download failed: HTTP ${res.status}`, copied: [] };
    const total = parseInt(res.headers.get('content-length') || '0', 10) || DLSSNR_DL.zipBytes;
    let received = 0;
    let last = 0;
    const src = Readable.fromWeb(res.body as never);
    src.on('data', (chunk: Buffer) => {
      received += chunk.length;
      if (received - last > 2 * 1048576 || received === total) {
        last = received;
        progress(received, total, 'download');
      }
    });
    await pipeline(src, fs.createWriteStream(zip));
    progress(received, total, 'verify');
    execFileSync(SYS_TAR, ['-xf', zip, '-C', tmp]);
    const found = findDvBin(tmp, DLSSNR_RUNTIME);
    if (!found)
      return { ok: false, error: DLSSNR_RUNTIME + ' not in the downloaded zip, nothing installed', copied: [] };
    const dh = sha256File(found);
    if (dh !== DLSSNR_DL.dllSha256)
      return { ok: false, error: `runtime checksum mismatch (${dh.slice(0, 12)}...), nothing installed`, copied: [] };
    fs.mkdirSync(DLSSNR_DIR, { recursive: true });
    const dest = path.join(DLSSNR_DIR, DLSSNR_RUNTIME);
    try {
      if (fileExists(dest)) fs.chmodSync(dest, 0o666);
    } catch {
      /* best effort */
    }
    fs.copyFileSync(found, dest);
    return { ok: true, copied: [DLSSNR_RUNTIME], sha256: dh, known: DLSSNR_DL.label };
  } catch (e) {
    return { ok: false, error: String(e), copied: [] };
  } finally {
    try {
      fs.rmSync(tmp, { recursive: true, force: true });
    } catch {
      /* temp cleanup best effort */
    }
  }
});

ipcMain.handle('dlssnr-open-download', () => {
  shell.openExternal(DLSSNR_DL.page);
  return true;
});

ipcMain.handle('dlssnr-choose', async () => {
  const r = await dialog.showOpenDialog(win!, {
    title: 'Select nvngx_dlssnr.dll (or a .zip containing it)',
    properties: ['openFile'],
    filters: [
      { name: 'DLSS 5 runtime', extensions: ['dll', 'zip'] },
      { name: 'All files', extensions: ['*'] },
    ],
  });
  return r.canceled ? null : r.filePaths[0] || null;
});

// --- Live mode: the real-time pipeline host (engine/live/smv-live.exe) -----------------------------
// The exe captures the target window via Windows.Graphics.Capture and presents it enhanced into a
// click-through topmost overlay. Backend today: DLSS Frame Generation at (gen+1)x (its own copy of
// the Streamline runtime sits beside it); the design target is backend-agnostic (server route for
// RIFE/GMFSS/FRUC + VSR/HDR/sharpen/restore is the planned phase 2). Gotcha (bisected in the
// smv-live PoC): the overlay must hold foreground for the driver to generate, so the exe
// force-activates its own window at start.
const LIVE_DIR = path.join(ENGINE, 'live');
const LIVE_EXE = path.join(LIVE_DIR, 'smv-live.exe');
// Resident offline host: the render (src/render/native.ts) keeps `smv-live.exe --offline
// --resident` alive between renders (engines loaded, about 300 MB of VRAM idle) and finds it through
// this named pipe, one per app process. The render spawns it detached; this process quits it when a live session starts and at app exit
// (a busy host ignores the request and leaves on its own idle limit).
const OFFLINE_PIPE_NAME = 'smv-offline-' + process.pid;
function offlineHostQuit() {
  try {
    const fd = fs.openSync('\\\\.\\pipe\\' + OFFLINE_PIPE_NAME, 'r+');
    try {
      fs.writeSync(fd, 'quit\n');
    } finally {
      fs.closeSync(fd);
    }
  } catch {
    /* no idle host */
  }
}

// THE merged log (one file for everything): the render (src/render/cli.ts) tees
// its lines to the same literal path, so live session lines land chronologically
// between render runs. Append-only; cli.ts owns the size cap.
const SMV_LOG = path.join(os.tmpdir(), 'smv-engine.log');
function liveLog(text: string) {
  try {
    const lines = text
      .replace(/\r/g, '')
      .split('\n')
      .filter((l) => l.trim() !== '');
    if (!lines.length) return;
    const stamp = new Date().toISOString().slice(0, 19).replace('T', ' ');
    fs.appendFileSync(SMV_LOG, lines.map((l) => `[${stamp}] [live] ${l}\n`).join(''));
  } catch {
    /* the log must never break live */
  }
}
let liveProc: ChildProcess | null = null;
let liveStopping = false;
let liveRestartPending = false; // renderer asked to relaunch the session with new settings
// Resident host: every server-backend session runs `smv-live.exe --resident`,
// which keeps the process alive after the session with what the session loaded: the TensorRT
// engines (about 300 MB of VRAM for RIFE, the next session of that model starts in well under
// a second instead of 2.5 s). Sessions then start and stop over the
// exe's stdin ("start<TAB>args", "stop", "quit"); the exe prints "live session ended: exit N,
// host resident" where a non-resident run exited. The host is freed (quit) when a model other
// than the one it holds is selected in the panel or when DLSS-G starts, when the app quits, and
// by the exe itself after an idle limit (10 min) or when a session used something it cannot
// keep (DLSS-G, native DLSS 5, the TrueHDR bridge, an error exit).
let liveResident = false; // liveProc was spawned with --resident
let liveIdle = false; // the resident host is alive with no session running
let liveRunModel = ''; // the model of the running (or last) session on liveProc
let liveHeld = ''; // the model the idle resident host keeps loaded (= its last session's)
let liveResolved: string | null = null; // the running session's target hwnd (revives reuse it)
let liveRestarts = 0; // exit 4 / 6 revive count of the current target
let liveStopTimer: NodeJS.Timeout | null = null; // "stop" grace: kill the host if the line never comes
const LIVE_STOP_GRACE_MS = 3000;
const LIVE_ENDED_RE = /^live session ended: exit (\d+), host resident/;
let liveModel = 'dlssg'; // mirrored from the renderer's Live settings (lv-opts): the hotkey
// and countdown paths spawn with whatever the panel currently shows
let liveLabel = ''; // user-facing effective-model name for the exe's loading message/HUD
let liveNote = ''; // substitution note (e.g. Smooth Motion live runs GMFSS), same destination
let liveDlssMode = 'dlaa'; // the DLSS mode (dlssScaleArg): the working size's share of the presented size
// (server backends only; dlssg has no such input)
let liveDlssCustom = 100; // Custom: that share in %
let liveFit = 'window'; // 'window' = 1:1 overlay, 'fill' = the target's monitor upscaled (server
// only), 'monitor' = whole-screen capture 1:1 (every model incl. dlssg)
let liveTarget = 60; // adaptive output fps target, inherited from the renderer's Speed selectors
let liveSharpen = 0; // live RCAS strength, inherited from the Sharpen controls (0 = off)
let liveVsr = false; // RTX VSR as the live upscaler, inherited from the RTX VSR checkbox
let liveUpH = 0; // "Upscale to" height as the live internal render size (0 = off), inherited from the selector
let liveRestore = false; // Real-ESRGAN on every presented frame, inherited from the Restore checkbox
let liveDlssnr = false; // DLSS 5 Neural Rendering once per captured frame, inherited from the NVIDIA DLSS 5 checkbox
let liveNrStructure = 1; // DLSS 5 Structure Intensity 0..2
let liveNrTone = 1; // DLSS 5 Tone Intensity 0..2
let liveNrStyle = 1; // DLSS 5 Style: 0 Default, 1 Natural, 2 Cinematic
let liveNrPasses = 1; // DLSS 5 passes, 1..10 (chained per frame)
let liveRtxHdr = false; // live TrueHDR, inherited from the RTX HDR checkbox
let liveHdrColor = 'vivid'; // RTX HDR colour mode + tone knobs, inherited from the HDR sliders
let liveHdrSat = 0; // SDK Saturation (drives the rtx colour mode)
let liveHdrCon = 100; // SDK Contrast (100 = neutral)
let liveHdrVib = 0; // Dynamic Vibrance intensity 0..1 (per-session opt-in, like renders)
let liveHdrSb = 0; // Dynamic Vibrance saturation boost 0..1
let liveHud = true; // on-screen fps/latency readout (panel checkbox; off -> --no-hud)
let liveHudLat = true; // latency segment of that readout (off -> --no-hud-latency)

// Live events go to the main window (not a captured e.sender): the ` hotkey starts sessions
// with no IPC event at all, and the panel must reflect those too.
const sendLive = (channel: string, ...payload: unknown[]) => {
  try {
    if (win && !win.webContents.isDestroyed()) win.webContents.send(channel, ...payload);
  } catch {
    /* window gone */
  }
};

// SMV's own top-level window handle, so hotkey mode never overlays the app itself
function ownHwnd(): string {
  try {
    return win ? '0x' + win.getNativeWindowHandle().readBigUInt64LE(0).toString(16) : '0x0';
  } catch {
    return '0x0';
  }
}

// write one command line to the resident host; false when the pipe is gone
function liveCommand(line: string): boolean {
  try {
    if (liveProc?.stdin && !liveProc.stdin.destroyed) {
      liveProc.stdin.write(line + '\n');
      return true;
    }
  } catch {
    /* the host died under us; the close handler follows */
  }
  return false;
}

// end the idle resident host (a different model is about to run, or the app quits): "quit"
// lets it release the engines and exit on its own, the kill covers a wedged one
function liveQuitIdle() {
  if (!liveProc || !liveIdle) return;
  const p = liveProc;
  liveLog('quitting the idle resident host');
  if (!liveCommand('quit')) {
    try {
      p.kill();
    } catch {
      /* gone */
    }
  }
  const t = setTimeout(() => {
    try {
      if (p.exitCode === null) p.kill();
    } catch {
      /* gone */
    }
  }, 2000);
  t.unref();
  liveProc = null;
  liveIdle = false;
  liveResident = false;
}

// a session ended: the exe exited (code from 'close') or, on the resident host, printed its
// "live session ended" line and stays alive. Same policy either way: a settings restart or an
// exit 4 / 6 revive relaunches the SAME target, anything else reports lv-done.
function onLiveSessionEnd(code: number | null) {
  if (liveStopTimer) {
    clearTimeout(liveStopTimer);
    liveStopTimer = null;
  }
  const resolved = liveResolved;
  const restarts = liveRestarts;
  const busy = () => liveProc !== null && !liveIdle;
  // settings restart (renderer 'lv-restart', e.g. the DLSS mode changed mid-run):
  // relaunch the SAME target with the CURRENT lv-opts mirror; not counted against the cap
  if (liveRestartPending && !liveStopping && resolved) {
    liveRestartPending = false;
    liveLog('restarting with new settings');
    setTimeout(() => {
      if (!busy() && !liveStopping) startLiveSession(resolved, restarts);
    }, 400);
    return;
  }
  // exit 4 = target window resized (the overlay cannot resize in place); exit 6 = the stall
  // watchdog killed a wedged enhancement engine. Both revive the SAME session with the SAME
  // configuration: a mid-session downgrade (e.g. to the eager model path) would read as
  // "it suddenly got slow" to the user, worse than a brief hiccup at full speed. Bounded by
  // the shared restart cap (interactive resizing alone can fire many restarts).
  if ((code === 4 || code === 6) && !liveStopping && restarts < 20 && resolved) {
    liveLog(`session exit ${code}, reviving (restart ${restarts + 1})`);
    if (code === 6) sendLive('lv-out', 'live engine stalled, reviving the session\n');
    setTimeout(() => {
      if (!busy() && !liveStopping) startLiveSession(resolved, restarts + 1);
    }, 400);
    return;
  }
  if (code === 6) sendLive('lv-out', 'live engine keeps stalling, giving up on this session\n');
  liveLog(`stopped (exit ${code})`);
  sendLive('lv-done', code);
}

// one overlay session; explicit hwnd (GUI picker) or the foreground window (` hotkey)
function startLiveSession(hwnd: string | null, restarts = 0) {
  if (liveProc && !liveIdle) return;
  liveStopping = false;
  liveRestartPending = false;
  offlineHostQuit(); // an idle offline render host gives its VRAM to the live session
  const args = hwnd ? ['--hwnd', hwnd] : ['--fg', '--exclude', ownHwnd()];
  // ONE knob: the fps target from the Speed selectors. Adaptive models resample to it; fixed
  // pipelines (DLSS 4.5) approximate it in the exe with the nearest whole multiple of the
  // captured window's measured rate, capped by the model (no --gen: the exe derives it).
  const target = String(Math.min(1000, Math.max(10, Math.round(Number(liveTarget) || 60))));
  if (liveModel && liveModel !== 'dlssg') {
    args.push('--backend', liveModel);
    // user-facing name + substitution note for the exe's loading message/HUD (the raw
    // backend id read as the wrong model when live substitutes, e.g. Smooth Motion -> GMFSS)
    if (liveLabel) args.push('--label', liveLabel);
    if (liveNote) args.push('--note', liveNote);
    // RIFE/GMFSS/blend resample adaptively to the fps target inherited from the Speed
    // selectors. No --gen: with only a target the exe derives the slot count per pair from
    // the measured source rate (ceil(target/source)+1, VRAM-clamped), so high targets are
    // not capped by a fixed 16-slot ceiling.
    args.push('--target', target);
    // live effects, inherited from the file-render Sharpen / RTX VSR settings
    const sharp = Math.min(1, Math.max(0, Number(liveSharpen) || 0));
    if (sharp > 0) args.push('--sharpen', sharp.toFixed(2));
    // "Upscale to": the server resizes the model frame to this height
    // first (VSR when enlarging), then fits it to the canvas; so VSR can now engage outside
    // fill mode too. Without it an upscale exists only in fill mode.
    const upH = Math.max(0, Math.round(Number(liveUpH) || 0));
    if (upH > 0) args.push('--upscale', String(upH));
    // the host gives RTX VSR the one resize that enlarges (the fit after the model, else the capture to the
    // working size before it) and skips it with a line when neither does
    if (liveVsr) args.push('--rtx-vsr');
    // Restore: Real-ESRGAN first on every presented frame; costs most
    // of a 1080p frame budget, the panel hint says so
    if (liveRestore) args.push('--restore');
    // NVIDIA DLSS 5: Neural Rendering once per captured frame inside
    // the exe (SDR domain, before the model; the tweens inherit it); a session whose pass cannot
    // run goes on without it and logs why. Only sent when the runtime is installed (the
    // renderer gates on dlssnr-ready).
    if (liveDlssnr)
      args.push(
        '--dlssnr',
        '--nr-structure',
        String(liveNrStructure),
        '--nr-tone',
        String(liveNrTone),
        '--nr-style',
        String(liveNrStyle),
      );
    if (liveDlssnr && liveNrPasses > 1) args.push('--nr-passes', String(liveNrPasses));
    // live TrueHDR: SDR window expanded to HDR out, inherited from the RTX HDR
    // controls; the exe forwards these to the server only when its HDR live mode is on
    if (liveRtxHdr) {
      args.push('--rtx-hdr');
      if (liveHdrColor !== 'vivid') args.push('--hdr-color', liveHdrColor, '--hdr-saturation', String(liveHdrSat));
      if (liveHdrCon !== 100) args.push('--hdr-contrast', String(liveHdrCon));
      if (liveHdrVib > 0) args.push('--hdr-vibrance', String(liveHdrVib));
      if (liveHdrSb > 0) args.push('--hdr-satboost', String(liveHdrSb));
    }
  } else {
    // DLSS 4.5 generates whole in-between frames: the exe derives the count from the target
    // and the measured capture rate at runtime (capped at 6x by the model)
    args.push('--target', target);
  }
  // the DLSS mode: the working size as its share of the presented size (the window, or the Fill rect); the
  // host resolves Auto by that size, and no flag = DLAA (the presented size itself)
  const scaleArg = dlssScaleArg(liveDlssMode, liveDlssCustom);
  if (liveModel !== 'dlssg' && scaleArg) args.push('--scale', scaleArg);
  if (liveModel !== 'dlssg' && liveFit === 'fill') args.push('--fit', 'fill');
  if (liveFit === 'monitor') args.push('--fit', 'monitor'); // whole-screen: all models incl. dlssg
  if (!liveHud) args.push('--no-hud');
  else if (!liveHudLat) args.push('--no-hud-latency'); // meter on, latency segment hidden
  // Every server backend runs inside smv-live.exe (its native host), the only live route; a
  // session the host cannot run ends with its reason on the status line.
  // resident host: every server backend has something worth keeping (the native engines); the
  // exe itself exits after a session it cannot stay resident for. SMV_LIVE_RESIDENT=0 restores
  // one process per session (A/B harness).
  const resident = liveModel !== 'dlssg' && process.env.SMV_LIVE_RESIDENT !== '0';
  liveResolved = hwnd; // hotkey mode learns the resolved hwnd from the exe's log line
  liveRestarts = restarts;
  liveRunModel = liveModel;
  if (liveProc && liveIdle) {
    if (resident && liveResident && liveCommand('start\t' + args.join('\t'))) {
      liveIdle = false;
      liveLog(`session started on the resident host: ${args.join(' ')}`);
      sendLive('lv-started');
      return;
    }
    // a model the resident host cannot serve (or a dead pipe): free it, spawn fresh
    liveQuitIdle();
  }
  if (resident) args.push('--resident');
  const p = spawn(LIVE_EXE, args, { cwd: LIVE_DIR });
  liveProc = p;
  liveResident = resident;
  liveIdle = false;
  let errBuf = ''; // partial trailing line, so the hwnd is only parsed out of COMPLETE lines
  liveLog(`session started: smv-live.exe ${args.join(' ')}`);
  sendLive('lv-started');
  // a write into a host that just exited raises EPIPE on the stream: never let it throw
  p.stdin?.on('error', () => {});
  p.stderr?.on('data', (b: Buffer) => {
    const t = b.toString();
    // a chunk boundary inside "target window: hwnd=0x..." used to resolve a TRUNCATED handle,
    // and the next settings restart then respawned on a window that does not exist
    errBuf += t;
    const lines = errBuf.split(/\r?\n/);
    errBuf = lines.pop() ?? '';
    for (const line of lines) {
      const m = /target window: hwnd=0x0*([0-9a-fA-F]+)/.exec(line);
      if (m && liveProc === p && !liveIdle) liveResolved = '0x' + m[1].toLowerCase();
      const e = LIVE_ENDED_RE.exec(line);
      if (e && liveProc === p) {
        // hotkey mode (--fg) has no hwnd of its own: it learns the target from the exe's
        // "target window:" line, so a session that never printed one left the exit 4 / 6
        // revive with nothing to revive (a GMFSS session died on a resize that way: the
        // title carried an en dash and the line was dropped). The exe
        // repeats the resolved target on this line, so take it from there as the fallback.
        if (!liveResolved) {
          const h = /target=0x0*([0-9a-fA-F]+)/.exec(line);
          if (h) liveResolved = '0x' + h[1].toLowerCase();
        }
        // the resident host stays alive: this line IS the session's exit
        liveIdle = true;
        liveHeld = liveRunModel;
        onLiveSessionEnd(Number(e[1]));
      }
    }
    liveLog(t);
    sendLive('lv-out', t);
  });
  p.on('close', (code) => {
    liveRestoreMouse(false);
    if (liveProc !== p) return; // an idle host quit by liveQuitIdle, already forgotten
    const wasIdle = liveIdle;
    liveProc = null;
    liveIdle = false;
    liveResident = false;
    if (wasIdle) {
      // the resident host left on its own (idle limit, stdin closed): no session was running
      liveLog(`resident host exited (${code})`);
      return;
    }
    onLiveSessionEnd(code);
  });
  p.on('error', (err) => {
    liveProc = null;
    liveIdle = false;
    liveResident = false;
    liveLog('spawn error: ' + err);
    sendLive('lv-out', 'live spawn error: ' + err + '\n');
    sendLive('lv-done', -1);
  });
}

ipcMain.handle('lv-ready', () => fileExists(LIVE_EXE));

// Bundled example clip (Big Buck Bunny, CC-BY Blender Foundation): drives the first-page
// settings + before/after preview before the user has picked any video.
// The example preview clip ships as its own extraResource (samples/ is otherwise local
// test material and stays out of both git and the package).
const EXAMPLE_MP4 = app.isPackaged
  ? path.join(process.resourcesPath, 'samples', 'example.mp4')
  : path.join(ROOT, 'samples', 'example.mp4');
ipcMain.handle('example-path', () => (fileExists(EXAMPLE_MP4) ? EXAMPLE_MP4 : null));

// window picker: parse the exe's "0xHWND<TAB>title" UTF-8 lines
ipcMain.handle('lv-list', () => {
  return new Promise<{ hwnd: string; title: string }[]>((resolve) => {
    const p = spawn(LIVE_EXE, ['--list'], { cwd: LIVE_DIR });
    let out = '';
    p.stdout.on('data', (b: Buffer) => (out += b.toString('utf8')));
    p.on('close', () =>
      resolve(
        out
          .split(/\r?\n/)
          .map((l) => {
            const t = l.indexOf('\t');
            return t > 0 ? { hwnd: l.slice(0, t), title: l.slice(t + 1) } : null;
          })
          .filter((w): w is { hwnd: string; title: string } => w !== null),
      ),
    );
    p.on('error', () => resolve([]));
  });
});

// the Smooth It Live! button: after the renderer's countdown, target the foreground window
// (the user clicked the window they want during the countdown; --exclude keeps SMV itself out)
ipcMain.on('lv-start-fg', () => startLiveSession(null));
// end the running session: "stop" on the resident host (it ends the session and stays), a
// kill otherwise; the grace timer kills a host whose "ended" line never comes (the native
// engine load polls the stop; the kill keeps Stop instant for a stage that does not). A stop
// already pending keeps its
// timer: a hotkey pressed every 2 s used to re-arm the grace each time, so the kill never
// fired and the panel sat on "loading the model" for the whole build.
function liveEndSession() {
  if (!liveProc || liveIdle) return;
  const p = liveProc;
  if (liveResident && liveCommand('stop')) {
    if (liveStopTimer) return;
    liveStopTimer = setTimeout(() => {
      liveStopTimer = null;
      if (liveProc === p && !liveIdle) {
        liveLog('stop grace elapsed, killing the host');
        try {
          p.kill();
        } catch {
          /* already gone */
        }
      }
    }, LIVE_STOP_GRACE_MS);
    return;
  }
  try {
    p.kill();
  } catch {
    /* already gone */
  }
}

// relaunch a running session so spawn-time settings (the DLSS mode) take effect; no-op when idle
ipcMain.on('lv-restart', () => {
  if (liveProc && !liveIdle && !liveStopping) {
    liveRestartPending = true;
    liveEndSession();
  }
});
ipcMain.on(
  'lv-opts',
  (
    _e,
    opts: {
      model: string;
      label?: string;
      note?: string;
      dlssmode?: string; // the DLSS mode (dlssScaleArg)
      dlsscustom?: number; // Custom: the working size in % of the presented size
      fit: string;
      target?: number;
      sharpen?: number;
      rtxvsr?: boolean;
      uph?: number;
      restore?: boolean;
      dlssnr?: boolean;
      nrstructure?: number;
      nrtone?: number;
      nrstyle?: number; // DLSS 5 Style 0 Default / 1 Natural / 2 Cinematic
      nrpasses?: number; // DLSS 5 passes 1..10
      rtxhdr?: boolean;
      hdrcolor?: string;
      hdrsat?: number;
      hdrcon?: number;
      hdrvib?: number;
      hdrsb?: number;
      hud?: boolean;
      hudlat?: boolean;
    },
  ) => {
    liveModel = opts.model;
    liveLabel = opts.label ?? '';
    liveNote = opts.note ?? '';
    liveDlssMode = opts.dlssmode ?? 'dlaa';
    liveDlssCustom = opts.dlsscustom ?? 100;
    liveFit = opts.fit;
    if (opts.target !== undefined) liveTarget = opts.target;
    liveSharpen = opts.sharpen ?? 0;
    liveVsr = !!opts.rtxvsr;
    liveUpH = opts.uph ?? 0;
    liveRestore = !!opts.restore;
    liveDlssnr = !!opts.dlssnr;
    liveNrStructure = opts.nrstructure ?? 1;
    liveNrTone = opts.nrtone ?? 1;
    liveNrStyle = opts.nrstyle ?? 1;
    liveNrPasses = Math.min(10, Math.max(1, Math.round(opts.nrpasses ?? 1)));
    liveRtxHdr = !!opts.rtxhdr;
    liveHdrColor = opts.hdrcolor ?? 'vivid';
    liveHdrSat = opts.hdrsat ?? 0;
    liveHdrCon = opts.hdrcon ?? 100;
    liveHdrVib = opts.hdrvib ?? 0;
    liveHdrSb = opts.hdrsb ?? 0;
    liveHud = opts.hud !== false;
    liveHudLat = opts.hudlat !== false;
    // the panel moved to another model: whatever the idle resident host keeps loaded (its
    // engines) goes right away (user rule: a model must not hold VRAM
    // through a session of another model)
    if (liveIdle && liveModel !== liveHeld) liveQuitIdle();
  },
);

// The global Live toggle hotkey, Lossless-Scaling-style: press it in any app and the window
// you are in goes live; press it again to stop. Global shortcuts swallow the key system-wide
// while the app runs - inherent to the feature, hence the panel lets the user pick the key.
// Default ` (backtick); the renderer restores a persisted choice at boot via 'lv-hotkey'.
let liveHotkey = '';
function liveToggle() {
  if (liveProc && !liveIdle) stopLive();
  else startLiveSession(null);
}
function registerLiveHotkey(acc: string): boolean {
  if (!fileExists(LIVE_EXE)) return false;
  if (liveHotkey) globalShortcut.unregister(liveHotkey);
  const ok = globalShortcut.register(acc, liveToggle);
  if (ok) liveHotkey = acc;
  else if (liveHotkey) globalShortcut.register(liveHotkey, liveToggle); // keep the old key working
  console.log(ok ? `live hotkey registered: ${acc}` : `live hotkey registration FAILED: ${acc}`);
  return ok;
}
ipcMain.handle('lv-hotkey', (_e, acc: string) => registerLiveHotkey(String(acc || '`')));
app.whenReady().then(() => registerLiveHotkey('`'));
app.on('will-quit', () => globalShortcut.unregisterAll());

// Fill's mouse mapping (the exe's FillMouse) holds the pointer speed and a cursor clip while the cursor is on the
// stretched picture and keeps them in this marker until it gives them back: a host that was killed (the stop grace,
// the app quitting) or crashed cannot, so the exe's --restore-mouse does it (GetTempPath's order: TMP, then TEMP)
const LIVE_MOUSE_MARKER = path.join(process.env.TMP || process.env.TEMP || os.tmpdir(), 'smv-live-mouse.txt');
function liveRestoreMouse(sync: boolean) {
  if (!fileExists(LIVE_MOUSE_MARKER)) return;
  liveLog('restoring the pointer speed and clip a killed Fill session held');
  try {
    if (sync) execFileSync(LIVE_EXE, ['--restore-mouse'], { cwd: LIVE_DIR, timeout: 5000 });
    else execFile(LIVE_EXE, ['--restore-mouse'], { cwd: LIVE_DIR, timeout: 5000 }, () => {});
  } catch {
    /* the next live session restores it */
  }
}

// hard = the app is leaving: kill outright (no "stop" grace, the overlay must not outlive the GUI)
function stopLive(hard = false) {
  liveStopping = true;
  if (!liveProc) return;
  if (hard) {
    try {
      liveProc.kill();
    } catch {
      /* already gone */
    }
    liveRestoreMouse(true);
    return;
  }
  if (liveIdle) liveQuitIdle();
  else if (liveProc.pid) liveEndSession();
}
ipcMain.handle('lv-stop', () => stopLive());
ipcMain.handle('fruc-ready', () => {
  const bridge = fileExists(path.join(NVOFFRUC_DIR, 'nvoffruc_bridge.dll'));
  const dll = fileExists(path.join(NVOFFRUC_DIR, 'NvOFFRUC.dll'));
  return { ready: bridge && dll, bridge, dll, dir: NVOFFRUC_DIR };
});
ipcMain.handle('fruc-open-download', () => {
  shell.openExternal(OF_SDK_URL);
  return true;
});
ipcMain.handle('fruc-install', (_e, source?: string) => {
  if (!source)
    return { ok: false, error: 'No Optical Flow SDK selected. Use "Get from NVIDIA", then "Choose .zip".', copied: [] };
  return installFruc(source);
});
ipcMain.handle('fruc-choose', async () => {
  const r = await dialog.showOpenDialog(win!, {
    title: 'Select the Optical Flow SDK .zip',
    properties: ['openFile'],
    filters: [{ name: 'Zip', extensions: ['zip'] }],
  });
  return r.canceled ? null : r.filePaths[0] || null;
});

// --- Dolby Vision export tool: readiness + install (mirrors the RTX flow) -------------------------
// DV Profile 8.1 export layers a Dolby Vision RPU on top of the HDR10 render, then tags the MP4 with a
// dvvC box the engine writes itself (see hdr10_meta.inject_dv_config) - so the bundled ffmpeg is enough
// to mux DV and NO GPAC/MP4Box is needed. The one non-bundled piece is dovi_tool (open source, but
// Dolby-adjacent enough that the user fetches it deliberately, like the NVIDIA DLLs); this app copies
// dovi_tool.exe into engine/dvtools. "Ready" = dovi_tool present.
const DV_DIR = path.join(ENGINE, 'dvtools');
const DOVI_BIN = 'dovi_tool.exe';
// The general releases page (newest at the top) so the user always grabs the latest build; the UI
// names the exact Windows asset to pick.
const DOVI_URL = 'https://github.com/quietvoid/dovi_tool/releases/';

// Recursive case-insensitive search for a file by name under root (dovi_tool.exe can sit in a nested
// folder inside its release zip).
const findDvBin = (root: string, name: string): string | null => findFilesNamed(root, [name])[0] || null;

// Copy a single tool file out of a chosen source (its release .zip, a folder, or the file directly)
// into an engine subdir. Shared by the Dolby Vision (dovi_tool), HDR10+ (hdr10plus_tool) and DLSS 5
// runtime installers.
const installBin = (source: string, binName: string, destDir: string): InstallResult =>
  installFiles(source, [binName], destDir, { what: 'the selection', where: binName + ' into ' + destDir });

ipcMain.handle('dv-ready', () => {
  const dovi = fileExists(path.join(DV_DIR, DOVI_BIN));
  return { dovi, ready: dovi, dir: DV_DIR };
});

ipcMain.handle('dv-open-download', () => {
  shell.openExternal(DOVI_URL);
  return true;
});

ipcMain.handle('dv-install', (_e, source: string) => installBin(source, DOVI_BIN, DV_DIR));

// Picker: the UI only ever talks about the release .zip, so the default filter is zip-only; a bare
// dovi_tool.exe is still accepted SILENTLY via the "All files" fallback (installBin handles both).
ipcMain.handle('dv-choose', async () => {
  const r = await dialog.showOpenDialog(win!, {
    title: 'Select the dovi_tool release .zip',
    properties: ['openFile'],
    filters: [
      { name: 'Zip', extensions: ['zip'] },
      { name: 'All files', extensions: ['*'] },
    ],
  });
  return r.canceled ? null : r.filePaths[0] || null;
});

// --- HDR10+ export tool: readiness + install (mirrors the Dolby Vision flow) ----------------------
// HDR10+ export embeds ST 2094-40 dynamic metadata into the HDR10 render; the engine collects the
// per-frame stats itself and the user-installed hdr10plus_tool injects the SEI (see _hp_export).
const HP_DIR = path.join(ENGINE, 'hptools');
const HP_BIN = 'hdr10plus_tool.exe';
const HP_URL = 'https://github.com/quietvoid/hdr10plus_tool/releases/';

ipcMain.handle('hp-ready', () => {
  const ready = fileExists(path.join(HP_DIR, HP_BIN));
  return { ready, dir: HP_DIR };
});

ipcMain.handle('hp-open-download', () => {
  shell.openExternal(HP_URL);
  return true;
});

ipcMain.handle('hp-install', (_e, source: string) => installBin(source, HP_BIN, HP_DIR));

// Picker: zip-only default filter like the DV one; a bare hdr10plus_tool.exe still works
// silently through "All files".
ipcMain.handle('hp-choose', async () => {
  const r = await dialog.showOpenDialog(win!, {
    title: 'Select the hdr10plus_tool release .zip',
    properties: ['openFile'],
    filters: [
      { name: 'Zip', extensions: ['zip'] },
      { name: 'All files', extensions: ['*'] },
    ],
  });
  return r.canceled ? null : r.filePaths[0] || null;
});

let current: ChildProcess | null = null;
let currentOut: string | null = null; // output path of the in-flight run, for .part cleanup on Cancel
// Long renders (16K, overnight batch) must survive display/system sleep, and their progress should show
// on the taskbar even when the window is unfocused. Both are driven from the run lifecycle below.
let sleepBlocker: number | null = null;
const keepAwake = (on: boolean) => {
  if (on) {
    if (sleepBlocker === null || !powerSaveBlocker.isStarted(sleepBlocker))
      sleepBlocker = powerSaveBlocker.start('prevent-app-suspension'); // system stays awake; the display may still sleep
  } else if (sleepBlocker !== null) {
    if (powerSaveBlocker.isStarted(sleepBlocker)) powerSaveBlocker.stop(sleepBlocker);
    sleepBlocker = null;
  }
};
const taskbarProgress = (frac: number) => {
  try {
    win?.setProgressBar(frac);
  } catch {
    /* no window */
  }
};

// Live progress thumbnail: the render overwrites this PNG about once a second (see
// SMV_LIVE_PREVIEW below); the renderer polls it by mtime and shows the frame being written.
const LIVE_PNG = path.join(app.getPath('userData'), 'preview', 'live.png');
ipcMain.handle('live-path', () => LIVE_PNG);

// Cooperative pause flag: the engine (SMV_PAUSE_FILE) checks this file at each source-pair
// boundary. Creating it holds generation after the queued frames finish encoding; removing it
// resumes. The renderer's Pause/Resume button drives it via the 'pause'/'resume' IPC below.
const PAUSE_FLAG = path.join(app.getPath('userData'), 'preview', 'pause.flag');
const clearPause = () => {
  try {
    fs.unlinkSync(PAUSE_FLAG);
  } catch {
    /* not paused */
  }
};
ipcMain.on('pause', () => {
  try {
    fs.writeFileSync(PAUSE_FLAG, '');
  } catch {
    /* ignore */
  }
});
ipcMain.on('resume', clearPause);
// A fresh renderer page (app launch OR a reload) can't manage an engine started by the previous page.
// If a reload slipped past the key guard in createWindow, that engine is orphaned - still streaming
// PROGRESS - and the next run would race it (the 1%->15%->1% ping-pong). So on every renderer load,
// kill any in-flight engine and clear the pause flag: the page comes up clean/idle. On first launch
// current is null, so this is a harmless no-op that just clears any stale pause flag.
ipcMain.on('renderer-ready', () => {
  const c = current;
  current = null;
  if (c && c.pid) {
    try {
      execFile('taskkill', ['/pid', String(c.pid), '/T', '/F'], () => {});
    } catch {
      /* already gone */
    }
  }
  clearPause();
});

// Live-preview Hide toggle: the engine (SMV_LIVE_OFF_FILE) skips producing the thumbnail while this
// file exists, so hiding it reclaims the per-second snapshot cost, not just the UI. The renderer
// sends its persisted preference on toggle and at each run start.
const LIVE_OFF_FLAG = path.join(app.getPath('userData'), 'preview', 'live_off.flag');
ipcMain.on('live-off', (_e, off: boolean) => {
  try {
    if (off) fs.writeFileSync(LIVE_OFF_FLAG, '');
    else fs.unlinkSync(LIVE_OFF_FLAG);
  } catch {
    /* already in the desired state */
  }
});

// The renderer's render request (one field per GUI control; see engineArgs for what each becomes).
type RunOpts = {
  input: string;
  multi: number;
  output: string;
  fps?: number;
  sharpen?: number;
  restore?: boolean;
  dlssmode?: string; // the DLSS mode (dlssScaleArg)
  dlsscustom?: number; // Custom: the working size in % of the output
  dlssnr?: boolean;
  nrstructure?: number;
  nrtone?: number;
  nrstyle?: number; // DLSS 5 Style 0 Default / 1 Natural / 2 Cinematic
  nrpasses?: number; // DLSS 5 passes 1..10
  interp?: boolean;
  model?: string;
  upscale?: number;
  rtxvsr?: boolean;
  rtxhdr?: boolean;
  dv?: boolean;
  hp?: boolean;
  codec?: string;
  hdrcolor?: string;
  hdrsat?: number;
  hdrcon?: number;
  hdrsb?: number;
  hdrvib?: number;
};

// The GUI's DLSS mode as the CLI's --scale value: auto / quality / balanced / performance / ultra
// by name, Custom as the working size's share of the output (33..100 %); null = DLAA, the default.
function dlssScaleArg(mode?: string, custom?: number): string | null {
  if (mode === 'custom') {
    const pct = Math.min(100, Math.max(1, Math.round(custom ?? 100)));
    return pct >= 100 ? null : (pct / 100).toFixed(2);
  }
  return mode && ['auto', 'quality', 'balanced', 'performance', 'ultra'].includes(mode) ? mode : null;
}

// The render command line for one request (pure: the GUI state in, render.py's argv out; the TS
// orchestrator dist/render/cli.js takes the same argv).
function engineArgs(opts: RunOpts): string[] {
  const args = [opts.input, String(opts.multi), opts.output];
  // Output codec family (hevc default / av1 / vvc); the engine owns encoder pick + fallbacks.
  if (opts.codec && opts.codec !== 'hevc') args.push('--codec', opts.codec);
  // Interpolation is the default; interp === false means the user only wants the sharpen pass,
  // so tell the engine to skip frame generation (and ignore any fps/multi) entirely.
  if (opts.interp === false) args.push('--no-interp');
  else {
    if (opts.model === 'rife') args.push('--rife'); // RIFE 4.26 heavy backend instead of GMFSS (bundled)
    if (opts.model === 'rifedrba') args.push('--rife-drba'); // RIFE with DRBA anime-pacing timing
    if (opts.model === 'dlssg') args.push('--dlssg'); // "DLSS 4.5" (Frame Generation) backend instead of GMFSS
    if (opts.model === 'fruc') args.push('--fruc'); // "NVIDIA Smooth Motion" backend instead of GMFSS
    if (opts.model === 'lsfg') args.push('--lsfg'); // Frame Blend: flow-warp interpolation
    if (opts.model === 'nvof') args.push('--nvof'); // NVIDIA Optical Flow: hardware flow + splat, native host only
    if (opts.fps && opts.fps > 0) args.push('--fps', String(opts.fps));
  }
  // DLSS mode (the mode selector): the working size Restore's output, DLSS 5 and the model run at,
  // NVIDIA's share of the output (plan.ts workPlan); DLAA, the default, is the output itself
  const scaleArg = dlssScaleArg(opts.dlssmode, opts.dlsscustom);
  if (scaleArg) args.push('--scale', scaleArg);
  // FSR-style RCAS sharpening strength (GUI checkbox + slider). 0/omitted = off, leaving the
  // frames value-preserving; >0 enables the in-engine RCAS pass. Works with or without interp.
  if (opts.sharpen && opts.sharpen > 0) args.push('--sharpen', String(opts.sharpen));
  // AI detail restoration (GUI Restore checkbox): Real-ESRGAN animevideov3 once per source frame,
  // first (it folds straight to the working size). Works with or without interpolation.
  if (opts.restore) args.push('--restore');
  // NVIDIA DLSS 5 Neural Rendering (GUI checkbox + the two sliders): once per source frame at the
  // working size, after Restore and the resize, before the interpolation (NVIDIA's order). The
  // renderer only sends it when the user-supplied runtime is installed (dlssnr-ready).
  if (opts.dlssnr)
    args.push(
      '--dlssnr',
      '--nr-structure',
      String(opts.nrstructure ?? 1),
      '--nr-tone',
      String(opts.nrtone ?? 1),
      '--nr-style',
      String(opts.nrstyle ?? 1),
    );
  if (opts.dlssnr && (opts.nrpasses ?? 1) > 1)
    args.push('--nr-passes', String(Math.min(10, Math.max(1, Math.round(opts.nrpasses ?? 1)))));
  // Resize factor (an arbitrary float, source height -> chosen target height), computed by the
  // renderer from the resolution selector. >1 enables the upscale pass; <1 is a downscale the
  // engine FOLDS into the decode (whole pipeline runs at the output size - also what keeps 4K
  // sources inside TRT-safe flow shapes, so dropping it here re-breaks 4K GMFSS renders).
  // Without --rtx-vsr an upscale is a Lanczos3 resize; with it, RTX Video Super Resolution.
  if (opts.upscale && opts.upscale > 0 && opts.upscale !== 1) args.push('--upscale', String(opts.upscale));
  // RTX VSR: the real RTX Video SDK (the engine/rtxvideo CUDA bridge) for an enlarging resize: the
  // final one from the working size (a mode below DLAA or an upscale), else the one before the
  // model; the plan skips it when nothing enlarges. Falls back to Lanczos3 if the bridge or the RTX
  // Video runtime is unavailable.
  if (opts.rtxvsr) args.push('--rtx-vsr');
  // RTX HDR (TrueHDR): convert the output to HDR10. Works with or without --upscale (when both are
  // on, the RTX bridge does VSR then TrueHDR in one pass). The engine masters at a fixed 1000-nit
  // peak and writes the HDR10 metadata, so there is no per-display nits knob; it falls back to an
  // SDR render if the bridge is unavailable.
  if (opts.rtxhdr) {
    args.push('--rtx-hdr');
    // HDR colour controls: mode (vivid default / rtx / raw), the SDK Saturation (drives rtx and raw
    // modes; inert in vivid), and the vibrance boost (vivid/rtx modes).
    if (opts.hdrcolor && opts.hdrcolor !== 'vivid') {
      args.push('--hdr-color', opts.hdrcolor);
      if (typeof opts.hdrsat === 'number') args.push('--hdr-saturation', String(opts.hdrsat));
    }
    if (opts.hdrvib && opts.hdrvib > 0) args.push('--hdr-vibrance', String(opts.hdrvib));
    if (opts.hdrsb && opts.hdrsb > 0) args.push('--hdr-satboost', String(opts.hdrsb));
    // RTX HDR tone curve (SDK 0..200, 100 = neutral; the GUI shows the App's -100..100 scale).
    if (typeof opts.hdrcon === 'number' && opts.hdrcon !== 100) args.push('--hdr-contrast', String(opts.hdrcon));
    // Dolby Vision Profile 8.1 export on top of the HDR10 render (needs dovi_tool in engine/dvtools).
    if (opts.dv) args.push('--dv');
    // HDR10+ dynamic metadata on top of the HDR10 render (needs hdr10plus_tool in engine/hptools).
    if (opts.hp) args.push('--hdr10plus');
  }
  return args;
}

// The engine's environment. PYTHONUTF8 keeps the dynamo ONNX exporter's unicode logs from
// crashing the engine during first-run TRT builds. The TRT cache deliberately gets NO override
// here: GUI and CLI runs share the one in-app cache next to the engine
// (engine/trt_cache_safe_to_delete); a separate AppData cache would split them, so GUI renders
// would rebuild engines the CLI cache already has. SMV_LIVE_PREVIEW makes the render drop a small PNG
// of the frame being written about once a second; the renderer polls it for the live progress
// thumbnail.
function engineEnv(): NodeJS.ProcessEnv {
  try {
    fs.mkdirSync(path.join(app.getPath('userData'), 'preview'), { recursive: true });
  } catch {
    /* exists */
  }
  return {
    ...process.env,
    SMV_LIVE_PREVIEW: LIVE_PNG,
    SMV_PAUSE_FILE: PAUSE_FLAG,
    SMV_LIVE_OFF_FILE: LIVE_OFF_FLAG,
    SMV_OFFLINE_HOST_PIPE: OFFLINE_PIPE_NAME, // the resident offline host of this app process
  };
}

// Relay the engine's output to the renderer and drive the taskbar progress from it; finish the
// run lifecycle on close or spawn error.
function pipeEngineOutput(proc: ChildProcess, send: (channel: string, ...payload: unknown[]) => void) {
  const onData = (buf: Buffer) => {
    const txt = buf.toString();
    // Drive the Windows taskbar progress from the engine's "PROGRESS k/total" lines (last one wins).
    const hits = [...txt.matchAll(/PROGRESS (\d+)\/(\d+)/g)];
    const last = hits[hits.length - 1];
    if (last) {
      const tot = Number(last[2]);
      if (tot > 0) taskbarProgress(Number(last[1]) / tot);
    }
    if (txt) send('engine-out', txt);
  };
  // NvOFFRUC.dll printfs "Optical Flow Grid Size: N" to stdout on handle create, and the text can
  // arrive SPLIT across chunks (an orphan "4" once reached the log), so a per-chunk regex is not
  // enough: line-buffer stdout, strip matching COMPLETE lines, and flush the tail on close. stderr
  // (the engine's own output, incl. PROGRESS) stays unbuffered for realtime progress.
  let outCarry = '';
  const stripGridSize = (s: string) => s.replace(/^.*Optical Flow Grid Size:.*\r?\n?/gm, '');
  const onStdout = (buf: Buffer) => {
    outCarry += buf.toString();
    const nl = outCarry.lastIndexOf('\n');
    if (nl < 0) return;
    const txt = stripGridSize(outCarry.slice(0, nl + 1));
    outCarry = outCarry.slice(nl + 1);
    if (txt) onData(Buffer.from(txt));
  };
  const finish = () => {
    current = null;
    clearPause();
    keepAwake(false);
    taskbarProgress(-1);
  };
  proc.stdout!.on('data', onStdout);
  proc.stderr!.on('data', onData);
  proc.on('close', (code) => {
    const tail = stripGridSize(outCarry);
    outCarry = '';
    if (tail) send('engine-out', tail);
    finish();
    send('engine-done', code);
  });
  proc.on('error', (err) => {
    finish();
    send('engine-out', 'spawn error: ' + err);
    send('engine-done', -1);
  });
}

ipcMain.on('run', (e, opts: RunOpts) => {
  const args = engineArgs(opts);
  const env = engineEnv();
  clearPause(); // start unpaused: never inherit a stale flag from a previous (e.g. killed) run
  // The render orchestrator is TypeScript: this Electron binary run as plain
  // node on dist/render/cli.js, which speaks render.py's stderr protocol and exit codes and runs
  // every render on the native host. SMV_ENGINE_DIR = the engine folder
  // (packaged: resources/engine, outside the asar the script is read from).
  const proc = spawn(process.execPath, [RENDER_CLI, ...args], {
    cwd: ENGINE,
    env: { ...env, ELECTRON_RUN_AS_NODE: '1', SMV_ENGINE_DIR: ENGINE },
  });
  current = proc;
  currentOut = opts.output || null;
  keepAwake(true);
  try {
    win?.setProgressBar(2, { mode: 'indeterminate' });
  } catch {
    /* warm-up: activity shown before the first PROGRESS line */
  }
  // The engine keeps emitting stdout/stderr (and eventually 'close') asynchronously. If the renderer
  // window was closed mid-render, e.sender is destroyed and e.sender.send() throws "Object has been
  // destroyed", which is an UNCAUGHT exception in the main process and kills the whole app. Guard every
  // send: skip when the WebContents is gone, and try/catch as a backstop against a check/send race.
  const send = (channel: string, ...payload: unknown[]) => {
    try {
      if (!e.sender.isDestroyed()) e.sender.send(channel, ...payload);
    } catch {
      /* renderer went away between the isDestroyed check and the send; nothing to update */
    }
  };
  pipeEngineOutput(proc, send);
});

// Awaited by the renderer: it re-probes resumability only after this resolves, so the
// "Interrupted render found" hint can never race the cleanup below and flash on a cancel.
ipcMain.handle('cancel', async () => {
  const c = current;
  const out = currentOut;
  if (!c || !c.pid) return;
  await new Promise<void>((res) => execFile('taskkill', ['/pid', String(c.pid), '/T', '/F'], () => res()));
  // The engine renders into "<base>.part<ext>" and promotes it to the real name only at
  // success, so a cancelled run leaves a dead .part remnant (plus the stage temps and the
  // crash-resume artifacts - an explicit Cancel means the user is abandoning the render, so
  // its resume assets go too; use Pause or just close the app to keep a render resumable).
  // The killed processes release their handles asynchronously: retry for ~3s, then give up
  // (a still-locked file just stays until the next run overwrites it).
  if (!out) return;
  const ext = path.extname(out);
  const part = out.slice(0, out.length - ext.length) + '.part' + ext;
  let remaining = [
    '',
    '.video.tmp.mp4',
    '.video.mp4',
    '.video2.mp4',
    '.videofull.mp4',
    '.salv.mp4',
    '.salv2.mp4',
    '.trim.mp4',
    '.resume.json',
    '.resume.json.tmp',
    '.concat.txt',
  ].map((s) => part + s);
  for (let attempt = 0; attempt < 10 && remaining.length; attempt++) {
    await new Promise((r) => setTimeout(r, 300));
    remaining = remaining.filter((p) => {
      try {
        fs.unlinkSync(p);
        return false;
      } catch (e) {
        return (e as NodeJS.ErrnoException).code !== 'ENOENT'; // locked: keep retrying
      }
    });
  }
});

// Crash/exit resume probe: given the intended FINAL output path, report whether a resumable
// partial render sits next to it (the engine's stage-1 video + .resume.json sidecar; see the
// resume block in src/render/resume.ts). The renderer uses this to flip Smooth It! to Resume and to
// tell the user where the render will pick up. The engine itself re-validates the settings
// signature, so a stale positive here just means the button said Resume and the run starts
// fresh with a log notice - never a wrong render.
ipcMain.handle('check-resume', (_e, out: string) => {
  try {
    const ext = path.extname(out);
    const part = out.slice(0, out.length - ext.length) + '.part' + ext;
    if (!fs.existsSync(part + '.video.mp4') && !fs.existsSync(part + '.video2.mp4')) return null;
    const meta = JSON.parse(fs.readFileSync(part + '.resume.json', 'utf8'));
    return { pair: Number(meta.pair) || 0, total: Number(meta.total) || 0 };
  } catch {
    return null; // no sidecar (or unreadable): not resumable
  }
});

// The renderer fires this when a job (or the whole batch) finishes; show a native notification only when
// the window is unfocused, so someone who tabbed away during a long render is told it is done.
ipcMain.on('render-complete', (_e, body: string) => {
  if (win && !win.isFocused() && Notification.isSupported()) {
    try {
      new Notification({ title: 'Smooth My Video', body: body || 'Render complete' }).show();
    } catch {
      /* notifications unavailable on this system */
    }
  }
});

// Before/after preview: render ONE source frame at the current spatial settings (RTX HDR when opts.hdr,
// FSR/CAS sharpen when opts.sharpen > 0; no interpolation, no encode) and hand back the two PNG paths
// for the renderer's side-by-side pane. render/preview.js (the render's decode and
// the native host's pass chain, no python) writes <prefix>_original.png and _processed.png; a
// fixed prefix is reused each call (the renderer cache-busts its img src) so previews never pile up.
// Resolves with { error } instead when the frame or the RTX bridge is unavailable.
ipcMain.handle(
  'preview',
  (
    _e,
    opts: {
      input: string;
      frame?: number | string;
      sharpen?: number;
      hdr?: boolean;
      color?: string;
      saturation?: number;
      vibrance?: number;
      satboost?: number;
      contrast?: number;
      upscale?: number;
      rtxvsr?: boolean;
      restore?: boolean;
      dlssmode?: string; // the DLSS mode (dlssScaleArg)
      dlsscustom?: number; // Custom: the working size in % of the output
      dlssnr?: boolean;
      nrstructure?: number;
      nrtone?: number;
      nrstyle?: number; // DLSS 5 Style 0 Default / 1 Natural / 2 Cinematic
      nrpasses?: number; // DLSS 5 passes 1..10
      nrmask?: boolean;
    },
  ) => {
    return new Promise((resolve) => {
      const dir = path.join(app.getPath('userData'), 'preview');
      try {
        fs.mkdirSync(dir, { recursive: true });
      } catch {
        /* already exists */
      }
      const prefix = path.join(dir, 'frame');
      const args = [PREVIEW_CLI, opts.input, '--out', prefix, '--frame', String(opts.frame ?? 'mid')];
      if (opts.sharpen && opts.sharpen > 0) args.push('--sharpen', String(opts.sharpen));
      if (opts.restore) args.push('--restore');
      const pScale = dlssScaleArg(opts.dlssmode, opts.dlsscustom);
      if (pScale) args.push('--scale', pScale);
      if (opts.dlssnr)
        args.push(
          '--dlssnr',
          '--nr-structure',
          String(opts.nrstructure ?? 1),
          '--nr-tone',
          String(opts.nrtone ?? 1),
          '--nr-style',
          String(opts.nrstyle ?? 1),
        );
      if (opts.dlssnr && (opts.nrpasses ?? 1) > 1)
        args.push('--nr-passes', String(Math.min(10, Math.max(1, Math.round(opts.nrpasses ?? 1)))));
      if (opts.dlssnr && opts.nrmask) args.push('--nr-mask'); // heat map of the DLSS 5 change, <prefix>_nrmask.png
      if (opts.upscale && opts.upscale > 0 && opts.upscale !== 1) args.push('--upscale', String(opts.upscale));
      if (opts.rtxvsr) args.push('--rtx-vsr'); // the plan runs it on an enlarging resize only
      if (opts.hdr)
        args.push(
          '--rtx-hdr',
          '--hdr-color',
          String(opts.color ?? 'vivid'),
          '--hdr-saturation',
          String(opts.saturation ?? 0),
          '--hdr-vibrance',
          String(opts.vibrance ?? 0),
          '--hdr-satboost',
          String(opts.satboost ?? 0),
          '--hdr-contrast',
          String(opts.contrast ?? 100),
        );
      // like the render: Electron's own node runs the compiled module (no python)
      const env = { ...process.env, ELECTRON_RUN_AS_NODE: '1', SMV_ENGINE_DIR: ENGINE };
      execFile(process.execPath, args, { cwd: ENGINE, env, maxBuffer: 1 << 20 }, (err, stdout, stderr) => {
        if (err) {
          resolve({ error: String(stderr || err).slice(-400) });
          return;
        }
        const m = /frame (\d+)\/(\d+)/.exec(String(stdout ?? ''));
        // "nrmask=<touched %>/<mean change, 8-bit> <path>" is present only when the DLSS 5 pass ran
        // with --nr-mask; a pass that degraded to the plain frame writes no mask and the pane shows
        // the processed picture as usual.
        const mm = /nrmask=([\d.]+)%\/([\d.]+) /.exec(String(stdout ?? ''));
        resolve({
          original: prefix + '_original.png',
          processed: prefix + '_processed.png',
          frame: m ? Number(m[1]) : null,
          total: m ? Number(m[2]) : null,
          nrmask: mm ? prefix + '_nrmask.png' : null,
          nrmaskPct: mm ? Number(mm[1]) : null,
          nrmaskMean: mm ? Number(mm[2]) : null,
        });
      });
    });
  },
);
