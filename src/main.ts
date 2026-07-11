import { app, BrowserWindow, ipcMain, dialog, screen, shell, powerSaveBlocker, Notification } from 'electron';
import { spawn, execFile, execFileSync, ChildProcess } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';

const ROOT = path.join(__dirname, '..');
// When packaged, the engine ships as an unpacked extraResource (the Python files and
// runtime must be real files on disk, not inside app.asar). The renderer and icon stay
// under ROOT (Electron reads those from the asar fine).
const ENGINE = app.isPackaged ? path.join(process.resourcesPath, 'engine') : path.join(ROOT, 'engine');
// Bundled, relocatable python-build-standalone runtime (full stdlib + torch/cupy stack).
// Its python.exe sits at the runtime root, not under a Scripts/ subdir like a venv.
const RUNTIME_PY = path.join(ENGINE, 'runtime', 'python.exe');
const ENGINE_SCRIPT = path.join(ENGINE, 'gmfss_interp.py');
const PREVIEW_SCRIPT = path.join(ENGINE, 'preview.py');
// Prefer ffprobe bundled at engine/bin (portable build); fall back to PATH for dev.
const FFPROBE = fs.existsSync(path.join(ENGINE, 'bin', 'ffprobe.exe'))
  ? path.join(ENGINE, 'bin', 'ffprobe.exe')
  : 'ffprobe';
// The real cold-start when a video is selected is the before/after PREVIEW: it spawns the bundled Python,
// imports cv2 + torch, and creates a CUDA context (~6s cold, ~2s warm; ffprobe itself is ~0.07s and the
// preview decodes with cv2, not ffmpeg). Warm that engine at launch, while the welcome screen is up, so
// the first preview pays the warm ~2s instead of the cold ~6s. This pre-loads torch's DLLs into the OS
// cache and initialises the CUDA driver/device, which a fresh preview process then skips. It cannot go
// below the ~2s per-process CUDA context cost (that would need a persistent preview daemon). Best-effort.
function warmPreviewEngine() {
  try {
    spawn(pyExe(), ['-c', 'import cv2, torch; torch.zeros(1, device="cuda" if torch.cuda.is_available() else "cpu")'], {
      cwd: ENGINE,
      env: { ...process.env, PYTHONUTF8: '1' },
      stdio: 'ignore',
    }).on('error', () => {});
  } catch {
    /* best-effort */
  }
}

function pyExe(): string {
  return fs.existsSync(RUNTIME_PY) ? RUNTIME_PY : 'python';
}

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
    createWindow();
    warmPreviewEngine();
    checkForUpdate();
  });
}
app.on('window-all-closed', () => {
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

ipcMain.handle('refresh-rate', () => {
  // TRUE refresh rate of the monitor the app window is on (fallback: primary), kept FRACTIONAL so a
  // 59.94 / 359.99 Hz panel feeds the renderer its exact rate - drives "match screen" and its decimal
  // precision (see decimalsOf/targetDecimals in the renderer). Only float noise is trimmed to 3 dp.
  // NOTE: on Windows the OS display API usually reports whole Hz, so this is commonly an integer anyway;
  // a truly fractional rate would need a native DXGI/DWM query, which we don't do.
  try {
    const d = win ? screen.getDisplayMatching(win.getBounds()) : screen.getPrimaryDisplay();
    const hz = d.displayFrequency || screen.getPrimaryDisplay().displayFrequency || 60;
    return Math.round(hz * 1000) / 1000;
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
// NvOFFRUC ("Nvidia Smooth Motion"): our bridge (nvoffruc_bridge.dll) is locally built and ships,
// but NvOFFRUC.dll + cudart64_110.dll are NVIDIA proprietary and user-installed from the Optical
// Flow SDK .zip - same EULA-gated, non-redistributable pattern as the RTX feature DLLs.
const NVOFFRUC_DIR = path.join(ENGINE, 'nvoffruc');
const NVOFFRUC_DLLS = ['NvOFFRUC.dll', 'cudart64_110.dll'];
const OF_SDK_URL = 'https://developer.nvidia.com/opticalflow/download';
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

// Copy the two feature DLLs out of a chosen source (an extracted SDK folder or an SDK .zip) into
// engine/rtxvideo. Zips are handled with Windows' bundled bsdtar, extracting only the two members.
function installRtx(source: string): { ok: boolean; error?: string; copied: string[] } {
  try {
    fs.mkdirSync(RTX_DIR, { recursive: true });
  } catch {
    /* exists */
  }
  let srcFiles: string[] = [];
  try {
    if (/\.zip$/i.test(source)) {
      const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'smv-rtx-'));
      execFileSync(SYS_TAR, ['-xf', source, '-C', tmp, '*nvngx_vsr.dll', '*nvngx_truehdr.dll']);
      const found: string[] = [];
      const walk = (d: string) => {
        for (const e of fs.readdirSync(d, { withFileTypes: true })) {
          const p = path.join(d, e.name);
          if (e.isDirectory()) walk(p);
          else if (RTX_FEATURE_DLLS.includes(e.name)) found.push(p);
        }
      };
      walk(tmp);
      // The SDK ships arm64 + x64 (dev/rel) copies of each DLL; take the x64 release build.
      const pick = (n: string) => {
        const all = found.filter((f) => path.basename(f) === n);
        return all.find((f) => /x64[\\/]+rel/i.test(f)) || all[0];
      };
      srcFiles = RTX_FEATURE_DLLS.map(pick).filter((f): f is string => !!f);
    } else {
      const dir = findFeatureDllDir(source) || (fileExists(path.join(source, RTX_FEATURE_DLLS[0])) ? source : null);
      if (dir) srcFiles = RTX_FEATURE_DLLS.map((n) => path.join(dir, n));
    }
  } catch (e) {
    return { ok: false, error: String(e), copied: [] };
  }
  const present = srcFiles.filter(fileExists);
  if (present.length < RTX_FEATURE_DLLS.length)
    return {
      ok: false,
      error: 'nvngx_vsr.dll / nvngx_truehdr.dll not found in the selected RTX Video SDK',
      copied: [],
    };
  const copied: string[] = [];
  try {
    for (const f of present) {
      const dest = path.join(RTX_DIR, path.basename(f));
      // Overwrite any existing copy (a newer SDK release replaces the old DLLs), even one a previous
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
      error:
        'Could not overwrite the RTX DLLs in engine/rtxvideo (' +
        String(e) +
        '). If a render is running, stop it and try again.',
      copied,
    };
  }
  return { ok: true, copied };
}

// Copy NvOFFRUC.dll + cudart64_110.dll out of a chosen Optical Flow SDK .zip (or extracted folder)
// into engine/nvoffruc, beside the locally built bridge. Mirrors installRtx.
function installFruc(source: string): { ok: boolean; error?: string; copied: string[] } {
  try { fs.mkdirSync(NVOFFRUC_DIR, { recursive: true }); } catch { /* exists */ }
  let srcFiles: string[] = [];
  try {
    const found: string[] = [];
    const walk = (d: string) => { for (const e of fs.readdirSync(d, { withFileTypes: true })) {
      const p = path.join(d, e.name);
      if (e.isDirectory()) walk(p); else if (NVOFFRUC_DLLS.includes(e.name)) found.push(p);
    } };
    if (/\.zip$/i.test(source)) {
      const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'smv-fruc-'));
      execFileSync(SYS_TAR, ['-xf', source, '-C', tmp, '*NvOFFRUC.dll', '*cudart64_110.dll']);
      walk(tmp);
    } else {
      walk(source);
    }
    // The SDK ships win32 + win64 copies; take the x64 build (basename filter drops the __MACOSX ._ junk).
    const pick = (n: string) => { const all = found.filter((f) => path.basename(f) === n);
      return all.find((f) => /win64/i.test(f)) || all[0]; };
    srcFiles = NVOFFRUC_DLLS.map(pick).filter((f): f is string => !!f);
  } catch (e) { return { ok: false, error: String(e), copied: [] }; }
  const present = srcFiles.filter(fileExists);
  if (present.length < NVOFFRUC_DLLS.length)
    return { ok: false, error: 'NvOFFRUC.dll / cudart64_110.dll not found in the selected Optical Flow SDK', copied: [] };
  const copied: string[] = [];
  try {
    for (const f of present) {
      const dest = path.join(NVOFFRUC_DIR, path.basename(f));
      try { if (fileExists(dest)) fs.chmodSync(dest, 0o666); } catch { /* best effort */ }
      fs.copyFileSync(f, dest);
      copied.push(path.basename(f));
    }
  } catch (e) {
    return { ok: false, error: 'Could not write the FRUC DLLs into engine/nvoffruc (' + String(e)
      + '). If a render is running, stop it and try again.', copied };
  }
  return { ok: true, copied };
}

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

// "Nvidia Smooth Motion" (NvOFFRUC) runtime: ready only when our bridge AND NvOFFRUC.dll are present.
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
function findDvBin(root: string, name: string): string | null {
  const want = name.toLowerCase();
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
      else if (e.name.toLowerCase() === want) return p;
    }
  }
  return null;
}

// Copy a single tool .exe out of a chosen source (its release .zip, a folder, or the .exe directly)
// into an engine subdir. Zips are handled with Windows' bundled bsdtar. Shared by the Dolby Vision
// (dovi_tool) and HDR10+ (hdr10plus_tool) installers.
function installBin(
  source: string,
  binName: string,
  destDir: string,
): { ok: boolean; error?: string; copied: string[] } {
  try {
    fs.mkdirSync(destDir, { recursive: true });
  } catch {
    /* exists */
  }
  let searchDir = source;
  let tmp: string | null = null;
  try {
    if (/\.zip$/i.test(source)) {
      tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'smv-tool-'));
      execFileSync(SYS_TAR, ['-xf', source, '-C', tmp]);
      searchDir = tmp;
    } else if (!fs.statSync(source).isDirectory()) {
      searchDir = path.dirname(source); // a picked .exe: search its folder
    }
    const found = findDvBin(searchDir, binName);
    if (!found) return { ok: false, error: binName + ' not found in the selection', copied: [] };
    const dest = path.join(destDir, binName);
    try {
      if (fileExists(dest)) fs.chmodSync(dest, 0o666);
    } catch {
      /* best effort */
    }
    fs.copyFileSync(found, dest);
    return { ok: true, copied: [binName] };
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

// Live progress thumbnail: the engine overwrites this JPEG about once a second during a render
// (see SMV_LIVE_PREVIEW below); the renderer polls it by mtime and shows the frame being written.
const LIVE_JPG = path.join(app.getPath('userData'), 'preview', 'live.jpg');
ipcMain.handle('live-path', () => LIVE_JPG);

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

ipcMain.on(
  'run',
  (
    e,
    opts: {
      input: string;
      multi: number;
      output: string;
      fps?: number;
      sharpen?: number;
      restore?: boolean;
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
    },
  ) => {
    const args = ['-u', ENGINE_SCRIPT, opts.input, String(opts.multi), opts.output];
    // Output codec family (hevc default / av1 / vvc); the engine owns encoder pick + fallbacks.
    if (opts.codec && opts.codec !== 'hevc') args.push('--codec', opts.codec);
    // Interpolation is the default; interp === false means the user only wants the sharpen pass,
    // so tell the engine to skip frame generation (and ignore any fps/multi) entirely.
    if (opts.interp === false) args.push('--no-interp');
    else {
      if (opts.model === 'fruc') args.push('--fruc'); // "Nvidia Smooth Motion" backend instead of GMFSS
      if (opts.fps && opts.fps > 0) args.push('--fps', String(opts.fps));
    }
    // FSR-style RCAS sharpening strength (GUI checkbox + slider). 0/omitted = off, leaving the
    // frames value-preserving; >0 enables the in-engine RCAS pass. Works with or without interp.
    if (opts.sharpen && opts.sharpen > 0) args.push('--sharpen', String(opts.sharpen));
    // AI detail restoration (GUI Restore checkbox): Real-ESRGAN animevideov3 on every output
    // frame, before the upscale. Works with or without interpolation.
    if (opts.restore) args.push('--restore');
    // Upscale factor (an arbitrary float, source height -> chosen target height), computed by the
    // renderer from the resolution selector. >1 enables the upscale pass. Without --rtx-vsr this is a
    // bicubic resize; with it, RTX Video Super Resolution (any target resolution, no integer-scale
    // limit), which the engine degrades to bicubic if the RTX runtime is absent.
    if (opts.upscale && opts.upscale > 1) args.push('--upscale', String(opts.upscale));
    // RTX VSR: use the real RTX Video SDK (the engine/rtxvideo CUDA bridge) for the upscale step.
    // Only meaningful alongside --upscale (it supplies the target resolution). Falls back to bicubic
    // if the bridge or RTX Video runtime is unavailable.
    if (opts.rtxvsr && opts.upscale && opts.upscale > 1) args.push('--rtx-vsr');
    // RTX HDR (TrueHDR): convert the output to HDR10. Works with or without --upscale (when both are
    // on, the RTX bridge does VSR then TrueHDR in one pass). The engine masters at a fixed 1000-nit
    // peak (its --hdr-nits default) and writes the HDR10 metadata, so there is no per-display nits
    // knob; it falls back to an SDR render if the bridge is unavailable.
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
    // PYTHONUTF8 keeps the dynamo ONNX exporter's unicode logs from crashing the engine
    // during first-run TRT builds; SMV_TRT_CACHE is a guaranteed writable cache location.
    // SMV_LIVE_PREVIEW makes the engine drop a small JPEG of the frame being written about
    // once a second; the renderer polls it for the live progress thumbnail.
    try {
      fs.mkdirSync(path.join(app.getPath('userData'), 'preview'), { recursive: true });
    } catch {
      /* exists */
    }
    const env = {
      ...process.env,
      PYTHONUTF8: '1',
      SMV_TRT_CACHE: path.join(app.getPath('userData'), 'trt_cache'),
      SMV_LIVE_PREVIEW: LIVE_JPG,
      SMV_PAUSE_FILE: PAUSE_FLAG,
      SMV_LIVE_OFF_FILE: LIVE_OFF_FLAG,
    };
    clearPause(); // start unpaused: never inherit a stale flag from a previous (e.g. killed) run
    const proc = spawn(pyExe(), args, { cwd: ENGINE, env });
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
    proc.stdout.on('data', onStdout);
    proc.stderr.on('data', onData);
    proc.on('close', (code) => {
      const tail = stripGridSize(outCarry);
      outCarry = '';
      if (tail) send('engine-out', tail);
      current = null;
      clearPause();
      keepAwake(false);
      taskbarProgress(-1);
      send('engine-done', code);
    });
    proc.on('error', (err) => {
      current = null;
      clearPause();
      keepAwake(false);
      taskbarProgress(-1);
      send('engine-out', 'spawn error: ' + err);
      send('engine-done', -1);
    });
  },
);

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
// resume block in gmfss_interp.py). The renderer uses this to flip Smooth It! to Resume and to
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
// for the renderer's side-by-side pane. preview.py writes <prefix>_original.png and _processed.png; a
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
      nits?: number;
      color?: string;
      saturation?: number;
      vibrance?: number;
      satboost?: number;
      contrast?: number;
      upscale?: number;
      rtxvsr?: boolean;
      restore?: boolean;
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
      const args = ['-u', PREVIEW_SCRIPT, opts.input, '--out', prefix, '--frame', String(opts.frame ?? 'mid')];
      if (opts.sharpen && opts.sharpen > 0) args.push('--sharpen', String(opts.sharpen));
      if (opts.restore) args.push('--restore');
      if (opts.upscale && opts.upscale > 1) {
        args.push('--upscale', String(opts.upscale));
        if (opts.rtxvsr) args.push('--rtx-vsr');
      }
      if (opts.hdr)
        args.push(
          '--rtx-hdr',
          '--hdr-nits',
          String(opts.nits ?? 1000),
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
      const env = { ...process.env, PYTHONUTF8: '1' };
      execFile(pyExe(), args, { cwd: ENGINE, env }, (err, stdout, stderr) => {
        if (err) {
          resolve({ error: String(stderr || err).slice(-400) });
          return;
        }
        const m = /frame (\d+)\/(\d+)/.exec(String(stdout ?? ''));
        resolve({
          original: prefix + '_original.png',
          processed: prefix + '_processed.png',
          frame: m ? Number(m[1]) : null,
          total: m ? Number(m[2]) : null,
        });
      });
    });
  },
);
