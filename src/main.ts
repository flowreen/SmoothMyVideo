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
    title: 'SmoothMyVideo',
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
  });
}
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});
app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});

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

let current: ChildProcess | null = null;
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
      upscale?: number;
      rtxvsr?: boolean;
      rtxhdr?: boolean;
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
    proc.stdout.on('data', onData);
    proc.stderr.on('data', onData);
    proc.on('close', (code) => {
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

ipcMain.on('cancel', () => {
  const c = current;
  if (c && c.pid) {
    execFile('taskkill', ['/pid', String(c.pid), '/T', '/F'], () => {});
  }
});

// The renderer fires this when a job (or the whole batch) finishes; show a native notification only when
// the window is unfocused, so someone who tabbed away during a long render is told it is done.
ipcMain.on('render-complete', (_e, body: string) => {
  if (win && !win.isFocused() && Notification.isSupported()) {
    try {
      new Notification({ title: 'SmoothMyVideo', body: body || 'Render complete' }).show();
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
