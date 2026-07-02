import { app, BrowserWindow, ipcMain, dialog, screen, shell } from 'electron';
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
  ? path.join(ENGINE, 'bin', 'ffprobe.exe') : 'ffprobe';

function pyExe(): string {
  return fs.existsSync(RUNTIME_PY) ? RUNTIME_PY : 'python';
}

let win: BrowserWindow | null = null;

function createWindow() {
  win = new BrowserWindow({
    width: 780,
    height: 700,
    title: 'SmoothMyVideo',
    backgroundColor: '#1b1b1b',
    icon: path.join(ROOT, 'icon.ico'),
    webPreferences: { nodeIntegration: true, contextIsolation: false },
  });
  win.setMenuBarVisibility(false);
  win.loadFile(path.join(ROOT, 'renderer', 'index.html'));
}

// Single instance only. A second launch shares this profile dir, and Chromium's disk/GPU cache and
// Local Storage locks (held by the first instance) make the second window come up empty ("Unable to
// move the cache: Access is denied", "Gpu Cache Creation failed"); it would also fight the first
// instance over the preview PNGs and the TRT cache. So a second launch focuses the running window.
if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (win) { if (win.isMinimized()) win.restore(); win.show(); win.focus(); }
  });
  app.whenReady().then(createWindow);
}
app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });
app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });

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
  const r = await dialog.showSaveDialog(win!, {
    defaultPath,
    filters: [{ name: 'MP4 video', extensions: ['mp4'] }],
  });
  return r.canceled ? null : (r.filePath || null);
});

ipcMain.handle('probe', async (_e, file: string) => {
  return new Promise((resolve) => {
    execFile(FFPROBE, ['-v', 'error', '-select_streams', 'v:0',
      '-show_entries', 'stream=width,height,r_frame_rate,codec_name,nb_frames,color_transfer',
      '-show_entries', 'format=duration', '-of', 'json', file],
      (err, stdout) => {
        if (err) { resolve({ error: String(err) }); return; }
        try { resolve(JSON.parse(stdout)); } catch { resolve({ error: 'probe parse failed' }); }
      });
  });
});

ipcMain.handle('refresh-rate', () => {
  // Refresh rate of the monitor the app window is on (fallback: primary), rounded up so a
  // 59.94 / 143.9 Hz panel targets a clean 60 / 144. Feeds the renderer's "match screen" option.
  try {
    const d = win ? screen.getDisplayMatching(win.getBounds()) : screen.getPrimaryDisplay();
    return Math.ceil(d.displayFrequency || screen.getPrimaryDisplay().displayFrequency || 60);
  } catch { return 60; }
});

ipcMain.handle('screen-size', () => {
  // Physical pixel resolution of the monitor the window is on (fallback: primary). display.size is
  // in logical DIPs, so multiply by the scale factor to get the real panel resolution. Feeds the
  // renderer's "RTX Video Super Resolution: upscale to screen" target.
  try {
    const d = win ? screen.getDisplayMatching(win.getBounds()) : screen.getPrimaryDisplay();
    const f = d.scaleFactor || 1;
    return { width: Math.round(d.size.width * f), height: Math.round(d.size.height * f) };
  } catch { return { width: 0, height: 0 }; }
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
const fileExists = (p: string) => { try { return fs.existsSync(p); } catch { return false; } };

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
  } catch { /* unreadable root */ }
  return seeds.find(hasBoth) || null;
}

// Look in the usual download spots for an extracted SDK folder or a recognizable SDK .zip.
function scanForSdk(): { folder: string | null; zip: string | null } {
  const roots: string[] = [];
  for (const k of ['downloads', 'desktop', 'home'] as const) { try { roots.push(app.getPath(k)); } catch { /* none */ } }
  let folder: string | null = null;
  for (const r of roots) { folder = findFeatureDllDir(r); if (folder) break; }
  let zip: string | null = null;
  for (const r of roots) {
    try {
      const hit = fs.readdirSync(r, { withFileTypes: true })
        .find((e) => e.isFile() && /\.zip$/i.test(e.name) && /rtx.*video.*sdk/i.test(e.name));
      if (hit) { zip = path.join(r, hit.name); break; }
    } catch { /* unreadable root */ }
  }
  return { folder, zip };
}

// Copy the two feature DLLs out of a chosen source (an extracted SDK folder or an SDK .zip) into
// engine/rtxvideo. Zips are handled with Windows' bundled bsdtar, extracting only the two members.
function installRtx(source: string): { ok: boolean; error?: string; copied: string[] } {
  try { fs.mkdirSync(RTX_DIR, { recursive: true }); } catch { /* exists */ }
  let srcFiles: string[] = [];
  try {
    if (/\.zip$/i.test(source)) {
      const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'smv-rtx-'));
      execFileSync(SYS_TAR, ['-xf', source, '-C', tmp, '*nvngx_vsr.dll', '*nvngx_truehdr.dll']);
      const found: string[] = [];
      const walk = (d: string) => { for (const e of fs.readdirSync(d, { withFileTypes: true })) {
        const p = path.join(d, e.name);
        if (e.isDirectory()) walk(p); else if (RTX_FEATURE_DLLS.includes(e.name)) found.push(p);
      } };
      walk(tmp);
      // The SDK ships arm64 + x64 (dev/rel) copies of each DLL; take the x64 release build.
      const pick = (n: string) => { const all = found.filter((f) => path.basename(f) === n);
        return all.find((f) => /x64[\\/]+rel/i.test(f)) || all[0]; };
      srcFiles = RTX_FEATURE_DLLS.map(pick).filter((f): f is string => !!f);
    } else {
      const dir = findFeatureDllDir(source) || (fileExists(path.join(source, RTX_FEATURE_DLLS[0])) ? source : null);
      if (dir) srcFiles = RTX_FEATURE_DLLS.map((n) => path.join(dir, n));
    }
  } catch (e) { return { ok: false, error: String(e), copied: [] }; }
  const present = srcFiles.filter(fileExists);
  if (present.length < RTX_FEATURE_DLLS.length)
    return { ok: false, error: 'nvngx_vsr.dll / nvngx_truehdr.dll not found in the selected RTX Video SDK', copied: [] };
  const copied: string[] = [];
  try {
    for (const f of present) {
      const dest = path.join(RTX_DIR, path.basename(f));
      // Overwrite any existing copy (a newer SDK release replaces the old DLLs), even one a previous
      // extraction left read-only; clearing the flag first avoids an EPERM on copy.
      try { if (fileExists(dest)) fs.chmodSync(dest, 0o666); } catch { /* best effort */ }
      fs.copyFileSync(f, dest);
      copied.push(path.basename(f));
    }
  } catch (e) {
    return { ok: false, error: 'Could not overwrite the RTX DLLs in engine/rtxvideo (' + String(e)
      + '). If a render is running, stop it and try again.', copied };
  }
  return { ok: true, copied };
}

ipcMain.handle('rtx-ready', () => {
  const bridge = fileExists(path.join(RTX_DIR, 'rtxvideo_cuda.dll'));
  return {
    vsr: bridge && fileExists(path.join(RTX_DIR, 'nvngx_vsr.dll')),
    hdr: bridge && fileExists(path.join(RTX_DIR, 'nvngx_truehdr.dll')),
    bridge, dir: RTX_DIR,
  };
});

ipcMain.handle('rtx-open-download', () => { shell.openExternal(RTX_SDK_URL); return true; });

// Install from a given source (the picked .zip), or auto-detect one when none is passed.
ipcMain.handle('rtx-install', (_e, source?: string) => {
  let src = source;
  if (!src) { const s = scanForSdk(); src = s.folder || s.zip || undefined; }
  if (!src) return { ok: false, error: 'No RTX Video SDK found in Downloads/Desktop. Use "Get from NVIDIA", then "Choose..."', copied: [] };
  return installRtx(src);
});

// Manual picker fallback: a folder (extracted SDK) or a .zip.
ipcMain.handle('rtx-choose', async (_e, mode: 'dir' | 'zip') => {
  const r = await dialog.showOpenDialog(win!, mode === 'dir'
    ? { title: 'Select the extracted RTX Video SDK folder', properties: ['openDirectory'] }
    : { title: 'Select the RTX Video SDK .zip', properties: ['openFile'], filters: [{ name: 'Zip', extensions: ['zip'] }] });
  return r.canceled ? null : (r.filePaths[0] || null);
});

let current: ChildProcess | null = null;

// Live progress thumbnail: the engine overwrites this JPEG about once a second during a render
// (see SMV_LIVE_PREVIEW below); the renderer polls it by mtime and shows the frame being written.
const LIVE_JPG = path.join(app.getPath('userData'), 'preview', 'live.jpg');
ipcMain.handle('live-path', () => LIVE_JPG);

ipcMain.on('run', (e, opts: { input: string; multi: number; output: string; fps?: number; sharpen?: number; interp?: boolean; upscale?: number; rtxvsr?: boolean; rtxhdr?: boolean; codec?: string; hdrcolor?: string; hdrsat?: number; hdrcon?: number; hdrsb?: number; hdrvib?: number }) => {
  const args = ['-u', ENGINE_SCRIPT, opts.input, String(opts.multi), opts.output];
  // Output codec family (hevc default / av1 / vvc); the engine owns encoder pick + fallbacks.
  if (opts.codec && opts.codec !== 'hevc') args.push('--codec', opts.codec);
  // Interpolation is the default; interp === false means the user only wants the sharpen pass,
  // so tell the engine to skip frame generation (and ignore any fps/multi) entirely.
  if (opts.interp === false) args.push('--no-interp');
  else if (opts.fps && opts.fps > 0) args.push('--fps', String(opts.fps));
  // FSR-style RCAS sharpening strength (GUI checkbox + slider). 0/omitted = off, leaving the
  // frames value-preserving; >0 enables the in-engine RCAS pass. Works with or without interp.
  if (opts.sharpen && opts.sharpen > 0) args.push('--sharpen', String(opts.sharpen));
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
  try { fs.mkdirSync(path.join(app.getPath('userData'), 'preview'), { recursive: true }); } catch { /* exists */ }
  const env = { ...process.env, PYTHONUTF8: '1',
    SMV_TRT_CACHE: path.join(app.getPath('userData'), 'trt_cache'),
    SMV_LIVE_PREVIEW: LIVE_JPG };
  const proc = spawn(pyExe(), args, { cwd: ENGINE, env });
  current = proc;
  const onData = (buf: Buffer) => e.sender.send('engine-out', buf.toString());
  proc.stdout.on('data', onData);
  proc.stderr.on('data', onData);
  proc.on('close', (code) => { current = null; e.sender.send('engine-done', code); });
  proc.on('error', (err) => { current = null; e.sender.send('engine-out', 'spawn error: ' + err); e.sender.send('engine-done', -1); });
});

ipcMain.on('cancel', () => {
  const c = current;
  if (c && c.pid) { execFile('taskkill', ['/pid', String(c.pid), '/T', '/F'], () => {}); }
});

// Before/after preview: render ONE source frame at the current spatial settings (RTX HDR when opts.hdr,
// FSR/CAS sharpen when opts.sharpen > 0; no interpolation, no encode) and hand back the two PNG paths
// for the renderer's side-by-side pane. preview.py writes <prefix>_original.png and _processed.png; a
// fixed prefix is reused each call (the renderer cache-busts its img src) so previews never pile up.
// Resolves with { error } instead when the frame or the RTX bridge is unavailable.
ipcMain.handle('preview', (_e, opts: { input: string; frame?: number | string; sharpen?: number;
    hdr?: boolean; nits?: number; color?: string; saturation?: number; vibrance?: number; satboost?: number; contrast?: number;
    upscale?: number; rtxvsr?: boolean }) => {
  return new Promise((resolve) => {
    const dir = path.join(app.getPath('userData'), 'preview');
    try { fs.mkdirSync(dir, { recursive: true }); } catch { /* already exists */ }
    const prefix = path.join(dir, 'frame');
    const args = ['-u', PREVIEW_SCRIPT, opts.input, '--out', prefix, '--frame', String(opts.frame ?? 'mid')];
    if (opts.sharpen && opts.sharpen > 0) args.push('--sharpen', String(opts.sharpen));
    if (opts.upscale && opts.upscale > 1) {
      args.push('--upscale', String(opts.upscale));
      if (opts.rtxvsr) args.push('--rtx-vsr');
    }
    if (opts.hdr) args.push('--rtx-hdr', '--hdr-nits', String(opts.nits ?? 1000),
      '--hdr-color', String(opts.color ?? 'vivid'),
      '--hdr-saturation', String(opts.saturation ?? 0), '--hdr-vibrance', String(opts.vibrance ?? 0),
      '--hdr-satboost', String(opts.satboost ?? 0), '--hdr-contrast', String(opts.contrast ?? 100));
    const env = { ...process.env, PYTHONUTF8: '1' };
    execFile(pyExe(), args, { cwd: ENGINE, env }, (err, stdout, stderr) => {
      if (err) { resolve({ error: String(stderr || err).slice(-400) }); return; }
      const m = /frame (\d+)\/(\d+)/.exec(String(stdout ?? ''));
      resolve({ original: prefix + '_original.png', processed: prefix + '_processed.png',
        frame: m ? Number(m[1]) : null, total: m ? Number(m[2]) : null });
    });
  });
});
