import { app, BrowserWindow, ipcMain, dialog, screen } from 'electron';
import { spawn, execFile, ChildProcess } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';

const ROOT = path.join(__dirname, '..');
// When packaged, the engine ships as an unpacked extraResource (the Python files and
// runtime must be real files on disk, not inside app.asar). The renderer and icon stay
// under ROOT (Electron reads those from the asar fine).
const ENGINE = app.isPackaged ? path.join(process.resourcesPath, 'engine') : path.join(ROOT, 'engine');
// Bundled, relocatable python-build-standalone runtime (full stdlib + torch/cupy stack).
// Its python.exe sits at the runtime root, not under a Scripts/ subdir like a venv.
const RUNTIME_PY = path.join(ENGINE, 'runtime', 'python.exe');
const ENGINE_SCRIPT = path.join(ENGINE, 'gmfss_interp.py');
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

app.whenReady().then(createWindow);
app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });
app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });

ipcMain.handle('pick-video', async (_e, defaultPath?: string) => {
  const r = await dialog.showOpenDialog(win!, {
    defaultPath,
    properties: ['openFile'],
    filters: [{ name: 'Video', extensions: ['mp4', 'mkv', 'mov', 'avi', 'webm', 'm4v', 'wmv', 'ts'] }],
  });
  return r.canceled ? null : r.filePaths[0];
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
      '-show_entries', 'stream=width,height,r_frame_rate,codec_name,nb_frames',
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

let current: ChildProcess | null = null;

ipcMain.on('run', (e, opts: { input: string; multi: number; output: string; fps?: number; sharpen?: number; interp?: boolean }) => {
  const args = ['-u', ENGINE_SCRIPT, opts.input, String(opts.multi), opts.output];
  // Interpolation is the default; interp === false means the user only wants the sharpen pass,
  // so tell the engine to skip frame generation (and ignore any fps/multi) entirely.
  if (opts.interp === false) args.push('--no-interp');
  else if (opts.fps && opts.fps > 0) args.push('--fps', String(opts.fps));
  // FSR-style RCAS sharpening strength (GUI checkbox + slider). 0/omitted = off, leaving the
  // frames value-preserving; >0 enables the in-engine RCAS pass. Works with or without interp.
  if (opts.sharpen && opts.sharpen > 0) args.push('--sharpen', String(opts.sharpen));
  // PYTHONUTF8 keeps the dynamo ONNX exporter's unicode logs from crashing the engine
  // during first-run TRT builds; SMV_TRT_CACHE is a guaranteed writable cache location.
  const env = { ...process.env, PYTHONUTF8: '1',
    SMV_TRT_CACHE: path.join(app.getPath('userData'), 'trt_cache') };
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
