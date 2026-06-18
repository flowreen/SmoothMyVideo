import { app, BrowserWindow, ipcMain, dialog } from 'electron';
import { spawn, execFile, ChildProcess } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';

const ROOT = path.join(__dirname, '..');
const ENGINE = path.join(ROOT, 'engine');
const VENV_PY = path.join(ENGINE, '.venv', 'Scripts', 'python.exe');
const ENGINE_SCRIPT = path.join(ENGINE, 'gmfss_interp.py');

function pyExe(): string {
  return fs.existsSync(VENV_PY) ? VENV_PY : 'python';
}

let win: BrowserWindow | null = null;

function createWindow() {
  win = new BrowserWindow({
    width: 780,
    height: 760,
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
    execFile('ffprobe', ['-v', 'error', '-select_streams', 'v:0',
      '-show_entries', 'stream=width,height,r_frame_rate,codec_name,nb_frames',
      '-show_entries', 'format=duration', '-of', 'json', file],
      (err, stdout) => {
        if (err) { resolve({ error: String(err) }); return; }
        try { resolve(JSON.parse(stdout)); } catch { resolve({ error: 'probe parse failed' }); }
      });
  });
});

let current: ChildProcess | null = null;

ipcMain.on('run', (e, opts: { input: string; multi: number; output: string }) => {
  const args = ['-u', ENGINE_SCRIPT, opts.input, String(opts.multi), opts.output];
  const proc = spawn(pyExe(), args, { cwd: ENGINE });
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
