// Main process: the window, its saved size and its creation.
import { app, BrowserWindow } from 'electron';
import * as path from 'path';
import * as fs from 'fs';
import { ROOT } from './main-paths';
import { current } from './main-render';

export let win: BrowserWindow | null = null;

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

export function createWindow() {
  const st = loadWindowSize();
  win = new BrowserWindow({
    width: st.width,
    height: st.height,
    minWidth: 680,
    minHeight: 640,
    title: 'Smooth My Video',
    backgroundColor: '#1b1b1b',
    icon: path.join(ROOT, 'assets', 'icon.ico'),
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
