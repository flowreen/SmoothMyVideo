// Main process: the file dialogs, the source probe, the display's refresh rate, peak brightness and size.
import { ipcMain, dialog, screen } from 'electron';
import { execFile, execFileSync } from 'child_process';
import { FFPROBE } from './main-paths';
import { win } from './main-window';
import { LIVE_DIR, LIVE_EXE } from './main-live';

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
// integer-only on Windows, so the exact rational comes from QueryDisplayConfig through the host's
// `smv-live.exe --hz` (one "numerator denominator" line per active display path), matched to the
// Electron display by nearest integer. Cached: rates only change on display mode switches.
let exactHz: number[] = [];
let exactHzAt = 0;
function queryExactRates(): Promise<number[]> {
  return new Promise((resolve) => {
    execFile(LIVE_EXE, ['--hz'], { cwd: LIVE_DIR, timeout: 8000, windowsHide: true }, (err, stdout) => {
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
    });
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

// The peak (nits) of the HDR display the app window is on, from `smv-live.exe --peak-nits X Y` (Windows' Max
// Luminance, the level NVIDIA's own RTX Video HDR expands to); 1000 when that display is not in HDR mode (the usual
// HDR10 master). File renders and the preview pass it as --hdr-nits, so RTX HDR targets the user's screen.
export function displayPeakNits(): number {
  try {
    const b = win ? win.getBounds() : screen.getPrimaryDisplay().bounds;
    const p = screen.dipToScreenPoint({ x: Math.round(b.x + b.width / 2), y: Math.round(b.y + b.height / 2) });
    const out = execFileSync(LIVE_EXE, ['--peak-nits', String(p.x), String(p.y)], { cwd: LIVE_DIR, timeout: 8000 });
    const n = parseInt(out.toString(), 10);
    return n > 0 ? n : 1000;
  } catch {
    return 1000;
  }
}
ipcMain.handle('peak-nits', () => displayPeakNits());

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
