// Stamp icon.ico into the dev electron.exe so `npm start` shows the SmoothMyVideo icon in the
// Windows taskbar. The window icon (BrowserWindow `icon`) and the packaged exe (electron-builder
// `win.icon`) were always right; the taskbar button of a DEV launch falls back to the process
// executable's icon - stock electron.exe's Electron logo - because our AppUserModelID has no
// registered Start Menu shortcut. Runs from postinstall (after electron's own postinstall has
// downloaded the binary), so a fresh `npm run setup` reapplies it. Best-effort: never fails install.
// Also makes "Smooth My Video.lnk" in the repo root: the app icon on a double-click shortcut to SmoothMyVideo.cmd (a .cmd
// cannot carry an icon of its own, and a shortcut holds absolute paths, so each clone makes its own; git-ignored).
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

async function stamp(): Promise<void> {
  if (process.platform !== 'win32') return;
  const exe = path.join(import.meta.dirname, '..', 'node_modules', 'electron', 'dist', 'electron.exe');
  const ico = path.join(import.meta.dirname, '..', 'assets', 'icon.ico');
  if (!fs.existsSync(exe)) {
    // electron >= 43.2 has no postinstall of its own: the binary downloads lazily on first
    // launch. Fetch it now (same script index.js runs) so setup ends fully provisioned and
    // the icon stamp below has an exe to patch.
    const installJs = path.join(import.meta.dirname, '..', 'node_modules', 'electron', 'install.js');
    if (!fs.existsSync(installJs)) return;
    const r = spawnSync(process.execPath, [installJs], { stdio: 'inherit' });
    if (r.status !== 0 || !fs.existsSync(exe)) return;
  }
  if (!fs.existsSync(ico)) return;
  try {
    const { rcedit } = await import('rcedit');
    await rcedit(exe, { icon: ico });
    console.log('[dev-icon] stamped icon.ico into dev electron.exe');
  } catch (e) {
    console.warn(`[dev-icon] skipped (${(e as Error).message}); dev taskbar keeps the Electron logo`);
  }
}

// the shortcut through Windows' own shortcut object (PowerShell, paths in the environment so any folder name works);
// minimized, so the launcher's console never shows
function shortcut(): void {
  if (process.platform !== 'win32') return;
  const root = path.join(import.meta.dirname, '..');
  const ps =
    '$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:LNK_PATH); $s.TargetPath = $env:LNK_TARGET; ' +
    "$s.WorkingDirectory = $env:LNK_DIR; $s.IconLocation = $env:LNK_ICON + ',0'; $s.WindowStyle = 7; " +
    "$s.Description = 'Smooth My Video'; $s.Save()";
  const r = spawnSync('powershell', ['-NoProfile', '-NonInteractive', '-Command', ps], {
    env: {
      ...process.env,
      LNK_PATH: path.join(root, 'Smooth My Video.lnk'),
      LNK_TARGET: path.join(root, 'SmoothMyVideo.cmd'),
      LNK_DIR: root,
      LNK_ICON: path.join(root, 'assets', 'icon.ico'),
    },
    stdio: 'ignore',
  });
  if (r.status === 0) console.log('[dev-icon] made "Smooth My Video.lnk" (the app icon) beside SmoothMyVideo.cmd');
  else console.warn(`[dev-icon] no shortcut (powershell rc ${r.status}); double-click SmoothMyVideo.cmd`);
}

await stamp();
shortcut();
