// Stamp icon.ico into the dev electron.exe so `npm start` shows the SmoothMyVideo icon in the
// Windows taskbar. The window icon (BrowserWindow `icon`) and the packaged exe (electron-builder
// `win.icon`) were always right; the taskbar button of a DEV launch falls back to the process
// executable's icon - stock electron.exe's Electron logo - because our AppUserModelID has no
// registered Start Menu shortcut. Runs from postinstall (after electron's own postinstall has
// downloaded the binary), so a fresh `npm run setup` reapplies it. Best-effort: never fails install.
const path = require('path');
const fs = require('fs');

(async () => {
  if (process.platform !== 'win32') return;
  const exe = path.join(__dirname, '..', 'node_modules', 'electron', 'dist', 'electron.exe');
  const ico = path.join(__dirname, '..', 'icon.ico');
  if (!fs.existsSync(exe)) {
    // electron >= 43.2 has no postinstall of its own: the binary downloads lazily on first
    // launch. Fetch it now (same script index.js runs) so setup ends fully provisioned and
    // the icon stamp below has an exe to patch.
    const installJs = path.join(__dirname, '..', 'node_modules', 'electron', 'install.js');
    if (!fs.existsSync(installJs)) return;
    const r = require('child_process').spawnSync(process.execPath, [installJs], { stdio: 'inherit' });
    if (r.status !== 0 || !fs.existsSync(exe)) return;
  }
  if (!fs.existsSync(ico)) return;
  try {
    // rcedit 5 is pure ESM with a named export; rcedit 4 was a CJS function export
    const mod = await import('rcedit');
    await (mod.rcedit || mod.default)(exe, { icon: ico });
    console.log('[dev-icon] stamped icon.ico into dev electron.exe');
  } catch (e) {
    console.warn(`[dev-icon] skipped (${e.message}); dev taskbar keeps the Electron logo`);
  }
})();
