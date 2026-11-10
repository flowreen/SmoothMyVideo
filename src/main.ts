// The main process's entry: one instance only, the start-up and the app's lifecycle. Everything else lives in the
// main-<name>.ts modules beside this file, one concern each; they import each other's names.
import { app, BrowserWindow } from 'electron';
import * as path from 'path';
import { checkEngineCache } from './render/cache';
import { ENGINE } from './main-paths';
import { createWindow, win } from './main-window';
import { applyPendingUpdate, checkForUpdate, finishUpdate } from './main-update';
import './main-dialogs';
import { retireOldRuntimes } from './main-setup';
import { fsrWarmProc, liveSpareDrop, offlineHostQuit, stopLive } from './main-live';
import './main-render';
import './main-preview';

// Single instance only. A second launch shares this profile dir, and Chromium's disk/GPU cache and
// Local Storage locks (held by the first instance) make the second window come up empty ("Unable to
// move the cache: Access is denied", "Gpu Cache Creation failed"); it would also fight the first
// instance over the preview PNGs and the model cache. So a second launch focuses the running window.
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
    // a newer version unpacked by an earlier launch: this launch applies it instead of opening (applyPendingUpdate)
    if (applyPendingUpdate()) return;
    // an earlier version's TensorRT engines leave the model cache, before any live session or
    // render starts; a cache an earlier version kept elsewhere (the engine folder, 1.0.x's AppData)
    // is moved or removed (src/render/cache.ts)
    console.log(checkEngineCache(ENGINE, [path.join(app.getPath('userData'), 'trt_cache')]));
    // superseded runtimes (DLSS 5's retired builds, an RTX Video file that is not the pinned one) go before
    // any host can load them
    for (const line of retireOldRuntimes()) console.log(line);
    createWindow();
    finishUpdate();
    checkForUpdate();
  });
}
app.on('window-all-closed', () => {
  stopLive(true); // never leave a headless FG overlay (or an idle resident host) running after the GUI is gone
  liveSpareDrop();
  offlineHostQuit();
  fsrWarmProc?.kill();
  if (process.platform !== 'darwin') app.quit();
});
app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
