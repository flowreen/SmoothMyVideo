import {
  app,
  BrowserWindow,
  ipcMain,
  dialog,
  screen,
  shell,
  powerSaveBlocker,
  Notification,
  globalShortcut,
  net,
} from 'electron';
import { spawn, execFile, execFileSync, ChildProcess } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import * as crypto from 'crypto';
import * as ofs from 'original-fs';
import { Readable } from 'stream';
import { pipeline } from 'stream/promises';
import { checkEngineCache, engineCacheDir } from './render/cache';

const ROOT = path.join(__dirname, '..');
// When packaged, the engine ships as an unpacked extraResource (the host exe, its DLLs and
// the ONNX must be real files on disk, not inside app.asar). The renderer and icon stay
// under ROOT (Electron reads those from the asar fine).
const ENGINE = app.isPackaged ? path.join(process.resourcesPath, 'engine') : path.join(ROOT, 'engine');
// The render orchestrator and the before/after preview, compiled next to this file.
const RENDER_CLI = path.join(__dirname, 'render', 'cli.js');
const PREVIEW_CLI = path.join(__dirname, 'render', 'preview.js');
// Prefer ffprobe bundled at engine/bin (portable build); fall back to PATH for dev.
const FFPROBE = fs.existsSync(path.join(ENGINE, 'bin', 'ffprobe.exe'))
  ? path.join(ENGINE, 'bin', 'ffprobe.exe')
  : 'ffprobe';
// The real cold-start when a video is selected is the before/after PREVIEW: any enabled pass starts the
// native host, which creates a CUDA context. We deliberately do NOT warm that at launch: on hybrid-GPU
// laptops the CUDA-context creation saturates the RTX and stalls Chromium's GPU compositor for several
// seconds, so a user who selects a file during that window sees the (instantly-probed, ~100ms) file
// info fail to paint until the warmup finishes - the exact "video info loads slowly" symptom. The first
// preview instead pays its own cold start behind its own spinner, which never blocks the file info. The
// preview spawn is also fenced behind a composited frame in the renderer (see loadVideo) so the info is
// always on screen before the host starts.

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
    // a newer version unpacked by an earlier launch: this launch applies it instead of opening (applyPendingUpdate)
    if (applyPendingUpdate()) return;
    // an earlier version's TensorRT engines leave the model cache, before any live session or
    // render starts; a cache an earlier version kept elsewhere (the engine folder, 1.0.x's AppData)
    // is moved or removed (src/render/cache.ts)
    console.log(checkEngineCache(ENGINE, [path.join(app.getPath('userData'), 'trt_cache')]));
    // a retired DLSS 5 runtime (DLSSNR_RETIRED) goes before any host can load it
    const nr = retireOldDlssnr();
    if (nr) console.log(nr);
    createWindow();
    finishUpdate();
    checkForUpdate();
  });
}
app.on('window-all-closed', () => {
  stopLive(true); // never leave a headless FG overlay (or an idle resident host) running after the GUI is gone
  offlineHostQuit();
  fsrWarmProc?.kill();
  if (process.platform !== 'darwin') app.quit();
});
app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});

// --- Update check and automatic update -----------------------------------------------------------------------------
// One GitHub releases/latest fetch a few seconds after start (never in the way of the preview-engine warm-up), silent
// on any failure (offline, rate limit, no release). A newer tag shows the renderer's one-line notice and, in a packaged
// copy with the notice's Automatic updates tick on, first checks what an update needs here (milliseconds, before any
// byte is downloaded: the copy's release-files.txt, the folder taking new files and letting go of its own, the free
// space, the bundled 7zr.exe unpacking a probe archive packed as the release is, no two failures of this version);
// anything missing = the notice with the reason and the tick greyed. Otherwise the release archive downloads in the
// background into <app folder>\update\<version>\ (a cut download resumes at the next launch), its SHA256 is checked
// against the notes' `sha256(win.7z):` line, 7zr.exe unpacks it there, and every file is checked against the new
// version's release-files.txt before the tree is marked ready. The swap runs the way Chrome's does: the big Update
// button restarts into the new version at once (a running render pauses first, a Live session stops), otherwise the
// next launch applies it before a window opens. The NEW version's native helper (engine\update\smv-update.exe, copied to %TEMP%)
// does the swap once this process has exited: it replaces only the files the old version's list names, so the user's
// own files in the folder (videos, renders, the one-click downloads, the model cache) stay where they are; the new
// version removes update\<version>\ (the old files) at its start. Levers: SMV_AUTO_UPDATE=0 (the notice only),
// SMV_UPDATE_API=URL (another latest-release JSON: the gates' local server), SMV_UPDATE_TRIES=N (the helper's tries a
// file), SMV_UPDATE_7ZR=PATH (another unpacker: the gates' failing-probe case).
const UPDATE_REPO = 'flowreen/SmoothMyVideo';
const UPDATE_API = process.env.SMV_UPDATE_API || `https://api.github.com/repos/${UPDATE_REPO}/releases/latest`;
const UPDATE_DIR = 'update'; // under the app folder: <version>\<archive>, <version>\x\<the unpacked tree>
const UPDATE_READY = '.ready'; // in x\ once the tree is complete
const UPDATE_FAILED = 'failed.txt'; // the helper's note that the swap did not happen: the next launch opens normally
const UPDATE_FAILS = 'failed.json'; // { version, count, why }: a download or unpack that failed twice is not tried again
const RELEASE_LIST = path.join('resources', 'release-files.txt'); // "<sha256> <size> <path>" a line (release-files.ts)
const UPDATE_7ZR = () => process.env.SMV_UPDATE_7ZR || path.join(ENGINE, 'update', '7zr.exe');
const UPDATE_PROBE = path.join(ENGINE, 'update', 'probe.7z'); // smv-update.exe packed as the release is (build.bat)
type UpdateWhy = 'list' | 'locked' | 'space' | 'unpack' | 'failed'; // the renderer's update.why.* texts
type ListEntry = { sha: string; size: number; rel: string };
type Release = {
  tag_name?: string;
  html_url?: string;
  body?: string;
  assets?: Array<{ name?: string; size?: number; browser_download_url?: string }>;
};
let autoUpdateOn = true; // the renderer's Automatic updates tick ('update-auto'); its default
let updateAbort: AbortController | null = null; // the download or unpack under way
let updateReady: { version: string; rel: string } | null = null; // an unpacked tree, relative to the app folder
// the app folder of a packaged copy (the one holding SmoothMyVideo.exe); a dev tree runs no automatic update
const appTop = (): string | null => {
  if (!app.isPackaged) return null;
  const top = path.dirname(process.execPath);
  return path.dirname(top) === top ? null : top; // never a drive root
};
// Electron's fs treats a path through an .asar as the archive's inside, and a mere existsSync on another version's
// app.asar opens and caches that archive (its tree can then never be deleted: EPERM, probed), so every look at or
// delete of a tree holding an app.asar (an unpacked version, the previous version's folder) runs with the hooks off
function withNoAsar<T>(fn: () => T): T {
  const proc = process as NodeJS.Process & { noAsar?: boolean };
  const was = proc.noAsar;
  proc.noAsar = true;
  try {
    return fn();
  } finally {
    proc.noAsar = was;
  }
}
const rmTree = (p: string) => withNoAsar(() => fs.rmSync(p, { recursive: true, force: true }));
const asarExists = (p: string) => withNoAsar(() => fileExists(p));
// the update's own lines go to the console and to the helper's log (%TEMP%\smv-update.log), one record of a whole update
function ulog(m: string): void {
  console.log(m);
  try {
    fs.appendFileSync(path.join(os.tmpdir(), 'smv-update.log'), `[${new Date().toISOString()}] app: ${m}\n`);
  } catch {
    /* nowhere to log */
  }
}
function sendUpdate(channel: string, payload: unknown): void {
  try {
    if (win && !win.webContents.isDestroyed()) win.webContents.send(channel, payload);
  } catch {
    /* window gone */
  }
}
function newerVersion(a: string, b: string): boolean {
  // true when dotted-numeric a > b (non-numeric parts count as 0; length-agnostic)
  const pa = a.split('.').map((x) => parseInt(x, 10) || 0);
  const pb = b.split('.').map((x) => parseInt(x, 10) || 0);
  for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
    if ((pa[i] || 0) !== (pb[i] || 0)) return (pa[i] || 0) > (pb[i] || 0);
  }
  return false;
}
// the unpacked, checked tree of a version (update\<version>\x\.ready marks it)
function readyTree(top: string, version: string): string | null {
  const x = path.join(top, UPDATE_DIR, version, 'x');
  return fileExists(path.join(x, UPDATE_READY)) ? treeIn(x) : null;
}
// the one folder under an unpack folder holding this exe's name and the asar
function treeIn(x: string): string | null {
  try {
    for (const e of fs.readdirSync(x, { withFileTypes: true })) {
      const d = path.join(x, e.name);
      if (
        e.isDirectory() &&
        fileExists(path.join(d, path.basename(process.execPath))) &&
        asarExists(path.join(d, 'resources', 'app.asar'))
      )
        return d;
    }
  } catch {
    /* unreadable */
  }
  return null;
}
// the newest unpacked version newer than this build, waiting under update\
function pendingUpdate(top: string): { version: string; tree: string } | null {
  let best: { version: string; tree: string } | null = null;
  try {
    for (const e of fs.readdirSync(path.join(top, UPDATE_DIR), { withFileTypes: true })) {
      if (!e.isDirectory() || !newerVersion(e.name, app.getVersion())) continue;
      const tree = readyTree(top, e.name);
      if (tree && (!best || newerVersion(e.name, best.version))) best = { version: e.name, tree };
    }
  } catch {
    /* no update folder */
  }
  return best;
}
// the swap helper exe (engine\update\smv-update.exe, its source in build_src there): the NEW version's copy, so its swap
// rules ship with it, copied to %TEMP% as soon as that version is unpacked (the antivirus has scanned the fresh copy
// long before it runs) and run from there as this process exits
const updateHelperPath = () => path.join(os.tmpdir(), 'smv-update-apply.exe');
function prepareHelper(tree: string): boolean {
  try {
    const own = path.join(tree, 'resources', 'engine', 'update', 'smv-update.exe');
    const src = fileExists(own) ? own : path.join(ENGINE, 'update', 'smv-update.exe');
    const dst = updateHelperPath();
    if (!fileExists(dst) || sha256File(dst) !== sha256File(src)) fs.copyFileSync(src, dst);
    return true;
  } catch (e) {
    ulog(`update: the swap helper could not be copied to %TEMP%: ${e}`);
    return false;
  }
}
function launchSwap(top: string, rel: string, relaunch: boolean): boolean {
  try {
    if (!prepareHelper(path.join(top, rel))) return false;
    fs.rmSync(path.join(top, UPDATE_DIR, UPDATE_FAILED), { force: true });
    // A GUI-subsystem exe run from %TEMP%: the detached spawn it needs to outlive this process opens no console (a
    // detached console program gets none and every console child of it opens a window of its own), and it holds nothing
    // inside the app folder whose files it moves. Its working directory is %TEMP% too: this process's own (the app
    // folder, as Explorer starts it) would be inherited.
    const child = spawn(
      updateHelperPath(),
      [String(process.pid), top, rel, relaunch ? '1' : '0', path.basename(process.execPath)],
      { detached: true, stdio: 'ignore', windowsHide: true, cwd: os.tmpdir() },
    );
    child.unref();
    return true;
  } catch (e) {
    ulog(`update: the swap helper did not start: ${e}`);
    return false;
  }
}
// the start of a launch: an unpacked newer version waiting = the helper swaps it in once this process exits and starts
// it (the Chrome way: an update the button was not pressed for applies at the next launch); after a swap the helper
// could not finish, the launch opens normally and the button offers the retry
function applyPendingUpdate(): boolean {
  const top = appTop();
  const pend = top ? pendingUpdate(top) : null;
  if (!top || !pend || fileExists(path.join(top, UPDATE_DIR, UPDATE_FAILED))) return false;
  if (!launchSwap(top, path.relative(top, pend.tree), true)) return false;
  ulog(`update: ${pend.version} is applied at this launch`);
  app.quit();
  return true;
}
// a newer version's download cut short, waiting to resume
const hasPartial = (dir: string): boolean => {
  try {
    return fs.readdirSync(dir).some((n) => n.endsWith('.part'));
  } catch {
    return false;
  }
};
// after the window of a launch: update\<version>\ of this version or an older one goes (the swap's old files and its
// emptied tree: the rollback net until this version started), so do an unfinished probe and the notes of versions no
// longer newer; a newer version's unpacked tree (its swap failed: the button retries) or cut download stays
function finishUpdate(): void {
  const top = appTop();
  if (!top) return;
  const upd = path.join(top, UPDATE_DIR);
  const gone: string[] = [];
  try {
    const pend = pendingUpdate(top);
    const fails = readFails(top);
    for (const e of fs.readdirSync(upd, { withFileTypes: true })) {
      const p = path.join(upd, e.name);
      if (e.isDirectory()) {
        const waits = newerVersion(e.name, app.getVersion()) && (readyTree(top, e.name) || hasPartial(p));
        if (!waits) gone.push(p);
      } else if (
        e.name === UPDATE_FAILS
          ? !fails || !newerVersion(fails.version, app.getVersion())
          : e.name === UPDATE_FAILED && !pend
      ) {
        gone.push(p);
      }
    }
  } catch {
    return; // no update folder
  }
  if (!gone.length) return;
  setTimeout(() => {
    for (const p of gone) {
      try {
        rmTree(p);
        ulog(`update: removed ${p}`);
      } catch (e) {
        ulog(`update: ${p} stays (${e})`);
      }
    }
    try {
      if (!fs.readdirSync(upd).length) fs.rmdirSync(upd);
    } catch {
      /* in use or gone */
    }
  }, 3000);
}
function checkForUpdate(): void {
  setTimeout(async () => {
    const top = appTop();
    const pend = top ? pendingUpdate(top) : null; // unpacked earlier, its swap did not happen (failed.txt)
    let rel: Release = {};
    try {
      // Chromium's network stack: the system proxy applies, as for the download
      const res = await net.fetch(UPDATE_API, {
        headers: { accept: 'application/vnd.github+json', 'user-agent': 'SmoothMyVideo' },
      });
      if (res.ok) rel = (await res.json()) as Release;
    } catch {
      /* offline / API unreachable: stay silent */
    }
    const latest = String(rel.tag_name || '').replace(/^v/i, '');
    const url = rel.html_url || `https://github.com/${UPDATE_REPO}/releases/`;
    if (top && pend) {
      updateReady = { version: pend.version, rel: path.relative(top, pend.tree) };
      prepareHelper(pend.tree);
      sendUpdate('update-available', { version: pend.version, url });
      sendUpdate('update-failed', { version: pend.version, retry: true });
      sendUpdate('update-ready', { version: pend.version });
    }
    if (!latest || !newerVersion(latest, app.getVersion())) return;
    if (!pend) sendUpdate('update-available', { version: latest, url });
    void autoUpdate(rel, latest);
  }, 5000);
}
// a version's release-files.txt (scripts\release-files.ts writes it into every release): null = none or unreadable
function readReleaseList(file: string): ListEntry[] | null {
  try {
    const out: ListEntry[] = [];
    for (const line of ofs.readFileSync(file, 'utf8').split(/\r?\n/)) {
      if (!line || line.startsWith('#')) continue;
      const m = /^([0-9a-f]{64}) (\d+) (.+)$/.exec(line);
      if (!m) return null;
      out.push({ sha: m[1], size: Number(m[2]), rel: m[3] });
    }
    return out.length ? out : null;
  } catch {
    return null;
  }
}
// through Electron's unpatched fs: a path through an app.asar is read as the file it is
function hashInto(p: string, h: crypto.Hash): Promise<void> {
  return new Promise((resolve, reject) => {
    ofs
      .createReadStream(p)
      .on('data', (d: string | Buffer) => h.update(d))
      .on('error', reject)
      .on('end', () => resolve());
  });
}
async function sha256Of(p: string): Promise<string> {
  const h = crypto.createHash('sha256');
  await hashInto(p, h);
  return h.digest('hex');
}
function readFails(top: string): { version: string; count: number; why: string } | null {
  try {
    const f = JSON.parse(fs.readFileSync(path.join(top, UPDATE_DIR, UPDATE_FAILS), 'utf8'));
    return f && typeof f.version === 'string' && typeof f.count === 'number' ? f : null;
  } catch {
    return null;
  }
}
function noteFail(top: string, version: string, why: string): void {
  const f = readFails(top);
  const count = f && f.version === version ? f.count + 1 : 1;
  try {
    fs.mkdirSync(path.join(top, UPDATE_DIR), { recursive: true });
    fs.writeFileSync(path.join(top, UPDATE_DIR, UPDATE_FAILS), JSON.stringify({ version, count, why }));
  } catch {
    /* the folder took the download, so it takes this */
  }
}
// 7zr.exe x: a console program started hidden (no window); it checks every file's CRC itself
function unpack7z(archive: string, dest: string, signal?: AbortSignal, timeout = 0): Promise<void> {
  return new Promise((resolve, reject) =>
    execFile(
      UPDATE_7ZR(),
      ['x', '-y', '-bd', '-bso0', '-bsp0', '-o' + dest, archive],
      { signal, timeout, windowsHide: true, maxBuffer: 1 << 20 },
      (e) => (e ? reject(e) : resolve()),
    ),
  );
}
// what stops an automatic update in this copy, checked before any byte is downloaded; null = nothing
async function updateBlocker(top: string, version: string, size: number): Promise<UpdateWhy | null> {
  const list = path.join(top, RELEASE_LIST);
  if (!readReleaseList(list)) return 'list'; // the swap replaces only the files this list names
  const fails = readFails(top);
  if (fails && fails.version === version && fails.count >= 2) return 'failed';
  // the folder takes new files and lets go of its own (Program Files, Controlled Folder Access or a read-only drive
  // stop it here): a folder made under update\, this copy's list renamed and back
  const probe = path.join(top, UPDATE_DIR, 'probe');
  try {
    rmTree(probe);
    fs.mkdirSync(probe, { recursive: true });
    fs.renameSync(list, list + '.probe');
    fs.renameSync(list + '.probe', list);
  } catch {
    try {
      if (!fileExists(list) && fileExists(list + '.probe')) fs.renameSync(list + '.probe', list);
    } catch {
      /* the list stays renamed: the next launch reports 'list' */
    }
    return 'locked';
  }
  try {
    const st = fs.statfsSync(top); // the archive plus its unpacked tree (about three times the archive)
    if (size && st.bavail * st.bsize < size * 5) return 'space';
  } catch {
    /* no figure: try */
  }
  // the unpacker runs on this PC (an antivirus or a policy may stop it): the probe archive, packed as the release is
  try {
    await unpack7z(UPDATE_PROBE, probe, undefined, 30000);
    return fileExists(path.join(probe, 'smv-update.exe')) ? null : 'unpack';
  } catch {
    return 'unpack';
  } finally {
    try {
      rmTree(probe);
    } catch {
      /* finishUpdate takes it */
    }
  }
}
// the background download, check and unpack of a newer release into <app folder>\update\<version>\; a refusal or a
// failure leaves the notice's link as the way to the new build
async function autoUpdate(rel: Release, version: string): Promise<void> {
  const top = appTop();
  if (!top || !autoUpdateOn || process.env.SMV_AUTO_UPDATE === '0' || updateAbort) return;
  if (updateReady && !newerVersion(version, updateReady.version)) return;
  const asset = (rel.assets || []).find((a) => /\.7z$/i.test(a.name || '') && a.browser_download_url);
  const hash = /sha256\(win\.7z\):\s*([0-9a-f]{64})/i.exec(rel.body || '')?.[1]?.toLowerCase();
  if (!asset || !hash) {
    ulog('update: the release has no archive asset or no hash line: the notice only');
    return;
  }
  const have = readyTree(top, version); // unpacked by an earlier launch, the swap still to come
  if (have) {
    updateReady = { version, rel: path.relative(top, have) };
    prepareHelper(have);
    sendUpdate('update-ready', { version });
    return;
  }
  const size = asset.size || 0;
  const why = await updateBlocker(top, version, size);
  if (why) {
    ulog(`update: automatic update off in this copy (${why}): the notice only`);
    sendUpdate('update-blocked', { version, why });
    return;
  }
  const dir = path.join(top, UPDATE_DIR, version);
  const x = path.join(dir, 'x');
  const archive = path.join(dir, asset.name || 'SmoothMyVideo.7z');
  const part = archive + '.part';
  const ac = new AbortController();
  updateAbort = ac;
  let stage: 'download' | 'check' = 'download';
  try {
    fs.mkdirSync(dir, { recursive: true });
    // a finished download from an earlier launch is kept; a cut one resumes from its last byte
    if (!(fileExists(archive) && fs.statSync(archive).size === size && (await sha256Of(archive)) === hash)) {
      fs.rmSync(archive, { force: true });
      let h = crypto.createHash('sha256');
      let had = fileExists(part) ? fs.statSync(part).size : 0;
      if (had && size && had < size) await hashInto(part, h);
      else had = 0;
      // Chromium's network stack (net.fetch), not Node's: a large body the file cannot take as fast as it arrives
      // leaves Node's undici parser paused when the socket ends, and its assert(!this.paused) kills the main process
      // (seen with the local test server); the hash runs over the chunks as they land
      const res = await net.fetch(asset.browser_download_url as string, {
        headers: { 'user-agent': 'SmoothMyVideo', ...(had ? { range: `bytes=${had}-` } : {}) },
        cache: 'no-store', // the resume is ours; the browser cache keeps no copy of the archive
        signal: ac.signal,
      });
      if (had && res.status !== 206) {
        had = 0; // the whole file came instead of the rest: start over
        h = crypto.createHash('sha256');
      }
      if (!res.ok || !res.body) throw new Error(`download failed: HTTP ${res.status}`);
      if (had) ulog(`update: the download of ${version} resumes at byte ${had}`);
      const total = size || had + (parseInt(res.headers.get('content-length') || '0', 10) || 0);
      let received = had;
      let last = 0;
      const src = Readable.fromWeb(res.body as never);
      src.on('data', (chunk: Buffer) => {
        h.update(chunk);
        received += chunk.length;
        const now = Date.now();
        if (now - last > 250) {
          last = now;
          sendUpdate('update-progress', { version, received, total, stage: 'download' });
        }
      });
      await pipeline(src, fs.createWriteStream(part, { flags: had ? 'a' : 'w' }), { signal: ac.signal });
      stage = 'check';
      sendUpdate('update-progress', { version, received, total, stage: 'verify' });
      const got = h.digest('hex');
      if (got !== hash) {
        fs.rmSync(part, { force: true });
        throw new Error(`checksum mismatch (${got.slice(0, 12)}...)`);
      }
      fs.renameSync(part, archive);
    }
    stage = 'check';
    sendUpdate('update-progress', { version, received: 0, total: 0, stage: 'unpack' });
    rmTree(x);
    fs.mkdirSync(x, { recursive: true });
    await unpack7z(archive, x, ac.signal);
    // every file of the new version against its own list (a cut unpack, a file an antivirus took), before any swap
    const tree = treeIn(x);
    if (!tree) throw new Error('the unpacked archive holds no app folder');
    const list = readReleaseList(path.join(tree, RELEASE_LIST));
    if (!list) throw new Error('the new version has no release-files.txt');
    for (const f of list) {
      const p = path.join(tree, f.rel);
      if (!ofs.existsSync(p) || ofs.statSync(p).size !== f.size || (await sha256Of(p)) !== f.sha)
        throw new Error(`the unpacked ${f.rel} differs from the new version's list`);
    }
    fs.writeFileSync(path.join(x, UPDATE_READY), version);
    fs.rmSync(archive, { force: true });
    updateReady = { version, rel: path.relative(top, tree) };
    prepareHelper(tree);
    ulog(`update: ${version} is unpacked and checked, applied by the Update button or at the next launch`);
    sendUpdate('update-ready', { version });
  } catch (e) {
    const aborted = ac.signal.aborted;
    // a cut download keeps its part for the next launch; a wrong checksum, a failed unpack or a failed check is noted
    // (twice for one version = not downloaded again) and its folder goes
    if (aborted || stage === 'download') {
      ulog(
        `update: ${aborted ? 'stopped (Automatic updates off)' : `download cut (${e}), it resumes at the next launch`}`,
      );
    } else {
      noteFail(top, version, String(e));
      ulog(`update: failed (${e}): the notice only`);
      try {
        rmTree(dir);
      } catch {
        /* best effort */
      }
    }
    sendUpdate('update-failed', { version, aborted });
  } finally {
    if (updateAbort === ac) updateAbort = null;
  }
}
ipcMain.on('update-auto', (_e, on: boolean) => {
  autoUpdateOn = !!on;
  if (!autoUpdateOn && updateAbort) updateAbort.abort(); // a download under way stops; an unpacked tree stays
});
// a running render pauses through the pause flag; the host says PAUSED once it holds (ten seconds at most: a stage that
// does not poll the flag ends with the app, and the exit resume picks the render up from its last saved frames)
function pauseRenderForUpdate(c: ChildProcess, capMs = 10000): Promise<boolean> {
  return new Promise((resolve) => {
    let tailText = '';
    let done = false;
    const finish = (ok: boolean) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      c.stdout?.off('data', onData);
      c.stderr?.off('data', onData);
      c.off('exit', onExit);
      resolve(ok);
    };
    const onData = (d: Buffer) => {
      tailText = (tailText + d.toString()).slice(-512);
      if (/(^|\n)PAUSED/.test(tailText)) finish(true);
    };
    const onExit = () => finish(false);
    const timer = setTimeout(() => finish(false), capMs);
    c.stdout?.on('data', onData);
    c.stderr?.on('data', onData); // the render's say() lines, PAUSED among them, come on stderr
    c.once('exit', onExit);
    try {
      fs.mkdirSync(path.dirname(PAUSE_FLAG), { recursive: true });
      fs.writeFileSync(PAUSE_FLAG, '');
    } catch {
      finish(false);
    }
  });
}
// the Update button: a running render pauses (the new version resumes it from its partial output when the same video
// is rendered again), a Live session stops, then this process exits, the helper swaps the files, the new version starts
ipcMain.on('update-apply', async () => {
  const top = appTop();
  const ready = updateReady;
  if (!top || !ready) return;
  if (current) {
    const already = fileExists(PAUSE_FLAG); // the user paused it before: the host already holds
    const held = already || (await pauseRenderForUpdate(current));
    ulog(`update: the render ${held ? 'paused' : 'did not report a pause within 10 s'} for the update`);
  }
  if (liveProc) {
    ulog('update: the Live session stops for the update');
    stopLive(true);
  }
  // app.quit() skips window-all-closed, where these go at a normal close; a host still busy with the paused render
  // is ended by the helper (it ends whatever still runs from the app folder)
  offlineHostQuit();
  fsrWarmProc?.kill();
  if (!launchSwap(top, ready.rel, true)) return;
  app.quit();
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
function displayPeakNits(): number {
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

// --- RTX Video runtime: readiness + one-click install --------------------------------------------
// The opt-in RTX features (VSR / TrueHDR) run through the compiled CUDA bridge (rtxvideo_cuda.dll,
// which ships with the app) plus NVIDIA's two RTX Video feature DLLs. Those feature DLLs are NVIDIA
// proprietary and NON-redistributable, so they are never bundled; the user downloads NVIDIA's RTX
// Video SDK (a deliberate, EULA-gated action) and this app drops the two DLLs into engine/rtxvideo
// for them. A feature is "ready" only when the bridge AND its feature DLL are present there.
const RTX_DIR = path.join(ENGINE, 'rtxvideo');
const RTX_FEATURE_DLLS = ['nvngx_vsr.dll', 'nvngx_truehdr.dll'];
const RTX_SDK_URL = 'https://developer.nvidia.com/rtx-video-sdk/getting-started';
// NvOFFRUC ("NVIDIA Smooth Motion"): our bridge (nvoffruc_bridge.dll) is locally built and ships,
// but NvOFFRUC.dll + cudart64_110.dll are NVIDIA proprietary and user-installed from the Optical
// Flow SDK .zip - same EULA-gated, non-redistributable pattern as the RTX feature DLLs.
const NVOFFRUC_DIR = path.join(ENGINE, 'nvoffruc');
const NVOFFRUC_DLLS = ['NvOFFRUC.dll', 'cudart64_110.dll'];
const OF_SDK_URL = 'https://developer.nvidia.com/opticalflow/download';
// "DLSS 4.5" (DLSS Frame Generation): fully bundled, no installer - the offline host exe
// (engine/dlssg/dlssg2f.exe, built from build_src) plus NVIDIA's redistributable Streamline
// runtime (~10 MB, licenses alongside). Readiness = all files present (False means a broken
// install); actual GPU/driver support surfaces as a clear engine error at render time.
const DLSSG_DIR = path.join(ENGINE, 'dlssg');
const DLSSG_FILES = [
  'dlssg2f.exe',
  'sl.interposer.dll',
  'sl.common.dll',
  'sl.dlss_g.dll',
  'sl.pcl.dll',
  'sl.reflex.dll',
  'nvngx_dlssg.dll',
];
// "NVIDIA DLSS 5" (Neural Rendering): the caller shim engine/dlssnr/nvngx.dll (built from build_src;
// the native host loads the runtime through it) ships, but the NR runtime nvngx_dlssnr.dll is NOT shipped: NVIDIA
// publishes no download (no SDK, no driver copy, no NVIDIA App override as of 2026-09), the only
// NVIDIA copy sits inside NBA 2K27. So the app offers a one-click download of the community build
// every DLSS 5 tool pulls from (RankFTW/rhi-repo release assets, the RenoDX author's RTX 40 + 50
// rebuild), pinned to one asset and verified twice (zip SHA256 from the GitHub API digest, then the
// DLL inside). nvngx_dlss.dll is the DLSS SR runtime; the NGX core only warns when it is absent
// (probe-verified), so it is reported, never required.
const DLSSNR_DIR = path.join(ENGINE, 'dlssnr');
const DLSSNR_HOST = ['nvngx.dll'];
const DLSSNR_RUNTIME = 'nvngx_dlssnr.dll';
const DLSSNR_SR = 'nvngx_dlss.dll';
// The one-click source: the pinned rhi-repo asset, and the ONE hash this app holds: the
// SHA256 of the nvngx_dlssnr.dll inside it (probe-verified on the host). The download refuses to install any other
// DLL. Bump tag, asset and hash together when moving to a
// newer build and add the replaced one to DLSSNR_RETIRED (the zip's own digest is at
// https://api.github.com/repos/RankFTW/rhi-repo/releases).
const DLSSNR_DL = {
  url: 'https://github.com/RankFTW/rhi-repo/releases/download/dlssnr-310.9.16/nvngx_dlssnr_310.9.16.zip',
  page: 'https://github.com/RankFTW/rhi-repo/releases/tag/dlssnr-310.9.16',
  dllSha256: 'cdfa6c1ec784867182164c7836f324f087b5c9f374d0c9fe46f54999feabf3ed',
  label: '310.9.16 (rhi-repo, RTX 40 + 50)',
  zipBytes: 192378839,
};
// Superseded one-click builds (size + SHA256 of nvngx_dlssnr.dll). An installed copy, downloaded or dropped, is
// removed at app start so the DLSS 5 row offers DLSSNR_DL's build: SMV offers one build only.
const DLSSNR_RETIRED = [
  {
    label: '310.8.SF-v2',
    bytes: 165830144,
    sha256: '6eb209e764f39872625debd6abaf45e2bb6322f6f270f781f70c059ae30b3927',
  },
];
/** Remove an installed retired runtime; '' when there is none. One held by a running host stays and is
 *  retried next start. The size test comes first, so only a same-size file gets hashed. */
function retireOldDlssnr(): string {
  const p = path.join(DLSSNR_DIR, DLSSNR_RUNTIME);
  let old: (typeof DLSSNR_RETIRED)[number] | undefined;
  try {
    const size = fs.statSync(p).size;
    old = DLSSNR_RETIRED.find((r) => r.bytes === size && r.sha256 === sha256File(p));
  } catch {
    return '';
  }
  if (!old) return '';
  try {
    fs.rmSync(p);
  } catch {
    return `DLSS 5: the retired ${old.label} runtime is in use, retried next start`;
  }
  return `DLSS 5: removed the retired ${old.label} runtime, the download offers ${DLSSNR_DL.label}`;
}
function sha256File(p: string): string {
  const h = crypto.createHash('sha256');
  const fd = fs.openSync(p, 'r');
  try {
    const buf = Buffer.alloc(1 << 20);
    for (;;) {
      const n = fs.readSync(fd, buf, 0, buf.length, null);
      if (n <= 0) break;
      h.update(buf.subarray(0, n));
    }
  } finally {
    fs.closeSync(fd);
  }
  return h.digest('hex');
}
const SYS_TAR = path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'tar.exe');
const fileExists = (p: string) => {
  try {
    return fs.existsSync(p);
  } catch {
    return false;
  }
};

type InstallResult = { ok: boolean; error?: string; copied: string[] };

// Every file under root whose name matches one of `names` (case-insensitive; a release zip can
// nest its payload, and the __MACOSX ._ copies fall out because their basename differs).
function findFilesNamed(root: string, names: string[]): string[] {
  const want = new Set(names.map((n) => n.toLowerCase()));
  const hits: string[] = [];
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
      else if (want.has(e.name.toLowerCase())) hits.push(p);
    }
  }
  return hits;
}

// The one zip-extract-and-copy flow behind every user-installed runtime (the RTX feature DLLs,
// the NvOFFRUC pair, dovi_tool / hdr10plus_tool, the DLSS 5 runtime). `source` is a .zip (only
// the wanted members are extracted, with Windows' bundled bsdtar, into a temp dir that is removed
// again), an extracted folder, or a picked file (its folder is searched). `prefer` breaks ties
// when an archive ships several copies of a name (the RTX SDK's arm64 / x64 dev / rel builds, the
// Optical Flow SDK's win32 / win64). Every wanted name must be found, else nothing is copied.
function installFiles(
  source: string,
  names: string[],
  destDir: string,
  opts: { what: string; where: string; prefer?: RegExp },
): InstallResult {
  try {
    fs.mkdirSync(destDir, { recursive: true });
  } catch {
    /* exists */
  }
  let tmp: string | null = null;
  const srcFiles: string[] = [];
  try {
    let searchDir = source;
    if (/\.zip$/i.test(source)) {
      tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'smv-install-'));
      try {
        execFileSync(SYS_TAR, ['-xf', source, '-C', tmp, ...names.map((n) => '*' + n)]);
      } catch {
        // bsdtar's member patterns are case-sensitive and fail when one matches nothing (a
        // DOVI_TOOL.EXE member, a renamed SDK layout): take the whole archive and search it
        execFileSync(SYS_TAR, ['-xf', source, '-C', tmp]);
      }
      searchDir = tmp;
    } else if (!fs.statSync(source).isDirectory()) {
      searchDir = path.dirname(source);
    }
    const found = findFilesNamed(searchDir, names);
    for (const n of names) {
      const all = found.filter((f) => path.basename(f).toLowerCase() === n.toLowerCase());
      const hit = (opts.prefer && all.find((f) => opts.prefer!.test(f))) || all[0];
      if (hit) srcFiles.push(hit);
    }
    if (srcFiles.length < names.length)
      return { ok: false, error: names.join(' / ') + ' not found in ' + opts.what, copied: [] };
    const copied: string[] = [];
    try {
      for (const f of srcFiles) {
        const dest = path.join(destDir, path.basename(f));
        // Overwrite any existing copy (a newer release replaces the old files), even one a previous
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
        error: 'Could not write ' + opts.where + ' (' + String(e) + '). If a render is running, stop it and try again.',
        copied,
      };
    }
    return { ok: true, copied };
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

// The two RTX Video feature DLLs into engine/rtxvideo (the SDK ships arm64 + x64 dev/rel copies;
// take the x64 release build).
const installRtx = (source: string): InstallResult =>
  installFiles(source, RTX_FEATURE_DLLS, RTX_DIR, {
    what: 'the selected RTX Video SDK',
    where: 'the RTX DLLs in engine/rtxvideo',
    prefer: /x64[\\/]+rel/i,
  });

// NvOFFRUC.dll + cudart64_110.dll into engine/nvoffruc, beside the locally built bridge (the SDK
// ships win32 + win64 copies; take the x64 build).
const installFruc = (source: string): InstallResult =>
  installFiles(source, NVOFFRUC_DLLS, NVOFFRUC_DIR, {
    what: 'the selected Optical Flow SDK',
    where: 'the FRUC DLLs in engine/nvoffruc',
    prefer: /win64/i,
  });

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

// Install from the picked or dropped .zip.
ipcMain.handle('rtx-install', (_e, source: string) => installRtx(source));

// Manual picker: the RTX Video SDK .zip.
ipcMain.handle('rtx-choose', async () => {
  const r = await dialog.showOpenDialog(win!, {
    title: 'Select the RTX Video SDK .zip',
    properties: ['openFile'],
    filters: [{ name: 'Zip', extensions: ['zip'] }],
  });
  return r.canceled ? null : r.filePaths[0] || null;
});

// "NVIDIA Smooth Motion" (NvOFFRUC) runtime: ready only when our bridge AND NvOFFRUC.dll are present.
// "DLSS 4.5" (DLSS Frame Generation): bundled, so this only guards against a broken install.
// DLSS 4.5 runs on RTX 40 / 50 GPUs with Windows' hardware-accelerated GPU scheduling on (HwSchMode 2); read once a run.
// The renderer makes it the default model only when both hold
let dlssgSupport: Promise<{ gpu: string; hags: boolean; supported: boolean }> | null = null;
function dlssgSupportOnce(): Promise<{ gpu: string; hags: boolean; supported: boolean }> {
  if (!dlssgSupport)
    dlssgSupport = new Promise((resolve) => {
      execFile(
        'nvidia-smi',
        ['--query-gpu=name', '--format=csv,noheader'],
        { timeout: 5000, windowsHide: true },
        (e1, o1) => {
          const gpu = e1 ? '' : String(o1).split(/\r?\n/)[0].trim();
          const key = 'HKLM\\SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers';
          execFile('reg', ['query', key, '/v', 'HwSchMode'], { timeout: 5000, windowsHide: true }, (e2, o2) => {
            const hags = !e2 && /HwSchMode\s+REG_DWORD\s+0x2\b/i.test(String(o2));
            resolve({ gpu, hags, supported: hags && /RTX\s*(40|50)\d{2}\b|\bAda\b|Blackwell/i.test(gpu) });
          });
        },
      );
    });
  return dlssgSupport;
}

ipcMain.handle('dlssg-ready', async () => {
  const missing = DLSSG_FILES.filter((f) => !fileExists(path.join(DLSSG_DIR, f)));
  const s = await dlssgSupportOnce();
  return { ready: missing.length === 0, missing, dir: DLSSG_DIR, gpu: s.gpu, hags: s.hags, supported: s.supported };
});

// "NVIDIA DLSS 5" (Neural Rendering): ready when the shipped caller shim AND the user-supplied
// runtime are present in engine/dlssnr (see the DLSSNR_* constants).
ipcMain.handle('dlssnr-ready', () => {
  const host = DLSSNR_HOST.every((f) => fileExists(path.join(DLSSNR_DIR, f)));
  const runtime = fileExists(path.join(DLSSNR_DIR, DLSSNR_RUNTIME));
  const sr = fileExists(path.join(DLSSNR_DIR, DLSSNR_SR));
  return {
    ready: host && runtime,
    host,
    runtime,
    sr,
    dir: DLSSNR_DIR,
    file: DLSSNR_RUNTIME,
    srFile: DLSSNR_SR,
    download: { page: DLSSNR_DL.page, mb: Math.round(DLSSNR_DL.zipBytes / 1048576) },
  };
});

// One-click: download the pinned rhi-repo zip into a temp dir (progress to the renderer as
// 'dlssnr-progress'), extract with bsdtar, verify the DLL hash, copy it into engine/dlssnr. A
// mismatch aborts before the copy, so a replaced release asset can never install anything.
ipcMain.handle('dlssnr-download', async () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'smv-dlssnr-'));
  const zip = path.join(tmp, 'runtime.zip');
  const progress = (received: number, total: number, stage: string) => {
    try {
      if (win && !win.webContents.isDestroyed()) win.webContents.send('dlssnr-progress', { received, total, stage });
    } catch {
      /* window gone */
    }
  };
  try {
    const res = await fetch(DLSSNR_DL.url, { headers: { 'user-agent': 'SmoothMyVideo' } });
    if (!res.ok || !res.body) return { ok: false, error: `download failed: HTTP ${res.status}`, copied: [] };
    const total = parseInt(res.headers.get('content-length') || '0', 10) || DLSSNR_DL.zipBytes;
    let received = 0;
    let last = 0;
    const src = Readable.fromWeb(res.body as never);
    src.on('data', (chunk: Buffer) => {
      received += chunk.length;
      if (received - last > 2 * 1048576 || received === total) {
        last = received;
        progress(received, total, 'download');
      }
    });
    await pipeline(src, fs.createWriteStream(zip));
    progress(received, total, 'verify');
    execFileSync(SYS_TAR, ['-xf', zip, '-C', tmp]);
    const found = findDvBin(tmp, DLSSNR_RUNTIME);
    if (!found)
      return { ok: false, error: DLSSNR_RUNTIME + ' not in the downloaded zip, nothing installed', copied: [] };
    const dh = sha256File(found);
    if (dh !== DLSSNR_DL.dllSha256)
      return { ok: false, error: `runtime checksum mismatch (${dh.slice(0, 12)}...), nothing installed`, copied: [] };
    fs.mkdirSync(DLSSNR_DIR, { recursive: true });
    const dest = path.join(DLSSNR_DIR, DLSSNR_RUNTIME);
    try {
      if (fileExists(dest)) fs.chmodSync(dest, 0o666);
    } catch {
      /* best effort */
    }
    fs.copyFileSync(found, dest);
    return { ok: true, copied: [DLSSNR_RUNTIME], sha256: dh, known: DLSSNR_DL.label };
  } catch (e) {
    return { ok: false, error: String(e), copied: [] };
  } finally {
    try {
      fs.rmSync(tmp, { recursive: true, force: true });
    } catch {
      /* temp cleanup best effort */
    }
  }
});

ipcMain.handle('dlssnr-open-download', () => {
  shell.openExternal(DLSSNR_DL.page);
  return true;
});

// "AnimeJaNai HD", Restore's second model: the author's 2x ONNX (CC BY-NC-SA 4.0, so never bundled: package.json's
// filter drops engine/animejanai). One click downloads the pinned release archive, checks its SHA256, extracts the
// one model with Windows' bsdtar, checks the model's SHA256 and has the host convert it into the .pkl the app keeps
// like every other model's weights (--convert-restore-model: the same weights for the kernels, the frames identical);
// the UI says when the installed .pkl is not the verified one. The host gets it as --restore-model.
const AJ_FILE = '2x_AnimeJaNai_HD_V3.1_Balanced_SPANF3_b8f64_unshuffle_fp16.onnx';
const AJ_PKL = AJ_FILE.replace(/\.onnx$/, '.pkl');
// the host's conversion of the author's ONNX into the .pkl (written only when it reads back as the ONNX's weights)
function ajConvert(onnx: string, pkl: string): boolean {
  try {
    execFileSync(path.join(ENGINE, 'live', 'smv-live.exe'), ['--convert-restore-model', onnx, pkl], {
      stdio: 'ignore',
      windowsHide: true,
    });
    return fileExists(pkl);
  } catch {
    return false;
  }
}
// engine/animejanai, the model's own folder like engine/artcnn. An install that holds the author's ONNX (in
// engine/restore, the folder's earlier name, or here) gets it moved here and converted once, at start, so an update
// keeps the download; a read-only or locked install keeps the old file and the setup row offers the download.
function ajDir(): string {
  const dir = path.join(ENGINE, 'animejanai');
  const oldDir = path.join(ENGINE, 'restore');
  const old = path.join(oldDir, AJ_FILE);
  const onnx = path.join(dir, AJ_FILE);
  const pkl = path.join(dir, AJ_PKL);
  try {
    if (fs.existsSync(old) && !fs.existsSync(onnx)) {
      fs.mkdirSync(dir, { recursive: true });
      fs.renameSync(old, onnx);
    }
    if (fs.existsSync(oldDir) && fs.readdirSync(oldDir).length === 0) fs.rmdirSync(oldDir);
    if (fs.existsSync(onnx) && (fs.existsSync(pkl) || ajConvert(onnx, pkl))) fs.rmSync(onnx);
  } catch {
    /* left as it is */
  }
  return dir;
}
const AJ_DIR = ajDir();
const AJ_DL = {
  url: 'https://github.com/the-database/mpv-upscale-2x_animejanai/releases/download/3.9.0/mpv-upscale-2x_animejanai-overlay-3.9.0.7z',
  page: 'https://github.com/the-database/mpv-upscale-2x_animejanai/releases/tag/3.9.0',
  licence: 'https://creativecommons.org/licenses/by-nc-sa/4.0/',
  member: 'animejanai/onnx/' + AJ_FILE,
  archiveSha256: '8933e1003dc9bdf5bc9cd4bb8da91051c28e03129841846b17bb65f1c0446ab0',
  onnxSha256: 'b308e93c3dda3c4f9f968295b2b7b8d1c0583a711bca53a08d51134d9d1cbd5c',
  // the host's conversion of that ONNX (the same bytes every time)
  pklSha256: 'e48fd48fbafb752fad9db5c599bcdcc600a82a82de24ed496f13c96bdd843363',
  archiveBytes: 42945210,
};
const ajPath = () => path.join(AJ_DIR, AJ_PKL);
/** The host's Restore model arguments for a model id: 'animejanai' = --restore-model with the installed file;
 * 'anime4k' = --restore-anime4k (Anime4K's Restore CNN UL, inside the host); none = Real-ESRGAN. */
function restoreModelArgs(id?: string): string[] {
  if (id === 'anime4k') return ['--restore-anime4k'];
  return id === 'animejanai' && fileExists(ajPath()) ? ['--restore-model', ajPath()] : [];
}

ipcMain.handle('aj-ready', () => {
  const ready = fileExists(ajPath());
  let sha256 = '';
  try {
    if (ready) sha256 = sha256File(ajPath());
  } catch {
    /* the hash is informational */
  }
  const known = sha256 ? sha256 === AJ_DL.pklSha256 : undefined;
  return { ready, known, sha256, file: AJ_PKL, mb: Math.round(AJ_DL.archiveBytes / 1048576) };
});

ipcMain.handle('aj-download', async () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'smv-animejanai-'));
  const archive = path.join(tmp, 'release.7z');
  const progress = (received: number, total: number, stage: string) => {
    try {
      if (win && !win.webContents.isDestroyed()) win.webContents.send('aj-progress', { received, total, stage });
    } catch {
      /* window gone */
    }
  };
  try {
    const res = await fetch(AJ_DL.url, { headers: { 'user-agent': 'SmoothMyVideo' } });
    if (!res.ok || !res.body) return { ok: false, error: `download failed: HTTP ${res.status}` };
    const total = parseInt(res.headers.get('content-length') || '0', 10) || AJ_DL.archiveBytes;
    let received = 0;
    let last = 0;
    const src = Readable.fromWeb(res.body as never);
    src.on('data', (chunk: Buffer) => {
      received += chunk.length;
      if (received - last > 2 * 1048576 || received === total) {
        last = received;
        progress(received, total, 'download');
      }
    });
    await pipeline(src, fs.createWriteStream(archive));
    progress(received, total, 'verify');
    const ah = sha256File(archive);
    if (ah !== AJ_DL.archiveSha256)
      return { ok: false, error: `archive checksum mismatch (${ah.slice(0, 12)}...), nothing installed` };
    execFileSync(SYS_TAR, ['-xf', archive, '-C', tmp, AJ_DL.member]);
    const found = path.join(tmp, ...AJ_DL.member.split('/'));
    if (!fileExists(found)) return { ok: false, error: AJ_FILE + ' not in the downloaded archive, nothing installed' };
    const mh = sha256File(found);
    if (mh !== AJ_DL.onnxSha256)
      return { ok: false, error: `model checksum mismatch (${mh.slice(0, 12)}...), nothing installed` };
    fs.mkdirSync(AJ_DIR, { recursive: true });
    if (!ajConvert(found, ajPath())) return { ok: false, error: 'the model did not convert, nothing installed' };
    const ph = sha256File(ajPath());
    return { ok: true, sha256: ph, known: ph === AJ_DL.pklSha256 };
  } catch (e) {
    return { ok: false, error: String(e) };
  } finally {
    try {
      fs.rmSync(tmp, { recursive: true, force: true });
    } catch {
      /* temp cleanup best effort */
    }
  }
});

ipcMain.handle('aj-open', (_e, which: string) => {
  shell.openExternal(which === 'licence' ? AJ_DL.licence : AJ_DL.page);
  return true;
});

// --- Live mode: the real-time pipeline host (engine/live/smv-live.exe) -----------------------------
// The exe captures the target window via Windows.Graphics.Capture and presents it enhanced into a
// click-through topmost overlay. Backend today: DLSS Frame Generation at (gen+1)x (its own copy of
// the Streamline runtime sits beside it); the design target is backend-agnostic (server route for
// RIFE/GMFSS/FRUC + VSR/HDR/sharpen/restore is the planned phase 2). Gotcha (bisected in the
// smv-live PoC): the overlay must hold foreground for the driver to generate, so the exe
// force-activates its own window at start.
const LIVE_DIR = path.join(ENGINE, 'live');
const LIVE_EXE = path.join(LIVE_DIR, 'smv-live.exe');
// Resident offline host: the render (src/render/native.ts) keeps `smv-live.exe --offline
// --resident` alive between renders (engines loaded, about 300 MB of VRAM idle) and finds it through
// this named pipe, one per app process. The render spawns it detached; this process quits it when a live session starts and at app exit
// (a busy host ignores the request and leaves on its own idle limit).
const OFFLINE_PIPE_NAME = 'smv-offline-' + process.pid;
function offlineHostQuit() {
  try {
    const fd = fs.openSync('\\\\.\\pipe\\' + OFFLINE_PIPE_NAME, 'r+');
    try {
      fs.writeSync(fd, 'quit\n');
    } finally {
      fs.closeSync(fd);
    }
  } catch {
    /* no idle host */
  }
}

// THE merged log (one file for everything): the render (src/render/cli.ts) tees
// its lines to the same literal path, so live session lines land chronologically
// between render runs. Append-only; cli.ts owns the size cap.
const SMV_LOG = path.join(os.tmpdir(), 'smv-engine.log');
function liveLog(text: string) {
  try {
    const lines = text
      .replace(/\r/g, '')
      .split('\n')
      .filter((l) => l.trim() !== '');
    if (!lines.length) return;
    const stamp = new Date().toISOString().slice(0, 19).replace('T', ' ');
    fs.appendFileSync(SMV_LOG, lines.map((l) => `[${stamp}] [live] ${l}\n`).join(''));
  } catch {
    /* the log must never break live */
  }
}
let liveProc: ChildProcess | null = null;
let liveStopping = false;
let liveRestartPending = false; // renderer asked to relaunch the session with new settings
// Resident host: every server-backend session runs `smv-live.exe --resident`,
// which keeps the process alive after the session with what the session loaded (the kernel
// module, so the next session of that model starts faster). Sessions then start and stop over the
// exe's stdin ("start<TAB>args", "stop", "quit"); the exe prints "live session ended: exit N,
// host resident" where a non-resident run exited. The host is freed (quit) when a model other
// than the one it holds is selected in the panel or when DLSS-G starts, when a file render starts, when the app quits, and
// by the exe itself after an idle limit (10 min) or when a session used something it cannot
// keep (DLSS-G, native DLSS 5, the TrueHDR bridge, an error exit).
let liveResident = false; // liveProc was spawned with --resident
let liveIdle = false; // the resident host is alive with no session running
let liveRunModel = ''; // the model of the running (or last) session on liveProc
let liveHeld = ''; // the model the idle resident host keeps loaded (= its last session's)
let liveBuildHeld = false; // the idle host stayed for an engine build its session left running (its ended line says so)
let liveRevivePending = false; // a restart / revive timer is armed: it reuses the idle host in 400 ms
let liveResolved: string | null = null; // the running session's target hwnd (revives reuse it)
let liveRestarts = 0; // exit 4 / 6 revive count of the current target
let liveStopTimer: NodeJS.Timeout | null = null; // "stop" grace: kill the host if the line never comes
const LIVE_STOP_GRACE_MS = 3000;
const LIVE_ENDED_RE = /^live session ended: exit (\d+), host resident/;
let liveModel = 'dlssg'; // mirrored from the renderer's Live settings (lv-opts): the hotkey
// and countdown paths spawn with whatever the panel currently shows
let liveLabel = ''; // user-facing effective-model name for the exe's loading message/HUD
let liveNote = ''; // substitution note (e.g. Smooth Motion live runs GMFSS), same destination
let liveHudNote = ''; // the HUD's opening note in the GUI language (the exe shows it as is)
let liveDlssMode = 'dlaa'; // the DLSS mode (dlssScaleArg): the working size's share of the presented size
// (every backend; below DLAA, DLSS 4.5 runs on the echo route's frames)
let liveFit = 'monitor'; // 'monitor' = whole-screen capture 1:1 (every model incl. dlssg; the default),
// 'window' = 1:1 overlay, 'fill' = the target's monitor upscaled (server only)
let liveTarget = 60; // adaptive output fps target, inherited from the renderer's Speed selectors
let liveSharpen = 0; // live Adaptive Sharpen strength 0..2, inherited from the Sharpen controls (0 = off)
let liveVsr = false; // RTX VSR as the live upscaler, inherited from the RTX VSR checkbox
let liveVsrQ = 4; // RTX VSR's quality level 1..4, inherited from its slider (4 = Ultra)
/** RTX VSR's quality level from the renderer's slider: 1..4, anything else = 4 (Ultra, the default). */
function vsrQ(v: number | undefined): number {
  const q = Math.round(Number(v));
  return q >= 1 && q <= 4 ? q : 4;
}
/** The host flag for a quality level; none for 4, the host's own default. */
function vsrQArgs(q: number): string[] {
  return q !== 4 ? ['--vsr-quality', String(q)] : [];
}
let liveFsrUp = false; // AMD FSR 4 as the live upscaler instead (its checkbox; the two exclude each other)
let liveA4kUp = false; // Anime4K as the live upscaler instead (its checkbox; RTX VSR and AMD FSR 4 clear it)
let liveA4kAA = false; // its Mode A+A box: Restore CNN UL on the first x2 pass's 2x frame too
let liveAcnUp = false; // ArtCNN as the live upscaler instead (its checkbox; the other three clear it)
let liveXessUp = false; // Intel XeSS as the live upscaler instead (its checkbox; the other four clear it)
let liveUpH = 0; // "Upscale to" height as the live internal render size (0 = off), inherited from the selector
let liveRestore = false; // Real-ESRGAN on every presented frame, inherited from the Restore checkbox
let liveRestoreModel = ''; // Restore's model id ('animejanai' = the installed AnimeJaNai file, 'anime4k', '' = Real-ESRGAN)
let liveDlssnr = false; // DLSS 5 Neural Rendering once per captured frame, inherited from the NVIDIA DLSS 5 checkbox
let liveNrStructure = 1; // DLSS 5 Structure Intensity 0..2
let liveNrTone = 1; // DLSS 5 Tone Intensity 0..2
let liveNrSkin = -1; // DLSS 5 Skin Structure 0..2, -1 = NVIDIA's default (follows Structure)
let liveNrAutoMask = 1; // DLSS 5 Auto mask (NVIDIA's character mask) 1 on / 0 off
let liveNrStyle = 0; // DLSS 5 Style: 0 Default (NVIDIA's, the app's default), 1 Natural, 2 Cinematic
let liveNrPasses = 1; // DLSS 5 passes, 1..20 (chained per frame)
let liveRtxHdr = false; // live TrueHDR, inherited from the RTX HDR checkbox
let liveHdrColor = 'raw'; // RTX HDR colour mode + tone knobs, inherited from the HDR sliders
let liveHdrSat = 100; // SDK Saturation (100 = NVIDIA's default)
let liveHdrCon = 100; // SDK Contrast (100 = neutral)
let liveHdrVib = 0; // Dynamic Vibrance intensity 0..1 (per-session opt-in, like renders)
let liveHdrSb = 0; // Dynamic Vibrance saturation boost 0..1
let liveHud = true; // on-screen fps/latency readout (panel checkbox; off -> --no-hud)
let liveHudLat = true; // latency segment of that readout (off -> --no-hud-latency)
let liveGpuFit = true; // "Fit to the GPU" (off -> --no-gpu-fit: the target holds, no video memory fit)

// Live events go to the main window (not a captured e.sender): the ` hotkey starts sessions
// with no IPC event at all, and the panel must reflect those too.
const sendLive = (channel: string, ...payload: unknown[]) => {
  try {
    if (win && !win.webContents.isDestroyed()) win.webContents.send(channel, ...payload);
  } catch {
    /* window gone */
  }
};

// SMV's own top-level window handle, so hotkey mode never overlays the app itself
function ownHwnd(): string {
  try {
    return win ? '0x' + win.getNativeWindowHandle().readBigUInt64LE(0).toString(16) : '0x0';
  } catch {
    return '0x0';
  }
}

// write one command line to the resident host; false when the pipe is gone
function liveCommand(line: string): boolean {
  try {
    if (liveProc?.stdin && !liveProc.stdin.destroyed) {
      liveProc.stdin.write(line + '\n');
      return true;
    }
  } catch {
    /* the host died under us; the close handler follows */
  }
  return false;
}

// end the idle resident host (a different model is about to run, or the app quits): "quit"
// lets it release the engines and exit on its own, the kill covers a wedged one
function liveQuitIdle() {
  if (!liveProc || !liveIdle) return;
  const p = liveProc;
  liveLog('quitting the idle resident host');
  if (!liveCommand('quit')) {
    try {
      p.kill();
    } catch {
      /* gone */
    }
  }
  const t = setTimeout(() => {
    try {
      if (p.exitCode === null) p.kill();
    } catch {
      /* gone */
    }
  }, 2000);
  t.unref();
  liveProc = null;
  liveIdle = false;
  liveResident = false;
  liveBuildHeld = false;
}

// a session ended: the exe exited (code from 'close') or, on the resident host, printed its
// "live session ended" line and stays alive. Same policy either way: a settings restart or an
// exit 4 / 6 revive relaunches the SAME target, anything else reports lv-done.
function onLiveSessionEnd(code: number | null) {
  if (liveStopTimer) {
    clearTimeout(liveStopTimer);
    liveStopTimer = null;
  }
  const resolved = liveResolved;
  const restarts = liveRestarts;
  const busy = () => liveProc !== null && !liveIdle;
  // settings restart (renderer 'lv-restart', e.g. the DLSS mode changed mid-run):
  // relaunch the SAME target with the CURRENT lv-opts mirror; not counted against the cap
  if (liveRestartPending && !liveStopping && resolved) {
    liveRestartPending = false;
    liveLog('restarting with new settings');
    liveRevivePending = true;
    setTimeout(() => {
      liveRevivePending = false;
      if (!busy() && !liveStopping) startLiveSession(resolved, restarts);
    }, 400);
    return;
  }
  // exit 4 = target window resized (the overlay cannot resize in place), moved monitor, or Whole
  // screen's monitor left the desktop (unplugged; restarts on the current monitor); exit 6 = the stall
  // watchdog killed a wedged enhancement engine. Both revive the SAME session with the SAME
  // configuration: a mid-session downgrade (e.g. to the eager model path) would read as
  // "it suddenly got slow" to the user, worse than a brief hiccup at full speed.
  // Bounded by the shared restart cap (interactive resizing alone can fire many restarts).
  if ((code === 4 || code === 6) && !liveStopping && restarts < 20 && resolved) {
    liveLog(`session exit ${code}, reviving (restart ${restarts + 1})`);
    if (code === 6) sendLive('lv-out', 'live engine stalled, reviving the session\n');
    liveRevivePending = true;
    setTimeout(() => {
      liveRevivePending = false;
      if (!busy() && !liveStopping) startLiveSession(resolved, restarts + 1);
    }, 400);
    return;
  }
  if (code === 6) sendLive('lv-out', 'live engine keeps stalling, giving up on this session\n');
  liveLog(`stopped (exit ${code})`);
  sendLive('lv-done', code);
}

// one overlay session; explicit hwnd (GUI picker) or the foreground window (` hotkey)
function startLiveSession(hwnd: string | null, restarts = 0) {
  if (liveProc && !liveIdle) return;
  liveStopping = false;
  liveRestartPending = false;
  offlineHostQuit(); // an idle offline render host gives its VRAM to the live session
  // Whole screen uses the foreground window only to pick the monitor, so SMV itself may be it
  const args = hwnd ? ['--hwnd', hwnd] : liveFit === 'monitor' ? ['--fg'] : ['--fg', '--exclude', ownHwnd()];
  // ONE knob: the fps target from the Speed selectors. Adaptive models resample to it; fixed
  // pipelines (DLSS 4.5) approximate it in the exe with the nearest whole multiple of the
  // captured window's measured rate, capped by the model (no --gen: the exe derives it).
  const target = String(Math.min(10000, Math.max(10, Math.round(Number(liveTarget) || 60))));
  // DLSS 4.5 is the exe's default backend (no --backend); with any effect below the exe runs the effects on every
  // captured frame and DLSS 4.5 generates from those frames
  // Intel XeSS-FG rides the fsrfg backend with its own bridge (--xess-fg)
  if (liveModel === 'xefg') args.push('--backend', 'fsrfg', '--xess-fg');
  else if (liveModel && liveModel !== 'dlssg') args.push('--backend', liveModel);
  // user-facing name + substitution note for the exe's loading message/HUD (the raw
  // backend id read as the wrong model when live substitutes, e.g. Smooth Motion -> GMFSS)
  if (liveLabel) args.push('--label', liveLabel);
  if (liveNote) args.push('--note', liveNote);
  if (liveHudNote) args.push('--hud-note', liveHudNote);
  // RIFE/GMFSS resample adaptively to the fps target inherited from the Speed selectors. No --gen: with only a
  // target the exe derives the slot count per pair from the measured source rate (ceil(target/source)+1,
  // VRAM-clamped), so high targets are not capped by a fixed 16-slot ceiling; DLSS 4.5 takes the whole multiple
  // of the measured rate nearest the target (past one pass's limit in two passes)
  args.push('--target', target);
  // live effects, inherited from the file-render Sharpen / RTX VSR settings
  const sharp = Math.min(2, Math.max(0, Number(liveSharpen) || 0));
  if (sharp > 0) args.push('--sharpen', sharp.toFixed(2));
  // "Upscale to": the server resizes the model frame to this height
  // first (VSR when enlarging), then fits it to the canvas; so VSR can now engage outside
  // fill mode too. Without it an upscale exists only in fill mode.
  const upH = Math.max(0, Math.round(Number(liveUpH) || 0));
  if (upH > 0) args.push('--upscale', String(upH));
  // the host gives RTX VSR the one resize that enlarges (the fit after the model, else the capture to the
  // working size before it) and skips it with a line when neither does
  if (liveVsr) args.push('--rtx-vsr', ...vsrQArgs(liveVsrQ));
  else if (liveFsrUp) args.push('--fsr-upscale');
  else if (liveA4kUp) args.push('--anime4k', ...(liveA4kAA ? ['--anime4k-aa'] : []));
  else if (liveAcnUp) args.push('--artcnn');
  else if (liveXessUp) args.push('--xess-upscale');
  // Restore: Real-ESRGAN first on every presented frame; costs most
  // of a 1080p frame budget, the panel hint says so
  if (liveRestore) args.push('--restore', ...restoreModelArgs(liveRestoreModel));
  // NVIDIA DLSS 5: Neural Rendering once per captured frame inside
  // the exe (SDR domain, before the model; the tweens inherit it); a session whose pass cannot
  // run goes on without it and logs why. Only sent when the runtime is installed (the
  // renderer gates on dlssnr-ready).
  if (liveDlssnr)
    args.push(
      '--dlssnr',
      '--nr-structure',
      String(liveNrStructure),
      '--nr-tone',
      String(liveNrTone),
      '--nr-style',
      String(liveNrStyle),
    );
  if (liveDlssnr && liveNrPasses > 1) args.push('--nr-passes', String(liveNrPasses));
  if (liveDlssnr && liveNrSkin >= 0) args.push('--nr-skin', String(liveNrSkin));
  if (liveDlssnr && !liveNrAutoMask) args.push('--nr-automask', '0');
  // live TrueHDR: SDR window expanded to HDR out, inherited from the RTX HDR
  // controls; the exe forwards these to the server only when its HDR live mode is on
  if (liveRtxHdr) {
    args.push('--rtx-hdr');
    if (liveHdrColor !== 'raw') args.push('--hdr-color', liveHdrColor);
    if (liveHdrSat !== 100) args.push('--hdr-saturation', String(liveHdrSat));
    if (liveHdrCon !== 100) args.push('--hdr-contrast', String(liveHdrCon));
    if (liveHdrVib > 0) args.push('--hdr-vibrance', String(liveHdrVib));
    if (liveHdrSb > 0) args.push('--hdr-satboost', String(liveHdrSb));
  }
  // the DLSS mode: the working size as its share of the presented size (the window, or the Fill rect); no
  // flag = DLAA (the presented size itself)
  const scaleArg = dlssScaleArg(liveDlssMode);
  if (scaleArg) args.push('--scale', scaleArg);
  if (liveFit === 'fill') args.push('--fit', 'fill');
  if (liveFit === 'monitor') args.push('--fit', 'monitor'); // whole-screen: all models incl. dlssg
  if (!liveHud) args.push('--no-hud');
  else if (!liveHudLat) args.push('--no-hud-latency'); // meter on, latency segment hidden
  if (!liveGpuFit) args.push('--no-gpu-fit');
  liveSetVals = liveSetNow();
  liveSetAj = liveAjFile();
  // Every server backend runs inside smv-live.exe (its native host), the only live route; a
  // session the host cannot run ends with its reason on the status line.
  // resident host: every server backend has something worth keeping (the native engines); the
  // exe itself exits after a session it cannot stay resident for. SMV_LIVE_RESIDENT=0 restores
  // one process per session (A/B harness).
  const resident = liveModel !== 'dlssg' && process.env.SMV_LIVE_RESIDENT !== '0';
  liveResolved = hwnd; // hotkey mode learns the resolved hwnd from the exe's log line
  liveRestarts = restarts;
  liveRunModel = liveModel;
  if (liveProc && liveIdle) {
    if (resident && liveResident && liveCommand('start\t' + args.join('\t'))) {
      liveIdle = false;
      liveLog(`session started on the resident host: ${args.join(' ')}`);
      sendLive('lv-started');
      return;
    }
    // a model the resident host cannot serve (or a dead pipe): free it, spawn fresh
    liveQuitIdle();
  }
  if (resident) args.push('--resident');
  // windowsHide: a packaged app has no console, so a console-subsystem child would get a window of its own
  const p = spawn(LIVE_EXE, args, { cwd: LIVE_DIR, windowsHide: true });
  liveProc = p;
  liveResident = resident;
  liveIdle = false;
  let errBuf = ''; // partial trailing line, so the hwnd is only parsed out of COMPLETE lines
  liveLog(`session started: smv-live.exe ${args.join(' ')}`);
  sendLive('lv-started');
  // a write into a host that just exited raises EPIPE on the stream: never let it throw
  p.stdin?.on('error', () => {});
  p.stderr?.on('data', (b: Buffer) => {
    const t = b.toString();
    // a chunk boundary inside "target window: hwnd=0x..." used to resolve a TRUNCATED handle,
    // and the next settings restart then respawned on a window that does not exist
    errBuf += t;
    const lines = errBuf.split(/\r?\n/);
    errBuf = lines.pop() ?? '';
    for (const line of lines) {
      const m = /target window: hwnd=0x0*([0-9a-fA-F]+)/.exec(line);
      if (m && liveProc === p && !liveIdle) liveResolved = '0x' + m[1].toLowerCase();
      const e = LIVE_ENDED_RE.exec(line);
      if (e && liveProc === p) {
        // hotkey mode (--fg) has no hwnd of its own: it learns the target from the exe's
        // "target window:" line, so a session that never printed one left the exit 4 / 6
        // revive with nothing to revive (a GMFSS session died on a resize that way: the
        // title carried an en dash and the line was dropped). The exe
        // repeats the resolved target on this line, so take it from there as the fallback.
        if (!liveResolved) {
          const h = /target=0x0*([0-9a-fA-F]+)/.exec(line);
          if (h) liveResolved = '0x' + h[1].toLowerCase();
        }
        // the resident host stays alive: this line IS the session's exit
        liveIdle = true;
        liveHeld = liveRunModel;
        liveBuildHeld = line.includes('a running engine build');
        onLiveSessionEnd(Number(e[1]));
      }
    }
    liveLog(t);
    sendLive('lv-out', t);
  });
  p.on('close', (code) => {
    liveRestoreMouse(false);
    if (liveProc !== p) return; // an idle host quit by liveQuitIdle, already forgotten
    const wasIdle = liveIdle;
    liveProc = null;
    liveIdle = false;
    liveResident = false;
    if (wasIdle) {
      // the resident host left on its own (idle limit, stdin closed): no session was running
      liveLog(`resident host exited (${code})`);
      return;
    }
    onLiveSessionEnd(code);
  });
  p.on('error', (err) => {
    liveProc = null;
    liveIdle = false;
    liveResident = false;
    liveLog('spawn error: ' + err);
    sendLive('lv-out', 'live spawn error: ' + err + '\n');
    sendLive('lv-done', -1);
  });
}

ipcMain.handle('lv-ready', () => fileExists(LIVE_EXE));

// the Smooth It Live! button: after the renderer's countdown, target the foreground window
// (the user clicked the window they want during the countdown; --exclude keeps SMV itself out
// outside Whole screen)
ipcMain.on('lv-start-fg', () => {
  startLiveSession(null);
});
// end the running session: "stop" on the resident host (it ends the session and stays), a
// kill otherwise; the grace timer kills a host whose "ended" line never comes (the native
// engine load polls the stop; the kill keeps Stop instant for a stage that does not). A stop
// already pending keeps its
// timer: a hotkey pressed every 2 s used to re-arm the grace each time, so the kill never
// fired and the panel sat on "loading the model" for the whole build.
function liveEndSession() {
  if (!liveProc || liveIdle) return;
  const p = liveProc;
  if (liveResident && liveCommand('stop')) {
    if (liveStopTimer) return;
    liveStopTimer = setTimeout(() => {
      liveStopTimer = null;
      if (liveProc === p && !liveIdle) {
        liveLog('stop grace elapsed, killing the host');
        try {
          p.kill();
        } catch {
          /* already gone */
        }
      }
    }, LIVE_STOP_GRACE_MS);
    return;
  }
  try {
    p.kill();
  } catch {
    /* already gone */
  }
}

// relaunch a running session so spawn-time settings (the DLSS mode) take effect; no-op when idle
ipcMain.on('lv-restart', () => {
  if (liveProc && !liveIdle && !liveStopping) {
    liveRestartPending = true;
    liveEndSession();
  }
});
ipcMain.on(
  'lv-opts',
  (
    _e,
    opts: {
      model: string;
      label?: string;
      note?: string;
      hudnote?: string;
      dlssmode?: string; // the DLSS mode (dlssScaleArg)
      fit: string;
      target?: number;
      sharpen?: number;
      rtxvsr?: boolean;
      vsrquality?: number; // RTX VSR's quality level 1..4
      fsrup?: boolean;
      anime4k?: boolean;
      a4kaa?: boolean; // its Mode A+A box
      artcnn?: boolean;
      xess?: boolean;
      uph?: number;
      restore?: boolean;
      restoreModel?: string; // Restore's model id ('animejanai' / 'anime4k'; none = Real-ESRGAN)
      dlssnr?: boolean;
      nrstructure?: number;
      nrtone?: number;
      nrskin?: number; // DLSS 5 Skin Structure 0..2, -1 = NVIDIA's default
      nrautomask?: number; // DLSS 5 Auto mask 1 / 0
      nrstyle?: number; // DLSS 5 Style 0 Default / 1 Natural / 2 Cinematic
      nrpasses?: number; // DLSS 5 passes 1..20
      rtxhdr?: boolean;
      hdrcolor?: string;
      hdrsat?: number;
      hdrcon?: number;
      hdrvib?: number;
      hdrsb?: number;
      hud?: boolean;
      hudlat?: boolean;
      gpufit?: boolean;
    },
  ) => {
    liveModel = opts.model;
    liveLabel = opts.label ?? '';
    liveNote = opts.note ?? '';
    liveHudNote = opts.hudnote ?? '';
    liveDlssMode = opts.dlssmode ?? 'dlaa';
    liveFit = opts.fit;
    if (opts.target !== undefined) liveTarget = opts.target;
    liveSharpen = opts.sharpen ?? 0;
    liveVsr = !!opts.rtxvsr;
    liveVsrQ = vsrQ(opts.vsrquality);
    liveFsrUp = !!opts.fsrup;
    liveA4kUp = !!opts.anime4k;
    liveA4kAA = !!opts.a4kaa;
    liveAcnUp = !!opts.artcnn;
    liveXessUp = !!opts.xess;
    liveUpH = opts.uph ?? 0;
    liveRestore = !!opts.restore;
    liveRestoreModel = opts.restoreModel ?? '';
    liveDlssnr = !!opts.dlssnr;
    liveNrStructure = opts.nrstructure ?? 1;
    liveNrTone = opts.nrtone ?? 1;
    liveNrSkin = opts.nrskin ?? -1;
    liveNrAutoMask = opts.nrautomask === 0 ? 0 : 1;
    liveNrStyle = opts.nrstyle ?? 0;
    liveNrPasses = Math.min(20, Math.max(1, Math.round(opts.nrpasses ?? 1)));
    liveRtxHdr = !!opts.rtxhdr;
    liveHdrColor = opts.hdrcolor ?? 'raw';
    liveHdrSat = opts.hdrsat ?? 100;
    liveHdrCon = opts.hdrcon ?? 100;
    liveHdrVib = opts.hdrvib ?? 0;
    liveHdrSb = opts.hdrsb ?? 0;
    liveHud = opts.hud !== false;
    liveHudLat = opts.hudlat !== false;
    liveGpuFit = opts.gpufit !== false;
    // the panel moved to another model: whatever the idle resident host keeps loaded (its
    // engines) goes right away (user rule: a model must not hold VRAM
    // through a session of another model)
    if (liveIdle && liveModel !== liveHeld) liveQuitIdle();
    liveSendSets();
  },
);

// the running session's values a stdin `set` line changes in place (the host's nativeLiveSets): the Sharpen strength
// (0 = off), RTX VSR's quality while RTX VSR is the upscaler, the RTX HDR sliders while RTX HDR runs, DLSS
// 5 on / off and its Structure / Tone / passes (the host builds more passes in place, memory permitting), Anime4K's
// Mode A+A, the upscaler / Restore by kind (a Restore the session started without starts in place), the HUD's latency
// segment while the HUD shows; a value whose effect the session does not run takes effect with the next session's
// arguments
let liveSetVals: Record<string, number> | null = null;
// AnimeJaNai's file as the host holds it (the spawn's --restore-model, or the last `restoremodel` sent): the host needs
// it before `restore 2` to start AnimeJaNai in place
let liveSetAj = '';
const liveAjFile = () => (liveRestore && liveRestoreModel === 'animejanai' && fileExists(ajPath()) ? ajPath() : '');
function liveSetNow(): Record<string, number> {
  const out: Record<string, number> = {};
  out.sharpen = Math.min(2, Math.max(0, Number(liveSharpen) || 0));
  if (liveVsr) out.vsrq = liveVsrQ;
  if (liveRtxHdr) Object.assign(out, { hdrsat: liveHdrSat, hdrcon: liveHdrCon, hdrvib: liveHdrVib, hdrsb: liveHdrSb });
  // DLSS 5's values always (held by the host while DLSS 5 is off, for its start in place), then DLSS 5 on / off
  Object.assign(out, {
    nrstyle: liveNrStyle,
    nrstruct: liveNrStructure,
    nrtone: liveNrTone,
    nrskin: liveNrSkin,
    nrautomask: liveNrAutoMask,
    nrpasses: liveNrPasses,
    dlssnr: liveDlssnr ? 1 : 0,
  });
  // Anime4K's Mode A+A as the spawn passes it, before the upscaler's kind: an Anime4K started in place takes the
  // value the host holds
  out.a4kaa = liveA4kUp && liveA4kAA ? 1 : 0;
  // the upscaler, Restore and RTX HDR by kind (the host switches the kind its session runs off and on again in place)
  out.up = liveVsr ? 1 : liveFsrUp ? 2 : liveA4kUp ? 3 : liveAcnUp ? 4 : liveXessUp ? 5 : 0;
  out.restore = liveRestore ? (liveRestoreModel === 'animejanai' ? 2 : liveRestoreModel === 'anime4k' ? 3 : 1) : 0;
  out.rtxhdr = liveRtxHdr ? 1 : 0;
  out.hud = liveHud ? 1 : 0;
  if (liveHud) out.hudlat = liveHudLat ? 1 : 0;
  return out;
}
function liveSendSets() {
  if (!liveProc || liveIdle || liveStopping || !liveSetVals) return;
  const aj = liveAjFile();
  if (aj && aj !== liveSetAj && liveCommand(`set\trestoremodel\t${aj}`)) liveSetAj = aj;
  // a key the spawn did not carry (an effect ticked since) goes too: the host takes it in place or says why not
  for (const [k, v] of Object.entries(liveSetNow())) {
    if (liveSetVals[k] === v) continue;
    if (liveCommand(`set\t${k}\t${k === 'sharpen' ? v.toFixed(2) : v}`)) liveSetVals[k] = v;
  }
}

// The global Live toggle hotkey, Lossless-Scaling-style: press it in any app and the window
// you are in goes live; press it again to stop. Global shortcuts swallow the key system-wide
// while the app runs - inherent to the feature, hence the panel lets the user pick the key.
// Default ` (backtick); the renderer restores a persisted choice at boot via 'lv-hotkey'.
let liveHotkey = '';
function liveToggle() {
  if (liveProc && !liveIdle) stopLive();
  else startLiveSession(null);
}
function registerLiveHotkey(acc: string): boolean {
  if (!fileExists(LIVE_EXE)) return false;
  if (liveHotkey) globalShortcut.unregister(liveHotkey);
  const ok = globalShortcut.register(acc, liveToggle);
  if (ok) liveHotkey = acc;
  else if (liveHotkey) globalShortcut.register(liveHotkey, liveToggle); // keep the old key working
  console.log(ok ? `live hotkey registered: ${acc}` : `live hotkey registration FAILED: ${acc}`);
  return ok;
}
ipcMain.handle('lv-hotkey', (_e, acc: string) => registerLiveHotkey(String(acc || '`')));
app.whenReady().then(() => registerLiveHotkey('`'));
app.on('will-quit', () => globalShortcut.unregisterAll());

// Fill's mouse mapping (the exe's FillMouse) holds the pointer speed and a cursor clip while the cursor is on the
// stretched picture and keeps them in this marker until it gives them back: a host that was killed (the stop grace,
// the app quitting) or crashed cannot, so the exe's --restore-mouse does it (GetTempPath's order: TMP, then TEMP)
const LIVE_MOUSE_MARKER = path.join(process.env.TMP || process.env.TEMP || os.tmpdir(), 'smv-live-mouse.txt');
function liveRestoreMouse(sync: boolean) {
  if (!fileExists(LIVE_MOUSE_MARKER)) return;
  liveLog('restoring the pointer speed and clip a killed Fill session held');
  try {
    if (sync) execFileSync(LIVE_EXE, ['--restore-mouse'], { cwd: LIVE_DIR, timeout: 5000, windowsHide: true });
    else execFile(LIVE_EXE, ['--restore-mouse'], { cwd: LIVE_DIR, timeout: 5000, windowsHide: true }, () => {});
  } catch {
    /* the next live session restores it */
  }
}

// AMD FSR 4's first session after a driver update waits while the driver compiles FSR's shaders (frame generation
// ~15 s for vkd3d-proton's Vulkan pipelines at any size, the upscaler ~47 s in D3D12 per exe location and upscale ratio); the
// driver's own cache keeps them after. A Live session at another ratio compiles in its first frame, with the host's
// stall watchdog held for it. While an FSR route is picked (the model, the upscale box) the renderer asks for a warm-up: ONE tiny
// offline session of the picked routes (2 black 320x180 frames, the upscaler to 640x360) in a below-normal host, once
// per driver and FSR build: the route keys in the model cache's fsr4_warm.txt (emptied with the cache, ~1 s to redo).
const FSR_WARM_W = 320;
const FSR_WARM_H = 180;
const FSR_WARM_STAMP = 'fsr4_warm.txt';
let fsrWarmProc: ChildProcess | null = null;
const fsrWarmWant = { fg: false, up: false }; // asked while a warm-up, a live session or a render ran: tried after it
function fsrWarmKeys(): { fg: string; up: string } | null {
  // the driver = System32's nvapi64.dll (every driver install replaces it), a route = its DLLs, by size + write time
  const id = (f: string) => {
    const s = fs.statSync(f);
    return `${path.basename(f)} ${s.size} ${Math.round(s.mtimeMs)}`;
  };
  const ids = (dir: string, files: string[]) => files.map((f) => id(path.join(ENGINE, dir, f))).join(', ');
  try {
    const drv = id(path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'nvapi64.dll'));
    return {
      fg: `fg ${drv}, ${ids('fsrfg', ['smv_fsrfg_bridge.dll', 'smv_vkd3d_d3d12core.dll', 'amd_fidelityfx_framegeneration_dx12.dll'])}`,
      up: `up ${drv}, ${ids('fsrup', ['smv_fsrup_bridge.dll', 'amd_fidelityfx_upscaler_dx12.dll'])}`,
    };
  } catch {
    return null; // no NVIDIA driver or an FSR file missing: the session itself says so
  }
}
function fsrWarm(fg: boolean, up: boolean) {
  fsrWarmWant.fg ||= fg;
  fsrWarmWant.up ||= up;
  if (fsrWarmProc || (liveProc && !liveIdle) || current || !fileExists(LIVE_EXE)) return;
  const keys = fsrWarmKeys();
  if (!keys) return;
  const cache = engineCacheDir(ENGINE);
  const stampPath = path.join(cache, FSR_WARM_STAMP);
  let have: string[] = [];
  try {
    have = fs.readFileSync(stampPath, 'utf8').split(/\r?\n/).filter(Boolean);
  } catch {
    /* never warmed with this cache */
  }
  const doFg = fsrWarmWant.fg && !have.includes(keys.fg);
  const doUp = fsrWarmWant.up && !have.includes(keys.up);
  fsrWarmWant.fg = fsrWarmWant.up = false;
  if (!doFg && !doUp) return;
  try {
    fs.mkdirSync(cache, { recursive: true });
  } catch {
    return;
  }
  const args = ['--offline', '--w', String(FSR_WARM_W), '--h', String(FSR_WARM_H), '--frames', '2'];
  args.push('--pixfmt', 'rgb24', '--out-pixfmt', 'rgb24', '--cache', cache);
  args.push(...(doFg ? ['--multi', '2', '--fsrfg'] : ['--no-interp']));
  if (doUp) args.push('--out-w', String(2 * FSR_WARM_W), '--out-h', String(2 * FSR_WARM_H), '--fsr-upscale');
  const what = [doFg ? 'frame generation' : '', doUp ? 'upscaler' : ''].filter(Boolean).join(' + ');
  const t0 = Date.now();
  let err = '';
  const p = spawn(LIVE_EXE, args, { cwd: LIVE_DIR, stdio: ['pipe', 'ignore', 'pipe'], windowsHide: true });
  fsrWarmProc = p;
  try {
    if (p.pid) os.setPriority(p.pid, os.constants.priority.PRIORITY_BELOW_NORMAL);
  } catch {
    /* normal priority then */
  }
  p.stderr?.on('data', (d) => (err += d));
  p.stdin?.on('error', () => {});
  p.stdin?.end(Buffer.alloc(FSR_WARM_W * FSR_WARM_H * 3 * 2));
  p.on('error', () => {});
  p.on('close', (code) => {
    fsrWarmProc = null;
    // a run that ended by itself is stamped, a failed one too (its session would fail the same way); a killed one not
    if (code !== null) {
      const add = [doFg ? keys.fg : '', doUp ? keys.up : ''].filter(Boolean);
      const keep = have.filter((l) => !add.some((a) => a.slice(0, 3) === l.slice(0, 3)));
      try {
        fs.writeFileSync(stampPath, [...keep, ...add].join('\n') + '\n');
      } catch {
        /* warmed again next time */
      }
    }
    liveLog(
      `AMD FSR 4 shader warm-up (${what}): exit ${code === null ? 'killed' : code}, ${((Date.now() - t0) / 1000).toFixed(1)} s`,
    );
    if (code !== 0) liveLog(err.slice(-2000));
    if (fsrWarmWant.fg || fsrWarmWant.up) fsrWarm(false, false);
  });
}
ipcMain.on('fsr-warm', (_e, o: { fg?: boolean; up?: boolean }) => fsrWarm(!!o?.fg, !!o?.up));

// hard = the app is leaving: kill outright (no "stop" grace, the overlay must not outlive the GUI)
function stopLive(hard = false) {
  liveStopping = true;
  if (!liveProc) return;
  if (hard) {
    try {
      liveProc.kill();
    } catch {
      /* already gone */
    }
    liveRestoreMouse(true);
    return;
  }
  if (liveIdle) liveQuitIdle();
  else if (liveProc.pid) liveEndSession();
}
ipcMain.handle('lv-stop', () => stopLive());
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

// --- Dolby Vision + HDR10+ export tools: readiness + one-click install -----------------------------
// DV Profile 8.1 export layers a Dolby Vision RPU on top of the HDR10 render, then tags the MP4 with a
// dvvC box the engine writes itself (see hdr10_meta.inject_dv_config) - so the bundled ffmpeg is enough
// to mux DV and NO GPAC/MP4Box is needed. HDR10+ export embeds ST 2094-40 dynamic metadata into the
// HDR10 render; the engine collects the per-frame stats itself and hdr10plus_tool injects the SEI (see
// _hp_export). The two non-bundled pieces are quietvoid's MIT tools: one click downloads the pinned
// Windows x64 release zip, checks its SHA256 (the GitHub API digest) before anything is extracted,
// extracts the exe with Windows' bsdtar and copies it into engine/dvtools / engine/hptools. "Ready" =
// the exe present. Bump version, url, page, hash and size together.
type PinnedZip = { version: string; url: string; page: string; zipSha256: string; zipBytes: number };
const DV_DIR = path.join(ENGINE, 'dvtools');
const DOVI_BIN = 'dovi_tool.exe';
const DOVI_DL: PinnedZip = {
  version: '2.3.4',
  url: 'https://github.com/quietvoid/dovi_tool/releases/download/2.3.4/dovi_tool-2.3.4-x86_64-pc-windows-msvc.zip',
  page: 'https://github.com/quietvoid/dovi_tool/releases/tag/2.3.4',
  zipSha256: '0d9733d311dfe49f9dca0c350f7e6d978182673d333c6f07f7365daf9e6c81eb',
  zipBytes: 1346827,
};
const HP_DIR = path.join(ENGINE, 'hptools');
const HP_BIN = 'hdr10plus_tool.exe';
const HP_DL: PinnedZip = {
  version: '1.7.2',
  url: 'https://github.com/quietvoid/hdr10plus_tool/releases/download/1.7.2/hdr10plus_tool-1.7.2-x86_64-pc-windows-msvc.zip',
  page: 'https://github.com/quietvoid/hdr10plus_tool/releases/tag/1.7.2',
  zipSha256: '82b2d560073941b14c6511a431f429e33e134e5caefb60d7e8f6f6e6da8e16ba',
  zipBytes: 948827,
};

// Recursive case-insensitive search for a file by name under root (a release zip can nest its payload).
const findDvBin = (root: string, name: string): string | null => findFilesNamed(root, [name])[0] || null;

// One-click install of a pinned release zip: download into a temp dir (progress to the renderer as
// `channel`), check the zip's SHA256 before anything is extracted (a replaced asset installs nothing),
// extract it with bsdtar and copy each wanted file into destDir: `files` = [name in the zip, name in
// destDir] pairs, all found before any is copied. The temp dir is removed again.
async function installPinnedZip(
  dl: PinnedZip,
  files: Array<[string, string]>,
  destDir: string,
  channel: string,
): Promise<InstallResult> {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'smv-tool-'));
  const zip = path.join(tmp, 'tool.zip');
  const progress = (received: number, total: number, stage: string) => {
    try {
      if (win && !win.webContents.isDestroyed()) win.webContents.send(channel, { received, total, stage });
    } catch {
      /* window gone */
    }
  };
  try {
    const res = await fetch(dl.url, { headers: { 'user-agent': 'SmoothMyVideo' } });
    if (!res.ok || !res.body) return { ok: false, error: `download failed: HTTP ${res.status}`, copied: [] };
    const total = parseInt(res.headers.get('content-length') || '0', 10) || dl.zipBytes;
    let received = 0;
    const src = Readable.fromWeb(res.body as never);
    src.on('data', (chunk: Buffer) => {
      received += chunk.length;
      progress(received, total, 'download');
    });
    await pipeline(src, fs.createWriteStream(zip));
    progress(received, total, 'verify');
    const zh = sha256File(zip);
    if (zh !== dl.zipSha256)
      return { ok: false, error: `checksum mismatch (${zh.slice(0, 12)}...), nothing installed`, copied: [] };
    execFileSync(SYS_TAR, ['-xf', zip, '-C', tmp]);
    const found = files.map(([member]) => findDvBin(tmp, member));
    const missing = files.filter((_, k) => !found[k]).map(([member]) => member);
    if (missing.length)
      return { ok: false, error: missing.join(', ') + ' not in the downloaded zip, nothing installed', copied: [] };
    fs.mkdirSync(destDir, { recursive: true });
    files.forEach(([, name], k) => {
      const dest = path.join(destDir, name);
      try {
        if (fileExists(dest)) fs.chmodSync(dest, 0o666);
      } catch {
        /* best effort */
      }
      fs.copyFileSync(found[k] as string, dest);
    });
    return { ok: true, copied: files.map(([, name]) => name) };
  } catch (e) {
    return { ok: false, error: String(e), copied: [] };
  } finally {
    try {
      fs.rmSync(tmp, { recursive: true, force: true });
    } catch {
      /* temp cleanup best effort */
    }
  }
}
const toolMb = (dl: PinnedZip) => (dl.zipBytes / 1048576).toFixed(1);

ipcMain.handle('dv-ready', () => {
  const dovi = fileExists(path.join(DV_DIR, DOVI_BIN));
  return { dovi, ready: dovi, dir: DV_DIR, ver: DOVI_DL.version, mb: toolMb(DOVI_DL) };
});

ipcMain.handle('dv-open-download', () => {
  shell.openExternal(DOVI_DL.page);
  return true;
});

ipcMain.handle('dv-download', () => installPinnedZip(DOVI_DL, [[DOVI_BIN, DOVI_BIN]], DV_DIR, 'dv-progress'));

ipcMain.handle('hp-ready', () => {
  const ready = fileExists(path.join(HP_DIR, HP_BIN));
  return { ready, dir: HP_DIR, ver: HP_DL.version, mb: toolMb(HP_DL) };
});

ipcMain.handle('hp-open-download', () => {
  shell.openExternal(HP_DL.page);
  return true;
});

ipcMain.handle('hp-download', () => installPinnedZip(HP_DL, [[HP_BIN, HP_BIN]], HP_DIR, 'hp-progress'));

// --- Intel XeSS: readiness + one-click install ----------------------------------------------------
// XeSS-SR runs through the bundled engine/xess/smv_xess_bridge.dll and Intel's libxess.dll, which is not
// bundled: one click downloads Intel's pinned XeSS SDK release zip (github.com/intel/xess), checks its
// SHA256 (the GitHub API digest) before anything is extracted, and copies libxess.dll (XeSS-SR),
// libxess_fg.dll + libxell.dll (XeSS frame generation and its required XeLL) and Intel's licence into
// engine/xess, unmodified (the licence allows redistribution without modification). "Ready" = libxess.dll
// present (XeSS-SR); "fg" = libxess_fg.dll + libxell.dll present (XeSS-FG through the bundled
// smv_xefg_bridge.dll). Bump version, url, page, hash and size together.
const XESS_DIR = path.join(ENGINE, 'xess');
const XESS_DL: PinnedZip = {
  version: '3.0.2',
  url: 'https://github.com/intel/xess/releases/download/v3.0.2/XeSS_SDK_3.0.2.zip',
  page: 'https://github.com/intel/xess/releases/tag/v3.0.2',
  zipSha256: '88b8a373f30e33f3558a77a93e634f11b8132fc3047ea1a8edeead32b8471990',
  zipBytes: 76839749,
};
const XESS_FILES: Array<[string, string]> = [
  ['libxess.dll', 'libxess.dll'],
  ['libxess_fg.dll', 'libxess_fg.dll'],
  ['libxell.dll', 'libxell.dll'],
  ['LICENSE.txt', 'XeSS_LICENSE.txt'],
];

ipcMain.handle('xess-ready', () => {
  const ready = fileExists(path.join(XESS_DIR, 'libxess.dll'));
  const fg = ['libxess_fg.dll', 'libxell.dll'].every((f) => fileExists(path.join(XESS_DIR, f)));
  return { ready, fg, dir: XESS_DIR, ver: XESS_DL.version, mb: toolMb(XESS_DL) };
});

ipcMain.handle('xess-open-download', () => {
  shell.openExternal(XESS_DL.page);
  return true;
});

ipcMain.handle('xess-download', () => installPinnedZip(XESS_DL, XESS_FILES, XESS_DIR, 'xess-progress'));

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

// Live progress thumbnail: the render overwrites this PNG about once a second (see
// SMV_LIVE_PREVIEW below); the renderer polls it by mtime and shows the frame being written.
const LIVE_PNG = path.join(app.getPath('userData'), 'preview', 'live.png');
ipcMain.handle('live-path', () => LIVE_PNG);

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

// The renderer's render request (one field per GUI control; see engineArgs for what each becomes).
type RunOpts = {
  input: string;
  multi: number;
  output: string;
  fps?: number;
  sharpen?: number;
  restore?: boolean;
  restoreModel?: string; // Restore's model id ('animejanai' / 'anime4k'; none = Real-ESRGAN)
  dlssmode?: string; // the DLSS mode (dlssScaleArg)
  dlssnr?: boolean;
  nrstructure?: number;
  nrtone?: number;
  nrskin?: number; // DLSS 5 Skin Structure 0..2, -1 = NVIDIA's default
  nrautomask?: number; // DLSS 5 Auto mask 1 / 0
  nrstyle?: number; // DLSS 5 Style 0 Default / 1 Natural / 2 Cinematic
  nrpasses?: number; // DLSS 5 passes 1..20
  interp?: boolean;
  model?: string;
  upscale?: number;
  rtxvsr?: boolean;
  vsrquality?: number; // RTX VSR's quality level 1..4
  fsrup?: boolean;
  anime4k?: boolean;
  a4kaa?: boolean; // its Mode A+A box
  artcnn?: boolean;
  xess?: boolean;
  rtxhdr?: boolean;
  dv?: boolean;
  dvprofile?: string; // Dolby Vision 8.1 (HDR10 base, the default) or 8.4 (HLG base)
  hp?: boolean;
  codec?: string;
  nogpufit?: boolean;
  hdrcolor?: string;
  hdrsat?: number;
  hdrcon?: number;
  hdrnits?: number;
  hdrsb?: number;
  hdrvib?: number;
};

// The GUI's DLSS mode as the CLI's --scale value: quality / balanced / performance / ultra by name, a
// whole percent 1..99 (the slider between the modes) as its share; null = DLAA, the default.
function dlssScaleArg(mode?: string): string | null {
  if (mode && ['quality', 'balanced', 'performance', 'ultra'].includes(mode)) return mode;
  const pct = mode && /^\d+$/.test(mode) ? Number(mode) : 0;
  return pct >= 1 && pct <= 99 ? (pct / 100).toFixed(2) : null;
}

// The render command line for one request (pure: the GUI state in, render.py's argv out; the TS
// orchestrator dist/render/cli.js takes the same argv).
function engineArgs(opts: RunOpts): string[] {
  const args = [opts.input, String(opts.multi), opts.output];
  // Output codec family (hevc default / av1 / vvc); the engine owns encoder pick + fallbacks.
  if (opts.codec && opts.codec !== 'hevc') args.push('--codec', opts.codec);
  // Interpolation is the default; interp === false means the user only wants the sharpen pass,
  // so tell the engine to skip frame generation (and ignore any fps/multi) entirely.
  if (opts.interp === false) args.push('--no-interp');
  else {
    if (opts.model === 'rife') args.push('--rife'); // RIFE 4.25 backend instead of GMFSS (bundled)
    if (opts.model === 'rifedrba') args.push('--rife-drba'); // RIFE with DRBA anime-pacing timing
    if (opts.model === 'dlssg') args.push('--dlssg'); // "DLSS 4.5" (Frame Generation) backend instead of GMFSS
    if (opts.model === 'fruc') args.push('--fruc'); // "NVIDIA Smooth Motion" backend instead of GMFSS
    if (opts.model === 'fsrfg') args.push('--fsrfg'); // AMD FSR frame generation (our bundled bridge)
    if (opts.model === 'xefg') args.push('--xess-fg'); // Intel XeSS-FG (our bridge + Intel's downloaded DLLs)
    if (opts.model === 'nvof') args.push('--nvof'); // NVIDIA Optical Flow: hardware flow + splat, native host only
    if (opts.fps && opts.fps > 0) args.push('--fps', String(opts.fps));
  }
  // DLSS mode (the mode selector): the working size Restore's output, DLSS 5 and the model run at,
  // NVIDIA's share of the output (plan.ts workPlan); DLAA, the default, is the output itself
  const scaleArg = dlssScaleArg(opts.dlssmode);
  if (scaleArg) args.push('--scale', scaleArg);
  // Adaptive Sharpen strength 0..2 (GUI checkbox + slider, 1 = the author's default). 0/omitted = off,
  // leaving the frames value-preserving; >0 enables the host's sharpen pass. Works with or without interp.
  if (opts.sharpen && opts.sharpen > 0) args.push('--sharpen', String(opts.sharpen));
  // AI detail restoration (GUI Restore rows): Real-ESRGAN animevideov3, or the installed AnimeJaNai, once per source
  // frame, first (it folds straight to the working size). Works with or without interpolation.
  if (opts.restore) args.push('--restore', ...restoreModelArgs(opts.restoreModel));
  // NVIDIA DLSS 5 Neural Rendering (GUI checkbox + the two sliders): once per source frame at the
  // working size, after Restore and the resize, before the interpolation. The
  // renderer only sends it when the user-supplied runtime is installed (dlssnr-ready).
  if (opts.dlssnr)
    args.push(
      '--dlssnr',
      '--nr-structure',
      String(opts.nrstructure ?? 1),
      '--nr-tone',
      String(opts.nrtone ?? 1),
      '--nr-style',
      String(opts.nrstyle ?? 0),
    );
  if (opts.dlssnr && (opts.nrpasses ?? 1) > 1)
    args.push('--nr-passes', String(Math.min(20, Math.max(1, Math.round(opts.nrpasses ?? 1)))));
  if (opts.dlssnr && (opts.nrskin ?? -1) >= 0) args.push('--nr-skin', String(opts.nrskin));
  if (opts.dlssnr && opts.nrautomask === 0) args.push('--nr-automask', '0');
  // Resize factor (an arbitrary float, source height -> chosen target height), computed by the
  // renderer from the resolution selector. >1 enables the upscale pass; <1 is a downscale the
  // engine FOLDS into the decode (whole pipeline runs at the output size - also what keeps 4K
  // sources inside TRT-safe flow shapes, so dropping it here re-breaks 4K GMFSS renders).
  // Without --rtx-vsr an upscale is a Lanczos3 resize; with it, RTX Video Super Resolution.
  if (opts.upscale && opts.upscale > 0 && opts.upscale !== 1) args.push('--upscale', String(opts.upscale));
  // RTX VSR: the real RTX Video SDK (the engine/rtxvideo CUDA bridge) for an enlarging resize: the
  // final one from the working size (a mode below DLAA or an upscale), else the one before the
  // model; the plan skips it when nothing enlarges. Falls back to Lanczos3 if the bridge or the RTX
  // Video runtime is unavailable. AMD FSR 4 (bundled), Anime4K or ArtCNN takes the same resize when its box is ticked
  // instead.
  if (opts.rtxvsr) args.push('--rtx-vsr', ...vsrQArgs(vsrQ(opts.vsrquality)));
  else if (opts.fsrup) args.push('--fsr-upscale');
  else if (opts.anime4k) args.push('--anime4k', ...(opts.a4kaa ? ['--anime4k-aa'] : []));
  else if (opts.artcnn) args.push('--artcnn');
  else if (opts.xess) args.push('--xess-upscale');
  // "Fit to the GPU" off: no video memory fit (the batch stays as planned, slower when memory runs short)
  if (opts.nogpufit) args.push('--no-gpu-fit');
  // RTX HDR (TrueHDR): convert the output to HDR10. Works with or without --upscale (when both are
  // on, the RTX bridge does VSR then TrueHDR in one pass). The peak = the Peak box, else the app
  // window's display (as NVIDIA's own RTX Video HDR does), the HDR10 metadata follows it; it falls
  // back to an SDR render if the bridge is unavailable.
  if (opts.rtxhdr) {
    args.push('--rtx-hdr', '--hdr-nits', String(opts.hdrnits && opts.hdrnits > 0 ? opts.hdrnits : displayPeakNits()));
    // HDR colour controls: mode (raw = NVIDIA's result, the default / vivid / rtx), the SDK
    // Saturation (inert in vivid), and Dynamic Vibrance.
    if (opts.hdrcolor && opts.hdrcolor !== 'raw') args.push('--hdr-color', opts.hdrcolor);
    if (typeof opts.hdrsat === 'number' && opts.hdrsat !== 100) args.push('--hdr-saturation', String(opts.hdrsat));
    if (opts.hdrvib && opts.hdrvib > 0) args.push('--hdr-vibrance', String(opts.hdrvib));
    if (opts.hdrsb && opts.hdrsb > 0) args.push('--hdr-satboost', String(opts.hdrsb));
    // RTX HDR tone curve (SDK 0..200, 100 = neutral; the GUI shows the App's -100..100 scale).
    if (typeof opts.hdrcon === 'number' && opts.hdrcon !== 100) args.push('--hdr-contrast', String(opts.hdrcon));
    // Dolby Vision export (needs dovi_tool in engine/dvtools): Profile 8.1 on the HDR10 render, or 8.4 with the video
    // written as HLG.
    if (opts.dv) args.push('--dv');
    if (opts.dv && opts.dvprofile === '8.4') args.push('--dv-profile', '8.4');
    // HDR10+ dynamic metadata on top of the HDR10 render (needs hdr10plus_tool in engine/hptools).
    if (opts.hp) args.push('--hdr10plus');
  }
  return args;
}

// The engine's environment. PYTHONUTF8 keeps the dynamo ONNX exporter's unicode logs from
// crashing the engine during first-run TRT builds. The TRT cache deliberately gets NO override
// here: GUI and CLI runs share the one in-app cache in the app's top folder
// (model_cache_safe_to_delete); a separate AppData cache would split them, so GUI renders
// would rebuild engines the CLI cache already has. SMV_LIVE_PREVIEW makes the render drop a small PNG
// of the frame being written about once a second; the renderer polls it for the live progress
// thumbnail.
function engineEnv(): NodeJS.ProcessEnv {
  try {
    fs.mkdirSync(path.join(app.getPath('userData'), 'preview'), { recursive: true });
  } catch {
    /* exists */
  }
  return {
    ...process.env,
    SMV_LIVE_PREVIEW: LIVE_PNG,
    SMV_PAUSE_FILE: PAUSE_FLAG,
    SMV_LIVE_OFF_FILE: LIVE_OFF_FLAG,
    SMV_OFFLINE_HOST_PIPE: OFFLINE_PIPE_NAME, // the resident offline host of this app process
  };
}

// Relay the engine's output to the renderer and drive the taskbar progress from it; finish the
// run lifecycle on close or spawn error.
function pipeEngineOutput(proc: ChildProcess, send: (channel: string, ...payload: unknown[]) => void) {
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
  const finish = () => {
    current = null;
    clearPause();
    keepAwake(false);
    taskbarProgress(-1);
  };
  proc.stdout!.on('data', onStdout);
  proc.stderr!.on('data', onData);
  proc.on('close', (code) => {
    const tail = stripGridSize(outCarry);
    outCarry = '';
    if (tail) send('engine-out', tail);
    finish();
    send('engine-done', code);
  });
  proc.on('error', (err) => {
    finish();
    send('engine-out', 'spawn error: ' + err);
    send('engine-done', -1);
  });
}

ipcMain.on('run', (e, opts: RunOpts) => {
  const args = engineArgs(opts);
  const env = engineEnv();
  // an idle resident live host gives its VRAM to the render, unless a restart / revive is about to reuse it or it
  // stayed to finish an engine build (that build makes the next Live start warm)
  if (!liveRevivePending && !liveBuildHeld) liveQuitIdle();
  clearPause(); // start unpaused: never inherit a stale flag from a previous (e.g. killed) run
  // The render orchestrator is TypeScript: this Electron binary run as plain
  // node on dist/render/cli.js, which speaks render.py's stderr protocol and exit codes and runs
  // every render on the native host. SMV_ENGINE_DIR = the engine folder
  // (packaged: resources/engine, outside the asar the script is read from).
  const proc = spawn(process.execPath, [RENDER_CLI, ...args], {
    cwd: ENGINE,
    env: { ...env, ELECTRON_RUN_AS_NODE: '1', SMV_ENGINE_DIR: ENGINE },
  });
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
  pipeEngineOutput(proc, send);
});

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
  ]
    .flatMap((s) => (s && s !== '.video.tmp.mp4' ? [s, '.b' + s] : [s])) // + the file split's chunk B (<part>.b.*)
    .map((s) => part + s);
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
// resume block in src/render/resume.ts). The renderer uses this to flip Smooth It! to Resume and to
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

type PreviewOpts = {
  input: string;
  frame?: number | string;
  sharpen?: number;
  hdr?: boolean;
  color?: string;
  saturation?: number;
  nits?: number;
  vibrance?: number;
  satboost?: number;
  contrast?: number;
  upscale?: number;
  rtxvsr?: boolean;
  vsrquality?: number; // RTX VSR's quality level 1..4
  fsrup?: boolean;
  anime4k?: boolean;
  a4kaa?: boolean; // its Mode A+A box
  artcnn?: boolean;
  xess?: boolean;
  restore?: boolean;
  restoreModel?: string; // Restore's model id ('animejanai' / 'anime4k'; none = Real-ESRGAN)
  dlssmode?: string; // the DLSS mode (dlssScaleArg)
  dlssnr?: boolean;
  nrstructure?: number;
  nrtone?: number;
  nrskin?: number; // DLSS 5 Skin Structure 0..2, -1 = NVIDIA's default
  nrautomask?: number; // DLSS 5 Auto mask 1 / 0
  nrstyle?: number; // DLSS 5 Style 0 Default / 1 Natural / 2 Cinematic
  nrpasses?: number; // DLSS 5 passes 1..20
  nrmask?: boolean;
};

// render/preview.js's command line for the current settings, writing its PNGs under `prefix`
function previewArgv(opts: PreviewOpts, prefix: string): string[] {
  const args = [PREVIEW_CLI, opts.input, '--out', prefix, '--frame', String(opts.frame ?? 'mid')];
  if (opts.sharpen && opts.sharpen > 0) args.push('--sharpen', String(opts.sharpen));
  if (opts.restore) args.push('--restore', ...restoreModelArgs(opts.restoreModel));
  const pScale = dlssScaleArg(opts.dlssmode);
  if (pScale) args.push('--scale', pScale);
  if (opts.dlssnr)
    args.push(
      '--dlssnr',
      '--nr-structure',
      String(opts.nrstructure ?? 1),
      '--nr-tone',
      String(opts.nrtone ?? 1),
      '--nr-style',
      String(opts.nrstyle ?? 0),
    );
  if (opts.dlssnr && (opts.nrpasses ?? 1) > 1)
    args.push('--nr-passes', String(Math.min(20, Math.max(1, Math.round(opts.nrpasses ?? 1)))));
  if (opts.dlssnr && (opts.nrskin ?? -1) >= 0) args.push('--nr-skin', String(opts.nrskin));
  if (opts.dlssnr && opts.nrautomask === 0) args.push('--nr-automask', '0');
  if (opts.dlssnr && opts.nrmask) args.push('--nr-mask'); // heat map of the DLSS 5 change, <prefix>_nrmask.png
  if (opts.upscale && opts.upscale > 0 && opts.upscale !== 1) args.push('--upscale', String(opts.upscale));
  if (opts.rtxvsr)
    args.push('--rtx-vsr', ...vsrQArgs(vsrQ(opts.vsrquality))); // the plan runs it on an enlarging resize only
  else if (opts.fsrup) args.push('--fsr-upscale');
  else if (opts.anime4k) args.push('--anime4k', ...(opts.a4kaa ? ['--anime4k-aa'] : []));
  else if (opts.artcnn) args.push('--artcnn');
  else if (opts.xess) args.push('--xess-upscale');
  if (opts.hdr)
    args.push(
      '--rtx-hdr',
      '--hdr-color',
      String(opts.color ?? 'raw'),
      '--hdr-saturation',
      String(opts.saturation ?? 100),
      '--hdr-vibrance',
      String(opts.vibrance ?? 0),
      '--hdr-satboost',
      String(opts.satboost ?? 0),
      '--hdr-contrast',
      String(opts.contrast ?? 100),
      '--hdr-nits',
      String(opts.nits && opts.nits > 0 ? opts.nits : displayPeakNits()),
    );
  return args;
}

// The PNG prefix under userData\preview, one fixed name per kind, reused each call (the renderer cache-busts its img
// src) so previews never pile up.
function previewPrefix(name: string): string {
  const dir = path.join(app.getPath('userData'), 'preview');
  try {
    fs.mkdirSync(dir, { recursive: true });
  } catch {
    /* already exists */
  }
  return path.join(dir, name);
}

// Before/after preview: render ONE source frame at the current spatial settings (RTX HDR when opts.hdr,
// Adaptive Sharpen when opts.sharpen > 0; no interpolation, no encode) and hand back the two PNG paths
// for the renderer's side-by-side pane. render/preview.js (the render's decode and
// the native host's pass chain, no python) writes <prefix>_original.png and _processed.png.
// Resolves with { error } instead when the frame or the RTX bridge is unavailable.
ipcMain.handle('preview', (_e, opts: PreviewOpts) => {
  return new Promise((resolve) => {
    const prefix = previewPrefix('frame');
    const args = previewArgv(opts, prefix);
    // like the render: Electron's own node runs the compiled module (no python)
    const env = { ...process.env, ELECTRON_RUN_AS_NODE: '1', SMV_ENGINE_DIR: ENGINE };
    execFile(process.execPath, args, { cwd: ENGINE, env, maxBuffer: 1 << 20 }, (err, stdout, stderr) => {
      if (err) {
        resolve({ error: String(stderr || err).slice(-400) });
        return;
      }
      const m = /frame (\d+)\/(\d+)/.exec(String(stdout ?? ''));
      // "nrmask=<touched %>/<mean change, 8-bit> <path>" is present only when the DLSS 5 pass ran
      // with --nr-mask; a pass that degraded to the plain frame writes no mask and the pane shows
      // the processed picture as usual.
      const mm = /nrmask=([\d.]+)%\/([\d.]+) /.exec(String(stdout ?? ''));
      resolve({
        original: prefix + '_original.png',
        processed: prefix + '_processed.png',
        frame: m ? Number(m[1]) : null,
        total: m ? Number(m[2]) : null,
        nrmask: mm ? prefix + '_nrmask.png' : null,
        nrmaskPct: mm ? Number(mm[1]) : null,
        nrmaskMean: mm ? Number(mm[2]) : null,
      });
    });
  });
});

// Compare: the pairs from frame on (one, or Save clip's `pairs`) through the Video tab's model (modelA) and the one
// it is compared to (modelB) at the preview's settings and the render's multiplier; render/preview.js compare writes
// the real frames <prefix>_real<p>.png and each model's new frames <prefix>_a<o>.png / _b<o>.png (o = the output frame
// from frame on, never a multiple of multi). Resolves with { error } instead when a model or the frames are unavailable.
ipcMain.handle(
  'compare',
  (_e, opts: PreviewOpts & { modelA: string; modelB: string; multi: number; pairs?: number }) => {
    return new Promise((resolve) => {
      const prefix = previewPrefix((opts.pairs ?? 1) > 1 ? 'clip' : 'compare');
      const args = [
        ...previewArgv(opts, prefix),
        '--model-a',
        opts.modelA,
        '--model-b',
        opts.modelB,
        '--multi',
        String(opts.multi),
        '--pairs',
        String(opts.pairs ?? 1),
      ];
      const env = { ...process.env, ELECTRON_RUN_AS_NODE: '1', SMV_ENGINE_DIR: ENGINE };
      execFile(process.execPath, args, { cwd: ENGINE, env, maxBuffer: 1 << 20 }, (err, stdout, stderr) => {
        if (err) {
          resolve({ error: String(stderr || err).slice(-400) });
          return;
        }
        const m = /compare frame (\d+)\/(\d+) (\d+)x(\d+) multi=(\d+) pairs=(\d+) .*vsr=(\d) restore=(\d) nr=(\d)/.exec(
          String(stdout ?? ''),
        );
        if (!m) {
          resolve({ error: 'no compare result: ' + String(stdout ?? '').slice(-200) });
          return;
        }
        const n = Number(m[5]),
          pairs = Number(m[6]);
        const reals = Array.from({ length: pairs + 1 }, (_, p) => `${prefix}_real${p}.png`);
        const news = (tag: string) =>
          Array.from({ length: pairs * n }, (_, o) => o)
            .filter((o) => o % n)
            .map((o) => `${prefix}_${tag}${o}.png`);
        resolve({
          real0: reals[0],
          real1: reals[1],
          reals,
          a: news('a'),
          b: news('b'),
          frame: Number(m[1]),
          total: Number(m[2]),
          w: Number(m[3]),
          h: Number(m[4]),
          multi: n,
          pairs,
          vsr: m[7] === '1',
          restore: m[8] === '1',
          nr: m[9] === '1',
        });
      });
    });
  },
);

// Compare's Save image / Save clip: where to write the PNG or the lossless MKV the renderer composes
ipcMain.handle('pick-save', async (_e, defaultPath: string, kind: 'image' | 'clip') => {
  const r = await dialog.showSaveDialog(win!, {
    defaultPath,
    filters: [
      kind === 'clip' ? { name: 'Lossless video', extensions: ['mkv'] } : { name: 'PNG image', extensions: ['png'] },
    ],
  });
  return r.canceled ? null : r.filePath || null;
});

// Compare's Save clip: the renderer's composed frames (RGBA, w x h) into a lossless FFV1 MKV at fps, one clip at a
// time: clip-begin starts ffmpeg, each clip-frame resolves once ffmpeg took the frame, clip-end resolves to its exit.
let clip: { p: ChildProcess; done: Promise<number>; ended: boolean; err: string } | null = null;
ipcMain.handle('clip-begin', (_e, o: { out: string; w: number; h: number; fps: string }) => {
  if (clip) clip.p.kill();
  const ffmpeg = fs.existsSync(path.join(ENGINE, 'bin', 'ffmpeg.exe'))
    ? path.join(ENGINE, 'bin', 'ffmpeg.exe')
    : 'ffmpeg';
  const p = spawn(
    ffmpeg,
    // FFV1 keeps every RGB value; gbrp drops only the canvas's alpha
    ['-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', `${o.w}x${o.h}`, '-r', o.fps, '-i', '-'].concat([
      '-c:v',
      'ffv1',
      '-level',
      '3',
      '-pix_fmt',
      'gbrp',
      o.out,
    ]),
    { windowsHide: true, stdio: ['pipe', 'ignore', 'pipe'] },
  );
  const c = { p, done: Promise.resolve(0), ended: false, err: '' };
  c.done = new Promise<number>((res) => {
    p.once('error', (e) => {
      c.err += e.message;
      c.ended = true;
      res(-1);
    });
    p.once('close', (code: number | null) => {
      c.ended = true;
      res(code ?? 1);
    });
  });
  p.stderr!.on('data', (b: Buffer) => (c.err += b.toString('utf8')));
  p.stdin!.on('error', () => {
    /* ffmpeg left early: clip-end reports its exit and stderr */
  });
  clip = c;
  return true;
});
ipcMain.handle('clip-frame', (_e, data: Uint8Array) => {
  const c = clip;
  if (!c || c.ended) return false;
  if (c.p.stdin!.write(data)) return true;
  // ffmpeg's pipe is full: on once it drains, or false when ffmpeg ends first
  return Promise.race([new Promise((res) => c.p.stdin!.once('drain', () => res(true))), c.done.then(() => false)]);
});
ipcMain.handle('clip-end', async () => {
  const c = clip;
  clip = null;
  if (!c) return { rc: -1, error: 'no clip running' };
  c.p.stdin!.end();
  return { rc: await c.done, error: c.err.trim().slice(-300) };
});
