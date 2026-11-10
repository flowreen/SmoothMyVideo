// Main process: the update check and the automatic update.
import { app, ipcMain, net } from 'electron';
import { spawn, execFile, ChildProcess } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import * as crypto from 'crypto';
import * as ofs from 'original-fs';
import { Readable } from 'stream';
import { pipeline } from 'stream/promises';
import { ENGINE } from './main-paths';
import { win } from './main-window';
import { fileExists, sha256File } from './main-setup';
import { fsrWarmProc, liveProc, liveSpareDrop, offlineHostQuit, stopLive } from './main-live';
import { PAUSE_FLAG, current } from './main-render';

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
export function applyPendingUpdate(): boolean {
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
export function finishUpdate(): void {
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
export function checkForUpdate(): void {
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
  liveSpareDrop();
  offlineHostQuit();
  fsrWarmProc?.kill();
  if (!launchSwap(top, ready.rel, true)) return;
  app.quit();
});
