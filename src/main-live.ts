// Main process: Live mode (the host process and its spare, settings in place, the hotkey, the FSR 4 warm-up).
import { app, ipcMain, globalShortcut } from 'electron';
import { spawn, execFile, execFileSync, ChildProcess } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import { engineCacheDir } from './render/cache';
import { ENGINE } from './main-paths';
import { win } from './main-window';
import { ajPath, fileExists, restoreModelArgs } from './main-setup';
import { current, dlssScaleArg } from './main-render';

// --- Live mode: the real-time pipeline host (engine/live/smv-live.exe) -----------------------------
// The exe captures the target window via Windows.Graphics.Capture and presents it enhanced into a
// click-through topmost overlay. Backend today: DLSS Frame Generation at (gen+1)x (its own copy of
// NVIDIA's frame generation DLL sits beside it); the design target is backend-agnostic (server route for
// RIFE/GMFSS/FRUC + VSR/HDR/sharpen/restore is the planned phase 2). Gotcha (bisected in the
// smv-live PoC): the overlay must hold foreground for the driver to generate, so the exe
// force-activates its own window at start.
export const LIVE_DIR = path.join(ENGINE, 'live');
export const LIVE_EXE = path.join(LIVE_DIR, 'smv-live.exe');
// Resident offline host: the render (src/render/native.ts) keeps `smv-live.exe --offline
// --resident` alive between renders (engines loaded, about 300 MB of VRAM idle) and finds it through
// this named pipe, one per app process. The render spawns it detached; this process quits it when a live session starts and at app exit
// (a busy host ignores the request and leaves on its own idle limit).
export const OFFLINE_PIPE_NAME = 'smv-offline-' + process.pid;
export function offlineHostQuit() {
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
export let liveProc: ChildProcess | null = null;
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
export let liveBuildHeld = false; // the idle host stayed for an engine build its session left running (its ended line says so)
export let liveRevivePending = false; // a restart / revive timer is armed: it reuses the idle host in 400 ms
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
/** RTX VSR's quality level from the renderer: 1..4 (its slider), or 16..19 = the same levels of NVIDIA's
 * high-bitrate models (the Disable artifact removal tick); anything else = 4 (Ultra, the default). */
export function vsrQ(v: number | undefined): number {
  const q = Math.round(Number(v));
  return (q >= 1 && q <= 4) || (q >= 16 && q <= 19) ? q : 4;
}
/** The host flag for a quality level; none for 4, the host's own default. */
export function vsrQArgs(q: number): string[] {
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
export function liveQuitIdle() {
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
  liveSpareEnsure();
}

// The spare host. A Live session that runs DLSS 4.5 spends a second or two in its device's and NGX's start before
// its frames come. While DLSS 4.5 is the panel's interpolator and Live is in use (the Live tab is open, or a session ran
// since the app started) the app keeps ONE `smv-live.exe --prewarm` waiting with that start already paid: about 240 MB
// of memory, 32 MB of video memory, no window. The next DLSS 4.5 session goes to it as the resident host's "start"
// line and it ends with that session, like every DLSS 4.5 host. A spare nobody took ends when another model is
// picked, a file render starts or the app quits. SMV_LIVE_PREWARM=0 = no spare (every session starts its own host).
let liveSpare: ChildProcess | null = null;
let liveSpareReady = false; // it printed its "prewarmed" line: a session can go to it
let liveSpareOut = ''; // what it printed so far: the session that takes it shows these lines first
let liveSpareFails = 0; // spares that ended before they were ready (a GPU without DLSS 4.5): two = no more tries
let liveUiLive = false; // the window shows the Live tab (lv-opts)
let liveUsed = false; // a Live session started since the app did
let liveSpareHold = 0; // no new spare before this time (ms): a session's own start comes first
function liveSpareWanted(): boolean {
  return (
    liveModel === 'dlssg' &&
    (liveUiLive || liveUsed) &&
    !current &&
    liveSpareFails < 2 &&
    process.env.SMV_LIVE_PREWARM !== '0' &&
    fileExists(LIVE_EXE)
  );
}
export function liveSpareEnsure() {
  if (liveSpare || !liveSpareWanted() || Date.now() < liveSpareHold) return;
  const p = spawn(LIVE_EXE, ['--prewarm'], { cwd: LIVE_DIR, windowsHide: true });
  liveSpare = p;
  liveSpareReady = false;
  liveSpareOut = '';
  liveLog('spare host started: smv-live.exe --prewarm');
  p.stdin?.on('error', () => {});
  p.stderr?.on('data', (b: Buffer) => {
    if (liveSpare !== p) return;
    liveSpareOut += b.toString();
    if (!liveSpareReady && liveSpareOut.includes('prewarmed for DLSS 4.5')) {
      liveSpareReady = true;
      liveSpareFails = 0;
      liveLog('spare host ready');
    }
  });
  const gone = (why: string) => {
    if (liveSpare !== p) return; // a session took it: that session's handlers report its end
    if (!liveSpareReady) liveSpareFails++;
    liveLog(`spare host ${why}\n${liveSpareOut}`);
    liveSpare = null;
    liveSpareReady = false;
    liveSpareOut = '';
  };
  p.on('close', (code) => gone(`exited (${code})`));
  p.on('error', (err) => gone('spawn error: ' + err));
}
export function liveSpareDrop() {
  if (!liveSpare) return;
  const p = liveSpare;
  liveSpare = null;
  liveSpareReady = false;
  liveSpareOut = '';
  liveLog('spare host ended (not wanted now)');
  try {
    p.stdin?.write('quit\n');
  } catch {
    /* gone */
  }
  const t = setTimeout(() => {
    try {
      if (p.exitCode === null) p.kill();
    } catch {
      /* gone */
    }
  }, 2000);
  t.unref();
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
  // a DLSS 4.5 session goes to the spare host when one is ready
  const spare = liveModel === 'dlssg' && !resident && liveSpare && liveSpareReady ? liveSpare : null;
  const spareOut = spare ? liveSpareOut : '';
  if (spare) {
    liveSpare = null;
    liveSpareReady = false;
    liveSpareOut = '';
    spare.stderr?.removeAllListeners('data');
  }
  // windowsHide: a packaged app has no console, so a console-subsystem child would get a window of its own
  const p = spare ?? spawn(LIVE_EXE, args, { cwd: LIVE_DIR, windowsHide: true });
  liveProc = p;
  liveResident = resident;
  liveIdle = false;
  liveUsed = true;
  let errBuf = ''; // partial trailing line, so the hwnd is only parsed out of COMPLETE lines
  liveLog(`session started${spare ? ' on the spare host' : ''}: smv-live.exe ${args.join(' ')}`);
  sendLive('lv-started');
  // a write into a host that just exited raises EPIPE on the stream: never let it throw
  p.stdin?.on('error', () => {});
  const onErr = (t: string) => {
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
  };
  if (spareOut) onErr(spareOut);
  p.stderr?.on('data', (b: Buffer) => onErr(b.toString()));
  if (spare) {
    try {
      spare.stdin?.write('start\t' + args.join('\t') + '\n');
    } catch {
      /* the spare died under us; its close handler below reports the session's end */
    }
  }
  // the next spare only once this session's own start is over (two starts at once slow both): a resize
  // or stall revive then finds one ready
  if (liveModel === 'dlssg') {
    liveSpareHold = Date.now() + 5000;
    setTimeout(liveSpareEnsure, 5100).unref();
  }
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
      live?: boolean; // the window shows the Live tab
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
    liveUiLive = !!opts.live;
    if (liveSpareWanted()) liveSpareEnsure();
    else liveSpareDrop();
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
export let fsrWarmProc: ChildProcess | null = null;
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
export function stopLive(hard = false) {
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
