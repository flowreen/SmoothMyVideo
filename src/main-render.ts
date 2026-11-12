// Main process: a file render (the progress thumbnail, pause and resume, the engine's arguments and output,
// cancel, the resume check).
import { app, ipcMain, powerSaveBlocker, Notification } from 'electron';
import { spawn, execFile, ChildProcess } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import { ENGINE, RENDER_CLI } from './main-paths';
import { win } from './main-window';
import { displayPeakNits } from './main-dialogs';
import { restoreModelArgs } from './main-setup';
import {
  OFFLINE_PIPE_NAME,
  liveBuildHeld,
  liveQuitIdle,
  liveRevivePending,
  liveSpareDrop,
  liveSpareEnsure,
  vsrQ,
  vsrQArgs,
} from './main-live';

export let current: ChildProcess | null = null;
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
export const PAUSE_FLAG = path.join(app.getPath('userData'), 'preview', 'pause.flag');
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
  hdrcolor?: string;
  hdrsat?: number;
  hdrcon?: number;
  hdrnits?: number;
  hdrsb?: number;
  hdrvib?: number;
};

// The GUI's DLSS mode as the CLI's --scale value: quality / balanced / performance / ultra by name, a
// whole percent 1..99 (the slider between the modes) as its share; null = DLAA, the default.
export function dlssScaleArg(mode?: string): string | null {
  if (mode && ['quality', 'balanced', 'performance', 'ultra'].includes(mode)) return mode;
  const pct = mode && /^\d+$/.test(mode) ? Number(mode) : 0;
  return pct >= 1 && pct <= 99 ? (pct / 100).toFixed(2) : null;
}

// The render command line for one request (pure: the GUI state in, the argv of the render
// orchestrator dist/render/cli.js out).
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
    if (opts.model === 'nvvfg') args.push('--nvvfg'); // NVIDIA Video Frame Generation (NVIDIA's downloaded runtime)
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
  // engine FOLDS into the decode (the whole pipeline runs at the output size).
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

// The engine's environment. The model cache deliberately gets NO override
// here: GUI and CLI runs share the one in-app cache in the app's top folder
// (model_cache_safe_to_delete); a separate AppData cache would split them, so GUI renders
// would rebuild what the CLI cache already has. SMV_LIVE_PREVIEW makes the render drop a small PNG
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
    liveSpareEnsure();
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
  liveSpareDrop(); // the render gets the spare host's memory too; one starts again when the render is over
  clearPause(); // start unpaused: never inherit a stale flag from a previous (e.g. killed) run
  // The render orchestrator is TypeScript: this Electron binary run as plain
  // node on dist/render/cli.js, which runs
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
    .flatMap((s) => (s && s !== '.video.tmp.mp4' ? [s, '.b' + s, '.c' + s] : [s])) // + the file split's later chunks (<part>.b.*, <part>.c.*)
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
