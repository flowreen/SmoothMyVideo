// Main process: the before / after preview and Compare's Save image / Save clip.
import { app, ipcMain, dialog } from 'electron';
import { spawn, execFile, ChildProcess } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import { ENGINE, PREVIEW_CLI } from './main-paths';
import { win } from './main-window';
import { displayPeakNits } from './main-dialogs';
import { restoreModelArgs } from './main-setup';
import { vsrQ, vsrQArgs } from './main-live';
import { dlssScaleArg } from './main-render';

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
// the native host's pass chain) writes <prefix>_original.png and _processed.png.
// Resolves with { error } instead when the frame or the RTX bridge is unavailable.
ipcMain.handle('preview', (_e, opts: PreviewOpts) => {
  return new Promise((resolve) => {
    const prefix = previewPrefix('frame');
    const args = previewArgv(opts, prefix);
    // like the render: Electron's own node runs the compiled module
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
