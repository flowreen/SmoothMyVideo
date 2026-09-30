// The offline render: ported from render.py,
// every stderr line as render.py wrote it. This process probes, plans, picks the encoder, resumes,
// spawns the decoder / encoder and the host (smv-live.exe --offline, one-shot or the resident host
// behind its named pipe), relays the host's lines and finishes the container; no frame passes
// through it (only the progress thumbnail's small dump, thumbPng).
import { ChildProcess, spawn, spawnSync } from 'child_process';
import { createHash } from 'crypto';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import * as readline from 'readline';
import { dvExport, hpExport } from './dvhp';
import {
  chooseEncoder,
  decodeCmd,
  decodeFilters,
  encodePlan,
  finalizeOutput,
  fullChroma,
  HdrStats,
  HpFrame,
  pyErrText,
  pyReprStr,
  Say,
} from './encode';
import { autoCandidates, isGmfss, nvofRefusal, RenderArgs, WorkPlan, workPlan } from './plan';
import { thumbPng } from './preview';
import {
  decodeRgbVf,
  frameCount,
  needMkv,
  outputRate,
  probe,
  probeTracks,
  sourceBits,
  Stream,
  tag,
  vfrConform,
} from './probe';
import { pyFixed, pyFloatRepr, pyG, pyRound } from './pyfmt';
import {
  applyResume,
  ArgNs,
  concatCopy,
  emitPausePreview,
  nativeHdrPrefix,
  parseRenderArgv,
  partsOf,
  Progress,
  pyFloatParse,
  resumeCleanup,
  resumeSig,
  tryResume,
  winNormAbs,
  writeResumeSidecar,
} from './resume';

/** sys.exit: a string is written to stderr with a newline and exits 1, a number is the exit code. */
export class RenderExit extends Error {
  constructor(public code: number | string) {
    super(typeof code === 'string' ? code : `exit ${code}`);
  }
}

const FRAG_FLAGS = ['-movflags', '+frag_keyframe+empty_moov+default_base_moof'];
const HDR_NITS = 1000;
const HDR_MASTER_PRIM = 'display-p3';

export function engineDir(): string {
  return process.env.SMV_ENGINE_DIR || path.resolve(__dirname, '..', '..', 'engine');
}

export function tool(engine: string, name: string): string {
  const exe = path.join(engine, 'bin', name + '.exe');
  return fs.existsSync(exe) && fs.statSync(exe).isFile() ? exe : name;
}

function isFile(p: string): boolean {
  try {
    return fs.statSync(p).isFile();
  } catch {
    return false;
  }
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

/** The resident offline host's control pipe name (render.py _offline_pipe_name). */
function offlinePipeName(exe: string): string {
  return (
    process.env.SMV_OFFLINE_HOST_PIPE ||
    'smv-offline-' + createHash('md5').update(winNormAbs(exe), 'utf8').digest('hex').slice(0, 8)
  );
}

function openPipe(p: string, flags: string): number | null | -1 {
  try {
    return fs.openSync(p, flags);
  } catch (e) {
    return (e as NodeJS.ErrnoException).code === 'ENOENT' ? null : -1; // -1 = exists but busy
  }
}

/** Ask an idle resident offline host to exit (render.py _offline_host_quit). */
export function offlineHostQuit(exe: string): void {
  const fd = openPipe('\\\\.\\pipe\\' + offlinePipeName(exe), 'r+');
  if (fd === null || fd === -1) return;
  try {
    fs.writeSync(fd, 'quit\n');
  } catch {
    /* except OSError: pass */
  } finally {
    fs.closeSync(fd);
  }
}

function sleepMs(ms: number): void {
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms);
}

function waitExit(p: ChildProcess): Promise<number> {
  return new Promise((res) => {
    if (p.exitCode !== null) res(p.exitCode);
    else p.once('close', (c: number | null) => res(c ?? 1));
  });
}

/** Line reader over a blocking pipe fd (the resident host's control pipe). */
function* fdLines(fd: number): Generator<string> {
  const buf = Buffer.alloc(4096);
  let acc = Buffer.alloc(0);
  for (;;) {
    let n: number;
    try {
      n = fs.readSync(fd, buf, 0, buf.length, null);
    } catch {
      return;
    }
    if (!n) return;
    acc = Buffer.concat([acc, buf.subarray(0, n)]);
    let i: number;
    while ((i = acc.indexOf(10)) >= 0) {
      yield acc.subarray(0, i).toString('utf8').replace(/\r+$/, '');
      acc = acc.subarray(i + 1);
    }
  }
}

/** Offline Auto's working size by the free video memory. A working size below the source is folded into the decode,
 * so the host is asked before the render plans (`--offline --fit-work`: it prices Auto's pick and every DLSS mode below
 * it, the model, the passes around it and the output's buffers, against NVAPI's free figure less a reserve, and names
 * the first that fits). A partial render keeps the mode its resume signature was made with: the signature carries
 * work_fit only when the fit lowered the mode, so every other render keeps its signature. A mode other than Auto, or
 * DLSS 4.5 (hostArgs null), gets the plan as asked; a string = why --scale is refused. */
export function autoFit(o: {
  exe: string;
  st: Stream;
  w: number;
  h: number;
  upscaleF: number;
  scale: string | null;
  hostArgs: string[] | null; // the model and the passes the host prices
  ns: ArgNs;
  resumeJson: string | null; // null = no resume
  env: NodeJS.ProcessEnv;
  say: Say;
}): { plan: WorkPlan; sigNs: ArgNs } | string {
  const base = workPlan(o.st, o.w, o.h, o.upscaleF, o.scale);
  if (typeof base === 'string') return base;
  const cands = autoCandidates(o.st, o.w, o.h, o.upscaleF);
  if (
    String(o.scale ?? '')
      .trim()
      .toLowerCase() !== 'auto' ||
    !o.hostArgs ||
    cands.length < 2
  )
    return { plan: base, sigNs: o.ns };
  const lowered = (i: number, why: string) => ({
    plan: { ...cands[i].plan, note: cands[i].plan.note.replace(/^DLSS mode \w+:/, `DLSS mode Auto (${why}):`) },
    sigNs: { ...o.ns, work_fit: cands[i].mode },
  });
  if (o.resumeJson && fs.existsSync(o.resumeJson)) {
    let sig: unknown = null;
    try {
      sig = JSON.parse(fs.readFileSync(o.resumeJson, 'utf8'))?.sig;
    } catch {
      sig = null;
    }
    if (sig === resumeSig(o.ns, o.env)) return { plan: base, sigNs: o.ns };
    for (let i = 1; i < cands.length; i++)
      if (sig === resumeSig({ ...o.ns, work_fit: cands[i].mode }, o.env))
        return lowered(i, `${cands[i].mode}, as the partial render it resumes`);
  }
  const list = cands.map((c) => `${c.plan.workW}:${c.plan.workH}:${c.plan.w}:${c.plan.h}`).join(',');
  const out = ['--out-w', String(base.outW), '--out-h', String(base.outH)];
  const r = spawnSync(o.exe, ['--offline', ...o.hostArgs, ...out, '--fit-work', list], {
    encoding: 'utf8',
    windowsHide: true,
    timeout: 20000,
    env: o.env,
  });
  const m = /OFFLINE FIT (\d+) (\d+) (\d+) (\d+)/.exec(r.stdout || '');
  if (!m) {
    o.say('offline Auto: the video memory fit did not answer, Auto keeps its own pick\n');
    return { plan: base, sigNs: o.ns };
  }
  const i = Math.min(cands.length - 1, parseInt(m[1], 10));
  if (i === 0) return { plan: base, sigNs: o.ns };
  const room = parseInt(m[2], 10),
    needI = parseInt(m[4], 10);
  return lowered(
    i,
    `${cands[i].mode}; the free video memory fits it: ${cands[0].mode} needs about ${m[3]} MiB, ${room} MiB are free` +
      (needI > room ? ', it does not fit at any mode: the render runs anyway' : ''),
  );
}

/** One render: resolves to the exit code; throws RenderExit for sys.exit. `say` writes one stderr
 * text. */
async function nativeRoute(argv: string[], say: Say, env: NodeJS.ProcessEnv): Promise<number> {
  const ENGINE = engineDir();
  const FFMPEG = tool(ENGINE, 'ffmpeg'),
    FFPROBE = tool(ENGINE, 'ffprobe');
  const ns = parseRenderArgv(argv);
  const args = ns as unknown as RenderArgs & {
    input: string;
    output: string | null;
    fps: number | null;
    codec: string;
    multi: number;
  };
  const inp = path.win32.resolve(args.input);
  const SHARPEN = clamp(args.sharpen, 0.0, 1.0);
  const NO_INTERP = args.no_interp,
    FRUC_MODE = args.fruc,
    DLSSG_MODE = args.dlssg;
  const RIFE_MODE = args.rife || args.rife_drba,
    DRBA_MODE = args.rife_drba,
    LSFG_MODE = args.lsfg;
  const NVOF_MODE = args.nvof;
  if ([FRUC_MODE, DLSSG_MODE, RIFE_MODE, LSFG_MODE, NVOF_MODE].filter(Boolean).length > 1) {
    throw new RenderExit(
      '--fruc, --dlssg, --rife/--rife-drba, --lsfg and --nvof are ' +
        'mutually exclusive interpolation backends; pick one',
    );
  }
  let UPSCALE_F = args.upscale <= 0 ? 1.0 : clamp(args.upscale, 1.0 / 16, 16.0);
  let UPSCALE = UPSCALE_F !== 1.0;
  const RTX_VSR = args.rtx_vsr;
  let RTX_HDR = args.rtx_hdr;
  const DV_EXPORT = args.dv,
    HP_EXPORT = args.hdr10plus,
    CODEC = args.codec;
  const HDR_SAT = clamp(args.hdr_saturation, 0, 200),
    HDR_CON = clamp(args.hdr_contrast, 0, 200);
  const HDR_COLOR = args.hdr_color;
  const HDR_VIBRANCE = clamp(args.hdr_vibrance, 0.0, 1.0),
    HDR_SATBOOST = clamp(args.hdr_satboost, 0.0, 1.0);

  let { w: W, h: H, num, den, nb: NB, st: ST } = probe(FFPROBE, inp);
  const vfr = vfrConform(ST, num, den);
  [num, den] = [vfr.num, vfr.den];
  const VFR_DEC = vfr.flags;
  if (vfr.note) say(vfr.note);
  powerNotice(say);

  const SRC_CODEC = String(ST.codec_name || '').toLowerCase();
  const SRC_PIX = String(ST.pix_fmt || 'yuv420p');
  const SRC_BITS = sourceBits(ST, SRC_PIX);
  const TEN_BIT = SRC_BITS >= 10;
  const CHROMA444 = fullChroma(ST);
  const FPS_MODE = args.fps !== null && args.fps > 0;
  if (DLSSG_MODE && !NO_INTERP && (FPS_MODE || !(args.multi >= 2 && args.multi <= 6))) {
    throw new RenderExit(
      'DLSS Frame Generation supports whole multipliers from 2x to 6x on the source grid ' +
        '(no --fps resampling). Pick 2x-6x, or use GMFSS for anything else.',
    );
  }
  if (!NO_INTERP && !FPS_MODE && args.multi < 2) {
    throw new RenderExit(
      '--multi 1 generates no frames: use --no-interp for the passes alone, ' + '--multi 2 or more, or --fps',
    );
  }
  NB = frameCount(FFPROBE, inp, NB, num / den);
  const rate = outputRate(args.multi, FPS_MODE ? args.fps : null, NO_INTERP, num, den);
  const rateStr = rate.rateStr,
    outLabel = rate.outLabel;
  let ratio = rate.ratio;
  const tr = probeTracks(FFPROBE, inp);
  const outPath = args.output
    ? path.win32.resolve(args.output)
    : inp.slice(0, inp.length - path.win32.extname(inp).length) +
      (NO_INTERP ? '_sharpened' : `_${outLabel}fps`) +
      (needMkv(tr.aud, tr.sub) ? '.mkv' : '.mp4');
  const OUT_IS_MKV = outPath.toLowerCase().endsWith('.mkv');
  const oe = path.win32.extname(outPath);
  const ob = outPath.slice(0, outPath.length - oe.length);
  const WORK_PATH = ob + '.part' + oe;

  const NATIVE_EXE = path.join(ENGINE, 'live', 'smv-live.exe');
  if (NVOF_MODE) {
    const why = nvofRefusal(args, FPS_MODE);
    if (why) throw new RenderExit('NVIDIA Optical Flow: ' + why);
  }
  if (!isFile(NATIVE_EXE)) throw new RenderExit(`the render host ${NATIVE_EXE} is missing; reinstall SmoothMyVideo`);
  let fragCopy = CODEC === 'vvc';
  const P = partsOf(WORK_PATH, ob, OUT_IS_MKV);
  const RESUMABLE = !env.SMV_NO_RESUME;
  const DEC_FMT = TEN_BIT ? 'rgb48le' : 'rgb24';
  const OUT_RAW_FMT = 'rgb48le';
  if (DRBA_MODE && !FPS_MODE) ratio = args.multi;
  const totalPairs = NB ? Math.max(1, NB - 1) : 0;
  const totalUnits = NO_INTERP ? NB : totalPairs;
  // (the host only enlarges: workPlan's decode is never larger than the working size or the output)
  const kind = NVOF_MODE
    ? 'nvof'
    : NO_INTERP
      ? 'echo'
      : isGmfss(args)
        ? 'gmfss'
        : DRBA_MODE
          ? 'drba'
          : FRUC_MODE
            ? 'fruc'
            : DLSSG_MODE
              ? 'dlssg'
              : 'rife';
  // W x H = the decode, WORK = the DLSS mode x the output (DLSS 5 and the model); on Auto the working size fits the
  // free video memory (autoFit), a resumed render keeps the mode it was made with
  const fitArgs =
    kind === 'dlssg'
      ? null
      : [
          '--multi',
          String(Math.trunc(args.multi)),
          ...(kind === 'rife' ? [] : [kind === 'echo' ? '--no-interp' : '--' + kind]),
          ...(args.dlssnr
            ? ['--dlssnr', '--nr-passes', String(Math.min(10, Math.max(1, Math.round(args.nr_passes ?? 1))))]
            : []),
          ...(RTX_HDR ? ['--rtx-hdr'] : []),
          ...(RTX_VSR ? ['--rtx-vsr'] : []),
          ...(args.restore ? ['--restore'] : []),
          ...(args.no_gpu_fit ? ['--no-gpu-fit'] : []),
        ];
  const fit = autoFit({
    exe: NATIVE_EXE,
    st: ST,
    w: W,
    h: H,
    upscaleF: UPSCALE_F,
    scale: args.work_scale,
    hostArgs: fitArgs,
    ns,
    resumeJson: RESUMABLE ? P.resumeJson : null,
    env,
    say,
  });
  if (typeof fit === 'string') throw new RenderExit(fit);
  const plan = fit.plan;
  [W, H] = [plan.w, plan.h];
  const [WORK_W, WORK_H] = [plan.workW, plan.workH];
  UPSCALE = plan.outW !== W || plan.outH !== H;
  UPSCALE_F = plan.outH / H;
  const IMG_SCALE_VF = plan.vf;
  say(plan.note);
  const sig = resumeSig(fit.sigNs, env);
  const [OUT_W, OUT_H] = [plan.outW, plan.outH];
  if (kind === 'fruc') {
    const frdir = env.SMV_NVOFFRUC_DIR || path.join(ENGINE, 'nvoffruc');
    const miss = ['nvoffruc_bridge.dll', 'NvOFFRUC.dll', 'cudart64_110.dll'].filter(
      (d) => !isFile(path.join(frdir, d)),
    );
    if (miss.length) {
      throw new RenderExit(
        `NVIDIA Smooth Motion (NvOFFRUC) unavailable: ${miss[0]} not found in ${frdir}. ` +
          'It needs a Turing..Blackwell GPU with the Optical Flow engine, and NvOFFRUC.dll ' +
          "installed in engine/nvoffruc (from NVIDIA's Optical Flow SDK .zip); or drop --fruc " +
          'to use GMFSS.',
      );
    }
  } else if (kind === 'dlssg') {
    const dgdir = env.SMV_DLSSG_DIR || path.join(ENGINE, 'dlssg');
    const miss = [
      'dlssg2f.exe',
      'sl.interposer.dll',
      'sl.common.dll',
      'sl.dlss_g.dll',
      'sl.pcl.dll',
      'sl.reflex.dll',
      'nvngx_dlssg.dll',
    ].filter((d) => !isFile(path.join(dgdir, d)));
    if (miss.length)
      throw new RenderExit(`DLSS Frame Generation unavailable: DLSS host not found: ${miss[0]} is missing in ${dgdir}`);
  }
  // the model lines render.py writes on this route
  if (NO_INTERP)
    say('no-interp mode: GMFSS interpolation disabled (re-encode at source fps with optional FSR sharpen)\n');
  else if (FRUC_MODE) say('Using the NVIDIA Smooth Motion backend for interpolation (NVIDIA Optical Flow)\n');
  else if (DLSSG_MODE) say('Using the DLSS Frame Generation backend for interpolation (DLSS 4.5)\n');
  else if (NVOF_MODE) say('Using the NVIDIA Optical Flow backend for interpolation (native host)\n');
  else if (RIFE_MODE) say('Using the RIFE backend for interpolation (4.26 heavy, native host)\n');
  else if (LSFG_MODE) say('Using the Frame Blend backend for interpolation (RIFE 4.26 heavy flow, native host)\n');
  if (kind === 'gmfss') say('Using the GMFSS backend for interpolation (native host)\n');

  // RTX HDR / DV / HDR10+ (render.py's upscale / HDR backend block, native half)
  const SRC_HDR_IN = ['smpte2084', 'arib-std-b67'].includes(String(tag(ST, 'color_transfer') || ''));
  if (RTX_HDR && (OUT_W > 8192 || OUT_H > 8192)) {
    say(
      `[rtx] RTX HDR is limited to 8192px outputs (${OUT_W}x${OUT_H} requested): ` +
        'TrueHDR at this size oversubscribes GPU memory, which can hard-crash the ' +
        'system (DPC watchdog). Rendering SDR; lower the upscale target to combine ' +
        'it with HDR.\n',
    );
    RTX_HDR = false;
  }
  if (RTX_HDR && SRC_HDR_IN) {
    say(
      `[rtx] source is already HDR (transfer ${tag(ST, 'color_transfer')}); TrueHDR ` +
        'converts SDR only, skipping the HDR conversion (source HDR signalling is ' +
        'carried through as-is)\n',
    );
    RTX_HDR = false;
  }
  const DOVI_EXE = path.join(ENGINE, 'dvtools', 'dovi_tool.exe');
  const HP_EXE = path.join(ENGINE, 'hptools', 'hdr10plus_tool.exe');
  const wantDv = DV_EXPORT && CODEC === 'hevc' && !OUT_IS_MKV && isFile(DOVI_EXE);
  const wantHp = HP_EXPORT && CODEC === 'hevc' && !OUT_IS_MKV && isFile(HP_EXE);
  let HDR_ACTIVE = false;
  if (RTX_HDR) {
    const rtxDir = env.SMV_RTXVIDEO_DIR || path.join(ENGINE, 'rtxvideo');
    const miss = ['rtxvideo_cuda.dll', 'nvngx_truehdr.dll'].filter((f) => !isFile(path.join(rtxDir, f)));
    if (miss.length)
      say(`[rtx] unavailable, falling back (Lanczos3 upscale / SDR): ${miss.join(', ')} not in ${rtxDir}\n`);
    else {
      HDR_ACTIVE = true;
      const vib = HDR_VIBRANCE > 0 ? `, vib ${pyG(HDR_VIBRANCE)}` : '';
      const sb = HDR_SATBOOST > 0 ? `, sb ${pyG(HDR_SATBOOST)}` : '';
      say(
        `RTX HDR ready (TrueHDR ${HDR_NITS} nits, contrast ${HDR_CON - 100}, saturation ${HDR_SAT - 100}, ` +
          `colour ${HDR_COLOR}${vib}${sb}) HDR10 (BT.2020 PQ) @ ${OUT_W}x${OUT_H}\n`,
      );
    }
  }
  const DV_ACTIVE = wantDv && HDR_ACTIVE;
  if (DV_EXPORT && !DV_ACTIVE) {
    const why = OUT_IS_MKV
      ? 'output is MKV (DV export needs MP4)'
      : !isFile(DOVI_EXE)
        ? 'dovi_tool not found in engine/dvtools'
        : 'RTX HDR is not active';
    say(`[dv] Dolby Vision export skipped: ${why}\n`);
  } else if (DV_ACTIVE) say('Dolby Vision Profile 8.1 export ON (plays as HDR10 where DV is unsupported)\n');
  const HP_ACTIVE = wantHp && HDR_ACTIVE;
  if (HP_EXPORT && !HP_ACTIVE) {
    const why = OUT_IS_MKV
      ? 'output is MKV (HDR10+ export needs MP4)'
      : !isFile(HP_EXE)
        ? 'hdr10plus_tool not found in engine/hptools'
        : 'RTX HDR is not active';
    say(`[hdr10+] HDR10+ export skipped: ${why}\n`);
  } else if (HP_ACTIVE) say('HDR10+ dynamic metadata ON (plays as HDR10 where HDR10+ is unsupported)\n');
  let nr: { structure: number; tone: number; style: number; passes: number } | null = null;
  if (args.dlssnr) {
    const nrDir = env.SMV_DLSSNR_DIR || path.join(ENGINE, 'dlssnr');
    const miss = ['nvngx.dll', 'nvngx_dlssnr.dll'].filter((f) => !isFile(path.join(nrDir, f)));
    if (miss.length) say(`[dlss5] unavailable, skipping: DLSS 5 files missing in ${nrDir}: ${miss.join(', ')}\n`);
    else
      nr = {
        structure: clamp(args.nr_structure, 0.0, 2.0),
        tone: clamp(args.nr_tone, 0.0, 2.0),
        style: [0, 1, 2].includes(args.nr_style) ? args.nr_style : 1,
        passes: Math.min(10, Math.max(1, Math.round(args.nr_passes ?? 1))),
      };
  }

  // resume (render.py's _try_resume block + the native HDR prefix)
  const grid = {
    noInterp: NO_INTERP,
    fpsMode: FPS_MODE,
    drbaMode: DRBA_MODE,
    multi: args.multi,
    totalPairs,
    ratio,
    nb: NB,
  };
  const rctx = {
    ...grid,
    ffmpeg: FFMPEG,
    ffprobe: FFPROBE,
    fragCopy,
    fragFlags: FRAG_FLAGS,
    concatTxt: P.concatTxt,
    parts: P,
    sig,
    resumable: RESUMABLE,
    say,
  };
  const rz = tryResume(rctx);
  fragCopy = rctx.fragCopy;
  const R = applyResume(rz, P.vidPart, NB, totalUnits, say);
  let dvhpNote = '';
  let hdrPrefix: { maxcll: number; maxfall: number; l1: number[][]; hp: HpFrame[] } | null = null;
  if (HDR_ACTIVE && RESUMABLE) {
    const hp = nativeHdrPrefix(P.hdrFrames, R.active, R.outBase, DV_ACTIVE, HP_ACTIVE, say);
    hdrPrefix = hp.prefix;
    if (hp.note) dvhpNote = hp.note;
  }
  // the decode's colour conversion: inside the downscale's chain, else its own (zimg, empty = swscale's)
  const [decFilters, pipeDiscard] = decodeFilters(
    R.skipSrc,
    VFR_DEC,
    IMG_SCALE_VF.length ? IMG_SCALE_VF : decodeRgbVf(ST, W, H),
  );

  // the encoder
  const fatal = (m: string): never => {
    throw new RenderExit(m);
  };
  const [venc, useNvenc] = chooseEncoder(
    FFMPEG,
    CODEC,
    OUT_W,
    OUT_H,
    outLabel,
    R.active ? R.venc : '',
    path.win32.basename(P.vidPart),
    say,
    fatal,
  );
  if (venc === 'libvvenc') fragCopy = true;
  const ENC_IN_FMT = HDR_ACTIVE
    ? 'x2rgb10le'
    : NO_INTERP && !(SHARPEN > 0 || UPSCALE || nr || args.restore)
      ? DEC_FMT
      : OUT_RAW_FMT;
  const enc = encodePlan(
    {
      ffmpeg: FFMPEG,
      inp,
      venc,
      useNvenc,
      hdrActive: HDR_ACTIVE,
      chroma444: CHROMA444,
      st: ST,
      outLabel,
      outW: OUT_W,
      outH: OUT_H,
      rateStr,
      encInFmt: ENC_IN_FMT,
      aud: tr.aud,
      sub: tr.sub,
      hasAttach: tr.hasAttach,
      outIsMkv: OUT_IS_MKV,
      resumable: RESUMABLE,
      resumeActive: R.active,
      vidPart: P.vidPart,
      vidPart2: P.vidPart2,
      workPath: WORK_PATH,
      dvOrHp: DV_ACTIVE || HP_ACTIVE,
    },
    env,
  );
  for (const n of enc.notes) say(n);
  const HDR_MKV_2STAGE = HDR_ACTIVE && OUT_IS_MKV;
  const sidecar = (k: number, t: number) => writeResumeSidecar(P.resumeJson, sig, k, t, venc, 0, 0);
  if (RESUMABLE) sidecar(R.skipSrc, totalUnits);
  const sharpNote = SHARPEN > 0 ? `  sharpen(rcas)=${pyG(SHARPEN)}` : '';
  const upNote = UPSCALE ? `  upscale=${pyG(UPSCALE_F)}x->${OUT_W}x${OUT_H}` : '';
  const hdrNote = HDR_ACTIVE ? '  HDR10(TrueHDR,BT.2020 PQ)' : '';
  const nrNote = nr ? `  dlss5(structure ${pyG(nr.structure)}, tone ${pyG(nr.tone)})` : '';
  say(
    `encode: ${venc} visually-lossless -> ${enc.outPix}  (source ${SRC_CODEC || '?'} ${SRC_BITS}bit ${SRC_PIX})` +
      `${nrNote}${sharpNote}${upNote}${hdrNote}\n`,
  );

  // the host's flags
  const cacheDir = env.SMV_TRT_CACHE || path.join(ENGINE, 'trt_cache_safe_to_delete');
  fs.mkdirSync(cacheDir, { recursive: true });
  const nargs = [
    '--w',
    String(W),
    '--h',
    String(H),
    '--multi',
    String(Math.trunc(args.multi)),
    '--frames',
    String(NB || 0),
    '--pixfmt',
    DEC_FMT,
    '--out-pixfmt',
    ENC_IN_FMT,
    '--script',
    path.join(ENGINE, 'live_server.py'),
    '--cache',
    cacheDir,
  ];
  if (kind === 'nvof') nargs.push('--nvof');
  else if (kind === 'echo') nargs.push('--no-interp');
  if (kind === 'gmfss') nargs.push('--gmfss');
  if (kind === 'drba') nargs.push('--drba');
  if (kind === 'fruc') nargs.push('--fruc');
  if (kind === 'dlssg') nargs.push('--dlssg');
  if (FPS_MODE && !NO_INTERP) nargs.push('--fps-ratio', pyFloatRepr(ratio as number));
  if (OUT_W !== W || OUT_H !== H) {
    nargs.push('--out-w', String(OUT_W), '--out-h', String(OUT_H));
    if (RTX_VSR) nargs.push('--rtx-vsr');
  }
  if (SHARPEN > 0) nargs.push('--sharpen', pyG(SHARPEN));
  if (args.restore) nargs.push('--restore');
  if (args.no_gpu_fit) nargs.push('--no-gpu-fit');
  // the working size: the host runs Restore and the resize to it on the decoded frame, DLSS 5 and
  // the model at it, then the final resize to the output (sent only when it changes something, so
  // an older host stays usable for a same-size render)
  if (WORK_W !== W || WORK_H !== H || args.restore) nargs.push('--work-w', String(WORK_W), '--work-h', String(WORK_H));
  if (nr)
    nargs.push(
      '--dlssnr',
      '--nr-structure',
      pyG(nr.structure),
      '--nr-tone',
      pyG(nr.tone),
      '--nr-style',
      String(nr.style),
    );
  if (nr && nr.passes > 1) nargs.push('--nr-passes', String(nr.passes));
  // HDR video (PQ or HLG): DLSS 5 sees its SDR range, as Live does on an HDR desktop, and RIFE finds
  // the motion on the same SDR view of the decoded picture
  if (SRC_HDR_IN) nargs.push('--src-hdr', String(tag(ST, 'color_transfer') || '') === 'arib-std-b67' ? 'hlg' : 'pq');
  const hdrStats = WORK_PATH + '.hdrstats.json';
  if (HDR_ACTIVE) {
    nargs.push(
      '--rtx-hdr',
      '--hdr-color',
      HDR_COLOR,
      '--hdr-contrast',
      String(HDR_CON),
      '--hdr-saturation',
      String(HDR_SAT),
      '--hdr-vibrance',
      pyG(HDR_VIBRANCE),
      '--hdr-satboost',
      pyG(HDR_SATBOOST),
      '--hdr-stats',
      hdrStats,
    );
    if (DV_ACTIVE) nargs.push('--hdr-dv');
    if (HP_ACTIVE) nargs.push('--hdr-hp');
    try {
      fs.unlinkSync(hdrStats);
    } catch {
      /* except OSError: pass */
    }
  }
  const PAUSE_FILE = env.SMV_PAUSE_FILE;
  if (PAUSE_FILE) nargs.push('--pause-file', PAUSE_FILE);
  // the GUI's progress thumbnail: the host dumps a small frame about once a second
  // and says THUMB, onLine turns it into the PNG the renderer polls; no env var = no cost (CLI)
  const LIVE = env.SMV_LIVE_PREVIEW || '',
    THUMB_RAW = LIVE + '.raw';
  if (LIVE) {
    nargs.push('--thumb', THUMB_RAW);
    if (env.SMV_LIVE_OFF_FILE) nargs.push('--thumb-off', env.SMV_LIVE_OFF_FILE);
  }
  if (HDR_ACTIVE && RESUMABLE) nargs.push('--hdr-frames', P.hdrFrames);
  if (R.active) {
    const drop = NO_INTERP ? 0 : FPS_MODE || DRBA_MODE ? R.pairSkip : 1 + R.pairSkip;
    nargs.push('--resume-pair', String(R.skipSrc), '--resume-out', String(R.outBase), '--resume-drop', String(drop));
    if (pipeDiscard) nargs.push('--skip-in', String(pipeDiscard));
  }

  const progress = new Progress({
    encTarget: enc.target,
    baseBytes: R.baseBytes,
    outPath,
    hdrMkv2stage: HDR_MKV_2STAGE,
    resumable: RESUMABLE,
    sidecar,
    say,
  });
  const preview = () =>
    emitPausePreview({
      ffmpeg: FFMPEG,
      ffprobe: FFPROBE,
      fragCopy,
      fragFlags: FRAG_FLAGS,
      concatTxt: P.concatTxt,
      resumable: RESUMABLE,
      stage2Maps: enc.stage2Maps,
      outIsMkv: OUT_IS_MKV,
      ob,
      encTarget: enc.target,
      resumeActive: R.active,
      parts: P,
      inp,
      say,
    });
  let nk = 0,
    nout = 0;
  const onLine = (ln: string) => {
    if (ln.startsWith('PROGRESS ')) {
      const m = /^\S+\s+([^/\s]+)/.exec(ln);
      const k = m && /^\s*[+-]?\d+\s*$/.test(m[1]) ? parseInt(m[1], 10) : NaN;
      if (Number.isNaN(k)) say(ln + '\n');
      else {
        nk = k;
        progress.step(nk, totalUnits);
      }
    } else if (ln.startsWith('PAUSED') || ln.startsWith('RESUMED')) {
      say(ln + '\n');
      if (ln.startsWith('PAUSED')) preview();
    } else if (ln === 'THUMB') {
      if (LIVE) thumbPng(THUMB_RAW, LIVE, SRC_HDR_IN);
    } else if (ln.startsWith('OUTFRAMES ')) {
      const v = ln.split(/\s+/)[1];
      if (v !== undefined && /^\s*[+-]?\d+\s*$/.test(v)) nout = parseInt(v, 10);
    } else if (ln) say('[native] ' + ln + '\n');
  };
  const decCmd = decodeCmd(FFMPEG, inp, VFR_DEC, decFilters, DEC_FMT);
  const spawnOpts = { cwd: ENGINE, windowsHide: true };

  // the resident host (render.py _native_resident_render), else one exe for this render
  let nrc: number | null = null;
  // the encoder's exit code: a host can finish writing every frame before a dying encoder closes
  // its pipe (NVENC refusing a tiny frame), so the host's code alone cannot tell a good render
  let encRc = 0;
  let encRes: ChildProcess | null = null;
  if ((env.SMV_OFFLINE_RESIDENT || '') !== '0')
    nrc = await residentRender(
      NATIVE_EXE,
      ENGINE,
      nargs,
      onLine,
      say,
      (fin, fout) => {
        const decp = spawn(decCmd[0], decCmd.slice(1), { ...spawnOpts, stdio: ['inherit', fin, 'inherit'] });
        encRes = spawn(enc.cmd[0], enc.cmd.slice(1), { ...spawnOpts, stdio: [fout, 'inherit', 'inherit'] });
        return [decp, encRes];
      },
      env,
    );
  // residentRender waited for both children, so the exit code is in
  if (encRes !== null) encRc = (encRes as ChildProcess).exitCode ?? 1;
  if (nrc === null) {
    const dec = spawn(decCmd[0], decCmd.slice(1), { ...spawnOpts, stdio: ['inherit', 'pipe', 'inherit'] });
    const exe = spawn(NATIVE_EXE, ['--offline', ...nargs], { ...spawnOpts, stdio: [dec.stdout!, 'pipe', 'pipe'] });
    const encp = spawn(enc.cmd[0], enc.cmd.slice(1), { ...spawnOpts, stdio: [exe.stdout!, 'inherit', 'inherit'] });
    dec.stdout!.destroy(); // the exe and the encoder hold their own ends
    exe.stdout!.destroy();
    const rl = readline.createInterface({ input: exe.stderr!, crlfDelay: Infinity });
    rl.on('line', (l) => onLine(l.replace(/\r+$/, '')));
    const closed = new Promise<void>((res) => rl.once('close', () => res()));
    nrc = await waitExit(exe);
    await closed;
    [encRc] = await Promise.all([waitExit(encp), waitExit(dec)]);
  }
  const encFailed = `the encoder (${venc}) failed (exit ${encRc}); its error is in the lines above`;
  if (LIVE) {
    try {
      fs.unlinkSync(THUMB_RAW);
    } catch {
      /* none written */
    }
  }
  if (nrc !== 0) {
    if (DLSSG_MODE && nrc === 3) {
      throw new RenderExit(
        `this GPU does not support ${args.multi}x DLSS multi-frame generation (RTX 50 series ` +
          'does up to 6x, RTX 40 series only 2x); pick a lower multiplier or GMFSS',
      );
    }
    if (DLSSG_MODE && nrc === 4) {
      if (RESUMABLE) writeResumeSidecar(P.resumeJson, sig, nk, totalPairs, venc, 0, 0);
      const pct = totalPairs ? pyRound((100 * nk) / totalPairs) : 0;
      say(
        `DLSS_PREEMPTED ${pct}\n` +
          'DLSS Frame Generation was preempted (a video with NVIDIA RTX Video enhancement was most ' +
          'likely playing) and could not recover after 4 restarts. Your ' +
          `progress (~${pct}%) is saved and the render can be resumed: close that video, then click ` +
          'Resume to continue where it left off. (Or switch to GMFSS, which RTX Video does not ' +
          'affect.)\n',
      );
      throw new RenderExit(1);
    }
    if (encRc !== 0) throw new RenderExit(encFailed); // the host's write failed because the encoder died
    throw new RenderExit(`native offline host failed (exit ${nrc}); the [native] lines above name the reason`);
  }
  if (encRc !== 0) throw new RenderExit(encFailed);
  let rtx: HdrStats | null = null;
  if (HDR_ACTIVE) {
    let hs: { maxcll: number; maxfall: number; l1?: number[][]; hp?: HpFrame[] };
    try {
      hs = JSON.parse(fs.readFileSync(hdrStats, 'utf8'));
      fs.unlinkSync(hdrStats);
    } catch (e) {
      throw new RenderExit(`native offline host: the HDR statistics are missing (${pyErrText(e)})`);
    }
    rtx = { maxcll: Math.trunc(hs.maxcll), maxfall: Math.trunc(hs.maxfall), l1: hs.l1 || [], hp: hs.hp || [] };
    if (hdrPrefix) {
      rtx.maxcll = Math.max(rtx.maxcll as number, hdrPrefix.maxcll);
      rtx.maxfall = Math.max(rtx.maxfall as number, hdrPrefix.maxfall);
      rtx.l1 = hdrPrefix.l1.concat(rtx.l1 as number[][]);
      rtx.hp = hdrPrefix.hp.concat(rtx.hp as HpFrame[]);
    }
  }

  // _finish
  finalizeOutput({
    ffmpeg: FFMPEG,
    inp,
    workPath: WORK_PATH,
    encTarget: enc.target,
    resumable: RESUMABLE,
    resumeActive: R.active,
    vidPart: P.vidPart,
    vidPart2: P.vidPart2,
    vidFull: P.vidFull,
    hdrActive: HDR_ACTIVE,
    outIsMkv: OUT_IS_MKV,
    hpActive: HP_ACTIVE,
    dvActive: DV_ACTIVE,
    resumeDvhpNote: dvhpNote,
    fragCopy,
    fragFlags: FRAG_FLAGS,
    stage2Maps: enc.stage2Maps,
    rtx,
    nits: HDR_NITS,
    masterPrim: HDR_MASTER_PRIM,
    hpExport: () => hpExport(WORK_PATH, FFMPEG, HP_EXE, rtx, HDR_NITS, HDR_MASTER_PRIM, rateStr, outLabel, say),
    dvExport: () =>
      dvExport(WORK_PATH, FFMPEG, DOVI_EXE, rtx, HDR_NITS, HDR_MASTER_PRIM, rateStr, outLabel, OUT_W, OUT_H, say),
    concatCopy: (parts, dst) =>
      concatCopy(
        { ffmpeg: FFMPEG, ffprobe: FFPROBE, fragCopy, fragFlags: FRAG_FLAGS, concatTxt: P.concatTxt },
        parts,
        dst,
      ),
    resumeCleanup: () => resumeCleanup(P),
    say,
  });
  if (fs.existsSync(WORK_PATH)) fs.renameSync(WORK_PATH, outPath);
  const outFrames = nout ? nout : null;
  if (outFrames !== null) say(`OUTFRAMES ${outFrames}\n`);
  if (NO_INTERP) {
    say(`done ${nout} frames (${SHARPEN > 0 ? 'RCAS-sharpened' : 're-encoded'}, native host) -> ${outPath}\n`);
  } else {
    const npairs = nout && !FPS_MODE ? Math.floor((nout - 1) / Math.trunc(args.multi)) : Math.max(0, (NB || 0) - 1);
    say(`done ${npairs} pairs (native host) -> ${outPath}\n`);
  }
  return 0;
}

/** render.py _offline_host_connect + _native_resident_render: the item over the control pipe, the
 * decoder and the encoder on the host's frame pipes. null = no host served it (nothing spawned). */
async function residentRender(
  exe: string,
  engine: string,
  itemArgs: string[],
  onLine: (l: string) => void,
  say: Say,
  spawnPair: (fin: number, fout: number) => ChildProcess[],
  env: NodeJS.ProcessEnv,
): Promise<number | null> {
  const name = offlinePipeName(exe);
  const pipe = '\\\\.\\pipe\\' + name;
  let fd = openPipe(pipe, 'r+');
  let host: ChildProcess | null = null;
  let hostExit: number | null = null;
  if (fd === null) {
    const log = env.SMV_OFFLINE_HOST_LOG;
    let err: number | 'ignore' = 'ignore';
    try {
      if (log) err = fs.openSync(log, 'a');
      host = spawn(exe, ['--offline', '--resident', '--pipe', name, '--script', path.join(engine, 'live_server.py')], {
        detached: true,
        stdio: ['ignore', 'ignore', err],
        windowsHide: true,
      });
      host.on('error', () => {
        hostExit = -1;
      });
      host.on('exit', (c) => {
        hostExit = c ?? 1;
      });
      host.unref();
    } catch (e) {
      say(`[native] offline host spawn failed: ${(e as Error).message}\n`);
      return null;
    } finally {
      if (typeof err === 'number') fs.closeSync(err);
    }
  }
  const deadline = Date.now() + (host ? 10000 : 2000);
  for (;;) {
    if (fd !== null && fd !== -1) break;
    if (host) {
      await new Promise((r) => setImmediate(r)); // let the child's exit event land
      if (hostExit !== null) {
        say(`[native] offline host exited at start (exit ${hostExit})\n`);
        return null;
      }
    }
    if (Date.now() > deadline) return null;
    sleepMs(50);
    fd = openPipe(pipe, 'r+');
  }
  const cfd = fd as number;
  try {
    fs.writeSync(cfd, 'start\t' + itemArgs.join('\t') + '\n');
    const it = fdLines(cfd);
    let ready = false;
    for (let r = it.next(); !r.done; r = it.next()) {
      const ln = r.value;
      if (ln === 'offline pipes ready') {
        ready = true;
        break;
      }
      if (ln.startsWith('offline done:')) break;
      onLine(ln);
    }
    if (!ready) return null;
    const fin = fs.openSync(pipe + '-in', 'w');
    let fout: number;
    try {
      fout = fs.openSync(pipe + '-out', 'r');
    } catch (e) {
      fs.closeSync(fin);
      throw e;
    }
    let kids: ChildProcess[];
    try {
      kids = spawnPair(fin, fout);
    } finally {
      fs.closeSync(fin);
      fs.closeSync(fout);
    }
    fs.writeSync(cfd, 'go\n');
    say(`[native] offline host: resident host on ${pipe}\n`);
    let rc = 1;
    for (let r = it.next(); !r.done; r = it.next()) {
      const ln = r.value;
      if (ln.startsWith('offline done: exit ')) {
        const v = ln.slice(ln.lastIndexOf(' ') + 1);
        rc = /^\s*[+-]?\d+\s*$/.test(v) ? parseInt(v, 10) : 1;
        break;
      }
      onLine(ln);
    }
    await Promise.all(kids.map(waitExit));
    return rc;
  } finally {
    fs.closeSync(cfd);
  }
}

/** render.py _power_notice: one line when the GPU runs under a reduced power limit. */
function powerNotice(say: Say): void {
  try {
    const r = spawnSync('nvidia-smi', ['-q', '-d', 'POWER'], { windowsHide: true, timeout: 5000, encoding: 'utf8' });
    if (r.error || r.status !== 0) return;
    const vals: Record<string, number> = {};
    for (const line of r.stdout.split(/\r\n|\r|\n/)) {
      const i = line.indexOf(':');
      if (i < 0) continue;
      const k = line.slice(0, i).trim(),
        v = line
          .slice(i + 1)
          .trim()
          .split(' ')[0];
      if (['Current Power Limit', 'Default Power Limit', 'Max Power Limit'].includes(k) && !(k in vals)) {
        try {
          vals[k] = pyFloatParse(v);
        } catch {
          /* except ValueError: pass */
        }
      }
    }
    const cur = vals['Current Power Limit'],
      dflt = vals['Default Power Limit'],
      mx = vals['Max Power Limit'];
    if (cur && dflt && cur < 0.9 * dflt) {
      say(
        `note: GPU power limit is ${pyFixed(cur, 0)} W (board default ${pyFixed(dflt, 0)} W` +
          (mx ? `, max ${pyFixed(mx, 0)} W` : '') +
          ') - a quiet/Silent power profile is ' +
          'active, so this render runs proportionally slower; switch the Windows/vendor ' +
          'power mode for full speed\n',
      );
    }
  } catch {
    /* a missing / odd nvidia-smi never breaks a render */
  }
}

export const SESSION_LOG = path.join(os.tmpdir(), 'smv-engine.log');

/** render.py's crash policy: the full stack into the session log, one short line for the GUI. */
function crashLine(e: unknown, argv: string[]): string {
  const x = e as Error;
  try {
    fs.appendFileSync(SESSION_LOG, `[${stamp()}] ${pyArgv(argv)}\n${x && x.stack ? x.stack : String(e)}\n\n`, 'utf8');
  } catch {
    /* except OSError: pass */
  }
  if (process.env.SMV_TRACE === '1' && x && x.stack) return x.stack + '\n';
  return `ERROR: ${x && x.name ? x.name : 'Error'}: ${Array.from(String(x && x.message !== undefined ? x.message : e))
    .slice(0, 500)
    .join('')}\n(full traceback: ${SESSION_LOG})\n`;
}

export function stamp(): string {
  const d = new Date(),
    p = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

/** sys.argv as python prints it in the session log header. */
export function pyArgv(argv: string[]): string {
  return '[' + [path.join(__dirname, 'cli.js'), ...argv].map(pyReprStr).join(', ') + ']';
}

/** One render, render.py's contract. Resolves to the exit code; every line goes through say. */
export async function renderRoute(argv: string[], say: Say, env: NodeJS.ProcessEnv = process.env): Promise<number> {
  try {
    parseRenderArgv(argv);
  } catch (e) {
    say(`render: error: ${(e as Error).message}\n`); // argparse's exit code
    return 2;
  }
  try {
    return await nativeRoute(argv, say, env);
  } catch (e) {
    if (e instanceof RenderExit) {
      if (typeof e.code === 'string') {
        say(e.code + '\n');
        return 1;
      }
      return e.code;
    }
    say(crashLine(e, argv));
    return 1;
  }
}
