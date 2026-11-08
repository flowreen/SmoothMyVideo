// The offline render: ported from render.py,
// every stderr line as render.py wrote it. This process probes, plans, picks the encoder, resumes,
// spawns the decoder / encoder and the host (smv-live.exe --offline, one-shot or the resident host
// behind its named pipe), relays the host's lines and finishes the container; no frame passes
// through it (only the progress thumbnail's small dump, thumbPng).
import { ChildProcess, execFileSync, spawn } from 'child_process';
import { createHash } from 'crypto';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import * as readline from 'readline';
import { Readable } from 'stream';
import { engineCacheDir } from './cache';
import { dvExport, hpExport } from './dvhp';
import {
  chooseEncoder,
  codecEncoder,
  decodeCmd,
  decodeFilters,
  encodePlan,
  encWorksAsync,
  finalizeOutput,
  fullChroma,
  HdrStats,
  HpFrame,
  presetWorks,
  pyErrText,
  pyReprStr,
  Say,
  saveSubsBeside,
} from './encode';
import { isGmfss, nvofRefusal, RenderArgs, workPlan } from './plan';
import { thumbPng } from './preview';
import { decodeRgbVf, frameCount, needMkv, outputRate, probe, probeTracks, sourceBits, tag, vfrConform } from './probe';
import { pyFixed, pyFloatRepr, pyG, pyRound } from './pyfmt';
import {
  applyResume,
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
const HDR_MASTER_PRIM = 'display-p3';

/** TrueHDR's peak: --hdr-nits (the app passes its display's), else the primary display's as the host reads it (0 when
 * it is not in HDR mode), else 1000; inside the SDK's 400..2000, the same rule as the host's thdrPeakNits. */
function hdrPeakNits(asked: number, exe: string): number {
  let n = asked;
  if (!(n > 0)) {
    try {
      n = parseInt(execFileSync(exe, ['--peak-nits'], { cwd: path.dirname(exe), timeout: 8000 }).toString(), 10);
    } catch {
      n = 0;
    }
  }
  if (!(n > 0)) n = 1000;
  return Math.min(2000, Math.max(400, Math.round(n)));
}

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

/** DLSS 4.5's passes for a whole multiplier, the smallest pass first: an even multiple splits 2x then the rest (x4 =
 * 2 x 2, x6 = 2 x 3); above the GPU's own limit every pass stays within it, ascending (x8 = 2 x 4, x12 = 2 x 6,
 * x16 = 4 x 4, x18 = 3 x 6). A generated midpoint is the same frame whatever the pass count, and each later pass works
 * on pairs half as far apart, where DLSS 4.5's flow is better: against 1000 fps truth, 2x then 3x beats one 6x pass by
 * 0.8 dB on pairs 30 ms apart and ties at 6 ms; above the limit the pipe between passes also carries fewer frames.
 * `onePass` keeps a multiple up to the limit in one pass: RTX HDR (its statistics come from one host) and a source
 * above 100 fps (pairs under 10 ms, where one pass is as good and 0.3 dB better at 4 ms). [] when no product of
 * passes within the limit makes m (x7, x11, x13, x14 at a limit of 6). */
export function dlssgPasses(m: number, limit: number, onePass = false): number[] {
  if (m < 2) return [];
  if (m <= limit) return onePass || m % 2 || m < 4 ? [m] : [2, m / 2];
  // the fewest passes within the limit, ascending
  const split = (n: number): number[] => {
    if (n <= limit) return n >= 2 ? [n] : [];
    for (let a = Math.min(limit, n - 1); a >= 2; a--) {
      if (n % a) continue;
      const rest = split(n / a);
      if (rest.length) return [a, ...rest].sort((x, y) => x - y);
    }
    return [];
  };
  return split(m);
}

/** DLSS 4.5 at a target between whole multiples ([multiple, keep]): DLSS 4.5 makes only the evenly spaced frames between
 * two originals, so the render makes the smallest multiple its passes reach whose every keep-th frame lands within
 * 0.5 % of the target (23.976 -> 59.94 fps = 5x, every 2nd frame: still evenly spaced), keeping at least every 6th;
 * none = [the first multiple reaching the target within 3 % (Live's rule), 1]; [0, 0] = no multiple up to 64. */
export function dlssgFpsPlan(target: number, src: number, limit: number): [number, number] {
  for (let m = 2; m <= 64; m++) {
    if (!dlssgPasses(m, limit).length) continue;
    for (let k = 1; k <= 6 && k < m; k++) if (Math.abs((m * src) / k - target) <= 0.005 * target) return [m, k];
  }
  for (let m = 2; m <= 64; m++) if (dlssgPasses(m, limit).length && m * src * 1.03 >= target) return [m, 1];
  return [0, 0];
}

/** "<GPU>, <driver>" from nvidia-smi, '' when it is missing, fails or takes over 5 s. */
function gpuDriverId(): Promise<string> {
  return new Promise((resolve) => {
    try {
      const p = spawn('nvidia-smi', ['--query-gpu=name,driver_version', '--format=csv,noheader'], {
        windowsHide: true,
        stdio: ['ignore', 'pipe', 'ignore'],
      });
      const timer = setTimeout(() => p.kill(), 5000);
      let out = '';
      p.stdout!.setEncoding('utf8');
      p.stdout!.on('data', (d: string) => (out += d));
      p.once('error', () => {
        clearTimeout(timer);
        resolve('');
      });
      p.once('close', (code) => {
        clearTimeout(timer);
        resolve(code === 0 ? (out.split(/\r?\n/)[0] || '').trim() : '');
      });
    } catch {
      resolve('');
    }
  });
}

/** The card's used and total video memory in MiB from nvidia-smi (every process's, as NVAPI counts it); null when it is
 * missing, fails, takes over 5 s or lists more than one GPU. */
function gpuMemory(): Promise<{ used: number; total: number } | null> {
  return new Promise((resolve) => {
    try {
      const p = spawn('nvidia-smi', ['--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], {
        windowsHide: true,
        stdio: ['ignore', 'pipe', 'ignore'],
      });
      const timer = setTimeout(() => p.kill(), 5000);
      let out = '';
      p.stdout!.setEncoding('utf8');
      p.stdout!.on('data', (d: string) => (out += d));
      p.once('error', () => {
        clearTimeout(timer);
        resolve(null);
      });
      p.once('close', (code) => {
        clearTimeout(timer);
        const rows = out.split(/\r?\n/).filter((l) => l.trim());
        const v = rows.length === 1 ? rows[0].split(',').map((s) => parseInt(s, 10)) : [];
        resolve(code === 0 && v.length === 2 && v.every((n) => n > 0) ? { used: v[0], total: v[1] } : null);
      });
    } catch {
      resolve(null);
    }
  });
}

/** The most DLSS 4.5 makes in one pass on this GPU and driver = Streamline's numFramesToGenerateMax + 1 (the host's
 * `server max N` line): one 2-frame 256x128 render the first time, then dlssg_limit.json in the engine cache, keyed
 * by the GPU and driver; 0 = unknown. SMV_DLSSG_LIMIT=2..6 sets it (tests). */
export async function dlssgLimit(
  exe: string,
  engine: string,
  cacheDir: string,
  env: NodeJS.ProcessEnv,
): Promise<number> {
  const forced = parseInt(env.SMV_DLSSG_LIMIT || '', 10);
  if (forced >= 2 && forced <= 6) return forced;
  const id = await gpuDriverId();
  const file = path.join(cacheDir, 'dlssg_limit.json');
  try {
    const j = JSON.parse(fs.readFileSync(file, 'utf8'));
    if (id && j.id === id && j.limit >= 2) return j.limit;
  } catch {
    /* not measured on this driver yet */
  }
  fs.mkdirSync(cacheDir, { recursive: true });
  const [w, h] = [256, 128];
  const log = await new Promise<string>((resolve) => {
    let text = '';
    try {
      const p = spawn(
        exe,
        [
          '--offline',
          '--w',
          String(w),
          '--h',
          String(h),
          '--multi',
          '2',
          '--frames',
          '2',
          '--pixfmt',
          'rgb24',
          '--out-pixfmt',
          'rgb24',
          '--script',
          path.join(engine, 'live_server.py'),
          '--cache',
          cacheDir,
          '--dlssg',
        ],
        { cwd: engine, windowsHide: true, stdio: ['pipe', 'ignore', 'pipe'] },
      );
      const timer = setTimeout(() => p.kill(), 60000);
      p.stderr!.setEncoding('utf8');
      p.stderr!.on('data', (d: string) => (text += d));
      p.once('error', () => {
        clearTimeout(timer);
        resolve('');
      });
      p.once('close', () => {
        clearTimeout(timer);
        resolve(text);
      });
      p.stdin!.on('error', () => {
        /* the host ended before reading its frames */
      });
      p.stdin!.end(Buffer.alloc(w * h * 3 * 2));
    } catch {
      resolve('');
    }
  });
  const m = /server max (\d+)/.exec(log);
  const limit = m && parseInt(m[1], 10) >= 1 ? parseInt(m[1], 10) + 1 : 0;
  if (limit && id) {
    try {
      fs.writeFileSync(file, JSON.stringify({ id, limit }) + '\n');
    } catch {
      /* measured again next time */
    }
  }
  return limit;
}

/** Line reader over a blocking pipe fd (the resident host's control pipe); null = the pipe ended. `next` blocks the
 * event loop (the handshake, before any child runs); `nextAsync` reads on the thread pool, needed once children
 * joined by Node pipes run (a cascade's later passes): while the loop is blocked, Node's own copy of a pipe it handed
 * to the next process stays open, and when the loop runs again it takes part of that stream or cuts it. Never write
 * to the fd while a `nextAsync` read is pending (synchronous I/O on one pipe handle waits for the read). */
class FdLines {
  private readonly fd: number;
  private readonly buf = Buffer.alloc(4096);
  private acc = Buffer.alloc(0);
  constructor(fd: number) {
    this.fd = fd;
  }
  private line(): string | null {
    const i = this.acc.indexOf(10);
    if (i < 0) return null;
    const ln = this.acc.subarray(0, i).toString('utf8').replace(/\r+$/, '');
    this.acc = this.acc.subarray(i + 1);
    return ln;
  }
  private add(n: number): boolean {
    if (!n) return false;
    this.acc = Buffer.concat([this.acc, this.buf.subarray(0, n)]);
    return true;
  }
  next(): string | null {
    for (;;) {
      const ln = this.line();
      if (ln !== null) return ln;
      let n = 0;
      try {
        n = fs.readSync(this.fd, this.buf, 0, this.buf.length, null);
      } catch {
        /* the pipe ended */
      }
      if (!this.add(n)) return null;
    }
  }
  async nextAsync(): Promise<string | null> {
    for (;;) {
      const ln = this.line();
      if (ln !== null) return ln;
      const n = await new Promise<number>((res) =>
        fs.read(this.fd, this.buf, 0, this.buf.length, null, (e, k) => res(e ? 0 : k)),
      );
      if (!this.add(n)) return null;
    }
  }
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
  const SHARPEN = clamp(args.sharpen, 0.0, 2.0);
  const NO_INTERP = args.no_interp,
    FRUC_MODE = args.fruc,
    FSRFG_MODE = args.fsrfg,
    XEFG_MODE = args.xess_fg,
    DLSSG_MODE = args.dlssg;
  const RIFE_MODE = args.rife || args.rife_drba,
    DRBA_MODE = args.rife_drba;
  const NVOF_MODE = args.nvof;
  if ([FRUC_MODE, FSRFG_MODE, XEFG_MODE, DLSSG_MODE, RIFE_MODE, NVOF_MODE].filter(Boolean).length > 1) {
    throw new RenderExit(
      '--fruc, --fsrfg, --xess-fg, --dlssg, --rife/--rife-drba and --nvof are ' +
        'mutually exclusive interpolation backends; pick one',
    );
  }
  let UPSCALE_F = args.upscale <= 0 ? 1.0 : clamp(args.upscale, 1.0 / 16, 16.0);
  let UPSCALE = UPSCALE_F !== 1.0;
  const RTX_VSR = args.rtx_vsr;
  // RTX VSR's quality level goes to the host only when it is not the default Ultra (4)
  const VSR_Q = (args.vsr_quality ?? 4) !== 4 ? ['--vsr-quality', String(args.vsr_quality)] : [];
  const FSR_UP = args.fsr_upscale && !RTX_VSR;
  const A4K_UP = args.anime4k && !RTX_VSR && !FSR_UP;
  const ACN_UP = args.artcnn && !RTX_VSR && !FSR_UP && !A4K_UP;
  const XESS_UP = args.xess_upscale && !RTX_VSR && !FSR_UP && !A4K_UP && !ACN_UP;
  let RTX_HDR = args.rtx_hdr;
  const DV_EXPORT = args.dv,
    HP_EXPORT = args.hdr10plus,
    CODEC = args.codec;
  // Dolby Vision 8.4 = the HDR video written as HLG (its base layer); 8.1 = HDR10
  const DV_PROFILE = args.dv_profile === '8.4' ? '8.4' : '8.1';
  const HDR_SAT = clamp(args.hdr_saturation, 0, 200),
    HDR_CON = clamp(args.hdr_contrast, 0, 200);
  const HDR_COLOR = args.hdr_color;
  const HDR_VIBRANCE = clamp(args.hdr_vibrance, 0.0, 1.0),
    HDR_SATBOOST = clamp(args.hdr_satboost, 0.0, 1.0);

  // the encoder check and the power query are whole child processes (~0.2 s and ~0.09 s): both run beside the probes
  const ENC_FIRST = codecEncoder(CODEC);
  const encFirstOk = encWorksAsync(FFMPEG, ENC_FIRST);
  const power = powerQuery();

  let { w: W, h: H, num, den, nb: NB, st: ST } = probe(FFPROBE, inp);
  const vfr = vfrConform(ST, num, den);
  [num, den] = [vfr.num, vfr.den];
  const VFR_DEC = vfr.flags;
  if (vfr.note) say(vfr.note);
  powerNotice(await power, say);

  const SRC_CODEC = String(ST.codec_name || '').toLowerCase();
  const SRC_PIX = String(ST.pix_fmt || 'yuv420p');
  const SRC_BITS = sourceBits(ST, SRC_PIX);
  const TEN_BIT = SRC_BITS >= 10;
  const CHROMA444 = fullChroma(ST);
  let FPS_MODE = args.fps !== null && args.fps > 0;
  // DLSS 4.5 at a target between whole multiples (dlssgFpsPlan): the render runs as that multiple and the encode keeps
  // every DG_KEEP-th frame; from here on it is an ordinary whole-multiple render
  let DG_KEEP = 1;
  if (DLSSG_MODE && !NO_INTERP && FPS_MODE) {
    const want = args.fps as number;
    const lim =
      (await dlssgLimit(path.join(ENGINE, 'live', 'smv-live.exe'), ENGINE, engineCacheDir(ENGINE, env), env)) || 6;
    const [m, k] = dlssgFpsPlan(want, num / den, lim);
    if (!m) throw new RenderExit(`DLSS Frame Generation cannot reach ${pyG(want)} fps on this GPU; use GMFSS.`);
    const got = ((m * num) / (den * k)).toFixed(3);
    say(
      k > 1
        ? `DLSS Frame Generation at ${pyG(want)} fps: ${m}x, then every ${k === 2 ? '2nd' : k === 3 ? '3rd' : k + 'th'} ` +
            `frame = ${got} fps (evenly spaced: DLSS 4.5 makes whole multiples only)\n`
        : `DLSS Frame Generation at ${pyG(want)} fps: ${m}x = ${got} fps, the first whole multiple that reaches it ` +
            '(no multiple of the source lands on it)\n',
    );
    args.multi = m;
    args.fps = null;
    FPS_MODE = false;
    DG_KEEP = k;
  }
  if (!NO_INTERP && !FPS_MODE && args.multi < 2) {
    throw new RenderExit(
      '--multi 1 generates no frames: use --no-interp for the passes alone, ' + '--multi 2 or more, or --fps',
    );
  }
  NB = frameCount(FFPROBE, inp, NB, num / den);
  const rate = outputRate(args.multi, FPS_MODE ? args.fps : null, NO_INTERP, num, den);
  // the host's frames arrive at IN_RATE; with DG_KEEP > 1 the file holds every DG_KEEP-th of them at rateStr
  const IN_RATE = rate.rateStr;
  const rateStr = DG_KEEP > 1 ? `${num * args.multi}/${den * DG_KEEP}` : rate.rateStr,
    outLabel = DG_KEEP > 1 ? pyRound((num * args.multi) / (den * DG_KEEP)) : rate.outLabel;
  let ratio = rate.ratio;
  const tr = probeTracks(FFPROBE, inp);
  // MKV keeps its timestamps in whole milliseconds (ffmpeg's matroska muxer: a 1 ms time base, no finer scale), so above
  // 1000 fps its frames' times collide: such a render writes .mp4 and saves the subtitle tracks beside it
  const [rateNum, rateDen] = rateStr.split('/').map(Number);
  const OUT_FPS = rateNum / (rateDen || 1);
  const MKV_OK = OUT_FPS <= 1000;
  const outPath = args.output
    ? path.win32.resolve(args.output)
    : inp.slice(0, inp.length - path.win32.extname(inp).length) +
      (NO_INTERP ? '_sharpened' : `_${outLabel}fps`) +
      (needMkv(tr.aud, tr.sub) && MKV_OK ? '.mkv' : '.mp4');
  const OUT_IS_MKV = outPath.toLowerCase().endsWith('.mkv');
  if (OUT_IS_MKV && !MKV_OK)
    throw new RenderExit(
      `MKV keeps time in whole milliseconds and cannot hold ${pyG(OUT_FPS)} frames a second (over 1000); ` +
        'save the render as .mp4 (its subtitles are then saved beside it)',
    );
  const SUBS_BESIDE = !OUT_IS_MKV && !MKV_OK && tr.sub.length > 0;
  const oe = path.win32.extname(outPath);
  const ob = outPath.slice(0, outPath.length - oe.length);
  const WORK_PATH = ob + '.part' + oe;

  const NATIVE_EXE = path.join(ENGINE, 'live', 'smv-live.exe');
  if (NVOF_MODE) {
    const why = nvofRefusal(args, FPS_MODE);
    if (why) throw new RenderExit('NVIDIA Optical Flow: ' + why);
  }
  if (!isFile(NATIVE_EXE)) throw new RenderExit(`the render host ${NATIVE_EXE} is missing; reinstall SmoothMyVideo`);
  // Dolby Vision 8.4's HLG video holds light up to BT.2100's 1000-nit nominal peak, so TrueHDR's peak stays at or
  // below it there (nothing above it clips in the conversion)
  const DV84_WANT =
    DV_EXPORT &&
    DV_PROFILE === '8.4' &&
    CODEC === 'hevc' &&
    !OUT_IS_MKV &&
    isFile(path.join(ENGINE, 'dvtools', 'dovi_tool.exe'));
  const HDR_PEAK = RTX_HDR ? hdrPeakNits(args.hdr_nits, NATIVE_EXE) : 1000;
  const HDR_NITS = DV84_WANT ? Math.min(HDR_PEAK, 1000) : HDR_PEAK;
  if (RTX_HDR && HDR_NITS < HDR_PEAK)
    say(`RTX HDR peak ${HDR_PEAK} -> ${HDR_NITS} nits: Dolby Vision 8.4's HLG video holds light up to 1000 nits\n`);
  let fragCopy = CODEC === 'vvc';
  const P = partsOf(WORK_PATH, ob, OUT_IS_MKV);
  // kept frames that skip some of the host's do not line up with source pairs, which a resume counts in
  const RESUMABLE = !env.SMV_NO_RESUME && DG_KEEP === 1;
  if (DG_KEEP > 1 && !env.SMV_NO_RESUME)
    say('resume is off for this render (its frames skip some of the generated ones)\n');
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
            : FSRFG_MODE
              ? 'fsrfg'
              : XEFG_MODE
                ? 'xefg'
                : DLSSG_MODE
                  ? 'dlssg'
                  : 'rife';
  // W x H = the decode, WORK = the DLSS mode x the output (DLSS 5 and the model)
  const plan = workPlan(ST, W, H, UPSCALE_F, args.work_scale, !!args.dlssnr, kind === 'dlssg');
  if (typeof plan === 'string') throw new RenderExit(plan);
  [W, H] = [plan.w, plan.h];
  const [WORK_W, WORK_H] = [plan.workW, plan.workH];
  UPSCALE = plan.outW !== W || plan.outH !== H;
  UPSCALE_F = plan.outH / H;
  const IMG_SCALE_VF = plan.vf;
  say(plan.note);
  const sig = resumeSig(ns, env);
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
  } else if (kind === 'fsrfg') {
    const fgdir = env.SMV_FSRFG_DIR || path.join(ENGINE, 'fsrfg');
    if (!isFile(path.join(fgdir, 'smv_fsrfg_bridge.dll')))
      throw new RenderExit(`AMD FSR frame generation unavailable: smv_fsrfg_bridge.dll not found in ${fgdir}`);
  } else if (kind === 'xefg') {
    const xdir = env.SMV_XESS_DIR || path.join(ENGINE, 'xess');
    const miss = ['smv_xefg_bridge.dll', 'libxess_fg.dll', 'libxell.dll'].filter((d) => !isFile(path.join(xdir, d)));
    if (miss.length)
      throw new RenderExit(
        `Intel XeSS frame generation unavailable: ${miss[0]} not found in ${xdir} ` +
          "(the app's Intel XeSS setup installs Intel's files)",
      );
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
  // DLSS 4.5 above one pass's limit = a cascade (dlssgPasses): each later pass its own host, fed the previous pass's
  // frames through a pipe (two hosts in a pipe make the same frames as two separate passes)
  let DG_PASSES = [Math.trunc(args.multi)];
  let DG_LIMIT = 6;
  if (kind === 'dlssg' && args.multi > 2) {
    DG_LIMIT = (await dlssgLimit(NATIVE_EXE, ENGINE, engineCacheDir(ENGINE, env), env)) || 6;
    DG_PASSES = dlssgPasses(Math.trunc(args.multi), DG_LIMIT, !!RTX_HDR || num / den > 100);
    if (!DG_PASSES.length) {
      const reach = (m: number, step: number) => {
        while (m >= 2 && m <= 1000 && !dlssgPasses(m, DG_LIMIT).length) m += step;
        return m;
      };
      const near = [reach(args.multi - 1, -1), reach(args.multi + 1, 1)].filter((m) => m >= 2 && m <= 1000);
      throw new RenderExit(
        `DLSS Frame Generation cannot make ${args.multi}x on this GPU: one pass makes up to ${DG_LIMIT}x, and ` +
          `${args.multi}x is no product of such passes. Pick ${near.map((m) => m + 'x').join(' or ')}, or use GMFSS.`,
      );
    }
    if (DG_PASSES.length > 1)
      say(
        `DLSS Frame Generation ${args.multi}x in ${DG_PASSES.length} passes (${DG_PASSES.map((m) => m + 'x').join(' then ')}): ` +
          "the smallest first on the original frames, each later one between the previous pass's frames, which sit " +
          `closer together (one pass makes up to ${DG_LIMIT}x on this GPU)\n`,
      );
  }
  const CASCADE = DG_PASSES.length > 1;
  // the model lines render.py writes on this route
  if (NO_INTERP)
    say('no-interp mode: GMFSS interpolation disabled (re-encode at source fps with optional Adaptive Sharpen)\n');
  else if (FRUC_MODE) say('Using the NVIDIA Smooth Motion backend for interpolation (NVIDIA Optical Flow)\n');
  else if (FSRFG_MODE) say('Using the AMD FSR frame generation backend for interpolation (FSR 4, native host)\n');
  else if (XEFG_MODE) say('Using the Intel XeSS frame generation backend for interpolation (XeSS-FG, native host)\n');
  else if (DLSSG_MODE) say('Using the DLSS Frame Generation backend for interpolation (DLSS 4.5)\n');
  else if (NVOF_MODE) say('Using the NVIDIA Optical Flow backend for interpolation (native host)\n');
  else if (RIFE_MODE) say('Using the RIFE backend for interpolation (4.25, native host)\n');
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
          `colour ${HDR_COLOR}${vib}${sb}) ${DV84_WANT ? 'HLG (BT.2020) for Dolby Vision 8.4' : 'HDR10 (BT.2020 PQ)'} ` +
          `@ ${OUT_W}x${OUT_H}\n`,
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
  } else if (DV_ACTIVE)
    say(
      DV_PROFILE === '8.4'
        ? 'Dolby Vision Profile 8.4 export ON (an HLG video; plays as HLG where DV is unsupported)\n'
        : 'Dolby Vision Profile 8.1 export ON (plays as HDR10 where DV is unsupported)\n',
    );
  // the video is HLG: Dolby Vision 8.4's base layer
  const HLG_OUT = DV_ACTIVE && DV_PROFILE === '8.4';
  const HP_ACTIVE = wantHp && HDR_ACTIVE && !HLG_OUT;
  if (HP_EXPORT && !HP_ACTIVE) {
    const why = HLG_OUT
      ? 'Dolby Vision 8.4 writes an HLG video and HDR10+ needs HDR10 (PQ)'
      : OUT_IS_MKV
        ? 'output is MKV (HDR10+ export needs MP4)'
        : !isFile(HP_EXE)
          ? 'hdr10plus_tool not found in engine/hptools'
          : 'RTX HDR is not active';
    say(`[hdr10+] HDR10+ export skipped: ${why}\n`);
  } else if (HP_ACTIVE) say('HDR10+ dynamic metadata ON (plays as HDR10 where HDR10+ is unsupported)\n');
  // RTX HDR's statistics (MaxCLL, Dolby Vision, HDR10+) come per output frame from one host
  if (CASCADE && HDR_ACTIVE)
    throw new RenderExit(
      `DLSS Frame Generation at ${args.multi}x runs ${DG_PASSES.length} passes, which RTX HDR does not join yet: ` +
        `turn RTX HDR off, or pick ${DG_LIMIT}x or less.`,
    );
  let nr: { structure: number; tone: number; style: number; passes: number } | null = null;
  if (args.dlssnr) {
    const nrDir = env.SMV_DLSSNR_DIR || path.join(ENGINE, 'dlssnr');
    const miss = ['nvngx.dll', 'nvngx_dlssnr.dll'].filter((f) => !isFile(path.join(nrDir, f)));
    if (miss.length) say(`[dlss5] unavailable, skipping: DLSS 5 files missing in ${nrDir}: ${miss.join(', ')}\n`);
    else
      nr = {
        structure: clamp(args.nr_structure, 0.0, 2.0),
        tone: clamp(args.nr_tone, 0.0, 2.0),
        style: [0, 1, 2].includes(args.nr_style) ? args.nr_style : 0,
        passes: Math.min(20, Math.max(1, Math.round(args.nr_passes ?? 1))),
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
  // THE FILE SPLIT's eligibility (on unless SMV_FILE_SPLIT=0; RIFE at a whole multiple, SDR, no DLSS 5, a constant
  // rate, 600+ frames) and its resume: chunk A keeps the render's work files (P, its sidecar the render's with the
  // split's S, so a one-chain resume can still continue it), chunk B has its own (Q = <work>.b.*, sidecar signed
  // sigB); a later start with both sidecars on the same S resumes each half from its own cut
  const SPLIT_OK =
    env.SMV_FILE_SPLIT !== '0' &&
    kind === 'rife' &&
    RESUMABLE &&
    !CASCADE &&
    !HDR_ACTIVE &&
    !FPS_MODE &&
    DG_KEEP === 1 &&
    !nr &&
    VFR_DEC.length === 0 &&
    NB >= 600;
  const S0 = SPLIT_OK ? Math.floor(NB / 2) : 0;
  const Q = partsOf(WORK_PATH + '.b', ob + '.b', OUT_IS_MKV);
  const sigB = sig + '|split-b';
  const metaOf = (f: string): Record<string, unknown> | null => {
    try {
      const m = JSON.parse(fs.readFileSync(f, 'utf8'));
      return m && typeof m === 'object' && !Array.isArray(m) ? m : null;
    } catch {
      return null;
    }
  };
  const mA = metaOf(P.resumeJson),
    mB = metaOf(Q.resumeJson);
  const SPLIT_RESUME = S0 > 0 && mA?.sig === sig && mB?.sig === sigB && mA?.split === S0 && mB?.split === S0;
  const halfResume = (parts: typeof P, hsig: string, pairs: number) =>
    tryResume({ ...rctx, totalPairs: pairs, concatTxt: parts.concatTxt, parts, sig: hsig });
  // a half's banked frames (chunk A: outputs 0.., chunk B: its own from output S x multi + 1 on); a part that cannot be
  // continued is cleaned by tryResume and that half starts over
  const rzA = SPLIT_RESUME ? halfResume(P, sig, S0 + 1) : null;
  const rzB = SPLIT_RESUME ? halfResume(Q, sigB, NB - S0) : null;
  if (!SPLIT_RESUME) resumeCleanup(Q); // a split no longer resumed as one: chunk B's leftovers go (A's may resume below)
  const rz = SPLIT_RESUME ? null : tryResume(rctx);
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
    `${num}/${den}`,
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
    presetWorks(ENC_FIRST, await encFirstOk),
  );
  if (venc === 'libvvenc') fragCopy = true;
  const ENC_IN_FMT = HDR_ACTIVE
    ? 'x2rgb10le'
    : NO_INTERP && !(SHARPEN > 0 || UPSCALE || nr || args.restore)
      ? DEC_FMT
      : OUT_RAW_FMT;
  // the host runs the encoder's zimg conversion itself, in ffmpeg's avfilter DLL from engine\bin (where it loads it):
  // one host feeds the encoder (no DLSS 4.5 cascade pass between); SMV_OUT_CONV=0 = the encoder converts as before
  const HOST_CONV =
    (env.SMV_OUT_CONV || '') !== '0' &&
    !CASCADE &&
    ['avutil-61.dll', 'avfilter-12.dll'].every((d) => fs.existsSync(path.join(ENGINE, 'bin', d)));
  const encArgs = {
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
    sar: plan.sar,
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
    hlg: HLG_OUT,
    inRateStr: IN_RATE,
    keep: DG_KEEP,
    subsBeside: SUBS_BESIDE,
    hostConv: HOST_CONV,
  };
  const enc = encodePlan(encArgs, env);
  for (const n of enc.notes) say(n);
  // THE FILE SPLIT: two chains at once on the source frames [0, S] and [S, NB), chunk B a resume at output frame
  // S x multi + 1 (its host drops its first frame, the real frame S chunk A ends on); a fresh one only when no one-chain
  // resume is pending, and chunk B only when a second chain fits the card's video memory (else one chain as usual).
  // Each half writes its own part and, once continued, its own continuation; runSplit's caller joins the four in order
  const SPLIT_S = SPLIT_RESUME || (SPLIT_OK && !R.active) ? S0 : 0;
  const encQ = { ...encArgs, vidPart: Q.vidPart, vidPart2: Q.vidPart2 };
  const encAr = SPLIT_S ? encodePlan({ ...encArgs, resumeActive: true }, env) : null; // chunk A continued -> P.vidPart2
  const encBf = SPLIT_S ? encodePlan({ ...encQ, resumeActive: false }, env) : null; // chunk B -> Q.vidPart
  const encBr = SPLIT_S ? encodePlan({ ...encQ, resumeActive: true }, env) : null; // chunk B continued -> Q.vidPart2
  const HDR_MKV_2STAGE = HDR_ACTIVE && OUT_IS_MKV;
  const sidecar = (k: number, t: number) => writeResumeSidecar(P.resumeJson, sig, k, t, venc, 0, 0);
  if (RESUMABLE && !SPLIT_RESUME) sidecar(R.skipSrc, totalUnits); // a split resume keeps both halves' sidecars
  const sharpNote = SHARPEN > 0 ? `  sharpen(adaptive)=${pyG(SHARPEN)}` : '';
  const upNote = UPSCALE ? `  upscale=${pyG(UPSCALE_F)}x->${OUT_W}x${OUT_H}` : '';
  const hdrNote = HDR_ACTIVE ? '  HDR10(TrueHDR,BT.2020 PQ)' : '';
  const nrNote = nr ? `  dlss5(structure ${pyG(nr.structure)}, tone ${pyG(nr.tone)})` : '';
  say(
    `encode: ${venc} visually-lossless -> ${enc.outPix}  (source ${SRC_CODEC || '?'} ${SRC_BITS}bit ${SRC_PIX})` +
      `${nrNote}${sharpNote}${upNote}${hdrNote}\n`,
  );

  // the host's flags
  const cacheDir = engineCacheDir(ENGINE, env);
  fs.mkdirSync(cacheDir, { recursive: true });
  const nargs = [
    '--w',
    String(W),
    '--h',
    String(H),
    '--multi',
    String(DG_PASSES[0]),
    '--frames',
    String(NB || 0),
    '--pixfmt',
    DEC_FMT,
    '--out-pixfmt',
    CASCADE ? OUT_RAW_FMT : ENC_IN_FMT,
    ...(enc.hostVf ? ['--out-vf', enc.hostVf, '--out-sws', enc.hostSws] : []),
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
  if (kind === 'fsrfg') nargs.push('--fsrfg');
  if (kind === 'xefg') nargs.push('--xess-fg');
  if (kind === 'dlssg') nargs.push('--dlssg');
  if (FPS_MODE && !NO_INTERP) nargs.push('--fps-ratio', pyFloatRepr(ratio as number));
  // a cascade's first pass ends at the working size: its last pass makes the final resize, and RTX VSR / the FSR
  // upscaler stay on the one resize that enlarges (here the decode's to the working size only when no final one exists)
  const lastResize = OUT_W !== WORK_W || OUT_H !== WORK_H;
  const [P1_W, P1_H] = CASCADE ? [WORK_W, WORK_H] : [OUT_W, OUT_H];
  if (P1_W !== W || P1_H !== H) {
    nargs.push('--out-w', String(P1_W), '--out-h', String(P1_H));
    if (!CASCADE || !lastResize) {
      if (RTX_VSR) nargs.push('--rtx-vsr', ...VSR_Q);
      if (FSR_UP) nargs.push('--fsr-upscale');
      if (A4K_UP) nargs.push('--anime4k');
      if (A4K_UP && args.anime4k_aa) nargs.push('--anime4k-aa');
      if (ACN_UP) nargs.push('--artcnn');
      if (XESS_UP) nargs.push('--xess-upscale');
    }
  }
  if (SHARPEN > 0) nargs.push('--sharpen', pyG(SHARPEN));
  if (args.restore)
    nargs.push(
      ...(args.restore_model
        ? ['--restore-model', args.restore_model]
        : args.restore_anime4k
          ? ['--restore-anime4k']
          : ['--restore']),
    );
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
      '--hdr-nits',
      String(HDR_NITS),
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
  const drop = NO_INTERP ? 0 : FPS_MODE || DRBA_MODE ? R.pairSkip : 1 + R.pairSkip;
  if (R.active) {
    // a cascade's banked outputs sit on the last pass's grid: the first pass restarts at the source frame (its real
    // frame goes on to the next pass) and the last pass drops what the encoder holds
    if (CASCADE)
      nargs.push(
        '--resume-pair',
        String(R.skipSrc),
        '--resume-out',
        String(R.skipSrc * DG_PASSES[0]),
        '--resume-drop',
        '0',
      );
    else
      nargs.push('--resume-pair', String(R.skipSrc), '--resume-out', String(R.outBase), '--resume-drop', String(drop));
    if (pipeDiscard) nargs.push('--skip-in', String(pipeDiscard));
  }
  /** a cascade's later pass k: the previous pass's frames in at the working size, DLSS 4.5 at its factor, the last
   * pass out to the encoder with the final resize */
  const passArgs = (k: number): string[] => {
    const before = DG_PASSES.slice(0, k).reduce((a, b) => a * b, 1);
    const last = k === DG_PASSES.length - 1;
    const a = [
      '--w',
      String(WORK_W),
      '--h',
      String(WORK_H),
      '--multi',
      String(DG_PASSES[k]),
      '--frames',
      String(NB ? (NB - 1) * before + 1 : 0),
      '--pixfmt',
      OUT_RAW_FMT,
      '--out-pixfmt',
      last ? ENC_IN_FMT : OUT_RAW_FMT,
      '--script',
      path.join(ENGINE, 'live_server.py'),
      '--cache',
      cacheDir,
      '--dlssg',
    ];
    if (last && lastResize) {
      a.push('--out-w', String(OUT_W), '--out-h', String(OUT_H));
      if (RTX_VSR) a.push('--rtx-vsr', ...VSR_Q);
      if (FSR_UP) a.push('--fsr-upscale');
      if (A4K_UP) a.push('--anime4k');
      if (A4K_UP && args.anime4k_aa) a.push('--anime4k-aa');
      if (ACN_UP) a.push('--artcnn');
      if (XESS_UP) a.push('--xess-upscale');
    }
    // PQ / HLG frames go to DLSS 4.5 as HDR10, as in the first pass
    if (SRC_HDR_IN) a.push('--src-hdr', String(tag(ST, 'color_transfer') || '') === 'arib-std-b67' ? 'hlg' : 'pq');
    if (R.active) {
      const out = last ? R.outBase : R.skipSrc * before * DG_PASSES[k];
      a.push(
        '--resume-pair',
        String(R.skipSrc * before),
        '--resume-out',
        String(out),
        '--resume-drop',
        String(last ? drop : 0),
      );
    }
    return a;
  };

  const progCtx = {
    encTarget: enc.target,
    baseBytes: R.baseBytes,
    outPath,
    hdrMkv2stage: HDR_MKV_2STAGE,
    resumable: RESUMABLE, // off while the file split decides and once it runs two chains
    sidecar,
    say,
  };
  const progress = new Progress(progCtx);
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
  // in a cascade the first pass's PROGRESS counts the render and the last pass's OUTFRAMES its frames
  const onLine1 = CASCADE ? (ln: string) => (ln.startsWith('OUTFRAMES ') ? undefined : onLine(ln)) : onLine;
  const passKids: ChildProcess[] = [];
  const passRead: Promise<void>[] = [];
  /** a cascade's later passes after `src` (the first pass's frames), each fed the one before; returns the last
   * pass's frames for the encoder */
  const spawnPasses = (src: number | Readable): number | Readable => {
    let s = src;
    for (let k = 1; k < DG_PASSES.length; k++) {
      const last = k === DG_PASSES.length - 1;
      const p = spawn(NATIVE_EXE, ['--offline', ...passArgs(k)], { ...spawnOpts, stdio: [s, 'pipe', 'pipe'] });
      if (typeof s !== 'number') s.destroy(); // the pass holds its own end
      const rl = readline.createInterface({ input: p.stderr!, crlfDelay: Infinity });
      rl.on('line', (l) => {
        const ln = l.replace(/\r+$/, '');
        if (ln.startsWith('OUTFRAMES ')) {
          if (last) onLine(ln);
        } else if (!ln.startsWith('PROGRESS ') && ln !== 'THUMB') onLine(`pass ${k + 1}: ${ln}`);
      });
      passRead.push(new Promise<void>((res) => rl.once('close', () => res())));
      passKids.push(p);
      s = p.stdout!;
    }
    return s;
  };

  // the resident host (render.py _native_resident_render), else one exe for this render
  let nrc: number | null = null;
  // the encoder's exit code: a host can finish writing every frame before a dying encoder closes
  // its pipe (NVENC refusing a tiny frame), so the host's code alone cannot tell a good render
  let encRc = 0;
  let encRes: ChildProcess | null = null;
  // the decoder's exit code: a decode that dies at its first frame leaves the host 0 frames and a clean exit
  let decRc = 0;
  let decRes: ChildProcess | null = null;
  /** THE FILE SPLIT's two chains (one-shot hosts): chunk A = the source frames 0..S (outputs 0..S x multi) into
   * P.vidPart (continued: P.vidPart2), chunk B = from frame S as a resume at output S x multi + 1 into Q.vidPart
   * (continued: Q.vidPart2); a resumed half starts at its own banked frames (rzA / rzB), a finished one does not run;
   * PROGRESS = A's pairs + B's past S, OUTFRAMES = both, chunk A alone keeps the thumbnail; null = a fresh split whose
   * chunk B would not fit the card's video memory: chunk A stopped, the caller renders one chain */
  const runSplit = async (): Promise<{ nrc: number; encRc: number; decRc: number } | null> => {
    const S = SPLIT_S,
      m = Math.trunc(args.multi);
    const aEnd = S * m + 1, // chunk A's outputs 0..S x multi, chunk B's from aEnd on
      bLen = (NB - S - 1) * m;
    const cA = rzA ? rzA.c : 0,
      cB = rzB ? rzB.c : 0;
    const aDone = cA >= aEnd,
      bDone = cB >= bLen;
    // the next output frame c on the render's grid -> its source pair and the leading outputs its host drops (the real
    // frame + the slots already banked), as the one-chain resume (resumeMap + drop)
    const at = (c: number) => ({ p: Math.floor((c - 1) / m), drop: 1 + ((c - 1) % m) });
    const ra = at(cA || 1),
      rb = at(aEnd + cB);
    if (SPLIT_RESUME)
      say(
        `resume: continuing the previous split render: chunk 1 ${aDone ? 'done' : 'from output frame ' + cA}, ` +
          `chunk 2 ${bDone ? 'done' : 'from output frame ' + (aEnd + cB)}\n`,
      );
    const vf = IMG_SCALE_VF.length ? IMG_SCALE_VF : decodeRgbVf(ST, W, H);
    // chunk A = the whole render's decode cut after frame S (a resumed one decodeFilters' cut at its pair), chunk B =
    // decodeFilters' cut at its pair: one numbering
    const rate = `${num}/${den}`;
    const pA = cA ? ra.p : 0;
    const decA = decodeCmd(FFMPEG, inp, VFR_DEC, decodeFilters(pA, VFR_DEC, vf, rate)[0], DEC_FMT);
    decA.splice(decA.lastIndexOf('-f'), 0, '-frames:v', String(S + 1 - pA));
    const [fB, discB] = decodeFilters(rb.p, VFR_DEC, vf, rate);
    const decB = decodeCmd(FFMPEG, inp, VFR_DEC, fB, DEC_FMT);
    const argsA = [...nargs];
    argsA[argsA.indexOf('--frames') + 1] = String(S + 1);
    if (cA) argsA.push('--resume-pair', String(pA), '--resume-out', String(cA), '--resume-drop', String(ra.drop));
    const argsB: string[] = [];
    for (let i = 0; i < nargs.length; i++) {
      if (nargs[i] === '--thumb' || nargs[i] === '--thumb-off') i++;
      else argsB.push(nargs[i]);
    }
    argsB.push('--resume-pair', String(rb.p), '--resume-out', String(aEnd + cB), '--resume-drop', String(rb.drop));
    if (discB) argsB.push('--skip-in', String(discB));
    let pa = aDone ? S : pA,
      pb = bDone ? NB - 1 : rb.p,
      outA = aDone ? aEnd : 0,
      outB = bDone ? aEnd + cB : 0,
      aLines = 0;
    const step = () => progress.step(pa + Math.max(0, pb - S), totalUnits);
    let chainUp: () => void = () => {};
    const up = new Promise<void>((res) => (chainUp = res));
    const outOf = (ln: string) => {
      const v = ln.split(/\s+/)[1];
      return v !== undefined && /^\s*[+-]?\d+\s*$/.test(v) ? parseInt(v, 10) : 0;
    };
    // both hosts read the one pause flag: the first PAUSED of a pause (from either chunk, chunk B's too once chunk A is
    // done) and the first RESUMED after it reach the render, the other chunk's copy is dropped; the preview = chunk A's
    let pausedShown = false;
    const pauseLine = (ln: string) => {
      const paused = ln.startsWith('PAUSED');
      if (paused !== pausedShown) {
        pausedShown = paused;
        onLine(ln);
      }
    };
    const lineA = (ln: string) => {
      if (ln.startsWith('PROGRESS ')) {
        pa = parseInt(ln.slice(9), 10) || pa;
        step();
        if (++aLines >= 2) chainUp(); // its second pair: every part of chunk A's chain holds its memory
      } else if (ln.startsWith('OUTFRAMES ')) outA = outOf(ln);
      else if (ln.startsWith('PAUSED') || ln.startsWith('RESUMED')) pauseLine(ln);
      else onLine(ln);
    };
    const lineB = (ln: string) => {
      if (ln.startsWith('PROGRESS ')) {
        pb = parseInt(ln.slice(9), 10) || pb;
        step();
      } else if (ln.startsWith('OUTFRAMES ')) outB = outOf(ln);
      else if (ln.startsWith('PAUSED') || ln.startsWith('RESUMED')) pauseLine(ln);
      else if (ln && ln !== 'THUMB') say(`[native] chunk 2: ${ln}\n`);
    };
    const chain = (dec: string[], hargs: string[], ecmd: string[], lineFn: (ln: string) => void) => {
      const d = spawn(dec[0], dec.slice(1), { ...spawnOpts, stdio: ['inherit', 'pipe', 'inherit'] });
      const x = spawn(NATIVE_EXE, ['--offline', ...hargs], { ...spawnOpts, stdio: [d.stdout!, 'pipe', 'pipe'] });
      const e = spawn(ecmd[0], ecmd.slice(1), { ...spawnOpts, stdio: [x.stdout!, 'inherit', 'inherit'] });
      d.stdout!.destroy(); // the exe and the encoder hold their own ends
      x.stdout!.destroy();
      const rl = readline.createInterface({ input: x.stderr!, crlfDelay: Infinity });
      rl.on('line', (l) => lineFn(l.replace(/\r+$/, '')));
      return { d, x, e, closed: new Promise<void>((res) => rl.once('close', () => res())) };
    };
    const cmdA = cA ? (encAr as typeof enc).cmd : enc.cmd,
      cmdB = cB ? (encBr as typeof enc).cmd : (encBf as typeof enc).cmd;
    // the sidecars while the halves run: chunk A's = the render's with S and both halves' progress (the GUI's resume
    // probe reads it), chunk B's = sigB with S; a kill from here on resumes each half
    const commit = () => {
      writeResumeSidecar(Q.resumeJson, sigB, 0, 0, venc, 0, 0, S);
      progCtx.sidecar = (k: number, t: number) => writeResumeSidecar(P.resumeJson, sig, k, t, venc, 0, 0, S);
      progCtx.resumable = RESUMABLE;
      progCtx.sidecar(pa + Math.max(0, pb - S), totalUnits);
    };
    const sides: ReturnType<typeof chain>[] = [];
    if (aDone || bDone) {
      // one half left (or none): it runs alone, no memory fit to decide
      commit();
      if (!aDone) sides.push(chain(decA, argsA, cmdA, lineA));
      if (!bDone) sides.push(chain(decB, argsB, cmdB, lineB));
    } else {
      // chunk B beside chunk A only when a second chain of chunk A's size fits with the host's reserve (max(512 MiB,
      // 4 %); SMV_VRAM_MIB=N = a smaller card): measured on chunk A's own chain after its second pair (decoder, host
      // and encoder all holding their memory); else a fresh split stops chunk A and the render runs one chain, a
      // resumed one runs chunk B after chunk A (both halves' work kept)
      const before = await gpuMemory();
      progCtx.resumable = false;
      const a = chain(decA, argsA, cmdA, lineA);
      a.x.once('exit', () => chainUp());
      await up;
      const after = before && a.x.exitCode === null ? await gpuMemory() : null;
      const cap = after ? Math.min(after.total, parseInt(env.SMV_VRAM_MIB || '', 10) || after.total) : 0;
      const held = before && after ? after.used - before.used : 0;
      const left = after ? Math.round(cap - after.used - Math.max(512, cap * 0.04)) : 0;
      const fits = !!after && held > 0 && held <= left;
      const why =
        after && held > 0
          ? `file split: one chain holds ${held} MiB of video memory, ${Math.max(0, left)} MiB left beside it: `
          : 'file split: no video memory figure for a second chain: ';
      if (!fits && !SPLIT_RESUME) {
        for (const p of [a.d, a.x, a.e]) if (p.exitCode === null) p.kill();
        await Promise.all([waitExit(a.x), waitExit(a.e), waitExit(a.d), a.closed]);
        try {
          fs.unlinkSync(P.vidPart);
        } catch {
          /* none written */
        }
        progCtx.resumable = RESUMABLE;
        nout = 0;
        say(why + 'one chain\n');
        return null;
      }
      commit();
      sides.push(a);
      if (fits) say(why + `two chains at once on source frames 0..${S} and ${S}..${NB - 1}\n`);
      else {
        say(why + 'chunk 2 after chunk 1\n');
        await Promise.all([waitExit(a.x), a.closed]);
      }
      sides.push(chain(decB, argsB, cmdB, lineB));
    }
    const codes = await Promise.all(
      sides.map(async (s) => {
        const x = await waitExit(s.x);
        await s.closed;
        const [e, d] = await Promise.all([waitExit(s.e), waitExit(s.d)]);
        return { x, e, d };
      }),
    );
    nout = outA + Math.max(0, outB - aEnd); // chunk B's OUTFRAMES counts the aEnd frames before it too
    const first = (k: 'x' | 'e' | 'd') => codes.map((c) => c[k]).find((v) => v !== 0) ?? 0;
    return { nrc: first('x'), encRc: first('e'), decRc: first('d') };
  };
  let splitRan = false;
  if (SPLIT_S) {
    const r = await runSplit();
    if (r) {
      nrc = r.nrc;
      encRc = r.encRc;
      decRc = r.decRc;
      splitRan = true;
    }
  }
  if (!splitRan && (env.SMV_OFFLINE_RESIDENT || '') !== '0')
    nrc = await residentRender(
      NATIVE_EXE,
      ENGINE,
      nargs,
      onLine1,
      say,
      (fin, fout) => {
        decRes = spawn(decCmd[0], decCmd.slice(1), { ...spawnOpts, stdio: ['inherit', fin, 'inherit'] });
        const src = CASCADE ? spawnPasses(fout) : fout;
        encRes = spawn(enc.cmd[0], enc.cmd.slice(1), { ...spawnOpts, stdio: [src, 'inherit', 'inherit'] });
        if (typeof src !== 'number') src.destroy();
        return [decRes, ...passKids, encRes];
      },
      env,
    );
  // residentRender waited for both children, so the exit codes are in
  if (encRes !== null) encRc = (encRes as ChildProcess).exitCode ?? 1;
  if (decRes !== null) decRc = (decRes as ChildProcess).exitCode ?? 1;
  if (nrc === null) {
    const dec = spawn(decCmd[0], decCmd.slice(1), { ...spawnOpts, stdio: ['inherit', 'pipe', 'inherit'] });
    const exe = spawn(NATIVE_EXE, ['--offline', ...nargs], { ...spawnOpts, stdio: [dec.stdout!, 'pipe', 'pipe'] });
    const src = CASCADE ? spawnPasses(exe.stdout!) : exe.stdout!;
    const encp = spawn(enc.cmd[0], enc.cmd.slice(1), { ...spawnOpts, stdio: [src, 'inherit', 'inherit'] });
    dec.stdout!.destroy(); // the exe and the encoder hold their own ends
    if (typeof src !== 'number') src.destroy();
    const rl = readline.createInterface({ input: exe.stderr!, crlfDelay: Infinity });
    rl.on('line', (l) => onLine1(l.replace(/\r+$/, '')));
    const closed = new Promise<void>((res) => rl.once('close', () => res()));
    nrc = await waitExit(exe);
    await closed;
    [encRc, decRc] = await Promise.all([waitExit(encp), waitExit(dec)]);
  }
  // a stage that fails closes its input pipe, so every stage before it fails writing into it (the decoder last, exit
  // -22): the failure starts at the last stage that failed, and a later pass's code decides the render's like the
  // first pass's (exit 3 / 4 included)
  const passRcs = await Promise.all(passKids.map(waitExit));
  await Promise.all(passRead);
  let lastFailed = -1;
  passRcs.forEach((c, i) => {
    if (c !== 0) lastFailed = i;
  });
  if (lastFailed >= 0 && encRc === 0) nrc = passRcs[lastFailed];
  const encFailed = `the encoder (${venc}) failed (exit ${encRc}); its error is in the lines above`;
  const decFailed = `the decoder failed (exit ${decRc}); its error is in the lines above`;
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
    // a decoder that exited too only wrote into the failed host's closed pipe
    if (lastFailed >= 0)
      throw new RenderExit(
        `DLSS 4.5 pass ${lastFailed + 2} of ${DG_PASSES.length} failed (exit ${nrc}); its [native] pass ` +
          `${lastFailed + 2} lines above name the reason`,
      );
    throw new RenderExit(`native offline host failed (exit ${nrc}); the [native] lines above name the reason`);
  }
  if (encRc !== 0) throw new RenderExit(encFailed);
  if (decRc !== 0) throw new RenderExit(decFailed); // the host finished on the frames it got: a short or empty video
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

  // the split's video = its pieces in order (chunk A's part, its continuation, chunk B's part, its continuation), one
  // stream copy into P.vidFull, which the finish remuxes as a one-part render's
  if (splitRan) {
    const pieces = [P.vidPart, P.vidPart2, Q.vidPart, Q.vidPart2].filter((f) => fs.existsSync(f));
    const fc = { ffmpeg: FFMPEG, ffprobe: FFPROBE, fragCopy, fragFlags: FRAG_FLAGS, concatTxt: P.concatTxt };
    if (!concatCopy(fc, pieces, P.vidFull))
      throw new RenderExit('file split: joining the two halves failed; the render stays resumable');
  }
  // _finish
  finalizeOutput({
    ffmpeg: FFMPEG,
    inp,
    workPath: WORK_PATH,
    encTarget: enc.target,
    resumable: RESUMABLE,
    resumeActive: R.active,
    vidPart: splitRan ? P.vidFull : P.vidPart,
    vidPart2: P.vidPart2,
    vidFull: P.vidFull,
    hdrActive: HDR_ACTIVE,
    outIsMkv: OUT_IS_MKV,
    hpActive: HP_ACTIVE,
    dvActive: DV_ACTIVE,
    hlg: HLG_OUT,
    resumeDvhpNote: dvhpNote,
    fragCopy,
    fragFlags: FRAG_FLAGS,
    stage2Maps: enc.stage2Maps,
    cropped: enc.crop,
    rtx,
    nits: HDR_NITS,
    masterPrim: HDR_MASTER_PRIM,
    hpExport: () => hpExport(WORK_PATH, FFMPEG, HP_EXE, rtx, HDR_NITS, HDR_MASTER_PRIM, rateStr, outLabel, say),
    dvExport: () =>
      dvExport(
        WORK_PATH,
        FFMPEG,
        DOVI_EXE,
        rtx,
        HDR_NITS,
        HDR_MASTER_PRIM,
        rateStr,
        outLabel,
        OUT_W,
        OUT_H,
        say,
        DV_PROFILE,
      ),
    concatCopy: (parts, dst) =>
      concatCopy(
        { ffmpeg: FFMPEG, ffprobe: FFPROBE, fragCopy, fragFlags: FRAG_FLAGS, concatTxt: P.concatTxt },
        parts,
        dst,
      ),
    resumeCleanup: () => {
      resumeCleanup(P);
      resumeCleanup(Q); // the file split's chunk B
    },
    say,
  });
  if (fs.existsSync(WORK_PATH)) fs.renameSync(WORK_PATH, outPath);
  if (SUBS_BESIDE && fs.existsSync(outPath)) say(saveSubsBeside(FFMPEG, FFPROBE, inp, outPath, tr.sub));
  // the file's frames: every DG_KEEP-th of the host's (framestep keeps frame 0, DG_KEEP, 2 DG_KEEP ...)
  const outFrames = nout ? (DG_KEEP > 1 ? Math.floor((nout - 1) / DG_KEEP) + 1 : nout) : null;
  if (outFrames !== null) say(`OUTFRAMES ${outFrames}\n`);
  if (NO_INTERP) {
    say(`done ${nout} frames (${SHARPEN > 0 ? 'sharpened' : 're-encoded'}, native host) -> ${outPath}\n`);
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
    const it = new FdLines(cfd);
    let ready = false;
    for (let ln = it.next(); ln !== null; ln = it.next()) {
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
    // off the event loop from here on: a cascade's later passes are joined by Node pipes (FdLines)
    for (let ln = await it.nextAsync(); ln !== null; ln = await it.nextAsync()) {
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

/** nvidia-smi's POWER report, or null when it is missing, fails or takes over 5 s. */
function powerQuery(): Promise<string | null> {
  return new Promise((resolve) => {
    try {
      const p = spawn('nvidia-smi', ['-q', '-d', 'POWER'], { windowsHide: true, stdio: ['ignore', 'pipe', 'ignore'] });
      const timer = setTimeout(() => p.kill(), 5000);
      let out = '';
      p.stdout!.setEncoding('utf8');
      p.stdout!.on('data', (d: string) => (out += d));
      p.once('error', () => {
        clearTimeout(timer);
        resolve(null);
      });
      p.once('close', (code) => {
        clearTimeout(timer);
        resolve(code === 0 ? out : null);
      });
    } catch {
      resolve(null);
    }
  });
}

/** render.py _power_notice: one line when the GPU runs under a reduced power limit (`report` = powerQuery's). */
export function powerNotice(report: string | null, say: Say): void {
  if (report === null) return;
  try {
    const vals: Record<string, number> = {};
    for (const line of report.split(/\r\n|\r|\n/)) {
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
