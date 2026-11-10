// Resume, pause preview and progress: the resume after a crash or an exit (the fragmented stage-1
// file, the settings signature, the salvage + decode-verified gapless cut, the output-index to
// source-grid mapping, the native host's per-frame HDR statistics prefix), the pause preview, the
// PROGRESS / SIZE heartbeat. The native host keeps the per-frame HDR records itself. Every
// stderr line goes through `say`; child processes run synchronously with no console window and
// their output dropped (see encode.runChecked).
import { spawnSync, SpawnSyncOptions, SpawnSyncReturns } from 'child_process';
import { createHash } from 'crypto';
import * as fs from 'fs';
import * as path from 'path';
import { PyRuntimeError, pyExcRepr, pyJsonDumps } from './dvhp';
import { CalledProcessError, Env, HpFrame, pyErrText, pyReprStr, Say } from './encode';
import { pyFixed, pyFloatRepr } from './pyfmt';

// --- argument value parsing ---------------------------------------------------------------------
export class PyValueError extends Error {}

const PY_FLOAT_RE =
  /^[+-]?(?:(?:\d(?:_?\d)*(?:\.(?:\d(?:_?\d)*)?)?|\.\d(?:_?\d)*)(?:[eE][+-]?\d(?:_?\d)*)?|inf(?:inity)?|nan)$/i;

/** A float argument's text as a number (Python's float literal forms: `_` separators, inf, nan). */
export function pyFloatParse(s: string): number {
  const t = s.trim();
  if (!PY_FLOAT_RE.test(t)) throw new PyValueError(`could not convert string to float: ${pyReprStr(s)}`);
  const u = t.replace(/_/g, '').toLowerCase();
  if (u.endsWith('nan')) return NaN;
  if (u.endsWith('inf') || u.endsWith('infinity')) return u.startsWith('-') ? -Infinity : Infinity;
  return Number(u);
}

/** An int argument's text as a number (base 10, `_` separators allowed). */
export function pyIntParse(s: string): number {
  const t = s.trim();
  if (!/^[+-]?\d(?:_?\d)*$/.test(t)) throw new PyValueError(`invalid literal for int() with base 10: ${pyReprStr(s)}`);
  return Number(t.replace(/_/g, ''));
}

function pyMax(a: number, b: number): number {
  return b > a ? b : a; // the first one unless the second is larger (a NaN first stays)
}

// --- the render's arguments -----------------------------------------------------------------------
type Kind = 'str' | 'int' | 'float' | 'bool';
interface Spec {
  dest: string;
  kind: Kind;
  def: string | number | boolean | null;
  nargsOpt?: number;
  choices?: (string | number)[];
}

/** The render's options in declaration order (flag -> dest, type, default, the bare flag's value). */
const OPTS: Record<string, Spec> = {
  // the DLSS mode (auto, dlaa, quality, balanced, performance, ultra) or a number in (0, 1]: the
  // working size's share of the output (plan.ts workPlan checks it)
  '--scale': { dest: 'work_scale', kind: 'str', def: null },
  '--fps': { dest: 'fps', kind: 'float', def: null },
  '--sharpen': { dest: 'sharpen', kind: 'float', def: 0.0, nargsOpt: 1.0 },
  '--no-interp': { dest: 'no_interp', kind: 'bool', def: false },
  '--upscale': { dest: 'upscale', kind: 'float', def: 1.0, nargsOpt: 1.5 },
  '--rtx-vsr': { dest: 'rtx_vsr', kind: 'bool', def: false },
  '--vsr-quality': { dest: 'vsr_quality', kind: 'int', def: 4, choices: [1, 2, 3, 4] },
  '--fsr-upscale': { dest: 'fsr_upscale', kind: 'bool', def: false },
  '--anime4k': { dest: 'anime4k', kind: 'bool', def: false },
  '--anime4k-aa': { dest: 'anime4k_aa', kind: 'bool', def: false },
  '--artcnn': { dest: 'artcnn', kind: 'bool', def: false },
  '--xess-upscale': { dest: 'xess_upscale', kind: 'bool', def: false },
  '--rtx-hdr': { dest: 'rtx_hdr', kind: 'bool', def: false },
  '--dv': { dest: 'dv', kind: 'bool', def: false },
  '--dv-profile': { dest: 'dv_profile', kind: 'str', def: '8.1', choices: ['8.1', '8.4'] },
  '--hdr10plus': { dest: 'hdr10plus', kind: 'bool', def: false },
  '--codec': { dest: 'codec', kind: 'str', def: 'hevc', choices: ['hevc', 'av1', 'vvc'] },
  '--restore': { dest: 'restore', kind: 'bool', def: false },
  // Restore's model file (the installed AnimeJaNai model; the host's --restore-model), none = Real-ESRGAN
  '--restore-model': { dest: 'restore_model', kind: 'str', def: null },
  '--restore-anime4k': { dest: 'restore_anime4k', kind: 'bool', def: false },
  '--no-gpu-fit': { dest: 'no_gpu_fit', kind: 'bool', def: false },
  '--dlssnr': { dest: 'dlssnr', kind: 'bool', def: false },
  '--nr-structure': { dest: 'nr_structure', kind: 'float', def: 1.0 },
  '--nr-tone': { dest: 'nr_tone', kind: 'float', def: 1.0 },
  '--nr-skin': { dest: 'nr_skin', kind: 'float', def: -1.0 },
  '--nr-automask': { dest: 'nr_automask', kind: 'int', def: 1, choices: [0, 1] },
  '--nr-style': { dest: 'nr_style', kind: 'int', def: 0, choices: [0, 1, 2] },
  '--nr-passes': {
    dest: 'nr_passes',
    kind: 'int',
    def: 1,
    choices: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
  },
  '--hdr-saturation': { dest: 'hdr_saturation', kind: 'int', def: 100 },
  '--hdr-contrast': { dest: 'hdr_contrast', kind: 'int', def: 100 },
  '--hdr-nits': { dest: 'hdr_nits', kind: 'int', def: 0 },
  '--hdr-color': { dest: 'hdr_color', kind: 'str', def: 'raw', choices: ['raw', 'vivid', 'rtx'] },
  '--hdr-vibrance': { dest: 'hdr_vibrance', kind: 'float', def: 0.0 },
  '--hdr-satboost': { dest: 'hdr_satboost', kind: 'float', def: 0.0 },
  '--fruc': { dest: 'fruc', kind: 'bool', def: false },
  '--fsrfg': { dest: 'fsrfg', kind: 'bool', def: false },
  '--xess-fg': { dest: 'xess_fg', kind: 'bool', def: false },
  '--nvvfg': { dest: 'nvvfg', kind: 'bool', def: false },
  '--dlssg': { dest: 'dlssg', kind: 'bool', def: false },
  '--rife': { dest: 'rife', kind: 'bool', def: false },
  '--rife-drba': { dest: 'rife_drba', kind: 'bool', def: false },
  '--nvof': { dest: 'nvof', kind: 'bool', def: false },
};
const POS: Spec[] = [
  { dest: 'input', kind: 'str', def: null },
  { dest: 'multi', kind: 'int', def: null },
  { dest: 'output', kind: 'str', def: null },
];
const KIND: Record<string, Kind> = Object.fromEntries([...POS, ...Object.values(OPTS)].map((s) => [s.dest, s.kind]));

export type ArgValue = string | number | boolean | null;
/** The parsed arguments: every dest, typed by its spec (KIND says which numbers are floats). */
export type ArgNs = Record<string, ArgValue>;

function convert(spec: Spec, flag: string, s: string): ArgValue {
  const v = spec.kind === 'int' ? pyIntParse(s) : spec.kind === 'float' ? pyFloatParse(s) : s;
  if (spec.choices && !spec.choices.includes(v as string | number))
    throw new Error(`argument ${flag}: invalid choice: ${s}`);
  return v;
}

/** Whether a token looks like an option (a negative number does not: no flag is a number). */
function isOption(t: string): boolean {
  if (!t.startsWith('-') || t === '-') return false;
  if (/^-\d+$|^-\d*\.\d+$/.test(t)) return false;
  return !t.includes(' ') || (t.startsWith('--') && t.split('=')[0] in OPTS);
}

/** The render's command line parser, for the command lines the app builds: exact long flags,
 * `--flag value` / `--flag=value`, the two bare-flag values, and the positionals as ONE block
 * (input multi [output]). Anything else throws: an unknown or malformed argument, and the forms the
 * app never builds (abbreviated flags, positionals split around options, `--`). */
export function parseRenderArgv(argv: string[]): ArgNs {
  const ns: ArgNs = {};
  for (const s of Object.values(OPTS)) ns[s.dest] = s.def;
  const pos: string[] = [];
  let block = -1; // index in argv where the positional block started
  for (let i = 0; i < argv.length; i++) {
    const t = argv[i];
    if (!isOption(t)) {
      if (block >= 0 && i !== block + pos.length) throw new Error('positionals must be one block');
      if (block < 0) block = i;
      pos.push(t);
      continue;
    }
    const eq = t.indexOf('=');
    const flag = eq >= 0 ? t.slice(0, eq) : t;
    const spec = OPTS[flag];
    if (!spec) throw new Error(`unrecognized arguments: ${t}`);
    if (spec.kind === 'bool') {
      if (eq >= 0) throw new Error(`argument ${flag}: ignored explicit argument`);
      ns[spec.dest] = true;
    } else if (eq >= 0) {
      ns[spec.dest] = convert(spec, flag, t.slice(eq + 1));
    } else if (i + 1 < argv.length && !isOption(argv[i + 1])) {
      ns[spec.dest] = convert(spec, flag, argv[++i]);
    } else if (spec.nargsOpt !== undefined) {
      ns[spec.dest] = spec.nargsOpt;
    } else throw new Error(`argument ${flag}: expected one argument`);
  }
  if (pos.length < 2) throw new Error('the following arguments are required: input, multi');
  if (pos.length > 3) throw new Error(`unrecognized arguments: ${pos.slice(3).join(' ')}`);
  ns.input = pos[0];
  ns.multi = convert(POS[1], 'multi', pos[1]);
  ns.output = pos.length > 2 ? pos[2] : null;
  return ns;
}

function pyFloatJson(x: number): string {
  return Number.isNaN(x) ? 'NaN' : !Number.isFinite(x) ? (x > 0 ? 'Infinity' : '-Infinity') : pyFloatRepr(x);
}

function argJson(dest: string, v: ArgValue): string {
  if (typeof v === 'number' && KIND[dest] === 'float') return pyFloatJson(v);
  return pyJsonDumps(v);
}

/** A path's identity on Windows: absolute, backslashes, lower case. */
export function winNormAbs(p: string): string {
  return path.win32.resolve(p).replace(/\//g, '\\').toLowerCase();
}

/** A file's modification time in whole seconds, from sec + nsec * 1e-9 in double (the resume
 * signature's form). */
function pyMtimeInt(p: string): number {
  const ns = fs.statSync(p, { bigint: true }).mtimeNs;
  return Math.trunc(Number(ns / 1000000000n) + Number(ns % 1000000000n) * 1e-9);
}

/** The resume signature: sha1 of the JSON text (pyJsonDumps, keys sorted) of the parsed arguments
 * + the source identity + SMV_CQ. A mismatch means the partial video came from other settings. */
export function resumeSig(ns: ArgNs, env: Env = process.env): string {
  const inp = path.win32.resolve(ns.input as string);
  const d: Record<string, string> = {};
  // no_gpu_fit changes no pixel by itself
  for (const [k, v] of Object.entries(ns)) if (k !== 'no_gpu_fit') d[k] = argJson(k, v);
  const input = winNormAbs(ns.input as string);
  d.input = pyJsonDumps(input);
  d.output = ns.output ? pyJsonDumps(winNormAbs(ns.output as string)) : 'null';
  d.__src = `[${pyJsonDumps(input)}, ${fs.statSync(inp).size}, ${pyMtimeInt(inp)}]`;
  d.__cq = pyJsonDumps(env.SMV_CQ || '');
  const keys = Object.keys(d).sort();
  const text = '{' + keys.map((k) => pyJsonDumps(k) + ': ' + d[k]).join(', ') + '}';
  return createHash('sha1').update(text, 'utf8').digest('hex');
}

// --- the resume assets ------------------------------------------------------------------------------
export interface Parts {
  vidPart: string;
  vidPart2: string;
  vidFull: string;
  resumeJson: string;
  hdrFrames: string;
  concatTxt: string;
  salv: string;
  salv2: string;
  trim: string;
  preview: string;
}

/** The work files of an output whose base (path minus extension) is `ob` and work path `workPath`. */
export function partsOf(workPath: string, ob: string, outIsMkv: boolean): Parts {
  return {
    vidPart: workPath + '.video.mp4',
    vidPart2: workPath + '.video2.mp4',
    vidFull: workPath + '.videofull.mp4',
    resumeJson: workPath + '.resume.json',
    hdrFrames: workPath + '.hdrframes.txt',
    concatTxt: workPath + '.concat.txt',
    salv: workPath + '.salv.mp4',
    salv2: workPath + '.salv2.mp4',
    trim: workPath + '.trim.mp4',
    preview: ob + (outIsMkv ? '.preview.mkv' : '.preview.mp4'),
  };
}

/** Drop every resume artifact. */
export function resumeCleanup(p: Parts): void {
  for (const f of [
    p.vidPart,
    p.vidPart2,
    p.vidFull,
    p.resumeJson,
    p.hdrFrames,
    p.concatTxt,
    p.salv,
    p.salv2,
    p.trim,
    p.preview,
  ]) {
    try {
      fs.unlinkSync(f);
    } catch {
      /* a file that is missing or cannot be removed is skipped */
    }
  }
}

/** Atomic replace of the resume json; a failed write is ignored. split > 0 = the file
 * split's source frame S (both halves' sidecars carry it, native.ts resumes each half from its own work files). */
export function writeResumeSidecar(
  resumeJson: string,
  sig: string,
  pair: number,
  total: number,
  venc: string,
  maxcll: number,
  maxfall: number,
  split = 0,
): void {
  const text =
    `{"sig": ${pyJsonDumps(sig)}, "pair": ${pair}, "total": ${total}, "venc": ${pyJsonDumps(venc)}, ` +
    `"maxcll": ${pyFloatJson(maxcll)}, "maxfall": ${pyFloatJson(maxfall)}` +
    (split > 0 ? `, "split": ${split}` : '') +
    '}';
  try {
    const tmp = resumeJson + '.tmp';
    fs.writeFileSync(tmp, text, 'utf8');
    fs.renameSync(tmp, resumeJson);
  } catch {
    /* the previous sidecar stays */
  }
}

/** A failed rename / delete on Windows, with the two texts the render's lines quote: the message
 * ("[WinError N] text: 'path'") and the repr form ("PermissionError(13, 'text')"). */
class PyOSError extends Error {
  constructor(
    text: string,
    public repr: string,
  ) {
    super(text);
  }
}
const WIN_ERR: Record<string, [number, number, string, string]> = {
  ENOENT: [2, 2, 'FileNotFoundError', 'The system cannot find the file specified'],
  EACCES: [5, 13, 'PermissionError', 'Access is denied'],
  EPERM: [5, 13, 'PermissionError', 'Access is denied'],
  EBUSY: [32, 13, 'PermissionError', 'The process cannot access the file because it is being used by another process'],
};
function winErr(e: unknown, a: string, b?: string): unknown {
  const m = WIN_ERR[(e as NodeJS.ErrnoException).code || ''];
  if (!m) return e;
  return new PyOSError(
    `[WinError ${m[0]}] ${m[3]}: ${pyReprStr(a)}` + (b !== undefined ? ` -> ${pyReprStr(b)}` : ''),
    `${m[2]}(${m[1]}, ${pyReprStr(m[3])})`,
  );
}
function pyReplace(a: string, b: string): void {
  try {
    fs.renameSync(a, b);
  } catch (e) {
    throw winErr(e, a, b);
  }
}
function pyRemove(p: string): void {
  try {
    fs.unlinkSync(p);
  } catch (e) {
    throw winErr(e, p);
  }
}

/** The message text of whatever tryResume catches. */
function pyStr(e: unknown): string {
  const x = e as NodeJS.ErrnoException;
  if (x && x.code === 'ENOENT' && x.syscall && x.syscall.startsWith('spawn'))
    return '[WinError 2] The system cannot find the file specified';
  if (
    e instanceof PyOSError ||
    e instanceof CalledProcessError ||
    e instanceof PyRuntimeError ||
    e instanceof PyValueError
  )
    return e.message;
  return pyErrText(e);
}

/** The repr-form text of whatever the pause preview catches. */
function pyRepr(e: unknown): string {
  if (e instanceof PyOSError) return e.repr;
  if (e instanceof PyValueError) return `ValueError(${pyReprStr(e.message)})`;
  return pyExcRepr(e);
}

// --- salvage, cut, concat ----------------------------------------------------------------------------
function run(cmd: string[], opts: SpawnSyncOptions): SpawnSyncReturns<Buffer> {
  const r = spawnSync(cmd[0], cmd.slice(1), {
    windowsHide: true,
    maxBuffer: Infinity,
    ...opts,
  }) as SpawnSyncReturns<Buffer>;
  if (r.error) throw r.error; // a program that cannot start
  return r;
}

export interface FfCtx {
  ffmpeg: string;
  ffprobe: string;
  fragCopy: boolean;
  fragFlags: string[];
  concatTxt: string;
}

function fragFor(c: FfCtx, dst: string): string[] {
  return c.fragCopy && dst.toLowerCase().endsWith('.mp4') ? c.fragFlags : [];
}

/** Error-tolerant video-only stream copy; true on success. +discardcorrupt drops
 * the torn tail packet the demuxer flags corrupt: av1 muxing chokes on its cut OBU and
 * leaves an mp4 with no moov while ffmpeg exits 0, so a killed AV1 render never resumed; hevc
 * decoded around it (at most one decodable packet is dropped now, one frame re-rendered). */
export function ffCopy(c: FfCtx, src: string, dst: string, extra: string[] = []): boolean {
  const cmd = [
    c.ffmpeg,
    '-v',
    'error',
    '-y',
    '-err_detect',
    'ignore_err',
    '-fflags',
    '+discardcorrupt',
    '-i',
    src,
    '-map',
    '0:v:0',
    '-c',
    'copy',
    ...extra,
    ...fragFor(c, dst),
    dst,
  ];
  return run(cmd, { stdio: 'ignore' }).status === 0 && fs.existsSync(dst);
}

/** A file's video packets: [pts in decode order, keyframe packet indices, stream time_base]. */
export function ffPackets(ffprobe: string, file: string): [number[], number[], number] {
  const cmd = [
    ffprobe,
    '-v',
    'error',
    '-select_streams',
    'v:0',
    '-show_entries',
    'packet=pts,flags',
    '-show_entries',
    'stream=time_base',
    '-of',
    'csv=p=0',
    file,
  ];
  const r = run(cmd, { stdio: ['ignore', 'pipe', 'ignore'] });
  if (r.status !== 0) throw new CalledProcessError(r.status ?? 1, cmd);
  const pts: number[] = [],
    keys: number[] = [];
  let tb = 0.0;
  for (const ln of r.stdout.toString('utf8').split(/\r\n|\r|\n/)) {
    const f = ln.trim().split(',');
    if (!f[0]) continue;
    if (f[0].includes('/')) {
      const [n, d] = f[0].split('/');
      tb = pyIntParse(n) / Math.max(1, pyIntParse(d));
    } else if (f[0] !== 'N/A') {
      if (f.length > 1 && f[1].includes('K')) keys.push(pts.length);
      pts.push(pyIntParse(f[0]));
    }
  }
  return [pts, keys, tb];
}

/** The number of frames that decode from startTime to EOF. */
export function decodedCount(ffmpeg: string, file: string, startTime: number): number {
  const r = run(
    [
      ffmpeg,
      '-v',
      'error',
      '-err_detect',
      'explode',
      '-ss',
      pyFixed(startTime, 6),
      '-i',
      file,
      '-map',
      '0:v:0',
      '-vf',
      'scale=64:36',
      '-f',
      'rawvideo',
      '-pix_fmt',
      'gray',
      '-',
    ],
    { stdio: ['ignore', 'pipe', 'ignore'] },
  );
  return Math.floor(r.stdout.length / (64 * 36));
}

/** The largest packet count c <= cap whose decode-order prefix displays gaplessly. */
export function cleanCut(pts: number[], cap: number): number {
  if (pts.length < 2) return 0;
  const s = [...pts].sort((a, b) => a - b);
  let step = Infinity;
  for (let i = 1; i < s.length; i++) if (s[i] > s[i - 1]) step = Math.min(step, s[i] - s[i - 1]);
  if (step === Infinity) throw new PyValueError('min() iterable argument is empty');
  const base = s[0];
  let runmax = -1,
    best = 0;
  for (let idx = 1; idx <= pts.length; idx++) {
    runmax = pyMax(runmax, pts[idx - 1]);
    if (runmax - base === (idx - 1) * step && idx <= cap) best = idx;
  }
  return best;
}

/** Stream-copy concat of same-parameter video parts. */
export function concatCopy(c: FfCtx, parts: string[], dst: string): boolean {
  // the list file ends each line with "\r\n"
  fs.writeFileSync(
    c.concatTxt,
    parts.map((p) => "file '" + path.win32.resolve(p).replace(/\\/g, '/').replace(/'/g, "'\\''") + "'\r\n").join(''),
    'utf8',
  );
  const cmd = [
    c.ffmpeg,
    '-v',
    'error',
    '-y',
    '-f',
    'concat',
    '-safe',
    '0',
    '-i',
    c.concatTxt,
    '-map',
    '0:v:0',
    '-c',
    'copy',
    ...fragFor(c, dst),
    dst,
  ];
  let ok: boolean;
  try {
    ok = run(cmd, { stdio: 'ignore' }).status === 0 && fs.existsSync(dst);
  } finally {
    try {
      fs.unlinkSync(c.concatTxt);
    } catch {
      /* a list file that cannot be removed stays */
    }
  }
  return ok;
}

// --- the resume point ------------------------------------------------------------------------------
/** The render's emission grid, as tryResume reads it. ratio = output / source rate (--fps, DRBA). */
export interface Grid {
  noInterp: boolean;
  fpsMode: boolean;
  drbaMode: boolean;
  multi: number;
  totalPairs: number;
  ratio: number | null;
  nb: number;
}

/** The latest output index a cut may keep: the render still emits at least one more frame. */
export function resumeCap(g: Grid): number {
  if (g.noInterp) return g.nb;
  if (!g.fpsMode && !g.drbaMode) return 1 + g.multi * (g.totalPairs - 1);
  return Math.max(0, Math.ceil(g.totalPairs * (g.ratio as number) - 0.5) - 1);
}

/** Output-frame index c back onto the source grid: (source pair, banked slots of that pair). */
export function resumeMap(c: number, g: Grid): { p: number; skip: number } {
  if (g.noInterp) return { p: c, skip: 0 };
  if (!g.fpsMode && !g.drbaMode) return { p: Math.floor((c - 1) / g.multi), skip: (c - 1) % g.multi };
  const ratio = g.ratio as number;
  let p = Math.trunc((c + 0.5) / ratio);
  while (p > 0 && Math.ceil(p * ratio - 0.5) > c) p -= 1;
  while (Math.ceil((p + 1) * ratio - 0.5) <= c) p += 1;
  return { p, skip: c - Math.ceil(p * ratio - 0.5) };
}

export interface Resume {
  c: number;
  p: number;
  skip: number;
  venc: string;
  maxcll: number;
  maxfall: number;
}

export interface ResumeCtx extends Grid, FfCtx {
  parts: Parts;
  sig: string;
  resumable: boolean;
  say: Say;
}

function metaNum(v: unknown): number {
  if (!v) return 0.0;
  return typeof v === 'string' ? pyFloatParse(v) : Number(v);
}

/** Salvage and trim a partial render, map the cut to the source grid; null =
 * render fresh (the partial is cleaned up when it cannot be used). A libvvenc prefix turns
 * ctx.fragCopy on. */
export function tryResume(x: ResumeCtx): Resume | null {
  const P = x.parts;
  if (!(x.resumable && fs.existsSync(P.resumeJson) && (fs.existsSync(P.vidPart) || fs.existsSync(P.vidPart2))))
    return null;
  let meta: Record<string, unknown>;
  try {
    const m = JSON.parse(fs.readFileSync(P.resumeJson, 'utf8'));
    meta = m && typeof m === 'object' && !Array.isArray(m) ? m : {};
  } catch {
    meta = {};
  }
  if (meta.sig !== x.sig) {
    x.say('resume: found a partial render from DIFFERENT settings/source; starting fresh\n');
    resumeCleanup(P);
    return null;
  }
  if (meta.venc === 'libvvenc') x.fragCopy = true;
  try {
    if (fs.existsSync(P.vidPart2)) {
      if (ffCopy(x, P.vidPart2, P.salv2)) {
        if (fs.existsSync(P.vidPart)) {
          if (!concatCopy(x, [P.vidPart, P.salv2], P.vidFull))
            throw new PyRuntimeError('concat of previous resume parts failed');
          pyReplace(P.vidFull, P.vidPart);
          pyRemove(P.salv2);
        } else pyReplace(P.salv2, P.vidPart);
      }
      pyRemove(P.vidPart2);
    }
    if (!ffCopy(x, P.vidPart, P.salv)) throw new PyRuntimeError('partial video unreadable');
    const [pts, keys, tb] = ffPackets(x.ffprobe, P.salv);
    const nf = pts.length;
    if (!nf || !keys.length || !tb) throw new PyRuntimeError('no usable frames in the partial video');
    const k = keys[keys.length - 1];
    const good = k + Math.min(decodedCount(x.ffmpeg, P.salv, pts[k] * tb), nf - k);
    const c = cleanCut(pts.slice(0, good), resumeCap(x));
    if (c < 1) throw new PyRuntimeError('no usable frames in the partial video');
    if (c === nf) pyReplace(P.salv, P.vidPart);
    else {
      if (!ffCopy(x, P.salv, P.trim, ['-frames:v', String(c)]))
        throw new PyRuntimeError('trim to the cut point failed');
      const got = ffPackets(x.ffprobe, P.trim)[0].length;
      if (got !== c) throw new PyRuntimeError(`trim produced ${got} frames, wanted ${c}`);
      pyReplace(P.trim, P.vidPart);
      pyRemove(P.salv);
    }
    const { p, skip } = resumeMap(c, x);
    return {
      c,
      p,
      skip,
      venc: meta.venc === undefined ? '' : (meta.venc as string),
      maxcll: metaNum(meta.maxcll),
      maxfall: metaNum(meta.maxfall),
    };
  } catch (e) {
    x.say(`resume: could not continue the partial render (${pyStr(e)}); starting fresh\n`);
    resumeCleanup(P);
    return null;
  }
}

export interface ResumeState {
  active: boolean;
  outBase: number;
  skipSrc: number;
  pairSkip: number;
  venc: string;
  baseBytes: number;
}

/** What follows tryResume: the resume state and the two lines that announce it
 * (the HDR statistics of the banked frames: see nativeHdrPrefix). */
export function applyResume(rz: Resume | null, vidPart: string, nb: number, totalUnits: number, say: Say): ResumeState {
  if (!rz) return { active: false, outBase: 0, skipSrc: 0, pairSkip: 0, venc: '', baseBytes: 0 };
  const baseBytes = fs.statSync(vidPart).size;
  say(
    `resume: continuing the previous render from source frame ${rz.p}/${nb || '?'} (${rz.c} output frames ` +
      `already rendered, ${pyFixed(baseBytes / 1e6, 0)} MB banked)\n`,
  );
  say(`PROGRESS ${rz.p}/${totalUnits}\n`);
  return { active: true, outBase: rz.c, skipSrc: rz.p, pairSkip: rz.skip, venc: rz.venc, baseBytes };
}

export interface HdrPrefix {
  maxcll: number;
  maxfall: number;
  l1: number[][];
  hp: HpFrame[];
}

/** The HDR statistics prefix of a resumable HDR render: a resumed render keeps the
 * banked frames' records of the host's per-frame statistics file (rewritten to exactly those lines,
 * the host appends after them) and takes their maxima; a fresh one deletes the file. `note` = the
 * RESUME_DVHP_NOTE a short file sets (null = unchanged). */
export function nativeHdrPrefix(
  hdrFrames: string,
  resumeActive: boolean,
  outBase: number,
  dv: boolean,
  hp: boolean,
  say: Say,
): { prefix: HdrPrefix | null; note: string | null } {
  if (!resumeActive) {
    try {
      fs.unlinkSync(hdrFrames);
    } catch {
      /* no statistics file to delete */
    }
    return { prefix: null, note: null };
  }
  const kept: string[] = [],
    pl1: number[][] = [],
    php: HpFrame[] = [];
  let mc = 0.0,
    mf = 0.0;
  try {
    const text = new TextDecoder('utf-8', { fatal: true }).decode(fs.readFileSync(hdrFrames)).replace(/\r\n?/g, '\n');
    for (const m of text.matchAll(/[^\n]*\n|[^\n]+$/g)) {
      const ln = m[0];
      if (kept.length === outBase) break;
      const t = ln
        .trim()
        .split(/\s+/)
        .filter((s) => s.length > 0);
      if (t.length < 2 || !ln.endsWith('\n')) break; // a torn last line
      mc = pyMax(mc, pyFloatParse(t[0]));
      mf = pyMax(mf, pyFloatParse(t[1]));
      let j = 2;
      if (dv) {
        pl1.push(t.slice(j + 1, j + 4).map(pyIntParse));
        j += 4;
      }
      if (hp) {
        const v = t.slice(j + 1, j + 14).map(pyIntParse);
        if (!v.length) throw new Error('IndexError');
        php.push({ avg: v[0], maxscl: v.slice(1, 4), dist: v.slice(4, 13) });
      }
      kept.push(ln);
    }
  } catch {
    /* an unreadable file or a malformed line: the records read so far are kept */
  }
  let note: string | null = null;
  if ((dv || hp) && kept.length !== outBase) {
    note =
      `stats rebuild got ${kept.length}/${outBase} banked frames; Dolby Vision / HDR10+ export will be skipped ` +
      '(the HDR10 output itself is complete)';
    say(`resume: ${note}\n`);
  }
  fs.writeFileSync(hdrFrames, kept.join(''), 'utf8');
  return {
    prefix: { maxcll: Math.min(65535, Math.ceil(mc)), maxfall: Math.min(65535, Math.ceil(mf)), l1: pl1, hp: php },
    note,
  };
}

// --- pause preview ---------------------------------------------------------------------------------
export interface PreviewCtx extends FfCtx {
  resumable: boolean;
  stage2Maps: string[];
  outIsMkv: boolean;
  ob: string;
  encTarget: string;
  resumeActive: boolean;
  parts: Parts;
  inp: string;
  say: Say;
}

function lastChars(s: string, n: number): string {
  const a = Array.from(s);
  return a.slice(Math.max(0, a.length - n)).join('');
}

/** On pause, the frames so far + the source's audio / subtitles into a
 * playable <output>.preview file (PREVIEW_READY), best-effort. */
export function emitPausePreview(x: PreviewCtx): void {
  if (!x.resumable || !x.stage2Maps.length) return;
  const ext = x.outIsMkv ? '.mkv' : '.mp4';
  const prev = x.ob + '.preview' + ext,
    tmp = x.ob + '.preview.building' + ext;
  let cat: string | null = null;
  try {
    let vid = x.encTarget;
    if (x.resumeActive) {
      cat = x.ob + '.preview.cat.mp4';
      vid = concatCopy(x, [x.parts.vidPart, x.parts.vidPart2], cat) ? cat : x.encTarget;
    }
    if (!fs.existsSync(vid) || !fs.statSync(vid).isFile() || fs.statSync(vid).size < 1 << 18) {
      x.say('PREVIEW_PENDING too few frames rendered so far\n');
      return;
    }
    const r = run(
      [
        x.ffmpeg,
        '-v',
        'error',
        '-y',
        '-i',
        vid,
        '-i',
        x.inp,
        '-map',
        '0:v:0',
        '-c:v',
        'copy',
        ...x.stage2Maps,
        '-max_interleave_delta',
        '0',
        '-shortest',
        ...fragFor(x, tmp),
        tmp,
      ],
      { stdio: ['ignore', 'ignore', 'pipe'] },
    );
    if (r.status !== 0 || !fs.existsSync(tmp) || !fs.statSync(tmp).isFile()) {
      const err = r.stderr.toString('utf8').replace(/\r\n?/g, '\n').trim().replace(/\n/g, ' | ');
      throw new PyRuntimeError(lastChars(err, 200) || `exit ${r.status}`);
    }
    pyReplace(tmp, prev);
    x.say(`PREVIEW_READY ${prev}\n`);
  } catch (e) {
    try {
      fs.unlinkSync(tmp);
    } catch {
      /* no half-built preview to remove */
    }
    x.say(`[preview] partial preview skipped (${Array.from(pyRepr(e)).slice(0, 200).join('')})\n`);
  } finally {
    if (cat) {
      try {
        fs.unlinkSync(cat);
      } catch {
        /* a concat file that cannot be removed stays */
      }
    }
  }
}

// --- progress ----------------------------------------------------------------------------------------
export interface ProgressCtx {
  encTarget: string;
  baseBytes: number;
  outPath: string;
  hdrMkv2stage: boolean;
  resumable: boolean;
  sidecar: (k: number, total: number) => void;
  say: Say;
}

/** The PROGRESS heartbeat, the resume sidecar refresh, the SIZE projection and
 * the one-time free-space warning. */
export class Progress {
  private diskWarned = false;
  constructor(private x: ProgressCtx) {}

  step(k: number, total: number): void {
    const x = this.x;
    x.say(`PROGRESS ${k}/${total}\n`);
    if (x.resumable) x.sidecar(k, total);
    const frac = total ? k / total : 0.0;
    if (frac < 0.01) return;
    try {
      const st0 = fs.statSync(x.encTarget); // libuv's stat reads the live size through a handle
      if (!st0.isFile()) return; // a directory has no size to project
      const cur = st0.size + x.baseBytes;
      if (cur < 1 << 20) return;
      const proj = Math.trunc(cur / frac);
      x.say(`SIZE ${cur} ${proj}\n`);
      if (!this.diskWarned && frac >= 0.05) {
        this.diskWarned = true;
        const st = fs.statfsSync(path.win32.dirname(x.outPath) || '.');
        const free = Number(st.bavail) * Number(st.bsize);
        const need = proj - cur + (x.hdrMkv2stage || x.resumable ? proj : 0) + (1 << 30);
        if (need > free) {
          x.say(
            `warning: projected output ~${pyFixed(proj / 1e9, 1)} GB` +
              (x.hdrMkv2stage ? ' (x2 transiently for the HDR MKV remux)' : '') +
              ` but only ${pyFixed(free / 1e9, 1)} GB free on the output drive - ` +
              'the render may fail; free up space or change the output location\n',
          );
        }
      }
    } catch {
      /* no size this time: the heartbeat goes on */
    }
  }
}
