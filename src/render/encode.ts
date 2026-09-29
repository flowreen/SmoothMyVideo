// The encoder: engine/render_encode.py ported line by line, the comments
// there carry the reasoning (the codec ladder, the CPU fallbacks and the RAM preflight, the
// quality-first flags, the colour signalling, the track passthrough, the two-stage HDR / resume
// finish). Every function is fed the decisions the orchestrator already made and returns a record
// or an argv; the ones that talk while they work take its `say` / `fatal` callables. The DV and
// HDR10+ exports and the resume concat / cleanup arrive as callables in the
// Finalize record. Child processes run synchronously with no console window, as in python.
import { spawnSync } from 'child_process';
import * as fs from 'fs';
import * as os from 'os';
import { injectHdr10 } from './hdr10meta';
import { MP4_AUDIO_OK, MKV_SUB_COPY_OK, sizeMatrix, Stream, SWS_ACCURATE, tag, Track, ZSC_MATRIX } from './probe';
import { pyFixed } from './pyfmt';

export type Env = Record<string, string | undefined>;
export type Say = (line: string) => void;
export type Fatal = (msg: string) => never;

export const NVENC_MAX = 8192;

/** python subprocess.CalledProcessError, message included (it reaches the GUI log verbatim). */
export class CalledProcessError extends Error {
  constructor(
    public returncode: number,
    public cmd: string[],
  ) {
    super(`Command '${pyReprList(cmd)}' returned non-zero exit status ${returncode}.`);
  }
}

/** python repr() of a str: quote choice and escapes as CPython prints them. */
export function pyReprStr(s: string): string {
  const q = s.includes("'") && !s.includes('"') ? '"' : "'";
  let out = q;
  for (const ch of s) {
    const c = ch.codePointAt(0) as number;
    if (ch === '\\') out += '\\\\';
    else if (ch === q) out += '\\' + q;
    else if (ch === '\n') out += '\\n';
    else if (ch === '\r') out += '\\r';
    else if (ch === '\t') out += '\\t';
    else if (c < 0x20 || c === 0x7f) out += '\\x' + c.toString(16).padStart(2, '0');
    else out += ch;
  }
  return out + q;
}

function pyReprList(xs: string[]): string {
  return '[' + xs.map(pyReprStr).join(', ') + ']';
}

const PY_ERRNO: Record<string, [number, string]> = {
  ENOENT: [2, 'No such file or directory'],
  EACCES: [13, 'Permission denied'],
  EPERM: [13, 'Permission denied'],
  EEXIST: [17, 'File exists'],
  EISDIR: [21, 'Is a directory'],
  ENOSPC: [28, 'No space left on device'],
};

/** str(e) as python prints the exception: an fs error as OSError's "[Errno N] text: 'path'". */
export function pyErrText(e: unknown): string {
  const x = e as NodeJS.ErrnoException;
  const m = x && x.code ? PY_ERRNO[x.code] : undefined;
  if (m) return `[Errno ${m[0]}] ${m[1]}` + (x.path ? `: ${pyReprStr(x.path)}` : '');
  return x instanceof Error ? x.message : String(e);
}

/** subprocess.run(cmd, check=True, creationflags=CREATE_NO_WINDOW): throws CalledProcessError on
 * a non-zero exit (or when the program cannot start, which python raises as OSError). The child's
 * output is dropped: under CREATE_NO_WINDOW with no std handles passed, python's child wrote to
 * its own hidden console, so none of it ever reached the log (measured by the 6b gate). */
export function runChecked(cmd: string[]): void {
  const r = spawnSync(cmd[0], cmd.slice(1), { windowsHide: true, stdio: 'ignore' });
  if (r.error) throw r.error;
  if (r.status !== 0) throw new CalledProcessError(r.status ?? 1, cmd);
}

// --- decode -----------------------------------------------------------------------------------
/** (the decode -vf chain, the frames to discard from the pipe instead). */
export function decodeFilters(resumeSkip: number, vfrDec: string[], imgScaleVf: string[]): [string[], number] {
  const vfr = vfrDec.length > 0;
  const filters = resumeSkip && !vfr ? [`select=gte(n\\,${resumeSkip}),setpts=PTS-STARTPTS`] : [];
  return [filters.concat(imgScaleVf), resumeSkip && vfr ? resumeSkip : 0];
}

/** The raw-frame decoder argv (stdout = the frame pipe); -sws_flags = accurate rounding for the conversions
 * swscale still does (the pack to the pipe's format, a source zimg does not take). */
export function decodeCmd(ffmpeg: string, inp: string, vfrDec: string[], filters: string[], decFmt: string): string[] {
  return [
    ffmpeg,
    '-v',
    'error',
    '-i',
    inp,
    '-sws_flags',
    SWS_ACCURATE,
    ...vfrDec,
    ...(filters.length ? ['-vf', filters.join(',')] : []),
    '-f',
    'rawvideo',
    '-pix_fmt',
    decFmt,
    '-',
  ];
}

// --- the encoder ------------------------------------------------------------------------------
/** Real availability check: open the encoder on one black frame of `size`. */
export function encWorks(ffmpeg: string, name: string, size = '256x256', fast = false): boolean {
  const args = [
    '-hide_banner',
    '-v',
    'error',
    '-f',
    'lavfi',
    '-i',
    `color=c=black:s=${size}:d=1:r=24`,
    '-frames:v',
    '1',
  ];
  if (fast) {
    args.push('-vf', 'format=yuv420p10le');
    if (name === 'libsvtav1') args.push('-preset', '12');
    else if (name === 'libvvenc') args.push('-preset', 'faster');
  }
  try {
    const r = spawnSync(ffmpeg, args.concat(['-c:v', name, '-f', 'null', '-']), { windowsHide: true, stdio: 'ignore' });
    return !r.error && r.status === 0;
  } catch {
    return false;
  }
}

/** Available physical RAM in GB (libuv reads GlobalMemoryStatusEx ullAvailPhys, as python did). */
export function availRamGb(): number {
  try {
    return os.freemem() / 1e9;
  } catch {
    return 0.0;
  }
}

/** [venc, useNvenc]: the encoder this render runs after the probes and fallbacks. works / ram
 * default to the real probes (a gate substitutes them). */
export function chooseEncoder(
  ffmpeg: string,
  codec: string,
  outW: number,
  outH: number,
  outLabel: number,
  resumeVenc: string,
  vidPartName: string,
  say: Say,
  fatal: Fatal,
  works: typeof encWorks = encWorks,
  ram: typeof availRamGb = availRamGb,
): [string, boolean] {
  let venc = ({ av1: 'av1_nvenc', vvc: 'libvvenc' } as Record<string, string>)[codec] ?? 'hevc_nvenc';
  if (outW > NVENC_MAX || outH > NVENC_MAX) {
    const order = codec === 'vvc' || outW * outH > 90_000_000 ? ['libvvenc', 'libsvtav1'] : ['libsvtav1', 'libvvenc'];
    say(
      `${outW}x${outH} exceeds the ${NVENC_MAX}px NVENC/HEVC limit; probing CPU ` +
        'encoders at the output size (one-time, up to ~1 min)...\n',
    );
    const hit = order.find((cand) => works(ffmpeg, cand, `${outW}x${outH}`, true));
    if (hit === undefined) fatal(`no bundled encoder can encode ${outW}x${outH}; lower the upscale target`);
    venc = hit as string;
    say(`encoding ${outW}x${outH} with ${venc} (${venc === 'libvvenc' ? 'H.266/VVC' : 'AV1'}, CPU)\n`);
    const mp = (outW * outH) / 1e6;
    const need = (venc === 'libvvenc' ? 0.36 : 0.55) * mp + 6.0;
    const avail = ram();
    if (avail && avail < need) {
      fatal(
        `${outW}x${outH} needs ~${pyFixed(need, 0)} GB of free RAM (the CPU encoder alone ` +
          `holds ~40 GB of frames in flight at this size) but only ${pyFixed(avail, 0)} GB is ` +
          'available. Close other applications, lower the upscale target, or run on a ' +
          'machine with more memory; proceeding anyway can freeze or hard-crash the ' +
          'whole system (DPC watchdog).',
      );
    }
    say(`RAM preflight: ~${pyFixed(need, 0)} GB needed, ${pyFixed(avail, 0)} GB available\n`);
  } else if (venc === 'libvvenc' && !works(ffmpeg, venc)) {
    say('libvvenc (H.266/VVC) unavailable in this ffmpeg; using HEVC instead\n');
    venc = 'hevc_nvenc';
  }
  let useNvenc = venc.endsWith('_nvenc');
  if (useNvenc && !works(ffmpeg, venc)) {
    say(`NVENC (${venc}) unavailable on this device; falling back to CPU libsvtav1\n`);
    venc = 'libsvtav1';
    useNvenc = false;
  }
  if (venc === 'libsvtav1' && outLabel > 240) {
    if (works(ffmpeg, 'libvvenc', `${outW}x${outH}`, true)) {
      say(`libsvtav1 caps at 240 fps (${outLabel} fps requested); encoding H.266/VVC instead\n`);
      venc = 'libvvenc';
    } else {
      fatal(
        `no available encoder can write ${outLabel} fps: libsvtav1 caps at 240 fps ` +
          'and libvvenc is unavailable. Lower the output multiplier/fps.',
      );
    }
  }
  if (resumeVenc && resumeVenc !== venc) {
    fatal(
      `resume: the interrupted render used ${resumeVenc} but this run would encode with ` +
        `${venc} (encoder availability changed). Fix the encoder (e.g. the NVIDIA driver) ` +
        `to continue, or delete '${vidPartName}' and its .resume.json next ` +
        'to the output to render fresh.',
    );
  }
  return [venc, useNvenc];
}

/** Output pixel format: always 10 bit, 4:4:4 kept where the encoder allows it (NVENC takes its 10 bits in the
 * high bits of 16, as p010le does: handed 16-bit samples it drops the low six, half a code dark on every plane). */
export function outPixfmt(venc: string, useNvenc: boolean, hdrActive: boolean, chroma444: boolean): string {
  if (venc === 'libvvenc') return 'yuv420p10le';
  if (hdrActive) return useNvenc ? 'p010le' : 'yuv420p10le';
  if (chroma444 && (venc === 'h264_nvenc' || venc === 'hevc_nvenc')) return 'yuv444p10msble';
  return useNvenc ? 'p010le' : 'yuv420p10le';
}

/** [the encoder's quality argv, the stderr note or null]; the measured policy sits on
 * render_encode.quality_args. */
export function qualityArgs(
  venc: string,
  useNvenc: boolean,
  outLabel: number,
  dvOrHp: boolean,
  ultra: boolean,
  env: Env = process.env,
): [string[], string | null] {
  let note: string | null = null;
  let qargs: string[];
  if (useNvenc) {
    if (env.SMV_ENC_LOSSLESS === '1') {
      return [
        ['-preset', 'p7', '-tune', 'lossless', '-rc', 'constqp', '-qp', '0', ...(dvOrHp ? ['-bf', '0'] : [])],
        `encoder: SMV_ENC_LOSSLESS=1, ${venc} constant QP 0 lossless\n`,
      ];
    }
    const cq = env.SMV_CQ || (venc === 'av1_nvenc' ? '22' : '17');
    const la = outLabel > 120 ? '0' : '1';
    qargs = [
      '-preset',
      'p7',
      '-tune',
      'hq',
      '-rc',
      'vbr',
      '-cq',
      cq,
      '-b:v',
      '0',
      '-multipass',
      'fullres',
      '-rc-lookahead',
      la,
      '-spatial-aq',
      '1',
      '-temporal-aq',
      '1',
    ];
    // Split-frame encoding stays OFF, pinned so no driver's `auto` can turn it on. Two strips were
    // 1.15x (1080p) / 1.30x (4K) faster for RIFE 2x, but the strips are separate slices with the
    // loop filter off across them, which leaves an unfiltered line at the same row of every frame
    // (1080p: rows 575 / 576, +2 to 3 levels), a repeatable, identifiable artifact, so splitting
    // stays off. SMV_NVENC_SPLIT overrides the mode for measurement only.
    if (venc === 'hevc_nvenc' || venc === 'av1_nvenc') {
      const split = env.SMV_NVENC_SPLIT || '15';
      qargs.push('-split_encode_mode', split);
      if (split !== '15') note = `encoder: ${venc} split frame encoding mode ${split} (SMV_NVENC_SPLIT)\n`;
    }
    if (venc === 'h264_nvenc' || venc === 'hevc_nvenc') qargs.push('-qp_cb_offset', '-2', '-qp_cr_offset', '-2');
    if (dvOrHp) qargs.push('-bf', '0');
  } else if (venc === 'libvvenc') {
    qargs = ['-qp', '17', '-preset', 'fast', '-qpa', '0'];
    if (ultra) qargs.push('-vvenc-params', 'maxparallelframes=2');
  } else {
    qargs = ['-crf', '17', '-preset', '6'];
  }
  return [qargs, note];
}

/** The HEVC profile follows the pixel format; other encoders pick their own. */
export function profileArgs(venc: string, outPix: string): string[] {
  if (venc === 'hevc_nvenc' && outPix === 'yuv444p10msble') return ['-profile:v', 'rext'];
  if (venc === 'hevc_nvenc' && outPix === 'p010le') return ['-profile:v', 'main10'];
  return [];
}

/** True for a source with no YUV matrix of its own: an RGB, palette or gray pixel format, or the gbr matrix. */
export function rgbSource(st: Stream): boolean {
  return String(st.color_space || '') === 'gbr' || /rgb|bgr|gbr|^pal|^gray|^ya|^mono/.test(String(st.pix_fmt || ''));
}

/** True for a source whose every pixel has its own colour (4:4:4 YUV, RGB, a palette): the output keeps 4:4:4
 * where the encoder allows it. */
export function fullChroma(st: Stream): boolean {
  const pix = String(st.pix_fmt || '');
  return pix.includes('444') || String(st.color_space || '') === 'gbr' || /rgb|bgr|gbr|^pal/.test(pix);
}

/** [setparams fields, -color_* flags, the matrix the encoder's conversion uses]: the source signalling carried
 * through, or HDR10 forced. An RGB source has no YUV matrix or range to carry: the YUV output takes BT.709 (BT.601
 * below HD, the size rule of an untagged decode) at TV range. An untagged YUV source converts with the matrix
 * players assume for the output's size, and the conversion names it on the output. */
export function colorArgs(hdrActive: boolean, st: Stream, outW: number, outH: number): [string[], string[], string] {
  if (hdrActive) {
    return [
      ['range=tv', 'colorspace=bt2020nc', 'color_trc=smpte2084', 'color_primaries=bt2020'],
      ['-color_range', 'tv', '-colorspace', 'bt2020nc', '-color_trc', 'smpte2084', '-color_primaries', 'bt2020'],
      'bt2020nc',
    ];
  }
  const rgb = rgbSource(st);
  const sp: string[] = [],
    color: string[] = [];
  let matrix = sizeMatrix(outW, outH);
  for (const [spOpt, flag, key] of [
    ['range', '-color_range', 'color_range'],
    ['colorspace', '-colorspace', 'color_space'],
    ['color_trc', '-color_trc', 'color_transfer'],
    ['color_primaries', '-color_primaries', 'color_primaries'],
  ]) {
    let v = tag(st, key);
    if (rgb && key === 'color_space') v = matrix;
    else if (rgb && key === 'color_range') v = 'tv';
    if (v) {
      sp.push(`${spOpt}=${v}`);
      color.push(flag, String(v));
      if (key === 'color_space') matrix = String(v);
    }
  }
  return [sp, color, matrix];
}

/** The encode -vf: the colour tags, the host's RGB frames (encInFmt) to YUV through zimg with the output's matrix
 * and range (chroma subsampled with Lanczos3), the encoder's pixel format. A matrix outside ZSC_MATRIX or another
 * input format leaves the conversion to swscale. */
export function encodeVf(sp: string[], outPix: string, encInFmt = '', matrix = ''): string {
  const tags = sp.length ? ['setparams=' + sp.join(':')] : [];
  const planarIn = ({ rgb24: 'gbrp', rgb48le: 'gbrp16le', x2rgb10le: 'gbrp10le' } as Record<string, string>)[encInFmt];
  if (!planarIn || !Object.prototype.hasOwnProperty.call(ZSC_MATRIX, matrix))
    return tags.concat([`format=${outPix}`]).join(',');
  const range = sp.includes('range=pc') ? 'full' : 'limited';
  const planarOut = outPix.startsWith('yuv444') ? 'yuv444p10le' : 'yuv420p10le';
  return tags
    .concat([
      `format=${planarIn}`,
      `zscale=matrix=${ZSC_MATRIX[matrix]}:range=${range}:filter=lanczos:dither=none`,
      `format=${planarOut}`,
    ])
    .concat(planarOut === outPix ? [] : [`format=${outPix}`])
    .join(',');
}

/** [the passthrough -map argv, the stderr notes]. */
export function trackMaps(outIsMkv: boolean, aud: Track[], sub: Track[], hasAttach: boolean): [string[], string[]] {
  const maps: string[] = [],
    drops: string[] = [],
    notes: string[] = [];
  if (outIsMkv) {
    maps.push('-map', '1:a?');
  } else {
    const good = aud.filter(([, c]) => MP4_AUDIO_OK.has(c)).map(([i]) => i);
    for (const i of good) maps.push('-map', `1:${i ?? 'None'}`);
    if (good.length < aud.length) drops.push(`${aud.length - good.length} audio track(s) (codec not mp4-compatible)`);
  }
  maps.push('-c:a', 'copy');
  if (outIsMkv && sub.length) {
    maps.push('-map', '1:s?', '-c:s', 'copy');
    sub.forEach(([, c], j) => {
      if (!MKV_SUB_COPY_OK.has(c)) maps.push(`-c:s:${j}`, 'srt');
    });
  } else if (sub.length) {
    drops.push(`${sub.length} subtitle track(s) (mp4 output; use .mkv to keep them)`);
  }
  if (outIsMkv && hasAttach) maps.push('-map', '1:t?');
  maps.push('-map_chapters', '1');
  const nSub = outIsMkv ? sub.length : 0;
  const nAud = outIsMkv ? aud.length : aud.filter(([, c]) => MP4_AUDIO_OK.has(c)).length;
  if (nAud > 1 || nSub || (outIsMkv && hasAttach)) {
    notes.push(
      `passthrough: ${nAud} audio, ${nSub} subtitle track(s)` +
        `${outIsMkv && hasAttach ? ', fonts' : ''}, chapters -> ${outIsMkv ? 'mkv' : 'mp4'}\n`,
    );
  }
  for (const d of drops) notes.push(`passthrough: dropping ${d}\n`);
  return [maps, notes];
}

export interface Encode {
  outPix: string;
  qargs: string[];
  prof: string[];
  color: string[];
  vf: string;
  tq: string[];
  maps: string[];
  stage2Maps: string[];
  in2: string[];
  target: string;
  frag: string[];
  cmd: string[];
  notes: string[];
}

export interface EncodeInput {
  ffmpeg: string;
  inp: string;
  venc: string;
  useNvenc: boolean;
  hdrActive: boolean;
  chroma444: boolean;
  st: Stream;
  outLabel: number;
  outW: number;
  outH: number;
  rateStr: string;
  encInFmt: string;
  aud: Track[];
  sub: Track[];
  hasAttach: boolean;
  outIsMkv: boolean;
  resumable: boolean;
  resumeActive: boolean;
  vidPart: string;
  vidPart2: string;
  workPath: string;
  dvOrHp: boolean;
}

/** The encode command and everything it is built from (render_encode.encode_plan). */
export function encodePlan(e: EncodeInput, env: Env = process.env): Encode {
  const outPix = outPixfmt(e.venc, e.useNvenc, e.hdrActive, e.chroma444);
  const ultra = e.outW > NVENC_MAX || e.outH > NVENC_MAX;
  const [qargs, speedNote] = qualityArgs(e.venc, e.useNvenc, e.outLabel, e.dvOrHp, ultra, env);
  const prof = profileArgs(e.venc, outPix);
  const [sp, color, matrix] = colorArgs(e.hdrActive, e.st, e.outW, e.outH);
  const vf = encodeVf(sp, outPix, e.encInFmt, matrix);
  const tq = ultra ? ['-threads', '1', '-thread_queue_size', '1'] : [];
  let [maps, trackNotes] = trackMaps(e.outIsMkv, e.aud, e.sub, e.hasAttach);
  const notes = (speedNote ? [speedNote] : []).concat(trackNotes);
  let target: string, stage2Maps: string[], in2: string[];
  if (e.resumable) {
    target = e.resumeActive ? e.vidPart2 : e.vidPart;
    [stage2Maps, maps] = [maps, []];
    in2 = [];
  } else if (e.hdrActive && e.outIsMkv) {
    target = e.workPath + '.video.tmp.mp4';
    [stage2Maps, maps] = [maps, []];
    in2 = [];
  } else {
    target = e.workPath;
    stage2Maps = [];
    in2 = ['-i', e.inp];
  }
  const frag = e.resumable ? ['-movflags', '+frag_keyframe+empty_moov+default_base_moof'] : [];
  const cmd = [
    e.ffmpeg,
    '-v',
    'error',
    '-y',
    '-f',
    'rawvideo',
    '-pix_fmt',
    e.encInFmt,
    '-s',
    `${e.outW}x${e.outH}`,
    '-r',
    e.rateStr,
    ...tq,
    '-i',
    '-',
    ...in2,
    '-map',
    '0:v:0',
    ...maps,
    '-c:v',
    e.venc,
    '-sws_flags',
    SWS_ACCURATE,
    '-vf',
    vf,
    '-max_interleave_delta',
    '0',
    ...qargs,
    ...prof,
    ...color,
    ...(e.venc === 'libvvenc' ? ['-strict', 'experimental'] : []),
    ...frag,
    target,
  ];
  return { outPix, qargs, prof, color, vf, tq, maps, stage2Maps, in2, target, frag, cmd, notes };
}

// --- the finish -------------------------------------------------------------------------------
/** What the TrueHDR pass measured during the render (read only after the loop filled it). */
export interface HpFrame {
  avg: number;
  maxscl: number[];
  dist: number[];
}
export interface HdrStats {
  maxcll?: number;
  maxfall?: number;
  l1?: number[][];
  hp?: HpFrame[];
}

/** [MaxCLL, MaxFALL] measured by the TrueHDR pass, 0 / 0 without one. */
export function hdrLight(rtx: HdrStats | null): [number, number] {
  return [Math.trunc((rtx && rtx.maxcll) || 0), Math.trunc((rtx && rtx.maxfall) || 0)];
}

/** Stamp HDR10 static metadata into an mp4; best-effort, a failure is a note. */
export function writeHdr10Metadata(
  target: string,
  hdrActive: boolean,
  rtx: HdrStats | null,
  nits: number,
  masterPrim: string,
  say: Say,
): void {
  if (!hdrActive || !String(target).toLowerCase().endsWith('.mp4')) return;
  try {
    const [cll, fall] = hdrLight(rtx);
    if (injectHdr10(target, nits, 0.0, cll, fall, masterPrim)) {
      say(`HDR10 metadata: mastered ${nits} nits (${masterPrim}), measured MaxCLL ${cll} / MaxFALL ${fall} nits\n`);
    }
  } catch (e) {
    say(`HDR10 metadata: skipped (${pyErrText(e)})\n`);
  }
}

/** Stream-copy remux of the finished video with the source's passthrough tracks. */
export function remuxTracks(
  ffmpeg: string,
  inp: string,
  stage2Maps: string[],
  fragCopy: boolean,
  fragFlags: string[],
  vidSrc: string,
  dst: string,
  say: Say,
): void {
  const frag = fragCopy && dst.toLowerCase().endsWith('.mp4') ? fragFlags : [];
  runChecked([
    ffmpeg,
    '-v',
    'error',
    '-y',
    '-i',
    vidSrc,
    '-i',
    inp,
    '-map',
    '0:v:0',
    '-c:v',
    'copy',
    ...stage2Maps,
    '-max_interleave_delta',
    '0',
    ...frag,
    dst,
  ]);
  if (frag.length) {
    say(
      'vvc: final mp4 written fragmented (resume-safe layout; plays in ' +
        'modern players, streams without faststart)\n',
    );
  }
}

/** Everything the finish reads, captured by the orchestrator at call time (render_encode.Finalize).
 * hpExport / dvExport = the HDR10+ and DV exports of the finished work path. */
export interface Finalize {
  ffmpeg: string;
  inp: string;
  workPath: string;
  encTarget: string;
  resumable: boolean;
  resumeActive: boolean;
  vidPart: string;
  vidPart2: string;
  vidFull: string;
  hdrActive: boolean;
  outIsMkv: boolean;
  hpActive: boolean;
  dvActive: boolean;
  resumeDvhpNote: string;
  fragCopy: boolean;
  fragFlags: string[];
  stage2Maps: string[];
  rtx: HdrStats | null;
  nits: number;
  masterPrim: string;
  hpExport: () => void;
  dvExport: () => void;
  concatCopy: (parts: string[], dst: string) => boolean;
  resumeCleanup: () => void;
  say: Say;
}

function removeQuiet(p: string): void {
  try {
    fs.unlinkSync(p);
  } catch {
    /* python: except OSError: pass */
  }
}

/** Finish the container (the routes sit on render_encode.finalize_output). */
export function finalizeOutput(f: Finalize): void {
  const hdr10 = (target: string) => writeHdr10Metadata(target, f.hdrActive, f.rtx, f.nits, f.masterPrim, f.say);
  const remux = (vidSrc: string, dst: string) =>
    remuxTracks(f.ffmpeg, f.inp, f.stage2Maps, f.fragCopy, f.fragFlags, vidSrc, dst, f.say);

  const hdrMkv2stage = f.hdrActive && f.outIsMkv;
  if (!f.resumable) {
    if (!hdrMkv2stage) {
      hdr10(f.workPath);
      if (f.hpActive) f.hpExport();
      if (f.dvActive) f.dvExport();
      return;
    }
    hdr10(f.encTarget);
    try {
      remux(f.encTarget, f.workPath);
      fs.unlinkSync(f.encTarget);
      f.say('HDR10 metadata carried into the MKV as native MasteringMetadata/MaxCLL elements\n');
    } catch (e) {
      removeQuiet(f.workPath);
      f.say(
        `final MKV remux failed (${pyErrText(e)}); the HDR video (with metadata, without ` +
          `the extra tracks) was kept at ${f.encTarget}\n`,
      );
    }
    return;
  }
  let vid = f.vidPart;
  try {
    if (f.resumeActive) {
      if (!f.concatCopy([f.vidPart, f.vidPart2], f.vidFull)) {
        throw new Error('concat of the banked prefix + continuation failed');
      }
      vid = f.vidFull;
    }
    if (f.hdrActive && f.outIsMkv) {
      const tmp = f.workPath + '.video.tmp.mp4';
      runChecked([
        f.ffmpeg,
        '-v',
        'error',
        '-y',
        '-i',
        vid,
        '-map',
        '0:v:0',
        '-c',
        'copy',
        ...(f.fragCopy ? f.fragFlags : []),
        tmp,
      ]);
      hdr10(tmp);
      remux(tmp, f.workPath);
      fs.unlinkSync(tmp);
      f.say('HDR10 metadata carried into the MKV as native MasteringMetadata/MaxCLL elements\n');
    } else {
      remux(vid, f.workPath);
      if (f.hdrActive) {
        hdr10(f.workPath);
        if (f.resumeDvhpNote && (f.hpActive || f.dvActive)) {
          f.say(`resume: ${f.resumeDvhpNote}\n`);
        } else {
          if (f.hpActive) f.hpExport(); // HDR10+ first, DV second (see finalize_output)
          if (f.dvActive) f.dvExport();
        }
      }
    }
    f.resumeCleanup();
  } catch (e) {
    removeQuiet(f.workPath);
    f.say(
      `final remux failed (${pyErrText(e)}); the rendered video stream was kept at ` +
        `${vid} and the render stays resumable\n`,
    );
  }
}
