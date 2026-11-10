// The encoder: the codec ladder, the CPU fallbacks and the RAM preflight, the
// quality-first flags, the colour signalling, the track passthrough, the two-stage HDR / resume
// finish. Every function is fed the decisions the orchestrator already made and returns a record
// or an argv; the ones that talk while they work take its `say` / `fatal` callables. The DV and
// HDR10+ exports and the resume concat / cleanup arrive as callables in the
// Finalize record. Child processes run synchronously with no console window.
import { spawn, spawnSync } from 'child_process';
import * as fs from 'fs';
import * as os from 'os';
import { injectHdr10 } from './hdr10meta';
import { MP4_AUDIO_OK, MKV_SUB_COPY_OK, sizeMatrix, Stream, SWS_ACCURATE, tag, Track, ZSC_MATRIX } from './probe';
import { pyFixed } from './pyfmt';

export type Env = Record<string, string | undefined>;
export type Say = (line: string) => void;
export type Fatal = (msg: string) => never;

export const NVENC_MAX = 8192;
/** The smallest even frame [w, h] each NVENC encoder opens; below it NVENC refuses ("Frame dimensions are less than
 * the minimum supported value"). */
export const NVENC_MIN: Record<string, [number, number]> = { hevc_nvenc: [130, 34], av1_nvenc: [130, 66] };

/** True when NVENC encoder `venc` refuses a w x h frame. */
export function belowNvencMin(venc: string, w: number, h: number): boolean {
  const m = NVENC_MIN[venc];
  return m !== undefined && (w < m[0] || h < m[1]);
}

/** A child process that ended with a non-zero exit code; the message reaches the GUI log verbatim. */
export class CalledProcessError extends Error {
  constructor(
    public returncode: number,
    public cmd: string[],
  ) {
    super(`Command '${pyReprList(cmd)}' returned non-zero exit status ${returncode}.`);
  }
}

/** A string in Python's repr form: its quote choice and its escapes. */
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

/** An error's text for the log: an fs error as "[Errno N] text: 'path'". */
export function pyErrText(e: unknown): string {
  const x = e as NodeJS.ErrnoException;
  const m = x && x.code ? PY_ERRNO[x.code] : undefined;
  if (m) return `[Errno ${m[0]}] ${m[1]}` + (x.path ? `: ${pyReprStr(x.path)}` : '');
  return x instanceof Error ? x.message : String(e);
}

/** Runs cmd to its end with no console window: throws CalledProcessError on a non-zero exit, or
 * the spawn error when the program cannot start. The child's output is dropped, so none of it
 * reaches the log. */
export function runChecked(cmd: string[]): void {
  const r = spawnSync(cmd[0], cmd.slice(1), { windowsHide: true, stdio: 'ignore' });
  if (r.error) throw r.error;
  if (r.status !== 0) throw new CalledProcessError(r.status ?? 1, cmd);
}

// --- decode -----------------------------------------------------------------------------------
/** (the decode -vf chain, the frames to discard from the pipe instead). select counts the stream's frames while the
 * decoder's own rate conversion runs after it (a video that starts after the file gets its first frame repeated
 * there): the fps filter first (rate = the source's num/den) puts select on the numbering of the frames a plain decode
 * emits, so a cut at resumeSkip continues that decode exactly. */
export function decodeFilters(
  resumeSkip: number,
  vfrDec: string[],
  imgScaleVf: string[],
  rate: string,
): [string[], number] {
  const vfr = vfrDec.length > 0;
  const filters =
    resumeSkip && !vfr ? [`fps=${rate}:start_time=0,select=gte(n\\,${resumeSkip}),setpts=PTS-STARTPTS`] : [];
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
/** The NVENC encoders whose quality call asks for lookahead level 1 (qualityArgs). */
const NVENC_LEVEL = new Set(['hevc_nvenc', 'av1_nvenc']);

/** encWorks's ffmpeg arguments: one black frame of `size` through the encoder into the null muxer; `level` opens
 * the encoder as the quality call does, with a lookahead frame and lookahead level 1. */
function encWorksArgs(name: string, size: string, fast: boolean, level: boolean): string[] {
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
  const enc = ['-c:v', name];
  if (level) enc.push('-rc-lookahead', '1', '-lookahead_level', '1');
  return args.concat(enc, ['-f', 'null', '-']);
}

/** Real availability check: open the encoder on one black frame of `size`. */
export function encWorks(ffmpeg: string, name: string, size = '256x256', fast = false, level = false): boolean {
  try {
    const r = spawnSync(ffmpeg, encWorksArgs(name, size, fast, level), { windowsHide: true, stdio: 'ignore' });
    return !r.error && r.status === 0;
  } catch {
    return false;
  }
}

/** encWorks at the default size, with the lookahead level where the encoder's quality call takes one, started now
 * and answered when its ffmpeg exits: the route starts it before the probes, so the check's ~0.2 s runs beside them
 * instead of after. */
export function encWorksAsync(ffmpeg: string, name: string): Promise<boolean> {
  return new Promise((resolve) => {
    try {
      const args = encWorksArgs(name, '256x256', false, NVENC_LEVEL.has(name));
      const p = spawn(ffmpeg, args, { windowsHide: true, stdio: 'ignore' });
      p.once('error', () => resolve(false));
      p.once('exit', (code) => resolve(code === 0));
    } catch {
      resolve(false);
    }
  });
}

/** chooseEncoder's `works` with one answer known: `name` at the default size and mode, opened the way encWorksAsync
 * opens it, returns `ok`; every other check runs encWorks. */
export function presetWorks(name: string, ok: boolean): typeof encWorks {
  return (ffmpeg, n, size, fast, level) =>
    n === name && size === undefined && !fast && !!level === NVENC_LEVEL.has(name)
      ? ok
      : encWorks(ffmpeg, n, size, fast, level);
}

/** The encoder a codec asks for first (chooseEncoder's start, before the size and availability fallbacks). */
export function codecEncoder(codec: string): string {
  return ({ av1: 'av1_nvenc', vvc: 'libvvenc' } as Record<string, string>)[codec] ?? 'hevc_nvenc';
}

/** Available physical RAM in GB (libuv reads GlobalMemoryStatusEx ullAvailPhys). */
export function availRamGb(): number {
  try {
    return os.freemem() / 1e9;
  } catch {
    return 0.0;
  }
}

/** [venc, useNvenc, nvencLevel]: the encoder this render runs after the probes and fallbacks, and whether NVENC
 * opens with the quality call's lookahead level. works / ram default to the real probes (a gate substitutes them). */
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
): [string, boolean, boolean] {
  let venc = codecEncoder(codec);
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
  if (venc === 'av1_nvenc' && belowNvencMin(venc, outW, outH)) {
    // AV1 has no crop a decoder applies, so the HEVC route's padding cannot shrink it back: the CPU AV1 encoder
    say(
      `${outW}x${outH} is below av1_nvenc's smallest frame (${NVENC_MIN.av1_nvenc.join('x')}); ` +
        'encoding AV1 on the CPU with libsvtav1\n',
    );
    venc = 'libsvtav1';
  }
  let useNvenc = venc.endsWith('_nvenc');
  // The lookahead level is a device capability, and ffmpeg refuses to open an encoder asked for a level the device
  // lacks: NVENC is opened with the level first, then without it.
  const nvencLevel = useNvenc && NVENC_LEVEL.has(venc) && works(ffmpeg, venc, undefined, false, true);
  if (useNvenc && !nvencLevel) {
    if (!works(ffmpeg, venc)) {
      say(`NVENC (${venc}) unavailable on this device; falling back to CPU libsvtav1\n`);
      venc = 'libsvtav1';
      useNvenc = false;
    } else if (NVENC_LEVEL.has(venc)) {
      say(`NVENC (${venc}) has no lookahead level on this device; encoding without it\n`);
    }
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
  return [venc, useNvenc, nvencLevel];
}

/** Output pixel format: always 10 bit, 4:4:4 kept where the encoder allows it (NVENC takes its 10 bits in the
 * high bits of 16, as p010le does: handed 16-bit samples it drops the low six, half a code dark on every plane). */
export function outPixfmt(venc: string, useNvenc: boolean, hdrActive: boolean, chroma444: boolean): string {
  if (venc === 'libvvenc') return 'yuv420p10le';
  if (hdrActive) return useNvenc ? 'p010le' : 'yuv420p10le';
  if (chroma444 && (venc === 'h264_nvenc' || venc === 'hevc_nvenc')) return 'yuv444p10msble';
  return useNvenc ? 'p010le' : 'yuv420p10le';
}

/** [the encoder's quality argv, the stderr note or null]. */
export function qualityArgs(
  venc: string,
  useNvenc: boolean,
  outLabel: number,
  dvOrHp: boolean,
  ultra: boolean,
  env: Env = process.env,
  nvencLevel = true,
): [string[], string | null] {
  let qargs: string[];
  if (useNvenc) {
    if (env.SMV_ENC_LOSSLESS === '1') {
      return [
        ['-preset', 'p7', '-tune', 'lossless', '-rc', 'constqp', '-qp', '0', ...(dvOrHp ? ['-bf', '0'] : [])],
        `encoder: SMV_ENC_LOSSLESS=1, ${venc} constant QP 0 lossless\n`,
      ];
    }
    // HEVC and AV1: lookahead level 1 at a lookahead depth of 1 at every output rate (fewer bits at equal PSNR /
    // VMAF on most clips; on tween-dense streams above 120 fps it beats no lookahead, where depth 1 alone loses
    // ~3 dB on HEVC and changes nothing on AV1). The level needs a lookahead frame, so both keep depth 1 above
    // 120 fps. A device without the level, and SMV_NVENC_LEVEL=0, encode without it.
    const level = nvencLevel && env.SMV_NVENC_LEVEL !== '0' && NVENC_LEVEL.has(venc);
    // AV1's CQ goes with its lookahead level: the level makes a CQ spend fewer bits and, on anime, lose 1.7 to
    // 2.7 dB, so 19 is the highest CQ that keeps, on every test clip, the PSNR and VMAF of 22 without the level.
    const cq = env.SMV_CQ || (venc === 'av1_nvenc' ? (level ? '19' : '22') : '17');
    const la = level || outLabel <= 120 ? '1' : '0';
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
    if (level) qargs.push('-lookahead_level', '1');
    // Split-frame encoding stays OFF, pinned so no driver's `auto` can turn it on. Two strips were
    // 1.15x (1080p) / 1.30x (4K) faster for RIFE 2x, but the strips are separate slices with the
    // loop filter off across them, which leaves an unfiltered line at the same row of every frame
    // (1080p: rows 575 / 576, +2 to 3 levels), a repeatable, identifiable artifact, so splitting
    // stays off.
    if (venc === 'hevc_nvenc' || venc === 'av1_nvenc') qargs.push('-split_encode_mode', '15');
    if (venc === 'h264_nvenc' || venc === 'hevc_nvenc') qargs.push('-qp_cb_offset', '-2', '-qp_cr_offset', '-2');
    if (dvOrHp) qargs.push('-bf', '0');
  } else if (venc === 'libvvenc') {
    qargs = ['-qp', '17', '-preset', 'fast', '-qpa', '0'];
    if (ultra) qargs.push('-vvenc-params', 'maxparallelframes=2');
  } else {
    qargs = ['-crf', '17', '-preset', '6'];
  }
  return [qargs, null];
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
export function colorArgs(
  hdrActive: boolean,
  st: Stream,
  outW: number,
  outH: number,
  hlg = false,
): [string[], string[], string] {
  // the host's HDR frames are BT.2020 PQ (the -vf tags); `hlg` = the video is written as HLG, encodeVf converts them
  if (hdrActive) {
    return [
      ['range=tv', 'colorspace=bt2020nc', 'color_trc=smpte2084', 'color_primaries=bt2020'],
      [
        '-color_range',
        'tv',
        '-colorspace',
        'bt2020nc',
        '-color_trc',
        hlg ? 'arib-std-b67' : 'smpte2084',
        '-color_primaries',
        'bt2020',
      ],
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
 * input format leaves the conversion to swscale. `pad` = [w, h, padded w, padded h]: the frame padded on the planar
 * format (pad and fillborders take no semi-planar input), the padding = the edge pixels smeared, so no hard edge
 * rings into the visible blocks. `cut` = [w, h]: the frame cropped to it before the conversion (an odd size into
 * 4:2:0, which zimg refuses). `hlg` = the PQ frames converted to HLG in the same zimg step, BT.2100's 1000-nit
 * nominal peak (203 nits = HLG 75 %, BT.2408); the HDR frames always take the zimg route (gbrp10le, BT.2020). */
const PLANAR_IN: Record<string, string> = { rgb24: 'gbrp', rgb48le: 'gbrp16le', x2rgb10le: 'gbrp10le' };
/** The host's RGB frames go through zimg (encodeVf's route for a known matrix and input format), not swscale. */
export function zimgRoute(encInFmt: string, matrix: string): boolean {
  return (
    Object.prototype.hasOwnProperty.call(PLANAR_IN, encInFmt) &&
    Object.prototype.hasOwnProperty.call(ZSC_MATRIX, matrix)
  );
}
export function encodeVf(
  sp: string[],
  outPix: string,
  encInFmt = '',
  matrix = '',
  pad: number[] | null = null,
  cut: number[] | null = null,
  hlg = false,
): string {
  const tags = sp.length ? ['setparams=' + sp.join(':')] : [];
  const planarOut = outPix.startsWith('yuv444') ? 'yuv444p10le' : 'yuv420p10le';
  const padF = pad
    ? [`pad=${pad[2]}:${pad[3]}:0:0`, `fillborders=right=${pad[2] - pad[0]}:bottom=${pad[3] - pad[1]}:mode=smear`]
    : [];
  const cutF = cut ? [`crop=${cut[0]}:${cut[1]}:0:0`] : [];
  const planarIn = PLANAR_IN[encInFmt];
  if (!zimgRoute(encInFmt, matrix)) {
    const planar = padF.length || cutF.length ? [`format=${planarOut}`, ...cutF, ...padF] : [];
    return tags.concat(planar, [`format=${outPix}`]).join(',');
  }
  const range = sp.includes('range=pc') ? 'full' : 'limited';
  return tags
    .concat([`format=${planarIn}`], cutF, [
      `zscale=matrix=${ZSC_MATRIX[matrix]}:range=${range}${hlg ? ':transfer=arib-std-b67:npl=1000' : ''}:filter=lanczos:` +
        'dither=none',
      `format=${planarOut}`,
    ])
    .concat(padF)
    .concat(planarOut === outPix ? [] : [`format=${outPix}`])
    .join(',');
}

/** [the passthrough -map argv, the stderr notes]. */
export function trackMaps(
  outIsMkv: boolean,
  aud: Track[],
  sub: Track[],
  hasAttach: boolean,
  subsBeside = false,
): [string[], string[]] {
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
  } else if (sub.length && subsBeside) {
    notes.push(`passthrough: ${sub.length} subtitle track(s) go beside the video (MKV cannot hold over 1000 fps)\n`);
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
  crop: boolean;
  hostVf: string; // the conversion the host runs itself (--out-vf), '' = the encoder converts as before
  hostSws: string; // the host's --out-sws (the encoder's -sws_flags), '' with no hostVf
}

export interface EncodeInput {
  ffmpeg: string;
  inp: string;
  venc: string;
  useNvenc: boolean;
  nvencLevel: boolean; // NVENC opens with the quality call's lookahead level (chooseEncoder)
  hdrActive: boolean;
  chroma444: boolean;
  st: Stream;
  outLabel: number;
  outW: number;
  outH: number;
  sar: [number, number]; // the output's pixel shape (the plan's), written on the stream when not square
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
  hlg?: boolean; // the HDR video as HLG (Dolby Vision 8.4's base layer), converted from the host's PQ frames
  inRateStr?: string; // the host's frame rate when the file keeps only every `keep`-th frame (rateStr = the file's)
  keep?: number; // > 1: the encode keeps every keep-th frame (DLSS 4.5 at a target between whole multiples)
  subsBeside?: boolean; // an MP4 above 1000 fps: its subtitle tracks are saved beside it (saveSubsBeside), not dropped
  hostConv?: boolean; // the host may run the encoder's zimg conversion itself (ffmpeg's avfilter DLL in the host)
}

/** The source's subtitle tracks as files beside `out` (an MP4 above 1000 fps: MKV cannot time it, MP4 holds none of
 * them): `<base>.<lang>.<ext>`, a later track of the same language `<base>.<lang>.<k>.<ext>`; ASS / SSA, SRT and
 * WebVTT as they are, mov_text as SRT, PGS as .sup, any other kind skipped. Returns the log line. */
export function saveSubsBeside(ffmpeg: string, ffprobe: string, inp: string, out: string, sub: Track[]): string {
  const base = out.replace(/\.[^.\\/]+$/, '');
  const langs = new Map<number, string>();
  const pr = spawnSync(
    ffprobe,
    [
      '-v',
      'error',
      '-select_streams',
      's',
      '-show_entries',
      'stream=index:stream_tags=language',
      '-of',
      'csv=p=0',
      inp,
    ],
    { encoding: 'utf8', windowsHide: true },
  );
  for (const ln of String(pr.stdout || '').split(/\r?\n/)) {
    const [i, l] = ln.trim().split(',');
    if (i && l && /^[A-Za-z-]{1,8}$/.test(l)) langs.set(Number(i), l.toLowerCase());
  }
  // codec -> (muxer, extension, -c:s)
  const kinds: Record<string, [string, string, string]> = {
    ass: ['ass', '.ass', 'copy'],
    ssa: ['ass', '.ass', 'copy'],
    subrip: ['srt', '.srt', 'copy'],
    srt: ['srt', '.srt', 'copy'],
    webvtt: ['webvtt', '.vtt', 'copy'],
    mov_text: ['srt', '.srt', 'srt'],
    hdmv_pgs_subtitle: ['sup', '.sup', 'copy'],
  };
  const count = new Map<string, number>();
  const saved: string[] = [],
    skipped: string[] = [];
  for (const [i, c] of sub) {
    const k = kinds[c];
    if (!k || i === null) {
      skipped.push(c);
      continue;
    }
    const lang = langs.get(i) || 'und';
    const n = (count.get(lang) || 0) + 1;
    count.set(lang, n);
    const dst = `${base}.${lang}${n > 1 ? '.' + n : ''}${k[1]}`;
    const r = spawnSync(ffmpeg, ['-v', 'error', '-y', '-i', inp, '-map', `0:${i}`, '-c:s', k[2], '-f', k[0], dst], {
      windowsHide: true,
    });
    if (r.status === 0 && fs.existsSync(dst)) saved.push(dst.slice(dst.lastIndexOf('\\') + 1));
    else skipped.push(c);
  }
  return (
    `subtitles: ${saved.length} track(s) saved beside the video (MKV cannot hold over 1000 fps)` +
    (saved.length ? `: ${saved.join(', ')}` : '') +
    (skipped.length ? `; not saved: ${skipped.join(', ')}` : '') +
    '\n'
  );
}

/** The encode command and everything it is built from. */
export function encodePlan(e: EncodeInput, env: Env = process.env): Encode {
  const outPix = outPixfmt(e.venc, e.useNvenc, e.hdrActive, e.chroma444);
  const ultra = e.outW > NVENC_MAX || e.outH > NVENC_MAX;
  const [qargs, speedNote] = qualityArgs(e.venc, e.useNvenc, e.outLabel, e.dvOrHp, ultra, env, e.nvencLevel);
  const prof = profileArgs(e.venc, outPix);
  const [sp, color, matrix] = colorArgs(e.hdrActive, e.st, e.outW, e.outH, !!e.hlg);
  // 4:2:0 holds no odd size (HEVC and VVC crop in steps of 2, zimg refuses it, NVENC rounds it down itself): an odd
  // frame is written at the even floor.
  const sub = outPix.startsWith('yuv444') ? 1 : 2;
  const [encW, encH] = [e.outW - (e.outW % sub), e.outH - (e.outH % sub)];
  const cut = encW !== e.outW || encH !== e.outH ? [encW, encH] : null;
  // A frame below hevc_nvenc's smallest: encoded padded up to it, then the HEVC conformance window (the SPS crop every
  // decoder applies) cuts it back; the container's own size fields keep the padded size until the finish's
  // stream-copy remux rewrites them from the SPS.
  const crop = e.venc === 'hevc_nvenc' && belowNvencMin(e.venc, encW, encH);
  const [minW, minH] = NVENC_MIN.hevc_nvenc;
  const pad = crop ? [encW, encH, Math.max(encW, minW), Math.max(encH, minH)] : null;
  // the frames arrive as raw planes with no pixel shape: an anamorphic one is tagged on them, the encoder and the
  // container take it from there (the stream SAR, the bitstream's VUI)
  const sarTag = e.sar[0] !== e.sar[1] ? `,setsar=${e.sar[0]}/${e.sar[1]}` : '';
  const keep = e.keep && e.keep > 1 ? e.keep : 1;
  // the host runs the zimg route to p010le itself (its --out-vf / --out-sws in ffmpeg's own avfilter DLL, the same bytes)
  // for SDR when nothing is cut, padded or stepped: ffmpeg then ingests p010le, half the bytes of rgb48le, and its -vf
  // only tags the frames (setparams, the colour the encoder's VUI and format negotiation read from them: untagged
  // frames get the stream's transfer / primaries unknown and a colour conversion of their own)
  const hostVf =
    e.hostConv && !e.hdrActive && outPix === 'p010le' && !pad && !cut && keep === 1 && zimgRoute(e.encInFmt, matrix)
      ? encodeVf(sp, outPix, e.encInFmt, matrix, null, null, false)
      : '';
  // the encoder-side zimg step stamps its matrix on the frames; with the host converting, setparams does, so an
  // untagged source's output still names the matrix its conversion used
  const hostSp = sp.some((s) => s.startsWith('colorspace=')) ? sp : sp.concat(`colorspace=${matrix}`);
  const vf = hostVf
    ? ['setparams=' + hostSp.join(':')].concat(sarTag ? [sarTag.slice(1)] : []).join(',')
    : (keep > 1 ? `framestep=${keep},` : '') + encodeVf(sp, outPix, e.encInFmt, matrix, pad, cut, !!e.hlg) + sarTag;
  const bsf = crop ? ['-bsf:v', `hevc_metadata=width=${encW}:height=${encH}`] : [];
  const tq = ultra ? ['-threads', '1', '-thread_queue_size', '1'] : [];
  let [maps, trackNotes] = trackMaps(e.outIsMkv, e.aud, e.sub, e.hasAttach, !!e.subsBeside);
  const notes = (speedNote ? [speedNote] : []).concat(trackNotes);
  if (cut) notes.push(`encoder: 4:2:0 holds no odd size; ${e.outW}x${e.outH} is written as ${encW}x${encH}\n`);
  if (pad) {
    notes.push(
      `encoder: ${encW}x${encH} is below hevc_nvenc's smallest frame (${minW}x${minH}); encoded padded to ` +
        `${pad[2]}x${pad[3]}, the HEVC stream cropped back to ${encW}x${encH}\n`,
    );
  }
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
    hostVf ? outPix : e.encInFmt,
    '-s',
    `${e.outW}x${e.outH}`,
    '-r',
    keep > 1 && e.inRateStr ? e.inRateStr : e.rateStr,
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
    ...(vf ? ['-vf', vf] : []),
    '-max_interleave_delta',
    '0',
    ...qargs,
    ...prof,
    ...color,
    ...bsf,
    ...(e.venc === 'libvvenc' ? ['-strict', 'experimental'] : []),
    ...frag,
    target,
  ];
  return {
    outPix,
    qargs,
    prof,
    color,
    vf,
    tq,
    maps,
    stage2Maps,
    in2,
    target,
    frag,
    cmd,
    notes,
    crop,
    hostVf,
    hostSws: hostVf ? SWS_ACCURATE : '',
  };
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

/** Everything the finish reads, captured by the orchestrator at call time.
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
  hlg?: boolean; // the video is HLG (Dolby Vision 8.4): no HDR10 static boxes
  resumeDvhpNote: string;
  fragCopy: boolean;
  fragFlags: string[];
  stage2Maps: string[];
  cropped: boolean;
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
    /* a file that cannot be removed stays */
  }
}

/** Finish the container. */
export function finalizeOutput(f: Finalize): void {
  const hdr10 = (target: string) =>
    writeHdr10Metadata(target, f.hdrActive && !f.hlg, f.rtx, f.nits, f.masterPrim, f.say);
  const remux = (vidSrc: string, dst: string) =>
    remuxTracks(f.ffmpeg, f.inp, f.stage2Maps, f.fragCopy, f.fragFlags, vidSrc, dst, f.say);

  const hdrMkv2stage = f.hdrActive && f.outIsMkv;
  if (!f.resumable) {
    if (!hdrMkv2stage) {
      if (f.cropped) {
        // the one-stage encode's container still names the padded size: a stream-copy remux rewrites it from the SPS
        const tmp = f.workPath.replace(/(\.[^.\\/]*)$/, '.remux$1');
        runChecked([
          f.ffmpeg,
          '-v',
          'error',
          '-y',
          '-i',
          f.workPath,
          '-map',
          '0',
          '-c',
          'copy',
          '-max_interleave_delta',
          '0',
          tmp,
        ]);
        fs.renameSync(tmp, f.workPath);
      }
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
