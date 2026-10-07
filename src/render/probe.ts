// The source probe: engine/render_probe.py ported line by line, the
// comments there carry the reasoning. Pure functions over the ffprobe stream record or the
// input path; the callers write every stderr line themselves. ffprobe runs synchronously, as
// in python (subprocess.check_output), with no console window.
import { execFileSync } from 'child_process';
import { pyG, pyRound } from './pyfmt';

/** One ffprobe stream record (the fields render.py reads; values as ffprobe's JSON has them). */
export type Stream = Record<string, string | number | undefined>;

export interface Probe {
  w: number;
  h: number;
  num: number;
  den: number;
  nb: number;
  st: Stream;
}

function ffprobeText(ffprobe: string, args: string[]): string {
  return execFileSync(ffprobe, args, { encoding: 'utf8', windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] });
}

/** python int() of a digit string ("" and junk are errors in python; callers guard as it did). */
function toInt(s: string): number {
  if (!/^\s*[+-]?\d+\s*$/.test(s)) throw new Error(`invalid literal for int(): '${s}'`);
  return parseInt(s, 10);
}

/** (str(v or dflt).split("/") + ["1"])[:2] */
function splitRate(v: unknown, dflt: string): [string, string] {
  const parts = String(v || dflt)
    .split('/')
    .concat(['1']);
  return [parts[0], parts[1]];
}

/** The first video stream: w, h, rate num / den, nb_frames or 0, the stream record. */
export function probe(ffprobe: string, path: string): Probe {
  const out = ffprobeText(ffprobe, [
    '-v',
    'error',
    '-select_streams',
    'v:0',
    '-show_entries',
    'stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,codec_name,pix_fmt,' +
      'bits_per_raw_sample,color_space,color_transfer,color_primaries,color_range,sample_aspect_ratio',
    '-of',
    'json',
    path,
  ]);
  const st: Stream = (JSON.parse(out).streams || [{}])[0];
  const w = toInt(String(st.width)),
    h = toInt(String(st.height));
  const [num, den] = splitRate(st.r_frame_rate, '0/1');
  const nb = /^\d+$/.test(String(st.nb_frames || '')) ? toInt(String(st.nb_frames)) : 0;
  return { w, h, num: toInt(num), den: toInt(den || '1'), nb, st };
}

/** A colour / format tag of the stream, null when absent or a placeholder. */
export function tag(st: Stream, key: string): string | number | null {
  const v = st[key];
  return v && !['unknown', 'reserved', 'N/A'].includes(String(v)) ? v : null;
}

/** The shape of the source's pixels, reduced: ffprobe's sample aspect ratio (the container's, else the bitstream's),
 * [1, 1] when square, absent or invalid. An anamorphic source (a DVD's 720x480 at 32:27 or 8:9, 1440x1080 at 4:3)
 * stores its picture squeezed and the player stretches it by this ratio. */
export function sourceSar(st: Stream): [number, number] {
  const m = /^(\d+):(\d+)$/.exec(String(st.sample_aspect_ratio || ''));
  const [n, d] = m ? [parseInt(m[1], 10), parseInt(m[2], 10)] : [0, 0];
  if (!(n > 0 && d > 0)) return [1, 1];
  const gcd = (a: number, b: number): number => (b ? gcd(b, a % b) : a);
  const g = gcd(n, d);
  return [n / g, d / g];
}

/** Bit depth of the source samples: the probe's own field, else read off the pixel format. */
export function sourceBits(st: Stream, srcPix: string): number {
  const bpr = st.bits_per_raw_sample;
  if (bpr && /^\d+$/.test(String(bpr))) return toInt(String(bpr));
  if (['p10', '10le', '10be'].some((s) => srcPix.includes(s))) return 10;
  if (['p12', '12le', '12be'].some((s) => srcPix.includes(s))) return 12;
  if (['p16', '16le', '16be'].some((s) => srcPix.includes(s))) return 16;
  return 8;
}

/** nb when the container reports it, else an estimate from the container duration. */
export function frameCount(ffprobe: string, path: string, nb: number, srcFps: number): number {
  if (nb) return nb;
  try {
    const s = ffprobeText(ffprobe, ['-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', path]).trim();
    const dur = Number(s);
    if (s === '' || Number.isNaN(dur)) return 0; // python float() raises ValueError
    return Math.max(0, pyRound(dur * srcFps));
  } catch {
    return 0;
  }
}

export interface Vfr {
  num: number;
  den: number;
  flags: string[];
  note: string | null;
}

/** The average rate and `-fps_mode cfr` when the stream is VFR, else the container rate unchanged. */
export function vfrConform(st: Stream, num: number, den: number): Vfr {
  const [as, ds] = splitRate(st.avg_frame_rate, '0/1');
  const an = toInt(as || '0'),
    ad = toInt(ds || '1');
  if (an > 0 && ad > 0 && num > 0 && Math.abs(an / ad - num / den) / (num / den) > 0.005) {
    const note =
      `VFR source: container rate ${num}/${den} (${pyG(num / den)} fps) but the stream ` +
      `averages ${an}/${ad} (${pyG(an / ad)} fps); decoding at the constant average ` +
      'rate so duration and audio sync are preserved\n';
    return { num: an, den: ad, flags: ['-fps_mode', 'cfr', '-r', `${an}/${ad}`], note };
  }
  return { num, den, flags: [], note: null };
}

export interface OutRate {
  rateStr: string;
  outLabel: number;
  ratio: number | null;
  tgtNum: number | null;
  tgtDen: number | null;
}

/** The encoder rate string, the fps label and, in --fps mode, the ratio and the snapped target
 * rational. fps = the --fps target, or null when the render is not in --fps mode. */
export function outputRate(multi: number, fps: number | null, noInterp: boolean, num: number, den: number): OutRate {
  const srcFps = num / den;
  let ratio: number | null = null,
    tgtNum: number | null = null,
    tgtDen: number | null = null;
  let rateStr: string, outLabel: number;
  if (noInterp) {
    rateStr = `${num}/${den}`;
    outLabel = pyRound(srcFps);
  } else if (fps !== null) {
    if (Math.abs(fps - pyRound(fps)) < 0.005) {
      tgtNum = pyRound(fps);
      tgtDen = 1;
    } else if (Math.abs((fps * 1001) / 1000 - pyRound((fps * 1001) / 1000)) < 0.005) {
      tgtNum = pyRound((fps * 1001) / 1000) * 1000;
      tgtDen = 1001;
    }
    if (tgtNum) {
      ratio = (tgtNum * den) / ((tgtDen as number) * num);
      rateStr = `${tgtNum}/${tgtDen}`;
      outLabel = pyRound(tgtNum / (tgtDen as number));
    } else {
      ratio = fps / srcFps;
      rateStr = pyG(fps);
      outLabel = pyRound(fps);
    }
  } else {
    rateStr = `${num * multi}/${den}`;
    outLabel = pyRound(srcFps * multi);
  }
  return { rateStr, outLabel, ratio, tgtNum, tgtDen };
}

// track passthrough: the container rule sits on render_probe.py's comment block
export const MP4_AUDIO_OK = new Set(['aac', 'ac3', 'eac3', 'mp3', 'alac', 'opus', 'flac']);
export const MKV_SUB_COPY_OK = new Set(['ass', 'ssa', 'subrip', 'srt', 'hdmv_pgs_subtitle', 'dvd_subtitle', 'webvtt']);

export type Track = [number | null, string]; // (absolute index, codec)

/** The audio and subtitle streams and whether attachments exist; a probe failure = none. */
export function probeTracks(ffprobe: string, path: string): { aud: Track[]; sub: Track[]; hasAttach: boolean } {
  const aud: Track[] = [],
    sub: Track[] = [];
  let hasAttach = false;
  try {
    const ts =
      JSON.parse(
        ffprobeText(ffprobe, [
          '-v',
          'error',
          '-show_entries',
          'stream=index,codec_type,codec_name',
          '-of',
          'json',
          path,
        ]),
      ).streams || [];
    for (const s of ts) {
      const ty = s.codec_type || '',
        c = String(s.codec_name || '').toLowerCase();
      if (ty === 'audio') aud.push([s.index ?? null, c]);
      else if (ty === 'subtitle') sub.push([s.index ?? null, c]);
      else if (ty === 'attachment') hasAttach = true;
    }
  } catch {
    /* probe failure: no extra tracks */
  }
  return { aud, sub, hasAttach };
}

/** True when the default output must be .mkv (subtitles or mp4-incompatible audio). */
export function needMkv(aud: Track[], sub: Track[]): boolean {
  return sub.length > 0 || aud.some(([, c]) => !MP4_AUDIO_OK.has(c));
}

const UNTAGGED = ['', 'unknown', 'unspecified'];

// The colour conversions run through zimg (zscale): exact matrix and range math and Lanczos3 chroma, where
// swscale's default conversions read an 8-bit source 0.9 to 1.5 codes dark, a 10-bit one 0.4, and write the
// encoder's 10-bit luma 1.2 codes bright. swscale keeps what zimg does not take (RGB, palette and gray sources,
// alpha, semi-planar formats, other matrices), with its accurate rounding.
export const SWS_ACCURATE = 'accurate_rnd+full_chroma_int+full_chroma_inp+bitexact';
/** zimg's names of the YUV matrices both conversions take. */
export const ZSC_MATRIX: Record<string, string> = {
  bt709: '709',
  smpte170m: '170m',
  bt470bg: '470bg',
  bt2020nc: '2020_ncl',
};

/** The matrix of an untagged w x h picture, as players assume it: BT.709 from HD up, BT.601 below. */
export function sizeMatrix(w: number, h: number): string {
  return Math.min(w, h) >= 600 || Math.max(w, h) >= 1024 ? 'bt709' : 'smpte170m';
}

/** The planar RGB format the decode's zimg conversion ends in (the pack to rgb24 / rgb48le follows). */
function planarRgb(st: Stream): string {
  return sourceBits(st, String(st.pix_fmt || 'yuv420p')) >= 10 ? 'gbrp16le' : 'gbrp';
}

/** [horizontal, vertical] chroma subsampling of a YUV pixel format (yuv420p10le -> [2, 2]); an unknown one = 4:2:0. */
export function chromaSub(pix: string): [number, number] {
  const m = /^yuvj?4([0-4])([0-4])p/.exec(pix);
  if (m)
    return (
      (
        { '44': [1, 1], '22': [2, 1], '20': [2, 2], '11': [4, 1], '10': [4, 4], '40': [1, 2] } as Record<
          string,
          [number, number]
        >
      )[m[1] + m[2]] ?? [2, 2]
    );
  if (/^(nv24|nv42|p41[026])/.test(pix)) return [1, 1];
  if (/^(nv16|nv20|p21[026]|yuyv422|uyvy422|yvyu422|y21[02])/.test(pix)) return [2, 1];
  return [2, 2];
}

/** [the pad before a zimg conversion, the crop after it], or null when zimg cannot take the frame: it takes only
 * sizes the chroma subsampling divides. Missing rows are padded and the RGB cropped back, bit for bit the taller
 * frame's conversion (the padding's luma reaches no visible pixel, the last chroma row stays whole). A missing column
 * cannot be: ffmpeg's pad (and hstack) overwrite the chroma column the last pixel shares with the padding, so that
 * frame goes to swscale (null). Both empty when the size divides. */
export function zimgPadCrop(pix: string, w: number, h: number): [string[], string[]] | null {
  const [sx, sy] = chromaSub(pix);
  if (w % sx) return null;
  const ph = Math.ceil(h / sy) * sy;
  return ph === h ? [[], []] : [[`pad=${w}:${ph}:0:0`], [`crop=${w}:${h}:0:0`]];
}

/** The -vf chain of a decode at the source size: planar YUV to RGB (`fmt`, the pipe's planar format by default)
 * through zimg. Empty = swscale converts (a source that is not planar YUV without alpha, or a matrix outside
 * ZSC_MATRIX). */
export function decodeRgbVf(st: Stream, w: number, h: number, fmt = planarRgb(st)): string[] {
  const pix = String(st.pix_fmt || '');
  if (!/^yuvj?4[0-4]{2}p(\d+(le|be))?$/.test(pix)) return [];
  const sp: string[] = [];
  const cs = String(st.color_space || '');
  if (UNTAGGED.includes(cs)) sp.push('colorspace=' + sizeMatrix(w, h));
  else if (!Object.prototype.hasOwnProperty.call(ZSC_MATRIX, cs)) return [];
  if (UNTAGGED.includes(String(st.color_range || ''))) sp.push('range=tv');
  const pc = zimgPadCrop(pix, w, h);
  if (!pc) return [];
  const [pad, crop] = pc;
  return (sp.length ? [`setparams=${sp.join(':')}`] : []).concat(
    pad,
    ['zscale=filter=lanczos:dither=none', `format=${fmt}`],
    crop,
  );
}

/** The -vf chain for a decode-side downscale of a w x h source to dw x dh: the decode's zimg conversion to float
 * planes, then zimg's Lanczos at radius 7 on the source's own codes (the host's shrink, kLzDown), rounded by zimg to
 * the pipe's integers (swscale's pack of float planes is 0.36 codes bright at 8 bits). In linear light, radius 3, 4
 * and 6 measured 1.5 to 1.7 dB under it through SMV's enlarge back. What zimg cannot read: swscale's Lanczos 7. */
export function dscaleVf(st: Stream, w: number, h: number, dw: number, dh: number): string[] {
  const conv = decodeRgbVf(st, w, h, 'gbrpf32le');
  if (!conv.length) return [`scale=${dw}:${dh}:flags=lanczos+accurate_rnd+full_chroma_int+full_chroma_inp:param0=7`];
  return conv.concat([`zscale=w=${dw}:h=${dh}:filter=lanczos:param_a=7:dither=none`, `format=${planarRgb(st)}`]);
}
