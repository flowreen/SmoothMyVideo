// The before/after preview: engine/preview.py's job without python. ONE
// source frame at the current spatial settings -> <out>_original.png and <out>_processed.png
// (+ <out>_nrmask.png), and preview.py's one stdout line. The frame comes from the render's own
// decode (ffmpeg at the render's pixel format, the downscale folded in exactly as the render
// folds it), the processed side runs the render's own pass chain in the native host
// (smv-live.exe --offline --no-interp, one frame, NVIDIA's order at the render's working size:
// restore -> resize to the working size -> DLSS 5 -> the final resize / RTX VSR -> RCAS ->
// TrueHDR), so the pane is the render's frame. This process only converts for display: the
// PQ tonemaps, the 1:1 resize of the original pane, the DLSS 5 change mask and the PNG files;
// that arithmetic follows preview.py's numpy float32 (Math.fround), whose comments carry the
// reasoning. Two deliberate changes: the frame is decoded by ffmpeg like the render (preview.py
// used cv2's own decode), and a 10-bit source keeps its 16-bit samples up to the display encode.
// `node dist/render/preview.js <preview.py's arguments>`; main.ts runs it under Electron's node.
import { spawn } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import * as zlib from 'zlib';
import { INFERNO } from './inferno';
import { engineDir, tool } from './native';
import { workPlan } from './plan';
import { frameCount, probe, sourceBits, vfrConform } from './probe';
import { pyFixed, pyG } from './pyfmt';

const f = Math.fround;

export interface PreviewArgs {
  input: string;
  frame: string;
  out: string;
  rtx_hdr: boolean;
  sharpen: number;
  restore: boolean;
  scale: string | null; // --scale: the render's DLSS mode or working-size share
  upscale: number;
  rtx_vsr: boolean;
  dlssnr: boolean;
  nr_structure: number;
  nr_tone: number;
  nr_style: number;
  nr_passes: number;
  nr_mask: boolean;
  hdr_color: string;
  hdr_vibrance: number;
  hdr_satboost: number;
  hdr_saturation: number;
  hdr_contrast: number;
}

/** preview.py's argparse flags. */
export function parsePreviewArgv(argv: string[]): PreviewArgs {
  const a: PreviewArgs = {
    input: '',
    frame: 'mid',
    out: '',
    rtx_hdr: false,
    sharpen: 0,
    restore: false,
    scale: null,
    upscale: 1,
    rtx_vsr: false,
    dlssnr: false,
    nr_structure: 1,
    nr_tone: 1,
    nr_style: 1,
    nr_passes: 1,
    nr_mask: false,
    hdr_color: 'vivid',
    hdr_vibrance: 0,
    hdr_satboost: 0,
    hdr_saturation: 0,
    hdr_contrast: 100,
  };
  const num = (k: string, v: string | undefined, int = false): number => {
    const x = v === undefined ? NaN : Number(v);
    if (v === undefined || v.trim() === '' || Number.isNaN(x) || (int && !Number.isInteger(x))) {
      throw new Error(`argument ${k}: invalid ${int ? 'int' : 'float'} value: '${v ?? ''}'`);
    }
    return x;
  };
  const flags: Record<string, keyof PreviewArgs> = {
    '--rtx-hdr': 'rtx_hdr',
    '--restore': 'restore',
    '--rtx-vsr': 'rtx_vsr',
    '--dlssnr': 'dlssnr',
    '--nr-mask': 'nr_mask',
  };
  const floats: Record<string, keyof PreviewArgs> = {
    '--sharpen': 'sharpen',
    '--upscale': 'upscale',
    '--nr-structure': 'nr_structure',
    '--nr-tone': 'nr_tone',
    '--hdr-vibrance': 'hdr_vibrance',
    '--hdr-satboost': 'hdr_satboost',
  };
  const ints: Record<string, keyof PreviewArgs> = {
    '--nr-style': 'nr_style',
    '--nr-passes': 'nr_passes',
    '--hdr-saturation': 'hdr_saturation',
    '--hdr-contrast': 'hdr_contrast',
  };
  const rec = a as unknown as Record<string, unknown>;
  const pos: string[] = [];
  for (let i = 0; i < argv.length; i++) {
    const k = argv[i];
    if (k in flags) rec[flags[k]] = true;
    else if (k in floats) rec[floats[k]] = num(k, argv[++i]);
    else if (k in ints) rec[ints[k]] = num(k, argv[++i], true);
    else if (k === '--frame' || k === '--out' || k === '--hdr-color' || k === '--scale') {
      const v = argv[++i];
      if (v === undefined) throw new Error(`argument ${k}: expected one argument`);
      if (k === '--frame') a.frame = v;
      else if (k === '--out') a.out = v;
      else if (k === '--scale') a.scale = v;
      else a.hdr_color = v;
    } else if (k.startsWith('-') && k.length > 1) throw new Error(`unrecognized arguments: ${k}`);
    else pos.push(k);
  }
  if (pos.length !== 1)
    throw new Error(
      pos.length ? `unrecognized arguments: ${pos.slice(1).join(' ')}` : 'the following arguments are required: input',
    );
  a.input = pos[0];
  if (!a.out) throw new Error('the following arguments are required: --out');
  if (![0, 1, 2].includes(a.nr_style))
    throw new Error(`argument --nr-style: invalid choice: ${a.nr_style} (choose from 0, 1, 2)`);
  if (!(a.nr_passes >= 1 && a.nr_passes <= 10))
    throw new Error(`argument --nr-passes: invalid choice: ${a.nr_passes} (choose from 1 to 10)`);
  if (!['vivid', 'rtx', 'raw'].includes(a.hdr_color)) {
    throw new Error(`argument --hdr-color: invalid choice: '${a.hdr_color}' (choose from 'vivid', 'rtx', 'raw')`);
  }
  return a;
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

function isFile(p: string): boolean {
  try {
    return fs.statSync(p).isFile();
  } catch {
    return false;
  }
}

/** Run a child with `input` on its stdin; its whole stdout, its stderr text and the exit code. */
function run(
  exe: string,
  args: string[],
  input: Buffer | null,
  cwd: string,
): Promise<{ rc: number; out: Buffer; err: string }> {
  return new Promise((res) => {
    const p = spawn(exe, args, { cwd, windowsHide: true, stdio: [input ? 'pipe' : 'ignore', 'pipe', 'pipe'] });
    const out: Buffer[] = [],
      err: Buffer[] = [];
    p.stdout!.on('data', (b: Buffer) => out.push(b));
    p.stderr!.on('data', (b: Buffer) => err.push(b));
    if (input) {
      p.stdin!.on('error', () => {
        /* the child left early: its exit code says why */
      });
      p.stdin!.end(input);
    }
    p.once('error', (e) => res({ rc: -1, out: Buffer.alloc(0), err: e.message }));
    p.once('close', (c: number | null) =>
      res({ rc: c ?? 1, out: Buffer.concat(out), err: Buffer.concat(err).toString('utf8') }),
    );
  });
}

// ---- display arithmetic (preview.py, numpy float32) -----------------------------------------
const M2020_709 = [
  [1.6605, -0.5876, -0.0728],
  [-0.1246, 1.1329, -0.0083],
  [-0.0182, -0.1006, 1.1187],
].map((r) => r.map(f));
const PQ_M1 = 0.1593017578125,
  PQ_M2 = 78.84375,
  PQ_C1 = 0.8359375,
  PQ_C2 = 18.8515625,
  PQ_C3 = 18.6875;
const L0 = f(0.2126),
  L1 = f(0.7152),
  L2 = f(0.0722);

/** SMPTE ST 2084 EOTF on one float32 code value in [0,1] (display-linear, 1.0 == 10000 nits). */
function pqToLinear(e: number): number {
  const ep = f(Math.pow(clamp(e, 0, 1), f(1.0 / PQ_M2)));
  const a = Math.max(f(ep - f(PQ_C1)), 0),
    b = Math.max(f(f(PQ_C2) - f(f(PQ_C3) * ep)), f(1e-6));
  return f(Math.pow(f(a / b), f(1.0 / PQ_M1)));
}

/** A code -> value table over `levels` codes (every PQ / sRGB decode here reads a quantised code). */
function table(levels: number, fn: (i: number) => number): Float32Array {
  const t = new Float32Array(levels);
  for (let i = 0; i < levels; i++) t[i] = fn(i);
  return t;
}

/** numpy's float32 median (the mean of the two middle values for an even count), by selection. */
function median(v: Float32Array, n: number): number {
  const a = v.subarray(0, n);
  const k = n >> 1;
  select(a, k);
  if (n & 1) return a[k];
  let lo = -Infinity;
  for (let i = 0; i < k; i++) if (a[i] > lo) lo = a[i];
  return f(f(lo + a[k]) / 2);
}

/** Hoare quickselect: a[k] = the k-th smallest, everything left of it no larger. */
function select(a: Float32Array, k: number): void {
  let l = 0,
    r = a.length - 1;
  while (l < r) {
    const x = a[(l + r) >> 1];
    let i = l,
      j = r;
    while (i <= j) {
      while (a[i] < x) i++;
      while (a[j] > x) j--;
      if (i <= j) {
        const t = a[i];
        a[i] = a[j];
        a[j] = t;
        i++;
        j--;
      }
    }
    if (k <= j) r = j;
    else if (k >= i) l = i;
    else return;
  }
}

/** linear BT.2020 (3 planes interleaved) -> linear BT.709 clipped at 0, in place, and its luminance. */
function to709(lin: Float32Array, lum: Float32Array): void {
  const [m0, m1, m2] = M2020_709;
  for (let p = 0, o = 0; o < lin.length; p++, o += 3) {
    const r = lin[o],
      g = lin[o + 1],
      b = lin[o + 2];
    const x0 = Math.max(f(f(f(r * m0[0]) + f(g * m0[1])) + f(b * m0[2])), 0);
    const x1 = Math.max(f(f(f(r * m1[0]) + f(g * m1[1])) + f(b * m1[2])), 0);
    const x2 = Math.max(f(f(f(r * m2[0]) + f(g * m2[1])) + f(b * m2[2])), 0);
    lin[o] = x0;
    lin[o + 1] = x1;
    lin[o + 2] = x2;
    lum[p] = f(f(f(L0 * x0) + f(L1 * x1)) + f(L2 * x2));
  }
}

/** _shoulder_srgb: exposure k applied, the Reinhard shoulder above the knee, sRGB uint8. */
function shoulderSrgb(lin: Float32Array, k: number): Buffer {
  const out = Buffer.alloc(lin.length),
    kf = f(k),
    inv24 = f(1 / 2.4);
  for (let i = 0; i < lin.length; i++) {
    const x = f(lin[i] * kf);
    let y = x;
    if (!(x <= 0.75)) {
      const t = f(f(x - 0.75) / 0.25);
      y = f(0.75 + f(0.25 * f(t / f(1.0 + Math.abs(t)))));
    }
    const s = y <= f(0.0031308) ? f(y * f(12.92)) : f(f(f(1.055) * f(Math.pow(Math.max(y, 0), inv24))) - f(0.055));
    out[i] = Math.trunc(f(clamp(s, 0, 1) * 255));
  }
  return out;
}

/** Median of the luminances above `thr`. */
function litMedian(lum: Float32Array, thr: number): number | null {
  const v = new Float32Array(lum.length);
  let n = 0;
  for (let i = 0; i < lum.length; i++) if (lum[i] > thr) v[n++] = lum[i];
  return n ? median(v, n) : null;
}

/** A frame as display samples: `px` codes of `bits` depth (8: uint8, 16: uint16 LE), RGB. */
interface Frame {
  w: number;
  h: number;
  bits: 8 | 16;
  px: Buffer;
}

function codes(fr: Frame): { n: number; at: (i: number) => number; levels: number } {
  const n = fr.w * fr.h * 3;
  if (fr.bits === 8) return { n, at: (i) => fr.px[i], levels: 256 };
  return { n, at: (i) => fr.px.readUInt16LE(2 * i), levels: 65536 };
}

/** The source as linear sRGB luminance (the exposure anchor of _tonemap). */
function srcLum(fr: Frame): Float32Array {
  const c = codes(fr);
  const d = f(c.levels - 1);
  const lut = table(c.levels, (v) => {
    const s = f(v / d);
    return s <= f(0.04045) ? f(s / f(12.92)) : f(Math.pow(f(f(s + f(0.055)) / f(1.055)), f(2.4)));
  });
  const lum = new Float32Array(fr.w * fr.h);
  for (let p = 0, i = 0; i < c.n; p++, i += 3) {
    lum[p] = f(f(f(L0 * lut[c.at(i)]) + f(L1 * lut[c.at(i + 1)])) + f(L2 * lut[c.at(i + 2)]));
  }
  return lum;
}

/** x2rgb10le words (R in bits 29..20) as linear BT.709 and its luminance. */
function hdrLinear(words: Buffer, w: number, h: number): { lin: Float32Array; lum: Float32Array } {
  const lut = table(1024, (c) => pqToLinear(f(c / 1023.0)));
  const lin = new Float32Array(w * h * 3),
    lum = new Float32Array(w * h);
  for (let p = 0; p < w * h; p++) {
    const u = words.readUInt32LE(4 * p);
    lin[3 * p] = lut[(u >>> 20) & 1023];
    lin[3 * p + 1] = lut[(u >>> 10) & 1023];
    lin[3 * p + 2] = lut[u & 1023];
  }
  to709(lin, lum);
  return { lin, lum };
}

/** _tonemap: the TrueHDR output (x2rgb10le words) to sRGB, anchored to the source. */
function tonemapHdr(words: Buffer, w: number, h: number, src: Frame): Buffer {
  const { lin, lum } = hdrLinear(words, w, h);
  const ms = litMedian(srcLum(src), f(1e-4)),
    mh = litMedian(lum, f(1e-6));
  const k = ms !== null && mh !== null ? ms / Math.max(1e-9, mh) : 1.0;
  return shoulderSrgb(lin, k);
}

/** _tonemap_pq: an already-PQ frame to sRGB, self-anchored (median luminance at 0.2 linear). */
function tonemapPq(fr: Frame): Buffer {
  const c = codes(fr);
  const d = f(c.levels - 1);
  const lut = table(c.levels, (v) => pqToLinear(f(v / d)));
  const lin = new Float32Array(c.n),
    lum = new Float32Array(fr.w * fr.h);
  for (let i = 0; i < c.n; i++) lin[i] = lut[c.at(i)];
  to709(lin, lum);
  const m = litMedian(lum, f(1e-5));
  return shoulderSrgb(lin, m !== null ? 0.2 / Math.max(1e-9, m) : 1.0);
}

/** A frame as display uint8 (torch's clamp * 255 then round half to even for 16-bit samples). */
function toU8(fr: Frame): Buffer {
  if (fr.bits === 8) return fr.px;
  const c = codes(fr),
    out = Buffer.alloc(c.n);
  for (let i = 0; i < c.n; i++) {
    const x = f(f(c.at(i) / 65535) * 255),
      r = Math.round(x);
    out[i] = r - x === 0.5 && r & 1 ? r - 1 : r;
  }
  return out;
}

/** cv2.resize INTER_CUBIC on uint8 RGB (A = -0.75, replicated border, cv2's 11-bit fixed point). */
function resizeCubic(src: Buffer, sw: number, sh: number, dw: number, dh: number): Buffer {
  const coefs = (n: number, sn: number) => {
    const scale = 1 / (n / sn),
      ofs = new Int32Array(n * 4),
      w = new Int32Array(n * 4);
    for (let d = 0; d < n; d++) {
      let fx = f((d + 0.5) * scale - 0.5);
      const sx = Math.floor(fx);
      fx = f(fx - sx);
      const A = -0.75,
        x1 = fx + 1,
        x2 = 1 - fx;
      const c0 = f(((A * x1 - 5 * A) * x1 + 8 * A) * x1 - 4 * A);
      const c1 = f(((A + 2) * fx - (A + 3)) * fx * fx + 1);
      const c2 = f(((A + 2) * x2 - (A + 3)) * x2 * x2 + 1);
      const c3 = f(1 - c0 - c1 - c2);
      [c0, c1, c2, c3].forEach((c, k) => {
        w[4 * d + k] = clamp(Math.round(c * 2048), -32768, 32767);
        ofs[4 * d + k] = clamp(sx + k - 1, 0, sn - 1);
      });
    }
    return { ofs, w };
  };
  const hx = coefs(dw, sw),
    vy = coefs(dh, sh);
  const rows = new Int32Array(sh * dw * 3);
  for (let y = 0; y < sh; y++) {
    for (let x = 0; x < dw; x++) {
      for (let c = 0; c < 3; c++) {
        let s = 0;
        for (let k = 0; k < 4; k++) s += src[(y * sw + hx.ofs[4 * x + k]) * 3 + c] * hx.w[4 * x + k];
        rows[(y * dw + x) * 3 + c] = s;
      }
    }
  }
  const out = Buffer.alloc(dw * dh * 3);
  for (let y = 0; y < dh; y++) {
    for (let i = 0; i < dw * 3; i++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += rows[vy.ofs[4 * y + k] * dw * 3 + i] * vy.w[4 * y + k];
      out[y * dw * 3 + i] = clamp((s + (1 << 21)) >> 22, 0, 255);
    }
  }
  return out;
}

// ---- PNG ------------------------------------------------------------------------------------
const CRC = (() => {
  const t = new Int32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c;
  }
  return t;
})();

function chunk(type: string, data: Buffer): Buffer {
  const td = Buffer.concat([Buffer.from(type, 'latin1'), data]);
  let c = -1;
  for (let i = 0; i < td.length; i++) c = CRC[(c ^ td[i]) & 255] ^ (c >>> 8);
  const len = Buffer.alloc(4),
    crc = Buffer.alloc(4);
  len.writeUInt32BE(data.length);
  crc.writeUInt32BE((c ^ -1) >>> 0);
  return Buffer.concat([len, td, crc]);
}

/** An 8-bit RGB PNG of w x h interleaved samples. */
export function writePng(file: string, rgb: Buffer, w: number, h: number): void {
  const raw = Buffer.alloc((w * 3 + 1) * h);
  for (let y = 0; y < h; y++) rgb.copy(raw, y * (w * 3 + 1) + 1, y * w * 3, (y + 1) * w * 3); // filter 0 per row
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0);
  ihdr.writeUInt32BE(h, 4);
  ihdr[8] = 8;
  ihdr[9] = 2; // 8-bit truecolour, no interlace
  fs.writeFileSync(
    file,
    Buffer.concat([
      Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
      chunk('IHDR', ihdr),
      chunk('IDAT', zlib.deflateSync(raw, { level: 3 })),
      chunk('IEND', Buffer.alloc(0)),
    ]),
  );
}

/** The DLSS 5 change mask: the processed pane dimmed to grey under a heat map of the change. */
/** The render's progress thumbnail (render.py _live_preview's formatting worker): the
 * host's "SMVT" dump (smv-live.exe --thumb: uint32 w, h, fmt 0 rgb24 / 1 rgb48le / 2 x2rgb10le,
 * then the samples) to the GUI's PNG, tmp then replace. SDR as is; a PQ source carried through
 * gets the pane's self-anchored _tonemap_pq; a TrueHDR output the same self-anchored exposure
 * (deviation: python anchored it to the SDR frame before TrueHDR, which the host's writer thread
 * never holds). Never throws: a thumbnail must never break the render. */
export function thumbPng(raw: string, png: string, srcPq: boolean): void {
  try {
    const b = fs.readFileSync(raw);
    if (b.length < 16 || b.toString('latin1', 0, 4) !== 'SMVT') return;
    const w = b.readUInt32LE(4),
      h = b.readUInt32LE(8),
      fmt = b.readUInt32LE(12),
      px = b.subarray(16);
    if (px.length < w * h * (fmt === 2 ? 4 : fmt === 1 ? 6 : 3)) return;
    let rgb: Buffer;
    if (fmt === 2) {
      const { lin, lum } = hdrLinear(px, w, h);
      const m = litMedian(lum, f(1e-5));
      rgb = shoulderSrgb(lin, m !== null ? 0.2 / Math.max(1e-9, m) : 1.0);
    } else {
      const fr: Frame = { w, h, bits: fmt === 1 ? 16 : 8, px };
      rgb = srcPq ? tonemapPq(fr) : toU8(fr);
    }
    const tmp = png + '.tmp.png';
    writePng(tmp, rgb, w, h);
    fs.renameSync(tmp, png);
  } catch {
    /* except Exception: pass */
  }
}

/** The DLSS 5 change map scaled from the decoded size to the processed pane: bilinear with
 * half-pixel centres. */
function scaleMap(m: Float32Array, w: number, h: number, ow: number, oh: number): Float32Array {
  const out = new Float32Array(ow * oh);
  for (let y = 0; y < oh; y++) {
    const fy = Math.min(Math.max(((y + 0.5) * h) / oh - 0.5, 0), h - 1);
    const y0 = Math.floor(fy),
      y1 = Math.min(y0 + 1, h - 1),
      wy = fy - y0;
    for (let x = 0; x < ow; x++) {
      const fx = Math.min(Math.max(((x + 0.5) * w) / ow - 0.5, 0), w - 1);
      const x0 = Math.floor(fx),
        x1 = Math.min(x0 + 1, w - 1),
        wx = fx - x0;
      const top = m[y0 * w + x0] + (m[y0 * w + x1] - m[y0 * w + x0]) * wx;
      const bot = m[y1 * w + x0] + (m[y1 * w + x1] - m[y1 * w + x0]) * wx;
      out[y * ow + x] = top + (bot - top) * wy;
    }
  }
  return out;
}

function nrMask(delta: Float32Array, proc: Buffer): { png: Buffer; touched: number; mean: number } {
  const out = Buffer.alloc(proc.length),
    g45 = f(0.45);
  let touched = 0,
    sum = 0;
  for (let p = 0; p < delta.length; p++) {
    const d = delta[p];
    if (d > f(1.0 / 255)) touched++;
    sum += d;
    const m = clamp(f(d * 8.0), 0, 1),
      li = Math.trunc(f(m * 255)) * 3,
      a1 = f(1.0 - m);
    const grey = f(((proc[3 * p] * 4899 + proc[3 * p + 1] * 9617 + proc[3 * p + 2] * 1868 + 8192) >> 14) * g45);
    for (let c = 0; c < 3; c++) out[3 * p + c] = Math.trunc(clamp(f(f(grey * a1) + f(INFERNO[li + c] * m)), 0, 255));
  }
  return { png: out, touched: (touched / delta.length) * 100.0, mean: (sum / delta.length) * 255.0 };
}

// ---- the preview ----------------------------------------------------------------------------
/** preview.py main(): writes the PNGs and resolves to its stdout line. */
export async function preview(a: PreviewArgs, env: NodeJS.ProcessEnv = process.env): Promise<string> {
  const ENGINE = engineDir();
  const FFMPEG = tool(ENGINE, 'ffmpeg'),
    FFPROBE = tool(ENGINE, 'ffprobe');
  const inp = path.win32.resolve(a.input);
  const strength = clamp(a.sharpen, 0.0, 1.0);
  const pr = probe(FFPROBE, inp);
  const vfr = vfrConform(pr.st, pr.num, pr.den);
  const n = frameCount(FFPROBE, inp, pr.nb, vfr.num / vfr.den);
  let idx: number;
  if (a.frame === 'mid') idx = Math.max(0, Math.floor(n / 2));
  else if (/^\s*[+-]?\d+\s*$/.test(a.frame)) idx = Math.max(0, Math.min(n - 1, parseInt(a.frame, 10)));
  else throw new Error(`invalid literal for int() with base 10: '${a.frame}'`);
  const transfer = String(pr.st.color_transfer ?? '').toLowerCase();
  const srcHdr = transfer === 'smpte2084' || transfer === 'arib-std-b67';
  let doHdr = a.rtx_hdr && !srcHdr;
  const up = a.upscale <= 0 ? 1.0 : clamp(a.upscale, 1.0 / 16, 16.0);
  const plan = workPlan(pr.st, pr.w, pr.h, up, a.scale);
  if (typeof plan === 'string') throw new Error(plan);
  const W = plan.w,
    H = plan.h;
  const [OW, OH] = [plan.outW, plan.outH];
  const [WW, WH] = [plan.workW, plan.workH];
  if (doHdr && (OW > 8192 || OH > 8192)) doHdr = false;
  const bits: 8 | 16 = sourceBits(pr.st, String(pr.st.pix_fmt || 'yuv420p')) >= 10 ? 16 : 8;
  const DEC_FMT = bits === 16 ? 'rgb48le' : 'rgb24',
    bpp = bits === 16 ? 6 : 3;

  // the frame: the render's decode (its pixel format and downscale fold), seeked to idx; the
  // original pane decodes without the fold (vf empty, the source size)
  const decode = async (at: number, vf: string[] = plan.vf, dw: number = W, dh: number = H) => {
    const ss = at > 0 ? ['-ss', String(Math.max(0, ((at - 0.5) * vfr.den) / vfr.num))] : [];
    const r = await run(
      FFMPEG,
      [
        '-v',
        'error',
        ...ss,
        '-i',
        inp,
        '-an',
        '-sn',
        '-dn',
        ...(vf.length ? ['-vf', vf.join(',')] : []),
        '-frames:v',
        '1',
        '-f',
        'rawvideo',
        '-pix_fmt',
        DEC_FMT,
        '-',
      ],
      null,
      ENGINE,
    );
    return r.out.length >= dw * dh * bpp ? r.out.subarray(0, dw * dh * bpp) : null;
  };
  let at = idx;
  let px = await decode(at);
  if (!px && idx > 0) px = await decode((at = 0)); // a missed seek: frame 0, like preview.py
  if (!px) throw new Error('could not read a frame from ' + a.input);
  const orig: Frame = { w: W, h: H, bits, px };
  // the before pane: the source itself when the working size folds the decode
  let before: Frame = orig;
  if (plan.vf.length) {
    const full = await decode(at, [], pr.w, pr.h);
    if (full) before = { w: pr.w, h: pr.h, bits, px: full };
  }

  let vsrUsed = false,
    restoreUsed = false,
    nrUsed = false;
  let proc: Frame | null = null,
    hdrWords: Buffer | null = null,
    delta: Float32Array | null = null;
  let needVsr = OW > W && OH > H && a.rtx_vsr;
  const rtxDir = env.SMV_RTXVIDEO_DIR || path.join(ENGINE, 'rtxvideo');
  if ((needVsr || doHdr) && !['rtxvideo_cuda.dll', 'nvngx_truehdr.dll'].every((d) => isFile(path.join(rtxDir, d)))) {
    needVsr = false; // like preview.py: no RTX Video bridge means bicubic + SDR
    doHdr = false;
  }
  const nrDir = env.SMV_DLSSNR_DIR || path.join(ENGINE, 'dlssnr');
  const nrOn = a.dlssnr && ['nvngx.dll', 'nvngx_dlssnr.dll'].every((d) => isFile(path.join(nrDir, d)));
  if (strength > 0 || a.restore || OW !== W || OH !== H || doHdr || nrOn) {
    const cacheDir = env.SMV_TRT_CACHE || path.join(ENGINE, 'trt_cache_safe_to_delete');
    fs.mkdirSync(cacheDir, { recursive: true });
    const outFmt = doHdr ? 'x2rgb10le' : DEC_FMT;
    const args = [
      '--offline',
      '--w',
      String(W),
      '--h',
      String(H),
      '--frames',
      '1',
      '--no-interp',
      '--pixfmt',
      DEC_FMT,
      '--out-pixfmt',
      outFmt,
      '--script',
      path.join(ENGINE, 'live_server.py'),
      '--cache',
      cacheDir,
    ];
    if (OW !== W || OH !== H) {
      args.push('--out-w', String(OW), '--out-h', String(OH));
      if (needVsr) args.push('--rtx-vsr');
    }
    if (strength > 0) args.push('--sharpen', pyG(strength));
    if (a.restore) args.push('--restore');
    if (WW !== W || WH !== H || a.restore) args.push('--work-w', String(WW), '--work-h', String(WH));
    const deltaFile = path.resolve(a.out) + '_nrdelta.f32';
    if (nrOn) {
      args.push(
        '--dlssnr',
        '--nr-structure',
        pyG(clamp(a.nr_structure, 0.0, 2.0)),
        '--nr-tone',
        pyG(clamp(a.nr_tone, 0.0, 2.0)),
        '--nr-style',
        String(a.nr_style),
      );
      if (a.nr_passes > 1) args.push('--nr-passes', String(a.nr_passes));
      if (a.nr_mask) {
        try {
          fs.unlinkSync(deltaFile);
        } catch {
          /* not there */
        }
        args.push('--nr-delta', deltaFile);
      }
    }
    if (doHdr) {
      args.push(
        '--rtx-hdr',
        '--hdr-color',
        a.hdr_color,
        '--hdr-contrast',
        String(clamp(a.hdr_contrast, 0, 200)),
        '--hdr-saturation',
        String(clamp(a.hdr_saturation, 0, 200)),
        '--hdr-vibrance',
        pyG(clamp(a.hdr_vibrance, 0.0, 1.0)),
        '--hdr-satboost',
        pyG(clamp(a.hdr_satboost, 0.0, 1.0)),
      );
    }
    const r = await run(path.join(ENGINE, 'live', 'smv-live.exe'), args, px, path.join(ENGINE, 'GMFSS_Fortuna'));
    const outBytes = OW * OH * (doHdr ? 4 : bpp);
    if (r.rc !== 0 || r.out.length < outBytes) {
      throw new Error(`the native host failed (exit ${r.rc}): ` + r.err.trim().split(/\r?\n/).slice(-4).join(' | '));
    }
    if (doHdr) hdrWords = r.out.subarray(0, outBytes);
    else proc = { w: OW, h: OH, bits, px: r.out.subarray(0, outBytes) };
    vsrUsed = needVsr && !/RTX VSR (unavailable|skipped|demoted)|VSR run failed|VSR eval failed/.test(r.err);
    restoreUsed = a.restore && !/\[restore\] unavailable|restore failed/.test(r.err);
    nrUsed = nrOn && /DLSS 5 Neural Rendering ready/.test(r.err) && !/\[dlss5\]/.test(r.err);
    if (nrUsed && a.nr_mask) {
      try {
        const b = fs.readFileSync(deltaFile);
        fs.unlinkSync(deltaFile);
        // DLSS 5 runs at the working size (the map is WW x WH)
        if (b.length === WW * WH * 4) {
          const m = new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.length));
          delta = WW === OW && WH === OH ? m : scaleMap(m, WW, WH, OW, OH);
        }
      } catch {
        /* no map: the pane shows the processed picture as usual */
      }
    }
  } else proc = orig; // nothing enabled: the render would leave the frame as is

  // display conversion (preview.py: a PQ source gets the self-anchored tonemap on both panes)
  const pq = transfer === 'smpte2084';
  let origDisp = pq ? tonemapPq(before) : toU8(before);
  const procDisp = hdrWords ? tonemapHdr(hdrWords, OW, OH, before) : pq ? tonemapPq(proc!) : toU8(proc!);
  if (OW !== before.w || OH !== before.h) origDisp = resizeCubic(origDisp, before.w, before.h, OW, OH); // 1:1 zoom with the processed pane
  const outDir = path.dirname(path.resolve(a.out));
  if (outDir) fs.mkdirSync(outDir, { recursive: true });
  const pOrig = a.out + '_original.png',
    pProc = a.out + '_processed.png';
  writePng(pOrig, origDisp, OW, OH);
  writePng(pProc, procDisp, OW, OH);
  let maskTag = '';
  if (delta) {
    const pMask = a.out + '_nrmask.png';
    const mk = nrMask(delta, procDisp);
    writePng(pMask, mk.png, OW, OH);
    maskTag = ` nrmask=${pyFixed(mk.touched, 1)}%/${pyFixed(mk.mean, 2)} ${pMask}`;
  }
  return (
    `preview frame ${idx}/${n} ${OW}x${OH} hdr=${+doHdr} sharpen=${pyG(strength)} srchdr=${+srcHdr} up=${pyG(up)} ` +
    `vsr=${+vsrUsed} restore=${+restoreUsed} nr=${+nrUsed} -> ${pOrig} | ${pProc}${maskTag}\n`
  );
}

if (require.main === module) {
  let a: PreviewArgs;
  try {
    a = parsePreviewArgv(process.argv.slice(2));
  } catch (e) {
    process.stderr.write(`preview: error: ${(e as Error).message}\n`);
    process.exit(2);
  }
  preview(a).then(
    (line) => {
      process.stdout.write(line);
    },
    (e: Error) => {
      process.stderr.write(`ERROR: ${e.message}\n`);
      process.exitCode = 1;
    },
  );
}
