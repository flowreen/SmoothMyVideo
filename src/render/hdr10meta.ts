// HDR10 static metadata (mdcv + clli) and the Dolby Vision configuration box (dvvC) injected
// into an MP4's video sample entry (priority 24 step 6b): engine/hdr10_meta.py ported line by
// line, the module docstring there carries the reasoning (why the LGPL ffmpeg cannot write
// these, the box layouts, the moov-tail-only rewrite, idempotence). All fields big-endian.
import * as fs from 'fs';
import { pyRound } from './pyfmt';

type Pair = [number, number];
type Gbr = [Pair, Pair, Pair];

// mastering-display gamuts in mdcv's 0.00002 units, box order Green, Blue, Red
export const P3_GBR: Gbr = [
  [13250, 34500],
  [7500, 3000],
  [34000, 16000],
];
export const BT2020_GBR: Gbr = [
  [8500, 39850],
  [6550, 2300],
  [35400, 14600],
];
export const BT709_GBR: Gbr = [
  [15000, 30000],
  [7500, 3000],
  [32000, 16500],
];
export const D65: Pair = [15635, 16450];
export const DCI: Pair = [15700, 17550];

export const MASTERING_COLORSPACES: Record<string, [Gbr, Pair]> = {
  'display-p3': [P3_GBR, D65],
  'dci-p3': [P3_GBR, DCI],
  bt2020: [BT2020_GBR, D65],
  bt709: [BT709_GBR, D65],
};
export const DEFAULT_COLORSPACE = 'display-p3';

type Box = [string, number, number, number]; // (type, box_start, payload_start, box_end)

function u32(buf: Buffer, off: number): number {
  return buf.readUInt32BE(off);
}
function u64(buf: Buffer, off: number): bigint {
  return buf.readBigUInt64BE(off);
}

/** (type, box_start, payload_start, box_end) for each box in [start, end). */
function* iterBoxes(buf: Buffer, start: number, end: number): Generator<Box> {
  let off = start;
  while (off + 8 <= end) {
    let size = u32(buf, off);
    const typ = buf.toString('latin1', off + 4, off + 8);
    let hdr = 8;
    if (size === 1) {
      // 64-bit largesize
      size = Number(u64(buf, off + 8));
      hdr = 16;
    } else if (size === 0) {
      // extends to the end of the parent
      size = end - off;
    }
    if (size < hdr || off + size > end) return;
    yield [typ, off, off + hdr, off + size];
    off += size;
  }
}

function find(buf: Buffer, start: number, end: number, typ: string): [number, number, number] | null {
  for (const [t, bs, ps, be] of iterBoxes(buf, start, end)) if (t === typ) return [bs, ps, be];
  return null;
}

/** (base, buf): the file from the first top-level moov box to the end, and that box's absolute
 * offset; null without a moov. Reads only box headers until moov (see hdr10_meta._read_tail). */
function readTail(path: string): [number, Buffer] | null {
  const fd = fs.openSync(path, 'r');
  try {
    const size = fs.fstatSync(fd).size;
    const hdr = Buffer.alloc(16);
    let off = 0,
      base: number | null = null;
    while (off + 8 <= size) {
      const got = fs.readSync(fd, hdr, 0, 16, off);
      if (got < 8) break;
      let bsize = u32(hdr, 0);
      const typ = hdr.toString('latin1', 4, 8);
      let hlen = 8;
      if (bsize === 1) {
        // 64-bit largesize
        if (got < 16) break;
        bsize = Number(u64(hdr, 8));
        hlen = 16;
      } else if (bsize === 0) {
        // extends to the end of the file
        bsize = size - off;
      }
      if (bsize < hlen || off + bsize > size) break;
      if (typ === 'moov') {
        base = off;
        break;
      }
      off += bsize;
    }
    if (base === null) return null;
    const buf = Buffer.alloc(size - base);
    let pos = 0;
    while (pos < buf.length) {
      const n = fs.readSync(fd, buf, pos, buf.length - pos, base + pos);
      if (n <= 0) break;
      pos += n;
    }
    return [base, buf.subarray(0, pos)];
  } finally {
    fs.closeSync(fd);
  }
}

/** Add delta to the 32-bit (or 64-bit largesize) size field of the box at boxStart. */
function grow(buf: Buffer, boxStart: number, delta: number): void {
  const size = u32(buf, boxStart);
  if (size === 1) packQ(buf, u64(buf, boxStart + 8) + BigInt(delta), boxStart + 8);
  else packU(buf, size + delta, boxStart, 4);
}

/** The trak whose handler is 'vide'. */
function videoTrak(buf: Buffer, moovPs: number, moovBe: number): [number, number, number] | null {
  for (const [t, bs, ps, be] of iterBoxes(buf, moovPs, moovBe)) {
    if (t !== 'trak') continue;
    const mdia = find(buf, ps, be, 'mdia');
    if (!mdia) continue;
    const hdlr = find(buf, mdia[1], mdia[2], 'hdlr');
    if (hdlr && buf.toString('latin1', hdlr[1] + 8, hdlr[1] + 12) === 'vide') return [bs, ps, be];
  }
  return null;
}

/** python struct ">H" / ">I" refuse out-of-range values; so does this, with struct.error's text. */
function packU(buf: Buffer, v: number, off: number, bytes: 2 | 4): void {
  if (!Number.isInteger(v) || v < 0 || v >= 2 ** (8 * bytes)) {
    throw new Error(
      bytes === 2 ? "'H' format requires 0 <= number <= 65535" : "'I' format requires 0 <= number <= 4294967295",
    );
  }
  if (bytes === 2) buf.writeUInt16BE(v, off);
  else buf.writeUInt32BE(v, off);
}

function packQ(buf: Buffer, v: bigint, off: number): void {
  if (v < 0n || v >= 2n ** 64n) throw new Error("'Q' format requires 0 <= number <= 18446744073709551615");
  buf.writeBigUInt64BE(v, off);
}

function mdcvClli(
  primaries: Gbr,
  white: Pair,
  maxNits: number,
  minNits: number,
  maxcll: number,
  maxfall: number,
): Buffer {
  const [g, b, r] = primaries;
  const mdcv = Buffer.alloc(8 + 24),
    clli = Buffer.alloc(8 + 4);
  mdcv.writeUInt32BE(mdcv.length, 0);
  mdcv.write('mdcv', 4, 'latin1');
  [g[0], g[1], b[0], b[1], r[0], r[1], white[0], white[1]].forEach((v, i) => packU(mdcv, v, 8 + 2 * i, 2));
  packU(mdcv, pyRound(maxNits * 10000), 24, 4);
  packU(mdcv, pyRound(minNits * 10000), 28, 4);
  clli.writeUInt32BE(clli.length, 0);
  clli.write('clli', 4, 'latin1');
  clli.writeUInt16BE(Math.trunc(maxcll) & 0xffff, 8);
  clli.writeUInt16BE(Math.trunc(maxfall) & 0xffff, 10);
  return Buffer.concat([mdcv, clli]);
}

/** The shared box surgery (hdr10_meta._insert_into_sample_entry): append buildBox() to the video
 * sample entry in place, patch every stco / co64 offset past the insert, grow the ancestors up to
 * moov, rewrite the tail from moov on. False when skipped (present already, or not the expected
 * moov / trak / ... / stsd shape); throws only on real I/O errors. */
function insertIntoSampleEntry(path: string, buildBox: () => Buffer, skipTypes: string[]): boolean {
  const tail = readTail(path);
  if (!tail) return false;
  const [base, buf] = tail;
  const moov = find(buf, 0, buf.length, 'moov');
  if (!moov) return false;
  const trak = videoTrak(buf, moov[1], moov[2]);
  if (!trak) return false;
  const mdia = find(buf, trak[1], trak[2], 'mdia');
  const minf = mdia ? find(buf, mdia[1], mdia[2], 'minf') : null;
  const stbl = minf ? find(buf, minf[1], minf[2], 'stbl') : null;
  const stsd = stbl ? find(buf, stbl[1], stbl[2], 'stsd') : null;
  if (!stsd || !stbl || !minf || !mdia) return false;
  const sample = iterBoxes(buf, stsd[1] + 8, stsd[2]).next();
  if (sample.done) return false;
  const [, sStart, sPs, sEnd] = sample.value;
  for (const [t] of iterBoxes(buf, sPs + 78, sEnd)) if (skipTypes.includes(t)) return false;

  const add = buildBox();
  const delta = add.length;
  const insertAt = sEnd;
  const absInsert = base + insertAt;
  for (const [t, , ps] of iterBoxes(buf, stbl[1], stbl[2])) {
    if (t === 'stco') {
      const cnt = u32(buf, ps + 4);
      for (let i = 0; i < cnt; i++) {
        const o = ps + 8 + 4 * i;
        const v = u32(buf, o);
        if (v >= absInsert) packU(buf, v + delta, o, 4);
      }
    } else if (t === 'co64') {
      const cnt = u32(buf, ps + 4);
      for (let i = 0; i < cnt; i++) {
        const o = ps + 8 + 8 * i;
        const v = u64(buf, o);
        if (v >= BigInt(absInsert)) packQ(buf, v + BigInt(delta), o);
      }
    }
  }
  for (const bstart of [sStart, stsd[0], stbl[0], minf[0], mdia[0], trak[0], moov[0]]) grow(buf, bstart, delta);

  const out = Buffer.concat([buf.subarray(0, insertAt), add, buf.subarray(insertAt)]);
  const fd = fs.openSync(path, 'r+');
  try {
    let pos = 0;
    while (pos < out.length) pos += fs.writeSync(fd, out, pos, out.length - pos, base + pos);
    fs.ftruncateSync(fd, base + out.length);
  } finally {
    fs.closeSync(fd);
  }
  return true;
}

/** Add mdcv + clli to the video sample entry of the MP4 at path, in place; an unknown colorspace
 * falls back to the default. True if written, false if skipped. */
export function injectHdr10(
  path: string,
  maxNits = 1000,
  minNits = 0.0,
  maxcll = 0,
  maxfall = 0,
  colorspace: string = DEFAULT_COLORSPACE,
): boolean {
  const [primaries, white] = Object.prototype.hasOwnProperty.call(MASTERING_COLORSPACES, colorspace)
    ? MASTERING_COLORSPACES[colorspace]
    : MASTERING_COLORSPACES[DEFAULT_COLORSPACE];
  return insertIntoSampleEntry(path, () => mdcvClli(primaries, white, maxNits, minNits, maxcll, maxfall), [
    'mdcv',
    'clli',
  ]);
}

// Dolby Vision Profile 8.1 signalling: the reasoning sits on hdr10_meta.py's dvvC block
const DV_LEVEL_CAPS = [
  22118400, 27648000, 49766400, 62208000, 124416000, 199065600, 248832000, 398131200, 497664000, 995328000, 1990656000,
];

/** Dolby Vision level from the luminance pixel rate w * h * fps, clamped to the top level. */
export function dvLevel(width: number, height: number, fps: number): number {
  const pps = width * height * Math.max(1.0, fps);
  for (let i = 0; i < DV_LEVEL_CAPS.length; i++) if (pps <= DV_LEVEL_CAPS[i]) return i + 1;
  return DV_LEVEL_CAPS.length;
}

function dvConfigBox(profile: number, level: number, blCompat: number): Buffer {
  const bits =
    (BigInt(profile & 0x7f) << 41n) |
    (BigInt(level & 0x3f) << 35n) |
    (1n << 34n) |
    (1n << 32n) |
    (BigInt(blCompat & 0xf) << 28n);
  const box = Buffer.alloc(8 + 24);
  box.writeUInt32BE(box.length, 0);
  box.write('dvvC', 4, 'latin1');
  box[8] = 1;
  box[9] = 0;
  for (let i = 0; i < 6; i++) box[10 + i] = Number((bits >> BigInt(8 * (5 - i))) & 0xffn);
  return box;
}

/** Stamp a DV configuration box (dvvC) into the video sample entry in place (default Profile
 * 8.1, HDR10-compatible); idempotent. True if written, false if skipped. */
export function injectDvConfig(
  path: string,
  width: number,
  height: number,
  fps: number,
  profile = 8,
  blCompat = 1,
): boolean {
  return insertIntoSampleEntry(path, () => dvConfigBox(profile, dvLevel(width, height, fps), blCompat), [
    'dvvC',
    'dvcC',
  ]);
}
