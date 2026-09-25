// Dolby Vision Profile 8.1 and HDR10+ exports of a finished HDR10 MP4 (priority 24 step 6c):
// render_encode.py's dv_export / hp_export and their scene-shot grouping ported line by line, the
// comments there carry the reasoning (one metadata block per SHOT, the L1 / SceneInfo layouts,
// HDR10+ before DV, why the remux re-stamps the HDR10 boxes). Both are best-effort: any failure is
// one "[dv]" / "[hdr10+]" line quoting python's repr() of the error, and the HDR10 file stays.
// The stats are ints (rtxvideo / the native host's TrueHDR pass), so the configs hold only ints and
// strings; pyJsonDumps writes them as python's json.dump does.
import { spawnSync } from 'child_process';
import * as fs from 'fs';
import { CalledProcessError, hdrLight, HdrStats, HpFrame, pyReprStr, runChecked, Say } from './encode';
import { injectDvConfig, injectHdr10 } from './hdr10meta';
import { pyFloatRepr, pyRound } from './pyfmt';

export const SHOT_CUT = 50; // 12-bit PQ codes: consecutive avg step that marks a hard scene cut
export const SHOT_DRIFT = 100; // avg spread within one shot before a fade forces a new shot
export const SHOT_MIN_S = 0.2; // minimum shot length, seconds

/** Scene-shot [start, end) ranges of a per-frame average-brightness series (12-bit PQ codes). */
export function sceneBounds(avgs: number[], fps: number): [number, number][] {
  if (!avgs.length) throw new Error('list index out of range'); // python: avgs[0] raises
  const minLen = Math.max(2, pyRound(fps * SHOT_MIN_S));
  const bounds: [number, number][] = [];
  let s = 0,
    lo = avgs[0],
    hi = avgs[0];
  for (let i = 1; i < avgs.length; i++) {
    const av = avgs[i];
    lo = Math.min(lo, av);
    hi = Math.max(hi, av);
    if (i - s >= minLen && (Math.abs(av - avgs[i - 1]) > SHOT_CUT || hi - lo > SHOT_DRIFT)) {
      bounds.push([s, i]);
      s = i;
      lo = av;
      hi = av;
    }
  }
  bounds.push([s, avgs.length]);
  return bounds;
}

// python min() / max() over a shot; a loop, since a spread call caps near 100k frames
function minOf(xs: number[]): number {
  let m = xs[0];
  for (const x of xs) if (x < m) m = x;
  return m;
}
function maxOf(xs: number[]): number {
  let m = xs[0];
  for (const x of xs) if (x > m) m = x;
  return m;
}

/** Per-shot [start, duration, min, avg, max] of per-frame DV L1 triples. */
export function l1Shots(l1: number[][], fps: number): number[][] {
  return sceneBounds(
    l1.map((f) => f[1]),
    fps,
  ).map(([s, e]) => {
    const seg = l1.slice(s, e);
    return [
      s,
      e - s,
      minOf(seg.map((f) => f[0])),
      pyRound(seg.reduce((a, f) => a + f[1], 0) / seg.length),
      maxOf(seg.map((f) => f[2])),
    ];
  });
}

/** Luminance in the HDR10+ metadata's 0.1-nit units -> 12-bit PQ code (ST 2084 inverse). */
export function pq12(tenths: number): number {
  const p = Math.pow(Math.max(tenths, 0) / 100000.0, 0.1593017578125);
  return pyRound(4095.0 * Math.pow((0.8359375 + 18.8515625 * p) / (1.0 + 18.6875 * p), 78.84375));
}

/** python json.dumps with the default separators and ensure_ascii; integral numbers print as
 * ints (every value here is a python int), others as python's float repr. */
export function pyJsonDumps(v: unknown): string {
  if (v === null || v === undefined) return 'null';
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  if (typeof v === 'number') {
    if (Number.isNaN(v)) return 'NaN';
    if (!Number.isFinite(v)) return v > 0 ? 'Infinity' : '-Infinity';
    return Number.isInteger(v) ? String(v) : pyFloatRepr(v);
  }
  if (typeof v === 'string') {
    let out = '"';
    for (const ch of v) {
      const c = ch.codePointAt(0) as number;
      if (ch === '"') out += '\\"';
      else if (ch === '\\') out += '\\\\';
      else if (ch === '\n') out += '\\n';
      else if (ch === '\r') out += '\\r';
      else if (ch === '\t') out += '\\t';
      else if (ch === '\b') out += '\\b';
      else if (ch === '\f') out += '\\f';
      else if (c < 0x20 || c > 0x7e) {
        const u = c > 0xffff ? [0xd800 + ((c - 0x10000) >> 10), 0xdc00 + ((c - 0x10000) & 0x3ff)] : [c];
        out += u.map((x) => '\\u' + x.toString(16).padStart(4, '0')).join('');
      } else out += ch;
    }
    return out + '"';
  }
  if (Array.isArray(v)) return '[' + v.map(pyJsonDumps).join(', ') + ']';
  return (
    '{' +
    Object.entries(v as Record<string, unknown>)
      .map(([k, x]) => pyJsonDumps(k) + ': ' + pyJsonDumps(x))
      .join(', ') +
    '}'
  );
}

/** python repr() of the errors these exports can raise (the failure line quotes repr(e)[:200]). */
export class PyRuntimeError extends Error {}
export function pyExcRepr(e: unknown): string {
  if (e instanceof CalledProcessError)
    return `CalledProcessError(${e.returncode}, [${e.cmd.map(pyReprStr).join(', ')}])`;
  if (e instanceof PyRuntimeError) return `RuntimeError(${pyReprStr(e.message)})`;
  const x = e as NodeJS.ErrnoException;
  if (x && x.code === 'ENOENT') {
    // a program that cannot start (CreateProcess, WinError 2) vs a file call (os.replace etc.)
    return x.syscall && x.syscall.startsWith('spawn')
      ? "FileNotFoundError(2, 'The system cannot find the file specified', None, 2, None)"
      : "FileNotFoundError(2, 'The system cannot find the file specified')";
  }
  if (x && (x.code === 'EACCES' || x.code === 'EPERM')) return "PermissionError(13, 'Access is denied')";
  return `Exception(${pyReprStr(x instanceof Error ? x.message : String(e))})`;
}

function cut(s: string, n: number): string {
  return Array.from(s).slice(0, n).join('');
}

function removeQuiet(p: string): void {
  try {
    fs.unlinkSync(p);
  } catch {
    /* python: except OSError: pass */
  }
}

/** The dovi_tool generate config: one L1 block per scene shot + L6. */
export function dvConfig(l1: number[][], nits: number, cll: number, fall: number, fps: number): object {
  return {
    cm_version: 'V29',
    length: l1.length,
    level6: {
      max_display_mastering_luminance: nits,
      min_display_mastering_luminance: 1,
      max_content_light_level: cll,
      max_frame_average_light_level: fall,
    },
    shots: l1Shots(l1, fps).map(([s, d, mn, av, mx]) => ({
      start: s,
      duration: d,
      metadata_blocks: [{ Level1: { min_pq: mn, avg_pq: av, max_pq: mx } }],
    })),
  };
}

/** The hdr10plus_tool metadata JSON: one SceneInfo per frame, constant within each shot. */
export function hpConfig(hp: HpFrame[], fps: number): object {
  const scenes = sceneBounds(
    hp.map((f) => pq12(f.avg)),
    fps,
  );
  const sinfo: object[] = [];
  scenes.forEach(([s, e], sid) => {
    const seg = hp.slice(s, e);
    const dist: number[] = [];
    for (let k = 0; k < 9; k++) dist.push(pyRound(seg.reduce((a, f) => a + f.dist[k], 0) / seg.length));
    dist[1] = maxOf(seg.map((f) => f.dist[1]));
    const lum = {
      AverageRGB: pyRound(seg.reduce((a, f) => a + f.avg, 0) / seg.length),
      LuminanceDistributions: { DistributionIndex: [1, 5, 10, 25, 50, 75, 90, 95, 99], DistributionValues: dist },
      MaxScl: [0, 1, 2].map((c) => maxOf(seg.map((f) => f.maxscl[c]))),
    };
    for (let i = 0; i < seg.length; i++) {
      sinfo.push({
        LuminanceParameters: lum,
        NumberOfWindows: 1,
        TargetedSystemDisplayMaximumLuminance: 0,
        SceneFrameIndex: i,
        SceneId: sid,
        SequenceFrameIndex: s + i,
      });
    }
  });
  return {
    JSONInfo: { HDR10plusProfile: 'A', Version: '1.0' },
    SceneInfo: sinfo,
    SceneInfoSummary: {
      SceneFirstFrameIndex: scenes.map(([s]) => s),
      SceneFrameNumbers: scenes.map(([s, e]) => e - s),
    },
    ToolInfo: { Tool: 'SmoothMyVideo', Version: '1.0' },
  };
}

/** The extract / remux argv both exports share. */
function extractCmd(ffmpeg: string, mp4Path: string, hevc: string): string[] {
  return [
    ffmpeg,
    '-v',
    'error',
    '-y',
    '-i',
    mp4Path,
    '-map',
    '0:v:0',
    '-c',
    'copy',
    '-bsf:v',
    'hevc_mp4toannexb',
    '-f',
    'hevc',
    hevc,
  ];
}

function remuxCmd(ffmpeg: string, rateStr: string, es: string, mp4Path: string, tmpOut: string): string[] {
  return [
    ffmpeg,
    '-v',
    'error',
    '-y',
    '-f',
    'hevc',
    '-r',
    rateStr,
    '-i',
    es,
    '-i',
    mp4Path,
    '-map',
    '0:v:0',
    '-map',
    '1:a?',
    '-map',
    '1:s?',
    '-c',
    'copy',
    '-color_primaries',
    'bt2020',
    '-color_trc',
    'smpte2084',
    '-colorspace',
    'bt2020nc',
    '-color_range',
    'tv',
    '-tag:v',
    'hvc1',
    '-max_interleave_delta',
    '0',
    tmpOut,
  ];
}

/** The finished HDR10 MP4 -> Dolby Vision Profile 8.1 in place (render_encode.dv_export). */
export function dvExport(
  mp4Path: string,
  ffmpeg: string,
  doviExe: string,
  rtx: HdrStats | null,
  nits: number,
  masterPrim: string,
  rateStr: string,
  outLabel: number,
  outW: number,
  outH: number,
  say: Say,
): void {
  const l1 = rtx && rtx.l1 ? Array.from(rtx.l1) : [];
  if (!l1.length) {
    say('[dv] no per-frame metadata collected; kept the HDR10 file\n');
    return;
  }
  const base = mp4Path + '.dvwork';
  const [hevc, rpu, dvhevc, cfg, tmpOut] = ['.hevc', '.rpu', '.dv.hevc', '.json', '.mp4'].map((s) => base + s);
  const [cll, fall] = hdrLight(rtx);
  try {
    runChecked(extractCmd(ffmpeg, mp4Path, hevc));
    fs.writeFileSync(cfg, pyJsonDumps(dvConfig(l1, nits, cll, fall, outLabel || 24)));
    runChecked([doviExe, 'generate', '-j', cfg, '-o', rpu]);
    runChecked([doviExe, 'inject-rpu', '-i', hevc, '--rpu-in', rpu, '-o', dvhevc]);
    runChecked(remuxCmd(ffmpeg, rateStr, dvhevc, mp4Path, tmpOut));
    injectDvConfig(tmpOut, outW, outH, outLabel);
    injectHdr10(tmpOut, nits, 0.0, cll, fall, masterPrim);
    fs.renameSync(tmpOut, mp4Path);
    say('Dolby Vision: Profile 8.1 written (HDR10-compatible)\n');
  } catch (e) {
    say(`[dv] export failed (${cut(pyExcRepr(e), 200)}); kept the HDR10 file at ${mp4Path}\n`);
  } finally {
    for (const p of [hevc, rpu, dvhevc, cfg, tmpOut]) removeQuiet(p);
  }
}

/** HDR10+ (ST 2094-40) dynamic metadata into the finished HDR10 MP4 in place (render_encode.hp_export). */
export function hpExport(
  mp4Path: string,
  ffmpeg: string,
  hpExe: string,
  rtx: HdrStats | null,
  nits: number,
  masterPrim: string,
  rateStr: string,
  outLabel: number,
  say: Say,
): void {
  const hp = rtx && rtx.hp ? Array.from(rtx.hp) : [];
  if (!hp.length) {
    say('[hdr10+] no per-frame metadata collected; kept the HDR10 file\n');
    return;
  }
  const base = mp4Path + '.hpwork';
  const [hevc, hphevc, cfg, tmpOut] = ['.hevc', '.hp.hevc', '.json', '.mp4'].map((s) => base + s);
  const [cll, fall] = hdrLight(rtx);
  try {
    runChecked(extractCmd(ffmpeg, mp4Path, hevc));
    fs.writeFileSync(cfg, pyJsonDumps(hpConfig(hp, outLabel || 24)));
    const r = spawnSync(hpExe, ['inject', '-i', hevc, '-j', cfg, '-o', hphevc], {
      windowsHide: true,
      encoding: 'utf8',
    });
    if (r.error) throw r.error;
    if (r.status !== 0) {
      const text = ((r.stderr || '') + (r.stdout || '')).replace(/\r\n/g, '\n').trim();
      throw new PyRuntimeError('hdr10plus_tool inject: ' + Array.from(text).slice(-300).join(''));
    }
    runChecked(remuxCmd(ffmpeg, rateStr, hphevc, mp4Path, tmpOut));
    injectHdr10(tmpOut, nits, 0.0, cll, fall, masterPrim);
    fs.renameSync(tmpOut, mp4Path);
    say('HDR10+: dynamic metadata written (HDR10-compatible)\n');
  } catch (e) {
    say(`[hdr10+] export failed (${cut(pyExcRepr(e), 200)}); kept the HDR10 file at ${mp4Path}\n`);
  } finally {
    for (const p of [hevc, hphevc, cfg, tmpOut]) removeQuiet(p);
  }
}
