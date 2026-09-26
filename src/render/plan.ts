// The render plan: engine/render_plan.py ported line by line, the
// comments there carry the reasoning (the image scale / downscale fold, the output size, the
// native offline eligibility, the DLSS + RTX Video two-pass split). Pure functions; the callers
// write every stderr line and run the passes.
import { createHash } from 'crypto';
import * as fs from 'fs';
import { dscaleVf, Stream } from './probe';
import { pyFloatRepr, pyG, pyFixed, pyInt, pyRound } from './pyfmt';

/** The render.py arguments the plan reads, typed as argparse has them (floats vs ints matter:
 * python prints a float argument as "2.0"). */
export interface RenderArgs {
  multi: number; // int
  scale: number | null; // float
  upscale: number; // float
  sharpen: number; // float
  rife: boolean;
  rife_drba: boolean;
  lsfg: boolean;
  fruc: boolean;
  dlssg: boolean;
  nvof: boolean;
  no_interp: boolean;
  rtx_vsr: boolean;
  rtx_hdr: boolean;
  restore: boolean;
  dlssnr: boolean;
  dv: boolean;
  hdr10plus: boolean;
  hdr_saturation: number; // int
  hdr_contrast: number; // int
  hdr_color: string;
  hdr_vibrance: number; // float
  hdr_satboost: number; // float
  nr_structure: number; // float
  nr_tone: number; // float
  nr_style: number; // int
}

export type Env = Record<string, string | undefined>;

export interface ScalePlan {
  w: number;
  h: number;
  upscaleF: number;
  upscale: boolean;
  vf: string[];
  note: string | null;
}

/** The image scale / downscale fold for a w x h source with the (clamped) --upscale factor and
 * the --scale slider; pass1 = this process is pass 1 of the DLSS + RTX two-pass split. */
export function scalePlan(
  st: Stream,
  w: number,
  h: number,
  upscaleF: number,
  scale: number | null,
  pass1: boolean,
): ScalePlan {
  let vf: string[] = [],
    note: string | null = null;
  const userS = scale && scale > 0 && scale < 1.0 ? scale : 1.0;
  let img: [number, number] | null = null;
  if (userS < Math.min(1.0, upscaleF)) {
    const sw = Math.max(64, pyRound(w * userS) & ~1);
    const sh = Math.max(64, pyRound(h * userS) & ~1);
    const f = Math.max(1.0 / 16, Math.min(16.0, (upscaleF * h) / sh));
    const [ow, oh] = outputSize(sw, sh, f, f !== 1.0);
    // the host only enlarges: an output that the rounding would leave smaller than the processed
    // size (an image scale within a pixel of a downscale target) folds the downscale instead
    if ((sw !== w || sh !== h) && (pass1 || (ow >= sw && oh >= sh))) img = [sw, sh];
  }
  if (img) {
    const [sw, sh] = img;
    if (pass1) {
      upscaleF = 1.0;
      note =
        `image scale: processing at ${sw}x${sh} of ${w}x${h} ` +
        `(${pyFixed(userS * 100, 0)}%), pass 2 restores the output size\n`;
    } else {
      upscaleF = Math.max(1.0 / 16, Math.min(16.0, (upscaleF * h) / sh));
      note =
        `image scale: processing at ${sw}x${sh} of ${w}x${h} ` +
        `(${pyFixed(userS * 100, 0)}%), output size restored by the upscale pass\n`;
    }
    vf = dscaleVf(st, w, h, sw, sh);
    w = sw;
    h = sh;
  } else if (upscaleF < 1.0) {
    let [sw, sh] = outputSize(w, h, upscaleF, true);
    sw = Math.max(64, sw);
    sh = Math.max(64, sh);
    if (sw !== w || sh !== h) {
      note =
        `downscale folded into the decode: whole pipeline runs at ` +
        `${sw}x${sh} (was ${w}x${h}), linear-light spline36\n`;
      vf = dscaleVf(st, w, h, sw, sh);
      w = sw;
      h = sh;
    }
    upscaleF = 1.0;
  }
  return { w, h, upscaleF, upscale: upscaleF !== 1.0, vf, note };
}

/** The output resolution: the factor on both dimensions, each rounded down to even. */
export function outputSize(w: number, h: number, upscaleF: number, upscale: boolean): [number, number] {
  if (upscale) return [Math.floor(pyRound(w * upscaleF) / 2) * 2, Math.floor(pyRound(h * upscaleF) / 2) * 2];
  return [w, h];
}

/** GMFSS is the default model: no other model flag and frames are generated. */
export function isGmfss(args: RenderArgs): boolean {
  return !(args.rife || args.rife_drba || args.lsfg || args.fruc || args.dlssg || args.nvof || args.no_interp);
}

/** Why an --nvof render cannot run, or null. */
export function nvofRefusal(args: RenderArgs, fpsMode: boolean): string | null {
  if (args.no_interp) return '--no-interp renders no tweens, pick no model instead';
  if (fpsMode)
    return (
      'the target fps is not a whole multiple of the source rate; this model renders ' +
      'integer multiples only (Speed: Multiplier, or a target that divides evenly)'
    );
  return null;
}

/** True when this process must split the render (it is not itself one of the passes). */
export function twopassNeeded(
  dlssg: boolean,
  rtxVsr: boolean,
  rtxHdr: boolean,
  dlssnr: boolean,
  noInterp: boolean,
  env: Env = process.env,
): boolean {
  return dlssg && (rtxVsr || rtxHdr || dlssnr) && !noInterp && env.SMV_TWOPASS_PHASE === undefined;
}

export interface TwoPass {
  inter: string;
  t1: number;
  t2: number;
  g1: number;
  pass1: string[];
  pass2: string[];
}

/** The two-pass split: the intermediate's path, pass 1's and pass 2's work units, pass 1's
 * share of the one progress bar, and the two render argument lists. */
export function twopassPlan(
  args: RenderArgs,
  inp: string,
  outBase: string,
  outPath: string,
  codec: string,
  nb: number,
  sharpen: number,
  imgScaleVf: string[],
  upscale: boolean,
  upscaleF: number,
): TwoPass {
  let fp: string;
  try {
    const st = fs.statSync(inp, { bigint: true });
    fp = `${inp}|${st.mtimeNs}|${st.size}|${args.multi}|${codec}|10`;
  } catch {
    fp = `${inp}|${args.multi}|${codec}|10`;
  }
  const sig = createHash('md5').update(fp, 'utf8').digest('hex').slice(0, 8);
  const inter = outBase + `.dlss-interp-${sig}.mkv`;
  const t1 = Math.max(1, nb - 1);
  const t2 = args.multi * t1 + 1;
  const g1 = pyInt(0.6 * t2);

  let p1Scale: string[], p2Up: string[];
  if (imgScaleVf.length && upscale) {
    p1Scale = ['--scale', pyG(args.scale as number)];
    p2Up = ['--upscale', pyG(upscaleF, 6)];
  } else if (imgScaleVf.length) {
    p1Scale = ['--upscale', pyFloatRepr(args.upscale)];
    p2Up = [];
  } else {
    p1Scale = [];
    p2Up = upscale ? ['--upscale', pyFloatRepr(args.upscale)] : [];
  }
  const pass1 = [inp, String(args.multi), inter, '--dlssg', '--codec', codec, ...p1Scale];
  const pass2 = [inter, '1', outPath, '--no-interp', '--codec', codec, ...p2Up];
  if (args.rtx_vsr) pass2.push('--rtx-vsr');
  if (args.rtx_hdr)
    pass2.push(
      '--rtx-hdr',
      '--hdr-saturation',
      String(args.hdr_saturation),
      '--hdr-contrast',
      String(args.hdr_contrast),
      '--hdr-color',
      args.hdr_color,
      '--hdr-vibrance',
      pyFloatRepr(args.hdr_vibrance),
      '--hdr-satboost',
      pyFloatRepr(args.hdr_satboost),
    );
  if (args.restore) pass2.push('--restore');
  if (args.dlssnr)
    pass2.push(
      '--dlssnr',
      '--nr-structure',
      pyFloatRepr(args.nr_structure),
      '--nr-tone',
      pyFloatRepr(args.nr_tone),
      '--nr-style',
      String(args.nr_style),
    );
  if (sharpen > 0) pass2.push('--sharpen', pyFloatRepr(args.sharpen));
  if (args.dv) pass2.push('--dv');
  if (args.hdr10plus) pass2.push('--hdr10plus');
  return { inter, t1, t2, g1, pass1, pass2 };
}
