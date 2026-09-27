// The render plan: engine/render_plan.py ported line by line, the
// comments there carry the reasoning (the image scale / downscale fold, the output size, the
// native offline eligibility). Pure functions; the callers write every stderr line.
import { dscaleVf, Stream } from './probe';
import { pyFixed, pyRound } from './pyfmt';

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
  nvidia_order: boolean;
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
  nr_passes: number; // int, 1..10: DLSS 5 chained per frame
}

export interface ScalePlan {
  w: number;
  h: number;
  upscaleF: number;
  upscale: boolean;
  vf: string[];
  note: string | null;
}

/** The image scale / downscale fold for a w x h source with the (clamped) --upscale factor and
 * the --scale slider. */
export function scalePlan(st: Stream, w: number, h: number, upscaleF: number, scale: number | null): ScalePlan {
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
    if ((sw !== w || sh !== h) && ow >= sw && oh >= sh) img = [sw, sh];
  }
  if (img) {
    const [sw, sh] = img;
    upscaleF = Math.max(1.0 / 16, Math.min(16.0, (upscaleF * h) / sh));
    note =
      `image scale: processing at ${sw}x${sh} of ${w}x${h} ` +
      `(${pyFixed(userS * 100, 0)}%), output size restored by the upscale pass\n`;
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
