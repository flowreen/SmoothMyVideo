// The render plan: engine/render_plan.py ported line by line, the
// comments there carry the reasoning (the image scale / downscale fold, the output size, the
// native offline eligibility). Pure functions; the callers write every stderr line.
import { dscaleVf, Stream } from './probe';
import { pyFixed, pyRound } from './pyfmt';

/** The render.py arguments the plan reads, typed as argparse has them (floats vs ints matter:
 * python prints a float argument as "2.0"). */
export interface RenderArgs {
  multi: number; // int
  work_scale: string | null; // --scale: a DLSS mode or a number, the working size's share of the output
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
  no_gpu_fit: boolean; // --no-gpu-fit: no video memory fit (the host's batch and Auto's working size)
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

/** NVIDIA's DLSS modes: the working size's share of the output per axis (DLSS Programming Guide:
 * DLAA 1, Quality 1 / 1.5, Balanced 1 / 1.724, Performance 1 / 2, Ultra Performance 1 / 3). */
export const DLSS_MODES: Record<string, number> = {
  dlaa: 1.0,
  quality: 1 / 1.5,
  balanced: 1 / 1.724,
  performance: 1 / 2,
  ultra: 1 / 3,
};

/** NVIDIA's Auto by the output's pixel count: below 1080p DLAA, up to 1440p Quality, up to 4K
 * Performance, above it Ultra Performance. */
export function autoMode(ow: number, oh: number): string {
  const px = ow * oh;
  if (px < 1920 * 1080) return 'dlaa';
  if (px <= 2560 * 1440) return 'quality';
  if (px <= 3840 * 2160) return 'performance';
  return 'ultra';
}

/** --scale: a mode name (auto, dlaa, quality, balanced, performance, ultra) or a number in (0, 1];
 * absent = DLAA. null = not one of those. */
export function workFactor(scale: string | null, ow: number, oh: number): { mode: string; factor: number } | null {
  const s = (scale ?? 'dlaa').trim().toLowerCase();
  if (s === 'auto') {
    const m = autoMode(ow, oh);
    return { mode: `Auto (${m})`, factor: DLSS_MODES[m] };
  }
  if (s in DLSS_MODES) return { mode: s, factor: DLSS_MODES[s] };
  const f = s === '' ? NaN : Number(s);
  if (!(f > 0 && f <= 1)) return null;
  return { mode: `custom ${pyFixed(f * 100, 1)} %`, factor: f };
}

/** The largest working size the interpolation reaches: 3840x2160 pixels. */
export const WORK_MAX_PX = 3840 * 2160;

export interface WorkPlan {
  w: number; // the decode: the source, or the working size when that is smaller (the fold)
  h: number;
  workW: number; // the working size: DLSS 5 and the model
  workH: number;
  outW: number;
  outH: number;
  vf: string[]; // the decoder's downscale, empty = none
  note: string;
}

/** DLSS 5's GPU work never completes on a 64 px wide frame (the GPU resets; 102 and 128 px wide run, and so does a
 * 64 px high one): with DLSS 5 on, a working size below the output is at least this wide (the host's kNrMinWorkW, the
 * same rule live). */
export const NR_MIN_WORK_W = 128;

/** The pass sizes for a w x h source: the output (the --upscale factor, clamped), the working size =
 * the DLSS mode (--scale) x the output (even, at most WORK_MAX_PX keeping the aspect; the share raised, aspect kept,
 * until the shorter side is 64 px and, with DLSS 5 (nr), the width NR_MIN_WORK_W),
 * and the decode (a working size below the source folds the downscale into the decode, linear-light
 * Lanczos3). A string = why --scale is refused. */
export function workPlan(
  st: Stream,
  w: number,
  h: number,
  upscaleF: number,
  scale: string | null,
  nr = false,
): WorkPlan | string {
  const [ow, oh] = outputSize(w, h, upscaleF, upscaleF !== 1.0);
  const wf = workFactor(scale, ow, oh);
  if (!wf)
    return `--scale takes a DLSS mode (auto, dlaa, quality, balanced, performance, ultra) or a number in (0, 1], not '${scale}'`;
  // DLAA is the output itself (an odd source width stays odd, no 1-px fold)
  let ww = ow;
  let wh = oh;
  let raised = '';
  if (wf.factor !== 1) {
    const fFloor = Math.max(64 / ow, 64 / oh);
    const fNr = nr ? NR_MIN_WORK_W / ow : 0;
    const k = Math.max(wf.factor, fFloor, fNr);
    if (k > wf.factor)
      raised =
        fNr >= fFloor
          ? `, raised to DLSS 5's minimum width, ${NR_MIN_WORK_W} px, aspect kept`
          : ', raised to the 64 px floor, aspect kept';
    ww = Math.min(ow, Math.max(nr ? NR_MIN_WORK_W : 64, pyRound(ow * k) & ~1));
    wh = Math.min(oh, Math.max(64, pyRound(oh * k) & ~1));
  }
  let capped = false;
  if (ww * wh > WORK_MAX_PX) {
    const k = Math.sqrt(WORK_MAX_PX / (ww * wh));
    ww = Math.floor(ww * k) & ~1;
    wh = Math.floor(wh * k) & ~1;
    capped = true;
  }
  let vf: string[] = [];
  let [dw, dh] = [w, h];
  if (ww < w || wh < h) {
    vf = dscaleVf(st, w, h, ww, wh);
    [dw, dh] = [ww, wh];
  }
  const note =
    `DLSS mode ${wf.mode}: working size ${ww}x${wh}${raised} for the ${ow}x${oh} output (source ${w}x${h}` +
    (vf.length ? ', the downscale folded into the decode, linear-light Lanczos3' : '') +
    (capped ? "; capped at 3840x2160, the interpolation's reach" : '') +
    ')\n';
  return { w: dw, h: dh, workW: ww, workH: wh, outW: ow, outH: oh, vf, note };
}

/** Auto's candidates for its video memory fit, its own pick first: the plan of every DLSS mode from that pick down to
 * Ultra Performance (the render host prices them, --fit-work). */
export function autoCandidates(
  st: Stream,
  w: number,
  h: number,
  upscaleF: number,
  nr = false,
): { mode: string; plan: WorkPlan }[] {
  const [ow, oh] = outputSize(w, h, upscaleF, upscaleF !== 1.0);
  const order = Object.keys(DLSS_MODES);
  const out: { mode: string; plan: WorkPlan }[] = [];
  for (const m of order.slice(order.indexOf(autoMode(ow, oh)))) {
    const p = workPlan(st, w, h, upscaleF, m, nr);
    if (typeof p !== 'string') out.push({ mode: m, plan: p });
  }
  return out;
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
