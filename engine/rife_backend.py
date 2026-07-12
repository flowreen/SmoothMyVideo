"""RIFE 4.26-heavy interpolation backend, with optional DRBA timing for anime.

Model code and weights are vendored from routineLife1/DRBA (MIT, engine/rife/DRBA-LICENSE.txt),
which itself packages hzwer's Practical-RIFE 4.26 heavy checkpoint (MIT). Two modes:

* Plain RIFE (`--rife`): pair-based, arbitrary timestep - exposes the engine's standard
  `reuse(a, b, scale)` / `inference(a, b, reuse, t)` interface, so both existing render loops
  (on-grid multi and --fps resample) drive it exactly like GMFSS. `reuse` caches the two IFNet
  feature encodes per pair (the flow itself is timestep-dependent and cannot be cached).
* DRBA (`--rife-drba`): triple-based - `inference_ts_drba(I0, I1, I2, ts, reuse)` renders one
  window of output times around the centre frame, biasing timesteps per-region via the
  DistanceRatioMap so nonlinear (character) motion keeps its original pace while linear motion
  (pans) smooths fully. The engine's dedicated DRBA window loop drives this; `linear=True`
  matches upstream's inference script. The `reuse` return chains flow state between windows.

Inputs are the engine's standard frame tensors: [1,3,H,W] float RGB in [0,1], padded to a
multiple of 64 (the engine pads to max(64, 64/scale) which satisfies IFNet at every scale).
"""
import os

import torch

from rife.IFNet_HDv3 import IFNet
from rife.drm import calc_drm_rife, warp

_DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _convert(param):
    # upstream checkpoint keys carry a DataParallel "module." prefix
    return {k.replace("module.", ""): v for k, v in param.items() if "module." in k}


class RIFE:
    def __init__(self, weights_dir=None, scale=1.0, device=_DEV):
        if weights_dir is None:
            weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rife")
        self.device = device
        self.ifnet = IFNet().to(device).eval()
        self.ifnet.load_state_dict(
            _convert(torch.load(os.path.join(weights_dir, "flownet.pkl"),
                                map_location="cpu", weights_only=True)), strict=False)
        self.pad_size = 64
        self.set_scale(scale)

    def set_scale(self, scale):
        # flow scale, GMFSS semantics: 1.0 normally, 0.5 for 4K+ (the engine decides)
        self.scale = scale
        self.scale_list = [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]

    # --- plain pair interface (matches the GMFSS wrapper the render loops call) ---------------
    @torch.inference_mode()
    def reuse(self, a, b, scale=None):
        # per-pair feature encodes; IFNet accepts them precomputed (the DRBA path relies on it)
        return self.ifnet.encode(a[:, :3]), self.ifnet.encode(b[:, :3])

    @torch.inference_mode()
    @torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu")
    def inference(self, a, b, reuse=None, t=0.5):
        if t <= 0:
            return a
        if t >= 1:
            return b
        f0, f1 = reuse if reuse is not None else (None, None)
        return self.ifnet(torch.cat((a, b), 1), timestep=t,
                          scale_list=self.scale_list, f0=f0, f1=f1)[0]

    # --- DRBA triple interface (adapted from upstream models/rife.py, MIT) --------------------
    def calc_flow(self, a, b, f0=None, f1=None):
        # flow at the coarsest pyramid level only (significantly faster, near-lossless for DRM)
        timestep = (a[:, :1].clone() * 0 + 1) * 0.5
        f0 = self.ifnet.encode(a[:, :3]) if f0 is None else f0
        f1 = self.ifnet.encode(b[:, :3]) if f1 is None else f1
        flow, _, _ = self.ifnet.block0(torch.cat((a[:, :3], b[:, :3], f0, f1, timestep), 1),
                                       None, scale=self.scale_list[0])
        flow50, flow51 = flow[:, :2], flow[:, 2:]
        warp_method = 'avg'
        flow05 = -1 * warp(flow50, flow50, None, warp_method)
        flow15 = -1 * warp(flow51, flow51, None, warp_method)
        ones_mask = flow05.clone() * 0 + 1
        mask05 = warp(ones_mask, flow50, None, warp_method)
        mask15 = warp(ones_mask, flow51, None, warp_method)
        gap05 = mask05 < 0.999
        gap15 = mask15 < 0.999
        flow05[gap05] = (ones_mask * max(flow05.shape[2], flow05.shape[3]))[gap05]
        flow15[gap15] = (ones_mask * max(flow15.shape[2], flow15.shape[3]))[gap15]
        return flow05 * 2, flow15 * 2, f0, f1

    @torch.inference_mode()
    @torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu")
    def inference_ts_drba(self, I0, I1, I2, ts, reuse=None, linear=True):
        """One DRBA window: output frames at times `ts` in [0.5, 1.5) around centre frame I1
        (ts = offset + 1). Returns (frames, reuse_state) - pass the state to the next window."""
        flow10, _flow01, f1, f0 = self.calc_flow(I1, I0) if not reuse else reuse
        if reuse is None:
            flow12, flow21, f1, f2 = self.calc_flow(I1, I2)
        else:
            flow12, flow21, f1, f2 = self.calc_flow(I1, I2, f0=reuse[2])
        output = []
        for t in ts:
            if t == 1:
                output.append(I1)
            elif t < 1:
                tt = 1 - t
                drm = calc_drm_rife(tt, flow10, flow12, linear)
                out = self.ifnet(torch.cat((I1, I0), 1), timestep=drm['drm_t1_t01'],
                                 scale_list=self.scale_list, f0=f1, f1=f0)[0]
                output.append(out)
            else:
                tt = t - 1
                drm = calc_drm_rife(tt, flow10, flow12, linear)
                out = self.ifnet(torch.cat((I1, I2), 1), timestep=drm['drm_t1_t12'],
                                 scale_list=self.scale_list, f0=f1, f1=f2)[0]
                output.append(out)
        # the next window's left flow state is this window's right one, reversed
        return output, (flow21, flow12, f2, f1)
