"""RIFE 4.26-heavy: the IFNet with its weights, for the size-free ONNX export (onnx_export.py reads
`ifnet` and `scale_list`), and DRBA's coarse flow (`calc_flow`, the harness gates' reference).

Model code and weights are vendored from routineLife1/DRBA (MIT, engine/rife/DRBA-LICENSE.txt),
which itself packages hzwer's Practical-RIFE 4.26 heavy checkpoint (MIT).

Inputs are frame tensors: [1,3,H,W] float RGB in [0,1], padded to a multiple of 64.
"""
import os

import torch

from rife.IFNet_HDv3 import IFNet
from rife.drm import warp

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
        self.set_scale(scale)

    def set_scale(self, scale):
        # flow scale, GMFSS semantics: 1.0 normally, 0.5 for 4K+ (the engine decides)
        self.scale = scale
        self.scale_list = [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]

    # --- DRBA's flow (adapted from upstream models/rife.py, MIT) -----------------------------
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
