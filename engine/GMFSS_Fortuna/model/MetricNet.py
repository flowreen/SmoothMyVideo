import torch
import torch.nn as nn
import torch.nn.functional as F

from model.gmflow.geometry import forward_backward_consistency_check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


backwarp_tenGrid = {}


def _grid_axis(n, device):
    # SMV: torch.linspace(-1, 1, n) for a SIZE-FREE export. linspace's
    # length specializes under torch.export, so the export path spells out onnxscript's own
    # aten_linspace (step = 2 / (n - 1) in fp32, split at i < n / 2) with n taken off the arange.
    # The torch.maximum no-ops sit between every multiply and its add: in a size-free graph this
    # runs on the GPU, where TensorRT would fuse multiply-add into one rounding, while the
    # per-size export's host-folded constant rounded twice; split, the engine is bit-exact with
    # the per-size one (checked on metricnet at 480x288 and 960x544).
    i = torch.arange(n, device=device, dtype=torch.float32)
    steps_f = i[-1:] + 1.0
    sm1 = steps_f - 1.0
    step = (1.0 - (-1.0)) / sm1
    lo = torch.full_like(i, -1e30)
    a = torch.maximum(step * i, lo)
    b = torch.maximum(step * (sm1 - i), lo)
    return torch.where(i < steps_f / 2.0, -1.0 + a, 1.0 - b)


def backwarp(tenIn, tenflow):
    if torch.compiler.is_compiling():
        # the export path (SMV): the grid without linspace, see _grid_axis
        h, w = tenflow.shape[2], tenflow.shape[3]
        hor = _grid_axis(w, tenflow.device).view(1, 1, 1, w).expand(tenflow.shape[0], -1, h, -1)
        ver = _grid_axis(h, tenflow.device).view(1, 1, h, 1).expand(tenflow.shape[0], -1, -1, w)
        grid = torch.cat([hor, ver], 1)
        tenflow = torch.cat([tenflow[:, 0:1, :, :] / ((tenIn.shape[3] - 1.0) / 2.0), tenflow[:, 1:2, :, :] / ((tenIn.shape[2] - 1.0) / 2.0)], 1)
        return torch.nn.functional.grid_sample(input=tenIn, grid=(grid + tenflow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)
    # ONNX-export traces run this with FakeTensors; caching one poisons the dict for every
    # later export AND the real eager path at the same shape (SMV fix), so fakes
    # bypass the cache entirely.
    fake = isinstance(tenflow, torch._subclasses.fake_tensor.FakeTensor)
    grid = None if fake else backwarp_tenGrid.get(str(tenflow.shape))
    if grid is None:
        tenHor = torch.linspace(start=-1.0, end=1.0, steps=tenflow.shape[3], dtype=tenflow.dtype, device=tenflow.device).view(1, 1, 1, -1).repeat(1, 1, tenflow.shape[2], 1)
        tenVer = torch.linspace(start=-1.0, end=1.0, steps=tenflow.shape[2], dtype=tenflow.dtype, device=tenflow.device).view(1, 1, -1, 1).repeat(1, 1, 1, tenflow.shape[3])

        grid = torch.cat([tenHor, tenVer], 1) if fake else torch.cat([tenHor, tenVer], 1).to(device)
        if not fake:
            backwarp_tenGrid[str(tenflow.shape)] = grid
    # end

    tenflow = torch.cat([tenflow[:, 0:1, :, :] / ((tenIn.shape[3] - 1.0) / 2.0), tenflow[:, 1:2, :, :] / ((tenIn.shape[2] - 1.0) / 2.0)], 1)

    return torch.nn.functional.grid_sample(input=tenIn, grid=(grid + tenflow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)


class MetricNet(nn.Module):
    def __init__(self):
        super(MetricNet, self).__init__()
        self.metric_in = nn.Conv2d(14, 64, 3, 1, 1)
        self.metric_net1 = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, 1, 1)
        )
        self.metric_net2 = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, 1, 1)
        )
        self.metric_net3 = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, 1, 1)
        )
        self.metric_out = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(64, 2, 3, 1, 1)
        )

    def forward(self, img0, img1, flow01, flow10):
        metric0 = F.l1_loss(img0, backwarp(img1, flow01), reduction='none').mean([1], True)
        metric1 = F.l1_loss(img1, backwarp(img0, flow10), reduction='none').mean([1], True)

        fwd_occ, bwd_occ = forward_backward_consistency_check(flow01, flow10)

        flow01 = torch.cat([flow01[:, 0:1, :, :] / ((flow01.shape[3] - 1.0) / 2.0), flow01[:, 1:2, :, :] / ((flow01.shape[2] - 1.0) / 2.0)], 1)
        flow10 = torch.cat([flow10[:, 0:1, :, :] / ((flow10.shape[3] - 1.0) / 2.0), flow10[:, 1:2, :, :] / ((flow10.shape[2] - 1.0) / 2.0)], 1)
        
        img = torch.cat((img0, img1), 1)
        metric = torch.cat((-metric0, -metric1), 1)
        flow = torch.cat((flow01, flow10), 1)
        occ = torch.cat((fwd_occ.unsqueeze(1), bwd_occ.unsqueeze(1)), 1)

        feat = self.metric_in(torch.cat((img, metric, flow, occ), 1))
        feat = self.metric_net1(feat) + feat
        feat = self.metric_net2(feat) + feat
        feat = self.metric_net3(feat) + feat
        metric = self.metric_out(feat)

        metric = torch.tanh(metric) * 10

        return metric[:, :1], metric[:, 1:2]
