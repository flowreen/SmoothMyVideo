import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backwarp_tenGrid = {}


def warp(tenInput, tenFlow):
    if torch.compiler.is_compiling():
        # Export/compile path: linspace specializes its length under symbolic shapes, so an
        # exported graph bakes the example H/W into the grid and a dynamic-shape TRT engine
        # then fails at every other resolution. arange stays symbolic; values match linspace
        # to float32 LSB. Eager keeps the cached linspace path below, bit-identical to before.
        h, w = tenFlow.shape[2], tenFlow.shape[3]
        tenHorizontal = (torch.arange(w, device=tenFlow.device, dtype=torch.float32)
                         * (2.0 / (w - 1)) - 1.0).view(1, 1, 1, w).expand(
            tenFlow.shape[0], -1, h, -1)
        tenVertical = (torch.arange(h, device=tenFlow.device, dtype=torch.float32)
                       * (2.0 / (h - 1)) - 1.0).view(1, 1, h, 1).expand(
            tenFlow.shape[0], -1, -1, w)
        grid = torch.cat([tenHorizontal, tenVertical], 1)
    else:
        k = (str(tenFlow.device), str(tenFlow.size()))
        if k not in backwarp_tenGrid:
            tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device).view(
                1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
            tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device).view(
                1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
            backwarp_tenGrid[k] = torch.cat(
                [tenHorizontal, tenVertical], 1).to(device)
        grid = backwarp_tenGrid[k]

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (grid + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)