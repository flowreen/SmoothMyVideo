import torch
import torch.nn.functional as F


def coords_grid(b, h, w, homogeneous=False, device=None):
    # build on the target device: CPU-alloc + .to() breaks CUDA graph capture
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device),
                          indexing="ij")  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid


def generate_window_grid(h_min, h_max, w_min, w_max, len_h, len_w, device=None):
    assert device is not None

    x, y = torch.meshgrid([torch.linspace(w_min, w_max, len_w, device=device),
                           torch.linspace(h_min, h_max, len_h, device=device)],
                          indexing="ij")
    grid = torch.stack((x, y), -1).transpose(0, 1).float()  # [H, W, 2]

    return grid


_center_cache = {}  # (h, w, device) -> c; creating from Python floats is a CPU->GPU
# copy either way, which breaks CUDA graph capture, so cache it (warmup populates)


def normalize_coords(coords, h, w):
    # coords: [B, H, W, 2]
    # FakeTensors (ONNX-export traces) bypass the cache: a cached fake poisons later real
    # eager calls at the same shape (same class of bug as MetricNet.backwarp, 2026-08-28)
    if torch.compiler.is_compiling():
        # export path (SMV 2026-09-21, ONNX-in-exe): torch.tensor() from symbolic sizes would pin
        # the size, so the same (coords - c) / c per channel from scalar arithmetic
        cx, cy = (w - 1) / 2., (h - 1) / 2.
        return torch.cat(((coords[..., 0:1] - cx) / cx, (coords[..., 1:2] - cy) / cy), -1)
    if isinstance(coords, torch._subclasses.fake_tensor.FakeTensor):
        c = torch.tensor([(w - 1) / 2., (h - 1) / 2.], dtype=torch.float32, device=coords.device)
        return (coords - c) / c
    key = (h, w, coords.device)
    c = _center_cache.get(key)
    if c is None:
        c = torch.tensor([(w - 1) / 2., (h - 1) / 2.], dtype=torch.float32, device=coords.device)
        _center_cache[key] = c
    return (coords - c) / c  # [-1, 1]


def bilinear_sample(img, sample_coords, mode='bilinear', padding_mode='zeros', return_mask=False):
    # img: [B, C, H, W]
    # sample_coords: [B, 2, H, W] in image scale
    if sample_coords.size(1) != 2:  # [B, H, W, 2]
        sample_coords = sample_coords.permute(0, 3, 1, 2)

    b, _, h, w = sample_coords.shape

    # Normalize to [-1, 1]
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1

    grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, 2]

    img = F.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=True)

    if return_mask:
        mask = (x_grid >= -1) & (y_grid >= -1) & (x_grid <= 1) & (y_grid <= 1)  # [B, H, W]

        return img, mask

    return img


def flow_warp(feature, flow, mask=False, padding_mode='zeros'):
    b, c, h, w = feature.size()
    assert flow.size(1) == 2

    grid = coords_grid(b, h, w, device=flow.device) + flow  # [B, 2, H, W]

    return bilinear_sample(feature, grid, padding_mode=padding_mode,
                           return_mask=mask)


def forward_backward_consistency_check(fwd_flow, bwd_flow,
                                       alpha=0.01,
                                       beta=0.5
                                       ):
    # fwd_flow, bwd_flow: [B, 2, H, W]
    # alpha and beta values are following UnFlow (https://arxiv.org/abs/1711.07837)
    assert fwd_flow.dim() == 4 and bwd_flow.dim() == 4
    assert fwd_flow.size(1) == 2 and bwd_flow.size(1) == 2
    flow_mag = torch.norm(fwd_flow, dim=1) + torch.norm(bwd_flow, dim=1)  # [B, H, W]

    warped_bwd_flow = flow_warp(bwd_flow, fwd_flow)  # [B, 2, H, W]
    warped_fwd_flow = flow_warp(fwd_flow, bwd_flow)  # [B, 2, H, W]

    diff_fwd = torch.norm(fwd_flow + warped_bwd_flow, dim=1)  # [B, H, W]
    diff_bwd = torch.norm(bwd_flow + warped_fwd_flow, dim=1)

    threshold = alpha * flow_mag + beta

    fwd_occ = (diff_fwd > threshold).float()  # [B, H, W]
    bwd_occ = (diff_bwd > threshold).float()

    return fwd_occ, bwd_occ
