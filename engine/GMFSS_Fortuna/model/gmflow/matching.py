import torch
import torch.nn.functional as F

from .geometry import coords_grid, generate_window_grid, normalize_coords


def global_correlation_softmax(feature0, feature1,
                               pred_bidir_flow=False,
                               ):
    # global correlation
    b, c, h, w = feature0.shape
    feature0 = feature0.view(b, c, -1).permute(0, 2, 1)  # [B, H*W, C]
    feature1 = feature1.view(b, c, -1)  # [B, C, H*W]

    correlation = torch.matmul(feature0, feature1).view(b, h, w, h, w) / (c ** 0.5)  # [B, H, W, H, W]

    # flow from softmax
    init_grid = coords_grid(b, h, w, device=correlation.device)  # [B, 2, H, W]
    grid = init_grid.view(b, 2, -1).permute(0, 2, 1)  # [B, H*W, 2]

    correlation = correlation.view(b, h * w, h * w)  # [B, H*W, H*W]

    if pred_bidir_flow:
        correlation = torch.cat((correlation, correlation.permute(0, 2, 1)), dim=0)  # [2*B, H*W, H*W]
        init_grid = init_grid.repeat(2, 1, 1, 1)  # [2*B, 2, H, W]
        grid = grid.repeat(2, 1, 1)  # [2*B, H*W, 2]
        b = b * 2

    prob = F.softmax(correlation, dim=-1)  # [B, H*W, H*W]

    correspondence = torch.matmul(prob, grid).view(b, h, w, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]

    # when predicting bidirectional flow, flow is the concatenation of forward flow and backward flow
    flow = correspondence - init_grid

    return flow, prob


# The per-pixel matmuls below run as batched gemms with batch count b*h*w. CUDA caps a
# kernel launch's gridDim.z at 65535, and TensorRT-RTX 1.6.1 (Myelin/cuDNN-graph) emits
# batched-gemm launches that hit that cap unguarded: any engine whose local-correlation
# grid exceeds 65535 pixels builds fine but every enqueueV3 fails with kernel-launch
# err 1 (the measured big-shape boundary: scale-1 grid 336x192 = 64512 clean, 208x368 =
# 76544 broken). Chunking the pixel dim keeps every gemm under the cap and is exact
# (each pixel's softmax is independent). Eager output is bit-identical (fp32 verified).
GEMM_BATCH_CHUNK = 61440


def local_correlation_softmax(feature0, feature1, local_radius,
                              padding_mode='zeros',
                              ):
    b, c, h, w = feature0.size()
    coords_init = coords_grid(b, h, w, device=feature0.device)  # [B, 2, H, W]
    coords = coords_init.view(b, 2, -1).permute(0, 2, 1)  # [B, H*W, 2]

    local_h = 2 * local_radius + 1
    local_w = 2 * local_radius + 1

    window_grid = generate_window_grid(-local_radius, local_radius,
                                       -local_radius, local_radius,
                                       local_h, local_w, device=feature0.device)  # [2R+1, 2R+1, 2]
    window_grid = window_grid.reshape(-1, 2).repeat(b, 1, 1, 1)  # [B, 1, (2R+1)^2, 2]
    sample_coords = coords.unsqueeze(-2) + window_grid  # [B, H*W, (2R+1)^2, 2]

    if torch.compiler.is_compiling():
        # export path (SMV 2026-09-21, ONNX-in-exe): the chunk loop below unrolls by H * W,
        # which a size-free export cannot hold. The same math on the whole grid, the two
        # per-pixel products as multiply + sum over one axis (no batched gemm, so no launch cap);
        # a different summation order: equivalent, measured closer to eager fp32 than the
        # chunked engine (harness\onnx\gmflow_check.py)
        valid = ((sample_coords[:, :, :, 0] >= 0) & (sample_coords[:, :, :, 0] < w)
                 & (sample_coords[:, :, :, 1] >= 0) & (sample_coords[:, :, :, 1] < h))
        window_feature = F.grid_sample(feature1, normalize_coords(sample_coords, h, w),
                                       padding_mode=padding_mode, align_corners=True
                                       )  # [B, C, H*W, (2R+1)^2]
        corr = (feature0.view(b, c, h * w, 1) * window_feature).sum(1) / (c ** 0.5)
        corr = corr.masked_fill(~valid, torch.finfo(corr.dtype).min / 2)
        prob = F.softmax(corr, -1)  # [B, H*W, (2R+1)^2]
        correspondence = (prob.unsqueeze(-1) * sample_coords).sum(-2)  # [B, H*W, 2]
        correspondence = correspondence.view(b, h, w, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]
        return correspondence - coords_init, None

    feature0_view = feature0.permute(0, 2, 3, 1).view(b, h * w, 1, c)  # [B, H*W, 1, C]

    hw = h * w
    step = max(1, GEMM_BATCH_CHUNK // b)
    outs = []
    for i in range(0, hw, step):
        sc = sample_coords[:, i:i + step]  # [B, chunk, (2R+1)^2, 2]
        # exclude coords that are out of image space
        valid = ((sc[:, :, :, 0] >= 0) & (sc[:, :, :, 0] < w)
                 & (sc[:, :, :, 1] >= 0) & (sc[:, :, :, 1] < h))
        # normalize coordinates to [-1, 1]
        window_feature = F.grid_sample(feature1, normalize_coords(sc, h, w),
                                       padding_mode=padding_mode, align_corners=True
                                       ).permute(0, 2, 1, 3)  # [B, chunk, C, (2R+1)^2]
        corr = torch.matmul(feature0_view[:, i:i + step], window_feature
                            ).view(b, -1, local_h * local_w) / (c ** 0.5)
        # mask invalid locations (finfo.min/2: -1e9 overflows a masked_fill in fp16)
        corr = corr.masked_fill(~valid, torch.finfo(corr.dtype).min / 2)
        prob = F.softmax(corr, -1)  # [B, chunk, (2R+1)^2]
        outs.append(torch.matmul(prob.unsqueeze(-2), sc).squeeze(-2))  # [B, chunk, 2]

    correspondence = torch.cat(outs, dim=1).view(b, h, w, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]

    flow = correspondence - coords_init

    # match_prob is no longer materialized whole (chunked away); no caller reads it
    return flow, None
