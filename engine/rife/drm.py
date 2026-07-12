# DistanceRatioMap for DRBA timing, trimmed to the RIFE path from routineLife1/DRBA (MIT, see
# DRBA-LICENSE.txt). Upstream: models/drm.py + the distance_calculator helper from utils/tools.py;
# the GMFSS variants were dropped. Falls back to the pure-torch softsplat when cupy is unavailable
# (parity with upstream's check_cupy_env behaviour).
import torch

try:
    from rife.softsplat import softsplat as warp
except Exception:  # noqa: BLE001 - any cupy/NVRTC trouble: the torch fallback is exact, just slower
    from rife.softsplat_torch import softsplat as warp


def distance_calculator(_x):
    dtype = _x.dtype
    u, v = _x[:, 0:1].float(), _x[:, 1:].float()
    return torch.sqrt(u ** 2 + v ** 2).to(dtype)


def get_drm_t(drm, t, precision=1e-3):
    """Move the whole DRM tensor (values in (0,1), defined as sitting at timestep 0.5) toward
    timestep t by bisection, keeping every value's original proportion (see upstream docstring
    for the worked example)."""
    dtype = drm.dtype
    _x, b = 0.5, 0.5
    l, r = 0, 1
    # float is suggested for drm calculation to avoid overflow
    x_drm, b_drm = drm.float().clone(), drm.float().clone()
    l_drm, r_drm = x_drm.clone() * 0, x_drm.clone() * 0 + 1
    while abs(_x - t) > precision:
        if _x > t:
            r = _x
            _x = _x - (_x - l) * b
            r_drm = x_drm.clone()
            x_drm = x_drm - (x_drm - l_drm) * b_drm
        if _x < t:
            l = _x
            _x = _x + (r - _x) * b
            l_drm = x_drm.clone()
            x_drm = x_drm + (r_drm - x_drm) * b_drm
    return x_drm.to(dtype)


def calc_drm_rife(t, flow10, flow12, linear=False):
    # Compute the distance using the optical flow and distance calculator
    d10 = distance_calculator(flow10) + 1e-4
    d12 = distance_calculator(flow12) + 1e-4

    # Calculate the distance ratio map
    drm10 = d10 / (d10 + d12)
    drm12 = d12 / (d10 + d12)

    if linear:
        drm_t0_unaligned = drm10 * t * 2
        drm_t1_unaligned = drm12 * t * 2
    else:
        drm_t0_unaligned = get_drm_t(drm10, t)
        drm_t1_unaligned = get_drm_t(drm12, t)

    warp_method = 'avg'
    # For RIFE, drm must be aligned with the time of the intermediate frame; with the reversed
    # input order used by the backend (I1 first), the t1-aligned maps are the ones needed.
    drm_t1_t01 = warp(drm_t1_unaligned, flow10 * drm_t1_unaligned, None, warp_method)
    drm_t1_t12 = warp(drm_t0_unaligned, flow12 * drm_t0_unaligned, None, warp_method)

    ones_mask = drm10.clone() * 0 + 1

    mask_t1_t01 = warp(ones_mask, flow10 * drm_t1_unaligned, None, warp_method)
    mask_t1_t12 = warp(ones_mask, flow12 * drm_t0_unaligned, None, warp_method)

    gap_t1_t01 = mask_t1_t01 < 0.999
    gap_t1_t12 = mask_t1_t12 < 0.999

    drm_t1_t01[gap_t1_t01] = drm_t1_unaligned[gap_t1_t01]
    drm_t1_t12[gap_t1_t12] = drm_t0_unaligned[gap_t1_t12]

    return {
        "drm_t1_t01": drm_t1_t01,
        "drm_t1_t12": drm_t1_t12
    }
