"""AMD FidelityFX RCAS (Robust Contrast-Adaptive Sharpening), shared by the render engine
(gmfss_interp.py, the GUI "FSR" toggle / --sharpen) and the single-frame preview (preview.py), so the
preview shows exactly the sharpen a full render will apply. Moved verbatim out of gmfss_interp.py.
"""
import torch
from torch.nn import functional as F

RCAS_LIMIT = 0.1875 - 1e-6   # AMD FSR_RCAS_LIMIT = 0.25 - 1/16


def rcas(img, con):
    """AMD FidelityFX RCAS (Robust Contrast-Adaptive Sharpening) on a [1,3,H,W] float image in [0,1].

    This is the sharpen AMD FSR and Lossless Scaling's FSR mode use. RCAS limits its sharpening lobe
    to the 4-neighbour min/max (no overshoot or ringing) and attenuates it in noisy regions (the
    FSR_RCAS_DENOISE term), so it crisps real edges without amplifying fine texture/grain into mush
    even at high strength. The lobe is one
    scalar per pixel applied to all three channels, so it cannot decorrelate them into colour speckle.
    con in (0, 1] scales the strength (1 = AMD's max, sharpness 0); con <= 0 is a no-op.
    """
    p = F.pad(img, (1, 1, 1, 1), mode="replicate")        # replicate so edges don't sharpen to black
    b, h = p[..., 0:-2, 1:-1], p[..., 2:, 1:-1]           # up, down
    d, f = p[..., 1:-1, 0:-2], p[..., 1:-1, 2:]           # left, right
    e = img                                               # centre
    def luma(x): return x[:, 1:2] + 0.5 * (x[:, 0:1] + x[:, 2:3])   # green-weighted, as in FSR
    bL, dL, eL, fL, hL = luma(b), luma(d), luma(e), luma(f), luma(h)
    rng = torch.maximum(torch.maximum(torch.maximum(bL, dL), torch.maximum(fL, hL)), eL) \
        - torch.minimum(torch.minimum(torch.minimum(bL, dL), torch.minimum(fL, hL)), eL)
    nz = (0.25 * (bL + dL + fL + hL) - eL).abs() / (rng + 1e-4)
    nz = -0.5 * nz.clamp(0.0, 1.0) + 1.0                   # ~1 on flat areas, ->0.5 in noise
    mn4 = torch.minimum(torch.minimum(b, d), torch.minimum(f, h))
    mx4 = torch.maximum(torch.maximum(b, d), torch.maximum(f, h))
    hit_min = mn4 / (4.0 * mx4 + 1e-4)
    hit_max = (1.0 - mx4) / (4.0 * mn4 - 4.0 - 1e-4)       # denom strictly < 0: no divide-by-zero
    lobe = torch.maximum(-hit_min, hit_max).amax(dim=1, keepdim=True)   # most-limiting channel
    lobe = lobe.clamp(max=0.0).clamp(min=-RCAS_LIMIT) * con * nz
    return ((e + lobe * (b + d + f + h)) / (1.0 + 4.0 * lobe)).clamp(0.0, 1.0)
