"""Real-ESRGAN anime-video detail restoration for the --restore pass.

SRVGGNetCompact is vendored VERBATIM from Real-ESRGAN (realesrgan/archs/srvgg_arch.py,
https://github.com/xinntao/Real-ESRGAN, BSD 3-Clause (c) 2021 Xintao Wang - see
REALESRGAN-LICENSE.txt next to this file) with only the basicsr ARCH_REGISTRY dependency
removed, so the bundled realesr-animevideov3 weights load key-for-key. That model is the
project's official anime VIDEO variant: a compact 16-conv VGG body whose convolutions all run
at the SOURCE resolution (only the final PixelShuffle emits the 4x image), so it is cheap
enough to run per output frame. load() builds it from the weights bundled next to this file
(state dict under 'params'; shapes verified: 64 feat, 16 convs, PReLU, 4x = 48-channel last
conv) and returns an eval-mode fp16 CUDA module.
"""
import os

import torch
from torch import nn as nn
from torch.nn import functional as F

WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "realesr-animevideov3.pth")


def weights_hash():
    """Short content fingerprint of the bundled weights, for the TRT engine cache name (a
    weight swap must be a cache miss, mirroring trt_runtime's WEIGHTS_TAG scheme)."""
    import hashlib
    h = hashlib.md5()
    with open(WEIGHTS, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


class SRVGGNetCompact(nn.Module):
    """A compact VGG-style network structure for super-resolution.

    It is a compact network structure, which performs upsampling in the last layer and no convolution is
    conducted on the HR feature space.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_conv (int): Number of convolution layers in the body network. Default: 16.
        upscale (int): Upsampling factor. Default: 4.
        act_type (str): Activation type, options: 'relu', 'prelu', 'leakyrelu'. Default: prelu.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'):
        super(SRVGGNetCompact, self).__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        self.body = nn.ModuleList()
        # the first conv
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        # the first activation
        if act_type == 'relu':
            activation = nn.ReLU(inplace=True)
        elif act_type == 'prelu':
            activation = nn.PReLU(num_parameters=num_feat)
        elif act_type == 'leakyrelu':
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)

        # the body structure
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            # activation
            if act_type == 'relu':
                activation = nn.ReLU(inplace=True)
            elif act_type == 'prelu':
                activation = nn.PReLU(num_parameters=num_feat)
            elif act_type == 'leakyrelu':
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.body.append(activation)

        # the last conv
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        # upsample
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for i in range(0, len(self.body)):
            out = self.body[i](out)

        out = self.upsampler(out)
        # add the nearest upsampled image, so that the network learns the residual
        base = F.interpolate(x, scale_factor=self.upscale, mode='nearest')
        out += base
        return out


def fit(out, oh, ow):
    """Resize the net's 4x output to (oh, ow): box filter for exact integer downscales (the
    correct antialias for them, and cheapest), area for other downscales, bicubic only when
    upscaling past 4x. Shared by the render engine (_restore) and the preview pane so the
    pane shows exactly what a render produces."""
    h, w = out.shape[-2], out.shape[-1]
    if (oh, ow) == (h, w):
        return out
    if oh < h and h % oh == 0 and w % ow == 0 and h // oh == w // ow:
        return F.avg_pool2d(out, h // oh)
    if oh < h:
        return F.interpolate(out, size=(oh, ow), mode="area")
    return F.interpolate(out, size=(oh, ow), mode="bicubic", align_corners=False).clamp(0.0, 1.0)


def load(device):
    """The bundled realesr-animevideov3 as an eval fp16 module on `device`.

    fp16 is explicit (weights + input halved by the caller) rather than autocast so the pass
    behaves identically whether or not the caller sits inside an autocast region; the compact
    conv/PReLU/PixelShuffle stack is fp16-safe.
    """
    sd = torch.load(WEIGHTS, map_location="cpu", weights_only=True)["params"]
    net = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16,
                          upscale=4, act_type="prelu")
    net.load_state_dict(sd, strict=True)
    return net.eval().half().to(device)
