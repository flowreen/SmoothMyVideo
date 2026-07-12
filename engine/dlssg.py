"""DLSS 4.5 Frame Generation driven from Python via a bare-bones D3D12 host process.

This is the "DLSS 4.5" interpolation model: NVIDIA's game Frame Generation (Streamline
sl.dlss_g) coerced into offline two-frame video interpolation. DLSS-FG has no offline API -
it only exists inside a game's D3D12 swap chain presentation loop - so engine/dlssg/dlssg2f.exe
IS that loop: a minimal host that presents each source frame to a real (offscreen) swap chain,
satisfies the SDK contract (flat depth + zero motion-vector tags, Reflex/PCL markers, per-frame
camera constants), runs with DLSSGFlags::eShowOnlyInterpolatedFrame so only the AI-generated
in-between frame reaches the native swap chain, and reads it straight back from the last
presented buffer (verified bit-identical to a screen capture; no screen dependency).

Unlike FRUC there is no arbitrary-t API: DLSS-FG generates exactly the evenly spaced frame(s)
between consecutive presents - 1 to 5 of them (multi-frame generation), i.e. 2x to 6x - so this
model is on-grid integer multipliers only. The host is spawned once per render (`--server W H
--gen N`, raw RGBA8 frames on stdin, the N generated frames per source frame on stdout, in
temporal order) and fed the source stream in presentation order; state is temporal, so frames
must arrive strictly consecutively (the on-grid loop guarantees that). The 5-frame ceiling is
the SDK's; the per-GPU limit (numFramesToGenerateMax, e.g. 1 on RTX 40 series without MFG) is
reported in the handshake and enforced with a clear error.

Everything the model needs ships with the app in engine/dlssg/ (the host exe + NVIDIA's
redistributable Streamline runtime: sl.interposer/sl.common/sl.dlss_g/sl.pcl/sl.reflex +
nvngx_dlssg, ~10 MB, licenses alongside). Runtime support still requires an RTX 40/50 GPU,
a recent driver, and Windows hardware-accelerated GPU scheduling; the host reports a clear
error on stderr when any of those are missing.
"""
import os
import subprocess

import numpy as np
import torch

DLSSG_DIR = os.environ.get(
    "SMV_DLSSG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "dlssg"))
_EXE = "dlssg2f.exe"
_RUNTIME = ("sl.interposer.dll", "sl.common.dll", "sl.dlss_g.dll",
            "sl.pcl.dll", "sl.reflex.dll", "nvngx_dlssg.dll")


def exe_path(dlssg_dir=None):
    return os.path.join(dlssg_dir or DLSSG_DIR, _EXE)


def available(dlssg_dir=None):
    """True when the host exe and the full Streamline runtime are present (they are bundled, so
    this is normally always true; a False means a broken install). GPU/driver/OS support is only
    knowable by actually starting the host - the GUI probe stays a file check, the render errors
    clearly. Never raises."""
    d = dlssg_dir or DLSSG_DIR
    return (os.path.isfile(exe_path(d))
            and all(os.path.isfile(os.path.join(d, f)) for f in _RUNTIME))


class DLSSG:
    """One DLSS-FG host process for a fixed frame size and multiplier. interpolate(a, b) returns
    the gen_frames generated frames (evenly spaced, temporal order) as fresh [1,3,H,W] RGB float
    tensors. Frames MUST be fed as strictly consecutive pairs (b of one call is a of the next),
    mirroring how a game presents; the wrapper keeps the host's input stream in presentation
    order and skips re-sending a frame it already presented."""

    def __init__(self, width, height, gen_frames=1, dlssg_dir=None):
        d = dlssg_dir or DLSSG_DIR
        exe = exe_path(d)
        if not os.path.isfile(exe):
            raise FileNotFoundError(f"DLSS host not found: {exe}")
        self.w, self.h = int(width), int(height)
        self.gen = int(gen_frames)
        self._nbytes = self.w * self.h * 4
        # stderr inherits ours, so the host's "DLSS-G ready: SL x.y NGX model ..." line and any
        # SDK warnings land in the engine log the GUI already streams.
        self._proc = subprocess.Popen(
            [exe, "--server", str(self.w), str(self.h), "--gen", str(self.gen)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, cwd=d)
        line = self._proc.stdout.readline()  # handshake, printed only after the SDK is fully up
        if not line.startswith(b"DLSSG READY"):
            rc = self._proc.wait(timeout=5)
            if rc == 3:  # host: requested multiplier beyond numFramesToGenerateMax for this GPU
                raise RuntimeError(
                    f"this GPU does not support {self.gen + 1}x DLSS multi-frame generation "
                    "(RTX 50 series does up to 6x, RTX 40 series only 2x); pick a lower "
                    "multiplier or GMFSS")
            raise RuntimeError(
                "DLSS Frame Generation host failed to start"
                + (" (GPU/driver/OS unsupported)" if rc == 2 else f" (exit {rc})")
                + "; it needs an RTX 40/50 GPU, a recent driver and Windows HW-accelerated "
                  "GPU scheduling ON. See the engine log for the SDK's own error.")
        self._last = None      # the tensor object of the most recently presented frame
        self._alpha = torch.full((self.h, self.w, 1), 255, dtype=torch.uint8, device="cuda")

    def _send(self, img):
        """[1,3,H,W] RGB float in [0,1] (CUDA) -> raw RGBA8 frame on the host's stdin."""
        x = (img[0].clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)         # [3,H,W]
        rgba = torch.cat((x.permute(1, 2, 0), self._alpha), dim=2)           # [H,W,4] RGBA
        self._proc.stdin.write(rgba.contiguous().cpu().numpy().tobytes())
        self._proc.stdin.flush()

    def _recv(self):
        buf = self._proc.stdout.read(self._nbytes)
        if len(buf) < self._nbytes:
            rc = self._proc.poll()
            if rc == 3:
                raise RuntimeError(
                    f"this GPU does not support {self.gen + 1}x DLSS multi-frame generation "
                    "(RTX 50 series does up to 6x, RTX 40 series only 2x); pick a lower "
                    "multiplier or GMFSS")
            raise RuntimeError(f"DLSS host died mid-render (exit {rc}); see the engine log")
        o = torch.from_numpy(np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 4).copy())
        o = o.to("cuda")[..., :3].permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return o

    def interpolate(self, I0, I1):
        """The gen_frames DLSS-FG generated frames between consecutive frames I0 and I1, as a
        list in temporal order (evenly spaced at j/(gen+1))."""
        if self._last is not I0:
            # first pair, or a stream discontinuity (e.g. resume): present I0 first. The host
            # emits gen_frames per present after its warmup; when I0 is mid-stream the frames it
            # produces belong to the previous (unrelated) pair, so read them and drop them.
            mid_stream = self._last is not None
            self._send(I0)
            if mid_stream:
                for _ in range(self.gen):
                    self._recv()
        self._send(I1)
        self._last = I1
        return [self._recv() for _ in range(self.gen)]

    def close(self):
        p = getattr(self, "_proc", None)
        if p is not None and p.poll() is None:
            try:
                p.stdin.close()
                p.wait(timeout=5)
            except Exception:
                p.kill()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
