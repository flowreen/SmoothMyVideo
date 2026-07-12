"""NVIDIA Optical Flow FRUC (NvOFFRUC) driven from Python via a small CUDA bridge DLL.

This is the "NVIDIA Smooth Motion" interpolation model: hardware optical-flow frame interpolation on
the OFA engine, offered as the deliberately inferior / faster alternative to GMFSS. It is the same
class of technique as NVIDIA's driver-level Smooth Motion (OFA optical flow + blend), just exposed
offline through the Optical Flow SDK's FRUC library.

Like engine/rtxvideo.py this reaches NVIDIA's DLL through a tiny compiled C bridge
(engine/nvoffruc/nvoffruc_bridge.dll, built from build_src) because NvOFFRUC.dll must be loaded via
the SDK's signature-checked SecureLoadLibrary, not by ctypes directly. The bridge, NvOFFRUC.dll and
cudart64_110.dll all live together in engine/nvoffruc/; NvOFFRUC.dll + cudart are NVIDIA proprietary,
user-installed from their Optical Flow SDK download and gitignored, so this module degrades to
"unavailable" when they are absent (the GUI keeps the option locked, the CLI errors clearly).

Data path per interpolated frame mirrors the RTX bridge: torch [1,3,H,W] RGB float in [0,1] on CUDA
is quantised to uint8 and packed into a [H,W,4] BGRA buffer (FRUC's "ARGB" dword = little-endian
bytes B,G,R,A); its data_ptr goes straight to the bridge (zero copy). The bridge interpolates at an
arbitrary fraction t in (0,1) and writes a BGRA output we reorder back to a [1,3,oH,oW] RGB float
tensor. Interpolation runs at the source resolution, exactly where GMFSS runs, so the rest of the
pipe (upscale / RCAS / HDR / encode) is unchanged.
"""
import os
import ctypes

import torch

NVOFFRUC_DIR = os.environ.get(
    "SMV_NVOFFRUC_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "nvoffruc"))
_BRIDGE = "nvoffruc_bridge.dll"


def _add_cuda_search():
    """Put torch's lib dir (and any nvidia-*-cu* wheel bin dirs) on the Windows DLL search so the
    bridge and NvOFFRUC.dll resolve their cudart. Best effort; identical to rtxvideo.py."""
    base = os.path.dirname(os.path.abspath(torch.__file__))
    dirs = [os.path.join(base, "lib")]
    nv = os.path.join(os.path.dirname(base), "nvidia")
    if os.path.isdir(nv):
        dirs += [os.path.join(nv, sub, "bin") for sub in os.listdir(nv)]
    for d in dirs:
        if os.path.isdir(d):
            try:
                os.add_dll_directory(d)
            except OSError:
                pass


def bridge_path(nvoffruc_dir=None):
    return os.path.join(nvoffruc_dir or NVOFFRUC_DIR, _BRIDGE)


def _load(nvoffruc_dir):
    dll = bridge_path(nvoffruc_dir)
    if not os.path.isfile(dll):
        raise FileNotFoundError(f"NvOFFRUC bridge not found: {dll} (build engine/nvoffruc/build_src)")
    _add_cuda_search()
    os.add_dll_directory(nvoffruc_dir)     # bridge dll + NvOFFRUC.dll + cudart64_110.dll live here
    lib = ctypes.CDLL(dll)                  # cdecl; on x64 the ABI is uniform
    cvp, I, U, D = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint, ctypes.c_double
    lib.nvoffruc_last_error.restype = ctypes.c_char_p
    lib.nvoffruc_probe.restype = I
    lib.nvoffruc_create.argtypes = [U, U]
    lib.nvoffruc_create.restype = I
    lib.nvoffruc_interpolate.argtypes = [cvp, cvp, cvp, D, ctypes.POINTER(I)]
    lib.nvoffruc_interpolate.restype = I
    lib.nvoffruc_reset.restype = I
    lib.nvoffruc_destroy.restype = None
    return lib


def available(nvoffruc_dir=None):
    """True when the bridge is built AND NvOFFRUC.dll is installed beside it and NVIDIA-signed.
    Cheap enough for the GUI 'fruc-ready' probe; never raises."""
    nvoffruc_dir = nvoffruc_dir or NVOFFRUC_DIR
    if not os.path.isfile(bridge_path(nvoffruc_dir)):
        return False
    try:
        return bool(_load(nvoffruc_dir).nvoffruc_probe())
    except OSError:
        return False


class NvOFFRUC:
    """One FRUC instance for a fixed frame size. interpolate() returns a fresh RGB float tensor for
    each requested fraction t; the caller drives the same _pair_fracs grid it uses for GMFSS."""

    def __init__(self, width, height, nvoffruc_dir=None):
        self._lib = _load(nvoffruc_dir or NVOFFRUC_DIR)
        self.w, self.h = int(width), int(height)
        self.repeats = 0                    # count of frames FRUC fell back to repeating (low conf.)
        dev = "cuda"
        # Persistent ARGB (BGRA dword) surfaces, allocated BEFORE create so their device pointers can
        # be registered with FRUC once and reused: prev/cur are filled in place each call, out is read.
        self._a = torch.empty((self.h, self.w, 4), dtype=torch.uint8, device=dev)
        self._b = torch.empty((self.h, self.w, 4), dtype=torch.uint8, device=dev)
        self._o = torch.empty((self.h, self.w, 4), dtype=torch.uint8, device=dev)
        self._a[..., 3] = 255; self._b[..., 3] = 255
        self._flag = ctypes.c_int(0)
        rc = self._lib.nvoffruc_create(self.w, self.h)
        if rc != 0:
            raise RuntimeError(f"nvoffruc_create failed: {self._err()} (rc {rc})")

    def _err(self):
        return (self._lib.nvoffruc_last_error() or b"").decode("utf-8", "replace")

    def _pack(self, img, dst):
        """[1,3,H,W] RGB float in [0,1] -> BGRA uint8 into dst (in place)."""
        x = (img[0].clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)   # [3,H,W] RGB
        dst[..., 0] = x[2]; dst[..., 1] = x[1]; dst[..., 2] = x[0]     # B, G, R (A already 255)
        return dst

    def interpolate(self, I0, I1, t):
        """Interpolate the frame at fraction t in (0,1) between RGB float frames I0 and I1
        ([1,3,H,W], [0,1], CUDA). Returns a new [1,3,H,W] RGB float tensor."""
        self._pack(I0, self._a)                                        # ARGB staging surfaces the
        self._pack(I1, self._b)                                        # bridge copies into FRUC's own
        # Cross-context sync fences live IN THE BRIDGE (cuCtxSynchronize at entry and before return,
        # added 2026-07-11): CUDA does not order the bridge context's null-stream copies against
        # torch's streams, and unfenced every tween came out sliced at horizontal seams under torch
        # 2.13 (partially-written surfaces; torch 2.12 won the race by luck). Nothing to do here -
        # but never pair this wrapper with a pre-2026-07-11 nvoffruc_bridge.dll.
        rc = self._lib.nvoffruc_interpolate(
            ctypes.c_void_p(self._a.data_ptr()),
            ctypes.c_void_p(self._b.data_ptr()),
            ctypes.c_void_p(self._o.data_ptr()),
            ctypes.c_double(float(t)), ctypes.byref(self._flag))
        if rc != 0:
            raise RuntimeError(f"nvoffruc_interpolate failed: {self._err()} (rc {rc})")
        if self._flag.value:
            self.repeats += 1
        o = self._o.to(torch.float32) / 255.0                          # [H,W,4] BGRA
        # Clean Smooth Motion: return NvOFFRUC's warp verbatim, no post-processing. (An earlier
        # neighbourhood clamp that muted out-of-range warp colour was removed by request. It altered
        # every frame, and the artifacts it targeted - ghosts and rainbow bands - come from NvOFFRUC's
        # optical flow, which post-processing cannot actually fix; verified against philipl's FFmpeg
        # baseline, which produces the same result. GMFSS is the quality path for hard motion.)
        return torch.stack((o[..., 2], o[..., 1], o[..., 0]), dim=0).unsqueeze(0)  # BGRA -> [1,3,H,W] RGB

    def reset(self):
        """Clear FRUC's internal OFA temporal-hint state (recreate the handle, keep surfaces).

        DO NOT USE in the streaming render path: on current drivers (found 2026-07-11) every
        interpolate() after a reset collapses to an exact copy of I1 (MAD 0.0) instead of warping -
        the engine feeds strictly consecutive pairs, so hints are always valid and no reset is ever
        needed there. Kept only for experiments with non-consecutive pairs (the old bisection use),
        where its output must be re-validated first."""
        rc = self._lib.nvoffruc_reset()
        if rc != 0:
            raise RuntimeError(f"nvoffruc_reset failed: {self._err()} (rc {rc})")

    def close(self):
        if getattr(self, "_lib", None) is not None:
            self._lib.nvoffruc_destroy()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
