"""NVIDIA RTX Video SDK (VSR / TrueHDR) driven from Python via a small CUDA bridge DLL.

The RTX Video SDK is NGX based and its entry points live in a static import lib, not a DLL, so it
cannot be reached by ctypes alone (a C export shim is needed). engine/rtxvideo/
therefore ships a tiny compiled bridge, rtxvideo_cuda.dll, built from NVIDIA's
rtx_video_api_cuda_impl.cpp (the SDK's CUDA convenience layer) plus a path shim. It exports the
C functions used here. The two NGX feature DLLs (nvngx_vsr.dll, nvngx_truehdr.dll) sit beside it
in the same folder, which is how NGX locates them (it resolves them relative to the loading
module, so this works regardless of the process working directory).

Data path per frame (VSR): a torch [1,3,H,W] RGB float tensor in [0,1] on CUDA is quantised to
uint8 and packed into a [H,W,4] BGRA buffer (the SDK's "ARGB" dword = little-endian bytes
B,G,R,A), whose data_ptr is passed straight into rtx_video_api_cuda_evaluate_deviceptr (zero copy
the same way trt_runtime.py binds torch tensors into TensorRT). VSR upscales it into an arbitrary
[oH,oW,4] BGRA output rectangle, which is reordered back to a [1,3,oH,oW] RGB float tensor. The
output resolution is unrestricted: probing showed clean, crash-free upscales to any aspect-
preserving target well past 8K (16K worked on a 24 GB GPU), so the caller picks an exact target
resolution rather than an integer 2x/3x/4x multiple. The bridge shares torch's primary CUDA
context (created with cuContext=NULL), so no separate context is made.

The feature DLLs are NVIDIA proprietary and not redistributable, so engine/rtxvideo/ is
gitignored and absent from a fresh clone; construction raises if the bridge or feature DLLs are
missing and the caller falls back to bicubic.
"""
import os
import ctypes

import torch

RTX_DIR = os.environ.get(
    "SMV_RTXVIDEO_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "rtxvideo"))

_API_SUCCESS = 1


class _RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_uint32), ("top", ctypes.c_uint32),
                ("right", ctypes.c_uint32), ("bottom", ctypes.c_uint32)]


class _VSR(ctypes.Structure):
    _fields_ = [("QualityLevel", ctypes.c_uint32)]            # 0 Bicubic .. 4 Ultra


class _THDR(ctypes.Structure):
    _fields_ = [("Contrast", ctypes.c_uint32), ("Saturation", ctypes.c_uint32),
                ("MiddleGray", ctypes.c_uint32), ("MaxLuminance", ctypes.c_uint32)]


def _add_cuda_search():
    """Put torch's lib dir and the nvidia-*-cu12 wheel bin dirs on the Windows DLL search so the
    bridge can resolve cudart64_12.dll (the driver nvcuda.dll lives in System32). Best effort."""
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


def _load(rtx_dir):
    dll = os.path.join(rtx_dir, "rtxvideo_cuda.dll")
    if not os.path.isfile(dll):
        raise FileNotFoundError(f"RTX Video bridge not found: {dll}")
    if not os.path.isfile(os.path.join(rtx_dir, "nvngx_vsr.dll")):
        raise FileNotFoundError(f"nvngx_vsr.dll missing in {rtx_dir}")
    _add_cuda_search()
    os.add_dll_directory(rtx_dir)          # bridge dll + the two NGX feature dlls live here
    lib = ctypes.CDLL(dll)                  # cdecl; on x64 the ABI is uniform
    cvp, I, U = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint
    lib.rtxv_set_model_path.argtypes = [ctypes.c_wchar_p]
    lib.rtxv_set_model_path.restype = None
    lib.rtx_video_api_cuda_create.argtypes = [cvp, cvp, I, U, U]
    lib.rtx_video_api_cuda_create.restype = U
    lib.rtx_video_api_cuda_evaluate_deviceptr.argtypes = [
        cvp, cvp, _RECT, _RECT, ctypes.POINTER(_VSR), ctypes.POINTER(_THDR)]
    lib.rtx_video_api_cuda_evaluate_deviceptr.restype = U
    lib.rtx_video_api_cuda_shutdown.restype = None
    return lib


class RTXVideo:
    """A loaded RTX Video feature (VSR / TrueHDR) bound to one input resolution and an arbitrary
    output resolution (out_w x out_h). VSR places no restriction on the output rectangle, so the
    caller passes the exact target dimensions (out = input when not upscaling). Raises on any setup
    failure so the caller can fall back to bicubic. run_vsr() / run_hdr() are per frame."""

    def __init__(self, width, height, out_w, out_h, vsr=True, hdr=False, vsr_quality=4,
                 hdr_max_nits=1000, rtx_dir=RTX_DIR):
        # Make sure torch's primary CUDA context exists and is current on this thread before the
        # bridge retains it (create is called with cuContext=NULL -> cuDevicePrimaryCtxRetain).
        torch.zeros(8, device="cuda")
        torch.cuda.synchronize()

        self.lib = _load(rtx_dir)
        self.lib.rtxv_set_model_path(rtx_dir)
        self.W, self.H = width, height
        self.oW, self.oH = (out_w, out_h) if vsr else (width, height)
        self.vsr, self.hdr, self.vsr_q = vsr, hdr, vsr_quality

        r = self.lib.rtx_video_api_cuda_create(None, None, 0, int(bool(hdr)), int(bool(vsr)))
        if r != _API_SUCCESS:
            raise RuntimeError("rtx_video_api_cuda_create failed (VSR/TrueHDR unavailable on this "
                               "GPU/driver, or feature DLLs not found)")

        # Reused GPU staging buffers. Input is packed BGRA uint8 (pitch = 4*W); alpha is constant.
        self._src = torch.empty((height, width, 4), dtype=torch.uint8, device="cuda")
        self._src[..., 3] = 255
        self._dst = torch.empty((self.oH, self.oW, 4), dtype=torch.uint8, device="cuda")
        self._vsr = _VSR(int(vsr_quality))
        # TrueHDR knobs: Contrast/Saturation 100 = neutral, MiddleGray 50, MaxLuminance = the HDR10
        # target peak in nits (1000 is a standard mastering peak; the SDK allows 400..2000).
        self._thdr = _THDR(100, 100, 50, max(400, min(2000, int(hdr_max_nits))))

    def run_vsr(self, rgb):
        """Upscale a [1,3,H,W] RGB float tensor in [0,1] on CUDA; returns [1,3,oH,oW] RGB float."""
        t8 = (rgb[0].clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)   # [3,H,W] planes R,G,B
        self._src[..., 0] = t8[2]      # B
        self._src[..., 1] = t8[1]      # G
        self._src[..., 2] = t8[0]      # R
        torch.cuda.synchronize()        # finish the packing (default stream) before the bridge reads
        r = self.lib.rtx_video_api_cuda_evaluate_deviceptr(
            ctypes.c_void_p(self._src.data_ptr()), ctypes.c_void_p(self._dst.data_ptr()),
            _RECT(0, 0, self.W, self.H), _RECT(0, 0, self.oW, self.oH),
            ctypes.byref(self._vsr), ctypes.byref(self._thdr))
        torch.cuda.synchronize()        # finish the eval before torch reads _dst
        if r != _API_SUCCESS:
            raise RuntimeError("rtx_video_api_cuda_evaluate_deviceptr failed")
        d = self._dst                   # [oH,oW,4] bytes B,G,R,A
        return torch.stack([d[..., 2], d[..., 1], d[..., 0]], dim=0).float().div_(255.0).unsqueeze(0)

    def run_hdr(self, rgb):
        """SDR to HDR (and optional VSR, when this instance was created with vsr=True) on a
        [1,3,H,W] RGB float tensor in [0,1] on CUDA. Returns the packed 10:10:10:2 bytes at (oH,oW),
        which are ffmpeg's x2rgb10le (B in the low 10 bits, matching the model's channel-0=blue):
        PQ-encoded, BT.2020 primaries, full-range RGB10 (the DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020
        the SDK viewer uses for ABGR10). The caller feeds these to ffmpeg as -pix_fmt x2rgb10le and
        tags the stream HDR10."""
        t8 = (rgb[0].clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)   # [3,H,W] planes R,G,B
        self._src[..., 0] = t8[2]      # B
        self._src[..., 1] = t8[1]      # G
        self._src[..., 2] = t8[0]      # R
        torch.cuda.synchronize()
        r = self.lib.rtx_video_api_cuda_evaluate_deviceptr(
            ctypes.c_void_p(self._src.data_ptr()), ctypes.c_void_p(self._dst.data_ptr()),
            _RECT(0, 0, self.W, self.H), _RECT(0, 0, self.oW, self.oH),
            ctypes.byref(self._vsr), ctypes.byref(self._thdr))
        torch.cuda.synchronize()
        if r != _API_SUCCESS:
            raise RuntimeError("rtx_video_api_cuda_evaluate_deviceptr (TrueHDR) failed")
        return self._dst.cpu().numpy().tobytes()   # [oH,oW,4] packed 10:10:10:2 == x2rgb10le

    def close(self):
        try:
            self.lib.rtx_video_api_cuda_shutdown()
        except Exception:  # noqa: BLE001 - best-effort, process is exiting anyway
            pass
