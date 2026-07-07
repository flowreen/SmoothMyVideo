"""NVIDIA RTX Video SDK (VSR / TrueHDR) driven from Python via a small CUDA bridge DLL.

The RTX Video SDK is NGX based and its entry points live in a static import lib, not a DLL, so it
cannot be reached by ctypes alone (a C export shim is needed). engine/rtxvideo/
therefore ships a tiny compiled bridge, rtxvideo_cuda.dll, built from NVIDIA's
rtx_video_api_cuda_impl.cpp (the SDK's CUDA convenience layer) plus a path shim. It exports the
C functions used here. The two NGX feature DLLs (nvngx_vsr.dll, nvngx_truehdr.dll) sit beside it
in the same folder, which is how NGX locates them (it resolves them relative to the loading
module, so this works regardless of the process working directory).

VSR and TrueHDR run as SEPARATE evals (rtx_video_api_cuda_evaluate_vsr_deviceptr and
..._thdr_deviceptr), not the SDK's fused VSR->THDR pass, so the engine can apply its RCAS sharpen
between them at the output resolution (the max-quality order: upscale the clean frame, sharpen at
final res, then expand to HDR). run_vsr does VSR only and returns an SDR float tensor; run_hdr does
TrueHDR only on an already-final-resolution SDR frame and returns packed 10-bit.

Data path per frame: a torch [1,3,H,W] RGB float tensor in [0,1] on CUDA is quantised to uint8 and
packed into a [H,W,4] BGRA buffer (the SDK's "ARGB" dword = little-endian bytes B,G,R,A), whose
data_ptr is passed straight into the bridge (zero copy the same way trt_runtime.py binds torch
tensors into TensorRT). VSR upscales it into an arbitrary [oH,oW,4] BGRA output rectangle, reordered
back to a [1,3,oH,oW] RGB float tensor. The output resolution is unrestricted: probing showed clean,
crash-free upscales to any aspect-preserving target well past 8K (16K worked on a 24 GB GPU), so the
caller picks an exact target resolution rather than an integer 2x/3x/4x multiple. The bridge shares
torch's primary CUDA context (created with cuContext=NULL), so no separate context is made.

The feature DLLs are NVIDIA proprietary and not redistributable, so engine/rtxvideo/ is
gitignored and absent from a fresh clone; construction raises if the bridge or feature DLLs are
missing and the caller falls back to bicubic.
"""
import os
import ctypes
import math

import torch

RTX_DIR = os.environ.get(
    "SMV_RTXVIDEO_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "rtxvideo"))

_API_SUCCESS = 1

# Vibrance reference chroma: the ICtCp chroma magnitude treated as "fully saturated" (no boost).
# Measured on sample content: mean scene chroma ~0.04, saturated anime fills ~0.10-0.15.
_VIBRANCE_CREF = 0.12

# SMPTE ST 2084 (PQ) EOTF constants, used to measure MaxCLL/MaxFALL from the TrueHDR output so the
# HDR10 content-light metadata reflects the actual frames rather than a guess.
_PQ_M1, _PQ_M2 = 0.1593017578125, 78.84375
_PQ_C1, _PQ_C2, _PQ_C3 = 0.8359375, 18.8515625, 18.6875


def _pq_to_linear(e):
    """SMPTE ST 2084 EOTF: PQ code' in [0,1] -> display-linear in [0,1] (1.0 == 10000 nits)."""
    ep = e.clamp(0.0, 1.0).pow(1.0 / _PQ_M2)
    return ((ep - _PQ_C1).clamp(min=0.0) / (_PQ_C2 - _PQ_C3 * ep).clamp(min=1e-6)).pow(1.0 / _PQ_M1)


def _linear_to_pq(lin):
    """SMPTE ST 2084 inverse EOTF: display-linear in [0,1] (1.0 == 10000 nits) -> PQ code' in [0,1]."""
    lm = lin.clamp(min=0.0).pow(_PQ_M1)
    return ((_PQ_C1 + _PQ_C2 * lm) / (1.0 + _PQ_C3 * lm)).pow(_PQ_M2)


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
    # Split single-feature entries: VSR only (8-bit in/out) and TrueHDR only (8-bit in, packed
    # 10-bit out), so a sharpen pass can run between them at the output resolution.
    lib.rtx_video_api_cuda_evaluate_vsr_deviceptr.argtypes = [
        cvp, cvp, _RECT, _RECT, ctypes.POINTER(_VSR)]
    lib.rtx_video_api_cuda_evaluate_vsr_deviceptr.restype = U
    lib.rtx_video_api_cuda_evaluate_thdr_deviceptr.argtypes = [
        cvp, cvp, _RECT, _RECT, ctypes.POINTER(_THDR)]
    lib.rtx_video_api_cuda_evaluate_thdr_deviceptr.restype = U
    lib.rtx_video_api_cuda_shutdown.restype = None
    return lib


class RTXVideo:
    """A loaded RTX Video feature set (VSR and/or TrueHDR) bound to one input resolution (width x
    height) and the final output resolution (out_w x out_h). out_w/out_h is the size VSR upscales TO
    and the size TrueHDR runs AT; pass out == in when not upscaling. VSR places no restriction on the
    output rectangle. Raises on any setup failure so the caller can fall back to bicubic / SDR.

    VSR and TrueHDR are run as separate passes (run_vsr then, after the caller's sharpen, run_hdr) so
    sharpening can land at the output resolution between them. Both are per frame."""

    def __init__(self, width, height, out_w, out_h, vsr=True, hdr=False, vsr_quality=4,
                 hdr_max_nits=1000, hdr_contrast=100, hdr_saturation=0, hdr_middlegray=50,
                 hdr_color="vivid", hdr_vibrance=0.0, hdr_satboost=0.0, collect_l1=False,
                 rtx_dir=RTX_DIR):
        # Make sure torch's primary CUDA context exists and is current on this thread before the
        # bridge retains it (create is called with cuContext=NULL -> cuDevicePrimaryCtxRetain).
        torch.zeros(8, device="cuda")
        torch.cuda.synchronize()

        self.lib = _load(rtx_dir)
        self.lib.rtxv_set_model_path(rtx_dir)
        self.W, self.H = width, height
        # The final output resolution: VSR upscales (W,H) -> (oW,oH) and TrueHDR runs at (oW,oH).
        # The caller passes out == in when there is no upscale.
        self.oW, self.oH = out_w, out_h
        self.vsr, self.hdr, self.vsr_q = bool(vsr), bool(hdr), vsr_quality

        r = self.lib.rtx_video_api_cuda_create(None, None, 0, int(bool(hdr)), int(bool(vsr)))
        if r != _API_SUCCESS:
            raise RuntimeError("rtx_video_api_cuda_create failed (VSR/TrueHDR unavailable on this "
                               "GPU/driver, or feature DLLs not found)")

        self._vsr = _VSR(int(vsr_quality))
        # TrueHDR tone controls (SDK ranges from nvsdk_ngx_defs_truehdr.h): Contrast 0..200, Saturation
        # 0..200, MiddleGray 10..100. The SDK defaults are 100/100/50, but its "neutral" Saturation 100
        # measurably oversaturates versus the SDR source (the SDR->HDR model adds vibrance of its own),
        # so hdr_saturation defaults to 0 = faithful (matches the source's saturation); 100 is the vivid
        # look, in between trades them. MaxLuminance 400..2000 (def 1000) is the mastering peak the PQ
        # values are shaped to and is written into the HDR10 mastering-display metadata (see
        # gmfss_interp / hdr10_meta), so one file tone-maps to any display without a per-monitor knob.
        self._cll = 0.0    # running MaxCLL  (brightest maxRGB pixel, nits) over all HDR frames
        self._fall = 0.0   # running MaxFALL (brightest frame-average maxRGB, nits)
        # Per-frame Dolby Vision L1 (min/avg/max PQ brightness), collected only for the --dv export so
        # a normal HDR render pays nothing. One triple is appended per run_hdr call, in output order.
        self._collect_l1 = bool(collect_l1)
        self._l1 = []
        self._thdr = _THDR(max(0, min(200, int(hdr_contrast))),
                           max(0, min(200, int(hdr_saturation))),
                           max(10, min(100, int(hdr_middlegray))),
                           max(400, min(2000, int(hdr_max_nits))))

        # Reused GPU staging buffers, packed BGRA uint8 (pitch = 4*width), alpha constant. VSR takes a
        # source-size input and writes an output-size result; TrueHDR takes an output-size input (the
        # already-upscaled, sharpened frame) and writes the packed 10-bit output. Only the enabled
        # features allocate their buffers.
        if self.vsr:
            self._vsr_in = torch.empty((self.H, self.W, 4), dtype=torch.uint8, device="cuda")
            self._vsr_in[..., 3] = 255
            self._vsr_out = torch.empty((self.oH, self.oW, 4), dtype=torch.uint8, device="cuda")
        if self.hdr:
            self._hdr_in = torch.empty((self.oH, self.oW, 4), dtype=torch.uint8, device="cuda")
            self._hdr_in[..., 3] = 255
            self._hdr_out = torch.empty((self.oH, self.oW, 4), dtype=torch.uint8, device="cuda")
            # Colour handling (the model rotates hues - it greens/cyans the blues - even at saturation 0):
            #   vivid (default) - keep TrueHDR's luminance (the HDR expansion), take the SDR source's hue
            #                     AND chroma in ICtCp (see _ictcp_correct): faithful colour, cyan rotation
            #                     removed, saturation matching the source. The SDK Saturation knob (_thdr)
            #                     is inert here because the model's chroma is dropped entirely.
            #   rtx             - keep TrueHDR's luminance, take the SDR source's hue, but take TrueHDR's own
            #                     chroma MAGNITUDE (floored at the source) in ICtCp (see _rtx_correct). The SDK
            #                     Saturation knob (_thdr) drives the magnitude here, so the slider edits
            #                     saturation like real RTX TrueHDR while the model's cyan/teal rotation is gone.
            #   raw             - emit TrueHDR's colour unmodified (the cyan-shifted reference).
            # All keep the linear BT.709 -> linear BT.2020 primaries matrix (BT.2087); vivid and rtx also need
            # the BT.2100 ICtCp matrices below.
            # The Dynamic Vibrance analog on top of vivid/rtx (inert in raw mode), mirroring NVIDIA's
            # separate vibrance filter with its two controls: _vibrance (Intensity, 0..1) is a chroma
            # gain weighted toward LOW-chroma pixels (muted colours pop, saturated colours and skin
            # stay put) and _satboost (Saturation boost, 0..1 = +0..100%) is a uniform chroma gain,
            # independent of the TrueHDR SDK Saturation that RTX HDR's own slider drives. Both are
            # hue-safe because Ct/Cp scale uniformly per pixel (see _vibrance_gain).
            self._color_mode = hdr_color if hdr_color in ("vivid", "rtx", "raw") else "vivid"
            self._vibrance = max(0.0, min(1.0, float(hdr_vibrance)))
            self._satboost = max(0.0, min(1.0, float(hdr_satboost)))
            self._m709_2020 = torch.tensor(
                [[0.6274, 0.3293, 0.0433], [0.0691, 0.9195, 0.0114], [0.0164, 0.0880, 0.8956]],
                dtype=torch.float32, device="cuda")
            # BT.2100 ICtCp (vivid): linear BT.2020 RGB -> LMS, and PQ L'M'S' -> ICtCp, plus the
            # numerically-inverted returns. Hue is ~the ICtCp Ct/Cp angle, so working here lets vivid
            # restore the source hue while keeping TrueHDR's intensity (I) and chroma magnitude.
            _rgb2lms = torch.tensor([[1688., 2146., 262.], [683., 2951., 462.], [99., 309., 3688.]],
                                    dtype=torch.float64) / 4096.0
            _lms2ictcp = torch.tensor([[2048., 2048., 0.], [6610., -13613., 7003.],
                                       [17933., -17390., -543.]], dtype=torch.float64) / 4096.0
            self._rgb2lms = _rgb2lms.to(torch.float32).cuda()
            self._lms2ictcp = _lms2ictcp.to(torch.float32).cuda()
            self._lms2rgb = torch.linalg.inv(_rgb2lms).to(torch.float32).cuda()
            self._ictcp2lms = torch.linalg.inv(_lms2ictcp).to(torch.float32).cuda()

    def run_vsr(self, rgb):
        """RTX VSR only on a [1,3,H,W] RGB float tensor in [0,1] on CUDA; returns a [1,3,oH,oW] RGB
        float tensor (SDR 8-bit, so a sharpen/HDR pass can follow). TrueHDR is not applied here."""
        t8 = (rgb[0].clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)   # [3,H,W] planes R,G,B
        self._vsr_in[..., 0] = t8[2]    # B
        self._vsr_in[..., 1] = t8[1]    # G
        self._vsr_in[..., 2] = t8[0]    # R
        torch.cuda.synchronize()        # finish the packing (default stream) before the bridge reads
        r = self.lib.rtx_video_api_cuda_evaluate_vsr_deviceptr(
            ctypes.c_void_p(self._vsr_in.data_ptr()), ctypes.c_void_p(self._vsr_out.data_ptr()),
            _RECT(0, 0, self.W, self.H), _RECT(0, 0, self.oW, self.oH), ctypes.byref(self._vsr))
        torch.cuda.synchronize()        # finish the eval before torch reads the output
        if r != _API_SUCCESS:
            raise RuntimeError("rtx_video_api_cuda_evaluate_vsr_deviceptr failed")
        d = self._vsr_out               # [oH,oW,4] bytes B,G,R,A
        return torch.stack([d[..., 2], d[..., 1], d[..., 0]], dim=0).float().div_(255.0).unsqueeze(0)

    def run_hdr(self, rgb):
        """RTX TrueHDR only (SDR -> HDR10) on a [1,3,oH,oW] RGB float tensor in [0,1] on CUDA, already
        at the final/output resolution (VSR and any RCAS sharpen have run upstream). Returns the packed
        10:10:10:2 bytes, which are ffmpeg's x2rgb10le (B in the low 10 bits, matching the model's
        channel-0=blue): PQ-encoded, BT.2020 primaries, full-range RGB10 (the
        DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020 the SDK viewer uses for ABGR10). The caller feeds
        these to ffmpeg as -pix_fmt x2rgb10le and tags the stream HDR10."""
        t8 = (rgb[0].clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)   # [3,oH,oW] planes R,G,B
        self._hdr_in[..., 0] = t8[2]    # B
        self._hdr_in[..., 1] = t8[1]    # G
        self._hdr_in[..., 2] = t8[0]    # R
        torch.cuda.synchronize()
        r = self.lib.rtx_video_api_cuda_evaluate_thdr_deviceptr(
            ctypes.c_void_p(self._hdr_in.data_ptr()), ctypes.c_void_p(self._hdr_out.data_ptr()),
            _RECT(0, 0, self.oW, self.oH), _RECT(0, 0, self.oW, self.oH), ctypes.byref(self._thdr))
        torch.cuda.synchronize()
        if r != _API_SUCCESS:
            raise RuntimeError("rtx_video_api_cuda_evaluate_thdr_deviceptr failed")
        if self._color_mode == "vivid":
            return self._ictcp_correct(rgb, self._hdr_out)    # source hue & chroma @ TrueHDR luma
        if self._color_mode == "rtx":
            return self._rtx_correct(rgb, self._hdr_out)      # source hue + TrueHDR chroma magnitude @ TrueHDR luma
        try:
            self._measure_light(self._hdr_out)
        except Exception:  # noqa: BLE001 - measurement is best-effort; never fail a render over it
            pass
        return self._hdr_out.cpu().numpy().tobytes()   # [oH,oW,4] packed 10:10:10:2 == x2rgb10le

    def _vibrance_gain(self, ct, cp):
        """The Dynamic Vibrance analog: scale Ct/Cp by the product of the uniform Saturation boost
        (1 + _satboost) and the Intensity term, a gain weighted toward LOW-chroma pixels so muted
        colours pop while already-saturated colours (and skin) stay put. The Intensity weight falls
        linearly from full boost at zero chroma to none at _VIBRANCE_CREF, roughly the ICtCp chroma
        of strongly saturated content. Per-pixel uniform Ct/Cp scaling preserves the hue angle, so
        neither control can reintroduce a cast. No-op when both are 0."""
        if self._vibrance <= 0.0 and self._satboost <= 0.0:
            return ct, cp
        c = torch.hypot(ct, cp)
        g = (1.0 + self._satboost) * (1.0 + self._vibrance * (1.0 - (c / _VIBRANCE_CREF).clamp(0.0, 1.0)))
        return ct * g, cp * g

    def _ictcp_correct(self, rgb, packed):
        """Faithful HDR (the default vivid mode). Keeps TrueHDR's luminance (the real HDR expansion) but
        rebuilds colour in ICtCp (BT.2100), the constant-intensity, hue-linear space, from the
        colorimetric SDR source's hue AND chroma - so the picture keeps exactly the source's colours at
        HDR brightness, unlike the TrueHDR model, which greens/cyans the blues. The SDK Saturation is
        inert here (the model's chroma is dropped); only _vibrance modifies chroma. Packs x2rgb10le."""
        u = packed.view(torch.int32).squeeze(-1)                             # [H,W]  B|G<<10|R<<20|A<<30
        thdr = torch.stack([(u >> 20) & 1023, (u >> 10) & 1023, u & 1023], 0).float().div_(1023.0)
        lin_t = _pq_to_linear(thdr)                                          # TrueHDR linear BT.2020 [3,H,W]
        y_t = 0.2627 * lin_t[0] + 0.6780 * lin_t[1] + 0.0593 * lin_t[2]      # BT.2020 luminance
        s = rgb[0].clamp(0.0, 1.0)                                           # SDR source, gamma BT.709
        lin709 = torch.where(s < 0.081, s / 4.5, ((s + 0.099) / 1.099).clamp(min=0.0).pow(1.0 / 0.45))
        lin = torch.einsum("ij,jhw->ihw", self._m709_2020, lin709)          # source linear BT.2020
        y_s = (0.2627 * lin[0] + 0.6780 * lin[1] + 0.0593 * lin[2]).clamp_(min=1e-6)
        lin_f = lin * (y_t / y_s)                                            # source chroma @ TrueHDR luma

        def _to_ictcp(linrgb):
            lms = torch.einsum("ij,jhw->ihw", self._rgb2lms, linrgb.clamp(min=0.0))
            return torch.einsum("ij,jhw->ihw", self._lms2ictcp, _linear_to_pq(lms))

        ic_f = _to_ictcp(lin_f)                                             # I, Ct, Cp: source hue & sat @ TrueHDR luma
        ct, cp = self._vibrance_gain(ic_f[1], ic_f[2])
        ictcp = torch.stack([ic_f[0], ct, cp], 0)
        lms = _pq_to_linear(torch.einsum("ij,jhw->ihw", self._ictcp2lms, ictcp))
        out = torch.einsum("ij,jhw->ihw", self._lms2rgb, lms).clamp_(0.0, 1.0)   # linear BT.2020
        return self._pack_out(out)

    def _rtx_correct(self, rgb, packed):
        """RTX-faithful HDR. Keeps the source hue but takes TrueHDR's chroma MAGNITUDE (driven by the SDK
        Saturation knob), floored at the source, so the SDK Saturation slider edits saturation like real RTX
        TrueHDR while the model's cyan/teal hue rotation is removed."""
        u = packed.view(torch.int32).squeeze(-1)                             # [H,W]  B|G<<10|R<<20|A<<30
        thdr = torch.stack([(u >> 20) & 1023, (u >> 10) & 1023, u & 1023], 0).float().div_(1023.0)
        lin_t = _pq_to_linear(thdr)                                          # TrueHDR linear BT.2020 [3,H,W]
        y_t = 0.2627 * lin_t[0] + 0.6780 * lin_t[1] + 0.0593 * lin_t[2]      # BT.2020 luminance
        s = rgb[0].clamp(0.0, 1.0)                                           # SDR source, gamma BT.709
        lin709 = torch.where(s < 0.081, s / 4.5, ((s + 0.099) / 1.099).clamp(min=0.0).pow(1.0 / 0.45))
        lin = torch.einsum("ij,jhw->ihw", self._m709_2020, lin709)          # source linear BT.2020
        y_s = (0.2627 * lin[0] + 0.6780 * lin[1] + 0.0593 * lin[2]).clamp_(min=1e-6)
        lin_f = lin * (y_t / y_s)                                            # source chroma @ TrueHDR luma

        def _to_ictcp(linrgb):
            lms = torch.einsum("ij,jhw->ihw", self._rgb2lms, linrgb.clamp(min=0.0))
            return torch.einsum("ij,jhw->ihw", self._lms2ictcp, _linear_to_pq(lms))

        ic_f = _to_ictcp(lin_f)                                             # I, Ct, Cp: source hue & sat @ TrueHDR luma
        # Take TrueHDR's chroma MAGNITUDE (the SDK Saturation knob drives it) but keep the source hue. The
        # gain d rescales the source Ct/Cp to TrueHDR's magnitude (floored at the source so it never
        # desaturates below the colorimetric source), preserving the source hue ANGLE so the model's
        # cyan/teal rotation is gone while the SDK Saturation slider still edits saturation.
        ic_t = _to_ictcp(lin_t)                                     # TrueHDR's own ICtCp
        mag_f = torch.hypot(ic_f[1], ic_f[2]).clamp_(min=1e-8)      # source chroma magnitude
        mag = torch.maximum(torch.hypot(ic_t[1], ic_t[2]), mag_f)   # TrueHDR magnitude, floored at source
        d = mag / mag_f
        ct, cp = self._vibrance_gain(ic_f[1] * d, ic_f[2] * d)
        ictcp = torch.stack([ic_f[0], ct, cp], 0)
        lms = _pq_to_linear(torch.einsum("ij,jhw->ihw", self._ictcp2lms, ictcp))
        out = torch.einsum("ij,jhw->ihw", self._lms2rgb, lms).clamp_(0.0, 1.0)   # linear BT.2020
        return self._pack_out(out)

    def _pack_out(self, out):
        """Shared vivid/rtx tail: accumulate MaxCLL/MaxFALL (and DV L1 when collecting) from the
        corrected linear BT.2020 frame `out` ([3,H,W]), then PQ-encode and pack to x2rgb10le bytes."""
        try:                                                                # MaxCLL/FALL from corrected px
            mx = out.max(0).values
            self._cll = max(self._cll, float(mx.max().item()) * 10000.0)
            self._fall = max(self._fall, float(mx.mean().item()) * 10000.0)
        except Exception:  # noqa: BLE001
            pass
        code = _linear_to_pq(out).mul_(1023.0).round_().clamp_(0, 1023).to(torch.int64)
        self._accum_l1(code.amax(0))                                        # DV L1 (maxRGB PQ); no-op unless --dv
        pk = (code[2] & 1023) | ((code[1] & 1023) << 10) | ((code[0] & 1023) << 20) | (3 << 30)
        return pk.cpu().numpy().astype("<u4").tobytes()                      # [H,W] -> x2rgb10le bytes

    def _accum_l1(self, maxrgb10):
        """Append one Dolby Vision L1 (per-frame min/avg/max brightness) from the frame's per-pixel
        max-RGB PQ code (10-bit int tensor), scaled to the 12-bit values the DV RPU carries. No-op
        unless created with collect_l1=True; one .tolist() sync per frame keeps the cost to ~one
        reduction (negligible beside the TrueHDR inference)."""
        if not self._collect_l1:
            return
        b = maxrgb10.float()
        stats = torch.stack([b.amin(), b.mean(), b.amax()]).mul_(4095.0 / 1023.0)
        mn, av, mx = stats.round_().clamp_(0, 4095).to(torch.int32).tolist()
        self._l1.append((mn, av, mx))

    def _measure_light(self, packed):
        """Accumulate MaxCLL/MaxFALL from one packed 10-bit PQ frame ([H,W,4] uint8 ==
        little-endian 10:10:10:2, B in the low 10 bits). maxRGB per CTA-861.3: per-pixel max of the
        linear R/G/B in nits; MaxCLL is the peak over all pixels, MaxFALL the peak frame average."""
        u = packed.view(torch.int32).squeeze(-1)                  # [H,W], B|G<<10|R<<20|A<<30
        code = torch.maximum(torch.maximum(u & 1023, (u >> 10) & 1023), (u >> 20) & 1023)
        ep = (code.to(torch.float32) / 1023.0).pow_(1.0 / _PQ_M2)  # PQ code' in [0,1]
        nits = ((ep - _PQ_C1).clamp_(min=0.0) / (_PQ_C2 - _PQ_C3 * ep).clamp_(min=1e-6)) \
            .pow_(1.0 / _PQ_M1).mul_(10000.0)                     # maxRGB luminance, nits
        self._cll = max(self._cll, float(nits.max().item()))
        self._fall = max(self._fall, float(nits.mean().item()))
        self._accum_l1(code)                                      # raw mode: DV L1 from the same maxRGB PQ code

    @property
    def maxcll(self):
        """Measured MaxCLL in nits (rounded up, clamped to the clli box's uint16), 0 if no frames."""
        return min(65535, int(math.ceil(self._cll)))

    @property
    def maxfall(self):
        """Measured MaxFALL in nits (rounded up, clamped to the clli box's uint16), 0 if no frames."""
        return min(65535, int(math.ceil(self._fall)))

    @property
    def l1(self):
        """Per-frame Dolby Vision L1 triples (min_pq, avg_pq, max_pq; 12-bit) in output-frame order,
        one per run_hdr call; empty unless the instance was created with collect_l1=True."""
        return self._l1

    def close(self):
        try:
            self.lib.rtx_video_api_cuda_shutdown()
        except Exception:  # noqa: BLE001 - best-effort, process is exiting anyway
            pass
