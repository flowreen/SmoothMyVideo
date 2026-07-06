"""Single-frame preview for the GUI's before/after pane.

Renders ONE source frame at the current spatial settings and writes <out>_original.png (the untouched
source) and <out>_processed.png. The processed side applies the SAME passes in the SAME order as a full
render (to_bytes in gmfss_interp.py): AI detail restoration first when --restore (the shared
realesr.py, eager - one frame needs no TRT engine), then the upscale, then FSR RCAS sharpening in SDR
(the shared rcas.py, the exact kernel the render uses), then RTX TrueHDR when --rtx-hdr. No
interpolation, no encode, so the GUI can scrub settings and see the effect before a full render.

The HDR result is PQ/BT.2020, which a normal canvas cannot show, so it is tonemapped to sRGB for
display: exposure is anchored to the SDR source (median luminance match), so midtones keep the source
brightness and only the highlight expansion differs; a Reinhard shoulder above the knee rolls the
expanded highlights toward white instead of clipping. (An earlier 99th-percentile auto-exposure made
the whole preview dim and washed out, because the brightest HDR highlight was dragging every midtone
down.) RCAS runs on the CPU when HDR is off, so a sharpen-only preview never touches CUDA, and when neither
sharpen nor HDR is requested the frame is copied straight through without even importing torch (the
GUI auto-loads a preview on every video select, so the do-nothing case stays instant and the pane
labels it "Unchanged"). --upscale (with optional --rtx-vsr) runs the spatial upscale first, exactly
like a render: upscale to the output resolution, RCAS there, HDR last; VSR degrades to bicubic when
the RTX runtime is absent, mirroring the engine.

HDR sources: when the input itself is already HDR (PQ transfer), TrueHDR is skipped, it is an
SDR-to-HDR model and the engine skips it for HDR sources too; sharpening still applies (on the
decoded PQ pixels, exactly as a render does), and BOTH panes are tonemapped for display so the
preview does not read flat and washed out. RTX VSR / upscale preview is a later addition.
"""
import os
import sys
import argparse
import subprocess

import numpy as np
import cv2

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ENGINE_DIR)

# linear BT.2020 -> linear BT.709, for tonemapping the PQ HDR output down to an sRGB preview.
_M2020_709 = np.array([[1.6605, -0.5876, -0.0728], [-0.1246, 1.1329, -0.0083],
                       [-0.0182, -0.1006, 1.1187]], np.float32)

# SMPTE ST 2084 (PQ) constants, numpy twin of the ones in rtxvideo.py (no torch dependency here).
_PQ_M1, _PQ_M2 = 0.1593017578125, 78.84375
_PQ_C1, _PQ_C2, _PQ_C3 = 0.8359375, 18.8515625, 18.6875


def _ffprobe():
    exe = os.path.join(ENGINE_DIR, "bin", "ffprobe.exe")
    return exe if os.path.isfile(exe) else "ffprobe"


def _src_transfer(path):
    """The source's color_transfer tag ('' when unknown); smpte2084/arib-std-b67 mark HDR sources."""
    try:
        out = subprocess.check_output(
            [_ffprobe(), "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=color_transfer", "-of", "csv=p=0", path],
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return out.decode("utf-8", "replace").strip().rstrip(",").lower()   # csv adds a trailing comma
    except Exception:  # noqa: BLE001 - probe is best-effort; treat unknown as SDR
        return ""


def _pq_to_linear_np(e):
    """SMPTE ST 2084 EOTF in numpy: PQ code' in [0,1] -> display-linear (1.0 == 10000 nits)."""
    ep = np.power(np.clip(e, 0.0, 1.0), 1.0 / _PQ_M2)
    return np.power(np.clip(ep - _PQ_C1, 0.0, None) / np.clip(_PQ_C2 - _PQ_C3 * ep, 1e-6, None),
                    1.0 / _PQ_M1)


def _read_frame(path, which):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idx = max(0, n // 2) if which == "mid" else max(0, min(n - 1, int(which)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, bgr = cap.read()
    if not ok:                                   # seek can miss on some codecs; fall back to frame 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("could not read a frame from " + path)
    return bgr, idx, n


def _lum709(a):
    return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]


def _shoulder_srgb(x):
    """Shared display encode: linear BT.709 with exposure applied -> sRGB uint8, per-channel Reinhard
    shoulder above the knee so expanded highlights roll toward white instead of clipping."""
    knee = 0.75
    t = (x - knee) / (1.0 - knee)
    y = np.where(x <= knee, x, knee + (1.0 - knee) * (t / (1.0 + np.abs(t))))
    srgb = np.where(y <= 0.0031308, y * 12.92,
                    1.055 * np.power(np.clip(y, 0.0, None), 1 / 2.4) - 0.055)
    return (np.clip(srgb, 0.0, 1.0) * 255).astype(np.uint8)


def _tonemap(lin2020, src_rgb_u8):
    """Linear BT.2020 (1.0 == 10000 nits) -> sRGB uint8 RGB, anchored to the SDR source.

    Exposure: scale so the HDR frame's median luminance equals the SDR source's (in linear 709), so
    midtones read exactly as bright as the original and the preview never dims or washes out.
    """
    lin709 = np.clip(lin2020 @ _M2020_709.T, 0.0, None)
    s = src_rgb_u8.astype(np.float32) / 255.0
    lin_s = np.where(s <= 0.04045, s / 12.92, np.power((s + 0.055) / 1.055, 2.4))
    ls, lh = _lum709(lin_s), _lum709(lin709)
    # Per-image lit-content masks: the HDR frame may be at the upscaled output resolution while the
    # SDR reference stays at source resolution, so one shared mask cannot index both.
    ms, mh = ls > 1e-4, lh > 1e-6
    k = (float(np.median(ls[ms])) / max(1e-9, float(np.median(lh[mh])))) if (ms.any() and mh.any()) else 1.0
    return _shoulder_srgb(lin709 * k)


def _tonemap_pq(rgb_u8):
    """Display an already-HDR (PQ) frame on the sRGB preview: decode the PQ code values to linear
    BT.2020, anchor exposure so the median luminance lands at a typical SDR diffuse level (0.2
    linear), and roll the highlights with the shared shoulder. Self-anchored, because an HDR source
    has no SDR reference frame to match against."""
    lin2020 = _pq_to_linear_np(rgb_u8.astype(np.float32) / 255.0)
    lin709 = np.clip(lin2020 @ _M2020_709.T, 0.0, None)
    lh = _lum709(lin709)
    m = lh > 1e-5
    k = (0.20 / max(1e-9, float(np.median(lh[m])))) if m.any() else 1.0
    return _shoulder_srgb(lin709 * k)


def _make_rtx(w, h, ow, oh, need_vsr, need_hdr, args):
    """One RTX Video instance for whatever the preview needs (VSR to ow x oh and/or TrueHDR at
    ow x oh), configured exactly like the render engine's. Raises if the bridge/runtime is absent."""
    import rtxvideo
    return rtxvideo.RTXVideo(
        w, h, ow, oh, vsr=need_vsr, hdr=need_hdr, hdr_max_nits=max(400, min(2000, args.hdr_nits)),
        hdr_contrast=max(0, min(200, args.hdr_contrast)),
        hdr_saturation=max(0, min(200, args.hdr_saturation)),
        hdr_middlegray=max(10, min(100, args.hdr_middlegray)),
        hdr_color=args.hdr_color, hdr_vibrance=max(0.0, min(1.0, args.hdr_vibrance)),
        hdr_satboost=max(0.0, min(1.0, args.hdr_satboost)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--frame", default="mid", help="source frame index, or 'mid' (default)")
    ap.add_argument("--out", required=True,
                    help="output prefix; writes <out>_original.png and <out>_processed.png")
    ap.add_argument("--rtx-hdr", action="store_true", help="convert the processed frame to HDR (tonemapped)")
    ap.add_argument("--sharpen", type=float, default=0.0, help="FSR RCAS sharpen strength 0..1")
    ap.add_argument("--restore", action="store_true",
                    help="AI detail restoration (Real-ESRGAN animevideov3), like the render's --restore")
    ap.add_argument("--upscale", type=float, default=1.0,
                    help="spatial upscale factor for the processed side (1 = off, clamped to 8)")
    ap.add_argument("--rtx-vsr", action="store_true",
                    help="use RTX Video Super Resolution for --upscale (falls back to bicubic)")
    ap.add_argument("--hdr-nits", type=int, default=1000)
    ap.add_argument("--hdr-color", choices=["vivid", "rtx", "raw"], default="vivid")
    ap.add_argument("--hdr-vibrance", type=float, default=0.0)
    ap.add_argument("--hdr-satboost", type=float, default=0.0)
    ap.add_argument("--hdr-saturation", type=int, default=0)
    ap.add_argument("--hdr-contrast", type=int, default=100)
    ap.add_argument("--hdr-middlegray", type=int, default=50)
    args = ap.parse_args()
    strength = max(0.0, min(1.0, args.sharpen))

    bgr, idx, n = _read_frame(args.input, args.frame)
    rgb_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    transfer = _src_transfer(args.input)
    src_hdr = transfer in ("smpte2084", "arib-std-b67")
    do_hdr = args.rtx_hdr and not src_hdr   # TrueHDR is SDR-to-HDR; the engine skips it on HDR sources
    h0, w0 = bgr.shape[0], bgr.shape[1]
    up = max(1.0, min(16.0, args.upscale))
    ow, oh = ((round(w0 * up) // 2) * 2, (round(h0 * up) // 2) * 2) if up > 1.0 else (w0, h0)
    if do_hdr and (ow > 8192 or oh > 8192):
        do_hdr = False   # mirror the engine: TrueHDR past 8192px oversubscribes VRAM (DPC-watchdog risk)
    vsr_used = False
    restore_used = False

    if strength > 0 or do_hdr or up > 1.0 or args.restore:
        import torch                       # heavy import, skipped entirely on the unchanged fast path
        from rcas import rcas              # the render engine's own FSR RCAS kernel
        need_vsr = up > 1.0 and args.rtx_vsr
        dev = "cuda" if (do_hdr or need_vsr or args.restore) else "cpu"
        t = torch.from_numpy(rgb_u8).to(dev).float().div(255.0).permute(2, 0, 1).unsqueeze(0).contiguous()
        rtx = None
        if need_vsr or do_hdr:
            try:
                rtx = _make_rtx(w0, h0, ow, oh, need_vsr, do_hdr, args)
            except Exception:  # noqa: BLE001 - like the render: no bridge means bicubic + SDR
                rtx, do_hdr, need_vsr = None, False, False
        try:
            # Same order as a render (to_bytes): restore, upscale, RCAS at the OUTPUT resolution, HDR.
            if args.restore:
                try:
                    import realesr
                    net = realesr.load(torch.device("cuda"))
                    r = net(t.half()).clamp(0.0, 1.0)          # fp16 4x reconstruction (eager: one frame)
                    # Without VSR the reconstruction IS the upscale source, exactly like a render;
                    # otherwise back to source size so VSR upscales the restored frame below.
                    as_up = up > 1.0 and not need_vsr
                    t = realesr.fit(r, oh if as_up else h0, ow if as_up else w0).float()
                    restore_used = True
                except Exception:  # noqa: BLE001 - degrade like the render: unrestored frame
                    pass
            if up > 1.0 and t.shape[-1] != ow:
                if rtx is not None and need_vsr:
                    try:
                        t = rtx.run_vsr(t)
                        vsr_used = True
                    except Exception:  # noqa: BLE001 - degrade to bicubic like the render does
                        pass
                if not vsr_used:
                    import torch.nn.functional as tf
                    t = tf.interpolate(t, size=(oh, ow), mode="bicubic", align_corners=False).clamp(0.0, 1.0)
            if strength > 0:
                t = rcas(t, strength)
            if do_hdr and rtx is not None:
                packed = rtx.run_hdr(t)
                u = np.frombuffer(packed, "<u4").reshape(oh, ow)
                code = np.stack([(u >> 20) & 1023, (u >> 10) & 1023, u & 1023], -1).astype(np.float32) / 1023.0
                import rtxvideo
                lin = rtxvideo._pq_to_linear(torch.from_numpy(code)).numpy()
                proc_rgb = _tonemap(lin, rgb_u8)
            else:
                proc_rgb = (t[0].permute(1, 2, 0).clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy())
        finally:
            if rtx is not None:
                rtx.close()
    else:
        proc_rgb = rgb_u8                  # nothing enabled: the render would leave the frame as is

    # Display conversion: a PQ (HDR10) source cannot be shown raw on the sRGB canvas (it reads flat
    # and washed out), so both panes get the self-anchored tonemap. HLG is shown as-is (it is
    # designed to degrade acceptably on SDR displays). The do_hdr output was already tonemapped.
    orig_disp = _tonemap_pq(rgb_u8) if transfer == "smpte2084" else rgb_u8
    proc_disp = _tonemap_pq(proc_rgb) if (transfer == "smpte2084" and not do_hdr) else proc_rgb

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    # Match the processed side's OUTPUT resolution so the before/after panes zoom 1:1 identically
    # (same native pixel size -> same magnification, pixel-aligned). Bicubic - the SAME resampler the
    # non-VSR upscale path uses (gmfss_interp _upscale) - so the original is the honest baseline: when
    # VSR runs, the processed side is genuinely sharper; when it doesn't, both panes are the same
    # bicubic image and the renderer labels it "plain bicubic upscale" rather than faking a difference.
    if up > 1.0 and (orig_disp.shape[1] != ow or orig_disp.shape[0] != oh):
        orig_disp = cv2.resize(orig_disp, (ow, oh), interpolation=cv2.INTER_CUBIC)
    p_orig, p_proc = args.out + "_original.png", args.out + "_processed.png"
    cv2.imwrite(p_orig, cv2.cvtColor(orig_disp, cv2.COLOR_RGB2BGR))
    cv2.imwrite(p_proc, cv2.cvtColor(proc_disp, cv2.COLOR_RGB2BGR))
    sys.stdout.write(f"preview frame {idx}/{n} {ow}x{oh} hdr={int(do_hdr)} sharpen={strength:g} "
                     f"srchdr={int(src_hdr)} up={up:g} vsr={int(vsr_used)} "
                     f"restore={int(restore_used)} -> {p_orig} | {p_proc}\n")


if __name__ == "__main__":
    main()
