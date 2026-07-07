"""Inject HDR10 static metadata (mastering display + content light level) into an MP4.

The bundled ffmpeg is the BtbN **LGPL** build: it has no libx265, `hevc_nvenc` exposes no
mastering/CLL options, and the `hevc_metadata` bitstream filter carries only the VUI colour
fields (primaries/transfer/matrix), not the mastering-display or content-light SEI. So there is
no ffmpeg path on this build to write HDR10 static metadata. This module adds it directly at the
container level, the same place ffmpeg's own mov muxer would: two ISOBMFF boxes inside the video
sample entry, next to the `hvcC`/`colr` boxes that are already there.

- `mdcv` - Mastering Display Colour Volume (ISO/IEC 23001-17 / SMPTE ST 2086): the display the
  content was mastered on (grading-monitor primaries - Display P3 + D65 white by default - plus
  peak/black luminance). This is the box a
  TV uses to tone-map: it says "graded for a 1000-nit display", so a 400-nit panel knows to roll
  the highlights down instead of guessing. Without it a player only has the PQ/BT.2020 signalling
  and assumes a default peak.
- `clli` - Content Light Level (CTA-861.3): MaxCLL (brightest pixel) and MaxFALL (brightest
  frame-average), measured from the actual frames. A secondary hint that lets a display avoid
  over-dimming content that never reaches the mastering peak.

Both are codec-agnostic (they live in the visual sample entry, valid for hvc1/hev1/av01), and are
written so one file tone-maps correctly on both a 1000-nit and a 400-nit display with no
per-display input. The insert grows the sample entry and its ancestor boxes; chunk offsets
(`stco`/`co64`) are patched for any layout (the app's encode is moov-at-end, where they do not
move, but faststart is handled too). Injection is idempotent: it is a no-op if the boxes already
exist.

All multi-byte fields are big-endian (ISOBMFF). Chromaticities are in units of 0.00002
(value = coord * 50000); luminances in units of 0.0001 cd/m2 (value = nits * 10000).
"""
import struct
import sys

# Mastering-display gamuts in mdcv's 0.00002 units (coord * 50000). Box order is Green, Blue, Red
# (SMPTE ST 2086 / ISO 23001-17), not RGB. Real HDR masters carry their grading monitor's primaries
# (almost always a P3 gamut inside a BT.2020 container), which is why a player prints explicit
# chromaticities for them; the nominal BT.2020 set instead collapses to the bare name "bt.2020".
P3_GBR     = ((13250, 34500), (7500, 3000), (34000, 16000))  # G(0.265,0.690) B(0.150,0.060) R(0.680,0.320)
BT2020_GBR = ((8500, 39850), (6550, 2300), (35400, 14600))   # G(0.170,0.797) B(0.131,0.046) R(0.708,0.292)
BT709_GBR  = ((15000, 30000), (7500, 3000), (32000, 16500))  # G(0.300,0.600) B(0.150,0.060) R(0.640,0.330)
D65 = (15635, 16450)   # (0.3127, 0.3290) white - sRGB / Display-P3 / BT.709 / BT.2020
DCI = (15700, 17550)   # (0.3140, 0.3510) white - DCI theatrical (SMPTE RP431-2), greener than D65

# Mastering colorspace name -> (GBR primaries, white point), keyed by mpv's colour-space names so the
# --hdr-mastering-prim flag takes a string a player would recognise. display-p3 and dci-p3 share the P3
# gamut and differ only in white point (D65 vs DCI theatrical white). display-p3 is the default: the
# de-facto grading gamut and a faithful bound for SDR-sourced TrueHDR output (whose real gamut sits
# within P3), so the file reads like a normal HDR master. The stream's own colr/VUI primaries stay
# BT.2020 either way - this box is only the mastering-display hint.
MASTERING_COLORSPACES = {
    "display-p3": (P3_GBR, D65),   # P3 primaries + D65 white (Apple Display P3)
    "dci-p3":     (P3_GBR, DCI),   # P3 primaries + DCI theatrical white (SMPTE RP431-2)
    "bt2020":     (BT2020_GBR, D65),
    "bt709":      (BT709_GBR, D65),
}
DEFAULT_COLORSPACE = "display-p3"


def _iter_boxes(buf, start, end):
    """Yield (type, box_start, payload_start, box_end) for each box in [start, end)."""
    off = start
    while off + 8 <= end:
        size = struct.unpack_from(">I", buf, off)[0]
        typ = bytes(buf[off + 4:off + 8])
        hdr = 8
        if size == 1:                                   # 64-bit largesize
            size = struct.unpack_from(">Q", buf, off + 8)[0]
            hdr = 16
        elif size == 0:                                 # extends to the end of the parent
            size = end - off
        if size < hdr or off + size > end:
            return
        yield typ, off, off + hdr, off + size
        off += size


def _find(buf, start, end, typ):
    for t, bs, ps, be in _iter_boxes(buf, start, end):
        if t == typ:
            return bs, ps, be
    return None


def _grow(buf, box_start, delta):
    """Add `delta` to the 32-bit (or 64-bit largesize) size field of the box at box_start."""
    size = struct.unpack_from(">I", buf, box_start)[0]
    if size == 1:
        big = struct.unpack_from(">Q", buf, box_start + 8)[0]
        struct.pack_into(">Q", buf, box_start + 8, big + delta)
    else:
        struct.pack_into(">I", buf, box_start, size + delta)


def _video_trak(buf, moov_ps, moov_be):
    """Return (trak_start, trak_payload_start, trak_end) of the trak whose handler is 'vide'."""
    for t, bs, ps, be in _iter_boxes(buf, moov_ps, moov_be):
        if t != b"trak":
            continue
        mdia = _find(buf, ps, be, b"mdia")
        if not mdia:
            continue
        hdlr = _find(buf, mdia[1], mdia[2], b"hdlr")
        if hdlr and bytes(buf[hdlr[1] + 8:hdlr[1] + 12]) == b"vide":
            return bs, ps, be
    return None


def _mdcv_clli(primaries, white, max_nits, min_nits, maxcll, maxfall):
    g, b, r = primaries
    mdcv_payload = struct.pack(
        ">8H2I", g[0], g[1], b[0], b[1], r[0], r[1], white[0], white[1],
        int(round(max_nits * 10000)), int(round(min_nits * 10000)))
    clli_payload = struct.pack(">2H", int(maxcll) & 0xFFFF, int(maxfall) & 0xFFFF)
    mdcv = struct.pack(">I", 8 + len(mdcv_payload)) + b"mdcv" + mdcv_payload
    clli = struct.pack(">I", 8 + len(clli_payload)) + b"clli" + clli_payload
    return mdcv + clli


def _insert_into_sample_entry(path, build_box, skip_types):
    """Shared ISOBMFF surgery: append child box(es) to the video sample entry of the MP4 at `path`,
    in place. `build_box()` returns the bytes to insert (called only when nothing in `skip_types` is
    already present, so the operation is idempotent). Patches every stco/co64 chunk offset that points
    past the insertion point and grows the sample entry + all ancestor boxes up to moov. Returns True
    if written, False if skipped (already present, or not the expected moov/trak/.../stsd shape).
    Raises only on real I/O errors."""
    with open(path, "rb") as f:
        buf = bytearray(f.read())
    n = len(buf)
    moov = _find(buf, 0, n, b"moov")
    if not moov:
        return False
    trak = _video_trak(buf, moov[1], moov[2])
    if not trak:
        return False
    mdia = _find(buf, trak[1], trak[2], b"mdia")
    minf = _find(buf, mdia[1], mdia[2], b"minf") if mdia else None
    stbl = _find(buf, minf[1], minf[2], b"stbl") if minf else None
    stsd = _find(buf, stbl[1], stbl[2], b"stsd") if stbl else None
    if not stsd:
        return False
    # stsd is a FullBox: 4 bytes version/flags + 4 bytes entry_count, then the sample entries.
    sample = next(_iter_boxes(buf, stsd[1] + 8, stsd[2]), None)
    if not sample:
        return False
    _s_type, s_start, s_ps, s_end = sample
    # Idempotent: children begin after the 78-byte VisualSampleEntry header.
    for t, *_ in _iter_boxes(buf, s_ps + 78, s_end):
        if t in skip_types:
            return False

    new = build_box()
    delta = len(new)
    insert_at = s_end

    # Patch chunk-offset tables first (positions are at their original offsets). Any offset that
    # points past the insertion point shifts by delta. moov-at-end: offsets are < insert_at, so
    # nothing moves; faststart (moov before mdat): offsets are >= insert_at and all shift.
    for t, _bs, ps, _be in _iter_boxes(buf, stbl[1], stbl[2]):
        if t == b"stco":
            cnt = struct.unpack_from(">I", buf, ps + 4)[0]
            base = ps + 8
            for i in range(cnt):
                o = base + 4 * i
                v = struct.unpack_from(">I", buf, o)[0]
                if v >= insert_at:
                    struct.pack_into(">I", buf, o, v + delta)
        elif t == b"co64":
            cnt = struct.unpack_from(">I", buf, ps + 4)[0]
            base = ps + 8
            for i in range(cnt):
                o = base + 8 * i
                v = struct.unpack_from(">Q", buf, o)[0]
                if v >= insert_at:
                    struct.pack_into(">Q", buf, o, v + delta)

    # Grow the sample entry and every ancestor up to moov by delta.
    for bstart in (s_start, stsd[0], stbl[0], minf[0], mdia[0], trak[0], moov[0]):
        _grow(buf, bstart, delta)

    buf[insert_at:insert_at] = new
    with open(path, "wb") as f:
        f.write(buf)
    return True


def inject_hdr10(path, max_nits=1000, min_nits=0.0, maxcll=0, maxfall=0,
                 colorspace=DEFAULT_COLORSPACE):
    """Add mdcv + clli to the video sample entry of the MP4 at `path`, in place.

    `colorspace` names the mastering-display gamut + white point (see MASTERING_COLORSPACES, e.g.
    display-p3 / dci-p3 / bt2020 / bt709); an unrecognised name falls back to the default.
    `min_nits` defaults to 0 - a perfect-black mastering reference, as OLED-graded HDR10 declares.
    It is metadata only: actual black reproduction comes from the PQ stream, not from this field.
    Returns True if written, False if skipped (already present or unexpected structure).
    """
    primaries, white = MASTERING_COLORSPACES.get(colorspace, MASTERING_COLORSPACES[DEFAULT_COLORSPACE])
    return _insert_into_sample_entry(
        path, lambda: _mdcv_clli(primaries, white, max_nits, min_nits, maxcll, maxfall),
        (b"mdcv", b"clli"))


# --- Dolby Vision Profile 8.1 signaling (dvvC box) --------------------------------------------
# A DV 8.1 stream is plain HDR10 (BT.2020 PQ) HEVC carrying a Dolby Vision RPU in-band (added to the
# elementary stream by dovi_tool) plus this one small container box that tags the track as Dolby
# Vision. Non-DV players ignore the box and the RPU and just play the HDR10 base; DV displays read
# them. Writing it here (the same box surgery inject_hdr10 does) means the bundled LGPL ffmpeg - which
# cannot emit a dvcC/dvvC box itself - is enough to mux Dolby Vision, so no GPAC/MP4Box is needed.
# This is our own code writing a documented box format (Dolby's public ISOBMFF spec); it uses no
# Dolby or GPAC source and performs no patented processing - it only tags an already-built stream.
_DV_LEVEL_CAPS = [22118400, 27648000, 49766400, 62208000, 124416000, 199065600,
                  248832000, 398131200, 497664000, 995328000, 1990656000]


def dv_level(width, height, fps):
    """Dolby Vision level from the luminance pixel rate (w*h*fps), per the DV ISOBMFF level table:
    level 5 covers 1080p60, 9 covers 2160p60, 11+ the 8K tiers. Clamped to the top defined level."""
    pps = width * height * max(1.0, float(fps))
    for i, cap in enumerate(_DV_LEVEL_CAPS, 1):
        if pps <= cap:
            return i
    return len(_DV_LEVEL_CAPS)


def _dv_config_box(profile, level, bl_compat):
    # DOVIDecoderConfigurationRecord: 2 version bytes, then a 48-bit packed field (7-bit profile,
    # 6-bit level, rpu/el/bl present flags, 4-bit BL signal compatibility id, 28 reserved), then 16
    # reserved bytes. rpu present + bl present, no enhancement layer = single-layer Profile 8.1.
    bits = ((profile & 0x7F) << 41) | ((level & 0x3F) << 35) | (1 << 34) | (0 << 33) | (1 << 32) \
        | ((bl_compat & 0xF) << 28)
    rec = bytes([1, 0]) + bits.to_bytes(6, "big") + b"\x00" * 16
    return struct.pack(">I", 8 + len(rec)) + b"dvvC" + rec


def inject_dv_config(path, width, height, fps, profile=8, bl_compat=1):
    """Stamp a Dolby Vision configuration box (dvvC) into the video sample entry of the MP4 at
    `path`, in place, tagging a DV-injected HEVC (RPU already in-band) as Profile `profile` - default
    8 with bl_compat 1 = the HDR10-compatible Profile 8.1. The dv_level is derived from
    width*height*fps. Same box surgery as inject_hdr10; idempotent. Returns True if written, False if
    skipped/unsupported. Verified: ffprobe reads the result as a DOVI configuration record."""
    return _insert_into_sample_entry(
        path, lambda: _dv_config_box(profile, dv_level(width, height, fps), bl_compat),
        (b"dvvC", b"dvcC"))


def _dump(path):
    """Print the mastering-display / content-light boxes found in the video sample entry."""
    with open(path, "rb") as f:
        buf = bytearray(f.read())
    n = len(buf)
    moov = _find(buf, 0, n, b"moov")
    trak = _video_trak(buf, moov[1], moov[2]) if moov else None
    if not trak:
        print("no video trak"); return
    mdia = _find(buf, trak[1], trak[2], b"mdia")
    minf = _find(buf, mdia[1], mdia[2], b"minf")
    stbl = _find(buf, minf[1], minf[2], b"stbl")
    stsd = _find(buf, stbl[1], stbl[2], b"stsd")
    sample = next(_iter_boxes(buf, stsd[1] + 8, stsd[2]))
    print("sample entry:", sample[0].decode(errors="replace"))
    for t, _bs, ps, _be in _iter_boxes(buf, sample[2] + 78, sample[3]):
        if t == b"mdcv":
            vals = struct.unpack_from(">8H2I", buf, ps)
            print("  mdcv", vals)
        elif t == b"clli":
            print("  clli", struct.unpack_from(">2H", buf, ps))


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "inject":
        nits = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
        cll = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        fall = int(sys.argv[5]) if len(sys.argv) > 5 else 0
        cs = sys.argv[6] if len(sys.argv) > 6 else DEFAULT_COLORSPACE
        print("wrote" if inject_hdr10(sys.argv[2], nits, maxcll=cll, maxfall=fall, colorspace=cs)
              else "skipped")
    elif len(sys.argv) >= 3 and sys.argv[1] == "dump":
        _dump(sys.argv[2])
    else:
        print("usage: hdr10_meta.py inject <file.mp4> [nits] [maxcll] [maxfall] "
              "[display-p3|dci-p3|bt2020|bt709] | dump <file.mp4>")
