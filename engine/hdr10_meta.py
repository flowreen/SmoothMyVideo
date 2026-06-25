"""Inject HDR10 static metadata (mastering display + content light level) into an MP4.

The bundled ffmpeg is the BtbN **LGPL** build: it has no libx265, `hevc_nvenc` exposes no
mastering/CLL options, and the `hevc_metadata` bitstream filter carries only the VUI colour
fields (primaries/transfer/matrix), not the mastering-display or content-light SEI. So there is
no ffmpeg path on this build to write HDR10 static metadata. This module adds it directly at the
container level, the same place ffmpeg's own mov muxer would: two ISOBMFF boxes inside the video
sample entry, next to the `hvcC`/`colr` boxes that are already there.

- `mdcv` - Mastering Display Colour Volume (ISO/IEC 23001-17 / SMPTE ST 2086): the display the
  content was mastered on (BT.2020 primaries, D65 white, peak/black luminance). This is the box a
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

# BT.2020 primaries and D65 white point in mdcv's 0.00002 units (coord * 50000). The box order is
# Green, Blue, Red (SMPTE ST 2086 / ISO 23001-17), not RGB.
BT2020_GBR = ((8500, 39850), (6550, 2300), (35400, 14600))   # G, B, R  (0.170,0.797)(0.131,0.046)(0.708,0.292)
D65 = (15635, 16450)                                          # (0.3127, 0.3290)

_CONTAINERS = (b"moov", b"trak", b"mdia", b"minf", b"stbl")   # plain container boxes (8-byte header)


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


def inject_hdr10(path, max_nits=1000, min_nits=0.0001, maxcll=0, maxfall=0,
                 primaries=BT2020_GBR, white=D65):
    """Add mdcv + clli to the video sample entry of the MP4 at `path`, in place.

    Returns True if written, False if skipped (already present, or the structure was not the
    expected moov/trak/.../stsd shape). Raises only on real I/O errors.
    """
    with open(path, "rb") as f:
        buf = bytearray(f.read())
    n = len(buf)

    moov = _find(buf, 0, n, b"moov")
    if not moov:
        return False
    _, moov_ps, moov_be = moov
    trak = _video_trak(buf, moov_ps, moov_be)
    if not trak:
        return False
    _, trak_ps, trak_be = trak
    mdia = _find(buf, trak_ps, trak_be, b"mdia")
    minf = _find(buf, mdia[1], mdia[2], b"minf") if mdia else None
    stbl = _find(buf, minf[1], minf[2], b"stbl") if minf else None
    stsd = _find(buf, stbl[1], stbl[2], b"stsd") if stbl else None
    if not stsd:
        return False
    # stsd is a FullBox: 4 bytes version/flags + 4 bytes entry_count, then the sample entries.
    entries_start = stsd[1] + 8
    sample = next(_iter_boxes(buf, entries_start, stsd[2]), None)
    if not sample:
        return False
    s_type, s_start, s_ps, s_end = sample
    # Idempotent: skip if mdcv/clli already sit in the sample entry (children begin after the
    # 78-byte VisualSampleEntry header).
    for t, *_ in _iter_boxes(buf, s_ps + 78, s_end):
        if t in (b"mdcv", b"clli"):
            return False

    new = _mdcv_clli(primaries, white, max_nits, min_nits, maxcll, maxfall)
    delta = len(new)
    insert_at = s_end

    # Patch chunk-offset tables first (positions are at their original offsets). Any offset that
    # points past the insertion point shifts by delta. moov-at-end: offsets are < insert_at, so
    # nothing moves; faststart (moov before mdat): offsets are >= insert_at and all shift.
    for t, bs, ps, be in _iter_boxes(buf, stbl[1], stbl[2]):
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
    for t, bs, ps, be in _iter_boxes(buf, sample[2] + 78, sample[3]):
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
        print("wrote" if inject_hdr10(sys.argv[2], nits, maxcll=cll, maxfall=fall) else "skipped")
    elif len(sys.argv) >= 3 and sys.argv[1] == "dump":
        _dump(sys.argv[2])
    else:
        print("usage: hdr10_meta.py inject <file.mp4> [nits] [maxcll] [maxfall] | dump <file.mp4>")
