"""Side-by-side comparison export for SmoothMyVideo.

Takes the original video and its interpolated (higher-fps) render and exports
one video: original on the LEFT, smoothed on the RIGHT, each pane labelled
with its frame rate. Pane order is decided by measured fps, so the two input
files can be passed (or drag-and-dropped) in any order.

Both panes are conformed to the higher frame rate (the left pane simply
repeats frames at its native cadence, which is exactly what makes the
difference visible) and to a common height. Uses the ffmpeg bundled in
engine/bin, encodes 10-bit HEVC via NVENC with an SVT-AV1 CPU fallback.

Usage:  python tools/compare.py <video A> <video B> [-o out.mp4]
        (or drop two files onto tools/compare.cmd)

Note: if one input is an HDR (PQ) render and the other is SDR, the panes will
not be colour-comparable; the script warns but still exports.
"""

import argparse
import json
import os
import subprocess
import sys
import time

BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine", "bin")
FFMPEG = os.path.join(BIN, "ffmpeg.exe")
FFPROBE = os.path.join(BIN, "ffprobe.exe")

FONTS = [
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
]


def probe(path):
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries",
         "stream=width,height,r_frame_rate,color_transfer,color_primaries,color_space"
         ":format=duration",
         "-of", "json", path],
        capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout.strip():
        sys.exit(f"ffprobe failed on {path}:\n{out.stderr}")
    data = json.loads(out.stdout)
    st = data["streams"][0]
    num, den = (int(x) for x in st["r_frame_rate"].split("/"))
    return {
        "path": path,
        "w": st["width"],
        "h": st["height"],
        "fps_frac": st["r_frame_rate"],
        "fps": num / den,
        "duration": float(data.get("format", {}).get("duration", 0) or 0),
        "transfer": st.get("color_transfer", ""),
        "primaries": st.get("color_primaries", ""),
        "space": st.get("color_space", ""),
    }


def unique_out(base):
    cand, n = base, 2
    root, ext = os.path.splitext(base)
    while os.path.exists(cand):
        cand = f"{root}_{n}{ext}"
        n += 1
    return cand


def drawtext(label, fontsize, pq=False):
    font = next((f for f in FONTS if os.path.exists(f)), None)
    if not font:
        return ""
    font = font.replace(":", "\\:")
    # In PQ video, full white encodes a 10000-nit signal; label at ~200 nits instead.
    color = "0x949494" if pq else "white"
    return (f",drawtext=text='{label}':fontfile='{font}':fontsize={fontsize}"
            f":fontcolor={color}:box=1:boxcolor=black@0.5:boxborderw={max(4, fontsize // 4)}"
            f":x=16:y=16")


def fps_label(v):
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return f"{s} fps"


def matched_color(low, high):
    """Colour tags shared by both inputs (only comparable when transfers match)."""
    if not low["transfer"] or low["transfer"] != high["transfer"]:
        return {}
    keys = (("color_trc", "transfer"), ("color_primaries", "primaries"),
            ("colorspace", "space"))
    return {flag: low[k] for flag, k in keys if low[k] and low[k] == high[k]}


def build_filter(low, high, height, names=("Original", "Smoothed")):
    fontsize = max(16, round(height * 0.035))
    pq = low["transfer"] == "smpte2084" and high["transfer"] == "smpte2084"
    lab_l = drawtext(f"{names[0]}  {fps_label(low['fps'])}", fontsize, pq)
    lab_r = drawtext(f"{names[1]}  {fps_label(high['fps'])}", fontsize, pq)
    chain = "fps={fps},scale=-2:{h}:flags=lanczos,format=yuv420p10le,setsar=1"
    left = chain.format(fps=high["fps_frac"], h=height) + lab_l
    right = chain.format(fps=high["fps_frac"], h=height) + lab_r
    # hstack resets the frames' transfer/primaries to unspecified, and the
    # encoder takes colour from the frame props (the -color_* output options
    # are ignored), so matched tags must be restamped inside the graph.
    stamp = ""
    tags = matched_color(low, high)
    if tags:
        stamp = ",setparams=" + ":".join(f"{k}={v}" for k, v in tags.items())
    return f"[0:v]{left}[l];[1:v]{right}[r];[l][r]hstack=shortest=1{stamp}[v]"


def eta_text(seconds):
    s = max(0, int(round(seconds)))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def run_encode(low, high, fc, out_path, encoder, total_frames):
    if encoder == "hevc_nvenc":
        vargs = ["-c:v", "hevc_nvenc", "-preset", "p5", "-tune", "hq",
                 "-rc", "vbr", "-cq", "18", "-b:v", "0",
                 "-spatial-aq", "1", "-temporal-aq", "1",
                 "-profile:v", "main10", "-pix_fmt", "p010le", "-tag:v", "hvc1"]
    else:  # libsvtav1
        vargs = ["-c:v", "libsvtav1", "-crf", "22", "-preset", "7",
                 "-pix_fmt", "yuv420p10le"]
    cmd = [FFMPEG, "-hide_banner", "-nostats", "-v", "error", "-y",
           "-i", low["path"], "-i", high["path"],
           "-filter_complex", fc, "-map", "[v]", "-map", "0:a?",
           *vargs, "-c:a", "aac", "-b:a", "192k", "-shortest",
           "-movflags", "+faststart", "-progress", "pipe:1", out_path]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    start, frame, fps = time.time(), 0, 0.0
    for line in proc.stdout:
        key, _, val = line.strip().partition("=")
        if key == "frame":
            frame = int(val)
        elif key == "fps":
            fps = float(val)
        elif key == "progress":  # one block of stats is complete
            rate = fps if fps > 0 else frame / max(time.time() - start, 1e-6)
            if total_frames and rate > 0:
                pct = min(100, 100 * frame // total_frames)
                eta = eta_text(max(0, total_frames - frame) / rate)
                msg = (f"frame {frame}/{total_frames}  {pct}%  "
                       f"{rate:.0f} fps  ETA {eta}")
            else:
                msg = f"frame {frame}  {rate:.0f} fps"
            print(f"\r  {msg}   ", end="", flush=True)
    proc.wait()
    print()
    return proc.returncode


def ask_for_video(what):
    while True:
        p = input(f"Drag the {what} video into this window and press Enter: ")
        p = p.strip().strip('"').strip()
        if p and os.path.isfile(p):
            return p
        print("  that's not a file, try again.")


def main():
    ap = argparse.ArgumentParser(description="Export a left/right fps comparison video.")
    ap.add_argument("videos", nargs="*",
                    help="the original and the smoothed video, any order")
    ap.add_argument("-o", "--out", help="output path (default: <smoothed>_sidebyside.mp4)")
    ap.add_argument("--labels", nargs=2, metavar=("LEFT", "RIGHT"),
                    default=("Original", "Smoothed"),
                    help="pane label words for the lower- and higher-fps video "
                         "(default: Original Smoothed); the measured fps is always appended")
    args = ap.parse_args()

    videos = [p for p in args.videos if os.path.isfile(p)]
    for bad in set(args.videos) - set(videos):
        print(f"skipping (not a file): {bad}")
    if len(videos) > 2:
        sys.exit("give exactly TWO videos: the original and the smoothed render.")
    if not videos:
        videos.append(ask_for_video("first"))
    if len(videos) == 1:
        videos.append(ask_for_video("other"))

    a, b = (probe(p) for p in videos)
    low, high = (a, b) if a["fps"] <= b["fps"] else (b, a)
    print(f"left  (original): {os.path.basename(low['path'])}  "
          f"{low['w']}x{low['h']} @ {low['fps']:.3f} fps")
    print(f"right (smoothed): {os.path.basename(high['path'])}  "
          f"{high['w']}x{high['h']} @ {high['fps']:.3f} fps")
    if low["fps"] == high["fps"]:
        print("note: both inputs have the same fps; keeping the given order.")
    if low["transfer"] != high["transfer"]:
        print(f"WARNING: colour transfer differs ({low['transfer'] or 'sdr'} vs "
              f"{high['transfer'] or 'sdr'}); the panes will not be colour-comparable.")

    height = max(low["h"], high["h"])
    out_path = args.out or unique_out(
        os.path.splitext(high["path"])[0] + "_sidebyside.mp4")
    fc = build_filter(low, high, height, tuple(args.labels))

    # hstack=shortest + -shortest: the output runs for the shorter input.
    duration = min(d for d in (low["duration"], high["duration"]) if d > 0) \
        if (low["duration"] > 0 or high["duration"] > 0) else 0
    total_frames = round(duration * high["fps"])

    print(f"encoding {low['w'] * height // low['h'] + high['w'] * height // high['h']}"
          f"x{height} @ {high['fps']:.3f} fps -> {out_path}")
    rc = run_encode(low, high, fc, out_path, "hevc_nvenc", total_frames)
    if rc != 0:
        print("hevc_nvenc failed (output too wide for NVENC?), retrying with "
              "SVT-AV1 on CPU...")
        rc = run_encode(low, high, fc, out_path, "libsvtav1", total_frames)
    if rc != 0:
        sys.exit("encode failed.")
    mb = os.path.getsize(out_path) / 1e6
    print(f"done: {out_path} ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
