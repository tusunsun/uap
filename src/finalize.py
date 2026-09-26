"""Mux picture + score into the delivery files.

    python3 src/finalize.py            # output/未知接触_UNIDENTIFIED_1080p.mp4 (fits GitHub's 100 MB limit)
    python3 src/finalize.py --hq       # also a high-bitrate master in build/
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import imageio_ffmpeg  # noqa: E402
from core import ROOT  # noqa: E402
from timeline import TOTAL  # noqa: E402

FF = imageio_ffmpeg.get_ffmpeg_exe()
BUILD = os.path.join(ROOT, "build")
OUT = os.path.join(ROOT, "output")


def run(cmd):
    print(" ".join(cmd[:6]), "...")
    subprocess.run(cmd, check=True)


def encode(src, wav, dst, target_mb=None, crf=None, preset="slow"):
    audio_kbps = 192
    common_v = ["-c:v", "libx264", "-preset", preset, "-tune", "grain", "-pix_fmt", "yuv420p", "-profile:v", "high",
                "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-movflags", "+faststart"]
    if target_mb:
        kbps = int(target_mb * 8192 / TOTAL - audio_kbps)
        log = os.path.join(BUILD, "x264pass")
        run([FF, "-y", "-loglevel", "error", "-i", src, *common_v, "-b:v", f"{kbps}k", "-pass", "1", "-passlogfile",
             log, "-an", "-f", "mp4", os.devnull])
        run([FF, "-y", "-loglevel", "error", "-i", src, "-i", wav, *common_v, "-b:v", f"{kbps}k", "-maxrate",
             f"{int(kbps * 2.2)}k", "-bufsize", f"{kbps * 4}k", "-pass", "2", "-passlogfile", log, "-c:a", "aac",
             "-b:a", f"{audio_kbps}k", "-shortest", "-metadata", "title=未知接触 UNIDENTIFIED", dst])
    else:
        run([FF, "-y", "-loglevel", "error", "-i", src, "-i", wav, *common_v, "-crf", str(crf), "-c:a", "aac", "-b:a",
             "256k", "-shortest", "-metadata", "title=未知接触 UNIDENTIFIED", dst])
    print("wrote", dst, f"{os.path.getsize(dst) / 1e6:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hq", action="store_true")
    ap.add_argument("--target-mb", type=float, default=92)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    src = os.path.join(BUILD, "video_noaudio.mp4")
    wav = os.path.join(BUILD, "score.wav")
    encode(src, wav, os.path.join(OUT, "未知接触_UNIDENTIFIED_1080p.mp4"), target_mb=a.target_mb)
    if a.hq:
        encode(src, wav, os.path.join(BUILD, "未知接触_UNIDENTIFIED_1080p_master.mp4"), crf=17)
