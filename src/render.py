"""Frame renderer / encoder.

    python3 src/render.py --still 33.5 41        # PNG stills into build/stills
    python3 src/render.py --sheet                 # contact sheet of every scene
    python3 src/render.py --range 30 40 [--half]  # preview clip (no audio)
    python3 src/render.py --full [--jobs 4]       # full film -> build/video_noaudio.mp4
"""
import argparse
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import imageio_ffmpeg  # noqa: E402
from core import FPS, H, W, ROOT, chroma_shift, grain, letterbox, to_uint8, vignette, ease_out  # noqa: E402
from timeline import TIMELINE, TOTAL  # noqa: E402

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
BUILD = os.path.join(ROOT, "build")


def locate(t):
    for i, (start, sc) in enumerate(TIMELINE):
        if start <= t < start + sc.dur:
            return i
    return len(TIMELINE) - 1


def render_scene(i, t, f):
    start, sc = TIMELINE[i]
    sc.ensure()
    lt = t - start
    img = sc.render(lt, f)
    if sc.vig:
        img = vignette(img, sc.vig)
    if sc.grain:
        img = grain(img, sc.grain, f)
    if sc.ca:
        img = chroma_shift(img, sc.ca)
    img = sc.overlay(img, lt, f)
    if sc.lbox:
        img = letterbox(img, sc.lbox)
    return img


def render_frame(f):
    t = f / FPS
    i = locate(t)
    start, sc = TIMELINE[i]
    lt = t - start
    img = render_scene(i, t, f)
    if sc.xfade and lt < sc.xfade and i > 0:
        prev = render_scene(i - 1, t, f)
        k = lt / sc.xfade
        k = k * k * (3 - 2 * k)
        img = prev * (1 - k) + img * k
    if sc.fade_in and lt < sc.fade_in:
        img *= (lt / sc.fade_in) ** 1.5
    if sc.fade_out and lt > sc.dur - sc.fade_out:
        img *= max(0.0, (sc.dur - lt) / sc.fade_out) ** 1.5
    if sc.flash_in and lt < sc.flash_in:
        img += (1 - ease_out(lt / sc.flash_in, 2)) * 1.2
    return to_uint8(img)


def ffmpeg_writer(path, crf=12, preset="veryfast", w=W, h=H):
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(FPS), "-i", "-", "-vf", "scale=out_color_matrix=bt709:out_range=tv",
           "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
           "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def render_segment(args):
    seg, f0, f1, path, half = args
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return seg, 0.0
    t0 = time.time()
    w, h = (W // 2, H // 2) if half else (W, H)
    tmp = path + ".part.mp4"
    p = ffmpeg_writer(tmp, w=w, h=h)
    for f in range(f0, f1):
        fr = render_frame(f)
        if half:
            import cv2
            fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    p.wait()
    os.replace(tmp, path)
    return seg, time.time() - t0


def render_range(t0, t1, out, jobs=4, half=False, seglen=48, tag="seg"):
    f0, f1 = int(round(t0 * FPS)), int(round(t1 * FPS))
    segdir = os.path.join(BUILD, tag)
    os.makedirs(segdir, exist_ok=True)
    tasks = []
    for k, s in enumerate(range(f0, f1, seglen)):
        e = min(f1, s + seglen)
        tasks.append((k, s, e, os.path.join(segdir, f"{s:06d}_{e:06d}{'_h' if half else ''}.mp4"), half))
    t_start = time.time()
    if jobs > 1:
        with mp.Pool(jobs) as pool:
            for n, (seg, dt) in enumerate(pool.imap_unordered(render_segment, tasks)):
                print(f"  segment {seg + 1}/{len(tasks)} done in {dt:.1f}s  (elapsed {time.time() - t_start:.0f}s)",
                      flush=True)
    else:
        for task in tasks:
            seg, dt = render_segment(task)
            print(f"  segment {seg + 1}/{len(tasks)} done in {dt:.1f}s", flush=True)
    lst = os.path.join(segdir, "list.txt")
    with open(lst, "w") as fh:
        for task in tasks:
            fh.write(f"file '{task[3]}'\n")
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", out],
                   check=True)
    print("wrote", out)


def stills(times, outdir):
    from PIL import Image
    os.makedirs(outdir, exist_ok=True)
    for t in times:
        f = int(round(t * FPS))
        t0 = time.time()
        fr = render_frame(f)
        p = os.path.join(outdir, f"still_{t:07.2f}.png")
        Image.fromarray(fr).save(p)
        print(f"{p}  ({time.time() - t0:.2f}s)")


def sheet(outpath, per_scene=3, cols=6):
    from PIL import Image
    import cv2
    thumbs = []
    for start, sc in TIMELINE:
        for k in range(per_scene):
            t = start + sc.dur * (k + 0.5) / per_scene
            fr = render_frame(int(round(t * FPS)))
            thumbs.append(cv2.resize(fr, (W // 5, H // 5), interpolation=cv2.INTER_AREA))
    rows = math.ceil(len(thumbs) / cols)
    tw, th = W // 5, H // 5
    canvas = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for n, im in enumerate(thumbs):
        r, c = divmod(n, cols)
        canvas[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = im
    Image.fromarray(canvas).save(outpath)
    print("wrote", outpath)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--still", nargs="*", type=float)
    ap.add_argument("--range", nargs=2, type=float)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    os.makedirs(BUILD, exist_ok=True)
    if a.still:
        stills(a.still, os.path.join(BUILD, "stills"))
    if a.sheet:
        sheet(a.out or os.path.join(BUILD, "sheet.png"))
    if a.range:
        out = a.out or os.path.join(BUILD, f"preview_{a.range[0]:.1f}_{a.range[1]:.1f}.mp4")
        render_range(a.range[0], a.range[1], out, a.jobs, a.half, tag="preview")
    if a.full:
        render_range(0, TOTAL, a.out or os.path.join(BUILD, "video_noaudio.mp4"), a.jobs, a.half, tag="full")
