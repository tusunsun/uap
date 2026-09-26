"""Rendering core: frame buffer helpers, procedural noise, light, text and post-processing.

All frames are float32 arrays shaped (H, W, 3) in display space (0..1, values above 1
are highlights that get compressed by the tone curve in `to_uint8`).
"""
import functools
import math
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1920, 1080, 24
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_DIR = os.path.join(ROOT, "assets", "fonts")

FONTS = {
    "serif_black": "NotoSerifSC-Black.ttf",
    "serif_bold": "NotoSerifSC-Bold.ttf",
    "serif_light": "NotoSerifSC-Light.ttf",
    "sans_black": "NotoSansSC-Black.ttf",
    "sans_bold": "NotoSansSC-Bold.ttf",
    "sans": "NotoSansSC-Medium.ttf",
    "kai": "LXGWWenKai-Regular.ttf",
    "kai_bold": "LXGWWenKai-Bold.ttf",
    "mono": "JetBrainsMono-Regular.ttf",
    "mono_bold": "JetBrainsMono-Bold.ttf",
    "plex": "IBMPlexMono-Regular.ttf",
    "plex_bold": "IBMPlexMono-Bold.ttf",
    "hand": "NothingYouCouldDo-Regular.ttf",
    "baskerville": "LibreBaskerville-Regular.ttf",
    "gloock": "Gloock-Regular.ttf",
    "shoulders": "BigShoulders-Bold.ttf",
    "italiana": "Italiana-Regular.ttf",
}


@functools.lru_cache(None)
def font(name, size):
    return ImageFont.truetype(os.path.join(FONT_DIR, FONTS[name]), int(size))


# --------------------------------------------------------------------------- math

def clamp01(x):
    return min(1.0, max(0.0, x))


def smooth(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def ease_out(x, p=3):
    x = clamp01(x)
    return 1 - (1 - x) ** p


def ease_in(x, p=3):
    return clamp01(x) ** p


def ease_io(x):
    x = clamp01(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ramp(t, a, b):
    """0 before a, 1 after b, smooth in between."""
    if b <= a:
        return 1.0 if t >= a else 0.0
    return smooth((t - a) / (b - a))


def window(t, a, b, fin=0.4, fout=0.4):
    """Opacity envelope visible between a and b with fades."""
    return min(ramp(t, a, a + fin), 1 - ramp(t, b - fout, b))


def lerp(a, b, x):
    return a + (b - a) * x


def hash01(*args):
    h = 0
    for a in args:
        h = (h * 1000003 + int(a * 1000 + 7919)) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0x5BD1E995) & 0xFFFFFFFF
    h ^= h >> 15
    return (h & 0xFFFFFF) / float(0xFFFFFF)


@functools.lru_cache(None)
def grid(w=W, h=H):
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    return xs, ys


def blank(color=(0, 0, 0), w=W, h=H):
    img = np.empty((h, w, 3), np.float32)
    img[:] = np.array(color, np.float32)
    return img


def vgradient(top, bottom, w=W, h=H, power=1.0):
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None] ** power
    top = np.array(top, np.float32)
    bottom = np.array(bottom, np.float32)
    col = top * (1 - t) + bottom * t
    return np.broadcast_to(col, (h, w, 3)).copy()


# --------------------------------------------------------------------------- noise

def fbm(w, h, scale, octaves=5, persistence=0.5, seed=0, lacunarity=2.0):
    """Fractal value noise in [0, 1], `scale` = size in px of the largest feature."""
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, total, s = 1.0, 0.0, float(scale)
    for _ in range(octaves):
        cx = max(2, int(math.ceil(w / s)) + 3)
        cy = max(2, int(math.ceil(h / s)) + 3)
        g = rng.random((cy, cx)).astype(np.float32)
        up = cv2.resize(g, (int(cx * s), int(cy * s)), interpolation=cv2.INTER_CUBIC)
        ox = int(rng.integers(0, max(1, up.shape[1] - w)))
        oy = int(rng.integers(0, max(1, up.shape[0] - h)))
        out += amp * up[oy:oy + h, ox:ox + w]
        total += amp
        amp *= persistence
        s /= lacunarity
        if s < 1.5:
            break
    out /= total
    lo, hi = np.percentile(out, 0.5), np.percentile(out, 99.5)
    return np.clip((out - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)


def noise1d(n, scale, octaves=4, seed=0, persistence=0.5):
    rng = np.random.default_rng(seed)
    out = np.zeros(n, np.float32)
    amp, total, s = 1.0, 0.0, float(scale)
    for _ in range(octaves):
        c = max(2, int(n / s) + 3)
        g = rng.random(c).astype(np.float32)
        x = np.linspace(0, (n - 1) / s, n)
        i = np.floor(x).astype(int)
        f = x - i
        f = f * f * (3 - 2 * f)
        out += amp * (g[i] * (1 - f) + g[i + 1] * f)
        total += amp
        amp *= persistence
        s /= 2
        if s < 1:
            break
    return out / total


def ridge_mask(w, h, base, amp, scale, seed=0, octaves=5, sharp=False, persistence=0.5):
    """Silhouette mask (1 = ground) below a noisy horizon line."""
    n = noise1d(w, scale, octaves, seed, persistence)
    if sharp:
        n = 1 - np.abs(n * 2 - 1)
    top = base - amp * n
    ys = np.arange(h, dtype=np.float32)[:, None]
    return np.clip(ys - top[None, :] + 0.5, 0, 1).astype(np.float32), top


# --------------------------------------------------------------------------- light

def add_glow(img, x, y, r, color, intensity=1.0, kind="gauss", reach=None):
    """Additive radial light. kind='gauss' (soft ball) or 'halo' (1/(1+d^2) wide falloff)."""
    h, w = img.shape[:2]
    if reach is None:
        reach = 3.2 * r if kind == "gauss" else 14 * r
    x0, x1 = int(max(0, x - reach)), int(min(w, x + reach + 1))
    y0, y1 = int(max(0, y - reach)), int(min(h, y + reach + 1))
    if x0 >= x1 or y0 >= y1 or r <= 0:
        return img
    xs = np.arange(x0, x1, dtype=np.float32)[None, :] - x
    ys = np.arange(y0, y1, dtype=np.float32)[:, None] - y
    d2 = (xs * xs + ys * ys) / (r * r)
    if kind == "gauss":
        g = np.exp(-d2)
    else:
        g = 1.0 / (1.0 + d2)
        g -= 1.0 / (1.0 + (reach / r) ** 2)
        np.maximum(g, 0, out=g)
    img[y0:y1, x0:x1] += g[..., None] * (np.array(color, np.float32) * intensity)
    return img


def add_ellipse_glow(img, x, y, rx, ry, angle, color, intensity=1.0, reach=3.0):
    h, w = img.shape[:2]
    R = max(rx, ry) * reach
    x0, x1 = int(max(0, x - R)), int(min(w, x + R + 1))
    y0, y1 = int(max(0, y - R)), int(min(h, y + R + 1))
    if x0 >= x1 or y0 >= y1:
        return img
    xs = np.arange(x0, x1, dtype=np.float32)[None, :] - x
    ys = np.arange(y0, y1, dtype=np.float32)[:, None] - y
    c, s = math.cos(angle), math.sin(angle)
    u = (xs * c + ys * s) / rx
    v = (-xs * s + ys * c) / ry
    g = np.exp(-(u * u + v * v))
    img[y0:y1, x0:x1] += g[..., None] * (np.array(color, np.float32) * intensity)
    return img


def blur(img, sigma):
    if sigma <= 0:
        return img
    if sigma > 6:
        f = 4
        h, w = img.shape[:2]
        small = cv2.resize(img, (w // f, h // f), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), sigma / f)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(img, (0, 0), sigma)


def bloom(img, thresh=0.75, strength=0.6, radius=40, wide=1.0):
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    b = np.maximum(small - thresh, 0)
    b1 = cv2.GaussianBlur(b, (0, 0), radius / 16)
    b2 = cv2.GaussianBlur(b, (0, 0), radius / 4)
    bb = b1 * 0.55 + b2 * 0.45 * wide
    img += cv2.resize(bb, (w, h), interpolation=cv2.INTER_LINEAR) * strength
    return img


def splat_points(img, xs, ys, cols):
    """Bilinear additive splat of N points (xs, ys float arrays, cols (N,3))."""
    h, w = img.shape[:2]
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    fx = (xs - x0).astype(np.float32)
    fy = (ys - y0).astype(np.float32)
    flat = img.reshape(-1, 3)
    for dx, dy, wt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xx, yy = x0 + dx, y0 + dy
        ok = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        if not ok.any():
            continue
        idx = (yy[ok] * w + xx[ok])
        for c in range(3):
            flat[:, c] += np.bincount(idx, weights=cols[ok, c] * wt[ok], minlength=h * w).astype(np.float32)
    return img


def star_layer(w, h, n, seed=0, density=None, mag_pow=3.0, bright=1.0, tint=True, glow=True):
    """Static star texture. `density` optional (h,w) map biasing star placement."""
    rng = np.random.default_rng(seed)
    if density is not None:
        p = density.ravel().astype(np.float64) + 1e-9
        p /= p.sum()
        idx = rng.choice(p.size, n, p=p)
        ys = (idx // w).astype(np.float32) + rng.random(n)
        xs = (idx % w).astype(np.float32) + rng.random(n)
    else:
        xs = rng.random(n) * w
        ys = rng.random(n) * h
    mag = rng.random(n) ** mag_pow * bright
    if tint:
        temp = rng.random(n)
        cols = np.stack([
            0.85 + 0.15 * temp,
            0.85 + 0.1 * rng.random(n),
            1.0 - 0.25 * temp,
        ], 1) * mag[:, None]
    else:
        cols = np.repeat(mag[:, None], 3, 1)
    img = np.zeros((h, w, 3), np.float32)
    splat_points(img, xs, ys, cols.astype(np.float32))
    if glow:
        big = mag > 0.55 * bright
        g = np.zeros_like(img)
        splat_points(g, xs[big], ys[big], cols[big].astype(np.float32) * 6)
        img += cv2.GaussianBlur(g, (0, 0), 2.2) + cv2.GaussianBlur(g, (0, 0), 7) * 0.4
    return img


def warp(img, scale=1.0, angle=0.0, cx=None, cy=None, tx=0.0, ty=0.0, out_w=W, out_h=H, border=cv2.BORDER_REFLECT):
    """Scale/rotate `img` about its point (cx, cy) and place that point at output centre + (tx, ty)."""
    h, w = img.shape[:2]
    if cx is None:
        cx = w / 2
    if cy is None:
        cy = h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    M[0, 2] += out_w / 2 - cx + tx
    M[1, 2] += out_h / 2 - cy + ty
    return cv2.warpAffine(img, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=border)


def shake_offsets(t, amp, freq=9.0, seed=0):
    a = math.sin(t * freq * 1.3 + seed) * 0.6 + math.sin(t * freq * 2.9 + seed * 3) * 0.4
    b = math.sin(t * freq * 1.7 + seed * 5) * 0.6 + math.sin(t * freq * 3.3 + seed * 7) * 0.4
    return a * amp, b * amp


def shake(img, dx, dy, rot=0.0, zoom=1.0):
    if abs(dx) < 0.05 and abs(dy) < 0.05 and rot == 0 and zoom == 1.0:
        return img
    return warp(img, zoom, rot, tx=dx, ty=dy)


# --------------------------------------------------------------------------- text

@functools.lru_cache(maxsize=4096)
def text_patch(text, fname, size, color=(255, 255, 255), tracking=0.0, stroke=0, stroke_color=(0, 0, 0)):
    """Render text -> (rgb float32 (h,w,3), alpha float32 (h,w)). tracking in em units."""
    f = font(fname, size)
    pad = int(size * 0.35) + stroke * 2
    if tracking == 0:
        l, t, r, b = f.getbbox(text, stroke_width=stroke)
        tw = r
        th_top, th_bot = min(t, 0), b
    else:
        adv = [f.getlength(ch) + tracking * size for ch in text]
        tw = int(sum(adv) - tracking * size) + 1
        asc, desc = f.getmetrics()
        th_top, th_bot = 0, asc + desc
    asc, desc = f.getmetrics()
    hh = max(th_bot, asc + desc) - th_top + 2 * pad
    ww = int(tw) + 2 * pad
    im = Image.new("L", (max(1, ww), max(1, hh)), 0)
    d = ImageDraw.Draw(im)
    if stroke:
        sim = Image.new("L", im.size, 0)
        sd = ImageDraw.Draw(sim)
    if tracking == 0:
        d.text((pad, pad - th_top), text, font=f, fill=255)
        if stroke:
            sd.text((pad, pad - th_top), text, font=f, fill=255, stroke_width=stroke, stroke_fill=255)
    else:
        x = pad
        for ch in text:
            d.text((x, pad - th_top), ch, font=f, fill=255)
            if stroke:
                sd.text((x, pad - th_top), ch, font=f, fill=255, stroke_width=stroke, stroke_fill=255)
            x += f.getlength(ch) + tracking * size
    a = np.asarray(im, np.float32) / 255.0
    col = np.array(color, np.float32) / 255.0
    rgb = np.broadcast_to(col, a.shape + (3,)).copy()
    if stroke:
        sa = np.asarray(sim, np.float32) / 255.0
        scol = np.array(stroke_color, np.float32) / 255.0
        rgb = rgb * a[..., None] + scol * (1 - a[..., None])
        a = np.maximum(a, sa)
    rgb.setflags(write=False)
    a.setflags(write=False)
    return rgb, a, pad


def text_size(text, fname, size, tracking=0.0):
    f = font(fname, size)
    if tracking == 0:
        return f.getlength(text)
    return sum(f.getlength(ch) + tracking * size for ch in text) - tracking * size


def blit(img, rgb, a, x, y, opacity=1.0, mode="normal"):
    if opacity <= 0.002:
        return img
    h, w = img.shape[:2]
    ph, pw = a.shape
    x, y = int(round(x)), int(round(y))
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + pw), min(h, y + ph)
    if x0 >= x1 or y0 >= y1:
        return img
    sa = a[y0 - y:y1 - y, x0 - x:x1 - x][..., None] * opacity
    sr = rgb[y0 - y:y1 - y, x0 - x:x1 - x]
    reg = img[y0:y1, x0:x1]
    if mode == "add":
        reg += sr * sa
    elif mode == "multiply":
        reg *= 1 - sa + sr * sa
    else:
        reg *= 1 - sa
        reg += sr * sa
    return img


def draw_text(img, text, x, y, fname, size, color=(255, 255, 255), opacity=1.0, anchor="lt",
              tracking=0.0, mode="normal", stroke=0, stroke_color=(0, 0, 0), shadow=0.0):
    """anchor: first char l/c/r, second t/m/b (t = top of em box, b = baseline-ish bottom)."""
    rgb, a, pad = text_patch(text, fname, size, tuple(color), tracking, stroke, tuple(stroke_color))
    tw = a.shape[1] - 2 * pad
    asc, desc = font(fname, size).getmetrics()
    if anchor[0] == "c":
        x -= tw / 2
    elif anchor[0] == "r":
        x -= tw
    if anchor[1] == "m":
        y -= (asc + desc) / 2
    elif anchor[1] == "b":
        y -= asc
    if shadow > 0:
        sh = np.zeros_like(rgb)
        blit(img, sh, cv2.GaussianBlur(np.ascontiguousarray(a), (0, 0), size * 0.08 + 1), x - pad + size * 0.04,
             y - pad + size * 0.06, opacity * shadow)
    blit(img, rgb, a, x - pad, y - pad, opacity, mode)
    return tw


def darken_region(img, x0, y0, x1, y1, amount=0.6, soft=120):
    """Soft dark scrim behind text."""
    h, w = img.shape[:2]
    X0, Y0 = int(max(0, x0 - soft)), int(max(0, y0 - soft))
    X1, Y1 = int(min(w, x1 + soft)), int(min(h, y1 + soft))
    if X0 >= X1 or Y0 >= Y1:
        return img
    xs = np.arange(X0, X1, dtype=np.float32)
    ys = np.arange(Y0, Y1, dtype=np.float32)
    mx = np.clip(np.minimum(xs - (x0 - soft), (x1 + soft) - xs) / soft, 0, 1)
    my = np.clip(np.minimum(ys - (y0 - soft), (y1 + soft) - ys) / soft, 0, 1)
    m = (my[:, None] * mx[None, :])
    m = m * m * (3 - 2 * m)
    img[Y0:Y1, X0:X1] *= (1 - amount * m)[..., None]
    return img


# ------------------------------------------------------------------ documentary graphics

RED = (214, 40, 40)
AMBER = (255, 176, 58)


def caption(img, t, t0, t1, tag, title=None, body=(), source=None, x=110, bottom=H - 150, scrim=0.55,
            title_size=50, body_size=30, width_hint=None, accent=RED, title_font="serif_bold"):
    """Lower-left documentary caption block with staggered animated reveal."""
    if t < t0 or t > t1:
        return img
    rows = []
    rows.append(("tag", tag, "sans_bold", 26, 0.0))
    if title:
        rows.append(("title", title, title_font, title_size, 0.18))
    for i, line in enumerate(body):
        rows.append(("body", line, "serif_light", body_size, 0.45 + 0.22 * i))
    if source:
        rows.append(("src", source, "serif_light", 22, 0.55 + 0.22 * len(body)))
    gaps = {"tag": 18, "title": 18, "body": 10, "src": 0}
    heights = {"tag": 36, "title": title_size * 1.3, "body": body_size * 1.45, "src": 30}
    total = sum(heights[r[0]] + gaps[r[0]] for r in rows)
    y = bottom - total
    out = 1 - ramp(t, t1 - 0.45, t1)
    if scrim > 0:
        wmax = max(text_size(r[1], r[2], r[3], 0.12 if r[0] == "tag" else 0.02) for r in rows)
        darken_region(img, x - 20, y - 10, x + wmax + 20, bottom + 10, scrim * min(1, ramp(t, t0, t0 + 0.4)) * out, 160)
    for kind, text, fname, size, delay in rows:
        a = ramp(t, t0 + delay, t0 + delay + 0.45) * out
        dy = (1 - ease_out((t - t0 - delay) / 0.6)) * 14
        if kind == "tag":
            barh = 30 * ease_out((t - t0) / 0.35)
            img[int(y + 3):int(y + 3 + barh), int(x):int(x + 6)] = np.array(accent, np.float32) / 255 * out + \
                img[int(y + 3):int(y + 3 + barh), int(x):int(x + 6)] * (1 - out)
            slide = (1 - ease_out((t - t0) / 0.5)) * -24
            draw_text(img, text, x + 20 + slide, y, fname, size, (235, 235, 235), a, tracking=0.12)
        elif kind == "title":
            draw_text(img, text, x, y + dy, fname, size, (255, 255, 255), a, tracking=0.02, shadow=0.6)
        elif kind == "body":
            draw_text(img, text, x, y + dy, fname, size, (222, 222, 222), a, tracking=0.02, shadow=0.5)
        else:
            draw_text(img, text, x, y + dy + 6, fname, size, (170, 170, 170), a, tracking=0.02)
        y += heights[kind] + gaps[kind]
    return img


def subtitle(img, t, t0, t1, zh, en=None, y=H - 120, color=(255, 255, 255)):
    if t < t0 or t > t1:
        return img
    a = window(t, t0, t1, 0.12, 0.2)
    draw_text(img, zh, W / 2, y - (46 if en else 0), "sans_bold", 44, color, a, anchor="ct", shadow=0.9,
              stroke=2, stroke_color=(0, 0, 0))
    if en:
        draw_text(img, en, W / 2, y + 12, "sans", 28, (215, 215, 215), a, anchor="ct", shadow=0.9)
    return img


def recreation_tag(img, opacity=1.0, text="情景再现  RECREATION", x=None, y=60):
    if opacity <= 0:
        return img
    tw = text_size(text, "sans", 20, 0.1)
    if x is None:
        x = W - 110 - tw
    xx, yy = int(x - 14), int(y - 8)
    img[yy:yy + 40, xx:xx + int(tw) + 28] *= 1 - 0.45 * opacity
    col = np.array((1, 1, 1), np.float32) * 0.8 * opacity
    img[yy, xx:xx + int(tw) + 28] += col
    img[yy + 39, xx:xx + int(tw) + 28] += col
    img[yy:yy + 40, xx] += col
    img[yy:yy + 40, xx + int(tw) + 27] += col
    draw_text(img, text, x, y, "sans", 20, (230, 230, 230), opacity * 0.9, tracking=0.1)
    return img


def chapter_card(img, t, t0, t1, num, zh, en, y=H / 2):
    if t < t0 or t > t1:
        return img
    a = window(t, t0, t1, 0.8, 0.8)
    k = ease_out((t - t0) / 3.0)
    draw_text(img, num, W / 2, y - 120, "mono", 26, (200, 200, 200), a, anchor="ct", tracking=0.6)
    draw_text(img, zh, W / 2, y - 60, "serif_black", 96, (255, 255, 255), a, anchor="ct",
              tracking=0.35 + 0.15 * (1 - k), shadow=0.7)
    lw = int(360 * ease_out((t - t0 - 0.4) / 1.2))
    if lw > 0:
        img[int(y + 78):int(y + 80), int(W / 2 - lw):int(W / 2 + lw)] += np.float32(0.8 * a)
    draw_text(img, en, W / 2, y + 100, "mono", 24, (190, 190, 190), a, anchor="ct", tracking=0.5)
    return img


def chapter_tag(img, t, t0, t1, text, x=110, y=64):
    a = window(t, t0, t1, 0.6, 0.6)
    if a <= 0:
        return img
    draw_text(img, text, x, y, "mono", 20, (220, 220, 220), a * 0.85, tracking=0.35)
    return img


# --------------------------------------------------------------------------- post

@functools.lru_cache(None)
def _vignette(strength, w=W, h=H):
    xs, ys = grid(w, h)
    u = (xs - w / 2) / (w / 2)
    v = (ys - h / 2) / (h / 2)
    r2 = u * u * 0.8 + v * v
    m = 1 - strength * np.clip(r2 * 0.75, 0, 1.4) ** 1.5
    return np.clip(m, 0, 1)[..., None].astype(np.float32)


def vignette(img, strength=0.5):
    img *= _vignette(round(strength, 2), img.shape[1], img.shape[0])
    return img


@functools.lru_cache(None)
def _grain_bank(n=10, seed=11):
    rng = np.random.default_rng(seed)
    bank = []
    for _ in range(n):
        g = rng.standard_normal((H // 2 + 2, W // 2 + 2)).astype(np.float32)
        g = cv2.GaussianBlur(g, (0, 0), 0.6)
        g = cv2.resize(g, (W + 4, H + 4), interpolation=cv2.INTER_LINEAR)
        bank.append(g / (g.std() + 1e-6))
    return bank


def grain(img, amount, frame):
    if amount <= 0:
        return img
    bank = _grain_bank()
    g = bank[frame % len(bank)]
    ox, oy = (frame * 7) % 4, (frame * 3) % 4
    gg = g[oy:oy + H, ox:ox + W]
    lum = np.clip(img.mean(axis=2), 0, 1)
    k = amount * (0.12 + 2.6 * lum * (1 - lum))
    img += (gg * k)[..., None]
    return img


def chroma_shift(img, px):
    px = int(round(px))
    if px <= 0:
        return img
    out = img.copy()
    out[:, px:, 0] = img[:, :-px, 0]
    out[:, :-px, 2] = img[:, px:, 2]
    return out


def letterbox(img, ratio=2.39, amount=1.0):
    if amount <= 0:
        return img
    bar = int(round((H - W / ratio) / 2 * amount))
    if bar > 0:
        img[:bar] = 0
        img[H - bar:] = 0
    return img


def pillarbox(img, ratio=4 / 3, soft=0):
    bw = int((W - H * ratio) / 2)
    if bw > 0:
        img[:, :bw] = 0
        img[:, W - bw:] = 0
    return img


def desaturate(img, amount=1.0, tint=None):
    lum = img @ np.array([0.299, 0.587, 0.114], np.float32)
    grey = lum[..., None]
    if tint is not None:
        grey = grey * np.array(tint, np.float32)
    if amount >= 1:
        return np.ascontiguousarray(np.broadcast_to(grey, img.shape)) if tint is None else grey
    return img * (1 - amount) + grey * amount


def grade(img, lift=(0, 0, 0), gamma=(1, 1, 1), gain=(1, 1, 1)):
    img = img * np.array(gain, np.float32) + np.array(lift, np.float32)
    g = np.array(gamma, np.float32)
    if np.any(g != 1):
        np.maximum(img, 0, out=img)
        img = img ** (1 / g)
    return img


def contrast(img, c=1.2, pivot=0.4):
    return (img - pivot) * c + pivot


def flicker(img, t, amount=0.05, seed=0):
    f = 1 + amount * (hash01(int(t * FPS), seed) - 0.5) * 2
    img *= f
    return img


def scanlines(img, amount=0.15, period=3):
    img[::period] *= 1 - amount
    return img


def glitch(img, amount, frame, seed=0):
    """Horizontal band displacement + RGB split. amount 0..1"""
    if amount <= 0.01:
        return img
    rng = np.random.default_rng(frame * 131 + seed)
    out = img.copy()
    nb = int(3 + amount * 14)
    for _ in range(nb):
        y0 = int(rng.integers(0, H - 10))
        hh = int(rng.integers(4, int(20 + 120 * amount)))
        dx = int(rng.normal(0, 60 * amount))
        band = img[y0:y0 + hh]
        out[y0:y0 + hh] = np.roll(band, dx, axis=1)
        if rng.random() < 0.3 * amount:
            out[y0:y0 + hh] *= np.array([rng.random() * 1.5, rng.random(), rng.random() * 1.5], np.float32)
    return chroma_shift(out, 4 + 22 * amount)


def to_uint8(img):
    x = np.maximum(img, 0)
    knee = 0.82
    hi = x > knee
    if hi.any():
        x = np.where(hi, knee + (1 - knee) * np.tanh((x - knee) / (1 - knee)), x)
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


# --------------------------------------------------------------------------- misc drawing

def pil_canvas(w=W, h=H, mode="L"):
    im = Image.new(mode, (w, h), 0)
    return im, ImageDraw.Draw(im)


def pil_to_alpha(im):
    return np.asarray(im, np.float32) / 255.0


def add_mask_color(img, mask, color, opacity=1.0, mode="normal"):
    col = np.array(color, np.float32)
    m = mask[..., None] * opacity
    if mode == "add":
        img += m * col
    else:
        img *= 1 - m
        img += m * col
    return img


def aa_mask_from_poly(polys, w=W, h=H, ss=2):
    """Anti-aliased polygon mask using supersampled cv2 fill."""
    m = np.zeros((h * ss, w * ss), np.uint8)
    for p in polys:
        pts = (np.asarray(p, np.float32) * ss).astype(np.int32)
        cv2.fillPoly(m, [pts], 255, lineType=cv2.LINE_AA)
    if ss > 1:
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA)
    return m.astype(np.float32) / 255.0
