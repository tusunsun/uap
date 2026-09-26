"""Reusable procedural building blocks: IR targeting-pod display, skies, paper, radar."""
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw

from core import (FPS, H, W, add_glow, add_ellipse_glow, fbm, font, grid, hash01, smooth, splat_points,
                  star_layer, clamp01)


# ============================================================ IR targeting pod (FLIR) view

class FlirView:
    """Grey-scale infrared targeting-pod view with HUD symbology (half-res sensor, full-res HUD)."""

    def __init__(self, mode="gimbal", seed=3):
        self.mode = mode
        self.sw, self.sh = 960, 540
        rng = np.random.default_rng(seed)
        self.rng = rng
        if mode == "gimbal":
            self.cloud = fbm(3200, 900, 260, 6, 0.55, seed)
            self.cloud2 = fbm(3200, 900, 90, 5, 0.5, seed + 1)
        elif mode == "gofast":
            n = fbm(3000, 1400, 60, 5, 0.6, seed)
            s = fbm(3000, 1400, 14, 3, 0.6, seed + 2)
            self.sea = (n * 0.7 + s * 0.3)
            self.sea = cv2.GaussianBlur(self.sea, (0, 0), 1.0)
        elif mode == "hellfire":
            n = fbm(2400, 1400, 120, 5, 0.55, seed)
            s = fbm(2400, 1400, 20, 3, 0.6, seed + 2)
            self.sea = n * 0.6 + s * 0.4
        else:  # tictac: near uniform sky
            self.sky = fbm(1400, 800, 300, 4, 0.5, seed)
        self.noise = [rng.standard_normal((self.sh, self.sw)).astype(np.float32) for _ in range(6)]

    # ---- sensor image (0..1 grey, half res)
    def sensor(self, t, obj, bank=0.0):
        sw, sh = self.sw, self.sh
        if self.mode == "gimbal":
            ox = int(t * 170) % 1600
            c = self.cloud[:, ox:ox + 1800]
            c2 = self.cloud2[:, ox:ox + 1800]
            big = np.zeros((900, 1800), np.float32)
            ys = np.linspace(0, 1, 900, dtype=np.float32)[:, None]
            horizon = 0.62
            dens = np.clip((c * 0.7 + c2 * 0.3 - 0.48) * 3.2, 0, 1) * np.clip((ys - horizon + 0.05) * 8, 0, 1)
            sky = 0.56 + 0.10 * (1 - ys)
            big[:] = sky
            big = big * (1 - dens) + (0.78 + 0.12 * c2) * dens
            M = cv2.getRotationMatrix2D((900, 450), bank, 1.0)
            M[0, 2] += sw / 2 - 900
            M[1, 2] += sh / 2 - 450 + 40
            img = cv2.warpAffine(big, M, (sw, sh), borderMode=cv2.BORDER_REFLECT)
        elif self.mode == "gofast":
            ox = int(t * 520) % 1500
            oy = int(t * 180) % 600
            img = self.sea[oy:oy + sh, ox:ox + sw].copy()
            img = 0.30 + 0.35 * img
        elif self.mode == "hellfire":
            ox = int(t * 60) % 1200
            oy = int(600 + 40 * math.sin(t * 0.3))
            img = self.sea[oy:oy + sh, ox:ox + sw].copy()
            img = 0.18 + 0.25 * img
        else:
            img = cv2.resize(self.sky[100:100 + 540, 200:200 + 960], (sw, sh))
            ys = np.linspace(0, 1, sh, dtype=np.float32)[:, None]
            img = 0.50 + 0.06 * img + 0.05 * ys
            img = np.broadcast_to(img, (sh, sw)).copy()
        if obj is not None:
            self.draw_object(img, t, **obj)
        return img

    def draw_object(self, img, t, x, y, size, angle=0.0, polarity=-1, kind="saucer", aura=1.0, alpha=1.0):
        sw, sh = self.sw, self.sh
        x, y = x * sw, y * sh
        if not (-200 < x < sw + 200 and -200 < y < sh + 200):
            return
        m = np.zeros((sh * 2, sw * 2), np.uint8)
        c = (int(x * 2), int(y * 2))
        s2 = size * 2
        if kind == "saucer":
            cv2.ellipse(m, c, (int(s2), int(s2 * 0.34)), angle, 0, 360, 255, -1, cv2.LINE_AA)
            ca, sa = math.cos(math.radians(angle)), math.sin(math.radians(angle))
            for sgn in (-1, 1):
                px = int(c[0] + sgn * ca * s2 * 0.62)
                py = int(c[1] + sgn * sa * s2 * 0.62)
                cv2.ellipse(m, (px, py), (int(s2 * 0.34), int(s2 * 0.40)), angle, 0, 360, 255, -1, cv2.LINE_AA)
            cv2.ellipse(m, c, (int(s2 * 0.42), int(s2 * 0.52)), angle, 0, 360, 255, -1, cv2.LINE_AA)
        elif kind == "tictac":
            cv2.ellipse(m, c, (int(s2), int(s2 * 0.42)), angle, 0, 360, 255, -1, cv2.LINE_AA)
        else:  # blob / orb
            cv2.circle(m, c, int(s2), 255, -1, cv2.LINE_AA)
        mk = cv2.resize(m, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        mk = cv2.GaussianBlur(mk, (0, 0), 0.9)
        glow = cv2.GaussianBlur(mk, (0, 0), size * 0.9 + 2)
        if polarity < 0:   # black-hot: dark body, light halo
            img += glow * 0.35 * aura * alpha
            img *= 1 - mk * 0.85 * alpha
        else:              # white-hot
            img += glow * 0.55 * aura * alpha
            img += mk * 0.6 * alpha

    def finish(self, img, f, polarity_flip=False, blur=0.7, noise=0.028):
        if polarity_flip:
            img = 1.05 - img
        img = cv2.GaussianBlur(img, (0, 0), blur)
        img = img + self.noise[f % 6] * noise
        img[::2] *= 0.985
        big = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        return np.repeat(big[..., None], 3, axis=2)

    # ---- HUD
    def hud(self, img, t, f, label_mode="IR", zoom="NAR", polarity="BHT", gate=None, bank=0.0,
            heading=0.0, rng_nm=None, alt=25000, mach=0.62, az=None, el=None, extra=(), opacity=1.0):
        im = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(im)
        cx, cy = W // 2, H // 2
        fm = font("plex", 30)
        fs = font("plex", 24)
        white = 255
        # centre reticle
        g = 44
        for sx, sy in ((1, 0), (-1, 0)):
            d.line([(cx + sx * g, cy), (cx + sx * 190, cy)], fill=white, width=3)
            for k in range(1, 5):
                xx = cx + sx * (g + k * 36)
                d.line([(xx, cy - 8), (xx, cy + 8)], fill=white, width=2)
        d.line([(cx, cy - g), (cx, cy - 130)], fill=white, width=3)
        d.line([(cx, cy + g), (cx, cy + 130)], fill=white, width=3)
        # artificial horizon (banked)
        a = math.radians(bank)
        for sgn in (-1, 1):
            x0 = cx + sgn * 260 * math.cos(a)
            y0 = cy + sgn * 260 * math.sin(a)
            x1 = cx + sgn * 560 * math.cos(a)
            y1 = cy + sgn * 560 * math.sin(a)
            d.line([(x0, y0 + 60), (x1, y1 + 60)], fill=white, width=3)
            d.line([(x1, y1 + 60), (x1 - sgn * 0 + 14 * math.sin(a) * sgn, y1 + 60 + 26)], fill=white, width=3)
        # outer corner brackets
        m = 120
        L = 70
        for (x, y, dx, dy) in ((m, m, 1, 1), (W - m, m, -1, 1), (m, H - m, 1, -1), (W - m, H - m, -1, -1)):
            d.line([(x, y), (x + dx * L, y)], fill=white, width=3)
            d.line([(x, y), (x, y + dy * L)], fill=white, width=3)
        # tracking gate
        if gate is not None:
            gx, gy, gw, gh = gate
            gx *= W
            gy *= H
            l2 = 26
            for (x, y, dx, dy) in ((gx - gw, gy - gh, 1, 1), (gx + gw, gy - gh, -1, 1), (gx - gw, gy + gh, 1, -1),
                                   (gx + gw, gy + gh, -1, -1)):
                d.line([(x, y), (x + dx * l2, y)], fill=white, width=3)
                d.line([(x, y), (x, y + dy * l2)], fill=white, width=3)
        # heading tape
        base = heading
        for k in range(-6, 7):
            hv = base + k * 5
            xx = cx + (k * 5 - (base % 5)) * 14
            tick = 16 if int(round(hv)) % 10 == 0 else 8
            d.line([(xx, 150), (xx, 150 + tick)], fill=white, width=2)
            if int(round(hv)) % 10 == 0:
                d.text((xx - 20, 118), f"{int(round(hv)) % 360:03d}", font=fs, fill=white)
        d.polygon([(cx, 176), (cx - 9, 192), (cx + 9, 192)], fill=white)
        # text blocks
        d.text((160, 220), label_mode, font=fm, fill=white)
        d.text((160, 262), zoom, font=fm, fill=white)
        d.text((160, 304), polarity, font=fm, fill=white)
        d.text((160, 346), f"{1.0 + 0.5 * math.floor(t * 0.2) % 2:.1f}", font=fm, fill=white)
        d.text((W - 330, 220), "TRK AUTO", font=fm, fill=white)
        d.text((W - 330, 262), f"LST  {int(t * 10) % 3 + 1}", font=fm, fill=white)
        if az is not None:
            d.text((cx - 150, H - 210), f"{abs(az):02.0f}{'L' if az < 0 else 'R'}", font=fm, fill=white)
        if el is not None:
            d.text((cx + 80, H - 210), f"{el:+03.0f}", font=fm, fill=white)
        if rng_nm is not None:
            d.text((160, H - 250), f"RNG {rng_nm:4.1f}", font=fm, fill=white)
        d.text((W - 330, H - 290), f"{alt:05d} ASL", font=fm, fill=white)
        d.text((W - 330, H - 250), f"M {mach:.2f}", font=fm, fill=white)
        secs = int(t) + 12
        d.text((W - 330, H - 330), f"00:{secs // 60:02d}:{secs % 60:02d}", font=fs, fill=white)
        for k, s in enumerate(extra):
            d.text((160, 400 + k * 42), s, font=fm, fill=white)
        # left elevation scale
        for k in range(9):
            yy = cy - 200 + k * 50
            d.line([(92, yy), (92 + (18 if k % 2 == 0 else 10), yy)], fill=white, width=2)
        a = np.asarray(im, np.float32) / 255.0 * opacity
        sh = cv2.GaussianBlur(a, (0, 0), 2.0)
        img *= (1 - sh * 0.55)[..., None]
        img += (a * 0.95)[..., None] * (1 - img * 0.5)
        return img


# ============================================================ skies & space

def milky_way(w, h, seed=5, angle=-28, stars=26000, band_width=0.12):
    xs, ys = grid(w, h)
    a = math.radians(angle)
    nx, ny = -math.sin(a), math.cos(a)
    d = ((xs - w / 2) * nx + (ys - h / 2) * ny) / h
    along = ((xs - w / 2) * math.cos(a) + (ys - h / 2) * math.sin(a)) / w
    n1 = fbm(w, h, 420, 6, 0.55, seed)
    n2 = fbm(w, h, 90, 6, 0.6, seed + 1)
    n3 = fbm(w, h, 30, 4, 0.6, seed + 2)
    width = band_width * (1 + 0.5 * (n1 - 0.5))
    band = np.exp(-(d / width) ** 2)
    core = np.exp(-((along + 0.08) / 0.22) ** 2)
    dust = np.exp(-((d + 0.01 * (n1 - 0.5)) / (width * 0.32)) ** 2) * np.clip((n2 - 0.38) * 2.6, 0, 1)
    lum = band * (0.25 + 0.75 * n1 * (0.7 + 0.3 * n3)) * (0.45 + 0.9 * core)
    lum *= 1 - 0.9 * dust
    lum = lum.astype(np.float32)
    warm = np.array([1.0, 0.80, 0.62], np.float32)
    cool = np.array([0.55, 0.66, 1.0], np.float32)
    mixk = np.clip(core * 1.2 * band, 0, 1)[..., None]
    col = (warm * mixk + cool * (1 - mixk)) * lum[..., None] * 0.42
    neb = np.clip((fbm(w, h, 160, 5, 0.55, seed + 3) - 0.62) * 3, 0, 1) * band
    col += neb[..., None] * np.array([0.35, 0.08, 0.14], np.float32) * 0.5
    dens = (0.25 + band * 3.0 * (1 - 0.7 * dust)).astype(np.float32)
    col += star_layer(w, h, stars, seed + 9, density=dens, mag_pow=3.2, bright=1.0)
    col += star_layer(w, h, stars * 3, seed + 10, density=dens, mag_pow=1.5, bright=0.18, glow=False)
    return col.astype(np.float32)


def night_sky(w, h, top=(0.01, 0.015, 0.04), horizon=(0.05, 0.07, 0.12), stars=3000, seed=1, star_bright=0.8):
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    img = np.array(top, np.float32) * (1 - t) + np.array(horizon, np.float32) * t
    img = np.broadcast_to(img, (h, w, 3)).copy()
    if stars:
        img += star_layer(w, h, stars, seed, bright=star_bright)
    return img


def cloud_layer(w, h, scale=300, seed=0, cover=0.5, sharp=3.0):
    n = fbm(w, h, scale, 6, 0.55, seed)
    return np.clip((n - (1 - cover)) * sharp, 0, 1).astype(np.float32)


# ============================================================ papers

def paper(w, h, base=(0.90, 0.86, 0.76), seed=2, stains=0.35, fibers=0.05):
    n1 = fbm(w, h, 400, 5, 0.5, seed)
    n2 = fbm(w, h, 6, 2, 0.5, seed + 1)
    fib = fbm(w, h, 40, 3, 0.5, seed + 2)
    img = np.empty((h, w, 3), np.float32)
    img[:] = np.array(base, np.float32)
    img *= (1 - stains * 0.25 * (n1 - 0.5) * 2)[..., None]
    img *= (1 - fibers * (n2 - 0.5) * 2)[..., None]
    img *= (1 - 0.04 * (fib - 0.5))[..., None]
    xs, ys = grid(w, h)
    u = (xs - w / 2) / (w / 2)
    v = (ys - h / 2) / (h / 2)
    edge = np.clip((u * u) ** 3 + (v * v) ** 3, 0, 1)
    img *= (1 - 0.25 * edge)[..., None]
    img[..., 2] *= 1 - 0.08 * edge
    return img


def hatch(h, w, period=6.0, angle=0.0):
    xs, ys = grid(w, h)
    a = math.radians(angle)
    p = (xs * math.sin(a) + ys * math.cos(a)) * (2 * math.pi / period)
    return (0.5 + 0.5 * np.sin(p)).astype(np.float32)


def ink_hatch(tone, pattern, soft=0.12):
    """tone 0..1 (1 = full ink); pattern from hatch(); returns ink coverage."""
    return np.clip((tone - pattern) / soft + 0.5, 0, 1).astype(np.float32)


# ============================================================ radar PPI

class RadarScope:
    def __init__(self, size=900, seed=4, color=(0.25, 1.0, 0.45)):
        self.size = size
        self.color = np.array(color, np.float32)
        s = size
        ys, xs = np.mgrid[0:s, 0:s].astype(np.float32)
        self.dx, self.dy = xs - s / 2, ys - s / 2
        self.r = np.sqrt(self.dx ** 2 + self.dy ** 2) / (s / 2)
        self.th = (np.arctan2(self.dx, -self.dy)) % (2 * math.pi)
        rng = np.random.default_rng(seed)
        clut = fbm(s, s, 22, 4, 0.6, seed)
        self.clutter = (np.clip((clut - 0.55) * 4, 0, 1) * np.exp(-(self.r / 0.22) ** 2)).astype(np.float32)
        grain_ = fbm(s, s, 3, 2, 0.5, seed + 1)
        self.speckle = np.clip((grain_ - 0.72) * 5, 0, 1).astype(np.float32) * (self.r < 1)
        self.disc = np.clip((1 - self.r) * s / 4, 0, 1).astype(np.float32)
        im = Image.new("L", (s, s), 0)
        d = ImageDraw.Draw(im)
        c = s / 2
        for k in range(1, 6):
            rr = c * k / 5 * 0.98
            d.ellipse([c - rr, c - rr, c + rr, c + rr], outline=110, width=2)
        for k in range(0, 360, 30):
            a = math.radians(k)
            d.line([(c, c), (c + math.sin(a) * c * 0.98, c - math.cos(a) * c * 0.98)], fill=55, width=1)
        for k in range(0, 360, 5):
            a = math.radians(k)
            r0 = c * (0.94 if k % 30 else 0.9)
            d.line([(c + math.sin(a) * r0, c - math.cos(a) * r0),
                    (c + math.sin(a) * c * 0.98, c - math.cos(a) * c * 0.98)], fill=140, width=2)
        self.graticule = np.asarray(im, np.float32) / 255.0

    def frame(self, t, blips, period=4.0, persistence=0.85):
        """blips: list of (r 0..1, theta rad, strength). returns (s,s,3) float."""
        sweep = (t / period * 2 * math.pi) % (2 * math.pi)
        age = ((sweep - self.th) % (2 * math.pi)) / (2 * math.pi)
        decay = np.exp(-age * 6.0 / persistence)
        beam = np.exp(-(age * 2 * math.pi / 0.02) ** 2) * 0.9
        echo = self.clutter * 0.9 + self.speckle * 0.25
        s = self.size
        bl = np.zeros((s, s), np.float32)
        for (r, th, st) in blips:
            x = s / 2 + math.sin(th) * r * s / 2
            y = s / 2 - math.cos(th) * r * s / 2
            x0, x1 = int(max(0, x - 14)), int(min(s, x + 15))
            y0, y1 = int(max(0, y - 14)), int(min(s, y + 15))
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            bl[y0:y1, x0:x1] += st * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 30.0)
        wedge = np.exp(-age * 9.0) * 0.10
        lum = (echo + bl * 2.4) * decay + beam * (0.35 + echo) + wedge
        lum = lum * self.disc + self.graticule * 0.75 + 0.05 * self.disc
        out = lum[..., None] * self.color
        return out.astype(np.float32)


# ============================================================ newspaper page

def newspaper(masthead, date_line, headline_lines, sub_lines, seed=0, w=1400, h=1900, head_font="gloock",
              head_size=118, photo=None, masthead_font="gloock"):
    """Generate a vintage newspaper front page as float RGB."""
    rng = np.random.default_rng(seed)
    im = Image.new("L", (w, h), 236)
    d = ImageDraw.Draw(im)
    mf = font(masthead_font, 120)
    tw = d.textlength(masthead, font=mf)
    d.text(((w - tw) / 2, 40), masthead, font=mf, fill=20)
    d.line([(50, 190), (w - 50, 190)], fill=20, width=4)
    fd = font("baskerville", 26)
    d.text((60, 200), date_line, font=fd, fill=30)
    d.text((w - 60 - d.textlength("PRICE FIVE CENTS", font=fd), 200), "PRICE FIVE CENTS", font=fd, fill=30)
    d.line([(50, 242), (w - 50, 242)], fill=20, width=2)
    y = 262
    hf = font(head_font, head_size)
    for line in headline_lines:
        tw = d.textlength(line, font=hf)
        sc = min(1.0, (w - 110) / tw)
        f2 = font(head_font, int(head_size * sc))
        tw = d.textlength(line, font=f2)
        d.text(((w - tw) / 2, y), line, font=f2, fill=12)
        y += int(head_size * sc * 1.08)
    y += 14
    sf = font("baskerville", 38)
    for line in sub_lines:
        tw = d.textlength(line, font=sf)
        d.text(((w - tw) / 2, y), line, font=sf, fill=25)
        y += 52
    y += 16
    d.line([(50, y), (w - 50, y)], fill=20, width=2)
    y += 20
    cols = 6
    cw = (w - 100) / cols
    photo_box = None
    if photo is not None:
        photo_box = (50 + cw * 2, y + 10, 50 + cw * 4 - 12, y + 520)
    body = font("baskerville", 17)
    words = "the object was seen by several residents who reported that it moved at great speed across the sky and " \
            "made no sound officials declined to comment further on the matter while witnesses described a bright " \
            "light that appeared to hover before departing to the northwest late in the evening".split()
    for c in range(cols):
        x0 = 50 + c * cw + 8
        yy = y
        if c in (2, 3) and photo_box:
            yy = photo_box[3] + 20
        sub_head_done = False
        while yy < h - 60:
            if not sub_head_done and rng.random() < 0.5:
                d.text((x0, yy), " ".join(rng.choice(words, 3)).upper(), font=font("baskerville", 20), fill=20)
                yy += 34
                sub_head_done = True
            line = ""
            while d.textlength(line, font=body) < cw - 30:
                line += rng.choice(words) + " "
            d.text((x0, yy), line.strip(), font=body, fill=40)
            yy += 23
            if rng.random() < 0.04:
                yy += 20
                sub_head_done = False
        if c < cols - 1:
            d.line([(50 + (c + 1) * cw, y), (50 + (c + 1) * cw, h - 60)], fill=90, width=1)
    arr = np.asarray(im, np.float32) / 255.0
    if photo is not None:
        x0, y0, x1, y1 = [int(v) for v in photo_box]
        ph = cv2.resize(photo, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
        arr[y0:y1, x0:x1] = ph
    # halftone / print texture
    n = fbm(w, h, 200, 4, 0.5, seed + 5)
    fine = fbm(w, h, 3, 2, 0.5, seed + 6)
    arr = arr * (0.92 + 0.08 * n) * (0.96 + 0.06 * fine)
    col = np.stack([arr * 0.97, arr * 0.93, arr * 0.82], -1)
    return col.astype(np.float32)


def rotate_paste(img, tex, cx, cy, scale, angle, opacity=1.0, shadow=0.5):
    """Place texture `tex` onto img centred at (cx,cy) with scale/rotation, plus soft shadow."""
    th, tw = tex.shape[:2]
    M = cv2.getRotationMatrix2D((tw / 2, th / 2), angle, scale)
    M[0, 2] += cx - tw / 2
    M[1, 2] += cy - th / 2
    out = cv2.warpAffine(tex, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    m = cv2.warpAffine(np.ones((th, tw), np.float32), M, (W, H), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if shadow > 0:
        M2 = M.copy()
        M2[0, 2] += 18 * scale
        M2[1, 2] += 26 * scale
        sm = cv2.warpAffine(np.ones((th, tw), np.float32), M2, (W // 4, H // 4), flags=cv2.INTER_LINEAR) \
            if False else None
        sm = cv2.warpAffine(np.ones((th, tw), np.float32), M2, (W, H), flags=cv2.INTER_LINEAR)
        sm = cv2.GaussianBlur(cv2.resize(sm, (W // 4, H // 4)), (0, 0), 6)
        sm = cv2.resize(sm, (W, H))
        img *= (1 - shadow * sm * opacity)[..., None]
    m = (m * opacity)[..., None]
    img *= 1 - m
    img += out * m
    return img


def city_lights(w, h, horizon, seed=0, n=9000, spread=1.0, color=(1.0, 0.75, 0.42), grid_like=False):
    """Perspective field of city lights below a horizon line."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.float32)
    z = rng.random(n) ** 1.6
    depth = 1 / (0.05 + z)
    yy = horizon + (h - horizon) * (1 - np.exp(-z * 3.2)) * 1.05
    xx = w / 2 + (rng.random(n) - 0.5) * w * spread * (0.4 + 2.4 * z)
    if grid_like:
        gx = np.round(xx / (6 + 40 * z)) * (6 + 40 * z)
        xx = np.where(rng.random(n) < 0.7, gx, xx)
    b = (0.25 + rng.random(n) ** 3 * 1.4) * (0.35 + 0.65 * z)
    c = np.array(color, np.float32)[None, :] * (0.8 + 0.4 * rng.random((n, 1))).astype(np.float32)
    c[:, 2] *= (0.7 + 0.8 * rng.random(n)).astype(np.float32)
    splat_points(img, xx.astype(np.float32), yy.astype(np.float32), (c * b[:, None]).astype(np.float32))
    glow = cv2.GaussianBlur(img, (0, 0), 3) * 2.0 + cv2.GaussianBlur(img, (0, 0), 14) * 3.0
    return img + glow
