"""Act II–III: 1896 airships to the 1977 Wow! signal."""
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw

from core import (H, W, FPS, add_ellipse_glow, add_glow, blank, bloom, caption, chapter_card, chapter_tag, draw_text,
                  ease_in, ease_io, ease_out, fbm, glitch, grid, hash01, lerp, noise1d, ramp, recreation_tag,
                  ridge_mask, smooth, star_layer, subtitle, text_size, warp, window, clamp01, shake_offsets,
                  darken_region, desaturate, font, splat_points, contrast, flicker, scanlines)
from elements import (RadarScope, hatch, ink_hatch, milky_way, newspaper, paper, rotate_paste, night_sky)
from scene import Scene


def desk_bg(seed=0, lamp=(760, 420)):
    wood = cv2.resize(fbm(480, 1400, 40, 5, 0.55, seed), (W * 2, H), interpolation=cv2.INTER_CUBIC)[:, :W]
    grain_ = cv2.resize(fbm(1920, 90, 6, 3, 0.6, seed + 1), (W, H), interpolation=cv2.INTER_LINEAR)
    v = 0.6 * wood + 0.4 * grain_
    col = np.stack([0.20 + 0.12 * v, 0.11 + 0.07 * v, 0.06 + 0.03 * v], -1).astype(np.float32)
    xs, ys = grid()
    d2 = ((xs - lamp[0]) ** 2 + (ys - lamp[1]) ** 2) / (900.0 ** 2)
    col *= (0.25 + 1.0 * np.exp(-d2 * 1.6))[..., None]
    return col


def film_damage(img, t, f, amount=1.0, seed=0):
    rng = np.random.default_rng(f * 7 + seed)
    for _ in range(rng.integers(0, 3)):
        x = int(rng.uniform(0, W))
        wdt = int(rng.integers(1, 3))
        img[:, x:x + wdt] *= 1 - 0.35 * amount * rng.random()
    if rng.random() < 0.25 * amount:
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        cv2.circle(img, (int(x), int(y)), int(rng.uniform(2, 6)), (0.9, 0.9, 0.9) if rng.random() < 0.5 else
                   (0.0, 0.0, 0.0), -1, cv2.LINE_AA)
    return img


def spin_in(t, t0, dur=1.1, turns=2.0, s0=0.03, s1=0.5, a1=-4):
    u = clamp01((t - t0) / dur)
    e = ease_out(u, 3)
    return lerp(s0, s1, e), lerp(a1 + 360 * turns, a1, e), u


# ----------------------------------------------------------------------------- S08

def airship_engraving(w=620, h=470, seed=81):
    im = np.full((h, w), 0.85, np.float32)
    ys = np.arange(h, dtype=np.float32)[:, None]
    hh = hatch(h, w, 5, 0)
    tone = 0.25 + 0.5 * (1 - ys / h)
    im -= 0.7 * ink_hatch(tone, hh, 0.2)
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (w // 2, 150), (170, 42), -6, 0, 360, 255, -1, cv2.LINE_AA)
    cv2.rectangle(m, (w // 2 - 40, 190), (w // 2 + 40, 214), 255, -1)
    for dx in (-30, 30):
        cv2.line(m, (w // 2 + dx, 184), (w // 2 + dx, 192), 255, 2)
    body = m.astype(np.float32) / 255
    im = im * (1 - body) + body * 0.12
    beam = np.zeros((h, w), np.uint8)
    cv2.fillPoly(beam, [np.array([[w // 2 + 10, 212], [w // 2 + 190, h], [w // 2 - 40, h]], np.int32)], 255,
                 cv2.LINE_AA)
    bm = cv2.GaussianBlur(beam, (0, 0), 8).astype(np.float32) / 255
    im = im + bm * 0.5
    roofs = np.zeros((h, w), np.uint8)
    x = 0
    rng = np.random.default_rng(seed)
    while x < w:
        bw = int(rng.uniform(40, 90))
        bh = int(rng.uniform(40, 110))
        cv2.fillPoly(roofs, [np.array([[x, h], [x, h - bh], [x + bw // 2, h - bh - 30], [x + bw, h - bh], [x + bw, h]],
                                      np.int32)], 255)
        x += bw
    r = roofs.astype(np.float32) / 255
    im = im * (1 - r) + r * 0.08
    return np.clip(im, 0, 1)


class Airship(Scene):
    dur = 10.0
    grain = 0.04
    vig = 0.6

    def setup(self):
        self.desk = desk_bg(3)
        self.paper1 = newspaper("The Evening Courier", "SATURDAY, NOVEMBER 21, 1896",
                                ["THE MYSTERIOUS", "AIRSHIP IS SEEN"],
                                ["Strange Light Crosses the Night Sky,", "Hundreds of Citizens Watch in Wonder"],
                                seed=8, photo=airship_engraving())
        self.paper2 = newspaper("The Daily Tribune", "SATURDAY, APRIL 10, 1897",
                                ["THOUSANDS SEE", "LIGHTS IN THE SKY"],
                                ["Mysterious Visitor Again Reported", "Over the City at Midnight"], seed=9)

    def render(self, t, f):
        img = self.desk.copy()
        s, a, u = spin_in(t, 2.2, 1.0, 2.0, 0.03, 0.50, -5)
        if t > 2.2:
            k = (t - 3.2) / 6.8
            s2 = s * (1 + 0.08 * max(0, k))
            img = rotate_paste(img, self.paper1, 820 + 30 * max(0, k), 560, s2, a, 1.0, 0.6)
        s, a, u = spin_in(t, 5.6, 0.9, 1.5, 0.03, 0.44, 7)
        if t > 5.6:
            img = rotate_paste(img, self.paper2, 1180, 600, s, a, 1.0, 0.6)
        img = desaturate(img, 0.35)
        return img

    def overlay(self, img, t, f):
        chapter_card(img, t, 0.0, 2.3, "CHAPTER  II", "飞 碟 时 代", "THE  SAUCER  AGE")
        recreation_tag(img, window(t, 2.4, 9.8))
        caption(img, t, 3.2, 9.8, "1896—1897  ·  美国",
                "“神秘飞艇”目击潮",
                ("从加利福尼亚到芝加哥，大批民众报告夜空中出现带探照灯的“飞艇”——",
                 "而那时，世界上还没有任何能完成这种飞行的飞行器。"),
                bottom=H - 100, title_size=48)
        return img


# ----------------------------------------------------------------------------- S09

class LosAngeles1942(Scene):
    dur = 10.0
    grain = 0.07
    vig = 0.7

    def setup(self):
        rng = np.random.default_rng(91)
        sky = np.zeros((H, W, 3), np.float32)
        ys = np.arange(H, dtype=np.float32)[:, None, None]
        sky += 0.03 + 0.05 * (ys / H)
        haze = fbm(W, H, 300, 5, 0.55, 92)
        sky += (haze * 0.05)[..., None]
        self.sky = sky
        m = np.zeros((H, W), np.uint8)
        x = -20
        while x < W:
            bw = int(rng.uniform(40, 130))
            bh = int(rng.uniform(40, 170))
            if abs(x - 900) < 60:
                bh = 330
                cv2.rectangle(m, (x + 10, 1000 - bh - 80), (x + bw - 10, 1000 - bh), 255, -1)
                cv2.fillPoly(m, [np.array([[x + 10, 1000 - bh - 80], [x + bw // 2, 1000 - bh - 150],
                                           [x + bw - 10, 1000 - bh - 80]], np.int32)], 255, cv2.LINE_AA)
            cv2.rectangle(m, (x, 1000 - bh), (x + bw, H), 255, -1)
            x += bw + int(rng.uniform(0, 8))
        cv2.rectangle(m, (0, 990), (W, H), 255, -1)
        self.city = cv2.GaussianBlur(m, (0, 0), 1.0).astype(np.float32) / 255
        self.beams = [(rng.uniform(80, 1840), rng.uniform(0.0, 6.0), rng.uniform(0.4, 1.0)) for _ in range(10)]
        self.target = (1010, 300)
        self.bursts = [(rng.uniform(0.2, 9.8), rng.normal(0, 150), rng.normal(0, 90), rng.uniform(0.5, 1.2))
                       for _ in range(60)]
        self.puff_tex = fbm(W // 4, H // 4, 10, 3, 0.6, 93)
        self.q = (W // 4, H // 4)
        qx, qy = np.meshgrid(np.arange(self.q[0], dtype=np.float32), np.arange(self.q[1], dtype=np.float32))
        self.qx, self.qy = qx * 4, qy * 4

    def render(self, t, f):
        img = self.sky.copy()
        tx = self.target[0] + 30 * math.sin(t * 0.3)
        ty = self.target[1] + 12 * math.sin(t * 0.21)
        beam = np.zeros(self.q[::-1], np.float32)
        for (bx, ph, str_) in self.beams:
            by = 1000.0
            sway = 60 * math.sin(t * 0.35 + ph)
            ax, ay = tx + sway - bx, ty - by
            L = math.hypot(ax, ay)
            ax, ay = ax / L, ay / L
            px, py = self.qx - bx, self.qy - by
            s = px * ax + py * ay
            d = np.abs(px * ay - py * ax)
            wdt = 6 + s * 0.035
            val = np.exp(-(d / wdt) ** 2) * (s > 0) * np.exp(-s / 2400) * str_
            beam += val
        conv = np.exp(-((self.qx - tx) ** 2 + (self.qy - ty) ** 2) / (70 ** 2))
        beam += conv * 0.6
        # flak smoke puffs darken the lit haze
        smoke = np.zeros_like(beam)
        flash = np.zeros_like(img)
        for (bt, dx, dy, sz) in self.bursts:
            age = t - bt
            if age < 0 or age > 6:
                continue
            x, y = tx + dx, ty + dy
            r = (20 + 45 * math.sqrt(age)) * sz
            g = np.exp(-((self.qx - x) ** 2 + (self.qy - y) ** 2) / (r * r))
            smoke += g * math.exp(-age / 3.5) * 0.9
            if age < 0.12:
                add_glow(flash, x, y, 10 * sz, (1, 1, 1), 5.0 * (1 - age / 0.12))
                add_glow(flash, x, y, 60 * sz, (1, 0.95, 0.9), 0.6 * (1 - age / 0.12), kind="halo")
        smoke *= 0.6 + 0.8 * self.puff_tex
        beam = beam * (1 - np.clip(smoke, 0, 0.85)) + np.clip(smoke, 0, 1) * 0.05
        beam_full = cv2.resize(beam, (W, H), interpolation=cv2.INTER_LINEAR)
        img += beam_full[..., None] * 0.55
        add_ellipse_glow(img, tx, ty, 26, 12, 0.1, (1, 1, 1), 0.9)
        img += flash
        # tracers
        rng = np.random.default_rng(int(t * 6))
        for k in range(5):
            x0 = rng.uniform(200, 1700)
            u = (t * 6) % 1
            x1, y1 = lerp(x0, tx + rng.normal(0, 200), u), lerp(1000, ty + rng.normal(0, 80), u)
            x2, y2 = lerp(x0, tx, max(0, u - 0.08)), lerp(1000, ty, max(0, u - 0.08))
            cv2.line(img, (int(x2), int(y2)), (int(x1), int(y1)), (0.9, 0.9, 0.9), 2, cv2.LINE_AA)
        c = self.city[..., None]
        img = img * (1 - c) + c * 0.01
        bloom(img, 0.6, 0.6, 40)
        img = desaturate(img, 1.0)
        img = contrast(img, 1.15, 0.2)
        img = flicker(img, t, 0.06, 9)
        img = film_damage(img, t, f)
        dx, dy = shake_offsets(t, 2.5, 3.0)
        return warp(img, 1.04 + 0.03 * t / 10, 0, tx=dx, ty=dy)

    def overlay(self, img, t, f):
        recreation_tag(img, window(t, 0.2, 9.8))
        chapter_tag(img, t, 0.2, 9.8, "CHAPTER II  ·  THE SAUCER AGE")
        caption(img, t, 0.8, 9.8, "1942 年 2 月 25 日  ·  美国 洛杉矶",
                "“洛杉矶之战”",
                ("珍珠港事件后不到三个月，防空警报在凌晨响彻全城。",
                 "高射炮向夜空射出约 1400 发炮弹——它们究竟在向什么开火，至今成谜。"),
                bottom=H - 100, title_size=52)
        return img


# ----------------------------------------------------------------------------- S10

class Arnold1947(Scene):
    dur = 10.0
    grain = 0.05
    vig = 0.55

    def setup(self):
        ys = np.arange(H, dtype=np.float32)[:, None, None]
        sky = np.array([0.30, 0.42, 0.62], np.float32) * (1 - ys / H) + np.array([0.75, 0.80, 0.85], np.float32) * (ys / H)
        sky = np.broadcast_to(sky, (H, W, 3)).copy()
        xs, yy = grid()
        # the big volcano (Rainier) on the right
        cx, top = 1320, 330
        prof = np.abs(xs - cx) / 820
        n = noise1d(W, 90, 5, 101)[None, :]
        height = top + 520 * prof ** 0.85 + 30 * (n - 0.5)
        mtn = np.clip(yy - height, 0, 1)
        ridges = fbm(W, H, 40, 5, 0.6, 102)
        streaks = cv2.resize(fbm(W // 6, H, 30, 4, 0.6, 103), (W, H))
        snow = 0.80 + 0.12 * ridges + 0.08 * streaks
        rock = ((ridges * 0.5 + streaks * 0.5) > 0.66) & (yy > height + 90)
        shade = np.clip((xs - cx) / 600, -1, 1)
        col = snow[..., None] * np.array([0.95, 0.96, 1.0], np.float32)
        col = np.where(rock[..., None], np.array([0.25, 0.26, 0.30], np.float32) + 0.1 * ridges[..., None], col)
        col *= (1 - 0.35 * np.clip(shade, 0, 1))[..., None]
        sky = sky * (1 - mtn[..., None]) + col * mtn[..., None]
        # lower range
        m2, _ = ridge_mask(W, H, 840, 170, 260, seed=104, octaves=6, sharp=True)
        rc = np.array([0.20, 0.23, 0.28], np.float32) * (0.7 + 0.5 * ridges[..., None])
        sky = sky * (1 - m2[..., None]) + rc * m2[..., None]
        # cloud deck
        cl = np.clip((fbm(W, H, 160, 5, 0.55, 105) - 0.45) * 3, 0, 1) * np.clip((yy - 800) / 120, 0, 1)
        sky = sky * (1 - cl[..., None]) + cl[..., None] * np.array([0.92, 0.93, 0.95], np.float32)
        self.bg = sky.astype(np.float32)
        # cockpit frame
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [np.array([[0, 0], [W, 0], [W, 60], [0, 140]], np.int32)], 255, cv2.LINE_AA)
        cv2.fillPoly(m, [np.array([[0, 0], [120, 0], [260, H], [0, H]], np.int32)], 255, cv2.LINE_AA)
        cv2.line(m, (1650, 60), (1760, H), 255, 26, cv2.LINE_AA)
        cv2.fillPoly(m, [np.array([[0, H - 70], [W, H - 120], [W, H], [0, H]], np.int32)], 255, cv2.LINE_AA)
        self.frame = cv2.GaussianBlur(m, (0, 0), 3).astype(np.float32) / 255

    def objects(self, t):
        out = []
        for i in range(9):
            u = (t - 1.2) / 7.5 - i * 0.035
            x = lerp(-100, 2100, u)
            y = 430 + i * 12 + 14 * math.sin(u * 30 + i * 0.7) + 50 * u
            glint = max(0.0, math.sin(u * 55 + i * 1.3)) ** 12
            out.append((x, y, glint, math.sin(u * 30 + i * 0.7)))
        return out

    def render(self, t, f):
        img = self.bg.copy()
        for (x, y, glint, tilt) in self.objects(t):
            if -60 < x < W + 60:
                m = np.zeros((80, 120), np.uint8)
                pts = np.array([[8, 40], [60, 28 + 8 * tilt], [112, 40], [60, 48 - 4 * tilt]], np.int32)
                cv2.fillPoly(m, [pts], 255, cv2.LINE_AA)
                cv2.ellipse(m, (60, 40), (36, int(8 + 4 * abs(tilt))), 0, 180, 360, 255, -1, cv2.LINE_AA)
                mf = cv2.resize(m, (60, 40)).astype(np.float32) / 255
                x0, y0 = int(x - 30), int(y - 20)
                if 0 <= x0 < W - 60 and 0 <= y0 < H - 40:
                    reg = img[y0:y0 + 40, x0:x0 + 60]
                    reg *= 1 - mf[..., None] * 0.85
                    reg += mf[..., None] * 0.15
                add_glow(img, x, y, 6, (1, 1, 1), 3.0 * glint)
                add_glow(img, x, y, 30, (1, 0.97, 0.9), 0.8 * glint, kind="halo")
        fr = self.frame[..., None]
        img = img * (1 - fr) + fr * np.array([0.03, 0.03, 0.035], np.float32)
        bloom(img, 0.9, 0.4, 30)
        img = desaturate(img, 0.75, tint=None)
        img = img * np.array([1.04, 1.0, 0.92], np.float32)
        dx, dy = shake_offsets(t, 4, 2.2, 3)
        return warp(img, 1.03, 0.3 * math.sin(t * 0.7), tx=dx, ty=dy)

    def overlay(self, img, t, f):
        recreation_tag(img, window(t, 0.2, 9.8), y=100)
        chapter_tag(img, t, 0.2, 9.8, "CHAPTER II  ·  THE SAUCER AGE", y=100, x=300)
        caption(img, t, 1.0, 9.8, "1947 年 6 月 24 日  ·  美国华盛顿州 雷尼尔山",
                "“它们像碟子掠过水面一样飞行”",
                ("私人飞行员肯尼斯·阿诺德目击九个闪亮物体以惊人速度掠过群山。",
                 "报纸称之为“飞碟”（flying saucer）——这个词从此传遍全世界。"),
                "—— 肯尼斯·阿诺德", x=300, bottom=H - 140, title_size=50)
        return img


# ----------------------------------------------------------------------------- S11

class Roswell(Scene):
    dur = 10.0
    grain = 0.04
    vig = 0.6

    def setup(self):
        self.desk = desk_bg(5, lamp=(900, 450))
        self.paper = newspaper("Roswell Daily Record", "ROSWELL, NEW MEXICO,  TUESDAY, JULY 8, 1947",
                               ["RAAF Captures Flying Saucer", "On Ranch in Roswell Region"],
                               ["No Details of Flying Disk Are Revealed"], seed=11, head_size=104,
                               masthead_font="gloock")

    def stamp(self, img, t, t0, text, x, y, angle, size=54):
        if t < t0:
            return img
        u = clamp01((t - t0) / 0.14)
        s = lerp(1.8, 1.0, ease_in(u, 2))
        a = ramp(t, t0, t0 + 0.05)
        tw = text_size(text, "sans_black", size, 0.05)
        pad = 26
        wd, hd = int(tw + pad * 2), int(size * 1.7)
        patch = np.zeros((hd + 20, wd + 20, 3), np.float32)
        pa = np.zeros((hd + 20, wd + 20), np.float32)
        cv2.rectangle(pa, (10, 10), (10 + wd, 10 + hd), 1.0, 6)
        rgb_im = Image.new("L", (wd + 20, hd + 20), 0)
        d = ImageDraw.Draw(rgb_im)
        d.text((10 + pad, 10 + hd * 0.18), text, font=font("sans_black", size), fill=255)
        pa = np.maximum(pa, np.asarray(rgb_im, np.float32) / 255)
        tex = fbm(wd + 20, hd + 20, 8, 3, 0.6, int(t0 * 10))
        pa *= np.clip((tex - 0.25) * 3, 0, 1)
        patch[:] = np.array([0.72, 0.08, 0.07], np.float32)
        M = cv2.getRotationMatrix2D(((wd + 20) / 2, (hd + 20) / 2), angle, s)
        M[0, 2] += x - (wd + 20) / 2
        M[1, 2] += y - (hd + 20) / 2
        pw = cv2.warpAffine(pa, M, (W, H))
        m = (pw * 0.9 * a)[..., None]
        img *= 1 - m
        img += m * patch[0, 0]
        return img

    def render(self, t, f):
        img = self.desk.copy()
        s, a, u = spin_in(t, 0.0, 1.0, 2.0, 0.03, 0.56, -3)
        k = max(0.0, t - 1.0) / 9
        img = rotate_paste(img, self.paper, 960, 600 + 60 * k, s * (1 + 0.25 * ease_io(k)), a, 1.0, 0.6)
        img = self.stamp(img, t, 4.6, "次日：“只是气象气球”", 1150, 470, 6, 70)
        img = self.stamp(img, t, 6.9, "1994：“莫古尔计划”高空气球", 900, 690, -5, 60)
        if 4.6 < t < 4.75 or 6.9 < t < 7.05:
            dx, dy = shake_offsets(t, 8, 30)
            img = warp(img, 1.0, 0, tx=dx, ty=dy)
        img = desaturate(img, 0.25)
        return img

    def overlay(self, img, t, f):
        recreation_tag(img, window(t, 0.2, 9.8))
        caption(img, t, 1.2, 9.8, "1947 年 7 月 8 日  ·  美国新墨西哥州 罗斯威尔",
                "军方宣布：“捕获飞碟”",
                ("罗斯威尔陆军机场发布新闻稿，称在牧场回收了一个“飞行圆盘”。",
                 "第二天，说法变成了气象气球——这场争议延续了近八十年。"),
                bottom=H - 90, title_size=52)
        return img


# ----------------------------------------------------------------------------- S12

DOC_LINES = [
    ("DEPARTMENT OF THE AIR FORCE", 0),
    ("PROJECT BLUE BOOK", 0),
    ("REPORT ON UNIDENTIFIED FLYING OBJECT(S)", 0),
    ("", 0),
    ("1. DATE-TIME GROUP:  ## JULY 1952   2140 LOCAL", 1),
    ("2. LOCATION:  ####################", 1),
    ("3. SOURCE:  MILITARY PILOT / GROUND RADAR", 1),
    ("4. DESCRIPTION:  ROUND, BRIGHT WHITE LIGHT, NO SOUND", 1),
    ("5. COURSE:  HOVERED, THEN ACCELERATED RAPIDLY TO NW", 1),
    ("6. DURATION:  APPROX 12 MINUTES", 1),
    ("7. WEATHER:  CLEAR, UNLIMITED VISIBILITY", 1),
    ("8. INTERCEPTION:  #################", 1),
    ("", 0),
    ("9. CONCLUSION:", 1),
]


class BlueBook(Scene):
    dur = 12.5
    grain = 0.04
    vig = 0.55

    def setup(self):
        self.desk = desk_bg(7, lamp=(900, 380))
        pw, ph = 1300, 1700
        base = paper(pw, ph, base=(0.90, 0.88, 0.80), seed=121, stains=0.5)
        self.pw, self.ph = pw, ph
        self.paper = base

    def doc(self, t):
        im = Image.new("L", (self.pw, self.ph), 0)
        d = ImageDraw.Draw(im)
        fnt = font("plex_bold", 34)
        y = 130
        chars = int(max(0, t - 0.3) * 58)
        n = 0
        redact = []
        for line, kind in DOC_LINES:
            if kind == 0 and line:
                tw = d.textlength(line, font=font("plex_bold", 38))
                show = line[:max(0, chars - n)]
                d.text(((self.pw - tw) / 2, y), show, font=font("plex_bold", 38), fill=255)
            else:
                show = line[:max(0, chars - n)]
                if "#" in show:
                    i0 = line.index("#")
                    x0 = 110 + d.textlength(line[:i0], font=fnt)
                    x1 = 110 + d.textlength(line[:len(line.rstrip("#"))] if False else line, font=fnt)
                    if len(show) > i0:
                        redact.append((x0, y, x0 + d.textlength(show[i0:], font=fnt), y + 40))
                    show = show.replace("#", " ")
                d.text((110, y), show, font=fnt, fill=255)
            n += len(line)
            y += 64 if line else 40
        ink = np.asarray(im, np.float32) / 255
        return ink, redact, chars >= n, y

    def render(self, t, f):
        ink, redact, done, yend = self.doc(t)
        pap = self.paper.copy()
        pap *= 1 - 0.85 * ink[..., None]
        for (x0, y0, x1, y1) in redact:
            pap[int(y0):int(y1), int(x0):int(x1)] = 0.04
        # stamp
        if t > 5.4:
            u = clamp01((t - 5.4) / 0.12)
            s = lerp(1.6, 1.0, ease_in(u, 2))
            st = Image.new("L", (760, 170), 0)
            d = ImageDraw.Draw(st)
            d.rectangle([6, 6, 754, 164], outline=255, width=10)
            d.text((50, 22), "UNIDENTIFIED", font=font("plex_bold", 104), fill=255)
            sa = np.asarray(st, np.float32) / 255 * np.clip((fbm(760, 170, 6, 3, 0.6, 7) - 0.2) * 3, 0, 1)
            M = cv2.getRotationMatrix2D((380, 85), 9, s)
            M[0, 2] += 560 - 380
            M[1, 2] += yend + 110 - 85
            sw = cv2.warpAffine(sa, M, (self.pw, self.ph))[..., None] * 0.88
            pap = pap * (1 - sw) + sw * np.array([0.70, 0.08, 0.07], np.float32)
        img = self.desk.copy()
        k = t / self.dur
        cx = 900 + 60 * math.sin(k * 2)
        img = rotate_paste(img, pap, cx, 560 + 300 - 520 * ease_io(k), 0.9 - 0.12 * ease_io(k), -2 + 3 * k, 1.0, 0.5)
        if 5.4 < t < 5.6:
            dx, dy = shake_offsets(t, 10, 40)
            img = warp(img, 1.0, 0, tx=dx, ty=dy)
        return img

    def overlay(self, img, t, f):
        recreation_tag(img, window(t, 0.2, 12.3))
        # statistics
        a = window(t, 6.6, 12.4, 0.4, 0.4)
        if a > 0:
            img = darken_region(img, 1180, 250, 1800, 800, 0.75 * a, 180)
            n1 = int(12618 * ease_out((t - 6.7) / 1.8, 3))
            n2 = int(701 * ease_out((t - 8.2) / 1.5, 3))
            draw_text(img, f"{n1:,}", 1240, 280, "shoulders", 170, (255, 255, 255), a, tracking=0.02)
            draw_text(img, "份目击报告   REPORTS", 1250, 470, "sans_bold", 34, (220, 220, 220), a, tracking=0.1)
            a2 = window(t, 8.2, 12.4, 0.3, 0.4)
            draw_text(img, f"{n2:,}", 1240, 530, "shoulders", 170, (235, 50, 45), a2, tracking=0.02)
            draw_text(img, "起至今“无法识别”   UNIDENTIFIED", 1250, 720, "sans_bold", 34, (220, 220, 220), a2,
                      tracking=0.1)
        caption(img, t, 0.8, 12.3, "1952—1969  ·  美国空军“蓝皮书计划”",
                "官方调查，持续了 17 年",
                ("美国空军系统收集并调查全国的不明飞行物报告。",
                 "1969 年计划终止，但留下了数百个无法解释的案例。"),
                bottom=H - 90, title_size=50)
        return img


# ----------------------------------------------------------------------------- S13

class Radar1952(Scene):
    dur = 10.0
    grain = 0.035
    vig = 0.6

    def setup(self):
        self.rs = RadarScope(880, seed=131)
        rng = np.random.default_rng(132)
        self.targets = [(rng.uniform(0.35, 0.8), rng.uniform(4.9, 5.9), rng.uniform(0.6, 1.0),
                         rng.uniform(-0.02, 0.02), rng.uniform(-0.03, 0.03)) for _ in range(7)]
        room = np.zeros((H, W, 3), np.float32)
        xs, ys = grid()
        room += (np.exp(-((xs - 620) ** 2 + (ys - 540) ** 2) / (700 ** 2)) * 0.10)[..., None] * \
            np.array([0.3, 1.0, 0.5], np.float32)
        ring = np.zeros((H, W), np.uint8)
        cv2.circle(ring, (620, 540), 470, 255, 34, cv2.LINE_AA)
        r = cv2.GaussianBlur(ring, (0, 0), 3).astype(np.float32) / 255
        room = room * (1 - r[..., None]) + r[..., None] * 0.06
        self.room = room

    def render(self, t, f):
        blips = []
        for (r, th, st, dr, dth) in self.targets:
            jump = math.floor((t + r * 3) / 2.2)
            rr = r + dr * t + 0.05 * math.sin(jump * 3.1 + th)
            tt = th + dth * t + 0.08 * math.cos(jump * 2.3 + r * 10)
            blips.append((rr, tt, st))
        sc = self.rs.frame(t + 2.0, blips, period=4.0)
        img = self.room.copy()
        s = sc.shape[0]
        img[540 - s // 2:540 - s // 2 + s, 620 - s // 2:620 - s // 2 + s] += sc
        bloom(img, 0.3, 0.9, 30)
        img = scanlines(img, 0.12, 3)
        return img

    def overlay(self, img, t, f):
        lines = ["WASHINGTON NATIONAL AIRPORT", "ARTC CENTER  ·  MEW RADAR", "", "JULY 19-20, 1952",
                 "2340  7 UNIDENTIFIED TARGETS", "      SPEED 100-130 MPH", "0015  TARGETS OVER WHITE HOUSE",
                 "      AND CAPITOL", "0300  F-94 INTERCEPTORS ARRIVE", "      TARGETS DISAPPEAR", "",
                 "JULY 26-27, 1952", "      TARGETS RETURN"]
        chars = int(max(0, t - 0.4) * 50)
        n = 0
        for i, line in enumerate(lines):
            show = line[:max(0, chars - n)]
            n += len(line)
            if show:
                draw_text(img, show, 1180, 170 + i * 44, "plex", 28, (110, 255, 150), 0.9, tracking=0.05)
        recreation_tag(img, window(t, 0.2, 9.8))
        caption(img, t, 1.0, 9.8, "1952 年 7 月  ·  美国 华盛顿",
                "雷达上的不明目标，出现在白宫与国会上空",
                ("华盛顿国家机场雷达连续两个周末捕捉到多个不明目标，",
                 "空军派出战斗机拦截——目标却一次次在战机抵达前消失。"),
                bottom=H - 70, title_size=46, x=110)
        return img


# ----------------------------------------------------------------------------- S14

class Seti(Scene):
    dur = 12.5
    grain = 0.04
    vig = 0.55
    lbox = 2.39

    def setup(self):
        self.sky = milky_way(2600, 1500, seed=141, angle=58, band_width=0.11)
        m = np.zeros((H, W), np.uint8)
        gm, top = ridge_mask(W, H, 880, 40, 600, seed=142)
        m[gm > 0.5] = 255
        for i, (x, sc) in enumerate(((330, 1.0), (760, 0.72), (1080, 0.55), (1330, 0.42), (1540, 0.34),
                                     (1700, 0.28))):
            base = int(top[min(W - 1, x)]) + 4
            hh = int(240 * sc)
            cv2.fillPoly(m, [np.array([[x - 40 * sc, base], [x + 40 * sc, base], [x + 10 * sc, base - hh],
                                       [x - 10 * sc, base - hh]], np.int32)], 255, cv2.LINE_AA)
            cx, cy = x, base - hh
            ang = -35
            cv2.ellipse(m, (int(cx), int(cy)), (int(190 * sc), int(70 * sc)), ang, 0, 180, 255, -1, cv2.LINE_AA)
            cv2.ellipse(m, (int(cx), int(cy)), (int(190 * sc), int(18 * sc)), ang, 180, 360, 255, -1, cv2.LINE_AA)
            fx = cx + math.cos(math.radians(ang - 90)) * 150 * sc
            fy = cy + math.sin(math.radians(ang - 90)) * 150 * sc
            for sgn in (-1, 1):
                ex = cx + sgn * math.cos(math.radians(ang)) * 170 * sc
                ey = cy + sgn * math.sin(math.radians(ang)) * 170 * sc
                cv2.line(m, (int(ex), int(ey)), (int(fx), int(fy)), 255, max(1, int(4 * sc)), cv2.LINE_AA)
            cv2.circle(m, (int(fx), int(fy)), max(2, int(9 * sc)), 255, -1, cv2.LINE_AA)
        self.ground = cv2.GaussianBlur(m, (0, 0), 0.8).astype(np.float32) / 255
        ys = np.arange(H, dtype=np.float32)[:, None, None]
        self.hglow = np.exp(-((ys - 860) / 120) ** 2) * np.array([0.10, 0.10, 0.18], np.float32)
        # golden record texture
        R = 700
        yy, xx = np.mgrid[0:R, 0:R].astype(np.float32)
        dx, dy = xx - R / 2, yy - R / 2
        r = np.sqrt(dx * dx + dy * dy) / (R / 2)
        self.rec_r = r
        self.rec_th = np.arctan2(dy, dx)
        grooves = 0.85 + 0.15 * np.sin(r * 700)
        self.rec_grooves = grooves.astype(np.float32)
        self.rec_mask = np.clip((1 - r) * R / 3, 0, 1).astype(np.float32)
        self.rec_label = ((r < 0.34) & (r > 0.03)).astype(np.float32)
        self.stars = star_layer(W, H, 5000, seed=143, bright=0.8)

    def record(self, t):
        R = self.rec_r.shape[0]
        la = t * 0.35
        spec = np.abs(np.cos(2 * (self.rec_th - la))) ** 18
        base = np.array([0.55, 0.40, 0.14], np.float32)
        col = base[None, None, :] * (0.45 + 0.35 * self.rec_grooves[..., None])
        col = col + spec[..., None] * np.array([1.0, 0.85, 0.5], np.float32) * 1.4 * (self.rec_r > 0.36)[..., None]
        col = col * (1 - self.rec_label[..., None]) + self.rec_label[..., None] * \
            np.array([0.62, 0.48, 0.20], np.float32) * (0.9 + 0.1 * np.cos(self.rec_th * 3))[..., None]
        hole = (self.rec_r < 0.03)[..., None]
        col = col * (1 - hole)
        return col.astype(np.float32), self.rec_mask

    def render(self, t, f):
        if t < 6.5:
            k = t / 6.5
            img = warp(self.sky, 0.86 + 0.04 * k, 8 * k, ty=-40)
            img += self.hglow
            g = self.ground[..., None]
            img = img * (1 - g) + g * np.array([0.004, 0.005, 0.01], np.float32)
            bloom(img, 0.6, 0.3, 30)
            if t > 6.0:
                img *= 1 - (t - 6.0) / 0.5
            return img
        lt = t - 6.5
        img = self.stars.copy() * 0.8
        rec, m = self.record(lt)
        rec = rec * m[..., None]
        s = 0.80 + 0.10 * lt / 6
        tilt = 0.55 + 0.1 * math.sin(lt * 0.4)
        R = rec.shape[0]
        M = np.array([[s * math.cos(0.2), -s * math.sin(0.2), 0], [s * tilt * math.sin(0.2), s * tilt * math.cos(0.2), 0]],
                     np.float32)
        M[0, 2] = 1500 - (M[0, 0] * R / 2 + M[0, 1] * R / 2)
        M[1, 2] = 560 - (M[1, 0] * R / 2 + M[1, 1] * R / 2)
        rw = cv2.warpAffine(rec, M, (W, H))
        mw = cv2.warpAffine(m, M, (W, H))[..., None]
        img = img * (1 - mw) + rw
        add_glow(img, 1500 - 200, 560 - 100, 40, (1.0, 0.9, 0.7), 0.2 + 0.15 * math.sin(lt), kind="halo")
        bloom(img, 0.8, 0.5, 40)
        if lt < 0.5:
            img *= lt / 0.5
        return img

    def overlay(self, img, t, f):
        chapter_card(img, t, 0.3, 5.9, "CHAPTER  III", "我 们 也 在 呼 唤", "WE  ARE  CALLING  TOO", y=H / 2 - 40)
        items = [("1960", "奥兹玛计划：人类首次用射电望远镜倾听外星文明的信号"),
                 ("1974", "阿雷西博信息：向 2.5 万光年外的 M13 星团发出人类的“自我介绍”"),
                 ("1977", "旅行者号携带“金唱片”出发——如今它已是离地球最远的人造物体")]
        for i, (yr, txt) in enumerate(items):
            t0 = 6.8 + i * 1.5
            a = window(t, t0, 12.4, 0.5, 0.4)
            if a <= 0:
                continue
            y = 330 + i * 120
            dx = (1 - ease_out((t - t0) / 0.7)) * -30
            draw_text(img, yr, 130 + dx, y, "shoulders", 64, (255, 214, 120), a, tracking=0.05)
            draw_text(img, txt, 270 + dx, y + 20, "serif_bold", 27, (240, 240, 240), a, shadow=0.8)
        return img


# ----------------------------------------------------------------------------- S15

class WowSignal(Scene):
    dur = 10.0
    grain = 0.035
    vig = 0.55

    def setup(self):
        pw, ph = 1700, 1300
        base = paper(pw, ph, base=(0.93, 0.92, 0.87), seed=151, stains=0.3)
        im = Image.new("L", (pw, ph), 0)
        d = ImageDraw.Draw(im)
        rng = np.random.default_rng(152)
        fnt = font("plex", 30)
        ncol = 24
        self.wow_col = 13
        cw = 58
        x0 = 150
        for r in range(26):
            y = 120 + r * 42
            for c in range(ncol):
                v = rng.random()
                ch = " " if v < 0.45 else ("1" if v < 0.85 else str(rng.integers(2, 4)))
                if c == self.wow_col and 9 <= r <= 14:
                    ch = "6EQUJ5"[r - 9]
                d.text((x0 + c * cw, y), ch, font=fnt, fill=255)
        for y in range(90, ph, 42):
            pass
        ink = np.asarray(im, np.float32) / 255
        pap = base * (1 - 0.75 * ink[..., None] * np.array([0.9, 0.9, 0.85], np.float32))
        for y in range(40, ph, 84):
            cv2.circle(pap, (50, y), 14, (0.2, 0.18, 0.16), -1, cv2.LINE_AA)
            cv2.circle(pap, (pw - 50, y), 14, (0.2, 0.18, 0.16), -1, cv2.LINE_AA)
        self.pap = pap
        self.cx = x0 + self.wow_col * cw + 10
        self.cy = 120 + 11.5 * 42 + 18
        self.pw, self.ph = pw, ph
        self.desk = desk_bg(9, lamp=(960, 500)) * 0.7

    def render(self, t, f):
        pap = self.pap.copy()
        u = ease_io(clamp01((t - 2.0) / 1.2))
        if u > 0:
            m = np.zeros((self.ph, self.pw), np.uint8)
            cv2.ellipse(m, (int(self.cx), int(self.cy)), (46, 160), 4, -90, -90 + 360 * u, 255, 7, cv2.LINE_AA)
            mm = cv2.GaussianBlur(m, (0, 0), 0.8).astype(np.float32)[..., None] / 255
            pap = pap * (1 - mm * 0.85) + mm * 0.85 * np.array([0.78, 0.12, 0.10], np.float32)
        a = ramp(t, 3.4, 3.9)
        if a > 0:
            im = Image.new("L", (400, 200), 0)
            d = ImageDraw.Draw(im)
            d.text((10, 10), "Wow!", font=font("hand", 110), fill=255)
            w_ = np.asarray(im, np.float32) / 255
            reveal = np.clip((np.arange(400)[None, :] - 400 * (t - 3.4) / 0.6), -1, 0) + 1
            w_ = w_ * (1 - reveal) if False else w_ * np.clip(1 - (np.arange(400)[None, :] / 400 - (t - 3.4) / 0.6) * 8,
                                                                0, 1)
            x0, y0 = int(self.cx - 400), int(self.cy - 120)
            reg = pap[y0:y0 + 200, x0:x0 + 400]
            reg *= 1 - w_[..., None] * 0.85
            reg += w_[..., None] * 0.85 * np.array([0.78, 0.12, 0.10], np.float32)
        img = self.desk.copy()
        k = ease_io(t / self.dur)
        s = 0.62 + 0.55 * k
        cx = 960 - (self.cx - self.pw / 2) * s * k
        cy = 560 - (self.cy - self.ph / 2) * s * k
        img = rotate_paste(img, pap, cx, cy, s, 1.5 - 1.5 * k, 1.0, 0.5)
        return img

    def overlay(self, img, t, f):
        recreation_tag(img, window(t, 0.2, 9.8))
        caption(img, t, 0.6, 9.8, "1977 年 8 月 15 日  ·  美国俄亥俄州 “大耳朵”射电望远镜",
                "持续 72 秒的神秘信号",
                ("天文学家杰里·埃曼在打印纸上圈出这串字符，写下“Wow!”。",
                 "此后人类无数次将望远镜对准同一片星空——它再也没有出现。"),
                bottom=H - 80, title_size=52)
        return img
