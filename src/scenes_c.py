"""Act IV–V: sightings 1957–2010 and the disclosure era 2004–2025."""
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw

from core import (H, W, FPS, RED, add_ellipse_glow, add_glow, blank, bloom, caption, chapter_card, chapter_tag,
                  draw_text, ease_in, ease_io, ease_out, fbm, glitch, grid, hash01, lerp, noise1d, ramp,
                  recreation_tag, ridge_mask, smooth, star_layer, subtitle, text_size, warp, window, clamp01,
                  shake_offsets, darken_region, desaturate, font, splat_points, contrast, flicker, scanlines,
                  aa_mask_from_poly, vgradient, chroma_shift)
from elements import FlirView, city_lights, night_sky, cloud_layer, paper
from scene import Scene


def add_patch(img, patch, cx, cy):
    ph, pw = patch.shape[:2]
    x0, y0 = int(cx - pw / 2), int(cy - ph / 2)
    X0, Y0 = max(0, x0), max(0, y0)
    X1, Y1 = min(W, x0 + pw), min(H, y0 + ph)
    if X1 > X0 and Y1 > Y0:
        img[Y0:Y1, X0:X1] += patch[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    return img


def mix(img, mask, color):
    m = mask[..., None]
    return img * (1 - m) + m * np.array(color, np.float32)


# ----------------------------------------------------------------------------- S16 montage

CARDS = [
    ("1957", "美国 得克萨斯州 利维兰", ["多名司机报告：一个发光的蛋形物体靠近时，", "汽车引擎与车灯同时熄灭。"]),
    ("1964", "美国 新墨西哥州 索科罗", ["警员朗尼·萨莫拉目击一个蛋形飞行器，", "旁边站着两个“小个子身影”。"]),
    ("1976", "伊朗 德黑兰", ["两架 F-4 战机前去拦截一个发光物体——", "逼近时，武器与通讯系统相继失灵。"]),
    ("1981", "中国 西部多省", ["7 月 24 日夜，多地民众同时目击", "夜空中一个巨大的螺旋状光团。"]),
    ("1986", "美国 阿拉斯加上空", ["日航 1628 号货机机组报告：", "巨大的发光物体伴飞近 50 分钟。"]),
    ("1994", "津巴布韦 鲁瓦 阿列尔小学", ["62 名小学生在课间目睹飞行器", "降落在校园旁的树林中。"]),
]


class Montage(Scene):
    dur = 17.5
    grain = 0.05
    vig = 0.55
    CARD = 2.5
    START = 2.5

    def setup(self):
        rng = np.random.default_rng(161)
        xs, ys = grid()
        # 1 Levelland
        bg = vgradient((0.01, 0.015, 0.04), (0.05, 0.06, 0.10))
        bg += star_layer(W, H, 1500, 162, bright=0.6)
        fieldm = (ys > 640).astype(np.float32)
        bg = mix(bg, fieldm, (0.02, 0.02, 0.025))
        road = aa_mask_from_poly([[(1250, 640), (1270, 640), (1700, H), (900, H)]])
        bg = mix(bg, road * 0.9, (0.07, 0.07, 0.075))
        self.bg1 = bg
        # 2 Socorro
        bg = vgradient((0.10, 0.14, 0.30), (0.95, 0.55, 0.30), power=1.6)
        g = (ys > 700).astype(np.float32)
        bg = mix(bg, g, (0.32, 0.22, 0.14))
        bg *= (0.85 + 0.3 * fbm(W, H, 30, 4, 0.6, 163))[..., None] ** (g[..., None])
        m = np.zeros((H, W), np.uint8)
        for _ in range(160):
            x, y = rng.uniform(0, W), rng.uniform(710, H)
            r = int(4 + (y - 700) / 25 * rng.uniform(0.5, 1.5))
            cv2.circle(m, (int(x), int(y)), r, 255, -1, cv2.LINE_AA)
        hills, _ = ridge_mask(W, H, 700, 60, 400, 164)
        bg = mix(bg, hills * 0.8, (0.25, 0.14, 0.12))
        bg = mix(bg, m.astype(np.float32) / 255, (0.08, 0.06, 0.04))
        self.bg2 = bg
        # 3 Tehran
        bg = vgradient((0.0, 0.0, 0.02), (0.02, 0.03, 0.05))
        bg += star_layer(W, H, 2500, 165, bright=0.7)
        cm, _ = ridge_mask(W, H, 900, 40, 300, 166)
        bg += city_lights(W, H, 900, 167, n=5000) * cm[..., None] * 0.6
        self.bg3 = bg
        # 4 China 1981
        bg = vgradient((0.01, 0.01, 0.03), (0.03, 0.04, 0.08))
        bg += star_layer(W, H, 4000, 168, bright=0.8)
        mm, _ = ridge_mask(W, H, 930, 260, 380, 169, sharp=True)
        bg = mix(bg, mm, (0.005, 0.006, 0.01))
        self.bg4 = bg
        s = 900
        yy, xx = np.mgrid[0:s, 0:s].astype(np.float32) - s / 2
        self.sp_r = np.sqrt(xx * xx + yy * yy) + 1e-3
        self.sp_th = np.arctan2(yy, xx)
        # 5 JAL
        bg = vgradient((0.0, 0.0, 0.015), (0.02, 0.025, 0.04))
        bg += star_layer(W, H, 3000, 170, bright=0.6)
        gm = (ys > 760).astype(np.float32)
        bg = mix(bg, gm, (0.01, 0.012, 0.018))
        bg += city_lights(W, H, 760, 171, n=900, color=(1.0, 0.8, 0.6)) * gm[..., None] * 0.3
        fr = aa_mask_from_poly([[(0, 0), (W, 0), (W, 120), (0, 60)], [(0, 0), (160, 0), (330, H), (0, H)],
                                [(0, H - 90), (W, H - 150), (W, H), (0, H)], [(W - 200, 0), (W, 0), (W, H), (W - 60, H)]])
        self.frame5 = fr
        self.bg5 = bg
        # 6 Ariel School
        bg = vgradient((0.28, 0.45, 0.75), (0.75, 0.82, 0.88))
        g = (ys > 640).astype(np.float32)
        gtex = fbm(W, H, 40, 5, 0.6, 172)
        bg = mix(bg, g, (0.45, 0.40, 0.22))
        bg *= (0.8 + 0.4 * gtex)[..., None] ** g[..., None]
        m = np.zeros((H, W), np.uint8)
        for _ in range(60):
            x = rng.uniform(700, W)
            y = rng.uniform(560, 660)
            r = int(rng.uniform(30, 80))
            cv2.circle(m, (int(x), int(y)), r, 255, -1, cv2.LINE_AA)
            cv2.rectangle(m, (int(x - 4), int(y)), (int(x + 4), int(y + 60)), 255, -1)
        trees = cv2.GaussianBlur(m, (0, 0), 2).astype(np.float32) / 255
        trees *= 0.7 + 0.5 * fbm(W, H, 12, 3, 0.6, 173)
        self.trees6 = np.clip(trees, 0, 1)
        kids = np.zeros((H, W), np.uint8)
        for i in range(14):
            x = 120 + i * 60 + rng.uniform(-10, 10)
            hgt = rng.uniform(110, 150)
            y = H - 30 - rng.uniform(0, 30)
            cv2.circle(kids, (int(x), int(y - hgt)), 14, 255, -1, cv2.LINE_AA)
            cv2.rectangle(kids, (int(x - 16), int(y - hgt + 14)), (int(x + 16), int(y - hgt * 0.4)), 255, -1)
            cv2.rectangle(kids, (int(x - 12), int(y - hgt * 0.4)), (int(x + 12), int(y)), 255, -1)
        self.kids6 = cv2.GaussianBlur(kids, (0, 0), 1).astype(np.float32) / 255
        self.bg6 = bg

    def card_bg(self, i, lt, f):
        if i == 0:
            img = self.bg1.copy()
            ox, oy = 1330 + 20 * math.sin(lt * 1.5), 430
            p = 0.8 + 0.2 * math.sin(lt * 9)
            add_ellipse_glow(img, ox, oy, 90, 36, 0.1, (1.0, 0.55, 0.25), 1.6 * p)
            add_ellipse_glow(img, ox, oy, 34, 14, 0.1, (1.0, 0.95, 0.8), 2.5 * p)
            add_glow(img, ox, oy, 200, (1.0, 0.5, 0.2), 0.35 * p, kind="halo")
            add_ellipse_glow(img, 1300, 820, 300, 60, 0, (1.0, 0.5, 0.25), 0.25 * p)
            head = 0.0 if (lt > 1.0 and not (1.25 < lt < 1.35)) else 1.0
            for dx in (-40, 40):
                add_glow(img, 1260 + dx, 900, 16, (1.0, 0.95, 0.8), 2.5 * head)
                add_glow(img, 1260 + dx, 900, 90, (1.0, 0.9, 0.7), 0.4 * head, kind="halo")
        elif i == 1:
            img = self.bg2.copy()
            rise = ease_in(lt / 2.5, 2) * 160
            cx, cy = 1330, 650 - rise
            m = np.zeros((H, W), np.uint8)
            cv2.ellipse(m, (int(cx), int(cy)), (95, 52), 0, 0, 360, 255, -1, cv2.LINE_AA)
            for dx in (-60, 60):
                cv2.line(m, (int(cx + dx * 0.6), int(cy + 40)), (int(cx + dx), int(cy + 70)), 255, 5, cv2.LINE_AA)
            mk = m.astype(np.float32) / 255
            img = mix(img, mk, (0.78, 0.78, 0.80))
            shade = aa_mask_from_poly([[(cx - 95, cy), (cx + 95, cy), (cx + 95, cy + 60), (cx - 95, cy + 60)]]) * mk
            img *= (1 - 0.45 * shade)[..., None]
            img = mix(img, aa_mask_from_poly([[(cx - 10, cy - 18), (cx + 10, cy - 18), (cx + 10, cy + 2),
                                              (cx - 10, cy + 2)]]) * 0.8, (0.7, 0.1, 0.1))
            fl = 0.8 + 0.2 * math.sin(lt * 40)
            add_ellipse_glow(img, cx, cy + 80, 30, 60, 0, (0.6, 0.7, 1.0), 1.4 * fl)
            add_ellipse_glow(img, cx, cy + 90, 60, 40, 0, (1.0, 0.6, 0.3), 1.0 * fl)
        elif i == 2:
            img = self.bg3.copy()
            cols = [(1, 0.2, 0.2), (0.3, 1, 0.4), (0.4, 0.5, 1), (1, 0.6, 0.2)]
            c = cols[int(lt * 8) % 4]
            add_glow(img, 1380, 430, 14, (1, 1, 1), 3.0)
            add_glow(img, 1380, 430, 60, c, 1.6, kind="halo")
            hud = np.zeros((H, W), np.uint8)
            cx, cy = 1180, 520
            for k in range(-3, 4):
                y = int(cy + k * 110 + 20 * math.sin(lt))
                if k:
                    cv2.line(hud, (cx - 260, y), (cx - 90, y), 255, 2, cv2.LINE_AA)
                    cv2.line(hud, (cx + 90, y), (cx + 260, y), 255, 2, cv2.LINE_AA)
            cv2.circle(hud, (cx, cy), 10, 255, 2, cv2.LINE_AA)
            cv2.line(hud, (cx - 40, cy), (cx - 12, cy), 255, 2, cv2.LINE_AA)
            cv2.line(hud, (cx + 12, cy), (cx + 40, cy), 255, 2, cv2.LINE_AA)
            h = hud.astype(np.float32) / 255
            img += h[..., None] * np.array([0.3, 1.0, 0.45], np.float32) * 0.7
            if int(lt * 3) % 2 == 0:
                draw_text(img, "WPN  CTRL  FAIL", 1060, 760, "plex_bold", 34, (255, 60, 50), 0.9, tracking=0.1)
        elif i == 3:
            img = self.bg4.copy()
            rot = lt * 0.5
            arms = np.cos(2 * (self.sp_th - np.log(self.sp_r) * 1.4 - rot))
            spiral = np.clip(arms, 0, 1) ** 3 * np.exp(-self.sp_r / 260) * (self.sp_r > 8)
            core = np.exp(-(self.sp_r / 22) ** 2)
            v = spiral * 0.9 + core * 3
            v = cv2.GaussianBlur(v.astype(np.float32), (0, 0), 3)
            add_patch(img, v[..., None] * np.array([0.75, 0.85, 1.0], np.float32), 1350, 400)
        elif i == 4:
            img = self.bg5.copy()
            for (bx, by) in ((1180, 470), (1460, 440)):
                p = 0.7 + 0.3 * math.sin(lt * 3 + bx)
                for r in range(3):
                    for c in range(5):
                        add_glow(img, bx + c * 16 - 32, by + r * 14 - 14, 5, (1.0, 0.75, 0.4), 2.0 * p)
                add_glow(img, bx, by, 110, (1.0, 0.6, 0.3), 0.5 * p, kind="halo")
            img = mix(img, self.frame5, (0.01, 0.01, 0.012))
        else:
            img = self.bg6.copy()
            img = mix(img, self.trees6, (0.10, 0.16, 0.08))
            cx, cy = 1320 + 10 * math.sin(lt * 2), 590
            m = np.zeros((H, W), np.uint8)
            cv2.ellipse(m, (int(cx), int(cy)), (80, 22), 0, 0, 360, 255, -1, cv2.LINE_AA)
            cv2.ellipse(m, (int(cx), int(cy - 12)), (34, 22), 0, 180, 360, 255, -1, cv2.LINE_AA)
            mk = m.astype(np.float32) / 255
            img = mix(img, mk, (0.82, 0.84, 0.88))
            add_glow(img, cx - 30, cy - 8, 8, (1, 1, 1), 2.5 * (0.5 + 0.5 * math.sin(lt * 5)))
            img = mix(img, self.kids6, (0.03, 0.03, 0.04))
        return img

    def render(self, t, f):
        if t < self.START:
            img = blank((0, 0, 0))
            return img
        i = min(5, int((t - self.START) / self.CARD))
        lt = t - self.START - i * self.CARD
        img = self.card_bg(i, lt, f)
        img = darken_region(img, 0, 150, 950, 930, 0.6, 250)
        bloom(img, 0.7, 0.5, 40)
        z = 1.0 + 0.05 * lt / self.CARD
        img = warp(img, z, 0)
        if lt < 0.18:
            img = glitch(img, 1 - lt / 0.18, f, i)
        return img

    def overlay(self, img, t, f):
        if t < self.START:
            chapter_card(img, t, 0.0, self.START, "CHAPTER  IV", "目   击", "SIGHTINGS")
            return img
        i = min(5, int((t - self.START) / self.CARD))
        lt = t - self.START - i * self.CARD
        yr, place, lines = CARDS[i]
        a = ramp(lt, 0.0, 0.12) * (1 - ramp(lt, self.CARD - 0.12, self.CARD))
        s = 1.0 + 0.25 * (1 - ease_out(lt / 0.25))
        draw_text(img, yr, 110, 200, "shoulders", int(260 * s), (255, 255, 255), a, tracking=0.02)
        img[520:524, 116:116 + int(420 * ease_out(lt / 0.5))] += np.float32(0.9 * a)
        draw_text(img, place, 116, 548, "sans_bold", 40, (255, 196, 90), a, tracking=0.08)
        for k, line in enumerate(lines):
            aa = a * ramp(lt, 0.15 + k * 0.15, 0.45 + k * 0.15)
            draw_text(img, line, 116, 624 + k * 58, "serif_bold", 38, (240, 240, 240), aa, shadow=0.8)
        recreation_tag(img, a * 0.9)
        chapter_tag(img, t, self.START, self.dur, "CHAPTER IV  ·  SIGHTINGS")
        return img


# ----------------------------------------------------------------------------- S17 Rendlesham

class Rendlesham(Scene):
    dur = 10.0
    grain = 0.06
    vig = 0.65

    def setup(self):
        rng = np.random.default_rng(171)
        self.bg = vgradient((0.005, 0.01, 0.02), (0.03, 0.05, 0.07))
        self.fog = fbm(W * 2, H, 260, 5, 0.55, 172)
        trees = []
        for _ in range(95):
            trees.append((rng.uniform(0, 1) ** 0.8, rng.uniform(-80, W + 80), rng.uniform(-0.03, 0.03)))
        trees.sort(key=lambda a: -a[0])
        far = np.zeros((H, W), np.float32)
        near = np.zeros((H, W), np.float32)
        for z, x, lean in trees:
            wdt = lerp(90, 5, z)
            gy = lerp(H + 40, 690, z)
            m = np.zeros((H, W), np.uint8)
            top = -50
            pts = np.array([[x - wdt / 2, gy], [x - wdt / 2 * 0.8 + lean * 900, top], [x + wdt / 2 * 0.8 + lean * 900, top],
                            [x + wdt / 2, gy]], np.int32)
            cv2.fillPoly(m, [pts], 255, cv2.LINE_AA)
            for b in range(int(rng.integers(0, 4))):
                by = rng.uniform(80, gy * 0.6)
                bl = rng.uniform(40, 200) * (1 - z * 0.7)
                sgn = rng.choice([-1, 1])
                cv2.line(m, (int(x + lean * (gy - by)), int(by)), (int(x + sgn * bl), int(by - bl * 0.4)), 255,
                         max(1, int(wdt * 0.15)), cv2.LINE_AA)
            mm = m.astype(np.float32) / 255
            if z > 0.55:
                fogk = (z - 0.55) / 0.45
                far = far * (1 - mm) + mm * (1 - 0.85 * fogk) * 0.8
            else:
                near = np.maximum(near, mm)
        self.far = far
        self.near = cv2.GaussianBlur(near, (0, 0), 1.2)
        ys = np.arange(H, dtype=np.float32)[:, None]
        self.ground = np.clip((ys - 700) / 300, 0, 1)

    def render(self, t, f):
        img = self.bg.copy()
        ox = int(t * 25) % W
        fog = self.fog[:, ox:ox + W]
        img += (fog * 0.06)[..., None] * np.array([0.6, 0.75, 1.0], np.float32)
        lx, ly = 1120 + 30 * math.sin(t * 0.4), 610 + 8 * math.sin(t * 1.3)
        pulse = 0.5 + 0.5 * math.sin(t * 5.5)
        add_glow(img, lx, ly, 280, (0.9, 0.3, 0.25), 0.25 + 0.15 * pulse, kind="halo", reach=1400)
        add_glow(img, lx, ly, 18, (1.0, 0.3, 0.2), 2.5 * pulse + 0.5)
        for k in range(3):
            a = t * 2.5 + k * 2.09
            add_glow(img, lx + math.cos(a) * 55, ly + 30 + math.sin(a) * 10, 7, (0.6, 0.75, 1.0), 2.5)
        add_glow(img, lx, ly - 30, 10, (1.0, 1.0, 1.0), 1.5 + 1.5 * (int(t * 2) % 2))
        f_ = self.far[..., None]
        img = img * (1 - f_) + f_ * (img * 0.25 + np.array([0.02, 0.03, 0.04], np.float32))
        fogl = np.zeros_like(img)
        add_glow(fogl, lx, ly, 200, (1.0, 0.4, 0.3), 0.55 + 0.25 * pulse, kind="halo", reach=1100)
        add_glow(fogl, lx, ly, 26, (1.0, 0.5, 0.4), 1.2 * pulse + 0.3)
        img += fogl * (0.5 + 0.7 * fog[..., None])
        img = mix(img, self.ground * 0.8, (0.01, 0.012, 0.015))
        # flashlight beams
        for sx, ph in ((500, 0.0), (1500, 1.7)):
            ang = -1.35 + 0.18 * math.sin(t * 0.9 + ph)
            add_ellipse_glow(img, sx + math.cos(ang) * 500, H + math.sin(ang) * 500, 520, 70, ang,
                             (0.85, 0.9, 1.0), 0.10)
        n = self.near[..., None]
        img = img * (1 - n) + n * np.array([0.004, 0.005, 0.008], np.float32)
        bloom(img, 0.5, 0.7, 50)
        dx, dy = shake_offsets(t, 9, 1.8, 5)
        return warp(img, 1.06 + 0.02 * t / 10, 0.6 * math.sin(t * 0.8), tx=dx, ty=dy)

    def overlay(self, img, t, f):
        recreation_tag(img, window(t, 0.2, 9.8))
        chapter_tag(img, t, 0.2, 9.8, "CHAPTER IV  ·  SIGHTINGS")
        caption(img, t, 0.8, 9.8, "1980 年 12 月  ·  英国 伦德尔舍姆森林",
                "北约空军基地外的森林之谜",
                ("驻英美军人员在森林中目击发光物体。基地副指挥官查尔斯·霍尔特中校",
                 "亲自带队调查，录下现场录音，并以正式备忘录上报英国国防部。"),
                bottom=H - 90, title_size=50)
        return img


# ----------------------------------------------------------------------------- S18 Belgium

class BelgianTriangle(Scene):
    dur = 7.5
    grain = 0.05
    vig = 0.6

    def setup(self):
        rng = np.random.default_rng(181)
        bg = vgradient((0.01, 0.015, 0.035), (0.10, 0.08, 0.09))
        bg += star_layer(W, H, 1200, 182, bright=0.5)
        cl = cloud_layer(W, H, 300, 183, 0.45)
        bg = bg * (1 - cl[..., None] * 0.5) + cl[..., None] * 0.5 * np.array([0.10, 0.08, 0.09], np.float32)
        m = np.zeros((H, W), np.uint8)
        win = np.zeros((H, W), np.uint8)
        x = -20
        while x < W:
            bw = int(rng.uniform(90, 170))
            bh = int(rng.uniform(90, 180))
            gy = H
            base = 920
            if abs(x - 1500) < 80:
                cv2.rectangle(m, (x + 30, base - 380), (x + 80, base), 255, -1)
                cv2.fillPoly(m, [np.array([[x + 30, base - 380], [x + 55, base - 520], [x + 80, base - 380]],
                                          np.int32)], 255, cv2.LINE_AA)
            cv2.rectangle(m, (x, base - bh), (x + bw, gy), 255, -1)
            cv2.fillPoly(m, [np.array([[x - 5, base - bh], [x + bw // 2, base - bh - 70], [x + bw + 5, base - bh]],
                                      np.int32)], 255, cv2.LINE_AA)
            for wy in range(base - bh + 20, H - 20, 42):
                for wx in range(x + 14, x + bw - 20, 36):
                    if rng.random() < 0.35:
                        cv2.rectangle(win, (wx, wy), (wx + 14, wy + 20), 255, -1)
            x += bw + int(rng.uniform(4, 20))
        town = m.astype(np.float32) / 255
        wn = win.astype(np.float32) / 255
        bg = mix(bg, town, (0.012, 0.012, 0.018))
        bg += wn[..., None] * np.array([1.0, 0.70, 0.35], np.float32) * 0.7
        bg += cv2.GaussianBlur(wn, (0, 0), 12)[..., None] * np.array([1.0, 0.6, 0.3], np.float32) * 0.5
        self.bg = bg

    def render(self, t, f):
        img = self.bg.copy()
        k = t / self.dur
        cx, cy = lerp(700, 1150, k), lerp(330, 300, k)
        s = 1.0 + 0.1 * k
        pts = []
        corners = []
        for a in (-90, 30, 150):
            r = 360 * s
            ang = math.radians(a + 4 * math.sin(t * 0.3))
            px, py = cx + math.cos(ang) * r, cy + math.sin(ang) * r * 0.5
            corners.append((px, py))
        m = aa_mask_from_poly([corners], ss=2)
        m = cv2.GaussianBlur(m, (0, 0), 6)
        img = img * (1 - 0.92 * m[..., None]) + m[..., None] * 0.92 * np.array([0.015, 0.015, 0.02], np.float32)
        for (px, py) in corners:
            ix, iy = cx + (px - cx) * 0.8, cy + (py - cy) * 0.8
            add_glow(img, ix, iy, 13, (1.0, 1.0, 0.95), 2.8)
            add_glow(img, ix, iy, 80, (0.9, 0.95, 1.0), 0.35, kind="halo")
        pulse = 0.5 + 0.5 * math.sin(t * 6)
        add_glow(img, cx, cy + 10, 16, (1.0, 0.15, 0.1), 2.4 * pulse)
        add_glow(img, cx, cy + 10, 70, (1.0, 0.2, 0.1), 0.5 * pulse, kind="halo")
        bloom(img, 0.6, 0.6, 50)
        dx, dy = shake_offsets(t, 3, 1.5, 8)
        return warp(img, 1.04, 0, tx=dx, ty=dy)

    def overlay(self, img, t, f):
        recreation_tag(img, window(t, 0.2, 7.3))
        chapter_tag(img, t, 0.2, 7.3, "CHAPTER IV  ·  SIGHTINGS")
        caption(img, t, 0.5, 7.4, "1989—1990  ·  比利时",
                "黑色三角",
                ("上千名民众报告目击巨大的三角形飞行器，底部闪着三盏白灯与一盏红灯。",
                 "比利时空军出动 F-16 战机追踪，雷达数次短暂锁定目标。"),
                bottom=H - 90, title_size=56)
        return img


# ----------------------------------------------------------------------------- S19 Phoenix

class PhoenixLights(Scene):
    dur = 7.5
    grain = 0.05
    vig = 0.6

    def setup(self):
        bg = vgradient((0.01, 0.012, 0.03), (0.12, 0.08, 0.10), power=1.8)
        bg += star_layer(W, H, 1800, 191, bright=0.55)
        mm, top = ridge_mask(W, H, 620, 90, 380, 192, sharp=True)
        bg = mix(bg, mm, (0.02, 0.018, 0.025))
        lights = city_lights(W, H, 640, 193, n=26000, spread=1.8, grid_like=True)
        gm = np.clip((grid()[1] - 640) / 10, 0, 1)
        bg += lights * gm[..., None] * 0.9
        self.bg = bg

    def render(self, t, f):
        img = self.bg.copy()
        k = t / self.dur
        cx, cy = lerp(1350, 800, k), lerp(300, 330, k)
        for i in range(-3, 4):
            x = cx + i * 95
            y = cy + abs(i) * 42
            fl = 0.9 + 0.1 * math.sin(t * 3 + i)
            add_glow(img, x, y, 9, (1.0, 0.72, 0.35), 3.2 * fl)
            add_glow(img, x, y, 60, (1.0, 0.6, 0.25), 0.4 * fl, kind="halo")
        bloom(img, 0.55, 0.6, 50)
        return warp(img, 1.02 + 0.03 * k, 0)

    def overlay(self, img, t, f):
        recreation_tag(img, window(t, 0.2, 7.3))
        chapter_tag(img, t, 0.2, 7.3, "CHAPTER IV  ·  SIGHTINGS")
        caption(img, t, 0.5, 7.4, "1997 年 3 月 13 日  ·  美国 凤凰城",
                "凤凰城之光",
                ("数千人目击一组呈 V 字形排列的巨大灯光，静静掠过城市上空。",
                 "时任州长后来公开承认：那天晚上，他本人也看到了。"),
                bottom=H - 90, title_size=56)
        return img


# ----------------------------------------------------------------------------- S20 Hangzhou

BOARD = [("北京", "20:35", 1), ("广州", "20:40", 2), ("成都", "20:45", 1), ("西安", "20:55", 2),
         ("深圳", "21:05", 1), ("厦门", "21:10", 2)]


class Hangzhou(Scene):
    dur = 10.0
    grain = 0.045
    vig = 0.55

    def setup(self):
        bg = vgradient((0.01, 0.012, 0.03), (0.10, 0.07, 0.07), power=1.5)
        cl = cloud_layer(W, H, 280, 201, 0.5)
        ys = np.arange(H, dtype=np.float32)[:, None]
        bg = bg * (1 - cl[..., None] * 0.6) + cl[..., None] * 0.6 * np.array([0.14, 0.09, 0.07], np.float32) * \
            np.clip((ys / 600), 0.2, 1)[..., None]
        gm = (ys > 600).astype(np.float32) * np.ones((1, W), np.float32)
        bg = mix(bg, gm, (0.01, 0.012, 0.016))
        lights = np.zeros((H, W, 3), np.float32)
        vx, vy = 900, 600
        pts, cols = [], []
        for side in (-1, 1):
            for k in range(60):
                z = k / 60
                d = 1 / (0.02 + z * 0.98)
                x = vx + side * 900 / d * 1.4
                y = vy + 480 / d * 1.0
                if y > H:
                    continue
                pts.append((x, y))
                cols.append((1.0, 0.85, 0.5) if k % 2 else (0.5, 0.7, 1.0))
        for k in range(40):
            z = k / 40
            d = 1 / (0.02 + z * 0.98)
            pts.append((vx, vy + 480 / d))
            cols.append((0.9, 0.9, 1.0))
        p = np.array(pts, np.float32)
        c = np.array(cols, np.float32) * 3
        splat_points(lights, p[:, 0], p[:, 1], c)
        lights = lights + cv2.GaussianBlur(lights, (0, 0), 4) * 3 + cv2.GaussianBlur(lights, (0, 0), 15) * 3
        bg += lights
        term = aa_mask_from_poly([[(1300, 600), (1920, 560), (1920, 700), (1300, 640)]])
        bg = mix(bg, term, (0.02, 0.02, 0.025))
        rng = np.random.default_rng(202)
        for k in range(50):
            x = rng.uniform(1320, 1900)
            y = lerp(598, 560, (x - 1300) / 620) + rng.uniform(15, 40)
            add_glow(bg, x, y, 3, (1.0, 0.85, 0.6), 1.2)
        self.bg = bg
        self.flaps = {}

    def render(self, t, f):
        if t < 5.0:
            img = self.bg.copy()
            ox, oy = 1000 + 10 * math.sin(t), 300 + 5 * math.sin(t * 1.7)
            for j in range(26):
                add_glow(img, ox - j * 9, oy + j * 7, 10 - j * 0.3, (1.0, 0.9, 0.7), 0.5 * (1 - j / 26))
            add_glow(img, ox, oy, 12, (1, 1, 1), 3.0)
            add_glow(img, ox, oy, 90, (1.0, 0.85, 0.6), 0.5, kind="halo")
            bloom(img, 0.6, 0.6, 40)
            img = warp(img, 1.0 + 0.04 * t / 5, 0)
            if t > 4.8:
                img *= 1 - (t - 4.8) / 0.2
            return img
        lt = t - 5.0
        img = blank((0.015, 0.015, 0.02))
        x0, y0 = 170, 150
        rowh = 88
        img[y0 - 10:y0 + 70, x0 - 20:W - x0 + 20] = (0.05, 0.08, 0.16)
        for i, (city, sched, kind) in enumerate(BOARD):
            y = y0 + 90 + i * rowh
            img[y:y + rowh - 10, x0 - 20:W - x0 + 20] = (0.03, 0.03, 0.035)
            for cx0 in range(x0, W - x0, 64):
                img[y + (rowh - 10) // 2, cx0:cx0 + 58] *= 0.4
        return img

    def overlay(self, img, t, f):
        recreation_tag(img, window(t, 0.2, 9.8))
        if t >= 5.0:
            lt = t - 5.0
            x0, y0 = 170, 150
            rowh = 88
            for k, h in enumerate(("始发地  FROM", "计划  SCHED", "状态  STATUS")):
                draw_text(img, h, x0 + [0, 520, 900][k], y0 + 6, "sans_bold", 34, (230, 230, 230), 1.0, tracking=0.08)
            for i, (city, sched, kind) in enumerate(BOARD):
                y = y0 + 102 + i * rowh
                draw_text(img, city, x0, y, "sans_bold", 44, (240, 240, 240), 1.0)
                draw_text(img, sched, x0 + 520, y, "plex_bold", 44, (240, 240, 240), 1.0)
                settle = 0.4 + i * 0.35
                if lt < settle:
                    status, col = ("正常  ON TIME", (120, 230, 140)) if lt < settle - 0.35 else \
                        ("".join("▒█▓"[int((lt * 30 + j + i) % 3)] for j in range(6)), (200, 200, 200))
                else:
                    status, col = (("延误  DELAYED", (255, 190, 60)) if kind == 1 else ("备降  DIVERTED", (255, 70, 60)))
                draw_text(img, status, x0 + 900, y, "sans_bold", 44, col, 1.0, tracking=0.04)
            draw_text(img, "杭州萧山国际机场  HANGZHOU XIAOSHAN", W - x0, 104, "sans_bold", 30, (200, 200, 200),
                      1.0, anchor="rt", tracking=0.06)
        caption(img, t, 0.6, 9.8, "2010 年 7 月 7 日  ·  中国 杭州萧山机场",
                "不明飞行物出现，机场临时关闭",
                ("当晚，机场上空出现不明飞行物，空管部门紧急关闭航路，",
                 "多个航班延误或备降外地——事件原因至今没有明确结论。"),
                bottom=H - 70, title_size=50)
        return img


# ----------------------------------------------------------------------------- S21 Navy FLIR

class NavyFlir(Scene):
    dur = 12.5
    grain = 0.02
    vig = 0.35

    def setup(self):
        self.tic = FlirView("tictac", 211)
        self.gof = FlirView("gofast", 212)
        self.gim = FlirView("gimbal", 213)

    def render(self, t, f):
        if t < 5.0:
            x = 0.5 + 0.01 * math.sin(t * 2)
            if t > 4.2:
                x -= ((t - 4.2) / 0.5) ** 2.5 * 0.9
            o = dict(x=x, y=0.48 + 0.006 * math.sin(t * 5), size=12, angle=4, polarity=-1, kind="tictac", aura=1.3)
            s = self.tic.sensor(t, o)
            img = self.tic.finish(s, f)
            img = self.tic.hud(img, t, f, zoom="NAR", polarity="BHT", gate=(0.5, 0.48, 60, 40), bank=2,
                               heading=190 + t, rng_nm=None, alt=20000, mach=0.58, az=None, el=None,
                               extra=("TGT", "RDR ASSIST"))
            if t < 2.2:
                img *= 0.35 + 0.65 * ramp(t, 0.0, 2.2)
        elif t < 8.75:
            lt = t - 5.0
            o = dict(x=0.5 + 0.01 * math.sin(lt * 3), y=0.5 + 0.01 * math.sin(lt * 4.1), size=6, polarity=1,
                     kind="blob", aura=1.5)
            s = self.gof.sensor(lt, o)
            img = self.gof.finish(s, f)
            img = self.gof.hud(img, lt, f, zoom="NAR", polarity="WHT", gate=(0.5, 0.5, 50, 50), bank=-8,
                               heading=45 - lt * 2, rng_nm=4.4, alt=25010, mach=0.61, az=-43, el=-26,
                               extra=("TRK",))
        else:
            lt = t - 8.75
            rot = -20 - 12 * lt
            o = dict(x=0.5 + 0.01 * math.sin(lt), y=0.5, size=34, angle=rot, polarity=-1, kind="saucer", aura=1.2)
            s = self.gim.sensor(lt + 3, o, bank=-24)
            img = self.gim.finish(s, f)
            img = self.gim.hud(img, lt, f, zoom="NAR", polarity="BHT", gate=(0.5, 0.5, 130, 70), bank=-24,
                               heading=350, alt=25004, mach=0.56, az=-50, el=-2, extra=("GIMBAL",))
        for cut in (5.0, 8.75):
            if cut <= t < cut + 0.15:
                img = glitch(img, 1 - (t - cut) / 0.15, f, 5)
        return img

    def overlay(self, img, t, f):
        if t < 2.4:
            a = window(t, 0.0, 2.4, 0.3, 0.4)
            img = darken_region(img, 500, 350, 1420, 730, 0.7 * a, 200)
            chapter_card(img, t, 0.0, 2.4, "CHAPTER  V", "解   密", "DISCLOSURE")
            return img
        if t < 5.0:
            recreation_tag(img, window(t, 2.4, 4.9), "情景再现 · 2004 美国海军“FLIR1”红外画面")
            caption(img, t, 2.4, 5.0, "2004 年 11 月  ·  美国“尼米兹”号航母打击群",
                    "“Tic Tac”：没有机翼，没有尾焰",
                    ("F/A-18 飞行员目击一个约 12 米长的白色椭圆物体，", "在战机接近时瞬间加速离去。"),
                    bottom=H - 140, title_size=46, scrim=0.8)
        elif t < 8.75:
            recreation_tag(img, window(t, 5.0, 8.7), "情景再现 · 2015 美国海军“GOFAST”红外画面")
            subtitle(img, t, 5.2, 6.9, "“哇！锁定了！”", "“Whoa! Got it!”")
            subtitle(img, t, 7.0, 8.7, "“那到底是什么东西？！”", "“What the f*** is that thing?”")
        else:
            recreation_tag(img, window(t, 8.75, 12.4), "情景再现 · 2015 美国海军“GIMBAL”红外画面")
            subtitle(img, t, 8.9, 10.6, "“那边有一整支编队……”", "“There's a whole fleet of them.”")
            subtitle(img, t, 10.7, 12.4, "“它们全都在逆风飞行！风速 120 节！”",
                     "“They're all going against the wind. The wind's 120 knots to the west.”")
        return img


# ----------------------------------------------------------------------------- S22 News

HEADLINES = [
    ("2017.12.16", "《纽约时报》披露：五角大楼曾秘密运行 UFO 研究计划"),
    ("2020.04.27", "美国国防部正式公开三段海军 UFO 视频，承认“无法识别”"),
    ("2021.06.25", "美国情报总监办公室报告：144 起事件，仅 1 起得到解释"),
]
TICKER = ("  ◆  2022.05.17  美国国会举行半个多世纪以来首次 UAP 公开听证会"
          "  ◆  2022.07  五角大楼成立“全域异常解决办公室”（AARO）"
          "  ◆  2023.09  NASA 发布 UAP 独立研究报告，呼吁以科学方法研究"
          "  ◆  2023.07.26  前情报官员在国会宣誓作证")


class NewsMontage(Scene):
    dur = 10.0
    grain = 0.02
    vig = 0.35

    def setup(self):
        self.fl = FlirView("tictac", 221)
        self.bg = vgradient((0.02, 0.04, 0.10), (0.01, 0.02, 0.05))
        xs, ys = grid()
        self.streak_x = xs
        self.streak_y = ys

    def render(self, t, f):
        img = self.bg.copy()
        for k in range(6):
            ph = (t * 0.08 + k / 6) % 1
            x = lerp(-400, W + 400, ph)
            add_ellipse_glow(img, x, 300 + k * 120, 600, 14, -0.25, (0.2, 0.4, 1.0), 0.18)
        seg = min(2, int(t / 3.34))
        lt = t - seg * 3.34
        # inset video panel
        px0, py0, pw, ph = 140, 150, 1020, 574
        if seg < 2:
            o = dict(x=0.5 - (0.0 if seg == 0 else 0.1 * math.sin(lt)), y=0.5, size=12 if seg == 0 else 26,
                     angle=4 if seg == 0 else -20 - lt * 6, polarity=-1, kind="tictac" if seg == 0 else "saucer")
            s = self.fl.sensor(t, o)
            s = cv2.GaussianBlur(s, (0, 0), 0.7)
            v = cv2.resize(s, (pw, ph))
            img[py0:py0 + ph, px0:px0 + pw] = v[..., None]
        else:
            img[py0:py0 + ph, px0:px0 + pw] = (0.03, 0.05, 0.12)
        img[py0 - 3:py0, px0 - 3:px0 + pw + 3] = 0.8
        img[py0 + ph:py0 + ph + 3, px0 - 3:px0 + pw + 3] = 0.8
        img[py0:py0 + ph, px0 - 3:px0] = 0.8
        img[py0:py0 + ph, px0 + pw:px0 + pw + 3] = 0.8
        if lt < 0.15:
            img = glitch(img, 1 - lt / 0.15, f, 7)
        return img

    def overlay(self, img, t, f):
        seg = min(2, int(t / 3.34))
        lt = t - seg * 3.34
        # right column info
        if seg == 0:
            draw_text(img, "AATIP", 1230, 190, "shoulders", 120, (255, 255, 255), ramp(lt, 0.1, 0.4))
            draw_text(img, "先进航空航天威胁识别计划", 1236, 340, "sans_bold", 36, (220, 220, 220), ramp(lt, 0.3, 0.6))
            draw_text(img, "2007 — 2012", 1236, 400, "plex_bold", 34, (255, 196, 90), ramp(lt, 0.5, 0.8))
        elif seg == 1:
            draw_text(img, "FLIR1", 1230, 190, "shoulders", 90, (255, 255, 255), ramp(lt, 0.1, 0.3))
            draw_text(img, "GIMBAL", 1230, 290, "shoulders", 90, (255, 255, 255), ramp(lt, 0.3, 0.5))
            draw_text(img, "GOFAST", 1230, 390, "shoulders", 90, (255, 255, 255), ramp(lt, 0.5, 0.7))
            draw_text(img, "官方认定：“不明空中现象”", 1236, 510, "sans_bold", 34, (255, 196, 90), ramp(lt, 0.8, 1.1))
        else:
            n = int(144 * ease_out(lt / 1.2))
            draw_text(img, f"{n}", 1230, 150, "shoulders", 220, (255, 255, 255), 1.0)
            draw_text(img, "起 UAP 事件（2004—2021）", 1236, 420, "sans_bold", 36, (220, 220, 220), ramp(lt, 0.2, 0.5))
            draw_text(img, "仅 1 起", 1230, 480, "sans_black", 80, (255, 70, 60), ramp(lt, 1.2, 1.4))
            draw_text(img, "得到解释（一只泄气的气球）", 1236, 590, "sans_bold", 34, (220, 220, 220), ramp(lt, 1.4, 1.7))
            draw_text(img, "美国国家情报总监办公室 · 初步评估报告", 180, 400, "sans_bold", 36, (200, 210, 230),
                      ramp(lt, 0.2, 0.5))
            draw_text(img, "PRELIMINARY ASSESSMENT:", 180, 300, "plex_bold", 40, (255, 255, 255), ramp(lt, 0.1, 0.4))
            draw_text(img, "UNIDENTIFIED AERIAL PHENOMENA", 180, 350, "plex_bold", 40, (255, 255, 255),
                      ramp(lt, 0.1, 0.4))
        # lower third
        y0 = 800
        date, head = HEADLINES[seg]
        img[y0:y0 + 70, 140:480] = np.array([0.80, 0.08, 0.08], np.float32)
        draw_text(img, "突发  BREAKING", 160, y0 + 12, "sans_black", 38, (255, 255, 255), 1.0, tracking=0.04)
        wbar = int(1640 * ease_out(lt / 0.4))
        img[y0 + 70:y0 + 160, 140:140 + wbar] = np.array([0.96, 0.96, 0.96], np.float32)
        if wbar > 600:
            draw_text(img, head, 170, y0 + 88, "sans_black", 46, (20, 20, 30), ramp(lt, 0.2, 0.4))
        img[y0:y0 + 70, 480:480 + int(300 * ease_out(lt / 0.3))] = np.array([0.08, 0.10, 0.20], np.float32)
        draw_text(img, date, 500, y0 + 14, "plex_bold", 36, (255, 255, 255), ramp(lt, 0.15, 0.3))
        # ticker
        img[H - 70:H, :] = np.array([0.05, 0.07, 0.16], np.float32)
        img[H - 70:H, :200] = np.array([0.80, 0.08, 0.08], np.float32)
        tw = text_size(TICKER, "sans_bold", 30)
        off = (t * 230) % tw
        for k in range(3):
            draw_text(img, TICKER, 200 - off + k * tw, H - 58, "sans_bold", 30, (235, 235, 235), 1.0)
        img[H - 70:H, :200] = np.array([0.80, 0.08, 0.08], np.float32)
        draw_text(img, "UAP 档案", 30, H - 58, "sans_black", 30, (255, 255, 255), 1.0)
        # live bug
        if int(t * 1.5) % 2 == 0:
            add_glow(img, 170, 110, 8, (1.0, 0.1, 0.1), 2.0)
        draw_text(img, "LIVE", 190, 94, "plex_bold", 30, (255, 255, 255), 1.0, tracking=0.1)
        draw_text(img, "WORLD NEWS  ·  新闻", W - 140, 90, "sans_bold", 28, (255, 255, 255), 0.9, anchor="rt",
                  tracking=0.1)
        chapter_tag(img, t, 0.2, 9.8, "CHAPTER V  ·  DISCLOSURE", x=330, y=98)
        return img


# ----------------------------------------------------------------------------- S23 Hearing

class Hearing(Scene):
    dur = 12.5
    grain = 0.04
    vig = 0.65

    def setup(self):
        rng = np.random.default_rng(231)
        bg = vgradient((0.13, 0.10, 0.08), (0.04, 0.03, 0.03))
        xs, ys = grid()
        wall = fbm(W, H, 120, 4, 0.5, 232)
        bg *= (0.8 + 0.4 * wall)[..., None]
        for x in range(120, W, 260):
            bg[:560, x:x + 18] *= 1.35
        add_ellipse_glow(bg, 960, 180, 900, 200, 0, (0.6, 0.42, 0.25), 0.45)
        add_ellipse_glow(bg, 960, 820, 800, 200, 0, (0.35, 0.25, 0.16), 0.3)
        dais = aa_mask_from_poly([[(0, 470), (W, 470), (W, 610), (0, 610)]])
        bg = mix(bg, dais, (0.12, 0.07, 0.04))
        bg *= (1 - 0.3 * (np.abs(ys - 540) < 3))[..., None]
        m = np.zeros((H, W), np.uint8)
        for i in range(11):
            x = 110 + i * 170 + rng.uniform(-10, 10)
            cv2.ellipse(m, (int(x), 408), (26, 32), 0, 0, 360, 255, -1, cv2.LINE_AA)
            cv2.ellipse(m, (int(x), 490), (62, 60), 0, 180, 360, 255, -1, cv2.LINE_AA)
            cv2.rectangle(m, (int(x - 50), 520), (int(x + 50), 548), 255, -1)
        bg = mix(bg, m.astype(np.float32) / 255, (0.02, 0.018, 0.02))
        for i in range(11):
            x = 110 + i * 170
            bg[520:548, int(x - 50):int(x + 50)] = (0.75, 0.72, 0.62)
        table = aa_mask_from_poly([[(250, 900), (1670, 900), (1800, H), (120, H)]])
        bg = mix(bg, table, (0.06, 0.035, 0.02))
        wit = np.zeros((H, W), np.uint8)
        cv2.ellipse(wit, (960, 740), (80, 96), 0, 0, 360, 255, -1, cv2.LINE_AA)
        cv2.ellipse(wit, (960, 1010), (270, 210), 0, 180, 360, 255, -1, cv2.LINE_AA)
        cv2.rectangle(wit, (690, 1000), (1230, H), 255, -1)
        cv2.line(wit, (1180, 900), (1120, 760), 255, 8, cv2.LINE_AA)
        cv2.ellipse(wit, (1115, 752), (16, 22), -20, 0, 360, 255, -1, cv2.LINE_AA)
        self.wit = cv2.GaussianBlur(wit, (0, 0), 2).astype(np.float32) / 255
        self.bg = bg
        self.flashes = sorted(rng.uniform(0.3, 8.5, 22))
        self.flash_pos = [(rng.uniform(150, 1770), rng.uniform(650, 1000)) for _ in range(22)]

    def render(self, t, f):
        img = self.bg.copy()
        rim = cv2.GaussianBlur(self.wit, (0, 0), 6) - self.wit
        img += np.clip(rim, 0, 1)[..., None] * np.array([0.5, 0.4, 0.3], np.float32) * 0.8
        img = mix(img, self.wit, (0.008, 0.007, 0.008))
        glob = 0.0
        for ft, (fx, fy) in zip(self.flashes, self.flash_pos):
            age = t - ft
            if 0 <= age < 0.12:
                k = 1 - age / 0.12
                add_glow(img, fx, fy, 20, (1, 1, 1), 4 * k)
                add_glow(img, fx, fy, 200, (0.9, 0.95, 1.0), 0.8 * k, kind="halo", reach=1500)
                glob = max(glob, k)
        img *= 1 + glob * 0.6
        cool = ramp(t, 8.6, 9.2)
        if cool > 0:
            img = img * (1 - cool * 0.55) * (1 - cool * 0.4 + cool * 0.4 * np.array([0.7, 0.85, 1.2], np.float32))
        bloom(img, 0.7, 0.5, 40)
        dx, dy = shake_offsets(t, 3, 0.8, 2)
        return warp(img, 1.05 + 0.04 * t / self.dur, 0, tx=dx, ty=dy)

    def overlay(self, img, t, f):
        recreation_tag(img, window(t, 0.2, 12.3))
        chapter_tag(img, t, 0.2, 12.3, "CHAPTER V  ·  DISCLOSURE")
        a = window(t, 0.4, 5.7, 0.5, 0.4)
        if a > 0:
            img = darken_region(img, 260, 120, 1660, 420, 0.6 * a, 160)
            draw_text(img, "“我在履行公务时获悉，存在一个持续数十年的", W / 2, 150, "serif_bold", 52,
                      (255, 255, 255), a, anchor="ct", shadow=0.8)
            draw_text(img, "UAP 坠毁回收与逆向工程项目。”", W / 2, 230, "serif_bold", 52, (255, 255, 255),
                      ramp(t, 0.9, 1.3) * a, anchor="ct", shadow=0.8)
            draw_text(img, "—— 前美国情报官员 大卫·格鲁什，国会宣誓证词", W / 2, 330, "sans", 30, (210, 210, 210),
                      ramp(t, 1.4, 1.8) * a, anchor="ct")
        a = window(t, 5.9, 8.6, 0.25, 0.4)
        if a > 0:
            s = 1.0 + 0.08 * (1 - ease_out((t - 5.9) / 0.3))
            img = darken_region(img, 360, 150, 1560, 360, 0.65 * a, 160)
            draw_text(img, "“一些回收物中……带有生物遗骸。”", W / 2, 200, "serif_black", int(66 * s), (255, 255, 255),
                      a, anchor="ct", shadow=0.8)
            draw_text(img, "“Biologics came with some of these recoveries.”", W / 2, 310, "sans", 30,
                      (210, 210, 210), a, anchor="ct")
        a = window(t, 9.1, 12.4, 0.4, 0.4)
        if a > 0:
            img = darken_region(img, 300, 140, 1620, 430, 0.7 * a, 160)
            draw_text(img, "2024 年 3 月  ·  五角大楼 AARO 历史记录报告", W / 2, 150, "sans_bold", 34, (150, 200, 255), a,
                      anchor="ct", tracking=0.06)
            draw_text(img, "“没有证据表明，任何 UAP 目击", W / 2, 220, "serif_bold", 54, (255, 255, 255),
                      ramp(t, 9.4, 9.8) * a, anchor="ct", shadow=0.8)
            draw_text(img, "代表着外星技术。”", W / 2, 300, "serif_bold", 54, (255, 255, 255), ramp(t, 9.6, 10.0) * a,
                      anchor="ct", shadow=0.8)
        tag = "2023 年 7 月 26 日  ·  美国众议院 UAP 听证会" if t < 8.8 else "官方的另一种声音"
        caption(img, t, 0.4 if t < 8.8 else 8.8, 8.8 if t < 8.8 else 12.4, tag, None, (), bottom=H - 90)
        return img


# ----------------------------------------------------------------------------- S24 Hellfire

class Hellfire(Scene):
    dur = 7.5
    grain = 0.02
    vig = 0.35
    HIT = 4.0

    def setup(self):
        self.v = FlirView("hellfire", 241)

    def orb(self, t):
        return 0.62 - 0.035 * t, 0.46 + 0.01 * math.sin(t * 2)

    def render(self, t, f):
        ox, oy = self.orb(t)
        o = dict(x=ox, y=oy, size=7, polarity=1, kind="blob", aura=1.6)
        s = self.v.sensor(t, o)
        if 2.9 < t < self.HIT:
            u = (t - 2.9) / (self.HIT - 2.9)
            mx, my = lerp(1.1, ox, u), lerp(0.85, oy, u)
            sw, sh = self.v.sw, self.v.sh
            for j in range(12):
                uu = max(0, u - j * 0.02)
                x, y = lerp(1.1, ox, uu) * sw, lerp(0.85, oy, uu) * sh
                cv2.circle(s, (int(x), int(y)), 2, 0.6 * (1 - j / 12) + 0.3, -1, cv2.LINE_AA)
            cv2.circle(s, (int(mx * sw), int(my * sh)), 4, 1.0, -1, cv2.LINE_AA)
        if self.HIT <= t < self.HIT + 0.6:
            k = 1 - (t - self.HIT) / 0.6
            yy, xx = np.mgrid[0:self.v.sh, 0:self.v.sw].astype(np.float32)
            d2 = (xx - ox * self.v.sw) ** 2 + (yy - oy * self.v.sh) ** 2
            s += np.exp(-d2 / (20 + 400 * (1 - k)) ** 2 * 4) * 0.8 * k
            rng = np.random.default_rng(int(t * 100))
            for _ in range(10):
                a = rng.uniform(0, 6.28)
                r = (1 - k) * rng.uniform(20, 80)
                cv2.circle(s, (int(ox * self.v.sw + math.cos(a) * r), int(oy * self.v.sh + math.sin(a) * r)), 1,
                           0.9 * k, -1)
        img = self.v.finish(s, f, noise=0.035)
        gx, gy = ox, oy
        img = self.v.hud(img, t, f, label_mode="IR", zoom="MFOV", polarity="WHT", gate=(gx, gy, 45, 45), bank=0,
                         heading=110 + t, rng_nm=None, alt=12500, mach=0.28, az=None, el=None, extra=("MQ-9",))
        if self.HIT <= t < self.HIT + 0.5:
            dx, dy = shake_offsets(t, 14 * (1 - (t - self.HIT) / 0.5), 30)
            img = warp(img, 1.0, 0, tx=dx, ty=dy)
        return img

    def overlay(self, img, t, f):
        recreation_tag(img, window(t, 0.2, 7.3), "情景再现 · MQ-9 无人机红外画面")
        caption(img, t, 0.4, 7.4, "2024 年 10 月  ·  也门近海",
                "导弹命中之后，它继续飞行",
                ("2025 年 9 月，美国国会听证会上公开了这段视频：",
                 "一枚“地狱火”导弹击中发光球体，而球体似乎安然无恙地继续前进。"),
                bottom=H - 150, title_size=48, scrim=0.8)
        return img
