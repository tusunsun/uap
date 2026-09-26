"""Act I: cold open, hook, and ancient records (74 BC, Song dynasty, 1561)."""
import math

import cv2
import numpy as np

from core import (H, W, FPS, add_ellipse_glow, add_glow, blank, bloom, caption, chapter_card, chapter_tag, draw_text,
                  ease_in, ease_io, ease_out, fbm, glitch, grid, hash01, lerp, noise1d, ramp, recreation_tag,
                  ridge_mask, smooth, star_layer, subtitle, text_size, warp, window, clamp01, shake_offsets,
                  darken_region)
from elements import FlirView, hatch, ink_hatch, milky_way, paper
from scene import Scene


# ----------------------------------------------------------------------------- S01

class ColdOpen(Scene):
    dur = 5.0
    grain = 0.03
    vig = 0.0

    def render(self, t, f):
        img = blank((0.0, 0.0, 0.0))
        rng = np.random.default_rng(f)
        for _ in range(3):
            y = int(rng.integers(0, H))
            img[y:y + 1] += 0.015 * rng.random()
        return img

    def overlay(self, img, t, f):
        l1 = "本片内容取材自真实历史文献、政府解密档案与新闻报道"
        l2 = "BASED ON HISTORICAL RECORDS, DECLASSIFIED FILES AND NEWS REPORTS"
        out = 1 - ramp(t, 4.2, 4.9)
        n1 = int(clamp01((t - 0.5) / 1.9) * len(l1))
        n2 = int(clamp01((t - 2.1) / 1.6) * len(l2))
        tw1 = text_size(l1, "sans", 36, 0.08)
        x1 = W / 2 - tw1 / 2
        if n1 > 0:
            draw_text(img, l1[:n1], x1, H / 2 - 50, "sans", 36, (235, 235, 235), out, tracking=0.08)
        tw2 = text_size(l2, "mono", 20, 0.12)
        if n2 > 0:
            draw_text(img, l2[:n2], W / 2 - tw2 / 2, H / 2 + 14, "mono", 20, (150, 150, 150), out, tracking=0.12)
        # blinking cursor
        if (int(t * 2.5) % 2 == 0) and t < 4.2:
            if n1 < len(l1):
                cx = x1 + text_size(l1[:n1], "sans", 36, 0.08) + 8
                cy = H / 2 - 44
            else:
                cx = W / 2 - tw2 / 2 + text_size(l2[:n2], "mono", 20, 0.12) + 6
                cy = H / 2 + 16
            img[int(cy):int(cy + 34 if n1 < len(l1) else cy + 22), int(cx):int(cx + 14)] += 0.85 * out
        return img


# ----------------------------------------------------------------------------- S02

class FlirTeaser(Scene):
    dur = 7.5
    grain = 0.02
    vig = 0.35

    def setup(self):
        self.v = FlirView("gimbal", seed=7)

    def obj(self, t):
        zoom = 1.0 + 1.1 * smooth((t - 1.0) / 0.35)
        rot = -8 - 38 * smooth((t - 3.6) / 3.2)
        x = 0.5 + 0.012 * math.sin(t * 1.7) + 0.006 * math.sin(t * 7.3)
        y = 0.5 + 0.01 * math.sin(t * 1.3 + 1) + 0.005 * math.sin(t * 9.1)
        return dict(x=x, y=y, size=17 * zoom, angle=rot, polarity=-1, kind="saucer", aura=1.2), zoom

    def render(self, t, f):
        o, zoom = self.obj(t)
        s = self.v.sensor(t, o, bank=-24)
        img = self.v.finish(s, f)
        gate = (o["x"], o["y"], 70 * zoom, 40 * zoom)
        img = self.v.hud(img, t, f, zoom="NAR" if t > 1.1 else "WFOV", polarity="BHT", gate=gate, bank=-24,
                         heading=348 + t * 0.8, rng_nm=None, alt=25004, mach=0.56, az=-54 + t * 0.4, el=-2,
                         extra=("GIMBAL",) if t > 1.2 else ())
        if t > 7.05:
            img = glitch(img, (t - 7.05) / 0.45, f, 3)
        return img

    def overlay(self, img, t, f):
        recreation_tag(img, window(t, 0.2, 7.3, 0.3, 0.2), "情景再现 · 2015 美国海军“GIMBAL”红外画面")
        subtitle(img, t, 1.6, 4.1, "“看那东西，伙计！”", "“Look at that thing, dude!”")
        subtitle(img, t, 4.5, 7.3, "“它在旋转！”", "“It's rotating!”")
        return img


# ----------------------------------------------------------------------------- S03

class Hook(Scene):
    dur = 7.5
    grain = 0.035
    vig = 0.5

    def setup(self):
        self.stars = star_layer(W, H, 5000, seed=21, bright=0.9)

    def render(self, t, f):
        img = blank((0.0, 0.0, 0.0))
        k = ramp(t, 3.8, 7.5) * 0.9
        if k > 0:
            img += self.stars * k
        return img

    def overlay(self, img, t, f):
        if 0.5 <= t < 3.9:
            lt = t - 0.5
            s = 1.25 - 0.25 * ease_out(lt / 0.22, 3) + 0.03 * lt
            a = ramp(lt, 0, 0.06) * (1 - ramp(t, 3.3, 3.9))
            size = int(118 * s)
            draw_text(img, "这 不 是 第 一 次", W / 2, H / 2, "serif_black", size, (255, 255, 255), a,
                      anchor="cm", tracking=0.1, shadow=0.0)
            if lt < 0.25:
                img += (1 - lt / 0.25) * 0.35
        a1 = window(t, 4.0, 7.6, 0.9, 0.5)
        a2 = window(t, 4.8, 7.6, 0.9, 0.5)
        draw_text(img, "人类仰望同一片天空，", W / 2, H / 2 - 40, "serif_light", 54, (240, 240, 240), a1,
                  anchor="cm", tracking=0.15)
        draw_text(img, "已有数千年。", W / 2, H / 2 + 50, "serif_light", 54, (240, 240, 240), a2, anchor="cm",
                  tracking=0.15)
        return img


# ----------------------------------------------------------------------------- S04

class MilkyWay(Scene):
    dur = 10.0
    grain = 0.04
    vig = 0.55
    lbox = 2.39
    xfade = 1.2

    def setup(self):
        self.sky = milky_way(2600, 1500, seed=5, angle=-32)
        # ground silhouette with standing stones
        mask, top = ridge_mask(W, H, 900, 90, 700, seed=3, octaves=5)
        m = (mask * 255).astype(np.uint8)
        rng = np.random.default_rng(4)
        cx0 = 1250
        stones = []
        for k in range(9):
            ang = (k / 9) * math.pi * 2 + 0.3
            sx = cx0 + math.cos(ang) * 330
            depth = 0.75 + 0.25 * math.sin(ang)
            stones.append((depth, sx))
        stones.sort()
        base_y = 905
        for depth, sx in stones:
            hh = 150 * depth
            ww = 42 * depth
            y0 = base_y - hh + (1 - depth) * -20
            pts = np.array([[sx - ww / 2, base_y], [sx - ww / 2 + 4, y0], [sx + ww / 2 - 3, y0 + 3],
                            [sx + ww / 2, base_y]], np.int32)
            cv2.fillPoly(m, [pts], 255, cv2.LINE_AA)
        # a few trilithon lintels
        for a, b in ((0, 1), (3, 4), (6, 7)):
            d1, x1 = stones[a]
            d2, x2 = stones[b]
            if abs(x1 - x2) < 170:
                yy = base_y - 150 * min(d1, d2) - 4
                cv2.rectangle(m, (int(min(x1, x2) - 22), int(yy - 26 * d1)), (int(max(x1, x2) + 22), int(yy + 6)),
                              255, -1, cv2.LINE_AA)
        # lone figure
        fx, fy = 700, int(top[700]) + 2
        cv2.ellipse(m, (fx, fy - 58), (9, 11), 0, 0, 360, 255, -1, cv2.LINE_AA)
        cv2.fillPoly(m, [np.array([[fx - 12, fy - 46], [fx + 12, fy - 46], [fx + 16, fy], [fx - 16, fy]], np.int32)],
                     255, cv2.LINE_AA)
        self.ground = m.astype(np.float32) / 255
        ys = np.arange(H, dtype=np.float32)[:, None]
        self.hglow = (np.exp(-((ys - 880) / 140) ** 2))[..., None] * np.array([0.10, 0.12, 0.16], np.float32)
        self.hglow2 = (np.exp(-((ys - 900) / 50) ** 2))[..., None] * np.array([0.25, 0.12, 0.06], np.float32)

    def render(self, t, f):
        k = t / self.dur
        img = warp(self.sky, 0.88 + 0.07 * ease_io(k), -4 + 3 * k, tx=0, ty=-60 + 30 * k)
        img += self.hglow + self.hglow2
        # meteor
        if 4.3 < t < 5.1:
            u = (t - 4.3) / 0.8
            x0, y0 = 1500 - 700 * u, 250 + 260 * u
            for j in range(18):
                uu = u - j * 0.012
                if uu < 0:
                    break
                add_glow(img, 1500 - 700 * uu, 250 + 260 * uu, 2.2, (1.0, 0.95, 0.85),
                         (1 - j / 18) * 1.6 * math.sin(math.pi * u))
        g = self.ground[..., None]
        img = img * (1 - g) + g * np.array([0.006, 0.008, 0.014], np.float32)
        bloom(img, 0.6, 0.35, 30)
        return img

    def overlay(self, img, t, f):
        chapter_card(img, t, 1.2, 9.4, "CHAPTER  I", "古 老 的 记 载", "ANCIENT  RECORDS", y=H / 2 - 60)
        return img


# ----------------------------------------------------------------------------- S05

def soldier_mask(m, x, y, s, facing, rng, spear=True):
    hh = 46 * s
    cv2.rectangle(m, (int(x - 5 * s), int(y - hh * 0.74)), (int(x + 5 * s), int(y)), 255, -1)
    cv2.circle(m, (int(x), int(y - hh * 0.84)), max(2, int(5 * s)), 255, -1, cv2.LINE_AA)
    cv2.ellipse(m, (int(x + facing * 7 * s), int(y - hh * 0.45)), (max(2, int(6 * s)), max(3, int(12 * s))), 0, 0,
                360, 255, -1, cv2.LINE_AA)
    if spear:
        lean = facing * rng.uniform(2, 10) * s
        cv2.line(m, (int(x - facing * 6 * s), int(y + 2)), (int(x - facing * 6 * s + lean), int(y - hh * 1.75)), 255,
                 max(1, int(1.6 * s)), cv2.LINE_AA)


class Plutarch(Scene):
    dur = 10.0
    grain = 0.05
    vig = 0.5
    lbox = 2.39
    flash_in = 0.0

    IMPACT = 5.0

    def setup(self):
        ys = np.arange(H, dtype=np.float32)
        stops_y = [0, 140, 380, 540, 630, 660]
        cols = np.array([[0.02, 0.03, 0.07], [0.04, 0.05, 0.11], [0.20, 0.12, 0.16], [0.58, 0.28, 0.17],
                         [0.98, 0.60, 0.30], [1.0, 0.70, 0.40]], np.float32)
        sky = np.stack([np.interp(ys, stops_y, cols[:, c]) for c in range(3)], -1).astype(np.float32)
        sky = np.broadcast_to(sky[:, None, :], (H, W, 3)).copy()
        n = cv2.resize(fbm(1600, 360, 180, 6, 0.55, 12), (W, 720))
        n2 = cv2.resize(fbm(1600, 360, 50, 4, 0.6, 13), (W, 720))
        dens = np.clip((n - 0.42) * 2.4, 0, 1)
        yy = np.arange(720, dtype=np.float32)[:, None]
        dens *= np.clip((650 - yy) / 120, 0, 1)
        light = np.clip((yy - 200) / 430, 0, 1) ** 1.6 * (0.5 + 0.5 * n2)
        ccol = np.array([0.08, 0.06, 0.09], np.float32) * (1 - light[..., None]) + \
            np.array([1.0, 0.50, 0.25], np.float32) * light[..., None]
        sky[:720] = sky[:720] * (1 - dens[..., None]) + ccol * dens[..., None]
        hills, _ = ridge_mask(W, H, 650, 50, 420, seed=8, octaves=5)
        sky = sky * (1 - hills[..., None] * 0.85) + hills[..., None] * 0.85 * np.array([0.22, 0.13, 0.13], np.float32)
        gy = np.clip((ys - 655) / 300, 0, 1)[:, None, None]
        groundcol = np.array([0.24, 0.13, 0.07], np.float32) * (1 - gy) + np.array([0.03, 0.02, 0.02], np.float32) * gy
        gmask = np.clip(ys - 656, 0, 1)[:, None, None]
        tex = fbm(W, H, 60, 4, 0.6, 14)[..., None]
        sky = sky * (1 - gmask) + groundcol * (0.8 + 0.4 * tex) * gmask
        self.bg = sky.astype(np.float32)
        # armies
        m = np.zeros((H, W), np.uint8)
        rng = np.random.default_rng(3)
        for r in range(5):
            s = 0.55 + r * 0.32
            yb = 668 + r * 52
            for side, (xa, xb, facing) in enumerate(((-40, 790 - r * 60, 1), (1130 + r * 60, 1960, -1))):
                n_ = int((xb - xa) / (17 * s))
                for i in range(n_):
                    x = xa + (i + rng.random() * 0.6) * (xb - xa) / n_
                    soldier_mask(m, x, yb + rng.uniform(-3, 3), s, facing, rng)
                for i in range(3):
                    bx = xa + rng.uniform(0.1, 0.9) * (xb - xa)
                    top = yb - 46 * s * 2.6
                    cv2.line(m, (int(bx), int(yb)), (int(bx), int(top)), 255, max(1, int(2 * s)), cv2.LINE_AA)
                    cv2.rectangle(m, (int(bx), int(top)), (int(bx + facing * 22 * s), int(top + 26 * s)), 255, -1)
        self.army = cv2.GaussianBlur(m, (0, 0), 0.6).astype(np.float32) / 255
        self.canvas_tex = (0.97 + 0.03 * fbm(W, H, 5, 2, 0.5, 30))[..., None]

    def obj_pos(self, t):
        u = clamp01((t - 0.6) / (self.IMPACT - 0.6))
        u = u ** 1.7
        return lerp(1130, 962, u), lerp(40, 690, u), u

    def render(self, t, f):
        img = self.bg.copy()
        # sky bursting open
        burst = window(t, 0.3, 2.6, 0.2, 1.8)
        if burst > 0:
            img[:700] += burst * 0.25 * np.array([1.0, 0.55, 0.3], np.float32) * \
                np.exp(-((np.arange(700, dtype=np.float32) - 60) / 320) ** 2)[:, None, None]
            add_glow(img, 1140, 40, 160, (1.0, 0.7, 0.45), burst * 1.2, kind="halo")
        x, y, u = self.obj_pos(t)
        if t < self.IMPACT:
            if t > 0.6:
                # trail
                for j in range(40):
                    tt = t - j * 0.035
                    if tt < 0.6:
                        break
                    px, py, _ = self.obj_pos(tt)
                    fall = 1 - j / 40
                    add_glow(img, px, py, 10 + j * 1.2, (1.0, 0.48, 0.18), 0.55 * fall ** 1.5)
                ang = math.atan2(690 - 40, 962 - 1130)
                add_ellipse_glow(img, x, y, 30, 17, ang, (0.95, 0.97, 1.0), 2.4)
                add_ellipse_glow(img, x, y, 70, 40, ang, (1.0, 0.7, 0.4), 0.9)
                add_glow(img, x, y, 90, (1.0, 0.65, 0.35), 0.6 + u, kind="halo")
                add_glow(img, 960, 700, 300, (1.0, 0.55, 0.25), 0.35 * u ** 2, kind="halo")
        else:
            lt = t - self.IMPACT
            pulse = 0.85 + 0.15 * math.sin(lt * 5)
            add_glow(img, 962, 684, 26, (0.95, 0.97, 1.0), 2.6 * pulse)
            add_ellipse_glow(img, 962, 690, 120, 30, 0, (1.0, 0.75, 0.45), 1.3 * pulse)
            add_glow(img, 962, 684, 120, (1.0, 0.7, 0.4), 1.3 * pulse, kind="halo")
            # dust wave along the ground
            ring = 80 + lt * 520
            k = math.exp(-lt * 1.3)
            add_ellipse_glow(img, 962, 700, ring, ring * 0.12, 0, (0.9, 0.6, 0.35), 0.8 * k, reach=1.6)
            if lt < 0.45:
                img += (1 - lt / 0.45) ** 2 * 1.4 * np.array([1.0, 0.92, 0.8], np.float32)
        a = self.army[..., None]
        img = img * (1 - a) + a * np.array([0.012, 0.008, 0.01], np.float32)
        bloom(img, 0.7, 0.55, 50)
        img *= self.canvas_tex
        # camera: slow push + impact shake
        amp = 0.0
        if t > self.IMPACT:
            amp = 22 * math.exp(-(t - self.IMPACT) * 2.2)
        dx, dy = shake_offsets(t, amp, 17)
        img = warp(img, 1.0 + 0.045 * t / self.dur, 0, cx=962, cy=640, tx=dx, ty=150 + dy)
        return img

    def overlay(self, img, t, f):
        recreation_tag(img, window(t, 0.5, 9.7), y=160)
        chapter_tag(img, t, 0.4, 9.7, "CHAPTER I  ·  ANCIENT RECORDS", y=160)
        caption(img, t, 1.4, 9.8, "公元前 74 年  ·  小亚细亚",
                "“天空骤然裂开，一个巨大的燃烧物体坠落在两军之间”",
                ("罗马统帅卢库鲁斯与本都国王米特拉达梯的大军即将交战，",
                 "那物体“形如酒瓮，色如熔银”。两军惊骇，各自罢兵。"),
                "—— 普鲁塔克《希腊罗马名人传 · 卢库鲁斯传》", bottom=520, title_size=44, body_size=29)
        return img


# ----------------------------------------------------------------------------- S06

class YangzhouPearl(Scene):
    dur = 12.5
    grain = 0.03
    vig = 0.35

    COLS = ["嘉祐中扬州有一珠甚大天晦多见",
            "初微开其房光自吻中出如横一金线",
            "俄顷忽张壳其大如半席壳中白光如银",
            "珠大如拳烂然不可正视",
            "倏然远去其行如飞浮于波中杳杳如日"]
    STARTS = [0.2, 1.3, 3.2, 5.0, 8.5]
    PX, PY = 780, 600

    def setup(self):
        pap = paper(W, H, base=(0.90, 0.86, 0.77), seed=41, stains=0.5)
        xs = np.arange(W, dtype=np.float32)
        ys = np.arange(H, dtype=np.float32)[:, None]
        ink = np.zeros((H, W), np.float32)
        tex = fbm(W, H, 18, 4, 0.6, 42)
        streak = cv2.resize(fbm(W // 4, H, 40, 4, 0.6, 43), (W, H))
        mist = fbm(W, H, 220, 4, 0.5, 44)

        def layer(base, amp, scale, seed, env, strength, fade, oct_=6):
            n = noise1d(W, scale, oct_, seed, 0.55)
            n2 = noise1d(W, scale / 5, 3, seed + 1, 0.5)
            top = base - amp * env * (0.65 * n + 0.35 * n2)
            dy = ys - top[None, :]
            m = np.clip(dy, 0, 1)
            val = m * (0.25 + 0.75 * np.exp(-np.maximum(dy, 0) / fade))
            val *= 0.55 + 0.45 * tex * (0.6 + 0.4 * streak)
            return val * strength, top

        env_far = np.exp(-((xs - 420) / 520) ** 2) * 1.1 + np.exp(-((xs - 1300) / 300) ** 2) * 0.6 + 0.2
        v, _ = layer(620, 430, 380, 51, env_far, 0.30, 110)
        v *= np.clip(1 - (mist - 0.3) * 1.2 * np.clip((ys - 380) / 240, 0, 1), 0, 1)
        ink = ink + v * (1 - ink)
        env_mid = np.exp(-((xs - 250) / 380) ** 2) * 1.0 + np.exp(-((xs - 1050) / 260) ** 2) * 0.45 + 0.1
        v, _ = layer(700, 300, 300, 52, env_mid, 0.55, 80)
        v *= np.clip(1 - (mist - 0.25) * 1.4 * np.clip((ys - 560) / 160, 0, 1), 0, 1)
        ink = ink + v * (1 - ink)
        env_near = np.clip(1 - (xs - 60) / 700, 0, 1) ** 1.2
        v, top_near = layer(1080, 330, 240, 53, env_near, 0.9, 70)
        ink = ink + v * (1 - ink)
        m = np.zeros((H, W), np.uint8)
        rng = np.random.default_rng(55)
        for _ in range(900):
            x = rng.uniform(0, 640)
            y = top_near[int(x)] + rng.uniform(-8, 60)
            cv2.circle(m, (int(x), int(y)), int(rng.uniform(2, 7)), int(rng.uniform(120, 255)), -1, cv2.LINE_AA)
        for _ in range(26):
            x0 = rng.uniform(80, 1700)
            y0 = rng.uniform(690, 1000)
            L = rng.uniform(40, 260)
            cv2.line(m, (int(x0), int(y0)), (int(x0 + L), int(y0 + rng.uniform(-2, 2))), int(rng.uniform(40, 90)), 2,
                     cv2.LINE_AA)
        # reeds bottom right
        for _ in range(120):
            x0 = rng.uniform(1560, 1940)
            y0 = rng.uniform(1010, 1090)
            hgt = rng.uniform(60, 170)
            bend = rng.uniform(-40, 30)
            pts = np.array([[x0, y0], [x0 + bend * 0.3, y0 - hgt * 0.5], [x0 + bend, y0 - hgt]], np.int32)
            cv2.polylines(m, [pts], False, int(rng.uniform(150, 240)), int(rng.uniform(1, 3)), cv2.LINE_AA)
        # boat with a figure
        bx, by = 1240, 820
        cv2.ellipse(m, (bx, by), (70, 12), 0, 0, 180, 235, -1, cv2.LINE_AA)
        cv2.circle(m, (bx - 12, by - 30), 7, 235, -1, cv2.LINE_AA)
        cv2.fillPoly(m, [np.array([[bx - 30, by - 36], [bx + 6, by - 36], [bx - 12, by - 46]], np.int32)], 235,
                     cv2.LINE_AA)
        cv2.rectangle(m, (bx - 18, by - 24), (bx - 6, by - 2), 235, -1)
        cv2.line(m, (bx + 20, by + 30), (bx + 60, by - 90), 200, 2, cv2.LINE_AA)
        mm = cv2.GaussianBlur(m, (0, 0), 0.8).astype(np.float32) / 255
        ink = ink + mm * (1 - ink)
        inkcol = np.array([0.07, 0.065, 0.07], np.float32)
        base = pap * (1 - ink[..., None]) + inkcol * ink[..., None]
        # calligraphy columns are added to the painting (so they receive the same lighting)
        self.base = base.astype(np.float32)
        self.ripple = (0.6 + 0.4 * np.sin(np.arange(700, dtype=np.float32)[:, None] * 0.55 +
                                            fbm(600, 700, 30, 3, 0.5, 60) * 8)).astype(np.float32)
        yy = np.arange(H, dtype=np.float32)[:, None, None]
        self.redsky = np.clip(1 - yy / 650, 0, 1) ** 1.5

    def pearl(self, t):
        """returns x, y, opening(0..1), line(0..1), scale"""
        x, y, sc = self.PX, self.PY + 5 * math.sin(t * 1.4), 1.0
        line = ramp(t, 1.2, 3.0)
        opening = ramp(t, 3.3, 4.6)
        if t > 8.8:
            u = ease_in((t - 8.8) / 1.9, 2.2)
            x = lerp(self.PX, 1650, min(u, 1.2))
            y = lerp(self.PY, 572, min(u, 1.0))
            sc = lerp(1.0, 0.18, min(u, 1.0))
        return x, y, opening, line, sc

    def render(self, t, f):
        img = self.base.copy()
        # calligraphy
        for ci, (col, st) in enumerate(zip(self.COLS, self.STARTS)):
            cx = 1800 - ci * 62
            for k, ch in enumerate(col):
                a = ramp(t, st + k * 0.075, st + k * 0.075 + 0.25)
                if a <= 0:
                    break
                draw_text(img, ch, cx, 92 + k * 45, "kai_bold", 40, (26, 22, 20), a * 0.92, anchor="ct")
        if t > 9.8:
            a = ramp(t, 9.8, 10.1)
            sx, sy = 1500, 830
            img[sy:sy + 76, sx:sx + 76] = img[sy:sy + 76, sx:sx + 76] * (1 - 0.9 * a) + \
                np.array([0.70, 0.12, 0.10], np.float32) * 0.9 * a
            for i, ch in enumerate("梦溪笔谈"):
                draw_text(img, ch, sx + 57 - (i // 2) * 38, sy + 4 + (i % 2) * 36, "kai_bold", 32,
                          (245, 225, 210), a * 0.95, anchor="ct")
        x, y, opening, line, sc = self.pearl(t)
        inten = (0.25 * line + 1.0 * opening) * (1 - 0.8 * ramp(t, 9.0, 11.0))
        # night ambience, lit by the pearl
        amb = np.array([0.50, 0.50, 0.58], np.float32)
        light = np.zeros((H // 4, W // 4, 3), np.float32)
        add_glow(light, x / 4, y / 4, 34 * (0.4 + 0.6 * sc), (1.0, 0.80, 0.50), inten * 0.75, kind="halo",
                 reach=600)
        light = cv2.resize(light, (W, H), interpolation=cv2.INTER_LINEAR)
        red = ramp(t, 5.0, 7.0) * (1 - ramp(t, 9.5, 11.5))
        img *= amb + light
        if red > 0:
            img *= 1 + self.redsky * (np.array([0.45, -0.05, -0.15], np.float32) * red)
        # the pearl itself
        if line > 0 and opening < 1:
            add_ellipse_glow(img, x, y, 10 + 150 * line * (1 - opening * 0.5), 3 + 20 * opening, 0,
                             (1.0, 0.78, 0.35), 2.2 * line)
        if opening > 0:
            r = 17 * sc
            add_ellipse_glow(img, x, y, (58 + 20 * opening) * sc, 26 * opening * sc + 2, 0, (0.92, 0.94, 1.0),
                             0.8 * opening)
            add_glow(img, x, y, r, (1.0, 1.0, 1.0), 2.4 * opening)
            add_glow(img, x, y, r * 2.5, (1.0, 0.85, 0.55), 0.45 * opening, kind="halo")
            # reflection on the lake
            ry = 2 * 735 - y
            h0, h1 = int(ry - 120 * sc), int(ry + 160 * sc)
            x0, x1 = int(x - 60 * sc), int(x + 60 * sc)
            if h1 > h0 and x1 > x0 and h0 > 0:
                reg = img[h0:min(H, h1), x0:x1]
                yy = np.arange(reg.shape[0], dtype=np.float32)[:, None] - (ry - h0)
                xx = np.arange(reg.shape[1], dtype=np.float32)[None, :] - (x - x0)
                g = np.exp(-(xx / (22 * sc + 2)) ** 2 - (yy / (110 * sc + 4)) ** 2)
                rp = self.ripple[:reg.shape[0], :reg.shape[1]]
                reg += (g * rp)[..., None] * np.array([1.0, 0.85, 0.6], np.float32) * 0.9 * opening
        if t > 8.8:
            for j in range(1, 16):
                xx, yy, _, _, ss = self.pearl(t - j * 0.03)
                add_glow(img, xx, yy, 14 * ss + 3, (1.0, 0.85, 0.55), 0.6 * (1 - j / 16) * opening)
        bloom(img, 0.9, 0.35, 36)
        return warp(img, 1.0 + 0.05 * t / self.dur, 0, cx=900, cy=560, tx=0, ty=0)

    def overlay(self, img, t, f):
        chapter_tag(img, t, 0.3, 12.2, "CHAPTER I  ·  ANCIENT RECORDS", y=48)
        caption(img, t, 1.0, 12.3, "北宋 · 嘉祐年间（1056—1063） · 扬州",
                "沈括《梦溪笔谈》：湖上“明珠”",
                ("一个会发光的“大珠”在扬州湖泊间出没十余年，“居民行人常常见之”。",
                 "它开壳放光，照得林木皆有影子，而后“其行如飞”，杳然远去。"),
                "—— 沈括《梦溪笔谈 · 卷二十一 · 异事》", bottom=H - 90, title_size=46, body_size=29)
        return img


# ----------------------------------------------------------------------------- S07

class Nuremberg(Scene):
    dur = 10.0
    grain = 0.04
    vig = 0.5

    def setup(self):
        rng = np.random.default_rng(71)
        base = paper(W, H, base=(0.88, 0.80, 0.62), seed=72, stains=0.8, fibers=0.07)
        hh = hatch(H, W, 7.0, 0)
        hd = hatch(H, W, 5.5, 45)
        self.hd = hd
        ys = np.arange(H, dtype=np.float32)[:, None]
        n = fbm(W, H, 300, 4, 0.5, 73)
        tone = 0.12 + 0.42 * np.clip(1 - ys / 800, 0, 1) ** 1.4 + 0.12 * (n - 0.5)
        sky_ink = ink_hatch(tone, hh, 0.18) * (ys < 840)
        img = base * (1 - 0.85 * sky_ink[..., None])
        # sun with blood-red crescent
        m = np.zeros((H, W), np.uint8)
        sx, sy, sr = 1180, 300, 92
        cv2.circle(m, (sx, sy), sr + 6, 255, -1, cv2.LINE_AA)
        clear = m.astype(np.float32) / 255
        img = img * (1 - clear[..., None]) + base * clear[..., None] * np.array([1.0, 0.95, 0.75], np.float32)
        cres = np.zeros((H, W), np.uint8)
        cv2.circle(cres, (sx, sy), sr, 255, -1, cv2.LINE_AA)
        cv2.circle(cres, (sx + 30, sy - 20), sr - 6, 0, -1, cv2.LINE_AA)
        c = cres.astype(np.float32) / 255
        img = img * (1 - c[..., None] * 0.8) + c[..., None] * 0.8 * np.array([0.62, 0.10, 0.08], np.float32)
        ol = np.zeros((H, W), np.uint8)
        cv2.circle(ol, (sx, sy), sr, 255, 3, cv2.LINE_AA)
        for k in range(24):
            a = k / 24 * 2 * math.pi
            cv2.line(ol, (int(sx + math.cos(a) * (sr + 14)), int(sy + math.sin(a) * (sr + 14))),
                     (int(sx + math.cos(a) * (sr + 44)), int(sy + math.sin(a) * (sr + 44))), 255, 3, cv2.LINE_AA)
        # hill on the left and town
        fill = np.zeros((H, W), np.uint8)
        pts = [[0, 740], [120, 720], [260, 760], [420, 840], [560, 900], [560, H], [0, H]]
        cv2.fillPoly(fill, [np.array(pts, np.int32)], 255, cv2.LINE_AA)
        cv2.polylines(ol, [np.array(pts[:5], np.int32)], False, 255, 4, cv2.LINE_AA)
        hillm = fill.astype(np.float32) / 255
        x = 520
        shade = np.zeros((H, W), np.uint8)
        while x < W + 40:
            bw = int(rng.uniform(60, 140))
            bh = int(rng.uniform(70, 170))
            gy = 900 + int(rng.uniform(-12, 12))
            tower = rng.random() < 0.18
            if tower:
                bw = int(rng.uniform(40, 60))
                bh = int(rng.uniform(150, 210))
            body = [[x, H], [x, gy - bh], [x + bw, gy - bh], [x + bw, H]]
            roof_h = bh * (0.9 if tower else 0.55)
            roof = [[x - 6, gy - bh], [x + bw / 2, gy - bh - roof_h], [x + bw + 6, gy - bh]]
            for p in (body, roof):
                cv2.fillPoly(fill, [np.array(p, np.int32)], 255, cv2.LINE_AA)
                cv2.polylines(ol, [np.array(p, np.int32)], True, 255, 3, cv2.LINE_AA)
            cv2.fillPoly(shade, [np.array([[x + bw * 0.55, gy - bh], [x + bw, gy - bh], [x + bw, H],
                                           [x + bw * 0.55, H]], np.int32)], 255)
            cv2.fillPoly(shade, [np.array([[x + bw / 2, gy - bh - roof_h], [x + bw + 6, gy - bh],
                                           [x + bw / 2, gy - bh]], np.int32)], 255)
            for wy in range(int(gy - bh + 20), gy - 20, 38):
                for wx in range(x + 10, x + bw - 18, 26):
                    cv2.rectangle(ol, (wx, wy), (wx + 10, wy + 16), 255, -1)
            x += bw + int(rng.uniform(-10, 6))
        fm = fill.astype(np.float32) / 255
        img = img * (1 - fm[..., None]) + base * fm[..., None]
        hv = hatch(H, W, 5.0, 90)
        sm = shade.astype(np.float32) / 255 * (ys > 600)
        img *= 1 - 0.75 * (ink_hatch(0.55, hv, 0.2) * sm)[..., None]
        img *= 1 - 0.7 * (ink_hatch(0.45 + 0.2 * (ys - 740) / 340, hd, 0.2) * hillm)[..., None]
        # the black spear lying on the hill, with smoke
        sp = np.array([[80, 790], [470, 870], [470, 885], [60, 812]], np.int32)
        cv2.fillPoly(ol, [sp], 255, cv2.LINE_AA)
        olm = cv2.GaussianBlur(ol, (0, 0), 0.7).astype(np.float32) / 255
        img *= 1 - 0.9 * olm[..., None]
        self.base = img.astype(np.float32)
        self.smoke_tex = fbm(W, H, 40, 4, 0.6, 74)
        # objects
        objs = []
        palette = [(0.66, 0.14, 0.10), (0.24, 0.32, 0.50), (0.09, 0.08, 0.07), (0.66, 0.14, 0.10),
                   (0.24, 0.32, 0.50)]
        for i in range(44):
            objs.append(dict(kind="ball", x=rng.uniform(140, 1800), y=rng.uniform(150, 640), r=rng.uniform(13, 26),
                             col=palette[i % 5], ph=rng.uniform(0, 6.28), w=rng.uniform(0.4, 1.3),
                             A=rng.uniform(15, 70), t0=rng.uniform(0.2, 3.8)))
        for i in range(6):
            objs.append(dict(kind="cross", x=rng.uniform(250, 1700), y=rng.uniform(180, 600), r=rng.uniform(30, 44),
                             col=palette[i % 2], ph=rng.uniform(0, 6.28), w=rng.uniform(0.3, 0.9),
                             A=rng.uniform(10, 40), t0=rng.uniform(0.5, 3.0)))
        for i, (tx, ty, ang) in enumerate(((430, 330, -18), (1540, 520, 24))):
            objs.append(dict(kind="tube", x=tx, y=ty, r=150, ang=ang, col=(0.20, 0.28, 0.44), ph=i * 2.0, w=0.5,
                             A=20, t0=0.8 + i * 0.9))
        self.objs = objs
        fall_ids = rng.choice(44, 12, replace=False)
        for k in fall_ids:
            objs[k]["fall"] = rng.uniform(6.2, 8.6)

    def render(self, t, f):
        img = self.base.copy()
        col = np.zeros((H, W, 3), np.uint8)
        alpha = np.zeros((H, W), np.uint8)
        ink = np.zeros((H, W), np.uint8)
        shade = np.zeros((H, W), np.uint8)
        smoke = np.zeros((H // 2, W // 2), np.uint8)
        for o in self.objs:
            s = ease_out((t - o["t0"]) / 0.5, 3)
            if s <= 0:
                continue
            x = o["x"] + o["A"] * math.sin(o["w"] * t + o["ph"])
            y = o["y"] + o["A"] * 0.6 * math.cos(o["w"] * 1.3 * t + o["ph"] * 1.7)
            jit = 2 * math.sin(t * 13 + o["ph"] * 5)
            x += jit
            if "fall" in o and t > o["fall"]:
                u = (t - o["fall"])
                y += 180 * u * u
                for j in range(1, 12):
                    yy = y - j * 26 - 180 * u * u * 0.2
                    cv2.circle(smoke, (int(x / 2 + math.sin(j + t) * 4), int(yy / 2)), int(5 + j * 1.5), 255, -1,
                               cv2.LINE_AA)
                if y > 860:
                    continue
            c255 = tuple(int(v * 255) for v in o["col"])
            if o["kind"] == "ball":
                r = int(o["r"] * s)
                if r < 2:
                    continue
                cv2.circle(col, (int(x), int(y)), r, c255, -1, cv2.LINE_AA)
                cv2.circle(alpha, (int(x), int(y)), r, 255, -1, cv2.LINE_AA)
                cv2.circle(shade, (int(x), int(y)), r, 255, -1, cv2.LINE_AA)
                cv2.circle(shade, (int(x - r * 0.3), int(y - r * 0.3)), int(r * 0.85), 0, -1, cv2.LINE_AA)
                cv2.circle(ink, (int(x), int(y)), r, 255, 3, cv2.LINE_AA)
            elif o["kind"] == "cross":
                r = o["r"] * s
                a = 0.3 * math.sin(t * 0.8 + o["ph"])
                ca, sa = math.cos(a), math.sin(a)
                wdt = r * 0.28
                pts = []
                for (px, py) in ((-wdt, -r), (wdt, -r), (wdt, -wdt), (r, -wdt), (r, wdt), (wdt, wdt), (wdt, r),
                                 (-wdt, r), (-wdt, wdt), (-r, wdt), (-r, -wdt), (-wdt, -wdt)):
                    pts.append([x + px * ca - py * sa, y + px * sa + py * ca])
                p = np.array(pts, np.int32)
                cv2.fillPoly(col, [p], c255, cv2.LINE_AA)
                cv2.fillPoly(alpha, [p], 255, cv2.LINE_AA)
                cv2.polylines(ink, [p], True, 255, 3, cv2.LINE_AA)
            else:
                L = o["r"] * s
                a = math.radians(o["ang"] + 4 * math.sin(t * 0.6 + o["ph"]))
                ca, sa = math.cos(a), math.sin(a)
                wd = 30
                p = np.array([[x - ca * L - sa * wd, y - sa * L + ca * wd], [x + ca * L - sa * wd, y + sa * L + ca * wd],
                              [x + ca * L + sa * wd, y + sa * L - ca * wd], [x - ca * L + sa * wd, y - sa * L - ca * wd]],
                             np.int32)
                cv2.fillPoly(col, [p], (220, 205, 170), cv2.LINE_AA)
                cv2.fillPoly(alpha, [p], 255, cv2.LINE_AA)
                cv2.polylines(ink, [p], True, 255, 4, cv2.LINE_AA)
                for k in range(-2, 3):
                    bx, by = x + ca * L * k * 0.36, y + sa * L * k * 0.36
                    cv2.circle(col, (int(bx), int(by)), 20, (30, 28, 25), -1, cv2.LINE_AA)
                    cv2.circle(ink, (int(bx), int(by)), 20, 255, 2, cv2.LINE_AA)
        a = alpha.astype(np.float32)[..., None] / 255
        c = col.astype(np.float32) / 255
        wash = self.base * 0 + c * 1.15
        img = img * (1 - a) + (wash * np.array([0.98, 0.95, 0.85], np.float32)) * a
        sh = shade.astype(np.float32) / 255
        img *= 1 - 0.75 * (ink_hatch(0.55, self.hd, 0.2) * sh)[..., None]
        img *= 1 - 0.9 * (ink.astype(np.float32) / 255)[..., None]
        sm = cv2.resize(cv2.GaussianBlur(smoke, (0, 0), 3), (W, H)).astype(np.float32) / 255
        if sm.max() > 0:
            img *= 1 - 0.45 * (sm * (0.5 + 0.5 * self.smoke_tex))[..., None]
        k = t / self.dur
        return warp(img, 1.12 - 0.08 * ease_io(k), 0, cx=W / 2 + 120 - 240 * k, cy=H / 2 + 40)

    def overlay(self, img, t, f):
        recreation_tag(img, window(t, 0.3, 9.8))
        a = window(t, 0.3, 9.8)
        img = darken_region(img, 700, 40, 1220, 100, 0.25 * a, 80)
        draw_text(img, "Nürnberg  ·  14. April  1561", W / 2, 44, "gloock", 42, (240, 228, 200), a * 0.9,
                  anchor="ct", tracking=0.08)
        caption(img, t, 1.2, 9.8, "1561 年 4 月 14 日  ·  德意志 纽伦堡",
                "黎明时分，天空中上演了一场“空战”",
                ("许多市民目睹球体、十字与筒状物在太阳周围交错飞行，持续约一小时，",
                 "随后仿佛燃烧着坠向城外，在浓烟中消失。"),
                "—— 汉斯·格拉泽 木刻新闻传单（藏于苏黎世中央图书馆）", bottom=H - 90, title_size=46, body_size=29)
        return img
