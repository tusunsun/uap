"""Act VI and finale: interstellar visitors, Mars, flash recap, title, end card."""
import math

import cv2
import numpy as np

from core import (H, W, FPS, add_ellipse_glow, add_glow, blank, bloom, caption, chapter_card, chapter_tag, draw_text,
                  ease_in, ease_io, ease_out, fbm, glitch, grid, hash01, lerp, ramp, recreation_tag, smooth,
                  star_layer, text_size, warp, window, clamp01, shake_offsets, darken_region, desaturate, contrast,
                  chroma_shift, vignette)
from elements import milky_way
from scene import Scene


def au_px(r):
    return 150.0 * np.power(np.maximum(r, 0), 0.6)


# ----------------------------------------------------------------------------- S25 interstellar

class Interstellar(Scene):
    dur = 10.0
    grain = 0.03
    vig = 0.5
    CX, CY = 1080, 520
    TILT = 0.42
    ROT = math.radians(-14)
    PLANETS = [(0.39, (0.8, 0.7, 0.6)), (0.72, (1.0, 0.9, 0.7)), (1.0, (0.5, 0.7, 1.0)), (1.52, (1.0, 0.5, 0.3)),
               (5.2, (1.0, 0.8, 0.6)), (9.5, (1.0, 0.9, 0.7))]

    def setup(self):
        self.stars = star_layer(W, H, 6000, 251, bright=0.7) + milky_way(W, H, 252, angle=20, stars=4000) * 0.35
        m = np.zeros((H * 2, W * 2), np.uint8)
        for a, _ in self.PLANETS:
            r = au_px(a)
            cv2.ellipse(m, (self.CX * 2, self.CY * 2), (int(r * 2), int(r * 2 * self.TILT)), math.degrees(self.ROT),
                        0, 360, 255, 2, cv2.LINE_AA)
        self.orbits = cv2.resize(m, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255

    def proj(self, r, th):
        rp = au_px(r)
        x, y = rp * np.cos(th), rp * np.sin(th) * self.TILT
        c, s = math.cos(self.ROT), math.sin(self.ROT)
        return self.CX + x * c - y * s, self.CY + x * s + y * c

    def hyper(self, e, q, orient, n=400):
        p = q * (1 + e)
        lim = math.acos(-1 / e) * 0.985
        th = np.linspace(-lim, lim, n)
        r = p / (1 + e * np.cos(th))
        r = np.minimum(r, 14)
        return self.proj(r, th + orient)

    def draw_path(self, img, xs, ys, upto, color, width=3):
        n = int(len(xs) * clamp01(upto))
        if n < 2:
            return None
        m = np.zeros((H, W), np.uint8)
        pts = np.stack([xs[:n], ys[:n]], 1).astype(np.int32)
        cv2.polylines(m, [pts], False, 255, width, cv2.LINE_AA)
        mm = m.astype(np.float32) / 255
        img += mm[..., None] * np.array(color, np.float32) * 1.2
        img += cv2.GaussianBlur(mm, (0, 0), 6)[..., None] * np.array(color, np.float32) * 1.5
        return xs[n - 1], ys[n - 1]

    def render(self, t, f):
        img = self.stars.copy()
        img += self.orbits[..., None] * np.array([0.35, 0.55, 0.8], np.float32) * 0.7
        add_glow(img, self.CX, self.CY, 16, (1.0, 0.9, 0.7), 3.0)
        add_glow(img, self.CX, self.CY, 90, (1.0, 0.7, 0.4), 0.6, kind="halo")
        for i, (a, col) in enumerate(self.PLANETS):
            th = t * 0.9 / a ** 1.5 + i * 1.7
            x, y = self.proj(np.array([a]), np.array([th]))
            add_glow(img, float(x[0]), float(y[0]), 4.5, col, 2.2)
        xs, ys = self.hyper(1.20, 0.26, 2.3)
        p1 = self.draw_path(img, xs, ys, (t - 2.2) / 3.2, (1.0, 0.7, 0.25))
        xs2, ys2 = self.hyper(6.1, 1.36, -0.6)
        p2 = self.draw_path(img, xs2, ys2, (t - 5.2) / 3.6, (1.0, 0.25, 0.2))
        self.p1, self.p2 = p1, p2
        for p in (p1, p2):
            if p is not None:
                add_glow(img, p[0], p[1], 6, (1, 1, 1), 3.0)
        bloom(img, 0.7, 0.5, 40)
        if t < 2.4:
            img *= 0.35 + 0.65 * ramp(t, 1.4, 2.4)
        return warp(img, 1.0 + 0.05 * t / self.dur, 0)

    def overlay(self, img, t, f):
        if t < 2.4:
            chapter_card(img, t, 0.0, 2.4, "CHAPTER  VI", "星 际 来 客", "VISITORS  FROM  THE  STARS")
            return img
        chapter_tag(img, t, 2.4, 9.8, "CHAPTER VI  ·  VISITORS FROM THE STARS")
        labels = [(getattr(self, "p1", None), "1I/ʻOUMUAMUA  ·  2017", "首个被确认的星际天体 · 雪茄状 · 非引力加速",
                   (255, 190, 80), 2.4),
                  (getattr(self, "p2", None), "3I/ATLAS  ·  2025", "第三个星际天体 · 速度约 58 公里/秒", (255, 90, 70), 5.4)]
        for p, l1, l2, col, t0 in labels:
            if p is None:
                continue
            a = ramp(t, t0, t0 + 0.4)
            x, y = p[0] * (1 + 0.05 * t / self.dur) - W / 2 * 0.05 * t / self.dur, p[1]
            x = min(x, W - 700)
            draw_text(img, l1, x + 24, y - 60, "plex_bold", 30, col, a, tracking=0.05)
            draw_text(img, l2, x + 24, y - 20, "sans_bold", 26, (230, 230, 230), a)
        caption(img, t, 2.6, 9.8, "2017  ·  2025  ·  来自太阳系之外", "星际来客",
                ("2017 年，人类首次发现穿越太阳系的星际天体“奥陌陌”；2025 年 7 月，第三位访客 3I/ATLAS 现身。",
                 "有学者猜测它们可能是人造探测器——主流观点认为，它们是来自其他恒星系的彗星或小天体。"),
                bottom=H - 80, title_size=54, body_size=28)
        return img


# ----------------------------------------------------------------------------- S26 Mars

class Mars(Scene):
    dur = 7.5
    grain = 0.03
    vig = 0.5
    R = 430

    def setup(self):
        tw, th = 2048, 1024
        n1 = fbm(tw, th, 300, 6, 0.55, 261)
        n2 = fbm(tw, th, 60, 5, 0.6, 262)
        n3 = fbm(tw, th, 12, 3, 0.6, 263)
        dark = np.clip((n1 - 0.55) * 3, 0, 1)
        base = np.array([0.75, 0.36, 0.17], np.float32)
        darkc = np.array([0.30, 0.15, 0.09], np.float32)
        tex = base * (1 - dark[..., None]) + darkc * dark[..., None]
        tex *= (0.8 + 0.3 * n2 + 0.1 * n3)[..., None]
        lat = np.linspace(90, -90, th, dtype=np.float32)[:, None]
        cap = np.clip((np.abs(lat) - 72) / 6, 0, 1) * (0.7 + 0.3 * n2)
        tex = tex * (1 - cap[..., None]) + cap[..., None] * np.array([0.95, 0.93, 0.9], np.float32)
        rng = np.random.default_rng(264)
        for _ in range(110):
            x, y = rng.uniform(0, tw), rng.uniform(100, th - 100)
            r = rng.uniform(3, 14) ** 1.2
            cv2.circle(tex, (int(x), int(y)), int(r), (0.62, 0.30, 0.15), 1, cv2.LINE_AA)
        self.tex = np.concatenate([tex, tex[:, :64]], 1).astype(np.float32)
        s = 2 * self.R
        yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
        u = (xx - self.R) / self.R
        v = (yy - self.R) / self.R
        rr = u * u + v * v
        self.inside = (rr <= 1).astype(np.float32)
        z = np.sqrt(np.clip(1 - rr, 0, 1))
        self.lon0 = np.arctan2(u, z)
        self.latp = (0.5 - np.arcsin(np.clip(-v, -1, 1)) / math.pi) * (th - 1)
        L = np.array([-0.6, -0.35, 0.72])
        L /= np.linalg.norm(L)
        self.lamb = np.clip(u * L[0] + v * L[1] + z * L[2], 0, 1)
        self.rim = np.clip(1 - z, 0, 1) ** 3 * self.inside
        self.edge_aa = np.clip((1 - np.sqrt(rr)) * self.R, 0, 1)
        self.stars = star_layer(W, H, 5000, 265, bright=0.7)
        halo = np.zeros((s + 200, s + 200), np.float32)
        yy2, xx2 = np.mgrid[0:s + 200, 0:s + 200].astype(np.float32)
        d = np.sqrt((xx2 - s / 2 - 100) ** 2 + (yy2 - s / 2 - 100) ** 2) / self.R
        self.halo = (np.exp(-((d - 1.0) / 0.03) ** 2) * (d > 0.98)).astype(np.float32)

    def planet(self, rot):
        th, tw = self.tex.shape[0], self.tex.shape[1] - 64
        mapx = ((self.lon0 + rot) / (2 * math.pi) % 1.0) * tw
        col = cv2.remap(self.tex, mapx.astype(np.float32), self.latp.astype(np.float32), cv2.INTER_LINEAR)
        light = self.lamb[..., None] ** 0.8
        col = col * (0.03 + 1.25 * light) + self.rim[..., None] * np.array([0.9, 0.5, 0.35], np.float32) * 0.5 * \
            self.lamb[..., None]
        return col * self.edge_aa[..., None]

    def render(self, t, f):
        img = self.stars.copy()
        p = self.planet(t * 0.12 + 1.0)
        s = p.shape[0]
        cx, cy = 1260, 560
        a = self.edge_aa[..., None]
        reg = img[cy - s // 2:cy + s // 2, cx - s // 2:cx + s // 2]
        reg *= 1 - a
        reg += p
        hs = self.halo.shape[0]
        y0, x0 = cy - hs // 2, cx - hs // 2
        Y1, X1 = min(H, y0 + hs), min(W, x0 + hs)
        img[max(0, y0):Y1, max(0, x0):X1] += self.halo[max(0, y0) - y0:Y1 - y0, max(0, x0) - x0:X1 - x0, None] * \
            np.array([1.0, 0.55, 0.35], np.float32) * 0.5
        bloom(img, 0.8, 0.3, 30)
        return warp(img, 1.0 + 0.06 * t / self.dur, 0, cx=1260, cy=560, tx=300, ty=20)

    def overlay(self, img, t, f):
        chapter_tag(img, t, 0.2, 7.3, "CHAPTER VI  ·  VISITORS FROM THE STARS")
        a = window(t, 0.6, 7.4, 0.4, 0.4)
        n = int(6000 * ease_out((t - 0.8) / 2.2, 3))
        draw_text(img, f"{n:,}+", 110, 220, "shoulders", 180, (255, 255, 255), a)
        draw_text(img, "颗已确认的系外行星", 116, 430, "sans_bold", 40, (230, 230, 230), a, tracking=0.06)
        draw_text(img, "CONFIRMED EXOPLANETS", 116, 490, "plex", 26, (180, 180, 180), a, tracking=0.2)
        caption(img, t, 1.2, 7.4, "2025 年 9 月  ·  NASA 毅力号火星车",
                "火星岩石中的“潜在生命迹象”",
                ("毅力号在火星古河道采集的岩石中，发现了迄今“最清晰的潜在生命迹象”。",
                 "宇宙中可能孕育生命的家园，或许远比我们想象的多。"),
                bottom=H - 80, title_size=50)
        return img


# ----------------------------------------------------------------------------- S27 recap

class Recap(Scene):
    dur = 7.5
    grain = 0.05
    vig = 0.5
    PICKS = [("FlirTeaser", 3.0), ("Plutarch", 5.3), ("YangzhouPearl", 6.2), ("Nuremberg", 5.0),
             ("LosAngeles1942", 5.0), ("Arnold1947", 4.5), ("Radar1952", 3.0), ("Montage", 11.5),
             ("Rendlesham", 5.0), ("BelgianTriangle", 4.0), ("PhoenixLights", 4.0), ("Hangzhou", 2.5),
             ("NavyFlir", 3.5), ("Hearing", 2.05), ("Hellfire", 4.1), ("Interstellar", 8.5)]

    def setup(self):
        from timeline import TIMELINE
        self.by_name = {type(sc).__name__: sc for _, sc in TIMELINE}

    def render(self, t, f):
        if t >= 5.0:
            return blank((0, 0, 0))
        i = min(len(self.PICKS) - 1, int(t / 0.3125))
        name, lt = self.PICKS[i]
        sc = self.by_name[name]
        sc.ensure()
        img = sc.render(lt + (t % 0.3125), f)
        img = desaturate(img, 0.55)
        img = contrast(img, 1.25, 0.35)
        k = (t % 0.3125) / 0.3125
        img *= 0.55 + 0.45 * k
        img += (1 - k) ** 4 * 0.5
        img = chroma_shift(img, 6)
        z = 1.08 - 0.06 * k
        return warp(img, z, 0)

    def overlay(self, img, t, f):
        words = ["看见。", "记录。", "否认。", "解密。"]
        if t < 5.0:
            i = min(3, int(t / 1.25))
            lt = t - i * 1.25
            s = 1.0 + 0.3 * (1 - ease_out(lt / 0.2))
            img = darken_region(img, 660, 400, 1260, 680, 0.55, 200)
            draw_text(img, words[i], W / 2, H / 2, "serif_black", int(150 * s), (255, 255, 255), 1.0, anchor="cm",
                      tracking=0.15, shadow=0.9)
        else:
            a = window(t, 5.2, 7.2, 0.6, 0.35)
            draw_text(img, "但我们，仍然不知道——", W / 2, H / 2, "serif_light", 60, (235, 235, 235), a, anchor="cm",
                      tracking=0.2)
        return img


# ----------------------------------------------------------------------------- S28 title

class Title(Scene):
    dur = 10.0
    grain = 0.03
    vig = 0.45
    flash_in = 0.9
    ECX, ECY, ER = 960, 2700, 1880

    def setup(self):
        self.sky = milky_way(2400, 1400, 281, angle=-15, band_width=0.10) * 0.8
        xs, ys = grid()
        dx, dy = xs - self.ECX, ys - self.ECY
        d = np.sqrt(dx * dx + dy * dy)
        self.d = d
        self.inside = np.clip(self.ER - d, 0, 1).astype(np.float32)
        h = d - self.ER
        self.atmo = (np.exp(-np.maximum(h, 0) / 14) * (h > -2)).astype(np.float32)
        self.atmo_in = (np.exp(np.minimum(h, 0) / 40) * (h <= 0)).astype(np.float32)
        ang = np.arctan2(dx, -dy)
        self.ang = ang
        rng = np.random.default_rng(282)
        cities = np.zeros((H, W, 3), np.float32)
        n = 30000
        px = rng.uniform(0, W, n)
        py = rng.uniform(self.ECY - self.ER, H, n)
        cl = fbm(W, H, 90, 5, 0.6, 283)
        keep = cl[np.clip(py.astype(int), 0, H - 1), np.clip(px.astype(int), 0, W - 1)] > 0.55
        px, py = px[keep], py[keep]
        b = rng.random(px.size) ** 2 * 0.9
        col = np.stack([b * 1.0, b * 0.75, b * 0.4], 1).astype(np.float32)
        from core import splat_points
        splat_points(cities, px.astype(np.float32), py.astype(np.float32), col)
        cities += cv2.GaussianBlur(cities, (0, 0), 3) * 2
        self.cities = cities * self.inside[..., None]
        clouds = fbm(W, H, 200, 6, 0.55, 284)
        self.clouds = (np.clip((clouds - 0.45) * 2, 0, 1) * self.inside).astype(np.float32)

    def render(self, t, f):
        k = t / self.dur
        img = warp(self.sky, 0.85, -3 + 2 * k, ty=-120 + 30 * k)
        sun_x = 1430
        sunup = ramp(t, 3.0, 5.0)
        dir_ = np.exp(-((self.ang - math.atan2(sun_x - self.ECX, self.ECY)) / 0.35) ** 2)
        earth = self.cities * (1 - 0.6 * sunup * dir_[..., None])
        earth += self.clouds[..., None] * np.array([0.05, 0.07, 0.12], np.float32) * (0.4 + 2.5 * dir_[..., None] *
                                                                                    (0.3 + sunup))
        img = img * (1 - self.inside[..., None]) + earth
        atm_col = np.array([0.25, 0.55, 1.0], np.float32)
        warmth = dir_[..., None] * np.array([1.0, 0.55, 0.25], np.float32)
        img += self.atmo[..., None] * (atm_col * (0.35 + 0.9 * dir_[..., None]) + warmth * (0.6 + 1.2 * sunup))
        img += self.atmo_in[..., None] * atm_col * 0.12 * (0.3 + dir_[..., None])
        limb_y = self.ECY - math.sqrt(self.ER ** 2 - (sun_x - self.ECX) ** 2)
        sy = limb_y + 30 - 60 * sunup
        pk = ramp(t, 3.2, 4.2)
        add_glow(img, sun_x, sy, 30, (1.0, 0.9, 0.75), 2.5 * pk)
        add_glow(img, sun_x, sy, 220, (1.0, 0.75, 0.5), 0.9 * (0.2 + pk), kind="halo", reach=2400)
        add_ellipse_glow(img, sun_x, sy, 900, 5, 0, (0.5, 0.7, 1.0), 1.2 * pk, reach=2.0)
        for a in range(6):
            ang = a * math.pi / 3 + 0.2
            add_ellipse_glow(img, sun_x, sy, 190, 3, ang, (1.0, 0.95, 0.85), 0.6 * pk, reach=2.0)
        bloom(img, 0.7, 0.6, 60)
        return warp(img, 1.06 - 0.05 * ease_io(k), 0, ty=10 - 30 * ease_io(k))

    def overlay(self, img, t, f):
        a = 1 - ramp(t, 9.2, 10.0)
        s = 1.0 + 0.2 * (1 - ease_out(t / 0.35)) + 0.03 * t / 10
        tr = 0.55 - 0.15 * ease_out(t / 8)
        draw_text(img, "未 知 接 触", W / 2, 380, "serif_black", int(170 * s), (255, 255, 255), a, anchor="cm",
                  tracking=tr, shadow=0.7)
        la = ramp(t, 0.6, 1.4) * a
        lw = int(520 * ease_out((t - 0.6) / 1.5))
        if lw > 0:
            img[498:500, W // 2 - lw:W // 2 + lw] += np.float32(0.85 * la)
        draw_text(img, "UNIDENTIFIED  ·  A  HISTORY  OF  CONTACT", W / 2, 530, "mono", 30, (220, 220, 220), la,
                  anchor="ct", tracking=0.45)
        ta = ramp(t, 4.8, 5.6) * a
        draw_text(img, "从古至今，他们一直都在？", W / 2, 612, "serif_light", 50, (255, 235, 210), ta, anchor="ct",
                  tracking=0.25, shadow=0.6)
        return img


# ----------------------------------------------------------------------------- S29 end

class EndCard(Scene):
    dur = 10.0
    grain = 0.03
    vig = 0.5
    fade_out = 1.2

    def setup(self):
        self.sky = milky_way(2400, 1400, 291, angle=-35, band_width=0.10) * 0.55

    def render(self, t, f):
        img = warp(self.sky, 0.82 + 0.03 * t / 10, 4 * t / 10)
        if 5.5 < t < 9.5:
            u = (t - 5.5)
            if u < 2.0:
                x, y = lerp(300, 1150, ease_io(u / 2.0)), lerp(200, 260, ease_io(u / 2.0))
            elif u < 3.0:
                x, y = 1150, 260
            else:
                v = ease_in((u - 3.0) / 0.25, 2)
                x, y = lerp(1150, 2400, v), lerp(260, -200, v)
            add_glow(img, x, y, 2.5, (1.0, 1.0, 1.0), 2.0)
            add_glow(img, x, y, 12, (0.8, 0.9, 1.0), 0.3, kind="halo")
        return img

    def overlay(self, img, t, f):
        a = window(t, 0.4, 5.0, 0.8, 0.6)
        draw_text(img, "继续仰望天空。", W / 2, H / 2 - 40, "serif_bold", 72, (255, 255, 255), a, anchor="cm",
                  tracking=0.3, shadow=0.6)
        draw_text(img, "KEEP  WATCHING  THE  SKIES", W / 2, H / 2 + 50, "mono", 28, (200, 200, 200), a, anchor="cm",
                  tracking=0.4)
        b = window(t, 5.2, 9.9, 0.8, 0.8)
        lines = ["本片所有画面均为程序生成的情景再现，配乐与音效为程序合成。",
                 "事件、引文与数据取材自公开的历史文献、政府报告与新闻报道。",
                 "部分事件至今仍存争议；美国官方调查迄今未发现外星技术的确凿证据。"]
        for i, l in enumerate(lines):
            draw_text(img, l, W / 2, H / 2 + 120 + i * 48, "sans", 28, (190, 190, 190), b, anchor="ct", tracking=0.05)
        draw_text(img, "未 知 接 触", W / 2, H / 2 - 110, "serif_black", 64, (255, 255, 255), b, anchor="cm",
                  tracking=0.4)
        draw_text(img, "UNIDENTIFIED  ·  A  HISTORY  OF  CONTACT", W / 2, H / 2 - 20, "mono", 22, (200, 200, 200), b,
                  anchor="cm", tracking=0.4)
        return img
