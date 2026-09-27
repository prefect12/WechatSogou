"""Procedural pixel backgrounds: ruined City Z at night, space / moon, abstract."""
import math

import numpy as np

from engine import (W, H, XX, YY, MaskDraw, shift, dilate, hexc, radial, fbm, vnoise, BAYER, clamp01, lerp)


def P(**kw):
    return {k: (hexc(v) if isinstance(v, str) else np.asarray(v, np.float32)) for k, v in kw.items()}


NIGHT = P(top='#070a1f', mid='#1c1840', low='#5a2a4e', hor='#8a4450', cloud='#3a3560', cloud_hi='#8a7aa8',
          far='#2b2a52', mid_b='#1d1b3a', near='#141228', win='#ffb45a', ground='#2c2536', ground2='#181420',
          fire='#ff7a2a', rim='#9aa6ff')
BURST = P(top='#12021a', mid='#3a0636', low='#a0184a', hor='#ff6a3a', cloud='#4a1040', cloud_hi='#ff5ab0',
          far='#3a1440', mid_b='#260c30', near='#18081e', win='#ff8a5a', ground='#3a1a34', ground2='#1c0a1c',
          fire='#ff4a6a', rim='#ff7ad8')
CHARGE = P(top='#05010f', mid='#1c0640', low='#4a1a8a', hor='#9a5aff', cloud='#26104a', cloud_hi='#b07aff',
           far='#1e1440', mid_b='#150c2e', near='#0c0820', win='#c08aff', ground='#221a36', ground2='#100a1c',
           fire='#b06aff', rim='#c0a0ff')
DAWN = P(top='#2a4a9a', mid='#5a8ad8', low='#f0b87a', hor='#ffe0a0', cloud='#5a5078', cloud_hi='#ffd6a0',
         far='#6a6a9a', mid_b='#4a4a78', near='#2e2c4a', win='#ffe0a0', ground='#4a4052', ground2='#2a2434',
         fire='#ffb060', rim='#fff0c0')


def mix_pal(a, b, k):
    return {key: a[key] + (b[key] - a[key]) * k for key in a}


def vgrad(fr, y0, y1, c0, c1, levels=10, rows=None):
    g = np.clip((YY - y0) / max(1.0, (y1 - y0)), 0, 1)
    g = np.floor(g * levels + BAYER * 0.999) / levels
    col = c0 + (c1 - c0) * g[..., None]
    if rows is None:
        fr.img[:] = col
    else:
        fr.img[rows] = col[rows]


class City:
    def __init__(self, seed=3):
        rs = np.random.RandomState(seed)
        self.layers = []
        spec = [(0.12, 70, (22, 62), (10, 30), 'far', 0.55),
                (0.3, 46, (30, 85), (16, 38), 'mid_b', 0.45),
                (0.55, 26, (36, 110), (26, 56), 'near', 0.35)]
        for par, n, hr, wr, ck, broken in spec:
            items = []
            x = -1100
            while x < 1100:
                w = rs.uniform(*wr)
                h = rs.uniform(*hr)
                poly = [(0, 0), (0, -h)]
                if rs.rand() < broken:
                    k = rs.randint(3, 6)
                    for j in range(1, k):
                        poly.append((w * j / k, -h + rs.uniform(-h * 0.35, h * 0.18)))
                else:
                    if rs.rand() < 0.4:
                        poly += [(w * 0.3, -h), (w * 0.3, -h - rs.uniform(4, 12)), (w * 0.4, -h - rs.uniform(4, 12)),
                                 (w * 0.4, -h)]
                poly += [(w, -h + rs.uniform(-10, 10)), (w, 0)]
                wins = []
                for wy in np.arange(-h + 6, -6, 6 if par > 0.2 else 5):
                    for wx in np.arange(3, w - 3, 5 if par > 0.2 else 4):
                        r = rs.rand()
                        if r < (0.07 if par < 0.2 else 0.16):
                            wins.append((wx, wy, rs.uniform(0.3, 1.0)))
                items.append(dict(x=x, w=w, h=h, poly=poly, wins=wins, fire=rs.rand() < 0.12))
                x += w + rs.uniform(-4, 10) + (rs.uniform(10, 50) if (par > 0.5 and rs.rand() < 0.35) else 0)
            self.layers.append((par, items, ck))
        self.rubble = []
        for _ in range(140):
            x = rs.uniform(-1200, 1200)
            r = rs.uniform(2, 9)
            k = rs.randint(4, 7)
            pts = [(math.cos(i / k * 6.28) * r * rs.uniform(0.6, 1.3), -abs(math.sin(i / k * 6.28)) * r *
                    rs.uniform(0.4, 1.0)) for i in range(k)]
            self.rubble.append((x, pts, rs.uniform(0.6, 1.1)))
        self.fg = []
        for _ in range(18):
            x = rs.uniform(-1300, 1300)
            r = rs.uniform(14, 40)
            k = rs.randint(5, 8)
            pts = [(math.cos(i / k * 6.28) * r * rs.uniform(0.6, 1.3), -abs(math.sin(i / k * 6.28)) * r *
                    rs.uniform(0.3, 0.8)) for i in range(k)]
            self.fg.append((x, pts))

    def draw(self, fr, t, pal=NIGHT, ship=0.0, clouds=True, split=None, fg=False, sun=None, cloud_speed=3.0):
        cam = fr.cam
        hy = cam.w2s(0, 0, 0.25)[1]
        # ---------------- sky
        top_y = hy - 260 * (1 + (cam.zoom - 1) * 0.25)
        g = np.clip((YY - top_y) / max(1.0, (hy - top_y)), 0, 1)
        c = np.where(g[..., None] < 0.55, pal['top'] + (pal['mid'] - pal['top']) * (g[..., None] / 0.55),
                     np.where(g[..., None] < 0.85, pal['mid'] + (pal['low'] - pal['mid']) * ((g[..., None] - 0.55) / 0.3),
                              pal['low'] + (pal['hor'] - pal['low']) * ((g[..., None] - 0.85) / 0.15)))
        q = 18
        fr.img[:] = np.floor(c * q + BAYER[..., None] * 0.999) / q
        if sun is not None:
            sx, sy = cam.w2s(sun[0], sun[1], 0.05)
            radial(fr.img, sx, sy, 150, pal['hor'], 0.6, steps=7)
            radial(fr.glow, sx, sy, 26, (1, 0.95, 0.8), 1.4, steps=4)
        # ---------------- clouds
        if clouds:
            zl = 1 + (cam.zoom - 1) * 0.15
            wx = (XX - W / 2) / zl + cam.x * 0.15 + t * cloud_speed
            wy = (YY - H / 2) / zl + cam.y * 0.15
            d = fbm(wx / 70.0, wy / 22.0, 4, seed=5)
            d2 = fbm(wx / 70.0, (wy - 3) / 22.0, 4, seed=5)
            band = np.clip(1 - np.abs((YY - (hy - 150)) / 110.0), 0, 1)
            dens = (d - 0.47) * 3.2 * band
            if split is not None:
                sx, sy, ang, width = split
                px, py = cam.w2s(sx, sy, 0.15)
                nx, ny = -math.sin(ang), math.cos(ang)
                dist = np.abs((XX - px) * nx + (YY - py) * ny)
                wob = width * (1 + 0.25 * (vnoise(XX / 17.0, YY / 17.0, 9) - 0.5))
                dens = dens * np.clip((dist - wob) / 26.0, 0, 1)
            m1 = dens > 0.18
            m2 = dens > 0.45
            fr.img[m1] = pal['cloud']
            fr.img[m2] = pal['cloud'] * 0.7 + pal['cloud_hi'] * 0.3
            lit = m1 & (d > d2 + 0.012)
            fr.img[lit] = pal['cloud_hi'] * 0.85
            edge = m1 & ~shift(m1, 0, -1)
            fr.img[edge] = pal['cloud_hi']
        # ---------------- ship
        if ship > 0:
            sx, sy = cam.w2s(40, -300, 0.08)
            zl = (1 + (cam.zoom - 1) * 0.08) * ship
            md = MaskDraw().ellipse(sx, sy, 150 * zl, 18 * zl)
            md.ellipse(sx, sy - 12 * zl, 60 * zl, 16 * zl)
            md.poly([(sx - 40 * zl, sy + 10 * zl), (sx + 40 * zl, sy + 10 * zl), (sx + 20 * zl, sy + 34 * zl),
                     (sx - 20 * zl, sy + 34 * zl)])
            hull = md.get()
            fr.img[hull] = hexc('#0e0c1c')
            top = hull & ~shift(hull, 0, -1)
            fr.img[top] = pal['rim'] * 0.6
            for i in range(24):
                a = i / 24 * 6.283
                lx = sx + math.cos(a) * 130 * zl
                ly = sy + math.sin(a) * 12 * zl + 4 * zl
                if math.sin(a) > -0.2:
                    on = 0.5 + 0.5 * math.sin(t * 6 + i * 0.9)
                    xi, yi = int(lx), int(ly)
                    if 0 <= xi < W and 0 <= yi < H:
                        fr.glow[yi, xi] += np.array([1.0, 0.3, 0.6]) * (0.6 + on)
            radial(fr.glow, sx, sy + 34 * zl, 30 * zl, hexc('#ff4aa0'), 0.7 + 0.3 * math.sin(t * 3), steps=4)
        # ---------------- building layers (haze fill below the far skyline)
        hz = cam.w2s(0, 0, 0.12)[1] - 2
        rows = YY >= hz
        gg = np.clip((YY - hz) / 40.0, 0, 1)
        gg = np.floor(gg * 5 + BAYER * 0.999) / 5
        hc = pal['low'] * 0.55 + pal['far'] * 0.45
        col = hc + (pal['near'] - hc) * gg[..., None]
        fr.img[rows] = col[rows]
        for par, items, ck in self.layers:
            zl = 1 + (cam.zoom - 1) * par
            col = pal[ck]
            md = MaskDraw()
            wmd = MaskDraw()
            fires = []
            for b in items:
                bx0, by0 = cam.w2s(b['x'], 0, par)
                if bx0 > W + 10 or bx0 + b['w'] * zl < -10:
                    continue
                pts = [(bx0 + x * zl, by0 + y * zl) for x, y in b['poly']]
                md.poly(pts)
                if zl * (0.8 if par < 0.2 else 1) > 0.55:
                    for wx, wy, lv in b['wins']:
                        if (lv + 0.07 * math.sin(t * 2 + wx)) > 0.72:
                            x0, y0 = bx0 + wx * zl, by0 + wy * zl
                            wmd.d.rectangle([x0, y0, x0 + max(0, zl - 1), y0 + max(0, zl * 1.4 - 1)], fill=1)
                if b['fire']:
                    fires.append((bx0 + b['w'] * zl * 0.5, by0 - b['h'] * zl * 0.8, zl))
            m = md.get()
            fr.img[m] = col
            rim = m & ~shift(m, -1, 0) & ~shift(m, 0, -1)
            fr.img[rim] = col * 0.5 + pal['rim'] * 0.35
            low = m & (YY > hy - 20 * zl)
            fr.img[low] = fr.img[low] * 0.9 + pal['fire'] * 0.1
            wm = wmd.get() & m
            fr.img[wm] = pal['win'] * (0.55 + par * 0.4)
            fr.glow[wm] += pal['win'] * 0.25
            for fx_, fy_, zl_ in fires:
                fl = 0.8 + 0.2 * math.sin(t * 13 + fx_) + 0.1 * math.sin(t * 29 + fy_)
                radial(fr.glow, fx_, fy_, 28 * zl_, pal['fire'], 0.35 * fl, steps=4)
                radial(fr.img, fx_, fy_ + 6 * zl_, 10 * zl_, pal['fire'], 0.6 * fl, steps=3)
            # haze between layers
            haze = np.clip((YY - (hy - 60)) / 80.0, 0, 1)[..., None]
            fr.img[m] = fr.img[m] * (1 - haze[m] * 0.25) + pal['low'] * haze[m] * 0.25
        # ---------------- ground
        gy_far = cam.w2s(0, 0, 0.55)[1]
        gy = cam.w2s(0, 0, 1.0)[1]
        rows = YY >= gy_far
        g = np.clip((YY - gy_far) / max(1.0, H - gy_far), 0, 1)
        g = np.floor(g * 8 + BAYER * 0.999) / 8
        col = pal['ground2'] + (pal['ground'] - pal['ground2']) * g[..., None]
        fr.img[rows] = col[rows]
        # street cracks / texture
        tex = vnoise((XX - W / 2) / cam.zoom / 9 + cam.x / 9, (YY - gy) / cam.zoom / 2.5, 21)
        crack = rows & (tex > 0.82) & (YY > gy - 2)
        fr.img[crack] = fr.img[crack] * 0.6
        edge = rows & ~shift(rows, 0, -1)
        fr.img[edge] = pal['rim'] * 0.25 + pal['ground'] * 0.6
        # rubble on character plane
        md = MaskDraw()
        for x, pts, k in self.rubble:
            sx, sy = cam.w2s(x, 2, 1.0)
            if -40 < sx < W + 40:
                md.poly([(sx + px * cam.zoom, sy + py * cam.zoom) for px, py in pts])
        m = md.get()
        fr.img[m] = pal['ground'] * 1.25 + pal['rim'] * 0.05
        top = m & ~shift(m, 0, -1)
        fr.img[top] = pal['ground'] * 1.2 + pal['rim'] * 0.25
        ol = dilate(m) & ~m
        fr.img[ol] = pal['ground2'] * 0.7
        self._gy = gy

    def draw_fg(self, fr, pal=NIGHT, par=1.35):
        cam = fr.cam
        md = MaskDraw()
        zl = 1 + (cam.zoom - 1) * par
        for x, pts in self.fg:
            sx, sy = cam.w2s(x, 10, par)
            if -80 < sx < W + 80:
                md.poly([(sx + px * zl, sy + py * zl) for px, py in pts])
        m = md.get()
        fr.img[m] = pal['near'] * 0.5
        top = m & ~shift(m, 0, -1)
        fr.img[top] = pal['rim'] * 0.3


class Space:
    def __init__(self, seed=8):
        rs = np.random.RandomState(seed)
        n = 420
        self.sx = rs.uniform(-W, 2 * W, n)
        self.sy = rs.uniform(-H * 2, 2 * H, n)
        self.sb = rs.uniform(0.2, 1.0, n) ** 2
        self.sp = rs.uniform(0, 6.28, n)
        self.craters = [(rs.uniform(-600, 600), rs.uniform(4, 60), rs.uniform(6, 34)) for _ in range(40)]

    def stars(self, fr, t, par=0.03, bright=1.0):
        cam = fr.cam
        x = (self.sx - cam.x * par) % (W * 1.5) - W * 0.25
        y = (self.sy - cam.y * par) % (H * 1.5) - H * 0.25
        tw = self.sb * (0.7 + 0.3 * np.sin(t * 4 + self.sp)) * bright
        xi = x.astype(int)
        yi = y.astype(int)
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        np.add.at(fr.img, (yi[ok], xi[ok]), (tw[ok, None] * np.array([0.9, 0.95, 1.0])))
        big = ok & (self.sb > 0.7)
        np.add.at(fr.glow, (yi[big], xi[big]), tw[big, None] * np.array([0.5, 0.6, 1.0]))

    def earth(self, fr, t, x, y, r, par=0.1, light=(-0.6, -0.3)):
        cam = fr.cam
        cx, cy = cam.w2s(x, y, par)
        R = r * (1 + (cam.zoom - 1) * par)
        radial(fr.glow, cx, cy, R * 1.25, hexc('#3a8aff'), 0.5, power=1.5, steps=6)
        dx = (XX - cx) / R
        dy = (YY - cy) / R
        d2 = dx * dx + dy * dy
        m = d2 < 1
        if not m.any():
            return
        z = np.sqrt(np.clip(1 - d2, 0, 1))
        u = np.arctan2(dx, z) + t * 0.05
        v = dy
        land = fbm(u * 2.2 + 3, v * 2.2, 4, 31) > 0.52
        cl = fbm(u * 4 + 9 + t * 0.03, v * 5, 3, 37) > 0.58
        col = np.where(land[..., None], hexc('#3f8f4a'), hexc('#1d4fb3'))
        col = np.where(cl[..., None], hexc('#f0f4ff'), col)
        lx, ly = light
        lam = np.clip(-(dx * lx + dy * ly) + z * 0.4, 0, 1)
        lam = np.floor(lam * 5 + BAYER * 0.99) / 5
        shade = 0.12 + 0.95 * lam
        fr.img[m] = (col * shade[..., None])[m]
        rim = m & (d2 > 0.86)
        fr.img[rim] = fr.img[rim] * 0.5 + hexc('#7ac0ff') * 0.5 * shade[rim][:, None]

    def moon_ground(self, fr, t, y=0.0):
        cam = fr.cam
        gy = cam.w2s(0, y, 1.0)[1]
        curve = ((XX - W / 2) / W) ** 2 * 14 * cam.zoom
        rows = YY >= gy + curve
        g = np.clip((YY - gy) / max(1.0, H - gy), 0, 1)
        g = np.floor(g * 6 + BAYER * 0.999) / 6
        col = hexc('#9a98a6') * (1 - g[..., None] * 0.55)
        fr.img[rows] = col[rows]
        md = MaskDraw()
        md2 = MaskDraw()
        for x, dz, r in self.craters:
            sx, sy = cam.w2s(x, y + dz * 0.8, 1.0)
            rr = r * cam.zoom
            md.ellipse(sx, sy, rr, rr * 0.28)
            md2.ellipse(sx + rr * 0.08, sy + rr * 0.05, rr * 0.85, rr * 0.2)
        m = md.get() & rows
        m2 = md2.get() & rows
        fr.img[m] = fr.img[m] * 0.72
        fr.img[m & ~m2] = hexc('#c8c6d2') * 0.9
        edge = rows & ~shift(rows, 0, -1)
        fr.img[edge] = hexc('#e8e6f0')


def bg_radial(fr, cx, cy, c_in, c_out, r=260, levels=9):
    d = np.hypot(XX - cx, YY - cy) / r
    d = np.clip(d, 0, 1)
    d = np.floor(d * levels + BAYER * 0.999) / levels
    fr.img[:] = np.asarray(c_in, np.float32) + (np.asarray(c_out, np.float32) - np.asarray(c_in, np.float32)) * d[..., None]


def bg_streaks(fr, t, c0, c1, speed=900, direction=-1, seed=0, dens=0.6):
    """Horizontal energy streak field (inside a beam / super speed)."""
    rows = np.arange(H)
    x = XX + t * speed * direction * -1
    n = vnoise(x / 60.0, YY / 2.0, seed)
    n2 = vnoise(x / 18.0, YY / 1.0, seed + 5)
    v = np.clip((n * 0.7 + n2 * 0.3 - (1 - dens)) * 2.5, 0, 1)
    v = np.floor(v * 5 + BAYER * 0.99) / 5
    fr.img[:] = np.asarray(c0, np.float32) + (np.asarray(c1, np.float32) - np.asarray(c0, np.float32)) * v[..., None]
