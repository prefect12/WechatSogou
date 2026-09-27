"""Effects: particles, debris, shockwaves, impact stars, speed lines, lightning,
energy orbs and beams.  All world-space, rendered through the camera."""
import math

import numpy as np

from engine import (W, H, XX, YY, DT, MaskDraw, shift, dilate, hexc, radial, vnoise, BAYER, ease_out,
                    clamp01)

SPARK, DOT, DUST, PIX, EMBER = 0, 1, 2, 3, 4


class Particles:
    def __init__(self, cap=6000):
        self.cap = cap
        self.pos = np.zeros((cap, 2), np.float32)
        self.vel = np.zeros((cap, 2), np.float32)
        self.life = np.zeros(cap, np.float32)
        self.maxl = np.ones(cap, np.float32)
        self.size = np.ones(cap, np.float32)
        self.c0 = np.zeros((cap, 3), np.float32)
        self.c1 = np.zeros((cap, 3), np.float32)
        self.drag = np.zeros(cap, np.float32)
        self.grav = np.zeros(cap, np.float32)
        self.mode = np.zeros(cap, np.int8)
        self.n = 0
        self.attract = None   # (x, y, strength)

    def emit(self, pos, vel, life, size=1.0, c0=(1, 1, 1), c1=None, drag=0.0, grav=0.0, mode=SPARK):
        pos = np.atleast_2d(np.asarray(pos, np.float32))
        vel = np.atleast_2d(np.asarray(vel, np.float32))
        k = max(len(pos), len(vel))
        k = min(k, self.cap - self.n)
        if k <= 0:
            return
        s = slice(self.n, self.n + k)
        self.pos[s] = np.broadcast_to(pos, (max(len(pos), k), 2))[:k]
        self.vel[s] = np.broadcast_to(vel, (max(len(vel), k), 2))[:k]
        self.life[s] = np.broadcast_to(np.asarray(life, np.float32), (k,))
        self.maxl[s] = self.life[s]
        self.size[s] = np.broadcast_to(np.asarray(size, np.float32), (k,))
        c0 = np.asarray(c0, np.float32)
        self.c0[s] = np.broadcast_to(c0, (k, 3))
        self.c1[s] = np.broadcast_to(np.asarray(c1 if c1 is not None else c0, np.float32), (k, 3))
        self.drag[s] = drag
        self.grav[s] = grav
        self.mode[s] = mode
        self.n += k

    def burst(self, rng, x, y, n, speed=(60, 240), ang=(0, 360), life=(0.2, 0.6), size=(1, 2),
              c0=(1, 1, 1), c1=None, drag=2.0, grav=0.0, mode=SPARK, spread=0.0):
        a = np.radians(rng.uniform(ang[0], ang[1], n))
        sp = rng.uniform(speed[0], speed[1], n)
        vel = np.stack([np.cos(a) * sp, np.sin(a) * sp], -1)
        pos = np.stack([np.full(n, x) + rng.uniform(-spread, spread, n),
                        np.full(n, y) + rng.uniform(-spread, spread, n)], -1)
        self.emit(pos, vel, rng.uniform(life[0], life[1], n), rng.uniform(size[0], size[1], n), c0, c1,
                  drag, grav, mode)

    def update(self, dt=DT, ground=None):
        n = self.n
        if n == 0:
            return
        v = self.vel[:n]
        if self.attract is not None:
            ax, ay, st = self.attract
            d = np.array([ax, ay], np.float32) - self.pos[:n]
            L = np.hypot(d[:, 0], d[:, 1])[:, None] + 4
            v += d / L * st * dt
        v *= (1 - np.clip(self.drag[:n] * dt, 0, 1))[:, None]
        v[:, 1] += self.grav[:n] * dt
        self.pos[:n] += v * dt
        if ground is not None:
            hit = (self.pos[:n, 1] > ground) & (self.mode[:n] != DUST)
            self.pos[:n, 1][hit] = ground
            v[hit, 1] *= -0.3
            v[hit, 0] *= 0.6
        self.life[:n] -= dt
        alive = self.life[:n] > 0
        if not alive.all():
            idx = np.nonzero(alive)[0]
            m = len(idx)
            for arr in (self.pos, self.vel, self.life, self.maxl, self.size, self.c0, self.c1, self.drag,
                        self.grav, self.mode):
                arr[:m] = arr[idx]
            self.n = m

    def draw(self, fr, layer_modes=None):
        n = self.n
        if n == 0:
            return
        cam = fr.cam
        age = 1 - self.life[:n] / self.maxl[:n]
        col = self.c0[:n] + (self.c1[:n] - self.c0[:n]) * age[:, None]
        fade = (1 - age)
        sp = cam.pts(self.pos[:n])
        mode = self.mode[:n]
        # sparks: streaks along velocity, additive
        for md in (SPARK, EMBER, DOT):
            sel = np.nonzero(mode == md)[0]
            if len(sel) == 0:
                continue
            c = col[sel] * (fade[sel] ** 0.7)[:, None] * (1.6 if md == SPARK else 1.2)
            if md == SPARK:
                vs = self.vel[sel] * cam.zoom * 0.028
                steps = 5
                for j in range(steps):
                    q = sp[sel] - vs * (j / steps)
                    self._add(fr.glow, q, c * (1 - j / steps * 0.7))
            else:
                self._add(fr.glow, sp[sel], c)
                big = sel[self.size[sel] * cam.zoom >= 1.6]
                if len(big):
                    cb = col[big] * (fade[big] ** 0.7)[:, None] * 0.55
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        self._add(fr.glow, sp[big] + np.array([dx, dy]), cb)
        sel = np.nonzero(mode == PIX)[0]
        if len(sel):
            xi = np.round(sp[sel, 0]).astype(int)
            yi = np.round(sp[sel, 1]).astype(int)
            ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            fr.img[yi[ok], xi[ok]] = col[sel][ok]
        sel = np.nonzero(mode == DUST)[0]
        for i in sel:
            r = self.size[i] * cam.zoom * (0.6 + age[i] * 0.9)
            a = (1 - age[i]) ** 1.3 * 0.75
            self._blend_circle(fr.img, sp[i, 0], sp[i, 1], r, col[i], a)

    @staticmethod
    def _add(layer, p, c):
        xi = np.round(p[:, 0]).astype(int)
        yi = np.round(p[:, 1]).astype(int)
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        np.add.at(layer, (yi[ok], xi[ok]), c[ok])

    @staticmethod
    def _blend_circle(img, cx, cy, r, c, a):
        if r < 0.5:
            return
        x0, x1 = int(max(0, cx - r)), int(min(W, cx + r + 1))
        y0, y1 = int(max(0, cy - r)), int(min(H, cy + r + 1))
        if x1 <= x0 or y1 <= y0:
            return
        d = (XX[y0:y1, x0:x1] - cx) ** 2 + (YY[y0:y1, x0:x1] - cy) ** 2
        m = d <= r * r
        # 2-level dithered puff: dense core, dithered edge
        edge = (d > (r * 0.6) ** 2) & m & (BAYER[y0:y1, x0:x1] < 0.5)
        core = m & ~((d > (r * 0.6) ** 2) & ~edge)
        reg = img[y0:y1, x0:x1]
        reg[core] = reg[core] * (1 - a) + np.asarray(c) * a


class Debris:
    def __init__(self):
        self.items = []

    def spawn(self, rng, x, y, n, speed=(80, 260), ang=(200, 340), size=(1.5, 4), color=(0.4, 0.38, 0.45),
              grav=500, life=(1.5, 3.0), spin=12, glow=None):
        for _ in range(n):
            a = math.radians(rng.uniform(*ang))
            s = rng.uniform(*speed)
            k = rng.randint(4, 7)
            r = rng.uniform(*size)
            shape = [(math.cos(i / k * 6.283) * r * rng.uniform(0.6, 1.2),
                      math.sin(i / k * 6.283) * r * rng.uniform(0.6, 1.2)) for i in range(k)]
            self.items.append(dict(p=np.array([x + rng.uniform(-3, 3), y + rng.uniform(-3, 3)], np.float32),
                                   v=np.array([math.cos(a) * s, math.sin(a) * s], np.float32),
                                   rot=rng.uniform(0, 6.28), w=rng.uniform(-spin, spin), shape=shape,
                                   col=np.asarray(color, np.float32) * rng.uniform(0.8, 1.15), g=grav,
                                   life=rng.uniform(*life), age=0.0, glow=glow))

    def update(self, dt=DT, ground=0.0):
        keep = []
        for d in self.items:
            d['v'][1] += d['g'] * dt
            d['v'] *= (1 - 0.3 * dt)
            d['p'] += d['v'] * dt
            d['rot'] += d['w'] * dt
            if ground is not None and d['p'][1] > ground and d['g'] > 0:
                d['p'][1] = ground
                d['v'][1] *= -0.35
                d['v'][0] *= 0.7
                d['w'] *= 0.7
            d['age'] += dt
            if d['age'] < d['life']:
                keep.append(d)
        self.items = keep

    def draw(self, fr, light=(-0.6, -0.8)):
        cam = fr.cam
        if not self.items:
            return
        # batch by drawing each into shared masks (colors individually)
        for d in self.items:
            c, s = math.cos(d['rot']), math.sin(d['rot'])
            pts = [(d['p'][0] + x * c - y * s, d['p'][1] + x * s + y * c) for x, y in d['shape']]
            sp = cam.pts(np.array(pts, np.float32))
            if sp[:, 0].max() < 0 or sp[:, 0].min() >= W or sp[:, 1].max() < 0 or sp[:, 1].min() >= H:
                continue
            md = MaskDraw()
            md.poly(sp)
            m = md.get()
            if not m.any():
                cx, cy = sp.mean(0)
                xi, yi = int(cx), int(cy)
                if 0 <= xi < W and 0 <= yi < H:
                    fr.img[yi, xi] = d['col']
                continue
            fade = clamp01((d['life'] - d['age']) / 0.3)
            top = m & ~shift(m, 0, -1)
            fr.img[m] = fr.img[m] * (1 - fade) + d['col'] * 0.75 * fade
            fr.img[top] = fr.img[top] * (1 - fade) + np.minimum(d['col'] * 1.35 + 0.05, 1) * fade
            ol = dilate(m) & ~m
            fr.img[ol] = fr.img[ol] * (1 - fade * 0.7) + np.array([0.06, 0.04, 0.08]) * fade * 0.7
            if d['glow'] is not None:
                fr.glow[m] += np.asarray(d['glow'], np.float32) * fade


class Rings:
    """Expanding shockwave rings (optionally squashed = ground plane)."""

    def __init__(self):
        self.items = []

    def add(self, x, y, rmax, dur=0.45, squash=1.0, color=(1, 1, 1), width=6, inten=1.5, delay=0.0,
            solid=True):
        self.items.append(dict(x=x, y=y, rmax=rmax, dur=dur, squash=squash, color=np.asarray(color, np.float32),
                               width=width, inten=inten, age=-delay, solid=solid))

    def update(self, dt=DT):
        for r in self.items:
            r['age'] += dt
        self.items = [r for r in self.items if r['age'] < r['dur']]

    def draw(self, fr):
        cam = fr.cam
        for r in self.items:
            if r['age'] < 0:
                continue
            k = r['age'] / r['dur']
            rad = r['rmax'] * ease_out(k, 3) * cam.zoom
            wd = max(1.0, r['width'] * (1 - k) * cam.zoom)
            if rad < 1:
                continue
            cx, cy = cam.w2s(r['x'], r['y'])
            sq = r['squash']
            md = MaskDraw().ellipse(cx, cy, rad, rad * sq)
            outer = md.get()
            ri = max(0, rad - wd)
            inner = MaskDraw().ellipse(cx, cy, ri, ri * sq).get() if ri > 0.5 else np.zeros((H, W), bool)
            ring = outer & ~inner
            a = (1 - k) ** 1.5
            fr.glow[ring] += r['color'] * r['inten'] * a
            if r['solid'] and k < 0.5:
                edge = ring & ~shift(ring, 0, 1) if sq < 1 else ring & ~dilate(inner, 1)
                fr.img[edge] = fr.img[edge] * 0.3 + 0.7
            # faint filled disk right after birth: air pressure
            if k < 0.25:
                fr.glow[inner] += r['color'] * 0.25 * (1 - k * 4)


class Stars:
    """Impact star-bursts (jagged flash)."""

    def __init__(self):
        self.items = []

    def add(self, rng, x, y, size, color=(1, 0.8, 0.3), dur=0.16, spikes=10):
        n = spikes * 2
        rr = [size * (rng.uniform(0.75, 1.3) if i % 2 == 0 else rng.uniform(0.25, 0.4)) for i in range(n)]
        self.items.append(dict(x=x, y=y, rr=rr, color=np.asarray(color, np.float32), dur=dur, age=0.0,
                               a0=rng.uniform(0, 6.28)))

    def update(self, dt=DT):
        for s in self.items:
            s['age'] += dt
        self.items = [s for s in self.items if s['age'] < s['dur']]

    def draw(self, fr):
        cam = fr.cam
        for s in self.items:
            k = s['age'] / s['dur']
            sc = (0.55 + 0.6 * ease_out(k, 2)) * cam.zoom * (1 - k * 0.2)
            cx, cy = cam.w2s(s['x'], s['y'])
            n = len(s['rr'])
            pts = [(cx + math.cos(s['a0'] + i / n * 6.283) * r * sc, cy + math.sin(s['a0'] + i / n * 6.283) * r * sc)
                   for i, r in enumerate(s['rr'])]
            m = MaskDraw().poly(pts).get()
            core = MaskDraw().poly([(cx + (x - cx) * 0.5, cy + (y - cy) * 0.5) for x, y in pts]).get()
            fr.img[m] = s['color']
            fr.img[core] = 1.0
            fr.glow[m] += s['color'] * 1.2 * (1 - k)
            fr.glow[core] += 1.5 * (1 - k)
            ol = dilate(m) & ~m
            fr.img[ol] = fr.img[ol] * 0.3


def speed_lines(out, rng, cx, cy, n=60, r_in=(0.45, 0.8), color=(1, 1, 1), alpha=0.8, width=(1.5, 4.5)):
    """Anime focus lines converging to (cx, cy) in screen space - drawn on final image."""
    if alpha <= 0:
        return
    md = MaskDraw()
    R = math.hypot(W, H)
    for _ in range(n):
        a = rng.uniform(0, 6.283)
        ri = rng.uniform(*r_in) * R * 0.5
        w = rng.uniform(*width) / R * 2
        p_in = (cx + math.cos(a) * ri, cy + math.sin(a) * ri)
        pa = (cx + math.cos(a - w) * R, cy + math.sin(a - w) * R)
        pb = (cx + math.cos(a + w) * R, cy + math.sin(a + w) * R)
        md.poly([p_in, pa, pb])
    m = md.get()
    out[m] = out[m] * (1 - alpha) + np.asarray(color, np.float32) * alpha


def hlines(out, rng, n=30, color=(1, 1, 1), alpha=0.6, direction=1, lens=(30, 140), ys=(0, H)):
    """Horizontal streaks (dash / motion)."""
    md = MaskDraw()
    for _ in range(n):
        y = rng.uniform(*ys)
        x = rng.uniform(-60, W + 60)
        L = rng.uniform(*lens)
        th = rng.choice([1, 1, 1, 2])
        md.poly([(x, y), (x + L * direction, y - th / 2), (x + L * direction, y + th / 2)])
    m = md.get()
    out[m] = out[m] * (1 - alpha) + np.asarray(color, np.float32) * alpha


def vlines(out, rng, n=30, color=(1, 1, 1), alpha=0.6, lens=(30, 140)):
    md = MaskDraw()
    for _ in range(n):
        x = rng.uniform(0, W)
        y = rng.uniform(-60, H + 60)
        L = rng.uniform(*lens)
        th = rng.choice([1, 1, 2])
        md.poly([(x, y), (x - th / 2, y + L), (x + th / 2, y + L)])
    m = md.get()
    out[m] = out[m] * (1 - alpha) + np.asarray(color, np.float32) * alpha


def bolt_points(rng, p0, p1, depth=5, disp=0.25):
    pts = [np.asarray(p0, np.float32), np.asarray(p1, np.float32)]
    for d in range(depth):
        new = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            m = (a + b) / 2
            v = b - a
            L = np.hypot(*v)
            nrm = np.array([-v[1], v[0]]) / (L + 1e-6)
            m = m + nrm * rng.uniform(-1, 1) * L * disp
            new += [m, b]
        pts = new
    return pts


def lightning(fr, rng, p0, p1, color=(0.6, 0.8, 1), inten=1.5, branches=2, depth=5, width=1):
    cam = fr.cam
    pts = bolt_points(rng, p0, p1, depth)
    sp = cam.pts(np.array(pts))
    md = MaskDraw().line(sp, width)
    for _ in range(branches):
        i = rng.randint(1, len(pts) - 2)
        a = pts[i]
        d = np.asarray(p1) - np.asarray(p0)
        L = np.hypot(*d)
        ang = math.atan2(d[1], d[0]) + rng.uniform(-1.2, 1.2)
        b = a + np.array([math.cos(ang), math.sin(ang)]) * L * rng.uniform(0.15, 0.35)
        bp = cam.pts(np.array(bolt_points(rng, a, b, depth - 2)))
        md.line(bp, 1)
    m = md.get()
    fr.img[m] = 1.0
    halo = dilate(m, 1) & ~m
    fr.glow[m] += np.asarray(color, np.float32) * inten
    fr.glow[halo] += np.asarray(color, np.float32) * inten * 0.5


def orb(fr, x, y, r, t, c_out=(0.8, 0.3, 1.0), c_mid=(0.5, 0.7, 1.0), inten=1.0, seed=0):
    cam = fr.cam
    cx, cy = cam.w2s(x, y)
    R = r * cam.zoom
    if R < 0.5:
        return
    radial(fr.glow, cx, cy, R * 3.2, c_out, 0.9 * inten, power=2.2, steps=6)
    ang = np.arctan2(YY - cy, XX - cx)
    dist = np.hypot(XX - cx, YY - cy)
    wob = 1 + 0.12 * (vnoise(np.cos(ang) * 3 + t * 5, np.sin(ang) * 3 + t * 4, seed) - 0.5) * 2
    body = dist < R * wob
    mid = dist < R * 0.75 * wob
    core = dist < R * 0.45
    fr.img[body] = np.asarray(c_out, np.float32)
    fr.img[mid] = np.asarray(c_mid, np.float32)
    fr.img[core] = 1.0
    fr.glow[body] += np.asarray(c_out, np.float32) * 0.8 * inten
    fr.glow[mid] += np.asarray(c_mid, np.float32) * 0.8 * inten
    fr.glow[core] += 1.2 * inten
    # swirl bands inside
    sw = (np.sin(ang * 3 + dist / max(R, 1) * 6 - t * 9) > 0.6) & body & ~core
    fr.img[sw] = fr.img[sw] * 0.5 + 0.5


def beam(fr, x0, y0, ang_deg, length, width, t, c_out=(0.55, 0.25, 1.0), c_mid=(0.45, 0.75, 1.0),
         inten=1.0, seed=0):
    """Thick wobbling energy beam from (x0,y0) toward angle."""
    cam = fr.cam
    a = math.radians(ang_deg)
    u = np.array([math.cos(a), math.sin(a)], np.float32)
    n = np.array([-u[1], u[0]], np.float32)
    steps = 48
    s = np.linspace(0, length, steps)
    layers = [(1.55, c_out, 0.9, False), (1.0, c_mid, 1.0, True), (0.5, (1, 1, 1), 1.4, True)]
    for li, (wk, col, gi, solid) in enumerate(layers):
        left, right = [], []
        for j, sj in enumerate(s):
            ramp = min(1.0, (sj + 6) / 30.0)
            wob = 1 + 0.12 * math.sin(sj * 0.09 - t * 38 + li) + 0.1 * (float(
                vnoise(np.array([sj * 0.05 + t * 12]), np.array([li * 3.0 + seed]))[0]) - 0.5)
            w = width * wk * wob * ramp
            p = np.array([x0, y0], np.float32) + u * sj
            left.append(p + n * w)
            right.append(p - n * w)
        pts = cam.pts(np.array(left + right[::-1]))
        m = MaskDraw().poly(pts).get()
        if solid:
            fr.img[m] = np.asarray(col, np.float32)
        fr.glow[m] += np.asarray(col, np.float32) * gi * inten
    cx, cy = cam.w2s(x0, y0)
    radial(fr.glow, cx, cy, width * 3.5 * cam.zoom, c_out, 1.2 * inten, steps=5)


def cracks(fr, rng_seed, x, y, length, prog, color=(1.0, 0.5, 0.2), n=7, inten=1.2, squash=0.25):
    """Glowing ground fissures spreading from (x, y) - squashed onto the ground plane."""
    rs = np.random.RandomState(rng_seed)
    cam = fr.cam
    md = MaskDraw()
    for i in range(n):
        a = rs.uniform(0, 6.283)
        pts = [np.array([x, y], np.float32)]
        L = length * rs.uniform(0.5, 1.0) * prog
        segs = 6
        cur = pts[0].copy()
        for j in range(segs):
            a += rs.uniform(-0.6, 0.6)
            cur = cur + np.array([math.cos(a), math.sin(a) * squash]) * (L / segs)
            pts.append(cur.copy())
        md.line(cam.pts(np.array(pts)), 1)
    m = md.get()
    fr.img[m] = np.minimum(np.asarray(color, np.float32) * 1.4, 1)
    fr.glow[m] += np.asarray(color, np.float32) * inten
    fr.glow[dilate(m) & ~m] += np.asarray(color, np.float32) * inten * 0.3
