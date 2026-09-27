"""Shared shot machinery: event timing, afterimages, smears, glints, hit helper."""
import math

import numpy as np

from engine import (W, H, FPS, DT, Camera, Frame, MaskDraw, hexc, radial, remap, letterbox, dilate)
from fx import Particles, Debris, Rings, Stars, SPARK, DOT, DUST, PIX, EMBER

LB = 22


class Shot:
    dur = 2.0
    SFX = []

    def __init__(self, seed=0):
        self.rng = np.random.RandomState(seed + 101)
        self.cam = Camera()
        self.parts = Particles()
        self.debris = Debris()
        self.rings = Rings()
        self.stars = Stars()
        self.i = 0
        self.t = 0.0
        self.setup()

    def setup(self):
        pass

    def nframes(self):
        return int(round(self.dur * FPS))

    def ev(self, te):
        return self.i == int(round(te * FPS))

    def every(self, n):
        return self.i % n == 0

    def render(self, i):
        self.i = i
        self.t = i / FPS
        return self.frame(self.t)

    def fx_step(self, ground=0.0):
        self.parts.update(ground=ground)
        self.debris.update(ground=ground)
        self.rings.update()
        self.stars.update()

    def fx_draw(self, fr, parts=True):
        self.debris.draw(fr)
        self.rings.draw(fr)
        if parts:
            self.parts.draw(fr)
        self.stars.draw(fr)

    def hit(self, x, y, size=1.0, color=(1.0, 0.8, 0.3), trauma=0.6, n=40, ring=True):
        rng = self.rng
        self.stars.add(rng, x, y, 16 * size, color)
        self.parts.burst(rng, x, y, n, speed=(120 * size, 420 * size), life=(0.15, 0.45), c0=(1, 1, 0.85),
                         c1=color, drag=4.0)
        if ring:
            self.rings.add(x, y, 42 * size, dur=0.35, color=color, width=5 * size)
            self.rings.add(x, y, 26 * size, dur=0.22, color=(1, 1, 1), width=3 * size)
        self.cam.add_trauma(trauma)

    def embers(self, n=2, color0=(1.0, 0.55, 0.2), color1=(0.5, 0.05, 0.1), y=(-4, 0), spread=260, up=(25, 70)):
        rng = self.rng
        x = self.cam.x + rng.uniform(-spread, spread, n)
        yy = rng.uniform(*y, n)
        vx = rng.uniform(-25, 25, n)
        vy = -rng.uniform(*up, n)
        self.parts.emit(np.stack([x, yy], -1), np.stack([vx, vy], -1), rng.uniform(1.5, 3.5, n),
                        rng.uniform(1, 2.2, n), color0, color1, drag=0.2, grav=-4, mode=EMBER)


def snap(f):
    return (f.x, f.y, dict(f.pose), f.air, f.f)


def ghost(fr, ftr, s, color, alpha):
    keep = (ftr.x, ftr.y, ftr.pose, ftr.air, ftr.f, ftr.cape, ftr.hair, ftr.hole, ftr.dissolve, ftr.shake)
    ftr.x, ftr.y, ftr.pose, ftr.air, ftr.f = s
    ftr.cape, ftr.hair, ftr.hole, ftr.dissolve, ftr.shake = None, [], None, None, 0
    _, A, _ = ftr.render(fr.cam, outline=False)
    (ftr.x, ftr.y, ftr.pose, ftr.air, ftr.f, ftr.cape, ftr.hair, ftr.hole, ftr.dissolve, ftr.shake) = keep
    ftr._J = None
    c = np.asarray(color, np.float32)
    fr.img[A] = fr.img[A] * (1 - alpha) + c * alpha
    fr.glow[A] += c * alpha * 0.45


def trail(fr, ftr, hist, color, n=4, every=2, alpha=0.55):
    """Draw afterimages from a history list of snapshots (oldest first)."""
    sel = hist[::-1][every::every][:n]
    for k, s in enumerate(sel[::-1]):
        a = alpha * (k + 1) / (len(sel) + 1)
        ghost(fr, ftr, s, color, a)


def smear(fr, pts, r, color=(1, 0.95, 0.8), alpha=0.85):
    cam = fr.cam
    n = len(pts)
    if n < 2:
        return
    md = MaskDraw()
    for k in range(n - 1):
        a = cam.pts(pts[k])
        b = cam.pts(pts[k + 1])
        rr = r * cam.zoom * (0.25 + 0.75 * (k + 1) / n)
        md.capsule(a, b, rr * 0.6, rr)
    m = md.get()
    c = np.asarray(color, np.float32)
    fr.img[m] = fr.img[m] * (1 - alpha) + c * alpha
    fr.glow[m] += c * 0.35


def glint(img, glow, cx, cy, size, k=1.0):
    if k <= 0 or size < 1:
        return
    s = size * k
    md = MaskDraw()
    md.poly([(cx, cy - s), (cx + s * 0.16, cy), (cx, cy + s), (cx - s * 0.16, cy)])
    md.poly([(cx - s, cy), (cx, cy - s * 0.16), (cx + s, cy), (cx, cy + s * 0.16)])
    m = md.get()
    img[m] = 1.0
    if glow is not None:
        glow[m] += 1.2 * k
        radial(glow, cx, cy, s * 0.8, (1, 0.95, 0.8), 0.8 * k, steps=3)


def rim_dir(fr, ftr, light_xy):
    """Screen direction from the fighter's chest toward a world-space light."""
    J = ftr.joints()
    c = J['shoulder']
    return (light_xy[0] - c[0], light_xy[1] - c[1])


def fade_ui(fr, a, color=(0, 0, 0)):
    if a <= 0:
        return

    def f(o):
        o[:] = o * (1 - a) + np.asarray(color, np.float32) * a
    fr.ui.append(f)


def lb_ui(fr, h=LB):
    fr.ui.append(lambda o: letterbox(o, h))


def post(fr, fn):
    fr.post.setdefault('pre_ui', []).append(fn)
