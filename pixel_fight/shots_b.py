"""Storyboard part 2: clash & speed fight -> moon -> return -> roaring cannon -> serious punch -> end."""
import math

import numpy as np

from engine import *
from fighters import Fighter, pose, draw, aura
from fx import *
from backgrounds import City, Space, NIGHT, BURST, CHARGE, DAWN, mix_pal, bg_radial, bg_streaks
from shotbase import *

AURA = hexc('#ff3ad8')


def boros_aura(fr, L, t, k=0.42):
    aura(fr, L[1], AURA, t, k, height=10, seed=3)


# =====================================================================  S10
class S10_Clash(Shot):
    dur = 5.0
    SFX = [(0.0, 'dash', 1.0), (0.44, 'clash', 1.0), (0.45, 'explode', 0.9)] + \
          [(0.98 + k * 0.29, 'hit' if k % 2 else 'hit2', 0.9) for k in range(8)] + \
          [(3.58, 'punch_big', 1.0), (3.72, 'whoosh', 0.9), (4.62, 'ting', 0.9)]

    PAIRS = [('punch', 'block', 'S'), ('block', 'kick', 'B'), ('kick_hi', 'block', 'S'), ('dodge_side', 'punch', 'B'),
             ('uppercut', 'block', 'S'), ('block', 'spin_kick', 'B'), ('punch', 'punch', 'S'), ('block', 'jab_hi', 'B')]

    def setup(self):
        self.city = City(6)
        s = self.sai = Fighter('saitama', -160, 0, 1)
        b = self.bor = Fighter('boros', 160, 0, -1)
        b.form = 'burst'
        for f in (s, b):
            f.settle(wind=(-100, 0))
        rng = self.rng
        self.ex = []
        for k, (ps, pb, att) in enumerate(self.PAIRS):
            side = 1 if k % 3 != 1 else -1
            cx = rng.uniform(-110, 110)
            air = k in (2, 3, 5, 7)
            cy = rng.uniform(-150, -80) if air else 0
            self.ex.append(dict(te=0.98 + k * 0.29, side=side, cx=cx, cy=cy, air=air, ps=ps, pb=pb, att=att,
                                zoom=rng.uniform(1.15, 1.55)))
        self.hist_s = []
        self.hist_b = []
        self.prev = None

    def place(self, e):
        s, b = self.sai, self.bor
        sd = e['side']
        s.f, b.f = sd, -sd
        s.pose, b.pose = pose(e['ps']), pose(e['pb'])
        gap = 20 if 'kick' not in e['ps'] + e['pb'] else 24
        s.x, b.x = e['cx'] - gap * sd, e['cx'] + (gap + 6) * sd
        s.air = b.air = e['air']
        if e['air']:
            s.y = b.y = e['cy']
        else:
            s.y = b.y = 0

    def contact(self, e):
        a = self.sai if e['att'] == 'S' else self.bor
        J = a.joints()
        pn = e['ps'] if e['att'] == 'S' else e['pb']
        if 'kick' in pn:
            return J['toef'] if pn != 'spin_kick' else J['toef']
        if pn == 'uppercut':
            return J['fistf']
        return J['fistf'] if a.pose['sf'] > a.pose['sb'] else J['fistb']

    def frame(self, t):
        s, b, cam = self.sai, self.bor, self.cam
        rng = self.rng
        tq = math.floor(t * 30) / 30
        pal = BURST
        show = True
        streak = None
        flash_k = 0.0
        if t < 0.45:
            k = ease_in(remap(t, 0.0, 0.45), 2)
            s.x, b.x = lerp(-170, -34, k), lerp(170, 40, k)
            s.f, b.f = 1, -1
            s.air = b.air = False
            s.y = b.y = 0
            s.pose = pose('dash') if t < 0.36 else pose('punch', sf=88)
            b.pose = pose('dash') if t < 0.36 else pose('punch', sf=92)
            cam.x, cam.y, cam.zoom = 0, -55, 1.0 + 0.4 * k
        elif t < 0.95:
            k = ease_out(remap(t, 0.68, 0.9))
            s.x, b.x = lerp(-34, -62, k), lerp(40, 68, k)
            if t > 0.68:
                s.pose, b.pose = pose('guard'), pose('guard')
            cam.x, cam.y, cam.zoom = 0, -55, 1.4 - 0.2 * k
        elif t < 3.3:
            idx = min(7, int((t - 0.98) / 0.29)) if t >= 0.98 else -1
            if idx < 0:
                show = False
            else:
                e = self.ex[idx]
                lt = t - e['te']
                self.place(e)
                if lt > 0.19:
                    show = False
                    nxt = self.ex[idx + 1] if idx + 1 < len(self.ex) else None
                    if nxt:
                        streak = (e, nxt, remap(lt, 0.19, 0.29))
                c = self.contact(e)
                cam.x = e['cx'] * 0.6
                cam.y = (e['cy'] if e['air'] else 0) * 0.8 - 50
                cam.zoom = e['zoom'] + lt * 0.4
                if self.ev(e['te']):
                    self.hit(c[0], c[1], 1.4, color=(1, 0.5, 0.9) if e['att'] == 'B' else (1, 0.85, 0.3),
                             trauma=0.55, n=50)
                    self.parts.burst(rng, c[0], c[1], 20, speed=(200, 500), ang=(0, 360), life=(0.1, 0.3),
                                     c0=(1, 1, 1), c1=(0.6, 0.6, 1), drag=4)
                if lt < 0.05:
                    flash_k = 0.25
        else:
            s.f, b.f = 1, -1
            if t < 3.7:
                s.air = b.air = False
                s.x, b.x = -12, 34
                s.y = b.y = 0
                s.pose = pose('block')
                b.pose = pose('spin_kick') if tq < 3.52 else pose('kick_hi')
                cam.x, cam.y, cam.zoom = 10, -60, 1.6 + (t - 3.3) * 0.5
            else:
                k = ease_in(remap(t, 3.7, 4.45), 2)
                s.air = True
                s.x = lerp(-12, -150, k)
                s.y = lerp(-50, -900, k)
                s.pose = pose('knocked', rot=(t - 3.7) * 900)
                b.pose = Track([(3.7, pose('kick_hi')), (4.0, pose('look_up', head=-40), 'out')])(tq)
                u = ease_io(remap(t, 3.7, 4.6))
                cam.x = lerp(10, -30, u)
                cam.y = lerp(-60, -190, u)
                cam.zoom = lerp(1.6, 0.95, u)
                if t < 4.45 and self.every(1):
                    self.parts.emit(np.array([[s.x, s.y]]), np.array([[rng.uniform(-10, 10), rng.uniform(0, 20)]]),
                                    1.4, rng.uniform(4, 7), (0.95, 0.9, 1.0), (0.4, 0.3, 0.5), drag=2, mode=DUST)
        if self.ev(0.45):
            p = (3, -58)
            self.hit(p[0], p[1], 3.2, color=(1, 0.6, 0.9), trauma=1.5, n=160)
            self.rings.add(p[0], p[1], 300, dur=0.8, color=(1, 0.7, 1), width=16)
            self.rings.add(p[0], p[1], 200, dur=0.55, color=(1, 1, 0.8), width=8, delay=0.05)
            self.rings.add(0, 0, 400, dur=1.0, squash=0.18, color=(1, 0.6, 0.8), width=10)
            self.debris.spawn(rng, 0, 0, 60, speed=(100, 420), ang=(200, 340), size=(2, 7),
                              color=pal['ground'] * 1.8, grav=450)
            for d in (-1, 1):
                self.parts.burst(rng, d * 30, -4, 40, speed=(150, 420), ang=(270 - 90 * d - 12, 270 - 90 * d + 12),
                                 life=(0.6, 1.2), size=(6, 14), c0=pal['ground'] * 2.0, c1=pal['ground'], drag=2.5,
                                 mode=DUST)
        if self.ev(3.58):
            fp = b.joints()['toef']
            self.hit(fp[0], fp[1], 2.2, color=(1, 0.5, 0.9), trauma=1.1, n=90)
        cam.tick()
        fr = Frame(cam)
        self.city.draw(fr, t + 20, pal, cloud_speed=14)
        if self.every(3):
            self.embers(2, color0=(1, 0.4, 0.8), color1=(0.4, 0.0, 0.3))
        self.fx_step(0.0)
        for f in (b, s):
            f.step(t, (-200 * f.f if 0.95 < t < 3.3 else -140, -100 if t > 3.7 else 0))
        self.hist_s.append(snap(s))
        self.hist_b.append(snap(b))
        amb = (0.8, 0.72, 0.9)
        if t < 0.45:
            trail(fr, s, self.hist_s, hexc('#ffe070'), n=4, every=2, alpha=0.55)
            trail(fr, b, self.hist_b, AURA, n=4, every=2, alpha=0.55)
        if streak is not None:
            e0, e1, k = streak
            md = MaskDraw()
            for who, col in (('s', hexc('#ffe070')), ('b', AURA)):
                sd0 = e0['side'] if who == 's' else -e0['side']
                sd1 = e1['side'] if who == 's' else -e1['side']
                p0 = np.array([e0['cx'] - 20 * sd0, (e0['cy'] if e0['air'] else 0) - 45])
                p1 = np.array([e1['cx'] - 20 * sd1, (e1['cy'] if e1['air'] else 0) - 45])
                a = cam.pts(p0 + (p1 - p0) * max(0, k - 0.4))
                c = cam.pts(p0 + (p1 - p0) * min(1, k + 0.3))
                m = MaskDraw().capsule(a, c, 1.5, 3).get()
                fr.img[m] = 1.0
                fr.glow[m] += col * 1.5
                fr.glow[dilate(m, 2) & ~m] += col * 0.5
        if show:
            Lb = b.render(cam, ambient=amb, rims=[((0, -1), hexc('#ff6ad8'), 0.5, 1)])
            boros_aura(fr, Lb, t, 0.35)
            draw(fr, Lb)
            draw(fr, s.render(cam, ambient=amb, rims=[((0, -1), hexc('#ffd080'), 0.5, 1)]))
        if 0.45 <= t < 0.8 and self.every(2):
            for _ in range(3):
                a = rng.uniform(0, 6.28)
                p0 = np.array([3, -58])
                lightning(fr, rng, p0, p0 + np.array([math.cos(a), math.sin(a)]) * rng.uniform(40, 110),
                          color=(1, 0.6, 1), inten=1.4, branches=2)
        self.fx_draw(fr)
        if t > 4.5:
            sp = cam.w2s(-150, -900)
            g = math.sin(math.pi * remap(t, 4.6, 4.95))
            sp = (sp[0] if 0 < sp[0] < W else W * 0.3, max(LB + 8, sp[1]))
            fr.ui.append(lambda o: glint(o, None, sp[0], sp[1], 10, g))
        if 0.45 <= t < 0.62:
            k = int((t - 0.45) * FPS)
            if k < 3:
                post(fr, lambda o: impact_frame(o, 0, c_dark=(0, 0, 0), c_light=(1, 1, 1)))
            elif k < 6:
                post(fr, lambda o: impact_frame(o, 1, c_dark=(1, 0.2, 0.7), c_light=(0.05, 0, 0.05)))
            else:
                post(fr, lambda o: chroma(o, 3))
        if 3.58 <= t < 3.66:
            post(fr, lambda o: impact_frame(o, 1, c_dark=(1, 0.9, 0.9), c_light=(0.1, 0, 0.1)))
        if flash_k:
            fr.ui.append(lambda o: o.__setitem__(slice(None), o * (1 - flash_k) + flash_k))
        rng2 = np.random.RandomState(self.i // 2)
        if t < 0.45:
            fr.ui.append(lambda o: hlines(o, rng2, 30, color=(1, 1, 1), alpha=0.45, direction=1))
        if 0.95 < t < 3.3:
            fr.ui.append(lambda o: speed_lines(o, rng2, W / 2, H / 2, n=36, color=(1, 1, 1), alpha=0.25))
        if 3.7 < t < 4.4:
            post(fr, lambda o: vsmear(o, 4))
        lb_ui(fr)
        return fr.finish()


# =====================================================================  S11
class S11_Ascent(Shot):
    dur = 3.4
    SFX = [(0.0, 'rush_up', 1.0), (2.8, 'whoosh', 0.8)]

    def setup(self):
        s = self.sai = Fighter('saitama', 0, -60, 1)
        s.air = True
        s.pose = pose('fly_up')
        s.settle(wind=(0, 900))
        self.sp = Space(8)
        self.yt = Track([(0, -60), (3.4, -3400, 'in')])

    def frame(self, t):
        s, cam = self.sai, self.cam
        rng = self.rng
        s.y = self.yt(t)
        s.pose = pose('fly_up', rot=8 * math.sin(t * 3), head=-10 + 6 * math.sin(t * 2))
        s.expr = 'blank'
        cam.x, cam.y, cam.zoom = 0, s.y - 12, 1.8
        cam.add_trauma(0.02)
        cam.tick()
        fr = Frame(cam)
        alt = clamp01(-s.y / 3000.0)
        top = BURST['top'] * (1 - alt) + np.array([0.0, 0.0, 0.02]) * alt
        low = BURST['low'] * (1 - alt) ** 2 + hexc('#0a1440') * alt * (1 - alt) * 2
        g = np.clip(YY / H, 0, 1)
        col = top + (low - top) * g[..., None]
        fr.img[:] = np.floor(col * 16 + BAYER[..., None] * 0.999) / 16
        if alt > 0.35:
            self.sp.stars(fr, t, bright=remap(alt, 0.35, 0.7))
        # clouds rushing past (world altitudes -1400..-500)
        wy = (YY - H / 2) / cam.zoom + cam.y
        band = np.clip(1 - np.abs(wy + 950) / 450.0, 0, 1)
        if band.max() > 0:
            d = fbm(XX / 50.0, wy / 14.0, 4, seed=5)
            dens = (d - 0.42) * 3 * band
            m1 = dens > 0.15
            fr.img[m1] = BURST['cloud']
            m2 = dens > 0.4
            fr.img[m2] = BURST['cloud_hi'] * 0.8
        # earth curvature (late)
        if alt > 0.5:
            rise = ease_out(remap(alt, 0.5, 1.0)) * 70
            R = 900.0
            cx, cy = W / 2, H + R - rise
            d2 = np.hypot(XX - cx, YY - cy)
            m = d2 < R
            fr.img[m] = hexc('#1d4fb3') * (0.5 + 0.5 * np.clip((R - d2[m]) / 60.0, 0, 1))[:, None]
            rim = (d2 >= R) & (d2 < R + 6)
            fr.glow[rim] += hexc('#4aa0ff') * 0.6
        # moon approaching
        if t > 2.7:
            k = ease_in(remap(t, 2.7, 3.4))
            R = 90 + 300 * k
            cy = -R + 40 + k * 160
            m = np.hypot(XX - W / 2, YY - cy) < R
            fr.img[m] = hexc('#9a98a6') * (0.7 + 0.3 * vnoise(XX[m] / 9.0, YY[m] / 9.0, 3))[:, None]
        # heat plasma while punching through the atmosphere
        heat = math.sin(math.pi * remap(alt, 0.02, 0.45))
        if self.every(1):
            J = s.joints()
            for _ in range(3):
                p = J['hip'] + np.array([rng.uniform(-10, 10), rng.uniform(-40, 30)])
                self.parts.emit(p[None], np.array([[rng.uniform(-30, 30), 900 + rng.uniform(0, 400)]]),
                                0.25, 1.3, (1, 0.8, 0.5) if heat > 0.2 else (0.8, 0.8, 1), (0.8, 0.2, 0.1),
                                mode=SPARK)
        self.fx_step(None)
        self.parts.draw(fr)
        s.step(t, (30 * math.sin(t * 7), 1400))
        L = s.render(cam, ambient=(0.85, 0.8, 0.9), rims=[((0, -1), (1, 0.6, 0.3), 1.2 * heat, 2)])
        if heat > 0.05:
            hc = cam.w2s(*s.point('head'))
            radial(fr.glow, hc[0], hc[1] - 6, 46, (1, 0.45, 0.15), 1.0 * heat, steps=5)
            md = MaskDraw().poly([(hc[0] - 22, hc[1] + 8), (hc[0], hc[1] - 26), (hc[0] + 22, hc[1] + 8),
                                  (hc[0] + 10, hc[1] + 70), (hc[0] - 10, hc[1] + 70)])
            m = md.get()
            fr.glow[m] += np.array([1.0, 0.5, 0.2]) * 0.35 * heat
        draw(fr, L)
        rng2 = np.random.RandomState(self.i)
        fr.ui.append(lambda o: vlines(o, rng2, 30, color=(1, 1, 1), alpha=0.35, lens=(40, 160)))
        lb_ui(fr)
        a = remap(t, 3.25, 3.4)
        fade_ui(fr, a, (1, 1, 1))
        return fr.finish()


# =====================================================================  S12
class S12_Moon(Shot):
    dur = 3.7
    SFX = [(0.3, 'land', 1.0), (0.32, 'rumble', 0.6), (2.2, 'charge_small', 0.7), (2.62, 'jump', 1.0),
           (2.62, 'explode', 0.7)]

    def setup(self):
        s = self.sai = Fighter('saitama', 20, 0, -1)
        s.pose = pose('fly_up')
        s.tatter = 0.15
        s.settle(wind=(0, 300))
        self.sp = Space(9)
        self.pt = Track([(0, pose('fly_up')), (0.28, pose('fly_up')), (0.31, pose('land'), 'snap'),
                         (0.85, pose('land')), (1.2, pose('idle'), 'out'), (2.2, pose('idle')),
                         (2.5, pose('crouch'), 'out'), (2.62, pose('crouch')), (2.66, pose('fly', rot=-40), 'snap')])

    def frame(self, t):
        s, cam = self.sai, self.cam
        rng = self.rng
        tq = math.floor(t * 30) / 30
        s.pose = self.pt(tq)
        if t < 0.3:
            s.air = False
            s.pose['lift'] = 260 * (1 - ease_in(remap(t, 0, 0.3), 2))
        elif t < 2.62:
            s.air = False
        else:
            k = ease_in(remap(t, 2.62, 3.0), 2)
            s.air = True
            s.x = 20 - 500 * k
            s.y = -38 - 700 * k
        s.shake = 1.0 if 2.3 < t < 2.62 else 0.0
        if self.ev(0.3):
            cam.add_trauma(0.9)
            self.rings.add(20, 0, 160, dur=0.8, squash=0.2, color=(0.9, 0.9, 1), width=6)
            for a0 in (180, 330):
                self.parts.burst(rng, 20, -2, 16, speed=(60, 200), ang=(a0, a0 + 30), life=(1.2, 2.2), size=(3, 7),
                                 c0=hexc('#c8c6d2'), c1=hexc('#5a5866'), drag=1.6, grav=10, mode=DUST, spread=6)
            self.debris.spawn(rng, 20, -2, 25, speed=(60, 200), ang=(200, 340), size=(1.5, 4),
                              color=hexc('#a8a6b4'), grav=70)
        if self.ev(2.62):
            cam.add_trauma(1.3)
            self.rings.add(20, 0, 260, dur=1.0, squash=0.2, color=(1, 1, 1), width=10)
            self.rings.add(20, -20, 120, dur=0.5, color=(1, 0.9, 0.6), width=8)
            for a0 in (178, 200, 318, 338):
                self.parts.burst(rng, 20, -2, 14, speed=(80, 300), ang=(a0, a0 + 24), life=(1.4, 2.6), size=(4, 10),
                                 c0=hexc('#d8d6e2'), c1=hexc('#4a4856'), drag=1.3, grav=12, mode=DUST, spread=10)
            self.debris.spawn(rng, 20, -2, 50, speed=(80, 320), ang=(190, 350), size=(2, 6), color=hexc('#a8a6b4'),
                              grav=60, life=(2.5, 4))
            self.parts.burst(rng, 20, -10, 60, speed=(200, 500), ang=(190, 350), life=(0.2, 0.6), c0=(1, 1, 1),
                             c1=(0.6, 0.7, 1), drag=2)
        if 2.25 < t < 2.62 and self.every(2):
            self.parts.burst(rng, 20 + rng.uniform(-40, 40), -1, 2, speed=(10, 30), ang=(260, 280), life=(0.8, 1.4),
                             size=(2, 4), c0=hexc('#c8c6d2'), c1=hexc('#5a5866'), grav=-10, mode=DUST)
        cam.x = -30 + 6 * t
        cam.y = -70
        cam.zoom = 1.05 + 0.08 * ease_io(remap(t, 0.4, 2.5))
        cam.tick(decay=1.0)
        fr = Frame(cam)
        self.sp.stars(fr, t)
        self.sp.earth(fr, t, -120, -58, 42, par=0.1, light=(0.8, -0.3))
        self.sp.moon_ground(fr, t)
        if t > 2.62:
            cracks(fr, 12, 20, 0, 160, ease_out(remap(t, 2.62, 3.0)), color=(0.9, 0.9, 1.0), inten=0.4, n=9)
        self.fx_step(0.0)
        s.step(t, (0, 200) if t < 0.3 else (-40, 0))
        if t < 3.0:
            draw(fr, s.render(cam, light=(0.8, -0.6), ambient=(0.95, 0.95, 1.0),
                              rims=[((-1, -0.3), hexc('#6aa8ff'), 0.6, 1)]))
        if t > 2.62:
            hp = cam.w2s(s.x, s.y)
            k = remap(t, 2.62, 3.1)
            md = MaskDraw().capsule(cam.w2s(20, -40), hp, 1, 3)
            m = md.get()
            fr.glow[m] += np.array([1, 0.9, 0.6]) * (1 - k) * 1.5
        self.fx_draw(fr)
        lb_ui(fr)
        fr.ui.append(lambda o: subtitle(o, '埼玉', '……这里是月球？', t, 1.0, 1.4))
        a = 1 - remap(t, 0, 0.12)
        fade_ui(fr, a, (1, 1, 1))
        return fr.finish()


# =====================================================================  S13
class S13_Return(Shot):
    dur = 2.4
    SFX = [(0.0, 'meteor', 1.0), (0.55, 'explode', 1.0), (0.56, 'boom', 1.0), (0.6, 'rumble', 0.9)]

    def setup(self):
        self.city = City(7)
        s = self.sai = Fighter('saitama', -50, 0, 1)
        s.tatter = 0.35
        s.settle(wind=(-60, 0))
        b = self.bor = Fighter('boros', 70, -70, -1)
        b.air = True
        b.form = 'burst'
        b.pose = pose('float')
        b.settle(wind=(0, -200))
        self.p0 = np.array([-330.0, -360.0])
        self.p1 = np.array([-50.0, -10.0])

    def frame(self, t):
        s, b, cam = self.sai, self.bor, self.cam
        rng = self.rng
        k = ease_in(remap(t, 0, 0.55), 1.4)
        head = self.p0 + (self.p1 - self.p0) * k
        if self.ev(0.55):
            cam.add_trauma(1.5)
            x = self.p1[0]
            self.parts.burst(rng, x, -8, 150, speed=(80, 420), ang=(185, 355), life=(1.2, 2.4), size=(8, 22),
                             c0=hexc('#7a6878'), c1=hexc('#2a1c2a'), drag=1.8, grav=-10, mode=DUST, spread=12)
            self.debris.spawn(rng, x, -6, 60, speed=(120, 480), ang=(195, 345), size=(2, 7),
                              color=BURST['ground'] * 1.8, grav=420)
            self.parts.burst(rng, x, -10, 120, speed=(200, 700), ang=(180, 360), life=(0.3, 0.8), c0=(1, 1, 0.8),
                             c1=(1, 0.4, 0.1), drag=2)
            self.rings.add(x, 0, 380, dur=1.0, squash=0.2, color=(1, 0.7, 0.4), width=12)
            self.rings.add(x, -30, 200, dur=0.6, color=(1, 1, 0.9), width=10)
        cam.x = lerp(-60, 0, ease_out(remap(t, 0, 0.55))) + 0
        cam.y = lerp(-140, -70, ease_out(remap(t, 0, 0.55)))
        cam.zoom = 1.0 + 0.3 * ease_io(remap(t, 0.8, 2.4))
        if t < 0.55 and self.every(4):
            cam.add_trauma(0.15)
        cam.tick()
        fr = Frame(cam)
        self.city.draw(fr, t + 30, BURST, cloud_speed=10)
        if t < 0.55:
            u = (self.p1 - self.p0) / np.hypot(*(self.p1 - self.p0))
            tail = head - u * 160
            md = MaskDraw().capsule(cam.pts(tail), cam.pts(head), 0.5, 7 * cam.zoom)
            m = md.get()
            fr.img[m] = (1, 0.9, 0.6)
            fr.glow[m] += np.array([1.0, 0.5, 0.15]) * 1.2
            md = MaskDraw().capsule(cam.pts(head - u * 60), cam.pts(head), 0.5, 3.5 * cam.zoom)
            fr.img[md.get()] = 1.0
            hp = cam.w2s(*head)
            radial(fr.glow, hp[0], hp[1], 40, (1, 0.6, 0.2), 1.3, steps=5)
            self.parts.emit(head[None] + rng.uniform(-4, 4, (3, 2)), -u[None] * 200 + rng.uniform(-60, 60, (3, 2)),
                            0.35, 1.5, (1, 0.9, 0.5), (1, 0.2, 0.05), drag=3, mode=SPARK)
        self.fx_step(0.0)
        b.expr = 'grin' if t > 1.3 else 'calm'
        b.pose = pose('float', head=-6 + 3 * math.sin(t * 3))
        b.y = -72 + 3 * math.sin(t * 2.5)
        b.step(t, (0, -300))
        Lb = b.render(cam, ambient=(0.8, 0.7, 0.9), rims=[((-1, 0.4), (1, 0.6, 0.3), 0.8 * (t > 0.5), 1)])
        boros_aura(fr, Lb, t, 0.35)
        draw(fr, Lb)
        if t > 0.55:
            s.step(t, (-80, 0))
            s.pose = pose('idle2')
            draw(fr, s.render(cam, ambient=(0.8, 0.72, 0.85), rims=[((1, -0.3), AURA, 0.6, 1)]))
            if t < 1.4:
                gp = cam.w2s(-50, 0)
                radial(fr.glow, gp[0], gp[1], 90 * cam.zoom, (1, 0.5, 0.2), 1.2 * (1 - remap(t, 0.55, 1.4)), steps=5)
        self.fx_draw(fr)
        if 0.55 <= t < 0.66:
            k2 = int((t - 0.55) * FPS)
            post(fr, (lambda o: o * 0 + 1) if k2 < 3 else (lambda o: impact_frame(o, 0, c_light=(1, 0.8, 0.5))))
        lb_ui(fr)
        fr.ui.append(lambda o: subtitle(o, '波罗斯', '哈哈哈！你果然回来了！', t, 1.3, 1.1, color=(0.7, 0.8, 1)))
        return fr.finish()


# =====================================================================  S14
def cannon_geom(b):
    J = b.joints()
    m = (J['fistf'] + J['fistb']) / 2
    sh = J['shoulder']
    d = m - sh
    d = d / (np.hypot(*d) + 1e-6)
    return m + d * 12, d


class S14_Charge(Shot):
    dur = 4.7
    SFX = [(0.1, 'charge_big', 1.0), (2.05, 'banner', 0.9), (2.05, 'shout_boros', 0.8), (4.25, 'compress', 1.0)]

    def setup(self):
        self.city = City(8)
        b = self.bor = Fighter('boros', 55, -122, -1)
        b.air = True
        b.form = 'burst'
        b.pose = pose('cannon', sf=60, sb=64, rot=12)
        b.settle(wind=(0, -200))
        s = self.sai = Fighter('saitama', -70, 0, 1)
        s.pose = pose('look_up')
        s.tatter = 0.4
        s.settle(wind=(150, -60))
        self.r = Track([(0, 0), (0.25, 2), (2.0, 15, 'io'), (3.1, 18), (4.2, 28, 'io'), (4.45, 21, 'out'),
                        (4.7, 22)])

    def frame(self, t):
        b, s, cam = self.bor, self.sai, self.cam
        rng = self.rng
        close = 2.0 <= t < 3.1
        pal = mix_pal(BURST, CHARGE, ease_io(remap(t, 0, 2.0)))
        r = self.r(t)
        b.expr = 'grin'
        b.pose = pose('cannon', sf=60, sb=64, rot=12, head=-4 + 2 * math.sin(t * 20) * remap(t, 3, 4.5))
        b.shake = 0.3 + 1.2 * remap(t, 3.1, 4.4)
        o, d = cannon_geom(b)
        J = b.joints()
        if close:
            hc = J['head']
            cam.x, cam.y = hc[0] - 12, hc[1] + 8
            cam.zoom = 4.0 + 0.4 * remap(t, 2.0, 3.1)
        else:
            cam.x, cam.y = -5, -92
            cam.zoom = 0.82 + (0.06 * ease_io(remap(t, 3.1, 4.4)) if t > 3.1 else 0.04 * remap(t, 0, 2))
        if self.every(5):
            cam.add_trauma(0.08 + 0.25 * remap(t, 0, 4.4))
        if self.ev(4.2):
            cam.add_trauma(0.8)
        cam.tick()
        fr = Frame(cam)
        self.city.draw(fr, t + 40, pal, cloud_speed=-25)
        # swirling intake
        self.parts.attract = (o[0], o[1], 1600)
        n = 6 if t < 4.2 else 2
        a = rng.uniform(0, 6.28, n)
        rr = rng.uniform(60, 180, n)
        pos = np.stack([o[0] + np.cos(a) * rr, o[1] + np.sin(a) * rr], -1)
        tang = np.stack([-np.sin(a), np.cos(a)], -1) * 120
        cols = [(0.8, 0.6, 1.0), (0.5, 0.8, 1.0), (1, 0.5, 0.9)]
        self.parts.emit(pos, tang, rr / 260.0, 1.5, cols[self.i % 3], (1, 1, 1), drag=0.5, mode=DOT)
        if self.every(3):
            x = rng.uniform(-200, 200)
            self.debris.spawn(rng, x, -1, 1, speed=(30, 80), ang=(255, 285), size=(1.5, 5),
                              color=pal['ground'] * 1.7, grav=-90, life=(2.5, 3.5), spin=4)
        self.fx_step(None)
        self.parts.attract = None
        s.step(t, (220, -60))
        if not close:
            draw(fr, s.render(cam, ambient=(0.65, 0.6, 0.85), rims=[((1, -1), hexc('#c08aff'), 0.9, 1)]))
        b.step(t, (0, -250))
        Lb = b.render(cam, ambient=(0.6, 0.55, 0.85) if close else (0.75, 0.7, 0.9),
                      rims=[((d[0], d[1]), hexc('#c0a0ff'), 1.1, 2 if close else 1)])
        boros_aura(fr, Lb, t, 0.3)
        draw(fr, Lb)
        self.fx_draw(fr, parts=True)
        orb(fr, o[0], o[1], r * (1 + 0.04 * math.sin(t * 40)), t, c_out=(0.65, 0.25, 1.0), c_mid=(0.45, 0.75, 1.0),
            inten=0.9 + 0.5 * remap(t, 3.1, 4.4))
        if self.every(2) and r > 4:
            for _ in range(1 + int(2 * remap(t, 2, 4.4))):
                aa = rng.uniform(0, 6.28)
                p1 = o + np.array([math.cos(aa), math.sin(aa)]) * r * rng.uniform(1.8, 4.0)
                lightning(fr, rng, o, p1, color=(0.6, 0.6, 1.0), inten=1.3, branches=1, depth=4)
        if close:
            fr.ui.append(lambda o_: banner(o_, t - 2.05, '崩星咆哮炮', 'COLLAPSING  STAR  ROARING  CANNON',
                                           color=(0.55, 0.7, 1.0), side=-1, y=150))
        lbh = LB + 8 * remap(t, 3.1, 4.4)
        fr.ui.append(lambda o_: letterbox(o_, lbh))
        return fr.finish(bloom_k=1.0)


# =====================================================================  S15
class S15_Beam(Shot):
    dur = 2.6
    SFX = [(0.0, 'beam', 1.0), (0.02, 'explode', 1.0), (1.0, 'rumble', 0.8)]

    def setup(self):
        self.city = City(8)
        b = self.bor = Fighter('boros', 55, -122, -1)
        b.air = True
        b.form = 'burst'
        b.pose = pose('cannon', sf=60, sb=64, rot=12)
        b.settle(wind=(0, -200))
        s = self.sai = Fighter('saitama', -70, 0, 1)
        s.pose = pose('guard', lean=0, sf=80, ef=100, sb=70, eb=110)
        s.tatter = 0.5
        s.settle(wind=(-300, 200))

    def frame(self, t):
        b, s, cam = self.bor, self.sai, self.cam
        rng = self.rng
        o, d = cannon_geom(b)
        tgt = np.array([s.x, -40.0])
        v = tgt - o
        ang = math.degrees(math.atan2(v[1], v[0]))
        L = np.hypot(*v) + 400
        length = L * ease_out(remap(t, 0, 0.12), 2)
        close = t >= 1.0
        if close:
            cam.x, cam.y, cam.zoom = s.x + 4, -44, 2.3 + 0.2 * remap(t, 1.0, 2.6)
        else:
            cam.x, cam.y, cam.zoom = -5, -92, 0.9
        if self.every(3):
            cam.add_trauma(0.35)
        cam.tick()
        fr = Frame(cam)
        self.city.draw(fr, t + 45, CHARGE, cloud_speed=-40)
        # ground eruption where the beam lands
        gx = s.x
        if self.every(2):
            self.debris.spawn(rng, gx + rng.uniform(-20, 20), -2, 3, speed=(150, 420), ang=(200, 340), size=(2, 6),
                              color=CHARGE['ground'] * 1.8, grav=450)
            self.parts.burst(rng, gx, -4, 12, speed=(200, 600), ang=(190, 350), life=(0.2, 0.6), c0=(1, 1, 1),
                             c1=(0.5, 0.4, 1.0), drag=2)
        if self.every(12):
            self.rings.add(gx, 0, 180, dur=0.6, squash=0.22, color=(0.7, 0.6, 1.0), width=8)
        self.fx_step(0.0)
        b.step(t, (0, -300))
        if not close:
            Lb = b.render(cam, ambient=(0.7, 0.65, 0.95), rims=[((d[0], d[1]), (1, 1, 1), 1.0, 1)])
            draw(fr, Lb)
        beam(fr, o[0], o[1], ang, length, 22 if not close else 26, t, c_out=(0.55, 0.25, 1.0),
             c_mid=(0.45, 0.75, 1.0), inten=1.1)
        s.step(t, (-500, 300))
        s.tatter = 0.5 + 0.3 * remap(t, 0, 2.6)
        Ls = s.render(cam, ambient=(0.05, 0.03, 0.1), rims=[((-d[0], -d[1]), (1, 1, 1), 1.4, 1)])
        if close:
            rgbs, As, _ = Ls
            sil = np.clip(rgbs, 0, 1)
            fr.ui.append(lambda o_: o_.__setitem__(As, sil[As]))
            if self.every(2):
                J = s.joints()
                p = J['neck'] + rng.uniform(-6, 6, 2)
                self.parts.emit(p[None], np.array([[-rng.uniform(100, 220), rng.uniform(40, 120)]]), 0.7, 1,
                                (1, 1, 1), (0.7, 0.6, 1), mode=PIX)
        self.fx_draw(fr)
        if t < 0.1:
            post(fr, lambda o_: o_ * 0 + 1)
        post(fr, lambda o_: o_ * np.array([0.92, 0.9, 1.08]))
        lbh = LB + 8
        fr.ui.append(lambda o_: letterbox(o_, lbh))
        return fr.finish(bloom_k=1.0)


# =====================================================================  S16
class S16_Serious(Shot):
    dur = 3.7
    SFX = [(0.0, 'beam_inside', 1.0), (1.4, 'silence', 1.0), (1.55, 'slam', 1.0), (1.8, 'serious', 1.0),
           (2.6, 'charge_small', 0.8), (3.35, 'clench', 1.0)]

    def setup(self):
        s = self.sai = Fighter('saitama', -20, 0, 1)
        s.tatter = 0.8
        s.expr = 'serious'
        s.pose = pose('walk_a')
        s.settle(wind=(-500, 0))

    def frame(self, t):
        s, cam = self.sai, self.cam
        rng = self.rng
        tq = math.floor(t * 30) / 30
        seg = 0 if t < 1.4 else (1 if t < 2.6 else 2)
        fr = None
        if seg == 0:
            ph = (tq * 4) % 2
            s.pose = pose('walk_a') if ph < 1 else pose('walk_b')
            s.x = -20 + 14 * t
            cam.x, cam.y, cam.zoom = s.x + 6, -44, 2.1
            if self.every(20):
                cam.add_trauma(0.25)
        elif seg == 1:
            s.pose = pose('idle', head=-4)
            J = s.joints()
            cam.x, cam.y = J['head'][0] - 7, J['head'][1] + 4
            cam.zoom = 6.2 + 0.4 * remap(t, 1.4, 2.6)
            if self.ev(1.55):
                cam.add_trauma(0.6)
        else:
            s.pose = pose('serious_wind')
            J = s.joints()
            cam.x, cam.y = J['fistf'][0] + 10, J['fistf'][1] - 4
            cam.zoom = 4.6 + 0.6 * remap(t, 2.6, 3.7)
            if self.ev(3.35):
                cam.add_trauma(0.9)
        cam.tick()
        fr = Frame(cam)
        bg_streaks(fr, t, hexc('#3a1a8a'), hexc('#e8e0ff'), speed=1400 if seg != 1 else 500, direction=1, seed=seg)
        if self.every(1):
            J = s.joints()
            p = J['neck'] + np.array([-8, rng.uniform(-4, 30)])
            self.parts.emit(p[None], np.array([[-rng.uniform(150, 300), rng.uniform(-40, 40)]]), 0.6, 1,
                            (1, 1, 1), (0.7, 0.6, 1), mode=PIX)
        self.fx_step(None)
        s.step(t, (-700, 0))
        if seg == 0:
            L = s.render(cam, ambient=(0.12, 0.08, 0.2), rims=[((1, 0), (1, 1, 1), 1.3, 2), ((-1, 0), hexc('#a06aff'), 0.6, 1)])
        elif seg == 1:
            L = s.render(cam, light=(0.9, -0.4), ambient=(0.5, 0.45, 0.65),
                         rims=[((1, -0.2), (1, 1, 1), 0.9, 3), ((-1, 0), hexc('#a06aff'), 0.5, 2)])
        else:
            L = s.render(cam, light=(0.9, -0.4), ambient=(0.55, 0.5, 0.7), rims=[((1, 0), (1, 0.9, 0.9), 0.9, 3)])
        draw(fr, L)
        self.parts.draw(fr)
        if seg == 2:
            fp = cam.w2s(*s.point('fistf'))
            p = remap(t, 2.6, 3.6)
            radial(fr.glow, fp[0], fp[1], 60, (1, 0.3, 0.2), 0.5 + 0.8 * p, steps=5)
            if self.every(2):
                for _ in range(1 + int(p * 3)):
                    a = rng.uniform(0, 6.28)
                    c = s.point('fistf')
                    lightning(fr, rng, c, c + np.array([math.cos(a), math.sin(a)]) * rng.uniform(6, 14),
                              color=(1, 0.5, 0.3), inten=1.3, branches=1, depth=3)
            rng2 = np.random.RandomState(self.i // 2)
            fr.ui.append(lambda o: speed_lines(o, rng2, fp[0], fp[1], n=60, color=(1, 1, 1), alpha=0.5 * p,
                                               r_in=(0.3, 0.6)))
        if seg == 1:
            def ui(o):
                p = ease_out(remap(t, 1.55, 1.7), 3)
                if t >= 1.55:
                    size = int(round(80 - 44 * p))
                    blit_text(o, '认真系列', W * 0.2 + cam.ox, H * 0.42 + cam.oy, color=(1, 1, 1), color2=(1, 0.75, 0.7),
                              outline=(0.7, 0.05, 0.08), path=FONT_CJK, size=size, thick=2, shadow=3, spacing=2)
                if t >= 1.8:
                    a = remap(t, 1.8, 1.95)
                    blit_text(o, 'SERIOUS  SERIES', W * 0.2, H * 0.42 + 30, color=(1, 0.3, 0.3), outline=(0, 0, 0),
                              path=FONT_BOLD, size=12, spacing=2, alpha=a)
            fr.ui.append(ui)
            post(fr, lambda o: o * np.array([1.0, 0.92, 0.95]))
        lbh = LB + 8
        fr.ui.append(lambda o: letterbox(o, lbh))
        return fr.finish(bloom_k=0.9)


# =====================================================================  S17
class S17_SeriousPunch(Shot):
    dur = 3.8
    SFX = [(0.0, 'beam', 0.8), (0.2, 'serious_punch', 1.0), (0.22, 'explode', 1.0), (0.5, 'shockwave', 1.0),
           (1.25, 'sky_split', 1.0), (1.3, 'slam', 0.9), (2.5, 'dissolve', 0.9)]

    def setup(self):
        self.city = City(9)
        s = self.sai = Fighter('saitama', -40, 0, 1)
        s.tatter = 0.85
        s.expr = 'serious'
        s.pose = pose('serious_wind')
        s.settle(wind=(-500, 0))
        b = self.bor = Fighter('boros', 20, -80, -1)
        b.air = True
        b.form = 'burst'
        b.pose = pose('hurt', lean=-20)
        b.settle(wind=(300, 0))

    def frame(self, t):
        s, b, cam = self.sai, self.bor, self.cam
        rng = self.rng
        seg = 0 if t < 1.2 else (1 if t < 2.4 else 2)
        if seg == 0:
            s.pose = pose('serious_wind') if t < 0.2 else pose('serious_punch')
            fp = s.joints()['fistf']
            if self.ev(0.2):
                self.hit(fp[0] + 5, fp[1], 3.5, color=(1, 0.9, 0.7), trauma=1.5, n=200)
                for k in range(5):
                    self.rings.add(fp[0] + 20 + k * 40, fp[1], 120 + k * 80, dur=0.7 + k * 0.1, color=(1, 1, 1),
                                   width=10, squash=1.4, delay=k * 0.05)
            k = ease_io(remap(t, 0.5, 1.2))
            cam.x, cam.y, cam.zoom = lerp(0, 60, k), lerp(-58, -80, k), lerp(1.5, 0.75, k)
            if t > 0.2 and self.every(3):
                cam.add_trauma(0.5)
        elif seg == 1:
            cam.x, cam.y, cam.zoom = 0, -250, 0.9
            if self.ev(1.25):
                cam.add_trauma(1.0)
                self.rings.add(0, -170, 600, dur=1.4, squash=0.3, color=(1, 1, 1), width=18)
                self.rings.add(0, -170, 420, dur=1.0, squash=0.3, color=(1, 0.9, 0.7), width=10, delay=0.1)
        else:
            J = b.joints()
            cam.x, cam.y, cam.zoom = J['hip'][0] - 6, J['hip'][1] - 12, 2.1
        cam.tick()
        fr = Frame(cam)
        if seg == 0:
            self.city.draw(fr, t + 50, CHARGE, cloud_speed=60 if t > 0.2 else -40)
        elif seg == 1:
            k = ease_out(remap(t, 1.25, 2.4), 2)
            pal = mix_pal(CHARGE, DAWN, k)
            self.city.draw(fr, t + 50, pal, split=(0, -72, 0.32, 10 + 110 * k), sun=(0, -60), cloud_speed=30)
        else:
            self.city.draw(fr, t + 50, DAWN, split=(0, -72, 0.32, 120), sun=(0, -60), cloud_speed=4)
        self.fx_step(None)
        if seg == 0:
            fp = s.joints()['fistf']
            if t < 0.2:
                beam(fr, 420, fp[1] - 2, 180, 420 - fp[0] - 4, 30, t, inten=1.1)
            else:
                p = ease_out(remap(t, 0.2, 1.0), 2)
                front = fp[0] + 700 * p
                if front < 460:
                    beam(fr, 460, fp[1] - 2, 180, 460 - front, 30 * (1 - p * 0.7), t, inten=1.1)
                spread = math.radians(8 + 30 * p)
                Lc = 50 + 900 * p
                apex = np.array([fp[0] + 2, fp[1]])
                for wk, col, gi in ((1.0, (0.7, 0.6, 1.0), 0.8), (0.55, (1, 1, 1), 1.5)):
                    pts = [apex, apex + np.array([math.cos(spread * wk), -math.sin(spread * wk)]) * Lc,
                           apex + np.array([Lc * 1.05, 0]),
                           apex + np.array([math.cos(spread * wk), math.sin(spread * wk)]) * Lc]
                    m = MaskDraw().poly(cam.pts(np.array(pts))).get()
                    fr.img[m] = np.asarray(col)
                    fr.glow[m] += np.asarray(col) * gi
                if self.every(1):
                    self.parts.burst(rng, fp[0] + 10, fp[1], 20, speed=(300, 900), ang=(-40, 40), life=(0.2, 0.6),
                                     c0=(1, 1, 1), c1=(0.6, 0.5, 1), drag=1)
            s.step(t, (-600, 0) if t < 0.2 else (-300, 0))
            draw(fr, s.render(cam, light=(0.9, -0.3), ambient=(0.5, 0.45, 0.7) if t < 0.2 else (0.9, 0.85, 1.0),
                              rims=[((1, 0), (1, 1, 1), 1.2, 2)]))
        elif seg == 2:
            prog = ease_in(remap(t, 2.45, 3.6), 1.4) * 1.35
            J = b.joints()
            b.dissolve = (J['hip'][0] - 16, J['hip'][1] - 16, prog, 60)
            b.pose = pose('hurt', lean=-20, head=-30)
            b.expr = 'calm'
            b.step(t, (300, 0))
            L = b.render(cam, light=(-0.8, -0.5), ambient=(0.95, 0.85, 0.85), rims=[((-1, -0.2), (1, 0.9, 0.7), 0.9, 2)])
            draw(fr, L)
            band = b.last_band
            if band is not None and band.any():
                ys, xs = np.nonzero(band)
                sel = rng.choice(len(xs), min(len(xs), 14), replace=False)
                wx = (xs[sel] - W / 2 - cam.ox) / cam.zoom + cam.x
                wy = (ys[sel] - H / 2 - cam.oy) / cam.zoom + cam.y
                n = len(sel)
                self.parts.emit(np.stack([wx, wy], -1), np.stack([rng.uniform(30, 120, n), rng.uniform(-80, 10, n)], -1),
                                rng.uniform(0.5, 1.4, n), rng.uniform(1, 2, n), (1, 0.9, 0.6), (1, 0.3, 0.1),
                                drag=1.0, grav=-30, mode=EMBER)
        self.fx_draw(fr)
        if 0.2 <= t < 0.5:
            k = int((t - 0.2) * FPS)
            mode = (k // 2) % 3
            if mode == 0:
                post(fr, lambda o: impact_frame(o, 0, c_dark=(0, 0, 0), c_light=(1, 1, 1)))
            elif mode == 1:
                post(fr, lambda o: impact_frame(o, 1, c_dark=(1, 1, 1), c_light=(0, 0, 0)))
            else:
                post(fr, lambda o: impact_frame(o, 0, c_dark=(0.85, 0.05, 0.1), c_light=(1, 0.95, 0.9)))
        if 0.5 <= t < 0.6:
            post(fr, lambda o: chroma(o, 4))
        if seg == 1:
            def ui(o):
                if t >= 1.3:
                    p = ease_out(remap(t, 1.3, 1.45), 3)
                    size = int(round(96 - 52 * p))
                    blit_text(o, '认真一拳', W / 2 + cam.ox, H / 2 + 26 + cam.oy, color=(1, 1, 1), color2=(1, 0.8, 0.5),
                              outline=(0.75, 0.05, 0.08), path=FONT_CJK, size=size, thick=2, shadow=3, spacing=3)
                if t >= 1.55:
                    blit_text(o, 'SERIOUS  PUNCH', W / 2, H / 2 + 58, color=(1, 0.3, 0.25), outline=(0, 0, 0),
                              path=FONT_BOLD, size=13, spacing=3, alpha=remap(t, 1.55, 1.7))
                fl = 1 - remap(t, 1.2, 1.5)
                o[:] = o * (1 - fl) + fl
            fr.ui.append(ui)
        if seg == 2:
            fr.ui.append(lambda o: subtitle(o, '波罗斯', '……原来如此……差距……竟有这么大……', t, 2.5, 1.3,
                                            color=(0.7, 0.8, 1)))
        lb_ui(fr)
        return fr.finish(bloom_k=0.9)


# =====================================================================  S18
class S18_End(Shot):
    dur = 6.2
    SFX = [(0.0, 'wind', 0.6), (0.0, 'calm', 0.45), (2.3, 'shout', 1.0), (3.6, 'end_hit', 1.0), (4.0, 'ting', 0.9)]

    def setup(self):
        self.city = City(9)
        s = self.sai = Fighter('saitama', 0, 0, 1)
        s.tatter = 0.9
        s.pose = pose('idle')
        s.settle(wind=(-90, 0))

    def frame(self, t):
        s, cam = self.sai, self.cam
        rng = self.rng
        if t < 2.3:
            cam.x, cam.y, cam.zoom = 10, -95, 0.92 + 0.08 * ease_io(t / 2.3)
            cam.tick()
            fr = Frame(cam)
            self.city.draw(fr, t + 60, DAWN, split=(0, -72, 0.32, 120), sun=(0, -60), cloud_speed=2)
            sx, sy = cam.w2s(0, -60, 0.05)
            md = MaskDraw()
            for k in range(11):
                a = math.radians(40 + k * 10 + 3 * math.sin(t * 0.6 + k))
                w = math.radians(1.2 + (k % 3))
                L = 400
                md.poly([(sx, sy), (sx + math.cos(a - w) * L, sy + math.sin(a - w) * L),
                         (sx + math.cos(a + w) * L, sy + math.sin(a + w) * L)])
            m = md.get() & (BAYER < 0.55)
            fr.glow[m] += np.array([1, 0.9, 0.6]) * 0.18
            if self.every(3):
                self.embers(1, color0=(1, 0.95, 0.8), color1=(0.6, 0.5, 0.4), up=(10, 30))
            self.fx_step(None)
            s.step(t, (-100, 0))
            s.pose = pose('idle', head=-6)
            draw(fr, s.render(cam, light=(0.3, -0.9), ambient=(1.0, 0.95, 0.9), rims=[((0, -1), (1, 0.9, 0.6), 0.6, 1)]))
            self.parts.draw(fr)
            lb_ui(fr)
            fr.ui.append(lambda o: subtitle(o, '埼玉', '……又是一拳。', t, 0.6, 1.6))
            return fr.finish()
        if t < 3.6:
            s.expr = 'shout'
            s.pose = pose('idle', head=-18, lean=-6)
            J = s.joints()
            cam.x, cam.y, cam.zoom = J['head'][0] + 2, J['head'][1] + 6, 4.4
            if self.ev(2.3):
                cam.add_trauma(0.9)
            if self.every(6):
                cam.add_trauma(0.2)
            cam.tick()
            fr = Frame(cam)
            bg_radial(fr, W / 2, H / 2, hexc('#fff2a0'), hexc('#ff8a30'), r=260)
            s.step(t, (-200, 0))
            draw(fr, s.render(cam, ambient=(1, 1, 1)))
            rng2 = np.random.RandomState(self.i // 3)
            hc = cam.w2s(*J['head'])
            fr.ui.append(lambda o: speed_lines(o, rng2, hc[0], hc[1], n=70, color=(0.1, 0.05, 0.0), alpha=0.8,
                                               r_in=(0.55, 0.85)))
            lb_ui(fr)
            fr.ui.append(lambda o: subtitle(o, '埼玉', '可恶啊啊啊————！！', t, 2.35, 1.25))
            return fr.finish(vignette=0.2)
        cam.x, cam.y, cam.zoom = 0, 0, 1
        cam.tick()
        fr = Frame(cam)
        fr.img[:] = hexc('#07040c')
        if self.every(2):
            self.parts.emit(np.array([[rng.uniform(-190, 190), 110]]), np.array([[rng.uniform(-10, 10), -rng.uniform(20, 50)]]),
                            4.0, 1.5, (1, 0.7, 0.3), (0.5, 0.1, 0.05), mode=EMBER)
        self.fx_step(None)
        self.parts.draw(fr)

        def ui(o):
            p = ease_out(remap(t, 3.6, 3.8), 3)
            size = int(round(90 - 46 * p))
            blit_text(o, '一拳超人', W / 2, H / 2 - 22, color=(1, 0.95, 0.5), color2=(1, 0.35, 0.1),
                      outline=(0.4, 0.02, 0.02), path=FONT_CJK, size=size, thick=2, shadow=3, spacing=4,
                      glowc=(1, 0.5, 0.1))
            if t > 3.9:
                a = remap(t, 3.9, 4.2)
                blit_text(o, 'SAITAMA   VS   BOROS', W / 2, H / 2 + 14, color=(1, 1, 1), outline=(0, 0, 0),
                          path=FONT_BOLD, size=12, spacing=3, alpha=a)
                blit_text(o, '— 像素同人动画 · PIXEL FAN ANIMATION —', W / 2, H / 2 + 38, color=(0.7, 0.7, 0.8),
                          outline=None, alpha=a * 0.9)
            g = math.sin(math.pi * remap(t, 4.0, 4.45))
            glint(o, None, W / 2 + 60, H / 2 - 34, 18, g)
            f = remap(t, 5.6, 6.2)
            o[:] = o * (1 - f)
            fl = 1 - remap(t, 3.6, 3.75)
            o[:] = o * (1 - fl) + fl
        fr.ui.append(ui)
        return fr.finish(vignette=0.5)
