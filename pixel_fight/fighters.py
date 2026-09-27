"""Skeletal pixel fighters: Saitama & Boros.

Bodies are built from vector parts rasterised without anti-aliasing onto the
pixel grid, then cel-shaded per part (shadow / highlight bands computed by
shifting the part mask along the light direction), outlined, and rim-lit by
the scene's dynamic lights.  Cape and hair are verlet chains.
"""
import math

import numpy as np

from engine import (W, H, XX, YY, MaskDraw, shift, dilate, erode, hexc, vnoise, BAYER, DT)


def D(a, f):
    r = math.radians(a)
    return np.array([math.sin(r) * f, math.cos(r)], np.float32)


def rot(v, deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([v[0] * c - v[1] * s, v[0] * s + v[1] * c], np.float32)


OUTLINE = hexc('#160c1e')

SPECS = {
    'saitama': dict(
        scale=1.0, head_r=7.4, neck=2.2, torso=23.0, sh_drop=3.4, ua=12.5, fa=12.0, fist_r=3.9,
        th=17.0, shn=17.0, foot=6.4, foot_h=2.9, torso_hw=(6.6, 5.5), ua_r=(3.0, 2.7),
        fa_r=(2.8, 3.3), th_r=(4.0, 3.4), sh_r=(3.3, 3.0),
        skin=hexc('#f4c89c'), suit=hexc('#f8d43a'), glove=hexc('#dc2a2c'), belt=hexc('#221f2c'),
        buckle=hexc('#e9c85a'), cape=hexc('#f1f0ec')),
    'boros': dict(
        scale=1.12, head_r=7.6, neck=2.6, torso=24.0, sh_drop=4.2, ua=13.0, fa=12.4, fist_r=4.6,
        th=17.4, shn=17.4, foot=7.4, foot_h=3.3, torso_hw=(9.4, 6.2), ua_r=(3.9, 3.3),
        fa_r=(3.3, 4.0), th_r=(4.8, 3.8), sh_r=(3.5, 3.9),
        skin=hexc('#9cc6ea'), suit=hexc('#2c2f6c'), armor=hexc('#eab83c'), trim=hexc('#fff3cf'),
        gem=hexc('#ff3a78'), hair=hexc('#3a5fd8'), hair2=hexc('#77e6ff'), eye=hexc('#ff3a2a'),
        burst=hexc('#4a3c8e'), lines=hexc('#ffb13c')),
}

BASE = dict(lean=0.0, head=0.0, sf=8.0, ef=10.0, sb=-8.0, eb=12.0, hf=5.0, kf=3.0, hb=-5.0,
            kb=3.0, foot=0.0, rot=0.0, lift=0.0)

POSES = {
    'idle': dict(lean=0, head=0, sf=4, ef=6, sb=-4, eb=8, hf=4, kf=2, hb=-4, kb=2),
    'idle2': dict(lean=2, head=4, sf=6, ef=10, sb=-6, eb=12, hf=5, kf=4, hb=-4, kb=3),
    'boros_idle': dict(lean=8, head=-4, sf=28, ef=55, sb=18, eb=65, hf=18, kf=22, hb=-16, kb=12),
    'guard': dict(lean=12, head=-2, sf=45, ef=95, sb=25, eb=105, hf=22, kf=28, hb=-20, kb=14),
    'crouch': dict(lean=30, head=-12, sf=35, ef=40, sb=5, eb=50, hf=70, kf=115, hb=15, kb=105),
    'dash': dict(lean=42, head=-30, sf=-45, ef=35, sb=-60, eb=25, hf=55, kf=70, hb=-45, kb=35),
    'fly': dict(lean=70, head=-55, sf=-70, ef=15, sb=-80, eb=10, hf=-40, kf=25, hb=-55, kb=35),
    'punch_wind': dict(lean=-6, head=4, sf=-35, ef=115, sb=60, eb=35, hf=22, kf=22, hb=-26, kb=6),
    'punch': dict(lean=22, head=-10, sf=90, ef=0, sb=-30, eb=70, hf=38, kf=28, hb=-36, kb=4),
    'punch_b': dict(lean=22, head=-10, sf=-30, ef=70, sb=90, eb=0, hf=38, kf=28, hb=-36, kb=4),
    'jab_hi': dict(lean=16, head=-8, sf=110, ef=-5, sb=-25, eb=80, hf=32, kf=24, hb=-30, kb=6),
    'uppercut': dict(lean=-18, head=-25, sf=160, ef=10, sb=-30, eb=60, hf=10, kf=10, hb=-30, kb=20),
    'kick': dict(lean=-24, head=-2, sf=-25, ef=45, sb=45, eb=70, hf=100, kf=-2, hb=-10, kb=10),
    'kick_hi': dict(lean=-38, head=5, sf=-35, ef=50, sb=40, eb=70, hf=135, kf=0, hb=-5, kb=12),
    'spin_kick': dict(lean=-30, head=10, sf=60, ef=40, sb=-40, eb=40, hf=120, kf=5, hb=-30, kb=40),
    'block': dict(lean=-4, head=8, sf=70, ef=110, sb=60, eb=120, hf=25, kf=25, hb=-22, kb=10),
    'hurt': dict(lean=-32, head=-24, sf=55, ef=20, sb=75, eb=10, hf=40, kf=35, hb=18, kb=40),
    'knocked': dict(lean=-70, head=-40, sf=130, ef=20, sb=150, eb=10, hf=60, kf=30, hb=30, kb=50),
    'dodge_back': dict(lean=-30, head=-8, sf=12, ef=25, sb=-10, eb=20, hf=10, kf=10, hb=-22, kb=14),
    'dodge_duck': dict(lean=38, head=-18, sf=18, ef=30, sb=-4, eb=35, hf=48, kf=70, hb=-16, kb=50),
    'dodge_side': dict(lean=10, head=-25, sf=6, ef=12, sb=-8, eb=14, hf=12, kf=14, hb=-12, kb=10),
    'kneel': dict(lean=25, head=18, sf=35, ef=25, sb=15, eb=30, hf=85, kf=95, hb=-5, kb=120),
    'clench': dict(lean=18, head=12, sf=40, ef=120, sb=30, eb=125, hf=26, kf=35, hb=-26, kb=18),
    'power': dict(lean=-16, head=-24, sf=-50, ef=35, sb=-62, eb=30, hf=28, kf=12, hb=-28, kb=10),
    'float': dict(lean=6, head=-6, sf=24, ef=40, sb=14, eb=50, hf=12, kf=35, hb=-8, kb=45),
    'cannon': dict(lean=-8, head=-6, sf=82, ef=4, sb=88, eb=8, hf=20, kf=25, hb=-18, kb=35),
    'fly_up': dict(lean=0, head=-10, sf=150, ef=10, sb=-10, eb=20, hf=5, kf=10, hb=-5, kb=25),
    'land': dict(lean=28, head=-6, sf=40, ef=25, sb=-30, eb=25, hf=75, kf=110, hb=15, kb=95),
    'serious_wind': dict(lean=6, head=-4, sf=-62, ef=118, sb=72, eb=18, hf=44, kf=48, hb=-42, kb=10),
    'serious_punch': dict(lean=26, head=-8, sf=94, ef=0, sb=-52, eb=55, hf=48, kf=36, hb=-46, kb=0),
    'walk_a': dict(lean=4, head=0, sf=-14, ef=10, sb=16, eb=12, hf=22, kf=8, hb=-18, kb=18),
    'walk_b': dict(lean=4, head=0, sf=16, ef=12, sb=-14, eb=10, hf=-18, kf=18, hb=22, kb=8),
    'look_up': dict(lean=-6, head=-28, sf=4, ef=8, sb=-4, eb=8, hf=5, kf=3, hb=-5, kb=3),
}


def pose(name, **kw):
    p = dict(BASE)
    p.update(POSES[name])
    p.update(kw)
    return p


class Chain:
    def __init__(self, n, seg, stiff=0.0, damp=0.94):
        self.n, self.seg, self.stiff, self.damp = n, seg, stiff, damp
        self.p = None
        self.q = None

    def reset(self, anchor, d):
        d = np.asarray(d, np.float32)
        d = d / (np.hypot(*d) + 1e-6)
        self.p = anchor[None] + d[None] * (np.arange(self.n, dtype=np.float32)[:, None] * self.seg)
        self.q = self.p.copy()

    def step(self, anchor, force, rest=None, dt=DT):
        anchor = np.asarray(anchor, np.float32)
        if self.p is None or np.hypot(*(self.p[0] - anchor)) > self.seg * 6:
            self.reset(anchor, rest if rest is not None else (0, 1))
        v = (self.p - self.q) * self.damp
        self.q = self.p.copy()
        self.p = self.p + v + np.asarray(force, np.float32) * dt * dt
        if rest is not None and self.stiff > 0:
            rd = np.asarray(rest, np.float32)
            rd = rd / (np.hypot(*rd) + 1e-6)
            tgt = anchor[None] + rd[None] * (np.arange(self.n)[:, None] * self.seg)
            self.p += (tgt - self.p) * self.stiff
        self.p[0] = anchor
        for i in range(self.n - 1):
            d = self.p[i + 1] - self.p[i]
            L = float(np.hypot(*d))
            if L > 1e-6:
                self.p[i + 1] = self.p[i] + d / L * self.seg


class Fighter:
    def __init__(self, kind, x=0.0, y=0.0, f=1):
        self.kind = kind
        self.S = SPECS[kind]
        self.x, self.y, self.f = x, y, f
        self.air = False
        self.pose = pose('idle' if kind == 'saitama' else 'boros_idle')
        self.expr = 'blank' if kind == 'saitama' else 'calm'
        self.form = 'armor'
        self.eye_open = 1.0
        self.tatter = 0.0
        self.hole = None          # (dx, dy, r) relative to hip, world units
        self.dissolve = None      # (ox, oy, progress, R) world
        self.cracks = 0.0
        self.shake = 0.0
        self.t = 0.0
        sc = self.S['scale']
        if kind == 'saitama':
            self.cape = Chain(10, 5.2 * sc, stiff=0.0, damp=0.95)
            self.hair = []
        else:
            self.cape = None
            self.hair = [Chain(5, 5.6 * sc, 0.10), Chain(5, 5.0 * sc, 0.12), Chain(4, 5.0 * sc, 0.14),
                         Chain(4, 4.0 * sc, 0.16), Chain(3, 4.0 * sc, 0.2)]
        self._J = None

    # ------------------------------------------------------------ skeleton
    def joints(self):
        S, p, f = self.S, self.pose, self.f
        sc = S['scale']
        g = lambda k: p.get(k, BASE[k])
        lean = g('lean')
        hip = np.zeros(2, np.float32)
        up = D(180 - lean, f)
        front = np.array([-up[1] * f, up[0] * f], np.float32)
        J = {}
        J['hip'] = hip
        J['neck'] = hip + up * S['torso'] * sc
        J['shoulder'] = hip + up * (S['torso'] - S['sh_drop']) * sc
        J['head'] = J['neck'] + D(180 - lean - g('head'), f) * (S['neck'] + S['head_r']) * sc
        for side, sa, ea, off in (('f', g('sf'), g('ef'), 1.2), ('b', g('sb'), g('eb'), -1.6)):
            sh = J['shoulder'] + np.array([off * f * sc, 0], np.float32)
            el = sh + D(sa, f) * S['ua'] * sc
            wr = el + D(sa + ea, f) * S['fa'] * sc
            J['sh' + side], J['el' + side], J['wr' + side] = sh, el, wr
            J['fist' + side] = wr + D(sa + ea, f) * S['fist_r'] * 0.55 * sc
        for side, ha, ka, off in (('f', g('hf'), g('kf'), 1.4), ('b', g('hb'), g('kb'), -1.4)):
            h0 = hip + np.array([off * f * sc, 0], np.float32)
            kn = h0 + D(ha, f) * S['th'] * sc
            an = kn + D(ha - ka, f) * S['shn'] * sc
            toe = an + D(ha - ka + 90 + g('foot'), f) * S['foot'] * sc
            J['hp' + side], J['kn' + side], J['an' + side], J['toe' + side] = h0, kn, an, toe
        J['up'] = up
        J['front'] = front
        r = g('rot')
        if r:
            rr = r * f
            for k in list(J.keys()):
                if k in ('up', 'front'):
                    J[k] = rot(J[k], rr)
                else:
                    J[k] = rot(J[k], rr)
        if self.air:
            off = np.array([self.x, self.y], np.float32)
        else:
            low = max(J['anf'][1], J['anb'][1], J['toef'][1], J['toeb'][1]) + S['foot_h'] * 0.6 * sc
            off = np.array([self.x, self.y - low - g('lift')], np.float32)
        if self.shake:
            off = off + np.array([vnoise(np.array([self.t * 60]), np.array([1.0]))[0] - 0.5,
                                  vnoise(np.array([self.t * 60]), np.array([7.0]))[0] - 0.5]) * self.shake * 2
        for k in J:
            if k not in ('up', 'front'):
                J[k] = J[k] + off
        self._J = J
        return J

    def point(self, name):
        J = self._J if self._J is not None else self.joints()
        return J[name]

    # ------------------------------------------------------------ physics
    def step(self, t, wind=(0, 0), dt=DT):
        self.t = t
        J = self.joints()
        sc = self.S['scale']
        f = self.f
        wx, wy = wind
        if self.cape is not None:
            anc = J['neck'] - J['front'] * 3.0 * sc + J['up'] * -1.0
            n = self.cape.n
            i = np.arange(n, dtype=np.float32)
            flutter = np.sin(t * 13.0 - i * 0.8)[:, None] * np.array([0.0, 1.0]) * (abs(wx) * 0.55 + 60)
            gust = 1 + 0.4 * np.sin(t * 5.3 + i * 0.3)
            force = np.stack([np.full(n, wx) * gust, np.full(n, 420.0 + wy)], -1) + flutter
            self.cape.step(anc, force, rest=None, dt=dt)
        for k, ch in enumerate(self.hair):
            base_ang = [-0.25, 0.05, 0.35, -0.6, 0.6][k]
            anc = J['head'] + (J['front'] * -0.55 + J['up'] * (0.45 - k * 0.12)) * self.S['head_r'] * sc
            rest = -J['front'] * 1.0 + J['up'] * base_ang
            n = ch.n
            i = np.arange(n, dtype=np.float32)
            flutter = np.sin(t * 11.0 - i * 1.1 + k)[:, None] * np.array([0.0, 1.0]) * (abs(wx) * 0.4 + 40)
            force = np.stack([np.full(n, wx), np.full(n, 150.0 + wy)], -1) + flutter
            ch.step(anc, force, rest=rest, dt=dt)

    def settle(self, steps=90, wind=(0, 0)):
        for i in range(steps):
            self.step(-steps * DT + i * DT, wind)

    # ------------------------------------------------------------ drawing
    def render(self, cam, light=(-0.55, -0.83), ambient=(1, 1, 1), rims=(), outline=True):
        J = self.joints()
        S, f = self.S, self.f
        sc = S['scale'] * cam.zoom
        P = lambda k: cam.pts(J[k])
        rgb = np.zeros((H, W, 3), np.float32)
        A = np.zeros((H, W), bool)
        E = np.zeros((H, W, 3), np.float32)
        L = np.asarray(light, np.float32)
        L = L / (np.hypot(*L) + 1e-6)

        def put(mask, base, rpx=3.0, line=True, emis=None, hi=True, shade=True):
            if not mask.any():
                return
            base = np.asarray(base, np.float32)
            k = max(1, int(round(rpx * 0.42)))
            if line:
                edge = mask & ~erode(mask) & A
            rgb[mask] = base
            if shade:
                sh = mask & ~shift(mask, int(round(-L[0] * k)), int(round(-L[1] * k)))
                rgb[sh] = base * np.array([0.60, 0.56, 0.74], np.float32)
                if hi:
                    kh = max(1, k // 2)
                    hl = mask & ~shift(mask, int(round(L[0] * kh)), int(round(L[1] * kh))) & ~sh
                    rgb[hl] = np.minimum(base * 1.14 + 0.07, 1.0)
            if line:
                rgb[edge] = base * np.array([0.3, 0.26, 0.4], np.float32)
            if emis is not None:
                E[mask] += np.asarray(emis, np.float32)
            else:
                E[mask] = 0
            A[mask] = True

        def limb(a, b, ra, rb, col, **kw):
            m = MaskDraw().capsule(P(a), P(b), ra * sc, rb * sc).get()
            put(m, col, rpx=(ra + rb) * 0.5 * sc, **kw)
            return m

        def shape(pts_world, col, rpx=4, **kw):
            m = MaskDraw().poly(cam.pts(np.asarray(pts_world, np.float32))).get()
            put(m, col, rpx=rpx * sc, **kw)
            return m

        up, fr = J['up'], J['front']
        hw_t, hw_b = S['torso_hw']
        sco = S['scale']
        dark = 0.72
        burst = self.form == 'burst'

        # ---------- behind-body layers
        if self.cape is not None:
            self._draw_cape(cam, put)
        for k, ch in enumerate(self.hair):
            if ch.p is None:
                continue
            pts = cam.pts(ch.p)
            n = len(pts)
            left, right = [], []
            for i in range(n):
                a = pts[max(0, i - 1)]
                b = pts[min(n - 1, i + 1)]
                d = b - a
                d = d / (np.hypot(*d) + 1e-6)
                nn = np.array([-d[1], d[0]])
                w = (3.4 - 3.2 * i / (n - 1)) * sc * (1.0 if k < 3 else 0.8)
                left.append(pts[i] + nn * w)
                right.append(pts[i] - nn * w)
            tip = pts[-1] + (pts[-1] - pts[-2]) * 0.6
            poly = left + [tip] + right[::-1]
            m = MaskDraw().poly(poly).get()
            hc = S['hair'] * (0.8 if k % 2 else 1.0)
            if burst:
                put(m, S['hair2'] * 0.9, rpx=3 * sc, emis=S['hair2'] * 0.15)
            else:
                put(m, hc, rpx=3 * sc)

        # ---------- back limbs
        if self.kind == 'saitama':
            suit, glove = S['suit'], S['glove']
            limb('shb', 'elb', *S['ua_r'], suit * dark)
            limb('elb', 'wrb', *S['fa_r'], glove * dark)
            self._cuff(cam, put, J['elb'], J['wrb'], 3.5, glove * dark, sc)
            limb('wrb', 'fistb', S['fist_r'] * 0.9, S['fist_r'], glove * dark)
            limb('hpb', 'knb', *S['th_r'], suit * dark)
            limb('knb', 'anb', *S['sh_r'], glove * dark)
            self._cuff(cam, put, J['knb'], J['anb'], 3.9, glove * dark, sc)
            limb('anb', 'toeb', S['foot_h'] * 0.62, S['foot_h'] * 0.5, glove * dark)
        else:
            body = S['burst'] if burst else S['suit']
            arm = S['burst'] if burst else S['armor']
            skin = S['burst'] if burst else S['skin']
            if not burst:
                self._pauldron(cam, put, J['shb'], up, fr, S['armor'] * dark, sc)
            limb('shb', 'elb', *S['ua_r'], body * dark)
            limb('elb', 'wrb', *S['fa_r'], arm * dark)
            limb('wrb', 'fistb', S['fist_r'] * 0.9, S['fist_r'], (skin if burst else arm) * dark)
            limb('hpb', 'knb', *S['th_r'], body * dark)
            limb('knb', 'anb', *S['sh_r'], arm * dark)
            limb('anb', 'toeb', S['foot_h'] * 0.62, S['foot_h'] * 0.5, arm * dark)

        # ---------- torso
        neck, hip = J['neck'], J['hip']
        tor = [neck + fr * hw_t * sco, neck - fr * hw_t * sco, hip - fr * hw_b * sco, hip + fr * hw_b * sco]
        if self.kind == 'saitama':
            shape(tor, S['suit'], rpx=hw_t)
            m = MaskDraw().circle(*P('shoulder'), hw_t * 0.75 * sc).get()
            put(m, S['suit'], rpx=hw_t * sc, line=False)
            belt = [hip + up * 3.6 + fr * hw_b * 1.08 * sco, hip + up * 3.6 - fr * hw_b * 1.08 * sco,
                    hip + up * 0.6 - fr * hw_b * 1.08 * sco, hip + up * 0.6 + fr * hw_b * 1.08 * sco]
            shape(belt, S['belt'], rpx=2)
            bk = cam.pts(hip + up * 2.1 + fr * hw_b * 0.95 * sco)
            m = MaskDraw().circle(bk[0], bk[1], 1.4 * sc).get()
            put(m, S['buckle'], rpx=1.4 * sc, line=False)
        else:
            if burst:
                shape(tor, S['burst'], rpx=hw_t)
            else:
                shape(tor, S['suit'], rpx=hw_t)
                plate = [neck + fr * hw_t * 1.12 * sco + up * 1.0, neck - fr * hw_t * 1.05 * sco + up * 1.0,
                         hip + up * S['torso'] * 0.42 * sco - fr * hw_t * 0.95 * sco,
                         hip + up * S['torso'] * 0.30 * sco + fr * hw_b * 1.3 * sco]
                shape(plate, S['armor'], rpx=hw_t)
                gm = cam.pts(neck - up * 6.5 * sco + fr * hw_t * 0.62 * sco)
                m = MaskDraw().circle(gm[0], gm[1], 1.8 * sc).get()
                put(m, S['gem'], rpx=2 * sc, line=False, emis=S['gem'] * 0.9)
            belt = [hip + up * 4.4 + fr * hw_b * 1.1 * sco, hip + up * 4.4 - fr * hw_b * 1.1 * sco,
                    hip + up * 0.8 - fr * hw_b * 1.1 * sco, hip + up * 0.8 + fr * hw_b * 1.1 * sco]
            shape(belt, S['armor'] if not burst else S['burst'] * 0.7, rpx=2)

        # ---------- front leg
        if self.kind == 'saitama':
            limb('hpf', 'knf', *S['th_r'], S['suit'])
            limb('knf', 'anf', *S['sh_r'], S['glove'])
            self._cuff(cam, put, J['knf'], J['anf'], 3.9, S['glove'], sc)
            limb('anf', 'toef', S['foot_h'] * 0.62, S['foot_h'] * 0.5, S['glove'])
        else:
            body = S['burst'] if burst else S['suit']
            arm = S['burst'] if burst else S['armor']
            limb('hpf', 'knf', *S['th_r'], body)
            limb('knf', 'anf', *S['sh_r'], arm)
            limb('anf', 'toef', S['foot_h'] * 0.62, S['foot_h'] * 0.5, arm)

        # ---------- head
        hr = S['head_r']
        skin = S['skin'] if not burst else (S['skin'] * 0.75 + S['burst'] * 0.25)
        limb('neck', 'head', 2.6 if self.kind == 'saitama' else 3.4, 2.6, skin * 0.9, line=False)
        hc = P('head')
        R = hr * sc
        if self.kind == 'saitama':
            m = MaskDraw().ellipse(hc[0], hc[1], R * 0.97, R * 1.04).get()
            put(m, skin, rpx=R)
        else:
            m = MaskDraw().circle(hc[0], hc[1], R).get()
            put(m, skin, rpx=R)
            self._crown(cam, put, hc, R, f, burst)
        self._face(hc, R, f, rgb, E, A, L)

        # ---------- front arm
        if self.kind == 'saitama':
            limb('shf', 'elf', *S['ua_r'], S['suit'])
            limb('elf', 'wrf', *S['fa_r'], S['glove'])
            self._cuff(cam, put, J['elf'], J['wrf'], 3.5, S['glove'], sc)
            limb('wrf', 'fistf', S['fist_r'] * 0.9, S['fist_r'], S['glove'])
        else:
            body = S['burst'] if burst else S['suit']
            arm = S['burst'] if burst else S['armor']
            limb('shf', 'elf', *S['ua_r'], body)
            limb('elf', 'wrf', *S['fa_r'], arm)
            limb('wrf', 'fistf', S['fist_r'] * 0.9, S['fist_r'], S['skin'] * 0.8 if burst else arm)
            if not burst:
                self._pauldron(cam, put, J['shf'], up, fr, S['armor'], sc)

        # ---------- glowing burst markings
        if self.kind == 'boros' and burst:
            md = MaskDraw()
            lw = max(1, int(round(0.7 * sc)))
            for a, b in (('shf', 'elf'), ('elf', 'wrf'), ('hpf', 'knf'), ('knf', 'anf'), ('shb', 'elb'),
                         ('hpb', 'knb')):
                md.line([P(a), P(b)], lw)
            c1 = cam.pts(neck - fr * 2 * sco)
            c2 = cam.pts(hip + up * 8 + fr * 2 * sco)
            c3 = cam.pts(hip + up * 8 - fr * 3 * sco)
            md.line([c1, c2], lw).line([c1, c3], lw)
            m = md.get() & A
            rgb[m] = S['lines']
            E[m] += S['lines'] * 0.7

        # ---------- damage: hole / dissolve / cracks
        if self.hole is not None:
            dx, dy, r = self.hole
            c = cam.pts(J['hip'] + np.array([dx * f, dy]))
            rr = r * cam.zoom
            if rr > 0.6:
                ang = np.arctan2(YY - c[1], XX - c[0])
                dist = np.hypot(XX - c[0], YY - c[1])
                wob = vnoise(np.cos(ang) * 2 + 5, np.sin(ang) * 2 + 5, 3) * 0.55 + 0.7
                hm = dist < rr * wob
                rim = dilate(hm, max(1, int(rr * 0.15))) & ~hm & A
                A[hm] = False
                rgb[rim] = hexc('#ff5a7a')
                E[rim] += hexc('#ff2a6a') * 1.0
        if self.cracks > 0 and self.kind == 'boros' and not burst:
            c = cam.pts(J['neck'] - up * 7)
            md = MaskDraw()
            rs = np.random.RandomState(4)
            for i in range(6):
                a0 = rs.uniform(0, 6.28)
                pts = [c]
                cur = np.array(c, np.float32)
                for j in range(4):
                    a0 += rs.uniform(-0.8, 0.8)
                    cur = cur + np.array([math.cos(a0), math.sin(a0)]) * 2.8 * sc * self.cracks
                    pts.append(cur.copy())
                md.line(pts, 1)
            m = md.get() & A
            rgb[m] = hexc('#ffe0ff')
            E[m] += hexc('#ff4ad8') * 0.9
        dmask = None
        if self.dissolve is not None:
            ox, oy, prog, Rw = self.dissolve
            c = cam.pts(np.array([ox, oy], np.float32))
            dist = np.hypot(XX - c[0], YY - c[1]) / (Rw * cam.zoom)
            nz = vnoise(XX * 0.35, YY * 0.35, 11) * 0.35
            field = dist + nz
            gone = field < prog
            band = (field < prog + 0.08) & ~gone & A
            A[gone] = False
            rgb[band] = hexc('#fff0c0')
            E[band] += hexc('#ff9a3a') * 2.0
            dmask = band

        # ---------- outline, lighting
        if outline:
            ol = dilate(A) & ~A
            rgb[ol] = OUTLINE
            A = A | ol
        rgb *= np.asarray(ambient, np.float32)
        for (d, col, inten, wpx) in rims:
            d = np.asarray(d, np.float32)
            d = d / (np.hypot(*d) + 1e-6)
            wpx = max(1, int(wpx))
            rim = A & ~shift(A, int(round(d[0] * wpx)), int(round(d[1] * wpx)))
            rgb[rim] += np.asarray(col, np.float32) * inten
            E[rim] += np.asarray(col, np.float32) * inten * 0.25
        self.last_band = dmask
        return rgb, A, E

    # ------------------------------------------------------------ pieces
    def _draw_cape(self, cam, put):
        ch = self.cape
        if ch.p is None:
            return
        sc = self.S['scale'] * cam.zoom
        n = ch.n - int(round(self.tatter * 2.5))
        pts = cam.pts(ch.p[:n])
        left, right = [], []
        for i in range(n):
            a = pts[max(0, i - 1)]
            b = pts[min(n - 1, i + 1)]
            d = b - a
            d = d / (np.hypot(*d) + 1e-6)
            nn = np.array([-d[1], d[0]])
            w = (2.4 + 6.0 * (i / (n - 1)) ** 0.8) * sc
            left.append(pts[i] + nn * w)
            right.append(pts[i] - nn * w)
        teeth = []
        a, b = left[-1], right[-1]
        tn = 5
        rs = np.random.RandomState(7)
        d = pts[-1] - pts[-2]
        d = d / (np.hypot(*d) + 1e-6)
        for j in range(tn + 1):
            q = a + (b - a) * (j / tn)
            amp = (1.5 + self.tatter * 5 * rs.uniform(0.3, 1.0)) * sc if j % 2 else 0
            teeth.append(q + d * amp)
        poly = left + teeth + right[::-1]
        m = MaskDraw().poly(poly).get()
        put(m, self.S['cape'], rpx=4 * sc)

    def _cuff(self, cam, put, a, b, r, col, sc):
        a = np.asarray(a)
        b = np.asarray(b)
        p0 = cam.pts(a + (b - a) * 0.08)
        p1 = cam.pts(a + (b - a) * 0.3)
        m = MaskDraw().capsule(p0, p1, r * sc, r * 0.92 * sc).get()
        put(m, col, rpx=r * sc)

    def _pauldron(self, cam, put, sh, up, fr, col, sc):
        c = cam.pts(sh + up * 1.5)
        r = 5.8 * sc
        u = cam.pts(sh + up * 10) - c
        u = u / (np.hypot(*u) + 1e-6)
        n = np.array([-u[1], u[0]])
        pts = []
        for i in range(14):
            a = i / 14 * 2 * math.pi
            rr = r * (1.0 if i not in (2, 4, 10, 12) else 1.45)
            v = u * math.cos(a) * rr * 0.9 + n * math.sin(a) * rr * 1.1
            pts.append(c + v)
        m = MaskDraw().poly(pts).get()
        put(m, col, rpx=r)

    def _crown(self, cam, put, hc, R, f, burst):
        S = self.S
        pts = []
        spikes = [(-160, 1.25), (-135, 1.7), (-112, 1.95), (-90, 1.8), (-70, 1.55), (-48, 1.25), (-25, 1.0)]
        for i, (a, l) in enumerate(spikes):
            a = math.radians(a if f > 0 else -180 - a)
            a2 = a + math.radians(12 if f > 0 else -12)
            pts.append(hc + np.array([math.cos(a - 0.2 * f), math.sin(a - 0.2 * f)]) * R * 0.95)
            pts.append(hc + np.array([math.cos(a2), math.sin(a2)]) * R * l)
        pts.append(hc + np.array([math.cos(math.radians(-10 if f > 0 else -170)),
                                  math.sin(math.radians(-10))]) * R * 0.9)
        pts.append(hc + np.array([-0.9 * f, 0.4]) * R)
        m = MaskDraw().poly(pts).get()
        if burst:
            put(m, S['hair2'] * 0.9, rpx=R, emis=S['hair2'] * 0.12)
        else:
            put(m, S['hair'], rpx=R)

    def _face(self, hc, R, f, rgb, E, A, L):
        md = lambda: MaskDraw()
        dark = OUTLINE
        if self.kind == 'saitama':
            eye = hc + np.array([0.45 * R * f, -0.02 * R])
            mo = hc + np.array([0.6 * R * f, 0.46 * R])
            if self.expr == 'serious' and R >= 9:
                al = [eye + np.array([-0.2 * R * f, 0.02 * R]), eye + np.array([0.2 * R * f, -0.05 * R]),
                      eye + np.array([0.12 * R * f, 0.08 * R])]
                m = md().poly(al).get()
                rgb[m] = 1.0
                m = md().circle(eye[0] + 0.04 * R * f, eye[1], max(0.5, 0.05 * R)).get()
                rgb[m] = dark
                br = [eye + np.array([-0.3 * R * f, -0.22 * R]), eye + np.array([0.28 * R * f, -0.1 * R])]
                m = md().line(br, max(1, 0.1 * R)).get()
                rgb[m] = dark
                m = md().line([mo - np.array([0.14 * R * f, -0.02 * R]), mo + np.array([0.1 * R * f, 0.04 * R])],
                              1).get()
                rgb[m] = dark * 1.5
            elif self.expr == 'serious':
                m = md().line([eye - np.array([0.1 * R * f, 0]), eye + np.array([0.15 * R * f, -0.05 * R])],
                              1).get()
                rgb[m] = dark
                m = md().line([eye + np.array([-0.2 * R * f, -0.22 * R]), eye + np.array([0.2 * R * f, -0.14 * R])],
                              1).get()
                rgb[m] = dark
            elif self.expr == 'shout':
                w = max(1, 0.1 * R)
                m = md().line([eye + np.array([-0.16 * R * f, -0.12 * R]), eye + np.array([0.1 * R * f, 0]),
                               eye + np.array([-0.16 * R * f, 0.12 * R])], w).get()
                rgb[m] = dark
                mc = hc + np.array([0.55 * R * f, 0.45 * R])
                m = md().ellipse(mc[0], mc[1], 0.26 * R, 0.3 * R).get()
                rgb[m] = dark
                m2 = md().ellipse(mc[0], mc[1] + 0.14 * R, 0.16 * R, 0.12 * R).get() & m
                rgb[m2] = hexc('#e0404a')
            else:
                m = md().ellipse(eye[0], eye[1], max(0.5, 0.07 * R), max(0.9, 0.13 * R)).get()
                rgb[m] = dark
                if R >= 7:
                    m = md().line([mo - np.array([0.1 * R * f, 0]), mo + np.array([0.08 * R * f, 0])], 1).get()
                    rgb[m] = dark * 2.0
            sp = hc + np.asarray(L) * 0.58 * R + np.array([0.08 * R * f, 0])
            m = md().ellipse(sp[0], sp[1], max(0.6, 0.16 * R), max(0.5, 0.1 * R)).get()
            rgb[m] = 1.0
            E[m] += 0.35
        else:
            eye = hc + np.array([0.36 * R * f, -0.12 * R])
            er = 0.3 * R
            if R < 7:
                m = md().circle(eye[0], eye[1], 0.6).get()
                rgb[m] = self.S['eye']
                E[m] += self.S['eye'] * 1.5
            else:
                m = md().ellipse(eye[0], eye[1], er * 1.08, max(0.5, er * self.eye_open)).get()
                ring = dilate(m) & ~m
                rgb[ring] = dark
                rgb[m] = hexc('#fff4de')
                if self.eye_open > 0.2:
                    m2 = md().circle(eye[0] + er * 0.25 * f, eye[1], er * 0.58).get() & m
                    rgb[m2] = self.S['eye']
                    E[m2] += self.S['eye'] * (1.4 if self.form == 'burst' else 0.8)
                    m3 = md().ellipse(eye[0] + er * 0.3 * f, eye[1], max(0.5, er * 0.16), er * 0.45).get() & m
                    rgb[m3] = dark
            mo = hc + np.array([0.52 * R * f, 0.5 * R])
            w = 0.34 * R if self.expr != 'grin' else 0.46 * R
            m = md().line([mo - np.array([w * f, 0.05 * R]), mo + np.array([w * 0.6 * f, -0.08 * R])],
                          max(1, 0.1 * R)).get()
            rgb[m] = dark
            if R >= 10:
                for j in range(3):
                    tx = mo + np.array([(-w + j * w * 0.6) * f, -0.02 * R])
                    m = md().poly([tx, tx + np.array([0.12 * R * f, 0]), tx + np.array([0.06 * R * f, 0.16 * R])]).get()
                    rgb[m] = 1.0


def draw(frame, layer, alpha=1.0, tint=None):
    rgb, A, E = layer
    if tint is not None:
        c = np.asarray(tint, np.float32)
        if alpha >= 1:
            frame.img[A] = c
        else:
            frame.img[A] = frame.img[A] * (1 - alpha) + c * alpha
        frame.glow[A] += c * 0.25 * alpha
        return
    if alpha >= 1:
        frame.img[A] = rgb[A]
        frame.glow[A] *= 0.15
    else:
        frame.img[A] = frame.img[A] * (1 - alpha) + rgb[A] * alpha
    frame.glow += E * alpha


def aura(frame, A, color, t, strength=1.0, height=9, seed=0, core=(1, 1, 1)):
    """Pixel flame aura rising off a silhouette mask."""
    if strength <= 0:
        return
    xs = np.arange(W, dtype=np.float32)
    tongue = (0.25 + 0.75 * vnoise(xs * 0.22, np.full(W, t * 9.0), seed)) * height
    acc = np.zeros((H, W), np.float32)
    for k in range(1, height + 1):
        m = shift(A, 0, k)
        acc += m * (tongue[None, :] > k) * (1 - k / (height + 1))
    halo = dilate(A, 2) & ~A
    acc = np.clip(acc + halo * 0.5, 0, 1)
    acc[A] = 0
    q = np.floor(acc * 3 + BAYER * 0.99) / 3
    col = np.asarray(color, np.float32)
    frame.glow += q[..., None] * col * strength
    inner = (q > 0.66)
    frame.glow[inner] += np.asarray(core, np.float32) * 0.5 * strength
