"""Pixel stickmen: Saitama (cream stick, red fists, tiny cape) and Boros (violet stick, spikes, one red eye)."""
import math

import numpy as np

from engine import W, H, XX, YY, MaskDraw, dilate, shift, hexc, vnoise
from fighters import Fighter, SPECS, POSES, pose, Chain, OUTLINE

SPECS['stick_s'] = dict(scale=1.0, head_r=5.8, neck=2.2, torso=19.0, sh_drop=2.2, ua=10.5, fa=10.5, fist_r=2.2,
                        th=14.0, shn=14.0, foot=4.5, foot_h=1.6)
SPECS['stick_b'] = dict(SPECS['stick_s'], scale=1.12, head_r=6.0)

POSES.update({
    'block_up': dict(lean=0, head=10, sf=168, ef=70, sb=158, eb=80, hf=40, kf=60, hb=-35, kb=45),
    'tuck': dict(lean=15, head=-10, sf=60, ef=90, sb=40, eb=100, hf=110, kf=140, hb=90, kb=140),
    'hammer_up': dict(lean=-20, head=-10, sf=175, ef=15, sb=170, eb=20, hf=30, kf=60, hb=-20, kb=50),
    'hammer_down': dict(lean=45, head=-20, sf=75, ef=0, sb=70, eb=5, hf=20, kf=40, hb=-30, kb=50),
    'bored': dict(lean=-2, head=16, sf=4, ef=6, sb=-4, eb=8, hf=4, kf=2, hb=-4, kb=2),
    'stance': dict(lean=10, head=-4, sf=55, ef=85, sb=35, eb=95, hf=26, kf=30, hb=-24, kb=16),
})

COLS = {
    'saitama': dict(body=hexc('#fff2cc'), fist=hexc('#ff3a30'), cape=hexc('#f4f4ff'), glow=hexc('#ffd070')),
    'boros': dict(body=hexc('#b06cff'), fist=hexc('#c890ff'), hair=hexc('#46c8ff'), eye=hexc('#ff2a2a'),
                  glow=hexc('#ff4ad8')),
}


class Stick(Fighter):
    def __init__(self, who, x=0.0, f=1):
        super().__init__('stick_s' if who == 'saitama' else 'stick_b', x, 0.0, f)
        self.who = who
        self.C = COLS[who]
        self.cape = Chain(7, 3.4, damp=0.93) if who == 'saitama' else None
        self.hair = []
        self.burst = False
        self.visible = True

    def render(self, cam, rims=(), outline=True, ambient=None):
        J = self.joints()
        z = cam.zoom * self.S['scale']
        r = max(0.9, 1.45 * z)
        P = lambda k: cam.pts(J[k])
        rgb = np.zeros((H, W, 3), np.float32)
        A = np.zeros((H, W), bool)
        E = np.zeros((H, W, 3), np.float32)
        C = self.C
        body = np.minimum(C['body'] * (1.2 if self.burst else 1.0), 1)
        f = self.f

        def fill(m, c, e=0.0):
            rgb[m] = c
            A[m] = True
            if e:
                E[m] += np.asarray(c) * e

        # cape ribbon
        if self.cape is not None and self.cape.p is not None:
            pts = cam.pts(self.cape.p)
            n = len(pts)
            left, right = [], []
            for i in range(n):
                a, b = pts[max(0, i - 1)], pts[min(n - 1, i + 1)]
                d = (b - a) / (np.hypot(*(b - a)) + 1e-6)
                nn = np.array([-d[1], d[0]])
                w = (0.8 + 2.6 * i / (n - 1)) * z
                left.append(pts[i] + nn * w)
                right.append(pts[i] - nn * w)
            fill(MaskDraw().poly(left + right[::-1]).get(), C['cape'] * 0.9, 0.05)
        hc = P('head')
        R = self.S['head_r'] * z
        # boros hair spikes (behind head)
        if self.who == 'boros':
            md = MaskDraw()
            for a, l in ((168, 1.8), (188, 2.3), (208, 2.6), (230, 2.3), (252, 1.8), (272, 1.4)):
                a = a if f > 0 else 180 - a
                ar = math.radians(a)
                sp = 0.38
                b1 = hc + np.array([math.cos(ar - sp), math.sin(ar - sp)]) * R * 0.8
                b2 = hc + np.array([math.cos(ar + sp), math.sin(ar + sp)]) * R * 0.8
                tip = hc + np.array([math.cos(ar), math.sin(ar)]) * R * l
                md.poly([b1, tip, b2])
            fill(md.get(), C['hair'], 0.45 if self.burst else 0.12)
        # back limbs
        md = MaskDraw()
        for a, b in (('shb', 'elb'), ('elb', 'wrb'), ('hpb', 'knb'), ('knb', 'anb'), ('anb', 'toeb')):
            md.capsule(P(a), P(b), r * 0.95, r * 0.95)
        fill(md.get(), body * 0.6, 0.05)
        fr = max(1.3, self.S['fist_r'] * z * 0.8)
        fill(MaskDraw().circle(*P('fistb'), fr).get(), C['fist'] * 0.62, 0.1)
        # body
        md = MaskDraw()
        for a, b in (('hip', 'neck'), ('neck', 'head'), ('hpf', 'knf'), ('knf', 'anf'), ('anf', 'toef'), ('shf', 'elf'),
                     ('elf', 'wrf')):
            md.capsule(P(a), P(b), r, r)
        md.circle(hc[0], hc[1], R)
        fill(md.get(), body, 0.35 if self.burst else 0.12)
        fill(MaskDraw().circle(*P('fistf'), fr).get(), C['fist'], 0.25)
        if self.who == 'saitama':
            sp = hc + np.array([-0.35 * R, -0.45 * R])
            m = MaskDraw().circle(sp[0], sp[1], max(0.5, 0.18 * R)).get()
            rgb[m] = 1.0
            E[m] += 0.6
        else:
            e = hc + np.array([0.38 * R * f, -0.08 * R])
            m = MaskDraw().circle(e[0], e[1], max(0.6, 0.24 * R)).get()
            rgb[m] = C['eye']
            E[m] += C['eye'] * 2.0
        # dissolve
        if self.dissolve is not None:
            ox, oy, prog, Rw = self.dissolve
            c = cam.w2s(ox, oy)
            field = np.hypot(XX - c[0], YY - c[1]) / (Rw * cam.zoom) + vnoise(XX * 0.35, YY * 0.35, 11) * 0.3
            gone = field < prog
            band = (field < prog + 0.1) & ~gone & A
            A[gone] = False
            rgb[band] = hexc('#fff0c0')
            E[band] += hexc('#ff9a3a') * 2.0
        if outline:
            ol = dilate(A) & ~A
            rgb[ol] = OUTLINE
            A = A | ol
        for (d, col, inten, wpx) in rims:
            d = np.asarray(d, np.float32)
            d = d / (np.hypot(*d) + 1e-6)
            rim = A & ~shift(A, int(round(d[0] * wpx)), int(round(d[1] * wpx)))
            rgb[rim] += np.asarray(col, np.float32) * inten
            E[rim] += np.asarray(col, np.float32) * inten * 0.3
        return rgb, A, E
