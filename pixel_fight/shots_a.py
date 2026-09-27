"""Storyboard part 1: intro -> VS -> first exchange -> normal punch -> regeneration -> meteoric burst."""
import math

import numpy as np

from engine import *
from fighters import Fighter, pose, draw, aura
from fx import *
from backgrounds import City, NIGHT, BURST, mix_pal, bg_radial
from shotbase import *


# =====================================================================  S01
class S01_Establish(Shot):
    dur = 4.0
    SFX = [(0.0, 'wind', 0.7), (0.0, 'drone', 0.7), (2.9, 'boom_soft', 0.5)]

    def setup(self):
        self.city = City(3)
        self.sai = Fighter('saitama', -72, 0, 1)
        self.bor = Fighter('boros', 80, 0, -1)
        for f in (self.sai, self.bor):
            f.settle(wind=(-160, 0))
        self.camy = Track([(0, -250), (0.5, -250), (3.4, -52, 'io'), (4.0, -50)])
        self.camz = Track([(0, 0.92), (4.0, 1.12, 'io')])

    def frame(self, t):
        cam = self.cam
        cam.x = -6 + 4 * t
        cam.y = self.camy(t)
        cam.zoom = self.camz(t)
        cam.tick()
        fr = Frame(cam)
        self.city.draw(fr, t, NIGHT, ship=1.0)
        if self.every(2):
            self.embers(2)
        self.fx_step(ground=None)
        amb = (0.55, 0.55, 0.78)
        for f in (self.bor, self.sai):
            f.step(t, (-170, 0))
        lb = self.bor.render(cam, ambient=amb, rims=[((-1, -0.1), NIGHT['fire'], 0.7, 1)])
        draw(fr, lb)
        draw(fr, self.sai.render(cam, ambient=amb, rims=[((1, -0.2), NIGHT['fire'], 0.7, 1)]))
        self.fx_draw(fr)
        self.city.draw_fg(fr)
        fade = 1 - remap(t, 0, 0.8)

        def ui(o):
            letterbox(o, LB)
            if t > 1.2:
                s = 'Z市 · 废墟区'
                n = int((t - 1.2) * 12)
                blit_text(o, s[:n], 14, LB + 14, color=(1, 0.85, 0.5), anchor='l')
                if t > 2.0:
                    blit_text(o, 'CITY-Z  RUINS', 15, LB + 30, color=(0.7, 0.7, 0.9), path=FONT_BOLD, size=8,
                              anchor='l', spacing=1, alpha=remap(t, 2.0, 2.3))
            o *= (1 - fade)
        fr.ui.append(ui)
        return fr.finish(vignette=0.5)


# =====================================================================  S02
class S02_BorosEye(Shot):
    dur = 2.5
    SFX = [(0.0, 'drone', 0.5), (1.0, 'eye', 0.9), (1.22, 'boom', 0.8)]

    def setup(self):
        b = self.bor = Fighter('boros', 0, 0, -1)
        b.pose = pose('boros_idle', head=-8)
        b.eye_open = 0.05
        b.settle(wind=(150, -20))
        self.zoom = Track([(0, 4.0), (1.1, 4.5, 'io'), (1.32, 7.2, 'expo'), (2.5, 7.6)])
        self.eye = Track([(0, 0.06), (0.95, 0.06), (1.08, 1.0, 'out')])
        self.dust = Particles()

    def frame(self, t):
        b, cam = self.bor, self.cam
        b.eye_open = self.eye(t)
        J = b.joints()
        R = b.S['head_r'] * b.S['scale']
        eye = J['head'] + np.array([0.36 * R * b.f, -0.12 * R])
        k = ease_io(remap(t, 1.1, 1.4))
        tgt = J['head'] * (1 - k) + eye * k
        cam.x = tgt[0] + 8 * (1 - k) - 2 * t
        cam.y = tgt[1] + 2
        cam.zoom = self.zoom(t)
        if self.ev(1.22):
            cam.add_trauma(0.7)
        cam.tick()
        fr = Frame(cam)
        bg_radial(fr, W * 0.7, H * 0.35, hexc('#3a1838'), hexc('#07040f'), r=300)
        if self.every(3):
            self.parts.emit(np.array([[cam.x + self.rng.uniform(-40, 40), cam.y + 30]]),
                            np.array([[self.rng.uniform(-15, 15), -self.rng.uniform(4, 12)]]), 3.0,
                            1.0, (1, 0.5, 0.25), (0.4, 0.1, 0.1), mode=DOT)
        self.fx_step(None)
        self.parts.draw(fr)
        b.step(t, (160, -20))
        draw(fr, b.render(cam, light=(0.6, -0.8), ambient=(0.5, 0.5, 0.75),
                          rims=[((1, -0.5), hexc('#ff6a3a'), 0.8, 2), ((-1, 0), hexc('#6a6aff'), 0.4, 1)]))
        ex, ey = cam.w2s(*eye)
        if t > 0.98:
            p = remap(t, 0.98, 1.25)
            inten = (1.6 - 1.0 * p) * (0.8 + 0.2 * math.sin(t * 30))
            radial(fr.glow, ex, ey, 40 + 30 * p, hexc('#ff3a2a'), 0.8 * inten, steps=6)
            band = np.clip(1 - np.abs(XX[0] - ex) / (W * 0.8), 0, 1) ** 2
            y0 = int(ey)
            if 1 <= y0 < H - 1:
                fr.glow[y0, :] += band[:, None] * hexc('#ff5a4a') * 1.5 * inten
                fr.glow[y0 - 1:y0 + 2, :] += band[:, None] * hexc('#ff2a2a') * 0.5 * inten
        fr.ui.append(lambda o: letterbox(o, LB))
        fr.ui.append(lambda o: subtitle(o, '波罗斯', '预言中的强者……就是你吗？', t, 0.15, 2.3, color=(0.7, 0.8, 1)))
        return fr.finish(vignette=0.6)


# =====================================================================  S03
class S03_SaitamaFace(Shot):
    dur = 2.4
    SFX = [(0.0, 'wind', 0.5), (0.85, 'ting', 0.8)]

    def setup(self):
        s = self.sai = Fighter('saitama', 0, 0, 1)
        s.pose = pose('idle', head=2)
        s.settle(wind=(-170, 0))

    def frame(self, t):
        s, cam = self.sai, self.cam
        J = s.joints()
        cam.x = J['head'][0] - 6 + 3 * t
        cam.y = J['head'][1] + 10
        cam.zoom = 3.6 + 0.25 * t
        cam.tick()
        fr = Frame(cam)
        bg_radial(fr, W * 0.2, H * 0.7, hexc('#5a2a2a'), hexc('#0c0812'), r=320)
        if self.every(3):
            self.parts.emit(np.array([[cam.x + self.rng.uniform(-50, 50), cam.y + 30]]),
                            np.array([[self.rng.uniform(-20, 5), -self.rng.uniform(4, 12)]]), 3.0,
                            1.0, (1, 0.6, 0.3), (0.4, 0.1, 0.1), mode=DOT)
        self.fx_step(None)
        self.parts.draw(fr)
        s.step(t, (-180, 0))
        draw(fr, s.render(cam, ambient=(0.8, 0.75, 0.85), rims=[((-1, 0.2), hexc('#ff8a4a'), 0.6, 2)]))
        R = s.S['head_r'] * cam.zoom
        L = np.array([-0.55, -0.83])
        L = L / np.hypot(*L)
        hc = cam.w2s(*J['head'])
        sp = np.array(hc) + L * 0.58 * R + np.array([0.08 * R, 0])
        g = math.sin(math.pi * remap(t, 0.85, 1.25))
        fr.ui.append(lambda o: glint(o, None, sp[0], sp[1], 22, g))
        fr.ui.append(lambda o: letterbox(o, LB))
        fr.ui.append(lambda o: subtitle(o, '埼玉', '我只是个兴趣使然的英雄。', t, 0.2, 2.2))
        return fr.finish(vignette=0.6)


# =====================================================================  S04
class S04_Versus(Shot):
    dur = 2.9
    SFX = [(0.05, 'whoosh', 0.8), (0.12, 'whoosh', 0.7), (0.55, 'slam', 1.0), (0.85, 'whoosh', 0.4),
           (2.7, 'whoosh', 0.6)]

    def setup(self):
        self.sai = Fighter('saitama', 0, 0, 1)
        self.sai.pose = pose('guard')
        self.bor = Fighter('boros', 0, 0, -1)
        self.bor.pose = pose('guard')
        for f in (self.sai, self.bor):
            f.settle(wind=(-200 * f.f, 0))
        self.shake = Camera()

    def panel(self, t, ftr, bg_in, bg_out, rim, slide):
        cam = Camera(0, -48, 2.25)
        cam.x = (40 if ftr.f > 0 else -44) + slide
        cam.y = -44
        fr = Frame(cam)
        bg_radial(fr, W / 2 - (60 if ftr.f > 0 else -60), H / 2, bg_in, bg_out, r=260)
        ftr.step(t, (-240 * ftr.f, 0))
        L = ftr.render(cam, light=(0.4 * ftr.f, -0.9), rims=[((ftr.f, -0.3), rim, 0.7, 2)])
        return fr, L

    def frame(self, t):
        e1 = ease_out(remap(t, 0.0, 0.35), 4)
        e2 = ease_out(remap(t, 0.08, 0.43), 4)
        sh = self.shake
        if self.ev(0.55):
            sh.add_trauma(1.0)
        sh.tick()
        drift = t * 4
        frL, LL = self.panel(t, self.sai, hexc('#ffc23a'), hexc('#a02a10'), hexc('#fff0a0'),
                             (1 - e1) * 170 - drift)
        rng = np.random.RandomState(self.i // 2)
        draw(frL, LL)
        outL = frL.finish(raw=True, vignette=0.2)
        speed_lines(outL, rng, W * 0.25, H * 0.55, n=40, color=(1, 0.95, 0.7), alpha=0.35, r_in=(0.6, 0.9))
        frR, LR = self.panel(t, self.bor, hexc('#8a3aff'), hexc('#12052a'), hexc('#ff6ad8'),
                             -(1 - e2) * 170 + drift)
        aura(frR, LR[1], hexc('#b04aff'), t, 0.4)
        draw(frR, LR)
        outR = frR.finish(raw=True, vignette=0.2)
        speed_lines(outR, rng, W * 0.75, H * 0.45, n=40, color=(0.8, 0.6, 1), alpha=0.3, r_in=(0.6, 0.9))
        split = W / 2 + (H / 2 - YY) * 0.38 + sh.ox
        mL = XX < split
        out = np.where(mL[..., None], outL, outR)
        # glowing divider with electric jitter
        jit = (vnoise(YY[:, 0] * 0.3, np.full(H, t * 20), 4) - 0.5) * 4
        d = np.abs(XX - split - jit[:, None])
        k = remap(t, 0.3, 0.5)
        out[d < 1.2 * k] = 1.0
        out[(d >= 1.2) & (d < 3.5) & (k > 0)] = out[(d >= 1.2) & (d < 3.5) & (k > 0)] * 0.4 + np.array([1, 0.8, 1]) * 0.6
        # VS slam
        if t >= 0.4:
            p = ease_out(remap(t, 0.4, 0.55), 3)
            size = int(round(110 - 58 * p))
            blit_text(out, 'VS', W / 2 + sh.ox, H / 2 - 4 + sh.oy, color=(1, 1, 0.7), color2=(1, 0.3, 0.1),
                      outline=(0.08, 0.0, 0.05), path=FONT_BOLD, size=size, thick=2, shadow=3, glowc=(1, 0.6, 0.2))
        if t >= 0.55:
            a = 1 - remap(t, 0.55, 0.75)
            out[:] = out * (1 - a * 0.8) + a * 0.8
        # name plates
        for (side, cn, en, tag, col) in ((-1, '埼玉', 'SAITAMA', '兴趣使然的英雄', (1, 0.85, 0.2)),
                                         (1, '波罗斯', 'BOROS', '宇宙霸主', (0.8, 0.5, 1))):
            p = ease_out(remap(t, 0.75 if side < 0 else 0.85, 1.1 if side < 0 else 1.2), 4)
            if p <= 0:
                continue
            x = W / 2 + side * 118 + side * (1 - p) * 200
            y = H - 44 if side < 0 else 40
            blit_text(out, cn, x, y, color=(1, 1, 1), color2=col, outline=(0, 0, 0), path=FONT_CJK, size=24,
                      thick=2, shadow=2)
            blit_text(out, en, x, y + 18, color=col, outline=(0, 0, 0), path=FONT_BOLD, size=10, spacing=2)
            blit_text(out, tag, x, y + 32, color=(0.9, 0.9, 0.9), outline=(0, 0, 0), size=16, alpha=p)
        a = remap(t, 2.65, 2.9)
        out[:] = out * (1 - a) + a
        return quantize(out)


# =====================================================================  S05
class S05_Fight(Shot):
    dur = 1.9
    SFX = [(0.0, 'wind', 0.6), (0.15, 'ready', 0.8), (0.85, 'slam', 1.0), (0.85, 'fight', 1.0)]

    def setup(self):
        self.city = City(3)
        self.sai = Fighter('saitama', -62, 0, 1)
        self.sai.pose = pose('idle2')
        self.bor = Fighter('boros', 64, 0, -1)
        self.bor.pose = pose('guard')
        for f in (self.sai, self.bor):
            f.settle(wind=(-160, 0))

    def frame(self, t):
        cam = self.cam
        cam.x = 0
        cam.y = -48
        cam.zoom = 1.25 + 0.15 * ease_out(remap(t, 0.85, 1.9))
        if self.ev(0.85):
            cam.add_trauma(0.8)
            self.rings.add(0, -90, 240, dur=0.6, color=(1, 0.7, 0.3), width=10, squash=0.6)
        cam.tick()
        fr = Frame(cam)
        self.city.draw(fr, t + 4, NIGHT, ship=0.6)
        if self.every(2):
            self.embers(2)
        if self.every(4):
            self.parts.emit(np.array([[cam.x + 220, self.rng.uniform(-6, 0)]]), np.array([[-160, -8]]), 2.5, 7,
                            NIGHT['ground'] * 1.6, NIGHT['ground'], mode=DUST)
        self.fx_step(None)
        for f in (self.bor, self.sai):
            f.step(t, (-170, 0))
        amb = (0.72, 0.7, 0.88)
        draw(fr, self.bor.render(cam, ambient=amb, rims=[((-1, -0.2), NIGHT['fire'], 0.7, 1)]))
        draw(fr, self.sai.render(cam, ambient=amb, rims=[((1, -0.2), NIGHT['fire'], 0.7, 1)]))
        self.fx_draw(fr)

        def ui(o):
            letterbox(o, LB * (1 - ease_out(remap(t, 0.8, 1.1))))
            if 0.12 < t < 0.8:
                p = ease_out_back(remap(t, 0.12, 0.3))
                blit_text(o, 'READY', W / 2, H / 2 - 30 + (1 - p) * -40, color=(1, 1, 1), color2=(0.6, 0.8, 1),
                          outline=(0.05, 0.05, 0.2), path=FONT_BOLD, size=30, thick=2, shadow=2,
                          alpha=1 - remap(t, 0.65, 0.8))
            if t >= 0.85:
                p = ease_out(remap(t, 0.85, 0.97), 3)
                size = int(round(120 - 76 * p))
                a = 1 - remap(t, 1.6, 1.9)
                blit_text(o, 'FIGHT!', W / 2 + cam.ox, H / 2 - 22 + cam.oy, color=(1, 1, 0.6), color2=(1, 0.25, 0.1),
                          outline=(0.1, 0, 0.02), path=FONT_BOLD, size=size, thick=2, shadow=3, alpha=a,
                          glowc=(1, 0.5, 0.1))
            fl = 1 - remap(t, 0.85, 1.05)
            if t >= 0.85:
                o[:] = o * (1 - fl * 0.7) + fl * 0.7
        fr.ui.append(ui)
        return fr.finish()


# =====================================================================  S06
class S06_Rush(Shot):
    dur = 3.4
    SFX = [(0.0, 'wind', 0.4), (0.26, 'dash', 1.0), (0.42, 'swish', 0.6)] + \
          [(0.5 + k * 0.1, 'swish', 0.55) for k in range(19)] + [(2.42, 'whoosh', 0.9), (2.6, 'slowmo', 0.8)]

    def setup(self):
        self.city = City(3)
        s = self.sai = Fighter('saitama', -8, 0, 1)
        s.pose = pose('idle2')
        b = self.bor = Fighter('boros', 120, 0, -1)
        b.pose = pose('guard')
        for f in (s, b):
            f.settle(wind=(-160, 0))
        bk = [(0, pose('guard')), (0.12, pose('crouch')), (0.26, pose('crouch')), (0.3, pose('dash')),
              (0.42, pose('dash')), (0.46, pose('punch_wind'))]
        t0 = 0.5
        for k in range(19):
            pz = 'punch' if k % 2 == 0 else 'punch_b'
            var = dict(sf=90 + self.rng.uniform(-18, 14) if pz == 'punch' else -30,
                       sb=90 + self.rng.uniform(-18, 14) if pz == 'punch_b' else -30)
            bk.append((t0 + k * 0.1, pose(pz, **var)))
            bk.append((t0 + k * 0.1 + 0.05, pose('guard', lean=18)))
        bk += [(2.4, pose('punch_wind', lean=0)), (2.55, pose('punch', sf=100, lean=30)), (3.4, pose('punch', sf=102, lean=32))]
        self.bpose = Track([(a, p, 'snap') for a, p in bk])
        self.bx = Track([(0, 130), (0.26, 134), (0.44, 48, 'out'), (2.4, 46), (2.55, 38, 'out'), (3.4, 34)])
        sk = [(0, pose('idle2'))]
        seq = ['dodge_back', 'dodge_side', 'dodge_duck', 'dodge_side', 'dodge_back', 'dodge_duck']
        for k in range(10):
            sk.append((0.5 + k * 0.19, pose(seq[k % len(seq)])))
        sk += [(2.42, pose('dodge_side')), (2.55, pose('dodge_duck', lean=45, kf=80)),
               (2.9, pose('punch_wind', lean=10, hf=40, kf=50)), (3.4, pose('punch_wind', lean=12, hf=42, kf=52))]
        self.spose = Track([(a, p, 'snap') for a, p in sk])
        self.sx = Track([(0, -8), (0.5, -8), (1.0, -12), (1.5, -6), (2.0, -12), (2.42, -8), (2.55, -4), (3.4, -2)])
        self.hist_b = []
        self.hist_s = []
        self.fist = []

    def frame(self, t):
        s, b, cam = self.sai, self.bor, self.cam
        tq = math.floor(t * 30) / 30
        slow = t > 2.6
        b.pose = self.bpose(tq if not slow else 2.6 + (tq - 2.6) * 0.3)
        s.pose = self.spose(tq if not slow else 2.6 + (tq - 2.6) * 0.3)
        b.x = self.bx(t)
        s.x = self.sx(t)
        s.expr = 'blank'
        mid = (s.x + b.x) / 2
        zk = ease_io(remap(t, 0.2, 0.6))
        z2 = ease_io(remap(t, 2.55, 3.3))
        cam.x = lerp(55, mid, zk) + z2 * -10
        cam.y = lerp(-50, -52, zk) + z2 * -8
        cam.zoom = lerp(1.2, 1.7, zk) + z2 * 1.0
        if self.ev(0.26):
            self.parts.burst(self.rng, b.x + 10, -2, 30, speed=(60, 200), ang=(-60, 10), life=(0.4, 0.9), size=(4, 9),
                             c0=NIGHT['ground'] * 1.8, c1=NIGHT['ground'], drag=3, mode=DUST)
            self.rings.add(b.x + 5, 0, 60, dur=0.4, squash=0.25, color=(0.8, 0.8, 1), width=4)
            cam.add_trauma(0.4)
        # flurry: each punch releases an air-pressure blast behind saitama
        for k in range(19):
            te = 0.5 + k * 0.1 + 0.03
            if self.ev(te):
                y = self.rng.uniform(-80, -30)
                x = s.x - self.rng.uniform(40, 120)
                self.rings.add(x, y, self.rng.uniform(18, 34), dur=0.3, color=(0.8, 0.85, 1), width=3, inten=0.9)
                self.parts.burst(self.rng, x, y, 8, speed=(80, 220), ang=(150, 210), life=(0.1, 0.3),
                                 c0=(1, 1, 1), c1=(0.6, 0.7, 1), drag=5)
                if k % 3 == 0:
                    self.debris.spawn(self.rng, x - 30, -2, 6, speed=(60, 200), ang=(200, 300),
                                      color=NIGHT['ground'] * 1.6)
                    self.parts.burst(self.rng, x - 30, -4, 4, speed=(20, 60), ang=(200, 340), life=(0.5, 1.0),
                                     size=(5, 9), c0=NIGHT['ground'] * 1.7, c1=NIGHT['ground'], mode=DUST)
                cam.add_trauma(0.18)
        if self.ev(2.42):
            cam.add_trauma(0.5)
            self.rings.add(s.x - 40, -70, 70, dur=0.5, color=(0.8, 0.8, 1), width=6)
        cam.tick()
        fr = Frame(cam)
        self.city.draw(fr, t + 6, NIGHT, ship=0.4)
        if self.every(3):
            self.embers(1)
        self.fx_step(0.0)
        for f in (b, s):
            f.step(t, (-170 if t < 2.6 else -60, 0))
        self.hist_b.append(snap(b))
        self.hist_s.append(snap(s))
        amb = (0.74, 0.72, 0.9)
        if 0.26 < t < 0.5:
            trail(fr, b, self.hist_b, hexc('#6a7aff'), n=4, every=2, alpha=0.6)
        if 0.5 < t < 2.6:
            trail(fr, s, self.hist_s, hexc('#ffe070'), n=2, every=3, alpha=0.4)
        lb = b.render(cam, ambient=amb, rims=[((-1, -0.2), NIGHT['fire'], 0.6, 1)])
        J = b.joints()
        fk = 'fistf' if b.pose['sf'] > b.pose['sb'] else 'fistb'
        self.fist.append(J[fk].copy())
        self.fist = self.fist[-4:]
        if 0.46 < t < 2.62:
            smear(fr, self.fist, 4.5, color=(1, 0.95, 0.75), alpha=0.7)
            for k in range(3):
                jit = np.array([self.rng.uniform(-6, 2), self.rng.uniform(-26, 16)])
                smear(fr, [p + jit for p in self.fist], 3.6, color=(0.9, 0.85, 1.0), alpha=0.35)
        draw(fr, lb)
        draw(fr, s.render(cam, ambient=amb, rims=[((1, -0.2), NIGHT['fire'], 0.6, 1)]))
        self.fx_draw(fr)
        rng = np.random.RandomState(self.i // 2)
        if 0.26 < t < 0.46:
            fr.ui.append(lambda o: hlines(o, rng, 26, color=(1, 1, 1), alpha=0.5, direction=1))
        if t > 2.6:
            k = remap(t, 2.6, 2.8)
            hc = cam.w2s(*s.point('head'))
            fr.ui.append(lambda o: speed_lines(o, rng, hc[0], hc[1], n=50, color=(1, 1, 1), alpha=0.35 * k,
                                               r_in=(0.5, 0.85)))
            post(fr, lambda o: o * np.array([0.85, 0.9, 1.1]) * 0.95)
        lb_ui(fr)
        return fr.finish()


# =====================================================================  S07
class S07_NormalPunch(Shot):
    dur = 3.4
    SFX = [(0.06, 'punch_big', 1.0), (0.07, 'impact', 1.0), (0.32, 'whoosh', 1.0), (0.9, 'crash', 1.0),
           (0.95, 'rumble', 0.8), (2.1, 'wind', 0.5)]

    def setup(self):
        self.city = City(3)
        s = self.sai = Fighter('saitama', -2, 0, 1)
        s.pose = pose('punch_wind', lean=12, hf=42, kf=52)
        b = self.bor = Fighter('boros', 36, 0, -1)
        b.pose = pose('punch', sf=102, lean=32)
        for f in (s, b):
            f.settle(wind=(-120, 0))
        self.bld = dict(x=440, w=70, h=150)
        self.hist_b = []

    def frame(self, t):
        s, b, cam = self.sai, self.bor, self.cam
        tq = math.floor(t * 30) / 30
        rng = self.rng
        # saitama
        if t < 0.05:
            s.pose = pose('punch_wind', lean=12, hf=42, kf=52)
        elif t < 2.0:
            s.pose = pose('punch', sf=86, lean=18)
        else:
            s.pose = Track([(2.0, pose('punch', sf=86, lean=18)), (2.35, pose('idle2'), 'out')])(tq)
        # boros
        if t < 0.3:
            b.pose = pose('punch', sf=102, lean=32) if t < 0.06 else pose('hurt', lean=-10)
            b.x = 36
            b.air = False
        else:
            b.air = True
            p = remap(t, 0.3, 0.88)
            b.pose = pose('knocked', rot=-(t - 0.3) * 720)
            b.x = 36 + (self.bld['x'] - 36) * ease_in(p, 1.3)
            b.y = -46 - 30 * math.sin(p * math.pi) * 0.6
        if self.ev(0.06):
            b.hole = (0, -17, 11)
            b.cracks = 1.0
            fp = s.point('fistf')
            self.hit(fp[0] + 6, fp[1], 2.2, color=(1, 0.8, 0.3), trauma=1.0, n=120)
            self.rings.add(fp[0], fp[1], 160, dur=0.5, color=(1, 0.9, 0.6), width=10)
            self.parts.burst(rng, fp[0] + 10, fp[1], 60, speed=(200, 600), ang=(-35, 35), life=(0.2, 0.5),
                             c0=(1, 0.8, 1), c1=(0.8, 0.2, 0.9), drag=3)
        if self.ev(0.88):
            bx = self.bld['x']
            cam.add_trauma(1.2)
            self.debris.spawn(rng, bx, -80, 70, speed=(100, 420), ang=(190, 350), size=(2, 7),
                              color=hexc('#4a4466'), grav=420)
            self.debris.spawn(rng, bx, -40, 30, speed=(60, 260), ang=(200, 340), size=(1.5, 4),
                              color=hexc('#6a6488'), grav=420)
            self.parts.burst(rng, bx, -60, 70, speed=(40, 220), ang=(180, 360), life=(1.2, 2.4), size=(8, 18),
                             c0=hexc('#6a6078'), c1=hexc('#2a2436'), drag=1.6, mode=DUST, spread=30)
            self.parts.burst(rng, bx, -60, 80, speed=(150, 500), life=(0.3, 0.8), c0=(1, 0.9, 0.6),
                             c1=(1, 0.3, 0.1), drag=2)
            self.rings.add(bx, -60, 120, dur=0.6, color=(1, 0.6, 0.3), width=8)
            self.rings.add(bx, 0, 220, dur=0.8, squash=0.22, color=(1, 0.7, 0.4), width=6)
        # camera
        if t < 0.3:
            cam.x, cam.y, cam.zoom = 14, -58, 2.3 - t
        elif t < 2.0:
            k = ease_io(remap(t, 0.3, 0.9))
            cam.x = lerp(14, self.bld['x'] - 40, k)
            cam.y = lerp(-58, -70, k)
            cam.zoom = lerp(2.0, 1.1, k) - 0.05 * remap(t, 0.9, 2.0)
        else:
            cam.x, cam.y, cam.zoom = s.x + 12, -52, 1.9 + 0.2 * remap(t, 2.0, 3.4)
        cam.tick()
        fr = Frame(cam)
        self.city.draw(fr, t + 9, NIGHT, ship=0.3)
        # the building boros smashes into
        bd = self.bld
        md = MaskDraw()
        if t < 0.88:
            x0, y0 = cam.w2s(bd['x'] - 10, -bd['h'])
            x1, y1 = cam.w2s(bd['x'] + bd['w'], 0)
            md.d.rectangle([x0, y0, x1, y1], fill=1)
        else:
            pts = [(bd['x'] - 10, 0), (bd['x'] - 10, -40), (bd['x'] + 5, -55), (bd['x'] + 18, -30), (bd['x'] + 30, -62),
                   (bd['x'] + 45, -44), (bd['x'] + bd['w'], -70), (bd['x'] + bd['w'], 0)]
            md.poly(cam.pts(np.array(pts, np.float32)))
        m = md.get()
        fr.img[m] = hexc('#2a2644')
        win = m & (((XX.astype(int) // max(1, int(3 * cam.zoom))) % 3 == 0) &
                   ((YY.astype(int) // max(1, int(4 * cam.zoom))) % 3 == 0))
        fr.img[win] = hexc('#ffb45a') * 0.6
        edge = m & ~shift(m, -1, 0)
        fr.img[edge] = hexc('#8a8aff') * 0.5
        if t > 0.88:
            k = 1 - remap(t, 0.88, 3.0)
            c = cam.w2s(bd['x'] + 20, -40)
            radial(fr.glow, c[0], c[1], 70 * cam.zoom, (1, 0.45, 0.15), 0.9 * k, steps=5)
        self.fx_step(0.0)
        for f in (b, s):
            f.step(t, (-150, 0))
        self.hist_b.append(snap(b))
        amb = (0.75, 0.72, 0.9)
        if 0.3 < t < 0.9:
            trail(fr, b, self.hist_b, hexc('#ff4ad8'), n=5, every=2, alpha=0.5)
        if t < 0.88 or t > 2.0:
            draw(fr, b.render(cam, ambient=amb, rims=[((-1, 0), (1, 0.8, 0.4), 0.8, 1)]))
        draw(fr, s.render(cam, ambient=amb, rims=[((1, -0.3), NIGHT['fire'], 0.6, 1)]))
        self.fx_draw(fr)
        # post: impact frames, whip smear
        if 0.06 <= t < 0.28:
            k = int((t - 0.06) * FPS)
            if k < 3:
                post(fr, lambda o: impact_frame(o, 0, c_dark=(0, 0, 0), c_light=(1, 1, 1)))
            elif k < 6:
                post(fr, lambda o: impact_frame(o, 1, c_dark=(0.9, 0.1, 0.15), c_light=(0.05, 0, 0.05)))
            elif k < 9:
                post(fr, lambda o: chroma(o, 3))
        if 0.3 < t < 0.8:
            r = 10 * math.sin(math.pi * remap(t, 0.3, 0.8))
            post(fr, lambda o: hsmear(o, r))
        rng2 = np.random.RandomState(self.i // 2)
        if 0.06 < t < 0.35:
            fp = cam.w2s(*s.point('fistf'))
            fr.ui.append(lambda o: speed_lines(o, rng2, fp[0], fp[1], n=70, color=(1, 1, 1), alpha=0.6,
                                               r_in=(0.25, 0.6)))
        lb_ui(fr)
        if t > 2.0:
            fr.ui.append(lambda o: subtitle(o, '埼玉', '……嗯，还挺结实的。', t, 2.1, 1.3))
        return fr.finish()


# =====================================================================  S08
class S08_Regen(Shot):
    dur = 3.3
    SFX = [(0.0, 'rumble', 0.4), (0.3, 'regen', 0.9), (2.0, 'power_small', 0.7), (2.55, 'eye', 0.8)]

    def setup(self):
        self.city = City(4)
        b = self.bor = Fighter('boros', 0, 0, -1)
        b.pose = pose('kneel')
        b.hole = (0, -17, 11)
        b.cracks = 1.0
        b.settle(wind=(80, 0))
        self.hole = Track([(0, 11), (0.35, 11), (1.9, 0, 'io')])
        self.bp = Track([(0, pose('kneel')), (2.0, pose('kneel')), (2.5, pose('boros_idle', head=-10), 'out'),
                         (3.3, pose('boros_idle', head=-12))])

    def frame(self, t):
        b, cam = self.bor, self.cam
        tq = math.floor(t * 30) / 30
        b.pose = self.bp(tq)
        r = self.hole(t)
        b.hole = (0, -17, r) if r > 0.3 else None
        b.expr = 'grin' if t > 2.3 else 'calm'
        J = b.joints()
        hc = J['hip'] + np.array([0, -17])
        cam.x = J['hip'][0] - 8
        cam.y = J['hip'][1] - 16 - 10 * remap(t, 2.0, 2.6)
        cam.zoom = 2.2 + 0.4 * ease_io(remap(t, 0, 3.3))
        if self.ev(2.0):
            cam.add_trauma(0.5)
            self.rings.add(J['hip'][0], 0, 90, dur=0.5, squash=0.25, color=(1, 0.4, 0.9), width=5)
        cam.tick()
        fr = Frame(cam)
        self.city.draw(fr, t + 12, NIGHT, ship=0.0)
        if r > 0.3 and self.every(1):
            n = 5
            a = self.rng.uniform(0, 6.28, n)
            d = self.rng.uniform(30, 55, n)
            pos = np.stack([hc[0] + np.cos(a) * d, hc[1] + np.sin(a) * d], -1)
            vel = -np.stack([np.cos(a), np.sin(a)], -1) * (d[:, None] * 2.2)
            self.parts.emit(pos, vel, 0.45, self.rng.uniform(1, 2, n), (0.9, 0.5, 1.0), (1, 1, 1), mode=DOT)
        if self.every(3):
            self.parts.burst(self.rng, J['hip'][0] + self.rng.uniform(-60, 60), -2, 1, speed=(5, 15), ang=(250, 290),
                             life=(1.5, 2.5), size=(6, 12), c0=NIGHT['ground'] * 1.5, c1=NIGHT['ground'], mode=DUST)
        self.fx_step(None)
        b.step(t, (70, 0))
        L = b.render(cam, ambient=(0.7, 0.68, 0.9),
                     rims=[((0, -1), hexc('#ff4ad8'), 0.3 + 0.3 * (r > 0.3), 1), ((1, 0), NIGHT['fire'], 0.5, 1)])
        cracks(fr, 5, J['hip'][0], 0, 70, 1.0, color=(0.6, 0.3, 0.9), inten=0.5)
        draw(fr, L)
        if r > 0.3:
            c = cam.w2s(*hc)
            radial(fr.glow, c[0], c[1], 30 * cam.zoom, hexc('#ff3ad0'), 0.5, steps=4)
        if t > 2.5:
            R = b.S['head_r'] * b.S['scale']
            e = J['head'] + np.array([0.36 * R * b.f, -0.12 * R])
            ec = cam.w2s(*e)
            g = math.sin(math.pi * remap(t, 2.55, 3.0))
            fr.ui.append(lambda o: glint(o, None, ec[0], ec[1], 16, g))
        self.parts.draw(fr)
        lb_ui(fr)
        fr.ui.append(lambda o: subtitle(o, '波罗斯', '有意思……真是有意思！', t, 0.9, 2.3, color=(0.7, 0.8, 1)))
        return fr.finish()


# =====================================================================  S09
class S09_Burst(Shot):
    dur = 4.3
    SFX = [(0.0, 'charge', 0.9), (0.95, 'explode', 1.0), (0.95, 'burst', 1.0), (1.35, 'banner', 0.8),
           (1.0, 'aura_loop', 0.8)]

    def setup(self):
        self.city = City(5)
        b = self.bor = Fighter('boros', 0, 0, -1)
        b.pose = pose('clench')
        b.settle(wind=(30, -100))
        self.bp = Track([(0, pose('boros_idle')), (0.3, pose('clench'), 'out'), (0.95, pose('clench', lean=24)),
                         (1.05, pose('power'), 'snap'), (4.3, pose('power', head=-18, lean=-12))])

    def frame(self, t):
        b, cam = self.bor, self.cam
        rng = self.rng
        tq = math.floor(t * 30) / 30
        b.pose = self.bp(tq)
        pre = remap(t, 0.2, 0.95)
        b.shake = 1.4 * pre if t < 0.95 else 0.6
        b.cracks = pre if t < 0.95 else 0
        if self.ev(0.95):
            b.form = 'burst'
            J = b.joints()
            for key in ('neck', 'shf', 'shb', 'elf', 'knf', 'knb', 'hip'):
                p = J[key]
                self.debris.spawn(rng, p[0], p[1], 6, speed=(120, 380), ang=(0, 360), size=(2, 5),
                                  color=hexc('#eab83c'), grav=380, glow=hexc('#ff8a2a') * 0.4)
            self.rings.add(0, -50, 260, dur=0.7, color=(1, 0.4, 0.9), width=12)
            self.rings.add(0, -50, 180, dur=0.5, color=(1, 1, 1), width=6)
            self.rings.add(0, 0, 330, dur=0.9, squash=0.2, color=(1, 0.5, 0.8), width=8)
            self.parts.burst(rng, 0, -50, 160, speed=(150, 600), life=(0.3, 0.9), c0=(1, 0.9, 1), c1=(1, 0.2, 0.7),
                             drag=2.5)
            cam.add_trauma(1.4)
        pal = mix_pal(NIGHT, BURST, ease_out(remap(t, 0.95, 1.4)))
        z = 1.7 if t < 0.95 else lerp(1.25, 1.75, ease_io(remap(t, 1.0, 4.3)))
        cam.x, cam.y, cam.zoom = 0, -54 - (6 if t > 0.95 else 0), z + 0.05 * pre
        if t < 0.95 and self.every(6):
            cam.add_trauma(0.15 * pre)
        if t > 0.95 and self.every(10):
            cam.add_trauma(0.12)
        cam.tick()
        fr = Frame(cam)
        self.city.draw(fr, t + 15, pal, ship=0.0, cloud_speed=12 if t > 1 else 3)
        J = b.joints()
        if t < 0.95:
            cracks(fr, 9, 0, 0, 120, pre, color=(1, 0.4, 0.8), inten=1.0 * pre)
        else:
            cracks(fr, 9, 0, 0, 170, 1.0, color=(1, 0.45, 0.8), inten=1.2)
        # rising rocks & aura flames
        if (t < 0.95 and self.every(4)) or (t > 0.95 and self.every(2)):
            x = rng.uniform(-150, 150)
            self.debris.spawn(rng, x, -1, 1, speed=(20, 60), ang=(260, 280), size=(1.5, 4.5),
                              color=pal['ground'] * 1.7, grav=-40 - 60 * (t > 0.95), life=(2, 3), spin=3)
        if t > 0.95:
            n = 3
            xs = J['hip'][0] + rng.uniform(-18, 18, n)
            ys = J['hip'][1] + rng.uniform(-40, 30, n)
            self.parts.emit(np.stack([xs, ys], -1), np.stack([rng.uniform(-20, 20, n), -rng.uniform(80, 200, n)], -1),
                            rng.uniform(0.3, 0.7, n), 1.2, (0.8, 0.5, 0.8), (0.5, 0.05, 0.4), drag=1, mode=EMBER)
        self.fx_step(None)
        b.step(t, (0, -500) if t > 0.95 else (20, -150))
        L = b.render(cam, ambient=pal['rim'] * 0.3 + 0.55, rims=[((0, -1), hexc('#ff6ad8'), 0.45 if t > 0.95 else 0.2, 1),
                                                                ((1, 0), hexc('#ff6a3a'), 0.4, 1)])
        if t > 0.95:
            aura(fr, L[1], hexc('#ff3ad8'), t, 0.42, height=11, seed=3)
            c = cam.w2s(*J['shoulder'])
            radial(fr.glow, c[0], c[1], 90 * cam.zoom, hexc('#ff2ab0'), 0.14 + 0.05 * math.sin(t * 20), steps=6)
        draw(fr, L)
        if t < 0.95 and self.every(3):
            for _ in range(2):
                a = rng.uniform(0, 6.28)
                p0 = J['shoulder'] + np.array([math.cos(a), math.sin(a)]) * 10
                p1 = p0 + np.array([math.cos(a), math.sin(a)]) * rng.uniform(15, 35) * pre
                lightning(fr, rng, p0, p1, color=(1, 0.4, 0.9), inten=1.2, branches=1, depth=4)
        if t > 0.95 and self.every(4):
            a = rng.uniform(0, 6.28)
            p0 = J['shoulder'] + np.array([math.cos(a), math.sin(a)]) * 12
            lightning(fr, rng, p0, p0 + np.array([math.cos(a), math.sin(a)]) * 50, color=(1, 0.5, 1), inten=1.5,
                      branches=2)
        self.fx_draw(fr)
        if 0.95 <= t < 1.12:
            k = int((t - 0.95) * FPS)
            if k < 2:
                post(fr, lambda o: o * 0 + 1)
            elif k < 5:
                post(fr, lambda o: impact_frame(o, 0, c_dark=(0.1, 0, 0.12), c_light=(1, 0.7, 1)))
            else:
                post(fr, lambda o: chroma(o, 2))
        lb_ui(fr)
        fr.ui.append(lambda o: banner(o, t - 1.35, '流星爆发', 'METEORIC  BURST', color=(1, 0.35, 0.85), side=1, y=64))
        return fr.finish(bloom_k=0.75, thresh=0.8)
