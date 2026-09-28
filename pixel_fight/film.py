"""Stickman fight: one continuous choreographed brawl, built for impact.

Everything is authored in *story time*.  Every hit declares a hit-stop; the time map
freezes story time on contact (while shake / sparks keep running in real time), which is
what gives each blow its weight.  Smears, afterimages, skid dust and landing dust are
generated automatically from the motion itself.
"""
import math
from collections import defaultdict, deque

import numpy as np

from engine import *
from fighters import pose, draw, aura
from stick import Stick
from fx import *
from backgrounds import City, NIGHT, BURST, CHARGE, DAWN, mix_pal
from shotbase import smear, ghost, glint

S_GLOW = hexc('#ffd070')
B_GLOW = hexc('#ff4ad8')
BLOCK = np.array([0.75, 0.9, 1.0], np.float32)

KINDS = {
    'block': dict(stop=0.035, star=0.7, n=16, ring=0.7, trauma=0.22, zk=0.02, block=True, sfx=('hit2', 0.5)),
    'blockh': dict(stop=0.07, star=1.3, n=34, ring=1.4, trauma=0.45, zk=0.05, block=True, crack=True,
                   sfx=('hit2', 0.9)),
    'light': dict(stop=0.05, star=1.0, n=22, ring=0.9, trauma=0.3, zk=0.03, count=True, sfx=('hit', 0.75)),
    'medium': dict(stop=0.085, star=1.5, n=40, ring=1.3, trauma=0.5, zk=0.06, count=True, sfx=('hit', 1.0)),
    'heavy': dict(stop=0.15, star=2.3, n=90, ring=2.2, trauma=0.9, zk=0.12, impact=3, chroma=3, count=True,
                  lines=True, sfx=('punch_big', 1.0)),
    'clash': dict(stop=0.18, star=2.8, n=130, ring=3.0, trauma=1.15, zk=0.14, impact=3, chroma=4, bolts=True,
                  lines=True, sfx=('clash', 1.0)),
    'slam': dict(stop=0.12, star=2.0, n=60, trauma=1.0, zk=0.1, ground=1.0, impact=2, sfx=('explode', 0.8)),
    'slam_big': dict(stop=0.17, star=3.0, n=110, trauma=1.4, zk=0.15, ground=1.8, impact=3, chroma=4,
                     sfx=('explode', 1.0)),
    'wall': dict(stop=0.06, star=1.6, n=40, trauma=0.8, zk=0.05, wall=True, sfx=('crash', 0.9)),
    'land': dict(stop=0.0, n=0, trauma=0.3, ground=0.35, sfx=('land', 0.45)),
    'power': dict(stop=0.1, n=160, trauma=1.3, zk=0.1, impact=2, sfx=('burst', 1.0)),
    'beam': dict(stop=0.05, n=0, trauma=1.0, flash=0.9, sfx=('beam', 1.0)),
    'mega': dict(stop=0.4, star=4.5, n=260, ring=5.0, trauma=1.7, zk=0.25, impact=9, chroma=6, lines=True,
                 sfx=('serious_punch', 1.0)),
    'mark': dict(stop=0.0, n=0),
}


class Hit:
    def __init__(self, ts, kind, att=None, joint='fistf', pos=None, d=None, cb=None):
        self.ts, self.kind, self.att, self.joint, self.pos, self.d, self.cb = ts, kind, att, joint, pos, d, cb
        self.k = KINDS[kind]


class Actor:
    def __init__(self, st):
        self.st = st
        self.pk, self.xk, self.fk, self.hk = [], [], [], []

    def p(self, t, name, e='snap', **kw):
        self.pk.append((t, pose(name, **kw), e))
        return self

    def at(self, t, x, lift=0.0, e='io'):
        self.xk.append((t, np.array([x, lift], np.float32), e))
        return self

    def arc(self, t0, t1, x0, l0, x1, l1, h=0.0, ease='lin'):
        n = max(2, int((t1 - t0) * 40))
        for k in range(n + 1):
            u = k / n
            ue = EASES[ease](u)
            self.xk.append((t0 + (t1 - t0) * u, np.array([lerp(x0, x1, ue), lerp(l0, l1, ue) + h * 4 * u * (1 - u)],
                                                        np.float32), 'lin'))
        return self

    def face(self, t, f):
        self.fk.append((t, f))
        return self

    def hide(self, t0, t1):
        self.hk.append((t0, t1))
        return self

    def done(self):
        self.P = Track(self.pk)
        self.X = Track(self.xk)
        self.fk.sort(key=lambda k: k[0])

    def pos(self, s):
        return self.X(s)

    def apply(self, s):
        st = self.st
        st.pose = dict(self.P(s))
        x, lift = self.X(s)
        st.x = float(x)
        st.pose['lift'] = max(0.0, float(lift))
        f = self.fk[0][1]
        for t, ff in self.fk:
            if s >= t:
                f = ff
        st.f = f
        st.visible = not any(a <= s < b for a, b in self.hk)


class Bldg:
    def __init__(self, x, w, h):
        self.x, self.w, self.h = x, w, h
        self.state = 0
        self.hole = None


class Film:
    def __init__(self):
        self.S = Stick('saitama', -70, 1)
        self.B = Stick('boros', 70, -1)
        self.a = Actor(self.S)
        self.b = Actor(self.B)
        self.hits, self.sfx_story, self.slow, self.camk = [], [], [], []
        self.bldgs = [Bldg(-330, 54, 128), Bldg(-540, 74, 168), Bldg(330, 60, 140)]
        self.script()
        self.a.done()
        self.b.done()
        self.CAM = Track(self.camk)
        self.build_timemap()
        self.reset()

    # ================================================================ choreography
    def cam(self, t, e='io', **kw):
        d = dict(ws=1.0, wb=1.0, zb=1.3, yo=0.0, k=0.14)
        d.update(kw)
        self.camk.append((t, d, e))

    def hit(self, ts, kind, att=None, joint='fistf', **kw):
        self.hits.append(Hit(ts, kind, att, joint, **kw))

    def snd(self, ts, name, g=1.0):
        self.sfx_story.append((ts, name, g))

    def script(self):
        a, b = self.a, self.b
        S, B = 'S', 'B'
        self.cam(0.0, zb=1.5)
        self.cam(1.0, zb=1.3)
        # ---------------- A: face-off
        a.at(0, -70).p(0, 'bored').face(0, 1)
        b.at(0, 70).p(0, 'stance').face(0, -1)
        # ---------------- B: opening exchange
        b.p(0.95, 'stance').p(1.02, 'crouch', 'out').at(1.02, 70).p(1.1, 'dash').at(1.2, -38, e='out')
        self.snd(1.08, 'dash')
        a.p(1.1, 'bored').p(1.16, 'block')
        b.p(1.2, 'jab_hi')
        self.hit(1.22, 'block', B)
        b.p(1.28, 'stance', lean=16).p(1.34, 'punch_b')
        self.hit(1.36, 'block', B, 'fistb')
        b.p(1.42, 'stance').p(1.5, 'kick')
        self.hit(1.53, 'blockh', B, 'toef')
        a.at(1.5, -70).at(1.64, -80, e='out')
        b.p(1.62, 'stance').p(1.7, 'uppercut')
        a.p(1.64, 'dodge_back')
        self.snd(1.7, 'swish')
        a.p(1.84, 'punch_wind').p(1.9, 'punch').at(1.84, -80).at(1.92, -72, e='out')
        self.hit(1.92, 'medium', S)
        b.p(1.92, 'hurt').at(1.92, -38).at(2.18, 12, e='out')
        b.p(2.25, 'stance')
        # flurry
        a.p(2.3, 'block')
        b.p(2.3, 'dash').at(2.3, 12).at(2.42, -42, e='out')
        self.snd(2.3, 'dash')
        rs = np.random.RandomState(3)
        for k in range(11):
            t = 2.45 + k * 0.07
            if k % 2 == 0:
                b.p(t, 'punch', sf=90 + rs.uniform(-20, 18), lean=22)
            else:
                b.p(t, 'punch_b', sb=90 + rs.uniform(-20, 18), lean=22)
            b.p(t + 0.035, 'stance', lean=20)
            self.hit(t + 0.02, 'block', B, 'fistf' if k % 2 == 0 else 'fistb')
            a.p(t + 0.02, 'block', head=rs.uniform(-8, 10), lean=rs.uniform(-8, 0))
        b.p(3.24, 'stance').p(3.3, 'kick_hi')
        a.p(3.24, 'dodge_duck')
        self.snd(3.3, 'swish')
        # teleport behind
        a.hide(3.38, 3.45).at(3.38, -72, e='lin').at(3.45, -8, e='lin').face(3.44, -1)
        self.snd(3.38, 'swish', 0.8)
        a.p(3.45, 'punch_wind').p(3.52, 'punch')
        self.hit(3.54, 'heavy', S, d=(-1, -0.1))
        b.p(3.54, 'knocked', rot=0, e='lin').p(3.96, 'knocked', rot=-540, e='lin')
        b.arc(3.54, 3.96, -42, 0, -306, 18, h=26)
        self.hit(3.96, 'wall', B, 'hip')
        b.p(3.97, 'hurt', rot=0).at(3.97, -312, 20).at(4.3, -312, 16)
        a.p(3.8, 'bored')
        self.cam(3.5, ws=0.6)
        # ---------------- C: boros comeback + air combo
        b.face(4.55, 1).p(4.5, 'crouch').at(4.5, -312, 0, e='out').p(4.6, 'dash').at(4.6, -312, 0).at(4.86, -44, e='out')
        self.hit(4.6, 'wall', B, 'hip')
        self.snd(4.6, 'dash')
        self.cam(4.6, ws=1.0)
        a.p(4.7, 'bored')
        b.p(4.86, 'uppercut')
        self.hit(4.88, 'heavy', B, d=(0.3, -1))
        a.p(4.88, 'knocked', rot=0, e='lin').p(5.28, 'knocked', rot=-360, e='lin')
        a.at(4.88, -8, 0).at(5.28, 4, 110, e='out')
        b.p(4.98, 'tuck').at(4.98, -44, 0).at(5.28, -24, 100, e='out')
        self.snd(4.98, 'whoosh')
        b.p(5.28, 'punch')
        self.hit(5.3, 'medium', B)
        a.p(5.3, 'hurt').at(5.4, 12, 114)
        b.p(5.38, 'stance', lean=10).p(5.42, 'punch_b').at(5.42, -16, 106)
        self.hit(5.44, 'medium', B, 'fistb')
        a.at(5.52, 20, 118)
        b.p(5.5, 'spin_kick')
        self.hit(5.56, 'medium', B, 'toef')
        a.p(5.56, 'hurt', rot=-30).at(5.72, 32, 128, e='out')
        b.at(5.6, -8, 110).at(5.78, 18, 166, e='out').p(5.7, 'hammer_up')
        self.snd(5.7, 'whoosh')
        b.p(5.84, 'hammer_down')
        self.hit(5.86, 'heavy', B, d=(0.2, 1))
        a.p(5.86, 'knocked', rot=-60, e='lin').p(6.02, 'idle', rot=-90)
        a.at(5.86, 32, 128).at(6.02, 36, 0, e='in')
        self.hit(6.02, 'slam', S, 'hip', pos=(36, 0))
        b.at(6.1, 18, 160).at(6.5, -10, 0, e='in').p(6.4, 'tuck').p(6.48, 'land').p(6.75, 'stance')
        self.hit(6.5, 'land', B, 'hip', pos=(-10, 0))
        a.p(6.9, 'idle', rot=-90).p(7.15, 'crouch', rot=0, e='out').p(7.4, 'bored', e='out')
        b.at(6.8, -10).at(7.5, -40, e='io')
        # ---------------- D: saitama's normal punch
        a.p(7.8, 'bored').p(7.88, 'crouch').p(7.96, 'dash').at(7.96, 36).at(8.1, -6, e='out')
        self.snd(7.96, 'dash')
        a.p(8.08, 'punch_wind').p(8.14, 'punch')
        self.hit(8.16, 'heavy', S, d=(-1, -0.05))
        b.p(8.16, 'knocked', rot=0, e='lin').p(8.78, 'knocked', rot=-900, e='lin')
        b.arc(8.16, 8.78, -40, 0, -770, 16, h=12)
        self.hit(8.35, 'wall', B, 'hip', pos=(-330, -40))
        self.hit(8.56, 'wall', B, 'hip', pos=(-540, -40))
        self.hit(8.78, 'slam', B, 'hip', pos=(-770, 0))
        b.p(8.79, 'idle', rot=90).at(8.79, -770, 0)
        self.cam(8.1, ws=1.0, k=0.14)
        self.cam(8.2, ws=0.0, zb=0.62, k=0.3, e='lin')
        self.cam(9.0, ws=0.0, zb=0.7, k=0.2)
        a.p(8.5, 'bored')
        # boros powers up
        b.p(9.3, 'idle', rot=90).p(9.55, 'kneel', rot=0, e='out').p(9.78, 'clench', e='out').p(9.84, 'power')
        self.hit(9.84, 'power', B, 'shoulder', cb=lambda f: f.power_up())
        b.p(10.3, 'power')
        # ---------------- E: teleport assault
        self.cam(10.3, ws=0.0, zb=0.7)
        self.cam(10.45, ws=1.0, wb=0.5, zb=1.0, k=0.3)
        a.p(10.3, 'stance').face(10.3, -1)
        b.hide(10.4, 10.55).at(10.4, -770, e='lin').at(10.55, -38, e='lin').face(10.55, 1)
        self.snd(10.4, 'swish')
        b.p(10.55, 'stance').p(10.58, 'punch')
        a.p(10.56, 'block')
        self.hit(10.61, 'blockh', B)
        a.at(10.6, -6).at(10.68, 0, e='out')
        b.hide(10.68, 10.76).at(10.68, -38, e='lin').at(10.76, 32, e='lin').face(10.76, -1)
        self.snd(10.68, 'swish')
        a.face(10.74, 1).p(10.74, 'block')
        b.p(10.76, 'stance').p(10.79, 'kick')
        self.hit(10.82, 'blockh', B, 'toef')
        a.at(10.82, 0).at(10.9, -6, e='out')
        b.hide(10.88, 10.96).at(10.88, 32, e='lin').at(10.96, -22, 58, e='lin').face(10.96, 1)
        self.snd(10.88, 'swish')
        a.p(10.94, 'block_up')
        b.p(10.96, 'hammer_up').p(11.0, 'hammer_down')
        self.hit(11.02, 'blockh', B, d=(0.1, 1))
        b.hide(11.08, 11.16).at(11.08, -22, 58, e='lin').at(11.16, -40, 0, e='lin').face(11.16, 1)
        self.snd(11.08, 'swish')
        a.face(11.14, -1).p(11.14, 'block')
        b.p(11.16, 'stance').p(11.19, 'spin_kick')
        self.hit(11.22, 'blockh', B, 'toef')
        b.hide(11.28, 11.36).at(11.28, -40, e='lin').at(11.36, 52, e='lin').face(11.36, -1)
        self.snd(11.28, 'swish')
        a.face(11.34, 1).p(11.34, 'punch_wind')
        b.p(11.36, 'punch_wind').p(11.41, 'punch')
        a.p(11.41, 'punch')
        self.hit(11.44, 'clash', S, pos=(22, -44))
        a.at(11.44, -6).at(11.7, -52, e='out').p(11.46, 'stance')
        b.at(11.44, 52).at(11.7, 92, e='out').p(11.46, 'stance')
        # ---------------- F: air fight
        a.p(11.85, 'crouch').p(11.95, 'tuck').at(11.95, -52, 0).at(12.28, -30, 140, e='out').p(12.26, 'stance')
        b.p(11.85, 'crouch').p(11.95, 'tuck').at(11.95, 92, 0).at(12.28, 44, 150, e='out').p(12.26, 'stance')
        self.snd(11.95, 'jump', 0.6)
        self.cam(11.9, zb=1.25, k=0.18)
        a.p(12.34, 'dash').at(12.34, -30, 140).at(12.44, 6, 146, e='out').p(12.42, 'punch')
        b.p(12.4, 'block')
        self.hit(12.46, 'block', S)
        b.p(12.52, 'kick').at(12.52, 44, 150)
        a.p(12.5, 'block')
        self.hit(12.56, 'block', B, 'toef')
        a.p(12.62, 'punch_wind').p(12.68, 'punch')
        b.p(12.62, 'punch_wind').p(12.68, 'punch')
        self.hit(12.71, 'clash', S, pos=(24, -178))
        a.at(12.71, 6, 146).at(12.9, -40, 172, e='out').p(12.74, 'stance')
        b.at(12.71, 44, 150).at(12.9, 86, 128, e='out').p(12.74, 'stance')
        a.p(12.96, 'dash').at(12.96, -40, 172).at(13.08, 8, 152, e='out').p(13.06, 'punch')
        b.p(12.96, 'dash').at(12.96, 86, 128).at(13.08, 46, 152, e='out').p(13.06, 'punch')
        self.snd(12.96, 'dash')
        self.hit(13.1, 'clash', S, pos=(28, -186))
        a.at(13.1, 8, 152).at(13.25, -6, 156, e='out').p(13.14, 'stance')
        b.at(13.1, 46, 152).at(13.25, 60, 150, e='out').p(13.14, 'stance')
        b.p(13.26, 'spin_kick').at(13.28, 40, 152, e='out')
        self.hit(13.34, 'medium', B, 'toef')
        a.p(13.34, 'hurt', rot=-40, e='lin').p(13.56, 'stance', rot=0).at(13.34, -6, 156).at(13.56, -62, 186, e='out')
        a.p(13.6, 'dash').at(13.6, -62, 186).at(13.75, 24, 160, e='out').p(13.73, 'punch')
        self.snd(13.6, 'dash')
        b.p(13.6, 'stance').at(13.6, 60, 150)
        self.hit(13.77, 'heavy', S, d=(0.6, -0.8))
        b.p(13.77, 'knocked', rot=0, e='lin').p(14.05, 'knocked', rot=-360, e='lin').at(13.77, 60, 150)
        b.at(14.05, 112, 250, e='out')
        a.p(13.85, 'dash').at(13.85, 24, 160).at(14.1, 76, 244, e='out').p(14.12, 'kick_hi')
        b.p(14.08, 'block')
        self.hit(14.17, 'blockh', S, 'toef', d=(0.3, -1))
        b.at(14.25, 100, 280, e='out').p(14.2, 'hammer_up')
        b.p(14.31, 'hammer_down')
        self.hit(14.34, 'heavy', B, d=(-0.2, 1))
        a.p(14.34, 'knocked', rot=0, e='lin').p(14.56, 'idle', rot=-90).at(14.34, 76, 244).at(14.56, 40, 0, e='in')
        self.hit(14.56, 'slam_big', S, 'hip', pos=(40, 0))
        b.at(14.4, 100, 280).at(15.2, 150, 200, e='io').p(14.5, 'float')
        self.cam(14.3, zb=1.1, k=0.22)
        # ---------------- G: roaring cannon
        a.p(15.5, 'idle', rot=-90).p(15.75, 'crouch', rot=0, e='out').p(16.0, 'stance', e='out')
        a.p(16.6, 'look_up')
        b.p(15.3, 'cannon', sf=58, sb=62, rot=14, e='out')
        self.hit(15.3, 'mark', B, cb=lambda f: f.set('charge', True))
        self.snd(15.3, 'charge_big')
        self.cam(15.3, zb=0.82, k=0.1)
        self.hit(17.6, 'beam', B, cb=lambda f: f.fire())
        b.p(17.6, 'cannon', sf=58, sb=62, rot=14, lean=-14)
        a.p(17.62, 'block')
        a.p(18.2, 'crouch').p(18.32, 'fly', rot=-30).at(18.32, 40, 0).at(18.78, 110, 176, e='in')
        self.snd(18.32, 'dash')
        self.hit(18.8, 'mark', B, cb=lambda f: f.set('beam_on', False))
        # ---------------- H: serious punch
        self.slow.append((18.8, 19.3, 0.28))
        a.p(18.8, 'serious_wind').face(18.8, 1).at(18.8, 110, 176)
        b.p(18.84, 'hurt', lean=-12)
        self.cam(18.7, zb=0.9, k=0.2)
        self.cam(18.85, zb=1.7, k=0.12)
        a.p(19.28, 'serious_punch')
        self.hit(19.3, 'mega', S, cb=lambda f: f.mega())
        self.cam(19.3, zb=1.7, k=0.12)
        self.cam(19.5, zb=0.75, k=0.1, yo=-20)
        b.at(19.3, 150, 200).at(20.6, 236, 236, e='out').p(19.3, 'knocked', rot=0, e='lin').p(20.6, 'knocked',
                                                                                                  rot=-120, e='lin')
        a.at(19.7, 110, 176).at(20.6, 110, 0, e='in').p(20.2, 'fly_up').p(20.58, 'land').p(20.95, 'bored', e='out')
        self.hit(20.6, 'land', S, 'hip', pos=(110, 0))
        self.cam(20.6, ws=1.0, wb=0.0, zb=0.7, k=0.08, yo=-40)
        self.cam(23.4, ws=1.0, wb=0.0, zb=0.75, yo=-40)
        self.hit(21.0, 'mark', S, cb=lambda f: f.set('ko', True))
        self.end = 23.4

    # ================================================================ time map
    def build_timemap(self):
        hits = sorted(self.hits, key=lambda h: h.ts)
        frames, trig = [], defaultdict(list)
        s, fz, hi, i = 0.0, 0.0, 0, 0
        while s < self.end:
            if fz > 1e-9:
                fz -= DT
                frames.append(s)
                i += 1
                continue
            rate = 1.0
            for a, b, r in self.slow:
                if a <= s < b:
                    rate = r
            nxt = s + DT * rate
            if hi < len(hits) and hits[hi].ts <= nxt:
                s = hits[hi].ts
                while hi < len(hits) and hits[hi].ts <= s + 1e-9:
                    trig[i].append(hits[hi])
                    fz = max(fz, hits[hi].k.get('stop', 0))
                    hi += 1
            else:
                s = nxt
            frames.append(s)
            i += 1
        self.frames = frames
        self.trig = trig
        self.n = len(frames)
        fa = np.array(frames)
        real = lambda ts: float(np.searchsorted(fa, ts - 1e-9)) * DT
        ev = []
        for h in hits:
            if 'sfx' in h.k:
                ev.append((real(h.ts), h.k['sfx'][0], h.k['sfx'][1]))
            if h.kind == 'mega':
                ev += [(real(h.ts) + 0.3, 'shockwave', 1.0), (real(h.ts) + 0.9, 'sky_split', 0.9)]
            if h.kind == 'slam_big':
                ev.append((real(h.ts) + 0.02, 'boom', 0.8))
            if h.kind == 'power':
                ev.append((real(h.ts) + 0.1, 'aura_loop', 0.6))
        for ts, name, g in self.sfx_story:
            ev.append((real(ts), name, g))
        self.real = real
        self.sfx_events = sorted(ev)
        self.duration = self.n * DT

    # ================================================================ state
    def reset(self):
        self.rng = np.random.RandomState(1234)
        self.cam = Camera(0, -50, 1.25)
        self.parts = Particles(8000)
        self.debris = Debris()
        self.rings = Rings()
        self.stars = Stars()
        self.decals = []
        self.hist = {'S': deque(maxlen=12), 'B': deque(maxlen=12)}
        self.ext = {k: {j: deque(maxlen=6) for j in ('fistf', 'fistb', 'toef', 'toeb')} for k in 'SB'}
        self.zk = 0.0
        self.post_imp = (-1, 0, 0)
        self.chroma_until = (-1, 0)
        self.flash = 0.0
        self.lines = (-1, None)
        self.front = 'S'
        self.combo = {'S': [0, -99], 'B': [0, -99]}
        self.flags = dict(charge=False, beam_on=False, ko=False)
        self.beam_tgt = None
        self.mega_i = None
        self.prev_lift = {'S': 0.0, 'B': 0.0}
        self.prev_x = {'S': -70.0, 'B': 70.0}
        for st in (self.S, self.B):
            st.burst = False
            st.dissolve = None
        self.a.apply(0)
        self.b.apply(0)
        self.S.settle(wind=(-120, 0))
        self.i = -1

    def set(self, k, v):
        self.flags[k] = v

    def power_up(self):
        self.B.burst = True
        c = self.B.point('shoulder')
        self.rings.add(c[0], c[1], 260, dur=0.8, color=(1, 0.4, 0.9), width=12)
        self.rings.add(c[0], 0, 320, dur=1.0, squash=0.2, color=(1, 0.5, 0.8), width=8)
        self.debris.spawn(self.rng, c[0], 0, 40, speed=(100, 360), ang=(200, 340), size=(2, 6),
                          color=NIGHT['ground'] * 1.8, grav=420)

    def fire(self):
        self.flags['beam_on'] = True
        self.flags['charge'] = False
        self.beam_tgt = self.S.point('shoulder').copy()

    def mega(self):
        self.mega_i = self.i

    def orb_geom(self):
        J = self.B.joints()
        m = (J['fistf'] + J['fistb']) / 2
        d = m - J['shoulder']
        d = d / (np.hypot(*d) + 1e-6)
        return m + d * 10, d

    def actor(self, k):
        return self.S if k == 'S' else self.B

    # ---------------------------------------------------------------- hit feedback
    def do_hit(self, h):
        k = h.k
        rng = self.rng
        att = self.actor(h.att) if h.att else None
        if h.pos is not None:
            p = np.array(h.pos, np.float32)
        else:
            p = att.point(h.joint).copy()
        if h.cb:
            h.cb(self)
        d = np.array(h.d if h.d is not None else (att.f if att else 1, -0.15), np.float32)
        d = d / (np.hypot(*d) + 1e-6)
        col = BLOCK if k.get('block') else (S_GLOW if h.att == 'S' else B_GLOW)
        if h.att:
            self.front = h.att
        if k.get('star'):
            self.stars.add(rng, p[0], p[1], 14 * k['star'], col, dur=0.1 + 0.04 * k['star'])
        n = k.get('n', 0)
        if n:
            base = math.degrees(math.atan2(d[1], d[0]))
            m = int(n * 0.7)
            self.parts.burst(rng, p[0], p[1], m, speed=(150, 520 * (0.6 + 0.2 * k.get('star', 1))),
                             ang=(base - 40, base + 40), life=(0.12, 0.45), c0=(1, 1, 0.9), c1=col, drag=3.5)
            self.parts.burst(rng, p[0], p[1], n - m, speed=(60, 260), life=(0.1, 0.35), c0=(1, 1, 1), c1=col,
                             drag=4)
        if k.get('ring'):
            self.rings.add(p[0], p[1], 34 * k['ring'], dur=0.3 + 0.05 * k['ring'], color=col, width=3 + 2 * k['ring'])
            if k['ring'] > 1.2:
                self.rings.add(p[0], p[1], 22 * k['ring'], dur=0.22, color=(1, 1, 1), width=3, delay=0.02)
        self.cam.add_trauma(k.get('trauma', 0))
        self.zk += k.get('zk', 0)
        if k.get('impact'):
            self.post_imp = (self.i, k['impact'], 1 if h.kind == 'mega' else 0)
        if k.get('chroma'):
            self.chroma_until = (self.i + int(k.get('stop', 0.1) * FPS) + 6, k['chroma'])
        if k.get('flash'):
            self.flash = max(self.flash, k['flash'])
        if k.get('lines'):
            self.lines = (self.i + int(k.get('stop', 0.1) * FPS) + 4, p.copy())
        if k.get('bolts'):
            self.bolt_until = (self.i + 20, p.copy())
        if k.get('crack'):
            x = self.actor('S' if h.att == 'B' else 'B').x
            self.decals.append(dict(x=x, r=0, t=self.i, seed=self.i, crack=0.6))
            self.parts.burst(rng, x, -1, 10, speed=(40, 140), ang=(190, 350), life=(0.4, 0.8), size=(3, 6),
                             c0=NIGHT['ground'] * 2, c1=NIGHT['ground'], drag=3, mode=DUST)
        if k.get('ground'):
            g = k['ground']
            x = p[0]
            self.decals.append(dict(x=x, r=18 * g, t=self.i, seed=self.i, crack=g))
            for a0 in (180, 200, 320, 340):
                self.parts.burst(rng, x, -2, int(10 * g), speed=(80, 300 * g), ang=(a0, a0 + 20), life=(0.6, 1.4),
                                 size=(4, 6 + 6 * g), c0=NIGHT['ground'] * 2.2, c1=NIGHT['ground'], drag=2.2,
                                 mode=DUST)
            self.debris.spawn(rng, x, -2, int(22 * g), speed=(100, 320 * g), ang=(200, 340), size=(1.5, 3 + 2 * g),
                              color=NIGHT['ground'] * 1.9, grav=450)
            self.rings.add(x, 0, 120 * g, dur=0.5 + 0.2 * g, squash=0.22, color=(1, 0.8, 0.6), width=4 + 3 * g)
        if k.get('wall'):
            bd = min(self.bldgs, key=lambda q: abs(q.x + q.w / 2 - p[0]))
            if abs(bd.x + bd.w / 2 - p[0]) < 90:
                cx = bd.x + bd.w / 2
                if bd.state == 0:
                    bd.state = 1
                    bd.hole = (p[1] if h.pos is None else -40)
                    nd = 30
                else:
                    bd.state = 2
                    nd = 70
                self.debris.spawn(rng, cx, -40, nd, speed=(80, 380), ang=(180, 360), size=(2, 7),
                                  color=hexc('#4a4466'), grav=420)
                self.parts.burst(rng, cx, -45, nd, speed=(30, 180), life=(1.0, 2.0), size=(6, 14),
                                 c0=hexc('#6a6078'), c1=hexc('#2a2436'), drag=1.8, mode=DUST, spread=18)
        if h.kind == 'power':
            self.parts.burst(rng, p[0], p[1], 120, speed=(150, 600), life=(0.3, 0.9), c0=(1, 0.9, 1),
                             c1=(1, 0.2, 0.7), drag=2.5)
        if h.kind == 'mega':
            for j in range(5):
                self.rings.add(p[0] + 20 + j * 45, p[1], 120 + j * 90, dur=0.7 + j * 0.12, color=(1, 1, 1), width=12,
                               squash=1.3, delay=0.25 + j * 0.05)
        if k.get('count') and h.att:
            c = self.combo[h.att]
            c[0] = c[0] + 1 if self.i - c[1] < 70 else 1
            c[1] = self.i
            other = 'B' if h.att == 'S' else 'S'
            self.combo[other][0] = 0

    # ---------------------------------------------------------------- simulation
    def step(self, i):
        self.i = i
        s = self.frames[i]
        t = i * DT
        frozen = i > 0 and self.frames[i - 1] == s
        rng = self.rng
        self.a.apply(s)
        self.b.apply(s)
        S, B = self.S, self.B
        if self.mega_i is not None:
            prog = ease_in(remap(s, 19.34, 20.55), 1.3) * 1.4
            c = B.point('shoulder')
            B.dissolve = (c[0] - 12 * B.f, c[1] + 4, prog, 44)
            if s >= 20.58:
                B.visible = False
        for h in self.trig.get(i, []):
            self.do_hit(h)
        wind = (-90, 0) if not self.flags['beam_on'] else (-600, 300)
        if not frozen:
            for st in (S, B):
                st.step(t, wind)
        # motion history
        for key, st in (('S', S), ('B', B)):
            if not frozen:
                self.hist[key].append((st.x, st.y, dict(st.pose), st.air, st.f))
                J = st.joints()
                for j in self.ext[key]:
                    self.ext[key][j].append(J[j].copy())
            lift = st.pose.get('lift', 0)
            vx = (st.x - self.prev_x[key]) / DT if not frozen else 0
            if not frozen and st.visible and lift < 2 and abs(vx) > 160 and self.every(2):
                self.parts.burst(rng, st.x - np.sign(vx) * 4, -1, 2, speed=(20, 60), ang=(200, 340),
                                 life=(0.3, 0.6), size=(3, 6), c0=NIGHT['ground'] * 2, c1=NIGHT['ground'],
                                 drag=3, mode=DUST)
                self.parts.burst(rng, st.x, -1, 3, speed=(80, 200), ang=(190, 240) if vx > 0 else (300, 350),
                                 life=(0.1, 0.3), c0=(1, 0.9, 0.6), c1=(1, 0.4, 0.1), drag=4)
            self.prev_x[key] = st.x
            self.prev_lift[key] = lift
        # charge: particle intake, rising rocks
        if self.flags['charge'] and not frozen:
            o, d = self.orb_geom()
            self.parts.attract = (o[0], o[1], 1700)
            n = 6
            a = rng.uniform(0, 6.28, n)
            rr = rng.uniform(60, 190, n)
            pos = np.stack([o[0] + np.cos(a) * rr, o[1] + np.sin(a) * rr], -1)
            tang = np.stack([-np.sin(a), np.cos(a)], -1) * 130
            cols = [(0.8, 0.6, 1.0), (0.5, 0.8, 1.0), (1, 0.5, 0.9)]
            self.parts.emit(pos, tang, rr / 260.0, 1.5, cols[i % 3], (1, 1, 1), drag=0.5, mode=DOT)
            if self.every(3):
                self.debris.spawn(rng, rng.uniform(-150, 260), -1, 1, speed=(30, 80), ang=(255, 285), size=(1.5, 5),
                                  color=NIGHT['ground'] * 1.7, grav=-90, life=(2.5, 3.5), spin=4)
            if self.every(5):
                self.cam.add_trauma(0.06 + 0.2 * remap(s, 15.3, 17.6))
        else:
            self.parts.attract = None
        if self.flags['beam_on'] and not frozen:
            gx = self.beam_tgt[0]
            if self.every(2):
                self.debris.spawn(rng, gx + rng.uniform(-20, 20), -2, 3, speed=(150, 420), ang=(200, 340),
                                  size=(2, 6), color=CHARGE['ground'] * 1.8, grav=450)
                self.parts.burst(rng, gx, -4, 12, speed=(200, 600), ang=(190, 350), life=(0.2, 0.6), c0=(1, 1, 1),
                                 c1=(0.5, 0.4, 1.0), drag=2)
            if self.every(12):
                self.rings.add(gx, 0, 160, dur=0.6, squash=0.22, color=(0.7, 0.6, 1.0), width=8)
            if self.every(3):
                self.cam.add_trauma(0.3)
        # boros aura embers
        if B.burst and B.visible and not frozen and self.every(2):
            J = B.joints()
            n = 2
            xs = J['hip'][0] + rng.uniform(-10, 10, n)
            ys = J['hip'][1] + rng.uniform(-30, 20, n)
            self.parts.emit(np.stack([xs, ys], -1), np.stack([rng.uniform(-20, 20, n), -rng.uniform(60, 160, n)], -1),
                            rng.uniform(0.3, 0.6, n), 1.2, (0.9, 0.5, 0.9), (0.5, 0.05, 0.4), drag=1, mode=EMBER)
        # serious punch: disintegration embers + shock stream
        if B.dissolve is not None and B.visible and not frozen:
            J = B.joints()
            ox, oy, prog, Rw = B.dissolve
            pts = []
            for a_, b_ in (('hip', 'neck'), ('neck', 'head'), ('shf', 'wrf'), ('shb', 'wrb'), ('hpf', 'anf'),
                           ('hpb', 'anb')):
                for u in np.linspace(0, 1, 6):
                    pts.append(J[a_] + (J[b_] - J[a_]) * u)
            pts = np.array(pts)
            fd = np.hypot(pts[:, 0] - ox, pts[:, 1] - oy) / Rw
            sel = pts[np.abs(fd - prog) < 0.12]
            if len(sel):
                sel = sel + rng.uniform(-2, 2, sel.shape)
                n = len(sel)
                self.parts.emit(sel, np.stack([rng.uniform(40, 160, n), rng.uniform(-90, 10, n)], -1),
                                rng.uniform(0.5, 1.3, n), rng.uniform(1, 2, n), (1, 0.9, 0.6), (1, 0.3, 0.1),
                                drag=1.0, grav=-30, mode=EMBER)
        if self.mega_i is not None and not frozen and s < 20.2:
            fp = S.point('fistf')
            self.parts.burst(rng, fp[0] + 10, fp[1], 16, speed=(300, 900), ang=(-35, 35), life=(0.2, 0.6),
                             c0=(1, 1, 1), c1=(0.8, 0.6, 1), drag=1)
        if s > 21 and self.every(3):
            self.parts.emit(np.array([[self.cam.x + rng.uniform(-220, 220), 2]]),
                            np.array([[rng.uniform(-10, 10), -rng.uniform(10, 30)]]), 3.0, 1.2, (1, 0.95, 0.8),
                            (0.6, 0.5, 0.4), mode=EMBER)
        dt_fx = DT * (0.25 if frozen else 1.0)
        self.parts.update(dt_fx, ground=0.0)
        self.debris.update(dt_fx, ground=0.0)
        self.rings.update(dt_fx)
        self.stars.update(DT * (0.35 if frozen else 1.0))
        self.update_cam(s, frozen)

    def every(self, n):
        return self.i % n == 0

    def update_cam(self, s, frozen):
        c = self.CAM(s)
        pts, ws = [], []
        for st, w in ((self.S, c['ws']), (self.B, c['wb'])):
            if st.visible and w > 0.01:
                pts.append(st.point('shoulder'))
                ws.append(w)
        if not pts:
            pts, ws = [self.S.point('shoulder')], [1.0]
        pts = np.array(pts)
        ws = np.array(ws)
        ctr = (pts * ws[:, None]).sum(0) / ws.sum()
        if len(pts) > 1 and ws.min() > 0.3:
            sx = abs(pts[0, 0] - pts[1, 0])
            sy = abs(pts[0, 1] - pts[1, 1])
        else:
            sx = sy = 0
        z = min(W * 0.62 / (sx + 110), H * 0.62 / (sy + 90)) * c['zb']
        z = float(np.clip(z, 0.5, 2.6))
        ty = min(ctr[1] + c['yo'], -30 / max(z, 0.6))
        cam = self.cam
        k = c['k']
        if not frozen:
            cam.x += (ctr[0] - cam.x) * k
            cam.y += (ty - cam.y) * k
            cam.zoom += (z - cam.zoom) * min(1, k * 0.8)
        self.zk *= 0.86
        self.flash *= 0.8
        cam.zoom_eff = cam.zoom * (1 + self.zk)
        cam.tick(decay=1.8)

    # ================================================================ rendering
    def palette(self, s):
        if s < 9.84:
            return NIGHT, None, None
        if s < 15.3:
            return mix_pal(NIGHT, BURST, ease_out(remap(s, 9.84, 10.3))), None, None
        if s < 19.3:
            return mix_pal(BURST, CHARGE, ease_io(remap(s, 15.3, 17.0))), None, None
        k = ease_out(remap(s, 19.4, 20.6), 2)
        return mix_pal(CHARGE, DAWN, k), k, (0, -60)

    def draw(self, i):
        s = self.frames[i]
        t = i * DT
        cam = self.cam
        base_zoom = cam.zoom
        cam.zoom = cam.zoom_eff
        fr = Frame(cam)
        pal, split_k, sun = self.palette(s)
        split = None
        if split_k is not None:
            zl = 1 + (cam.zoom - 1) * 0.15
            split = (cam.x * 0.15, (H * 0.28 - H / 2) / zl + cam.y * 0.15, 0.3, 8 + 120 * split_k)
        City.draw(CITY, fr, t, pal, ship=0.0, split=split, sun=sun,
                  cloud_speed=4 if s < 9.8 else (14 if s < 19.3 else 40 * (1 - remap(s, 19.3, 21)) + 3))
        self.draw_bldgs(fr, pal)
        self.draw_decals(fr, i)
        drw = np.random.RandomState(i * 7 + 1)
        # teleport streaks
        for act, col in ((self.a, S_GLOW), (self.b, B_GLOW)):
            for t0, t1 in act.hk:
                if t0 <= s < t1 + 0.06:
                    p0 = act.pos(t0 - 1e-4)
                    p1 = act.pos(t1)
                    u = remap(s, t0, t1)
                    w0 = np.array([p0[0], -p0[1] - 40])
                    w1 = np.array([p1[0], -p1[1] - 40])
                    a_ = cam.pts(w0 + (w1 - w0) * max(0, u - 0.5))
                    b_ = cam.pts(w0 + (w1 - w0) * min(1, u + 0.2))
                    m = MaskDraw().capsule(a_, b_, 1.0, 2.5 * cam.zoom).get()
                    fr.img[m] = 1.0
                    fr.glow[m] += col * 1.6
                    fr.glow[dilate(m, 2) & ~m] += col * 0.4
        order = ['B', 'S'] if self.front == 'S' else ['S', 'B']
        beam_sil = None
        for key in order:
            st = self.actor(key)
            if not st.visible:
                continue
            hist = self.hist[key]
            if len(hist) > 3:
                v = math.hypot(hist[-1][0] - hist[-3][0], hist[-1][2].get('lift', 0) - hist[-3][2].get('lift', 0))
                if v > 7:
                    col = S_GLOW if key == 'S' else B_GLOW
                    sel = list(hist)[-9::2][:-1]
                    for j, snap_ in enumerate(sel):
                        ghost(fr, st, snap_, col, 0.18 + 0.12 * j)
            for j, dq in self.ext[key].items():
                if len(dq) >= 3:
                    sp = np.hypot(*(dq[-1] - dq[-3])) / 2
                    if sp > 3.2:
                        col = (1, 0.96, 0.85) if key == 'S' else (0.95, 0.8, 1.0)
                        smear(fr, list(dq)[-5:], 2.6 * st.S['scale'], color=col, alpha=min(0.9, sp / 8))
            rims = [((0, -1), S_GLOW if key == 'B' else B_GLOW, 0.35, 1)]
            if self.flags['charge'] or self.flags['beam_on']:
                rims.append(((1, -1), hexc('#c0a0ff'), 0.8, 1))
            L = st.render(cam, rims=rims)
            if key == 'B' and st.burst:
                aura(fr, L[1], B_GLOW, t, 0.3, height=8, seed=3)
            if key == 'S' and self.flags['beam_on']:
                beam_sil = L
            draw(fr, L)
        self.debris.draw(fr)
        self.rings.draw(fr)
        self.parts.draw(fr)
        self.stars.draw(fr)
        # charge orb / beam
        if self.flags['charge'] or self.flags['beam_on']:
            o, d = self.orb_geom()
            r = 3 + 20 * ease_io(remap(s, 15.3, 17.4)) if self.flags['charge'] else 12
            orb(fr, o[0], o[1], r * (1 + 0.05 * math.sin(t * 40)), t, c_out=(0.65, 0.25, 1.0), c_mid=(0.45, 0.75, 1.0),
                inten=0.9 + 0.4 * remap(s, 16.5, 17.5))
            if self.every(2) and r > 4:
                for _ in range(1 + int(2 * remap(s, 16, 17.5))):
                    aa = drw.uniform(0, 6.28)
                    p1 = o + np.array([math.cos(aa), math.sin(aa)]) * r * drw.uniform(1.8, 4.0)
                    lightning(fr, drw, o, p1, color=(0.6, 0.6, 1.0), inten=1.3, branches=1, depth=4)
            if self.flags['beam_on']:
                v = self.beam_tgt - o
                ang = math.degrees(math.atan2(v[1], v[0]))
                ln = (np.hypot(*v) + 300) * ease_out(remap(s, 17.6, 17.7), 2)
                beam(fr, o[0], o[1], ang, ln, 20, t, inten=1.1)
        # serious punch shock cone
        if self.mega_i is not None and s < 20.3:
            fp = self.S.point('fistf')
            p = ease_out(remap(s, 19.3, 20.2), 2)
            spread = math.radians(8 + 30 * p)
            Lc = 60 + 900 * p
            apex = np.array([fp[0] + 2, fp[1]])
            fade = 1 - remap(s, 20.0, 20.3)
            for wk, col, gi in ((1.0, (0.7, 0.6, 1.0), 0.8), (0.55, (1, 1, 1), 1.5)):
                pts = [apex, apex + np.array([math.cos(spread * wk), -math.sin(spread * wk)]) * Lc,
                       apex + np.array([Lc * 1.05, 0]),
                       apex + np.array([math.cos(spread * wk), math.sin(spread * wk)]) * Lc]
                m = MaskDraw().poly(cam.pts(np.array(pts))).get()
                fr.img[m] = fr.img[m] * (1 - fade) + np.asarray(col) * fade
                fr.glow[m] += np.asarray(col) * gi * fade
        if 18.8 <= s < 19.3:
            fp = cam.w2s(*self.S.point('fistf'))
            p = remap(s, 18.8, 19.3)
            radial(fr.glow, fp[0], fp[1], 40 * cam.zoom, (1, 0.3, 0.2), 0.5 + 1.2 * p, steps=5)
            if self.every(2):
                c = self.S.point('fistf')
                for _ in range(2 + int(p * 3)):
                    aa = drw.uniform(0, 6.28)
                    lightning(fr, drw, c, c + np.array([math.cos(aa), math.sin(aa)]) * drw.uniform(6, 16),
                              color=(1, 0.5, 0.3), inten=1.4, branches=1, depth=3)
        if hasattr(self, 'bolt_until') and i < self.bolt_until[0] and self.every(2):
            p0 = self.bolt_until[1]
            for _ in range(3):
                aa = drw.uniform(0, 6.28)
                lightning(fr, drw, p0, p0 + np.array([math.cos(aa), math.sin(aa)]) * drw.uniform(30, 80),
                          color=(1, 0.8, 1), inten=1.4, branches=2)
        # ---- post
        i0, nimp, style = self.post_imp
        if i0 >= 0 and i0 <= i < i0 + nimp:
            k = i - i0
            if style == 0:
                fnp = (lambda o: impact_frame(o, 0)) if k % 2 == 0 else \
                    (lambda o: impact_frame(o, 1, c_dark=(0.95, 0.15, 0.2), c_light=(0.05, 0, 0.05)))
            else:
                mode = (k // 2) % 3
                fnp = [lambda o: impact_frame(o, 0), lambda o: impact_frame(o, 1, c_dark=(1, 1, 1), c_light=(0, 0, 0)),
                       lambda o: impact_frame(o, 0, c_dark=(0.85, 0.05, 0.1), c_light=(1, 0.95, 0.9))][mode]
            fr.post.setdefault('pre_ui', []).append(fnp)
        elif i < self.chroma_until[0]:
            kk = self.chroma_until[1]
            fr.post.setdefault('pre_ui', []).append(lambda o: chroma(o, kk))
        if beam_sil is not None and self.S.visible:
            rgbs, As, _ = beam_sil
            sil = np.clip(rgbs * 0.08 + 0.02, 0, 1)
            edge = As & ~erode(As)
            fr.ui.append(lambda o: (o.__setitem__(As, sil[As]), o.__setitem__(edge, 1.0)))
        if self.flash > 0.01:
            fl = self.flash
            fr.ui.append(lambda o: o.__setitem__(slice(None), o * (1 - fl) + fl))
        if i < self.lines[0]:
            lp = cam.w2s(*self.lines[1])
            rl = np.random.RandomState(i // 2)
            fr.ui.append(lambda o: speed_lines(o, rl, lp[0], lp[1], n=60, color=(1, 1, 1), alpha=0.55, r_in=(0.25, 0.6)))
        if 18.8 <= s < 19.3:
            lp = cam.w2s(*self.S.point('fistf'))
            rl = np.random.RandomState(i // 2)
            fr.ui.append(lambda o: speed_lines(o, rl, lp[0], lp[1], n=70, color=(1, 1, 1), alpha=0.45,
                                               r_in=(0.35, 0.7)))
        self.ui(fr, i, s, t)
        cam.zoom = base_zoom
        return fr.finish(bloom_k=0.85, vignette=0.4)

    def draw_bldgs(self, fr, pal):
        cam = fr.cam
        for bd in self.bldgs:
            md = MaskDraw()
            if bd.state < 2:
                x0, y0 = cam.w2s(bd.x, -bd.h)
                x1, y1 = cam.w2s(bd.x + bd.w, 0)
                md.d.rectangle([x0, y0, x1, y1], fill=1)
            else:
                pts = [(bd.x, 0), (bd.x, -34), (bd.x + bd.w * 0.2, -48), (bd.x + bd.w * 0.4, -26),
                       (bd.x + bd.w * 0.6, -56), (bd.x + bd.w * 0.8, -38), (bd.x + bd.w, -64), (bd.x + bd.w, 0)]
                md.poly(cam.pts(np.array(pts, np.float32)))
            m = md.get()
            if not m.any():
                continue
            fr.img[m] = pal['near'] * 1.5 + 0.03
            step = max(2, int(round(5 * cam.zoom)))
            win = m & ((XX.astype(int) % step) < max(1, step // 2)) & ((YY.astype(int) % (step + 1)) < max(1, step // 2))
            fr.img[win] = pal['win'] * 0.45
            edge = m & ~shift(m, -1, 0)
            fr.img[edge] = pal['rim'] * 0.5
            top = m & ~shift(m, 0, -1)
            fr.img[top] = pal['rim'] * 0.6
            if bd.state == 1:
                c = cam.w2s(bd.x + bd.w / 2, bd.hole)
                hm = MaskDraw().ellipse(c[0], c[1], 12 * cam.zoom, 15 * cam.zoom).get() & m
                fr.img[hm] = 0.02
                rim = dilate(hm, 1) & m & ~hm
                fr.img[rim] = (1, 0.6, 0.3)
                fr.glow[rim] += np.array([1, 0.4, 0.1]) * 0.6

    def draw_decals(self, fr, i):
        cam = fr.cam
        for d in self.decals:
            age = (i - d['t']) * DT
            if d['r'] > 0:
                c = cam.w2s(d['x'], 0)
                rr = d['r'] * cam.zoom
                m = MaskDraw().ellipse(c[0], c[1] + 1, rr, rr * 0.18).get()
                fr.img[m] = fr.img[m] * 0.35
                rim = dilate(m, 1) & ~m & (YY < c[1] + 1)
                fr.img[rim] = fr.img[rim] * 0.5 + 0.25
            if age < 2.0 and d['crack']:
                k = 1 - age / 2.0
                cracks(fr, d['seed'], d['x'], 0, 50 + 60 * d['crack'], ease_out(remap(age, 0, 0.2)),
                       color=(1, 0.55, 0.25), inten=1.2 * k, n=6)

    def ui(self, fr, i, s, t):
        combo = {k: list(v) for k, v in self.combo.items()}

        def f(o):
            if t < 1.25:
                p = ease_out(remap(t, 0.25, 0.4), 3)
                a = 1 - remap(t, 1.0, 1.25)
                if t >= 0.25:
                    size = int(round(110 - 66 * p))
                    blit_text(o, 'FIGHT!', W / 2, H / 2 - 30, color=(1, 1, 0.7), color2=(1, 0.3, 0.1),
                              outline=(0.1, 0, 0.02), path=FONT_BOLD, size=size, thick=2, shadow=3, alpha=a,
                              glowc=(1, 0.5, 0.1))
                    fl = 1 - remap(t, 0.25, 0.45)
                    o[:] = o * (1 - fl * 0.6) + fl * 0.6
            for key, (n, last) in combo.items():
                age = (i - last) * DT
                if n >= 2 and age < 1.3:
                    pop = 1 + 0.6 * (1 - ease_out(remap(age, 0, 0.12)))
                    a = 1 - remap(age, 1.0, 1.3)
                    x = 44 if key == 'B' else W - 44
                    col = (1, 0.85, 0.3) if key == 'S' else (1, 0.5, 1.0)
                    blit_text(o, str(n), x - 8, 34, color=(1, 1, 1), color2=col, outline=(0, 0, 0), path=FONT_BOLD,
                              size=int(22 * pop), thick=2, shadow=2, alpha=a)
                    blit_text(o, 'HITS', x + 22, 38, color=col, outline=(0, 0, 0), path=FONT_BOLD, size=10,
                              spacing=1, alpha=a)
            if s >= 21.0:
                p = ease_out(remap(s, 21.0, 21.15), 3)
                size = int(round(130 - 76 * p))
                blit_text(o, 'K.O.', W / 2, H / 2 - 36, color=(1, 1, 0.7), color2=(1, 0.25, 0.1),
                          outline=(0.12, 0, 0.02), path=FONT_BOLD, size=size, thick=2, shadow=3,
                          glowc=(1, 0.5, 0.1))
                fl = 1 - remap(s, 21.0, 21.2)
                o[:] = o * (1 - fl * 0.7) + fl * 0.7
            fo = remap(s, 22.8, 23.4)
            o[:] = o * (1 - fo)
        fr.ui.append(f)


CITY = City(3)
