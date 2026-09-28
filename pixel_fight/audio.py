"""Fully synthesized soundtrack: chiptune score that follows the edit + procedural SFX."""
import wave

import numpy as np

SR = 44100
RS = np.random.RandomState(7)


def T(d):
    return np.arange(int(d * SR)) / SR


def noise(d):
    return RS.uniform(-1, 1, int(d * SR))


def boxlp(x, L):
    """Variable-length moving-average lowpass (L scalar or per-sample array)."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    n = len(x)
    L = np.clip(np.broadcast_to(np.asarray(L, np.float64), (n,)).astype(int), 1, None)
    i = np.arange(n) + 1
    j = np.clip(i - L, 0, None)
    return (c[i] - c[j]) / (i - j)


def hp(x, L=4):
    return x - boxlp(x, L)


def dec(d, k):
    return np.exp(-T(d) * k)


def sweep(f0, f1, d, wave_='sine', curve=1.0):
    t = T(d)
    f = f0 + (f1 - f0) * (t / d) ** curve
    ph = np.cumsum(f) / SR
    return osc(ph, wave_)


def osc(ph, w, duty=0.5):
    ph = ph % 1.0
    if w == 'sine':
        return np.sin(ph * 2 * np.pi)
    if w == 'square':
        return np.where(ph < duty, 1.0, -1.0)
    if w == 'saw':
        return ph * 2 - 1
    if w == 'tri':
        return 1 - 4 * np.abs(ph - 0.5)
    raise ValueError(w)


def tone(f, d, w='sine', duty=0.5):
    return osc(T(d) * f, w, duty)


def fit(x, n):
    out = np.zeros(n)
    m = min(n, len(x))
    out[:m] = x[:m]
    return out


def mixl(*xs):
    n = max(len(x) for x in xs)
    return sum(fit(x, n) for x in xs)


# ------------------------------------------------------------------ sfx
def kick(d=0.35, f0=150, f1=42, k=9):
    return sweep(f0, f1, d, curve=0.35) * dec(d, k)


def crack(d=0.1, k=40):
    return hp(noise(d), 3) * dec(d, k)


def boom(d=1.6, f0=70, f1=28, lvl=1.0):
    return (sweep(f0, f1, d, curve=0.5) * dec(d, 2.8) * 1.2 + boxlp(noise(d), 30) * dec(d, 2.0) * 3.0) * lvl


def whoosh(d=0.4, peak=0.5, bright=3):
    t = T(d) / d
    L = 40 - (40 - bright) * np.exp(-((t - peak) / 0.22) ** 2)
    env = np.sin(np.pi * t) ** 2
    return boxlp(noise(d), L) * env * 3.5


def ring(fs, d=1.2, k=3.0):
    return sum(tone(f, d) * dec(d, k * (1 + i * 0.3)) for i, f in enumerate(fs)) / len(fs)


def crush(x, bits=6):
    q = 2 ** bits
    return np.round(x * q) / q


def punch(big=False):
    if big:
        return np.tanh(mixl(kick(0.6, 180, 38, 5) * 1.4, crack(0.15, 25) * 0.9, boom(1.0, 60, 30, 0.6)) * 1.6)
    return np.tanh(mixl(kick(0.25, 170, 50, 14), crack(0.08, 45) * 0.8) * 1.5)


def explode(d=2.2):
    n = boxlp(noise(d), np.linspace(6, 60, int(d * SR)))
    return np.tanh(mixl(n * dec(d, 1.6) * 3.5, kick(0.8, 90, 30, 4) * 1.2, crack(0.2, 20) * 0.6) * 1.3)


def rumble(d=2.0):
    t = T(d) / d
    return boxlp(noise(d), 90) * np.sin(np.pi * t) * 6


def roar(d, L=25, mod=6.0, base=55):
    t = T(d)
    n = boxlp(noise(d), L) * 4
    s = tone(base, d, 'saw') * 0.35
    am = 0.75 + 0.25 * np.sin(t * mod * 2 * np.pi)
    return np.tanh((n + s) * am * 1.5)


def sparkle(d=1.2, dens=90):
    out = np.zeros(int(d * SR))
    for _ in range(int(d * dens)):
        i = RS.randint(0, len(out) - 3000)
        g = tone(RS.uniform(2500, 6000), 0.05) * dec(0.05, 60) * RS.uniform(0.2, 0.6)
        out[i:i + len(g)] += g
    return out


def voice_wah(d=1.2, f0=420, f1=230):
    t = T(d)
    f = f0 + (f1 - f0) * (t / d) + 12 * np.sin(t * 2 * np.pi * 7)
    ph = np.cumsum(f) / SR
    x = osc(ph, 'square', 0.3) * 0.5 + osc(ph * 2, 'square', 0.25) * 0.2
    return boxlp(x, 4) * np.minimum(1, (d - t) * 4) * np.minimum(1, t * 30)


def sfx(name, dur=None):
    if name == 'wind':
        d = dur or 4.0
        t = T(d)
        am = 0.55 + 0.45 * np.sin(t * 0.9) * np.sin(t * 2.3 + 1)
        return boxlp(noise(d), 18) * am * 2.2 * np.minimum(1, t * 2) * np.minimum(1, (d - t) * 3)
    if name in ('drone', 'calm'):
        d = dur or 4.0
        t = T(d)
        x = boxlp(tone(55, d, 'saw') + tone(82.4, d, 'saw') * 0.6 + tone(110.3, d, 'saw') * 0.3, 18)
        return x * np.minimum(1, t) * np.minimum(1, (d - t) * 2) * 0.8
    if name == 'boom_soft':
        return boom(2.0, 60, 30, 0.5)
    if name == 'boom':
        return boom(2.0, 80, 26, 1.0)
    if name == 'eye':
        return mixl(sweep(300, 1400, 0.3, curve=2) * dec(0.3, 3) * 0.4, ring([1800, 2700, 3900], 0.9, 4) * 0.6)
    if name == 'ting':
        t = T(1.0)
        return (tone(2637, 1.0) + tone(3951, 1.0) * 0.5) * dec(1.0, 5) * (1 + 0.1 * np.sin(t * 40)) * 0.5
    if name == 'whoosh':
        return whoosh(0.4)
    if name == 'swish':
        return whoosh(0.13, 0.4, 2) * 0.6
    if name == 'slam':
        return np.tanh(mixl(punch(True), ring([220, 330, 495], 0.8, 5) * 0.4))
    if name == 'ready':
        return mixl(tone(880, 0.1, 'square', 0.25) * 0.3, np.concatenate([np.zeros(int(0.12 * SR)),
                                                                           tone(1318, 0.15, 'square', 0.25) * 0.3]))
    if name == 'fight':
        d = 0.9
        ch = sum(tone(f, d, 'square', 0.3) for f in (220, 329.6, 440, 554.4)) / 4
        return boxlp(ch, 3) * dec(d, 3) * 0.8
    if name == 'dash':
        return mixl(whoosh(0.35, 0.3, 2), kick(0.2, 120, 50, 18) * 0.7)
    if name == 'slowmo':
        return mixl(sweep(700, 70, 0.9, curve=0.6) * dec(0.9, 1.5) * 0.35, boxlp(noise(0.9), 40) * dec(0.9, 2) * 2)
    if name == 'punch_big':
        return punch(True)
    if name in ('impact', 'end_hit'):
        return np.tanh(mixl(boom(2.0, 90, 28, 1.2), ring([140, 211, 297], 1.6, 2) * 0.4) * 1.2)
    if name == 'crash':
        x = explode(2.4)
        for _ in range(40):
            i = RS.randint(int(0.1 * SR), int(1.5 * SR))
            c = crack(0.03, 80) * RS.uniform(0.2, 0.6)
            x[i:i + len(c)] += c
        return x
    if name == 'rumble':
        return rumble(2.2)
    if name == 'regen':
        t = T(1.7)
        return sweep(200, 700, 1.7, curve=1.5) * (0.5 + 0.5 * np.sin(t * 2 * np.pi * 14)) * 0.3 * np.minimum(1, (1.7 - t) * 3)
    if name in ('power_small', 'charge_small'):
        t = T(1.0)
        return mixl(sweep(150, 600, 1.0, 'saw', 1.5) * 0.15, boxlp(noise(1.0), np.linspace(40, 5, SR)) * t * 1.5)
    if name == 'charge':
        t = T(1.0)
        s = sweep(60, 260, 1.0, 'saw', 1.4) * (0.6 + 0.4 * np.sin(t * 2 * np.pi * (8 + 20 * t)))
        return np.tanh(mixl(boxlp(s, 6) * 0.6, boxlp(noise(1.0), np.linspace(50, 6, SR)) * t * 2.5) * 1.4)
    if name == 'burst':
        return mixl(explode(2.5), sparkle(1.5, 60) * 0.7, ring([660, 990, 1320], 1.5, 2.5) * 0.3)
    if name == 'aura_loop':
        d = dur or 3.0
        t = T(d)
        return roar(d, 30, 9, 58) * 0.5 * np.minimum(1, t * 3) * np.minimum(1, (d - t) * 3)
    if name == 'banner':
        return mixl(whoosh(0.2, 0.3, 2), ring([1500, 2250, 3375], 0.7, 5) * 0.4)
    if name == 'explode':
        return explode(2.2)
    if name == 'clash':
        return np.tanh(mixl(punch(True) * 1.2, ring([800, 1131, 1697, 2400], 1.8, 1.8) * 0.5, boom(2.0, 70, 25)) * 1.2)
    if name == 'hit':
        return punch() * 0.9
    if name == 'hit2':
        return mixl(punch() * 0.8, ring([1200, 1800], 0.3, 12) * 0.2)
    if name == 'rush_up':
        d = dur or 3.4
        t = T(d)
        L = np.linspace(40, 6, len(t))
        return boxlp(noise(d), L) * 3 * np.minimum(1, t * 4) * np.minimum(1, (d - t) * 6)
    if name == 'land':
        return mixl(boom(1.8, 70, 28), crack(0.1, 30) * 0.5)
    if name == 'jump':
        return mixl(boom(1.6, 90, 30), whoosh(0.5, 0.2, 2) * 0.8)
    if name == 'meteor':
        return mixl(sweep(2400, 300, 0.6, curve=0.7) * 0.25 * np.linspace(0.2, 1, int(0.6 * SR)),
                    boxlp(noise(0.6), np.linspace(30, 4, int(0.6 * SR))) * np.linspace(0.2, 3, int(0.6 * SR)))
    if name == 'charge_big':
        d = dur or 4.5
        t = T(d)
        f = 40 + 90 * (t / d) ** 1.5
        ph = np.cumsum(f) / SR
        s = osc(ph, 'saw') * 0.5 + osc(ph * 1.5, 'saw') * 0.3
        lfo = 0.6 + 0.4 * np.sin(np.cumsum(3 + 25 * (t / d) ** 2) / SR * 2 * np.pi)
        n = boxlp(noise(d), np.linspace(60, 8, len(t))) * (t / d) * 3
        return np.tanh((boxlp(s, 8) * lfo + n) * (0.4 + 0.8 * t / d)) * np.minimum(1, (d - t) * 20)
    if name == 'shout_boros':
        d = 0.9
        t = T(d)
        f = 110 - 25 * t
        ph = np.cumsum(f) / SR
        x = np.tanh((osc(ph, 'saw') + osc(ph * 1.01, 'saw') + boxlp(noise(d), 6)) * 2.5)
        return boxlp(x, 5) * np.minimum(1, t * 20) * np.minimum(1, (d - t) * 4) * 0.5
    if name == 'compress':
        d = 0.45
        t = T(d)
        return boxlp(noise(d), np.linspace(40, 2, len(t))) * (t / d) ** 3 * 4
    if name == 'beam':
        d = dur or 2.6
        t = T(d)
        return roar(d, 8, 14, 45) * np.minimum(1, t * 30) * np.minimum(1, (d - t) * 4) * 1.1
    if name == 'beam_inside':
        d = 1.4
        t = T(d)
        return boxlp(roar(d, 20, 10, 40), 12) * 1.5 * np.minimum(1, t * 10)
    if name == 'silence':
        return np.zeros(10)
    if name == 'serious':
        d = 2.0
        ch = sum(tone(f, d, 'saw') for f in (110, 164.8, 220, 277.2, 329.6)) / 5
        t = T(d)
        return mixl(kick(0.8, 90, 35, 3) * 1.2, boxlp(ch, 5) * np.minimum(1, t * 3) * np.minimum(1, (d - t) * 2) * 0.6)
    if name == 'clench':
        return mixl(crack(0.25, 12) * 0.5, kick(0.3, 100, 40, 10))
    if name == 'serious_punch':
        d = 3.0
        x = mixl(kick(1.2, 200, 25, 2.5) * 1.5, boxlp(noise(d), np.linspace(2, 80, int(d * SR))) * dec(d, 1.3) * 4,
                 crack(0.3, 10))
        return np.tanh(x * 2.0)
    if name == 'shockwave':
        return whoosh(1.3, 0.3, 3) * 1.4
    if name == 'sky_split':
        d = 2.2
        t = T(d)
        ch = sum(tone(f, d, 'square', 0.25) for f in (440, 554.4, 659.3, 880)) / 4
        sw = boxlp(noise(d), np.linspace(30, 2, len(t))) * np.minimum(1, t * 2) * 1.5
        return mixl(boxlp(ch, 4) * np.minimum(1, t * 8) * dec(d, 1.0) * 0.5, sw * dec(d, 1.5), sparkle(2.0, 50) * 0.5)
    if name == 'dissolve':
        return mixl(sparkle(1.3, 140) * 0.8, boxlp(noise(1.3), 12) * dec(1.3, 2) * 0.8)
    if name == 'shout':
        return voice_wah(1.2) * 0.9
    raise KeyError(name)


# ------------------------------------------------------------------ music
NOTE = {'C': -9, 'C#': -8, 'D': -7, 'D#': -6, 'E': -5, 'F': -4, 'F#': -3, 'G': -2, 'G#': -1, 'A': 0, 'A#': 1, 'B': 2}


def hz(n):
    name, octv = n[:-1], int(n[-1])
    return 440.0 * 2 ** ((NOTE[name] + (octv - 4) * 12) / 12)


class Score:
    def __init__(self, dur):
        self.buf = np.zeros(int(dur * SR) + SR * 4)

    def add(self, t, x, g=1.0):
        i = int(t * SR)
        if i < 0 or i >= len(self.buf):
            return
        n = min(len(x), len(self.buf) - i)
        self.buf[i:i + n] += x[:n] * g

    def note(self, t, d, f, w='square', g=0.2, duty=0.5, k=3.0, lp=2):
        x = tone(f, d, w, duty) * dec(d, k) * np.minimum(1, (d - T(d)) * 60) * np.minimum(1, T(d) * 400)
        self.add(t, boxlp(x, lp) if lp > 1 else x, g)

    def drums(self, t, g=1.0):
        pass


BPM = 150
BEAT = 60.0 / BPM
PROG = [('A', ['A4', 'C5', 'E5', 'C5']), ('F', ['F4', 'A4', 'C5', 'A4']), ('G', ['G4', 'B4', 'D5', 'B4']),
        ('E', ['E4', 'G#4', 'B4', 'G#4'])]
BASS = {'A': 'A2', 'F': 'F2', 'G': 'G2', 'E': 'E2'}
MELODY = [[('A5', 1), ('C6', .5), ('B5', .5), ('A5', 1), ('E5', 1)],
          [('F5', 1.5), ('G5', .5), ('A5', 1), ('C6', 1)],
          [('B5', 1), ('A5', .5), ('G5', .5), ('D5', 1), ('G5', 1)],
          [('G#5', 2), ('B5', 1), ('E6', 1)]]


def battle(sc, t0, t1, drums=True, melody=True, arps=True, g=1.0):
    bar = BEAT * 4
    t = t0
    k = 0
    kk = kick(0.3, 140, 45, 12)
    sn = mixl(hp(noise(0.16), 3) * dec(0.16, 18) * 0.7, tone(190, 0.1) * dec(0.1, 30) * 0.5)
    hh = hp(noise(0.04), 2) * dec(0.04, 90) * 0.35
    while t < t1 - 0.01:
        root, arp = PROG[k % 4]
        for e in range(8):
            te = t + e * BEAT / 2
            if te >= t1:
                break
            f = hz(BASS[root]) * (2 if e % 2 else 1)
            sc.note(te, BEAT / 2 * 0.9, f, 'square', 0.16 * g, duty=0.25, k=4, lp=3)
        if arps:
            for s in range(16):
                te = t + s * BEAT / 4
                if te >= t1:
                    break
                sc.note(te, BEAT / 4 * 0.8, hz(arp[s % 4]), 'square', 0.045 * g, duty=0.125, k=10, lp=2)
        if melody:
            tt = t
            for n, d in MELODY[k % 4]:
                if tt >= t1:
                    break
                sc.note(tt, d * BEAT * 0.95, hz(n), 'square', 0.07 * g, duty=0.5, k=1.2, lp=4)
                sc.note(tt, d * BEAT * 0.95, hz(n) * 1.005, 'tri', 0.05 * g, k=1.2)
                tt += d * BEAT
        if drums:
            for b in range(4):
                te = t + b * BEAT
                if te >= t1:
                    break
                if b in (0, 2):
                    sc.add(te, kk, 0.55 * g)
                if b == 2:
                    sc.add(te - BEAT / 4, kk, 0.35 * g)
                if b in (1, 3):
                    sc.add(te, sn, 0.5 * g)
                for h in range(2):
                    sc.add(te + h * BEAT / 2, hh, 0.5 * g)
        t += bar
        k += 1


def pad(sc, t0, d, freqs, g=0.12, attack=0.8, release=1.0):
    t = T(d)
    x = sum(tone(f, d, 'saw') + tone(f * 1.004, d, 'saw') for f in freqs) / (2 * len(freqs))
    x = boxlp(x, 10) * np.minimum(1, t / attack) * np.minimum(1, (d - t) / release)
    sc.add(t0, x, g)


def roll(sc, t0, t1, g=0.4):
    sn = hp(noise(0.1), 3) * dec(0.1, 30)
    t = t0
    while t < t1:
        p = (t - t0) / (t1 - t0)
        sc.add(t, sn, g * (0.3 + 0.7 * p))
        t += BEAT / 4 * (1 - 0.5 * p)


def build_score(starts, total):
    sc = Score(total)
    s = starts
    kk = kick(0.4, 100, 40, 8)
    # intro: dark pad + heartbeat
    pad(sc, 0.0, s['S04'] - 0.0, [hz('A2'), hz('E3'), hz('C4')], g=0.10, attack=2.5)
    t = 0.5
    while t < s['S04']:
        sc.add(t, kk, 0.35)
        sc.add(t + 0.25, kk, 0.2)
        t += 1.1
    # VS + ready: roll into the fight
    roll(sc, s['S04'] + 0.6, s['S05'] + 0.85, 0.35)
    pad(sc, s['S04'] + 0.55, s['S05'] + 0.85 - s['S04'] - 0.55, [hz('E3'), hz('B3'), hz('G#4')], g=0.08, attack=0.3)
    # battle theme
    battle(sc, s['S05'] + 0.85, s['S08'], g=1.0)
    battle(sc, s['S08'], s['S09'], drums=False, melody=False, g=0.7)
    battle(sc, s['S09'] + 0.95, s['S11'], g=1.1)
    battle(sc, s['S11'], s['S13'], drums=False, melody=True, arps=True, g=0.6)
    battle(sc, s['S13'] + 0.55, s['S14'], g=1.0)
    # cannon charge: pedal + accelerating roll
    t = s['S14']
    while t < s['S15'] - 0.3:
        sc.note(t, BEAT / 2 * 0.9, hz('E2'), 'square', 0.16, duty=0.25, k=4, lp=3)
        sc.note(t + BEAT / 4, BEAT / 4, hz('E3'), 'square', 0.06, duty=0.25, k=6, lp=3)
        t += BEAT / 2
    roll(sc, s['S14'] + 2.5, s['S15'] - 0.3, 0.3)
    # serious series: silence -> heroic swell
    pad(sc, s['S16'] + 1.8, 1.9, [hz('A2'), hz('E3'), hz('A3'), hz('C#4'), hz('E4')], g=0.16, attack=1.2, release=0.1)
    # sky split: triumphant A major phrase
    t0 = s['S17'] + 1.25
    pad(sc, t0, 2.4, [hz('A2'), hz('E3'), hz('A3'), hz('C#4'), hz('E4')], g=0.14, attack=0.05, release=1.2)
    for i, (n, d) in enumerate([('E5', 1), ('A5', 1), ('C#6', 1), ('E6', 2)]):
        sc.note(t0 + sum(x[1] for x in [('E5', 1), ('A5', 1), ('C#6', 1), ('E6', 2)][:i]) * BEAT * 0.8,
                d * BEAT * 0.8, hz(n), 'square', 0.07, k=1.0, lp=4)
    # ending: calm pad, final chord on the title card
    pad(sc, s['S18'], 2.3, [hz('F#3'), hz('A3'), hz('C#4'), hz('E4')], g=0.09, attack=0.6)
    pad(sc, s['S18'] + 3.6, 2.6, [hz('A2'), hz('E3'), hz('A3'), hz('C#4')], g=0.14, attack=0.02, release=2.0)
    t0 = s['S18'] + 3.6
    for i, n in enumerate(['A4', 'C#5', 'E5', 'A5']):
        sc.note(t0 + i * 0.12, 1.6 - i * 0.12, hz(n), 'square', 0.06, duty=0.25, k=1.5, lp=3)
    return sc.buf


def build(shots, path):
    starts = {}
    t = 0.0
    events = []
    for cls in shots:
        tag = cls.__name__[:3]
        starts[tag] = t
        for ev in cls.SFX:
            te, name, g = ev[:3]
            events.append((t + te, name, g, cls.dur - te))
        t += cls.dur
    total = t
    buf = np.zeros(int(total * SR) + SR * 4)
    for te, name, g, rem in events:
        x = sfx(name, dur=rem)
        i = int(te * SR)
        n = min(len(x), len(buf) - i)
        buf[i:i + n] += x[:n] * g * 0.55
    music = build_score(starts, total)
    n = min(len(buf), len(music))
    mix = buf[:n] + music[:n] * 0.9
    mix = mix[:int(total * SR)]
    fo = int(0.9 * SR)
    mix[-fo:] *= np.linspace(1, 0, fo) ** 2
    mix = np.tanh(mix * 1.1) * 0.92
    mix /= max(1e-6, np.abs(mix).max()) / 0.95
    # tiny stereo widening via delayed copy
    d = int(0.012 * SR)
    left = mix
    right = np.concatenate([mix[:d] * 0, mix[:-d]]) * 0.35 + mix * 0.65
    st = np.stack([left, right], -1)
    pcm = (np.clip(st, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return total


def build_film(film, path):
    """Soundtrack for the stickman fight: hit-synced SFX + a driving chiptune loop."""
    global BEAT
    total = film.duration
    buf = np.zeros(int(total * SR) + SR * 4)
    for te, name, g in film.sfx_events:
        x = sfx(name, dur=None)
        i = int(te * SR)
        if i >= len(buf):
            continue
        n = min(len(x), len(buf) - i)
        buf[i:i + n] += x[:n] * g * 0.6
    sc = Score(total)
    BEAT = 60.0 / 172
    r = film.real
    tC, tF, tW, tP, tKO = film.tC, film.tF, film.tW, film.tP, film.tKO
    t_fight = 0.25
    roll(sc, 0.0, t_fight, 0.35)
    battle(sc, t_fight, r(9.84), g=1.0)
    sc.add(r(9.84), kick(0.6, 120, 30, 3), 0.8)
    battle(sc, r(9.84), r(15.42), g=1.15)
    # launched to the moon: drums drop out, airy arps; moon: sparse pad
    battle(sc, r(15.42), r(17.2), drums=False, melody=True, g=0.7)
    pad(sc, r(17.2), r(18.6) - r(17.2), [hz('E3'), hz('B3'), hz('G4')], g=0.1, attack=0.3, release=0.3)
    roll(sc, r(18.3), r(18.6), 0.3)
    battle(sc, r(18.6), r(tC), g=1.15)
    t = r(tC)
    while t < r(tF):
        sc.note(t, BEAT / 2 * 0.9, hz('E2'), 'square', 0.18, duty=0.25, k=4, lp=3)
        sc.note(t + BEAT / 4, BEAT / 4, hz('E3'), 'square', 0.07, duty=0.25, k=6, lp=3)
        t += BEAT / 2
    roll(sc, r(tC + 1.1), r(tF), 0.35)
    battle(sc, r(tF), r(tW), g=1.1, melody=False)
    t_punch = r(tP)
    pad(sc, t_punch + 0.8, 3.2, [hz('A2'), hz('E3'), hz('A3'), hz('C#4'), hz('E4')], g=0.15, attack=0.05, release=1.4)
    mel = [('E5', 1), ('A5', 1), ('C#6', 1), ('E6', 2)]
    tt = t_punch + 0.8
    for n_, d in mel:
        sc.note(tt, d * BEAT * 0.9, hz(n_), 'square', 0.07, k=1.0, lp=4)
        tt += d * BEAT
    t_ko = r(tKO)
    battle(sc, t_ko, total, drums=True, melody=False, arps=True, g=0.6)
    pad(sc, t_ko, total - t_ko, [hz('A2'), hz('E3'), hz('A3'), hz('C#4')], g=0.12, attack=0.02, release=1.0)
    music = sc.buf
    n = min(len(buf), len(music))
    mix = buf[:n] + music[:n] * 0.8
    mix = mix[:int(total * SR)]
    fo = int(0.6 * SR)
    mix[-fo:] *= np.linspace(1, 0, fo) ** 2
    mix = np.tanh(mix * 1.1) * 0.92
    mix /= max(1e-6, np.abs(mix).max()) / 0.95
    d = int(0.012 * SR)
    right = np.concatenate([np.zeros(d), mix[:-d]]) * 0.35 + mix * 0.65
    pcm = (np.clip(np.stack([mix, right], -1), -1, 1) * 32767).astype(np.int16)
    with wave.open(path, 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
