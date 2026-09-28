"""Tiny 3D renderer for the hero shots: perspective camera, sphere ray-casting, projected boxes.

Everything is evaluated per pixel on the native 384x216 grid and then dithered, so the
3D shots keep the same chunky pixel-art look as the rest of the film.
"""
import math

import numpy as np
from PIL import Image, ImageDraw

from engine import W, H, XX, YY, BAYER, hexc, radial, ease_out, ease_in, remap, clamp01


def nrm(v):
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


# ------------------------------------------------------------------ 3D value noise
def hash3(ix, iy, iz, seed=0):
    h = (ix * 374761393 + iy * 668265263 + iz * 1274126177 + seed * 1442695041) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF).astype(np.float32) / float(0xFFFFFF)


def vnoise3(x, y, z, seed=0):
    x0, y0, z0 = np.floor(x), np.floor(y), np.floor(z)
    fx, fy, fz = x - x0, y - y0, z - z0
    ix, iy, iz = x0.astype(np.int64), y0.astype(np.int64), z0.astype(np.int64)
    sx, sy, sz = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy), fz * fz * (3 - 2 * fz)
    out = 0
    for dx in (0, 1):
        wx = sx if dx else 1 - sx
        for dy in (0, 1):
            wy = sy if dy else 1 - sy
            for dz in (0, 1):
                wz = sz if dz else 1 - sz
                out = out + hash3(ix + dx, iy + dy, iz + dz, seed) * wx * wy * wz
    return out


def fbm3(p, octaves=4, seed=0, scale=1.0):
    tot, amp, norm, f = 0.0, 1.0, 0.0, scale
    for o in range(octaves):
        tot = tot + vnoise3(p[..., 0] * f, p[..., 1] * f, p[..., 2] * f, seed + o * 13) * amp
        norm += amp
        amp *= 0.5
        f *= 2.03
    return tot / norm


# ------------------------------------------------------------------ camera
class Cam3:
    def __init__(self, pos, target, fov=50.0, up=(0, 1, 0), roll=0.0):
        self.pos = np.asarray(pos, np.float64)
        f = nrm(np.asarray(target, np.float64) - self.pos)
        r = nrm(np.cross(up, f))
        u = np.cross(f, r)
        if roll:
            c, s = math.cos(math.radians(roll)), math.sin(math.radians(roll))
            r, u = r * c + u * s, u * c - r * s
        self.f, self.r, self.u = f, r, u
        self.k = 1.0 / math.tan(math.radians(fov) / 2)

    def rays(self):
        sx = (XX - W / 2) / (H / 2)
        sy = -(YY - H / 2) / (H / 2)
        d = self.f[None, None] * self.k + self.r[None, None] * sx[..., None] + self.u[None, None] * sy[..., None]
        return nrm(d)

    def project(self, P):
        v = np.asarray(P, np.float64) - self.pos
        z = v @ self.f
        x = v @ self.r
        y = v @ self.u
        zz = np.maximum(z, 1e-3)
        return W / 2 + x / zz * self.k * (H / 2), H / 2 - y / zz * self.k * (H / 2), z

    def pick(self, sx, sy, C=(0, 0, 0), R=1.0):
        """Point on a sphere under a screen pixel (for placing features)."""
        d = nrm(self.f * self.k + self.r * (sx - W / 2) / (H / 2) - self.u * (sy - H / 2) / (H / 2))
        oc = self.pos - np.asarray(C)
        b = d @ oc
        c = oc @ oc - R * R
        t = -b - math.sqrt(max(0.0, b * b - c))
        return nrm(self.pos + d * t - np.asarray(C))


def sphere(cam, rays, C=(0, 0, 0), R=1.0):
    oc = cam.pos - np.asarray(C)
    b = rays @ oc
    c = oc @ oc - R * R
    disc = b * b - c
    hit = disc > 0
    t = -b - np.sqrt(np.maximum(disc, 0))
    P = cam.pos + rays * t[..., None]
    N = (P - np.asarray(C)) / R
    dmin = np.sqrt(np.maximum(oc @ oc - b * b, 0))
    return hit & (t > 0), N, dmin


def stars(img, seed=3, n=500):
    rs = np.random.RandomState(seed)
    xs = rs.randint(0, W, n)
    ys = rs.randint(0, H, n)
    br = rs.uniform(0.15, 1.0, n) ** 2
    img[ys, xs] = np.maximum(img[ys, xs], br[:, None] * np.array([0.9, 0.95, 1.0]))


def q(v, levels=6):
    return np.floor(np.clip(v, 0, 1) * levels + BAYER * 0.999) / levels


def tangent_basis(p):
    a = np.array([0.0, 1.0, 0.0]) if abs(p[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    t1 = nrm(np.cross(a, p))
    t2 = np.cross(p, t1)
    return t1, t2


# ================================================================== EARTH: the serious-punch cloud cut
def earth_cut(fr, u, t, shake=(0.0, 0.0)):
    """u: 0..1 shot progress. Oblique orbital view; a straight gash races across the cloud deck."""
    img, glow = fr.img, fr.glow
    img[:] = hexc('#020308')
    stars(img, 11)
    ang = -0.3 + 0.07 * u
    cam = Cam3((1.02 * math.sin(ang), 0.78 - 0.04 * u, -1.02 * math.cos(ang)), (0.1, 0.62, 0.3), fov=70,
               roll=-16 + 3 * u)
    rays = cam.rays()
    hit, N, dmin = sphere(cam, rays)
    # atmosphere halo just beyond the limb
    halo = (~hit) & (dmin < 1.07)
    hk = np.clip((1.07 - dmin) / 0.07, 0, 1) ** 2
    img[halo] = img[halo] + (hexc('#4a9cff') * hk[halo][:, None] * 0.9)
    glow[halo] += hexc('#3a8aff') * hk[halo][:, None] * 0.4
    p0 = cam.pick(W * 0.22, H * 0.86)
    p1 = cam.pick(W * 0.86, H * 0.3)
    n = nrm(np.cross(p0, p1))
    tdir = np.cross(n, p0)
    Q = N
    L = nrm(np.array([-0.6, 0.75, -0.35]))
    lam = np.clip(Q @ L, 0, 1)
    view = np.clip(-(rays * Q).sum(-1), 0, 1)
    land = fbm3(Q * 2.4 + 5, 5, 31) > 0.54
    base = np.where(land[..., None], hexc('#3f7a3a'), hexc('#1b4fb0'))
    deep = fbm3(Q * 5 + 1, 3, 9)
    base = np.where(land[..., None], base * (0.85 + 0.3 * deep[..., None]), base * (0.8 + 0.4 * deep[..., None]))
    cl = fbm3(Q * 3.6 + np.array([t * 0.01, 0, 0]), 5, 37)
    cloud = cl > 0.47
    # ---- the cut
    dist = Q @ n                                    # sin(angular distance to the great circle)
    along = np.arctan2(Q @ tdir, Q @ p0)
    L_cut = 2.0 * ease_out(clamp01(u / 0.75), 2.2)
    w = 0.03 + 0.022 * clamp01(u * 1.5)
    taper = np.clip((L_cut - along) / 0.12, 0, 1) ** 0.6
    inside_len = (along > -0.02) & (along < L_cut)
    wl = w * taper
    gash = inside_len & (np.abs(dist) < wl)
    wall = inside_len & (np.abs(dist) >= wl) & (np.abs(dist) < wl * 1.9 + 0.004)
    shadow = gash & (np.abs(dist) > wl * 0.45) & ((dist * (L @ n)) > 0)
    # ---- origin burst: radial streaks blown out of the cloud deck
    a0 = np.arccos(np.clip(Q @ p0, -1, 1))
    t1, t2 = tdir, n
    bearing = np.arctan2(Q @ t2, Q @ t1)
    Rb = 0.05 + 0.2 * ease_out(clamp01(u / 0.5), 2)
    streak = np.sin(bearing * 34 + 3 * np.sin(bearing * 5)) > 0.15
    burst = a0 < Rb
    cloud = np.where(burst, streak & (a0 > 0.03) & (cl > 0.3), cloud)
    cloud = (cloud | wall) & ~gash
    ctex = 0.72 + 0.34 * fbm3(Q * 14 + 2, 3, 43) + 0.25 * (cl - 0.47)
    ccol = hexc('#e8ecf8')[None, None] * np.clip(ctex, 0.6, 1.05)[..., None]
    col = np.where(cloud[..., None], ccol, base)
    col = np.where(wall[..., None], hexc('#ffffff'), col)
    # cloud shadows on the ground (fake thickness)
    cs = fbm3((Q - L * 0.02) * 3.6 + np.array([t * 0.01, 0, 0]), 5, 37) > 0.47
    col = np.where((~cloud & cs)[..., None], col * 0.72, col)
    col = np.where(shadow[..., None], col * 0.55, col)
    shade = 0.1 + 1.0 * q(lam * 0.85 + 0.15, 7)
    col = col * shade[..., None]
    rim = (1 - view) ** 3
    col = col + hexc('#5aa8ff') * rim[..., None] * 0.7
    img[hit] = col[hit]
    # shockwave ring racing out from the origin
    for lag in (0.0, 0.15):
        rr = 0.03 + 1.1 * ease_out(clamp01((u - lag) / 0.9), 1.7)
        ring = hit & (np.abs(a0 - rr) < 0.006 + 0.004 * (1 - view))
        img[ring] = 1.0
        glow[ring] += np.array([1.0, 0.95, 0.85]) * 0.9 * (1 - u)
    sx, sy, _ = cam.project(p0 * 1.0)
    radial(glow, sx, sy, 40, (1, 0.95, 0.8), 0.9 * (1 - u) + 0.15, steps=5)


# ================================================================== MOON: shock rings from the jump
_rs = np.random.RandomState(8)
CRATERS = nrm(_rs.normal(size=(260, 3)))
CRAD = _rs.uniform(0.012, 0.09, 260) ** 1.4 * 2.2


def moon_impact(fr, u, t):
    img, glow = fr.img, fr.glow
    img[:] = hexc('#020206')
    stars(img, 5)
    cam = Cam3((-0.55 + 0.1 * u, 0.62, -1.35 + 0.08 * u), (0.1, -0.12, 0.0), fov=64, roll=12)
    rays = cam.rays()
    hit, N, dmin = sphere(cam, rays)
    Q = N
    L = nrm(np.array([0.7, 0.5, -0.4]))
    lam = np.clip(Q @ L, 0, 1)
    maria = fbm3(Q * 1.7 + 3, 4, 17)
    alb = 0.62 + 0.25 * fbm3(Q * 7, 3, 23) - 0.22 * (maria > 0.55)
    # craters: dark floor, shadowed on the side facing the light, bright rim
    cosang = Q @ CRATERS.T
    angs = np.arccos(np.clip(cosang, -1, 1))
    rel = angs / CRAD[None, None]
    inside = rel < 1.0
    rimz = (rel >= 1.0) & (rel < 1.25)
    towardL = ((Q[..., None, :] - CRATERS[None, None]) @ L) > 0
    floor_sh = (inside & towardL).any(-1)
    floor = inside.any(-1)
    rim_b = rimz.any(-1)
    alb = np.where(floor, alb * 0.8, alb)
    alb = np.where(floor_sh, alb * 0.45, alb)
    alb = np.where(rim_b & ~floor, alb * 1.15, alb)
    # impact features
    p0 = cam.pick(W * 0.56, H * 0.63)
    t1, t2 = tangent_basis(p0)
    a0 = np.arccos(np.clip(Q @ p0, -1, 1))
    bear = np.arctan2(Q @ t2, Q @ t1)
    sp = 0.5 + 0.5 * np.sin(bear * 47 + 4 * np.sin(bear * 7)) * np.sin(bear * 19 + 1.3)
    spikes = sp > 0.55
    g = ease_out(clamp01(u / 0.8), 2.2)
    # ejecta rays
    Rej = 0.08 + 0.75 * g
    rays_b = (a0 < Rej) & (sp > 0.62)
    alb = np.where(rays_b, np.minimum(1.0, alb + 0.35 * (1 - a0 / Rej)), alb)
    # central crater with spiky rim
    cr = 0.07
    alb = np.where(a0 < cr, alb * (0.25 + 1.5 * (a0 / cr) ** 3), alb)
    crim = (a0 >= cr) & (a0 < cr * (1.25 + 0.4 * sp))
    alb = np.where(crim, 1.1, alb)
    # concentric shock rings (with jagged streaks), each casting a thin shadow outward
    ring_mask = np.zeros(a0.shape, bool)
    for k, (r_end, lag) in enumerate(((0.13, 0.0), (0.24, 0.07), (0.37, 0.14), (0.52, 0.21))):
        r = cr + r_end * ease_out(clamp01((u - lag) / 0.7), 2.0)
        wk = 0.008 + 0.018 * sp ** 2
        band = (np.abs(a0 - r) < wk) & (sp > 0.2)
        shade_out = (a0 - r > wk) & (a0 - r < wk + 0.014)
        alb = np.where(shade_out, alb * 0.55, alb)
        alb = np.where(band, np.maximum(alb, 1.05), alb)
        ring_mask |= band
    lit = 0.06 + 1.0 * q(lam * 0.9 + 0.1, 7)
    col = hexc('#d4d2dc') * (alb * lit)[..., None]
    view = np.clip(-(rays * Q).sum(-1), 0, 1)
    img[hit] = col[hit]
    glow[hit & ring_mask] += np.array([1.0, 0.95, 0.85]) * 0.25 * (1 - u * 0.5)
    halo = (~hit) & (dmin < 1.012)
    img[halo] = 0.55
    # debris flying off the surface (analytic 3D ballistics, no state)
    rs = np.random.RandomState(12)
    n = 420
    dirs = nrm(rs.normal(size=(n, 3)) * np.array([1, 1, 1]))
    tang = dirs - p0[None] * (dirs @ p0)[:, None]
    tang = nrm(tang)
    up_amt = rs.uniform(0.15, 1.0, n)
    v = nrm(tang + p0[None] * up_amt[:, None]) * rs.uniform(0.15, 0.9, n)[:, None]
    tau = u * 1.4
    P = p0[None] * 1.005 + v * tau
    sx, sy, z = cam.project(P)
    ok = (z > 0.05) & (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
    size = np.clip(1.6 / z, 0, 3)
    for j in np.nonzero(ok)[0]:
        x, y = int(sx[j]), int(sy[j])
        s = int(size[j] > 1.4)
        img[y:y + 1 + s, x:x + 1 + s] = 0.92 * (1 - 0.3 * u)
    # Saitama: a streak leaving the impact
    d_out = nrm(p0 * 0.8 + nrm(cam.pos - p0) * 0.6 + t2 * 0.35)
    head = p0 + d_out * (0.05 + 2.2 * ease_in(clamp01(u / 0.9), 1.4))
    tail = p0 + d_out * 0.02
    (hx, hy, hz), (tx, ty, tz) = cam.project(head), cam.project(tail)
    if hz > 0.05:
        md = Image.new('1', (W, H))
        ImageDraw.Draw(md).line([(tx, ty), (hx, hy)], fill=1, width=2)
        m = np.array(md, bool)
        img[m] = 1.0
        glow[m] += np.array([1.0, 0.9, 0.7]) * 1.2
        radial(glow, hx, hy, 18, (1, 0.9, 0.7), 1.4, steps=4)
    sx0, sy0, _ = cam.project(p0)
    radial(glow, sx0, sy0, 36, (1, 0.95, 0.85), 1.0 * (1 - u) ** 2 + 0.15, steps=5)


# ================================================================== CITY: aerial shot, shockwave dome flattens the blocks
_crs = np.random.RandomState(21)
BLOCKS = []
for i in range(-7, 8):
    for j in range(-7, 8):
        cx, cz = i * 74 + _crs.uniform(-10, 10), j * 74 + _crs.uniform(-10, 10)
        if math.hypot(cx, cz) < 80:
            continue
        w, d = _crs.uniform(24, 50), _crs.uniform(24, 50)
        h = _crs.uniform(28, 150) * (1.0 if _crs.rand() > 0.15 else 1.6)
        BLOCKS.append((cx, cz, w, d, h, _crs.uniform(0.85, 1.15)))


def city_blast(fr, u, t, pal, x0=0.0):
    img, glow = fr.img, fr.glow
    a = 0.55 + 0.3 * u
    R = 360 - 40 * u
    cam = Cam3((R * math.sin(a), 120 + 30 * u, -R * math.cos(a)), (0, 55, 0), fov=64, roll=-8)
    # sky + ground
    g = np.clip(YY / H, 0, 1)
    img[:] = pal['top'] + (pal['low'] - pal['top']) * g[..., None]
    rays = cam.rays()
    tg = np.where(rays[..., 1] < -1e-3, -cam.pos[1] / np.minimum(rays[..., 1], -1e-3), 1e9)
    gp = cam.pos + rays * tg[..., None]
    ground = tg < 5000
    fog = np.clip(tg / 3000.0, 0, 1)
    road = ((np.abs(((gp[..., 0] + 37) % 74) - 37) < 6) | (np.abs(((gp[..., 2] + 37) % 74) - 37) < 6))
    gc = np.where(road[..., None], pal['ground2'] * 0.9, pal['ground'] * 1.1)
    gc = gc * (1 - fog[..., None]) + pal['low'] * fog[..., None]
    img[ground] = gc[ground]
    # shock radius
    rw = 15 + 520 * ease_out(clamp01(u / 0.95), 1.5)
    # flash-lit ground near the dome
    gd = np.hypot(gp[..., 0], gp[..., 2])
    near = ground & (gd < rw)
    img[near] = img[near] * 0.75 + np.array([1.0, 0.5, 0.8]) * 0.12
    # buildings (painter's order)
    im = Image.new('RGB', (W, H))
    im.paste(Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)))
    dr = ImageDraw.Draw(im)
    order = sorted(BLOCKS, key=lambda b: -((b[0] - cam.pos[0]) ** 2 + (b[1] - cam.pos[2]) ** 2))
    Ld = nrm(np.array([0.3, 0.8, -0.5]))
    debris = []
    for (cx, cz, w, d, h, kc) in order:
        dist = math.hypot(cx, cz)
        col_k = clamp01((rw - dist) / 70.0)
        hh = h * (1 - 0.75 * col_k)
        x0_, x1_, z0_, z1_ = cx - w / 2, cx + w / 2, cz - d / 2, cz + d / 2
        faces = [((0, 1, 0), [(x0_, hh, z0_), (x1_, hh, z0_), (x1_, hh, z1_), (x0_, hh, z1_)]),
                 ((0, 0, -1), [(x0_, 0, z0_), (x1_, 0, z0_), (x1_, hh, z0_), (x0_, hh, z0_)]),
                 ((0, 0, 1), [(x0_, 0, z1_), (x1_, 0, z1_), (x1_, hh, z1_), (x0_, hh, z1_)]),
                 ((-1, 0, 0), [(x0_, 0, z0_), (x0_, 0, z1_), (x0_, hh, z1_), (x0_, hh, z0_)]),
                 ((1, 0, 0), [(x1_, 0, z0_), (x1_, 0, z1_), (x1_, hh, z1_), (x1_, hh, z0_)])]
        fog_b = clamp01(math.hypot(cx - cam.pos[0], cz - cam.pos[2]) / 1800.0)
        for nv, pts in faces:
            nv = np.array(nv, float)
            ctr = np.mean(pts, 0)
            if nv @ (cam.pos - ctr) <= 0:
                continue
            sx, sy, z = cam.project(np.array(pts))
            if (z <= 1).any():
                continue
            lamb = 0.35 + 0.65 * max(0.0, nv @ Ld)
            to_c = nrm(-ctr * np.array([1, 0, 1]))
            flash = max(0.0, nv @ to_c) * (1 - clamp01((dist - rw) / 200.0)) * (1 - u) * 1.4
            base = (pal['near'] * 2.6 if nv[1] == 0 else pal['far'] * 2.0) * kc
            c = base * lamb + np.array([1.0, 0.55, 0.85]) * flash * 0.6
            c = c * (1 - fog_b) + pal['low'] * fog_b
            dr.polygon(list(zip(sx, sy)), fill=tuple(int(v) for v in np.clip(c * 255, 0, 255)))
            if nv[1] == 0 and hh > 20 and col_k < 0.3 and z.mean() < 900:
                for fy in np.arange(8, hh - 4, 10):
                    for fx in np.linspace(0.2, 0.8, 3):
                        p = np.array(pts[0]) + (np.array(pts[1]) - np.array(pts[0])) * fx + np.array([0, fy, 0])
                        wx, wy, wz = cam.project(p[None])
                        if kc * 7 % 1 > 0.35:
                            dr.point((float(wx[0]), float(wy[0])), fill=(255, 180, 110))
        if col_k > 0 and dist < rw:
            tb = dist
            debris.append((cx, cz, h, rw - tb))
    img[:] = np.asarray(im, np.float32) / 255.0
    # flying chunks from buildings the wave already passed
    rs = np.random.RandomState(4)
    for (cx, cz, h, since) in debris:
        tau = since / 900.0 * 1.2
        for k in range(4):
            dirv = nrm(np.array([cx, 0, cz])) * rs.uniform(60, 180) + np.array([0, rs.uniform(40, 160), 0])
            p = np.array([cx + rs.uniform(-10, 10), h * rs.uniform(0.5, 1.0), cz + rs.uniform(-10, 10)])
            p = p + dirv * tau + np.array([0, -300, 0]) * tau * tau
            if p[1] < 0:
                continue
            sx, sy, z = cam.project(p[None])
            if z[0] > 1 and 0 <= sx[0] < W - 2 and 0 <= sy[0] < H - 2:
                s = 2 if z[0] < 500 else 1
                img[int(sy[0]):int(sy[0]) + s, int(sx[0]):int(sx[0]) + s] = pal['near'] * 3.0
    # dome shell (fresnel rim) - ground hemisphere
    hit, N, _ = sphere(cam, rays, (0, 0, 0), rw)
    oc = cam.pos
    b = rays @ oc
    tt = -b - np.sqrt(np.maximum(b * b - (oc @ oc - rw * rw), 0))
    P = cam.pos + rays * tt[..., None]
    shell = hit & (P[..., 1] > 0)
    fres = (1 - np.abs((rays * N).sum(-1))) ** 2
    k = (1 - u) ** 0.7
    img[shell] = img[shell] * (1 - 0.35 * k) + np.array([1.0, 0.8, 1.0]) * (0.2 + 0.9 * fres[shell, None]) * k
    glow[shell] += np.array([1.0, 0.5, 0.9]) * (fres[shell, None] * 0.9) * k
    # ground ring + dust wall
    th = np.linspace(0, 2 * math.pi, 180)
    ringp = np.stack([np.cos(th) * rw, np.zeros_like(th), np.sin(th) * rw], -1)
    sx, sy, z = cam.project(ringp)
    ok = z > 1
    mk = Image.new('1', (W, H))
    dd = ImageDraw.Draw(mk)
    pts = [(float(a_), float(b_)) for a_, b_, o in zip(sx, sy, ok) if o]
    if len(pts) > 2:
        dd.line(pts, fill=1, width=2)
    m = np.array(mk, bool)
    img[m] = 1.0
    glow[m] += np.array([1.0, 0.7, 0.9]) * 0.8 * k
    for j in range(0, 180, 3):
        if ok[j]:
            rr = max(1.0, 26.0 / max(z[j], 1) * 60)
            radial(img, sx[j], sy[j] - rr * 0.5, rr, pal['ground'] * 2.5, 0.5 * k, steps=3)
    # the clash itself
    cx_, cy_, cz_ = cam.project(np.array([[0, 30, 0]]))
    radial(glow, cx_[0], cy_[0], 50, (1, 0.9, 1), 2.0 * (1 - u) + 0.5, steps=5)
    radial(img, cx_[0], cy_[0], 8, (1, 1, 1), 1.0, steps=2)
