"""Core pixel renderer: low-res canvas, masks, bloom, dithering, text, camera.

Everything is drawn natively at W x H (an arcade-style 384x216 grid) and only
upscaled with nearest-neighbour at the very end, so every effect lands on the
same crisp pixel grid.
"""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 384, 216
SCALE = 5
FPS = 60
DT = 1.0 / FPS

YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)

_B4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]],
               np.float32) / 16.0
BAYER = np.tile(_B4, (H // 4, W // 4))


def hexc(h, k=1.0):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32) * k


def clamp01(x):
    return max(0.0, min(1.0, x))


def lerp(a, b, t):
    return a + (b - a) * t


def remap(t, a, b):
    if b == a:
        return 1.0 if t >= b else 0.0
    return clamp01((t - a) / (b - a))


def ease_out(t, p=3):
    return 1 - (1 - clamp01(t)) ** p


def ease_in(t, p=3):
    return clamp01(t) ** p


def ease_io(t):
    t = clamp01(t)
    return t * t * (3 - 2 * t)


def ease_out_back(t, s=1.9):
    t = clamp01(t) - 1
    return t * t * ((s + 1) * t + s) + 1


def ease_out_expo(t):
    t = clamp01(t)
    return 1.0 if t >= 1 else 1 - 2 ** (-10 * t)


EASES = {'lin': lambda t: clamp01(t), 'out': ease_out, 'in': ease_in, 'io': ease_io,
         'back': ease_out_back, 'expo': ease_out_expo, 'step': lambda t: 0.0 if t < 1 else 1.0,
         'snap': lambda t: ease_out(t, 5)}


def mix(a, b, t):
    """Interpolate floats, tuples/arrays or dicts of floats."""
    if isinstance(a, dict):
        out = dict(a)
        for k, v in b.items():
            out[k] = mix(a.get(k, v), v, t)
        return out
    if isinstance(a, (tuple, list, np.ndarray)):
        return np.asarray(a, np.float32) + (np.asarray(b, np.float32) - np.asarray(a, np.float32)) * t
    return a + (b - a) * t


class Track:
    """Keyframes [(t, value, ease)] - ease is applied on the segment ending at that key."""

    def __init__(self, keys):
        self.keys = sorted(keys, key=lambda k: k[0])

    def __call__(self, t):
        ks = self.keys
        if t <= ks[0][0]:
            return ks[0][1]
        for i in range(1, len(ks)):
            if t <= ks[i][0]:
                t0, v0 = ks[i - 1][0], ks[i - 1][1]
                t1, v1 = ks[i][0], ks[i][1]
                e = ks[i][2] if len(ks[i]) > 2 else 'io'
                return mix(v0, v1, EASES[e](remap(t, t0, t1)))
        return ks[-1][1]


# ----------------------------------------------------------------- noise
def hash2(ix, iy, seed=0):
    h = (ix.astype(np.int64) * 374761393 + iy.astype(np.int64) * 668265263 + seed * 1442695041) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF).astype(np.float32) / float(0xFFFFFF)


def vnoise(x, y, seed=0):
    x = np.asarray(x, np.float32)
    y = np.asarray(y, np.float32)
    x0 = np.floor(x)
    y0 = np.floor(y)
    fx = x - x0
    fy = y - y0
    ix = x0.astype(np.int64)
    iy = y0.astype(np.int64)
    a = hash2(ix, iy, seed)
    b = hash2(ix + 1, iy, seed)
    c = hash2(ix, iy + 1, seed)
    d = hash2(ix + 1, iy + 1, seed)
    sx = fx * fx * (3 - 2 * fx)
    sy = fy * fy * (3 - 2 * fy)
    return (a + (b - a) * sx) + ((c + (d - c) * sx) - (a + (b - a) * sx)) * sy


def fbm(x, y, octaves=4, seed=0, lac=2.0, gain=0.5):
    tot = 0.0
    amp = 1.0
    norm = 0.0
    f = 1.0
    for i in range(octaves):
        tot = tot + vnoise(x * f, y * f, seed + i * 17) * amp
        norm += amp
        amp *= gain
        f *= lac
    return tot / norm


def n1(t, seed=0):
    """Smooth 1D noise in [-1, 1]."""
    return float(vnoise(np.array([t]), np.array([seed * 3.7]), seed)[0]) * 2 - 1


# ----------------------------------------------------------------- masks
class MaskDraw:
    """Aliased (pixel-perfect) vector drawing into a boolean mask."""

    def __init__(self):
        self.im = Image.new('1', (W, H), 0)
        self.d = ImageDraw.Draw(self.im)

    def poly(self, pts, fill=1):
        pts = [(float(x), float(y)) for x, y in pts]
        if len(pts) >= 3:
            self.d.polygon(pts, fill=fill)
        return self

    def circle(self, x, y, r, fill=1):
        if r < 0.5:
            self.d.point((x, y), fill=fill)
        else:
            self.d.ellipse([x - r, y - r, x + r, y + r], fill=fill)
        return self

    def ellipse(self, x, y, rx, ry, fill=1):
        self.d.ellipse([x - rx, y - ry, x + rx, y + ry], fill=fill)
        return self

    def line(self, pts, w=1, fill=1):
        pts = [(float(x), float(y)) for x, y in pts]
        self.d.line(pts, fill=fill, width=max(1, int(round(w))))
        return self

    def capsule(self, a, b, ra, rb, fill=1):
        a = np.asarray(a, np.float32)
        b = np.asarray(b, np.float32)
        d = b - a
        L = float(np.hypot(*d))
        if L > 1e-3:
            u = d / L
            n = np.array([-u[1], u[0]])
            self.poly([a + n * ra, b + n * rb, b - n * rb, a - n * ra], fill)
        self.circle(a[0], a[1], ra, fill)
        self.circle(b[0], b[1], rb, fill)
        return self

    def get(self):
        return np.array(self.im, dtype=bool)


def shift(m, dx, dy):
    """s[y, x] = m[y + dy, x + dx]  (out of range -> False/0)."""
    dx = int(dx)
    dy = int(dy)
    out = np.zeros_like(m)
    h, w = m.shape[:2]
    y0, y1 = max(0, -dy), min(h, h - dy)
    x0, x1 = max(0, -dx), min(w, w - dx)
    if y1 > y0 and x1 > x0:
        out[y0:y1, x0:x1] = m[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    return out


def dilate(m, r=1, diag=False):
    for _ in range(r):
        o = m | shift(m, 1, 0) | shift(m, -1, 0) | shift(m, 0, 1) | shift(m, 0, -1)
        if diag:
            o |= shift(m, 1, 1) | shift(m, -1, -1) | shift(m, 1, -1) | shift(m, -1, 1)
        m = o
    return m


def erode(m, r=1):
    return ~dilate(~m, r)


def box_blur(a, r):
    if r <= 0:
        return a
    k = 2 * r + 1
    p = np.pad(a, ((r + 1, r), (0, 0), (0, 0)))
    c = np.cumsum(p, axis=0)
    a = (c[k:] - c[:-k]) / k
    p = np.pad(a, ((0, 0), (r + 1, r), (0, 0)))
    c = np.cumsum(p, axis=1)
    return (c[:, k:] - c[:, :-k]) / k


def radial(layer, cx, cy, r, color, intensity=1.0, power=2.0, steps=0, squash=1.0):
    """Additive radial glow; optional dithered banding for a pixel look."""
    if r <= 0.5 or intensity <= 0:
        return
    x0, x1 = int(max(0, cx - r - 1)), int(min(W, cx + r + 2))
    y0, y1 = int(max(0, cy - r * squash - 1)), int(min(H, cy + r * squash + 2))
    if x1 <= x0 or y1 <= y0:
        return
    dx = (XX[y0:y1, x0:x1] - cx) / r
    dy = (YY[y0:y1, x0:x1] - cy) / (r * squash)
    v = np.clip(1 - np.sqrt(dx * dx + dy * dy), 0, 1) ** power
    if steps:
        v = np.floor(v * steps + BAYER[y0:y1, x0:x1]) / steps
    layer[y0:y1, x0:x1] += v[..., None] * (np.asarray(color, np.float32) * intensity)


def paint(img, mask, color, alpha=1.0):
    color = np.asarray(color, np.float32)
    if alpha >= 1:
        img[mask] = color
    else:
        img[mask] = img[mask] * (1 - alpha) + color * alpha


def dither_q(v, levels):
    return np.floor(v * levels + BAYER) / levels


# ----------------------------------------------------------------- camera
class Camera:
    def __init__(self, x=0.0, y=-45.0, zoom=1.0):
        self.x, self.y, self.zoom = x, y, zoom
        self.trauma = 0.0
        self.ox = self.oy = 0.0
        self.t = 0.0

    def add_trauma(self, a):
        self.trauma = min(1.5, self.trauma + a)

    def tick(self, dt=DT, decay=1.6):
        self.t += dt
        s = self.trauma ** 2 * 7.0
        self.ox = s * n1(self.t * 38, 1)
        self.oy = s * n1(self.t * 38, 2)
        self.trauma = max(0.0, self.trauma - decay * dt)

    def w2s(self, x, y, par=1.0):
        zl = 1 + (self.zoom - 1) * par
        return ((x - self.x * par) * zl + W / 2 + self.ox * par,
                (y - self.y * par) * zl + H / 2 + self.oy * par)

    def pts(self, arr):
        a = np.asarray(arr, np.float32)
        return np.stack([(a[..., 0] - self.x) * self.zoom + W / 2 + self.ox,
                         (a[..., 1] - self.y) * self.zoom + H / 2 + self.oy], -1)


# ----------------------------------------------------------------- frame
class Frame:
    def __init__(self, cam, bg=(0, 0, 0)):
        self.cam = cam
        self.img = np.zeros((H, W, 3), np.float32)
        self.img[:] = np.asarray(bg, np.float32)
        self.glow = np.zeros((H, W, 3), np.float32)
        self.ui = []          # deferred UI draw callbacks (after grading)
        self.post = {}

    def finish(self, bloom_k=0.9, thresh=0.72, gain=(1, 1, 1), lift=(0, 0, 0),
               vignette=0.35, sat=1.0, levels=40, raw=False):
        img, glow = self.img, self.glow
        bright = np.maximum(img - thresh, 0) * 1.4 + glow
        s1 = bright.reshape(H // 2, 2, W // 2, 2, 3).mean((1, 3))
        b1 = box_blur(box_blur(s1, 2), 2)
        s2 = s1.reshape(H // 4, 2, W // 4, 2, 3).mean((1, 3))
        b2 = box_blur(box_blur(s2, 5), 5)
        b1 = b1.repeat(2, 0).repeat(2, 1)
        b2 = b2.repeat(4, 0).repeat(4, 1)
        out = img + glow + (b1 * 0.9 + b2 * 1.1) * bloom_k
        over = np.clip(out.max(-1, keepdims=True) - 1, 0, 1.5)
        out = out + over * 0.55
        if sat != 1.0:
            lum = out.mean(-1, keepdims=True)
            out = lum + (out - lum) * sat
        out = out * np.asarray(gain, np.float32) + np.asarray(lift, np.float32)
        if vignette:
            d = ((XX - W / 2) / (W / 2)) ** 2 + ((YY - H / 2) / (H / 2)) ** 2
            out = out * (1 - vignette * np.clip(d - 0.35, 0, 1))[..., None]
        out = np.where(out < 0.8, out, 1 - 0.2 * np.exp(-(out - 0.8) / 0.2))
        out = np.clip(out, 0, 1)
        for fn in self.post.get('pre_ui', []):
            out = fn(out)
        for fn in self.ui:
            fn(out)
        if raw:
            return out
        return quantize(out, levels)


def quantize(out, levels=40):
    out = np.floor(np.clip(out, 0, 1) * levels + BAYER[..., None] * 0.999) / levels
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def upscale(u8, scanline=0.9):
    big = u8.repeat(SCALE, 0).repeat(SCALE, 1)
    if scanline < 1:
        big[SCALE - 1::SCALE] = (big[SCALE - 1::SCALE] * scanline).astype(np.uint8)
    return big


# ----------------------------------------------------------------- post fx
def chroma(out, k):
    k = int(k)
    if k <= 0:
        return out
    o = out.copy()
    o[..., 0] = shift(out[..., 0], -k, 0)
    o[..., 2] = shift(out[..., 2], k, 0)
    return o


def hsmear(out, r):
    r = int(r)
    if r <= 0:
        return out
    k = 2 * r + 1
    p = np.pad(out, ((0, 0), (r + 1, r), (0, 0)), mode='edge')
    c = np.cumsum(p, axis=1)
    return (c[:, k:] - c[:, :-k]) / k


def vsmear(out, r):
    return hsmear(out.transpose(1, 0, 2), r).transpose(1, 0, 2)


def impact_frame(out, mode=0, c_dark=(0.02, 0.0, 0.03), c_light=(1, 1, 1), thresh=None):
    """Anime-style 2-tone impact frame (adaptive threshold keeps silhouettes readable)."""
    lum = out.max(-1) * 0.6 + out.mean(-1) * 0.4
    if thresh is None:
        thresh = float(np.percentile(lum, 74))
    m = lum > thresh
    if mode == 1:
        m = ~m
    o = np.empty_like(out)
    o[:] = np.asarray(c_dark, np.float32)
    o[m] = np.asarray(c_light, np.float32)
    return o


def flash(out, color, a):
    if a <= 0:
        return out
    return out * (1 - a) + np.asarray(color, np.float32) * a


# ----------------------------------------------------------------- text
FONT_PIX = '/usr/share/fonts/opentype/unifont/unifont.otf'
FONT_CJK = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
FONT_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
_fonts = {}


def font(path, size):
    k = (path, size)
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype(path, size)
    return _fonts[k]


def text_mask(s, path=FONT_PIX, size=16, scale=1, spacing=0):
    f = font(path, size)
    widths = [f.getbbox(ch)[2] if ch != ' ' else size // 2 for ch in s]
    wtot = sum(widths) + spacing * max(0, len(s) - 1) + 4
    im = Image.new('1', (int(wtot), int(size * 1.4) + 4), 0)
    d = ImageDraw.Draw(im)
    d.fontmode = '1'
    x = 1
    for ch, w in zip(s, widths):
        d.text((x, 1), ch, font=f, fill=1)
        x += w + spacing
    m = np.array(im, dtype=bool)
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return m
    m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    if scale != 1:
        m = m.repeat(scale, 0).repeat(scale, 1)
    return m


def blit_text(out, s, x, y, color=(1, 1, 1), outline=(0, 0, 0), path=FONT_PIX, size=16,
              scale=1, anchor='c', color2=None, shadow=0, alpha=1.0, spacing=0, thick=1,
              glowc=None):
    """Draw text into a final (H,W,3) image. x,y is the anchor point."""
    m = text_mask(s, path, size, scale, spacing)
    th, tw = m.shape
    pad = thick + 2 + shadow
    M = np.zeros((th + pad * 2, tw + pad * 2), bool)
    M[pad:pad + th, pad:pad + tw] = m
    ol = dilate(M, thick, diag=True) & ~M
    if anchor == 'c':
        x0, y0 = int(round(x - tw / 2)) - pad, int(round(y - th / 2)) - pad
    elif anchor == 'l':
        x0, y0 = int(round(x)) - pad, int(round(y - th / 2)) - pad
    else:
        x0, y0 = int(round(x - tw)) - pad, int(round(y - th / 2)) - pad
    hh, ww = M.shape
    sx0, sy0 = max(0, -x0), max(0, -y0)
    sx1, sy1 = min(ww, W - x0), min(hh, H - y0)
    if sx1 <= sx0 or sy1 <= sy0:
        return
    reg = out[y0 + sy0:y0 + sy1, x0 + sx0:x0 + sx1]
    Mc = M[sy0:sy1, sx0:sx1]
    Oc = ol[sy0:sy1, sx0:sx1]
    if shadow:
        Sh = shift(M | ol, -shadow, -shadow)[sy0:sy1, sx0:sx1]
        reg[Sh] = reg[Sh] * (1 - alpha * 0.8)
    if glowc is not None:
        G = dilate(M, 3, diag=True)[sy0:sy1, sx0:sx1] & ~Oc & ~Mc
        reg[G] = reg[G] * (1 - alpha * 0.5) + np.asarray(glowc, np.float32) * alpha * 0.5
    if outline is not None:
        reg[Oc] = reg[Oc] * (1 - alpha) + np.asarray(outline, np.float32) * alpha
    if color2 is None:
        reg[Mc] = reg[Mc] * (1 - alpha) + np.asarray(color, np.float32) * alpha
    else:
        g = np.linspace(0, 1, hh)[sy0:sy1][:, None, None]
        grad = np.asarray(color, np.float32) * (1 - g) + np.asarray(color2, np.float32) * g
        grad = np.broadcast_to(grad, reg.shape)
        reg[Mc] = reg[Mc] * (1 - alpha) + grad[Mc] * alpha


def letterbox(out, h):
    h = int(round(h))
    if h > 0:
        out[:h] = 0
        out[H - h:] = 0


def subtitle(out, speaker, line, t, a_in=0.0, dur=99, color=(1, 0.9, 0.4)):
    """Typewriter subtitle living inside the bottom letterbox bar."""
    if t < a_in or t > a_in + dur:
        return
    lt = t - a_in
    n = int(lt * 22)
    s = line[:n]
    alpha = min(1.0, (a_in + dur - t) / 0.2) if dur < 99 else 1.0
    full = speaker + '：' + line
    m = text_mask(full, FONT_PIX, 16)
    x0 = W / 2 - m.shape[1] / 2
    if speaker:
        blit_text(out, speaker + '：', x0, H - 12, color=color, outline=(0, 0, 0), anchor='l', alpha=alpha)
        x0 += text_mask(speaker + '：', FONT_PIX, 16).shape[1] + 1
    if s:
        blit_text(out, s, x0, H - 12, color=(1, 1, 1), outline=(0, 0, 0), anchor='l', alpha=alpha)


def banner(out, t, title, sub, color=(1, 0.3, 0.8), side=1, y=62):
    """Sliding move-name banner (fighting-game super move callout)."""
    if t < 0 or t > 1.9:
        return
    a = ease_out(remap(t, 0, 0.22), 4)
    b = ease_in(remap(t, 1.55, 1.9), 3)
    off = (1 - a) * 320 * side + b * -320 * side
    h = 40
    y0 = int(y - h / 2)
    sl = 10
    band = np.zeros((H, W), bool)
    md = MaskDraw()
    md.poly([(0 + off, y0), (W + off, y0 - sl), (W + off, y0 + h - sl), (0 + off, y0 + h)])
    band = md.get()
    edge = dilate(band, 2) & ~band
    out[band] = out[band] * 0.15 + np.array([0.03, 0.02, 0.06]) * 0.85
    out[edge] = np.asarray(color, np.float32)
    blit_text(out, title, W / 2 + off - 20 * side, y - 7, color=(1, 1, 1), color2=color,
              outline=(0.05, 0, 0.08), path=FONT_CJK, size=26, thick=2, glowc=color, spacing=2)
    blit_text(out, sub, W / 2 + off + 30 * side, y + 12, color=color, outline=(0, 0, 0),
              path=FONT_BOLD, size=10, spacing=2)
