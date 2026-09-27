"""Render driver.

  python render.py sheet S07 [cols]      contact sheet of a shot -> out/sheet_S07.png
  python render.py all [--jobs 4]        render every shot, add audio -> out/saitama_vs_boros.mp4
"""
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import W, H, SCALE, FPS, upscale  # noqa: E402
import shots_a  # noqa: E402

try:
    import shots_b  # noqa: E402
except ImportError:  # pragma: no cover
    shots_b = None

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return 'ffmpeg'


def all_shots():
    mods = [shots_a] + ([shots_b] if shots_b else [])
    out = []
    for m in mods:
        names = sorted(n for n in dir(m) if n[:1] == 'S' and n[1:3].isdigit())
        out += [getattr(m, n) for n in names]
    return out


def find(tag):
    for c in all_shots():
        if c.__name__.startswith(tag):
            return c
    raise SystemExit('no shot ' + tag)


def sheet(tag, cols=4, rows=4):
    cls = find(tag)
    shot = cls(seed=int(tag[1:3]))
    n = shot.nframes()
    picks = set(np.linspace(0, n - 1, cols * rows).astype(int))
    tiles = []
    t0 = time.time()
    for i in range(n):
        img = shot.render(i)
        if i in picks:
            tiles.append(img)
    dt = (time.time() - t0) / n
    rws = [np.concatenate(tiles[r * cols:(r + 1) * cols], 1) for r in range(len(tiles) // cols)]
    im = np.concatenate(rws, 0)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, 'sheet_%s.png' % tag)
    Image.fromarray(im).save(p)
    print('%s: %d frames, %.3fs/frame -> %s' % (cls.__name__, n, dt, p))


def render_shot(args):
    idx, cls_name = args
    cls = [c for c in all_shots() if c.__name__ == cls_name][0]
    shot = cls(seed=idx)
    path = os.path.join(OUT, 'seg_%02d.mp4' % idx)
    cmd = [ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', '%dx%d' % (W * SCALE, H * SCALE), '-r', str(FPS), '-i', '-',
           '-c:v', 'libx264', '-preset', 'medium', '-crf', '17', '-tune', 'animation', '-pix_fmt', 'yuv420p', path]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for i in range(shot.nframes()):
        pr.stdin.write(upscale(shot.render(i)).tobytes())
    pr.stdin.close()
    pr.wait()
    print('  %-22s %5.1fs' % (cls_name, time.time() - t0), flush=True)
    return path


def render_all(jobs=4):
    os.makedirs(OUT, exist_ok=True)
    shots = all_shots()
    work = sorted(enumerate(c.__name__ for c in shots), key=lambda w: -shots[w[0]].dur)
    t0 = time.time()
    with Pool(jobs) as pool:
        pool.map(render_shot, work, chunksize=1)
    lst = os.path.join(OUT, 'segments.txt')
    with open(lst, 'w') as f:
        for i in range(len(shots)):
            f.write("file 'seg_%02d.mp4'\n" % i)
    import audio
    wav = os.path.join(OUT, 'audio.wav')
    audio.build(shots, wav)
    final = os.path.join(OUT, 'saitama_vs_boros.mp4')
    subprocess.check_call([ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', lst,
                           '-i', wav, '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-shortest',
                           '-movflags', '+faststart', final])
    print('done in %.0fs -> %s' % (time.time() - t0, final))


if __name__ == '__main__':
    if sys.argv[1] == 'sheet':
        sheet(sys.argv[2], *(int(a) for a in sys.argv[3:]))
    elif sys.argv[1] == 'all':
        render_all(int(sys.argv[3]) if len(sys.argv) > 3 else 4)
