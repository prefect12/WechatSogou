"""Render driver for the stickman fight.

  python render.py sheet [cols] [rows] [t0] [t1]   contact sheet -> out/sheet.png
  python render.py all [jobs]                      full video    -> out/saitama_vs_boros.mp4
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
from film import Film  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return 'ffmpeg'


def sheet(cols=6, rows=5, t0=0.0, t1=None):
    film = Film()
    t1 = film.duration if t1 is None else t1
    picks = set(np.linspace(int(t0 * FPS), int(t1 * FPS) - 1, cols * rows).astype(int))
    tiles = []
    tt = time.time()
    last = max(picks)
    for i in range(last + 1):
        film.step(i)
        if i in picks:
            tiles.append(film.draw(i))
    print('%d frames (%.1fs real), %.3fs/frame' % (film.n, film.duration, (time.time() - tt) / len(picks)))
    rws = [np.concatenate(tiles[r * cols:(r + 1) * cols], 1) for r in range(len(tiles) // cols)]
    os.makedirs(OUT, exist_ok=True)
    Image.fromarray(np.concatenate(rws, 0)).save(os.path.join(OUT, 'sheet.png'))


def render_chunk(args):
    ci, a, b = args
    film = Film()
    path = os.path.join(OUT, 'chunk_%02d.mp4' % ci)
    cmd = [ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', '%dx%d' % (W * SCALE, H * SCALE), '-r', str(FPS), '-i', '-',
           '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-tune', 'animation', '-pix_fmt', 'yuv420p', path]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for i in range(b):
        film.step(i)
        if i >= a:
            pr.stdin.write(upscale(film.draw(i)).tobytes())
    pr.stdin.close()
    pr.wait()
    print('  chunk %d [%d,%d) %.0fs' % (ci, a, b, time.time() - t0), flush=True)
    return path


def render_all(jobs=4):
    os.makedirs(OUT, exist_ok=True)
    film = Film()
    n = film.n
    k = jobs * 2
    # later frames are cheaper to reach but the simulation prefix grows - equal splits are fine
    bounds = np.linspace(0, n, k + 1).astype(int)
    work = [(ci, int(bounds[ci]), int(bounds[ci + 1])) for ci in range(k)]
    t0 = time.time()
    with Pool(jobs) as pool:
        pool.map(render_chunk, work[::-1], chunksize=1)
    lst = os.path.join(OUT, 'chunks.txt')
    with open(lst, 'w') as f:
        for ci in range(k):
            f.write("file 'chunk_%02d.mp4'\n" % ci)
    import audio
    wav = os.path.join(OUT, 'audio.wav')
    audio.build_film(film, wav)
    final = os.path.join(OUT, 'saitama_vs_boros.mp4')
    subprocess.check_call([ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', lst,
                           '-i', wav, '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-shortest',
                           '-movflags', '+faststart', final])
    print('done in %.0fs -> %s (%.1fs)' % (time.time() - t0, final, film.duration))


if __name__ == '__main__':
    if sys.argv[1] == 'sheet':
        sheet(*(float(x) if '.' in x else int(x) for x in sys.argv[2:]))
    elif sys.argv[1] == 'all':
        render_all(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
