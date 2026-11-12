#!/usr/bin/env python3
"""Vision parity on board: PIL-vs-OpenCV preprocess, RKNN-vs-torch features."""
import os
import subprocess

import numpy as np

DIR = os.environ.get('GPI_STUDENT_DIR', '')
BIN = os.environ.get('GPI_STUDENT_BIN', os.path.join(os.environ.get('GPI_BUILD_DIR', 'build'), 'student_v1'))
VP = DIR + '/vp'
SUB = ['overfit64-000', 'overfit64-001', 'overfit64-004', 'overfit64-005', 'overfit64-020', 'overfit64-034']
IMG = {
    'overfit64-000': '000000000139.jpg',
}


def load(p):
    return np.fromfile(p, dtype=np.float32)


rows = [l.rstrip('\n').split('\t') for l in open(DIR + '/bench64.tsv') if l.strip()]
img_of = {r[0]: os.path.basename(r[1]) for r in rows}

print('%-14s %-10s %-10s %-10s %-10s %-10s' % ('sample', 'px_max', 'px_rms', 'v_max', 'v_rel_l2', 'v_cos'))
worst = [0, 0, 0, 0]
for sid in SUB:
    jpg = DIR + '/images/' + img_of[sid]
    pre = VP + '/' + sid + '-board-pixels.f32'
    subprocess.run([BIN, DIR, 'pre', jpg, pre], capture_output=True, check=True)
    px_ref = load(VP + '/' + sid + '-pixels.f32')
    px_brd = load(pre)
    px_max = float(np.abs(px_ref - px_brd).max())
    px_rms = float(np.sqrt(((px_ref - px_brd) ** 2).mean()))

    out = VP + '/' + sid + '-rkmn.f32'
    r = subprocess.run([BIN, DIR, 'vraw', VP + '/' + sid + '-pixels.f32', out], capture_output=True, text=True)
    assert 'VRAW_OK' in r.stdout, r.stdout + r.stderr
    a = load(VP + '/' + sid + '-torch.f32')
    b = load(out)
    v_max = float(np.abs(a - b).max())
    rel = float(np.linalg.norm(a - b) / np.linalg.norm(a))
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    # RKNN on board-preprocessed pixels (full vision path drift)
    out2 = VP + '/' + sid + '-rkmn-boardpx.f32'
    subprocess.run([BIN, DIR, 'vraw', pre, out2], capture_output=True, check=True)
    c = load(out2)
    v_max2 = float(np.abs(a - c).max())
    print('%-14s %-10.2e %-10.2e %-10.2e %-10.2e %-10.7f  (board-px v_max %.2e)' % (
        sid, px_max, px_rms, v_max, rel, cos, v_max2))
    worst = [max(worst[0], px_max), max(worst[1], px_rms), max(worst[2], v_max), max(worst[3], rel)]
print('worst: px_max %.2e px_rms %.2e vision_max %.2e vision_rel %.2e' % tuple(worst))
