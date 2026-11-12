#!/usr/bin/env python3
"""Cross-check C++ and Python parsers on the same raw token ids."""
import os
import subprocess
import sys

sys.path.insert(0, '/tmp')
from protocol_parse import parse_and_map, COORD_BASE, BOX_END, IM_END, NONE

DIR = os.environ.get('GPI_STUDENT_DIR', '')
BIN = os.environ.get('GPI_STUDENT_BIN', os.path.join(os.environ.get('GPI_BUILD_DIR', 'build'), 'student_v1'))
C = lambda v: COORD_BASE + v

CASES = [
    ('box_compact', [C(100), C(200), C(300), C(400), IM_END], 640, 480),
    ('box_strict', [C(100), C(200), C(300), C(400), BOX_END, IM_END], 640, 480),
    ('none_compact', [NONE, IM_END], 640, 480),
    ('none_strict', [NONE, BOX_END, IM_END], 640, 480),
    ('incomplete_no_end', [C(100), C(200), C(300), C(400)], 640, 480),
    ('unexpected_text', [C(100), C(200), C(300), C(400), 34607, IM_END], 640, 480),
    ('extra_coord', [C(100), C(200), C(300), C(400), C(500), IM_END], 640, 480),
    ('short_prefix', [C(100), C(200), C(300), BOX_END, IM_END], 640, 480),
    ('two_coords', [C(100), C(200), IM_END], 640, 480),
    ('dup_end', [C(100), C(200), C(300), C(400), IM_END, IM_END], 640, 480),
    ('after_end', [C(100), C(200), C(300), C(400), IM_END, 151649], 640, 480),
    ('empty', [], 640, 480),
    ('no_end_with_box', [C(100), C(200), C(300), C(400), BOX_END], 640, 480),
    ('degenerate', [C(500), C(500), C(500), C(500), IM_END], 640, 480),
    ('out_of_image', [C(0), C(0), C(999), C(999), IM_END], 480, 640),
    ('bad_prefix_none_junk', [NONE, 42, IM_END], 640, 480),
    ('unknown_ids', [150000, 151000, 152000, 153000, IM_END], 480, 640),
]


def run_cpp(ids, w, h):
    arg = ','.join(str(t) for t in ids)
    r = subprocess.run([BIN, DIR, 'ptest', arg, str(w), str(h)], capture_output=True, text=True)
    line = [l for l in r.stdout.splitlines() if l.startswith('PTEST')][0]
    kv = {}
    for tok in line.split():
        if '=' in tok:
            k, v = tok.split('=', 1)
            kv[k] = v
    xyxy = kv['xyxy'].split(',')
    return dict(status=kv['status'], valid=int(kv['valid']), strict=int(kv['strict']),
                none=int(kv['none']), box_end=int(kv['box_end']), ended=int(kv['ended']),
                historic=int(kv['historic']), box_ok=int(kv['box_ok']),
                out_of_image=int(kv['out_of_image']),
                box=[float(x) for x in xyxy], failure=kv['failure'])


print('%-22s %-18s %-18s %-7s %s' % ('case', 'cpp status', 'py status', 'same?', 'note'))
bad = 0
for name, ids, w, h in CASES:
    cpp = run_cpp(ids, w, h)
    py = parse_and_map(list(ids), w, h)
    same = (cpp['status'] == py['status'] and cpp['valid'] == int(py['valid']) and
            cpp['strict'] == int(py['strict']) and cpp['none'] == int(py['none']) and
            cpp['box_end'] == int(py['has_box_end']) and cpp['ended'] == int(py['ended']) and
            cpp['historic'] == int(py['historical_extractable']) and
            cpp['box_ok'] == int(py['box_ok']) and cpp['out_of_image'] == int(py['out_of_image']))
    dbox = max(abs(a - b) for a, b in zip(cpp['box'], py['box']))
    same = same and dbox < 1e-2
    if not same:
        bad += 1
    print('%-22s %-18s %-18s %-7s %s' % (name, cpp['status'], py['status'],
                                         'yes' if same else 'NO',
                                         '' if same else 'cpp=%s py=%s dbox=%.3f' % (cpp, py, dbox)))
print('cases %d, mismatches %d' % (len(CASES), bad))
sys.exit(1 if bad else 0)
