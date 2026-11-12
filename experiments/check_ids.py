#!/usr/bin/env python3
"""Compare C++ prompt token ids against the Python reference for all 64 samples."""
import os
import subprocess
import sys

DIR = os.environ.get('GPI_STUDENT_DIR', '')
BIN = os.environ.get('GPI_STUDENT_BIN', os.path.join(os.environ.get('GPI_BUILD_DIR', 'build'), 'student_v1'))
rows = [l.rstrip('\n').split('\t') for l in open(DIR + '/bench64.tsv') if l.strip()]
bad = 0
for f in rows:
    sid, _, query, _, _, _, _, _, _, ref = f
    out = subprocess.run([BIN, DIR, 'ids', query], capture_output=True, text=True).stdout
    line = [l for l in out.splitlines() if l.startswith('IDS')][0]
    got = [int(x) for x in line.split()[2:]]
    want = [int(x) for x in ref.split(',')]
    if got != want:
        bad += 1
        first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
        print('MISMATCH', sid, 'len', len(got), len(want), 'first_diff', first, got[first:first + 3], want[first:first + 3])
print('checked', len(rows), 'mismatches', bad)
sys.exit(1 if bad else 0)
