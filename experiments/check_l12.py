#!/usr/bin/env python3
"""L1/L2 checks on board against the PC float reference tokens.

L1: PC-computed prompt embeddings -> board LLM only (prompt length from the manifest).
L2: PC pixels -> board vision -> board LLM.
GP_LLM selects the rkllm build (w8a8 / fp16).
"""
import os
import os
import subprocess

DIR = os.environ.get('GPI_STUDENT_DIR', '')
BIN = os.environ.get('GPI_STUDENT_BIN', os.path.join(os.environ.get('GPI_BUILD_DIR', 'build'), 'student_v1'))
VP = DIR + '/vp'

rows = [l.rstrip('\n').split('\t') for l in open(DIR + '/bench64.tsv') if l.strip()]
info = {r[0]: dict(query=r[2], prompt=len(r[9].split(','))) for r in rows}
exp = {}
for l in open(DIR + '/expected_tokens.tsv'):
    sid, toks = l.rstrip('\n').split('\t')
    exp[sid] = [int(x) for x in toks.split(',')]


def run(args):
    r = subprocess.run([BIN, DIR] + args, capture_output=True, text=True)
    line = [l for l in r.stdout.splitlines() if l.startswith('IDS')]
    return ([int(x) for x in line[0].split()[2:]] if line else None), r.stdout


print('%-14s %-8s %-8s %s' % ('sample', 'L1', 'L2', 'detail'))
ok1 = ok2 = 0
for sid in sorted(exp):
    ids1, _ = run(['emb', DIR + '/l1/' + sid + '.f32', str(info[sid]['prompt'])])
    ids2, _ = run(['pix', VP + '/' + sid + '-pixels.f32', info[sid]['query']])
    s1, s2 = ids1 == exp[sid], ids2 == exp[sid]
    ok1 += s1
    ok2 += s2
    print('%-14s %-8s %-8s %s' % (sid, 'same' if s1 else 'DIFF', 'same' if s2 else 'DIFF',
                                  '' if s2 else 'got %s' % (ids2,)))
print('L1 identical %d/%d, L2 identical %d/%d' % (ok1, len(exp), ok2, len(exp)))
