import os
import struct

import os
b = open(os.path.join(os.environ.get('GPI_STUDENT_DIR', '.'), 'vocab.bin'), 'rb').read()
cnt = struct.unpack('<I', b[:4])[0]
off = 4
d = {}
for i in range(cnt):
    tid, l = struct.unpack('<II', b[off:off + 8])
    off += 8
    d[tid] = b[off:off + l].decode('utf-8', 'replace')
    off += l
for k in (151645, 151649, 151650, 151651, 151652, 151653, 151654, 151655, 151656, 151660, 151666, 151667, 151668, 151669, 123183):
    print(k, repr(d.get(k)))
