#!/usr/bin/env python3
"""Generate ids_multi.bin and tg_multi.bin for multi-step training."""
import sys
import numpy as np
from pathlib import Path

n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 100
T = 128
SEED = 42

corpus_path = Path('ref/full_test/corpus.bin')
if not corpus_path.exists():
    print(f'ERROR: {corpus_path} not found')
    sys.exit(1)

corpus = np.fromfile(corpus_path, dtype=np.uint8)
n = len(corpus)
maxpos = n - T - 1
print(f'Corpus: {n} bytes, maxpos={maxpos}')

ids_multi = np.zeros(n_steps * T, dtype=np.int32)
tg_multi = np.zeros(n_steps * T, dtype=np.int32)

state = SEED
for s in range(n_steps):
    state = (state * 25173 + 13849) % 65536
    pos = state % maxpos
    ids_multi[s*T:(s+1)*T] = corpus[pos:pos+T]
    tg_multi[s*T:(s+1)*T] = corpus[pos+1:pos+T+1]

Path('ref/full_test/ids_multi.bin').write_bytes(ids_multi.tobytes())
Path('ref/full_test/tg_multi.bin').write_bytes(tg_multi.tobytes())
print(f'Wrote {n_steps*T} ids + {n_steps*T} tg')
