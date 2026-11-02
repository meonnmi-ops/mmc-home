#!/usr/bin/env python3
"""Generate ids_multi.bin and tg_multi.bin. Usage: gen_multi_batches.py [N_STEPS] [BATCH]"""
import sys
import numpy as np
from pathlib import Path

n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 100
BATCH   = int(sys.argv[2]) if len(sys.argv) > 2 else 1
T = 128
SEED = 42

corpus_path = Path('ref/full_test/corpus.bin')
if not corpus_path.exists(): sys.exit(f'ERROR: {corpus_path} not found')
corpus = np.fromfile(corpus_path, dtype=np.uint8)
maxpos = len(corpus) - T - 1
print(f'Corpus: {len(corpus)}B, N_STEPS={n_steps}, BATCH={BATCH}, T={T}')

ids = np.zeros(n_steps * BATCH * T, dtype=np.int32)
tgs = np.zeros(n_steps * BATCH * T, dtype=np.int32)

state = SEED
for s in range(n_steps):
    for b in range(BATCH):
        state = (state * 25173 + 13849) % 65536
        pos = state % maxpos
        idx = (s*BATCH + b) * T
        ids[idx:idx+T] = corpus[pos:pos+T]
        tgs[idx:idx+T] = corpus[pos+1:pos+T+1]

Path('ref/full_test/ids_multi.bin').write_bytes(ids.tobytes())
Path('ref/full_test/tg_multi.bin').write_bytes(tgs.tobytes())
print(f'Wrote {n_steps*BATCH*T} ids + {n_steps*BATCH*T} tg ({n_steps}x{BATCH}x{T})')
