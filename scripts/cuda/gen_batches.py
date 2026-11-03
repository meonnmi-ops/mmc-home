import numpy as np, sys
N, B = int(sys.argv[1]), int(sys.argv[2]); NV = int(sys.argv[3]) if len(sys.argv) > 3 else 30
DATA = sys.argv[4] if len(sys.argv) > 4 else 'experiments/run_F/data'
T = 32
corpus = np.fromfile('ref/full_test/corpus.bin', dtype=np.uint8)
rng = np.random.RandomState(42)
pos = rng.randint(0, len(corpus) - T - 2, size=N * B + NV)
for s, p in enumerate(pos):
    corpus[p:p+T].astype(np.int32).tofile(f'{DATA}/batch_{s+1}_ids.bin')
    corpus[p+1:p+T+1].astype(np.int32).tofile(f'{DATA}/batch_{s+1}_tg.bin')
print(f"[OK] {N*B+NV} batches (train 1..{N*B}, val {N*B+1}..{N*B+NV}) → {DATA}")
