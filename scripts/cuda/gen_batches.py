import numpy as np, sys
N, B = int(sys.argv[1]), int(sys.argv[2])
T = 32
corpus = np.fromfile('ref/full_test/corpus.bin', dtype=np.uint8)
rng = np.random.RandomState(42)
pos = rng.randint(0, len(corpus) - T - 2, size=N * B + 30)
for s, p in enumerate(pos):
    corpus[p:p+T].astype(np.int32).tofile(f'/tmp/batch_{s+1}_ids.bin')
    corpus[p+1:p+T+1].astype(np.int32).tofile(f'/tmp/batch_{s+1}_tg.bin')
print(f"[OK] {N*B+30} batches (train 1..{N*B}, val {N*B+1}..{N*B+30})")
