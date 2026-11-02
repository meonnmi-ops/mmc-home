import numpy as np, sys
D, DFF = int(sys.argv[1]), int(sys.argv[2])
T, L, V, TMAX = 32, 3, 256, 32
rng = np.random.RandomState(1234)
n = lambda *s, sd=0.02: (rng.randn(*s) * sd).astype(np.float32)
res = 0.02 / np.sqrt(2 * L)          # residual-branch outputs scaled down
w = {
 'wte': n(V, D), 'wpe': n(TMAX, D),
 'lnfg': np.ones(D, np.float32), 'lnfb': np.zeros(D, np.float32),
 'wq': n(L, D, D), 'wk': n(L, D, D), 'wv': n(L, D, D), 'wo': n(L, D, D, sd=res),
 'ln1g': np.ones((L, D), np.float32), 'ln1b': np.zeros((L, D), np.float32),
 'w1': n(L, D, DFF), 'b1': np.zeros((L, DFF), np.float32),
 'w2': n(L, DFF, D, sd=res), 'b2': np.zeros((L, D), np.float32),
 'ln2g': np.ones((L, D), np.float32), 'ln2b': np.zeros((L, D), np.float32),
 'dy': n(T, D),
}
for k, v in w.items(): v.astype(np.float32).tofile(f'experiments/run_F/data/fm_{k}.bin')
np.fromfile('experiments/run_F/data/batch_1_ids.bin', dtype=np.int32).tofile('experiments/run_F/data/fm_ids.bin')
np.fromfile('experiments/run_F/data/batch_1_tg.bin', dtype=np.int32).tofile('experiments/run_F/data/fm_tg.bin')
print(f"[OK] init D={D} DFF={DFF}, params ~ {sum(v.size for k,v in w.items() if k!='dy'):,}")
