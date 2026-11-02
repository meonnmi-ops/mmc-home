import numpy as np, sys
N = int(sys.argv[1])
tr = [float(np.fromfile(f'/tmp/loss_s{k}.bin', dtype=np.float32)[0]) for k in range(1, N+1)]
va = [float(np.fromfile(f'/tmp/loss_val{v}.bin', dtype=np.float32)[0]) for v in range(1, 31)]
for a in range(0, N, N//10):
    print(f"  steps {a+1:5d}-{a+N//10:5d}  train mean = {np.mean(tr[a:a+N//10]):.4f}")
print(f"  VAL mean = {np.mean(va):.4f}  std = {np.std(va):.4f}   (unigram 2.410 | bigram 1.314)")
