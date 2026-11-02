import numpy as np, sys
from pathlib import Path
N = int(sys.argv[1]); NV = int(sys.argv[2]) if len(sys.argv) > 2 else 30
D = Path('experiments/run_F/loss')
def _rd(f):
    if not f.exists() or f.stat().st_size == 0: return None
    a = np.fromfile(f, dtype=np.float32)
    return float(a[0]) if a.size else None
tr = [_rd(D/f'loss_s{k}.bin') for k in range(1, N+1)]
va = [_rd(D/f'loss_val{v}.bin') for v in range(1, NV+1)]
trv = [x for x in tr if x is not None]
vav = [x for x in va if x is not None]
print(f"train got {len(trv)}/{N}  val got {len(vav)}/{NV}")
if trv:
    for a in range(0, len(trv), max(1, len(trv)//10)):
        e = min(a + len(trv)//10, len(trv))
        print(f"  steps {a+1:5d}-{e:5d}  train mean = {np.mean(trv[a:e]):.4f}")
if vav:
    print(f"  VAL mean={np.mean(vav):.4f}  std={np.std(vav):.4f}  sem={np.std(vav)/np.sqrt(len(vav)):.4f}  n={len(vav)}")
