"""Full attention block bwd: PyTorch vs MMC."""
import torch, numpy as np
from pathlib import Path

T, D, H = 8, 32, 4
HD = D // H
B = Path('ref/bwd_test'); B.mkdir(parents=True, exist_ok=True)
torch.manual_seed(4)

if not (B/'aXn.bin').exists():
    Xn   = torch.randn(T, D, dtype=torch.float64)
    Wqkv = torch.randn(3*D, D, dtype=torch.float64) * 0.3
    bqkv = torch.randn(3*D, dtype=torch.float64) * 0.1
    Wo   = torch.randn(D, D, dtype=torch.float64) * 0.3
    bo   = torch.randn(D, dtype=torch.float64) * 0.1
    dY   = torch.randn(T, D, dtype=torch.float64)
    Xn.numpy().tofile(B/'aXn.bin')
    Wqkv.numpy().tofile(B/'aWqkv.bin')
    bqkv.numpy().tofile(B/'abqkv.bin')
    Wo.numpy().tofile(B/'aWo.bin')
    bo.numpy().tofile(B/'abo.bin')
    dY.numpy().tofile(B/'adY.bin')
    print("Generated inputs")

Xn   = torch.from_numpy(np.fromfile(B/'aXn.bin', dtype=np.float64).reshape(T, D)).requires_grad_(True)
Wqkv = torch.from_numpy(np.fromfile(B/'aWqkv.bin', dtype=np.float64).reshape(3*D, D)).requires_grad_(True)
bqkv = torch.from_numpy(np.fromfile(B/'abqkv.bin', dtype=np.float64)).requires_grad_(True)
Wo   = torch.from_numpy(np.fromfile(B/'aWo.bin', dtype=np.float64).reshape(D, D)).requires_grad_(True)
bo   = torch.from_numpy(np.fromfile(B/'abo.bin', dtype=np.float64)).requires_grad_(True)
dY   = torch.from_numpy(np.fromfile(B/'adY.bin', dtype=np.float64).reshape(T, D))

# Forward
qkv = Xn @ Wqkv.T + bqkv
Q, K, V = qkv.split(D, dim=-1)
qh = Q.view(T, H, HD).transpose(0, 1)
kh = K.view(T, H, HD).transpose(0, 1)
vh = V.view(T, H, HD).transpose(0, 1)
mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
att = (qh @ kh.transpose(-2, -1)) / (HD ** 0.5)
att = att.masked_fill(mask == 0, float('-inf'))
att = torch.softmax(att, dim=-1)
y = att @ vh
y = y.transpose(0, 1).contiguous().view(T, D)
O = y
Y = O @ Wo.T + bo
Y.backward(dY)

refs = {
    'adWqkv': Wqkv.grad.numpy(),
    'adbqkv': bqkv.grad.numpy(),
    'adWo':   Wo.grad.numpy(),
    'adbo':   bo.grad.numpy(),
    'adXn':   Xn.grad.numpy(),
}

print()
print("=== attention block backward comparison ===")
worst = 0.0
missing = 0
for name, ref in refs.items():
    f = B / f'{name}.bin'
    if not f.exists():
        print(f"  ❌ {name:8s}  FILE MISSING")
        missing += 1
        continue
    mmc = np.fromfile(f, dtype=np.float64).reshape(ref.shape)
    d = np.abs(mmc - ref).max()
    worst = max(worst, d)
    flag = "✅" if d < 1e-10 else "❌"
    print(f"  {flag} {name:8s}  max={d:.4e}")

if missing > 0:
    print(f"\n❌ {missing} files missing — run MMC first")
    import sys; sys.exit(1)

if worst < 1e-10:
    print(f"\n🏆 PASS — attention backward < 1e-10")
else:
    print(f"\n❌ FAIL (worst={worst:.4e})")
