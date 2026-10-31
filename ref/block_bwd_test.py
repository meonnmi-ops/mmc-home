"""Full transformer block bwd: PyTorch vs MMC."""
import torch, numpy as np, sys
from pathlib import Path

T, D, DFF, H = 8, 32, 64, 4
HD = D // H
B = Path('ref/block_test'); B.mkdir(parents=True, exist_ok=True)
torch.manual_seed(5)

if not (B/'bX.bin').exists():
    torch.save({}, B/'_init.pt')
    data = {
        'bX':    torch.randn(T, D),
        'bWqkv': torch.randn(3*D, D) * 0.3,
        'bbqkv': torch.randn(3*D) * 0.1,
        'bWo':   torch.randn(D, D) * 0.3,
        'bbo':   torch.randn(D) * 0.1,
        'bW1':   torch.randn(DFF, D) * 0.3,
        'bb1':   torch.randn(DFF) * 0.1,
        'bW2':   torch.randn(D, DFF) * 0.3,
        'bb2':   torch.randn(D) * 0.1,
        'bln1w': torch.ones(D),
        'bln1b': torch.zeros(D),
        'bln2w': torch.ones(D),
        'bln2b': torch.zeros(D),
        'bdY':   torch.randn(T, D),
    }
    for k, v in data.items():
        v.double().numpy().tofile(B/f'{k}.bin')
    print("Generated inputs")

def L(name): return torch.from_numpy(np.fromfile(B/f'{name}.bin', dtype=np.float64))
def R(name, shape): return L(name).reshape(shape)

X    = R('bX', (T, D)).requires_grad_(True)
Wqkv = R('bWqkv', (3*D, D)).requires_grad_(True)
bqkv = L('bbqkv').requires_grad_(True)
Wo   = R('bWo', (D, D)).requires_grad_(True)
bo   = L('bbo').requires_grad_(True)
W1   = R('bW1', (DFF, D)).requires_grad_(True)
b1   = L('bb1').requires_grad_(True)
W2   = R('bW2', (D, DFF)).requires_grad_(True)
b2   = L('bb2').requires_grad_(True)
ln1w = L('bln1w').requires_grad_(True)
ln1b = L('bln1b').requires_grad_(True)
ln2w = L('bln2w').requires_grad_(True)
ln2b = L('bln2b').requires_grad_(True)
dY   = R('bdY', (T, D))

# Block forward
xn1 = torch.nn.functional.layer_norm(X, (D,), ln1w, ln1b, eps=1e-5)
qkv = xn1 @ Wqkv.T + bqkv
Q, K, V = qkv.split(D, dim=-1)
qh = Q.view(T, H, HD).transpose(0, 1)
kh = K.view(T, H, HD).transpose(0, 1)
vh = V.view(T, H, HD).transpose(0, 1)
mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
att = (qh @ kh.transpose(-2, -1)) / (HD ** 0.5)
att = att.masked_fill(mask == 0, float('-inf'))
att = torch.softmax(att, dim=-1)
y = att @ vh
O = y.transpose(0, 1).contiguous().view(T, D)
proj = O @ Wo.T + bo
x1 = X + proj
xn2 = torch.nn.functional.layer_norm(x1, (D,), ln2w, ln2b, eps=1e-5)
fc1 = xn2 @ W1.T + b1
g = torch.nn.functional.gelu(fc1, approximate='tanh')
fc2 = g @ W2.T + b2
x2 = x1 + fc2

x2.backward(dY)

refs = {
    'bdx':    X.grad.numpy(),
    'bdWqkv': Wqkv.grad.numpy(),
    'bdbqkv': bqkv.grad.numpy(),
    'bdWo':   Wo.grad.numpy(),
    'bdbo':   bo.grad.numpy(),
    'bdW1':   W1.grad.numpy(),
    'bdb1':   b1.grad.numpy(),
    'bdW2':   W2.grad.numpy(),
    'bdb2':   b2.grad.numpy(),
    'bdln1w': ln1w.grad.numpy(),
    'bdln1b': ln1b.grad.numpy(),
    'bdln2w': ln2w.grad.numpy(),
    'bdln2b': ln2b.grad.numpy(),
}

print()
print("=== Transformer block backward ===")
worst = 0.0
missing = 0
for name, ref in refs.items():
    f = B / f'{name}.bin'
    if not f.exists():
        print(f"  ❌ {name:8s} MISSING")
        missing += 1
        continue
    mmc = np.fromfile(f, dtype=np.float64).reshape(ref.shape)
    d = np.abs(mmc - ref).max()
    worst = max(worst, d)
    flag = "✅" if d < 1e-10 else "❌"
    print(f"  {flag} {name:8s}  max={d:.4e}")

if missing > 0:
    print(f"\n❌ {missing} files missing")
    sys.exit(1)

if worst < 1e-10:
    print(f"\n🏆 BLOCK BACKWARD PASS — worst={worst:.4e}")
else:
    print(f"\n❌ FAIL worst={worst:.4e}")
    sys.exit(1)
