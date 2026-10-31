"""Generate weights, compute PyTorch AdamW step, compare with MMC."""
import torch, numpy as np, sys
from pathlib import Path

T, D, DFF, H, V, TMAX = 4, 8, 16, 2, 4, 4
B = Path('ref/train_test'); B.mkdir(parents=True, exist_ok=True)
torch.manual_seed(42)

if not (B/'Wqkv.bin').exists():
    torch.save({}, B/'_init.pt')
    data = {
        'wte':  torch.randn(V, D) * 0.1,
        'wpe':  torch.randn(TMAX, D) * 0.1,
        'Wqkv': torch.randn(3*D, D) * 0.1,
        'bqkv': torch.randn(3*D) * 0.01,
        'Wo':   torch.randn(D, D) * 0.1,
        'bo':   torch.randn(D) * 0.01,
        'W1':   torch.randn(DFF, D) * 0.1,
        'b1':   torch.randn(DFF) * 0.01,
        'W2':   torch.randn(D, DFF) * 0.1,
        'b2':   torch.randn(D) * 0.01,
        'ln1w': torch.ones(D),
        'ln1b': torch.zeros(D),
        'ln2w': torch.ones(D),
        'ln2b': torch.zeros(D),
        'lnfw': torch.ones(D),
        'lnfb': torch.zeros(D),
    }
    for k, v in data.items():
        v.double().numpy().tofile(B/f'{k}.bin')
    ids = torch.randint(0, V, (T,))
    tg = torch.randint(0, V, (T,))
    ids.numpy().astype(np.int32).tofile(B/'ids.bin')
    tg.numpy().astype(np.int32).tofile(B/'tg.bin')
    print("Generated inputs")

def L(n): return torch.from_numpy(np.fromfile(B/f'{n}.bin', dtype=np.float64))
def R(n, s): return L(n).reshape(s)

wte  = R('wte', (V, D)).requires_grad_(True)
wpe  = R('wpe', (TMAX, D)).requires_grad_(True)
Wqkv = R('Wqkv', (3*D, D)).requires_grad_(True)
bqkv = L('bqkv').requires_grad_(True)
Wo   = R('Wo', (D, D)).requires_grad_(True)
bo   = L('bo').requires_grad_(True)
W1   = R('W1', (DFF, D)).requires_grad_(True)
b1   = L('b1').requires_grad_(True)
W2   = R('W2', (D, DFF)).requires_grad_(True)
b2   = L('b2').requires_grad_(True)
ln1w = L('ln1w').requires_grad_(True)
ln1b = L('ln1b').requires_grad_(True)
ln2w = L('ln2w').requires_grad_(True)
ln2b = L('ln2b').requires_grad_(True)
lnfw = L('lnfw').requires_grad_(True)
lnfb = L('lnfb').requires_grad_(True)
ids  = torch.from_numpy(np.fromfile(B/'ids.bin', dtype=np.int32).astype(np.int64))
tg   = torch.from_numpy(np.fromfile(B/'tg.bin', dtype=np.int32).astype(np.int64))

# Forward
emb = wte[ids] + wpe[:T]
xn1 = torch.nn.functional.layer_norm(emb, (D,), ln1w, ln1b, eps=1e-5)
qkv = xn1 @ Wqkv.T + bqkv
Q, K, V_ = qkv.split(D, dim=-1)
HD = D // H
qh = Q.view(T, H, HD).transpose(0,1)
kh = K.view(T, H, HD).transpose(0,1)
vh = V_.view(T, H, HD).transpose(0,1)
mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
att = (qh @ kh.transpose(-2,-1)) / (HD ** 0.5)
att = att.masked_fill(mask == 0, float('-inf'))
att = torch.softmax(att, dim=-1)
y = att @ vh
O = y.transpose(0,1).contiguous().view(T, D)
proj = O @ Wo.T + bo
x1 = emb + proj
xn2 = torch.nn.functional.layer_norm(x1, (D,), ln2w, ln2b, eps=1e-5)
fc1 = xn2 @ W1.T + b1
g = torch.nn.functional.gelu(fc1, approximate='tanh')
fc2 = g @ W2.T + b2
x2 = x1 + fc2
xnf = torch.nn.functional.layer_norm(x2, (D,), lnfw, lnfb, eps=1e-5)
logits = xnf @ wte.T
loss = torch.nn.functional.cross_entropy(logits, tg)
loss.backward()

loss_val = loss.item()
print(f"PyTorch loss = {loss_val:.10f}")

# AdamW step=1
lr, b1c, b2c, eps, wd = 0.001, 0.9, 0.999, 1e-8, 0.01
def adamw_torch(w, g_):
    with torch.no_grad():
        m = b1c * 0 + (1-b1c) * g_
        v = b2c * 0 + (1-b2c) * g_ * g_
        mh = m / (1 - b1c)
        vh_ = v / (1 - b2c)
        return w * (1 - lr * wd) - lr * mh / (torch.sqrt(vh_) + eps)

updates = {
    'wte':  adamw_torch(wte, wte.grad),
    'wpe':  adamw_torch(wpe, wpe.grad),
    'Wqkv': adamw_torch(Wqkv, Wqkv.grad),
    'bqkv': adamw_torch(bqkv, bqkv.grad),
    'Wo':   adamw_torch(Wo, Wo.grad),
    'bo':   adamw_torch(bo, bo.grad),
    'W1':   adamw_torch(W1, W1.grad),
    'b1':   adamw_torch(b1, b1.grad),
    'W2':   adamw_torch(W2, W2.grad),
    'b2':   adamw_torch(b2, b2.grad),
    'ln1w': adamw_torch(ln1w, ln1w.grad),
    'ln1b': adamw_torch(ln1b, ln1b.grad),
    'ln2w': adamw_torch(ln2w, ln2w.grad),
    'ln2b': adamw_torch(ln2b, ln2b.grad),
    'lnfw': adamw_torch(lnfw, lnfw.grad),
    'lnfb': adamw_torch(lnfb, lnfb.grad),
}
for k, v in updates.items():
    torch.from_numpy(v.numpy()).to(B/f'ref_{k}.bin') if False else None
    # Save via numpy
    v.detach().numpy().tofile(B/f'ref_{k}.bin')

# Compare with MMC outputs
print()
print("=== After 1 AdamW step — MMC vs PyTorch ===")
worst = 0.0
missing = 0
for k in updates.keys():
    f = B/f'out_{k}.bin'
    if not f.exists():
        print(f"  ❌ out_{k}.bin MISSING")
        missing += 1
        continue
    mmc = np.fromfile(f, dtype=np.float64).reshape(updates[k].shape)
    ref = updates[k].numpy()
    d = np.abs(mmc - ref).max()
    worst = max(worst, d)
    flag = "✅" if d < 1e-10 else "❌"
    print(f"  {flag} {k:6s}  max={d:.4e}")

if missing > 0:
    print(f"\n❌ {missing} missing")
    sys.exit(1)
if worst < 1e-10:
    print(f"\n🏆 TRAIN STEP PASS — worst={worst:.4e}")
else:
    print(f"\n❌ FAIL worst={worst:.4e}")
    sys.exit(1)
