"""L=2 block training: PyTorch ref vs MMC."""
import torch, numpy as np, sys
from pathlib import Path

T, D, DFF, H, V, TMAX, L = 4, 8, 16, 2, 4, 4, 2
B = Path('ref/multi_test'); B.mkdir(parents=True, exist_ok=True)
torch.manual_seed(42)

if not (B/'Wqkv.bin').exists():
    # Per-block weights (flat layout: L × per_block)
    def gen(fn, per_block_count):
        w = [torch.randn(per_block_count) * 0.1 for _ in range(L)]
        torch.cat(w).double().numpy().tofile(B/fn)
    gen('Wqkv.bin', 3*D*D)
    gen('bqkv.bin', 3*D)
    gen('Wo.bin', D*D)
    gen('bo.bin', D)
    gen('W1.bin', DFF*D)
    gen('b1.bin', DFF)
    gen('W2.bin', D*DFF)
    gen('b2.bin', D)
    torch.cat([torch.ones(D) for _ in range(L)]).double().numpy().tofile(B/'ln1w.bin')
    torch.cat([torch.zeros(D) for _ in range(L)]).double().numpy().tofile(B/'ln1b.bin')
    torch.cat([torch.ones(D) for _ in range(L)]).double().numpy().tofile(B/'ln2w.bin')
    torch.cat([torch.zeros(D) for _ in range(L)]).double().numpy().tofile(B/'ln2b.bin')
    torch.randn(V, D).double().numpy().tofile(B/'wte.bin')
    torch.randn(TMAX, D).double().numpy().tofile(B/'wpe.bin')
    torch.ones(D).double().numpy().tofile(B/'lnfw.bin')
    torch.zeros(D).double().numpy().tofile(B/'lnfb.bin')
    torch.randint(0, V, (T,)).numpy().astype(np.int32).tofile(B/'ids.bin')
    torch.randint(0, V, (T,)).numpy().astype(np.int32).tofile(B/'tg.bin')
    print("Generated")

def Ld(n): return np.fromfile(B/f'{n}.bin', dtype=np.float64)
def Rs(a, shape): return torch.from_numpy(a.reshape(shape)).requires_grad_(True)

# Load and split L blocks
Wqkv_all = Rs(Ld('Wqkv'), (L, 3*D, D))
bqkv_all = Rs(Ld('bqkv'), (L, 3*D))
Wo_all   = Rs(Ld('Wo'),   (L, D, D))
bo_all   = Rs(Ld('bo'),   (L, D))
W1_all   = Rs(Ld('W1'),   (L, DFF, D))
b1_all   = Rs(Ld('b1'),   (L, DFF))
W2_all   = Rs(Ld('W2'),   (L, D, DFF))
b2_all   = Rs(Ld('b2'),   (L, D))
ln1w_all = Rs(Ld('ln1w'), (L, D))
ln1b_all = Rs(Ld('ln1b'), (L, D))
ln2w_all = Rs(Ld('ln2w'), (L, D))
ln2b_all = Rs(Ld('ln2b'), (L, D))
wte  = Rs(Ld('wte'),  (V, D))
wpe  = Rs(Ld('wpe'),  (TMAX, D))
lnfw = Rs(Ld('lnfw'), (D,))
lnfb = Rs(Ld('lnfb'), (D,))
ids = torch.from_numpy(np.fromfile(B/'ids.bin', dtype=np.int32).astype(np.int64))
tg  = torch.from_numpy(np.fromfile(B/'tg.bin', dtype=np.int32).astype(np.int64))

# Forward (L blocks)
x = wte[ids] + wpe[:T]
for i in range(L):
    xn1 = torch.nn.functional.layer_norm(x, (D,), ln1w_all[i], ln1b_all[i], eps=1e-5)
    qkv = xn1 @ Wqkv_all[i].T + bqkv_all[i]
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
    proj = O @ Wo_all[i].T + bo_all[i]
    x1 = x + proj
    xn2 = torch.nn.functional.layer_norm(x1, (D,), ln2w_all[i], ln2b_all[i], eps=1e-5)
    fc1 = xn2 @ W1_all[i].T + b1_all[i]
    g = torch.nn.functional.gelu(fc1, approximate='tanh')
    fc2 = g @ W2_all[i].T + b2_all[i]
    x = x1 + fc2

xnf = torch.nn.functional.layer_norm(x, (D,), lnfw, lnfb, eps=1e-5)
logits = xnf @ wte.T
loss = torch.nn.functional.cross_entropy(logits, tg)
loss.backward()
print(f"PyTorch loss = {loss.item():.10f}")

# AdamW
lr, b1c, b2c, eps, wd = 0.02, 0.9, 0.999, 1e-8, 0.01
def adamw(w, g):
    with torch.no_grad():
        m = (1-b1c) * g
        v = (1-b2c) * g * g
        mh = m / (1 - b1c)
        vh_ = v / (1 - b2c)
        return w * (1 - lr * wd) - lr * mh / (torch.sqrt(vh_) + eps)

updates = {}
for name, w_all in [('Wqkv', Wqkv_all), ('bqkv', bqkv_all), ('Wo', Wo_all), ('bo', bo_all),
                     ('W1', W1_all), ('b1', b1_all), ('W2', W2_all), ('b2', b2_all),
                     ('ln1w', ln1w_all), ('ln1b', ln1b_all), ('ln2w', ln2w_all), ('ln2b', ln2b_all)]:
    updates[name] = adamw(w_all, w_all.grad).flatten().numpy()
updates['wte'] = adamw(wte, wte.grad).numpy()
updates['wpe'] = adamw(wpe, wpe.grad).numpy()
updates['lnfw'] = adamw(lnfw, lnfw.grad).numpy()
updates['lnfb'] = adamw(lnfb, lnfb.grad).numpy()

# Compare
print()
print("=== After 1 AdamW step (L=2) ===")
worst = 0.0
missing = 0
for name, ref in updates.items():
    f = B/f'out_{name}.bin'
    if not f.exists():
        print(f"  ❌ out_{name}.bin MISSING")
        missing += 1
        continue
    mmc = np.fromfile(f, dtype=np.float64)
    d = np.abs(mmc.ravel() - ref.ravel()).max()
    worst = max(worst, d)
    flag = "✅" if d < 1e-10 else "❌"
    print(f"  {flag} {name:6s}  max={d:.4e}")

if missing > 0:
    print(f"\n❌ {missing} missing")
    sys.exit(1)
if worst < 1e-10:
    print(f"\n🏆 MULTI-BLOCK TRAIN PASS — worst={worst:.4e}")
else:
    print(f"\n❌ FAIL worst={worst:.4e}")
    sys.exit(1)
