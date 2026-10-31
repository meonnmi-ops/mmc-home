"""Parameterized L-block training step ref (reads D, H, DFF from env)."""
import torch, numpy as np, sys, os
from pathlib import Path

T    = int(os.environ.get('T',    4))
TMAX = int(os.environ.get('TMAX', 4))
V    = int(os.environ.get('V',    4))
L    = int(os.environ.get('L',    2))
D    = int(os.environ['D'])
H    = int(os.environ['H'])
DFF  = int(os.environ['DFF'])
HD   = D // H

B = Path('ref/multi_test'); B.mkdir(parents=True, exist_ok=True)
torch.manual_seed(42)

# Only regen when shape changed (keeps out_* between step 2 & step 5)
shape_marker = B / '.shape'
expected = f'{D}_{DFF}_{H}'
needs_regen = (not shape_marker.exists()) or (shape_marker.read_text().strip() != expected)
if needs_regen:
    for f in B.glob('*.bin'): f.unlink()
    for f in B.glob('*.npy'): f.unlink()

def gen(fn, per_block):
    parts = [torch.randn(*per_block) * 0.1 for _ in range(L)]
    torch.stack(parts).double().numpy().tofile(B/fn)
gen('Wqkv.bin', (3*D, D)); gen('bqkv.bin', (3*D,))
gen('Wo.bin', (D, D));     gen('bo.bin', (D,))
gen('W1.bin', (DFF, D));   gen('b1.bin', (DFF,))
gen('W2.bin', (D, DFF));   gen('b2.bin', (D,))
torch.stack([torch.ones(D) for _ in range(L)]).double().numpy().tofile(B/'ln1w.bin')
torch.stack([torch.zeros(D) for _ in range(L)]).double().numpy().tofile(B/'ln1b.bin')
torch.stack([torch.ones(D) for _ in range(L)]).double().numpy().tofile(B/'ln2w.bin')
torch.stack([torch.zeros(D) for _ in range(L)]).double().numpy().tofile(B/'ln2b.bin')
torch.randn(V, D).double().numpy().tofile(B/'wte.bin')
torch.randn(TMAX, D).double().numpy().tofile(B/'wpe.bin')
torch.ones(D).double().numpy().tofile(B/'lnfw.bin')
torch.zeros(D).double().numpy().tofile(B/'lnfb.bin')
torch.randint(0, V, (T,)).numpy().astype(np.int32).tofile(B/'ids.bin')
torch.randint(0, V, (T,)).numpy().astype(np.int32).tofile(B/'tg.bin')
shape_marker.write_text(expected)

def Ld(n): return np.fromfile(B/f'{n}.bin', dtype=np.float64)
def Rs(a, s): return torch.from_numpy(a.reshape(s).copy()).requires_grad_(True)

Wqkv = Rs(Ld('Wqkv'), (L, 3*D, D)); bqkv = Rs(Ld('bqkv'), (L, 3*D))
Wo   = Rs(Ld('Wo'),   (L, D, D));   bo   = Rs(Ld('bo'),   (L, D))
W1   = Rs(Ld('W1'),   (L, DFF, D)); b1   = Rs(Ld('b1'),   (L, DFF))
W2   = Rs(Ld('W2'),   (L, D, DFF)); b2   = Rs(Ld('b2'),   (L, D))
ln1w = Rs(Ld('ln1w'), (L, D));      ln1b = Rs(Ld('ln1b'), (L, D))
ln2w = Rs(Ld('ln2w'), (L, D));      ln2b = Rs(Ld('ln2b'), (L, D))
wte  = Rs(Ld('wte'),  (V, D));      wpe  = Rs(Ld('wpe'),  (TMAX, D))
lnfw = Rs(Ld('lnfw'), (D,));        lnfb = Rs(Ld('lnfb'), (D,))
ids = torch.from_numpy(np.fromfile(B/'ids.bin', dtype=np.int32).astype(np.int64))
tg  = torch.from_numpy(np.fromfile(B/'tg.bin', dtype=np.int32).astype(np.int64))

x = wte[ids] + wpe[:T]
for i in range(L):
    xn1 = torch.nn.functional.layer_norm(x, (D,), ln1w[i], ln1b[i], eps=1e-5)
    qkv = xn1 @ Wqkv[i].T + bqkv[i]
    Q, K, V_ = qkv.split(D, dim=-1)
    qh = Q.view(T, H, HD).transpose(0,1); kh = K.view(T, H, HD).transpose(0,1); vh = V_.view(T, H, HD).transpose(0,1)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    att = (qh @ kh.transpose(-2,-1)) / (HD ** 0.5)
    att = att.masked_fill(mask == 0, float('-inf'))
    att = torch.softmax(att, dim=-1)
    y = att @ vh
    O = y.transpose(0,1).contiguous().view(T, D)
    x1 = x + (O @ Wo[i].T + bo[i])
    xn2 = torch.nn.functional.layer_norm(x1, (D,), ln2w[i], ln2b[i], eps=1e-5)
    g = torch.nn.functional.gelu(xn2 @ W1[i].T + b1[i], approximate='tanh')
    x = x1 + (g @ W2[i].T + b2[i])

xnf = torch.nn.functional.layer_norm(x, (D,), lnfw, lnfb, eps=1e-5)
loss = torch.nn.functional.cross_entropy(xnf @ wte.T, tg)
loss.backward()

lr, b1c, b2c, eps, wd = 0.02, 0.9, 0.999, 1e-8, 0.01
def adamw(w, g):
    with torch.no_grad():
        mh = ((1-b1c) * g) / (1 - b1c)
        vh_ = ((1-b2c) * g * g) / (1 - b2c)
        return w * (1 - lr * wd) - lr * mh / (torch.sqrt(vh_) + eps)

updates = {}
for name, w_all in [('Wqkv',Wqkv),('bqkv',bqkv),('Wo',Wo),('bo',bo),('W1',W1),('b1',b1),
                     ('W2',W2),('b2',b2),('ln1w',ln1w),('ln1b',ln1b),('ln2w',ln2w),('ln2b',ln2b)]:
    updates[name] = adamw(w_all, w_all.grad).flatten().numpy()
for name, w in [('wte',wte),('wpe',wpe),('lnfw',lnfw),('lnfb',lnfb)]:
    updates[name] = adamw(w, w.grad).flatten().numpy()

loss_val = loss.item()
print(f"  PyTorch loss = {loss_val:.10f}")

# Compare with relative tolerance
all_ok = True
max_rel = 0.0
missing = 0
for name, ref in updates.items():
    f = B/f'out_{name}.bin'
    if not f.exists():
        missing += 1; continue
    mmc = np.fromfile(f, dtype=np.float64).ravel()
    ref_flat = ref.ravel()
    abs_d = np.abs(mmc - ref_flat).max()
    scale = max(np.abs(ref_flat).max(), 1e-3)
    rel_d = abs_d / scale
    max_rel = max(max_rel, rel_d)
    if rel_d > 1e-8:
        all_ok = False
        print(f"    {name}: rel={rel_d:.2e}")

if missing == len(updates):
    print("  (waiting for MMC)")
    sys.exit(0)
if missing > 0:
    print(f"  ❌ {missing} missing")
    sys.exit(1)
if all_ok:
    print(f"  🏆 PASS (D={D}) max_rel={max_rel:.2e}")
    sys.exit(0)
print(f"  ❌ FAIL max_rel={max_rel:.2e}")
sys.exit(1)
