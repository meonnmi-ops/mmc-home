"""full_train_gen.py — PyTorch reference for the FULL Phase 1 model.

Usage — same 3-command dance as train_multi_gen:
  1) python3 ref/full_train_gen.py     # init weights if missing + compute ref
  2) ./build/train_full                # MMC does fwd+bwd+AdamW, writes out_*
  3) python3 ref/full_train_gen.py     # compare out_* vs PyTorch
"""
import sys
import numpy as np
import torch
from pathlib import Path

T, TMAX, D, DFF, H, V, L = 128, 128, 256, 1024, 8, 256, 6
HD = D // H
B = Path('ref/full_test'); B.mkdir(parents=True, exist_ok=True)

TSHAPE = {'Wqkv': (L, 3*D, D), 'bqkv': (L, 3*D), 'Wo': (L, D, D), 'bo': (L, D),
          'W1': (L, DFF, D), 'b1': (L, DFF), 'W2': (L, D, DFF), 'b2': (L, D),
          'ln1w': (L, D), 'ln1b': (L, D), 'ln2w': (L, D), 'ln2b': (L, D)}
SIZES = {'Wqkv': L*3*D*D, 'bqkv': L*3*D, 'Wo': L*D*D, 'bo': L*D,
         'W1': L*DFF*D, 'b1': L*DFF, 'W2': L*D*DFF, 'b2': L*D,
         'ln1w': L*D, 'ln1b': L*D, 'ln2w': L*D, 'ln2b': L*D,
         'wte': V*D, 'wpe': TMAX*D, 'lnfw': D, 'lnfb': D}

TOTAL_PARAMS = sum(SIZES.values())
assert TOTAL_PARAMS == 4_837_376, TOTAL_PARAMS


def init_weights():
    """GPT-2 style init (std=0.02), deterministic via dedicated generator."""
    g = torch.Generator().manual_seed(42)

    def rnd(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float64) * 0.02

    for n in ['Wqkv', 'Wo', 'W1', 'W2', 'bqkv', 'bo', 'b1', 'b2']:
        per = TSHAPE[n][1:]
        torch.stack([rnd(*per) for _ in range(L)]).numpy().tofile(B / f'{n}.bin')
    torch.ones(L, D, dtype=torch.float64).numpy().tofile(B / 'ln1w.bin')
    torch.zeros(L, D, dtype=torch.float64).numpy().tofile(B / 'ln1b.bin')
    torch.ones(L, D, dtype=torch.float64).numpy().tofile(B / 'ln2w.bin')
    torch.zeros(L, D, dtype=torch.float64).numpy().tofile(B / 'ln2b.bin')
    rnd(V, D).numpy().tofile(B / 'wte.bin')
    rnd(TMAX, D).numpy().tofile(B / 'wpe.bin')
    torch.ones(D, dtype=torch.float64).numpy().tofile(B / 'lnfw.bin')
    torch.zeros(D, dtype=torch.float64).numpy().tofile(B / 'lnfb.bin')
    print("initialized fresh weights (seed 42, std=0.02)")


def Ld(n):
    return np.fromfile(B / f'{n}.bin', dtype=np.float64)


def Ld0(n):
    """load float64 file, or zeros if absent (AdamW state on fresh start)."""
    f = B / f'{n}.bin'
    if f.exists() and f.stat().st_size == SIZES[n] * 8:
        return np.fromfile(f, dtype=np.float64)
    return np.zeros(SIZES[n], dtype=np.float64)


def Rs(a, shape):
    return torch.from_numpy(a.reshape(shape).copy()).requires_grad_(True)


def scalar_bin(name, dtype, default):
    f = B / name
    if f.exists() and f.stat().st_size >= np.dtype(dtype).itemsize:
        arr = np.fromfile(f, dtype=dtype)
        if arr.size >= 1:
            return arr[0]
    return default


# ---------- 1) weights ----------
if not (B / 'Wqkv.bin').exists():
    init_weights()

W = {n: Rs(Ld(n), TSHAPE[n]) for n in TSHAPE}
wte = Rs(Ld('wte'), (V, D))
wpe = Rs(Ld('wpe'), (TMAX, D))
lnfw = Rs(Ld('lnfw'), (D,))
lnfb = Rs(Ld('lnfb'), (D,))

# ---------- 2) batch ----------
if not (B / 'ids.bin').exists() or not (B / 'tg.bin').exists():
    rng = np.random.RandomState(7)
    rng.randint(0, V, T).astype(np.int32).tofile(B / 'ids.bin')
    rng.randint(0, V, T).astype(np.int32).tofile(B / 'tg.bin')
    print("wrote demo ids.bin / tg.bin (random bytes, seed 7)")
ids_np = np.fromfile(B / 'ids.bin', dtype=np.int32)[:T]
tg_np = np.fromfile(B / 'tg.bin', dtype=np.int32)[:T]
ids = torch.from_numpy(ids_np.astype(np.int64))
tg = torch.from_numpy(tg_np.astype(np.int64))

# ---------- 3) lr / step (must match train_full.mmc) ----------
lr = float(scalar_bin('lr.bin', np.float64, 1e-3))
step = int(scalar_bin('step.bin', np.int32, 1))

# ---------- 4) forward ----------
x = wte[ids] + wpe[:T]
for i in range(L):
    xn1 = torch.nn.functional.layer_norm(x, (D,), W['ln1w'][i], W['ln1b'][i], eps=1e-5)
    qkv = xn1 @ W['Wqkv'][i].T + W['bqkv'][i]
    Q, K, V_ = qkv.split(D, dim=-1)
    qh = Q.view(T, H, HD).transpose(0, 1)
    kh = K.view(T, H, HD).transpose(0, 1)
    vh = V_.view(T, H, HD).transpose(0, 1)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    att = (qh @ kh.transpose(-2, -1)) / (HD ** 0.5)
    att = att.masked_fill(mask == 0, float('-inf'))
    att = torch.softmax(att, dim=-1)
    y = att @ vh
    O = y.transpose(0, 1).contiguous().view(T, D)
    x1 = x + (O @ W['Wo'][i].T + W['bo'][i])
    xn2 = torch.nn.functional.layer_norm(x1, (D,), W['ln2w'][i], W['ln2b'][i], eps=1e-5)
    gg = torch.nn.functional.gelu(xn2 @ W['W1'][i].T + W['b1'][i], approximate='tanh')
    x = x1 + (gg @ W['W2'][i].T + W['b2'][i])

xnf = torch.nn.functional.layer_norm(x, (D,), lnfw, lnfb, eps=1e-5)
logits = xnf @ wte.T
loss = torch.nn.functional.cross_entropy(logits, tg)
loss.backward()
print(f"PyTorch loss = {loss.item():.10f}   (step={step}, lr={lr:.2e})")

# ---------- 5) AdamW with persistent state (matches MMC exactly) ----------
b1c, b2c, epsaw, wd = 0.9, 0.999, 1e-8, 0.01
bc1 = 1.0 - b1c ** step
bc2 = 1.0 - b2c ** step

tensors = dict(W)
tensors.update({'wte': wte, 'wpe': wpe, 'lnfw': lnfw, 'lnfb': lnfb})

results = {}
for name, w in tensors.items():
    m = torch.from_numpy(Ld0(f'm_{name}').copy())
    v = torch.from_numpy(Ld0(f'v_{name}').copy())
    with torch.no_grad():
        g = w.grad
        m2 = b1c * m + (1.0 - b1c) * g
        v2 = b2c * v + (1.0 - b2c) * g * g
        mh = m2 / bc1
        vhh = v2 / bc2
        w2 = w * (1.0 - lr * wd) - lr * mh / (torch.sqrt(vhh) + epsaw)
    results[name] = (w2.flatten().numpy(), m2.flatten().numpy(), v2.flatten().numpy())

# ---------- 6) compare ----------
missing = [f'out_{pfx}{n}.bin' for n in results for pfx in ('', 'm_', 'v_')
           if not (B / f'out_{pfx}{n}.bin').exists()]
if missing:
    print(f"({len(missing)} out_* files missing — waiting for MMC)")
    sys.exit(0)

rows, worst, worst_name = [], 0.0, ''
for name in results:
    dmax = 0.0
    for pfx, idx in (('', 0), ('m_', 1), ('v_', 2)):
        mmc = np.fromfile(B / f'out_{pfx}{name}.bin', dtype=np.float64)
        ref = results[name][idx]
        d = float(np.abs(mmc - ref).max()) if mmc.size == ref.size else float('inf')
        dmax = max(dmax, d)
    rows.append((name, dmax))
    if dmax > worst:
        worst, worst_name = dmax, name

for name, d in rows:
    print(f"  {'✅' if d < 1e-8 else '❌'} {name:6s}  max={d:.3e}")

if worst < 1e-8:
    print(f"\n🏆 FULL-STEP PASS (D={D}, L={L}, T={T}) — worst={worst:.3e} ({worst_name})")
else:
    print(f"\n❌ FAIL — worst={worst:.3e} ({worst_name})")
    sys.exit(1)
