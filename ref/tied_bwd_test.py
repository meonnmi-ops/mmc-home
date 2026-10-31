"""Tied head + embedding + LN_f bwd: PyTorch vs MMC."""
import torch, numpy as np, sys
from pathlib import Path

T, D, V = 3, 6, 4
B = Path('ref/tied_test'); B.mkdir(parents=True, exist_ok=True)
torch.manual_seed(7)

if not (B/'ids.bin').exists():
    ids     = torch.randint(0, V, (T,))
    wte     = torch.randn(V, D)
    wpe     = torch.randn(T, D)
    lnfw    = torch.randn(D)
    lnfb    = torch.randn(D)
    targets = torch.randint(0, V, (T,))
    ids.numpy().astype(np.int32).tofile(B/'ids.bin')
    wte.double().numpy().tofile(B/'wte.bin')
    wpe.double().numpy().tofile(B/'wpe.bin')
    lnfw.double().numpy().tofile(B/'lnfw.bin')
    lnfb.double().numpy().tofile(B/'lnfb.bin')
    targets.numpy().astype(np.int32).tofile(B/'targets.bin')
    print("Generated inputs")

ids     = torch.from_numpy(np.fromfile(B/'ids.bin', dtype=np.int32).astype(np.int64))
wte     = torch.from_numpy(np.fromfile(B/'wte.bin', dtype=np.float64).reshape(V, D)).requires_grad_(True)
wpe     = torch.from_numpy(np.fromfile(B/'wpe.bin', dtype=np.float64).reshape(T, D)).requires_grad_(True)
lnfw    = torch.from_numpy(np.fromfile(B/'lnfw.bin', dtype=np.float64)).requires_grad_(True)
lnfb    = torch.from_numpy(np.fromfile(B/'lnfb.bin', dtype=np.float64)).requires_grad_(True)
targets = torch.from_numpy(np.fromfile(B/'targets.bin', dtype=np.int32).astype(np.int64))

x = wte[ids] + wpe
xn = torch.nn.functional.layer_norm(x, (D,), lnfw, lnfb, eps=1e-5)
logits = xn @ wte.T
loss = torch.nn.functional.cross_entropy(logits, targets)
loss.backward()

refs = {
    'd_wte':  wte.grad.numpy(),
    'd_wpe':  wpe.grad.numpy(),
    'd_lnfw': lnfw.grad.numpy(),
    'd_lnfb': lnfb.grad.numpy(),
}
print(f"\nref loss = {loss.item():.10f}")
print()
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
    print(f"\n❌ {missing} missing")
    sys.exit(1)
if worst < 1e-10:
    print(f"\n🏆 TIED HEAD BACKWARD PASS — worst={worst:.4e}")
else:
    print(f"\n❌ FAIL worst={worst:.4e}")
    sys.exit(1)
