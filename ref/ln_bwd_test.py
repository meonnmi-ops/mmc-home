"""Generate random inputs, compute PyTorch ref grads, compare with MMC."""
import torch, numpy as np
from pathlib import Path

D = 16; T = 8
B = Path('ref/bwd_test'); B.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)

# --- Step 1: generate inputs (idempotent — only if not exist) ---
if not (B/'x.bin').exists():
    x = torch.randn(T, D, dtype=torch.float64)
    gamma = torch.randn(D, dtype=torch.float64)
    beta = torch.randn(D, dtype=torch.float64)
    dout = torch.randn(T, D, dtype=torch.float64)

    x.numpy().tofile(B/'x.bin')
    gamma.numpy().tofile(B/'gamma.bin')
    beta.numpy().tofile(B/'beta.bin')
    dout.numpy().tofile(B/'dout.bin')
    print("Generated inputs")

# --- Step 2: compute PyTorch reference ---
x = torch.from_numpy(np.fromfile(B/'x.bin', dtype=np.float64).reshape(T, D)).requires_grad_(True)
gamma = torch.from_numpy(np.fromfile(B/'gamma.bin', dtype=np.float64)).requires_grad_(True)
beta  = torch.from_numpy(np.fromfile(B/'beta.bin', dtype=np.float64)).requires_grad_(True)
dout  = torch.from_numpy(np.fromfile(B/'dout.bin', dtype=np.float64).reshape(T, D))

mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, unbiased=False, keepdim=True)
inv = 1.0 / torch.sqrt(var + 1e-5)
xn = (x - mean) * inv * gamma + beta
xn.backward(dout)

np.save(B/'dx_ref.npy', x.grad.numpy())
np.save(B/'dgamma_ref.npy', gamma.grad.numpy())
np.save(B/'dbeta_ref.npy', beta.grad.numpy())

# --- Step 3: compare if MMC output exists ---
if (B/'dx.bin').exists():
    dx_mmc = np.fromfile(B/'dx.bin', dtype=np.float64).reshape(T, D)
    dg_mmc = np.fromfile(B/'dgamma.bin', dtype=np.float64)
    db_mmc = np.fromfile(B/'dbeta.bin', dtype=np.float64)
    dx_ref = x.grad.numpy()
    dg_ref = gamma.grad.numpy()
    db_ref = beta.grad.numpy()

    print()
    print("=== LN backward comparison ===")
    print(f"  dx:      max={np.abs(dx_mmc - dx_ref).max():.4e}")
    print(f"  dgamma:  max={np.abs(dg_mmc - dg_ref).max():.4e}")
    print(f"  dbeta:   max={np.abs(db_mmc - db_ref).max():.4e}")

    tol = 1e-10
    if max(np.abs(dx_mmc-dx_ref).max(), np.abs(dg_mmc-dg_ref).max(), np.abs(db_mmc-db_ref).max()) < tol:
        print(f"\n🏆 PASS — LN backward matches PyTorch < {tol}")
    else:
        print(f"\n❌ FAIL")
