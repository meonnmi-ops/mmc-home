"""GELU(tanh-approx) backward: PyTorch ref vs MMC."""
import torch, numpy as np
from pathlib import Path

N = 128
B = Path('ref/bwd_test'); B.mkdir(parents=True, exist_ok=True)
torch.manual_seed(1)

if not (B/'gx.bin').exists():
    x = torch.randn(N, dtype=torch.float64)
    dout = torch.randn(N, dtype=torch.float64)
    x.numpy().tofile(B/'gx.bin')
    dout.numpy().tofile(B/'gdout.bin')
    print("Generated inputs")

x = torch.from_numpy(np.fromfile(B/'gx.bin', dtype=np.float64)).requires_grad_(True)
dout = torch.from_numpy(np.fromfile(B/'gdout.bin', dtype=np.float64))

y = torch.nn.functional.gelu(x, approximate='tanh')
y.backward(dout)

dx_ref = x.grad.numpy()

if (B/'gdx.bin').exists():
    dx_mmc = np.fromfile(B/'gdx.bin', dtype=np.float64)
    print()
    print("=== GELU backward comparison ===")
    print(f"  dx:  max={np.abs(dx_mmc - dx_ref).max():.4e}")
    tol = 1e-10
    if np.abs(dx_mmc - dx_ref).max() < tol:
        print(f"\n🏆 PASS — GELU backward matches PyTorch < {tol}")
    else:
        print("\n❌ FAIL")
