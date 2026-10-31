"""matmul_t backward: PyTorch ref vs MMC."""
import torch, numpy as np
from pathlib import Path

T, J, K = 8, 16, 32
B = Path('ref/bwd_test'); B.mkdir(parents=True, exist_ok=True)
torch.manual_seed(2)

if not (B/'mX.bin').exists():
    X  = torch.randn(T, K, dtype=torch.float64)
    W  = torch.randn(J, K, dtype=torch.float64)
    b  = torch.randn(J,    dtype=torch.float64)
    dY = torch.randn(T, J, dtype=torch.float64)
    X.numpy().tofile(B/'mX.bin')
    W.numpy().tofile(B/'mW.bin')
    b.numpy().tofile(B/'mb.bin')
    dY.numpy().tofile(B/'mdY.bin')
    print("Generated inputs")

X  = torch.from_numpy(np.fromfile(B/'mX.bin', dtype=np.float64).reshape(T, K)).requires_grad_(True)
W  = torch.from_numpy(np.fromfile(B/'mW.bin', dtype=np.float64).reshape(J, K)).requires_grad_(True)
b  = torch.from_numpy(np.fromfile(B/'mb.bin', dtype=np.float64)).requires_grad_(True)
dY = torch.from_numpy(np.fromfile(B/'mdY.bin', dtype=np.float64).reshape(T, J))

Y = X @ W.T + b
Y.backward(dY)

dX_ref = X.grad.numpy()
dW_ref = W.grad.numpy()
db_ref = b.grad.numpy()

if (B/'mdX.bin').exists():
    dX_mmc = np.fromfile(B/'mdX.bin', dtype=np.float64).reshape(T, K)
    dW_mmc = np.fromfile(B/'mdW.bin', dtype=np.float64).reshape(J, K)
    db_mmc = np.fromfile(B/'mdb.bin', dtype=np.float64)

    print()
    print("=== matmul_t backward comparison ===")
    print(f"  dX:  max={np.abs(dX_mmc-dX_ref).max():.4e}")
    print(f"  dW:  max={np.abs(dW_mmc-dW_ref).max():.4e}")
    print(f"  db:  max={np.abs(db_mmc-db_ref).max():.4e}")

    worst = max(np.abs(dX_mmc-dX_ref).max(),
                np.abs(dW_mmc-dW_ref).max(),
                np.abs(db_mmc-db_ref).max())
    tol = 1e-10
    if worst < tol:
        print(f"\n🏆 PASS — matmul_t backward < {tol}")
    else:
        print(f"\n❌ FAIL (worst={worst:.4e})")
