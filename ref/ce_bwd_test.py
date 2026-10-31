"""CE backward: PyTorch ref vs MMC."""
import torch, numpy as np
from pathlib import Path

T, V = 16, 32
B = Path('ref/bwd_test'); B.mkdir(parents=True, exist_ok=True)
torch.manual_seed(3)

if not (B/'ceL.bin').exists():
    logits = torch.randn(T, V, dtype=torch.float64) * 2.0  # wider scale
    targets = torch.randint(0, V, (T,))
    logits.numpy().tofile(B/'ceL.bin')
    targets.numpy().astype(np.int32).tofile(B/'ceT.bin')
    print("Generated inputs")

logits = torch.from_numpy(
    np.fromfile(B/'ceL.bin', dtype=np.float64).reshape(T, V)
).requires_grad_(True)
targets = torch.from_numpy(np.fromfile(B/'ceT.bin', dtype=np.int32).astype(np.int64))

loss = torch.nn.functional.cross_entropy(logits, targets)
loss.backward()

loss_ref = loss.item()
dL_ref = logits.grad.numpy()

if (B/'cedL.bin').exists():
    loss_mmc = np.fromfile(B/'ceLOSS.bin', dtype=np.float64)[0]
    dL_mmc = np.fromfile(B/'cedL.bin', dtype=np.float64).reshape(T, V)

    print()
    print("=== CE forward/backward comparison ===")
    print(f"  loss:  ref={loss_ref:.10f}  mmc={loss_mmc:.10f}")
    print(f"  dlogits: max={np.abs(dL_mmc-dL_ref).max():.4e}")

    tol = 1e-10
    worst = max(abs(loss_mmc - loss_ref), np.abs(dL_mmc-dL_ref).max())
    if worst < tol:
        print(f"\n🏆 PASS — CE backward < {tol}")
    else:
        print(f"\n❌ FAIL (worst={worst:.4e})")
