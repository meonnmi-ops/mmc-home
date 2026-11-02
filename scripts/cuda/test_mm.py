import numpy as np
import subprocess
import sys

M, K, N = 128, 256, 1024
np.random.seed(42)
A = (np.random.randn(M, K) * 0.1).astype(np.float32)
B = (np.random.randn(K, N) * 0.1).astype(np.float32)

A.tofile('/tmp/A.bin')
B.tofile('/tmp/B.bin')
print(f"Generated A({M}x{K}) B({K}x{N})")

r = subprocess.run(['/tmp/test_mm'], capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print("STDERR:", r.stderr)
    sys.exit(1)

C_gpu = np.fromfile('/tmp/C.bin', dtype=np.float32).reshape(M, N)
C_ref = A @ B

abs_diff = np.abs(C_ref - C_gpu)
max_abs = abs_diff.max()
max_rel = (abs_diff / (np.abs(C_ref) + 1e-8)).max()
print(f"Shape: {C_ref.shape}")
print(f"Max abs diff: {max_abs:.3e}")
print(f"Max rel diff: {max_rel:.3e}")
print("PASS" if max_abs < 1e-3 else "FAIL")
