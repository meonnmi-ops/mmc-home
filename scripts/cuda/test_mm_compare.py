import numpy as np
import subprocess

M, K, N = 128, 256, 1024
np.random.seed(42)
A = (np.random.randn(M, K) * 0.1).astype(np.float32)
B = (np.random.randn(K, N) * 0.1).astype(np.float32)
A.tofile('/tmp/A.bin'); B.tofile('/tmp/B.bin')

print("=" * 60)
print(f"Shape: {M}x{K}x{N}")
print(f"FLOPs: {2*M*K*N/1e6:.2f} MFLOP")
print("=" * 60)

# Reference
C_ref = A @ B

# Naive
r1 = subprocess.run(['/tmp/test_mm'], capture_output=True, text=True)
print("\n[Naive kernel]")
print(r1.stdout.strip())
C1 = np.fromfile('/tmp/C.bin', dtype=np.float32).reshape(M, N)
diff1 = np.abs(C_ref - C1).max()

# Tiled
r2 = subprocess.run(['/tmp/test_mm_tiled'], capture_output=True, text=True)
print("\n[Tiled kernel]")
print(r2.stdout.strip())
C2 = np.fromfile('/tmp/C.bin', dtype=np.float32).reshape(M, N)
diff2 = np.abs(C_ref - C2).max()

print("\n" + "=" * 60)
print(f"Naive abs diff: {diff1:.3e}")
print(f"Tiled abs diff: {diff2:.3e}")
print("PASS" if diff1 < 1e-3 and diff2 < 1e-3 else "FAIL")
print("=" * 60)
