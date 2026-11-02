import numpy as np
import subprocess, sys

M, D = 128, 256
eps  = 1e-5

np.random.seed(42)
x     = (np.random.randn(M, D) * 0.5).astype(np.float32)
gamma = (np.random.randn(D) * 0.1 + 1.0).astype(np.float32)
beta  = (np.random.randn(D) * 0.1).astype(np.float32)
dy    = (np.random.randn(M, D) * 0.1).astype(np.float32)

x.tofile('/tmp/ln_x.bin')
dy.tofile('/tmp/ln_dy.bin')
gamma.tofile('/tmp/ln_gamma.bin')
beta.tofile('/tmp/ln_beta.bin')

# ---- Reference forward
mu   = x.mean(axis=1, keepdims=True)
var  = x.var(axis=1, keepdims=True)
rstd = 1.0 / np.sqrt(var + eps)
xhat = (x - mu) * rstd
y_ref = xhat * gamma + beta

# ---- Reference backward
dgamma_ref = (dy * xhat).sum(axis=0)
dbeta_ref  = dy.sum(axis=0)
dxhat      = dy * gamma
dx_ref     = rstd * (dxhat
                     - dxhat.mean(axis=1, keepdims=True)
                     - xhat * (dxhat * xhat).mean(axis=1, keepdims=True))

r = subprocess.run(['/tmp/test_ln'], capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print(r.stderr); sys.exit(1)

y      = np.fromfile('/tmp/ln_y.bin',      dtype=np.float32).reshape(M, D)
dx     = np.fromfile('/tmp/ln_dx.bin',     dtype=np.float32).reshape(M, D)
dgamma = np.fromfile('/tmp/ln_dgamma.bin', dtype=np.float32)
dbeta  = np.fromfile('/tmp/ln_dbeta.bin',  dtype=np.float32)

def chk(name, a, b):
    d = np.abs(a - b).max()
    print(f"{name:10s} max_abs_diff = {d:.3e}  {'PASS' if d < 1e-3 else 'FAIL'}")
    return d

print("=" * 50)
ok = all([
    chk('y',      y,      y_ref)      < 1e-3,
    chk('dx',     dx,     dx_ref)     < 1e-3,
    chk('dgamma', dgamma, dgamma_ref) < 1e-3,
    chk('dbeta',  dbeta,  dbeta_ref)  < 1e-3,
])
print("=" * 50)
print("ALL PASS" if ok else "SOME FAILED")
