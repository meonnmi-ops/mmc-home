import numpy as np, subprocess, sys

T, D, H = 4, 32, 2
hd = D // H
np.random.seed(42)

X  = (np.random.randn(T, D) * 0.3).astype(np.float32)
Wq = (np.random.randn(D, D) * 0.3).astype(np.float32)
Wk = (np.random.randn(D, D) * 0.3).astype(np.float32)
Wv = (np.random.randn(D, D) * 0.3).astype(np.float32)
Wo = (np.random.randn(D, D) * 0.3).astype(np.float32)
dY = (np.random.randn(T, D) * 0.1).astype(np.float32)

X.tofile('/tmp/bn_x.bin'); Wq.tofile('/tmp/bn_wq.bin')
Wk.tofile('/tmp/bn_wk.bin'); Wv.tofile('/tmp/bn_wv.bin')
Wo.tofile('/tmp/bn_wo.bin'); dY.tofile('/tmp/bn_dy.bin')

# Forward
Q = X @ Wq; K = X @ Wk; V = X @ Wv
sc = 1.0/np.sqrt(hd)
Qh = Q.reshape(T,H,hd).transpose(1,0,2)
Kh = K.reshape(T,H,hd).transpose(1,0,2)
Vh = V.reshape(T,H,hd).transpose(1,0,2)
scores = np.einsum('htd,hcd->htc', Qh, Kh) * sc
smax = scores.max(axis=-1, keepdims=True)
exps = np.exp(scores - smax)
S = exps / exps.sum(axis=-1, keepdims=True)
Ah = np.einsum('htc,hcd->htd', S, Vh)
A = Ah.transpose(1,0,2).reshape(T,D)
O = A @ Wo
Y = X + O

# Backward
dWo = A.T @ dY
dA = dY @ Wo.T
dAh = dA.reshape(T,H,hd).transpose(1,0,2)
dVh = np.einsum('htc,htd->hcd', S, dAh)
dS_pre = np.einsum('htd,hcd->htc', dAh, Vh)
row_dot = np.einsum('htc,htc->ht', S, dS_pre)[..., None]
dS = S * (dS_pre - row_dot) * sc
dQh = np.einsum('htc,hcd->htd', dS, Kh)
dKh = np.einsum('htc,htd->hcd', dS, Qh)
dQ = dQh.transpose(1,0,2).reshape(T,D)
dK = dKh.transpose(1,0,2).reshape(T,D)
dV = dVh.transpose(1,0,2).reshape(T,D)
dWq = X.T @ dQ; dWk = X.T @ dK; dWv = X.T @ dV
dX = dY + dQ @ Wq.T + dK @ Wk.T + dV @ Wv.T

r = subprocess.run(['/tmp/test_block_nl'], capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print("STDERR:", r.stderr); sys.exit(1)

def chk(name, ref, shape):
    got = np.fromfile(f'/tmp/bn_{name}.bin', dtype=np.float32).reshape(shape)
    d = np.abs(got - ref).max()
    print(f"  {name:8s} max_abs = {d:.3e}  {'PASS' if d<1e-3 else 'FAIL'}")
    return d < 1e-3

print("=" * 50)
ok = all([chk('y_out', Y, (T,D)), chk('dx', dX, (T,D)),
          chk('dwq', dWq, (D,D)), chk('dwk', dWk, (D,D)),
          chk('dwv', dWv, (D,D)), chk('dwo', dWo, (D,D))])
print("=" * 50)
print("ALL PASS" if ok else "SOME FAILED")
