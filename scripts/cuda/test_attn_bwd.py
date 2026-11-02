import numpy as np, subprocess, sys

T, D, H = 128, 256, 8
head_dim = D // H
np.random.seed(42)
Q  = (np.random.randn(T, D) * 0.3).astype(np.float32)
K  = (np.random.randn(T, D) * 0.3).astype(np.float32)
V  = (np.random.randn(T, D) * 0.3).astype(np.float32)
dO = (np.random.randn(T, D) * 0.1).astype(np.float32)
Q.tofile('/tmp/attn_q.bin'); K.tofile('/tmp/attn_k.bin')
V.tofile('/tmp/attn_v.bin'); dO.tofile('/tmp/attn_do.bin')

scale = 1.0 / np.sqrt(head_dim)
Qh = Q.reshape(T,H,head_dim).transpose(1,0,2)
Kh = K.reshape(T,H,head_dim).transpose(1,0,2)
Vh = V.reshape(T,H,head_dim).transpose(1,0,2)
dOh= dO.reshape(T,H,head_dim).transpose(1,0,2)

scores = np.einsum('htd,hcd->htc', Qh, Kh) * scale
smax = scores.max(axis=-1, keepdims=True)
exps = np.exp(scores - smax)
S = exps / exps.sum(axis=-1, keepdims=True)

dV_h = np.einsum('htc,htd->hcd', S, dOh)
dS_pre = np.einsum('htd,hcd->htc', dOh, Vh)
row_dot = np.einsum('htc,htc->ht', S, dS_pre)[..., None]
dS = S * (dS_pre - row_dot) * scale
dQ_h = np.einsum('htc,hcd->htd', dS, Kh)
dK_h = np.einsum('htc,htd->hcd', dS, Qh)
dQ_ref = dQ_h.transpose(1,0,2).reshape(T,D)
dK_ref = dK_h.transpose(1,0,2).reshape(T,D)
dV_ref = dV_h.transpose(1,0,2).reshape(T,D)

r = subprocess.run(['/tmp/test_attn_bwd'], capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print("STDERR:", r.stderr); sys.exit(1)

dQ = np.fromfile('/tmp/attn_dq.bin', dtype=np.float32).reshape(T,D)
dK = np.fromfile('/tmp/attn_dk.bin', dtype=np.float32).reshape(T,D)
dV = np.fromfile('/tmp/attn_dv.bin', dtype=np.float32).reshape(T,D)

ok = True
for name, a, b in [('dQ', dQ, dQ_ref), ('dK', dK, dK_ref), ('dV', dV, dV_ref)]:
    d = np.abs(a-b).max()
    s = 'PASS' if d < 1e-3 else 'FAIL'
    if d >= 1e-3: ok = False
    print(f"{name} max_abs = {d:.3e}  {s}")
print("ALL PASS" if ok else "SOME FAILED")
