import numpy as np, subprocess, sys

T, D, H = 128, 256, 8
head_dim = D // H
np.random.seed(42)

Q = (np.random.randn(T, D) * 0.3).astype(np.float32)
K = (np.random.randn(T, D) * 0.3).astype(np.float32)
V = (np.random.randn(T, D) * 0.3).astype(np.float32)
Q.tofile('/tmp/attn_q.bin')
K.tofile('/tmp/attn_k.bin')
V.tofile('/tmp/attn_v.bin')

# Reference: multi-head attention
# Split D -> H heads of size head_dim
scale = 1.0 / np.sqrt(head_dim)
Qh = Q.reshape(T, H, head_dim).transpose(1, 0, 2)  # [H, T, head_dim]
Kh = K.reshape(T, H, head_dim).transpose(1, 0, 2)
Vh = V.reshape(T, H, head_dim).transpose(1, 0, 2)

scores = np.einsum('htd,hcd->htc', Qh, Kh) * scale  # [H, T, T]
# softmax over last axis
scores_max = scores.max(axis=-1, keepdims=True)
exp_s = np.exp(scores - scores_max)
softmax = exp_s / exp_s.sum(axis=-1, keepdims=True)
out_h = np.einsum('htc,hcd->htd', softmax, Vh)  # [H, T, head_dim]
out_ref = out_h.transpose(1, 0, 2).reshape(T, D)

r = subprocess.run(['/tmp/test_attn'], capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print("STDERR:", r.stderr); sys.exit(1)

O = np.fromfile('/tmp/attn_out.bin', dtype=np.float32).reshape(T, D)
d = np.abs(O - out_ref).max()
print(f"Max abs diff: {d:.3e}  {'PASS' if d < 1e-3 else 'FAIL'}")
