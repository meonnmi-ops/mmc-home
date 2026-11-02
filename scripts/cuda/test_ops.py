import numpy as np, subprocess, sys

np.random.seed(42)

# --- GELU data
n = 128*256
x_g  = (np.random.randn(n) * 0.5).astype(np.float32)
dy_g = (np.random.randn(n) * 0.1).astype(np.float32)
x_g.tofile('/tmp/gelu_x.bin'); dy_g.tofile('/tmp/gelu_dy.bin')

c = 0.7978845608
inner = c * (x_g + 0.044715 * x_g**3)
t = np.tanh(inner)
y_g_ref  = 0.5 * x_g * (1 + t)
dt = c * (1 + 3 * 0.044715 * x_g**2)
dx_g_ref = dy_g * (0.5*(1+t) + 0.5*x_g*(1-t*t)*dt)

# --- Softmax+CE data
T, V = 128, 256
logits = (np.random.randn(T, V) * 2.0).astype(np.float32)
tg = np.random.randint(0, V, size=T).astype(np.int32)
logits.tofile('/tmp/sm_logits.bin'); tg.tofile('/tmp/sm_tg.bin')

mx = logits.max(axis=1, keepdims=True)
e = np.exp(logits - mx)
probs_ref = e / e.sum(axis=1, keepdims=True)
loss_ref = -(logits[np.arange(T), tg] - mx.flatten() - np.log(e.sum(axis=1)))
onehot = np.zeros_like(logits); onehot[np.arange(T), tg] = 1.0
dlogits_ref = (probs_ref - onehot) / T

# --- AdamW data
an = 1024
w0 = (np.random.randn(an) * 0.1).astype(np.float32)
g0 = (np.random.randn(an) * 0.01).astype(np.float32)
m0 = (np.random.randn(an) * 0.001).astype(np.float32)
v0 = np.abs(np.random.randn(an) * 0.001).astype(np.float32)
w0.tofile('/tmp/aw_w.bin'); g0.tofile('/tmp/aw_g.bin')
m0.tofile('/tmp/aw_m.bin'); v0.tofile('/tmp/aw_v.bin')

lr, b1, b2, eps, wd = 1e-3, 0.9, 0.999, 1e-8, 0.01
stp = 5
bc1 = 1 - b1**stp; bc2 = 1 - b2**stp
m1 = b1*m0 + (1-b1)*g0
v1 = b2*v0 + (1-b2)*g0*g0
w1_ref = w0 * (1 - lr*wd) - lr * (m1/bc1) / (np.sqrt(v1/bc2) + eps)

# --- Run
r = subprocess.run(['/tmp/test_ops'], capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print("STDERR:", r.stderr); sys.exit(1)

# --- Verify
print("=" * 60)
ok = True
def chk(name, a, b, tol=1e-3):
    global ok
    d = np.abs(a - b).max()
    s = "PASS" if d < tol else "FAIL"
    if d >= tol: ok = False
    print(f"{name:22s} max_abs = {d:.3e}  {s}")

chk("GELU fwd",  np.fromfile('/tmp/gelu_y.bin',  dtype=np.float32), y_g_ref)
chk("GELU bwd",  np.fromfile('/tmp/gelu_dx.bin', dtype=np.float32), dx_g_ref)

chk("Softmax probs", np.fromfile('/tmp/sm_probs.bin', dtype=np.float32).reshape(T,V), probs_ref)
chk("CE loss",       np.fromfile('/tmp/sm_loss.bin',  dtype=np.float32), loss_ref)
chk("dlogits",       np.fromfile('/tmp/sm_dlogits.bin', dtype=np.float32).reshape(T,V), dlogits_ref)

chk("AdamW w",       np.fromfile('/tmp/aw_w_out.bin', dtype=np.float32), w1_ref)

print("=" * 60)
print("ALL PASS" if ok else "SOME FAILED")
