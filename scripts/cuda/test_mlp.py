import numpy as np, subprocess, sys

T, D, DFF = 4, 32, 64
np.random.seed(42)
X  = (np.random.randn(T, D)    * 0.3).astype(np.float32)
W1 = (np.random.randn(D, DFF)  * 0.3).astype(np.float32)
b1 = (np.random.randn(DFF)     * 0.1).astype(np.float32)
W2 = (np.random.randn(DFF, D)  * 0.3).astype(np.float32)
b2 = (np.random.randn(D)       * 0.1).astype(np.float32)
dY = (np.random.randn(T, D)    * 0.1).astype(np.float32)

X.tofile('/tmp/mlp_x.bin'); W1.tofile('/tmp/mlp_w1.bin')
b1.tofile('/tmp/mlp_b1.bin'); W2.tofile('/tmp/mlp_w2.bin')
b2.tofile('/tmp/mlp_b2.bin'); dY.tofile('/tmp/mlp_dy.bin')

# Forward
H1  = X @ W1 + b1
G   = 0.5*H1*(1+np.tanh(0.7978845608*(H1+0.044715*H1**3)))
Y   = G @ W2 + b2

# Backward
dW2 = G.T @ dY
db2 = dY.sum(axis=0)
dG  = dY @ W2.T
inner = 0.7978845608*(H1+0.044715*H1**3)
t = np.tanh(inner)
dt = 0.7978845608*(1+3*0.044715*H1**2)
dH1 = dG * (0.5*(1+t) + 0.5*H1*(1-t*t)*dt)
dW1 = X.T @ dH1
db1 = dH1.sum(axis=0)
dX  = dH1 @ W1.T

# Run CUDA
r = subprocess.run(['/tmp/test_mlp'], capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print("STDERR:", r.stderr); sys.exit(1)

# Verify
def chk(name, ref, shape):
    got = np.fromfile(f'/tmp/mlp_{name}.bin', dtype=np.float32).reshape(shape)
    d = np.abs(got - ref).max()
    s = 'PASS' if d < 1e-3 else 'FAIL'
    print(f"  {name:5s} max_abs = {d:.3e}  {s}")
    return d < 1e-3

print("=" * 50)
ok = all([
    chk('y',   Y,   (T, D)),
    chk('dw1', dW1, (D, DFF)),
    chk('db1', db1, (DFF,)),
    chk('dw2', dW2, (DFF, D)),
    chk('db2', db2, (D,)),
    chk('dx',  dX,  (T, D)),
])
print("=" * 50)
print("ALL PASS" if ok else "SOME FAILED")
