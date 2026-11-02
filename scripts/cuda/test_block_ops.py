import numpy as np, subprocess, sys

M, D = 4, 32
eps = 1e-5
np.random.seed(42)
X  = (np.random.randn(M,D)*0.3).astype(np.float32)
G  = (np.random.randn(D)*0.1+1.0).astype(np.float32)
B  = (np.random.randn(D)*0.1).astype(np.float32)
dY = (np.random.randn(M,D)*0.1).astype(np.float32)
X.tofile('/tmp/bo_x.bin'); G.tofile('/tmp/bo_g.bin')
B.tofile('/tmp/bo_b.bin'); dY.tofile('/tmp/bo_dy.bin')

# Reference
mu = X.mean(axis=1, keepdims=True)
va = X.var(axis=1, keepdims=True)
r  = 1.0/np.sqrt(va+eps)
xh = (X-mu)*r
Y_ref = xh*G + B

dgamma_ref = (dY*xh).sum(axis=0)
dbeta_ref  = dY.sum(axis=0)
dxhat = dY*G
dx_ref = r*(dxhat - dxhat.mean(axis=1, keepdims=True)
            - xh*(dxhat*xh).mean(axis=1, keepdims=True))

# GELU
inner = 0.7978845608*(X+0.044715*X**3)
t = np.tanh(inner)
gy_ref = 0.5*X*(1+t)
dt = 0.7978845608*(1+3*0.044715*X**2)
gdx_ref = dY*(0.5*(1+t) + 0.5*X*(1-t*t)*dt)

r = subprocess.run(['/tmp/test_block_ops'], capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print("STDERR:", r.stderr); sys.exit(1)

def chk(name, ref, shape):
    got = np.fromfile(f'/tmp/bo_{name}.bin', dtype=np.float32).reshape(shape)
    d = np.abs(got - ref).max()
    print(f"  {name:10s} max_abs = {d:.3e}  {'PASS' if d<1e-3 else 'FAIL'}")
    return d < 1e-3

print("=" * 55)
ok = all([
    chk('y',        Y_ref,     (M,D)),
    chk('dx',       dx_ref,    (M,D)),
    chk('dg',       dgamma_ref,(D,)),
    chk('db',       dbeta_ref, (D,)),
    chk('gelu_y',   gy_ref,    (M,D)),
    chk('gelu_dx',  gdx_ref,   (M,D)),
])
print("=" * 55)
print("ALL PASS" if ok else "SOME FAILED")
