import numpy as np, subprocess, sys

T, D, DFF, H, L, V, TMAX = 16, 32, 64, 2, 3, 256, 16
hd = D // H; eps = 1e-5
np.random.seed(42)
def rn(*s): return (np.random.randn(*s) * 0.3).astype(np.float32)

ids = np.random.randint(0, V, T).astype(np.int32)
tg  = np.random.randint(0, V, T).astype(np.int32)
wte = rn(V, D); wpe = rn(TMAX, D)
lnfg = (np.random.randn(D)*0.1+1.0).astype(np.float32)
lnfb = (np.random.randn(D)*0.1).astype(np.float32)
Wq=[rn(D,D) for _ in range(L)]; Wk=[rn(D,D) for _ in range(L)]
Wv=[rn(D,D) for _ in range(L)]; Wo=[rn(D,D) for _ in range(L)]
ln1g=[(np.random.randn(D)*0.1+1.0).astype(np.float32) for _ in range(L)]
ln1b=[(np.random.randn(D)*0.1).astype(np.float32) for _ in range(L)]
W1=[rn(D,DFF) for _ in range(L)]; b1=[(np.random.randn(DFF)*0.1).astype(np.float32) for _ in range(L)]
W2=[rn(DFF,D) for _ in range(L)]; b2=[(np.random.randn(D)*0.1).astype(np.float32) for _ in range(L)]
ln2g=[(np.random.randn(D)*0.1+1.0).astype(np.float32) for _ in range(L)]
ln2b=[(np.random.randn(D)*0.1).astype(np.float32) for _ in range(L)]

ids.tofile('/tmp/fm_ids.bin'); tg.tofile('/tmp/fm_tg.bin')
wte.tofile('/tmp/fm_wte.bin'); wpe.tofile('/tmp/fm_wpe.bin')
lnfg.tofile('/tmp/fm_lnfg.bin'); lnfb.tofile('/tmp/fm_lnfb.bin')
np.concatenate(Wq).tofile('/tmp/fm_wq.bin')
np.concatenate(Wk).tofile('/tmp/fm_wk.bin')
np.concatenate(Wv).tofile('/tmp/fm_wv.bin')
np.concatenate(Wo).tofile('/tmp/fm_wo.bin')
np.concatenate(ln1g).tofile('/tmp/fm_ln1g.bin')
np.concatenate(ln1b).tofile('/tmp/fm_ln1b.bin')
np.concatenate(W1).tofile('/tmp/fm_w1.bin')
np.concatenate(b1).tofile('/tmp/fm_b1.bin')
np.concatenate(W2).tofile('/tmp/fm_w2.bin')
np.concatenate(b2).tofile('/tmp/fm_b2.bin')
np.concatenate(ln2g).tofile('/tmp/fm_ln2g.bin')
np.concatenate(ln2b).tofile('/tmp/fm_ln2b.bin')

def ln_fwd(x,g,b):
    mu=x.mean(-1,keepdims=True); va=x.var(-1,keepdims=True)
    r=1.0/np.sqrt(va+eps); return (x-mu)*r*g+b
def ln_bwd(dy,x,g):
    mu=x.mean(-1,keepdims=True); va=x.var(-1,keepdims=True)
    r=1.0/np.sqrt(va+eps); xh=(x-mu)*r; dxhat=dy*g
    dx=r*(dxhat - dxhat.mean(-1,keepdims=True)
          - xh*(dxhat*xh).mean(-1,keepdims=True))
    return dx, (dy*xh).sum(0), dy.sum(0)
def gf(x): return 0.5*x*(1+np.tanh(0.7978845608*(x+0.044715*x**3)))
def gb(dy,x):
    t=np.tanh(0.7978845608*(x+0.044715*x**3))
    dt=0.7978845608*(1+3*0.044715*x**2)
    return dy*(0.5*(1+t)+0.5*x*(1-t*t)*dt)
def mhsa(Q,K,V):
    Qh=Q.reshape(T,H,hd).transpose(1,0,2); Kh=K.reshape(T,H,hd).transpose(1,0,2)
    Vh=V.reshape(T,H,hd).transpose(1,0,2); sc_=1.0/np.sqrt(hd)
    s_=np.einsum('htd,hcd->htc',Qh,Kh)*sc_
    mx=s_.max(-1,keepdims=True); ex=np.exp(s_-mx); S=ex/ex.sum(-1,keepdims=True)
    Ah=np.einsum('htc,hcd->htd',S,Vh)
    return Ah.transpose(1,0,2).reshape(T,D)
def mhsa_bwd(dA,Q,K,V):
    sc_=1.0/np.sqrt(hd)
    Qh=Q.reshape(T,H,hd).transpose(1,0,2); Kh=K.reshape(T,H,hd).transpose(1,0,2)
    Vh=V.reshape(T,H,hd).transpose(1,0,2); dAh=dA.reshape(T,H,hd).transpose(1,0,2)
    s_=np.einsum('htd,hcd->htc',Qh,Kh)*sc_
    mx=s_.max(-1,keepdims=True); ex=np.exp(s_-mx); S=ex/ex.sum(-1,keepdims=True)
    dVh=np.einsum('htc,htd->hcd',S,dAh)
    dSp=np.einsum('htd,hcd->htc',dAh,Vh)
    rd=np.einsum('htc,htc->ht',S,dSp)[...,None]
    dS=S*(dSp-rd)*sc_
    dQh=np.einsum('htc,hcd->htd',dS,Kh)
    dKh=np.einsum('htc,htd->hcd',dS,Qh)
    return (dQh.transpose(1,0,2).reshape(T,D),
            dKh.transpose(1,0,2).reshape(T,D),
            dVh.transpose(1,0,2).reshape(T,D))

# Forward
x0 = wte[ids] + wpe[:T]
xs = [x0.copy()]; caches = []
x = x0
for b in range(L):
    h1 = ln_fwd(x, ln1g[b], ln1b[b])
    Q=h1@Wq[b]; K=h1@Wk[b]; Vv=h1@Wv[b]
    A = mhsa(Q,K,Vv); O=A@Wo[b]; X1 = x+O
    h2 = ln_fwd(X1, ln2g[b], ln2b[b])
    pre = h2@W1[b]+b1[b]; hg=gf(pre); M=hg@W2[b]+b2[b]
    x = X1 + M
    caches.append(dict(h1=h1,Q=Q,K=K,V=Vv,A=A,X1=X1,h2=h2,pre=pre,hg=hg))
    xs.append(x.copy())
Y = x
xnf = ln_fwd(Y, lnfg, lnfb)
logits = xnf @ wte.T
mx = logits.max(-1, keepdims=True)
e = np.exp(logits - mx)
probs = e / e.sum(-1, keepdims=True)
loss = -(logits[np.arange(T), tg] - mx.flatten() - np.log(e.sum(-1)))

# Backward
dlogits = probs.copy()
dlogits[np.arange(T), tg] -= 1.0
dlogits /= T

# Tied head bwd
dxnf = dlogits @ wte
dwte_ref = dlogits.T @ xnf

# LNf bwd
dY, dlnfg_ref, dlnfb_ref = ln_bwd(dxnf, Y, lnfg)

# Block chain bwd
dX = dY.copy()
grads = [None]*L
for b in range(L-1,-1,-1):
    c = caches[b]
    dX1 = dX.copy()
    dW2_ = c['hg'].T @ dX; db2_ = dX.sum(0)
    dpre = gb(dX @ W2[b].T, c['pre'])
    dW1_ = c['h2'].T @ dpre; db1_ = dpre.sum(0)
    dh2 = dpre @ W1[b].T; dX1 += dh2
    dh2_dx, dln2g_, dln2b_ = ln_bwd(dh2, c['X1'], ln2g[b]); dX1 += dh2_dx
    dO = dX1; dWo_ = c['A'].T @ dO; dA = dO @ Wo[b].T
    dQ,dK,dV = mhsa_bwd(dA, c['Q'], c['K'], c['V'])
    dWq_ = c['h1'].T @ dQ; dWk_ = c['h1'].T @ dK; dWv_ = c['h1'].T @ dV
    dh1 = dQ@Wq[b].T + dK@Wk[b].T + dV@Wv[b].T
    dX = dX1.copy()
    dh1_dx, dln1g_, dln1b_ = ln_bwd(dh1, xs[b], ln1g[b]); dX += dh1_dx
    grads[b] = dict(dWq=dWq_, dWk=dWk_, dWv=dWv_, dWo=dWo_,
                    dW1=dW1_, db1=db1_, dW2=dW2_, db2=db2_,
                    dln1g=dln1g_, dln1b=dln1b_,
                    dln2g=dln2g_, dln2b=dln2b_)

# Emb bwd
for t in range(T):
    dwte_ref[ids[t]] += dX[t]
dwpe_ref = np.zeros_like(wpe)
dwpe_ref[:T] = dX

# Run
# AdamW step (step=1, m=0, v=0)
def adamw_step(w, g, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.01, step=1):
    bc1 = 1 - b1**step
    bc2 = 1 - b2**step
    m = (1-b1)*g
    v = (1-b2)*g*g
    return w * (1 - lr*wd) - lr * (m/bc1) / (np.sqrt(v/bc2) + eps)

wte_new  = adamw_step(wte, dwte_ref)
wpe_new  = adamw_step(wpe, dwpe_ref)
lnfg_new = adamw_step(lnfg, dlnfg_ref)
lnfb_new = adamw_step(lnfb, dlnfb_ref)
Wq_new   = [adamw_step(Wq[b],  grads[b]['dWq'])  for b in range(L)]
Wk_new   = [adamw_step(Wk[b],  grads[b]['dWk'])  for b in range(L)]
Wv_new   = [adamw_step(Wv[b],  grads[b]['dWv'])  for b in range(L)]
Wo_new   = [adamw_step(Wo[b],  grads[b]['dWo'])  for b in range(L)]
ln1g_new = [adamw_step(ln1g[b], grads[b]['dln1g']) for b in range(L)]
ln1b_new = [adamw_step(ln1b[b], grads[b]['dln1b']) for b in range(L)]
W1_new   = [adamw_step(W1[b],  grads[b]['dW1'])  for b in range(L)]
b1_new   = [adamw_step(b1[b],  grads[b]['db1'])  for b in range(L)]
W2_new   = [adamw_step(W2[b],  grads[b]['dW2'])  for b in range(L)]
b2_new   = [adamw_step(b2[b],  grads[b]['db2'])  for b in range(L)]
ln2g_new = [adamw_step(ln2g[b], grads[b]['dln2g']) for b in range(L)]
ln2b_new = [adamw_step(ln2b[b], grads[b]['dln2b']) for b in range(L)]

r = subprocess.run(['/tmp/test_train_step'], capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0: print("STDERR:", r.stderr); sys.exit(1)

def chk(n, ref, shape):
    got = np.fromfile(f'/tmp/fm_{n}.bin', dtype=np.float32).reshape(shape)
    d = np.abs(got - ref).max()
    print(f"  {n:8s} max_abs = {d:.3e}  {'PASS' if d < 1e-3 else 'FAIL'}")
    return d < 1e-3

dWq_ref = np.stack([g["dWq"] for g in grads])
dWk_ref = np.stack([g["dWk"] for g in grads])
dWv_ref = np.stack([g["dWv"] for g in grads])
dWo_ref = np.stack([g["dWo"] for g in grads])

print("=" * 55)
ok = all([
    chk('y', Y, (T,D)),
    chk('probs', probs, (T,V)),
    chk('loss', loss, (T,)),
    chk('dx', dX, (T,D)),
    chk('dwte', dwte_ref, (V,D)),
    chk('dwpe', dwpe_ref, (TMAX,D)),
    chk('dlnfg', dlnfg_ref, (D,)),
    chk('dlnfb', dlnfb_ref, (D,)),
    chk('dwq', dWq_ref, (L,D,D)),
    chk('dwk', dWk_ref, (L,D,D)),
    chk('dwv', dWv_ref, (L,D,D)),
    chk('dwo', dWo_ref, (L,D,D)),
])
# --- Weight update checks (AdamW step 1) ---
def chk_w(name, ref, shape):
    got = np.fromfile(f'/tmp/ts_s1_{name}.bin', dtype=np.float32).reshape(shape)
    d = np.abs(got - ref).max()
    print(f"  {name:8s} max_abs = {d:.3e}  {'PASS' if d < 1e-3 else 'FAIL'}")
    return d < 1e-3

print()
print("--- AdamW updated weights (step 1) ---")
ok_w = all([
    chk_w('wte',  wte_new,  (V, D)),
    chk_w('wpe',  wpe_new,  (TMAX, D)),
    chk_w('lnfg', lnfg_new, (D,)),
    chk_w('lnfb', lnfb_new, (D,)),
    chk_w('wq',   np.stack(Wq_new),   (L, D, D)),
    chk_w('wk',   np.stack(Wk_new),   (L, D, D)),
    chk_w('wv',   np.stack(Wv_new),   (L, D, D)),
    chk_w('wo',   np.stack(Wo_new),   (L, D, D)),
    chk_w('ln1g', np.stack(ln1g_new), (L, D)),
    chk_w('ln1b', np.stack(ln1b_new), (L, D)),
    chk_w('w1',   np.stack(W1_new),   (L, D, DFF)),
    chk_w('b1',   np.stack(b1_new),   (L, DFF)),
    chk_w('w2',   np.stack(W2_new),   (L, DFF, D)),
    chk_w('b2',   np.stack(b2_new),   (L, D)),
    chk_w('ln2g', np.stack(ln2g_new), (L, D)),
    chk_w('ln2b', np.stack(ln2b_new), (L, D)),
])

print("=" * 55)
def chk_w2(name, ref, shape):
    got = np.fromfile(f'/tmp/ts_s2_{name}.bin', dtype=np.float32).reshape(shape)
    d = np.abs(got - ref).max()
    print(f"  s2_{name:6s} max_abs = {d:.3e}  {'PASS' if d < 1e-3 else 'FAIL'}")
    return d < 1e-3

print()
# ===== NEW step-2 refs: REAL (w1 from ts_s1, g2 from fm_s2) =====
def ld_s1(name, shape):
    return np.fromfile(f'/tmp/ts_s1_{name}.bin', dtype=np.float32).reshape(shape)
def ld_g2(name, shape):
    return np.fromfile(f'/tmp/fm_s2_{name}.bin', dtype=np.float32).reshape(shape)

wte_w1  = ld_s1('wte',  (V, D));     wpe_w1  = ld_s1('wpe',  (TMAX, D))
lnfg_w1 = ld_s1('lnfg', (D,));       lnfb_w1 = ld_s1('lnfb', (D,))
Wq_w1   = ld_s1('wq',   (L, D, D));  Wk_w1   = ld_s1('wk',   (L, D, D))
Wv_w1   = ld_s1('wv',   (L, D, D));  Wo_w1   = ld_s1('wo',   (L, D, D))
ln1g_w1 = ld_s1('ln1g', (L, D));     ln1b_w1 = ld_s1('ln1b', (L, D))
W1_w1   = ld_s1('w1',   (L, D, DFF));b1_w1   = ld_s1('b1',   (L, DFF))
W2_w1   = ld_s1('w2',   (L, DFF, D));b2_w1   = ld_s1('b2',   (L, D))
ln2g_w1 = ld_s1('ln2g', (L, D));     ln2b_w1 = ld_s1('ln2b', (L, D))

dwte_g2  = ld_g2('dwte',  (V, D));    dwpe_g2  = ld_g2('dwpe',  (TMAX, D))
dlnfg_g2 = ld_g2('dlnfg', (D,));      dlnfb_g2 = ld_g2('dlnfb', (D,))
dWq_g2   = ld_g2('dwq',   (L, D, D)); dWk_g2   = ld_g2('dwk',   (L, D, D))
dWv_g2   = ld_g2('dwv',   (L, D, D)); dWo_g2   = ld_g2('dwo',   (L, D, D))
dln1g_g2 = ld_g2('dln1g', (L, D));    dln1b_g2 = ld_g2('dln1b', (L, D))
dW1_g2   = ld_g2('dw1',   (L, D, DFF));db1_g2  = ld_g2('db1',   (L, DFF))
dW2_g2   = ld_g2('dw2',   (L, DFF, D));db2_g2  = ld_g2('db2',   (L, D))
dln2g_g2 = ld_g2('dln2g', (L, D));    dln2b_g2 = ld_g2('dln2b', (L, D))
dwte_g1  = np.fromfile('/tmp/fm_dwte.bin', dtype=np.float32).reshape((V, D))
dwpe_g1  = np.fromfile('/tmp/fm_dwpe.bin', dtype=np.float32).reshape((TMAX, D))
dlnfg_g1 = np.fromfile('/tmp/fm_dlnfg.bin', dtype=np.float32).reshape((D,))
dlnfb_g1 = np.fromfile('/tmp/fm_dlnfb.bin', dtype=np.float32).reshape((D,))
dWq_g1   = np.fromfile('/tmp/fm_dwq.bin', dtype=np.float32).reshape((L, D, D))
dWk_g1   = np.fromfile('/tmp/fm_dwk.bin', dtype=np.float32).reshape((L, D, D))
dWv_g1   = np.fromfile('/tmp/fm_dwv.bin', dtype=np.float32).reshape((L, D, D))
dWo_g1   = np.fromfile('/tmp/fm_dwo.bin', dtype=np.float32).reshape((L, D, D))
dln1g_g1 = np.fromfile('/tmp/fm_dln1g.bin', dtype=np.float32).reshape((L, D))
dln1b_g1 = np.fromfile('/tmp/fm_dln1b.bin', dtype=np.float32).reshape((L, D))
dW1_g1   = np.fromfile('/tmp/fm_dw1.bin', dtype=np.float32).reshape((L, D, DFF))
db1_g1   = np.fromfile('/tmp/fm_db1.bin', dtype=np.float32).reshape((L, DFF))
dW2_g1   = np.fromfile('/tmp/fm_dw2.bin', dtype=np.float32).reshape((L, DFF, D))
db2_g1   = np.fromfile('/tmp/fm_db2.bin', dtype=np.float32).reshape((L, D))
dln2g_g1 = np.fromfile('/tmp/fm_dln2g.bin', dtype=np.float32).reshape((L, D))
dln2b_g1 = np.fromfile('/tmp/fm_dln2b.bin', dtype=np.float32).reshape((L, D))


y1 = np.fromfile('/tmp/fm_y.bin', dtype=np.float32)
y2 = np.fromfile('/tmp/fm_s2_y.bin', dtype=np.float32)
dy = np.abs(y1 - y2).max()
print(f"  sanity: |y2-y1| = {dy:.3e}  {'CHANGED (good)' if dy > 1e-4 else 'SAME (BAD)'}")

def adamw_step2(w1, g1, g2, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.01):
    m1 = (1-b1)*g1
    v1 = (1-b2)*g1*g1
    m2 = b1*m1 + (1-b1)*g2
    v2 = b2*v1 + (1-b2)*g2*g2
    bc1 = 1 - b1**2
    bc2 = 1 - b2**2
    return w1*(1-lr*wd) - lr*(m2/bc1) / (np.sqrt(v2/bc2) + eps)

wte_2  = adamw_step2(wte_w1,  dwte_g1,  dwte_g2)
wpe_2  = adamw_step2(wpe_w1,  dwpe_g1,  dwpe_g2)
lnfg_2 = adamw_step2(lnfg_w1, dlnfg_g1, dlnfg_g2)
lnfb_2 = adamw_step2(lnfb_w1, dlnfb_g1, dlnfb_g2)
Wq_2   = np.stack([adamw_step2(Wq_w1[b], dWq_g1[b], dWq_g2[b]) for b in range(L)])
Wk_2   = np.stack([adamw_step2(Wk_w1[b], dWk_g1[b], dWk_g2[b]) for b in range(L)])
Wv_2   = np.stack([adamw_step2(Wv_w1[b], dWv_g1[b], dWv_g2[b]) for b in range(L)])
Wo_2   = np.stack([adamw_step2(Wo_w1[b], dWo_g1[b], dWo_g2[b]) for b in range(L)])
ln1g_2 = np.stack([adamw_step2(ln1g_w1[b], dln1g_g1[b], dln1g_g2[b]) for b in range(L)])
ln1b_2 = np.stack([adamw_step2(ln1b_w1[b], dln1b_g1[b], dln1b_g2[b]) for b in range(L)])
W1_2   = np.stack([adamw_step2(W1_w1[b], dW1_g1[b], dW1_g2[b]) for b in range(L)])
b1_2   = np.stack([adamw_step2(b1_w1[b], db1_g1[b], db1_g2[b]) for b in range(L)])
W2_2   = np.stack([adamw_step2(W2_w1[b], dW2_g1[b], dW2_g2[b]) for b in range(L)])
b2_2   = np.stack([adamw_step2(b2_w1[b], db2_g1[b], db2_g2[b]) for b in range(L)])
ln2g_2 = np.stack([adamw_step2(ln2g_w1[b], dln2g_g1[b], dln2g_g2[b]) for b in range(L)])
ln2b_2 = np.stack([adamw_step2(ln2b_w1[b], dln2b_g1[b], dln2b_g2[b]) for b in range(L)])

print("--- AdamW step 2 (REAL: w1 + g2 from fresh fwd+bwd) ---")
ok_w2 = all([
    chk_w2('wte',  wte_2,  (V, D)),
    chk_w2('wpe',  wpe_2,  (TMAX, D)),
    chk_w2('lnfg', lnfg_2, (D,)),
    chk_w2('lnfb', lnfb_2, (D,)),
    chk_w2('wq',   np.stack(Wq_2),   (L, D, D)),
    chk_w2('wk',   np.stack(Wk_2),   (L, D, D)),
    chk_w2('wv',   np.stack(Wv_2),   (L, D, D)),
    chk_w2('wo',   np.stack(Wo_2),   (L, D, D)),
    chk_w2('ln1g', np.stack(ln1g_2), (L, D)),
    chk_w2('ln1b', np.stack(ln1b_2), (L, D)),
    chk_w2('w1',   np.stack(W1_2),   (L, D, DFF)),
    chk_w2('b1',   np.stack(b1_2),   (L, DFF)),
    chk_w2('w2',   np.stack(W2_2),   (L, DFF, D)),
    chk_w2('b2',   np.stack(b2_2),   (L, D)),
    chk_w2('ln2g', np.stack(ln2g_2), (L, D)),
    chk_w2('ln2b', np.stack(ln2b_2), (L, D)),
])

# ===== Loss trajectory =====
print()
print("--- Loss trajectory ---")
losses = []
for k in range(1, 501):
    try:
        v = np.fromfile(f'/tmp/loss_s{k}.bin', dtype=np.float32)
        if len(v) > 0:
            losses.append((k, float(v[0])))
    except FileNotFoundError:
        break
for k, v in losses:
    print(f"  loss_s{k:2d} = {v:.6f}")
if len(losses) >= 2:
    first, last = losses[0][1], losses[-1][1]
    print()
    print(f"  Trajectory: {first:.6f} -> {last:.6f}  (delta = {last-first:+.6f})")
    if last < first - 1e-4:
        print("  ✓ LOSS DECREASED — training is working")
    elif last < first:
        print("  ~ Small decrease (bump N for clarity)")
    else:
        print("  ✗ No decrease — check lr/data")

# ===== Validation loss =====
print()
print("--- Validation loss (30 held-out batches) ---")
vals = []
for v in range(1, 31):
    try:
        vv = np.fromfile(f'/tmp/loss_val{v}.bin', dtype=np.float32)
        if len(vv) > 0:
            vals.append((v, float(vv[0])))
    except FileNotFoundError:
        break
for v, val in vals:
    print(f"  val_{v:2d} = {val:.6f}")
if len(vals) >= 1 and len(losses) >= 2:
    arr = np.array([v for _, v in vals])
    print()
    print(f"  Train final: {losses[-1][1]:.6f}")
    print(f"  Val mean:    {arr.mean():.6f}")
    print(f"  Val median:  {np.median(arr):.6f}")
    print(f"  Val std:     {arr.std():.6f}")
    print(f"  Val min/max: {arr.min():.4f} / {arr.max():.4f}")
    print()
    # Reasonable check: val_mean vs train_final gap < 2.0 is fine
    gap = arr.mean() - losses[-1][1]
    if gap < 2.0:
        print(f"  ✓ Generalization OK (train->val gap = {gap:.3f})")
    else:
        print(f"  ✗ Gap too large: {gap:.3f}")

print("ALL PASS (grads + step1 + step2)" if (ok and ok_w and ok_w2) else "SOME FAILED")
