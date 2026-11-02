import numpy as np, subprocess, sys

T, D, DFF, H, L, V, TMAX = 4, 32, 64, 2, 3, 256, 8
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
r = subprocess.run(['/tmp/test_full_model'], capture_output=True, text=True)
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
print("=" * 55)
print("ALL PASS" if ok else "SOME FAILED")
