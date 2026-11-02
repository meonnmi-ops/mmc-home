import numpy as np, subprocess, sys

T, D, DFF, H, L = 4, 32, 64, 2, 3
hd = D // H; eps = 1e-5
np.random.seed(42)
def rn(*shape): return (np.random.randn(*shape) * 0.3).astype(np.float32)

X = rn(T,D); dY = (np.random.randn(T,D) * 0.1).astype(np.float32)
Wq=[rn(D,D) for _ in range(L)]; Wk=[rn(D,D) for _ in range(L)]
Wv=[rn(D,D) for _ in range(L)]; Wo=[rn(D,D) for _ in range(L)]
ln1g=[(np.random.randn(D)*0.1+1.0).astype(np.float32) for _ in range(L)]
ln1b=[(np.random.randn(D)*0.1).astype(np.float32) for _ in range(L)]
W1=[rn(D,DFF) for _ in range(L)]; b1=[(np.random.randn(DFF)*0.1).astype(np.float32) for _ in range(L)]
W2=[rn(DFF,D) for _ in range(L)]; b2=[(np.random.randn(D)*0.1).astype(np.float32) for _ in range(L)]
ln2g=[(np.random.randn(D)*0.1+1.0).astype(np.float32) for _ in range(L)]
ln2b=[(np.random.randn(D)*0.1).astype(np.float32) for _ in range(L)]

X.tofile('/tmp/lb_x.bin'); dY.tofile('/tmp/lb_dy.bin')
np.concatenate(Wq).tofile('/tmp/lb_wq.bin')
np.concatenate(Wk).tofile('/tmp/lb_wk.bin')
np.concatenate(Wv).tofile('/tmp/lb_wv.bin')
np.concatenate(Wo).tofile('/tmp/lb_wo.bin')
np.concatenate(ln1g).tofile('/tmp/lb_ln1g.bin')
np.concatenate(ln1b).tofile('/tmp/lb_ln1b.bin')
np.concatenate(W1).tofile('/tmp/lb_w1.bin')
np.concatenate(b1).tofile('/tmp/lb_b1.bin')
np.concatenate(W2).tofile('/tmp/lb_w2.bin')
np.concatenate(b2).tofile('/tmp/lb_b2.bin')
np.concatenate(ln2g).tofile('/tmp/lb_ln2g.bin')
np.concatenate(ln2b).tofile('/tmp/lb_ln2b.bin')

def ln_fwd(x,g,b):
    mu=x.mean(-1,keepdims=True); va=x.var(-1,keepdims=True)
    r=1.0/np.sqrt(va+eps); xh=(x-mu)*r
    return xh*g+b, r, mu
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
    Qh=Q.reshape(T,H,hd).transpose(1,0,2)
    Kh=K.reshape(T,H,hd).transpose(1,0,2)
    Vh=V.reshape(T,H,hd).transpose(1,0,2)
    sc=1.0/np.sqrt(hd)
    s_=np.einsum('htd,hcd->htc',Qh,Kh)*sc
    mx=s_.max(-1,keepdims=True); ex=np.exp(s_-mx)
    S=ex/ex.sum(-1,keepdims=True)
    Ah=np.einsum('htc,hcd->htd',S,Vh)
    return Ah.transpose(1,0,2).reshape(T,D), S
def mhsa_bwd(dA,Q,K,V,S):
    sc=1.0/np.sqrt(hd)
    Qh=Q.reshape(T,H,hd).transpose(1,0,2)
    Kh=K.reshape(T,H,hd).transpose(1,0,2)
    Vh=V.reshape(T,H,hd).transpose(1,0,2)
    dAh=dA.reshape(T,H,hd).transpose(1,0,2)
    dVh=np.einsum('htc,htd->hcd',S,dAh)
    dSp=np.einsum('htd,hcd->htc',dAh,Vh)
    rd=np.einsum('htc,htc->ht',S,dSp)[...,None]
    dS=S*(dSp-rd)*sc
    dQh=np.einsum('htc,hcd->htd',dS,Kh)
    dKh=np.einsum('htc,htd->hcd',dS,Qh)
    return (dQh.transpose(1,0,2).reshape(T,D),
            dKh.transpose(1,0,2).reshape(T,D),
            dVh.transpose(1,0,2).reshape(T,D))

def block_fwd(x,b):
    h1,r1,m1 = ln_fwd(x,ln1g[b],ln1b[b])
    Q=h1@Wq[b]; K=h1@Wk[b]; V=h1@Wv[b]
    A,S = mhsa(Q,K,V); O=A@Wo[b]; X1=x+O
    h2,r2,m2 = ln_fwd(X1,ln2g[b],ln2b[b])
    pre = h2@W1[b]+b1[b]; hg=gf(pre); M=hg@W2[b]+b2[b]
    return X1+M, dict(h1=h1,Q=Q,K=K,V=V,A=A,S=S,X1=X1,h2=h2,pre=pre,hg=hg,
                      r1=r1,m1=m1,r2=r2,m2=m2)
def block_bwd(dY,c,x,b):
    dX1 = dY.copy()
    dW2 = c['hg'].T@dY; db2=dY.sum(0)
    dpre = gb(dY@W2[b].T, c['pre'])
    dW1 = c['h2'].T@dpre; db1=dpre.sum(0)
    dh2 = dpre@W1[b].T; dX1 += dh2
    dh2_dx,dln2g,dln2b = ln_bwd(dh2, c['X1'], ln2g[b]); dX1 += dh2_dx
    dO = dX1; dWo = c['A'].T@dO; dA = dO@Wo[b].T
    dQ,dK,dV = mhsa_bwd(dA, c['Q'], c['K'], c['V'], c['S'])
    dWq = c['h1'].T@dQ; dWk = c['h1'].T@dK; dWv = c['h1'].T@dV
    dh1 = dQ@Wq[b].T+dK@Wk[b].T+dV@Wv[b].T
    dX = dX1.copy()
    dh1_dx,dln1g,dln1b = ln_bwd(dh1, x, ln1g[b]); dX += dh1_dx
    return dX, dict(dWq=dWq,dWk=dWk,dWv=dWv,dWo=dWo,dW1=dW1,db1=db1,
                    dW2=dW2,db2=db2,dln1g=dln1g,dln1b=dln1b,dln2g=dln2g,dln2b=dln2b)

x = X.copy(); xs=[x.copy()]; caches=[]
for b in range(L):
    y,c = block_fwd(x,b)
    caches.append(c); x = y; xs.append(x.copy())
Y = x

dXg = dY.copy(); grads=[None]*L
for b in range(L-1,-1,-1):
    dXg,g = block_bwd(dXg, caches[b], xs[b], b)
    grads[b] = g

r = subprocess.run(['/tmp/test_lblock'], capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0: print("STDERR:", r.stderr); sys.exit(1)

def chk(n,ref):
    got = np.fromfile(f'/tmp/lb_{n}.bin', dtype=np.float32).reshape(ref.shape)
    d = np.abs(got-ref).max()
    print(f"  {n:10s} max_abs = {d:.3e}  {'PASS' if d<1e-3 else 'FAIL'}")
    return d<1e-3

dWq_ref = np.concatenate([g['dWq'] for g in grads])
dWk_ref = np.concatenate([g['dWk'] for g in grads])
dWv_ref = np.concatenate([g['dWv'] for g in grads])
dWo_ref = np.concatenate([g['dWo'] for g in grads])

print("="*60)
ok = all([chk('y',Y), chk('dx',dXg),
          chk('dwq',dWq_ref), chk('dwk',dWk_ref),
          chk('dwv',dWv_ref), chk('dwo',dWo_ref)])
print("="*60)
print("ALL PASS" if ok else "SOME FAILED")
