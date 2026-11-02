import numpy as np, subprocess, sys

T, D, DFF, H = 4, 32, 64, 2
hd = D // H; eps = 1e-5
np.random.seed(42)

def rn(*shape): return (np.random.randn(*shape) * 0.3).astype(np.float32)
X=rn(T,D); Wq=rn(D,D); Wk=rn(D,D); Wv=rn(D,D); Wo=rn(D,D)
ln1g=(np.random.randn(D)*0.1+1.0).astype(np.float32)
ln1b=(np.random.randn(D)*0.1).astype(np.float32)
W1=rn(D,DFF); b1=(np.random.randn(DFF)*0.1).astype(np.float32)
W2=rn(DFF,D); b2=(np.random.randn(D)*0.1).astype(np.float32)
ln2g=(np.random.randn(D)*0.1+1.0).astype(np.float32)
ln2b=(np.random.randn(D)*0.1).astype(np.float32)
dY=(np.random.randn(T,D)*0.1).astype(np.float32)

for n,v in [('x',X),('wq',Wq),('wk',Wk),('wv',Wv),('wo',Wo),
            ('ln1g',ln1g),('ln1b',ln1b),('w1',W1),('b1',b1),
            ('w2',W2),('b2',b2),('ln2g',ln2g),('ln2b',ln2b),('dy',dY)]:
    v.tofile(f'/tmp/fb_{n}.bin')

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
    sc_=np.einsum('htd,hcd->htc',Qh,Kh)*sc
    mx=sc_.max(-1,keepdims=True); ex=np.exp(sc_-mx)
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

# Forward
h1,_,_=ln_fwd(X,ln1g,ln1b)
Q=h1@Wq; K=h1@Wk; V=h1@Wv
A,S=mhsa(Q,K,V); O=A@Wo
X1=X+O
h2,_,_=ln_fwd(X1,ln2g,ln2b)
pre=h2@W1+b1; hg=gf(pre); M=hg@W2+b2
Y=X1+M

# Backward
dX1=dY.copy()
dW2=hg.T@dY; db2=dY.sum(0)
dhg=dY@W2.T; dpre=gb(dhg,pre)
dW1=h2.T@dpre; db1=dpre.sum(0)
dh2=dpre@W1.T                    # [T,D] — grad wrt LN2 output
dX1 += dh2                        # path through MLP
dh2_dx,dln2g,dln2b=ln_bwd(dh2,X1,ln2g)
dX1 += dh2_dx
dO=dX1; dWo=A.T@dO; dA=dO@Wo.T
dQ,dK,dV=mhsa_bwd(dA,Q,K,V,S)
dWq=h1.T@dQ; dWk=h1.T@dK; dWv=h1.T@dV
dh1=dQ@Wq.T+dK@Wk.T+dV@Wv.T
dX=dX1.copy()
dh1_dx,dln1g,dln1b=ln_bwd(dh1,X,ln1g)
dX += dh1_dx

r=subprocess.run(['/tmp/test_full_block'],capture_output=True,text=True)
print(r.stdout)
if r.returncode!=0: print("STDERR:",r.stderr); sys.exit(1)

def chk(n,ref,shape):
    got=np.fromfile(f'/tmp/fb_{n}.bin',dtype=np.float32).reshape(shape)
    d=np.abs(got-ref).max()
    s='PASS' if d<1e-3 else 'FAIL'
    print(f"  {n:10s} max_abs = {d:.3e}  {s}")
    return d<1e-3

print("="*60)
ok=all([chk('y',Y,(T,D)), chk('dx',dX,(T,D)),
        chk('dwq',dWq,(D,D)), chk('dwk',dWk,(D,D)),
        chk('dwv',dWv,(D,D)), chk('dwo',dWo,(D,D)),
        chk('dln1g',dln1g,(D,)), chk('dln1b',dln1b,(D,)),
        chk('dw1',dW1,(D,DFF)), chk('db1',db1,(DFF,)),
        chk('dw2',dW2,(DFF,D)), chk('db2',db2,(D,)),
        chk('dln2g',dln2g,(D,)), chk('dln2b',dln2b,(D,))])
print("="*60)
print("ALL PASS" if ok else "SOME FAILED")
