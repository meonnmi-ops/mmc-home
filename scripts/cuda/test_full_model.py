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
dY   = (np.random.randn(T,D)*0.1).astype(np.float32)
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
dY.tofile('/tmp/fm_dy.bin')
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
def gf(x): return 0.5*x*(1+np.tanh(0.7978845608*(x+0.044715*x**3)))
def mhsa(Q,K,V):
    Qh=Q.reshape(T,H,hd).transpose(1,0,2); Kh=K.reshape(T,H,hd).transpose(1,0,2)
    Vh=V.reshape(T,H,hd).transpose(1,0,2); sc_=1.0/np.sqrt(hd)
    s_=np.einsum('htd,hcd->htc',Qh,Kh)*sc_
    mx=s_.max(-1,keepdims=True); ex=np.exp(s_-mx); S=ex/ex.sum(-1,keepdims=True)
    Ah=np.einsum('htc,hcd->htd',S,Vh)
    return Ah.transpose(1,0,2).reshape(T,D)

# forward
x = wte[ids] + wpe[:T]
for b in range(L):
    h1 = ln_fwd(x, ln1g[b], ln1b[b])
    Q=h1@Wq[b]; K=h1@Wk[b]; Vv=h1@Wv[b]
    A = mhsa(Q,K,Vv); O=A@Wo[b]; X1 = x + O
    h2 = ln_fwd(X1, ln2g[b], ln2b[b])
    pre = h2@W1[b]+b1[b]; hg = gf(pre); M = hg@W2[b]+b2[b]
    x = X1 + M
Y = x
xnf = ln_fwd(Y, lnfg, lnfb)
logits = xnf @ wte.T
mx = logits.max(-1, keepdims=True)
e = np.exp(logits - mx)
probs = e / e.sum(-1, keepdims=True)
loss = -(logits[np.arange(T), tg] - mx.flatten() - np.log(e.sum(-1)))

r = subprocess.run(['/tmp/test_full_model'], capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0: print("STDERR:", r.stderr); sys.exit(1)

def chk(n, ref, shape):
    got = np.fromfile(f'/tmp/fm_{n}.bin', dtype=np.float32).reshape(shape)
    d = np.abs(got - ref).max()
    print(f"  {n:8s} max_abs = {d:.3e}  {'PASS' if d < 1e-3 else 'FAIL'}")
    return d < 1e-3

print("=" * 55)
ok = all([chk('y', Y, (T,D)), chk('probs', probs, (T,V)), chk('loss', loss, (T,))])
print("=" * 55)
print("FORWARD PASS" if ok else "FORWARD FAIL")
