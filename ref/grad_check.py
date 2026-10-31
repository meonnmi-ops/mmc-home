"""
grad_check.py — NYANLIN Cave 2: numerical vs analytical gradient (float64)
Run: cd ~/mmc_publish/ref && python grad_check.py
"""
import numpy as np
import torch
import torch.nn.functional as F


def loss_fn(model, idx):
    logits = model(idx)
    sl = logits[:, :-1, :].contiguous()
    st = idx[:, 1:].contiguous()
    return F.cross_entropy(sl.view(-1, sl.size(-1)), st.view(-1))


def check(model, idx, sample=None, eps=1e-5, tag="",
          per_tensor=8, atol=1e-7, rtol=1e-4):
    model.double()   # critical: finite diff needs float64
    model.zero_grad()
    loss = loss_fn(model, idx)
    loss.backward()

    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if sample is not None:
        rng = np.random.default_rng(42)
        sel = rng.choice(len(params), min(sample, len(params)), replace=False)
        params = [params[i] for i in sel]

    print(f"\n[{tag}] loss={loss.item():.6f}  tensors={len(params)}  eps={eps}  dtype=float64")
    print(f"{'param':<18s} {'idx':<6s} {'analytical':>13s} {'numerical':>13s} {'abs_err':>9s} {'rel_err':>9s}")

    worst_ratio = 0.0
    worst_name = ""
    total = n_pass = n_fail = 0

    for name, p in params:
        g = p.grad.detach().clone().view(-1)
        d = p.data.view(-1)
        n_elem = d.numel()

        if n_elem <= per_tensor:
            idxs = list(range(n_elem))
        else:
            idxs = np.random.default_rng(1).choice(n_elem, per_tensor, replace=False).tolist()

        for i in idxs:
            i = int(i)
            old = d[i].item()

            d[i] = old + eps
            with torch.no_grad():
                lp = loss_fn(model, idx).item()

            d[i] = old - eps
            with torch.no_grad():
                lm = loss_fn(model, idx).item()

            d[i] = old

            num_g = (lp - lm) / (2 * eps)
            ana_g = g[i].item()
            abs_e = abs(num_g - ana_g)
            denom = max(abs(num_g), abs(ana_g), 1e-8)
            rel_e = abs_e / denom

            tol = atol + rtol * max(abs(num_g), abs(ana_g))
            ratio = abs_e / tol
            if ratio > worst_ratio:
                worst_ratio = ratio
                worst_name = f"{name}[{i}]"
            if abs_e <= tol:
                n_pass += 1
            else:
                n_fail += 1

            total += 1
            if total <= 12:
                print(f"{name:<18s} {i:<6d} {ana_g:>13.5e} {num_g:>13.5e} {abs_e:>9.2e} {rel_e:>9.2e}")

    print(f"[{tag}] checked={total}  pass={n_pass}  fail={n_fail}  worst_ratio={worst_ratio:.2f} at {worst_name}")
    print(f"[{tag}] criterion: abs_err < {atol} + {rtol}*max(|g|)")
    return worst_ratio, n_fail


def run_mini():
    import ref_model as rm
    orig = (rm.D, rm.L, rm.H, rm.D_FF, rm.VOCAB, rm.SEQ)
    try:
        rm.D, rm.L, rm.H, rm.D_FF = 32, 1, 2, 64
        rm.VOCAB, rm.SEQ = 16, 8
        torch.manual_seed(0)
        mini = rm.GPT2Ref()
        n = sum(p.numel() for p in mini.parameters())
        print(f"\nMini model: {n:,} params (D=32 L=1 H=2 D_FF=64 V=16 T=8)")
        torch.manual_seed(0)
        idx_mini = torch.randint(0, rm.VOCAB, (1, rm.SEQ))
        return check(mini, idx_mini, sample=None, eps=1e-5, tag="MINI", per_tensor=8)
    finally:
        rm.D, rm.L, rm.H, rm.D_FF, rm.VOCAB, rm.SEQ = orig


def run_real():
    import ref_model as rm
    torch.manual_seed(rm.SEED)
    real = rm.GPT2Ref()
    rng = np.random.default_rng(rm.SEED)
    idx = torch.from_numpy(rng.integers(0, rm.VOCAB, size=(1, rm.SEQ), dtype=np.int64))
    return check(real, idx, sample=40, eps=1e-5, tag="REAL", per_tensor=3)


def main():
    print("="*70)
    print("NYANLIN Cave 2 — Gradient Check (float64)")
    print("="*70)
    r1, f1 = run_mini()
    r2, f2 = run_real()
    print("\n" + "="*70)
    if f1 == 0 and f2 == 0:
        print(f"✅ CAVE 2 PASSED (mini worst_ratio={r1:.2f}, real={r2:.2f})")
    else:
        print(f"❌ CAVE 2 FAILED (mini fail={f1}, real fail={f2})")
    print("="*70)


if __name__ == '__main__':
    main()
