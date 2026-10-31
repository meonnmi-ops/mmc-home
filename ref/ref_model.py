"""
ref_model.py — NYANLIN Phase 1 PyTorch reference model
For MMC forward/backward verification (Cave 1 & 2)

Config (LOCKED):
  VOCAB=256 (byte-level)  SEQ=128
  D=256  L=6  H=8  D_FF=1024  SEED=42

Outputs (in --out dir):
  weights.npz     — all model weights (float32, named for MMC)
  input_ids.npy   — random test input (1, SEQ) int32
  ref_logits.npy  — forward logits (1, SEQ, VOCAB) float32
  ref_grads.npz   — gradients from causal LM loss
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import argparse

# ===== LOCKED CONFIG =====
VOCAB = 256
SEQ   = 128
D     = 256
L     = 6
H     = 8
D_FF  = 1024
SEED  = 42
DTYPE = torch.float32

# ===== MODEL =====
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1_w = nn.Parameter(torch.ones(D))
        self.ln_1_b = nn.Parameter(torch.zeros(D))
        self.qkv_w  = nn.Parameter(torch.empty(3*D, D))
        self.qkv_b  = nn.Parameter(torch.zeros(3*D))
        self.proj_w = nn.Parameter(torch.empty(D, D))
        self.proj_b = nn.Parameter(torch.zeros(D))
        self.ln_2_w = nn.Parameter(torch.ones(D))
        self.ln_2_b = nn.Parameter(torch.zeros(D))
        self.fc1_w  = nn.Parameter(torch.empty(D_FF, D))
        self.fc1_b  = nn.Parameter(torch.zeros(D_FF))
        self.fc2_w  = nn.Parameter(torch.empty(D, D_FF))
        self.fc2_b  = nn.Parameter(torch.zeros(D))
        for w in (self.qkv_w, self.proj_w, self.fc1_w, self.fc2_w):
            nn.init.normal_(w, std=0.02)

    def forward(self, x):
        B, T, _ = x.shape
        Dh = D // H

        # --- Attention (pre-LN, GPT-2 style) ---
        h = F.layer_norm(x, (D,), self.ln_1_w, self.ln_1_b)
        qkv = h @ self.qkv_w.T + self.qkv_b           # (B,T,3D)
        q, k, v = qkv.split(D, dim=-1)

        q = q.view(B, T, H, Dh).transpose(1, 2)       # (B,H,T,Dh)
        k = k.view(B, T, H, Dh).transpose(1, 2)
        v = v.view(B, T, H, Dh).transpose(1, 2)

        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1,1,T,T)
        att = (q @ k.transpose(-2, -1)) / (Dh ** 0.5)
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v                                   # (B,H,T,Dh)
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        x = x + (y @ self.proj_w.T + self.proj_b)

        # --- MLP ---
        h = F.layer_norm(x, (D,), self.ln_2_w, self.ln_2_b)
        h = h @ self.fc1_w.T + self.fc1_b
        h = F.gelu(h, approximate='tanh')             # GPT-2 GELU
        h = h @ self.fc2_w.T + self.fc2_b
        x = x + h
        return x


class GPT2Ref(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(VOCAB, D)
        self.wpe = nn.Embedding(SEQ, D)
        nn.init.normal_(self.wte.weight, std=0.02)
        nn.init.normal_(self.wpe.weight, std=0.02)
        self.blocks = nn.ModuleList([Block() for _ in range(L)])
        self.ln_f_w = nn.Parameter(torch.ones(D))
        self.ln_f_b = nn.Parameter(torch.zeros(D))

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for blk in self.blocks:
            x = blk(x)
        x = F.layer_norm(x, (D,), self.ln_f_w, self.ln_f_b)
        logits = x @ self.wte.weight.T                # tied head
        return logits


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def save_all(model, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. weights
    W = {}
    W['wte']    = model.wte.weight.detach().cpu().numpy()
    W['wpe']    = model.wpe.weight.detach().cpu().numpy()
    W['ln_f_w'] = model.ln_f_w.detach().cpu().numpy()
    W['ln_f_b'] = model.ln_f_b.detach().cpu().numpy()
    for i, b in enumerate(model.blocks):
        for n in ('ln_1_w','ln_1_b','qkv_w','qkv_b','proj_w','proj_b',
                  'ln_2_w','ln_2_b','fc1_w','fc1_b','fc2_w','fc2_b'):
            W[f'b{i}_{n}'] = getattr(b, n).detach().cpu().numpy()
    np.savez(out/'weights.npz', **W)
    print(f"weights: {len(W)} arrays, {sum(v.size for v in W.values()):,} values")

    # 2. input
    rng = np.random.default_rng(SEED)
    input_ids = rng.integers(0, VOCAB, size=(1, SEQ), dtype=np.int32)
    np.save(out/'input_ids.npy', input_ids)
    print(f"input_ids: shape={input_ids.shape} dtype={input_ids.dtype}")

    idx = torch.from_numpy(input_ids.astype(np.int64))

    # 3. forward
    model.eval()
    with torch.no_grad():
        logits = model(idx)
    np.save(out/'ref_logits.npy', logits.detach().cpu().numpy())
    print(f"ref_logits: shape={tuple(logits.shape)} "
          f"mean={logits.mean():.6f} std={logits.std():.6f}")

    # 4. backward (causal LM loss: predict next token)
    model.train()
    model.zero_grad()
    logits = model(idx)
    shift_logits  = logits[:, :-1, :].contiguous()
    shift_targets = idx[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, VOCAB),
        shift_targets.view(-1),
    )
    loss.backward()
    G = {n: p.grad.detach().cpu().numpy()
         for n, p in model.named_parameters() if p.grad is not None}
    np.savez(out/'ref_grads.npz', **G)
    print(f"ref_grads: {len(G)} arrays, loss={loss.item():.6f}")

    # sanity checks
    assert not np.isnan(logits.detach().cpu().numpy()).any(), "NaN in logits!"
    assert not np.isinf(logits.detach().cpu().numpy()).any(), "Inf in logits!"
    for n, g in G.items():
        assert np.isfinite(g).all(), f"non-finite grad: {n}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='./ref', help='output directory')
    ap.add_argument('--seed', type=int, default=SEED)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = GPT2Ref()
    n = count_params(model)
    print("=" * 60)
    print(f"NYANLIN Phase 1 Ref Model")
    print(f"  VOCAB={VOCAB} SEQ={SEQ} D={D} L={L} H={H} D_FF={D_FF}")
    print(f"  Params: {n:,}  ({n/1e6:.4f}M)")
    print("=" * 60)

    save_all(model, args.out)

    print("=" * 60)
    print(f"DONE. All outputs in: {Path(args.out).resolve()}")
    print("Verify MMC forward against ref_logits.npy (tol 1e-6)")


if __name__ == '__main__':
    main()
