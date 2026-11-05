# 2026-11-02 — MMC GPU + Hybrid Training (BINDING)

## Full GPU support
All operations: matmul, LN, GELU, softmax, AdamW, embedding, loss

## CPU+GPU Hybrid (3 modes)
- Data parallel (batch split)
- Model parallel (layer split)
- ZeRO-Offload (optimizer split)

## Target: 350M Assistant-level Nyanlin on GTX 1660 Ti 6GB
## Timeline: 8 weeks

## This is locked. No alternatives.

## 2026-11-02 — MMC Name & Identity (BINDING)

**MMC = Myanmar Code = Myanmar Programming Language**

- **MM** = Myanmar (မြန်မာလူမျိုး အများစု အသုံးပြုတဲ့ အတိုကောက်)
- **C** = Code / Programming Language
- **= မြန်မာတွေ နားလည်လွယ်အောင် တမင် ပေးထားတဲ့ နာမည်**

**Purpose:** မြန်မာစာနဲ့ မြန်မာတွေ အလွယ်တကူ နားလည်ပြီး မြန်မာ AI တည်ဆောက်နိုင်ဖို့

**This name is fixed. Do not rename.**

## 2026-11-02 — 100% MMC Standard (BINDING)

### User's definition of "100% MMC"
- Terminal မှာ ရိုက်တဲ့ command ကအစ
- ငါ ရေးတဲ့ code ကအစ
- run လုပ်တဲ့ output ကအစ
- **အားလုံး မြန်မာစာ MMC**

### Transition rule
- လမ်းသွားလမ်းလာ မအောင်မြင်ချိန်ထိ C/CUDA/Python = "တုတ်ချောင်း"
- လမ်းအောင်မြင်တဲ့ တစ်နေ့ — scaffolding အားလုံး ဖျက်
- ဘာကုဒ်မှ မရှိမပါစေရ

### 3 Phases
- Phase 1 (1 month): Prove pathway (GPU training + 350M) with scaffolding
- Phase 2 (2-3 months): Replace scaffolding (MMC --cuda, MMC tests)
- Phase 3 (6-12 months): Native MMC (shell, file ops, codegen)

### Current honest state
- Compiler: 100% MMC
- Training source: 100% MMC
- Build: C + Python (scaffold)
- GPU: CUDA C++ (scaffold)
- Tests: Python (scaffold)
- Shell: bash (scaffold)

### This is LOCKED. Do not forget.
## 2026-11-05 — Full GPU support: ပြီးစီး (GPU=CPU သက်သေအားလုံး)
Locked list (2026-11-02): matmul, LN, GELU, softmax, AdamW, embedding, loss — ၇ မျိုးလုံး ပြီး
- matmul 64x64x64 — GPU = CPU ✓
- LayerNorm — GPU = CPU ✓ (benchmark 50,000x256: 5.2x)
- GELU — GPU = CPU ✓ (benchmark 10,000,000: 47.4x)
- softmax — GPU = CPU ✓
- ReLU 1,000,000 တန်ဖိုး — GPU = CPU ✓
- Transpose 512x512 — GPU = CPU ✓
- Attention score 64x64, d=16 — GPU = CPU ✓
- Embedding 32 rows, V=100, D=16 — GPU = CPU ✓ (commit 7c6389c)
- AdamW 65536 params, 3 steps — GPU = CPU ✓ (commit 7c6389c)
- Cross-Entropy loss 16 rows, V=32 — GPU = CPU ✓ (commit 7c6389c)
Fixpoint binary sha256 မပြောင်း: bf78d0515e0c429083a03d16ea46a55b7d36f2e811455b9aa6bfae277df3fde8
တစ်ခါပြေး စစ်ဆေးရန် — scripts/mmc_gpu/run_all.sh
## 2026-11-05 — Backward pass အဆင့် ၁ (GPU=CPU)
- matmul 64x64x64 — GPU = CPU ✓ (test file archive ပြန်သိမ်း)
- softmax 16 rows, V=32 — GPU = CPU ✓ (test file archive ပြန်သိမ်း)
- Backward linear: dW = X^T dY, dX = dY W^T (R=32, K=16, N=32) — GPU = CPU ✓
- run_all.sh — test ၁၁ ဖိုင် အားလုံး PASS (ဒီ commit မတိုင်မီ စစ်ဆေးပြီး)
## 2026-11-05 — Backward chain အပြည့်အစုံ (GPU=CPU)
- တစ်ဖိုင်ထဲမှာ GPU kernel ၅ ခု — mm_k, softmax_row, ce_bwd (အသစ်: dP = P - onehot), bwd_dw, bwd_dx
- Forward: Y = X·W → P = softmax(Y); CE gradient dP = P - onehot(label)
- Backward: dW = X^T·dP, dX = dP·W (bwd_dx က W layout K×V နဲ့ကိုက်အောင် ချိန်) — R=8, K=16, V=32
- Y, P, dP, dW, dX အားလုံး GPU = CPU ✓
- run_all.sh — test ၁၂ ဖိုင် အားလုံး PASS
## 2026-11-05 — Training step ပထမဆုံး အပြည့်အစုံ (GPU=CPU + loss ကျ)
- Forward: Y = X·W → P = softmax(Y) → CE gradient dP = P - onehot → dW = X^T·dP
- Update: AdamW locked kernel (lr=0.01, beta1=0.9, beta2=0.999, t=1) — W, mv in-place update
- ဒုတိယ forward: Y2 = X·W_new → P2 — CE loss ကျဆင်းမှု စစ်ဆေး (loss0 → loss1 ကျ)
- Y, P, dP, dW, W_new, mv, Y2, P2 အားလုံး GPU = CPU ✓
- run_all.sh — test ၁၃ ဖိုင် အားလုံး PASS
## 2026-11-05 — Training loop ၁၀ အဆင့် (တစ်ဆင့်ချင်း GPU=CPU)
- task-k training step ကို loop အဖြစ် တိုး — step ၁ ကနေ ၁၀
- တစ်ဆင့်ချင်းစီမှာ P, dW, W_new, mv အားလုံး GPU = CPU ✓ (1e-6, 2304 တန်ဖိုး/step)
- AdamW t = step (1..10) — mv momentum သည် step တွေကြား ဆက်တည်ရှိ
- CE loss ကို တစ်ဆင့်ချင်း ပရင့် — ကျဆင်းမှု curve မြင်ရ + နောက်ဆုံး gate
- run_all.sh — test ၁၄ ဖိုင် အားလုံး PASS
## 2026-11-05 — Attention forward block (GPU=CPU)
- att_row (locked, scale 0.25 = 1/√16) → softmax_row → mm_k — kernel အသစ်မလို၊ ပေါင်းစပ်အသစ်
- S = (Q·K^T)·0.25, A = softmax row လိုက်, O = A·V — T=8 seq, d=16 head
- S (64), A (64), O (128) အားလုံး GPU = CPU ✓ (1e-6)
- A row sum = 1.0 — attention weight တွေက probability ဖြစ်နေတာ မြင်ရ
- causal mask မပါသေး — autoregressive LM အတွက် နောက် task
- run_all.sh — test ၁၅ ဖိုင် အားလုံး PASS
## 2026-11-05 — Causal mask attention (autoregressive path)
- mask_row kernel အသစ် — S ရဲ့ အပေါ်တြိဂံ (j > row) ကို -1e9 ထား — att_row (locked) ပြီးချင်း တိုက်ရိုက်ဆက်
- mask → softmax_row (locked) → O = A·V — masked score တွေက softmax ပြီးရင် အတိအကျ 0 ဖြစ်သွား
- A အပေါ်တြိဂံ 28/28 = 0 အတိအကျ — row sum တွေ အားလုံး 1.0 — position ရှေ့ကအရာ မမြင်ရ (causal ✓)
- သက်သေ — O[0] = -0.4 = V[0] အတိအကျ (position 0 က ကိုယ့်ကိုယ်ကိုပဲ မြင်လို့ ကိုယ့် V ပဲ ရ)
- S (64), A (64), O (128) အားလုံး GPU = CPU ✓ (1e-6)
- run_all.sh — test ၁၆ ဖိုင် အားလုံး PASS
