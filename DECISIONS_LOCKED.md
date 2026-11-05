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
