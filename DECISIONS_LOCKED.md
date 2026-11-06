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
## 2026-11-05 — Attention backward (GPU=CPU)
- kernel အသစ် ၃ ခု — smax_bwd_row (dS = A·(dA − rowdot) — softmax backward), scale_row (×0.25 — locked att_row ရဲ့ scale နဲ့အတူ), tpose_rc (R×C non-square transpose — V^T အတွက်)
- locked ၅ ခု မူရင်းအတိုင်း ပြန်သုံး — att_row, mask_row, softmax_row, mm_k, tpose_row (square)
- gradient chain — dV = A^T·dO, dA = dO·V^T, dS = A·(dA−rowdot)·0.25, dQ = dS·K, dK = dS^T·Q
- dS အပေါ်တြိဂံ 28/28 = 0 အတိအကျ — causal mask က gradient ပိတ် (autoregressive ✓)
- row 0 ရဲ့ dS တွေ အားလုံး 0 — position 0 က ကိုယ့်ကိုယ်ကိုပဲ မြင်လို့ softmax gradient သုံ (သင်္ချာအရ မှန်)
- dQ (128), dK (128), dV (128), dS (64) အားလုံး GPU = CPU ✓ (1e-6)
- run_all.sh — test ၁၇ ဖိုင် အားလုံး PASS
## 2026-11-05 — Attention with learnable projections (GPU=CPU)
- Q = x·Wq, K = x·Wk, V = x·Wv (locked mm_k) — Wq Wk Wv က learnable weight (16×16 စီ)
- ပြီးမှ locked causal attention chain — att_row → mask_row → softmax_row → O = A·V
- kernel အသစ် ၀ — locked ၄ ခုရဲ့ သန့်သန့် ပေါင်းစပ် (composition)
- Q (128), K (128), V (128), S (64), A (64), O (128) အားလုံး GPU = CPU ✓ (1e-6)
- A အပေါ်တြိဂံ 28/28 = 0, row sum 1.0, O[0] = V[0] အတိအကျ — projection တွေကြားထဲမှာလည်း causal မှန်
- နောက် task — Wq Wk Wv ကို AdamW နဲ့ train (task-q — attention ၁ ဆင့် training)
- run_all.sh — test ၁၈ ဖိုင် အားလုံး PASS
## 2026-11-05 — Attention training step (GPU=CPU)
- Wq Wk Wv ကို AdamW နဲ့ ၁ ဆင့် train — forward + loss + backward + update + second forward အားလုံး GPU ပေါ်မှာ
- forward — Q = x·Wq, K = x·Wk, V = x·Wv → att_row → mask_row → softmax_row → O = A·V (locked အားလုံး)
- loss = 0.5·Σ(O−Y)² — dO = O−Y — kernel အသစ် mse_bwd ၁ ခုပဲ (MMC_TID elementwise)
- backward — dV = A^T·dO, dA = dO·V^T, dS = A·(dA−rowdot)·0.25, dQ = dS·K, dK = dS^T·Q (task-o chain)
- dWq = X^T·dQ, dWk = X^T·dK, dWv = X^T·dV — locked bwd_dw (task-k) ပြန်သုံး
- locked AdamW t=1 — Wq Wk Wv သုံးခုလုံး update, mv interleaved zero-init
- A (64), O (128), dO (128), dW ×3 (768), mv ×3 (1536), W_new ×3 (768), O2 (128) — စုစုပေါင်း 3520 တန်ဖိုး GPU = CPU ✓ (1e-6)
- loss 1.274930 → 1.210640 — LOSS DECREASED — attention တစ်ခုလုံးကို GPU ပေါ်မှာ train လို့ရပြီ
- dS အပေါ်တြိဂံ 28/28 = 0, dQ row 0 အားလုံး 0 — causal gradient block ဆက်တည်ရှိ
- run_all.sh — test ၁၉ ဖိုင် အားလုံး PASS
## 2026-11-06 — Attention training loop (GPU=CPU)
- task-q ရဲ့ ၁ ဆင့်ကို ၁၀ ဆင့် loop အဖြစ် တိုးချဲ့ — kernel အသစ် ၀ ခု၊ locked kernels အားလုံး ပြန်သုံး
- AdamW state (m v) GPU ပေါ်မှာ step တွေကြား တည်ရှိ — adamw_k recurrence m=0.9m+0.1g, v=0.999v+0.001g²
- tstep 1→10 — bias correction pow(0.9,t) pow(0.999,t) တစ်ဆင့်ချင်း ပြောင်း
- forward + backward + update တစ်ပတ်လုံး MMC loop ထဲမှာ — ၂၂ launch × ၁၀ ဆင့်
- loss တစ်ဆင့်ချင်း GPU = CPU ✓ (1e-6) — final Wq Wk Wv + m v + step-10 dO dW အားလုံး GPU = CPU ✓
- loss 1.274930 → 0.958722 — ၁၀ ဆင့်လုံး ဆင့်ချင်းကျ — step 1/2 တန်ဖိုးတွေ task-q နဲ့ အတိကျတူ
- bug fix — hOs copy မှာ byte ၁၂၈ (တန်ဖိုး ၁၆) ပဲကူးမိ — ၁၀၂၄ နဲ့ ပြင် — GPU compute က အစကတည်းက မှန်ခဲ့
- run_all.sh — test ၂၀ ဖိုင် အားလုံး PASS
## 2026-11-06 — Finite-difference gradient check (calculus verifies backward)
- task-s — backward chain ကို သင်္ချာ finite-difference နဲ့ စစ် — kernel အသစ် ၀ ခု၊ locked ၁၁ ခုတည်း
- T=4, K=8, d=8 — weight ၁၉၂ ခုလုံး (Wq ၆၄ + Wk ၆၄ + Wv ၆၄) — central difference eps 0.0001
- weight တစ်ခုချင်း ±0.0001 ခေါက် — forward ကို GPU ပေါ်မှာ ပြန်ပြေး — fd = (L+ − L−)/0.0002
- analytic chain (mse_bwd tpose mm_k smax_bwd_row scale_row bwd_dw) vs fd — tol 0.000001
- ၁၉၂/၁၉၂ အတိကျ — max err ~1e-12 (Wq 1e-12, Wk 1e-12, Wv 0) — restore ပြန်စစ် loss အတိတူ 0.325346
- FD check က GPU=CPU ထက် တစ်ဆင့် နက် — formula ကိုယ်တိုင်က သင်္ချာမှန်ကြောင်း သက်သေ (self-consistent မဟုတ်)
- negative control (sandbox) — scale_row ဖယ် → dWq/dWk 4× မှား → mismatch ၁၂၃ ခု ဖမ်းမိ — စစ်ဆေးချက်မှာ စွမ်းအားရှိ
- run_all.sh — test ၂၁ ဖိုင် အားလုံး PASS
## 2026-11-06 — Mini-batch training B=4 (gradient accumulation, one AdamW per step)
- task-t — sample ၄ ခု တစ်လောတည်း — per-sample gradient ကို GPU ပေါ်မှာ ထုတ် — host မှာ ပေါင်း — mean ×0.25 — AdamW ၁ ခေါက်တည်း
- kernel အသစ် ၀ — locked ၁၁ ခုတည်း — bwd_dw က overwrite ဖြစ်လို့ accumulation က host-side (loss sum လုပ်နည်းအတိုင်း)
- T=4, K=8, d=8 — sample ၄ ခု data မတူ ((i+3b)%7, (3i+b)%5) — weight တစ်စုတည်း Wq Wk Wv (၆၄ ခုစီ)
- step တိုင်း sample တိုင်း dWq dWk dWv GPU=CPU — ၇၆၈၀ တန်း (tol 0.000001) — per-sample loss လည်း GPU=CPU
- batch loss = ၄ ခု ပေါင်း ×0.25 — GPU=CPU ၁၀ ဆင့် — loss 0.321258 → 0.267066 တိကျတိကျ ကျဆင်း (min drop 0.0045)
- AdamW m v GPU ပေါ်မှာ persist — နောက်ဆုံး Wq Wk Wv m v အားလုံး GPU=CPU (tol 0.000001)
- sandbox: python replica ဂဏန်းတိတိကျ — negative control (scale_row ပိတ် → exit 236) — စစ်ဆေးချက်မှာ စွမ်းအားရှိ
- run_all.sh — test ၂၂ ဖိုင် အားလုံး PASS
## 2026-11-06 — Full Transformer block forward (GPU=CPU)
- embed(+PE) → QKV → causal attention → softmax → attn-out → residual → LN → W1 → GELU → W2 → residual → LN → logits → MSE loss — full block forward တစ်ခုလုံး ဆက်တိုက်
- kernel အသစ် ၀ — locked ၆ ခုရဲ့ သန့်သန့်ပေါင်းစပ် (mm_k att_row mask_row softmax_row gelu_kernel ln_fwd_row)
- dims — T=8 seq, K=8, d=8, H=32 (MLP hidden), V=11 vocab — embed table 11×8 + positional encoding + learnable Wq Wk Wv W1 W2 Wo
- GPU မှာ matmul/attention/softmax/LN/GELU အားလုံး — host မှာ embed+PE တပ်ဆင် + residual ၂ ခု + loss ပေါင်း (task-t အစဉ်အလာ)
- elementwise 984 (S A Att H1 F1 G F2 H3 logits) + loss + masked-zero 28 + rowsum 4 — စုစုပေါင်း 1017 စစ်ဆေးချက် GPU = CPU ✓ (1e-6)
- loss = 2.970168 — sandbox CPU stub, python replica (ဂဏန်းတိတိကျ), real GPU သုံးခုလုံး 10dp အတိကျတူ
- A အပေါ်တြိဂံ 28/28 = 0, row sum 1.000000000000 — LN ၂ ဆင့် + GELU ကြားထဲမှာလည်း causal မှန်
- negative control — mask_row ပစ်ရင် bad=855 → exit 87 — စစ်ဆေးချက် ဖမ်းမိတာ အတည်ပြု
- run_all.sh — test ၂၃ ဖိုင် အားလုံး PASS

## 2026-11-06 — MLP half-block training step (GPU=CPU)

- gelu_bwd kernel အသစ် ၁ ခု — dF1 = dG .* gelu'(F1) — tanh-approx derivative 0.5(1+t) + 0.5x(1-t*t)*0.7978845608*(1+3*0.044715*x*x), t = tanh(0.7978845608*(x+0.044715*x*x*x)) — launch3p3i, mse_bwd signature အတူ (pad ၂ ခု)
- MLP backward တစ်ခုလုံး GPU ပေါ်မှာ — dF2 = mse_bwd(F2-Y) → dG = dF2·W2^T (mm_k) → dF1 = gelu_bwd(dG, F1) → dW1 = H1^T·dF1, dW2 = G^T·dF2 (bwd_dw)
- gradient T=8 အပေါ် sum (mean မဟုတ်) — loss = 0.5*sum sq, တန်ဖိုး 64 (T·d)
- AdamW t=1 — W1 (256) W2 (256) နှစ်ခုလုံး — lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, bias correction, mv interleaved — locked adamw_k ပြန်သုံး
- second forward — W1n W2n နဲ့ F1b = H1·W1n → Gb = gelu_kernel → F2b = Gb·W2n — H1 (LN1 output) မပြောင်း
- LOSS DECREASED gate — loss2 < loss1 မဖြစ်ရင် bad++ — loss 0.534658 → 0.513591
- 3776 elementwise (10×256 + 3×64 + 2×512) + 2 loss GPU = CPU ✓ (1e-6) — GPU / sandbox stub / python replica ၃ ခုလုံး 10 ဂဏန်းအတိတူ
- negative control — gelu_bwd launch ပိတ်လိုက် → MISMATCH bad=1493, exit 213 — check က အလုပ်လုပ်တယ်
- run_all.sh — test ၂၄ ဖိုင် အားလုံး PASS

## 2026-11-06 — LayerNorm backward (GPU=CPU)
- ln_bwd_row kernel အသစ် ၁ ခု — dX က kernel ထဲမှာ — dgamma dbeta က CPU mirror + GPU finite-difference နှစ်ခါစစ်
- mean/var gradient အပြည့်အစုံ — dx = rstd*(dy*g - s1/D - xh*s2/D) — s1 = sum(dy*g) — s2 = sum(dy*g*xh) — kernel ထဲ mean/var/rstd ပြန်တွက် (forward အတိုင်း)
- dgamma[j] = sum_t(dy*xh) — dbeta[j] = sum_t(dy) — CPU mirror တွက် — GPU finite-difference နဲ့ တိုက်စစ်
- GPU = CPU ✓ 1e-6 — 208 elementwise — y 64 + dX 64 + FD dX 64 + FD dgamma 8 + FD dbeta 8 + restore gate 1 (lossr = loss0, 1e-9)
- finite-difference central diff h=0.0001 — dgamma/dbeta FD တစ်ဆယ်လုံးအတိတူ — dX fd=0.1625031119 vs 0.1625031334 (truncation 2.2e-8, gate 1e-6 အတွင်း)
- sample — dX0 0.1625031334 — dgamma0 5.8291892698 — dbeta0 -4.3069359836 — loss 32.022148
- sandbox negative control — ln_bwd_row ထဲက xh*s2/D term ဖြတ်လိုက် → dX0 gpu=-8.9189782549 — bad=128 (mirror 64 + FD 64) exit 128 — mean/var gradient term သီးသန့်သက်သေ — dgamma/dbeta FD မထိခိုင်
- T=8, D=8, eps 0.000001 — ln_fwd_row + ln_bwd_row locked — GPU test record task-r ၂၀ → s ၂၁ → t ၂၂ → u ၂၃ → v ၂၄ → w ၂၅
- run_all.sh — test ၂၅ ဖိုင် အားလုံး PASS

## 2026-11-06 — Full block backward (GPU=CPU)

- block_bwd_test.mmc — transformer block တစ်ခုလုံး (forward + backward + AdamW + second forward) GPU ပေါ်မှာ တစ်ဆို့က် — kernel အသစ် ၀ — locked kernels ၁၆ ခု စုစည်းသုံး
- backward chain — mse_bwd → ln_bwd_row (LN2) → gelu_bwd → ln_bwd_row (LN1) → smax_bwd_row + scale_row → dQ dK dV → bwd_dw (dWq dWk dWv dWo dW1 dW2)
- residual gradient — dHt = dHm + dH2g — dXt = dHt + dXq + dXk + dXv — CPU mirror နဲ့ GPU တူ
- AdamW ၆ ခု — Wq Wk Wv Wo W1 W2 (64/64/64/88/256/256) — m/v interleaved — bias correction — lr 0.01, beta1 0.9, beta2 0.999, eps 1e-8
- second forward (updated weights) — loss1 2.970168 → loss2 2.104753 — LOSS DECREASED gate
- စစ်ဆေးချက် — elementwise ၅၆၄၈ (forward ၉၈၄ + backward ၄၆၆၄) + loss ၂ + causal mask ၂၈ + rowsum ၄ — gate 1e-6 — bad=0, exit 0
- sample ၁၃ ခု — dX0_0 -1.2257127092, dWq0 0.0001309928, Wq_new -0.3099992367 — GPU = sandbox CPU stub = python replica ဂဏန်း ၁၀ လုံး တူ
- sandbox သက်သေ — generated C byte-identical (a487d778) — NC: loss2 scale 1.5 ပျက်စေရင် MISMATCH bad=1 exit 1 — fail-closed
- dims — T=8, K=8, d=8, H=32, V=11, eps 0.000001 — run_all.sh 26/26 PASS (glob ကြောင့် run_all ပြင်စရာမလို) — GPU test record w 25 → x 26
