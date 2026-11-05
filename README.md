# MMC — Myanmar Machine Code

> မြန်မာစာ keyword တွေနဲ့ ရေးထားတဲ့ programming language —
> C ကို compile လုပ်တယ်။ **Self-hosting** (fixpoint verified)။

> 🚀 **GPU ပေါ် ပြေးချင်ပါသလား?** — GPU op ၁၀ မျိုး စစ်ဆေးပြီး — အားလုံး GPU = CPU MATCH ✅ (GTX 1660 Ti: GELU 47.4x, LayerNorm 5.2x)

---

## 🏆 Self-Hosting Proof

**Reproducible fixpoint test** — the compiler compiles its own source,
and the output is bit-identical across compilation stages.

Stage 1 (gcc from C) → C1.c → gcc → Stage 2
Stage 2 (self)       → C2.c

C1.c == C2.c  ✅

**SHA-256 (identical):**

9876ee473982bc607c4eb05bead79b480494f3a8b5427bdbffdcc0e003d4cead

**Verify it yourself:**

    make verify

---

## ✨ Features

- 🇲🇲 **Burmese keywords** — တည်ဆောက်, လုပ်ငန်း, သုံးသပ်ချက်အရ, ဖော်ပြ
- 🔄 **Self-hosting** — compiler written in MMC
- 🎵 **Music-note bitwise operators** — ♩ ♪ ♫ ♬ ♭ ♯ ♮
- 💾 **Compiles to portable C** — runs anywhere C runs
- 📦 **Small runtime** — 48 KB C

---

## 🚀 Quick Start

Build the compiler:

    make

Compile an example:

    ./build/mmc examples/hello.mmc build/hello.c

Build and run:

    gcc build/hello.c runtime/runtime_v5.c runtime/runtime_glue.c \
        -I runtime -o build/hello -lm
    ./build/hello

**Output:**

    မင်္ဂလာပါ MMC v11!

---

## 📐 Language Reference

| Keyword | C Equivalent |
|---|---|
| တည်ဆောက် | let / declaration |
| လုပ်ငန်း | function |
| သုံးသပ်ချက်အရ | if |
| မဟုတ်လျှင် | else |
| ပြည့်အောင် | while |
| ထပ်ကာထပ်ကာ | for |
| ပြန်ပေး | return |
| ဖော်ပြ | print |
| ကိန်း | int |
| ဒသမ | double |
| စာသား | char* |
| စာရင်း | MMCArray* |
| အစုအဝေး | struct |

**Bitwise (music notes):**

| Symbol | Operation |
|---|---|
| ♩ | AND |
| ♪ | OR |
| ♫ | XOR |
| ♬ | NOT |
| ♭ | left shift |
| ♯ | right shift |
| ♮ | address-of |

---

## 🔧 Verify Self-Hosting

    ./scripts/verify_selfhost.sh

5 steps: build stage 1 → C1 → build stage 2 → C2 → assert C1 == C2.

---

## 🏗️ Architecture

    compiler_v13.mmc  (MMC source, 111 KB)
            ↓
       Lexer (MMC)
            ↓
       Parser (MMC)
            ↓
       Codegen → C (MMC)
            ↓
       C1.c  (70 KB)
            ↓ gcc + runtime_v5.c
       mmc_v1  (self-hosted compiler)

Runtime (48 KB C): string ops, arrays, file I/O, matrices.

---

## ⚠️ Honest Limitations

- ❌ Bootstrap requires C — Stage 0 needs gcc
- ❌ Runtime is C — not yet written in MMC
- ❌ No tensor/inference — AI engine not yet implemented
- ⚠️ Work in progress — expect bugs

We tell the truth. `--No Fake No Simulate--`

---

## 📜 License

MIT — see LICENSE file.

---


## 🚀 GPU လမ်းညွှန် (ကမ္ဘာလုံး = CUDA Kernel)

MMC ကနေ NVIDIA GPU ပေါ် တိုက်ရိုက် ပြေးနိုင်တယ် — No Fake, No Simulate — အောက်မှာ ပြထားတဲ့ ရလဒ်အားလုံးကို GTX 1660 Ti (sm_75) ပေါ်မှာ တကယ်ပြေး အတည်ပြုပြီးသား။

### ၁။ Kernel ရေးပုံ — ကမ္ဘာလုံး

    ကမ္ဘာလုံး လုပ်ငန်း gelu_kernel(ဒသမ* x, ဒသမ* y, ကိန်း n) void {
        တည်ဆောက် ကိန်း i = MMC_TID
        သုံးသပ်ချက်အရ(i < n) {
            တည်ဆောက် ဒသမ v = x[i]
            y[i] = 0.5 * v * (1.0 + tanh(0.7978845608 * (v + 0.044715 * v * v * v)))
        }
    }

- ကမ္ဘာလုံး လုပ်ငန်း → C code ထဲမှာ __global__ void ဖြစ်သွားတယ်
- MMC_TID = thread အညွှန်း (blockIdx.x × blockDim.x + threadIdx.x)
- MMC_BID = block အညွှန်း — row အလိုက် လုပ်ငန်း (LayerNorm) အတွက်

### ၂။ GPU runtime လုပ်ငန်းများ

| လုပ်ငန်း | အလုပ် |
|---|---|
| mmc_gpu_malloc(bytes) | GPU memory ချုပ် |
| mmc_gpu_copy_to(dev, host, bytes) | host → GPU ကူး |
| mmc_gpu_copy_from(host, dev, bytes) | GPU → host ကူး |
| mmc_gpu_launch2pi(kern, a, b, n, grid, block) | kernel ဖွင့် (pointer ၂ + int ၁) |
| mmc_gpu_launch_ln(kern, x, g, b, y, D, eps, M) | LayerNorm kernel ဖွင့် (M = row အရေအတွက်) |
| mmc_gpu_launch3p3i(kern, a, b, c, M, N, K, grid, block) | Matmul kernel ဖွင့် |
| cudaDeviceSynchronize() | GPU ပြီးအောင် စောင့် |
| mmc_now_ms() | အချိန်တိုင်း (ms) — benchmark အတွက် |

### ၃။ ပြေးပုံ — အဆင့် ၃ ဆင့်

    ./bin/mmc my_gpu.mmc my_gpu.c
    cp my_gpu.c my_gpu.cu
    nvcc -O2 -arch=sm_75 -I runtime my_gpu.cu runtime/runtime_v5.c runtime/runtime_glue.c runtime/runtime_gpu.cu -o my_gpu -lm
    ./my_gpu

လိုအပ်တာ — WSL ထဲမှာ CUDA toolkit (nvcc --version) နဲ့ NVIDIA driver (nvidia-smi) ရှိရမယ်။

### ၄။ စမ်းသပ်မှု ရလဒ်များ — GPU = CPU အားလုံး

| စမ်းသပ်မှု | အရွယ်အစား | ရလဒ် |
|---|---|---|
| GELU | ၈ တန်ဖိုး | GPU = CPU ✓ |
| LayerNorm | ၂ row × ၄ | GPU = CPU ✓ |
| Softmax | row လိုက် | GPU = CPU ✓ |
| Matmul | 64×64×64 | GPU = CPU ✓ |
| ReLU | 1,000,000 တန်ဖိုး | GPU = CPU ✓ |
| Transpose | 512x512 | GPU = CPU ✓ |
| Attention score | 64x64, d=16 | GPU = CPU ✓ |
| Embedding | 32 rows x V=100 x D=16 | GPU = CPU ✓ |
| AdamW | 65536 params x 3 steps | GPU = CPU ✓ |
| Cross-Entropy loss | 16 rows x V=32 | GPU = CPU ✓ |
| Backward linear (dW, dX) | R=32, K=16, N=32 | GPU = CPU ✓ |
| Backward chain (Y, P, dP, dW, dX) | R=8, K=16, V=32 | GPU = CPU ✓ |
| Training step (fwd + bwd + AdamW) | R=8, K=16, V=32 | GPU = CPU ✓ |
| Training loop (10 AdamW steps) | R=8, K=16, V=32 | GPU = CPU ✓ |
| Attention forward (S, A, O) | T=8 seq, d=16 | GPU = CPU ✓ |
| Attention + causal mask (autoregressive) | T=8 seq, d=16 | GPU = CPU ✓ |
| Attention backward (dQ, dK, dV, dS) | T=8 seq, d=16 | GPU = CPU ✓ |
| GELU Benchmark | ၁၀,၀၀၀,၀၀၀ | GPU 8.6ms vs CPU 406ms — 47.4x မြန် ✓ |
| LayerNorm Benchmark | 50,000×256 | GPU 43.8ms vs CPU 228ms — 5.2x မြန် ✓ |

စမ်းသပ်ဖိုင်အားလုံး — scripts/mmc_gpu/ ထဲမှာ (test ၁၇ ဖိုင် + bench ၂ ဖိုင်) — အားလုံး တစ်ခါပြေးရန် run_all.sh

## 🙏 Credits

Built with love for the Myanmar developer community. 🇲🇲

