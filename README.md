# MMC — Myanmar Machine Code

> မြန်မာစာ keyword တွေနဲ့ ရေးထားတဲ့ programming language —
> C ကို compile လုပ်တယ်။ **Self-hosting** (fixpoint verified)။

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

## 🙏 Credits

Built with love for the Myanmar developer community. 🇲🇲
