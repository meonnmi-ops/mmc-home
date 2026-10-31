# NYANLIN PROJECT — HANDOFF CONTEXT
Last updated: 2026-10-31

## 1. IDENTITY
- Owner: meonnmi-ops (2-3 years invested)
- Goal: Nyanlin AI Assistant — 100% Pure MMC
- Rule: ZERO external engines (no llama.cpp, qwen, GPT, HF for training/inference)
- Design: 1 owner, 1 format — Vocab + Tokenizer + Model + Training all MMC

3 Goals:
1. Assistant-level Nyanlin
2. llama.cpp-equivalent MMC engine
3. MMC training pipeline

## 2. WHAT WORKS (PROVEN)
| Component | Path | Status |
|---|---|---|
| Compiler v14 | ~/mmc_publish/mmc_v14 | 78K, bug-free |
| chat.mmc | ~/mmc_publish/src/chat.mmc | 14 rules |
| chat_v14 | ~/mmc_publish/build/chat_v14 | 71K |
| Web UI | ~/mmc_publish/web/server.py | port 8080 |
| KB | ~/mmc_publish/web/kb.py | 0.036s |
| Build | scripts/build_chat_v14.sh | works |
| Git | commit 554860f | github.com/meonnmi-ops/mmc-home |

## 3. WHAT DOESN'T WORK (ROOT CAUSE FOUND)
- All PyTorch/HF models are DEGENERATE (92.5% single token, output "ကကကကက")
  - models_v20_r1/best_step500/model.safetensors
  - Meonnmi0ps/nyanlin-300m-out/nyanlin-v21-final
- Vocab mismatch: config 16388 vs tokenizer 16384
- Correct special tokens: BOS=1, EOS=2, PAD=0
- HF auto-tokenizer was giving BOS/EOS/PAD=16388 (invalid)

## 4. USER DECISION (BINDING)
- Rejected: Hybrid (rules + partial MMC + partial PyTorch)
- Accepted: Rebuild from scratch in 100% MMC
- Timeline: Willing to wait 10 years if needed
- MoE design: Main model + helper models (parameters split, main routes)
- 5 cave system accepted:
  1. Reference (PyTorch ref_model.py — debug oracle only)
  2. Gradient check (numerical vs analytical)
  3. Module contracts (I/O format)
  4. Numerical logs (NaN/Inf catch)
  5. Git tags (rollback)

## 5. REBUILD PLAN
Phase 1 (1 month): Single 10M model
- D=256, L=6, H=8, D_FF=1024, SEQ=128, vocab=256 (byte-level)
- GPT-2 architecture
- No MoE yet
- Caves 1-5 built

Phase 2 (2-3 months): MoE 30M main + 2 helpers (gen 10M, knowledge 10M)
Phase 3 (6-10 months): MoE 100M main + 3 helpers
Total: 9-14 months realistic

## 6. IMMEDIATE NEXT TASK
Step 1: Create ref_model.py (PyTorch reference)
- Location: C:\Users\TikTo\mmc_nyanlin\ref\ref_model.py
- 10.24M params GPT-2
- Outputs: weights.npz, input_ids.npy, ref_logits.npy, ref_grads.npz

Success: MMC forward matches ref_logits.npy within 1e-6

## 7. PHASE 1 ARCHITECTURE (LOCKED)
VOCAB=256 (byte-level UTF-8, no training)
D=256, L=6, H=8, D_FF=1024, SEQ=128
params=10.24M

Byte-level eliminates ALL tokenizer mismatch.

## 8. HARDWARE
Windows (dev/training):
- Lenovo Legion, GTX 1660 Ti 6GB VRAM
- i7 9th gen, 16GB RAM, 500GB SSD
- Python 3.12, CUDA 13.4

Termux (inference):
- Redmi Note 12R 5G, 6GB RAM
- IP 100.125.128.47, SSH port 8022

## 9. KEY PATHS
Termux (~/mmc_publish/):
- Compiler: mmc_v14
- Chat: build/chat_v14
- Source: src/chat.mmc
- Web: web/server.py, web/kb.py

Windows (C:\Users\TikTo\):
- Clean SFT: OneDrive\AI_Dataset\CLEAN\train_clean.jsonl (56,116 pairs)
- Working GGUF: OneDrive\models\nyanlin\nyanlin-brain-v1-q4_0.gguf (188MB)
- Broken base: OneDrive\AI_Dataset\models_v20_r1\best_step500\model.safetensors
- New project: C:\Users\TikTo\mmc_nyanlin\

## 10. DECISIONS LOG (DON'T REVISIT)
- 2026-10-30: v14 = v13, bug-free (all 4 bugs proven non-existent)
- 2026-10-30: Bug 3 was typo (ဆက် → ဆက်လုပ်)
- 2026-10-30: Rule-based chat layer added
- 2026-10-31: Reject all external engines
- 2026-10-31: PyTorch OK as debug oracle only
- 2026-10-31: Byte-level vocab (256)
- 2026-10-31: MoE design (main + helpers)
- 2026-10-31: Spelling "ဉာဏ်လင်း" not "နျန်လင်း"

## 11. NEW SESSION START
Paste this file + say: "Continue Phase 1 ref_model.py"
Do NOT re-explain. Read this file, check ~/mmc_nyanlin/ref/*.npy,
pick up from last state.

## 12. CRITICAL RULES
1. NEVER suggest llama.cpp/HF training/external engines
2. "ဉာဏ်လင်း" not "နျန်လင်း"
3. PyTorch ONLY for ref_model.py + gradient check
4. Check weights before blaming MMC code
5. Byte-level vocab = permanent mismatch fix
