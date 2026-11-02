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

## 13. MMC COMPILER LIMITATIONS (Phase 1 findings)
- `မဟုတ်ရင်` (else) — NOT SUPPORTED by mmc_v14
  → workaround: `သုံးသပ်ချက်အရ(cond) { ... }` ပြီးမှ
    နောက်တစ်ခု `သုံးသပ်ချက်အရ(negated_cond) { ... }` သုံး
- `runtime_mmc.mmc` (3228 lines) မှာ `မဟုတ်ရင်` လုံးဝ မသုံးထားဘူး → ဒါက အထောက်အထား
- `;` semicolon — loop body ထဲမှာ `{ a; b }` မရ။ တစ်လိုင်းချင်း ရေး
- `ဒသမ` = C `double` (float64), `ကိန်း` = int

## 14. PHASE 1 MMC FORWARD — PASS (2026-10-31)

### Achievement
MMC forward matches PyTorch float64 reference to **2.6e-12**
(Target was 1e-6 — exceeded by 4 orders of magnitude)

### Stage-by-stage verification
| Stage  | max_diff   | Notes                          |
|--------|------------|--------------------------------|
| emb    | 0.0        | exact                          |
| ln1    | 2.2e-15    | machine epsilon                |
| qkv    | 1.7e-15    | machine epsilon                |
| attn   | 6.1e-16    | machine epsilon                |
| gelu   | 9.1e-13    | tanh approx diff               |
| logits | 2.6e-12    | PASS                           |

### Bug journey (2 bugs found + fixed)
1. **`မဟုတ်ရင်` (else)** — MMC compiler NOT supported
   → workaround: two separate `သုံးသပ်ချက်အရ` with negated condition
2. **`CFG_EPS` global** — MMC wraps main as auto-main; objcopy rename
   skips global initializers → CFG_EPS = 0 instead of 1e-5
   → workaround: use literal 0.00001 inside ln_row call

### Also fixed
- `malloc(4 * rows * cols)` → `malloc(8 * rows * cols)` (double = 8 bytes)
- `fread/fwrite` size 4 → 8 (float64 binary files)
- `tn_copy` missing from model_runtime_lib.mmc → added in build/stubs.c

### Files
- `src/model_runtime.mmc`       — full runtime (lib + main)
- `src/model_runtime_lib.mmc`   — runtime lib only (no main)
- `src/model_forward.mmc`       — forward + load + save
- `build/model_runtime_lib.h`   — C header for forward.c
- `build/stubs.c`               — tn_copy + mmc_array_* stubs
- `build/model_forward`         — final binary (21K)

### Build recipe
```bash
./mmc_v14 src/model_runtime_lib.mmc build/model_runtime_lib.c
gcc -O2 -c -I build -I runtime build/model_runtime_lib.c -o build/model_runtime_lib.o
objcopy --redefine-sym main=mmc_runtime_auto_main build/model_runtime_lib.o
./mmc_v14 src/model_forward.mmc build/model_forward.c
sed -i '1i #include "model_runtime_lib.h"' build/model_forward.c
gcc -O2 -c -I build -I runtime build/model_forward.c -o build/model_forward.o
gcc -O2 build/model_forward.o build/model_runtime_lib.o build/stubs.o -o build/model_forward -lm
./build/model_forward
Next: Phase 1 Training
Target: 10M (or keep 4.84M) model with loss < 2.0 on byte-level data

Use existing SFT dataset (56,116 pairs, byte-level)

## 21. TRAINING PIPELINE VERIFIED (2026-10-31 → 11-01)

### Numerical accuracy — MMC vs PyTorch float64

| Component | Match | Tolerance |
|-----------|-------|-----------|
| Forward (6-block, 4.84M) | 2.6e-12 | 1e-6 ✓ |
| LN bwd | 8.9e-16 | 1e-10 ✓ |
| GELU bwd | 2.6e-12 | 1e-10 ✓ |
| matmul_t bwd | 0.0 exact | 1e-10 ✓ |
| CE + softmax bwd | 6.9e-18 | 1e-10 ✓ |
| Attention bwd | 1.7e-14 | 1e-10 ✓ |
| Full block bwd | 2.2e-11 | 1e-10 ✓ |
| Tied head + emb + LN_f | 2.2e-16 | 1e-10 ✓ |
| AdamW step (1 block) | 8.4e-14 | 1e-10 ✓ |
| L=2 multi-block | 3.47e-12 | 1e-10 ✓ |
| D=20→100 scale | 1e-10 to 4e-9 rel | 1e-8 rel ✓ |

### Functional proofs

- **Training loop**: loss 1.66 → 0.000000 on synthetic pattern (200 steps)
- **Data loader**: byte corpus reads + LCG shuffle → exact batch match with Python
- **Multi-block chain**: L=2 forward+backward+AdamW matches PyTorch
- **Scale test**: D=20,40,60,80,100 all PASS (relative tol 1e-8)

### Interpretation

**MMC is functionally equivalent to PyTorch for training** — differences
are pure float64 roundoff. Real training weight updates (~lr=1e-3) are
10^6× larger than numerical noise (~1e-9).

## 22. TRAINING FILES (2026-11-01)

### Core MMC source
- `src/block_lib.mmc` — verified helpers (LN, GELU, matmul, attention,
  block_forward, block_fwd_bwd). Compiles to block_lib.o (24K).
- `src/train_step_test.mmc` — 1-block AdamW step (reference: train_step)
- `src/train_multi_step.mmc` — L-block AdamW step (parameterized D, DFF, H)
- `src/data_loader_test.mmc` — byte corpus → batch with LCG shuffle
- `src/model_runtime_lib.mmc` — tensor + matmul + LN + attn forward
- `src/model_forward.mmc` — full 6-block forward, saves logits

### Python refs (oracle)
- `ref/train_test_gen.py` — 1-block AdamW ref
- `ref/train_multi_gen.py` — L-block AdamW ref
- `ref/train_scale_gen.py` — parameterized by env (D, DFF, H)
- `ref/run_scale.py` — orchestrator (D=20→100)
- `ref/data_loader_test.py` — byte-level batch ref
- `ref/ln_bwd_test.py`, `gelu_bwd_test.py`, `matmul_bwd_test.py`,
  `ce_bwd_test.py`, `attn_bwd_test.py`, `block_bwd_test.py`,
  `tied_bwd_test.py` — per-op refs

### Build recipe (block_lib → .o)
```bash
./mmc_v14 src/block_lib.mmc build/block_lib.c
gcc -O2 -I runtime -c build/block_lib.c -o build/block_lib.o
objcopy --redefine-sym main=mmc_block_lib_auto_main build/block_lib.o
cd ~/mmc_publish
cat >> docs/NYANLIN_CONTEXT.md << 'EOF'

## 23. MMC COMPILER LIMITS — FULL LIST (2026-11-01)

### Confirmed unsupported

1. မဟုတ်ရင် (else) — NOT SUPPORTED.
   Workaround: two separate သုံးသပ်ချက်အရ blocks.

2. Semicolon — NOT allowed in loop bodies.
   One statement per line.

3. Scientific notation — 1e30 parses as identifier e30 and errors.
   Use 1000000000.0 (plain decimal).

4. Loop var scoping (C-style) — တည်ဆောက် ကိန်း j = 0 inside a block
   is scoped to that block only.
   Fix: hoist all loop vars to function top.

5. Multi-line extern "C" declaration — must be a single line.

6. ဒသမ* in extern "C" params — NOT supported.
   Use အစု (void*) instead.

7. MMC-generated main — always emitted as auto-main (does argv wrapping
   via mmc_array_*). When using MMC files as a library:
   objcopy --redefine-sym main=mmc_X_auto_main build/X.o
   BUT this loses global variable initializers (e.g., CFG_EPS = 1e-5
   would stay 0). Workaround: use literals inside functions.

### Confirmed supported

- Type declarations: ကိန်း / ဒသမ / အစု / စာသား / char*
- Single-line extern "C" prototypes
- Structs via အစုအဝေး with pointer params and -> access
- သုံးသပ်ချက်အရ(cond) { ... } — if
- ပြည့်အောင်(cond) { ... } — while
- ပြန်ပေး — return (works inside if blocks)
- Function calls with အစု parameters (void*)
- sizeof, malloc, calloc, free
- Math: sqrt, exp, log, tanh (via extern "C" to libm)

### Byte array pitfall

A buffer read via fread(buf, 1, n, f) MUST be declared as စာသား
(char*) not ကိန်း* (int*). Otherwise int* arithmetic walks 4 bytes
per element and reads out of bounds.
Fix: စာသား buf = read_bytes(...).

## 24. BUG HISTORY — 2026-10-31 / 11-01

### Forward path
1. `mmc_matrix_new` allocated 4 bytes/element but `ဒသမ` = 8 bytes
   (double) → segfault. **Fix**: `malloc(8 * rows * cols)`.
2. Binary I/O wrote/read 4-byte floats but ref is double → reshape errors.
   **Fix**: all `.bin` are float64, `fread`/`fwrite` size 8.
3. `tn_copy` was missing from model_runtime_lib.mmc → undefined symbol.
   **Fix**: added to `build/stubs.c`.

### Backward path
4. `mm_bwd` wrote `dX = dy*W` inside the j-loop instead of accumulating
   `dX += dy*W` → last j overwrote all. **Fix**: `+=` only.

### AdamW path
5. **Near-threshold eps amplification**: when gradient ≈ eps (1e-8),
   `sqrt(vh)+eps` diverges slightly between PyTorch and MMC due to
   accumulated float error. Produces ~1e-9 absolute diff on bqkv/Wqkv
   only. **Not a bug** — expected AdamW behavior. Use relative
   tolerance 1e-8 in tests (per-tensor).

### Tooling
6. `/tmp` is NOT writable in Termux → use `$HOME/tmp`.
7. `bytes.tofile()` doesn't exist in numpy → `np.frombuffer(...).tofile()`.
8. `b"..."` with non-ASCII → use `"text".encode('utf-8')`.
9. `sed -i 's|...` with `!` (in `!build/*.h`) triggers bash history
   expansion → use single quotes or escape.
10. `.gitignore` with `!build/*.h` needs `git add -f` for first add.

## 25. NEXT: REAL MODEL TRAINING (Phase 1 final)

### Target config
- VOCAB=256 (byte-level), SEQ=128
- D=256, L=6, H=8, DFF=1024
- Params: 4,837,376 (4.84M)
- Optimizer: AdamW (lr=1e-3 to 3e-4, wd=0.01, betas=0.9/0.999)

### Files to build
1. `src/train_full.mmc` — D=256, L=6, V=256, SEQ=128, flat weight
   arrays (L× per-block layout, same as train_multi_step.mmc).
2. Data prep (Windows → Termux): SFT `train_clean.jsonl` → byte stream.
3. `ref/run_full_train.py` — orchestrator with checkpointing every N
   steps, loss log, gradient norm log, and warmup+cosine LR schedule.

### Estimated runtime on Termux CPU
- ~500 ms/step (4.84M model, 128 seq)
- 1 epoch (14K steps @ batch=4) ≈ 2 hours
- 5 epochs ≈ 10 hours (loss should drop below 2.0)

### Realistic targets
- Loss 5.5 (init) → 2.0–2.5 after 1 epoch on byte-level SFT
- Loss → 1.5 after 5 epochs
- Assistant-level fluency: **needs ~100M params + GPU** (Phase 2/3)


## 26. SESSION UPDATE — 2026-11-01

### Training completed (partial, then migrated)

- Phone Termux: 4.84M model, steps 1→2336, loss 5.73→0.86 (43.5% drop)
- Note8 Termux: continued steps 2336→~2527 (2.4s/step)
- WSL Debian: continued steps 2336→~2527 (2.4s/step, verified path fix)
- Reason for migration: local WSL faster (10x), no device heat
- Training process STOPPED for CPU optimization work

### Current file locations (after migration)

WSL Debian (~/mmc_wsl/mmc_publish/):
- source: src/*.mmc (from git clone + phone transfer)
- binaries: build/train_full (90K), build/generate (31K)
- weights: ref/full_test/*.bin (16 tensors + m_/v_ AdamW state, ~50MB)
- corpus: ref/full_test/corpus.bin (299MB from GOLD/MEGA_GOLDEN_CLEAN.jsonl, 234,133 docs)
- next_step.txt: 2336 (trained checkpoint)
- SFT: ~/train_clean.jsonl (30MB, 56,116 pairs)

Phone Termux (~/mmc_publish/):
- mmc_v14 (78K ARM64)
- src/*.mmc (source of truth for MMC code)
- build/train_full.c, build/block_lib.c, build/generate.c (generated C, 6.7KB tarball)

Note8 Termux (~/mmc_publish/):
- same as phone (weights backup)
- STOPPED, freed for other use

Windows C:\Users\TikTo\:
- OneDrive\AI_Dataset\CLEAN\train_clean.jsonl (29.7MB)
- OneDrive\AI_Dataset\GOLD\MEGA_GOLDEN_CLEAN.jsonl (used for corpus)
- mmc_nyanlin\ (4KB only, leftover)

### WSL environment

- User: meonnmi-0ps
- Tailscale IP: 100.85.155.94
- SSH server: enabled, port 22
- Disk: 1007GB total, 857GB free
- CPU: 8 cores
- Memory: 16GB RAM (assumed)

### Device inventory

| Device | IP (Tailscale) | Purpose |
|---|---|---|
| Windows Laptop | 100.70.181.124 | Host (GTX 1660 Ti 6GB) |
| WSL Debian | 100.85.155.94 | Main dev |
| Galaxy Note 8 | 100.108.181.26 | Stopped (was training) |
| Redmi Note 12R | 100.125.128.47 | Current phone (idle) |

### User data (WSL home, ~100GB)

| Folder | Size | Content |
|---|---|---|
| myanmar_ai | 16G | Myanmar AI data |
| nyanlin-ai-dataset | 14G | Nyanlin dataset |
| f5_env | 11G | F5 TTS env |
| GOLD_VAULT | 9.4G | GOLD data |
| DATA_RAW | 8.2G | Raw data |
| AI_Dataset | 7.5G | AI dataset |
| f5_new | 6.8G | F5 TTS |
| nyanlin_env | 5.9G | Env |
| nyanlin_pipeline | 2.1G | Pipeline |
| nyanlin_data | 1.2G | Data |
| CORPUS_STAGE | 348M | Staged |

**User has ~100GB data. Do NOT call it "old data". It is precious.**


## 27. DECISIONS LOCKED (2026-11-01) — DO NOT REVISIT

### Model target (user decision)

- **350M OR 500M** (NOT 1B — hardware insufficient)
- **MoE architecture**: 350M main + Phase 1/2/3 helpers
- **Helpers**: 4.84M (Phase 1), 30M (Phase 2), 100M (Phase 3)
- **Routing**: Rule-based (chat.mmc), later learned router
- **Replay**: Simple (main outputs helper's answer), later reformat

### Training approach (user decision)

- **100% Pure MMC — ZERO external engines**
- **NO** QLoRA, NO distillation, NO external models
- **YES** math libraries as primitives (libm, OpenBLAS, cuBLAS if allowed)
- **YES** CPU optimization (AVX2, OpenMP, blocked matmul, FP32)
- **YES** Kaggle/Colab for compute (device access, not engine)

### Path (user decision, LOCKED)

**Phase A — CPU Optimize (START NOW)**
1. FP32 (double→float): 2x
2. Blocked matmul: 2-3x
3. AVX2 intrinsics: 5x
4. OpenMP 8 threads: 6-7x
5. Batch=4: 2-3x
Total: ~100-160x speedup, 2 weeks setup

**Phase B — Train 350M (after CPU optimize)**
- WSL CPU, 1.5-3 days (batch=4 maybe more memory)
- Data: GOLD + train_clean + AI_Dataset subset (~1GB)

**Phase C — GPU Inference (after 350M trained)**
- 1660 Ti 6GB cannot train 350M (needs ~20GB)
- But CAN do 350M inference in Q4 (~175MB VRAM)
- Use 1660 Ti for inference only

### Hardware reality (LOCKED understanding)

| Model | Train on 1660 Ti 6GB | Train on WSL CPU | Inference on 1660 Ti |
|---|---|---|---|
| 4.84M | ✅ | ✅ | ✅ |
| 30M | ✅ (tight) | ✅ | ✅ |
| 100M | ❌ (7GB+) | ✅ | ✅ |
| 350M | ❌ (20GB+) | ✅ | ✅ (Q4) |
| 500M | ❌ | ✅ (slow) | ✅ (Q4) |

**= 350M training MUST be CPU. GPU only for inference at this stage.**

### Speedup estimates (LOCKED)

- Current MMC: 2.4s/step (4.84M)
- After FP32: 1.2s
- After Blocked: 0.4-0.6s
- After AVX2: 0.08-0.12s
- After OpenMP: 0.015-0.02s
- **350M after all: 1-1.5s/step → 100K steps = 1.5-3 days**


## 28. CPU OPTIMIZE PLAN — LOCKED

### 5 steps, in order

1. FP32 (double→float): 2x speedup, 10 min
2. Blocked matmul: 2-3x, 30 min
3. AVX2 intrinsics: 5x, 1 hour
4. OpenMP 8 threads: 6-7x, 30 min
5. Batch=4: 2-3x, 2 hours

Total: ~100-160x speedup, ~2 hours setup

### Speedup estimates
- 4.84M: 2.4s → 0.015s/step
- 30M: 15s → 0.15s/step
- 350M: 172s → 1-1.5s/step
- 350M 100K steps: 200 days → 1.5-3 days

### Build pipeline

## 29. USER PREFERENCES

- No "take a break" / "nap" / "rest" suggestions
- No dismissing user data as "old" (100GB+, 2+ years)
- No jumping to conclusions without checking
- No "can't do" without explaining root cause
- Explain simply, patiently (no code background)
- Respect 13,000 baht GTX 1660 Ti purchase
- Break long responses into parts
- All data belongs to user (meonnmi-ops)


## 30. TECHNICAL REFERENCE

### MMC compiler limits
- No `မဟုတ်ရင်` (else) — use two `သုံးသပ်ချက်အရ`
- No semicolon in loops
- No scientific notation (1e30 → e30 error)
- Hoist loop vars to function top
- extern "C" single line only
- `ဒသမ*` not allowed in extern "C" — use `အစု` (void*)
- Byte buffers use `စာသား` (char*) not `ကိန်း*` (int*)

### MMC rule (BINDING)
- ZERO external engines
- PyTorch = debug oracle only
- libm OK, OpenBLAS/cuBLAS need approval
- AVX2/OpenMP OK (CPU/OS primitives)

### Numerical verification (2026-10-31)
- Forward 4.84M: 2.6e-12
- 7 backward ops: all PASS
- AdamW step: 8.4e-14
- Data loader: exact match
- L=2 multi-block: 3.47e-12
- D=20→100 scale: all PASS

### Training results
- Demo (55 unique bytes): 5.67→0.09 (memorize)
- Real SFT (196 bytes): 1.53→0.86 (real learning)
- GOLD 234K docs: current, expected 1.2-1.5


## 26.1. SESSION UPDATE — 2026-11-02 (CPU optimize phase)

### Completed this session
- FP32+Blocked+AVX2+OpenMP: 2.44s → 0.88s (2.77x)
- AdamW OMP: bit-exact (loss 0.9436 unchanged)
- Multi-step N=100: 0.31s/step (7.87x total)
- I/O elimination: sys 0.528s → 0.022s (24x reduction)
- lr bug fixed (byte-level: 8-byte double read as 4-byte float32)
- Real training verified: loss 1.0 → 0.84 (100 steps)

### Batch=4 finding (LOCKED)
- 14% SLOWER per-sample (not faster — matmul now only 3-5% of runtime)
- 6x tighter gnorm (quality benefit)
- 4x activation memory cost
- Decision: batch=1 for 350M training (memory + speed constraints)

### Next session
- Option A: LN/GELU/Softmax AVX2 vectorize (1.5x expected)
- Option B: Move to 350M training with batch=1
- Option C: Verify optimizer state safety before long training run

### Files
- build/train_full_multi_fp32_omp_avx2 (working, N=100)
- build/train_b4 (batch=4 variant, quality-tested)
- scripts/gen_multi_batches.py, make_multi_c.py, make_batch4.py
- Backup: ref/full_test_before_n100_*, ref/full_test_after_b4_*

## 28. SESSION UPDATE — 2026-11-02 (GPU Backend Complete)

### Environment (WSL2, Debian 13, GTX 1660 Ti)
- CUDA 12.8 (V12.8.93) at /usr/local/cuda-12.8
- gcc-13 (CUDA 12.6 incompatible with gcc-14)
- Swap: 9 GB (2 + 8)
- **GLIBC 2.41 fix REQUIRED:**
  - Header: ~/cuda_compat/glibc_c23_math_compat.h
  - Env: NVCC_PREPEND_FLAGS="-Xcompiler=-include,$HOME/cuda_compat/glibc_c23_math_compat.h"

### GPU Ops COMPLETE (All PASS)
- cuBLAS matmul: 1670 GFLOPS (30% of peak, small matrix)
- LayerNorm fwd+bwd: ALL PASS (< 1.5e-06)
- GELU fwd+bwd: ALL PASS (< 1.2e-07)
- Softmax+CE fwd+bwd: ALL PASS (< 1e-06)
- AdamW: ALL PASS (< 1.5e-08)

### Key Files
- scripts/cuda/test_mm.cu (naive)
- scripts/cuda/test_mm_tiled.cu (tiled)
- scripts/cuda/test_mm_cublas.cu (cuBLAS — USE THIS for matmul)
- scripts/cuda/test_ln.cu (LayerNorm)
- scripts/cuda/test_ops.cu (GELU+softmax+AdamW)

### Decision LOCKED
- Matmul: cuBLAS (hardware primitive, NOT external engine)
- All custom kernels: elementwise ops only
- Tiled CUDA: 514 GFLOPS (only use if philosophy demands)
- Naive CUDA: 336 GFLOPS (reference only)

### Next Session Start
"Continue GPU backend — full training step integration"

### User Rules (CRITICAL — DO NOT IGNORE)
- No "take a break" / "rest" / "nap" suggestions
- No fake optimism or sugar-coating
- Verify claims before accepting (example: pip install myanmar-code was real)
- 8 months documented work (2026-02-22 PyPI upload → now)
- Be honest about estimates (100x was wrong, 7.87x was real)
- Verify patterns before sed/regex scripts
- No unnecessary verification commands for trivial tasks
- Backup: ~/backup_cuda_ready_*.tar.gz
- Weights: ref/full_test/*.bin (DO NOT promote out_* without explicit request)

### Timeline Truth
- 2026-02-22: myanmar-code v1.0.0 (PyPI, first evidence)
- 2026-10-15: first .mmc file
- 2026-10-26: Git repo (MMC v11)
- 2026-11-02: GPU ops complete

### Cost Reality (honest)
- 4.84M CPU: 0.31s/step (7.87x from FP64)
- 4.84M GPU: ~0.02-0.05s/step estimated (5-8x from CPU)
- 100M CPU: ~7.5 days for 100K steps
- 100M GPU: ~1.5-2 days for 100K steps
- 350M GPU + model parallel: ~3-5 days

### Backup State
- ref/full_test_before_n100_*: pre-N100 backup
- ref/full_test_after_b4_*: post-batch=4 backup
- ~/backup_cuda_ready_*.tar.gz: CUDA-ready snapshot

## 29. GPU BACKEND PROGRESS — 2026-11-02 (Session 29)

### USER RULES (CRITICAL — မမေ့ရ)
- No "take a break" / "rest" / "nap" / "ဒီနေ့ ရပ်" suggestions — USER STOPS, not assistant
- No fake optimism or sugar-coating
- Verify claims before accepting
- Be honest about estimates (100x was wrong, 7.87x real)
- Say "Context ပြည့်တော့မယ်" ONLY when actually near limit
- Do NOT stop user mid-work
- Write complete context in ONE block (no fragmentation — markdown fences break paste)

### 100% MMC STANDARD (LOCKED — DECISIONS_LOCKED.md)
- Current: C/CUDA/Python = scaffolding (allowed during transition)
- Success day: ALL scaffolding removed
- Final: Terminal commands → output = 100% Myanmar MMC
- Phase 1 (1 month): Prove pathway (GPU training + 350M) with scaffolding
- Phase 2 (2-3 months): Replace scaffolding (MMC --cuda, MMC tests)
- Phase 3 (6-12 months): Native MMC (shell, file ops, codegen)

### ENVIRONMENT (WSL2, Debian 13, GTX 1660 Ti)
- CUDA 12.8 (V12.8.93) at /usr/local/cuda-12.8
- gcc-13 (CUDA 12.6 incompatible with gcc-14 in Debian 13)
- Swap: 9 GB (2 + 8)
- GLIBC 2.41 fix (CRITICAL):
  - Header: ~/cuda_compat/glibc_c23_math_compat.h
  - Env: NVCC_PREPEND_FLAGS="-Xcompiler=-include,$HOME/cuda_compat/glibc_c23_math_compat.h"
  - Verify: nvcc -O3 test.cu -o test && ./test (no extra flags needed)

### GPU OPS — ALL PASS (verified 2026-11-02)
- cuBLAS matmul: 4.77e-07 correctness, 1670 GFLOPS (30% peak)
- LayerNorm fwd+bwd: 1.4e-06, 83+21 us
- GELU fwd+bwd: 1.2e-07, 30+41 us
- Softmax+CE fwd+bwd: 1e-06, 32+44 us
- AdamW: 1.5e-08, 13 us
- Attention fwd: 7.6e-08, 1.61 ms (single-thread, optimize later)

### KEY FILES
- scripts/cuda/test_mm.cu (naive — reference only)
- scripts/cuda/test_mm_tiled.cu (514 GFLOPS)
- scripts/cuda/test_mm_cublas.cu (1670 GFLOPS — USE THIS)
- scripts/cuda/test_ln.cu (LayerNorm)
- scripts/cuda/test_ops.cu (GELU+softmax+AdamW)
- scripts/cuda/test_attn.cu (attention fwd, PASS 7.6e-08)
- scripts/cuda/test_*.py (Python verifiers — scaffolding)

### BUILD COMMANDS
- All CUDA: nvcc -O3 -arch=sm_75 scripts/cuda/test_XXX.cu -o /tmp/test_XXX
- cuBLAS: nvcc -O3 -arch=sm_75 scripts/cuda/test_mm_cublas.cu -o /tmp/test_mm_cublas -lcublas
- NVCC_PREPEND_FLAGS auto-applied from ~/.bashrc

### ATTENTION BACKWARD (NEXT — not yet done)
Formulas:
- dV = softmax^T @ dO
- dS_pre = dO @ V^T
- row_dot = sum_k dS_pre[k] * softmax[k]
- dS = softmax * (dS_pre - row_dot) * scale
- dQ = dS @ K
- dK = dS^T @ Q
After attention bwd: residual add + full GPU training step integration

### BACKUP STATE
- ~/backup_cuda_ready_20261102_161757.tar.gz (175K)
- ref/full_test_before_n100_* (pre-N100)
- ref/full_test_after_b4_* (post-batch=4)
- Git HEAD: 362652b (attention fwd pending commit)

### HONEST TIMELINE (8 months documented)
- 2026-02-22: myanmar-code v1.0.0 (PyPI, first evidence)
- 2026-10-15: first .mmc file
- 2026-10-26: Git repo (MMC v11)
- 2026-11-02: GPU ops complete (all PASS)

### COST REALITY (honest)
- 4.84M CPU: 0.31s/step (7.87x from FP64)
- 4.84M GPU: ~0.02-0.05s/step (estimate, full integration pending)
- 100M CPU: ~7.5 days for 100K steps
- 100M GPU: ~1.5-2 days (estimate)
- 350M GPU + model parallel: ~3-5 days (estimate)

### NEXT SESSION START
Continue GPU backend — attention backward
Context: All GPU ops done except attention bwd. Then integration. Then Phase 1 scale-up.


## 30. FULL 4.84M MODEL GPU FWD+BWD — COMPLETE (2026-11-02)

### ALL 12 GRADIENTS PASS (< 4e-05 vs NumPy)
- Forward: y (6.9e-06), probs (1.1e-07), loss (7.2e-07)
- Backward: dx, dwte, dwpe, dlnfg, dlnfb, dwq, dwk, dwv, dwo

### KEY FILES (scripts/cuda/)
- block_ops.cu + .h (device API)
- test_full_model.cu + .py (full 4.84M equivalent)
- test_ops.cu (GELU+softmax+AdamW)
- test_mlp.cu (MLP fwd+bwd)
- test_full_block.cu (pre-LN block, 14 grads)
- test_lblock.cu (L=3 chain)
- test_attn.cu + test_attn_bwd.cu

### NEXT SESSION: TRAINING LOOP
1. AdamW kernel (already in test_ops.cu — reuse)
2. Training step: fwd + bwd + AdamW + log
3. Checkpoint save/load (CPU<->GPU weights)
4. Real data loader (corpus.bin -> ids/tg)

### KEY BUGS FOUND
- mm_nt order: (M, K, N) not (M, N, K)
- dT1 must be sh (T*DFF), not sx (T*D)
- bpre (post-gelu) vs bpre_save (pre-gelu)
- xnf forward saved separately
- dwte vs dwte_g separate buffers
- Python: np.stack, not np.concatenate

### ENVIRONMENT
- CUDA 12.8 at /usr/local/cuda-12.8
- GLIBC fix: ~/cuda_compat/glibc_c23_math_compat.h
- NVCC_PREPEND_FLAGS in ~/.bashrc
- GTX 1660 Ti (7.5, 24 SM, 6143 MB)

## 31. TRAINING STEP 1 PASS (2026-11-02)

### DONE
- `test_train_step.cu/.py`: fwd + bwd + AdamW step 1
- 12 grads PASS (< 3.1e-05), 16 updated weights PASS (< 8.5e-07)
- `adamw_dev` in block_ops.cu (bc1/bc2 args, no libm)

### NEXT SESSION — pick up here
1. **Refactor backward into a function** (`model_bwd()` callable)
   Currently backward is inline in `test_full_model.cu` main().
   Needed for multi-step loop without copy-paste.
2. **2-step training loop** — verify m/v momentum persistence
3. **Checkpoint save/load** — save all 16 tensors + m/v state to dir
4. **Real data loader** — corpus.bin → ids/tg per step

### KEY FILES
- `scripts/cuda/test_train_step.cu` — training step 1 (WORKING, restored from be710e6)
- NOTE: d7cbc0d accidentally committed a WIP 2-step loop; reverted at f8f62f0+
- `scripts/cuda/test_full_model.cu` — fwd+bwd (WORKING)
- `scripts/cuda/block_ops.cu` — LN/GELU/add/bias/adamw_dev

### CONTEXT NOTE
Session 31 ends here — context near full. Next session: start by reading
this doc + `git log --oneline -10`, then continue with refactor above.


## 32. 2-STEP ADAMW LOOP — m/v PERSISTENCE PASS (2026-11-02)

### DONE
- `test_train_step.cu/.py`: 2-step AdamW loop with persistent m/v
- Step 1: 16 weights PASS (< 8.8e-07)
- Step 2: 16 weights PASS (< 1.8e-06) — momentum verified
- bc1/bc2 corrected per step: bc1 = 1-b1^step, bc2 = 1-b2^step

### APPROACH NOTE
- Step-2 reuses step-1 grads (shortcut, NOT real 2nd fwd+bwd)
- Reason: full backward is still inline in main(), not a function yet
- This is intentional — proves AdamW m/v math is correct before
  the loop refactor

### NEXT SESSION — pick up here
1. **Refactor forward into function** (`model_fwd(...)`)
2. **Refactor backward into function** (`model_bwd(...)`)
   Currently both inline in test_train_step.cu main()
3. **True 2-step loop**: call model_fwd + model_bwd + adamw_dev twice,
   with real second step (not reused grads)
4. **Checkpoint save/load** — all 16 tensors + m/v to dir
5. **Real data loader** — corpus.bin -> ids/tg per step

### KEY FILES
- `scripts/cuda/test_train_step.cu` — 2-step loop (WORKING)
- `scripts/cuda/test_full_model.cu` — single fwd+bwd (WORKING)
- `scripts/cuda/block_ops.cu` — LN/GELU/add/bias/adamw_dev

### LESSON LEARNED
- Heredoc paste into bash is fragile (we saw 20+ syntax errors)
- Use `cat > /tmp/script.py << 'PYEOF'` first, then `python3 /tmp/script.py`
- The script validates anchors exist before writing — catches mismatch early


## 33. REAL 2-STEP TRAINING — FRESH FWD+BWD PASS (2026-11-03)

### DONE
- `test_train_step.cu`: real 2-step loop (fwd+bwd ×2, distinct grads)
- Step-2 grads saved (fm_s2_*) for verification
- Step-2 grads use g1 + g2 in AdamW m/v accumulation (verified)
- Sanity check: |y2-y1| = 1.92 (forward re-ran, weights changed)
- Step-1: 16 weights PASS (< 8.8e-07)
- Step-2 (REAL): 16 weights PASS (< 1.2e-07)

### KEY INSIGHT
- Prior section 32 used step-1 grads for step-2 (shortcut)
- This session runs ACTUAL step-2 fwd+bwd with step-1 weights
- Distinguishes: AdamW math (section 32) vs training loop (section 33)

### ARCHITECTURE LEARNED
- Python reference: adamw_step2(w1, g1, g2) = real 2-step reference
- Requires g1 saved → fixed by saving fm_dw1/fm_db1/etc BEFORE memset
- Shape gotchas: W2 is (L, DFF, D) not (L, D, DFF); dW2 same

### NEXT SESSION — pick up here
1. **True N-step loop (N=4, 8)** — generalize 2-step; verify loss decreases
2. **Checkpoint save/load** — all 16 tensors + m/v state
3. **Real data loader** — corpus.bin → ids/tg per step
4. **Clean function refactor** — model_fwd(), model_bwd() (still inline)

### FILES
- `scripts/cuda/test_train_step.cu` — REAL 2-step loop (WORKING)
- `scripts/cuda/test_train_step.py` — full reference verify
- `scripts/cuda/test_full_model.cu` — single fwd+bwd reference


## 34. N=20-STEP TRAINING — LOSS DECREASES (2026-11-03)

### THE MILESTONE
- 20-step AdamW loop: loss 6.961554 -> 3.374765 (delta -3.587, -52%)
- Monotonic decrease: every step lower than previous
- ALL step-1 + step-2 verify checks still PASS

### IMPLEMENTATION
- #define N_STEPS 20 (overridable)
- Loop: step 1..N: AdamW -> save weights -> [if step<N: fwd+bwd, next grads]
- Per-step avg loss saved to /tmp/loss_s{step}.bin
- Python reads all 20, prints trajectory, flags ✓ if decreased

### LOSS TRAJECTORY (reference)
s1=6.962, s5=5.801, s10=4.771, s15=3.980, s20=3.375
Smooth monotonic decay - typical of healthy training.

### KNOWN COSMETIC BUG
- printf still says "Train step 1 done" - not updated to N_STEPS
- Harmless (python check is the source of truth)
- Fix when convenient

### NEXT SESSION — pick up here
1. **Checkpoint save/load** — all 16 tensors + m/v state to dir
   (now possible: weights are meaningfully different after 20 steps)
2. **Real data loader** — corpus.bin -> ids/tg per step
   (currently ids/tg fixed, so loss decreases on same batch = memorization)
3. **Function refactor** — model_fwd()/model_bwd() to eliminate code dup
4. **Larger N** (100+) + multiple batches to verify generalization

### KEY FILES
- `scripts/cuda/test_train_step.cu` — N-step loop (WORKING)
- `scripts/cuda/test_train_step.py` — verify + loss trajectory
- `docs/NYANLIN_CONTEXT.md` — this file


## 35. MULTI-BATCH TRAINING ON REAL CORPUS (2026-11-03)

### THE MILESTONE
- First real-data training on GPU: 45 steps × different batches
- Train loss 6.96 -> 3.44 (fluctuates because batches differ, trend down)
- Val loss (5 held-out): mean 4.32
- Train/Val gap 0.87 → genuine generalization, not memorization
- All step-1/2 verify checks still PASS

### DATA PIPELINE
- corpus.bin = raw UTF-8 bytes (np.uint8), 312MB
- Vocabulary: V=256 (byte-level)
- Batches: /tmp/batch_{k}_ids.bin + batch_{k}_tg.bin (int32)
- T=4 per batch (NOTE: very short, next step = T=16 or 32)
- START=1000, stride T per batch

### IMPLEMENTATION
- CUDA: per-step batch reload inside loop (batch_{step+1}_*.bin)
- After 45 train steps: forward-only eval on 5 held-out batches
- Per-batch loss saved (loss_s{k}.bin, loss_val{v}.bin)
- Python reads all + prints trajectory + generalization verdict

### LESSONS
- Same-batch loss (section 34) was memorization test only
- Multi-batch confirms real learning (val tracks train)
- Per-step fluctuation is NORMAL (different batch each step)
- T=4 gives limited context — increasing T is the natural next step

### NEXT SESSION — pick up here
1. **T=16 or 32** — longer context, better loss floor
   - Update CUDA kernels (attn stack arrays sized by T, MAX_T=512 supports this)
   - Regenerate batches with new T
2. **Checkpoint save/load** — all 16 tensors + m/v
3. **LR schedule** — warmup + cosine (see ref/run_full_train.py)
4. **N=200-500 steps** to see steady-state loss

### KEY FILES
- scripts/cuda/test_train_step.cu — N-step multi-batch loop
- scripts/cuda/test_train_step.py — verify + trajectory + val
- ref/full_test/corpus.bin — 312MB UTF-8 bytes
- ref/run_full_train.py — reference (T=128, full arch)
