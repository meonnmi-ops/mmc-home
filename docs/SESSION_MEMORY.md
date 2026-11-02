# SESSION MEMORY — Nyanlin training handoff
Last updated: 2026-11-03 (Claude session, hit rate limit at section 42)

## WHERE WE ARE
Custom CUDA GPT-2 (no PyTorch). Model: D=64, DFF=128, H=2, L=3, V=256, T=32.
Best val loss: 0.854 (Run E: B=8, N=6000, lr 1.5e-3, n_val=256, sem 0.019).
Bigram baseline: 1.314. Train ~= val (0.847 vs 0.854) -> no overfit.
Params ~118K, tokens seen ~1.5M. Model uses context (not bigram-level).

## JUST LAUNCHED (STATUS UNKNOWN — CHECK FIRST)
Run F: capacity test, D=128 DFF=256 H=4, lr 1e-3, N=6000, B=8, nval=256.
Command was:
  nohup bash scripts/cuda/run_acc.sh 128 256 4 1e-3 6000 8 256 > /tmp/run_F.txt 2>&1 &
Result to /tmp/run_F.txt. Expected ~10+ min. CHECK THIS FIRST.

## DECISION TREE FOR RUN F
- val < 0.80  -> capacity was limiting, go bigger (D=256)
- val ~= 0.85 -> steps/data limiting, increase N (not D)
- loss spike  -> lower lr (try 5e-4)
- NaN/diverg  -> lr too high for D=128, drop to 5e-4 or 3e-4

## CRITICAL: /tmp IS VOLATILE (WSL restart wipes it)
After restart, rebuild:
  nvcc -O3 -arch=sm_75 -c scripts/cuda/block_ops.cu -o /tmp/block_ops.o
Verify batches need:
  python3 scripts/gen_multi_batches.py 45 1 30
Verify needs /tmp/fm_dy.bin (binary reads it, contents unused) - regen if missing.

## KEY FILES
- scripts/cuda/test_train_step.cu  - training loop
- scripts/cuda/block_ops.cu/.h     - LN/GELU/add/bias/adamw device API
- scripts/cuda/run_acc.sh          - run wrapper: run_acc.sh D DFF H LR N B NVAL
- scripts/gen_multi_batches.py     - batch generator (args: N B VAL)
- docs/NYANLIN_CONTEXT.md          - full history (sections 1..42)

## COMPILE FLAGS (standard)
  nvcc -O3 -arch=sm_75 -DUSE_LR_SCHEDULE=1 -DN_STEPS=N -DWARMUP=N/20 ...
  (block_ops.o linked separately, see above)

## BUGS FOUND THIS SESSION (already fixed)
1. Causal mask leak in attention - model was seeing future tokens.
2. LR calculation was wrong - fixed warmup+cosine.
3. Init weights too large - training blocked. Fixed -> loss 2.15 -> 1.32.
4. Grad accumulation added: 8 batches per step, average gradients.
5. n_val was 30 (too noisy) -> 256 for stable stats.

## NEXT STEPS (in order)
1. Check /tmp/run_F.txt for Run F result -> apply decision tree above.
2. Checkpoint save/load (still not done) - needed for long runs.
3. Data diversity: currently START=1000 region only (~7KB of 312MB corpus).
   Consider random START per step or stride over full corpus.
4. If capacity wins: D=128, DFF=256, H=4 -> D=256, DFF=512, H=8.
5. If steps win: N=12000 with tuned schedule.

## WHAT TO PASTE TO RESUME (next AI session)
"""
Picking up Nyanlin CUDA GPT-2 training. Read docs/SESSION_MEMORY.md first,
then docs/NYANLIN_CONTEXT.md sections 40-42 for full context.
State: D=64 model, val 0.854 (Run E). Run F (D=128) was launched, status
unknown - check /tmp/run_F.txt first, then apply decision tree in memory.
/tmp is volatile - rebuild block_ops.o if WSL restarted.
"""
