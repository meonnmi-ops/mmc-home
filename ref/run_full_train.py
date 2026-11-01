"""run_full_train.py — Phase 1 orchestrator: drives ./build/train_full in a loop.

Per step: clean stale out_* -> sample SEQ=128 window -> write ids/tg/lr/step
-> run ./build/train_full -> on success rename out_* -> *  (commit point).

Features: warmup+cosine LR, JSONL loss log, checkpoints, auto-resume.
step.bin = GLOBAL optimizer step (AdamW bias correction); LR schedule is
indexed within this run. batch=1 (true batch=4 = Phase 1.5).
"""
import argparse, json, math, os, re, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np

SEQ, TMAX, D, DFF, H, V, L = 128, 128, 256, 1024, 8, 256, 6
NAMES = ['Wqkv', 'bqkv', 'Wo', 'bo', 'W1', 'b1', 'W2', 'b2',
         'ln1w', 'ln1b', 'ln2w', 'ln2b', 'wte', 'wpe', 'lnfw', 'lnfb']
SIZES = {'Wqkv': L*3*D*D, 'bqkv': L*3*D, 'Wo': L*D*D, 'bo': L*D,
         'W1': L*DFF*D, 'b1': L*DFF, 'W2': L*D*DFF, 'b2': L*D,
         'ln1w': L*D, 'ln1b': L*D, 'ln2w': L*D, 'ln2b': L*D,
         'wte': V*D, 'wpe': TMAX*D, 'lnfw': D, 'lnfb': D}
TOTAL = sum(SIZES.values())

ap = argparse.ArgumentParser()
ap.add_argument('--steps', type=int, default=500, help='steps to run this session')
ap.add_argument('--lr', type=float, default=1e-3, help='peak learning rate')
ap.add_argument('--warmup', type=int, default=50, help='linear warmup steps')
ap.add_argument('--ckpt-every', type=int, default=200)
ap.add_argument('--log-every', type=int, default=10)
ap.add_argument('--seed', type=int, default=1234)
ap.add_argument('--start', type=int, default=0, help='global first step (0 = auto from next_step.txt)')
ap.add_argument('--no-ckpt', action='store_true')
args = ap.parse_args()

ROOT = Path.home() / 'mmc_wsl/mmc_publish'
os.chdir(ROOT)
B = Path('ref/full_test'); B.mkdir(parents=True, exist_ok=True)

# ---------- preconditions ----------
if not (ROOT / 'build/train_full').exists():
    sys.exit("❌ build/train_full missing — run: bash scripts/build_train_full.sh")
missing = [f'{n}.bin' for n in NAMES if not (B / f'{n}.bin').exists()]
if missing:
    sys.exit(f"❌ weights missing: {missing}\n   run: python3 ref/full_train_gen.py")
if not (B / 'corpus.bin').exists():
    sys.exit("❌ corpus.bin missing — run: python3 ref/prep_corpus.py (--jsonl FILE | --text FILE)")

corpus = np.fromfile(B / 'corpus.bin', dtype=np.uint8)
ncorp = len(corpus)
if ncorp < SEQ + 2:
    sys.exit(f"❌ corpus too small ({ncorp} bytes, need >= {SEQ+2})")

# ---------- resume ----------
start = args.start
if start <= 0:
    ns = B / 'next_step.txt'
    start = int(ns.read_text().strip()) if ns.exists() else 1
end = start + args.steps - 1

print("=" * 62)
print(f" Phase 1 full training — {TOTAL:,} params (T={SEQ}, D={D}, L={L}, H={H})")
print(f" corpus {ncorp:,} bytes | steps {start}..{end} | peak lr {args.lr:g} (warmup {args.warmup})")
print("=" * 62)


def lr_at(prog):
    """warmup + cosine within this run; prog = 1..steps."""
    if prog <= args.warmup:
        return args.lr * prog / max(1, args.warmup)
    t = (prog - args.warmup) / max(1, args.steps - args.warmup)
    return max(args.lr * 0.1, args.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t))))


LOG = (B / 'loss_log.jsonl').open('a')
losses, t_session = [], time.time()
failed = False

for s in range(start, end + 1):
    prog = s - start + 1
    lr_now = lr_at(prog)

    # 1) hygiene: drop stale outputs from any crashed previous step
    for f in B.glob('out_*.bin'):
        f.unlink()

    # 2) sample batch (deterministic per (seed, global step))
    rng = np.random.RandomState((args.seed * 1_000_003 + s) % (2**31))
    pos = int(rng.randint(0, ncorp - SEQ - 1))
    corpus[pos:pos + SEQ].astype(np.int32).tofile(B / 'ids.bin')
    corpus[pos + 1:pos + SEQ + 1].astype(np.int32).tofile(B / 'tg.bin')

    # 3) control files
    np.array([lr_now], dtype=np.float64).tofile(B / 'lr.bin')
    np.array([s], dtype=np.int32).tofile(B / 'step.bin')

    # 4) run one MMC step
    t0 = time.time()
    r = subprocess.run(['./build/train_full'], capture_output=True, text=True)
    dt = time.time() - t0

    m = re.search(r'loss = ([0-9.eE+-]+)', r.stdout)
    g = re.search(r'gnorm = ([0-9.eE+-]+)', r.stdout)
    if m is None or r.returncode != 0:
        print(f"\n❌ step {s} FAILED (rc={r.returncode}, {dt:.1f}s)")
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        failed = True
        break
    loss = float(m.group(1))
    gnorm = float(g.group(1)) if g else float('nan')

    # 5) commit: out_* -> *  (crash-safe: only after success)
    for f in B.glob('out_*.bin'):
        f.replace(B / f.name[4:])
    (B / 'next_step.txt').write_text(str(s + 1))

    losses.append(loss)
    LOG.write(json.dumps({'step': s, 'loss': loss, 'gnorm': gnorm,
                          'lr': lr_now, 'sec': round(dt, 3),
                          'pos': pos}) + '\n')
    LOG.flush()

    if (prog - 1) % args.log_every == 0 or s == end:
        el = time.time() - t_session
        print(f"step {s:5d}  loss {loss:8.4f}  gnorm {gnorm:9.4f}  "
              f"lr {lr_now:.2e}  {dt:5.2f}s/step  elapsed {el/60:5.1f}m")

    if not args.no_ckpt and s % args.ckpt_every == 0:
        ck = B / f'ckpt_{s}'; ck.mkdir(exist_ok=True)
        for name in NAMES:
            for pfx in ('', 'm_', 'v_'):
                src = B / f'{pfx}{name}.bin'
                if src.exists():
                    shutil.copy2(src, ck / src.name)
        shutil.copy2(B / 'step.bin', ck / 'step.bin')
        (ck / 'next_step.txt').write_text(str(s + 1))
        print(f"  ⛁ checkpoint -> {ck}")

LOG.close()

# ---------- summary ----------
print()
if losses:
    k = min(20, len(losses))
    mean_last = sum(losses[-k:]) / k
    drop = 100.0 * (1.0 - losses[-1] / losses[0]) if losses[0] > 0 else 0.0
    print(f"first {losses[0]:.4f}  ->  last {losses[-1]:.4f}   ({drop:.1f}% drop)")
    print(f"mean of last {k}: {mean_last:.4f}")
    print(f"loss log: {B/'loss_log.jsonl'}")
if failed:
    print(f"⚠️  stopped at step {s}; weights/m/v on disk are the last good commit")
    print(f"   resume with: python3 ref/run_full_train.py --steps {args.steps}")
    sys.exit(1)
else:
    print("✅ session complete — resume anytime with: python3 ref/run_full_train.py --steps N")
