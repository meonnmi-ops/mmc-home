"""Data loader test: MMC vs Python, LCG shuffle + batch extraction."""
import numpy as np, sys
from pathlib import Path

SEQ, BATCH, SEED = 128, 4, 42
B = Path('ref/data_test'); B.mkdir(parents=True, exist_ok=True)

# Generate corpus
if not (B/'corpus.bin').exists():
    english = b"The quick brown fox jumps over the lazy dog. " * 30
    myanmar = "မြန်မာစာ စမ်းသပ်မှု ဒေတာ။ ".encode('utf-8') * 20
    text = english + myanmar
    np.frombuffer(text, dtype=np.uint8).tofile(B/'corpus.bin')
    print(f"Generated corpus: {len(text)} bytes")

buf = np.fromfile(B/'corpus.bin', dtype=np.uint8)
n = len(buf)
print(f"corpus size = {n} bytes")

# Same LCG as MMC: state = (state * 25173 + 13849) % 65536
state = SEED
positions = []
for b in range(BATCH):
    state = (state * 25173 + 13849) % 65536
    pos = state % (n - SEQ - 1)
    positions.append(pos)
    print(f"batch {b}: pos = {pos}")

inp_ref = np.zeros((BATCH, SEQ), dtype=np.int32)
tgt_ref = np.zeros((BATCH, SEQ), dtype=np.int32)
for b, pos in enumerate(positions):
    inp_ref[b] = buf[pos:pos+SEQ].astype(np.int32)
    tgt_ref[b] = buf[pos+1:pos+SEQ+1].astype(np.int32)

if not (B/'inp.bin').exists():
    print("❌ inp.bin MISSING — run MMC first")
    sys.exit(1)

mmc_inp = np.fromfile(B/'inp.bin', dtype=np.int32).reshape(BATCH, SEQ)
mmc_tgt = np.fromfile(B/'tgt.bin', dtype=np.int32).reshape(BATCH, SEQ)

d_inp = int(np.abs(mmc_inp - inp_ref).max())
d_tgt = int(np.abs(mmc_tgt - tgt_ref).max())

print()
print("=== Data loader comparison ===")
print(f"  inp:  max={d_inp}  {'✅' if d_inp == 0 else '❌'}")
print(f"  tgt:  max={d_tgt}  {'✅' if d_tgt == 0 else '❌'}")

if d_inp == 0 and d_tgt == 0:
    print("\n🏆 DATA LOADER PASS — batches match exactly")
else:
    print("\n❌ FAIL — sample values")
    print(f"  mmc_inp[0,:10] = {mmc_inp[0,:10]}")
    print(f"  ref_inp[0,:10] = {inp_ref[0,:10]}")
    sys.exit(1)
