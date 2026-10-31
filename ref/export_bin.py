"""export_bin.py — convert phase1 npz/npy to raw float32 .bin for MMC"""
import numpy as np
from pathlib import Path

SRC = Path(__file__).parent / "phase1"
DST = Path(__file__).parent / "phase1" / "bin"
DST.mkdir(exist_ok=True)

def save_raw(name, arr, dtype=np.float32):
    a = np.ascontiguousarray(arr, dtype=dtype)
    a.tofile(DST / f"{name}.bin")
    print(f"  {name}.bin: shape={a.shape} dtype={a.dtype} bytes={a.nbytes}")

W = np.load(SRC / "weights.npz")
for k in W.files:
    save_raw(k, W[k])

# input_ids as int32
ids = np.load(SRC / "input_ids.npy").astype(np.int32)
ids.tofile(DST / "input_ids.bin")
print(f"  input_ids.bin: shape={ids.shape} dtype=int32 bytes={ids.nbytes}")

# also keep ref_logits.npy (numpy side) — no need to bin it

# write a manifest with metadata
import json
meta = {"vocab":256, "seq":128, "d":256, "l":6, "h":8, "d_ff":1024,
        "n_weights": len(W.files), "param_count": int(sum(W[k].size for k in W.files))}
(DST / "meta.json").write_text(json.dumps(meta, indent=2))
print(f"  meta.json: {meta}")

print(f"\n✅ All binaries in: {DST}")
