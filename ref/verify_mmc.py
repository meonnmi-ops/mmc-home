"""verify_mmc.py — compare MMC forward output vs ref_logits.npy
Usage:
  python verify_mmc.py                    # smoke test (ref vs ref)
  python verify_mmc.py path/to/mmc.npy    # real MMC output
"""
import sys
import numpy as np
from pathlib import Path
from logs import check, save_log

REF = Path(__file__).parent / "phase1" / "ref_logits.npy"
TOL = 1e-5

def main():
    ref = np.load(REF)
    check("ref_logits", ref)

    if len(sys.argv) > 1:
        mmc = np.load(sys.argv[1])
        tag = "mmc-vs-ref"
        print(f"\nMMC file: {sys.argv[1]}")
    else:
        # smoke: add tiny noise to ref, confirm diff reports correctly
        rng = np.random.default_rng(0)
        mmc = ref + rng.normal(0, 1e-7, ref.shape).astype(np.float32)
        tag = "smoke"
        print("\nSMOKE MODE (ref + noise)")

    check("mmc_logits", mmc)

    if mmc.shape != ref.shape:
        raise AssertionError(f"shape mismatch: mmc={mmc.shape} ref={ref.shape}")

    diff = np.abs(mmc.astype(np.float64) - ref.astype(np.float64))
    flat = diff.ravel()
    idx = np.unravel_index(flat.argmax(), diff.shape)

    print(f"\n--- diff report ---")
    print(f"  max       = {diff.max():.4e}")
    print(f"  mean      = {diff.mean():.4e}")
    print(f"  p50       = {np.percentile(flat, 50):.4e}")
    print(f"  p99       = {np.percentile(flat, 99):.4e}")
    print(f"  p99.9     = {np.percentile(flat, 99.9):.4e}")
    print(f"  argmax    = {idx}")
    print(f"  ref@argmax= {ref[idx]:.6e}")
    print(f"  mmc@argmax= {mmc[idx]:.6e}")

    ok = bool(diff.max() < TOL)
    entry = {
        "tag": tag, "tol": TOL, "ok": ok,
        "max": float(diff.max()), "mean": float(diff.mean()),
        "p99": float(np.percentile(flat, 99)), "argmax": [int(i) for i in idx],
    }
    save_log(entry)

    print("="*60)
    if ok:
        print(f"✅ PASS  max_diff={diff.max():.4e} < tol={TOL:.0e}")
    else:
        print(f"❌ FAIL  max_diff={diff.max():.4e} >= tol={TOL:.0e}")
        print("  Check contracts.md §1-7 in order. Do NOT modify ref to match.")
    print("="*60)
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
