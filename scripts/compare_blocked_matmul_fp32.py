#!/usr/bin/env python3
"""Compare deterministic FP32 matmul probe outputs and report timings."""
import argparse
import re
from pathlib import Path

import numpy as np


CASES = (
    ("qkv", 128, 768, 256),
    ("attn_proj", 128, 256, 256),
    ("fc1", 128, 1024, 256),
    ("fc2", 128, 256, 1024),
)
ARRAYS = ("out", "dX", "dW", "db")
TIMING = re.compile(
    r"^\s*(\w+)\s+fwd=\s*([0-9.]+) ms\s+bwd=\s*([0-9.]+) ms\s*$",
    re.MULTILINE,
)


def timings(path):
    found = {name: (float(fwd), float(bwd))
             for name, fwd, bwd in TIMING.findall(Path(path).read_text())}
    missing = {case[0] for case in CASES} - set(found)
    if missing:
        raise ValueError(f"timing log {path} is missing cases: {sorted(missing)}")
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline", type=Path)
    ap.add_argument("blocked", type=Path)
    ap.add_argument("--baseline-log", type=Path, required=True)
    ap.add_argument("--blocked-log", type=Path, required=True)
    ap.add_argument("--baseline-label", default="baseline")
    ap.add_argument("--candidate-label", default="blocked")
    ap.add_argument("--atol", type=float, default=2e-5)
    ap.add_argument("--rtol", type=float, default=2e-5)
    args = ap.parse_args()

    base = np.fromfile(args.baseline, dtype=np.float32)
    opt = np.fromfile(args.blocked, dtype=np.float32)
    if base.size != opt.size:
        raise SystemExit(f"FAIL output length differs: baseline={base.size}, blocked={opt.size}")
    if not np.isfinite(base).all() or not np.isfinite(opt).all():
        raise SystemExit("FAIL output contains NaN or Inf")

    base_t = timings(args.baseline_log)
    opt_t = timings(args.blocked_log)
    offset = 0
    overall_ok = True
    print(f"Numerical comparison: atol={args.atol:g}, rtol={args.rtol:g}")
    for name, T, J, K in CASES:
        sizes = (T * J, T * K, J * K, J)
        for array_name, size in zip(ARRAYS, sizes):
            end = offset + size
            a, b = base[offset:end], opt[offset:end]
            close = np.allclose(a, b, atol=args.atol, rtol=args.rtol)
            max_abs = float(np.max(np.abs(a - b))) if size else 0.0
            print(f"  {'PASS' if close else 'FAIL'} {name}.{array_name}: max_abs={max_abs:.3g}")
            overall_ok &= bool(close)
            offset = end

        bf, bb = base_t[name]
        of, ob = opt_t[name]
        print(f"  speed {name}: fwd {bf / of:.2f}x, bwd {bb / ob:.2f}x "
              f"({args.baseline_label} {bf:.3f}/{bb:.3f} ms, "
              f"{args.candidate_label} {of:.3f}/{ob:.3f} ms)")

    if offset != base.size:
        raise SystemExit(f"FAIL unexpected trailing values: {base.size - offset}")
    if not overall_ok:
        raise SystemExit("FAIL: blocked matmul differs beyond tolerance")
    print(f"ALL {args.candidate_label.upper()} FP32 MATMUL OUTPUTS MATCH")


if __name__ == "__main__":
    main()
