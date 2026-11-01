"""Runtime validator for the NYANLIN Phase 1 reference/MMC file contract.

Run from this directory with: python contracts.py
Or pass another output directory: python contracts.py --dir path/to/phase1
"""
import argparse
from pathlib import Path

import numpy as np


VOCAB = 256
SEQ = 128
D = 256
LAYERS = 6
HEADS = 8
D_FF = 1024
EXPECTED_PARAMS = 4_837_376
EXPECTED_KEYS = {
    "wte", "wpe", "ln_f_w", "ln_f_b",
    *{
        f"b{i}_{name}"
        for i in range(LAYERS)
        for name in (
            "ln_1_w", "ln_1_b", "qkv_w", "qkv_b", "proj_w", "proj_b",
            "ln_2_w", "ln_2_b", "fc1_w", "fc1_b", "fc2_w", "fc2_b",
        )
    },
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def validate(out_dir):
    out = Path(out_dir)
    require(out.is_dir(), f"output directory not found: {out}")

    # Input token IDs: row-major (B=1, T=128), int32, byte vocabulary.
    ids = np.load(out / "input_ids.npy", allow_pickle=False)
    require(ids.dtype == np.int32, f"input_ids dtype {ids.dtype} != int32")
    require(ids.shape == (1, SEQ), f"input_ids shape {ids.shape} != (1, {SEQ})")
    require(ids.min() >= 0 and ids.max() < VOCAB, "input_ids outside [0, 256)")
    print(f"PASS input_ids: shape={ids.shape}, dtype={ids.dtype}, range=[{ids.min()}, {ids.max()}]")

    # Named model weights, all float32 and laid out (out_features, in_features).
    with np.load(out / "weights.npz", allow_pickle=False) as archive:
        weights = {key: archive[key] for key in archive.files}
    require(set(weights) == EXPECTED_KEYS,
            f"weight keys differ: missing={sorted(EXPECTED_KEYS - set(weights))}, "
            f"extra={sorted(set(weights) - EXPECTED_KEYS)}")

    expected_shapes = {
        "wte": (VOCAB, D),
        "wpe": (SEQ, D),
        "ln_f_w": (D,),
        "ln_f_b": (D,),
    }
    block_shapes = {
        "ln_1_w": (D,), "ln_1_b": (D,),
        "qkv_w": (3 * D, D), "qkv_b": (3 * D,),
        "proj_w": (D, D), "proj_b": (D,),
        "ln_2_w": (D,), "ln_2_b": (D,),
        "fc1_w": (D_FF, D), "fc1_b": (D_FF,),
        "fc2_w": (D, D_FF), "fc2_b": (D,),
    }
    for i in range(LAYERS):
        expected_shapes.update({f"b{i}_{name}": shape for name, shape in block_shapes.items()})

    for key, expected in expected_shapes.items():
        arr = weights[key]
        require(arr.shape == expected, f"{key} shape {arr.shape} != {expected}")
        require(arr.dtype == np.float32, f"{key} dtype {arr.dtype} != float32")
        require(np.isfinite(arr).all(), f"non-finite weight values in {key}")

    n_params = sum(arr.size for arr in weights.values())
    require(n_params == EXPECTED_PARAMS,
            f"parameter count {n_params:,} != {EXPECTED_PARAMS:,}")
    print(f"PASS weights: {len(weights)} arrays, {n_params:,} float32 values")

    # Forward reference logits: tied LM head, no bias or scaling.
    logits = np.load(out / "ref_logits.npy", allow_pickle=False)
    require(logits.dtype == np.float32, f"ref_logits dtype {logits.dtype} != float32")
    require(logits.shape == (1, SEQ, VOCAB),
            f"ref_logits shape {logits.shape} != (1, {SEQ}, {VOCAB})")
    require(np.isfinite(logits).all(), "non-finite values in ref_logits")
    print(f"PASS ref_logits: shape={logits.shape}, dtype={logits.dtype}, "
          f"mean={logits.mean():.6f}, std={logits.std():.6f}")

    # Backward reference gradients use exactly the weight names and shapes.
    with np.load(out / "ref_grads.npz", allow_pickle=False) as archive:
        grads = {key: archive[key] for key in archive.files}
    require(set(grads) == EXPECTED_KEYS,
            f"gradient keys differ: missing={sorted(EXPECTED_KEYS - set(grads))}, "
            f"extra={sorted(set(grads) - EXPECTED_KEYS)}")
    for key, grad in grads.items():
        require(grad.shape == weights[key].shape,
                f"{key} gradient shape {grad.shape} != weight shape {weights[key].shape}")
        require(grad.dtype == np.float32, f"{key} gradient dtype {grad.dtype} != float32")
        require(np.isfinite(grad).all(), f"non-finite gradient values in {key}")
    print(f"PASS ref_grads: {len(grads)} finite float32 arrays match weight shapes")
    print("ALL PHASE 1 FILE CONTRACTS PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=str(Path(__file__).parent / "phase1"),
                        help="directory containing the four ref_model.py outputs")
    args = parser.parse_args()
    validate(args.dir)


if __name__ == "__main__":
    main()
