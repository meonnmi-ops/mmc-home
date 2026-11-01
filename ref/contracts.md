# NYANLIN Phase 1 — Reference/MMC File Contract

Version 1.0.0 · Locked after Cave 2 gradient-check pass

This document defines the files produced by `ref_model.py` and consumed by
the pure MMC implementation. The reference model is a debug oracle; MMC must
match it. Do not change the reference to make a mismatching MMC result pass.

## Locked model configuration

| Name | Value |
|---|---:|
| Vocabulary (byte-level) | 256 |
| Sequence length | 128 |
| Width `D` | 256 |
| Transformer blocks | 6 |
| Attention heads | 8 |
| Head width `Dh` | 32 |
| Feed-forward width | 1024 |
| Parameter count | 4,837,376 |
| Tensor dtype | float32, little-endian |

## Files

Run `python ref_model.py --out ref/phase1` from the repository root. It writes:

| File | Shape / contents |
|---|---|
| `input_ids.npy` | `int32`, `(1, 128)`, values in `[0, 256)` |
| `weights.npz` | 76 named `float32` arrays, 4,837,376 values total |
| `ref_logits.npy` | `float32`, `(1, 128, 256)` |
| `ref_grads.npz` | 76 `float32` gradient arrays matching weight names/shapes |

The default seed is 42. The input IDs use NumPy's `default_rng(seed)`; model
initialization uses `torch.manual_seed(seed)`. Use the same seed when
reproducing reference files.

## Weight names and shapes

Top-level arrays:

| Key | Shape |
|---|---|
| `wte` | `(256, 256)` |
| `wpe` | `(128, 256)` |
| `ln_f_w`, `ln_f_b` | `(256,)` each |

For each block `i` from 0 through 5:

| Suffix | Shape |
|---|---|
| `ln_1_w`, `ln_1_b` | `(256,)` |
| `qkv_w` | `(768, 256)` |
| `qkv_b` | `(768,)` |
| `proj_w` | `(256, 256)` |
| `proj_b` | `(256,)` |
| `ln_2_w`, `ln_2_b` | `(256,)` |
| `fc1_w` | `(1024, 256)` |
| `fc1_b` | `(1024,)` |
| `fc2_w` | `(256, 1024)` |
| `fc2_b` | `(256,)` |

Names are case-sensitive (`b0_qkv_w`, for example). Matrix weights use
`(out_features, in_features)` layout, so MMC computes `y = W @ x + b`.
QKV rows are contiguous: `[0:256]` is Q, `[256:512]` is K, and `[512:768]`
is V. The LM head is tied to `wte`: `logits = x @ wte.T`, with no bias or
scaling.

## Forward math MMC must match

Each block is pre-LayerNorm with residual attention and MLP:

1. LayerNorm uses `eps=1e-5` and biased variance (divide by `D`, not `D-1`).
2. Project normalized input to Q, K, V, then reshape to `(B, H, T, Dh)`.
3. Compute attention scores `(Q @ K.T) / sqrt(Dh)`; mask future positions
   with negative infinity before stable softmax.
4. Combine values, project, and add the attention residual.
5. Apply LayerNorm, linear `fc1`, GPT-2 tanh-approximate GELU, linear `fc2`,
   and add the MLP residual.
6. Apply final LayerNorm and the tied embedding head.

The GELU definition is

```text
0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

Do not substitute erf GELU, rotary embeddings, ALiBi, an untied head, or a
different attention-mask convention.

## Loss and gradients

The reference loss is causal next-token cross entropy:

```text
cross_entropy(logits[:, :-1, :], input_ids[:, 1:])
```

Every gradient array must have the same name and shape as its corresponding
weight and contain only finite float32 values.

## Validation and MMC comparison

Run the file-contract validator from the `ref` directory:

```bash
python contracts.py
```

It validates IDs, exact weight/gradient keys, shapes, dtypes, finiteness,
parameter count, and logit shape. After MMC writes `mmc_logits.npy`, compare
it with `ref/phase1/ref_logits.npy`; report max, mean, p99 absolute error and
argmax location. Target: `max(abs(mmc_logits - ref_logits)) < 1e-5` for
float32 accumulation. The existing `verify_mmc.py` handles that comparison.
