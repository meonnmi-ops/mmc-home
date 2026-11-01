#!/usr/bin/env python3
"""Replace FP32 mm_fwd/mm_bwd in generated C with AVX2 implementations."""
import argparse
import re
from pathlib import Path


def function_span(text, name):
    pattern = re.compile(rf"\bint\s+{re.escape(name)}\s*\(")
    for match in pattern.finditer(text):
        brace = text.find("{", match.end())
        semi = text.find(";", match.end())
        if brace < 0 or (semi >= 0 and semi < brace):
            continue
        depth = 0
        i = brace
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return match.start(), i + 1
            i += 1
    raise ValueError(f"function definition not found: {name}")


def replace_function(text, name, replacement):
    start, end = function_span(text, name)
    return text[:start] + replacement + text[end:]


MM_FWD = r'''int mm_fwd(float* out, float* X, float* W, float* b, int T, int J, int K) {
    int t, j, k;
    for (t = 0; t < T; ++t) {
        for (j = 0; j < J; ++j) {
            __m256 acc = _mm256_setzero_ps();
            k = 0;
            for (; k + 7 < K; k += 8) {
                __m256 xv = _mm256_loadu_ps(X + t * K + k);
                __m256 wv = _mm256_loadu_ps(W + j * K + k);
                acc = _mm256_add_ps(acc, _mm256_mul_ps(xv, wv));
            }
            float lanes[8];
            float sum = b[j];
            _mm256_storeu_ps(lanes, acc);
            for (int lane = 0; lane < 8; ++lane) sum += lanes[lane];
            for (; k < K; ++k) sum += X[t * K + k] * W[j * K + k];
            out[t * J + j] = sum;
        }
    }
    return 0;
}'''


MM_BWD = r'''int mm_bwd(float* dX, float* dW, float* db, float* dout, float* X, float* W, int T, int J, int K, int add_dX) {
    int t, j, k;
    (void)add_dX;
    for (t = 0; t < T; ++t) {
        for (j = 0; j < J; ++j) db[j] += dout[t * J + j];
    }
    for (t = 0; t < T; ++t) {
        for (j = 0; j < J; ++j) {
            const __m256 dy = _mm256_set1_ps(dout[t * J + j]);
            k = 0;
            for (; k + 7 < K; k += 8) {
                float *dw = dW + j * K + k;
                float *dx = dX + t * K + k;
                const float *x = X + t * K + k;
                const float *w = W + j * K + k;
                _mm256_storeu_ps(dw, _mm256_add_ps(_mm256_loadu_ps(dw), _mm256_mul_ps(dy, _mm256_loadu_ps(x))));
                _mm256_storeu_ps(dx, _mm256_add_ps(_mm256_loadu_ps(dx), _mm256_mul_ps(dy, _mm256_loadu_ps(w))));
            }
            for (; k < K; ++k) {
                dW[j * K + k] += dout[t * J + j] * X[t * K + k];
                dX[t * K + k] += dout[t * J + j] * W[j * K + k];
            }
        }
    }
    return 0;
}'''


OMP_MM_FWD = r'''int mm_fwd(float* out, float* X, float* W, float* b, int T, int J, int K) {
    int t, j, k;
    #pragma omp parallel for schedule(static)
    for (t = 0; t < T; ++t) {
        for (j = 0; j < J; ++j) {
            __m256 acc = _mm256_setzero_ps();
            k = 0;
            for (; k + 7 < K; k += 8) {
                __m256 xv = _mm256_loadu_ps(X + t * K + k);
                __m256 wv = _mm256_loadu_ps(W + j * K + k);
                acc = _mm256_add_ps(acc, _mm256_mul_ps(xv, wv));
            }
            float lanes[8];
            float sum = b[j];
            _mm256_storeu_ps(lanes, acc);
            for (int lane = 0; lane < 8; ++lane) sum += lanes[lane];
            for (; k < K; ++k) sum += X[t * K + k] * W[j * K + k];
            out[t * J + j] = sum;
        }
    }
    return 0;
}'''


OMP_MM_BWD = r'''int mm_bwd(float* dX, float* dW, float* db, float* dout, float* X, float* W, int T, int J, int K, int add_dX) {
    int t, j, k;
    (void)add_dX;
    /* Each j owns one db row and one dW row. T reduction order is unchanged. */
    #pragma omp parallel for schedule(static)
    for (j = 0; j < J; ++j) {
        float db_sum = db[j];
        for (t = 0; t < T; ++t) {
            const __m256 dy = _mm256_set1_ps(dout[t * J + j]);
            db_sum += dout[t * J + j];
            for (k = 0; k + 7 < K; k += 8) {
                float *dw = dW + j * K + k;
                const float *x = X + t * K + k;
                _mm256_storeu_ps(dw, _mm256_add_ps(_mm256_loadu_ps(dw), _mm256_mul_ps(dy, _mm256_loadu_ps(x))));
            }
            for (; k < K; ++k) dW[j * K + k] += dout[t * J + j] * X[t * K + k];
        }
        db[j] = db_sum;
    }
    /* Each t owns one dX row. J reduction order is unchanged. */
    #pragma omp parallel for schedule(static)
    for (t = 0; t < T; ++t) {
        for (j = 0; j < J; ++j) {
            const __m256 dy = _mm256_set1_ps(dout[t * J + j]);
            for (k = 0; k + 7 < K; k += 8) {
                float *dx = dX + t * K + k;
                const float *w = W + j * K + k;
                _mm256_storeu_ps(dx, _mm256_add_ps(_mm256_loadu_ps(dx), _mm256_mul_ps(dy, _mm256_loadu_ps(w))));
            }
            for (; k < K; ++k) dX[t * K + k] += dout[t * J + j] * W[j * K + k];
        }
    }
    return 0;
}'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fp32-c", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--openmp", action="store_true",
                    help="emit OpenMP-parallel AVX2 functions")
    args = ap.parse_args()

    text = args.fp32_c.read_text(encoding="utf-8")
    if "float*" not in text:
        raise ValueError("input translation unit is not FP32")
    text = replace_function(text, "mm_fwd", OMP_MM_FWD if args.openmp else MM_FWD)
    text = replace_function(text, "mm_bwd", OMP_MM_BWD if args.openmp else MM_BWD)
    if "#include <immintrin.h>" not in text:
        text = "#include <immintrin.h>\n" + text
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    mode = "OpenMP AVX2" if args.openmp else "AVX2"
    print(f"Wrote {args.out} with {mode} FP32 mm_fwd/mm_bwd")


if __name__ == "__main__":
    main()
