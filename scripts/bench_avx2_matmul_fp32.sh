#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CC="${CC:-gcc}"
OBJCOPY="${OBJCOPY:-objcopy}"
COMMON_FLAGS=(-O3 -mavx2 -mfma -funroll-loops -I build -I runtime)

for file in build/block_lib_fp32_blocked.o build/block_lib_fp32_avx2.o build/stubs.o; do
    if [[ ! -f "$file" ]]; then
        printf 'Missing required file: %s\nRun: bash scripts/build_train_full_fp32_avx2.sh\n' "$file" >&2
        exit 1
    fi
done

"$CC" "${COMMON_FLAGS[@]}" -c scripts/bench_blocked_matmul_fp32.c \
    -o build/bench_avx2_matmul_fp32.o
"$CC" -O3 -mavx2 -mfma build/bench_avx2_matmul_fp32.o build/block_lib_fp32_blocked.o build/stubs.o \
    -o build/bench_matmul_fp32_blocked -lm
"$CC" -O3 -mavx2 -mfma build/bench_avx2_matmul_fp32.o build/block_lib_fp32_avx2.o build/stubs.o \
    -o build/bench_matmul_fp32_avx2 -lm

printf '\n=== Blocked FP32 ===\n'
./build/bench_matmul_fp32_blocked build/matmul_fp32_blocked.bin \
    | tee build/matmul_fp32_blocked.log
printf '\n=== AVX2 FP32 ===\n'
./build/bench_matmul_fp32_avx2 build/matmul_fp32_avx2.bin \
    | tee build/matmul_fp32_avx2.log

printf '\n=== Compare outputs and speed ===\n'
python3 scripts/compare_blocked_matmul_fp32.py \
    build/matmul_fp32_blocked.bin build/matmul_fp32_avx2.bin \
    --baseline-log build/matmul_fp32_blocked.log \
    --blocked-log build/matmul_fp32_avx2.log \
    --baseline-label blocked --candidate-label avx2

printf '\nProbe files are under build/. Training weights and checkpoints were not used.\n'
