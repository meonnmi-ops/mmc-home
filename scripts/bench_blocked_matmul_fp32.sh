#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CC="${CC:-gcc}"
OBJCOPY="${OBJCOPY:-objcopy}"
COMMON_FLAGS=(-O3 -march=native -funroll-loops -I build -I runtime)

for file in build/block_lib.c build/block_lib_fp32_blocked.o build/stubs.o; do
    if [[ ! -f "$file" ]]; then
        printf 'Missing required file: %s\n' "$file" >&2
        exit 1
    fi
done

if ! grep -Eq 'int[[:space:]]+mm_fwd\(float[[:space:]]*\*' build/block_lib.c; then
    printf 'build/block_lib.c is not the expected FP32 baseline.\n' >&2
    exit 1
fi

"$CC" "${COMMON_FLAGS[@]}" -c build/block_lib.c \
    -o build/block_lib_fp32_baseline_probe.o
"$OBJCOPY" --redefine-sym main=mmc_block_lib_auto_main \
    build/block_lib_fp32_baseline_probe.o
"$CC" "${COMMON_FLAGS[@]}" -c scripts/bench_blocked_matmul_fp32.c \
    -o build/bench_blocked_matmul_fp32.o

"$CC" -O3 -march=native build/bench_blocked_matmul_fp32.o \
    build/block_lib_fp32_baseline_probe.o build/stubs.o \
    -o build/bench_matmul_fp32_baseline -lm
"$CC" -O3 -march=native build/bench_blocked_matmul_fp32.o \
    build/block_lib_fp32_blocked.o build/stubs.o \
    -o build/bench_matmul_fp32_blocked -lm

printf '\n=== Baseline FP32 ===\n'
./build/bench_matmul_fp32_baseline build/matmul_fp32_baseline.bin \
    | tee build/matmul_fp32_baseline.log
printf '\n=== Blocked FP32 ===\n'
./build/bench_matmul_fp32_blocked build/matmul_fp32_blocked.bin \
    | tee build/matmul_fp32_blocked.log

printf '\n=== Compare outputs and speed ===\n'
python3 scripts/compare_blocked_matmul_fp32.py \
    build/matmul_fp32_baseline.bin build/matmul_fp32_blocked.bin \
    --baseline-log build/matmul_fp32_baseline.log \
    --blocked-log build/matmul_fp32_blocked.log

printf '\nProbe files are under build/. Training weights and checkpoints were not used.\n'
