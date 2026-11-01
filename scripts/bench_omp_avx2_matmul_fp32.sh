#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CC="${CC:-gcc}"
OBJCOPY="${OBJCOPY:-objcopy}"
OMP_FLAGS=(-O3 -mavx2 -mfma -fopenmp -funroll-loops -I build -I runtime)

for file in build/block_lib_fp32_avx2.o build/block_lib_fp32_omp_avx2.o build/stubs.o; do
    if [[ ! -f "$file" ]]; then
        printf 'Missing required file: %s\nRun: bash scripts/build_train_full_fp32_omp_avx2.sh\n' "$file" >&2
        exit 1
    fi
done

"$CC" "${OMP_FLAGS[@]}" -c scripts/bench_blocked_matmul_fp32.c -o build/bench_omp_avx2_matmul_fp32.o
"$CC" -O3 -mavx2 -mfma -fopenmp build/bench_omp_avx2_matmul_fp32.o \
    build/block_lib_fp32_avx2.o build/stubs.o -o build/bench_matmul_fp32_avx2 -lm
"$CC" -O3 -mavx2 -mfma -fopenmp build/bench_omp_avx2_matmul_fp32.o \
    build/block_lib_fp32_omp_avx2.o build/stubs.o -o build/bench_matmul_fp32_omp_avx2 -lm

printf '\n=== AVX2 FP32, one thread ===\n'
OMP_NUM_THREADS=1 ./build/bench_matmul_fp32_avx2 build/matmul_fp32_avx2.bin \
    | tee build/matmul_fp32_avx2.log
printf '\n=== OpenMP + AVX2 FP32, 8 threads ===\n'
OMP_NUM_THREADS=8 ./build/bench_matmul_fp32_omp_avx2 build/matmul_fp32_omp_avx2.bin \
    | tee build/matmul_fp32_omp_avx2.log

printf '\n=== Compare outputs and speed ===\n'
python3 scripts/compare_blocked_matmul_fp32.py \
    build/matmul_fp32_avx2.bin build/matmul_fp32_omp_avx2.bin \
    --baseline-log build/matmul_fp32_avx2.log \
    --blocked-log build/matmul_fp32_omp_avx2.log \
    --baseline-label avx2 --candidate-label omp_avx2

printf '\nProbe files are under build/. Training weights and checkpoints were not used.\n'
