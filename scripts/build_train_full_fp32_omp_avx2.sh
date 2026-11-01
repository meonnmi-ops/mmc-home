#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CC="${CC:-gcc}"
OBJCOPY="${OBJCOPY:-objcopy}"
OMP_FLAGS=(-O3 -mavx2 -mfma -fopenmp -funroll-loops -I build -I runtime)

for file in build/block_lib_blocked.c build/train_full.c build/stubs.o; do
    if [[ ! -f "$file" ]]; then
        printf 'Missing required file: %s\nRun: bash scripts/build_train_full_fp32_blocked.sh\n' "$file" >&2
        exit 1
    fi
done

python3 scripts/transplant_avx2_matmul_fp32.py \
    --fp32-c build/block_lib_blocked.c --openmp \
    --out build/block_lib_omp_avx2.c

"$CC" "${OMP_FLAGS[@]}" -c build/block_lib_omp_avx2.c -o build/block_lib_fp32_omp_avx2.o
"$OBJCOPY" --redefine-sym main=mmc_block_lib_auto_main build/block_lib_fp32_omp_avx2.o
"$CC" "${OMP_FLAGS[@]}" -c build/train_full.c -o build/train_full_fp32_omp_avx2.o
"$CC" -O3 -mavx2 -mfma -fopenmp build/train_full_fp32_omp_avx2.o \
    build/block_lib_fp32_omp_avx2.o build/stubs.o -o build/train_full_fp32_omp_avx2 -lm

ls -lh build/train_full_fp32_avx2 build/train_full_fp32_omp_avx2
printf '\nOpenMP+AVX2 binary built. It has not been run and will not alter training state.\n'
