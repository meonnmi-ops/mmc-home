#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CC="${CC:-gcc}"
OBJCOPY="${OBJCOPY:-objcopy}"

if [[ -z "${MMC_COMPILER:-}" ]]; then
    if command -v mmc_v14 >/dev/null 2>&1; then
        MMC_COMPILER="$(command -v mmc_v14)"
    elif [[ -x ./mmc_v14 ]]; then
        MMC_COMPILER=./mmc_v14
    elif [[ -x ./build/mmc_v14_native ]]; then
        MMC_COMPILER=./build/mmc_v14_native
    else
        # WSL checkout may not contain the phone's mmc_v14 binary. Bootstrap
        # a native compiler from the checked-in Stage 0 C and MMC sources.
        mkdir -p build
        "$CC" bootstrap/compiler_v11_stage0.c runtime/runtime_v5.c \
            runtime/runtime_glue.c -I runtime -o build/mmc_v11_native -lm
        ./build/mmc_v11_native src/compiler_v13.mmc build/compiler_v13_native.c
        "$CC" -O2 build/compiler_v13_native.c runtime/runtime_v5.c \
            runtime/runtime_glue.c -I runtime -o build/mmc_v13_native -lm
        ./build/mmc_v13_native src/compiler_v14.mmc build/compiler_v14_native.c
        "$CC" -O2 build/compiler_v14_native.c runtime/runtime_v5.c \
            runtime/runtime_glue.c -I runtime -o build/mmc_v14_native -lm
        MMC_COMPILER=./build/mmc_v14_native
    fi
fi

if [[ ! -x "$MMC_COMPILER" ]]; then
    printf 'MMC compiler is not executable: %s\n' "$MMC_COMPILER" >&2
    exit 1
fi

for file in build/train_full.c build/stubs.o; do
    if [[ ! -f "$file" ]]; then
        printf 'Missing required file: %s\n' "$file" >&2
        exit 1
    fi
done

"$MMC_COMPILER" src/block_lib.mmc build/block_lib_blocked_native.c
python3 scripts/transplant_blocked_matmul_fp32.py \
    --native-c build/block_lib_blocked_native.c \
    --fp32-c build/block_lib.c \
    --out build/block_lib_blocked.c

"$CC" -O3 -march=native -funroll-loops -I build -I runtime \
    -c build/block_lib_blocked.c -o build/block_lib_fp32_blocked.o
"$OBJCOPY" --redefine-sym main=mmc_block_lib_auto_main \
    build/block_lib_fp32_blocked.o

"$CC" -O3 -march=native -funroll-loops -I build -I runtime \
    -c build/train_full.c -o build/train_full_fp32_blocked.o
"$CC" -O3 -march=native build/train_full_fp32_blocked.o \
    build/block_lib_fp32_blocked.o build/stubs.o \
    -o build/train_full_fp32_blocked -lm

ls -lh build/train_full_fp32 build/train_full_fp32_blocked
printf '\nBlocked binary built. It has not been run and will not alter training state.\n'
