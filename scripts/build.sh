#!/bin/sh
# scripts/build.sh — MMC compiler rebuild + fixpoint verify (proven chain)
set -e
cd "$(dirname "$0")/.."
./bin/mmc src/compiler_v14.mmc build/compiler_gen.c
gcc -O2 -I runtime build/compiler_gen.c runtime/runtime_v5.c runtime/runtime_glue.c -o build/mmc_build -lm
./build/mmc_build src/compiler_v14.mmc build/compiler_verify.c
diff -q build/compiler_gen.c build/compiler_verify.c || { echo "FIXPOINT FAIL"; exit 1; }
cp build/mmc_build bin/mmc
echo "=== bin/mmc updated — FIXPOINT OK ==="
