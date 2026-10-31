#!/bin/bash
# build_train_full.sh — compile src/train_full.mmc -> build/train_full
set -e
cd ~/mmc_publish

echo "[1/4] MMC -> C"
./mmc_v14 src/train_full.mmc build/train_full.c 2>&1 | grep -E "Parse|Generated|Error" || true

echo "[2/4] include block_lib.h"
sed -i '1i #include "block_lib.h"' build/train_full.c

echo "[3/4] compile object"
gcc -O2 -I runtime -I build -c build/train_full.c -o build/train_full.o

echo "[4/4] link"
gcc -O2 build/train_full.o build/block_lib.o build/stubs.o -o build/train_full -lm

echo ""
echo "=== build/train_full ready (4.84M params, T=128, D=256, L=6) ==="
