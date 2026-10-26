#!/bin/bash
set -e
echo "=== MMC Self-Hosting Verification ==="
cd "$(dirname "$0")/.."

# Build directory
mkdir -p build

echo ""
echo "[1/5] Build Stage 1 (from C source)"
gcc bootstrap/compiler_v11_stage0.c \
    runtime/runtime_v5.c \
    runtime/runtime_glue.c \
    -I runtime -o build/mmc_v1 -lm 2>&1 | grep -E "error" || true
if [ ! -f build/mmc_v1 ]; then
    echo "❌ Stage 1 build FAILED"
    exit 1
fi
echo "✓ build/mmc_v1 ($(ls -la build/mmc_v1 | awk '{print $5}') bytes)"

echo ""
echo "[2/5] Stage 1 compiles source → C1.c"
./build/mmc_v1 src/compiler_v13.mmc build/C1.c 2>&1 | tail -3
echo "✓ C1.c: $(wc -c < build/C1.c) bytes"

echo ""
echo "[3/5] Build Stage 2 (from C1.c)"
gcc build/C1.c \
    runtime/runtime_v5.c \
    runtime/runtime_glue.c \
    -I runtime -o build/mmc_v2 -lm 2>&1 | grep -E "error" || true
if [ ! -f build/mmc_v2 ]; then
    echo "❌ Stage 2 build FAILED"
    exit 1
fi
echo "✓ build/mmc_v2 ($(ls -la build/mmc_v2 | awk '{print $5}') bytes)"

echo ""
echo "[4/5] Stage 2 compiles source → C2.c"
./build/mmc_v2 src/compiler_v13.mmc build/C2.c 2>&1 | tail -3
echo "✓ C2.c: $(wc -c < build/C2.c) bytes"

echo ""
echo "[5/5] Fixpoint check"
if diff -q build/C1.c build/C2.c > /dev/null 2>&1; then
    echo ""
    echo "✅✅✅ FIXPOINT VERIFIED — SELF-HOSTING PROVEN ✅✅✅"
    echo ""
    sha256sum build/C1.c build/C2.c
else
    echo "❌ NOT FIXPOINT"
    echo "Diff lines: $(diff build/C1.c build/C2.c | wc -l)"
    diff build/C1.c build/C2.c | head -20
    exit 1
fi
