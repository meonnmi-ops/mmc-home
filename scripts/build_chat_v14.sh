#!/data/data/com.termux/files/usr/bin/bash
# Build chat.mmc with v14 compiler — verified 2026-10-30
set -e

echo "═══ v14 chat.mmc build ═══"
echo ""

# Step 1: Generate C
echo "[1/3] Generating C..."
./mmc_v14 src/chat.mmc build/chat_v14.c

# Step 2: Patch headers
echo ""
echo "[2/3] Patching headers..."
sed -i '/#include "runtime_glue.h"/d' build/chat_v14.c
sed -i '/#include "runtime_v5.h"/a #include "chat_decls.h"' build/chat_v14.c

# Step 3: Compile + link
echo ""
echo "[3/3] Compiling + linking..."
gcc -O3 -I runtime -I build \
    build/chat_v14.c build/chat_loader.c build/v49_final.o \
    -o build/chat_v14 -lm

echo ""
echo "✅ Success: build/chat_v14 ($(ls -lh build/chat_v14 | awk '{print $5}'))"
echo ""
echo "Run: ./build/chat_v14 path/to/model.gguf"
