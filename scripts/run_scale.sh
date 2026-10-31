#!/bin/bash
# Progressively scale D and verify at each step
set -e
cd ~/mmc_publish

cp src/train_multi_step.mmc $HOME/tmp/train_multi_step.mmc.orig

for D in 20 40 60 80 100; do
    DFF=$((D * 2))
    H=4
    
    echo ""
    echo "═══════════════════════════════════════════"
    echo "  D = $D, DFF = $DFF, H = $H, HD = $((D/H))"
    echo "═══════════════════════════════════════════"
    
    # Patch config
    sed -i "s/^\(\s*တည်ဆောက် ကိန်း D\s*=\s*\)[0-9]*/\1$D/" src/train_multi_step.mmc
    sed -i "s/^\(\s*တည်ဆောက် ကိန်း DFF\s*=\s*\)[0-9]*/\1$DFF/" src/train_multi_step.mmc
    sed -i "s/^\(\s*တည်ဆောက် ကိန်း H\s*=\s*\)[0-9]*/\1$H/" src/train_multi_step.mmc
    
    # 1. Generate ref (no out_ yet)
    D=$D DFF=$DFF H=$H python3 ref/train_scale_gen.py 2>&1 | grep -E "PyTorch|waiting"
    
    # 2. Build MMC
    ./mmc_v14 src/train_multi_step.mmc build/train_multi_step.c >/dev/null 2>&1
    sed -i '1i #include "block_lib.h"' build/train_multi_step.c
    gcc -O2 -I runtime -I build -c build/train_multi_step.c -o build/train_multi_step.o
    gcc -O2 build/train_multi_step.o build/block_lib.o build/stubs.o -o build/train_multi_step -lm
    
    # 3. Run MMC
    ./build/train_multi_step 2>&1 | grep -E "loss|DONE"
    
    # 4. Compare
    D=$D DFF=$DFF H=$H python3 ref/train_scale_gen.py 2>&1 | grep -E "PASS|FAIL|missing"
done

mv $HOME/tmp/train_multi_step.mmc.orig src/train_multi_step.mmc
echo ""
echo "✅ restored original D=8"
