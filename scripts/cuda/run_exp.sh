#!/bin/bash
# usage: run_exp.sh D DFF H LR
D=$1; DFF=$2; H=$3; LR=$4
cd ~/mmc_wsl/mmc_publish
python3 /tmp/gen_init.py $D $DFF || exit 1
cp scripts/cuda/test_train_step.cu scripts/cuda/ts_exp.cu
sed -i "s/const int T=32, D=32, DFF=64, H=2,/const int T=32, D=$D, DFF=$DFF, H=$H,/; s/adamw_lr = 1e-3f/adamw_lr = ${LR}f/" scripts/cuda/ts_exp.cu
grep -n "const int T=\|adamw_lr = " scripts/cuda/ts_exp.cu | head -2
rm -f /tmp/loss_*.bin
nvcc -O3 -arch=sm_75 -DUSE_LR_SCHEDULE=1 -DN_STEPS=2000 -DWARMUP=100 scripts/cuda/ts_exp.cu /tmp/block_ops.o -o /tmp/ts_exp -lcublas 2>&1 | grep -E "error"
( time /tmp/ts_exp > /dev/null ) 2>&1 | grep real
python3 /tmp/read_loss.py 2000
