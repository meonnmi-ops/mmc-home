#!/bin/bash
# usage: run_acc.sh D DFF H LR N BATCH NV [RUN_NAME]
#   RUN_NAME default: run_F (uses experiments/run_F/data/)
D=$1; DFF=$2; H=$3; LR=$4; N=$5; B=$6; NV=${7:-256}; RUN=${8:-run_F}
DATA="experiments/$RUN/data"

cd ~/mmc_wsl/mmc_publish
mkdir -p "$DATA/tmp"
export TMPDIR=$PWD/$DATA/tmp

echo "[run_acc] config: D=$D DFF=$DFF H=$H LR=$LR N=$N B=$B NV=$NV RUN=$RUN"

# --- Generate data ---
python3 scripts/cuda/gen_batches.py $N $B $NV "$DATA" || exit 1
python3 scripts/cuda/gen_init.py $D $DFF "$DATA" || exit 1

# --- Copy source & apply per-run paths ---
cp scripts/cuda/test_train_step.cu scripts/cuda/ts_exp.cu
sed -i "s|experiments/run_F/data/|$DATA/|g" scripts/cuda/ts_exp.cu
sed -i "s/const int T=32, D=32, DFF=64, H=2,/const int T=32, D=$D, DFF=$DFF, H=$H,/; s/adamw_lr = 1e-3f/adamw_lr = ${LR}f/" scripts/cuda/ts_exp.cu

# --- Compile ---
rm -f "$DATA/loss_*.bin"
nvcc -O3 -arch=sm_75 -DUSE_LR_SCHEDULE=1 -DN_STEPS=$N -DBATCH=$B -DNVAL=$NV -DWARMUP=$((N/20)) \
  scripts/cuda/ts_exp.cu "$DATA/block_ops.o" -o "$DATA/ts_exp" -lcublas 2>&1 | grep -E "error"
[ -f "$DATA/ts_exp" ] || { echo "COMPILE FAILED"; exit 1; }

# --- Run ---
( time "$DATA/ts_exp" > /dev/null ) 2>&1 | grep real

# --- Read results ---
python3 scripts/cuda/read_loss.py $N $NV "$DATA"
rm -f scripts/cuda/ts_exp.cu
