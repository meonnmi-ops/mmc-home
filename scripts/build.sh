#!/bin/sh
# scripts/build.sh v2 — fixpoint convergence loop (behavior patch ပြီးရင် ၂ ခေါက် ကြာနိုင်)
set -e
cd "$(dirname "$0")/.."
./bin/mmc src/compiler_v14.mmc build/fix_a.c
n=1
while [ $n -le 4 ]; do
  gcc -O2 -I runtime build/fix_a.c runtime/runtime_v5.c runtime/runtime_glue.c -o build/mmc_stage -lm
  ./build/mmc_stage src/compiler_v14.mmc build/fix_b.c
  if diff -q build/fix_a.c build/fix_b.c > /dev/null 2>&1; then
    echo "FIXPOINT OK — iteration $n"
    cp build/mmc_stage bin/mmc
    sha256sum bin/mmc
    exit 0
  fi
  echo "iteration $n — ကွဲနေသေး — နောက်အကျော့ (ပုံမှန်)..."
  cp build/fix_b.c build/fix_a.c
  n=$((n+1))
done
echo "FIXPOINT FAIL — ၄ ခေါက်မှာလည်း မရောက်"
exit 1
