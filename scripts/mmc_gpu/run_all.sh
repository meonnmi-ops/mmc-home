#!/bin/bash
# run_all.sh — GPU test အားလုံး တစ်ခါပြေး (MMC — No Fake No Simulate)
# သုံးပုံ: bash scripts/mmc_gpu/run_all.sh
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO" || exit 1
OUT=/tmp/mmc_gpu_all
mkdir -p "$OUT"
rm -f /tmp/ALL_GPU_OK
pass=0
fail=0
failed=""
for f in scripts/mmc_gpu/*_test.mmc; do
  name=$(basename "$f" .mmc)
  printf '── %s ──\n' "$name"
  if ./bin/mmc "$f" "$OUT/$name.c" > "$OUT/$name.log" 2>&1 \
     && cp "$OUT/$name.c" "$OUT/$name.cu" \
     && nvcc -O2 -arch=sm_75 -I runtime "$OUT/$name.cu" runtime/runtime_v5.c runtime/runtime_glue.c runtime/runtime_gpu.cu -o "$OUT/$name" -lm 2>> "$OUT/$name.log" \
     && "$OUT/$name" >> "$OUT/$name.log" 2>&1; then
    echo "   PASS"
    pass=$((pass+1))
  else
    echo "   FAIL (log: $OUT/$name.log)"
    fail=$((fail+1))
    failed="$failed $name"
  fi
done
echo ""
echo "PASS: $pass / $((pass+fail))"
if [ "$fail" -eq 0 ]; then
  touch /tmp/ALL_GPU_OK
  echo "ALL GPU TESTS PASS"
else
  echo "FAIL:$failed"
fi
