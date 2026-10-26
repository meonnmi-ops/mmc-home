#!/usr/bin/env bash
# Build pure-MMC nyanlin runtime with -O2.  usage: scripts/build.sh [tag]
set -e
cd "$(dirname "$0")/.."
V=${1:-runtime_current}
./build/mmc_v13 src/runtime_mmc.mmc build/${V}.c
sed '/#include "runtime_v5.h"/d; /#include "runtime_glue.h"/d' build/${V}.c > build/${V}_standalone.c
python3 - "$V" << 'PY'
import sys
from pathlib import Path
v = sys.argv[1]
p = Path('build') / f'{v}_standalone.c'
code = p.read_text()
s = code.find('int main(int argc, char** argv)')
if s >= 0:
    bs = code.find('{', s); d = 0; e = -1
    for i in range(bs, len(code)):
        if code[i] == '{': d += 1
        elif code[i] == '}':
            d -= 1
            if d == 0: e = i + 1; break
    p.write_text(code[:s] + code[e:])
    print('✅ main stripped')
PY
gcc -O2 -c build/${V}_standalone.c -I . -o build/${V}.o
gcc -O2 src/cli.c build/${V}.o -o build/nyanlin -lm
echo "✅ build/nyanlin  (MMC $V, -O2)"
