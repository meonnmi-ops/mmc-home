#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
V=${1:-runtime_pipeline}

./build/mmc_v13 src/dataset_pipeline.mmc build/${V}.c
sed '/#include "runtime_v5.h"/d; /#include "runtime_glue.h"/d' build/${V}.c > build/${V}_s.c
python3 - "$V" << 'PY'
import sys
from pathlib import Path
v = sys.argv[1]
p = Path('build') / f'{v}_s.c'
code = p.read_text()
s = code.find('int main(int argc, char** argv)')
if s >= 0:
    bs = code.find('{', s); d=0; e=-1
    for i in range(bs, len(code)):
        if code[i]=='{': d+=1
        elif code[i]=='}':
            d-=1
            if d==0: e=i+1; break
    p.write_text(code[:s]+code[e:])
PY

cat > tests/ext_proto.h << 'HEOF'
int json_extract_text(void* line, int len, char* out, int cap);
HEOF

# json_decode + pipeline ကို ပေါင်း .o တစ်ခုတည်း
./build/mmc_v13 src/json_decode_mmc.mmc build/json_decode.c
sed '/#include "runtime_v5.h"/d; /#include "runtime_glue.h"/d' build/json_decode.c > build/json_decode_s.c
python3 - << 'PY'
from pathlib import Path
p = Path('build/json_decode_s.c')
c = p.read_text()
s = c.find('int main(int argc, char** argv)')
if s >= 0:
    bs = c.find('{', s); d=0; e=-1
    for i in range(bs, len(c)):
        if c[i]=='{': d+=1
        elif c[i]=='}':
            d-=1
            if d==0: e=i+1; break
    p.write_text(c[:s]+c[e:])
PY

gcc -O2 -c build/json_decode_s.c -I . -o build/json_decode.o
gcc -O2 -include tests/ext_proto.h -c build/${V}_s.c -I . -o build/${V}.o
gcc -O2 tests/test_pipeline.c build/${V}.o build/json_decode.o -o build/test_pipeline -lm
echo "✅ build/test_pipeline"
