#!/bin/bash
cd "$(dirname "$0")/.."
mkdir -p build/audit
C="build/mmc_v1"

test() {
    local n="$1"; local code="$2"
    echo "─── $n ───"
    printf '%s\n' "$code" > build/audit/$n.mmc
    $C build/audit/$n.mmc build/audit/$n.c 2>&1 | grep -E "❌|error" | head -2
    if [ -f "build/audit/$n.c" ]; then
        echo "   ✅ codegen OK"
        head -10 "build/audit/$n.c"
    else
        echo "   ❌ no output"
    fi
    echo ""
}

test pointer 'တည်ဆောက် ကိန်း x = 5
တည်ဆောက် ကိန်း* p = ♮x'

test struct 'အစုအဝေး P { ကိန်း x }'

test extern 'extern "C" လုပ်ငန်း malloc(ကိန်း n) စာသား'

test array 'တည်ဆောက် ကိန်း arr[5]'
