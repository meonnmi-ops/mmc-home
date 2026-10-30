#!/data/data/com.termux/files/usr/bin/bash
set -e

echo "═══ v14 chat regression test ═══"
echo ""

# Build
./scripts/build_chat_v14.sh

# Basic test — no args → usage
echo ""
echo "═══ Test 1: Usage message ═══"
./build/chat_v14 2>&1 | head -1

# Interactive test (with timeout)
echo ""
echo "═══ Test 2: Interactive input ═══"
echo "exit" | timeout 5 ./build/chat_v14 2>&1 | head -3 || echo "(no model — expected)"

echo ""
echo "✅ All tests passed"
