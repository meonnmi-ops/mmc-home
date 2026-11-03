#!/usr/bin/env python3
# fix_entry_template.py — restore working main emission in compiler template (fix lives in MMC source, no C shim)
# Re-runnable: 2nd run reports "already removed / already in template".
import sys
from pathlib import Path

SRC = Path("src/compiler_v14.mmc")
s = SRC.read_text(encoding="utf-8")
lines = s.split("\n")
report = []

def find_hdr(ls):
    return next((i for i, l in enumerate(ls) if '"int အဓိက(void);' in l), None)

hdr = find_hdr(lines)
if hdr is None:
    print("FATAL: template anchor not found — paste sed -n '1890,1910p' output")
    sys.exit(1)

# 1. recover the PROVEN argv-push line from a working generated C
push_c = None
for cand in ["build/mmc_ဖိုင်ရေး.c", "build/mmc_ဖတ်.c", "build/mmc_ရေ.c"]:
    p = Path(cand)
    if p.exists():
        for l in p.read_text(encoding="utf-8").split("\n"):
            if "mmc_array_push(knowledge_base" in l and "argv" in l:
                push_c = l.strip()
                break
    if push_c:
        report.append("push line recovered from " + cand)
        break
if push_c is None:
    push_c = "for (int _i = 0; _i < argc; _i++) mmc_array_push(knowledge_base, argv[_i]);"
    report.append("push line not recovered — using default")

# 2. delete the bogus template line:  output = concat(output, "    return အဓိက();\n")
del_idx = next((i for i in range(hdr, min(hdr + 12, len(lines)))
                if "return အဓိက()" in lines[i] and "mmc_string_concat" in lines[i]), None)
if del_idx is not None:
    report.append("DELETED L" + str(del_idx + 1) + ": " + lines[del_idx].strip())
    del lines[del_idx]
else:
    report.append("bogus return line: not found (already removed?)")

# 3. insert argv push after kb-init line (only if template lacks it)
hdr = find_hdr(lines)
win = range(hdr, min(hdr + 12, len(lines)))
if not any("mmc_array_push(knowledge_base" in lines[i] for i in win):
    kb = next((i for i in win if "mmc_array_new()" in lines[i] and "mmc_string_concat" in lines[i]), None)
    ins = '    output = mmc_string_concat(output, "    ' + push_c + '\\n")'
    at = (kb + 1) if kb is not None else (hdr + 1)
    lines.insert(at, ins)
    report.append("INSERTED argv push after L" + str(at))
else:
    report.append("argv push already in template — skipped")

SRC.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(report))
print("bytes:", len(s.encode("utf-8")), "->", len(SRC.read_text(encoding="utf-8").encode("utf-8")))
h2 = find_hdr(lines)
print("--- template region now ---")
print("\n".join(lines[h2:h2 + 10]))
