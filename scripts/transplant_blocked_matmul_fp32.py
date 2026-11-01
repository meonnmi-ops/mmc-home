#!/usr/bin/env python3
"""Replace only mm_fwd/mm_bwd in the existing FP32 generated C file.

The optimized implementations are generated from block_lib.mmc. This helper
transplants those two functions into the existing FP32 C translation unit,
changing only their local/signature scalar type from double to float.
"""
import argparse
import re
from pathlib import Path


def function_span(text, name):
    pattern = re.compile(rf"\bint\s+{re.escape(name)}\s*\(")
    for match in pattern.finditer(text):
        brace = text.find("{", match.end())
        semi = text.find(";", match.end())
        if brace < 0 or (semi >= 0 and semi < brace):
            continue

        depth = 0
        in_string = False
        in_char = False
        in_line_comment = False
        in_block_comment = False
        escaped = False
        i = brace
        while i < len(text):
            ch = text[i]
            nxt = text[i + 1] if i + 1 < len(text) else ""

            if in_line_comment:
                if ch == "\n":
                    in_line_comment = False
            elif in_block_comment:
                if ch == "*" and nxt == "/":
                    in_block_comment = False
                    i += 1
            elif in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif in_char:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == "'":
                    in_char = False
            elif ch == "/" and nxt == "/":
                in_line_comment = True
                i += 1
            elif ch == "/" and nxt == "*":
                in_block_comment = True
                i += 1
            elif ch == '"':
                in_string = True
            elif ch == "'":
                in_char = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return match.start(), i + 1
            i += 1

    raise ValueError(f"function definition not found: {name}")


def get_function(text, name):
    start, end = function_span(text, name)
    return text[start:end]


def replace_function(text, name, replacement):
    start, end = function_span(text, name)
    return text[:start] + replacement + text[end:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-c", required=True,
                        help="C generated from the blocked MMC source")
    parser.add_argument("--fp32-c", required=True,
                        help="existing FP32 C translation unit")
    parser.add_argument("--out", required=True,
                        help="new candidate FP32 C translation unit")
    args = parser.parse_args()

    native = Path(args.native_c).read_text(encoding="utf-8")
    fp32 = Path(args.fp32_c).read_text(encoding="utf-8")
    replacements = {}

    for name in ("mm_fwd", "mm_bwd"):
        old = get_function(fp32, name)
        new = get_function(native, name)
        if "float*" not in old:
            raise ValueError(f"baseline {name} is not an FP32 function")
        new = re.sub(r"\bdouble\b", "float", new)
        if "double*" in new or "float*" not in new:
            raise ValueError(f"failed to lower {name} pointers to float")
        replacements[name] = new

    for name in ("mm_fwd", "mm_bwd"):
        fp32 = replace_function(fp32, name, replacements[name])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(fp32, encoding="utf-8")
    print(f"Wrote {out} with blocked FP32 mm_fwd/mm_bwd")


if __name__ == "__main__":
    main()
