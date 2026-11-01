#!/usr/bin/env python3
"""Add OpenMP parallel-for to adamw() in MMC-generated C.

Transforms:
    int i = 0;
    while ((i < n)) {
    ...body...
    i = (i + 1);
    }
into:
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
    ...body...
    }

Idempotent: running twice is safe.
"""
import re
import sys
from pathlib import Path


def patch(text):
    if '#pragma omp parallel for' in text and 'for (int i = 0; i < n; i++)' in text:
        return text, 'already'

    func_re = re.compile(r'int\s+adamw\s*\([^)]*\)\s*\{')
    fm = func_re.search(text)
    if not fm:
        return text, 'no-adamw'

    loop_re = re.compile(
        r'int\s+i\s*=\s*0;\s*\n'
        r'(\s*)while\s*\(\(i\s*<\s*n\)\)\s*\{\n'
        r'(.*?)\n'
        r'\s*i\s*=\s*\(i\s*\+\s*1\);\s*\n'
        r'\}',
        re.DOTALL,
    )
    lm = loop_re.search(text, fm.end())
    if not lm:
        return text, 'no-loop'

    between = text[fm.end():lm.start()]
    if 'ld64' in between or 'sv64' in between:
        return text, 'wrong-func'

    indent = lm.group(1)
    body = lm.group(2)
    new = (
        '#pragma omp parallel for\n'
        f'{indent}for (int i = 0; i < n; i++) {{\n'
        f'{body}\n'
        f'{indent}}}'
    )
    return text[:lm.start()] + new + text[lm.end():], 'patched'


def main():
    if len(sys.argv) < 2:
        print('Usage: add_omp_to_adamw.py <file.c> [output.c]')
        sys.exit(2)
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src
    if not src.exists():
        print(f'ERROR: {src} not found')
        sys.exit(1)

    text = src.read_text()
    new_text, status = patch(text)

    if status == 'patched':
        dst.write_text(new_text)
        print(f'✓ Patched {dst}: #pragma omp parallel for added to adamw()')
    elif status == 'already':
        print(f'• {dst}: already patched, no change')
    else:
        print(f'✗ ERROR: {status} (file unchanged)')
        sys.exit(1)


if __name__ == '__main__':
    main()
