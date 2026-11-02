#!/usr/bin/env python3
"""Add batch=N gradient accumulation to build/train_full_multi.c."""
import re, sys
from pathlib import Path

BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 4
src = Path('build/train_full_multi.c')
if not src.exists(): sys.exit('ERROR: build/train_full_multi.c not found')
if BATCH <= 1:
    print(f'BATCH={BATCH}, no patching needed'); sys.exit(0)

text = src.read_text()
if 'int BATCH =' in text: sys.exit('ERROR: already patched')

def err(m): print(f'ERROR: {m}'); sys.exit(1)

# 1. BATCH constant
m = re.search(r'int N_STEPS = \d+;', text)
if not m: err('N_STEPS not found')
text = text[:m.end()] + f'\n    int BATCH = {BATCH};' + text[m.end():]

# 2. Enlarge ids/tg loads
text = text.replace('N_STEPS * T);', 'N_STEPS * BATCH * T);')

# 3. dlogits scaling (matches C format with parens)
for old, new in [('(p - 1.0) / T)', '(p - 1.0) / (T * BATCH))'),
                 ('(p / T)', '(p / (T * BATCH))')]:
    if old in text: text = text.replace(old, new, 1)

# 4. g_wte head: = to accumulate (C format: g_wte[((v * D) + d)] = s;)
if 'g_wte[((v * D) + d)] = s;' in text:
    text = text.replace(
        'g_wte[((v * D) + d)] = s;',
        'g_wte[((v * D) + d)] = (g_wte[((v * D) + d)] + s);', 1)
else:
    for line in text.split('\n'):
        if 'g_wte' in line and '= s' in line and 'calloc' not in line:
            print(f'  Found similar: {line.strip()[:120]}')
    err('g_wte head pattern not found')

# 5. g_wpe: = to accumulate (C format: g_wpe[((t * D) + d)] = dx_cur[((t * D) + d)];)
if 'g_wpe[((t * D) + d)] = dx_cur[((t * D) + d)];' in text:
    text = text.replace(
        'g_wpe[((t * D) + d)] = dx_cur[((t * D) + d)];',
        'g_wpe[((t * D) + d)] = (g_wpe[((t * D) + d)] + dx_cur[((t * D) + d)]);', 1)
else:
    err('g_wpe pattern not found')

# 6. Find outer loop body
main_idx = text.find('int main(')
if main_idx < 0: err('main not found')
i_s_loop = text.find('while (s < N_STEPS) {', main_idx)
if i_s_loop < 0: err('outer loop not found')
i_body_start = text.find('\n', i_s_loop) + 1

m_close = re.search(r'step = step \+ 1;\s*\n\s*s = s \+ 1;\s*\n\s*\}', text[i_body_start:])
if not m_close: err('outer loop close not found')
i_outer_close = i_body_start + m_close.start()
body = text[i_body_start:i_outer_close]

# 7. Split body
i_zs = body.find('zero_float(gWqkv')
if i_zs < 0: err('zero_float start not found')
i_lnfb = body.rfind('zero_float(g_lnfb')
if i_lnfb < 0: err('zero_float end not found')
i_ze = body.find('\n', i_lnfb) + 1

i_gs = body.find('gtotal = sqrt(gtotal)')
if i_gs < 0: err('gtotal sqrt not found')
m_adamw = re.search(r'i = 0;\s*\n\s*while \(\(i < L\)\)', body[i_gs:])
if not m_adamw: err('adamw section not found')
i_ad = i_gs + m_adamw.start()

chunk_c = body[i_ze:i_ad]
chunk_d = body[i_ad:]

if 'while ((t < T))' not in chunk_c: err('forward embedding not in chunk_c')

# 8. Extract declarations (strip 'float'/'double' prefix)
def strip_decl(block, var):
    pat = rf'(?:float|double)\s+{var}\s*=\s*0\.0f?;'
    m = re.search(pat, block)
    if m:
        block = block[:m.start()] + f'{var} = 0.0f;' + block[m.end():]
    return block

chunk_c = strip_decl(chunk_c, 'loss')
chunk_c = strip_decl(chunk_c, 'gtotal')

# 9. Rebuild
new_body = (
    body[i_zs:i_ze] +
    '    float loss = 0.0f;\n'
    '    float gtotal = 0.0f;\n'
    '    float loss_total = 0.0f;\n'
    '    int b = 0;\n'
    '    while (b < BATCH) {\n'
    '    t = 0;\n'
    '    while (t < T) { ids[t] = ids_multi[(s*BATCH + b)*T + t]; tg[t] = tg_multi[(s*BATCH + b)*T + t]; t = t + 1; }\n'
    '    t = 0;\n' +
    chunk_c +
    '    loss_total = loss_total + loss;\n'
    '    b = b + 1;\n'
    '    }\n'
    '    loss = loss_total / BATCH;\n' +
    chunk_d
)

new_text = text[:i_body_start] + new_body + text[i_outer_close:]

dst = Path('build/train_full_multi_b4.c')
dst.write_text(new_text)
print(f'✓ Wrote {dst}: {len(text)}→{len(new_text)} bytes, BATCH={BATCH}')
