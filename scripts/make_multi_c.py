#!/usr/bin/env python3
"""Multi-step patch: build/train_full.c (FP32, L=3) -> build/train_full_multi.c."""
import sys
from pathlib import Path

n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 100
src = Path('build/train_full.c'); dst = Path('build/train_full_multi.c')
if not src.exists(): sys.exit('ERROR: build/train_full.c not found')
text = src.read_text(); orig_len = len(text)

# Sanity
if 'int L = 3;' not in text:
    sys.exit('ERROR: L=3 not in train_full.c (run sed first)')

# 1. N_STEPS
text = text.replace('int L = 3;', f'int L = 3;\n    int N_STEPS = {n_steps};', 1)

# FIX lr bug: ld64 -> ld32 (lr.bin converted to float32, 4 bytes)
text = text.replace(
    'float* lr_p = ld64("/home/meonnmi-0ps/mmc_wsl/mmc_publish/ref/full_test/lr.bin", 1);',
    'float* lr_p = ld32("/home/meonnmi-0ps/mmc_wsl/mmc_publish/ref/full_test/lr.bin", 1);',
    1
)

# 2. zero_float helper
if 'int zero_float(' not in text:
    helper = (
        'int zero_float(float* p, int n) {\n'
        '    int i = 0;\n'
        '    while (i < n) { p[i] = 0.0f; i = i + 1; }\n'
        '    return 0;\n'
        '}\n\n'
    )
    idx = text.find('int main(')
    if idx < 0: sys.exit('ERROR: main() not found')
    text = text[:idx] + helper + text[idx:]

# 3. ids/tg -> multi
old_ids = 'int* ids = ld32("/home/meonnmi-0ps/mmc_wsl/mmc_publish/ref/full_test/ids.bin", T);'
old_tg  = 'int* tg = ld32("/home/meonnmi-0ps/mmc_wsl/mmc_publish/ref/full_test/tg.bin", T);'
if old_ids not in text or old_tg not in text:
    sys.exit('ERROR: ids/tg load not found')
text = text.replace(old_ids,
    'int* ids_multi = ld32("/home/meonnmi-0ps/mmc_wsl/mmc_publish/ref/full_test/ids_multi.bin", N_STEPS * T);\n'
    '    int* tg_multi = ld32("/home/meonnmi-0ps/mmc_wsl/mmc_publish/ref/full_test/tg_multi.bin", N_STEPS * T);\n'
    '    int* ids = (int*)calloc(T, 4);\n'
    '    int* tg = (int*)calloc(T, 4);',
    1)
text = text.replace(old_tg, '', 1)

# 4. dx_cur: add dx_prev_buf
old_dxcur = 'float* dx_cur = calloc((T * D), 8);'
if old_dxcur not in text: sys.exit('ERROR: dx_cur not found')
text = text.replace(old_dxcur,
    old_dxcur + '\n    float* dx_prev_buf = (float*)calloc(T * D, 4);', 1)

# 5. dx_prev: calloc -> zero_float + dx_prev_buf (4 spaces indent)
old_block = '    float* dx_prev = calloc((T * D), 8);\n    block_fwd_bwd(dx_prev,'
if old_block not in text: sys.exit('ERROR: dx_prev block not found')
text = text.replace(old_block,
    '    zero_float(dx_prev_buf, T * D);\n    block_fwd_bwd(dx_prev_buf,', 1)

# 6. free/swap (4 spaces indent)
old_free = '    free(dx_cur);\n    dx_cur = dx_prev;'
if old_free not in text: sys.exit('ERROR: free/swap not found')
text = text.replace(old_free,
    '    float* dx_tmp = dx_cur;\n    dx_cur = dx_prev_buf;\n    dx_prev_buf = dx_tmp;',
    1)

# 7. Loop boundary: first "while ((t < T))" after "int main("
main_idx = text.find('int main(')
if main_idx < 0: sys.exit('ERROR: main() not found')
fwd_idx = text.find('while ((t < T))', main_idx)
if fwd_idx < 0: sys.exit('ERROR: while ((t < T)) not found')
sv_idx = text.find('sv64(', fwd_idx)
if sv_idx < 0: sys.exit('ERROR: sv64 not found')

pre = text[:fwd_idx]
body = text[fwd_idx:sv_idx]
post = text[sv_idx:]

zero_block = (
    '    // ===== Multi-step loop =====\n'
    '    int s = 0;\n'
    '    while (s < N_STEPS) {\n'
    '    t = 0;\n'
    '    while (t < T) { ids[t] = ids_multi[s*T + t]; tg[t] = tg_multi[s*T + t]; t = t + 1; }\n'
    '    t = 0;\n'
)
for g in ['gWqkv, L*3*D*D', 'gbqkv, L*3*D', 'gWo, L*D*D', 'gbo, L*D',
          'gW1, L*DFF*D', 'gb1, L*DFF', 'gW2, L*D*DFF', 'gb2, L*D',
          'gln1w, L*D', 'gln1b, L*D', 'gln2w, L*D', 'gln2b, L*D',
          'g_wte, V*D', 'g_wpe, TMAX*D', 'g_lnfw, D', 'g_lnfb, D']:
    zero_block += f'    zero_float({g});\n'

loop_close = '\n    step = step + 1;\n    s = s + 1;\n    }\n\n'

result = pre + zero_block + body + loop_close + post
dst.write_text(result)
print(f'✓ Wrote {dst}: {orig_len}→{len(result)} bytes, N_STEPS={n_steps}')
