#!/usr/bin/env python3
"""Patch src/train_full.mmc -> src/train_full_multi.mmc"""
import sys
from pathlib import Path

n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 100

def find_main_bounds(text):
    start = text.find('လုပ်ငန်း main() {')
    if start < 0:
        return -1, -1
    depth = 0
    i = start
    while i < len(text):
        if text[i] == '{': depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0: return start, i + 1
        i += 1
    return -1, -1

def patch(m, n):
    # 1. N_STEPS after H = 8
    if 'ကိန်း H = 8' not in m: raise ValueError('H=8 not found')
    m = m.replace('ကိန်း H = 8', 'ကိန်း H = 8\n    တည်ဆောက် ကိန်း N_STEPS = ' + str(n), 1)

    # 2. Batch load: ids.bin -> ids_multi.bin
    old_ids = 'တည်ဆောက် ကိန်း* ids = ld32("/home/meonnmi-0ps/mmc_wsl/mmc_publish/ref/full_test/ids.bin", T)'
    old_tg = 'တည်ဆောက် ကိန်း* tg = ld32("/home/meonnmi-0ps/mmc_wsl/mmc_publish/ref/full_test/tg.bin", T)'
    if old_ids not in m or old_tg not in m: raise ValueError('ids/tg load not found')
    new_batch = (
        'တည်ဆောက် ကိန်း* ids_multi = ld32("/home/meonnmi-0ps/mmc_wsl/mmc_publish/ref/full_test/ids_multi.bin", N_STEPS*T)\n'
        '    တည်ဆောက် ကိန်း* tg_multi = ld32("/home/meonnmi-0ps/mmc_wsl/mmc_publish/ref/full_test/tg_multi.bin", N_STEPS*T)\n'
        '    တည်ဆောက် ကိန်း* ids = calloc(T, 4)\n'
        '    တည်ဆောက် ကိန်း* tg = calloc(T, 4)'
    )
    m = m.replace(old_ids, new_batch, 1)
    m = m.replace(old_tg, '', 1)

    # 3. dx_prev_buf after dx_cur
    old_dx = 'တည်ဆောက် ဒသမ* dx_cur = calloc(T*D, 8)'
    if old_dx not in m: raise ValueError('dx_cur not found')
    m = m.replace(old_dx, old_dx + '\n    တည်ဆောက် ဒသမ* dx_prev_buf = calloc(T*D, 8)', 1)

    # 4. Block loop: calloc -> zero+swap
    old_alloc = 'တည်ဆောက် ဒသမ* dx_prev = calloc(T*D, 8)\n        block_fwd_bwd(dx_prev,'
    if old_alloc not in m: raise ValueError('block calloc not found')
    m = m.replace(old_alloc, 'zero_double(dx_prev_buf, T*D)\n        block_fwd_bwd(dx_prev_buf,', 1)

    old_free = 'free(dx_cur)\n        dx_cur = dx_prev\n        i = i - 1'
    if old_free not in m: raise ValueError('free(dx_cur) not found')
    m = m.replace(old_free, 'တည်ဆောက် ဒသမ* dx_tmp = dx_cur\n        dx_cur = dx_prev_buf\n        dx_prev_buf = dx_tmp\n        i = i - 1', 1)

    # 5. Wrap step body in loop
    fwd = '// ===== Forward: xs[0] = emb'
    sv = '// ===== Save updated weights'
    if fwd not in m or sv not in m: raise ValueError('Forward/Save markers not found')
    i_f = m.find(fwd); i_s = m.find(sv)
    if i_f >= i_s: raise ValueError('marker order wrong')

    pre = m[:i_f]
    body = m[i_f:i_s]
    post = m[i_s:]

    zero_block = (
        '    // ===== Multi-step loop =====\n'
        '    တည်ဆောက် ကိန်း s = 0\n'
        '    ပြည့်အောင်(s < N_STEPS) {\n'
        '    t = 0\n'
        '    ပြည့်အောင်(t < T) {\n'
        '    ids[t] = ids_multi[s*T + t]\n'
        '    tg[t] = tg_multi[s*T + t]\n'
        '    t = t + 1\n'
        '    }\n'
    )
    for g in ['gWqkv, L*3*D*D', 'gbqkv, L*3*D', 'gWo, L*D*D', 'gbo, L*D',
              'gW1, L*DFF*D', 'gb1, L*DFF', 'gW2, L*D*DFF', 'gb2, L*D',
              'gln1w, L*D', 'gln1b, L*D', 'gln2w, L*D', 'gln2b, L*D',
              'g_wte, V*D', 'g_wpe, TMAX*D', 'g_lnfw, D', 'g_lnfb, D']:
        zero_block += '    zero_double(' + g + ')\n'

    loop_close = '\n    step = step + 1\n    s = s + 1\n    }\n\n'
    return pre + zero_block + body + loop_close + post

src = Path('src/train_full.mmc'); dst = Path('src/train_full_multi.mmc')
if not src.exists(): print('ERROR: train_full.mmc missing'); sys.exit(1)

text = src.read_text()
m_start, m_end = find_main_bounds(text)
if m_start < 0: print('ERROR: main() not found'); sys.exit(1)

before = text[:m_start]; old_main = text[m_start:m_end]; after = text[m_end:]

try:
    new_main = patch(old_main, n_steps)
except ValueError as e:
    print(f'ERROR: {e}'); sys.exit(1)

helper = (
    'လုပ်ငန်း zero_double(ဒသမ* p, ကိန်း n) ကိန်း {\n'
    '    တည်ဆောက် ကိန်း i = 0\n'
    '    ပြည့်အောင်(i < n) {\n'
    '    p[i] = 0.0\n'
    '    i = i + 1\n'
    '    }\n'
    '    ပြန်ပေး 0\n'
    '}\n\n'
)
if 'zero_double' not in before:
    before = before + helper

dst.write_text(before + new_main + after)
print(f'✓ Wrote {dst} ({len(before+new_main+after)} bytes, N_STEPS={n_steps})')
