#!/usr/bin/env python3
# wsl_gen_l3scale.py — task-ae (WSL) — dims scale-up of the task-ad L=3 chain.
# Base = scripts/mmc_gpu/block_l3chain_test.mmc (task-ad main, 4272 lines,
# sha f409b517…). In-line transform only — line count stays 4272:
#   dims      : T 8→16, K 8→16, d 8→16, H 32→64, V 11→32
#   loop bounds: 64→256, 88→512, 256→1024, 128→512, 176→1024, 512→2048
#   adamw_k   : 64→256 (qkv,wpe), 88→512 (wte), 256→1024 (w1,w2)
#   gelu      : 256→1024 (fwd 2pi, bwd 3p3i)
#   data      : (3t+1) mod 11 → mod V  (8 sites; hWk init %11 kept)
#   anchor    : step-1 CE ln(11)=2.397895 → ln(32)=3.465736
#   MATCH/LD  : rewritten (17 AdamW matrices — task-ad banner said 18, actual
#               adamw_k launches are 17 — and gate-eval counts MEASURED by
#               gate-reach instrumentation on the CPU stub: 167139 total)
#   allocs/copies/strides/sample idx: KEPT (all ≥ new naturals — verified)
# Modes: main | nc1 (mirror CE-grad-scale break) | nc2 (mirror chain-splice break)
# Usage: wsl_gen_l3scale.py [mode] [ANCHOR]
import hashlib, re, sys

MODE = sys.argv[1] if len(sys.argv) > 1 else 'main'
ANCHOR = sys.argv[2] if len(sys.argv) > 2 else '3.465736'
SRC = 'scripts/mmc_gpu/block_l3chain_test.mmc'
DST = 'scripts/mmc_gpu/block_l3scale_test.mmc'
if MODE != 'main':
    DST = '/tmp/ae_nc.mmc'
SRC_SHA = 'f409b517406509e577c01ff1fc67b50cb5f8c9c5b86e4632dfe3a70f5875c0ee'

text = open(SRC, encoding='utf-8').read()
lines = text.split('\n')
assert hashlib.sha256(open(SRC, 'rb').read()).hexdigest() == SRC_SHA, 'task-ad base sha changed'
assert len(lines) == 4273 and lines[-1] == '', f'base line count {len(lines)}'

cnt = {'dim': 0, 'lb': 0, 'aw': 0, 'gelu': 0, 'mod': 0, 'mc': 0, 'rs': 0,
       'anchor': 0, 'match': 0, 'ld': 0}
out = []
for l in lines:
    s = l
    # 1) dim variables
    m = re.match(r'^    တည်ဆောက် ကိန်း (T|K|d|H|V) = (8|32|11)$', s)
    if m:
        newv = {'T': '16', 'K': '16', 'd': '16', 'H': '64', 'V': '32'}[m.group(1)]
        s = f'    တည်ဆောက် ကိန်း {m.group(1)} = {newv}'
        cnt['dim'] += 1
    # 2) loop bounds
    mlb = re.match(r'^(\s*)ပြည့်အောင်\((\w+) < (64|88|256|128|176|512|9)\)( \{)?$', s)
    if mlb:
        var, b = mlb.group(2), mlb.group(3)
        if b == '9':
            assert var == 'step', f'unexpected <9 loop var {var}'
        else:
            newb = {'64': '256', '88': '512', '256': '1024',
                    '128': '512', '176': '1024', '512': '2048'}[b]
            s = f'{mlb.group(1)}ပြည့်အောင်({var} < {newb}){mlb.group(4) or ""}'
            cnt['lb'] += 1
    # 3) adamw_k sizes (launch lines only — kernel definition untouched)
    #    atomic single-pass regex: all three numbers must match the same literal
    if 'adamw_k' in s and 'launch' in s:
        _AW = {'64': '256', '88': '512', '256': '1024'}
        s2 = re.sub(r', (64|88|256), (step|t), 0, \1, \1\)',
                    lambda m: f", {_AW[m.group(1)]}, {m.group(2)}, 0, {_AW[m.group(1)]}, {_AW[m.group(1)]})", s)
        assert s2 != s, f'adamw line without size pattern: {s[:90]}'
        s = s2
        cnt['aw'] += 1
    # 3b) malloc/copy sizes — the runtime API takes BYTES (natural doubles × 8):
    #     64→128 (gamma/beta d), 512→2048 (T×K/T×d/K×d/T×T), 704→4096 (T×V/V×K),
    #     1024→4096 (m+v qkv/wpe), 1408→8192 (m+v wte/wo), 2048→8192 (d×H/T×H),
    #     4096→16384 (m+v W1/W2)
    if ('mmc_malloc(' in s or 'mmc_gpu_malloc(' in s or
            'mmc_gpu_copy_to(' in s or 'mmc_gpu_copy_from(' in s):
        s2 = re.sub(r'\b(64|512|704|1024|1408|2048|4096)\)(\s*)$',
                    lambda m: {'64': '128', '512': '2048', '704': '4096',
                               '1024': '4096', '1408': '8192', '2048': '8192',
                               '4096': '16384'}[m.group(1)] + ')' + m.group(2), s)
        assert s2 != s, f'malloc/copy line without size pattern: {s[:90]}'
        s = s2
        cnt['mc'] += 1
    # 4) gelu launches (launch lines only — kernel definition untouched)
    if ('gelu_kernel' in s or 'gelu_bwd' in s) and 'launch' in s:
        s2 = s.replace(', 256, 1, 256)', ', 1024, 1, 1024)')
        s2 = s2.replace(', 256, 0, 0, 1, 256)', ', 1024, 0, 0, 1, 1024)')
        assert s2 != s, f'gelu line without 256 pattern: {s[:90]}'
        s = s2
        cnt['gelu'] += 1
    # 5) data pattern (3t+1) mod 11 → mod V
    if '(3 *' in s and '% 11' in s:
        s = s.replace('% 11', '% V')
        cnt['mod'] += 1
    # 5b) rowsum row-7 offsets — formula-based (row 7 of T×T attention = 7*T,
    #     row 7 of T×V probs = 7*V) — was literal 56 / 77 at T=8, V=11
    if '[56 +' in s:
        s = re.sub(r'\[56 \+', '[7 * T +', s)
        cnt['rs'] += 1
    if '[77 +' in s:
        s = re.sub(r'\[77 \+', '[7 * V +', s)
        cnt['rs'] += 1
    # 6) anchor
    if 'eAnch = lgpu - 2.397895' in s:
        s = s.replace('eAnch = lgpu - 2.397895', f'eAnch = lgpu - {ANCHOR}')
        cnt['anchor'] += 1
    out.append(s)

text = '\n'.join(out)
assert cnt == {'dim': 5, 'lb': 125, 'aw': 17, 'gelu': 6, 'mod': 8, 'mc': 698, 'rs': 6,
               'anchor': 1, 'match': 0, 'ld': 0}, f'inline counts {cnt}'

# 7) MATCH line (whole-line replace)
mi = [k for k, l in enumerate(out) if l.startswith('        printf("BLOCK SEQCE MATCH')]
assert len(mi) == 1, f'MATCH lines {len(mi)}'
MATCH_NEW = ('        printf("BLOCK SEQCE MATCH — L=3 chained three-block deep supervision '
             'in pure MMC on GPU at scaled dims: forward chain dx0 → block-1 → block-2 → block-3 '
             '(each block = attention + FFN + residual + centered LN, shared gamma/beta) — '
             'trainable embedding (wte V×K one-hot-gathered via mm_k: E = M@wte + MP@wpe) + '
             'tied output head (logits = H @ wte^T via dEt = tpose_rc(dE), ALL THREE blocks deep '
             'supervision, loss = CE1+CE2+CE3) — CE loss = -(1/T)·Σ_t log P[t,y_t] with '
             'shift-by-one token targets (x_t=(3t+1) mod 32, y_t = x_((t+1) mod T)) — '
             'step-1 CE = ln(32) byte-exact per block (uniform softmax, separable wte init + '
             'centered LN) — CE gradient (P−Y)/T computed on host from device softmax — '
             'wte gradient = embedding path (M^T@dX0) + block-1 head path (dLg1^T@H3) + '
             'block-2 head path (dLg2^T@H3B) + block-3 head path (dLg3^T@H3C) summed on host; '
             'wpe gradient = MP^T@dX0 — L=3 chain fully spliced: block-3 backward dX2 → '
             'block-2 dh3gB splice before LN-backward → dX1 → block-1 dh3g splice → dX0 — '
             'mirror chain identical (rdXtC → rdH3B splice → rdXtB → rdH3 splice → rdX0) — '
             '17 AdamW matrices ({Wq Wk Wv W1 W2}×3 + wte + wpe) lrs=1/step persistent TRUE m/v — '
             '167139 gate-condition evaluations over the green 8-step run (29742 forward '
             'elementwise + 81426 backward elementwise + 48609 optimizer-state elementwise + '
             '5811 weight-update magnitude checks + 384 masked-zero + 1152 rowsum + 6 loss-sync '
             '+ 8 anchor + 1 LOSS DECREASED; green-run gate-reach counts, stub-measured) '
             'all GPU = CPU (T=16, K=16, d=16, H=64, V=32) — zero new kernels — locked kernels '
             'only — dims scale-up of the L=3 chain proven in pure MMC on GPU\\n")')
out[mi[0]] = MATCH_NEW
cnt['match'] = 1

# 8) LOSS DECREASED line: 18 → 17 matrices
li = [k for k, l in enumerate(out) if l.startswith('        printf("LOSS DECREASED')]
assert len(li) == 1, f'LOSS DECREASED lines {len(li)}'
assert out[li[0]].count('18 matrices') == 1
out[li[0]] = out[li[0]].replace('18 matrices', '17 matrices')
cnt['ld'] = 1

text = '\n'.join(out)

# 9) NC modes
if MODE == 'nc1':
    NC = ('                rdLC[q1C] = (rP3[q1C] - hY[q1C]) / T',
          '                rdLC[q1C] = (rP3[q1C] - hY[q1C])')
elif MODE == 'nc2':
    NC = ('                rdH3B[hc4] = rdH3B[hc4] + rdXtC[hc4]',
          '                rdH3B[hc4] = rdH3B[hc4] + 0.0 * rdXtC[hc4]')
if MODE != 'main':
    assert text.count(NC[0]) == 1, f'NC pattern count={text.count(NC[0])}'
    text = text.replace(NC[0], NC[1])

open(DST, 'w').write(text)
n = len(text.split('\n')) - 1  # trailing newline
assert n == 4272, f'unexpected line count {n}'
print(f'GEN_OK mode={MODE} lines={n} bytes={len(text.encode()) + 1}')
print(f'DST_SHA={hashlib.sha256(open(DST, "rb").read()).hexdigest()}')
print(f'SHA16={hashlib.sha256(open(DST, "rb").read()).hexdigest()[:16]}')
