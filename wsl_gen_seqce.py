#!/usr/bin/env python3
# gen_seqce.py — task-ac: sequence CE / token targets in pure MMC on GPU
# Base = task-ab block_l2chain_test.mmc (2783 lines, sha 4941aa4d…).
# Transforms: trainable wte/hPE (AdamW), tied head (logits = H @ wte^T via
# dEt = tpose_rc(dE)), CE loss = -(1/T)·Σ_t log P[t,y_t] with softmax_row,
# one-hot embedding forward (M@wte + MP@wpe via mm_k), embedding backward
# (bwd_dw on one-hots), 12 AdamW matrices (Wq Wk Wv W1 W2 ×2 + wte + wpe).
# Zero new kernels. Modes: main | nc1 (mirror CE-grad-scale break) | nc2
# (mirror tied-head break).
import hashlib
import re
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else 'main'
SRC = 'scripts/mmc_gpu/block_l2chain_test.mmc'
DST = 'scripts/mmc_gpu/block_seqce_test.mmc'
SRC_SHA = '4941aa4dcefb861b48db5a22b009826615189aaad3335db74d363774dc89dc6a'
ANCHOR = '2.397895'          # step-1 block-1 CE = ln(11) (uniform softmax at init)
YT = '(3 * ((ilC + 1) % T) + 1) % 11'   # y_t = x_((t+1) mod T), x_t = (3t+1)%11

src = open(SRC).read().splitlines()
assert len(src) == 2783, f"task-ab base line count changed: {len(src)}"
assert hashlib.sha256(open(SRC, 'rb').read()).hexdigest() == SRC_SHA, "task-ab base sha changed"

def seg(a, b):
    return src[a-1:b]

def ind(lines, n=8):
    return [(' ' * n + l if l.strip() else l) for l in lines]

def sub(lines, pairs, tag=''):
    out, hits = [], {k: 0 for k, _ in pairs}
    for l in lines:
        for old, new in pairs:
            if old in l:
                l = l.replace(old, new)
                hits[old] += 1
        out.append(l)
    for old, _ in pairs:
        assert hits[old] > 0, f"[{tag}] pattern not found: {old!r}"
    return out

def repl_block(lines, old, new, tag=''):
    """Replace a consecutive run of lines equal to `old` (list) with `new` (list)."""
    n = len(old)
    for i in range(len(lines) - n + 1):
        if lines[i:i+n] == old:
            return lines[:i] + new + lines[i+n:]
    raise AssertionError(f"[{tag}] block not found: {old[:2]}…")

def drop(lines, toks, tag):
    out = [l for l in lines if not any(t in l for t in toks)]
    assert len(out) < len(lines), f"[{tag}] nothing dropped"
    return out

def insert_after(lines, anchor, block, tag=''):
    for i, l in enumerate(lines):
        if anchor in l:
            return lines[:i+1] + block + lines[i+1:]
    raise AssertionError(f"[{tag}] anchor not found: {anchor!r}")

def insert_before(lines, anchor, block, tag=''):
    for i, l in enumerate(lines):
        if anchor in l:
            return lines[:i] + block + lines[i:]
    raise AssertionError(f"[{tag}] anchor not found: {anchor!r}")

# ---------- segments ----------
K     = seg(1, 231)
HEAD  = seg(232, 238)
HMAL  = seg(239, 434)
DMAL  = seg(435, 550)
VARS  = seg(551, 558)
INIT  = seg(559, 673)
CPY   = seg(674, 701)
LOOP  = seg(702, 718)
GFWD1 = seg(719, 773)
GFWD2 = seg(774, 828)
MFWD  = seg(829, 1210)
FCHK1 = seg(1211, 1252)
MZ1   = seg(1253, 1271)
FCHK2 = seg(1272, 1313)
MZ2   = seg(1314, 1332)
LOSSG = seg(1333, 1353)
GBWD2 = seg(1354, 1447)
GBWD1A= seg(1448, 1455)
SPL_G = seg(1456, 1462)
GBWD1B= seg(1463, 1548)
MBWD2 = seg(1549, 1876)
MBWD1A= seg(1877, 1911)
SPL_M = seg(1912, 1916)
MBWD1B= seg(1917, 2209)
BCHK  = seg(2210, 2371)
AGPU1 = seg(2372, 2395)
AGPU2 = seg(2396, 2419)
IGPU  = seg(2420, 2491)
AMIR1 = seg(2492, 2557)
AMIR2 = seg(2558, 2623)
ACHK1 = seg(2624, 2689)
ACHK2 = seg(2690, 2755)
SAMPLES = seg(2756, 2764)
TAIL  = seg(2765, 2783)

# segment anchors
assert HEAD[0].startswith('လုပ်ငန်း main()')
assert 'hE = mmc_malloc(704)' in HMAL[0]
assert 'dx0 = mmc_gpu_malloc(512)' in DMAL[0]
assert 'mm_k, dx0, dwq, dq, T, d, K, T, 1' in GFWD1[0]
assert 'mm_k, dh3, dwo, dlog' in ''.join(GFWD1)
assert 'mm_k, dh3B, dwoB, dlogB' in ''.join(GFWD2)
assert 'mse_bwd, dlogB, dy, ddlogB' in GBWD2[0]
assert 'mse_bwd, dlog, dy, ddlog' in GBWD1A[0]
assert 'copy_from(hDH3, dh3g, 512)' in SPL_G[0]
assert 'q1B = 0' in MBWD2[0] and 'rdWoB[w1aB * V + w1bB]' in ''.join(MBWD2)
assert 'q1 = 0' in MBWD1A[0] and 'rdWo[w1a * V + w1b]' in ''.join(MBWD1A)
assert 'hc2' in SPL_M[0]
assert 'rdXt[w20a] = rdHt[w20a]' in ''.join(MBWD1B)
assert 'bc1 = 0' in BCHK[0] and 'hDWoB[bc3B]' in ''.join(BCHK)
assert 'adamw_k, dwq, dgwq' in AGPU1[0]
assert 'ipdwq' in IGPU[0]
assert 'aw1 = 0' in AMIR1[0] and 'aw1B = 0' in AMIR2[0]
assert 'bc4 = 0' in ACHK1[0] and 'bc5B' in ''.join(ACHK2)
assert 'dWq1_0' in SAMPLES[0]
assert 'loss first' in TAIL[0]

# ---------- HMAL / DMAL / VARS ----------
HMAL = drop(HMAL, ['* hWo = mmc_malloc(704)', '* hWoB = mmc_malloc(704)',
                   '* hDWo = mmc_malloc(704)', '* hDWoB = mmc_malloc(704)',
                   '* rdWo = mmc_malloc(704)', '* rdWoB = mmc_malloc(704)',
                   '* hWo2 = mmc_malloc(704)', '* hWo2B = mmc_malloc(704)'], 'HMAL')
HMAL += [
    '    တည်ဆောက် ဒသမ* hM = mmc_malloc(704)',
    '    တည်ဆောက် ဒသမ* hMP = mmc_malloc(512)',
    '    တည်ဆောက် ဒသမ* hP1 = mmc_malloc(704)',
    '    တည်ဆောက် ဒသမ* hP2 = mmc_malloc(704)',
    '    တည်ဆောက် ဒသမ* rP1 = mmc_malloc(704)',
    '    တည်ဆောက် ဒသမ* rP2 = mmc_malloc(704)',
    '    တည်ဆောက် ဒသမ* rX0 = mmc_malloc(512)',
    '    တည်ဆောက် ဒသမ* hX0e = mmc_malloc(512)',
    '    တည်ဆောက် ဒသမ* hX0p = mmc_malloc(512)',
    '    တည်ဆောက် ဒသမ* hE2 = mmc_malloc(704)',
    '    တည်ဆောက် ဒသမ* hPE2 = mmc_malloc(512)',
    '    တည်ဆောက် ဒသမ* hDWte = mmc_malloc(704)',
    '    တည်ဆောက် ဒသမ* hDWpe = mmc_malloc(512)',
    '    တည်ဆောက် ဒသမ* hDWEe = mmc_malloc(704)',
    '    တည်ဆောက် ဒသမ* hDWE1 = mmc_malloc(704)',
    '    တည်ဆောက် ဒသမ* hDWE2 = mmc_malloc(704)',
    '    တည်ဆောက် ဒသမ* hDWPe = mmc_malloc(512)',
    '    တည်ဆောက် ဒသမ* rdWte = mmc_malloc(704)',
    '    တည်ဆောက် ဒသမ* rdWpe = mmc_malloc(512)',
    '    တည်ဆောက် ဒသမ* rdWteE = mmc_malloc(704)',
    '    တည်ဆောက် ဒသမ* rdWte1 = mmc_malloc(704)',
    '    တည်ဆောက် ဒသမ* rdWte2 = mmc_malloc(704)',
    '    တည်ဆောက် ဒသမ* rdWpeE = mmc_malloc(512)',
    '    တည်ဆောက် ဒသမ* rmve = mmc_malloc(1408)',
    '    တည်ဆောက် ဒသမ* rmvpe = mmc_malloc(1024)',
    '    တည်ဆောက် ဒသမ* hmve = mmc_malloc(1408)',
    '    တည်ဆောက် ဒသမ* hmvpe = mmc_malloc(1024)',
]
DMAL = drop(DMAL, ['* dwo = mmc_gpu_malloc(704)', '* dwot = mmc_gpu_malloc(704)',
                   '* dgwo = mmc_gpu_malloc(704)', '* dwoB = mmc_gpu_malloc(704)',
                   '* dwotB = mmc_gpu_malloc(704)', '* dgwoB = mmc_gpu_malloc(704)',
                   '* dy = mmc_gpu_malloc(704)'], 'DMAL')
DMAL += [
    '    တည်ဆောက် ဒသမ* dM = mmc_gpu_malloc(704)',
    '    တည်ဆောက် ဒသမ* dMP = mmc_gpu_malloc(512)',
    '    တည်ဆောက် ဒသမ* dE = mmc_gpu_malloc(704)',
    '    တည်ဆောက် ဒသမ* dPE = mmc_gpu_malloc(512)',
    '    တည်ဆောက် ဒသမ* dEt = mmc_gpu_malloc(704)',
    '    တည်ဆောက် ဒသမ* dX0e = mmc_gpu_malloc(512)',
    '    တည်ဆောက် ဒသမ* dX0p = mmc_gpu_malloc(512)',
    '    တည်ဆောက် ဒသမ* dP1 = mmc_gpu_malloc(704)',
    '    တည်ဆောက် ဒသမ* dP2 = mmc_gpu_malloc(704)',
    '    တည်ဆောက် ဒသမ* dX0g = mmc_gpu_malloc(512)',
    '    တည်ဆောက် ဒသမ* dgwE = mmc_gpu_malloc(704)',
    '    တည်ဆောက် ဒသမ* dgwPE = mmc_gpu_malloc(512)',
    '    တည်ဆောက် ဒသမ* dgwEemb = mmc_gpu_malloc(704)',
    '    တည်ဆောက် ဒသမ* dgwPEemb = mmc_gpu_malloc(512)',
    '    တည်ဆောက် ဒသမ* dgwEB1 = mmc_gpu_malloc(704)',
    '    တည်ဆောက် ဒသမ* dgwEB2 = mmc_gpu_malloc(704)',
    '    တည်ဆောက် ဒသမ* dmve = mmc_gpu_malloc(1408)',
    '    တည်ဆောက် ဒသမ* dmvpe = mmc_gpu_malloc(1024)',
]
VARS = VARS + [
    '    တည်ဆောက် ဒသမ rsPF0 = 0',
    '    တည်ဆောက် ဒသမ rsPF7 = 0',
    '    တည်ဆောက် ဒသမ rsPF0B = 0',
    '    တည်ဆောက် ဒသမ rsPF7B = 0',
]
LOOP = LOOP + [
    '        တည်ဆောက် ဒသမ rsP0 = 0',
    '        တည်ဆောက် ဒသမ rsP7 = 0',
    '        တည်ဆောက် ဒသမ rsP0B = 0',
    '        တည်ဆောက် ဒသမ rsP7B = 0',
]

# ---------- INIT: drop static X0 build; hY -> one-hot; add hM/hMP + rmv zeros ----------
iX0 = next(i for i, l in enumerate(INIT) if 'iX = 0' in l)
assert 'hX0[iX * K + jX] = hE[tid * K + jX] + hPE[iX * d + jX]' in ''.join(INIT)
INIT = INIT[:iX0] + INIT[iX0+10:]
assert 'iX' not in ''.join(INIT), "hX0 build not fully dropped"
# drop the iwo loop (5 lines: decl/while/body/inc/close) AND the iwoB loop —
# substring drop would leave the closing brace, so remove by index
iwo0 = next(i for i, l in enumerate(INIT) if 'iwo = 0' in l)
assert INIT[iwo0+4].strip() == '}', "iwo loop shape changed"
INIT = INIT[:iwo0] + INIT[iwo0+5:]
iwoB0 = next(i for i, l in enumerate(INIT) if 'iwoB = 0' in l)
assert INIT[iwoB0+4].strip() == '}', "iwoB loop shape changed"
INIT = INIT[:iwoB0] + INIT[iwoB0+5:]
assert 'iwo' not in ''.join(INIT)
bal = sum(l.count('{') - l.count('}') for l in INIT)
assert bal == 0, f"INIT brace balance {bal}"
iY = next(i for i, l in enumerate(INIT) if 'irY = 0' in l)
assert INIT[iY+4].strip() == 'hY[irY * V + icY] = ((irY * 3 + icY) % 7) * 0.1 - 0.3'
INIT = INIT[:iY] + [
    '    တည်ဆောက် ကိန်း irY = 0',
    '    ပြည့်အောင်(irY < T) {',
    '        တည်ဆောက် ကိန်း icY = 0',
    '        ပြည့်အောင်(icY < V) {',
    '            hY[irY * V + icY] = 0',
    '            icY = icY + 1',
    '        }',
    '        တည်ဆောက် ကိန်း ytY = (3 * ((irY + 1) % T) + 1) % 11',
    '        hY[irY * V + ytY] = 1',
    '        irY = irY + 1',
    '    }',
] + INIT[iY+9:]
INIT += [
    '    တည်ဆောက် ကိန်း iM = 0',
    '    ပြည့်အောင်(iM < T) {',
    '        တည်ဆောက် ကိန်း jM = 0',
    '        ပြည့်အောင်(jM < V) {',
    '            hM[iM * V + jM] = 0',
    '            jM = jM + 1',
    '        }',
    '        တည်ဆောက် ကိန်း xtM = (3 * iM + 1) % 11',
    '        hM[iM * V + xtM] = 1',
    '        iM = iM + 1',
    '    }',
    '    တည်ဆောက် ကိန်း iPp = 0',
    '    ပြည့်အောင်(iPp < T) {',
    '        တည်ဆောက် ကိန်း jPp = 0',
    '        ပြည့်အောင်(jPp < T) {',
    '            hMP[iPp * T + jPp] = 0',
    '            သုံးသပ်ချက်အရ(jPp == iPp) { hMP[iPp * T + jPp] = 1 }',
    '            jPp = jPp + 1',
    '        }',
    '        iPp = iPp + 1',
    '    }',
    '    တည်ဆောက် ကိန်း iz4 = 0',
    '    ပြည့်အောင်(iz4 < 176) {',
    '        rmve[iz4] = 0',
    '        iz4 = iz4 + 1',
    '    }',
    '    တည်ဆောက် ကိန်း iz5 = 0',
    '    ပြည့်အောင်(iz5 < 128) {',
    '        rmvpe[iz5] = 0',
    '        iz5 = iz5 + 1',
    '    }',
]

# ---------- CPY ----------
CPY = drop(CPY, ['mmc_gpu_copy_to(dx0, hX0, 512)', 'mmc_gpu_copy_to(dy, hY, 704)',
                 'mmc_gpu_copy_to(dwo, hWo, 704)', 'mmc_gpu_copy_to(dwoB, hWoB, 704)'], 'CPY')
CPY += [
    '    mmc_gpu_copy_to(dM, hM, 704)',
    '    mmc_gpu_copy_to(dMP, hMP, 512)',
    '    mmc_gpu_copy_to(dE, hE, 704)',
    '    mmc_gpu_copy_to(dPE, hPE, 512)',
    '    mmc_gpu_copy_to(dmve, rmve, 1408)',
    '    mmc_gpu_copy_to(dmvpe, rmvpe, 1024)',
]

# ---------- EMB_F: device embedding forward + tied-head transpose ----------
EMB_F = [
    '            mmc_gpu_launch3p3i(mm_k, dM, dE, dX0e, T, K, V, T, 1)',
    '            cudaDeviceSynchronize()',
    '            mmc_gpu_launch3p3i(mm_k, dMP, dPE, dX0p, T, d, T, T, 1)',
    '            cudaDeviceSynchronize()',
    '            mmc_gpu_copy_from(hX0e, dX0e, 512)',
    '            mmc_gpu_copy_from(hX0p, dX0p, 512)',
    '            တည်ဆောက် ကိန်း ixE = 0',
    '            ပြည့်အောင်(ixE < 64) {',
    '                hX0[ixE] = hX0e[ixE] + hX0p[ixE]',
    '                ixE = ixE + 1',
    '            }',
    '            mmc_gpu_copy_to(dx0, hX0, 512)',
    '            mmc_gpu_launch3p3i(tpose_rc, dE, dEt, dE, V, K, 0, K, 1)',
    '            cudaDeviceSynchronize()',
]
MIR_EMB = [
    '            တည်ဆောက် ကိန်း ixv = 0',
    '            ပြည့်အောင်(ixv < T) {',
    '                တည်ဆောက် ကိန်း jxv = 0',
    '                ပြည့်အောင်(jxv < K) {',
    '                    တည်ဆောက် ဒသမ xemV = 0',
    '                    တည်ဆောက် ကိန်း kxv = 0',
    '                    ပြည့်အောင်(kxv < V) {',
    '                        xemV = xemV + hM[ixv * V + kxv] * hE[kxv * K + jxv]',
    '                        kxv = kxv + 1',
    '                    }',
    '                    တည်ဆောက် ဒသမ xemP = 0',
    '                    တည်ဆောက် ကိန်း pxv = 0',
    '                    ပြည့်အောင်(pxv < T) {',
    '                        xemP = xemP + hMP[ixv * T + pxv] * hPE[pxv * d + jxv]',
    '                        pxv = pxv + 1',
    '                    }',
    '                    rX0[ixv * K + jxv] = xemV + xemP',
    '                    jxv = jxv + 1',
    '                }',
    '                ixv = ixv + 1',
    '            }',
]

# ---------- GFWD1 / GFWD2: tied head + vocab softmax + CE loss ----------
def head_fwd(lines, blk, tag):
    if blk == 1:
        o_mm, n_mm = 'mmc_gpu_launch3p3i(mm_k, dh3, dwo, dlog, T, V, d, T, 1)', \
                     'mmc_gpu_launch3p3i(mm_k, dh3, dEt, dlog, T, V, K, T, 1)'
        o_sm, cp = 'mmc_gpu_copy_from(hLg, dlog, 704)', 'mmc_gpu_copy_from(hP1, dP1, 704)'
        o_ls = ['တည်ဆောက် ကိန်း ilG = 0', 'ပြည့်အောင်(ilG < 88) {',
                'တည်ဆောက် ဒသမ eG = hLg[ilG] - hY[ilG]', 'lgpu = lgpu + eG * eG',
                'ilG = ilG + 1', '}', 'lgpu = lgpu * 0.5']
        v = 'lgpu'
    else:
        o_mm, n_mm = 'mmc_gpu_launch3p3i(mm_k, dh3B, dwoB, dlogB, T, V, d, T, 1)', \
                     'mmc_gpu_launch3p3i(mm_k, dh3B, dEt, dlogB, T, V, K, T, 1)'
        o_sm, cp = 'mmc_gpu_copy_from(hLgB, dlogB, 704)', 'mmc_gpu_copy_from(hP2, dP2, 704)'
        o_ls = ['တည်ဆောက် ကိန်း ilGB = 0', 'ပြည့်အောင်(ilGB < 88) {',
                'တည်ဆောက် ဒသမ eGB = hLgB[ilGB] - hY[ilGB]', 'lgpuB = lgpuB + eGB * eGB',
                'ilGB = ilGB + 1', '}', 'lgpuB = lgpuB * 0.5']
        v = 'lgpuB'
    lines = sub(lines, [(o_mm, n_mm)], tag + '-mm')
    lines = insert_after(lines, o_sm, ['            ' + cp], tag + '-cp')
    i = next(i for i, l in enumerate(lines) if l.strip() == o_ls[0])
    assert [l.strip() for l in lines[i:i+7]] == o_ls, f"[{tag}] loss loop mismatch"
    lv = f'ilC{blk}'
    new_ls = [
        f'            တည်ဆောက် ကိန်း {lv} = 0',
        f'            ပြည့်အောင်({lv} < T) {{',
        f'                တည်ဆောက် ကိန်း yt{blk} = (3 * (({lv} + 1) % T) + 1) % 11',
        f'                {v} = {v} + log(hP{blk}[{lv} * V + yt{blk}])',
        f'                {lv} = {lv} + 1',
        '            }',
        f'            {v} = (0 - {v}) / T',
    ]
    lines = lines[:i] + new_ls + lines[i+7:]
    si = next(i for i, l in enumerate(lines) if n_mm in l)
    lines = lines[:si+2] + [
        f'            mmc_gpu_launch3p3i(softmax_row, dlog{"" if blk == 1 else "B"}, dP{blk}, dlog{"" if blk == 1 else "B"}, V, T, 0, T, 1)',
        '            cudaDeviceSynchronize()',
    ] + lines[si+2:]
    return lines

GFWD1 = head_fwd(GFWD1, 1, 'GFWD1')
GFWD2 = head_fwd(GFWD2, 2, 'GFWD2')

# ---------- MFWD: rX0, tied mirror heads, mirror vocab softmax + CE ----------
n_hx0 = sum('hX0[' in l for l in MFWD)
assert n_hx0 >= 4, f"MFWD hX0 count {n_hx0}"
MFWD = [l.replace('hX0[', 'rX0[') for l in MFWD]
MFWD = sub(MFWD, [
    ('racc6 = racc6 + rH3[rI6 * d + rT6] * hWo[rT6 * V + rC6]',
     'racc6 = racc6 + rH3[rI6 * d + rT6] * hE[rC6 * K + rT6]'),
    ('racc6B = racc6B + rH3B[rI6B * d + rT6B] * hWoB[rT6B * V + rC6B]',
     'racc6B = racc6B + rH3B[rI6B * d + rT6B] * hE[rC6B * K + rT6B]'),
], 'MFWD-heads')

def mir_softmax(v, pn, tag):
    return [
        f'            တည်ဆောက် ကိန်း rJ7{tag} = 0',
        f'            ပြည့်အောင်(rJ7{tag} < T) {{',
        f'                တည်ဆောက် ဒသမ rmxV{tag} = {v}[rJ7{tag} * V]',
        f'                တည်ဆောက် ကိန်း rJ8{tag} = 1',
        f'                ပြည့်အောင်(rJ8{tag} < V) {{',
        f'                    သုံးသပ်ချက်အရ({v}[rJ7{tag} * V + rJ8{tag}] > rmxV{tag}) {{ rmxV{tag} = {v}[rJ7{tag} * V + rJ8{tag}] }}',
        f'                    rJ8{tag} = rJ8{tag} + 1',
        '                }',
        f'                တည်ဆောက် ဒသမ rsmV{tag} = 0',
        f'                တည်ဆောက် ကိန်း rJ9{tag} = 0',
        f'                ပြည့်အောင်(rJ9{tag} < V) {{',
        f'                    rsmV{tag} = rsmV{tag} + exp({v}[rJ7{tag} * V + rJ9{tag}] - rmxV{tag})',
        f'                    rJ9{tag} = rJ9{tag} + 1',
        '                }',
        f'                တည်ဆောက် ကိန်း rJA{tag} = 0',
        f'                ပြည့်အောင်(rJA{tag} < V) {{',
        f'                    {pn}[rJ7{tag} * V + rJA{tag}] = exp({v}[rJ7{tag} * V + rJA{tag}] - rmxV{tag}) / rsmV{tag}',
        f'                    rJA{tag} = rJA{tag} + 1',
        '                }',
        f'                rJ7{tag} = rJ7{tag} + 1',
        '            }',
    ]

i = next(i for i, l in enumerate(MFWD) if 'ilC = 0' in l)
MFWD = MFWD[:i] + mir_softmax('rLg', 'rP1', '') + [
    '            တည်ဆောက် ကိန်း ilC = 0',
    '            ပြည့်အောင်(ilC < T) {',
    '                တည်ဆောက် ကိန်း ytC = (3 * ((ilC + 1) % T) + 1) % 11',
    '                lcpu = lcpu + log(rP1[ilC * V + ytC])',
    '                ilC = ilC + 1',
    '            }',
    '            lcpu = (0 - lcpu) / T',
] + MFWD[i+7:]
i = next(i for i, l in enumerate(MFWD) if 'ilCB = 0' in l)
MFWD = MFWD[:i] + mir_softmax('rLgB', 'rP2', 'B') + [
    '            တည်ဆောက် ကိန်း ilCB = 0',
    '            ပြည့်အောင်(ilCB < T) {',
    '                တည်ဆောက် ကိန်း ytCB = (3 * ((ilCB + 1) % T) + 1) % 11',
    '                lcpuB = lcpuB + log(rP2[ilCB * V + ytCB])',
    '                ilCB = ilCB + 1',
    '            }',
    '            lcpuB = (0 - lcpuB) / T',
] + MFWD[i+7:]

# ---------- FCHK / MZ: P gates + P rowsums ----------
FCHK1 = insert_before(FCHK1, 'ldiff = lgpu - lcpu', [
    '            တည်ဆောက် ကိန်း iC4 = 0',
    '            ပြည့်အောင်(iC4 < 88) {',
    '                တည်ဆောက် ဒသမ p1 = hP1[iC4] - rP1[iC4]',
    '                သုံးသပ်ချက်အရ(p1 < 0) { p1 = 0 - p1 }',
    '                သုံးသပ်ချက်အရ(p1 > 0.000001) { bad = bad + 1 }',
    '                iC4 = iC4 + 1',
    '            }',
], 'FCHK1-P')
FCHK2 = insert_before(FCHK2, 'ldiffB = lgpuB - lcpuB', [
    '            တည်ဆောက် ကိန်း iC4B = 0',
    '            ပြည့်အောင်(iC4B < 88) {',
    '                တည်ဆောက် ဒသမ p2 = hP2[iC4B] - rP2[iC4B]',
    '                သုံးသပ်ချက်အရ(p2 < 0) { p2 = 0 - p2 }',
    '                သုံးသပ်ချက်အရ(p2 > 0.000001) { bad = bad + 1 }',
    '                iC4B = iC4B + 1',
    '            }',
], 'FCHK2-P')
MZ1 += [
    '            တည်ဆောက် ကိန်း jRSP = 0',
    '            ပြည့်အောင်(jRSP < V) {',
    '                rsP0 = rsP0 + hP1[jRSP]',
    '                rsP7 = rsP7 + hP1[77 + jRSP]',
    '                jRSP = jRSP + 1',
    '            }',
    '            သုံးသပ်ချက်အရ(rsP0 < 0.999999999) { bad = bad + 1 }',
    '            သုံးသပ်ချက်အရ(rsP0 > 1.000000001) { bad = bad + 1 }',
    '            သုံးသပ်ချက်အရ(rsP7 < 0.999999999) { bad = bad + 1 }',
    '            သုံးသပ်ချက်အရ(rsP7 > 1.000000001) { bad = bad + 1 }',
]
MZ2 += [
    '            တည်ဆောက် ကိန်း jRSPB = 0',
    '            ပြည့်အောင်(jRSPB < V) {',
    '                rsP0B = rsP0B + hP2[jRSPB]',
    '                rsP7B = rsP7B + hP2[77 + jRSPB]',
    '                jRSPB = jRSPB + 1',
    '            }',
    '            သုံးသပ်ချက်အရ(rsP0B < 0.999999999) { bad = bad + 1 }',
    '            သုံးသပ်ချက်အရ(rsP0B > 1.000000001) { bad = bad + 1 }',
    '            သုံးသပ်ချက်အရ(rsP7B < 0.999999999) { bad = bad + 1 }',
    '            သုံးသပ်ချက်အရ(rsP7B > 1.000000001) { bad = bad + 1 }',
]

# ---------- LOSSG: anchor + step-8 P rowsum capture ----------
LOSSG = sub(LOSSG, [('eAnch = lgpu - 2.970168', f'eAnch = lgpu - {ANCHOR}')], 'LOSSG-anchor')
LOSSG = insert_after(LOSSG, 'rsFB7 = rs7B', [
    '            rsPF0 = rsP0',
    '            rsPF7 = rsP7',
    '            rsPF0B = rsP0B',
    '            rsPF7B = rsP7B',
], 'LOSSG-capture')

# ---------- GBWD2 / GBWD1A heads: CE grad -> tied head backward ----------
GBWD2 = repl_block(GBWD2, [
    '            mmc_gpu_launch3p3i(mse_bwd, dlogB, dy, ddlogB, 88, 0, 0, 1, 88)',
    '            cudaDeviceSynchronize()',
    '            mmc_gpu_launch3p3i(bwd_dw, dh3B, ddlogB, dgwoB, V, d, T, d, 1)',
    '            cudaDeviceSynchronize()',
    '            mmc_gpu_launch3p3i(tpose_rc, dwoB, dwotB, dwoB, d, V, 0, V, 1)',
    '            cudaDeviceSynchronize()',
    '            mmc_gpu_launch3p3i(mm_k, ddlogB, dwotB, dh3gB, T, d, V, T, 1)',
    '            cudaDeviceSynchronize()',
], [
    '            တည်ဆောက် ကိန်း ig1B = 0',
    '            ပြည့်အောင်(ig1B < 88) {',
    '                hDLB[ig1B] = (hP2[ig1B] - hY[ig1B]) / T',
    '                ig1B = ig1B + 1',
    '            }',
    '            mmc_gpu_copy_to(ddlogB, hDLB, 704)',
    '            mmc_gpu_launch3p3i(bwd_dw, ddlogB, dh3B, dgwEB2, d, V, T, V, 1)',
    '            cudaDeviceSynchronize()',
    '            mmc_gpu_launch3p3i(mm_k, ddlogB, dE, dh3gB, T, d, V, T, 1)',
    '            cudaDeviceSynchronize()',
    '            mmc_gpu_copy_from(hDWE2, dgwEB2, 704)',
], 'GBWD2-head')
GBWD2 = drop(GBWD2, ['mmc_gpu_copy_from(hDLB, ddlogB, 704)', 'mmc_gpu_copy_from(hDWoB, dgwoB, 704)'], 'GBWD2-dup')

GBWD1A = repl_block(GBWD1A, [
    '            mmc_gpu_launch3p3i(mse_bwd, dlog, dy, ddlog, 88, 0, 0, 1, 88)',
    '            cudaDeviceSynchronize()',
    '            mmc_gpu_launch3p3i(bwd_dw, dh3, ddlog, dgwo, V, d, T, d, 1)',
    '            cudaDeviceSynchronize()',
    '            mmc_gpu_launch3p3i(tpose_rc, dwo, dwot, dwo, d, V, 0, V, 1)',
    '            cudaDeviceSynchronize()',
    '            mmc_gpu_launch3p3i(mm_k, ddlog, dwot, dh3g, T, d, V, T, 1)',
    '            cudaDeviceSynchronize()',
], [
    '            တည်ဆောက် ကိန်း ig1 = 0',
    '            ပြည့်အောင်(ig1 < 88) {',
    '                hDL[ig1] = (hP1[ig1] - hY[ig1]) / T',
    '                ig1 = ig1 + 1',
    '            }',
    '            mmc_gpu_copy_to(ddlog, hDL, 704)',
    '            mmc_gpu_launch3p3i(bwd_dw, ddlog, dh3, dgwEB1, d, V, T, V, 1)',
    '            cudaDeviceSynchronize()',
    '            mmc_gpu_launch3p3i(mm_k, ddlog, dE, dh3g, T, d, V, T, 1)',
    '            cudaDeviceSynchronize()',
    '            mmc_gpu_copy_from(hDWE1, dgwEB1, 704)',
], 'GBWD1A-head')
GBWD1B = drop(GBWD1B, ['mmc_gpu_copy_from(hDL, ddlog, 704)', 'mmc_gpu_copy_from(hDWo, dgwo, 704)'], 'GBWD1B-dup')

# ---------- EMB_B_G: device embedding backward + wte/wpe grad assembly ----------
EMB_B_G = [
    '            mmc_gpu_copy_to(dX0g, hDXt, 512)',
    '            mmc_gpu_launch3p3i(bwd_dw, dM, dX0g, dgwEemb, d, V, T, V, 1)',
    '            cudaDeviceSynchronize()',
    '            mmc_gpu_launch3p3i(bwd_dw, dMP, dX0g, dgwPEemb, d, T, T, T, 1)',
    '            cudaDeviceSynchronize()',
    '            mmc_gpu_copy_from(hDWEe, dgwEemb, 704)',
    '            mmc_gpu_copy_from(hDWPe, dgwPEemb, 512)',
    '            တည်ဆောက် ကိန်း iew = 0',
    '            ပြည့်အောင်(iew < 88) {',
    '                hDWte[iew] = hDWEe[iew] + hDWE1[iew] + hDWE2[iew]',
    '                iew = iew + 1',
    '            }',
    '            တည်ဆောက် ကိန်း iep = 0',
    '            ပြည့်အောင်(iep < 64) {',
    '                hDWpe[iep] = hDWPe[iep]',
    '                iep = iep + 1',
    '            }',
    '            mmc_gpu_copy_to(dgwE, hDWte, 704)',
    '            mmc_gpu_copy_to(dgwPE, hDWpe, 512)',
]

# ---------- MBWD2 / MBWD1A: mirror tied-head backward ----------
MBWD2 = sub(MBWD2, [
    ('rdLB[q1B] = rLgB[q1B] - hY[q1B]', 'rdLB[q1B] = (rP2[q1B] - hY[q1B]) / T'),
    ('ပြည့်အောင်(w1aB < d) {', 'ပြည့်အောင်(w1aB < V) {'),
    ('w1sB = w1sB + rH3B[w1cB * d + w1aB] * rdLB[w1cB * V + w1bB]',
     'w1sB = w1sB + rdLB[w1cB * V + w1aB] * rH3B[w1cB * d + w1bB]'),
    ('ပြည့်အောင်(w1bB < V) {', 'ပြည့်အောင်(w1bB < d) {'),
    ('rdWoB[w1aB * V + w1bB] = w1sB', 'rdWte2[w1aB * K + w1bB] = w1sB'),
    ('w2sB = w2sB + rdLB[w2aB * V + w2cB] * hWoB[w2bB * V + w2cB]',
     'w2sB = w2sB + rdLB[w2aB * V + w2cB] * hE[w2cB * K + w2bB]'),
], 'MBWD2-head')
MBWD1A = sub(MBWD1A, [
    ('rdL[q1] = rLg[q1] - hY[q1]', 'rdL[q1] = (rP1[q1] - hY[q1]) / T'),
    ('ပြည့်အောင်(w1a < d) {', 'ပြည့်အောင်(w1a < V) {'),
    ('w1s = w1s + rH3[w1c * d + w1a] * rdL[w1c * V + w1b]',
     'w1s = w1s + rdL[w1c * V + w1a] * rH3[w1c * d + w1b]'),
    ('ပြည့်အောင်(w1b < V) {', 'ပြည့်အောင်(w1b < d) {'),
    ('rdWo[w1a * V + w1b] = w1s', 'rdWte1[w1a * K + w1b] = w1s'),
    ('w2s = w2s + rdL[w2a * V + w2c] * hWo[w2b * V + w2c]',
     'w2s = w2s + rdL[w2a * V + w2c] * hE[w2c * K + w2b]'),
], 'MBWD1A-head')

# ---------- MBWD1B: rX0 + mirror embedding backward ----------
n_hx0b = sum('hX0[' in l for l in MBWD1B)
assert n_hx0b == 3, f"MBWD1B hX0 count {n_hx0b}"
MBWD1B = [l.replace('hX0[', 'rX0[') for l in MBWD1B]
EMB_B_M = [
    '            တည်ဆောက် ကိန်း w21a = 0',
    '            ပြည့်အောင်(w21a < V) {',
    '                တည်ဆောက် ကိန်း w21b = 0',
    '                ပြည့်အောင်(w21b < d) {',
    '                    တည်ဆောက် ဒသမ w21s = 0',
    '                    တည်ဆောက် ကိန်း w21c = 0',
    '                    ပြည့်အောင်(w21c < T) {',
    '                        w21s = w21s + hM[w21c * V + w21a] * rdXt[w21c * d + w21b]',
    '                        w21c = w21c + 1',
    '                    }',
    '                    rdWteE[w21a * K + w21b] = w21s',
    '                    w21b = w21b + 1',
    '                }',
    '                w21a = w21a + 1',
    '            }',
    '            တည်ဆောက် ကိန်း w22a = 0',
    '            ပြည့်အောင်(w22a < T) {',
    '                တည်ဆောက် ကိန်း w22b = 0',
    '                ပြည့်အောင်(w22b < d) {',
    '                    တည်ဆောက် ဒသမ w22s = 0',
    '                    တည်ဆောက် ကိန်း w22c = 0',
    '                    ပြည့်အောင်(w22c < T) {',
    '                        w22s = w22s + hMP[w22a * T + w22c] * rdXt[w22c * d + w22b]',
    '                        w22c = w22c + 1',
    '                    }',
    '                    rdWpeE[w22a * d + w22b] = w22s',
    '                    w22b = w22b + 1',
    '                }',
    '                w22a = w22a + 1',
    '            }',
    '            တည်ဆောက် ကိန်း w23a = 0',
    '            ပြည့်အောင်(w23a < 88) {',
    '                rdWte[w23a] = rdWteE[w23a] + rdWte1[w23a] + rdWte2[w23a]',
    '                w23a = w23a + 1',
    '            }',
    '            တည်ဆောက် ကိန်း w23b = 0',
    '            ပြည့်အောင်(w23b < 64) {',
    '                rdWpe[w23b] = rdWpeE[w23b]',
    '                w23b = w23b + 1',
    '            }',
]

# ---------- BCHK: X0/wpe gates + wte grad renames ----------
i = next(i for i, l in enumerate(BCHK) if 'bc1 = bc1 + 1' in l)
BCHK = BCHK[:i] + [
    '                တည်ဆောက် ဒသမ ec42 = hX0[bc1] - rX0[bc1]',
    '                သုံးသပ်ချက်အရ(ec42 < 0) { ec42 = 0 - ec42 }',
    '                သုံးသပ်ချက်အရ(ec42 > 0.000001) { bad = bad + 1 }',
    '                တည်ဆောက် ဒသမ ec43 = hDWpe[bc1] - rdWpe[bc1]',
    '                သုံးသပ်ချက်အရ(ec43 < 0) { ec43 = 0 - ec43 }',
    '                သုံးသပ်ချက်အရ(ec43 > 0.000001) { bad = bad + 1 }',
] + BCHK[i:]
BCHK = sub(BCHK, [
    ('hDWo[bc3] - rdWo[bc3]', 'hDWte[bc3] - rdWte[bc3]'),
    ('hDWoB[bc3B] - rdWoB[bc3B]', 'hDWE2[bc3B] - rdWte2[bc3B]'),
], 'BCHK-grads')

# ---------- AGPU / IGPU / AMIR / ACHK: wte/wpe take the Wo slots ----------
AGPU1 = sub(AGPU1, [
    ('adamw_k, dwo, dgwo, dmvo, 88, step, 0, 88, 88', 'adamw_k, dE, dgwE, dmve, 88, step, 0, 88, 88'),
    ('mmc_gpu_copy_from(hWo2, dwo, 704)', 'mmc_gpu_copy_from(hE2, dE, 704)'),
    ('mmc_gpu_copy_from(hmvo, dmvo, 1408)', 'mmc_gpu_copy_from(hmve, dmve, 1408)'),
], 'AGPU1')
AGPU2 = sub(AGPU2, [
    ('adamw_k, dwoB, dgwoB, dmvoB, 88, step, 0, 88, 88', 'adamw_k, dPE, dgwPE, dmvpe, 64, step, 0, 64, 64'),
    ('mmc_gpu_copy_from(hWo2B, dwoB, 704)', 'mmc_gpu_copy_from(hPE2, dPE, 512)'),
    ('mmc_gpu_copy_from(hmvoB, dmvoB, 1408)', 'mmc_gpu_copy_from(hmvpe, dmvpe, 1024)'),
], 'AGPU2')
IGPU = sub(IGPU, [
    ('တည်ဆောက် ကိန်း ipdwo = 0', 'တည်ဆောက် ကိန်း ipdwte = 0'),
    ('ပြည့်အောင်(ipdwo < 88) {', 'ပြည့်အောင်(ipdwte < 88) {'),
    ('hWo2[ipdwo] = hWo[ipdwo] + (lrs * (hWo2[ipdwo] - hWo[ipdwo]))',
     'hE2[ipdwte] = hE[ipdwte] + (lrs * (hE2[ipdwte] - hE[ipdwte]))'),
    ('ipdwo = ipdwo + 1', 'ipdwte = ipdwte + 1'),
    ('mmc_gpu_copy_to(dwo, hWo2, 704)', 'mmc_gpu_copy_to(dE, hE2, 704)'),
    ('တည်ဆောက် ကိန်း ipdwoB = 0', 'တည်ဆောက် ကိန်း ipdwpe = 0'),
    ('ပြည့်အောင်(ipdwoB < 88) {', 'ပြည့်အောင်(ipdwpe < 64) {'),
    ('hWo2B[ipdwoB] = hWoB[ipdwoB] + (lrs * (hWo2B[ipdwoB] - hWoB[ipdwoB]))',
     'hPE2[ipdwpe] = hPE[ipdwpe] + (lrs * (hPE2[ipdwpe] - hPE[ipdwpe]))'),
    ('ipdwoB = ipdwoB + 1', 'ipdwpe = ipdwpe + 1'),
    ('mmc_gpu_copy_to(dwoB, hWo2B, 704)', 'mmc_gpu_copy_to(dPE, hPE2, 512)'),
], 'IGPU')
AMIR1 = sub(AMIR1, [
    ('amo = 0.9 * rmvo[2 * aw4] + 0.1 * rdWo[aw4]', 'ame = 0.9 * rmve[2 * aw4] + 0.1 * rdWte[aw4]'),
    ('avo = 0.999 * rmvo[2 * aw4 + 1] + 0.001 * rdWo[aw4] * rdWo[aw4]',
     'ave = 0.999 * rmve[2 * aw4 + 1] + 0.001 * rdWte[aw4] * rdWte[aw4]'),
    ('rmvo[2 * aw4] = amo', 'rmve[2 * aw4] = ame'),
    ('rmvo[2 * aw4 + 1] = avo', 'rmve[2 * aw4 + 1] = ave'),
    ('amho = amo / (1.0 - pow(0.9, step))', 'ameh = ame / (1.0 - pow(0.9, step))'),
    ('avho = avo / (1.0 - pow(0.999, step))', 'aveh = ave / (1.0 - pow(0.999, step))'),
    ('hWo[aw4] = hWo[aw4] - (lrs * 0.01) * amho / (sqrt(avho) + 0.00000001)',
     'hE[aw4] = hE[aw4] - (lrs * 0.01) * ameh / (sqrt(aveh) + 0.00000001)'),
], 'AMIR1')
AMIR2 = sub(AMIR2, [
    ('ပြည့်အောင်(aw4B < 88) {', 'ပြည့်အောင်(aw4B < 64) {'),
    ('amoB = 0.9 * rmvoB[2 * aw4B] + 0.1 * rdWoB[aw4B]', 'amp = 0.9 * rmvpe[2 * aw4B] + 0.1 * rdWpe[aw4B]'),
    ('avoB = 0.999 * rmvoB[2 * aw4B + 1] + 0.001 * rdWoB[aw4B] * rdWoB[aw4B]',
     'avp = 0.999 * rmvpe[2 * aw4B + 1] + 0.001 * rdWpe[aw4B] * rdWpe[aw4B]'),
    ('rmvoB[2 * aw4B] = amoB', 'rmvpe[2 * aw4B] = amp'),
    ('rmvoB[2 * aw4B + 1] = avoB', 'rmvpe[2 * aw4B + 1] = avp'),
    ('amhoB = amoB / (1.0 - pow(0.9, step))', 'amhp = amp / (1.0 - pow(0.9, step))'),
    ('avhoB = avoB / (1.0 - pow(0.999, step))', 'avhp = avp / (1.0 - pow(0.999, step))'),
    ('hWoB[aw4B] = hWoB[aw4B] - (lrs * 0.01) * amhoB / (sqrt(avhoB) + 0.00000001)',
     'hPE[aw4B] = hPE[aw4B] - (lrs * 0.01) * amhp / (sqrt(avhp) + 0.00000001)'),
], 'AMIR2')
ACHK1 = sub(ACHK1, [
    ('hWo2[bc5] - hWo[bc5]', 'hE2[bc5] - hE[bc5]'),
    ('hmvo[2 * bc5] - rmvo[2 * bc5]', 'hmve[2 * bc5] - rmve[2 * bc5]'),
    ('hmvo[2 * bc5 + 1] - rmvo[2 * bc5 + 1]', 'hmve[2 * bc5 + 1] - rmve[2 * bc5 + 1]'),
], 'ACHK1')
ACHK2 = sub(ACHK2, [
    ('ပြည့်အောင်(bc5B < 88) {', 'ပြည့်အောင်(bc5B < 64) {'),
    ('hWo2B[bc5B] - hWoB[bc5B]', 'hPE2[bc5B] - hPE[bc5B]'),
    ('hmvoB[2 * bc5B] - rmvoB[2 * bc5B]', 'hmvpe[2 * bc5B] - rmvpe[2 * bc5B]'),
    ('hmvoB[2 * bc5B + 1] - rmvoB[2 * bc5B + 1]', 'hmvpe[2 * bc5B + 1] - rmvpe[2 * bc5B + 1]'),
], 'ACHK2')

# ---------- NC transforms ----------
if MODE == 'nc1':
    # mirror CE-grad scale break: block-2 mirror grad loses the /T
    MBWD2 = sub(MBWD2, [('rdLB[q1B] = (rP2[q1B] - hY[q1B]) / T',
                         'rdLB[q1B] = (rP2[q1B] - hY[q1B])')], 'NC1')
if MODE == 'nc2':
    MBWD2 = sub(MBWD2, [('w2sB = w2sB + rdLB[w2aB * V + w2cB] * hE[w2cB * K + w2bB]',
                         'w2sB = w2sB + 0.0 * rdLB[w2aB * V + w2cB] * hE[w2cB * K + w2bB]')], 'NC2')

# ---------- SAMPLES / TAIL ----------
SAMPLES = insert_after(SAMPLES, 'mvq2_0', [
    '        printf("sample: dWte_0 gpu=%.10f cpu=%.10f\\n", hDWte[0], rdWte[0])',
    '        printf("sample: dWpe_0 gpu=%.10f cpu=%.10f\\n", hDWpe[0], rdWpe[0])',
    '        printf("sample: wte_new gpu=%.10f cpu=%.10f\\n", hE2[0], hE[0])',
    '        printf("sample: wpe_new gpu=%.10f cpu=%.10f\\n", hPE2[0], hPE[0])',
    '        printf("sample: mve_0 gpu=%.10f cpu=%.10f\\n", hmve[0], rmve[0])',
    '        printf("sample: mvpe_0 gpu=%.10f cpu=%.10f\\n", hmvpe[0], rmvpe[0])',
], 'SAMPLES')
TAIL = insert_after(TAIL, 'A2 row0 sum', [
    '    printf("P1 row0 sum=%.12f row7 sum=%.12f\\n", rsPF0, rsPF7)',
    '    printf("P2 row0 sum=%.12f row7 sum=%.12f\\n", rsPF0B, rsPF7B)',
], 'TAIL-P')
TAIL = insert_after(TAIL, 'mvq2_final', [
    '    printf("sample: wte_final gpu=%.10f cpu=%.10f\\n", hE2[0], hE[0])',
    '    printf("sample: wpe_final gpu=%.10f cpu=%.10f\\n", hPE2[0], hPE[0])',
    '    printf("sample: mve_final gpu=%.10f cpu=%.10f\\n", hmve[0], rmve[0])',
    '    printf("sample: mvpe_final gpu=%.10f cpu=%.10f\\n", hmvpe[0], rmvpe[0])',
], 'TAIL-finals')

# ---------- gate accounting (executed conditions over the 8-step run) ----------
# per step, loop-bound aware:
fwd_exec = (6*64 + 2*256 + 88 + 88) * 2        # iC1..iC4 + iC1B..iC4B elementwise
sync_exec = 3                                   # ldiff + ldiffB + ldiffT
mz_exec = 28 + 28                               # masked-zero (causal A / A2)
rs_exec = 4 + 4 + 4 + 4                         # rowsums: A, A2, P1, P2 (two-sided)
bwd_exec = (19*64 + 4*256 + 2*88) + (17*64 + 4*256 + 2*88)   # bc1..bc3 + bc1B..bc3B
anch_exec = 1
ld_exec = 1                                     # LOSS DECREASED site
per_step = fwd_exec + sync_exec + mz_exec + rs_exec + bwd_exec + anch_exec + ld_exec
total_exec = per_step * 8
GATES = (f"{total_exec} gate conditions over the 8-step run ({fwd_exec} forward "
         f"elementwise + {bwd_exec} backward elementwise + {mz_exec} masked-zero + "
         f"{rs_exec} rowsum + {sync_exec} loss-sync + {anch_exec} anchor + {ld_exec} "
         f"LOSS DECREASED per step)")
BANNER = ("BLOCK SEQCE MATCH — sequence-level cross-entropy training with trainable embedding "
          "(wte V×K one-hot-gathered via mm_k: E = M@wte + MP@wpe) + tied output head "
          "(logits = H @ wte^T via dEt = tpose_rc(dE), both blocks deep supervision) — CE loss = "
          "-(1/T)·Σ_t log P[t,y_t] with shift-by-one token targets (x_t=(3t+1) mod 11, y_t = "
          "x_((t+1) mod T)) — step-1 CE = ln(11) byte-exact (uniform softmax, separable wte init + "
          "centered LN) — CE gradient (P−Y)/T computed on host from device softmax — wte gradient = "
          "embedding path (M^T@dX0) + block-1 head path (dLg1^T@H3) + block-2 head path "
          "(dLg2^T@H3B) summed on host; wpe gradient = MP^T@dX0 — L=2 chain preserved (dX1 → "
          "block-1 dH3 before LN-backward) — 12 AdamW matrices (Wq Wk Wv W1 W2 ×2 + wte + wpe) "
          "lrs=1/step persistent TRUE m/v — " + GATES + " all GPU = CPU (T=8, K=8, d=8, H=32, "
          "V=11) — zero new kernels — locked kernels only (mse_bwd retired) — sequence CE "
          "token-target training proven in pure MMC on GPU")
LD_TEXT = ("LOSS DECREASED — 8 forward loss pairs 8 AdamW steps (t=1..8) inverse-step lr schedule "
           "12 matrices persistent true-gradient m/v — sequence CE with tied head and trainable "
           "embedding works on GPU")
old_ld = [l for l in TAIL if 'LOSS DECREASED' in l][0].strip()
TAIL = sub(TAIL, [(old_ld, f'printf("{LD_TEXT}\\n")')], 'TAIL-LD')
old_match = [l for l in TAIL if 'BLOCK L2CHAIN MATCH' in l][0].strip()
TAIL = sub(TAIL, [(old_match, f'printf("{BANNER}\\n")')], 'TAIL-MATCH')
TAIL = [l.replace('BLOCK L2CHAIN MISMATCH', 'BLOCK SEQCE MISMATCH') for l in TAIL]

# ---------- assemble ----------
out = []
out += K
out += HEAD
out += HMAL
out += DMAL
out += VARS
out += INIT
out += CPY
out += LOOP
out += EMB_F
out += MIR_EMB
out += GFWD1
out += GFWD2
out += MFWD
out += FCHK1
out += MZ1
out += FCHK2
out += MZ2
out += LOSSG
out += GBWD2
out += GBWD1A
out += SPL_G
out += GBWD1B
out += EMB_B_G
out += MBWD2
out += MBWD1A
out += SPL_M
out += MBWD1B
out += EMB_B_M
out += BCHK
out += AGPU1
out += AGPU2
out += IGPU
out += AMIR1
out += AMIR2
out += ACHK1
out += ACHK2
out += SAMPLES
out += TAIL

body = '\n'.join(out) + '\n'

# ---------- final integrity asserts ----------
assert body.count('mse_bwd') == 2, body.count('mse_bwd')            # K-seg def + banner mention
assert body.count('log(') == 4, body.count('log(')                  # 4 CE loss loops
assert 'hWo' not in body, "hWo leaked"
assert 'dgwo' not in body, "dgwo leaked"
assert 'dwot' not in body, "dwot leaked"
assert body.count('dEt') >= 4
assert body.count('adamw_k, dE, dgwE') == 1 and body.count('adamw_k, dPE, dgwPE') == 1
assert body.count('BLOCK SEQCE MATCH') == 1

open(DST, 'w').write(body)
sha = hashlib.sha256(body.encode()).hexdigest()
print(f"written {DST} mode={MODE}: {len(out)} lines")
print(f"SEQCE_SHA {sha}")
print(f"GATES {GATES}")
