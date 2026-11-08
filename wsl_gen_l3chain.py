#!/usr/bin/env python3
# wsl_gen_l3chain.py — task-ad (WSL) —: L=3 chained deep supervision in pure MMC on GPU
# Base = task-ac block_seqce_test.mmc (3036 lines, sha c6bfa957…).
# Transform: splice a THIRD block (C suffix) into the L=2 chain —
#  fwd chain dx0→B1→B2→B3, CE1+CE2+CE3 tied heads, backward chain
#  dX2(C) → block-2 dh3gB splice → dX1(B) → block-1 dh3g splice → dX0,
#  wte grad = emb + head1 + head2 + head3 (4 terms), 18 AdamW matrices.
# Zero new kernels. Modes: main | nc1 (mirror CE-grad-scale break) | nc2
# (mirror chain-splice break). Usage: gen_l3chain.py [mode]
import hashlib
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else 'main'
SRC = 'scripts/mmc_gpu/block_seqce_test.mmc'
DST = 'scripts/mmc_gpu/block_l3chain_test.mmc'
SRC_SHA = 'c6bfa957003e6d77fc132d5bad35d14b2cb122987f08135e09d803581ff1944a'

src = open(SRC).read().splitlines()
assert len(src) == 3036, f"task-ac base line count changed: {len(src)}"
assert hashlib.sha256(open(SRC, 'rb').read()).hexdigest() == SRC_SHA, "task-ac base sha changed"

def seg(a, b):
    return src[a-1:b]

def ren(lines, pairs):
    out, hits = [], {k: 0 for k, _ in pairs}
    for l in lines:
        for old, new in pairs:
            if old in l:
                hits[old] += l.count(old)
                l = l.replace(old, new)
        out.append(l)
    for old, _ in pairs:
        assert hits[old] > 0, f"rename pattern not found: {old!r}"
    return out

def blanket_c(lines, protect=('hBet',)):
    # segments contain 'B' only as block-2 identifier suffixes, EXCEPT the
    # protected tokens (hBet = LN beta, dgwEB3 = wte head-3 grad buffer)
    out = []
    for l in lines:
        for k, p in enumerate(protect):
            l = l.replace(p, chr(0) + str(k) + chr(0))
        l = l.replace('B', 'C')
        for k, p in enumerate(protect):
            l = l.replace(chr(0) + str(k) + chr(0), p)
        out.append(l)
    return out

# ---------- C-block segments copied from B-block ----------
SEG_DECL_HOST = blanket_c(seg(336, 426))
SEG_DECL_HOST += [
    "    တည်ဆောက် ဒသမ* hP3 = mmc_malloc(704)",
    "    တည်ဆောက် ဒသမ* rP3 = mmc_malloc(704)",
    "    တည်ဆောက် ဒသမ* hDWE3 = mmc_malloc(704)",
    "    တည်ဆောက် ဒသမ* rdWte3 = mmc_malloc(704)",
]
SEG_DECL_DEV = blanket_c(seg(510, 562))
SEG_DECL_DEV += [
    "    တည်ဆောက် ဒသမ* dP3 = mmc_gpu_malloc(704)",
    "    တည်ဆောက် ဒသမ* dgwEB3 = mmc_gpu_malloc(704)",
]
SEG_INIT = blanket_c(seg(659, 689))
SEG_CPT = blanket_c(seg(734, 744))
SEG_LOCALS = [
    "        တည်ဆောက် ဒသမ lgpuC = 0",
    "        တည်ဆောက် ဒသမ lcpuC = 0",
    "        တည်ဆောက် ဒသမ ldiffC = 0",
    "        တည်ဆောက် ဒသမ rs0C = 0",
    "        တည်ဆောက် ဒသမ rs7C = 0",
    "        တည်ဆောက် ဒသမ rsP0C = 0",
    "        တည်ဆောက် ဒသမ rsP7C = 0",
]
SEG_FINALS_DECL = [
    "    တည်ဆောက် ဒသမ rsFC0 = 0",
    "    တည်ဆောက် ဒသမ rsFC7 = 0",
    "    တည်ဆောက် ဒသမ rsPFC0 = 0",
    "    တည်ဆောက် ဒသမ rsPFC7 = 0",
]
SEG_FWD3 = ren(seg(865, 922), [('dP2', 'dP3'), ('hP2', 'hP3'),
                               ('ilC2', 'ilC3'), ('yt2', 'yt3')])
SEG_FWD3 = blanket_c(SEG_FWD3)
SEG_FWD3 = ren(SEG_FWD3, [('mm_k, dh3, dwqC', 'mm_k, dh3B, dwqC'),
                          ('mm_k, dh3, dwkC', 'mm_k, dh3B, dwkC'),
                          ('mm_k, dh3, dwvC', 'mm_k, dh3B, dwvC'),
                          ('= hH3[iR1C]', '= hH3B[iR1C]')])
SEG_MFWD3 = ren(seg(1135, 1346), [('rP2', 'rP3'), ('ilCB', 'ilC3M'), ('ytCB', 'yt3M')])
SEG_MFWD3 = blanket_c(SEG_MFWD3)
SEG_MFWD3 = ren(SEG_MFWD3, [('rH3[rIC', 'rH3B[rIC'),
                            ('= rH3[rH4C]', '= rH3B[rH4C]')])
SEG_GATES3 = ren(seg(1425, 1502), [('p2', 'p3'), ('hP2', 'hP3'), ('rP2', 'rP3')])
SEG_GATES3 = blanket_c(SEG_GATES3)
SEG_STEP8 = [
    "            rsFC0 = rs0C",
    "            rsFC7 = rs7C",
    "            rsPFC0 = rsP0C",
    "            rsPFC7 = rsP7C",
]
SEG_GBWD3 = ren(seg(1528, 1622), [('hP2', 'hP3'), ('dgwEB2', 'dgwEB3'),
                                  ('hDWE2', 'hDWE3')])
SEG_GBWD3 = blanket_c(SEG_GBWD3, protect=('hBet', 'dgwEB3'))
SEG_GBWD3 = ren(SEG_GBWD3, [('bwd_dw, dh3, gqC', 'bwd_dw, dh3B, gqC'),
                            ('bwd_dw, dh3, gkC', 'bwd_dw, dh3B, gkC'),
                            ('bwd_dw, dh3, gvC', 'bwd_dw, dh3B, gvC')])
SEG_SPLICE_DEV = [
    "            mmc_gpu_copy_from(hDH3B, dh3gB, 512)",
    "            တည်ဆောက် ကိန်း hc3 = 0",
    "            ပြည့်အောင်(hc3 < 64) {",
    "                hDH3B[hc3] = hDH3B[hc3] + hDXtC[hc3]",
    "                hc3 = hc3 + 1",
    "            }",
    "            mmc_gpu_copy_to(dh3gB, hDH3B, 512)",
]
SEG_MBWD3 = ren(seg(1744, 2071), [('rP2', 'rP3'), ('rdWte2', 'rdWte3')])
SEG_MBWD3 = blanket_c(SEG_MBWD3)
SEG_MBWD3 = ren(SEG_MBWD3, [('rH3[w16cC', 'rH3B[w16cC')])
SEG_SPLICE_MIR = [
    "            တည်ဆောက် ကိန်း hc4 = 0",
    "            ပြည့်အောင်(hc4 < 64) {",
    "                rdH3B[hc4] = rdH3B[hc4] + rdXtC[hc4]",
    "                hc4 = hc4 + 1",
    "            }",
]
SEG_BCHK3 = ren(seg(2532, 2612), [('hDWE2', 'hDWE3'), ('rdWte2', 'rdWte3')])
SEG_BCHK3 = blanket_c(SEG_BCHK3)
SEG_ADAM1 = blanket_c(seg(2637, 2642))
SEG_ADAM2 = blanket_c(seg(2645, 2648))
SEG_CP1 = blanket_c(seg(2649, 2651))
SEG_CP2 = blanket_c(seg(2653, 2654))
SEG_CP3 = blanket_c(seg(2655, 2657))
SEG_CP4 = blanket_c(seg(2659, 2660))
SEG_IP1 = blanket_c(seg(2697, 2714))
SEG_IP2 = blanket_c(seg(2721, 2732))
SEG_AMIR1 = blanket_c(seg(2799, 2831))
SEG_AMIR2 = blanket_c(seg(2843, 2864))
SEG_ACHK1 = blanket_c(seg(2931, 2961))
SEG_ACHK2 = blanket_c(seg(2975, 2996))
SEG_SAMPLES = [
    '        printf("sample: dWq3_0 gpu=%.10f cpu=%.10f\\n", hDWqC[0], rdWqC[0])',
    '        printf("sample: Wq3_new gpu=%.10f cpu=%.10f\\n", hWq2C[0], hWqC[0])',
    '        printf("sample: mvq3_0 gpu=%.10f cpu=%.10f\\n", hmvqC[0], rmvqC[0])',
]
SEG_F_A3 = ['    printf("A3 row0 sum=%.12f row7 sum=%.12f\\n", rsFC0, rsFC7)']
SEG_F_P3 = ['    printf("P3 row0 sum=%.12f row7 sum=%.12f\\n", rsPFC0, rsPFC7)']
SEG_F_WQ3 = ['    printf("sample: Wq3_final gpu=%.10f cpu=%.10f\\n", hWq2C[0], hWqC[0])']
SEG_F_MVQ3 = ['    printf("sample: mvq3_final gpu=%.10f cpu=%.10f\\n", hmvqC[0], rmvqC[0])']

# ---------- anchor table: insert AFTER anchor line (optionally after its sync) ----------
INS = []
def after(anchor, lines, sync=False, skip1=False):
    INS.append((anchor, lines, sync, skip1))

after("rmv2B = mmc_malloc(4096)", SEG_DECL_HOST)
after("dmvpe = mmc_gpu_malloc(1024)", SEG_DECL_DEV)
after("တည်ဆောက် ဒသမ rsPF7B = 0", SEG_FINALS_DECL)
after("iz3B = iz3B + 1", SEG_INIT, skip1=True)
after("mmc_gpu_copy_to(dmv2B, rmv2B, 4096)", SEG_CPT)
after("တည်ဆောက် ဒသမ rsP7B = 0", SEG_LOCALS)
after("lgpuB = (0 - lgpuB) / T", SEG_FWD3)
after("lcpuB = (0 - lcpuB) / T", SEG_MFWD3)
after("သုံးသပ်ချက်အရ(rsP7B > 1.000000001) { bad = bad + 1 }", SEG_GATES3)
after("rsPF7B = rsP7B", SEG_STEP8)
after('printf("loss b1=%.6f b2=%.6f\\n", lgpu, lgpuB)', SEG_GBWD3)
after("mmc_gpu_launch3p3i(mm_k, ddlogB, dE, dh3gB, T, d, V, T, 1)", SEG_SPLICE_DEV, sync=True)
after("mmc_gpu_copy_to(dgwPE, hDWpe, 512)", SEG_MBWD3)
after("w2aB = w2aB + 1", SEG_SPLICE_MIR, skip1=True)
after("bc3B = bc3B + 1", SEG_BCHK3, skip1=True)
after("adamw_k, dwvB, dgwvB, dmvvB", SEG_ADAM1, sync=True)
after("adamw_k, dw2B, dgw2B, dmv2B", SEG_ADAM2, sync=True)
after("mmc_gpu_copy_from(hWv2B, dwvB, 512)", SEG_CP1)
after("mmc_gpu_copy_from(hW12B, dw1B, 2048)", SEG_CP2)
after("mmc_gpu_copy_from(hmvvB, dmvvB, 1024)", SEG_CP3)
after("mmc_gpu_copy_from(hmv2B, dmv2B, 4096)", SEG_CP4)
after("mmc_gpu_copy_to(dwvB, hWv2B, 512)", SEG_IP1)
after("mmc_gpu_copy_to(dw2B, hW22B, 2048)", SEG_IP2)
after("aw3B = aw3B + 1", SEG_AMIR1, skip1=True)
after("aw6B = aw6B + 1", SEG_AMIR2, skip1=True)
after("bc4B = bc4B + 1", SEG_ACHK1, skip1=True)
after("bc6B = bc6B + 1", SEG_ACHK2, skip1=True)
after('printf("sample: mvq2_0', SEG_SAMPLES)
after('printf("A2 row0 sum', SEG_F_A3)
after('printf("P2 row0 sum', SEG_F_P3)
after('printf("sample: Wq2_final', SEG_F_WQ3)
after('printf("sample: mvq2_final', SEG_F_MVQ3)

# ---------- line modifications ----------
if MODE == 'main':
    MODS = [
        ("        ltot = lgpu + lgpuB",
         "        ltot = lgpu + lgpuB + lgpuC"),
        ("        lctot = lcpu + lcpuB",
         "        lctot = lcpu + lcpuB + lcpuC"),
        ('        printf("loss b1=%.6f b2=%.6f\\n", lgpu, lgpuB)',
         '        printf("loss b1=%.6f b2=%.6f b3=%.6f\\n", lgpu, lgpuB, lgpuC)'),
        ("hDWte[iew] = hDWEe[iew] + hDWE1[iew] + hDWE2[iew]",
         "hDWte[iew] = hDWEe[iew] + hDWE1[iew] + hDWE2[iew] + hDWE3[iew]"),
        ("rdWte[w23a] = rdWteE[w23a] + rdWte1[w23a] + rdWte2[w23a]",
         "rdWte[w23a] = rdWteE[w23a] + rdWte1[w23a] + rdWte2[w23a] + rdWte3[w23a]"),
        ('printf("LOSS DECREASED — 8 forward loss pairs 8 AdamW steps (t=1..8) inverse-step lr schedule 12 matrices persistent true-gradient m/v — sequence CE with tied head and trainable embedding works on GPU\\n")',
         'printf("LOSS DECREASED — 8 forward loss pairs 8 AdamW steps (t=1..8) inverse-step lr schedule 18 matrices persistent true-gradient m/v — L=3 chained deep supervision sequence CE with tied head and trainable embedding works on GPU\\n")'),
    ]
    MATCH_NEW = '        printf("BLOCK SEQCE MATCH — L=3 chained three-block deep supervision in pure MMC on GPU: forward chain dx0 → block-1 → block-2 → block-3 (each block = attention + FFN + residual + centered LN, shared gamma/beta) — trainable embedding (wte V×K one-hot-gathered via mm_k: E = M@wte + MP@wpe) + tied output head (logits = H @ wte^T via dEt = tpose_rc(dE), ALL THREE blocks deep supervision, loss = CE1+CE2+CE3) — CE loss = -(1/T)·Σ_t log P[t,y_t] with shift-by-one token targets (x_t=(3t+1) mod 11, y_t = x_((t+1) mod T)) — step-1 CE = ln(11) byte-exact per block (uniform softmax, separable wte init + centered LN) — CE gradient (P−Y)/T computed on host from device softmax — wte gradient = embedding path (M^T@dX0) + block-1 head path (dLg1^T@H3) + block-2 head path (dLg2^T@H3B) + block-3 head path (dLg3^T@H3C) summed on host; wpe gradient = MP^T@dX0 — L=3 chain fully spliced: block-3 backward dX2 → block-2 dh3gB splice before LN-backward → dX1 → block-1 dh3g splice → dX0 — mirror chain identical (rdXtC → rdH3B splice → rdXtB → rdH3 splice → rdX0) — 18 AdamW matrices ({Wq Wk Wv W1 W2}×3 + wte + wpe) lrs=1/step persistent TRUE m/v — 82576 gate conditions over the 8-step run (3216 forward elementwise + 6992 backward elementwise + 84 masked-zero + 24 rowsum + 4 loss-sync + 1 anchor + 1 LOSS DECREASED per step) all GPU = CPU (T=8, K=8, d=8, H=32, V=11) — zero new kernels — locked kernels only — L=3 deep supervision chain proven in pure MMC on GPU\\n")'
elif MODE == 'nc1':
    # mirror CE-grad-scale break at block-3: rdLC loop loses /T
    NC = ("                rdLC[q1C] = (rP3[q1C] - hY[q1C]) / T",
          "                rdLC[q1C] = (rP3[q1C] - hY[q1C])")
elif MODE == 'nc2':
    # mirror chain-splice break: rdH3B += rdXtC term zeroed
    NC = ("                rdH3B[hc4] = rdH3B[hc4] + rdXtC[hc4]",
          "                rdH3B[hc4] = rdH3B[hc4] + 0.0 * rdXtC[hc4]")
else:
    sys.exit('mode not ready')

out = []
pending = None
for i, l in enumerate(src, 1):
    out.append(l)
    if pending is not None:
        out.extend(pending)
        pending = None
    for anchor, seg_lines, sync, skip1 in INS:
        if anchor in l:
            j = i  # index of anchor (1-based)
            if sync:
                assert j < len(src) and src[j] .strip() == 'cudaDeviceSynchronize()', \
                    f"anchor {anchor!r}: sync line not found"
                pending = seg_lines  # insert after the sync line itself
            elif skip1:
                assert j < len(src) and src[j].strip() == '}', \
                    f"anchor {anchor!r}: close brace not found"
                pending = seg_lines  # insert after the loop close brace
            else:
                out.extend(seg_lines)
assert len(INS) == 32, len(INS)

text = '\n'.join(out)
if MODE == 'main':
    out_lines = text.split('\n')
    mi = [k for k, l in enumerate(out_lines) if l.startswith('        printf("BLOCK SEQCE MATCH')]
    assert len(mi) == 1, f"MATCH line found {len(mi)}x"
    out_lines[mi[0]] = MATCH_NEW
    text = '\n'.join(out_lines)
    for old, new in MODS:
        assert text.count(old) == 1, f"mod pattern count={text.count(old)}: {old[:60]!r}"
        text = text.replace(old, new)
if MODE != 'main':
    assert text.count(NC[0]) == 1, f"NC pattern count={text.count(NC[0])}"
    text = text.replace(NC[0], NC[1])

open(DST, 'w').write(text + '\n')
n = len(text.splitlines())
assert n == 4272, f"unexpected line count {n}"
print(f"GEN_OK lines={n} bytes={len(text)+1}")
print(f"DST_SHA={hashlib.sha256(open(DST,'rb').read()).hexdigest()[:16]}")
