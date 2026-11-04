#!/usr/bin/env python3
# pointer-decl cast emission ကို အရင်းခံ source (src/compiler_v14.mmc) ထဲ မှန်ကန်စွာ ထည့်တာ
import re, shutil, sys

MMC = 'src/compiler_v14.mmc'
PRI = '/tmp/gen_pristine.c'
CUR = 'build/compiler_gen.c'

def dump(lines, a, b, tag=''):
    print('---- %s ----' % tag)
    for i in range(max(0, a), min(len(lines), b)):
        print('%5d | %s' % (i+1, lines[i]))

mlines = open(MMC, encoding='utf-8').read().split('\n')

# ၁) .mmc မှာ template ရှိပြီးသားဆို — paren bug ပြင်ပြီး ရပ်
fp_mmc = [i for i, l in enumerate(mlines) if '"*)("' in l]
if fp_mmc:
    j0 = fp_mmc[0]
    print('MMC_ALREADY_PATCHED — line %d: %s' % (j0+1, mlines[j0].strip()))
    fixed = False
    for j in range(max(0, j0-5), min(len(mlines), j0+7)):
        m = re.search(r'"[^"]*\)\)[^"]*"', mlines[j])
        if m:
            seg = m.group(0).replace('))', ')', 1)
            mlines[j] = mlines[j][:m.start()] + seg + mlines[j][m.end():]
            print('PAREN_FIXED line %d: %s' % (j+1, mlines[j].strip()))
            fixed = True
            break
    if fixed:
        shutil.copy(MMC, 'archive/backups/compiler_v14.mmc.pre_castfix.bak')
        open(MMC, 'w', encoding='utf-8').write('\n'.join(mlines))
        print('SAVED — paren fix done')
    else:
        dump(mlines, j0-6, j0+8, 'mmc region (bug မတွေ့)')
    sys.exit(0)

# ၂) လက်ဖြင့် patch ခြေရာကနေ C အချက်အလက် ဖတ်
cur = open(CUR, encoding='utf-8').read().split('\n')
fp_c = [i for i, l in enumerate(cur) if '"*)("' in l]
if not fp_c:
    print('HANDPATCH_NOT_FOUND — build/compiler_gen.c ထဲ cast ခြေရာ မရှိ')
    sys.exit(2)
p_line = fp_c[0]
print('hand-patch ခြေရာ C line %d:' % (p_line+1))
print('   ', cur[p_line].strip()[:200])

type_arg = None
for m in re.finditer(r'[\w]+\s*\(\s*cs\s*,\s*(.+?)\)\s*;', cur[p_line]):
    a = m.group(1).strip()
    if not a.startswith('"'):
        type_arg = a
        break

pri = open(PRI, encoding='utf-8').read().split('\n')
i_c_eq = None
for k in range(max(0, p_line-10), min(len(pri), p_line+11)):
    if re.search(r'"[^"]* = [^"]*"', pri[k]):
        i_c_eq = k
        break
if i_c_eq is None:
    print('PRISTINE_EQ_NOT_FOUND')
    dump(pri, p_line-10, p_line+12, 'pristine region')
    sys.exit(2)
print('pristine " = " line %d: %s' % (i_c_eq+1, pri[i_c_eq].strip()))

if not type_arg:
    varargs = []
    for k in range(max(0, i_c_eq-8), i_c_eq):
        m = re.match(r'\s*[\w]+\s*\(\s*cs\s*,\s*(.+?)\)\s*;\s*$', pri[k])
        if m and not m.group(1).strip().startswith('"'):
            varargs.append(m.group(1).strip())
    if len(varargs) >= 2:
        type_arg = varargs[-2]
    else:
        print('TYPE_ARG_NOT_FOUND — varargs:', varargs)
        dump(pri, i_c_eq-9, i_c_eq+4, 'pristine region')
        sys.exit(2)
print('type_arg =', type_arg)

fname = None
for k in range(i_c_eq, max(0, i_c_eq-400), -1):
    l = pri[k]
    ls = l.strip()
    if re.match(r'^[A-Za-z_][\w \t\*]*\**\s*[A-Za-z_]\w*\s*\(', l) and not ls.startswith(('if','for','while','switch','return','else')):
        m = re.search(r'([A-Za-z_]\w*)\s*\(', l)
        if m:
            fname = m.group(1)
            break
print('function =', fname)

# ၃) .mmc ထဲ နေရာ ချိတ်
ml_i = None
if fname:
    for i, l in enumerate(mlines):
        if re.search(r'\b%s\s*\(' % re.escape(fname), l):
            ml_i = i
            break
if ml_i is None:
    ta = type_arg
    for i, l in enumerate(mlines):
        if ta in l or ta.replace('->', '.') in l:
            for k in range(i+1, min(len(mlines), i+8)):
                if re.search(r'"[^"]* = [^"]*"', mlines[k]):
                    ml_i = i
                    break
        if ml_i:
            break
if ml_i is None:
    print('MMC_FUNC_NOT_FOUND — .mmc ထဲ function နေရာ မတွေ့')
    sys.exit(2)

depth = 0; seen = False; b_end = ml_i
for i in range(ml_i, min(ml_i+500, len(mlines))):
    depth += mlines[i].count('{') - mlines[i].count('}')
    if '{' in mlines[i]:
        seen = True
    if seen and depth <= 0:
        b_end = i
        break
print('.mmc function lines %d-%d' % (ml_i+1, b_end+1))

beq = [i for i in range(ml_i, b_end+1) if re.search(r'"[^"]* = [^"]*"', mlines[i])]
if len(beq) != 1:
    print('AMBIGUOUS_EQ — " = " လိုင်း %d ခု ရှိနေ' % len(beq))
    dump(mlines, ml_i, min(b_end+1, ml_i+80), 'mmc function')
    sys.exit(2)
i_eq = beq[0]
print('.mmc " = " line %d: %s' % (i_eq+1, mlines[i_eq].strip()))

ta = type_arg
cands = [i for i in range(max(ml_i, i_eq-8), i_eq) if ta in mlines[i]]
if not cands:
    ta2 = ta.replace('->', '.')
    cands = [i for i in range(max(ml_i, i_eq-8), i_eq) if ta2 in mlines[i]]
    if cands:
        ta = ta2
if not cands:
    print('MMC_TYPE_LINE_NOT_FOUND — type_arg(%s) ကို .mmc ထဲ မတွေ့' % ta)
    dump(mlines, ml_i, min(b_end+1, ml_i+80), 'mmc function')
    sys.exit(2)
i_ty = cands[-1]
print('.mmc type line %d: %s' % (i_ty+1, mlines[i_ty].strip()))

i_gen = None
for i in range(i_eq+1, min(i_eq+6, b_end+1)):
    if mlines[i].strip():
        i_gen = i
        break
if i_gen is None or re.search(r'";|\\n"', mlines[i_gen]):
    print('GEN_LINE_NOT_FOUND သို့ tail နဲ့ ရောနေ')
    dump(mlines, i_eq-2, i_eq+7, 'around eq')
    sys.exit(2)
print('.mmc gen line %d: %s' % (i_gen+1, mlines[i_gen].strip()))

def swap_lit(line, new):
    return re.sub(r'"[^"]*"', lambda m: '"' + new + '"', line, count=1)

L1 = swap_lit(mlines[i_eq], ' = (')
L2 = mlines[i_ty]
L3 = swap_lit(mlines[i_eq], '*)(')
L5 = swap_lit(mlines[i_eq], ')')

new = mlines[:i_eq] + [L1, L2, L3] + mlines[i_eq+1:i_gen+1] + [L5] + mlines[i_gen+1:]
shutil.copy(MMC, 'archive/backups/compiler_v14.mmc.pre_cast.bak')
open(MMC, 'w', encoding='utf-8').write('\n'.join(new))
print('PATCHED — %d လိုင်း → %d လိုင်း' % (len(mlines), len(new)))
print('---- patch နောက် region ----')
for i in range(max(0, i_eq-6), min(len(new), i_eq+10)):
    print('%5d | %s' % (i+1, new[i]))
print('OK_CAST_MMC_PATCH')
