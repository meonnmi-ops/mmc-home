#!/usr/bin/env python3
import io, re, shutil, sys, glob

H = 'runtime/runtime_glue.h'
s = io.open(H, encoding='utf-8').read()
s0 = s
for fname in ('mmc_gpu_launch2pi', 'mmc_gpu_launch_ln'):
    if fname + '_impl' in s:
        print('%s — ပြင်ပြီးသား — ချန်' % fname)
        continue
    m = re.search(r'^\s*(?:extern\s+)?int\s+' + fname + r'\s*\(([^)]*)\)\s*;', s, re.M)
    if not m:
        print('ABORT — %s decl မတွေ့ runtime_glue.h ထဲ' % fname)
        sys.exit(1)
    params = [p.strip() for p in m.group(1).split(',') if p.strip()]
    names = []
    for i, p in enumerate(params):
        toks = p.replace('*', ' * ').split()
        last = toks[-1] if toks else ''
        bad = (last == '') or (not last.isidentifier()) or last in ('void','int','double','float','char','long','short','unsigned','signed','const')
        names.append('a%d' % (i + 1) if bad else last)
    names[0] = 'kern'
    decl = 'int %s_impl(%s);' % (fname, m.group(1))
    macro = '#define %s(%s) %s_impl((void*)(kern), %s)' % (fname, ', '.join(names), fname, ', '.join('(%s)' % n for n in names[1:]))
    s = s[:m.start()] + decl + '\n' + macro + s[m.end():]
    print('MACRO OK: %s' % fname)
if s != s0:
    shutil.copy(H, 'archive/backups/runtime_glue.h.pre_launchcast.bak')
    io.open(H, 'w', encoding='utf-8').write(s)
    print('GLUE.H PATCH OK')
else:
    print('GLUE.H — ပြောင်းလဲမှု မရှိ')

for f in sorted(glob.glob('runtime/*.c') + glob.glob('runtime/*.cu') + glob.glob('runtime/*.h')):
    if f.endswith('runtime_glue.h'):
        continue
    s = io.open(f, encoding='utf-8').read()
    s2 = re.sub(r'\bmmc_gpu_launch2pi\b(?!_impl)', 'mmc_gpu_launch2pi_impl', s)
    s2 = re.sub(r'\bmmc_gpu_launch_ln\b(?!_impl)', 'mmc_gpu_launch_ln_impl', s2)
    if s2 != s:
        shutil.copy(f, 'archive/backups/%s.pre_launchcast.bak' % f.split('/')[-1])
        io.open(f, 'w', encoding='utf-8').write(s2)
        print('RENAME OK: %s' % f)
print('ALL DONE')
