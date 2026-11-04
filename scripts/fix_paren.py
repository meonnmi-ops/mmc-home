#!/usr/bin/env python3
import io, shutil, sys
P = 'src/compiler_v14.mmc'
OLD = 'mmc_string_concat(gen_expr(init), "))")'
NEW = 'mmc_string_concat(gen_expr(init), ")")'
s = io.open(P, encoding='utf-8').read()
c = s.count(OLD)
print('ရှာတွေ့: %d (၁ ဖြစ်ရမယ်)' % c)
if c != 1:
    print('ABORT — ဘာမှ မပြင်ရသေး')
    sys.exit(1)
shutil.copy(P, 'archive/backups/compiler_v14.mmc.pre_parenfix.bak')
io.open(P, 'w', encoding='utf-8').write(s.replace(OLD, NEW, 1))
print('PAREN FIX OK')
