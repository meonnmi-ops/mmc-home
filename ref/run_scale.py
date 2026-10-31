"""Scale test runner — pure Python (no bash set -e quirks)."""
import subprocess, os, sys, re
from pathlib import Path

os.chdir(str(Path.home() / 'mmc_publish'))

# Backup
orig = Path('src/train_multi_step.mmc').read_text()

def patch(D, DFF, H):
    s = orig
    s = re.sub(r'(တည်ဆောက် ကိန်း D\s*=\s*)\d+', rf'\g<1>{D}', s)
    s = re.sub(r'(တည်ဆောက် ကိန်း DFF\s*=\s*)\d+', rf'\g<1>{DFF}', s)
    s = re.sub(r'(တည်ဆောက် ကိန်း H\s*=\s*)\d+', rf'\g<1>{H}', s)
    Path('src/train_multi_step.mmc').write_text(s)

def run(cmd, env=None, check=True, quiet=False):
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if not quiet and r.stdout.strip():
        print(r.stdout.strip())
    if r.stderr.strip() and r.returncode != 0:
        print("STDERR:", r.stderr.strip()[:500])
    if check and r.returncode != 0:
        print(f"❌ FAILED: {' '.join(cmd)}")
        return False
    return True

all_pass = True
for D in [20, 40, 60, 80, 100]:
    DFF = D * 2
    H = 4
    print()
    print("═" * 50)
    print(f"  D={D}, DFF={DFF}, H={H}, HD={D//H}")
    print("═" * 50)

    patch(D, DFF, H)
    env = os.environ.copy()
    env.update({'D': str(D), 'DFF': str(DFF), 'H': str(H)})

    # Step 1: gen ref
    print("  [1/4] Gen ref...")
    r = subprocess.run(['python3','ref/train_scale_gen.py'], env=env, capture_output=True, text=True)
    for line in r.stdout.split('\n'):
        if 'PyTorch' in line or 'waiting' in line or 'Generated' in line:
            print(line)

    # Step 2: build MMC
    print("  [2/4] Build MMC...")
    if not run(['./mmc_v14','src/train_multi_step.mmc','build/train_multi_step.c'], quiet=True):
        all_pass = False; break
    # add include
    c = Path('build/train_multi_step.c').read_text()
    if '#include "block_lib.h"' not in c.split('\n')[0]:
        Path('build/train_multi_step.c').write_text('#include "block_lib.h"\n' + c)
    if not run(['gcc','-O2','-I','runtime','-I','build','-c','build/train_multi_step.c','-o','build/train_multi_step.o'], quiet=True):
        all_pass = False; break
    if not run(['gcc','-O2','build/train_multi_step.o','build/block_lib.o','build/stubs.o','-o','build/train_multi_step','-lm'], quiet=True):
        all_pass = False; break

    # Step 3: run MMC
    print("  [3/4] Run MMC...")
    r = subprocess.run(['./build/train_multi_step'], capture_output=True, text=True)
    for line in r.stdout.split('\n'):
        if 'loss' in line or 'DONE' in line:
            print(line)

    # Step 4: compare
    print("  [4/4] Compare...")
    r = subprocess.run(['python3','ref/train_scale_gen.py'], env=env, capture_output=True, text=True)
    for line in r.stdout.split('\n'):
        if 'PASS' in line or 'FAIL' in line or 'missing' in line:
            print(line)
    if r.returncode != 0:
        all_pass = False
        break

# Restore original
Path('src/train_multi_step.mmc').write_text(orig)
print()
if all_pass:
    print("🎉 ALL D VALUES PASS")
else:
    print("❌ Some failed — debug above")
