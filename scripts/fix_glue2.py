#!/usr/bin/env python3
import io, shutil

def insert_before_last_endif(s, block):
    lines = s.rstrip('\n').split('\n')
    idx = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == '#endif':
            idx = i
            break
    if idx is None:
        return '\n'.join(lines + block) + '\n', 'append'
    return '\n'.join(lines[:idx] + block + lines[idx:]) + '\n', 'before last #endif'

H = 'runtime/runtime_glue.h'
s = io.open(H, encoding='utf-8').read()
block = ['']
if 'mmc_gpu_launch3p3i' not in s:
    block += [
        '/* matmul kernel launch — 3 pointers + 3 ints */',
        'int mmc_gpu_launch3p3i_impl(void* kern, void* a, void* b, void* c, int d1, int d2, int d3, int grid, int block);',
        '#define mmc_gpu_launch3p3i(kern, a, b, c, d1, d2, d3, grid, block) mmc_gpu_launch3p3i_impl((void*)(kern), (a), (b), (c), (d1), (d2), (d3), (grid), (block))',
    ]
if 'mmc_now_ms' not in s:
    block += ['/* benchmark timing (ms) */', 'double mmc_now_ms(void);']
if len(block) > 1:
    shutil.copy(H, 'archive/backups/runtime_glue.h.pre_step1.bak')
    s2, where = insert_before_last_endif(s, block)
    io.open(H, 'w', encoding='utf-8').write(s2)
    print('GLUE.H OK — %s' % where)
else:
    print('glue.h — ရှိပြီးသား — ချန်')

C = 'runtime/runtime_gpu.cu'
s = io.open(C, encoding='utf-8').read()
if 'mmc_gpu_launch3p3i_impl' not in s:
    shutil.copy(C, 'archive/backups/runtime_gpu.cu.pre_step1.bak')
    impl = (
        '\n/* matmul kernel launch — 3 pointers + 3 ints */\n'
        'int mmc_gpu_launch3p3i_impl(void* kern, void* a, void* b, void* c, int d1, int d2, int d3, int grid, int block) {\n'
        '    void* args[] = {&a, &b, &c, &d1, &d2, &d3};\n'
        '    cudaError_t err = cudaLaunchKernel(kern, dim3(grid), dim3(block), args, 0, 0);\n'
        '    if (err != cudaSuccess) {\n'
        '        fprintf(stderr, "mmc_gpu_launch3p3i_impl error: %s\\n", cudaGetErrorString(err));\n'
        '        return -1;\n'
        '    }\n'
        '    return 0;\n'
        '}\n'
    )
    if not s.endswith('\n'):
        s += '\n'
    io.open(C, 'w', encoding='utf-8').write(s + impl)
    print('GPU.CU OK — append')
else:
    print('gpu.cu — ရှိပြီးသား — ချန်')

G = 'runtime/runtime_glue.c'
s = io.open(G, encoding='utf-8').read()
if 'mmc_now_ms' not in s:
    shutil.copy(G, 'archive/backups/runtime_glue.c.pre_step1.bak')
    impl = (
        '\n/* benchmark timing — monotonic ms */\n'
        '#include <time.h>\n'
        'double mmc_now_ms(void) {\n'
        '    struct timespec ts;\n'
        '    clock_gettime(CLOCK_MONOTONIC, &ts);\n'
        '    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;\n'
        '}\n'
    )
    if not s.endswith('\n'):
        s += '\n'
    io.open(G, 'w', encoding='utf-8').write(s + impl)
    print('GLUE.C OK — append')
else:
    print('glue.c — ရှိပြီးသား — ချန်')
print('ALL DONE')
