#ifdef __cplusplus
extern "C" {
#endif
/* runtime_glue.h - prototypes for the glue helpers provided by
   runtime_glue.c. Generated self-hosted code (v4+) includes this after
   runtime_v5.h so calls to these helpers compile cleanly. */
#ifndef RUNTIME_GLUE_H
#define RUNTIME_GLUE_H

#include <stddef.h>

void* mmc_malloc(size_t n);
void mmc_free(void* p);
int mmc_string_compare(const char* a, const char* b);
char* mmc_string_concat(const char* a, const char* b);
char* mmc_string_substring(const char* s, int a, int b);
char* mmc_int_to_string(int n);
double mmc_string_to_float(const char* s);
void mmc_print_int(int n);
void mmc_print_float(double d);
void mmc_print_newline();
char* mmc_read_file(const char* filename);
int mmc_write_file(const char* fn, const char* c);
int mmc_array_size(MMCArray* arr);
void* mmc_array_get(MMCArray* arr, int idx);

#endif /* RUNTIME_GLUE_H */

#ifdef __cplusplus
}
#endif
int mmc_gpu_launch0(void (*kernel)(void));
#ifndef MMC_GLUE_GPU_EXTRA
#define MMC_GLUE_GPU_EXTRA
#define MMC_TID (((int)blockIdx.x) * ((int)blockDim.x) + ((int)threadIdx.x))
void* mmc_gpu_malloc(int bytes);
int mmc_gpu_copy_to(void* dst, void* src, int bytes);
int mmc_gpu_copy_from(void* dst, void* src, int bytes);
int mmc_gpu_launch2pi_impl(void* kern, void* a, void* b, int n, int grid, int block);
#define mmc_gpu_launch2pi(kern, a, b, n, grid, block) mmc_gpu_launch2pi_impl((void*)(kern), (a), (b), (n), (grid), (block))
#endif
#ifndef MMC_BID
#define MMC_BID ((int)blockIdx.x)

/* matmul kernel launch — 3 pointers + 3 ints */
int mmc_gpu_launch3p3i_impl(void* kern, void* a, void* b, void* c, int d1, int d2, int d3, int grid, int block);
#define mmc_gpu_launch3p3i(kern, a, b, c, d1, d2, d3, grid, block) mmc_gpu_launch3p3i_impl((void*)(kern), (a), (b), (c), (d1), (d2), (d3), (grid), (block))
/* benchmark timing (ms) */
#ifdef __cplusplus
extern "C" {
#endif
double mmc_now_ms(void);
#ifdef __cplusplus
}
#endif
#endif
int mmc_gpu_launch_ln_impl(void* kern, void* x, void* g, void* b, void* y, int D, double eps, int M);
#define mmc_gpu_launch_ln(kern, x, g, b, y, D, eps, M) mmc_gpu_launch_ln_impl((void*)(kern), (x), (g), (b), (y), (D), (eps), (M))
