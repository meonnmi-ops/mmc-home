#include <cuda_runtime.h>
#include <stdio.h>

int mmc_gpu_launch0(void (*kernel)(void)) {
    cudaError_t err = cudaLaunchKernel((const void*)kernel, dim3(1), dim3(1), NULL, 0, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "mmc_gpu_launch0 error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

void* mmc_gpu_malloc(int bytes) {
    void* p = 0;
    if (cudaMalloc(&p, bytes) != cudaSuccess) return 0;
    return p;
}

int mmc_gpu_copy_to(void* dst, void* src, int bytes) {
    return cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess ? 0 : 1;
}

int mmc_gpu_copy_from(void* dst, void* src, int bytes) {
    return cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost) == cudaSuccess ? 0 : 1;
}

/* 2 pointers + int ပါတဲ့ kernel အတွက် launch (MMC မှာ & မရှိတာကို C glue နဲ့ ဖြေရှင်း) */
int mmc_gpu_launch2pi_impl(void* kern, void* a, void* b, int n, int grid, int block) {
    void* args[3];
    args[0] = &a; args[1] = &b; args[2] = &n;
    cudaError_t err = cudaLaunchKernel(kern, dim3(grid), dim3(block), args, 0, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "mmc_gpu_launch2pi_impl error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int mmc_gpu_launch_ln_impl(void* kern, void* x, void* g, void* b, void* y, int D, double eps, int M) {
    void* args[6];
    args[0] = &x; args[1] = &g; args[2] = &b; args[3] = &y; args[4] = &D; args[5] = &eps;
    cudaError_t err = cudaLaunchKernel(kern, dim3(M), dim3(1), args, 0, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "mmc_gpu_launch_ln_impl error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

/* matmul kernel launch — 3 pointers + 3 ints */
int mmc_gpu_launch3p3i_impl(void* kern, void* a, void* b, void* c, int d1, int d2, int d3, int grid, int block) {
    void* args[] = {&a, &b, &c, &d1, &d2, &d3};
    cudaError_t err = cudaLaunchKernel(kern, dim3(grid), dim3(block), args, 0, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "mmc_gpu_launch3p3i_impl error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}
