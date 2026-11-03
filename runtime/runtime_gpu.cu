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
