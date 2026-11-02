#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// C[M,N] = A[M,K] @ B[K,N] — naive CUDA
__global__ void mm_fwd_kernel(const float* A, const float* B, float* C,
                               int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; k++) sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;
}

static void load(const char* p, float* b, int n) {
    FILE* f = fopen(p, "rb"); if (!f) { perror(p); exit(1); }
    fread(b, sizeof(float), n, f); fclose(f);
}
static void save(const char* p, float* b, int n) {
    FILE* f = fopen(p, "wb"); if (!f) { perror(p); exit(1); }
    fwrite(b, sizeof(float), n, f); fclose(f);
}

int main() {
    int M = 128, K = 256, N = 1024;
    size_t sa = (size_t)M*K, sb = (size_t)K*N, sc = (size_t)M*N;

    float *A = (float*)malloc(sa*sizeof(float));
    float *B = (float*)malloc(sb*sizeof(float));
    float *C = (float*)malloc(sc*sizeof(float));

    load("/tmp/A.bin", A, sa);
    load("/tmp/B.bin", B, sb);

    float *dA, *dB, *dC;
    cudaMalloc(&dA, sa*sizeof(float));
    cudaMalloc(&dB, sb*sizeof(float));
    cudaMalloc(&dC, sc*sizeof(float));

    cudaMemcpy(dA, A, sa*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sb*sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N+15)/16, (M+15)/16);

    // Warmup + timed run
    mm_fwd_kernel<<<grid, block>>>(dA, dB, dC, M, K, N);
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < 10; i++)
        mm_fwd_kernel<<<grid, block>>>(dA, dB, dC, M, K, N);
    cudaEventRecord(t1);
    cudaDeviceSynchronize();
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);

    cudaMemcpy(C, dC, sc*sizeof(float), cudaMemcpyDeviceToHost);
    save("/tmp/C.bin", C, sc);

    printf("GPU mm_fwd: %dx%dx%d, %.3f ms/run (10-run avg)\n", M, K, N, ms/10);
    printf("GFLOPS: %.2f\n", (2.0*M*K*N)/(ms/10/1000.0)/1e9);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(A); free(B); free(C);
    return 0;
}
