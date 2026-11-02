#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

static void load(const char* p, float* b, int n) {
    FILE* f = fopen(p, "rb"); if (!f) { perror(p); exit(1); }
    fread(b, sizeof(float), n, f); fclose(f);
}
static void save(const char* p, float* b, int n) {
    FILE* f = fopen(p, "wb"); if (!f) { perror(p); exit(1); }
    fwrite(b, sizeof(float), n, f); fclose(f);
}

int main(int argc, char** argv) {
    int M = 128, K = 256, N = 1024;
    if (argc >= 4) { M = atoi(argv[1]); K = atoi(argv[2]); N = atoi(argv[3]); }

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

    cublasHandle_t handle;
    cublasCreate(&handle);

    // Row-major C[M,N] = A[M,K] @ B[K,N]
    // cuBLAS is column-major → compute C^T = B^T @ A^T
    // cuBLAS call: C_col(N,M) = B_col(N,K) @ A_col(K,M)
    //   cublasSgemm(handle, op, op, N, M, K, alpha, B, N, A, K, beta, C, N)
    const float alpha = 1.0f, beta = 0.0f;

    // Warmup
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < 100; i++)
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
    cudaEventRecord(t1);
    cudaDeviceSynchronize();
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);

    cudaMemcpy(C, dC, sc*sizeof(float), cudaMemcpyDeviceToHost);
    save("/tmp/C.bin", C, sc);

    printf("GPU mm_cublas: %dx%dx%d, %.4f ms/run (100-run avg)\n", M, K, N, ms/100);
    printf("GFLOPS: %.2f\n", (2.0*M*K*N)/(ms/100/1000.0)/1e9);

    cublasDestroy(handle);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(A); free(B); free(C);
    return 0;
}
