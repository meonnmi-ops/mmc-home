#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE 32

// C[M,N] = A[M,K] @ B[K,N] — tiled with shared memory
__global__ void mm_fwd_tiled(const float* A, const float* B, float* C,
                              int M, int K, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        // Load tile into shared memory
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        if (row < M && a_col < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (b_row < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial dot product from tile
        #pragma unroll
        for (int k = 0; k < TILE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
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

    dim3 block(TILE, TILE);
    dim3 grid((N+TILE-1)/TILE, (M+TILE-1)/TILE);

    // Warmup
    mm_fwd_tiled<<<grid, block>>>(dA, dB, dC, M, K, N);
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < 100; i++)
        mm_fwd_tiled<<<grid, block>>>(dA, dB, dC, M, K, N);
    cudaEventRecord(t1);
    cudaDeviceSynchronize();
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);

    cudaMemcpy(C, dC, sc*sizeof(float), cudaMemcpyDeviceToHost);
    save("/tmp/C.bin", C, sc);

    printf("GPU mm_fwd_tiled: %dx%dx%d, %.4f ms/run (100-run avg)\n", M, K, N, ms/100);
    printf("GFLOPS: %.2f\n", (2.0*M*K*N)/(ms/100/1000.0)/1e9);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(A); free(B); free(C);
    return 0;
}
