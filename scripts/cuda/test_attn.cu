#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define MAX_T 512

// Multi-head attention forward. One thread per (head, row).
// Q, K, V: [T, D] row-major. H heads, head_dim = D/H.
__global__ void attn_fwd_multihead(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ out,
    int T, int D, int H, float scale)
{
    int h    = blockIdx.x;
    int row  = blockIdx.y;
    int head_dim = D / H;

    float s[MAX_T];  // scores for this (head, row) — local memory

    // Step 1: s[col] = Q[row] . K[col] * scale
    for (int col = 0; col < T; col++) {
        float sum = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            sum += Q[row * D + h * head_dim + d]
                 * K[col * D + h * head_dim + d];
        }
        s[col] = sum * scale;
    }

    // Step 2: softmax (max, exp, sum, normalize)
    float mx = -1e30f;
    for (int col = 0; col < T; col++)
        if (s[col] > mx) mx = s[col];

    float se = 0.0f;
    for (int col = 0; col < T; col++) {
        s[col] = expf(s[col] - mx);
        se += s[col];
    }
    for (int col = 0; col < T; col++)
        s[col] /= se;

    // Step 3: out[row, h*head_dim + d] = sum_col s[col] * V[col, h*head_dim + d]
    for (int d = 0; d < head_dim; d++) {
        float sum = 0.0f;
        for (int col = 0; col < T; col++) {
            sum += s[col] * V[col * D + h * head_dim + d];
        }
        out[row * D + h * head_dim + d] = sum;
    }
}

static void ld(const char* p, void* b, size_t by) {
    FILE* f = fopen(p, "rb"); if (!f) { perror(p); exit(1); }
    fread(b, 1, by, f); fclose(f);
}
static void sv(const char* p, const void* b, size_t by) {
    FILE* f = fopen(p, "wb"); if (!f) { perror(p); exit(1); }
    fwrite(b, 1, by, f); fclose(f);
}

int main(int argc, char** argv) {
    int T = 128, D = 256, H = 8;
    if (argc >= 4) { T = atoi(argv[1]); D = atoi(argv[2]); H = atoi(argv[3]); }

    if (T > MAX_T) { fprintf(stderr, "T > MAX_T\n"); return 1; }

    float scale = 1.0f / sqrtf((float)(D / H));
    size_t sz = (size_t)T * D;

    float *hQ = (float*)malloc(sz*sizeof(float));
    float *hK = (float*)malloc(sz*sizeof(float));
    float *hV = (float*)malloc(sz*sizeof(float));
    float *hO = (float*)malloc(sz*sizeof(float));
    ld("/tmp/attn_q.bin", hQ, sz*sizeof(float));
    ld("/tmp/attn_k.bin", hK, sz*sizeof(float));
    ld("/tmp/attn_v.bin", hV, sz*sizeof(float));

    float *dQ, *dK, *dV, *dO;
    cudaMalloc(&dQ, sz*sizeof(float));
    cudaMalloc(&dK, sz*sizeof(float));
    cudaMalloc(&dV, sz*sizeof(float));
    cudaMalloc(&dO, sz*sizeof(float));
    cudaMemcpy(dQ, hQ, sz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dK, hK, sz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, hV, sz*sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(H, T);
    dim3 block(1);

    // Warmup
    attn_fwd_multihead<<<grid, block>>>(dQ, dK, dV, dO, T, D, H, scale);
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < 100; i++)
        attn_fwd_multihead<<<grid, block>>>(dQ, dK, dV, dO, T, D, H, scale);
    cudaEventRecord(t1);
    cudaDeviceSynchronize();
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);

    cudaMemcpy(hO, dO, sz*sizeof(float), cudaMemcpyDeviceToHost);
    sv("/tmp/attn_out.bin", hO, sz*sizeof(float));

    printf("Attention fwd: %.4f ms/run  [T=%d, D=%d, H=%d]\n", ms/100, T, D, H);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    free(hQ); free(hK); free(hV); free(hO);
    return 0;
}
