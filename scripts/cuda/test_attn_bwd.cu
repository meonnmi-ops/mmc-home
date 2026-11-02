#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define MAX_T 512

__global__ void attn_bwd_multihead(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ dO,
    float* __restrict__ dQ,
    float* __restrict__ dK,
    float* __restrict__ dV,
    int T, int D, int H, float scale)
{
    int h = blockIdx.x;
    int row = blockIdx.y;
    int head_dim = D / H;
    int off = h * head_dim;

    float scores[MAX_T], softm[MAX_T], dsp[MAX_T], dsc[MAX_T];

    for (int c = 0; c < T; c++) {
        float s = 0.0f;
        for (int d = 0; d < head_dim; d++)
            s += Q[row*D + off + d] * K[c*D + off + d];
        scores[c] = s * scale;
    }
    float mx = -1e30f;
    for (int c = 0; c < T; c++) if (scores[c] > mx) mx = scores[c];
    float sm = 0.0f;
    for (int c = 0; c < T; c++) { softm[c] = expf(scores[c]-mx); sm += softm[c]; }
    for (int c = 0; c < T; c++) softm[c] /= sm;

    for (int c = 0; c < T; c++)
        for (int d = 0; d < head_dim; d++)
            atomicAdd(&dV[c*D + off + d], softm[c] * dO[row*D + off + d]);

    for (int c = 0; c < T; c++) {
        float s = 0.0f;
        for (int d = 0; d < head_dim; d++)
            s += dO[row*D + off + d] * V[c*D + off + d];
        dsp[c] = s;
    }

    float dot = 0.0f;
    for (int c = 0; c < T; c++) dot += dsp[c] * softm[c];
    for (int c = 0; c < T; c++) dsc[c] = softm[c] * (dsp[c] - dot) * scale;

    for (int d = 0; d < head_dim; d++) {
        float s = 0.0f;
        for (int c = 0; c < T; c++) s += dsc[c] * K[c*D + off + d];
        dQ[row*D + off + d] = s;
    }

    for (int c = 0; c < T; c++)
        for (int d = 0; d < head_dim; d++)
            atomicAdd(&dK[c*D + off + d], dsc[c] * Q[row*D + off + d]);
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
    float *hdO = (float*)malloc(sz*sizeof(float));
    float *hdQ = (float*)malloc(sz*sizeof(float));
    float *hdK = (float*)malloc(sz*sizeof(float));
    float *hdV = (float*)malloc(sz*sizeof(float));

    ld("/tmp/attn_q.bin",  hQ,  sz*sizeof(float));
    ld("/tmp/attn_k.bin",  hK,  sz*sizeof(float));
    ld("/tmp/attn_v.bin",  hV,  sz*sizeof(float));
    ld("/tmp/attn_do.bin", hdO, sz*sizeof(float));

    float *dQin,*dKin,*dVin,*dO,*dQ,*dK,*dV;
    cudaMalloc(&dQin, sz*sizeof(float));
    cudaMalloc(&dKin, sz*sizeof(float));
    cudaMalloc(&dVin, sz*sizeof(float));
    cudaMalloc(&dO,   sz*sizeof(float));
    cudaMalloc(&dQ,   sz*sizeof(float));
    cudaMalloc(&dK,   sz*sizeof(float));
    cudaMalloc(&dV,   sz*sizeof(float));
    cudaMemcpy(dQin, hQ,  sz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dKin, hK,  sz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dVin, hV,  sz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dO,   hdO, sz*sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(H, T), block(1);

    cudaMemset(dQ, 0, sz*sizeof(float));
    cudaMemset(dK, 0, sz*sizeof(float));
    cudaMemset(dV, 0, sz*sizeof(float));
    attn_bwd_multihead<<<grid, block>>>(dQin, dKin, dVin, dO, dQ, dK, dV, T, D, H, scale);
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaMemset(dQ, 0, sz*sizeof(float));
    cudaMemset(dK, 0, sz*sizeof(float));
    cudaMemset(dV, 0, sz*sizeof(float));
    cudaEventRecord(t0);
    for (int it = 0; it < 10; it++) {
        cudaMemset(dQ, 0, sz*sizeof(float));
        cudaMemset(dK, 0, sz*sizeof(float));
        cudaMemset(dV, 0, sz*sizeof(float));
        attn_bwd_multihead<<<grid, block>>>(dQin, dKin, dVin, dO, dQ, dK, dV, T, D, H, scale);
    }
    cudaEventRecord(t1);
    cudaDeviceSynchronize();
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);

    cudaMemcpy(hdQ, dQ, sz*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hdK, dK, sz*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hdV, dV, sz*sizeof(float), cudaMemcpyDeviceToHost);
    sv("/tmp/attn_dq.bin", hdQ, sz*sizeof(float));
    sv("/tmp/attn_dk.bin", hdK, sz*sizeof(float));
    sv("/tmp/attn_dv.bin", hdV, sz*sizeof(float));

    printf("Attention bwd: %.4f ms/run (10-avg incl. memsets)\n", ms/10);

    cudaFree(dQin); cudaFree(dKin); cudaFree(dVin); cudaFree(dO);
    cudaFree(dQ);   cudaFree(dK);   cudaFree(dV);
    free(hQ); free(hK); free(hV); free(hdO);
    free(hdQ); free(hdK); free(hdV);
    return 0;
}
