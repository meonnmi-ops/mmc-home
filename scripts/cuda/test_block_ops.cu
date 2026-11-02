#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "block_ops.h"

static void ld(const char* p, float* b, int n) {
    FILE* f=fopen(p,"rb"); if(!f){perror(p);exit(1);} fread(b,4,n,f); fclose(f);
}
static void sv(const char* p, float* b, int n) {
    FILE* f=fopen(p,"wb"); if(!f){perror(p);exit(1);} fwrite(b,4,n,f); fclose(f);
}

int main() {
    const int M=4, D=32;
    float eps = 1e-5f;
    size_t sx = M*D;

    // ---- Test LN fwd ----
    float *hX=(float*)malloc(sx*4), *hG=(float*)malloc(D*4), *hB=(float*)malloc(D*4);
    ld("/tmp/bo_x.bin", hX, sx);
    ld("/tmp/bo_g.bin", hG, D);
    ld("/tmp/bo_b.bin", hB, D);

    float *dX, *dG, *dB, *dY, *dMean, *dRstd;
    cudaMalloc(&dX,sx*4); cudaMalloc(&dG,D*4); cudaMalloc(&dB,D*4);
    cudaMalloc(&dY,sx*4); cudaMalloc(&dMean,M*4); cudaMalloc(&dRstd,M*4);
    cudaMemcpy(dX,hX,sx*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dG,hG,D*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dB,hB,D*4,cudaMemcpyHostToDevice);

    ln_fwd_dev(dX, dG, dB, dY, dMean, dRstd, M, D, eps);
    cudaDeviceSynchronize();

    float* hY=(float*)malloc(sx*4);
    cudaMemcpy(hY,dY,sx*4,cudaMemcpyDeviceToHost);
    sv("/tmp/bo_y.bin", hY, sx);

    // ---- Test LN bwd ----
    float* hdy=(float*)malloc(sx*4);
    ld("/tmp/bo_dy.bin", hdy, sx);
    float *dDY, *dDX, *dDgamma, *dDbeta;
    cudaMalloc(&dDY,sx*4); cudaMalloc(&dDX,sx*4);
    cudaMalloc(&dDgamma,D*4); cudaMalloc(&dDbeta,D*4);
    cudaMemcpy(dDY,hdy,sx*4,cudaMemcpyHostToDevice);
    cudaMemset(dDgamma,0,D*4); cudaMemset(dDbeta,0,D*4);
    ln_bwd_dev(dDY, dX, dG, dMean, dRstd, dDX, dDgamma, dDbeta, M, D);
    cudaDeviceSynchronize();

    float *hDX=(float*)malloc(sx*4), *hDg=(float*)malloc(D*4), *hDb=(float*)malloc(D*4);
    cudaMemcpy(hDX,dDX,sx*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(hDg,dDgamma,D*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(hDb,dDbeta,D*4,cudaMemcpyDeviceToHost);
    sv("/tmp/bo_dx.bin", hDX, sx);
    sv("/tmp/bo_dg.bin", hDg, D);
    sv("/tmp/bo_db.bin", hDb, D);

    // ---- Test GELU fwd+bwd ----
    float *dY2, *dDY2;
    cudaMalloc(&dY2,sx*4); cudaMalloc(&dDY2,sx*4);
    gelu_fwd_dev(dX, dY2, sx);
    cudaDeviceSynchronize();
    float *hY2=(float*)malloc(sx*4);
    cudaMemcpy(hY2,dY2,sx*4,cudaMemcpyDeviceToHost);
    sv("/tmp/bo_gelu_y.bin", hY2, sx);

    gelu_bwd_dev(dDY, dX, dDY2, sx);
    cudaDeviceSynchronize();
    float *hDX2=(float*)malloc(sx*4);
    cudaMemcpy(hDX2,dDY2,sx*4,cudaMemcpyDeviceToHost);
    sv("/tmp/bo_gelu_dx.bin", hDX2, sx);

    printf("block_ops all passed through\n");
    return 0;
}
