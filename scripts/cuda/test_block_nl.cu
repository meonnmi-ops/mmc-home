#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define MAX_T 512
static cublasHandle_t gb;

// ---- Attention fwd (1 thread per (head,row)) ----
__global__ void attn_fwd(const float* __restrict__ Q, const float* __restrict__ K,
                         const float* __restrict__ V, float* __restrict__ out,
                         int T, int D, int H, float scale) {
    int h = blockIdx.x, row = blockIdx.y;
    int hd = D/H, off = h*hd;
    float s[MAX_T];
    for (int c=0; c<T; c++) {
        float su=0.0f;
        for (int d=0; d<hd; d++) su += Q[row*D+off+d]*K[c*D+off+d];
        s[c]=su*scale;
    }
    float mx=-1e30f;
    for (int c=0; c<T; c++) if (s[c]>mx) mx=s[c];
    float se=0.0f;
    for (int c=0; c<T; c++){ s[c]=expf(s[c]-mx); se+=s[c]; }
    for (int c=0; c<T; c++) s[c]/=se;
    for (int d=0; d<hd; d++) {
        float su=0.0f;
        for (int c=0; c<T; c++) su += s[c]*V[c*D+off+d];
        out[row*D+off+d] = su;
    }
}

// ---- Attention bwd (1 thread per (head,row)) ----
__global__ void attn_bwd(const float* __restrict__ Q, const float* __restrict__ K,
                         const float* __restrict__ V, const float* __restrict__ dO,
                         float* __restrict__ dQ, float* __restrict__ dK, float* __restrict__ dV,
                         int T, int D, int H, float scale) {
    int h=blockIdx.x, row=blockIdx.y;
    int hd=D/H, off=h*hd;
    float sc[MAX_T], sm[MAX_T], dp[MAX_T], ds[MAX_T];
    for (int c=0;c<T;c++){
        float su=0.0f;
        for (int d=0;d<hd;d++) su+=Q[row*D+off+d]*K[c*D+off+d];
        sc[c]=su*scale;
    }
    float mx=-1e30f;
    for (int c=0;c<T;c++) if (sc[c]>mx) mx=sc[c];
    float se=0.0f;
    for (int c=0;c<T;c++){ sm[c]=expf(sc[c]-mx); se+=sm[c]; }
    for (int c=0;c<T;c++) sm[c]/=se;
    for (int c=0;c<T;c++)
        for (int d=0;d<hd;d++)
            atomicAdd(&dV[c*D+off+d], sm[c]*dO[row*D+off+d]);
    for (int c=0;c<T;c++){
        float su=0.0f;
        for (int d=0;d<hd;d++) su+=dO[row*D+off+d]*V[c*D+off+d];
        dp[c]=su;
    }
    float dot=0.0f;
    for (int c=0;c<T;c++) dot+=dp[c]*sm[c];
    for (int c=0;c<T;c++) ds[c]=sm[c]*(dp[c]-dot)*scale;
    for (int d=0;d<hd;d++){
        float su=0.0f;
        for (int c=0;c<T;c++) su+=ds[c]*K[c*D+off+d];
        dQ[row*D+off+d]=su;
    }
    for (int c=0;c<T;c++)
        for (int d=0;d<hd;d++)
            atomicAdd(&dK[c*D+off+d], ds[c]*Q[row*D+off+d]);
}

// ---- Row-major matmul via cuBLAS ----
static void mm(const float* A, const float* B, float* C, int M, int K, int N) {
    float a=1.0f, b=0.0f;
    cublasSgemm(gb, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &a, B, N, A, K, &b, C, N);
}
static void mm_nt(const float* A, const float* B, float* C, int M, int K, int N) {
    float a=1.0f, b=0.0f;
    cublasSgemm(gb, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &a, B, K, A, K, &b, C, N);
}
static void mm_tn(const float* A, const float* B, float* C, int M, int K, int N) {
    float a=1.0f, b=0.0f;
    cublasSgemm(gb, CUBLAS_OP_N, CUBLAS_OP_T, N, M, K, &a, B, N, A, M, &b, C, N);
}

static void ld(const char* p, float* b, int n){
    FILE* f=fopen(p,"rb"); if(!f){perror(p);exit(1);} fread(b,4,n,f); fclose(f);
}
static void sv(const char* p, float* b, int n){
    FILE* f=fopen(p,"wb"); if(!f){perror(p);exit(1);} fwrite(b,4,n,f); fclose(f);
}

int main() {
    cublasCreate(&gb);
    const int T=4, D=32, H=2;
    int sx=T*D, sD=D*D;
    float scale = 1.0f/sqrtf((float)(D/H));

    // Load inputs
    float *hX=(float*)malloc(sx*4), *hWq=(float*)malloc(sD*4), *hWk=(float*)malloc(sD*4),
          *hWv=(float*)malloc(sD*4), *hWo=(float*)malloc(sD*4), *hdY=(float*)malloc(sx*4);
    ld("/tmp/bn_x.bin", hX, sx); ld("/tmp/bn_wq.bin", hWq, sD);
    ld("/tmp/bn_wk.bin", hWk, sD); ld("/tmp/bn_wv.bin", hWv, sD);
    ld("/tmp/bn_wo.bin", hWo, sD); ld("/tmp/bn_dy.bin", hdY, sx);

    // Device allocs
    float *dX,*dWq,*dWk,*dWv,*dWo,*dQ,*dK,*dV,*dA,*dO,*dY,*dYup;
    float *dWq_g,*dWk_g,*dWv_g,*dWo_g,*dX_g,*dQ_g,*dK_g,*dV_g,*dA_g;
    cudaMalloc(&dX,sx*4); cudaMalloc(&dWq,sD*4); cudaMalloc(&dWk,sD*4);
    cudaMalloc(&dWv,sD*4); cudaMalloc(&dWo,sD*4);
    cudaMalloc(&dQ,sx*4); cudaMalloc(&dK,sx*4); cudaMalloc(&dV,sx*4);
    cudaMalloc(&dA,sx*4); cudaMalloc(&dO,sx*4); cudaMalloc(&dY,sx*4);
    cudaMalloc(&dYup,sx*4);
    cudaMalloc(&dWq_g,sD*4); cudaMalloc(&dWk_g,sD*4); cudaMalloc(&dWv_g,sD*4);
    cudaMalloc(&dWo_g,sD*4); cudaMalloc(&dX_g,sx*4);
    cudaMalloc(&dQ_g,sx*4); cudaMalloc(&dK_g,sx*4); cudaMalloc(&dV_g,sx*4);
    cudaMalloc(&dA_g,sx*4);

    cudaMemcpy(dX,hX,sx*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dWq,hWq,sD*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dWk,hWk,sD*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dWv,hWv,sD*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dWo,hWo,sD*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dYup,hdY,sx*4,cudaMemcpyHostToDevice);

    // === Forward ===
    mm(dX,dWq,dQ,T,D,D);
    mm(dX,dWk,dK,T,D,D);
    mm(dX,dWv,dV,T,D,D);
    dim3 grd(H,T), blk(1);
    attn_fwd<<<grd,blk>>>(dQ,dK,dV,dA,T,D,H,scale);
    cudaDeviceSynchronize();
    mm(dA,dWo,dO,T,D,D);
    // Y = X + O
    { float* hY=(float*)malloc(sx*4); float* hO=(float*)malloc(sx*4);
      cudaMemcpy(hO,dO,sx*4,cudaMemcpyDeviceToHost);
      for (int i=0;i<sx;i++) hY[i]=hX[i]+hO[i];
      cudaMemcpy(dY,hY,sx*4,cudaMemcpyHostToDevice);
      sv("/tmp/bn_y.bin",hY,sx); free(hY); free(hO); }

    // === Backward ===
    // dWo = A^T @ dY
    mm_tn(dA,dYup,dWo_g,D,T,D);
    // dA = dY @ Wo^T
    mm_nt(dYup,dWo,dA_g,T,D,D);
    // dQ,dK,dV from attention bwd
    cudaMemset(dQ_g,0,sx*4); cudaMemset(dK_g,0,sx*4); cudaMemset(dV_g,0,sx*4);
    attn_bwd<<<grd,blk>>>(dQ,dK,dV,dA_g,dQ_g,dK_g,dV_g,T,D,H,scale);
    cudaDeviceSynchronize();
    // dWq = X^T @ dQ, etc
    mm_tn(dX,dQ_g,dWq_g,D,T,D);
    mm_tn(dX,dK_g,dWk_g,D,T,D);
    mm_tn(dX,dV_g,dWv_g,D,T,D);
    // dX_attn = dQ@Wq^T + dK@Wk^T + dV@Wv^T
    cudaMemset(dX_g,0,sx*4);
    { float* hQ=(float*)malloc(sx*4); float* hK2=(float*)malloc(sx*4);
      float* hV2=(float*)malloc(sx*4); float* hXg=(float*)malloc(sx*4);
      cudaMemcpy(hQ,dQ_g,sx*4,cudaMemcpyDeviceToHost);
      cudaMemcpy(hK2,dK_g,sx*4,cudaMemcpyDeviceToHost);
      cudaMemcpy(hV2,dV_g,sx*4,cudaMemcpyDeviceToHost);
      cudaMemcpy(hXg,dX_g,sx*4,cudaMemcpyDeviceToHost);
      float* t1=(float*)malloc(sx*4); float* t2=(float*)malloc(sx*4);
      float* t3=(float*)malloc(sx*4);
      // use cublas to compute t1=Q@Wq^T etc — but easier: do on host
      for (int i=0;i<T;i++) for (int j=0;j<D;j++) {
          float s1=0,s2=0,s3=0;
          for (int k=0;k<D;k++) {
              s1 += hQ[i*D+k]*hWq[j*D+k];
              s2 += hK2[i*D+k]*hWk[j*D+k];
              s3 += hV2[i*D+k]*hWv[j*D+k];
          }
          hXg[i*D+j] = s1+s2+s3 + hdY[i*D+j];  // + residual
      }
      cudaMemcpy(dX_g,hXg,sx*4,cudaMemcpyHostToDevice);
      free(hQ); free(hK2); free(hV2); free(hXg); free(t1); free(t2); free(t3);
    }

    // Save outputs
    float *hY=(float*)malloc(sx*4), *hdXg=(float*)malloc(sx*4);
    float *hdWqg=(float*)malloc(sD*4), *hdWkg=(float*)malloc(sD*4),
          *hdWvg=(float*)malloc(sD*4), *hdWog=(float*)malloc(sD*4);
    cudaMemcpy(hY,dY,sx*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(hdXg,dX_g,sx*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(hdWqg,dWq_g,sD*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(hdWkg,dWk_g,sD*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(hdWvg,dWv_g,sD*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(hdWog,dWo_g,sD*4,cudaMemcpyDeviceToHost);
    sv("/tmp/bn_y_out.bin", hY, sx);
    sv("/tmp/bn_dx.bin", hdXg, sx);
    sv("/tmp/bn_dwq.bin", hdWqg, sD);
    sv("/tmp/bn_dwk.bin", hdWkg, sD);
    sv("/tmp/bn_dwv.bin", hdWvg, sD);
    sv("/tmp/bn_dwo.bin", hdWog, sD);

    printf("Block-noLN fwd+bwd done: T=%d D=%d H=%d\n", T, D, H);
    return 0;
}
