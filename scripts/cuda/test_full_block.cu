#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "block_ops.h"

#define MAX_T 512
static cublasHandle_t gb;

__global__ void attn_fwd_k(const float* Q, const float* K, const float* V,
                           float* out, int T, int D, int H, float sc) {
    int h=blockIdx.x, row=blockIdx.y; int hd=D/H, off=h*hd; float s[MAX_T];
    for (int c=0;c<T;c++){ float su=0; for (int d=0;d<hd;d++) su+=Q[row*D+off+d]*K[c*D+off+d]; s[c]=su*sc; }
    float mx=-1e30f; for (int c=0;c<T;c++) if (s[c]>mx) mx=s[c];
    float se=0; for (int c=0;c<T;c++){ s[c]=expf(s[c]-mx); se+=s[c]; }
    for (int c=0;c<T;c++) s[c]/=se;
    for (int d=0;d<hd;d++){ float su=0; for (int c=0;c<T;c++) su+=s[c]*V[c*D+off+d]; out[row*D+off+d]=su; }
}
__global__ void attn_bwd_k(const float* Q, const float* K, const float* V,
                           const float* dO, float* dQ, float* dK, float* dV,
                           int T, int D, int H, float sc) {
    int h=blockIdx.x, row=blockIdx.y; int hd=D/H, off=h*hd;
    float s[MAX_T],sm[MAX_T],dp[MAX_T],ds[MAX_T];
    for (int c=0;c<T;c++){ float su=0; for (int d=0;d<hd;d++) su+=Q[row*D+off+d]*K[c*D+off+d]; s[c]=su*sc; }
    float mx=-1e30f; for (int c=0;c<T;c++) if (s[c]>mx) mx=s[c];
    float se=0; for (int c=0;c<T;c++){ sm[c]=expf(s[c]-mx); se+=sm[c]; }
    for (int c=0;c<T;c++) sm[c]/=se;
    for (int c=0;c<T;c++) for (int d=0;d<hd;d++) atomicAdd(&dV[c*D+off+d], sm[c]*dO[row*D+off+d]);
    for (int c=0;c<T;c++){ float su=0; for (int d=0;d<hd;d++) su+=dO[row*D+off+d]*V[c*D+off+d]; dp[c]=su; }
    float dot=0; for (int c=0;c<T;c++) dot+=dp[c]*sm[c];
    for (int c=0;c<T;c++) ds[c]=sm[c]*(dp[c]-dot)*sc;
    for (int d=0;d<hd;d++){ float su=0; for (int c=0;c<T;c++) su+=ds[c]*K[c*D+off+d]; dQ[row*D+off+d]=su; }
    for (int c=0;c<T;c++) for (int d=0;d<hd;d++) atomicAdd(&dK[c*D+off+d], ds[c]*Q[row*D+off+d]);
}
// Row sum for bias grads
__global__ void colsum_k(const float* x, float* out, int T, int D) {
    int j = blockIdx.x*blockDim.x+threadIdx.x; if (j>=D) return;
    float s = 0; for (int t=0;t<T;t++) s += x[t*D+j]; out[j] = s;
}

// cublas row-major wrappers
static void mm(const float* A,const float* B,float* C,int M,int K,int N){
    float a=1,b=0; cublasSgemm(gb,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&a,B,N,A,K,&b,C,N);
}
static void mm_nt(const float* A,const float* B,float* C,int M,int K,int N){
    float a=1,b=0; cublasSgemm(gb,CUBLAS_OP_T,CUBLAS_OP_N,N,M,K,&a,B,K,A,K,&b,C,N);
}
static void mm_tn(const float* A,const float* B,float* C,int M,int K,int N){
    float a=1,b=0; cublasSgemm(gb,CUBLAS_OP_N,CUBLAS_OP_T,N,M,K,&a,B,N,A,M,&b,C,N);
}

static void ld(const char* p, float* b, int n) {
    FILE* f=fopen(p,"rb"); if(!f){perror(p);exit(1);} fread(b,4,n,f); fclose(f);
}
static void sv(const char* p, float* b, int n) {
    FILE* f=fopen(p,"wb"); if(!f){perror(p);exit(1);} fwrite(b,4,n,f); fclose(f);
}

#define ALLOC(ptr, sz) cudaMalloc(&ptr, (sz)*4)

int main() {
    cublasCreate(&gb);
    const int T=4, D=32, DFF=64, H=2;
    const float eps = 1e-5f;
    const float sc = 1.0f/sqrtf((float)(D/H));
    size_t sx = T*D, sh = T*DFF, sD = D*D, sDFF = D*DFF;

    // Load inputs
    float *hX=(float*)malloc(sx*4), *hdY=(float*)malloc(sx*4);
    float *hWq=(float*)malloc(sD*4), *hWk=(float*)malloc(sD*4),
          *hWv=(float*)malloc(sD*4), *hWo=(float*)malloc(sD*4);
    float *hln1g=(float*)malloc(D*4), *hln1b=(float*)malloc(D*4);
    float *hW1=(float*)malloc(sDFF*4), *hb1=(float*)malloc(DFF*4);
    float *hW2=(float*)malloc(sDFF*4), *hb2=(float*)malloc(D*4);
    float *hln2g=(float*)malloc(D*4), *hln2b=(float*)malloc(D*4);
    ld("/tmp/fb_x.bin", hX, sx); ld("/tmp/fb_dy.bin", hdY, sx);
    ld("/tmp/fb_wq.bin", hWq, sD); ld("/tmp/fb_wk.bin", hWk, sD);
    ld("/tmp/fb_wv.bin", hWv, sD); ld("/tmp/fb_wo.bin", hWo, sD);
    ld("/tmp/fb_ln1g.bin", hln1g, D); ld("/tmp/fb_ln1b.bin", hln1b, D);
    ld("/tmp/fb_w1.bin", hW1, sDFF); ld("/tmp/fb_b1.bin", hb1, DFF);
    ld("/tmp/fb_w2.bin", hW2, sDFF); ld("/tmp/fb_b2.bin", hb2, D);
    ld("/tmp/fb_ln2g.bin", hln2g, D); ld("/tmp/fb_ln2b.bin", hln2b, D);

    // === Device alloc ===
    float *dX, *dWq, *dWk, *dWv, *dWo, *dln1g, *dln1b;
    float *dW1, *db1, *dW2, *db2, *dln2g, *dln2b, *dYup;
    ALLOC(dX, sx); ALLOC(dWq,sD); ALLOC(dWk,sD); ALLOC(dWv,sD); ALLOC(dWo,sD);
    ALLOC(dln1g,D); ALLOC(dln1b,D);
    ALLOC(dW1,sDFF); ALLOC(db1,DFF); ALLOC(dW2,sDFF); ALLOC(db2,D);
    ALLOC(dln2g,D); ALLOC(dln2b,D); ALLOC(dYup, sx);

    // Activations
    float *dh1, *dm1, *dr1, *dQ, *dK, *dV, *dA, *dO, *dX1;
    float *dh2, *dm2, *dr2, *dhg, *dM, *dY;
    ALLOC(dh1, sx); ALLOC(dm1, T); ALLOC(dr1, T);
    ALLOC(dQ, sx); ALLOC(dK, sx); ALLOC(dV, sx); ALLOC(dA, sx); ALLOC(dO, sx);
    ALLOC(dX1, sx);
    ALLOC(dh2, sx); ALLOC(dm2, T); ALLOC(dr2, T);
    ALLOC(dhg, sh); ALLOC(dM, sx); ALLOC(dY, sx);

    // Grad buffers
    float *dWq_g, *dWk_g, *dWv_g, *dWo_g, *dln1g_g, *dln1b_g;
    float *dW1_g, *db1_g, *dW2_g, *db2_g, *dln2g_g, *dln2b_g, *dX_g;
    ALLOC(dWq_g,sD); ALLOC(dWk_g,sD); ALLOC(dWv_g,sD); ALLOC(dWo_g,sD);
    ALLOC(dln1g_g,D); ALLOC(dln1b_g,D);
    ALLOC(dW1_g,sDFF); ALLOC(db1_g,DFF); ALLOC(dW2_g,sDFF); ALLOC(db2_g,D);
    ALLOC(dln2g_g,D); ALLOC(dln2b_g,D); ALLOC(dX_g, sx);

    // Temp
    float *dT1, *dT2, *dT3;
    ALLOC(dT1, sh); ALLOC(dT2, sx); ALLOC(dT3, sx);
    float *dX1_fwd, *dT_sh, *dQ_g, *dK_g, *dV_g;
    ALLOC(dX1_fwd, sx); ALLOC(dT_sh, sh);
    ALLOC(dQ_g, sx); ALLOC(dK_g, sx); ALLOC(dV_g, sx);

    // Upload
    cudaMemcpy(dX, hX, sx*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dWq, hWq, sD*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dWk, hWk, sD*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dWv, hWv, sD*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dWo, hWo, sD*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dln1g, hln1g, D*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dln1b, hln1b, D*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dW1, hW1, sDFF*4, cudaMemcpyHostToDevice);
    cudaMemcpy(db1, hb1, DFF*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dW2, hW2, sDFF*4, cudaMemcpyHostToDevice);
    cudaMemcpy(db2, hb2, D*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dln2g, hln2g, D*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dln2b, hln2b, D*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dYup, hdY, sx*4, cudaMemcpyHostToDevice);

    // ========== FORWARD ==========
    dim3 grd(H,T), blk(1);

    // h1 = LN1(X)
    ln_fwd_dev(dX, dln1g, dln1b, dh1, dm1, dr1, T, D, eps);

    // Q=h1@Wq, K=h1@Wk, V=h1@Wv, A=Attn, O=A@Wo
    mm(dh1, dWq, dQ, T, D, D);
    mm(dh1, dWk, dK, T, D, D);
    mm(dh1, dWv, dV, T, D, D);
    attn_fwd_k<<<grd,blk>>>(dQ, dK, dV, dA, T, D, H, sc);
    cudaDeviceSynchronize();
    mm(dA, dWo, dO, T, D, D);

    // X1 = X + O
    cudaMemcpy(dX1, dX, sx*4, cudaMemcpyDeviceToDevice);
    add_dev(dX1, dO, sx);
    cudaMemcpy(dX1_fwd, dX1, sx*4, cudaMemcpyDeviceToDevice);  // save forward X1

    // h2 = LN2(X1)
    ln_fwd_dev(dX1, dln2g, dln2b, dh2, dm2, dr2, T, D, eps);

    // hg = GELU(h2@W1+b1), M = hg@W2+b2
    mm(dh2, dW1, dhg, T, D, DFF);
    add_bias_dev(dhg, db1, T, DFF);
    // save pre-gelu to dhg (need for gelu bwd)
    // We'll re-use dT1 to store pre-gelu
    cudaMemcpy(dT1, dhg, sh*4, cudaMemcpyDeviceToDevice);
    gelu_fwd_dev(dT1, dhg, sh);
    mm(dhg, dW2, dM, T, DFF, D);
    add_bias_dev(dM, db2, T, D);

    // Y = X1 + M
    cudaMemcpy(dY, dX1, sx*4, cudaMemcpyDeviceToDevice);
    add_dev(dY, dM, sx);

    // Save Y
    { float* hY=(float*)malloc(sx*4); cudaMemcpy(hY,dY,sx*4,cudaMemcpyDeviceToHost); sv("/tmp/fb_y.bin",hY,sx); free(hY); }

    // ========== BACKWARD ==========
    // dY upstream = dYup
    // dX1 = dYup (residual 2)
    cudaMemcpy(dX1, dYup, sx*4, cudaMemcpyDeviceToDevice);
    // dW2 = hg^T @ dYup, db2 = colsum(dYup)
    mm_tn(dhg, dYup, dW2_g, DFF, T, D);
    colsum_k<<<(D+255)/256,256>>>(dYup, db2_g, T, D);
    // dhg = dYup @ W2^T
    mm_nt(dYup, dW2, dT_sh, T, D, DFF);      // dT_sh = dYup @ W2^T [T,DFF]
    gelu_bwd_dev(dT_sh, dT1, dT_sh, sh);     // in-place: dT_sh = dM_in
    // dW1 = h2^T @ dM_in, db1 = colsum(dM_in)
    mm_tn(dh2, dT_sh, dW1_g, D, T, DFF);
    colsum_k<<<(DFF+255)/256,256>>>(dT_sh, db1_g, T, DFF);
    // dX1 += dM_in @ W1^T (path through MLP)
    mm_nt(dT_sh, dW1, dT3, T, DFF, D);      // dT3 = dM_in @ W1^T = dh2 [T,D]
    add_dev(dX1, dT3, sx);
    // LN2_bwd: dh2 → dX1 (path through LN2)
    cudaMemset(dln2g_g, 0, D*4);
    cudaMemset(dln2b_g, 0, D*4);
    ln_bwd_dev(dT3, dX1_fwd, dln2g, dm2, dr2, dT2, dln2g_g, dln2b_g, T, D);
    add_dev(dX1, dT2, sx);

    // dO = dX1
    cudaMemcpy(dO, dX1, sx*4, cudaMemcpyDeviceToDevice);
    // dWo = A^T @ dO, dA = dO @ Wo^T
    mm_tn(dA, dO, dWo_g, D, T, D);
    mm_nt(dO, dWo, dA, T, D, D);
    // Attention bwd (clean)
    cudaMemset(dQ_g, 0, sx*4); cudaMemset(dK_g, 0, sx*4); cudaMemset(dV_g, 0, sx*4);
    attn_bwd_k<<<grd,blk>>>(dQ,dK,dV,dA,dQ_g,dK_g,dV_g, T,D,H,sc);
    cudaDeviceSynchronize();
    // dWq = h1^T @ dQ, dWk = h1^T @ dK, dWv = h1^T @ dV
    mm_tn(dh1, dQ_g, dWq_g, D, T, D);
    mm_tn(dh1, dK_g, dWk_g, D, T, D);
    mm_tn(dh1, dV_g, dWv_g, D, T, D);
    // dh1 = dQ@Wq^T + dK@Wk^T + dV@Wv^T
    mm_nt(dQ_g, dWq, dT1, T, D, D);
    mm_nt(dK_g, dWk, dT2, T, D, D);
    add_dev(dT1, dT2, sx);
    mm_nt(dV_g, dWv, dT2, T, D, D);
    add_dev(dT1, dT2, sx);
    // dX += dX1 (residual 1)
    cudaMemcpy(dX_g, dX1, sx*4, cudaMemcpyDeviceToDevice);
    // dX += LN1_bwd(dh1)
    cudaMemset(dln1g_g, 0, D*4);
    cudaMemset(dln1b_g, 0, D*4);
    ln_bwd_dev(dT1, dX, dln1g, dm1, dr1, dT2, dln1g_g, dln1b_g, T, D);
    add_dev(dX_g, dT2, sx);

    // ========== SAVE GRADS ==========
    float *hY=(float*)malloc(sx*4), *hdXg=(float*)malloc(sx*4);
    float *hdWqg=(float*)malloc(sD*4), *hdWkg=(float*)malloc(sD*4),
          *hdWvg=(float*)malloc(sD*4), *hdWog=(float*)malloc(sD*4);
    float *hdln1g=(float*)malloc(D*4), *hdln1b=(float*)malloc(D*4);
    float *hdW1g=(float*)malloc(sDFF*4), *hdb1g=(float*)malloc(DFF*4);
    float *hdW2g=(float*)malloc(sDFF*4), *hdb2g=(float*)malloc(D*4);
    float *hdln2g=(float*)malloc(D*4), *hdln2b=(float*)malloc(D*4);

    cudaMemcpy(hdXg, dX_g, sx*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hdWqg, dWq_g, sD*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hdWkg, dWk_g, sD*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hdWvg, dWv_g, sD*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hdWog, dWo_g, sD*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hdln1g, dln1g_g, D*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hdln1b, dln1b_g, D*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hdW1g, dW1_g, sDFF*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hdb1g, db1_g, DFF*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hdW2g, dW2_g, sDFF*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hdb2g, db2_g, D*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hdln2g, dln2g_g, D*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hdln2b, dln2b_g, D*4, cudaMemcpyDeviceToHost);

    sv("/tmp/fb_dx.bin", hdXg, sx);
    sv("/tmp/fb_dwq.bin", hdWqg, sD); sv("/tmp/fb_dwk.bin", hdWkg, sD);
    sv("/tmp/fb_dwv.bin", hdWvg, sD); sv("/tmp/fb_dwo.bin", hdWog, sD);
    sv("/tmp/fb_dln1g.bin", hdln1g, D); sv("/tmp/fb_dln1b.bin", hdln1b, D);
    sv("/tmp/fb_dw1.bin", hdW1g, sDFF); sv("/tmp/fb_db1.bin", hdb1g, DFF);
    sv("/tmp/fb_dw2.bin", hdW2g, sDFF); sv("/tmp/fb_db2.bin", hdb2g, D);
    sv("/tmp/fb_dln2g.bin", hdln2g, D); sv("/tmp/fb_dln2b.bin", hdln2b, D);

    printf("Full block fwd+bwd: T=%d D=%d DFF=%d H=%d\n", T, D, DFF, H);
    return 0;
}
