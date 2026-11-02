#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "block_ops.h"

#define MAX_T 512
static cublasHandle_t gb;

__global__ void attn_fwd_k(const float* Q,const float* K,const float* V,float* out,
                           int T,int D,int H,float sc){
    int h=blockIdx.x,row=blockIdx.y; int hd=D/H,off=h*hd; float s[MAX_T];
    for (int c=0;c<T;c++){ float su=0; for (int d=0;d<hd;d++) su+=Q[row*D+off+d]*K[c*D+off+d]; s[c]=su*sc; }
    float mx=-1e30f; for (int c=0;c<T;c++) if (s[c]>mx) mx=s[c];
    float se=0; for (int c=0;c<T;c++){ s[c]=expf(s[c]-mx); se+=s[c]; }
    for (int c=0;c<T;c++) s[c]/=se;
    for (int d=0;d<hd;d++){ float su=0; for (int c=0;c<T;c++) su+=s[c]*V[c*D+off+d]; out[row*D+off+d]=su; }
}
__global__ void attn_bwd_k(const float* Q,const float* K,const float* V,const float* dO,
                           float* dQ,float* dK,float* dV,int T,int D,int H,float sc){
    int h=blockIdx.x,row=blockIdx.y; int hd=D/H,off=h*hd;
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
__global__ void colsum_k(const float* x, float* out, int T, int D){
    int j=blockIdx.x*blockDim.x+threadIdx.x; if (j>=D) return;
    float s=0; for (int t=0;t<T;t++) s+=x[t*D+j]; out[j]=s;
}

static void mm(const float* A,const float* B,float* C,int M,int K,int N){
    float a=1,b=0; cublasSgemm(gb,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&a,B,N,A,K,&b,C,N);
}
static void mm_nt(const float* A,const float* B,float* C,int M,int K,int N){
    float a=1,b=0; cublasSgemm(gb,CUBLAS_OP_T,CUBLAS_OP_N,N,M,K,&a,B,K,A,K,&b,C,N);
}
static void mm_tn(const float* A,const float* B,float* C,int M,int K,int N){
    float a=1,b=0; cublasSgemm(gb,CUBLAS_OP_N,CUBLAS_OP_T,N,M,K,&a,B,N,A,M,&b,C,N);
}
static void ld(const char* p,float* b,int n){ FILE* f=fopen(p,"rb"); if(!f){perror(p);exit(1);} fread(b,4,n,f); fclose(f); }
static void sv(const char* p,float* b,int n){ FILE* f=fopen(p,"wb"); if(!f){perror(p);exit(1);} fwrite(b,4,n,f); fclose(f); }
#define ALLOC(ptr, sz) cudaMalloc(&ptr, (sz)*4)

int main(){
    cublasCreate(&gb);
    const int T=4, D=32, DFF=64, H=2, L=3;
    const float eps=1e-5f;
    const float sc = 1.0f/sqrtf((float)(D/H));
    size_t sx=T*D, sh=T*DFF, sD=D*D, sDFF=D*DFF;

    // Host weights: L blocks × (Wq,Wk,Wv,Wo,ln1g,ln1b,W1,b1,W2,b2,ln2g,ln2b)
    float *hx=(float*)malloc(sx*4), *hdy=(float*)malloc(sx*4);
    float *hWq=(float*)malloc(L*sD*4), *hWk=(float*)malloc(L*sD*4),
          *hWv=(float*)malloc(L*sD*4), *hWo=(float*)malloc(L*sD*4);
    float *hln1g=(float*)malloc(L*D*4), *hln1b=(float*)malloc(L*D*4);
    float *hW1=(float*)malloc(L*sDFF*4), *hb1=(float*)malloc(L*DFF*4);
    float *hW2=(float*)malloc(L*sDFF*4), *hb2=(float*)malloc(L*D*4);
    float *hln2g=(float*)malloc(L*D*4), *hln2b=(float*)malloc(L*D*4);
    ld("/tmp/lb_x.bin", hx, sx); ld("/tmp/lb_dy.bin", hdy, sx);
    ld("/tmp/lb_wq.bin", hWq, L*sD); ld("/tmp/lb_wk.bin", hWk, L*sD);
    ld("/tmp/lb_wv.bin", hWv, L*sD); ld("/tmp/lb_wo.bin", hWo, L*sD);
    ld("/tmp/lb_ln1g.bin", hln1g, L*D); ld("/tmp/lb_ln1b.bin", hln1b, L*D);
    ld("/tmp/lb_w1.bin", hW1, L*sDFF); ld("/tmp/lb_b1.bin", hb1, L*DFF);
    ld("/tmp/lb_w2.bin", hW2, L*sDFF); ld("/tmp/lb_b2.bin", hb2, L*D);
    ld("/tmp/lb_ln2g.bin", hln2g, L*D); ld("/tmp/lb_ln2b.bin", hln2b, L*D);

    // Upload weights
    float *dWq,*dWk,*dWv,*dWo,*dln1g,*dln1b,*dW1,*db1,*dW2,*db2,*dln2g,*dln2b;
    ALLOC(dWq,L*sD); ALLOC(dWk,L*sD); ALLOC(dWv,L*sD); ALLOC(dWo,L*sD);
    ALLOC(dln1g,L*D); ALLOC(dln1b,L*D);
    ALLOC(dW1,L*sDFF); ALLOC(db1,L*DFF); ALLOC(dW2,L*sDFF); ALLOC(db2,L*D);
    ALLOC(dln2g,L*D); ALLOC(dln2b,L*D);
    cudaMemcpy(dWq,hWq,L*sD*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dWk,hWk,L*sD*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dWv,hWv,L*sD*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dWo,hWo,L*sD*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dln1g,hln1g,L*D*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dln1b,hln1b,L*D*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dW1,hW1,L*sDFF*4,cudaMemcpyHostToDevice);
    cudaMemcpy(db1,hb1,L*DFF*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dW2,hW2,L*sDFF*4,cudaMemcpyHostToDevice);
    cudaMemcpy(db2,hb2,L*D*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dln2g,hln2g,L*D*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dln2b,hln2b,L*D*4,cudaMemcpyHostToDevice);

    // Grad buffers
    float *dWq_g,*dWk_g,*dWv_g,*dWo_g,*dln1g_g,*dln1b_g;
    float *dW1_g,*db1_g,*dW2_g,*db2_g,*dln2g_g,*dln2b_g;
    ALLOC(dWq_g,L*sD); ALLOC(dWk_g,L*sD); ALLOC(dWv_g,L*sD); ALLOC(dWo_g,L*sD);
    ALLOC(dln1g_g,L*D); ALLOC(dln1b_g,L*D);
    ALLOC(dW1_g,L*sDFF); ALLOC(db1_g,L*DFF); ALLOC(dW2_g,L*sDFF); ALLOC(db2_g,L*D);
    ALLOC(dln2g_g,L*D); ALLOC(dln2b_g,L*D);
    cudaMemset(dWq_g,0,L*sD*4); cudaMemset(dWk_g,0,L*sD*4);
    cudaMemset(dWv_g,0,L*sD*4); cudaMemset(dWo_g,0,L*sD*4);
    cudaMemset(dln1g_g,0,L*D*4); cudaMemset(dln1b_g,0,L*D*4);
    cudaMemset(dW1_g,0,L*sDFF*4); cudaMemset(db1_g,0,L*DFF*4);
    cudaMemset(dW2_g,0,L*sDFF*4); cudaMemset(db2_g,0,L*D*4);
    cudaMemset(dln2g_g,0,L*D*4); cudaMemset(dln2b_g,0,L*D*4);

    // Inputs/outputs
    float *dX, *dYup;
    ALLOC(dX, sx); ALLOC(dYup, sx);
    cudaMemcpy(dX, hx, sx*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dYup, hdy, sx*4, cudaMemcpyHostToDevice);

    // Per-block saved activations
    float *dh1, *dQ, *dK, *dV, *dA, *dO, *dX1_fwd, *dh2, *dpre, *dT1;
    float *dm1, *dr1, *dm2, *dr2;
    ALLOC(dh1, L*sx); ALLOC(dQ, L*sx); ALLOC(dK, L*sx); ALLOC(dV, L*sx);
    ALLOC(dA, L*sx); ALLOC(dO, L*sx); ALLOC(dX1_fwd, L*sx);
    ALLOC(dh2, L*sx); ALLOC(dpre, L*sh);
    ALLOC(dT1, sh);
    ALLOC(dm1, L*T); ALLOC(dr1, L*T); ALLOC(dm2, L*T); ALLOC(dr2, L*T);

    // xs chain
    float *dXs; ALLOC(dXs, (L+1)*sx);
    cudaMemcpy(dXs, dX, sx*4, cudaMemcpyDeviceToDevice);  // xs[0] = X

    dim3 grd(H,T), blk(1);

    // ============ FORWARD L BLOCKS ============
    for (int b = 0; b < L; b++) {
        float *xin = dXs + b*sx;
        float *xout = dXs + (b+1)*sx;
        float *bWq = dWq + b*sD, *bWk = dWk + b*sD, *bWv = dWv + b*sD, *bWo = dWo + b*sD;
        float *bln1g = dln1g + b*D, *bln1b = dln1b + b*D;
        float *bW1 = dW1 + b*sDFF, *bb1 = db1 + b*DFF;
        float *bW2 = dW2 + b*sDFF, *bb2 = db2 + b*D;
        float *bln2g = dln2g + b*D, *bln2b = dln2b + b*D;

        float *bh1 = dh1 + b*sx;
        float *bQ = dQ + b*sx, *bK = dK + b*sx, *bV = dV + b*sx;
        float *bA = dA + b*sx, *bO = dO + b*sx;
        float *bX1 = dX1_fwd + b*sx;
        float *bh2 = dh2 + b*sx;
        float *bpre = dpre + b*sh;
        float *bm1 = dm1 + b*T, *br1 = dr1 + b*T;
        float *bm2 = dm2 + b*T, *br2 = dr2 + b*T;

        ln_fwd_dev(xin, bln1g, bln1b, bh1, bm1, br1, T, D, eps);
        mm(bh1, bWq, bQ, T, D, D);
        mm(bh1, bWk, bK, T, D, D);
        mm(bh1, bWv, bV, T, D, D);
        attn_fwd_k<<<grd,blk>>>(bQ, bK, bV, bA, T, D, H, sc);
        cudaDeviceSynchronize();
        mm(bA, bWo, bO, T, D, D);
        cudaMemcpy(bX1, xin, sx*4, cudaMemcpyDeviceToDevice);
        add_dev(bX1, bO, sx);
        ln_fwd_dev(bX1, bln2g, bln2b, bh2, bm2, br2, T, D, eps);
        mm(bh2, bW1, bpre, T, D, DFF);
        add_bias_dev(bpre, bb1, T, DFF);
        // save pre-gelu, then gelu
        float *bpre_save = dT1;  // temp
        cudaMemcpy(bpre_save, bpre, sh*4, cudaMemcpyDeviceToDevice);
        gelu_fwd_dev(bpre_save, bpre, sh);
        mm(bpre, bW2, xout, T, DFF, D);
        add_bias_dev(xout, bb2, T, D);
        // Y = X1 + M  (store in xout)
        add_dev(xout, bX1, sx);
    }

    // Save Y
    { float* hy=(float*)malloc(sx*4); cudaMemcpy(hy, dXs+L*sx, sx*4, cudaMemcpyDeviceToHost); sv("/tmp/lb_y.bin",hy,sx); free(hy); }

    // ============ BACKWARD (reverse) ============
    float *dXg;
    ALLOC(dXg, sx);
    cudaMemcpy(dXg, dYup, sx*4, cudaMemcpyDeviceToDevice);

    // temp buffers for block bwd
    float *dT_dxd = dX;              // reuse dX as temp (sx)
    float *dT_dxs = dT1;             // sh (careful!)
    // Actually use dYup as scratch? No — keep dYup. Allocate temp.
    float *dT3, *dT_sh, *dQ_g, *dK_g, *dV_g, *dX1_bwd, *dT_dx_safe;
    ALLOC(dT3, sh);
    ALLOC(dT_sh, sh);
    ALLOC(dQ_g, sx); ALLOC(dK_g, sx); ALLOC(dV_g, sx);
    ALLOC(dX1_bwd, sx);
    ALLOC(dT_dx_safe, sx);

    for (int b = L-1; b >= 0; b--) {
        float *xin = dXs + b*sx;
        float *bWq = dWq + b*sD, *bWk = dWk + b*sD, *bWv = dWv + b*sD, *bWo = dWo + b*sD;
        float *bW1 = dW1 + b*sDFF, *bW2 = dW2 + b*sDFF;
        float *bln1g = dln1g + b*D, *bln2g = dln2g + b*D;

        float *bh1 = dh1 + b*sx;
        float *bQ = dQ + b*sx, *bK = dK + b*sx, *bV = dV + b*sx;
        float *bA = dA + b*sx;
        float *bX1 = dX1_fwd + b*sx;
        float *bh2 = dh2 + b*sx;
        float *bpre = dpre + b*sh;
        float *bm1 = dm1 + b*T, *br1 = dr1 + b*T;
        float *bm2 = dm2 + b*T, *br2 = dr2 + b*T;

        // dY = dXg (grad wrt this block's output)
        // dX1 = dY (residual 2)
        cudaMemcpy(dX1_bwd, dXg, sx*4, cudaMemcpyDeviceToDevice);

        // dW2 = hg^T @ dY ; hg stored in bpre (post-gelu); need pre-gelu for gelu_bwd
        // Recompute pre-gelu from h2@W1+b1... OR: save pre-gelu forward
        // Use dT_sh to reconstruct pre-gelu
        mm(bh2, bW1, dT_sh, T, D, DFF);
        add_bias_dev(dT_sh, db1 + b*DFF, T, DFF);
        // bpre currently holds post-gelu (hg)
        // dW2 = hg^T @ dY
        mm_tn(bpre, dXg, dW2_g + b*sDFF, DFF, T, D);
        colsum_k<<<(D+255)/256,256>>>(dXg, db2_g + b*D, T, D);
        // dhg = dY @ W2^T → [T,DFF]
        mm_nt(dXg, bW2, dT3, T, D, DFF);
        // dM_in = gelu_bwd(dhg, pre_gelu) — dT_sh is pre_gelu
        gelu_bwd_dev(dT3, dT_sh, dT3, sh);  // dT3 = dM_in
        // dW1 = h2^T @ dM_in
        mm_tn(bh2, dT3, dW1_g + b*sDFF, D, T, DFF);
        colsum_k<<<(DFF+255)/256,256>>>(dT3, db1_g + b*DFF, T, DFF);
        // dX1 += dM_in @ W1^T
        mm_nt(dT3, bW1, dT_sh, T, DFF, D);   // dT_sh = dM_in @ W1^T [T,D]
        add_dev(dX1_bwd, dT_sh, sx);
        // LN2_bwd: dh2 = dM_in @ W1^T → dX1 += LN2_bwd(dh2)
        cudaMemset(dln2g_g + b*D, 0, D*4);
        cudaMemset(dln2b_g + b*D, 0, D*4);
        ln_bwd_dev(dT_sh, bX1, bln2g, bm2, br2, dT3, dln2g_g + b*D, dln2b_g + b*D, T, D);
        add_dev(dX1_bwd, dT3, sx);

        // dO = dX1
        // dWo = A^T @ dO, dA = dO @ Wo^T
        mm_tn(bA, dX1_bwd, dWo_g + b*sD, D, T, D);
        mm_nt(dX1_bwd, bWo, dT_sh, T, D, D);
        // Attention bwd
        cudaMemset(dQ_g, 0, sx*4); cudaMemset(dK_g, 0, sx*4); cudaMemset(dV_g, 0, sx*4);
        attn_bwd_k<<<grd,blk>>>(bQ, bK, bV, dT_sh, dQ_g, dK_g, dV_g, T, D, H, sc);
        cudaDeviceSynchronize();
        // dWq = h1^T @ dQ, dWk = h1^T @ dK, dWv = h1^T @ dV
        mm_tn(bh1, dQ_g, dWq_g + b*sD, D, T, D);
        mm_tn(bh1, dK_g, dWk_g + b*sD, D, T, D);
        mm_tn(bh1, dV_g, dWv_g + b*sD, D, T, D);
        // dh1 = dQ@Wq^T + dK@Wk^T + dV@Wv^T
        mm_nt(dQ_g, bWq, dT3, T, D, D);
        mm_nt(dK_g, bWk, dT_sh, T, D, D);
        add_dev(dT3, dT_sh, sx);
        mm_nt(dV_g, bWv, dT_sh, T, D, D);
        add_dev(dT3, dT_sh, sx);
        // dX += dX1 (residual 1)
        cudaMemcpy(dXg, dX1_bwd, sx*4, cudaMemcpyDeviceToDevice);
        // dX += LN1_bwd(dh1)
        cudaMemset(dln1g_g + b*D, 0, D*4);
        cudaMemset(dln1b_g + b*D, 0, D*4);
        ln_bwd_dev(dT3, xin, bln1g, bm1, br1, dT_sh, dln1g_g + b*D, dln1b_g + b*D, T, D);
        add_dev(dXg, dT_sh, sx);
        // dXg is now dX_in for next (previous) block
    }

    // Save grads
    float *hout=(float*)malloc(sx*4);
    cudaMemcpy(hout, dXg, sx*4, cudaMemcpyDeviceToHost);
    sv("/tmp/lb_dx.bin", hout, sx);
    free(hout);

    // Save weight grads for block 0 (test same weights on each block)
    float* hwg=(float*)malloc(L*sD*4);
    cudaMemcpy(hwg, dWq_g, L*sD*4, cudaMemcpyDeviceToHost);
    sv("/tmp/lb_dwq.bin", hwg, L*sD);
    cudaMemcpy(hwg, dWk_g, L*sD*4, cudaMemcpyDeviceToHost); sv("/tmp/lb_dwk.bin", hwg, L*sD);
    cudaMemcpy(hwg, dWv_g, L*sD*4, cudaMemcpyDeviceToHost); sv("/tmp/lb_dwv.bin", hwg, L*sD);
    cudaMemcpy(hwg, dWo_g, L*sD*4, cudaMemcpyDeviceToHost); sv("/tmp/lb_dwo.bin", hwg, L*sD);
    free(hwg);

    printf("L-block fwd+bwd: T=%d D=%d DFF=%d H=%d L=%d\n", T,D,DFF,H,L);
    return 0;
}
