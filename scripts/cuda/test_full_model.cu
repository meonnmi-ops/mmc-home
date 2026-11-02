#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "block_ops.h"

#define MAX_T 512
static cublasHandle_t gb;

// ---------- attention kernels ----------
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
// emb forward: xs[t] = wte[ids[t]] + wpe[t]
__global__ void emb_fwd_k(const int* ids, const float* wte, const float* wpe,
                          float* xs, int T, int D){
    int t = blockIdx.x, d = threadIdx.x;
    if (d >= D) return;
    xs[t*D+d] = wte[ids[t]*D+d] + wpe[t*D+d];
}
// emb backward: scatter dWte[ids[t]] += dX[t] ; dWpe[t] = dX[t]
__global__ void emb_bwd_k(const int* ids, const float* dX,
                          float* dWte, float* dWpe, int T, int D){
    int t = blockIdx.x, d = threadIdx.x;
    if (d >= D) return;
    atomicAdd(&dWte[ids[t]*D+d], dX[t*D+d]);
    dWpe[t*D+d] = dX[t*D+d];
}
// CE softmax: probs + loss + dlogits
__global__ void softmax_ce_fwd(const float* logits, const int* tg, float* probs,
                               float* loss, int T, int V){
    int row = blockIdx.x, tid = threadIdx.x;
    extern __shared__ float s[];
    float v = (tid < V) ? logits[row*V+tid] : -1e30f;
    s[tid] = v; __syncthreads();
    for (int st=blockDim.x>>1; st>0; st>>=1){ if (tid<st) s[tid]=fmaxf(s[tid],s[tid+st]); __syncthreads(); }
    float mx = s[0]; __syncthreads();
    float e = (tid < V) ? expf(v-mx) : 0.0f;
    s[tid] = e; __syncthreads();
    for (int st=blockDim.x>>1; st>0; st>>=1){ if (tid<st) s[tid]+=s[tid+st]; __syncthreads(); }
    float sum = s[0];
    if (tid < V) probs[row*V+tid] = e / sum;
    if (tid == 0) loss[row] = -((logits[row*V + tg[row]] - mx) - logf(sum));
}
__global__ void softmax_ce_bwd(const float* probs, const int* tg, float* dlogits,
                               int T, int V, float invT){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= T*V) return;
    int r = i/V, c = i%V;
    float p = probs[i];
    float oh = (c == tg[r]) ? 1.0f : 0.0f;
    dlogits[i] = (p - oh) * invT;
}
// tied head fwd: logits = xnf @ wte.T
__global__ void tied_head_fwd(const float* xnf, const float* wte, float* logits,
                              int T, int V, int D){
    int t = blockIdx.x, v = threadIdx.x;
    if (v >= V) return;
    float s = 0.0f;
    for (int d = 0; d < D; d++) s += xnf[t*D+d] * wte[v*D+d];
    logits[t*V+v] = s;
}
// tied head bwd: dxnf[t] = dlogits[t] @ wte ; dWte[v] += sum_t dlogits[t,v] * xnf[t]
__global__ void tied_head_bwd_dxnf(const float* dlogits, const float* wte,
                                    float* dxnf, int T, int V, int D){
    int t = blockIdx.x, d = threadIdx.x;
    if (d >= D) return;
    float s = 0.0f;
    for (int v = 0; v < V; v++) s += dlogits[t*V+v] * wte[v*D+d];
    dxnf[t*D+d] = s;
}
__global__ void tied_head_bwd_dwte(const float* dlogits, const float* xnf,
                                    float* dWte, int T, int V, int D){
    int v = blockIdx.x, d = threadIdx.x;
    if (d >= D) return;
    float s = 0.0f;
    for (int t = 0; t < T; t++) s += dlogits[t*V+v] * xnf[t*D+d];
    atomicAdd(&dWte[v*D+d], s);
}

// ---------- cublas helpers ----------
static void mm(const float* A,const float* B,float* C,int M,int K,int N){
    float a=1,b=0; cublasSgemm(gb,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&a,B,N,A,K,&b,C,N);
}
static void mm_nt(const float* A,const float* B,float* C,int M,int K,int N){
    float a=1,b=0; cublasSgemm(gb,CUBLAS_OP_T,CUBLAS_OP_N,N,M,K,&a,B,K,A,K,&b,C,N);
}
static void mm_tn(const float* A,const float* B,float* C,int M,int K,int N){
    float a=1,b=0; cublasSgemm(gb,CUBLAS_OP_N,CUBLAS_OP_T,N,M,K,&a,B,N,A,M,&b,C,N);
}
static void ld(const char* p,void* b,int n){ FILE* f=fopen(p,"rb"); if(!f){perror(p);exit(1);} fread(b,4,n,f); fclose(f); }
static void sv(const char* p,void* b,int n){ FILE* f=fopen(p,"wb"); if(!f){perror(p);exit(1);} fwrite(b,4,n,f); fclose(f); }
#define ALLOC(ptr, sz) cudaMalloc(&ptr, (sz)*4)

int main(){
    cublasCreate(&gb);
    const int T=4, D=32, DFF=64, H=2, L=3, V=256, TMAX=8;
    const float eps=1e-5f;
    const float sc = 1.0f/sqrtf((float)(D/H));
    size_t sx=T*D, sh=T*DFF, sD=D*D, sDFF=D*DFF;

    // Load host weights
    int* hids=(int*)malloc(T*4); int* htg=(int*)malloc(T*4);
    ld("/tmp/fm_ids.bin", hids, T); ld("/tmp/fm_tg.bin", htg, T);
    float *hwte=(float*)malloc(V*D*4), *hwpe=(float*)malloc(TMAX*D*4);
    float *hlnfg=(float*)malloc(D*4), *hlnfb=(float*)malloc(D*4);
    ld("/tmp/fm_wte.bin", hwte, V*D);
    ld("/tmp/fm_wpe.bin", hwpe, TMAX*D);
    ld("/tmp/fm_lnfg.bin", hlnfg, D); ld("/tmp/fm_lnfb.bin", hlnfb, D);
    float *hdY=(float*)malloc(sx*4); ld("/tmp/fm_dy.bin", hdY, sx);

    float *hWq=(float*)malloc(L*sD*4), *hWk=(float*)malloc(L*sD*4),
          *hWv=(float*)malloc(L*sD*4), *hWo=(float*)malloc(L*sD*4);
    float *hln1g=(float*)malloc(L*D*4), *hln1b=(float*)malloc(L*D*4);
    float *hW1=(float*)malloc(L*sDFF*4), *hb1=(float*)malloc(L*DFF*4);
    float *hW2=(float*)malloc(L*sDFF*4), *hb2=(float*)malloc(L*D*4);
    float *hln2g=(float*)malloc(L*D*4), *hln2b=(float*)malloc(L*D*4);
    ld("/tmp/fm_wq.bin", hWq, L*sD); ld("/tmp/fm_wk.bin", hWk, L*sD);
    ld("/tmp/fm_wv.bin", hWv, L*sD); ld("/tmp/fm_wo.bin", hWo, L*sD);
    ld("/tmp/fm_ln1g.bin", hln1g, L*D); ld("/tmp/fm_ln1b.bin", hln1b, L*D);
    ld("/tmp/fm_w1.bin", hW1, L*sDFF); ld("/tmp/fm_b1.bin", hb1, L*DFF);
    ld("/tmp/fm_w2.bin", hW2, L*sDFF); ld("/tmp/fm_b2.bin", hb2, L*D);
    ld("/tmp/fm_ln2g.bin", hln2g, L*D); ld("/tmp/fm_ln2b.bin", hln2b, L*D);

    // Device allocs
    int* dids; int* dtg;
    cudaMalloc(&dids, T*4); cudaMalloc(&dtg, T*4);
    cudaMemcpy(dids, hids, T*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dtg, htg, T*4, cudaMemcpyHostToDevice);

    float *dwte, *dwpe, *dlnfg, *dlnfb;
    ALLOC(dwte, V*D); ALLOC(dwpe, TMAX*D); ALLOC(dlnfg, D); ALLOC(dlnfb, D);
    cudaMemcpy(dwte, hwte, V*D*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dwpe, hwpe, TMAX*D*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dlnfg, hlnfg, D*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dlnfb, hlnfb, D*4, cudaMemcpyHostToDevice);

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

    // --- Forward activations ---
    float *dXs; ALLOC(dXs, (L+1)*sx);
    float *dh1,*dQ,*dK,*dV,*dA,*dO,*dX1_fwd,*dh2,*dpre;
    ALLOC(dh1,L*sx); ALLOC(dQ,L*sx); ALLOC(dK,L*sx); ALLOC(dV,L*sx);
    ALLOC(dA,L*sx); ALLOC(dO,L*sx); ALLOC(dX1_fwd,L*sx);
    ALLOC(dh2,L*sx); ALLOC(dpre,L*sh);
    float *dpre_save; ALLOC(dpre_save,L*sh);
    float *dm1,*dr1,*dm2,*dr2;
    ALLOC(dm1,L*T); ALLOC(dr1,L*T); ALLOC(dm2,L*T); ALLOC(dr2,L*T);
    float *dxnf; ALLOC(dxnf, sx);
    float *dlogits; ALLOC(dlogits, T*V);
    float *dprobs; ALLOC(dprobs, T*V);
    float *dloss; ALLOC(dloss, T);

    dim3 grd(H,T), blk(1);

    // 1) emb: xs[0] = wte[ids] + wpe
    emb_fwd_k<<<T, D>>>(dids, dwte, dwpe, dXs, T, D);
    cudaDeviceSynchronize();

    // 2) L blocks
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
        float *bpre_save = dpre_save + b*sh;
        float *bm1 = dm1 + b*T, *br1 = dr1 + b*T;
        float *bm2 = dm2 + b*T, *br2 = dr2 + b*T;

        ln_fwd_dev(xin, bln1g, bln1b, bh1, bm1, br1, T, D, eps);
        mm(bh1, bWq, bQ, T, D, D); mm(bh1, bWk, bK, T, D, D); mm(bh1, bWv, bV, T, D, D);
        attn_fwd_k<<<grd,blk>>>(bQ, bK, bV, bA, T, D, H, sc);
        cudaDeviceSynchronize();
        mm(bA, bWo, bO, T, D, D);
        cudaMemcpy(bX1, xin, sx*4, cudaMemcpyDeviceToDevice);
        add_dev(bX1, bO, sx);
        ln_fwd_dev(bX1, bln2g, bln2b, bh2, bm2, br2, T, D, eps);
        mm(bh2, bW1, bpre, T, D, DFF);
        add_bias_dev(bpre, bb1, T, DFF);
        cudaMemcpy(bpre_save, bpre, sh*4, cudaMemcpyDeviceToDevice);
        gelu_fwd_dev(bpre_save, bpre, sh);
        mm(bpre, bW2, xout, T, DFF, D);
        add_bias_dev(xout, bb2, T, D);
        add_dev(xout, bX1, sx);
    }

    // 3) LN_f
    float *dmlf; ALLOC(dmlf, T); float *drlf; ALLOC(drlf, T);
    ln_fwd_dev(dXs + L*sx, dlnfg, dlnfb, dxnf, dmlf, drlf, T, D, eps);

    // 4) Tied head
    tied_head_fwd<<<T, V>>>(dxnf, dwte, dlogits, T, V, D);
    cudaDeviceSynchronize();

    // 5) CE
    int sm_b = 1; while (sm_b < V) sm_b <<= 1;
    int sm_sh = sm_b * sizeof(float);
    softmax_ce_fwd<<<T, sm_b, sm_sh>>>(dlogits, dtg, dprobs, dloss, T, V);
    cudaDeviceSynchronize();

    // save Y, probs, loss
    { float* h=(float*)malloc(sx*4); cudaMemcpy(h, dXs+L*sx, sx*4, cudaMemcpyDeviceToHost); sv("/tmp/fm_y.bin",h,sx); free(h); }
    { float* h=(float*)malloc(T*V*4); cudaMemcpy(h, dprobs, T*V*4, cudaMemcpyDeviceToHost); sv("/tmp/fm_probs.bin",h,T*V); free(h); }
    { float* h=(float*)malloc(T*4); cudaMemcpy(h, dloss, T*4, cudaMemcpyDeviceToHost); sv("/tmp/fm_loss.bin",h,T); free(h); }

    // ============ BACKWARD ============
    // 5') dlogits from CE
    float invT = 1.0f/T;
    int dlg_b = (T*V + 255) / 256;
    softmax_ce_bwd<<<dlg_b, 256>>>(dprobs, dtg, dlogits, T, V, invT);
    cudaDeviceSynchronize();

    // 4') Tied head bwd: dxnf = dlogits @ wte, dWte_head = dlogits^T @ xnf (scatter to dwte)
    float *ddwte_head; ALLOC(ddwte_head, V*D);
    cudaMemset(ddwte_head, 0, V*D*4);
    tied_head_bwd_dxnf<<<T, D>>>(dlogits, dwte, dxnf, T, V, D);    // NOTE: use saved xnf? We overwrote dxnf. Need xnf saved.
    cudaDeviceSynchronize();
    // ^ BUG: xnf got overwritten by dxnf. Need to save xnf forward first.
    // FIX: save xnf forward before tied head bwd
    // We'll fix in next iteration — for now save we'll recompute xnf later.
    // Actually — save xnf before tied_head_bwd. Let me handle by using separate buf.
    // (for now, skip; add buffer)
    float *dxnf_save; ALLOC(dxnf_save, sx);

    // (Skip — need forward xnf saved; restructure below)

    printf("Full model fwd+bwd (part 1): T=%d D=%d L=%d V=%d\n", T,D,L,V);
    return 0;
}
