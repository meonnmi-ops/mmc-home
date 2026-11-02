#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "block_ops.h"

#ifndef N_STEPS
#define N_STEPS 20
#endif

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
static void sv_loss(const char* path, const float* dloss, int T) {
    float* h = (float*)malloc(T*4);
    cudaMemcpy(h, dloss, T*4, cudaMemcpyDeviceToHost);
    float s = 0; for (int i = 0; i < T; i++) s += h[i]; s /= T;
    FILE* f = fopen(path, "wb"); if(f){ fwrite(&s,4,1,f); fclose(f); }
    free(h);
}
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
    // ===== BACKWARD (clean) =====
    float *dxnf_fwd; ALLOC(dxnf_fwd, sx);
    ln_fwd_dev(dXs + L*sx, dlnfg, dlnfb, dxnf_fwd, dmlf, drlf, T, D, eps);

    // Separate grad buffers (forward weights untouched)
    float *dWq_g,*dWk_g,*dWv_g,*dWo_g,*dln1g_g,*dln1b_g;
    float *dW1_g,*db1_g,*dW2_g,*db2_g,*dln2g_g,*dln2b_g;
    float *dwte_g,*dwpe_g,*dlnfg_g,*dlnfb_g;
    ALLOC(dWq_g,L*sD);ALLOC(dWk_g,L*sD);ALLOC(dWv_g,L*sD);ALLOC(dWo_g,L*sD);
    ALLOC(dln1g_g,L*D);ALLOC(dln1b_g,L*D);
    ALLOC(dW1_g,L*sDFF);ALLOC(db1_g,L*DFF);ALLOC(dW2_g,L*sDFF);ALLOC(db2_g,L*D);
    ALLOC(dln2g_g,L*D);ALLOC(dln2b_g,L*D);
    ALLOC(dwte_g,V*D);ALLOC(dwpe_g,TMAX*D);
    ALLOC(dlnfg_g,D);ALLOC(dlnfb_g,D);
    cudaMemset(dWq_g,0,L*sD*4);cudaMemset(dWk_g,0,L*sD*4);
    cudaMemset(dWv_g,0,L*sD*4);cudaMemset(dWo_g,0,L*sD*4);
    cudaMemset(dln1g_g,0,L*D*4);cudaMemset(dln1b_g,0,L*D*4);
    cudaMemset(dW1_g,0,L*sDFF*4);cudaMemset(db1_g,0,L*DFF*4);
    cudaMemset(dW2_g,0,L*sDFF*4);cudaMemset(db2_g,0,L*D*4);
    cudaMemset(dln2g_g,0,L*D*4);cudaMemset(dln2b_g,0,L*D*4);
    cudaMemset(dwte_g,0,V*D*4);cudaMemset(dwpe_g,0,TMAX*D*4);
    cudaMemset(dlnfg_g,0,D*4);cudaMemset(dlnfb_g,0,D*4);

    // Temp buffers
    float *dY_temp,*dX1_temp,*dhg_temp,*dpre_temp,*dh2_temp;
    float *dA_temp,*dQ_t,*dK_t,*dV_t,*dh1_temp,*dX_temp;
    float *dT_sx_a,*dT_sh_a;
    ALLOC(dY_temp,sx);ALLOC(dX1_temp,sx);ALLOC(dhg_temp,sh);
    ALLOC(dpre_temp,sh);ALLOC(dh2_temp,sx);
    ALLOC(dA_temp,sx);ALLOC(dQ_t,sx);ALLOC(dK_t,sx);ALLOC(dV_t,sx);
    ALLOC(dh1_temp,sx);ALLOC(dX_temp,sx);
    ALLOC(dT_sx_a,sx);ALLOC(dT_sh_a,sh);

    // 4') Tied head bwd
    tied_head_bwd_dxnf<<<T,D>>>(dlogits, dwte, dxnf, T,V,D);
    tied_head_bwd_dwte<<<V,D>>>(dlogits, dxnf_fwd, dwte_g, T,V,D);
    cudaDeviceSynchronize();

    // 3') LNf bwd
    cudaMemset(dlnfg_g,0,D*4);cudaMemset(dlnfb_g,0,D*4);
    ln_bwd_dev(dxnf, dXs+L*sx, dlnfg, dmlf, drlf, dY_temp, dlnfg_g, dlnfb_g, T,D);
    cudaDeviceSynchronize();

    // 2') Block chain
    for (int b=L-1; b>=0; b--) {
        float *bWq=dWq+b*sD,*bWk=dWk+b*sD,*bWv=dWv+b*sD,*bWo=dWo+b*sD;
        float *bW1=dW1+b*sDFF,*bW2=dW2+b*sDFF;
        float *bln1g=dln1g+b*D,*bln2g=dln2g+b*D;
        float *bh1=dh1+b*sx;
        float *bQ=dQ+b*sx,*bK=dK+b*sx,*bV=dV+b*sx;
        float *bA=dA+b*sx;
        float *bX1=dX1_fwd+b*sx;
        float *bh2=dh2+b*sx;
        float *bpre_save=dpre_save+b*sh;
        float *bpre=dpre+b*sh;
        float *bm1=dm1+b*T,*br1=dr1+b*T;
        float *bm2=dm2+b*T,*br2=dr2+b*T;

        cudaMemcpy(dX1_temp, dY_temp, sx*4, cudaMemcpyDeviceToDevice);
        mm_tn(bpre, dY_temp, dW2_g+b*sDFF, DFF, T, D);
        colsum_k<<<(D+255)/256,256>>>(dY_temp, db2_g+b*D, T, D);
        mm_nt(dY_temp, bW2, dhg_temp, T, D, DFF);
        gelu_bwd_dev(dhg_temp, bpre_save, dpre_temp, sh);
        mm_tn(bh2, dpre_temp, dW1_g+b*sDFF, D, T, DFF);
        colsum_k<<<(DFF+255)/256,256>>>(dpre_temp, db1_g+b*DFF, T, DFF);
        mm_nt(dpre_temp, bW1, dh2_temp, T, DFF, D);
        add_dev(dX1_temp, dh2_temp, sx);
        ln_bwd_dev(dh2_temp, bX1, bln2g, bm2, br2, dT_sx_a, dln2g_g+b*D, dln2b_g+b*D, T,D);
        add_dev(dX1_temp, dT_sx_a, sx);
        mm_tn(bA, dX1_temp, dWo_g+b*sD, D, T, D);
        mm_nt(dX1_temp, bWo, dA_temp, T, D, D);
        cudaMemset(dQ_t,0,sx*4);cudaMemset(dK_t,0,sx*4);cudaMemset(dV_t,0,sx*4);
        attn_bwd_k<<<grd,blk>>>(bQ, bK, bV, dA_temp, dQ_t, dK_t, dV_t, T, D, H, sc);
        cudaDeviceSynchronize();
        mm_tn(bh1, dQ_t, dWq_g+b*sD, D, T, D);
        mm_tn(bh1, dK_t, dWk_g+b*sD, D, T, D);
        mm_tn(bh1, dV_t, dWv_g+b*sD, D, T, D);
        mm_nt(dQ_t, bWq, dh1_temp, T, D, D);
        mm_nt(dK_t, bWk, dT_sx_a, T, D, D);
        add_dev(dh1_temp, dT_sx_a, sx);
        mm_nt(dV_t, bWv, dT_sx_a, T, D, D);
        add_dev(dh1_temp, dT_sx_a, sx);
        cudaMemcpy(dX_temp, dX1_temp, sx*4, cudaMemcpyDeviceToDevice);
        ln_bwd_dev(dh1_temp, dXs+b*sx, bln1g, bm1, br1, dT_sx_a, dln1g_g+b*D, dln1b_g+b*D, T,D);
        add_dev(dX_temp, dT_sx_a, sx);
        cudaMemcpy(dY_temp, dX_temp, sx*4, cudaMemcpyDeviceToDevice);
    }

    // 1') Emb bwd
    emb_bwd_k<<<T, D>>>(dids, dY_temp, dwte_g, dwpe_g, T, D);
    cudaDeviceSynchronize();

    // Save grads
    float *h=(float*)malloc(sx*4);
    cudaMemcpy(h, dY_temp, sx*4, cudaMemcpyDeviceToHost);
    sv("/tmp/fm_dx.bin", h, sx); free(h);

    h=(float*)malloc(V*D*4);
    cudaMemcpy(h, dwte_g, V*D*4, cudaMemcpyDeviceToHost);
    sv("/tmp/fm_dwte.bin", h, V*D); free(h);

    h=(float*)malloc(TMAX*D*4);
    cudaMemcpy(h, dwpe_g, TMAX*D*4, cudaMemcpyDeviceToHost);
    sv("/tmp/fm_dwpe.bin", h, TMAX*D); free(h);

    h=(float*)malloc(D*4);
    cudaMemcpy(h, dlnfg_g, D*4, cudaMemcpyDeviceToHost);
    sv("/tmp/fm_dlnfg.bin", h, D);
    cudaMemcpy(h, dlnfb_g, D*4, cudaMemcpyDeviceToHost);
    sv("/tmp/fm_dlnfb.bin", h, D); free(h);

    h=(float*)malloc(L*sD*4);
    cudaMemcpy(h,dWq_g,L*sD*4,cudaMemcpyDeviceToHost);sv("/tmp/fm_dwq.bin",h,L*sD);
    cudaMemcpy(h,dWk_g,L*sD*4,cudaMemcpyDeviceToHost);sv("/tmp/fm_dwk.bin",h,L*sD);
    cudaMemcpy(h,dWv_g,L*sD*4,cudaMemcpyDeviceToHost);sv("/tmp/fm_dwv.bin",h,L*sD);
    cudaMemcpy(h,dWo_g,L*sD*4,cudaMemcpyDeviceToHost);sv("/tmp/fm_dwo.bin",h,L*sD);
    free(h);

    sv_loss("/tmp/loss_s1.bin", dloss, T);

    // ===== PERSISTENT M/V (shared across step 1 & 2) =====
    #define MK_MV(NM, N) float* mv_##NM##_m; float* mv_##NM##_v; \
        cudaMalloc(&mv_##NM##_m, (N)*4); cudaMalloc(&mv_##NM##_v, (N)*4); \
        cudaMemset(mv_##NM##_m, 0, (N)*4); cudaMemset(mv_##NM##_v, 0, (N)*4);
    MK_MV(wte, V*D);   MK_MV(wpe, TMAX*D);
    MK_MV(lnfg, D);    MK_MV(lnfb, D);
    MK_MV(wq, L*sD);   MK_MV(wk, L*sD);   MK_MV(wv, L*sD);   MK_MV(wo, L*sD);
    MK_MV(ln1g, L*D);  MK_MV(ln1b, L*D);
    MK_MV(w1, L*sDFF); MK_MV(b1, L*DFF);
    MK_MV(w2, L*sDFF); MK_MV(b2, L*D);
    MK_MV(ln2g, L*D);  MK_MV(ln2b, L*D);
    #undef MK_MV

    // ===== ADAMW 2-STEP LOOP (persistent m/v, step-1 grads reused) =====
    float adamw_lr = 1e-3f, adamw_b1 = 0.9f, adamw_b2 = 0.999f;
    float adamw_eps = 1e-8f, adamw_wd = 0.01f;
    for (int step = 1; step <= N_STEPS; step++) {
        float bc1 = (step == 1) ? (1.0f - 0.9f)   : (1.0f - 0.9f*0.9f);
        float bc2 = (step == 1) ? (1.0f - 0.999f) : (1.0f - 0.999f*0.999f);

        #define ADAMW(W, G, NM, N) adamw_dev((W), (G), mv_##NM##_m, mv_##NM##_v, \
            adamw_lr, adamw_b1, adamw_b2, adamw_eps, adamw_wd, bc1, bc2, (N))
        ADAMW(dwte,  dwte_g,  wte,  V*D);
        ADAMW(dwpe,  dwpe_g,  wpe,  TMAX*D);
        ADAMW(dlnfg, dlnfg_g, lnfg, D);
        ADAMW(dlnfb, dlnfb_g, lnfb, D);
        ADAMW(dWq,   dWq_g,   wq,   L*sD);
        ADAMW(dWk,   dWk_g,   wk,   L*sD);
        ADAMW(dWv,   dWv_g,   wv,   L*sD);
        ADAMW(dWo,   dWo_g,   wo,   L*sD);
        ADAMW(dln1g, dln1g_g, ln1g, L*D);
        ADAMW(dln1b, dln1b_g, ln1b, L*D);
        ADAMW(dW1,   dW1_g,   w1,   L*sDFF);
        ADAMW(db1,   db1_g,   b1,   L*DFF);
        ADAMW(dW2,   dW2_g,   w2,   L*sDFF);
        ADAMW(db2,   db2_g,   b2,   L*D);
        ADAMW(dln2g, dln2g_g, ln2g, L*D);
        ADAMW(dln2b, dln2b_g, ln2b, L*D);
        #undef ADAMW
        cudaDeviceSynchronize();

        #define SAVE_S(NM, PTR, N) do { \
            float* _h = (float*)malloc((N)*4); \
            cudaMemcpy(_h, (PTR), (N)*4, cudaMemcpyDeviceToHost); \
            char _p[64]; snprintf(_p, sizeof(_p), "/tmp/ts_s%d_" NM ".bin", step); \
            sv(_p, _h, (N)); free(_h); \
        } while(0)
        SAVE_S("wte",  dwte,  V*D);
        SAVE_S("wpe",  dwpe,  TMAX*D);
        SAVE_S("lnfg", dlnfg, D);
        SAVE_S("lnfb", dlnfb, D);
        SAVE_S("wq",   dWq,   L*sD);
        SAVE_S("wk",   dWk,   L*sD);
        SAVE_S("wv",   dWv,   L*sD);
        SAVE_S("wo",   dWo,   L*sD);
        SAVE_S("ln1g", dln1g, L*D);
        SAVE_S("ln1b", dln1b, L*D);
        SAVE_S("w1",   dW1,   L*sDFF);
        SAVE_S("b1",   db1,   L*DFF);
        SAVE_S("w2",   dW2,   L*sDFF);
        SAVE_S("b2",   db2,   L*D);
        SAVE_S("ln2g", dln2g, L*D);
        SAVE_S("ln2b", dln2b, L*D);
        #undef SAVE_S

        // ===== STEP 2: Re-run fwd+bwd with UPDATED weights =====
        if (step < N_STEPS) {
            // --- FORWARD (step-2) ---
            emb_fwd_k<<<T, D>>>(dids, dwte, dwpe, dXs, T, D);
            cudaDeviceSynchronize();
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
            ln_fwd_dev(dXs + L*sx, dlnfg, dlnfb, dxnf, dmlf, drlf, T, D, eps);
            tied_head_fwd<<<T, V>>>(dxnf, dwte, dlogits, T, V, D);
            cudaDeviceSynchronize();
            softmax_ce_fwd<<<T, sm_b, sm_sh>>>(dlogits, dtg, dprobs, dloss, T, V);
            cudaDeviceSynchronize();

            { char _lp[64]; snprintf(_lp, sizeof(_lp), "/tmp/loss_s%d.bin", step+1); sv_loss(_lp, dloss, T); }

            if (step == 1) {
            { float* h=(float*)malloc(sx*4); cudaMemcpy(h, dXs+L*sx, sx*4, cudaMemcpyDeviceToHost); sv("/tmp/fm_s2_y.bin",h,sx); free(h); }
            { float* h=(float*)malloc(T*4); cudaMemcpy(h, dloss, T*4, cudaMemcpyDeviceToHost); sv("/tmp/fm_s2_loss.bin",h,T); free(h); }

            // Save g1 for ALL tensors before step-2 memset (needed for python verify)
            { float* h=(float*)malloc(L*sDFF*4); cudaMemcpy(h,dW1_g,L*sDFF*4,cudaMemcpyDeviceToHost);sv("/tmp/fm_dw1.bin",h,L*sDFF); free(h); }
            { float* h=(float*)malloc(L*DFF*4); cudaMemcpy(h,db1_g,L*DFF*4,cudaMemcpyDeviceToHost);sv("/tmp/fm_db1.bin",h,L*DFF); free(h); }
            { float* h=(float*)malloc(L*sDFF*4); cudaMemcpy(h,dW2_g,L*sDFF*4,cudaMemcpyDeviceToHost);sv("/tmp/fm_dw2.bin",h,L*sDFF); free(h); }
            { float* h=(float*)malloc(L*D*4); cudaMemcpy(h,db2_g,L*D*4,cudaMemcpyDeviceToHost);sv("/tmp/fm_db2.bin",h,L*D); free(h); }
            { float* h=(float*)malloc(L*D*4); cudaMemcpy(h,dln1g_g,L*D*4,cudaMemcpyDeviceToHost);sv("/tmp/fm_dln1g.bin",h,L*D); cudaMemcpy(h,dln1b_g,L*D*4,cudaMemcpyDeviceToHost);sv("/tmp/fm_dln1b.bin",h,L*D); free(h); }
            { float* h=(float*)malloc(L*D*4); cudaMemcpy(h,dln2g_g,L*D*4,cudaMemcpyDeviceToHost);sv("/tmp/fm_dln2g.bin",h,L*D); cudaMemcpy(h,dln2b_g,L*D*4,cudaMemcpyDeviceToHost);sv("/tmp/fm_dln2b.bin",h,L*D); free(h); }

            }

            // --- ZERO GRADS for step 2 ---
            cudaMemset(dWq_g,0,L*sD*4);cudaMemset(dWk_g,0,L*sD*4);
            cudaMemset(dWv_g,0,L*sD*4);cudaMemset(dWo_g,0,L*sD*4);
            cudaMemset(dln1g_g,0,L*D*4);cudaMemset(dln1b_g,0,L*D*4);
            cudaMemset(dW1_g,0,L*sDFF*4);cudaMemset(db1_g,0,L*DFF*4);
            cudaMemset(dW2_g,0,L*sDFF*4);cudaMemset(db2_g,0,L*D*4);
            cudaMemset(dln2g_g,0,L*D*4);cudaMemset(dln2b_g,0,L*D*4);
            cudaMemset(dwte_g,0,V*D*4);cudaMemset(dwpe_g,0,TMAX*D*4);
            cudaMemset(dlnfg_g,0,D*4);cudaMemset(dlnfb_g,0,D*4);

            // --- BACKWARD (step-2) ---
            softmax_ce_bwd<<<dlg_b, 256>>>(dprobs, dtg, dlogits, T, V, invT);
            cudaDeviceSynchronize();
            cudaMemcpy(dxnf_fwd, dxnf, sx*4, cudaMemcpyDeviceToDevice);
            tied_head_bwd_dxnf<<<T,D>>>(dlogits, dwte, dxnf, T,V,D);
            tied_head_bwd_dwte<<<V,D>>>(dlogits, dxnf_fwd, dwte_g, T,V,D);
            cudaDeviceSynchronize();
            ln_bwd_dev(dxnf, dXs+L*sx, dlnfg, dmlf, drlf, dY_temp, dlnfg_g, dlnfb_g, T,D);
            cudaDeviceSynchronize();
            for (int b=L-1; b>=0; b--) {
                float *bWq=dWq+b*sD,*bWk=dWk+b*sD,*bWv=dWv+b*sD,*bWo=dWo+b*sD;
                float *bW1=dW1+b*sDFF,*bW2=dW2+b*sDFF;
                float *bln1g=dln1g+b*D,*bln2g=dln2g+b*D;
                float *bh1=dh1+b*sx;
                float *bQ=dQ+b*sx,*bK=dK+b*sx,*bV=dV+b*sx;
                float *bA=dA+b*sx;
                float *bX1=dX1_fwd+b*sx;
                float *bh2=dh2+b*sx;
                float *bpre_save=dpre_save+b*sh;
                float *bpre=dpre+b*sh;
                float *bm1=dm1+b*T,*br1=dr1+b*T;
                float *bm2=dm2+b*T,*br2=dr2+b*T;

                cudaMemcpy(dX1_temp, dY_temp, sx*4, cudaMemcpyDeviceToDevice);
                mm_tn(bpre, dY_temp, dW2_g+b*sDFF, DFF, T, D);
                colsum_k<<<(D+255)/256,256>>>(dY_temp, db2_g+b*D, T, D);
                mm_nt(dY_temp, bW2, dhg_temp, T, D, DFF);
                gelu_bwd_dev(dhg_temp, bpre_save, dpre_temp, sh);
                mm_tn(bh2, dpre_temp, dW1_g+b*sDFF, D, T, DFF);
                colsum_k<<<(DFF+255)/256,256>>>(dpre_temp, db1_g+b*DFF, T, DFF);
                mm_nt(dpre_temp, bW1, dh2_temp, T, DFF, D);
                add_dev(dX1_temp, dh2_temp, sx);
                ln_bwd_dev(dh2_temp, bX1, bln2g, bm2, br2, dT_sx_a, dln2g_g+b*D, dln2b_g+b*D, T,D);
                add_dev(dX1_temp, dT_sx_a, sx);
                mm_tn(bA, dX1_temp, dWo_g+b*sD, D, T, D);
                mm_nt(dX1_temp, bWo, dA_temp, T, D, D);
                cudaMemset(dQ_t,0,sx*4);cudaMemset(dK_t,0,sx*4);cudaMemset(dV_t,0,sx*4);
                attn_bwd_k<<<grd,blk>>>(bQ, bK, bV, dA_temp, dQ_t, dK_t, dV_t, T, D, H, sc);
                cudaDeviceSynchronize();
                mm_tn(bh1, dQ_t, dWq_g+b*sD, D, T, D);
                mm_tn(bh1, dK_t, dWk_g+b*sD, D, T, D);
                mm_tn(bh1, dV_t, dWv_g+b*sD, D, T, D);
                mm_nt(dQ_t, bWq, dh1_temp, T, D, D);
                mm_nt(dK_t, bWk, dT_sx_a, T, D, D);
                add_dev(dh1_temp, dT_sx_a, sx);
                mm_nt(dV_t, bWv, dT_sx_a, T, D, D);
                add_dev(dh1_temp, dT_sx_a, sx);
                cudaMemcpy(dX_temp, dX1_temp, sx*4, cudaMemcpyDeviceToDevice);
                ln_bwd_dev(dh1_temp, dXs+b*sx, bln1g, bm1, br1, dT_sx_a, dln1g_g+b*D, dln1b_g+b*D, T,D);
                add_dev(dX_temp, dT_sx_a, sx);
                cudaMemcpy(dY_temp, dX_temp, sx*4, cudaMemcpyDeviceToDevice);
            }
            emb_bwd_k<<<T, D>>>(dids, dY_temp, dwte_g, dwpe_g, T, D);
            cudaDeviceSynchronize();

            if (step == 1) {
            { float* h=(float*)malloc(V*D*4); cudaMemcpy(h, dwte_g, V*D*4, cudaMemcpyDeviceToHost); sv("/tmp/fm_s2_dwte.bin",h,V*D); free(h); }
            { float* h=(float*)malloc(TMAX*D*4); cudaMemcpy(h, dwpe_g, TMAX*D*4, cudaMemcpyDeviceToHost); sv("/tmp/fm_s2_dwpe.bin",h,TMAX*D); free(h); }
            { float* h=(float*)malloc(D*4); cudaMemcpy(h, dlnfg_g, D*4, cudaMemcpyDeviceToHost); sv("/tmp/fm_s2_dlnfg.bin",h,D); cudaMemcpy(h, dlnfb_g, D*4, cudaMemcpyDeviceToHost); sv("/tmp/fm_s2_dlnfb.bin",h,D); free(h); }
            { float* h=(float*)malloc(L*sD*4); cudaMemcpy(h,dWq_g,L*sD*4,cudaMemcpyDeviceToHost);sv("/tmp/fm_s2_dwq.bin",h,L*sD); cudaMemcpy(h,dWk_g,L*sD*4,cudaMemcpyDeviceToHost);sv("/tmp/fm_s2_dwk.bin",h,L*sD); cudaMemcpy(h,dWv_g,L*sD*4,cudaMemcpyDeviceToHost);sv("/tmp/fm_s2_dwv.bin",h,L*sD); cudaMemcpy(h,dWo_g,L*sD*4,cudaMemcpyDeviceToHost);sv("/tmp/fm_s2_dwo.bin",h,L*sD); free(h); }
            { float* h=(float*)malloc(L*D*4); cudaMemcpy(h,dln1g_g,L*D*4,cudaMemcpyDeviceToHost);sv("/tmp/fm_s2_dln1g.bin",h,L*D); cudaMemcpy(h,dln1b_g,L*D*4,cudaMemcpyDeviceToHost);sv("/tmp/fm_s2_dln1b.bin",h,L*D); cudaMemcpy(h,dln2g_g,L*D*4,cudaMemcpyDeviceToHost);sv("/tmp/fm_s2_dln2g.bin",h,L*D); cudaMemcpy(h,dln2b_g,L*D*4,cudaMemcpyDeviceToHost);sv("/tmp/fm_s2_dln2b.bin",h,L*D); free(h); }
            { float* h=(float*)malloc(L*sDFF*4); cudaMemcpy(h,dW1_g,L*sDFF*4,cudaMemcpyDeviceToHost);sv("/tmp/fm_s2_dw1.bin",h,L*sDFF); cudaMemcpy(h,dW2_g,L*sDFF*4,cudaMemcpyDeviceToHost);sv("/tmp/fm_s2_dw2.bin",h,L*sDFF); free(h); }
            { float* h=(float*)malloc(L*DFF*4); cudaMemcpy(h,db1_g,L*DFF*4,cudaMemcpyDeviceToHost);sv("/tmp/fm_s2_db1.bin",h,L*DFF); free(h); }
            { float* h=(float*)malloc(L*D*4); cudaMemcpy(h,db2_g,L*D*4,cudaMemcpyDeviceToHost);sv("/tmp/fm_s2_db2.bin",h,L*D); free(h); }
            }
        }
    }

    printf("Train step 1 done: T=%d D=%d L=%d V=%d\\n", T,D,L,V);
    return 0;
}
