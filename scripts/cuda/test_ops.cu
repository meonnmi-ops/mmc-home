#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// ---------- GELU (tanh approx) ----------
__global__ void gelu_fwd(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    float c = 0.7978845608f;
    float inner = c * (v + 0.044715f * v * v * v);
    y[i] = 0.5f * v * (1.0f + tanhf(inner));
}

__global__ void gelu_bwd(const float* dy, const float* x, float* dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    float c = 0.7978845608f;
    float inner = c * (v + 0.044715f * v * v * v);
    float t = tanhf(inner);
    float dt = c * (1.0f + 3.0f * 0.044715f * v * v);
    float g = 0.5f * (1.0f + t) + 0.5f * v * (1.0f - t*t) * dt;
    dx[i] = dy[i] * g;
}

// ---------- Softmax + CE ----------
__global__ void softmax_ce_fwd(const float* logits, const int* tg,
                               float* probs, float* loss,
                               int T, int V) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float s[];

    float v = (tid < V) ? logits[row*V + tid] : -1e30f;
    s[tid] = v; __syncthreads();
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (tid < st) s[tid] = fmaxf(s[tid], s[tid+st]);
        __syncthreads();
    }
    float mx = s[0]; __syncthreads();

    float e = (tid < V) ? expf(v - mx) : 0.0f;
    s[tid] = e; __syncthreads();
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (tid < st) s[tid] += s[tid+st];
        __syncthreads();
    }
    float sum = s[0];

    if (tid < V) probs[row*V + tid] = e / sum;
    if (tid == 0)
        loss[row] = -((logits[row*V + tg[row]] - mx) - logf(sum));
}

__global__ void softmax_ce_bwd(const float* probs, const int* tg,
                               float* dlogits, int T, int V, float invT) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T*V) return;
    int r = i / V;
    int c = i % V;
    float p = probs[i];
    float oh = (c == tg[r]) ? 1.0f : 0.0f;
    dlogits[i] = (p - oh) * invT;
}

// ---------- AdamW ----------
__global__ void adamw_step(float* w, const float* g, float* m, float* v,
                           float lr, float b1, float b2, float eps, float wd,
                           float bc1, float bc2, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    m[i] = b1 * m[i] + (1.0f - b1) * g[i];
    v[i] = b2 * v[i] + (1.0f - b2) * g[i] * g[i];
    float mh = m[i] / bc1;
    float vh = v[i] / bc2;
    w[i] = w[i] * (1.0f - lr * wd) - lr * mh / (sqrtf(vh) + eps);
}

// ---------- helpers ----------
static void ld(const char* p, void* b, size_t by) {
    FILE* f = fopen(p, "rb"); if (!f) { perror(p); exit(1); }
    fread(b, 1, by, f); fclose(f);
}
static void sv(const char* p, const void* b, size_t by) {
    FILE* f = fopen(p, "wb"); if (!f) { perror(p); exit(1); }
    fwrite(b, 1, by, f); fclose(f);
}

// ---------- main ----------
int main() {
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    // === GELU (n = 128*256) ===
    int n = 128 * 256;
    float *hx  = (float*)malloc(n*sizeof(float));
    float *hdy = (float*)malloc(n*sizeof(float));
    float *hy  = (float*)malloc(n*sizeof(float));
    float *hdx = (float*)malloc(n*sizeof(float));
    ld("/tmp/gelu_x.bin",  hx,  n*sizeof(float));
    ld("/tmp/gelu_dy.bin", hdy, n*sizeof(float));

    float *d_x, *d_y, *d_dy, *d_dxx;
    cudaMalloc(&d_x, n*sizeof(float));
    cudaMalloc(&d_y, n*sizeof(float));
    cudaMalloc(&d_dy, n*sizeof(float));
    cudaMalloc(&d_dxx, n*sizeof(float));
    cudaMemcpy(d_x,  hx,  n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dy, hdy, n*sizeof(float), cudaMemcpyHostToDevice);

    int th = 256, bl = (n + th - 1) / th;
    cudaEventRecord(t0);
    for (int i = 0; i < 100; i++) gelu_fwd<<<bl, th>>>(d_x, d_y, n);
    cudaEventRecord(t1); cudaDeviceSynchronize();
    float g_fwd = 0; cudaEventElapsedTime(&g_fwd, t0, t1);

    cudaEventRecord(t0);
    for (int i = 0; i < 100; i++) gelu_bwd<<<bl, th>>>(d_dy, d_x, d_dxx, n);
    cudaEventRecord(t1); cudaDeviceSynchronize();
    float g_bwd = 0; cudaEventElapsedTime(&g_bwd, t0, t1);

    cudaMemcpy(hy,  d_y,  n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hdx, d_dxx, n*sizeof(float), cudaMemcpyDeviceToHost);
    sv("/tmp/gelu_y.bin",  hy,  n*sizeof(float));
    sv("/tmp/gelu_dx.bin", hdx, n*sizeof(float));

    // === Softmax + CE (T=128, V=256) ===
    int T = 128, V = 256;
    size_t sz = (size_t)T * V;
    float *hlog = (float*)malloc(sz*sizeof(float));
    int   *htg  = (int*)malloc(T*sizeof(int));
    float *hpr  = (float*)malloc(sz*sizeof(float));
    float *hls  = (float*)malloc(T*sizeof(float));
    float *hdlg = (float*)malloc(sz*sizeof(float));
    ld("/tmp/sm_logits.bin", hlog, sz*sizeof(float));
    ld("/tmp/sm_tg.bin",     htg,  T*sizeof(int));

    float *d_log, *d_pr, *d_dlg, *d_loss;
    int   *d_tg;
    cudaMalloc(&d_log, sz*sizeof(float));
    cudaMalloc(&d_pr,  sz*sizeof(float));
    cudaMalloc(&d_dlg, sz*sizeof(float));
    cudaMalloc(&d_loss, T*sizeof(float));
    cudaMalloc(&d_tg, T*sizeof(int));
    cudaMemcpy(d_log, hlog, sz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tg,  htg,  T*sizeof(int), cudaMemcpyHostToDevice);

    int smb = 1; while (smb < V) smb <<= 1;
    int sm_sh = smb * sizeof(float);

    cudaEventRecord(t0);
    for (int i = 0; i < 100; i++)
        softmax_ce_fwd<<<T, smb, sm_sh>>>(d_log, d_tg, d_pr, d_loss, T, V);
    cudaEventRecord(t1); cudaDeviceSynchronize();
    float s_fwd = 0; cudaEventElapsedTime(&s_fwd, t0, t1);

    cudaMemcpy(hpr, d_pr,   sz*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hls, d_loss, T*sizeof(float), cudaMemcpyDeviceToHost);

    float invT = 1.0f / T;
    int sm_bl = (T*V + 255) / 256;
    cudaEventRecord(t0);
    for (int i = 0; i < 100; i++)
        softmax_ce_bwd<<<sm_bl, 256>>>(d_pr, d_tg, d_dlg, T, V, invT);
    cudaEventRecord(t1); cudaDeviceSynchronize();
    float s_bwd = 0; cudaEventElapsedTime(&s_bwd, t0, t1);

    cudaMemcpy(hdlg, d_dlg, sz*sizeof(float), cudaMemcpyDeviceToHost);
    sv("/tmp/sm_probs.bin",   hpr,  sz*sizeof(float));
    sv("/tmp/sm_loss.bin",    hls,  T*sizeof(float));
    sv("/tmp/sm_dlogits.bin", hdlg, sz*sizeof(float));

    // === AdamW (n = 1024) ===
    int an = 1024;
    float *h_w = (float*)malloc(an*sizeof(float));
    float *h_g = (float*)malloc(an*sizeof(float));
    float *h_m = (float*)malloc(an*sizeof(float));
    float *h_v = (float*)malloc(an*sizeof(float));
    ld("/tmp/aw_w.bin", h_w, an*sizeof(float));
    ld("/tmp/aw_g.bin", h_g, an*sizeof(float));
    ld("/tmp/aw_m.bin", h_m, an*sizeof(float));
    ld("/tmp/aw_v.bin", h_v, an*sizeof(float));

    float *d_w, *d_g, *d_m, *d_v;
    cudaMalloc(&d_w, an*sizeof(float));
    cudaMalloc(&d_g, an*sizeof(float));
    cudaMalloc(&d_m, an*sizeof(float));
    cudaMalloc(&d_v, an*sizeof(float));
    cudaMemcpy(d_w, h_w, an*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, h_g, an*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, h_m, an*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, an*sizeof(float), cudaMemcpyHostToDevice);

    float lr = 1e-3f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, wd = 0.01f;
    int stp = 5;
    float bc1 = 1.0f - powf(b1, (float)stp);
    float bc2 = 1.0f - powf(b2, (float)stp);
    int aw_bl = (an + 255) / 256;

    cudaEventRecord(t0);
    for (int i = 0; i < 100; i++)
        adamw_step<<<aw_bl, 256>>>(d_w, d_g, d_m, d_v, lr, b1, b2, eps, wd, bc1, bc2, an);
    cudaEventRecord(t1); cudaDeviceSynchronize();
    float a_ms = 0; cudaEventElapsedTime(&a_ms, t0, t1);

    // single-step correctness (reset state first)
    cudaMemcpy(d_w, h_w, an*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, h_m, an*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, an*sizeof(float), cudaMemcpyHostToDevice);
    adamw_step<<<aw_bl, 256>>>(d_w, d_g, d_m, d_v, lr, b1, b2, eps, wd, bc1, bc2, an);
    cudaDeviceSynchronize();

    cudaMemcpy(h_w, d_w, an*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_m, d_m, an*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_v, d_v, an*sizeof(float), cudaMemcpyDeviceToHost);
    sv("/tmp/aw_w_out.bin", h_w, an*sizeof(float));
    sv("/tmp/aw_m_out.bin", h_m, an*sizeof(float));
    sv("/tmp/aw_v_out.bin", h_v, an*sizeof(float));

    printf("GELU fwd: %.4f ms/run\n", g_fwd/100);
    printf("GELU bwd: %.4f ms/run\n", g_bwd/100);
    printf("Softmax+CE fwd: %.4f ms/run\n", s_fwd/100);
    printf("Softmax+CE bwd: %.4f ms/run\n", s_bwd/100);
    printf("AdamW step: %.4f ms/run\n", a_ms/100);

    // cleanup
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_dy); cudaFree(d_dxx);
    cudaFree(d_log); cudaFree(d_pr); cudaFree(d_dlg); cudaFree(d_loss); cudaFree(d_tg);
    cudaFree(d_w); cudaFree(d_g); cudaFree(d_m); cudaFree(d_v);
    free(hx); free(hdy); free(hy); free(hdx);
    free(hlog); free(htg); free(hpr); free(hls); free(hdlg);
    free(h_w); free(h_g); free(h_m); free(h_v);
    return 0;
}
