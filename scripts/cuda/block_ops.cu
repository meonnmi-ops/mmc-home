#include <math.h>
#include <cuda_runtime.h>

// ---------- Kernels ----------
__global__ void ln_fwd_k(const float* __restrict__ x, const float* __restrict__ g,
                         const float* __restrict__ b, float* __restrict__ y,
                         float* __restrict__ mean, float* __restrict__ rstd,
                         int M, int D, float eps) {
    int row = blockIdx.x, tid = threadIdx.x;
    extern __shared__ float sd[];
    float v = (tid < D) ? x[row*D+tid] : 0.0f;
    sd[tid] = v; __syncthreads();
    for (int s=blockDim.x>>1; s>0; s>>=1) { if (tid<s) sd[tid]+=sd[tid+s]; __syncthreads(); }
    float m = sd[0]/(float)D; __syncthreads();
    sd[tid] = (tid<D) ? (v-m)*(v-m) : 0.0f; __syncthreads();
    for (int s=blockDim.x>>1; s>0; s>>=1) { if (tid<s) sd[tid]+=sd[tid+s]; __syncthreads(); }
    float va = sd[0]/(float)D; float r = rsqrtf(va+eps);
    if (tid < D) y[row*D+tid] = (v-m)*r*g[tid] + b[tid];
    if (tid == 0) { mean[row]=m; rstd[row]=r; }
}

__global__ void ln_bwd_k(const float* __restrict__ dy, const float* __restrict__ x,
                         const float* __restrict__ g, const float* __restrict__ mean,
                         const float* __restrict__ rstd,
                         float* __restrict__ dx, float* __restrict__ dgamma,
                         float* __restrict__ dbeta, int M, int D) {
    int row = blockIdx.x, tid = threadIdx.x;
    extern __shared__ float sd[];
    float m = mean[row], r = rstd[row];
    float dyv = (tid<D) ? dy[row*D+tid] : 0.0f;
    float xv  = (tid<D) ? x[row*D+tid]  : 0.0f;
    float xhat = (tid<D) ? (xv-m)*r : 0.0f;
    float dxhat = (tid<D) ? dyv*g[tid] : 0.0f;
    sd[tid] = dxhat; __syncthreads();
    for (int s=blockDim.x>>1; s>0; s>>=1) { if (tid<s) sd[tid]+=sd[tid+s]; __syncthreads(); }
    float sum1 = sd[0]; __syncthreads();
    sd[tid] = dxhat*xhat; __syncthreads();
    for (int s=blockDim.x>>1; s>0; s>>=1) { if (tid<s) sd[tid]+=sd[tid+s]; __syncthreads(); }
    float sum2 = sd[0];
    if (tid < D) {
        dx[row*D+tid] = r*(dxhat - sum1/(float)D - xhat*sum2/(float)D);
        atomicAdd(&dgamma[tid], dyv*xhat);
        atomicAdd(&dbeta[tid],  dyv);
    }
}

__global__ void gelu_fwd_k(const float* x, float* y, int n) {
    int i = blockIdx.x*blockDim.x+threadIdx.x; if (i>=n) return;
    float v = x[i];
    float inner = 0.7978845608f*(v + 0.044715f*v*v*v);
    y[i] = 0.5f*v*(1+tanhf(inner));
}

__global__ void gelu_bwd_k(const float* dy, const float* x, float* dx, int n) {
    int i = blockIdx.x*blockDim.x+threadIdx.x; if (i>=n) return;
    float v = x[i];
    float inner = 0.7978845608f*(v + 0.044715f*v*v*v);
    float t = tanhf(inner);
    float dt = 0.7978845608f*(1 + 3*0.044715f*v*v);
    dx[i] = dy[i]*(0.5f*(1+t) + 0.5f*v*(1-t*t)*dt);
}

__global__ void add_k(float* a, const float* b, int n) {
    int i = blockIdx.x*blockDim.x+threadIdx.x; if (i>=n) return;
    a[i] += b[i];
}

__global__ void add_bias_k(float* y, const float* b, int T, int D) {
    int i = blockIdx.x*blockDim.x+threadIdx.x; if (i>=T*D) return;
    y[i] += b[i % D];
}


__global__ void adamw_step_k(float* w, const float* g, float* m, float* v,
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

// ---------- Device API ----------
extern "C" {

void ln_fwd_dev(const float* d_x, const float* d_g, const float* d_b,
                float* d_y, float* d_mean, float* d_rstd,
                int M, int D, float eps) {
    int block = 1; while (block < D) block <<= 1; if (block > 1024) block = 1024;
    int sh = block * sizeof(float);
    ln_fwd_k<<<M, block, sh>>>(d_x, d_g, d_b, d_y, d_mean, d_rstd, M, D, eps);
}

void ln_bwd_dev(const float* d_dy, const float* d_x, const float* d_g,
                const float* d_mean, const float* d_rstd,
                float* d_dx, float* d_dgamma, float* d_dbeta,
                int M, int D) {
    int block = 1; while (block < D) block <<= 1; if (block > 1024) block = 1024;
    int sh = block * sizeof(float);
    ln_bwd_k<<<M, block, sh>>>(d_dy, d_x, d_g, d_mean, d_rstd,
                                d_dx, d_dgamma, d_dbeta, M, D);
}

void gelu_fwd_dev(const float* d_x, float* d_y, int n) {
    gelu_fwd_k<<<(n+255)/256, 256>>>(d_x, d_y, n);
}

void gelu_bwd_dev(const float* d_dy, const float* d_x, float* d_dx, int n) {
    gelu_bwd_k<<<(n+255)/256, 256>>>(d_dy, d_x, d_dx, n);
}

void add_dev(float* d_a, const float* d_b, int n) {
    add_k<<<(n+255)/256, 256>>>(d_a, d_b, n);
}

void add_bias_dev(float* d_y, const float* d_b, int T, int D) {
    int n = T*D;
    add_bias_k<<<(n+255)/256, 256>>>(d_y, d_b, T, D);
}


void adamw_dev(float* d_w, const float* d_g, float* d_m, float* d_v,
               float lr, float b1, float b2, float eps, float wd,
               float bc1, float bc2, int n) {
    adamw_step_k<<<(n+255)/256, 256>>>(d_w, d_g, d_m, d_v, lr, b1, b2, eps, wd, bc1, bc2, n);
}

}  // extern "C"
