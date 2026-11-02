#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// LayerNorm forward: y = (x - mean) / sqrt(var + eps) * gamma + beta
__global__ void ln_fwd_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    float* __restrict__ mean_out,
    float* __restrict__ rstd_out,
    int M, int D, float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float sdata[];

    float val  = (tid < D) ? x[row * D + tid] : 0.0f;
    sdata[tid] = val;
    __syncthreads();

    // Reduce sum -> mean
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean = sdata[0] / (float)D;
    __syncthreads();

    // Reduce sum of centered squared -> var
    sdata[tid] = (tid < D) ? (val - mean) * (val - mean) : 0.0f;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float var  = sdata[0] / (float)D;
    float rstd = rsqrtf(var + eps);
    __syncthreads();

    if (tid < D)
        y[row * D + tid] = (val - mean) * rstd * gamma[tid] + beta[tid];
    if (tid == 0) {
        mean_out[row] = mean;
        rstd_out[row] = rstd;
    }
}

// LayerNorm backward
__global__ void ln_bwd_kernel(
    const float* __restrict__ dy,
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ mean_in,
    const float* __restrict__ rstd_in,
    float* __restrict__ dx,
    float* __restrict__ dgamma,
    float* __restrict__ dbeta,
    int M, int D)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float sdata[];

    float m   = mean_in[row];
    float r   = rstd_in[row];
    float dyv = (tid < D) ? dy[row * D + tid] : 0.0f;
    float xv  = (tid < D) ? x[row * D + tid]  : 0.0f;
    float xhat = (tid < D) ? (xv - m) * r : 0.0f;
    float dxhat = (tid < D) ? dyv * gamma[tid] : 0.0f;

    // sum(dxhat)
    sdata[tid] = dxhat;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float s_dxhat = sdata[0];
    __syncthreads();

    // sum(dxhat * xhat)
    sdata[tid] = dxhat * xhat;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float s_dxhat_xhat = sdata[0];

    if (tid < D) {
        dx[row * D + tid] = r * (dxhat - s_dxhat / (float)D
                                       - xhat * s_dxhat_xhat / (float)D);
        atomicAdd(&dgamma[tid], dyv * xhat);
        atomicAdd(&dbeta[tid],  dyv);
    }
}

static void load(const char* p, float* b, int n) {
    FILE* f = fopen(p, "rb"); if (!f) { perror(p); exit(1); }
    fread(b, sizeof(float), n, f); fclose(f);
}
static void save(const char* p, float* b, int n) {
    FILE* f = fopen(p, "wb"); if (!f) { perror(p); exit(1); }
    fwrite(b, sizeof(float), n, f); fclose(f);
}

int main(int argc, char** argv) {
    int M = 128, D = 256;
    float eps = 1e-5f;
    if (argc >= 3) { M = atoi(argv[1]); D = atoi(argv[2]); }

    size_t sx = (size_t)M*D;

    float *x, *gamma, *beta, *dy;
    cudaMallocHost(&x,     sx*sizeof(float));
    cudaMallocHost(&dy,    sx*sizeof(float));
    cudaMallocHost(&gamma, D*sizeof(float));
    cudaMallocHost(&beta,  D*sizeof(float));

    load("/tmp/ln_x.bin",     x,     sx);
    load("/tmp/ln_dy.bin",    dy,    sx);
    load("/tmp/ln_gamma.bin", gamma, D);
    load("/tmp/ln_beta.bin",  beta,  D);

    float *dx   = (float*)malloc(sx*sizeof(float));
    float *y    = (float*)malloc(sx*sizeof(float));
    float *mean = (float*)malloc(M*sizeof(float));
    float *rstd = (float*)malloc(M*sizeof(float));
    float *dgamma = (float*)malloc(D*sizeof(float));
    float *dbeta  = (float*)malloc(D*sizeof(float));

    for (int i = 0; i < D; i++) { dgamma[i] = 0.0f; dbeta[i] = 0.0f; }

    float *d_x, *d_g, *d_b, *d_y, *d_dy, *d_dx, *d_dgamma, *d_dbeta, *d_mean, *d_rstd;
    cudaMalloc(&d_x,      sx*sizeof(float));
    cudaMalloc(&d_y,      sx*sizeof(float));
    cudaMalloc(&d_dy,     sx*sizeof(float));
    cudaMalloc(&d_dx,     sx*sizeof(float));
    cudaMalloc(&d_g,      D*sizeof(float));
    cudaMalloc(&d_b,      D*sizeof(float));
    cudaMalloc(&d_dgamma, D*sizeof(float));
    cudaMalloc(&d_dbeta,  D*sizeof(float));
    cudaMalloc(&d_mean,   M*sizeof(float));
    cudaMalloc(&d_rstd,   M*sizeof(float));

    cudaMemcpy(d_x,  x,     sx*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dy, dy,    sx*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g,  gamma, D*sizeof(float),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,  beta,  D*sizeof(float),  cudaMemcpyHostToDevice);

    // blockDim = next power of 2 >= D, capped at 1024
    int block = 1; while (block < D) block <<= 1;
    if (block > 1024) block = 1024;
    int shared_bytes = block * sizeof(float);

    // ---- Forward
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int it = 0; it < 100; it++)
        ln_fwd_kernel<<<M, block, shared_bytes>>>(d_x, d_g, d_b, d_y, d_mean, d_rstd, M, D, eps);
    cudaEventRecord(t1);
    cudaDeviceSynchronize();
    float fwd_ms = 0; cudaEventElapsedTime(&fwd_ms, t0, t1);

    cudaMemcpy(y,    d_y,    sx*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(mean, d_mean, M*sizeof(float),  cudaMemcpyDeviceToHost);
    cudaMemcpy(rstd, d_rstd, M*sizeof(float),  cudaMemcpyDeviceToHost);

    // ---- Backward
    cudaMemset(d_dgamma, 0, D*sizeof(float));
    cudaMemset(d_dbeta,  0, D*sizeof(float));

    cudaEventRecord(t0);
    for (int it = 0; it < 100; it++)
        ln_bwd_kernel<<<M, block, shared_bytes>>>(d_dy, d_x, d_g, d_mean, d_rstd, d_dx, d_dgamma, d_dbeta, M, D);
    cudaEventRecord(t1);
    cudaDeviceSynchronize();
    float bwd_ms = 0; cudaEventElapsedTime(&bwd_ms, t0, t1);

    // For dgamma/dbeta use single (non-loop) accurate pass
    cudaMemset(d_dgamma, 0, D*sizeof(float));
    cudaMemset(d_dbeta,  0, D*sizeof(float));
    ln_bwd_kernel<<<M, block, shared_bytes>>>(d_dy, d_x, d_g, d_mean, d_rstd, d_dx, d_dgamma, d_dbeta, M, D);
    cudaDeviceSynchronize();

    cudaMemcpy(dx,     d_dx,     sx*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dgamma, d_dgamma, D*sizeof(float),  cudaMemcpyDeviceToHost);
    cudaMemcpy(dbeta,  d_dbeta,  D*sizeof(float),  cudaMemcpyDeviceToHost);

    save("/tmp/ln_y.bin",      y,      sx);
    save("/tmp/ln_dx.bin",     dx,     sx);
    save("/tmp/ln_dgamma.bin", dgamma, D);
    save("/tmp/ln_dbeta.bin",  dbeta,  D);

    printf("LN fwd: %.4f ms/run (100-avg)  [%dx%d]\n", fwd_ms/100, M, D);
    printf("LN bwd: %.4f ms/run (100-avg)\n", bwd_ms/100);

    cudaFreeHost(x); cudaFreeHost(dy); cudaFreeHost(gamma); cudaFreeHost(beta);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_dy); cudaFree(d_dx);
    cudaFree(d_g); cudaFree(d_b); cudaFree(d_dgamma); cudaFree(d_dbeta);
    cudaFree(d_mean); cudaFree(d_rstd);
    free(y); free(dx); free(mean); free(rstd); free(dgamma); free(dbeta);
    return 0;
}
