#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// MLP: Y = GELU(X @ W1 + b1) @ W2 + b2
// X[T,D], W1[D,DFF], b1[DFF], W2[DFF,D], b2[D]
// D=32, T=4, DFF=64

static cublasHandle_t gb;

__global__ void add_bias(float* y, const float* b, int T, int Dd) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= T*Dd) return;
    y[i] += b[i % Dd];
}

__global__ void gelu_fwd(const float* x, float* y, int n) {
    int i = blockIdx.x*blockDim.x+threadIdx.x; if (i>=n) return;
    float v = x[i];
    float inner = 0.7978845608f*(v + 0.044715f*v*v*v);
    y[i] = 0.5f*v*(1+tanhf(inner));
}

__global__ void gelu_bwd(const float* dy, const float* x, float* dx, int n) {
    int i = blockIdx.x*blockDim.x+threadIdx.x; if (i>=n) return;
    float v = x[i];
    float inner = 0.7978845608f*(v + 0.044715f*v*v*v);
    float t = tanhf(inner);
    float dt = 0.7978845608f*(1 + 3*0.044715f*v*v);
    dx[i] = dy[i]*(0.5f*(1+t) + 0.5f*v*(1-t*t)*dt);
}

// C_row[M,N] = A_row[M,K] @ B_row[K,N]
static void mm(const float* A, const float* B, float* C, int M, int K, int N) {
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(gb, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha, B, N, A, K, &beta, C, N);
}

// C_row[M,N] = A_row[M,K] @ B_row[N,K]^T   (B is [N,K])
static void mm_nt(const float* A, const float* B, float* C, int M, int K, int N) {
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(gb, CUBLAS_OP_T, CUBLAS_OP_N,
                N, M, K, &alpha, B, K, A, K, &beta, C, N);
}

// C_row[M,N] = A_row[K,M]^T @ B_row[K,N]   (A is [K,M])
static void mm_tn(const float* A, const float* B, float* C, int M, int K, int N) {
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(gb, CUBLAS_OP_N, CUBLAS_OP_T,
                N, M, K, &alpha, B, N, A, M, &beta, C, N);
}

static void ld(const char* p, float* b, int n) {
    FILE* f = fopen(p, "rb"); if (!f) { perror(p); exit(1); }
    fread(b, 4, n, f); fclose(f);
}
static void sv(const char* p, float* b, int n) {
    FILE* f = fopen(p, "wb"); if (!f) { perror(p); exit(1); }
    fwrite(b, 4, n, f); fclose(f);
}

int main() {
    cublasCreate(&gb);
    const int T=4, D=32, DFF=64;
    int sx=T*D, sh=T*DFF;

    // Host buffers
    float *hX=(float*)malloc(sx*4), *hW1=(float*)malloc(D*DFF*4), *hb1=(float*)malloc(DFF*4),
          *hW2=(float*)malloc(DFF*D*4), *hb2=(float*)malloc(D*4),
          *hdY=(float*)malloc(sx*4);
    ld("/tmp/mlp_x.bin", hX, sx); ld("/tmp/mlp_w1.bin", hW1, D*DFF);
    ld("/tmp/mlp_b1.bin", hb1, DFF); ld("/tmp/mlp_w2.bin", hW2, DFF*D);
    ld("/tmp/mlp_b2.bin", hb2, D); ld("/tmp/mlp_dy.bin", hdY, sx);

    // Device
    float *dX,*dW1,*db1,*dW2,*db2,*ddY;
    cudaMalloc(&dX,sx*4); cudaMalloc(&dW1,D*DFF*4); cudaMalloc(&db1,DFF*4);
    cudaMalloc(&dW2,DFF*D*4); cudaMalloc(&db2,D*4); cudaMalloc(&ddY,sx*4);
    cudaMemcpy(dX,hX,sx*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dW1,hW1,D*DFF*4,cudaMemcpyHostToDevice);
    cudaMemcpy(db1,hb1,DFF*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dW2,hW2,DFF*D*4,cudaMemcpyHostToDevice);
    cudaMemcpy(db2,hb2,D*4,cudaMemcpyHostToDevice);
    cudaMemcpy(ddY,hdY,sx*4,cudaMemcpyHostToDevice);

    // Activations
    float *h1, *g, *y;
    cudaMalloc(&h1, sh*4); cudaMalloc(&g, sh*4); cudaMalloc(&y, sx*4);

    // === Forward ===
    mm(dX, dW1, h1, T, D, DFF);
    add_bias<<<(sh+255)/256,256>>>(h1, db1, T, DFF);
    gelu_fwd<<<(sh+255)/256,256>>>(h1, g, sh);
    mm(g, dW2, y, T, DFF, D);
    add_bias<<<(sx+255)/256,256>>>(y, db2, T, D);

    // Save Y for verification
    float* hY=(float*)malloc(sx*4);
    cudaMemcpy(hY,y,sx*4,cudaMemcpyDeviceToHost);
    sv("/tmp/mlp_y.bin", hY, sx);

    // === Backward ===
    // dY upstream: hdY
    // db2 = sum rows of dY
    // dG = dY @ W2^T   → [T, DFF]
    // dW2 = G^T @ dY   → [DFF, D]
    // dH1 = dG * gelu'(H1)
    // db1 = sum rows of dH1
    // dW1 = X^T @ dH1  → [D, DFF]
    // dX = dH1 @ W1^T  → [T, D]

    float *dG, *dH1, *dXout, *dW1out, *dW2out, *db1out, *db2out;
    cudaMalloc(&dG, sh*4); cudaMalloc(&dH1, sh*4);
    cudaMalloc(&dXout, sx*4);
    cudaMalloc(&dW1out, D*DFF*4); cudaMalloc(&dW2out, DFF*D*4);
    cudaMalloc(&db1out, DFF*4); cudaMalloc(&db2out, D*4);
    cudaMemset(dW1out,0,D*DFF*4); cudaMemset(dW2out,0,DFF*D*4);
    cudaMemset(db1out,0,DFF*4); cudaMemset(db2out,0,D*4);

    // dG = dY @ W2^T
    mm_nt(ddY, dW2, dG, T, D, DFF);
    // dW2 = G^T @ dY
    mm_tn(g, ddY, dW2out, DFF, T, D);
    // db2 = column sum of dY
    {
        float *hdY2=(float*)malloc(sx*4);
        cudaMemcpy(hdY2,ddY,sx*4,cudaMemcpyDeviceToHost);
        float* db2h=(float*)calloc(D,4);
        for (int t=0;t<T;t++) for (int d=0;d<D;d++) db2h[d]+=hdY2[t*D+d];
        cudaMemcpy(db2out,db2h,D*4,cudaMemcpyHostToDevice);
        free(hdY2); free(db2h);
    }
    // dH1 = dG * gelu'(H1)
    gelu_bwd<<<(sh+255)/256,256>>>(dG, h1, dH1, sh);
    // dW1 = X^T @ dH1
    mm_tn(dX, dH1, dW1out, D, T, DFF);
    // db1 = column sum of dH1
    {
        float* hdH=(float*)malloc(sh*4);
        cudaMemcpy(hdH,dH1,sh*4,cudaMemcpyDeviceToHost);
        float* db1h=(float*)calloc(DFF,4);
        for (int t=0;t<T;t++) for (int d=0;d<DFF;d++) db1h[d]+=hdH[t*DFF+d];
        cudaMemcpy(db1out,db1h,DFF*4,cudaMemcpyHostToDevice);
        free(hdH); free(db1h);
    }
    // dX = dH1 @ W1^T
    mm_nt(dH1, dW1, dXout, T, DFF, D);

    // Save grads
    float *hdX=(float*)malloc(sx*4), *hdW1=(float*)malloc(D*DFF*4), *hdW2=(float*)malloc(DFF*D*4),
          *hdb1=(float*)malloc(DFF*4), *hdb2=(float*)malloc(D*4);
    cudaMemcpy(hdX,dXout,sx*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(hdW1,dW1out,D*DFF*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(hdW2,dW2out,DFF*D*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(hdb1,db1out,DFF*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(hdb2,db2out,D*4,cudaMemcpyDeviceToHost);
    sv("/tmp/mlp_dx.bin", hdX, sx);
    sv("/tmp/mlp_dw1.bin", hdW1, D*DFF);
    sv("/tmp/mlp_dw2.bin", hdW2, DFF*D);
    sv("/tmp/mlp_db1.bin", hdb1, DFF);
    sv("/tmp/mlp_db2.bin", hdb2, D);

    printf("MLP fwd+bwd done: T=%d D=%d DFF=%d\n", T, D, DFF);
    return 0;
}
