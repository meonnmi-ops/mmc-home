#ifndef BLOCK_OPS_H
#define BLOCK_OPS_H

#ifdef __cplusplus
extern "C" {
#endif

void ln_fwd_dev(const float* d_x, const float* d_g, const float* d_b,
                float* d_y, float* d_mean, float* d_rstd,
                int M, int D, float eps);

void ln_bwd_dev(const float* d_dy, const float* d_x, const float* d_g,
                const float* d_mean, const float* d_rstd,
                float* d_dx, float* d_dgamma, float* d_dbeta,
                int M, int D);

void gelu_fwd_dev(const float* d_x, float* d_y, int n);
void gelu_bwd_dev(const float* d_dy, const float* d_x, float* d_dx, int n);
void add_dev(float* d_a, const float* d_b, int n);
void add_bias_dev(float* d_y, const float* d_b, int T, int D);


void adamw_dev(float* d_w, const float* d_g, float* d_m, float* d_v,
               float lr, float b1, float b2, float eps, float wd,
               int step, int n);

#ifdef __cplusplus
}
#endif
#endif
