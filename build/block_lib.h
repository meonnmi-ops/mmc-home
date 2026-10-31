#ifndef BLOCK_LIB_H
#define BLOCK_LIB_H

int ln_fwd_row(void* xn, void* x, void* g, void* b, int D, void* mo, void* io);
int ln_bwd_row(void* dx, void* dg, void* db, void* dout, void* x, void* g, int D, int add);
int mm_fwd(void* out, void* X, void* W, void* b, int T, int J, int K);
int mm_bwd(void* dX, void* dW, void* db, void* dout, void* X, void* W, int T, int J, int K);
int gelu_bwd(void* dx, void* dout, void* x, int n, int add);
int block_forward(void* x2_out, void* X, void* Wqkv, void* bqkv, void* Wo, void* bo,
                  void* W1, void* b1, void* W2, void* b2,
                  void* ln1w, void* ln1b, void* ln2w, void* ln2b,
                  int T, int D, int DFF, int H);

int block_fwd_bwd(void* dx,
    void* dWqkv, void* dbqkv, void* dWo, void* dbo,
    void* dW1, void* db1, void* dW2, void* db2,
    void* dln1w, void* dln1b, void* dln2w, void* dln2b,
    void* X, void* Wqkv, void* bqkv, void* Wo, void* bo,
    void* W1, void* b1, void* W2, void* b2,
    void* ln1w, void* ln1b, void* ln2w, void* ln2b,
    void* dOut, int T, int D, int DFF, int H);

#endif
