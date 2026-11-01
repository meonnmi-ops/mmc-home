#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int mm_fwd(float *out, float *X, float *W, float *b, int T, int J, int K);
int mm_bwd(float *dX, float *dW, float *db, float *dout, float *X, float *W,
           int T, int J, int K, int add_dX);

typedef struct {
    const char *name;
    int T;
    int J;
    int K;
} Case;

static const Case cases[] = {
    {"qkv", 128, 768, 256},
    {"attn_proj", 128, 256, 256},
    {"fc1", 128, 1024, 256},
    {"fc2", 128, 256, 1024},
};

static uint32_t rng_state = 0x4e59414eu;

static float next_value(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return (float)(rng_state >> 8) / 8388608.0f - 1.0f;
}

static void fill(float *values, size_t n) {
    for (size_t i = 0; i < n; ++i) values[i] = next_value();
}

static void zero(float *values, size_t n) {
    memset(values, 0, n * sizeof(*values));
}

static double elapsed_ms(struct timespec start, struct timespec end) {
    return ((double)(end.tv_sec - start.tv_sec) * 1000.0) +
           ((double)(end.tv_nsec - start.tv_nsec) / 1000000.0);
}

static int write_values(FILE *file, const float *values, size_t n) {
    return fwrite(values, sizeof(*values), n, file) == n;
}

static int run_case(const Case *c, FILE *out) {
    const size_t nx = (size_t)c->T * (size_t)c->K;
    const size_t nw = (size_t)c->J * (size_t)c->K;
    const size_t ny = (size_t)c->T * (size_t)c->J;
    const int repeats = 5;
    float *X = malloc(nx * sizeof(float));
    float *W = malloc(nw * sizeof(float));
    float *b = malloc((size_t)c->J * sizeof(float));
    float *Y = malloc(ny * sizeof(float));
    float *dout = malloc(ny * sizeof(float));
    float *dX = malloc(nx * sizeof(float));
    float *dW = malloc(nw * sizeof(float));
    float *db = malloc((size_t)c->J * sizeof(float));

    if (!X || !W || !b || !Y || !dout || !dX || !dW || !db) {
        fprintf(stderr, "allocation failed for %s\n", c->name);
        free(X); free(W); free(b); free(Y); free(dout);
        free(dX); free(dW); free(db);
        return 0;
    }

    fill(X, nx);
    fill(W, nw);
    fill(b, (size_t)c->J);
    fill(dout, ny);

    for (int i = 0; i < 2; ++i) {
        if (mm_fwd(Y, X, W, b, c->T, c->J, c->K) != 0) return 0;
    }
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < repeats; ++i) {
        if (mm_fwd(Y, X, W, b, c->T, c->J, c->K) != 0) return 0;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    const double fwd_ms = elapsed_ms(start, end) / repeats;

    zero(dX, nx); zero(dW, nw); zero(db, (size_t)c->J);
    if (mm_bwd(dX, dW, db, dout, X, W, c->T, c->J, c->K, 0) != 0) return 0;
    double bwd_total_ms = 0.0;
    for (int i = 0; i < repeats; ++i) {
        zero(dX, nx); zero(dW, nw); zero(db, (size_t)c->J);
        clock_gettime(CLOCK_MONOTONIC, &start);
        if (mm_bwd(dX, dW, db, dout, X, W, c->T, c->J, c->K, 0) != 0) return 0;
        clock_gettime(CLOCK_MONOTONIC, &end);
        bwd_total_ms += elapsed_ms(start, end);
    }
    const double bwd_ms = bwd_total_ms / repeats;

    /* The output stream is fixed-order float32 data for the Python comparator. */
    const int ok = write_values(out, Y, ny) && write_values(out, dX, nx) &&
                   write_values(out, dW, nw) && write_values(out, db, (size_t)c->J);
    printf("%-10s fwd=%8.3f ms  bwd=%8.3f ms\n", c->name, fwd_ms, bwd_ms);

    free(X); free(W); free(b); free(Y); free(dout);
    free(dX); free(dW); free(db);
    return ok;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s OUTPUT.bin\n", argv[0]);
        return 2;
    }
    FILE *out = fopen(argv[1], "wb");
    if (!out) {
        perror(argv[1]);
        return 2;
    }
    const size_t count = sizeof(cases) / sizeof(cases[0]);
    for (size_t i = 0; i < count; ++i) {
        if (!run_case(&cases[i], out)) {
            fprintf(stderr, "benchmark failed for %s\n", cases[i].name);
            fclose(out);
            return 1;
        }
    }
    if (fclose(out) != 0) {
        perror("closing output");
        return 1;
    }
    return 0;
}
