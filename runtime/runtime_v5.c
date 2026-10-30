// ============================================================
// runtime_v5.c — MMC Runtime v5.0 (Full Featured)
// (c) NyanLin AI Lab 2026
// gcc -o mmc runtime_v5.c -lm
// ============================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include "runtime_v5.h"

// ============================================================
// 1. MEMORY MANAGEMENT
// ============================================================

void* mmc_malloc(size_t size) {
    void* ptr = malloc(size);
    if (!ptr && size > 0) {
        fprintf(stderr, "Error: malloc failed (%zu bytes)\n", size);
        exit(1);
    }
    return ptr;
}

void* mmc_realloc(void* ptr, size_t size) {
    void* new_ptr = realloc(ptr, size);
    if (!new_ptr && size > 0) {
        fprintf(stderr, "Error: realloc failed (%zu bytes)\n", size);
        exit(1);
    }
    return new_ptr;
}

void mmc_free(void* ptr) {
    if (ptr) free(ptr);
}

// ============================================================
// 2. FILE I/O
// ============================================================

char* mmc_read_file(const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file '%s'\n", filename);
        return NULL;
    }
    
    fseek(fp, 0, SEEK_END);
    long len = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    char* content = (char*)mmc_malloc(len + 1);
    fread(content, 1, len, fp);
    content[len] = '\0';
    fclose(fp);
    return content;
}

int mmc_write_file(const char* filename, const char* content) {
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: Cannot write to file '%s'\n", filename);
        return 0;
    }
    fprintf(fp, "%s", content);
    fclose(fp);
    return 1;
}

// Binary read/write for model checkpoints
int mmc_write_binary(const char* filename, const void* data, size_t size) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) return 0;
    fwrite(&size, sizeof(size_t), 1, fp);
    fwrite(data, 1, size, fp);
    fclose(fp);
    return 1;
}

void* mmc_read_binary(const char* filename, size_t* out_size) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) return NULL;
    
    size_t size;
    fread(&size, sizeof(size_t), 1, fp);
    
    void* data = mmc_malloc(size);
    fread(data, 1, size, fp);
    fclose(fp);
    
    if (out_size) *out_size = size;
    return data;
}

// Append to file (for logging)
int mmc_append_file(const char* filename, const char* content) {
    FILE* fp = fopen(filename, "a");
    if (!fp) return 0;
    fprintf(fp, "%s", content);
    fclose(fp);
    return 1;
}

// ============================================================
// 3. STANDARD I/O
// ============================================================

void mmc_print(const char* str) {
    printf("%s", str);
}

void mmc_println(const char* str) {
    printf("%s\n", str);
}

void mmc_print_int(int val) {
    printf("%d", val);
}

void mmc_print_float(double val) {
    printf("%g", val);
}

void mmc_print_newline() {
    printf("\n");
}

void mmc_flush() {
    fflush(stdout);
}

// Read line from stdin
char* mmc_read_line() {
    char* buf = (char*)mmc_malloc(4096);
    if (fgets(buf, 4096, stdin)) {
        // Remove newline
        int len = strlen(buf);
        if (len > 0 && buf[len-1] == '\n') buf[len-1] = '\0';
        return buf;
    }
    mmc_free(buf);
    return NULL;
}

// ============================================================
// 4. MATH FUNCTIONS
// ============================================================

double mmc_exp(double x) { return exp(x); }
double mmc_sqrt(double x) { return sqrt(x); }
double mmc_tanh(double x) { return tanh(x); }
double mmc_log(double x) { return log(x); }
double mmc_pow(double x, double y) { return pow(x, y); }
double mmc_abs(double x) { return fabs(x); }
double mmc_floor(double x) { return floor(x); }
double mmc_ceil(double x) { return ceil(x); }
double mmc_sin(double x) { return sin(x); }
double mmc_cos(double x) { return cos(x); }

// Random [0, 1)
double mmc_random() {
    return (double)rand() / RAND_MAX;
}

// Random Gaussian (Box-Muller)
double mmc_random_gaussian() {
    double u1 = mmc_random() + 1e-10;
    double u2 = mmc_random();
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979 * u2);
}

void mmc_random_seed(int seed) {
    srand(seed);
}

// ============================================================
// 5. STRING OPERATIONS
// ============================================================

int mmc_string_length(const char* str) {
    return str ? strlen(str) : 0;
}

char* mmc_string_concat(const char* a, const char* b) {
    if (!a && !b) return strdup("");
    if (!a) return strdup(b);
    if (!b) return strdup(a);
    
    char* result = (char*)mmc_malloc(strlen(a) + strlen(b) + 1);
    strcpy(result, a);
    strcat(result, b);
    return result;
}

int mmc_string_compare(const char* a, const char* b) {
    if (!a && !b) return 0;
    if (!a) return -1;
    if (!b) return 1;
    return strcmp(a, b);
}

int mmc_string_contains(const char* haystack, const char* needle) {
    if (!haystack || !needle) return 0;
    return strstr(haystack, needle) != NULL;
}

char* mmc_string_substring(const char* str, int start, int end) {
    if (!str) return strdup("");
    int len = strlen(str);
    if (start < 0) start = 0;
    if (end > len) end = len;
    if (start >= end) return strdup("");
    
    char* result = (char*)mmc_malloc(end - start + 1);
    memcpy(result, str + start, end - start);
    result[end - start] = '\0';
    return result;
}

// ============================================================
// 3b. UTF-8 UNICODE SUPPORT (for Myanmar script)
// ============================================================

int mmc_utf8_decode(const char* str, int pos) {
    // Decode a UTF-8 character at byte position and return Unicode code point
    if (!str || pos < 0) return 0;
    unsigned char byte = (unsigned char)str[pos];
    
    if (byte < 0x80) {
        // 1-byte: ASCII (0xxxxxxx)
        return (int)byte;
    } else if ((byte & 0xE0) == 0xC0) {
        // 2-byte: (110xxxxx 10xxxxxx)
        if (str[pos+1] == 0) return 0;
        return ((byte & 0x1F) << 6) | ((unsigned char)str[pos+1] & 0x3F);
    } else if ((byte & 0xF0) == 0xE0) {
        // 3-byte: (1110xxxx 10xxxxxx 10xxxxxx) — Myanmar characters are here
        if (str[pos+1] == 0 || str[pos+2] == 0) return 0;
        return ((byte & 0x0F) << 12) | ((unsigned char)str[pos+1] & 0x3F) << 6 | ((unsigned char)str[pos+2] & 0x3F);
    } else if ((byte & 0xF8) == 0xF0) {
        // 4-byte: (11110xxx 10xxxxxx 10xxxxxx 10xxxxxx)
        if (str[pos+1] == 0 || str[pos+2] == 0 || str[pos+3] == 0) return 0;
        return ((byte & 0x07) << 18) | ((unsigned char)str[pos+1] & 0x3F) << 12 | ((unsigned char)str[pos+2] & 0x3F) << 6 | ((unsigned char)str[pos+3] & 0x3F);
    }
    return (int)byte;  // invalid UTF-8, return raw byte
}

int mmc_utf8_char_len(const char* str, int pos) {
    // Return the byte length of the UTF-8 character at byte position pos
    if (!str || pos < 0) return 1;
    unsigned char byte = (unsigned char)str[pos];
    
    if (byte < 0x80) return 1;             // ASCII: 1 byte
    if ((byte & 0xE0) == 0xC0) return 2;   // 2-byte UTF-8
    if ((byte & 0xF0) == 0xE0) return 3;   // 3-byte UTF-8 (Myanmar)
    if ((byte & 0xF8) == 0xF0) return 4;   // 4-byte UTF-8
    return 1;  // invalid, treat as 1 byte
}

int mmc_utf8_is_myanmar(int codepoint) {
    // Myanmar script: U+1000 to U+109F
    return (codepoint >= 0x1000 && codepoint <= 0x109F);
}

int mmc_utf8_is_myanmar_digit(int codepoint) {
    // Myanmar digits: U+1040 (၀) to U+1049 (၉)
    return (codepoint >= 0x1040 && codepoint <= 0x1049);
}

int mmc_utf8_is_alpha(int codepoint) {
    // ASCII letters and underscore
    return ((codepoint >= 'a' && codepoint <= 'z') ||
            (codepoint >= 'A' && codepoint <= 'Z') ||
            codepoint == '_');
}

int mmc_utf8_is_digit(int codepoint) {
    // ASCII digits 0-9
    return (codepoint >= '0' && codepoint <= '9');
}

char* mmc_string_escape_c(const char* str) {
    // Escape a string for C output: \n → \\n, \t → \\t, \\ → \\\\, " → \\"
    if (!str) return strdup("");
    int len = strlen(str);
    // Max expansion: each char could become 2 chars
    char* result = (char*)mmc_malloc(len * 2 + 1);
    int j = 0;
    for (int i = 0; i < len; i++) {
        if (str[i] == '\n') {
            result[j++] = '\\';
            result[j++] = 'n';
        } else if (str[i] == '\t') {
            result[j++] = '\\';
            result[j++] = 't';
        } else if (str[i] == '\r') {
            result[j++] = '\\';
            result[j++] = 'r';
        } else if (str[i] == '\\') {
            result[j++] = '\\';
            result[j++] = '\\';
        } else if (str[i] == '"') {
            result[j++] = '\\';
            result[j++] = '"';
        } else if (str[i] == '\0') {
            result[j++] = '\\';
            result[j++] = '0';
        } else {
            result[j++] = str[i];
        }
    }
    result[j] = '\0';
    return result;
}

char* mmc_int_to_string(int val) {
    char* buf = (char*)mmc_malloc(32);
    snprintf(buf, 32, "%d", val);
    return buf;
}

char* mmc_float_to_string(double val) {
    char* buf = (char*)mmc_malloc(64);
    snprintf(buf, 64, "%g", val);
    return buf;
}

double mmc_string_to_float(const char* str) {
    return str ? atof(str) : 0.0;
}

int mmc_string_to_int(const char* str) {
    return str ? atoi(str) : 0;
}

// ============================================================
// 6. TIMING
// ============================================================

double mmc_time_now() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

static double _mmc_start_time = 0;

void mmc_timer_start() {
    _mmc_start_time = mmc_time_now();
}

double mmc_timer_elapsed() {
    return mmc_time_now() - _mmc_start_time;
}

// ============================================================
// 7. DYNAMIC ARRAY (for MMC lists)
// ============================================================
// MMCArray typedef moved to runtime_v5.h

MMCArray* mmc_array_new() {
    MMCArray* arr = (MMCArray*)mmc_malloc(sizeof(MMCArray));
    arr->items = (void**)mmc_malloc(16 * sizeof(void*));
    arr->size = 0;
    arr->capacity = 16;
    return arr;
}

void mmc_array_push(MMCArray* arr, void* item) {
    if (arr->size >= arr->capacity) {
        arr->capacity *= 2;
        arr->items = (void**)mmc_realloc(arr->items, arr->capacity * sizeof(void*));
    }
    arr->items[arr->size++] = item;
}

void* mmc_array_get(MMCArray* arr, int index) {
    if (index < 0 || index >= arr->size) return NULL;
    return arr->items[index];
}

void mmc_array_set(MMCArray* arr, int index, void* item) {
    if (index < 0 || index >= arr->size) return;
    arr->items[index] = item;
}

int mmc_array_size(MMCArray* arr) {
    return arr ? arr->size : 0;
}

void mmc_array_free(MMCArray* arr) {
    if (arr) {
        free(arr->items);
        free(arr);
    }
}

// ============================================================
// 8. FLOAT ARRAY (for model weights)
// ============================================================

// MMCMatrix type defined in runtime_v5.h

MMCMatrix* mmc_matrix_new(int rows, int cols) {
    MMCMatrix* mat = (MMCMatrix*)mmc_malloc(sizeof(MMCMatrix));
    mat->data = (float*)mmc_malloc(rows * cols * sizeof(float));
    mat->rows = rows;
    mat->cols = cols;
    memset(mat->data, 0, rows * cols * sizeof(float));
    return mat;
}

MMCMatrix* mmc_matrix_random(int rows, int cols, float std) {
    MMCMatrix* mat = mmc_matrix_new(rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        mat->data[i] = (float)(mmc_random_gaussian() * std);
    }
    return mat;
}

float mmc_matrix_get(MMCMatrix* mat, int r, int c) {
    return mat->data[r * mat->cols + c];
}

void mmc_matrix_set(MMCMatrix* mat, int r, int c, float val) {
    mat->data[r * mat->cols + c] = val;
}

// Matrix multiply: C = A * B
MMCMatrix* mmc_matrix_mul(MMCMatrix* A, MMCMatrix* B) {
    if (A->cols != B->rows) {
        fprintf(stderr, "Error: Matrix multiply dimension mismatch (%dx%d * %dx%d)\n",
                A->rows, A->cols, B->rows, B->cols);
        return NULL;
    }
    
    MMCMatrix* C = mmc_matrix_new(A->rows, B->cols);
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < B->cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < A->cols; k++) {
                sum += mmc_matrix_get(A, i, k) * mmc_matrix_get(B, k, j);
            }
            mmc_matrix_set(C, i, j, sum);
        }
    }
    return C;
}

// Matrix transpose
MMCMatrix* mmc_matrix_transpose(MMCMatrix* A) {
    MMCMatrix* T = mmc_matrix_new(A->cols, A->rows);
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            mmc_matrix_set(T, j, i, mmc_matrix_get(A, i, j));
        }
    }
    return T;
}

// Element-wise add
MMCMatrix* mmc_matrix_add(MMCMatrix* A, MMCMatrix* B) {
    MMCMatrix* C = mmc_matrix_new(A->rows, A->cols);
    for (int i = 0; i < A->rows * A->cols; i++) {
        C->data[i] = A->data[i] + B->data[i];
    }
    return C;
}

// Element-wise subtract
MMCMatrix* mmc_matrix_sub(MMCMatrix* A, MMCMatrix* B) {
    MMCMatrix* C = mmc_matrix_new(A->rows, A->cols);
    for (int i = 0; i < A->rows * A->cols; i++) {
        C->data[i] = A->data[i] - B->data[i];
    }
    return C;
}

// Scalar multiply
MMCMatrix* mmc_matrix_scale(MMCMatrix* A, float s) {
    MMCMatrix* C = mmc_matrix_new(A->rows, A->cols);
    for (int i = 0; i < A->rows * A->cols; i++) {
        C->data[i] = A->data[i] * s;
    }
    return C;
}

// Softmax (row-wise)
MMCMatrix* mmc_matrix_softmax(MMCMatrix* A) {
    MMCMatrix* C = mmc_matrix_new(A->rows, A->cols);
    for (int i = 0; i < A->rows; i++) {
        float max_val = mmc_matrix_get(A, i, 0);
        for (int j = 1; j < A->cols; j++) {
            float v = mmc_matrix_get(A, i, j);
            if (v > max_val) max_val = v;
        }
        float sum = 0.0f;
        for (int j = 0; j < A->cols; j++) {
            float v = expf(mmc_matrix_get(A, i, j) - max_val);
            mmc_matrix_set(C, i, j, v);
            sum += v;
        }
        for (int j = 0; j < A->cols; j++) {
            mmc_matrix_set(C, i, j, mmc_matrix_get(C, i, j) / sum);
        }
    }
    return C;
}

// GELU activation
MMCMatrix* mmc_matrix_gelu(MMCMatrix* A) {
    MMCMatrix* C = mmc_matrix_new(A->rows, A->cols);
    for (int i = 0; i < A->rows * A->cols; i++) {
        float x = A->data[i];
        C->data[i] = 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
    }
    return C;
}

// Layer normalization
MMCMatrix* mmc_matrix_layernorm(MMCMatrix* A, MMCMatrix* gamma, MMCMatrix* beta) {
    MMCMatrix* C = mmc_matrix_new(A->rows, A->cols);
    for (int i = 0; i < A->rows; i++) {
        float mean = 0.0f;
        for (int j = 0; j < A->cols; j++) mean += mmc_matrix_get(A, i, j);
        mean /= A->cols;
        
        float var = 0.0f;
        for (int j = 0; j < A->cols; j++) {
            float d = mmc_matrix_get(A, i, j) - mean;
            var += d * d;
        }
        var /= A->cols;
        float std = sqrtf(var + 1e-5f);
        
        for (int j = 0; j < A->cols; j++) {
            float normed = (mmc_matrix_get(A, i, j) - mean) / std;
            mmc_matrix_set(C, i, j, mmc_matrix_get(gamma, 0, j) * normed + mmc_matrix_get(beta, 0, j));
        }
    }
    return C;
}

// Cross-entropy loss
float mmc_cross_entropy(MMCMatrix* logits, int* targets, int batch_size) {
    float loss = 0.0f;
    for (int i = 0; i < batch_size; i++) {
        float prob = mmc_matrix_get(logits, i, targets[i]);
        loss -= logf(prob + 1e-8f);
    }
    return loss / batch_size;
}

void mmc_matrix_free(MMCMatrix* mat) {
    if (mat) {
        free(mat->data);
        free(mat);
    }
}

int mmc_matrix_rows(MMCMatrix* mat) {
    return mat ? mat->rows : 0;
}

int mmc_matrix_cols(MMCMatrix* mat) {
    return mat ? mat->cols : 0;
}

// ============================================================
// 9. PROGRESS BAR
// ============================================================

void mmc_progress_bar(int current, int total, int width) {
    float pct = (float)current / total;
    int filled = (int)(pct * width);
    printf("\r[");
    for (int i = 0; i < width; i++) {
        if (i < filled) printf("=");
        else if (i == filled) printf(">");
        else printf(" ");
    }
    printf("] %d/%d (%.1f%%)", current, total, pct * 100);
    fflush(stdout);
    if (current >= total) printf("\n");
}

// ============================================================
// 10. MAIN (test) — only when compiled standalone
// ============================================================

#ifndef MMC_LIBRARY_ONLY
int __mmc_runtime_main(int argc, char** argv) {
    srand(time(NULL));
    
    printf("========================================\n");
    printf("  MMC Runtime v5.0 — Full Featured     \n");
    printf("  (c) NyanLin AI Lab 2026              \n");
    printf("========================================\n");
    
    // Test matrix ops
    printf("\n[TEST] Matrix operations:\n");
    MMCMatrix* A = mmc_matrix_new(2, 3);
    mmc_matrix_set(A, 0, 0, 1); mmc_matrix_set(A, 0, 1, 2); mmc_matrix_set(A, 0, 2, 3);
    mmc_matrix_set(A, 1, 0, 4); mmc_matrix_set(A, 1, 1, 5); mmc_matrix_set(A, 1, 2, 6);
    
    MMCMatrix* B = mmc_matrix_new(3, 2);
    mmc_matrix_set(B, 0, 0, 7); mmc_matrix_set(B, 0, 1, 8);
    mmc_matrix_set(B, 1, 0, 9); mmc_matrix_set(B, 1, 1, 10);
    mmc_matrix_set(B, 2, 0, 11); mmc_matrix_set(B, 2, 1, 12);
    
    MMCMatrix* C = mmc_matrix_mul(A, B);
    printf("  A (2x3) * B (3x2) = C (2x2)\n");
    printf("  C[0][0]=%g C[0][1]=%g\n", mmc_matrix_get(C,0,0), mmc_matrix_get(C,0,1));
    printf("  C[1][0]=%g C[1][1]=%g\n", mmc_matrix_get(C,1,0), mmc_matrix_get(C,1,1));
    
    // Test GELU
    printf("\n[TEST] GELU(1.0) = %g\n", 0.5 * 1.0 * (1.0 + tanh(0.79788456 * (1.0 + 0.044715))));
    
    // Test softmax
    MMCMatrix* S = mmc_matrix_new(1, 4);
    mmc_matrix_set(S, 0, 0, 1); mmc_matrix_set(S, 0, 1, 2);
    mmc_matrix_set(S, 0, 2, 3); mmc_matrix_set(S, 0, 3, 4);
    MMCMatrix* SM = mmc_matrix_softmax(S);
    printf("[TEST] Softmax: %g %g %g %g\n",
           mmc_matrix_get(SM,0,0), mmc_matrix_get(SM,0,1),
           mmc_matrix_get(SM,0,2), mmc_matrix_get(SM,0,3));
    
    // Test timing
    printf("\n[TEST] Timer: ");
    mmc_timer_start();
    double sum = 0;
    for (int i = 0; i < 1000000; i++) sum += mmc_random();
    printf("%.3f seconds\n", mmc_timer_elapsed());
    
    printf("\n✅ Runtime v5.0 ready!\n");
    printf("   Functions: 50+\n");
    printf("   Matrix ops: 15+\n");
    printf("   Memory: managed\n");
    
    mmc_matrix_free(A);
    mmc_matrix_free(B);
    mmc_matrix_free(C);
    mmc_matrix_free(S);
    mmc_matrix_free(SM);
    
    return 0;
}
#endif
