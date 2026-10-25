// runtime_v5.h — MMC Runtime Header
#ifndef RUNTIME_V5_H
#define RUNTIME_V5_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

// Memory
void* mmc_malloc(size_t size);
void* mmc_realloc(void* ptr, size_t size);
void mmc_free(void* ptr);

// File I/O
char* mmc_read_file(const char* filename);
int mmc_write_file(const char* filename, const char* content);
int mmc_write_binary(const char* filename, const void* data, size_t size);
void* mmc_read_binary(const char* filename, size_t* out_size);
int mmc_append_file(const char* filename, const char* content);

// I/O
void mmc_print(const char* str);
void mmc_println(const char* str);
void mmc_print_int(int val);
void mmc_print_float(double val);
void mmc_print_newline();
void mmc_flush();
char* mmc_read_line();

// Math
double mmc_exp(double x);
double mmc_sqrt(double x);
double mmc_tanh(double x);
double mmc_log(double x);
double mmc_pow(double x, double y);
double mmc_abs(double x);
double mmc_random();
double mmc_random_gaussian();
void mmc_random_seed(int seed);

// String
int mmc_string_length(const char* str);
char* mmc_string_concat(const char* a, const char* b);
int mmc_string_compare(const char* a, const char* b);
char* mmc_int_to_string(int val);
char* mmc_float_to_string(double val);
double mmc_string_to_float(const char* str);
int mmc_string_to_int(const char* str);
int mmc_string_contains(const char* haystack, const char* needle);
char* mmc_string_substring(const char* str, int start, int end);

// UTF-8 Unicode support (for Myanmar script)
int mmc_utf8_decode(const char* str, int pos);         // returns Unicode code point at byte position
char* mmc_utf8_encode(int codepoint);                    // codepoint -> UTF-8 string (malloc'd)
int mmc_utf8_char_len(const char* str, int pos);        // returns byte length of UTF-8 char at pos
int mmc_utf8_is_myanmar(int codepoint);                 // checks if codepoint is Myanmar script (U+1000-U+109F)
int mmc_utf8_is_myanmar_digit(int codepoint);           // checks if codepoint is Myanmar digit (U+1040-U+1049)
int mmc_utf8_is_alpha(int codepoint);                   // checks if codepoint is ASCII letter or _
int mmc_utf8_is_digit(int codepoint);                   // checks if codepoint is ASCII digit 0-9

// C string escaping (for MMC compiler codegen)
char* mmc_string_escape_c(const char* str);

// Timing
double mmc_time_now();
void mmc_timer_start();
double mmc_timer_elapsed();

// Progress
void mmc_progress_bar(int current, int total, int width);

// Matrix type (must be before matrix functions)
typedef struct {
    float* data;
    int rows;
    int cols;
} MMCMatrix;

// Dynamic Array type (for MMC lists)
typedef struct {
    void** data;
    int size;
    int capacity;
} MMCArray;

MMCArray* mmc_array_new();
void mmc_array_push(MMCArray* arr, void* item);
void* mmc_array_get(MMCArray* arr, int index);
void mmc_array_set(MMCArray* arr, int index, void* item);
int mmc_array_size(MMCArray* arr);
void mmc_array_free(MMCArray* arr);

MMCMatrix* mmc_matrix_new(int rows, int cols);
MMCMatrix* mmc_matrix_random(int rows, int cols, float std);
float mmc_matrix_get(MMCMatrix* mat, int r, int c);
void mmc_matrix_set(MMCMatrix* mat, int r, int c, float val);
MMCMatrix* mmc_matrix_mul(MMCMatrix* A, MMCMatrix* B);
MMCMatrix* mmc_matrix_transpose(MMCMatrix* A);
MMCMatrix* mmc_matrix_add(MMCMatrix* A, MMCMatrix* B);
MMCMatrix* mmc_matrix_sub(MMCMatrix* A, MMCMatrix* B);
MMCMatrix* mmc_matrix_scale(MMCMatrix* A, float s);
MMCMatrix* mmc_matrix_softmax(MMCMatrix* A);
MMCMatrix* mmc_matrix_gelu(MMCMatrix* A);
MMCMatrix* mmc_matrix_layernorm(MMCMatrix* A, MMCMatrix* gamma, MMCMatrix* beta);
float mmc_cross_entropy(MMCMatrix* logits, int* targets, int batch_size);

// Backprop primitives (added for NN training)
MMCMatrix* mmc_matrix_fill(MMCMatrix* A, float v);
MMCMatrix* mmc_matrix_ew_mul(MMCMatrix* A, MMCMatrix* B);
MMCMatrix* mmc_gelu_backward(MMCMatrix* pre, MMCMatrix* grad_out);
MMCMatrix* mmc_softmax_ce_grad(MMCMatrix* probs, int target);
void mmc_matrix_apply_update(MMCMatrix* W, MMCMatrix* grad_scaled);  // W -= grad_scaled (in-place)

// Save/load matrix to/from binary file
int mmc_matrix_save(MMCMatrix* mat, const char* filename);
MMCMatrix* mmc_matrix_load(const char* filename);
void mmc_matrix_free(MMCMatrix* mat);
int mmc_matrix_rows(MMCMatrix* mat);
int mmc_matrix_cols(MMCMatrix* mat);

// Generic value-to-string (C11 _Generic — type dispatch at compile time)
static inline char* mmc_identity_str(char* s) { return s; }
static inline char* mmc_identity_cstr(const char* s) { return (char*)s; }
#define mmc_to_string(x) _Generic((x), \
    char*: mmc_identity_str, \
    const char*: mmc_identity_cstr, \
    double: mmc_float_to_string, \
    float: mmc_float_to_string, \
    int: mmc_int_to_string \
)(x)


/* File-based memory — added 2026-10-12 */
void mmc_memory_save_to(const char* filename, const char* key, const char* value);
char* mmc_memory_load_from(const char* filename, const char* key);

/* Dataset fuzzy search — added 2026-10-12 */
char* mmc_dataset_search(const char* filename, const char* keyword);

/* File scan helpers — added 2026-10-12 */
int mmc_file_line_count(const char* filename);
char* mmc_file_get_line(const char* filename, int index);
char* mmc_int_to_str(int n);

/* Bigram prediction — added 2026-10-12 */
char* mmc_model_predict(const char* filename, const char* prefix);

/* System with stdin capture — added 2026-10-12 */
char* mmc_system_stdin(const char* cmd, const char* input);
/* Trigram predict — added 2026-10-12 */
char* mmc_trigram_predict(const char* filename, const char* w1, const char* w2);
/* HTTP server — added 2026-10-12 */
int mmc_http_serve(int port);
#endif

/* Added: Character code decoder for MMC Brain Inference */
int mmc_char_code(const char* str, int pos);

// ============================================
// Nyan Lin AI – Memory System
// ============================================
void mmc_memory_save(const char* key, const char* value);
char* mmc_memory_load(const char* key);
int mmc_random_int(int min, int max);

// ============================================
// JSON Reader for Nyan Lin AI
// ============================================
char* mmc_json_read(const char* filename);
int mmc_json_count(const char* filename);
char* mmc_json_get_instruction(const char* filename, int index);
char* mmc_json_get_output(const char* filename, int index);
char* mmc_string_trim(const char* str);
char* mmc_string_split(const char* str, const char* delim, int index);
int mmc_string_is_number(const char* str);
char* mmc_string_split_v2(const char* str, const char* delim, int index);
char* mmc_json_read_cached(const char* filename);
char* mmc_json_read_cached(const char* filename);
char* mmc_json_read_cached(const char* filename);
char* mmc_json_read_cached(const char* filename);

void mmc_xor_train();

void mmc_xor_train_and_save(const char* filename);
void mmc_xor_load(const char* filename);
int mmc_xor_predict(float x1, float x2);

void mmc_bigram_train(const char* text);
void mmc_bigram_generate(const char* start, int max_len);

void mmc_trigram_train(const char* text);
void mmc_trigram_generate(const char* seed, int max_len);

void mmc_trigram_train_file(const char* filename);

void mmc_trigram_train_file(const char* filename);

void mmc_fgram_train(const char* text);
void mmc_fgram_generate(const char* seed, int max_len);
void mmc_fgram_train_file(const char* filename);

void mmc_char_train(const char* text);
void mmc_char3_build(const char* text);
void mmc_char3_generate(const char* text, const char* seed, int max_chars);

void mmc_c5_train(const char* text);
void mmc_c5_generate(const char* text, const char* seed, int max_chars);
void mmc_c5_train_file(const char* filename);
void mmc_c5_generate_from_file(const char* seed, int max_chars);

void mmc_word_train(const char* text);
void mmc_word_generate(const char* seed_word, int max_words_out);
void mmc_word_train_file(const char* filename);

void mmc_tt_forward(const int* tokens, int T, float* out_logits);
void mmc_tt_attention(const int* tokens, int T);
