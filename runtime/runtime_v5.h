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
    void** items;
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
void mmc_matrix_free(MMCMatrix* mat);
int mmc_matrix_rows(MMCMatrix* mat);
int mmc_matrix_cols(MMCMatrix* mat);

#endif
