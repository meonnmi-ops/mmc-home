#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "runtime_v5.h"

/* Only functions NOT in runtime_v5.c */
void* mmc_malloc(size_t n) { void* p = malloc(n); if (p) memset(p, 0, n); return p; }
void mmc_free(void* p) { if (p) free(p); }
int mmc_string_compare(const char* a, const char* b) { if (!a && !b) return 0; if (!a) return -1; if (!b) return 1; return strcmp(a, b); }
char* mmc_string_substring(const char* s, int a2, int b2) { if (!s) return NULL; int L = strlen(s); if (a2 < 0) a2 = 0; if (b2 > L) b2 = L; if (b2 <= a2) { char* e = (char*)malloc(1); e[0] = 0; return e; } int n2 = b2 - a2; char* r = (char*)malloc(n2 + 1); memcpy(r, s + a2, n2); r[n2] = 0; return r; }
char* mmc_int_to_string(int n) { char* r = (char*)malloc(32); sprintf(r, "%d", n); return r; }
double mmc_string_to_float(const char* s) { return s ? atof(s) : 0.0; }
void mmc_print_int(int n) { printf("%d", n); }
void mmc_print_float(double d) { printf("%g", d); }
void mmc_print_newline() { printf("\n"); }
char* mmc_read_file(const char* filename) { FILE* f = fopen(filename, "rb"); if (!f) return NULL; fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET); char* buf = (char*)malloc(len + 1); if (buf) { size_t rd = fread(buf, 1, len, f); buf[rd] = 0; } fclose(f); return buf; }
int mmc_write_file(const char* fn, const char* c) { FILE* f = fopen(fn, "wb"); if (!f) return -1; fputs(c ? c : "", f); fclose(f); return 0; }

/* mmc_array_size and mmc_array_get (not in runtime_v5.c) */
int mmc_array_size(MMCArray* arr) {
    return arr ? arr->size : 0;
}

void* mmc_array_get(MMCArray* arr, int idx) {
    if (!arr || idx < 0 || idx >= arr->size) return NULL;
    return arr->data[idx];
}
