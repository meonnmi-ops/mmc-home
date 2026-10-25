/* runtime_glue.h - prototypes for the glue helpers provided by
   runtime_glue.c. Generated self-hosted code (v4+) includes this after
   runtime_v5.h so calls to these helpers compile cleanly. */
#ifndef RUNTIME_GLUE_H
#define RUNTIME_GLUE_H

#include <stddef.h>

void* mmc_malloc(size_t n);
void mmc_free(void* p);
int mmc_string_compare(const char* a, const char* b);
char* mmc_string_concat(const char* a, const char* b);
char* mmc_string_substring(const char* s, int a, int b);
char* mmc_int_to_string(int n);
double mmc_string_to_float(const char* s);
void mmc_print_int(int n);
void mmc_print_float(double d);
void mmc_print_newline();
char* mmc_read_file(const char* filename);
int mmc_write_file(const char* fn, const char* c);
int mmc_array_size(MMCArray* arr);
void* mmc_array_get(MMCArray* arr, int idx);

#endif /* RUNTIME_GLUE_H */
