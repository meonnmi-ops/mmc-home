#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>

typedef struct {
    void** data;
    int size;
    int capacity;
} MMCArray;

MMCArray* mmc_array_new() {
    MMCArray* arr = (MMCArray*)malloc(sizeof(MMCArray));
    arr->size = 0;
    arr->capacity = 4;
    arr->data = malloc(sizeof(void*) * arr->capacity);
    return arr;
}

void mmc_array_push(MMCArray* arr, void* item) {
    if (arr->size >= arr->capacity) {
        arr->capacity *= 2;
        arr->data = realloc(arr->data, sizeof(void*) * arr->capacity);
    }
    arr->data[arr->size++] = item;
}

void mmc_println(const char* format, ...) {
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    printf("\n");
}

char* mmc_read_line() {
    char* line = malloc(256);
    if (fgets(line, 256, stdin)) {
        line[strcspn(line, "\n")] = 0;
        return line;
    }
    return "";
}

int mmc_string_length(const char* str) {
    if (!str) return 0;
    int len = 0;
    while (str[len]) len++;
    return len;
}

int mmc_string_contains(const char* str, const char* substr) {
    if (!str || !substr) return 0;
    return strstr(str, substr) ? 1 : 0;
}

int mmc_char_code(const char* str, int idx) {
    if (!str || idx < 0 || idx >= strlen(str)) return 0;
    // Multi-byte UTF-8 Handling for Myanmar Characters
    unsigned char c1 = str[idx];
    unsigned char c2 = str[idx+1];
    unsigned char c3 = str[idx+2];
    if (c1 == 0xE1 && c2 == 0x80) return 4096 + (c3 - 0x80);
    return c1;
}

char* mmc_utf8_encode(int code_point) {
    char* res = (char*)malloc(4 * sizeof(char));
    if (code_point >= 4096 && code_point <= 4128) {
        int offset = code_point - 4096;
        res[0] = 0xE1;
        res[1] = 0x80;
        res[2] = 0x80 + offset;
        res[3] = '\0';
    } else {
        res[0] = '?';
        res[1] = '\0';
    }
    return res;
}

// ============================================


// ============================================


// ============================================
// Nyan Lin AI – Memory System
// ============================================

void mmc_memory_save(const char* key, const char* value) {
    /* Read existing, update key, rewrite — overwrites in place */
    #define MEM_MAX_LINES 512
    static char lines[MEM_MAX_LINES][512];
    int n = 0;

    FILE* f = fopen("nyanlin_memory.txt", "r");
    if (f) {
        while (n < MEM_MAX_LINES && fgets(lines[n], 512, f)) n++;
        fclose(f);
    }

    /* Build "key = " prefix for exact match */
    char prefix[280];
    snprintf(prefix, sizeof(prefix), "%s = ", key);

    int replaced = 0;
    for (int i = 0; i < n; i++) {
        if (strncmp(lines[i], prefix, strlen(prefix)) == 0) {
            snprintf(lines[i], 512, "%s = %s\n", key, value);
            replaced = 1;
            break;
        }
    }

    /* Not found — append at end */
    if (!replaced && n < MEM_MAX_LINES) {
        snprintf(lines[n], 512, "%s = %s\n", key, value);
        n++;
    }

    /* Rewrite file */
    f = fopen("nyanlin_memory.txt", "w");
    if (f) {
        for (int i = 0; i < n; i++) fputs(lines[i], f);
        fclose(f);
    }
}

char* mmc_memory_load(const char* key) {
    static char result[256];
    result[0] = 0;

    FILE* f = fopen("nyanlin_memory.txt", "r");
    if (!f) return result;

    /* Exact prefix: "key = " */
    char prefix[280];
    snprintf(prefix, sizeof(prefix), "%s = ", key);
    size_t plen = strlen(prefix);

    char line[512];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, prefix, plen) == 0) {
            char* val = line + plen;
            size_t vlen = strlen(val);
            while (vlen > 0 && (val[vlen-1] == '\n' || val[vlen-1] == '\r'))
                val[--vlen] = 0;
            strncpy(result, val, 255);
            result[255] = 0;
            /* Don't return yet — keep scanning for LAST match */
        }
    }
    fclose(f);
    return result;
}

int mmc_random_int(int min, int max) {
    return min + (rand() % (max - min + 1));
}

// ============================================
// JSON Reader for Nyan Lin AI
// ============================================
#include <ctype.h>

char* mmc_json_read(const char* filename) {
    FILE* f = fopen(filename, "r");
    if (!f) return "";
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buffer = (char*)malloc(size + 1);
    fread(buffer, 1, size, f);
    buffer[size] = '\0';
    fclose(f);
    return buffer;
}

int mmc_json_count(const char* filename) {
    char* json = mmc_json_read(filename);
    if (!json || strlen(json) == 0) return 0;
    int count = 0;
    char* ptr = json;
    while ((ptr = strstr(ptr, "\"instruction\"")) != NULL) {
        count++;
        ptr++;
    }
    free(json);
    return count;
}

char* mmc_json_get_field(const char* filename, int index, const char* field) {
    char* json = mmc_json_read(filename);
    if (!json) return "";
    
    static char result[2048];
    char search[64];
    snprintf(search, sizeof(search), "\"%s\"", field);
    
    char* ptr = json;
    int count = 0;
    
    while ((ptr = strstr(ptr, search)) != NULL) {
        if (count == index) {
            // Skip to colon and opening quote
            ptr = strchr(ptr, ':');
            if (!ptr) break;
            ptr = strchr(ptr, '"');
            if (!ptr) break;
            ptr++;
            
            // Find closing quote (handle escaped quotes)
            char* start = ptr;
            int len = 0;
            while (*ptr && !(*ptr == '"' && *(ptr-1) != '\\')) {
                ptr++;
                len++;
            }
            
            strncpy(result, start, len);
            result[len] = '\0';
            free(json);
            return result;
        }
        count++;
        ptr++;
    }
    
    free(json);
    return "";
}

char* mmc_json_get_instruction(const char* filename, int index) {
    return mmc_json_get_field(filename, index, "instruction");
}

char* mmc_json_get_output(const char* filename, int index) {
    return mmc_json_get_field(filename, index, "output");
}

// ============================================
// JSON Reader for Nyan Lin AI
// ============================================
#include <ctype.h>







/* ═══════════════════════════════════════════════════
   Trigram v2 — UTF-8 safe (256 byte table)
   ═══════════════════════════════════════════════════ */
#define TRI_N 256
static int g_trigram[TRI_N][TRI_N][TRI_N];

void mmc_trigram_train(const char* text) {
    int i, j, k;
    for (i = 0; i < TRI_N; i++)
        for (j = 0; j < TRI_N; j++)
            for (k = 0; k < TRI_N; k++)
                g_trigram[i][j][k] = 0;
    
    int len = strlen(text);
    for (i = 0; i < len - 2; i++) {
        unsigned char c1 = (unsigned char)text[i];
        unsigned char c2 = (unsigned char)text[i+1];
        unsigned char c3 = (unsigned char)text[i+2];
        g_trigram[c1][c2][c3]++;
    }
    printf("[Trigram trained on %d bytes]\n", len);
}

void mmc_trigram_generate(const char* seed, int max_len) {
    srand(42);
    int slen = strlen(seed);
    if (slen < 2) { printf("[seed need 2+ bytes]\n"); return; }
    
    unsigned char c1 = (unsigned char)seed[slen-2];
    unsigned char c2 = (unsigned char)seed[slen-1];
    
    printf("[Generated] %s", seed);
    
    int i, j;
    for (i = 0; i < max_len; i++) {
        int total = 0;
        for (j = 0; j < TRI_N; j++) total += g_trigram[c1][c2][j];
        if (total == 0) break;
        int r = rand() % total;
        int cum = 0;
        unsigned char c3 = 0;
        for (j = 0; j < TRI_N; j++) {
            cum += g_trigram[c1][c2][j];
            if (cum > r) { c3 = (unsigned char)j; break; }
        }
        putchar(c3);
        c1 = c2;
        c2 = c3;
    }
    printf("\n");
}

/* Trigram train from file */
void mmc_trigram_train_file(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) { printf("[Cannot open %s]\n", filename); return; }
    
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    if (size > 500000) size = 500000;
    
    char* buf = malloc(size + 1);
    if (!buf) { fclose(f); return; }
    fread(buf, 1, size, f);
    buf[size] = 0;
    fclose(f);
    
    mmc_trigram_train(buf);
    free(buf);
}

/* 5-gram byte-level text generator */
#define FG_N 256
static int g_fgram[FG_N][FG_N][FG_N][FG_N][FG_N];  // too big!

// Better: hash-based sparse 5-gram
#define FG_HASH 65536
typedef struct { unsigned char ctx[4]; unsigned char next; int count; } FGEntry;
static FGEntry g_fg[FG_HASH];
static int g_fg_n = 0;

static unsigned int fg_hash4(unsigned char a, unsigned char b, unsigned char c, unsigned char d) {
    return ((unsigned int)a * 73856093u ^ (unsigned int)b * 19349663u ^ 
            (unsigned int)c * 83492791u ^ (unsigned int)d * 2654435761u) % FG_HASH;
}

void mmc_fgram_train(const char* text) {
    memset(g_fg, 0, sizeof(g_fg));
    g_fg_n = 0;
    int len = strlen(text);
    for (int i = 0; i < len - 4; i++) {
        unsigned int h = fg_hash4(text[i], text[i+1], text[i+2], text[i+3]);
        unsigned char nc = (unsigned char)text[i+4];
        while (g_fg[h].count > 0) {
            if (g_fg[h].ctx[0]==(unsigned char)text[i] && 
                g_fg[h].ctx[1]==(unsigned char)text[i+1] &&
                g_fg[h].ctx[2]==(unsigned char)text[i+2] &&
                g_fg[h].ctx[3]==(unsigned char)text[i+3] &&
                g_fg[h].next==nc) {
                g_fg[h].count++;
                goto next;
            }
            h = (h + 1) % FG_HASH;
        }
        g_fg[h].ctx[0] = (unsigned char)text[i];
        g_fg[h].ctx[1] = (unsigned char)text[i+1];
        g_fg[h].ctx[2] = (unsigned char)text[i+2];
        g_fg[h].ctx[3] = (unsigned char)text[i+3];
        g_fg[h].next = nc;
        g_fg[h].count = 1;
        g_fg_n++;
        next:;
    }
    printf("[5-gram trained: %d unique, %d bytes]\n", g_fg_n, len);
}

void mmc_fgram_generate(const char* seed, int max_len) {
    srand(42);
    int slen = strlen(seed);
    if (slen < 4) { printf("[seed need 4+ bytes]\n"); return; }
    
    unsigned char ctx[4] = {seed[slen-4], seed[slen-3], seed[slen-2], seed[slen-1]};
    printf("[Generated] %s", seed);
    
    for (int i = 0; i < max_len; i++) {
        unsigned int h = fg_hash4(ctx[0], ctx[1], ctx[2], ctx[3]);
        unsigned char next_char = 0;
        int total = 0;
        int start_h = h;
        // gather total count
        while (g_fg[h].count > 0) {
            if (g_fg[h].ctx[0]==ctx[0] && g_fg[h].ctx[1]==ctx[1] &&
                g_fg[h].ctx[2]==ctx[2] && g_fg[h].ctx[3]==ctx[3]) {
                total += g_fg[h].count;
            }
            h = (h + 1) % FG_HASH;
            if (h == start_h) break;
        }
        if (total == 0) break;
        int r = rand() % total;
        int cum = 0;
        h = start_h;
        while (g_fg[h].count > 0) {
            if (g_fg[h].ctx[0]==ctx[0] && g_fg[h].ctx[1]==ctx[1] &&
                g_fg[h].ctx[2]==ctx[2] && g_fg[h].ctx[3]==ctx[3]) {
                cum += g_fg[h].count;
                if (cum > r) { next_char = g_fg[h].next; break; }
            }
            h = (h + 1) % FG_HASH;
            if (h == start_h) break;
        }
        if (next_char == 0) break;
        putchar(next_char);
        ctx[0] = ctx[1]; ctx[1] = ctx[2]; ctx[2] = ctx[3]; ctx[3] = next_char;
    }
    printf("\n");
}

void mmc_fgram_train_file(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) { printf("[Cannot open %s]\n", filename); return; }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size > 500000) size = 500000;
    char* buf = malloc(size + 1);
    if (!buf) { fclose(f); return; }
    fread(buf, 1, size, f);
    buf[size] = 0;
    fclose(f);
    mmc_fgram_train(buf);
    free(buf);
}

/* ═══════════════════════════════════════════════════
   Char-level n-gram for Myanmar UTF-8 (3-byte chunks)
   ═══════════════════════════════════════════════════ */

// Parse UTF-8: return byte length of char at pos
static int utf8_len(const char* s, int pos) {
    unsigned char c = (unsigned char)s[pos];
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

// Char storage
#define MAX_CHARS 100000
static int g_char_starts[MAX_CHARS];  // byte position of each char
static int g_char_len;
static int g_char_total = 0;

void mmc_char_train(const char* text) {
    int len = strlen(text);
    g_char_total = 0;
    int pos = 0;
    while (pos < len && g_char_total < MAX_CHARS) {
        g_char_starts[g_char_total] = pos;
        int clen = utf8_len(text, pos);
        pos += clen;
        g_char_total++;
    }
    g_char_len = len;
    printf("[Char-trained: %d chars from %d bytes]\n", g_char_total, len);
}

// 3-char context hash table
#define CTX3_N 65536
typedef struct { int c1, c2, c3; int next_idx; int count; } CTX3Entry;
static CTX3Entry g_ctx3[CTX3_N];

static unsigned int ctx3_hash(int c1, int c2, int c3) {
    return ((c1 * 73856093u) ^ (c2 * 19349663u) ^ (c3 * 83492791u)) % CTX3_N;
}

// Char code at index
static int char_code_at(const char* text, int idx) {
    if (idx >= g_char_total) return -1;
    int pos = g_char_starts[idx];
    // Simple hash of 3-byte sequence
    unsigned char a = (unsigned char)text[pos];
    unsigned char b = (pos+1 < g_char_len) ? (unsigned char)text[pos+1] : 0;
    unsigned char c = (pos+2 < g_char_len) ? (unsigned char)text[pos+2] : 0;
    return a * 65536 + b * 256 + c;
}

void mmc_char3_build(const char* text) {
    memset(g_ctx3, 0, sizeof(g_ctx3));
    for (int i = 0; i + 3 < g_char_total; i++) {
        int c1 = char_code_at(text, i);
        int c2 = char_code_at(text, i+1);
        int c3 = char_code_at(text, i+2);
        int next = char_code_at(text, i+3);
        unsigned int h = ctx3_hash(c1, c2, c3);
        while (g_ctx3[h].count > 0) {
            if (g_ctx3[h].c1 == c1 && g_ctx3[h].c2 == c2 && 
                g_ctx3[h].c3 == c3 && g_ctx3[h].next_idx == next) {
                g_ctx3[h].count++;
                goto next_i;
            }
            h = (h + 1) % CTX3_N;
        }
        g_ctx3[h].c1 = c1;
        g_ctx3[h].c2 = c2;
        g_ctx3[h].c3 = c3;
        g_ctx3[h].next_idx = next;
        g_ctx3[h].count = 1;
        next_i:;
    }
    printf("[Char-3gram built from %d chars]\n", g_char_total);
}

void mmc_char3_generate(const char* text, const char* seed, int max_chars) {
    // Find seed position
    int seed_len = strlen(seed);
    int found = -1;
    for (int i = 0; i + 3 < g_char_total; i++) {
        int pos = g_char_starts[i];
        if (pos + seed_len <= g_char_len && strncmp(text + pos, seed, seed_len) == 0) {
            found = i;
            break;
        }
    }
    if (found < 0) { printf("[seed not found]\n"); return; }
    
    srand(42);
    printf("[Generated] %s", seed);
    
    int i1 = found, i2 = found+1, i3 = found+2;
    for (int n = 0; n < max_chars; n++) {
        if (i3 + 3 >= g_char_total) break;
        int c1 = char_code_at(text, i1);
        int c2 = char_code_at(text, i2);
        int c3 = char_code_at(text, i3);
        
        int total = 0;
        unsigned int h = ctx3_hash(c1, c2, c3);
        int start_h = h;
        while (g_ctx3[h].count > 0) {
            if (g_ctx3[h].c1==c1 && g_ctx3[h].c2==c2 && g_ctx3[h].c3==c3)
                total += g_ctx3[h].count;
            h = (h + 1) % CTX3_N;
            if (h == start_h) break;
        }
        if (total == 0) break;
        
        int r = rand() % total;
        int cum = 0;
        int chosen = -1;
        h = start_h;
        while (g_ctx3[h].count > 0) {
            if (g_ctx3[h].c1==c1 && g_ctx3[h].c2==c2 && g_ctx3[h].c3==c3) {
                cum += g_ctx3[h].count;
                if (cum > r) { chosen = g_ctx3[h].next_idx; break; }
            }
            h = (h + 1) % CTX3_N;
            if (h == start_h) break;
        }
        if (chosen < 0) break;
        
        // Find next char index with matching code
        int ni = -1;
        for (int k = i3 + 1; k < g_char_total; k++) {
            if (char_code_at(text, k) == chosen) { ni = k; break; }
        }
        if (ni < 0) break;
        
        // Print char
        int pos = g_char_starts[ni];
        int clen = utf8_len(text, pos);
        for (int b = 0; b < clen && pos + b < g_char_len; b++) putchar(text[pos+b]);
        
        i1 = i2; i2 = i3; i3 = ni;
    }
    printf("\n");
}

/* ═══════════════════════════════════════════════════
   Char 5-gram — Myanmar UTF-8 (4-char context)
   ═══════════════════════════════════════════════════ */

#define C5_HASH 262144
typedef struct {
    unsigned int h1, h2, h3, h4;  // char-code hashes
    int next_char_idx;
    int count;
} C5Entry;
static C5Entry g_c5[C5_HASH];
static char* g_c5_text = NULL;

static unsigned int c5_hash(unsigned int a, unsigned int b, unsigned int c, unsigned int d) {
    unsigned long long h = (unsigned long long)a * 0x9E3779B97F4A7C15ULL;
    h ^= (unsigned long long)b * 0xBF58476D1CE4E5B9ULL;
    h ^= (unsigned long long)c * 0x94D049BB133111EBULL;
    h ^= (unsigned long long)d * 0x2545F4914F6CDD1DULL;
    return (unsigned int)(h % C5_HASH);
}

static int c5_char_at(const char* text, int idx) {
    if (idx < 0 || idx >= g_char_total) return -1;
    int pos = g_char_starts[idx];
    unsigned int h = 0;
    int clen = utf8_len(text, pos);
    for (int b = 0; b < clen && pos + b < g_char_len; b++) {
        h = h * 256 + (unsigned char)text[pos + b];
    }
    return (int)(h & 0x7FFFFFFF);
}

void mmc_c5_train(const char* text) {
    memset(g_c5, 0, sizeof(g_c5));
    for (int i = 0; i + 4 < g_char_total; i++) {
        int c1 = c5_char_at(text, i);
        int c2 = c5_char_at(text, i+1);
        int c3 = c5_char_at(text, i+2);
        int c4 = c5_char_at(text, i+3);
        int next = c5_char_at(text, i+4);
        if (c1 < 0 || c2 < 0 || c3 < 0 || c4 < 0 || next < 0) continue;
        
        unsigned int h = c5_hash(c1, c2, c3, c4);
        for (int probe = 0; probe < 200; probe++) {
            unsigned int idx = (h + probe) % C5_HASH;
            if (g_c5[idx].count == 0) {
                g_c5[idx].h1 = c1; g_c5[idx].h2 = c2;
                g_c5[idx].h3 = c3; g_c5[idx].h4 = c4;
                g_c5[idx].next_char_idx = next;
                g_c5[idx].count = 1;
                break;
            } else if (g_c5[idx].h1 == (unsigned int)c1 && 
                       g_c5[idx].h2 == (unsigned int)c2 &&
                       g_c5[idx].h3 == (unsigned int)c3 &&
                       g_c5[idx].h4 == (unsigned int)c4 &&
                       g_c5[idx].next_char_idx == next) {
                g_c5[idx].count++;
                break;
            }
        }
    }
    printf("[Char-5gram trained]\n");
}

void mmc_c5_generate(const char* text, const char* seed, int max_chars) {
    int seed_len = strlen(seed);
    int found = -1;
    for (int i = 0; i + 4 < g_char_total; i++) {
        int pos = g_char_starts[i];
        if (pos + seed_len <= g_char_len && strncmp(text + pos, seed, seed_len) == 0) {
            found = i;
            break;
        }
    }
    if (found < 0) { printf("[seed not found]\n"); return; }
    
    srand(42);
    printf("[Generated] %s", seed);
    
    int i1 = found, i2 = found+1, i3 = found+2, i4 = found+3;
    for (int n = 0; n < max_chars; n++) {
        if (i4 + 4 >= g_char_total) break;
        int c1 = c5_char_at(text, i1);
        int c2 = c5_char_at(text, i2);
        int c3 = c5_char_at(text, i3);
        int c4 = c5_char_at(text, i4);
        
        int total = 0;
        unsigned int h = c5_hash(c1, c2, c3, c4);
        for (int p = 0; p < 200; p++) {
            unsigned int idx = (h + p) % C5_HASH;
            if (g_c5[idx].count == 0) break;
            if (g_c5[idx].h1 == (unsigned int)c1 && 
                g_c5[idx].h2 == (unsigned int)c2 &&
                g_c5[idx].h3 == (unsigned int)c3 &&
                g_c5[idx].h4 == (unsigned int)c4) {
                total += g_c5[idx].count;
            }
        }
        if (total == 0) break;
        
        int r = rand() % total;
        int cum = 0, chosen = -1;
        for (int p = 0; p < 200; p++) {
            unsigned int idx = (h + p) % C5_HASH;
            if (g_c5[idx].count == 0) break;
            if (g_c5[idx].h1 == (unsigned int)c1 && 
                g_c5[idx].h2 == (unsigned int)c2 &&
                g_c5[idx].h3 == (unsigned int)c3 &&
                g_c5[idx].h4 == (unsigned int)c4) {
                cum += g_c5[idx].count;
                if (cum > r) { chosen = g_c5[idx].next_char_idx; break; }
            }
        }
        if (chosen < 0) break;
        
        int ni = -1;
        for (int k = i4 + 1; k < g_char_total; k++) {
            if (c5_char_at(text, k) == chosen) { ni = k; break; }
        }
        if (ni < 0) break;
        
        int pos = g_char_starts[ni];
        int clen = utf8_len(text, pos);
        for (int b = 0; b < clen && pos + b < g_char_len; b++) putchar(text[pos+b]);
        
        i1 = i2; i2 = i3; i3 = i4; i4 = ni;
    }
    printf("\n");
}

void mmc_c5_train_file(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) { printf("[Cannot open %s]\n", filename); return; }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size > 500000) size = 500000;
    char* buf = malloc(size + 1);
    if (!buf) { fclose(f); return; }
    fread(buf, 1, size, f);
    buf[size] = 0;
    fclose(f);
    
    mmc_char_train(buf);
    mmc_c5_train(buf);
    g_c5_text = buf;  // save for later
    printf("[5-gram trained on %ld bytes]\n", size);
}

// For generation, need the text
void mmc_c5_generate_from_file(const char* seed, int max_chars) {
    if (!g_c5_text) { printf("[Train first]\n"); return; }
    mmc_c5_generate(g_c5_text, seed, max_chars);
}

/* ═══════════════════════════════════════════════════
   Word-level Markov Chain — Myanmar
   ═══════════════════════════════════════════════════ */

#define WM_HASH 16384
#define WM_MAX_WORDS 10000
#define WM_WORD_LEN 64

typedef struct {
    char word[WM_WORD_LEN];
    int next_ids[64];
    int next_counts[64];
    int next_n;
    int total;
} WMEntry;

static WMEntry g_wm[WM_HASH];
static char* g_wm_text = NULL;

static unsigned int wm_hash(const char* w) {
    unsigned int h = 5381;
    while (*w) h = ((h << 5) + h) + (unsigned char)*w++;
    return h % WM_HASH;
}

static WMEntry* wm_get(const char* word, int create) {
    unsigned int h = wm_hash(word);
    for (int p = 0; p < 100; p++) {
        unsigned int i = (h + p) % WM_HASH;
        if (g_wm[i].word[0] == 0) {
            if (!create) return NULL;
            strncpy(g_wm[i].word, word, WM_WORD_LEN-1);
            g_wm[i].next_n = 0;
            g_wm[i].total = 0;
            return &g_wm[i];
        }
        if (strcmp(g_wm[i].word, word) == 0) return &g_wm[i];
    }
    return NULL;
}

/* Simple Myanmar word splitter — splits at spaces and punctuation */
static int wm_split(const char* text, char words[][WM_WORD_LEN], int max_words) {
    int n = 0;
    int wi = 0;
    int len = strlen(text);
    char buf[WM_WORD_LEN];
    int bi = 0;
    
    for (int i = 0; i <= len && n < max_words; i++) {
        unsigned char c = (i < len) ? (unsigned char)text[i] : ' ';
        int is_space = (c == ' ' || c == '\n' || c == '\t' || c == '\r');
        int is_punct = (c == '.' || c == ',');
        
        if (is_space || is_punct || i == len) {
            if (bi > 0) {
                buf[bi] = 0;
                strncpy(words[n], buf, WM_WORD_LEN-1);
                words[n][WM_WORD_LEN-1] = 0;
                n++;
                bi = 0;
            }
            wi++;
        } else {
            if (bi < WM_WORD_LEN - 1) {
                buf[bi++] = (char)c;
            } else {
                buf[bi] = 0;
                strncpy(words[n], buf, WM_WORD_LEN-1);
                n++;
                bi = 0;
                buf[bi++] = (char)c;
            }
        }
    }
    return n;
}

void mmc_word_train(const char* text) {
    memset(g_wm, 0, sizeof(g_wm));
    
    static char words[WM_MAX_WORDS][WM_WORD_LEN];
    int n = wm_split(text, words, WM_MAX_WORDS);
    printf("[Word split: %d words]\n", n);
    
    for (int i = 0; i + 1 < n; i++) {
        WMEntry* e = wm_get(words[i], 1);
        if (!e) continue;
        WMEntry* ne = wm_get(words[i+1], 1);
        if (!ne) continue;
        
        int next_id = (int)(ne - g_wm);
        int found = 0;
        for (int j = 0; j < e->next_n; j++) {
            if (e->next_ids[j] == next_id) {
                e->next_counts[j]++;
                e->total++;
                found = 1;
                break;
            }
        }
        if (!found && e->next_n < 64) {
            e->next_ids[e->next_n] = next_id;
            e->next_counts[e->next_n] = 1;
            e->next_n++;
            e->total++;
        }
    }
    printf("[Word Markov trained]\n");
}

void mmc_word_generate(const char* seed_word, int max_words_out) {
    WMEntry* cur = wm_get(seed_word, 0);
    if (!cur) { printf("[seed word not found]\n"); return; }
    
    srand(42);
    printf("[Generated] %s", seed_word);
    
    for (int n = 0; n < max_words_out; n++) {
        if (cur->total == 0) break;
        int r = rand() % cur->total;
        int cum = 0;
        int chosen_id = -1;
        for (int j = 0; j < cur->next_n; j++) {
            cum += cur->next_counts[j];
            if (cum > r) { chosen_id = cur->next_ids[j]; break; }
        }
        if (chosen_id < 0) break;
        cur = &g_wm[chosen_id];
        printf(" %s", cur->word);
    }
    printf("\n");
}

void mmc_word_train_file(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) { printf("[Cannot open %s]\n", filename); return; }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size > 500000) size = 500000;
    char* buf = malloc(size + 1);
    if (!buf) { fclose(f); return; }
    fread(buf, 1, size, f);
    buf[size] = 0;
    fclose(f);
    
    mmc_word_train(buf);
    if (g_wm_text) free(g_wm_text);
    g_wm_text = buf;
    printf("[Word model on %ld bytes]\n", size);
}

/* ═══════════════════════════════════════════════════
   Tiny Transformer — Self-Attention LM
   ═══════════════════════════════════════════════════ */

#define TT_DIM 32
#define TT_VOCAB 64    // char classes
#define TT_CTX 16
#define TT_HEAD 4
#define TT_DK (TT_DIM / TT_HEAD)

static float g_tt_emb[TT_VOCAB][TT_DIM];
static float g_tt_pos[TT_CTX][TT_DIM];
static float g_tt_wq[TT_DIM][TT_DIM];
static float g_tt_wk[TT_DIM][TT_DIM];
static float g_tt_wv[TT_DIM][TT_DIM];
static float g_tt_wo[TT_DIM][TT_DIM];
static int   g_tt_ready = 0;

static void tt_init_rand() {
    if (g_tt_ready) return;
    srand(1337);
    for (int i = 0; i < TT_VOCAB; i++)
        for (int j = 0; j < TT_DIM; j++)
            g_tt_emb[i][j] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    for (int i = 0; i < TT_CTX; i++)
        for (int j = 0; j < TT_DIM; j++)
            g_tt_pos[i][j] = ((float)rand() / RAND_MAX - 0.5f) * 0.05f;
    for (int i = 0; i < TT_DIM; i++)
        for (int j = 0; j < TT_DIM; j++) {
            g_tt_wq[i][j] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
            g_tt_wk[i][j] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
            g_tt_wv[i][j] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
            g_tt_wo[i][j] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
    g_tt_ready = 1;
}

/* Matmul helper */
static void tt_matmul(float* A, float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < K; k++) s += A[i*K+k] * B[k*N+j];
            C[i*N+j] = s;
        }
}

/* Softmax row */
static void tt_softmax_row(float* x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

/* Single forward pass on token sequence */
void mmc_tt_forward(const int* tokens, int T, float* out_logits) {
    tt_init_rand();
    if (T > TT_CTX) T = TT_CTX;
    
    /* 1. Embedding + Positional */
    static float X[TT_CTX][TT_DIM];
    for (int t = 0; t < T; t++) {
        int tok = tokens[t] % TT_VOCAB;
        for (int d = 0; d < TT_DIM; d++)
            X[t][d] = g_tt_emb[tok][d] + g_tt_pos[t][d];
    }
    
    /* 2. Q = X * Wq, K = X * Wk, V = X * Wv */
    static float Q[TT_CTX][TT_DIM], K[TT_CTX][TT_DIM], V[TT_CTX][TT_DIM];
    for (int t = 0; t < T; t++) {
        for (int d = 0; d < TT_DIM; d++) {
            float sq = 0, sk = 0, sv = 0;
            for (int k = 0; k < TT_DIM; k++) {
                sq += X[t][k] * g_tt_wq[k][d];
                sk += X[t][k] * g_tt_wk[k][d];
                sv += X[t][k] * g_tt_wv[k][d];
            }
            Q[t][d] = sq; K[t][d] = sk; V[t][d] = sv;
        }
    }
    
    /* 3. Attention scores + softmax */
    static float attn[TT_CTX][TT_CTX];
    for (int i = 0; i < T; i++) {
        for (int j = 0; j < T; j++) {
            float s = 0;
            for (int d = 0; d < TT_DIM; d++) s += Q[i][d] * K[j][d];
            s /= sqrtf((float)TT_DIM);
            attn[i][j] = (j <= i) ? s : -1e9f;  // causal
        }
        tt_softmax_row(attn[i], T);
    }
    
    /* 4. Output: O = attn * V * Wo */
    static float O[TT_CTX][TT_DIM], H[TT_CTX][TT_DIM];
    for (int i = 0; i < T; i++) {
        for (int d = 0; d < TT_DIM; d++) {
            float s = 0;
            for (int j = 0; j < T; j++) s += attn[i][j] * V[j][d];
            O[i][d] = s;
        }
    }
    for (int t = 0; t < T; t++) {
        for (int d = 0; d < TT_DIM; d++) {
            float s = 0;
            for (int k = 0; k < TT_DIM; k++) s += O[t][k] * g_tt_wo[k][d];
            H[t][d] = s + X[t][d];  // residual
        }
    }
    
    /* 5. Output projection (last position -> vocab logits) */
    int last = T - 1;
    for (int v = 0; v < TT_VOCAB; v++) {
        float s = 0;
        for (int d = 0; d < TT_DIM; d++) s += H[last][d] * g_tt_emb[v][d];
        out_logits[v] = s;
    }
}

/* Attention weights dump for inspection */
void mmc_tt_attention(const int* tokens, int T) {
    tt_init_rand();
    if (T > TT_CTX) T = TT_CTX;
    
    static float X[TT_CTX][TT_DIM];
    for (int t = 0; t < T; t++) {
        int tok = tokens[t] % TT_VOCAB;
        for (int d = 0; d < TT_DIM; d++)
            X[t][d] = g_tt_emb[tok][d] + g_tt_pos[t][d];
    }
    
    printf("[Attention Matrix %dx%d]\n", T, T);
    for (int i = 0; i < T; i++) {
        for (int j = 0; j < T; j++) {
            float s = 0;
            for (int d = 0; d < TT_DIM; d++) {
                float q = 0, k = 0;
                for (int kk = 0; kk < TT_DIM; kk++) {
                    q += X[i][kk] * g_tt_wq[kk][d];
                    k += X[j][kk] * g_tt_wk[kk][d];
                }
                s += q * k;
            }
            s /= sqrtf((float)TT_DIM);
            if (j > i) s = -1e9f;
            printf("%.2f ", s);
        }
        printf("\n");
    }
}

/* ═══════════════════════════════════════════════════
   String helpers — added 2026-10-12 for nyanlin_v5.mmc
   ═══════════════════════════════════════════════════ */

/* Split string by delimiter — return nth token.
   4-slot rotating buffer — allows 3-4 consecutive calls. */
char* mmc_string_split_v2(const char* str, const char* delim, int index) {
    char* buf = (char*)malloc(512);
    if (!buf) return NULL;
    buf[0] = 0;
    if (!str || !delim || index < 0) return buf;
    int dlen = (int)strlen(delim);
    if (dlen == 0) return buf;
    const char* p   = str;
    int         cur = 0;
    while (*p) {
        const char* next    = strstr(p, delim);
        int         seg_len = next ? (int)(next - p) : (int)strlen(p);
        if (cur == index) {
            if (seg_len > 511) seg_len = 511;
            memcpy(buf, p, seg_len);
            buf[seg_len] = 0;
            return buf;
        }
        cur++;
        if (!next) break;
        p = next + dlen;
    }
    return buf;
}

/* String → int. Handles ASCII 0-9 AND Myanmar ၀-၉ (U+1040-1049) */
int mmc_string_to_int(const char* str) {
    if (!str) return 0;
    const unsigned char* p = (const unsigned char*)str;
    int sign = 1, result = 0;

    while (*p == ' ' || *p == '\t') p++;
    if (*p == '-') { sign = -1; p++; }
    else if (*p == '+') p++;

    while (*p) {
        if (*p >= '0' && *p <= '9') {
            result = result * 10 + (*p - '0');
            p++;
        } else if (p[0] == 0xE1 && p[1] == 0x81 &&
                   p[2] >= 0x80 && p[2] <= 0x89) {
            result = result * 10 + (p[2] - 0x80);
            p += 3;
        } else break;
    }
    return result * sign;
}

/* Trim leading/trailing whitespace (static buffer) */
char* mmc_string_trim(const char* str) {
    if (!str) { char* b = (char*)malloc(1); if (b) b[0]=0; return b; }
    const char* start = str;
    while (*start == 32 || *start == 9 || *start == 10 || *start == 13)
        start++;
    const char* end = start + strlen(start);
    while (end > start &&
           (end[-1] == 32 || end[-1] == 9 || end[-1] == 10 || end[-1] == 13))
        end--;
    size_t len = (size_t)(end - start);
    char* buf = (char*)malloc(len + 1);
    if (!buf) return NULL;
    memcpy(buf, start, len);
    buf[len] = 0;
    return buf;
}

/* ═══════════════════════════════════════════════════
   String helpers — added 2026-10-12 for nyanlin_v5.mmc
   ═══════════════════════════════════════════════════ */

/* ═══════════════════════════════════════════════════
   String helpers — added 2026-10-12 for nyanlin_v5.mmc
   ═══════════════════════════════════════════════════ */


/* String concat — append — added 2026-10-12 */
char* mmc_string_concat(const char* a, const char* b) {
    size_t la = a ? strlen(a) : 0;
    size_t lb = b ? strlen(b) : 0;
    char* buf = (char*)malloc(la + lb + 1);
    if (!buf) return NULL;
    buf[0] = 0;
    if (a) strcat(buf, a);
    if (b) strcat(buf, b);
    return buf;
}

/* File-based memory — added 2026-10-12 */
void mmc_memory_save_to(const char* filename, const char* key, const char* value) {
    static char lines[65536][512];
    int n = 0;
    FILE* f = fopen(filename, "r");
    if (f) {
        while (n < 65536 && fgets(lines[n], 512, f)) n++;
        fclose(f);
    }
    char prefix[280];
    snprintf(prefix, sizeof(prefix), "%s = ", key);
    size_t plen = strlen(prefix);
    int replaced = 0;
    for (int i = 0; i < n; i++) {
        if (strncmp(lines[i], prefix, plen) == 0) {
            snprintf(lines[i], 512, "%s = %s\n", key, value);
            replaced = 1;
            break;
        }
    }
    if (!replaced && n < 65536) {
        snprintf(lines[n], 512, "%s = %s\n", key, value);
        n++;
    }
    f = fopen(filename, "w");
    if (f) {
        for (int i = 0; i < n; i++) fputs(lines[i], f);
        fclose(f);
    }
}

char* mmc_memory_load_from(const char* filename, const char* key) {
    static char result[256];
    result[0] = 0;
    FILE* f = fopen(filename, "r");
    if (!f) return result;
    char prefix[280];
    snprintf(prefix, sizeof(prefix), "%s = ", key);
    size_t plen = strlen(prefix);
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, prefix, plen) == 0) {
            char* val = line + plen;
            size_t vlen = strlen(val);
            while (vlen > 0 && (val[vlen-1] == 10 || val[vlen-1] == 13))
                val[--vlen] = 0;
            strncpy(result, val, 255);
            result[255] = 0;
        }
    }
    fclose(f);
    return result;
}

/* Dataset fuzzy search — added 2026-10-12 */
char* mmc_dataset_search(const char* filename, const char* keyword) {
    static char result[512];
    result[0] = 0;
    if (!filename || !keyword || keyword[0] == 0) return result;
    FILE* f = fopen(filename, "r");
    if (!f) return result;
    char line[1024];
    int klen = strlen(keyword);
    int best_score = 0;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == 10) continue;
        char* eq = strstr(line, " = ");
        if (!eq) continue;
        *eq = 0;
        char* key_part = line;
        char* val_part = eq + 3;
        size_t vl = strlen(val_part);
        while (vl > 0 && (val_part[vl-1] == 10 || val_part[vl-1] == 13))
            val_part[--vl] = 0;
        /* Priority scoring */
        int score = 0;
        int kp_len = strlen(key_part);
        if (strcmp(key_part, keyword) == 0) {
            score = 100;  /* exact */
        } else if (kp_len > klen && strncmp(key_part, keyword, klen) == 0 && key_part[klen] == 32) {
            score = 80;   /* prefix + space */
        } else if (strstr(key_part, keyword)) {
            score = 40;   /* substring */
        }
        if (score > best_score) {
            best_score = score;
            strncpy(result, val_part, 511);
            result[511] = 0;
            if (score == 100) break;  /* exact = best, stop */
        }
    }
    fclose(f);
    return result;
}

/* File scan helpers — added 2026-10-12 */
int mmc_file_line_count(const char* filename) {
    FILE* f = fopen(filename, "r");
    if (!f) return 0;
    int n = 0;
    char line[2048];
    while (fgets(line, sizeof(line), f)) n++;
    fclose(f);
    return n;
}

char* mmc_file_get_line(const char* filename, int index) {
    static char result[2048];
    result[0] = 0;
    if (index < 0) return result;
    FILE* f = fopen(filename, "r");
    if (!f) return result;
    int i = 0;
    char line[2048];
    while (fgets(line, sizeof(line), f)) {
        if (i == index) {
            size_t len = strlen(line);
            while (len > 0 && (line[len-1] == 10 || line[len-1] == 13))
                line[--len] = 0;
            strncpy(result, line, 2047);
            result[2047] = 0;
            fclose(f);
            return result;
        }
        i++;
    }
    fclose(f);
    return result;
}

char* mmc_int_to_str(int n) {
    static char buf[32];
    snprintf(buf, sizeof(buf), "%d", n);
    return buf;
}

/* Bigram prediction — added 2026-10-12 */
char* mmc_model_predict(const char* filename, const char* prefix) {
    static char result[512];
    result[0] = 0;
    if (!filename || !prefix || prefix[0] == 0) return result;
    FILE* f = fopen(filename, "r");
    if (!f) return result;
    char pf[300];
    snprintf(pf, sizeof(pf), "%s_", prefix);
    size_t plen = strlen(pf);
    int best = 0;
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, pf, plen) != 0) continue;
        char* eq = strstr(line, " = ");
        if (!eq) continue;
        int cnt = atoi(eq + 3);
        if (cnt > best) {
            best = cnt;
            char* v = eq + 3;
            size_t vl = strlen(v);
            while (vl > 0 && (v[vl-1] == 10 || v[vl-1] == 13)) v[--vl] = 0;
            strncpy(result, v, 511);
            result[511] = 0;
        }
    }
    fclose(f);
    return result;
}

/* System with stdin capture — added 2026-10-12 */
char* mmc_system_stdin(const char* cmd, const char* input) {
    static char output[8192];
    output[0] = 0;
    if (!cmd) return output;
    const char* tmp = ".mmc_system_in.tmp";
    FILE* f = fopen(tmp, "w");
    if (!f) return output;
    if (input) fputs(input, f);
    fclose(f);
    char full[2048];
    snprintf(full, sizeof(full), "%s < %s 2>&1", cmd, tmp);
    FILE* p = popen(full, "r");
    if (!p) return output;
    size_t n = fread(output, 1, sizeof(output) - 1, p);
    output[n] = 0;
    pclose(p);
    size_t len = strlen(output);
    while (len > 0 && (output[len-1] == 10 || output[len-1] == 13))
        output[--len] = 0;
    return output;
}


/* Trigram predict — added 2026-10-12 */
char* mmc_trigram_predict(const char* filename, const char* w1, const char* w2) {
    static char result[512];
    result[0] = 0;
    if (!filename || !w1 || !w2) return result;
    FILE* f = fopen(filename, "r");
    if (!f) return result;
    char pf[600];
    snprintf(pf, sizeof(pf), "%s_%s_", w1, w2);
    size_t plen = strlen(pf);
    int best = 0;
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, pf, plen) != 0) continue;
        char* eq = strstr(line, " = ");
        if (!eq) continue;
        int cnt = atoi(eq + 3);
        if (cnt > best) {
            best = cnt;
            char* next = line + plen;
            char* end = strstr(next, " = ");
            if (end) *end = 0;
            strncpy(result, next, 511);
            result[511] = 0;
        }
    }
    fclose(f);
    return result;
}


/* ═══════════════════════════════════════════════════
   HTTP Brain Server — added 2026-10-12
   ═══════════════════════════════════════════════════ */
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

static void http_url_decode(char* s) {
    char* out = s;
    while (*s) {
        if (*s == '%' && s[1] && s[2]) {
            int hi = (s[1] >= 'A') ? (s[1] - 'A' + 10) : (s[1] - '0');
            int lo = (s[2] >= 'A') ? (s[2] - 'A' + 10) : (s[2] - '0');
            *out++ = (char)((hi << 4) | lo);
            s += 3;
        } else if (*s == '+') {
            *out++ = ' ';
            s++;
        } else {
            *out++ = *s++;
        }
    }
    *out = 0;
}

static void http_html_escape(const char* in, char* out, int max) {
    int oi = 0;
    for (int i = 0; in[i] && oi < max - 8; i++) {
        char c = in[i];
        if (c == '<') { strcpy(out+oi, "&lt;"); oi += 4; }
        else if (c == '>') { strcpy(out+oi, "&gt;"); oi += 4; }
        else if (c == '&') { strcpy(out+oi, "&amp;"); oi += 5; }
        else { out[oi++] = c; }
    }
    out[oi] = 0;
}

static const char* http_brain_lookup(const char* q_raw) {
    static char result[512];
    static char q[1024];
    if (!q_raw) q_raw = "";
    strncpy(q, q_raw, 1023);
    q[1023] = 0;
    /* Strip trailing punctuation + whitespace */
    int len = strlen(q);
    while (len > 0) {
        unsigned char ch = (unsigned char)q[len-1];
        if (ch == 63 || ch == 33 || ch == 46 || ch == 44 ||
            ch == 32 || ch == 9 || ch == 10 || ch == 13) {
            q[--len] = 0;
        } else break;
    }
    /* Strip leading whitespace */
    char* p = q;
    while (*p == 32 || *p == 9) p++;
    /* 1. Exact on stripped */
    strncpy(result, mmc_memory_load_from("dataset_mm_en.txt", p), 511);
    result[511] = 0;
    if (result[0]) return result;
    /* 2. Fuzzy on stripped */
    strncpy(result, mmc_dataset_search("dataset_mm_en.txt", p), 511);
    result[511] = 0;
    if (result[0]) return result;
    /* 3. First word only */
    char w1[256];
    int wi = 0;
    while (p[wi] && p[wi] != 32 && wi < 255) { w1[wi] = p[wi]; wi++; }
    w1[wi] = 0;
    if (wi > 0 && strcmp(w1, p) != 0) {
        strncpy(result, mmc_memory_load_from("dataset_mm_en.txt", w1), 511);
        result[511] = 0;
        if (result[0]) return result;
        strncpy(result, mmc_dataset_search("dataset_mm_en.txt", w1), 511);
        result[511] = 0;
        if (result[0]) return result;
    }
    return "(dataset ထဲ မရှိပါ)";
}

int mmc_http_serve(int port) {
    int server = socket(AF_INET, SOCK_STREAM, 0);
    if (server < 0) { printf("socket failed\n"); return 1; }
    int opt = 1;
    setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons((unsigned short)port);
    if (bind(server, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        printf("bind failed\n"); close(server); return 1;
    }
    if (listen(server, 8) < 0) { printf("listen failed\n"); close(server); return 1; }
    printf("[HTTP] http://localhost:%d\n", port);
    fflush(stdout);
    while (1) {
        struct sockaddr_in cli;
        socklen_t cl = sizeof(cli);
        int fd = accept(server, (struct sockaddr*)&cli, &cl);
        if (fd < 0) continue;
        char req[4096];
        int n = read(fd, req, sizeof(req) - 1);
        if (n <= 0) { close(fd); continue; }
        req[n] = 0;
        char method[16] = {0}, path[2048] = {0};
        sscanf(req, "%15s %2047s", method, path);
        char* q = strstr(path, "?q=");
        char query[1024] = "";
        if (q) { strncpy(query, q + 3, 1023); query[1023] = 0; }
        http_url_decode(query);
        char answer[512] = "";
        if (query[0]) strncpy(answer, http_brain_lookup(query), 511);
        char qe[2048], ae[1024];
        http_html_escape(query, qe, sizeof(qe));
        http_html_escape(answer, ae, sizeof(ae));
        char body[8192];
        snprintf(body, sizeof(body),
            "<!doctype html><meta charset=utf-8>"
            "<title>Nyan Lin Brain</title>"
            "<style>body{font-family:monospace;background:#111;color:#0f0;padding:20px}"
            "input{width:70%;padding:8px;background:#222;color:#0f0;border:1px solid #0f0}"
            "button{padding:8px;background:#0f0;color:#000;border:0}"
            ".a{margin-top:20px;padding:10px;border:1px solid #0f0}</style>"
            "<h1>Nyan Lin Brain</h1>"
            "<form method=get><input name=q value=\"%s\" autofocus>"
            "<button>Ask</button></form>"
            "<div class=a><b>Q:</b> %s<br><b>A:</b> %s</div>",
            qe, qe, ae);
        char resp[9216];
        int blen = strlen(body);
        int rlen = snprintf(resp, sizeof(resp),
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/html; charset=utf-8\r\n"
            "Content-Length: %d\r\n"
            "Connection: close\r\n\r\n%s",
            blen, body);
        write(fd, resp, rlen);
        close(fd);
        printf("[HTTP] Q=%s\n", query);
        fflush(stdout);
    }
    close(server);
    return 0;
}

