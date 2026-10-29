#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int json_extract_text(char*, int, char*, int);
extern int mmc_bpe_encode(void*, char*, int, void*, int);
extern int mmc_gguf_parse_tensors(void*, int);
extern int mmc_gguf_parse_tokenizer(void*, int);

#define TEXT_CAP  131072
#define TOKEN_CAP 32768

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s model.gguf out.bin < input.jsonl\n", argv[0]);
        return 1;
    }

    /* 1. Load GGUF */
    FILE* f = fopen(argv[1], "rb");
    if (!f) { perror("gguf"); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    void* buf = malloc(sz);
    if (fread(buf, 1, sz, f) != (size_t)sz) { perror("read"); return 1; }
    fclose(f);
    fprintf(stderr, "gguf=%ld bytes\n", sz);

    /* 2. Parse (0 = success) */
    int r1 = mmc_gguf_parse_tensors(buf, sz);
    int r2 = mmc_gguf_parse_tokenizer(buf, sz);
    fprintf(stderr, "parse: tensors=%d tokenizer=%d\n", r1, r2);
    if (r1 != 0 || r2 != 0) { fprintf(stderr, "parse failed\n"); return 1; }

    /* 3. Stream JSONL → tokens */
    FILE* fout = fopen(argv[2], "wb");
    if (!fout) { perror("out"); return 1; }

    char lbuf[262144], tbuf[TEXT_CAP];
    unsigned short ibuf[TOKEN_CAP];
    int lines = 0, total = 0, enc_fail = 0;

    while (fgets(lbuf, sizeof(lbuf), stdin)) {
        int ln = strlen(lbuf);
        if (ln < 10) continue;
        int tl = json_extract_text(lbuf, ln, tbuf, TEXT_CAP);
        if (tl < 20) continue;
        int tc = mmc_bpe_encode(buf, tbuf, tl, ibuf, TOKEN_CAP);
        if (tc <= 0) {
            enc_fail++;
            /* Debug: log first 5 failures */
            if (enc_fail <= 5) {
                int ll = (int)strlen(lbuf);
                fprintf(stderr, "FAIL[%d] code=%d lbuflen=%d head=%.80s\n",
                        enc_fail, tc, ll, lbuf);
            }
            continue;
        }
        fwrite(ibuf, 2, tc, fout);
        total += tc;
        lines++; if (lines % 1000 == 0) fprintf(stderr, "progress: %d lines, %d tokens\n", lines, total);
    }
    fclose(fout);
    fprintf(stderr, "lines=%d total_tokens=%d enc_fail=%d\n", lines, total, enc_fail);
    return 0;
}
