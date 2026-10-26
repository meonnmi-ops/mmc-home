#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int mmc_gguf_parse_tensors(void* buf, int size);
int mmc_gguf_parse_tokenizer(void* buf, int size);
int gguf_data_start();
int gguf_tensor_offset(int i);
int gguf_tensor_type(int i);
int gguf_tensor_dim(int i, int d);
int gguf_token_count();
int sampling_set(double t, int k, int seed);
int decode_set(double rep, int norep);
int stop_set(char* s);
int stop_hit();
int mmc_generate(void* buf, int ds, int d, int d_ff, int n_layers, int n_heads, int vocab,
    int ong, int onb, int lm_head, int lm_type,
    int max_seq, int tok_off, int pos_off,
    char* prompt, int prompt_len, int max_new, int eos_id,
    void* out, int out_cap, void* kc, void* vc, void* logits);
int gen_count();
int gen_id(int i);

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("usage: %s model.gguf \"question\" [max_new] [temp] [top_k] [seed]\n", argv[0]);
        return 1;
    }
    FILE* f = fopen(argv[1], "rb");
    if (!f) { printf("cannot open model\n"); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    void* buf = malloc(sz + 16);
    fread(buf, 1, sz, f); fclose(f);

    mmc_gguf_parse_tensors(buf, (int)sz);
    int ds = gguf_data_start();
    if (mmc_gguf_parse_tokenizer(buf, (int)sz) != 0) { printf("tokenizer fail\n"); return 1; }

    int tok_o = gguf_tensor_offset(4);
    int pos_o = gguf_tensor_offset(3);
    int onb   = gguf_tensor_offset(1);
    int ong   = gguf_tensor_offset(2);
    int lmo   = gguf_tensor_offset(0);
    int lmt   = gguf_tensor_type(0);
    int vocab = gguf_tensor_dim(4, 1);

    int max_new = argc > 3 ? atoi(argv[3]) : 40;
    double temp = argc > 4 ? atof(argv[4]) : 1.0;
    int top_k   = argc > 5 ? atoi(argv[5]) : 1;
    int seed    = argc > 6 ? atoi(argv[6]) : 42;
    int MAXS = 64;

    char chat[4096];
    if (strstr(argv[2], "<user>") != NULL) {
        snprintf(chat, sizeof(chat), "%s", argv[2]);
    } else {
        snprintf(chat, sizeof(chat), "<user> %s\n<nyanlin>", argv[2]);
    }

    stop_set("\n");          /* answer ends at newline (training format) */
    decode_set(1.15, 6);     /* anti-repeat: engine v2 defaults */
    sampling_set(temp, top_k, seed);

    double* kc = calloc((size_t)24 * MAXS * 1024, sizeof(double));
    double* vc = calloc((size_t)24 * MAXS * 1024, sizeof(double));
    double* logits = malloc((size_t)vocab * sizeof(double));
    char out[4096];
    memset(out, 0, sizeof out);

    clock_t t0 = clock();
    int n = mmc_generate(buf, ds, 1024, 4096, 24, 16, vocab, ong, onb, lmo, lmt,
                         MAXS, tok_o, pos_o, chat, (int)strlen(chat),
                         max_new, 2, out, (int)sizeof(out), kc, vc, logits);
    clock_t t1 = clock();

    int s0 = 0;
    while (s0 < n && (out[s0]=='\n' || out[s0]==' ' || out[s0]=='\r' || out[s0]=='\t')) s0++;
    int s1 = n;
    while (s1 > s0 && (out[s1-1]=='\n' || out[s1-1]==' ' || out[s1-1]=='\r' || out[s1-1]=='\t')) s1--;

    printf("\n========================================\n");
    printf("Q: %s\n", argv[2]);
    printf("----------------------------------------\n");
    printf("Nyanlin: ");
    fwrite(out + s0, 1, s1 - s0, stdout);
    printf("\n========================================\n");
    printf("gen=%d  time=%.1fs (%.2fs/token)  temp=%.2f top_k=%d\n",
           gen_count(), (double)(t1-t0)/CLOCKS_PER_SEC,
           gen_count() ? (double)(t1-t0)/CLOCKS_PER_SEC/gen_count() : 0.0,
           temp, top_k);
    printf("stop_reason=%d (0=max_new 1=newline 2=eos 3=step_err)\n", stop_hit());
    printf("ids:");
    for (int i = 0; i < gen_count(); i++) printf(" %d", gen_id(i));
    printf("\n");

    free(kc); free(vc); free(logits); free(buf);
    return 0;
}
