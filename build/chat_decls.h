/* GGUF + mmc_generate prototypes — impl in v49_final.o */
#ifndef CHAT_DECLS_H
#define CHAT_DECLS_H
#include <stddef.h>
void* mmc_load_binary(const char* filename);
int   mmc_load_binary_size(void);
int   mmc_gguf_parse_tensors(void* buf, int size);
int   mmc_gguf_parse_tokenizer(void* buf, int size);
int   gguf_data_start(void);
int   gguf_tensor_offset(int i);
int   gguf_tensor_type(int i);
int   gguf_tensor_dim(int i, int d);
int   sampling_set(double t, int k, int seed);
int   decode_set(double rep, int norep);
int   stop_set(const char* s);
int   mmc_generate(void* buf, int ds, int d, int d_ff, int n_layers,
                   int n_heads, int vocab, int ong, int onb,
                   int lm_head, int lm_type, int max_seq,
                   int tok_off, int pos_off, const char* prompt,
                   int prompt_len, int max_new, int eos_id,
                   void* out_ptr, int out_cap, void* kc_ptr,
                   void* vc_ptr, void* logits_ptr);
#endif
