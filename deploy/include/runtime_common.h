#ifndef LLAMA2C_DEPLOY_RUNTIME_COMMON_H
#define LLAMA2C_DEPLOY_RUNTIME_COMMON_H

#include "runtime_types.h"

void arena_init(RuntimeArena *arena, unsigned char *base, size_t size);
void arena_reset(RuntimeArena *arena);
void *arena_alloc(RuntimeArena *arena, size_t size, size_t align);

void runtime_model_init(RuntimeModel *model);
void runtime_model_destroy(RuntimeModel *model);

void runtime_malloc_state(RuntimeState *state, const RuntimeConfig *config, int group_size);
void runtime_free_state(RuntimeState *state);

int grouped_scale_count(int n, int group_size);

void dequantize_tensor(QuantizedTensor *qx, float *x, int n, int group_size);
void quantize_tensor(QuantizedTensor *qx, float *x, int n, int group_size);
QuantizedTensor *init_quantized_tensors(void **ptr, int n, int size_each, int group_size);

void rmsnorm_ref(float *out, const float *x, const float *weight, int size);
void softmax_ref(float *x, int size);
void matmul_ref(float *xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d, int group_size);

int compare_tokens(const void *a, const void *b);
char *decode_token(RuntimeTokenizer *tokenizer, int prev_token, int token);
void safe_printf_piece(char *piece);
int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size);
int encode_text(
    RuntimeTokenizer *tokenizer,
    char *text,
    int8_t bos,
    int8_t eos,
    int *tokens,
    int token_capacity,
    int *n_tokens
);

void build_sampler(
    RuntimeSampler *sampler,
    int vocab_size,
    float temperature,
    float topp,
    int top_k,
    float repetition_penalty,
    int no_repeat_ngram_size,
    unsigned long long rng_seed
);
void free_sampler(RuntimeSampler *sampler);
int sample_token(RuntimeSampler *sampler, float *logits, const int *history, int history_len);

void read_stdin_line(const char *guide, char *buffer, size_t bufsize);

void apply_rope_inplace(float *q, float *k, int dim, int kv_dim, int head_size, int pos);

void runtime_load_embedding_row(const RuntimeModel *model, int token, float *out);
float *runtime_decode_transformer_step(RuntimeBackend *backend, int token, int pos);

#endif
