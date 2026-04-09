#ifndef LLAMA2C_DEPLOY_RUNTIME_COMMON_H
#define LLAMA2C_DEPLOY_RUNTIME_COMMON_H

#include "runtime_types.h"

// runtime_common 汇总部署运行时的公共能力：
// - arena 与 state 生命周期管理
// - q80 量化辅助与参考数学实现
// - tokenizer 编解码与 sampler
// - decode 调度器中的共享步骤（RoPE、embedding 装载等）

void arena_init(RuntimeArena *arena, unsigned char *base, size_t size);
void arena_reset(RuntimeArena *arena);
// 从固定 arena 中按对齐要求切出一段空间；返回 NULL 表示空间不足。
void *arena_alloc(RuntimeArena *arena, size_t size, size_t align);

void runtime_model_init(RuntimeModel *model);
void runtime_model_destroy(RuntimeModel *model);

// 按模型配置在固定 arena 中分配运行时中间缓冲和 KV cache。
void runtime_malloc_state(RuntimeState *state, const RuntimeConfig *config, int group_size);
void runtime_free_state(RuntimeState *state);

// 计算 q80/grouped quantization 所需的 scale 组数。
int grouped_scale_count(int n, int group_size);

void dequantize_tensor(QuantizedTensor *qx, float *x, int n, int group_size);
void quantize_tensor(QuantizedTensor *qx, float *x, int n, int group_size);
// 把连续内存解释成若干个 QuantizedTensor 视图，不复制底层数据。
QuantizedTensor *init_quantized_tensors(void **ptr, int n, int size_each, int group_size);

void rmsnorm_ref(float *out, const float *x, const float *weight, int size);
void softmax_ref(float *x, int size);
void matmul_ref(float *xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d, int group_size);

int compare_tokens(const void *a, const void *b);
char *decode_token(RuntimeTokenizer *tokenizer, int prev_token, int token);
void safe_printf_piece(char *piece);
int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size);
// 将文本编码成 token 序列；token_capacity 不足时返回 -1。
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
// 基于 logits 和已执行历史 token 采样下一个 token。
int sample_token(RuntimeSampler *sampler, float *logits, const int *history, int history_len);

void read_stdin_line(const char *guide, char *buffer, size_t bufsize);

// 对当前 token 的 q/k 向量原地施加 RoPE 位置编码。
void apply_rope_inplace(float *q, float *k, int dim, int kv_dim, int head_size, int pos);

// 从 q80 embedding 表中按 token 解出一行到 float 缓冲。
void runtime_load_embedding_row(const RuntimeModel *model, int token, float *out);

// 部署/验证共用的单 token decode 调度器，返回 state->logits。
float *runtime_decode_transformer_step(RuntimeBackend *backend, int token, int pos);

#endif
