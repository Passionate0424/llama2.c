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

// 对当前 token 的 q/k 行原地施加 RoPE 位置编码。
void apply_rope_inplace(float *q, float *k, int dim, int kv_dim, int head_size, int pos);
void runtime_apply_rope_to_qk_row(float *q, float *k_row, int dim, int kv_dim, int head_size, int pos);

// 从 q80 embedding 表中按 token 解出一行到 float 缓冲。
void runtime_load_embedding_row(const RuntimeModel *model, int token, float *out);

// 统一把某一层的历史 KV 暴露成 layer view，避免 backend 继续吃裸 float* 历史 KV。
void runtime_init_kv_cache_layer_view(
    RuntimeKvCacheLayerView *view,
    const SharedBufferDesc *data_region,
    const SharedBufferDesc *scale_region,
    float *legacy_float_data,
    int seq_len,
    int kv_dim,
    int head_size,
    int kv_mul,
    size_t data_stride_bytes,
    size_t scale_stride_bytes
);

// 统一计算某层某时刻的 KV data byte offset，并按 view 解释数据。
size_t runtime_kv_cache_row_data_offset(const RuntimeKvCacheLayerView *view, int time_idx);
const float *runtime_kv_cache_head_ptr(const RuntimeKvCacheLayerView *view, int time_idx, int head_idx);
void runtime_kv_cache_extract_row(const RuntimeKvCacheLayerView *view, int time_idx, float *dst_row);
void runtime_kv_cache_write_row(RuntimeKvCacheLayerView *view, int time_idx, const float *src_row);
void runtime_kv_cache_write_row_int8(RuntimeKvCacheLayerView *view, int time_idx, const float *src_row);
void runtime_kv_cache_read_row_int8(const RuntimeKvCacheLayerView *view, int time_idx, float *dst_row);
int runtime_kv_cache_uses_int8_data(const RuntimeKvCacheLayerView *view);

// 控制运行时是否优先走最小 int8 data 路径；默认关闭，仅作为实验开关。
void runtime_set_kv_int8_mode(int enabled);
int runtime_get_kv_int8_mode(void);

// 刷新 RuntimeState 中 KV_MAIN data/scale region 的地址口径。
void runtime_refresh_kv_main_regions(RuntimeState *state, const RuntimeConfig *config);
void runtime_seed_kv_main_regions(RuntimeState *state, const RuntimeConfig *config, int kv_dim);
int runtime_kv_has_legacy_float_backing(const RuntimeState *state);

// 生成 key/value row 时统一计算本层 float backing 起点。
float *runtime_key_cache_layer_ptr(RuntimeState *state, int layer_idx, int seq_len, int kv_dim);
float *runtime_value_cache_layer_ptr(RuntimeState *state, int layer_idx, int seq_len, int kv_dim);
float *runtime_key_cache_row_ptr(RuntimeState *state, int layer_idx, int pos, int seq_len, int kv_dim);
float *runtime_value_cache_row_ptr(RuntimeState *state, int layer_idx, int pos, int seq_len, int kv_dim);

// 当前阶段把 q/k/v 中间结果直接收口到 row 级 buffer。
void runtime_linear_qkv_row(RuntimeBackend *backend, float *q, float *k_row, float *v_row, const float *x, int layer_idx);
void runtime_clear_transient_kv(RuntimeState *state, int kv_dim);
void runtime_init_layer_kv_views(
    RuntimeModel *model,
    int layer_idx,
    RuntimeKvCacheLayerView *key_view,
    RuntimeKvCacheLayerView *value_view,
    int bind_hw_regions
);
void runtime_commit_kv_rows(
    RuntimeModel *model,
    RuntimeKvCacheLayerView *key_view,
    RuntimeKvCacheLayerView *value_view,
    int pos,
    const float *key_row,
    const float *value_row
);
void runtime_trace_kv_row(FILE *fp, const char *label, const RuntimeKvCacheLayerView *view, int time_idx);

// 部署/验证共用的单 token decode 调度器，返回 state->logits。
float *runtime_decode_transformer_step(RuntimeBackend *backend, int token, int pos);

#endif
