#ifndef LLAMA2C_DEPLOY_RUNTIME_TYPES_H
#define LLAMA2C_DEPLOY_RUNTIME_TYPES_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

// 这里统一定义部署版/验证版共用的数据结构，避免后续多份 C 文件各自复制一套。

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} RuntimeConfig;

typedef struct {
    int8_t *q;
    float *s;
} QuantizedTensor;

typedef struct {
    QuantizedTensor *q_tokens;
    float *rms_att_weight;
    float *rms_ffn_weight;
    QuantizedTensor *wq;
    QuantizedTensor *wk;
    QuantizedTensor *wv;
    QuantizedTensor *wo;
    QuantizedTensor *w1;
    QuantizedTensor *w2;
    QuantizedTensor *w3;
    float *rms_final_weight;
    QuantizedTensor *wcls;
} RuntimeWeights;

typedef struct {
    float *x;
    float *xb;
    float *xb2;
    float *hb;
    float *hb2;
    QuantizedTensor xq;
    QuantizedTensor hq;
    float *q;
    float *k;
    float *v;
    float *att;
    float *logits;
    float *key_cache;
    float *value_cache;
} RuntimeState;

typedef struct {
    RuntimeConfig config;
    RuntimeWeights weights;
    RuntimeState state;
    const unsigned char *raw_model_data;
    size_t raw_model_size;
    int group_size;
} RuntimeModel;

typedef struct {
    unsigned char *base;
    size_t size;
    size_t used;
} RuntimeArena;

// 共享缓冲必须显式区分 CPU 访问地址和给 DMA/MMIO 编程用的物理地址。
typedef struct {
    // CPU 实际可解引用的指针。
    // 在 host / HW_STUB 环境下通常指向本地静态数组；
    // 在 SoC 环境下可替换成 uncached alias 对应的虚拟地址映射。
    void *cpu_ptr;
    // CPU 使用的 uncached alias 地址口径（用于日志、寄存器编程参数准备）。
    uintptr_t cpu_uncached_addr;
    // DMA / MMIO 编程使用的物理总线地址口径。
    uint32_t phys_addr;
    size_t size;
} SharedBufferDesc;

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char **vocab;
    float *vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
} RuntimeTokenizer;

typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex *probindex;
    float temperature;
    float topp;
    int top_k;
    float repetition_penalty;
    int no_repeat_ngram_size;
    unsigned long long rng_state;
} RuntimeSampler;

typedef enum {
    RUNTIME_BACKEND_SWREF = 0,
    RUNTIME_BACKEND_HWSTUB = 1,
} RuntimeBackendKind;

typedef enum {
    RUNTIME_MODE_GENERATE = 0,
    RUNTIME_MODE_CHAT = 1,
} RuntimeRunMode;

typedef struct {
    float temperature;
    float top_p;
    int top_k;
    float repetition_penalty;
    int no_repeat_ngram_size;
    int max_new_tokens;
} RuntimeDecodeConfig;

typedef struct {
    const char *label;
    float max_abs_err;
    float mean_abs_err;
    int mismatch_count;
    int elem_count;
} VerifyMetric;

struct RuntimeBackend;

typedef struct RuntimeBackend RuntimeBackend;

#endif
