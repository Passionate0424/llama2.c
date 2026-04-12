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
    // q80 量化后的 int8 数据区。
    int8_t *q;
    // 每个 group 对应一份缩放系数。
    float *s;
} QuantizedTensor;

// 共享缓冲必须显式区分 CPU 访问地址和给 DMA/MMIO 编程用的物理地址。
typedef struct {
    // CPU 实际可解引用的指针。
    // 在 intel_host_stub / HW_STUB 环境下通常指向本地静态数组；
    // 在 SoC 环境下可替换成 uncached alias 对应的虚拟地址映射。
    void *cpu_ptr;
    // CPU 使用的 uncached alias 地址口径（用于日志、寄存器编程参数准备）。
    uintptr_t cpu_uncached_addr;
    // DMA / MMIO 编程使用的物理总线地址口径。
    uint32_t phys_addr;
    size_t size;
} SharedBufferDesc;

// KV 访问视图：
// 1) 先把 backend 对历史 KV 的正式输入从裸 float* 收口成 view；
// 2) 当前阶段仍允许用 legacy_float_data 绑定旧的 float backing，便于逐步迁移；
// 3) 后续切到 KV_MAIN int8 data + scale metadata 时，优先替换 view 的解释逻辑，
//    而不是再次修改 backend 算子签名。
typedef struct {
    const SharedBufferDesc *data_region;
    const SharedBufferDesc *scale_region;
    // 迁移期保留的 float backing，仅供 SW_REF/HW_STUB 与旧验证路径复用。
    float *legacy_float_data;
    int seq_len;
    int kv_dim;
    int head_size;
    int kv_mul;
    size_t data_stride_bytes;
    size_t scale_stride_bytes;
} RuntimeKvCacheLayerView;

typedef struct {
    // token embedding，当前按整张 q80 表原位映射。
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
    // 若 shared_classifier 为真，wcls 会与 q_tokens 共享底层资产。
    QuantizedTensor *wcls;
} RuntimeWeights;

typedef struct {
    // x 表示当前 token 在各层之间流动的主残差流。
    float *x;
    // xb / xb2 是 attention 与 FFN 子图复用的中间缓冲。
    float *xb;
    float *xb2;
    // hb / hb2 是 FFN hidden 维度上的中间结果。
    float *hb;
    float *hb2;
    // xq / hq 是激活侧量化后的临时视图，不拥有独立生命周期。
    QuantizedTensor xq;
    QuantizedTensor hq;
    // 当前 token 的 q 向量。
    float *q;
    // 当前阶段仍暂存局部 k/v 中间结果；
    // 后续 decode 主路径会逐步收敛为直接写入最终槽位，不再依赖这两个字段作为正式接口。
    float *k;
    float *v;
    // attention 缓冲按 [head][time] 组织。
    float *att;
    // 最终 lm_head 输出的 logits。
    float *logits;
    // 完整历史 KV 的旧 float backing。
    // 当前只允许作为迁移期内部存储，不允许继续作为 backend/verify 的正式裸指针合同。
    float *key_cache;
    float *value_cache;
    // KV_MAIN 一级窗口下的二级子区描述对象。
    SharedBufferDesc key_cache_main_data_region;
    SharedBufferDesc key_cache_main_scale_region;
    SharedBufferDesc value_cache_main_data_region;
    SharedBufferDesc value_cache_main_scale_region;
} RuntimeState;

typedef struct {
    RuntimeConfig config;
    RuntimeWeights weights;
    RuntimeState state;
    // 原始模型头文件数组及其大小，主要给布局校验使用。
    const unsigned char *raw_model_data;
    size_t raw_model_size;
    // q80 权重/激活量化采用的 group size。
    int group_size;
} RuntimeModel;

typedef struct {
    unsigned char *base;
    size_t size;
    size_t used;
} RuntimeArena;

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
    int debug_dump_first_step;
    int sample_calls;
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
