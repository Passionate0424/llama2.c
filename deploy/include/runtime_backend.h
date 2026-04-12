#ifndef LLAMA2C_DEPLOY_RUNTIME_BACKEND_H
#define LLAMA2C_DEPLOY_RUNTIME_BACKEND_H

#include "runtime_types.h"

// backend 接口被收敛成“算子级”边界：
// - frontend / decode 调度器只负责决定执行顺序；
// - backend 负责给出每个算子的具体实现；
// - 这样 CPU 参考实现与 HW_STUB/未来真实硬件后端可以复用同一条 decode 主链。
//
// 当前阶段最重要的接口收敛是：
// - qk_matmul / av_matmul 不再长期接受裸 float* 历史 KV；
// - runtime 先把历史 KV 收口成 layer view，再由 backend 通过统一解释规则访问。
typedef struct RuntimeBackendOps {
    const char *name;
    int (*init)(RuntimeBackend *backend, RuntimeModel *model);
    void (*reset)(RuntimeBackend *backend);
    void (*destroy)(RuntimeBackend *backend);

    // Attention/FFN 主路径算子。
    void (*rmsnorm)(RuntimeBackend *backend, float *out, const float *x, const float *weight, int size);
    void (*linear_qkv)(RuntimeBackend *backend, float *q, float *k, float *v, const float *x, int layer_idx);
    void (*linear_attn_o)(RuntimeBackend *backend, float *out, const float *x, int layer_idx);
    void (*qk_matmul)(RuntimeBackend *backend, float *att, const float *q, const RuntimeKvCacheLayerView *key_view, int pos, int head_idx);
    void (*softmax_row)(RuntimeBackend *backend, float *row, int size);
    void (*av_matmul)(RuntimeBackend *backend, float *out, const float *att, const RuntimeKvCacheLayerView *value_view, int pos, int head_idx);
    void (*linear_ffn_w1)(RuntimeBackend *backend, float *out, const float *x, int layer_idx);
    void (*linear_ffn_w3)(RuntimeBackend *backend, float *out, const float *x, int layer_idx);
    void (*gate_mul)(RuntimeBackend *backend, float *out, const float *w1_out, const float *w3_out, int size);
    void (*linear_ffn_w2)(RuntimeBackend *backend, float *out, const float *x, int layer_idx);
    void (*residual_add)(RuntimeBackend *backend, float *dst, const float *src, int size);
    void (*final_norm)(RuntimeBackend *backend, float *out, const float *x);
    void (*lm_head)(RuntimeBackend *backend, float *logits, const float *x);
} RuntimeBackendOps;

struct RuntimeBackend {
    RuntimeBackendKind kind;
    RuntimeModel *model;
    const RuntimeBackendOps *ops;
    // trace_fp / trace_enabled 主要给 HW_STUB / adapter 调试链路使用。
    FILE *trace_fp;
    int trace_enabled;
};

// 绑定软件参考后端。该后端以 q80 数学路径为准，便于和部署实现保持一致。
int runtime_backend_bind_swref(RuntimeBackend *backend, RuntimeModel *model);

// 绑定硬件桩后端。该后端当前仍用软件结果，但保留硬件提交/等待边界。
int runtime_backend_bind_hwstub(RuntimeBackend *backend, RuntimeModel *model);

#endif
