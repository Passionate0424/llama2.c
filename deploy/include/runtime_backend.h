#ifndef LLAMA2C_DEPLOY_RUNTIME_BACKEND_H
#define LLAMA2C_DEPLOY_RUNTIME_BACKEND_H

#include "runtime_types.h"

typedef struct RuntimeBackendOps {
    const char *name;
    int (*init)(RuntimeBackend *backend, RuntimeModel *model);
    void (*reset)(RuntimeBackend *backend);
    void (*destroy)(RuntimeBackend *backend);
    void (*rmsnorm)(RuntimeBackend *backend, float *out, const float *x, const float *weight, int size);
    void (*linear_qkv)(RuntimeBackend *backend, float *q, float *k, float *v, const float *x, int layer_idx);
    void (*linear_attn_o)(RuntimeBackend *backend, float *out, const float *x, int layer_idx);
    void (*qk_matmul)(RuntimeBackend *backend, float *att, const float *q, const float *key_cache, int pos, int head_idx);
    void (*softmax_row)(RuntimeBackend *backend, float *row, int size);
    void (*av_matmul)(RuntimeBackend *backend, float *out, const float *att, const float *value_cache, int pos, int head_idx);
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
    FILE *trace_fp;
    int trace_enabled;
};

int runtime_backend_bind_swref(RuntimeBackend *backend, RuntimeModel *model);
int runtime_backend_bind_hwstub(RuntimeBackend *backend, RuntimeModel *model);

#endif
