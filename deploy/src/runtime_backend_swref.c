#include "runtime_backend.h"

#include <math.h>
#include <string.h>

#include "runtime_common.h"

static int swref_group_size(RuntimeBackend *backend) {
    return backend->model->group_size;
}

static int swref_init(RuntimeBackend *backend, RuntimeModel *model) {
    backend->kind = RUNTIME_BACKEND_SWREF;
    backend->model = model;
    backend->trace_enabled = 0;
    backend->trace_fp = NULL;
    return 0;
}

static void swref_reset(RuntimeBackend *backend) {
    RuntimeState *state = &backend->model->state;
    int kv_dim = (backend->model->config.dim * backend->model->config.n_kv_heads) / backend->model->config.n_heads;
    // SW_REF 的 reset 目标很明确：清掉所有随 token 演化的运行时状态。
    memset(state->x, 0, (size_t)backend->model->config.dim * sizeof(float));
    memset(state->xb, 0, (size_t)backend->model->config.dim * sizeof(float));
    memset(state->xb2, 0, (size_t)backend->model->config.dim * sizeof(float));
    memset(state->hb, 0, (size_t)backend->model->config.hidden_dim * sizeof(float));
    memset(state->hb2, 0, (size_t)backend->model->config.hidden_dim * sizeof(float));
    memset(state->q, 0, (size_t)backend->model->config.dim * sizeof(float));
    memset(state->k, 0, (size_t)kv_dim * sizeof(float));
    memset(state->v, 0, (size_t)kv_dim * sizeof(float));
    memset(state->att, 0, (size_t)backend->model->config.n_heads * (size_t)backend->model->config.seq_len * sizeof(float));
    memset(state->logits, 0, (size_t)backend->model->config.vocab_size * sizeof(float));
    memset(state->key_cache, 0, (size_t)backend->model->config.n_layers * (size_t)backend->model->config.seq_len *
           (size_t)((backend->model->config.dim * backend->model->config.n_kv_heads) / backend->model->config.n_heads) * sizeof(float));
    memset(state->value_cache, 0, (size_t)backend->model->config.n_layers * (size_t)backend->model->config.seq_len *
           (size_t)((backend->model->config.dim * backend->model->config.n_kv_heads) / backend->model->config.n_heads) * sizeof(float));
}

static void swref_destroy(RuntimeBackend *backend) {
    (void)backend;
}

static void swref_rmsnorm(RuntimeBackend *backend, float *out, const float *x, const float *weight, int size) {
    (void)backend;
    rmsnorm_ref(out, x, weight, size);
}

static void swref_linear_qkv(RuntimeBackend *backend, float *q, float *k, float *v, const float *x, int layer_idx) {
    RuntimeModel *model = backend->model;
    RuntimeState *state = &model->state;
    int dim = model->config.dim;
    int kv_dim = (model->config.dim * model->config.n_kv_heads) / model->config.n_heads;
    // 这里保持和 runq 数学一致：先量化输入，再走 q80 matmul。
    memcpy(state->xb, x, (size_t)dim * sizeof(float));
    quantize_tensor(&state->xq, state->xb, dim, swref_group_size(backend));
    matmul_ref(q, &state->xq, model->weights.wq + layer_idx, dim, dim, swref_group_size(backend));
    matmul_ref(k, &state->xq, model->weights.wk + layer_idx, dim, kv_dim, swref_group_size(backend));
    matmul_ref(v, &state->xq, model->weights.wv + layer_idx, dim, kv_dim, swref_group_size(backend));
}

static void swref_linear_attn_o(RuntimeBackend *backend, float *out, const float *x, int layer_idx) {
    RuntimeModel *model = backend->model;
    int dim = model->config.dim;
    RuntimeState *state = &model->state;
    memcpy(state->xb, x, (size_t)dim * sizeof(float));
    quantize_tensor(&state->xq, state->xb, dim, swref_group_size(backend));
    matmul_ref(out, &state->xq, model->weights.wo + layer_idx, dim, dim, swref_group_size(backend));
}

static void swref_qk_matmul(RuntimeBackend *backend, float *att, const float *q, const float *key_cache, int pos, int head_idx) {
    RuntimeModel *model = backend->model;
    int dim = model->config.dim;
    int kv_dim = (model->config.dim * model->config.n_kv_heads) / model->config.n_heads;
    int kv_mul = model->config.n_heads / model->config.n_kv_heads;
    int head_size = dim / model->config.n_heads;
    const float *q_head = q + head_idx * head_size;
    // 当前只实现最小 decode 参考路径：单 head 的 QK^T 行得分。
    for (int t = 0; t <= pos; ++t) {
        const float *k_cached = key_cache + t * kv_dim + (head_idx / kv_mul) * head_size;
        float score = 0.0f;
        for (int i = 0; i < head_size; ++i) score += q_head[i] * k_cached[i];
        att[t] = score / sqrtf((float)head_size);
    }
}

static void swref_softmax_row(RuntimeBackend *backend, float *row, int size) {
    (void)backend;
    softmax_ref(row, size);
}

static void swref_av_matmul(RuntimeBackend *backend, float *out, const float *att, const float *value_cache, int pos, int head_idx) {
    RuntimeModel *model = backend->model;
    int dim = model->config.dim;
    int kv_dim = (model->config.dim * model->config.n_kv_heads) / model->config.n_heads;
    int kv_mul = model->config.n_heads / model->config.n_kv_heads;
    int head_size = dim / model->config.n_heads;
    memset(out, 0, (size_t)head_size * sizeof(float));
    for (int t = 0; t <= pos; ++t) {
        const float *v_cached = value_cache + t * kv_dim + (head_idx / kv_mul) * head_size;
        float a = att[t];
        for (int i = 0; i < head_size; ++i) out[i] += a * v_cached[i];
    }
}

static void swref_linear_ffn_w1(RuntimeBackend *backend, float *out, const float *x, int layer_idx) {
    RuntimeModel *model = backend->model;
    int dim = model->config.dim;
    RuntimeState *state = &model->state;
    memcpy(state->xb, x, (size_t)dim * sizeof(float));
    quantize_tensor(&state->xq, state->xb, dim, swref_group_size(backend));
    matmul_ref(out, &state->xq, model->weights.w1 + layer_idx, dim, model->config.hidden_dim, swref_group_size(backend));
}

static void swref_linear_ffn_w3(RuntimeBackend *backend, float *out, const float *x, int layer_idx) {
    RuntimeModel *model = backend->model;
    int dim = model->config.dim;
    RuntimeState *state = &model->state;
    memcpy(state->xb, x, (size_t)dim * sizeof(float));
    quantize_tensor(&state->xq, state->xb, dim, swref_group_size(backend));
    matmul_ref(out, &state->xq, model->weights.w3 + layer_idx, dim, model->config.hidden_dim, swref_group_size(backend));
}

static void swref_gate_mul(RuntimeBackend *backend, float *out, const float *w1_out, const float *w3_out, int size) {
    (void)backend;
    // FFN gate 保持和模型定义一致：SiLU(w1) * w3。
    for (int i = 0; i < size; ++i) {
        float val = w1_out[i];
        val *= (1.0f / (1.0f + expf(-val)));
        out[i] = val * w3_out[i];
    }
}

static void swref_linear_ffn_w2(RuntimeBackend *backend, float *out, const float *x, int layer_idx) {
    RuntimeModel *model = backend->model;
    RuntimeState *state = &model->state;
    memcpy(state->hb, x, (size_t)model->config.hidden_dim * sizeof(float));
    quantize_tensor(&state->hq, state->hb, model->config.hidden_dim, swref_group_size(backend));
    matmul_ref(out, &state->hq, model->weights.w2 + layer_idx, model->config.hidden_dim, model->config.dim, swref_group_size(backend));
}

static void swref_residual_add(RuntimeBackend *backend, float *dst, const float *src, int size) {
    (void)backend;
    for (int i = 0; i < size; ++i) dst[i] += src[i];
}

static void swref_final_norm(RuntimeBackend *backend, float *out, const float *x) {
    rmsnorm_ref(out, x, backend->model->weights.rms_final_weight, backend->model->config.dim);
}

static void swref_lm_head(RuntimeBackend *backend, float *logits, const float *x) {
    RuntimeModel *model = backend->model;
    RuntimeState *state = &model->state;
    memcpy(state->x, x, (size_t)model->config.dim * sizeof(float));
    quantize_tensor(&state->xq, state->x, model->config.dim, swref_group_size(backend));
    matmul_ref(logits, &state->xq, model->weights.wcls, model->config.dim, model->config.vocab_size, swref_group_size(backend));
}

static float *swref_forward_logits(RuntimeBackend *backend, int token, int pos) {
    // 第一版部署 CPU 路径直接复用完整 runq-compatible 前向，先保证端到端可跑。
    return runtime_forward_logits_swref(backend->model, token, pos, swref_group_size(backend));
}

static const RuntimeBackendOps SWREF_OPS = {
    "sw_ref",
    swref_init,
    swref_reset,
    swref_destroy,
    swref_rmsnorm,
    swref_linear_qkv,
    swref_linear_attn_o,
    swref_qk_matmul,
    swref_softmax_row,
    swref_av_matmul,
    swref_linear_ffn_w1,
    swref_linear_ffn_w3,
    swref_gate_mul,
    swref_linear_ffn_w2,
    swref_residual_add,
    swref_final_norm,
    swref_lm_head,
    swref_forward_logits,
};

int runtime_backend_bind_swref(RuntimeBackend *backend, RuntimeModel *model) {
    backend->ops = &SWREF_OPS;
    return backend->ops->init(backend, model);
}
