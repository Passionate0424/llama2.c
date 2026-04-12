#include "runtime_backend.h"
#include "runtime_hw_adapter.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "runtime_common.h"

static void stub_trace(RuntimeBackend *backend, const char *label) {
    if (backend->trace_enabled && backend->trace_fp) {
        fprintf(backend->trace_fp, "[HW_STUB] %s\n", label);
    }
}

static int hwstub_group_size(RuntimeBackend *backend) {
    // HW_STUB 与 SW_REF 必须共享同一组量化粒度，否则 compare 将失去意义。
    return backend->model->group_size;
}

static int hwstub_init(RuntimeBackend *backend, RuntimeModel *model) {
    runtime_hw_adapter_init();
    backend->kind = RUNTIME_BACKEND_HWSTUB;
    backend->model = model;
    backend->trace_enabled = 0;
    backend->trace_fp = NULL;
    return 0;
}

static void hwstub_reset(RuntimeBackend *backend) {
    // HW_STUB 第一版不做真实硬件计算，但保留调用链和 trace。
    stub_trace(backend, "reset");
    {
        RuntimeState *state = &backend->model->state;
        int dim = backend->model->config.dim;
        int kv_dim = (backend->model->config.dim * backend->model->config.n_kv_heads) / backend->model->config.n_heads;
        memset(state->x, 0, (size_t)dim * sizeof(float));
        memset(state->xb, 0, (size_t)dim * sizeof(float));
        memset(state->xb2, 0, (size_t)dim * sizeof(float));
        memset(state->q, 0, (size_t)dim * sizeof(float));
        memset(state->k, 0, (size_t)kv_dim * sizeof(float));
        memset(state->v, 0, (size_t)kv_dim * sizeof(float));
    }
}

static void hwstub_destroy(RuntimeBackend *backend) {
    stub_trace(backend, "destroy");
    (void)backend;
}

static void hwstub_rmsnorm(RuntimeBackend *backend, float *out, const float *x, const float *weight, int size) {
    stub_trace(backend, "rmsnorm");
    rmsnorm_ref(out, x, weight, size);
}

static void hwstub_linear_qkv(RuntimeBackend *backend, float *q, float *k, float *v, const float *x, int layer_idx) {
    RuntimeState *state = &backend->model->state;
    int dim = backend->model->config.dim;
    int kv_dim = (backend->model->config.dim * backend->model->config.n_kv_heads) / backend->model->config.n_heads;
    RuntimeHwLinearJob linear_job;
    stub_trace(backend, "linear_qkv");
    runtime_hw_adapter_trace_op("linear_qkv", layer_idx, dim);
    // HW_STUB 当前分两段执行：
    // 1. 先走 adapter/job 提交边界，验证硬件调用顺序；
    // 2. 再落回软件 q80 matmul，保证结果仍可与 SW_REF 逐项比较。
    if (runtime_hw_prepare_linear_job(&linear_job, "linear_qkv", layer_idx, (size_t)dim) == 0) {
        (void)runtime_hw_submit_job("linear", &linear_job);
        (void)runtime_hw_wait_done(0u);
    }
    memcpy(state->xb, x, (size_t)dim * sizeof(float));
    quantize_tensor(&state->xq, state->xb, dim, hwstub_group_size(backend));
    matmul_ref(q, &state->xq, backend->model->weights.wq + layer_idx, dim, dim, hwstub_group_size(backend));
    matmul_ref(k, &state->xq, backend->model->weights.wk + layer_idx, dim, kv_dim, hwstub_group_size(backend));
    matmul_ref(v, &state->xq, backend->model->weights.wv + layer_idx, dim, kv_dim, hwstub_group_size(backend));
}

static void hwstub_linear_attn_o(RuntimeBackend *backend, float *out, const float *x, int layer_idx) {
    RuntimeState *state = &backend->model->state;
    int dim = backend->model->config.dim;
    stub_trace(backend, "linear_attn_o");
    runtime_hw_adapter_trace_op("linear_attn_o", layer_idx, dim);
    memcpy(state->xb, x, (size_t)dim * sizeof(float));
    quantize_tensor(&state->xq, state->xb, dim, hwstub_group_size(backend));
    matmul_ref(out, &state->xq, backend->model->weights.wo + layer_idx, dim, dim, hwstub_group_size(backend));
}

static void hwstub_qk_matmul(RuntimeBackend *backend, float *att, const float *q, const RuntimeKvCacheLayerView *key_view, int pos, int head_idx) {
    RuntimeModel *model = backend->model;
    int dim = model->config.dim;
    int head_size = dim / model->config.n_heads;
    const float *q_head = q + head_idx * head_size;
    stub_trace(backend, "qk_matmul");
    runtime_hw_adapter_trace_op("qk_matmul", head_idx, pos + 1);
    for (int t = 0; t <= pos; ++t) {
        const float *k_cached = runtime_kv_cache_head_ptr(key_view, t, head_idx);
        float score = 0.0f;
        for (int i = 0; i < head_size; ++i) score += q_head[i] * k_cached[i];
        att[t] = score / sqrtf((float)head_size);
    }
}

static void hwstub_softmax_row(RuntimeBackend *backend, float *row, int size) {
    stub_trace(backend, "softmax_row");
    runtime_hw_adapter_trace_op("softmax_row", 0, size);
    softmax_ref(row, size);
}

static void hwstub_av_matmul(RuntimeBackend *backend, float *out, const float *att, const RuntimeKvCacheLayerView *value_view, int pos, int head_idx) {
    RuntimeModel *model = backend->model;
    int dim = model->config.dim;
    int head_size = dim / model->config.n_heads;
    stub_trace(backend, "av_matmul");
    runtime_hw_adapter_trace_op("av_matmul", head_idx, pos + 1);
    memset(out, 0, (size_t)head_size * sizeof(float));
    for (int t = 0; t <= pos; ++t) {
        const float *v_cached = runtime_kv_cache_head_ptr(value_view, t, head_idx);
        for (int i = 0; i < head_size; ++i) out[i] += att[t] * v_cached[i];
    }
}

static void hwstub_linear_ffn_w1(RuntimeBackend *backend, float *out, const float *x, int layer_idx) {
    RuntimeState *state = &backend->model->state;
    int dim = backend->model->config.dim;
    stub_trace(backend, "linear_ffn_w1");
    runtime_hw_adapter_trace_op("linear_ffn_w1", layer_idx, dim);
    memcpy(state->xb, x, (size_t)dim * sizeof(float));
    quantize_tensor(&state->xq, state->xb, dim, hwstub_group_size(backend));
    matmul_ref(out, &state->xq, backend->model->weights.w1 + layer_idx, dim, backend->model->config.hidden_dim, hwstub_group_size(backend));
}

static void hwstub_linear_ffn_w3(RuntimeBackend *backend, float *out, const float *x, int layer_idx) {
    RuntimeState *state = &backend->model->state;
    int dim = backend->model->config.dim;
    stub_trace(backend, "linear_ffn_w3");
    runtime_hw_adapter_trace_op("linear_ffn_w3", layer_idx, dim);
    memcpy(state->xb, x, (size_t)dim * sizeof(float));
    quantize_tensor(&state->xq, state->xb, dim, hwstub_group_size(backend));
    matmul_ref(out, &state->xq, backend->model->weights.w3 + layer_idx, dim, backend->model->config.hidden_dim, hwstub_group_size(backend));
}

static void hwstub_gate_mul(RuntimeBackend *backend, float *out, const float *w1_out, const float *w3_out, int size) {
    stub_trace(backend, "gate_mul");
    runtime_hw_adapter_trace_op("gate_mul", 0, size);
    for (int i = 0; i < size; ++i) {
        float val = w1_out[i];
        val *= (1.0f / (1.0f + expf(-val)));
        out[i] = val * w3_out[i];
    }
}

static void hwstub_linear_ffn_w2(RuntimeBackend *backend, float *out, const float *x, int layer_idx) {
    RuntimeState *state = &backend->model->state;
    stub_trace(backend, "linear_ffn_w2");
    runtime_hw_adapter_trace_op("linear_ffn_w2", layer_idx, backend->model->config.hidden_dim);
    memcpy(state->hb, x, (size_t)backend->model->config.hidden_dim * sizeof(float));
    quantize_tensor(&state->hq, state->hb, backend->model->config.hidden_dim, hwstub_group_size(backend));
    matmul_ref(out, &state->hq, backend->model->weights.w2 + layer_idx, backend->model->config.hidden_dim, backend->model->config.dim, hwstub_group_size(backend));
}

static void hwstub_residual_add(RuntimeBackend *backend, float *dst, const float *src, int size) {
    stub_trace(backend, "residual_add");
    runtime_hw_adapter_trace_op("residual_add", 0, size);
    for (int i = 0; i < size; ++i) dst[i] += src[i];
}

static void hwstub_final_norm(RuntimeBackend *backend, float *out, const float *x) {
    stub_trace(backend, "final_norm");
    runtime_hw_adapter_trace_op("final_norm", 0, backend->model->config.dim);
    rmsnorm_ref(out, x, backend->model->weights.rms_final_weight, backend->model->config.dim);
}

static void hwstub_lm_head(RuntimeBackend *backend, float *logits, const float *x) {
    RuntimeState *state = &backend->model->state;
    stub_trace(backend, "lm_head");
    runtime_hw_adapter_trace_op("lm_head", 0, backend->model->config.dim);
    // 最终分类头也保持与部署态一致的量化口径，避免“前面像硬件、最后一步变纯 float”。
    memcpy(state->x, x, (size_t)backend->model->config.dim * sizeof(float));
    quantize_tensor(&state->xq, state->x, backend->model->config.dim, hwstub_group_size(backend));
    matmul_ref(logits, &state->xq, backend->model->weights.wcls, backend->model->config.dim, backend->model->config.vocab_size, hwstub_group_size(backend));
}

static const RuntimeBackendOps HWSTUB_OPS = {
    "hw_stub",
    hwstub_init,
    hwstub_reset,
    hwstub_destroy,
    hwstub_rmsnorm,
    hwstub_linear_qkv,
    hwstub_linear_attn_o,
    hwstub_qk_matmul,
    hwstub_softmax_row,
    hwstub_av_matmul,
    hwstub_linear_ffn_w1,
    hwstub_linear_ffn_w3,
    hwstub_gate_mul,
    hwstub_linear_ffn_w2,
    hwstub_residual_add,
    hwstub_final_norm,
    hwstub_lm_head,
};

int runtime_backend_bind_hwstub(RuntimeBackend *backend, RuntimeModel *model) {
    backend->ops = &HWSTUB_OPS;
    return backend->ops->init(backend, model);
}
