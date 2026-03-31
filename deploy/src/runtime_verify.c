#include "runtime_verify.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "runtime_assets.h"
#include "runtime_common.h"

static VerifyMetric compare_tensor(const char *label, const float *ref, const float *dut, int n, float tol) {
    VerifyMetric metric;
    float max_abs = 0.0f;
    float sum_abs = 0.0f;
    int mismatch = 0;
    metric.label = label;
    metric.elem_count = n;
    for (int i = 0; i < n; ++i) {
        float diff = fabsf(ref[i] - dut[i]);
        if (diff > max_abs) max_abs = diff;
        sum_abs += diff;
        if (diff > tol) mismatch++;
    }
    metric.max_abs_err = max_abs;
    metric.mean_abs_err = (n > 0) ? (sum_abs / n) : 0.0f;
    metric.mismatch_count = mismatch;
    return metric;
}

static void print_metric(const VerifyMetric *metric, float tol) {
    printf("[VERIFY] %-14s max=%.6f mean=%.6f mismatch=%d/%d tol=%.6f %s\n",
        metric->label,
        metric->max_abs_err,
        metric->mean_abs_err,
        metric->mismatch_count,
        metric->elem_count,
        tol,
        metric->mismatch_count == 0 ? "PASS" : "FAIL");
}

int runtime_run_verify_suite(void) {
    RuntimeModel model;
    RuntimeBackend swref;
    RuntimeBackend hwstub;
    RuntimeState *state;
    float *tmp_a = NULL;
    float *tmp_b = NULL;
    float *tmp_c = NULL;
    float *tmp_d = NULL;
    int dim;
    int hidden_dim;
    int kv_dim;
    int head_size;
    int fail_count = 0;
    const float tol = 1e-5f;

    runtime_model_init(&model);
    if (runtime_load_default_model(&model) != 0) return 1;
    runtime_backend_bind_swref(&swref, &model);
    runtime_backend_bind_hwstub(&hwstub, &model);
    swref.trace_enabled = 0;
    hwstub.trace_enabled = 1;
    hwstub.trace_fp = stdout;

    state = &model.state;
    dim = model.config.dim;
    hidden_dim = model.config.hidden_dim;
    kv_dim = (model.config.dim * model.config.n_kv_heads) / model.config.n_heads;
    head_size = dim / model.config.n_heads;

    tmp_a = (float *)calloc((size_t)hidden_dim > (size_t)dim ? (size_t)hidden_dim : (size_t)dim, sizeof(float));
    tmp_b = (float *)calloc((size_t)hidden_dim > (size_t)dim ? (size_t)hidden_dim : (size_t)dim, sizeof(float));
    tmp_c = (float *)calloc((size_t)hidden_dim > (size_t)dim ? (size_t)hidden_dim : (size_t)dim, sizeof(float));
    tmp_d = (float *)calloc((size_t)hidden_dim > (size_t)dim ? (size_t)hidden_dim : (size_t)dim, sizeof(float));
    if (!tmp_a || !tmp_b || !tmp_c || !tmp_d) {
        fprintf(stderr, "runtime_run_verify_suite: 临时缓冲分配失败\n");
        fail_count = 1;
        goto cleanup;
    }

    memcpy(tmp_a, model.weights.token_embedding_table, (size_t)dim * sizeof(float));

    swref.ops->rmsnorm(&swref, tmp_b, tmp_a, model.weights.rms_att_weight, dim);
    hwstub.ops->rmsnorm(&hwstub, tmp_c, tmp_a, model.weights.rms_att_weight, dim);
    {
        VerifyMetric metric = compare_tensor("rmsnorm", tmp_b, tmp_c, dim, tol);
        print_metric(&metric, tol);
        if (metric.mismatch_count) fail_count++;
    }

    swref.ops->linear_qkv(&swref, state->q, state->k, state->v, tmp_b, 0);
    hwstub.ops->linear_qkv(&hwstub, tmp_c, tmp_d, tmp_a, tmp_b, 0);
    {
        VerifyMetric metric = compare_tensor("linear_qkv_q", state->q, tmp_c, dim, tol);
        print_metric(&metric, tol);
        if (metric.mismatch_count) fail_count++;
    }

    memcpy(state->key_cache, state->k, (size_t)kv_dim * sizeof(float));
    swref.ops->qk_matmul(&swref, tmp_b, state->q, state->key_cache, 0, 0);
    hwstub.ops->qk_matmul(&hwstub, tmp_c, state->q, state->key_cache, 0, 0);
    {
        VerifyMetric metric = compare_tensor("qk_matmul", tmp_b, tmp_c, 1, tol);
        print_metric(&metric, tol);
        if (metric.mismatch_count) fail_count++;
    }

    tmp_b[0] = 1.0f; tmp_b[1] = 0.5f; tmp_b[2] = -0.25f; tmp_b[3] = 0.25f;
    memcpy(tmp_c, tmp_b, 4 * sizeof(float));
    swref.ops->softmax_row(&swref, tmp_b, 4);
    hwstub.ops->softmax_row(&hwstub, tmp_c, 4);
    {
        VerifyMetric metric = compare_tensor("softmax_row", tmp_b, tmp_c, 4, tol);
        print_metric(&metric, tol);
        if (metric.mismatch_count) fail_count++;
    }

    for (int i = 0; i < hidden_dim; ++i) {
        tmp_b[i] = sinf((float)i * 0.1f);
        tmp_c[i] = cosf((float)i * 0.1f);
    }
    swref.ops->gate_mul(&swref, tmp_d, tmp_b, tmp_c, hidden_dim);
    hwstub.ops->gate_mul(&hwstub, tmp_a, tmp_b, tmp_c, hidden_dim);
    {
        VerifyMetric metric = compare_tensor("gate_mul", tmp_d, tmp_a, hidden_dim, tol);
        print_metric(&metric, tol);
        if (metric.mismatch_count) fail_count++;
    }

    for (int i = 0; i < dim; ++i) {
        tmp_b[i] = (float)i * 0.01f;
        tmp_c[i] = (float)i * 0.02f;
    }
    memcpy(tmp_d, tmp_b, (size_t)dim * sizeof(float));
    memcpy(tmp_a, tmp_b, (size_t)dim * sizeof(float));
    swref.ops->residual_add(&swref, tmp_d, tmp_c, dim);
    hwstub.ops->residual_add(&hwstub, tmp_a, tmp_c, dim);
    {
        VerifyMetric metric = compare_tensor("residual_add", tmp_d, tmp_a, dim, tol);
        print_metric(&metric, tol);
        if (metric.mismatch_count) fail_count++;
    }

cleanup:
    free(tmp_a);
    free(tmp_b);
    free(tmp_c);
    free(tmp_d);
    runtime_model_destroy(&model);
    return fail_count == 0 ? 0 : 1;
}
