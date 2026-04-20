#include "runtime_common.h"
#include "runtime_backend.h"
#include "runtime_hw_adapter.h"
#include "runtime_memory_map.h"

#include <ctype.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// 部署版第一阶段直接把运行时工作区收口成固定 arena。
// 这样可以避免沿用 OpenLA500 现有 runq.c 那种零散 calloc/malloc 风格，
// 更接近后续 SoC 上“静态资产 + arena + shared RAM”的落地形态。
static unsigned char g_runtime_arena_storage[RUNTIME_ARENA_SIZE]
#if defined(__GNUC__)
__attribute__((section(".runtime_arena"), aligned(64)))
#endif
;

static RuntimeArena g_runtime_arena;
static int g_runtime_arena_inited = 0;

int grouped_scale_count(int n, int group_size) {
    if (group_size <= 0 || n <= 0) return 0;
    return (n + group_size - 1) / group_size;
}

static void runtime_prepare_arena(void) {
    // arena 以“整次模型加载/初始化”为粒度复位，
    // 不支持运行中碎片化回收，这正是部署态想要的固定布局行为。
    if (!g_runtime_arena_inited) {
        arena_init(&g_runtime_arena, g_runtime_arena_storage, sizeof(g_runtime_arena_storage));
        g_runtime_arena_inited = 1;
    }
    arena_reset(&g_runtime_arena);
}

void arena_init(RuntimeArena *arena, unsigned char *base, size_t size) {
    arena->base = base;
    arena->size = size;
    arena->used = 0;
}

void arena_reset(RuntimeArena *arena) {
    arena->used = 0;
}

void *arena_alloc(RuntimeArena *arena, size_t size, size_t align) {
    size_t aligned = arena->used;
    size_t mask = align - 1;
    if (align != 0 && (aligned & mask) != 0) {
        aligned = (aligned + mask) & ~mask;
    }
    if (aligned + size > arena->size) {
        return NULL;
    }
    arena->used = aligned + size;
    return arena->base + aligned;
}

void runtime_model_init(RuntimeModel *model) {
    memset(model, 0, sizeof(*model));
}

void runtime_model_destroy(RuntimeModel *model) {
    if (model->weights.q_tokens) free(model->weights.q_tokens);
    if (model->weights.wq) free(model->weights.wq);
    if (model->weights.wk) free(model->weights.wk);
    if (model->weights.wv) free(model->weights.wv);
    if (model->weights.wo) free(model->weights.wo);
    if (model->weights.w1) free(model->weights.w1);
    if (model->weights.w2) free(model->weights.w2);
    if (model->weights.w3) free(model->weights.w3);
    if (model->weights.wcls && model->weights.wcls != model->weights.q_tokens) {
        free(model->weights.wcls);
    }
    runtime_free_state(&model->state);
    memset(model, 0, sizeof(*model));
}

void runtime_malloc_state(RuntimeState *state, const RuntimeConfig *config, int group_size) {
    int kv_dim = (config->dim * config->n_kv_heads) / config->n_heads;
    int max_kv_dim = (RUNTIME_MODEL_MAX_DIM * RUNTIME_MODEL_MAX_KV_HEADS) / RUNTIME_MODEL_MAX_HEADS;
    int xq_scale_count = grouped_scale_count(config->dim, group_size);
    int hq_scale_count = grouped_scale_count(config->hidden_dim, group_size);
    void *ptr = NULL;

    if (group_size <= 0 ||
        config->dim > RUNTIME_MODEL_MAX_DIM ||
        config->hidden_dim > RUNTIME_MODEL_MAX_HIDDEN ||
        config->n_layers > RUNTIME_MODEL_MAX_LAYERS ||
        config->seq_len > RUNTIME_MODEL_MAX_SEQ_LEN ||
        config->vocab_size > RUNTIME_MODEL_MAX_VOCAB ||
        kv_dim > max_kv_dim ||
        xq_scale_count <= 0 ||
        hq_scale_count <= 0) {
        fprintf(stderr, "runtime_malloc_state: 当前模型或 group_size 超过固定 arena 上限，请先更新内存规划\n");
        exit(EXIT_FAILURE);
    }

    runtime_prepare_arena();
    memset(state, 0, sizeof(*state));
    ptr = arena_alloc(&g_runtime_arena, (size_t)config->dim * sizeof(float), 64);
    state->x = (float *)ptr;
    ptr = arena_alloc(&g_runtime_arena, (size_t)config->dim * sizeof(float), 64);
    state->xb = (float *)ptr;
    ptr = arena_alloc(&g_runtime_arena, (size_t)config->dim * sizeof(float), 64);
    state->xb2 = (float *)ptr;
    ptr = arena_alloc(&g_runtime_arena, (size_t)config->hidden_dim * sizeof(float), 64);
    state->hb = (float *)ptr;
    ptr = arena_alloc(&g_runtime_arena, (size_t)config->hidden_dim * sizeof(float), 64);
    state->hb2 = (float *)ptr;
    ptr = arena_alloc(&g_runtime_arena, (size_t)config->dim * sizeof(int8_t), 64);
    state->xq.q = (int8_t *)ptr;
    ptr = arena_alloc(&g_runtime_arena, (size_t)xq_scale_count * sizeof(float), 64);
    state->xq.s = (float *)ptr;
    ptr = arena_alloc(&g_runtime_arena, (size_t)config->hidden_dim * sizeof(int8_t), 64);
    state->hq.q = (int8_t *)ptr;
    ptr = arena_alloc(&g_runtime_arena, (size_t)hq_scale_count * sizeof(float), 64);
    state->hq.s = (float *)ptr;
    ptr = arena_alloc(&g_runtime_arena, (size_t)config->dim * sizeof(float), 64);
    state->q = (float *)ptr;
    ptr = arena_alloc(&g_runtime_arena, (size_t)kv_dim * sizeof(float), 64);
    state->k = (float *)ptr;
    ptr = arena_alloc(&g_runtime_arena, (size_t)kv_dim * sizeof(float), 64);
    state->v = (float *)ptr;
    ptr = arena_alloc(&g_runtime_arena, (size_t)config->n_heads * (size_t)config->seq_len * sizeof(float), 64);
    state->att = (float *)ptr;
    ptr = arena_alloc(&g_runtime_arena, (size_t)config->vocab_size * sizeof(float), 64);
    state->logits = (float *)ptr;
    ptr = arena_alloc(&g_runtime_arena, (size_t)config->n_layers * (size_t)config->seq_len * (size_t)kv_dim * sizeof(float), 64);
    state->key_cache = (float *)ptr;
    ptr = arena_alloc(&g_runtime_arena, (size_t)config->n_layers * (size_t)config->seq_len * (size_t)kv_dim * sizeof(float), 64);
    state->value_cache = (float *)ptr;

    runtime_refresh_kv_main_regions(state, config);
    runtime_seed_kv_main_regions(state, config, kv_dim);

    // 当前 token 的 k/v row buffer 最终会直接提交到 KV_MAIN float 数据面。
    state->key_cache_main_data_region.cpu_ptr = state->key_cache;
    state->value_cache_main_data_region.cpu_ptr = state->value_cache;

    if (!state->x || !state->xb || !state->xb2 || !state->hb || !state->hb2 ||
        !state->xq.q || !state->xq.s || !state->hq.q || !state->hq.s ||
        !state->q || !state->k || !state->v || !state->att || !state->logits ||
        !state->key_cache || !state->value_cache) {
        fprintf(stderr, "runtime_malloc_state: arena 空间不足，请增大 RUNTIME_ARENA_SIZE\n");
        exit(EXIT_FAILURE);
    }

    // arena 不保证清零，这里显式清空，保证行为与原先 calloc 一致。
    memset(state->x, 0, (size_t)config->dim * sizeof(float));
    memset(state->xb, 0, (size_t)config->dim * sizeof(float));
    memset(state->xb2, 0, (size_t)config->dim * sizeof(float));
    memset(state->hb, 0, (size_t)config->hidden_dim * sizeof(float));
    memset(state->hb2, 0, (size_t)config->hidden_dim * sizeof(float));
    memset(state->xq.q, 0, (size_t)config->dim * sizeof(int8_t));
    memset(state->xq.s, 0, (size_t)xq_scale_count * sizeof(float));
    memset(state->hq.q, 0, (size_t)config->hidden_dim * sizeof(int8_t));
    memset(state->hq.s, 0, (size_t)hq_scale_count * sizeof(float));
    memset(state->q, 0, (size_t)config->dim * sizeof(float));
    memset(state->k, 0, (size_t)kv_dim * sizeof(float));
    memset(state->v, 0, (size_t)kv_dim * sizeof(float));
    memset(state->att, 0, (size_t)config->n_heads * (size_t)config->seq_len * sizeof(float));
    memset(state->logits, 0, (size_t)config->vocab_size * sizeof(float));
    memset(state->key_cache, 0, (size_t)config->n_layers * (size_t)config->seq_len * (size_t)kv_dim * sizeof(float));
    memset(state->value_cache, 0, (size_t)config->n_layers * (size_t)config->seq_len * (size_t)kv_dim * sizeof(float));
}

void runtime_free_state(RuntimeState *state) {
    // arena 模式下，这里不 free 单个指针，只做结构清零。
    memset(state, 0, sizeof(*state));
}

void dequantize_tensor(QuantizedTensor *qx, float *x, int n, int group_size) {
    if (group_size <= 0) {
        fprintf(stderr, "dequantize_tensor: 非法 group_size=%d\n", group_size);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n; ++i) {
        x[i] = (float)qx->q[i] * qx->s[i / group_size];
    }
}

void quantize_tensor(QuantizedTensor *qx, float *x, int n, int group_size) {
    if (group_size <= 0) {
        fprintf(stderr, "quantize_tensor: 非法 group_size=%d\n", group_size);
        exit(EXIT_FAILURE);
    }
    // 激活侧允许最后一个 group 不满，这和当前 260K hidden_dim=172 的实际情况一致。
    // 因此这里按 ceil(n / group_size) 处理，不再强行要求整除。
    int num_groups = (n + group_size - 1) / group_size;
    const float qmax = 127.0f;
    for (int group = 0; group < num_groups; ++group) {
        int start = group * group_size;
        int end = start + group_size;
        if (end > n) end = n;
        float wmax = 0.0f;
        for (int i = start; i < end; ++i) {
            float val = fabsf(x[i]);
            if (val > wmax) wmax = val;
        }
        float scale = wmax / qmax;
        if (scale == 0.0f) scale = 1.0f / qmax;
        qx->s[group] = scale;
        for (int i = start; i < end; ++i) {
            qx->q[i] = (int8_t)roundf(x[i] / scale);
        }
    }
}

QuantizedTensor *init_quantized_tensors(void **ptr, int n, int size_each, int group_size) {
    if (group_size <= 0 || (size_each % group_size) != 0) {
        fprintf(stderr, "init_quantized_tensors: 非法 group_size=%d 或 size_each=%d 不能整除\n", group_size, size_each);
        exit(EXIT_FAILURE);
    }
    void *p = *ptr;
    QuantizedTensor *res = (QuantizedTensor *)malloc((size_t)n * sizeof(QuantizedTensor));
    if (!res) {
        fprintf(stderr, "init_quantized_tensors: QuantizedTensor 元数据分配失败\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n; ++i) {
        res[i].q = (int8_t *)p;
        p = (int8_t *)p + size_each;
        res[i].s = (float *)p;
        p = (float *)p + grouped_scale_count(size_each, group_size);
    }
    *ptr = p;
    return res;
}

void runtime_init_kv_cache_layer_view(
    RuntimeKvCacheLayerView *view,
    const SharedBufferDesc *data_region,
    int seq_len,
    int kv_dim,
    int head_size,
    int kv_mul,
    size_t data_stride_bytes
) {
    if (!view) {
        return;
    }
    memset(view, 0, sizeof(*view));
    memset(&view->data_region_desc, 0, sizeof(view->data_region_desc));
    if (data_region) {
        view->data_region_desc = *data_region;
        view->data_region = &view->data_region_desc;
    } else {
        view->data_region = NULL;
    }
    view->seq_len = seq_len;
    view->kv_dim = kv_dim;
    view->head_size = head_size;
    view->kv_mul = kv_mul;
    view->data_stride_bytes = data_stride_bytes;
}


size_t runtime_kv_cache_row_data_offset(const RuntimeKvCacheLayerView *view, int time_idx) {
    if (!view || time_idx < 0) {
        return 0u;
    }
    return (size_t)time_idx * view->data_stride_bytes;
}

void runtime_kv_cache_write_row(RuntimeKvCacheLayerView *view, int time_idx, const float *src_row) {
    size_t row_off;
    if (!view || !src_row || !view->data_region || !view->data_region->cpu_ptr) {
        return;
    }
    row_off = runtime_kv_cache_row_data_offset(view, time_idx);
    memcpy((unsigned char *)view->data_region->cpu_ptr + row_off,
           src_row,
           (size_t)view->kv_dim * sizeof(float));
}

void runtime_kv_cache_extract_row(const RuntimeKvCacheLayerView *view, int time_idx, float *dst_row) {
    size_t row_off;
    if (!view || !dst_row || !view->data_region || !view->data_region->cpu_ptr) {
        return;
    }
    row_off = runtime_kv_cache_row_data_offset(view, time_idx);
    memcpy(dst_row,
           (const unsigned char *)view->data_region->cpu_ptr + row_off,
           (size_t)view->kv_dim * sizeof(float));
}

const float *runtime_kv_cache_head_ptr(const RuntimeKvCacheLayerView *view, int time_idx, int head_idx) {
    int kv_head_idx;
    size_t row_off;
    if (!view || view->kv_mul <= 0 || !view->data_region || !view->data_region->cpu_ptr) {
        return NULL;
    }
    kv_head_idx = head_idx / view->kv_mul;
    row_off = runtime_kv_cache_row_data_offset(view, time_idx);
    return (const float *)((const unsigned char *)view->data_region->cpu_ptr + row_off) +
           (size_t)kv_head_idx * (size_t)view->head_size;
}


void runtime_refresh_kv_main_regions(RuntimeState *state, const RuntimeConfig *config) {
    int kv_dim;
    size_t float_bytes_per_row;
    if (!state || !config) {
        return;
    }
    kv_dim = (config->dim * config->n_kv_heads) / config->n_heads;
    float_bytes_per_row = (size_t)kv_dim * sizeof(float);
    state->key_cache_main_data_region.cpu_uncached_addr = RUNTIME_KEY_CACHE_MAIN_DATA_UNC;
    state->key_cache_main_data_region.phys_addr = RUNTIME_KEY_CACHE_MAIN_DATA_PHYS;
    state->key_cache_main_data_region.size = (size_t)config->n_layers * (size_t)config->seq_len * float_bytes_per_row;
    state->value_cache_main_data_region.cpu_uncached_addr = RUNTIME_VALUE_CACHE_MAIN_DATA_UNC;
    state->value_cache_main_data_region.phys_addr = RUNTIME_VALUE_CACHE_MAIN_DATA_PHYS;
    state->value_cache_main_data_region.size = (size_t)config->n_layers * (size_t)config->seq_len * float_bytes_per_row;
}

void runtime_seed_kv_main_regions(RuntimeState *state, const RuntimeConfig *config, int kv_dim) {
    size_t total_bytes = (size_t)config->n_layers * (size_t)config->seq_len * (size_t)kv_dim * sizeof(float);
    if (!state) {
        return;
    }
    state->key_cache_main_data_region.cpu_ptr = state->key_cache;
    state->key_cache_main_data_region.size = total_bytes;
    state->value_cache_main_data_region.cpu_ptr = state->value_cache;
    state->value_cache_main_data_region.size = total_bytes;
}

int runtime_kv_has_legacy_float_backing(const RuntimeState *state) {
    return state && state->key_cache && state->value_cache;
}

float *runtime_key_cache_layer_ptr(RuntimeState *state, int layer_idx, int seq_len, int kv_dim) {
    return state->key_cache + (size_t)layer_idx * (size_t)seq_len * (size_t)kv_dim;
}

float *runtime_value_cache_layer_ptr(RuntimeState *state, int layer_idx, int seq_len, int kv_dim) {
    return state->value_cache + (size_t)layer_idx * (size_t)seq_len * (size_t)kv_dim;
}

float *runtime_key_cache_row_ptr(RuntimeState *state, int layer_idx, int pos, int seq_len, int kv_dim) {
    return runtime_key_cache_layer_ptr(state, layer_idx, seq_len, kv_dim) + (size_t)pos * (size_t)kv_dim;
}

float *runtime_value_cache_row_ptr(RuntimeState *state, int layer_idx, int pos, int seq_len, int kv_dim) {
    return runtime_value_cache_layer_ptr(state, layer_idx, seq_len, kv_dim) + (size_t)pos * (size_t)kv_dim;
}

void runtime_linear_qkv_row(RuntimeBackend *backend, float *q, float *k_row, float *v_row, const float *x, int layer_idx) {
    backend->ops->linear_qkv(backend, q, k_row, v_row, x, layer_idx);
}

void runtime_clear_transient_kv(RuntimeState *state, int kv_dim) {
    if (!state || kv_dim <= 0) {
        return;
    }
    if (state->k) memset(state->k, 0, (size_t)kv_dim * sizeof(float));
    if (state->v) memset(state->v, 0, (size_t)kv_dim * sizeof(float));
}

void runtime_init_layer_kv_views(
    RuntimeModel *model,
    int layer_idx,
    RuntimeKvCacheLayerView *key_view,
    RuntimeKvCacheLayerView *value_view,
    int bind_hw_regions
) {
    RuntimeConfig *config = &model->config;
    RuntimeState *state = &model->state;
    int kv_dim = (config->dim * config->n_kv_heads) / config->n_heads;
    int head_size = config->dim / config->n_heads;
    int kv_mul = config->n_heads / config->n_kv_heads;
    const SharedBufferDesc *key_data_region = &state->key_cache_main_data_region;
    const SharedBufferDesc *value_data_region = &state->value_cache_main_data_region;
    size_t row_stride_bytes = (size_t)kv_dim * sizeof(float);
    size_t layer_offset_bytes = (size_t)layer_idx * (size_t)config->seq_len * row_stride_bytes;

    runtime_init_kv_cache_layer_view(
        key_view,
        key_data_region,
        config->seq_len,
        kv_dim,
        head_size,
        kv_mul,
        row_stride_bytes);
    runtime_init_kv_cache_layer_view(
        value_view,
        value_data_region,
        config->seq_len,
        kv_dim,
        head_size,
        kv_mul,
        row_stride_bytes);
    if (bind_hw_regions) {
        key_data_region = runtime_hw_adapter_key_cache_main_data();
        value_data_region = runtime_hw_adapter_value_cache_main_data();

        key_view->data_region_desc = *key_data_region;
        key_view->data_region_desc.cpu_ptr = key_data_region->cpu_ptr ?
            ((unsigned char *)key_data_region->cpu_ptr + layer_offset_bytes) : NULL;
        key_view->data_region_desc.cpu_uncached_addr = key_data_region->cpu_uncached_addr + layer_offset_bytes;
        key_view->data_region_desc.phys_addr = key_data_region->phys_addr + (uint32_t)layer_offset_bytes;
        key_view->data_region_desc.size = key_data_region->size >= layer_offset_bytes ?
            (key_data_region->size - layer_offset_bytes) : 0u;
        key_view->data_region = &key_view->data_region_desc;

        value_view->data_region_desc = *value_data_region;
        value_view->data_region_desc.cpu_ptr = value_data_region->cpu_ptr ?
            ((unsigned char *)value_data_region->cpu_ptr + layer_offset_bytes) : NULL;
        value_view->data_region_desc.cpu_uncached_addr = value_data_region->cpu_uncached_addr + layer_offset_bytes;
        value_view->data_region_desc.phys_addr = value_data_region->phys_addr + (uint32_t)layer_offset_bytes;
        value_view->data_region_desc.size = value_data_region->size >= layer_offset_bytes ?
            (value_data_region->size - layer_offset_bytes) : 0u;
        value_view->data_region = &value_view->data_region_desc;
    }
}

void runtime_commit_kv_rows(
    RuntimeModel *model,
    RuntimeKvCacheLayerView *key_view,
    RuntimeKvCacheLayerView *value_view,
    int pos,
    const float *key_row,
    const float *value_row
) {
    (void)model;
    runtime_kv_cache_write_row(key_view, pos, key_row);
    runtime_kv_cache_write_row(value_view, pos, value_row);
}

void runtime_trace_kv_row(FILE *fp, const char *label, const RuntimeKvCacheLayerView *view, int time_idx) {
    if (!fp || !label || !view) {
        return;
    }
    fprintf(fp, "[KV_TRACE] %s t=%d row_off=0x%zx cpu=0x%08llx phys=0x%08x\n",
        label,
        time_idx,
        runtime_kv_cache_row_data_offset(view, time_idx),
        view->data_region ? (unsigned long long)view->data_region->cpu_uncached_addr : 0ull,
        view->data_region ? view->data_region->phys_addr : 0u);
}

const RuntimeKvCacheLayerView *runtime_key_view(const RuntimeKvCacheLayerView *view) {
    return view;
}

const RuntimeKvCacheLayerView *runtime_value_view(const RuntimeKvCacheLayerView *view) {
    return view;
}

void runtime_write_kv_rows(
    RuntimeModel *model,
    RuntimeKvCacheLayerView *key_view,
    RuntimeKvCacheLayerView *value_view,
    int pos,
    const float *key_row,
    const float *value_row
) {
    runtime_commit_kv_rows(model, key_view, value_view, pos, key_row, value_row);
}

void runtime_extract_kv_row(const RuntimeKvCacheLayerView *view, int time_idx, float *dst_row) {
    runtime_kv_cache_extract_row(view, time_idx, dst_row);
}

void runtime_apply_rope_to_qk_row(float *q, float *k_row, int dim, int kv_dim, int head_size, int pos) {
    apply_rope_inplace(q, k_row, dim, kv_dim, head_size, pos);
}

void runtime_bind_kv_view_to_hw_regions(RuntimeKvCacheLayerView *view, int is_value_region) {
    if (!view) {
        return;
    }
    if (is_value_region) {
        view->data_region = runtime_hw_adapter_value_cache_main_data();
    } else {
        view->data_region = runtime_hw_adapter_key_cache_main_data();
    }
}


void rmsnorm_ref(float *out, const float *x, const float *weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; ++j) ss += x[j] * x[j];
    ss /= (float)size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; ++j) out[j] = weight[j] * (ss * x[j]);
}

void softmax_ref(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; ++i) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; ++i) x[i] /= sum;
}

void matmul_ref(float *xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d, int group_size) {
    for (int i = 0; i < d; ++i) {
        float val = 0.0f;
        int in = i * n;
        for (int j_group_start = 0; j_group_start < n; j_group_start += group_size) {
            int32_t group_ival = 0;
            for (int k_offset = 0; k_offset < group_size; ++k_offset) {
                int current_j = j_group_start + k_offset;
                if (current_j < n) {
                    group_ival += ((int32_t)x->q[current_j]) * ((int32_t)w->q[in + current_j]);
                }
            }
            val += ((float)group_ival) * w->s[(in + j_group_start) / group_size] * x->s[j_group_start / group_size];
        }
        xout[i] = val;
    }
}

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str);
}

char *decode_token(RuntimeTokenizer *tokenizer, int prev_token, int token) {
    if (token < 0 || token >= tokenizer->vocab_size) return "?";
    char *piece = tokenizer->vocab[token];
    if (prev_token == 1 && piece[0] == ' ') piece++;
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char *)tokenizer->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf_piece(char *piece) {
    if (piece == NULL || piece[0] == '\0') return;
    if (piece[1] == '\0') {
        unsigned char byte_val = (unsigned char)piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) return;
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    TokenIndex tok = { .str = str, .id = 0 };
    TokenIndex *res = (TokenIndex *)bsearch(&tok, sorted_vocab, (size_t)vocab_size, sizeof(TokenIndex), compare_tokens);
    return res ? res->id : -1;
}

int encode_text(
    RuntimeTokenizer *tokenizer,
    char *text,
    int8_t bos,
    int8_t eos,
    int *tokens,
    int token_capacity,
    int *n_tokens
) {
    if (tokenizer->sorted_vocab == NULL) {
        tokenizer->sorted_vocab = (TokenIndex *)malloc((size_t)tokenizer->vocab_size * sizeof(TokenIndex));
        if (!tokenizer->sorted_vocab) {
            fprintf(stderr, "encode_text: sorted_vocab 分配失败\n");
            exit(EXIT_FAILURE);
        }
        for (int i = 0; i < tokenizer->vocab_size; ++i) {
            tokenizer->sorted_vocab[i].str = tokenizer->vocab[i];
            tokenizer->sorted_vocab[i].id = i;
        }
        qsort(tokenizer->sorted_vocab, (size_t)tokenizer->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    if (text == NULL) {
        fprintf(stderr, "encode_text: text 不能为空\n");
        exit(EXIT_FAILURE);
    }

    char *str_buffer = (char *)malloc((size_t)(tokenizer->max_token_length * 2 + 3));
    if (!str_buffer) {
        fprintf(stderr, "encode_text: 临时缓冲分配失败\n");
        exit(EXIT_FAILURE);
    }

    size_t str_len = 0;
    int count = 0;
    if (bos) {
        if (count >= token_capacity) {
            free(str_buffer);
            return -1;
        }
        tokens[count++] = 1;
    }
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", tokenizer->sorted_vocab, tokenizer->vocab_size);
        if (dummy_prefix >= 0) {
            if (count >= token_capacity) {
                free(str_buffer);
                return -1;
            }
            tokens[count++] = dummy_prefix;
        }
    }

    for (char *c = text; *c != '\0'; ++c) {
        // 按 UTF-8 codepoint 聚合后再查词表，避免把中文等多字节字符错误拆散。
        if ((*c & 0xC0) != 0x80) {
            str_len = 0;
        }
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';
        if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }
        int id = str_lookup(str_buffer, tokenizer->sorted_vocab, tokenizer->vocab_size);
        if (id != -1) {
            if (count >= token_capacity) {
                free(str_buffer);
                return -1;
            }
            tokens[count++] = id;
        } else {
            for (size_t i = 0; i < str_len; ++i) {
                if (count >= token_capacity) {
                    free(str_buffer);
                    return -1;
                }
                tokens[count++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0;
    }

    while (1) {
        float best_score = -1e10f;
        int best_id = -1;
        int best_idx = -1;
        for (int i = 0; i < count - 1; ++i) {
            sprintf(str_buffer, "%s%s", tokenizer->vocab[tokens[i]], tokenizer->vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer, tokenizer->sorted_vocab, tokenizer->vocab_size);
            if (id != -1 && tokenizer->vocab_scores[id] > best_score) {
                best_score = tokenizer->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }
        if (best_idx == -1) break;
        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < count - 1; ++i) tokens[i] = tokens[i + 1];
        count--;
    }

    if (eos) {
        if (count >= token_capacity) {
            free(str_buffer);
            return -1;
        }
        tokens[count++] = 2;
    }
    *n_tokens = count;
    free(str_buffer);
    return 0;
}

static int sample_argmax(float *probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; ++i) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

static int compare_prob_index(const void *a, const void *b) {
    float diff = ((ProbIndex *)b)->prob - ((ProbIndex *)a)->prob;
    if (diff > 0) return 1;
    if (diff < 0) return -1;
    return 0;
}

static int runtime_env_flag_enabled(const char *name) {
    const char *value = getenv(name);
    if (!value || !value[0]) return 0;
    if (strcmp(value, "0") == 0) return 0;
    if (strcmp(value, "false") == 0 || strcmp(value, "FALSE") == 0) return 0;
    if (strcmp(value, "off") == 0 || strcmp(value, "OFF") == 0) return 0;
    return 1;
}

static unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (unsigned int)((*state * 0x2545F4914F6CDD1Dull) >> 32);
}

static float random_f32(unsigned long long *state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

static int sample_mult_candidates(const ProbIndex *candidates, int n, float total_prob, float coin) {
    float r = coin * total_prob;
    float cdf = 0.0f;
    for (int i = 0; i < n; ++i) {
        cdf += candidates[i].prob;
        if (r < cdf) return candidates[i].index;
    }
    return candidates[n - 1].index;
}

static int collect_top_k_probabilities(
    const float *probabilities,
    int vocab_size,
    int top_k,
    ProbIndex *top_candidates
) {
    int limit = top_k;
    int count = 0;
    if (limit <= 0 || limit > vocab_size) limit = vocab_size;
    for (int i = 0; i < vocab_size; ++i) {
        float prob = probabilities[i];
        int pos;
        if (prob <= 0.0f) continue;
        if (count == limit && prob <= top_candidates[count - 1].prob) continue;
        pos = count;
        if (count < limit) {
            count++;
        } else {
            pos = count - 1;
        }
        while (pos > 0 && top_candidates[pos - 1].prob < prob) {
            if (pos < limit) top_candidates[pos] = top_candidates[pos - 1];
            pos--;
        }
        top_candidates[pos].index = i;
        top_candidates[pos].prob = prob;
    }
    return count;
}

static int truncate_top_p_candidates(ProbIndex *candidates, int count, float topp, float *kept_sum) {
    float cumulative_prob = 0.0f;
    int kept = count;
    *kept_sum = 0.0f;
    if (count <= 0) return 0;
    if (topp <= 0.0f || topp >= 1.0f) {
        for (int i = 0; i < count; ++i) *kept_sum += candidates[i].prob;
        return count;
    }
    for (int i = 0; i < count; ++i) {
        cumulative_prob += candidates[i].prob;
        if (cumulative_prob > topp) {
            kept = i + 1;
            break;
        }
    }
    for (int i = 0; i < kept; ++i) {
        *kept_sum += candidates[i].prob;
    }
    if (*kept_sum <= 0.0f) {
        for (int i = 0; i < count; ++i) {
            *kept_sum += candidates[i].prob;
        }
        return count;
    }
    return kept;
}

static void debug_dump_first_step_candidates(
    const ProbIndex *top_k_candidates,
    int top_k_count,
    const ProbIndex *top_p_candidates,
    int top_p_count,
    float top_p_sum,
    int sampled_token
) {
    fprintf(stderr, "[SAMPLER] first-step top_k count=%d top_p count=%d top_p_sum=%.6f sampled=%d\n",
            top_k_count, top_p_count, top_p_sum, sampled_token);
    fprintf(stderr, "[SAMPLER] top_k:");
    for (int i = 0; i < top_k_count; ++i) {
        fprintf(stderr, " (%d,%.6f)", top_k_candidates[i].index, top_k_candidates[i].prob);
    }
    fprintf(stderr, "\n");
    fprintf(stderr, "[SAMPLER] top_p:");
    for (int i = 0; i < top_p_count; ++i) {
        fprintf(stderr, " (%d,%.6f)", top_p_candidates[i].index, top_p_candidates[i].prob);
    }
    fprintf(stderr, "\n");
}

static void apply_repetition_penalty(float *logits, int vocab_size, const int *history, int history_len, float penalty) {
    unsigned char seen_once[512] = {0};
    if (penalty <= 1.0f) return;
    for (int h = 0; h < history_len; ++h) {
        int tok = history[h];
        if (tok < 0 || tok >= vocab_size) continue;
        if (vocab_size <= (int)sizeof(seen_once)) {
            if (seen_once[tok]) continue;
            seen_once[tok] = 1;
        } else {
            int already_seen = 0;
            for (int prev = 0; prev < h; ++prev) {
                if (history[prev] == tok) {
                    already_seen = 1;
                    break;
                }
            }
            if (already_seen) continue;
        }
        if (logits[tok] < 0.0f) logits[tok] *= penalty;
        else logits[tok] /= penalty;
    }
}

static void apply_no_repeat_ngram(float *logits, int vocab_size, const int *history, int history_len, int ngram_size) {
    if (ngram_size <= 0 || history_len < ngram_size - 1) return;
    int prefix_len = ngram_size - 1;
    int banned[512];
    int banned_count = 0;
    for (int i = 0; i <= history_len - ngram_size; ++i) {
        int match = 1;
        for (int j = 0; j < prefix_len; ++j) {
            if (history[i + j] != history[history_len - prefix_len + j]) {
                match = 0;
                break;
            }
        }
        if (!match) continue;
        int cand = history[i + prefix_len];
        int exists = 0;
        for (int k = 0; k < banned_count; ++k) {
            if (banned[k] == cand) {
                exists = 1;
                break;
            }
        }
        if (!exists && banned_count < 512) banned[banned_count++] = cand;
    }
    for (int i = 0; i < banned_count; ++i) {
        int tok = banned[i];
        if (tok >= 0 && tok < vocab_size) logits[tok] = -1e30f;
    }
}

void build_sampler(
    RuntimeSampler *sampler,
    int vocab_size,
    float temperature,
    float topp,
    int top_k,
    float repetition_penalty,
    int no_repeat_ngram_size,
    unsigned long long rng_seed
) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->top_k = top_k;
    sampler->repetition_penalty = repetition_penalty;
    sampler->no_repeat_ngram_size = no_repeat_ngram_size;
    sampler->rng_state = rng_seed;
    sampler->debug_dump_first_step = runtime_env_flag_enabled("RUNQ_DEBUG_FIRST_STEP");
    sampler->sample_calls = 0;
    sampler->probindex = (ProbIndex *)malloc((size_t)vocab_size * sizeof(ProbIndex));
    if (!sampler->probindex) {
        fprintf(stderr, "build_sampler: probindex 分配失败\n");
        exit(EXIT_FAILURE);
    }
}

void free_sampler(RuntimeSampler *sampler) {
    free(sampler->probindex);
    sampler->probindex = NULL;
}

int sample_token(RuntimeSampler *sampler, float *logits, const int *history, int history_len) {
    ProbIndex *candidates = sampler->probindex;
    int candidate_count = 0;
    int kept_count = 0;
    float kept_sum = 0.0f;
    int next = 0;
    if (sampler->temperature == 0.0f) {
        apply_repetition_penalty(logits, sampler->vocab_size, history, history_len, sampler->repetition_penalty);
        apply_no_repeat_ngram(logits, sampler->vocab_size, history, history_len, sampler->no_repeat_ngram_size);
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        const float temperature = sampler->temperature > 1e-6f ? sampler->temperature : 1e-6f;
        for (int q = 0; q < sampler->vocab_size; ++q) logits[q] /= temperature;
        apply_repetition_penalty(logits, sampler->vocab_size, history, history_len, sampler->repetition_penalty);
        apply_no_repeat_ngram(logits, sampler->vocab_size, history, history_len, sampler->no_repeat_ngram_size);
        softmax_ref(logits, sampler->vocab_size);
        candidate_count = collect_top_k_probabilities(logits, sampler->vocab_size, sampler->top_k, candidates);
        if (candidate_count <= 0) {
            next = sample_argmax(logits, sampler->vocab_size);
        } else {
            const float coin = random_f32(&sampler->rng_state);
            kept_count = truncate_top_p_candidates(candidates, candidate_count, sampler->topp, &kept_sum);
            next = sample_mult_candidates(candidates, kept_count, kept_sum, coin);
            if (sampler->debug_dump_first_step && sampler->sample_calls == 0) {
                debug_dump_first_step_candidates(candidates, candidate_count, candidates, kept_count, kept_sum, next);
            }
        }
    }
    sampler->sample_calls++;
    return next;
}

void read_stdin_line(const char *guide, char *buffer, size_t bufsize) {
    printf("%s", guide);
    if (fgets(buffer, (int)bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') buffer[len - 1] = '\0';
    }
}

void apply_rope_inplace(float *q, float *k, int dim, int kv_dim, int head_size, int pos) {
    for (int i = 0; i < dim; i += 2) {
        int head_dim = i % head_size;
        float freq = 1.0f / powf(10000.0f, (float)head_dim / head_size);
        float val = (float)pos * freq;
        float fcr = cosf(val);
        float fci = sinf(val);
        int rotn = i < kv_dim ? 2 : 1;
        for (int v_idx = 0; v_idx < rotn; ++v_idx) {
            float *vec = (v_idx == 0) ? q : k;
            float v0 = vec[i];
            float v1 = vec[i + 1];
            vec[i] = v0 * fcr - v1 * fci;
            vec[i + 1] = v0 * fci + v1 * fcr;
        }
    }
}

void runtime_load_embedding_row(const RuntimeModel *model, int token, float *out) {
    const RuntimeConfig *config = &model->config;
    const QuantizedTensor *q_tokens = model->weights.q_tokens;
    int dim = config->dim;
    int group_size = model->group_size;
    int base = token * dim;

    if (token < 0 || token >= config->vocab_size) {
        fprintf(stderr, "runtime_load_embedding_row: token=%d 超出 vocab_size=%d\n", token, config->vocab_size);
        exit(EXIT_FAILURE);
    }

    // token embedding 在资产里是一整张 q80 表。
    // 这里按“单行”解码，既保留和原数学一致的结果，又避免整表展开到 heap。
    for (int i = 0; i < dim; ++i) {
        int flat_idx = base + i;
        out[i] = (float)q_tokens[0].q[flat_idx] * q_tokens[0].s[flat_idx / group_size];
    }
}

float *runtime_decode_transformer_step(RuntimeBackend *backend, int token, int pos) {
    RuntimeModel *model = backend->model;
    RuntimeConfig *config = &model->config;
    RuntimeWeights *weights = &model->weights;
    RuntimeState *state = &model->state;
    int dim = config->dim;
    int kv_dim = (config->dim * config->n_kv_heads) / config->n_heads;
    int hidden_dim = config->hidden_dim;
    int head_size = dim / config->n_heads;
    int kv_mul = config->n_heads / config->n_kv_heads;

    // 这条路径是部署/验证共用的“单 token decode 调度器”。
    // 关键点是：frontend 不再调用整图 forward，而是严格通过 backend 的算子级接口推进每一步。
    runtime_load_embedding_row(model, token, state->x);
    runtime_refresh_kv_main_regions(state, config);
    for (int layer = 0; layer < config->n_layers; ++layer) {
        float *key_cache_row = runtime_key_cache_row_ptr(state, layer, pos, config->seq_len, kv_dim);
        float *value_cache_row = runtime_value_cache_row_ptr(state, layer, pos, config->seq_len, kv_dim);
        RuntimeKvCacheLayerView key_view;
        RuntimeKvCacheLayerView value_view;

        runtime_init_layer_kv_views(model, layer, &key_view, &value_view,
            backend->kind == RUNTIME_BACKEND_HWSTUB);
        // 当前 deploy V1 主线保持 float attention core；
        // 这里的 rmsnorm 已前移为 Fusion Preview 候选，但默认执行仍是独立 float 算子。
        backend->ops->rmsnorm(backend, state->xb, state->x, weights->rms_att_weight + layer * dim, dim);
        runtime_linear_qkv_row(backend, state->q, key_cache_row, value_cache_row, state->xb, layer);

        // RoPE 和 KV cache 仍放在公共调度层：
        // 1. backend 只关心算子语义；
        // 2. 位置编码与 cache 组织属于跨算子的 runtime 责任。
        runtime_apply_rope_to_qk_row(state->q, key_cache_row, dim, kv_dim, head_size, pos);
        runtime_commit_kv_rows(model, &key_view, &value_view, pos, key_cache_row, value_cache_row);
        runtime_clear_transient_kv(state, kv_dim);

        // xb 在 attention 子图里作为“所有 head 拼接后的上下文向量”。
        memset(state->xb, 0, (size_t)dim * sizeof(float));
        for (int h = 0; h < config->n_heads; ++h) {
            float *att_head = state->att + h * config->seq_len;
            float *xb_head = state->xb + h * head_size;
            backend->ops->qk_matmul(backend, att_head, state->q, runtime_key_view(&key_view), pos, h);
            // softmax_row 当前也已进入 Fusion Preview 候选，
            // 但执行层仍保持 `qk_matmul -> softmax_row -> av_matmul` 的分算子 float 路径。
            backend->ops->softmax_row(backend, att_head, pos + 1);
            backend->ops->av_matmul(backend, xb_head, att_head, runtime_value_view(&value_view), pos, h);
        }

        backend->ops->linear_attn_o(backend, state->xb2, state->xb, layer);
        backend->ops->residual_add(backend, state->x, state->xb2, dim);

        // FFN 前的 rmsnorm 与 attention 前的口径一致：
        // 已进入 Fusion Preview 候选，但当前默认执行仍保持独立 float 算子。
        backend->ops->rmsnorm(backend, state->xb, state->x, weights->rms_ffn_weight + layer * dim, dim);
        backend->ops->linear_ffn_w1(backend, state->hb, state->xb, layer);
        backend->ops->linear_ffn_w3(backend, state->hb2, state->xb, layer);
        backend->ops->gate_mul(backend, state->hb, state->hb, state->hb2, hidden_dim);
        backend->ops->linear_ffn_w2(backend, state->xb, state->hb, layer);
        backend->ops->residual_add(backend, state->x, state->xb, dim);
    }
    backend->ops->final_norm(backend, state->x, state->x);
    backend->ops->lm_head(backend, state->logits, state->x);
    return state->logits;
}
