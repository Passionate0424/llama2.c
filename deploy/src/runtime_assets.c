#include "runtime_assets.h"

#include <stdlib.h>
#include <string.h>

#include "runtime_common.h"

// 第一版默认资产直接放在 deploy/assets/stories260K_qat_best 下。
// 后续切模型时，只需切换目录 tag，不改运行时代码。
#include "../assets/stories260K_qat_best/stories_data.h"
#include "../assets/stories260K_qat_best/tok512.h"

static void read_embedded_bytes(
    const unsigned char **cursor,
    const unsigned char *end,
    void *out,
    size_t len,
    const char *what
) {
    if ((size_t)(end - *cursor) < len) {
        fprintf(stderr, "read_embedded_bytes: 读取 %s 时头文件资产截断\n", what);
        exit(EXIT_FAILURE);
    }
    memcpy(out, *cursor, len);
    *cursor += len;
}

static int validate_group_divisible(const char *label, int total, int group_size) {
    if (group_size <= 0 || total <= 0) {
        fprintf(stderr, "runtime_load_default_model: %s 大小=%d 或 group_size=%d 非法\n", label, total, group_size);
        return -1;
    }
    if ((total % group_size) != 0) {
        fprintf(stderr, "runtime_load_default_model: %s 大小 %d 不能被 group_size=%d 整除\n", label, total, group_size);
        return -1;
    }
    return 0;
}

static int validate_model_layout(
    const RuntimeConfig *cfg,
    size_t raw_model_size,
    int header_size,
    int shared_classifier,
    int group_size
) {
    size_t used = (size_t)header_size;
    int head_size;
    if ((size_t)header_size > raw_model_size) {
        fprintf(stderr, "runtime_load_default_model: header_size=%d 超过模型总大小=%zu\n", header_size, raw_model_size);
        return -1;
    }
    if (cfg->dim <= 0 || cfg->hidden_dim <= 0 || cfg->n_layers <= 0 || cfg->n_heads <= 0 ||
        cfg->n_kv_heads <= 0 || cfg->vocab_size <= 0 || cfg->seq_len <= 0) {
        fprintf(stderr, "runtime_load_default_model: config 字段非法\n");
        return -1;
    }
    if ((cfg->dim % cfg->n_heads) != 0 || (cfg->n_heads % cfg->n_kv_heads) != 0) {
        fprintf(stderr, "runtime_load_default_model: head 配置非法 dim=%d n_heads=%d n_kv_heads=%d\n",
            cfg->dim, cfg->n_heads, cfg->n_kv_heads);
        return -1;
    }
    head_size = cfg->dim / cfg->n_heads;
    if (validate_group_divisible("token_embedding", cfg->vocab_size * cfg->dim, group_size) != 0) return -1;
    if (validate_group_divisible("wq", cfg->dim * (cfg->n_heads * head_size), group_size) != 0) return -1;
    if (validate_group_divisible("wk", cfg->dim * (cfg->n_kv_heads * head_size), group_size) != 0) return -1;
    if (validate_group_divisible("wv", cfg->dim * (cfg->n_kv_heads * head_size), group_size) != 0) return -1;
    if (validate_group_divisible("wo", (cfg->n_heads * head_size) * cfg->dim, group_size) != 0) return -1;
    if (validate_group_divisible("w1", cfg->dim * cfg->hidden_dim, group_size) != 0) return -1;
    if (validate_group_divisible("w2", cfg->hidden_dim * cfg->dim, group_size) != 0) return -1;
    if (validate_group_divisible("w3", cfg->dim * cfg->hidden_dim, group_size) != 0) return -1;
    if (validate_group_divisible("wcls", cfg->dim * cfg->vocab_size, group_size) != 0) return -1;

    // fp32 RMS 权重区
    used += (size_t)cfg->n_layers * (size_t)cfg->dim * sizeof(float);
    used += (size_t)cfg->n_layers * (size_t)cfg->dim * sizeof(float);
    used += (size_t)cfg->dim * sizeof(float);

    // 一个 q80 tensor 占：int8 数据 + float scale 数组
#define ADD_Q80_TENSOR(elem_count) \
    do { \
        size_t _n = (size_t)(elem_count); \
        used += _n * sizeof(int8_t); \
        used += (size_t)grouped_scale_count((int)_n, group_size) * sizeof(float); \
    } while (0)

    ADD_Q80_TENSOR(cfg->vocab_size * cfg->dim);
    for (int i = 0; i < cfg->n_layers; ++i) {
        (void)i;
        ADD_Q80_TENSOR(cfg->dim * (cfg->n_heads * head_size));
        ADD_Q80_TENSOR(cfg->dim * (cfg->n_kv_heads * head_size));
        ADD_Q80_TENSOR(cfg->dim * (cfg->n_kv_heads * head_size));
        ADD_Q80_TENSOR((cfg->n_heads * head_size) * cfg->dim);
        ADD_Q80_TENSOR(cfg->dim * cfg->hidden_dim);
        ADD_Q80_TENSOR(cfg->hidden_dim * cfg->dim);
        ADD_Q80_TENSOR(cfg->dim * cfg->hidden_dim);
    }
    if (!shared_classifier) {
        ADD_Q80_TENSOR(cfg->dim * cfg->vocab_size);
    }
#undef ADD_Q80_TENSOR

    if (used > raw_model_size) {
        fprintf(stderr, "runtime_load_default_model: 推导权重布局越界 used=%zu raw_model_size=%zu\n", used, raw_model_size);
        return -1;
    }
    return 0;
}

int runtime_load_default_model(RuntimeModel *model) {
    const int header_size = 256;
    const unsigned char *cursor = stories260K_q80_bin;
    const unsigned char *end = stories260K_q80_bin + stories260K_q80_bin_len;
    uint32_t magic_number = 0;
    int version = 0;
    uint8_t shared_classifier = 0;
    int group_size = 0;
    void *weights_ptr = NULL;

    // 模型头文件是 xxd 风格数组，这里直接从内存中解析 runq q80 header。
    runtime_model_init(model);
    model->raw_model_data = stories260K_q80_bin;
    model->raw_model_size = (size_t)stories260K_q80_bin_len;
    model->group_size = 0;

    read_embedded_bytes(&cursor, end, &magic_number, sizeof(magic_number), "magic");
    if (magic_number != 0x616b3432) {
        fprintf(stderr, "runtime_load_default_model: 模型 magic 错误 0x%x\n", magic_number);
        return -1;
    }
    read_embedded_bytes(&cursor, end, &version, sizeof(version), "version");
    if (version != 2) {
        fprintf(stderr, "runtime_load_default_model: 仅支持 version 2，当前为 %d\n", version);
        return -1;
    }
    read_embedded_bytes(&cursor, end, &model->config, sizeof(RuntimeConfig), "config");
    read_embedded_bytes(&cursor, end, &shared_classifier, sizeof(shared_classifier), "shared_classifier");
    read_embedded_bytes(&cursor, end, &group_size, sizeof(group_size), "group_size");
    if (group_size <= 0) {
        fprintf(stderr, "runtime_load_default_model: group size 非法 %d\n", group_size);
        return -1;
    }
    model->group_size = group_size;
    if (validate_model_layout(&model->config, model->raw_model_size, header_size, shared_classifier, group_size) != 0) {
        return -1;
    }

    weights_ptr = (void *)(model->raw_model_data + header_size);
    {
        // 这里只做一次性的权重元信息映射，避免运行时重复解析。
        RuntimeConfig *cfg = &model->config;
        RuntimeWeights *weights = &model->weights;
        float *fptr = (float *)weights_ptr;
        int head_size = cfg->dim / cfg->n_heads;

        weights->rms_att_weight = fptr; fptr += cfg->n_layers * cfg->dim;
        weights->rms_ffn_weight = fptr; fptr += cfg->n_layers * cfg->dim;
        weights->rms_final_weight = fptr; fptr += cfg->dim;

        weights_ptr = (void *)fptr;
        // token embedding 保持为 q80 资产原位映射。
        // 运行时改成“按 token 行解码”，避免再把整张 embedding 表复制一份到 heap。
        weights->q_tokens = init_quantized_tensors(&weights_ptr, 1, cfg->vocab_size * cfg->dim, group_size);
        weights->wq = init_quantized_tensors(&weights_ptr, cfg->n_layers, cfg->dim * (cfg->n_heads * head_size), group_size);
        weights->wk = init_quantized_tensors(&weights_ptr, cfg->n_layers, cfg->dim * (cfg->n_kv_heads * head_size), group_size);
        weights->wv = init_quantized_tensors(&weights_ptr, cfg->n_layers, cfg->dim * (cfg->n_kv_heads * head_size), group_size);
        weights->wo = init_quantized_tensors(&weights_ptr, cfg->n_layers, (cfg->n_heads * head_size) * cfg->dim, group_size);
        weights->w1 = init_quantized_tensors(&weights_ptr, cfg->n_layers, cfg->dim * cfg->hidden_dim, group_size);
        weights->w2 = init_quantized_tensors(&weights_ptr, cfg->n_layers, cfg->hidden_dim * cfg->dim, group_size);
        weights->w3 = init_quantized_tensors(&weights_ptr, cfg->n_layers, cfg->dim * cfg->hidden_dim, group_size);
        weights->wcls = shared_classifier ? weights->q_tokens :
            init_quantized_tensors(&weights_ptr, 1, cfg->dim * cfg->vocab_size, group_size);
    }

    runtime_malloc_state(&model->state, &model->config, model->group_size);
    return runtime_validate_model_layout(model);
}

int runtime_load_default_tokenizer(RuntimeTokenizer *tokenizer, int vocab_size) {
    const unsigned char *cursor = tok512_bin;
    const unsigned char *end = tok512_bin + tok512_bin_len;
    int len = 0;
    // tokenizer 同样从头文件数组中解析，第一版不再依赖外部 tok512.bin。
    memset(tokenizer, 0, sizeof(*tokenizer));
    tokenizer->vocab_size = vocab_size;
    tokenizer->vocab = (char **)malloc((size_t)vocab_size * sizeof(char *));
    tokenizer->vocab_scores = (float *)malloc((size_t)vocab_size * sizeof(float));
    if (!tokenizer->vocab || !tokenizer->vocab_scores) {
        fprintf(stderr, "runtime_load_default_tokenizer: tokenizer 元信息分配失败\n");
        return -1;
    }
    for (int i = 0; i < 256; ++i) {
        tokenizer->byte_pieces[i * 2] = (unsigned char)i;
        tokenizer->byte_pieces[i * 2 + 1] = '\0';
    }

    read_embedded_bytes(&cursor, end, &tokenizer->max_token_length, sizeof(tokenizer->max_token_length), "max_token_length");
    for (int i = 0; i < vocab_size; ++i) {
        read_embedded_bytes(&cursor, end, tokenizer->vocab_scores + i, sizeof(float), "token_score");
        read_embedded_bytes(&cursor, end, &len, sizeof(len), "token_len");
        if (len < 0 || (size_t)(end - cursor) < (size_t)len) {
            fprintf(stderr, "runtime_load_default_tokenizer: token %d 长度非法 %d\n", i, len);
            return -1;
        }
        tokenizer->vocab[i] = (char *)malloc((size_t)len + 1);
        if (!tokenizer->vocab[i]) {
            fprintf(stderr, "runtime_load_default_tokenizer: vocab[%d] 分配失败\n", i);
            return -1;
        }
        memcpy(tokenizer->vocab[i], cursor, (size_t)len);
        tokenizer->vocab[i][len] = '\0';
        cursor += len;
    }
    if (cursor > end) {
        fprintf(stderr, "runtime_load_default_tokenizer: tokenizer 解析越界\n");
        return -1;
    }
    return 0;
}

void runtime_free_tokenizer(RuntimeTokenizer *tokenizer) {
    if (tokenizer->vocab) {
        for (int i = 0; i < tokenizer->vocab_size; ++i) {
            free(tokenizer->vocab[i]);
        }
        free(tokenizer->vocab);
    }
    free(tokenizer->vocab_scores);
    free(tokenizer->sorted_vocab);
    memset(tokenizer, 0, sizeof(*tokenizer));
}

int runtime_validate_model_layout(const RuntimeModel *model) {
    const unsigned char *base = model->raw_model_data;
    const unsigned char *end = model->raw_model_data + model->raw_model_size;
    const RuntimeConfig *cfg = &model->config;
    const RuntimeWeights *w = &model->weights;
    int head_size = cfg->dim / cfg->n_heads;
    int q_tokens_elems = cfg->vocab_size * cfg->dim;
    int qkv_elems = cfg->dim * (cfg->n_heads * head_size);
    int kv_elems = cfg->dim * (cfg->n_kv_heads * head_size);
    int wo_elems = (cfg->n_heads * head_size) * cfg->dim;
    int w13_elems = cfg->dim * cfg->hidden_dim;
    int w2_elems = cfg->hidden_dim * cfg->dim;

#define CHECK_RANGE(ptr, bytes, label) \
    do { \
        const unsigned char *p = (const unsigned char *)(ptr); \
        if (p < base || p + (bytes) > end) { \
            fprintf(stderr, "runtime_validate_model_layout: %s 越界\n", label); \
            return -1; \
        } \
    } while (0)

    CHECK_RANGE(w->rms_att_weight, (size_t)cfg->n_layers * cfg->dim * sizeof(float), "rms_att_weight");
    CHECK_RANGE(w->rms_ffn_weight, (size_t)cfg->n_layers * cfg->dim * sizeof(float), "rms_ffn_weight");
    CHECK_RANGE(w->rms_final_weight, (size_t)cfg->dim * sizeof(float), "rms_final_weight");
    CHECK_RANGE(w->q_tokens[0].q, (size_t)q_tokens_elems * sizeof(int8_t), "q_tokens.q");
    CHECK_RANGE(w->q_tokens[0].s, (size_t)grouped_scale_count(q_tokens_elems, model->group_size) * sizeof(float), "q_tokens.s");
    CHECK_RANGE(w->wq[0].q, (size_t)qkv_elems * sizeof(int8_t), "wq[0].q");
    CHECK_RANGE(w->wk[0].q, (size_t)kv_elems * sizeof(int8_t), "wk[0].q");
    CHECK_RANGE(w->wv[0].q, (size_t)kv_elems * sizeof(int8_t), "wv[0].q");
    CHECK_RANGE(w->wo[0].q, (size_t)wo_elems * sizeof(int8_t), "wo[0].q");
    CHECK_RANGE(w->w1[0].q, (size_t)w13_elems * sizeof(int8_t), "w1[0].q");
    CHECK_RANGE(w->w2[0].q, (size_t)w2_elems * sizeof(int8_t), "w2[0].q");
    CHECK_RANGE(w->w3[0].q, (size_t)w13_elems * sizeof(int8_t), "w3[0].q");
#undef CHECK_RANGE

    return 0;
}
