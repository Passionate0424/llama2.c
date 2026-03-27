/*
 * runq_coarse_runtime_explore.c
 *
 * 作用：
 * - 保持 runq.c 基线不变
 * - 在单独文件中验证“粗粒度 linear op + prefill/decode 分段”的软件 runtime 结构
 * - 当前仍走 software fallback 数学，不接硬件 MMIO
 *
 * 说明：
 * - 这是结构探索工具，不是最终 runtime
 * - 它通过 include runq.c 复用模型加载、tokenizer、采样等基础能力
 * - 然后在本文件中重写 forward()/generate() 做结构化实验
 */

#define matmul base_matmul
#define forward base_forward
#define generate base_generate
#define main base_main
#include "../runq.c"
#undef matmul
#undef forward
#undef generate
#undef main

#define ACCEL_MAX_PREFILL_BATCH 8

typedef enum {
    ACCEL_OP_LINEAR_GENERIC = 0,
    ACCEL_OP_LINEAR_ATTN_Q,
    ACCEL_OP_LINEAR_ATTN_K,
    ACCEL_OP_LINEAR_ATTN_V,
    ACCEL_OP_LINEAR_ATTN_O,
    ACCEL_OP_LINEAR_FFN_W1,
    ACCEL_OP_LINEAR_FFN_W2,
    ACCEL_OP_LINEAR_FFN_W3,
    ACCEL_OP_LINEAR_CLS,
    ACCEL_OP_RMSNORM_ATTN,
    ACCEL_OP_RMSNORM_FFN,
    ACCEL_OP_RMSNORM_FINAL,
    ACCEL_OP_QK_SCORE,
    ACCEL_OP_SOFTMAX_AV
} AccelOp;

/*
 * AccelOp:
 * 站在“软件提交粗粒度 op”视角，对 Transformer/LLM 主干算子做分类。
 *
 * 注意：
 * - 这里的 LINEAR_* 并不是说硬件里有很多不同的矩阵乘模块；
 * - 它们在底层通常仍然会落到同一个 GEMM/LINEAR engine；
 * - 区分这些 op 的目的是让软件调度能表达“这次线性层属于 attention 还是 FFN”。
 */

typedef enum {
    ACCEL_PHASE_PREFILL = 0,
    ACCEL_PHASE_DECODE
} AccelPhase;

/*
 * AccelPhase:
 * - PREFILL: 仍在消费 prompt，未来希望尽量把 N 凑大（比如 N=4/N=8）
 * - DECODE : 已进入自回归生成，通常是 N=1
 *
 * 当前探索版里，phase 已进入 descriptor，但 prefill 还没有真正 batched 执行，
 * 主要先通过 n_dim_hint 和 tile trace 把“未来硬件希望如何被喂饱”编码进软件合同。
 */

typedef enum {
    ACCEL_FMT_INT8 = 0,
    ACCEL_FMT_FP32,
    ACCEL_FMT_ACC32_RAW,
    ACCEL_FMT_MID16
} AccelFmt;

/*
 * AccelFmt:
 * 这里不是当前都真实用到了，而是提前给软件/硬件合同留出数据域。
 *
 * 当前实际使用：
 * - LINEAR 输入: INT8
 * - 大多数输出 : FP32
 *
 * 预留：
 * - ACC32_RAW: 以后可以直接承接 GEMM 输出的 int32 累加结果
 * - MID16    : 以后可以作为 norm/softmax/层间传递的中间格式
 */

typedef enum {
    ACCEL_SCALE_NONE = 0,
    ACCEL_SCALE_PER_TENSOR,
    ACCEL_SCALE_PER_ROW,
    ACCEL_SCALE_PER_TOKEN
} AccelScaleMode;

/*
 * AccelScaleMode:
 * 用于表达量化 contract，而不是表达具体实现细节。
 *
 * 当前主线假设：
 * - 输入激活: PER_TOKEN
 * - 权重    : PER_ROW
 */

typedef enum {
    ACCEL_BACKEND_SOFTWARE = 0,
    ACCEL_BACKEND_MOCK_MMIO,
    ACCEL_BACKEND_COMPARE
} AccelBackend;

/*
 * AccelBackend:
 * - SOFTWARE : 直接走原始软件数学，作为参考实现
 * - MOCK_MMIO: 先把 coarse-op 翻译成“模拟寄存器/descriptor”，再走软件数学
 * - COMPARE  : 同时跑 SOFTWARE 和 MOCK_MMIO，并逐元素比较输出
 *
 * 目的：
 * 在真正接 RTL 之前，先确认 descriptor 路径本身不会改坏数学。
 */

typedef struct {
    void* data;
    void* meta;
    AccelFmt fmt;
    AccelScaleMode scale_mode;
    int rows;
    int cols;
} AccelTensor;

/*
 * AccelTensor:
 * 软件/硬件交界处的“张量视图”。
 *
 * 这里显式区分：
 * - data : 主数据地址
 * - meta : sideband，比如 scale / metadata
 * - fmt  : 当前张量在哪个数据域里
 * - scale_mode : 这个张量的量化粒度
 *
 * 当前探索版还没有真的把这些地址映射到硬件，但 mock_mmio 已经会把它们写进 trace。
 */

typedef struct {
    AccelOp op;
    AccelPhase phase;
    int layer;
    int token_pos;
    int head_idx;
    int elem_count;
    int n_dim_hint;
    AccelTensor in_tensor;
    AccelTensor weight_tensor;
    AccelTensor out_tensor;
    AccelTensor aux_tensor0;
    AccelTensor aux_tensor1;
    QuantizedTensor* xq;
    QuantizedTensor* wq;
    float* out;
    int k_dim;
    int m_dim;
} AccelDesc;

/*
 * AccelDesc:
 * 粗粒度 op 的核心 descriptor。
 *
 * 可以把它理解成“以后真正给硬件提交任务时的软件合同”。
 *
 * 其中：
 * - op / phase / layer / token_pos / head_idx: 描述当前任务属于哪个子路径
 * - elem_count: 对 norm / softmax 这类逐元素或逐行 op 很重要
 * - n_dim_hint: 软件告诉硬件“这次更像 prefill N=4 还是 decode N=1”
 * - in/weight/out/aux*: 对应主输入、权重、输出，以及额外输入（如 key_cache / value_cache）
 * - xq/wq/out/k_dim/m_dim: 仍然保留给软件 fallback 内核复用，属于探索阶段的实现细节
 */

typedef struct {
    uint32_t opcode;
    uint32_t layer;
    uint32_t token_pos;
    uint32_t phase;
    uint32_t head_idx;
    uint32_t elem_count;
    uint32_t m_dim;
    uint32_t k_dim;
    uint32_t n_dim;
    uint32_t in_fmt;
    uint32_t weight_fmt;
    uint32_t out_fmt;
    uint32_t aux0_fmt;
    uint32_t aux1_fmt;
    uint32_t in_scale_mode;
    uint32_t w_scale_mode;
    uintptr_t in_data_addr;
    uintptr_t in_meta_addr;
    uintptr_t w_data_addr;
    uintptr_t w_meta_addr;
    uintptr_t aux0_data_addr;
    uintptr_t aux0_meta_addr;
    uintptr_t aux1_data_addr;
    uintptr_t aux1_meta_addr;
    uintptr_t out_data_addr;
} MockMmioLinearRegs;

/*
 * MockMmioLinearRegs:
 * 不是最终硬件寄存器定义，而是“把 AccelDesc 翻译成硬件友好字段”的模拟版本。
 *
 * 为什么要有它：
 * - 让我们在软件里先看到未来 MMIO/descriptor 大概会长什么样
 * - 让 trace 更接近真实硬件视角
 * - 以后 RTL 设计寄存器协议时，可以直接参考这里的字段集合
 */

typedef struct {
    int tile_idx;
    int m_tile_idx;
    int k_tile_idx;
    int n_tile_dim;
    int m_tile_dim;
    int k_tile_dim;
    int input_use_bank;
    int input_prefetch_bank;
    int output_bank;
    int prefetch_next;
    int k_tile_first;
    int k_tile_last;
} LinearTileTask;

/*
 * LinearTileTask:
 * 针对 LINEAR 类 op 的 tile 展开结果。
 *
 * 目的：
 * - 按当前硬件假设（M=32, K=64, N<=4）把一个 coarse LINEAR op 展开成多个 tile 任务
 * - 在软件里先模拟“如果要充分利用 ping-pong buffer，任务会怎么切”
 *
 * 注意：
 * - 这仍然只是 mock runtime，不是真实硬件执行
 * - 但它已经把 bank/use/prefetch/out 的语义带出来了
 */

typedef struct {
    AccelBackend backend;
    FILE* trace_fp;
    int prefill_chunk_hint;
    int active_n_hint;
    AccelFmt linear_out_fmt;
    AccelFmt norm_out_fmt;
    AccelFmt attn_out_fmt;
    uint64_t submit_count;
    uint64_t tile_submit_count;
    int next_input_use_bank;
    int next_input_prefetch_bank;
    int next_output_bank;
    int next_mid_bank;
    int next_score_bank;
} AccelRuntimeCtx;

/*
 * AccelRuntimeCtx:
 * 探索版 runtime 的全局上下文。
 *
 * 当前主要承担：
 * - 记录当前后端类型
 * - 记录 prefill 希望的 chunk hint
 * - 维护 mock ping-pong bank 的轮换状态
 * - 输出 trace
 */

static AccelRuntimeCtx g_rt = {ACCEL_BACKEND_SOFTWARE, NULL, 4, 1, ACCEL_FMT_FP32, ACCEL_FMT_FP32, ACCEL_FMT_FP32, 0, 0, 0, 1, 0, 0, 1};
static AccelPhase g_phase = ACCEL_PHASE_DECODE;

static const char* op_name(AccelOp op) {
    switch (op) {
        case ACCEL_OP_LINEAR_GENERIC: return "LINEAR_GENERIC";
        case ACCEL_OP_LINEAR_ATTN_Q:  return "LINEAR_ATTN_Q";
        case ACCEL_OP_LINEAR_ATTN_K:  return "LINEAR_ATTN_K";
        case ACCEL_OP_LINEAR_ATTN_V:  return "LINEAR_ATTN_V";
        case ACCEL_OP_LINEAR_ATTN_O:  return "LINEAR_ATTN_O";
        case ACCEL_OP_LINEAR_FFN_W1:  return "LINEAR_FFN_W1";
        case ACCEL_OP_LINEAR_FFN_W2:  return "LINEAR_FFN_W2";
        case ACCEL_OP_LINEAR_FFN_W3:  return "LINEAR_FFN_W3";
        case ACCEL_OP_LINEAR_CLS:     return "LINEAR_CLS";
        case ACCEL_OP_RMSNORM_ATTN:   return "RMSNORM_ATTN";
        case ACCEL_OP_RMSNORM_FFN:    return "RMSNORM_FFN";
        case ACCEL_OP_RMSNORM_FINAL:  return "RMSNORM_FINAL";
        case ACCEL_OP_QK_SCORE:       return "QK_SCORE";
        case ACCEL_OP_SOFTMAX_AV:     return "SOFTMAX_AV";
        default: return "UNKNOWN";
    }
}

static const char* phase_name(AccelPhase phase) {
    return (phase == ACCEL_PHASE_PREFILL) ? "PREFILL" : "DECODE";
}

static const char* fmt_name(AccelFmt fmt) {
    switch (fmt) {
        case ACCEL_FMT_INT8: return "INT8";
        case ACCEL_FMT_FP32: return "FP32";
        case ACCEL_FMT_ACC32_RAW: return "ACC32_RAW";
        case ACCEL_FMT_MID16: return "MID16";
        default: return "UNKNOWN_FMT";
    }
}

static const char* scale_name(AccelScaleMode mode) {
    switch (mode) {
        case ACCEL_SCALE_NONE: return "NONE";
        case ACCEL_SCALE_PER_TENSOR: return "PER_TENSOR";
        case ACCEL_SCALE_PER_ROW: return "PER_ROW";
        case ACCEL_SCALE_PER_TOKEN: return "PER_TOKEN";
        default: return "UNKNOWN_SCALE";
    }
}

static int ceil_div_i(int a, int b) {
    return (a + b - 1) / b;
}

static int min_i(int a, int b) {
    return (a < b) ? a : b;
}

static AccelFmt parse_fmt(const char* s) {
    if (strcmp(s, "fp32") == 0) return ACCEL_FMT_FP32;
    if (strcmp(s, "acc32") == 0) return ACCEL_FMT_ACC32_RAW;
    if (strcmp(s, "mid16") == 0) return ACCEL_FMT_MID16;
    if (strcmp(s, "int8") == 0) return ACCEL_FMT_INT8;
    fprintf(stderr, "unknown format: %s\n", s);
    exit(EXIT_FAILURE);
}

static float apply_vector_output_contract(float* data, int count, AccelFmt fmt) {
    float scale = 1.0f;
    if (fmt == ACCEL_FMT_FP32) {
        return scale;
    }
    if (fmt == ACCEL_FMT_ACC32_RAW) {
        for (int i = 0; i < count; i++) {
            int32_t q = (int32_t)lrintf(data[i]);
            data[i] = (float)q;
        }
        return scale;
    }
    if (fmt == ACCEL_FMT_MID16) {
        float maxabs = 0.0f;
        for (int i = 0; i < count; i++) {
            float a = fabsf(data[i]);
            if (a > maxabs) maxabs = a;
        }
        scale = (maxabs > 1e-8f) ? (maxabs / 32767.0f) : 1e-8f;
        for (int i = 0; i < count; i++) {
            int q = (int)lrintf(data[i] / scale);
            if (q > 32767) q = 32767;
            if (q < -32767) q = -32767;
            data[i] = ((float)q) * scale;
        }
        return scale;
    }
    if (fmt == ACCEL_FMT_INT8) {
        float maxabs = 0.0f;
        for (int i = 0; i < count; i++) {
            float a = fabsf(data[i]);
            if (a > maxabs) maxabs = a;
        }
        scale = (maxabs > 1e-8f) ? (maxabs / 127.0f) : 1e-8f;
        for (int i = 0; i < count; i++) {
            int q = (int)lrintf(data[i] / scale);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            data[i] = ((float)q) * scale;
        }
        return scale;
    }
    return scale;
}

static void apply_matrix_output_contract(float* data, int rows, int cols, AccelFmt fmt, float* meta) {
    for (int r = 0; r < rows; r++) {
        float scale = apply_vector_output_contract(data + r * cols, cols, fmt);
        if (meta != NULL) {
            meta[r] = scale;
        }
    }
}

/*
 * trace_linear_op:
 * 打印 coarse-op 级别的 descriptor trace。
 *
 * 这个 trace 更接近“CPU 提交给硬件的一次任务”。
 * 后面配合 trace_linear_tile，可以同时看到：
 * - 上层 coarse-op
 * - 下层 tile 展开
 */
static void trace_linear_op(const AccelDesc* desc, const MockMmioLinearRegs* regs) {
    FILE* fp = g_rt.trace_fp ? g_rt.trace_fp : stderr;
    fprintf(
        fp,
        "TRACE submit=%llu backend=%d phase=%s op=%s layer=%d token=%d head=%d elems=%d m=%d k=%d n_hint=%d in_fmt=%s out_fmt=%s in_scale=%s w_scale=%s in_data=0x%p in_meta=0x%p w_data=0x%p w_meta=0x%p aux0_data=0x%p aux0_meta=0x%p aux1_data=0x%p aux1_meta=0x%p out_data=0x%p mmio{opcode=%u layer=%u token=%u phase=%u head=%u elems=%u m=%u k=%u n=%u in_fmt=%u w_fmt=%u out_fmt=%u aux0_fmt=%u aux1_fmt=%u in_scale=%u w_scale=%u}\n",
        (unsigned long long)g_rt.submit_count,
        (int)g_rt.backend,
        phase_name(desc->phase),
        op_name(desc->op),
        desc->layer,
        desc->token_pos,
        desc->head_idx,
        desc->elem_count,
        desc->m_dim,
        desc->k_dim,
        desc->n_dim_hint,
        fmt_name(desc->in_tensor.fmt),
        fmt_name(desc->out_tensor.fmt),
        scale_name(desc->in_tensor.scale_mode),
        scale_name(desc->weight_tensor.scale_mode),
        desc->in_tensor.data,
        desc->in_tensor.meta,
        desc->weight_tensor.data,
        desc->weight_tensor.meta,
        desc->aux_tensor0.data,
        desc->aux_tensor0.meta,
        desc->aux_tensor1.data,
        desc->aux_tensor1.meta,
        desc->out_tensor.data,
        regs->opcode,
        regs->layer,
        regs->token_pos,
        regs->phase,
        regs->head_idx,
        regs->elem_count,
        regs->m_dim,
        regs->k_dim,
        regs->n_dim,
        regs->in_fmt,
        regs->weight_fmt,
        regs->out_fmt,
        regs->aux0_fmt,
        regs->aux1_fmt,
        regs->in_scale_mode,
        regs->w_scale_mode
    );
    if (g_rt.trace_fp) {
        fflush(g_rt.trace_fp);
    }
}

/*
 * trace_linear_tile:
 * 打印 LINEAR op 展开后的 tile 级 trace。
 *
 * 重点关注字段：
 * - use_bank
 * - prefetch_bank
 * - out_bank
 * - k_first / k_last
 *
 * 这些字段是为了模拟：
 * - 输入侧 ping-pong buffer
 * - 输出 bank 轮换
 * - K 维分块和 overlap
 */
static void trace_linear_tile(const AccelDesc* desc, const LinearTileTask* task) {
    FILE* fp = g_rt.trace_fp ? g_rt.trace_fp : stderr;
    fprintf(
        fp,
        "TRACE_TILE submit=%llu tile=%d op=%s layer=%d token=%d phase=%s m_tile=%d k_tile=%d m_dim=%d k_dim=%d n_dim=%d use_bank=%d prefetch_bank=%d out_bank=%d prefetch_next=%d k_first=%d k_last=%d\n",
        (unsigned long long)g_rt.tile_submit_count,
        task->tile_idx,
        op_name(desc->op),
        desc->layer,
        desc->token_pos,
        phase_name(desc->phase),
        task->m_tile_idx,
        task->k_tile_idx,
        task->m_tile_dim,
        task->k_tile_dim,
        task->n_tile_dim,
        task->input_use_bank,
        task->input_prefetch_bank,
        task->output_bank,
        task->prefetch_next,
        task->k_tile_first,
        task->k_tile_last
    );
    if (g_rt.trace_fp) {
        fflush(g_rt.trace_fp);
    }
}

/*
 * trace_linear_tiling_summary:
 * 打印一个 coarse LINEAR op 被切成多少 tile，以及开始时使用哪组 bank。
 * 这个摘要用来看“软件有没有开始按硬件的 tile/buffer 方式组织任务”。
 */
static void trace_linear_tiling_summary(const AccelDesc* desc, int task_count) {
    FILE* fp = g_rt.trace_fp ? g_rt.trace_fp : stderr;
    fprintf(
        fp,
        "TRACE_TILING_SUMMARY submit=%llu op=%s layer=%d token=%d phase=%s task_count=%d start_use_bank=%d start_prefetch_bank=%d start_out_bank=%d n_hint=%d\n",
        (unsigned long long)g_rt.submit_count,
        op_name(desc->op),
        desc->layer,
        desc->token_pos,
        phase_name(desc->phase),
        task_count,
        g_rt.next_input_use_bank,
        g_rt.next_input_prefetch_bank,
        g_rt.next_output_bank,
        desc->n_dim_hint
    );
    if (g_rt.trace_fp) {
        fflush(g_rt.trace_fp);
    }
}

/*
 * build_linear_tile_tasks:
 * 按当前硬件假设，把一个 coarse LINEAR op 切成多个 tile。
 *
 * 当前采用的固定假设来自现有 RTL：
 * - M tile = 32
 * - K tile = 64
 * - N hint <= 4
 *
 * 注意：
 * - 这里的 bank 轮换是“软件 mock 调度”
 * - 目的是验证软件是否已经开始按硬件数据流思考
 * - 不是精确模拟 RTL 所有细节
 */

static int build_linear_tile_tasks(const AccelDesc* desc, LinearTileTask* tasks, int max_tasks) {
    int m_tiles = ceil_div_i(desc->m_dim, 32);
    int k_tiles = ceil_div_i(desc->k_dim, 64);
    int n_tile_dim = min_i(desc->n_dim_hint <= 0 ? 1 : desc->n_dim_hint, 4);
    int count = 0;
    int use_bank_seed = g_rt.next_input_use_bank;
    int prefetch_bank_seed = g_rt.next_input_prefetch_bank;
    int out_bank = g_rt.next_output_bank;

    for (int mt = 0; mt < m_tiles; mt++) {
        int m_dim = min_i(desc->m_dim - mt * 32, 32);
        int use_bank = use_bank_seed;
        int prefetch_bank = prefetch_bank_seed;
        for (int kt = 0; kt < k_tiles; kt++) {
            int k_dim = min_i(desc->k_dim - kt * 64, 64);
            LinearTileTask* t;
            if (count >= max_tasks) {
                return count;
            }
            t = &tasks[count];
            memset(t, 0, sizeof(*t));
            t->tile_idx = count;
            t->m_tile_idx = mt;
            t->k_tile_idx = kt;
            t->n_tile_dim = n_tile_dim;
            t->m_tile_dim = m_dim;
            t->k_tile_dim = k_dim;
            t->input_use_bank = use_bank;
            t->input_prefetch_bank = prefetch_bank;
            t->output_bank = out_bank;
            t->prefetch_next = (kt + 1 < k_tiles) ? 1 : 0;
            t->k_tile_first = (kt == 0);
            t->k_tile_last = (kt + 1 == k_tiles);
            count++;
            if (kt + 1 < k_tiles) {
                int tmp = use_bank;
                use_bank = prefetch_bank;
                prefetch_bank = tmp;
            }
        }
    }

    g_rt.next_input_use_bank = g_rt.next_input_prefetch_bank;
    g_rt.next_input_prefetch_bank = 1 - g_rt.next_input_use_bank;
    g_rt.next_output_bank = 1 - g_rt.next_output_bank;
    return count;
}

static void execute_linear_op_software(const AccelDesc* desc, float* out) {
    base_matmul(out, desc->xq, desc->wq, desc->k_dim, desc->m_dim);
}

static void execute_linear_op_batched_software(
    const QuantizedTensor* x_cols,
    int cols,
    QuantizedTensor* wq,
    int k_dim,
    int m_dim,
    float* out_batch
) {
    int i, c;
    #pragma omp parallel for private(i, c)
    for (i = 0; i < m_dim; i++) {
        float vals[ACCEL_MAX_PREFILL_BATCH];
        int in = i * k_dim;
        for (c = 0; c < cols; c++) {
            vals[c] = 0.0f;
        }

        for (int j = 0; j <= k_dim - GS; j += GS) {
            int32_t ivals[ACCEL_MAX_PREFILL_BATCH];
            float w_scale = wq->s[(in + j) / GS];
            for (c = 0; c < cols; c++) {
                ivals[c] = 0;
            }
            for (int k = 0; k < GS; k++) {
                int8_t wv = wq->q[in + j + k];
                for (c = 0; c < cols; c++) {
                    ivals[c] += ((int32_t)x_cols[c].q[j + k]) * ((int32_t)wv);
                }
            }
            for (c = 0; c < cols; c++) {
                vals[c] += ((float)ivals[c]) * w_scale * x_cols[c].s[j / GS];
            }
        }

        for (c = 0; c < cols; c++) {
            out_batch[c * m_dim + i] = vals[c];
        }
    }
}

static void quantize_batch_contiguous(
    float* in_batch,
    int cols,
    int dim,
    int8_t* q_buf,
    float* s_buf,
    QuantizedTensor* qts
) {
    int groups = dim / GS;
    for (int c = 0; c < cols; c++) {
        qts[c].q = q_buf + c * dim;
        qts[c].s = s_buf + c * groups;
        quantize(&qts[c], in_batch + c * dim, dim);
    }
}

/*
 * execute_*_software:
 * 这些函数都调用 runq.c 里的原始数学，实现“软件参考路径”。
 *
 * 它们的目的不是提速，而是：
 * - 作为 compare backend 的 reference
 * - 保证 descriptor/mock 路径不改坏数学
 */
static void execute_rmsnorm_op_software(const float* in, float* out, const float* weight, int size) {
    rmsnorm(out, (float*)in, (float*)weight, size);
}

static void execute_qk_score_op_software(
    float* att,
    const float* q,
    const float* key_cache_base,
    int pos,
    int head_idx,
    int head_size,
    int kv_mul,
    int kv_dim
) {
    for (int t = 0; t <= pos; t++) {
        const float* k = key_cache_base + t * kv_dim + (head_idx / kv_mul) * head_size;
        float score = 0.0f;
        for (int i = 0; i < head_size; i++) {
            score += q[i] * k[i];
        }
        score /= sqrtf((float)head_size);
        att[t] = score;
    }
}

static void execute_softmax_av_op_software(
    float* out,
    float* att,
    const float* value_cache_base,
    int pos,
    int head_idx,
    int head_size,
    int kv_mul,
    int kv_dim
) {
    memset(out, 0, (size_t)head_size * sizeof(float));
    softmax(att, pos + 1);
    for (int t = 0; t <= pos; t++) {
        const float* v = value_cache_base + t * kv_dim + (head_idx / kv_mul) * head_size;
        float a = att[t];
        for (int i = 0; i < head_size; i++) {
            out[i] += a * v[i];
        }
    }
}

static void execute_rmsnorm_batch_software(
    float* out_batch,
    const float* in_batch,
    const float* weight,
    int rows,
    int elem_count
) {
    for (int r = 0; r < rows; r++) {
        rmsnorm(out_batch + r * elem_count, (float*)(in_batch + r * elem_count), (float*)weight, elem_count);
    }
}

static void execute_qk_score_heads_batched_software(
    float* att_base,
    const float* q_base,
    const float* key_cache_base,
    int pos,
    int n_heads,
    int head_size,
    int kv_mul,
    int kv_dim,
    int seq_len
) {
    for (int h = 0; h < n_heads; h++) {
        execute_qk_score_op_software(
            att_base + h * seq_len,
            q_base + h * head_size,
            key_cache_base,
            pos,
            h,
            head_size,
            kv_mul,
            kv_dim
        );
    }
}

static void execute_softmax_av_heads_batched_software(
    float* out_base,
    float* att_base,
    const float* value_cache_base,
    int pos,
    int n_heads,
    int head_size,
    int kv_mul,
    int kv_dim,
    int seq_len
) {
    for (int h = 0; h < n_heads; h++) {
        execute_softmax_av_op_software(
            out_base + h * head_size,
            att_base + h * seq_len,
            value_cache_base,
            pos,
            h,
            head_size,
            kv_mul,
            kv_dim
        );
    }
}

/*
 * build_mock_mmio_regs:
 * 把抽象 descriptor 翻译成“更像硬件寄存器”的字段集合。
 *
 * 当前这是软件和未来 RTL 之间最重要的桥。
 */
static void build_mock_mmio_regs(const AccelDesc* desc, MockMmioLinearRegs* regs) {
    regs->opcode = (uint32_t)desc->op;
    regs->layer = (uint32_t)(desc->layer + 1);
    regs->token_pos = (uint32_t)desc->token_pos;
    regs->phase = (uint32_t)desc->phase;
    regs->head_idx = (uint32_t)(desc->head_idx < 0 ? 0 : desc->head_idx);
    regs->elem_count = (uint32_t)desc->elem_count;
    regs->m_dim = (uint32_t)desc->m_dim;
    regs->k_dim = (uint32_t)desc->k_dim;
    regs->n_dim = (uint32_t)desc->n_dim_hint;
    regs->in_fmt = (uint32_t)desc->in_tensor.fmt;
    regs->weight_fmt = (uint32_t)desc->weight_tensor.fmt;
    regs->out_fmt = (uint32_t)desc->out_tensor.fmt;
    regs->aux0_fmt = (uint32_t)desc->aux_tensor0.fmt;
    regs->aux1_fmt = (uint32_t)desc->aux_tensor1.fmt;
    regs->in_scale_mode = (uint32_t)desc->in_tensor.scale_mode;
    regs->w_scale_mode = (uint32_t)desc->weight_tensor.scale_mode;
    regs->in_data_addr = (uintptr_t)desc->in_tensor.data;
    regs->in_meta_addr = (uintptr_t)desc->in_tensor.meta;
    regs->w_data_addr = (uintptr_t)desc->weight_tensor.data;
    regs->w_meta_addr = (uintptr_t)desc->weight_tensor.meta;
    regs->aux0_data_addr = (uintptr_t)desc->aux_tensor0.data;
    regs->aux0_meta_addr = (uintptr_t)desc->aux_tensor0.meta;
    regs->aux1_data_addr = (uintptr_t)desc->aux_tensor1.data;
    regs->aux1_meta_addr = (uintptr_t)desc->aux_tensor1.meta;
    regs->out_data_addr = (uintptr_t)desc->out_tensor.data;
}

/*
 * execute_linear_op_mock_mmio:
 * 先输出 coarse-op trace
 * 再输出 tile/bank trace
 * 最后仍然回落到软件数学执行
 *
 * 它的意义在于：
 * - 让软件开始“按硬件思维”组织任务
 * - 但又不需要真实硬件就能验证行为一致性
 */
static void execute_linear_op_mock_mmio(const AccelDesc* desc, float* out) {
    MockMmioLinearRegs regs;
    LinearTileTask tasks[256];
    int task_count;
    int start_use_bank = g_rt.next_input_use_bank;
    int start_prefetch_bank = g_rt.next_input_prefetch_bank;
    int start_out_bank = g_rt.next_output_bank;
    build_mock_mmio_regs(desc, &regs);
    trace_linear_op(desc, &regs);
    task_count = build_linear_tile_tasks(desc, tasks, 256);
    g_rt.next_input_use_bank = start_use_bank;
    g_rt.next_input_prefetch_bank = start_prefetch_bank;
    g_rt.next_output_bank = start_out_bank;
    trace_linear_tiling_summary(desc, task_count);
    for (int ti = 0; ti < task_count; ti++) {
        g_rt.tile_submit_count++;
        trace_linear_tile(desc, &tasks[ti]);
    }
    g_rt.next_input_use_bank = (task_count > 0) ? (1 - start_use_bank) : start_use_bank;
    g_rt.next_input_prefetch_bank = 1 - g_rt.next_input_use_bank;
    g_rt.next_output_bank = (task_count > 0) ? (1 - start_out_bank) : start_out_bank;
    execute_linear_op_software(desc, out);
}

/*
 * compare_*:
 * compare backend 的核心检查。
 *
 * 只要 mock 路径和软件路径任何一项输出不一致，就立刻退出。
 * 这样可以尽早发现 descriptor 组织是否引入了行为偏差。
 */
static void compare_linear_outputs(const AccelDesc* desc, const float* ref, const float* dut) {
    for (int i = 0; i < desc->m_dim; i++) {
        float diff = fabsf(ref[i] - dut[i]);
        if (diff > 1e-6f) {
            fprintf(
                stderr,
                "COMPARE_MISMATCH op=%s layer=%d token=%d idx=%d ref=%f dut=%f diff=%f\n",
                op_name(desc->op),
                desc->layer,
                desc->token_pos,
                i,
                ref[i],
                dut[i],
                diff
            );
            exit(EXIT_FAILURE);
        }
    }
}

static void compare_float_vectors(const char* tag, const AccelDesc* desc, const float* ref, const float* dut, int count) {
    for (int i = 0; i < count; i++) {
        float diff = fabsf(ref[i] - dut[i]);
        if (diff > 1e-6f) {
            fprintf(
                stderr,
                "COMPARE_MISMATCH tag=%s op=%s layer=%d token=%d head=%d idx=%d ref=%f dut=%f diff=%f\n",
                tag,
                op_name(desc->op),
                desc->layer,
                desc->token_pos,
                desc->head_idx,
                i,
                ref[i],
                dut[i],
                diff
            );
            exit(EXIT_FAILURE);
        }
    }
}

/*
 * submit_*:
 * 这是 coarse-op runtime 最核心的接口层。
 *
 * 未来如果真接硬件：
 * - 就是在这些函数里把 desc 写入 MMIO / 队列 / descriptor ring
 * - 然后等待硬件完成或轮询状态
 *
 * 当前阶段：
 * - SOFTWARE: 直接软件执行
 * - MOCK_MMIO: 先做 descriptor/trace，再软件执行
 * - COMPARE: 两边都跑，并逐元素比对
 */
static void submit_linear_op(const AccelDesc* desc) {
    float* out = desc->out;
    g_rt.submit_count++;
    switch (g_rt.backend) {
        case ACCEL_BACKEND_SOFTWARE:
            execute_linear_op_software(desc, out);
            apply_matrix_output_contract(out, desc->out_tensor.rows, desc->out_tensor.cols, desc->out_tensor.fmt, (float*)desc->out_tensor.meta);
            break;
        case ACCEL_BACKEND_MOCK_MMIO:
            execute_linear_op_mock_mmio(desc, out);
            apply_matrix_output_contract(out, desc->out_tensor.rows, desc->out_tensor.cols, desc->out_tensor.fmt, (float*)desc->out_tensor.meta);
            break;
        case ACCEL_BACKEND_COMPARE: {
            float* ref = (float*)malloc((size_t)desc->m_dim * sizeof(float));
            float* dut = (float*)malloc((size_t)desc->m_dim * sizeof(float));
            if (ref == NULL || dut == NULL) {
                fprintf(stderr, "malloc failed in compare backend\n");
                exit(EXIT_FAILURE);
            }
            execute_linear_op_software(desc, ref);
            apply_matrix_output_contract(ref, desc->out_tensor.rows, desc->out_tensor.cols, desc->out_tensor.fmt, NULL);
            execute_linear_op_mock_mmio(desc, dut);
            apply_matrix_output_contract(dut, desc->out_tensor.rows, desc->out_tensor.cols, desc->out_tensor.fmt, NULL);
            compare_linear_outputs(desc, ref, dut);
            memcpy(out, dut, (size_t)desc->m_dim * sizeof(float));
            free(ref);
            free(dut);
            break;
        }
        default:
            fprintf(stderr, "unknown backend %d\n", (int)g_rt.backend);
            exit(EXIT_FAILURE);
    }
}

static void submit_linear_op_batched(
    const AccelDesc* desc,
    QuantizedTensor* x_cols,
    int cols,
    float* out_batch
) {
    MockMmioLinearRegs regs;
    g_rt.submit_count++;
    build_mock_mmio_regs(desc, &regs);
    switch (g_rt.backend) {
        case ACCEL_BACKEND_SOFTWARE:
            execute_linear_op_batched_software(x_cols, cols, desc->wq, desc->k_dim, desc->m_dim, out_batch);
            apply_matrix_output_contract(out_batch, cols, desc->m_dim, desc->out_tensor.fmt, (float*)desc->out_tensor.meta);
            break;
        case ACCEL_BACKEND_MOCK_MMIO: {
            LinearTileTask tasks[256];
            int task_count;
            int start_use_bank = g_rt.next_input_use_bank;
            int start_prefetch_bank = g_rt.next_input_prefetch_bank;
            int start_out_bank = g_rt.next_output_bank;
            trace_linear_op(desc, &regs);
            task_count = build_linear_tile_tasks(desc, tasks, 256);
            g_rt.next_input_use_bank = start_use_bank;
            g_rt.next_input_prefetch_bank = start_prefetch_bank;
            g_rt.next_output_bank = start_out_bank;
            trace_linear_tiling_summary(desc, task_count);
            for (int ti = 0; ti < task_count; ti++) {
                g_rt.tile_submit_count++;
                trace_linear_tile(desc, &tasks[ti]);
            }
            g_rt.next_input_use_bank = (task_count > 0) ? (1 - start_use_bank) : start_use_bank;
            g_rt.next_input_prefetch_bank = 1 - g_rt.next_input_use_bank;
            g_rt.next_output_bank = (task_count > 0) ? (1 - start_out_bank) : start_out_bank;
            execute_linear_op_batched_software(x_cols, cols, desc->wq, desc->k_dim, desc->m_dim, out_batch);
            apply_matrix_output_contract(out_batch, cols, desc->m_dim, desc->out_tensor.fmt, (float*)desc->out_tensor.meta);
            break;
        }
        case ACCEL_BACKEND_COMPARE: {
            float* ref = (float*)malloc((size_t)cols * (size_t)desc->m_dim * sizeof(float));
            float* dut = (float*)malloc((size_t)cols * (size_t)desc->m_dim * sizeof(float));
            if (ref == NULL || dut == NULL) {
                fprintf(stderr, "malloc failed in compare batched linear backend\n");
                exit(EXIT_FAILURE);
            }
            for (int c = 0; c < cols; c++) {
                base_matmul(ref + c * desc->m_dim, &x_cols[c], desc->wq, desc->k_dim, desc->m_dim);
            }
            apply_matrix_output_contract(ref, cols, desc->m_dim, desc->out_tensor.fmt, NULL);
            {
                LinearTileTask tasks[256];
                int task_count;
                int start_use_bank = g_rt.next_input_use_bank;
                int start_prefetch_bank = g_rt.next_input_prefetch_bank;
                int start_out_bank = g_rt.next_output_bank;
                trace_linear_op(desc, &regs);
                task_count = build_linear_tile_tasks(desc, tasks, 256);
                g_rt.next_input_use_bank = start_use_bank;
                g_rt.next_input_prefetch_bank = start_prefetch_bank;
                g_rt.next_output_bank = start_out_bank;
                trace_linear_tiling_summary(desc, task_count);
                for (int ti = 0; ti < task_count; ti++) {
                    g_rt.tile_submit_count++;
                    trace_linear_tile(desc, &tasks[ti]);
                }
                g_rt.next_input_use_bank = (task_count > 0) ? (1 - start_use_bank) : start_use_bank;
                g_rt.next_input_prefetch_bank = 1 - g_rt.next_input_use_bank;
                g_rt.next_output_bank = (task_count > 0) ? (1 - start_out_bank) : start_out_bank;
            }
            execute_linear_op_batched_software(x_cols, cols, desc->wq, desc->k_dim, desc->m_dim, dut);
            apply_matrix_output_contract(dut, cols, desc->m_dim, desc->out_tensor.fmt, NULL);
            compare_float_vectors("linear_batch", desc, ref, dut, cols * desc->m_dim);
            memcpy(out_batch, dut, (size_t)cols * (size_t)desc->m_dim * sizeof(float));
            free(ref);
            free(dut);
            break;
        }
        default:
            fprintf(stderr, "unknown backend %d\n", (int)g_rt.backend);
            exit(EXIT_FAILURE);
    }
}

static void submit_rmsnorm_op(const AccelDesc* desc) {
    MockMmioLinearRegs regs;
    float* out = desc->out;
    g_rt.submit_count++;
    build_mock_mmio_regs(desc, &regs);
    switch (g_rt.backend) {
        case ACCEL_BACKEND_SOFTWARE:
            execute_rmsnorm_op_software((const float*)desc->in_tensor.data, out, (const float*)desc->weight_tensor.data, desc->elem_count);
            apply_matrix_output_contract(out, desc->out_tensor.rows, desc->out_tensor.cols, desc->out_tensor.fmt, (float*)desc->out_tensor.meta);
            break;
        case ACCEL_BACKEND_MOCK_MMIO:
            trace_linear_op(desc, &regs);
            execute_rmsnorm_op_software((const float*)desc->in_tensor.data, out, (const float*)desc->weight_tensor.data, desc->elem_count);
            apply_matrix_output_contract(out, desc->out_tensor.rows, desc->out_tensor.cols, desc->out_tensor.fmt, (float*)desc->out_tensor.meta);
            break;
        case ACCEL_BACKEND_COMPARE: {
            float* ref = (float*)malloc((size_t)desc->elem_count * sizeof(float));
            float* dut = (float*)malloc((size_t)desc->elem_count * sizeof(float));
            if (ref == NULL || dut == NULL) {
                fprintf(stderr, "malloc failed in compare rmsnorm backend\n");
                exit(EXIT_FAILURE);
            }
            execute_rmsnorm_op_software((const float*)desc->in_tensor.data, ref, (const float*)desc->weight_tensor.data, desc->elem_count);
            apply_matrix_output_contract(ref, desc->out_tensor.rows, desc->out_tensor.cols, desc->out_tensor.fmt, NULL);
            trace_linear_op(desc, &regs);
            execute_rmsnorm_op_software((const float*)desc->in_tensor.data, dut, (const float*)desc->weight_tensor.data, desc->elem_count);
            apply_matrix_output_contract(dut, desc->out_tensor.rows, desc->out_tensor.cols, desc->out_tensor.fmt, NULL);
            compare_float_vectors("rmsnorm", desc, ref, dut, desc->elem_count);
            memcpy(out, dut, (size_t)desc->elem_count * sizeof(float));
            free(ref);
            free(dut);
            break;
        }
        default:
            fprintf(stderr, "unknown backend %d\n", (int)g_rt.backend);
            exit(EXIT_FAILURE);
    }
}

static void submit_qk_score_op(
    const AccelDesc* desc,
    const float* q,
    const float* key_cache_base,
    int pos,
    int head_size,
    int kv_mul,
    int kv_dim
) {
    MockMmioLinearRegs regs;
    float* out = desc->out;
    g_rt.submit_count++;
    build_mock_mmio_regs(desc, &regs);
    switch (g_rt.backend) {
        case ACCEL_BACKEND_SOFTWARE:
            execute_qk_score_op_software(out, q, key_cache_base, pos, desc->head_idx, head_size, kv_mul, kv_dim);
            break;
        case ACCEL_BACKEND_MOCK_MMIO:
            trace_linear_op(desc, &regs);
            execute_qk_score_op_software(out, q, key_cache_base, pos, desc->head_idx, head_size, kv_mul, kv_dim);
            break;
        case ACCEL_BACKEND_COMPARE: {
            float* ref = (float*)malloc((size_t)(pos + 1) * sizeof(float));
            float* dut = (float*)malloc((size_t)(pos + 1) * sizeof(float));
            if (ref == NULL || dut == NULL) {
                fprintf(stderr, "malloc failed in compare qk backend\n");
                exit(EXIT_FAILURE);
            }
            execute_qk_score_op_software(ref, q, key_cache_base, pos, desc->head_idx, head_size, kv_mul, kv_dim);
            trace_linear_op(desc, &regs);
            execute_qk_score_op_software(dut, q, key_cache_base, pos, desc->head_idx, head_size, kv_mul, kv_dim);
            compare_float_vectors("qk_score", desc, ref, dut, pos + 1);
            memcpy(out, dut, (size_t)(pos + 1) * sizeof(float));
            free(ref);
            free(dut);
            break;
        }
        default:
            fprintf(stderr, "unknown backend %d\n", (int)g_rt.backend);
            exit(EXIT_FAILURE);
    }
}

static void submit_softmax_av_op(
    const AccelDesc* desc,
    float* att,
    const float* value_cache_base,
    int pos,
    int head_size,
    int kv_mul,
    int kv_dim
) {
    MockMmioLinearRegs regs;
    float* out = desc->out;
    g_rt.submit_count++;
    build_mock_mmio_regs(desc, &regs);
    switch (g_rt.backend) {
        case ACCEL_BACKEND_SOFTWARE:
            execute_softmax_av_op_software(out, att, value_cache_base, pos, desc->head_idx, head_size, kv_mul, kv_dim);
            apply_matrix_output_contract(out, desc->out_tensor.rows, desc->out_tensor.cols, desc->out_tensor.fmt, (float*)desc->out_tensor.meta);
            break;
        case ACCEL_BACKEND_MOCK_MMIO:
            trace_linear_op(desc, &regs);
            execute_softmax_av_op_software(out, att, value_cache_base, pos, desc->head_idx, head_size, kv_mul, kv_dim);
            apply_matrix_output_contract(out, desc->out_tensor.rows, desc->out_tensor.cols, desc->out_tensor.fmt, (float*)desc->out_tensor.meta);
            break;
        case ACCEL_BACKEND_COMPARE: {
            float* ref_att = (float*)malloc((size_t)(pos + 1) * sizeof(float));
            float* dut_att = (float*)malloc((size_t)(pos + 1) * sizeof(float));
            float* ref_out = (float*)malloc((size_t)head_size * sizeof(float));
            float* dut_out = (float*)malloc((size_t)head_size * sizeof(float));
            if (ref_att == NULL || dut_att == NULL || ref_out == NULL || dut_out == NULL) {
                fprintf(stderr, "malloc failed in compare softmax_av backend\n");
                exit(EXIT_FAILURE);
            }
            memcpy(ref_att, att, (size_t)(pos + 1) * sizeof(float));
            memcpy(dut_att, att, (size_t)(pos + 1) * sizeof(float));
            execute_softmax_av_op_software(ref_out, ref_att, value_cache_base, pos, desc->head_idx, head_size, kv_mul, kv_dim);
            apply_matrix_output_contract(ref_out, desc->out_tensor.rows, desc->out_tensor.cols, desc->out_tensor.fmt, NULL);
            trace_linear_op(desc, &regs);
            execute_softmax_av_op_software(dut_out, dut_att, value_cache_base, pos, desc->head_idx, head_size, kv_mul, kv_dim);
            apply_matrix_output_contract(dut_out, desc->out_tensor.rows, desc->out_tensor.cols, desc->out_tensor.fmt, NULL);
            compare_float_vectors("softmax_av_out", desc, ref_out, dut_out, head_size);
            compare_float_vectors("softmax_av_att", desc, ref_att, dut_att, pos + 1);
            memcpy(out, dut_out, (size_t)head_size * sizeof(float));
            memcpy(att, dut_att, (size_t)(pos + 1) * sizeof(float));
            free(ref_att);
            free(dut_att);
            free(ref_out);
            free(dut_out);
            break;
        }
        default:
            fprintf(stderr, "unknown backend %d\n", (int)g_rt.backend);
            exit(EXIT_FAILURE);
    }
}

static void submit_rmsnorm_batch_op(
    const AccelDesc* desc,
    float* out_batch,
    const float* in_batch,
    const float* weight,
    int rows,
    int elem_count
) {
    MockMmioLinearRegs regs;
    g_rt.submit_count++;
    build_mock_mmio_regs(desc, &regs);
    switch (g_rt.backend) {
        case ACCEL_BACKEND_SOFTWARE:
            execute_rmsnorm_batch_software(out_batch, in_batch, weight, rows, elem_count);
            apply_matrix_output_contract(out_batch, rows, elem_count, desc->out_tensor.fmt, (float*)desc->out_tensor.meta);
            break;
        case ACCEL_BACKEND_MOCK_MMIO:
            trace_linear_op(desc, &regs);
            fprintf(g_rt.trace_fp ? g_rt.trace_fp : stderr,
                    "TRACE_BATCH submit=%llu op=%s rows=%d cols=%d mid_bank=%d\n",
                    (unsigned long long)g_rt.submit_count,
                    op_name(desc->op),
                    rows,
                    elem_count,
                    g_rt.next_mid_bank);
            g_rt.next_mid_bank = 1 - g_rt.next_mid_bank;
            execute_rmsnorm_batch_software(out_batch, in_batch, weight, rows, elem_count);
            apply_matrix_output_contract(out_batch, rows, elem_count, desc->out_tensor.fmt, (float*)desc->out_tensor.meta);
            break;
        case ACCEL_BACKEND_COMPARE: {
            float* ref = (float*)malloc((size_t)rows * (size_t)elem_count * sizeof(float));
            float* dut = (float*)malloc((size_t)rows * (size_t)elem_count * sizeof(float));
            if (ref == NULL || dut == NULL) {
                fprintf(stderr, "malloc failed in compare rmsnorm batch backend\n");
                exit(EXIT_FAILURE);
            }
            execute_rmsnorm_batch_software(ref, in_batch, weight, rows, elem_count);
            apply_matrix_output_contract(ref, rows, elem_count, desc->out_tensor.fmt, NULL);
            trace_linear_op(desc, &regs);
            fprintf(g_rt.trace_fp ? g_rt.trace_fp : stderr,
                    "TRACE_BATCH submit=%llu op=%s rows=%d cols=%d mid_bank=%d\n",
                    (unsigned long long)g_rt.submit_count,
                    op_name(desc->op),
                    rows,
                    elem_count,
                    g_rt.next_mid_bank);
            g_rt.next_mid_bank = 1 - g_rt.next_mid_bank;
            execute_rmsnorm_batch_software(dut, in_batch, weight, rows, elem_count);
            apply_matrix_output_contract(dut, rows, elem_count, desc->out_tensor.fmt, NULL);
            compare_float_vectors("rmsnorm_batch", desc, ref, dut, rows * elem_count);
            memcpy(out_batch, dut, (size_t)rows * (size_t)elem_count * sizeof(float));
            free(ref);
            free(dut);
            break;
        }
        default:
            fprintf(stderr, "unknown backend %d\n", (int)g_rt.backend);
            exit(EXIT_FAILURE);
    }
}

static void submit_attn_heads_batch_op(
    const AccelDesc* desc,
    float* out_base,
    float* att_base,
    const float* q_base,
    const float* cache_base,
    int pos,
    int n_heads,
    int head_size,
    int kv_mul,
    int kv_dim,
    int seq_len,
    int is_softmax_av
) {
    MockMmioLinearRegs regs;
    g_rt.submit_count++;
    build_mock_mmio_regs(desc, &regs);
    switch (g_rt.backend) {
        case ACCEL_BACKEND_SOFTWARE:
            if (is_softmax_av) {
                execute_softmax_av_heads_batched_software(out_base, att_base, cache_base, pos, n_heads, head_size, kv_mul, kv_dim, seq_len);
                apply_matrix_output_contract(out_base, n_heads, head_size, desc->out_tensor.fmt, (float*)desc->out_tensor.meta);
            } else {
                execute_qk_score_heads_batched_software(att_base, q_base, cache_base, pos, n_heads, head_size, kv_mul, kv_dim, seq_len);
            }
            break;
        case ACCEL_BACKEND_MOCK_MMIO:
            trace_linear_op(desc, &regs);
            fprintf(g_rt.trace_fp ? g_rt.trace_fp : stderr,
                    "TRACE_BATCH submit=%llu op=%s heads=%d elems=%d score_bank=%d out_bank=%d\n",
                    (unsigned long long)g_rt.submit_count,
                    op_name(desc->op),
                    n_heads,
                    pos + 1,
                    g_rt.next_score_bank,
                    g_rt.next_output_bank);
            if (is_softmax_av) {
                execute_softmax_av_heads_batched_software(out_base, att_base, cache_base, pos, n_heads, head_size, kv_mul, kv_dim, seq_len);
                apply_matrix_output_contract(out_base, n_heads, head_size, desc->out_tensor.fmt, (float*)desc->out_tensor.meta);
                g_rt.next_output_bank = 1 - g_rt.next_output_bank;
            } else {
                execute_qk_score_heads_batched_software(att_base, q_base, cache_base, pos, n_heads, head_size, kv_mul, kv_dim, seq_len);
                g_rt.next_score_bank = 1 - g_rt.next_score_bank;
            }
            break;
        case ACCEL_BACKEND_COMPARE:
            trace_linear_op(desc, &regs);
            fprintf(g_rt.trace_fp ? g_rt.trace_fp : stderr,
                    "TRACE_BATCH submit=%llu op=%s heads=%d elems=%d score_bank=%d out_bank=%d\n",
                    (unsigned long long)g_rt.submit_count,
                    op_name(desc->op),
                    n_heads,
                    pos + 1,
                    g_rt.next_score_bank,
                    g_rt.next_output_bank);
            if (is_softmax_av) {
                execute_softmax_av_heads_batched_software(out_base, att_base, cache_base, pos, n_heads, head_size, kv_mul, kv_dim, seq_len);
                apply_matrix_output_contract(out_base, n_heads, head_size, desc->out_tensor.fmt, (float*)desc->out_tensor.meta);
                g_rt.next_output_bank = 1 - g_rt.next_output_bank;
            } else {
                execute_qk_score_heads_batched_software(att_base, q_base, cache_base, pos, n_heads, head_size, kv_mul, kv_dim, seq_len);
                g_rt.next_score_bank = 1 - g_rt.next_score_bank;
            }
            break;
        default:
            fprintf(stderr, "unknown backend %d\n", (int)g_rt.backend);
            exit(EXIT_FAILURE);
    }
}

static void run_linear_op(
    AccelOp op,
    int layer,
    int token_pos,
    float* out,
    QuantizedTensor* xq,
    QuantizedTensor* wq,
    int k_dim,
    int m_dim
) {
    AccelDesc desc = {
        .op = op,
        .phase = g_phase,
        .layer = layer,
        .token_pos = token_pos,
        .n_dim_hint = (g_phase == ACCEL_PHASE_PREFILL) ? g_rt.prefill_chunk_hint : 1,
        .in_tensor = {
            .data = xq->q,
            .meta = xq->s,
            .fmt = ACCEL_FMT_INT8,
            .scale_mode = ACCEL_SCALE_PER_TOKEN,
            .rows = 1,
            .cols = k_dim
        },
        .weight_tensor = {
            .data = wq->q,
            .meta = wq->s,
            .fmt = ACCEL_FMT_INT8,
            .scale_mode = ACCEL_SCALE_PER_ROW,
            .rows = m_dim,
            .cols = k_dim
        },
        .out_tensor = {
            .data = out,
            .meta = NULL,
            .fmt = g_rt.linear_out_fmt,
            .scale_mode = ACCEL_SCALE_NONE,
            .rows = 1,
            .cols = m_dim
        },
        .xq = xq,
        .wq = wq,
        .out = out,
        .k_dim = k_dim,
        .m_dim = m_dim
    };
    submit_linear_op(&desc);
}

static void run_linear_op_batched(
    AccelOp op,
    int layer,
    int token_pos,
    float* out_batch,
    QuantizedTensor* x_cols,
    void* in_q_data,
    void* in_q_meta,
    int cols,
    QuantizedTensor* wq,
    int k_dim,
    int m_dim
) {
    AccelDesc desc;
    memset(&desc, 0, sizeof(desc));
    desc.op = op;
    desc.phase = g_phase;
    desc.layer = layer;
    desc.token_pos = token_pos;
    desc.head_idx = 0;
    desc.elem_count = cols;
    desc.n_dim_hint = cols;
    desc.in_tensor.data = in_q_data;
    desc.in_tensor.meta = in_q_meta;
    desc.in_tensor.fmt = ACCEL_FMT_INT8;
    desc.in_tensor.scale_mode = ACCEL_SCALE_PER_TOKEN;
    desc.in_tensor.rows = cols;
    desc.in_tensor.cols = k_dim;
    desc.weight_tensor.data = wq->q;
    desc.weight_tensor.meta = wq->s;
    desc.weight_tensor.fmt = ACCEL_FMT_INT8;
    desc.weight_tensor.scale_mode = ACCEL_SCALE_PER_ROW;
    desc.weight_tensor.rows = m_dim;
    desc.weight_tensor.cols = k_dim;
    desc.out_tensor.data = out_batch;
    desc.out_tensor.fmt = g_rt.linear_out_fmt;
    desc.out_tensor.scale_mode = ACCEL_SCALE_NONE;
    desc.out_tensor.rows = cols;
    desc.out_tensor.cols = m_dim;
    desc.wq = wq;
    desc.k_dim = k_dim;
    desc.m_dim = m_dim;
    submit_linear_op_batched(&desc, x_cols, cols, out_batch);
}

/*
 * run_*_op:
 * 这些是 forward() 中真正调用的“软件提交 op”的入口。
 *
 * 它们把当前模型内部的局部变量（q/k/v、权重、cache、输出缓冲）
 * 翻译成统一的 AccelDesc。
 */

static void run_rmsnorm_op(
    AccelOp op,
    int layer,
    int token_pos,
    float* out,
    float* in,
    float* weight,
    int elem_count
) {
    AccelDesc desc;
    memset(&desc, 0, sizeof(desc));
    desc.op = op;
    desc.phase = g_phase;
    desc.layer = layer;
    desc.token_pos = token_pos;
    desc.head_idx = -1;
    desc.elem_count = elem_count;
    desc.n_dim_hint = g_rt.active_n_hint;
    desc.in_tensor.data = in;
    desc.in_tensor.fmt = ACCEL_FMT_FP32;
    desc.in_tensor.scale_mode = ACCEL_SCALE_NONE;
    desc.in_tensor.rows = 1;
    desc.in_tensor.cols = elem_count;
    desc.weight_tensor.data = weight;
    desc.weight_tensor.fmt = ACCEL_FMT_FP32;
    desc.weight_tensor.scale_mode = ACCEL_SCALE_NONE;
    desc.weight_tensor.rows = 1;
    desc.weight_tensor.cols = elem_count;
    desc.out_tensor.data = out;
    desc.out_tensor.fmt = g_rt.norm_out_fmt;
    desc.out_tensor.scale_mode = ACCEL_SCALE_NONE;
    desc.out_tensor.rows = 1;
    desc.out_tensor.cols = elem_count;
    desc.out = out;
    submit_rmsnorm_op(&desc);
}

static void run_rmsnorm_batch(
    AccelOp op,
    int layer,
    int start_token_pos,
    float* out_batch,
    float* in_batch,
    float* weight,
    int cols,
    int elem_count
) {
    AccelDesc desc;
    memset(&desc, 0, sizeof(desc));
    desc.op = op;
    desc.phase = g_phase;
    desc.layer = layer;
    desc.token_pos = start_token_pos;
    desc.head_idx = -1;
    desc.elem_count = elem_count;
    desc.n_dim_hint = cols;
    desc.in_tensor.data = in_batch;
    desc.in_tensor.fmt = ACCEL_FMT_FP32;
    desc.in_tensor.scale_mode = ACCEL_SCALE_NONE;
    desc.in_tensor.rows = cols;
    desc.in_tensor.cols = elem_count;
    desc.weight_tensor.data = weight;
    desc.weight_tensor.fmt = ACCEL_FMT_FP32;
    desc.weight_tensor.scale_mode = ACCEL_SCALE_NONE;
    desc.weight_tensor.rows = 1;
    desc.weight_tensor.cols = elem_count;
    desc.out_tensor.data = out_batch;
    desc.out_tensor.fmt = g_rt.norm_out_fmt;
    desc.out_tensor.scale_mode = ACCEL_SCALE_NONE;
    desc.out_tensor.rows = cols;
    desc.out_tensor.cols = elem_count;
    submit_rmsnorm_batch_op(&desc, out_batch, in_batch, weight, cols, elem_count);
}

static void run_qk_score_heads_batch(
    int layer,
    int token_pos,
    float* att_base,
    float* q_base,
    float* key_cache_base,
    int n_heads,
    int elem_count,
    int head_size,
    int kv_mul,
    int kv_dim,
    int seq_len
) {
    AccelDesc desc;
    memset(&desc, 0, sizeof(desc));
    desc.op = ACCEL_OP_QK_SCORE;
    desc.phase = g_phase;
    desc.layer = layer;
    desc.token_pos = token_pos;
    desc.head_idx = -1;
    desc.elem_count = elem_count;
    desc.n_dim_hint = g_rt.active_n_hint;
    desc.in_tensor.data = q_base;
    desc.in_tensor.fmt = ACCEL_FMT_FP32;
    desc.in_tensor.scale_mode = ACCEL_SCALE_NONE;
    desc.in_tensor.rows = n_heads;
    desc.in_tensor.cols = head_size;
    desc.aux_tensor0.data = key_cache_base;
    desc.aux_tensor0.fmt = ACCEL_FMT_FP32;
    desc.aux_tensor0.scale_mode = ACCEL_SCALE_NONE;
    desc.out_tensor.data = att_base;
    desc.out_tensor.fmt = ACCEL_FMT_FP32;
    desc.out_tensor.scale_mode = ACCEL_SCALE_NONE;
    desc.out_tensor.rows = n_heads;
    desc.out_tensor.cols = elem_count;
    submit_attn_heads_batch_op(&desc, NULL, att_base, q_base, key_cache_base, token_pos, n_heads, head_size, kv_mul, kv_dim, seq_len, 0);
}

static void run_softmax_av_heads_batch(
    int layer,
    int token_pos,
    float* out_base,
    float* att_base,
    float* value_cache_base,
    int n_heads,
    int elem_count,
    int head_size,
    int kv_mul,
    int kv_dim,
    int seq_len
) {
    AccelDesc desc;
    memset(&desc, 0, sizeof(desc));
    desc.op = ACCEL_OP_SOFTMAX_AV;
    desc.phase = g_phase;
    desc.layer = layer;
    desc.token_pos = token_pos;
    desc.head_idx = -1;
    desc.elem_count = elem_count;
    desc.n_dim_hint = g_rt.active_n_hint;
    desc.in_tensor.data = att_base;
    desc.in_tensor.fmt = ACCEL_FMT_FP32;
    desc.in_tensor.scale_mode = ACCEL_SCALE_NONE;
    desc.in_tensor.rows = n_heads;
    desc.in_tensor.cols = elem_count;
    desc.aux_tensor0.data = value_cache_base;
    desc.aux_tensor0.fmt = ACCEL_FMT_FP32;
    desc.aux_tensor0.scale_mode = ACCEL_SCALE_NONE;
    desc.out_tensor.data = out_base;
    desc.out_tensor.fmt = g_rt.attn_out_fmt;
    desc.out_tensor.scale_mode = ACCEL_SCALE_NONE;
    desc.out_tensor.rows = n_heads;
    desc.out_tensor.cols = head_size;
    submit_attn_heads_batch_op(&desc, out_base, att_base, NULL, value_cache_base, token_pos, n_heads, head_size, kv_mul, kv_dim, seq_len, 1);
}

static void run_qk_score_op(
    int layer,
    int token_pos,
    int head_idx,
    float* att_out,
    float* q,
    float* key_cache_base,
    int elem_count,
    int head_size,
    int kv_mul,
    int kv_dim
) {
    AccelDesc desc;
    memset(&desc, 0, sizeof(desc));
    desc.op = ACCEL_OP_QK_SCORE;
    desc.phase = g_phase;
    desc.layer = layer;
    desc.token_pos = token_pos;
    desc.head_idx = head_idx;
    desc.elem_count = elem_count;
    desc.n_dim_hint = g_rt.active_n_hint;
    desc.in_tensor.data = q;
    desc.in_tensor.fmt = ACCEL_FMT_FP32;
    desc.in_tensor.scale_mode = ACCEL_SCALE_NONE;
    desc.in_tensor.rows = 1;
    desc.in_tensor.cols = head_size;
    desc.aux_tensor0.data = key_cache_base;
    desc.aux_tensor0.fmt = ACCEL_FMT_FP32;
    desc.aux_tensor0.scale_mode = ACCEL_SCALE_NONE;
    desc.out_tensor.data = att_out;
    desc.out_tensor.fmt = ACCEL_FMT_FP32;
    desc.out_tensor.scale_mode = ACCEL_SCALE_NONE;
    desc.out_tensor.rows = 1;
    desc.out_tensor.cols = elem_count;
    desc.out = att_out;
    submit_qk_score_op(&desc, q, key_cache_base, token_pos, head_size, kv_mul, kv_dim);
}

static void run_softmax_av_op(
    int layer,
    int token_pos,
    int head_idx,
    float* out,
    float* att,
    float* value_cache_base,
    int elem_count,
    int head_size,
    int kv_mul,
    int kv_dim
) {
    AccelDesc desc;
    memset(&desc, 0, sizeof(desc));
    desc.op = ACCEL_OP_SOFTMAX_AV;
    desc.phase = g_phase;
    desc.layer = layer;
    desc.token_pos = token_pos;
    desc.head_idx = head_idx;
    desc.elem_count = elem_count;
    desc.n_dim_hint = g_rt.active_n_hint;
    desc.in_tensor.data = att;
    desc.in_tensor.fmt = ACCEL_FMT_FP32;
    desc.in_tensor.scale_mode = ACCEL_SCALE_NONE;
    desc.in_tensor.rows = 1;
    desc.in_tensor.cols = elem_count;
    desc.aux_tensor0.data = value_cache_base;
    desc.aux_tensor0.fmt = ACCEL_FMT_FP32;
    desc.aux_tensor0.scale_mode = ACCEL_SCALE_NONE;
    desc.out_tensor.data = out;
    desc.out_tensor.fmt = g_rt.attn_out_fmt;
    desc.out_tensor.scale_mode = ACCEL_SCALE_NONE;
    desc.out_tensor.rows = 1;
    desc.out_tensor.cols = head_size;
    desc.out = out;
    submit_softmax_av_op(&desc, att, value_cache_base, token_pos, head_size, kv_mul, kv_dim);
}

float* forward(Transformer* transformer, int token, int pos) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    memcpy(x, w->token_embedding_table + token*dim, dim * sizeof(float));

    for (int l = 0; l < p->n_layers; l++) {
        /*
         * 注意这里的 forward() 已经不再只是“直接写数学”。
         * 它是在模拟未来的软件 runtime：
         * - 先提交 RMSNORM_ATTN
         * - 再提交 LINEAR_ATTN_Q/K/V
         * - 再提交 QK_SCORE / SOFTMAX_AV
         * - 再提交 LINEAR_ATTN_O
         * - 再提交 RMSNORM_FFN
         * - 再提交 FFN 线性层
         *
         * 也就是说，现在已经是“按粗粒度 op 图来组织软件执行”了。
         */
        run_rmsnorm_op(ACCEL_OP_RMSNORM_ATTN, l, pos, s->xb, x, w->rms_att_weight + l*dim, dim);

        quantize(&s->xq, s->xb, dim);
        run_linear_op(ACCEL_OP_LINEAR_ATTN_Q, l, pos, s->q, &s->xq, w->wq + l, dim, dim);
        run_linear_op(ACCEL_OP_LINEAR_ATTN_K, l, pos, s->k, &s->xq, w->wk + l, dim, kv_dim);
        run_linear_op(ACCEL_OP_LINEAR_ATTN_V, l, pos, s->v, &s->xq, w->wv + l, dim, kv_dim);

        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1;
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k;
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        {
            int loff = l * p->seq_len * kv_dim;
            float* key_cache_row = s->key_cache + loff + pos * kv_dim;
            float* value_cache_row = s->value_cache + loff + pos * kv_dim;
            memcpy(key_cache_row, s->k, kv_dim * sizeof(*key_cache_row));
            memcpy(value_cache_row, s->v, kv_dim * sizeof(*value_cache_row));

            #pragma omp parallel for
            for (int h = 0; h < p->n_heads; h++) {
                float* q = s->q + h * head_size;
                float* att = s->att + h * p->seq_len;
                {
                    float* xb = s->xb + h * head_size;
                    run_qk_score_op(l, pos, h, att, q, s->key_cache + loff, pos + 1, head_size, kv_mul, kv_dim);
                    run_softmax_av_op(l, pos, h, xb, att, s->value_cache + loff, pos + 1, head_size, kv_mul, kv_dim);
                }
            }
        }

        quantize(&s->xq, s->xb, dim);
        run_linear_op(ACCEL_OP_LINEAR_ATTN_O, l, pos, s->xb2, &s->xq, w->wo + l, dim, dim);

        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        run_rmsnorm_op(ACCEL_OP_RMSNORM_FFN, l, pos, s->xb, x, w->rms_ffn_weight + l*dim, dim);

        quantize(&s->xq, s->xb, dim);
        run_linear_op(ACCEL_OP_LINEAR_FFN_W1, l, pos, s->hb, &s->xq, w->w1 + l, dim, hidden_dim);
        run_linear_op(ACCEL_OP_LINEAR_FFN_W3, l, pos, s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);

        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        quantize(&s->hq, s->hb, hidden_dim);
        run_linear_op(ACCEL_OP_LINEAR_FFN_W2, l, pos, s->xb, &s->hq, w->w2 + l, hidden_dim, dim);

        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    run_rmsnorm_op(ACCEL_OP_RMSNORM_FINAL, -1, pos, x, x, w->rms_final_weight, dim);
    quantize(&s->xq, x, dim);
    run_linear_op(ACCEL_OP_LINEAR_CLS, -1, pos, s->logits, &s->xq, w->wcls, dim, p->vocab_size);
    return s->logits;
}

static int run_prefill_phase(
    Transformer *transformer,
    Tokenizer *tokenizer,
    int *prompt_tokens,
    int num_prompt_tokens,
    int steps,
    int *token_io,
    int *pos_io,
    long *start_io
) {
    g_phase = ACCEL_PHASE_PREFILL;
    while (*pos_io < steps && *pos_io < num_prompt_tokens - 1) {
        /*
         * 这是保留的“单 token prefill”路径。
         *
         * 当前主线已经改成 run_prefill_phase_batched()，
         * 这里主要保留作对照和局部调试使用。
         *
         * 即便如此，这里仍会维护 active_n_hint，
         * 让 descriptor / trace 保持和新的软件合同一致。
         */
        int remaining_prompt_edges = (num_prompt_tokens - 1) - *pos_io;
        g_rt.active_n_hint = (remaining_prompt_edges < g_rt.prefill_chunk_hint) ? remaining_prompt_edges : g_rt.prefill_chunk_hint;
        if (g_rt.active_n_hint <= 0) {
            g_rt.active_n_hint = 1;
        }
        int token = *token_io;
        int pos = *pos_io;
        int next = prompt_tokens[pos + 1];

        forward(transformer, token, pos);
        (*pos_io)++;

        if (next == 1) {
            *token_io = next;
            return 0;
        }

        {
            char* piece = decode(tokenizer, token, next);
            safe_printf(piece);
            fflush(stdout);
        }

        *token_io = next;
        if (*start_io == 0) {
            *start_io = time_in_ms();
        }
    }
    return 1;
}

static void process_prefill_chunk(
    Transformer* transformer,
    int* prompt_tokens,
    int start_pos,
    int cols
) {
    /*
     * process_prefill_chunk:
     * batched prefill 的核心。
     *
     * 这里做的事情不是“只给 n_dim_hint=4”，
     * 而是真正按 cols<=4 一次处理多个 token 的：
     * - RMSNorm
     * - Q/K/V 线性层
     * - QK_SCORE / SOFTMAX_AV
     * - O 投影
     * - FFN 的 W1/W3/W2
     *
     * 这就是当前软件里最接近“尽可能喂满 32x4 硬件”的执行原型。
     */
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;
    int groups_dim = dim / GS;
    int groups_hidden = hidden_dim / GS;

    float* x_batch = (float*)calloc((size_t)cols * (size_t)dim, sizeof(float));
    float* xb_batch = (float*)calloc((size_t)cols * (size_t)dim, sizeof(float));
    float* xb2_batch = (float*)calloc((size_t)cols * (size_t)dim, sizeof(float));
    float* q_batch = (float*)calloc((size_t)cols * (size_t)dim, sizeof(float));
    float* k_batch = (float*)calloc((size_t)cols * (size_t)kv_dim, sizeof(float));
    float* v_batch = (float*)calloc((size_t)cols * (size_t)kv_dim, sizeof(float));
    float* hb_batch = (float*)calloc((size_t)cols * (size_t)hidden_dim, sizeof(float));
    float* hb2_batch = (float*)calloc((size_t)cols * (size_t)hidden_dim, sizeof(float));
    int8_t* xq_buf = (int8_t*)calloc((size_t)cols * (size_t)dim, sizeof(int8_t));
    float* xq_s_buf = (float*)calloc((size_t)cols * (size_t)groups_dim, sizeof(float));
    int8_t* hq_buf = (int8_t*)calloc((size_t)cols * (size_t)hidden_dim, sizeof(int8_t));
    float* hq_s_buf = (float*)calloc((size_t)cols * (size_t)groups_hidden, sizeof(float));
    QuantizedTensor* xq_cols = (QuantizedTensor*)calloc((size_t)cols, sizeof(QuantizedTensor));
    QuantizedTensor* hq_cols = (QuantizedTensor*)calloc((size_t)cols, sizeof(QuantizedTensor));

    if (!x_batch || !xb_batch || !xb2_batch || !q_batch || !k_batch || !v_batch ||
        !hb_batch || !hb2_batch || !xq_buf || !xq_s_buf || !hq_buf || !hq_s_buf ||
        !xq_cols || !hq_cols) {
        fprintf(stderr, "malloc failed in process_prefill_chunk\n");
        exit(EXIT_FAILURE);
    }

    for (int c = 0; c < cols; c++) {
        memcpy(
            x_batch + c * dim,
            w->token_embedding_table + prompt_tokens[start_pos + c] * dim,
            (size_t)dim * sizeof(float)
        );
    }

    for (int l = 0; l < p->n_layers; l++) {
        run_rmsnorm_batch(ACCEL_OP_RMSNORM_ATTN, l, start_pos, xb_batch, x_batch, w->rms_att_weight + l * dim, cols, dim);

        quantize_batch_contiguous(xb_batch, cols, dim, xq_buf, xq_s_buf, xq_cols);
        run_linear_op_batched(ACCEL_OP_LINEAR_ATTN_Q, l, start_pos, q_batch, xq_cols, xq_buf, xq_s_buf, cols, w->wq + l, dim, dim);
        run_linear_op_batched(ACCEL_OP_LINEAR_ATTN_K, l, start_pos, k_batch, xq_cols, xq_buf, xq_s_buf, cols, w->wk + l, dim, kv_dim);
        run_linear_op_batched(ACCEL_OP_LINEAR_ATTN_V, l, start_pos, v_batch, xq_cols, xq_buf, xq_s_buf, cols, w->wv + l, dim, kv_dim);

        for (int c = 0; c < cols; c++) {
            int pos = start_pos + c;
            float* qv = q_batch + c * dim;
            float* kv = k_batch + c * kv_dim;
            float* vv = v_batch + c * kv_dim;

            for (int i = 0; i < dim; i += 2) {
                int head_dim = i % head_size;
                float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
                float val = pos * freq;
                float fcr = cosf(val);
                float fci = sinf(val);
                int rotn = i < kv_dim ? 2 : 1;
                for (int v = 0; v < rotn; v++) {
                    float* vec = v == 0 ? qv : kv;
                    float v0 = vec[i];
                    float v1 = vec[i + 1];
                    vec[i] = v0 * fcr - v1 * fci;
                    vec[i + 1] = v0 * fci + v1 * fcr;
                }
            }

            memcpy(s->key_cache + l * p->seq_len * kv_dim + pos * kv_dim, kv, (size_t)kv_dim * sizeof(float));
            memcpy(s->value_cache + l * p->seq_len * kv_dim + pos * kv_dim, vv, (size_t)kv_dim * sizeof(float));
        }

        memset(xb_batch, 0, (size_t)cols * (size_t)dim * sizeof(float));
        for (int c = 0; c < cols; c++) {
            int pos = start_pos + c;
            int loff = l * p->seq_len * kv_dim;
            float* qv = q_batch + c * dim;
            float* att_base = s->att;
            float* xb_heads = xb_batch + c * dim;
            run_qk_score_heads_batch(l, pos, att_base, qv, s->key_cache + loff, p->n_heads, pos + 1, head_size, kv_mul, kv_dim, p->seq_len);
            run_softmax_av_heads_batch(l, pos, xb_heads, att_base, s->value_cache + loff, p->n_heads, pos + 1, head_size, kv_mul, kv_dim, p->seq_len);
        }

        quantize_batch_contiguous(xb_batch, cols, dim, xq_buf, xq_s_buf, xq_cols);
        run_linear_op_batched(ACCEL_OP_LINEAR_ATTN_O, l, start_pos, xb2_batch, xq_cols, xq_buf, xq_s_buf, cols, w->wo + l, dim, dim);
        for (int c = 0; c < cols; c++) {
            for (int i = 0; i < dim; i++) {
                x_batch[c * dim + i] += xb2_batch[c * dim + i];
            }
        }

        run_rmsnorm_batch(ACCEL_OP_RMSNORM_FFN, l, start_pos, xb_batch, x_batch, w->rms_ffn_weight + l * dim, cols, dim);
        quantize_batch_contiguous(xb_batch, cols, dim, xq_buf, xq_s_buf, xq_cols);
        run_linear_op_batched(ACCEL_OP_LINEAR_FFN_W1, l, start_pos, hb_batch, xq_cols, xq_buf, xq_s_buf, cols, w->w1 + l, dim, hidden_dim);
        run_linear_op_batched(ACCEL_OP_LINEAR_FFN_W3, l, start_pos, hb2_batch, xq_cols, xq_buf, xq_s_buf, cols, w->w3 + l, dim, hidden_dim);

        for (int c = 0; c < cols; c++) {
            for (int i = 0; i < hidden_dim; i++) {
                float val = hb_batch[c * hidden_dim + i];
                val *= (1.0f / (1.0f + expf(-val)));
                val *= hb2_batch[c * hidden_dim + i];
                hb_batch[c * hidden_dim + i] = val;
            }
        }

        quantize_batch_contiguous(hb_batch, cols, hidden_dim, hq_buf, hq_s_buf, hq_cols);
        run_linear_op_batched(ACCEL_OP_LINEAR_FFN_W2, l, start_pos, xb_batch, hq_cols, hq_buf, hq_s_buf, cols, w->w2 + l, hidden_dim, dim);
        for (int c = 0; c < cols; c++) {
            for (int i = 0; i < dim; i++) {
                x_batch[c * dim + i] += xb_batch[c * dim + i];
            }
        }
    }

    free(x_batch);
    free(xb_batch);
    free(xb2_batch);
    free(q_batch);
    free(k_batch);
    free(v_batch);
    free(hb_batch);
    free(hb2_batch);
    free(xq_buf);
    free(xq_s_buf);
    free(hq_buf);
    free(hq_s_buf);
    free(xq_cols);
    free(hq_cols);
}

static int run_prefill_phase_batched(
    Transformer *transformer,
    Tokenizer *tokenizer,
    int *prompt_tokens,
    int num_prompt_tokens,
    int steps,
    int *token_io,
    int *pos_io,
    long *start_io
) {
    /*
     * 这条才是当前主线使用的 prefill 路径。
     *
     * 它会把 prompt 按 chunk 切成最多 4 列，
     * 然后交给 process_prefill_chunk() 做真正的 batched 执行。
     */
    g_phase = ACCEL_PHASE_PREFILL;
    while (*pos_io < steps && *pos_io < num_prompt_tokens - 1) {
        int remaining_prompt_edges = (num_prompt_tokens - 1) - *pos_io;
        int cols = min_i(remaining_prompt_edges, g_rt.prefill_chunk_hint);
        cols = min_i(cols, ACCEL_MAX_PREFILL_BATCH);
        if (cols <= 0) {
            cols = 1;
        }
        g_rt.active_n_hint = cols;

        process_prefill_chunk(transformer, prompt_tokens, *pos_io, cols);

        for (int c = 0; c < cols; c++) {
            int token = prompt_tokens[*pos_io + c];
            int next = prompt_tokens[*pos_io + c + 1];
            char* piece;

            if (next == 1) {
                *token_io = next;
                *pos_io += c + 1;
                return 0;
            }

            piece = decode(tokenizer, token, next);
            safe_printf(piece);
            fflush(stdout);

            if (*start_io == 0) {
                *start_io = time_in_ms();
            }
        }

        *pos_io += cols;
        *token_io = prompt_tokens[*pos_io];
    }
    return 1;
}

static int run_decode_phase(
    Transformer *transformer,
    Tokenizer *tokenizer,
    Sampler *sampler,
    int steps,
    int *token_io,
    int *pos_io,
    long *start_io
) {
    g_phase = ACCEL_PHASE_DECODE;
    g_rt.active_n_hint = 1;
    while (*pos_io < steps) {
        int token = *token_io;
        int pos = *pos_io;
        float* logits = forward(transformer, token, pos);
        int next = sample(sampler, logits);

        (*pos_io)++;
        if (next == 1) {
            *token_io = next;
            return 0;
        }

        {
            char* piece = decode(tokenizer, token, next);
            safe_printf(piece);
            fflush(stdout);
        }

        *token_io = next;
        if (*start_io == 0) {
            *start_io = time_in_ms();
        }
    }
    return 1;
}

/*
 * generate():
 * 当前探索版顶层生成流程。
 *
 * 和原始 runq.c 最大的结构区别：
 * - 显式拆成 prefill / decode 两个阶段
 * - prefill 主线使用 run_prefill_phase_batched()
 *
 * 其中：
 * - prefill 会尽量按 chunk<=4 组织多 token 线性层执行
 * - decode 仍保持单 token 自回归路径
 *
 * 注意：
 * - 这仍然是软件 fallback + mock runtime
 * - 不是实际 MMIO / RTL 执行
 */
void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    {
        int num_prompt_tokens = 0;
        int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int));
        long start = 0;
        int token;
        int pos = 0;

        encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
        if (num_prompt_tokens < 1) {
            fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
            exit(EXIT_FAILURE);
        }

        token = prompt_tokens[0];
        if (!run_prefill_phase_batched(transformer, tokenizer, prompt_tokens, num_prompt_tokens, steps, &token, &pos, &start)) {
            printf("\n");
            if (pos > 1) {
                long end = time_in_ms();
                fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
            }
            free(prompt_tokens);
            return;
        }

        (void)run_decode_phase(transformer, tokenizer, sampler, steps, &token, &pos, &start);
        printf("\n");
        if (pos > 1) {
            long end = time_in_ms();
            fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
        }
        free(prompt_tokens);
    }
}

/*
 * explore_error_usage():
 * 打印这个探索工具的 CLI 用法。
 *
 * 注意这里暴露的参数已经不只是原始 runq 参数，
 * 还包括：
 * - backend 选择
 * - prefill chunk hint
 * - 各类输出格式合同
 * - trace 路径
 */
static void explore_error_usage(void) {
    fprintf(stderr, "Usage: runq_coarse_runtime_explore <checkpoint> [options]\n");
    fprintf(stderr, "Example: runq_coarse_runtime_explore model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -r <string> backend: software|mock_mmio|compare, default software\n");
    fprintf(stderr, "  -c <int>    prefill chunk hint carried in descriptors, default 4\n");
    fprintf(stderr, "  -u <string> optional trace file path for descriptor/MMIO dumps\n");
    fprintf(stderr, "  -L <string> linear out fmt: fp32|acc32|mid16|int8, default fp32\n");
    fprintf(stderr, "  -N <string> norm out fmt: fp32|acc32|mid16|int8, default fp32\n");
    fprintf(stderr, "  -A <string> attn out fmt: fp32|acc32|mid16|int8, default fp32\n");
    exit(EXIT_FAILURE);
}

/*
 * parse_backend:
 * 命令行参数选择当前运行模式：
 * - software
 * - mock_mmio
 * - compare
 */
static AccelBackend parse_backend(const char* s) {
    if (strcmp(s, "software") == 0) return ACCEL_BACKEND_SOFTWARE;
    if (strcmp(s, "mock_mmio") == 0) return ACCEL_BACKEND_MOCK_MMIO;
    if (strcmp(s, "compare") == 0) return ACCEL_BACKEND_COMPARE;
    fprintf(stderr, "unknown backend: %s\n", s);
    exit(EXIT_FAILURE);
}

/*
 * main():
 * 这个探索工具现在的定位是：
 * - 不改 runq.c 基线
 * - 在独立文件里形成 coarse-op 软件合同
 * - 能输出 descriptor trace
 * - 能输出 tile/bank trace
 * - 能做 mock 路径与软件路径的一致性校验
 *
 * 额外职责：
 * - 解析输出格式合同参数
 * - 初始化 mock ping-pong bank 上下文
 */
int main(int argc, char *argv[]) {
    char *checkpoint_path = NULL;
    char *tokenizer_path = "tokenizer.bin";
    char *backend_name = "software";
    char *trace_path = NULL;
    char *linear_fmt_name = "fp32";
    char *norm_fmt_name = "fp32";
    char *attn_fmt_name = "fp32";
    float temperature = 1.0f;
    float topp = 0.9f;
    int steps = 256;
    char *prompt = NULL;
    unsigned long long rng_seed = 0;

    if (argc < 2) {
        explore_error_usage();
    }
    checkpoint_path = argv[1];

    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc) { explore_error_usage(); }
        if (argv[i][0] != '-') { explore_error_usage(); }
        if (strlen(argv[i]) != 2) { explore_error_usage(); }
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'r') { backend_name = argv[i + 1]; }
        else if (argv[i][1] == 'c') { g_rt.prefill_chunk_hint = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'u') { trace_path = argv[i + 1]; }
        else if (argv[i][1] == 'L') { linear_fmt_name = argv[i + 1]; }
        else if (argv[i][1] == 'N') { norm_fmt_name = argv[i + 1]; }
        else if (argv[i][1] == 'A') { attn_fmt_name = argv[i + 1]; }
        else { explore_error_usage(); }
    }

    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0f) temperature = 0.0f;
    if (topp < 0.0f || 1.0f < topp) topp = 0.9f;
    if (steps < 0) steps = 0;
    if (g_rt.prefill_chunk_hint <= 0) g_rt.prefill_chunk_hint = 4;
    g_rt.backend = parse_backend(backend_name);
    g_rt.linear_out_fmt = parse_fmt(linear_fmt_name);
    g_rt.norm_out_fmt = parse_fmt(norm_fmt_name);
    g_rt.attn_out_fmt = parse_fmt(attn_fmt_name);
    g_rt.submit_count = 0;
    g_rt.tile_submit_count = 0;
    g_rt.next_input_use_bank = 0;
    g_rt.next_input_prefetch_bank = 1;
    g_rt.next_output_bank = 0;
    g_rt.active_n_hint = 1;
    if (trace_path != NULL) {
        g_rt.trace_fp = fopen(trace_path, "w");
        if (g_rt.trace_fp == NULL) {
            fprintf(stderr, "failed to open trace file: %s\n", trace_path);
            exit(EXIT_FAILURE);
        }
    }

    {
        Transformer transformer;
        build_transformer(&transformer, checkpoint_path);
        if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len;

        Tokenizer tokenizer;
        build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

        Sampler sampler;
        build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

        generate(&transformer, &tokenizer, &sampler, prompt, steps);

        free_sampler(&sampler);
        free_tokenizer(&tokenizer);
        free_transformer(&transformer);
    }
    if (g_rt.trace_fp != NULL) {
        fclose(g_rt.trace_fp);
        g_rt.trace_fp = NULL;
    }
    return 0;
}
