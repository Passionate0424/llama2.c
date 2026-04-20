#ifndef LLAMA2C_DEPLOY_RUNTIME_HW_ADAPTER_H
#define LLAMA2C_DEPLOY_RUNTIME_HW_ADAPTER_H

#include "runtime_types.h"

// 这层负责把“语义级后端调用”翻译成当前 MMIO 优先的软件/硬件边界外壳。
// 第一版先保留地址口径、共享缓冲边界和 trace，不直接做真实 MMIO/DMA。
// 设计目标：
// 1) 让上层 backend 不直接依赖 MMIO 寄存器细节；
// 2) 固化 shared buffer 的三套口径（cpu_ptr / uncached / phys）；
// 3) 在 HW_STUB 阶段保留最小观测与 DMA stub，并继续收敛到 MMIO 优先主线。

typedef struct {
    // 语义标签，便于 trace/调试和后续 profile。
    const char *op_name;
    int layer_idx;
    size_t elem_count;
    // 线性算子通常依赖输入/输出/参数三个共享区。
    const SharedBufferDesc *in;
    const SharedBufferDesc *out;
    const SharedBufferDesc *param;
} RuntimeHwLinearJob;

typedef struct {
    const char *op_name;
    int layer_idx;
    size_t elem_count;
    // 后处理算子多数只需要 in/out，但这里保留 scratch 口便于扩展。
    const SharedBufferDesc *in;
    const SharedBufferDesc *out;
    const SharedBufferDesc *scratch;
} RuntimeHwPostJob;

void runtime_hw_adapter_init(void);
void runtime_hw_adapter_dump_layout(FILE *fp);
void runtime_hw_adapter_trace_op(const char *op_name, int layer_idx, int elem_count);

int runtime_hw_prepare_post_job(RuntimeHwPostJob *job, const char *op_name, int layer_idx, size_t elem_count);
int runtime_hw_dma_load(const SharedBufferDesc *dst, size_t dst_offset, const void *src, size_t bytes);
int runtime_hw_dma_store(void *dst, const SharedBufferDesc *src, size_t src_offset, size_t bytes);
void runtime_hw_soft_reset(void);

const SharedBufferDesc *runtime_hw_adapter_shared_in(void);
const SharedBufferDesc *runtime_hw_adapter_shared_out(void);
const SharedBufferDesc *runtime_hw_adapter_shared_param(void);
const SharedBufferDesc *runtime_hw_adapter_shared_kv(void);
const SharedBufferDesc *runtime_hw_adapter_key_cache_main_data(void);
const SharedBufferDesc *runtime_hw_adapter_value_cache_main_data(void);
const SharedBufferDesc *runtime_hw_adapter_shared_scratch(void);
const SharedBufferDesc *runtime_hw_adapter_shared_trace(void);

#endif
