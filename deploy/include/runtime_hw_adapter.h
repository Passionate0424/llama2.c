#ifndef LLAMA2C_DEPLOY_RUNTIME_HW_ADAPTER_H
#define LLAMA2C_DEPLOY_RUNTIME_HW_ADAPTER_H

#include "runtime_types.h"

// 这层负责把“语义级后端调用”翻译成“真实硬件作业提交”的外壳。
// 第一版先保留地址口径、共享缓冲边界和 trace，不直接做真实 MMIO/DMA。
// 设计目标：
// 1) 让上层 backend 不直接依赖 MMIO 寄存器细节；
// 2) 固化 shared buffer 的三套口径（cpu_ptr / uncached / phys）；
// 3) 在 HW_STUB 阶段也能完整演练“组包 -> DMA -> 提交 -> 等待”的调用顺序。

typedef struct {
    // 语义标签，便于 trace/调试和后续 profile。
    const char *op_name;
    int layer_idx;
    size_t elem_count;
    // 线性算子通常依赖输入/输出/参数三个共享区。
    const SharedBufferDesc *in;
    const SharedBufferDesc *out;
    const SharedBufferDesc *param;
    // queue-shadow：
    // - 第一版先在软件侧把 DMA 级 entry 组包并落到 CMDQ backing storage；
    // - 默认执行语义仍走 legacy submit，不在这里切默认 queue mode。
    uint32_t cmdq_shadow_offset;
    uint32_t cmdq_shadow_count;
    uint32_t submit_job_id;
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

// queue 模式的最小合同：
// 1) 先冻结 entry 大小、关键字段和 ring 容量推导方式；
// 2) 先不冻结更重的 profile/debug 扩展；
// 3) 先保持与当前 RTL dma_rdwr 已有命令字段尽量对齐。
//
// 当前约定：
// - CMDQ entry = 32B
// - CMPQ entry = 16B
// - CMDQ / CMPQ 都按 ring queue 使用
// - 默认路径仍然保留 legacy submit，不自动切 queue mode

typedef enum {
    // 与当前 dma_rdwr 的最小语义对齐：
    // 0 = 从外部 RAM 读入本地 SRAM / staging buffer
    // 1 = 从本地 SRAM / output buffer 写回外部 RAM
    RUNTIME_HW_CMD_DMA_READ  = 0,
    RUNTIME_HW_CMD_DMA_WRITE = 1,
} RuntimeHwCmdOpcode;

typedef enum {
    RUNTIME_HW_CMP_PENDING = 0,
    RUNTIME_HW_CMP_DONE    = 1,
    RUNTIME_HW_CMP_ERROR   = 2,
} RuntimeHwCmpStatus;

typedef struct {
    // word0:
    // - opcode/flags/qos 先冻结为最小控制字段；
    // - 这样既能覆盖当前 dma_rdwr，也给后续扩展留出少量空间。
    uint8_t opcode;
    uint8_t flags;
    uint8_t qos_class;
    uint8_t reserved0;

    // word1:
    // 当前主地址口径直接对齐 dma_rdwr 的 cmd_addr_i。
    uint32_t addr;

    // word2:
    // 当前长度/stride 字段直接对齐 dma_rdwr 的输入口。
    uint16_t len_bytes;
    uint8_t stride_en;
    uint8_t reserved1;

    // word3:
    uint16_t line_size;
    uint16_t line_stride;

    // word4:
    // job_id 先由软件分配，completion 回写时用同一值对应。
    uint32_t job_id;

    // word5~7:
    // 先保留给后续扩展，例如 src/dst bank、profiling tag、doorbell class 等。
    uint32_t user0;
    uint32_t reserved2;
    uint32_t reserved3;
} RuntimeHwCmdEntry;

typedef struct {
    // 最小 completion 合同：
    // - 回报 job_id
    // - 回报状态
    // - 回报错误码
    // - 回报粗粒度 cycle 计数
    uint32_t job_id;
    uint16_t status;
    uint16_t error_code;
    uint32_t cycles;
    uint32_t info0;
} RuntimeHwCmpEntry;

#define RUNTIME_HW_CMD_ENTRY_SIZE ((size_t)32u)
#define RUNTIME_HW_CMP_ENTRY_SIZE ((size_t)16u)

void runtime_hw_adapter_init(void);
void runtime_hw_adapter_dump_layout(FILE *fp);
void runtime_hw_adapter_trace_op(const char *op_name, int layer_idx, int elem_count);

int runtime_hw_prepare_linear_job(RuntimeHwLinearJob *job, const char *op_name, int layer_idx, size_t elem_count);
int runtime_hw_prepare_post_job(RuntimeHwPostJob *job, const char *op_name, int layer_idx, size_t elem_count);
int runtime_hw_dma_load(const SharedBufferDesc *dst, size_t dst_offset, const void *src, size_t bytes);
int runtime_hw_dma_store(void *dst, const SharedBufferDesc *src, size_t src_offset, size_t bytes);
int runtime_hw_submit_job(const char *job_kind, const void *job);
int runtime_hw_wait_done(uint32_t timeout_cycle);
void runtime_hw_soft_reset(void);

int runtime_hw_init_cmd_entry(
    RuntimeHwCmdEntry *entry,
    RuntimeHwCmdOpcode opcode,
    uint32_t job_id,
    uint32_t addr,
    uint16_t len_bytes,
    uint8_t qos_class,
    uint8_t stride_en,
    uint16_t line_size,
    uint16_t line_stride
);
int runtime_hw_init_cmp_entry(
    RuntimeHwCmpEntry *entry,
    uint32_t job_id,
    RuntimeHwCmpStatus status,
    uint16_t error_code,
    uint32_t cycles,
    uint32_t info0
);
size_t runtime_hw_cmdq_capacity(void);
size_t runtime_hw_cmpq_capacity(void);
int runtime_hw_queue_layout_is_valid(void);

const SharedBufferDesc *runtime_hw_adapter_shared_in(void);
const SharedBufferDesc *runtime_hw_adapter_shared_out(void);
const SharedBufferDesc *runtime_hw_adapter_shared_param(void);
const SharedBufferDesc *runtime_hw_adapter_shared_kv(void);
const SharedBufferDesc *runtime_hw_adapter_shared_scratch(void);
const SharedBufferDesc *runtime_hw_adapter_shared_trace(void);
const SharedBufferDesc *runtime_hw_adapter_shared_cmdq(void);
const SharedBufferDesc *runtime_hw_adapter_shared_cmpq(void);
const SharedBufferDesc *runtime_hw_adapter_shared_dbg2(void);

#endif
