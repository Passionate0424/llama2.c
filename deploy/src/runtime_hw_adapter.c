#include "runtime_hw_adapter.h"
#include "runtime_memory_map.h"

#include <stdio.h>
#include <string.h>

// 这层的职责不是“实现硬件算子”，而是把语义级 backend 调用翻译成
// SoC 实际能接受的 MMIO/DMA 作业提交流程。
// 第一版先把接口、地址口径和共享区边界冻结，真实寄存器协议后续替换进来。

#if defined(__GNUC__)
#define RUNTIME_SEC(section_name) __attribute__((section(section_name), aligned(64)))
#else
#define RUNTIME_SEC(section_name)
#endif

#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
#define RUNTIME_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#else
#define RUNTIME_STATIC_ASSERT(cond, msg)
#endif

RUNTIME_STATIC_ASSERT(sizeof(RuntimeHwCmdEntry) == RUNTIME_HW_CMD_ENTRY_SIZE, "RuntimeHwCmdEntry must stay 32B");
RUNTIME_STATIC_ASSERT(sizeof(RuntimeHwCmpEntry) == RUNTIME_HW_CMP_ENTRY_SIZE, "RuntimeHwCmpEntry must stay 16B");
RUNTIME_STATIC_ASSERT((RUNTIME_ACCEL_CMDQ_SIZE % RUNTIME_HW_CMD_ENTRY_SIZE) == 0, "CMDQ size must align to cmd entry size");
RUNTIME_STATIC_ASSERT((RUNTIME_ACCEL_CMPQ_SIZE % RUNTIME_HW_CMP_ENTRY_SIZE) == 0, "CMPQ size must align to cmp entry size");

// 共享区 backing storage：
// - 在 host/HW_STUB 上直接作为可访问内存；
// - 在 SoC 上可替换成实际 uncached alias 映射，接口层不变。
static unsigned char g_accel_io_in[RUNTIME_ACCEL_IO_IN_SIZE] RUNTIME_SEC(".accel_shared");
static unsigned char g_accel_io_out[RUNTIME_ACCEL_IO_OUT_SIZE] RUNTIME_SEC(".accel_shared");
static unsigned char g_accel_param[RUNTIME_ACCEL_PARAM_SIZE] RUNTIME_SEC(".accel_shared");
static unsigned char g_accel_kv[RUNTIME_ACCEL_KV_SHARED_SIZE] RUNTIME_SEC(".accel_shared");
static unsigned char g_accel_scratch[RUNTIME_ACCEL_SCRATCH_SIZE] RUNTIME_SEC(".accel_shared");
static unsigned char g_accel_trace[RUNTIME_ACCEL_TRACE_SIZE] RUNTIME_SEC(".accel_trace");

// command window backing storage：
// - 这块故意不复用现有 1MB shared window；
// - 目的是把 queue/control plane 和 data plane 解耦；
// - 当前先作为 host/HW_STUB 下的静态 backing storage，后续 SoC 上可映射到独立 SRAM 窗口。
static unsigned char g_accel_cmdq[RUNTIME_ACCEL_CMDQ_SIZE] RUNTIME_SEC(".accel_cmdq");
static unsigned char g_accel_cmpq[RUNTIME_ACCEL_CMPQ_SIZE] RUNTIME_SEC(".accel_cmpq");
static unsigned char g_accel_dbg2[RUNTIME_ACCEL_DBG2_SIZE] RUNTIME_SEC(".accel_dbg2");

typedef struct {
    uint32_t mmio_base_phys;
    uintptr_t mmio_base_uncached;
    SharedBufferDesc shared_in;
    SharedBufferDesc shared_out;
    SharedBufferDesc shared_param;
    SharedBufferDesc shared_kv;
    SharedBufferDesc shared_scratch;
    SharedBufferDesc shared_trace;
    SharedBufferDesc shared_cmdq;
    SharedBufferDesc shared_cmpq;
    SharedBufferDesc shared_dbg2;
} RuntimeHwAdapter;

static RuntimeHwAdapter g_hw_adapter = {
    RUNTIME_ACCEL_MMIO_BASE_PHYS,
    RUNTIME_ACCEL_MMIO_BASE_UNC,
    {g_accel_io_in, RUNTIME_ACCEL_IO_IN_UNC, RUNTIME_ACCEL_IO_IN_PHYS, RUNTIME_ACCEL_IO_IN_SIZE},
    {g_accel_io_out, RUNTIME_ACCEL_IO_OUT_UNC, RUNTIME_ACCEL_IO_OUT_PHYS, RUNTIME_ACCEL_IO_OUT_SIZE},
    {g_accel_param, RUNTIME_ACCEL_PARAM_UNC, RUNTIME_ACCEL_PARAM_PHYS, RUNTIME_ACCEL_PARAM_SIZE},
    {g_accel_kv, RUNTIME_ACCEL_KV_SHARED_UNC, RUNTIME_ACCEL_KV_SHARED_PHYS, RUNTIME_ACCEL_KV_SHARED_SIZE},
    {g_accel_scratch, RUNTIME_ACCEL_SCRATCH_UNC, RUNTIME_ACCEL_SCRATCH_PHYS, RUNTIME_ACCEL_SCRATCH_SIZE},
    {g_accel_trace, RUNTIME_ACCEL_TRACE_UNC, RUNTIME_ACCEL_TRACE_PHYS, RUNTIME_ACCEL_TRACE_SIZE},
    {g_accel_cmdq, RUNTIME_ACCEL_CMDQ_UNC, RUNTIME_ACCEL_CMDQ_PHYS, RUNTIME_ACCEL_CMDQ_SIZE},
    {g_accel_cmpq, RUNTIME_ACCEL_CMPQ_UNC, RUNTIME_ACCEL_CMPQ_PHYS, RUNTIME_ACCEL_CMPQ_SIZE},
    {g_accel_dbg2, RUNTIME_ACCEL_DBG2_UNC, RUNTIME_ACCEL_DBG2_PHYS, RUNTIME_ACCEL_DBG2_SIZE},
};

static int check_dma_args(const SharedBufferDesc *buf, size_t offset, size_t bytes, const char *tag) {
    if (!buf || !buf->cpu_ptr) {
        fprintf(stderr, "%s: 共享缓冲为空，无法执行 DMA 操作\n", tag);
        return -1;
    }
    if (offset > buf->size || bytes > (buf->size - offset)) {
        fprintf(stderr, "%s: 越界 offset=0x%zx bytes=0x%zx size=0x%zx\n", tag, offset, bytes, buf->size);
        return -1;
    }
    return 0;
}

static uint8_t clamp_qos_class(uint8_t qos_class) {
    // 当前 RTL dma_rdwr 只定义了 2-bit qos class。
    // 软件侧先在 builder 里收口，避免后面 queue mode 联调时出现隐式截断。
    return (uint8_t)(qos_class & 0x3u);
}

void runtime_hw_adapter_init(void) {
    // 第一版不做真实硬件初始化，仅清零共享区，保证每次启动状态可复现。
    memset(g_accel_io_in, 0, sizeof(g_accel_io_in));
    memset(g_accel_io_out, 0, sizeof(g_accel_io_out));
    memset(g_accel_param, 0, sizeof(g_accel_param));
    memset(g_accel_kv, 0, sizeof(g_accel_kv));
    memset(g_accel_scratch, 0, sizeof(g_accel_scratch));
    memset(g_accel_trace, 0, sizeof(g_accel_trace));
    memset(g_accel_cmdq, 0, sizeof(g_accel_cmdq));
    memset(g_accel_cmpq, 0, sizeof(g_accel_cmpq));
    memset(g_accel_dbg2, 0, sizeof(g_accel_dbg2));
}

void runtime_hw_adapter_dump_layout(FILE *fp) {
    fprintf(fp, "[HW_ADAPTER] MMIO phys=0x%08x uncached=0x%08llx\n",
        g_hw_adapter.mmio_base_phys,
        (unsigned long long)g_hw_adapter.mmio_base_uncached);
    fprintf(fp, "[HW_ADAPTER] IN      ptr=%p cpu=0x%08llx phys=0x%08x size=0x%zx\n",
        g_hw_adapter.shared_in.cpu_ptr,
        (unsigned long long)g_hw_adapter.shared_in.cpu_uncached_addr,
        g_hw_adapter.shared_in.phys_addr,
        g_hw_adapter.shared_in.size);
    fprintf(fp, "[HW_ADAPTER] OUT     ptr=%p cpu=0x%08llx phys=0x%08x size=0x%zx\n",
        g_hw_adapter.shared_out.cpu_ptr,
        (unsigned long long)g_hw_adapter.shared_out.cpu_uncached_addr,
        g_hw_adapter.shared_out.phys_addr,
        g_hw_adapter.shared_out.size);
    fprintf(fp, "[HW_ADAPTER] PARAM   ptr=%p cpu=0x%08llx phys=0x%08x size=0x%zx\n",
        g_hw_adapter.shared_param.cpu_ptr,
        (unsigned long long)g_hw_adapter.shared_param.cpu_uncached_addr,
        g_hw_adapter.shared_param.phys_addr,
        g_hw_adapter.shared_param.size);
    fprintf(fp, "[HW_ADAPTER] KV      ptr=%p cpu=0x%08llx phys=0x%08x size=0x%zx\n",
        g_hw_adapter.shared_kv.cpu_ptr,
        (unsigned long long)g_hw_adapter.shared_kv.cpu_uncached_addr,
        g_hw_adapter.shared_kv.phys_addr,
        g_hw_adapter.shared_kv.size);
    fprintf(fp, "[HW_ADAPTER] SCRATCH ptr=%p cpu=0x%08llx phys=0x%08x size=0x%zx\n",
        g_hw_adapter.shared_scratch.cpu_ptr,
        (unsigned long long)g_hw_adapter.shared_scratch.cpu_uncached_addr,
        g_hw_adapter.shared_scratch.phys_addr,
        g_hw_adapter.shared_scratch.size);
    fprintf(fp, "[HW_ADAPTER] TRACE   ptr=%p cpu=0x%08llx phys=0x%08x size=0x%zx\n",
        g_hw_adapter.shared_trace.cpu_ptr,
        (unsigned long long)g_hw_adapter.shared_trace.cpu_uncached_addr,
        g_hw_adapter.shared_trace.phys_addr,
        g_hw_adapter.shared_trace.size);
    fprintf(fp, "[HW_ADAPTER] CMDQ    ptr=%p cpu=0x%08llx phys=0x%08x size=0x%zx\n",
        g_hw_adapter.shared_cmdq.cpu_ptr,
        (unsigned long long)g_hw_adapter.shared_cmdq.cpu_uncached_addr,
        g_hw_adapter.shared_cmdq.phys_addr,
        g_hw_adapter.shared_cmdq.size);
    fprintf(fp, "[HW_ADAPTER] CMPQ    ptr=%p cpu=0x%08llx phys=0x%08x size=0x%zx\n",
        g_hw_adapter.shared_cmpq.cpu_ptr,
        (unsigned long long)g_hw_adapter.shared_cmpq.cpu_uncached_addr,
        g_hw_adapter.shared_cmpq.phys_addr,
        g_hw_adapter.shared_cmpq.size);
    fprintf(fp, "[HW_ADAPTER] DBG2    ptr=%p cpu=0x%08llx phys=0x%08x size=0x%zx\n",
        g_hw_adapter.shared_dbg2.cpu_ptr,
        (unsigned long long)g_hw_adapter.shared_dbg2.cpu_uncached_addr,
        g_hw_adapter.shared_dbg2.phys_addr,
        g_hw_adapter.shared_dbg2.size);
    fprintf(fp, "[HW_ADAPTER] CMDQ entry_size=0x%zx depth=%zu | CMPQ entry_size=0x%zx depth=%zu\n",
        RUNTIME_HW_CMD_ENTRY_SIZE,
        runtime_hw_cmdq_capacity(),
        RUNTIME_HW_CMP_ENTRY_SIZE,
        runtime_hw_cmpq_capacity());
}

void runtime_hw_adapter_trace_op(const char *op_name, int layer_idx, int elem_count) {
    fprintf(stdout, "[HW_ADAPTER] op=%s layer=%d elems=%d in_cpu=0x%08llx out_cpu=0x%08llx in_phys=0x%08x out_phys=0x%08x\n",
        op_name,
        layer_idx,
        elem_count,
        (unsigned long long)g_hw_adapter.shared_in.cpu_uncached_addr,
        (unsigned long long)g_hw_adapter.shared_out.cpu_uncached_addr,
        g_hw_adapter.shared_in.phys_addr,
        g_hw_adapter.shared_out.phys_addr);
}

int runtime_hw_prepare_linear_job(RuntimeHwLinearJob *job, const char *op_name, int layer_idx, size_t elem_count) {
    if (!job || !op_name) return -1;
    // 线性算子统一绑定 in/out/param 三块区域。
    // 后续接入真实硬件时，这里可以继续扩展 bank、stride、quant 等字段。
    job->op_name = op_name;
    job->layer_idx = layer_idx;
    job->elem_count = elem_count;
    job->in = &g_hw_adapter.shared_in;
    job->out = &g_hw_adapter.shared_out;
    job->param = &g_hw_adapter.shared_param;
    return 0;
}

int runtime_hw_prepare_post_job(RuntimeHwPostJob *job, const char *op_name, int layer_idx, size_t elem_count) {
    if (!job || !op_name) return -1;
    // 后处理算子统一绑定 in/out/scratch，便于 softmax/gate/residual 一类算子复用。
    job->op_name = op_name;
    job->layer_idx = layer_idx;
    job->elem_count = elem_count;
    job->in = &g_hw_adapter.shared_in;
    job->out = &g_hw_adapter.shared_out;
    job->scratch = &g_hw_adapter.shared_scratch;
    return 0;
}

int runtime_hw_dma_load(const SharedBufferDesc *dst, size_t dst_offset, const void *src, size_t bytes) {
    if (!src) return -1;
    if (check_dma_args(dst, dst_offset, bytes, "runtime_hw_dma_load") != 0) return -1;
    // host/HW_STUB 下用 memcpy 模拟 DMA，保持边界检查和接口语义一致。
    memcpy((unsigned char *)dst->cpu_ptr + dst_offset, src, bytes);
    return 0;
}

int runtime_hw_dma_store(void *dst, const SharedBufferDesc *src, size_t src_offset, size_t bytes) {
    if (!dst) return -1;
    if (check_dma_args(src, src_offset, bytes, "runtime_hw_dma_store") != 0) return -1;
    memcpy(dst, (const unsigned char *)src->cpu_ptr + src_offset, bytes);
    return 0;
}

int runtime_hw_submit_job(const char *job_kind, const void *job) {
    // 这里保留“提交”语义边界。真实硬件版本会在此处：
    // 1) 写 MMIO 描述符；
    //    或者把 descriptor enqueue 到独立的 CMDQ 窗口；
    // 2) 触发启动；
    // 3) 记录 job id / trace marker。
    // 当前默认仍然视为 legacy 路径，不在这里切换到 queue mode，
    // 这样可以先冻结地址/内存图，再单独做 RTL 联调。
    if (!job_kind || !job) return -1;
    fprintf(stdout, "[HW_ADAPTER] submit job_kind=%s\n", job_kind);
    return 0;
}

int runtime_hw_wait_done(uint32_t timeout_cycle) {
    // 第一版直接视为完成；保留 timeout 参数保证上层调用约定不变。
    (void)timeout_cycle;
    return 0;
}

void runtime_hw_soft_reset(void) {
    // 软复位时清空共享区，模拟硬件 reset 后的干净状态。
    runtime_hw_adapter_init();
}

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
) {
    if (!entry) return -1;

    // 这里冻结的是“最小 command 合同”，字段直接贴合当前 dma_rdwr：
    // - opcode      <-> cmd_rw_i
    // - addr        <-> cmd_addr_i
    // - len_bytes   <-> cmd_len_bytes_i
    // - qos_class   <-> cmd_qos_class_i
    // - stride_en   <-> cmd_stride_en_i
    // - line_size   <-> cmd_line_size_i
    // - line_stride <-> cmd_line_stride_i
    memset(entry, 0, sizeof(*entry));
    entry->opcode = (uint8_t)opcode;
    entry->flags = 0;
    entry->qos_class = clamp_qos_class(qos_class);
    entry->addr = addr;
    entry->len_bytes = len_bytes;
    entry->stride_en = (uint8_t)(stride_en ? 1u : 0u);
    entry->line_size = line_size;
    entry->line_stride = line_stride;
    entry->job_id = job_id;
    return 0;
}

int runtime_hw_init_cmp_entry(
    RuntimeHwCmpEntry *entry,
    uint32_t job_id,
    RuntimeHwCmpStatus status,
    uint16_t error_code,
    uint32_t cycles,
    uint32_t info0
) {
    if (!entry) return -1;
    memset(entry, 0, sizeof(*entry));
    entry->job_id = job_id;
    entry->status = (uint16_t)status;
    entry->error_code = error_code;
    entry->cycles = cycles;
    entry->info0 = info0;
    return 0;
}

size_t runtime_hw_cmdq_capacity(void) {
    return (size_t)(RUNTIME_ACCEL_CMDQ_SIZE / RUNTIME_HW_CMD_ENTRY_SIZE);
}

size_t runtime_hw_cmpq_capacity(void) {
    return (size_t)(RUNTIME_ACCEL_CMPQ_SIZE / RUNTIME_HW_CMP_ENTRY_SIZE);
}

int runtime_hw_queue_layout_is_valid(void) {
    // 这个检查主要给 bring-up / assert / 单测用：
    // 只验证 queue window 和 entry 大小是否自洽，不碰具体寄存器协议。
    if ((RUNTIME_ACCEL_CMDQ_SIZE % RUNTIME_HW_CMD_ENTRY_SIZE) != 0) return 0;
    if ((RUNTIME_ACCEL_CMPQ_SIZE % RUNTIME_HW_CMP_ENTRY_SIZE) != 0) return 0;
    if (runtime_hw_cmdq_capacity() == 0) return 0;
    if (runtime_hw_cmpq_capacity() == 0) return 0;
    return 1;
}

const SharedBufferDesc *runtime_hw_adapter_shared_in(void) {
    return &g_hw_adapter.shared_in;
}

const SharedBufferDesc *runtime_hw_adapter_shared_out(void) {
    return &g_hw_adapter.shared_out;
}

const SharedBufferDesc *runtime_hw_adapter_shared_param(void) {
    return &g_hw_adapter.shared_param;
}

const SharedBufferDesc *runtime_hw_adapter_shared_kv(void) {
    return &g_hw_adapter.shared_kv;
}

const SharedBufferDesc *runtime_hw_adapter_shared_scratch(void) {
    return &g_hw_adapter.shared_scratch;
}

const SharedBufferDesc *runtime_hw_adapter_shared_trace(void) {
    return &g_hw_adapter.shared_trace;
}

const SharedBufferDesc *runtime_hw_adapter_shared_cmdq(void) {
    return &g_hw_adapter.shared_cmdq;
}

const SharedBufferDesc *runtime_hw_adapter_shared_cmpq(void) {
    return &g_hw_adapter.shared_cmpq;
}

const SharedBufferDesc *runtime_hw_adapter_shared_dbg2(void) {
    return &g_hw_adapter.shared_dbg2;
}
