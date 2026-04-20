#include "runtime_hw_adapter.h"
#include "runtime_memory_map.h"

#include <stdio.h>
#include <string.h>

// 这层的职责不是“实现硬件算子”，而是把语义级 backend 调用翻译成
// SoC 实际能接受的 MMIO/DMA 作业提交流程。
// 当前阶段明确采用 MMIO 优先口径，仅保留共享区描述、trace 和最小 DMA stub。

#if defined(__GNUC__)
#define RUNTIME_SEC(section_name) __attribute__((section(section_name), aligned(64)))
#else
#define RUNTIME_SEC(section_name)
#endif

// 共享区 backing storage：
// - 在 host/HW_STUB 上直接作为可访问内存；
// - 在 SoC 上可替换成实际 uncached alias 映射，接口层不变。
static unsigned char g_accel_io_in[RUNTIME_ACCEL_IO_IN_SIZE] RUNTIME_SEC(".accel_shared");
static unsigned char g_accel_io_out[RUNTIME_ACCEL_IO_OUT_SIZE] RUNTIME_SEC(".accel_shared");
static unsigned char g_accel_param[RUNTIME_ACCEL_PARAM_SIZE] RUNTIME_SEC(".accel_shared");
static unsigned char g_key_cache_main_data[RUNTIME_KEY_CACHE_MAIN_DATA_SIZE] RUNTIME_SEC(".kv_main");
static unsigned char g_value_cache_main_data[RUNTIME_VALUE_CACHE_MAIN_DATA_SIZE] RUNTIME_SEC(".kv_main");
static unsigned char g_accel_scratch[RUNTIME_ACCEL_SCRATCH_SIZE] RUNTIME_SEC(".accel_shared");
static unsigned char g_accel_trace[RUNTIME_ACCEL_TRACE_SIZE] RUNTIME_SEC(".accel_trace");

typedef struct {
    uint32_t mmio_base_phys;
    uintptr_t mmio_base_uncached;
    SharedBufferDesc shared_in;
    SharedBufferDesc shared_out;
    SharedBufferDesc shared_param;
    SharedBufferDesc key_cache_main_data;
    SharedBufferDesc value_cache_main_data;
    SharedBufferDesc shared_scratch;
    SharedBufferDesc shared_trace;
} RuntimeHwAdapter;

static RuntimeHwAdapter g_hw_adapter = {
    RUNTIME_ACCEL_MMIO_BASE_PHYS,
    RUNTIME_ACCEL_MMIO_BASE_UNC,
    {g_accel_io_in, RUNTIME_ACCEL_IO_IN_UNC, RUNTIME_ACCEL_IO_IN_PHYS, RUNTIME_ACCEL_IO_IN_SIZE},
    {g_accel_io_out, RUNTIME_ACCEL_IO_OUT_UNC, RUNTIME_ACCEL_IO_OUT_PHYS, RUNTIME_ACCEL_IO_OUT_SIZE},
    {g_accel_param, RUNTIME_ACCEL_PARAM_UNC, RUNTIME_ACCEL_PARAM_PHYS, RUNTIME_ACCEL_PARAM_SIZE},
    {g_key_cache_main_data, RUNTIME_KEY_CACHE_MAIN_DATA_UNC, RUNTIME_KEY_CACHE_MAIN_DATA_PHYS, RUNTIME_KEY_CACHE_MAIN_DATA_SIZE},
    {g_value_cache_main_data, RUNTIME_VALUE_CACHE_MAIN_DATA_UNC, RUNTIME_VALUE_CACHE_MAIN_DATA_PHYS, RUNTIME_VALUE_CACHE_MAIN_DATA_SIZE},
    {g_accel_scratch, RUNTIME_ACCEL_SCRATCH_UNC, RUNTIME_ACCEL_SCRATCH_PHYS, RUNTIME_ACCEL_SCRATCH_SIZE},
    {g_accel_trace, RUNTIME_ACCEL_TRACE_UNC, RUNTIME_ACCEL_TRACE_PHYS, RUNTIME_ACCEL_TRACE_SIZE},
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

void runtime_hw_adapter_init(void) {
    // 第一版不做真实硬件初始化，仅清零共享区，保证每次启动状态可复现。
    memset(g_accel_io_in, 0, sizeof(g_accel_io_in));
    memset(g_accel_io_out, 0, sizeof(g_accel_io_out));
    memset(g_accel_param, 0, sizeof(g_accel_param));
    memset(g_key_cache_main_data, 0, sizeof(g_key_cache_main_data));
    memset(g_value_cache_main_data, 0, sizeof(g_value_cache_main_data));
    memset(g_accel_scratch, 0, sizeof(g_accel_scratch));
    memset(g_accel_trace, 0, sizeof(g_accel_trace));
}

void runtime_hw_adapter_dump_layout(FILE *fp) {
    // 布局打印既服务 bring-up，也服务文档化核对：
    // 通过一次输出同时确认 ptr / uncached / phys / size 四套信息是否一致。
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
    fprintf(fp, "[HW_ADAPTER] KEYDAT  ptr=%p cpu=0x%08llx phys=0x%08x size=0x%zx\n",
        g_hw_adapter.key_cache_main_data.cpu_ptr,
        (unsigned long long)g_hw_adapter.key_cache_main_data.cpu_uncached_addr,
        g_hw_adapter.key_cache_main_data.phys_addr,
        g_hw_adapter.key_cache_main_data.size);
    fprintf(fp, "[HW_ADAPTER] VALDAT  ptr=%p cpu=0x%08llx phys=0x%08x size=0x%zx\n",
        g_hw_adapter.value_cache_main_data.cpu_ptr,
        (unsigned long long)g_hw_adapter.value_cache_main_data.cpu_uncached_addr,
        g_hw_adapter.value_cache_main_data.phys_addr,
        g_hw_adapter.value_cache_main_data.size);
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

void runtime_hw_soft_reset(void) {
    // 软复位时清空共享区，模拟硬件 reset 后的干净状态。
    runtime_hw_adapter_init();
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
    return &g_hw_adapter.key_cache_main_data;
}

const SharedBufferDesc *runtime_hw_adapter_key_cache_main_data(void) {
    return &g_hw_adapter.key_cache_main_data;
}

const SharedBufferDesc *runtime_hw_adapter_value_cache_main_data(void) {
    return &g_hw_adapter.value_cache_main_data;
}

const SharedBufferDesc *runtime_hw_adapter_shared_scratch(void) {
    return &g_hw_adapter.shared_scratch;
}

const SharedBufferDesc *runtime_hw_adapter_shared_trace(void) {
    return &g_hw_adapter.shared_trace;
}
