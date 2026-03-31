#include "runtime_hw_adapter.h"
#include "runtime_memory_map.h"

#include <stdio.h>

// 这层的职责不是“实现硬件算子”，而是把语义级 backend 调用翻译成
// SoC 实际能接受的 MMIO/DMA 作业提交流程。第一版只把接口和日志立住。

typedef struct {
    uint32_t mmio_base_phys;
    uintptr_t mmio_base_uncached;
    SharedBufferDesc shared_in;
    SharedBufferDesc shared_out;
    SharedBufferDesc shared_param;
} RuntimeHwAdapter;

static RuntimeHwAdapter g_hw_adapter = {
    RUNTIME_ACCEL_MMIO_BASE_PHYS,
    RUNTIME_ACCEL_MMIO_BASE_UNC,
    {RUNTIME_ACCEL_IO_IN_UNC,  RUNTIME_ACCEL_IO_IN_PHYS,  RUNTIME_ACCEL_IO_IN_SIZE},
    {RUNTIME_ACCEL_IO_OUT_UNC, RUNTIME_ACCEL_IO_OUT_PHYS, RUNTIME_ACCEL_IO_OUT_SIZE},
    {RUNTIME_ACCEL_PARAM_UNC,  RUNTIME_ACCEL_PARAM_PHYS,  RUNTIME_ACCEL_PARAM_SIZE},
};

void runtime_hw_adapter_init(void) {
    // 第一版没有真实硬件状态需要初始化，这里主要用于后续扩展。
}

void runtime_hw_adapter_dump_layout(FILE *fp) {
    fprintf(fp, "[HW_ADAPTER] MMIO phys=0x%08x uncached=0x%08llx\n",
        g_hw_adapter.mmio_base_phys,
        (unsigned long long)g_hw_adapter.mmio_base_uncached);
    fprintf(fp, "[HW_ADAPTER] IN  cpu=0x%08llx phys=0x%08x size=0x%zx\n",
        (unsigned long long)g_hw_adapter.shared_in.cpu_uncached_addr,
        g_hw_adapter.shared_in.phys_addr,
        g_hw_adapter.shared_in.size);
    fprintf(fp, "[HW_ADAPTER] OUT cpu=0x%08llx phys=0x%08x size=0x%zx\n",
        (unsigned long long)g_hw_adapter.shared_out.cpu_uncached_addr,
        g_hw_adapter.shared_out.phys_addr,
        g_hw_adapter.shared_out.size);
    fprintf(fp, "[HW_ADAPTER] PAR cpu=0x%08llx phys=0x%08x size=0x%zx\n",
        (unsigned long long)g_hw_adapter.shared_param.cpu_uncached_addr,
        g_hw_adapter.shared_param.phys_addr,
        g_hw_adapter.shared_param.size);
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

const SharedBufferDesc *runtime_hw_adapter_shared_in(void) {
    return &g_hw_adapter.shared_in;
}

const SharedBufferDesc *runtime_hw_adapter_shared_out(void) {
    return &g_hw_adapter.shared_out;
}

const SharedBufferDesc *runtime_hw_adapter_shared_param(void) {
    return &g_hw_adapter.shared_param;
}
