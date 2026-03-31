#ifndef LLAMA2C_DEPLOY_RUNTIME_HW_ADAPTER_H
#define LLAMA2C_DEPLOY_RUNTIME_HW_ADAPTER_H

#include "runtime_types.h"

// 这层负责把“语义级后端调用”翻译成“真实硬件作业提交”的外壳。
// 第一版先保留地址口径、共享缓冲边界和 trace，不直接做真实 MMIO/DMA。

void runtime_hw_adapter_init(void);
void runtime_hw_adapter_dump_layout(FILE *fp);
void runtime_hw_adapter_trace_op(const char *op_name, int layer_idx, int elem_count);
const SharedBufferDesc *runtime_hw_adapter_shared_in(void);
const SharedBufferDesc *runtime_hw_adapter_shared_out(void);
const SharedBufferDesc *runtime_hw_adapter_shared_param(void);

#endif
