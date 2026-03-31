#ifndef LLAMA2C_DEPLOY_RUNTIME_MEMORY_MAP_H
#define LLAMA2C_DEPLOY_RUNTIME_MEMORY_MAP_H

// 统一维护当前部署版的地址口径，避免物理地址和 CPU 的 uncached alias 散落在代码里。
// 约定：
// - *_PHYS  ：给 DMA / MMIO / 硬件编程使用
// - *_UNC   ：给 CPU 直接访问使用

#define RUNTIME_UART_BASE_PHYS         0x1f000000u
#define RUNTIME_UART_BASE_UNC          0xbf000000u

#define RUNTIME_ACCEL_MMIO_BASE_PHYS   0x1f100000u
#define RUNTIME_ACCEL_MMIO_BASE_UNC    0xbf100000u

#define RUNTIME_CONFREG_BASE_PHYS      0x1f200000u
#define RUNTIME_CONFREG_BASE_UNC       0xbf200000u

#define RUNTIME_RAM_BASE_PHYS          0x1c000000u
#define RUNTIME_RAM_BASE_UNC           0xbc000000u

// deploy 规划中的共享 RAM 区：
// 物理地址：0x1c280000 ~ 0x1c37ffff
// CPU 访问别名：0xbc280000 ~ 0xbc37ffff
#define RUNTIME_ACCEL_SHARED_BASE_PHYS 0x1c280000u
#define RUNTIME_ACCEL_SHARED_BASE_UNC  0xbc280000u

#define RUNTIME_ACCEL_IO_IN_PHYS       0x1c280000u
#define RUNTIME_ACCEL_IO_IN_UNC        0xbc280000u
#define RUNTIME_ACCEL_IO_IN_SIZE       0x00020000u

#define RUNTIME_ACCEL_IO_OUT_PHYS      0x1c2a0000u
#define RUNTIME_ACCEL_IO_OUT_UNC       0xbc2a0000u
#define RUNTIME_ACCEL_IO_OUT_SIZE      0x00020000u

#define RUNTIME_ACCEL_PARAM_PHYS       0x1c2c0000u
#define RUNTIME_ACCEL_PARAM_UNC        0xbc2c0000u
#define RUNTIME_ACCEL_PARAM_SIZE       0x00040000u

// 当前阶段围绕 260K 模型固定部署，因此运行时 arena 也先按 260K 规模冻结。
#define RUNTIME_ARENA_SIZE             0x00100000u

#define RUNTIME_MODEL_MAX_DIM          64
#define RUNTIME_MODEL_MAX_HIDDEN       172
#define RUNTIME_MODEL_MAX_LAYERS       5
#define RUNTIME_MODEL_MAX_HEADS        8
#define RUNTIME_MODEL_MAX_KV_HEADS     4
#define RUNTIME_MODEL_MAX_SEQ_LEN      512
#define RUNTIME_MODEL_MAX_VOCAB        512

#endif
