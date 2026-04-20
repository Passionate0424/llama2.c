# 部署运行时说明

本文档说明 `deploy/` 目录下独立部署运行时的当前实现状态、使用方法和边界。这里描述的是“已经落地的实现口径”，不是规划口径。

## 目标

当前运行时的目标是提供一套独立于 `runq.c` 的部署/验证框架，满足以下几点：

- 可从 `.h` 头文件资产直接加载模型和 tokenizer
- 可生成 `CPU/HW` 两个部署产物
- `CPU/HW` 两条部署路径共享同一条算子级 decode 主链路
- 可通过 `runq_verify` 固定执行 `SW_REF -> HW_STUB -> compare`
- 为后续真实硬件后端预留共享 RAM、双地址口径和 adapter 分层

当前保留两条主线：

1. `runq_deploy`
   - 面向实际部署
   - 通过编译宏生成：
     - `runq_deploy_cpu`
     - `runq_deploy_hw`
2. `runq_verify`
   - 面向硬件验证
   - 固定执行 `SW_REF -> HW_STUB -> compare`

## 当前架构口径

### 1. backend API 已经收敛为算子级接口

部署主路径不再直接调用整图 `forward`。当前 `generate/chat` 都通过公共 decode 调度器，按算子级 backend API 推进单 token 推理。当前实际使用的算子接口包括：

- `rmsnorm`
- `linear_qkv`
- `qk_matmul`
- `softmax_row`
- `av_matmul`
- `linear_attn_o`
- `linear_ffn_w1`
- `linear_ffn_w3`
- `gate_mul`
- `linear_ffn_w2`
- `residual_add`
- `final_norm`
- `lm_head`

这意味着：

- `runq_deploy_cpu`
  - 通过 `SW_REF` backend 执行算子
- `runq_deploy_hw`
  - 当前仍通过 `HW_STUB` backend 执行算子
  - 但调用边界已经与 CPU 版一致

### 2. `runtime_hw_adapter` 当前是兼容层，不是真实硬件后端

当前 `runtime_hw_adapter` 的职责是冻结这几件事：

- 共享区边界
- command window 边界
- `cpu_ptr / cpu_uncached_addr / phys_addr` 三套口径
- 线性作业 / 后处理作业的基础接口形状
- `dma_load / dma_store / submit / wait_done / soft_reset` 这类后续真实硬件会用到的 API 入口

当前实现状态是：

- `HW_STUB` 仍然用软件数学实现算子结果
- adapter 主要负责：
  - 共享区 backing storage
  - command/completion window backing storage
  - 地址布局输出
  - host 环境下的 `memcpy` 型 DMA stub
  - 作业提交接口的占位与 trace

因此，当前 `runq_deploy_hw` 是“经过硬件后端边界和 adapter 兼容层”的版本，但还不是“真实 MMIO/DMA 硬件后端”。

当前还保留了一个独立 command window 地址窗口，作为历史/兼容占位：

- `CMDQ`
- `CMPQ`
- `DBG2`

需要强调的是：

- 现有 `ACCEL_SHARED` 1MB 不因为 queue 需求而改动
- 当前主线控制面继续采用 `legacy/fixed-job MMIO`
- command/completion 窗口不再代表当前要推进的 queue 命令硬件
- 若后续重新讨论 queue 路线，应另起阶段，不自动继承当前默认路径

因此，当前 deploy 主线应理解为：**MMIO 优先，queue 暂不需要。**

### 3. 共享 RAM / section 当前实现状态

当前已经补齐：

- `.model_assets`
- `.runtime_arena`
- `.accel_shared`
- `.accel_trace`
- `.accel_cmdq`
- `.accel_cmpq`
- `.accel_dbg2`

以及共享区/主数据面子区域：

- `KV_MAIN_WINDOW`
- `KEY_CACHE_MAIN_REGION`
- `VALUE_CACHE_MAIN_REGION`

- `ACCEL_IO_IN`
- `ACCEL_IO_OUT`
- `ACCEL_PARAM`
- `KEY_CACHE_MAIN_DATA`
- `KEY_CACHE_MAIN_SCALE`
- `VALUE_CACHE_MAIN_DATA`
- `VALUE_CACHE_MAIN_SCALE`
- `ACCEL_SCRATCH`
- `ACCEL_TRACE`

此外还保留了独立的 command window 子区域：

- `ACCEL_CMDQ`
- `ACCEL_CMPQ`
- `ACCEL_DBG2`

当前实现中的 command window 地址划分如下：

- `CMD_WINDOW`
  - 物理：`0x1c400000 ~ 0x1c40ffff`
  - CPU uncached alias：`0xbc400000 ~ 0xbc40ffff`
- `CMDQ`
  - 物理：`0x1c400000 ~ 0x1c403fff`
  - CPU uncached alias：`0xbc400000 ~ 0xbc403fff`
  - 大小：`16KB`
- `CMPQ`
  - 物理：`0x1c404000 ~ 0x1c404fff`
  - CPU uncached alias：`0xbc404000 ~ 0xbc404fff`
  - 大小：`4KB`
- `DBG2`
  - 物理：`0x1c405000 ~ 0x1c405fff`
  - CPU uncached alias：`0xbc405000 ~ 0xbc405fff`
  - 大小：`4KB`

这些窗口当前仅保留为地址规划/兼容占位，不再代表当前要推进的 queue 提交主线。
当前默认控制面不依赖 `CMDQ/CMPQ` ring、doorbell 或 completion 语义。
如果后续重新恢复 queue 路线，应单独重开规格，而不是默认沿用当前 bring-up 路径。

换言之：
- 当前主线 = `legacy/fixed-job MMIO`
- `CMDQ/CMPQ/DBG2` = 保留地址窗口，不是当前必经控制路径

需要注意：

- host 构建默认不强制加载 linker section 片段
- `deploy/ld/deploy_sections.ldh` 已经可以用于 SoC 侧链接接入
- 但真实固定物理地址落版仍需要由上层 LoongArch/OpenLA500 主链接脚本显式 `include`

## 目录说明

- `include/`
  - 统一接口、类型和地址口径
- `src/`
  - 前端、后端、adapter、验证和主入口实现
- `assets/`
  - 部署使用的头文件资产
- `tests/`
  - 后续固定 prompt / verify case 的位置
- `ld/`
  - linker section 片段

## 当前资产状态

当前正式资产可由下面脚本导出：

```powershell
python tools/export_deploy_headers.py --model-bin artifacts/stories260K/stories260K_q80.bin --tokenizer-bin artifacts/stories260K/tok512.bin
```

当前 deploy 状态文档默认对齐的 QAT best 资产为：

- `deploy/assets/qat_best_compare_finalpolishstrong_seq256/stories_data.h`
- `deploy/assets/qat_best_compare_finalpolishstrong_seq256/tok512.h`

其中：

- `deploy/assets/stories260K_qat_best/` 保留为官方导出基线口径；
- 本轮 deploy 状态、verify 与默认展示/workload 记录，不再把它写成当前默认 QAT best。

当前导出的头文件已经带有：

- `.model_assets` section 标注
- `aligned(64)` 对齐约束

## 构建方法

在仓库根目录执行：

```powershell
make runq_deploy_cpu
make runq_deploy_hw
make runq_verify
make export_deploy_assets
```

为了兼容旧脚本，下面这些旧目标名仍可继续使用：

```powershell
make runqdeploycpu
make runqdeployhw
make runqverify
make exportdeployassets
```

Windows/MSVC 下可使用：

```powershell
build_msvc.bat
```

如果要在 SoC 侧接入 linker section 片段，可显式传入：

```powershell
make runq_deploy_hw DEPLOY_LDFLAGS="-Wl,-T,deploy/ld/deploy_sections.ldh"
```

## 运行方法

### CPU 部署版

```powershell
.\runq_deploy_cpu -n 140 -i "Once upon a time"
```

### HW 部署版

```powershell
.\runq_deploy_hw -n 140 -i "Once upon a time"
```

当前 `HW` 版会：

- 经过 `HW_STUB` backend
- 输出共享区 `ptr/cpu/phys/size` 布局
- 在运行过程中输出 adapter trace

### 验证版

```powershell
.\runq_verify
```

它会固定执行：

1. `SW_REF`
2. `HW_STUB`
3. 对比关键算子输出
4. 输出逐项结果和最终 summary

当前 verify 已覆盖：

- `rmsnorm`
- `linear_qkv_q/k/v`
- `av_matmul`
- `qk_matmul`
- `softmax_row`
- `gate_mul`
- `residual_add`
- `kv_main_map`

其中 `qk_matmul` 已不再只是单元素检查，而是覆盖：

- `pos > 0`
- 多元素输出行
- 非零 `head_idx`

## 当前默认展示口径

当前内部固定的默认 decode/workload 真值来自 `deploy/include/runtime_decode_cfg.h`：

- `temperature = 0.93`
- `top_p = 0.9`
- `top_k = 40`
- `repetition_penalty = 1.05`
- `no_repeat_ngram_size = 3`
- `max_new_tokens = 140`
- prompts 固定分为 `stable` / `story` 两组

当前 deploy 状态可概括为：

- `runq_deploy_cpu = SW_REF`
- `runq_deploy_hw = HW_STUB + runtime_hw_adapter`
- deploy V1 默认执行主线 = `quantized linears + float attention core`
- `qk_matmul / softmax_row / av_matmul` 当前仍保持 float 执行语义

## 当前实现边界

当前阶段已经实现：

- 默认头文件资产加载
- `SW_REF` backend
- `HW_STUB` backend
- 算子级部署主路径
- `generate/chat` 前端
- arena 化 `RunState`
- 共享区 descriptor / adapter 兼容层
- 算子级 verify 与 summary 输出

其中 `chat` 当前边界为：

- 行为上对齐 OpenLA500 的多轮演示风格
- 首轮支持 `system prompt` 与 CLI `user prompt`
- 后续轮次继续读取 `User:`
- 增加了 token 容量检查和剩余上下文检查
- 主要用于运行时流程和串口交互演示，还不是完整 instruct/chat 产品形态

当前阶段尚未完成：

- 真实 `HW` backend
- 真实 MMIO/DMA 寄存器协议接入
- SoC 主链接脚本对 `deploy_sections.ldh` 的正式接入
- 当前最佳 QAT 权重的正式部署导出

当前默认控制面继续采用 `legacy/fixed-job MMIO`。
queue 命令硬件、queue submit/completion 软件路径不再属于这轮主线。

## 说明

`runq.c` 与 `runq_embedded.c` 当前保持不动，只作为历史基线参考。
