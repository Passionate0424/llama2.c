# 独立部署运行时实施记录

本文档记录 `deploy/` 运行时在当前实现阶段已经完成的关键收敛点。这里重点描述“实现过程里已经做了什么”，用于帮助后续继续接硬件或回顾本轮改动。

## 1. 本轮实现结论

本轮最大的变化不是再加一个新入口，而是把原先“骨架能跑”的运行时，继续收敛成更接近最终部署口径的结构：

- 部署主路径已经切到算子级 backend API
- `runq_deploy_cpu` 与 `runq_deploy_hw` 共享同一条单 token decode 调度链
- `runtime_hw_adapter` 已补成明确的硬件兼容层
- 共享 RAM 双地址口径已经体现在代码结构里
- `runq_verify` 已补齐最终 summary，并加强了 `qk_matmul` 覆盖

当前仍然没有完成的，是“真实硬件寄存器协议/MMIO/DMA 后端”，这一层还只是兼容层骨架。

## 2. 当前代码分层

### 2.1 frontend / common

当前部署主链路由公共 decode 调度器负责推进，每个 token 的推理过程都会经过：

- token embedding 按行解码
- attention 子图算子
- FFN 子图算子
- final norm
- lm head

这里最重要的收敛是：

- frontend 不再直接调用整图 `forward_logits`
- `generate/chat` 都走相同的算子级 backend 边界

这样做的目的，是避免 CPU 路径和 HW 路径在主链路上分叉。

### 2.2 backend

当前 backend 的正式语义接口已经冻结在算子级：

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

当前状态：

- `SW_REF`
  - 作为部署/验证金标准
- `HW_STUB`
  - 作为“假硬件后端”
  - 数学结果仍用软件算子实现
  - 但调用边界与共享区 trace 已经过 adapter

### 2.3 hw adapter

`runtime_hw_adapter` 当前已经明确成“硬件兼容层”，职责不是算子数学，而是冻结硬件接入边界：

- 共享区 descriptor
- `cpu_ptr / cpu_uncached_addr / phys_addr / size`
- linear/post 两类 job 描述结构
- `dma_load / dma_store / submit / wait_done / soft_reset`

当前 adapter 已经有：

- host 可运行的 backing storage
- `ACCEL_IO_IN / OUT / PARAM / KV / SCRATCH / TRACE`
- trace 输出
- `memcpy` 型 DMA stub

这意味着后续接真实硬件时，优先替换的是 adapter 内部逻辑，而不是重新改 frontend/backend 边界。

## 3. 本轮实现中的关键收敛

### 3.1 去掉主路径对整图 forward 的依赖

之前的 deploy 主路径仍然依赖一个整图 `forward_logits` 逃生口，这会让 backend API 只是“看起来像算子级”，实际运行却绕回 monolithic forward。

本轮已收敛为：

- 移除 backend 里的整图 `forward_logits`
- 用公共 decode 调度器显式串起算子
- `CPU/HW` 两个部署产物都经过这条路径

这是当前实现最重要的一步，因为它直接决定了后续真硬件替换时，不需要再拆一次主链路。

### 3.2 token embedding 改成按 token 行解码

之前运行时会把整张 embedding 表展开成 float 缓冲，这会额外占一份 heap。

本轮改成：

- 保留 q80 资产原位映射
- 推理时只按当前 token 解码一行 embedding

效果：

- 少了一份整表 float 拷贝
- 更贴近 SoC 上“静态资产 + 小工作区”的部署形态

### 3.3 共享 RAM 与 section 口径写进实现

本轮把共享 RAM 的冻结子区真正写进了实现：

- `IO_IN`
- `IO_OUT`
- `PARAM`
- `KV_SHARED`
- `SCRATCH`
- `TRACE`

并且把资产头、arena、共享区 section 都补上了代码/脚本支撑：

- `.model_assets`
- `.runtime_arena`
- `.accel_shared`
- `.accel_trace`

目前 host 默认仍不强制启用 linker 片段，这是为了保持本地构建可用；SoC 侧需要由主链接脚本继续接入。

### 3.4 verify 从“骨架”提升成可直接回归的工具

本轮 `runq_verify` 已经不只是逐项打印，而是：

- 仍固定执行 `SW_REF -> HW_STUB -> compare`
- 增强了 `qk_matmul` 的场景覆盖
- 结尾输出统一 summary

当前 verify 已通过的项目：

- `rmsnorm`
- `linear_qkv_q`
- `qk_matmul`
- `softmax_row`
- `gate_mul`
- `residual_add`

其中 `qk_matmul` 已覆盖：

- `pos > 0`
- 多元素行长
- 非零 `head_idx`

## 4. 当前已验证结果

本轮主机侧已经确认：

- `make runq_verify`
  - 通过
- `./runq_verify.exe`
  - 通过
  - 最终输出 `summary total=6 failed=0 status=PASS`
- `make runq_deploy_hw`
  - 通过
- `./runq_deploy_hw.exe -n 2 -i "Once upon a time"`
  - 通过
  - 会输出全部共享区的 `ptr/cpu/phys/size`
  - 会输出算子级 adapter trace
- CPU 部署路径也已通过临时产物验证：
  - `generate` 可正常输出文本

## 5. 目前仍未完成的部分

虽然这轮已经把接口和结构收敛了很多，但下面这些仍然没完成：

- 真实 `HW` backend
- 真实 MMIO 寄存器协议
- 真实 DMA 提交/完成等待
- SoC 主链接脚本对 `deploy_sections.ldh` 的正式接入
- 当前最佳 QAT checkpoint 的正式部署导出

换句话说，当前 `runq_deploy_hw` 是：

- 真实部署主链路
- 真实共享区/地址口径
- 真实 adapter 分层
- 但算子执行仍然是 `HW_STUB`

## 6. 对后续实现最重要的提醒

后续如果要继续接真实硬件，建议优先保持下面三点不再回退：

1. 不要把 frontend 再改回整图 `forward`
2. 不要让 HW 路径绕过 `runtime_hw_adapter`
3. 不要丢掉 `cpu_ptr / uncached / phys` 三套地址口径

这三点已经是当前实现里最值钱的“结构性收敛”。
