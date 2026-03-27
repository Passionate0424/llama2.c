# 粗粒度 Op 软件契约 V1

## 1. 目的

本文定义 `runq` 向“单 IP + 内部多引擎 + 粗粒度 op”硬件过渡时的软件契约。

目标不是立刻接真实 MMIO，而是先冻结软件侧边界，使后续：

- Python 先验实验
- C runtime 重构
- RTL descriptor / MMIO 设计

能围绕同一套合同推进。

对应探索实现：

- `E:\project\ciciec\9th\ref\llama2.c\tools\runq_coarse_runtime_explore.c`

---

## 2. 设计原则

1. 不修改 `runq.c` 基线语义
2. 先把 `matmul()` 升级成 `submit_linear_op(desc)` 边界
3. 先明确 `prefill / decode` 两种软件阶段
4. descriptor 中显式携带：
   - op 类型
   - phase
   - 输入/输出格式
   - scale 粒度
   - 形状信息
5. V1 先冻结 `LINEAR` 类合同，并形成 `RMSNorm / QK_SCORE / SOFTMAX_AV` 的软件合同原型

---

## 3. 粗粒度 op 分类

当前探索版合同已经覆盖 3 类粗粒度 op：

### `LINEAR` 族

- `LINEAR_ATTN_Q`
- `LINEAR_ATTN_K`
- `LINEAR_ATTN_V`
- `LINEAR_ATTN_O`
- `LINEAR_FFN_W1`
- `LINEAR_FFN_W2`
- `LINEAR_FFN_W3`
- `LINEAR_CLS`

这些 op 都由底层 `GEMM`/`matmul` 内核执行，但在软件调度层被视为不同 coarse op。

原因：

- 它们属于不同子路径
- 后续输出格式可能不同
- 将来是否接 `rmsnorm/softmax` 也不同

### `RMSNorm` 族

- `RMSNORM_ATTN`
- `RMSNORM_FFN`
- `RMSNORM_FINAL`

这些 op 当前仍使用软件浮点 `rmsnorm()` 作为参考数学，但已经纳入 descriptor / mock MMIO / compare 路径。

### `Attention NonLinear` 族

- `QK_SCORE`
- `SOFTMAX_AV`

其中：

- `QK_SCORE` 负责某个 head 在当前 `pos` 上生成一整行 attention score
- `SOFTMAX_AV` 负责对该行 score 做 softmax，并与 value cache 做加权求和

这两类 op 的合同已经形成，但它们仍然是软件 fallback 数学，不等于硬件数学已冻结。

---

## 4. 软件 phase 定义

### `PREFILL`

- 含义：仍在消费 prompt token，下一 token 已知
- 当前探索版中：
  - 仍按单 token `forward(token, pos)` 执行
  - 但 descriptor 中带 `n_dim_hint = prefill_chunk_hint`
- 目的：
  - 先把“未来硬件希望 `N=4/8`”这层意图编码进合同

### `DECODE`

- 含义：prompt 已消费完，开始 autoregressive sample
- descriptor 中固定：
  - `n_dim_hint = 1`

---

## 5. 数据格式定义

### 输入格式

- `ACCEL_FMT_INT8`
  - 当前 `LINEAR` 输入格式
  - 对应 `runq` 里的 `QuantizedTensor.q`

### 输出格式

当前探索版真实使用：

- `ACCEL_FMT_FP32`
  - `RMSNorm / QK_SCORE / SOFTMAX_AV` 当前工作格式
- `ACCEL_FMT_FP32`
  - `LINEAR` 当前实际输出存储格式

但合同里预留了：

- `ACCEL_FMT_ACC32_RAW`
- `ACCEL_FMT_MID16`

为后续硬件化 `linear_epilogue / rmsnorm / softmax_av` 准备。

---

## 6. scale 粒度定义

当前冻结为：

- 输入激活：
  - `ACCEL_SCALE_PER_TOKEN`
- 权重：
  - `ACCEL_SCALE_PER_ROW`
- 输出：
  - `ACCEL_SCALE_NONE`

这里反映的是当前主线量化契约：

- `W per-row`
- `A per-token`

---

## 7. V1 descriptor 字段

当前探索实现中的 `AccelDesc` 包含：

- `op`
- `phase`
- `layer`
- `token_pos`
- `head_idx`
- `elem_count`
- `n_dim_hint`
- `in_tensor`
- `weight_tensor`
- `out_tensor`
- `aux_tensor0`
- `aux_tensor1`
- `xq`
- `wq`
- `out`
- `k_dim`
- `m_dim`

其中最重要的是软件/硬件接口意义上的字段：

- `op`
- `phase`
- `token_pos`
- `head_idx`
- `elem_count`
- `n_dim_hint`
- `in_tensor`
- `weight_tensor`
- `out_tensor`
- `aux_tensor0`
- `aux_tensor1`
- `k_dim`
- `m_dim`

`xq / wq / out` 是当前软件 fallback 为了复用 `base_matmul()` 保留的实现细节。

---

## 8. Mock MMIO 映射

探索工具里还定义了 `MockMmioLinearRegs`，用于模拟以后真实硬件寄存器或 descriptor 队列的字段：

- `opcode`
- `layer`
- `token_pos`
- `phase`
- `head_idx`
- `elem_count`
- `m_dim`
- `k_dim`
- `n_dim`
- `in_fmt`
- `weight_fmt`
- `out_fmt`
- `aux0_fmt`
- `aux1_fmt`
- `in_scale_mode`
- `w_scale_mode`
- `in_data_addr`
- `in_meta_addr`
- `w_data_addr`
- `w_meta_addr`
- `aux0_data_addr`
- `aux0_meta_addr`
- `aux1_data_addr`
- `aux1_meta_addr`
- `out_data_addr`

这不是最终硬件寄存器定义，但已经足够作为 RTL/MMIO 设计输入。

---

## 9. backend 定义

探索工具支持 3 个 backend：

### `software`

- `LINEAR`：直接调用软件 `base_matmul()`
- `RMSNorm`：直接调用软件 `rmsnorm()`
- `QK_SCORE / SOFTMAX_AV`：直接调用软件 attention 子路径
- 用于基线行为验证

### `mock_mmio`

- 先把 descriptor 映射成 `MockMmioLinearRegs`
- 输出 trace
- 再调用对应的软件 fallback 数学

用途：

- 验证 descriptor 组织方式
- 验证 trace / MMIO 映射口径

### `compare`

- 同时跑：
  - `software`
  - `mock_mmio`
- 对每个 op 的输出逐元素比较
- 若不一致直接报错退出

用途：

- 验证 descriptor/MMIO 路径本身不改变数学

---

## 10. 已完成验证

### 已验证内容

1. `runq.c` 基线未修改
2. `runq_coarse_runtime_explore.c` 可独立编译运行
3. `software` backend 可正常推理
4. `mock_mmio` backend 可正常推理并输出 descriptor trace
5. `compare` backend 可正常推理，且对每个 coarse op 做逐元素对比
6. `mock_mmio` backend 已对 `LINEAR` 类 op 输出 tile-aware / bank-aware trace
7. `prefill` 已升级为真正的 batched 执行原型（当前默认 `N<=4`）
8. 已加入输出格式合同开关：
   - `-L` 控制 `LINEAR` 输出格式
   - `-N` 控制 `RMSNorm` 输出格式
   - `-A` 控制 `SOFTMAX_AV` 输出格式

### 已对拍结果

在 `stories260K_q80.bin` + `tok512.bin` + prompt `Once upon a time` 下：

- `runq.exe`
- `runq_embedded.exe`
- `runq_coarse_runtime_explore.exe`

输出一致：

```text
Once upon a time, there was a little girl named Lily. She loved to play outside in the phower and see all of her f
```

此外：

- `compare` backend 未触发 `COMPARE_MISMATCH`
- 多 prompt 对拍：
  - `""`
  - `"Once upon a time"`
  - `"Lily was sad because"`
  - `"The little boy found a"`
  均与 `runq.exe` 输出一致
- 说明 descriptor 路径当前没有改坏：
  - `LINEAR`
  - `RMSNorm`
  - `QK_SCORE`
  - `SOFTMAX_AV`
- 对 `LINEAR` 类 op，trace 中已经能看到：
  - `32x4x64` 风格 tile 切分
  - `use_bank / prefetch_bank / out_bank`
  - `k_tile_first / k_tile_last`
  这说明软件合同已经开始按乒乓 buffer 架构组织 `LINEAR` 任务
- `runq_coarse_runtime_explore_batched.exe` 下，多 prompt 对拍仍与 `runq.exe` 完全一致，说明 batched prefill 原型没有改坏行为
- `RMSNorm` 现在已通过 `TRACE_BATCH ... mid_bank=...` 表达 batch-aware / mid-buffer 语义
- `QK_SCORE / SOFTMAX_AV` 现在已通过 `TRACE_BATCH ... heads=... score_bank=... out_bank=...` 表达按 head batch 和 scratch/output bank 轮换语义
- 在非默认格式下，已能运行：
  - `-L acc32`
  - `-N mid16`
  - `-A int8`
  这说明 `ACC32_RAW / MID16 / INT8` 三种输出格式合同已经进入软件原型

---

## 11. 当前合同的边界

### 已经形成的软件合同

- `LINEAR` 类 coarse op 如何描述
- `RMSNorm` 类 coarse op 如何描述
- `QK_SCORE / SOFTMAX_AV` 如何拆分
- `prefill / decode` phase 如何表达
- `W per-row + A per-token` 如何进入 descriptor
- 未来 MMIO 最小字段集合大致是什么

### 还没冻结的部分

- `RMSNorm` 的硬件数学
- `softmax / QK / AV` 的硬件数学
- `ACC32_RAW / MID16 / INT8` 的完整输出格式合同
- `prefill N=4/8` 的真实 batched 执行逻辑

也就是说：

- 现在已经形成的是 **完整 coarse-op 软件合同原型**
- 但其中真正“冻结”的仍然是 `LINEAR` 主线
- `RMSNorm / QK_SCORE / SOFTMAX_AV` 当前是为硬件重构准备的探索性合同原型
- `LINEAR` 已经进一步扩展到：
  - tile-aware
  - bank-aware
  的 mock runtime 组织
- `prefill` 已从“只有 `n_dim_hint`”升级成真正的 batched 执行原型：
  - `Q/K/V/O`
  - `W1/W3/W2`
  在 prefill 阶段按 `N<=4` 一次处理多个 token
- `RMSNorm` 与 `SOFTMAX_AV` 虽然还没做真正的 RTL 对应实现，但软件合同层已经具备：
  - batched 语义
  - bank-aware/mock-bank 语义
  - 输出格式合同

---

## 12. 下一步建议

1. 保持 `runq.c` 不动
2. 继续在 `runq_coarse_runtime_explore.c` 里扩展：
   - `prefill N=8` 与更一般的 batched 执行
   - `MID16 / ACC32_RAW` 输出格式探索
3. 基于 `MockMmioLinearRegs` 反推：
   - RTL descriptor 寄存器组
   - MMIO 提交协议
4. 再开始在硬件侧定义：
   - `OP_CFG`
   - `IN_FMT`
   - `OUT_FMT`
   - `META_ADDR`
   - `NORM_CFG`
   - `SOFTMAX_CFG`

---

## 13. 结论

当前软件契约已经足以支持下一步工作：

- 可以继续做 C runtime 重构
- 可以开始定义硬件 MMIO descriptor
- 可以继续做 Python/C/RTL 三方对拍

并且最重要的一点已经验证：

- **`submit_linear_op(desc)` 这条软件边界是成立的**
