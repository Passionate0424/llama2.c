# 独立部署运行时实施记录

本文记录 `deploy/` 当前真实实现状态，目标是把“当前默认资产 / 默认 workload / 算子冻结 / 融合边界 / 控制面口径”统一写清楚。

## 1. 当前状态摘要
- 默认对齐的 QAT best 资产：`deploy/assets/qat_best_compare_finalpolishstrong_seq256/`
- 官方导出基线资产：`deploy/assets/stories260K_qat_best/`（保留为基线口径，不是本轮默认 QAT best）
- 默认 decode/workload 参考：`deploy/include/runtime_decode_cfg.h`
- 当前 deploy 主线：**MMIO 优先，queue 暂不需要**
- 当前 deploy V1：**quantized linears + float attention core**

## 2. 执行层冻结与融合边界
### 2.1 已冻结的执行层边界
- `linear_qkv`：已作为 fused coarse op 收口
- `linear_attn_o`
- `linear_ffn_w1`
- `linear_ffn_w3`
- `gate_mul`
- `linear_ffn_w2`
- `lm_head`

### 2.2 当前仍为 float attention core
- `qk_matmul`
- `softmax_row`
- `av_matmul`
- `rmsnorm`
- `residual_add`
- `final_norm`

### 2.3 Fusion Preview / 语义冻结层
- `rmsnorm`：已有 `post_engine` 通路基础，可前移为融合候选，但当前默认执行仍是独立 float 算子
- `softmax_row`：已有 `post_engine` 通路基础，可前移为融合候选，但当前默认执行仍是独立 float 算子
- `norm + linear`：后续局部融合候选

### 2.4 后续项
- `qk+softmax+av` 一体融合
- int `qk_matmul`
- int `av_matmul`
- 更激进 attention 深融合

## 3. 控制面口径
当前软件与 bring-up 路径统一收敛为：
- `legacy/fixed-job MMIO`
- 不把 queue 命令硬件、queue submit/completion 作为当前主线要求
- `CMDQ/CMPQ/DBG2` 只保留为历史/兼容地址窗口，不作为当前验收项

## 4. 共享区与地址口径
当前已保留并使用的共享区：
- `IO_IN`
- `IO_OUT`
- `PARAM`
- `KV_MAIN`
- `SCRATCH`
- `TRACE`

当前已保留但仅作为历史/兼容占位的地址窗口：
- `CMDQ`
- `CMPQ`
- `DBG2`

## 5. 运行时与验证
当前 `runq_verify` 已覆盖：
- `rmsnorm`
- `linear_qkv_q/k/v`
- `qk_matmul`
- `softmax_row`
- `av_matmul`
- `gate_mul`
- `residual_add`
- `kv_main_map`

当前验证口径对应：
- `SW_REF -> HW_STUB -> compare`
- 已冻结后的 V1 合同：`quantized linears + float attention core`

## 6. 当前阶段一句话
**MMIO 优先，queue 暂不需要；RMSNorm / Softmax 已前移为 Fusion Preview 候选，但仍不改变 deploy V1 的默认执行主线。**
