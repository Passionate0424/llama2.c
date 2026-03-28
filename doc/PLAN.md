# `stories260K` 完整硬件执行图 QAT 计划（含 42M vs 260K teacher 对照）

## Summary
- student 固定为 `stories260K`
- 最终目标固定为：`完整硬件执行图`，包括
  - `INT8(per-token activation) x INT8(per-row weight)`
  - `ACC32_RAW`
  - `MID16`
  - `RMSNorm proxy`
  - `SOFTMAX_ROW proxy`
- 优先级固定为：`速度/硬件利用率 > 精度可用 > 硬件改动`
- 精度要求固定为：`demo 可用`
- 训练策略固定为：`staged QAT + distillation`
- teacher 对照固定为：
  - `42M teacher`
  - `260K teacher`
- 这样能直接回答：`大 teacher 是否真的比 260K teacher 更有用`

## Key Changes
- 不更新 [coarse_op_software_contract_v1_zh.md](/e:/project/ciciec/9th/ref/llama2.c/doc/coarse_op_software_contract_v1_zh.md)
- 新建独立文档：
  - `doc/exec_graph_qat_exploration_zh.md`
- 总览文档只补结论：
  - [llm_quant_hw_codesign_zh.md](/e:/project/ciciec/9th/ref/llama2.c/doc/llm_quant_hw_codesign_zh.md)
- 基线代码完全不动：
  - `runq.c`
  - `runq_embedded.c`
  - 现有 coarse-op 合同与 runtime
- 新增/保留训练脚本路线：
  - 保留 [qat_exec_graph_explore.py](/e:/project/ciciec/9th/ref/llama2.c/tools/qat_exec_graph_explore.py) 做小规模先验
  - 正式训练主线使用独立脚本 `tools/qat_exec_graph_train.py`
- 本地准备 42M teacher checkpoint 作为主对照 teacher

## Implementation Changes
### 1. Teacher / Student 设定
- student 固定为 `stories260K.pt`
- teacher 两组固定：
  - `stories260K.pt`
  - `stories42M.pt`
- tokenizer 保持一致，避免词表差异污染对照
- 如果 42M teacher 本地尚未准备，先把下载与缓存纳入实验准备步骤

### 2. 训练图
- student 前向图插入可训练硬件代理：
  - `QATLinear`
  - `ACC32_RAW` 代理
  - `MID16` 代理
  - `ApproxRMSNormTrain`
  - `ApproxSoftmaxRowTrain`
- 最终部署目标图始终包含 `RMSNorm/softmax`
- 训练中采用 staged curriculum，而不是第一步全开

### 3. 训练阶段
- 阶段 1：`linear_mid16`
  - 图中只启用 `LINEAR + ACC32_RAW + MID16`
  - 分别用 `260K teacher` 和 `42M teacher` 训练
- 阶段 2：`linear_mid16_norm`
  - 在各自阶段 1 checkpoint 上加入 `RMSNorm proxy`
  - 继续分别跑 `260K teacher` 和 `42M teacher`
- 阶段 3：`linear_mid16_norm_softmax`
  - 在各自阶段 2 checkpoint 上加入 `SOFTMAX_ROW proxy`
  - 继续分别跑 `260K teacher` 和 `42M teacher`
- 每个阶段都使用相同数据、相同 student、相同训练预算，只改变 teacher

### 4. 损失函数
- 固定为蒸馏主导：
  - `loss = kl_distill + alpha * ce + beta * feature_mse`
- 默认不再只用当前小脚本里的 `mse + 0.25*ce`
- 第一版系数默认固定，不在主实验里做大搜索
- 若训练不稳定，再单独调损失权重，不和 teacher 对照混在一起

### 5. 数据与缓存
- 不依赖新增外部数据集下载作为主实验前提
- 数据来源固定为：
  - 现有 prompts / long text / calibration text
  - teacher 扩展生成文本
  - 若本地已有 TinyStories 预分词数据，则切换到正式数据流
- teacher logits / features 缓存到本地，避免重复前向
- 训练/验证切分固定，不能再只做单小集合 smoke

### 6. 硬件代理优先级
- `ACC32_RAW`
  - 视为必须项
  - 先对齐，不再允许过早回 `INT8`
- `MID16`
  - 视为主中间域
  - 是第二优先级
- `RMSNorm`
  - 当前粗近似保留为对照实现
  - 同时准备一版更稳但仍硬件可实现的改进代理
  - 如果阶段 2 明显失败，优先替换这里
- `SOFTMAX_ROW`
  - 先按 row-softmax 整体代理实现
  - 若阶段 3 失败，再单独定位 softmax 近似问题

## Test Plan
### 1. 对照实验矩阵
- `teacher=260K`, `phase=linear_mid16`
- `teacher=42M`, `phase=linear_mid16`
- `teacher=260K`, `phase=linear_mid16_norm`
- `teacher=42M`, `phase=linear_mid16_norm`
- `teacher=260K`, `phase=linear_mid16_norm_softmax`
- `teacher=42M`, `phase=linear_mid16_norm_softmax`

### 2. 统一输出指标
- `loss`
- `loss_delta`
- `cosine`
- `avg_prefix_match_tokens`
- `avg_token_match_ratio`
- 代表性生成文本前缀
- 训练曲线头尾
- 每阶段收敛速度与稳定性

### 3. 结论判定
- 若 `42M teacher` 在阶段 1 就明显优于 `260K teacher`
  - 说明大 teacher 对执行图对齐 QAT 有实质价值
- 若阶段 1 两者差距不大
  - 说明当前主瓶颈不在 teacher 规模，而在代理数学
- 若阶段 2 两者都显著恶化
  - 结论固定为 `RMSNorm 近似是主瓶颈`
- 若阶段 2 里 `42M teacher` 明显好于 `260K teacher`
  - 说明更强蒸馏能帮助 student 适应不完美 norm 近似
- 若阶段 3 相对阶段 2 没进一步恶化
  - softmax 不是第一矛盾
- 若阶段 3 明显恶化
  - `SOFTMAX_ROW proxy` 成为下一优先问题

### 4. 验收标准
- 以 `demo 可用` 为硬标准
- 任一阶段若生成明显退化、重复、空输出、语义崩坏，则该阶段视为失败
- 若阶段失败，不继续扩大硬件覆盖，而是先改对应硬件代理数学

## Assumptions
- `stories260K` 是唯一 student
- `42M` 是唯一大 teacher 主对照
- `260K vs 42M` 是最小可信对照，不再引入更多 teacher 规模以免实验面过大
- staged QAT 是训练策略，不改变最终部署目标图
- 文档职责严格分离：
  - coarse-op 文档不承载这条路线
  - 新文档承载完整细节
  - 总览文档只收结论
