# LLM 量化与现有 Accelerator 协同分析

本文记录基于 `stories260K` 小模型的本地 Win 实验结果，以及对现有 RTL 的最小修改方案判断。目标不是证明某个方案最终最优，而是先把三件事拆开：

1. 量化契约本身是否可行
2. 当前 `post_engine` 的固定点近似是否可接受
3. 如果要尽量少改 RTL，应该把修改集中在哪些数值边界

## 实验环境

- 模型：`stories260K.pt`
- tokenizer：`tok512.model`
- 代码基线：`9th/ref/llama2.c/model.py`
- 运行环境：`D:\anaconda3\python.exe`
- 实验脚本：`tools/eval_quant_schemes.py`
- 默认输出：`artifacts/quant_eval_report.json`

## 实验方案

### 1. 原版浮点

- 直接使用 `model.py` 中的 `Transformer`
- 作为所有方案的参考基线

### 2. 原版量化思路近似

- 方案名：`runq_g64_dyn`
- 目标：近似 `runq.c` 的数学
- 权重：`Q8_0`，按 `group_size=64` 做对称量化
- 激活：运行时按最后一维分组做动态 `absmax` 量化
- 注意：这是 PyTorch 中的 fake-quant / QDQ 代理，不是直接调用 `runq.c`

### 3. 硬件协同静态量化

- 方案名：
  - `static_w_tensor_a_layer`
  - `static_w_row_a_layer`
- 目标：模拟硬件友好的静态 `W8A8/PTQ`
- 权重：
  - `per-tensor` 或 `per-row`
- 激活：
  - 每层 1 个静态 scale
  - scale 由少量 calibration prompt 统计得到

### 4. 平衡型协同量化（新增）

- 方案名：
  - `codesign_w_row_a_token_dyn`
- 目标：在不过度接近 `runq` 的复杂 `per-group` 数学前提下，尽量保住精度并更贴近硬件列数据流
- 权重：
  - `per-row`
- 激活：
  - `per-token` 动态量化
  - 每个输入向量只给 1 个 scale，不沿 `K` 维分 group
- 含义：
  - 比 `runq_g64_dyn` 简单很多
  - 比纯静态 activation 更接近“协同设计”的折中点
### 5. 当前硬件数据流代理

- 方案名：
  - `hw_proxy_row_static_norm`
  - `hw_proxy_row_static_norm_softmax`
- 目标：评估“如果按当前 `post_engine` 的 fixed-point 思路直接上”，精度是否还能接受
- 代理内容：
  - `RMSNorm` 近似采用 `post_engine.sv` 中 `norm_prepare_invstd_q16` 的 `2^(-log2(x)/2)` 型路径
  - `softmax` 采用整数域近似代理
- 注意：这是数值代理，不是逐周期 RTL 仿真

### 6. 静态量化裁剪探索（本轮补充）

- 方案名：
  - `coarse_attn_row_static`
  - `coarse_attn_ffn13_row_static`
  - `coarse_ffn_w2_row_static`
  - `coarse_all_linear_no_cls_row_static`
  - `coarse_all_linear_with_cls_row_static`
- 目标：
  - 不是直接争论“全静态 vs 全动态”，而是先找出哪些子模块最适合优先静态化
  - 这更贴近硬件 V1 的真实落地路径，因为第一版往往不会把所有模块一次性都推到同一种量化契约
- 统一设置：
  - weight：`per-row INT8`
  - activation：离线 calibration 得到的 `per-layer static scale`
- 裁剪含义：
  - `attn`：仅量化 `Q/K/V/O`
  - `ffn13`：量化 `w1/w3`
  - `ffn_w2`：量化 `w2`
  - `all_linear_no_cls`：量化所有线性层，但保留最终 classifier 不量化
  - `all_linear_with_cls`：连最终 classifier 一起静态化

## 结果摘要

以同一段长文本做 teacher-forced 前向，得到：

| 方案 | loss | loss delta vs float | logits cosine | 生成结论 |
| --- | ---: | ---: | ---: | --- |
| float | 10.8303 | 0 | 1.0000 | 基线 |
| runq_g64_dyn | 10.8339 | +0.0036 | 0.999924 | 最接近基线 |
| static_w_tensor_a_layer | 10.7731 | -0.0572 | 0.999338 | 可用，生成结构基本保持 |
| static_w_row_a_layer | 10.8318 | +0.0015 | 0.999509 | logits 接近，但生成分叉更早 |
| codesign_w_row_a_token_dyn | 10.8309 | +0.0006 | 0.999892 | 协同折中方案里最有前景 |
| hw_proxy_row_static_norm | 3.1808 | -7.6494 | 0.523788 | 不可接受，生成崩坏 |
| hw_proxy_row_static_norm_softmax | 3.1266 | -7.7037 | 0.522807 | 同样不可接受 |

本轮新增的“静态量化裁剪探索”结果如下：

| 方案 | loss delta vs float | logits cosine | avg prefix match | avg token match | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| coarse_attn_row_static | -0.0159 | 0.999976 | 65.5 | 0.8250 | 只静态 attention，表现稳 |
| coarse_attn_ffn13_row_static | +0.0099 | 0.999949 | 55.0 | 0.7000 | 加上 `w1/w3` 后开始明显分叉 |
| coarse_ffn_w2_row_static | -0.0376 | 0.999663 | 69.0 | 0.8656 | `w2` 单独静态化也较稳 |
| coarse_all_linear_no_cls_row_static | -0.0216 | 0.999537 | 60.0 | 0.7563 | 大部分主干可静态化，但精度边际下降 |
| coarse_all_linear_with_cls_row_static | +0.0175 | 0.999492 | 34.5 | 0.4375 | 连 classifier 一起静态化后退化明显 |

其中 `avg prefix match` / `avg token match` 是基于 4 个 prompt 的 free-running generation 指标，更接近“用户会直接看到的生成质量”。

### 对结果的解释

- 先澄清一个指标语义：`cross-entropy loss` 是越小越好，不是越大越好。
- 但这里不能只看单条长文本上的 loss。因为这是单样本、teacher-forced 的局部指标，模型即使 free-running generation 已经崩坏，loss 也可能因为分布变尖、对某些目标 token 偶然更自信而变小。
- 对这类 LLM 量化实验，应联合判断：
  - loss / cosine
  - greedy generation 的前缀连续匹配
  - 文本是否出现明显重复、退化、空输出
- `runq_g64_dyn` 很稳，说明 `runq.c` 的动态 group 量化数学本身是可靠的。
- `static W8A8/PTQ` 没有立即把模型打坏，说明“静态量化”这条方向本身值得继续。
- `codesign_w_row_a_token_dyn` 比纯静态 activation 更稳，说明“保留轻量动态 activation、但放弃 `K-group`”是很有潜力的折中点。
- 本轮新增结果进一步说明：
  - “静态量化是否硬件友好”这个问题，答案是肯定的，而且不仅是概念上更友好，数值上也不是一上来就不可用。
  - 但“全链路一次性全静态化”不是最优探索顺序。attention-only 和 `ffn_w2`-only 的静态化更稳，适合作为第一批硬化对象。
  - 最终 classifier 对静态 activation 更敏感。`coarse_all_linear_with_cls_row_static` 明显差于 `coarse_all_linear_no_cls_row_static`，说明输出头最好晚一点再静态化，或者保留更高精度/轻量动态。
- 真正的问题不在量化本身，而在“当前固定点后处理数值路径太激进”。
- 当前 `post_engine` 的数值边界还不适合作为 LLM 主干的直接实现。

## 当前 RTL 中最伤精度的点

### 1. `dequant` 过早压回 `int8`

`post_engine.sv` 中的 `dequant_rne_s8` 会把结果直接 `saturate_int8_sext`：

- 文件：`workspace/accel_rtl/rtl/core/post_engine.sv`
- 行号：`856-877`

这对 `RMSNorm`、`softmax` 之前的中间表示不够。LLM 后处理更需要一个固定点中间域，而不是过早回到 `s8`。

### 2. `RMSNorm` 近似过于粗糙

- `norm_prepare_invstd_q16` 使用 `1/sqrt(x) ~ 2^(-log2(x)/2)` 幂次近似：
  - 行号：`771-801`
- `norm_pass3_transform` 只取 `src[15:0]`，并在较低位宽内完成缩放：
  - 行号：`971-996`

这套路径对 CNN 或简单后处理可能够用，但对 LLM residual stream 太粗，会把隐藏状态分布压坏。

### 3. 当前后处理接口默认语义偏“单次变换”

- `basic_post_transform` 的链路是 `dequant -> bias -> relu/silu`
- 行号：`899-923`

它适合卷积后处理，但不适合 `int32_acc -> higher precision fixed-point -> norm/softmax -> requant` 这种更长的 LLM 链路。

### 4. GEMM 与 POST 的系统握手仍然偏 softmax 专用

- `gemm32x4.sv` 当前使用 `softmax_ack_i / req_softmax_o`
- 文件：`workspace/accel_rtl/rtl/core/gemm32x4.sv`
- 行号：`18-24`, `359-375`

如果后续 `RMSNorm / requant / softmax` 都进入统一 post pipeline，这个接口语义太窄。

## 最小必要修改列表

按优先级排序如下。

### P0: 保留量化主干，先不要硬化动态 activation group quant

不建议把 `runq.c` 的运行时 `absmax` group 动态量化原样搬进硬件。它会把工作量消耗在归约、除法、逐组 scale 管理，而不是阵列和流式后处理。

推荐先把软件契约改为：

- weight：静态 `INT8`
- activation：静态 `per-tensor` 或 `per-row`
- 下游再由硬件执行 `GEMM + fixed-point post`

如果希望进一步降低 V1 风险，建议按模块分两步收敛：

- 第一步：优先探索 `attention` 或 `ffn_w2` 的静态化
- 第二步：扩展到 `all_linear_no_cls`
- 最终 classifier 先不要作为第一版静态化目标

### P1: 把 `dequant` 与 `requant` 拆开

在 `post_engine.sv` 中引入两个阶段：

- `INT32_ACC -> FIXED_MID`
- `FIXED_MID -> INT8`（可选）

这样 `RMSNorm` 和 `softmax` 可以工作在 `FIXED_MID` 域，不再被 `s8` 饱和过早截断。

这是最重要的硬件修改。

### P2: 重新定义 `RMSNorm` 的中间 Q 格式

推荐：

- 输入：`Q16.8` 或 `Q12.12` 一类的中间固定点
- `gamma`：`Q8.8`
- `inv_std`：至少保留 `Q0.16` 或更高

第一版先不要追求 `LayerNorm` 通用化，先把 `RMSNorm` 做稳。

### P3: 把 POST 握手从 softmax 专名改成通用后处理

将：

- `softmax_ack_i`
- `req_softmax_o`

抽象成：

- `post_ack_i`
- `req_post_o`

这样后续 `RMSNorm / requant / softmax` 都能走同一条 owner/调度路径，而不需要继续堆特殊分支。

### P4: 软硬件契约先收敛到静态 `W8A8/PTQ`

建议 V1 不使用 `runq` 式 `K` 维 `per-group` 数学，而采用更易映射到现有数据流的粒度：

- weight：`per-row`
- activation：`per-tensor` 或 `per-token/per-row`

原因不是内存，而是数据流闭合：

- `per-row/per-tensor` 可以“完整 dot-product 后只缩放一次”
- `runq` 式 `per-group` 需要“保留 group partial sum，再按 group scale 分段缩放后求和”

后者需要明显改动 GEMM 输出协议。

## 推荐的 V1 量化契约

### Weight

- `INT8` 对称量化
- 粒度：`per-row`
- scale 存储：每个输出通道 1 个 scale

### Activation

- `INT8` 对称量化
- 粒度：`per-tensor` 或 `per-token/per-row`
- scale 来源：离线 PTQ 校准

### 中间域

- GEMM 输出：`INT32_ACC`
- Post 前半段：`FIXED_MID`
- Norm / attention softmax：在 `FIXED_MID` 域完成
- 层边界：再根据需要 `requant -> INT8`

### 软件保留

V1 建议仍保留软件实现：

- 最终 vocab logits softmax / sampling
- 大词表采样相关逻辑
- 训练与校准脚本

V1 建议下放硬件：

- 线性层 GEMM
- Requant / dequant（改成中间固定点域）
- RMSNorm
- attention 内部的 row softmax

## 最终判断

- 我们不是“还没探索静态量化”，而是已经完成了第一轮，并且本轮又补做了按模块裁剪的静态探索。
- `静态 W8A8/PTQ + 固定点后处理` 方向本身是可行的。
- 从硬件实现角度，静态量化确实更友好，因为它避免了运行时 `absmax` 归约、除法和逐 group scale 管理。
- 现有硬件能承接这条路线的大部分数据流。
- 但更合理的路线不是“全链路纯静态一次到位”，而是：
  - V1 主干优先做 `per-row weight + static activation`
  - 先落在 `attention / ffn_w2 / 非 classifier 主干线性层`
  - classifier 与 norm/softmax 保留更高精度或轻量动态
- 但当前 `post_engine` 的数值边界还不够，需要围绕“不要过早压回 `s8`、把 `RMSNorm` 做稳、接口从 softmax 专用改成通用 post”做最小修改。
- 不建议 V1 继续追求完全兼容 `runq.c` 的动态 group activation 量化。那会显著提高硬件复杂度，而不是提高系统落地概率

## 执行图对齐 QAT 进展

本轮新增了一条不同于 coarse-op 路线的软件探索：

- `tools/qat_exec_graph_explore.py`
- `tools/qat_exec_graph_train.py`
- `doc/exec_graph_qat_exploration_zh.md`

这条路线的目标不是先定义 coarse-op descriptor，而是先验证：

- 如果软件侧显式模拟
  - `INT8(per-token A) x INT8(per-row W)`
  - `ACC32_RAW`
  - `MID16`
  - `RMSNorm / SOFTMAX_ROW`
- 再用 `QAT + 蒸馏` 去适配这张更接近真实硬件执行图的训练图，
- student 是否能在较小硬件改动前提下收敛到更适合现有数据通路的解。

当前结论如下：

> 口径说明：本节主要记录量化/硬件协同设计上的**方案建议与训练先验**，不等于 `deploy/` 当前已冻结实现。
> 当前 deploy V1 的真实默认执行主线仍是：`quantized linears + float attention core`。
> 其中 `rmsnorm / softmax_row` 可以因为 RTL 已有通路而被视为 Fusion Preview 候选，但不能据此推导出 deploy 默认路径已经切到 fused/hardware 执行。

> 当前默认对齐的 deploy QAT best 资产口径是：`deploy/assets/qat_best_compare_finalpolishstrong_seq256/`；
> `stories260K_qat_best/` 应视为官方导出基线，而不是本轮状态记录中的默认 QAT best。

> 因此，文中提到的 `ACC32_RAW / MID16 / RMSNorm / SOFTMAX_ROW` 路线，应理解为“后续硬件化/融合设计的候选空间”，而不是当前 deploy 已完成冻结的执行数学。


### 1. `LINEAR + ACC32_RAW + MID16` 是可训练的

在 GPU 上，`linear_only` 路线用更长训练后：

- `42M teacher` 语料 teacher：
  - `cosine`
    - 从短训练的 `0.8943`
    - 提升到 `0.9399`
- `260K teacher` 语料 teacher：
  - 在同等 `192` 步长训下可到 `0.9549`
- 真实 `TinyStories custom-512` 数据流下：
  - `linear_only` 长训已可到 `0.9586 ~ 0.9675`

这说明：

- 线性主数据通路存在继续训练的收益
- 当前 `0.8943` 不是这条路线的上限
- 在当前 260K student 场景下，更大的 teacher 不是第一瓶颈；训练预算本身同样重要
- 真实数据流比 fallback 小语料更关键

### 2. 当前主瓶颈仍然是 `RMSNorm/post`

更深入排查后发现，前一版 `RMSNorm` 训练代理存在量纲错误：

- 在整数域求出的 `inv_std` 已经隐含了 `input_scale`
- 最终输出又额外乘了一次 `input_scale`

这会把 `RMSNorm` 输出整体压坏。

修正这个问题后，再跑 GPU 训练：

- `260K teacher` 路线：
  - `+norm` 后 `cosine = 0.9199`
  - `+softmax` 后 `cosine = 0.9225`
- `42M teacher` 语料 teacher 路线：
  - `+norm` 后 `cosine = 0.9270`
  - `+softmax` 后 `cosine = 0.9209`

这说明：

- 先前掉到 `~0.49` 的结果，并不完全代表“硬件路线不可行”
- 更准确地说，主问题首先是 `RMSNorm/post` 的量纲和数学定义不对
- 修正后，完整执行图训练已经能进入 `0.92x` 区间
- 在真实 `TinyStories custom-512` 数据和长训下：
  - `+norm` 可到 `0.9707 ~ 0.9713`
  - `+softmax` 仍能保持在 `0.9473 ~ 0.9518`
- 当前主问题已从“整链会崩”转成“生成质量还需继续优化”
- demo 已从“乱码”进入“可读但重复和实体漂移仍较重”的阶段

### 3. softmax 不是当前第一矛盾

从最新训练结果看：

- 在修正过的 `norm` 代理下，`+softmax` 没有把结果再明显拉坏
- 因此当前仍应优先把 `norm` 做稳，再继续细化 softmax

因此当前优先级是：

1. 保持当前修正后的 `RMSNorm/post` 路线
2. 继续在真实数据上提升生成质量
3. 再看是否需要进一步细化 softmax

### 4. 大 teacher 有帮助，但不能掩盖 `norm/post` 问题

`42M teacher` 通过更强语料，对线性阶段是有帮助的；但在同等长训下，它并没有明显超过 `260K teacher`。在修正过的 `norm` 代理下，它也只是略优于 `260K teacher`，差距不大。

因此当前判断是：

- 更大 teacher 值得继续用
- 但主矛盾仍是 `norm/post` 数学，而不是 teacher 太弱
- 如果后续继续扩大 teacher 规模，收益大概率也会弱于先把 `norm/post` 做稳

### 5. 当前最有效的生成质量提升路线

在真实 `TinyStories custom-512` 数据流上继续迭代后，当前更适合继续打磨 demo 的主线已经比较明确：

- `official_ce_push`
- `improved norm`
- `improved softmax`
- `repetition penalty`
- `sampled demo`

代表结果：

- `artifacts/qat_exec_graph_train_task_260k_officialce768_topp_demo.json`
- `artifacts/qat_exec_graph_train_task_260k_finalpolishstrong768_demo.json`

这说明：

- 对 260K 这类小模型，后期训练更贴近原项目的 `CE` 目标，比继续强化蒸馏更有利于生成质量
- 轻量反重复损失能进一步压住模板化和连续重复
- 真正用于展示时，应以 `sampled` 结果为主，而不是只看 greedy
