# 执行图对齐 QAT 探索记录

本文记录一条不同于“粗粒度 Op 软件契约”的新路线：不把硬件侧做成语义很重的 coarse-op 调度器，而是尽量复用现有数据通路，在软件侧构造更贴近真实部署的训练图，并用 `QAT + 蒸馏` 去适配它。

这条路线的核心目标是：

1. 尽量发挥现有硬件数据通路，特别是 `GEMM + post` 组合的加速潜力
2. 在保证 demo 可用的前提下，把硬件改动压到较小
3. 用训练把模型主动拉向“真实硬件执行图”，而不是只做普通 `INT8 QDQ`

## 与 coarse-op 路线的边界

这条路线不写入：

- `coarse_op_software_contract_v1_zh.md`

原因很简单：

- coarse-op 路线强调的是：
  - `LINEAR_ATTN_Q`
  - `RMSNORM_FFN`
  - `SOFTMAX_AV`
  这样的显式 op 语义和 descriptor 契约
- 本文路线强调的是：
  - `现有数据通路 + 最小新增接口`
  - `训练图对齐`
  - `中间格式与后处理数学`

两条路线的目标有交集，但软件/硬件边界不同，文档应分开维护。

## 当前训练图定义

当前 student 固定为：

- `stories260K.pt`

训练图当前按阶段逐步拉入以下代理：

### 第一阶段：`LINEAR + ACC32_RAW + MID16`

- `INT8(per-token activation) x INT8(per-row weight)`
- `ACC32_RAW`
- `MID16`

对应训练代理：

- `QATLinear`

#### 代码对应：线性执行图代理是怎么接入的

下面这段来自 `tools/qat_exec_graph_explore.py`，它定义了线性层代理 `QATLinear`：

```python
class QATLinear(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 执行图语义：INT8(A) x INT8(W) -> ACC32 -> MID16。
        xq = qdq_lastdim_ste(x, self.act_mode, static_scale=self.act_scale)
        wq = qdq_weight_ste(self.weight, self.weight_mode)
        y = F.linear(xq.float(), wq, self.bias.float() if self.bias is not None else None)
        if self.use_acc32:
            y = qdq_acc32_ste(y)
        if self.use_mid16:
            y = qdq_mid16_token_ste(y)
        return y
```

这段代码对应的含义是：

- `qdq_lastdim_ste()`：把激活按 `per-token` fake quant 到 `INT8`
- `qdq_weight_ste()`：把权重按 `per-row` fake quant 到 `INT8`
- `qdq_acc32_ste()`：模拟 `GEMM` 后落到 `ACC32_RAW`
- `qdq_mid16_token_ste()`：模拟层间或 post 入口的 `MID16`

训练脚本里不是直接改 `Transformer.forward()`，而是在模型构建后统一替换模块。下面这段来自 `tools/qat_exec_graph_train.py`：

```python
def build_model_for_stage(...):
    student, _ = eqs.load_model(ckpt_path, device)
    cfg = qx.ExecGraphQATConfig(
        use_approx_norm=stage.use_approx_norm,
        use_approx_softmax="none" if not stage.use_approx_softmax else stage.softmax_impl,
        norm_impl=stage.norm_impl,
        softmax_impl=stage.softmax_impl,
        quantize_output=stage.quantize_output,
        linear_act_mode="dynamic_token",
        linear_weight_mode="per_row",
        use_acc32=True,
        use_mid16=True,
    )
    student = qx.prepare_qat_model(student, linear_scales, rms_scales, cfg)
```

这一步的意思是：

- 先加载原始 `stories260K.pt`
- 再根据当前阶段配置，把普通 `nn.Linear / RMSNorm / softmax` 替换成执行图代理
- 也就是说，训练主循环不变，变的是“student 前向图”

### 第二阶段：`+ RMSNorm proxy`

在第一阶段基础上加入：

- `ApproxRMSNormTrain`
- 或 `ApproxRMSNormTrainImproved`

### 第三阶段：`+ SOFTMAX_ROW proxy`

在第二阶段基础上加入：

- row-softmax 整体代理

## 当前 teacher 设置

当前 teacher 角色分为两类：

### `260K teacher`

- 与 student 同词表
- 可以直接做：
  - `KL distillation`
  - `feature distillation`

### `42M teacher`

- 默认大词表
- 不能直接和 `260K student` 做 logits 对齐蒸馏
- 当前用于：
  - 生成更强的文本/序列语料
  - 扩展 student 的 fallback 训练集

这意味着当前 `42M teacher` 的价值，主要体现在“训练语料质量更高”，而不是直接的同词表 logits 蒸馏。

#### 代码对应：蒸馏是怎么做的

蒸馏的关键有两层：

1. `260K teacher` 与 student 同词表，可直接做 `KL + feature distillation`
2. `42M teacher` 与 student 词表不同，因此只作为“文本 teacher”，不直接做 logits 对齐

teacher 定义代码在 `tools/qat_exec_graph_train.py` 里：

```python
@dataclass
class TeacherBundle:
    # same_vocab=True 表示可以直接做 logits / feature 蒸馏；
    # same_vocab=False 则更适合作为“文本/序列 teacher”来扩展训练语料。
    name: str
    model: nn.Module
    tokenizer: Tokenizer
    same_vocab: bool
```

实际加载逻辑是：

```python
def load_teacher_bundle(name: str, *, quant_eval_dir: str, device: str) -> TeacherBundle:
    if name == "260k":
        model, _ = eqs.load_model(checkpoint_path(quant_eval_dir), device)
        tokenizer = Tokenizer(tokenizer_model=tokenizer_path(quant_eval_dir))
        return TeacherBundle(name="260k", model=model.eval(), tokenizer=tokenizer, same_vocab=True)
    if name == "42m":
        model, _ = eqs.load_model(big_teacher_checkpoint_path(quant_eval_dir), device)
        tokenizer = Tokenizer(tokenizer_model=llama_tokenizer_path())
        return TeacherBundle(name="42m", model=model.eval(), tokenizer=tokenizer, same_vocab=False)
```

loss 组装逻辑在 `run_stage()`：

```python
ce = student.last_loss
kl = kl_distill_loss(s_logits, t_logits.float(), temperature)
feat = masked_feature_loss(s_feat, t_feat, y)
rep = repetition_unlikelihood_loss(
    s_logits,
    x,
    y,
    window=stage.rep_window,
)
loss = (
    stage.ce_weight * ce
    + stage.kl_weight * kl
    + stage.feat_weight * feat
    + stage.rep_weight * rep
)
```

这里四项的作用分别是：

- `CE`：对齐原项目的标准 next-token 训练目标
- `KL`：让 student logits 靠近 same-vocab teacher
- `feature`：让最终 residual stream 也靠近 teacher
- `rep`：直接惩罚重复 token，改善 demo 观感

所谓“`42M teacher` 文本 teacher”，代码上是通过把它生成的文本混入真实数据流来实现，而不是直接拿它的 logits 做监督：

```python
def build_teacher_corpus_texts(bundle: TeacherBundle, *, device: str) -> List[str]:
    texts = [eqs.LONG_TEST_TEXT]
    ...
    with torch.no_grad():
        for prompt in qx.GEN_PROMPTS:
            ids = bundle.tokenizer.encode(prompt, bos=True, eos=False)
            x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]
            greedy = bundle.model.generate(x, max_new_tokens=128, temperature=0.0)
            texts.append(bundle.tokenizer.decode(greedy[0].tolist()))
            sample = bundle.model.generate(x, max_new_tokens=128, temperature=0.8, top_k=32)
            texts.append(bundle.tokenizer.decode(sample[0].tolist()))
```

然后在真实 `task_custom_512` 数据流里按比例混进去：

```python
def build_mixed_text_task_batch_iter(...):
    task_iter = build_task_batch_factory(...)(split="train")
    ...
    while True:
        # 默认 3:1，把真实数据作为主语料，teacher 文本作为辅助语料。
        if mix_iter is not None and step % 4 == 0:
            yield next(mix_iter)
        else:
            yield next(task_iter)
```

也就是说：

- `260K teacher` 负责“硬监督”：`KL + feature`
- `42M teacher` 负责“软数据增强”：给 student 更多更像样的文本分布

## 已完成实验

相关脚本：

- `tools/qat_exec_graph_explore.py`
- `tools/qat_exec_graph_train.py`

当前已经完成：

1. CPU 版 QAT-like 先验验证
2. GPU 版三阶段 smoke 训练
3. `260K teacher` 路线
4. `42M teacher` 语料 teacher 路线
5. 改进版 `RMSNorm` 训练代理初版

结果文件：

- `artifacts/qat_exec_graph_explore.json`
- `artifacts/qat_exec_graph_train_gpu_260k.json`
- `artifacts/qat_exec_graph_train_gpu_42m.json`
- `artifacts/qat_exec_graph_train_gpu_42m_linear96.json`
- `artifacts/qat_exec_graph_train_gpu_260k_linear192.json`
- `artifacts/qat_exec_graph_train_gpu_42m_linear192.json`
- `artifacts/qat_exec_graph_train_gpu_260k_gentle_normv2.json`
- `artifacts/qat_exec_graph_train_gpu_260k_normv2.json`
- `artifacts/qat_exec_graph_train_gpu_260k_gentle_normv3.json`
- `artifacts/qat_exec_graph_train_gpu_42m_gentle_normv3.json`
- `artifacts/qat_exec_graph_train_task_260k_linear192_demo.json`
- `artifacts/qat_exec_graph_train_task_260k_gentle_norm192_demo.json`
- `artifacts/qat_exec_graph_train_task_260k_qualitypush768_demo.json`
- `artifacts/qat_exec_graph_train_task_260k_qualitypush2048_softmaxv2_demo.json`
- `artifacts/qat_exec_graph_train_task_260k_qualitypush768_reploss_demo.json`
- `artifacts/qat_exec_graph_train_task_260k_officialce768_demo.json`
- `artifacts/qat_exec_graph_train_task_260k_officialce768_topp_demo.json`
- `artifacts/demo_sampling_sweep_officialce768.json`
- `artifacts/qat_exec_graph_train_task_mix42m_officialce768_demo.json`

## 结果摘要

### 1. 线性主路径是可训练的，而且训练轮次继续增加仍然有效

在 `42M teacher` 语料 teacher 路线下：

- 短训练 `linear_mid16`：
  - `cosine = 0.8943`
- 更长训练 `linear_only`：
  - `cosine = 0.9399`

在 `260K teacher` 语料 teacher 路线下：

- 更长训练 `linear_only`：
  - `cosine = 0.9549`

这说明：

- `0.8943` 不是上限
- `LINEAR + ACC32_RAW + MID16` 这条主路径确实能继续靠更长训练获得收益
- 在当前这组对照里，“继续训练”带来的收益，比“单纯换更大 teacher 语料”更明显

### 2. `RMSNorm` 代理一度是主瓶颈，但根因首先是量纲错误

在更深入排查后，发现前一版 `ApproxRMSNormTrain` / `ApproxRMSNormTrainImproved` 都存在量纲错误：

- 在整数域上求出的 `inv_std` 已经隐含了 `input_scale`
- 最终输出又额外乘了一次 `input_scale`
- 这会把 `RMSNorm` 输出整体压坏

修正这个问题后，再跑 `gentle_norm + improved norm`：

- `260K teacher` 路线：
  - `+norm` 后 `cosine = 0.9199`
  - `+softmax` 后 `cosine = 0.9225`
- `42M teacher` 语料 teacher 路线：
  - `+norm` 后 `cosine = 0.9270`
  - `+softmax` 后 `cosine = 0.9209`

这说明：

- 先前掉到 `~0.49` 的主因，并不全是“硬件近似天生太差”
- 更大的问题是训练代理里的 `norm` 量纲写错了
- 修正后，完整执行图已经能进入 `0.92x` 量级，不再是灾难性失真

### 3. softmax 目前仍不是第一矛盾

在修正 `norm` 量纲之后：

- `+softmax` 没有把结果再明显拉坏
- `+softmax` 后仍能维持在 `0.92x` 左右

因此当前优先级仍应是：

1. 先把 `RMSNorm` 的硬件数学和中间格式定义彻底定稳
2. 再继续细化 `softmax`

### 4. 42M teacher 有帮助，但不是决定性因素

更大的 teacher 通过更强文本语料，确实让线性阶段结果更好：

- 260K teacher 语料路线短训：`~0.887`
- 42M teacher 语料路线短训：`~0.894`
- 42M teacher 语料路线长训：`~0.940`
- 260K teacher 语料路线长训：`~0.955`

在修正过的 `norm` 代理下：

- `42M teacher` 路线 `+norm` 稍优于 `260K teacher`
- 但差距不大，远小于“修正 norm 公式本身”带来的收益

因此当前判断是：

- 更大 teacher 有价值
- 但在当前 260K student 场景下，teacher 规模不是第一瓶颈
- 当前收益排序更像是：
  1. 修正 `norm/post` 量纲与数学
  2. 增加训练预算
  3. 再考虑更大的 teacher

## 真实 TinyStories custom-512 数据上的长训结果

在把训练从 fallback 小语料切到真实 `TinyStories custom-512` 数据流后，结果进一步说明：

### 1. `linear_only` 路线已经明显有效

- `task_custom_512 + linear_only + 192 steps`
  - `cosine = 0.9586`
- `task_custom_512 + linear_only + 768 steps(quality preset 的第一阶段)`
  - `cosine = 0.9675`

这说明：

- 真实数据比 fallback 小语料更能支撑 student 学到像样的语言结构
- 线性主路径不仅数值上成立，而且 demo 已从“乱码”进到“可读但重复明显”

### 2. 完整执行图在真实数据上也已经稳定

- `task_custom_512 + gentle_norm + improved norm + 768 steps`
  - `+norm` 后 `cosine = 0.9713`
  - `+softmax` 后 `cosine = 0.9473`
- `task_custom_512 + quality_push + improved norm + 768 steps`
  - `+norm` 后 `cosine = 0.9707`
  - `+softmax` 后 `cosine = 0.9518`

这说明：

- 完整执行图在真实数据和足够训练步数下已经是稳定可训的
- `RMSNorm` 和 `softmax` 都不再把系统拉回灾难性区间

### 3. demo 已经明显改善，但还没有到可直接展示的程度

当前 demo 文本样例已经具备：

- 基本句子结构
- TinyStories 风格的开头
- 可读的英文句子

但仍存在：

- 重复较重
- 实体漂移
- 情节推进弱

也就是说：

- 当前问题已经从“执行图会不会崩”变成“如何把生成质量继续往上推”
- 短板主要在生成质量，而不是硬件代理数学本身

## 最新补充结论

### 1. 长训在真实数据上继续有效

- `task_custom_512 + linear_only + 192 steps`
  - `cosine = 0.9586`
- `task_custom_512 + linear_only + 768 steps`
  - `cosine = 0.9675`

这说明：

- 线性主路径不只是“能训”，而且在真实数据上确实还能继续吃到训练预算收益

### 2. 当前改进版 norm 已经进入可用区间

- `task_custom_512 + gentle_norm + improved norm + 768 steps`
  - `+norm` 后 `cosine = 0.9713`
  - `+softmax` 后 `cosine = 0.9473`
- `task_custom_512 + quality_push + improved norm + 768 steps`
  - `+norm` 后 `cosine = 0.9707`
  - `+softmax` 后 `cosine = 0.9518`

这说明：

- 旧的 `~0.49x` 崩坏结论已经失效
- 在修正量纲、使用真实数据并拉长训练后，`norm` 代理已经进入可用区间

### 3. softmax 已成为下一阶段更细的优化点

- 当前 `softmax` 不再把系统拉回灾难性区间
- 但从 `+norm` 的 `0.97x` 回落到 `+softmax` 的 `0.95x` 左右，说明 softmax 现在开始成为更细粒度的优化点

### 4. demo 质量已从“不可读”提升到“可读但不够展示”

- 优点：
  - 开头稳定
  - 句法基本成形
  - TinyStories 风格明确
- 问题：
  - 重复较重
  - 实体漂移
  - 情节推进弱

更准确地说：

- 这条路线已经跨过“能不能用”的阶段
- 现在进入“怎么把 demo 做好看”的阶段

### 5. 混入 42M teacher 文本暂未带来决定性改进

- `task_custom_512 + mix_teacher_text(42M) + quality_push + 768 steps`
  - `+softmax` 后 `cosine = 0.9499`
- `task_custom_512 + mix_teacher_text(42M) + gentle_norm + 768 steps`
  - `+softmax` 后 `cosine = 0.9737`

它说明：

- 42M teacher 文本混入是可用的
- 但目前收益仍不如“真实数据 + 正确量纲 + 更长训练”本身显著
- teacher 文本可继续作为辅助，不是当前第一优先级
- 当前优化顺序仍应是：
  1. 线性主路径继续训练
  2. 生成质量优化
  3. teacher 文本混入作为辅助

### 6. 千步级以上长训仍然有效，但收益已开始放缓

- `task_custom_512 + quality_push + improved norm + improved softmax + 2048 steps`
  - `linear_mid16_quality` 后 `cosine = 0.9738`
  - `add_norm_quality` 后 `cosine = 0.9780`
  - `add_softmax_quality` 后 `cosine = 0.9763`

这说明：

- 继续拉长训练仍然有效
- 但收益已经从“显著改善”进入“边际改善”阶段

### 7. 反重复损失和更贴近官方 CE 的训练，确实能缓解重复

新增两组对照：

1. `quality_push + repetition penalty`
   - 文件：
     - `artifacts/qat_exec_graph_train_task_260k_qualitypush768_reploss_demo.json`
   - 结果：
     - `+softmax` 后 `cosine = 0.9733`
     - `max_repeat_run = 2`

2. `official_ce_push + repetition penalty`
   - 文件：
     - `artifacts/qat_exec_graph_train_task_260k_officialce768_demo.json`
   - 结果：
     - `+softmax` 后 `cosine = 0.9726`
     - `max_repeat_run = 1`

这说明：

- 参考原项目的 `CE` 主训练目标是有帮助的
- 加入小的重复抑制损失后，重复问题确实有所缓解
- 数值不会明显崩

### 8. 当前最好的 demo 已经从“可读”走到“勉强可展示”

和最早的乱码相比，现在的最佳样例已经具备：

- 基本正常的句法
- 明确的 TinyStories 风格
- 相对稳定的开头和句子结构

但仍有明显问题：

- 人物/实体重复
- 事件推进单薄
- 模板化句子偏多

所以当前更准确的判断是：

- 这条路线已经可以做“勉强可展示”的 demo
- 但距离“高质量展示”还有一段距离

### 9. 当前最有效的生成质量改进来自“官方 CE + 反重复”

继续围绕生成质量做迭代后，当前最有效的方向是：

- 后期训练更偏 `CE`
- 同时加入轻量 `repetition penalty`
- 保留完整执行图与真实 `TinyStories custom-512` 数据

对应实验：

- `artifacts/qat_exec_graph_train_task_260k_officialce768_demo.json`

这组结果说明：

- `+softmax` 后 `cosine = 0.9726`
- `distinct1 = 0.2514`
- `distinct2 = 0.4870`
- `max_repeat_run = 1`

也就是说：

- 数值没有明显下降
- 重复问题比只做 `quality_push` 更可控
- 当前最值得继续沿着这条线推进

补充最新结果：

- `task_custom_512 + official_ce_push + improved norm + improved softmax + 768 steps`
  - `+softmax` 后 `cosine = 0.9726`
  - `distinct1 = 0.2514`
  - `distinct2 = 0.4870`
  - `max_repeat_run = 1`
- 从该配置继续做 sampled demo 输出：
  - `top-p = 0.9` 的 sampled 文本明显比 greedy 更自然

### 10. sampled demo 明显优于 greedy demo

在加入 sampled demo 评估后发现：

- greedy 输出仍然偏模板化、偏重复
- `temperature = 1.0, top-p = 0.9` 的 sampled 输出通常更自然
- 对 260K 这种小模型，demo 不应只看 greedy

因此后续如果要做展示，建议：

- 训练继续按当前完整执行图推进
- demo 端优先展示 sampled 结果，而不是只展示 greedy
- `greedy` 更适合作为稳定性和回归基线

对当前 `260K` 模型，比较推荐的 sampled 口径是：

- `temperature = 0.95`
- `top-p = 0.9`
- `top-k = 40`

这组在当前 checkpoint 上兼顾了：

- 相对更自然的故事展开
- 更少的模板化重复
- 不会像高温采样那样明显跑偏

## 当前判断

目前可以比较明确地下结论：

1. 这条“执行图对齐的 `QAT + 蒸馏` 路线”是成立的
2. `LINEAR + ACC32_RAW + MID16` 主路径值得继续投入训练
3. 在修正量纲后，`RMSNorm/post` 已不再是“整链会崩”的主瓶颈，已经进入可用区间
4. 当前最主要的剩余问题是生成质量：重复、实体漂移、情节推进弱
5. 增加训练轮次对线性阶段和完整执行图都有效，但收益已开始进入边际改善阶段

## 下一步

下一步建议按这个顺序推进：

1. 继续在真实 `TinyStories custom-512` 数据上长训
2. 重点从“数值可行性”转到“生成质量提升”
3. 优先尝试：
   - 更长完整执行图训练
   - teacher 文本混入比例调整
   - 更贴近 demo 的训练/评估策略
4. `norm_v2` 和 `softmax_v2` 仍可继续细化，但它们现在已经不是唯一主线
