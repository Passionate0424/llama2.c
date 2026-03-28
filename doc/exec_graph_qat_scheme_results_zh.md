# 执行图对齐 QAT 方案与结论

本文不是工作日志，而是面向阅读者总结当前“执行图对齐 QAT”路线到底做了什么、做到哪一步、当前最佳方案是什么。

对应更完整的过程记录见：

- `doc/exec_graph_qat_exploration_zh.md`

对应主要代码见：

- `tools/qat_exec_graph_explore.py`
- `tools/qat_exec_graph_train.py`

---

## 1. 当前方案是什么

当前路线的核心思想是：

- 不把硬件做成语义很重的 coarse-op 调度器
- 尽量复用现有数据通路
- 在软件侧把 student 模型前向图替换成“更贴近真实硬件执行图”的代理
- 再用 `QAT + 蒸馏` 去适配这张训练图

student 固定为：

- `stories260K.pt`

当前训练图对应的硬件执行图语义是：

1. `INT8(per-token activation) x INT8(per-row weight)`
2. `ACC32_RAW`
3. `MID16`
4. `RMSNorm proxy`
5. `SOFTMAX_ROW proxy`

也就是说，我们不是只做普通的 `INT8 QDQ`，而是在训练时显式模拟：

- GEMM 的输入量化
- GEMM 输出累加域
- 中间格式
- 后处理近似

---

## 2. 当前最佳方案

如果目标是“数值好 + demo 较像样”，当前最推荐的配置是：

- 数据：`TinyStories custom-512`
- student：`stories260K.pt`
- 训练主线：
  - `official_ce_push`
  - `improved norm`
  - `improved softmax`
  - `repetition penalty`
- 演示口径：
  - 以 `sampled` 为主
  - 不以 `greedy` 为主

当前最值得保留的结果文件：

- 训练结果：
  - `artifacts/qat_exec_graph_train_task_260k_officialce768_topp_demo.json`
  - `artifacts/qat_exec_graph_train_task_260k_finalpolishstrong768_demo.json`
- 演示采样扫表：
  - `artifacts/demo_sampling_sweep_finalpolishstrong768.json`

当前更推荐的 sampled 演示参数：

- `temperature = 0.95`
- `top_p = 0.9`
- `top_k = 40`

---

## 3. 当前结果如何

### 3.1 数值结果

当前较好的数值结果大致如下：

- `linear_only` 在真实数据长训后：
  - `cosine` 可到 `0.96+`
- 完整执行图在真实数据长训后：
  - `cosine` 可到 `0.97+`

这说明：

- 路线已经不是“能不能训”的问题
- 执行图对齐代理已经可以稳定训练

### 3.2 demo 结果

当前 demo 的更准确评价是：

- `greedy`：
  - 还能看到模板化和重复
  - 更适合做回归与稳定性基线
- `sampled`：
  - 已经比 greedy 更自然
  - 可以作为当前的主要演示口径

如果用“展示效果”来分级：

- 乱码/不可读：已经完全跨过
- 可读但差：已经跨过
- 较好演示：目前已经接近，并基本达到
- 高质量自然叙事：还没达到

当前残留问题主要是：

- 实体漂移
- 模板化句式
- 故事推进不够自然

---

## 4. 训练代理是怎么做的

### 4.1 线性路径代理

下面这段来自 `tools/qat_exec_graph_explore.py`，定义了线性执行图代理：

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

它实际对应的是：

- 激活先按 `per-token` fake quant 到 `INT8`
- 权重按 `per-row` fake quant 到 `INT8`
- GEMM 输出先走 `ACC32_RAW`
- 然后再压到 `MID16`

这一步是在 student 前向图里做的，不是只在评估时做后处理。

### 4.2 后处理代理

`RMSNorm` 代理在 `tools/qat_exec_graph_explore.py` 里也有实现。

当前更稳定的是 `ApproxRMSNormTrainImproved`：

```python
class ApproxRMSNormTrainImproved(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_mid = qdq_mid16_token_ste(x.float())
        ex2 = torch.clamp((x_mid * x_mid).mean(dim=-1, keepdim=True), min=1e-8)
        invstd = torch.rsqrt(ex2)
        invstd_q16 = ste_round(invstd * 65535.0) / 65535.0
        gamma_q12 = ste_round(self.weight.float() * 4096.0) / 4096.0
        out = x_mid * invstd_q16
        out = out * gamma_q12
        return out
```

这段代码代表：

- 先把输入拉到 `MID16` 风格域
- 在原浮点量纲里算 `RMS`
- 再把 `inv_std` 和 `gamma` 量化成硬件更像能承接的系数

`softmax` 也有对应代理：

```python
def approx_softmax_scores_train_improved(scores: torch.Tensor) -> torch.Tensor:
    scores_mid = qdq_mid16_token_ste(torch.nan_to_num(scores.float(), neginf=-1.0e4, posinf=1.0e4))
    probs = F.softmax(scores_mid, dim=-1)
    probs_q16 = ste_round(probs * 65535.0) / 65535.0
    denom = probs_q16.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return (probs_q16 / denom).to(scores.dtype)
```

这说明我们不只是模拟线性量化，而是连 `RMSNorm / softmax` 的有限精度行为也显式纳入了训练图。

---

## 5. 这些代理是怎么接进模型里的

训练时并不改 `train.py` 的主循环，而是在模型构建后、优化器创建前，把 student 模型替换成代理版。

对应代码在 `tools/qat_exec_graph_train.py`：

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

也就是说：

- 原始 checkpoint 先加载出来
- 再把其中的线性层、norm、softmax 替换成代理
- 训练主循环本身不需要大改

---

## 6. 蒸馏是怎么做的

蒸馏有两层：

### 6.1 同词表 teacher：260K

因为 `260K teacher` 和 `260K student` 共用 `tok512.model`，所以可以直接做：

- `KL distillation`
- `feature distillation`

teacher 定义是：

```python
@dataclass
class TeacherBundle:
    name: str
    model: nn.Module
    tokenizer: Tokenizer
    same_vocab: bool
```

实际加载逻辑：

```python
def load_teacher_bundle(name: str, *, quant_eval_dir: str, device: str) -> TeacherBundle:
    if name == "260k":
        model, _ = eqs.load_model(checkpoint_path(quant_eval_dir), device)
        tokenizer = Tokenizer(tokenizer_model=tokenizer_path(quant_eval_dir))
        return TeacherBundle(name="260k", model=model.eval(), tokenizer=tokenizer, same_vocab=True)
```

### 6.2 不同词表 teacher：42M

`42M teacher` 使用默认大词表，不能直接和 `260K student` 做 logits 对齐。

所以它现在主要作为：

- 文本 teacher
- 用来生成更丰富的训练文本

对应加载逻辑：

```python
if name == "42m":
    model, _ = eqs.load_model(big_teacher_checkpoint_path(quant_eval_dir), device)
    tokenizer = Tokenizer(tokenizer_model=llama_tokenizer_path())
    return TeacherBundle(name="42m", model=model.eval(), tokenizer=tokenizer, same_vocab=False)
```

然后通过文本生成混进训练集：

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

这就实现了：

- `260K teacher` 负责硬监督
- `42M teacher` 负责软数据增强

---

## 7. loss 是怎么组装的

训练主循环里 loss 的组成在 `tools/qat_exec_graph_train.py` 里很清楚：

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

四项分别表示：

- `CE`：对齐原项目的 next-token 训练目标
- `KL`：让 student logits 接近 teacher
- `feature`：让 student 的最终 residual stream 接近 teacher
- `rep`：显式惩罚重复 token，高度针对 demo 质量

当前最有效的阶段预设是 `official_ce_push`，它更贴近原项目的 CE 训练思路：

```python
def official_ce_push_stages(...):
    return [
        TrainStage(
            name="linear_mid16_official",
            ce_weight=0.25,
            kl_weight=0.90,
            feat_weight=0.04,
            rep_weight=0.0,
        ),
        TrainStage(
            name="add_norm_official",
            ce_weight=0.70,
            kl_weight=0.15,
            feat_weight=0.01,
            rep_weight=0.08,
        ),
        TrainStage(
            name="add_softmax_official",
            ce_weight=0.85,
            kl_weight=0.05,
            feat_weight=0.0,
            rep_weight=0.12,
        ),
    ]
```

这组配置反映的思路是：

- 前期用蒸馏帮 student 对齐执行图
- 后期逐渐把重点转回 `CE`
- 同时用轻量反重复约束把 demo 质量拉上去

---

## 8. 为什么 sampled demo 比 greedy 更重要

当前我们已经在评估里同时保留：

- `greedy demo`
- `sampled demo`

采样逻辑是：

```python
@torch.no_grad()
def generate_with_top_p(model, idx, max_new_tokens, *, temperature, top_p):
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.params.max_seq_len else idx[:, -model.params.max_seq_len :]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)
        ...
        next_sorted = torch.multinomial(sorted_probs, num_samples=1)
        next_token = torch.gather(sorted_idx, -1, next_sorted)
        idx = torch.cat((idx, next_token), dim=1)
    return idx
```

当前对 260K 模型更适合展示的 sampled 口径是：

- `temperature = 0.95`
- `top_p = 0.9`
- `top_k = 40`

原因很简单：

- greedy 更适合回归测试
- sampled 更接近真实演示观感
- 对 260K 这种小模型，sampled 通常比 greedy 更自然

---

## 9. 当前最佳结果如何理解

当前最佳方向可以概括为：

- student：`stories260K.pt`
- 数据：`TinyStories custom-512`
- 训练：`official_ce_push + repetition penalty`
- 代理：`improved norm + improved softmax`
- 演示：`sampled(top-p)`

当前这个组合已经把路线推进到：

- 数值上：`0.97x` 级别
- 演示上：`greedy` 勉强可展示，`sampled` 接近较好演示

这也是我当前对外汇报时最推荐的一套口径。
