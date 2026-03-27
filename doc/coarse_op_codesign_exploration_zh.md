# 粗粒度 Op 协同探索记录

## 目标

围绕“CPU 提交粗粒度 op，硬件在单 IP 内完成有限融合”的路线，先做本地先验探索，回答三个问题：

1. 哪些算子族最值得先下放到硬件
2. `per-row weight + per-token activation` 是否足够稳
3. `MID16` 这类中间格式是否会明显恶化精度

## 本次新增/扩展

### Python 侧

- 扩展实验脚本：
  - `E:\project\ciciec\9th\ref\llama2.c\tools\eval_quant_schemes.py`
- 新增粗粒度 op 相关 scheme：
  - `coarse_attn_row_token`
  - `coarse_attn_ffn13_row_token`
  - `coarse_ffn_w2_row_token`
  - `coarse_attn_ffn13_row_token_mid16`
  - `coarse_all_linear_no_cls_row_token`
  - `coarse_all_linear_no_cls_row_token_mid16`
  - `coarse_all_linear_with_cls_row_token`
- 新增聚合指标：
  - `avg_prefix_match_tokens`
  - `avg_token_match_ratio`

### C 侧

- 新增 coarse-op probe：
  - `E:\project\ciciec\9th\ref\llama2.c\tools\runq_coarse_op_probe.c`
- 用途：
  - 读取 `runq` 模型配置
  - 输出每层线性算子粗粒度 op 列表
  - 统计 `attention / ffn_in / ffn_out / classifier` 的 MAC 占比
  - 评估 `chunk_n=1/4/8` 时的形状和片外写回量级

- 新增独立 runtime 结构探索工具：
  - `E:\project\ciciec\9th\ref\llama2.c\tools\runq_coarse_runtime_explore.c`
- 该工具在不修改 `runq.c` 基线的前提下，验证：
  - `AccelOp / AccelDesc`
  - `submit_linear_op()`
  - `forward()` 中的显式 coarse linear op 边界
  - `generate()` 拆成 `prefill` 与 `decode`
- 当前这一步仍然保持软件 fallback 数学不变，目的是先验证软件 runtime 结构可行性。

## 实验命令

### Python

```powershell
D:\anaconda3\python.exe tools\eval_quant_schemes.py --output artifacts\quant_eval_report_coarse_ops.json
```

### C baseline

```powershell
.\runq.exe artifacts\stories260K\stories260K_q80.bin -z artifacts\stories260K\tok512.bin -t 0.0 -n 40 -i "Once upon a time"
.\runq_embedded.exe -t 0.0 -n 40 -i "Once upon a time"
```

### C probe

```powershell
gcc -Ofast -D_WIN32 -I. -o tools\runq_coarse_op_probe.exe tools\runq_coarse_op_probe.c win.c -lm

tools\runq_coarse_op_probe.exe artifacts\stories260K\stories260K_q80.bin 1
tools\runq_coarse_op_probe.exe artifacts\stories260K\stories260K_q80.bin 4
tools\runq_coarse_op_probe.exe artifacts\stories260K\stories260K_q80.bin 8
```

## 结果摘要

### 1. C baseline 已确认

- `runq.exe` 与 `runq_embedded.exe` 在同 prompt 下输出一致
- 本地 40 token greedy 示例输出一致：
  - `Once upon a time, there was a little girl named Lily. She loved to play outside in the phower and see all of her f`
- 两者都报告：
  - `achieved tok/s: 6500.000000`
- `runq_coarse_runtime_explore.exe` 在同 prompt 下也得到相同文本输出
- `test.c` 也已在 Windows 兼容编译参数下通过：
  - `ALL OK`

这说明：

- `runq_embedded` 可作为后续“无文件系统/静态镜像”软件链的稳定基线
- 后面做 coarse-op runtime 时，`runq_embedded.c` 可以继续保持基线不变
- `runq_coarse_runtime_explore.c` 已证明 `submit_linear_op(desc)` 与 `prefill/decode` 结构化改造至少在当前软件 fallback 路径上没有破坏推理行为

### 2. Python：最值得先下放的是 attention projections

结果文件：

- `E:\project\ciciec\9th\ref\llama2.c\artifacts\quant_eval_report_coarse_ops.json`

关键结果：

| scheme | loss delta | cosine | avg prefix | avg token match |
|---|---:|---:|---:|---:|
| `codesign_w_row_a_token_dyn` | `+0.0006` | `0.999892` | `66.25` | `0.8344` |
| `coarse_attn_row_token` | `-0.0105` | `0.999989` | `79.50` | `0.9938` |
| `coarse_attn_ffn13_row_token` | `-0.0104` | `0.999966` | `69.50` | `0.8750` |
| `coarse_ffn_w2_row_token` | `-0.0042` | `0.999957` | `63.25` | `0.7906` |
| `coarse_attn_ffn13_row_token_mid16` | `-0.0052` | `0.999967` | `69.50` | `0.8750` |
| `coarse_all_linear_no_cls_row_token` | `-0.0162` | `0.999908` | `53.25` | `0.6750` |
| `coarse_all_linear_no_cls_row_token_mid16` | `-0.0160` | `0.999918` | `50.00` | `0.6344` |
| `coarse_all_linear_with_cls_row_token` | `-0.0174` | `0.999888` | `54.50` | `0.6875` |
| `hw_proxy_row_static_softmax_only` | `-2.1660` | `0.941102` | `3.00` | `0.0531` |
| `hw_proxy_row_static_norm` | `-7.6494` | `0.523788` | `0.00` | `0.0000` |

结论：

- `attention projections only` 几乎不伤生成，极适合作为 V1 硬件粗粒度 op
- `attention + w1/w3` 也很有前景，但明显不如 “attention only” 稳
- `w2` 单独下放虽然 cosine 也高，但文本一致性明显更差
- `MID16` 在 `attention + w1/w3` 这组实验里没有带来额外灾难，说明中间格式路线值得继续
- 当前硬件代理的 `softmax / norm` 仍然明显不稳，特别是 norm

### 2.1 C 软件契约原型已扩展到完整 coarse-op

在 `runq_coarse_runtime_explore.c` 中，当前已经形成并验证：

- `LINEAR_*`
- `RMSNORM_*`
- `QK_SCORE`
- `SOFTMAX_AV`

三类 coarse-op 的统一 descriptor / mock MMIO / compare 路径。

并且：

- `LINEAR` 路径已经增加 tile-aware / bank-aware mock 调度
- trace 中可看到：
  - `m_tile / k_tile`
  - `use_bank / prefetch_bank / out_bank`
  - `k_first / k_last`

这说明软件侧已经开始按“乒乓 buffer + tile overlap”的硬件思路组织 `LINEAR` 任务。

注意：

- 这仍然是 mock runtime，不是实际 MMIO 驱动
- `prefill N=4` 现在已经升级成真正的 batched 执行原型：
  - `Q/K/V/O`
  - `W1/W3/W2`
  在 prefill 阶段按 `N<=4` 一次处理多个 token
  - 并且多 prompt 对拍与 `runq.exe` 保持一致
- `RMSNorm` 和 `QK_SCORE / SOFTMAX_AV` 也继续往前推进了：
  - `RMSNorm` 已有 batch-aware / `mid_bank` mock 语义
  - `QK_SCORE / SOFTMAX_AV` 已有按 head batch 和 `score_bank / out_bank` mock 语义
  - 输出格式合同已能探索：
    - `LINEAR -> ACC32_RAW`
    - `RMSNorm -> MID16`
    - `SOFTMAX_AV -> INT8`

### 3. C probe：MAC 覆盖率很清楚

结果文件：

- `E:\project\ciciec\9th\ref\llama2.c\artifacts\runq_coarse_op_probe_n1.txt`
- `E:\project\ciciec\9th\ref\llama2.c\artifacts\runq_coarse_op_probe_n4.txt`
- `E:\project\ciciec\9th\ref\llama2.c\artifacts\runq_coarse_op_probe_n8.txt`

对 `stories260K_q80.bin`，`chunk_n=1` 时：

- `attn_macs = 61440`
- `ffn_in_macs = 110080`
- `ffn_out_macs = 55040`
- `cls_macs = 32768`
- `total_wo_cls = 226560`
- `attn + ffn_in` 占比：
  - `0.757062`
- `w2` 占比：
  - `0.242938`

这说明：

- 如果 V1 先下放 `attention + w1/w3`，已经覆盖了除 `wcls` 外线性层 MAC 的约 `75.7%`
- `w2` 的收益存在，但不是第一优先级

### 4. prefill / decode 适配方向明确

probe 对 `chunk_n=1/4/8` 的结果说明：

- `N` 变大时，MAC 和输出张量大小按比例放大
- 权重字节不变
- 这正适合做：
  - `decode: N=1`
  - `prefill: N=4 or 8`

也就是说：

- 软件必须显式把 prefill 和 decode 分开调度
- 否则现有 `32x4` 阵列在 decode 时天然只有 1 列忙

## 当前最值得推进的软件适配

### V1 主线

1. 软件契约统一成：
   - `W per-row`
   - `A per-token dynamic`
2. CPU 只提交粗粒度 op：
   - `LINEAR_ATTN`
   - `LINEAR_FFN13`
   - `LINEAR_FFN2`
   - `LINEAR_CLS`
3. 首批建议下放：
   - `wq/wk/wv/wo`
   - 可选 `w1/w3`
4. 暂缓：
   - `w2`
   - `wcls`
   - `RMSNorm`
   - attention `softmax/AV`

### 对硬件接口的含义

- `LINEAR` 类任务优先稳定
- 输出格式优先探索：
  - `ACC32_RAW`
  - `MID16`
- `softmax / norm` 不应再只是 `POST_CFG` bit 开关

## 总结

本轮先验探索支持以下判断：

1. `粗粒度 op 提交` 方向可行
2. `attention projections` 是最适合先交给硬件的 V1 路线
3. `MID16` 中间格式值得继续探索
4. `RMSNorm/softmax` 不能在当前数学定义下直接进入主线
5. 软件必须分出 `prefill` 和 `decode` 两条路径
