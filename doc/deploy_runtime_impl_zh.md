# 独立部署运行时实施记录

本文档记录 `deploy/` 新运行时骨架的落地过程、当前状态和结论。

## 1. 背景

当前任务不再以继续大规模训练为主，而是把已经得到的模型结果转成：

- 可部署的软件框架
- 可验证硬件正确性的验证框架

因此本轮新建独立运行时，而不继续沿用：

- `runq.c`
- `runq_embedded.c`

这种历史路径。

## 2. 本轮新增内容

本轮已经新增：

- `deploy/include/runtime_types.h`
- `deploy/include/runtime_common.h`
- `deploy/include/runtime_assets.h`
- `deploy/include/runtime_backend.h`
- `deploy/include/runtime_decode_cfg.h`
- `deploy/include/runtime_frontend.h`
- `deploy/include/runtime_verify.h`
- `deploy/src/runtime_common.c`
- `deploy/src/runtime_assets.c`
- `deploy/src/runtime_backend_swref.c`
- `deploy/src/runtime_backend_hwstub.c`
- `deploy/src/runtime_frontend.c`
- `deploy/src/runtime_verify.c`
- `deploy/src/runtime_hw_adapter.c`
- `deploy/src/runq_deploy.c`
- `deploy/src/runq_verify.c`
- `deploy/assets/stories260K_qat_best/stories_data.h`
- `deploy/assets/stories260K_qat_best/tok512.h`
- `deploy/README_zh.md`
- `tools/export_deploy_headers.py`

同时更新：

- `Makefile`
- `build_msvc.bat`

## 3. 当前架构结论

当前新运行时已经形成两条主线：

1. 部署线
   - `runq_deploy_cpu`
   - `runq_deploy_hw`
2. 验证线
   - `runq_verify`

其中：

- `CPU` 版本负责先跑通部署逻辑
- `HW` 版本负责保留后续硬件接入位置
- `VERIFY` 负责对比 `SW_REF` 与 `HW_STUB`

## 4. 当前结果

### 4.1 已完成

- 已完成默认头文件资产解析
- 已完成 `SW_REF` 后端最小实现
- 已完成 `HW_STUB` 后端最小实现
- 已完成 `generate/chat` 前端骨架
- 已完成基本 verify 骨架
- 已完成构建入口
- 已完成 `xxd` 风格部署资产导出脚本

本轮已经主机侧验证通过：

- `runq_deploy_cpu.exe` 可编译
- `runq_deploy_hw.exe` 可编译
- `runq_verify.exe` 可编译
- `runq_verify` 已通过：
  - `rmsnorm`
  - `linear_qkv_q`
  - `qk_matmul`
  - `softmax_row`
  - `gate_mul`
  - `residual_add`
- `export_deploy_headers.py` 已成功导出：
  - `deploy/assets/stories260K_qat_best/stories_data.h`
  - `deploy/assets/stories260K_qat_best/tok512.h`
- 已完成 `runq_deploy_cpu` 一轮完整主机侧回归，结果文件：
  - `artifacts/runq_deploy_cpu_regression_20260330_v2.json`
- 已完成修复后的定向回归，结果文件：
  - `artifacts/runq_deploy_cpu_fix_regression_20260330_v2.json`
- 已完成 arena 化后的回归，结果文件：
  - `artifacts/runq_deploy_arena_regression_20260331.json`

### 4.2 当前仍待继续

- 真实 `HW` backend
- 共享 RAM 的正式静态区/arena 落位
- linker section 规划文件
- 当前最佳 QAT 模型资产正式导出
- 对 `runq_deploy_cpu/hw` 做更完整的行为回归

## 5. 当前采用的工程策略

### 5.1 资产策略

当前已经不再只是“兼容引入”：

- `export_deploy_headers.py` 已能直接生成 OpenLA500 风格的 `xxd -i` 资产头文件
- 第一版导出链已经打通到：
  - `stories260K_q80.bin`
  - `tok512.bin`

需要说明的是：

- 当前 `deploy/assets/stories260K_qat_best/` 里的资产已经是正式 `xxd` 风格头文件
- 但它们当前仍然对应 `q80` 兼容模型资产
- 还不是“当前最佳 QAT checkpoint 的正式部署导出结果”

### 5.2 内存策略

当前代码阶段先允许一部分初始化时动态分配，以便更快把框架跑起来。

但目标不变：

- 后续收敛为“静态资产 + arena + 共享 RAM”
- 避免运行时持续零散 `malloc/calloc`

本轮进一步推进后：

- `RunState` 主工作区已经切换为固定 `arena`
- 部署主路径不再依赖一组零散 `calloc/free`
- 这一步更贴近后续 SoC 上“静态资产 + arena + 共享 RAM”的实际部署形态

### 5.3 验证策略

当前验证不是对 `runq.c` 做完全行为兼容，而是：

- 用新 `SW_REF` 作为金标准
- `HW_STUB` 先保证调用链与接口一致
- 后续再逐步替换成真实硬件实现

### 5.4 当前回归结论

`runq_deploy_cpu` 当前主机侧回归结论如下：

- `build`
  - 通过
- `generate`
  - 对固定 8 个 prompt 均可正常运行并输出文本
- `verify`
  - 通过
- `chat`
  - 已恢复为与 OpenLA500 `runq` 更接近的“安全多轮”语义
  - 首轮支持 `system prompt` + `CLI user prompt`
  - 后续轮次继续从串口/标准输入读取 `User:`
  - 当前主循环的 token feed 顺序仍保持正确：先确定当前输入 token，再执行 `forward`
  - 新增“剩余上下文不足”检测，避免多轮对话把上下文窗口静默写爆
  - 但当前 `260K TinyStories` 模型对 `[INST] ... [/INST]` 这类聊天模板响应较弱
  - 因此当前更适合作为：
    - 运行时流程验证
    - 串口交互路径验证
  - 暂不适合作为质量展示主模式

### 5.5 本轮已修复问题

本轮针对运行时骨架中已经暴露出的高优先级问题，完成了以下修复：

1. `history_tokens` 缓冲区越界风险
- `encode_text()` 已增加容量参数
- `generate/chat` 编码阶段现在会检查 token 缓冲区容量
- 超长 prompt 会明确报错，不再静默写爆

2. `chat` 主循环 token feed 顺序错误
- 已改成安全多轮实现
- 现在和 `generate` 一样：
  - 先确定当前输入 token
  - 再执行 `forward`
  - prompt token 只用于静默注入，不再错序进入 `next` 逻辑
  - assistant 结束 token 不再继续送入模型，而是直接切回下一轮 `User:`

3. `group_size` 假设过强
- 激活量化路径不再要求 `n % group_size == 0`
- 现在支持最后一个不满的 group
- 这与当前 `260K` 模型 `hidden_dim=172` 的情况对齐

4. 资产解析校验不足
- 已增加模型 header 大小与权重布局大小检查
- 已增加权重 tensor 和 `group_size` 的整除关系检查
- 已增加 tokenizer 基本边界检查

5. `-s` 参数文档/实现不一致
- 现在 `-s` 已真正接入
- 可用于固定随机种子，方便可复现实验

6. `HW_STUB` 不经过 adapter
- `runtime_backend_hwstub.c` 已显式调用 `runtime_hw_adapter`
- 现在 `runq_deploy_hw` / `runq_verify` 会输出共享缓冲和地址口径 trace

7. `RunState` 仍然依赖大量零散 heap
- 当前已改成固定 `arena`
- 运行时主工作区不再由多次 `calloc/free` 维护
- 这降低了后续接硬件前的内存布局漂移风险

8. `xq.s / hq.s` 的 scale buffer 建模不准确
- 现在已经统一改成按 `grouped_scale_count()` 计算
- 不再错误地按“每元素一个 scale”分配 arena 空间
- 这使得运行时内存布局和 q80/group quant 数学保持一致

修复后的定向回归结果表明：

- 长 prompt 保护：通过
- seed 可复现：通过
- verify：通过
- `runq_deploy_hw` 走 adapter trace：通过

## 6. 下一步

下一步建议继续按这个顺序推进：

1. 补 `ld/` 下 section 规划
2. 明确共享 RAM / uncached alias 地址组织
3. 把共享 RAM / uncached alias 真正落进代码与 section 规划
4. 让 `runq_verify` 输出更明确的误差摘要与更多算子覆盖
5. 收敛 `runq_deploy_cpu` 的默认展示行为
6. 再开始接真正的 `HW` backend
