# 部署运行时说明

本文档说明 `deploy/` 目录下新运行时骨架的用途、当前状态和基本用法。

## 目标

当前新增运行时不是要替换 `runq.c` 的数学实现，而是先搭出一套：

- 独立于 `runq.c` 的部署入口
- 可切 CPU/HW 编译目标
- 可从 `.h` 头文件资产加载模型
- 可用于后续硬件验证的统一框架

当前保留两条代码线：

1. `runq_deploy`
   - 面向实际部署
   - 通过编译宏生成：
     - `runq_deploy_cpu`
     - `runq_deploy_hw`
2. `runq_verify`
   - 面向硬件验证
   - 固定执行 `SW_REF -> HW_STUB -> compare`

## 目录说明

- `include/`
  - 统一接口和公共类型
- `src/`
  - 前端、后端、验证和主入口实现
- `assets/`
  - 部署使用的头文件资产
- `tests/`
  - 后续固定 prompt / verify case 的位置
- `ld/`
  - 预留给后续 linker section 规划

## 当前资产状态

当前正式资产可以直接由下面脚本导出：

```powershell
python tools/export_deploy_headers.py --model-bin artifacts/stories260K/stories260K_q80.bin --tokenizer-bin artifacts/stories260K/tok512.bin
```

导出目标固定为：

- `deploy/assets/stories260K_qat_best/stories_data.h`
- `deploy/assets/stories260K_qat_best/tok512.h`

## 构建方法

在仓库根目录执行：

```powershell
make runqdeploycpu
make runqdeployhw
make runqverify
```

Windows/MSVC 下也已经补了：

```powershell
build_msvc.bat
```

当前已经主机侧验证通过：

- `runq_deploy_cpu.exe`
- `runq_deploy_hw.exe`
- `runq_verify.exe`
- `artifacts/runq_deploy_cpu_regression_20260330_v2.json`
- `artifacts/runq_deploy_cpu_fix_regression_20260330_v2.json`
- `artifacts/runq_deploy_arena_regression_20260331.json`

## 运行方法

### CPU 部署版

```powershell
.\runq_deploy_cpu -n 140 -i "Once upon a time"
```

### HW 部署版

```powershell
.\runq_deploy_hw -n 140 -i "Once upon a time"
```

当前 `HW` 版第一阶段仍走 `HW_STUB` 路径，目的是先把部署接口和共享内存边界固定下来。

### 验证版

```powershell
.\runq_verify
```

它会固定执行：

1. `SW_REF`
2. `HW_STUB`
3. 对比关键算子输出

当前已经通过的对拍项有：

- `rmsnorm`
- `linear_qkv_q`
- `qk_matmul`
- `softmax_row`
- `gate_mul`
- `residual_add`

另外，`runq_deploy_cpu` 已经完成一轮主机侧完整回归：

- `build`：通过
- `generate`：通过
- `verify`：通过
- `chat`：当前已恢复为接近 OpenLA500 的“安全多轮”语义，支持首轮 `system prompt`、首轮 CLI `user prompt` 和后续 `User:` 继续输入，但 `260K TinyStories` 模型对聊天模板响应仍然偏弱

本轮还额外修复并验证通过：

- 超长 prompt 会被安全拦截，不再写爆 token 缓冲区
- `-s` seed 已真正生效，可复现同一输出
- `runq_deploy_hw` 已经会经过 `runtime_hw_adapter` 输出共享缓冲与地址 trace
- `RunState` 主工作区已切换为固定 `arena`
- `xq.s / hq.s` 的 scale buffer 已改成按 group 数建模，而不是按元素数建模

## 当前默认展示口径

当前内部固定的默认 decode 参数是：

- `temperature = 0.93`
- `top_p = 0.9`
- `top_k = 40`
- `repetition_penalty = 1.05`
- `no_repeat_ngram_size = 3`

## 当前实现边界

当前阶段已经实现：

- 默认资产加载
- `SW_REF` backend
- `HW_STUB` backend
- `generate/chat` 前端骨架
- 算子级 verify 骨架

其中 `chat` 当前的具体边界为：

- 行为上对齐 OpenLA500 的多轮演示风格
- 安全性上增加了 token 容量检查与“剩余上下文不足”提示
- 还不是完整 instruct/chat 产品形态，主要用于串口交互和运行时流程演示

当前阶段尚未完成：

- 真实 `HW` backend
- linker section 的正式落位
- 当前最佳 QAT 权重到部署头文件资产的正式导出
- 共享 RAM / uncached alias / phys 的正式 section 落位

## 说明

`runq.c` 与 `runq_embedded.c` 当前保持不动，只作为历史基线参考。
