# `260K` 独立部署运行时与验证运行时完整实施计划（更新版：双地址空间口径）

## 总结
围绕当前最佳 `260K` 模型结果，后续主线不再是继续大规模训练，而是落地一套：

- 可在 LoongArch/OpenLA500 类 SoC 上编译运行
- 可直接从 `.h` 头文件资产加载模型
- 可先跑 `CPU/SW_REF`，后续平滑切到 `HW`
- 可用于后续硬件 IP 正确性验证

的软件框架。

整体方案固定为：

- 主部署程序：`runq_deploy`
  - 一份源码
  - 编译期宏选择 `CPU` 或 `HW`
  - 生成：
    - `runq_deploy_cpu`
    - `runq_deploy_hw`
- 验证程序：`runq_verify`
  - 一个可执行文件
  - 内部固定执行：
    - `SW_REF`
    - `HW_STUB`
    - compare
- 模型/tokenizer 资产：
  - 统一导出为 OpenLA500 当前 `runq` 使用的 `xxd -i` 风格头文件
- 内存布局：
  - 必须同时区分
    - `物理/总线地址`
    - `CPU uncached alias 地址`
  - 共享 RAM 规划以后者给 CPU 用，前者给 DMA/MMIO 编程用
- 一致性策略：
  - 采用你明确给出的 LoongArch 无 cache 方案
  - 第一版不引入 flush/invalidate

---

## 代码结构与职责

```text
9th/ref/llama2.c/
├─ deploy/
│  ├─ include/
│  │  ├─ runtime_types.h
│  │  ├─ runtime_backend.h
│  │  ├─ runtime_frontend.h
│  │  ├─ runtime_assets.h
│  │  ├─ runtime_decode_cfg.h
│  │  └─ runtime_verify.h
│  ├─ src/
│  │  ├─ runq_deploy.c
│  │  ├─ runq_verify.c
│  │  ├─ runtime_frontend.c
│  │  ├─ runtime_assets.c
│  │  ├─ runtime_backend_swref.c
│  │  ├─ runtime_backend_hwstub.c
│  │  ├─ runtime_hw_adapter.c
│  │  ├─ runtime_verify.c
│  │  └─ runtime_common.c
│  ├─ assets/
│  │  └─ stories260K_qat_best/
│  │     ├─ stories_data.h
│  │     └─ tok512.h
│  ├─ ld/
│  │  └─ deploy_sections.ldh
│  ├─ tests/
│  │  ├─ test_verify_cases.json
│  │  └─ test_demo_prompts.json
│  └─ README_zh.md
├─ tools/
│  ├─ export_deploy_headers.py
│  └─ run_demo_showcase.py
└─ artifacts/
```

### 文件职责
- `runq_deploy.c`
  - 用户最终运行入口
  - 兼容原版 `runq` CLI 风格
- `runq_verify.c`
  - 固定验证入口，做 `SW_REF -> HW_STUB -> compare`
- `runtime_frontend.*`
  - tokenizer、prompt、prefill/decode 主循环、KV cache、generate/chat
- `runtime_backend.h`
  - 算子级语义接口
- `runtime_backend_swref.c`
  - 当前执行图的软件参考实现
- `runtime_backend_hwstub.c`
  - 硬件后端桩
- `runtime_hw_adapter.c`
  - 把语义接口翻译成 SoC MMIO/DMA 作业提交
- `runtime_assets.*`
  - 从 `stories_data.h` / `tok512.h` 解析模型和 tokenizer
- `runtime_decode_cfg.h`
  - 固定默认 demo 参数和 prompt 集
- `deploy_sections.ldh`
  - 追加 linker section 规划，控制模型资产 / arena / 共享 RAM 的放置

---

## 运行时与接口设计

## 1. 主部署程序 `runq_deploy`
- 源码：`deploy/src/runq_deploy.c`
- 编译宏：
  - `RUNQ_DEPLOY_CPU`
  - `RUNQ_DEPLOY_HW`
- 产物：
  - `runq_deploy_cpu`
  - `runq_deploy_hw`

### CLI 风格
第一版尽量兼容原版 `runq` / OpenLA500 `runq`：
- `-n`
- `-i`
- `-t`
- `-p`
- `-s`
- `-m`
- `-y`

不提供运行时 `--backend=cpu|hw`。

### 语义
- `runq_deploy_cpu`
  - 当前执行图 `SW_REF`
- `runq_deploy_hw`
  - 后续真实硬件后端
  - 第一版允许先接 `HW_STUB`

### 模式
第一版同时支持：
- `generate`
- `chat`

测试优先级：
1. `generate`
2. `chat`

---

## 2. 验证程序 `runq_verify`
- 源码：`deploy/src/runq_verify.c`
- 固定行为：
  1. 加载同一份 `.h` 资产
  2. 跑 `SW_REF`
  3. 跑 `HW_STUB`
  4. compare
  5. 输出 PASS/FAIL 和误差摘要

不提供用户侧模式切换。

---

## 3. backend API
文件：`deploy/include/runtime_backend.h`

冻结为算子级语义接口：
- `backend_init`
- `backend_reset`
- `backend_rmsnorm`
- `backend_linear_qkv`
- `backend_linear_attn_o`
- `backend_qk_matmul`
- `backend_softmax_row`
- `backend_av_matmul`
- `backend_linear_ffn_w1`
- `backend_linear_ffn_w3`
- `backend_gate_mul`
- `backend_linear_ffn_w2`
- `backend_residual_add`
- `backend_final_norm`
- `backend_lm_head`
- `backend_destroy`

约束：
- `runq_deploy` 内部调用这些接口，但不暴露算子细节给用户
- `runq_verify` 围绕这些接口做算子级验证
- `backend_lm_head` 第一版允许 software-only

---

## 4. `runtime_hw_adapter` 设计
这是关键更新项，必须存在。

文件：`deploy/src/runtime_hw_adapter.c`

职责：
- 不让 backend API 直接绑定 MMIO/DMA
- 将语义级调用翻译为：
  - MMIO 配置
  - DMA load/store
  - post/gemm 启动
  - done/status 等待
  - output bank/owner 协调

### Adapter 内部接口建议
- `hw_prepare_linear_job(...)`
- `hw_prepare_post_job(...)`
- `hw_dma_load(...)`
- `hw_dma_store(...)`
- `hw_submit_job(...)`
- `hw_wait_done(...)`
- `hw_soft_reset(...)`

### 必须区分的地址参数
每个共享 buffer 必须同时保留：
- `cpu_uncached_addr`
- `phys_bus_addr`

绝不能只保留一套地址。

---

## 资产导出链

## 1. 资产格式
新增：
- `tools/export_deploy_headers.py`

输出格式直接对齐 OpenLA500 当前 `runq` 使用的头文件：
- `xxd -i` 风格数组头文件

即类似：
```c
unsigned char stories260K_q80_bin[] = { ... };
unsigned char tok512_bin[] = { ... };
```

## 2. 目录与命名
固定目录：
- `deploy/assets/stories260K_qat_best/`

固定文件名：
- `stories_data.h`
- `tok512.h`

规则：
- 文件名固定
- 模型版本只靠目录 tag 区分
- 新 runtime 只认这两个固定名字

---

## SoC 地址空间与双地址口径（修正版）

## 1. 设备窗口地址（RTL / 总线口径）
这些是 SoC 译码与 DMA 编程应使用的物理/总线地址：

- RAM：`0x1c000000 ~ 0x1c7fffff`
- UART：`0x1f000000`
- ACC MMIO：`0x1f100000`
- CONFREG：`0x1f200000`

## 2. CPU uncached alias 地址（软件访问口径）
这些是 CPU 在无 cache 方案下实际访问设备/共享 RAM 时应使用的地址别名：

- RAM uncached alias：`0xbc000000 ~ 0xbc7fffff`
- UART uncached alias：`0xbf000000`
- ACC MMIO uncached alias：`0xbf100000`
- CONFREG uncached alias：`0xbf200000`

说明：
- 设备窗口地址给 DMA/MMIO 编程使用
- uncached alias 给 CPU 指针访问使用
- 两套地址必须同时存在于 runtime 设计里

## 3. 软件可执行镜像内存布局（链接脚本口径）
当前链接脚本真实基址：

- `isram`：`0x1c000000 ~ 0x1c07ffff`（512KB）
- `dsram`：`0x1c080000 ~ 0x1c3fffff`（3584KB）

第一版不改这两个基址定义，只在 `dsram` 内细分部署运行时布局。

---

## `dsram` 内详细分配（第一版冻结）

`dsram` 总范围：
- `0x1c080000 ~ 0x1c3fffff`

### A. 普通 `.data/.bss`
- 物理：`0x1c080000 ~ 0x1c0fffff`
- CPU uncached alias：`0xbc080000 ~ 0xbc0fffff`
- 大小：`512KB`
- 用途：
  - 常规 `.data/.bss`
  - 小型全局对象
  - 普通状态和配置

### B. 模型资产区 `MODEL_RO`
- 物理：`0x1c100000 ~ 0x1c17ffff`
- CPU uncached alias：`0xbc100000 ~ 0xbc17ffff`
- 大小：`512KB`
- 用途：
  - `stories_data.h`
  - `tok512.h`
- 说明：
  - 模型头文件资产固定放这里
  - 通过 `.model_assets` section 控制

### C. CPU 工作区 `RUNTIME_ARENA`
- 物理：`0x1c180000 ~ 0x1c27ffff`
- CPU uncached alias：`0xbc180000 ~ 0xbc27ffff`
- 大小：`1MB`
- 用途：
  - `RunState`
  - `KV cache`
  - `att/logits`
  - prompt token buffer
  - sampler scratch
- 说明：
  - 一次性初始化
  - 推理过程中不再大量 `malloc/calloc`

### D. 共享 RAM 区 `ACCEL_SHARED`
- 物理：`0x1c280000 ~ 0x1c37ffff`
- CPU uncached alias：`0xbc280000 ~ 0xbc37ffff`
- 大小：`1MB`
- 用途：
  - accelerator DMA 共享输入/输出区
  - 硬件后端 staging buffer
  - 中间结果交换区
  - 验证版 `HW_STUB` 共享区
- 说明：
  - 第一版就固定边界
  - CPU 必须用 `0xbc...` 访问
  - DMA/MMIO 必须用 `0x1c...` 编程

### E. 栈与安全余量
- 物理：`0x1c380000 ~ 0x1c3fffff`
- CPU uncached alias：不作为普通数据区使用
- 规划：
  - 保留现有 `64KB stack`
  - 其余作为系统余量，不分配给 runtime 主工作区

---

## `ACCEL_SHARED` 内部细分（第一版）
### 物理地址视图
- `0x1c280000 ~ 0x1c29ffff`
  - `ACCEL_IO_IN`
- `0x1c2a0000 ~ 0x1c2bffff`
  - `ACCEL_IO_OUT`
- `0x1c2c0000 ~ 0x1c2fffff`
  - `ACCEL_PARAM_STAGE`
- `0x1c300000 ~ 0x1c33ffff`
  - `ACCEL_KV_SHARED`
- `0x1c340000 ~ 0x1c36ffff`
  - `ACCEL_SCRATCH`
- `0x1c370000 ~ 0x1c37ffff`
  - `ACCEL_TRACE`

### CPU 访问视图
对应 uncached alias：
- `0xbc280000 ~ 0xbc29ffff`
- `0xbc2a0000 ~ 0xbc2bffff`
- `0xbc2c0000 ~ 0xbc2fffff`
- `0xbc300000 ~ 0xbc33ffff`
- `0xbc340000 ~ 0xbc36ffff`
- `0xbc370000 ~ 0xbc37ffff`

---

## linker section 规划
新增：
- `deploy/ld/deploy_sections.ldh`

需要控制的 section：
- `.model_assets`
  - 放到 `MODEL_RO`
- `.runtime_arena`
  - 放到 `RUNTIME_ARENA`
- `.accel_shared`
  - 放到 `ACCEL_SHARED`
- `.accel_trace`
  - 放到 `ACCEL_TRACE`

第一版原则：
- 不依赖默认 heap 承载主要推理工作区
- heap 只留给少量兼容性分配
- 部署版主推理路径禁止继续沿用 OpenLA500 当前大量零散 `malloc/calloc`

---

## 串口与输入输出
### 串口口径
- RTL / 设备窗口：`0x1f000000`
- 软件访问别名：`0xbf000000`

### 策略
第一版继续保留 OpenLA500 当前风格：
- `printf/puts`
- `fgets(stdin, ...)`

因此：
- `generate` 与 `chat` 均支持
- 但测试优先级仍为：
  1. `generate`
  2. `chat`

---

## 默认模型与展示参数
### 默认模型
固定为当前最佳主版本：
- `seq256 + final_polish_strong`
- 当前 deploy 默认对齐的 QAT best 资产目录：
  - `deploy/assets/qat_best_compare_finalpolishstrong_seq256/`
- `deploy/assets/stories260K_qat_best/` 保留为官方导出基线口径，不作为当前 deploy 状态文档的默认 QAT best。

### 默认展示参数
固定：
- `temperature=0.93`
- `top_p=0.9`
- `top_k=40`
- `repetition_penalty=1.05`
- `no_repeat_ngram_size=3`

### 默认 prompt
固定两组：
- 稳妥组
- 故事组

### 演示长度
- `120~160` 新 token

---

## 构建与验收
### 构建目标
新增：
- `runq_deploy_cpu`
- `runq_deploy_hw`
- `runq_verify`
- `export_deploy_assets`

### 验收要求
`runq_deploy_cpu`
- 能直接从 `.h` 资产启动
- 不依赖 `.bin`
- `generate/chat` 可运行
- 共享 RAM 相关 buffer 使用 uncached alias

`runq_deploy_hw`
- 能编译通过
- 走 `runtime_hw_adapter`
- 第一版允许后端为 `HW_STUB`
- DMA/MMIO 编程统一使用物理/总线地址

`runq_verify`
- 固定执行 `SW_REF -> HW_STUB -> compare`
- 当前对齐后的最小覆盖包括：
  - `rmsnorm`
  - `linear_qkv_q/k/v`
  - `av_matmul`
  - `qk_matmul`
  - `softmax_row`
  - `gate_mul`
  - `residual_add`
  - `kv_main_map`

---

## 默认假设
- 当前仅围绕 `260K` 模型实施
- DMA 必须使用共享 RAM
- CPU 访问共享 RAM 使用 uncached alias
- DMA/MMIO 编程使用物理/总线地址
- 一致性采用无 cache 方案，不在第一版加入 flush/invalidate
- `runq.c` 与 `runq_embedded.c` 保持不动
- 第一版重点冻结：
  - 代码结构
  - 资产格式
  - 双地址口径
  - `dsram` 内运行时布局
  - `hw adapter` 分层
