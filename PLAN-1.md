# 更新计划：先固定软件侧内存/对象模型，再通过实验收敛量化 KV 与 Queue v0 合同

## 1. 计划目标
本轮计划只面向**软件侧实现**，目标固定为以下四条：

1. 让完整历史 `KV cache` 直接驻留在可与硬件协同的固定主存区域，减少 CPU 侧中转搬运。
2. 让软件下发接口统一收敛到 `region + byte offset`，为后续硬件 primitive / queue / doorbell 下发提供稳定软件合同。
3. 让 `runq_deploy_hw` 成为唯一正式部署路径，`runq_deploy_cpu` 只保留为 reference/debug artifact。
4. 让 `QK / AV` 具备进入 `int8 GEMM` 路径的条件；必要时允许通过 `PTQ / QAT / 微调` 做数值收敛。

本计划中的内容默认都是后续实现的**确定性指导**。只有 `KV scale` 粒度这一项明确列为“先实验探索，再回填冻结”的例外项，其他部分不允许在实现阶段自行放宽、重解释或引入并行语义分支。

---

## 2. 术语与范围
### 2.1 术语固定
从本计划开始，禁止继续使用含糊的 `host` 一词。统一改为：

- `soc_cpu`：指 SoC 上的 CPU 端。
- `intel_host_stub`：指当前 Intel 主机上的先验验证环境。

后续代码、注释、测试说明、日志输出都必须使用上述术语，不允许混用 `host`。

### 2.2 本轮范围
本轮包含以下内容：

- 运行时内存分配与对象模型收敛
- `KV_MAIN` 固定窗口与 region 描述
- 量化 `KV cache` 的软件布局与实验收敛路径
- queue v0 的软件侧 primitive / 编码 / ring 生命周期合同
- `runtime_hw_adapter` 的软件侧 lowering / submit / completion 聚合接口
- `runtime_backend_hwstub` 的 queue-aware 验证路径
- `soc_cpu` 访问策略的 uncached 基线路径，以及必要时的 cached + flush/invalidate 优化

### 2.3 本轮非范围
以下内容不在本轮冻结范围内：

- 硬件 ISA 最终编码细节
- SoC linker / section policy / 真实物理落位集成
- RTL 必须落地的最终 doorbell 细节

`runtime_memory_map.h` 在本轮只代表**软件侧固定地址合同**。本轮不实现 SoC integration，不要求 `intel_host_stub` 真实映射到这些物理地址。

---

## 3. 软件侧总原则
### 3.1 主数据面原则
完整历史 `KV cache` 从 `RUNTIME_ARENA` 中移出，并固定到 `KV_MAIN`。从本轮开始：

- 完整历史 `KV cache` 只允许驻留在 `KV_MAIN`。
- `ACCEL_SHARED` 不再承担完整历史 `KV cache` 的正式数据面职责。
- 任何需要访问完整历史 `KV cache` 的 queue / DMA / lowering 路径，必须基于 `KV_MAIN region + byte offset`。

### 3.2 接口分层原则
从本轮开始，软件接口分成两层，且两层职责固定：

1. **CPU 算法视图**
   - 不再以完整历史 `KV cache` 的裸 `float *` 作为正式长期接口
   - 若 CPU 算法层需要访问 `KV cache`，必须通过 `kv view / accessor helper` 访问逻辑上的 `(layer, time, head, channel)` 视图
   - CPU 算法视图允许保留“按 token/head/channel 访问”的语义，但不允许继续把完整历史 `KV cache` 暴露成正式的裸指针接口

2. **部署/下发视图**
   - 使用 `KEY_CACHE_MAIN_REGION / VALUE_CACHE_MAIN_REGION` 以及其下的 data/scale 二级子区
   - 只允许使用 `region + byte offset`

禁止把这两层接口混用。尤其禁止在 builder / DMA lowering / queue encoder / backend / verify 中长期直接消费裸 `float *key_cache/value_cache`。

### 3.3 单一正式路径原则
从本轮开始：

- `runq_deploy_hw` 是唯一正式 deploy path。
- `runq_deploy_cpu` 只允许用于 reference/debug。
- `SW_REF` 只允许存在于 verify / gold path 中，不允许进入正式 `HW` 执行路径。

---

## 4. 内存与对象模型收敛
### 4.1 固定窗口定义
软件侧固定 `KV_MAIN_WINDOW`、一级窗口和二级子区：

- `KV_MAIN_WINDOW`
  - phys: `0x1c410000 ~ 0x1c4affff`
  - cpu uncached alias: `0xbc410000 ~ 0xbc4affff`
  - size: `640KiB`

- `KEY_CACHE_MAIN_REGION`
  - phys: `0x1c410000 ~ 0x1c45ffff`
  - cpu uncached alias: `0xbc410000 ~ 0xbc45ffff`
  - size: `320KiB`

- `VALUE_CACHE_MAIN_REGION`
  - phys: `0x1c460000 ~ 0x1c4affff`
  - cpu uncached alias: `0xbc460000 ~ 0xbc4affff`
  - size: `320KiB`

上述一级窗口地址、边界、容量在本轮冻结，不允许改动。

### 4.2 region 对象命名固定
`KV_MAIN` 的一级软件框架从本轮开始固定为“一级窗口 + data/scale 二级子区”。

一级窗口固定为：

- `KEY_CACHE_MAIN_REGION`
- `VALUE_CACHE_MAIN_REGION`

对应的二级子区 region 描述对象名称固定为：

- `key_cache_main_data_region`
- `key_cache_main_scale_region`
- `value_cache_main_data_region`
- `value_cache_main_scale_region`

其中每个一级窗口都必须由对应的 `*_data_region` 与 `*_scale_region` 组成。一级窗口名称用于容量与地址口径，二级子区名称用于实现、helper、builder、queue 与 offset 组织。

其中：

- `*_data_region` 承载量化后的 `int8` 数据区
- `*_scale_region` 承载对应的 scale metadata 区

一级窗口与二级子区的命名口径必须严格区分，不允许混用。一级窗口不等于 data 子区，也不等于 scale 子区。

region 描述对象至少包含以下字段：

- `cpu_ptr`
- `cpu_uncached_addr`
- `phys_addr`
- `size`

本轮对二级子区只冻结“必须存在 data/scale 分区”这一层框架；`*_scale_region` 的精确边界、容量与内部 metadata 布局不在本轮冻结，待实验结论形成后再回填固定。

如果后续需要扩展字段，只允许追加，不允许改变上述字段含义。
如果后续需要扩展字段，只允许追加，不允许改变上述字段含义。

### 4.3 backend 接口与 CPU 侧 KV 访问接口的确定性要求
从本轮开始，`runtime_backend.h`、`runtime_common.c` 与相关运行时对象必须满足以下要求：

- `runtime_backend.h` 中的 `qk_matmul / av_matmul` 必须优先完成接口改造，不允许继续把完整历史 `KV cache` 作为裸 `float *key_cache/value_cache` 传入 backend 合同。
- 完整历史 `KV cache` 不再作为正式的裸 `float *key_cache/value_cache` 接口长期保留。
- 任何 CPU 侧对完整历史 `KV cache` 的访问，都必须通过 `kv view / accessor helper` 完成。
- `kv view / accessor helper` 的底层数据来源必须绑定到 `key_cache_main_data_region / value_cache_main_data_region` 以及对应的 `*_scale_region`。
- builder / backend / verify 不允许再长期接受裸 `float *key_cache/value_cache` 作为正式接口。
- 若为了过渡需要短期保留旧字段，其语义也不得再作为正式合同对外扩散。
### 4.4 `RUNTIME_ARENA` 保留范围固定
`RUNTIME_ARENA` 只保留 `soc_cpu` / `intel_host_stub` 当前 token 工作集，固定包含：

- `x`
- `xb`
- `xb2`
- `hb`
- `hb2`
- `xq`
- `hq`
- `q`
- `att`
- `logits`

### 4.5 当前 token 的局部 `k/v` 处理规则冻结
从本轮开始，当前 token 的局部 `k/v` 不再作为正式长期对象保留在 `RUNTIME_ARENA` 中。

固定要求如下：

- 当前 token 的 `k/v` 只允许作为算子执行过程中的短生命周期中间结果存在。
- `runtime_common.c` 中“先写局部 `k/v`，再 `memcpy` 到完整历史 `KV cache`”的正式实现路径必须移除。
- 一旦完成当前 token 的 `k/v` 计算，必须直接写入或量化写入 `KV_MAIN` 的最终槽位。
- 不允许形成“局部 `k/v` 长期驻留 arena，再额外搬运到 `KV_MAIN`”的正式实现路径。

本条的目标是减少 CPU 侧额外中转搬运，并让当前 token 的 `k/v` 生成路径与最终历史布局保持一致。

### 4.6 `ACCEL_SHARED` 角色固定
`ACCEL_SHARED` 只允许承载以下内容：

- staged input/output
- staged params
- scratch
- debug
- trace
- bring-up fallback 所需的小块临时区

从本轮开始，`ACCEL_SHARED` 禁止承载：

- 完整历史 `KV cache`
- 完整历史 `KV cache` 的正式 staging 副本
- 任何替代 `KV_MAIN` 的主数据面语义

`ACCEL_KV_SHARED` 语义废弃，不允许继续作为完整历史 `KV cache` 或正式主数据面使用。

---

## 5. 量化 KV 路线固定
### 5.1 `KV_MAIN` 数据路线固定
本轮 `KV_MAIN` 的目标路线固定为：

- `KV_MAIN = int8 data + scale metadata`

也就是说，本轮不再把 `KV_MAIN` 冻结为纯 `float32 KV` 路线，而是按面向 `QK / AV int8 GEMM` 的量化 KV 路线推进。

### 5.2 已冻结项
以下项目在本轮冻结：

- 完整历史 `KV cache` 的主存归属固定为 `KV_MAIN`
- `QK / AV` 必须面向量化 KV 路线设计
- 访问 `KV_MAIN` 的下发接口必须统一为 `region + byte offset`
- 当前 token 的局部 `k/v` 必须直接写入或量化写入 `KV_MAIN` 最终槽位
- `KV_MAIN` 的一级软件框架固定为“data 区和 scale 区分离”
- builder / helper / queue / primitive 必须在该分区框架下组织地址与 offset

### 5.3 待实验探索项
`KV scale` 粒度在本轮**先做实验探索，不在当前版本计划中冻结**。

必须探索并对比的候选至少包括：

- 按 `kv_head` 一份 scale
- 按整行一份 scale
- 按 `group_size` 分组 scale

实验必须围绕以下目标进行：

- `QK / AV int8 GEMM` 的可实现性
- 精度退化情况
- `soc_cpu` 侧搬运与地址组织复杂度
- queue / primitive / region 组织复杂度
- 后续硬件原语组织难度

在实验结论形成之前：

- 不允许在实现中隐式固化某种 scale 粒度为最终合同
- 允许为实验性路径编写临时代码
- 但临时代码不得伪装为最终冻结接口

### 5.4 实验完成后的回填要求
一旦 scale 粒度实验有结论，必须在下一版计划中明确补齐以下内容：

- scale 粒度
- `*_scale_region` 的精确边界与容量
- `*_scale_region` 的内部 metadata 布局
- scale 的 byte offset 计算规则
- data 区与 scale 区的精确对应关系
- `QK / AV` 读取时的精确解释规则

在实验结论未形成之前，一级结构已经固定为“data 区和 scale 区分离”，不允许退回到 data/scale 行内交织方案；同时本轮不冻结 `*_scale_region` 的精确边界与容量，只冻结其必须作为独立二级子区存在。

---

## 6. KV 布局与 offset 合同
### 6.1 逻辑布局冻结
在不考虑 scale metadata 的前提下，`KV data` 的逻辑布局固定为：

- `[layer][time][kv_dim]`

其中：

- `layer` 范围：`[0, n_layers)`
- `time` 范围：`[0, seq_len)`
- `kv_dim = dim * n_kv_heads / n_heads`

该布局在 CPU 算法视图和部署/下发视图中必须完全一致，不允许各自定义不同布局。

### 6.2 数据区 byte offset 计算规则冻结
对 `KV data` 本体，逻辑位置 `(layer, time, channel)` 对应的 data byte offset 固定为：

`(((layer * seq_len) + time) * kv_dim + channel) * sizeof(int8_t)`

其中：

- `channel` 范围：`[0, kv_dim)`

任何 builder / DMA lowering / HW_STUB / 后续 RTL 对接逻辑，都必须使用这一公式来解释 **KV data 区**，不允许自行解释。

### 6.3 head 视图派生规则冻结
某个 KV head 的起始通道固定为：

- `kv_head_idx * head_size`

其中：

- `head_size = dim / n_heads`
- `kv_head_idx = head_idx / kv_mul`
- `kv_mul = n_heads / n_kv_heads`

因此，某个 head 的向量访问本质上是：

- 先按 `[layer][time][kv_dim]` 找到该时刻的 KV 行
- 再在该行内取 `[kv_head_idx * head_size, (kv_head_idx + 1) * head_size)` 这段连续通道

后续所有 `QK` / `AV` lowering 都必须沿用这套规则，不允许再引入第二套 head 布局解释。

---

## 7. Queue v0 软件合同冻结
### 7.1 Queue 组织形式固定为 ring
从本轮开始，Queue v0 固定为 ring queue，不允许再做成线性一次性队列。

固定分工如下：

- `CMDQ`：`SW producer / HW consumer`
- `CMPQ`：`HW producer / SW consumer`

### 7.2 `CMDQ / CMPQ` 语义固定
- `CMDQ` 用于承载待执行命令。软件写入 command entry，硬件读取并执行。
- `CMPQ` 用于承载完成回报。硬件写入 completion entry，软件读取并回收。

这两个队列职责不同，不允许混用。

### 7.3 生命周期管理规则冻结
queue slot 的复用通过 `head/tail` 推进实现，不通过“清空旧 entry”实现。

固定规则如下：

- `head` / `tail` 使用**单调递增计数器**
- `slot_index = counter % depth`
- 空队列条件：`tail == head`
- 满队列条件：`tail - head == depth`

旧 entry 的内容允许保留在内存中，不要求清零。slot 是否可复用只由 `head/tail` 关系决定。

### 7.4 completion 语义固定
`seq_id` 是唯一 primitive 身份标识，`trace_tag` 是唯一合法的软件观测标签，固定规则如下：

- `seq_id` 全局唯一，按提交顺序单调递增
- `CMPQ` completion 只回报 `seq_id`
- `trace_tag` 只允许用于软件日志、调试、trace 聚合与观测
- `trace_tag` 不允许进入硬件合同，不允许参与 completion 正确性判断
- `job_id` 这一术语从本轮开始废弃，统一改为 `trace_tag`

从本轮开始，completion 语义禁止依赖 CMDQ slot 位置。

### 7.5 CMDQ 回收语义固定
CMDQ 的 `head` 代表：

- 硬件已经接管该命令 entry，可复用其 slot

真正的执行完成语义不由 CMDQ `head` 表示，而由 `CMPQ` 中的 `seq_id` completion 表示。

### 7.6 opcode 落地范围固定
本轮必须落地的软件/硬件共同合同固定为：

- `DMA_LOAD`
- `DMA_STORE`
- `EXEC_GEMM`
- `EXEC_SOFTMAX`
- `EXEC_NORM`
- `CMPQ`

本轮仅保留编号与软件 helper 语义，不要求成为必须落地硬件原语的项目固定为：

- `WAIT`
- `BARRIER`
- `SET_QUANT`
- `SET_TILE`
- `SET_POST`

### 7.7 编码风格固定
Queue v0 继续采用 `major/minor` 编码，固定大类为：

- `MOVE`
- `EXEC`
- `SYNC`
- `CFG`

`word0` 固定包含以下字段：

- `opcode`
- `flags`
- `buf0`
- `buf1`
- `buf2`
- `version`

### 7.8 payload 单位固定
以下字段统一按 **byte offset** 解释：

- `local_offset`
- `src_offset`
- `dst_offset`
- `param_offset`
- `srcA_offset`
- `srcB_offset`

以下字段统一按 **byte address** 解释：

- `ext_addr_phys`

以下字段统一按 **bytes** 解释：

- `len_bytes`
- `line_size`
- `line_stride`

以下字段统一按 **logical element count** 解释：

- `M`
- `K`
- `N`
- `rows`
- `cols`
- `elem_count`

builder 必须在 enqueue 之前，把所有高层索引换算成上述固定单位，不允许把单位解释留给 backend、HW_STUB 或后续 RTL。

### 7.9 `MOVE / EXEC / SYNC / CFG` 解释规则固定
- `MOVE entry`
  - 只描述外存访问
  - 允许携带主存 phys 地址、长度、stride、qos

- `EXEC entry`
  - 只描述本地 buffer / bank 与算子配置
  - 不允许直接携带主存地址

- `WAIT`
  - 只针对 `seq_id`

- `CFG`
  - 本轮只冻结 opcode 预留编号
  - 本轮不冻结 payload 细节

---

## 8. `runtime_hw_adapter` 与 backend 改造顺序冻结
### 8.1 `runtime_hw_adapter` 的职责固定
下一步第一优先级固定为：

- `runtime_hw_adapter.h`
- `runtime_hw_adapter.c`

其职责固定为三层：

1. 语义级算子调用接入
2. primitive sequence lowering
3. queue v0 entry 编码与 submit / completion 聚合

在 adapter 合同稳定之前，不允许先改 backend 行为。

### 8.2 backend 改造顺序固定
`runtime_backend_hwstub.c` 必须在 adapter 合同稳定后再改造成 queue-aware stub，职责固定为：

- 验证 lowering 输出是否正确
- 验证 enqueue / submit / completion 聚合是否正确
- 保留数学 fallback 仅作为内部对拍工具

`runtime_backend_hwstub.c` 不允许反向定义 adapter 合同。

---

## 9. `soc_cpu` 访问策略
### 9.1 正确性基线路径固定
本轮 `soc_cpu` 访问 `KV_MAIN` 的正确性基线路径固定为：

- 默认使用 uncached alias

### 9.2 性能优化策略
在不改变 `region / offset / queue` 合同的前提下，本轮允许追加：

- cached alias + flush/invalidate 优化

也就是说：

- uncached 路径负责先打通正确性
- cached 优化可以在本轮做，但只作为性能策略，不允许改变软件合同层定义

---

## 10. Attention 首轮范围固定
Attention 第一轮的目标固定为打通以下三条路径：

- `memory path`
- `queue path`
- `primitive contract`

数值收敛优先级固定为：

1. `QK` 优先进入 `int8 GEMM` 路径
2. `AV` 优先进入 `int8 GEMM` 路径
3. `softmax` 继续由 `EXEC_SOFTMAX / post_engine` 负责，本轮不强制整数化

必要时允许通过 `PTQ / QAT / 微调` 支持 `QK / AV` 数值收敛。

---

## 11. 实现顺序冻结
实现顺序固定为以下八步，不允许调换：

1. 更新 `runtime_backend.h`，优先改造 `qk_matmul / av_matmul` 接口，去掉完整历史 `KV cache` 的裸 `float *key_cache/value_cache` backend 合同。
2. 更新 `runtime_memory_map.h`，补齐 `KV_MAIN_WINDOW`、`KEY_CACHE_MAIN_REGION`、`VALUE_CACHE_MAIN_REGION` 的固定软件地址合同，并为 key/value 的 data/scale 二级子区预留独立命名与地址口径；其中 `*_scale_region` 的精确边界与容量待实验结论形成后再冻结。
3. 更新 `runtime_types.h`，正式定义 `key_cache_main_data_region / key_cache_main_scale_region / value_cache_main_data_region / value_cache_main_scale_region` 所需的 region 描述能力，并去除完整历史 `KV cache` 的裸 `float *` 正式接口语义。
4. 更新 `runtime_common.c`，引入 `kv view / accessor helper`，其底层绑定到 `KEY_CACHE_MAIN_REGION / VALUE_CACHE_MAIN_REGION` 下的 data/scale 二级子区；移除“当前 token 的局部 `k/v` 先写局部，再搬运到历史区”的正式路径；若短期保留 `state->key_cache/value_cache` 旧字段，其语义只允许作为过渡字段，不再是正式接口，并移除当前 token 局部 `k/v` 的正式 arena 常驻语义。
5. 更新与 `KV cache` 访问相关的 builder / lowering / verify 路径，统一改为基于一级窗口与二级子区的 `region + byte offset`，不再长期接受裸 `float *key_cache/value_cache`。
6. 实现 `KV scale` 粒度实验路径，完成精度 / 搬运 / 组织复杂度对比，并形成结论。
7. 更新 `runtime_hw_adapter.[hc]`，完成 primitive lowering、queue v0 编码、submit / completion 聚合，并将软件观测字段统一改为 `trace_tag`。
8. 更新 `runtime_backend_hwstub.c`，让其成为 queue-aware stub，并完成软件侧合同验证。

---

## 12. 测试计划冻结
### 12.1 地址与容量
必须校验：

- `KEY_CACHE_MAIN_REGION = 320KiB`
- `VALUE_CACHE_MAIN_REGION = 320KiB`
- `KV_MAIN_WINDOW = 640KiB`
- `KV_MAIN_WINDOW` 与 `CMD_WINDOW` 不冲突
- `key/value data region` 必须各自完整落在对应一级窗口内
- `key/value scale region` 必须作为独立二级子区存在，并具备独立命名与地址口径
- 一级窗口与二级子区的命名口径一致
- 本轮不要求冻结 `*_scale_region` 的精确边界与容量，但要求其不与对应 data 子区语义混淆

### 12.2 Arena / KV 行为
必须校验：

- `runtime_backend.h` 中的 `qk_matmul / av_matmul` 已不再以裸 `float *key_cache/value_cache` 作为正式 backend 合同
- 完整历史 `KV cache` 已不再作为正式裸 `float *key_cache/value_cache` 接口保留
- CPU 侧完整历史 `KV cache` 访问必须经由 `kv view / accessor helper`
- `kv view / accessor helper` 的底层数据来源绑定到 `key_cache_main_data_region / key_cache_main_scale_region / value_cache_main_data_region / value_cache_main_scale_region`
- 通过 accessor 读写 `(layer, time, channel)` 时，与本计划固定的 offset 规则一致
- `builder / backend / verify` 不再长期接收裸 `float *key_cache/value_cache`
- `RUNTIME_ARENA` 中不再包含完整历史 `KV cache`
- 当前 token 的局部 `k/v` 不再作为正式长期 arena 对象保留
- `runtime_common.c` 已移除“局部 `k/v` 先落局部对象，再搬运到历史区”的正式路径
- 当前 token 的 `k/v` 计算完成后直接写入或量化写入 `KV_MAIN` 最终槽位
- `KV_MAIN` 一级结构固定为一级窗口 + data/scale 二级子区

### 12.3 KV 布局与 offset
必须校验：

- `KV data` 的逻辑布局固定为 `[layer][time][kv_dim]`
- `(layer, time, channel)` 到 data byte offset 的换算符合本计划中的固定公式
- head 访问路径符合 `kv_head_idx * head_size` 的固定派生规则

### 12.4 Queue 生命周期
必须校验：

- `CMDQ/CMPQ` 按 ring queue 工作
- `head/tail` 推进正确
- slot 复用正确
- reset 后 queue 状态恢复为空队列
- 旧 entry 不清零时，队列语义仍然正确

### 12.5 编码与下发
必须校验：

- builder 只使用 `region + byte offset` 下发
- builder / backend / verify 不再长期接受裸 `float *key_cache/value_cache`
- `MOVE / EXEC / WAIT / CFG` 的软件解释符合本计划
- payload 字段单位符合本计划定义
- 软件观测字段统一使用 `trace_tag`，completion 仍只认 `seq_id`

### 12.6 量化实验
必须产出：

- 不同 `KV scale` 粒度下的精度对比结果
- 不同 `KV scale` 粒度下的搬运/地址组织复杂度对比结果
- 选定方案的原因与回填建议

### 12.7 行为回归
必须校验：

- `runq_verify` 继续 PASS
- `runq_deploy_cpu` 继续可运行
- `runq_deploy_hw` 在 queue 单路径下跑通最小 generate

---

## 13. 禁止事项
以下做法从本轮开始一律禁止：

1. 在 builder / DMA lowering / queue encoder 中直接使用裸 `float *` 作为正式下发接口。
2. 继续把完整历史 `KV cache` 作为正式裸 `float *key_cache/value_cache` 接口向 backend / verify / adapter 扩散。
3. 把完整历史 `KV cache` 放回 `RUNTIME_ARENA`。
4. 把完整历史 `KV cache` 放进 `ACCEL_SHARED` 作为正式主数据面。
5. 重新引入 `host` 这种含糊术语。
6. 让 backend 反向定义 adapter 合同。
7. 让 completion 重新依赖 slot 编号而不是 `seq_id`。
8. 继续使用 `job_id` 作为正式术语，而不是统一改为 `trace_tag`。
9. 把 queue 做成一次性线性队列，或依赖“清空旧 entry”回收 slot。
10. 在 scale 粒度实验完成前，把任意实验性 scale 布局伪装为最终冻结合同。
11. 在 `KV_MAIN` 一级结构上回退到 data/scale 行内交织方案。

---

## 14. 当前阶段默认假设
1. `intel_host_stub` 只复用相同 region/offset/queue 合同，不模拟真实 SoC 物理落位。
2. 本轮冻结的是软件侧内存/对象模型/queue v0 的核心合同，不冻结硬件 ISA 细节。
3. `QK / AV` 进入 `int8 GEMM` 路线是本轮明确目标，而不是后续可选项。
4. `KV scale` 粒度通过实验探索后，在下一版计划中回填冻结。