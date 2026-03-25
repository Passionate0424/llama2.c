# runq_embedded 工作记录

本文档记录了在 `llama2.c` 仓库中为 `runq` 增加“无文件系统/头文件嵌入”运行方式的本次工作内容，重点覆盖：

- 为什么要增加 `runq_embedded`
- 本次新增/修改了哪些文件
- 如何在本地 Windows 上生成模型头文件、编译并运行
- 本次验证结果和结论
- 260K q80 模型输出质量为什么不如 `run`

## 1. 背景与目标

原始的 `runq.c` 面向 PC 文件系统环境，运行时依赖：

- 外部量化模型文件，例如 `stories260K_q80.bin`
- 外部 tokenizer 文件，例如 `tok512.bin`

但在 SoC/FPGA 场景里，通常没有方便的文件系统接口，更常见的方式是：

- 把模型和 tokenizer 作为静态数组编进镜像
- 或者把二进制放进 ROM / Flash / DDR，由程序直接按地址访问

因此本次工作的目标是：

1. 保留原始 `runq.c` 文件版用法不变
2. 新增一个嵌入式变体 `runq_embedded.c`
3. 用 `xxd -i` 风格头文件承载 q80 模型和 tokenizer
4. 在本地 Windows 上先验证这套无文件系统链路是通的

## 2. 本次新增与修改

### 2.1 新增文件

- `runq_embedded.c`
  - 复用原始 `runq.c` 的推理主逻辑
  - 改为从头文件中的静态数组加载模型和 tokenizer
- `tools/generate_embedded_260k.py`
  - 一键生成 260K 嵌入式资产
  - 负责下载、导出 q80、生成头文件
- `tools/generate_embedded_stories260k.py`
  - 兼容入口，内部转调 `generate_embedded_260k.py`
- `embedded/stories260K/stories260K_q80.h`
  - 生成后的 q80 模型头文件
- `embedded/stories260K/tok512.h`
  - 生成后的 tokenizer 头文件
- `.gitignore`
  - 忽略中间产物和本地验证输出

### 2.2 修改文件

- `Makefile`
  - 增加 `runembeddedwin` 目标
- `build_msvc.bat`
  - 增加 `runq_embedded.exe` 的构建入口
- `win.c`
- `win.h`
  - 修复 MinGW 下的 Windows 兼容问题
- `doc/stories260K.md`
  - 补充嵌入式 `runq` 的英文说明

## 3. runq_embedded 的实现方式

`runq_embedded.c` 不是把原始 `runq.c` 整体重写，而是尽量复用上游代码，只替换与“文件系统”直接耦合的部分。

实现方式是：

1. 通过 `#include "runq.c"` 复用原始推理代码
2. 覆盖以下几个入口：
   - `build_transformer`
   - `free_transformer`
   - `build_tokenizer`
   - `error_usage`
   - `main`
3. 保持 `forward`、`generate`、`chat`、`sample` 等推理路径不变

这样做的好处是：

- 和上游差异小
- 文件版 `runq` 仍可保留作基线
- 嵌入式版与文件版可以直接做输出对拍

## 4. 260K 嵌入式资产生成流程

本次固定使用官方 `stories260K.pt` 和 `tok512.model` 作为源资产，然后走 `llama2.c` 自己的导出链路。

### 4.1 生成命令

在仓库根目录执行：

```powershell
D:\anaconda3\python.exe tools\generate_embedded_stories260k.py --force
```

脚本会自动完成以下步骤：

1. 下载：
   - `stories260K.pt`
   - `tok512.model`
2. 调用：

```powershell
python export.py artifacts\stories260K\stories260K_q80.bin --version 2 --group-size 32 --checkpoint artifacts\stories260K\stories260K.pt
```

3. 调用：

```powershell
python tokenizer.py --tokenizer-model artifacts\stories260K\tok512.model
```

4. 调用 `xxd -i` 生成头文件：
   - `embedded/stories260K/stories260K_q80.h`
   - `embedded/stories260K/tok512.h`

### 4.2 为什么固定 `group_size=32`

`export.py` 的 version-2 q80 导出默认 `group_size=64`。  
对 260K 这种极小模型来说，这个设置会让量化误差偏大，容易出现输出塌缩，例如：

```text
Once upon upon upon upon ...
```

本次把导出参数固定为 `group_size=32`，原因有两个：

1. 它与 OpenLA500 参考头文件一致
2. 本地实测输出明显优于 `group_size=64`

实测结果表明，当前生成出的：

- `artifacts/stories260K/stories260K_q80.bin`
- `embedded/stories260K/stories260K_q80.h`

与 OpenLA500 中的 `stories_data.h` 对应字节流完全一致。

## 5. Windows 本地编译与运行

### 5.1 编译文件版和嵌入版

```powershell
make runwin
make runembeddedwin
```

生成结果：

- `runq.exe`
- `runq_embedded.exe`

### 5.2 运行文件版 q80

```powershell
.\runq.exe artifacts\stories260K\stories260K_q80.bin -z artifacts\stories260K\tok512.bin -t 0.0 -n 200
```

### 5.3 运行嵌入版 q80

```powershell
.\runq_embedded.exe -t 0.0 -n 200
```

### 5.4 采样参数说明

- `-t`
  - temperature，控制随机性
  - `0.0` 表示 greedy/argmax，每步都选当前最大概率 token
  - `1.0` 表示按原始概率分布采样
- `-p`
  - top-p / nucleus sampling
  - `0.9` 表示只在累计概率前 90% 的候选 token 中采样
- `-s`
  - 随机种子
  - 当 `-t > 0` 时，固定种子可以让随机采样结果可复现

## 6. 本次验证结果

### 6.1 文件版与嵌入版输出一致

在本地 Windows 上，以下两条命令已经完成对拍：

```powershell
.\runq.exe artifacts\stories260K\stories260K_q80.bin -z artifacts\stories260K\tok512.bin -t 0.0 -n 200
.\runq_embedded.exe -t 0.0 -n 200
```

结果是：输出逐字节一致。

这说明：

- 头文件化没有改变模型内容
- `runq_embedded.c` 没有引入新的推理行为偏差
- 嵌入式加载链路与文件版是一致的

### 6.2 runq_embedded 不依赖运行时 `.bin`

本次还从仓库外部目录直接启动了 `runq_embedded.exe`，验证它在没有当前目录 `.bin` 文件辅助的情况下仍然可以正常运行。

这说明它已经满足“无文件系统/静态嵌入资产”的目标。

### 6.3 q80 结果比 fp32 差是正常现象

对于这个 260K tiny model：

- `run` 使用 fp32 checkpoint，文本相对流畅
- `runq` / `runq_embedded` 使用 q80 checkpoint，文本质量明显下降

例如：

```powershell
.\run.exe artifacts\stories260K\stories260K.bin -z artifacts\stories260K\tok512.bin -t 0.0 -n 120
```

会输出比较正常的故事开头。

而：

```powershell
.\runq_embedded.exe -t 0.0 -n 120
```

则更容易出现：

```text
Once upon upon a time, there was a little girl named Lily...
```

或者在参数不合适时进一步塌缩成：

```text
Once upon upon upon upon ...
```

这不是嵌入式加载方案的问题，而是：

1. 260K 模型本身很小
2. q80 量化会进一步损失精度
3. `-t 0.0` 的 greedy 解码会把错误稳定放大

## 7. 对 OpenLA500 的结论

本次已经确认：

- OpenLA500 的 `stories_data.h`
- 我们用 `llama2.c` 官方导出链生成的 `stories260K_q80.bin`

两者对应的字节流完全相同。

同时：

- OpenLA500 的 `tok512.h`
- 我们生成的 `tok512.bin`

对应字节流也完全相同。

因此可以得出结论：

1. OpenLA500 使用的就是同一份 260K q80 模型和 tokenizer
2. 如果 OpenLA500 运行的是这份 `runq` 头文件资产，它的输出风格应与本地 `runq_embedded` 一致
3. OpenLA500 那边如果表现较好，原因更可能在：
   - 使用的是 `runc` / fp32 路径
   - 或者展示的是不同采样参数
   - 而不是头文件资产本身不同

## 8. 后续建议

如果下一步目标是 SoC/FPGA 集成，建议按下面思路继续推进：

1. 先把 `runq_embedded.c` 当作 PC 侧“无文件系统基线”
2. 再把 `embedded/stories260K/*.h` 作为板端统一输入资产
3. 板端优先验证：
   - 头文件能否正常链接
   - 模型加载是否成功
   - 输出是否与 PC 版 `runq_embedded.exe` 一致
4. 如果后面要提升效果，优先考虑：
   - 用 fp32 小模型先验证系统
   - 或继续研究更合适的量化策略
   - 而不是先怀疑头文件化流程

