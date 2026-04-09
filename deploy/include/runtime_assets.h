#ifndef LLAMA2C_DEPLOY_RUNTIME_ASSETS_H
#define LLAMA2C_DEPLOY_RUNTIME_ASSETS_H

#include "runtime_types.h"

// 资产加载层：
// 1. 负责从编译进程序的头文件数组解析模型与 tokenizer；
// 2. 对外屏蔽 q80 权重布局细节；
// 3. 在初始化阶段完成布局校验，避免运行中才暴露资产损坏问题。

// 从默认内嵌资产中加载模型配置、权重映射和运行时 state。
// 返回 0 表示成功，返回非 0 表示模型头或布局校验失败。
int runtime_load_default_model(RuntimeModel *model);

// 从默认内嵌 tokenizer 资产中解析词表与分数字段。
// vocab_size 需要与已加载模型的 vocab_size 保持一致。
int runtime_load_default_tokenizer(RuntimeTokenizer *tokenizer, int vocab_size);

// 释放 tokenizer 解析阶段分配的词表与辅助索引内存。
void runtime_free_tokenizer(RuntimeTokenizer *tokenizer);

// 校验模型权重指针是否都落在原始资产范围内。
// 该检查主要用于尽早发现头文件资产截断或布局推导错误。
int runtime_validate_model_layout(const RuntimeModel *model);

#endif
