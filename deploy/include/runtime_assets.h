#ifndef LLAMA2C_DEPLOY_RUNTIME_ASSETS_H
#define LLAMA2C_DEPLOY_RUNTIME_ASSETS_H

#include "runtime_types.h"

// 第一版先固定使用默认资产目录里的头文件数据。
int runtime_load_default_model(RuntimeModel *model);
int runtime_load_default_tokenizer(RuntimeTokenizer *tokenizer, int vocab_size);
void runtime_free_tokenizer(RuntimeTokenizer *tokenizer);
int runtime_validate_model_layout(const RuntimeModel *model);

#endif
