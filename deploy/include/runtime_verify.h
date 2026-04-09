#ifndef LLAMA2C_DEPLOY_RUNTIME_VERIFY_H
#define LLAMA2C_DEPLOY_RUNTIME_VERIFY_H

#include "runtime_backend.h"
#include "runtime_types.h"

// 验证入口层：
// 1. 固定组织 SW_REF -> HW_STUB -> compare 的验证流程；
// 2. 对外只暴露一套 suite 入口，便于脚本和主程序直接调用。

// 执行 deploy 运行时的整套验证用例。
// 返回 0 表示全部通过，返回非 0 表示至少一个 case 失败。
int runtime_run_verify_suite(void);

#endif
