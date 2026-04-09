#include "runtime_verify.h"

// 验证程序入口保持极简：
// 1. 不在 main 中重复组织用例；
// 2. 固定调用统一的 verify suite；
// 3. 让脚本和人工运行都只关心返回码。

int main(void) {
    return runtime_run_verify_suite();
}
