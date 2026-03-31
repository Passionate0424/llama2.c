#include "runtime_verify.h"

// 验证入口保持极简：固定执行 SW_REF -> HW_STUB -> compare。

int main(void) {
    return runtime_run_verify_suite();
}
