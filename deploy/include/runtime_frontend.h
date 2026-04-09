#ifndef LLAMA2C_DEPLOY_RUNTIME_FRONTEND_H
#define LLAMA2C_DEPLOY_RUNTIME_FRONTEND_H

#include "runtime_backend.h"
#include "runtime_types.h"

// RuntimeApp 把部署运行时对外需要持有的对象集中起来：
// - model/tokenizer：静态资产与解析后的运行时视图
// - sampler：采样策略与随机状态
// - backend：当前选中的算子后端实现
// - decode_cfg：本次运行使用的解码参数
// 这样主入口只需要传递一个 app，即可覆盖初始化、推理和销毁阶段。
typedef struct {
    RuntimeModel model;
    RuntimeTokenizer tokenizer;
    RuntimeSampler sampler;
    RuntimeBackend backend;
    RuntimeDecodeConfig decode_cfg;
} RuntimeApp;

// 初始化部署运行时：加载资产、创建 sampler，并绑定指定 backend。
// 返回 0 表示成功，返回非 0 表示初始化过程中有任一步失败。
int runtime_app_init(
    RuntimeApp *app,
    RuntimeBackendKind backend_kind,
    const RuntimeDecodeConfig *decode_cfg,
    unsigned long long seed
);

// 释放 RuntimeApp 持有的所有运行时资源。
void runtime_app_destroy(RuntimeApp *app);

// 执行标准 generate 模式，prompt 中的 token 会先静默送入模型，再进入采样输出阶段。
int runtime_run_generate(RuntimeApp *app, char *prompt, int steps);

// 执行 chat 模式，支持首轮 system prompt 与后续多轮 user/assistant 交互。
int runtime_run_chat(RuntimeApp *app, char *prompt, char *system_prompt, int steps);

#endif
