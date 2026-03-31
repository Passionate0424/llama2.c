#ifndef LLAMA2C_DEPLOY_RUNTIME_FRONTEND_H
#define LLAMA2C_DEPLOY_RUNTIME_FRONTEND_H

#include "runtime_backend.h"
#include "runtime_types.h"

typedef struct {
    RuntimeModel model;
    RuntimeTokenizer tokenizer;
    RuntimeSampler sampler;
    RuntimeBackend backend;
    RuntimeDecodeConfig decode_cfg;
} RuntimeApp;

int runtime_app_init(
    RuntimeApp *app,
    RuntimeBackendKind backend_kind,
    const RuntimeDecodeConfig *decode_cfg,
    unsigned long long seed
);
void runtime_app_destroy(RuntimeApp *app);
int runtime_run_generate(RuntimeApp *app, char *prompt, int steps);
int runtime_run_chat(RuntimeApp *app, char *prompt, char *system_prompt, int steps);

#endif
