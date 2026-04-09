#include "runtime_decode_cfg.h"
#include "runtime_frontend.h"
#include "runtime_hw_adapter.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 独立部署入口：
// 1. 保持 runq 风格参数；
// 2. 不运行时切 backend；
// 3. 通过编译宏分别生成 CPU/HW 两个可执行文件。

static void print_usage(void) {
    fprintf(stderr, "Usage: runq_deploy_[cpu|hw] [options]\n");
    fprintf(stderr, "Example: runq_deploy_cpu -n 140 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature, default 0.93\n");
    fprintf(stderr, "  -p <float>  top-p, default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default 133742\n");
    fprintf(stderr, "  -n <int>    number of steps, default 140\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default generate\n");
    fprintf(stderr, "  -y <string> system prompt in chat mode\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
    RuntimeApp app;
    RuntimeDecodeConfig cfg = RUNTIME_DECODE_ENHANCED;
    RuntimeBackendKind backend_kind = RUNTIME_BACKEND_SWREF;
    unsigned long long seed = 133742ULL;
    char *prompt = NULL;
    char *mode = "generate";
    char *system_prompt = NULL;
    int ret = 0;

#if defined(RUNQ_DEPLOY_HW)
    // backend 选择在编译期冻结，避免部署版在运行时携带多余的分叉逻辑。
    backend_kind = RUNTIME_BACKEND_HWSTUB;
#endif

    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) print_usage();
        if (argv[i][0] != '-') print_usage();
        if (strlen(argv[i]) != 2) print_usage();
        if (argv[i][1] == 't') cfg.temperature = (float)atof(argv[i + 1]);
        else if (argv[i][1] == 'p') cfg.top_p = (float)atof(argv[i + 1]);
        else if (argv[i][1] == 's') seed = strtoull(argv[i + 1], NULL, 10);
        else if (argv[i][1] == 'n') cfg.max_new_tokens = atoi(argv[i + 1]);
        else if (argv[i][1] == 'i') prompt = argv[i + 1];
        else if (argv[i][1] == 'm') mode = argv[i + 1];
        else if (argv[i][1] == 'y') system_prompt = argv[i + 1];
        else print_usage();
    }

#if defined(RUNQ_DEPLOY_EMBEDDED_DEFAULTS)
    // SoC 仿真入口通常没有 argv，给一组固定默认值，确保上电后能直接跑出可见 token。
    if (argc <= 1) {
        prompt = (char *)RUNTIME_PROMPTS_STABLE[0];
        mode = "generate";
    }
#endif

    if (!prompt) prompt = "";
    if (runtime_app_init(&app, backend_kind, &cfg, seed) != 0) {
        fprintf(stderr, "runq_deploy: 初始化失败\n");
        return 1;
    }
#if defined(RUNQ_DEPLOY_HW)
    // 硬件版第一阶段虽然仍是 HW_STUB，但至少把共享缓冲和地址口径显式打印出来。
    runtime_hw_adapter_dump_layout(stdout);
#endif
    if (strcmp(mode, "generate") == 0) {
        ret = runtime_run_generate(&app, prompt, cfg.max_new_tokens);
    } else if (strcmp(mode, "chat") == 0) {
        ret = runtime_run_chat(&app, prompt, system_prompt, cfg.max_new_tokens);
    } else {
        print_usage();
    }
    runtime_app_destroy(&app);
    return ret;
}
