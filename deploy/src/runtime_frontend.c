#include "runtime_frontend.h"

#include <stdlib.h>
#include <string.h>

#include "runtime_assets.h"
#include "runtime_common.h"

static int append_history_token(int *history_tokens, int history_capacity, int *history_len, int token, const char *tag) {
    // 这里记录“已经真正送入模型执行图”的 token 历史。
    // 采样器的 repetition penalty / no-repeat-ngram 应该基于真实上下文，而不是基于尚未执行的未来 token。
    if (*history_len >= history_capacity) {
        fprintf(stderr, "%s: history_tokens 容量不足，无法继续追加 token=%d\n", tag, token);
        return -1;
    }
    history_tokens[(*history_len)++] = token;
    return 0;
}

static int prepare_chat_turn(
    RuntimeApp *app,
    int pos,
    int steps,
    char *cli_user_prompt,
    char *system_prompt_cli,
    char *system_prompt,
    size_t system_prompt_size,
    char *user_prompt,
    size_t user_prompt_size,
    char *rendered_prompt,
    size_t rendered_prompt_size,
    int *prompt_tokens,
    int prompt_capacity,
    int *num_prompt_tokens
) {
    // 对齐 OpenLA500 / runq 的 demo 交互方式：
    // 1. 首轮可注入 system prompt
    // 2. 首轮可使用 CLI 给定 user prompt
    // 3. 后续轮次继续从串口/标准输入读取 User:
    if (pos == 0) {
        if (system_prompt_cli) {
            strncpy(system_prompt, system_prompt_cli, system_prompt_size - 1);
            system_prompt[system_prompt_size - 1] = '\0';
        } else {
            read_stdin_line("Enter system prompt (optional): ", system_prompt, system_prompt_size);
        }
    }

    if (pos == 0 && cli_user_prompt) {
        strncpy(user_prompt, cli_user_prompt, user_prompt_size - 1);
        user_prompt[user_prompt_size - 1] = '\0';
    } else {
        read_stdin_line("User: ", user_prompt, user_prompt_size);
    }

    if (pos == 0 && system_prompt[0] != '\0') {
        snprintf(
            rendered_prompt,
            rendered_prompt_size,
            "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]",
            system_prompt,
            user_prompt
        );
    } else {
        snprintf(rendered_prompt, rendered_prompt_size, "[INST] %s [/INST]", user_prompt);
    }

    if (encode_text(
        &app->tokenizer,
        rendered_prompt,
        1,
        0,
        prompt_tokens,
        prompt_capacity,
        num_prompt_tokens
    ) != 0) {
        fprintf(stderr, "runtime_run_chat: 当前轮 system/user prompt 过长，token 缓冲区不足\n");
        return -1;
    }
    if (*num_prompt_tokens < 1) {
        fprintf(stderr, "runtime_run_chat: 当前轮 chat prompt 编码后 token 数为 0\n");
        return -1;
    }
    if (*num_prompt_tokens > (steps - pos)) {
        fprintf(stderr, "runtime_run_chat: 剩余上下文不足，当前轮 prompt 需要 %d 个 token，剩余仅 %d\n",
                *num_prompt_tokens, steps - pos);
        return 1;
    }
    return 0;
}

int runtime_app_init(
    RuntimeApp *app,
    RuntimeBackendKind backend_kind,
    const RuntimeDecodeConfig *decode_cfg,
    unsigned long long seed
) {
    // 第一版固定使用默认资产和默认 decode 配置，先把主运行链路收敛下来。
    memset(app, 0, sizeof(*app));
    runtime_model_init(&app->model);
    if (runtime_load_default_model(&app->model) != 0) return -1;
    if (runtime_load_default_tokenizer(&app->tokenizer, app->model.config.vocab_size) != 0) {
        runtime_model_destroy(&app->model);
        return -1;
    }
    app->decode_cfg = *decode_cfg;
    build_sampler(
        &app->sampler,
        app->model.config.vocab_size,
        decode_cfg->temperature,
        decode_cfg->top_p,
        decode_cfg->top_k,
        decode_cfg->repetition_penalty,
        decode_cfg->no_repeat_ngram_size,
        seed
    );
    if (backend_kind == RUNTIME_BACKEND_SWREF) {
        return runtime_backend_bind_swref(&app->backend, &app->model);
    }
    return runtime_backend_bind_hwstub(&app->backend, &app->model);
}

void runtime_app_destroy(RuntimeApp *app) {
    if (app->backend.ops && app->backend.ops->destroy) {
        app->backend.ops->destroy(&app->backend);
    }
    free_sampler(&app->sampler);
    runtime_free_tokenizer(&app->tokenizer);
    runtime_model_destroy(&app->model);
    memset(app, 0, sizeof(*app));
}

int runtime_run_generate(RuntimeApp *app, char *prompt, int steps) {
    RuntimeBackend *backend = &app->backend;
    // history_tokens 保存完整生成历史，方便 repetition penalty / no-repeat-ngram 生效。
    int *history_tokens = (int *)malloc((size_t)(app->model.config.seq_len + 8) * sizeof(int));
    int num_prompt_tokens = 0;
    int history_len = 0;
    int token = 0;
    int next = 0;
    int pos = 0;
    if (!history_tokens) {
        fprintf(stderr, "runtime_run_generate: history_tokens 分配失败\n");
        return -1;
    }
    if (steps <= 0 || steps > app->model.config.seq_len) steps = app->model.config.seq_len;
    if (encode_text(
        &app->tokenizer,
        prompt,
        1,
        0,
        history_tokens,
        app->model.config.seq_len + 8,
        &num_prompt_tokens
    ) != 0) {
        fprintf(stderr, "runtime_run_generate: prompt 过长，token 缓冲区不足\n");
        free(history_tokens);
        return -1;
    }
    history_len = num_prompt_tokens;
    if (num_prompt_tokens < 1 && strlen(prompt) > 0) {
        fprintf(stderr, "runtime_run_generate: 非空 prompt 编码后 token 数为 0\n");
        free(history_tokens);
        return -1;
    }
    if (num_prompt_tokens < 1) {
        printf("Empty prompt, nothing to generate.\n");
        free(history_tokens);
        return 0;
    }

    // 每次生成前显式清空 backend 内部状态，避免上一次对话污染 KV cache。
    backend->ops->reset(backend);
    token = history_tokens[0];
    while (pos < steps) {
        float *logits = backend->ops->forward_logits(backend, token, pos);
        if (pos < num_prompt_tokens - 1) {
            next = history_tokens[pos + 1];
        } else {
            next = sample_token(&app->sampler, logits, history_tokens, history_len);
        }
        pos++;
        if (next == 1) break;
        char *piece = decode_token(&app->tokenizer, token, next);
        safe_printf_piece(piece);
        fflush(stdout);
        token = next;
        if (history_len < app->model.config.seq_len + 8) {
            history_tokens[history_len++] = next;
        }
    }
    printf("\n");
    free(history_tokens);
    return 0;
}

int runtime_run_chat(RuntimeApp *app, char *cli_user_prompt, char *system_prompt_cli, int steps) {
    RuntimeBackend *backend = &app->backend;
    char system_prompt[512] = {0};
    char user_prompt[512] = {0};
    char rendered_prompt[1152] = {0};
    int history_capacity = app->model.config.seq_len + 8;
    int prompt_capacity = app->model.config.seq_len + 128;
    int *history_tokens = (int *)malloc((size_t)history_capacity * sizeof(int));
    int *prompt_tokens = (int *)malloc((size_t)prompt_capacity * sizeof(int));
    int num_prompt_tokens = 0;
    int history_len = 0;
    int user_idx = 0;
    int user_turn = 1;
    int pos = 0;
    int token = 0;
    int next = 0;
    int ret = 0;
    if (!history_tokens || !prompt_tokens) {
        fprintf(stderr, "runtime_run_chat: token 缓冲区分配失败\n");
        free(history_tokens);
        free(prompt_tokens);
        return -1;
    }
    if (steps <= 0 || steps > app->model.config.seq_len) steps = app->model.config.seq_len;

    backend->ops->reset(backend);
    while (pos < steps) {
        if (user_turn) {
            int turn_status = prepare_chat_turn(
                app,
                pos,
                steps,
                cli_user_prompt,
                system_prompt_cli,
                system_prompt,
                sizeof(system_prompt),
                user_prompt,
                sizeof(user_prompt),
                rendered_prompt,
                sizeof(rendered_prompt),
                prompt_tokens,
                prompt_capacity,
                &num_prompt_tokens
            );
            // CLI user prompt 只在首轮使用一次，后续轮次统一走 stdin/串口输入。
            cli_user_prompt = NULL;
            if (turn_status < 0) {
                ret = -1;
                break;
            }
            if (turn_status > 0) {
                // 剩余上下文已经容不下新一轮 prompt，直接结束本次 demo。
                break;
            }
            user_idx = 0;
            user_turn = 0;
            printf("Assistant: ");
            fflush(stdout);
        }

        if (user_idx < num_prompt_tokens) {
            token = prompt_tokens[user_idx++];
        } else {
            token = next;
        }

        // 这一步保持和 OpenLA500 语义一致：
        // 当上一轮 assistant 已经给出结束 token，下一轮切回 User 输入，不再把结束 token 继续送入模型。
        if ((token == 2 || (token == 1 && pos > 0)) && !user_turn) {
            user_turn = 1;
            printf("\n");
            fflush(stdout);
            continue;
        }

        if (append_history_token(history_tokens, history_capacity, &history_len, token, "runtime_run_chat") != 0) {
            ret = -1;
            break;
        }

        float *logits = backend->ops->forward_logits(backend, token, pos);
        if (user_idx < num_prompt_tokens) {
            // 仍在静默注入当前轮 prompt token，不进入采样。
            next = prompt_tokens[user_idx];
        } else {
            next = sample_token(&app->sampler, logits, history_tokens, history_len);
        }
        pos++;

        if (user_idx >= num_prompt_tokens && next != 1 && next != 2 && !user_turn) {
            char *piece = decode_token(&app->tokenizer, token, next);
            safe_printf_piece(piece);
            fflush(stdout);
        }
    }
    printf("\n");
    free(history_tokens);
    free(prompt_tokens);
    return ret;
}
