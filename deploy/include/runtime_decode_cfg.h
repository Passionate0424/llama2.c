#ifndef LLAMA2C_DEPLOY_RUNTIME_DECODE_CFG_H
#define LLAMA2C_DEPLOY_RUNTIME_DECODE_CFG_H

#include "runtime_types.h"

// 当前阶段已经筛出来的默认展示配置，先在部署版里固定，后续再视需要开放更多 profile。
static const RuntimeDecodeConfig RUNTIME_DECODE_ENHANCED = {
    0.93f,
    0.90f,
    40,
    1.05f,
    3,
    140,
};

static const RuntimeDecodeConfig RUNTIME_DECODE_NATURAL = {
    0.95f,
    0.90f,
    40,
    1.10f,
    0,
    140,
};

static const char *RUNTIME_PROMPTS_STABLE[] = {
    "Once upon a time",
    "Ben opened the box and",
    "Tom was happy because",
    "Timmy went to the park and found",
};

static const char *RUNTIME_PROMPTS_STORY[] = {
    "One day, Lily found a little bird and",
    "Ben wanted to help his friend, so",
    "Lily and Tim went to the park because",
    "The boy opened the box and saw",
};

#define RUNTIME_PROMPTS_STABLE_COUNT ((int)(sizeof(RUNTIME_PROMPTS_STABLE) / sizeof(RUNTIME_PROMPTS_STABLE[0])))
#define RUNTIME_PROMPTS_STORY_COUNT  ((int)(sizeof(RUNTIME_PROMPTS_STORY) / sizeof(RUNTIME_PROMPTS_STORY[0])))

#endif
