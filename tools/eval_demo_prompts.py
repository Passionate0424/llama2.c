import argparse
import json
import os
import sys
from typing import Dict, List

import torch


REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

import eval_demo_sampling as eds
from tokenizer import Tokenizer


# 这组 prompt 不追求“任务覆盖面”，而是专门为现场 demo 挑选：
# 1. 句子短，降低前缀噪声；
# 2. 更容易触发一个完整的小故事；
# 3. 尽量减少人物过多导致的实体漂移。
PROMPT_BANK: Dict[str, List[str]] = {
    "stable": [
        "Once upon a time",
        "Lily found a",
        "Tom was happy because",
        "The little bird saw a",
    ],
    "lively": [
        "Ben opened the box and",
        "The puppy ran to the",
        "Mia wanted to help",
        "The red ball rolled into",
    ],
    "longer": [
        "One sunny morning, Lily and her mom",
        "Timmy went to the park and found",
        "The little boy looked under the tree and",
        "After the rain, the small village was",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="按 prompt 集合评估 demo 展示效果。")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.93)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repetition-penalty", type=float, default=1.10)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument(
        "--prompt-groups",
        default="stable,lively,longer",
        help="逗号分隔，可选 stable,lively,longer",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(REPO_DIR, "artifacts", "demo_prompt_eval.json"),
    )
    return parser.parse_args()


def prompt_score(stats: Dict[str, float]) -> float:
    # 这个分数不是训练目标，只是帮助从多条 prompt 中挑更适合现场展示的候选。
    return (
        1.00 * stats["distinct2"]
        + 0.35 * stats["distinct3"]
        - 0.90 * stats["repeat_trigram_ratio"]
        - 0.40 * stats["repeat_bigram_ratio"]
        - 0.30 * stats["tail_loop_ratio"]
        - 0.05 * max(0.0, float(stats["max_repeat_run"]) - 1.0)
    )


def selected_prompts(group_names: List[str]) -> List[Dict[str, str]]:
    prompts = []
    for group in group_names:
        for prompt in PROMPT_BANK[group]:
            prompts.append({"group": group, "prompt": prompt})
    return prompts


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    groups = [name.strip() for name in args.prompt_groups.split(",") if name.strip()]
    for group in groups:
        if group not in PROMPT_BANK:
            raise ValueError(f"未知 prompt 组：{group}")

    model = eds.load_model_from_checkpoint(args.checkpoint, args.device)
    tokenizer = Tokenizer(tokenizer_model=args.tokenizer)

    results = []
    for item in selected_prompts(groups):
        prompt = item["prompt"]
        ids = tokenizer.encode(prompt, bos=True, eos=False)
        x = torch.tensor(ids, dtype=torch.long, device=args.device)[None, ...]
        y = eds.generate_with_top_p(
            model,
            x,
            args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
        out_ids = y[0].tolist()
        stats = eds.distinct_stats(out_ids)
        text = tokenizer.decode(out_ids)
        row = {
            "group": item["group"],
            "prompt": prompt,
            "text_prefix": text[:480],
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            **stats,
        }
        row["score"] = prompt_score(row)
        results.append(row)

    results.sort(key=lambda row: row["score"], reverse=True)
    payload = {
        "checkpoint": args.checkpoint,
        "max_new_tokens": args.max_new_tokens,
        "decode": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
        },
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
