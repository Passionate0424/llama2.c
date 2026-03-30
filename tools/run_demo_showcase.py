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


# 这里固化当前阶段筛出来的更适合现场展示的 prompt。
SHOWCASE_PROMPTS: Dict[str, List[str]] = {
    "best": [
        "Once upon a time",
        "Ben opened the box and",
        "Tom was happy because",
        "Timmy went to the park and found",
    ],
    "stable": [
        "Once upon a time",
        "Lily found a",
        "Tom was happy because",
        "The little bird saw a",
    ],
    "story": [
        "One day, Lily found a little bird and",
        "Ben wanted to help his friend, so",
        "Lily and Tim went to the park because",
        "The boy opened the box and saw",
    ],
}


DECODE_PRESETS = {
    # 当前最强的“增强版展示”口径。
    "enhanced": {
        "temperature": 0.93,
        "top_p": 0.9,
        "top_k": 40,
        "repetition_penalty": 1.05,
        "no_repeat_ngram_size": 3,
    },
    # 如果想要稍微保守一点，可以用这个更轻的版本。
    "natural": {
        "temperature": 0.95,
        "top_p": 0.9,
        "top_k": 40,
        "repetition_penalty": 1.10,
        "no_repeat_ngram_size": 0,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="按当前最佳展示配置直接生成 demo 文本。")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--preset", choices=sorted(DECODE_PRESETS.keys()), default="enhanced")
    parser.add_argument("--prompt-set", choices=sorted(SHOWCASE_PROMPTS.keys()), default="best")
    parser.add_argument("--max-new-tokens", type=int, default=140)
    parser.add_argument(
        "--output",
        default=os.path.join(REPO_DIR, "artifacts", "demo_showcase.json"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    decode = dict(DECODE_PRESETS[args.preset])
    prompts = SHOWCASE_PROMPTS[args.prompt_set]

    model = eds.load_model_from_checkpoint(args.checkpoint, args.device)
    tokenizer = Tokenizer(tokenizer_model=args.tokenizer)

    results = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, bos=True, eos=False)
        x = torch.tensor(ids, dtype=torch.long, device=args.device)[None, ...]
        y = eds.generate_with_top_p(
            model,
            x,
            args.max_new_tokens,
            temperature=decode["temperature"],
            top_p=decode["top_p"],
            top_k=decode["top_k"],
            repetition_penalty=decode["repetition_penalty"],
            no_repeat_ngram_size=decode["no_repeat_ngram_size"],
        )
        out_ids = y[0].tolist()
        stats = eds.distinct_stats(out_ids)
        text = tokenizer.decode(out_ids)
        results.append(
            {
                "prompt": prompt,
                "text": text,
                **stats,
            }
        )

    payload = {
        "checkpoint": args.checkpoint,
        "preset": args.preset,
        "prompt_set": args.prompt_set,
        "max_new_tokens": args.max_new_tokens,
        "decode": decode,
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
