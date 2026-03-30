import argparse
import json
import os
import sys
from typing import List, Tuple

import torch
import torch.nn.functional as F


REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

from model import ModelArgs, Transformer
from tokenizer import Tokenizer


PROMPTS = [
    "",
    "Once upon a time",
    "Lily was sad because",
    "The little boy found a",
]


def load_model_from_checkpoint(checkpoint_path: str, device: str) -> Transformer:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gptconf = ModelArgs(**checkpoint["model_args"])
    model = Transformer(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key, value in list(state_dict.items()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    return model


def distinct_stats(ids: List[int]) -> dict:
    if not ids:
        return {
            "distinct1": 0.0,
            "distinct2": 0.0,
            "distinct3": 0.0,
            "max_repeat_run": 0,
            "repeat_bigram_ratio": 0.0,
            "repeat_trigram_ratio": 0.0,
            "tail_loop_ratio": 0.0,
        }
    unigrams = set(ids)
    bigrams = set(zip(ids, ids[1:])) if len(ids) > 1 else set()
    trigrams = set(zip(ids, ids[1:], ids[2:])) if len(ids) > 2 else set()
    max_run = 1
    cur = 1
    for i in range(1, len(ids)):
        if ids[i] == ids[i - 1]:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 1

    # 用重复 n-gram 比例观察“故事看似不坏，但在局部循环打转”的问题。
    bigram_total = max(0, len(ids) - 1)
    trigram_total = max(0, len(ids) - 2)
    repeat_bigram_ratio = 0.0
    repeat_trigram_ratio = 0.0
    if bigram_total > 0:
        repeat_bigram_ratio = 1.0 - (len(bigrams) / bigram_total)
    if trigram_total > 0:
        repeat_trigram_ratio = 1.0 - (len(trigrams) / trigram_total)

    # 句尾回环启发式：看最后 24 token 中，后半段与前半段重复多少。
    tail_loop_ratio = 0.0
    tail_span = min(24, len(ids))
    if tail_span >= 8:
        tail = ids[-tail_span:]
        half = tail_span // 2
        prefix = tail[:half]
        suffix = tail[-half:]
        matches = sum(1 for a, b in zip(prefix, suffix) if a == b)
        tail_loop_ratio = matches / max(1, half)

    return {
        "distinct1": len(unigrams) / max(1, len(ids)),
        "distinct2": len(bigrams) / max(1, len(ids) - 1),
        "distinct3": len(trigrams) / max(1, len(ids) - 2),
        "max_repeat_run": max_run,
        "repeat_bigram_ratio": repeat_bigram_ratio,
        "repeat_trigram_ratio": repeat_trigram_ratio,
        "tail_loop_ratio": tail_loop_ratio,
    }


@torch.no_grad()
def generate_with_top_p(
    model: Transformer,
    idx: torch.Tensor,
    max_new_tokens: int,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> torch.Tensor:
    # 统一支持 top-p + top-k，并额外支持 repetition penalty / no-repeat ngram。
    # 这样可以直接探索“只改解码，不改训练”能否把 demo 再拉好一点。
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.params.max_seq_len else idx[:, -model.params.max_seq_len :]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-6)

        # repetition penalty: 对已经生成过的 token 适度降温，
        # 尤其适合小模型故事生成时抑制“短模板来回念”。
        if repetition_penalty > 1.0:
            for b in range(idx.size(0)):
                seen_ids = torch.unique(idx[b])
                seen_logits = logits[b, seen_ids]
                penalized = torch.where(
                    seen_logits < 0,
                    seen_logits * repetition_penalty,
                    seen_logits / repetition_penalty,
                )
                logits[b, seen_ids] = penalized

        # no-repeat ngram: 直接阻断已经出现过的 n-gram 继续重复。
        if no_repeat_ngram_size > 0 and idx.size(1) >= no_repeat_ngram_size - 1:
            prefix_len = no_repeat_ngram_size - 1
            for b in range(idx.size(0)):
                tokens = idx[b].tolist()
                banned = set()
                if prefix_len == 0:
                    banned.update(tokens)
                else:
                    prefix = tuple(tokens[-prefix_len:])
                    for i in range(len(tokens) - no_repeat_ngram_size + 1):
                        ngram = tokens[i : i + no_repeat_ngram_size]
                        if tuple(ngram[:-1]) == prefix:
                            banned.add(ngram[-1])
                if banned:
                    logits[b, list(banned)] = float("-inf")

        probs = F.softmax(logits, dim=-1)

        if top_k > 0:
            v, _ = torch.topk(probs, min(top_k, probs.size(-1)))
            probs = probs.masked_fill(probs < v[:, [-1]], 0.0)

        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cdf = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cdf > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        next_sorted = torch.multinomial(sorted_probs, num_samples=1)
        next_token = torch.gather(sorted_idx, -1, next_sorted)
        idx = torch.cat((idx, next_token), dim=1)
    return idx


def parse_grid(grid: str) -> List[Tuple[float, float, int]]:
    configs = []
    for item in grid.split(";"):
        item = item.strip()
        if not item:
            continue
        temp_s, topp_s, topk_s = item.split(",")
        configs.append((float(temp_s), float(topp_s), int(topk_s)))
    return configs


def main():
    parser = argparse.ArgumentParser(description="扫一轮 demo 采样参数，挑最适合展示的口径。")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help=">1.0 时抑制重复 token。")
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0, help=">0 时阻断重复 n-gram。")
    parser.add_argument(
        "--grid",
        default="1.0,0.9,40;0.8,0.9,40;0.9,0.9,20;1.0,0.85,20;0.8,0.95,40",
        help="格式：temperature,top_p,top_k;...",
    )
    parser.add_argument("--output", default=os.path.join(REPO_DIR, "artifacts", "demo_sampling_sweep.json"))
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    model = load_model_from_checkpoint(args.checkpoint, args.device)
    tokenizer = Tokenizer(tokenizer_model=args.tokenizer)
    configs = parse_grid(args.grid)

    results = []
    for temperature, top_p, top_k in configs:
        samples = []
        merged_ids = []
        for prompt in PROMPTS:
            ids = tokenizer.encode(prompt, bos=True, eos=False)
            x = torch.tensor(ids, dtype=torch.long, device=args.device)[None, ...]
            y = generate_with_top_p(
                model,
                x,
                args.max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )
            out_ids = y[0].tolist()
            merged_ids.extend(out_ids)
            stats = distinct_stats(out_ids)
            text = tokenizer.decode(out_ids)
            samples.append(
                {
                    "prompt": prompt,
                    "text_prefix": text[:320],
                    "distinct1": stats["distinct1"],
                    "distinct2": stats["distinct2"],
                    "distinct3": stats["distinct3"],
                    "max_repeat_run": stats["max_repeat_run"],
                    "repeat_bigram_ratio": stats["repeat_bigram_ratio"],
                    "repeat_trigram_ratio": stats["repeat_trigram_ratio"],
                    "tail_loop_ratio": stats["tail_loop_ratio"],
                }
            )
        merged_stats = distinct_stats(merged_ids)
        results.append(
            {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": args.repetition_penalty,
                "no_repeat_ngram_size": args.no_repeat_ngram_size,
                "distinct1": merged_stats["distinct1"],
                "distinct2": merged_stats["distinct2"],
                "distinct3": merged_stats["distinct3"],
                "max_repeat_run": merged_stats["max_repeat_run"],
                "repeat_bigram_ratio": merged_stats["repeat_bigram_ratio"],
                "repeat_trigram_ratio": merged_stats["repeat_trigram_ratio"],
                "tail_loop_ratio": merged_stats["tail_loop_ratio"],
                "samples": samples,
            }
        )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
