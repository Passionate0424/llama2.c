import argparse
import copy
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from functools import partial
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

import eval_quant_schemes as eqs
import qat_exec_graph_explore as qx
from tinystories import Task
from tokenizer import Tokenizer


DEMO_PROMPTS = [
    "",
    "Once upon a time",
    "Lily was sad because",
    "The little boy found a",
]

SAMPLED_DEMO_CFG = {
    # 这是目前对 260K demo 更均衡的一组展示口径：
    # 既能保留一定故事推进，又比 greedy / 过冷采样更不容易模板化。
    "temperature": 0.95,
    "top_k": 40,
    "top_p": 0.9,
}


@dataclass
class TrainStage:
    # 用分阶段 curriculum 逐步把完整硬件执行图拉进训练目标，而不是一上来全开。
    name: str
    steps: int
    lr: float
    use_approx_norm: bool
    use_approx_softmax: bool
    ce_weight: float
    kl_weight: float
    feat_weight: float
    train_norm_weight: bool
    quantize_output: bool = False
    norm_impl: str = "rough"
    softmax_impl: str = "rough"
    rep_weight: float = 0.0
    rep_window: int = 0


@dataclass
class TeacherBundle:
    # same_vocab=True 表示可以直接做 logits / feature 蒸馏；
    # same_vocab=False 则更适合作为“文本/序列 teacher”来扩展训练语料。
    name: str
    model: nn.Module
    tokenizer: Tokenizer
    same_vocab: bool


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def data_root_exists() -> bool:
    data_dir = os.path.join(REPO_DIR, "data", "TinyStories_all_data")
    return os.path.isdir(data_dir)


def custom_512_data_exists() -> bool:
    bin_dir = os.path.join(REPO_DIR, "data", "tok512")
    model_path = os.path.join(REPO_DIR, "data", "tok512.model")
    if not os.path.isfile(model_path):
        return False
    if not os.path.isdir(bin_dir):
        return False
    return any(name.endswith(".bin") for name in os.listdir(bin_dir))


def tokenizer_path(quant_eval_dir: str) -> str:
    return os.path.join(quant_eval_dir, "tok512.model")


def checkpoint_path(quant_eval_dir: str) -> str:
    return os.path.join(quant_eval_dir, "stories260K.pt")


def big_teacher_checkpoint_path(quant_eval_dir: str) -> str:
    return os.path.join(quant_eval_dir, "stories42M.pt")


def llama_tokenizer_path() -> str:
    return os.path.join(REPO_DIR, "tokenizer.model")


def require_existing_file(filepath: str, hint: str) -> None:
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        return
    raise FileNotFoundError(f"缺少文件：{filepath}。请先手动准备：{hint}")


def ensure_teacher_artifacts(quant_eval_dir: str, need_42m: bool) -> None:
    # 训练脚本不再自动下载，统一依赖本地已经准备好的模型和 tokenizer。
    require_existing_file(checkpoint_path(quant_eval_dir), "stories260K.pt")
    require_existing_file(tokenizer_path(quant_eval_dir), "tok512.model")
    if need_42m:
        require_existing_file(big_teacher_checkpoint_path(quant_eval_dir), "stories42M.pt")
    require_existing_file(llama_tokenizer_path(), "默认 llama tokenizer.model")


def build_fallback_dataset(
    corpus_teachers: List[TeacherBundle],
    student_tokenizer: Tokenizer,
    *,
    device: str,
    seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # fallback 模式直接用 teacher 生成文本，再回落到 student tokenizer 切样本。
    texts = []
    for bundle in corpus_teachers:
        texts.extend(build_teacher_corpus_texts(bundle, device=device))
    return qx.build_samples(student_tokenizer, texts, seq_len)


def build_mixed_text_task_batch_iter(
    *,
    student_tokenizer: Tokenizer,
    corpus_teachers: List[TeacherBundle],
    batch_size: int,
    seq_len: int,
    device: str,
):
    # 用真实 TinyStories token 流做主训练数据，同时周期性插入大 teacher 生成文本切片，
    # 目标是进一步改善 free-running 生成质量，而不是只贴 teacher logits。
    task_iter = build_task_batch_factory(
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
        vocab_size=512,
        vocab_source="custom",
    )(split="train")

    xs_mix = None
    ys_mix = None
    if corpus_teachers:
        mix_texts = []
        for bundle in corpus_teachers:
            mix_texts.extend(build_teacher_corpus_texts(bundle, device=device))
        xs_mix, ys_mix = qx.build_samples(student_tokenizer, mix_texts, seq_len)

    mix_iter = None
    if xs_mix is not None and xs_mix.shape[0] > 0:
        mix_iter = make_in_memory_batch_iter(xs_mix, ys_mix, batch_size=batch_size, device=device)

    step = 0
    while True:
        step += 1
        # 默认 3:1，把真实数据作为主语料，teacher 文本作为辅助语料。
        if mix_iter is not None and step % 4 == 0:
            yield next(mix_iter)
        else:
            yield next(task_iter)


def build_task_batch_iter(
    *,
    split: str,
    batch_size: int,
    seq_len: int,
    device: str,
):
    return Task.iter_batches(
        split=split,
        batch_size=batch_size,
        max_seq_len=seq_len,
        vocab_size=32000,
        vocab_source="llama2",
        device=device,
        num_workers=0,
    )


def build_task_batch_factory(
    *,
    batch_size: int,
    seq_len: int,
    device: str,
    vocab_size: int,
    vocab_source: str,
):
    # 尽量复用原训练脚本的数据流组织方式，减少新脚本引入的变量。
    return partial(
        Task.iter_batches,
        batch_size=batch_size,
        max_seq_len=seq_len,
        vocab_size=vocab_size,
        vocab_source=vocab_source,
        device=device,
        num_workers=0,
    )


def make_in_memory_batch_iter(
    xs: torch.Tensor,
    ys: torch.Tensor,
    *,
    batch_size: int,
    device: str,
):
    indices = list(range(xs.shape[0]))
    cursor = 0
    while True:
        if cursor == 0:
            random.shuffle(indices)
        batch_ids = indices[cursor : cursor + batch_size]
        if len(batch_ids) < batch_size:
            random.shuffle(indices)
            cursor = 0
            continue
        cursor = (cursor + batch_size) % len(indices)
        x = xs[batch_ids].to(device)
        y = ys[batch_ids].to(device)
        yield x, y


def build_teacher_corpus_texts(bundle: TeacherBundle, *, device: str) -> List[str]:
    # 大 teacher 与 student 词表可以不同，但最终都回落成文本，再用 student tokenizer 重新切样本。
    texts = [eqs.LONG_TEST_TEXT]
    for prompt in qx.GEN_PROMPTS + eqs.CALIB_PROMPTS + eqs.TEST_PROMPTS:
        if prompt:
            texts.append(prompt)
    with torch.no_grad():
        for prompt in qx.GEN_PROMPTS:
            ids = bundle.tokenizer.encode(prompt, bos=True, eos=False)
            x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]
            greedy = bundle.model.generate(x, max_new_tokens=128, temperature=0.0)
            texts.append(bundle.tokenizer.decode(greedy[0].tolist()))
            sample = bundle.model.generate(x, max_new_tokens=128, temperature=0.8, top_k=32)
            texts.append(bundle.tokenizer.decode(sample[0].tolist()))
    return texts


def freeze_unselected_params(
    model: nn.Module,
    *,
    train_norm_weight: bool,
    attention_only: bool,
    train_output_weight: bool,
) -> None:
    # 默认先把可训练集合收口到 attention 路径，避免 FFN/分类头一起漂移，便于判断 QK/AV/KV 代理是否有效。
    for name, param in model.named_parameters():
        param.requires_grad = False
        if ".weight" in name and "attention." in name:
            param.requires_grad = True
        elif not attention_only and ".weight" in name and "feed_forward." in name:
            param.requires_grad = True
        if train_norm_weight and name.endswith(".weight") and ("attention_norm" in name or "ffn_norm" in name or name == "norm.weight"):
            param.requires_grad = True
        if train_output_weight and name == "output.weight":
            param.requires_grad = True


class FeatureCapture:
    """抓最终 residual stream，作为蒸馏里的 feature loss。"""

    def __init__(self, module: nn.Module):
        self.value = None
        self.hook = module.register_forward_hook(self._hook)

    def _hook(self, _mod, _inp, out):
        self.value = out.float()

    def close(self):
        self.hook.remove()


def build_model_for_stage(
    ckpt_path: str,
    tokenizer: Tokenizer,
    linear_scales,
    rms_scales,
    stage: TrainStage,
    device: str,
    *,
    kv_cache_mode: str,
    qk_score_mode: str,
    av_out_mode: str,
    attn_group_size: int,
    attention_only: bool,
    train_output_weight: bool,
) -> nn.Module:
    # 每个阶段都从同一 student 结构出发，只改变“哪些硬件代理被拉进图里”。
    student, _ = eqs.load_model(ckpt_path, device)
    cfg = qx.ExecGraphQATConfig(
        use_approx_norm=stage.use_approx_norm,
        use_approx_softmax="none" if not stage.use_approx_softmax else stage.softmax_impl,
        norm_impl=stage.norm_impl,
        softmax_impl=stage.softmax_impl,
        quantize_output=stage.quantize_output,
        linear_act_mode="dynamic_token",
        linear_weight_mode="per_row",
        use_acc32=True,
        use_mid16=True,
        kv_cache_mode=kv_cache_mode,
        qk_score_mode=qk_score_mode,
        av_out_mode=av_out_mode,
        attn_group_size=attn_group_size,
    )
    student = qx.prepare_qat_model(student, linear_scales, rms_scales, cfg)
    freeze_unselected_params(
        student,
        train_norm_weight=stage.train_norm_weight,
        attention_only=attention_only,
        train_output_weight=train_output_weight,
    )
    return student


def kl_distill_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    s_logp = F.log_softmax(student_logits / temperature, dim=-1)
    t_prob = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(s_logp, t_prob, reduction="batchmean") * (temperature * temperature)


def masked_feature_loss(student_feat: torch.Tensor, teacher_feat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = (target != -1).unsqueeze(-1).expand_as(student_feat)
    if not torch.any(mask):
        return torch.zeros((), device=student_feat.device)
    return F.mse_loss(student_feat[mask], teacher_feat[mask])


def repetition_unlikelihood_loss(
    logits: torch.Tensor,
    context_tokens: torch.Tensor,
    targets: torch.Tensor,
    *,
    window: int,
) -> torch.Tensor:
    # 最小反重复损失：
    # 对每个位置，把最近 window 个上下文 token 当作“负样本”，
    # 惩罚模型继续给这些 token 高概率，但排除当前位置真实 target。
    if window <= 0:
        return torch.zeros((), device=logits.device)

    probs = F.softmax(logits.float(), dim=-1)
    bsz, seqlen, vocab = probs.shape
    losses = []
    for t in range(seqlen):
        start = max(0, t - window)
        neg_ids = context_tokens[:, start : t + 1]
        neg_mask = torch.zeros((bsz, vocab), dtype=torch.bool, device=logits.device)
        neg_mask.scatter_(1, neg_ids, True)
        valid_tgt = targets[:, t] >= 0
        tgt_ids = targets[:, t].clamp(min=0)
        neg_mask[valid_tgt, tgt_ids[valid_tgt]] = False
        if not torch.any(neg_mask):
            continue
        p_neg = probs[:, t, :][neg_mask].clamp(min=1e-6, max=1.0 - 1e-6)
        losses.append(-torch.log1p(-p_neg).mean())
    if not losses:
        return torch.zeros((), device=logits.device)
    return torch.stack(losses).mean()


def distinct_stats(ids: List[int]) -> Dict[str, float]:
    toks = ids
    if not toks:
        return {
            "distinct1": 0.0,
            "distinct2": 0.0,
            "distinct3": 0.0,
            "max_repeat_run": 0.0,
            "repeat_bigram_ratio": 0.0,
            "repeat_trigram_ratio": 0.0,
            "tail_loop_ratio": 0.0,
        }
    unigrams = set(toks)
    bigrams = set(zip(toks, toks[1:])) if len(toks) > 1 else set()
    trigrams = set(zip(toks, toks[1:], toks[2:])) if len(toks) > 2 else set()
    max_run = 1
    cur = 1
    for i in range(1, len(toks)):
        if toks[i] == toks[i - 1]:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 1
    bigram_total = max(0, len(toks) - 1)
    trigram_total = max(0, len(toks) - 2)
    repeat_bigram_ratio = 0.0
    repeat_trigram_ratio = 0.0
    if bigram_total > 0:
        repeat_bigram_ratio = 1.0 - (len(bigrams) / bigram_total)
    if trigram_total > 0:
        repeat_trigram_ratio = 1.0 - (len(trigrams) / trigram_total)
    tail_loop_ratio = 0.0
    tail_span = min(24, len(toks))
    if tail_span >= 8:
        tail = toks[-tail_span:]
        half = tail_span // 2
        prefix = tail[:half]
        suffix = tail[-half:]
        matches = sum(1 for a, b in zip(prefix, suffix) if a == b)
        tail_loop_ratio = matches / max(1, half)
    return {
        "distinct1": len(unigrams) / max(1, len(toks)),
        "distinct2": len(bigrams) / max(1, len(toks) - 1),
        "distinct3": len(trigrams) / max(1, len(toks) - 2),
        "max_repeat_run": float(max_run),
        "repeat_bigram_ratio": repeat_bigram_ratio,
        "repeat_trigram_ratio": repeat_trigram_ratio,
        "tail_loop_ratio": tail_loop_ratio,
    }


@torch.no_grad()
def generate_with_top_p(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
) -> torch.Tensor:
    # 采样演示统一走 top-p + top-k，避免训练日志和单独扫表脚本口径不一致。
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.params.max_seq_len else idx[:, -model.params.max_seq_len :]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
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


def evaluate_model(
    model: nn.Module,
    teacher_logits: torch.Tensor,
    base_loss: float,
    tokenizer: Tokenizer,
    device: str,
) -> Dict[str, float]:
    long_ids = tokenizer.encode(eqs.LONG_TEST_TEXT, bos=True, eos=False)
    long_x = torch.tensor(long_ids, dtype=torch.long, device=device)[None, ...]
    result = eqs.evaluate_scheme(model, tokenizer, teacher_logits, base_loss)
    eqs.add_prompt_comparisons({"float": {"prompts": []}, "eval": result})
    eqs.add_scheme_aggregates({"eval": result})
    return {
        "loss": result["logits"]["loss"],
        "loss_delta": result["logits"]["loss_delta"],
        "cosine": result["logits"]["cosine"],
        "avg_prefix_match_tokens": result.get("aggregate", {}).get("avg_prefix_match_tokens"),
        "avg_token_match_ratio": result.get("aggregate", {}).get("avg_token_match_ratio"),
    }


def run_stage(
    stage: TrainStage,
    student: nn.Module,
    teacher: nn.Module,
    train_iter,
    *,
    device: str,
    temperature: float,
    eval_interval: int,
    tokenizer: Tokenizer,
    teacher_long_logits: torch.Tensor,
    base_loss: float,
    grad_accum_steps: int,
    weight_decay: float,
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    # 这里的 teacher 固定是 same-vocab teacher，用来做 KL / feature 蒸馏。
    # 大 teacher 如果词表不同，则只通过上面的 corpus_teachers 影响训练数据分布。
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=stage.lr,
        weight_decay=weight_decay,
    )
    teacher.eval()
    student.train()

    teacher_feat = FeatureCapture(teacher.norm)
    student_feat = FeatureCapture(student.norm)

    history = []
    for step in range(stage.steps):
        optimizer.zero_grad(set_to_none=True)
        # 更接近原项目训练方式：允许通过梯度累积扩大等效 batch。
        loss_total = 0.0
        ce_total = 0.0
        kl_total = 0.0
        feat_total = 0.0
        rep_total = 0.0
        for _micro in range(grad_accum_steps):
            x, y = next(train_iter)
            with torch.no_grad():
                t_logits = teacher(x, targets=y).detach()
                t_feat = teacher_feat.value.detach()

            s_logits = student(x, targets=y).float()
            s_feat = student_feat.value

            ce = student.last_loss
            kl = kl_distill_loss(s_logits, t_logits.float(), temperature)
            feat = masked_feature_loss(s_feat, t_feat, y)
            rep = repetition_unlikelihood_loss(
                s_logits,
                x,
                y,
                window=stage.rep_window,
            )
            loss = (
                stage.ce_weight * ce
                + stage.kl_weight * kl
                + stage.feat_weight * feat
                + stage.rep_weight * rep
            )
            (loss / grad_accum_steps).backward()

            loss_total += float(loss.item())
            ce_total += float(ce.item())
            kl_total += float(kl.item())
            feat_total += float(feat.item())
            rep_total += float(rep.item())

        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        row = {
            "step": step + 1,
            "loss": loss_total / grad_accum_steps,
            "ce": ce_total / grad_accum_steps,
            "kl": kl_total / grad_accum_steps,
            "feat": feat_total / grad_accum_steps,
            "rep": rep_total / grad_accum_steps,
        }
        if (step + 1) % eval_interval == 0 or step == 0 or step + 1 == stage.steps:
            student.eval()
            row["eval"] = evaluate_with_teacher_prompts(student, teacher_long_logits, base_loss, tokenizer, device)
            student.train()
        history.append(row)

    teacher_feat.close()
    student_feat.close()
    student.eval()
    final_eval = evaluate_with_teacher_prompts(student, teacher_long_logits, base_loss, tokenizer, device)
    return history, final_eval


def evaluate_with_teacher_prompts(
    model: nn.Module,
    teacher_long_logits: torch.Tensor,
    base_loss: float,
    tokenizer: Tokenizer,
    device: str,
) -> Dict[str, float]:
    result = eqs.evaluate_scheme(model, tokenizer, teacher_long_logits, base_loss)
    # 这里不依赖 prefix 对拍结果做决策，主看 logits/cosine；生成统计仍保留。
    eqs.add_scheme_aggregates({"model": result})
    demos = []
    sampled_demos = []
    all_ids = []
    with torch.no_grad():
        for prompt in DEMO_PROMPTS:
            ids = tokenizer.encode(prompt, bos=True, eos=False)
            x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]
            y = model.generate(x, max_new_tokens=80, temperature=0.0)
            out_ids = y[0].tolist()
            text = tokenizer.decode(out_ids)
            stats = distinct_stats(out_ids)
            all_ids.extend(out_ids)
            demos.append(
                {
                    "prompt": prompt,
                    # 只保留前缀，避免 JSON 里塞过长文本。
                    "text_prefix": text[:240],
                    "distinct1": stats["distinct1"],
                    "distinct2": stats["distinct2"],
                    "distinct3": stats["distinct3"],
                    "max_repeat_run": stats["max_repeat_run"],
                    "repeat_bigram_ratio": stats["repeat_bigram_ratio"],
                    "repeat_trigram_ratio": stats["repeat_trigram_ratio"],
                    "tail_loop_ratio": stats["tail_loop_ratio"],
                }
            )
            # 260K 模型在官方项目里 sample 模式通常比 greedy 更像样，demo 也应显式记录这一点。
            y_sample = generate_with_top_p(
                model,
                x,
                max_new_tokens=80,
                temperature=SAMPLED_DEMO_CFG["temperature"],
                top_p=SAMPLED_DEMO_CFG["top_p"],
                top_k=SAMPLED_DEMO_CFG["top_k"],
            )
            out_ids_sample = y_sample[0].tolist()
            text_sample = tokenizer.decode(out_ids_sample)
            stats_sample = distinct_stats(out_ids_sample)
            sampled_demos.append(
                {
                    "prompt": prompt,
                    "text_prefix": text_sample[:240],
                    "distinct1": stats_sample["distinct1"],
                    "distinct2": stats_sample["distinct2"],
                    "distinct3": stats_sample["distinct3"],
                    "max_repeat_run": stats_sample["max_repeat_run"],
                    "repeat_bigram_ratio": stats_sample["repeat_bigram_ratio"],
                    "repeat_trigram_ratio": stats_sample["repeat_trigram_ratio"],
                    "tail_loop_ratio": stats_sample["tail_loop_ratio"],
                }
            )
    overall_stats = distinct_stats(all_ids)
    return {
        "loss": result["logits"]["loss"],
        "loss_delta": result["logits"]["loss_delta"],
        "cosine": result["logits"]["cosine"],
        "avg_prefix_match_tokens": result.get("aggregate", {}).get("avg_prefix_match_tokens"),
        "avg_token_match_ratio": result.get("aggregate", {}).get("avg_token_match_ratio"),
        "distinct1": overall_stats["distinct1"],
        "distinct2": overall_stats["distinct2"],
        "distinct3": overall_stats["distinct3"],
        "max_repeat_run": overall_stats["max_repeat_run"],
        "repeat_bigram_ratio": overall_stats["repeat_bigram_ratio"],
        "repeat_trigram_ratio": overall_stats["repeat_trigram_ratio"],
        "tail_loop_ratio": overall_stats["tail_loop_ratio"],
        "demo_samples": demos,
        "sampled_demo_cfg": SAMPLED_DEMO_CFG,
        "sampled_demo_samples": sampled_demos,
    }


def default_stages(total_steps: int, base_lr: float, norm_impl: str, softmax_impl: str) -> List[TrainStage]:
    # 三阶段总步数严格受 total_steps 约束，便于 smoke / 正式实验共用一套脚本。
    s1 = max(1, total_steps // 4)
    s2 = max(1, total_steps // 4)
    s3 = max(1, total_steps - s1 - s2)
    return [
        TrainStage(
            name="linear_mid16",
            steps=s1,
            lr=base_lr,
            use_approx_norm=False,
            use_approx_softmax=False,
            ce_weight=0.20,
            kl_weight=1.00,
            feat_weight=0.10,
            train_norm_weight=False,
            norm_impl=norm_impl,
            softmax_impl=softmax_impl,
            rep_weight=0.0,
            rep_window=0,
        ),
        TrainStage(
            name="add_norm",
            steps=s2,
            lr=base_lr * 0.7,
            use_approx_norm=True,
            use_approx_softmax=False,
            ce_weight=0.25,
            kl_weight=1.00,
            feat_weight=0.15,
            train_norm_weight=True,
            norm_impl=norm_impl,
            softmax_impl=softmax_impl,
            rep_weight=0.0,
            rep_window=0,
        ),
        TrainStage(
            name="add_softmax",
            steps=s3,
            lr=base_lr * 0.5,
            use_approx_norm=True,
            use_approx_softmax=True,
            ce_weight=0.30,
            kl_weight=1.00,
            feat_weight=0.20,
            train_norm_weight=True,
            norm_impl=norm_impl,
            softmax_impl=softmax_impl,
            rep_weight=0.0,
            rep_window=0,
        ),
    ]


def linear_only_stages(total_steps: int, base_lr: float, norm_impl: str, softmax_impl: str) -> List[TrainStage]:
    return [
        TrainStage(
            name="linear_mid16_only",
            steps=max(1, total_steps),
            lr=base_lr,
            use_approx_norm=False,
            use_approx_softmax=False,
            ce_weight=0.20,
            kl_weight=1.00,
            feat_weight=0.10,
            train_norm_weight=False,
            norm_impl=norm_impl,
            softmax_impl=softmax_impl,
            rep_weight=0.0,
            rep_window=0,
        )
    ]


def gentle_norm_stages(total_steps: int, base_lr: float, norm_impl: str, softmax_impl: str) -> List[TrainStage]:
    # 更温和地把 norm 拉进训练图：
    # 先给 linear 更长的预热，再降低 norm/softmax 阶段的学习率和 loss 压力。
    s1 = max(1, total_steps // 2)
    s2 = max(1, total_steps // 3)
    s3 = max(1, total_steps - s1 - s2)
    return [
        TrainStage(
            name="linear_mid16",
            steps=s1,
            lr=base_lr,
            use_approx_norm=False,
            use_approx_softmax=False,
            ce_weight=0.15,
            kl_weight=1.00,
            feat_weight=0.08,
            train_norm_weight=False,
            norm_impl=norm_impl,
            softmax_impl=softmax_impl,
            rep_weight=0.0,
            rep_window=0,
        ),
        TrainStage(
            name="add_norm_gentle",
            steps=s2,
            lr=base_lr * 0.4,
            use_approx_norm=True,
            use_approx_softmax=False,
            ce_weight=0.15,
            kl_weight=0.80,
            feat_weight=0.08,
            train_norm_weight=True,
            norm_impl=norm_impl,
            softmax_impl=softmax_impl,
            rep_weight=0.02,
            rep_window=8,
        ),
        TrainStage(
            name="add_softmax_gentle",
            steps=s3,
            lr=base_lr * 0.25,
            use_approx_norm=True,
            use_approx_softmax=True,
            ce_weight=0.20,
            kl_weight=0.80,
            feat_weight=0.10,
            train_norm_weight=True,
            norm_impl=norm_impl,
            softmax_impl=softmax_impl,
            rep_weight=0.05,
            rep_window=12,
        ),
    ]


def quality_push_stages(total_steps: int, base_lr: float, norm_impl: str, softmax_impl: str) -> List[TrainStage]:
    # 这个预设更偏“demo 质量”：
    # 1. 线性阶段继续保留蒸馏主导，先把硬件执行图主路径训稳；
    # 2. norm/softmax 阶段提高 CE 占比，减弱 teacher 过强约束带来的模板化/重复问题；
    # 3. feature loss 适当减弱，避免 student 只会贴 teacher 中间表示而丢掉生成自由度。
    s1 = max(1, total_steps // 2)
    s2 = max(1, total_steps // 3)
    s3 = max(1, total_steps - s1 - s2)
    return [
        TrainStage(
            name="linear_mid16_quality",
            steps=s1,
            lr=base_lr,
            use_approx_norm=False,
            use_approx_softmax=False,
            ce_weight=0.20,
            kl_weight=1.00,
            feat_weight=0.06,
            train_norm_weight=False,
            norm_impl=norm_impl,
            softmax_impl=softmax_impl,
            rep_weight=0.0,
            rep_window=0,
        ),
        TrainStage(
            name="add_norm_quality",
            steps=s2,
            lr=base_lr * 0.35,
            use_approx_norm=True,
            use_approx_softmax=False,
            ce_weight=0.40,
            kl_weight=0.50,
            feat_weight=0.03,
            train_norm_weight=True,
            norm_impl=norm_impl,
            softmax_impl=softmax_impl,
            rep_weight=0.05,
            rep_window=10,
        ),
        TrainStage(
            name="add_softmax_quality",
            steps=s3,
            lr=base_lr * 0.20,
            use_approx_norm=True,
            use_approx_softmax=True,
            ce_weight=0.55,
            kl_weight=0.30,
            feat_weight=0.02,
            train_norm_weight=True,
            norm_impl=norm_impl,
            softmax_impl=softmax_impl,
            rep_weight=0.10,
            rep_window=16,
        ),
    ]


def official_ce_push_stages(total_steps: int, base_lr: float, norm_impl: str, softmax_impl: str) -> List[TrainStage]:
    # 更贴近原项目训练目标：
    # 1. 第一阶段仍保留蒸馏，先把执行图主路径对齐；
    # 2. 后两阶段明显提高 CE 权重，让 student 更像“自己学着说话”，而不是死贴 teacher；
    # 3. 保留少量 KL 和 rep penalty，避免完全失去对 teacher 与重复的约束。
    s1 = max(1, total_steps // 2)
    s2 = max(1, total_steps // 3)
    s3 = max(1, total_steps - s1 - s2)
    return [
        TrainStage(
            name="linear_mid16_official",
            steps=s1,
            lr=base_lr,
            use_approx_norm=False,
            use_approx_softmax=False,
            ce_weight=0.25,
            kl_weight=0.90,
            feat_weight=0.04,
            train_norm_weight=False,
            norm_impl=norm_impl,
            softmax_impl=softmax_impl,
            rep_weight=0.0,
            rep_window=0,
        ),
        TrainStage(
            name="add_norm_official",
            steps=s2,
            lr=base_lr * 0.35,
            use_approx_norm=True,
            use_approx_softmax=False,
            ce_weight=0.70,
            kl_weight=0.15,
            feat_weight=0.01,
            train_norm_weight=True,
            norm_impl=norm_impl,
            softmax_impl=softmax_impl,
            rep_weight=0.08,
            rep_window=12,
        ),
        TrainStage(
            name="add_softmax_official",
            steps=s3,
            lr=base_lr * 0.20,
            use_approx_norm=True,
            use_approx_softmax=True,
            ce_weight=0.85,
            kl_weight=0.05,
            feat_weight=0.0,
            train_norm_weight=True,
            norm_impl=norm_impl,
            softmax_impl=softmax_impl,
            rep_weight=0.12,
            rep_window=16,
        ),
    ]


def final_polish_stages(total_steps: int, base_lr: float, norm_impl: str, softmax_impl: str) -> List[TrainStage]:
    # 最后收尾阶段：尽量不再重新学习执行图，而是集中压重复、提升可读性。
    return [
        TrainStage(
            name="final_polish",
            steps=max(1, total_steps),
            lr=base_lr,
            use_approx_norm=True,
            use_approx_softmax=True,
            ce_weight=1.00,
            kl_weight=0.0,
            feat_weight=0.0,
            train_norm_weight=True,
            norm_impl=norm_impl,
            softmax_impl=softmax_impl,
            rep_weight=0.20,
            rep_window=24,
        )
    ]


def final_polish_strong_stages(total_steps: int, base_lr: float, norm_impl: str, softmax_impl: str) -> List[TrainStage]:
    # 更激进的 demo 精修：
    # 1. 继续保持 CE-only 主导；
    # 2. 进一步提高反重复强度和窗口；
    # 3. 降低学习率，避免把已经成形的语言结构训坏。
    return [
        TrainStage(
            name="final_polish_strong",
            steps=max(1, total_steps),
            lr=base_lr,
            use_approx_norm=True,
            use_approx_softmax=True,
            ce_weight=1.00,
            kl_weight=0.0,
            feat_weight=0.0,
            train_norm_weight=True,
            norm_impl=norm_impl,
            softmax_impl=softmax_impl,
            rep_weight=0.30,
            rep_window=32,
        )
    ]


def build_stages(preset: str, total_steps: int, base_lr: float, norm_impl: str, softmax_impl: str) -> List[TrainStage]:
    if preset == "default":
        return default_stages(total_steps, base_lr, norm_impl, softmax_impl)
    if preset == "linear_only":
        return linear_only_stages(total_steps, base_lr, norm_impl, softmax_impl)
    if preset == "gentle_norm":
        return gentle_norm_stages(total_steps, base_lr, norm_impl, softmax_impl)
    if preset == "quality_push":
        return quality_push_stages(total_steps, base_lr, norm_impl, softmax_impl)
    if preset == "official_ce_push":
        return official_ce_push_stages(total_steps, base_lr, norm_impl, softmax_impl)
    if preset == "final_polish":
        return final_polish_stages(total_steps, base_lr, norm_impl, softmax_impl)
    if preset == "final_polish_strong":
        return final_polish_strong_stages(total_steps, base_lr, norm_impl, softmax_impl)
    raise ValueError(preset)


def stage_checkpoint_path(output_path: str, stage_name: str) -> str:
    base, _ext = os.path.splitext(output_path)
    return f"{base}_{stage_name}.pt"


def load_teacher_bundle(name: str, *, quant_eval_dir: str, device: str) -> TeacherBundle:
    if name == "260k":
        model, _ = eqs.load_model(checkpoint_path(quant_eval_dir), device)
        tokenizer = Tokenizer(tokenizer_model=tokenizer_path(quant_eval_dir))
        return TeacherBundle(name="260k", model=model.eval(), tokenizer=tokenizer, same_vocab=True)
    if name == "42m":
        # 42M teacher 使用默认大词表 tokenizer，因此不能直接和 260K student 做 logits 对齐。
        model, _ = eqs.load_model(big_teacher_checkpoint_path(quant_eval_dir), device)
        tokenizer = Tokenizer(tokenizer_model=llama_tokenizer_path())
        return TeacherBundle(name="42m", model=model.eval(), tokenizer=tokenizer, same_vocab=False)
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser(description="More complete QAT + distillation training for exec-graph-aligned proxies.")
    parser.add_argument("--quant-eval-dir", default=os.path.join(REPO_DIR, "quant_eval"))
    parser.add_argument("--output", default=os.path.join(REPO_DIR, "artifacts", "qat_exec_graph_train.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--distill-teacher", choices=["260k"], default="260k")
    parser.add_argument("--corpus-teachers", default="260k", help="逗号分隔：260k,42m")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--norm-impl", choices=["rough", "improved"], default="rough")
    parser.add_argument("--softmax-impl", choices=["rough", "improved"], default="rough")
    parser.add_argument("--stage-preset", choices=["default", "linear_only", "gentle_norm", "quality_push", "official_ce_push", "final_polish", "final_polish_strong"], default="default")
    parser.add_argument("--dataset-mode", choices=["auto", "task", "fallback"], default="auto")
    parser.add_argument("--mix-teacher-text", action="store_true", help="在真实 custom-512 数据流中混入 teacher 生成文本。")
    parser.add_argument("--save-stage-checkpoints", action="store_true", help="保存每个阶段结束后的 student checkpoint，便于单独做 demo 采样。")
    parser.add_argument("--resume-from", default="", help="从已有 stage checkpoint 继续训练。")
    parser.add_argument("--student-ckpt", default="", help="显式指定 student 初始 checkpoint；为空时沿用 quant_eval/stories260K.pt")
    parser.add_argument("--tokenizer-path", default="", help="显式指定 tokenizer.model；为空时沿用 quant_eval/tok512.model")
    parser.add_argument("--kv-cache-mode", choices=["none", "per_kv_head", "per_row", "group64"], default="none")
    parser.add_argument("--qk-score-mode", choices=["none", "mid16_token", "mid16_layer"], default="none")
    parser.add_argument("--av-out-mode", choices=["none", "per_kv_head", "per_row", "group64"], default="none")
    parser.add_argument("--attn-group-size", type=int, default=64)
    parser.add_argument("--attention-only", action="store_true", help="只训练 attention.* 权重，避免 FFN 一起漂移。")
    parser.add_argument("--no-train-output-weight", action="store_true", help="关闭 output.weight 训练，只保留 attention 定向微调。")
    args = parser.parse_args()

    device = detect_device(args.device)
    seed_everything(args.seed)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    corpus_teacher_names = [name.strip() for name in args.corpus_teachers.split(",") if name.strip()]
    need_42m = "42m" in corpus_teacher_names
    ensure_teacher_artifacts(args.quant_eval_dir, need_42m=need_42m)

    ckpt_path = args.student_ckpt if args.student_ckpt else checkpoint_path(args.quant_eval_dir)
    tok_path = args.tokenizer_path if args.tokenizer_path else tokenizer_path(args.quant_eval_dir)
    student_tokenizer = Tokenizer(tokenizer_model=tok_path)

    # same-vocab teacher 负责 KL / feature 蒸馏，保证训练目标维度完全对齐。
    distill_teacher = load_teacher_bundle(args.distill_teacher, quant_eval_dir=args.quant_eval_dir, device=device)
    corpus_teachers = [load_teacher_bundle(name, quant_eval_dir=args.quant_eval_dir, device=device) for name in corpus_teacher_names]

    linear_scales, rms_scales = eqs.collect_scales(distill_teacher.model, student_tokenizer, device)

    long_ids = student_tokenizer.encode(eqs.LONG_TEST_TEXT, bos=True, eos=False)
    long_x = torch.tensor(long_ids, dtype=torch.long, device=device)[None, ...]
    with torch.no_grad():
        teacher_long_logits = distill_teacher.model(long_x, targets=long_x).float()
    base_loss = float(distill_teacher.model.last_loss.item())

    use_task_data = args.dataset_mode == "task" or (
        args.dataset_mode == "auto" and (custom_512_data_exists() or data_root_exists())
    )
    if use_task_data:
        # 优先使用 custom-512 真实数据流；如果没有，再退回 llama2 大词表数据流。
        if custom_512_data_exists():
            vocab_size = 512
            vocab_source = "custom"
            dataset_mode = "task_custom_512"
        else:
            vocab_size = 32000
            vocab_source = "llama2"
            dataset_mode = "task_llama2"
        if args.mix_teacher_text and vocab_source == "custom":
            train_iter = build_mixed_text_task_batch_iter(
                student_tokenizer=student_tokenizer,
                corpus_teachers=corpus_teachers,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                device=device,
            )
        else:
            task_factory = build_task_batch_factory(
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                device=device,
                vocab_size=vocab_size,
                vocab_source=vocab_source,
            )
            train_iter = task_factory(split="train")
        dataset_info = {
            "mode": dataset_mode,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "vocab_size": vocab_size,
            "vocab_source": vocab_source,
            "mix_teacher_text": bool(args.mix_teacher_text and vocab_source == "custom"),
        }
    else:
        xs, ys = build_fallback_dataset(corpus_teachers, student_tokenizer, device=device, seq_len=args.seq_len)
        train_iter = make_in_memory_batch_iter(xs, ys, batch_size=args.batch_size, device=device)
        dataset_info = {
            "mode": "fallback",
            "num_samples": int(xs.shape[0]),
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "corpus_teachers": corpus_teacher_names,
        }

    results = {
        "device": device,
        "distill_teacher": distill_teacher.name,
        "corpus_teachers": corpus_teacher_names,
        "dataset": dataset_info,
        "train": {
            "steps": args.steps,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_accum_steps": args.grad_accum_steps,
            "temperature": args.temperature,
            "eval_interval": args.eval_interval,
            "stage_preset": args.stage_preset,
            "kv_cache_mode": args.kv_cache_mode,
            "qk_score_mode": args.qk_score_mode,
            "av_out_mode": args.av_out_mode,
            "attn_group_size": args.attn_group_size,
            "student_ckpt": ckpt_path,
            "tokenizer_path": tok_path,
            "attention_only": args.attention_only,
            "train_output_weight": not args.no_train_output_weight,
        },
        "stages": [],
    }

    resume_state = None
    if args.resume_from:
        # 允许直接从某个最好阶段 checkpoint 继续做生成质量精修。
        resume_ckpt = torch.load(args.resume_from, map_location=device)
        resume_state = resume_ckpt["model"]

    student = None
    for stage in build_stages(args.stage_preset, args.steps, args.lr, args.norm_impl, args.softmax_impl):
        if student is None:
            student = build_model_for_stage(
                ckpt_path,
                student_tokenizer,
                linear_scales,
                rms_scales,
                stage,
                device,
                kv_cache_mode=args.kv_cache_mode,
                qk_score_mode=args.qk_score_mode,
                av_out_mode=args.av_out_mode,
                attn_group_size=args.attn_group_size,
                attention_only=args.attention_only,
                train_output_weight=not args.no_train_output_weight,
            )
            if resume_state is not None:
                student.load_state_dict(resume_state, strict=False)
        else:
            state = copy.deepcopy(student.state_dict())
            student = build_model_for_stage(
                ckpt_path,
                student_tokenizer,
                linear_scales,
                rms_scales,
                stage,
                device,
                kv_cache_mode=args.kv_cache_mode,
                qk_score_mode=args.qk_score_mode,
                av_out_mode=args.av_out_mode,
                attn_group_size=args.attn_group_size,
                attention_only=args.attention_only,
                train_output_weight=not args.no_train_output_weight,
            )
            student.load_state_dict(state, strict=False)

        history, final_eval = run_stage(
            stage,
            student,
            distill_teacher.model,
            train_iter,
            device=device,
            temperature=args.temperature,
            eval_interval=args.eval_interval,
            tokenizer=student_tokenizer,
            teacher_long_logits=teacher_long_logits,
            base_loss=base_loss,
            grad_accum_steps=args.grad_accum_steps,
            weight_decay=args.weight_decay,
        )
        results["stages"].append(
            {
                "config": asdict(stage),
                "history_head": history[:8],
                "history_tail": history[-8:],
                "final_eval": final_eval,
            }
        )
        if args.save_stage_checkpoints:
            ckpt = {
                "model": student.state_dict(),
                "model_args": {
                    "dim": student.params.dim,
                    "n_layers": student.params.n_layers,
                    "n_heads": student.params.n_heads,
                    "n_kv_heads": student.params.n_kv_heads,
                    "vocab_size": student.params.vocab_size,
                    "multiple_of": student.params.multiple_of,
                    "max_seq_len": student.params.max_seq_len,
                    "dropout": student.params.dropout,
                },
                "stage": asdict(stage),
                "dataset": dataset_info,
                "train": results["train"],
            }
            ckpt_path = stage_checkpoint_path(args.output, stage.name)
            torch.save(ckpt, ckpt_path)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    try:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    except UnicodeEncodeError:
        # Windows 控制台可能仍是 GBK；文件已成功写出，这里退化成最小 ASCII 提示，避免把已完成训练误判成失败。
        print(json.dumps({"output": args.output, "status": "written"}, ensure_ascii=True))


if __name__ == "__main__":
    main()
