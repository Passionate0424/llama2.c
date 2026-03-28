import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

import eval_quant_schemes as eqs
from model import Attention, apply_rotary_emb, repeat_kv
from tokenizer import Tokenizer

GEN_PROMPTS = [
    "",
    "Once upon a time",
    "Lily was sad because",
    "The little boy found a",
    "In a small village there was",
    "Tom saw a red ball and",
]


def ste_round(x: torch.Tensor) -> torch.Tensor:
    # 训练时保梯度、推理时仍走整数舍入语义。
    return x + (torch.round(x) - x).detach()


def qdq_lastdim_ste(x: torch.Tensor, mode: str, static_scale=None) -> torch.Tensor:
    x = x.float()
    if mode == "dynamic_token":
        # 对齐当前主线合同：每个 token 向量独立求 activation scale。
        scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    elif mode == "static_layer":
        # 静态模式保留入口，便于后续比较 PTQ/QAT 的 scale 选择差异。
        if static_scale is None:
            scale = x.abs().max().clamp(min=1e-8) / 127.0
        else:
            scale = torch.tensor(static_scale, device=x.device, dtype=x.dtype)
    else:
        raise ValueError(mode)
    q = torch.clamp(ste_round(x / scale), -127, 127)
    return q * scale


def qdq_weight_ste(w: torch.Tensor, mode: str) -> torch.Tensor:
    w = w.float()
    if mode == "per_row":
        scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
    elif mode == "per_tensor":
        scale = w.abs().max().clamp(min=1e-8) / 127.0
    else:
        raise ValueError(mode)
    q = torch.clamp(ste_round(w / scale), -127, 127)
    return q * scale


def qdq_acc32_ste(x: torch.Tensor) -> torch.Tensor:
    # 这里不再缩放，只模拟 GEMM 后落到 int32 累加域。
    return ste_round(x.float())


def qdq_mid16_token_ste(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    # MID16 用动态 token scale 先做软件探索；后续可替换成更贴近 RTL 的固定格式。
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 32767.0
    q = torch.clamp(ste_round(x / scale), -32767, 32767)
    return q * scale


class QATLinear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        *,
        weight_mode="per_row",
        act_mode="dynamic_token",
        act_scale=None,
        use_acc32=True,
        use_mid16=True,
    ):
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.weight_mode = weight_mode
        self.act_mode = act_mode
        self.act_scale = act_scale
        self.use_acc32 = use_acc32
        self.use_mid16 = use_mid16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 执行图语义：INT8(A) x INT8(W) -> ACC32 -> MID16。
        xq = qdq_lastdim_ste(x, self.act_mode, static_scale=self.act_scale)
        wq = qdq_weight_ste(self.weight, self.weight_mode)
        y = F.linear(xq.float(), wq, self.bias.float() if self.bias is not None else None)
        if self.use_acc32:
            y = qdq_acc32_ste(y)
        if self.use_mid16:
            y = qdq_mid16_token_ste(y)
        return y


class ApproxRMSNormTrain(nn.Module):
    def __init__(self, norm_module: nn.Module, input_scale: float):
        super().__init__()
        self.weight = norm_module.weight
        self.input_scale = float(max(input_scale, 1e-8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 训练代理显式复现当前 post_engine 的粗近似路径，而不是用理想浮点 RMSNorm。
        x_int = torch.clamp(ste_round(x.float() / self.input_scale), -32768, 32767)
        elems = x_int.shape[-1]
        lg2_elems = 0 if elems <= 1 else int(torch.floor(torch.log2(torch.tensor(float(elems)))).item())
        sumsq = (x_int * x_int).sum(dim=-1, keepdim=True)
        ex2 = torch.clamp(sumsq / float(1 << lg2_elems), min=1.0)
        lg2_var = torch.floor(torch.log2(ex2))
        sh = 16.0 - torch.floor(lg2_var / 2.0)
        sh_i = torch.clamp(sh, min=0.0, max=16.0)
        invstd = torch.pow(torch.full_like(sh_i, 2.0), sh_i)
        invstd = torch.where(sh_i >= 16.0, torch.full_like(invstd, 65535.0), invstd)
        invstd = torch.where(sh_i <= 0.0, torch.ones_like(invstd), invstd)
        gamma_q8 = ste_round(self.weight.float() * 256.0) / 256.0
        out = (x_int * invstd) / 256.0
        out = out * gamma_q8
        return out * self.input_scale


class ApproxRMSNormTrainImproved(nn.Module):
    def __init__(self, norm_module: nn.Module, input_scale: float):
        super().__init__()
        self.weight = norm_module.weight
        self.input_scale = float(max(input_scale, 1e-8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 改进版近似：
        # 1. 保留 int16/MID16 输入域；
        # 2. 在原浮点量纲下计算 RMS 与 inv_std，避免整数域量纲漂移；
        # 3. 仅把 inv_std / gamma 量化成硬件更容易承接的系数。
        x_mid = qdq_mid16_token_ste(x.float())
        ex2 = torch.clamp((x_mid * x_mid).mean(dim=-1, keepdim=True), min=1e-8)
        invstd = torch.rsqrt(ex2)
        invstd_q16 = ste_round(invstd * 65535.0) / 65535.0
        gamma_q12 = ste_round(self.weight.float() * 4096.0) / 4096.0
        out = x_mid * invstd_q16
        out = out * gamma_q12
        return out


def approx_softmax_scores_train(scores: torch.Tensor) -> torch.Tensor:
    # 先做 row-softmax 整体代理，重点验证训练图是否支持，不先追逐周期细节。
    scores = torch.nan_to_num(scores.float(), neginf=-1.0e4, posinf=1.0e4)
    row_max = scores.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    scale = row_max / 127.0
    s_int = torch.clamp(ste_round(scores / scale), -32768, 32767)
    max_v = s_int.amax(dim=-1, keepdim=True)
    centered = s_int - max_v
    mag = torch.clamp(-centered, 0, 255)
    num_q16 = torch.exp(-mag.float()) * 65535.0
    denom = num_q16.sum(dim=-1, keepdim=True).clamp(min=1.0)
    return (num_q16 / denom).to(scores.dtype)


def approx_softmax_scores_train_improved(scores: torch.Tensor) -> torch.Tensor:
    # 改进版 softmax：
    # 1. 先把 score 行量化到 MID16 风格域；
    # 2. 再在量化后的分数上做稳定 softmax；
    # 3. 输出概率再量化到 Q0.16，模拟硬件里有限精度概率。
    scores_mid = qdq_mid16_token_ste(torch.nan_to_num(scores.float(), neginf=-1.0e4, posinf=1.0e4))
    probs = F.softmax(scores_mid, dim=-1)
    probs_q16 = ste_round(probs * 65535.0) / 65535.0
    denom = probs_q16.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return (probs_q16 / denom).to(scores.dtype)


def make_attention_forward_train(module: Attention, use_approx_softmax: bool):
    def forward(self, x, freqs_cos, freqs_sin):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / (self.head_dim ** 0.5)
        mask = torch.full((1, 1, seqlen, seqlen), -1.0e4, device=scores.device)
        scores = scores + torch.triu(mask, diagonal=1)
        if use_approx_softmax == "rough":
            probs = approx_softmax_scores_train(scores.float())
        elif use_approx_softmax == "improved":
            probs = approx_softmax_scores_train_improved(scores.float())
        else:
            probs = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(probs, xv)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output

    return MethodType(forward, module)


def replace_module(root: nn.Module, name: str, module: nn.Module) -> None:
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], module)


@dataclass
class ExecGraphQATConfig:
    use_approx_softmax: str = "none"
    use_approx_norm: bool = True
    norm_impl: str = "rough"
    softmax_impl: str = "rough"
    quantize_output: bool = False
    linear_act_mode: str = "dynamic_token"
    linear_weight_mode: str = "per_row"
    use_acc32: bool = True
    use_mid16: bool = True


def prepare_qat_model(model: nn.Module, linear_scales, rms_scales, cfg: ExecGraphQATConfig) -> nn.Module:
    # 不改训练主循环，只在模型构建后插入可训练的硬件代理包装。
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            if name == "output" and not cfg.quantize_output:
                continue
            replace_module(
                model,
                name,
                QATLinear(
                    module,
                    weight_mode=cfg.linear_weight_mode,
                    act_mode=cfg.linear_act_mode,
                    act_scale=linear_scales.get(name),
                    use_acc32=cfg.use_acc32,
                    use_mid16=cfg.use_mid16,
                ),
            )

    if cfg.use_approx_norm:
        for name, module in list(model.named_modules()):
            if module.__class__.__name__ == "RMSNorm":
                if cfg.norm_impl == "rough":
                    repl = ApproxRMSNormTrain(module, rms_scales[name])
                elif cfg.norm_impl == "improved":
                    repl = ApproxRMSNormTrainImproved(module, rms_scales[name])
                else:
                    raise ValueError(cfg.norm_impl)
                replace_module(model, name, repl)

    if cfg.use_approx_softmax != "none":
        for module in model.modules():
            if isinstance(module, Attention):
                # Attention 内部只替换 softmax 数学，QKV/O 线性层仍走上面的 QATLinear。
                module.forward = make_attention_forward_train(module, use_approx_softmax=cfg.softmax_impl)
    return model


def build_corpus_texts(teacher: nn.Module, tokenizer: Tokenizer, device: str):
    # 用 teacher 生成的小语料做蒸馏微调，先验证软件层是否“能训动”。
    texts = [eqs.LONG_TEST_TEXT]
    for prompt in eqs.CALIB_PROMPTS + eqs.TEST_PROMPTS:
        if prompt:
            texts.append(prompt)
    with torch.no_grad():
        for prompt in GEN_PROMPTS:
            ids = tokenizer.encode(prompt, bos=True, eos=False)
            x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]
            greedy = teacher.generate(x, max_new_tokens=96, temperature=0.0)
            texts.append(tokenizer.decode(greedy[0].tolist()))
            sample = teacher.generate(x, max_new_tokens=96, temperature=0.8, top_k=16)
            texts.append(tokenizer.decode(sample[0].tolist()))
    return texts


def build_samples(tokenizer: Tokenizer, texts, seq_len: int):
    # 固定窗口切片，避免为这轮探索引入新的数据管线复杂度。
    xs = []
    ys = []
    stride = max(4, seq_len // 4)
    for text in texts:
        ids = tokenizer.encode(text, bos=True, eos=False)
        if len(ids) < 2:
            continue
        upper = max(1, len(ids) - 1)
        for start in range(0, upper, stride):
            chunk = ids[start : start + seq_len + 1]
            if len(chunk) < 2:
                continue
            x = chunk[:-1]
            y = chunk[1:]
            if len(x) < seq_len:
                pad = seq_len - len(x)
                x = x + [0] * pad
                y = y + [-1] * pad
            else:
                x = x[:seq_len]
                y = y[:seq_len]
            xs.append(x)
            ys.append(y)
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def split_samples(xs: torch.Tensor, ys: torch.Tensor, val_ratio: float, seed: int):
    total = xs.shape[0]
    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_count = max(1, int(total * val_ratio))
    train_ids = indices[val_count:]
    val_ids = indices[:val_count]
    return xs[train_ids], ys[train_ids], xs[val_ids], ys[val_ids]


def cache_teacher_logits(teacher: nn.Module, xs: torch.Tensor, ys: torch.Tensor, device: str):
    # 先缓存 teacher logits，避免训练时双模型同时前向拖慢探索速度。
    logits = []
    with torch.no_grad():
        for idx in range(xs.shape[0]):
            x = xs[idx : idx + 1].to(device)
            y = ys[idx : idx + 1].to(device)
            logit = teacher(x, targets=y).detach().cpu()
            logits.append(logit)
    return torch.cat(logits, dim=0)


def masked_kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, targets: torch.Tensor, temperature: float) -> torch.Tensor:
    mask = targets != -1
    if not mask.any():
        return torch.zeros([], device=student_logits.device)
    s = student_logits[mask] / temperature
    t = teacher_logits[mask] / temperature
    log_p = F.log_softmax(s, dim=-1)
    q = F.softmax(t, dim=-1)
    return F.kl_div(log_p, q, reduction="batchmean") * (temperature ** 2)


@torch.no_grad()
def evaluate_distill_batch(student: nn.Module, xs: torch.Tensor, ys: torch.Tensor, teacher_logits: torch.Tensor, device: str, temperature: float):
    student.eval()
    logits = student(xs.to(device), targets=ys.to(device)).float()
    targets = ys.to(device)
    teacher = teacher_logits.to(device)
    mask = (targets != -1).unsqueeze(-1)
    mse = F.mse_loss(logits[mask.expand_as(logits)], teacher[mask.expand_as(teacher)])
    kd = masked_kd_loss(logits, teacher, targets, temperature)
    ce = student.last_loss
    return {
        "mse": float(mse.item()),
        "kd": float(kd.item()),
        "ce": float(ce.item()),
    }


def run_qat_finetune(
    student: nn.Module,
    train_xs: torch.Tensor,
    train_ys: torch.Tensor,
    train_teacher_logits: torch.Tensor,
    val_xs: torch.Tensor,
    val_ys: torch.Tensor,
    val_teacher_logits: torch.Tensor,
    *,
    device: str,
    steps: int,
    batch_size: int,
    lr: float,
    temperature: float,
    kd_weight: float,
    mse_weight: float,
    ce_weight: float,
    eval_interval: int,
):
    # 更完整的 QAT+蒸馏：KL 蒸馏 + logits MSE + CE，多目标一起约束。
    student.train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.0)
    history = []
    indices = list(range(train_xs.shape[0]))
    for step in range(steps):
        batch_ids = [indices[(step * batch_size + offset) % len(indices)] for offset in range(batch_size)]
        x = train_xs[batch_ids].to(device)
        y = train_ys[batch_ids].to(device)
        t = train_teacher_logits[batch_ids].to(device)

        logits = student(x, targets=y).float()
        mask = (y != -1).unsqueeze(-1)
        mse = F.mse_loss(logits[mask.expand_as(logits)], t[mask.expand_as(t)])
        kd = masked_kd_loss(logits, t, y, temperature)
        ce = student.last_loss
        loss = mse_weight * mse + kd_weight * kd + ce_weight * ce

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        row = {
            "step": step + 1,
            "loss": float(loss.item()),
            "mse": float(mse.item()),
            "kd": float(kd.item()),
            "ce": float(ce.item()),
        }
        if (step + 1) % eval_interval == 0 or step == 0 or step + 1 == steps:
            val_metrics = evaluate_distill_batch(student, val_xs, val_ys, val_teacher_logits, device, temperature)
            row["val_mse"] = val_metrics["mse"]
            row["val_kd"] = val_metrics["kd"]
            row["val_ce"] = val_metrics["ce"]
            student.train()
        history.append(row)
    student.eval()
    return history


def summarize_scheme(result):
    agg = result.get("aggregate", {})
    return {
        "loss": result["logits"]["loss"],
        "loss_delta": result["logits"]["loss_delta"],
        "cosine": result["logits"]["cosine"],
        "avg_prefix_match_tokens": agg.get("avg_prefix_match_tokens"),
        "avg_token_match_ratio": agg.get("avg_token_match_ratio"),
    }


def main():
    parser = argparse.ArgumentParser(description="Explore QAT support for hardware-execution-graph-aligned proxies.")
    parser.add_argument("--quant-eval-dir", default=os.path.join(REPO_DIR, "quant_eval"))
    parser.add_argument("--output", default=os.path.join(REPO_DIR, "artifacts", "qat_exec_graph_explore.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--kd-weight", type=float, default=1.0)
    parser.add_argument("--mse-weight", type=float, default=0.5)
    parser.add_argument("--ce-weight", type=float, default=0.25)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    eqs.ensure_artifacts(args.quant_eval_dir)

    ckpt_path = os.path.join(args.quant_eval_dir, "stories260K.pt")
    tok_path = os.path.join(args.quant_eval_dir, "tok512.model")
    tokenizer = Tokenizer(tokenizer_model=tok_path)

    teacher, _ = eqs.load_model(ckpt_path, args.device)
    linear_scales, rms_scales = eqs.collect_scales(teacher, tokenizer, args.device)

    corpus_texts = build_corpus_texts(teacher, tokenizer, args.device)
    xs, ys = build_samples(tokenizer, corpus_texts, args.seq_len)
    train_xs, train_ys, val_xs, val_ys = split_samples(xs, ys, args.val_ratio, args.seed)
    train_teacher_logits = cache_teacher_logits(teacher, train_xs, train_ys, args.device)
    val_teacher_logits = cache_teacher_logits(teacher, val_xs, val_ys, args.device)

    long_ids = tokenizer.encode(eqs.LONG_TEST_TEXT, bos=True, eos=False)
    long_x = torch.tensor(long_ids, dtype=torch.long, device=args.device)[None, ...]
    with torch.no_grad():
        base_logits = teacher(long_x, targets=long_x).float()
    base_loss = float(teacher.last_loss.item())

    configs = {
        # 先只验证 GEMM 主数据通路和 MID16 中间域能否被训练适配。
        "exec_graph_linear_mid16": ExecGraphQATConfig(use_approx_norm=False, use_approx_softmax=False),
        # 再把当前 RMSNorm 近似拉进图里，观察 QAT 是否还能稳住。
        "exec_graph_norm": ExecGraphQATConfig(use_approx_softmax=False),
        # 最后再把 row-softmax 也拉进来，验证完整执行图训练可达性。
        "exec_graph_full": ExecGraphQATConfig(use_approx_softmax=True),
    }

    results = {
        "dataset": {
            "num_texts": len(corpus_texts),
            "num_samples": int(xs.shape[0]),
            "num_train_samples": int(train_xs.shape[0]),
            "num_val_samples": int(val_xs.shape[0]),
            "seq_len": args.seq_len,
        },
        "train": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "temperature": args.temperature,
            "kd_weight": args.kd_weight,
            "mse_weight": args.mse_weight,
            "ce_weight": args.ce_weight,
            "eval_interval": args.eval_interval,
        },
        "schemes": {},
    }

    for name, cfg in configs.items():
        student_ptq, _ = eqs.load_model(ckpt_path, args.device)
        student_ptq = prepare_qat_model(student_ptq, linear_scales, rms_scales, cfg)
        ptq_eval = eqs.evaluate_scheme(student_ptq, tokenizer, base_logits, base_loss)
        eqs.add_prompt_comparisons({"float": eqs.evaluate_scheme(teacher, tokenizer, base_logits, base_loss), name: ptq_eval})
        eqs.add_scheme_aggregates({name: ptq_eval})

        student_qat, _ = eqs.load_model(ckpt_path, args.device)
        student_qat = prepare_qat_model(student_qat, linear_scales, rms_scales, cfg)
        history = run_qat_finetune(
            student_qat,
            train_xs,
            train_ys,
            train_teacher_logits,
            val_xs,
            val_ys,
            val_teacher_logits,
            device=args.device,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            temperature=args.temperature,
            kd_weight=args.kd_weight,
            mse_weight=args.mse_weight,
            ce_weight=args.ce_weight,
            eval_interval=args.eval_interval,
        )
        qat_eval = eqs.evaluate_scheme(student_qat, tokenizer, base_logits, base_loss)
        eqs.add_prompt_comparisons({"float": eqs.evaluate_scheme(teacher, tokenizer, base_logits, base_loss), name: qat_eval})
        eqs.add_scheme_aggregates({name: qat_eval})
        val_metrics = evaluate_distill_batch(student_qat, val_xs, val_ys, val_teacher_logits, args.device, args.temperature)

        results["schemes"][name] = {
            "ptq_proxy": summarize_scheme(ptq_eval),
            "qat_proxy": summarize_scheme(qat_eval),
            "final_val_metrics": val_metrics,
            "train_history_head": history[:10],
            "train_history_tail": history[-10:],
        }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
