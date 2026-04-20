import argparse
import json
import math
import os
import sys
from collections import OrderedDict
from types import MethodType

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

from model import Attention, ModelArgs, Transformer, apply_rotary_emb, repeat_kv
from tokenizer import Tokenizer


MODEL_URLS = {
    "stories260K.pt": "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/stories260K.pt",
    "tok512.model": "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/tok512.model",
}

CALIB_PROMPTS = [
    "",
    "Once upon a time",
    "The little girl was happy because",
    "In a small village there was",
    "Tom saw a red ball and",
    "The dog ran into the garden and",
    "Lily looked at the sky and",
    "A boy named Timmy wanted to",
    "The red bird sat on the tree and",
    "Mom said we should go home because",
]

TEST_PROMPTS = [
    "",
    "Once upon a time",
    "Lily was sad because",
    "The little boy found a",
]

LONG_TEST_TEXT = (
    "Once upon a time, there was a little girl named Lily. She loved to play outside in the park. "
    "One day, she saw a big, red ball. She wanted to play with it, but it was too high. "
    "Lily's mom said, \"Lily, let's go to the park.\" Lily was sad and didn't know what to do."
)


def ensure_artifacts(quant_eval_dir: str) -> None:
    os.makedirs(quant_eval_dir, exist_ok=True)
    for name, url in MODEL_URLS.items():
        path = os.path.join(quant_eval_dir, name)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            continue
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)


def load_model(ckpt_path: str, device: str):
    checkpoint_dict = torch.load(ckpt_path, map_location=device)
    gptconf = ModelArgs(**checkpoint_dict["model_args"])
    model = Transformer(gptconf)
    state_dict = checkpoint_dict["model"]
    unwanted_prefix = "_orig_mod."
    for key, value in list(state_dict.items()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    return model, checkpoint_dict


def qdq_lastdim(x: torch.Tensor, group_size=None, mode="per_tensor", static_scale=None) -> torch.Tensor:
    x = x.float()
    if mode == "per_tensor":
        if static_scale is None:
            scale = x.abs().max().clamp(min=1e-8) / 127.0
        else:
            scale = torch.tensor(static_scale, device=x.device, dtype=x.dtype)
        return torch.clamp(torch.round(x / scale), -127, 127) * scale
    if mode == "per_token":
        scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
        return torch.clamp(torch.round(x / scale), -127, 127) * scale
    if mode == "group":
        assert group_size is not None
        n = x.shape[-1]
        parts = []
        for start in range(0, n, group_size):
            chunk = x[..., start : start + group_size]
            scale = chunk.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
            parts.append(torch.clamp(torch.round(chunk / scale), -127, 127) * scale)
        return torch.cat(parts, dim=-1)
    raise ValueError(mode)


def qdq_weight(w: torch.Tensor, mode="per_tensor", group_size=None) -> torch.Tensor:
    w = w.float()
    if mode == "per_tensor":
        scale = w.abs().max().clamp(min=1e-8) / 127.0
        return torch.clamp(torch.round(w / scale), -127, 127) * scale
    if mode == "per_row":
        scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
        return torch.clamp(torch.round(w / scale), -127, 127) * scale
    if mode == "group":
        assert group_size is not None
        flat = w.reshape(-1)
        parts = []
        for start in range(0, flat.numel(), group_size):
            chunk = flat[start : start + group_size]
            scale = chunk.abs().max().clamp(min=1e-8) / 127.0
            parts.append(torch.clamp(torch.round(chunk / scale), -127, 127) * scale)
        return torch.cat(parts).reshape_as(w)
    raise ValueError(mode)


def qdq_mid(x: torch.Tensor, mode="none", static_scale=None) -> torch.Tensor:
    x = x.float()
    if mode == "none":
        return x
    if mode == "bf16":
        return x.to(torch.bfloat16).float()
    if mode == "fp16":
        return x.to(torch.float16).float()
    if mode == "int16_token":
        scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 32767.0
        return torch.clamp(torch.round(x / scale), -32767, 32767) * scale
    if mode == "int16_layer":
        if static_scale is None:
            scale = x.abs().max().clamp(min=1e-8) / 32767.0
        else:
            scale = torch.tensor(static_scale, device=x.device, dtype=x.dtype)
        return torch.clamp(torch.round(x / scale), -32767, 32767) * scale
    raise ValueError(mode)


class FakeQuantEmbedding(nn.Module):
    def __init__(self, weight_qdq: torch.Tensor):
        super().__init__()
        self.register_buffer("weight_qdq", weight_qdq)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.weight_qdq)


class FakeQuantOutput(nn.Module):
    def __init__(self, weight_qdq: torch.Tensor, act_mode="per_tensor", act_scale=None, group_size=None, out_mode="none", out_scale=None):
        super().__init__()
        self.register_buffer("weight_qdq", weight_qdq)
        self.act_mode = act_mode
        self.act_scale = act_scale
        self.group_size = group_size
        self.out_mode = out_mode
        self.out_scale = out_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_mode == "static_layer":
            x = qdq_lastdim(x, mode="per_tensor", static_scale=self.act_scale)
        elif self.act_mode == "dynamic_token":
            x = qdq_lastdim(x, mode="per_token")
        elif self.act_mode == "dynamic_group":
            x = qdq_lastdim(x, mode="group", group_size=self.group_size)
        y = F.linear(x.float(), self.weight_qdq, None)
        return qdq_mid(y, mode=self.out_mode, static_scale=self.out_scale)


class FakeQuantLinear(nn.Module):
    def __init__(self, linear: nn.Linear, weight_qdq: torch.Tensor, act_mode="static_layer", act_scale=None, group_size=None, out_mode="none", out_scale=None):
        super().__init__()
        self.register_buffer("weight_qdq", weight_qdq)
        self.bias_buf = None if linear.bias is None else linear.bias.detach().float()
        self.act_mode = act_mode
        self.act_scale = act_scale
        self.group_size = group_size
        self.out_mode = out_mode
        self.out_scale = out_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_mode == "static_layer":
            x = qdq_lastdim(x, mode="per_tensor", static_scale=self.act_scale)
        elif self.act_mode == "dynamic_token":
            x = qdq_lastdim(x, mode="per_token")
        elif self.act_mode == "dynamic_group":
            x = qdq_lastdim(x, mode="group", group_size=self.group_size)
        y = F.linear(x.float(), self.weight_qdq, self.bias_buf)
        return qdq_mid(y, mode=self.out_mode, static_scale=self.out_scale)


class ApproxRMSNorm(nn.Module):
    def __init__(self, norm_module: nn.Module, input_scale: float):
        super().__init__()
        self.input_scale = float(input_scale)
        gamma_q8 = torch.clamp(torch.round(norm_module.weight.detach().float() * 256.0), -32768, 32767).to(torch.int32)
        self.register_buffer("gamma_q8", gamma_q8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_int = torch.clamp(torch.round(x.float() / self.input_scale), -32768, 32767).to(torch.int32)
        elems = x_int.shape[-1]
        lg2_elems = 0 if elems <= 1 else int(math.floor(math.log2(elems)))
        sumsq = (x_int.to(torch.int64) * x_int.to(torch.int64)).sum(dim=-1, keepdim=True)
        ex2 = sumsq >> lg2_elems
        ex2 = torch.clamp(ex2, min=1)
        lg2_var = torch.floor(torch.log2(ex2.float())).to(torch.int64)
        sh = 16 - (lg2_var >> 1)
        invstd = torch.where(
            sh >= 16,
            torch.full_like(sh, 65535),
            torch.where(sh <= 0, torch.ones_like(sh), (torch.ones_like(sh) << sh)),
        )
        out_int = (x_int.to(torch.int64) * invstd.to(torch.int64)) >> 8
        out_int = (out_int * self.gamma_q8.to(torch.int64)) >> 8
        return out_int.float() * self.input_scale


def approx_softmax_scores(scores: torch.Tensor) -> torch.Tensor:
    row_max = scores.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    scale = row_max / 127.0
    s_int = torch.clamp(torch.round(scores / scale), -32768, 32767).to(torch.int32)
    max_v = s_int.amax(dim=-1, keepdim=True)
    centered = s_int - max_v
    mag = torch.clamp(-centered, 0, 255).to(torch.int64)
    num_q16 = torch.round(torch.exp(-mag.float()) * 65535.0)
    denom = num_q16.sum(dim=-1, keepdim=True).clamp(min=1.0)
    probs = num_q16 / denom
    return probs.to(scores.dtype)


def qdq_kv_tensor(x: torch.Tensor, mode: str, group_size: int = 64) -> torch.Tensor:
    if mode == "none":
        return x
    if mode == "per_kv_head":
        # 最后一维正好是 head_dim，此处等价于“每个 token、每个 kv head 单独一份 scale”。
        return qdq_lastdim(x, mode="per_token")
    flat = x.reshape(*x.shape[:-2], -1)
    if mode == "per_row":
        # 把一个 token 的完整 kv_dim 当作一整行，模拟 deploy 里按行存 KV_MAIN 的方案。
        flat = qdq_lastdim(flat, mode="per_token")
        return flat.reshape_as(x)
    if mode == "group64":
        flat = qdq_lastdim(flat, mode="group", group_size=group_size)
        return flat.reshape_as(x)
    raise ValueError(mode)


def qdq_qk_scores(scores: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return scores
    if mode == "mid16_token":
        return qdq_mid(scores, mode="int16_token")
    if mode == "mid16_layer":
        return qdq_mid(scores, mode="int16_layer")
    raise ValueError(mode)


def make_attention_forward(
    module: Attention,
    use_approx_softmax: bool,
    *,
    kv_cache_mode: str = "none",
    qk_score_mode: str = "none",
    av_out_mode: str = "none",
    group_size: int = 64,
):
    def forward(self, x, freqs_cos, freqs_sin):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
        # KV cache 量化实验只在写入 cache 的边界注入，避免把误差和别的 attention 线性层混在一起。
        xk = qdq_kv_tensor(xk, kv_cache_mode, group_size=group_size)
        xv = qdq_kv_tensor(xv, kv_cache_mode, group_size=group_size)
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(torch.ones((seqlen, seqlen), device=scores.device, dtype=torch.bool), diagonal=1)
        if qk_score_mode != "none":
            finite_scores = scores.masked_fill(causal_mask.view(1, 1, seqlen, seqlen), 0.0)
            finite_scores = qdq_qk_scores(finite_scores, qk_score_mode)
            scores = finite_scores.masked_fill(causal_mask.view(1, 1, seqlen, seqlen), float("-inf"))
        else:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=scores.device)
            mask = torch.triu(mask, diagonal=1)
            scores = scores + mask
        if use_approx_softmax:
            probs = approx_softmax_scores(scores.float())
        else:
            probs = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(probs, xv)
        output = qdq_kv_tensor(output, av_out_mode, group_size=group_size)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output

    return MethodType(forward, module)


KV_QKAV_SCHEME_SPECS = OrderedDict({
    "kv_cache_head": {
        "kv_cache_mode": "per_kv_head",
        "qk_score_mode": "none",
        "av_out_mode": "none",
        "group_size": 64,
        "family": "kv_only",
    },
    "kv_cache_row": {
        "kv_cache_mode": "per_row",
        "qk_score_mode": "none",
        "av_out_mode": "none",
        "group_size": 64,
        "family": "kv_only",
    },
    "kv_cache_group64": {
        "kv_cache_mode": "group64",
        "qk_score_mode": "none",
        "av_out_mode": "none",
        "group_size": 64,
        "family": "kv_only",
    },
    "kv_av_head": {
        "kv_cache_mode": "per_kv_head",
        "qk_score_mode": "none",
        "av_out_mode": "per_kv_head",
        "group_size": 64,
        "family": "kv_av",
    },
    "kv_av_row": {
        "kv_cache_mode": "per_row",
        "qk_score_mode": "none",
        "av_out_mode": "per_row",
        "group_size": 64,
        "family": "kv_av",
    },
    "kv_av_group64": {
        "kv_cache_mode": "group64",
        "qk_score_mode": "none",
        "av_out_mode": "group64",
        "group_size": 64,
        "family": "kv_av",
    },
    "kv_qkav_head": {
        "kv_cache_mode": "per_kv_head",
        "qk_score_mode": "mid16_token",
        "av_out_mode": "per_kv_head",
        "group_size": 64,
        "family": "kv_qkav",
    },
    "kv_qkav_row": {
        "kv_cache_mode": "per_row",
        "qk_score_mode": "mid16_token",
        "av_out_mode": "per_row",
        "group_size": 64,
        "family": "kv_qkav",
    },
    "kv_qkav_group64": {
        "kv_cache_mode": "group64",
        "qk_score_mode": "mid16_token",
        "av_out_mode": "group64",
        "group_size": 64,
        "family": "kv_qkav",
    },
})


def kv_layout_summary(model: nn.Module, spec) -> dict:
    if spec is None:
        return {}
    attn = model.layers[0].attention
    kv_dim = attn.n_local_kv_heads * attn.head_dim
    n_layers = len(model.layers)
    seq_len = model.params.max_seq_len
    if spec["kv_cache_mode"] == "per_kv_head":
        scales_per_token_per_layer_per_tensor = attn.n_local_kv_heads
        layout_complexity = "中"
    elif spec["kv_cache_mode"] == "per_row":
        scales_per_token_per_layer_per_tensor = 1
        layout_complexity = "低"
    elif spec["kv_cache_mode"] == "group64":
        scales_per_token_per_layer_per_tensor = math.ceil(kv_dim / spec["group_size"])
        layout_complexity = "高"
    else:
        scales_per_token_per_layer_per_tensor = 0
        layout_complexity = "无"
    scales_per_token_per_layer = scales_per_token_per_layer_per_tensor * 2
    total_scale_values = scales_per_token_per_layer * n_layers * seq_len
    return {
        "family": spec["family"],
        "kv_cache_mode": spec["kv_cache_mode"],
        "qk_score_mode": spec["qk_score_mode"],
        "av_out_mode": spec["av_out_mode"],
        "kv_dim": kv_dim,
        "n_layers": n_layers,
        "seq_len": seq_len,
        "scales_per_token_per_layer": scales_per_token_per_layer,
        "scale_region_bytes_float32": int(total_scale_values * 4),
        "kv_data_bytes_int8": int(kv_dim * n_layers * seq_len * 2),
        "layout_complexity": layout_complexity,
    }


def build_kv_qkav_model(scheme: str, ckpt_path: str, tokenizer: Tokenizer, device: str, linear_scales, rms_scales):
    spec = KV_QKAV_SCHEME_SPECS[scheme]
    model = build_model("codesign_w_row_a_token_dyn", ckpt_path, tokenizer, device, linear_scales, rms_scales)
    for module in model.modules():
        if isinstance(module, Attention):
            module.forward = make_attention_forward(
                module,
                use_approx_softmax=False,
                kv_cache_mode=spec["kv_cache_mode"],
                qk_score_mode=spec["qk_score_mode"],
                av_out_mode=spec["av_out_mode"],
                group_size=spec["group_size"],
            )
    return model


def build_scheme_list(scheme_set: str):
    if scheme_set == "kv_qk_av":
        # 保留 float 与已有 attention 线性量化基线，便于直接判断新增误差是否来自 KV/QK/AV 注入点。
        return [
            "float",
            "codesign_w_row_a_token_dyn",
            "coarse_attn_row_token",
            *KV_QKAV_SCHEME_SPECS.keys(),
        ]
    return [
        "float",
        "runq_g64_dyn",
        "static_w_tensor_a_layer",
        "static_w_row_a_layer",
        "codesign_w_row_a_token_dyn",
        "coarse_attn_row_token",
        "coarse_attn_row_static",
        "coarse_attn_ffn13_row_token",
        "coarse_attn_ffn13_row_static",
        "coarse_attn_ffn2_row_static",
        "coarse_ffn_w2_row_token",
        "coarse_ffn_w2_row_static",
        "coarse_attn_ffn13_row_token_mid16",
        "hybrid_attn_token_ffn2_static",
        "coarse_all_linear_no_cls_row_static",
        "coarse_all_linear_no_cls_row_token",
        "coarse_all_linear_no_cls_row_token_mid16",
        "coarse_all_linear_with_cls_row_static",
        "coarse_all_linear_with_cls_row_token",
        "hw_proxy_row_static_softmax_only",
        "hw_proxy_codesign_row_token_softmax_only",
        "hw_proxy_row_static_norm",
        "hw_proxy_row_static_norm_softmax",
    ]


def resolve_eval_paths(args):
    if args.ckpt_path:
        ckpt_path = args.ckpt_path
        tok_path = args.tokenizer_path
        if not tok_path:
            raise ValueError("使用 --ckpt-path 时必须同时提供 --tokenizer-path")
        return ckpt_path, tok_path
    ensure_artifacts(args.quant_eval_dir)
    return (
        os.path.join(args.quant_eval_dir, "stories260K.pt"),
        os.path.join(args.quant_eval_dir, "tok512.model"),
    )


def attach_scheme_metadata(results, models):
    for scheme, scheme_result in results.items():
        spec = KV_QKAV_SCHEME_SPECS.get(scheme)
        if spec is None:
            continue
        scheme_result["layout"] = kv_layout_summary(models[scheme], spec)
        scheme_result["quant_focus"] = {
            "kv_cache": spec["kv_cache_mode"],
            "qk_score": spec["qk_score_mode"],
            "av_out": spec["av_out_mode"],
        }


def summarize_best_kv_scheme(results):
    candidates = []
    for scheme, scheme_result in results.items():
        if scheme not in KV_QKAV_SCHEME_SPECS:
            continue
        agg = scheme_result.get("aggregate", {})
        logits = scheme_result.get("logits", {})
        layout = scheme_result.get("layout", {})
        candidates.append({
            "scheme": scheme,
            "avg_token_match_ratio": agg.get("avg_token_match_ratio", 0.0),
            "avg_prefix_match_tokens": agg.get("avg_prefix_match_tokens", 0.0),
            "cosine": logits.get("cosine", 0.0),
            "loss_delta": logits.get("loss_delta", 0.0),
            "scale_region_bytes_float32": layout.get("scale_region_bytes_float32", 0),
            "layout_complexity": layout.get("layout_complexity", "无"),
        })
    candidates.sort(key=lambda item: (item["avg_token_match_ratio"], item["cosine"], -abs(item["loss_delta"])), reverse=True)
    return candidates


def recommend_scale_scheme(results):
    ranked = summarize_best_kv_scheme(results)
    if not ranked:
        return {}
    best = ranked[0]
    reason = "优先按 avg_token_match_ratio 与 cosine 排序"
    return {
        "recommended_scheme": best["scheme"],
        "reason": reason,
        "top_candidates": ranked[:3],
    }


def should_try_qat(results, token_match_threshold: float = 0.80, cosine_threshold: float = 0.999):
    family_scores = {}
    for scheme, scheme_result in results.items():
        spec = KV_QKAV_SCHEME_SPECS.get(scheme)
        if spec is None:
            continue
        family = spec["family"]
        agg = scheme_result.get("aggregate", {})
        logits = scheme_result.get("logits", {})
        score = (
            agg.get("avg_token_match_ratio", 0.0),
            logits.get("cosine", 0.0),
            -abs(logits.get("loss_delta", 0.0)),
        )
        if family not in family_scores or score > family_scores[family][0]:
            family_scores[family] = (score, scheme_result, scheme)
    kv_qkav = family_scores.get("kv_qkav")
    if kv_qkav is None:
        return {"need_qat": False, "reason": "未运行 kv_qkav 方案"}
    agg = kv_qkav[1].get("aggregate", {})
    logits = kv_qkav[1].get("logits", {})
    need_qat = agg.get("avg_token_match_ratio", 0.0) < token_match_threshold or logits.get("cosine", 0.0) < cosine_threshold
    return {
        "need_qat": need_qat,
        "best_kv_qkav_scheme": kv_qkav[2],
        "avg_token_match_ratio": agg.get("avg_token_match_ratio", 0.0),
        "cosine": logits.get("cosine", 0.0),
        "reason": "若 KV+QK+AV 组合下 token 漂移或 cosine 明显下滑，再进入 attention 定向 QAT",
    }


def build_overall_summary(results):
    return {
        "scale_recommendation": recommend_scale_scheme(results),
        "qat_decision": should_try_qat(results),
    }


def pick_build_model_fn(scheme: str):
    if scheme in KV_QKAV_SCHEME_SPECS:
        return build_kv_qkav_model
    return build_model




#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


#


# continue


def collect_scales(model: nn.Module, tokenizer: Tokenizer, device: str):
    linear_in_max = {}
    rms_in_max = {}
    hooks = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            def make_linear_hook(key):
                def hook(_mod, inputs):
                    x = inputs[0].detach().float()
                    linear_in_max[key] = max(linear_in_max.get(key, 0.0), float(x.abs().max().item()))
                return hook

            hooks.append(module.register_forward_pre_hook(make_linear_hook(name)))

        if module.__class__.__name__ == "RMSNorm":
            def make_norm_hook(key):
                def hook(_mod, inputs):
                    x = inputs[0].detach().float()
                    rms_in_max[key] = max(rms_in_max.get(key, 0.0), float(x.abs().max().item()))
                return hook

            hooks.append(module.register_forward_pre_hook(make_norm_hook(name)))

    for prompt in CALIB_PROMPTS:
        toks = tokenizer.encode(prompt, bos=True, eos=False)
        x = torch.tensor(toks, dtype=torch.long, device=device)[None, ...]
        with torch.no_grad():
            _ = model(x, targets=x)

    for hook in hooks:
        hook.remove()

    linear_scales = {k: max(v / 127.0, 1e-8) for k, v in linear_in_max.items()}
    rms_scales = {k: max(v / 32767.0, 1e-8) for k, v in rms_in_max.items()}
    return linear_scales, rms_scales


def replace_module(root: nn.Module, name: str, module: nn.Module) -> None:
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], module)


def selective_quantize_model(
    model: nn.Module,
    linear_scales,
    *,
    select_linears,
    select_output=False,
    weight_mode="per_row",
    act_mode="dynamic_token",
    out_mode="none",
):
    shared_weight = model.output.weight.detach().float()
    if select_output:
        model.output = FakeQuantOutput(
            qdq_weight(shared_weight, mode=weight_mode),
            act_mode=act_mode,
            act_scale=linear_scales.get("output"),
            out_mode=out_mode,
        )

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if name == "output":
            continue
        if not select_linears(name):
            continue
        replace_module(
            model,
            name,
            FakeQuantLinear(
                module,
                qdq_weight(module.weight, mode=weight_mode),
                act_mode=act_mode,
                act_scale=linear_scales.get(name),
                out_mode=out_mode,
            ),
        )
    return model


def selective_quantize_model_mixed(
    model: nn.Module,
    linear_scales,
    *,
    module_configs,
):
    shared_weight = model.output.weight.detach().float()
    output_cfg = module_configs.get("output")
    if output_cfg is not None:
        model.output = FakeQuantOutput(
            qdq_weight(shared_weight, mode=output_cfg["weight_mode"]),
            act_mode=output_cfg["act_mode"],
            act_scale=linear_scales.get("output"),
            out_mode=output_cfg.get("out_mode", "none"),
        )

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if name == "output":
            continue
        cfg = module_configs.get(name)
        if cfg is None:
            continue
        replace_module(
            model,
            name,
            FakeQuantLinear(
                module,
                qdq_weight(module.weight, mode=cfg["weight_mode"]),
                act_mode=cfg["act_mode"],
                act_scale=linear_scales.get(name),
                out_mode=cfg.get("out_mode", "none"),
            ),
        )
    return model


def op_group_selector(groups):
    groups = set(groups)

    def select(name: str) -> bool:
        if "attn" in groups and ".attention." in name:
            return True
        if "ffn_in" in groups and (name.endswith("feed_forward.w1") or name.endswith("feed_forward.w3")):
            return True
        if "ffn_out" in groups and name.endswith("feed_forward.w2"):
            return True
        return False

    return select


def build_model(scheme: str, ckpt_path: str, tokenizer: Tokenizer, device: str, linear_scales, rms_scales):
    model, _ = load_model(ckpt_path, device)
    shared_weight = model.output.weight.detach().float()

    if scheme == "float":
        return model

    if scheme == "runq_g64_dyn":
        shared_qdq = qdq_weight(shared_weight, mode="group", group_size=64)
        model.tok_embeddings = FakeQuantEmbedding(shared_qdq)
        model.output = FakeQuantOutput(shared_qdq, act_mode="dynamic_group", group_size=64)
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear) and name != "output":
                parts = name.split(".")
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], FakeQuantLinear(module, qdq_weight(module.weight, mode="group", group_size=64), act_mode="dynamic_group", group_size=64))
        return model

    if scheme == "static_w_tensor_a_layer":
        shared_qdq = qdq_weight(shared_weight, mode="per_tensor")
        model.tok_embeddings = FakeQuantEmbedding(shared_qdq)
        model.output = FakeQuantOutput(shared_qdq, act_mode="static_layer", act_scale=linear_scales["output"])
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear) and name != "output":
                parts = name.split(".")
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], FakeQuantLinear(module, qdq_weight(module.weight, mode="per_tensor"), act_mode="static_layer", act_scale=linear_scales[name]))
        return model

    if scheme == "static_w_row_a_layer":
        shared_qdq = qdq_weight(shared_weight, mode="per_row")
        model.tok_embeddings = FakeQuantEmbedding(shared_qdq)
        model.output = FakeQuantOutput(shared_qdq, act_mode="static_layer", act_scale=linear_scales["output"])
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear) and name != "output":
                parts = name.split(".")
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], FakeQuantLinear(module, qdq_weight(module.weight, mode="per_row"), act_mode="static_layer", act_scale=linear_scales[name]))
        return model

    if scheme == "codesign_w_row_a_token_dyn":
        shared_qdq = qdq_weight(shared_weight, mode="per_row")
        model.tok_embeddings = FakeQuantEmbedding(shared_qdq)
        model.output = FakeQuantOutput(shared_qdq, act_mode="dynamic_token")
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear) and name != "output":
                parts = name.split(".")
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], FakeQuantLinear(module, qdq_weight(module.weight, mode="per_row"), act_mode="dynamic_token"))
        return model

    if scheme == "coarse_attn_row_token":
        return selective_quantize_model(
            model,
            linear_scales,
            select_linears=op_group_selector(["attn"]),
            weight_mode="per_row",
            act_mode="dynamic_token",
            out_mode="none",
        )

    if scheme == "coarse_attn_row_static":
        return selective_quantize_model(
            model,
            linear_scales,
            select_linears=op_group_selector(["attn"]),
            weight_mode="per_row",
            act_mode="static_layer",
            out_mode="none",
        )

    if scheme == "coarse_attn_ffn13_row_token":
        return selective_quantize_model(
            model,
            linear_scales,
            select_linears=op_group_selector(["attn", "ffn_in"]),
            weight_mode="per_row",
            act_mode="dynamic_token",
            out_mode="none",
        )

    if scheme == "coarse_attn_ffn13_row_static":
        return selective_quantize_model(
            model,
            linear_scales,
            select_linears=op_group_selector(["attn", "ffn_in"]),
            weight_mode="per_row",
            act_mode="static_layer",
            out_mode="none",
        )

    if scheme == "coarse_attn_ffn2_row_static":
        return selective_quantize_model(
            model,
            linear_scales,
            select_linears=op_group_selector(["attn", "ffn_out"]),
            weight_mode="per_row",
            act_mode="static_layer",
            out_mode="none",
        )

    if scheme == "coarse_ffn_w2_row_token":
        return selective_quantize_model(
            model,
            linear_scales,
            select_linears=op_group_selector(["ffn_out"]),
            weight_mode="per_row",
            act_mode="dynamic_token",
            out_mode="none",
        )

    if scheme == "coarse_ffn_w2_row_static":
        return selective_quantize_model(
            model,
            linear_scales,
            select_linears=op_group_selector(["ffn_out"]),
            weight_mode="per_row",
            act_mode="static_layer",
            out_mode="none",
        )

    if scheme == "coarse_attn_ffn13_row_token_mid16":
        return selective_quantize_model(
            model,
            linear_scales,
            select_linears=op_group_selector(["attn", "ffn_in"]),
            weight_mode="per_row",
            act_mode="dynamic_token",
            out_mode="int16_token",
        )

    if scheme == "hybrid_attn_token_ffn2_static":
        module_configs = {}
        for name, module in list(model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if name == "output":
                continue
            if ".attention." in name:
                module_configs[name] = {
                    "weight_mode": "per_row",
                    "act_mode": "dynamic_token",
                }
            elif name.endswith("feed_forward.w2"):
                module_configs[name] = {
                    "weight_mode": "per_row",
                    "act_mode": "static_layer",
                }
        return selective_quantize_model_mixed(
            model,
            linear_scales,
            module_configs=module_configs,
        )

    if scheme == "coarse_all_linear_no_cls_row_static":
        return selective_quantize_model(
            model,
            linear_scales,
            select_linears=lambda _name: True,
            weight_mode="per_row",
            act_mode="static_layer",
            out_mode="none",
        )

    if scheme == "coarse_all_linear_no_cls_row_token":
        return selective_quantize_model(
            model,
            linear_scales,
            select_linears=lambda _name: True,
            weight_mode="per_row",
            act_mode="dynamic_token",
            out_mode="none",
        )

    if scheme == "coarse_all_linear_no_cls_row_token_mid16":
        return selective_quantize_model(
            model,
            linear_scales,
            select_linears=lambda _name: True,
            weight_mode="per_row",
            act_mode="dynamic_token",
            out_mode="int16_token",
        )

    if scheme == "coarse_all_linear_with_cls_row_static":
        return selective_quantize_model(
            model,
            linear_scales,
            select_linears=lambda _name: True,
            select_output=True,
            weight_mode="per_row",
            act_mode="static_layer",
            out_mode="none",
        )

    if scheme == "coarse_all_linear_with_cls_row_token":
        return selective_quantize_model(
            model,
            linear_scales,
            select_linears=lambda _name: True,
            select_output=True,
            weight_mode="per_row",
            act_mode="dynamic_token",
            out_mode="none",
        )

    if scheme == "hw_proxy_row_static_norm":
        model = build_model("static_w_row_a_layer", ckpt_path, tokenizer, device, linear_scales, rms_scales)
        for name, module in list(model.named_modules()):
            if module.__class__.__name__ == "RMSNorm":
                parts = name.split(".")
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], ApproxRMSNorm(module, rms_scales[name]))
        return model

    if scheme == "hw_proxy_row_static_softmax_only":
        model = build_model("static_w_row_a_layer", ckpt_path, tokenizer, device, linear_scales, rms_scales)
        for module in model.modules():
            if isinstance(module, Attention):
                module.forward = make_attention_forward(module, use_approx_softmax=True)
        return model

    if scheme == "hw_proxy_codesign_row_token_softmax_only":
        model = build_model("codesign_w_row_a_token_dyn", ckpt_path, tokenizer, device, linear_scales, rms_scales)
        for module in model.modules():
            if isinstance(module, Attention):
                module.forward = make_attention_forward(module, use_approx_softmax=True)
        return model

    if scheme == "hw_proxy_row_static_norm_softmax":
        model = build_model("hw_proxy_row_static_norm", ckpt_path, tokenizer, device, linear_scales, rms_scales)
        for module in model.modules():
            if isinstance(module, Attention):
                module.forward = make_attention_forward(module, use_approx_softmax=True)
        return model

    raise ValueError(scheme)


def evaluate_scheme(model: nn.Module, tokenizer: Tokenizer, base_logits: torch.Tensor, base_loss: float):
    def token_ids(prompt: str):
        return tokenizer.encode(prompt, bos=True, eos=False)

    def generate_ids(prompt: str, max_new=80):
        ids = token_ids(prompt)
        x = torch.tensor(ids, dtype=torch.long, device=base_logits.device)[None, ...]
        with torch.no_grad():
            y = model.generate(x, max_new, temperature=0.0)
        return y[0].tolist()

    long_ids = token_ids(LONG_TEST_TEXT)
    long_x = torch.tensor(long_ids, dtype=torch.long, device=base_logits.device)[None, ...]
    with torch.no_grad():
        logits = model(long_x, targets=long_x).float()
    diff = logits - base_logits
    res = {
        "logits": {
            "mse": float((diff * diff).mean().item()),
            "mae": float(diff.abs().mean().item()),
            "cosine": float(F.cosine_similarity(base_logits.reshape(1, -1), logits.reshape(1, -1)).item()),
            "loss": float(model.last_loss.item()),
            "loss_delta": float(model.last_loss.item() - base_loss),
        },
        "prompts": [],
    }
    for prompt in TEST_PROMPTS:
        q_ids = generate_ids(prompt)
        res["prompts"].append({"prompt": prompt, "ids": q_ids, "text": tokenizer.decode(q_ids)})
    return res


def add_prompt_comparisons(results):
    base_prompts = {entry["prompt"]: entry for entry in results["float"]["prompts"]}
    for scheme, scheme_result in results.items():
        if scheme == "float":
            continue
        for entry in scheme_result["prompts"]:
            prompt = entry["prompt"]
            base_ids = base_prompts[prompt]["ids"]
            prompt_len = len(base_ids) - 80
            gen_b = base_ids[prompt_len:]
            gen_q = entry["ids"][prompt_len:]
            prefix = 0
            for a, b in zip(gen_b, gen_q):
                if a == b:
                    prefix += 1
                else:
                    break
            total_match = sum(int(a == b) for a, b in zip(gen_b, gen_q)) / max(1, len(gen_b))
            entry["prefix_match_tokens"] = prefix
            entry["token_match_ratio"] = total_match
            entry["baseline_text_prefix"] = base_prompts[prompt]["text"][:220]
            entry["quant_text_prefix"] = entry["text"][:220]


def add_scheme_aggregates(results):
    for scheme, scheme_result in results.items():
        prompts = scheme_result["prompts"]
        if not prompts:
            continue
        scheme_result["aggregate"] = {
            "avg_prefix_match_tokens": float(sum(p.get("prefix_match_tokens", 0) for p in prompts) / len(prompts)),
            "avg_token_match_ratio": float(sum(p.get("token_match_ratio", 0.0) for p in prompts) / len(prompts)),
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate float, runq-style, and hardware-friendly quantization schemes.")
    parser.add_argument("--quant-eval-dir", default=os.path.join(REPO_DIR, "quant_eval"))
    parser.add_argument("--ckpt-path", default="", help="显式指定 checkpoint；为空时沿用 quant_eval 目录默认资产。")
    parser.add_argument("--tokenizer-path", default="", help="与 --ckpt-path 配套使用的 tokenizer.model 路径。")
    parser.add_argument("--scheme-set", choices=["full", "kv_qk_av"], default="full")
    parser.add_argument("--output", default=os.path.join(REPO_DIR, "artifacts", "quant_eval_report.json"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    ckpt_path, tok_path = resolve_eval_paths(args)
    tokenizer = Tokenizer(tokenizer_model=tok_path)

    base_model, _ = load_model(ckpt_path, args.device)
    linear_scales, rms_scales = collect_scales(base_model, tokenizer, args.device)

    schemes = build_scheme_list(args.scheme_set)

    models = OrderedDict()
    for scheme in schemes:
        build_fn = pick_build_model_fn(scheme)
        models[scheme] = build_fn(scheme, ckpt_path, tokenizer, args.device, linear_scales, rms_scales)

    long_ids = tokenizer.encode(LONG_TEST_TEXT, bos=True, eos=False)
    long_x = torch.tensor(long_ids, dtype=torch.long, device=args.device)[None, ...]
    with torch.no_grad():
        base_logits = models["float"](long_x, targets=long_x).float()
    base_loss = float(models["float"].last_loss.item())

    results = OrderedDict()
    for scheme, model in models.items():
        results[scheme] = evaluate_scheme(model, tokenizer, base_logits, base_loss)

    add_prompt_comparisons(results)
    add_scheme_aggregates(results)
    attach_scheme_metadata(results, models)

    report = {
        "checkpoint": ckpt_path,
        "tokenizer": tok_path,
        "scheme_set": args.scheme_set,
        "base_loss": base_loss,
        "long_test_text": LONG_TEST_TEXT,
        "calibration_prompts": CALIB_PROMPTS,
        "test_prompts": TEST_PROMPTS,
        "linear_scales": linear_scales,
        "rms_scales": rms_scales,
        "results": results,
        "summary": build_overall_summary(results),
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "checkpoint": ckpt_path,
        "scheme_set": args.scheme_set,
        "base_loss": base_loss,
        "schemes": {
            name: {
                "loss": res["logits"]["loss"],
                "loss_delta": res["logits"]["loss_delta"],
                "cosine": res["logits"]["cosine"],
                "avg_token_match_ratio": res.get("aggregate", {}).get("avg_token_match_ratio"),
            }
            for name, res in results.items()
        },
        "summary": report["summary"],
        "output": args.output,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
