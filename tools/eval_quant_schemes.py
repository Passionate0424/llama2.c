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


def make_attention_forward(module: Attention, use_approx_softmax: bool):
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
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=scores.device)
        mask = torch.triu(mask, diagonal=1)
        scores = scores + mask
        if use_approx_softmax:
            probs = approx_softmax_scores(scores.float())
        else:
            probs = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(probs, xv)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output

    return MethodType(forward, module)


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
    parser.add_argument("--output", default=os.path.join(REPO_DIR, "artifacts", "quant_eval_report.json"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    ensure_artifacts(args.quant_eval_dir)

    ckpt_path = os.path.join(args.quant_eval_dir, "stories260K.pt")
    tok_path = os.path.join(args.quant_eval_dir, "tok512.model")
    tokenizer = Tokenizer(tokenizer_model=tok_path)

    base_model, _ = load_model(ckpt_path, args.device)
    linear_scales, rms_scales = collect_scales(base_model, tokenizer, args.device)

    schemes = [
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

    models = OrderedDict()
    for scheme in schemes:
        models[scheme] = build_model(scheme, ckpt_path, tokenizer, args.device, linear_scales, rms_scales)

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

    report = {
        "base_loss": base_loss,
        "long_test_text": LONG_TEST_TEXT,
        "calibration_prompts": CALIB_PROMPTS,
        "test_prompts": TEST_PROMPTS,
        "linear_scales": linear_scales,
        "rms_scales": rms_scales,
        "results": results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "base_loss": base_loss,
        "schemes": {
            name: {
                "loss": res["logits"]["loss"],
                "loss_delta": res["logits"]["loss_delta"],
                "cosine": res["logits"]["cosine"],
            }
            for name, res in results.items()
        },
        "output": args.output,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
