"""Forward do Qwen3-MoE com experts vindos de fora.

Atenção, normas, embedding e ``lm_head`` ficam residentes. O gate
original não roda. Cada camada recebe só os experts quentes daquele
token; se a máscara não escolheu nenhum, o MoE soma zero e o residual
segue. É a máscara, não um fallback.

O desenho acompanha o Qwen3-MoE: RMSNorm, QK-norm, RoPE, GQA e SwiGLU.
Não é o bloco do ``transformers`` carregado inteiro — o expert não mora
no módulo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from moe_memory.offload.format import DTYPE_BF16, DTYPE_F16, DTYPE_F32
from moe_memory.offload.store import ResidentReader, tensor_from_raw, torch_dtype


@dataclass(frozen=True)
class QwenSpec:
    vocab_size: int
    hidden_size: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    moe_intermediate: int
    n_experts: int
    experts_per_tok: int
    rms_eps: float
    rope_theta: float
    dtype_code: int

    @property
    def dtype(self) -> torch.dtype:
        return torch_dtype(self.dtype_code)


@dataclass
class LayerWeights:
    input_norm: torch.Tensor
    post_norm: torch.Tensor
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    o: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    q_bias: torch.Tensor | None = None
    k_bias: torch.Tensor | None = None
    v_bias: torch.Tensor | None = None
    o_bias: torch.Tensor | None = None


@dataclass
class QwenWeights:
    spec: QwenSpec
    embed: torch.Tensor
    final_norm: torch.Tensor
    lm_head: torch.Tensor
    layers: list[LayerWeights]

    def nbytes(self) -> int:
        total = self.embed.nbytes + self.final_norm.nbytes + self.lm_head.nbytes
        for layer in self.layers:
            for name in ("input_norm", "post_norm", "q", "k", "v", "o", "q_norm", "k_norm"):
                total += getattr(layer, name).nbytes
            for name in ("q_bias", "k_bias", "v_bias", "o_bias"):
                bias = getattr(layer, name)
                if bias is not None:
                    total += bias.nbytes
        return int(total)


class KvCache:
    def __init__(self, n_layers: int) -> None:
        self.pairs: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * n_layers


def load_spec(config_path: Path, dtype_code: int) -> QwenSpec:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    hidden = int(data["hidden_size"])
    n_heads = int(data["num_attention_heads"])
    n_kv = int(data["num_key_value_heads"])
    head_dim = int(data["head_dim"]) if data.get("head_dim") else hidden // n_heads
    if n_heads % n_kv != 0:
        raise ValueError("num_attention_heads precisa ser múltiplo de num_key_value_heads")
    if head_dim % 2 != 0:
        raise ValueError("head_dim do RoPE precisa ser par")
    return QwenSpec(
        vocab_size=int(data["vocab_size"]),
        hidden_size=hidden,
        n_layers=int(data["num_hidden_layers"]),
        n_heads=n_heads,
        n_kv_heads=n_kv,
        head_dim=head_dim,
        moe_intermediate=int(data.get("moe_intermediate_size", data["intermediate_size"])),
        n_experts=int(data["num_experts"]),
        experts_per_tok=int(data.get("num_experts_per_tok", 8)),
        rms_eps=float(data.get("rms_norm_eps", 1e-6)),
        rope_theta=float(data.get("rope_theta", 1_000_000.0)),
        dtype_code=dtype_code,
    )


def load_resident(path: Path, spec: QwenSpec, device: torch.device) -> QwenWeights:
    blobs: dict[str, torch.Tensor] = {}
    with ResidentReader(path) as reader:
        for name, dtype_code, shape, raw in reader:
            blobs[name] = tensor_from_raw(raw, shape, torch_dtype(dtype_code)).to(device=device)
    tie = "lm_head.weight" not in blobs
    required = ["model.embed_tokens.weight", "model.norm.weight"]
    if not tie:
        required.append("lm_head.weight")
    for layer in range(spec.n_layers):
        prefix = f"model.layers.{layer}"
        required.extend(
            [
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.self_attn.q_norm.weight",
                f"{prefix}.self_attn.k_norm.weight",
            ]
        )
    missing = [name for name in required if name not in blobs]
    if missing:
        raise FileNotFoundError(f"peso residente ausente: {missing[0]}")
    embed = blobs["model.embed_tokens.weight"]
    lm_head = embed if tie else blobs["lm_head.weight"]
    layers: list[LayerWeights] = []
    q_out = spec.n_heads * spec.head_dim
    kv_out = spec.n_kv_heads * spec.head_dim
    for layer in range(spec.n_layers):
        prefix = f"model.layers.{layer}"
        built = LayerWeights(
            input_norm=_shaped(blobs, f"{prefix}.input_layernorm.weight", (spec.hidden_size,)),
            post_norm=_shaped(blobs, f"{prefix}.post_attention_layernorm.weight", (spec.hidden_size,)),
            q=_shaped(blobs, f"{prefix}.self_attn.q_proj.weight", (q_out, spec.hidden_size)),
            k=_shaped(blobs, f"{prefix}.self_attn.k_proj.weight", (kv_out, spec.hidden_size)),
            v=_shaped(blobs, f"{prefix}.self_attn.v_proj.weight", (kv_out, spec.hidden_size)),
            o=_shaped(blobs, f"{prefix}.self_attn.o_proj.weight", (spec.hidden_size, q_out)),
            q_norm=_shaped(blobs, f"{prefix}.self_attn.q_norm.weight", (spec.head_dim,)),
            k_norm=_shaped(blobs, f"{prefix}.self_attn.k_norm.weight", (spec.head_dim,)),
            q_bias=blobs.get(f"{prefix}.self_attn.q_proj.bias"),
            k_bias=blobs.get(f"{prefix}.self_attn.k_proj.bias"),
            v_bias=blobs.get(f"{prefix}.self_attn.v_proj.bias"),
            o_bias=blobs.get(f"{prefix}.self_attn.o_proj.bias"),
        )
        layers.append(built)
    _shaped_embed = _shaped(blobs, "model.embed_tokens.weight", (spec.vocab_size, spec.hidden_size))
    return QwenWeights(
        spec=spec,
        embed=_shaped_embed,
        final_norm=_shaped(blobs, "model.norm.weight", (spec.hidden_size,)),
        lm_head=lm_head if tie else _shaped(blobs, "lm_head.weight", (spec.vocab_size, spec.hidden_size)),
        layers=layers,
    )


def forward_token(
    weights: QwenWeights,
    cache: KvCache,
    token_id: int,
    position: int,
    before_moe,
    on_span=None,
) -> tuple[torch.Tensor, int]:
    """Um token. ``before_moe(camada)`` devolve os experts quentes e pode bloquear.

    ``on_span(nome, "begin"|"end")`` é opt-in. O intervalo ``expert`` cobre
    só ``apply_experts``; a espera do ensure fica em ``before_moe``.
    """

    spec = weights.spec
    if not 0 <= token_id < spec.vocab_size:
        raise ValueError(f"token {token_id} fora do vocabulário {spec.vocab_size}")
    hidden = weights.embed[token_id].view(1, 1, -1)
    executed = 0
    for index, layer in enumerate(weights.layers):
        normed = rms_norm(hidden, layer.input_norm, spec.rms_eps)
        if on_span is not None:
            on_span("attention", "begin")
        hidden = hidden + _attention(layer, normed, cache, index, position, spec)
        if on_span is not None:
            on_span("attention", "end")
        calls = before_moe(index)
        executed += len(calls)
        if on_span is not None:
            on_span("expert", "begin")
        hidden = hidden + apply_experts(rms_norm(hidden, layer.post_norm, spec.rms_eps), calls)
        if on_span is not None:
            on_span("expert", "end")
    logits = F.linear(rms_norm(hidden, weights.final_norm, spec.rms_eps), weights.lm_head)
    return logits.view(1, -1), executed


def apply_experts(
    hidden: torch.Tensor,
    calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    flat = hidden.reshape(hidden.shape[0] * hidden.shape[1], hidden.shape[-1])
    acc = torch.zeros_like(flat)
    for gate_w, up_w, down_w, coeff in calls:
        mid = F.silu(F.linear(flat, gate_w)) * F.linear(flat, up_w)
        acc = acc + F.linear(mid, down_w) * coeff.to(dtype=acc.dtype, device=acc.device)
    return acc.view_as(hidden)


def rms_norm(hidden: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    in_dtype = hidden.dtype
    squared = hidden.float().pow(2).mean(dim=-1, keepdim=True)
    normed = hidden.float() * torch.rsqrt(squared + eps)
    return (weight * normed.to(in_dtype)).to(in_dtype)


def _attention(
    layer: LayerWeights,
    hidden: torch.Tensor,
    cache: KvCache,
    index: int,
    position: int,
    spec: QwenSpec,
) -> torch.Tensor:
    batch, seq, _width = hidden.shape
    if seq != 1 or batch != 1:
        raise ValueError("este forward decodifica um token, batch 1")
    query = F.linear(hidden, layer.q, layer.q_bias).view(batch, seq, spec.n_heads, spec.head_dim)
    key = F.linear(hidden, layer.k, layer.k_bias).view(batch, seq, spec.n_kv_heads, spec.head_dim)
    value = F.linear(hidden, layer.v, layer.v_bias).view(batch, seq, spec.n_kv_heads, spec.head_dim)
    query = rms_norm(query, layer.q_norm, spec.rms_eps)
    key = rms_norm(key, layer.k_norm, spec.rms_eps)
    cos, sin = _rope(position, spec.head_dim, spec.rope_theta, hidden.device, hidden.dtype)
    query = _apply_rope(query.transpose(1, 2), cos, sin)
    key = _apply_rope(key.transpose(1, 2), cos, sin)
    value = value.transpose(1, 2)
    previous = cache.pairs[index]
    if previous is None:
        cache.pairs[index] = (key, value)
    else:
        cache.pairs[index] = (
            torch.cat([previous[0], key], dim=2),
            torch.cat([previous[1], value], dim=2),
        )
    full_k, full_v = cache.pairs[index]
    groups = spec.n_heads // spec.n_kv_heads
    full_k = _repeat_kv(full_k, groups)
    full_v = _repeat_kv(full_v, groups)
    mixed = F.scaled_dot_product_attention(query, full_k, full_v, is_causal=False)
    mixed = mixed.transpose(1, 2).reshape(batch, seq, spec.n_heads * spec.head_dim)
    return F.linear(mixed, layer.o, layer.o_bias)


def _rope(position: int, head_dim: int, theta: float, device: torch.device, dtype: torch.dtype):
    half = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    inv = 1.0 / (theta ** (half / head_dim))
    freqs = position * inv
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype)[None, None, None, :], emb.sin().to(dtype)[None, None, None, :]


def _apply_rope(states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return states * cos + _rotate_half(states) * sin


def _rotate_half(states: torch.Tensor) -> torch.Tensor:
    half = states.shape[-1] // 2
    return torch.cat((-states[..., half:], states[..., :half]), dim=-1)


def _repeat_kv(states: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return states
    batch, heads, seq, dim = states.shape
    expanded = states[:, :, None, :, :].expand(batch, heads, groups, seq, dim)
    return expanded.reshape(batch, heads * groups, seq, dim)


def _shaped(blobs: dict[str, torch.Tensor], name: str, shape: tuple[int, ...]) -> torch.Tensor:
    tensor = blobs[name]
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} tem shape {tuple(tensor.shape)}, esperado {shape}")
    return tensor


def dtype_code_of(name: str) -> int:
    table = {"BF16": DTYPE_BF16, "F16": DTYPE_F16, "F32": DTYPE_F32, "bfloat16": DTYPE_BF16, "float16": DTYPE_F16, "float32": DTYPE_F32}
    try:
        return table[name]
    except KeyError as exc:
        raise ValueError(f"dtype {name} não é BF16, F16 ou F32") from exc
