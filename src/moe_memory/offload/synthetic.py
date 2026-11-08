"""Checkpoint minúsculo no layout do Qwen3, para o teste e o ``--dry-run``.

Não é o 30B. As dimensões cabem na RAM do CI. O bloco do expert e o
arquivo residente seguem o mesmo formato do repack.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from moe_memory.offload.format import DTYPE_F32
from moe_memory.offload.store import ExpertStoreWriter, ResidentWriter, raw_from_tensor


def write_synthetic_bundle(
    out_dir: Path,
    *,
    n_layers: int = 2,
    n_experts: int = 4,
    hidden: int = 32,
    intermediate: int = 16,
    vocab: int = 64,
    n_heads: int = 4,
    n_kv: int = 2,
    head_dim: int = 8,
    seed: int = 0,
) -> Path:
    if min(n_layers, n_experts, hidden, intermediate, vocab, n_heads, n_kv, head_dim) < 1:
        raise ValueError("dimensões sintéticas inválidas")
    if n_heads % n_kv != 0 or head_dim % 2 != 0:
        raise ValueError("cabeças e head_dim não fecham")
    torch.manual_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    order = [(layer, expert) for layer in range(n_layers) for expert in range(n_experts)]
    store = ExpertStoreWriter(
        out_dir / "experts.moestore",
        n_layers=n_layers,
        n_experts=n_experts,
        hidden=hidden,
        intermediate=intermediate,
        dtype_code=DTYPE_F32,
        order=order,
    )
    resident = ResidentWriter(out_dir / "resident.bin")
    try:
        for layer, expert in order:
            gate = torch.randn(intermediate, hidden) * 0.02
            up = torch.randn(intermediate, hidden) * 0.02
            down = torch.randn(hidden, intermediate) * 0.02
            store.write(layer, expert, _raw(gate), _raw(up), _raw(down))
        _write_resident(
            resident,
            n_layers=n_layers,
            n_experts=n_experts,
            hidden=hidden,
            vocab=vocab,
            n_heads=n_heads,
            n_kv=n_kv,
            head_dim=head_dim,
        )
        nbytes = resident.nbytes
        resident.finish()
        store.finish(resident_nbytes=nbytes)
    except Exception:
        store.abort()
        raise
    config = {
        "model_type": "qwen3_moe",
        "vocab_size": vocab,
        "hidden_size": hidden,
        "num_hidden_layers": n_layers,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv,
        "head_dim": head_dim,
        "intermediate_size": intermediate,
        "moe_intermediate_size": intermediate,
        "num_experts": n_experts,
        "num_experts_per_tok": 1,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "torch_dtype": "float32",
        "tie_word_embeddings": False,
    }
    (out_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return out_dir


def _write_resident(resident: ResidentWriter, **spec: int) -> None:
    hidden = spec["hidden"]
    vocab = spec["vocab"]
    n_heads = spec["n_heads"]
    n_kv = spec["n_kv"]
    head_dim = spec["head_dim"]
    n_layers = spec["n_layers"]
    n_experts = spec["n_experts"]
    q_out = n_heads * head_dim
    kv_out = n_kv * head_dim
    _add(resident, "model.embed_tokens.weight", torch.randn(vocab, hidden) * 0.02)
    _add(resident, "model.norm.weight", torch.ones(hidden))
    _add(resident, "lm_head.weight", torch.randn(vocab, hidden) * 0.02)
    for layer in range(n_layers):
        prefix = f"model.layers.{layer}"
        _add(resident, f"{prefix}.input_layernorm.weight", torch.ones(hidden))
        _add(resident, f"{prefix}.post_attention_layernorm.weight", torch.ones(hidden))
        _add(resident, f"{prefix}.self_attn.q_proj.weight", torch.randn(q_out, hidden) * 0.02)
        _add(resident, f"{prefix}.self_attn.k_proj.weight", torch.randn(kv_out, hidden) * 0.02)
        _add(resident, f"{prefix}.self_attn.v_proj.weight", torch.randn(kv_out, hidden) * 0.02)
        _add(resident, f"{prefix}.self_attn.o_proj.weight", torch.randn(hidden, q_out) * 0.02)
        _add(resident, f"{prefix}.self_attn.q_norm.weight", torch.ones(head_dim))
        _add(resident, f"{prefix}.self_attn.k_norm.weight", torch.ones(head_dim))
        _add(resident, f"{prefix}.mlp.gate.weight", torch.randn(n_experts, hidden) * 0.02)


def _add(resident: ResidentWriter, name: str, tensor: torch.Tensor) -> None:
    code, shape, raw = raw_from_tensor(tensor)
    resident.add(name, code, shape, raw)


def _raw(tensor: torch.Tensor) -> bytes:
    _code, _shape, raw = raw_from_tensor(tensor)
    return raw
