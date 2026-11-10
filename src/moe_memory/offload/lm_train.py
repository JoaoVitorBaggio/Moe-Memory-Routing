"""Um passo de loss de linguagem no Qwen com o store, sem o pool de inferência.

O runtime de decode solta a vaga e dá ``detach`` no logit. Aqui o expert
é lido de novo para um tensor privado, a LoRA entra em volta do peso
congelado e o checkpoint de ativação relê o bloco no backward. O store
e o residente não são parâmetros. Treinam o LSTM e a LoRA.

A loss é a entropia cruzada do próximo token. Não há KL, balanceamento
nem penalidade de troca. Batch 1. O compute é float32: a GTX 1060 não
executa bf16.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from moe_memory.lstm_ahead_router import TwoHorizonLstmRouter
from moe_memory.offload.qwen import (
    QwenSpec,
    QwenWeights,
    _apply_rope,
    _repeat_kv,
    rms_norm,
)
from moe_memory.offload.store import ExpertStore, torch_dtype
from moe_memory.qwen3_adapter import language_model_loss

# Teto aprovado. O smoke recusa antes, para sobrar folga de página e de alocador.
CGROUP_BYTES = 9_663_676_416
LAUNCH_CEILING_BYTES = 8 * 1024**3
VRAM_BUDGET_BYTES = int(5.5 * 1024**3)
OVERHEAD_BYTES = 512 * 1024**2


def pascal_bf16_cuda_ok(capability: tuple[int, int] | None, bf16_supported: bool) -> bool:
    """sm_61 não entra. bf16 de CUDA começa em Ampere, capability 8."""

    if capability is None or capability[0] < 8:
        return False
    return bool(bf16_supported)


def refuse_cuda_bf16(device: torch.device, resident_dtype: torch.dtype) -> None:
    if device.type != "cuda":
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA não está disponível neste processo. O smoke fica na CPU.")
    capability = torch.cuda.get_device_capability(device)
    supported = bool(torch.cuda.is_bf16_supported())
    if resident_dtype == torch.bfloat16 and not pascal_bf16_cuda_ok(capability, supported):
        raise RuntimeError(
            f"GPU capability {capability[0]}.{capability[1]} não executa bf16 "
            f"(suporte reportado: {supported}). O residente está em bf16. "
            "O smoke usa CPU e float32. Não movo o residente inteiro para a GPU."
        )


def estimate_bytes(
    *,
    resident_bytes: int,
    vocab: int,
    hidden: int,
    intermediate: int,
    n_layers: int,
    n_experts: int,
    rank: int,
    seq_len: int,
    k1: int,
    lstm_embed: int,
    lstm_hidden: int,
) -> dict[str, int]:
    """Teto do processo, não medição. Experts de uma camada, não o store inteiro."""

    if min(resident_bytes, vocab, hidden, intermediate, n_layers, n_experts, rank, seq_len, k1) < 1:
        raise ValueError("estimativa com dimensão < 1")
    lora = n_layers * n_experts * 3 * rank * (hidden + intermediate) * 4
    lstm_params = (
        vocab * lstm_embed
        + 4 * lstm_hidden * lstm_embed
        + 4 * lstm_hidden * lstm_hidden
        + 8 * lstm_hidden
        + n_layers * (lstm_hidden * n_experts + n_experts)
    )
    lstm = lstm_params * 12
    per_expert = 3 * hidden * intermediate * 4
    distinct = min(n_experts, k1 * seq_len)
    experts = distinct * per_expert
    head = vocab * hidden * 4
    pieces = {
        "resident_bytes": resident_bytes,
        "lora_fp32_bytes": lora,
        "adam_lora_bytes": lora * 2,
        "lstm_with_adam_bytes": lstm,
        "expert_peak_fp32_bytes": experts,
        "lm_head_fp32_bytes": head,
        "overhead_bytes": OVERHEAD_BYTES,
    }
    pieces["total_bytes"] = sum(pieces.values())
    pieces["launch_ceiling_bytes"] = LAUNCH_CEILING_BYTES
    pieces["cgroup_bytes"] = CGROUP_BYTES
    pieces["vram_budget_bytes"] = VRAM_BUDGET_BYTES
    return pieces


def fingerprint(path: str | Path, prefix: int = 4096) -> dict[str, int | str]:
    file = Path(path)
    size = file.stat().st_size
    with file.open("rb") as handle:
        blob = handle.read(prefix)
    return {
        "bytes": size,
        "prefix_bytes": len(blob),
        "prefix_sha256": hashlib.sha256(blob).hexdigest(),
    }


class ExpertLoRA(nn.Module):
    """Delta nas três projeções. ``B`` nasce zero. O peso base não mora aqui."""

    def __init__(self, n_layers: int, n_experts: int, hidden: int, intermediate: int, rank: int, alpha: float) -> None:
        super().__init__()
        if rank < 1 or alpha <= 0:
            raise ValueError("rank >= 1 e alpha > 0")
        self.scale = alpha / rank
        self.gate_A = nn.Parameter(torch.empty(n_layers, n_experts, rank, hidden))
        self.gate_B = nn.Parameter(torch.zeros(n_layers, n_experts, intermediate, rank))
        self.up_A = nn.Parameter(torch.empty(n_layers, n_experts, rank, hidden))
        self.up_B = nn.Parameter(torch.zeros(n_layers, n_experts, intermediate, rank))
        self.down_A = nn.Parameter(torch.empty(n_layers, n_experts, rank, intermediate))
        self.down_B = nn.Parameter(torch.zeros(n_layers, n_experts, hidden, rank))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, parameter in self.named_parameters():
            if name.endswith("_B"):
                nn.init.zeros_(parameter)
                continue
            matrix = parameter.view(-1, parameter.shape[-2], parameter.shape[-1])
            for row in matrix:
                nn.init.kaiming_uniform_(row, a=5**0.5)


class OffloadLanguageModel(nn.Module):
    """Forward de uma janela. O store fica aberto e só é lido."""

    def __init__(
        self,
        weights: QwenWeights,
        router: TwoHorizonLstmRouter,
        store: ExpertStore,
        *,
        rank: int = 4,
        alpha: float = 8.0,
    ) -> None:
        super().__init__()
        spec = weights.spec
        if router.n_layers != spec.n_layers or router.n_experts != spec.n_experts:
            raise ValueError("o LSTM não casa com o store")
        if router.vocab_size != spec.vocab_size:
            raise ValueError("o vocabulário do LSTM não é o do residente")
        if store.header.n_layers != spec.n_layers or store.header.n_experts != spec.n_experts:
            raise ValueError("o cabeçalho do store não casa com o config")
        self.weights = weights
        self.store = store
        self.router = router
        self.lora = ExpertLoRA(spec.n_layers, spec.n_experts, spec.hidden_size, spec.moe_intermediate, rank, alpha)
        self.store_dtype = torch_dtype(store.header.dtype_code)
        for tensor in _resident_tensors(weights):
            tensor.requires_grad_(False)

    def forward_window(self, token_ids: torch.Tensor):
        if token_ids.ndim != 2 or token_ids.shape[0] != 1:
            raise ValueError("a janela de offload é [1, T]")
        if token_ids.shape[1] < 2:
            raise ValueError("a janela precisa de ao menos 2 tokens")
        plan, _state = self.router.plan(token_ids)
        hidden = self._embed(token_ids)
        spec = self.weights.spec
        for layer_index in range(spec.n_layers):
            coeffs = plan.soft_weights[:, :, layer_index, :]
            index = plan.hot_index[:, :, layer_index, :]
            valid = plan.hot_valid[:, :, layer_index, :]
            hidden = checkpoint(
                self._layer,
                hidden,
                coeffs,
                index,
                valid,
                torch.tensor(layer_index, device=hidden.device),
                use_reentrant=False,
            )
        normed = rms_norm(hidden, _as(self.weights.final_norm, hidden), spec.rms_eps)
        logits = F.linear(normed, _as(self.weights.lm_head, hidden))
        return logits, plan

    def _embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        rows = self.weights.embed.index_select(0, token_ids.reshape(-1))
        return rows.to(dtype=torch.float32).view(1, token_ids.shape[1], -1)

    def _layer(self, hidden, coeffs, index, valid, layer_index_t):
        spec = self.weights.spec
        layer_index = int(layer_index_t.item())
        layer = self.weights.layers[layer_index]
        normed = rms_norm(hidden, _as(layer.input_norm, hidden), spec.rms_eps)
        hidden = hidden + _attention_seq(layer, normed, spec)
        moe_in = rms_norm(hidden, _as(layer.post_norm, hidden), spec.rms_eps)
        hidden = hidden + self._moe(moe_in, layer_index, index, valid, coeffs)
        return hidden

    def _moe(self, hidden, layer_index, index, valid, coeffs):
        flat = hidden[0]
        pieces = []
        rows = []
        cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        seq, slots = index.shape[1], index.shape[2]
        for token in range(seq):
            for slot in range(slots):
                if not bool(valid[0, token, slot].item()):
                    continue
                expert = int(index[0, token, slot].item())
                if expert not in cache:
                    cache[expert] = self._read(layer_index, expert)
                gate, up, down = cache[expert]
                current = flat[token : token + 1]
                projected_gate = F.linear(current, gate) + self.lora.scale * _delta(
                    current, self.lora.gate_A[layer_index, expert], self.lora.gate_B[layer_index, expert]
                )
                projected_up = F.linear(current, up) + self.lora.scale * _delta(
                    current, self.lora.up_A[layer_index, expert], self.lora.up_B[layer_index, expert]
                )
                mid = F.silu(projected_gate) * projected_up
                down_y = F.linear(mid, down) + self.lora.scale * _delta(
                    mid, self.lora.down_A[layer_index, expert], self.lora.down_B[layer_index, expert]
                )
                pieces.append(down_y[0] * coeffs[0, token, slot])
                rows.append(token)
        out = flat.new_zeros(flat.shape)
        if pieces:
            out = out.index_add(0, torch.tensor(rows, device=flat.device), torch.stack(pieces, dim=0))
        return out.unsqueeze(0)

    def _read(self, layer: int, expert: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate_b, up_b, down_b = self.store.payload(layer, expert)
        layout = self.store.layout
        gate = _matrix(gate_b, self.store_dtype, (layout.intermediate, layout.hidden))
        up = _matrix(up_b, self.store_dtype, (layout.intermediate, layout.hidden))
        down = _matrix(down_b, self.store_dtype, (layout.hidden, layout.intermediate))
        return gate, up, down


def lm_step(model: OffloadLanguageModel, optimizer: torch.optim.Optimizer, token_ids: torch.Tensor) -> dict[str, float | int | bool]:
    model.router.train()
    model.lora.train()
    optimizer.zero_grad(set_to_none=True)
    logits, plan = model.forward_window(token_ids)
    loss = language_model_loss(logits, token_ids)
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError("loss de linguagem não é finita")
    loss.backward()
    report = grad_report(model)
    if not report["grads_finite"]:
        raise RuntimeError("há gradiente não finito")
    optimizer.step()
    report["loss"] = float(loss.detach())
    report["distinct_experts"] = distinct_experts(plan)
    report["hot_entropy"] = _hot_entropy(plan)
    report["router_entropy"] = _router_entropy(model, token_ids)
    return report


def grad_report(model: OffloadLanguageModel) -> dict[str, float | bool]:
    lstm = model.router.lstm.weight_ih_l0.grad
    lora_b = []
    finite = True
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all().item()):
            finite = False
        if name.startswith("lora.") and name.endswith("_B"):
            lora_b.append(parameter.grad.detach())
    lora_norm = torch.zeros(())
    if lora_b:
        lora_norm = torch.stack([item.float().pow(2).sum() for item in lora_b]).sum().sqrt()
    lstm_norm = 0.0 if lstm is None else float(lstm.detach().float().norm())
    return {
        "grads_finite": finite and lstm is not None and bool(torch.isfinite(lstm).all().item()),
        "lstm_grad_norm": lstm_norm,
        "lora_b_grad_norm": float(lora_norm),
        "lstm_grad_nonzero": lstm is not None and bool(lstm.detach().abs().sum() > 0),
        "lora_b_grad_nonzero": bool(lora_norm > 0),
    }


def distinct_experts(plan) -> int:
    valid = plan.hot_valid[0]
    index = plan.hot_index[0]
    tokens, layers, slots = torch.nonzero(valid, as_tuple=True)
    experts = index[tokens, layers, slots]
    return len({(int(layer), int(expert)) for layer, expert in zip(layers.tolist(), experts.tolist(), strict=True)})


def trainable_state(model: OffloadLanguageModel) -> dict:
    return {"router": model.router.state_dict(), "lora": model.lora.state_dict()}


def load_trainable(model: OffloadLanguageModel, payload: dict) -> None:
    model.router.load_state_dict(payload["router"])
    model.lora.load_state_dict(payload["lora"])


def save_checkpoint(path: str | Path, *, step: int, model: OffloadLanguageModel, optimizer: torch.optim.Optimizer, extra: dict) -> None:
    torch.save(
        {
            "version": 1,
            "kind": "qwen-lm-lora-lstm",
            "step": step,
            "router": model.router.state_dict(),
            "lora": model.lora.state_dict(),
            "optimizer": optimizer.state_dict(),
            **extra,
        },
        path,
    )


def _hot_entropy(plan) -> float:
    weights = plan.soft_weights.detach().float().clamp_min(0)
    total = weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    probs = weights / total
    ent = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    valid = plan.hot_valid.detach().any(dim=-1)
    if not bool(valid.any().item()):
        return 0.0
    return float(ent[valid].mean())


def _router_entropy(model: OffloadLanguageModel, token_ids: torch.Tensor) -> float:
    with torch.no_grad():
        decision, _state = model.router.forward(token_ids)
        probs = torch.softmax(decision.raw_scores.float(), dim=-1)
        ent = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    return float(ent.mean())


def _resident_tensors(weights: QwenWeights):
    yield weights.embed
    yield weights.final_norm
    yield weights.lm_head
    for layer in weights.layers:
        for name in ("input_norm", "post_norm", "q", "k", "v", "o", "q_norm", "k_norm"):
            yield getattr(layer, name)
        for name in ("q_bias", "k_bias", "v_bias", "o_bias"):
            bias = getattr(layer, name)
            if bias is not None:
                yield bias


def _as(weight: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor | None:
    if weight is None:
        return None
    if weight.dtype == ref.dtype and weight.device == ref.device:
        return weight
    return weight.to(dtype=ref.dtype, device=ref.device)


def _matrix(raw: bytes, dtype: torch.dtype, shape: tuple[int, int]) -> torch.Tensor:
    elem = torch.empty((), dtype=dtype).element_size()
    need = shape[0] * shape[1] * elem
    if len(raw) != need:
        raise ValueError(f"bloco do expert tem {len(raw)} bytes, esperado {need}")
    cpu = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape).clone()
    return cpu.to(dtype=torch.float32)


def _delta(hidden: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return F.linear(F.linear(hidden, left), right)


def _attention_seq(layer, hidden: torch.Tensor, spec: QwenSpec) -> torch.Tensor:
    batch, seq, _width = hidden.shape
    if batch != 1 or seq < 1:
        raise ValueError("a atenção do treino é batch 1")
    query = F.linear(hidden, _as(layer.q, hidden), _as(layer.q_bias, hidden))
    key = F.linear(hidden, _as(layer.k, hidden), _as(layer.k_bias, hidden))
    value = F.linear(hidden, _as(layer.v, hidden), _as(layer.v_bias, hidden))
    query = query.view(batch, seq, spec.n_heads, spec.head_dim)
    key = key.view(batch, seq, spec.n_kv_heads, spec.head_dim)
    value = value.view(batch, seq, spec.n_kv_heads, spec.head_dim)
    query = rms_norm(query, _as(layer.q_norm, hidden), spec.rms_eps)
    key = rms_norm(key, _as(layer.k_norm, hidden), spec.rms_eps)
    positions = torch.arange(seq, device=hidden.device)
    cos, sin = _rope_seq(positions, spec.head_dim, spec.rope_theta, hidden.dtype)
    query = _apply_rope(query.transpose(1, 2), cos, sin)
    key = _apply_rope(key.transpose(1, 2), cos, sin)
    value = value.transpose(1, 2)
    groups = spec.n_heads // spec.n_kv_heads
    mixed = F.scaled_dot_product_attention(
        query,
        _repeat_kv(key, groups),
        _repeat_kv(value, groups),
        is_causal=seq > 1,
    )
    mixed = mixed.transpose(1, 2).reshape(batch, seq, spec.n_heads * spec.head_dim)
    return F.linear(mixed, _as(layer.o, hidden), _as(layer.o_bias, hidden))


def _rope_seq(positions: torch.Tensor, head_dim: int, theta: float, dtype: torch.dtype):
    half = torch.arange(0, head_dim, 2, device=positions.device, dtype=torch.float32)
    inv = 1.0 / (theta ** (half / head_dim))
    freqs = positions.float().unsqueeze(1) * inv.unsqueeze(0)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype)[None, None, :, :], emb.sin().to(dtype)[None, None, :, :]
