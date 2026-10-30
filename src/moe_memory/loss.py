"""``L = L_tarefa + λ · C_mem + β · L_balanço``.

``L_tarefa`` é a entropia cruzada do próximo token, média por token.

``C_mem`` é o proxy do doc, especializado em ``k = 2``:

- treino (``weighted_mean``): média ponderada dos top-2 sob a
  probabilidade do roteador renormalizada nesses dois. O roteador aprende
  a ranquear; o gradiente passa pelos pesos, não pelo índice discreto.
- inferência (``sum``): soma dos custos dos top-2. Acesso realizado: os
  dois experts são tocados, sem desconto pela massa de probabilidade.

Cada termo de ``C_mem`` usa essa agregação:

- acesso: ``c(ℓ(e))``
- troca: ``w(ℓ(e), quente) = |c(quente) - c(ℓ(e))|``, zero se já está quente

A fórmula ``(1/T) Σ_t Σ_e p(t,e) · c(ℓ(e))`` fica no campo
``access_expected``. Ela não entra em ``L``: com ``k = 2`` o que entra é
a agregação dos top-2. O campo existe para a correlação proxy × emulador.

``L_balanço`` é o auxiliar do Switch, ``E · Σ_e f(e) P(e)``, com ``f``
fração das vagas top-k (soma 1) e sem gradiente. ``P`` é a média de
``p``. ``β`` multiplica a média desse termo entre as camadas MoE, e
``λ`` multiplica a média de ``C_mem``. A média entre camadas mantém ``λ``
e ``β`` na escala escrita para um roteador (``c`` ~ 1–20, ``β`` ~ 10⁻²),
não ``n_layers`` vezes maior.

A entropia do roteador só é monitorada. Não entra na loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from moe_memory.levels import HOT, N_LEVELS


@dataclass(frozen=True)
class LossTerms:
    total: torch.Tensor
    task: torch.Tensor
    c_mem: torch.Tensor
    balance: torch.Tensor
    entropy: torch.Tensor
    access: torch.Tensor
    swap: torch.Tensor
    access_expected: torch.Tensor

    def as_floats(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach()),
            "task": float(self.task.detach()),
            "c_mem": float(self.c_mem.detach()),
            "balance": float(self.balance.detach()),
            "entropy": float(self.entropy.detach()),
            "access": float(self.access.detach()),
            "swap": float(self.swap.detach()),
            "access_expected": float(self.access_expected.detach()),
        }


def swap_cost_vector(level_cost: torch.Tensor) -> torch.Tensor:
    """``w(ℓ, quente) = |c(quente) - c(ℓ)|`` para cada nível."""

    if level_cost.shape != (N_LEVELS,):
        raise ValueError(f"level_cost tem shape {tuple(level_cost.shape)}; esperado ({N_LEVELS},)")
    return (level_cost - level_cost[HOT]).abs()


def aggregate_topk(
    per_expert: torch.Tensor,
    probs: torch.Tensor,
    topk_index: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Agrega um valor por expert em ``[B, T]``.

    ``per_expert`` é ``[E]``. ``probs`` é ``[B, T, E]``. ``topk_index`` é ``[B, T, k]``.
    """

    selected = per_expert[topk_index]
    if mode == "weighted_mean":
        weight = probs.gather(-1, topk_index)
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return (weight * selected).sum(dim=-1)
    if mode == "sum":
        return selected.sum(dim=-1)
    raise ValueError(f"agregação desconhecida: {mode!r}")


def balance_loss(probs: torch.Tensor, topk_index: torch.Tensor) -> torch.Tensor:
    """Switch: ``E · Σ_e f(e) P(e)``. Mínimo 1 quando a carga é uniforme."""

    batch, seq, n_experts = probs.shape
    k = topk_index.shape[-1]
    flat_index = topk_index.reshape(-1)
    ones = torch.ones(flat_index.shape[0], device=probs.device, dtype=probs.dtype)
    counts = torch.zeros(n_experts, device=probs.device, dtype=probs.dtype)
    counts.scatter_add_(0, flat_index, ones)
    fraction = counts / (batch * seq * k)
    mean_prob = probs.mean(dim=(0, 1))
    return n_experts * torch.sum(fraction.detach() * mean_prob)


def router_entropy(probs: torch.Tensor) -> torch.Tensor:
    logged = probs.clamp_min(1e-9).log()
    return -(probs * logged).sum(dim=-1).mean()


def task_cross_entropy(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """CE do token ``t+1`` a partir da posição ``t``. Média por token."""

    return F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)),
        input_ids[:, 1:].reshape(-1),
    )


def compute_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    probs: torch.Tensor,
    topk_index: torch.Tensor,
    levels: torch.Tensor,
    level_cost: torch.Tensor,
    *,
    lambda_: float,
    beta: float,
    aggregation: str,
) -> LossTerms:
    """``aggregation`` vem de ``LossConfig.aggregation(training)``."""

    _check_shapes(logits, input_ids, probs, topk_index, levels, level_cost)
    n_layers = probs.shape[0]
    topk_index = topk_index.to(device=probs.device)
    levels = levels.to(device=probs.device)
    level_cost = level_cost.to(device=probs.device, dtype=probs.dtype)
    promotion = swap_cost_vector(level_cost)

    access_layers = []
    swap_layers = []
    balance_layers = []
    entropy_layers = []
    expected_layers = []
    for layer in range(n_layers):
        access_cost = level_cost[levels[layer]]
        swap_cost = promotion[levels[layer]]
        layer_probs = probs[layer]
        layer_index = topk_index[layer]
        access_layers.append(
            aggregate_topk(access_cost, layer_probs, layer_index, aggregation).mean()
        )
        swap_layers.append(
            aggregate_topk(swap_cost, layer_probs, layer_index, aggregation).mean()
        )
        balance_layers.append(balance_loss(layer_probs, layer_index))
        entropy_layers.append(router_entropy(layer_probs))
        expected_layers.append((layer_probs * access_cost).sum(dim=-1).mean())

    access = torch.stack(access_layers).mean()
    swap = torch.stack(swap_layers).mean()
    c_mem = access + swap
    balance = torch.stack(balance_layers).mean()
    task = task_cross_entropy(logits, input_ids)
    total = task + float(lambda_) * c_mem + float(beta) * balance
    return LossTerms(
        total=total,
        task=task,
        c_mem=c_mem,
        balance=balance,
        entropy=torch.stack(entropy_layers).mean().detach(),
        access=access,
        swap=swap,
        access_expected=torch.stack(expected_layers).mean().detach(),
    )


def _check_shapes(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    probs: torch.Tensor,
    topk_index: torch.Tensor,
    levels: torch.Tensor,
    level_cost: torch.Tensor,
) -> None:
    if logits.ndim != 3:
        raise ValueError(f"logits precisa ser [B, T, V], recebido {tuple(logits.shape)}")
    if input_ids.shape != logits.shape[:2]:
        raise ValueError(
            f"input_ids {tuple(input_ids.shape)} não casa com logits {tuple(logits.shape[:2])}"
        )
    if input_ids.size(1) < 2:
        raise ValueError("CE de próximo token precisa de seq_len >= 2")
    if probs.ndim != 4 or topk_index.ndim != 4:
        raise ValueError("probs e topk_index precisam ser [camadas, B, T, ...]")
    n_layers, batch, seq, n_experts = probs.shape
    if (batch, seq) != logits.shape[:2]:
        raise ValueError("probs não casa com o batch de logits")
    if topk_index.shape[:3] != (n_layers, batch, seq):
        raise ValueError(
            f"topk_index {tuple(topk_index.shape)} não casa com probs {tuple(probs.shape)}"
        )
    if levels.shape != (n_layers, n_experts):
        raise ValueError(
            f"levels {tuple(levels.shape)} não casa com [camadas, E] = {(n_layers, n_experts)}"
        )
    if level_cost.shape != (N_LEVELS,):
        raise ValueError(f"level_cost precisa ter {N_LEVELS} níveis")
