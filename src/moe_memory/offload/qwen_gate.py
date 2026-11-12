"""Gate original do Qwen3, congelado, com o top-k do config.

Não cria LSTM. O score é a projeção linear do hidden da camada, a
softmax e o top-k renormalizado (``norm_topk_prob``). A escolha e o
peso saem sem gradiente: a LoRA treina pelo expert, o gate não.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class QwenGateRouter(nn.Module):
    """Não tem parâmetro. O peso do gate mora no residente, congelado."""

    kind = "qwen"

    def __init__(self, *, vocab_size: int, n_layers: int, n_experts: int, experts_per_tok: int) -> None:
        super().__init__()
        if min(vocab_size, n_layers, n_experts, experts_per_tok) < 1:
            raise ValueError("o gate do Qwen precisa de dimensões >= 1")
        if experts_per_tok > n_experts:
            raise ValueError("experts por token passa do total de experts")
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.k1 = experts_per_tok
        self.k2 = experts_per_tok

    def train(self, mode: bool = True):
        return super().train(False)


def route_qwen(hidden: torch.Tensor, gate: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``hidden`` é ``[1, T, H]``. Devolve índice, máscara e peso ``[1, T, k]``."""

    if hidden.ndim != 3 or hidden.shape[0] != 1:
        raise ValueError("o gate do treino vê uma janela [1, T, H]")
    if gate.ndim != 2 or gate.shape[1] != hidden.shape[-1]:
        raise ValueError("o gate não casa com o hidden")
    if not 1 <= k <= gate.shape[0]:
        raise ValueError("k do gate está fora do número de experts")
    flat = hidden.reshape(-1, hidden.shape[-1])
    weight = gate if gate.device == flat.device and gate.dtype == flat.dtype else gate.to(device=flat.device, dtype=flat.dtype)
    with torch.no_grad():
        probs = torch.softmax(F.linear(flat, weight), dim=-1)
        chosen, index = torch.topk(probs, k, dim=-1)
        chosen = chosen / chosen.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    seq = hidden.shape[1]
    index = index.view(1, seq, k).to(dtype=torch.long)
    coeffs = chosen.view(1, seq, k).to(dtype=hidden.dtype)
    valid = torch.ones(1, seq, k, dtype=torch.bool, device=index.device)
    return index, valid, coeffs
