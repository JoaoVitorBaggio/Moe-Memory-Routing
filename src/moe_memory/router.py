"""Roteador linear que vê o estado oculto e o nível ℓ(e).

O logit do expert ``e`` é o produto do hidden projetado pela chave
``expert_key[e] + embed(ℓ(e))``. Trocar o expert de VRAM para SSD muda
a chave, então o mesmo token pontua o expert de outro jeito. ℓ(e) é
índice discreto: o gradiente atualiza a tabela de níveis, não a decisão
de cache (essa é do emulador, por batch).
"""

from __future__ import annotations

import torch
from torch import nn

from moe_memory.levels import N_LEVELS


class MemoryAwareRouter(nn.Module):
    def __init__(self, d_model: int, n_experts: int, n_levels: int = N_LEVELS) -> None:
        super().__init__()
        if n_experts < 1:
            raise ValueError("n_experts precisa ser >= 1")
        self.n_experts = n_experts
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.expert_keys = nn.Parameter(torch.empty(n_experts, d_model))
        self.level_embed = nn.Embedding(n_levels, d_model)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.normal_(self.expert_keys, std=0.02)
        nn.init.normal_(self.level_embed.weight, std=0.02)

    def forward(
        self,
        hidden: torch.Tensor,
        levels: torch.Tensor,
        *,
        use_level: bool,
    ) -> torch.Tensor:
        """``hidden`` é ``[B, T, D]``; ``levels`` é ``[E]`` long. Sai ``[B, T, E]``."""

        if levels.shape != (self.n_experts,):
            raise ValueError(
                f"levels tem shape {tuple(levels.shape)}; esperado ({self.n_experts},)"
            )
        keys = self.expert_keys
        if use_level:
            keys = keys + self.level_embed(levels.to(device=hidden.device))
        projected = self.proj(hidden)
        return projected @ keys.transpose(0, 1)
