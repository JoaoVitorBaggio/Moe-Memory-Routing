"""Fila de busca de experts ordenada pelo score do preditor.

Score alto sai na frente, esteja o expert na RAM ou no SSD. O home é o
nível do emulador (quente, morno, frio). Este módulo não troca o storage
do experimento atual: só decide a ordem e o alcance do prefetch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from moe_memory.levels import COLD, HOT, N_LEVELS, WARM
from moe_memory.lookahead import lookahead_for_experts, prefetch_layer_span


@dataclass(frozen=True)
class ExpertPull:
    layer: int
    expert_id: int
    score: float
    level: int


class ExpertMemoryManager:
    def __init__(self, n_layers: int, n_experts: int, homes: torch.Tensor) -> None:
        if n_layers < 1 or n_experts < 1:
            raise ValueError("a grade de experts precisa ser positiva")
        if homes.shape != (n_layers, n_experts):
            raise ValueError(
                f"homes tem shape {tuple(homes.shape)}; esperado {(n_layers, n_experts)}"
            )
        if homes.dtype != torch.long:
            raise ValueError("homes precisa ser long")
        self._check_levels(homes)
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.homes = homes.clone()
        self.pull_log: list[ExpertPull] = []

    def set_home(self, layer: int, expert_id: int, level: int) -> None:
        self._check_index(layer, expert_id)
        self._check_levels(torch.tensor([level]))
        self.homes[layer, expert_id] = level

    def prioritize(
        self,
        layer: int,
        expert_ids: torch.Tensor,
        scores: torch.Tensor,
    ) -> list[ExpertPull]:
        """Um conjunto de experts, do score maior para o menor."""

        if not 0 <= layer < self.n_layers:
            raise ValueError(f"camada {layer} fora de 0..{self.n_layers - 1}")
        if expert_ids.ndim != 1 or scores.shape != expert_ids.shape:
            raise ValueError("expert_ids e scores precisam ser vetores do mesmo tamanho")
        if expert_ids.numel() == 0:
            return []
        order = torch.argsort(scores, descending=True, stable=True)
        pulls: list[ExpertPull] = []
        for position in order.tolist():
            expert_id = int(expert_ids[position].item())
            self._check_index(layer, expert_id)
            pulls.append(
                ExpertPull(
                    layer=layer,
                    expert_id=expert_id,
                    score=float(scores[position].item()),
                    level=int(self.homes[layer, expert_id].item()),
                )
            )
        return pulls

    def pull(self, requests: list[ExpertPull]) -> list[ExpertPull]:
        """Reordena um lote inteiro por score e registra a ordem de busca."""

        ordered = sorted(requests, key=lambda item: (-item.score, item.layer, item.expert_id))
        self.pull_log.extend(ordered)
        return ordered

    def schedule(
        self,
        current_layer: int,
        expert_ids: torch.Tensor,
        scores: torch.Tensor,
        *,
        ssd_layers: int = 4,
    ) -> list[ExpertPull]:
        """Prefetch das camadas futuras, na ordem do score.

        A profundidade usa o home dos experts previstos na camada corrente.
        O mesmo conjunto é pedido em cada camada do alcance, e o SSD entra
        na fila como qualquer outro home.
        """

        if not 0 <= current_layer < self.n_layers:
            raise ValueError(f"camada {current_layer} fora de 0..{self.n_layers - 1}")
        if expert_ids.ndim != 1 or scores.shape != expert_ids.shape:
            raise ValueError("expert_ids e scores precisam ser vetores do mesmo tamanho")
        levels = self.homes[current_layer, expert_ids]
        depth = lookahead_for_experts(levels, ssd_layers=ssd_layers)
        requests: list[ExpertPull] = []
        for layer in prefetch_layer_span(current_layer, self.n_layers, depth):
            requests.extend(self.prioritize(layer, expert_ids, scores))
        return self.pull(requests)

    def _check_index(self, layer: int, expert_id: int) -> None:
        if not 0 <= layer < self.n_layers or not 0 <= expert_id < self.n_experts:
            raise ValueError(f"expert ({layer}, {expert_id}) fora da grade")

    @staticmethod
    def _check_levels(levels: torch.Tensor) -> None:
        valid = (levels >= 0) & (levels < N_LEVELS)
        if not bool(valid.all()):
            raise ValueError(f"nível fora de {HOT}/{WARM}/{COLD}")
