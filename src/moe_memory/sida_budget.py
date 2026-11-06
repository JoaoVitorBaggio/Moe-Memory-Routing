"""Gancho da comparação com o SiDA, no mesmo saldo de ativação.

O saldo é o ``k`` do preditor: tantos experts ativos por token em cada
camada, e portanto ``k × camadas`` forwards de expert por token. Isso
iguala a carga de computação ao top-k do MoE. Não há corrida, acurácia
nem latência neste módulo.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActivationBudget:
    k: int
    n_layers: int

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError(f"k={self.k} precisa ser >= 1")
        if self.n_layers < 1:
            raise ValueError(f"n_layers={self.n_layers} precisa ser >= 1")

    @property
    def experts_per_token_per_layer(self) -> int:
        return self.k

    @property
    def expert_forwards_per_token(self) -> int:
        return self.k * self.n_layers


def budgets_match(left: ActivationBudget, right: ActivationBudget) -> bool:
    return (
        left.experts_per_token_per_layer == right.experts_per_token_per_layer
        and left.expert_forwards_per_token == right.expert_forwards_per_token
    )


def describe_budget(budget: ActivationBudget) -> str:
    return (
        "gancho SiDA: mesma ativação, sem corrida\n"
        f"experts por token por camada: {budget.experts_per_token_per_layer}\n"
        f"forwards de expert por token: {budget.expert_forwards_per_token}\n"
        "a comparação grande não rodou; não há acurácia nem latência aqui"
    )
