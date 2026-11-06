"""Lookahead de prefetch em camadas de transformer.

A profundidade não é medida neste repositório. É a aproximação fechada
para o segundo modelo: RAM→GPU cabe em cerca de uma camada, SSD→GPU em
três ou quatro. O expert mais frio do conjunto previsto manda no atraso.
"""

from __future__ import annotations

import torch

from moe_memory.levels import COLD, HOT, WARM

RAM_TO_GPU_LAYERS = 1
SSD_TO_GPU_LAYERS = (3, 4)


def lookahead_layers(
    level: int,
    *,
    ram_layers: int = RAM_TO_GPU_LAYERS,
    ssd_layers: int = SSD_TO_GPU_LAYERS[1],
) -> int:
    """Camadas de folga para o expert chegar na GPU antes de ser usado."""

    if ram_layers != RAM_TO_GPU_LAYERS:
        raise ValueError(
            f"RAM→GPU está fechado em {RAM_TO_GPU_LAYERS} camada, recebido {ram_layers}"
        )
    if ssd_layers not in SSD_TO_GPU_LAYERS:
        raise ValueError(
            f"SSD→GPU está fechado em {SSD_TO_GPU_LAYERS} camadas, recebido {ssd_layers}"
        )
    if level == HOT:
        return 0
    if level == WARM:
        return ram_layers
    if level == COLD:
        return ssd_layers
    raise ValueError(f"nível desconhecido: {level}")


def lookahead_for_experts(
    levels: torch.Tensor,
    *,
    ram_layers: int = RAM_TO_GPU_LAYERS,
    ssd_layers: int = SSD_TO_GPU_LAYERS[1],
) -> int:
    """O atraso do conjunto é o do expert mais frio."""

    if levels.numel() == 0:
        raise ValueError("lookahead precisa de ao menos um expert")
    return max(
        lookahead_layers(int(level), ram_layers=ram_layers, ssd_layers=ssd_layers)
        for level in levels.reshape(-1).tolist()
    )


def prefetch_layer_span(current_layer: int, n_layers: int, depth: int) -> list[int]:
    """Camadas futuras cobertas por um prefetch disparado agora.

    ``depth`` 1 a partir da camada 0 devolve ``[1]``: a cópia da RAM
    precisa de uma camada de folga. A camada corrente não entra.
    """

    if n_layers < 1:
        raise ValueError(f"n_layers={n_layers} precisa ser >= 1")
    if not 0 <= current_layer < n_layers:
        raise ValueError(f"camada {current_layer} fora de 0..{n_layers - 1}")
    if depth < 0:
        raise ValueError(f"profundidade {depth} não pode ser negativa")
    start = current_layer + 1
    stop = min(n_layers, current_layer + 1 + depth)
    return list(range(start, stop))
