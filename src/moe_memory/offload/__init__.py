"""Movimento real dos experts do Qwen3-30B-A3B.

O ``TierPlan`` continua só listando ids. Este pacote é quem move o byte:

- ``store``: cada ``(camada, expert)`` num bloco contíguo, alinhado a 4 KiB;
- ``repack``: safetensors em streaming, com teto de escrita e fsync;
- ``warm`` / ``hot``: vagas fixas na RAM pinned e na VRAM;
- ``runtime``: forward do Qwen3 que só executa o expert quente.

Não entra nos quatro braços nem em ``configs/default.yaml``. A conta de
disco (checkpoint de ~61 GB, ~44 GB livres) está em ``space``.
"""

from moe_memory.offload.space import (
    REFERENCE_CHECKPOINT_BYTES,
    REFERENCE_FREE_BYTES,
    REFERENCE_WARM_RAM_BYTES,
    SpaceError,
    SpaceReport,
    assess_space,
)
from moe_memory.offload.stats import OffloadStats

__all__ = [
    "OffloadStats",
    "REFERENCE_CHECKPOINT_BYTES",
    "REFERENCE_FREE_BYTES",
    "REFERENCE_WARM_RAM_BYTES",
    "SpaceError",
    "SpaceReport",
    "assess_space",
]
