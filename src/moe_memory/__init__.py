"""MoE com roteamento consciente de memória (AGL11157 / UFRGS).

A loss, a tabela ``c``, ``E``, ``k`` e o proxy top-2 estão fechados em
``docs/funcao-de-custo.md``. Este pacote implementa esse desenho; não é
um MoE genérico com outros defaults.
"""

from moe_memory.config import load_config
from moe_memory.dense import DenseLanguageModel
from moe_memory.emulator import ResidencyEmulator
from moe_memory.levels import COLD, HOT, WARM
from moe_memory.loss import LossTerms, compute_loss
from moe_memory.model import MemoryMoEModel

__all__ = [
    "COLD",
    "DenseLanguageModel",
    "HOT",
    "LossTerms",
    "MemoryMoEModel",
    "ResidencyEmulator",
    "WARM",
    "compute_loss",
    "load_config",
]

__version__ = "0.1.0"
