"""Três níveis de residência. O índice é a posição do expert em ``c``."""

HOT = 0  # hbm / VRAM — barato de acessar, caro no orçamento B
WARM = 1  # ddr / RAM
COLD = 2  # nvme / SSD
N_LEVELS = 3
LEVEL_NAMES = ("hbm", "ddr", "nvme")


def swap_penalty(cost_src: float, cost_dst: float) -> float:
    """``w(a, b) = |c(b) - c(a)|``. Muda com ``c``; não há tabela solta."""

    return abs(cost_dst - cost_src)
