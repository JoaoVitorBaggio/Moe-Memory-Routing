"""Contadores do movimento. O token que espera soma o tempo no nível que faltou.

``warm_start_s`` e ``warm_start_bytes`` são o conjunto inicial, antes do
primeiro token. Não entram no stall. ``experts_executed_per_token`` é a
lista que o bench grava para comparar os modos.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class OffloadStats:
    stall_warm_s: float = 0.0
    stall_hot_s: float = 0.0
    ssd_bytes: int = 0
    warm_misses: int = 0
    hot_misses: int = 0
    warm_loads: int = 0
    tokens: int = 0
    elapsed_s: float = 0.0
    ram_peak_bytes: int = 0
    vram_peak_bytes: int = 0
    experts_executed: int = 0
    experts_executed_per_token: list[int] = field(default_factory=list)
    warm_start_s: float = 0.0
    warm_start_bytes: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_ssd(self, nbytes: int) -> None:
        with self._lock:
            self.ssd_bytes += nbytes
            self.warm_loads += 1

    def note_warm_miss(self, stall_s: float) -> None:
        with self._lock:
            self.warm_misses += 1
            self.stall_warm_s += stall_s

    def note_hot_miss(self, stall_s: float) -> None:
        with self._lock:
            self.hot_misses += 1
            self.stall_hot_s += stall_s

    def as_dict(self) -> dict[str, float | int | list[int]]:
        elapsed = self.elapsed_s
        return {
            "tokens": self.tokens,
            "elapsed_s": elapsed,
            "tokens_per_s": (self.tokens / elapsed) if elapsed > 0 else 0.0,
            "stall_warm_s": self.stall_warm_s,
            "stall_hot_s": self.stall_hot_s,
            "ssd_bytes": self.ssd_bytes,
            "ram_peak_bytes": self.ram_peak_bytes,
            "vram_peak_bytes": self.vram_peak_bytes,
            "warm_misses": self.warm_misses,
            "hot_misses": self.hot_misses,
            "warm_loads": self.warm_loads,
            "experts_executed": self.experts_executed,
            "experts_executed_per_token": list(self.experts_executed_per_token),
            "warm_start_s": self.warm_start_s,
            "warm_start_bytes": self.warm_start_bytes,
        }
