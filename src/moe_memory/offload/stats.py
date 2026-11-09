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
    # Fases de parede. Não substituem elapsed_s. prefetch fica None sem lookahead.
    resident_load_s: float | None = None
    prefetch_enqueue_s: float | None = None
    decode_loop_s: float | None = None
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

    def as_dict(self) -> dict:
        elapsed = self.elapsed_s
        denom = elapsed - self.warm_start_s
        excluding = (self.tokens / denom) if denom > 0 else None
        return {
            "tokens": self.tokens,
            "elapsed_s": elapsed,
            "tokens_per_s": (self.tokens / elapsed) if elapsed > 0 else 0.0,
            "tokens_per_s_excluding_warm_start": excluding,
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
            "resident_load_s": self.resident_load_s,
            "prefetch_enqueue_s": self.prefetch_enqueue_s,
            "decode_loop_s": self.decode_loop_s,
            "timing_definitions": TIMING_DEFINITIONS,
        }


TIMING_DEFINITIONS = {
    "elapsed_s": (
        "perf_counter da parede de decode(), de antes do warm-start até o "
        "fim do último token. Inclui warm-start, o enqueue do prefetch e o "
        "loop. Exclui a construção do OffloadRuntime e load_resident."
    ),
    "resident_load_s": (
        "perf_counter só de load_resident, durante a construção, fora de elapsed_s."
    ),
    "warm_start_s": (
        "perf_counter do conjunto inicial antes do token 0, com o score do "
        "roteador e a espera (drain) do SSD e da cópia para a VRAM. Está "
        "dentro de elapsed_s. Não é stall."
    ),
    "prefetch_enqueue_s": (
        "perf_counter das chamadas que enfileiram o prefetch. null se o modo "
        "não faz prefetch. Não espera o fim do SSD nem da cópia. Está dentro "
        "de elapsed_s."
    ),
    "decode_loop_s": "perf_counter só do loop de tokens. Está dentro de elapsed_s.",
    "tokens_per_s": "tokens / elapsed_s. O warm-start fica no denominador.",
    "tokens_per_s_excluding_warm_start": (
        "tokens / (elapsed_s - warm_start_s). O prefetch e o loop continuam "
        "no denominador. Não afirma que a carga dos dois modos é a mesma."
    ),
    "stall_warm_s": (
        "Soma da parede em que a thread de decode bloqueou em warm.ensure "
        "com o bloco ainda não pronto. Não é o tempo de read do worker."
    ),
    "stall_hot_s": (
        "Soma da parede em que a thread de decode bloqueou em hot.ensure. "
        "Na GPU isso sincroniza o event daquele expert no stream de cópia, "
        "não o device inteiro."
    ),
    "ssd_bytes": (
        "Bytes que o worker terminou de ler até a amostra de elapsed_s. "
        "Uma leitura ainda em voo não entra."
    ),
}
