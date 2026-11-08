"""Vagas fixas. Quem sai é quem o plano soltou, não um LRU escondido."""

from __future__ import annotations

import threading

import torch


class SlotPool:
    """``n_slots`` blocos do mesmo tamanho. A chave é ``(camada, expert)``."""

    def __init__(self, n_slots: int, block_nbytes: int, *, device: torch.device | None = None, pin: bool = False) -> None:
        if n_slots < 1:
            raise ValueError("o pool precisa de ao menos uma vaga")
        if block_nbytes < 1:
            raise ValueError("bloco precisa ter ao menos 1 byte")
        self.n_slots = n_slots
        self.block_nbytes = block_nbytes
        self.device = torch.device("cpu") if device is None else torch.device(device)
        self.pin = bool(pin) and self.device.type == "cpu" and torch.cuda.is_available()
        self.blocks: list[torch.Tensor] = []
        for _slot in range(n_slots):
            block = torch.empty(block_nbytes, dtype=torch.uint8, device=self.device)
            if self.pin:
                block = block.pin_memory()
            self.blocks.append(block)
        self.slot_of: dict[tuple[int, int], int] = {}
        self.key_of: list[tuple[int, int] | None] = [None] * n_slots
        self.free: list[int] = list(range(n_slots))
        self._lock = threading.Lock()

    @property
    def nbytes(self) -> int:
        return self.n_slots * self.block_nbytes

    def __contains__(self, key: tuple[int, int]) -> bool:
        with self._lock:
            return key in self.slot_of

    def has_free(self) -> bool:
        with self._lock:
            return bool(self.free)

    def occupied(self) -> list[tuple[int, int]]:
        with self._lock:
            return [key for key in self.key_of if key is not None]

    def reserve(self, key: tuple[int, int]) -> int | None:
        """Devolve a vaga. ``None`` se não há vaga livre e a chave é nova."""

        with self._lock:
            found = self.slot_of.get(key)
            if found is not None:
                return found
            if not self.free:
                return None
            slot = self.free.pop()
            self.slot_of[key] = slot
            self.key_of[slot] = key
            return slot

    def evict(self, key: tuple[int, int]) -> bool:
        with self._lock:
            slot = self.slot_of.pop(key, None)
            if slot is None:
                return False
            self.key_of[slot] = None
            self.free.append(slot)
            return True

    def tensor(self, key: tuple[int, int]) -> torch.Tensor:
        with self._lock:
            try:
                slot = self.slot_of[key]
            except KeyError as exc:
                raise KeyError(f"vaga sem {key}") from exc
            return self.blocks[slot]
