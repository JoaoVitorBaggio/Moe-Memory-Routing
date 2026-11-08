"""Pool morno. Threads leem o store; quem sai é o expert que o plano soltou.

A vaga é fixa. Um pedido com passo menor toma a vaga de um residente
cujo passo é maior: o token que vai executar agora não espera atrás de
um prefetch. A leitura em si não segura o lock.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import torch

from moe_memory.offload.pool import SlotPool
from moe_memory.offload.stats import OffloadStats


class WarmTier:
    def __init__(
        self,
        store,
        n_slots: int,
        *,
        stats: OffloadStats | None = None,
        n_threads: int = 1,
        pin: bool | None = None,
    ) -> None:
        if n_threads < 1:
            raise ValueError("n_threads precisa ser >= 1")
        self.store = store
        self.stats = stats if stats is not None else OffloadStats()
        self.pool = SlotPool(
            n_slots,
            int(store.header.block_nbytes),
            device=torch.device("cpu"),
            pin=True if pin is None else pin,
        )
        self.read_order: list[tuple[int, int]] = []
        self._wanted: dict[tuple[int, int], int] = {}
        self._events: dict[tuple[int, int], threading.Event] = {}
        self._inflight: set[tuple[int, int]] = set()
        self._cancel: set[tuple[int, int]] = set()
        self._holds: dict[tuple[int, int], int] = {}
        self._on_ready: list[Callable[[], None]] = []
        self._error: BaseException | None = None
        self._stop = False
        self._cv = threading.Condition()
        self._threads = [
            threading.Thread(target=self._worker, name=f"warm-{i}", daemon=True) for i in range(n_threads)
        ]
        for thread in self._threads:
            thread.start()

    def add_ready_callback(self, callback: Callable[[], None]) -> None:
        self._on_ready.append(callback)

    def sync(self, ranked: list[tuple[int, tuple[int, int]]]) -> None:
        """O conjunto pedido passa a ser exatamente ``ranked``. Passo menor é mais urgente."""

        wanted: dict[tuple[int, int], int] = {}
        for step, key in ranked:
            previous = wanted.get(key)
            wanted[key] = step if previous is None else min(previous, step)
        fire = False
        with self._cv:
            self._raise_if_error()
            for key in list(self._wanted):
                if key not in wanted:
                    self._cancel_locked(key)
            for key, step in wanted.items():
                self._request_locked(key, step)
            self._rebalance_locked()
            fire = any(event.is_set() for event in self._events.values())
            self._cv.notify_all()
        if fire:
            self._fire()

    def ensure(self, key: tuple[int, int]) -> None:
        """Espera o bloco. Se não estava pronto, conta miss e o tempo de espera."""

        with self._cv:
            self._raise_if_error()
            event = self._events.get(key)
            if event is None or key not in self._wanted:
                raise KeyError(f"expert {key} não foi pedido ao morno")
            ready = event.is_set()
        if ready:
            return
        t0 = _now()
        event.wait()
        with self._cv:
            self._raise_if_error()
            if key not in self.pool or not self._events[key].is_set():
                raise RuntimeError(f"expert {key} saiu do morno enquanto era esperado")
        self.stats.note_warm_miss(_now() - t0)

    def is_ready(self, key: tuple[int, int]) -> bool:
        with self._cv:
            event = self._events.get(key)
            return event is not None and event.is_set() and key in self.pool

    def tensor(self, key: tuple[int, int]) -> torch.Tensor:
        return self.pool.tensor(key)

    def hold(self, key: tuple[int, int]) -> bool:
        """Impede a troca da vaga enquanto a cópia quente lê o bloco."""

        with self._cv:
            event = self._events.get(key)
            if event is None or not event.is_set() or key not in self.pool:
                return False
            self._holds[key] = self._holds.get(key, 0) + 1
            return True

    def release_hold(self, key: tuple[int, int]) -> None:
        with self._cv:
            left = self._holds.get(key, 0) - 1
            if left <= 0:
                self._holds.pop(key, None)
            else:
                self._holds[key] = left
            if key not in self._holds and (key in self._cancel or key not in self._wanted):
                self.pool.evict(key)
                self._cancel.discard(key)
                self._events.pop(key, None)
            self._cv.notify_all()

    def drain(self, timeout: float = 30.0) -> None:
        """Espera o que cabe. Chave a mais, sem vaga, não trava o teste."""

        deadline = _now() + timeout
        while _now() < deadline:
            with self._cv:
                self._raise_if_error()
                pending = [key for key in self._wanted if not self._events[key].is_set()]
                if not pending:
                    return
                if not self._inflight and not self.pool.has_free():
                    return
            event = self._events[pending[0]]
            event.wait(timeout=0.05)
        raise TimeoutError("prefetch morno não terminou")

    def close(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        for thread in self._threads:
            thread.join(timeout=5)

    def _request_locked(self, key: tuple[int, int], step: int) -> None:
        previous = self._wanted.get(key)
        self._wanted[key] = step if previous is None else min(previous, step)
        self._cancel.discard(key)
        event = self._events.get(key)
        if event is None:
            event = threading.Event()
            self._events[key] = event
        if key in self.pool and key not in self._inflight:
            event.set()

    def _cancel_locked(self, key: tuple[int, int]) -> None:
        self._wanted.pop(key, None)
        if key in self._inflight or self._holds.get(key, 0) > 0:
            self._cancel.add(key)
            return
        self.pool.evict(key)
        self._events.pop(key, None)

    def _rebalance_locked(self) -> None:
        while True:
            pending = [
                key
                for key in self._wanted
                if not self._events[key].is_set() and key not in self._inflight and key not in self.pool
            ]
            if not pending or self.pool.has_free():
                return
            urgent = min(pending, key=lambda key: (self._wanted[key], key))
            victims = [
                key
                for key in self.pool.occupied()
                if key not in self._inflight
                and self._holds.get(key, 0) == 0
                and self._wanted.get(key, -1) > self._wanted[urgent]
            ]
            if not victims:
                return
            victim = max(victims, key=lambda key: (self._wanted[key], key))
            self.pool.evict(victim)
            self._events[victim] = threading.Event()

    def _pick_locked(self) -> tuple[int, int] | None:
        pending = []
        for key, step in self._wanted.items():
            if self._events[key].is_set() or key in self._inflight:
                continue
            pending.append((step, key))
        if not pending:
            return None
        pending.sort()
        key = pending[0][1]
        slot = self.pool.reserve(key)
        if slot is None:
            return None
        self._inflight.add(key)
        self.read_order.append(key)
        return key

    def _worker(self) -> None:
        while True:
            with self._cv:
                if self._stop:
                    return
                key = self._pick_locked()
                if key is None:
                    self._cv.wait()
                    continue
            try:
                dest = self.pool.tensor(key)
                self.store.read_into(key[0], key[1], dest)
                nbytes = int(self.store.header.raw_nbytes)
            except Exception as exc:
                with self._cv:
                    self._error = exc
                    self._inflight.discard(key)
                    event = self._events.get(key)
                    if event is not None:
                        event.set()
                    self._cv.notify_all()
                return
            fire = False
            with self._cv:
                self._inflight.discard(key)
                self.stats.add_ssd(nbytes)
                if key in self._cancel or key not in self._wanted:
                    self.pool.evict(key)
                    self._cancel.discard(key)
                    self._events.pop(key, None)
                else:
                    self._events[key].set()
                    fire = True
                self._cv.notify_all()
            if fire:
                self._fire()

    def _fire(self) -> None:
        for callback in self._on_ready:
            callback()

    def _raise_if_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("leitura do store falhou") from self._error


def _now() -> float:
    import time

    return time.perf_counter()
