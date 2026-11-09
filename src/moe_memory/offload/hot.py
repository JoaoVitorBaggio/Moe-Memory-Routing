"""Pool quente. A cópia sobe num stream CUDA próprio; o compute espera o event.

Sem CUDA a cópia é uma thread e um ``threading.Event``. O teste de GPU
não roda nesse caminho. A vaga continua fixa: 8 por camada mais a folga
entra como ``n_slots``, calculado por quem constrói.
"""

from __future__ import annotations

import threading

import torch

from moe_memory.offload.blocks import split_block
from moe_memory.offload.format import ExpertLayout
from moe_memory.offload.pool import SlotPool
from moe_memory.offload.stats import OffloadStats
from moe_memory.offload.warm import WarmTier


class HotTier:
    def __init__(
        self,
        warm: WarmTier,
        n_slots: int,
        *,
        layout: ExpertLayout,
        dtype: torch.dtype,
        device: torch.device,
        stats: OffloadStats | None = None,
        observer=None,
    ) -> None:
        if n_slots < 1:
            raise ValueError("o quente precisa de ao menos uma vaga")
        self.warm = warm
        self.layout = layout
        self.dtype = dtype
        self.device = torch.device(device)
        self.stats = stats if stats is not None else warm.stats
        self.observer = observer
        self.pool = SlotPool(n_slots, layout.block_nbytes, device=self.device, pin=False)
        self.stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        self._wanted: dict[tuple[int, int], int] = {}
        self._events: dict[tuple[int, int], threading.Event] = {}
        self._launched: set[tuple[int, int]] = set()
        self._cuda: dict[tuple[int, int], torch.cuda.Event] = {}
        self._cancel: set[tuple[int, int]] = set()
        self._error: BaseException | None = None
        self._stop = False
        self._cv = threading.Condition()
        warm.add_ready_callback(self._notify)
        self._thread = threading.Thread(target=self._worker, name="hot-copy", daemon=True)
        self._thread.start()

    def sync(self, ranked: list[tuple[int, tuple[int, int]]]) -> None:
        wanted: dict[tuple[int, int], int] = {}
        for step, key in ranked:
            previous = wanted.get(key)
            wanted[key] = step if previous is None else min(previous, step)
        with self._cv:
            self._raise_if_error()
            for key in list(self._wanted):
                if key not in wanted:
                    self._cancel_locked(key)
            for key, step in wanted.items():
                self._request_locked(key, step)
            self._rebalance_locked()
            self._cv.notify_all()

    def ensure(self, key: tuple[int, int]) -> None:
        with self._cv:
            self._raise_if_error()
            event = self._events.get(key)
            if event is None or key not in self._wanted:
                raise KeyError(f"expert {key} não foi pedido ao quente")
            ready = event.is_set()
            cuda_event = self._cuda.get(key)
        if ready:
            return
        t0 = _now()
        if cuda_event is not None:
            cuda_event.synchronize()
            with self._cv:
                self._cuda.pop(key, None)
                if key in self._wanted:
                    self._events[key].set()
                self._cv.notify_all()
        else:
            # O sync pode trocar o Event desta chave. Esperar o objeto
            # antigo para sempre perde o set do novo.
            while True:
                if event.wait(timeout=0.05):
                    break
                with self._cv:
                    self._raise_if_error()
                    current = self._events.get(key)
                    if current is None or key not in self._wanted:
                        raise KeyError(f"expert {key} não foi pedido ao quente")
                    if current.is_set():
                        break
                    event = current
        with self._cv:
            self._raise_if_error()
            if key not in self.pool or not self._events[key].is_set():
                raise RuntimeError(f"expert {key} saiu do quente enquanto era esperado")
        self.stats.note_hot_miss(_now() - t0)

    def views(self, key: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return split_block(self.pool.tensor(key), self.layout, self.dtype)

    def pending_sources(self) -> set[tuple[int, int]]:
        """Chaves cuja cópia ainda lê o bloco morno."""

        with self._cv:
            return {key for key in self._wanted if not self._events[key].is_set()}

    def is_ready(self, key: tuple[int, int]) -> bool:
        self._poll_cuda()
        with self._cv:
            event = self._events.get(key)
            return event is not None and event.is_set()

    def drain(self, timeout: float = 30.0) -> None:
        deadline = _now() + timeout
        while _now() < deadline:
            self._poll_cuda()
            with self._cv:
                self._raise_if_error()
                pending = [key for key in self._wanted if not self._events[key].is_set()]
                if not pending:
                    return
            self._events[pending[0]].wait(timeout=0.05)
        raise TimeoutError("cópia quente não terminou")

    def close(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self._thread.join(timeout=5)

    def _notify(self) -> None:
        with self._cv:
            self._cv.notify_all()

    def _request_locked(self, key: tuple[int, int], step: int) -> None:
        # O passo novo substitui o antigo. O min entre repetições já foi
        # feito em ``sync``. Segurar o passo velho deixa chave urgente
        # demais e, com a VRAM cheia, o token espera para sempre.
        self._wanted[key] = step
        self._cancel.discard(key)
        if key not in self._events:
            self._events[key] = threading.Event()
        if key in self.pool and key not in self._launched and key not in self._cuda:
            self._events[key].set()

    def _cancel_locked(self, key: tuple[int, int]) -> None:
        self._wanted.pop(key, None)
        if key in self._launched or key in self._cuda:
            self._cancel.add(key)
            return
        self.pool.evict(key)
        self._events.pop(key, None)

    def _rebalance_locked(self) -> None:
        while True:
            pending = [
                key
                for key in self._wanted
                if not self._events[key].is_set() and key not in self._launched and key not in self.pool
            ]
            if not pending or self.pool.has_free():
                return
            urgent = min(pending, key=lambda key: (self._wanted[key], key))
            victims = [
                key
                for key in self.pool.occupied()
                if key not in self._launched and self._wanted.get(key, 10**9) > self._wanted[urgent]
            ]
            if not victims:
                return
            victim = max(victims, key=lambda key: (self._wanted.get(key, 10**9), key))
            self.pool.evict(victim)
            if victim in self._wanted:
                self._events[victim] = threading.Event()
            else:
                self._events.pop(victim, None)

    def _worker(self) -> None:
        while True:
            self._poll_cuda()
            key = self._pick()
            if key is None:
                with self._cv:
                    if self._stop:
                        return
                    # Timeout curto: o notify do morno pode chegar antes deste wait.
                    self._cv.wait(timeout=0.01)
                continue
            held = False
            try:
                if not self.warm.hold(key):
                    with self._cv:
                        self._launched.discard(key)
                        if key not in self._cuda:
                            self.pool.evict(key)
                    continue
                held = True
                host = self.warm.tensor(key).detach()
                dest = self.pool.tensor(key)
                if self.stream is None:
                    copy_started = _now() if self.observer is not None else 0.0
                    dest.copy_(host)
                    if self.observer is not None:
                        self.observer.note_h2d_cpu(_now() - copy_started, int(host.nbytes))
                    with self._cv:
                        self._launched.discard(key)
                        if key in self._cancel or key not in self._wanted:
                            self.pool.evict(key)
                            self._cancel.discard(key)
                            self._events.pop(key, None)
                        else:
                            self._events[key].set()
                        self._cv.notify_all()
                else:
                    timing = self.observer is not None
                    start = torch.cuda.Event(enable_timing=True) if timing else None
                    event = torch.cuda.Event(enable_timing=timing)
                    with torch.cuda.stream(self.stream):
                        if start is not None:
                            start.record(self.stream)
                        dest.copy_(host, non_blocking=True)
                        event.record(self.stream)
                    if start is not None and self.observer is not None:
                        self.observer.note_h2d_gpu(self.stream, start, event, int(host.nbytes))
                    with self._cv:
                        self._launched.discard(key)
                        self._cuda[key] = event
                    held = False  # o poll solta o morno quando o event completa
            except Exception as exc:
                with self._cv:
                    self._error = exc
                    self._launched.discard(key)
                    event = self._events.get(key)
                    if event is not None:
                        event.set()
                    self._cv.notify_all()
                return
            finally:
                if held:
                    self.warm.release_hold(key)

    def _pick(self) -> tuple[int, int] | None:
        with self._cv:
            snapshot = list(self._wanted.items())
        ready: list[tuple[int, tuple[int, int]]] = []
        for key, step in snapshot:
            if self.warm.is_ready(key):
                ready.append((step, key))
        ready.sort()
        with self._cv:
            for _step, key in ready:
                if key not in self._wanted or self._events[key].is_set() or key in self._launched or key in self._cuda:
                    continue
                if self.pool.reserve(key) is None:
                    self._evict_for_locked(key)
                    if self.pool.reserve(key) is None:
                        return None
                self._launched.add(key)
                return key
        return None

    def _evict_for_locked(self, urgent: tuple[int, int]) -> None:
        """Solta um residente menos urgente para a chave que vai executar."""

        urgent_step = self._wanted.get(urgent, 0)
        victims = [
            key
            for key in self.pool.occupied()
            if key not in self._launched
            and key not in self._cuda
            and key != urgent
            and self._wanted.get(key, 10**9) > urgent_step
        ]
        if not victims:
            return
        victim = max(victims, key=lambda key: (self._wanted.get(key, 10**9), key))
        self.pool.evict(victim)
        if victim in self._wanted:
            self._events[victim] = threading.Event()
        else:
            self._events.pop(victim, None)

    def _poll_cuda(self) -> None:
        if not self._cuda:
            return
        done: list[tuple[int, int]] = []
        for key, event in list(self._cuda.items()):
            if event.query():
                done.append(key)
        if not done:
            return
        for key in done:
            self.warm.release_hold(key)
        with self._cv:
            for key in done:
                self._cuda.pop(key, None)
                if key in self._cancel or key not in self._wanted:
                    self.pool.evict(key)
                    self._cancel.discard(key)
                    self._events.pop(key, None)
                else:
                    self._events[key].set()
            self._cv.notify_all()

    def _raise_if_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("cópia para o quente falhou") from self._error


def _now() -> float:
    import time

    return time.perf_counter()
