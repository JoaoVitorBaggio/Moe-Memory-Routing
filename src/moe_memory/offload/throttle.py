"""Teto de escrita e fsync em pedaços.

Um download de ~61 GB, sem limite, encheu a page cache suja e travou o
PC. O teto segura a taxa; o fsync periódico devolve as páginas sujas
antes que elas virem um stall do sistema. ``DONTNEED`` solta do cache
o que já foi sincronizado, para a leitura seguinte não disputar RAM.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable


class IoThrottle:
    """Conta bytes escritos. ``max_bytes_per_sec`` None não dorme."""

    def __init__(
        self,
        fds: list[int] | None = None,
        *,
        max_bytes_per_sec: float | None = 80 * 1024 * 1024,
        fsync_every: int = 64 * 1024 * 1024,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
        fsync: Callable[[int], None] = os.fsync,
        advise: Callable[[int], None] | None = None,
    ) -> None:
        if fsync_every < 1:
            raise ValueError("fsync_every precisa ser >= 1")
        if max_bytes_per_sec is not None and max_bytes_per_sec <= 0:
            raise ValueError("max_bytes_per_sec precisa ser > 0 ou None")
        self.fds = list(fds or [])
        self.max_bytes_per_sec = max_bytes_per_sec
        self.fsync_every = fsync_every
        self._clock = clock
        self._sleep = sleep
        self._fsync = fsync
        self._advise = advise if advise is not None else _advise_dontneed
        self.written = 0
        self.since_sync = 0
        self.syncs = 0
        self._t0 = clock()

    def add_fd(self, fd: int) -> None:
        if fd not in self.fds:
            self.fds.append(fd)

    def account(self, n: int) -> None:
        if n < 0:
            raise ValueError("não se contabiliza byte negativo")
        if n == 0:
            return
        self.written += n
        self.since_sync += n
        while self.since_sync >= self.fsync_every:
            self.sync()
            self.since_sync -= self.fsync_every
        self._pace()

    def sync(self) -> None:
        for fd in self.fds:
            self._fsync(fd)
            self._advise(fd)
        self.syncs += 1

    def finish(self) -> None:
        self.sync()
        self.since_sync = 0

    def _pace(self) -> None:
        if self.max_bytes_per_sec is None:
            return
        target = self.written / self.max_bytes_per_sec
        delay = target - (self._clock() - self._t0)
        if delay > 0:
            self._sleep(delay)


def _advise_dontneed(fd: int) -> None:
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
