"""Uma linha por fase, no stderr, já com flush.

Serve para achar em que passo a memória estourou se o processo morrer
antes do JSON. Não sincroniza a GPU e não entra na agenda do expert.

Os números não são a mesma coisa:

- ``ru_maxrss_bytes`` e ``vm_hwm_bytes`` são picos do processo, não o uso agora;
- ``vm_rss_bytes`` é o RSS atual (``VmRSS``), com páginas de arquivo que
  estejam residentes no processo;
- ``vm_pin_bytes`` é o ``VmPin`` do kernel (mlock). A memória pinned do
  CUDA pode não aparecer aqui; ela costuma entrar no RSS e no
  ``memory.current``;
- ``cgroup_memory_current_bytes`` é o cgroup v2, se existir. Inclui cache
  de página e memória de kernel cobrados desse cgroup. Não é o RSS;
- ``tracked_pool_bytes`` é a soma dos ``SlotPool`` que ``release`` ainda
  não soltou. Não inclui o residente, o cache de página nem a GPU fora
  desses blocos.
"""

from __future__ import annotations

import resource
import sys
from pathlib import Path

from moe_memory.offload.pool import live_nbytes
from moe_memory.offload.profile import cgroup_memory_current


def snapshot() -> dict[str, int | None]:
    return {
        "ru_maxrss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "vm_rss_bytes": _status_bytes("VmRSS"),
        "vm_hwm_bytes": _status_bytes("VmHWM"),
        "vm_pin_bytes": _status_bytes("VmPin"),
        "cgroup_memory_current_bytes": cgroup_memory_current(),
        "tracked_pool_bytes": live_nbytes(),
    }


def emit_phase(name: str) -> dict[str, int | str | None]:
    snap = snapshot()
    row: dict[str, int | str | None] = {"phase": name, **snap}
    parts = [f"phase={name}"]
    for key, value in snap.items():
        parts.append(f"{key}={'null' if value is None else value}")
    print(" ".join(parts), file=sys.stderr, flush=True)
    return row


def _status_bytes(field: str) -> int | None:
    path = Path("/proc/self/status")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    prefix = field + ":"
    for line in text.splitlines():
        if line.startswith(prefix):
            bits = line.split()
            if len(bits) < 2:
                return None
            return int(bits[1]) * 1024
    return None
