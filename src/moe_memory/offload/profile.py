"""Instrumentação opt-in do decode. Desligada, não cria event nem sincroniza.

Os relógios da thread de decode são ``perf_counter`` (parede). O SSD é o
tempo do worker em volta do ``read_into``, e a espera na fila é outro
número: os dois podem se sobrepor entre threads e não entram em
``elapsed_s``. A cópia H2D e o compute na GPU são ``cuda.Event`` no
stream que de fato enfileira o trabalho. ``elapsed_time`` só roda depois
que ``elapsed_s`` já foi fechado, num ``stream.synchronize`` por stream,
não num ``torch.cuda.synchronize`` por token ou por camada.

Soma de estágios concorrentes não é o tempo de parede. O tempo para
enfileirar o prefetch não é o tempo em que a cópia termina.
"""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import torch


def thread_cpu_seconds() -> float | None:
    """CPU da thread chamadora. ``None`` se a plataforma não tiver ``thread_time``.

    Não é parede e não entra na soma de ``elapsed_s``.
    """

    fn = getattr(time, "thread_time", None)
    if fn is None:
        return None
    try:
        return float(fn())
    except (OSError, ValueError):
        return None


class HostSpan:
    """Parede e CPU da thread em volta de um trecho. Não sincroniza a GPU."""

    def __init__(self) -> None:
        self.wall_s = 0.0
        self.thread_cpu_s: float | None = None
        self._wall = 0.0
        self._cpu: float | None = None

    def __enter__(self) -> "HostSpan":
        self._wall = time.perf_counter()
        self._cpu = thread_cpu_seconds()
        return self

    def __exit__(self, *_exc: object) -> bool:
        self.wall_s = time.perf_counter() - self._wall
        end = thread_cpu_seconds()
        if self._cpu is None or end is None:
            self.thread_cpu_s = None
        else:
            self.thread_cpu_s = end - self._cpu
        return False


def stream_key(stream: object) -> int:
    """Identidade do stream CUDA, não do objeto Python.

    ``current_stream()`` devolve um wrapper novo a cada chamada. Usar
    ``id`` conta um synchronize por span.
    """

    handle = getattr(stream, "cuda_stream", None)
    if handle is None:
        return id(stream)
    return int(handle)


def percentile(values: list[float], pct: float) -> float | None:
    """Posto mais próximo. ``p50`` de ``[1, 2, 3, 4]`` é 2; ``p95`` é 4.

    Lista vazia devolve ``None``. Não inventa zero.
    """

    if not values:
        return None
    if not 0 < pct <= 100:
        raise ValueError("percentil precisa estar em (0, 100]")
    ordered = sorted(values)
    index = max(0, math.ceil(pct / 100 * len(ordered)) - 1)
    return float(ordered[index])


def summarize(values: list[float]) -> dict | None:
    """p50/p95 e a mediana da primeira metade contra a da segunda.

    Com um único valor, cedo e tarde são esse valor. A primeira metade é
    ``valores[:n//2]``; se ``n`` for 1, as duas metades usam o valor.
    """

    if not values:
        return None
    mid = len(values) // 2
    early = values[:mid] if mid else values
    late = values[mid:] if mid else values
    return {
        "n": len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "early_p50": percentile(early, 50),
        "late_p50": percentile(late, 50),
        "sum": float(sum(values)),
        "kind": "ver o campo clock do relatório; a soma não é parede se os intervalos se sobrepõem",
    }


def cgroup_memory_current() -> int | None:
    """Bytes do ``memory.current`` se o cgroup v2 expuser. Senão ``None``."""

    path = Path("/sys/fs/cgroup/memory.current")
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


class DecodeProfile:
    """Coleta limitada. Passou do teto, o resto do decode segue sem amostra."""

    def __init__(
        self,
        *,
        device: torch.device,
        max_tokens: int = 512,
        max_spans: int = 16384,
        max_io: int = 4096,
    ) -> None:
        if max_tokens < 1 or max_spans < 1 or max_io < 1:
            raise ValueError("tetos do perfil precisam ser >= 1")
        self.device = torch.device(device)
        self.max_tokens = max_tokens
        self.max_spans = max_spans
        self.max_io = max_io
        self.detail_truncated = False
        self.span_truncated = False
        self.io_truncated = False
        self.per_token: list[dict] = []
        self.resolve_wall_s: float | None = None
        self.resolve_stream_syncs = 0
        self.cgroup_memory_current_bytes_start: int | None = None
        self.cgroup_memory_current_bytes_end: int | None = None
        self._lock = threading.Lock()
        self._ssd_io: list[float] = []
        self._ssd_queue: list[float] = []
        self._ssd_bytes: list[int] = []
        self._h2d_cpu: list[float] = []
        self._h2d_spans: list[tuple] = []
        self._compute_spans: list[tuple] = []
        self._cpu_acc = {"attention": 0.0, "expert": 0.0}
        self._open_wall: dict[str, float] = {}
        self._open_event: dict[str, torch.cuda.Event] = {}
        self._cuda = self.device.type == "cuda" and torch.cuda.is_available()

    def accepting_token(self) -> bool:
        if len(self.per_token) >= self.max_tokens:
            self.detail_truncated = True
            return False
        return True

    def begin(self, kind: str) -> None:
        if kind not in self._cpu_acc:
            raise ValueError("span é attention ou expert")
        self._open_wall[kind] = time.perf_counter()
        if not self._cuda or self.span_truncated:
            return
        if len(self._compute_spans) >= self.max_spans:
            self.span_truncated = True
            return
        event = torch.cuda.Event(enable_timing=True)
        event.record(torch.cuda.current_stream(self.device))
        self._open_event[kind] = event

    def end(self, kind: str, token: int) -> None:
        started = self._open_wall.pop(kind, None)
        if started is not None:
            self._cpu_acc[kind] += time.perf_counter() - started
        event = self._open_event.pop(kind, None)
        if event is None or not self._cuda:
            return
        if len(self._compute_spans) >= self.max_spans:
            self.span_truncated = True
            return
        end = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream(self.device)
        end.record(stream)
        self._compute_spans.append((kind, token, event, end, stream))

    def consume_cpu(self) -> tuple[float, float]:
        attention = self._cpu_acc["attention"]
        expert = self._cpu_acc["expert"]
        self._cpu_acc["attention"] = 0.0
        self._cpu_acc["expert"] = 0.0
        return attention, expert

    def add_token(self, row: dict) -> None:
        if len(self.per_token) >= self.max_tokens:
            self.detail_truncated = True
            return
        self.per_token.append(row)

    def note_ssd_read(self, queue_s: float | None, io_s: float, nbytes: int) -> None:
        with self._lock:
            if len(self._ssd_io) >= self.max_io:
                self.io_truncated = True
                return
            self._ssd_io.append(io_s)
            if queue_s is not None:
                self._ssd_queue.append(queue_s)
            self._ssd_bytes.append(nbytes)

    def note_h2d_cpu(self, copy_s: float, nbytes: int) -> None:
        del nbytes
        with self._lock:
            if len(self._h2d_cpu) >= self.max_io:
                self.io_truncated = True
                return
            self._h2d_cpu.append(copy_s)

    def note_h2d_gpu(self, stream: torch.cuda.Stream, start: torch.cuda.Event, end: torch.cuda.Event, nbytes: int) -> None:
        del nbytes
        with self._lock:
            if len(self._h2d_spans) >= self.max_spans:
                self.span_truncated = True
                return
            self._h2d_spans.append((stream, start, end))

    def resolve(self) -> None:
        """Fecha events já gravados. Chamar depois de fixar ``elapsed_s``."""

        started = time.perf_counter()
        with self._lock:
            h2d_spans = list(self._h2d_spans)
            self._h2d_spans.clear()
        cuda_streams: list[torch.cuda.Stream] = []
        identities: set[int] = set()
        for item in self._compute_spans:
            stream = item[4]
            ident = stream_key(stream)
            if ident not in identities:
                identities.add(ident)
                cuda_streams.append(stream)
        for stream, _start, _end in h2d_spans:
            ident = stream_key(stream)
            if ident not in identities:
                identities.add(ident)
                cuda_streams.append(stream)
        for stream in cuda_streams:
            stream.synchronize()
            self.resolve_stream_syncs += 1
        gpu_attention: dict[int, float] = {}
        gpu_expert: dict[int, float] = {}
        for kind, token, start, end, _stream in self._compute_spans:
            seconds = float(start.elapsed_time(end)) / 1000.0
            bucket = gpu_attention if kind == "attention" else gpu_expert
            bucket[token] = bucket.get(token, 0.0) + seconds
        h2d_gpu: list[float] = []
        for _stream, start, end in h2d_spans:
            h2d_gpu.append(float(start.elapsed_time(end)) / 1000.0)
        for index, row in enumerate(self.per_token):
            row["attention_gpu_s"] = gpu_attention.get(index) if self._cuda else None
            row["expert_gpu_s"] = gpu_expert.get(index) if self._cuda else None
        self._h2d_gpu_resolved = h2d_gpu
        self._compute_spans.clear()
        self.resolve_wall_s = time.perf_counter() - started

    def report(self) -> dict:
        with self._lock:
            ssd_io = list(self._ssd_io)
            ssd_queue = list(self._ssd_queue)
            ssd_bytes = list(self._ssd_bytes)
            h2d_cpu = list(self._h2d_cpu)
        h2d_gpu = list(getattr(self, "_h2d_gpu_resolved", []))
        if self._cuda and h2d_gpu:
            h2d_kind: str | None = "cuda_event"
        elif h2d_cpu:
            h2d_kind = "cpu_memcpy"
        else:
            h2d_kind = None
        rows = self.per_token

        def column(name: str) -> list[float]:
            return [float(row[name]) for row in rows if row.get(name) is not None]

        gpu_supported = bool(self._cuda)
        return {
            "enabled": True,
            "detail_truncated": self.detail_truncated,
            "span_truncated": self.span_truncated,
            "io_truncated": self.io_truncated,
            "max_tokens": self.max_tokens,
            "max_spans": self.max_spans,
            "max_io": self.max_io,
            "resolve_wall_s": self.resolve_wall_s,
            "resolve_stream_syncs": self.resolve_stream_syncs,
            "device_synchronize": False,
            "cgroup_memory_current_bytes_start": self.cgroup_memory_current_bytes_start,
            "cgroup_memory_current_bytes_end": self.cgroup_memory_current_bytes_end,
            "cgroup_note": (
                "Leitura de memory.current se existir. É observação de "
                "contenção possível, não causa do tempo."
            ),
            "per_token": rows,
            "summary": {
                "clock": "perf_counter na thread de decode, salvo indicação",
                "router_wall_s": summarize(column("router_wall_s")),
                "schedule_enqueue_wall_s": summarize(column("schedule_enqueue_wall_s")),
                "forward_wall_s": summarize(column("forward_wall_s")),
                "stall_warm_wall_s": summarize(column("stall_warm_wall_s")),
                "stall_hot_wall_s": summarize(column("stall_hot_wall_s")),
                "attention_cpu_wall_s": summarize(column("attention_cpu_wall_s")),
                "expert_cpu_wall_s": summarize(column("expert_cpu_wall_s")),
                "token_wall_s": summarize(column("token_wall_s")),
                "unaccounted_wall_s": summarize(column("unaccounted_wall_s")),
                "attention_gpu_s": summarize(column("attention_gpu_s")) if gpu_supported else None,
                "expert_gpu_s": summarize(column("expert_gpu_s")) if gpu_supported else None,
                "schedule_ids_wall_s": summarize(column("schedule_ids_wall_s")),
                "schedule_lookahead_plan_wall_s": summarize(column("schedule_lookahead_plan_wall_s")),
                "schedule_lookahead_plan_thread_cpu_s": summarize(column("schedule_lookahead_plan_thread_cpu_s")),
                "schedule_rank_wall_s": summarize(column("schedule_rank_wall_s")),
                "schedule_hot_sync_wall_s": summarize(column("schedule_hot_sync_wall_s")),
                "schedule_hot_sync_lock_wait_s": summarize(column("schedule_hot_sync_lock_wait_s")),
                "schedule_hot_sync_locked_s": summarize(column("schedule_hot_sync_locked_s")),
                "schedule_warm_sync_wall_s": summarize(column("schedule_warm_sync_wall_s")),
                "schedule_warm_sync_lock_wait_s": summarize(column("schedule_warm_sync_lock_wait_s")),
                "schedule_warm_sync_locked_s": summarize(column("schedule_warm_sync_locked_s")),
                "schedule_warm_callback_wall_s": summarize(column("schedule_warm_callback_wall_s")),
                "schedule_enqueue_thread_cpu_s": summarize(column("schedule_enqueue_thread_cpu_s")),
                "schedule_unaccounted_wall_s": summarize(column("schedule_unaccounted_wall_s")),
            },
            "schedule_note": (
                "Tudo isto é a thread de decode dentro de _install. "
                "schedule_enqueue_wall_s é a parede do conjunto. As partes "
                "ids, lookahead_plan, rank, hot_sync e warm_sync são sequenciais "
                "nessa thread e aproximam essa parede; o que sobra é "
                "schedule_unaccounted_wall_s. lock_wait e locked estão dentro "
                "da parede do sync correspondente e não se somam de novo. "
                "callback está dentro da parede do warm_sync. thread_cpu é "
                "time.thread_time e não se soma à parede. null em "
                "lookahead_plan ou rank significa que o modo não chamou a "
                "função, não que durou zero. A parede pode absorver espera de "
                "lock. Não há cuda.Event.synchronize nem torch.cuda.synchronize "
                "dentro de _install. O plano fica no device do LSTM; o bench "
                "deixa o LSTM na CPU. Se a parede for muito maior que a CPU da "
                "thread, a thread não estava executando; isso ainda não separa "
                "lock de um spin de driver, e nesta máquina de teste não há CUDA."
            ),
            "ssd_worker": {
                "clock": "perf_counter no worker, em volta de read_into",
                "additive_to_elapsed": False,
                "overlaps": True,
                "io_s_sum": float(sum(ssd_io)) if ssd_io else None,
                "io_s": summarize(ssd_io),
                "queue_wait_s_sum": float(sum(ssd_queue)) if ssd_queue else None,
                "queue_wait_s": summarize(ssd_queue),
                "reads": len(ssd_io),
                "bytes": int(sum(ssd_bytes)) if ssd_bytes else None,
                "note": (
                    "io_s é serviço do read. queue_wait_s é da entrada do "
                    "pedido até o worker começar o read. Com mais de uma "
                    "thread a soma não é parede e não é stall."
                ),
            },
            "h2d": {
                "kind": h2d_kind,
                "additive_to_elapsed": False,
                "cpu_memcpy_s_sum": float(sum(h2d_cpu)) if h2d_cpu else None,
                "gpu_event_s_sum": float(sum(h2d_gpu)) if h2d_gpu else None,
                "note": (
                    "gpu_event_s_sum é a soma de cuda.Event no stream de "
                    "cópia, resolvida depois de elapsed_s. Não é a duração "
                    "do enqueue e pode sobrepor o compute."
                ),
            },
            "compute_gpu": None
            if not gpu_supported
            else {
                "kind": "cuda_event",
                "stream": "stream corrente do forward, não o stream da cópia",
                "additive_to_elapsed": False,
                "attention_s_sum": _sum_or_none(column("attention_gpu_s")),
                "expert_s_sum": _sum_or_none(column("expert_gpu_s")),
                "note": (
                    "Eventos em volta da atenção e de apply_experts. O "
                    "ensure do expert fica fora do intervalo. A resolução "
                    "é um synchronize por stream no fim, fora de elapsed_s."
                ),
            },
            "overhead_note": (
                "Desligado, este objeto não existe: sem event e sem "
                "synchronize da instrumentação. Ligado, o loop ganha "
                "perf_counter e, na GPU, events com enable_timing. O "
                "synchronize dos streams da instrumentação fica em "
                "resolve_wall_s, depois de elapsed_s."
            ),
        }


def _sum_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values))
