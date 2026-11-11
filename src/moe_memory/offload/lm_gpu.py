"""Compute de uma camada na GPU. O modelo inteiro não sobe.

O residente e o store ficam na CPU. A cópia usa memória pinned e um
stream à parte, no mesmo desenho do H2D da inferência. O Adam, o
gradiente mestre da LoRA e o LSTM continuam na RAM.

Pascal (capability < 8) não executa bf16: o compute nesse caso é fp32.
Ampere ou mais novo pode usar bf16.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

VRAM_OVERHEAD_BYTES = 256 * 1024**2


def compute_dtype_for(device: torch.device) -> torch.dtype:
    """bf16 só a partir de Ampere. A 1060 fica em fp32 na GPU."""

    if device.type != "cuda":
        return torch.float32
    capability = torch.cuda.get_device_capability(device)
    if capability[0] >= 8 and bool(torch.cuda.is_bf16_supported()):
        return torch.bfloat16
    return torch.float32


def host_compute_threads(cpu_count: int, cuda: bool) -> int:
    """Na GPU, uma thread fica livre para o prefetch e a cópia."""

    count = max(int(cpu_count), 1)
    if not cuda:
        return count
    return max(1, count - 1)


def _attention_bytes(hidden: int, n_heads: int, n_kv_heads: int, head_dim: int, elem: int) -> int:
    q = n_heads * head_dim * hidden * elem
    kv = n_kv_heads * head_dim * hidden * elem
    out = hidden * n_heads * head_dim * elem
    norms = (hidden * 2 + head_dim * 2) * elem
    bias = (n_heads * head_dim + 2 * n_kv_heads * head_dim + hidden) * elem
    return q + 2 * kv + out + norms + bias


def _expert_bytes(hidden: int, intermediate: int, elem: int) -> int:
    return 3 * hidden * intermediate * elem


def estimate_pinned_bytes(
    *,
    hidden: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    intermediate: int,
    k2: int,
    resident_elem: int = 2,
) -> int:
    """Um buffer pinned da camada. A cópia anterior já está na GPU."""

    if min(hidden, n_heads, n_kv_heads, head_dim, intermediate, k2, resident_elem) < 1:
        raise ValueError("pinned com dimensão < 1")
    return _attention_bytes(hidden, n_heads, n_kv_heads, head_dim, resident_elem) + k2 * _expert_bytes(
        hidden, intermediate, resident_elem
    )


def estimate_vram_bytes(
    *,
    hidden: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    intermediate: int,
    seq_len: int,
    k2: int,
    rank: int,
    compute_elem: int,
) -> dict[str, int]:
    """Duas camadas no device (a que computa e a que está copiando) mais a ativação."""

    if min(hidden, n_heads, n_kv_heads, head_dim, intermediate, seq_len, k2, rank, compute_elem) < 1:
        raise ValueError("VRAM com dimensão < 1")
    attention = _attention_bytes(hidden, n_heads, n_kv_heads, head_dim, compute_elem)
    expert = _expert_bytes(hidden, intermediate, compute_elem)
    lora = k2 * 3 * (rank * hidden + intermediate * rank) * compute_elem
    activation = seq_len * hidden * compute_elem * 16 + seq_len * n_heads * seq_len * compute_elem
    one = attention + k2 * expert
    total = 2 * one + lora + activation + VRAM_OVERHEAD_BYTES
    return {
        "vram_attention_bytes": attention,
        "vram_experts_bytes": k2 * expert,
        "vram_lora_bytes": lora,
        "vram_activation_bytes": activation,
        "vram_overhead_bytes": VRAM_OVERHEAD_BYTES,
        "vram_layers_on_device": 2,
        "vram_bytes": total,
    }


def assert_vram_fits(vram_bytes: int, budget: int) -> None:
    if budget < 1:
        raise ValueError("orçamento de VRAM precisa ser >= 1 byte")
    if vram_bytes > budget:
        raise ValueError(
            f"VRAM estimada {vram_bytes} bytes passa de {budget}. "
            "Não carrego o modelo inteiro na GPU."
        )


@dataclass
class _Pack:
    layer: object
    experts: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    pinned: list[torch.Tensor]


class LayerTransport:
    """H2D de uma camada. O stream de cópia não é o de compute."""

    def __init__(self, model) -> None:
        self.model = model
        self.device = model.compute_device
        self.dtype = model.compute_dtype
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self._ready: _Pack | None = None
        self._ready_index = -1
        self._pending: _Pack | None = None
        self._pending_index = -1
        self._pending_event: torch.cuda.Event | None = None
        self._h2d: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._d2h: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._compute: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self.h2d_bytes = 0
        self.d2h_bytes = 0

    def close(self) -> None:
        self.reset()

    def reset(self) -> None:
        if self._pending_event is not None:
            self._pending_event.synchronize()
        self._ready = None
        self._ready_index = -1
        self._pending = None
        self._pending_index = -1
        self._pending_event = None
        self._h2d.clear()
        self._d2h.clear()
        self._compute.clear()
        self.h2d_bytes = 0
        self.d2h_bytes = 0

    def ensure(self, layer_index: int, layer, blobs: dict[tuple[int, int], tuple[bytes, bytes, bytes]]) -> _Pack:
        if self._ready is not None and self._ready_index == layer_index:
            return self._ready
        if self._pending is not None and self._pending_index == layer_index:
            self._wait_pending()
            return self._ready
        if self._pending_event is not None:
            # A direção mudou. A cópia em voo precisa terminar antes de soltar o destino.
            self._pending_event.synchronize()
        self._pending = None
        self._pending_event = None
        pack, _event = self._load(layer_index, layer, blobs, block=True)
        self._ready = pack
        self._ready_index = layer_index
        return pack

    def stage_ahead(
        self,
        layer_index: int,
        layer,
        blobs: dict[tuple[int, int], tuple[bytes, bytes, bytes]],
    ) -> None:
        if self._pending is not None and self._pending_index == layer_index:
            return
        if self._ready is not None and self._ready_index == layer_index:
            return
        self._pending = None
        self._pending_event = None
        pack, event = self._load(layer_index, layer, blobs, block=False)
        self._pending = pack
        self._pending_index = layer_index
        self._pending_event = event

    def take_expert(self, layer: int, expert: int):
        key = (layer, expert)
        for pack in (self._ready, self._pending):
            if pack is None:
                continue
            found = pack.experts.get(key)
            if found is not None:
                return found
        return None

    def h2d_now(self, layer: int, expert: int, blobs: tuple[bytes, bytes, bytes]):
        gate, up, down = self._matrices(blobs, block=True)
        if self._ready is not None:
            self._ready.experts[(layer, expert)] = (gate, up, down)
        return gate, up, down

    def note_compute(self, start: torch.cuda.Event, end: torch.cuda.Event) -> None:
        self._compute.append((start, end))

    def note_d2h(self, start: torch.cuda.Event, end: torch.cuda.Event, nbytes: int) -> None:
        self._d2h.append((start, end))
        self.d2h_bytes += nbytes

    def resolve(self) -> dict[str, float | int]:
        """Lê os eventos. O synchronize só acontece com ``--time-log``."""

        if not self.model.time_log:
            return {
                "h2d_s": 0.0,
                "d2h_s": 0.0,
                "h2d_wait_s": float(self.model._times.get("h2d_wait_s", 0.0)),
                "h2d_wait_backward_s": float(self.model._times.get("h2d_wait_backward_s", 0.0)),
                "gpu_compute_s": 0.0,
                "h2d_bytes": self.h2d_bytes,
                "d2h_bytes": self.d2h_bytes,
            }
        torch.cuda.synchronize(self.device)
        return {
            "h2d_s": _elapsed(self._h2d),
            "d2h_s": _elapsed(self._d2h),
            "h2d_wait_s": float(self.model._times.get("h2d_wait_s", 0.0)),
            "h2d_wait_backward_s": float(self.model._times.get("h2d_wait_backward_s", 0.0)),
            "gpu_compute_s": _elapsed(self._compute),
            "h2d_bytes": self.h2d_bytes,
            "d2h_bytes": self.d2h_bytes,
        }

    def check_budget(self) -> None:
        peak = int(torch.cuda.max_memory_allocated(self.device))
        if peak > self.model.vram_budget:
            raise RuntimeError(
                f"VRAM alocada {peak} bytes passou do orçamento {self.model.vram_budget}. "
                "Não carrego o modelo inteiro na GPU."
            )

    def _wait_pending(self) -> None:
        event = self._pending_event
        if event is not None:
            if self.model.time_log:
                started = time.perf_counter()
                event.synchronize()
                self._note_wait(time.perf_counter() - started)
            else:
                torch.cuda.current_stream(self.device).wait_stream(self.copy_stream)
        if self._pending is not None:
            self._pending.pinned.clear()
        self._ready = self._pending
        self._ready_index = self._pending_index
        self._pending = None
        self._pending_event = None

    def _load(self, layer_index: int, layer, blobs: dict, *, block: bool):
        pinned: list[torch.Tensor] = []
        experts: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        start = end = None
        if self.model.time_log:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(self.copy_stream)
        with torch.cuda.stream(self.copy_stream):
            view = _GpuLayer(
                input_norm=self._h2d(layer.input_norm, pinned),
                post_norm=self._h2d(layer.post_norm, pinned),
                q=self._h2d(layer.q, pinned),
                k=self._h2d(layer.k, pinned),
                v=self._h2d(layer.v, pinned),
                o=self._h2d(layer.o, pinned),
                q_norm=self._h2d(layer.q_norm, pinned),
                k_norm=self._h2d(layer.k_norm, pinned),
                q_bias=self._h2d(layer.q_bias, pinned),
                k_bias=self._h2d(layer.k_bias, pinned),
                v_bias=self._h2d(layer.v_bias, pinned),
                o_bias=self._h2d(layer.o_bias, pinned),
            )
            for key, raw in blobs.items():
                experts[key] = self._matrices(raw, pinned)
        if end is not None:
            end.record(self.copy_stream)
            self._h2d.append((start, end))
        event = torch.cuda.Event()
        event.record(self.copy_stream)
        pack = _Pack(layer=view, experts=experts, pinned=pinned)
        if block:
            if self.model.time_log:
                started = time.perf_counter()
                event.synchronize()
                self._note_wait(time.perf_counter() - started)
            else:
                torch.cuda.current_stream(self.device).wait_stream(self.copy_stream)
            pack.pinned.clear()
        return pack, event

    def _h2d(self, weight: torch.Tensor | None, pinned: list[torch.Tensor]) -> torch.Tensor | None:
        if weight is None:
            return None
        host = weight.detach()
        if host.device.type != "cpu":
            host = host.to("cpu")
        if not host.is_pinned():
            host = host.pin_memory()
        pinned.append(host)
        gpu = host.to(device=self.device, dtype=self.dtype, non_blocking=True)
        self.h2d_bytes += int(host.nbytes)
        return gpu

    def _matrices(self, blobs: tuple[bytes, bytes, bytes], pinned: list[torch.Tensor] | None = None, block: bool = False):
        layout = self.model.store.layout
        shapes = (
            (layout.intermediate, layout.hidden),
            (layout.intermediate, layout.hidden),
            (layout.hidden, layout.intermediate),
        )
        out = []
        hold = pinned if pinned is not None else []
        start = end = None
        if block and self.model.time_log:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(self.copy_stream)
        with torch.cuda.stream(self.copy_stream):
            for raw, shape in zip(blobs, shapes, strict=True):
                host = _host_matrix(raw, self.model.store_dtype, shape)
                if not host.is_pinned():
                    host = host.pin_memory()
                hold.append(host)
                gpu = host.to(device=self.device, dtype=self.dtype, non_blocking=True)
                self.h2d_bytes += int(host.nbytes)
                out.append(gpu)
        if block:
            event = torch.cuda.Event()
            event.record(self.copy_stream)
            if end is not None:
                end.record(self.copy_stream)
                self._h2d.append((start, end))
            if self.model.time_log:
                started = time.perf_counter()
                event.synchronize()
                self._note_wait(time.perf_counter() - started)
            else:
                torch.cuda.current_stream(self.device).wait_stream(self.copy_stream)
            if pinned is None:
                hold.clear()
        return tuple(out)

    def _note_wait(self, waited: float) -> None:
        if self.model._io_phase == "forward":
            self.model._add_time("h2d_wait_s", waited)
        else:
            self.model._add_time("h2d_wait_backward_s", waited)


class _GpuLayer:
    def __init__(self, **weights) -> None:
        self.__dict__.update(weights)


def _host_matrix(raw: bytes, dtype: torch.dtype, shape: tuple[int, int]) -> torch.Tensor:
    cpu = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape)
    if not cpu.is_contiguous():
        cpu = cpu.contiguous()
    return cpu


def _elapsed(pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]]) -> float:
    total = 0.0
    for start, end in pairs:
        total += float(start.elapsed_time(end)) / 1000.0
    return total
