"""Compute de uma camada na GPU. O modelo inteiro não sobe.

O residente e o store ficam na CPU. A cópia usa memória pinned e um
stream à parte, no mesmo desenho do H2D da inferência. O Adam, o
gradiente mestre da LoRA e o LSTM continuam na RAM.

Pascal (capability < 8) não executa bf16: o compute nesse caso é fp32.
Ampere ou mais novo pode usar bf16.
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass

import torch

VRAM_OVERHEAD_BYTES = 256 * 1024**2
# Pico medido na 1060 com cache de 3 GiB e o top-8: 5478979584 bytes.
# A fórmula daquele desenho ficava perto de 4,2 GiB. A diferença entra
# só no encaixe do modo qwen, para o cache encolher antes do check_budget.
QWEN_VRAM_MEASURED_PEAK = 5_478_979_584
QWEN_VRAM_FIT_SLACK = 1024**3
QWEN_VRAM_MIN_MARGIN_BYTES = 300 * 1024**2
QWEN_VRAM_MEASURED_BASELINE_SEQ = 32
QWEN_VRAM_MEASURED_BASELINE_TOPK = 8


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
    vram_expert_cache_bytes: int = 0,
    ce_bytes: int = 0,
) -> dict[str, int]:
    """Duas camadas de atenção, o cache de experts e a fatia da cabeça."""

    if min(hidden, n_heads, n_kv_heads, head_dim, intermediate, seq_len, k2, rank, compute_elem) < 1:
        raise ValueError("VRAM com dimensão < 1")
    if vram_expert_cache_bytes < 0 or ce_bytes < 0:
        raise ValueError("cache de VRAM e a cabeça precisam ser >= 0")
    attention = _attention_bytes(hidden, n_heads, n_kv_heads, head_dim, compute_elem)
    expert = _expert_bytes(hidden, intermediate, compute_elem)
    lora = k2 * 3 * (rank * hidden + intermediate * rank) * compute_elem
    activation = seq_len * hidden * compute_elem * 16 + seq_len * n_heads * seq_len * compute_elem
    live_experts = 2 * k2 * expert
    total = 2 * attention + live_experts + vram_expert_cache_bytes + lora + activation + ce_bytes + VRAM_OVERHEAD_BYTES
    return {
        "vram_attention_bytes": attention,
        "vram_experts_bytes": k2 * expert,
        "vram_expert_cache_bytes": vram_expert_cache_bytes,
        "vram_lm_head_bytes": ce_bytes,
        "vram_lora_bytes": lora,
        "vram_activation_bytes": activation,
        "vram_overhead_bytes": VRAM_OVERHEAD_BYTES,
        "vram_layers_on_device": 2,
        "vram_bytes": total,
    }


def _activation_vram_bytes(seq_len: int, hidden: int, n_heads: int, compute_elem: int) -> int:
    return seq_len * hidden * compute_elem * 16 + seq_len * n_heads * seq_len * compute_elem


def qwen_vram_peak_extra_bytes(
    *,
    seq_len: int,
    vram_k: int,
    hidden: int,
    intermediate: int,
    n_heads: int,
    compute_elem: int,
    baseline_seq: int = QWEN_VRAM_MEASURED_BASELINE_SEQ,
    baseline_topk: int = QWEN_VRAM_MEASURED_BASELINE_TOPK,
) -> int:
    """Ajuste do pico medido (top-8, ctx 32) para outro K e comprimento de contexto."""

    expert = _expert_bytes(hidden, intermediate, compute_elem)
    topk_delta = 2 * (vram_k - baseline_topk) * expert
    act_delta = _activation_vram_bytes(seq_len, hidden, n_heads, compute_elem) - _activation_vram_bytes(
        baseline_seq, hidden, n_heads, compute_elem
    )
    return topk_delta + act_delta


def predict_qwen_vram_peak(
    requested_cache: int,
    cache_bytes: int,
    *,
    seq_len: int,
    vram_k: int,
    hidden: int,
    intermediate: int,
    n_heads: int,
    compute_elem: int,
) -> int:
    """Pico VRAM esperado no modo qwen, calibrado na 1060 com cache de 3 GiB e top-8."""

    return (
        QWEN_VRAM_MEASURED_PEAK
        - (requested_cache - cache_bytes)
        + qwen_vram_peak_extra_bytes(
            seq_len=seq_len,
            vram_k=vram_k,
            hidden=hidden,
            intermediate=intermediate,
            n_heads=n_heads,
            compute_elem=compute_elem,
        )
    )


def fit_qwen_vram_cache(
    formula_bytes: int,
    requested_cache: int,
    budget: int,
    slack: int,
    *,
    seq_len: int,
    vram_k: int,
    hidden: int,
    intermediate: int,
    n_heads: int,
    compute_elem: int,
    vram_cache_elem: int = 2,
) -> int:
    """Encolhe o cache de experts pelo pior caso entre a fórmula e o pico medido."""

    if min(seq_len, vram_k, hidden, intermediate, n_heads, compute_elem, vram_cache_elem) < 1:
        raise ValueError("fit qwen com dimensão < 1")
    margin = QWEN_VRAM_MIN_MARGIN_BYTES
    expert_slot = 3 * hidden * intermediate * vram_cache_elem
    peak_kw = {
        "seq_len": seq_len,
        "vram_k": vram_k,
        "hidden": hidden,
        "intermediate": intermediate,
        "n_heads": n_heads,
        "compute_elem": compute_elem,
    }

    def fits(cache: int) -> bool:
        if cache < 0:
            return False
        shrunk_formula = formula_bytes - (requested_cache - cache)
        if shrunk_formula + slack > budget:
            return False
        return predict_qwen_vram_peak(requested_cache, cache, **peak_kw) <= budget - margin

    cache = fit_vram_cache(formula_bytes, requested_cache, budget, slack)
    while cache > 0 and not fits(cache):
        cache -= expert_slot
    return max(0, cache)


def fit_vram_cache(formula_bytes: int, requested_cache: int, budget: int, slack: int) -> int:
    """Maior cache que não passa de ``requested`` e deixa a fórmula mais a folga no orçamento.

    ``formula_bytes`` já inclui ``requested_cache``. O restante da conta não muda
    quando o cache encolhe. Folga zero devolve o pedido se a fórmula já cabe.
    """

    if requested_cache < 0 or budget < 1 or slack < 0 or formula_bytes < requested_cache:
        raise ValueError("cache, orçamento ou fórmula inválidos")
    room = budget - slack - (formula_bytes - requested_cache)
    if room <= 0:
        return 0
    return min(requested_cache, room)


def assert_vram_fits(vram_bytes: int, budget: int) -> None:
    if budget < 1:
        raise ValueError("orçamento de VRAM precisa ser >= 1 byte")
    if vram_bytes > budget:
        raise ValueError(
            f"VRAM estimada {vram_bytes} bytes passa de {budget}. "
            "Não carrego o modelo inteiro na GPU."
        )


class VramExpertCache:
    """Experts já na VRAM. O LRU não expulsa a camada que está computando.

    O backward anda para trás, mas o uso (forward, recompute, backward)
    toca o expert e o traz para o fim da fila. A camada corrente fica presa.
    """

    def __init__(
        self,
        max_bytes: int,
        dtype: torch.dtype = torch.float32,
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        if max_bytes < 0:
            raise ValueError("cache de VRAM precisa ser >= 0")
        self.max_bytes = max_bytes
        self.dtype = dtype
        self.compute_dtype = compute_dtype or dtype
        self.expert_nbytes = 0
        self._items: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._order: list[tuple[int, int]] = []
        self._sizes: dict[tuple[int, int], int] = {}
        self._pinned: set[tuple[int, int]] = set()
        self.stored_bytes = 0
        self.hits = 0
        self.misses = 0
        self.avoided_bytes = 0

    @property
    def enabled(self) -> bool:
        return self.max_bytes > 0

    def set_pinned(self, keys: set[tuple[int, int]]) -> None:
        self._pinned = set(keys)

    def contains(self, key: tuple[int, int]) -> bool:
        if not self.enabled:
            return False
        try:
            return key in self._items
        except RuntimeError:
            return False

    def reset_counters(self) -> None:
        self.hits = 0
        self.misses = 0
        self.avoided_bytes = 0

    def clear(self) -> None:
        self._items.clear()
        self._order.clear()
        self._sizes.clear()
        self._pinned.clear()
        self.stored_bytes = 0

    def take(self, key: tuple[int, int]):
        if not self.enabled:
            return None
        found = self._items.get(key)
        if found is None:
            return None
        self.hits += 1
        self.avoided_bytes += self._sizes[key]
        self._order.remove(key)
        self._order.append(key)
        return self._for_compute(found)

    def note_miss(self) -> None:
        if self.enabled:
            self.misses += 1

    def put(self, key: tuple[int, int], tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
        if not self.enabled or key in self._items:
            if key in self._items:
                self._order.remove(key)
                self._order.append(key)
            return
        if tensors[0].dtype != self.dtype:
            tensors = tuple(tensor.to(dtype=self.dtype) for tensor in tensors)
        nbytes = sum(int(tensor.nbytes) for tensor in tensors)
        if nbytes > self.max_bytes:
            return
        while self.stored_bytes + nbytes > self.max_bytes:
            if not self._evict():
                return
        self._items[key] = tensors
        self._sizes[key] = nbytes
        self._order.append(key)
        self.stored_bytes += nbytes

    def stats(self) -> dict[str, int | str]:
        capacity = self.max_bytes // self.expert_nbytes if self.expert_nbytes else 0
        return {
            "max_bytes": self.max_bytes,
            "hits": self.hits,
            "misses": self.misses,
            "avoided_bytes": self.avoided_bytes,
            "bytes": self.stored_bytes,
            "dtype": "bfloat16" if self.dtype == torch.bfloat16 else "float32",
            "expert_nbytes": self.expert_nbytes,
            "capacity_experts": capacity,
        }

    def _for_compute(self, tensors):
        if all(tensor.dtype == self.compute_dtype for tensor in tensors):
            return tensors
        return tuple(tensor.to(dtype=self.compute_dtype) for tensor in tensors)

    def _evict(self) -> bool:
        for key in self._order:
            if key in self._pinned:
                continue
            self._order.remove(key)
            del self._items[key]
            self.stored_bytes -= self._sizes.pop(key)
            return True
        return False


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
        self.copy_stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        self._ready: _Pack | None = None
        self._ready_index = -1
        self._pending: _Pack | None = None
        self._pending_index = -1
        self._pending_event: torch.cuda.Event | None = None
        # Nomes distintos dos métodos de cópia. Uma lista `_h2d` sombreava `_h2d()`.
        self._h2d_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._d2h_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._compute_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self.h2d_bytes = 0
        self.d2h_bytes = 0
        self.resident_h2d_bytes = 0
        self._stored_for: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.vram_cache = VramExpertCache(0, compute_dtype=self.dtype)

    def close(self) -> None:
        self.reset()
        self.vram_cache.clear()

    def reset(self) -> None:
        if self._pending_event is not None:
            self._pending_event.synchronize()
        self._ready = None
        self._ready_index = -1
        self._pending = None
        self._pending_index = -1
        self._pending_event = None
        self._h2d_events.clear()
        self._d2h_events.clear()
        self._compute_events.clear()
        self.h2d_bytes = 0
        self.d2h_bytes = 0
        self._stored_for.clear()
        self.vram_cache.reset_counters()

    def set_vram_cache(self, max_bytes: int, dtype: torch.dtype | None = None) -> None:
        storage = self.dtype if dtype is None else dtype
        cache = VramExpertCache(max_bytes, dtype=storage, compute_dtype=self.dtype)
        layout = self.model.store.layout
        elem = torch.empty((), dtype=storage).element_size()
        cache.expert_nbytes = 3 * layout.hidden * layout.intermediate * elem
        self.vram_cache = cache

    def take_cached_expert(self, layer: int, expert: int):
        return self.vram_cache.take((layer, expert))

    def remember_expert(self, layer: int, expert: int, tensors, *, miss: bool) -> None:
        if miss:
            self.vram_cache.note_miss()
        key = (layer, expert)
        stored = self._stored_for.pop(key, None)
        self.vram_cache.put(key, stored if stored is not None else tensors)

    def ensure(self, layer_index: int, layer, blobs: dict[tuple[int, int], tuple[bytes, bytes, bytes]]) -> _Pack:
        if self._ready is not None and self._ready_index == layer_index:
            return self._ready
        if self._pending is not None and self._pending_index == layer_index:
            self._wait_pending()
            return self._ready
        if self._pending_event is not None:
            # A direção mudou. A cópia em voo precisa terminar antes de soltar o destino.
            self._sync_copy(self._pending_event)
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
        gate, up, down = self._matrices(blobs, block=True, key=(layer, expert))
        if self._ready is not None:
            self._ready.experts[(layer, expert)] = (gate, up, down)
        return gate, up, down

    def h2d_from_block(self, layer: int, expert: int, block: torch.Tensor):
        from moe_memory.offload.blocks import split_block

        views = split_block(block, self.model.store.layout, self.model.store_dtype)
        compute, stored = self._place_views(views, block=True)
        if stored[0].data_ptr() == views[0].data_ptr():
            stored = tuple(tensor.clone() for tensor in stored)
            if compute[0].data_ptr() == views[0].data_ptr():
                compute = stored
        key = (layer, expert)
        if stored[0].dtype != self.dtype:
            self._stored_for[key] = stored
        if self._ready is not None:
            self._ready.experts[key] = compute
        return compute

    def note_compute(self, start: torch.cuda.Event, end: torch.cuda.Event) -> None:
        self._compute_events.append((start, end))

    def note_d2h(self, start: torch.cuda.Event, end: torch.cuda.Event, nbytes: int) -> None:
        self._d2h_events.append((start, end))
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
                "h2d_bytes": self.h2d_bytes + self.resident_h2d_bytes,
                "d2h_bytes": self.d2h_bytes,
            }
        if self.device.type != "cuda":
            return {
                "h2d_s": 0.0,
                "d2h_s": 0.0,
                "h2d_wait_s": float(self.model._times.get("h2d_wait_s", 0.0)),
                "h2d_wait_backward_s": float(self.model._times.get("h2d_wait_backward_s", 0.0)),
                "gpu_compute_s": 0.0,
                "h2d_bytes": self.h2d_bytes + self.resident_h2d_bytes,
                "d2h_bytes": self.d2h_bytes,
            }
        torch.cuda.synchronize(self.device)
        return {
            "h2d_s": _elapsed(self._h2d_events),
            "d2h_s": _elapsed(self._d2h_events),
            "h2d_wait_s": float(self.model._times.get("h2d_wait_s", 0.0)),
            "h2d_wait_backward_s": float(self.model._times.get("h2d_wait_backward_s", 0.0)),
            "gpu_compute_s": _elapsed(self._compute_events),
            "h2d_bytes": self.h2d_bytes + self.resident_h2d_bytes,
            "d2h_bytes": self.d2h_bytes,
        }

    def check_budget(self) -> None:
        if self.device.type != "cuda":
            return
        peak = int(torch.cuda.max_memory_allocated(self.device))
        if peak > self.model.vram_budget:
            raise RuntimeError(
                f"VRAM alocada {peak} bytes passou do orçamento {self.model.vram_budget}. "
                "Não carrego o modelo inteiro na GPU."
            )

    def _wait_pending(self) -> None:
        event = self._pending_event
        if event is not None:
            self._sync_copy(event)
        if self._pending is not None:
            self._pending.pinned.clear()
        self._ready = self._pending
        self._ready_index = self._pending_index
        self._pending = None
        self._pending_event = None

    def _load(self, layer_index: int, layer, blobs: dict, *, block: bool):
        pinned: list[torch.Tensor] = []
        experts: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        start, end = self._open_span()
        with self._copy_scope():
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
                experts[key] = self._matrices(raw, pinned, key=key)
        self._close_span(start, end, self._h2d_events)
        event = self._dependency_event()
        pack = _Pack(layer=view, experts=experts, pinned=pinned)
        if block:
            self._sync_copy(event)
            pack.pinned.clear()
        return pack, event

    def _h2d(self, weight: torch.Tensor | None, pinned: list[torch.Tensor]) -> torch.Tensor | None:
        if weight is None:
            return None
        host = weight.detach()
        if host.device.type != "cpu":
            host = host.to("cpu")
        if self.device.type == "cuda" and not host.is_pinned():
            host = host.pin_memory()
        pinned.append(host)
        gpu = host.to(device=self.device, dtype=self.dtype, non_blocking=self.device.type == "cuda")
        self.h2d_bytes += int(host.nbytes)
        return gpu

    def _matrices(
        self,
        blobs: tuple[bytes, bytes, bytes],
        pinned: list[torch.Tensor] | None = None,
        block: bool = False,
        key: tuple[int, int] | None = None,
    ):
        layout = self.model.store.layout
        shapes = (
            (layout.intermediate, layout.hidden),
            (layout.intermediate, layout.hidden),
            (layout.hidden, layout.intermediate),
        )
        hosts = []
        hold = pinned if pinned is not None else []
        for raw, shape in zip(blobs, shapes, strict=True):
            host = _host_matrix(raw, self.model.store_dtype, shape)
            hosts.append(host)
        compute, stored = self._place_views(hosts, block=block, hold=hold)
        if key is not None and stored[0].dtype != self.dtype:
            self._stored_for[key] = stored
        return compute

    def _storage_dtype(self) -> torch.dtype:
        if self.vram_cache.enabled:
            return self.vram_cache.dtype
        return self.dtype

    def _place_views(self, hosts, *, block: bool, hold: list[torch.Tensor] | None = None):
        bucket = hold if hold is not None else []
        compute_out = []
        stored_out = []
        storage = self._storage_dtype()
        start, end = self._open_span() if block else (None, None)
        with self._copy_scope():
            for host in hosts:
                if self.device.type == "cuda" and not host.is_pinned():
                    host = host.pin_memory()
                bucket.append(host)
                stored = host.to(device=self.device, dtype=storage, non_blocking=self.device.type == "cuda")
                self.h2d_bytes += int(host.nbytes)
                compute = stored if stored.dtype == self.dtype else stored.to(dtype=self.dtype)
                compute_out.append(compute)
                stored_out.append(stored)
        if block:
            self._close_span(start, end, self._h2d_events)
            self._sync_copy(self._dependency_event())
            if hold is None:
                bucket.clear()
        return tuple(compute_out), tuple(stored_out)

    def _copy_scope(self):
        if self.copy_stream is None:
            return nullcontext()
        return torch.cuda.stream(self.copy_stream)

    def _open_span(self):
        if self.copy_stream is None or not self.model.time_log:
            return None, None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(self.copy_stream)
        return start, end

    def _close_span(self, start, end, bucket) -> None:
        if start is None or end is None or self.copy_stream is None:
            return
        end.record(self.copy_stream)
        bucket.append((start, end))

    def _dependency_event(self):
        if self.copy_stream is None:
            return None
        event = torch.cuda.Event()
        event.record(self.copy_stream)
        return event

    def _sync_copy(self, event) -> None:
        """Espera esta cópia antes de soltar o pinned. Não é um synchronize do device inteiro."""

        if event is None:
            return
        if self.model.time_log:
            started = time.perf_counter()
            event.synchronize()
            self._note_wait(time.perf_counter() - started)
            return
        event.synchronize()

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
