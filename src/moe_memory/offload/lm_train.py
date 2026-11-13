"""Um passo de loss de linguagem no Qwen com o store, sem o pool de inferência.

O runtime de decode solta a vaga e dá ``detach`` no logit. Aqui o expert
é lido de novo para um tensor privado, a LoRA entra em volta do peso
congelado e o checkpoint de ativação relê o bloco no backward. O store
e o residente não são parâmetros. Treinam o LSTM e a LoRA.

A loss é a entropia cruzada do próximo token. Não há KL, balanceamento
nem penalidade de troca. Batch 1. O compute é float32: a GTX 1060 não
executa bf16.

O backward da fatia não volta pelo ``SelectBackward``: o gradiente é
somado direto em ``.grad[camada, expert]`` depois do passo. Isso evita
materializar um zero do tensor compartilhado inteiro a cada expert.

``--expert-cache N`` guarda o bf16 recém-lido do store, antes do
float32. Não entra no grafo. Zero é o caminho de hoje.
``--expert-prefetch-layers D`` lê, numa thread, as próximas camadas
enquanto a CPU computa a corrente. Zero desliga.
``--device-compute cuda`` calcula uma camada na GPU. O padrão é a CPU.
``--lm-head-vram``, ``--prefetch-vram-aware`` e ``--vram-cache-dtype bf16``
também nascem desligados. Sem eles o passo é o de hoje.
``--sparse-adamw`` também nasce desligado. O AdamW denso continua o padrão.
"""

from __future__ import annotations

import contextvars
import ctypes
import gc
import hashlib
import json
import math
import resource
import shutil
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.checkpoint import checkpoint

# Só o passo pedido liga isto. Fora dele o rótulo é um contexto vazio.
_PROFILE_LABELS = False


class _EmptyRegion:
    def __enter__(self):
        return None

    def __exit__(self, *exc: object) -> bool:
        return False


_EMPTY_REGION = _EmptyRegion()


def _region(name: str):
    if not _PROFILE_LABELS:
        return _EMPTY_REGION
    return record_function(name)

from moe_memory.lstm_ahead_router import TwoHorizonLstmRouter
from moe_memory.offload.qwen import (
    QwenSpec,
    QwenWeights,
    _apply_rope,
    _repeat_kv,
    rms_norm,
)
from moe_memory.offload.store import ExpertStore, torch_dtype

# Folga de lançamento, não o limite do cgroup. O limite real sai de memory.max.
LAUNCH_CEILING_BYTES = 8 * 1024**3
VRAM_BUDGET_BYTES = int(5.5 * 1024**3)
OVERHEAD_BYTES = 512 * 1024**2
LM_HEAD_CHUNK = 4096
# O DONTNEED da PR32 deixou 1,0–1,6 GiB de página. É conselho, não garantia.
FILE_CACHE_RESIDUAL_BYTES = int(1.6 * 1024**3)


def pascal_bf16_cuda_ok(capability: tuple[int, int] | None, bf16_supported: bool) -> bool:
    """sm_61 não entra. bf16 de CUDA começa em Ampere, capability 8."""

    if capability is None or capability[0] < 8:
        return False
    return bool(bf16_supported)


def refuse_cuda_bf16(device: torch.device, resident_dtype: torch.dtype) -> None:
    if device.type != "cuda":
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA não está disponível neste processo. O smoke fica na CPU.")
    capability = torch.cuda.get_device_capability(device)
    supported = bool(torch.cuda.is_bf16_supported())
    if resident_dtype == torch.bfloat16 and not pascal_bf16_cuda_ok(capability, supported):
        raise RuntimeError(
            f"GPU capability {capability[0]}.{capability[1]} não executa bf16 "
            f"(suporte reportado: {supported}). O residente está em bf16. "
            "O smoke usa CPU e float32. Não movo o residente inteiro para a GPU."
        )


def read_cgroup_memory_max(paths: list[Path] | None = None) -> int | None:
    """Limite de memória do cgroup deste processo. ``None`` se for ilimitado ou ilegível."""

    for path in _cgroup_memory_paths() if paths is None else paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text or text == "max":
            return None
        try:
            value = int(text)
        except ValueError:
            continue
        if value <= 0 or value >= 1 << 62:
            return None
        return value
    return None


def _cgroup_memory_paths() -> list[Path]:
    found: list[Path] = []
    proc = Path("/proc/self/cgroup")
    if proc.is_file():
        for line in proc.read_text(encoding="utf-8").splitlines():
            parts = line.split(":", 2)
            if len(parts) == 3 and parts[0] == "0" and parts[1] == "":
                found.append(Path("/sys/fs/cgroup") / parts[2].lstrip("/") / "memory.max")
    found.append(Path("/sys/fs/cgroup/memory.max"))
    found.append(Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"))
    return found


def rss_bytes() -> int | None:
    status = Path("/proc/self/status")
    if not status.is_file():
        return None
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return None


def ru_maxrss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _expert_block_bf16(hidden: int, intermediate: int) -> int:
    return 3 * hidden * intermediate * 2


def _expert_prefetch_bytes(hidden: int, intermediate: int, layers: int, k2: int, max_bytes: int) -> int:
    """Teto do buffer de prefetch. Não é a cópia float32 nem o LRU."""

    if layers < 0 or k2 < 0 or max_bytes < 0:
        raise ValueError("prefetch precisa ser >= 0")
    if layers == 0:
        return 0
    auto = layers * max(k2, 1) * _expert_block_bf16(hidden, intermediate)
    if max_bytes == 0:
        return auto
    return min(auto, max_bytes)


def _expert_cache_bytes(hidden: int, intermediate: int, slots: int, max_bytes: int) -> int:
    """Bytes do LRU. O bloco é bf16, três matrizes, sem a cópia float32."""

    if slots < 0 or max_bytes < 0:
        raise ValueError("cache de experts precisa ser >= 0")
    block = _expert_block_bf16(hidden, intermediate)
    if slots == 0 and max_bytes == 0:
        return 0
    if max_bytes == 0:
        return slots * block
    if slots == 0:
        return max_bytes
    return min(slots * block, max_bytes)


def launch_allowed(estimate: dict[str, int | None]) -> bool:
    """O anônimo, já com o cache, precisa caber em 8 GiB.

    A conta do cgroup ainda soma 1,6 GiB de página residual. Com o cache
    de 64 experts essa soma passa de 9 GiB por cerca de 100 MiB, enquanto
    o pico medido de 7,63 GiB mais o cache continua abaixo. Se a conta sem
    o cache já cabia e o anônimo com o cache continua no teto, o passo
    começa. Sem cache, o cgroup estimado continua obrigatório.
    """

    total = int(estimate["total_bytes"] or 0)
    if total > LAUNCH_CEILING_BYTES:
        return False
    cgroup = estimate["cgroup_bytes"]
    cgroup_total = int(estimate["cgroup_total_bytes"] or 0)
    cache = int(estimate.get("expert_cache_bytes") or 0)
    if cgroup is None or cgroup_total <= cgroup:
        return True
    return cache > 0 and cgroup_total - cache <= int(cgroup) and total <= int(cgroup)


def estimate_bytes(
    *,
    resident_bytes: int,
    vocab: int,
    hidden: int,
    intermediate: int,
    n_layers: int,
    n_experts: int,
    rank: int,
    seq_len: int,
    k1: int,
    lstm_embed: int,
    lstm_hidden: int,
    n_heads: int | None = None,
    n_kv_heads: int | None = None,
    head_dim: int | None = None,
    cgroup_bytes: int | None = None,
    read_cgroup: bool = True,
    expert_cache_slots: int = 0,
    expert_cache_max_bytes: int = 0,
    expert_prefetch_layers: int = 0,
    expert_prefetch_bytes: int = 0,
    k2: int = 6,
    compute_cuda: bool = False,
    compute_elem: int = 4,
    vram_expert_cache_bytes: int = 0,
    lm_head_vram: bool = False,
    vram_cache_elem: int = 4,
    sparse_adam_steps: int = 0,
    trainable_lora: bool = True,
    trainable_lstm: bool = True,
    vram_k: int | None = None,
) -> dict[str, int | None]:
    """Teto do anon com a cabeça e o expert em pedaços. Não inclui o store inteiro.

    ``resident_page_cache_if_kept`` é o cache do arquivo se ninguém pedir
    ``DONTNEED``. Não entra na soma. ``cgroup_bytes`` vem do processo, ou
    do teste. ``expert_cache_*`` é o LRU bf16. ``expert_prefetch_*`` é o
    buffer da thread, também bf16, e pode estar cheio junto com o LRU.
    """

    if min(resident_bytes, vocab, hidden, intermediate, n_layers, n_experts, rank, seq_len, k1) < 1:
        raise ValueError("estimativa com dimensão < 1")
    if vram_cache_elem < 1:
        raise ValueError("o elemento do cache de VRAM precisa ser >= 1")
    if sparse_adam_steps < 0:
        raise ValueError("os passos do Adam esparso precisam ser >= 0")
    lora = n_layers * n_experts * 3 * rank * (hidden + intermediate) * 4 if trainable_lora else 0
    lstm_params = 0 if not trainable_lstm else (
        vocab * lstm_embed
        + 4 * lstm_hidden * lstm_embed
        + 4 * lstm_hidden * lstm_hidden
        + 8 * lstm_hidden
        + n_layers * (lstm_hidden * n_experts + n_experts)
    )
    one_expert = 3 * hidden * intermediate * 4
    expert_cache = _expert_cache_bytes(hidden, intermediate, expert_cache_slots, expert_cache_max_bytes)
    expert_prefetch = _expert_prefetch_bytes(
        hidden, intermediate, expert_prefetch_layers, k2, expert_prefetch_bytes
    )
    if n_heads is None or n_kv_heads is None or head_dim is None:
        attention = 0
    else:
        q = n_heads * head_dim * hidden * 4
        kv = n_kv_heads * head_dim * hidden * 4
        attention = q + 2 * kv + hidden * n_heads * head_dim * 4
    head = min(vocab, LM_HEAD_CHUNK) * hidden * 4
    if cgroup_bytes is None and read_cgroup:
        cgroup_bytes = read_cgroup_memory_max()
    per_expert = 3 * rank * (hidden + intermediate)
    dense_adam = lora * 2
    dense_grad = lora
    if sparse_adam_steps:
        cap_experts = min(n_layers * n_experts, sparse_adam_steps * n_layers * max(k2, 1))
        dense_adam = cap_experts * per_expert * 8
        dense_grad = n_layers * max(k2, 1) * per_expert * 4
    pieces: dict[str, int | None] = {
        "resident_bytes": resident_bytes,
        "lora_fp32_bytes": lora,
        "lora_grad_bytes": dense_grad,
        "lora_select_backward_bytes": 0,
        "adam_lora_bytes": dense_adam,
        "adam_lora_bytes_if_dense": lora * 2,
        "lstm_with_adam_bytes": lstm_params * 12,
        "lstm_grad_bytes": lstm_params * 4,
        "expert_fp32_bytes": one_expert,
        "expert_cache_bytes": expert_cache,
        "expert_prefetch_bytes": expert_prefetch,
        "attention_fp32_bytes": attention,
        "lm_head_fp32_bytes": head,
        "overhead_bytes": OVERHEAD_BYTES,
        "resident_page_cache_if_kept": resident_bytes,
        "file_cache_bytes": FILE_CACHE_RESIDUAL_BYTES,
        "seq_len": seq_len,
        "k1": k1,
    }
    largest = max(
        n_layers * n_experts * rank * hidden * 4,
        n_layers * n_experts * intermediate * rank * 4,
        n_layers * n_experts * hidden * rank * 4,
    )
    pieces["activation_bytes"] = seq_len * hidden * 4 * (n_layers + 8)
    pieces["optimizer_step_temporary_bytes"] = largest
    pieces["checkpoint_extra_bytes"] = 0
    pieces["checkpoint_file_bytes"] = lora + lora * 2 + lstm_params * 12
    skipped = {
        "resident_page_cache_if_kept",
        "seq_len",
        "k1",
        "checkpoint_file_bytes",
        "file_cache_bytes",
        "adam_lora_bytes_if_dense",
    }
    pinned = 0
    vram = {
        "vram_attention_bytes": 0,
        "vram_experts_bytes": 0,
        "vram_lora_bytes": 0,
        "vram_activation_bytes": 0,
        "vram_overhead_bytes": 0,
        "vram_layers_on_device": 0,
        "vram_bytes": 0,
        "vram_expert_cache_bytes": 0,
        "vram_lm_head_bytes": 0,
    }
    if compute_cuda:
        if n_heads is None or n_kv_heads is None or head_dim is None:
            raise ValueError("a estimativa da GPU precisa das dimensões da atenção")
        from moe_memory.offload.lm_gpu import estimate_pinned_bytes, estimate_vram_bytes

        live_k = max(k2 if vram_k is None else vram_k, 1)
        pinned = estimate_pinned_bytes(
            hidden=hidden,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            intermediate=intermediate,
            k2=live_k,
        )
        vram = estimate_vram_bytes(
            hidden=hidden,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            intermediate=intermediate,
            seq_len=seq_len,
            k2=live_k,
            rank=rank,
            compute_elem=compute_elem,
            vram_expert_cache_bytes=max(0, vram_expert_cache_bytes),
            ce_bytes=(vocab * hidden * 2) if lm_head_vram else min(vocab, LM_HEAD_CHUNK) * hidden * compute_elem,
        )
    pieces["pinned_bytes"] = pinned
    pieces.update(vram)
    released = vocab * hidden * 2 if compute_cuda and lm_head_vram else 0
    pieces["lm_head_cpu_released_bytes"] = released
    expert_stored = 3 * hidden * intermediate * vram_cache_elem
    pieces["vram_cache_capacity_experts"] = (
        max(0, vram_expert_cache_bytes) // expert_stored if compute_cuda and vram_expert_cache_bytes > 0 else 0
    )
    skipped.update({
        "vram_attention_bytes",
        "vram_experts_bytes",
        "vram_expert_cache_bytes",
        "vram_lm_head_bytes",
        "vram_lora_bytes",
        "vram_activation_bytes",
        "vram_overhead_bytes",
        "vram_layers_on_device",
        "vram_bytes",
        "lm_head_cpu_released_bytes",
        "vram_cache_capacity_experts",
    })
    pieces["total_bytes"] = (
        sum(value for key, value in pieces.items() if key not in skipped and value is not None) - released
    )
    pieces["cgroup_total_bytes"] = pieces["total_bytes"] + pieces["file_cache_bytes"]
    pieces["launch_ceiling_bytes"] = LAUNCH_CEILING_BYTES
    pieces["cgroup_bytes"] = cgroup_bytes
    pieces["vram_budget_bytes"] = VRAM_BUDGET_BYTES
    return pieces


def fingerprint(path: str | Path, prefix: int = 4096) -> dict[str, int | str]:
    file = Path(path)
    size = file.stat().st_size
    with file.open("rb") as handle:
        blob = handle.read(prefix)
    return {
        "bytes": size,
        "prefix_bytes": len(blob),
        "prefix_sha256": hashlib.sha256(blob).hexdigest(),
    }


class ExpertLoRA(nn.Module):
    """Delta nas três projeções. ``B`` nasce zero. O peso base não mora aqui."""

    def __init__(self, n_layers: int, n_experts: int, hidden: int, intermediate: int, rank: int, alpha: float) -> None:
        super().__init__()
        if rank < 1 or alpha <= 0:
            raise ValueError("rank >= 1 e alpha > 0")
        self.scale = alpha / rank
        self.gate_A = nn.Parameter(torch.empty(n_layers, n_experts, rank, hidden))
        self.gate_B = nn.Parameter(torch.zeros(n_layers, n_experts, intermediate, rank))
        self.up_A = nn.Parameter(torch.empty(n_layers, n_experts, rank, hidden))
        self.up_B = nn.Parameter(torch.zeros(n_layers, n_experts, intermediate, rank))
        self.down_A = nn.Parameter(torch.empty(n_layers, n_experts, rank, intermediate))
        self.down_B = nn.Parameter(torch.zeros(n_layers, n_experts, hidden, rank))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, parameter in self.named_parameters():
            if name.endswith("_B"):
                nn.init.zeros_(parameter)
                continue
            matrix = parameter.view(-1, parameter.shape[-2], parameter.shape[-1])
            for row in matrix:
                nn.init.kaiming_uniform_(row, a=5**0.5)


class OffloadLanguageModel(nn.Module):
    """Forward de uma janela. O store fica aberto e só é lido."""

    def __init__(
        self,
        weights: QwenWeights,
        router: TwoHorizonLstmRouter,
        store: ExpertStore,
        *,
        rank: int = 4,
        alpha: float = 8.0,
        use_lora: bool = True,
    ) -> None:
        super().__init__()
        spec = weights.spec
        if router.n_layers != spec.n_layers or router.n_experts != spec.n_experts:
            raise ValueError("o LSTM não casa com o store")
        if router.vocab_size != spec.vocab_size:
            raise ValueError("o vocabulário do LSTM não é o do residente")
        if store.header.n_layers != spec.n_layers or store.header.n_experts != spec.n_experts:
            raise ValueError("o cabeçalho do store não casa com o config")
        self.weights = weights
        self.store = store
        self.router = router
        self.router_kind = getattr(router, "kind", "lstm")
        self.use_lora = use_lora
        if use_lora:
            self.lora = ExpertLoRA(spec.n_layers, spec.n_experts, spec.hidden_size, spec.moe_intermediate, rank, alpha)
            self._lora_scale = self.lora.scale
        else:
            self.lora = None
            self._lora_scale = 0.0
            self._lora_zero = _zero_lora_bank(spec.hidden_size, spec.moe_intermediate)
        self.store_dtype = torch_dtype(store.header.dtype_code)
        self.memory_log = False
        self.time_log = False
        self.cache_sim_sizes: tuple[int, ...] = ()
        self.expert_cache = ExpertWeightCache(0, 0)
        self.prefetch = ExpertPrefetcher(self)
        self.store_reads = 0
        self.compute_device = torch.device("cpu")
        self.compute_dtype = torch.float32
        self.vram_budget = 0
        self.vram_expert_cache_bytes = 0
        self.vram_cache_dtype = torch.float32
        self.vram_cache_dtype_name = "fp32"
        self.vram_cache_sim_sizes: tuple[int, ...] = ()
        self.lm_head_vram = False
        self.lm_head_h2d_bytes = 0
        self.lm_head_cpu_released = False
        self._lm_head_weight: torch.Tensor | None = None
        self.prefetch_vram_aware = False
        self.gpu = None
        self._expert_plan: list[list[tuple[int, int]]] = []
        self.expert_live = 0
        self.expert_live_max = 0
        self._io_phase = "forward"
        self._times: dict[str, float] = {}
        self._read_seq: list[tuple[tuple[int, int], int]] = []
        for tensor in _resident_tensors(weights):
            tensor.requires_grad_(False)

    def forward_window(self, token_ids: torch.Tensor):
        if token_ids.ndim != 2 or token_ids.shape[0] != 1:
            raise ValueError("a janela de offload é [1, T]")
        if token_ids.shape[1] < 2:
            raise ValueError("a janela precisa de ao menos 2 tokens")
        _EXPERT_STATE["model"] = self
        qwen = self.router_kind == "qwen"
        self._qwen_routes = []
        if qwen:
            plan = None
            self._expert_plan = []
        else:
            started = time.perf_counter()
            with _region("router_plan"):
                plan, _state = self.router.plan(token_ids)
            self._add_time("router_plan_s", time.perf_counter() - started)
            if self.prefetch.enabled or self.gpu is not None:
                self._expert_plan = experts_by_layer(plan)
                if self.prefetch.enabled:
                    self.prefetch.set_plan(self._expert_plan)
        hidden = self._embed(token_ids)
        self._phase("embed")
        spec = self.weights.spec
        seq = token_ids.shape[1]
        for layer_index in range(spec.n_layers):
            self._phase(f"layer-{layer_index}")
            if qwen:
                coeffs = torch.zeros(1, seq, 1)
                index = torch.zeros(1, seq, 1, dtype=torch.long)
                valid = torch.zeros(1, seq, 1, dtype=torch.bool)
            else:
                coeffs = plan.soft_weights[:, :, layer_index, :]
                index = plan.hot_index[:, :, layer_index, :]
                valid = plan.hot_valid[:, :, layer_index, :]
            hidden = checkpoint(
                self._layer,
                hidden,
                coeffs,
                index,
                valid,
                torch.tensor(layer_index, device=hidden.device),
                use_reentrant=False,
            )
        self._phase(f"layers-done-{spec.n_layers}")
        started = time.perf_counter()
        with _region("cross_entropy"):
            normed = rms_norm(hidden, _as(self.weights.final_norm, hidden), spec.rms_eps)
            ce_hidden = normed[:, :-1].reshape(-1, normed.shape[-1])
            if self.gpu is not None:
                started_h2d = time.perf_counter()
                ce_hidden = ce_hidden.to(device=self.gpu.device, dtype=self.gpu.dtype)
                self._add_time("h2d_activation_s", time.perf_counter() - started_h2d)
            weight = self._lm_head_weight if self.lm_head_vram and self._lm_head_weight is not None else self.weights.lm_head
            loss = _chunked_cross_entropy(
                ce_hidden,
                weight,
                token_ids[:, 1:].reshape(-1),
                reuse_stats=self.lm_head_vram,
                h2d_counter=self._lm_head_h2d_counter(),
            )
        self._add_time("cross_entropy_forward_s", time.perf_counter() - started)
        self._phase("loss")
        if qwen:
            plan = _stack_qwen_routes(self._qwen_routes)
        return loss, plan

    def _embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        started = time.perf_counter()
        rows = self.weights.embed.index_select(0, token_ids.reshape(-1))
        hidden = rows.to(dtype=torch.float32).view(1, token_ids.shape[1], -1)
        self._add_time("embed_s", time.perf_counter() - started)
        return hidden

    def _layer(self, hidden, coeffs, index, valid, layer_index_t):
        spec = self.weights.spec
        layer_index = int(layer_index_t.item())
        layer = self.weights.layers[layer_index]
        if self._io_phase == "backward":
            with _region(f"checkpoint_recompute_layer_{layer_index}"):
                return self._layer_body(hidden, coeffs, index, valid, spec, layer, layer_index)
        return self._layer_body(hidden, coeffs, index, valid, spec, layer, layer_index)

    def set_vram_expert_cache(self, max_bytes: int, dtype: str = "fp32") -> None:
        if max_bytes < 0:
            raise ValueError("cache de VRAM precisa ser >= 0")
        if dtype not in ("fp32", "bf16"):
            raise ValueError("vram-cache-dtype é fp32 ou bf16")
        self.vram_expert_cache_bytes = int(max_bytes)
        self.vram_cache_dtype_name = dtype
        self.vram_cache_dtype = torch.bfloat16 if dtype == "bf16" else torch.float32
        if self.gpu is not None:
            self.gpu.set_vram_cache(self.vram_expert_cache_bytes, self.vram_cache_dtype)

    def set_prefetch_vram_aware(self, enabled: bool) -> None:
        self.prefetch_vram_aware = bool(enabled)

    def enable_lm_head_vram(self) -> None:
        """Sobe a cabeça uma vez. O backward reaproveita max e log Z do forward."""

        head = self.weights.lm_head.detach()
        device = self.gpu.device if self.gpu is not None else torch.device("cpu")
        stored = head.to(device=device)
        stored.requires_grad_(False)
        self._lm_head_weight = stored
        self.lm_head_vram = True
        nbytes = int(head.nbytes) if device.type == "cuda" else 0
        self.lm_head_h2d_bytes = nbytes
        if self.gpu is not None and nbytes:
            self.gpu.resident_h2d_bytes += nbytes
        self.lm_head_cpu_released = release_cpu_lm_head(self.weights, stored, device.type)

    def _lm_head_h2d_counter(self):
        gpu = self.gpu
        if gpu is None:
            return None

        def add(nbytes: int) -> None:
            gpu.h2d_bytes += int(nbytes)

        return add

    def set_compute_device(self, device: torch.device, vram_budget: int) -> None:
        """``cpu`` é o caminho de hoje. ``cuda`` calcula uma camada por vez."""

        if vram_budget < 1:
            raise ValueError("orçamento de VRAM precisa ser >= 1 byte")
        if device.type == "cpu":
            self.compute_device = torch.device("cpu")
            self.compute_dtype = torch.float32
            self.vram_budget = 0
            if self.gpu is not None:
                self.gpu.close()
            self.gpu = None
            return
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA não está disponível. O compute misto não sobe.")
        from moe_memory.offload.lm_gpu import LayerTransport, compute_dtype_for

        self.compute_device = torch.device(device)
        self.compute_dtype = compute_dtype_for(self.compute_device)
        self.vram_budget = int(vram_budget)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.gpu = LayerTransport(self)
        self.gpu.set_vram_cache(self.vram_expert_cache_bytes, self.vram_cache_dtype)
        torch.cuda.reset_peak_memory_stats(self.compute_device)

    def _layer_body(self, hidden, coeffs, index, valid, spec, layer, layer_index):
        direction = 1 if self._io_phase == "forward" else -1
        self.prefetch.arm(layer_index, direction)
        if self.gpu is not None:
            return self._layer_body_gpu(hidden, coeffs, index, valid, spec, layer, layer_index, direction)
        started = time.perf_counter()
        normed = rms_norm(hidden, _as(layer.input_norm, hidden), spec.rms_eps)
        self._add_time(self._phase_key("resident_s"), time.perf_counter() - started)
        with _region(f"attention_layer_{layer_index}"):
            hidden = hidden + _attention_seq(layer, normed, spec)
        started = time.perf_counter()
        moe_in = rms_norm(hidden, _as(layer.post_norm, hidden), spec.rms_eps)
        self._add_time(self._phase_key("resident_s"), time.perf_counter() - started)
        if self.router_kind == "qwen":
            index, valid, coeffs = self._route_qwen(moe_in, layer, layer_index)
        hidden = hidden + self._moe(moe_in, layer_index, index, valid, coeffs)
        return hidden

    def _layer_body_gpu(self, hidden, coeffs, index, valid, spec, layer, layer_index, direction):
        gpu = self.gpu
        if self._expert_plan and 0 <= layer_index < len(self._expert_plan):
            gpu.vram_cache.set_pinned(set(self._expert_plan[layer_index]))
        blobs = self._cached_expert_blobs(layer_index)
        started = time.perf_counter()
        pack = gpu.ensure(layer_index, layer, blobs)
        self._add_h2d_time("h2d_prepare_s", time.perf_counter() - started)
        started = time.perf_counter()
        hidden_g = hidden.to(device=gpu.device, dtype=gpu.dtype)
        coeffs_g = coeffs.to(device=gpu.device, dtype=gpu.dtype)
        self._add_h2d_time("h2d_activation_s", time.perf_counter() - started)
        compute_start = compute_end = None
        if self.time_log and gpu.device.type == "cuda":
            compute_start = torch.cuda.Event(enable_timing=True)
            compute_start.record()
        normed = rms_norm(hidden_g, _as(pack.layer.input_norm, hidden_g), spec.rms_eps)
        with _region(f"attention_layer_{layer_index}"):
            hidden_g = hidden_g + _attention_seq(pack.layer, normed, spec)
        moe_in = rms_norm(hidden_g, _as(pack.layer.post_norm, hidden_g), spec.rms_eps)
        if self.router_kind == "qwen":
            index, valid, coeffs_g = self._route_qwen(moe_in, layer, layer_index)
        hidden_g = hidden_g + self._moe(moe_in, layer_index, index, valid, coeffs_g)
        if compute_start is not None:
            compute_end = torch.cuda.Event(enable_timing=True)
            compute_end.record()
            gpu.note_compute(compute_start, compute_end)
        nxt = layer_index + direction
        if self._expert_plan and 0 <= nxt < len(self._expert_plan):
            gpu.stage_ahead(nxt, self.weights.layers[nxt], self._cached_expert_blobs(nxt))
        nbytes = hidden_g.numel() * hidden_g.element_size()
        d2h_start = d2h_end = None
        if self.time_log and gpu.device.type == "cuda":
            d2h_start = torch.cuda.Event(enable_timing=True)
            d2h_start.record()
        started = time.perf_counter()
        out = hidden_g.to(device="cpu", dtype=torch.float32)
        self._add_h2d_time("d2h_wait_s", time.perf_counter() - started)
        if d2h_start is not None:
            d2h_end = torch.cuda.Event(enable_timing=True)
            d2h_end.record()
            gpu.note_d2h(d2h_start, d2h_end, nbytes)
        gpu.check_budget()
        return out

    def _cached_expert_blobs(self, layer_index: int) -> dict[tuple[int, int], tuple[bytes, bytes, bytes]]:
        found: dict[tuple[int, int], tuple[bytes, bytes, bytes]] = {}
        if not self._expert_plan or not 0 <= layer_index < len(self._expert_plan):
            return found
        for key in self._expert_plan[layer_index]:
            blob = self.expert_cache.peek(key)
            if blob is not None:
                found[key] = blob
        return found

    def _add_h2d_time(self, name: str, value: float) -> None:
        if self._io_phase == "forward":
            self._add_time(name, value)
        else:
            stem, suffix = name.rsplit("_s", 1)
            self._add_time(f"{stem}_backward_s{suffix}", value)

    def _count_ram_hit(self, key: tuple[int, int]) -> None:
        if self.expert_cache.enabled:
            self.expert_cache.touch(key)

    def _route_qwen(self, moe_in: torch.Tensor, layer, layer_index: int):
        from moe_memory.offload.qwen_gate import route_qwen

        if layer.gate is None:
            raise RuntimeError("o modo qwen precisa do gate residente")
        index, valid, coeffs = route_qwen(moe_in, layer.gate, self.router.k2)
        keys = {(layer_index, int(expert)) for expert in index.detach().reshape(-1).tolist()}
        self.expert_cache.set_pinned(keys)
        if self.gpu is not None:
            self.gpu.vram_cache.set_pinned(keys)
        if self._io_phase == "forward":
            self._qwen_routes.append((index.detach().cpu(), valid.detach().cpu(), coeffs.detach().to(device="cpu", dtype=torch.float32)))
        return index, valid, coeffs

    def _moe(self, hidden, layer_index, index, valid, coeffs):
        flat = hidden[0]
        out = flat.new_zeros(flat.shape)
        groups: dict[int, list[tuple[int, int]]] = {}
        started = time.perf_counter()
        seq, slots = index.shape[1], index.shape[2]
        for token in range(seq):
            for slot in range(slots):
                if not bool(valid[0, token, slot].item()):
                    continue
                expert = int(index[0, token, slot].item())
                groups.setdefault(expert, []).append((token, slot))
        self._add_time(self._phase_key("moe_group_s"), time.perf_counter() - started)
        for expert, members in groups.items():
            rows = torch.tensor([token for token, _slot in members], device=flat.device)
            current = flat.index_select(0, rows)
            coeff = torch.stack([coeffs[0, token, slot] for token, slot in members])
            mixed = _apply_expert(self, current, coeff, layer_index, expert)
            out = out.index_add(0, rows, mixed)
        return out.unsqueeze(0)

    def _phase(self, name: str) -> None:
        if not self.memory_log:
            return
        print(
            json.dumps({"phase": name, "rss_bytes": rss_bytes(), "ru_maxrss_bytes": ru_maxrss_bytes()}),
            flush=True,
        )

    def _note_expert(self, delta: int) -> None:
        self.expert_live += delta
        if self.expert_live > self.expert_live_max:
            self.expert_live_max = self.expert_live

    def set_expert_cache(self, slots: int, max_bytes: int = 0) -> None:
        self.expert_cache = ExpertWeightCache(slots, max_bytes)

    def set_expert_prefetch(self, layers: int, max_bytes: int = 0) -> None:
        block = int(self.store.header.raw_nbytes)
        cap = max_bytes
        if layers > 0 and cap == 0:
            cap = layers * max(self.router.k2, 1) * block
        self.prefetch.start(layers, cap, block, block_nbytes=int(self.store.header.block_nbytes))

    def release_expert_cache(self) -> None:
        """Solta o LRU e junta a thread antes do checkpoint."""

        self.prefetch.stop()
        self.expert_cache.clear()
        if self.gpu is not None:
            self.gpu.close()

    def _read(self, layer: int, expert: int, kind: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (layer, expert)
        if self.gpu is not None:
            cached_gpu = self.gpu.take_cached_expert(layer, expert)
            if cached_gpu is not None:
                self._count_ram_hit(key)
                if self.time_log:
                    self._read_seq.append((key, int(self.store.header.raw_nbytes)))
                return cached_gpu
            ready = self.gpu.take_expert(layer, expert)
            if ready is not None:
                self._count_ram_hit(key)
                self.gpu.remember_expert(layer, expert, ready, miss=False)
                if self.time_log:
                    self._read_seq.append((key, int(self.store.header.raw_nbytes)))
                return ready
        if self.prefetch.enabled:
            blobs = None
            try:
                blobs, sync_s = self.prefetch.obtain(layer, expert, kind)
                if isinstance(blobs, torch.Tensor):
                    nbytes = int(self.store.header.raw_nbytes)
                    if sync_s is not None:
                        self._add_time(f"expert_read_{kind}_s", sync_s)
                        self._add_time(f"expert_read_{kind}_bytes", nbytes)
                    if self.time_log:
                        self._read_seq.append((key, nbytes))
                    if self.gpu is not None:
                        tensors = self.gpu.h2d_from_block(layer, expert, blobs)
                        self.gpu.remember_expert(layer, expert, tensors, miss=True)
                        return tensors
                    from moe_memory.offload.blocks import split_block

                    gate, up, down = split_block(blobs, self.store.layout, self.store_dtype)
                    return (
                        gate.to(dtype=torch.float32).clone(),
                        up.to(dtype=torch.float32).clone(),
                        down.to(dtype=torch.float32).clone(),
                    )
                nbytes = len(blobs[0]) + len(blobs[1]) + len(blobs[2])
                if sync_s is not None:
                    self._add_time(f"expert_read_{kind}_s", sync_s)
                    self._add_time(f"expert_read_{kind}_bytes", nbytes)
                gate_b, up_b, down_b = blobs
            finally:
                if isinstance(blobs, torch.Tensor):
                    self.prefetch.release_block(blobs)
        else:
            cached = self.expert_cache.lookup(key)
            if cached is None:
                started = time.perf_counter()
                with _region(f"expert_read_{kind}"):
                    gate_b, up_b, down_b = self.store.payload(layer, expert)
                self.store_reads += 1
                self._add_time(f"expert_read_{kind}_s", time.perf_counter() - started)
                nbytes = len(gate_b) + len(up_b) + len(down_b)
                self._add_time(f"expert_read_{kind}_bytes", nbytes)
                self.expert_cache.store(key, (gate_b, up_b, down_b), nbytes)
            else:
                gate_b, up_b, down_b = cached
                nbytes = len(gate_b) + len(up_b) + len(down_b)
        if self.time_log:
            self._read_seq.append((key, nbytes))
        if self.gpu is not None:
            tensors = self.gpu.h2d_now(layer, expert, (gate_b, up_b, down_b))
            self.gpu.remember_expert(layer, expert, tensors, miss=True)
            return tensors
        started = time.perf_counter()
        with _region(f"bf16_fp32_{kind}"):
            layout = self.store.layout
            gate = _matrix(gate_b, self.store_dtype, (layout.intermediate, layout.hidden))
            up = _matrix(up_b, self.store_dtype, (layout.intermediate, layout.hidden))
            down = _matrix(down_b, self.store_dtype, (layout.hidden, layout.intermediate))
        self._add_time(f"expert_convert_{kind}_s", time.perf_counter() - started)
        return gate, up, down

    def _add_time(self, key: str, value: float) -> None:
        if not self.time_log:
            return
        self._times[key] = self._times.get(key, 0.0) + value

    def _phase_key(self, name: str) -> str:
        if self._io_phase == "forward":
            return name
        stem, suffix = name.rsplit("_s", 1)
        return f"{stem}_recompute_s{suffix}"

    def begin_step_timing(self) -> None:
        self._times = {}
        self._read_seq = []
        self._io_phase = "forward"
        self.expert_cache.reset_counters()
        self.prefetch.reset_counters()
        if self.gpu is not None:
            self.gpu.reset()
            torch.cuda.reset_peak_memory_stats(self.compute_device)

    def timing_report(self, wall_s: float) -> dict[str, float | int | dict | None]:
        exclusive = (
            "router_plan_s",
            "embed_s",
            "resident_s",
            "attention_s",
            "expert_read_forward_s",
            "prefetch_wait_s",
            "h2d_wait_s",
            "expert_convert_forward_s",
            "expert_compute_forward_s",
            "cross_entropy_forward_s",
            "h2d_prepare_s",
            "h2d_activation_s",
            "d2h_wait_s",
            "moe_group_s",
            "backward_s",
            "optimizer_s",
            "release_s",
        )
        accounted = sum(float(self._times.get(key, 0.0)) for key in exclusive)
        report: dict[str, float | int | dict | None] = {key: float(self._times.get(key, 0.0)) for key in exclusive}
        for kind in ("forward", "recompute", "backward"):
            seconds = float(self._times.get(f"expert_read_{kind}_s", 0.0))
            nbytes = int(self._times.get(f"expert_read_{kind}_bytes", 0.0))
            report[f"expert_read_{kind}_s"] = seconds
            report[f"expert_read_{kind}_bytes"] = nbytes
            report[f"expert_read_{kind}_mb_s"] = (nbytes / 1e6 / seconds) if seconds else None
            report[f"expert_convert_{kind}_s"] = float(self._times.get(f"expert_convert_{kind}_s", 0.0))
            report[f"expert_compute_{kind}_s"] = float(self._times.get(f"expert_compute_{kind}_s", 0.0))
        report["attention_recompute_s"] = float(self._times.get("attention_recompute_s", 0.0))
        report["resident_recompute_s"] = float(self._times.get("resident_recompute_s", 0.0))
        report["cross_entropy_backward_s"] = float(self._times.get("cross_entropy_backward_s", 0.0))
        report["h2d_prepare_backward_s"] = float(self._times.get("h2d_prepare_backward_s", 0.0))
        report["h2d_activation_backward_s"] = float(self._times.get("h2d_activation_backward_s", 0.0))
        report["d2h_wait_backward_s"] = float(self._times.get("d2h_wait_backward_s", 0.0))
        report["moe_group_recompute_s"] = float(self._times.get("moe_group_recompute_s", 0.0))
        report["accounted_s"] = accounted
        report["outros_s"] = wall_s - accounted
        report["wall_s"] = wall_s
        report["torch_threads"] = torch.get_num_threads()
        keys = [item[0] for item in self._read_seq]
        report["experts_leituras"] = len(keys)
        report["experts_unicos"] = len(set(keys))
        report["experts_releituras"] = len(keys) - len(set(keys))
        report["cache_lru"] = {
            str(size): simulate_expert_lru(self._read_seq, size) for size in self.cache_sim_sizes
        }
        report["vram_cache_sim"] = {
            str(size): simulate_expert_lru(self._read_seq, size) for size in self.vram_cache_sim_sizes
        }
        report["lm_head_vram"] = self.lm_head_vram
        report["lm_head_h2d_bytes"] = self.lm_head_h2d_bytes
        report["vram_cache_dtype"] = self.vram_cache_dtype_name
        report["expert_cache"] = self.expert_cache.stats()
        report["expert_prefetch"] = self.prefetch.stats()
        report["vram_expert_cache"] = self.gpu.vram_cache.stats() if self.gpu is not None else {
            "max_bytes": 0,
            "hits": 0,
            "misses": 0,
            "avoided_bytes": 0,
            "bytes": 0,
        }
        report["store_reads"] = self.store_reads
        gpu_times = {
            "h2d_s": 0.0,
            "d2h_s": 0.0,
            "h2d_wait_s": float(self._times.get("h2d_wait_s", 0.0)),
            "h2d_wait_backward_s": float(self._times.get("h2d_wait_backward_s", 0.0)),
            "gpu_compute_s": 0.0,
            "h2d_bytes": 0,
            "d2h_bytes": 0,
        }
        if self.gpu is not None:
            gpu_times = self.gpu.resolve()
        report.update(gpu_times)
        report["optimizer_cpu_s"] = float(self._times.get("optimizer_s", 0.0))
        report["compute_device"] = self.compute_device.type
        report["compute_dtype"] = str(self.compute_dtype).removeprefix("torch.")
        return report


def lm_step(model: OffloadLanguageModel, optimizer: torch.optim.Optimizer, token_ids: torch.Tensor) -> dict[str, float | int | bool]:
    model.router.train()
    if model.lora is not None:
        model.lora.train()
    optimizer.zero_grad(set_to_none=True)
    model.begin_step_timing()
    wall = time.perf_counter()
    model._phase("forward")
    loss, plan = model.forward_window(token_ids)
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError("loss de linguagem não é finita")
    model._phase("backward")
    model._io_phase = "backward"
    started = time.perf_counter()
    try:
        loss.backward()
        if not _uses_sparse_adam(optimizer):
            flush_lora_slice_grads()
    except Exception:
        _SLICE_GRADS.clear()
        raise
    model._add_time("backward_s", time.perf_counter() - started)
    model._io_phase = "forward"
    model._phase("optimizer")
    report = grad_report(model)
    if not report["grads_finite"]:
        raise RuntimeError("há gradiente não finito")
    started = time.perf_counter()
    with _region("optimizer_step"):
        optimizer.step()
    model._add_time("optimizer_s", time.perf_counter() - started)
    report.update(optimizer_account(optimizer))
    report["loss"] = float(loss.detach())
    report["distinct_experts"] = distinct_experts(plan)
    report["hot_entropy"] = _hot_entropy(plan)
    report["router_entropy"] = None if model.router_kind == "qwen" else _router_entropy(model, token_ids)
    del loss, plan
    started = time.perf_counter()
    release_step_memory(optimizer)
    model._add_time("release_s", time.perf_counter() - started)
    report["rss_bytes"] = rss_bytes()
    report["ru_maxrss_bytes"] = ru_maxrss_bytes()
    report["store_reads"] = model.store_reads
    report["lm_head_vram"] = model.lm_head_vram
    report["lm_head_h2d_bytes"] = model.lm_head_h2d_bytes
    report["lm_head_cpu_released"] = model.lm_head_cpu_released
    if model.gpu is not None:
        report["h2d_bytes"] = model.gpu.h2d_bytes + model.gpu.resident_h2d_bytes
    report["expert_cache"] = model.expert_cache.stats()
    report["expert_prefetch"] = model.prefetch.stats()
    report["vram_expert_cache"] = model.gpu.vram_cache.stats() if model.gpu is not None else {
        "max_bytes": 0,
        "hits": 0,
        "misses": 0,
        "avoided_bytes": 0,
        "bytes": 0,
    }
    if model.time_log:
        report["time"] = model.timing_report(time.perf_counter() - wall)
    model._phase("after-step")
    return report


def scheduled_lr(
    step: int,
    total: int,
    base: float,
    *,
    warmup: int = 0,
    schedule: str = "constant",
    lr_min: float = 0.0,
) -> float:
    """Constante, ou warmup linear e cosseno até ``lr_min``. O passo 1 é o primeiro update."""

    if step < 1 or total < 1:
        raise ValueError("o schedule pede passo e total >= 1")
    if schedule not in {"constant", "cosine"}:
        raise ValueError("schedule é constant ou cosine")
    if warmup < 0 or base < 0 or lr_min < 0:
        raise ValueError("warmup, lr e piso precisam ser >= 0")
    if warmup and step <= warmup:
        return base * step / warmup
    if schedule == "constant":
        return base
    span = max(total - warmup, 1)
    progress = min(max((step - warmup) / span, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_min + (base - lr_min) * cosine


def set_learning_rate(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def evaluate_windows(model: OffloadLanguageModel, windows: list[list[int]]) -> dict[str, float | int]:
    """Loss média por token, sem gradiente. A perplexidade é exp dessa média."""

    if not windows:
        raise ValueError("avaliação sem janela")
    training = model.training
    model.eval()
    loss_sum = 0.0
    tokens = 0
    try:
        with torch.no_grad():
            for window in windows:
                ids = torch.tensor(window, dtype=torch.long).view(1, -1)
                loss, _plan = model.forward_window(ids)
                count = int(ids.shape[1] - 1)
                loss_sum += float(loss.detach()) * count
                tokens += count
    finally:
        model.train(training)
    mean = loss_sum / tokens
    return {
        "eval_loss": mean,
        "eval_perplexity": math.exp(mean),
        "eval_tokens": tokens,
        "eval_batches": len(windows),
    }


def replace_checkpoint(path: str | Path, **kwargs) -> None:
    """Grava o checkpoint treinável e fica só com o último diretório."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    save_checkpoint(temporary, **kwargs)
    if target.exists():
        shutil.rmtree(target)
    temporary.rename(target)


def _zero_lora_bank(hidden: int, intermediate: int) -> dict[str, torch.Tensor]:
    return {
        "gate_a": torch.zeros(1, hidden),
        "gate_b": torch.zeros(intermediate, 1),
        "up_a": torch.zeros(1, hidden),
        "up_b": torch.zeros(intermediate, 1),
        "down_a": torch.zeros(1, intermediate),
        "down_b": torch.zeros(hidden, 1),
    }


def _stack_qwen_routes(routes: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    if not routes:
        raise RuntimeError("o gate do Qwen não rodou")
    index = torch.stack([item[0] for item in routes], dim=2)
    valid = torch.stack([item[1] for item in routes], dim=2)
    coeffs = torch.stack([item[2] for item in routes], dim=2)

    class _Plan:
        hot_index = index
        hot_valid = valid
        soft_weights = coeffs

    return _Plan()


def make_optimizer(model: OffloadLanguageModel, lr: float, *, sparse: bool = False) -> torch.optim.Optimizer:
    """AdamW sem ``foreach``. O caminho agrupado monta temporários do tamanho da LoRA inteira.

    ``sparse=True`` guarda momento só da fatia tocada. O padrão continua o AdamW denso.
    """

    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not sparse:
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.0, foreach=False)
    from moe_memory.offload.sparse_adamw import SparseAdamW

    return SparseAdamW(params, lr=lr, weight_decay=0.0)


def malloc_trim() -> None:
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        return


def release_step_memory(optimizer: torch.optim.Optimizer) -> None:
    """Solta o gradiente e devolve arenas ao glibc antes do checkpoint."""

    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    malloc_trim()


def _uses_sparse_adam(optimizer: torch.optim.Optimizer) -> bool:
    from moe_memory.offload.sparse_adamw import SparseAdamW

    return isinstance(optimizer, SparseAdamW)


def optimizer_account(optimizer: torch.optim.Optimizer) -> dict[str, int]:
    """Tensores e parâmetros que o passo realmente atualizou, e o estado vivo."""

    if _uses_sparse_adam(optimizer):
        stats = optimizer.last_stats
        return {
            "optimizer_updated_tensors": int(stats["updated_tensors"]),
            "optimizer_updated_params": int(stats["updated_params"]),
            "optimizer_state_bytes": int(stats["state_bytes"]),
            "optimizer_moment_slices": int(stats["moment_slices"]),
        }
    tensors = 0
    params = 0
    state_bytes = 0
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                tensors += 1
                params += param.numel()
            state = optimizer.state.get(param, {})
            for key in ("exp_avg", "exp_avg_sq"):
                value = state.get(key)
                if torch.is_tensor(value):
                    state_bytes += value.numel() * value.element_size()
    return {
        "optimizer_updated_tensors": tensors,
        "optimizer_updated_params": params,
        "optimizer_state_bytes": state_bytes,
        "optimizer_moment_slices": tensors,
    }


def grad_report(model: OffloadLanguageModel) -> dict[str, float | bool]:
    lstm_module = getattr(model.router, "lstm", None)
    lstm = None if lstm_module is None else lstm_module.weight_ih_l0.grad
    lora_b = []
    finite = True
    b_ids = {id(model.lora.gate_B), id(model.lora.up_B), id(model.lora.down_B)}
    for _param, _layer, _expert, grad in _SLICE_GRADS:
        if not bool(torch.isfinite(grad).all().item()):
            finite = False
        if id(_param) in b_ids:
            lora_b.append(grad.detach())
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all().item()):
            finite = False
        if name.startswith("lora.") and name.endswith("_B"):
            lora_b.append(parameter.grad.detach())
    lora_norm = torch.zeros(())
    if lora_b:
        lora_norm = torch.stack([item.float().pow(2).sum() for item in lora_b]).sum().sqrt()
    lstm_norm = 0.0 if lstm is None else float(lstm.detach().float().norm())
    return {
        "grads_finite": finite and (not hasattr(model.router, "lstm") or (lstm is not None and bool(torch.isfinite(lstm).all().item()))),
        "lstm_grad_norm": lstm_norm,
        "lora_b_grad_norm": float(lora_norm),
        "lstm_grad_nonzero": lstm is not None and bool(lstm.detach().abs().sum() > 0),
        "lora_b_grad_nonzero": bool(lora_norm > 0),
    }


def distinct_experts(plan) -> int:
    valid = plan.hot_valid[0]
    index = plan.hot_index[0]
    tokens, layers, slots = torch.nonzero(valid, as_tuple=True)
    experts = index[tokens, layers, slots]
    return len({(int(layer), int(expert)) for layer, expert in zip(layers.tolist(), experts.tolist(), strict=True)})


def trainable_state(model: OffloadLanguageModel) -> dict:
    return {"router": model.router.state_dict(), "lora": model.lora.state_dict()}


def load_trainable(model: OffloadLanguageModel, payload: dict) -> None:
    model.router.load_state_dict(payload["router"])
    model.lora.load_state_dict(payload["lora"])


def persist_checkpoint(
    enabled: bool,
    path: str | Path,
    *,
    step: int,
    model: OffloadLanguageModel,
    optimizer: torch.optim.Optimizer,
    extra: dict,
) -> bool:
    """False não grava e não percorre parâmetros nem o estado do Adam."""

    if not enabled:
        return False
    save_checkpoint(path, step=step, model=model, optimizer=optimizer, extra=extra)
    return True


def save_checkpoint(path: str | Path, *, step: int, model: OffloadLanguageModel, optimizer: torch.optim.Optimizer, extra: dict) -> None:
    """Grava um tensor por arquivo. Não monta o estado inteiro numa cópia só."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    if _uses_sparse_adam(optimizer):
        optimizer.catch_up_all()
    entries: list[dict] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        rel = f"weights/{len(entries):04d}.bin"
        _write_tensor(directory / rel, parameter)
        entries.append(
            {
                "kind": "param",
                "name": name,
                "dtype": _dtype_name(parameter.dtype),
                "shape": list(parameter.shape),
                "file": rel,
            }
        )
        state = optimizer.state.get(parameter, {})
        for key, value in state.items():
            if not torch.is_tensor(value):
                entries.append({"kind": "optim-scalar", "name": name, "key": key, "value": _json_scalar(value)})
                continue
            rel = f"optim/{len(entries):04d}-{key}.bin"
            _write_tensor(directory / rel, value)
            entries.append(
                {
                    "kind": "optim",
                    "name": name,
                    "key": key,
                    "dtype": _dtype_name(value.dtype),
                    "shape": list(value.shape),
                    "file": rel,
                }
            )
        if _uses_sparse_adam(optimizer) and parameter.ndim >= 3:
            for (layer, expert), (exp_avg, exp_avg_sq) in optimizer.slice_entries(parameter):
                for key, value in (("exp_avg", exp_avg), ("exp_avg_sq", exp_avg_sq)):
                    rel = f"optim/{len(entries):04d}-{key}.bin"
                    _write_tensor(directory / rel, value)
                    entries.append(
                        {
                            "kind": "optim-slice",
                            "name": name,
                            "layer": layer,
                            "expert": expert,
                            "key": key,
                            "dtype": _dtype_name(value.dtype),
                            "shape": list(value.shape),
                            "file": rel,
                        }
                    )
    meta = {
        "version": 2,
        "kind": "qwen-lm-lora-lstm",
        "step": step,
        "entries": entries,
        "param_groups": [_jsonable_group(group) for group in optimizer.param_groups],
        **extra,
    }
    (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def _write_tensor(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = tensor.detach()
    if not raw.is_contiguous():
        raw = raw.contiguous()
    flat = raw.reshape(-1).view(torch.uint8)
    with path.open("wb") as handle:
        flat.numpy().tofile(handle)


def _read_tensor_into(path: Path, dest: torch.Tensor, entry: dict) -> None:
    if list(dest.shape) != list(entry["shape"]) or dest.dtype != _dtype_of(entry["dtype"]):
        raise ValueError(f"tensor {entry.get('name')} não casa com {entry['shape']} {entry['dtype']}")
    nbytes = dest.numel() * dest.element_size()
    blob = path.read_bytes()
    if len(blob) != nbytes:
        raise ValueError(f"{path} tem {len(blob)} bytes, esperado {nbytes}")
    dest.reshape(-1).view(torch.uint8).copy_(torch.frombuffer(bytearray(blob), dtype=torch.uint8))


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _dtype_of(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"dtype {name} não é do torch")
    return dtype


def _json_scalar(value):
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    raise ValueError(f"estado do otimizador não é tensor nem escalar: {type(value)}")


def _jsonable_group(group: dict) -> dict:
    saved = {}
    for key, value in group.items():
        if key == "params":
            continue
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, list) and all(isinstance(item, (int, float)) for item in value):
            saved[key] = value
        elif isinstance(value, (int, float, str, bool)) or value is None:
            saved[key] = value
    return saved


def load_checkpoint(path: str | Path, model: OffloadLanguageModel, optimizer: torch.optim.Optimizer) -> dict:
    directory = Path(path)
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    if meta.get("version") != 2 or meta.get("kind") != "qwen-lm-lora-lstm":
        raise ValueError("checkpoint não é o formato em streaming da LM")
    named = dict(model.named_parameters())
    pending: dict = {}
    for group, saved in zip(optimizer.param_groups, meta["param_groups"], strict=True):
        for key, value in saved.items():
            if key in group and key != "params":
                group[key] = value
    for entry in meta["entries"]:
        parameter = named[entry["name"]]
        if entry["kind"] == "param":
            _read_tensor_into(directory / entry["file"], parameter, entry)
            continue
        if entry["kind"] == "optim-slice":
            if not _uses_sparse_adam(optimizer):
                raise ValueError("checkpoint esparso pede SparseAdamW")
            pending.setdefault(parameter, []).append(entry)
            continue
        slot = optimizer.state[parameter]
        if entry["kind"] == "optim-scalar":
            slot[entry["key"]] = entry["value"]
            continue
        if entry["kind"] != "optim":
            raise ValueError(f"entrada de checkpoint desconhecida: {entry['kind']}")
        tensor = torch.empty(entry["shape"], dtype=_dtype_of(entry["dtype"]))
        _read_tensor_into(directory / entry["file"], tensor, entry)
        slot[entry["key"]] = tensor
    if _uses_sparse_adam(optimizer):
        _load_sparse_slices(directory, optimizer, pending)
    return meta


def _load_sparse_slices(directory: Path, optimizer, pending: dict) -> None:
    for parameter, entries in pending.items():
        paired: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
        for entry in entries:
            key = (int(entry["layer"]), int(entry["expert"]))
            tensor = torch.empty(entry["shape"], dtype=_dtype_of(entry["dtype"]))
            _read_tensor_into(directory / entry["file"], tensor, entry)
            paired.setdefault(key, {})[entry["key"]] = tensor
        step = int(optimizer.state[parameter]["step"].item())
        for (layer, expert), moments in paired.items():
            optimizer.load_slice(
                parameter,
                layer,
                expert,
                exp_avg=moments["exp_avg"],
                exp_avg_sq=moments["exp_avg_sq"],
                through=step,
            )
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.ndim < 3:
                continue
            state = optimizer.state.get(parameter)
            if not state or "step" not in state:
                continue
            optimizer.mark_through(parameter, int(state["step"].item()))


def _hot_entropy(plan) -> float:
    weights = plan.soft_weights.detach().float().clamp_min(0)
    total = weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    probs = weights / total
    ent = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    valid = plan.hot_valid.detach().any(dim=-1)
    if not bool(valid.any().item()):
        return 0.0
    return float(ent[valid].mean())


def _router_entropy(model: OffloadLanguageModel, token_ids: torch.Tensor) -> float:
    with torch.no_grad():
        decision, _state = model.router.forward(token_ids)
        probs = torch.softmax(decision.raw_scores.float(), dim=-1)
        ent = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    return float(ent.mean())


def _resident_tensors(weights: QwenWeights):
    yield weights.embed
    yield weights.final_norm
    yield weights.lm_head
    for layer in weights.layers:
        for name in ("input_norm", "post_norm", "q", "k", "v", "o", "q_norm", "k_norm"):
            yield getattr(layer, name)
        for name in ("q_bias", "k_bias", "v_bias", "o_bias", "gate"):
            bias = getattr(layer, name)
            if bias is not None:
                yield bias


def _as(weight: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor | None:
    if weight is None:
        return None
    if weight.dtype == ref.dtype and weight.device == ref.device:
        return weight
    with _region("bf16_fp32"):
        return weight.to(dtype=ref.dtype, device=ref.device)


def _apply_expert(model: OffloadLanguageModel, hidden, coeff, layer: int, expert: int) -> torch.Tensor:
    _EXPERT_STATE["model"] = model
    _EXPERT_STATE["layer"] = layer
    _EXPERT_STATE["expert"] = expert
    from moe_memory.offload.sparse_adamw import catch_up_lora_slice

    if model.lora is None:
        zero = model._lora_zero
        gate_a, gate_b, up_a, up_b, down_a, down_b = (
            zero["gate_a"], zero["gate_b"], zero["up_a"], zero["up_b"], zero["down_a"], zero["down_b"]
        )
    else:
        lora = model.lora
        for param in (lora.gate_A, lora.gate_B, lora.up_A, lora.up_B, lora.down_A, lora.down_B):
            catch_up_lora_slice(param, layer, expert)
        gate_a, gate_b = lora.gate_A[layer, expert], lora.gate_B[layer, expert]
        up_a, up_b = lora.up_A[layer, expert], lora.up_B[layer, expert]
        down_a, down_b = lora.down_A[layer, expert], lora.down_B[layer, expert]
    return _ExpertFn.apply(hidden, coeff, gate_a, gate_b, up_a, up_b, down_a, down_b)


class _ExpertFn(torch.autograd.Function):
    """Um expert de cada vez. O peso base não fica no grafo."""

    @staticmethod
    def forward(ctx, hidden, coeff, gate_a, gate_b, up_a, up_b, down_a, down_b):
        model = _EXPERT_STATE["model"]
        layer = _EXPERT_STATE["layer"]
        expert = _EXPERT_STATE["expert"]
        ctx.model = model
        ctx.layer = layer
        ctx.expert = expert
        ctx.scale = model._lora_scale
        kind = "forward" if model._io_phase == "forward" else "recompute"
        gate, up, down = model._read(layer, expert, kind)
        model._note_expert(1)
        started = time.perf_counter()
        try:
            with _region(f"expert_compute_{kind}"):
                with torch.no_grad():
                    projected = _swiglu(
                        hidden,
                        gate,
                        up,
                        down,
                        _align(gate_a, hidden),
                        _align(gate_b, hidden),
                        _align(up_a, hidden),
                        _align(up_b, hidden),
                        _align(down_a, hidden),
                        _align(down_b, hidden),
                        ctx.scale,
                    )
                    out = projected * coeff.unsqueeze(-1)
        finally:
            model._add_time(f"expert_compute_{kind}_s", time.perf_counter() - started)
            model._note_expert(-1)
            del gate, up, down
        ctx.save_for_backward(hidden, coeff, gate_a, gate_b, up_a, up_b, down_a, down_b)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        hidden, coeff, gate_a, gate_b, up_a, up_b, down_a, down_b = ctx.saved_tensors
        model = ctx.model
        with torch.enable_grad():
            inputs = [_as_grad_input(hidden, hidden), _as_grad_input(coeff, hidden)]
            inputs.extend(_as_grad_input(tensor, hidden) for tensor in (gate_a, gate_b, up_a, up_b, down_a, down_b))
            gate, up, down = model._read(ctx.layer, ctx.expert, "backward")
            model._note_expert(1)
            started = time.perf_counter()
            try:
                with _region("expert_compute_backward"):
                    projected = _swiglu(inputs[0], gate, up, down, *inputs[2:], ctx.scale)
                    out = projected * inputs[1].unsqueeze(-1)
                    grads = torch.autograd.grad(out, inputs, grad_out)
            finally:
                model._add_time("expert_compute_backward_s", time.perf_counter() - started)
                model._note_expert(-1)
                del gate, up, down
        if _RETURN_SLICE_GRADS:
            fixed = list(grads)
            for index in range(2, len(fixed)):
                grad = fixed[index]
                if grad is not None and grad.device.type != "cpu":
                    fixed[index] = grad.detach().to("cpu")
            return tuple(fixed)
        lora = model.lora
        for param, grad in (
            (lora.gate_A, grads[2]),
            (lora.gate_B, grads[3]),
            (lora.up_A, grads[4]),
            (lora.up_B, grads[5]),
            (lora.down_A, grads[6]),
            (lora.down_B, grads[7]),
        ):
            _queue_slice_grad(param, ctx.layer, ctx.expert, grad)
        return grads[0], grads[1], None, None, None, None, None, None


_EXPERT_STATE: dict = {}
# Gradiente da fatia, aplicado depois do backward. Não volta pelo SelectBackward.
_SLICE_GRADS: list[tuple[torch.nn.Parameter, int, int, torch.Tensor]] = []
_RETURN_SLICE_GRADS = False


def _align(tensor: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if tensor.device == ref.device and tensor.dtype == ref.dtype:
        return tensor
    return tensor.detach().to(device=ref.device, dtype=ref.dtype)


def _as_grad_input(tensor: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    out = tensor.detach()
    if out.device != ref.device or out.dtype != ref.dtype:
        out = out.to(device=ref.device, dtype=ref.dtype)
    return out.requires_grad_(True)


def _queue_slice_grad(param: torch.nn.Parameter, layer: int, expert: int, grad: torch.Tensor | None) -> None:
    if grad is None:
        return
    grad = grad.detach()
    if grad.device.type != "cpu":
        grad = grad.to("cpu")
    _SLICE_GRADS.append((param, int(layer), int(expert), grad))


def flush_lora_slice_grads() -> None:
    """Soma cada fatia no ``.grad`` do tensor compartilhado, uma vez."""

    for param, layer, expert, grad in _SLICE_GRADS:
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        param.grad[layer, expert].add_(grad)
    _SLICE_GRADS.clear()


@contextmanager
def legacy_lora_slice_grads():
    """Caminho antigo, só para o teste de equivalência. O treino não usa."""

    global _RETURN_SLICE_GRADS
    previous = _RETURN_SLICE_GRADS
    _RETURN_SLICE_GRADS = True
    try:
        yield
    finally:
        _RETURN_SLICE_GRADS = previous
        _SLICE_GRADS.clear()


def _swiglu(hidden, gate, up, down, gate_a, gate_b, up_a, up_b, down_a, down_b, scale):
    with _region("lora"):
        gate_delta = _delta(hidden, gate_a, gate_b)
        up_delta = _delta(hidden, up_a, up_b)
    projected_gate = F.linear(hidden, gate) + scale * gate_delta
    projected_up = F.linear(hidden, up) + scale * up_delta
    mid = F.silu(projected_gate) * projected_up
    with _region("lora"):
        down_delta = _delta(mid, down_a, down_b)
    return F.linear(mid, down) + scale * down_delta


_CE_COUNTER: contextvars.ContextVar = contextvars.ContextVar("lm_head_h2d_counter", default=None)


def _chunked_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    reuse_stats: bool = False,
    h2d_counter=None,
) -> torch.Tensor:
    token = _CE_COUNTER.set(h2d_counter)
    try:
        return _ChunkedCrossEntropy.apply(hidden, weight, targets, 1 if reuse_stats else 0)
    finally:
        _CE_COUNTER.reset(token)


class _ChunkedCrossEntropy(torch.autograd.Function):
    """CE do próximo token sem materializar a cabeça inteira em float32.

    ``reuse_stats`` guarda o máximo e o log Z do forward e o backward faz
    uma passada. Os dois caminhos usam os mesmos kernels e a mesma ordem.
    """

    @staticmethod
    def forward(ctx, hidden, weight, targets, reuse_stats):
        ctx.chunk = LM_HEAD_CHUNK
        ctx.reuse_stats = bool(int(reuse_stats))
        ctx.h2d_counter = _CE_COUNTER.get()
        with torch.no_grad():
            if ctx.reuse_stats:
                loss, running_max, log_z = _ce_value(
                    hidden, weight, targets, LM_HEAD_CHUNK, h2d_counter=ctx.h2d_counter, return_stats=True
                )
                ctx.save_for_backward(hidden, weight, targets, running_max, log_z)
            else:
                loss = _ce_value(hidden, weight, targets, LM_HEAD_CHUNK, h2d_counter=ctx.h2d_counter)
                ctx.save_for_backward(hidden, weight, targets)
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        saved = ctx.saved_tensors
        hidden, weight, targets = saved[:3]
        started = time.perf_counter()
        with _region("cross_entropy_backward"):
            if ctx.reuse_stats:
                grad_hidden = _ce_grad_hidden(
                    hidden,
                    weight,
                    targets,
                    ctx.chunk,
                    grad_loss,
                    running_max=saved[3],
                    log_z=saved[4],
                    h2d_counter=ctx.h2d_counter,
                )
            else:
                grad_hidden = _ce_grad_hidden(
                    hidden, weight, targets, ctx.chunk, grad_loss, h2d_counter=ctx.h2d_counter
                )
        model = _EXPERT_STATE.get("model")
        if model is not None:
            model._add_time("cross_entropy_backward_s", time.perf_counter() - started)
        return grad_hidden, None, None, None


def _ce_value(hidden, weight, targets, chunk: int, *, h2d_counter=None, return_stats: bool = False):
    if targets.device != hidden.device:
        targets = targets.to(hidden.device)
    rows = hidden.shape[0]
    running_max = hidden.new_full((rows,), float("-inf"))
    vocab = weight.shape[0]
    for start in range(0, vocab, chunk):
        logits = F.linear(hidden, _ce_block(weight, start, chunk, hidden, h2d_counter))
        running_max = torch.maximum(running_max, logits.detach().max(dim=-1).values)
    sum_exp = hidden.new_zeros(rows)
    target_logit = hidden.new_zeros(rows)
    index = torch.arange(rows, device=hidden.device)
    for start in range(0, vocab, chunk):
        block = _ce_block(weight, start, chunk, hidden, h2d_counter)
        logits = F.linear(hidden, block)
        sum_exp = sum_exp + (logits - running_max.unsqueeze(-1)).exp().sum(dim=-1)
        chosen = (targets >= start) & (targets < start + block.shape[0])
        local = (targets - start).clamp(0, block.shape[0] - 1)
        picked = logits[index, local]
        target_logit = torch.where(chosen, picked, target_logit)
    if return_stats:
        log_z = running_max + sum_exp.log()
        loss = (log_z - target_logit).mean()
        return loss, running_max, log_z
    nll = running_max + sum_exp.log() - target_logit
    return nll.mean()


def _ce_grad_hidden(
    hidden,
    weight,
    targets,
    chunk: int,
    grad_loss,
    *,
    running_max=None,
    log_z=None,
    h2d_counter=None,
) -> torch.Tensor:
    hidden = hidden.detach()
    if targets.device != hidden.device:
        targets = targets.to(hidden.device)
    rows = hidden.shape[0]
    vocab = weight.shape[0]
    with torch.no_grad():
        if running_max is None or log_z is None:
            running_max = hidden.new_full((rows,), float("-inf"))
            for start in range(0, vocab, chunk):
                block = _ce_block(weight, start, chunk, hidden, h2d_counter)
                logits = F.linear(hidden, block)
                running_max = torch.maximum(running_max, logits.max(dim=-1).values)
                del block, logits
            sum_exp = hidden.new_zeros(rows)
            for start in range(0, vocab, chunk):
                block = _ce_block(weight, start, chunk, hidden, h2d_counter)
                logits = F.linear(hidden, block)
                sum_exp = sum_exp + (logits - running_max.unsqueeze(-1)).exp().sum(dim=-1)
                del block, logits
            log_z = running_max + sum_exp.log()
        grad = hidden.new_zeros(hidden.shape)
        scale = grad_loss.to(device=hidden.device, dtype=hidden.dtype) / rows
        for start in range(0, vocab, chunk):
            block = _ce_block(weight, start, chunk, hidden, h2d_counter)
            logits = F.linear(hidden, block)
            probs = (logits - log_z.unsqueeze(-1)).exp()
            chosen = (targets >= start) & (targets < start + block.shape[0])
            if bool(chosen.any().item()):
                local = targets[chosen] - start
                probs[chosen, local] = probs[chosen, local] - 1
            grad = grad + probs.matmul(block) * scale
            del block, logits, probs
    return grad


def _ce_block(
    weight: torch.Tensor,
    start: int,
    chunk: int,
    hidden: torch.Tensor,
    h2d_counter=None,
) -> torch.Tensor:
    block = weight[start : start + chunk]
    if block.device == hidden.device and block.dtype == hidden.dtype:
        return block
    if h2d_counter is not None and block.device != hidden.device:
        h2d_counter(int(block.numel() * block.element_size()))
    return block.to(device=hidden.device, dtype=hidden.dtype)


def release_cpu_lm_head(weights, stored: torch.Tensor, device_type: str) -> bool:
    """Solta a cabeça na CPU quando a cópia nova não divide o storage do embedding."""

    head = weights.lm_head
    tied = head.data_ptr() == weights.embed.data_ptr() and head.numel() == weights.embed.numel()
    if device_type != "cuda" or tied:
        return False
    weights.lm_head = stored
    return True


def _matrix(raw: bytes, dtype: torch.dtype, shape: tuple[int, int]) -> torch.Tensor:
    elem = torch.empty((), dtype=dtype).element_size()
    need = shape[0] * shape[1] * elem
    if len(raw) != need:
        raise ValueError(f"bloco do expert tem {len(raw)} bytes, esperado {need}")
    cpu = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape).clone()
    return cpu.to(dtype=torch.float32)


def _delta(hidden: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return F.linear(F.linear(hidden, left), right)


class ExpertWeightCache:
    """LRU dos blocos como o store devolve, antes do float32.

    A recomputação do checkpoint e o backward da mesma camada vêm em
    seguida. Com dezenas de vagas, esses experts acabam de ser lidos e
    ficam no fim da fila: o backward não relê o SSD. A ordem reversa
    das camadas continua a de um LRU comum, a mesma do ``--expert-cache-sim``.
    """

    def __init__(self, slots: int, max_bytes: int) -> None:
        if slots < 0 or max_bytes < 0:
            raise ValueError("cache de experts precisa ser >= 0")
        self.slots = slots
        self.max_bytes = max_bytes
        self._items: OrderedDict[tuple[int, int], tuple[bytes, bytes, bytes]] = OrderedDict()
        self._sizes: dict[tuple[int, int], int] = {}
        self._pinned: set[tuple[int, int]] = set()
        self.stored_bytes = 0
        self.hits = 0
        self.misses = 0
        self.avoided_bytes = 0

    @property
    def enabled(self) -> bool:
        return self.slots > 0 or self.max_bytes > 0

    def set_pinned(self, keys: set[tuple[int, int]]) -> None:
        """Experts da camada corrente. O LRU não os expulsa."""

        self._pinned = set(keys)

    def contains(self, key: tuple[int, int]) -> bool:
        return key in self._items

    def peek(self, key: tuple[int, int]) -> tuple[bytes, bytes, bytes] | None:
        return self._items.get(key)

    def touch(self, key: tuple[int, int]) -> tuple[bytes, bytes, bytes] | None:
        found = self._items.get(key)
        if found is None:
            return None
        self.hits += 1
        self.avoided_bytes += self._sizes[key]
        self._items.move_to_end(key)
        return found

    def note_miss(self) -> None:
        if self.enabled:
            self.misses += 1

    def lookup(self, key: tuple[int, int]) -> tuple[bytes, bytes, bytes] | None:
        if not self.enabled:
            return None
        found = self.touch(key)
        if found is None:
            self.misses += 1
        return found

    def store(self, key: tuple[int, int], blobs: tuple[bytes, bytes, bytes], nbytes: int) -> None:
        if not self.enabled or key in self._items:
            if key in self._items:
                self._items.move_to_end(key)
            return
        if self.max_bytes > 0 and nbytes > self.max_bytes:
            return
        while self._items and self._needs_room(nbytes):
            if not self._evict():
                break
        if self._needs_room(nbytes):
            return
        self._items[key] = blobs
        self._sizes[key] = nbytes
        self.stored_bytes += nbytes

    def stored(self) -> list[tuple[bytes, bytes, bytes]]:
        return list(self._items.values())

    def stats(self) -> dict[str, int]:
        return {
            "slots": self.slots,
            "max_bytes": self.max_bytes,
            "hits": self.hits,
            "misses": self.misses,
            "avoided_bytes": self.avoided_bytes,
            "bytes": self.stored_bytes,
        }

    def reset_counters(self) -> None:
        self.hits = 0
        self.misses = 0
        self.avoided_bytes = 0

    def clear(self) -> None:
        self._items.clear()
        self._sizes.clear()
        self._pinned.clear()
        self.stored_bytes = 0

    def _needs_room(self, incoming: int) -> bool:
        if self.slots > 0 and len(self._items) >= self.slots:
            return True
        if self.max_bytes > 0 and self.stored_bytes + incoming > self.max_bytes:
            return True
        return False

    def _evict(self) -> bool:
        victim = None
        for key in self._items:
            if key not in self._pinned:
                victim = key
                break
        if victim is None:
            return False
        del self._items[victim]
        self.stored_bytes -= self._sizes.pop(victim)
        return True


def experts_by_layer(plan) -> list[list[tuple[int, int]]]:
    """Experts usados em cada camada, na ordem em que o forward os encontra."""

    valid = plan.hot_valid[0]
    index = plan.hot_index[0]
    seq, n_layers, slots = valid.shape
    layers: list[list[tuple[int, int]]] = []
    for layer in range(n_layers):
        found: list[tuple[int, int]] = []
        seen: set[int] = set()
        for token in range(seq):
            for slot in range(slots):
                if not bool(valid[token, layer, slot].item()):
                    continue
                expert = int(index[token, layer, slot].item())
                if expert in seen:
                    continue
                seen.add(expert)
                found.append((layer, expert))
        layers.append(found)
    return layers


class ExpertPrefetcher:
    """Uma thread lê o store enquanto a CPU computa a camada corrente.

    O ``pread`` (``os.pread`` ou ``libc.pread``) solta o GIL. A thread não
    chama o torch, então não abre outro pool de threads. Não altera o
    ioprio: o processo herda ``IOSchedulingClass=idle``.

    No forward a fila é ``l+1 .. l+D``. No backward, a recomputação anda
    para trás, e a fila é ``l-1 .. l-D``. O buffer é separado do LRU e
    cabe em ``max_bytes``. Ao entregar um bloco, ele sai do buffer e
    entra no LRU sem expulsar a camada corrente.
    """

    def __init__(self, model: OffloadLanguageModel) -> None:
        self.model = model
        self.depth = 0
        self.max_bytes = 0
        self._block = 0
        self._plan: list[list[tuple[int, int]]] = []
        self._keep: set[tuple[int, int]] = set()
        self._queue: list[tuple[int, int]] = []
        self._buffer: dict[tuple[int, int], tuple[bytes, bytes, bytes]] = {}
        self._sizes: dict[tuple[int, int], int] = {}
        self._buffer_bytes = 0
        self._inflight: set[tuple[int, int]] = set()
        self._claimed: set[tuple[int, int]] = set()
        self._stop = False
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._cv = threading.Condition()
        self.hits = 0
        self.wait_s = 0.0
        self.wait_forward_s = 0.0
        self.wait_backward_s = 0.0
        self.hidden_read_s = 0.0
        self.background_bytes = 0
        self.prefetch_wasted = 0
        self.prefetch_skipped_vram = 0
        self.vram_aware = False
        self._block_nbytes = 0
        self._ring: list[torch.Tensor] = []
        self._free_slots: list[int] = []
        self._slot_of: dict[int, int] = {}

    @property
    def enabled(self) -> bool:
        return self._thread is not None

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, layers: int, max_bytes: int, block: int, block_nbytes: int | None = None) -> None:
        if layers < 0 or max_bytes < 0 or block < 0:
            raise ValueError("prefetch precisa ser >= 0")
        self.stop()
        self.vram_aware = bool(getattr(self.model, "prefetch_vram_aware", False))
        self._block_nbytes = int(block if block_nbytes is None else block_nbytes)
        if layers == 0 or max_bytes == 0 or block == 0:
            self.depth = 0
            self.max_bytes = 0
            return
        self.depth = layers
        self.max_bytes = max_bytes
        self._block = block
        self._stop = False
        self._error = None
        if self.vram_aware:
            self._alloc_ring()
        self._thread = threading.Thread(target=self._run, name="expert-prefetch", daemon=False)
        self._thread.start()

    def stop(self) -> None:
        thread = None
        with self._cv:
            self._stop = True
            self._queue.clear()
            self._cv.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        with self._cv:
            self._thread = None
            self._buffer.clear()
            self._sizes.clear()
            self._buffer_bytes = 0
            self._inflight.clear()
            self._claimed.clear()
            self._queue.clear()
            self._ring.clear()
            self._free_slots.clear()
            self._slot_of.clear()
            self.depth = 0
        self.model.expert_cache.set_pinned(set())

    def set_plan(self, layers: list[list[tuple[int, int]]]) -> None:
        if not self.enabled:
            return
        with self._cv:
            self._plan = layers

    def arm(self, layer: int, direction: int) -> None:
        if not self.enabled or not self._plan:
            return
        if direction not in (1, -1):
            raise ValueError("o prefetch anda para a frente ou para trás")
        with self._cv:
            current = list(self._plan[layer]) if 0 <= layer < len(self._plan) else []
            wanted: list[tuple[int, int]] = []
            for step in range(1, self.depth + 1):
                nxt = layer + direction * step
                if 0 <= nxt < len(self._plan):
                    wanted.extend(self._plan[nxt])
            keep = set(current)
            keep.update(wanted)
            self._keep = keep
            self.model.expert_cache.set_pinned(set(current))
            self._drop_unwanted(keep)
            queued: list[tuple[int, int]] = []
            for key in wanted:
                in_vram = self._vram_has(key)
                if in_vram:
                    self.prefetch_skipped_vram += 1
                    if self.vram_aware:
                        continue
                if not self._present(key):
                    queued.append(key)
            self._queue = queued
            self._cv.notify_all()

    def obtain(self, layer: int, expert: int, kind: str) -> tuple[tuple[bytes, bytes, bytes], float | None]:
        """Devolve os bytes e, se a leitura foi síncrona, o tempo dela."""

        key = (layer, expert)
        while True:
            with self._cv:
                self._raise()
                cached = self.model.expert_cache.touch(key)
                if cached is not None:
                    return cached, None
                buffered = self._buffer.get(key)
                if buffered is not None:
                    self._take_buffer(key)
                    self.hits += 1
                    self.model.expert_cache.note_miss()
                    if isinstance(buffered, torch.Tensor) and self.model.expert_cache.enabled:
                        copied = self._bytes_of(buffered)
                        self._release_tensor_locked(buffered)
                        self.model.expert_cache.store(key, copied, self._nbytes(copied))
                        return copied, None
                    if not isinstance(buffered, torch.Tensor):
                        self.model.expert_cache.store(key, buffered, self._nbytes(buffered))
                    return buffered, None
                if key in self._inflight:
                    waited = self._wait_locked()
                    self._note_wait(waited)
                    continue
                if key in self._queue:
                    self._queue.remove(key)
                self._claimed.add(key)
            started = time.perf_counter()
            try:
                with _region(f"expert_read_{kind}"):
                    blobs = self._read_fresh(key) if self.vram_aware else self._payload(key)
            finally:
                with self._cv:
                    self._claimed.discard(key)
                    self._cv.notify_all()
            sync_s = time.perf_counter() - started
            with self._cv:
                self.model.store_reads += 1
                self.model.expert_cache.note_miss()
                if isinstance(blobs, torch.Tensor) and self.model.expert_cache.enabled:
                    blobs = self._bytes_of(blobs)
                if not isinstance(blobs, torch.Tensor):
                    self.model.expert_cache.store(key, blobs, self._nbytes(blobs))
            return blobs, sync_s

    def reset_counters(self) -> None:
        with self._cv:
            self.hits = 0
            self.wait_s = 0.0
            self.wait_forward_s = 0.0
            self.wait_backward_s = 0.0
            self.hidden_read_s = 0.0
            self.background_bytes = 0
            self.prefetch_wasted = 0
            self.prefetch_skipped_vram = 0
            self.model.store_reads = 0

    def stats(self) -> dict[str, float | int]:
        with self._cv:
            return {
                "layers": self.depth,
                "max_bytes": self.max_bytes,
                "hits": self.hits,
                "wait_s": self.wait_s,
                "wait_forward_s": self.wait_forward_s,
                "wait_backward_s": self.wait_backward_s,
                "hidden_read_s": self.hidden_read_s,
                "background_bytes": self.background_bytes,
                "buffer_bytes": self._buffer_bytes,
                "prefetch_wasted": self.prefetch_wasted,
                "prefetch_skipped_vram": self.prefetch_skipped_vram,
            }

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._stop and not self._queue:
                    self._cv.wait()
                if self._stop:
                    return
                if self._block > self.max_bytes:
                    self._queue.clear()
                    continue
                if self._buffer_bytes + self._block > self.max_bytes:
                    self._cv.wait(timeout=0.05)
                    continue
                key = self._queue.pop(0)
                if self._present(key) or key in self._claimed:
                    continue
                if self.vram_aware and self._vram_has(key):
                    continue
                slot = None
                if self.vram_aware:
                    if not self._free_slots:
                        self._queue.insert(0, key)
                        self._cv.wait(timeout=0.05)
                        continue
                    slot = self._free_slots.pop()
                self._inflight.add(key)
            try:
                started = time.perf_counter()
                blobs = self._read_ring(key, slot) if slot is not None else self._payload(key)
                elapsed = time.perf_counter() - started
            except Exception as exc:
                with self._cv:
                    self._error = exc
                    self._inflight.discard(key)
                    if slot is not None:
                        self._free_slots.append(slot)
                    self._cv.notify_all()
                return
            nbytes = self._nbytes(blobs)
            with self._cv:
                self._inflight.discard(key)
                self.hidden_read_s += elapsed
                self.background_bytes += nbytes
                self.model.store_reads += 1
                if not self._stop and key not in self._claimed and key not in self._buffer:
                    self._buffer[key] = blobs
                    self._sizes[key] = nbytes
                    self._buffer_bytes += nbytes
                    if slot is not None:
                        self._slot_of[id(blobs)] = slot
                else:
                    self.prefetch_wasted += 1
                    if slot is not None:
                        self._free_slots.append(slot)
                self._cv.notify_all()

    def _payload(self, key: tuple[int, int]) -> tuple[bytes, bytes, bytes]:
        # Fora do lock. os.pread e libc.pread soltam o GIL.
        return self.model.store.payload(key[0], key[1])

    def _alloc_ring(self) -> None:
        slots = max(1, self.max_bytes // max(self._block, 1))
        self._ring = []
        self._free_slots = []
        self._slot_of = {}
        for index in range(slots):
            buf = torch.empty(self._block_nbytes, dtype=torch.uint8)
            if torch.cuda.is_available():
                buf = buf.pin_memory()
            self._ring.append(buf)
            self._free_slots.append(index)

    def _read_ring(self, key: tuple[int, int], slot: int) -> torch.Tensor:
        dest = self._ring[slot]
        self.model.store.read_into(key[0], key[1], dest)
        return dest

    def _read_fresh(self, key: tuple[int, int]) -> torch.Tensor:
        """Leitura síncrona na thread do passo. Não espera uma vaga do anel."""

        dest = torch.empty(self._block_nbytes, dtype=torch.uint8)
        if torch.cuda.is_available():
            dest = dest.pin_memory()
        self.model.store.read_into(key[0], key[1], dest)
        return dest

    def _bytes_of(self, blobs: torch.Tensor) -> tuple[bytes, bytes, bytes]:
        layout = self.model.store.layout
        raw = bytes(blobs[: layout.raw_nbytes].numpy())
        gate = raw[: layout.gate_nbytes]
        up = raw[layout.gate_nbytes : layout.gate_nbytes + layout.up_nbytes]
        down = raw[layout.gate_nbytes + layout.up_nbytes : layout.raw_nbytes]
        return gate, up, down

    def release_block(self, blobs) -> None:
        if not isinstance(blobs, torch.Tensor):
            return
        with self._cv:
            self._release_tensor_locked(blobs)
            self._cv.notify_all()

    def _release_tensor_locked(self, blobs: torch.Tensor) -> None:
        slot = self._slot_of.pop(id(blobs), None)
        if slot is not None:
            self._free_slots.append(slot)

    def _vram_has(self, key: tuple[int, int]) -> bool:
        gpu = getattr(self.model, "gpu", None)
        cache = getattr(gpu, "vram_cache", None) if gpu is not None else None
        if cache is None or not getattr(cache, "enabled", False):
            return False
        contains = getattr(cache, "contains", None)
        if contains is None:
            return False
        try:
            return bool(contains(key))
        except RuntimeError:
            return False

    def _present(self, key: tuple[int, int]) -> bool:
        return (
            key in self._buffer
            or key in self._inflight
            or key in self._claimed
            or self.model.expert_cache.contains(key)
        )

    def _drop_unwanted(self, keep: set[tuple[int, int]]) -> None:
        for key in list(self._buffer):
            if key in keep:
                continue
            self._buffer_bytes -= self._sizes.pop(key)
            blobs = self._buffer.pop(key)
            self.prefetch_wasted += 1
            if isinstance(blobs, torch.Tensor):
                self._release_tensor_locked(blobs)

    def _take_buffer(self, key: tuple[int, int]) -> None:
        del self._buffer[key]
        self._buffer_bytes -= self._sizes.pop(key)
        self._cv.notify_all()

    def _wait_locked(self) -> float:
        started = time.perf_counter()
        self._cv.wait()
        return time.perf_counter() - started

    def _note_wait(self, waited: float) -> None:
        self.wait_s += waited
        if self.model._io_phase == "forward":
            self.wait_forward_s += waited
            self.model._add_time("prefetch_wait_s", waited)
        else:
            self.wait_backward_s += waited

    def _raise(self) -> None:
        if self._error is not None:
            raise RuntimeError("prefetch do expert falhou") from self._error

    def _nbytes(self, blobs) -> int:
        if isinstance(blobs, torch.Tensor):
            return self._block
        return len(blobs[0]) + len(blobs[1]) + len(blobs[2])


def simulate_expert_lru(reads: list[tuple[tuple[int, int], int]], capacity: int) -> dict[str, int]:
    """Hits de um LRU. Não lê o store e não muda o treino."""

    if capacity < 0:
        raise ValueError("capacidade do LRU precisa ser >= 0")
    cache: list[tuple[int, int]] = []
    hits = 0
    avoided = 0
    for key, nbytes in reads:
        if key in cache:
            hits += 1
            avoided += nbytes
            cache.remove(key)
            cache.append(key)
            continue
        if capacity == 0:
            continue
        if len(cache) >= capacity:
            cache.pop(0)
        cache.append(key)
    return {"hits": hits, "avoided_bytes": avoided, "reads": len(reads)}


def _attention_seq(layer, hidden: torch.Tensor, spec: QwenSpec) -> torch.Tensor:
    model = _EXPERT_STATE.get("model")
    started = time.perf_counter() if model is not None and model.time_log else None
    result = _attention_seq_body(layer, hidden, spec)
    if started is not None and model is not None:
        model._add_time(model._phase_key("attention_s"), time.perf_counter() - started)
    return result


def _attention_seq_body(layer, hidden: torch.Tensor, spec: QwenSpec) -> torch.Tensor:
    batch, seq, _width = hidden.shape
    if batch != 1 or seq < 1:
        raise ValueError("a atenção do treino é batch 1")
    with _region("qkvo"):
        query = F.linear(hidden, _as(layer.q, hidden), _as(layer.q_bias, hidden))
        key = F.linear(hidden, _as(layer.k, hidden), _as(layer.k_bias, hidden))
        value = F.linear(hidden, _as(layer.v, hidden), _as(layer.v_bias, hidden))
    query = query.view(batch, seq, spec.n_heads, spec.head_dim)
    key = key.view(batch, seq, spec.n_kv_heads, spec.head_dim)
    value = value.view(batch, seq, spec.n_kv_heads, spec.head_dim)
    query = rms_norm(query, _as(layer.q_norm, hidden), spec.rms_eps)
    key = rms_norm(key, _as(layer.k_norm, hidden), spec.rms_eps)
    positions = torch.arange(seq, device=hidden.device)
    cos, sin = _rope_seq(positions, spec.head_dim, spec.rope_theta, hidden.dtype)
    query = _apply_rope(query.transpose(1, 2), cos, sin)
    key = _apply_rope(key.transpose(1, 2), cos, sin)
    value = value.transpose(1, 2)
    groups = spec.n_heads // spec.n_kv_heads
    mixed = F.scaled_dot_product_attention(
        query,
        _repeat_kv(key, groups),
        _repeat_kv(value, groups),
        is_causal=seq > 1,
    )
    mixed = mixed.transpose(1, 2).reshape(batch, seq, spec.n_heads * spec.head_dim)
    with _region("qkvo"):
        return F.linear(mixed, _as(layer.o, hidden), _as(layer.o_bias, hidden))


_REGION_PREFIXES = (
    "router_plan",
    "cross_entropy",
    "attention_layer_",
    "checkpoint_recompute_layer_",
    "qkvo",
    "bf16_fp32",
    "expert_read_",
    "expert_compute_",
    "lora",
    "optimizer_step",
)


@contextmanager
def profile_lm_step(out_dir: str | Path, *, with_stack: bool = False, with_trace: bool = False):
    """Perfila só o bloco `with`. Sem profile_memory. A stack só se pedida."""

    global _PROFILE_LABELS
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    _PROFILE_LABELS = True
    try:
        with profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=False,
            with_stack=with_stack,
        ) as prof:
            yield prof
    finally:
        _PROFILE_LABELS = False
    write_step_profile(prof, directory, with_stack=with_stack, with_trace=with_trace)


def write_step_profile(prof, out_dir: Path, *, with_stack: bool, with_trace: bool) -> None:
    ops = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=50)
    (out_dir / "profile_ops_top50.txt").write_text(ops, encoding="utf-8")
    shapes = prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total", row_limit=30)
    (out_dir / "profile_shapes_top30.txt").write_text(shapes, encoding="utf-8")
    (out_dir / "profile_regions.json").write_text(
        json.dumps(_profile_summary(prof, with_stack=with_stack, with_trace=with_trace), indent=2),
        encoding="utf-8",
    )
    if with_trace:
        prof.export_chrome_trace(str(out_dir / "profile_trace.json"))


def _profile_summary(prof, *, with_stack: bool, with_trace: bool) -> dict:
    events = list(prof.key_averages())
    regions: dict[str, dict[str, float | int]] = {}
    total_calls = 0
    aten_calls = 0
    autograd_calls = 0
    small_calls = 0
    for event in events:
        count = int(event.count)
        total_calls += count
        name = str(event.key)
        if name.startswith("aten::"):
            aten_calls += count
        if "autograd" in name.lower() or name.startswith("Autograd"):
            autograd_calls += count
        self_us = float(getattr(event, "self_cpu_time_total", 0.0))
        if count and self_us / count < 100:
            small_calls += count
        if not any(name == prefix or name.startswith(prefix) for prefix in _REGION_PREFIXES):
            continue
        regions[name] = {
            "self_cpu_s": self_us / 1e6,
            "cpu_s": float(getattr(event, "cpu_time_total", 0.0)) / 1e6,
            "count": count,
        }
    return {
        "regions": regions,
        "calls": {
            "total": total_calls,
            "aten": aten_calls,
            "autograd": autograd_calls,
            "under_100us_self": small_calls,
        },
        "profile_memory": False,
        "with_stack": with_stack,
        "with_trace": with_trace,
        "overhead_note": (
            "Sem profile_memory e sem with_stack o buffer fica na ordem de dezenas "
            "a poucas centenas de MB durante o passo. O trace chrome pode passar disso. "
            "Não foi medido na 1060."
        ),
    }


def _rope_seq(positions: torch.Tensor, head_dim: int, theta: float, dtype: torch.dtype):
    half = torch.arange(0, head_dim, 2, device=positions.device, dtype=torch.float32)
    inv = 1.0 / (theta ** (half / head_dim))
    freqs = positions.float().unsqueeze(1) * inv.unsqueeze(0)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype)[None, None, :, :], emb.sin().to(dtype)[None, None, :, :]
