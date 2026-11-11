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
"""

from __future__ import annotations

import ctypes
import gc
import hashlib
import json
import resource
import time
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
) -> dict[str, int | None]:
    """Teto do anon com a cabeça e o expert em pedaços. Não inclui o store inteiro.

    ``resident_page_cache_if_kept`` é o cache do arquivo se ninguém pedir
    ``DONTNEED``. Não entra na soma. ``cgroup_bytes`` vem do processo, ou
    do teste.
    """

    if min(resident_bytes, vocab, hidden, intermediate, n_layers, n_experts, rank, seq_len, k1) < 1:
        raise ValueError("estimativa com dimensão < 1")
    lora = n_layers * n_experts * 3 * rank * (hidden + intermediate) * 4
    lstm_params = (
        vocab * lstm_embed
        + 4 * lstm_hidden * lstm_embed
        + 4 * lstm_hidden * lstm_hidden
        + 8 * lstm_hidden
        + n_layers * (lstm_hidden * n_experts + n_experts)
    )
    one_expert = 3 * hidden * intermediate * 4
    if n_heads is None or n_kv_heads is None or head_dim is None:
        attention = 0
    else:
        q = n_heads * head_dim * hidden * 4
        kv = n_kv_heads * head_dim * hidden * 4
        attention = q + 2 * kv + hidden * n_heads * head_dim * 4
    head = min(vocab, LM_HEAD_CHUNK) * hidden * 4
    if cgroup_bytes is None and read_cgroup:
        cgroup_bytes = read_cgroup_memory_max()
    pieces: dict[str, int | None] = {
        "resident_bytes": resident_bytes,
        "lora_fp32_bytes": lora,
        "lora_grad_bytes": lora,
        "lora_select_backward_bytes": 0,
        "adam_lora_bytes": lora * 2,
        "lstm_with_adam_bytes": lstm_params * 12,
        "lstm_grad_bytes": lstm_params * 4,
        "expert_fp32_bytes": one_expert,
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
    }
    pieces["total_bytes"] = sum(value for key, value in pieces.items() if key not in skipped and value is not None)
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
        self.lora = ExpertLoRA(spec.n_layers, spec.n_experts, spec.hidden_size, spec.moe_intermediate, rank, alpha)
        self.store_dtype = torch_dtype(store.header.dtype_code)
        self.memory_log = False
        self.time_log = False
        self.cache_sim_sizes: tuple[int, ...] = ()
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
        started = time.perf_counter()
        with _region("router_plan"):
            plan, _state = self.router.plan(token_ids)
        self._add_time("router_plan_s", time.perf_counter() - started)
        hidden = self._embed(token_ids)
        self._phase("embed")
        spec = self.weights.spec
        for layer_index in range(spec.n_layers):
            self._phase(f"layer-{layer_index}")
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
            loss = _chunked_cross_entropy(
                normed[:, :-1].reshape(-1, normed.shape[-1]),
                self.weights.lm_head,
                token_ids[:, 1:].reshape(-1),
            )
        self._add_time("cross_entropy_forward_s", time.perf_counter() - started)
        self._phase("loss")
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

    def _layer_body(self, hidden, coeffs, index, valid, spec, layer, layer_index):
        started = time.perf_counter()
        normed = rms_norm(hidden, _as(layer.input_norm, hidden), spec.rms_eps)
        self._add_time(self._phase_key("resident_s"), time.perf_counter() - started)
        with _region(f"attention_layer_{layer_index}"):
            hidden = hidden + _attention_seq(layer, normed, spec)
        started = time.perf_counter()
        moe_in = rms_norm(hidden, _as(layer.post_norm, hidden), spec.rms_eps)
        self._add_time(self._phase_key("resident_s"), time.perf_counter() - started)
        hidden = hidden + self._moe(moe_in, layer_index, index, valid, coeffs)
        return hidden

    def _moe(self, hidden, layer_index, index, valid, coeffs):
        flat = hidden[0]
        out = flat.new_zeros(flat.shape)
        groups: dict[int, list[tuple[int, int]]] = {}
        seq, slots = index.shape[1], index.shape[2]
        for token in range(seq):
            for slot in range(slots):
                if not bool(valid[0, token, slot].item()):
                    continue
                expert = int(index[0, token, slot].item())
                groups.setdefault(expert, []).append((token, slot))
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

    def _read(self, layer: int, expert: int, kind: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        started = time.perf_counter()
        with _region(f"expert_read_{kind}"):
            gate_b, up_b, down_b = self.store.payload(layer, expert)
        self._add_time(f"expert_read_{kind}_s", time.perf_counter() - started)
        nbytes = len(gate_b) + len(up_b) + len(down_b)
        self._add_time(f"expert_read_{kind}_bytes", nbytes)
        if self.time_log:
            self._read_seq.append(((layer, expert), nbytes))
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

    def timing_report(self, wall_s: float) -> dict[str, float | int | dict | None]:
        exclusive = (
            "router_plan_s",
            "embed_s",
            "resident_s",
            "attention_s",
            "expert_read_forward_s",
            "expert_convert_forward_s",
            "expert_compute_forward_s",
            "cross_entropy_forward_s",
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
        return report


def lm_step(model: OffloadLanguageModel, optimizer: torch.optim.Optimizer, token_ids: torch.Tensor) -> dict[str, float | int | bool]:
    model.router.train()
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
    report["loss"] = float(loss.detach())
    report["distinct_experts"] = distinct_experts(plan)
    report["hot_entropy"] = _hot_entropy(plan)
    report["router_entropy"] = _router_entropy(model, token_ids)
    del loss, plan
    started = time.perf_counter()
    release_step_memory(optimizer)
    model._add_time("release_s", time.perf_counter() - started)
    report["rss_bytes"] = rss_bytes()
    report["ru_maxrss_bytes"] = ru_maxrss_bytes()
    if model.time_log:
        report["time"] = model.timing_report(time.perf_counter() - wall)
    model._phase("after-step")
    return report


def make_optimizer(model: OffloadLanguageModel, lr: float) -> torch.optim.Optimizer:
    """AdamW sem ``foreach``. O caminho agrupado monta temporários do tamanho da LoRA inteira."""

    return torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=lr,
        weight_decay=0.0,
        foreach=False,
    )


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


def grad_report(model: OffloadLanguageModel) -> dict[str, float | bool]:
    lstm = model.router.lstm.weight_ih_l0.grad
    lora_b = []
    finite = True
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
        "grads_finite": finite and lstm is not None and bool(torch.isfinite(lstm).all().item()),
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


def save_checkpoint(path: str | Path, *, step: int, model: OffloadLanguageModel, optimizer: torch.optim.Optimizer, extra: dict) -> None:
    """Grava um tensor por arquivo. Não monta o estado inteiro numa cópia só."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
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
    for group, saved in zip(optimizer.param_groups, meta["param_groups"], strict=True):
        for key, value in saved.items():
            if key in group and key != "params":
                group[key] = value
    for entry in meta["entries"]:
        parameter = named[entry["name"]]
        if entry["kind"] == "param":
            _read_tensor_into(directory / entry["file"], parameter, entry)
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
    return meta


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
        for name in ("q_bias", "k_bias", "v_bias", "o_bias"):
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
    lora = model.lora
    return _ExpertFn.apply(
        hidden,
        coeff,
        lora.gate_A[layer, expert],
        lora.gate_B[layer, expert],
        lora.up_A[layer, expert],
        lora.up_B[layer, expert],
        lora.down_A[layer, expert],
        lora.down_B[layer, expert],
    )


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
        ctx.scale = model.lora.scale
        kind = "forward" if model._io_phase == "forward" else "recompute"
        gate, up, down = model._read(layer, expert, kind)
        model._note_expert(1)
        started = time.perf_counter()
        try:
            with _region(f"expert_compute_{kind}"):
                with torch.no_grad():
                    projected = _swiglu(hidden, gate, up, down, gate_a, gate_b, up_a, up_b, down_a, down_b, ctx.scale)
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
            inputs = [
                hidden.detach().requires_grad_(True),
                coeff.detach().requires_grad_(True),
                gate_a.detach().requires_grad_(True),
                gate_b.detach().requires_grad_(True),
                up_a.detach().requires_grad_(True),
                up_b.detach().requires_grad_(True),
                down_a.detach().requires_grad_(True),
                down_b.detach().requires_grad_(True),
            ]
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
            return grads
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


def _queue_slice_grad(param: torch.nn.Parameter, layer: int, expert: int, grad: torch.Tensor | None) -> None:
    if grad is None:
        return
    _SLICE_GRADS.append((param, int(layer), int(expert), grad.detach()))


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


def _chunked_cross_entropy(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return _ChunkedCrossEntropy.apply(hidden, weight, targets)


class _ChunkedCrossEntropy(torch.autograd.Function):
    """CE do próximo token sem materializar a cabeça inteira em float32."""

    @staticmethod
    def forward(ctx, hidden, weight, targets):
        ctx.save_for_backward(hidden, weight, targets)
        ctx.chunk = LM_HEAD_CHUNK
        with torch.no_grad():
            loss = _ce_value(hidden, weight, targets, LM_HEAD_CHUNK)
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        hidden, weight, targets = ctx.saved_tensors
        started = time.perf_counter()
        with _region("cross_entropy_backward"):
            grad_hidden = _ce_grad_hidden(hidden, weight, targets, ctx.chunk, grad_loss)
        model = _EXPERT_STATE.get("model")
        if model is not None:
            model._add_time("cross_entropy_backward_s", time.perf_counter() - started)
        return grad_hidden, None, None


def _ce_value(hidden, weight, targets, chunk: int) -> torch.Tensor:
    rows = hidden.shape[0]
    running_max = hidden.new_full((rows,), float("-inf"))
    vocab = weight.shape[0]
    for start in range(0, vocab, chunk):
        logits = F.linear(hidden, weight[start : start + chunk].to(dtype=hidden.dtype))
        running_max = torch.maximum(running_max, logits.detach().max(dim=-1).values)
    sum_exp = hidden.new_zeros(rows)
    target_logit = hidden.new_zeros(rows)
    index = torch.arange(rows, device=hidden.device)
    for start in range(0, vocab, chunk):
        block = weight[start : start + chunk].to(dtype=hidden.dtype)
        logits = F.linear(hidden, block)
        sum_exp = sum_exp + (logits - running_max.unsqueeze(-1)).exp().sum(dim=-1)
        chosen = (targets >= start) & (targets < start + block.shape[0])
        local = (targets - start).clamp(0, block.shape[0] - 1)
        picked = logits[index, local]
        target_logit = torch.where(chosen, picked, target_logit)
    nll = running_max + sum_exp.log() - target_logit
    return nll.mean()


def _ce_grad_hidden(hidden, weight, targets, chunk: int, grad_loss) -> torch.Tensor:
    hidden = hidden.detach()
    rows = hidden.shape[0]
    with torch.no_grad():
        running_max = hidden.new_full((rows,), float("-inf"))
        vocab = weight.shape[0]
        for start in range(0, vocab, chunk):
            block = weight[start : start + chunk].to(dtype=hidden.dtype)
            logits = F.linear(hidden, block)
            running_max = torch.maximum(running_max, logits.max(dim=-1).values)
            del block, logits
        sum_exp = hidden.new_zeros(rows)
        for start in range(0, vocab, chunk):
            block = weight[start : start + chunk].to(dtype=hidden.dtype)
            logits = F.linear(hidden, block)
            sum_exp = sum_exp + (logits - running_max.unsqueeze(-1)).exp().sum(dim=-1)
            del block, logits
        log_z = running_max + sum_exp.log()
        grad = hidden.new_zeros(hidden.shape)
        scale = grad_loss.to(dtype=hidden.dtype) / rows
        for start in range(0, vocab, chunk):
            block = weight[start : start + chunk].to(dtype=hidden.dtype)
            logits = F.linear(hidden, block)
            probs = (logits - log_z.unsqueeze(-1)).exp()
            chosen = (targets >= start) & (targets < start + block.shape[0])
            if bool(chosen.any().item()):
                local = targets[chosen] - start
                probs[chosen, local] = probs[chosen, local] - 1
            grad = grad + probs.matmul(block) * scale
            del block, logits, probs
    return grad


def _matrix(raw: bytes, dtype: torch.dtype, shape: tuple[int, int]) -> torch.Tensor:
    elem = torch.empty((), dtype=dtype).element_size()
    need = shape[0] * shape[1] * elem
    if len(raw) != need:
        raise ValueError(f"bloco do expert tem {len(raw)} bytes, esperado {need}")
    cpu = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape).clone()
    return cpu.to(dtype=torch.float32)


def _delta(hidden: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return F.linear(F.linear(hidden, left), right)


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
