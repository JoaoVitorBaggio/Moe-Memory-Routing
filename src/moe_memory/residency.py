"""Storage físico dos experts, alinhado ao ``ℓ(e)`` do emulador.

No CUDA (e no mock de VRAM usado no CI sem GPU):

- quente: o ``nn.Parameter`` fica no device de compute;
- morno: CPU pinned; ``stage_for_compute`` copia para a VRAM com
  ``non_blocking`` quando o top-k escolhe o expert;
- frio: memmap no disco; o parâmetro aponta para esse arquivo e só ganha
  storage de compute quando é promovido ou escolhido no forward.

A subida para o compute é temporária. Depois do ``optimizer.step`` (ou
entre batches da medição de parede) ``materialize`` devolve quem está
fora de ``B`` para o home. Trocas e bytes do LRU continuam só no
emulador: ``staged_loads`` / ``staged_bytes`` contam a cópia de compute,
que é o que o relógio de parede vê, e não entram em ``total_swaps``.

No CPU do smoke o backend é ``None`` e os parâmetros não mudam de
storage. O store ainda grava a cópia morna/fria para os testes de arquivo.

Procedimento da latência na GTX 1060: ``docs/plano-de-hardware.md``.
"""

from __future__ import annotations

from typing import Protocol

import torch
from torch import nn

from moe_memory.emulator import PhysicalExpertStore, ResidencyEmulator
from moe_memory.levels import COLD, HOT, WARM


class VramBackend(Protocol):
    def on_device(self, tensor: torch.Tensor) -> bool: ...

    def is_pinned(self, tensor: torch.Tensor) -> bool: ...

    def pin(self, tensor: torch.Tensor) -> torch.Tensor: ...

    def to_device(self, tensor: torch.Tensor, *, non_blocking: bool = False) -> torch.Tensor: ...


class CudaVram:
    """VRAM real. ``pin_memory`` e ``.to(cuda, non_blocking=True)``."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = torch.device("cuda") if device is None else device

    def on_device(self, tensor: torch.Tensor) -> bool:
        return tensor.device.type == "cuda"

    def is_pinned(self, tensor: torch.Tensor) -> bool:
        return tensor.device.type == "cpu" and tensor.is_pinned()

    def pin(self, tensor: torch.Tensor) -> torch.Tensor:
        cpu = tensor.detach().to(dtype=torch.float32, device="cpu").contiguous()
        if not cpu.is_pinned():
            cpu = cpu.pin_memory()
        return cpu

    def to_device(self, tensor: torch.Tensor, *, non_blocking: bool = False) -> torch.Tensor:
        if non_blocking and not self.is_pinned(tensor):
            non_blocking = False
        return tensor.detach().to(device=self.device, dtype=torch.float32, non_blocking=non_blocking)


class MockVram:
    """CI sem GPU. O byte fica na CPU; um conjunto marca quem 'ocupa VRAM'.

    ``prepare`` simula o ``model.to(cuda)``: todo parâmetro atual conta
    como residente. ``note_view`` propaga a marca para fatias do flat,
    porque o ``data_ptr`` de um offset não é o do tensor base.
    """

    def __init__(self) -> None:
        self._vram: set[int] = set()
        self._pinned: set[int] = set()
        self.last_non_blocking: list[bool] = []

    def prepare(self, module: nn.Module) -> None:
        for parameter in module.parameters():
            self._vram.add(parameter.data_ptr())

    def note_view(self, view: torch.Tensor, base: torch.Tensor) -> None:
        if base.data_ptr() in self._vram or view.data_ptr() in self._vram:
            self._vram.add(view.data_ptr())
        if base.data_ptr() in self._pinned:
            self._pinned.add(view.data_ptr())

    def on_device(self, tensor: torch.Tensor) -> bool:
        return tensor.data_ptr() in self._vram

    def is_pinned(self, tensor: torch.Tensor) -> bool:
        return tensor.data_ptr() in self._pinned and tensor.data_ptr() not in self._vram

    def pin(self, tensor: torch.Tensor) -> torch.Tensor:
        out = tensor.detach().to(dtype=torch.float32, device="cpu").contiguous().clone()
        self._pinned.add(out.data_ptr())
        return out

    def to_device(self, tensor: torch.Tensor, *, non_blocking: bool = False) -> torch.Tensor:
        self.last_non_blocking.append(bool(non_blocking))
        out = tensor.detach().to(dtype=torch.float32, device="cpu").contiguous().clone()
        self._vram.add(out.data_ptr())
        return out


class ExpertResidency:
    def __init__(
        self,
        emulator: ResidencyEmulator,
        model: nn.Module,
        backend: VramBackend | None,
    ) -> None:
        if model.n_layers != emulator.n_layers or model.n_experts != emulator.n_experts:
            raise ValueError(
                "emulador e modelo não têm a mesma grade de experts "
                f"({model.n_layers}×{model.n_experts} vs "
                f"{emulator.n_layers}×{emulator.n_experts})"
            )
        self.emulator = emulator
        self.model = model
        self.backend = backend
        self.optimizer: torch.optim.Optimizer | None = None
        self.loads = 0
        self.bytes_loaded = 0
        self._store: PhysicalExpertStore | None = None

    def materialize(self, optimizer: torch.optim.Optimizer | None = None) -> PhysicalExpertStore:
        """Alinha o storage ao ``ℓ(e)`` corrente. Não mexe nos contadores do LRU."""

        if optimizer is not None:
            self.optimizer = optimizer
        if self._store is None:
            self._store = PhysicalExpertStore(self.emulator.work_dir)
            self.emulator.bind_store(self._store)
        store = self._store
        for layer, expert_id, expert in _iter_experts(self.model):
            level = int(self.emulator.levels[layer, expert_id].item())
            params = list(expert.parameters())
            if self._already_home(params, level, store, layer, expert_id):
                continue
            flat = _flat(params)
            self._park(params, layer, expert_id, level, flat, store)
        return store

    def stage_for_compute(self, layer: int, expert_id: int) -> None:
        """Sobe o expert para o device de compute se o forward vai usá-lo."""

        if self.backend is None:
            return
        expert = self.model.blocks[layer].ffn.experts[expert_id]
        params = list(expert.parameters())
        if params and all(self.backend.on_device(parameter.data) for parameter in params):
            self._align_state(params, to_vram=True)
            return
        nbytes = sum(parameter.numel() * parameter.element_size() for parameter in params)
        for parameter in params:
            source = parameter.data
            parameter.data = self.backend.to_device(
                source, non_blocking=self.backend.is_pinned(source)
            )
        self.loads += 1
        self.bytes_loaded += nbytes
        self._align_state(params, to_vram=True)

    def _already_home(
        self,
        params: list[nn.Parameter],
        level: int,
        store: PhysicalExpertStore,
        layer: int,
        expert_id: int,
    ) -> bool:
        if self.backend is None or not params:
            return False
        on_device = [self.backend.on_device(parameter.data) for parameter in params]
        pinned = [self.backend.is_pinned(parameter.data) for parameter in params]
        key = (layer, expert_id)
        if level == HOT:
            return all(on_device) and key not in store._warm and key not in store._cold
        if level == WARM:
            return (not any(on_device)) and all(pinned) and key in store._warm
        if level == COLD:
            path = store._cold.get(key)
            return (
                (not any(on_device))
                and (not any(pinned))
                and path is not None
                and path.exists()
            )
        raise ValueError(f"nível desconhecido: {level}")

    def _park(
        self,
        params: list[nn.Parameter],
        layer: int,
        expert_id: int,
        level: int,
        flat: torch.Tensor,
        store: PhysicalExpertStore,
    ) -> None:
        if self.backend is None:
            store.place(layer, expert_id, level, flat)
            return
        if level == HOT:
            store.place(layer, expert_id, HOT, flat)
            if not all(self.backend.on_device(parameter.data) for parameter in params):
                _scatter(params, self.backend.to_device(flat), self.backend)
            self._align_state(params, to_vram=True)
            self._align_grad(params, to_vram=True)
            return
        if level == WARM:
            pinned = self.backend.pin(flat)
            store.adopt_warm(layer, expert_id, pinned)
            _scatter(params, pinned, self.backend)
            self._align_state(params, to_vram=False)
            self._align_grad(params, to_vram=False)
            return
        if level != COLD:
            raise ValueError(f"nível desconhecido: {level}")
        store.place(layer, expert_id, COLD, flat)
        _scatter(params, store.cold_view(layer, expert_id), self.backend)
        self._align_state(params, to_vram=False)
        self._align_grad(params, to_vram=False)

    def _align_state(self, params: list[nn.Parameter], *, to_vram: bool) -> None:
        optimizer = self.optimizer
        if optimizer is None or self.backend is None:
            return
        for parameter in params:
            state = optimizer.state.get(parameter)
            if not state:
                continue
            for key, value in list(state.items()):
                if not torch.is_tensor(value):
                    continue
                if to_vram:
                    if not self.backend.on_device(value):
                        state[key] = self.backend.to_device(value)
                elif self.backend.on_device(value):
                    state[key] = value.detach().to(device="cpu", dtype=value.dtype)

    def _align_grad(self, params: list[nn.Parameter], *, to_vram: bool) -> None:
        if self.backend is None:
            return
        for parameter in params:
            if parameter.grad is None:
                continue
            if to_vram:
                if not self.backend.on_device(parameter.grad):
                    parameter.grad = self.backend.to_device(parameter.grad)
            elif self.backend.on_device(parameter.grad):
                parameter.grad = parameter.grad.detach().to(device="cpu")


def attach_residency(
    model: nn.Module,
    emulator: ResidencyEmulator,
    device: torch.device,
    *,
    backend: VramBackend | None = None,
) -> ExpertResidency:
    """Liga o modelo ao emulador. CUDA real escolhe ``CudaVram``.

    Sem GPU, peça ``backend=MockVram()`` nos testes. ``device='cpu'`` com
    backend ``None`` é o smoke: não troca o storage do parâmetro.
    """

    if backend is None and device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device=cuda sem GPU visível. Na GTX 1060 instale o wheel CUDA; "
                "no CI use MockVram."
            )
        backend = CudaVram(device)
    residency = ExpertResidency(emulator, model, backend)
    model.bind_residency(residency)
    prepare = getattr(backend, "prepare", None)
    if callable(prepare):
        prepare(model)
    return residency


def _iter_experts(model: nn.Module):
    for layer, block in enumerate(model.blocks):
        for expert_id, expert in enumerate(block.ffn.experts):
            yield layer, expert_id, expert


def _flat(params: list[nn.Parameter]) -> torch.Tensor:
    pieces = [
        parameter.data.detach().reshape(-1).to(dtype=torch.float32, device="cpu")
        for parameter in params
    ]
    return torch.cat(pieces)


def _scatter(params: list[nn.Parameter], flat: torch.Tensor, backend: VramBackend | None) -> None:
    expected = sum(parameter.numel() for parameter in params)
    if flat.numel() != expected:
        raise RuntimeError(
            f"vetor do expert tem {flat.numel()} elementos; os parâmetros somam {expected}"
        )
    if flat.dtype != torch.float32:
        flat = flat.to(dtype=torch.float32)
    if not flat.is_contiguous():
        raise RuntimeError("home do expert precisa ser contíguo para fatiar sem cópia")
    note = getattr(backend, "note_view", None)
    offset = 0
    for parameter in params:
        count = parameter.numel()
        piece = flat.narrow(0, offset, count).view(parameter.shape)
        parameter.data = piece
        offset += count
        if callable(note):
            note(piece, flat)
