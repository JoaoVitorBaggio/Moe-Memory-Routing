"""Emulador de residência VRAM / RAM / SSD, um passo por batch.

Política do esqueleto: LRU com desempate pela contagem de despachos do
batch. Quem foi tocado neste batch fica mais recente; entre esses, quem
recebeu mais tokens no top-k fica mais quente. As ``B`` primeiras vagas
são o nível quente, as ``warm_slots`` seguintes o morno, o resto o frio.

O adversário LRU do protocolo reutiliza esta política com ``λ = 0``.
Não há segunda política neste esqueleto.

Contadores: uma troca por expert que mudou de nível; bytes =
``|Δℓ| × tamanho do expert`` (um salto frio→quente conta a cópia nos
dois degraus). ``w(a,b)`` não entra aqui — a loss cobra o surrogate;
o emulador conta o que a política fez, para a correlação do protocolo.

TODO(1060): na latência de parede, experts fora de ``B`` precisam estar
fisicamente fora da VRAM. ``PhysicalExpertStore`` já grava morno como
tensor de CPU (pinned se houver CUDA) e frio como memmap. O que falta é
trocar o storage do ``nn.Parameter`` por esse home *depois* do
``optimizer.step()``. No smoke de CPU os parâmetros continuam no device
de compute para o autograd não quebrar.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from moe_memory.levels import COLD, HOT, WARM

_NEVER_USED = -10**9


@dataclass(frozen=True)
class StepStats:
    swaps: int
    bytes_moved: int
    promotions: int
    demotions: int


class PhysicalExpertStore:
    """Cópia física dos pesos. Não substitui o ``nn.Parameter``."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._warm: dict[tuple[int, int], torch.Tensor] = {}
        self._cold: dict[tuple[int, int], Path] = {}

    def place(self, layer: int, expert: int, level: int, weight: torch.Tensor) -> None:
        key = (layer, expert)
        self._discard(key)
        if level == HOT:
            # Quente: o parâmetro no device de compute é o home.
            # TODO(1060): este ramo deve ser o storage CUDA do nn.Parameter.
            return
        flat = weight.detach().to(dtype=torch.float32, device="cpu").contiguous().reshape(-1)
        if level == WARM:
            if torch.cuda.is_available():
                flat = flat.pin_memory()
            self._warm[key] = flat
            return
        if level != COLD:
            raise ValueError(f"nível desconhecido: {level}")
        path = self.directory / f"layer{layer}_expert{expert}.f32"
        mapping = np.memmap(path, dtype=np.float32, mode="w+", shape=(flat.numel(),))
        mapping[:] = flat.numpy()
        mapping.flush()
        del mapping
        self._cold[key] = path

    def read(self, layer: int, expert: int, level: int) -> torch.Tensor:
        key = (layer, expert)
        if level == WARM:
            try:
                return self._warm[key]
            except KeyError as exc:
                raise KeyError(f"expert {(layer, expert)} não está no home morno") from exc
        if level == COLD:
            try:
                path = self._cold[key]
            except KeyError as exc:
                raise KeyError(f"expert {(layer, expert)} não está no home frio") from exc
            mapping = np.memmap(path, dtype=np.float32, mode="r")
            copied = np.empty(mapping.shape, dtype=np.float32)
            copied[:] = mapping
            del mapping
            return torch.from_numpy(copied)
        raise ValueError("o nível quente não tem cópia no store; use o nn.Parameter")

    @property
    def n_cold_files(self) -> int:
        return len(self._cold)

    @property
    def n_warm(self) -> int:
        return len(self._warm)

    def _discard(self, key: tuple[int, int]) -> None:
        self._warm.pop(key, None)
        path = self._cold.pop(key, None)
        if path is not None and path.exists():
            path.unlink()


class ResidencyEmulator:
    def __init__(
        self,
        n_layers: int,
        n_experts: int,
        hot_slots: int,
        warm_slots: int,
        expert_nbytes: int,
        work_dir: str | Path | None = None,
    ) -> None:
        if n_layers < 1 or n_experts < 2:
            raise ValueError("emulador precisa de ao menos 1 camada e 2 experts")
        if hot_slots < 1 or warm_slots < 1:
            raise ValueError("quente e morno precisam de ao menos uma vaga")
        if hot_slots + warm_slots > n_experts:
            raise ValueError("quente + morno não cabem em E")
        if expert_nbytes < 1:
            raise ValueError("expert_nbytes precisa ser positivo")
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.hot_slots = hot_slots
        self.warm_slots = warm_slots
        self.expert_nbytes = expert_nbytes
        self.work_dir = Path(work_dir) if work_dir is not None else Path("outputs/emulator_cold")
        self.levels = torch.empty(n_layers, n_experts, dtype=torch.long)
        self.last_used = torch.empty(n_layers, n_experts, dtype=torch.long)
        self.last_count = torch.empty(n_layers, n_experts, dtype=torch.long)
        self.total_swaps = 0
        self.total_bytes = 0
        self.tick = 0
        self._store: PhysicalExpertStore | None = None
        self.reset()

    def reset(self) -> None:
        """Experts ``0 .. B-1`` quentes, os próximos ``warm_slots`` mornos, o resto frio."""

        self.tick = 0
        self.total_swaps = 0
        self.total_bytes = 0
        self.levels = torch.full((self.n_layers, self.n_experts), COLD, dtype=torch.long)
        self.last_used = torch.full(
            (self.n_layers, self.n_experts), _NEVER_USED, dtype=torch.long
        )
        self.last_count = torch.zeros(self.n_layers, self.n_experts, dtype=torch.long)
        for expert in range(self.hot_slots):
            self.levels[:, expert] = HOT
            self.last_used[:, expert] = -expert
        for offset in range(self.warm_slots):
            expert = self.hot_slots + offset
            self.levels[:, expert] = WARM
            self.last_used[:, expert] = -(self.hot_slots + offset)

    def step(self, dispatched: torch.Tensor) -> StepStats:
        """``dispatched`` é ``[camadas, E]``: quantos slots top-k caíram em cada expert."""

        counts = self._counts(dispatched)
        self.tick += 1
        used = counts > 0
        self.last_used = self.last_used.clone()
        self.last_used[used] = self.tick
        self.last_count = counts
        return self._commit(self._targets())

    def set_hot_budget(self, hot_slots: int) -> StepStats:
        """Troca o teto ``B`` (0,5× / 1× / 2× na avaliação) e reatribui pelo LRU atual.

        Não apaga o carimbo de uso. Conta trocas e bytes desta reatribuição.
        """

        if hot_slots < 1 or hot_slots + self.warm_slots > self.n_experts:
            raise ValueError(
                f"hot_slots={hot_slots} não cabe com warm_slots={self.warm_slots} "
                f"e E={self.n_experts}"
            )
        self.hot_slots = hot_slots
        return self._commit(self._targets())

    def sync_physical(self, vectors: list[list[torch.Tensor]]) -> PhysicalExpertStore:
        """Grava o home de cada expert a partir do vetor achatado dos parâmetros."""

        self._check_vectors(vectors)
        store = PhysicalExpertStore(self.work_dir)
        for layer in range(self.n_layers):
            for expert in range(self.n_experts):
                store.place(layer, expert, int(self.levels[layer, expert].item()), vectors[layer][expert])
        self._store = store
        return store

    @property
    def physical_store(self) -> PhysicalExpertStore | None:
        return self._store

    def _counts(self, dispatched: torch.Tensor) -> torch.Tensor:
        if dispatched.shape != (self.n_layers, self.n_experts):
            raise ValueError(
                f"dispatched tem shape {tuple(dispatched.shape)}; "
                f"esperado {(self.n_layers, self.n_experts)}"
            )
        counts = dispatched.detach().to(device="cpu", dtype=torch.long)
        if torch.any(counts < 0):
            raise ValueError("dispatched não pode ter contagem negativa")
        return counts

    def _targets(self) -> torch.Tensor:
        levels = torch.full((self.n_layers, self.n_experts), COLD, dtype=torch.long)
        warm_end = self.hot_slots + self.warm_slots
        for layer in range(self.n_layers):
            order = self._rank(layer)
            levels[layer, order[: self.hot_slots]] = HOT
            levels[layer, order[self.hot_slots : warm_end]] = WARM
        return levels

    def _rank(self, layer: int) -> list[int]:
        last = self.last_used[layer].tolist()
        count = self.last_count[layer].tolist()
        return sorted(range(self.n_experts), key=lambda expert: (-last[expert], -count[expert], expert))

    def _commit(self, new_levels: torch.Tensor) -> StepStats:
        old = self.levels
        changed = new_levels != old
        distance = (new_levels - old).abs()
        swaps = int(changed.sum().item())
        bytes_moved = int(distance.sum().item()) * self.expert_nbytes
        promotions = int((changed & (new_levels < old)).sum().item())
        demotions = int((changed & (new_levels > old)).sum().item())
        self.levels = new_levels
        self.total_swaps += swaps
        self.total_bytes += bytes_moved
        return StepStats(
            swaps=swaps,
            bytes_moved=bytes_moved,
            promotions=promotions,
            demotions=demotions,
        )

    def _check_vectors(self, vectors: list[list[torch.Tensor]]) -> None:
        if len(vectors) != self.n_layers:
            raise ValueError(
                f"sync_physical esperava {self.n_layers} camadas, recebeu {len(vectors)}"
            )
        for layer, experts in enumerate(vectors):
            if len(experts) != self.n_experts:
                raise ValueError(
                    f"camada {layer} tem {len(experts)} experts; esperado {self.n_experts}"
                )
