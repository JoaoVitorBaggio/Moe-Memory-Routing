"""Decode batch 1. O expert que não chegou segura o token e entra no stall.

``lookahead`` pede ao SSD o morno de até ``L2`` e à VRAM o quente de até
``L1``, com os scores que o LSTM já emitiu. ``on-demand`` só pede na
hora do token. Os dois usam o mesmo forward e a mesma máscara.
"""

from __future__ import annotations

import resource
from pathlib import Path

import torch

from moe_memory.levels import HOT, WARM
from moe_memory.lstm_ahead_router import TwoHorizonLstmRouter, TwoHorizonState
from moe_memory.offload.hot import HotTier
from moe_memory.offload.qwen import KvCache, QwenWeights, forward_token, load_resident, load_spec
from moe_memory.offload.schedule import lookahead_plan
from moe_memory.offload.stats import OffloadStats
from moe_memory.offload.store import ExpertStore
from moe_memory.offload.warm import WarmTier
from moe_memory.tier_mask import TierPlan


class OffloadRuntime:
    def __init__(
        self,
        store_dir: str | Path,
        router: TwoHorizonLstmRouter,
        *,
        lookahead: bool,
        warm_slots: int,
        hot_slack: int = 0,
        device: torch.device | None = None,
        n_warm_threads: int = 1,
        direct: bool = True,
    ) -> None:
        if warm_slots < 1:
            raise ValueError("warm_slots precisa ser >= 1")
        if hot_slack < 0:
            raise ValueError("hot_slack precisa ser >= 0")
        self.lookahead = lookahead
        self.router = router
        self.device = torch.device("cpu") if device is None else torch.device(device)
        self.stats = OffloadStats()
        store_dir = Path(store_dir)
        self.store = ExpertStore.open(store_dir / "experts.moestore", direct=direct)
        header = self.store.header
        if header.n_layers != router.n_layers or header.n_experts != router.n_experts:
            raise ValueError(
                f"store {header.n_layers}×{header.n_experts} não casa com o roteador "
                f"{router.n_layers}×{router.n_experts}"
            )
        need = router.k2 * router.n_layers
        if warm_slots < need:
            raise ValueError(
                f"o morno cabe {router.k2} experts por camada, {need} vagas, e o pool tem {warm_slots}"
            )
        self.spec = load_spec(store_dir / "config.json", header.dtype_code)
        if self.spec.n_layers != router.n_layers or self.spec.n_experts != router.n_experts:
            raise ValueError("config.json não casa com o roteador")
        if self.spec.hidden_size != header.hidden or self.spec.moe_intermediate != header.intermediate:
            raise ValueError("config.json não casa com o bloco do expert")
        self.weights: QwenWeights = load_resident(store_dir / "resident.bin", self.spec, self.device)
        self.warm = WarmTier(self.store, warm_slots, stats=self.stats, n_threads=n_warm_threads)
        hot_slots = router.n_layers * router.k1 + hot_slack
        self.hot = HotTier(
            self.warm,
            hot_slots,
            layout=self.store.layout,
            dtype=self.weights.embed.dtype,
            device=self.device,
            stats=self.stats,
        )
        self.hot_slots = hot_slots
        self.warm_slots = warm_slots
        self.state: TwoHorizonState | None = None
        self.cache = KvCache(router.n_layers)
        self._plan: TierPlan | None = None
        self.last_logits: torch.Tensor | None = None
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def __enter__(self) -> OffloadRuntime:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def decode(self, token_ids: list[int]) -> OffloadStats:
        if not token_ids:
            raise ValueError("decode precisa de ao menos um token")
        self.router.eval()
        started = _now()
        try:
            with torch.no_grad():
                self._prefetch_initial()
                for position, token in enumerate(token_ids):
                    self._step(position, int(token))
        finally:
            self.stats.tokens = len(token_ids)
            self.stats.elapsed_s = _now() - started
            self.stats.ram_peak_bytes = max(_rss_bytes(), self.warm.pool.nbytes + self.weights.nbytes())
            self.stats.vram_peak_bytes = _vram_bytes(self.device, self.hot.pool.nbytes)
        return self.stats

    def close(self) -> None:
        self.hot.close()
        self.warm.close()
        self.store.close()

    def _step(self, position: int, token: int) -> torch.Tensor:
        ids = torch.tensor([[token]], dtype=torch.long)
        plan, self.state = self.router.plan(ids, self.state)
        self._plan = plan
        self._install(plan)
        logits, executed = forward_token(
            self.weights,
            self.cache,
            token,
            position,
            lambda layer: self._experts(plan, layer),
        )
        self.stats.experts_executed += executed
        self.last_logits = logits.detach()
        return logits

    def _experts(self, plan: TierPlan, layer: int):
        calls = []
        slots = plan.hot_index.shape[-1]
        for slot in range(slots):
            if not bool(plan.hot_valid[0, 0, layer, slot].item()):
                continue
            expert = int(plan.hot_index[0, 0, layer, slot].item())
            key = (layer, expert)
            self.warm.ensure(key)
            self.hot.ensure(key)
            gate, up, down = self.hot.views(key)
            coeff = plan.soft_weights[0, 0, layer, slot]
            calls.append((gate, up, down, coeff))
        return calls

    def _prefetch_initial(self) -> None:
        if not self.lookahead:
            return
        window = lookahead_plan(self.router, None)
        self.hot.sync(_ranked(window, hot=True))
        sources = self.hot.pending_sources()
        ranked = _ranked(window, hot=False)
        ranked.extend((0, key) for key in sources)
        self.warm.sync(ranked)

    def _install(self, plan: TierPlan) -> None:
        current_hot = list(plan.resident_ids(0, 0, HOT))
        current_warm = list(plan.resident_ids(0, 0, WARM))
        future_hot: list[tuple[int, tuple[int, int]]] = []
        future_warm: list[tuple[int, tuple[int, int]]] = []
        if self.lookahead and self.state is not None:
            window = lookahead_plan(self.router, self.state)
            future_hot = _ranked(window, hot=True, step_bias=1)
            future_warm = _ranked(window, hot=False, step_bias=1)
        self.hot.sync([(0, key) for key in current_hot] + future_hot)
        sources = self.hot.pending_sources()
        self.warm.sync(
            [(0, key) for key in current_warm]
            + [(0, key) for key in sources]
            + future_warm
        )


def _ranked(window, *, hot: bool, step_bias: int = 0) -> list[tuple[int, tuple[int, int]]]:
    from moe_memory.levels import HOT as HOT_LEVEL
    from moe_memory.levels import WARM as WARM_LEVEL

    level = HOT_LEVEL if hot else WARM_LEVEL
    ranked: list[tuple[int, tuple[int, int]]] = []
    steps = window.plan.warm_mask.shape[1]
    limit = window.hot_steps if hot else steps
    for step in range(limit):
        for key in window.plan.resident_ids(0, step, level):
            ranked.append((step + step_bias, key))
    return ranked


def _now() -> float:
    import time

    return time.perf_counter()


def _rss_bytes() -> int:
    # Linux: ru_maxrss em KiB.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _vram_bytes(device: torch.device, hot_pool: int) -> int:
    if device.type != "cuda":
        return 0
    return max(int(torch.cuda.max_memory_allocated(device)), hot_pool)
