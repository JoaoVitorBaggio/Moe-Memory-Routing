"""Mede a máscara nesta CPU. Não é tok/s e não é a GTX 1060.

Compara o planejador atual com a cópia congelada da regra anterior,
nos mesmos tensores. Se as máscaras divergirem, o script para.
"""

from __future__ import annotations

import os
import platform
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch

from moe_memory.lstm_ahead_router import TwoHorizonLstmRouter, TwoHorizonState
from moe_memory.offload.schedule import lookahead_plan
from moe_memory.tier_mask import TierState, plan_from_scores
from tests import tier_plan_reference as old


def _median_range(samples: list[float]) -> tuple[float, float, float]:
    ordered = sorted(samples)
    return ordered[len(ordered) // 2], ordered[0], ordered[-1]


def _time(fn, warmup: int, repeats: int) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return _median_range(samples)


def _peak(fn) -> int:
    tracemalloc.start()
    fn()
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def _masks_equal(new_plan, ref_plan) -> bool:
    return (
        torch.equal(new_plan.warm_mask, ref_plan.warm_mask)
        and torch.equal(new_plan.hot_mask, ref_plan.hot_mask)
        and torch.equal(new_plan.hot_index, ref_plan.hot_index)
        and torch.equal(new_plan.hot_valid, ref_plan.hot_valid)
        and torch.equal(new_plan.entered_warm, ref_plan.entered_warm)
        and torch.equal(new_plan.entered_hot, ref_plan.entered_hot)
        and torch.equal(new_plan.left_warm, ref_plan.left_warm)
        and torch.equal(new_plan.left_hot, ref_plan.left_hot)
    )


def _reference_lookahead(router, state):
    from moe_memory.offload.schedule import _initial_scores, _raw_at, _tier

    batch = int(state.raw_hist.shape[0])
    device = state.raw_hist.device
    dtype = router.heads[0].weight.dtype
    initial = _initial_scores(router, batch, device, dtype)
    executed = int(state.tier.step)
    warm = torch.empty(batch, router.l2, router.n_layers, router.n_experts, dtype=dtype, device=device)
    hot = torch.empty_like(warm)
    for delta in range(router.l2):
        position = executed + delta
        warm[:, delta] = _raw_at(router, state, position - router.l2, initial)
        if delta < router.l1:
            hot[:, delta] = _raw_at(router, state, position - router.l1, initial)
        else:
            hot[:, delta] = 0
    plan, _future = old.plan_from_scores(
        warm,
        hot,
        k1=router.k1,
        k2=router.k2,
        b2=router.b2,
        refresh_n1=router.refresh_n1,
        refresh_n2=router.refresh_n2,
        l1=router.l1,
        l2=router.l2,
        state=state.tier,
    )
    return plan


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main() -> None:
    torch.manual_seed(0)
    batch, steps, layers, experts = 1, 3, 48, 128
    k1, k2, b2, l1, l2 = 4, 6, 5, 1, 3
    warm_scores = torch.randn(batch, steps, layers, experts)
    hot_scores = warm_scores.clone()
    hot_scores[:, l1:] = 0
    state = TierState(
        warm=torch.zeros(batch, layers, experts, dtype=torch.bool),
        hot=torch.zeros(batch, layers, experts, dtype=torch.bool),
        step=10,
    )
    state.warm[:, :, :k2] = True
    state.hot[:, :, :k1] = True
    new_plan, _ = plan_from_scores(
        warm_scores, hot_scores, k1=k1, k2=k2, b2=b2, l1=l1, l2=l2, state=state
    )
    ref_plan, _ = old.plan_from_scores(
        warm_scores, hot_scores, k1=k1, k2=k2, b2=b2, l1=l1, l2=l2, state=state
    )
    if not _masks_equal(new_plan, ref_plan):
        raise SystemExit("plan_from_scores divergiu do oráculo; medição cancelada")

    router = TwoHorizonLstmRouter(
        vocab_size=32,
        n_layers=layers,
        n_experts=experts,
        l1=l1,
        l2=l2,
        k1=k1,
        k2=k2,
        b2=b2,
        embed_dim=8,
        hidden_size=8,
    )
    hist = torch.randn(batch, l2, layers, experts)
    tier = TierState(
        warm=state.warm.clone(),
        hot=state.hot.clone(),
        step=10,
    )
    ahead = TwoHorizonState(
        h=torch.zeros(1, batch, 8),
        c=torch.zeros(1, batch, 8),
        tier=tier,
        raw_hist=hist,
    )
    produced = lookahead_plan(router, ahead)
    reference = _reference_lookahead(router, ahead)
    if not _masks_equal(produced.plan, reference):
        raise SystemExit("lookahead_plan divergiu do oráculo; medição cancelada")

    warmup, repeats = 5, 21
    new_plan_s = _time(
        lambda: plan_from_scores(warm_scores, hot_scores, k1=k1, k2=k2, b2=b2, l1=l1, l2=l2, state=state),
        warmup,
        repeats,
    )
    old_plan_s = _time(
        lambda: old.plan_from_scores(warm_scores, hot_scores, k1=k1, k2=k2, b2=b2, l1=l1, l2=l2, state=state),
        warmup,
        repeats,
    )
    new_look_s = _time(lambda: lookahead_plan(router, ahead), warmup, repeats)
    old_look_s = _time(lambda: _reference_lookahead(router, ahead), warmup, repeats)

    def _fmt(sample: tuple[float, float, float]) -> str:
        median, low, high = sample
        return f"mediana {median * 1e3:.2f} ms  faixa {low * 1e3:.2f}–{high * 1e3:.2f} ms"

    print("ambiente", platform.platform())
    print("cpu", _cpu_model())
    print("cpus", os.cpu_count(), "torch", torch.__version__, "threads", torch.get_num_threads())
    print("formato", f"{batch}x{steps}x{layers}x{experts}", f"k1={k1} k2={k2} b2={b2} l1={l1} l2={l2}")
    print("warmup", warmup, "repeticoes", repeats)
    print("plan_from_scores novo", _fmt(new_plan_s))
    print("plan_from_scores antigo", _fmt(old_plan_s))
    print("plan ratio antigo/novo", f"{old_plan_s[0] / new_plan_s[0]:.2f}")
    print("lookahead_plan novo", _fmt(new_look_s))
    print("lookahead_plan antigo", _fmt(old_look_s))
    print("lookahead ratio antigo/novo", f"{old_look_s[0] / new_look_s[0]:.2f}")
    print(
        "tracemalloc pico plan novo/antigo bytes",
        _peak(lambda: plan_from_scores(warm_scores, hot_scores, k1=k1, k2=k2, b2=b2, l1=l1, l2=l2, state=state)),
        _peak(lambda: old.plan_from_scores(warm_scores, hot_scores, k1=k1, k2=k2, b2=b2, l1=l1, l2=l2, state=state)),
    )
    print("isto não é tokens/s e não prevê a GTX 1060")


if __name__ == "__main__":
    main()
