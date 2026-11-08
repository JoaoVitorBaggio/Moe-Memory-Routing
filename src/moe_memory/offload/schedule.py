"""Olha só o que o LSTM já decidiu.

O plano da posição ``u`` usa o score cru de ``u - L``. Quando o token
``t`` já passou pelo roteador, o morno de ``t+1 .. t+L2`` e o quente de
``t+1 .. t+L1`` estão fixados, sem ler o token que ainda não chegou.
O passo 0 da janela é o próximo token a executar. Passos ``>= l1`` têm
score quente desconhecido: a máscara quente desses passos não se usa.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from moe_memory.lstm_ahead_router import TwoHorizonLstmRouter, TwoHorizonState
from moe_memory.tier_mask import TierPlan, TierState, plan_from_scores


@dataclass(frozen=True)
class LookaheadWindow:
    plan: TierPlan
    executed: int
    hot_steps: int


def lookahead_plan(router: TwoHorizonLstmRouter, state: TwoHorizonState | None) -> LookaheadWindow:
    """Janela já determinada. Não avança o estado do roteador."""

    batch = 1 if state is None else int(state.raw_hist.shape[0])
    device = _device(router, state)
    dtype = router.heads[0].weight.dtype
    initial = _initial_scores(router, batch, device, dtype)
    executed = 0 if state is None or state.tier is None else int(state.tier.step)
    warm = torch.empty(batch, router.l2, router.n_layers, router.n_experts, dtype=dtype, device=device)
    hot = torch.empty_like(warm)
    for delta in range(router.l2):
        position = executed + delta
        warm[:, delta] = _raw_at(router, state, position - router.l2, initial)
        if delta < router.l1:
            hot[:, delta] = _raw_at(router, state, position - router.l1, initial)
        else:
            hot[:, delta] = 0
    tier = _tier(router, state, batch, device)
    plan, _future = plan_from_scores(
        warm,
        hot,
        k1=router.k1,
        k2=router.k2,
        b2=router.b2,
        refresh_n1=router.refresh_n1,
        refresh_n2=router.refresh_n2,
        l1=router.l1,
        l2=router.l2,
        state=tier,
    )
    return LookaheadWindow(plan=plan, executed=executed, hot_steps=router.l1)


def _initial_scores(router: TwoHorizonLstmRouter, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    zeros = torch.zeros(batch, 1, router.hidden_size, device=device, dtype=dtype)
    parts = [head(zeros) for head in router.heads]
    return torch.stack(parts, dim=2).squeeze(1)


def _raw_at(
    router: TwoHorizonLstmRouter,
    state: TwoHorizonState | None,
    index: int,
    initial: torch.Tensor,
) -> torch.Tensor:
    if index < 0:
        return initial
    if state is None or state.tier is None:
        raise RuntimeError(f"score cru {index} pedido antes de qualquer token")
    hist = state.raw_hist
    executed = int(state.tier.step)
    oldest = executed - hist.shape[1]
    if index < oldest or index >= executed:
        raise RuntimeError(f"score cru {index} fora do histórico {oldest}..{executed - 1}")
    return hist[:, index - oldest]


def _tier(router: TwoHorizonLstmRouter, state: TwoHorizonState | None, batch: int, device: torch.device) -> TierState | None:
    if state is None or state.tier is None:
        if state is None:
            return None
        return TierState(
            warm=torch.zeros(batch, router.n_layers, router.n_experts, dtype=torch.bool, device=device),
            hot=torch.zeros(batch, router.n_layers, router.n_experts, dtype=torch.bool, device=device),
            step=0,
        )
    return state.tier


def _device(router: TwoHorizonLstmRouter, state: TwoHorizonState | None) -> torch.device:
    if state is not None:
        return state.raw_hist.device
    return router.heads[0].weight.device
