"""LSTM de dois horizontes, em dimensão minúscula. Sem Qwen3 e sem peso baixado."""

import pytest
import torch
from torch import nn

from moe_memory.lstm_ahead_router import HEAD_INIT_STD, TwoHorizonLstmRouter


def _router(**overrides: int | None) -> TwoHorizonLstmRouter:
    kwargs = dict(
        vocab_size=16,
        n_layers=2,
        n_experts=6,
        l1=2,
        l2=4,
        k1=2,
        k2=3,
        b2=1,
        embed_dim=8,
        hidden_size=8,
        refresh_n1=None,
        refresh_n2=None,
    )
    kwargs.update(overrides)
    return TwoHorizonLstmRouter(**kwargs)


def test_one_trunk_and_lookahead_does_not_add_parameters():
    torch.manual_seed(0)
    short = _router(l1=1, l2=2)
    long = _router(l1=3, l2=9)
    assert sum(parameter.numel() for parameter in short.parameters()) == sum(
        parameter.numel() for parameter in long.parameters()
    )
    lstms = [module for module in short.modules() if isinstance(module, nn.LSTM)]
    assert len(lstms) == 1
    assert len(short.heads) == short.n_layers
    assert 0 < HEAD_INIT_STD < 0.02


def test_raw_scores_shift_by_each_horizon_and_start_near_uniform():
    torch.manual_seed(0)
    router = _router()
    tokens = torch.randint(0, 16, (2, 9))
    decision, state = router(tokens)
    assert decision.raw_scores.shape == (2, 9, 2, 6)
    assert decision.hot_scores.shape == decision.raw_scores.shape
    assert decision.warm_scores.shape == decision.raw_scores.shape
    assert torch.equal(decision.hot_scores[:, router.l1 :], decision.raw_scores[:, : 9 - router.l1])
    assert torch.equal(decision.warm_scores[:, router.l2 :], decision.raw_scores[:, : 9 - router.l2])
    assert torch.equal(decision.hot_scores[:, : router.l1], torch.zeros_like(decision.hot_scores[:, : router.l1]))
    assert torch.equal(
        decision.warm_scores[:, : router.l2],
        torch.zeros_like(decision.warm_scores[:, : router.l2]),
    )
    probs = decision.raw_scores.softmax(dim=-1)
    uniform = torch.full_like(probs, 1.0 / router.n_experts)
    assert torch.allclose(probs, uniform, atol=0.05)
    assert state.h.shape == (1, 2, 8)
    assert state.raw_hist.shape[1] == max(router.l1, router.l2)
    assert state.tier is None


def test_decision_uses_only_tokens_up_to_t_for_position_t_plus_l():
    torch.manual_seed(0)
    router = _router(l1=2, l2=4)
    tokens = torch.randint(0, 16, (2, 10))
    seen = 3
    base, _ = router(tokens)
    base_plan, _ = router.plan(tokens)
    later = tokens.clone()
    later[:, seen + 1 :] = (later[:, seen + 1 :] + 3) % 16
    changed, _ = router(later)
    changed_plan, _ = router.plan(later)
    hot_cut = seen + router.l1 + 1
    warm_cut = seen + router.l2 + 1
    assert torch.equal(base.hot_scores[:, :hot_cut], changed.hot_scores[:, :hot_cut])
    assert torch.equal(base.warm_scores[:, :warm_cut], changed.warm_scores[:, :warm_cut])
    assert torch.equal(base_plan.hot_mask[:, :hot_cut], changed_plan.hot_mask[:, :hot_cut])
    assert torch.equal(base_plan.warm_mask[:, :warm_cut], changed_plan.warm_mask[:, :warm_cut])

    used = tokens.clone()
    used[:, seen] = (used[:, seen] + 5) % 16
    assert not torch.equal(used[:, seen], tokens[:, seen])
    moved, _ = router(used)
    assert seen + router.l2 < tokens.shape[1]
    assert not torch.equal(moved.warm_scores[:, seen + router.l2], base.warm_scores[:, seen + router.l2])
    assert not torch.equal(moved.hot_scores[:, seen + router.l1], base.hot_scores[:, seen + router.l1])


def test_token_steps_match_the_full_plan():
    torch.manual_seed(1)
    router = _router(l1=1, l2=3, b2=2, k1=2, k2=4)
    tokens = torch.randint(0, 16, (2, 7))
    full, _ = router.plan(tokens)
    state = None
    warm = []
    hot = []
    weights = []
    for step in range(tokens.shape[1]):
        piece, state = router.plan(tokens[:, step : step + 1], state)
        warm.append(piece.warm_mask)
        hot.append(piece.hot_mask)
        weights.append(piece.soft_weights)
    assert torch.equal(torch.cat(warm, dim=1), full.warm_mask)
    assert torch.equal(torch.cat(hot, dim=1), full.hot_mask)
    assert torch.allclose(torch.cat(weights, dim=1), full.soft_weights, atol=1e-5)
    assert state is not None and state.tier is not None
    assert state.tier.step == tokens.shape[1]


def test_soft_weights_are_softmax_over_the_hot_set():
    torch.manual_seed(2)
    router = _router()
    tokens = torch.randint(0, 16, (2, 8))
    decision, _ = router(tokens)
    plan, _ = router.plan(tokens)
    assert not bool((plan.hot_mask & ~plan.warm_mask).any().item())
    gathered = torch.gather(decision.hot_scores, -1, plan.hot_index)
    masked = gathered.masked_fill(~plan.hot_valid, torch.finfo(gathered.dtype).min)
    expected = torch.softmax(masked.float(), dim=-1).to(plan.soft_weights.dtype)
    expected = expected * plan.hot_valid.to(dtype=expected.dtype)
    assert torch.allclose(plan.soft_weights, expected, atol=1e-5)
    active = plan.hot_valid.any(dim=-1)
    assert torch.allclose(plan.soft_weights.sum(dim=-1)[active], torch.ones_like(plan.soft_weights.sum(-1)[active]))


def test_constructor_rejects_a_bad_horizon_or_budget():
    with pytest.raises(ValueError):
        _router(l1=0)
    with pytest.raises(ValueError):
        _router(l2=1)
    with pytest.raises(ValueError):
        _router(k1=4, k2=3)
    with pytest.raises(ValueError):
        _router(b2=0)
    with pytest.raises(ValueError):
        _router(refresh_n2=0)
