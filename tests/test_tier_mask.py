"""Regra determinística do morno e do quente. Sem modelo e sem I/O."""

import pytest
import torch

from moe_memory.levels import COLD, HOT, WARM
from moe_memory.tier_mask import (
    b2_from_token_time,
    k2_from_ram_budget,
    plan_from_scores,
    swiglu_expert_nbytes,
)


def _plan(warm, hot, **overrides):
    warm_scores = warm if warm.ndim == 4 else warm.view(1, warm.shape[0], 1, warm.shape[-1])
    hot_scores = hot if hot.ndim == 4 else hot.view_as(warm_scores) if hot.shape == warm.shape else hot
    if hot_scores.ndim != 4:
        hot_scores = hot_scores.view_as(warm_scores)
    kwargs = dict(k1=1, k2=2, b2=1, refresh_n1=None, refresh_n2=None, l1=1, l2=2)
    kwargs.update(overrides)
    plan, state = plan_from_scores(warm_scores, hot_scores, **kwargs)
    return plan, state


def test_fill_then_replace_lowest_score_and_hot_stays_inside_warm():
    warm = torch.tensor(
        [
            [0.1, 0.4, 0.2, 0.3],
            [0.1, 0.4, 0.9, 0.3],
            [0.8, 0.1, 0.2, 0.05],
            [0.8, 0.1, 0.2, 0.70],
        ]
    )
    # No quente, o expert de fora do morno tem score enorme e não pode entrar.
    hot = torch.tensor(
        [
            [0.0, 5.0, 9.0, 0.0],
            [0.0, 0.2, 0.1, 8.0],
            [0.1, 9.0, 0.4, 7.0],
            [0.2, 9.0, 8.0, 0.3],
        ]
    )
    plan, _ = _plan(warm, hot, k1=1, k2=2, b2=1)
    warm_ids = [plan.resident_ids(0, step, WARM) for step in range(4)]
    hot_ids = [plan.resident_ids(0, step, HOT) for step in range(4)]
    assert warm_ids == [((0, 1),), ((0, 1), (0, 2)), ((0, 0), (0, 2)), ((0, 0), (0, 3))]
    assert hot_ids == [((0, 1),), ((0, 1),), ((0, 2),), ((0, 3),)]
    assert torch.equal(plan.n_new_warm[0, :, 0], torch.tensor([1, 1, 1, 1]))
    assert int(plan.n_new_warm.sum(dim=-1).max().item()) <= 1
    assert not bool((plan.hot_mask & ~plan.warm_mask).any().item())
    # Passo 2: o quente fica com o expert 2, que perde no score morno para o 0.
    assert plan.resident_ids(0, 2, HOT) == ((0, 2),)
    entered = plan.prefetch(0, 2)
    assert entered.warm == ((0, 0),)
    assert (0, 1) in entered.warm_release


def test_global_budget_is_shared_across_layers():
    # Camada 0 quer o expert 3 (0.9). Camada 1 quer o expert 0 (0.5).
    # b2=1 deixa entrar um só por token, o de maior score.
    layer0 = torch.tensor([0.1, 0.2, 0.3, 0.9])
    layer1 = torch.tensor([0.5, 0.4, 0.0, 0.0])
    warm = torch.stack([layer0, layer1], dim=0).unsqueeze(0).unsqueeze(0).repeat(1, 5, 1, 1)
    plan, _ = plan_from_scores(warm, warm.clone(), k1=1, k2=2, b2=1)
    totals = plan.n_new_warm.sum(dim=-1)[0].tolist()
    assert totals == [1, 1, 1, 1, 0]
    assert max(totals) <= 1
    assert plan.resident_ids(0, 0, WARM) == ((0, 3),)
    assert int(plan.warm_mask.sum(dim=-1).max().item()) <= 2
    assert not bool((plan.hot_mask & ~plan.warm_mask).any().item())


def test_tie_keeps_the_lower_index():
    warm = torch.tensor([[0.5, 0.5, 0.0]]).view(1, 1, 1, 3)
    plan, _ = plan_from_scores(warm, warm.clone(), k1=1, k2=1, b2=1)
    assert plan.resident_ids(0, 0, WARM) == ((0, 0),)


def test_same_scores_make_the_same_masks_and_chunks_match():
    torch.manual_seed(0)
    scores = torch.randn(2, 6, 3, 5)
    first, _ = plan_from_scores(scores, scores + 0.1, k1=2, k2=3, b2=2, l1=1, l2=3)
    second, _ = plan_from_scores(scores.clone(), (scores + 0.1).clone(), k1=2, k2=3, b2=2, l1=1, l2=3)
    assert torch.equal(first.warm_mask, second.warm_mask)
    assert torch.equal(first.hot_mask, second.hot_mask)
    assert torch.equal(first.hot_index, second.hot_index)
    assert torch.equal(first.n_new_warm, second.n_new_warm)
    assert torch.equal(first.entered_warm.sum(-1), first.n_new_warm)
    assert int(first.n_new_warm.sum(-1).max().item()) <= 2
    assert int(first.warm_mask.sum(-1).max().item()) <= 3
    assert torch.all(first.hot_mask.sum(-1) <= first.warm_mask.sum(-1))
    assert not bool((first.hot_mask & ~first.warm_mask).any().item())

    state = None
    parts = []
    for step in range(scores.shape[1]):
        piece, state = plan_from_scores(
            scores[:, step : step + 1],
            (scores + 0.1)[:, step : step + 1],
            k1=2,
            k2=3,
            b2=2,
            state=state,
        )
        parts.append(piece.warm_mask)
    assert torch.equal(torch.cat(parts, dim=1), first.warm_mask)


def test_periodic_refresh_holds_the_set_between_boundaries():
    warm = torch.zeros(1, 6, 1, 4)
    warm[0, 0, 0] = torch.tensor([0.9, 0.8, 0.1, 0.0])
    warm[0, 1:, 0] = torch.tensor([0.1, 0.0, 0.9, 0.8])
    hot = warm.clone()
    plan, _ = plan_from_scores(warm, hot, k1=1, k2=2, b2=2, refresh_n2=3, refresh_n1=3)
    assert plan.resident_ids(0, 0, WARM) == ((0, 0), (0, 1))
    assert plan.resident_ids(0, 1, WARM) == plan.resident_ids(0, 0, WARM)
    assert plan.resident_ids(0, 2, WARM) == plan.resident_ids(0, 0, WARM)
    assert torch.equal(plan.n_new_warm[0, 1], torch.zeros(1, dtype=torch.long))
    assert torch.equal(plan.n_new_warm[0, 2], torch.zeros(1, dtype=torch.long))
    assert plan.resident_ids(0, 3, WARM) == ((0, 2), (0, 3))
    assert int(plan.n_new_warm[0, 3].sum().item()) <= 2


def test_hot_refresh_holds_while_the_warm_set_is_stable():
    base = torch.tensor([0.9, 0.8, 0.1, 0.0])
    warm = base.view(1, 1, 1, 4).repeat(1, 4, 1, 1)
    hot = torch.stack(
        [
            torch.tensor([0.2, 0.9, 0.0, 0.0]),
            torch.tensor([0.9, 0.2, 0.0, 0.0]),
            torch.tensor([0.9, 0.2, 0.0, 0.0]),
            torch.tensor([0.2, 0.9, 0.0, 0.0]),
        ]
    ).view(1, 4, 1, 4)
    plan, _ = plan_from_scores(warm, hot, k1=1, k2=2, b2=2, refresh_n1=2)
    assert [plan.resident_ids(0, step, HOT) for step in range(4)] == [
        ((0, 1),),
        ((0, 1),),
        ((0, 0),),
        ((0, 0),),
    ]
    assert int(plan.n_new_hot[0, 1].sum().item()) == 0
    assert plan.prefetch(0, 1).hot == ()


def test_budget_helpers_follow_the_measurement_arithmetic():
    expert = swiglu_expert_nbytes(2048, 768, 2)
    assert expert == 9_437_184
    assert 48 * 128 * expert == 57_982_058_496
    ram_12 = 12 * 1024**3
    ram_13 = 13 * 1024**3
    assert k2_from_ram_budget(ram_12, expert, 48) == (ram_12 // expert) // 48
    assert k2_from_ram_budget(ram_13, expert, 48) == (ram_13 // expert) // 48
    # A nota registra ~1.400 corpos na faixa de 12–13 GB. O helper não arredonda.
    assert 1300 <= ram_12 // expert <= 1500
    assert 1300 <= ram_13 // expert <= 1500
    assert b2_from_token_time(0.068, 0.013) == 5
    with pytest.raises(ValueError):
        k2_from_ram_budget(-1, expert, 48)
    with pytest.raises(ValueError):
        plan_from_scores(torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 1, 2), k1=2, k2=1, b2=1)


def test_cold_is_not_a_prefetch_target():
    scores = torch.zeros(1, 1, 1, 2)
    scores[0, 0, 0, 1] = 1
    plan, _ = plan_from_scores(scores, scores, k1=1, k2=1, b2=1)
    with pytest.raises(ValueError):
        plan.resident_ids(0, 0, COLD)
