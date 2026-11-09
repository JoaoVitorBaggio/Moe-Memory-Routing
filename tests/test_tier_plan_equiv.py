"""A seleção nova tem de repetir a máscara antiga, não só parecer parecida."""

from __future__ import annotations

import pytest
import torch

from moe_memory.lstm_ahead_router import TwoHorizonLstmRouter, TwoHorizonState
from moe_memory.offload.schedule import lookahead_plan
from moe_memory.tier_mask import TierState, plan_from_scores
from tests import tier_plan_reference as old


def _same_plan(new, ref) -> None:
    names = (
        "warm_mask",
        "hot_mask",
        "entered_warm",
        "entered_hot",
        "left_warm",
        "left_hot",
        "hot_index",
        "hot_valid",
    )
    for name in names:
        assert torch.equal(getattr(new, name), getattr(ref, name)), name
    assert torch.equal(new.n_new_warm, ref.n_new_warm)
    assert torch.equal(new.n_new_hot, ref.n_new_hot)
    soft_ok = (new.soft_weights == ref.soft_weights) | (
        torch.isnan(new.soft_weights) & torch.isnan(ref.soft_weights)
    )
    assert bool(soft_ok.all()), "soft_weights"
    assert (new.k1, new.k2, new.b2, new.refresh_n1, new.refresh_n2) == (
        ref.k1,
        ref.k2,
        ref.b2,
        ref.refresh_n1,
        ref.refresh_n2,
    )


def _same_state(new, ref) -> None:
    assert torch.equal(new.warm, ref.warm)
    assert torch.equal(new.hot, ref.hot)
    assert new.step == ref.step


def _run(warm, hot, **kwargs):
    incoming = kwargs.get("state")
    snapshot = None
    if incoming is not None:
        snapshot = (incoming.warm.clone(), incoming.hot.clone(), incoming.step)
        kwargs = {
            **kwargs,
            "state": TierState(warm=snapshot[0].clone(), hot=snapshot[1].clone(), step=snapshot[2]),
        }
    new_plan, new_state = plan_from_scores(warm, hot, **kwargs)
    ref_kwargs = dict(kwargs)
    if snapshot is not None:
        ref_kwargs["state"] = TierState(warm=snapshot[0].clone(), hot=snapshot[1].clone(), step=snapshot[2])
    ref_plan, ref_out = old.plan_from_scores(warm, hot, **ref_kwargs)
    _same_plan(new_plan, ref_plan)
    _same_state(new_state, ref_out)
    if snapshot is not None:
        assert torch.equal(incoming.warm, snapshot[0])
        assert torch.equal(incoming.hot, snapshot[1])
        assert incoming.step == snapshot[2]
        assert torch.equal(kwargs["state"].warm, snapshot[0])
        assert torch.equal(kwargs["state"].hot, snapshot[1])


def _scores(gen, shape, dtype, kind):
    if kind == "tie":
        base = torch.randint(0, 3, shape, generator=gen, dtype=torch.int64).to(dtype)
        return base
    if kind == "neg":
        return torch.randn(shape, generator=gen).to(dtype) - 2
    return torch.randn(shape, generator=gen).to(dtype)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("kind", ["rand", "tie", "neg"])
def test_sorteio_contra_oraculo(seed, kind):
    gen = torch.Generator().manual_seed(seed + (0 if kind == "rand" else 100 if kind == "tie" else 200))
    batch, seq, layers, experts = 2, 4, 3, 11
    dtype = torch.float32
    warm = _scores(gen, (batch, seq, layers, experts), dtype, kind)
    hot = _scores(gen, (batch, seq, layers, experts), dtype, kind)
    state_warm = torch.zeros(batch, layers, experts, dtype=torch.bool)
    state_hot = torch.zeros(batch, layers, experts, dtype=torch.bool)
    if seed % 3 == 1:
        state_warm[:, :, :2] = True
        state_hot[:, :, :1] = True
    if seed % 3 == 2:
        state_warm[:] = True
        state_hot[:, :, :2] = True
    state = TierState(warm=state_warm, hot=state_hot, step=seed)
    _run(
        warm,
        hot,
        k1=1 if seed % 2 == 0 else 2,
        k2=1 if seed == 0 else (experts if seed == 5 else 4),
        b2=(0, 1, 5, 9)[seed % 4],
        refresh_n1=None if seed % 2 == 0 else 2,
        refresh_n2=None if seed < 3 else 3,
        state=state,
    )


def test_continuacao_de_chunk_igual_a_sequencia():
    gen = torch.Generator().manual_seed(7)
    warm = torch.randn(1, 4, 2, 8, generator=gen)
    hot = torch.randn(1, 4, 2, 8, generator=gen)
    full, _ = plan_from_scores(warm, hot, k1=2, k2=3, b2=2, refresh_n1=2, refresh_n2=2)
    first, state = plan_from_scores(warm[:, :2], hot[:, :2], k1=2, k2=3, b2=2, refresh_n1=2, refresh_n2=2)
    second, _ = plan_from_scores(
        warm[:, 2:],
        hot[:, 2:],
        k1=2,
        k2=3,
        b2=2,
        refresh_n1=2,
        refresh_n2=2,
        state=state,
    )
    ref, _ = old.plan_from_scores(warm, hot, k1=2, k2=3, b2=2, refresh_n1=2, refresh_n2=2)
    _same_plan(full, ref)
    assert torch.equal(first.warm_mask, full.warm_mask[:, :2])
    assert torch.equal(second.hot_index, full.hot_index[:, 2:])
    assert torch.equal(second.entered_warm, full.entered_warm[:, 2:])


def test_estado_de_entrada_nao_muda():
    warm = torch.randn(1, 2, 2, 6)
    hot = warm.clone()
    state = TierState(
        warm=torch.zeros(1, 2, 6, dtype=torch.bool),
        hot=torch.zeros(1, 2, 6, dtype=torch.bool),
        step=4,
    )
    before_w = state.warm.clone()
    before_h = state.hot.clone()
    plan_from_scores(warm, hot, k1=1, k2=2, b2=2, state=state)
    assert torch.equal(state.warm, before_w)
    assert torch.equal(state.hot, before_h)
    assert state.step == 4


def test_gradiente_do_peso_mole_igual_ao_oraculo():
    warm = torch.randn(1, 2, 2, 6)
    hot = torch.randn(1, 2, 2, 6, requires_grad=True)
    new, _ = plan_from_scores(warm, hot, k1=2, k2=3, b2=1)
    ref, _ = old.plan_from_scores(warm, hot, k1=2, k2=3, b2=1)
    _same_plan(new, ref)
    new.soft_weights.sum().backward(retain_graph=True)
    grad_new = hot.grad.detach().clone()
    hot.grad = None
    ref.soft_weights.sum().backward()
    assert torch.equal(grad_new, hot.grad)


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_nao_finito_cai_no_laco_antigo(value):
    scores = torch.zeros(1, 2, 2, 5)
    scores[0, 0, 1, 3] = value
    scores[0, 1, 0, 0] = value
    state = TierState(
        warm=torch.tensor([[[1, 1, 0, 0, 0], [1, 0, 0, 0, 0]]], dtype=torch.bool),
        hot=torch.tensor([[[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]]], dtype=torch.bool),
        step=1,
    )
    _run(scores, scores.clone(), k1=1, k2=2, b2=2, state=state)


def test_formato_do_bench_48_por_128():
    gen = torch.Generator().manual_seed(0)
    warm = torch.randn(1, 3, 48, 128, generator=gen)
    hot = warm.clone()
    hot[:, 1:] = 0
    state = TierState(
        warm=torch.zeros(1, 48, 128, dtype=torch.bool),
        hot=torch.zeros(1, 48, 128, dtype=torch.bool),
        step=10,
    )
    state.warm[:, :, :6] = True
    state.hot[:, :, :4] = True
    _run(warm, hot, k1=4, k2=6, b2=5, l1=1, l2=3, state=state)


def test_lookahead_plan_repete_o_oraculo():
    from moe_memory.offload.schedule import _initial_scores, _raw_at

    torch.manual_seed(0)
    router = TwoHorizonLstmRouter(
        vocab_size=8,
        n_layers=2,
        n_experts=7,
        l1=1,
        l2=3,
        k1=1,
        k2=3,
        b2=2,
        embed_dim=4,
        hidden_size=4,
    )
    hist = torch.randn(1, 3, 2, 7)
    tier = TierState(
        warm=torch.zeros(1, 2, 7, dtype=torch.bool),
        hot=torch.zeros(1, 2, 7, dtype=torch.bool),
        step=10,
    )
    tier.warm[:, :, :3] = True
    tier.hot[:, :, :1] = True
    state = TwoHorizonState(h=torch.zeros(1, 1, 4), c=torch.zeros(1, 1, 4), tier=tier, raw_hist=hist)
    produced = lookahead_plan(router, state)
    dtype = router.heads[0].weight.dtype
    initial = _initial_scores(router, 1, hist.device, dtype)
    warm = torch.empty(1, 3, 2, 7, dtype=dtype)
    hot = torch.empty_like(warm)
    for delta in range(router.l2):
        position = int(tier.step) + delta
        warm[:, delta] = _raw_at(router, state, position - router.l2, initial)
        if delta < router.l1:
            hot[:, delta] = _raw_at(router, state, position - router.l1, initial)
        else:
            hot[:, delta] = 0
    ref, _ = old.plan_from_scores(
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
    _same_plan(produced.plan, ref)


def test_prefetch_do_plano_igual():
    from moe_memory.tier_mask import HOT, WARM

    warm = torch.randn(1, 3, 2, 6)
    hot = warm.clone()
    new, _ = plan_from_scores(warm, hot, k1=1, k2=2, b2=2, l1=1, l2=3)
    ref, _ = old.plan_from_scores(warm, hot, k1=1, k2=2, b2=2, l1=1, l2=3)
    for step in range(3):
        assert new.prefetch(0, step) == ref.prefetch(0, step)
        assert new.resident_ids(0, step, WARM) == ref.resident_ids(0, step, WARM)
        assert new.resident_ids(0, step, HOT) == ref.resident_ids(0, step, HOT)


def test_vaga_vazia_ganha_da_troca_de_margem_maior():
    scores = torch.zeros(1, 1, 2, 4)
    scores[0, 0, 0, 2] = 0.1
    scores[0, 0, 1, 3] = 100.0
    scores[0, 0, 1, 0] = -50.0
    warm = torch.zeros(1, 2, 4, dtype=torch.bool)
    warm[0, 0, 0] = True
    warm[0, 1, :3] = True
    state = TierState(warm=warm, hot=warm.clone(), step=0)
    _run(scores, scores.clone(), k1=1, k2=3, b2=1, state=state)


def test_empate_escolhe_menor_camada_e_pior_no_maior_indice():
    scores = torch.zeros(1, 1, 2, 4)
    scores[0, 0, :, 1] = 1
    scores[0, 0, :, 2] = 1
    warm = torch.zeros(1, 2, 4, dtype=torch.bool)
    warm[0, :, 0] = True
    warm[0, :, 3] = True
    state = TierState(warm=warm, hot=warm.clone(), step=3)
    _run(scores, scores.clone(), k1=1, k2=2, b2=1, state=state)


def test_itens_do_batch_com_conjuntos_diferentes():
    scores = torch.arange(2 * 2 * 3 * 6, dtype=torch.float32).reshape(2, 2, 3, 6)
    warm = torch.zeros(2, 3, 6, dtype=torch.bool)
    warm[0] = True
    warm[1, :, :2] = True
    hot = torch.zeros_like(warm)
    hot[0, :, :1] = True
    hot[1, :, :1] = True
    state = TierState(warm=warm, hot=hot, step=2)
    _run(scores, scores.clone(), k1=2, k2=4, b2=5, refresh_n1=2, refresh_n2=2, state=state)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_dtypes_suportados(dtype):
    gen = torch.Generator().manual_seed(3)
    warm = torch.randn(2, 3, 4, 9, generator=gen).to(dtype)
    hot = torch.randn(2, 3, 4, 9, generator=gen).to(dtype)
    state = TierState(
        warm=torch.zeros(2, 4, 9, dtype=torch.bool),
        hot=torch.zeros(2, 4, 9, dtype=torch.bool),
        step=1,
    )
    state.warm[:, :, :4] = True
    state.hot[:, :, :2] = True
    _run(warm, hot, k1=2, k2=4, b2=3, state=state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sem CUDA neste host")
def test_cuda_repete_o_oraculo():
    warm = torch.randn(1, 2, 2, 8, device="cuda")
    hot = torch.randn(1, 2, 2, 8, device="cuda")
    _run(warm, hot, k1=2, k2=3, b2=2)
