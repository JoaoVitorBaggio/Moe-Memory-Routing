"""Proxy k=2: média ponderada no treino, soma na inferência, mais o Switch."""

import torch

from moe_memory.loss import aggregate_topk, balance_loss, compute_loss, swap_cost_vector


def _fixture():
    # Um token repetido nas duas posições (a CE precisa de T>=2).
    # probs: e1=0.5 (frio, c=20), e2=0.3 (morno, c=10); os outros 0.1.
    probs_token = torch.tensor([0.1, 0.5, 0.3, 0.1])
    probs = probs_token.view(1, 1, 1, 4).expand(1, 1, 2, 4).contiguous()
    topk = torch.tensor([1, 2]).view(1, 1, 1, 2).expand(1, 1, 2, 2).contiguous()
    levels = torch.tensor([[0, 2, 1, 0]])
    level_cost = torch.tensor([1.0, 10.0, 20.0])
    logits = torch.zeros(1, 2, 5)
    logits[0, 0, 3] = 50.0
    input_ids = torch.tensor([[1, 3]])
    return logits, input_ids, probs, topk, levels, level_cost


def test_train_aggregation_is_the_weighted_mean_of_the_top_two():
    logits, input_ids, probs, topk, levels, level_cost = _fixture()
    terms = compute_loss(
        logits,
        input_ids,
        probs,
        topk,
        levels,
        level_cost,
        lambda_=1.0,
        beta=0.0,
        aggregation="weighted_mean",
    )
    # pesos 0.5/0.8 e 0.3/0.8 sobre c = 20 e 10
    assert torch.allclose(terms.access, torch.tensor(16.25))
    # w(frio→quente)=19, w(morno→quente)=9
    assert torch.allclose(terms.swap, torch.tensor(15.25))
    assert torch.allclose(terms.c_mem, torch.tensor(31.5))
    # Σ_e p·c = 0.1*1 + 0.5*20 + 0.3*10 + 0.1*1 = 13.2, e não entra em L
    assert torch.allclose(terms.access_expected, torch.tensor(13.2))
    assert not torch.allclose(terms.access, terms.access_expected)


def test_infer_aggregation_sums_the_top_two_costs():
    logits, input_ids, probs, topk, levels, level_cost = _fixture()
    terms = compute_loss(
        logits,
        input_ids,
        probs,
        topk,
        levels,
        level_cost,
        lambda_=1.0,
        beta=0.0,
        aggregation="sum",
    )
    assert torch.allclose(terms.access, torch.tensor(30.0))
    assert torch.allclose(terms.swap, torch.tensor(28.0))
    assert torch.allclose(terms.c_mem, torch.tensor(58.0))


def test_lambda_and_beta_scale_the_documented_sum():
    logits, input_ids, probs, topk, levels, level_cost = _fixture()
    base = compute_loss(
        logits, input_ids, probs, topk, levels, level_cost,
        lambda_=0.0, beta=0.0, aggregation="weighted_mean",
    )
    only_balance = compute_loss(
        logits, input_ids, probs, topk, levels, level_cost,
        lambda_=0.0, beta=0.01, aggregation="weighted_mean",
    )
    only_memory = compute_loss(
        logits, input_ids, probs, topk, levels, level_cost,
        lambda_=0.5, beta=0.0, aggregation="weighted_mean",
    )
    assert torch.allclose(only_balance.total, base.task + 0.01 * base.balance)
    assert torch.allclose(only_memory.total, base.task + 0.5 * base.c_mem)
    assert base.task < 1e-4


def test_train_weights_receive_gradient_and_infer_sum_does_not():
    logits, input_ids, probs, topk, levels, level_cost = _fixture()
    leaf = torch.tensor([0.1, 0.5, 0.3, 0.1], requires_grad=True)
    soft = leaf.softmax(dim=0)
    train_probs = soft.view(1, 1, 1, 4).repeat(1, 1, 2, 1)
    terms = compute_loss(
        logits, input_ids, train_probs, topk, levels, level_cost,
        lambda_=1.0, beta=0.0, aggregation="weighted_mean",
    )
    terms.c_mem.backward()
    assert leaf.grad is not None
    assert leaf.grad.abs().sum() > 0

    infer = compute_loss(
        logits, input_ids, probs, topk, levels, level_cost,
        lambda_=1.0, beta=0.0, aggregation="sum",
    )
    assert infer.c_mem.requires_grad is False


def test_switch_balance_is_one_when_uniform_and_grows_when_collapsed():
    uniform_probs = torch.full((1, 2, 4), 0.25)
    # cada token manda uma vaga para cada expert: k=4, fração uniforme
    uniform_index = torch.tensor([[[0, 1, 2, 3], [0, 1, 2, 3]]])
    uniform = balance_loss(uniform_probs, uniform_index)
    assert torch.allclose(uniform, torch.tensor(1.0))

    collapsed_probs = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])
    collapsed_index = torch.tensor([[[0, 1], [0, 1]]])
    # f = counts / (2*2) com counts [2, 2, 0, 0] → f = [0.5, 0.5, 0, 0]
    # P = [1, 0, 0, 0]
    # E * Σ f P = 4 * 0.5 = 2
    collapsed = balance_loss(collapsed_probs, collapsed_index)
    assert torch.allclose(collapsed, torch.tensor(2.0))
    assert collapsed > uniform


def test_swap_vector_matches_absolute_gap():
    cost = torch.tensor([1.0, 10.0, 20.0])
    assert torch.equal(swap_cost_vector(cost), torch.tensor([0.0, 9.0, 19.0]))


def test_aggregate_unknown_mode_fails():
    probs = torch.full((1, 1, 2), 0.5)
    index = torch.tensor([[[0, 1]]])
    try:
        aggregate_topk(torch.tensor([1.0, 2.0]), probs, index, "median")
    except ValueError as exc:
        assert "median" in str(exc)
    else:
        raise AssertionError("modo desconhecido deveria falhar")
