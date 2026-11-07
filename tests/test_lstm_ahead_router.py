"""Esqueleto do LSTM adiantado, em dimensão minúscula. Sem OLMoE."""

import pytest
import torch
from torch import nn

import moe_memory.lstm_ahead_router as ahead
from moe_memory.lstm_ahead_router import (
    HEAD_INIT_STD,
    LstmAheadRouter,
    combine_selected,
)


class _ConstExpert(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(value))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden * 0 + self.anchor


def _tiny(**overrides: int) -> LstmAheadRouter:
    kwargs = dict(
        vocab_size=16,
        n_layers=2,
        n_experts=4,
        k=2,
        lookahead=3,
        embed_dim=8,
        hidden_size=8,
    )
    kwargs.update(overrides)
    return LstmAheadRouter(**kwargs)


def test_module_is_labeled_as_a_skeleton_and_lookahead_does_not_add_parameters():
    assert LstmAheadRouter.SKELETON is True
    assert "ESQUELETO" in (LstmAheadRouter.__doc__ or "")
    assert ahead.__doc__ is not None and "ESQUELETO" in ahead.__doc__
    short = _tiny(lookahead=1)
    long = _tiny(lookahead=5)
    short_count = sum(parameter.numel() for parameter in short.parameters())
    long_count = sum(parameter.numel() for parameter in long.parameters())
    assert short_count == long_count
    assert short.lookahead == 1
    assert long.lookahead == 5
    assert 0 < HEAD_INIT_STD < 0.02


def test_tiny_forward_reads_tokens_and_stays_near_uniform():
    torch.manual_seed(0)
    router = _tiny()
    tokens = torch.randint(0, 16, (2, 5))
    decision, state = router(tokens)
    assert decision.logits.shape == (2, 5, 2, 4)
    assert decision.probs.shape == decision.logits.shape
    assert decision.topk_index.shape == (2, 5, 2, 2)
    assert decision.soft_weights.shape == (2, 5, 2, 2)
    assert decision.lookahead == 3
    assert decision.logits.shape[1] == tokens.shape[1]
    assert torch.allclose(decision.probs.sum(dim=-1), torch.ones(2, 5, 2), atol=1e-5)
    uniform = torch.full_like(decision.probs, 1.0 / router.n_experts)
    assert torch.allclose(decision.probs, uniform, atol=0.05)
    gathered = decision.probs.gather(-1, decision.topk_index)
    assert torch.allclose(decision.soft_weights, gathered)
    assert torch.all(decision.soft_weights.sum(dim=-1) < 0.99)
    assert state[0].shape == (1, 2, 8)
    assert state[1].shape == (1, 2, 8)
    assert torch.equal(
        decision.topk_index,
        torch.topk(decision.logits, k=router.k, dim=-1).indices,
    )


def test_token_steps_match_the_full_sequence():
    torch.manual_seed(1)
    router = _tiny()
    tokens = torch.randint(0, 16, (2, 5))
    full, _ = router(tokens)
    state = None
    pieces = []
    for step in range(tokens.shape[1]):
        decision, state = router(tokens[:, step : step + 1], state)
        pieces.append(decision.logits)
    assert torch.allclose(torch.cat(pieces, dim=1), full.logits, atol=1e-5)


def test_soft_weights_reach_the_lstm_and_the_embedding():
    torch.manual_seed(2)
    router = _tiny()
    tokens = torch.randint(0, 16, (2, 5))
    decision, _ = router(tokens)
    decision.soft_weights.sum().backward()
    assert router.head.weight.grad is not None
    assert router.head.weight.grad.abs().sum() > 0
    assert router.embed.weight.grad is not None
    assert router.embed.weight.grad.abs().sum() > 0
    assert router.lstm.weight_ih_l0.grad is not None
    assert router.lstm.weight_ih_l0.grad.abs().sum() > 0


def test_mix_multiplies_by_lstm_probs_and_skips_idle_experts():
    experts = [_ConstExpert(float(expert_id * 10)) for expert_id in range(4)]
    hidden = torch.ones(1, 1, 4)
    index = torch.tensor([[[0, 2]]])
    weights = torch.tensor([[[0.2, 0.3]]])
    mixed = combine_selected(hidden, experts, index, weights)
    assert torch.allclose(mixed, torch.full_like(mixed, 6.0))
    mixed.sum().backward()
    assert experts[0].anchor.grad is not None
    assert experts[2].anchor.grad is not None
    assert experts[1].anchor.grad is None
    assert experts[3].anchor.grad is None

    torch.manual_seed(3)
    router = _tiny()
    tokens = torch.randint(0, 16, (1, 2))
    decision, _ = router(tokens)
    layer_hidden = torch.ones(1, 2, 4)
    layer_experts = [_ConstExpert(0.0) for _ in range(4)]
    mixed_layer = router.mix_layer(decision, 1, layer_hidden, layer_experts)
    assert mixed_layer.shape == (1, 2, 4)


def test_constructor_rejects_a_bad_width():
    with pytest.raises(ValueError):
        _tiny(k=0)
    with pytest.raises(ValueError):
        _tiny(k=5)
    with pytest.raises(ValueError):
        _tiny(lookahead=0)
