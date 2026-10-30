"""ℓ(e) entra no logit, e a mistura do FFN é a média ponderada dos gates."""

import torch
from torch import nn

from moe_memory.model import MixtureOfExperts
from moe_memory.router import MemoryAwareRouter


def test_level_changes_logits_only_when_the_router_is_allowed_to_see_them():
    router = MemoryAwareRouter(d_model=8, n_experts=4)
    with torch.no_grad():
        router.level_embed.weight.zero_()
        router.level_embed.weight[2].fill_(1.0)
    hidden = torch.randn(2, 3, 8)
    hot = torch.zeros(4, dtype=torch.long)
    cold = torch.full((4,), 2, dtype=torch.long)
    seen_hot = router(hidden, hot, use_level=True)
    seen_cold = router(hidden, cold, use_level=True)
    blind_hot = router(hidden, hot, use_level=False)
    blind_cold = router(hidden, cold, use_level=False)
    assert not torch.allclose(seen_hot, seen_cold)
    assert torch.allclose(blind_hot, blind_cold)


class _ConstExpert(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden * 0 + self.value + self.anchor


def test_dispatch_is_a_weighted_mean_and_skips_idle_experts():
    moe = MixtureOfExperts(d_model=4, d_ff=8, n_experts=4, k=2, use_level=True)
    for expert_id in range(4):
        moe.experts[expert_id] = _ConstExpert(float(expert_id))
    hidden = torch.ones(1, 1, 4)
    index = torch.tensor([[[0, 2]]])
    gates = torch.tensor([[[0.25, 0.75]]])
    mixed = moe._mix(hidden, index, gates)
    assert torch.allclose(mixed, torch.full_like(mixed, 1.5))
    mixed.sum().backward()
    assert moe.experts[0].anchor.grad is not None
    assert moe.experts[2].anchor.grad is not None
    assert moe.experts[1].anchor.grad is None
    assert moe.experts[3].anchor.grad is None


def test_level_embedding_gets_gradient_from_c_mem():
    from moe_memory.loss import compute_loss
    from moe_memory.model import MemoryMoEModel

    model = MemoryMoEModel(
        n_layers=1,
        d_model=16,
        n_heads=4,
        d_ff=32,
        n_experts=4,
        k=2,
        vocab_size=32,
        max_seq_len=8,
        use_level=True,
        tie_embeddings=True,
    )
    levels = torch.tensor([[0, 1, 2, 0]])
    input_ids = torch.randint(0, 32, (2, 4))
    output = model(input_ids, levels)
    terms = compute_loss(
        output.logits,
        input_ids,
        output.probs,
        output.topk_index,
        levels,
        torch.tensor([1.0, 10.0, 20.0]),
        lambda_=1.0,
        beta=0.0,
        aggregation="weighted_mean",
    )
    terms.c_mem.backward()
    grad = model.blocks[0].ffn.router.level_embed.weight.grad
    assert grad is not None
    assert grad.abs().sum() > 0


def test_tokenizer_matches_the_gpt2_vocab():
    from moe_memory.tokenizer import GPT2_VOCAB_SIZE, encode

    ids = encode("oi")
    assert ids
    assert all(0 <= token < GPT2_VOCAB_SIZE for token in ids)
