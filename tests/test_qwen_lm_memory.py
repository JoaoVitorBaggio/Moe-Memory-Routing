"""Pico de cópia do treino. Não abre o store de 58 GB."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from moe_memory.offload.lm_train import (
    _chunked_cross_entropy,
    read_cgroup_memory_max,
    rss_bytes,
)
from tests.test_qwen_lm_train import _model


def test_chunked_cross_entropy_matches_the_full_head():
    torch.manual_seed(0)
    hidden = torch.randn(5, 8, requires_grad=True)
    weight = torch.randn(23, 8, dtype=torch.bfloat16).float()
    targets = torch.randint(0, 23, (5,))
    loss = _chunked_cross_entropy(hidden, weight, targets)
    reference = F.cross_entropy(F.linear(hidden, weight), targets)
    assert torch.allclose(loss, reference, atol=1e-5)
    loss.backward(retain_graph=True)
    got = hidden.grad.detach().clone()
    hidden.grad = None
    reference.backward()
    assert torch.allclose(got, hidden.grad, atol=1e-4)


def test_bf16_head_chunk_matches_without_a_full_fp32_copy():
    torch.manual_seed(1)
    hidden = torch.randn(4, 6, requires_grad=True)
    weight = torch.randn(19, 6, dtype=torch.bfloat16)
    targets = torch.tensor([0, 3, 7, 18])
    loss = _chunked_cross_entropy(hidden, weight, targets)
    reference = F.cross_entropy(F.linear(hidden, weight.float()), targets)
    assert torch.allclose(loss, reference, atol=1e-4)
    loss.backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()


def test_one_step_keeps_a_single_expert_copy(tmp_path):
    model, store, _directory = _model(tmp_path)
    try:
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
        from moe_memory.offload.lm_train import lm_step

        stats = lm_step(model, optimizer, tokens)
        assert stats["grads_finite"] is True
        assert model.expert_live == 0
        assert 1 <= model.expert_live_max <= 2
    finally:
        store.close()


def test_time_log_partitions_the_step_and_simulates_the_lru(tmp_path):
    from moe_memory.offload.lm_train import lm_step, make_optimizer, simulate_expert_lru

    model, store, _directory = _model(tmp_path)
    try:
        model.time_log = True
        model.cache_sim_sizes = (0, 1, 8)
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
        stats = lm_step(model, make_optimizer(model, 1e-3), tokens)
        timing = stats["time"]
        assert abs(timing["accounted_s"] + timing["outros_s"] - timing["wall_s"]) < 1e-6
        assert timing["experts_leituras"] >= timing["experts_unicos"] >= 1
        assert timing["expert_read_backward_bytes"] > 0
        assert timing["cache_lru"]["0"]["hits"] == 0
    finally:
        store.close()
    reads = [((0, 1), 10), ((0, 1), 10), ((0, 2), 10), ((0, 1), 10)]
    assert simulate_expert_lru(reads, 1)["hits"] == 1
    assert simulate_expert_lru(reads, 2)["hits"] == 2
    assert simulate_expert_lru(reads, 2)["avoided_bytes"] == 20


def test_cgroup_parser_accepts_a_number_and_max(tmp_path):
    path = tmp_path / "memory.max"
    path.write_text("9663676416\n", encoding="utf-8")
    assert read_cgroup_memory_max([path]) == 9_663_676_416
    path.write_text("max\n", encoding="utf-8")
    assert read_cgroup_memory_max([path]) is None
    assert rss_bytes() is None or rss_bytes() > 0
