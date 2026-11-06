"""Folga de prefetch: RAM uma camada, SSD três ou quatro."""

import torch

from moe_memory.levels import COLD, HOT, WARM
from moe_memory.lookahead import lookahead_for_experts, lookahead_layers, prefetch_layer_span


def test_ram_is_one_layer_and_ssd_is_three_or_four():
    assert lookahead_layers(HOT) == 0
    assert lookahead_layers(WARM) == 1
    assert lookahead_layers(COLD) == 4
    assert lookahead_layers(COLD, ssd_layers=3) == 3
    try:
        lookahead_layers(COLD, ssd_layers=2)
    except ValueError as exc:
        assert "3" in str(exc)
    else:
        raise AssertionError("SSD fora de 3–4 deveria falhar")
    try:
        lookahead_layers(WARM, ram_layers=2)
    except ValueError as exc:
        assert "RAM" in str(exc)
    else:
        raise AssertionError("RAM diferente de 1 camada deveria falhar")


def test_the_coldest_expert_sets_the_depth():
    warm = torch.tensor([WARM, WARM])
    mixed = torch.tensor([WARM, COLD])
    assert lookahead_for_experts(warm) == 1
    assert lookahead_for_experts(mixed, ssd_layers=3) == 3


def test_prefetch_span_starts_at_the_next_layer_and_stops_at_the_stack():
    assert prefetch_layer_span(0, n_layers=4, depth=1) == [1]
    assert prefetch_layer_span(0, n_layers=4, depth=4) == [1, 2, 3]
    assert prefetch_layer_span(3, n_layers=4, depth=4) == []
    assert prefetch_layer_span(1, n_layers=4, depth=0) == []
