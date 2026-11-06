"""O gerente puxa pelo score, e o SSD é um home válido."""

import torch

from moe_memory.expert_manager import ExpertMemoryManager
from moe_memory.levels import COLD, HOT, WARM


def _homes() -> torch.Tensor:
    homes = torch.full((4, 4), WARM, dtype=torch.long)
    homes[0, 0] = COLD
    homes[1, 0] = COLD
    homes[2, 1] = HOT
    return homes


def test_higher_score_is_pulled_first_even_when_it_lives_on_ssd():
    manager = ExpertMemoryManager(4, 4, _homes())
    expert_ids = torch.tensor([1, 0])
    scores = torch.tensor([0.2, 0.9])
    ordered = manager.prioritize(0, expert_ids, scores)
    assert [item.expert_id for item in ordered] == [0, 1]
    assert ordered[0].level == COLD
    assert abs(ordered[0].score - 0.9) < 1e-5
    assert ordered[1].level == WARM


def test_schedule_covers_the_lookahead_and_keeps_score_order():
    manager = ExpertMemoryManager(4, 4, _homes())
    expert_ids = torch.tensor([1, 0])
    scores = torch.tensor([0.2, 0.9])
    pulled = manager.schedule(0, expert_ids, scores, ssd_layers=3)
    assert [(item.layer, item.expert_id) for item in pulled] == [
        (1, 0),
        (2, 0),
        (3, 0),
        (1, 1),
        (2, 1),
        (3, 1),
    ]
    assert pulled[0].level == COLD
    assert manager.pull_log == pulled


def test_hot_experts_do_not_prefetch_future_layers():
    homes = torch.full((3, 2), HOT, dtype=torch.long)
    manager = ExpertMemoryManager(3, 2, homes)
    pulled = manager.schedule(0, torch.tensor([0, 1]), torch.tensor([0.4, 0.8]))
    assert pulled == []


def test_set_home_accepts_ssd():
    homes = torch.full((2, 2), HOT, dtype=torch.long)
    manager = ExpertMemoryManager(2, 2, homes)
    manager.set_home(1, 0, COLD)
    assert int(manager.homes[1, 0].item()) == COLD
