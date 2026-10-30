"""LRU por batch, orçamento B e homes morno (CPU) / frio (memmap)."""

from pathlib import Path

import torch

from moe_memory.emulator import PhysicalExpertStore, ResidencyEmulator
from moe_memory.levels import COLD, HOT, WARM


def _emulator(tmp_path: Path, expert_nbytes: int = 16) -> ResidencyEmulator:
    return ResidencyEmulator(
        n_layers=1,
        n_experts=32,
        hot_slots=8,
        warm_slots=8,
        expert_nbytes=expert_nbytes,
        work_dir=tmp_path,
    )


def test_initial_placement_fills_three_levels(tmp_path: Path):
    emulator = _emulator(tmp_path)
    levels = emulator.levels[0]
    assert torch.equal(levels[:8], torch.full((8,), HOT))
    assert torch.equal(levels[8:16], torch.full((8,), WARM))
    assert torch.equal(levels[16:], torch.full((16,), COLD))
    assert emulator.step(torch.zeros(1, 32)).swaps == 0


def test_touching_a_cold_expert_promotes_it_under_the_budget(tmp_path: Path):
    emulator = _emulator(tmp_path, expert_nbytes=10)
    dispatched = torch.zeros(1, 32)
    dispatched[0, 31] = 4
    stats = emulator.step(dispatched)
    levels = emulator.levels[0]
    assert int((levels == HOT).sum()) == 8
    assert int(levels[31]) == HOT
    assert stats.swaps == 3
    assert stats.promotions == 1
    assert stats.demotions == 2
    # frio→quente (2) + quente→morno (1) + morno→frio (1)
    assert stats.bytes_moved == 4 * 10
    assert emulator.total_bytes == stats.bytes_moved


def test_half_and_double_budget_resize_the_hot_set(tmp_path: Path):
    emulator = _emulator(tmp_path)
    stats = emulator.set_hot_budget(4)
    assert int((emulator.levels == HOT).sum()) == 4
    assert set(torch.where(emulator.levels[0] == HOT)[0].tolist()) == {0, 1, 2, 3}
    assert stats.swaps > 0
    emulator.set_hot_budget(16)
    assert int((emulator.levels == HOT).sum()) == 16
    try:
        emulator.set_hot_budget(30)
    except ValueError:
        pass
    else:
        raise AssertionError("quente + morno acima de E deveria falhar")


def test_warm_home_is_cpu_and_cold_home_roundtrips(tmp_path: Path):
    store = PhysicalExpertStore(tmp_path)
    weight = torch.arange(8, dtype=torch.float32)
    store.place(0, 1, WARM, weight)
    warm = store.read(0, 1, WARM)
    assert warm.device.type == "cpu"
    assert torch.equal(warm, weight)
    if torch.cuda.is_available():
        assert warm.is_pinned()

    store.place(0, 2, COLD, weight + 1)
    cold = store.read(0, 2, COLD)
    assert torch.equal(cold, weight + 1)
    assert store.n_cold_files == 1

    store.place(0, 2, HOT, weight)
    assert store.n_cold_files == 0
    assert not (tmp_path / "layer0_expert2.f32").exists()
