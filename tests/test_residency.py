"""Residência física: mock de VRAM no CI, parâmetros quietos no CPU."""

import math
from pathlib import Path

import pytest
import torch

from moe_memory.emulator import ResidencyEmulator
from moe_memory.levels import COLD, HOT, WARM
from moe_memory.model import MemoryMoEModel
from moe_memory.residency import MockVram, attach_residency


def _model() -> MemoryMoEModel:
    return MemoryMoEModel(
        n_layers=1,
        d_model=16,
        n_heads=4,
        d_ff=32,
        n_experts=4,
        k=2,
        vocab_size=50,
        max_seq_len=16,
        use_level=True,
        tie_embeddings=True,
    )


def _emulator(tmp_path: Path, model: MemoryMoEModel) -> ResidencyEmulator:
    return ResidencyEmulator(
        n_layers=1,
        n_experts=4,
        hot_slots=1,
        warm_slots=1,
        expert_nbytes=model.expert_nbytes(),
        work_dir=tmp_path,
    )


def _params(model: MemoryMoEModel, expert_id: int):
    return list(model.blocks[0].ffn.experts[expert_id].parameters())


def _flat(params) -> torch.Tensor:
    return torch.cat([parameter.data.detach().reshape(-1).float().cpu() for parameter in params])


def _assert_homes(model, emulator, backend, directory: Path) -> None:
    for expert_id in range(4):
        level = int(emulator.levels[0, expert_id].item())
        params = _params(model, expert_id)
        on_device = [backend.on_device(parameter.data) for parameter in params]
        if level == HOT:
            assert all(on_device)
            assert not (directory / f"layer0_expert{expert_id}.f32").exists()
        else:
            assert not any(on_device)
        if level == WARM:
            assert all(backend.is_pinned(parameter.data) for parameter in params)
        if level == COLD:
            assert (directory / f"layer0_expert{expert_id}.f32").exists()


def test_cpu_smoke_does_not_swap_parameter_storage(tmp_path: Path):
    model = _model()
    emulator = _emulator(tmp_path, model)
    pointer = model.blocks[0].ffn.experts[0].fc1.weight.data_ptr()
    residency = attach_residency(model, emulator, torch.device("cpu"))
    store = residency.materialize()
    assert residency.backend is None
    assert residency.loads == 0
    assert model.blocks[0].ffn.experts[0].fc1.weight.data_ptr() == pointer
    assert store.n_cold_files == int((emulator.levels == COLD).sum().item())
    assert store.n_warm == int((emulator.levels == WARM).sum().item())


def test_mock_vram_parks_non_hot_experts_and_stages_them_for_compute(tmp_path: Path):
    torch.manual_seed(0)
    model = _model()
    emulator = _emulator(tmp_path, model)
    backend = MockVram()
    residency = attach_residency(model, emulator, torch.device("cpu"), backend=backend)
    assert backend.on_device(model.blocks[0].attn.qkv.weight.data)

    model.eval()
    ids = torch.randint(0, 50, (2, 8))
    with torch.no_grad():
        before = model(ids, emulator.levels).logits.clone()
    residency.materialize()
    _assert_homes(model, emulator, backend, tmp_path)
    with torch.no_grad():
        after = model(ids, emulator.levels).logits
    assert torch.equal(before, after)
    # O forward sobe quem o top-k tocou. materialize devolve o home.
    residency.materialize()
    _assert_homes(model, emulator, backend, tmp_path)

    cold_id = 3
    assert int(emulator.levels[0, cold_id].item()) == COLD
    parked = _flat(_params(model, cold_id)).clone()
    backend.last_non_blocking.clear()
    loads = residency.loads
    residency.stage_for_compute(0, cold_id)
    assert residency.loads == loads + 1
    assert backend.last_non_blocking
    assert not any(backend.last_non_blocking)
    assert torch.equal(parked, _flat(_params(model, cold_id)))
    assert all(backend.on_device(parameter.data) for parameter in _params(model, cold_id))
    # A segunda subida não copia de novo: o expert já está no compute.
    residency.stage_for_compute(0, cold_id)
    assert residency.loads == loads + 1
    bytes_before = emulator.total_bytes
    swaps_before = emulator.total_swaps
    residency.materialize()
    assert emulator.total_bytes == bytes_before
    assert emulator.total_swaps == swaps_before
    assert torch.equal(parked, _flat(_params(model, cold_id)))

    warm_id = 1
    assert int(emulator.levels[0, warm_id].item()) == WARM
    backend.last_non_blocking.clear()
    residency.stage_for_compute(0, warm_id)
    assert backend.last_non_blocking
    assert all(backend.last_non_blocking)


def test_promoting_a_cold_expert_loads_the_memmap_into_vram(tmp_path: Path):
    model = _model()
    emulator = _emulator(tmp_path, model)
    backend = MockVram()
    residency = attach_residency(model, emulator, torch.device("cpu"), backend=backend)
    residency.materialize()
    before = _flat(_params(model, 3)).clone()
    dispatched = torch.zeros(1, 4)
    dispatched[0, 3] = 9
    emulator.step(dispatched)
    assert int(emulator.levels[0, 3].item()) == HOT
    accounted = emulator.total_bytes
    residency.materialize()
    assert emulator.total_bytes == accounted
    assert all(backend.on_device(parameter.data) for parameter in _params(model, 3))
    assert not (tmp_path / "layer0_expert3.f32").exists()
    assert torch.equal(before, _flat(_params(model, 3)))


def test_two_steps_keep_homes_aligned_and_finite(tmp_path: Path):
    torch.manual_seed(0)
    model = _model()
    emulator = _emulator(tmp_path, model)
    backend = MockVram()
    residency = attach_residency(model, emulator, torch.device("cpu"), backend=backend)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    residency.optimizer = optimizer
    residency.materialize(optimizer)
    ids = torch.randint(0, 50, (2, 8))
    model.train()
    for _ in range(2):
        output = model(ids, emulator.levels)
        loss = output.logits.sum()
        assert math.isfinite(float(loss.detach()))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        emulator.step(output.dispatched)
        residency.materialize(optimizer)
    _assert_homes(model, emulator, backend, tmp_path)
    for parameter in model.parameters():
        assert torch.isfinite(parameter.data).all()
    assert backend.on_device(model.tok_emb.weight.data)
    assert residency.loads > 0


def test_requesting_cuda_without_a_gpu_points_at_the_mock(tmp_path: Path):
    if torch.cuda.is_available():
        pytest.skip("há GPU; o caminho sem device não é este")
    model = _model()
    emulator = _emulator(tmp_path, model)
    with pytest.raises(RuntimeError, match="MockVram"):
        attach_residency(model, emulator, torch.device("cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sem CUDA neste host")
def test_real_cuda_keeps_non_hot_experts_off_the_device(tmp_path: Path):
    model = _model().to("cuda")
    emulator = _emulator(tmp_path, model)
    residency = attach_residency(model, emulator, torch.device("cuda"))
    residency.materialize()
    warm = _params(model, 1)
    cold = _params(model, 2)
    assert all(parameter.device.type == "cpu" and parameter.is_pinned() for parameter in warm)
    assert all(parameter.device.type == "cpu" and not parameter.is_pinned() for parameter in cold)
    assert all(parameter.device.type == "cuda" for parameter in _params(model, 0))
    ids = torch.randint(0, 50, (2, 8), device="cuda")
    with torch.no_grad():
        output = model(ids, emulator.levels)
    assert output.logits.device.type == "cuda"
    assert math.isfinite(float(output.logits.sum()))
    residency.materialize()
    for expert_id in range(4):
        level = int(emulator.levels[0, expert_id].item())
        devices = {parameter.device.type for parameter in _params(model, expert_id)}
        if level == HOT:
            assert devices == {"cuda"}
        else:
            assert devices == {"cpu"}
