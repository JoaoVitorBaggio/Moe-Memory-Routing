"""AdamW esparso contra o AdamW denso, no CPU. CUDA fica marcado para o Fedora."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from moe_memory.offload.lm_train import (
    _queue_slice_grad,
    estimate_bytes,
    load_checkpoint,
    lm_step,
    make_optimizer,
    save_checkpoint,
)
from moe_memory.offload.sparse_adamw import SparseAdamW
from tests.test_qwen_lm_train import _model


def _run_pair(shape: tuple[int, ...], *, steps: int, lr: float, weight_decay: float, seed: int) -> None:
    torch.manual_seed(seed)
    base = torch.randn(*shape)
    dense = nn.Parameter(base.clone())
    sparse = nn.Parameter(base.clone())
    opt_dense = torch.optim.AdamW([dense], lr=lr, weight_decay=weight_decay, foreach=False)
    opt_sparse = SparseAdamW([sparse], lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed + 1)
    layers, experts = shape[0], shape[1]
    for step in range(steps):
        grad = torch.zeros_like(dense)
        chosen = []
        for layer in range(layers):
            for expert in range(experts):
                if int(torch.randint(0, 4, (1,), generator=generator)) != 0:
                    continue
                chosen.append((layer, expert))
        if not chosen:
            chosen.append((step % layers, step % experts))
        opt_dense.zero_grad(set_to_none=True)
        opt_sparse.zero_grad(set_to_none=True)
        for layer, expert in chosen:
            noise = torch.randn(dense[layer, expert].shape, generator=generator)
            grad[layer, expert] = noise
            _queue_slice_grad(sparse, layer, expert, noise.clone())
        dense.grad = grad
        opt_dense.step()
        opt_sparse.step()
    opt_sparse.catch_up_all()
    assert torch.equal(dense, sparse)
    exp_avg, exp_avg_sq = opt_sparse.materialize(sparse)
    assert torch.equal(exp_avg, opt_dense.state[dense]["exp_avg"])
    assert torch.equal(exp_avg_sq, opt_dense.state[dense]["exp_avg_sq"])
    assert int(opt_dense.state[dense]["step"].item()) == int(opt_sparse.state[sparse]["step"].item())


def test_sparse_adamw_matches_dense_with_random_sparsity():
    _run_pair((2, 5, 3, 4), steps=7, lr=1e-2, weight_decay=0.0, seed=0)
    _run_pair((2, 5, 3, 4), steps=7, lr=1e-2, weight_decay=0.05, seed=1)


def test_sparse_adamw_catches_up_experts_skipped_for_many_steps():
    shape = (2, 4, 3)
    base = torch.arange(24, dtype=torch.float32).reshape(shape)
    dense = nn.Parameter(base.clone())
    sparse = nn.Parameter(base.clone())
    opt_dense = torch.optim.AdamW([dense], lr=3e-2, weight_decay=0.02, foreach=False)
    opt_sparse = SparseAdamW([sparse], lr=3e-2, weight_decay=0.02)
    first = torch.randn(shape[-1])
    for step in range(12):
        grad = torch.zeros_like(dense)
        opt_dense.zero_grad(set_to_none=True)
        opt_sparse.zero_grad(set_to_none=True)
        if step == 0 or step == 11:
            grad[0, 1] = first
            _queue_slice_grad(sparse, 0, 1, first.clone())
        else:
            noise = torch.randn(shape[-1])
            grad[1, 2] = noise
            _queue_slice_grad(sparse, 1, 2, noise.clone())
        dense.grad = grad
        opt_dense.step()
        opt_sparse.step()
    opt_sparse.catch_up_all()
    assert torch.equal(dense, sparse)
    exp_avg, exp_avg_sq = opt_sparse.materialize(sparse)
    assert torch.equal(exp_avg, opt_dense.state[dense]["exp_avg"])
    assert torch.equal(exp_avg_sq, opt_dense.state[dense]["exp_avg_sq"])
    dense_bytes = opt_dense.state[dense]["exp_avg"].numel() * 8
    assert opt_sparse.state_bytes() < dense_bytes


def test_default_optimizer_stays_dense(tmp_path: Path):
    model, store, _directory = _model(tmp_path)
    try:
        optimizer = make_optimizer(model, 1e-3)
        assert isinstance(optimizer, torch.optim.AdamW)
        assert not isinstance(optimizer, SparseAdamW)
    finally:
        store.close()


def test_sparse_training_matches_dense_loss_and_checkpoint(tmp_path: Path):
    dense_model, dense_store, _left = _model(tmp_path / "dense")
    sparse_model, sparse_store, _right = _model(tmp_path / "sparse")
    try:
        opt_dense = make_optimizer(dense_model, 1e-2, sparse=False)
        opt_sparse = make_optimizer(sparse_model, 1e-2, sparse=True)
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
        for _ in range(3):
            left = lm_step(dense_model, opt_dense, tokens)
            right = lm_step(sparse_model, opt_sparse, tokens)
            assert left["loss"] == right["loss"]
            assert right["optimizer_updated_params"] <= left["optimizer_updated_params"]
            assert right["optimizer_state_bytes"] <= left["optimizer_state_bytes"]
            assert right["optimizer_updated_tensors"] >= 1
            assert right["optimizer_moment_slices"] >= 1
        opt_sparse.catch_up_all()
        for (name, left_param), (other, right_param) in zip(
            dense_model.named_parameters(), sparse_model.named_parameters(), strict=True
        ):
            assert name == other
            assert torch.equal(left_param, right_param)
        for name, param in sparse_model.named_parameters():
            if param.ndim < 3:
                continue
            exp_avg, exp_avg_sq = opt_sparse.materialize(param)
            dense_param = dict(dense_model.named_parameters())[name]
            assert torch.equal(exp_avg, opt_dense.state[dense_param]["exp_avg"])
            assert torch.equal(exp_avg_sq, opt_dense.state[dense_param]["exp_avg_sq"])
        path = tmp_path / "checkpoint"
        save_checkpoint(path, step=3, model=sparse_model, optimizer=opt_sparse, extra={"windows": [[1, 2, 3, 4, 5, 6]]})
        resumed, resumed_store, _directory = _model(tmp_path / "resumed")
        try:
            resumed_opt = make_optimizer(resumed, 1e-2, sparse=True)
            load_checkpoint(path, resumed, resumed_opt)
            for (name, kept), (other, loaded) in zip(
                sparse_model.named_parameters(), resumed.named_parameters(), strict=True
            ):
                assert name == other
                assert torch.equal(kept, loaded)
            again = lm_step(sparse_model, opt_sparse, tokens)
            loaded_step = lm_step(resumed, resumed_opt, tokens)
            assert again["loss"] == loaded_step["loss"]
            opt_sparse.catch_up_all()
            resumed_opt.catch_up_all()
            for (name, kept), (other, loaded) in zip(
                sparse_model.named_parameters(), resumed.named_parameters(), strict=True
            ):
                assert name == other
                assert torch.equal(kept, loaded)
        finally:
            resumed_store.close()
    finally:
        dense_store.close()
        sparse_store.close()


def test_sparse_estimate_shrinks_adam_and_keeps_vram():
    common = dict(
        resident_bytes=3_082_214_946,
        vocab=151_936,
        hidden=2048,
        intermediate=768,
        n_layers=48,
        n_experts=128,
        rank=4,
        seq_len=32,
        k1=4,
        lstm_embed=64,
        lstm_hidden=64,
        n_heads=32,
        n_kv_heads=4,
        head_dim=128,
        cgroup_bytes=9_663_676_416,
        read_cgroup=False,
        expert_cache_slots=0,
        expert_prefetch_layers=1,
        k2=6,
        compute_cuda=True,
        compute_elem=4,
        vram_expert_cache_bytes=3 * 1024**3,
        lm_head_vram=True,
        vram_cache_elem=2,
    )
    dense = estimate_bytes(**common)
    sparse = estimate_bytes(**common, sparse_adam_steps=3)
    assert sparse["adam_lora_bytes"] < dense["adam_lora_bytes"]
    assert sparse["lora_grad_bytes"] < dense["lora_grad_bytes"]
    assert sparse["vram_bytes"] == dense["vram_bytes"]
    assert sparse["adam_lora_bytes_if_dense"] == dense["adam_lora_bytes"]
    assert sparse["total_bytes"] < dense["total_bytes"]
    saved = dense["total_bytes"] - sparse["total_bytes"]
    assert saved > 800 * 1024**2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA só no Fedora")
def test_sparse_adamw_matches_dense_on_cuda():
    shape = (2, 3, 4, 2)
    torch.manual_seed(0)
    base = torch.randn(*shape, device="cuda")
    dense = nn.Parameter(base.clone())
    sparse = nn.Parameter(base.clone())
    opt_dense = torch.optim.AdamW([dense], lr=1e-2, weight_decay=0.01, foreach=False)
    opt_sparse = SparseAdamW([sparse], lr=1e-2, weight_decay=0.01)
    for step in range(5):
        grad = torch.zeros_like(dense)
        opt_dense.zero_grad(set_to_none=True)
        opt_sparse.zero_grad(set_to_none=True)
        noise = torch.randn(shape[-2:], device="cuda")
        layer, expert = step % 2, step % 3
        grad[layer, expert] = noise
        _queue_slice_grad(sparse, layer, expert, noise.detach().cpu())
        dense.grad = grad
        opt_dense.step()
        opt_sparse.step()
    opt_sparse.catch_up_all()
    assert torch.equal(dense, sparse)
