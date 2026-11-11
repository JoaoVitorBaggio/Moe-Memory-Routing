"""Pico de cópia do treino. Não abre o store de 58 GB."""

from __future__ import annotations

import json

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


def test_profile_step_writes_tables_without_a_trace(tmp_path):
    from moe_memory.offload.lm_train import lm_step, make_optimizer, profile_lm_step

    model, store, _directory = _model(tmp_path)
    try:
        tokens = torch.tensor([[1, 2, 3, 4]])
        out = tmp_path / "profile-step-1"
        with profile_lm_step(out, with_stack=False, with_trace=False):
            stats = lm_step(model, make_optimizer(model, 1e-3), tokens)
        assert stats["grads_finite"] is True
        assert (out / "profile_ops_top50.txt").is_file()
        assert (out / "profile_shapes_top30.txt").is_file()
        assert not (out / "profile_trace.json").exists()
        summary = json.loads((out / "profile_regions.json").read_text(encoding="utf-8"))
        assert summary["profile_memory"] is False
        assert summary["with_stack"] is False
        assert summary["calls"]["total"] > 0
        assert "router_plan" in summary["regions"]
        assert summary["regions"]["router_plan"]["count"] >= 1
        assert any(name.startswith("expert_compute_") for name in summary["regions"])
    finally:
        store.close()


def _grad_map(model):
    found = {}
    for name, parameter in model.named_parameters():
        found[name] = None if parameter.grad is None else parameter.grad.detach().clone()
    return found


def test_lora_grads_match_the_old_slice_path_and_skip_select_backward(tmp_path):
    from moe_memory.offload.lm_train import flush_lora_slice_grads, legacy_lora_slice_grads
    from torch.profiler import ProfilerActivity, profile

    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 1, 2, 3, 4]])
    legacy, legacy_store, _legacy_dir = _model(tmp_path / "legacy")
    current, current_store, _current_dir = _model(tmp_path / "current")
    try:
        with legacy_lora_slice_grads():
            loss_old, _plan = legacy.forward_window(tokens)
            loss_old.backward()
            flush_lora_slice_grads()
        loss_new, _plan = current.forward_window(tokens)
        loss_new.backward()
        flush_lora_slice_grads()
        assert torch.allclose(loss_old, loss_new, atol=0, rtol=0)
        for name, old in _grad_map(legacy).items():
            new = current.get_parameter(name).grad
            if old is None:
                assert new is None
                continue
            assert new is not None
            assert torch.allclose(old, new, atol=1e-6, rtol=1e-5), name
        opt_old = torch.optim.AdamW([parameter for parameter in legacy.parameters() if parameter.requires_grad], lr=1e-3, weight_decay=0.0, foreach=False)
        opt_new = torch.optim.AdamW([parameter for parameter in current.parameters() if parameter.requires_grad], lr=1e-3, weight_decay=0.0, foreach=False)
        opt_old.step()
        opt_new.step()
        for name, parameter in legacy.named_parameters():
            assert torch.allclose(parameter, current.get_parameter(name), atol=1e-6, rtol=1e-5), name

        hidden = current.weights.spec.hidden_size
        left = torch.randn(2, hidden)
        coeff = torch.tensor([1.0, 0.4])
        from moe_memory.offload.lm_train import _apply_expert

        for target, flag in ((legacy, True), (current, False)):
            target.zero_grad(set_to_none=True)
        with legacy_lora_slice_grads():
            out = _apply_expert(legacy, left, coeff, 0, 1) + _apply_expert(legacy, left + 0.1, coeff, 0, 1)
            out.sum().backward()
            flush_lora_slice_grads()
        out = _apply_expert(current, left, coeff, 0, 1) + _apply_expert(current, left + 0.1, coeff, 0, 1)
        out.sum().backward()
        flush_lora_slice_grads()
        assert torch.allclose(legacy.lora.gate_B.grad, current.lora.gate_B.grad, atol=1e-6, rtol=1e-5)

        def full_lora_zeros(model, legacy: bool) -> int:
            model.zero_grad(set_to_none=True)
            full = {tuple(parameter.shape) for name, parameter in model.named_parameters() if name.startswith("lora.")}
            loss, _plan = model.forward_window(tokens)
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=False) as prof:
                if legacy:
                    with legacy_lora_slice_grads():
                        loss.backward()
                        flush_lora_slice_grads()
                else:
                    loss.backward()
                    flush_lora_slice_grads()
            total = 0
            for event in prof.key_averages(group_by_input_shape=True):
                if event.key != "aten::zero_":
                    continue
                shapes = getattr(event, "input_shapes", None) or []
                if any(tuple(shape) in full for shape in shapes if shape):
                    total += int(event.count)
            return total

        with legacy_lora_slice_grads():
            old_zeros = full_lora_zeros(legacy, True)
        new_zeros = full_lora_zeros(current, False)
        lora_params = sum(1 for name, _parameter in current.named_parameters() if name.startswith("lora."))
        assert old_zeros > lora_params
        assert new_zeros <= lora_params
    finally:
        legacy_store.close()
        current_store.close()


def test_backward_is_faster_without_full_tensor_grads(tmp_path):
    import time

    from moe_memory.lstm_ahead_router import TwoHorizonLstmRouter
    from moe_memory.offload.lm_train import (
        OffloadLanguageModel,
        flush_lora_slice_grads,
        legacy_lora_slice_grads,
    )
    from moe_memory.offload.qwen import load_resident, load_spec
    from moe_memory.offload.store import ExpertStore
    from moe_memory.offload.synthetic import write_synthetic_bundle

    def build(path: Path):
        torch.manual_seed(1)
        write_synthetic_bundle(
            path,
            n_layers=4,
            n_experts=8,
            hidden=1024,
            intermediate=32,
            vocab=32,
            n_heads=8,
            n_kv=2,
            head_dim=32,
            seed=0,
        )
        store = ExpertStore.open(path / "experts.moestore", direct=True)
        spec = load_spec(path / "config.json", store.header.dtype_code)
        weights = load_resident(path / "resident.bin", spec, torch.device("cpu"))
        router = TwoHorizonLstmRouter(
            vocab_size=32,
            n_layers=4,
            n_experts=8,
            l1=1,
            l2=2,
            k1=2,
            k2=4,
            b2=2,
            embed_dim=16,
            hidden_size=16,
        )
        return OffloadLanguageModel(weights, router, store, rank=32, alpha=8.0), store

    tokens = torch.randint(1, 32, (1, 8))

    def median_backward(model, legacy: bool) -> float:
        samples = []
        for _ in range(3):
            model.zero_grad(set_to_none=True)
            loss, _plan = model.forward_window(tokens)
            started = time.perf_counter()
            if legacy:
                with legacy_lora_slice_grads():
                    loss.backward()
                    flush_lora_slice_grads()
            else:
                loss.backward()
                flush_lora_slice_grads()
            samples.append(time.perf_counter() - started)
        samples.sort()
        return samples[1]

    legacy, legacy_store = build(tmp_path / "legacy")
    current, current_store = build(tmp_path / "current")
    try:
        assert legacy.lora.gate_A.numel() * 4 > 1_000_000
        with legacy_lora_slice_grads():
            old_seconds = median_backward(legacy, True)
        new_seconds = median_backward(current, False)
        assert new_seconds < old_seconds
    finally:
        legacy_store.close()
        current_store.close()


def test_expert_cache_matches_grads_and_skips_rereads(tmp_path):
    from moe_memory.offload.lm_train import flush_lora_slice_grads

    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 1, 2, 3, 4]])
    cold, cold_store, _cold_dir = _model(tmp_path / "cold")
    warm, warm_store, _warm_dir = _model(tmp_path / "warm")
    try:
        warm.set_expert_cache(32)
        cold.time_log = True
        warm.time_log = True
        cold.cache_sim_sizes = (32,)
        warm.cache_sim_sizes = (32,)
        loss_cold, _plan = cold.forward_window(tokens)
        loss_warm, _plan = warm.forward_window(tokens)
        assert torch.equal(loss_cold, loss_warm)
        after_forward = warm.store_reads
        assert after_forward > 0
        loss_cold.backward()
        flush_lora_slice_grads()
        loss_warm.backward()
        flush_lora_slice_grads()
        assert torch.equal(loss_cold, loss_warm)
        for name, parameter in cold.named_parameters():
            old = parameter.grad
            new = warm.get_parameter(name).grad
            if old is None:
                assert new is None
                continue
            assert torch.equal(old, new), name
        assert warm.store_reads == after_forward
        assert warm.store_reads * 2 < cold.store_reads
        assert warm.expert_cache.hits > 0
        assert warm.expert_cache.hits + warm.expert_cache.misses == len(warm._read_seq)
        for blobs in warm.expert_cache.stored():
            assert all(isinstance(part, bytes) for part in blobs)
        timing = warm.timing_report(1.0)
        assert timing["expert_cache"]["hits"] == warm.expert_cache.hits
        assert timing["expert_cache"]["avoided_bytes"] > 0
        assert "32" in timing["cache_lru"]
        warm.release_expert_cache()
        assert warm.expert_cache.stored_bytes == 0
        assert warm.expert_cache.stored() == []
    finally:
        cold_store.close()
        warm_store.close()


def test_expert_cache_byte_limit_keeps_a_single_block(tmp_path):
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    model, store, _directory = _model(tmp_path)
    try:
        model.set_expert_cache(0, max_bytes=1)
        loss, _plan = model.forward_window(tokens)
        loss.backward()
        from moe_memory.offload.lm_train import flush_lora_slice_grads

        flush_lora_slice_grads()
        assert model.expert_cache.stored_bytes == 0
        assert model.expert_cache.misses > 0
        assert model.expert_cache.hits == 0
        model.set_expert_cache(0, max_bytes=10**9)
        model.store_reads = 0
        model.expert_cache.reset_counters()
        again, _plan = model.forward_window(tokens)
        assert torch.equal(again, loss)
        assert model.expert_cache.stored_bytes > 0
        assert model.expert_cache.stored_bytes <= 10**9
    finally:
        store.close()


def test_pinned_experts_stay_in_the_lru():
    from moe_memory.offload.lm_train import ExpertWeightCache

    cache = ExpertWeightCache(2, 0)
    blob = (b"g", b"u", b"d")
    cache.set_pinned({(0, 0)})
    cache.store((0, 0), blob, 3)
    cache.store((0, 1), blob, 3)
    cache.store((0, 2), blob, 3)
    assert cache.contains((0, 0))
    assert cache.contains((0, 2))
    assert not cache.contains((0, 1))
    cache.set_pinned({(0, 0), (0, 2)})
    cache.store((0, 3), blob, 3)
    assert cache.contains((0, 0))
    assert cache.contains((0, 2))
    assert not cache.contains((0, 3))


def test_prefetch_hits_once_and_the_thread_joins():
    import threading
    import time

    from moe_memory.offload.lm_train import ExpertPrefetcher, ExpertWeightCache

    class Store:
        def __init__(self):
            self.calls: list[tuple[int, int]] = []

        def payload(self, layer: int, expert: int):
            self.calls.append((layer, expert))
            time.sleep(0.02)
            return (b"g", b"u", b"d")

    class Model:
        def __init__(self):
            self.expert_cache = ExpertWeightCache(4, 0)
            self.store = Store()
            self.time_log = True
            self._times: dict[str, float] = {}
            self._io_phase = "forward"
            self.store_reads = 0

        def _add_time(self, key: str, value: float) -> None:
            self._times[key] = self._times.get(key, 0.0) + value

    model = Model()
    prefetch = ExpertPrefetcher(model)
    prefetch.start(1, max_bytes=3, block=3)
    try:
        prefetch.set_plan([[(0, 0)], [(1, 0)], [(2, 0)]])
        prefetch.arm(0, 1)
        time.sleep(0.15)
        blobs, sync_s = prefetch.obtain(1, 0, "forward")
        assert blobs == (b"g", b"u", b"d")
        assert sync_s is None
        assert prefetch.hits == 1
        assert model.store.calls.count((1, 0)) == 1
        again, again_sync = prefetch.obtain(1, 0, "forward")
        assert again == blobs
        assert again_sync is None
        assert model.store.calls.count((1, 0)) == 1
        assert prefetch.hidden_read_s > 0
        first, first_sync = prefetch.obtain(0, 0, "forward")
        assert first_sync is not None and first_sync > 0
        prefetch.arm(1, 1)
        _third, _third_sync = prefetch.obtain(2, 0, "forward")
        assert model.store.calls.count((2, 0)) == 1
    finally:
        prefetch.stop()
    assert not prefetch.alive
    assert not any(thread.name == "expert-prefetch" and thread.is_alive() for thread in threading.enumerate())


def test_prefetch_matches_grads_and_does_not_leak_a_thread(tmp_path):
    import threading

    from moe_memory.offload.lm_train import flush_lora_slice_grads, save_checkpoint

    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 1, 2, 3, 4]])
    plain, plain_store, _plain_dir = _model(tmp_path / "plain")
    ahead, ahead_store, ahead_dir = _model(tmp_path / "ahead")
    try:
        plain.set_expert_cache(8)
        ahead.set_expert_cache(8)
        ahead.set_expert_prefetch(2)
        loss_plain, _plan = plain.forward_window(tokens)
        loss_ahead, _plan = ahead.forward_window(tokens)
        assert torch.equal(loss_plain, loss_ahead)
        loss_plain.backward()
        flush_lora_slice_grads()
        loss_ahead.backward()
        flush_lora_slice_grads()
        assert torch.equal(loss_plain, loss_ahead)
        for name, parameter in plain.named_parameters():
            old = parameter.grad
            new = ahead.get_parameter(name).grad
            if old is None:
                assert new is None
                continue
            assert torch.equal(old, new), name
        assert ahead.store_reads == plain.store_reads
        assert ahead.prefetch.alive
        optimizer = torch.optim.AdamW(ahead.parameters(), lr=1e-3, foreach=False)
        ahead.release_expert_cache()
        assert not ahead.prefetch.alive
        save_checkpoint(ahead_dir / "checkpoint", step=1, model=ahead, optimizer=optimizer, extra={})
        assert not any(thread.name == "expert-prefetch" and thread.is_alive() for thread in threading.enumerate())
    finally:
        plain.release_expert_cache()
        ahead.release_expert_cache()
        plain_store.close()
        ahead_store.close()


def test_cgroup_parser_accepts_a_number_and_max(tmp_path):
    path = tmp_path / "memory.max"
    path.write_text("9663676416\n", encoding="utf-8")
    assert read_cgroup_memory_max([path]) == 9_663_676_416
    path.write_text("max\n", encoding="utf-8")
    assert read_cgroup_memory_max([path]) is None
    assert rss_bytes() is None or rss_bytes() > 0
