"""Smoke sintético da loss de linguagem. Não é o store de 58 GB."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
import torch

from moe_memory.lstm_ahead_router import TwoHorizonLstmRouter
from moe_memory.offload.lm_train import (
    LAUNCH_CEILING_BYTES,
    OffloadLanguageModel,
    estimate_bytes,
    fingerprint,
    launch_allowed,
    lm_step,
    load_checkpoint,
    make_optimizer,
    pascal_bf16_cuda_ok,
    persist_checkpoint,
    save_checkpoint,
)
from moe_memory.offload.qwen import KvCache, _attention, load_resident, load_spec
from moe_memory.offload.lm_train import _attention_seq
from moe_memory.offload.store import ExpertStore
from moe_memory.offload.synthetic import write_synthetic_bundle
from moe_memory.qwen_corpus import iter_arrow_text, iter_token_windows, wikitext_arrow_files
from moe_memory.qwen_tokenizer import bytes_to_unicode, load_qwen_tokenizer


def _bundle(tmp_path: Path):
    write_synthetic_bundle(
        tmp_path,
        n_layers=2,
        n_experts=4,
        hidden=32,
        intermediate=16,
        vocab=64,
        n_heads=4,
        n_kv=2,
        head_dim=8,
        seed=0,
    )
    return tmp_path


def _model(tmp_path: Path, rank: int = 2):
    torch.manual_seed(1)
    directory = _bundle(tmp_path)
    store = ExpertStore.open(directory / "experts.moestore", direct=True)
    spec = load_spec(directory / "config.json", store.header.dtype_code)
    weights = load_resident(directory / "resident.bin", spec, torch.device("cpu"))
    router = TwoHorizonLstmRouter(
        vocab_size=64,
        n_layers=2,
        n_experts=4,
        l1=1,
        l2=3,
        k1=2,
        k2=3,
        b2=2,
        embed_dim=16,
        hidden_size=16,
    )
    model = OffloadLanguageModel(weights, router, store, rank=rank, alpha=4.0)
    return model, store, directory


def test_estimate_only_does_not_need_the_tokenizer(tmp_path: Path):
    import subprocess
    import sys

    directory = _bundle(tmp_path / "bundle")
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/train_qwen_lm.py",
            "--estimate-only",
            "--store-dir",
            str(directory),
            "--model-dir",
            str(directory),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "fits_launch_ceiling" in completed.stdout
    assert "true" in completed.stdout.lower()


def test_pascal_does_not_run_bf16():
    assert pascal_bf16_cuda_ok((6, 1), False) is False
    assert pascal_bf16_cuda_ok((7, 5), True) is False
    assert pascal_bf16_cuda_ok((8, 6), True) is True
    assert pascal_bf16_cuda_ok(None, True) is False


def test_real_shape_estimate_fits_under_9gib_and_reads_the_given_cgroup():
    resident = 3_082_214_946
    nine = 9_663_676_416
    estimate = estimate_bytes(
        resident_bytes=resident,
        vocab=151_936,
        hidden=2048,
        intermediate=768,
        n_layers=48,
        n_experts=128,
        rank=4,
        seq_len=8,
        k1=4,
        lstm_embed=64,
        lstm_hidden=64,
        n_heads=32,
        n_kv_heads=4,
        head_dim=128,
        cgroup_bytes=nine,
        read_cgroup=False,
    )
    assert estimate["cgroup_bytes"] == nine
    assert estimate["lora_grad_bytes"] == estimate["lora_fp32_bytes"]
    assert estimate["expert_fp32_bytes"] == 3 * 2048 * 768 * 4
    assert estimate["lm_head_fp32_bytes"] < resident
    assert estimate["lm_head_fp32_bytes"] == 4096 * 2048 * 4
    assert estimate["file_cache_bytes"] == int(1.6 * 1024**3)
    assert estimate["checkpoint_extra_bytes"] == 0
    assert estimate["optimizer_step_temporary_bytes"] == 48 * 128 * 4 * 2048 * 4
    assert estimate["total_bytes"] < LAUNCH_CEILING_BYTES
    assert estimate["cgroup_total_bytes"] < nine
    assert estimate["resident_bytes"] == resident
    assert estimate["expert_cache_bytes"] == 0


def test_sixty_four_expert_cache_stays_under_the_anon_ceiling():
    nine = 9_663_676_416
    base = estimate_bytes(
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
        cgroup_bytes=nine,
        read_cgroup=False,
    )
    cached = estimate_bytes(
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
        cgroup_bytes=nine,
        read_cgroup=False,
        expert_cache_slots=64,
    )
    block = 3 * 2048 * 768 * 2
    assert cached["expert_cache_bytes"] == 64 * block
    assert cached["total_bytes"] == base["total_bytes"] + 64 * block
    assert cached["total_bytes"] < LAUNCH_CEILING_BYTES
    assert cached["cgroup_total_bytes"] > nine
    assert launch_allowed(base) is True
    assert launch_allowed(cached) is True


def test_cache_32_and_one_prefetch_layer_fit_under_9gib():
    nine = 9_663_676_416
    block = 3 * 2048 * 768 * 2
    estimate = estimate_bytes(
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
        cgroup_bytes=nine,
        read_cgroup=False,
        expert_cache_slots=32,
        expert_prefetch_layers=1,
        k2=6,
    )
    assert estimate["expert_cache_bytes"] == 32 * block
    assert estimate["expert_prefetch_bytes"] == 6 * block
    assert estimate["total_bytes"] < LAUNCH_CEILING_BYTES
    assert estimate["cgroup_total_bytes"] < nine
    assert launch_allowed(estimate) is True


def test_cuda_layer_budget_fits_under_5gb_and_9gib_host():
    from moe_memory.offload.lm_gpu import assert_vram_fits, host_compute_threads

    nine = 9_663_676_416
    budget = int(5 * 1024**3)
    estimate = estimate_bytes(
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
        cgroup_bytes=nine,
        read_cgroup=False,
        expert_cache_slots=32,
        expert_prefetch_layers=1,
        k2=6,
        compute_cuda=True,
        compute_elem=4,
    )
    assert estimate["vram_layers_on_device"] == 2
    assert estimate["pinned_bytes"] > 0
    assert estimate["vram_bytes"] < budget
    assert estimate["total_bytes"] < LAUNCH_CEILING_BYTES
    assert estimate["cgroup_total_bytes"] < nine
    assert launch_allowed(estimate) is True
    assert_vram_fits(int(estimate["vram_bytes"]), budget)
    with pytest.raises(ValueError, match="VRAM"):
        assert_vram_fits(int(estimate["vram_bytes"]), 1024)
    assert host_compute_threads(4, False) == 4
    assert host_compute_threads(4, True) == 3
    cached = estimate_bytes(
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
        cgroup_bytes=nine,
        read_cgroup=False,
        expert_cache_slots=8,
        expert_prefetch_layers=1,
        k2=6,
        compute_cuda=True,
        compute_elem=4,
        vram_expert_cache_bytes=2 * 1024**3,
    )
    assert cached["vram_expert_cache_bytes"] == 2 * 1024**3
    assert cached["vram_bytes"] < budget
    assert cached["cgroup_total_bytes"] < nine
    assert launch_allowed(cached) is True


def test_vram_expert_cache_does_not_evict_the_pinned_layer():
    from moe_memory.offload.lm_gpu import VramExpertCache

    cache = VramExpertCache(24)
    blob = (torch.zeros(1), torch.zeros(1), torch.zeros(1))
    cache.put((0, 0), blob)
    cache.put((0, 1), blob)
    cache.set_pinned({(0, 0)})
    cache.put((0, 2), blob)
    assert cache.take((0, 0)) is not None
    assert cache.take((0, 1)) is None
    assert cache.hits == 1
    assert cache.avoided_bytes == 12
    same = torch.ones(2)
    plain = VramExpertCache(100)
    plain.put((1, 0), (same, same, same))
    assert plain.take((1, 0))[0] is same


def test_bf16_vram_cache_returns_fp32_matching_the_direct_cast(tmp_path: Path):
    from moe_memory.offload.lm_gpu import LayerTransport, VramExpertCache

    source = torch.randn(4, 6, dtype=torch.bfloat16)
    direct = source.float()
    cache = VramExpertCache(source.nbytes * 3 * 2, dtype=torch.bfloat16, compute_dtype=torch.float32)
    cache.expert_nbytes = source.nbytes * 3
    cache.put((0, 0), (source, source, source))
    cache.put((0, 1), (source.clone(), source.clone(), source.clone()))
    assert cache.contains((0, 0)) and cache.contains((0, 1))
    assert cache.stats()["capacity_experts"] == 2
    taken = cache.take((0, 1))
    assert taken[0].dtype == torch.float32
    assert torch.equal(taken[0], direct)
    wide = VramExpertCache(source.nbytes * 3 * 2, dtype=torch.float32, compute_dtype=torch.float32)
    wide.put((0, 0), (direct, direct, direct))
    wide.put((0, 1), (direct.clone(), direct.clone(), direct.clone()))
    assert wide.contains((0, 1))
    assert not wide.contains((0, 0))

    model, store, _directory = _model(tmp_path)
    try:
        model.compute_device = torch.device("cpu")
        model.compute_dtype = torch.float32
        bf16 = LayerTransport(model)
        fp32 = LayerTransport(model)
        bf16.set_vram_cache(10**7, torch.bfloat16)
        fp32.set_vram_cache(10**7, torch.float32)
        hosts = (source, source, source)
        got_bf16, stored = bf16._place_views(hosts, block=True)
        got_fp32, _stored_fp32 = fp32._place_views(hosts, block=True)
        assert stored[0].dtype == torch.bfloat16
        assert got_bf16[0].dtype == torch.float32
        assert torch.equal(got_bf16[0], got_fp32[0])
        assert torch.equal(got_bf16[0], direct)
    finally:
        store.close()


def test_lm_head_vram_keeps_the_step_identical(tmp_path: Path):
    from moe_memory.offload.lm_train import lm_step, make_optimizer, release_cpu_lm_head

    plain, store_plain, _left = _model(tmp_path / "plain")
    flagged, store_flagged, _right = _model(tmp_path / "flag")
    try:
        flagged.enable_lm_head_vram()
        assert flagged.lm_head_cpu_released is False
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
        left = lm_step(plain, make_optimizer(plain, 1e-3), tokens)
        right = lm_step(flagged, make_optimizer(flagged, 1e-3), tokens)
        assert left["loss"] == right["loss"]
        for (name, base), (other_name, changed) in zip(plain.named_parameters(), flagged.named_parameters(), strict=True):
            assert name == other_name
            assert torch.equal(base, changed)
            if base.grad is not None:
                assert torch.equal(base.grad, changed.grad)
    finally:
        store_plain.close()
        store_flagged.close()

    shared = torch.randn(4, 3)
    tied = type("Weights", (), {"embed": shared, "lm_head": shared})()
    assert release_cpu_lm_head(tied, torch.randn(4, 3), "cuda") is False
    assert tied.lm_head.data_ptr() == shared.data_ptr()
    loose = type("Weights", (), {"embed": torch.randn(4, 3), "lm_head": torch.randn(4, 3)})()
    stored = torch.randn(4, 3)
    assert release_cpu_lm_head(loose, stored, "cuda") is True
    assert loose.lm_head is stored
    assert release_cpu_lm_head(loose, stored, "cpu") is False


def test_recommended_vram_budget_fits_in_5gib():
    from moe_memory.offload.lm_gpu import assert_vram_fits

    budget = 5 * 1024**3
    head = 151_936 * 2048 * 2
    estimate = estimate_bytes(
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
    assert estimate["vram_lm_head_bytes"] == head
    assert estimate["vram_cache_capacity_experts"] == 341
    assert estimate["vram_expert_cache_bytes"] == 3 * 1024**3
    assert estimate["vram_bytes"] <= budget
    assert_vram_fits(int(estimate["vram_bytes"]), budget)
    base = estimate_bytes(
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
    )
    assert estimate["total_bytes"] == base["total_bytes"] - head


def test_gpu_path_on_cpu_counts_the_ram_cache_and_reuses_vram(tmp_path: Path):
    from moe_memory.offload.lm_gpu import LayerTransport
    from moe_memory.offload.lm_train import flush_lora_slice_grads

    model, store, _directory = _model(tmp_path)
    try:
        assert model.gpu is None
        model.compute_device = torch.device("cpu")
        model.compute_dtype = torch.float32
        model.vram_budget = 10**12
        model.gpu = LayerTransport(model)
        model.set_expert_cache(8)
        model.set_vram_expert_cache(10**7)
        model.set_expert_prefetch(1)
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 1, 2, 3, 4]])
        loss, _plan = model.forward_window(tokens)
        vram_misses = model.gpu.vram_cache.misses
        loss.backward()
        flush_lora_slice_grads()
        ram = model.expert_cache.stats()
        assert ram["hits"] > 0
        assert ram["misses"] > 0
        assert model.gpu.vram_cache.hits > 0
        assert model.gpu.vram_cache.misses == vram_misses
        assert model.gpu.vram_cache.avoided_bytes > 0
    finally:
        model.release_expert_cache()
        store.close()


def test_layer_transport_copies_a_layer_on_cpu(tmp_path: Path):
    from moe_memory.offload.lm_gpu import LayerTransport

    model, store, _directory = _model(tmp_path)
    try:
        assert model.gpu is None
        model.compute_device = torch.device("cpu")
        model.compute_dtype = torch.float32
        model.vram_budget = 10**12
        model.time_log = True
        model._times = {}
        model._io_phase = "forward"
        transport = LayerTransport(model)
        assert callable(transport._h2d)
        layer = model.weights.layers[0]
        raw = model.store.payload(0, 0)
        pack = transport.ensure(0, layer, {(0, 0): raw})
        assert torch.allclose(pack.layer.input_norm, layer.input_norm.to(dtype=torch.float32))
        assert torch.allclose(pack.layer.q, layer.q.to(dtype=torch.float32))
        gate, _up, _down = transport.take_expert(0, 0)
        assert gate.shape[0] == model.store.layout.intermediate
        copied = transport.h2d_now(0, 1, model.store.payload(0, 1))
        assert copied[0].shape == gate.shape
        assert transport.take_expert(0, 1)[0] is copied[0]
        transport.stage_ahead(1, model.weights.layers[1], {})
        nxt = transport.ensure(1, model.weights.layers[1], {})
        assert nxt.layer.o.shape == model.weights.layers[1].o.shape
        report = transport.resolve()
        assert report["h2d_bytes"] > 0
        assert report["h2d_s"] == 0.0
        transport.check_budget()
        transport.close()
        assert transport._ready is None
    finally:
        store.close()


def test_default_compute_stays_on_cpu(tmp_path: Path):
    model, store, _directory = _model(tmp_path)
    try:
        assert model.compute_device.type == "cpu"
        assert model.gpu is None
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
        loss, _plan = model.forward_window(tokens)
        assert loss.device.type == "cpu"
    finally:
        store.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sem CUDA neste host")
def test_gpu_step_loss_matches_cpu_within_bf16_tolerance(tmp_path: Path):
    from moe_memory.offload.lm_train import flush_lora_slice_grads

    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 1, 2, 3, 4]])
    cpu, cpu_store, _cpu_dir = _model(tmp_path / "cpu")
    gpu, gpu_store, _gpu_dir = _model(tmp_path / "gpu")
    try:
        gpu.set_compute_device(torch.device("cuda"), vram_budget=int(5 * 1024**3))
        loss_cpu, _plan = cpu.forward_window(tokens)
        loss_gpu, _plan = gpu.forward_window(tokens)
        tol = 1e-2 if gpu.compute_dtype == torch.bfloat16 else 1e-4
        assert torch.allclose(loss_gpu.cpu(), loss_cpu, atol=tol, rtol=tol)
        loss_cpu.backward()
        flush_lora_slice_grads()
        loss_gpu.backward()
        flush_lora_slice_grads()
        assert gpu.lora.gate_B.grad is not None
        assert gpu.lora.gate_B.grad.device.type == "cpu"
        assert gpu.router.lstm.weight_ih_l0.grad.device.type == "cpu"
        assert torch.allclose(gpu.lora.gate_B.grad, cpu.lora.gate_B.grad, atol=tol * 2, rtol=tol * 2)
        assert int(torch.cuda.max_memory_allocated()) < int(5 * 1024**3)
    finally:
        gpu.release_expert_cache()
        cpu.release_expert_cache()
        cpu_store.close()
        gpu_store.close()


def test_one_token_attention_matches_decode_and_is_causal(tmp_path: Path):
    model, store, _directory = _model(tmp_path)
    try:
        layer = model.weights.layers[0]
        spec = model.weights.spec
        hidden = torch.randn(1, 1, spec.hidden_size)
        decoded = _attention(layer, hidden, KvCache(1), 0, 0, spec)
        sequenced = _attention_seq(layer, hidden, spec)
        assert torch.allclose(decoded, sequenced, atol=1e-5)
        longer = torch.randn(1, 4, spec.hidden_size)
        before = _attention_seq(layer, longer, spec)
        changed = longer.clone()
        changed[:, -1] += 3
        after = _attention_seq(layer, changed, spec)
        assert torch.allclose(before[:, 0], after[:, 0], atol=1e-5)
        assert not torch.allclose(before[:, -1], after[:, -1])
    finally:
        store.close()


def test_lm_step_updates_lstm_and_lora_and_leaves_the_store(tmp_path: Path):
    model, store, directory = _model(tmp_path)
    try:
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
        before_store = fingerprint(directory / "experts.moestore")
        before_resident = fingerprint(directory / "resident.bin")
        before_b = model.lora.gate_B.detach().clone()
        before_lstm = model.router.lstm.weight_ih_l0.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
        stats = lm_step(model, optimizer, tokens)
        assert stats["loss"] == stats["loss"]
        assert stats["grads_finite"] is True
        assert stats["lstm_grad_nonzero"] is True
        assert stats["lora_b_grad_nonzero"] is True
        assert stats["distinct_experts"] >= 1
        assert not torch.equal(model.lora.gate_B, before_b)
        assert not torch.equal(model.router.lstm.weight_ih_l0, before_lstm)
        assert fingerprint(directory / "experts.moestore") == before_store
        assert fingerprint(directory / "resident.bin") == before_resident
        assert all(not tensor.requires_grad for tensor in (model.weights.embed, model.weights.lm_head))
        names = set(model.state_dict())
        assert all(name.startswith("router.") or name.startswith("lora.") for name in names)
    finally:
        store.close()


def test_checkpoint_restores_routing_and_the_next_step(tmp_path: Path):
    model, store, directory = _model(tmp_path)
    try:
        first = torch.tensor([[4, 5, 6, 7]])
        second = torch.tensor([[8, 9, 10, 11]])
        optimizer = make_optimizer(model, 1e-2)
        lm_step(model, optimizer, first)
        path = directory / "checkpoint"
        save_checkpoint(path, step=1, model=model, optimizer=optimizer, extra={"windows": [first.view(-1).tolist()]})
        routed = model.router.plan(first)[0].hot_index.detach().clone()
        saved_avg = optimizer.state[model.lora.gate_B]["exp_avg"].detach().clone()
        other, other_store, _other_dir = _model(directory / "second")
        try:
            other_opt = make_optimizer(other, 1e-2)
            meta = load_checkpoint(path, other, other_opt)
            assert torch.equal(other.router.plan(first)[0].hot_index, routed)
            assert torch.equal(other_opt.state[other.lora.gate_B]["exp_avg"], saved_avg)
            again = lm_step(model, optimizer, second)
            resumed = lm_step(other, other_opt, second)
            assert again["loss"] == pytest.approx(resumed["loss"])
            assert torch.equal(model.lora.down_B, other.lora.down_B)
            assert meta["step"] == 1
            assert meta["kind"] == "qwen-lm-lora-lstm"
            assert (path / "meta.json").is_file()
            assert not (path / "checkpoint.pt").exists()
        finally:
            other_store.close()
    finally:
        store.close()


def _toy_tokenizer(directory: Path, *, bad_merge: bool = False) -> None:
    unicode_of = bytes_to_unicode()
    vocab = {char: index for index, char in enumerate(unicode_of.values())}
    left = unicode_of[ord("a")]
    right = unicode_of[ord("b")]
    vocab[left + right] = 300 if bad_merge else 256
    vocab["<|endoftext|>"] = 257
    blob = {
        "model": {"type": "BPE", "vocab": vocab, "merges": [f"{left} {right}"]},
        "added_tokens": [{"id": 257, "content": "<|endoftext|>", "special": True}],
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "tokenizer.json").write_text(json.dumps(blob), encoding="utf-8")
    (directory / "tokenizer_config.json").write_text(
        json.dumps({"eos_token": "<|endoftext|>", "tokenizer_class": "Qwen2Tokenizer"}),
        encoding="utf-8",
    )


def test_qwen_bpe_refuses_a_broken_merge_and_encodes_bytes(tmp_path: Path):
    good = tmp_path / "good"
    _toy_tokenizer(good)
    tokenizer = load_qwen_tokenizer(good)
    assert tokenizer.eos_id == 257
    assert tokenizer.vocab_size == 258
    encoded = tokenizer.encode("ab")
    assert encoded
    assert all(0 <= token < tokenizer.vocab_size for token in encoded)
    _toy_tokenizer(tmp_path / "bad", bad_merge=True)
    with pytest.raises(ValueError, match="buraco no BPE"):
        load_qwen_tokenizer(tmp_path / "bad")


def test_windows_keep_train_text_apart_from_validation(tmp_path: Path):
    _toy_tokenizer(tmp_path / "tok")
    tokenizer = load_qwen_tokenizer(tmp_path / "tok")
    root = tmp_path / "data" / "cache" / "Salesforce___wikitext" / "wikitext-2-raw-v1" / "hash"
    root.mkdir(parents=True)
    _write_arrow(root / "wikitext-train.arrow", ["Alpha beta.", "", "Gamma delta."])
    _write_arrow(root / "wikitext-validation.arrow", ["Valid only."])
    _write_arrow(root / "wikitext-test.arrow", ["Do not train."])
    train = wikitext_arrow_files(tmp_path / "data" / "cache", "train")
    valid = wikitext_arrow_files(tmp_path / "data" / "cache", "validation")
    assert len(train) == 1 and len(valid) == 1
    assert "test" not in train[0].name
    windows = list(iter_token_windows(iter_arrow_text(train), tokenizer, seq_len=4, limit=2))
    assert len(windows) == 2
    skipped = list(iter_token_windows(iter_arrow_text(train), tokenizer, seq_len=4, limit=1, skip=1))
    assert skipped == windows[1:]


def test_no_checkpoint_skips_files_and_marks_the_summary(tmp_path: Path):
    model, store, _directory = _model(tmp_path / "model")
    try:
        optimizer = make_optimizer(model, 1e-3)
        loss_before = lm_step(model, optimizer, torch.tensor([[1, 2, 3, 4, 5, 6]]))["loss"]
        def refuse_state(*_args, **_kwargs):
            raise AssertionError("não montar state_dict com --no-checkpoint")
        model.state_dict = refuse_state
        model.named_parameters = refuse_state
        model.lora.state_dict = refuse_state
        model.router.state_dict = refuse_state
        optimizer.state = refuse_state
        out = tmp_path / "out"
        saved = persist_checkpoint(
            False,
            out / "checkpoint",
            step=1,
            model=model,
            optimizer=optimizer,
            extra={"seq_len": 6},
        )
        assert saved is False
        out.mkdir()
        summary = {"checkpoint_saved": saved, "checkpoint": None, "steps": [{"loss": loss_before}]}
        (out / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        loaded = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        assert loaded["checkpoint_saved"] is False
        assert loaded["checkpoint"] is None
        assert loaded["steps"][0]["loss"] == loss_before
        assert not (out / "checkpoint").exists()
        assert not (out / "checkpoint.pt").exists()
        assert list(out.rglob("*.bin")) == []
    finally:
        store.close()


def test_default_checkpoint_still_saves(tmp_path: Path):
    model, store, _directory = _model(tmp_path / "model")
    try:
        optimizer = make_optimizer(model, 1e-3)
        stats = lm_step(model, optimizer, torch.tensor([[1, 2, 3, 4, 5, 6]]))
        out = tmp_path / "out"
        checkpoint = out / "checkpoint"
        saved = persist_checkpoint(
            True,
            checkpoint,
            step=1,
            model=model,
            optimizer=optimizer,
            extra={"seq_len": 6},
        )
        assert saved is True
        assert (checkpoint / "meta.json").is_file()
        assert list((checkpoint / "weights").glob("*.bin"))
        summary = {"checkpoint_saved": saved, "checkpoint": str(checkpoint), "steps": [{"loss": stats["loss"]}]}
        (out / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        loaded = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        assert loaded["checkpoint_saved"] is True
        assert loaded["steps"][0]["loss"] == stats["loss"]
    finally:
        store.close()


def _write_arrow(path: Path, rows: list[str]) -> None:
    table = pa.table({"text": rows})
    with pa.OSFile(str(path), "wb") as handle:
        with ipc.new_stream(handle, table.schema) as writer:
            writer.write_table(table)
