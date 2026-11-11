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


def _write_arrow(path: Path, rows: list[str]) -> None:
    table = pa.table({"text": rows})
    with pa.OSFile(str(path), "wb") as handle:
        with ipc.new_stream(handle, table.schema) as writer:
            writer.write_table(table)
