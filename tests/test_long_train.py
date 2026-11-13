"""Treino longo: hold-out, gate do Qwen e checkpoint só do que treina."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from moe_memory.offload.lm_train import (
    evaluate_windows,
    lm_step,
    make_optimizer,
    replace_checkpoint,
    scheduled_lr,
)
from moe_memory.offload.qwen import load_resident, load_spec
from moe_memory.offload.qwen_gate import QwenGateRouter, route_qwen
from moe_memory.offload.store import ExpertStore
from moe_memory.qwen_corpus import HOLDOUT_SPLIT, holdout_note, iter_arrow_text, iter_token_windows, wikitext_arrow_files
from tests.test_qwen_lm_train import _bundle, _model, _write_arrow


class _Tok:
    eos_id = 9

    def encode(self, text: str) -> list[int]:
        return [1 + (ord(char) % 7) for char in text if char != " "]


def test_holdout_windows_are_disjoint_and_stable(tmp_path: Path):
    root = tmp_path / "cache" / "Salesforce___wikitext" / "wikitext-2-raw-v1" / "hash"
    root.mkdir(parents=True)
    _write_arrow(root / "wikitext-train.arrow", ["alpha beta.", "", "gamma delta."])
    _write_arrow(root / "wikitext-validation.arrow", ["valid only.", "", "hold out."])
    train_files = wikitext_arrow_files(tmp_path / "cache", "train")
    hold_files = wikitext_arrow_files(tmp_path / "cache", HOLDOUT_SPLIT)
    tok = _Tok()
    train = list(iter_token_windows(iter_arrow_text(train_files), tok, seq_len=4, limit=8))
    hold = list(iter_token_windows(iter_arrow_text(hold_files), tok, seq_len=4, limit=8))
    again = list(iter_token_windows(iter_arrow_text(hold_files), tok, seq_len=4, limit=8))
    assert train and hold
    assert again == hold
    assert set(map(tuple, train)).isdisjoint(map(tuple, hold))
    assert "validation" in holdout_note()
    assert "test" in holdout_note()


def test_eval_is_deterministic_and_has_no_gradient(tmp_path: Path):
    model, store, _directory = _model(tmp_path)
    try:
        windows = [[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]]
        first = evaluate_windows(model, windows)
        second = evaluate_windows(model, windows)
        assert first == second
        assert first["eval_tokens"] == 10
        assert first["eval_perplexity"] == math.exp(first["eval_loss"])
        assert all(parameter.grad is None for parameter in model.parameters())
    finally:
        store.close()


def test_lm_step_grad_report_works_with_qwen_router_without_lstm(tmp_path: Path):
    directory = _bundle(tmp_path / "bundle")
    store = ExpertStore.open(directory / "experts.moestore", direct=True)
    try:
        spec = load_spec(directory / "config.json", store.header.dtype_code)
        weights = load_resident(directory / "resident.bin", spec, torch.device("cpu"), with_gate=True)
        router = QwenGateRouter(
            vocab_size=spec.vocab_size,
            n_layers=spec.n_layers,
            n_experts=spec.n_experts,
            experts_per_tok=4,
        )
        from moe_memory.offload.lm_train import OffloadLanguageModel

        model = OffloadLanguageModel(weights, router, store, rank=2, alpha=4.0, use_lora=True)
        optimizer = make_optimizer(model, 1e-3)
        stats = lm_step(model, optimizer, torch.tensor([[1, 2, 3, 4, 5, 6]]))
        assert stats["grads_finite"] is True
        assert stats["lstm_grad_norm"] == 0.0
        assert stats["lstm_grad_nonzero"] is False
        assert stats["lora_b_grad_nonzero"] is True
    finally:
        store.close()


def test_qwen_router_does_not_build_an_lstm(tmp_path: Path):
    directory = _bundle(tmp_path / "bundle")
    store = ExpertStore.open(directory / "experts.moestore", direct=True)
    try:
        spec = load_spec(directory / "config.json", store.header.dtype_code)
        weights = load_resident(directory / "resident.bin", spec, torch.device("cpu"), with_gate=True)
        router = QwenGateRouter(
            vocab_size=spec.vocab_size,
            n_layers=spec.n_layers,
            n_experts=spec.n_experts,
            experts_per_tok=spec.experts_per_tok,
        )
        from moe_memory.offload.lm_train import OffloadLanguageModel

        with_lora, bare = (
            OffloadLanguageModel(weights, router, store, rank=2, alpha=4.0, use_lora=True),
            OffloadLanguageModel(weights, router, store, rank=2, alpha=4.0, use_lora=False),
        )
        assert not hasattr(router, "lstm")
        assert not any("lstm" in name for name, _parameter in with_lora.named_parameters())
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
        loss, plan = with_lora.forward_window(tokens)
        base, _plan = bare.forward_window(tokens)
        assert torch.equal(loss, base)
        assert plan.hot_index.shape[-1] == spec.experts_per_tok
        assert int(plan.hot_valid.sum()) == tokens.shape[1] * spec.n_layers * spec.experts_per_tok
    finally:
        store.close()


def test_qwen_route_renormalizes_the_topk():
    hidden = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    gate = torch.tensor([[2.0, 0.0], [1.0, 0.0], [0.0, 3.0], [0.0, 1.0]])
    index, valid, coeffs = route_qwen(hidden, gate, k=2)
    assert index[0, 0].tolist() == [0, 1]
    assert index[0, 1].tolist() == [2, 3]
    assert bool(valid.all())
    assert torch.allclose(coeffs.sum(dim=-1), torch.ones(1, 2))


def test_schedule_is_constant_by_default_and_cosine_ends_at_the_floor():
    assert scheduled_lr(3, 10, 1e-4) == 1e-4
    assert scheduled_lr(1, 10, 1e-4, warmup=2) == 5e-5
    assert scheduled_lr(10, 10, 1e-4, warmup=2, schedule="cosine", lr_min=0.0) == 0.0


def test_rotating_checkpoint_keeps_trainable_state_and_resumes(tmp_path: Path):
    first, first_store, _left = _model(tmp_path / "first")
    second, second_store, _right = _model(tmp_path / "second")
    try:
        opt_first = make_optimizer(first, 1e-2)
        opt_second = make_optimizer(second, 1e-2)
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
        lm_step(first, opt_first, tokens)
        path = tmp_path / "checkpoint"
        replace_checkpoint(path, step=1, model=first, optimizer=opt_first, extra={"seed": 0})
        (path / "marker.txt").write_text("velho", encoding="utf-8")
        lm_step(first, opt_first, tokens)
        replace_checkpoint(path, step=2, model=first, optimizer=opt_first, extra={"seed": 0})
        assert not (tmp_path / ".checkpoint.tmp").exists()
        assert not (path / "marker.txt").exists()
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        names = [entry["name"] for entry in meta["entries"] if entry["kind"] == "param"]
        assert names
        assert all(name.startswith("lora.") or name.startswith("router.") for name in names)
        from moe_memory.offload.lm_train import load_checkpoint

        load_checkpoint(path, second, opt_second)
        for (name, kept), (other, loaded) in zip(first.named_parameters(), second.named_parameters(), strict=True):
            assert name == other
            assert torch.equal(kept, loaded)
        left = lm_step(first, opt_first, tokens)
        right = lm_step(second, opt_second, tokens)
        assert left["loss"] == right["loss"]
        for (name, kept), (other, loaded) in zip(first.named_parameters(), second.named_parameters(), strict=True):
            assert name == other
            assert torch.equal(kept, loaded)
    finally:
        first_store.close()
        second_store.close()
