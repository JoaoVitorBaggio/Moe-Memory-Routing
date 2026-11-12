"""Comparação do paper: top-k, VRAM do gate e janelas iguais."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from moe_memory.offload.lm_gpu import QWEN_VRAM_FIT_SLACK, QWEN_VRAM_MEASURED_PEAK, fit_vram_cache
from moe_memory.offload.lm_train import (
    estimate_bytes,
    evaluate_windows,
    load_checkpoint,
    lm_step,
    make_optimizer,
    replace_checkpoint,
)
from moe_memory.offload.qwen_gate import route_qwen
from moe_memory.qwen_corpus import paper_windows, target_tokens, windows_sha256
from tests.test_qwen_lm_train import _model

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Tok:
    eos_id = 2

    def encode(self, text: str) -> list[int]:
        return [3 + (ord(char) % 5) for char in text if char != " "]


def test_qwen_topk_renormalizes_and_matches_the_full_softmax():
    torch.manual_seed(0)
    hidden = torch.randn(1, 3, 6)
    gate = torch.randn(8, 6)
    full = torch.softmax(torch.nn.functional.linear(hidden[0], gate), dim=-1)
    for k in (4, 8):
        index, valid, coeffs = route_qwen(hidden, gate, k)
        assert bool(valid.all())
        assert torch.allclose(coeffs.sum(-1), torch.ones(1, 3))
        picked = full.gather(1, index[0])
        assert torch.allclose(coeffs[0], picked / picked.sum(-1, keepdim=True))
    top8, _, _ = route_qwen(hidden, gate, 8)
    top4, _, _ = route_qwen(hidden, gate, 4)
    assert top4[0].tolist() == [row[:4] for row in top8[0].tolist()]


def test_qwen_cache_shrinks_to_the_measured_budget_and_lstm_does_not():
    requested = 3 * 1024**3
    budget = 5 * 1024**3
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
        k2=6,
        compute_cuda=True,
        compute_elem=4,
        vram_expert_cache_bytes=requested,
        lm_head_vram=True,
        vram_cache_elem=2,
    )
    qwen = estimate_bytes(**common, vram_k=8, trainable_lstm=False)
    used = fit_vram_cache(int(qwen["vram_bytes"]), requested, budget, QWEN_VRAM_FIT_SLACK)
    assert used < requested
    assert int(qwen["vram_bytes"]) + QWEN_VRAM_FIT_SLACK - (requested - used) <= budget
    predicted = QWEN_VRAM_MEASURED_PEAK - (requested - used)
    assert predicted <= budget
    lstm = estimate_bytes(**common)
    assert fit_vram_cache(int(lstm["vram_bytes"]), requested, budget, 0) == requested


def test_paper_windows_are_stable_and_cover_the_token_budget():
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa " * 40
    windows = paper_windows(iter([text]), _Tok(), context_len=8, eval_tokens=30)
    again = paper_windows(iter([text]), _Tok(), context_len=8, eval_tokens=30)
    assert windows == again
    assert windows_sha256(windows) == windows_sha256(again)
    assert target_tokens(windows) >= 30
    wider = paper_windows(iter([text]), _Tok(), context_len=16, eval_tokens=30)
    assert windows_sha256(wider) != windows_sha256(windows)


def test_loaded_checkpoint_matches_the_in_memory_eval(tmp_path: Path):
    live, live_store, _left = _model(tmp_path / "live")
    restored, restored_store, _right = _model(tmp_path / "restored")
    try:
        optimizer = make_optimizer(live, 1e-2)
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
        lm_step(live, optimizer, tokens)
        windows = [[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]]
        before = evaluate_windows(live, windows)
        path = tmp_path / "checkpoint"
        replace_checkpoint(path, step=1, model=live, optimizer=optimizer, extra={"seed": 0})
        other = make_optimizer(restored, 1e-2)
        load_checkpoint(path, restored, other)
        after = evaluate_windows(restored, windows)
        assert before["eval_loss"] == after["eval_loss"]
        assert before["eval_tokens"] == after["eval_tokens"]
    finally:
        live_store.close()
        restored_store.close()


def test_night_batch_dry_run_lists_every_stage_without_touching_the_store():
    night = _load("night_batch_under_test", "night_batch.py")
    dense = _load("eval_dense_under_test", "eval_dense.py")
    steps = night.build_steps()
    text = night.render(steps)
    joined = "\n".join(" ".join(night.launch(step["cmd"])) for step in steps)
    assert "a-val-32" in text and "c-train-topk4" in text and "e-4b" in text
    assert "systemd-run" in joined
    assert "MemoryMax=9663676416" in joined
    assert "MemorySwapMax=0" in joined
    assert "OOMPolicy=stop" in joined
    assert "IOSchedulingClass=idle" in joined
    assert "MALLOC_ARENA_MAX=2" in joined
    assert "--qwen-topk 4" in joined
    assert "--qwen-topk 8" in joined
    assert "models/baselines/Qwen3-1.7B" in joined
    assert "rm " not in joined
    for step in steps:
        command = step["cmd"]
        for flag in ("--out", "--dest", "--windows-out"):
            if flag in command:
                assert "models/qwen3-30b-a3b" not in command[command.index(flag) + 1]
    disk = dense.DENSE_MODELS["Qwen/Qwen3-1.7B"]["disk_bytes"] + dense.DENSE_MODELS["Qwen/Qwen3-4B"]["disk_bytes"]
    assert disk < 43 * 1024**3
