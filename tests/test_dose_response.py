"""Dose–resposta no hold-out. CPU, sem treino e sem o modelo de 48 M.

O laço de ``eval.B_mult`` e as chamadas a ``set_hot_budget`` são um
emulador espião. O emulador real entra só no teste do carimbo do LRU.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    path = ROOT / "experiments" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


holdout = _load("holdout_dose_under_test", "run_holdout_eval.py")
import dose_response  # noqa: E402  — o hold-out acima põe experiments/ no path
from moe_memory.emulator import ResidencyEmulator, StepStats  # noqa: E402
from moe_memory.levels import HOT  # noqa: E402


class _SpyEmulator:
    def __init__(self, hot_slots: int = 8) -> None:
        self.hot_slots = hot_slots
        self.calls: list[tuple[str, int]] = []
        self.tick = 7
        self.last_used = object()
        self.last_count = object()
        self._store = None
        self.work_dir = "nao-criar"

    def reset(self) -> None:
        self.calls.append(("reset", self.hot_slots))
        self.tick = 0
        self.last_used = object()
        self.last_count = object()

    def set_hot_budget(self, hot_slots: int) -> StepStats:
        slots = int(hot_slots)
        self.calls.append(("set_hot_budget", slots))
        self.hot_slots = slots
        return StepStats(swaps=slots, bytes_moved=slots, promotions=0, demotions=0)


def _row(hot_slots: int) -> dict:
    return {
        "perplexity": 10.0 + hot_slots,
        "nll_nats": 2.0,
        "nll_nats_sum": 14.0,
        "bits_per_byte": 1.0,
        "tokens_scored": 7,
        "tokens_packed": 8,
        "sequences_scored": 1,
        "sequences_available": 4,
        "tokens_discarded": 0,
        "documents": 1,
        "utf8_bytes": 10,
        "source_utf8_bytes": 20,
        "truncated": True,
        "logical_batches": 1,
        "c_mem": 0.1,
        "balance": 1.0,
        "entropy": 0.2,
        "aggregation": "sum",
        "swaps": 0,
        "bytes_moved": 0,
        "promotions": 0,
        "demotions": 0,
        "hot": hot_slots,
        "warm": 8,
        "cold": 1,
        "lru_tick_end": 1,
    }


def _measure(monkeypatch, config, *, hot_slots: int = 8):
    emulator = _SpyEmulator(hot_slots)
    sentinel = object()
    scored: list[tuple[str, int]] = []

    def fake_pack(_cache, _config, split, _seq_len):
        return SimpleNamespace(split=split)

    def fake_score(model, packed, **kwargs):
        assert model is sentinel
        assert kwargs["reset_lru"] is False
        assert kwargs["emulator"] is emulator
        assert emulator.calls[-2] == ("reset", config.memory.hot_slots(config.model.E))
        assert emulator.calls[-1] == ("set_hot_budget", emulator.hot_slots)
        scored.append((packed.split, emulator.hot_slots))
        return _row(emulator.hot_slots)

    monkeypatch.setattr(holdout, "_cached_pack", fake_pack)
    monkeypatch.setattr(holdout, "score_split", fake_score)
    args = holdout.parse_args(["--splits", "valid,test", "--max-batches", "1"])
    record = holdout.measure_dose_response(
        kind=holdout.KIND_MOE,
        model=sentinel,
        emulator=emulator,
        config=config,
        args=args,
        device=torch.device("cpu"),
        logical_batch=2,
        forward_batch=1,
        lambda_weight=0.0,
        packed_cache={},
        seq_len=8,
    )
    return record, emulator, scored


def test_b_mult_loop_calls_set_hot_budget_after_reset_at_yaml_b(monkeypatch):
    from moe_memory.config import load_config

    config = load_config()
    record, emulator, scored = _measure(monkeypatch, config)
    assert scored == [
        ("valid", 4),
        ("test", 4),
        ("valid", 8),
        ("test", 8),
        ("valid", 16),
        ("test", 16),
    ]
    resets = [slots for name, slots in emulator.calls if name == "reset"]
    assert resets
    assert set(resets) == {8}
    measured = [slots for name, slots in emulator.calls if name == "set_hot_budget"]
    assert measured[:1] == [4]
    assert {4, 8, 16} <= set(measured)
    assert record["status"] == "ok"
    assert record["retrained"] is False
    assert record["method"] == "reset_lru_at_yaml_b_then_set_hot_budget"
    assert [point["multiplier"] for point in record["budgets"]] == [0.5, 1.0, 2.0]
    assert [point["hot_slots"] for point in record["budgets"]] == [4, 8, 16]
    for point in record["budgets"]:
        for split in ("valid", "test"):
            row = point["splits"][split]
            assert row["budget_reassign"]["swaps"] == point["hot_slots"]
            assert row["perplexity"] == 10.0 + point["hot_slots"]


def test_b_mult_order_comes_from_the_yaml(monkeypatch, tmp_path: Path):
    from moe_memory.config import load_config

    text = (ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
    text = text.replace("B_mult: [0.5, 1.0, 2.0]", "B_mult: [2.0, 0.5]")
    path = tmp_path / "flipped.yaml"
    path.write_text(text, encoding="utf-8")
    config = load_config(path)
    assert config.eval.B_mult == (2.0, 0.5)
    record, emulator, scored = _measure(monkeypatch, config)
    assert [hot for _split, hot in scored] == [16, 16, 4, 4]
    assert 8 not in [hot for _split, hot in scored]
    assert [point["multiplier"] for point in record["budgets"]] == [2.0, 0.5]
    assert [point["hot_slots"] for point in record["budgets"]] == [16, 4]
    assert ("set_hot_budget", 16) in emulator.calls
    assert ("set_hot_budget", 4) in emulator.calls
    assert ("reset", 8) in emulator.calls


def test_dense_dose_response_is_na_and_does_not_touch_the_emulator():
    from moe_memory.config import load_config

    class Boom:
        def set_hot_budget(self, _slots):
            raise AssertionError("denso chamou set_hot_budget")

        def reset(self):
            raise AssertionError("denso deu reset no LRU")

    config = load_config()
    args = holdout.parse_args([])
    record = holdout.measure_dose_response(
        kind=holdout.KIND_DENSE,
        model=object(),
        emulator=Boom(),
        config=config,
        args=args,
        device=torch.device("cpu"),
        logical_batch=2,
        forward_batch=2,
        lambda_weight=None,
        packed_cache={},
        seq_len=8,
    )
    assert record["status"] == "n/a"
    assert record["budgets"] is None
    assert record["retrained"] is False
    assert record["method"] is None
    assert "set_hot_budget" in record["reason"]


def test_set_hot_budget_keeps_the_reset_stamps(tmp_path: Path):
    emulator = ResidencyEmulator(
        n_layers=1,
        n_experts=32,
        hot_slots=8,
        warm_slots=8,
        expert_nbytes=16,
        work_dir=tmp_path,
    )
    dispatched = torch.zeros(1, 32, dtype=torch.long)
    dispatched[0, 31] = 5
    emulator.step(dispatched)
    assert emulator.tick == 1
    assert emulator.physical_store is None
    stats = dose_response.prepare_split_budget(emulator, 8, 4)
    assert stats.swaps > 0
    assert emulator.tick == 0
    assert emulator.hot_slots == 4
    assert int((emulator.levels == HOT).sum()) == 4
    assert int(emulator.last_used[0, 0]) == 0
    stamps = emulator.last_used
    counts = emulator.last_count
    emulator.set_hot_budget(16)
    assert emulator.last_used is stamps
    assert emulator.last_count is counts
    assert emulator.tick == 0
    assert emulator.hot_slots == 16
    assert int((emulator.levels == HOT).sum()) == 16
    assert emulator.warm_slots == 8
    assert emulator.physical_store is None
    assert emulator.work_dir == tmp_path
    assert list(tmp_path.glob("**/*.f32")) == []


def test_prepare_restores_yaml_b_with_set_hot_budget_then_resets(tmp_path: Path):
    emulator = ResidencyEmulator(
        n_layers=1,
        n_experts=32,
        hot_slots=8,
        warm_slots=8,
        expert_nbytes=16,
        work_dir=tmp_path,
    )
    emulator.set_hot_budget(16)
    calls: list[tuple] = []
    real_set = emulator.set_hot_budget
    real_reset = emulator.reset

    def spy_set(slots: int):
        calls.append(("set_hot_budget", int(slots)))
        return real_set(slots)

    def spy_reset():
        calls.append(("reset", emulator.hot_slots))
        real_reset()

    emulator.set_hot_budget = spy_set
    emulator.reset = spy_reset
    dose_response.prepare_split_budget(emulator, 8, 4)
    assert calls == [
        ("set_hot_budget", 8),
        ("reset", 8),
        ("set_hot_budget", 4),
    ]
    assert emulator.hot_slots == 4
    assert emulator.tick == 0
    assert int(emulator.last_used[0, 0]) == 0
    assert int((emulator.levels[0] == HOT).sum()) == 4


def test_table_writes_ppl_and_marks_dense_na(tmp_path: Path):
    from moe_memory.config import load_config

    config = load_config()
    summary = tmp_path / "holdout" / "summary.json"
    args = holdout.parse_args(["--summary", str(summary), "--outputs", str(tmp_path / "outputs")])

    def split_row(perplexity: float) -> dict:
        return {
            "perplexity": perplexity,
            "nll_nats": 2.0,
            "nll_nats_sum": 14.0,
            "bits_per_byte": 1.25,
            "tokens_scored": 7,
            "truncated": False,
        }

    dense = {
        "name": "baseline",
        "status": "ok",
        "path": str(tmp_path / "baseline" / "wikitext" / "checkpoint.pt"),
        "model_kind": "dense",
        "lambda": None,
        "context_matches_checkpoint": True,
        "lambda_in_grid": None,
        "splits": {"valid": split_row(40.0), "test": split_row(41.0)},
        "dose_response": dose_response.dense_record(config),
    }
    budgets = []
    for multiplier, slots, perplexity in ((0.5, 4, 12.5), (1.0, 8, 11.0), (2.0, 16, 10.25)):
        budgets.append(
            {
                "multiplier": multiplier,
                "hot_slots": slots,
                "yaml_reference": slots == 8,
                "splits": {
                    "valid": split_row(perplexity),
                    "test": split_row(perplexity + 0.5),
                },
            }
        )
    moe = {
        "name": "lambda-0.5",
        "status": "ok",
        "path": str(tmp_path / "wikitext" / "lambda-0.5" / "checkpoint.pt"),
        "model_kind": "moe",
        "lambda": 0.5,
        "context_matches_checkpoint": True,
        "lambda_in_grid": True,
        "splits": {"valid": split_row(11.0), "test": split_row(11.5)},
        "dose_response": {
            "status": "ok",
            "reason": None,
            "reference_hot_slots": 8,
            "method": "reset_lru_at_yaml_b_then_set_hot_budget",
            "retrained": False,
            "budgets": budgets,
        },
    }
    families = [
        holdout.Family("baseline", Path(dense["path"]), expected_kind="dense"),
        holdout.Family("lambda-0.5", Path(moe["path"]), expected_kind="moe", expected_lambda=0.5),
    ]
    written = holdout._write_outputs(args, config, families, [dense, moe], interrupted=False)
    dose_json = summary.parent / "dose_response.json"
    dose_md = summary.parent / "dose_response.md"
    assert dose_json in written
    assert dose_md in written
    payload = json.loads(dose_json.read_text(encoding="utf-8"))
    assert payload["retrained"] is False
    assert payload["protocol"] == "wikitext-2-dose-response"
    assert payload["B_reference_slots"] == 8
    assert payload["B_mult"] == [0.5, 1.0, 2.0]
    by_name = {run["name"]: run for run in payload["runs"]}
    assert by_name["baseline"]["dose_response"]["status"] == "n/a"
    assert by_name["baseline"]["dose_response"]["budgets"] is None
    assert by_name["lambda-0.5"]["dose_response"]["budgets"][1]["splits"]["valid"]["perplexity"] == 11.0
    table = dose_md.read_text(encoding="utf-8")
    assert "| baseline | dense | — | valid | n/a | n/a | n/a |" in table
    assert "| lambda-0.5 | moe | 0.5 | valid | 12.5000 | 11.0000 | 10.2500 |" in table
    assert "| lambda-0.5 | moe | 0.5 | test | 13.0000 | 11.5000 | 10.7500 |" in table
    assert "0.5×B (4)" in table
    assert "1×B (8)" in table
    assert "2×B (16)" in table
    holdout_summary = json.loads(summary.read_text(encoding="utf-8"))
    assert holdout_summary["dose_response"] is True
    assert holdout_summary["retrained"] is False
    assert holdout_summary["runs"][0]["dose_response"]["status"] == "n/a"


def test_single_checkpoint_without_summary_writes_the_table_beside_it(tmp_path: Path):
    from moe_memory.config import load_config

    config = load_config()
    checkpoint = tmp_path / "wikitext" / "lambda-0" / "checkpoint.pt"
    args = holdout.parse_args(["--checkpoint", str(checkpoint)])
    family = holdout.Family("lambda-0", checkpoint, expected_kind="moe", expected_lambda=0.0)
    row = {
        "perplexity": 3.0,
        "nll_nats": 1.0,
        "nll_nats_sum": 7.0,
        "bits_per_byte": 0.5,
        "tokens_scored": 7,
        "truncated": False,
    }
    run = {
        "name": "lambda-0",
        "status": "ok",
        "path": str(checkpoint),
        "model_kind": "moe",
        "lambda": 0.0,
        "context_matches_checkpoint": True,
        "lambda_in_grid": True,
        "splits": {"valid": dict(row), "test": dict(row)},
        "dose_response": {
            "status": "ok",
            "reason": None,
            "reference_hot_slots": 8,
            "method": "reset_lru_at_yaml_b_then_set_hot_budget",
            "retrained": False,
            "budgets": [
                {
                    "multiplier": multiplier,
                    "hot_slots": slots,
                    "yaml_reference": slots == 8,
                    "splits": {"valid": dict(row), "test": dict(row)},
                }
                for multiplier, slots in ((0.5, 4), (1.0, 8), (2.0, 16))
            ],
        },
    }
    holdout._write_outputs(args, config, [family], [run], interrupted=False)
    assert not (tmp_path / "outputs" / "holdout" / "summary.json").exists()
    assert (checkpoint.parent / "dose_response.json").is_file()
    assert (checkpoint.parent / "dose_response.md").is_file()
    assert "lambda-0" in (checkpoint.parent / "dose_response.md").read_text(encoding="utf-8")
    saved = json.loads((checkpoint.parent / "holdout.json").read_text(encoding="utf-8"))
    assert saved["runs"][0]["dose_response"]["status"] == "ok"
    assert saved["retrained"] is False
