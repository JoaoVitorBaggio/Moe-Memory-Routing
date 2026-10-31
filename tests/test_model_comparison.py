"""Manifesto dos quatro braços. CPU, sem treino e sem checkpoint real.

O modelo 2 não ganha política nova: o teste trava o rótulo, o schema e o
fato de que a linha copia o λ=0 do mesmo ResidencyEmulator.
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from moe_memory.comparison import (
    CODE_PATH_DENSE,
    CODE_PATH_LAMBDA0,
    POLICY_CLASS,
    SPLITS,
    TABLE_COLUMNS,
    WALL_FIELDS,
    ComparisonError,
    build_comparison,
    lambda_tag,
)
from moe_memory.config import load_config
from moe_memory.emulator import ResidencyEmulator

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments" / "run_model_comparison.py"


def _holdout_module():
    path = ROOT / "experiments" / "run_holdout_eval.py"
    spec = importlib.util.spec_from_file_location("holdout_for_comparison", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _arm(payload: dict, arm_id: str) -> dict:
    return next(arm for arm in payload["arms"] if arm["id"] == arm_id)


def _rows(payload: dict, arm_id: str, **filters):
    found = []
    for row in payload["rows"]:
        if row["arm_id"] != arm_id:
            continue
        if any(row[key] != value for key, value in filters.items()):
            continue
        found.append(row)
    return found


def _split(perplexity: float, swaps: int | None, *, kind: str) -> dict:
    row = {
        "perplexity": perplexity,
        "nll_nats": math.log(perplexity),
        "bits_per_byte": 0.4,
        "truncated": False,
        "swaps": None,
        "bytes_moved": None,
        "promotions": None,
        "demotions": None,
        "entropy": None,
        "c_mem": None,
    }
    if kind == "moe":
        assert swaps is not None
        row.update(
            swaps=swaps,
            bytes_moved=swaps * 8,
            promotions=1,
            demotions=1,
            entropy=0.2,
            c_mem=0.1,
        )
    return row


def _run(name: str, kind: str, lam: float | None, perplexity: float, swaps: int | None) -> dict:
    run = {
        "name": name,
        "status": "ok",
        "model_kind": kind,
        "lambda": lam,
        "splits": {split: _split(perplexity, swaps, kind=kind) for split in SPLITS},
    }
    if kind == "moe":
        run["residency"] = "lru_reset_per_split"
        run["memory_path"] = True
    else:
        run["memory_path"] = False
    return run


def _holdout(runs: list[dict]) -> dict:
    return {
        "protocol": "wikitext-2-holdout-ppl",
        "dose_response": False,
        "residency": "lru_reset_per_split",
        "runs": runs,
    }


def _full_holdout() -> dict:
    return _holdout(
        [
            _run("baseline", "dense", None, 20.0, None),
            _run("lambda-0", "moe", 0.0, 10.0, 3),
            _run("lambda-0.5", "moe", 0.5, 11.0, 4),
            _run("lambda-1", "moe", 1.0, 12.0, 5),
        ]
    )


def _dose_point(multiplier: float, hot_slots: int, perplexity: float, swaps: int) -> dict:
    return {
        "multiplier": multiplier,
        "hot_slots": hot_slots,
        "yaml_reference": multiplier == 1.0,
        "splits": {split: _split(perplexity, swaps, kind="moe") for split in SPLITS},
    }


def _moe_curve(lam: float, swaps: int, reference_ppl: float, *, clash: float | None = None) -> dict:
    one = reference_ppl if clash is None else clash
    return {
        "status": "ok",
        "reason": None,
        "reference_hot_slots": 8,
        "method": "reset_lru_at_yaml_b_then_set_hot_budget",
        "retrained": False,
        "budgets": [
            _dose_point(0.5, 4, 30.0 + lam, swaps),
            _dose_point(1.0, 8, one, swaps),
            _dose_point(2.0, 16, 8.0 + lam, swaps),
        ],
    }


def _dense_curve() -> dict:
    return {
        "status": "n/a",
        "reason": "O LM denso não tem residência nem teto quente.",
        "reference_hot_slots": 8,
        "method": None,
        "retrained": False,
        "budgets": None,
    }


def _dose_run(name: str, kind: str, lam: float | None, curve: dict) -> dict:
    return {
        "name": name,
        "status": "ok",
        "model_kind": kind,
        "lambda": lam,
        "dose_response": curve,
    }


def _full_dose(*, clash_lambda0: float | None = None) -> dict:
    runs = [
        _dose_run("baseline", "dense", None, _dense_curve()),
        _dose_run(
            "lambda-0",
            "moe",
            0.0,
            _moe_curve(0.0, 3, 10.0, clash=clash_lambda0),
        ),
        _dose_run("lambda-0.5", "moe", 0.5, _moe_curve(0.5, 4, 11.0)),
        _dose_run("lambda-1", "moe", 1.0, _moe_curve(1.0, 5, 12.0)),
    ]
    return {
        "protocol": "wikitext-2-dose-response",
        "retrained": False,
        "residency": "lru_reset_per_split_then_set_hot_budget",
        "dense_axis": "n/a",
        "runs": runs,
    }


def _holdout_with_curve(*, clash_lambda0: float | None = None) -> dict:
    payload = _full_holdout()
    payload["dose_response"] = True
    curves = {
        "baseline": _dense_curve(),
        "lambda-0": _moe_curve(0.0, 3, 10.0, clash=clash_lambda0),
        "lambda-0.5": _moe_curve(0.5, 4, 11.0),
        "lambda-1": _moe_curve(1.0, 5, 12.0),
    }
    for run in payload["runs"]:
        run["dose_response"] = curves[run["name"]]
    return payload


def test_four_arms_are_labeled_and_model2_is_not_a_new_policy():
    payload = build_comparison(load_config(), outputs=Path("outputs"))
    assert [arm["id"] for arm in payload["arms"]] == ["model1", "model2", "model3", "model4"]
    assert _arm(payload, "model1")["label"] == "MoE top-k λ=0"
    assert _arm(payload, "model2")["label"] == "MoE + LRU adversary"
    assert _arm(payload, "model3")["label"] == "Denso"
    assert _arm(payload, "model4")["label"] == "MoE λ>0"
    assert payload["identity"]["model1_equals_model2"] is True
    assert payload["identity"]["second_eviction_policy_class"] is None
    assert payload["identity"]["eviction_policy_class"] == "ResidencyEmulator"
    assert payload["identity"]["eviction_algorithm"] == "lru"
    assert payload["identity"]["dense_is_moe_lambda0"] is False
    assert payload["produces_measurements"] is False
    assert payload["trained_by_this_module"] is False
    assert payload["evaluated_by_this_module"] is False
    assert "segunda política" in payload["identity"]["note"]
    assert hasattr(ResidencyEmulator, payload["dose_hook"]["method"])
    assert payload["dose_hook"]["method"] == "set_hot_budget"
    assert payload["dose_hook"]["procedure"] == "reset_lru_at_yaml_b_then_set_hot_budget"
    assert payload["dose_hook"]["protocol"] == "wikitext-2-dose-response"
    assert payload["dose_hook"]["retrained"] is False
    assert payload["dose_hook"]["changes_policy"] is False


def test_model2_shares_the_lambda0_checkpoint_and_code_path():
    outputs = Path("outputs")
    payload = build_comparison(load_config(), outputs=outputs)
    model1 = _arm(payload, "model1")
    model2 = _arm(payload, "model2")
    assert model1["code_path"] == model2["code_path"] == CODE_PATH_LAMBDA0
    assert model1["checkpoint"] == model2["checkpoint"]
    assert model1["checkpoint"].endswith("outputs/wikitext/seed-0/lambda-0/checkpoint.pt")
    assert payload["checkpoint_seed"] == 0
    assert payload["identity"]["checkpoint_seed"] == 0
    assert model1["lambda"] == model2["lambda"] == 0.0
    assert model1["policy_class"] == model2["policy_class"] == POLICY_CLASS
    assert model2["same_code_path_as"] == "model1"
    assert model2["metrics_shared_with"] == "model1"
    assert model2["role"] == "lru_adversary"
    assert model1["metrics_shared_with"] is None
    assert model1["c_mem_in_loss"] == model2["c_mem_in_loss"] == "weight_zero"
    families = {row["family"] for row in payload["rows"] if row["arm_id"] == "model2"}
    assert families == {"lambda-0"}


def test_dense_is_not_moe_lambda_zero():
    payload = build_comparison(load_config())
    dense = _arm(payload, "model3")
    topk = _arm(payload, "model1")
    assert dense["model_kind"] == "dense"
    assert dense["model_class"] == "DenseLanguageModel"
    assert dense["code_path"] == CODE_PATH_DENSE
    assert dense["code_path"] != topk["code_path"]
    assert dense["lambda"] is None
    assert dense["policy_class"] is None
    assert dense["c_mem_in_loss"] == "absent"
    assert dense["checkpoint"].endswith("outputs/baseline/wikitext/seed-0/checkpoint.pt")
    assert dense["checkpoint"] != topk["checkpoint"]
    assert "Não é o MoE com λ=0" in dense["note"]


def test_model4_reuses_lru_only_for_positive_lambda():
    config = load_config()
    payload = build_comparison(config)
    model4 = _arm(payload, "model4")
    assert model4["policy_class"] == POLICY_CLASS
    assert model4["same_eviction_class_as"] == "model1"
    assert model4["same_code_path_as"] is None
    assert model4["code_path"] != CODE_PATH_LAMBDA0
    assert model4["c_mem_in_loss"] == "weighted"
    lambdas = [item["lambda"] for item in model4["variants"]]
    assert lambdas == [0.5, 1.0]
    assert 0.0 not in lambdas
    assert [lambda_tag(value) for value in lambdas] == ["0.5", "1"]
    row_lambdas = {row["lambda"] for row in payload["rows"] if row["arm_id"] == "model4"}
    assert row_lambdas == {0.5, 1.0}
    assert all(row["policy"] == "lru" for row in _rows(payload, "model4"))


def test_dose_budgets_follow_scaled_hot_slots_without_a_second_policy():
    config = load_config()
    payload = build_comparison(config)
    budgets = payload["dose_hook"]["budgets"]
    assert [item["B_mult"] for item in budgets] == list(config.eval.B_mult)
    assert [item["hot_slots"] for item in budgets] == [
        config.memory.scaled_hot_slots(config.model.E, multiplier)
        for multiplier in config.eval.B_mult
    ]
    assert [item["hot_slots"] for item in budgets] == [4, 8, 16]
    reference = [row for row in _rows(payload, "model2") if row["B_mult"] == 1.0]
    assert reference
    assert {row["source"] for row in reference} == {"holdout"}
    assert {row["hot_slots"] for row in reference} == {8}
    off = [row for row in _rows(payload, "model2") if row["B_mult"] != 1.0]
    assert {row["source"] for row in off} == {"dose_response"}
    assert {row["status"] for row in off} == {"pending"}
    dense_off = [row for row in _rows(payload, "model3") if row["B_mult"] != 1.0]
    assert {row["status"] for row in dense_off} == {"not_applicable"}
    assert {row["hot_slots"] for row in dense_off} == {None}
    assert {row["budget_applies"] for row in _rows(payload, "model3")} == {False}


def test_table_schema_stays_pending_until_a_measurement_is_supplied():
    payload = build_comparison(load_config())
    assert payload["columns"] == list(TABLE_COLUMNS)
    assert payload["rows"]
    assert len(payload["rows"]) == 30
    for row in payload["rows"]:
        assert list(row) == list(TABLE_COLUMNS)
        assert row["perplexity"] is None
        assert row["status"] in {"pending", "not_applicable"}
    assert payload["metrics_complete"] is False
    assert payload["holdout_ready"] is False
    assert payload["dose_ready"] is False
    assert payload["measurement_sources"] == []
    assert payload["conflicts"] == []
    reporting = [row for row in payload["rows"] if row["in_reporting_slice"]]
    assert {row["arm_id"] for row in reporting} == {"model2", "model3", "model4"}
    assert all(not row["in_reporting_slice"] for row in _rows(payload, "model1"))
    slice_note = payload["reporting_slice"]
    assert slice_note["id"] == "model2"
    assert slice_note["against"] == ["model3", "model4"]
    assert slice_note["holds"] == ["holdout", "dose_response"]
    assert slice_note["includes_subject_metrics_from"] == "model1"


def test_holdout_copies_lambda0_into_the_model2_slice():
    payload = build_comparison(load_config(), holdout=_full_holdout())
    model1 = _rows(payload, "model1", split="valid", B_mult=1.0)[0]
    model2 = _rows(payload, "model2", split="valid", B_mult=1.0)[0]
    assert model1["status"] == model2["status"] == "filled"
    assert model1["filled_from"] == model2["filled_from"] == "holdout"
    assert model2["perplexity"] == model1["perplexity"] == pytest.approx(10.0)
    assert model2["swaps"] == model1["swaps"] == 3
    assert model2["metrics_shared_with"] == "model1"
    assert model2["checkpoint"] == model1["checkpoint"]
    dense = _rows(payload, "model3", split="test", B_mult=1.0)[0]
    assert dense["perplexity"] == pytest.approx(20.0)
    assert dense["swaps"] is None
    assert dense["lambda"] is None
    penalty = _rows(payload, "model4", split="valid", B_mult=1.0)
    assert {row["lambda"]: row["perplexity"] for row in penalty} == {0.5: 11.0, 1.0: 12.0}
    assert payload["holdout_ready"] is True
    assert payload["dose_ready"] is False
    assert payload["metrics_complete"] is False
    assert payload["measurement_sources"] == ["holdout"]
    assert payload["produces_measurements"] is False
    dose_rows = [row for row in _rows(payload, "model2") if row["source"] == "dose_response"]
    assert {row["status"] for row in dose_rows} == {"pending"}


def test_holdout_rejects_a_separate_model2_run_and_a_dense_with_swaps():
    config = load_config()
    separate = _full_holdout()
    separate["runs"].append(_run("model2", "moe", 0.0, 10.0, 9))
    with pytest.raises(ComparisonError, match="lambda-0"):
        build_comparison(config, holdout=separate)
    dense_swaps = _full_holdout()
    dense_swaps["runs"][0]["splits"]["valid"]["swaps"] = 1
    with pytest.raises(ComparisonError, match="denso"):
        build_comparison(config, holdout=dense_swaps)


def test_dose_response_true_fills_the_adversary_slice_from_the_holdout_curve():
    config = load_config()
    only_holdout = build_comparison(config, holdout=_full_holdout())
    assert only_holdout["dose_ready"] is False
    embedded = build_comparison(config, holdout=_holdout_with_curve())
    assert embedded["dose_ready"] is True
    assert embedded["holdout_ready"] is True
    assert embedded["metrics_complete"] is True
    assert embedded["conflicts"] == []
    assert embedded["measurement_sources"] == ["holdout", "dose_response"]
    _assert_dose_slice(embedded)
    reference = _rows(embedded, "model2", split="valid", B_mult=1.0)[0]
    assert reference["filled_from"] == "holdout+dose_response"
    assert reference["perplexity"] == pytest.approx(10.0)
    from_file = build_comparison(config, holdout=_full_holdout(), dose=_full_dose())
    assert from_file["metrics_complete"] is True
    assert from_file["measurement_sources"] == ["holdout", "dose_response"]
    _assert_dose_slice(from_file)
    partial = _full_dose()
    partial["runs"] = [partial["runs"][0], partial["runs"][1]]
    sliced = build_comparison(config, holdout=_full_holdout(), dose=partial)
    assert sliced["dose_ready"] is False
    assert _rows(sliced, "model2", split="valid", B_mult=2.0)[0]["status"] == "filled"
    assert _rows(sliced, "model4", split="valid", B_mult=2.0)[0]["status"] == "pending"


def _assert_dose_slice(payload: dict) -> None:
    model1 = _rows(payload, "model1", split="test", B_mult=0.5)[0]
    model2 = _rows(payload, "model2", split="test", B_mult=0.5)[0]
    assert model1["status"] == model2["status"] == "filled"
    assert model2["filled_from"] == "dose_response"
    assert model2["perplexity"] == model1["perplexity"]
    assert model2["hot_slots"] == 4
    assert model2["metrics_shared_with"] == "model1"
    wide = [
        row
        for row in _rows(payload, "model4", split="valid", B_mult=2.0)
        if row["lambda"] == 1.0
    ]
    assert wide[0]["hot_slots"] == 16
    dense = _rows(payload, "model3", split="valid", B_mult=0.5)[0]
    assert dense["status"] == "not_applicable"


def test_dose_rejects_a_second_policy_a_dense_budget_and_a_conflicting_number():
    config = load_config()
    other = _full_dose()
    other["residency"] = "fifo"
    with pytest.raises(ComparisonError, match="set_hot_budget"):
        build_comparison(config, dose=other)
    invented = _full_dose()
    invented["runs"][1]["dose_response"]["method"] = "evict_fifo"
    with pytest.raises(ComparisonError, match="não há segunda política"):
        build_comparison(config, dose=invented)
    dense = _full_dose()
    dense["runs"][0]["dose_response"] = {
        "status": "ok",
        "method": "reset_lru_at_yaml_b_then_set_hot_budget",
        "retrained": False,
        "budgets": [_dose_point(1.0, 8, 20.0, 1)],
    }
    with pytest.raises(ComparisonError, match="denso"):
        build_comparison(config, dose=dense)
    clashed = build_comparison(
        config,
        holdout=_full_holdout(),
        dose=_full_dose(clash_lambda0=99.0),
    )
    assert clashed["metrics_complete"] is False
    assert clashed["conflicts"]
    row = _rows(clashed, "model2", split="valid", B_mult=1.0)[0]
    assert row["status"] == "conflict"
    assert row["perplexity"] is None
    twin = _rows(clashed, "model1", split="valid", B_mult=1.0)[0]
    assert twin["status"] == "conflict"


def test_wall_fields_from_holdout_are_copied_and_cpu_null_stays_filled():
    holdout = _full_holdout()
    for run in holdout["runs"]:
        for split in run["splits"].values():
            split.update(
                wall_p50_ms=None,
                wall_p95_ms=None,
                wall_clock="unavailable",
                wall_samples=0,
                wall_report_grade=False,
                wall_gpu=None,
                wall_forward_batch=2,
            )
    payload = build_comparison(load_config(), holdout=holdout)
    row = _rows(payload, "model2", split="valid", B_mult=1.0)[0]
    twin = _rows(payload, "model1", split="valid", B_mult=1.0)[0]
    assert row["status"] == "filled"
    assert row["perplexity"] == twin["perplexity"]
    assert row["wall_p50_ms"] is None
    assert row["wall_p95_ms"] is None
    assert row["wall_clock"] == "unavailable"
    assert row["wall_samples"] == 0
    assert row["wall_report_grade"] is False
    assert row["wall_forward_batch"] == 2
    assert twin["wall_clock"] == row["wall_clock"]
    dense = _rows(payload, "model3", split="test", B_mult=1.0)[0]
    assert dense["status"] == "filled"
    assert dense["swaps"] is None
    assert dense["wall_clock"] == "unavailable"


def test_seed0_checkpoint_wins_and_legacy_is_only_a_fallback(tmp_path: Path):
    outputs = tmp_path / "outputs"
    legacy = outputs / "wikitext" / "lambda-0" / "checkpoint.pt"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"legacy")
    payload = build_comparison(load_config(), outputs=outputs)
    assert _arm(payload, "model1")["checkpoint"] == str(legacy)
    assert _arm(payload, "model2")["checkpoint"] == str(legacy)
    canonical = outputs / "wikitext" / "seed-0" / "lambda-0" / "checkpoint.pt"
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"seed0")
    payload = build_comparison(load_config(), outputs=outputs)
    assert _arm(payload, "model1")["checkpoint"] == str(canonical)
    assert _arm(payload, "model2")["checkpoint"] == str(canonical)


def test_manifest_families_match_holdout_discovery(tmp_path: Path):
    holdout = _holdout_module()
    config = load_config()
    for value in config.loss.lambda_grid:
        assert lambda_tag(value) == holdout.lambda_tag(value)
    assert holdout.REPORT_SPLITS == SPLITS
    assert holdout.WALL_FIELDS == WALL_FIELDS
    discovered = holdout.discover_families(tmp_path, config)
    payload = build_comparison(config, outputs=tmp_path)
    by_family = {}
    for row in payload["rows"]:
        by_family.setdefault(row["family"], Path(row["checkpoint"]))
    assert [family.name for family in discovered] == [
        "baseline",
        "lambda-0",
        "lambda-0.5",
        "lambda-1",
    ]
    for family in discovered:
        assert by_family[family.name] == family.path
    assert "model2" not in by_family
    assert by_family["lambda-0"] == Path(_arm(payload, "model1")["checkpoint"])
    assert _arm(payload, "model1")["checkpoint"] == _arm(payload, "model2")["checkpoint"]


def test_runner_does_not_train_and_manifest_only_ignores_a_broken_holdout(tmp_path: Path):
    source = RUNNER.read_text(encoding="utf-8")
    for banned in ("train_one_batch", "prepare_train", "optimizer", "backward", "set_hot_budget("):
        assert banned not in source
    module = (ROOT / "src" / "moe_memory" / "comparison.py").read_text(encoding="utf-8")
    assert module.count("\nclass ") == 1
    assert "class ComparisonError" in module
    outputs = tmp_path / "outputs"
    broken = outputs / "holdout" / "summary.json"
    broken.parent.mkdir(parents=True)
    broken.write_text("{", encoding="utf-8")
    out = tmp_path / "comparison.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--manifest-only",
            "--outputs",
            str(outputs),
            "--out",
            str(out),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "MoE + LRU adversary" in completed.stdout
    assert "segunda política: nenhuma" in completed.stdout
    assert "não é número de relatório" in completed.stdout
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["identity"]["model1_equals_model2"] is True
    assert payload["metrics_complete"] is False
    probed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--outputs",
            str(outputs),
            "--out",
            str(tmp_path / "should-not-matter.json"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert probed.returncode == 2
    assert "ilegível" in probed.stderr
    usage = subprocess.run(
        [sys.executable, str(RUNNER), "--manifest-only", "--holdout", str(broken)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert usage.returncode == 1


def test_cli_fills_model2_from_holdout_and_rejects_another_policy(tmp_path: Path):
    holdout_path = tmp_path / "holdout.json"
    holdout_path.write_text(json.dumps(_full_holdout()), encoding="utf-8")
    out = tmp_path / "comparison.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--holdout",
            str(holdout_path),
            "--outputs",
            str(tmp_path / "outputs"),
            "--out",
            str(out),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    model1 = _rows(payload, "model1", split="valid", B_mult=1.0)[0]
    model2 = _rows(payload, "model2", split="valid", B_mult=1.0)[0]
    assert model2["perplexity"] == model1["perplexity"]
    assert model2["checkpoint"] == model1["checkpoint"]
    dose_path = tmp_path / "dose.json"
    dose = _full_dose()
    dose["residency"] = "fifo"
    dose_path.write_text(json.dumps(dose), encoding="utf-8")
    rejected_out = tmp_path / "rejected.json"
    rejected = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--dose",
            str(dose_path),
            "--outputs",
            str(tmp_path / "empty-outputs"),
            "--out",
            str(rejected_out),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode == 2
    assert "set_hot_budget" in rejected.stderr
    assert not rejected_out.exists()
