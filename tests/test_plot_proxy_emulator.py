"""Pós-processamento do jsonl: proxy × emulador, sem treinar."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "proxy_emulator" / "lambda-0.5" / "train.jsonl"


def _load():
    path = ROOT / "experiments" / "plot_proxy_emulator.py"
    spec = importlib.util.spec_from_file_location("plot_proxy_emulator_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


plot = _load()


def test_axis_ticks_land_on_one_two_five_steps():
    assert plot._ticks(0.38, 2.12) == [0.5, 1.0, 1.5, 2.0]
    assert plot._ticks(7.6, 42.4) == [10.0, 20.0, 30.0, 40.0]


def test_lambda_tag_matches_the_wikitext_layout():
    assert plot.lambda_tag(0.0) == "0"
    assert plot.lambda_tag(0.5) == "0.5"
    assert plot.lambda_tag(1.0) == "1"


def test_loader_drops_non_steps_nulls_and_bad_lines():
    loaded = plot.load_log(FIXTURE)
    assert loaded["bad_lines"] == 1
    assert loaded["skipped_rows"] == 2
    assert [row["step"] for row in loaded["rows"]] == [1, 2, 3, 4]
    assert [row["bytes_moved"] for row in loaded["rows"]] == [10, 20, 30, 40]
    assert loaded["rows"][0]["entropy"] == 2.0


def test_pearson_and_spearman_on_the_fixture_rows():
    rows = plot.load_log(FIXTURE)["rows"]
    stats = plot.correlate_rows(rows)
    assert stats["n_steps"] == 4
    assert stats["c_mem_vs_bytes"]["pearson"] == 1.0
    assert stats["c_mem_vs_bytes"]["spearman"] == 1.0
    assert stats["c_mem_vs_bytes"]["note"] is None
    assert stats["c_mem_vs_swaps"]["pearson"] == 0.8
    assert stats["c_mem_vs_swaps"]["spearman"] == 0.8


def test_spearman_is_one_when_only_the_order_matches():
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [1.0, 10.0, 100.0, 1000.0]
    assert plot.spearman(xs, ys) == 1.0
    assert plot.pearson(xs, ys) < 0.99


def test_ties_use_average_ranks():
    assert plot.average_ranks([1.0, 2.0, 2.0]) == [1.0, 2.5, 2.5]
    rho = plot.spearman([1.0, 2.0, 2.0], [1.0, 2.0, 3.0])
    assert math.isclose(rho, math.sqrt(3) / 2)


def test_constant_series_has_no_correlation():
    flat_proxy = [
        {"c_mem": 3.0, "bytes_moved": 10.0, "swaps": 2.0, "entropy": 1.0},
        {"c_mem": 3.0, "bytes_moved": 20.0, "swaps": 4.0, "entropy": 1.0},
    ]
    stats = plot.correlate_rows(flat_proxy)
    assert stats["c_mem_vs_bytes"]["pearson"] is None
    assert stats["c_mem_vs_bytes"]["spearman"] is None
    assert stats["c_mem_vs_bytes"]["note"] == "variância zero em c_mem"
    flat_emulator = [
        {"c_mem": 1.0, "bytes_moved": 5.0, "swaps": 2.0, "entropy": 1.0},
        {"c_mem": 2.0, "bytes_moved": 5.0, "swaps": 2.0, "entropy": 1.0},
    ]
    stats = plot.correlate_rows(flat_emulator)
    assert stats["c_mem_vs_bytes"]["note"] == "variância zero em bytes_moved"
    assert stats["c_mem_vs_swaps"]["note"] == "variância zero em swaps"


def test_script_writes_figures_csv_and_summary(tmp_path: Path):
    out = tmp_path / "proxy-emulator"
    code = plot.main(["--log", str(FIXTURE), "--out", str(out)])
    assert code == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["trained"] is False
    assert summary["cpu_only"] is True
    assert summary["n_steps"] == 4
    run = summary["runs"][0]
    assert run["label"] == "lambda-0.5"
    assert run["lambda"] == 0.5
    assert run["bad_lines"] == 1
    assert run["c_mem_vs_bytes"]["pearson"] == 1.0
    assert run["c_mem_vs_swaps"]["spearman"] == 0.8
    assert run["entropy_first"] == 2.0
    assert run["entropy_last"] == 0.5
    assert summary["pooled"]["c_mem_vs_bytes"]["pearson"] == 1.0
    assert summary["skipped_legacy"] == []

    with (out / "steps.csv").open(encoding="utf-8", newline="") as handle:
        steps = list(csv.DictReader(handle))
    assert [row["step"] for row in steps] == ["1", "2", "3", "4"]
    assert steps[2]["swaps"] == "4"

    with (out / "correlations.csv").open(encoding="utf-8", newline="") as handle:
        correlations = list(csv.DictReader(handle))
    pairs = {(row["label"], row["pair"]): row for row in correlations}
    assert pairs[("lambda-0.5", "c_mem_vs_bytes")]["pearson"] == "1"
    assert pairs[("pooled", "c_mem_vs_swaps")]["spearman"] == "0.8"

    entropy = (out / "entropy.svg").read_text(encoding="utf-8")
    assert entropy.startswith("<svg ")
    assert "Entropia do roteador por passo" in entropy
    assert "lambda-0.5" in entropy
    assert "<polyline " in entropy
    scatter = (out / "c_mem_vs_bytes.svg").read_text(encoding="utf-8")
    assert "Proxy C_mem" in scatter
    assert "<circle " in scatter
    assert (out / "c_mem_vs_swaps.svg").is_file()


def test_discovery_reads_lambda_directories(tmp_path: Path):
    root = tmp_path / "wikitext"
    for name, c_mem in (("lambda-0", 1.0), ("lambda-1", 4.0)):
        folder = root / name
        folder.mkdir(parents=True)
        lam = 0.0 if name == "lambda-0" else 1.0
        (folder / "train.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {
                        "event": "step",
                        "step": step,
                        "lambda": lam,
                        "c_mem": c_mem + step - 1,
                        "bytes_moved": (c_mem + step - 1) * 10,
                        "swaps": c_mem + step - 1,
                        "entropy": 1.2 if step == 1 else 0.9,
                    }
                )
                for step in (1, 2)
            )
            + "\n",
            encoding="utf-8",
        )
    found = plot.discover_logs(root)
    assert [path.parent.name for path in found] == ["lambda-0", "lambda-1"]
    out = tmp_path / "out"
    assert plot.main(["--root", str(root), "--out", str(out)]) == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert [run["label"] for run in summary["runs"]] == ["lambda-0", "lambda-1"]
    assert summary["pooled"]["n_steps"] == 4
    assert summary["skipped_legacy"] == []
    assert "<polyline " in (out / "entropy.svg").read_text(encoding="utf-8")


def _write_run(folder: Path, lam: float, c_mem: float) -> None:
    folder.mkdir(parents=True)
    (folder / "train.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "event": "step",
                    "step": step,
                    "lambda": lam,
                    "c_mem": c_mem + step - 1,
                    "bytes_moved": (c_mem + step - 1) * 10,
                    "swaps": c_mem + step - 1,
                    "entropy": 1.2 if step == 1 else 0.9,
                }
            )
            for step in (1, 2)
        )
        + "\n",
        encoding="utf-8",
    )


def test_discovery_reads_seed_layout_and_keeps_uncovered_legacy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    root = tmp_path / "wikitext"
    _write_run(root / "seed-0" / "lambda-0", 0.0, 1.0)
    _write_run(root / "seed-1" / "lambda-0.5", 0.5, 2.0)
    _write_run(root / "lambda-1", 1.0, 3.0)
    _write_run(root / "lambda-0", 0.0, 9.0)
    _write_run(root / "seed-notes" / "lambda-0", 0.0, 8.0)

    used, skipped = plot.partition_logs(root)
    assert [path.relative_to(root).as_posix() for path in used] == [
        "lambda-1/train.jsonl",
        "seed-0/lambda-0/train.jsonl",
        "seed-1/lambda-0.5/train.jsonl",
    ]
    assert [path.relative_to(root).as_posix() for path in skipped] == ["lambda-0/train.jsonl"]
    assert plot.discover_logs(root) == used

    out = tmp_path / "out"
    assert plot.main(["--root", str(root), "--out", str(out)]) == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert [run["label"] for run in summary["runs"]] == [
        "lambda-1",
        "seed-0/lambda-0",
        "seed-1/lambda-0.5",
    ]
    assert summary["skipped_legacy"] == [str(root / "lambda-0" / "train.jsonl")]
    assert summary["n_steps"] == 6
    assert "seed-notes" not in (out / "entropy.svg").read_text(encoding="utf-8")
    assert "layout antigo ignorado" in capsys.readouterr().out

    explicit = tmp_path / "explicit"
    code = plot.main(
        ["--log", str(root / "lambda-0" / "train.jsonl"), "--out", str(explicit)]
    )
    assert code == 0
    forced = json.loads((explicit / "summary.json").read_text(encoding="utf-8"))
    assert forced["runs"][0]["label"] == "lambda-0"
    assert forced["skipped_legacy"] == []
    assert forced["runs"][0]["mean_c_mem"] == 9.5


def test_missing_log_and_dense_baseline_do_not_invent_a_correlation(tmp_path: Path):
    missing = plot.main(["--log", str(tmp_path / "nope.jsonl"), "--out", str(tmp_path / "a")])
    assert missing == plot.EXIT_USAGE
    assert not (tmp_path / "a" / "summary.json").exists()

    dense = tmp_path / "baseline" / "train.jsonl"
    dense.parent.mkdir()
    dense.write_text(
        json.dumps(
            {
                "event": "step",
                "step": 1,
                "c_mem": None,
                "balance": None,
                "total": 1.0,
                "task": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "dense-out"
    code = plot.main(["--log", str(dense.parent), "--out", str(out)])
    assert code == plot.EXIT_USAGE
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_steps"] == 0
    assert summary["runs"][0]["c_mem_vs_bytes"]["pearson"] is None
    assert summary["runs"][0]["skip_reason"]
