"""Sementes de eval.seeds: iteração e pastas, sem treinar e sem baixar corpus.

O lançador é falso. Ele grava um done.json compatível com o alvo e devolve
0. Nada aqui sobe peso, GPU ou WikiText.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from moe_memory.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    path = ROOT / "experiments" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


seed_layout = _load("seed_layout_under_test", "seed_layout.py")
run_lambda_grid = _load("run_lambda_grid_multi_seed", "run_lambda_grid.py")
run_baseline = _load("run_baseline_multi_seed", "run_baseline.py")
train_wikitext = _load("train_wikitext_multi_seed", "train_wikitext.py")
holdout = _load("holdout_multi_seed", "run_holdout_eval.py")


def _yaml_with_seeds(tmp_path: Path, n_seeds: int) -> Path:
    text = (ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
    text = text.replace("seeds: 3", f"seeds: {n_seeds}", 1)
    path = tmp_path / f"seeds-{n_seeds}.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _moe_done(lam: float, seed: int, *, seq_len: int = 8, steps: int = 1) -> dict:
    return {
        "lambda": lam,
        "seed": seed,
        "seq_len": seq_len,
        "steps": steps,
        "epochs_completed": 0,
        "tokens": steps * seq_len,
        "finite": True,
    }


def _dense_done(seed: int, *, seq_len: int = 8, steps: int = 1) -> dict:
    return {
        "model": "dense",
        "memory_path": False,
        "corpus": "WikiText-2",
        "lambda": None,
        "seed": seed,
        "seq_len": seq_len,
        "steps": steps,
        "epochs_completed": 0,
        "tokens": steps * seq_len,
        "finite": True,
        "status": "done",
        "last_total": 1.0,
        "last_task": 1.0,
    }


def _flag(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def _accept(command: list[str], log_path: Path) -> int:
    work = Path(_flag(command, "--work-dir"))
    seed = int(_flag(command, "--seed"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if "--lambda" in command:
        payload = _moe_done(float(_flag(command, "--lambda")), seed)
    else:
        payload = _dense_done(seed)
    (work / "done.json").write_text(json.dumps(payload), encoding="utf-8")
    return 0


def test_seed_indices_follow_eval_seeds_from_the_yaml():
    assert seed_layout.seed_indices(1) == [0]
    assert seed_layout.seed_indices(3) == [0, 1, 2]
    with pytest.raises(ValueError):
        seed_layout.seed_indices(0)
    with pytest.raises(ValueError):
        seed_layout.seed_indices(True)  # type: ignore[arg-type]
    config = load_config()
    assert config.eval.seeds == 3
    assert seed_layout.seed_indices(config.eval.seeds) == [0, 1, 2]
    grid = run_lambda_grid.parse_args(["--device", "cpu", "--steps", "1"])
    request = run_lambda_grid.resolve_request(config, grid)
    assert request.seed is None
    assert request.seeds == [0, 1, 2]
    base = run_baseline.parse_args(["--device", "cpu", "--steps", "1"])
    resolved = run_baseline.resolve_request(config, base)
    assert resolved.seeds == [0, 1, 2]
    assert resolved.work_dir == ROOT / "outputs" / "baseline" / "wikitext"


def test_explicit_seed_does_not_expand_to_the_yaml_count():
    config = load_config()
    args = run_lambda_grid.parse_args(["--device", "cpu", "--steps", "1", "--seed", "1"])
    request = run_lambda_grid.resolve_request(config, args)
    assert request.seeds == [1]
    with pytest.raises(SystemExit):
        run_lambda_grid.parse_args(["--device", "cpu", "--seed", "-1"])
    with pytest.raises(SystemExit):
        run_baseline.parse_args(["--device", "cpu", "--seed", "-1"])


def test_direct_train_seed_one_uses_the_nested_directory():
    args = train_wikitext.parse_args(["--lambda", "0", "--seed", "1"])
    assert args.work_dir == ROOT / "outputs" / "wikitext" / "seed-1" / "lambda-0"


def test_lambda_grid_reads_yaml_seeds_and_writes_per_seed_paths(tmp_path: Path):
    config_path = _yaml_with_seeds(tmp_path, 2)
    config = load_config(config_path)
    output = tmp_path / "wikitext"
    calls: list[list[str]] = []

    def launch(command: list[str], log_path: Path) -> int:
        calls.append(command)
        return _accept(command, log_path)

    args = run_lambda_grid.parse_args(
        [
            "--config",
            str(config_path),
            "--device",
            "cpu",
            "--steps",
            "1",
            "--seq-len",
            "8",
            "--batch-size",
            "1",
            "--work-dir",
            str(output),
        ]
    )
    request = run_lambda_grid.resolve_request(config, args)
    summary = run_lambda_grid.run_grid(config, request, launch)
    assert request.seeds == [0, 1]
    assert summary["eval_seeds"] == 2
    assert summary["seeds"] == [0, 1]
    assert summary["seed"] is None
    assert len(summary["runs"]) == 6
    assert len(calls) == 6
    expected = []
    for seed in (0, 1):
        for lam, tag in ((0.0, "0"), (0.5, "0.5"), (1.0, "1")):
            expected.append((seed, lam, output / f"seed-{seed}" / f"lambda-{tag}"))
    got = []
    for command in calls:
        got.append(
            (
                int(_flag(command, "--seed")),
                float(_flag(command, "--lambda")),
                Path(_flag(command, "--work-dir")),
            )
        )
    assert got == expected
    assert all(run["status"] == "done" and run["legacy_layout"] is False for run in summary["runs"])
    assert [item["seed"] for item in summary["per_seed"]] == [0, 1]
    assert all(item["status"] == "done" for item in summary["per_seed"])
    assert summary["per_seed"][0]["work_dirs"] == [str(path) for _, _, path in expected[:3]]
    assert summary["per_seed"][1]["work_dirs"] == [str(path) for _, _, path in expected[3:]]
    saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert saved["per_seed"] == summary["per_seed"]
    assert [run["seed"] for run in saved["runs"]] == [0, 0, 0, 1, 1, 1]
    assert "perplexity" not in json.dumps(saved)


def test_lambda_seed_zero_keeps_a_legacy_directory_and_later_seeds_do_not(tmp_path: Path):
    config = load_config()
    output = tmp_path / "grid"
    legacy = output / "lambda-0"
    legacy.mkdir(parents=True)
    (legacy / "checkpoint.pt").write_bytes(b"")
    (legacy / "done.json").write_text(
        json.dumps(_moe_done(0.0, 0, steps=4, seq_len=8)),
        encoding="utf-8",
    )
    calls: list[tuple[int, float, Path]] = []

    def launch(command: list[str], log_path: Path) -> int:
        calls.append(
            (
                int(_flag(command, "--seed")),
                float(_flag(command, "--lambda")),
                Path(_flag(command, "--work-dir")),
            )
        )
        return _accept(command, log_path)

    request = Namespace(
        config=None,
        device="cpu",
        seq_len=8,
        batch_size=1,
        seed=None,
        seeds=[0, 1],
        steps=1,
        epochs=None,
        checkpoint_every=1,
        work_dir=output,
    )
    summary = run_lambda_grid.run_grid(config, request, launch)
    seed0 = [run for run in summary["runs"] if run["seed"] == 0 and run["lambda"] == 0.0][0]
    assert seed0["skipped"] is True
    assert seed0["legacy_layout"] is True
    assert seed0["work_dir"] == str(legacy)
    assert seed0["status"] == "done"
    assert not (output / "seed-0" / "lambda-0").exists()
    assert (0, 0.0) not in {(seed, lam) for seed, lam, _path in calls}
    fresh = [run for run in summary["runs"] if run["seed"] == 1 and run["lambda"] == 0.0][0]
    assert fresh["legacy_layout"] is False
    assert fresh["work_dir"] == str(output / "seed-1" / "lambda-0")
    assert fresh["status"] == "done"
    saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    by_seed = {item["seed"]: item for item in saved["per_seed"]}
    assert by_seed[0]["status"] == "done"
    assert str(legacy) in by_seed[0]["work_dirs"]
    assert str(output / "seed-1" / "lambda-0") in by_seed[1]["work_dirs"]


def test_baseline_reads_yaml_seeds_and_isolates_each_directory(tmp_path: Path):
    config_path = _yaml_with_seeds(tmp_path, 2)
    config = load_config(config_path)
    output = tmp_path / "baseline"
    calls: list[Path] = []

    def launch(command: list[str], log_path: Path) -> int:
        assert "--lambda" not in command
        assert _flag(command, "--corpus") == "WikiText-2"
        calls.append(Path(_flag(command, "--work-dir")))
        assert _flag(command, "--seed") == str(len(calls) - 1)
        return _accept(command, log_path)

    args = run_baseline.parse_args(
        [
            "--config",
            str(config_path),
            "--device",
            "cpu",
            "--steps",
            "1",
            "--seq-len",
            "8",
            "--batch-size",
            "1",
            "--work-dir",
            str(output),
        ]
    )
    request = run_baseline.resolve_request(config, args)
    summary = run_baseline.run_baseline(config, request, launch)
    assert request.seeds == [0, 1]
    assert calls == [output / "seed-0", output / "seed-1"]
    assert [run["work_dir"] for run in summary["runs"]] == [str(path) for path in calls]
    assert summary["per_seed"][0]["status"] == "done"
    assert summary["per_seed"][1]["logs"] == [str(output / "seed-1" / "train.log")]
    saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert saved["seeds"] == [0, 1]
    assert saved["eval_seeds"] == 2
    assert saved["lambda"] is None
    assert [item["status"] for item in saved["per_seed"]] == ["done", "done"]


def test_baseline_seed_zero_reuses_the_flat_directory(tmp_path: Path):
    config = load_config()
    output = tmp_path / "wikitext"
    output.mkdir()
    (output / "checkpoint.pt").write_bytes(b"")
    (output / "done.json").write_text(json.dumps(_dense_done(0)), encoding="utf-8")
    calls: list[int] = []

    def launch(command: list[str], log_path: Path) -> int:
        calls.append(int(_flag(command, "--seed")))
        return _accept(command, log_path)

    request = Namespace(
        config=None,
        corpus="WikiText-2",
        corpus_name="WikiText-2",
        device="cpu",
        seq_len=8,
        batch_size=1,
        seed=None,
        seeds=[0, 1],
        steps=1,
        epochs=None,
        checkpoint_every=1,
        work_dir=output,
    )
    summary = run_baseline.run_baseline(config, request, launch)
    assert calls == [1]
    assert summary["runs"][0]["skipped"] is True
    assert summary["runs"][0]["legacy_layout"] is True
    assert summary["runs"][0]["work_dir"] == str(output)
    assert summary["runs"][1]["work_dir"] == str(output / "seed-1")
    assert summary["runs"][1]["legacy_layout"] is False
    assert not (output / "seed-0").exists()
    saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert saved["per_seed"][0]["work_dirs"] == [str(output)]
    assert saved["per_seed"][1]["status"] == "done"
    assert saved["per_seed"][0]["status"] == "done"


def test_a_failed_seed_does_not_cancel_the_next_one(tmp_path: Path):
    config = load_config(_yaml_with_seeds(tmp_path, 2))
    output = tmp_path / "dense"

    def launch(command: list[str], log_path: Path) -> int:
        seed = int(_flag(command, "--seed"))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if seed == 0:
            log_path.write_text("uso\n", encoding="utf-8")
            return 1
        return _accept(command, log_path)

    request = Namespace(
        config=None,
        corpus="WikiText-2",
        corpus_name="WikiText-2",
        device="cpu",
        seq_len=8,
        batch_size=1,
        seed=None,
        seeds=[0, 1],
        steps=1,
        epochs=None,
        checkpoint_every=1,
        work_dir=output,
    )
    summary = run_baseline.run_baseline(config, request, launch)
    assert [run["status"] for run in summary["runs"]] == ["failed", "done"]
    assert [item["status"] for item in summary["per_seed"]] == ["failed", "done"]
    assert summary["per_seed"][0]["work_dirs"] == [str(output / "seed-0")]
    assert summary["per_seed"][1]["work_dirs"] == [str(output / "seed-1")]


def test_holdout_discovery_prefers_seed_zero_and_falls_back_to_the_flat_checkpoint(tmp_path: Path):
    config = load_config()
    families = holdout.discover_families(tmp_path, config)
    by_name = {family.name: family for family in families}
    assert by_name["baseline"].path == (
        tmp_path / "baseline" / "wikitext" / "seed-0" / "checkpoint.pt"
    )
    assert by_name["lambda-0"].path == tmp_path / "wikitext" / "seed-0" / "lambda-0" / "checkpoint.pt"
    assert by_name["lambda-0.5"].path == (
        tmp_path / "wikitext" / "seed-0" / "lambda-0.5" / "checkpoint.pt"
    )
    assert by_name["lambda-1"].path == tmp_path / "wikitext" / "seed-0" / "lambda-1" / "checkpoint.pt"

    legacy = tmp_path / "wikitext" / "lambda-0.5"
    legacy.mkdir(parents=True)
    (legacy / "checkpoint.pt").write_bytes(b"old")
    families = holdout.discover_families(tmp_path, config)
    found = {family.name: family.path for family in families}
    assert found["lambda-0.5"] == legacy / "checkpoint.pt"
    assert found["lambda-0"] == tmp_path / "wikitext" / "seed-0" / "lambda-0" / "checkpoint.pt"

    canonical = tmp_path / "wikitext" / "seed-0" / "lambda-0.5"
    canonical.mkdir(parents=True)
    (canonical / "checkpoint.pt").write_bytes(b"new")
    families = holdout.discover_families(tmp_path, config)
    found = {family.name: family.path for family in families}
    assert found["lambda-0.5"] == canonical / "checkpoint.pt"

    flat = tmp_path / "baseline" / "wikitext"
    flat.mkdir(parents=True)
    (flat / "checkpoint.pt").write_bytes(b"dense")
    families = holdout.discover_families(tmp_path, config)
    found = {family.name: family.path for family in families}
    assert found["baseline"] == flat / "checkpoint.pt"
    nested = flat / "seed-0"
    nested.mkdir()
    (nested / "checkpoint.pt").write_bytes(b"dense-new")
    families = holdout.discover_families(tmp_path, config)
    found = {family.name: family.path for family in families}
    assert found["baseline"] == nested / "checkpoint.pt"

    assert holdout.name_from_path(nested / "checkpoint.pt") == "baseline"
    assert holdout.name_from_path(flat / "checkpoint.pt") == "baseline"
    assert holdout.name_from_path(canonical / "checkpoint.pt") == "lambda-0.5"


def test_canonical_dirs_ignore_the_disk_and_resolve_does_not(tmp_path: Path):
    root = tmp_path / "wikitext"
    legacy = root / "lambda-1"
    legacy.mkdir(parents=True)
    (legacy / "train.jsonl").write_text("{}\n", encoding="utf-8")
    assert seed_layout.lambda_run_dir(root, 1.0, 0) == root / "seed-0" / "lambda-1"
    assert seed_layout.resolve_lambda_run_dir(root, 1.0, 0) == legacy
    assert seed_layout.resolve_lambda_run_dir(root, 1.0, 1) == root / "seed-1" / "lambda-1"
    occupied = root / "seed-0" / "lambda-1"
    occupied.mkdir(parents=True)
    (occupied / "done.json").write_text("{}", encoding="utf-8")
    assert seed_layout.resolve_lambda_run_dir(root, 1.0, 0) == occupied

    dense = tmp_path / "baseline"
    (dense / "train.log").parent.mkdir(parents=True)
    (dense / "train.log").write_text("old\n", encoding="utf-8")
    assert seed_layout.baseline_seed_dir(dense, 0) == dense / "seed-0"
    assert seed_layout.resolve_baseline_seed_dir(dense, 0) == dense
    assert seed_layout.resolve_baseline_seed_dir(dense, 2) == dense / "seed-2"
