"""Corrida curta de WikiText-2 e a política de retomada da grade de λ.

O passo real usa o modelo do YAML (fp32). A política de OOM/NaN não treina:
um lançador falso devolve o código de saída.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
WIKI = "\n".join(
    [
        " = Title = ",
        "",
        "The cat sat on the mat and looked at the moon for a long while.",
        "Then the dog ran past the gate and into the field.",
        "",
        " = Other = ",
        "",
        "Another article has several words so the block can close cleanly.",
        "",
    ]
)


def _load(name: str, filename: str):
    path = ROOT / "experiments" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


train_wikitext = _load("train_wikitext_under_test", "train_wikitext.py")
run_lambda_grid = _load("run_lambda_grid_under_test", "run_lambda_grid.py")


def _local_config(tmp_path: Path) -> Path:
    wiki = tmp_path / "wiki"
    stories = tmp_path / "stories"
    wiki.mkdir()
    stories.mkdir()
    (stories / "train.txt").write_text("Once upon a time.\n", encoding="utf-8")
    (stories / "valid.txt").write_text("Once upon a time.\n", encoding="utf-8")
    for name in ("train.txt", "valid.txt", "test.txt"):
        (wiki / name).write_text((WIKI + "\n") * 30, encoding="utf-8")
    text = (ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
    text = text.replace('TinyStories: ""', f'TinyStories: "{stories}"')
    text = text.replace('WikiText-2: ""', f'WikiText-2: "{wiki}"')
    path = tmp_path / "local.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _done(lam: float, *, steps: int, seq_len: int, seed: int = 0, epochs_completed: int = 0) -> dict:
    return {
        "lambda": lam,
        "seed": seed,
        "seq_len": seq_len,
        "batch_size": 1,
        "steps": steps,
        "epochs_completed": epochs_completed,
        "tokens": steps * seq_len,
        "target_steps": steps,
        "target_epochs": None,
        "finite": True,
        "last_total": 1.0,
        "last_task": 1.0,
        "last_grad_norm": 1.0,
    }


def test_defaults_follow_the_yaml_and_the_lambda_tag():
    args = train_wikitext.parse_args(["--lambda", "0.5"])
    assert args.epochs == 1
    assert args.steps is None
    assert args.batch_size == train_wikitext.DEFAULT_BATCH_SIZE
    assert args.seq_len is None
    assert args.device == "cpu"
    assert args.seed == 0
    assert train_wikitext.lambda_tag(0.0) == "0"
    assert train_wikitext.lambda_tag(0.5) == "0.5"
    assert train_wikitext.lambda_tag(1.0) == "1"
    with pytest.raises(SystemExit):
        train_wikitext.parse_args(["--lambda", "0", "--steps", "1", "--epochs", "1"])
    assert not train_wikitext.losses_are_finite(
        {"total": float("nan"), "task": 1.0, "c_mem": 1.0, "balance": 1.0, "grad_norm": 1.0}
    )
    assert train_wikitext.is_oom(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))


def test_log_trim_drops_steps_ahead_of_the_checkpoint(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"event":"step","step":1,"total":1}',
                '{"event":"step","step":2,"total":2}',
                '{"event":"nonfinite","step_attempt":3}',
                '{"event":"step","step":3,"total":9}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    train_wikitext._trim_log_to_checkpoint(path, steps=2)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row.get("step") for row in rows] == [1, 2, None]
    assert rows[2]["event"] == "nonfinite"


def test_sequence_order_is_stable_and_local_wikitext_packs(tmp_path: Path):
    from moe_memory.config import load_config

    first = train_wikitext.sequence_order(17, seed=3, epoch=0)
    again = train_wikitext.sequence_order(17, seed=3, epoch=0)
    other = train_wikitext.sequence_order(17, seed=3, epoch=1)
    assert torch.equal(first, again)
    assert not torch.equal(first, other)
    config = load_config(_local_config(tmp_path))
    sequences = train_wikitext.load_train_sequences(config, seq_len=8)
    assert sequences.ndim == 2 and sequences.size(1) == 8
    assert sequences.size(0) >= 2
    assert int(sequences.max()) < 50257


def test_lambda_outside_the_grid_exits_before_a_step(tmp_path: Path):
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments" / "train_wikitext.py"),
            "--config",
            str(_local_config(tmp_path)),
            "--lambda",
            "0.25",
            "--steps",
            "1",
            "--seq-len",
            "8",
            "--work-dir",
            str(tmp_path / "run"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == train_wikitext.EXIT_USAGE
    assert "não está na grade" in completed.stderr
    assert not (tmp_path / "run" / "checkpoint.pt").exists()


def test_no_checkpoint_keeps_the_log_and_skips_the_pt_file(tmp_path: Path):
    config = _local_config(tmp_path)
    work = tmp_path / "measure"
    completed = _run_train(config, work, steps=1, no_checkpoint=True)
    assert completed.returncode == 0, completed.stderr
    assert (work / "train.jsonl").is_file()
    assert not (work / "checkpoint.pt").exists()
    assert not (work / "checkpoint.pt.tmp").exists()
    done = json.loads((work / "done.json").read_text(encoding="utf-8"))
    assert done["checkpoint_saved"] is False
    assert done["steps"] == 1
    assert _steps(work / "train.jsonl")[0]["finite"] is True


def test_one_lambda_trains_and_resumes_on_cpu(tmp_path: Path):
    config = _local_config(tmp_path)
    work = tmp_path / "lambda-0"
    first = _run_train(config, work, steps=1)
    assert first.returncode == 0, first.stderr
    assert "dtype=fp32" in first.stdout
    assert "torch.compile=não" in first.stdout
    assert "WikiText-2" in first.stdout
    done = json.loads((work / "done.json").read_text(encoding="utf-8"))
    assert done["steps"] == 1
    assert done["lambda"] == 0.0
    assert done["finite"] is True
    assert done["checkpoint_saved"] is True
    assert done["seq_len"] == 8
    lines = _steps(work / "train.jsonl")
    assert [row["step"] for row in lines] == [1]
    assert lines[0]["beta"] == 0.01
    assert lines[0]["finite"] is True
    assert lines[0]["grad_norm"] > 0
    before = _weight_sum(work / "checkpoint.pt")
    blob = torch.load(work / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert all(tensor.dtype == torch.float32 for tensor in blob["model"].values())
    assert blob["architecture"]["E"] == 32
    assert blob["architecture"]["d_model"] == 256

    second = _run_train(config, work, steps=2)
    assert second.returncode == 0, second.stderr
    assert "retomando do passo 1" in second.stdout
    done = json.loads((work / "done.json").read_text(encoding="utf-8"))
    assert done["steps"] == 2
    lines = _steps(work / "train.jsonl")
    assert [row["step"] for row in lines] == [1, 2]
    assert [row["batch_rows"] for row in lines] == [2, 2]
    assert _weight_sum(work / "checkpoint.pt") != before

    third = _run_train(config, work, steps=3, batch_size=1)
    assert third.returncode == 0, third.stderr
    assert "retomando com batch 1" in third.stdout
    assert "retomando do passo 2" in third.stdout
    done = json.loads((work / "done.json").read_text(encoding="utf-8"))
    assert done["steps"] == 3
    assert done["batch_size"] == 1
    lines = _steps(work / "train.jsonl")
    assert [row["step"] for row in lines] == [1, 2, 3]
    assert lines[2]["batch_rows"] == 1
    assert list((work / "cold").glob("*.f32"))
    assert run_lambda_grid.is_complete(
        work / "done.json",
        lam=0.0,
        seed=0,
        seq_len=8,
        steps=2,
        epochs=None,
    )


def test_grid_shrinks_after_oom_skips_a_finished_lambda_and_keeps_going(tmp_path: Path):
    from moe_memory.config import load_config

    config_path = _local_config(tmp_path)
    config = load_config(config_path)
    output = tmp_path / "grid"
    # λ=0.5 já fechou uma época. A grade não pode relançar.
    middle = output / "lambda-0.5"
    middle.mkdir(parents=True)
    (middle / "done.json").write_text(
        json.dumps(_done(0.5, steps=4, seq_len=8, epochs_completed=1)),
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def launch(command: list[str], log_path: Path) -> int:
        calls.append(command)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        lam = float(command[command.index("--lambda") + 1])
        batch = int(command[command.index("--batch-size") + 1])
        work = Path(command[command.index("--work-dir") + 1])
        # Primeira tentativa de cada λ estoura; a segunda fecha o alvo.
        prior = sum(1 for previous in calls[:-1] if _lambda_of(previous) == lam)
        if prior == 0:
            log_path.write_text("out of memory\n", encoding="utf-8")
            return train_wikitext.EXIT_OOM
        (work / "done.json").write_text(
            json.dumps(_done(lam, steps=1, seq_len=8)),
            encoding="utf-8",
        )
        assert batch == 2
        return 0

    request = Namespace(
        config=config_path,
        device="cpu",
        seq_len=8,
        batch_size=4,
        seed=0,
        steps=1,
        epochs=None,
        checkpoint_every=1,
        work_dir=output,
    )
    summary = run_lambda_grid.run_grid(config, request, launch)
    assert [run["status"] for run in summary["runs"]] == ["done", "done", "done"]
    assert summary["runs"][1]["skipped"] is True
    launched = [_lambda_of(command) for command in calls]
    assert 0.5 not in launched
    assert launched.count(0.0) == 2
    assert launched.count(1.0) == 2
    assert summary["runs"][0]["recoveries"][0]["exit_code"] == train_wikitext.EXIT_OOM
    assert summary["runs"][0]["batch_size"] == 2
    saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert saved["lambda_grid"] == [0.0, 0.5, 1.0]
    assert saved["beta"] == 0.01
    assert "perplexity" not in json.dumps(saved)


def test_usage_error_does_not_shrink_the_batch(tmp_path: Path):
    from moe_memory.config import load_config

    config = load_config(_local_config(tmp_path))
    calls: list[int] = []

    def launch(command: list[str], log_path: Path) -> int:
        calls.append(int(command[command.index("--batch-size") + 1]))
        log_path.write_text("seq_len inválido\n", encoding="utf-8")
        return train_wikitext.EXIT_USAGE

    request = Namespace(
        config=None,
        device="cpu",
        seq_len=8,
        batch_size=4,
        seed=0,
        steps=1,
        epochs=None,
        checkpoint_every=1,
        work_dir=tmp_path / "grid",
    )
    summary = run_lambda_grid.run_grid(config, request, launch)
    assert calls == [4, 4, 4]
    assert all(run["status"] == "failed" for run in summary["runs"])
    assert run_lambda_grid.batch_ladder(4) == [4, 2, 1]
    assert run_lambda_grid.should_shrink(-9)
    assert run_lambda_grid.should_shrink(train_wikitext.EXIT_NONFINITE)
    assert not run_lambda_grid.should_shrink(train_wikitext.EXIT_USAGE)
    assert run_lambda_grid.should_resume_same(train_wikitext.EXIT_INTERRUPT)


def _lambda_of(command: list[str]) -> float:
    return float(command[command.index("--lambda") + 1])


def _run_train(
    config: Path,
    work: Path,
    *,
    steps: int,
    batch_size: int = 2,
    no_checkpoint: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(ROOT / "experiments" / "train_wikitext.py"),
        "--config",
        str(config),
        "--lambda",
        "0",
        "--steps",
        str(steps),
        "--seq-len",
        "8",
        "--batch-size",
        str(batch_size),
        "--checkpoint-every",
        "1",
        "--device",
        "cpu",
        "--work-dir",
        str(work),
    ]
    if no_checkpoint:
        command.append("--no-checkpoint")
    return subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _steps(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("event") == "step":
            rows.append(record)
    return rows


def _weight_sum(path: Path) -> float:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return float(blob["model"]["tok_emb.weight"].norm())
