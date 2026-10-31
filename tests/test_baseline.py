"""LM denso de referência: não é o MoE com λ=0.

A capacidade usa o YAML fechado. O passo real treina em CPU num corpus
local curto. A política de OOM não treina: um lançador falso devolve o
código de saída.
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from moe_memory.config import ModelConfig, load_config
from moe_memory.dense import (
    DenseLanguageModel,
    active_budget_gap,
    capacity_report,
    count_dense_parameters,
    count_moe_active_parameters,
    count_moe_parameters,
    prepare_baseline,
    train_dense_step,
)
from moe_memory.loss import task_cross_entropy
from moe_memory.model import MemoryMoEModel, MixtureOfExperts
from moe_memory.router import MemoryAwareRouter

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


train_baseline = _load("train_baseline_under_test", "train_baseline.py")
run_baseline = _load("run_baseline_under_test", "run_baseline.py")


def _local_config(tmp_path: Path) -> Path:
    wiki = tmp_path / "wiki"
    stories = tmp_path / "stories"
    wiki.mkdir()
    stories.mkdir()
    story = (
        "Once upon a time there was a small cat who loved the moon and the quiet field.\n"
        * 40
    )
    (stories / "train.txt").write_text(story, encoding="utf-8")
    (stories / "valid.txt").write_text(story, encoding="utf-8")
    for name in ("train.txt", "valid.txt", "test.txt"):
        (wiki / name).write_text((WIKI + "\n") * 30, encoding="utf-8")
    text = (ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
    text = text.replace('TinyStories: ""', f'TinyStories: "{stories}"')
    text = text.replace('WikiText-2: ""', f'WikiText-2: "{wiki}"')
    path = tmp_path / "local.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _dense_done(**overrides) -> dict:
    done = {
        "model": "dense",
        "memory_path": False,
        "corpus": "WikiText-2",
        "lambda": None,
        "seed": 0,
        "seq_len": 8,
        "batch_size": 2,
        "steps": 1,
        "epochs_completed": 0,
        "tokens": 16,
        "finite": True,
        "status": "done",
        "last_total": 1.5,
        "last_task": 1.5,
    }
    done.update(overrides)
    return done


def test_defaults_are_wikitext_not_a_lambda_run():
    args = train_baseline.parse_args([])
    assert args.corpus == "WikiText-2"
    assert args.epochs == 1
    assert args.steps is None
    assert args.batch_size == 4
    assert args.seed == 0
    assert args.seq_len is None
    assert args.device == "cpu"
    wiki_root = ROOT / "outputs" / "baseline" / "wikitext"
    assert args.work_dir == train_baseline.resolve_baseline_seed_dir(wiki_root, 0)
    with pytest.raises(SystemExit):
        train_baseline.parse_args(["--steps", "1", "--epochs", "1"])

    config = load_config()
    assert train_baseline.default_seq_len(config, "WikiText-2") == config.model.max_seq_len == 256
    assert train_baseline.canonical_corpus(config, "report") == "WikiText-2"
    runner = run_baseline.parse_args(["--device", "cpu"])
    request = run_baseline.resolve_request(config, runner)
    assert request.corpus_name == "WikiText-2"
    assert request.seq_len == 256
    assert request.seed is None
    assert request.seeds == [0, 1, 2]
    assert request.work_dir == ROOT / "outputs" / "baseline" / "wikitext"
    assert not train_baseline.losses_are_finite({"total": float("nan"), "task": 1.0, "grad_norm": 1.0})


def test_tinystories_is_optional_and_uses_the_smoke_length():
    stories = ROOT / "outputs" / "baseline" / "tinystories"
    args = train_baseline.parse_args(["--corpus", "TinyStories"])
    assert args.work_dir == train_baseline.resolve_baseline_seed_dir(stories, 0)
    dev = train_baseline.parse_args(["--corpus", "dev"])
    assert dev.work_dir == train_baseline.resolve_baseline_seed_dir(stories, 0)
    config = load_config()
    assert train_baseline.canonical_corpus(config, "dev") == "TinyStories"
    assert train_baseline.default_seq_len(config, "TinyStories") == config.smoke.seq_len
    request = run_baseline.resolve_request(
        config,
        run_baseline.parse_args(["--corpus", "TinyStories", "--device", "cpu"]),
    )
    assert request.corpus_name == "TinyStories"
    assert request.seq_len == config.smoke.seq_len
    assert request.work_dir == ROOT / "outputs" / "baseline" / "tinystories"


def test_dense_matches_active_budget_not_the_full_moe():
    config = load_config()
    model_cfg = config.model
    # 16_090_880 = tronco compartilhado + FFN de largura k*expert_hidden (1024).
    # 16_389_888 = tronco do MoE com roteador + k experts. 47_939_328 é o MoE inteiro.
    assert count_moe_parameters(model_cfg) == 47_939_328
    assert count_dense_parameters(model_cfg) == 16_090_880
    assert count_moe_active_parameters(model_cfg) == 16_389_888
    assert (
        count_moe_active_parameters(model_cfg) - count_dense_parameters(model_cfg)
        == active_budget_gap(model_cfg)
    )
    report = capacity_report(config)
    assert report["d_ff"] == model_cfg.k * model_cfg.expert_hidden == 1024
    assert report["memory_path"] is False
    assert report["router"] is False
    assert report["dense_parameters"] != report["moe_total_parameters"]

    torch.manual_seed(0)
    dense = DenseLanguageModel.from_config(config)
    assert dense.num_parameters() == 16_090_880
    assert dense.kind == "dense"
    assert dense.memory_path is False
    assert dense.d_ff == 1024
    assert not any(isinstance(module, MemoryAwareRouter) for module in dense.modules())
    assert not any(isinstance(module, MixtureOfExperts) for module in dense.modules())


def test_parameter_formulas_follow_the_constructors():
    cfg = ModelConfig(
        E=4,
        k=2,
        d_model=16,
        n_layers=2,
        n_heads=2,
        expert_hidden=4,
        max_seq_len=12,
        vocab_size=30,
        tokenizer="gpt2",
        dtype="fp32",
        tie_embeddings=True,
    )
    dense = DenseLanguageModel(
        n_layers=cfg.n_layers,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_ff=cfg.k * cfg.expert_hidden,
        vocab_size=cfg.vocab_size,
        max_seq_len=cfg.max_seq_len,
        tie_embeddings=True,
    )
    moe = MemoryMoEModel.from_model_config(cfg, use_level=True)
    moe_without_level = MemoryMoEModel.from_model_config(cfg, use_level=False)
    assert dense.num_parameters() == count_dense_parameters(cfg)
    assert moe.num_parameters() == moe_without_level.num_parameters() == count_moe_parameters(cfg)
    expert_n = sum(parameter.numel() for parameter in moe.blocks[0].ffn.experts[0].parameters())
    active = moe.num_parameters() - (cfg.E - cfg.k) * expert_n * cfg.n_layers
    assert count_moe_active_parameters(cfg) == active
    router_n = sum(parameter.numel() for parameter in moe.blocks[0].ffn.router.parameters())
    router_n *= cfg.n_layers
    bias_gap = cfg.n_layers * (cfg.k - 1) * cfg.d_model
    assert active - dense.num_parameters() == router_n + bias_gap
    assert active_budget_gap(cfg) == router_n + bias_gap

    untied = ModelConfig(
        E=cfg.E,
        k=cfg.k,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        expert_hidden=cfg.expert_hidden,
        max_seq_len=cfg.max_seq_len,
        vocab_size=cfg.vocab_size,
        tokenizer=cfg.tokenizer,
        dtype=cfg.dtype,
        tie_embeddings=False,
    )
    untied_dense = DenseLanguageModel(
        n_layers=untied.n_layers,
        d_model=untied.d_model,
        n_heads=untied.n_heads,
        d_ff=untied.k * untied.expert_hidden,
        vocab_size=untied.vocab_size,
        max_seq_len=untied.max_seq_len,
        tie_embeddings=False,
    )
    assert untied_dense.num_parameters() == count_dense_parameters(untied)
    assert active_budget_gap(untied) == active_budget_gap(cfg)


def test_step_is_task_loss_and_does_not_take_levels():
    torch.manual_seed(0)
    model = DenseLanguageModel(
        n_layers=1,
        d_model=8,
        n_heads=2,
        d_ff=4,
        vocab_size=20,
        max_seq_len=16,
        tie_embeddings=True,
    )
    ids = torch.randint(0, 20, (2, 6))
    expected = float(task_cross_entropy(model(ids), ids).detach())
    before = model.tok_emb.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    result = train_dense_step(model, optimizer, ids)
    assert result["total"] == result["task"] == pytest.approx(expected)
    assert math.isfinite(result["task"])
    assert result["grad_norm"] > 0
    assert not torch.equal(model.tok_emb.weight, before)
    with pytest.raises(TypeError):
        model(ids, torch.zeros(1, dtype=torch.long))


def test_checkpoint_save_keeps_the_live_adam_state(tmp_path: Path):
    config = load_config()
    model, optimizer = prepare_baseline(config, device=torch.device("cpu"), seed=0)
    ids = torch.randint(0, config.model.vocab_size, (2, 8))
    train_dense_step(model, optimizer, ids)
    parameter = model.tok_emb.weight
    live = optimizer.state[parameter]["exp_avg"]
    ptr = live.data_ptr()
    weight_ptr = parameter.data_ptr()
    copied = train_baseline._optimizer_state_cpu(optimizer)
    copied_tensor = next(iter(copied["state"].values()))["exp_avg"]
    assert copied_tensor.data_ptr() != ptr
    assert optimizer.state[parameter]["exp_avg"] is live
    state = {
        "model": model,
        "optimizer": optimizer,
        "config": config,
        "cursor": {
            "epoch": 0,
            "sequence_cursor": 2,
            "epochs_completed": 0,
            "steps": 1,
            "tokens": 16,
        },
        "corpus": "WikiText-2",
        "seed": 0,
        "seq_len": 8,
        "batch_size": 2,
        "lr": config.smoke.lr,
        "capacity": capacity_report(config),
        "checkpoint_path": tmp_path / "checkpoint.pt",
    }
    train_baseline._write_checkpoint(state)
    assert optimizer.state[parameter]["exp_avg"].data_ptr() == ptr
    assert optimizer.state[parameter]["exp_avg"].device == parameter.device
    assert model.tok_emb.weight.data_ptr() == weight_ptr
    blob = torch.load(tmp_path / "checkpoint.pt", map_location="cpu", weights_only=False)
    saved = next(iter(blob["optimizer"]["state"].values()))["exp_avg"]
    assert saved.device.type == "cpu"
    assert blob["model_kind"] == "dense"
    assert "emulator" not in blob
    train_dense_step(model, optimizer, ids)


def test_sources_do_not_call_the_memory_path():
    for name in ("train_baseline.py", "run_baseline.py"):
        text = (ROOT / "experiments" / name).read_text(encoding="utf-8")
        assert "prepare_train" not in text
        assert "ExpertResidency" not in text
        assert "ResidencyEmulator" not in text
        assert "compute_loss" not in text
        assert "torch.compile(" not in text
    dense = (ROOT / "src" / "moe_memory" / "dense.py").read_text(encoding="utf-8")
    assert "moe_memory.residency" not in dense
    assert "moe_memory.emulator" not in dense
    assert "ExpertResidency" not in dense
    assert "compute_loss(" not in dense
    assert "torch.compile(" not in dense


def test_lambda_flag_is_rejected_before_training(tmp_path: Path):
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments" / "train_baseline.py"),
            "--lambda",
            "0",
            "--steps",
            "1",
            "--work-dir",
            str(tmp_path / "run"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "unrecognized arguments" in completed.stderr
    assert not (tmp_path / "run").exists()


def test_wikitext_baseline_trains_and_resumes_on_cpu(tmp_path: Path):
    config = _local_config(tmp_path)
    work = tmp_path / "wikitext"
    first = _run_train(config, work, steps=1, corpus="WikiText-2")
    assert first.returncode == 0, first.stderr + first.stdout
    assert "Não é o MoE com λ=0" in first.stdout
    assert "dtype=fp32" in first.stdout
    assert "torch.compile=não" in first.stdout
    assert "loss=L_task" in first.stdout
    assert "WikiText-2" in first.stdout
    step_lines = [line for line in first.stdout.splitlines() if line.startswith("passo ")]
    assert step_lines
    assert "C_mem" not in step_lines[0]
    done = json.loads((work / "done.json").read_text(encoding="utf-8"))
    assert done["model"] == "dense"
    assert done["memory_path"] is False
    assert done["corpus"] == "WikiText-2"
    assert done["lambda"] is None
    assert done["status"] == "done"
    assert done["steps"] == 1
    assert done["tokens"] == 16
    assert done["finite"] is True
    assert done["last_total"] == done["last_task"]
    assert done["parameters"] == 16_090_880
    assert done["d_ff"] == 1024
    assert done["loss"] == "L_task"
    lines = _steps(work / "train.jsonl")
    assert [row["step"] for row in lines] == [1]
    assert lines[0]["model"] == "dense"
    assert lines[0]["total"] == lines[0]["task"]
    assert lines[0]["c_mem"] is None
    assert lines[0]["balance"] is None
    assert "swaps" not in lines[0]
    assert lines[0]["grad_norm"] > 0
    blob = torch.load(work / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert blob["model_kind"] == "dense"
    assert blob["architecture"]["kind"] == "dense"
    assert blob["architecture"]["memory_path"] is False
    assert blob["architecture"]["d_ff"] == 1024
    assert "E" not in blob["architecture"]
    assert all(tensor.dtype == torch.float32 for tensor in blob["model"].values())
    assert not (work / "cold").exists()
    before = float(blob["model"]["tok_emb.weight"].norm())

    second = _run_train(config, work, steps=2, corpus="WikiText-2")
    assert second.returncode == 0, second.stderr + second.stdout
    assert "retomando do passo 1" in second.stdout
    done = json.loads((work / "done.json").read_text(encoding="utf-8"))
    assert done["steps"] == 2
    lines = _steps(work / "train.jsonl")
    assert [row["step"] for row in lines] == [1, 2]
    assert [row["batch_rows"] for row in lines] == [2, 2]
    after = _weight_norm(work / "checkpoint.pt")
    assert after != before

    third = _run_train(config, work, steps=3, corpus="WikiText-2", batch_size=1)
    assert third.returncode == 0, third.stderr + third.stdout
    assert "retomando com batch 1" in third.stdout
    assert "retomando do passo 2" in third.stdout
    done = json.loads((work / "done.json").read_text(encoding="utf-8"))
    assert done["steps"] == 3
    assert done["batch_size"] == 1
    assert done["last_total"] == done["last_task"]
    lines = _steps(work / "train.jsonl")
    assert [row["step"] for row in lines] == [1, 2, 3]
    assert lines[2]["batch_rows"] == 1


def test_tinystories_optional_path_runs_one_step(tmp_path: Path):
    config = _local_config(tmp_path)
    work = tmp_path / "stories"
    completed = _run_train(config, work, steps=1, corpus="TinyStories", seq_len=None, batch_size=2)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert "TinyStories" in completed.stdout
    assert "desenvolvimento" in completed.stdout
    done = json.loads((work / "done.json").read_text(encoding="utf-8"))
    assert done["corpus"] == "TinyStories"
    assert done["seq_len"] == load_config().smoke.seq_len
    assert done["model"] == "dense"
    assert done["lambda"] is None
    assert done["last_total"] == done["last_task"]
    assert done["steps"] == 1


def test_runner_shrinks_after_oom_and_writes_a_comparable_summary(tmp_path: Path):
    from moe_memory.config import load_config as load

    config_path = _local_config(tmp_path)
    config = load(config_path)
    output = tmp_path / "grid"
    calls: list[list[str]] = []

    def launch(command: list[str], log_path: Path) -> int:
        calls.append(command)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        batch = int(command[command.index("--batch-size") + 1])
        work = Path(command[command.index("--work-dir") + 1])
        if len(calls) == 1:
            log_path.write_text("out of memory\n", encoding="utf-8")
            return train_baseline.EXIT_OOM
        assert batch == 2
        assert work == output / "seed-0"
        assert command[command.index("--seed") + 1] == "0"
        assert "--lambda" not in command
        assert "train_baseline.py" in command[1]
        assert command[command.index("--corpus") + 1] == "WikiText-2"
        (work / "done.json").write_text(
            json.dumps(_dense_done(batch_size=batch)),
            encoding="utf-8",
        )
        return 0

    request = Namespace(
        config=config_path,
        corpus="WikiText-2",
        corpus_name="WikiText-2",
        device="cpu",
        seq_len=8,
        batch_size=4,
        seed=0,
        steps=1,
        epochs=None,
        checkpoint_every=1,
        work_dir=output,
    )
    summary = run_baseline.run_baseline(config, request, launch)
    assert summary["model"] == "dense"
    assert summary["memory_path"] is False
    assert summary["corpus"] == "WikiText-2"
    assert summary["lambda"] is None
    assert summary["dtype"] == "fp32"
    assert summary["torch_compile"] is False
    assert summary["parameters"] == 16_090_880
    run = summary["runs"][0]
    assert run["status"] == "done"
    assert run["exit_code"] == 0
    assert run["batch_size"] == 2
    assert run["steps"] == 1
    assert run["tokens"] == 16
    assert run["last_total"] == run["last_task"] == 1.5
    assert run["lambda"] is None
    assert run["recoveries"][0]["exit_code"] == train_baseline.EXIT_OOM
    assert [item for item in summary["metric_fields"]] == [
        "status",
        "steps",
        "tokens",
        "last_total",
        "last_task",
    ]
    saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert saved["runs"][0]["status"] == "done"
    assert "perplexity" not in json.dumps(saved)


def test_usage_error_does_not_shrink_the_batch(tmp_path: Path):
    config = load_config(_local_config(tmp_path))
    calls: list[int] = []

    def launch(command: list[str], log_path: Path) -> int:
        calls.append(int(command[command.index("--batch-size") + 1]))
        log_path.write_text("seq_len inválido\n", encoding="utf-8")
        return train_baseline.EXIT_USAGE

    request = Namespace(
        config=None,
        corpus="WikiText-2",
        corpus_name="WikiText-2",
        device="cpu",
        seq_len=8,
        batch_size=4,
        seed=0,
        steps=1,
        epochs=None,
        checkpoint_every=1,
        work_dir=tmp_path / "grid",
    )
    summary = run_baseline.run_baseline(config, request, launch)
    assert calls == [4]
    assert summary["runs"][0]["status"] == "failed"
    assert run_baseline.batch_ladder(4) == [4, 2, 1]
    assert run_baseline.should_shrink(train_baseline.EXIT_OOM)
    assert not run_baseline.should_shrink(train_baseline.EXIT_USAGE)


def test_lambda_zero_done_json_is_not_a_finished_baseline(tmp_path: Path):
    path = tmp_path / "done.json"
    path.write_text(
        json.dumps(
            {
                "lambda": 0.0,
                "seed": 0,
                "seq_len": 8,
                "steps": 1,
                "finite": True,
                "last_total": 1.0,
                "last_task": 2.0,
            }
        ),
        encoding="utf-8",
    )
    assert not run_baseline.is_complete(
        path,
        corpus="WikiText-2",
        seed=0,
        seq_len=8,
        steps=1,
        epochs=None,
    )
    mislabeled = _dense_done()
    mislabeled["lambda"] = 0.0
    path.write_text(json.dumps(mislabeled), encoding="utf-8")
    assert not run_baseline.is_complete(
        path,
        corpus="WikiText-2",
        seed=0,
        seq_len=8,
        steps=1,
        epochs=None,
    )


def test_finished_baseline_is_not_relaunched(tmp_path: Path):
    config = load_config(_local_config(tmp_path))
    output = tmp_path / "wikitext"
    output.mkdir()
    (output / "done.json").write_text(json.dumps(_dense_done()), encoding="utf-8")
    calls: list[list[str]] = []

    def launch(command: list[str], log_path: Path) -> int:
        calls.append(command)
        return 0

    request = Namespace(
        config=None,
        corpus="WikiText-2",
        corpus_name="WikiText-2",
        device="cpu",
        seq_len=8,
        batch_size=4,
        seed=0,
        steps=1,
        epochs=None,
        checkpoint_every=1,
        work_dir=output,
    )
    summary = run_baseline.run_baseline(config, request, launch)
    assert calls == []
    assert summary["runs"][0]["skipped"] is True
    assert summary["runs"][0]["last_task"] == 1.5
    assert summary["runs"][0]["tokens"] == 16


def _run_train(
    config: Path,
    work: Path,
    *,
    steps: int,
    corpus: str,
    batch_size: int = 2,
    seq_len: int | None = 8,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(ROOT / "experiments" / "train_baseline.py"),
        "--config",
        str(config),
        "--corpus",
        corpus,
        "--steps",
        str(steps),
        "--batch-size",
        str(batch_size),
        "--checkpoint-every",
        "1",
        "--device",
        "cpu",
        "--work-dir",
        str(work),
    ]
    if seq_len is not None:
        command.extend(["--seq-len", str(seq_len)])
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


def _weight_norm(path: Path) -> float:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return float(blob["model"]["tok_emb.weight"].norm())
