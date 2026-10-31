"""Hold-out de perplexidade. Não usa os checkpoints multi-GB de outputs/.

O corte de CPU empacota um WikiText local minúsculo. O roundtrip denso/MoE
constrói o modelo do YAML (fp32) e avalia poucos batches — não é a corrida
da GTX 1060.
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
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
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


holdout = _load("holdout_under_test", "run_holdout_eval.py")
train_wikitext = _load("train_wikitext_for_holdout", "train_wikitext.py")
train_baseline = _load("train_baseline_for_holdout", "train_baseline.py")


def _local_config(tmp_path: Path) -> Path:
    wiki = tmp_path / "wiki"
    stories = tmp_path / "stories"
    wiki.mkdir()
    stories.mkdir()
    (stories / "train.txt").write_text("Once upon a time a small cat walked.\n", encoding="utf-8")
    (stories / "valid.txt").write_text("Once upon a time a small cat walked.\n", encoding="utf-8")
    for name in ("train.txt", "valid.txt", "test.txt"):
        (wiki / name).write_text((WIKI + "\n") * 20, encoding="utf-8")
    text = (ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
    text = text.replace('TinyStories: ""', f'TinyStories: "{stories}"')
    text = text.replace('WikiText-2: ""', f'WikiText-2: "{wiki}"')
    path = tmp_path / "local.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _cursor(steps: int = 3) -> dict:
    return {
        "epoch": 0,
        "sequence_cursor": steps,
        "epochs_completed": 0,
        "steps": steps,
        "tokens": steps * 8,
    }


def _save_dense(path: Path, config, model) -> None:
    blob = {
        "version": train_baseline.CHECKPOINT_VERSION,
        "model_kind": "dense",
        "corpus": "WikiText-2",
        "seed": 0,
        "seq_len": 8,
        "batch_size": 2,
        "lr": config.smoke.lr,
        "architecture": train_baseline._architecture(config),
        "cursor": _cursor(),
        "model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, path)


def _save_moe(path: Path, config, model, *, levels_poison: torch.Tensor) -> None:
    blob = {
        "version": train_wikitext.CHECKPOINT_VERSION,
        "lambda": 0.0,
        "seed": 0,
        "seq_len": 8,
        "batch_size": 2,
        "lr": config.smoke.lr,
        "architecture": train_wikitext._architecture(config),
        "cursor": _cursor(),
        "emulator": {
            "hot_slots": config.memory.hot_slots(config.model.E),
            "levels": levels_poison,
        },
        "model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, path)


def _ok_run() -> dict:
    row = {"truncated": False, "nll_nats": 2.0}
    return {
        "status": "ok",
        "context_matches_checkpoint": True,
        "model_kind": "dense",
        "lambda_in_grid": None,
        "splits": {"valid": dict(row), "test": dict(row)},
    }


def test_perplexity_and_bits_per_byte_use_nats():
    assert holdout.perplexity(0.0) == 1.0
    assert holdout.perplexity(math.log(10)) == pytest.approx(10.0)
    assert holdout.perplexity(float("nan")) is None
    assert holdout.perplexity(800.0) is None
    assert holdout.bits_per_byte(math.log(2), 2) == pytest.approx(0.5)
    assert holdout.bits_per_byte(1.0, 0) is None


def test_accumulate_nll_is_the_training_cross_entropy():
    from moe_memory.loss import task_cross_entropy

    logits = torch.randn(2, 5, 11)
    ids = torch.randint(0, 11, (2, 5))
    nll_sum, n_pred = holdout.accumulate_nll(logits, ids)
    mean = float(task_cross_entropy(logits, ids))
    assert n_pred == 2 * 4
    assert nll_sum / n_pred == pytest.approx(mean, rel=1e-6)


def test_utf8_bytes_skip_eot_and_keep_a_token_prefix():
    from moe_memory.tokenizer import load_gpt2_encoding

    encoding = load_gpt2_encoding()
    text = "The cat sat on the mat."
    other = "Another article."
    ids = encoding.encode(text) + [encoding.eot_token] + encoding.encode(other)
    ids.append(encoding.eot_token)
    assert holdout.utf8_bytes(ids, encoding) == len(text.encode()) + len(other.encode())
    prefix = encoding.encode(text)[:4]
    assert holdout.utf8_bytes(prefix, encoding) == len(encoding.decode(prefix).encode("utf-8"))
    assert holdout.utf8_bytes([encoding.eot_token], encoding) == 0


def test_logical_batches_match_the_training_tail():
    assert list(holdout.iter_logical(5, 2, None)) == [(0, 2), (2, 2), (4, 1)]
    assert list(holdout.iter_logical(5, 2, 1)) == [(0, 2)]
    assert holdout.logical_count(5, 2, None) == 3
    assert list(holdout.iter_forward(4, 5, 2)) == [(4, 2), (6, 2), (8, 1)]


def test_dispatched_sum_clones_and_stays_integral():
    first = torch.tensor([[1, 0], [2, 3]])
    second = torch.tensor([[4, 5], [0, 1]])
    total = holdout.add_dispatched(None, first)
    first[0, 0] = 99
    assert int(total[0, 0]) == 1
    total = holdout.add_dispatched(total, second)
    assert torch.equal(total, torch.tensor([[5, 5], [2, 4]]))
    assert total.dtype == torch.long


def test_router_tally_matches_balance_loss_of_the_whole_batch():
    from moe_memory.loss import balance_loss

    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(2, 4, 3, 5, generator=generator)
    probs = logits.softmax(dim=-1)
    topk = torch.topk(logits, k=2, dim=-1).indices
    whole = holdout.RouterTally()
    whole.add(probs, topk)
    split = holdout.RouterTally()
    split.add(probs[:, :2], topk[:, :2])
    split.add(probs[:, 2:], topk[:, 2:])
    expected = float(
        torch.stack([balance_loss(probs[layer], topk[layer]) for layer in range(2)]).mean()
    )
    assert whole.value() == pytest.approx(expected, rel=1e-5, abs=1e-5)
    assert split.value() == pytest.approx(expected, rel=1e-5, abs=1e-5)


def test_pack_matches_the_train_loader_and_stays_on_wikitext(tmp_path: Path):
    from moe_memory.config import load_config
    from moe_memory.tokenizer import load_gpt2_encoding

    config = load_config(_local_config(tmp_path))
    assert config.corpus.active == "dev"
    packed = holdout.pack_split(config, "train", seq_len=8)
    trained = train_wikitext.load_train_sequences(config, seq_len=8)
    assert torch.equal(packed.sequences, trained)
    assert packed.tokens_discarded == packed.tokens_in_stream % 8
    assert packed.sequences.size(0) * 8 + packed.tokens_discarded == packed.tokens_in_stream
    assert packed.sequences.size(0) >= 4
    encoding = load_gpt2_encoding()
    text = encoding.decode(
        [token for token in packed.sequences.reshape(-1).tolist() if token != encoding.eot_token]
    )
    assert "moon" in text
    assert "Once upon" not in text
    valid = holdout.pack_split(config, "valid", seq_len=8)
    assert valid.split == "valid"
    assert int(valid.sequences.size(0)) >= 4


def test_discovered_paths_are_the_four_training_work_dirs():
    from moe_memory.config import load_config

    config = load_config()
    families = holdout.discover_families(ROOT / "outputs", config)
    assert [family.name for family in families] == [
        "baseline",
        "lambda-0",
        "lambda-0.5",
        "lambda-1",
    ]
    assert list(config.loss.lambda_grid) == [0.0, 0.5, 1.0]
    by_name = {family.name: family for family in families}
    direct_baseline = train_baseline.parse_args([])
    assert by_name["baseline"].path == direct_baseline.work_dir / "checkpoint.pt"
    assert direct_baseline.work_dir.name in {"seed-0", "wikitext"}
    assert by_name["baseline"].expected_kind == "dense"
    assert by_name["baseline"].expected_lambda is None
    for value, name in ((0.0, "lambda-0"), (0.5, "lambda-0.5"), (1.0, "lambda-1")):
        args = train_wikitext.parse_args(["--lambda", str(value)])
        assert by_name[name].path == args.work_dir / "checkpoint.pt"
        assert args.work_dir.parent.name in {"seed-0", "wikitext"}
        assert by_name[name].expected_kind == "moe"
        assert by_name[name].expected_lambda == value


def test_lambda_zero_is_moe_and_dense_is_not():
    assert holdout.checkpoint_kind({"lambda": 0.0, "architecture": {"E": 32}}) == "moe"
    assert (
        holdout.checkpoint_kind({"model_kind": "dense", "architecture": {"kind": "dense"}})
        == "dense"
    )
    with pytest.raises(SystemExit) as caught:
        holdout.checkpoint_kind(
            {"model_kind": "moe", "architecture": {"kind": "dense", "E": 32}}
        )
    assert caught.value.code == holdout.EXIT_USAGE


def test_header_rejects_the_wrong_class_without_building_a_model(capsys):
    from moe_memory.config import load_config

    config = load_config()
    blob = {
        "version": train_wikitext.CHECKPOINT_VERSION,
        "lambda": 0.0,
        "architecture": train_wikitext._architecture(config),
        "emulator": {"hot_slots": config.memory.hot_slots(config.model.E), "levels": "nao-usar"},
        "cursor": _cursor(),
        "seed": 0,
        "seq_len": 8,
        "batch_size": 2,
        "lr": config.smoke.lr,
    }
    family = holdout.Family(
        "baseline",
        Path("outputs/baseline/wikitext/checkpoint.pt"),
        expected_kind="dense",
    )
    with pytest.raises(SystemExit) as caught:
        holdout.validate_header(blob, config, holdout.KIND_MOE, family)
    assert caught.value.code == holdout.EXIT_USAGE
    assert "denso" in capsys.readouterr().err


def test_tinystories_dense_checkpoint_is_not_the_wikitext_holdout():
    from moe_memory.config import load_config

    config = load_config()
    blob = {
        "version": 1,
        "model_kind": "dense",
        "corpus": "TinyStories",
        "architecture": {},
    }
    family = holdout.Family("dev", Path("checkpoint.pt"))
    with pytest.raises(SystemExit) as caught:
        holdout.validate_header(blob, config, "dense", family)
    assert caught.value.code == holdout.EXIT_USAGE


def test_architecture_mismatch_stops_before_load(monkeypatch, tmp_path: Path):
    from moe_memory.config import load_config

    def boom(*_args, **_kwargs):
        raise AssertionError("construiu o modelo")

    monkeypatch.setattr(holdout, "load_model", boom)
    config = load_config()
    arch = dict(train_baseline._architecture(config))
    arch["d_model"] = 128
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "version": 1,
            "model_kind": "dense",
            "corpus": "WikiText-2",
            "architecture": arch,
        },
        path,
    )
    args = holdout.parse_args(["--checkpoint", str(path)])
    family = holdout.Family("baseline", path, expected_kind="dense")
    with pytest.raises(SystemExit) as caught:
        holdout.evaluate_family(family, config, args, torch.device("cpu"), {})
    assert caught.value.code == holdout.EXIT_USAGE


def test_forward_batch_stays_inside_the_checkpoint_batch():
    blob = {"batch_size": 4}
    assert holdout.resolve_batches(blob, None) == (4, 4, False)
    assert holdout.resolve_batches(blob, 1) == (4, 1, False)
    assert holdout.resolve_batches(blob, 8) == (4, 4, True)


def test_parse_args_rejects_a_train_split_and_mixed_paths():
    with pytest.raises(SystemExit) as caught:
        holdout.parse_args(["--splits", "train"])
    assert caught.value.code == holdout.EXIT_USAGE
    with pytest.raises(SystemExit) as mixed:
        holdout.parse_args(
            ["--checkpoint", "a.pt", "--work-dir", "outputs/baseline/wikitext"]
        )
    assert mixed.value.code == holdout.EXIT_USAGE
    args = holdout.parse_args([])
    assert args.device == "cpu"
    assert args.splits == ("valid", "test")
    assert args.explicit is False
    assert args.seq_len is None
    assert args.batch_size is None
    assert args.max_batches is None


def test_protocol_complete_needs_both_full_splits():
    assert holdout.protocol_is_complete([_ok_run()])
    truncated = _ok_run()
    truncated["splits"]["test"] = dict(truncated["splits"]["test"], truncated=True)
    assert not holdout.protocol_is_complete([truncated])
    missing = {"status": "missing", "context_matches_checkpoint": False, "splits": {}}
    assert not holdout.protocol_is_complete([_ok_run(), missing])
    only_valid = _ok_run()
    del only_valid["splits"]["test"]
    assert not holdout.run_is_complete(only_valid)
    shifted = _ok_run()
    shifted["context_matches_checkpoint"] = False
    assert not holdout.run_is_complete(shifted)
    moe = _ok_run()
    moe["model_kind"] = "moe"
    moe["lambda_in_grid"] = False
    assert not holdout.run_is_complete(moe)


def test_missing_families_write_an_incomplete_summary(tmp_path: Path):
    summary = tmp_path / "summary.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments" / "run_holdout_eval.py"),
            "--outputs",
            str(tmp_path / "empty"),
            "--summary",
            str(summary),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == holdout.EXIT_INCOMPLETE, completed.stderr
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["protocol"] == "wikitext-2-holdout-ppl"
    assert payload["dose_response"] is False
    assert payload["protocol_complete"] is False
    assert [run["name"] for run in payload["runs"]] == [
        "baseline",
        "lambda-0",
        "lambda-0.5",
        "lambda-1",
    ]
    assert [run["status"] for run in payload["runs"]] == ["missing", "missing", "missing", "missing"]
    assert "incompleto" in completed.stderr
    explicit = subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments" / "run_holdout_eval.py"),
            "--checkpoint",
            str(tmp_path / "ausente.pt"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert explicit.returncode == holdout.EXIT_USAGE
    assert "não achei" in explicit.stderr


def test_dense_checkpoint_roundtrip_matches_token_cross_entropy(tmp_path: Path):
    from moe_memory.config import load_config
    from moe_memory.dense import DenseLanguageModel
    from moe_memory.loss import task_cross_entropy

    config_path = _local_config(tmp_path)
    config = load_config(config_path)
    packed = holdout.pack_split(config, "valid", seq_len=8)
    assert int(packed.sequences.size(0)) > 2
    torch.manual_seed(0)
    model = DenseLanguageModel.from_config(config).eval()
    device = torch.device("cpu")
    reference = holdout.score_split(
        model,
        packed,
        kind=holdout.KIND_DENSE,
        device=device,
        logical_batch=2,
        forward_batch=2,
        max_logical=1,
        log_every=1,
    )
    micro = holdout.score_split(
        model,
        packed,
        kind=holdout.KIND_DENSE,
        device=device,
        logical_batch=2,
        forward_batch=1,
        max_logical=2,
        log_every=50,
    )
    whole = holdout.score_split(
        model,
        packed,
        kind=holdout.KIND_DENSE,
        device=device,
        logical_batch=2,
        forward_batch=2,
        max_logical=2,
        log_every=50,
    )
    batch = packed.sequences[:2]
    with torch.no_grad():
        expected = float(task_cross_entropy(model(batch), batch))
    assert reference["nll_nats"] == pytest.approx(expected, rel=1e-5, abs=1e-5)
    assert reference["c_mem"] is None
    assert reference["swaps"] is None
    assert micro["nll_nats"] == pytest.approx(whole["nll_nats"], rel=1e-4, abs=1e-4)
    assert reference["tokens_scored"] == 2 * 7
    work = tmp_path / "baseline" / "wikitext"
    _save_dense(work / "checkpoint.pt", config, model)
    del model
    summary = tmp_path / "summary.json"
    with pytest.raises(SystemExit) as caught:
        holdout.main(
            [
                "--config",
                str(config_path),
                "--checkpoint",
                str(work / "checkpoint.pt"),
                "--summary",
                str(summary),
                "--max-batches",
                "1",
                "--device",
                "cpu",
                "--log-every",
                "1",
            ]
        )
    assert caught.value.code == holdout.EXIT_INCOMPLETE
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["protocol_complete"] is False
    assert payload["dose_response"] is False
    assert payload["packing"] == "non_overlapping_seq_len"
    run = payload["runs"][0]
    assert run["status"] == "ok"
    assert run["model_kind"] == "dense"
    assert run["lambda"] is None
    assert run["memory_path"] is False
    assert run["steps"] == 3
    assert run["checkpoint_seq_len"] == 8
    assert run["eval_seq_len"] == 8
    assert run["context_matches_checkpoint"] is True
    assert run["checkpoint_batch_size"] == 2
    assert run["parameters"] == 16_090_880
    row = run["splits"]["valid"]
    assert row["truncated"] is True
    assert row["nll_nats"] == pytest.approx(reference["nll_nats"], rel=1e-5, abs=1e-5)
    assert row["perplexity"] == pytest.approx(math.exp(row["nll_nats"]), rel=1e-6)
    assert row["bits_per_byte"] == pytest.approx(
        (row["nll_nats_sum"] / math.log(2)) / row["utf8_bytes"], rel=1e-6
    )
    assert row["tokens_scored"] == row["sequences_scored"] * 7
    assert "test" in run["splits"]
    per_run = json.loads((work / "holdout.json").read_text(encoding="utf-8"))
    assert per_run["scope"] == "run"
    assert per_run["protocol_complete"] is False
    assert per_run["runs"][0]["model_kind"] == "dense"
    assert not (work / "holdout_lru").exists()
    assert list(work.glob("**/*.f32")) == []
    holdout.main(
        [
            "--config",
            str(config_path),
            "--checkpoint",
            str(work / "checkpoint.pt"),
            "--summary",
            str(summary),
            "--device",
            "cpu",
            "--log-every",
            "50",
        ]
    )
    done = json.loads(summary.read_text(encoding="utf-8"))
    assert done["protocol_complete"] is True
    assert done["max_logical_batches"] is None
    for split_name in ("valid", "test"):
        split_row = done["runs"][0]["splits"][split_name]
        assert split_row["truncated"] is False
        assert split_row["sequences_scored"] == split_row["sequences_available"]
        assert split_row["perplexity"] == pytest.approx(math.exp(split_row["nll_nats"]), rel=1e-6)
    per_run_done = json.loads((work / "holdout.json").read_text(encoding="utf-8"))
    assert per_run_done["scope"] == "run"
    assert per_run_done["protocol_complete"] is True


def test_moe_lambda_zero_resets_lru_per_split_and_ignores_saved_levels(tmp_path: Path):
    from moe_memory.config import load_config
    from moe_memory.levels import COLD
    from moe_memory.loss import task_cross_entropy
    from moe_memory.model import MemoryMoEModel
    from moe_memory.smoke import build_emulator

    config_path = _local_config(tmp_path)
    config = load_config(config_path)
    packed = holdout.pack_split(config, "valid", seq_len=8)
    assert int(packed.sequences.size(0)) >= 4
    torch.manual_seed(0)
    model = MemoryMoEModel.from_config(config).eval()
    device = torch.device("cpu")
    emulator = build_emulator(config, model.expert_nbytes(), tmp_path / "unused_lru")
    fine = holdout.score_split(
        model,
        packed,
        kind=holdout.KIND_MOE,
        device=device,
        logical_batch=2,
        forward_batch=1,
        max_logical=2,
        log_every=50,
        emulator=emulator,
        config=config,
        lambda_weight=0.0,
    )
    coarse = holdout.score_split(
        model,
        packed,
        kind=holdout.KIND_MOE,
        device=device,
        logical_batch=2,
        forward_batch=2,
        max_logical=2,
        log_every=50,
        emulator=emulator,
        config=config,
        lambda_weight=0.0,
    )
    assert fine["swaps"] == coarse["swaps"]
    assert fine["bytes_moved"] == coarse["bytes_moved"]
    assert fine["lru_tick_end"] == 2
    assert fine["nll_nats"] == pytest.approx(coarse["nll_nats"], rel=1e-4, abs=1e-4)
    assert fine["c_mem"] == pytest.approx(coarse["c_mem"], rel=1e-4, abs=1e-4)
    assert fine["balance"] == pytest.approx(coarse["balance"], rel=1e-4, abs=1e-4)
    assert fine["aggregation"] == "sum"
    emulator.reset()
    batch = packed.sequences[:2]
    with torch.no_grad():
        reset_out = model(batch, emulator.levels)
        reset_nll = float(task_cross_entropy(reset_out.logits, batch))
        emulator.levels[:] = COLD
        cold_out = model(batch, emulator.levels)
        cold_nll = float(task_cross_entropy(cold_out.logits, batch))
    assert reset_nll == pytest.approx(
        holdout.score_split(
            model,
            packed,
            kind=holdout.KIND_MOE,
            device=device,
            logical_batch=2,
            forward_batch=2,
            max_logical=1,
            log_every=50,
            emulator=emulator,
            config=config,
            lambda_weight=0.0,
        )["nll_nats"],
        rel=1e-5,
        abs=1e-5,
    )
    assert reset_nll != pytest.approx(cold_nll, rel=1e-4, abs=1e-4)
    work = tmp_path / "wikitext" / "lambda-0"
    poison = torch.full((config.model.n_layers, config.model.E), COLD)
    _save_moe(work / "checkpoint.pt", config, model, levels_poison=poison)
    del model
    summary = tmp_path / "summary.json"
    with pytest.raises(SystemExit) as caught:
        holdout.main(
            [
                "--config",
                str(config_path),
                "--work-dir",
                str(work),
                "--summary",
                str(summary),
                "--max-batches",
                "1",
                "--device",
                "cpu",
                "--log-every",
                "1",
            ]
        )
    assert caught.value.code == holdout.EXIT_INCOMPLETE
    payload = json.loads(summary.read_text(encoding="utf-8"))
    run = payload["runs"][0]
    assert run["name"] == "lambda-0"
    assert run["model_kind"] == "moe"
    assert run["lambda"] == 0.0
    assert run["lambda_in_grid"] is True
    assert run["memory_path"] is True
    assert run["residency"] == "lru_reset_per_split"
    assert run["parameters"] == 47_939_328
    assert run["steps"] == 3
    valid = run["splits"]["valid"]
    test = run["splits"]["test"]
    assert valid["nll_nats"] == pytest.approx(reset_nll, rel=1e-5, abs=1e-5)
    assert valid["nll_nats"] != pytest.approx(cold_nll, rel=1e-4, abs=1e-4)
    assert valid["lru_tick_end"] == 1
    assert test["lru_tick_end"] == 1
    assert valid["truncated"] is True
    assert test["truncated"] is True
    assert not (work / "holdout_lru").exists()
    assert list(work.glob("**/*.f32")) == []
