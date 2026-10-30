"""O YAML de partida continua sendo o desenho fechado, mais a arquitetura fina."""

from pathlib import Path

import pytest

from moe_memory.config import load_config
from moe_memory.levels import swap_penalty

ROOT = Path(__file__).resolve().parents[1]


def test_default_yaml_locks_the_blueprint():
    config = load_config()
    assert config.path == ROOT / "configs" / "default.yaml"
    assert config.model.E == 32
    assert config.model.k == 2
    assert config.model.d_model == 256
    assert config.model.n_layers == 4
    assert config.model.n_heads == 4
    assert config.model.d_head == 64
    assert config.model.expert_hidden == 512
    assert config.model.max_seq_len == 256
    assert config.model.vocab_size == 50257
    assert config.model.tokenizer == "gpt2"
    assert config.model.dtype == "fp32"
    assert config.model.tie_embeddings is True
    assert config.memory.hot_slots(config.model.E) == 8
    assert config.memory.warm_slots == 8
    assert config.memory.emulator_granularity == "batch"
    assert config.memory.router_sees_level is True
    assert config.cost.as_tuple() == (1.0, 10.0, 20.0)
    assert config.loss.lambda_grid == (0.0, 0.5, 1.0)
    assert config.loss.beta == pytest.approx(0.01)
    assert config.loss.train_topk_agg == "weighted_mean"
    assert config.loss.infer_topk_agg == "sum"
    assert config.loss.aggregation(training=True) == "weighted_mean"
    assert config.loss.aggregation(training=False) == "sum"
    assert config.eval.B_mult == (0.5, 1.0, 2.0)
    assert config.eval.seeds == 3
    assert [config.memory.scaled_hot_slots(32, mult) for mult in config.eval.B_mult] == [4, 8, 16]
    assert config.corpus.dev == "TinyStories"
    assert config.corpus.report == "WikiText-2"
    assert config.hardware.gpu == "GTX 1060"
    assert config.smoke.lambda_weight == 0.5


def test_swap_penalty_is_absolute_cost_gap():
    hbm, ddr, nvme = load_config().cost.as_tuple()
    assert swap_penalty(hbm, hbm) == 0
    assert swap_penalty(ddr, hbm) == 9
    assert swap_penalty(nvme, hbm) == 19
    assert swap_penalty(nvme, ddr) == 10


def test_yaml_mentions_the_expert_upgrade_path():
    text = (ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
    assert "expert_hidden: 512" in text
    assert "768" in text


def test_loader_rejects_a_different_aggregation(tmp_path: Path):
    source = (ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
    broken = source.replace("train_topk_agg: weighted_mean", "train_topk_agg: mean")
    path = tmp_path / "broken.yaml"
    path.write_text(broken, encoding="utf-8")
    with pytest.raises(ValueError, match="weighted_mean"):
        load_config(path)
