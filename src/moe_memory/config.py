"""Carrega e valida ``configs/default.yaml``.

Os números do experimento ficam no YAML. Aqui só se recusa o que o código
não sabe executar (granularidade, dtype, nome da agregação top-2) ou o
que quebra a geometria (``k >= E``, orçamento que não cabe, sequência
acima do knob de 512).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from moe_memory.corpus import KNOWN_CORPORA
from moe_memory.levels import N_LEVELS

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PATH = _REPO_ROOT / "configs" / "default.yaml"
_MAX_SEQ_LEN_KNOB = 512


@dataclass(frozen=True)
class ModelConfig:
    E: int
    k: int
    d_model: int
    n_layers: int
    n_heads: int
    expert_hidden: int
    max_seq_len: int
    vocab_size: int
    tokenizer: str
    dtype: str
    tie_embeddings: bool

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


@dataclass(frozen=True)
class MemoryConfig:
    B_frac: float
    warm_slots: int
    emulator_granularity: str
    router_sees_level: bool

    def hot_slots(self, n_experts: int) -> int:
        return int(round(n_experts * self.B_frac))

    def scaled_hot_slots(self, n_experts: int, multiplier: float) -> int:
        """Teto quente em ``multiplier × B`` (dose–resposta na avaliação)."""

        slots = int(round(self.hot_slots(n_experts) * multiplier))
        if slots < 1 or slots + self.warm_slots > n_experts:
            raise ValueError(
                f"{multiplier}×B = {slots} vagas quentes não cabem com "
                f"warm_slots={self.warm_slots} e E={n_experts}"
            )
        return slots


@dataclass(frozen=True)
class CostConfig:
    hbm: float
    ddr: float
    nvme: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.hbm, self.ddr, self.nvme)


@dataclass(frozen=True)
class LossConfig:
    lambda_grid: tuple[float, ...]
    beta: float
    train_topk_agg: str
    infer_topk_agg: str

    def aggregation(self, training: bool) -> str:
        return self.train_topk_agg if training else self.infer_topk_agg


@dataclass(frozen=True)
class EvalConfig:
    B_mult: tuple[float, ...]
    seeds: int
    holdout_note: str


@dataclass(frozen=True)
class HardwareConfig:
    gpu: str
    vram_usable_gb: float
    ram_work_gb: float
    ssd_cold_gb: float
    wall_latency_on: str


@dataclass(frozen=True)
class CorpusConfig:
    dev: str
    report: str
    active: str
    cache_dir: str
    paths: dict[str, str]

    def resolve(self, which: str | None = None) -> str:
        """``None`` usa ``active``. ``dev`` / ``report`` escolhem o papel.

        O nome do corpus (``TinyStories``, ``WikiText-2``) também vale.
        """

        choice = self.active if which is None else which
        if choice == "dev":
            return self.dev
        if choice == "report":
            return self.report
        if choice == self.dev or choice == self.report:
            return choice
        raise ValueError(
            f"corpus {choice!r} não é 'dev', 'report', {self.dev!r} nem {self.report!r}"
        )


@dataclass(frozen=True)
class SmokeConfig:
    batch_size: int
    seq_len: int
    lr: float
    lambda_weight: float


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    memory: MemoryConfig
    cost: CostConfig
    loss: LossConfig
    eval: EvalConfig
    hardware: HardwareConfig
    corpus: CorpusConfig
    smoke: SmokeConfig
    path: Path


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    config_path = _DEFAULT_PATH if path is None else Path(path)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} não é um mapeamento YAML")
    config = _parse(raw, config_path)
    _validate(config)
    return config


def _parse(raw: dict[str, Any], path: Path) -> ExperimentConfig:
    model = _section(raw, "model")
    memory = _section(raw, "memory")
    cost = _section(raw, "c")
    loss = _section(raw, "loss")
    eval_ = _section(raw, "eval")
    hardware = _section(raw, "hardware_ref")
    corpus = _section(raw, "corpus")
    smoke = _section(raw, "smoke")
    return ExperimentConfig(
        model=ModelConfig(
            E=_as_int(model, "E"),
            k=_as_int(model, "k"),
            d_model=_as_int(model, "d_model"),
            n_layers=_as_int(model, "n_layers"),
            n_heads=_as_int(model, "n_heads"),
            expert_hidden=_as_int(model, "expert_hidden"),
            max_seq_len=_as_int(model, "max_seq_len"),
            vocab_size=_as_int(model, "vocab_size"),
            tokenizer=_as_str(model, "tokenizer"),
            dtype=_as_str(model, "dtype"),
            tie_embeddings=_as_bool(model, "tie_embeddings"),
        ),
        memory=MemoryConfig(
            B_frac=_as_float(memory, "B_frac"),
            warm_slots=_as_int(memory, "warm_slots"),
            emulator_granularity=_as_str(memory, "emulator_granularity"),
            router_sees_level=_as_bool(memory, "router_sees_level"),
        ),
        cost=CostConfig(
            hbm=_as_float(cost, "hbm"),
            ddr=_as_float(cost, "ddr"),
            nvme=_as_float(cost, "nvme"),
        ),
        loss=LossConfig(
            lambda_grid=tuple(_as_float_list(loss, "lambda_grid")),
            beta=_as_float(loss, "beta"),
            train_topk_agg=_as_str(loss, "train_topk_agg"),
            infer_topk_agg=_as_str(loss, "infer_topk_agg"),
        ),
        eval=EvalConfig(
            B_mult=tuple(_as_float_list(eval_, "B_mult")),
            seeds=_as_int(eval_, "seeds"),
            holdout_note=_as_str(eval_, "holdout_note"),
        ),
        hardware=HardwareConfig(
            gpu=_as_str(hardware, "gpu"),
            vram_usable_gb=_as_float(hardware, "vram_usable_gb"),
            ram_work_gb=_as_float(hardware, "ram_work_gb"),
            ssd_cold_gb=_as_float(hardware, "ssd_cold_gb"),
            wall_latency_on=_as_str(hardware, "wall_latency_on"),
        ),
        corpus=CorpusConfig(
            dev=_as_str(corpus, "dev"),
            report=_as_str(corpus, "report"),
            active=_as_str(corpus, "active"),
            cache_dir=_as_str(corpus, "cache_dir"),
            paths=_as_path_map(corpus, "paths"),
        ),
        smoke=SmokeConfig(
            batch_size=_as_int(smoke, "batch_size"),
            seq_len=_as_int(smoke, "seq_len"),
            lr=_as_float(smoke, "lr"),
            lambda_weight=_as_float(smoke, "lambda"),
        ),
        path=path,
    )


def _validate(config: ExperimentConfig) -> None:
    model = config.model
    memory = config.memory
    if model.E < 2:
        raise ValueError(f"E={model.E} precisa ser >= 2")
    if not 1 <= model.k < model.E:
        raise ValueError(f"k={model.k} precisa satisfazer 1 <= k < E={model.E}")
    if model.n_layers < 1:
        raise ValueError("n_layers precisa ser >= 1")
    if model.n_heads < 1 or model.d_model % model.n_heads != 0:
        raise ValueError(
            f"d_model={model.d_model} não é divisível por n_heads={model.n_heads}"
        )
    if model.expert_hidden < 1:
        raise ValueError("expert_hidden precisa ser >= 1")
    if not 1 <= model.max_seq_len <= _MAX_SEQ_LEN_KNOB:
        raise ValueError(
            f"max_seq_len={model.max_seq_len} fora de 1..{_MAX_SEQ_LEN_KNOB} "
            "(knob do WikiText-2)"
        )
    if model.vocab_size < 2:
        raise ValueError("vocab_size precisa ser >= 2")
    if model.dtype != "fp32":
        raise ValueError(
            f"dtype={model.dtype!r} não está implementado; o plano de hardware é fp32"
        )
    if model.tokenizer == "gpt2" and model.vocab_size != 50257:
        raise ValueError(
            f"tokenizer gpt2 tem vocab 50257, o YAML diz {model.vocab_size}"
        )
    if not model.tie_embeddings:
        warnings.warn(
            "tie_embeddings=false duplica a matriz do vocab e tende a passar de 50 M parâmetros",
            stacklevel=2,
        )

    if not 0 < memory.B_frac < 1:
        raise ValueError(f"B_frac={memory.B_frac} precisa estar em (0, 1)")
    hot = memory.hot_slots(model.E)
    if hot < 1 or memory.warm_slots < 1 or hot + memory.warm_slots >= model.E:
        raise ValueError(
            f"orçamento quente={hot}, morno={memory.warm_slots}, E={model.E} "
            "não deixa os três níveis ocupados no B de referência"
        )
    if memory.emulator_granularity != "batch":
        raise ValueError(
            f"granularidade {memory.emulator_granularity!r} não está implementada; "
            "o protocolo fecha emulador por batch"
        )
    for multiplier in config.eval.B_mult:
        memory.scaled_hot_slots(model.E, multiplier)

    costs = config.cost.as_tuple()
    if any(value <= 0 for value in costs):
        raise ValueError(f"custos c precisam ser positivos, recebido {costs}")
    if len(costs) != N_LEVELS:
        raise ValueError("c precisa ter hbm, ddr e nvme")

    if not config.loss.lambda_grid or any(value < 0 for value in config.loss.lambda_grid):
        raise ValueError(f"lambda_grid inválida: {config.loss.lambda_grid}")
    if 0.0 not in config.loss.lambda_grid:
        raise ValueError("lambda_grid precisa conter 0 (baseline 1, sem C_mem)")
    if config.loss.beta < 0:
        raise ValueError(f"beta={config.loss.beta} não pode ser negativo")
    if config.loss.train_topk_agg != "weighted_mean":
        raise ValueError(
            "train_topk_agg precisa ser 'weighted_mean' (proxy de treino, k=2)"
        )
    if config.loss.infer_topk_agg != "sum":
        raise ValueError("infer_topk_agg precisa ser 'sum' (custo realizado, k=2)")
    if config.eval.seeds < 1:
        raise ValueError("eval.seeds precisa ser >= 1")
    if config.smoke.batch_size < 1:
        raise ValueError("smoke.batch_size precisa ser >= 1")
    if not 2 <= config.smoke.seq_len <= model.max_seq_len:
        raise ValueError(
            f"smoke.seq_len={config.smoke.seq_len} precisa estar em "
            f"2..max_seq_len={model.max_seq_len}"
        )
    if config.smoke.lr <= 0:
        raise ValueError(f"smoke.lr={config.smoke.lr} precisa ser positivo")
    if config.smoke.lambda_weight not in config.loss.lambda_grid:
        raise ValueError(
            f"smoke.lambda={config.smoke.lambda_weight} não está na grade "
            f"{config.loss.lambda_grid}"
        )
    if config.corpus.active not in {"dev", "report"}:
        raise ValueError(
            f"corpus.active={config.corpus.active!r} precisa ser 'dev' ou 'report'"
        )
    names = {config.corpus.dev, config.corpus.report}
    unknown = names - KNOWN_CORPORA
    if unknown:
        known = ", ".join(sorted(KNOWN_CORPORA))
        raise ValueError(f"corpus desconhecido: {sorted(unknown)}. Implementados: {known}")
    missing = names - set(config.corpus.paths)
    extra = set(config.corpus.paths) - names
    if missing or extra:
        raise ValueError(
            "corpus.paths precisa ter exatamente "
            + ", ".join(sorted(names))
        )
    if not config.corpus.cache_dir.strip():
        raise ValueError("corpus.cache_dir não pode ser vazio")


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"falta a seção '{key}' no YAML")
    return value


def _require(section: dict[str, Any], key: str) -> Any:
    if key not in section:
        raise ValueError(f"falta a chave '{key}' no YAML")
    return section[key]


def _as_int(section: dict[str, Any], key: str) -> int:
    value = _require(section, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} precisa ser inteiro, recebido {value!r}")
    if int(value) != value:
        raise ValueError(f"{key} precisa ser inteiro, recebido {value!r}")
    return int(value)


def _as_float(section: dict[str, Any], key: str) -> float:
    value = _require(section, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} precisa ser numérico, recebido {value!r}")
    return float(value)


def _as_float_list(section: dict[str, Any], key: str) -> list[float]:
    value = _require(section, key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} precisa ser uma lista não vazia")
    numbers: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{key} tem item não numérico: {item!r}")
        numbers.append(float(item))
    return numbers


def _as_path_map(section: dict[str, Any], key: str) -> dict[str, str]:
    value = _require(section, key)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{key} precisa ser um mapeamento não vazio")
    paths: dict[str, str] = {}
    for name, path in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{key} tem chave inválida: {name!r}")
        if path is None:
            paths[name] = ""
        elif isinstance(path, str):
            paths[name] = path
        else:
            raise ValueError(f"{key}.{name} precisa ser texto, recebido {path!r}")
    return paths


def _as_str(section: dict[str, Any], key: str) -> str:
    value = _require(section, key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} precisa ser texto, recebido {value!r}")
    return value


def _as_bool(section: dict[str, Any], key: str) -> bool:
    value = _require(section, key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} precisa ser booleano, recebido {value!r}")
    return value
