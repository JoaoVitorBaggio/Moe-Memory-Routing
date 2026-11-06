"""Preditor LSTM do segundo modelo.

Uma LSTM de duas camadas lê o token e devolve um logit por expert.
A previsão é por token, não por camada do transformer. O backbone do
MoE grande fica congelado: o passo offline só atualiza este módulo e
destila os logits de um roteador que não precisa caber na GPU.

Se a acurácia desse roteador ficar abaixo do piso configurado, o
critério pede hidden 512. A função só devolve o tamanho. Não retreina
e não troca o peso sozinha.

O fallback periódico substitui a previsão pelo índice do roteador
verdadeiro, para cortar o erro que a LSTM acumula ao longo da sequência.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch import nn

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PATH = _REPO_ROOT / "configs" / "lstm_predictor.yaml"


@dataclass(frozen=True)
class LstmPredictorConfig:
    vocab_size: int
    n_experts: int
    k: int
    n_model_layers: int
    embedding_dim: int = 128
    hidden_size: int = 256
    n_lstm_layers: int = 2
    fallback_every: int = 8
    router_accuracy_floor: float = 0.9
    hidden_if_router_accuracy_drops: int = 512
    ssd_lookahead_layers: int = 4

    def validate(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size precisa ser >= 2")
        if self.n_experts < 2:
            raise ValueError("n_experts precisa ser >= 2")
        if not 1 <= self.k < self.n_experts:
            raise ValueError(f"k={self.k} precisa satisfazer 1 <= k < E={self.n_experts}")
        if self.n_model_layers < 1:
            raise ValueError("n_model_layers precisa ser >= 1")
        if self.embedding_dim < 1:
            raise ValueError("embedding_dim precisa ser >= 1")
        if self.hidden_size < 1:
            raise ValueError("hidden_size precisa ser >= 1")
        if self.n_lstm_layers < 1:
            raise ValueError("n_lstm_layers precisa ser >= 1")
        if self.fallback_every < 1:
            raise ValueError("fallback_every precisa ser >= 1")
        if not 0 < self.router_accuracy_floor <= 1:
            raise ValueError(
                f"router_accuracy_floor={self.router_accuracy_floor} precisa estar em (0, 1]"
            )
        if self.hidden_if_router_accuracy_drops < self.hidden_size:
            raise ValueError(
                "hidden_if_router_accuracy_drops precisa ser maior que hidden_size"
            )
        if self.ssd_lookahead_layers not in (3, 4):
            raise ValueError(
                f"ssd_lookahead_layers={self.ssd_lookahead_layers} precisa ser 3 ou 4"
            )


@dataclass(frozen=True)
class ExpertPrediction:
    logits: torch.Tensor
    index: torch.Tensor
    scores: torch.Tensor
    fallback_mask: torch.Tensor


def recommended_hidden_size(config: LstmPredictorConfig, router_accuracy: float) -> int:
    """Tamanho a usar na próxima construção. Não cria módulo e não treina."""

    if not 0.0 <= router_accuracy <= 1.0:
        raise ValueError(f"acurácia {router_accuracy} precisa estar em [0, 1]")
    if router_accuracy < config.router_accuracy_floor:
        return config.hidden_if_router_accuracy_drops
    return config.hidden_size


class ExpertLSTMPredictor(nn.Module):
    def __init__(self, config: LstmPredictorConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_size,
            num_layers=config.n_lstm_layers,
            batch_first=True,
        )
        self.head = nn.Linear(config.hidden_size, config.n_experts)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.embed.weight, std=0.02)
        for name, parameter in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(parameter)
            else:
                nn.init.zeros_(parameter)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """``token_ids`` é ``[B, T]`` long. Sai ``[B, T, E]``, um vetor por token."""

        if token_ids.ndim != 2:
            raise ValueError(f"token_ids tem shape {tuple(token_ids.shape)}; esperado [B, T]")
        embedded = self.embed(token_ids)
        sequence, _state = self.lstm(embedded)
        return self.head(sequence)

    def predict(
        self,
        token_ids: torch.Tensor,
        *,
        teacher_index: torch.Tensor | None = None,
        token_offset: int = 0,
    ) -> ExpertPrediction:
        """Top-k por token. Nos passos de fallback, o índice é o do roteador verdadeiro."""

        logits = self.forward(token_ids)
        batch, seq, _experts = logits.shape
        k = self.config.k
        index = torch.topk(logits, k=k, dim=-1).indices
        probs = logits.softmax(dim=-1)
        scores = probs.gather(-1, index)
        positions = torch.arange(seq, device=token_ids.device) + token_offset
        fallback = torch.remainder(positions, self.config.fallback_every) == 0
        if bool(fallback.any()):
            if teacher_index is None:
                raise ValueError(
                    "fallback periódico precisa do índice do roteador verdadeiro"
                )
            if teacher_index.shape != (batch, seq, k):
                raise ValueError(
                    f"teacher_index tem shape {tuple(teacher_index.shape)}; "
                    f"esperado {(batch, seq, k)}"
                )
            mask = fallback.view(1, seq, 1)
            index = torch.where(mask, teacher_index, index)
            teacher_scores = torch.ones(batch, seq, k, device=logits.device, dtype=scores.dtype)
            scores = torch.where(mask, teacher_scores, scores)
        return ExpertPrediction(
            logits=logits,
            index=index,
            scores=scores,
            fallback_mask=fallback,
        )


def freeze_module(module: nn.Module) -> None:
    """Congela o backbone. O preditor não entra aqui."""

    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """KL do professor congelado para o aluno. O gradiente não volta ao professor."""

    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"logits {tuple(student_logits.shape)} e {tuple(teacher_logits.shape)} diferem"
        )
    teacher = teacher_logits.detach().softmax(dim=-1)
    log_student = student_logits.log_softmax(dim=-1)
    return F.kl_div(log_student, teacher, reduction="batchmean")


def offline_train_step(
    predictor: ExpertLSTMPredictor,
    token_ids: torch.Tensor,
    teacher_logits: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    backbone: nn.Module | None = None,
) -> float:
    """Um passo só nos pesos do preditor. ``teacher_logits`` já vem de fora."""

    if backbone is not None:
        frozen = [parameter.requires_grad for parameter in backbone.parameters()]
        if any(frozen):
            raise RuntimeError("o backbone precisa estar congelado antes do passo offline")
    if not any(parameter.requires_grad for parameter in predictor.parameters()):
        raise RuntimeError("o preditor não tem parâmetro treinável")
    predictor.train()
    optimizer.zero_grad(set_to_none=True)
    loss = distillation_loss(predictor(token_ids), teacher_logits)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def load_lstm_config(path: str | Path | None = None) -> LstmPredictorConfig:
    config_path = _DEFAULT_PATH if path is None else Path(path)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} não é um mapeamento YAML")
    section = raw.get("lstm_predictor")
    if not isinstance(section, dict):
        raise ValueError(f"{config_path} não tem a seção lstm_predictor")
    config = LstmPredictorConfig(
        vocab_size=_as_int(section, "vocab_size"),
        n_experts=_as_int(section, "n_experts"),
        k=_as_int(section, "k"),
        n_model_layers=_as_int(section, "n_model_layers"),
        embedding_dim=_as_int(section, "embedding_dim"),
        hidden_size=_as_int(section, "hidden_size"),
        n_lstm_layers=_as_int(section, "n_lstm_layers"),
        fallback_every=_as_int(section, "fallback_every"),
        router_accuracy_floor=_as_float(section, "router_accuracy_floor"),
        hidden_if_router_accuracy_drops=_as_int(section, "hidden_if_router_accuracy_drops"),
        ssd_lookahead_layers=_as_int(section, "ssd_lookahead_layers"),
    )
    config.validate()
    return config


def _as_int(section: dict[str, Any], key: str) -> int:
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
        raise ValueError(f"{key} precisa ser inteiro, recebido {value!r}")
    return int(value)


def _as_float(section: dict[str, Any], key: str) -> float:
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} precisa ser numérico, recebido {value!r}")
    return float(value)
