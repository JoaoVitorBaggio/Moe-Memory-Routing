"""Um passo de treino. O atalho sintético não lê corpus e não é número de relatório.

``prepare_train`` + ``train_one_batch`` também servem para um batch real
(TinyStories). A loss impressa no smoke continua sendo de peso aleatório.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from moe_memory.config import ExperimentConfig
from moe_memory.emulator import ResidencyEmulator
from moe_memory.levels import COLD, HOT, WARM
from moe_memory.loss import compute_loss
from moe_memory.model import MemoryMoEModel
from moe_memory.residency import ExpertResidency, attach_residency


def fake_batch(config: ExperimentConfig, *, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randint(
        0,
        config.model.vocab_size,
        (config.smoke.batch_size, config.smoke.seq_len),
        generator=generator,
    )


def default_work_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "outputs" / "emulator_cold"


def build_emulator(
    config: ExperimentConfig,
    expert_nbytes: int,
    work_dir: str | Path | None = None,
) -> ResidencyEmulator:
    return ResidencyEmulator(
        n_layers=config.model.n_layers,
        n_experts=config.model.E,
        hot_slots=config.memory.hot_slots(config.model.E),
        warm_slots=config.memory.warm_slots,
        expert_nbytes=expert_nbytes,
        work_dir=work_dir,
    )


def level_cost_tensor(config: ExperimentConfig) -> torch.Tensor:
    return torch.tensor(config.cost.as_tuple(), dtype=torch.float32)


def prepare_train(
    config: ExperimentConfig,
    *,
    lambda_: float | None = None,
    work_dir: str | Path | None = None,
    device: torch.device | None = None,
    seed: int = 0,
) -> tuple[MemoryMoEModel, ResidencyEmulator, torch.optim.Optimizer, ExpertResidency]:
    """Modelo fp32, emulador, AdamW e a residência física.

    Com CUDA, os experts fora de ``B`` saem da VRAM antes do primeiro
    forward. No CPU o storage do parâmetro fica onde o autograd espera.
    """

    device = device or torch.device("cpu")
    work_dir = default_work_dir() if work_dir is None else work_dir
    lambda_weight = config.smoke.lambda_weight if lambda_ is None else float(lambda_)
    if lambda_weight not in config.loss.lambda_grid:
        raise ValueError(
            f"λ={lambda_weight} não está na grade {config.loss.lambda_grid}"
        )
    torch.manual_seed(seed)
    model = MemoryMoEModel.from_config(config).to(device=device, dtype=torch.float32)
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("o modelo saiu de fp32")
    emulator = build_emulator(config, model.expert_nbytes(), work_dir)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.smoke.lr)
    residency = attach_residency(model, emulator, device)
    residency.optimizer = optimizer
    if residency.backend is not None:
        residency.materialize(optimizer)
    return model, emulator, optimizer, residency


def train_one_batch(
    model: MemoryMoEModel,
    emulator: ResidencyEmulator,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    input_ids: torch.Tensor,
    *,
    lambda_weight: float,
    residency: ExpertResidency,
) -> dict[str, Any]:
    """Um forward, backward e ``AdamW.step``. O LRU anda depois do passo.

    ``materialize`` regrava o home com o ``ℓ(e)`` novo. ``staged_*`` é a
    cópia para o compute e não entra nos bytes do emulador.
    """

    if input_ids.ndim != 2 or input_ids.size(1) < 2:
        raise ValueError(f"input_ids precisa ser [B, T>=2], recebido {tuple(input_ids.shape)}")
    model.train()
    output = model(input_ids, emulator.levels)
    terms = compute_loss(
        output.logits,
        input_ids,
        output.probs,
        output.topk_index,
        emulator.levels,
        level_cost_tensor(config),
        lambda_=lambda_weight,
        beta=config.loss.beta,
        aggregation=config.loss.aggregation(training=True),
    )
    optimizer.zero_grad(set_to_none=True)
    terms.total.backward()
    grad_norm = _grad_norm(model)
    optimizer.step()
    stats = emulator.step(output.dispatched)
    store = residency.materialize(optimizer)
    counts = _level_counts(emulator)
    return {
        "parameters": model.num_parameters(),
        "lambda": lambda_weight,
        "beta": config.loss.beta,
        "grad_norm": grad_norm,
        "swaps": stats.swaps,
        "bytes_moved": stats.bytes_moved,
        "promotions": stats.promotions,
        "demotions": stats.demotions,
        "cold_files": store.n_cold_files,
        "warm_copies": store.n_warm,
        "expert_nbytes": model.expert_nbytes(),
        "physical_offload": residency.backend is not None,
        "staged_loads": residency.loads,
        "staged_bytes": residency.bytes_loaded,
        **counts,
        **terms.as_floats(),
    }


def run_train_step(
    config: ExperimentConfig,
    *,
    lambda_: float | None = None,
    work_dir: str | Path | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Forward, backward e um ``AdamW.step`` num batch falso. Depois atualiza o LRU."""

    device = device or torch.device("cpu")
    lambda_weight = config.smoke.lambda_weight if lambda_ is None else float(lambda_)
    model, emulator, optimizer, residency = prepare_train(
        config, lambda_=lambda_weight, work_dir=work_dir, device=device
    )
    return train_one_batch(
        model,
        emulator,
        optimizer,
        config,
        fake_batch(config).to(device),
        lambda_weight=lambda_weight,
        residency=residency,
    )


def run_eval_pass(
    config: ExperimentConfig,
    *,
    lambda_: float | None = None,
    work_dir: str | Path | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Um forward em ``eval`` com a agregação de inferência (soma dos top-2)."""

    device = device or torch.device("cpu")
    work_dir = default_work_dir() if work_dir is None else work_dir
    lambda_weight = config.smoke.lambda_weight if lambda_ is None else float(lambda_)
    torch.manual_seed(0)
    model = MemoryMoEModel.from_config(config).to(device=device, dtype=torch.float32)
    model.eval()
    emulator = build_emulator(config, model.expert_nbytes(), work_dir)
    input_ids = fake_batch(config).to(device)
    with torch.no_grad():
        output = model(input_ids, emulator.levels)
        terms = compute_loss(
            output.logits,
            input_ids,
            output.probs,
            output.topk_index,
            emulator.levels,
            level_cost_tensor(config),
            lambda_=lambda_weight,
            beta=config.loss.beta,
            aggregation=config.loss.aggregation(training=False),
        )
    floats = terms.as_floats()
    return {
        "parameters": model.num_parameters(),
        "lambda": lambda_weight,
        "perplexity_random_init": math.exp(floats["task"]),
        "aggregation": config.loss.infer_topk_agg,
        **_level_counts(emulator),
        **floats,
    }


def _grad_norm(model: MemoryMoEModel) -> float:
    total = torch.zeros((), dtype=torch.float32)
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        total = total + parameter.grad.detach().float().pow(2).sum()
    return float(total.sqrt())


def _level_counts(emulator: ResidencyEmulator) -> dict[str, int]:
    return {
        "hot": int((emulator.levels == HOT).sum().item()),
        "warm": int((emulator.levels == WARM).sum().item()),
        "cold": int((emulator.levels == COLD).sum().item()),
    }
