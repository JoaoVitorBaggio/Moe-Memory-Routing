"""Adapter do Qwen3-MoE com o LSTM externo. CPU, peso aleatório, sem download."""

import math

import pytest
import torch

from moe_memory.qwen3_adapter import (
    TwoLevelQwen3,
    build_tiny_qwen3_backbone,
    language_model_loss,
    training_step,
    transformers_qwen3_available,
)
from moe_memory.lstm_ahead_router import TwoHorizonLstmRouter


def _router(n_layers: int, n_experts: int, **overrides: int) -> TwoHorizonLstmRouter:
    kwargs = dict(
        vocab_size=48,
        n_layers=n_layers,
        n_experts=n_experts,
        l1=1,
        l2=2,
        k1=2,
        k2=4,
        b2=4,
        embed_dim=16,
        hidden_size=16,
    )
    kwargs.update(overrides)
    return TwoHorizonLstmRouter(**kwargs)


def _model(backend: str, lora_impl: str = "auto") -> TwoLevelQwen3:
    torch.manual_seed(0)
    backbone = build_tiny_qwen3_backbone(
        vocab_size=48,
        hidden_size=64,
        n_layers=2,
        n_experts=8,
        intermediate_size=32,
        backend=backend,
    )
    return TwoLevelQwen3(backbone, _router(2, 8), lora_rank=4, lora_alpha=8.0, lora_impl=lora_impl)


def _backends() -> list[str]:
    found = ["mock"]
    if transformers_qwen3_available():
        found.append("transformers")
    return found


def _nudge_base_experts(model: TwoLevelQwen3) -> None:
    touched = 0
    with torch.no_grad():
        for name, parameter in model.backbone.named_parameters():
            if "lora_" in name:
                continue
            if any(token in name for token in ("gate_proj", "up_proj", "down_proj", "gate_up_proj")):
                parameter.add_(0.25)
                touched += 1
    if touched == 0:
        raise AssertionError("não achei peso base de expert")


def test_tiny_backend_is_real_qwen3_when_transformers_imports():
    mock = build_tiny_qwen3_backbone(backend="mock", hidden_size=32, intermediate_size=16, n_experts=4)
    assert mock.__class__.__name__ == "MockQwen3Moe"
    if transformers_qwen3_available():
        real = build_tiny_qwen3_backbone(backend="transformers", hidden_size=64, intermediate_size=32)
        assert real.__class__.__name__ == "Qwen3MoeForCausalLM"


@pytest.mark.parametrize("backend", _backends())
def test_original_gate_is_never_used_and_experts_are(backend: str):
    model = _model(backend, lora_impl="local")
    calls = {"n": 0}

    def boom(*args, **kwargs):
        del args, kwargs
        calls["n"] += 1
        raise AssertionError("o gate original foi chamado")

    for block in model.blocks:
        block.gate.forward = boom
    tokens = torch.randint(1, 48, (2, 6))
    first = model(tokens).logits.detach()
    assert calls["n"] == 0
    with torch.no_grad():
        for block in model.blocks:
            block.gate.weight.add_(1.0)
    second = model(tokens).logits.detach()
    assert torch.equal(first, second)
    _nudge_base_experts(model)
    third = model(tokens).logits.detach()
    assert not torch.allclose(second, third)


@pytest.mark.parametrize("backend", _backends())
def test_gradient_reaches_router_and_lora_only(backend: str):
    impls = ["local"]
    if backend == "mock":
        try:
            import peft  # noqa: F401
        except ImportError:
            peft = None
        if peft is not None:
            impls.append("peft")
    for impl in impls:
        model = _model(backend, lora_impl=impl)
        tokens = torch.randint(1, 48, (2, 8))
        output = model(tokens)
        loss = language_model_loss(output.logits, tokens)
        loss.backward()
        router_hit = False
        lora_hit = False
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                assert name.startswith("router.") or "lora_" in name, name
            if parameter.grad is None:
                continue
            assert parameter.grad.abs().sum() == 0 or parameter.requires_grad
            if not parameter.requires_grad:
                assert parameter.grad.abs().sum() == 0
                continue
            if parameter.grad.abs().sum() == 0:
                continue
            if name.startswith("router."):
                router_hit = True
            if "lora_" in name:
                lora_hit = True
        assert router_hit, impl
        assert lora_hit, impl
        # O tronco entra no grafo, não só o bias da cabeça.
        assert model.router.lstm.weight_ih_l0.grad is not None
        assert model.router.lstm.weight_ih_l0.grad.abs().sum() > 0
        assert model.router.embed.weight.grad is not None
        assert model.router.embed.weight.grad.abs().sum() > 0


def test_loss_is_plain_next_token_cross_entropy():
    model = _model("mock", lora_impl="local")
    tokens = torch.randint(1, 48, (2, 6))
    logits = model(tokens).logits
    loss = language_model_loss(logits, tokens)
    expected = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        tokens[:, 1:].reshape(-1),
    )
    assert torch.allclose(loss, expected)


@pytest.mark.parametrize("backend", _backends())
def test_lm_loss_drops_on_a_repeated_batch(backend: str):
    torch.manual_seed(0)
    model = _model(backend, lora_impl="local")
    pattern = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8])
    batch = pattern.view(1, -1).repeat(4, 1)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=2e-2)
    losses = [float(training_step(model, optimizer, batch)) for _ in range(30)]
    assert all(math.isfinite(value) for value in losses)
    assert losses[-1] < losses[0] - 1e-2
