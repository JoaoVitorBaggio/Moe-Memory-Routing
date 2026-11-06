"""LSTM por token, fallback no roteador verdadeiro e passo offline."""

import torch
from torch import nn

from moe_memory.lstm_predictor import (
    ExpertLSTMPredictor,
    LstmPredictorConfig,
    distillation_loss,
    freeze_module,
    load_lstm_config,
    offline_train_step,
    recommended_hidden_size,
    reshape_teacher_logits,
)
from moe_memory.olmoe_teacher import load_teacher_trace, save_teacher_trace
from moe_memory.sida_budget import ActivationBudget, budgets_match


def _config(**overrides) -> LstmPredictorConfig:
    raw = dict(
        vocab_size=32,
        n_experts=4,
        k=2,
        n_model_layers=4,
        embedding_dim=8,
        hidden_size=16,
        n_lstm_layers=2,
        fallback_every=2,
        router_accuracy_floor=0.9,
        hidden_if_router_accuracy_drops=32,
        ssd_lookahead_layers=4,
    )
    raw.update(overrides)
    config = LstmPredictorConfig(**raw)
    config.validate()
    return config


def test_yaml_defaults_match_the_project_decision():
    config = load_lstm_config()
    assert config.hidden_size == 256
    assert config.n_lstm_layers == 2
    assert config.embedding_dim == 128
    assert config.hidden_if_router_accuracy_drops == 512
    assert config.k == 2
    assert config.n_experts == 64
    assert config.n_model_layers == 16
    predictor = ExpertLSTMPredictor(config)
    assert predictor.lstm.num_layers == 2
    assert predictor.lstm.hidden_size == 256
    assert predictor.embed.embedding_dim == 128
    assert predictor.head.out_features == 16 * 64


def test_each_layer_keeps_its_own_expert_logits():
    predictor = ExpertLSTMPredictor(_config(n_model_layers=3, n_experts=4))
    tokens = torch.randint(0, 32, (2, 5))
    logits = predictor(tokens)
    assert logits.shape == (2, 5, 3, 4)


def test_low_router_accuracy_recommends_a_wider_hidden_without_retraining():
    config = load_lstm_config()
    predictor = ExpertLSTMPredictor(_config())
    before = [parameter.detach().clone() for parameter in predictor.parameters()]
    assert recommended_hidden_size(config, router_accuracy=0.5) == 512
    assert recommended_hidden_size(config, router_accuracy=0.9) == 256
    assert recommended_hidden_size(config, router_accuracy=1.0) == 256
    after = list(predictor.parameters())
    assert all(torch.equal(left, right) for left, right in zip(before, after, strict=True))


def test_periodic_fallback_replaces_only_the_scheduled_tokens():
    predictor = ExpertLSTMPredictor(_config(fallback_every=2, n_model_layers=4))
    tokens = torch.zeros(1, 4, dtype=torch.long)
    teacher = torch.tensor([[[1, 2], [3, 0], [1, 2], [2, 3]]]).unsqueeze(2).expand(1, 4, 4, 2).contiguous()
    prediction = predictor.predict(tokens, teacher_index=teacher, token_offset=0)
    assert prediction.fallback_mask.tolist() == [True, False, True, False]
    assert torch.equal(prediction.index[0, 0], teacher[0, 0])
    assert torch.equal(prediction.index[0, 2], teacher[0, 2])
    assert torch.equal(prediction.scores[0, 0], torch.ones(4, 2))
    student = predictor.predict(tokens[:, 1:2], teacher_index=None, token_offset=1)
    assert student.fallback_mask.tolist() == [False]
    assert student.index.shape == (1, 1, 4, 2)


def test_fallback_without_the_true_router_is_rejected():
    predictor = ExpertLSTMPredictor(_config(fallback_every=1))
    tokens = torch.zeros(1, 1, dtype=torch.long)
    try:
        predictor.predict(tokens)
    except ValueError as exc:
        assert "roteador verdadeiro" in str(exc)
    else:
        raise AssertionError("o fallback sem professor deveria falhar")


def test_offline_step_updates_the_predictor_and_leaves_the_backbone_frozen():
    config = _config()
    predictor = ExpertLSTMPredictor(config)
    backbone = nn.Linear(4, 4)
    freeze_module(backbone)
    backbone_before = [parameter.detach().clone() for parameter in backbone.parameters()]
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=1e-2)
    tokens = torch.randint(0, config.vocab_size, (2, 3))
    teacher = torch.randn(2, 3, config.n_model_layers, config.n_experts)
    first = offline_train_step(predictor, tokens, teacher, optimizer, backbone=backbone)
    second = offline_train_step(predictor, tokens, teacher, optimizer, backbone=backbone)
    assert first >= 0.0 and second >= 0.0
    assert all(
        torch.equal(left, right)
        for left, right in zip(backbone_before, backbone.parameters(), strict=True)
    )
    assert all(not parameter.requires_grad for parameter in backbone.parameters())
    teacher_grad = teacher.clone().requires_grad_(True)
    loss = distillation_loss(predictor(tokens), teacher_grad)
    loss.backward()
    assert teacher_grad.grad is None or torch.all(teacher_grad.grad == 0)


def test_activation_budget_matches_sida_at_the_same_k():
    config = load_lstm_config()
    ours = ActivationBudget(k=config.k, n_layers=config.n_model_layers)
    sida = ActivationBudget(k=config.k, n_layers=config.n_model_layers)
    wider = ActivationBudget(k=config.k + 1, n_layers=config.n_model_layers)
    assert budgets_match(ours, sida)
    assert not budgets_match(ours, wider)
    assert ours.experts_per_token_per_layer == config.k
    assert ours.expert_forwards_per_token == config.k * config.n_model_layers
    predictor = ExpertLSTMPredictor(_config(k=config.k, n_experts=8))
    tokens = torch.randint(0, 32, (1, 2))
    prediction = predictor.predict(
        tokens,
        teacher_index=torch.zeros(1, 2, predictor.config.n_model_layers, config.k, dtype=torch.long),
    )
    assert prediction.index.shape[-1] == ours.experts_per_token_per_layer
    assert prediction.logits.shape[-2:] == (
        predictor.config.n_model_layers,
        predictor.config.n_experts,
    )


def test_dump_tensor_enters_the_step_without_pooling_layers(tmp_path):
    config = _config(n_model_layers=2, n_experts=4, vocab_size=16)
    predictor = ExpertLSTMPredictor(config)
    tokens = torch.randint(0, config.vocab_size, (2, 3))
    flat = torch.zeros(6, 2, 4)
    flat[:, 0, 0] = 4
    flat[:, 1, 3] = 4
    aligned = reshape_teacher_logits(flat, tokens, n_layers=2, n_experts=4)
    assert aligned.shape == (2, 3, 2, 4)
    assert not torch.allclose(aligned[:, :, 0], aligned[:, :, 1])
    student = predictor(tokens)
    assert student.shape == (2, 3, 2, 4)
    by_layer = distillation_loss(student, aligned)
    by_flat_rows = distillation_loss(
        student.reshape(2, 6, 4),
        aligned.reshape(2, 6, 4),
    )
    assert torch.allclose(by_layer, by_flat_rows)
    path = tmp_path / "traco.pt"
    save_teacher_trace(path, tokens.reshape(-1), flat, split="train", top_k=2)
    blob = load_teacher_trace(path)
    assert tuple(blob["teacher_logits"].shape) == (6, 2, 4)
    from_file = reshape_teacher_logits(blob, tokens, n_layers=2, n_experts=4)
    assert torch.equal(from_file, aligned)
    optimizer = torch.optim.SGD(predictor.parameters(), lr=0.0)
    value = offline_train_step(predictor, tokens, blob, optimizer)
    assert abs(value - float(by_layer.detach())) < 1e-5
