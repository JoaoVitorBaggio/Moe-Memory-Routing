"""Traço do gate com um router falso. Sem rede, sem corpus e sem OLMoE."""

from pathlib import Path

import torch
from torch import nn

from moe_memory.config import load_config
from moe_memory.corpus import documents_from_texts, resolve_cache_dir
from moe_memory.lstm_predictor import ExpertLSTMPredictor, LstmPredictorConfig
from moe_memory.olmoe_teacher import (
    DEFAULT_FRAGMENT_TOKENS,
    OLMOE_MODEL_ID,
    OLMOE_N_EXPERTS,
    OLMOE_N_LAYERS,
    TEACHER_TRACE_FORMAT,
    OlmoeLoadSpec,
    collect_router_logits,
    load_frozen_fragment,
    load_olmoe,
    load_router_choices,
    load_teacher_trace,
    lstm_expert_choices,
    olmoe_load_spec,
    require_local_olmoe,
    require_olmoe_gate_shape,
    save_router_choices,
    save_teacher_trace,
    take_token_ids,
    token_ids_from_documents,
    trace_router_logits,
)


class _Encoding:
    eot_token = 9

    def encode(self, text, disallowed_special=()):
        return [ord(char) % 5 + 1 for char in text]


class _Gate(nn.Module):
    def __init__(self, d_model: int, n_experts: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.gate(hidden).mean(dim=-1, keepdim=True)


class _Block(nn.Module):
    def __init__(self, d_model: int, n_experts: int) -> None:
        super().__init__()
        self.mlp = _Gate(d_model, n_experts)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden)


class _FakeOlmoe(nn.Module):
    def __init__(self, vocab: int, d_model: int, n_layers: int, n_experts: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList(_Block(d_model, n_experts) for _ in range(n_layers))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


def _replay(model: _FakeOlmoe, input_ids: torch.Tensor) -> torch.Tensor:
    hidden = model.embed(input_ids)
    pieces = []
    for layer in model.layers:
        pieces.append(layer.mlp.gate(hidden))
        hidden = layer(hidden)
    return torch.stack(pieces, dim=2)


def test_fake_gate_logits_align_token_by_token_and_keep_every_expert():
    model = _FakeOlmoe(vocab=20, d_model=4, n_layers=3, n_experts=5)
    token_ids = torch.tensor([1, 4, 2, 7, 3])
    traced = trace_router_logits(model, token_ids, seq_len=2)
    replayed = []
    with torch.no_grad():
        for start in range(0, token_ids.numel(), 2):
            replayed.append(_replay(model, token_ids[start : start + 2].view(1, -1))[0])
    assert traced.shape == (5, 3, 5)
    assert torch.allclose(traced, torch.cat(replayed, dim=0))
    assert traced.shape[-1] == 5
    one = collect_router_logits(model, token_ids[:2].view(1, -1))
    assert one.shape == (1, 2, 3, 5)


def test_saved_trace_records_the_per_layer_layout(tmp_path):
    token_ids = torch.tensor([2, 0, 1])
    teacher_logits = torch.arange(3 * 2 * 4, dtype=torch.float32).view(3, 2, 4)
    path = tmp_path / "traco.pt"
    save_teacher_trace(path, token_ids, teacher_logits, split="train", top_k=2)
    blob = load_teacher_trace(path)
    assert blob["format"] == TEACHER_TRACE_FORMAT
    assert blob["model_id"] == OLMOE_MODEL_ID
    assert blob["gate"] == "mlp.gate"
    assert blob["top_k"] == 2
    assert blob["stored_layout"] == "[n_tokens, n_layers, n_experts]"
    assert blob["offline_train_step_layout"] == "[B, T, n_layers, n_experts]"
    assert "um vetor" in blob["layout_note"]
    assert "por camada" in blob["layout_note"]
    assert torch.equal(blob["token_ids"], token_ids)
    assert torch.equal(blob["teacher_logits"], teacher_logits)
    assert blob["teacher_logits"].shape[-1] != blob["top_k"]


def test_olmoe_shape_check_is_sixteen_by_sixty_four():
    good = torch.zeros(2, OLMOE_N_LAYERS, OLMOE_N_EXPERTS)
    require_olmoe_gate_shape(good)
    try:
        require_olmoe_gate_shape(torch.zeros(2, 2, 4))
    except ValueError as exc:
        assert "16" in str(exc) and "64" in str(exc)
    else:
        raise AssertionError("shape que não é o gate do OLMoE deveria falhar")


def test_reader_ids_follow_wikitext_documents_not_a_download():
    documents = documents_from_texts("WikiText-2", ["aa", "", "b"])
    ids = token_ids_from_documents(documents, _Encoding())
    assert ids.tolist() == [3, 3, 9, 4, 9]


def test_load_spec_uses_the_corpus_cache_and_refuses_a_download():
    config = load_config()
    spec = olmoe_load_spec(config, model_dir=None)
    assert spec.model_id == OLMOE_MODEL_ID
    assert spec.cache_dir == resolve_cache_dir(config)
    assert spec.model_dir is None
    assert spec.local_files_only is True
    local = olmoe_load_spec(config, model_dir="/tmp/olmoe-snapshot")
    assert local.model_dir == Path("/tmp/olmoe-snapshot")
    assert local.cache_dir is None
    assert local.local_files_only is True


def test_token_cap_stops_before_the_rest_of_the_stream():
    assert DEFAULT_FRAGMENT_TOKENS == 2048

    def source():
        for item in range(10):
            if item >= 4:
                raise AssertionError("o leitor passou do teto")
            yield item

    assert take_token_ids(source(), 3).tolist() == [0, 1, 2]


def test_missing_cache_stops_before_any_model(tmp_path):
    spec = OlmoeLoadSpec(
        model_id=OLMOE_MODEL_ID,
        cache_dir=tmp_path,
        model_dir=None,
        local_files_only=True,
    )
    ran = False
    try:
        require_local_olmoe(spec)
        ran = True
    except FileNotFoundError as exc:
        text = str(exc)
        assert "não baixa" in text
        assert "não foi executado" in text
    assert ran is False
    try:
        load_olmoe(None)
    except RuntimeError as exc:
        assert "5.5 GB" in str(exc)
    else:
        raise AssertionError("o modelo inteiro não devia carregar")


def test_frozen_fragment_is_small_and_a_huge_file_is_refused(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    torch.save({"expert.weight": torch.ones(2, 2)}, snapshot / "fragment.pt")
    fragment = load_frozen_fragment(snapshot)
    assert set(fragment) == {"expert.weight"}
    assert fragment["expert.weight"].requires_grad is False
    assert torch.equal(fragment["expert.weight"], torch.ones(2, 2))

    huge = tmp_path / "huge"
    huge.mkdir()
    (huge / "config.json").write_text("{}", encoding="utf-8")
    torch.save({"expert.weight": torch.zeros(64)}, huge / "model.pt")
    try:
        load_frozen_fragment(huge, max_bytes=32)
    except FileNotFoundError as exc:
        assert "5.5 GB" in str(exc)
        assert "não foi executado" in str(exc)
    else:
        raise AssertionError("arquivo grande devia ser recusado")


def test_lstm_records_expert_choices_not_teacher_logits(tmp_path):
    config = LstmPredictorConfig(
        vocab_size=16,
        n_experts=4,
        k=2,
        n_model_layers=3,
        embedding_dim=8,
        hidden_size=8,
        n_lstm_layers=2,
        fallback_every=2,
        router_accuracy_floor=0.9,
        hidden_if_router_accuracy_drops=16,
        ssd_lookahead_layers=4,
    )
    config.validate()
    predictor = ExpertLSTMPredictor(config)
    tokens = torch.tensor([1, 2, 3, 4])
    choices = lstm_expert_choices(predictor, tokens)
    assert choices.shape == (4, 3, 2)
    assert choices.dtype == torch.long
    assert int(choices.max()) < 4
    assert not choices.requires_grad
    path = tmp_path / "choices.pt"
    save_router_choices(path, tokens, choices, split="train", fragment_tokens=DEFAULT_FRAGMENT_TOKENS)
    blob = load_router_choices(path)
    assert blob["router"] == "lstm"
    assert blob["fragment_tokens"] == 2048
    assert "teacher_logits" not in blob
    assert torch.equal(blob["token_ids"], tokens)
    assert torch.equal(blob["expert_index"], choices)
