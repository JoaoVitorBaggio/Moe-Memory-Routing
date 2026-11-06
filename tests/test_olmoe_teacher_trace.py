"""Traço do gate com um router falso. Sem rede, sem corpus e sem OLMoE."""

from pathlib import Path

import torch
from torch import nn

from moe_memory.config import load_config
from moe_memory.corpus import documents_from_texts, resolve_cache_dir
from moe_memory.olmoe_teacher import (
    OLMOE_MODEL_ID,
    OLMOE_N_EXPERTS,
    OLMOE_N_LAYERS,
    TEACHER_TRACE_FORMAT,
    collect_router_logits,
    load_teacher_trace,
    olmoe_load_spec,
    require_olmoe_gate_shape,
    save_teacher_trace,
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
    assert blob["offline_train_step_layout"] == "[B, T, E]"
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
