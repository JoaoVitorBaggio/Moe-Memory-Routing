"""Traço do roteador do OLMoE para o passo offline do preditor.

O leitor é o do repositório (``iter_documents`` / ``iter_token_ids``,
BPE do GPT-2 no WikiText-2). Os ids gravados são esses, na mesma ordem.
O OLMoE não usa esse tokenizer; a linha ``i`` do logit é o gate na
passagem do ``token_ids[i]``, não uma retokenização.

O gate do OLMoE é a linear ``mlp.gate``, uma por camada (16 camadas,
64 experts, top-8). Este módulo grava o logit completo de cada camada,
não o índice top-k. ``offline_train_step`` lê esse tensor
``[n_tokens, n_layers, n_experts]`` e a KL compara cada camada.

O peso sai do ``corpus.cache_dir`` que o loader já usa, ou de um
diretório local. ``local_files_only`` impede o download. Quem não tem o
peso em disco recebe erro. A comparação com o SiDA não entra aqui.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from moe_memory.corpus import documents_from_texts, iter_documents, iter_token_ids, resolve_cache_dir
from moe_memory.tokenizer import load_gpt2_encoding

OLMOE_MODEL_ID = "allenai/OLMoE-1B-7B-0924"
OLMOE_N_LAYERS = 16
OLMOE_N_EXPERTS = 64
OLMOE_TOP_K = 8
OLMOE_GATE = "mlp.gate"
TEACHER_TRACE_FORMAT = "olmoe-teacher-logits-v1"

LAYOUT_NOTE = (
    "teacher_logits tem shape [n_tokens, n_layers, n_experts]: "
    "um vetor de logits por camada, por token. Não é a média das camadas. "
    "offline_train_step aceita esse tensor, ou [B, T, n_layers, n_experts], "
    "e a KL compara cada camada no eixo dos experts. "
    "token_ids[i] alinha com teacher_logits[i]. "
    "O arquivo guarda os logits completos do router, não o índice top-k."
)


@dataclass(frozen=True)
class OlmoeLoadSpec:
    """De onde ler o peso. Não baixa e não instancia o modelo."""

    model_id: str
    cache_dir: Path | None
    model_dir: Path | None
    local_files_only: bool = True

    def __post_init__(self) -> None:
        if not self.local_files_only:
            raise ValueError("a carga do OLMoE fica em local_files_only")
        if self.model_dir is None and self.cache_dir is None:
            raise ValueError("falta cache_dir ou model_dir")


def olmoe_load_spec(config, model_dir: str | Path | None = None) -> OlmoeLoadSpec:
    """O cache é o ``corpus.cache_dir`` do loader. Um diretório explícito ganha dele."""

    if model_dir is not None:
        return OlmoeLoadSpec(
            model_id=OLMOE_MODEL_ID,
            cache_dir=None,
            model_dir=Path(model_dir),
            local_files_only=True,
        )
    return OlmoeLoadSpec(
        model_id=OLMOE_MODEL_ID,
        cache_dir=resolve_cache_dir(config),
        model_dir=None,
        local_files_only=True,
    )


def load_olmoe(config, model_dir: str | Path | None = None):
    """Carrega o peso já presente no disco. Não baixa."""

    spec = olmoe_load_spec(config, model_dir)
    try:
        from transformers import OlmoeForCausalLM
    except ImportError as exc:
        raise ImportError(
            "carregar o OLMoE precisa do pacote transformers, e do peso já em disco"
        ) from exc
    if spec.model_dir is not None:
        return OlmoeForCausalLM.from_pretrained(
            str(spec.model_dir),
            local_files_only=True,
        )
    return OlmoeForCausalLM.from_pretrained(
        spec.model_id,
        cache_dir=str(spec.cache_dir),
        local_files_only=True,
    )


def token_ids_from_documents(documents: Iterator[str] | list[str], encoding) -> torch.Tensor:
    """Ids na ordem do leitor, um por token, com o fim de documento."""

    ids = list(iter_token_ids(documents, encoding))
    if not ids:
        raise ValueError("o leitor não devolveu token")
    return torch.tensor(ids, dtype=torch.long)


def wikitext_token_ids(config, split: str) -> torch.Tensor:
    """WikiText-2 pelo leitor do repositório. Não retokeniza com o OLMoE."""

    documents = iter_documents(config, split, name="WikiText-2", streaming=False)
    return token_ids_from_documents(documents, load_gpt2_encoding())


def iter_gate_modules(model: nn.Module) -> list[nn.Module]:
    """``mlp.gate`` na ordem das camadas."""

    gates = [
        module
        for name, module in model.named_modules()
        if name == OLMOE_GATE or name.endswith("." + OLMOE_GATE)
    ]
    if not gates:
        raise ValueError(f"não achei {OLMOE_GATE} no modelo")
    return gates


def collect_router_logits(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Logits de cada ``mlp.gate`` num forward. Sai ``[B, T, n_layers, n_experts]``."""

    if input_ids.ndim != 2:
        raise ValueError(f"input_ids tem shape {tuple(input_ids.shape)}; esperado [B, T]")
    gates = iter_gate_modules(model)
    captured: list[torch.Tensor] = []
    handles = []

    def _hook(_module, _inputs, output) -> None:
        captured.append(output)

    for gate in gates:
        handles.append(gate.register_forward_hook(_hook))
    try:
        model(input_ids)
    finally:
        for handle in handles:
            handle.remove()
    if len(captured) != len(gates):
        raise RuntimeError(
            f"{OLMOE_GATE} disparou {len(captured)} vezes; há {len(gates)} gates"
        )
    stacked = torch.stack([logits.detach() for logits in captured], dim=2)
    batch, seq = input_ids.shape
    if stacked.shape[:2] != (batch, seq):
        raise RuntimeError(
            f"logit do gate {tuple(stacked.shape)} não alinha com tokens {(batch, seq)}"
        )
    return stacked


def trace_router_logits(
    model: nn.Module,
    token_ids: torch.Tensor,
    *,
    seq_len: int,
) -> torch.Tensor:
    """Percorre a sequência em blocos e concatena. O resto também entra.

    Sai ``[n_tokens, n_layers, n_experts]``, uma linha por token de
    ``token_ids``, na mesma ordem.
    """

    if token_ids.ndim != 1:
        raise ValueError(f"token_ids tem shape {tuple(token_ids.shape)}; esperado [n_tokens]")
    if token_ids.numel() == 0:
        raise ValueError("sem tokens")
    if seq_len < 1:
        raise ValueError("seq_len precisa ser >= 1")
    pieces: list[torch.Tensor] = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, int(token_ids.numel()), seq_len):
                chunk = token_ids[start : start + seq_len].view(1, -1)
                pieces.append(collect_router_logits(model, chunk)[0])
    finally:
        model.train(was_training)
    traced = torch.cat(pieces, dim=0)
    if traced.shape[0] != token_ids.shape[0]:
        raise RuntimeError("o traço não ficou alinhado token a token")
    return traced


def require_olmoe_gate_shape(teacher_logits: torch.Tensor) -> None:
    """O OLMoE desta leva é 16 × 64. Outro shape não é esse gate."""

    if teacher_logits.ndim != 3 or teacher_logits.shape[1:] != (OLMOE_N_LAYERS, OLMOE_N_EXPERTS):
        raise ValueError(
            f"{OLMOE_GATE} do {OLMOE_MODEL_ID} é "
            f"{OLMOE_N_LAYERS} camadas × {OLMOE_N_EXPERTS} experts; "
            f"recebido {tuple(teacher_logits.shape)}"
        )


def save_teacher_trace(
    path: str | Path,
    token_ids: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    split: str,
    top_k: int = OLMOE_TOP_K,
) -> None:
    """Grava ids e logits. ``top_k`` fica no arquivo e não corta o vetor."""

    if token_ids.ndim != 1 or token_ids.dtype != torch.long:
        raise ValueError("token_ids precisa ser long [n_tokens]")
    if teacher_logits.ndim != 3 or teacher_logits.shape[0] != token_ids.shape[0]:
        raise ValueError(
            "teacher_logits precisa ser [n_tokens, n_layers, n_experts] "
            f"alinhado a token_ids; shapes {tuple(token_ids.shape)} e {tuple(teacher_logits.shape)}"
        )
    if top_k < 1 or top_k > teacher_logits.shape[-1]:
        raise ValueError(f"top_k={top_k} não cabe em {teacher_logits.shape[-1]} experts")
    payload = {
        "format": TEACHER_TRACE_FORMAT,
        "model_id": OLMOE_MODEL_ID,
        "gate": OLMOE_GATE,
        "top_k": int(top_k),
        "split": split,
        "token_ids": token_ids.detach().cpu().long(),
        "teacher_logits": teacher_logits.detach().cpu().float(),
        "stored_layout": "[n_tokens, n_layers, n_experts]",
        "offline_train_step_layout": "[B, T, n_layers, n_experts]",
        "layout_note": LAYOUT_NOTE,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)


def load_teacher_trace(path: str | Path) -> dict:
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or blob.get("format") != TEACHER_TRACE_FORMAT:
        raise ValueError(f"{path} não é um traço {TEACHER_TRACE_FORMAT}")
    return blob
