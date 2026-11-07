"""Traço em que a LSTM é o roteador. Não é aluno do gate do OLMoE.

O script lê um trecho curto do WikiText-2 pelo leitor do repositório
e grava, por token, os experts que a LSTM escolheu. Não grava
``teacher_logits``.

O peso do OLMoE, se entrar, é um fragmento já no disco
(``local_files_only``). O fp16 inteiro não cabe em 5.5 GB, então o
modelo cheio não é carregado nem executado. Sem cache, o erro é claro
e o modelo não roda. Os tensores lidos ficam congelados: sem otimizador
e sem backward. A comparação com o SiDA não entra aqui.
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
ROUTER_TRACE_FORMAT = "lstm-router-choices-v1"
DEFAULT_FRAGMENT_TOKENS = 2048
MAX_FRAGMENT_BYTES = 8 * 1024 * 1024
_WEIGHT_SUFFIXES = (".safetensors", ".pt", ".pth", ".bin")

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
    """Recusa o modelo inteiro. O fp16 não cabe em 5.5 GB e não há download."""

    del config, model_dir
    raise RuntimeError(
        "o OLMoE em fp16 não cabe em 5.5 GB. "
        "Este passo não carrega o modelo inteiro e não baixa. "
        "Use load_frozen_fragment num snapshot local."
    )


def _hub_dir(cache_dir: Path, model_id: str) -> Path:
    return cache_dir / ("models--" + model_id.replace("/", "--"))


def _weight_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix in _WEIGHT_SUFFIXES
    ]


def locate_snapshot(spec: OlmoeLoadSpec) -> Path | None:
    """Acha o snapshot no disco. Não baixa e não abre o modelo."""

    if spec.model_dir is not None:
        return spec.model_dir if spec.model_dir.is_dir() else None
    if spec.cache_dir is None:
        return None
    root = _hub_dir(spec.cache_dir, spec.model_id)
    snapshots = root / "snapshots"
    if snapshots.is_dir():
        for candidate in sorted(path for path in snapshots.iterdir() if path.is_dir()):
            if (candidate / "config.json").is_file():
                return candidate
    if (root / "config.json").is_file():
        return root
    return None


def require_local_olmoe(spec: OlmoeLoadSpec) -> Path:
    """Devolve o snapshot local. Se faltar, erro claro e o modelo não roda."""

    if not spec.local_files_only:
        raise RuntimeError("a leitura do OLMoE fica em local_files_only")
    where = spec.model_dir if spec.model_dir is not None else spec.cache_dir
    snapshot = locate_snapshot(spec)
    if snapshot is None or not (snapshot / "config.json").is_file() or not _weight_files(snapshot):
        raise FileNotFoundError(
            f"não achei {spec.model_id} em {where}. "
            "O traço só lê cache local (local_files_only) e não baixa. "
            "O modelo não foi executado."
        )
    return snapshot


def _flatten_tensors(blob: object, prefix: str = "") -> dict[str, torch.Tensor]:
    if torch.is_tensor(blob):
        return {prefix or "weight": blob}
    if isinstance(blob, dict):
        found: dict[str, torch.Tensor] = {}
        for key, value in blob.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            found.update(_flatten_tensors(value, name))
        return found
    return {}


def _freeze(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    frozen: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        out = tensor.detach().to(device="cpu")
        out.requires_grad_(False)
        frozen[name] = out
    return frozen


def load_frozen_fragment(
    snapshot: Path,
    *,
    max_bytes: int = MAX_FRAGMENT_BYTES,
) -> dict[str, torch.Tensor]:
    """Lê só um pedaço pequeno do peso. Não instancia o OLMoE e não treina.

    Arquivo maior que ``max_bytes`` fica de fora. Se só houver o fp16
    inteiro, o erro diz que o modelo não foi executado.
    """

    if max_bytes < 1:
        raise ValueError("max_bytes precisa ser >= 1")
    if not (snapshot / "config.json").is_file():
        raise FileNotFoundError(
            f"{snapshot} não tem config.json. Não há download. O modelo não foi executado."
        )
    loaded: dict[str, torch.Tensor] = {}
    used = 0
    saw_weight = False
    for path in _weight_files(snapshot):
        saw_weight = True
        size = path.stat().st_size
        if path.suffix not in {".pt", ".pth", ".bin"}:
            continue
        if size > max_bytes or size > max_bytes - used:
            continue
        blob = torch.load(path, map_location="cpu", weights_only=True)
        tensors = _freeze(_flatten_tensors(blob))
        nbytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
        if nbytes > max_bytes - used:
            continue
        loaded.update(tensors)
        used += nbytes
        if used >= max_bytes:
            break
    if loaded:
        return loaded
    if not saw_weight:
        raise FileNotFoundError(
            f"{snapshot} não tem arquivo de peso. Não há download. O modelo não foi executado."
        )
    raise FileNotFoundError(
        "não há fragmento pequeno de OLMoE nesse cache. "
        "O fp16 inteiro não cabe em 5.5 GB; este passo não carrega o modelo "
        "inteiro e não baixa. O modelo não foi executado."
    )


def token_ids_from_documents(documents: Iterator[str] | list[str], encoding) -> torch.Tensor:
    """Ids na ordem do leitor, um por token, com o fim de documento."""

    ids = list(iter_token_ids(documents, encoding))
    if not ids:
        raise ValueError("o leitor não devolveu token")
    return torch.tensor(ids, dtype=torch.long)


def take_token_ids(tokens: Iterator[int], max_tokens: int) -> torch.Tensor:
    """Corta o leitor no teto. Não pede o resto do corpus."""

    if max_tokens < 1:
        raise ValueError("max_tokens precisa ser >= 1")
    ids: list[int] = []
    for token in tokens:
        ids.append(int(token))
        if len(ids) >= max_tokens:
            break
    if not ids:
        raise ValueError("o leitor não devolveu token")
    return torch.tensor(ids, dtype=torch.long)


def wikitext_token_ids(config, split: str, *, max_tokens: int | None = None) -> torch.Tensor:
    """WikiText-2 pelo leitor do repositório. Não retokeniza com o OLMoE.

    ``max_tokens`` limita o trecho. ``None`` lê o split inteiro.
    """

    documents = iter_documents(config, split, name="WikiText-2", streaming=False)
    encoding = load_gpt2_encoding()
    if max_tokens is None:
        return token_ids_from_documents(documents, encoding)
    return take_token_ids(iter_token_ids(documents, encoding), max_tokens)


def lstm_expert_choices(predictor: nn.Module, token_ids: torch.Tensor) -> torch.Tensor:
    """A LSTM escolhe os experts. Sem gradiente e sem roteador professor.

    ``token_ids`` é ``[n_tokens]``. Sai ``[n_tokens, n_layers, k]`` long.
    """

    if token_ids.ndim != 1 or token_ids.numel() == 0:
        raise ValueError("token_ids precisa ser [n_tokens] não vazio")
    predictor.eval()
    with torch.no_grad():
        logits = predictor(token_ids.view(1, -1))
        index = torch.topk(logits, k=predictor.config.k, dim=-1).indices
    choices = index[0].detach().cpu().long()
    if choices.shape[0] != token_ids.shape[0]:
        raise RuntimeError("a escolha não ficou alinhada token a token")
    return choices


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


def save_router_choices(
    path: str | Path,
    token_ids: torch.Tensor,
    expert_index: torch.Tensor,
    *,
    split: str,
    fragment_tokens: int,
) -> None:
    """Grava a escolha da LSTM. Não grava logits de gate."""

    if token_ids.ndim != 1 or token_ids.dtype != torch.long:
        raise ValueError("token_ids precisa ser long [n_tokens]")
    if expert_index.ndim != 3 or expert_index.shape[0] != token_ids.shape[0]:
        raise ValueError(
            "expert_index precisa ser [n_tokens, n_layers, k] alinhado a token_ids; "
            f"shapes {tuple(token_ids.shape)} e {tuple(expert_index.shape)}"
        )
    if fragment_tokens < 1:
        raise ValueError("fragment_tokens precisa ser >= 1")
    payload = {
        "format": ROUTER_TRACE_FORMAT,
        "model_id": OLMOE_MODEL_ID,
        "router": "lstm",
        "split": split,
        "fragment_tokens": int(fragment_tokens),
        "token_ids": token_ids.detach().cpu().long(),
        "expert_index": expert_index.detach().cpu().long(),
        "stored_layout": "[n_tokens, n_layers, k]",
        "layout_note": (
            "A LSTM é o roteador e escolhe os experts na hora. "
            "expert_index[i] alinha com token_ids[i]. "
            "Não há teacher_logits."
        ),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)


def load_router_choices(path: str | Path) -> dict:
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or blob.get("format") != ROUTER_TRACE_FORMAT:
        raise ValueError(f"{path} não é um traço {ROUTER_TRACE_FORMAT}")
    if "teacher_logits" in blob:
        raise ValueError(f"{path} ainda traz teacher_logits; o traço é a escolha da LSTM")
    return blob


def load_teacher_trace(path: str | Path) -> dict:
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or blob.get("format") != TEACHER_TRACE_FORMAT:
        raise ValueError(f"{path} não é um traço {TEACHER_TRACE_FORMAT}")
    return blob
