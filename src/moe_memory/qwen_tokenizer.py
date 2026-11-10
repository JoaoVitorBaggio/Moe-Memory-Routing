"""BPE do Qwen2/Qwen3 a partir dos arquivos locais.

O vocabulário do Qwen tem 151936 ids. O GPT-2 do ``configs/default.yaml``
tem 50257 e não serve. Este loader usa o ``tiktoken`` que já está na
máquina e o ``tokenizer.json`` do diretório do modelo. Não baixa nada
e não precisa de ``transformers``, ``tokenizers`` nem ``peft``.

O id do token base é a ordem do BPE. Os ``added_tokens`` do
``tokenizer.json`` do Qwen não entram no ``model.vocab``: ficam com o
id que o arquivo já traz, depois do BPE. Id repetido, id de added
token em cima de um token do BPE, ou merge fora de ordem recusa o
arquivo. O loader não remapeia e não corta token.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path

# O mesmo corte do Qwen2Tokenizer. ``tiktoken`` aceita ``\\p{L}``.
QWEN_PAT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    extra = 0
    for byte in range(256):
        if byte not in bs:
            bs.append(byte)
            cs.append(256 + extra)
            extra += 1
    return dict(zip(bs, [chr(code) for code in cs], strict=True))


@dataclass(frozen=True)
class QwenTokenizer:
    """``vocab_size`` é ``max(id) + 1``. O config do Qwen pode ser maior."""

    vocab_size: int
    eos_id: int
    encoding: object
    nfc: bool = False

    def encode(self, text: str) -> list[int]:
        if self.nfc:
            text = unicodedata.normalize("NFC", text)
        # Sem allowed_special o tiktoken parte o especial em bytes.
        return list(self.encoding.encode(text, allowed_special="all"))


def load_qwen_tokenizer(model_dir: str | Path) -> QwenTokenizer:
    directory = Path(model_dir)
    blob_path = directory / "tokenizer.json"
    if not blob_path.is_file():
        raise FileNotFoundError(f"não achei {blob_path}. Não vou baixar o tokenizador.")
    blob = json.loads(blob_path.read_text(encoding="utf-8"))
    vocab, merges, specials, nfc = _pieces(blob)
    defined = _audit(vocab, merges, specials)
    mergeable, special_ids = _ranks(vocab, specials)
    import tiktoken

    encoding = tiktoken.Encoding(
        name="qwen-local",
        pat_str=QWEN_PAT,
        mergeable_ranks=mergeable,
        special_tokens=special_ids,
    )
    eos = _eos_id(directory, special_ids)
    if encoding.n_vocab != defined:
        raise ValueError(
            f"tiktoken montou {encoding.n_vocab} ids e o arquivo fecha em {defined}"
        )
    return QwenTokenizer(vocab_size=defined, eos_id=eos, encoding=encoding, nfc=nfc)


def _pieces(blob: dict) -> tuple[dict[str, int], list[tuple[str, str]], dict[str, int], bool]:
    try:
        model = blob["model"]
        vocab = dict(model["vocab"])
        raw_merges = model["merges"]
    except (KeyError, TypeError) as exc:
        raise ValueError("tokenizer.json não tem model.vocab e model.merges") from exc
    merges: list[tuple[str, str]] = []
    for item in raw_merges:
        if isinstance(item, str):
            if not item or item.startswith("#"):
                continue
            left, right = item.split(" ", 1)
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            left, right = str(item[0]), str(item[1])
        else:
            raise ValueError("merge do tokenizer.json não é 'a b' nem um par")
        merges.append((left, right))
    specials: dict[str, int] = {}
    seen_ids: dict[int, str] = {}
    for token in blob.get("added_tokens", []):
        content = str(token["content"])
        token_id = int(token["id"])
        if content == "":
            raise ValueError("added token vazio")
        if content in specials:
            raise ValueError(f"added token {content!r} repetido")
        if token_id in seen_ids:
            raise ValueError(f"id {token_id} repetido em {content!r} e {seen_ids[token_id]!r}")
        specials[content] = token_id
        seen_ids[token_id] = content
    normalizer = blob.get("normalizer") or {}
    nfc = isinstance(normalizer, dict) and normalizer.get("type") == "NFC"
    return vocab, merges, specials, nfc


def _audit(vocab: dict[str, int], merges: list[tuple[str, str]], specials: dict[str, int]) -> int:
    """Devolve ``max(id) + 1``. Added token fora do ``model.vocab`` é o formato do Qwen."""

    ids = list(vocab.values())
    if len(set(ids)) != len(ids):
        raise ValueError("vocab do Qwen tem ids repetidos")
    if set(ids) != set(range(len(vocab))):
        missing = next(item for item in range(len(vocab)) if item not in set(ids))
        raise ValueError(f"buraco no BPE, o primeiro id ausente é {missing}")
    vocab_ids = set(ids)
    for content, token_id in specials.items():
        if content in vocab and vocab[content] != token_id:
            raise ValueError(
                f"added token {content!r} tem id {token_id} e o vocab tem {vocab[content]}. Não remapeio."
            )
        if content not in vocab and token_id in vocab_ids:
            owner = next(token for token, value in vocab.items() if value == token_id)
            raise ValueError(f"added token {content!r} usa o id {token_id} do BPE {owner!r}")
        if token_id < len(vocab) and vocab.get(content) != token_id:
            raise ValueError(f"especial {content!r} caiu no meio do BPE, id {token_id}")
    for index, (left, right) in enumerate(merges):
        merged = left + right
        got = vocab.get(merged)
        expect = 256 + index
        if got != expect:
            raise ValueError(
                f"merge {index} produziu id {got}, esperado {expect}. "
                "O BPE não está na ordem dos ids; recuso para não trocar o vocabulário."
            )
        if vocab.get(left, -1) >= expect or vocab.get(right, -1) >= expect:
            raise ValueError(f"merge {index} não respeita a ordem do BPE")
    defined = set(ids) | set(specials.values())
    if not defined:
        raise ValueError("vocab vazio")
    limit = max(defined) + 1
    hole = next((item for item in range(limit) if item not in defined), None)
    if hole is not None:
        raise ValueError(f"buraco nos ids, o primeiro é {hole}")
    return limit


def _ranks(vocab: dict[str, int], specials: dict[str, int]) -> tuple[dict[bytes, int], dict[str, int]]:
    unicode_of = bytes_to_unicode()
    byte_of = {char: byte for byte, char in unicode_of.items()}
    special_set = set(specials)
    mergeable: dict[bytes, int] = {}
    for token, token_id in vocab.items():
        if token in special_set:
            continue
        try:
            raw = bytes(byte_of[char] for char in token)
        except KeyError as exc:
            raise ValueError(f"token {token!r} não é byte-level nem especial") from exc
        if raw in mergeable:
            raise ValueError("dois tokens do vocab caem nos mesmos bytes")
        mergeable[raw] = int(token_id)
    return mergeable, dict(specials)


def _eos_id(directory: Path, specials: dict[str, int]) -> int:
    config_path = directory / "tokenizer_config.json"
    content = "<|endoftext|>"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        eos = config.get("eos_token", content)
        if isinstance(eos, dict):
            content = str(eos.get("content", content))
        elif isinstance(eos, str) and eos:
            content = eos
    if content not in specials:
        raise ValueError(f"eos {content!r} não está nos added tokens")
    return int(specials[content])
