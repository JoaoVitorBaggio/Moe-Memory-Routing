"""BPE do Qwen2/Qwen3 a partir dos arquivos locais.

O vocabulário do Qwen tem 151936 ids. O GPT-2 do ``configs/default.yaml``
tem 50257 e não serve. Este loader usa o ``tiktoken`` que já está na
máquina e o ``tokenizer.json`` do diretório do modelo. Não baixa nada
e não precisa de ``transformers``, ``tokenizers`` nem ``peft``.

O id do token base é a ordem do BPE. Se o arquivo não seguir isso, o
loader recusa em vez de treinar com ids trocados.
"""

from __future__ import annotations

import json
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
    vocab_size: int
    eos_id: int
    encoding: object

    def encode(self, text: str) -> list[int]:
        return list(self.encoding.encode(text, disallowed_special=()))


def load_qwen_tokenizer(model_dir: str | Path) -> QwenTokenizer:
    directory = Path(model_dir)
    blob_path = directory / "tokenizer.json"
    if not blob_path.is_file():
        raise FileNotFoundError(f"não achei {blob_path}. Não vou baixar o tokenizador.")
    blob = json.loads(blob_path.read_text(encoding="utf-8"))
    vocab, merges, specials = _pieces(blob)
    _audit(vocab, merges, specials)
    mergeable, special_ids = _ranks(vocab, specials)
    import tiktoken

    encoding = tiktoken.Encoding(
        name="qwen-local",
        pat_str=QWEN_PAT,
        mergeable_ranks=mergeable,
        special_tokens=special_ids,
    )
    eos = _eos_id(directory, special_ids)
    if encoding.n_vocab != len(vocab):
        raise ValueError(
            f"tiktoken montou {encoding.n_vocab} ids e o vocab tem {len(vocab)}"
        )
    return QwenTokenizer(vocab_size=len(vocab), eos_id=eos, encoding=encoding)


def _pieces(blob: dict) -> tuple[dict[str, int], list[tuple[str, str]], dict[str, int]]:
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
    for token in blob.get("added_tokens", []):
        specials[str(token["content"])] = int(token["id"])
    return vocab, merges, specials


def _audit(vocab: dict[str, int], merges: list[tuple[str, str]], specials: dict[str, int]) -> None:
    ids = set(vocab.values())
    if len(ids) != len(vocab):
        raise ValueError("vocab do Qwen tem ids repetidos")
    if specials and set(specials) - set(vocab):
        raise ValueError("added token fora do vocab")
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
    for content, token_id in specials.items():
        if token_id < 256 + len(merges) and vocab.get(content) != token_id:
            raise ValueError(f"especial {content} caiu no meio do BPE")


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
