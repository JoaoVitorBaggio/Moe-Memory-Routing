"""Ids do tokenizador Qwen. O arquivo de 11 MB não entra no git.

A comparação com a lib ``tokenizers`` prova o id. Um encode seguido de
decode não prova: dois mapas errados podem devolver o mesmo texto.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from moe_memory.qwen_tokenizer import bytes_to_unicode, load_qwen_tokenizer

# Cópias da Fedora. O mesmo SHA256 do Qwen3-30B-A3B público, sem peso.
TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
TOKENIZER_CONFIG_SHA256 = "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"
CONFIG_SHA256 = "2850ddb3bf7aecad20b611e2d44f3077fc8193f4827c93beddd4c02ad63c2297"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _outside_vocab(directory: Path, *, added_id: int = 257, content: str = "<|endoftext|>") -> None:
    unicode_of = bytes_to_unicode()
    vocab = {char: index for index, char in enumerate(unicode_of.values())}
    left = unicode_of[ord("a")]
    right = unicode_of[ord("b")]
    vocab[left + right] = 256
    directory.mkdir(parents=True, exist_ok=True)
    blob = {
        "model": {"type": "BPE", "vocab": vocab, "merges": [f"{left} {right}"]},
        "added_tokens": [{"id": added_id, "content": content, "special": True}],
        "normalizer": {"type": "NFC"},
    }
    (directory / "tokenizer.json").write_text(json.dumps(blob), encoding="utf-8")
    (directory / "tokenizer_config.json").write_text(
        json.dumps({"eos_token": content}),
        encoding="utf-8",
    )


def test_added_token_outside_model_vocab_keeps_its_id(tmp_path: Path):
    _outside_vocab(tmp_path)
    tokenizer = load_qwen_tokenizer(tmp_path)
    assert tokenizer.eos_id == 257
    assert tokenizer.vocab_size == 258
    assert tokenizer.encode("<|endoftext|>") == [257]
    assert tokenizer.encode("ab") == [256]


def test_added_token_that_collides_with_bpe_still_fails(tmp_path: Path):
    _outside_vocab(tmp_path, added_id=10, content="<|x|>")
    with pytest.raises(ValueError, match="usa o id 10"):
        load_qwen_tokenizer(tmp_path)


def test_remapped_added_token_still_fails(tmp_path: Path):
    unicode_of = bytes_to_unicode()
    vocab = {char: index for index, char in enumerate(unicode_of.values())}
    left = unicode_of[ord("a")]
    right = unicode_of[ord("b")]
    vocab[left + right] = 256
    vocab["<|endoftext|>"] = 257
    tmp_path.mkdir(parents=True, exist_ok=True)
    blob = {
        "model": {"type": "BPE", "vocab": vocab, "merges": [f"{left} {right}"]},
        "added_tokens": [{"id": 300, "content": "<|endoftext|>", "special": True}],
    }
    (tmp_path / "tokenizer.json").write_text(json.dumps(blob), encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"eos_token": "<|endoftext|>"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Não remapeio"):
        load_qwen_tokenizer(tmp_path)


def test_swapped_merge_ids_still_fail(tmp_path: Path):
    unicode_of = bytes_to_unicode()
    vocab = {char: index for index, char in enumerate(unicode_of.values())}
    pairs = [(unicode_of[ord("a")], unicode_of[ord("b")]), (unicode_of[ord("c")], unicode_of[ord("d")])]
    vocab[pairs[0][0] + pairs[0][1]] = 257
    vocab[pairs[1][0] + pairs[1][1]] = 256
    tmp_path.mkdir(parents=True, exist_ok=True)
    blob = {
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": [f"{pairs[0][0]} {pairs[0][1]}", f"{pairs[1][0]} {pairs[1][1]}"],
        },
        "added_tokens": [],
    }
    (tmp_path / "tokenizer.json").write_text(json.dumps(blob), encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|endoftext|>"}), encoding="utf-8")
    with pytest.raises(ValueError, match="merge 0"):
        load_qwen_tokenizer(tmp_path)


def test_hole_between_bpe_and_added_token_still_fails(tmp_path: Path):
    _outside_vocab(tmp_path, added_id=259)
    with pytest.raises(ValueError, match="buraco nos ids"):
        load_qwen_tokenizer(tmp_path)


def test_check_tokenizer_does_not_need_the_store(tmp_path: Path):
    _outside_vocab(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/train_qwen_lm.py",
            "--check-tokenizer",
            "--model-dir",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["eos_id"] == 257
    assert payload["vocab_size"] == 258
    assert payload["note"].startswith("ids do BPE")
    assert "resident.bin" not in completed.stdout


def _real_tokenizer_dir() -> Path | None:
    candidates = []
    env = os.environ.get("QWEN_TOKENIZER_DIR")
    if env:
        candidates.append(Path(env))
    candidates.append(Path("/tmp/qwen3-tokenizer"))
    for path in candidates:
        file = path / "tokenizer.json"
        if file.is_file() and _sha(file) == TOKENIZER_SHA256:
            return path
    return None


def test_fedora_tokenizer_ids_match_tokenizers_library():
    tokenizers = pytest.importorskip("tokenizers")
    directory = _real_tokenizer_dir()
    if directory is None:
        pytest.skip("tokenizer real do Qwen3 não está neste host")
    assert _sha(directory / "tokenizer_config.json") == TOKENIZER_CONFIG_SHA256
    assert _sha(directory / "config.json") == CONFIG_SHA256
    ours = load_qwen_tokenizer(directory)
    reference = tokenizers.Tokenizer.from_file(str(directory / "tokenizer.json"))
    assert ours.vocab_size == reference.get_vocab_size(with_added_tokens=True) == 151669
    assert ours.eos_id == 151645
    probes = [
        "Hello",
        "Olá, o João não vai à praça.",
        "Na\u0303o",
        "Não",
        "a  b\n\nc",
        "ção\t ",
        "<|im_end|>",
        "texto <|endoftext|> fim",
        "こんにちは",
        " ",
        "\n\n",
        "",
        "O preço é R$ 3,50.",
        "  leading",
        "trailing  ",
        "linha\r\noutra",
        "<|im_start|>sistema",
        "emoji 😀 e acento áàâãé",
        "O\u0301",
    ]
    for text in probes:
        assert ours.encode(text) == reference.encode(text).ids
    assert ours.encode("Hello") == [9707]
    assert ours.encode("<|im_end|>") == [151645]
    assert ours.encode(" ") == [220]
    assert ours.encode("Não") == [50799]
    assert 151643 in ours.encode("texto <|endoftext|> fim")
