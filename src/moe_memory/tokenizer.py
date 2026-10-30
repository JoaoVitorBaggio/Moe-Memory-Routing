"""Tokenizer GPT-2 via tiktoken (o mesmo BPE do ``gpt2`` da HuggingFace).

O smoke não tokeniza texto: o batch é de ids aleatórios. Este módulo é
a porta para TinyStories / WikiText-2 quando o loader de corpus entrar.
"""

from __future__ import annotations

GPT2_VOCAB_SIZE = 50257


def load_gpt2_encoding():
    import tiktoken

    encoding = tiktoken.get_encoding("gpt2")
    if encoding.n_vocab != GPT2_VOCAB_SIZE:
        raise RuntimeError(
            f"tiktoken gpt2 reportou vocab {encoding.n_vocab}, esperado {GPT2_VOCAB_SIZE}"
        )
    return encoding


def encode(text: str) -> list[int]:
    return load_gpt2_encoding().encode(text)
