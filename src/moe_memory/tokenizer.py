"""Tokenizer GPT-2 via tiktoken (o mesmo BPE do ``gpt2`` da HuggingFace).

O smoke sintético não tokeniza texto. TinyStories e WikiText-2 entram
por ``moe_memory.corpus``, que usa este encoding e o ``eot_token`` como
fronteira de documento.
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
