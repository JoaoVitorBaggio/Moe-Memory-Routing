#!/usr/bin/env python3
"""Carga de fumaça do WikiText-2. Não treina e não mede perplexidade.

Confere os splits oficiais (train, valid, test), o BPE do GPT-2 e um
batch no formato da CE do próximo token. Número de token aqui é tamanho
de dado, não métrica do relatório.

Uso::

    python3 experiments/smoke_wikitext.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moe_memory.config import load_config
from moe_memory.corpus import iter_batches, iter_documents, iter_token_ids, source_label, splits_of
from moe_memory.tokenizer import load_gpt2_encoding


def main() -> None:
    parser = argparse.ArgumentParser(description="Carga de fumaça do WikiText-2 (sem treino).")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--seq-len", type=int, default=None, help="Padrão: smoke.seq_len.")
    parser.add_argument("--batch-size", type=int, default=None, help="Padrão: smoke.batch_size.")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.corpus.report != "WikiText-2":
        raise SystemExit(
            f"corpus.report={config.corpus.report!r}; este script é o WikiText-2 da entrega"
        )
    seq_len = config.smoke.seq_len if args.seq_len is None else args.seq_len
    batch_size = config.smoke.batch_size if args.batch_size is None else args.batch_size
    if seq_len < 2 or batch_size < 1:
        raise SystemExit("seq_len >= 2 e batch_size >= 1")
    encoding = load_gpt2_encoding()
    print("Carga de fumaça do WikiText-2 (não é treino e não é número de relatório)")
    print(f"fonte: {source_label(config, 'WikiText-2')}")
    print(f"vocab={encoding.n_vocab}  eot={encoding.eot_token}  seq={seq_len}  batch={batch_size}")
    try:
        for split in splits_of("WikiText-2"):
            documents = list(iter_documents(config, split, name="WikiText-2", streaming=False))
            ids = list(iter_token_ids(documents, encoding))
            batch = next(
                iter_batches(ids, seq_len=seq_len, batch_size=batch_size, max_batches=1),
                None,
            )
            if batch is None:
                raise SystemExit(f"{split} não fechou um bloco de seq_len={seq_len}")
            print(
                f"{split}: documentos={len(documents)}  tokens={len(ids)}  "
                f"batch={tuple(batch.shape)}  ids={int(batch.min())}..{int(batch.max())}"
            )
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(
            "Não consegui ler o WikiText-2. Confira a rede, corpus.cache_dir "
            f"e corpus.paths.WikiText-2. Detalhe: {exc}"
        ) from exc


if __name__ == "__main__":
    main()
