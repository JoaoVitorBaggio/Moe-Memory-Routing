#!/usr/bin/env python3
"""Grava os logits do router do OLMoE no WikiText-2. Não treina o preditor.

O texto passa pelo leitor que o repositório já usa (WikiText-2, BPE do
GPT-2). O peso é o ``allenai/OLMoE-1B-7B-0924`` já em
``corpus.cache_dir``, ou em ``--model-dir``. Não há download.

O arquivo guarda, token a token, ``token_ids`` e ``teacher_logits``.
O gate é a linear ``mlp.gate``: 16 camadas, 64 experts, top-8. O traço
guarda o logit de cada camada, ``[n_tokens, 16, 64]``, não o índice
top-k. ``offline_train_step`` consome esse shape: a KL compara cada
camada, sem juntar as 16 num vetor só.

Uso, com o peso já no disco::

    python3 experiments/dump_olmoe_teacher.py --output outputs/olmoe-teacher/train.pt
    python3 experiments/dump_olmoe_teacher.py --model-dir /caminho/do/snapshot --output traco.pt

Não rode este script para a comparação com o SiDA. Ela continua suspensa.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moe_memory.config import load_config
from moe_memory.olmoe_teacher import (
    OLMOE_MODEL_ID,
    OLMOE_TOP_K,
    load_olmoe,
    olmoe_load_spec,
    require_olmoe_gate_shape,
    save_teacher_trace,
    trace_router_logits,
    wikitext_token_ids,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Traço dos logits do router do OLMoE.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None, help="Snapshot local. Sem isto, usa corpus.cache_dir.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--seq-len", type=int, default=None, help="Bloco do forward. Padrão: model.max_seq_len.")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if config.corpus.report != "WikiText-2":
        raise SystemExit(f"corpus.report={config.corpus.report!r}; este traço é o WikiText-2")
    seq_len = config.model.max_seq_len if args.seq_len is None else args.seq_len
    if seq_len < 1:
        raise SystemExit("seq_len precisa ser >= 1")
    if args.max_tokens is not None and args.max_tokens < 1:
        raise SystemExit("max_tokens precisa ser >= 1")
    spec = olmoe_load_spec(config, args.model_dir)
    token_ids = wikitext_token_ids(config, args.split)
    if args.max_tokens is not None:
        token_ids = token_ids[: args.max_tokens]
    model = load_olmoe(config, args.model_dir)
    teacher_logits = trace_router_logits(model, token_ids, seq_len=seq_len)
    require_olmoe_gate_shape(teacher_logits)
    save_teacher_trace(
        args.output,
        token_ids,
        teacher_logits,
        split=args.split,
        top_k=OLMOE_TOP_K,
    )
    where = spec.model_dir if spec.model_dir is not None else spec.cache_dir
    print(
        f"{OLMOE_MODEL_ID}  origem={where}  local_files_only={spec.local_files_only}\n"
        f"split={args.split}  tokens={token_ids.shape[0]}  "
        f"teacher_logits={tuple(teacher_logits.shape)}  "
        f"layout=[n_tokens, n_layers, n_experts]\n"
        f"arquivo={args.output}"
    )


if __name__ == "__main__":
    main()
