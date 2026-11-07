#!/usr/bin/env python3
"""Grava as escolhas da LSTM no WikiText-2. A LSTM é o roteador.

Não grava logits do gate do OLMoE e não treina. O texto é um trecho
curto do leitor que o repositório já usa. O padrão são 2048 tokens, não
o corpus inteiro.

O peso, se existir, é um fragmento já em ``corpus.cache_dir`` ou em
``--model-dir`` (``local_files_only``). O OLMoE em fp16 não cabe em
5.5 GB: o script não carrega o modelo inteiro e não baixa. Sem o cache,
sai com erro e não roda o modelo. Os tensores lidos ficam congelados.

Uso::

    python3 experiments/dump_olmoe_teacher.py --output outputs/olmoe-teacher/choices.pt
    python3 experiments/dump_olmoe_teacher.py --model-dir /caminho/do/snapshot --output choices.pt

Não rode este script para a comparação com o SiDA. Ela continua suspensa.
Não há passo de treino aqui.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moe_memory.config import load_config
from moe_memory.lstm_predictor import ExpertLSTMPredictor, load_lstm_config
from moe_memory.olmoe_teacher import (
    DEFAULT_FRAGMENT_TOKENS,
    MAX_FRAGMENT_BYTES,
    OLMOE_MODEL_ID,
    load_frozen_fragment,
    lstm_expert_choices,
    olmoe_load_spec,
    require_local_olmoe,
    save_router_choices,
    wikitext_token_ids,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A LSTM escolhe os experts num trecho curto do WikiText-2."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Snapshot local. Sem isto, usa corpus.cache_dir. Não baixa.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_FRAGMENT_TOKENS,
        help=(
            f"Trecho do WikiText-2, em tokens do leitor. Padrão: {DEFAULT_FRAGMENT_TOKENS}. "
            "Não é o corpus inteiro. O OLMoE em fp16 não cabe em 5.5 GB; "
            "o script só abre um fragmento local do peso e a LSTM escolhe os experts."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_tokens < 1:
        raise SystemExit("max_tokens precisa ser >= 1")
    config = load_config(args.config)
    if config.corpus.report != "WikiText-2":
        raise SystemExit(f"corpus.report={config.corpus.report!r}; este traço é o WikiText-2")
    spec = olmoe_load_spec(config, args.model_dir)
    try:
        snapshot = require_local_olmoe(spec)
        fragment = load_frozen_fragment(snapshot)
    except (FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    if any(tensor.requires_grad for tensor in fragment.values()):
        raise SystemExit("o fragmento do OLMoE precisa estar congelado")
    token_ids = wikitext_token_ids(config, args.split, max_tokens=args.max_tokens)
    predictor = ExpertLSTMPredictor(load_lstm_config())
    expert_index = lstm_expert_choices(predictor, token_ids)
    save_router_choices(
        args.output,
        token_ids,
        expert_index,
        split=args.split,
        fragment_tokens=args.max_tokens,
    )
    print(
        f"{OLMOE_MODEL_ID}  origem={snapshot}  local_files_only=True  "
        f"fragmento_bytes<={MAX_FRAGMENT_BYTES}\n"
        f"split={args.split}  teto={args.max_tokens}  tokens={token_ids.shape[0]}  "
        f"expert_index={tuple(expert_index.shape)}  layout=[n_tokens, n_layers, k]\n"
        f"arquivo={args.output}"
    )


if __name__ == "__main__":
    main()
