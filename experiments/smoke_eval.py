#!/usr/bin/env python3
"""Um forward de avaliação com a soma dos top-2 no custo.

A perplexidade impressa é ``exp(CE)`` de peso aleatório. Não é a
perplexidade do hold-out. O número de valid/test está em
``experiments/run_holdout_eval.py``.

Uso::

    python3 experiments/smoke_eval.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from moe_memory.config import load_config
from moe_memory.smoke import run_eval_pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Um forward sintético em modo de inferência.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = load_config(args.config)
    result = run_eval_pass(config, lambda_=args.lambda_, device=torch.device(args.device))
    slots = [
        config.memory.scaled_hot_slots(config.model.E, multiplier) for multiplier in config.eval.B_mult
    ]
    print("Forward de fumaça (init aleatória — não é resultado experimental)")
    print(f"parâmetros: {result['parameters']:,}")
    print(f"agregação do custo={result['aggregation']}  λ={result['lambda']}")
    print(
        f"L_task={result['task']:.4f}  C_mem={result['c_mem']:.4f}  "
        f"L_balanço={result['balance']:.4f}  L={result['total']:.4f}"
    )
    print(
        f"exp(L_task) em init aleatória={result['perplexity_random_init']:.1f}  "
        "(não é perplexidade de relatório)"
    )
    print(
        "dose–resposta ainda não rodou; vagas quentes previstas para "
        + ", ".join(f"{mult}×B→{slot}" for mult, slot in zip(config.eval.B_mult, slots))
    )


if __name__ == "__main__":
    main()
