#!/usr/bin/env python3
"""Um passo de treino num batch aleatório. Não é a corrida de WikiText-2.

Uso, a partir da raiz do repositório::

    python3 experiments/smoke_train.py
    python3 experiments/smoke_train.py --lambda 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from moe_memory.config import load_config
from moe_memory.smoke import run_train_step


def main() -> None:
    parser = argparse.ArgumentParser(description="Um passo sintético de treino (fp32, CPU ou CUDA).")
    parser.add_argument("--config", type=Path, default=None, help="YAML; padrão: configs/default.yaml")
    parser.add_argument(
        "--lambda",
        dest="lambda_",
        type=float,
        default=None,
        help="Um valor da grade. Padrão: smoke.lambda do YAML.",
    )
    parser.add_argument("--device", default="cpu", help="Padrão cpu. A 1060 entra na medição de parede, não neste smoke.")
    args = parser.parse_args()
    config = load_config(args.config)
    result = run_train_step(config, lambda_=args.lambda_, device=torch.device(args.device))
    print("Passo de fumaça (init aleatória — não é resultado experimental)")
    print(f"parâmetros: {result['parameters']:,}")
    print(f"λ={result['lambda']}  β={result['beta']}  agregação=weighted_mean")
    print(
        f"L_task={result['task']:.4f}  C_mem={result['c_mem']:.4f}  "
        f"L_balanço={result['balance']:.4f}  L={result['total']:.4f}"
    )
    print(
        f"acesso={result['access']:.4f}  troca={result['swap']:.4f}  "
        f"E[c] cheio (não entra em L)={result['access_expected']:.4f}"
    )
    print(f"entropia do roteador={result['entropy']:.4f}  |grad|={result['grad_norm']:.4f}")
    print(
        f"emulador: trocas={result['swaps']}  bytes={result['bytes_moved']}  "
        f"quente/morno/frio={result['hot']}/{result['warm']}/{result['cold']}  "
        f"arquivos frios={result['cold_files']}"
    )


if __name__ == "__main__":
    main()
