"""Imprime o saldo de ativação do gancho SiDA. Não mede e não treina."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moe_memory.lstm_predictor import load_lstm_config
from moe_memory.sida_budget import ActivationBudget, describe_budget


def main() -> None:
    config = load_lstm_config()
    budget = ActivationBudget(k=config.k, n_layers=config.n_model_layers)
    print(describe_budget(budget))


if __name__ == "__main__":
    main()
