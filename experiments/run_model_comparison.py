#!/usr/bin/env python3
"""Nomeia os quatro braços e monta a tabela de comparação. Não treina.

O modelo 2 não é outra política de eviction. ``ResidencyEmulator`` já é
LRU; o braço 2 é o mesmo MoE com λ=0 e o mesmo checkpoint da semente 0
do modelo 1 (``seed-0``, ou o layout antigo se for o único arquivo).
A diferença é a fatia de relatório: hold-out no B de referência e a
curva ``dose_response`` (reset no B do YAML, depois ``set_hot_budget``)
quando o JSON existir, contra o denso e contra λ>0.

Números, se existirem, vêm de fora:

- ``outputs/holdout/summary.json`` — perplexidade no B de referência.
  ``dose_response: true`` traz a curva em cada corrida.
- ``outputs/holdout/dose_response.json`` — protocolo
  ``wikitext-2-dose-response``. Sem esse arquivo e sem o bloco na
  summary, a dose fica ``pending``.

Este script não chama o eval, não chama ``set_hot_budget`` e não grava
checkpoint. ``pending`` não é número de relatório.

Uso, a partir da raiz::

    python3 experiments/run_model_comparison.py
    python3 experiments/run_model_comparison.py --manifest-only --out /tmp/comparison.json
    python3 experiments/run_model_comparison.py --holdout outputs/holdout/summary.json

Códigos: 1 uso, 2 JSON ou manifesto incompatível com o protocolo.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moe_memory.comparison import ComparisonError, build_comparison, format_summary
from moe_memory.config import load_config

EXIT_USAGE = 1
EXIT_SCHEMA = 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tabela dos quatro braços. O modelo 2 é o mesmo LRU com λ=0, "
            "não uma segunda política. Não treina."
        )
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--outputs",
        type=Path,
        default=ROOT / "outputs",
        help="Raiz dos checkpoints (o mesmo layout do hold-out).",
    )
    parser.add_argument(
        "--holdout",
        type=Path,
        default=None,
        help="summary.json do run_holdout_eval. Padrão: <outputs>/holdout/summary.json se existir.",
    )
    parser.add_argument(
        "--dose",
        type=Path,
        default=None,
        help=(
            "dose_response.json do hold-out. Sem arquivo, a curva fica pending "
            "salvo o summary já trazer dose_response true."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="JSON da comparação. Padrão: <outputs>/comparison/comparison.json.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Não lê hold-out nem dose, mesmo que o arquivo exista.",
    )
    args = parser.parse_args(argv)
    if args.manifest_only and (args.holdout is not None or args.dose is not None):
        die_usage("use --manifest-only sem --holdout e sem --dose")
    return args


def die_usage(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(EXIT_USAGE)


def load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise ComparisonError(f"JSON ilegível em {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ComparisonError(f"{path} não é um objeto JSON")
    return data


def resolve_optional(explicit: Path | None, default: Path, *, enabled: bool) -> dict | None:
    if not enabled:
        return None
    path = default if explicit is None else explicit
    if explicit is None and not path.is_file():
        return None
    if not path.is_file():
        raise ComparisonError(f"não achei {path}")
    return load_json(path)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> None:
    try:
        _main(argv)
    except ComparisonError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(EXIT_SCHEMA) from exc


def _main(argv: list[str] | None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    read = not args.manifest_only
    holdout_default = Path(args.outputs) / "holdout" / "summary.json"
    holdout = resolve_optional(args.holdout, holdout_default, enabled=read)
    if args.holdout is not None:
        dose_default = Path(args.holdout).parent / "dose_response.json"
    else:
        dose_default = holdout_default.parent / "dose_response.json"
    dose = resolve_optional(args.dose, dose_default, enabled=read)
    payload = build_comparison(
        config,
        outputs=args.outputs,
        holdout=holdout,
        dose=dose,
    )
    out = Path(args.out) if args.out is not None else Path(args.outputs) / "comparison" / "comparison.json"
    write_json(out, payload)
    print(format_summary(payload), flush=True)
    print(f"json={out}", flush=True)


if __name__ == "__main__":
    main()
