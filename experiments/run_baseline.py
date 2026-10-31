#!/usr/bin/env python3
"""Corrida do LM denso no WikiText-2, com a mesma retomada da grade de λ.

Não percorre ``λ``. O modelo não tem roteador nem residência: não é o
ponto ``λ=0`` da grade MoE. β, E, k, B e a tabela ``c`` continuam no YAML
para o MoE; aqui só entram na largura do FFN (``k * expert_hidden``) e
no tronco compartilhado. Hold-out e dose–resposta ficam de fora.

Se o filho morrer por OOM, loss não finita, crash ou kill -9, a próxima
tentativa desce o batch (4 → 2 → 1) e retoma o checkpoint. SIGTERM/SIGINT
no filho espera o passo em curso, grava o checkpoint e o runner repete o
mesmo batch. Erro de uso não reduz batch.

Uso, a partir da raiz, na GTX 1060 (corpus padrão = WikiText-2)::

    python3 experiments/run_baseline.py --device cuda

TinyStories é opcional e não é o número de relatório::

    python3 experiments/run_baseline.py --corpus TinyStories --steps 2 --device cpu

Saída da entrega em ``outputs/baseline/wikitext/`` (gitignored):
``summary.json``, ``done.json``, ``train.jsonl``, ``train.log``,
``checkpoint.pt``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import torch

from moe_memory.config import ExperimentConfig, load_config
from moe_memory.dense import MODEL_KIND, capacity_report
from run_lambda_grid import (
    MAX_INTERRUPTS,
    batch_ladder,
    should_resume_same,
    should_shrink,
    subprocess_launch,
)
from train_baseline import (
    NOTE,
    canonical_corpus,
    default_seq_len,
    default_work_dir,
)
from train_wikitext import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINT_EVERY,
    EXIT_INTERRUPT,
)

Launch = Callable[[list[str], Path], int]


def is_complete(
    done_path: Path,
    *,
    corpus: str,
    seed: int,
    seq_len: int,
    steps: int | None,
    epochs: int | None,
) -> bool:
    """O alvo pedido já está no ``done.json`` do LM denso."""

    if not done_path.is_file():
        return False
    try:
        done = json.loads(done_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if done.get("model") != MODEL_KIND or done.get("memory_path") is not False:
        return False
    if done.get("corpus") != corpus:
        return False
    if done.get("lambda") not in (None,):
        return False
    if int(done.get("seed", -1)) != int(seed):
        return False
    if int(done.get("seq_len", -1)) != int(seq_len):
        return False
    if not done.get("finite", False):
        return False
    if steps is not None:
        return int(done.get("steps", 0)) >= int(steps)
    if epochs is None:
        return False
    return int(done.get("epochs_completed", 0)) >= int(epochs)


def build_command(
    *,
    corpus_flag: str,
    config_path: Path | None,
    device: str,
    seq_len: int,
    batch_size: int,
    seed: int,
    steps: int | None,
    epochs: int | None,
    checkpoint_every: int,
    work_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "train_baseline.py"),
        "--corpus",
        corpus_flag,
        "--device",
        device,
        "--seq-len",
        str(seq_len),
        "--batch-size",
        str(batch_size),
        "--seed",
        str(seed),
        "--checkpoint-every",
        str(checkpoint_every),
        "--work-dir",
        str(work_dir),
    ]
    if config_path is not None:
        command.extend(["--config", str(config_path)])
    if steps is not None:
        command.extend(["--steps", str(steps)])
    else:
        command.extend(["--epochs", str(epochs)])
    return command


def run_baseline(
    config: ExperimentConfig,
    request: argparse.Namespace,
    launch: Launch,
) -> dict[str, Any]:
    work = Path(request.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    capacity = capacity_report(config)
    summary: dict[str, Any] = {
        "model": MODEL_KIND,
        "memory_path": False,
        "note": NOTE,
        "corpus": request.corpus_name,
        "device": request.device,
        "seq_len": request.seq_len,
        "seed": request.seed,
        "steps": request.steps,
        "epochs": request.epochs,
        "batch_size_start": request.batch_size,
        "dtype": "fp32",
        "torch_compile": False,
        "loss": "L_task",
        "lambda": None,
        "d_ff": capacity["d_ff"],
        "parameters": capacity["dense_parameters"],
        "moe_total_parameters": capacity["moe_total_parameters"],
        "moe_active_parameters": capacity["moe_active_parameters"],
        "metric_fields": ["status", "steps", "tokens", "last_total", "last_task"],
        "runs": [],
    }
    summary_path = work / "summary.json"
    print(NOTE)
    print(
        f"Baseline denso  corpus={request.corpus_name}  device={request.device}  "
        f"seq={request.seq_len}  batch={request.batch_size}  seed={request.seed}"
    )
    print("fp32, sem torch.compile. Não é a grade de λ. Hold-out e dose–resposta ficam fora.")
    if request.corpus_name != "WikiText-2":
        print("TinyStories é o caminho de dev. O relatório deste baseline é o WikiText-2.")
    _write_summary(summary_path, summary)
    result = run_one(config_path=request.config, request=request, launch=launch)
    summary["runs"].append(result)
    _write_summary(summary_path, summary)
    print(f"status={result['status']}  resumo: {summary_path}")
    if result["status"] != "done":
        print("alvo não cumprido")
    else:
        print(
            f"passos={result.get('steps')}  tokens={result.get('tokens')}  "
            f"L={result.get('last_total')}  L_task={result.get('last_task')}"
        )
    return summary


def run_one(
    *,
    config_path: Path | None,
    request: argparse.Namespace,
    launch: Launch,
) -> dict[str, Any]:
    work = Path(request.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    log_path = work / "train.log"
    done_path = work / "done.json"
    base = {
        "model": MODEL_KIND,
        "corpus": request.corpus_name,
        "work_dir": str(work),
        "log": str(log_path),
        "lambda": None,
    }
    if is_complete(
        done_path,
        corpus=request.corpus_name,
        seed=request.seed,
        seq_len=request.seq_len,
        steps=request.steps,
        epochs=request.epochs,
    ):
        done = json.loads(done_path.read_text(encoding="utf-8"))
        print(f"{request.corpus_name} já cumpriu o alvo; não relança")
        return {
            **base,
            "status": "done",
            "skipped": True,
            "exit_code": 0,
            "batch_size": done.get("batch_size"),
            "steps": done.get("steps"),
            "epochs_completed": done.get("epochs_completed"),
            "tokens": done.get("tokens"),
            "last_total": done.get("last_total"),
            "last_task": done.get("last_task"),
            "recoveries": [],
        }
    ladder = batch_ladder(request.batch_size)
    recoveries: list[dict[str, Any]] = []
    interrupts = 0
    index = 0
    while index < len(ladder):
        batch = ladder[index]
        command = build_command(
            corpus_flag=request.corpus,
            config_path=config_path,
            device=request.device,
            seq_len=request.seq_len,
            batch_size=batch,
            seed=request.seed,
            steps=request.steps,
            epochs=request.epochs,
            checkpoint_every=request.checkpoint_every,
            work_dir=work,
        )
        print(f"batch={batch}  (degrau {index + 1}/{len(ladder)})")
        code = launch(command, log_path)
        if code == 0:
            if not is_complete(
                done_path,
                corpus=request.corpus_name,
                seed=request.seed,
                seq_len=request.seq_len,
                steps=request.steps,
                epochs=request.epochs,
            ):
                return {
                    **base,
                    "status": "failed",
                    "skipped": False,
                    "exit_code": 0,
                    "reason": "saiu 0 sem done.json compatível com o alvo",
                    "batch_size": batch,
                    "recoveries": recoveries,
                    "log_tail": _tail(log_path),
                }
            done = json.loads(done_path.read_text(encoding="utf-8"))
            return {
                **base,
                "status": "done",
                "skipped": False,
                "exit_code": 0,
                "batch_size": batch,
                "steps": done.get("steps"),
                "epochs_completed": done.get("epochs_completed"),
                "tokens": done.get("tokens"),
                "last_total": done.get("last_total"),
                "last_task": done.get("last_task"),
                "recoveries": recoveries,
            }
        if should_resume_same(code):
            interrupts += 1
            recoveries.append({"event": "interrupt", "exit_code": code, "batch_size": batch})
            print(f"interrompido; retoma o mesmo batch ({interrupts}/{MAX_INTERRUPTS})")
            if interrupts > MAX_INTERRUPTS:
                break
            continue
        if should_shrink(code):
            recoveries.append({"event": "shrink", "exit_code": code, "batch_size": batch})
            print(f"saiu {code}; próximo batch menor se ainda houver degrau")
            index += 1
            continue
        return {
            **base,
            "status": "failed",
            "skipped": False,
            "exit_code": code,
            "reason": "erro de uso; o batch não foi reduzido",
            "batch_size": batch,
            "recoveries": recoveries,
            "log_tail": _tail(log_path),
        }
    return {
        **base,
        "status": "failed",
        "skipped": False,
        "reason": "lote mínimo ainda falhou (OOM, NaN, crash ou interrupção repetida)",
        "batch_size": ladder[-1],
        "recoveries": recoveries,
        "log_tail": _tail(log_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina o LM denso no WikiText-2, com retomada 4→2→1. Não é a grade de λ."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--corpus",
        default="WikiText-2",
        choices=("WikiText-2", "TinyStories", "report", "dev"),
        help="Padrão: WikiText-2. TinyStories ou dev é só desenvolvimento.",
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--device",
        default=None,
        help="Padrão: cuda se existir, senão cpu. Na 1060 passe --device cuda.",
    )
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Pasta da corrida. Padrão: outputs/baseline/wikitext (ou tinystories).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    args = parser.parse_args(argv)
    if args.steps is None and args.epochs is None:
        args.epochs = 1
    if args.steps is not None and args.epochs is not None:
        raise SystemExit("use só --steps ou só --epochs")
    if args.steps is not None and args.steps < 1:
        raise SystemExit("--steps precisa ser >= 1")
    if args.epochs is not None and args.epochs < 1:
        raise SystemExit("--epochs precisa ser >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size precisa ser >= 1")
    if args.seed < 0:
        raise SystemExit("--seed precisa ser >= 0")
    if args.work_dir is None:
        args.work_dir = default_work_dir(args.corpus)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.device == "cpu":
            print("device=cpu (sem CUDA nesta máquina; na GTX 1060 use --device cuda)")
    elif args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "device cuda sem GPU visível. Na GTX 1060 instale o wheel CUDA do PyTorch."
        )
    return args


def resolve_request(config: ExperimentConfig, args: argparse.Namespace) -> argparse.Namespace:
    """Fecha corpus e seq_len para o filho e o done.json falarem a mesma coisa."""

    args.corpus_name = canonical_corpus(config, args.corpus)
    if args.seq_len is None:
        args.seq_len = default_seq_len(config, args.corpus_name)
    if not 2 <= args.seq_len <= config.model.max_seq_len:
        raise SystemExit(
            f"seq_len={args.seq_len} fora de 2..max_seq_len={config.model.max_seq_len}"
        )
    return args


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _tail(path: Path, lines: int = 30) -> str:
    if not path.is_file():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    request = resolve_request(config, args)
    try:
        summary = run_baseline(config, request, subprocess_launch)
    except KeyboardInterrupt:
        print("baseline interrompido; o que já fechou está no summary.json", file=sys.stderr)
        raise SystemExit(EXIT_INTERRUPT) from None
    if any(run["status"] != "done" for run in summary["runs"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
