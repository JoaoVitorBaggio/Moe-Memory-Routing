#!/usr/bin/env python3
"""Grade λ ∈ {0, 0,5, 1} no WikiText-2, um treino depois do outro.

Cada valor de ``loss.lambda_grid`` é uma curva. β, E, k, B e a arquitetura
continuam no YAML. Este runner não mede perplexidade nem dose–resposta;
o hold-out é ``experiments/run_holdout_eval.py``.

Se o filho morrer por OOM, loss não finita, crash ou kill -9, a próxima
tentativa desce o batch (4 → 2 → 1) e retoma o checkpoint. SIGTERM/SIGINT
no filho espera o passo em curso, grava o checkpoint e a grade repete o
mesmo batch. Erro de uso não reduz batch. Nada disso pergunta no meio da
corrida. Um λ que esgota o lote 1 entra no resumo como falha e o próximo
λ segue.

``eval.seeds`` do YAML define quantas sementes correm: ``0 .. seeds-1``.
``--seed N`` limita a essa semente (retomada pontual). Cada semente repete
a grade inteira antes da próxima.

Uso, a partir da raiz, na GTX 1060::

    python3 experiments/run_lambda_grid.py --device cuda
    python3 experiments/run_lambda_grid.py --seed 1 --device cuda

Saída em ``outputs/wikitext/`` (gitignored): ``summary.json`` na raiz, e
por semente ``seed-<n>/lambda-<λ>/`` com ``train.jsonl``, ``train.log``,
``checkpoint.pt``. A semente 0 que já está em ``lambda-<λ>/`` (sem
``seed-0``) continua nesse lugar. O resumo lista o status e o caminho
de cada semente.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
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
from seed_layout import (
    LEGACY_LAYOUT_NOTE,
    lambda_run_dir,
    per_seed_report,
    resolve_lambda_run_dir,
    seed_indices,
)
from train_wikitext import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINT_EVERY,
    EXIT_CRASH,
    EXIT_INTERRUPT,
    EXIT_NONFINITE,
    EXIT_OOM,
    same_lambda,
)

MAX_INTERRUPTS = 3
Launch = Callable[[list[str], Path], int]


def apply_seed_request(config: ExperimentConfig, args: argparse.Namespace) -> None:
    """Preenche ``args.seeds`` a partir de ``eval.seeds`` ou de ``--seed``."""

    if args.seed is None:
        args.seeds = seed_indices(config.eval.seeds)
        return
    if args.seed < 0:
        raise SystemExit("--seed precisa ser >= 0")
    args.seeds = [int(args.seed)]


def iteration_seeds(request: argparse.Namespace) -> list[int]:
    """Sementes desta invocação. Sem ``seeds``, vale o ``seed`` único do pedido."""

    seeds = getattr(request, "seeds", None)
    if seeds is None:
        if request.seed is None:
            raise RuntimeError("seeds ausentes; chame resolve_request")
        return [int(request.seed)]
    chosen = [int(seed) for seed in seeds]
    if not chosen:
        raise SystemExit("a lista de sementes ficou vazia")
    if any(seed < 0 for seed in chosen):
        raise SystemExit("--seed precisa ser >= 0")
    return chosen


def batch_ladder(start: int) -> list[int]:
    """Tamanhos de batch desta tentativa, do pedido até 1."""

    if start < 1:
        raise ValueError("batch_size precisa ser >= 1")
    ladder: list[int] = []
    current = start
    while current not in ladder:
        ladder.append(current)
        if current == 1:
            break
        current = max(1, current // 2)
    return ladder


def should_shrink(exit_code: int) -> bool:
    """OOM, NaN, crash ou morte por sinal. Erro de uso (1 ou 2) fica de fora."""

    if exit_code in {EXIT_OOM, EXIT_NONFINITE, EXIT_CRASH}:
        return True
    return exit_code < 0 or exit_code >= 128


def should_resume_same(exit_code: int) -> bool:
    return exit_code == EXIT_INTERRUPT


def is_complete(
    done_path: Path,
    *,
    lam: float,
    seed: int,
    seq_len: int,
    steps: int | None,
    epochs: int | None,
) -> bool:
    """O alvo pedido já está no ``done.json``. Batch menor na retomada ainda conta."""

    if not done_path.is_file():
        return False
    try:
        done = json.loads(done_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if not same_lambda(done.get("lambda", -1.0), lam):
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
    lam: float,
    *,
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
        str(SCRIPT_DIR / "train_wikitext.py"),
        "--lambda",
        str(float(lam)),
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


def run_grid(config: ExperimentConfig, request: argparse.Namespace, launch: Launch) -> dict[str, Any]:
    output_root = Path(request.work_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    seeds = iteration_seeds(request)
    summary: dict[str, Any] = {
        "corpus": "WikiText-2",
        "device": request.device,
        "seq_len": request.seq_len,
        "seed": seeds[0] if len(seeds) == 1 else None,
        "seeds": seeds,
        "eval_seeds": config.eval.seeds,
        "steps": request.steps,
        "epochs": request.epochs,
        "beta": config.loss.beta,
        "batch_size_start": request.batch_size,
        "lambda_grid": [float(value) for value in config.loss.lambda_grid],
        "layout": "seed-<n>/lambda-<λ>",
        "layout_note": LEGACY_LAYOUT_NOTE,
        "runs": [],
        "per_seed": [],
    }
    summary_path = output_root / "summary.json"
    print(
        "Grade WikiText-2  λ="
        + ", ".join(str(value) for value in config.loss.lambda_grid)
        + f"  β={config.loss.beta}  device={request.device}  seq={request.seq_len}"
    )
    print(
        "sementes desta corrida: "
        + ", ".join(str(seed) for seed in seeds)
        + f"  (eval.seeds={config.eval.seeds})"
    )
    print("fp32, sem torch.compile. Hold-out e dose–resposta ficam fora desta grade.")
    total = len(seeds) * len(config.loss.lambda_grid)
    step = 0
    for seed in seeds:
        for lam in config.loss.lambda_grid:
            step += 1
            print(f"[{step}/{total}] seed={seed} λ={lam}")
            result = run_one(
                float(lam),
                config_path=request.config,
                request=request,
                launch=launch,
                seed=seed,
            )
            summary["runs"].append(result)
            _publish(summary_path, summary)
            print(f"seed={seed} λ={lam}  status={result['status']}")
    failed = [
        f"seed={run['seed']} λ={run['lambda']}"
        for run in summary["runs"]
        if run["status"] != "done"
    ]
    print(f"resumo: {summary_path}")
    if failed:
        print("sem alvo cumprido: " + ", ".join(failed))
    else:
        print("todos os λ de todas as sementes cumpriram o alvo")
    return summary


def run_one(
    lam: float,
    *,
    config_path: Path | None,
    request: argparse.Namespace,
    launch: Launch,
    seed: int | None = None,
) -> dict[str, Any]:
    if seed is None:
        seed = request.seed
    if seed is None:
        raise RuntimeError("semente ausente; chame resolve_request ou passe seed")
    seed = int(seed)
    root = Path(request.work_dir)
    canonical = lambda_run_dir(root, lam, seed)
    work = resolve_lambda_run_dir(root, lam, seed)
    work.mkdir(parents=True, exist_ok=True)
    log_path = work / "train.log"
    done_path = work / "done.json"
    base = {
        "seed": seed,
        "lambda": float(lam),
        "work_dir": str(work),
        "log": str(log_path),
        "legacy_layout": work != canonical,
    }
    if is_complete(
        done_path,
        lam=lam,
        seed=seed,
        seq_len=request.seq_len,
        steps=request.steps,
        epochs=request.epochs,
    ):
        done = json.loads(done_path.read_text(encoding="utf-8"))
        print(f"seed={seed} λ={lam} já cumpriu o alvo; não relança")
        return {
            **base,
            "status": "done",
            "skipped": True,
            "batch_size": done.get("batch_size"),
            "steps": done.get("steps"),
            "epochs_completed": done.get("epochs_completed"),
            "tokens": done.get("tokens"),
            "recoveries": [],
        }
    ladder = batch_ladder(request.batch_size)
    recoveries: list[dict[str, Any]] = []
    interrupts = 0
    index = 0
    while index < len(ladder):
        batch = ladder[index]
        command = build_command(
            lam,
            config_path=config_path,
            device=request.device,
            seq_len=request.seq_len,
            batch_size=batch,
            seed=seed,
            steps=request.steps,
            epochs=request.epochs,
            checkpoint_every=request.checkpoint_every,
            work_dir=work,
        )
        print(f"seed={seed} λ={lam}  batch={batch}  (degrau {index + 1}/{len(ladder)})")
        code = launch(command, log_path)
        if code == 0:
            if not is_complete(
                done_path,
                lam=lam,
                seed=seed,
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
            print(
                f"seed={seed} λ={lam} interrompido; "
                f"retoma o mesmo batch ({interrupts}/{MAX_INTERRUPTS})"
            )
            if interrupts > MAX_INTERRUPTS:
                break
            continue
        if should_shrink(code):
            recoveries.append({"event": "shrink", "exit_code": code, "batch_size": batch})
            print(f"seed={seed} λ={lam} saiu {code}; próximo batch menor se ainda houver degrau")
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


def subprocess_launch(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n$ " + " ".join(command) + "\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(completed.returncode)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina cada λ da grade no WikiText-2, em sequência, com retomada."
    )
    parser.add_argument("--config", type=Path, default=None)
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
        default=ROOT / "outputs" / "wikitext",
        help="Pasta da grade. Cada λ ganha lambda-<valor>/ dentro dela.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Uma semente só. Sem esta flag, percorre 0 .. eval.seeds-1 do YAML.",
    )
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
    if args.seed is not None and args.seed < 0:
        raise SystemExit("--seed precisa ser >= 0")
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
    """Fecha seq_len no max do YAML para o filho e o done.json falarem a mesma coisa."""

    if args.seq_len is None:
        args.seq_len = config.model.max_seq_len
    if not 2 <= args.seq_len <= config.model.max_seq_len:
        raise SystemExit(
            f"seq_len={args.seq_len} fora de 2..max_seq_len={config.model.max_seq_len}"
        )
    apply_seed_request(config, args)
    return args


def _publish(path: Path, summary: dict[str, Any]) -> None:
    summary["per_seed"] = per_seed_report(summary["runs"])
    _write_summary(path, summary)


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
    if config.corpus.report != "WikiText-2":
        raise SystemExit(f"corpus.report={config.corpus.report!r}; a entrega é WikiText-2")
    request = resolve_request(config, args)
    try:
        summary = run_grid(config, request, subprocess_launch)
    except KeyboardInterrupt:
        print("grade interrompida; o que já fechou está no summary.json", file=sys.stderr)
        raise SystemExit(EXIT_INTERRUPT) from None
    if any(run["status"] != "done" for run in summary["runs"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
