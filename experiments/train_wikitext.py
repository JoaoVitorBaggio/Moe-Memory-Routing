#!/usr/bin/env python3
"""Treino de um λ no WikiText-2 (split oficial train). Não avalia hold-out.

A perplexidade de valid/test é ``experiments/run_holdout_eval.py``.

Reusa ``prepare_train`` e ``train_one_batch``. fp32, sem ``torch.compile``.
λ, β, E, k, B e a arquitetura saem de ``configs/default.yaml``; este script
não os redefine. A loss do jsonl é de treino, não perplexidade de relatório.

Padrão overnight: uma época, ``seq_len = model.max_seq_len``, batch operacional
4 (a grade reduz se a VRAM estourar). ``--steps`` corta em passos de otimizador
e é exclusivo com ``--epochs``.

Uso, a partir da raiz::

    python3 experiments/train_wikitext.py --lambda 0.5 --device cuda
    python3 experiments/train_wikitext.py --lambda 0 --steps 2 --device cpu --seq-len 32

Retomar é rodar o mesmo comando: o checkpoint em ``--work-dir`` continua
o cursor de sequências. OOM e loss não finita saem com código próprio para
a grade encolher o batch sem alguém no meio. SIGTERM/SIGINT espera o passo
em curso e só então grava o checkpoint.

Este script treina uma semente (``--seed``, padrão 0). Quem percorre
``eval.seeds`` é ``experiments/run_lambda_grid.py``. O ``--work-dir``
padrão é ``outputs/wikitext/seed-<n>/lambda-<λ>/``. Se a semente 0 já
tiver artefato no layout antigo (``outputs/wikitext/lambda-<λ>/``) e a
pasta nova ainda estiver vazia, o padrão continua no lugar antigo.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import torch

from moe_memory.config import ExperimentConfig, load_config
from seed_layout import lambda_tag, resolve_lambda_run_dir
from moe_memory.corpus import iter_split, source_label
from moe_memory.smoke import prepare_train, train_one_batch

# Códigos estáveis para a grade. 2 fica com o argparse.
EXIT_USAGE = 1
EXIT_OOM = 3
EXIT_NONFINITE = 4
EXIT_CRASH = 5
EXIT_INTERRUPT = 6

# Partida operacional na GTX 1060 em fp32 com seq_len = max_seq_len.
# Não é eixo do protocolo (esse continua no YAML). A grade desce 4 → 2 → 1.
DEFAULT_BATCH_SIZE = 4
DEFAULT_CHECKPOINT_EVERY = 25
CHECKPOINT_VERSION = 1

# Sinal só arma a bandeira. O save corre entre passos, com o cursor já fechado.
_STOP = {"now": False}


def same_lambda(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-6)


def losses_are_finite(result: dict[str, Any]) -> bool:
    keys = ("total", "task", "c_mem", "balance", "grad_norm")
    return all(math.isfinite(float(result[key])) for key in keys)


def is_oom(exc: BaseException) -> bool:
    if "OutOfMemory" in type(exc).__name__:
        return True
    return "out of memory" in str(exc).lower()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treino de um λ no WikiText-2 (fp32, split train)."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--lambda",
        dest="lambda_",
        type=float,
        required=True,
        help="Um valor de loss.lambda_grid. Cada λ é uma curva separada.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Passos de otimizador. Exclusivo com --epochs.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Passagens pelo train. Padrão: 1 se --steps não vier.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Padrão: model.max_seq_len (256 no YAML; knob até 512).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Padrão operacional {DEFAULT_BATCH_SIZE}. Não é eixo do protocolo.",
    )
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Não grava checkpoint.pt, nem no --checkpoint-every. O log e o done.json continuam.",
    )
    args = parser.parse_args(argv)
    if args.steps is None and args.epochs is None:
        args.epochs = 1
    if args.steps is not None and args.epochs is not None:
        die(EXIT_USAGE, "use só --steps ou só --epochs")
    if args.steps is not None and args.steps < 1:
        die(EXIT_USAGE, "--steps precisa ser >= 1")
    if args.epochs is not None and args.epochs < 1:
        die(EXIT_USAGE, "--epochs precisa ser >= 1")
    if args.batch_size < 1:
        die(EXIT_USAGE, "--batch-size precisa ser >= 1")
    if args.seed < 0:
        die(EXIT_USAGE, "--seed precisa ser >= 0")
    if args.checkpoint_every < 1:
        die(EXIT_USAGE, "--checkpoint-every precisa ser >= 1")
    if args.work_dir is None:
        args.work_dir = resolve_lambda_run_dir(
            ROOT / "outputs" / "wikitext",
            args.lambda_,
            args.seed,
        )
    return args


def train(args: argparse.Namespace) -> None:
    """Uma curva. Sai com SystemExit nos códigos que a grade entende.

    O handler de SIGTERM/SIGINT fica em ``main``: teste que chama ``train``
    não troca o sinal do pytest.
    """

    config = load_config(args.config)
    if config.corpus.report != "WikiText-2":
        die(
            EXIT_USAGE,
            f"corpus.report={config.corpus.report!r}; a entrega é WikiText-2",
        )
    seq_len = config.model.max_seq_len if args.seq_len is None else args.seq_len
    if not 2 <= seq_len <= config.model.max_seq_len:
        die(
            EXIT_USAGE,
            f"seq_len={seq_len} fora de 2..max_seq_len={config.model.max_seq_len}",
        )
    if args.lambda_ not in config.loss.lambda_grid and not any(
        same_lambda(args.lambda_, item) for item in config.loss.lambda_grid
    ):
        die(
            EXIT_USAGE,
            f"λ={args.lambda_} não está na grade {list(config.loss.lambda_grid)}",
        )
    # O valor da grade (0.0, não um float vizinho) é o que a loss usa.
    lambda_weight = _grid_value(config, args.lambda_)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    cold_dir = work_dir / "cold"
    log_path = work_dir / "train.jsonl"
    checkpoint_path = work_dir / "checkpoint.pt"

    print("Treino WikiText-2 (não é perplexidade de hold-out nem dose–resposta)")
    print(f"fonte: {source_label(config, 'WikiText-2')}")
    budget = f"passos={args.steps}" if args.steps is not None else f"épocas={args.epochs}"
    print(
        f"λ={lambda_weight}  β={config.loss.beta}  {budget}  "
        f"seq={seq_len}  batch={args.batch_size}  device={args.device}  seed={args.seed}"
    )
    print("dtype=fp32  torch.compile=não")
    try:
        sequences = load_train_sequences(config, seq_len)
    except SystemExit:
        raise
    except Exception as exc:
        die(
            EXIT_USAGE,
            "Não consegui ler o WikiText-2 train. Confira a rede, corpus.cache_dir "
            f"e corpus.paths.WikiText-2. Detalhe: {exc}",
        )
    n_sequences = int(sequences.size(0))
    print(
        f"sequências={n_sequences}  tokens_por_época={n_sequences * seq_len}  "
        f"work_dir={work_dir}"
    )

    device = torch.device(args.device)
    try:
        model, emulator, optimizer, residency = prepare_train(
            config,
            lambda_=lambda_weight,
            work_dir=cold_dir,
            device=device,
            seed=args.seed,
        )
    except ValueError as exc:
        die(EXIT_USAGE, str(exc))
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        die(EXIT_USAGE, "o modelo saiu de fp32")
    print(f"parâmetros: {model.num_parameters():,}")

    cursor = {
        "epoch": 0,
        "sequence_cursor": 0,
        "epochs_completed": 0,
        "steps": 0,
        "tokens": 0,
    }
    if checkpoint_path.is_file():
        restored_batch = _restore(
            checkpoint_path,
            config,
            model,
            emulator,
            optimizer,
            residency,
            lambda_weight=lambda_weight,
            seed=args.seed,
            seq_len=seq_len,
            lr=config.smoke.lr,
        )
        cursor.update(restored_batch["cursor"])
        if restored_batch["batch_size"] != args.batch_size:
            print(
                f"retomando com batch {args.batch_size} "
                f"(checkpoint tinha {restored_batch['batch_size']})"
            )
        print(
            f"retomando do passo {cursor['steps']} "
            f"(época {cursor['epoch']}, cursor {cursor['sequence_cursor']})"
        )
        if cursor["sequence_cursor"] > n_sequences:
            die(
                EXIT_USAGE,
                "checkpoint andou mais sequências do que o train tem agora. "
                "O corpus mudou; use outro --work-dir.",
            )
        _trim_log_to_checkpoint(log_path, cursor["steps"])
    elif log_path.is_file():
        backup = log_path.with_suffix(".jsonl.bak")
        log_path.replace(backup)
        print(f"sem checkpoint; log antigo foi para {backup.name}")

    state: dict[str, Any] = {
        "model": model,
        "optimizer": optimizer,
        "emulator": emulator,
        "residency": residency,
        "config": config,
        "cursor": cursor,
        "lambda": lambda_weight,
        "seed": args.seed,
        "seq_len": seq_len,
        "batch_size": args.batch_size,
        "lr": config.smoke.lr,
        "checkpoint_path": checkpoint_path,
    }

    def save() -> None:
        if args.no_checkpoint:
            return
        _write_checkpoint(state)

    if _target_met(cursor, args):
        _finish(state, args, log_path)
        return

    order = sequence_order(n_sequences, args.seed, cursor["epoch"])
    while not _target_met(cursor, args):
        if _STOP["now"]:
            save()
            die(
                EXIT_INTERRUPT,
                "interrompido; checkpoint não foi gravado"
                if args.no_checkpoint
                else "interrompido; checkpoint no último passo completo",
            )
        if cursor["sequence_cursor"] >= n_sequences:
            cursor["epochs_completed"] += 1
            cursor["epoch"] += 1
            cursor["sequence_cursor"] = 0
            save()
            if args.steps is None and cursor["epochs_completed"] >= args.epochs:
                break
            order = sequence_order(n_sequences, args.seed, cursor["epoch"])
            continue
        take = min(args.batch_size, n_sequences - cursor["sequence_cursor"])
        index = order[cursor["sequence_cursor"] : cursor["sequence_cursor"] + take]
        batch = sequences.index_select(0, index).to(device)
        try:
            result = train_one_batch(
                model,
                emulator,
                optimizer,
                config,
                batch,
                lambda_weight=lambda_weight,
                residency=residency,
            )
        except RuntimeError as exc:
            if is_oom(exc):
                _release_cuda()
                die(EXIT_OOM, f"OOM no passo {cursor['steps'] + 1}: {exc}")
            raise
        if not losses_are_finite(result):
            _append_jsonl(
                log_path,
                {
                    "event": "nonfinite",
                    "step_attempt": cursor["steps"] + 1,
                    "lambda": lambda_weight,
                    "batch_size": take,
                    "total": result["total"],
                    "task": result["task"],
                    "grad_norm": result["grad_norm"],
                },
            )
            die(
                EXIT_NONFINITE,
                f"loss ou gradiente não finito no passo {cursor['steps'] + 1}; "
                "checkpoint bom não foi atualizado",
            )
        cursor["sequence_cursor"] += take
        cursor["steps"] += 1
        cursor["tokens"] += take * seq_len
        if cursor["steps"] % args.checkpoint_every == 0:
            save()
        record = _record(result, cursor=cursor, batch_rows=take, seq_len=seq_len)
        _append_jsonl(log_path, record)
        print(
            f"passo {cursor['steps']}: L={result['total']:.4f}  "
            f"L_task={result['task']:.4f}  C_mem={result['c_mem']:.4f}  "
            f"|grad|={result['grad_norm']:.4f}  trocas={result['swaps']}  "
            f"batch={take}"
        )
        if _STOP["now"]:
            save()
            die(
                EXIT_INTERRUPT,
                "interrompido; checkpoint não foi gravado"
                if args.no_checkpoint
                else "interrompido; checkpoint no último passo completo",
            )
    save()
    _finish(state, args, log_path)


def load_train_sequences(config: ExperimentConfig, seq_len: int) -> torch.Tensor:
    """Blocos ``[N, seq_len]`` do train, na ordem do loader, sem embaralhar aqui.

    O embaralhamento da época fica em ``sequence_order`` (o loader deixa
    isso para a corrida overnight). O resto que não fecha ``seq_len`` cai fora.
    """

    rows: list[torch.Tensor] = []
    stream = iter_split(
        config,
        "train",
        name="WikiText-2",
        seq_len=seq_len,
        batch_size=64,
        streaming=False,
    )
    try:
        for batch in stream:
            rows.append(batch.cpu())
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()
    if not rows:
        die(
            EXIT_USAGE,
            f"WikiText-2 train não fechou sequência de seq_len={seq_len}",
        )
    return torch.cat(rows, dim=0)


def sequence_order(n_sequences: int, seed: int, epoch: int) -> torch.Tensor:
    """Permutação estável da época. A semente da corrida mexe no peso e na ordem."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(_mix_seed(seed, epoch))
    return torch.randperm(n_sequences, generator=generator)


def _mix_seed(seed: int, epoch: int) -> int:
    return (int(seed) * 1_000_003 + int(epoch) * 97 + 17) % (2**31 - 1)


def _grid_value(config: ExperimentConfig, requested: float) -> float:
    for item in config.loss.lambda_grid:
        if same_lambda(item, requested):
            return float(item)
    return float(requested)


def _target_met(cursor: dict[str, int], args: argparse.Namespace) -> bool:
    if args.steps is not None:
        return cursor["steps"] >= args.steps
    return cursor["epochs_completed"] >= args.epochs


def _record(
    result: dict[str, Any],
    *,
    cursor: dict[str, int],
    batch_rows: int,
    seq_len: int,
) -> dict[str, Any]:
    return {
        "event": "step",
        "step": cursor["steps"],
        "epoch": cursor["epoch"],
        "epochs_completed": cursor["epochs_completed"],
        "sequence_cursor": cursor["sequence_cursor"],
        "batch_rows": batch_rows,
        "tokens": cursor["tokens"],
        "seq_len": seq_len,
        "lambda": result["lambda"],
        "beta": result["beta"],
        "total": result["total"],
        "task": result["task"],
        "c_mem": result["c_mem"],
        "balance": result["balance"],
        "entropy": result["entropy"],
        "grad_norm": result["grad_norm"],
        "swaps": result["swaps"],
        "bytes_moved": result["bytes_moved"],
        "hot": result["hot"],
        "warm": result["warm"],
        "cold": result["cold"],
        "finite": True,
    }


def _trim_log_to_checkpoint(path: Path, steps: int) -> None:
    """Tira do jsonl passos que o checkpoint ainda não fechou.

    Um crash entre dois checkpoints deixa linhas à frente do cursor. Sem
    este corte a retomada repetiria o número do passo. Evento ``nonfinite``
    fica: é o rastro da tentativa que não entrou no checkpoint.
    """

    if not path.is_file():
        return
    kept: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("event") == "step" and int(record.get("step", 0)) > steps:
            continue
        kept.append(line)
    text = ("\n".join(kept) + "\n") if kept else ""
    path.write_text(text, encoding="utf-8")


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def _architecture(config: ExperimentConfig) -> dict[str, Any]:
    model = config.model
    return {
        "E": model.E,
        "k": model.k,
        "d_model": model.d_model,
        "n_layers": model.n_layers,
        "n_heads": model.n_heads,
        "expert_hidden": model.expert_hidden,
        "max_seq_len": model.max_seq_len,
        "vocab_size": model.vocab_size,
        "tie_embeddings": model.tie_embeddings,
        "B_frac": config.memory.B_frac,
        "beta": config.loss.beta,
        "c": list(config.cost.as_tuple()),
    }


def _write_checkpoint(state: dict[str, Any]) -> None:
    path: Path = state["checkpoint_path"]
    payload = {
        "version": CHECKPOINT_VERSION,
        "lambda": state["lambda"],
        "seed": state["seed"],
        "seq_len": state["seq_len"],
        "batch_size": state["batch_size"],
        "lr": state["lr"],
        "architecture": _architecture(state["config"]),
        "cursor": dict(state["cursor"]),
        "model": {
            key: value.detach().to(device="cpu").clone()
            for key, value in state["model"].state_dict().items()
        },
        "optimizer": _optimizer_state_cpu(state["optimizer"]),
        "emulator": _emulator_state(state["emulator"]),
        "torch_rng": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng"] = torch.cuda.get_rng_state_all()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _optimizer_state_cpu(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    # state_dict() devolve os MESMOS dicts de estado do otimizador vivo;
    # mutar ``slot`` in-place movia exp_avg/exp_avg_sq para a CPU e quebrava
    # o próximo optimizer.step() (cuda:0 vs cpu). Copiar antes de converter.
    blob = optimizer.state_dict()
    blob["state"] = {
        index: {
            key: value.detach().to(device="cpu").clone() if torch.is_tensor(value) else value
            for key, value in slot.items()
        }
        for index, slot in blob["state"].items()
    }
    return blob


def _emulator_state(emulator) -> dict[str, Any]:
    return {
        "levels": emulator.levels.detach().cpu().clone(),
        "last_used": emulator.last_used.detach().cpu().clone(),
        "last_count": emulator.last_count.detach().cpu().clone(),
        "tick": int(emulator.tick),
        "total_swaps": int(emulator.total_swaps),
        "total_bytes": int(emulator.total_bytes),
        "hot_slots": int(emulator.hot_slots),
    }


def _restore(
    path: Path,
    config: ExperimentConfig,
    model,
    emulator,
    optimizer,
    residency,
    *,
    lambda_weight: float,
    seed: int,
    seq_len: int,
    lr: float,
) -> dict[str, Any]:
    try:
        blob = _torch_load(path)
    except Exception as exc:
        die(EXIT_USAGE, f"checkpoint ilegível em {path}: {exc}")
    if blob.get("version") != CHECKPOINT_VERSION:
        die(EXIT_USAGE, f"versão de checkpoint {blob.get('version')!r} não é {CHECKPOINT_VERSION}")
    if not same_lambda(blob["lambda"], lambda_weight):
        die(
            EXIT_USAGE,
            f"checkpoint é λ={blob['lambda']}, este comando é λ={lambda_weight}. "
            "Use outro --work-dir.",
        )
    if int(blob["seed"]) != int(seed) or int(blob["seq_len"]) != int(seq_len):
        die(
            EXIT_USAGE,
            "checkpoint não bate com --seed/--seq-len. Use outro --work-dir.",
        )
    if not math.isclose(float(blob["lr"]), float(lr), rel_tol=0.0, abs_tol=1e-12):
        die(EXIT_USAGE, "checkpoint não bate com smoke.lr. Use outro --work-dir.")
    if blob["architecture"] != _architecture(config):
        die(
            EXIT_USAGE,
            "checkpoint não bate com a arquitetura do YAML (E, k, B, β, c, d). "
            "O desenho não se retoma por cima de outro.",
        )
    if int(blob["emulator"]["hot_slots"]) != int(emulator.hot_slots):
        die(EXIT_USAGE, "checkpoint não bate com o teto B do emulador")
    model.load_state_dict(blob["model"])
    optimizer.load_state_dict(blob["optimizer"])
    emulator.levels = blob["emulator"]["levels"].to(dtype=torch.long).clone()
    emulator.last_used = blob["emulator"]["last_used"].to(dtype=torch.long).clone()
    emulator.last_count = blob["emulator"]["last_count"].to(dtype=torch.long).clone()
    emulator.tick = int(blob["emulator"]["tick"])
    emulator.total_swaps = int(blob["emulator"]["total_swaps"])
    emulator.total_bytes = int(blob["emulator"]["total_bytes"])
    residency.materialize(optimizer)
    _move_optimizer_to_parameters(optimizer)
    torch.set_rng_state(blob["torch_rng"])
    if torch.cuda.is_available() and blob.get("cuda_rng"):
        torch.cuda.set_rng_state_all(blob["cuda_rng"])
    return {"cursor": dict(blob["cursor"]), "batch_size": int(blob["batch_size"])}


def _move_optimizer_to_parameters(optimizer: torch.optim.Optimizer) -> None:
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            slot = optimizer.state.get(parameter)
            if not slot:
                continue
            for key, value in list(slot.items()):
                if torch.is_tensor(value) and value.device != parameter.device:
                    slot[key] = value.to(device=parameter.device)


def _finish(state: dict[str, Any], args: argparse.Namespace, log_path: Path) -> None:
    cursor = state["cursor"]
    last = _last_step(log_path)
    done = {
        "lambda": state["lambda"],
        "seed": state["seed"],
        "seq_len": state["seq_len"],
        "batch_size": state["batch_size"],
        "steps": cursor["steps"],
        "epochs_completed": cursor["epochs_completed"],
        "tokens": cursor["tokens"],
        "target_steps": args.steps,
        "target_epochs": args.epochs,
        "finite": True,
        "checkpoint_saved": not args.no_checkpoint,
        "last_total": None if last is None else last.get("total"),
        "last_task": None if last is None else last.get("task"),
        "last_grad_norm": None if last is None else last.get("grad_norm"),
    }
    path = state["checkpoint_path"].parent / "done.json"
    path.write_text(json.dumps(done, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"fim: passos={cursor['steps']}  épocas_fechadas={cursor['epochs_completed']}  "
        f"tokens={cursor['tokens']}  λ={state['lambda']}  β={state['config'].loss.beta}"
    )
    print(f"log={log_path}")
    print("perplexidade de valid/test: experiments/run_holdout_eval.py. Dose–resposta não entra aqui.")


def _last_step(log_path: Path) -> dict[str, Any] | None:
    if not log_path.is_file():
        return None
    last = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("event") == "step":
            last = record
    return last


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _release_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _install_signals() -> None:
    def handler(signum: int, _frame) -> None:
        _STOP["now"] = True
        print(
            f"sinal {signum}: o passo em curso termina e o checkpoint é gravado",
            file=sys.stderr,
        )

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def die(code: int, message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(code)


def main(argv: list[str] | None = None) -> None:
    _install_signals()
    try:
        args = parse_args(argv)
        train(args)
    except SystemExit:
        raise
    except Exception as exc:
        if is_oom(exc):
            _release_cuda()
            die(EXIT_OOM, f"OOM: {exc}")
        traceback.print_exc()
        die(EXIT_CRASH, f"falha inesperada: {exc}")


if __name__ == "__main__":
    main()
