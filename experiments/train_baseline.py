#!/usr/bin/env python3
"""Treino do LM denso de referência. Não é o MoE com λ=0.

λ=0 ainda usa experts, roteador, emulador e ``β L_balanço``; só zera o
peso de ``C_mem``. Aqui a loss é só ``L_tarefa`` (CE do próximo token).
Não há residência, troca, nem ``torch.compile``. fp32. A perplexidade de
valid/test é ``experiments/run_holdout_eval.py``.

O corpus padrão é o WikiText-2 (entrega). TinyStories é caminho opcional
de desenvolvimento: o contexto padrão desse caminho é o ``smoke.seq_len``,
como ``train_tinystories.py``. No WikiText-2 o contexto é
``model.max_seq_len`` (256 no YAML).

Padrão operacional, o mesmo da grade overnight: uma época, batch 4.
``--seed`` padrão é 0. Quem percorre ``eval.seeds`` é
``experiments/run_baseline.py``, um processo por semente. O
``--work-dir`` padrão é ``outputs/baseline/<corpus>/seed-<n>/``. Se a
semente 0 já tiver artefato na raiz antiga e ``seed-0`` ainda estiver
vazio, o padrão continua na raiz. A grade densa desce 4 → 2 → 1 se a
VRAM estourar. ``--steps`` corta em passos e é exclusivo com ``--epochs``.

Uso, a partir da raiz::

    python3 experiments/train_baseline.py --device cuda
    python3 experiments/train_baseline.py --steps 2 --device cpu --seq-len 32
    python3 experiments/train_baseline.py --corpus TinyStories --steps 2 --device cpu

Retomar é rodar o mesmo comando. O checkpoint copia o Adam; não move o
estado vivo de device (a mesma correção de ``train_wikitext.py``).
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import torch

from moe_memory.config import ExperimentConfig, load_config
from moe_memory.corpus import iter_split, source_label
from moe_memory.dense import (
    MODEL_KIND,
    capacity_report,
    prepare_baseline,
    train_dense_step,
)
from seed_layout import resolve_baseline_seed_dir
from train_wikitext import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINT_EVERY,
    EXIT_CRASH,
    EXIT_INTERRUPT,
    EXIT_NONFINITE,
    EXIT_OOM,
    EXIT_USAGE,
    _append_jsonl,
    _move_optimizer_to_parameters,
    _optimizer_state_cpu,
    _release_cuda,
    _torch_load,
    _trim_log_to_checkpoint,
    die,
    is_oom,
    sequence_order,
)

CHECKPOINT_VERSION = 1
NOTE = (
    "LM denso sem roteador, sem residência e sem C_mem. "
    "Não é o MoE com λ=0 (esse ainda tem experts, emulador e L_balanço)."
)

# Sinal só arma a bandeira. O save corre entre passos, com o cursor já fechado.
_STOP = {"now": False}


def corpus_slug(flag: str) -> str:
    if flag in {"WikiText-2", "report"}:
        return "wikitext"
    if flag in {"TinyStories", "dev"}:
        return "tinystories"
    raise ValueError(f"corpus {flag!r} não tem pasta de saída")


def default_work_dir(flag: str) -> Path:
    return ROOT / "outputs" / "baseline" / corpus_slug(flag)


def canonical_corpus(config: ExperimentConfig, flag: str) -> str:
    """Nome do corpus. O padrão do script já é WikiText-2, não ``corpus.active``."""

    if flag == "report":
        name = config.corpus.report
    elif flag == "dev":
        name = config.corpus.dev
    else:
        name = flag
    if flag in {"WikiText-2", "report"} and config.corpus.report != "WikiText-2":
        die(EXIT_USAGE, f"corpus.report={config.corpus.report!r}; a entrega é WikiText-2")
    if flag in {"TinyStories", "dev"} and config.corpus.dev != "TinyStories":
        die(EXIT_USAGE, f"corpus.dev={config.corpus.dev!r}; o dev é TinyStories")
    if name not in {"WikiText-2", "TinyStories"}:
        die(EXIT_USAGE, f"corpus {name!r} não é WikiText-2 nem TinyStories")
    return name


def default_seq_len(config: ExperimentConfig, corpus: str) -> int:
    """WikiText-2 usa o contexto do YAML. TinyStories usa o smoke (dev)."""

    if corpus == "WikiText-2":
        return config.model.max_seq_len
    return config.smoke.seq_len


def losses_are_finite(result: dict[str, Any]) -> bool:
    return all(math.isfinite(float(result[key])) for key in ("total", "task", "grad_norm"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LM denso de referência no WikiText-2 (fp32). Não é o MoE com λ=0."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--corpus",
        default="WikiText-2",
        choices=("WikiText-2", "TinyStories", "report", "dev"),
        help=(
            "Padrão: WikiText-2 (entrega). TinyStories ou dev é só o caminho "
            "opcional de desenvolvimento."
        ),
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
        help="WikiText-2: model.max_seq_len. TinyStories: smoke.seq_len.",
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
        args.work_dir = resolve_baseline_seed_dir(default_work_dir(args.corpus), args.seed)
    return args


def train(args: argparse.Namespace) -> None:
    """Uma corrida densa. Sai com SystemExit nos códigos que o runner entende.

    O handler de SIGTERM/SIGINT fica em ``main``: teste que chama ``train``
    não troca o sinal do pytest.
    """

    _STOP["now"] = False
    config = load_config(args.config)
    corpus = canonical_corpus(config, args.corpus)
    seq_len = default_seq_len(config, corpus) if args.seq_len is None else args.seq_len
    if not 2 <= seq_len <= config.model.max_seq_len:
        die(
            EXIT_USAGE,
            f"seq_len={seq_len} fora de 2..max_seq_len={config.model.max_seq_len}",
        )
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = work_dir / "train.jsonl"
    checkpoint_path = work_dir / "checkpoint.pt"
    capacity = capacity_report(config)

    print(NOTE)
    print(f"Baseline denso  corpus={corpus}  (hold-out: experiments/run_holdout_eval.py)")
    print(f"fonte: {source_label(config, corpus)}")
    if corpus != "WikiText-2":
        print("TinyStories é só desenvolvimento. O número de relatório deste baseline é WikiText-2.")
    budget = f"passos={args.steps}" if args.steps is not None else f"épocas={args.epochs}"
    print(
        f"{budget}  seq={seq_len}  batch={args.batch_size}  "
        f"device={args.device}  seed={args.seed}  lr={config.smoke.lr}"
    )
    print("dtype=fp32  torch.compile=não  loss=L_task")
    print(
        f"parâmetros denso={capacity['dense_parameters']:,}  d_ff={capacity['d_ff']} "
        f"(= k*expert_hidden)  MoE total={capacity['moe_total_parameters']:,}  "
        f"MoE ativos≈{capacity['moe_active_parameters']:,}  "
        f"folga={capacity['active_budget_gap']:,} (roteador + viés fc2)"
    )
    try:
        sequences = load_train_sequences(config, seq_len, corpus)
    except SystemExit:
        raise
    except Exception as exc:
        die(
            EXIT_USAGE,
            f"Não consegui ler o {corpus} train. Confira a rede, corpus.cache_dir "
            f"e corpus.paths.{corpus}. Detalhe: {exc}",
        )
    n_sequences = int(sequences.size(0))
    print(
        f"sequências={n_sequences}  tokens_por_época={n_sequences * seq_len}  "
        f"work_dir={work_dir}"
    )

    device = torch.device(args.device)
    try:
        model, optimizer = prepare_baseline(config, device=device, seed=args.seed)
    except ValueError as exc:
        die(EXIT_USAGE, str(exc))
    if model.num_parameters() != capacity["dense_parameters"]:
        die(EXIT_CRASH, "contagem de parâmetros do denso não bate com a fórmula")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        die(EXIT_USAGE, "o modelo saiu de fp32")

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
            optimizer,
            corpus=corpus,
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
        "config": config,
        "cursor": cursor,
        "corpus": corpus,
        "seed": args.seed,
        "seq_len": seq_len,
        "batch_size": args.batch_size,
        "lr": config.smoke.lr,
        "capacity": capacity,
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
            result = train_dense_step(model, optimizer, batch)
        except RuntimeError as exc:
            if is_oom(exc):
                _release_cuda()
                die(EXIT_OOM, f"OOM no passo {cursor['steps'] + 1}: {exc}")
            raise
        if result["total"] != result["task"]:
            die(EXIT_CRASH, "baseline denso: L e L_task divergiram")
        if not losses_are_finite(result):
            _append_jsonl(
                log_path,
                {
                    "event": "nonfinite",
                    "model": MODEL_KIND,
                    "step_attempt": cursor["steps"] + 1,
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
        record = _record(result, cursor=cursor, batch_rows=take, seq_len=seq_len, corpus=corpus)
        _append_jsonl(log_path, record)
        print(
            f"passo {cursor['steps']}: L={result['total']:.4f}  "
            f"L_task={result['task']:.4f}  |grad|={result['grad_norm']:.4f}  "
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


def load_train_sequences(config: ExperimentConfig, seq_len: int, corpus: str) -> torch.Tensor:
    """Blocos ``[N, seq_len]`` do train, na ordem do loader, sem embaralhar aqui.

    O embaralhamento da época fica em ``sequence_order``. O resto que não
    fecha ``seq_len`` cai fora. WikiText-2 cabe inteiro; TinyStories no Hub
    materializa o split (é o caminho de dev, não a corrida de relatório).
    """

    rows: list[torch.Tensor] = []
    stream = iter_split(
        config,
        "train",
        name=corpus,
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
        die(EXIT_USAGE, f"{corpus} train não fechou sequência de seq_len={seq_len}")
    return torch.cat(rows, dim=0)


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
    corpus: str,
) -> dict[str, Any]:
    return {
        "event": "step",
        "model": MODEL_KIND,
        "corpus": corpus,
        "step": cursor["steps"],
        "epoch": cursor["epoch"],
        "epochs_completed": cursor["epochs_completed"],
        "sequence_cursor": cursor["sequence_cursor"],
        "batch_rows": batch_rows,
        "tokens": cursor["tokens"],
        "seq_len": seq_len,
        "loss": "L_task",
        "total": result["total"],
        "task": result["task"],
        "c_mem": None,
        "balance": None,
        "grad_norm": result["grad_norm"],
        "finite": True,
    }


def _architecture(config: ExperimentConfig) -> dict[str, Any]:
    report = capacity_report(config)
    model = config.model
    return {
        "kind": MODEL_KIND,
        "memory_path": False,
        "d_model": model.d_model,
        "n_layers": model.n_layers,
        "n_heads": model.n_heads,
        "d_ff": report["d_ff"],
        "k_reference": model.k,
        "expert_hidden_reference": model.expert_hidden,
        "max_seq_len": model.max_seq_len,
        "vocab_size": model.vocab_size,
        "tie_embeddings": model.tie_embeddings,
    }


def _write_checkpoint(state: dict[str, Any]) -> None:
    path: Path = state["checkpoint_path"]
    payload = {
        "version": CHECKPOINT_VERSION,
        "model_kind": MODEL_KIND,
        "corpus": state["corpus"],
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
        "torch_rng": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng"] = torch.cuda.get_rng_state_all()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _restore(
    path: Path,
    config: ExperimentConfig,
    model,
    optimizer,
    *,
    corpus: str,
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
    if blob.get("model_kind") != MODEL_KIND:
        die(
            EXIT_USAGE,
            "checkpoint não é do LM denso. O MoE, inclusive com λ=0, não retoma neste script.",
        )
    if blob.get("corpus") != corpus:
        die(
            EXIT_USAGE,
            f"checkpoint é corpus={blob.get('corpus')!r}, este comando é {corpus}. "
            "Use outro --work-dir.",
        )
    if int(blob["seed"]) != int(seed) or int(blob["seq_len"]) != int(seq_len):
        die(EXIT_USAGE, "checkpoint não bate com --seed/--seq-len. Use outro --work-dir.")
    if not math.isclose(float(blob["lr"]), float(lr), rel_tol=0.0, abs_tol=1e-12):
        die(EXIT_USAGE, "checkpoint não bate com smoke.lr. Use outro --work-dir.")
    if blob["architecture"] != _architecture(config):
        die(
            EXIT_USAGE,
            "checkpoint não bate com a arquitetura densa derivada do YAML. "
            "Use outro --work-dir.",
        )
    model.load_state_dict(blob["model"])
    optimizer.load_state_dict(blob["optimizer"])
    _move_optimizer_to_parameters(optimizer)
    torch.set_rng_state(blob["torch_rng"])
    if torch.cuda.is_available() and blob.get("cuda_rng"):
        torch.cuda.set_rng_state_all(blob["cuda_rng"])
    return {"cursor": dict(blob["cursor"]), "batch_size": int(blob["batch_size"])}


def _finish(state: dict[str, Any], args: argparse.Namespace, log_path: Path) -> None:
    cursor = state["cursor"]
    last = _last_step(log_path)
    capacity = state["capacity"]
    done = {
        "model": MODEL_KIND,
        "memory_path": False,
        "note": NOTE,
        "corpus": state["corpus"],
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
        "status": "done",
        "loss": "L_task",
        "lambda": None,
        "last_total": None if last is None else last.get("total"),
        "last_task": None if last is None else last.get("task"),
        "last_grad_norm": None if last is None else last.get("grad_norm"),
        "parameters": capacity["dense_parameters"],
        "d_ff": capacity["d_ff"],
        "moe_total_parameters": capacity["moe_total_parameters"],
        "moe_active_parameters": capacity["moe_active_parameters"],
    }
    path = state["checkpoint_path"].parent / "done.json"
    path.write_text(json.dumps(done, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"fim: passos={cursor['steps']}  épocas_fechadas={cursor['epochs_completed']}  "
        f"tokens={cursor['tokens']}  L_task={done['last_task']}  corpus={state['corpus']}"
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


def _install_signals() -> None:
    def handler(signum: int, _frame) -> None:
        _STOP["now"] = True
        print(
            f"sinal {signum}: o passo em curso termina e o checkpoint é gravado",
            file=sys.stderr,
        )

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


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
