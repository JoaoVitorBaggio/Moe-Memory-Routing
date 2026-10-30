#!/usr/bin/env python3
"""Um passo curto de TinyStories. Não é a corrida de entrega.

O batch sai do loader (BPE do GPT-2), não de ids uniformes. O tamanho
padrão é o ``smoke`` do YAML (batch 2, seq 32), para o passo caber numa
checagem. A corrida overnight sobe o contexto até ``model.max_seq_len``
e troca o corpus para WikiText-2; isso não é este script.

A loss impressa é de pouquíssimos tokens em peso inicial. Não entra no
relatório.

Uso, a partir da raiz::

    python3 experiments/train_tinystories.py --steps 1
    python3 experiments/train_tinystories.py --steps 1 --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from moe_memory.config import load_config
from moe_memory.corpus import iter_split, source_label
from moe_memory.smoke import prepare_train, train_one_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Passo curto de TinyStories (fp32).")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seq-len", type=int, default=None, help="Padrão: smoke.seq_len.")
    parser.add_argument("--batch-size", type=int, default=None, help="Padrão: smoke.batch_size.")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=ROOT / "outputs" / "tinystories_cold",
        help="Memmap dos experts frios. outputs/ está no .gitignore.",
    )
    args = parser.parse_args()
    if args.steps < 1:
        raise SystemExit("--steps precisa ser >= 1")
    config = load_config(args.config)
    seq_len = config.smoke.seq_len if args.seq_len is None else args.seq_len
    batch_size = config.smoke.batch_size if args.batch_size is None else args.batch_size
    if not 2 <= seq_len <= config.model.max_seq_len:
        raise SystemExit(
            f"seq_len={seq_len} fora de 2..max_seq_len={config.model.max_seq_len}"
        )
    if batch_size < 1:
        raise SystemExit("batch_size precisa ser >= 1")
    lambda_weight = config.smoke.lambda_weight if args.lambda_ is None else args.lambda_
    print("Passo curto de TinyStories (não é resultado experimental)")
    print(f"fonte: {source_label(config, 'TinyStories')}")
    print(f"passos={args.steps}  batch={batch_size}  seq={seq_len}  device={args.device}")
    try:
        batches = iter_split(
            config,
            "train",
            name="TinyStories",
            seq_len=seq_len,
            batch_size=batch_size,
            max_batches=args.steps,
            streaming=True,
        )
        try:
            first = next(batches, None)
            if first is None:
                raise SystemExit(
                    "o loader não fechou um batch; texto curto demais ou corpus vazio"
                )
            device = torch.device(args.device)
            model, emulator, optimizer, residency = prepare_train(
                config,
                lambda_=lambda_weight,
                work_dir=args.work_dir,
                device=device,
            )
            seen = 0
            result = None
            for batch in _chain(first, batches):
                seen += 1
                result = train_one_batch(
                    model,
                    emulator,
                    optimizer,
                    config,
                    batch.to(device),
                    lambda_weight=lambda_weight,
                    residency=residency,
                )
                print(
                    f"passo {seen}: L={result['total']:.4f}  L_task={result['task']:.4f}  "
                    f"C_mem={result['c_mem']:.4f}  |grad|={result['grad_norm']:.4f}  "
                    f"trocas={result['swaps']}"
                )
        finally:
            batches.close()
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(
            "Não consegui ler o TinyStories. Confira a rede, corpus.cache_dir "
            f"e corpus.paths.TinyStories. Detalhe: {exc}"
        ) from exc
    assert result is not None
    if seen < args.steps:
        print(f"o corpus fechou {seen} batch(es); --steps pedia {args.steps}")
    print(f"parâmetros: {result['parameters']:,}  λ={result['lambda']}  β={result['beta']}")
    if result["physical_offload"]:
        print(
            f"residência física: subidas={result['staged_loads']}  "
            f"bytes_staged={result['staged_bytes']}  "
            "(cópia de compute; não entra no contador do LRU)"
        )
    else:
        print("residência física: desligada neste device. Na 1060 use --device cuda.")


def _chain(first: torch.Tensor, rest):
    yield first
    yield from rest


if __name__ == "__main__":
    main()
