#!/usr/bin/env python3
"""Cronômetro fino da latência de parede. Não inventa número.

Sem CUDA o script explica o procedimento e sai. Com CUDA, mede o forward
em fp32 no device e imprime p50/p95 locais. Isso não é a coluna do
relatório: o número que entra no trabalho é na GTX 1060, no hold-out do
WikiText-2, gravado por ``run_holdout_eval.py``. A curva de ``B`` usa o
mesmo forward, depois de ``prepare_split_budget``. O batch daqui é o
sintético do smoke. Sem CUDA o hold-out deixa os ms null.

O relógio cobre o forward (a cópia do expert entra). ``emulator.step`` e
``materialize`` ficam fora: o primeiro atualiza o LRU, o segundo devolve
quem não é quente para CPU pinned ou memmap. Bytes/trocas do emulador
não são latência.

Uso na 1060::

    python3 experiments/wall_latency.py --device cuda --batches 8

O procedimento completo está em ``docs/plano-de-hardware.md``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import torch

EXPLANATION = """\
Latência de parede — só na GTX 1060, com o modelo em CUDA, fp32, sem torch.compile.

1. ExpertResidency estaciona o home: quente na VRAM, morno em CPU pinned, frio em memmap.
2. Fora do cronômetro, materialize() alinha o storage a ℓ(e).
3. synchronize, forward em eval (a subida do expert entra no tempo), synchronize.
4. emulator.step atualiza o LRU. materialize() devolve quem está fora de B. Os dois ficam fora do cronômetro.
5. p50/p95 desses forwards. O batch do relatório é o hold-out do WikiText-2.

O emulador segue contando trocas e bytes à parte. Este script não grava results/ e não preenche a tabela.
A coluna do relatório é o JSON do hold-out, na GTX 1060. CPU: wall_clock=unavailable.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Mede o forward com residência física, se houver CUDA.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=ROOT / "outputs" / "wall_cold",
    )
    args = parser.parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        print(EXPLANATION)
        print("medição: não executada")
        return
    if args.batches < 1:
        raise SystemExit("--batches precisa ser >= 1")
    _measure(args)


def _measure(args: argparse.Namespace) -> None:
    from moe_memory.config import load_config
    from moe_memory.smoke import fake_batch, prepare_train

    config = load_config(args.config)
    device = torch.device("cuda")
    model, emulator, _optimizer, residency = prepare_train(
        config, work_dir=args.work_dir, device=device
    )
    model.eval()
    times: list[float] = []
    with torch.no_grad():
        for index in range(args.batches):
            batch = fake_batch(config, seed=index).to(device)
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = model(batch, emulator.levels)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - started)
            emulator.step(output.dispatched)
            residency.materialize()
    ordered = sorted(times)
    print("medição local na CUDA — não entra no relatório")
    print(f"batches={len(ordered)}  seq={config.smoke.seq_len}  batch={config.smoke.batch_size}")
    print(f"p50_ms={_percentile(ordered, 50) * 1e3:.3f}")
    print(f"p95_ms={_percentile(ordered, 95) * 1e3:.3f}")
    print(
        f"subidas={residency.loads}  bytes_staged={residency.bytes_loaded}  "
        f"trocas_lru={emulator.total_swaps}  bytes_lru={emulator.total_bytes}"
    )


def _percentile(ordered: list[float], percent: float) -> float:
    """Mesma conta de ``run_holdout_eval.percentile``."""

    from run_holdout_eval import percentile

    return percentile(ordered, percent)


if __name__ == "__main__":
    main()
