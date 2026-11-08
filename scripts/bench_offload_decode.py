#!/usr/bin/env python3
"""Decode batch 1 do Qwen3 com experts no store. Roteador sem treino serve.

Grava JSON em ``outputs/`` com tokens/s, stall por nível, bytes lidos do
SSD, pico de RAM e de VRAM e misses. ``--mode on-demand`` não adianta a
cópia; ``lookahead`` usa L2 no SSD e L1 na VRAM.

``--dry-run`` monta um modelo minúsculo e não lê ``models/qwen3-30b-a3b``.

O pool morno padrão reserva ~12 GiB. Na 1060, residentes (~3 GB) mais os
384 experts quentes (~3,6 GB) já encostam nos 6 GB: ``--hot-slack 0`` é
o padrão. Subir a folga pode não caber; o script recusa se passar de
``--vram-budget-gb``.

Exemplo::

    python3 scripts/bench_offload_decode.py --dry-run --tokens 4 \\
        --out outputs/offload-dry.json

    python3 scripts/bench_offload_decode.py \\
        --store-dir models/qwen3-30b-a3b-store \\
        --mode lookahead --tokens 32 \\
        --out outputs/offload-lookahead.json

    python3 scripts/bench_offload_decode.py \\
        --store-dir models/qwen3-30b-a3b-store \\
        --mode on-demand --tokens 32 \\
        --out outputs/offload-ondemand.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moe_memory.offload.bench import DISK_NOTE, run_bench  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--store-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mode", choices=("lookahead", "on-demand"), default="lookahead")
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--l1", type=int, default=1)
    parser.add_argument("--l2", type=int, default=3)
    parser.add_argument("--k1", type=int, default=8)
    parser.add_argument("--k2", type=int, default=None)
    parser.add_argument("--b2", type=int, default=5)
    parser.add_argument("--warm-slots", type=int, default=None, help="padrão: orçamento de ~12 GiB")
    parser.add_argument("--hot-slack", type=int, default=0, help="vagas de VRAM além de k1 por camada")
    parser.add_argument("--warm-threads", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument(
        "--vram-budget-gb",
        type=float,
        default=None,
        help="se a conta de residentes + vagas quentes passar disto, não aloca. Sem a flag, só avisa.",
    )
    parser.add_argument("--router-dim", type=int, default=64)
    args = parser.parse_args()
    print(DISK_NOTE)
    if args.dry_run and args.k1 == 8 and args.k2 is None and args.b2 == 5 and args.l2 == 3:
        # O default real (k1=8, b2=5, l2=3) não cabe no bundle sintético de 4 experts.
        args.k1 = 1
        args.b2 = 1
        args.l2 = 2
        args.device = "cpu"
        args.warm_threads = 1
    try:
        payload = run_bench(
            args.out,
            store_dir=args.store_dir,
            dry_run=args.dry_run,
            lookahead=args.mode == "lookahead",
            tokens=args.tokens,
            seed=args.seed,
            l1=args.l1,
            l2=args.l2,
            k1=args.k1,
            k2=args.k2,
            b2=args.b2,
            warm_slots=args.warm_slots,
            hot_slack=args.hot_slack,
            n_warm_threads=args.warm_threads,
            device=args.device,
            vram_budget_bytes=None if args.vram_budget_gb is None else int(args.vram_budget_gb * 1e9),
            router_dim=args.router_dim,
        )
    except (ValueError, FileNotFoundError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
    if payload.get("vram_warning"):
        print(payload["vram_warning"])
    print(
        f"mode={payload['mode']} tokens/s={payload['tokens_per_s']:.3f} "
        f"stall_warm_s={payload['stall_warm_s']:.4f} stall_hot_s={payload['stall_hot_s']:.4f} "
        f"ssd_bytes={payload['ssd_bytes']} warm_misses={payload['warm_misses']} "
        f"hot_misses={payload['hot_misses']} json={args.out}"
    )


if __name__ == "__main__":
    main()
