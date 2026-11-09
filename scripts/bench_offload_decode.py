#!/usr/bin/env python3
"""Decode batch 1 do Qwen3 com experts no store. Roteador sem treino serve.

Grava JSON em ``outputs/`` com tokens/s, stall por nível, bytes lidos do
SSD, pico de RAM e de VRAM, misses, ``experts_executed`` por token,
``warm_start_s`` e ``warm_start_bytes``. ``--mode on-demand`` não adianta
a cópia; ``lookahead`` usa L2 no SSD e L1 na VRAM. Os dois modos, por
padrão, carregam o conjunto inicial (``k1`` experts por camada) antes
do primeiro token, fora do ``b2``. ``--no-warm-start`` deixa o morno vazio.

``--profile`` acrescenta tempos por token e somas de SSD/H2D/GPU. Não
sincroniza o device por token. ``--trace`` grava a identidade da
execução. ``--paired`` roda on-demand e depois lookahead com o mesmo
peso, os mesmos tokens e os mesmos tetos, e falha se a execução divergir.
Não é replay: os dois modos usam o runtime. O prompt não entra; o KV
começa no primeiro token decodificado.

Uma corrida por vez, com MemoryMax=7G e MemorySwapMax=0. Não lance dois
processos juntos. O ``--out`` do pareado é um arquivo novo: não reescreve
os JSON antigos de um modo só.

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

Pareado, 128 tokens, perfil ligado, arquivo novo (repetir com outro ``--out``)::

    python3 scripts/bench_offload_decode.py \\
        --store-dir models/qwen3-30b-a3b-store \\
        --paired --profile --trace \\
        --tokens 128 --seed 0 \\
        --k1 4 --k2 6 --b2 5 --l1 1 --l2 3 \\
        --warm-slots 320 --hot-slack 0 --warm-threads 2 \\
        --vram-budget-gb 5.5 --device cuda \\
        --out outputs/paired-128-seed0-a.json

    python3 scripts/bench_offload_decode.py \\
        --store-dir models/qwen3-30b-a3b-store \\
        --paired --profile --trace \\
        --tokens 128 --seed 0 \\
        --k1 4 --k2 6 --b2 5 --l1 1 --l2 3 \\
        --warm-slots 320 --hot-slack 0 --warm-threads 2 \\
        --vram-budget-gb 5.5 --device cuda \\
        --out outputs/paired-128-seed0-b.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moe_memory.offload.bench import DISK_NOTE, MEMORY_RUN_NOTE, run_bench, run_paired  # noqa: E402


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
    parser.add_argument(
        "--no-warm-start",
        action="store_true",
        help="morno vazio: o b2 enche o conjunto durante o decode. O padrão carrega k1 experts por camada antes do primeiro token.",
    )
    parser.add_argument("--profile", action="store_true", help="tempos por token e somas de SSD/H2D/GPU; não muda a rota")
    parser.add_argument("--trace", action="store_true", help="grava a identidade da execução (ids, peso mole, prefixo de host)")
    parser.add_argument(
        "--paired",
        action="store_true",
        help="on-demand e depois lookahead, mesmos pesos e tokens; falha se a execução divergir",
    )
    parser.add_argument("--profile-max-tokens", type=int, default=512)
    args = parser.parse_args()
    print(DISK_NOTE)
    if args.paired and args.mode != "lookahead":
        raise SystemExit("--paired ignora --mode e roda os dois; não passe --mode on-demand junto")
    if args.paired or args.profile:
        print(MEMORY_RUN_NOTE)
    if args.dry_run and args.k1 == 8 and args.k2 is None and args.b2 == 5 and args.l2 == 3:
        # O default real (k1=8, b2=5, l2=3) não cabe no bundle sintético de 4 experts.
        args.k1 = 1
        args.b2 = 1
        args.l2 = 2
        args.device = "cpu"
        args.warm_threads = 1
    common = dict(
        store_dir=args.store_dir,
        dry_run=args.dry_run,
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
        warm_start=not args.no_warm_start,
        profile=args.profile,
        trace=args.trace or args.paired,
        profile_max_tokens=args.profile_max_tokens,
    )
    try:
        if args.paired:
            payload = run_paired(args.out, **common)
        else:
            payload = run_bench(args.out, lookahead=args.mode == "lookahead", **common)
    except (ValueError, FileNotFoundError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
    if payload.get("vram_warning"):
        print(payload["vram_warning"])
    if payload.get("kind") == "paired":
        print(
            f"paired match={payload['execution_match']} fingerprint={payload['fingerprint']} "
            f"json={args.out}"
        )
        for name, run in payload["runs"].items():
            print(
                f"mode={name} tokens/s={run['tokens_per_s']:.3f} "
                f"experts_executed={run['experts_executed']} "
                f"warm_start_s={run['warm_start_s']:.4f} ssd_bytes={run['ssd_bytes']}"
            )
        return
    print(
        f"mode={payload['mode']} warm_start={payload['warm_start']} "
        f"tokens/s={payload['tokens_per_s']:.3f} "
        f"experts_executed={payload['experts_executed']} "
        f"warm_start_s={payload['warm_start_s']:.4f} warm_start_bytes={payload['warm_start_bytes']} "
        f"stall_warm_s={payload['stall_warm_s']:.4f} stall_hot_s={payload['stall_hot_s']:.4f} "
        f"ssd_bytes={payload['ssd_bytes']} warm_misses={payload['warm_misses']} "
        f"hot_misses={payload['hot_misses']} json={args.out}"
    )


if __name__ == "__main__":
    main()
