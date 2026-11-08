#!/usr/bin/env python3
"""Repack do Qwen3-30B-A3B para um store de experts alinhado a 4 KiB.

Não carrega o modelo. Lê um tensor por vez, limita a escrita e dá fsync
em pedaços: um download de ~61 GB sem teto travou o PC por page cache suja.

Na máquina de 07/11/2026 o checkpoint tem cerca de 61 GB e sobram cerca
de 44 GB. Sem ``--delete-source-after-each-shard`` a saída não cabe
junto da fonte. A flag apaga cada shard depois do fsync. É destrutiva:
um crash no meio perde o shard já apagado (os bytes dele já foram
sincronizados). O padrão é não apagar, e nesse caso o script sai com
código 2 sem escrever o store.

Exemplo, a partir da raiz do repositório::

    python3 scripts/repack_qwen3_experts.py \\
        --model-dir models/qwen3-30b-a3b \\
        --out-dir models/qwen3-30b-a3b-store \\
        --max-mb-s 80 \\
        --fsync-mib 64 \\
        --delete-source-after-each-shard
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moe_memory.offload.repack import repack_model  # noqa: E402
from moe_memory.offload.space import SpaceError  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--delete-source-after-each-shard",
        action="store_true",
        help="apaga cada shard depois do fsync. Sem isto, fonte e saída precisam caber juntas.",
    )
    parser.add_argument(
        "--max-mb-s",
        type=float,
        default=80.0,
        help="teto de escrita em MB/s. 0 desliga o teto e mantém o fsync. O padrão é baixo de propósito.",
    )
    parser.add_argument("--fsync-mib", type=int, default=64, help="fsync a cada N MiB escritos")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.fsync_mib < 1:
        raise SystemExit("--fsync-mib precisa ser >= 1")
    if args.max_mb_s < 0:
        raise SystemExit("--max-mb-s precisa ser >= 0")
    max_bps = None if args.max_mb_s == 0 else args.max_mb_s * 1e6
    try:
        repack_model(
            args.model_dir,
            args.out_dir,
            delete_source=args.delete_source_after_each_shard,
            max_bytes_per_sec=max_bps,
            fsync_every=args.fsync_mib * 1024 * 1024,
            overwrite=args.overwrite,
        )
    except SpaceError as exc:
        print(exc.report.message, file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
