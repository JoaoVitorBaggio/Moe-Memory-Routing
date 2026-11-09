"""Decode batch 1 com o roteador sem treino. O JSON vai para ``outputs/``.

O dry-run monta um Qwen3 minúsculo e não toca no checkpoint de 61 GB.
No modo real, o pool morno padrão pede ~12 GiB. A VRAM da 1060 não
segura a rede residente, os 384 experts e uma folga grande ao mesmo
tempo: ``hot_slack`` 0 é o padrão, e o script recusa se a conta passar
de ``vram_budget_bytes``.

``warm_start`` (padrão) carrega ``k1`` experts por camada antes do
primeiro token, nos dois modos, fora do ``b2``. Assim
``experts_executed_per_token`` compara a mesma carga. ``warm_start``
falso é o morno vazio.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from moe_memory.lstm_ahead_router import TwoHorizonLstmRouter
from moe_memory.offload.runtime import OffloadRuntime
from moe_memory.offload.space import (
    REFERENCE_CHECKPOINT_BYTES,
    REFERENCE_FREE_BYTES,
    default_warm_slots,
)
from moe_memory.offload.store import ExpertStore
from moe_memory.offload.synthetic import write_synthetic_bundle

DISK_NOTE = (
    "O checkpoint em bf16 fica em torno de 61 GB e, na máquina de 07/11/2026, "
    "sobram cerca de 44 GB. O repack sem --delete-source-after-each-shard não cabe. "
    "A flag apaga cada shard depois do fsync e é destrutiva."
)


def run_bench(
    out_path: Path,
    *,
    store_dir: Path | None = None,
    dry_run: bool = False,
    lookahead: bool = True,
    tokens: int = 4,
    seed: int = 0,
    l1: int = 1,
    l2: int = 2,
    k1: int = 1,
    k2: int | None = None,
    b2: int = 1,
    warm_slots: int | None = None,
    hot_slack: int = 0,
    n_warm_threads: int = 1,
    device: str = "cpu",
    vram_budget_bytes: int | None = None,
    router_dim: int = 32,
    warm_start: bool = True,
) -> dict:
    if tokens < 1:
        raise ValueError("tokens precisa ser >= 1")
    if l2 < l1 or l1 < 1:
        raise ValueError("l2 >= l1 >= 1")
    if dry_run and store_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="offload-dry-")
        store_path = Path(tmp.name)
        write_synthetic_bundle(store_path, seed=seed)
        owns_tmp = tmp
    else:
        if store_dir is None:
            raise ValueError("store_dir é obrigatório fora do dry-run")
        store_path = Path(store_dir)
        owns_tmp = None
    try:
        return _run(
            out_path,
            store_path,
            dry_run=dry_run,
            lookahead=lookahead,
            tokens=tokens,
            seed=seed,
            l1=l1,
            l2=l2,
            k1=k1,
            k2=k2,
            b2=b2,
            warm_slots=warm_slots,
            hot_slack=hot_slack,
            n_warm_threads=n_warm_threads,
            device=device,
            vram_budget_bytes=vram_budget_bytes,
            router_dim=router_dim,
            warm_start=warm_start,
        )
    finally:
        if owns_tmp is not None:
            owns_tmp.cleanup()


def _run(
    out_path: Path,
    store_dir: Path,
    *,
    dry_run: bool,
    lookahead: bool,
    tokens: int,
    seed: int,
    l1: int,
    l2: int,
    k1: int,
    k2: int | None,
    b2: int,
    warm_slots: int | None,
    hot_slack: int,
    n_warm_threads: int,
    device: str,
    vram_budget_bytes: int | None,
    router_dim: int,
    warm_start: bool,
) -> dict:
    preview = ExpertStore.open(store_dir / "experts.moestore", direct=False)
    try:
        header = preview.header
        block = header.block_nbytes
        n_layers = header.n_layers
        n_experts = header.n_experts
        dtype_code = header.dtype_code
    finally:
        preview.close()
    spec_vocab = json.loads((store_dir / "config.json").read_text(encoding="utf-8"))["vocab_size"]
    if warm_slots is not None:
        slots = warm_slots
    elif dry_run:
        # O orçamento de ~12 GiB é o da máquina real. O dry-run não aloca isso.
        slots = n_layers * n_experts
    else:
        slots = default_warm_slots(block)
    if k2 is None:
        slack = b2 * l2
        room = slots - slack
        if room < n_layers:
            raise ValueError(
                f"pool morno de {slots} vagas não cobre {n_layers} camadas com folga de prefetch {slack}"
            )
        k2 = max(k1, min(n_experts, room // n_layers))
    if not 1 <= k1 <= k2 <= n_experts:
        raise ValueError(f"precisa 1 <= k1={k1} <= k2={k2} <= experts={n_experts}")
    if k2 * n_layers > slots:
        raise ValueError(f"k2×camadas={k2 * n_layers} passa das {slots} vagas mornas")
    hot_slots = n_layers * k1 + hot_slack
    dev = _device(device)
    resident_file = (store_dir / "resident.bin").stat().st_size
    vram_estimate = resident_file + hot_slots * block
    vram_warning = ""
    if vram_estimate > int(5.5e9):
        vram_warning = (
            f"VRAM estimada {vram_estimate} bytes: residentes {resident_file} mais "
            f"{hot_slots} vagas × {block} bytes. Passa de 5,5 GB. Na GTX 1060 a rede "
            "residente (~3 GB) e os 384 experts (~3,6 GB) não cabem juntos nos 6 GB; "
            "o processo pode morrer por falta de memória. --hot-slack 0 já não reserva folga."
        )
    if vram_budget_bytes is not None and vram_estimate > vram_budget_bytes:
        raise ValueError(vram_warning or f"VRAM estimada {vram_estimate} passa de {vram_budget_bytes}")
    router = TwoHorizonLstmRouter(
        vocab_size=int(spec_vocab),
        n_layers=n_layers,
        n_experts=n_experts,
        l1=l1,
        l2=l2,
        k1=k1,
        k2=k2,
        b2=b2,
        embed_dim=router_dim,
        hidden_size=router_dim,
    )
    generator = torch.Generator().manual_seed(seed)
    token_ids = torch.randint(0, int(spec_vocab), (tokens,), generator=generator).tolist()
    with OffloadRuntime(
        store_dir,
        router,
        lookahead=lookahead,
        warm_slots=slots,
        hot_slack=hot_slack,
        device=dev,
        n_warm_threads=n_warm_threads,
        direct=not dry_run,
        warm_start=warm_start,
    ) as runtime:
        stats = runtime.decode(token_ids)
        direct = runtime.store.direct
    payload = stats.as_dict()
    payload.update(
        {
            "mode": "lookahead" if lookahead else "on-demand",
            "warm_start": warm_start,
            "dry_run": dry_run,
            "device": str(dev),
            "direct_io": direct,
            "l1": l1,
            "l2": l2,
            "k1": k1,
            "k2": k2,
            "b2": b2,
            "warm_slots": slots,
            "hot_slots": hot_slots,
            "hot_slack": hot_slack,
            "block_nbytes": block,
            "n_layers": n_layers,
            "n_experts": n_experts,
            "dtype_code": dtype_code,
            "vram_estimate_bytes": vram_estimate,
            "vram_warning": vram_warning,
            "disk_note": DISK_NOTE,
            "reference_checkpoint_bytes": REFERENCE_CHECKPOINT_BYTES,
            "reference_free_bytes": REFERENCE_FREE_BYTES,
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA pedido e não há GPU")
        return torch.device("cuda")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError("device é cpu, cuda ou auto")
