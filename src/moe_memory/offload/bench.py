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
from moe_memory.offload.phase import emit_phase
from moe_memory.offload.profile import DecodeProfile
from moe_memory.offload.runtime import OffloadRuntime
from moe_memory.offload.trace import ExecutionTrace, TraceDivergence, compare_traces, module_sha, store_sha
from moe_memory.offload.space import (
    REFERENCE_CHECKPOINT_BYTES,
    REFERENCE_FREE_BYTES,
    default_warm_slots,
)
from moe_memory.offload.store import ExpertStore
from moe_memory.offload.synthetic import write_synthetic_bundle

MEMORY_ACCOUNTING = {
    "ru_maxrss_bytes": "Pico de RSS do processo (getrusage ru_maxrss). Não é o uso neste instante.",
    "vm_rss_bytes": "RSS atual, VmRSS de /proc/self/status. Inclui página de arquivo residente no processo.",
    "vm_hwm_bytes": "Pico de RSS, VmHWM.",
    "vm_pin_bytes": (
        "VmPin do kernel (mlock). A memória pinned do CUDA pode não aparecer aqui; "
        "ela entra no RSS e, em geral, no memory.current."
    ),
    "cgroup_memory_current_bytes": (
        "memory.current do cgroup v2, se o arquivo existir. Inclui cache de página "
        "e memória de kernel cobrados do cgroup. Não é igual ao RSS."
    ),
    "tracked_pool_bytes": (
        "Soma dos SlotPool cujo release ainda não rodou. Não inclui o residente, "
        "o cache de página nem bloco de GPU fora desses pools."
    ),
    "ram_peak_bytes": (
        "max(ru_maxrss no fim do decode, bytes lógicos do pool morno + residente). "
        "O residente na GPU entra nessa soma mesmo sem estar no RSS do host."
    ),
}

MEMORY_RUN_NOTE = (
    "Uma corrida por vez. No PC, MemoryMax=7G e MemorySwapMax=0. "
    "Não dispare lookahead e on-demand em paralelo. O modo pareado roda "
    "os dois em sequência, no mesmo processo, com os mesmos pesos."
)

CANNOT_PROVE = (
    "O perfil não aponta a causa da diferença de tok/s. "
    "elapsed_s inclui warm-start e o enqueue do prefetch e exclui load_resident. "
    "Stall não é o read do worker. Somas de SSD e de events CUDA não são parcelas da parede. "
    "Corridas simples não semeiam o LSTM: o mesmo --seed não repete o peso entre processos. "
    "O pareado repete a execução; bytes copiados ainda podem diferir por prefetch e despejo.",
)

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
    profile: bool = False,
    trace: bool = False,
    profile_max_tokens: int = 512,
    seed_router: bool = False,
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
            profile=profile,
            trace=trace,
            profile_max_tokens=profile_max_tokens,
            seed_router=seed_router,
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
    profile: bool = False,
    trace: bool = False,
    profile_max_tokens: int = 512,
    seed_router: bool = False,
    router_state: dict | None = None,
    write_json: bool = True,
    phases: list | None = None,
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
    label = "lookahead" if lookahead else "on-demand"
    run_phases: list[dict] = []

    def mark(name: str) -> None:
        run_phases.append(emit_phase(f"{label}-{name}"))
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
    if trace and tokens > profile_max_tokens:
        raise ValueError(
            f"traço de {tokens} tokens passa do teto {profile_max_tokens}; "
            "suba --profile-max-tokens. Traço truncado não certifica."
        )
    if seed_router:
        torch.manual_seed(seed)
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
    if router_state is not None:
        router.load_state_dict(router_state)
        router_init = "state_dict"
    elif seed_router:
        router_init = "manual_seed"
    else:
        router_init = "global_rng"
    mark("router-ready")
    limits = {
        "k1": k1,
        "k2": k2,
        "b2": b2,
        "l1": l1,
        "l2": l2,
        "warm_slots": slots,
        "hot_slots": hot_slots,
        "warm_start": warm_start,
        "n_layers": n_layers,
        "n_experts": n_experts,
    }
    generator = torch.Generator().manual_seed(seed)
    token_ids = torch.randint(0, int(spec_vocab), (tokens,), generator=generator).tolist()
    execution = None
    if trace:
        execution = ExecutionTrace(
            limits=limits,
            tokens=token_ids,
            router_sha=module_sha(router),
            store_sha256=store_sha(store_dir / "experts.moestore"),
            max_tokens=profile_max_tokens,
        )
    observer = DecodeProfile(device=dev, max_tokens=profile_max_tokens) if profile else None
    mark("trace-ready" if execution is not None else "trace-skipped")
    mark("before-runtime")
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
        profile=observer,
        trace=execution,
        phase=lambda step: mark(step),
    ) as runtime:
        stats = runtime.decode(token_ids)
        direct = runtime.store.direct
    mark("after-close")
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
            "prompt_ids": None,
            "attention_cache": (
                "começa no primeiro token decodificado; o bench não faz prefill. "
                "Sem prompt, o conjunto inicial usa o token 0 só no roteador."
            ),
            "router_init": router_init,
            "profile": observer.report() if observer is not None else None,
            "trace": execution.public() if execution is not None else None,
            "memory_phases": run_phases,
            "memory_accounting": MEMORY_ACCOUNTING,
        }
    )
    if phases is not None:
        phases.extend(run_phases)
    if execution is not None:
        payload["_trace"] = execution
    if seed_router or router_state is not None:
        payload["_router_state"] = {name: tensor.detach().cpu().clone() for name, tensor in router.state_dict().items()}
    if write_json:
        _write_public(out_path, payload)
    return payload


def run_paired(
    out_path: Path,
    **kwargs,
) -> dict:
    """Roda on-demand e depois lookahead com o mesmo peso, token e teto.

    Não é replay: cada modo usa o runtime de verdade. Se a execução divergir,
    o JSON ainda é gravado e a função levanta ``TraceDivergence``.
    """

    tokens = int(kwargs.get("tokens", 4))
    max_tokens = int(kwargs.get("profile_max_tokens", 512))
    if tokens > max_tokens:
        raise ValueError(
            f"pareado de {tokens} tokens passa do teto {max_tokens}; "
            "suba profile_max_tokens. A comparação não certifica traço truncado."
        )
    dry_run = bool(kwargs.get("dry_run", False))
    store_dir = kwargs.get("store_dir")
    seed = int(kwargs.get("seed", 0))
    if dry_run and store_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="offload-paired-")
        store_path = Path(tmp.name)
        write_synthetic_bundle(store_path, seed=seed)
        owns_tmp = tmp
    else:
        if store_dir is None:
            raise ValueError("store_dir é obrigatório fora do dry-run")
        store_path = Path(store_dir)
        owns_tmp = None
    shared = {
        "tokens": 4,
        "seed": seed,
        "l1": 1,
        "l2": 2,
        "k1": 1,
        "k2": None,
        "b2": 1,
        "warm_slots": None,
        "hot_slack": 0,
        "n_warm_threads": 1,
        "device": "cpu",
        "vram_budget_bytes": None,
        "router_dim": 32,
        "warm_start": True,
        "profile": False,
        "profile_max_tokens": max_tokens,
    }
    for key, value in kwargs.items():
        if key in shared:
            shared[key] = value
    shared["trace"] = True
    boundary: list[dict] = []

    def mark(name: str) -> None:
        boundary.append(emit_phase(name))

    try:
        mark("paired-start")
        demand = _run(
            out_path,
            store_path,
            **_run_kwargs(shared),
            dry_run=dry_run,
            lookahead=False,
            write_json=False,
            seed_router=True,
        )
        mark("after-on-demand")
        mark("before-lookahead")
        ahead = _run(
            out_path,
            store_path,
            **_run_kwargs(shared),
            dry_run=dry_run,
            lookahead=True,
            write_json=False,
            router_state=demand["_router_state"],
            seed_router=False,
        )
        mark("after-lookahead")
        left = demand.pop("_trace")
        right = ahead.pop("_trace")
        demand.pop("_router_state", None)
        ahead.pop("_router_state", None)
        try:
            fingerprint = compare_traces(left, right)
            match = True
            detail = None
            message = None
        except TraceDivergence as exc:
            fingerprint = None
            match = False
            detail = exc.detail
            message = str(exc)
        payload = {
            "kind": "paired",
            "execution_match": match,
            "fingerprint": fingerprint,
            "divergence": detail,
            "divergence_message": message,
            "seed": seed,
            "router_seeded": True,
            "order": ["on-demand", "lookahead"],
            "sequential": True,
            "memory_run_note": MEMORY_RUN_NOTE,
            "measurement_host": (
                "synthetic-dry-run"
                if dry_run
                else "processo local; não é medição da GTX 1060"
            ),
            "cannot_prove": CANNOT_PROVE,
            "memory_phases": boundary,
            "memory_accounting": MEMORY_ACCOUNTING,
            "pool_overlap": _pool_overlap(boundary),
            "runs": {"on-demand": demand, "lookahead": ahead},
        }
        mark("before-json")
        _write_public(out_path, payload)
        mark("after-json")
        _write_public(out_path, payload)
        if not match:
            raise TraceDivergence(message or "execução divergiu", detail or {})
        return payload
    finally:
        if owns_tmp is not None:
            owns_tmp.cleanup()


def _pool_overlap(phases: list[dict]) -> dict:
    by_name = {str(row["phase"]): row for row in phases}
    start = by_name.get("paired-start")
    after = by_name.get("after-on-demand")
    before = by_name.get("before-lookahead")
    if start is None or after is None or before is None:
        return {"known": False}
    base = int(start["tracked_pool_bytes"])
    left = int(after["tracked_pool_bytes"])
    right = int(before["tracked_pool_bytes"])
    retained = left - base
    return {
        "known": True,
        "tracked_pool_bytes_at_start": base,
        "tracked_pool_bytes_after_on_demand": left,
        "tracked_pool_bytes_before_lookahead": right,
        "on_demand_pool_bytes_still_live": retained,
        "second_pool_starts_with_first_still_live": retained > 0 or right > base,
    }


def _run_kwargs(shared: dict) -> dict:
    allowed = {
        "lookahead",
        "tokens",
        "seed",
        "l1",
        "l2",
        "k1",
        "k2",
        "b2",
        "warm_slots",
        "hot_slack",
        "n_warm_threads",
        "device",
        "vram_budget_bytes",
        "router_dim",
        "warm_start",
        "profile",
        "trace",
        "profile_max_tokens",
    }
    return {key: value for key, value in shared.items() if key in allowed}


def _write_public(path: Path, payload: dict) -> None:
    public = {key: value for key, value in payload.items() if not str(key).startswith("_")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(public, indent=2), encoding="utf-8")


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
