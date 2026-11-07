#!/usr/bin/env python3
"""Cotas de viabilidade do roteador LSTM adiantado. Não inventa banda nem tempo.

Lê só o ``config.json`` do OLMoE (arquivo local ou a URL do config no Hub).
Não baixa peso. ``S_e`` sai das dimensões do expert SwiGLU, sem bias.

Com CUDA, mede a banda de uma cópia host→GPU a partir de memória pinned.
``t_tok`` entra por argumento. Sem o argumento e com CUDA, o proxy é o
tempo de um expert SwiGLU de um token: não é o tempo de token do OLMoE.
Sem CUDA as duas medições são puladas e o processo sai sem erro.

``L_min`` e as cotas de ``K`` só são numéricas quando ``m``, ``BW``,
``t_tok`` e ``M_hot`` existem. O significado das cotas está em
``docs/metodo-router-lstm-adiantado.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch

OLMOE_CONFIG_URL = (
    "https://huggingface.co/allenai/OLMoE-1B-7B-0924/resolve/main/config.json"
)
MAX_CONFIG_BYTES = 2_000_000

# Bytes de armazenamento por parâmetro. Chave desconhecida não vira chute:
# o processo pede --dtype-bytes.
_DTYPE_BYTES = {
    "float32": 4,
    "fp32": 4,
    "torch.float32": 4,
    "float16": 2,
    "fp16": 2,
    "torch.float16": 2,
    "bfloat16": 2,
    "bf16": 2,
    "torch.bfloat16": 2,
    "float64": 8,
    "fp64": 8,
    "torch.float64": 8,
}


@dataclass(frozen=True)
class ExpertSize:
    """Um expert. ``nbytes`` é ``S_e`` e vale para cada camada do config."""

    model_type: str
    hidden_size: int
    intermediate_size: int
    n_layers: int
    n_experts: int
    experts_per_token: int
    dtype_name: str
    dtype_bytes: int
    dtype_bytes_source: str
    parameters: int
    nbytes: int

    @property
    def native_executed_slots(self) -> int:
        """Top-k de cada camada ao mesmo tempo, sem reuso entre camadas."""

        return self.experts_per_token * self.n_layers


@dataclass(frozen=True)
class GpuMeasurement:
    value: float | None
    status: str
    detail: str


def swiglu_expert_parameters(hidden_size: int, intermediate_size: int) -> int:
    """Parâmetros de um expert SwiGLU sem bias.

    ``gate``, ``up`` e ``down`` têm o mesmo total que ``gate_up`` fundido
    mais ``down``: ``3 * hidden * intermediate``. Norma, atenção e o
    roteador linear ficam de fora.
    """

    if hidden_size <= 0 or intermediate_size <= 0:
        raise ValueError("hidden_size e intermediate_size precisam ser > 0")
    return 3 * hidden_size * intermediate_size


def dtype_nbytes(name: str) -> int:
    key = name.strip().lower()
    if key not in _DTYPE_BYTES:
        known = ", ".join(sorted(set(_DTYPE_BYTES)))
        raise ValueError(
            f"torch_dtype {name!r} não tem tamanho conhecido ({known}). "
            "Passe --dtype-bytes. O script não escolhe um tamanho."
        )
    return _DTYPE_BYTES[key]


def expert_size_from_config(
    config: dict,
    *,
    dtype_bytes_override: int | None = None,
) -> ExpertSize:
    """``S_e`` a partir dos campos do config. Não lê tensor de peso."""

    if not isinstance(config, dict):
        raise ValueError("config.json precisa ser um objeto")
    model_type = config.get("model_type")
    if model_type != "olmoe":
        raise ValueError(
            f"model_type {model_type!r} não é olmoe. "
            "A conta do SwiGLU sem bias é a desse modelo; outro config não entra."
        )
    missing = [
        field
        for field in (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_experts",
            "num_experts_per_tok",
        )
        if field not in config
    ]
    if missing:
        raise ValueError("config.json sem campos do expert: " + ", ".join(missing))
    hidden = _positive_int(config["hidden_size"], "hidden_size")
    intermediate = _positive_int(config["intermediate_size"], "intermediate_size")
    n_layers = _positive_int(config["num_hidden_layers"], "num_hidden_layers")
    n_experts = _positive_int(config["num_experts"], "num_experts")
    per_token = _positive_int(config["num_experts_per_tok"], "num_experts_per_tok")
    if per_token > n_experts:
        raise ValueError("num_experts_per_tok passa de num_experts")
    if dtype_bytes_override is None:
        if "torch_dtype" not in config:
            raise ValueError("config.json sem torch_dtype. Passe --dtype-bytes.")
        dtype_name = str(config["torch_dtype"])
        width = dtype_nbytes(dtype_name)
        source = "torch_dtype"
    else:
        if dtype_bytes_override <= 0:
            raise ValueError("--dtype-bytes precisa ser > 0")
        dtype_name = str(config.get("torch_dtype", "override"))
        width = dtype_bytes_override
        source = "argument"
    parameters = swiglu_expert_parameters(hidden, intermediate)
    return ExpertSize(
        model_type=str(model_type),
        hidden_size=hidden,
        intermediate_size=intermediate,
        n_layers=n_layers,
        n_experts=n_experts,
        experts_per_token=per_token,
        dtype_name=dtype_name,
        dtype_bytes=width,
        dtype_bytes_source=source,
        parameters=parameters,
        nbytes=parameters * width,
    )


def lookahead_min(
    m: int,
    expert_bytes: int,
    bandwidth_bytes_per_s: float,
    t_tok_s: float,
) -> int:
    """Menor inteiro ``L`` com ``L >= ceil(m * S_e / (BW * t_tok))``."""

    if m < 0:
        raise ValueError("m precisa ser >= 0")
    if expert_bytes <= 0:
        raise ValueError("S_e precisa ser > 0")
    if not math.isfinite(bandwidth_bytes_per_s) or bandwidth_bytes_per_s <= 0:
        raise ValueError("BW precisa ser finito e > 0")
    if not math.isfinite(t_tok_s) or t_tok_s <= 0:
        raise ValueError("t_tok precisa ser finito e > 0")
    return math.ceil((m * expert_bytes) / (bandwidth_bytes_per_s * t_tok_s))


def memory_slot_sum_max(m_hot_bytes: int, expert_bytes: int) -> int:
    """Maior inteiro ``sum_l K_l`` com ``sum_l K_l * S_e <= M_hot``."""

    if m_hot_bytes < 0:
        raise ValueError("M_hot precisa ser >= 0")
    if expert_bytes <= 0:
        raise ValueError("S_e precisa ser > 0")
    return m_hot_bytes // expert_bytes


def bandwidth_slot_max(
    bandwidth_bytes_per_s: float,
    lookahead: int,
    t_tok_s: float,
    expert_bytes: int,
) -> int:
    """Maior inteiro ``K`` com ``K * S_e <= BW * L * t_tok``."""

    if lookahead < 0:
        raise ValueError("L precisa ser >= 0")
    if expert_bytes <= 0:
        raise ValueError("S_e precisa ser > 0")
    if not math.isfinite(bandwidth_bytes_per_s) or bandwidth_bytes_per_s <= 0:
        raise ValueError("BW precisa ser finito e > 0")
    if not math.isfinite(t_tok_s) or t_tok_s <= 0:
        raise ValueError("t_tok precisa ser finito e > 0")
    movable = bandwidth_bytes_per_s * lookahead * t_tok_s
    return math.floor(movable / expert_bytes)


def assert_config_url(url: str) -> None:
    """A URL do microbench aponta para ``config.json``. Peso não passa."""

    path = urllib.parse.urlparse(url).path
    if not path.endswith("/config.json"):
        raise ValueError(
            "a URL precisa terminar em /config.json. Peso (safetensors, bin, pt) não é baixado."
        )


def load_config_json(path: Path) -> dict:
    if path.suffix.lower() != ".json":
        raise ValueError("só JSON de config; pesos não são lidos")
    payload = path.read_bytes()
    if len(payload) > MAX_CONFIG_BYTES:
        raise ValueError("arquivo grande demais para um config.json")
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError("config.json precisa ser um objeto")
    return parsed


def fetch_config_json(url: str, *, timeout_s: float = 30.0) -> dict:
    assert_config_url(url)
    request = urllib.request.Request(url, headers={"User-Agent": "moe-memory-hw-microbench"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = response.read(MAX_CONFIG_BYTES + 1)
    except urllib.error.URLError as exc:
        raise ValueError(f"falha ao buscar config.json: {exc}") from exc
    if len(payload) > MAX_CONFIG_BYTES:
        raise ValueError("resposta grande demais para um config.json")
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError("config.json precisa ser um objeto")
    return parsed


def measure_pinned_h2d_bandwidth(
    nbytes: int,
    *,
    iters: int,
    warmup: int,
) -> GpuMeasurement:
    """Banda de cópia pinned host→GPU. Sem CUDA devolve ``no_cuda``."""

    if nbytes <= 0 or iters <= 0 or warmup < 0:
        raise ValueError("bench de banda: bytes e iters > 0, warmup >= 0")
    if not torch.cuda.is_available():
        return GpuMeasurement(
            None,
            "no_cuda",
            "sem CUDA; a cópia pinned host→GPU não foi medida",
        )
    try:
        host = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        device = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
        for _ in range(warmup):
            device.copy_(host, non_blocking=True)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            device.copy_(host, non_blocking=True)
        end.record()
        torch.cuda.synchronize()
        elapsed_s = start.elapsed_time(end) / 1000.0
    except RuntimeError as exc:
        return GpuMeasurement(None, "failed", f"banda não medida: {exc}")
    if elapsed_s <= 0:
        return GpuMeasurement(None, "failed", "intervalo de banda nulo")
    return GpuMeasurement(
        (nbytes * iters) / elapsed_s,
        "ok",
        f"copia pinned host→GPU de {nbytes} bytes, {iters} vezes, depois de {warmup} aquecimentos",
    )


def measure_expert_gemm_proxy(
    hidden_size: int,
    intermediate_size: int,
    *,
    iters: int,
    warmup: int,
) -> GpuMeasurement:
    """Tempo de um SwiGLU de um token. Proxy, não o ``t_tok`` do OLMoE."""

    if hidden_size <= 0 or intermediate_size <= 0 or iters <= 0 or warmup < 0:
        raise ValueError("proxy de t_tok: dims e iters > 0, warmup >= 0")
    if not torch.cuda.is_available():
        return GpuMeasurement(
            None,
            "no_cuda",
            "sem CUDA; o proxy de t_tok não foi medido",
        )
    try:
        device = torch.device("cuda")
        token = torch.randn(1, hidden_size, device=device)
        gate = torch.randn(intermediate_size, hidden_size, device=device)
        up = torch.randn(intermediate_size, hidden_size, device=device)
        down = torch.randn(hidden_size, intermediate_size, device=device)

        def once() -> torch.Tensor:
            activated = torch.nn.functional.silu(torch.nn.functional.linear(token, gate))
            mixed = activated * torch.nn.functional.linear(token, up)
            return torch.nn.functional.linear(mixed, down)

        for _ in range(warmup):
            once()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            once()
        end.record()
        torch.cuda.synchronize()
        elapsed_s = start.elapsed_time(end) / 1000.0
    except RuntimeError as exc:
        return GpuMeasurement(None, "failed", f"proxy de t_tok não medido: {exc}")
    if elapsed_s <= 0:
        return GpuMeasurement(None, "failed", "intervalo do proxy nulo")
    return GpuMeasurement(
        elapsed_s / iters,
        "ok",
        "um expert SwiGLU, um token, nas dims do config; não é o t_tok do OLMoE "
        "(sem atenção, sem os outros experts, sem tráfego de ativação)",
    )


def render_report(
    *,
    config_source: str,
    expert: ExpertSize,
    cuda: bool,
    bandwidth: GpuMeasurement | None,
    bandwidth_argument: float | None,
    t_tok_argument: float | None,
    token_proxy: GpuMeasurement | None,
    m: int | None,
    m_hot_bytes: int | None,
    lookahead: int | None,
) -> str:
    """Texto ``chave=valor``. Cota sem dado fica ``unavailable``."""

    lines = [
        "Microbench de viabilidade do roteador LSTM adiantado. Sem peso de modelo.",
        f"config_source={config_source}",
        f"model_type={expert.model_type}",
        f"hidden_size={expert.hidden_size}",
        f"intermediate_size={expert.intermediate_size}",
        f"num_hidden_layers={expert.n_layers}",
        f"num_experts={expert.n_experts}",
        f"num_experts_per_tok={expert.experts_per_token}",
        f"torch_dtype={expert.dtype_name}",
        f"dtype_bytes={expert.dtype_bytes}",
        f"dtype_bytes_source={expert.dtype_bytes_source}",
        f"expert_parameters={expert.parameters}",
        f"S_e_bytes={expert.nbytes}",
        "S_e_layout=swiglu_sem_bias_gate_up_down",
        "S_e_same_every_layer=true",
        "S_e_note=um intermediate_size no config; o script repete S_e em toda camada e nao desconta reserva, overlap nem ativacao",
        f"cuda={str(cuda).lower()}",
        f"native_executed_slots={expert.native_executed_slots}",
        "native_executed_note=num_experts_per_tok * num_hidden_layers, sem reuso entre camadas",
    ]

    if bandwidth_argument is not None:
        bw_value: float | None = bandwidth_argument
        bw_source = "argument"
        bw_detail = "BW passado em --bw-bytes-per-s; a medição desta corrida não substitui o argumento"
        lines.append("medição de banda: não usada; BW veio do argumento")
    elif bandwidth is None or bandwidth.value is None:
        bw_value = None
        bw_source = "no_cuda" if bandwidth is None or bandwidth.status == "no_cuda" else "failed"
        bw_detail = (
            "sem CUDA; a cópia pinned host→GPU não foi medida"
            if bw_source == "no_cuda"
            else (bandwidth.detail if bandwidth is not None else "banda ausente")
        )
        lines.append("medição de banda: não executada")
    else:
        bw_value = bandwidth.value
        bw_source = "pinned_h2d"
        bw_detail = bandwidth.detail
        lines.append("medição de banda: executada")
    lines.append(f"BW_bytes_per_s={_fmt_optional(bw_value)}")
    lines.append(f"BW_source={bw_source}")
    lines.append(f"BW_detail={bw_detail}")

    if t_tok_argument is not None:
        t_tok: float | None = t_tok_argument
        t_source = "argument"
        t_detail = "t_tok passado em --t-tok"
        lines.append("medição de t_tok: não usada; t_tok veio do argumento")
    elif token_proxy is None or token_proxy.value is None:
        t_tok = None
        t_source = "no_cuda" if token_proxy is None or token_proxy.status == "no_cuda" else "failed"
        if t_source == "no_cuda":
            t_source = "unavailable"
            t_detail = "sem --t-tok e sem CUDA; o proxy não foi medido"
        else:
            t_detail = token_proxy.detail if token_proxy is not None else "t_tok ausente"
        lines.append("medição de t_tok: não executada")
    else:
        t_tok = token_proxy.value
        t_source = "proxy_one_expert_swiglu"
        t_detail = token_proxy.detail
        lines.append("medição de t_tok: proxy, não é o token do OLMoE")
    lines.append(f"t_tok_s={_fmt_optional(t_tok)}")
    lines.append(f"t_tok_source={t_source}")
    lines.append(f"t_tok_detail={t_detail}")

    lines.append(f"m={m if m is not None else 'unavailable'}")
    lines.append(f"M_hot_bytes={m_hot_bytes if m_hot_bytes is not None else 'unavailable'}")
    lines.append("L_min_formula=ceil(m*S_e/(BW*t_tok))")
    lines.append("K_mem_formula=sum_l(K_l)<=M_hot/S_e")
    lines.append("K_bw_formula=K*S_e<=BW*L*t_tok")

    l_min: int | None = None
    if m is not None and bw_value is not None and t_tok is not None:
        l_min = lookahead_min(m, expert.nbytes, bw_value, t_tok)
        lines.append(f"L_min={l_min}")
    else:
        lines.append("L_min=unavailable")

    if m_hot_bytes is not None:
        k_mem = memory_slot_sum_max(m_hot_bytes, expert.nbytes)
        lines.append(f"K_mem_sum_max_slots={k_mem}")
        lines.append(f"K_mem_ratio={m_hot_bytes}/{expert.nbytes}")
        lines.append(
            "native_vs_memory="
            + ("fits" if expert.native_executed_slots <= k_mem else "exceeds")
        )
    else:
        lines.append("K_mem_sum_max_slots=unavailable")
        lines.append("K_mem_ratio=unavailable")
        lines.append("native_vs_memory=unavailable")

    if lookahead is not None:
        k_bw_l: int | None = lookahead
        k_bw_l_source = "argument"
    elif l_min is not None:
        k_bw_l = l_min
        k_bw_l_source = "L_min"
    else:
        k_bw_l = None
        k_bw_l_source = "unavailable"
    lines.append(f"K_bw_L={k_bw_l if k_bw_l is not None else 'unavailable'}")
    lines.append(f"K_bw_L_source={k_bw_l_source}")
    if k_bw_l is not None and bw_value is not None and t_tok is not None:
        k_bw = bandwidth_slot_max(bw_value, k_bw_l, t_tok, expert.nbytes)
        lines.append(f"K_bw_max_slots={k_bw}")
        lines.append(
            "native_vs_bandwidth="
            + ("fits" if expert.native_executed_slots <= k_bw else "exceeds")
        )
    else:
        lines.append("K_bw_max_slots=unavailable")
        lines.append("native_vs_bandwidth=unavailable")

    lines.append(
        "bounds_note=estimativa de viabilidade; a demanda sustentada tem de caber em BW; "
        "L maior nao cobre sobrecarga de banda; M_hot entra como foi dado"
    )
    lines.append(
        "K_note=K_mem_sum_max_slots e K_bw_max_slots limitam o total de experts, somado nas camadas"
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Mede banda pinned host→GPU e imprime L_min e as cotas de K. Sem CUDA, pula a GPU."
    )
    parser.add_argument("--config", type=Path, default=None, help="config.json local. Não lê peso.")
    parser.add_argument(
        "--config-url",
        default=None,
        help=f"URL de config.json. Padrão, se --config faltar: {OLMOE_CONFIG_URL}",
    )
    parser.add_argument("--t-tok", type=float, default=None, help="Tempo de um token, em segundos.")
    parser.add_argument("--m", type=int, default=None, help="Experts que ainda precisam ser transferidos.")
    parser.add_argument("--m-hot-bytes", type=int, default=None, help="Orçamento quente já líquido, em bytes.")
    parser.add_argument(
        "--lookahead",
        type=int,
        default=None,
        help="L da cota de banda. Sem isto, usa L_min quando ele existir.",
    )
    parser.add_argument(
        "--bw-bytes-per-s",
        type=float,
        default=None,
        help="Banda já medida, em bytes/s. Sem isto e com CUDA, o script mede.",
    )
    parser.add_argument(
        "--dtype-bytes",
        type=int,
        default=None,
        help="Bytes por parâmetro. Sem isto, usa torch_dtype do config.",
    )
    parser.add_argument(
        "--bench-bytes",
        type=int,
        default=None,
        help="Tamanho da cópia de banda. Sem isto, usa S_e.",
    )
    parser.add_argument("--bench-iters", type=int, default=10)
    parser.add_argument("--bench-warmup", type=int, default=2)
    args = parser.parse_args(argv)

    try:
        if args.config is not None:
            config = load_config_json(args.config)
            source = str(args.config)
        else:
            url = args.config_url or OLMOE_CONFIG_URL
            config = fetch_config_json(url)
            source = url
        expert = expert_size_from_config(config, dtype_bytes_override=args.dtype_bytes)
        _reject_bad_optional(args)
        bandwidth: GpuMeasurement | None = None
        token_proxy: GpuMeasurement | None = None
        if args.bw_bytes_per_s is None:
            bench_bytes = args.bench_bytes if args.bench_bytes is not None else expert.nbytes
            bandwidth = measure_pinned_h2d_bandwidth(
                bench_bytes,
                iters=args.bench_iters,
                warmup=args.bench_warmup,
            )
        if args.t_tok is None:
            token_proxy = measure_expert_gemm_proxy(
                expert.hidden_size,
                expert.intermediate_size,
                iters=args.bench_iters,
                warmup=args.bench_warmup,
            )
        report = render_report(
            config_source=source,
            expert=expert,
            cuda=torch.cuda.is_available(),
            bandwidth=bandwidth,
            bandwidth_argument=args.bw_bytes_per_s,
            t_tok_argument=args.t_tok,
            token_proxy=token_proxy,
            m=args.m,
            m_hot_bytes=args.m_hot_bytes,
            lookahead=args.lookahead,
        )
    except (ValueError, json.JSONDecodeError, OSError) as exc:
        raise SystemExit(str(exc))
    print(report, end="")


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} precisa ser inteiro > 0")
    return value


def _reject_bad_optional(args: argparse.Namespace) -> None:
    if args.m is not None and args.m < 0:
        raise ValueError("--m precisa ser >= 0")
    if args.m_hot_bytes is not None and args.m_hot_bytes < 0:
        raise ValueError("--m-hot-bytes precisa ser >= 0")
    if args.lookahead is not None and args.lookahead < 0:
        raise ValueError("--lookahead precisa ser >= 0")
    if args.t_tok is not None and (not math.isfinite(args.t_tok) or args.t_tok <= 0):
        raise ValueError("--t-tok precisa ser finito e > 0")
    if args.bw_bytes_per_s is not None and (
        not math.isfinite(args.bw_bytes_per_s) or args.bw_bytes_per_s <= 0
    ):
        raise ValueError("--bw-bytes-per-s precisa ser finito e > 0")
    if args.bench_iters < 1 or args.bench_warmup < 0:
        raise ValueError("--bench-iters precisa ser >= 1 e --bench-warmup >= 0")
    if args.bench_bytes is not None and args.bench_bytes <= 0:
        raise ValueError("--bench-bytes precisa ser > 0")


def _fmt_optional(value: float | None) -> str:
    if value is None:
        return "unavailable"
    if math.isfinite(value) and float(value).is_integer():
        return str(int(value))
    return format(value, ".12g")


if __name__ == "__main__":
    main()
