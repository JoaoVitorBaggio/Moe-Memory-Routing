#!/usr/bin/env python3
"""Perplexidade de hold-out no WikiText-2 (valid e test). Não treina.

Os quatro checkpoints da entrega são a semente 0 (gitignored). O treino
novo grava ``seed-0/``; se essa pasta ainda não tiver checkpoint e o
layout antigo tiver, a descoberta usa o antigo:

    outputs/baseline/wikitext/seed-0/checkpoint.pt       LM denso
    outputs/wikitext/seed-0/lambda-0/checkpoint.pt       MoE λ=0
    outputs/wikitext/seed-0/lambda-0.5/checkpoint.pt     MoE λ=0.5
    outputs/wikitext/seed-0/lambda-1/checkpoint.pt       MoE λ=1

Layout antigo da semente 0, ainda aceito:

    outputs/baseline/wikitext/checkpoint.pt
    outputs/wikitext/lambda-<λ>/checkpoint.pt

As sementes 1..N-1 ficam no disco do treino. Este script não as agrega.

O denso não é o MoE com λ=0: o arquivo denso carrega ``DenseLanguageModel``;
os outros carregam ``MemoryMoEModel`` (roteador, emulador, β). A CE é
``task_cross_entropy``, nos mesmos blocos não sobrepostos do treino
(BPE do GPT-2, ``<|endoftext|>`` entre artigos, ``seq_len`` do checkpoint).
Perplexidade = exp dessa média por token previsto.

No MoE o roteador vê ``ℓ(e)``. Cada split recomeça o LRU no ``B`` do YAML
e dá um passo de emulador por batch gravado no checkpoint. ``--batch-size``
só parte o forward; não muda o grão do LRU nem a PPL. O cache do fim do
treino não entra: senão valid e test compartilhariam residência com o
train. Os pesos ficam no device do forward (fp32). Não há ``sync_physical``
no memmap do treino. No CUDA, ``materialize`` estaciona o expert fora de
``B`` antes do cronômetro; a subida entra no forward. ``emulator.step``
fica fora do relógio. Sem CUDA, ``wall_clock`` é ``unavailable`` e os
milissegundos ficam null.

Depois do B do YAML, cada checkpoint MoE que este script avalia (a
semente 0 na descoberta, ou o arquivo pedido na linha de comando)
repete valid e test em ``eval.B_mult`` (0,5× / 1× / 2× no YAML de
partida). O teto é ``dose_response.prepare_split_budget`` (reset no B
do YAML e ``set_hot_budget``). Não há outro método de dose e não há
``--b-mult``. O mesmo ``score_split`` mede a perplexidade, ``swaps``,
``bytes_moved`` e a parede. O denso não tem esse eixo (n/a). Não há
retreino. A tabela sai em ``dose_response.md``. As sementes 1..N-1 não
entram na descoberta e não são agregadas.

O número de relatório da parede é na GTX 1060 (``wall_report_grade``).

Fora deste script: agregar as sementes além da 0 e a correlação
proxy×emulador. O rótulo de modelo 2 é o mesmo checkpoint ``λ=0`` da
semente 0 (não uma segunda política); quem empacota a comparação é
``run_model_comparison.py``. ``smoke_eval.py`` continua sendo init
aleatória.

Uso, a partir da raiz, na GTX 1060::

    python3 experiments/run_holdout_eval.py --device cuda

Um checkpoint, ou um corte de CPU (sai com código 7: não é o relatório).
``--seq-len`` diferente do checkpoint também sai com 7::

    python3 experiments/run_holdout_eval.py --checkpoint outputs/wikitext/seed-0/lambda-0.5/checkpoint.pt --device cuda
    python3 experiments/run_holdout_eval.py --work-dir outputs/baseline/wikitext/seed-0 --max-batches 1 --device cpu

Códigos: 1 uso, 3 OOM, 4 NLL não finita, 5 crash, 7 hold-out incompleto
(checkpoint ausente, split cortado, ``seq_len`` diferente do treino, ou
sem valid e test). O JSON é gravado mesmo no código 7.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import torch

from moe_memory.config import ExperimentConfig, load_config
from moe_memory.corpus import iter_batches, iter_documents, iter_token_ids, source_label
from moe_memory.dense import (
    MODEL_KIND,
    DenseLanguageModel,
    count_dense_parameters,
    count_moe_parameters,
)
from moe_memory.levels import COLD, HOT, WARM
from moe_memory.loss import compute_loss, task_cross_entropy
from moe_memory.model import MemoryMoEModel
from moe_memory.residency import ExpertResidency, attach_residency
from moe_memory.smoke import build_emulator, level_cost_tensor
from moe_memory.tokenizer import load_gpt2_encoding
import dose_response
from train_baseline import CHECKPOINT_VERSION as DENSE_CHECKPOINT_VERSION
from train_baseline import NOTE as DENSE_NOTE
from train_baseline import _architecture as dense_architecture
from seed_layout import (
    baseline_seed_dir,
    is_seed_dirname,
    lambda_run_dir,
    prefer_checkpoint,
)
from train_wikitext import (
    CHECKPOINT_VERSION as MOE_CHECKPOINT_VERSION,
    EXIT_CRASH,
    EXIT_NONFINITE,
    EXIT_OOM,
    EXIT_USAGE,
    _architecture as moe_architecture,
    _release_cuda,
    _torch_load,
    die,
    is_oom,
    lambda_tag,
    same_lambda,
)

EXIT_INCOMPLETE = 7
KIND_DENSE = MODEL_KIND
KIND_MOE = "moe"
PROTOCOL = "wikitext-2-holdout-ppl"
REPORT_SPLITS = ("valid", "test")

METRIC_NOTE = (
    "perplexity = exp(nll_nats). nll_nats é a média de task_cross_entropy: "
    "bloco não sobreposto de seq_len, a posição t prediz t+1, o primeiro "
    "token do bloco não é previsto, o resto que não fecha seq_len cai fora "
    "(o mesmo descarte do treino). bits_per_byte = (nll_nats_sum / ln(2)) / "
    "utf8_bytes, com utf8_bytes = UTF-8 do decode dos tokens empacotados que "
    "entraram na conta, sem <|endoftext|>. Não é a perplexidade por palavra "
    "dos papers de WikiText. λ não multiplica a PPL. No MoE, C_mem e entropia "
    "usam a agregação de inferência (sum); o balanço é o do batch lógico do "
    "checkpoint. O LRU recomeça em cada split, no B do YAML. A curva "
    "eval.B_mult usa prepare_split_budget (reset nesse B e set_hot_budget) "
    "e o mesmo score_split: perplexidade, swaps, bytes_moved e a parede. "
    "O denso é n/a nesse eixo. A descoberta é a semente 0. wall_p50_ms e "
    "wall_p95_ms são o forward; sem CUDA ficam null (wall_clock=unavailable). "
    "Número de relatório da parede: GTX 1060, batch do checkpoint, split "
    "inteiro (wall_report_grade). Agregar sementes além da 0 e a "
    "correlação proxy×emulador ficam fora. O rótulo de modelo 2 é o mesmo "
    "λ=0 da semente 0, empacotado por run_model_comparison.py."
)

WALL_CLOCK_CUDA = "cuda"
WALL_CLOCK_UNAVAILABLE = "unavailable"
WALL_FIELDS = (
    "wall_p50_ms",
    "wall_p95_ms",
    "wall_clock",
    "wall_samples",
    "wall_report_grade",
    "wall_gpu",
    "wall_forward_batch",
)

WALL_NOTE = (
    "Parede do forward, em ms, com torch.cuda.synchronize antes e depois. "
    "No MoE em CUDA, materialize() alinha o storage a ℓ(e) fora do "
    "cronômetro; a subida do expert entra no forward; emulator.step fica "
    "fora. p50/p95 usam o mesmo nearest-rank de experiments/wall_latency.py. "
    "wall_report_grade só é verdadeiro na GTX 1060 (o nome do device contém "
    "1060), com o batch do checkpoint e o split sem --max-batches. Outra "
    "CUDA ainda pode gravar os ms, com a flag falsa. Em CPU, ou sem CUDA, "
    "wall_clock='unavailable', wall_samples=0 e os ms ficam null. O script "
    "não inventa tempo. swaps e bytes_moved são do emulador nesse passe, "
    "não a latência nem a reatribuição de set_hot_budget (essa fica em "
    "budget_reassign). A curva de B não tem outro cronômetro: "
    "prepare_split_budget ajusta o teto e score_split mede o forward. "
    "Não há --b-mult nem atribuição direta de hot_slots."
)

MOE_NOTE = (
    "MoE com roteador e emulador. λ=0 ainda calcula C_mem (peso zero na loss "
    "de treino) e soma β L_balanço. Não é o LM denso."
)


def percentile(ordered: list[float], percent: float) -> float:
    """Nearest-rank. A mesma conta de ``experiments/wall_latency.py``."""

    if not ordered:
        die(EXIT_CRASH, "percentil sem amostra")
    if len(ordered) == 1:
        return ordered[0]
    rank = max(0, math.ceil(percent / 100 * len(ordered)) - 1)
    return ordered[min(rank, len(ordered) - 1)]


def cuda_gpu_name(device: torch.device) -> str | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    index = 0 if device.index is None else int(device.index)
    return str(torch.cuda.get_device_name(index))


def _is_gtx_1060(gpu: str | None) -> bool:
    return gpu is not None and "1060" in gpu


def _unavailable_wall(forward_batch: int) -> dict[str, Any]:
    return {
        "wall_p50_ms": None,
        "wall_p95_ms": None,
        "wall_clock": WALL_CLOCK_UNAVAILABLE,
        "wall_samples": 0,
        "wall_report_grade": False,
        "wall_gpu": None,
        "wall_forward_batch": int(forward_batch),
    }


def wall_from_samples(
    samples_s: list[float],
    *,
    gpu: str | None,
    forward_batch: int,
    logical_batch: int,
    truncated: bool,
) -> dict[str, Any]:
    """Monta o bloco de parede. Não mede e não chama CUDA."""

    if not samples_s:
        return _unavailable_wall(forward_batch)
    ordered = sorted(float(sample) for sample in samples_s)
    if not all(math.isfinite(sample) and sample >= 0 for sample in ordered):
        die(EXIT_CRASH, "amostra de parede não é finita e não-negativa")
    return {
        "wall_p50_ms": percentile(ordered, 50) * 1e3,
        "wall_p95_ms": percentile(ordered, 95) * 1e3,
        "wall_clock": WALL_CLOCK_CUDA,
        "wall_samples": len(ordered),
        "wall_report_grade": bool(
            _is_gtx_1060(gpu) and forward_batch == logical_batch and not truncated
        ),
        "wall_gpu": gpu,
        "wall_forward_batch": int(forward_batch),
    }


def assemble_wall(
    samples_s: list[float],
    *,
    device: torch.device,
    forward_batch: int,
    logical_batch: int,
    truncated: bool,
) -> dict[str, Any]:
    """p50/p95 se o forward rodou em CUDA. CPU descarta qualquer amostra."""

    if device.type != "cuda" or not torch.cuda.is_available():
        return _unavailable_wall(forward_batch)
    return wall_from_samples(
        samples_s,
        gpu=cuda_gpu_name(device),
        forward_batch=forward_batch,
        logical_batch=logical_batch,
        truncated=truncated,
    )


def _park(residency: ExpertResidency | None) -> None:
    """Alinha o storage ao ℓ(e). Fora do cronômetro. CPU não grava memmap."""

    if residency is None or residency.backend is None:
        return
    residency.materialize()


@dataclass(frozen=True)
class Family:
    name: str
    path: Path
    expected_kind: str | None = None
    expected_lambda: float | None = None


@dataclass(frozen=True)
class PackedSplit:
    split: str
    sequences: torch.Tensor
    seq_len: int
    documents: int
    tokens_in_stream: int
    tokens_discarded: int
    source_utf8_bytes: int
    eot_token: int


class RouterTally:
    """Balanço do Switch num batch lógico, somando microbatches.

    ``balance_loss`` não é a média dos pedaços: ``f(e)`` é a fração do
    batch inteiro. Os pedaços do forward entram aqui e o valor sai uma vez,
    no grão do batch do checkpoint.
    """

    def __init__(self) -> None:
        self.counts: torch.Tensor | None = None
        self.prob_sum: torch.Tensor | None = None
        self.positions = 0
        self.k: int | None = None

    def add(self, probs: torch.Tensor, topk_index: torch.Tensor) -> None:
        layers, batch, seq, n_experts = probs.shape
        k = int(topk_index.shape[-1])
        if self.counts is None or self.prob_sum is None:
            self.counts = torch.zeros(layers, n_experts, dtype=torch.float64)
            self.prob_sum = torch.zeros(layers, n_experts, dtype=torch.float64)
            self.k = k
        elif self.k != k or tuple(self.counts.shape) != (layers, n_experts):
            die(EXIT_CRASH, "tally de roteador mudou de shape no meio do batch")
        self.positions += int(batch * seq)
        self.prob_sum = self.prob_sum + probs.detach().to(dtype=torch.float64, device="cpu").sum(
            dim=(1, 2)
        )
        index = topk_index.detach().to(dtype=torch.long, device="cpu")
        ones = torch.ones(batch * seq * k, dtype=torch.float64)
        assert self.counts is not None
        for layer in range(layers):
            self.counts[layer].scatter_add_(0, index[layer].reshape(-1), ones)

    def value(self) -> float:
        if (
            self.positions < 1
            or self.counts is None
            or self.prob_sum is None
            or self.k is None
        ):
            die(EXIT_CRASH, "tally de roteador vazio")
        assert self.counts is not None and self.prob_sum is not None and self.k is not None
        n_experts = int(self.counts.shape[-1])
        layers: list[float] = []
        for layer in range(int(self.counts.shape[0])):
            fraction = self.counts[layer] / (self.positions * self.k)
            mean_prob = self.prob_sum[layer] / self.positions
            layers.append(float(n_experts * torch.sum(fraction * mean_prob)))
        return sum(layers) / len(layers)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Perplexidade de hold-out no WikiText-2 (valid e test, fp32). Não treina."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help="Arquivo checkpoint.pt. Repetível. Exclusivo com --work-dir.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Pasta com checkpoint.pt. Exclusivo com --checkpoint.",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=ROOT / "outputs",
        help="Raiz da descoberta. Padrão: outputs/ (baseline + λ da grade).",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="JSON único. Padrão da descoberta: outputs/holdout/summary.json.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Padrão: o seq_len gravado no checkpoint (256 na entrega).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Microbatch do forward. Padrão: o batch do checkpoint. "
            "Não muda o passo do LRU nem a PPL."
        ),
    )
    parser.add_argument(
        "--splits",
        default="valid,test",
        help="Padrão: valid,test. O relatório pede os dois.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Corta cada split em N batches lógicos. Não é o hold-out do relatório.",
    )
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args(argv)
    if args.checkpoint and args.work_dir is not None:
        die(EXIT_USAGE, "use só --checkpoint ou só --work-dir")
    if args.batch_size is not None and args.batch_size < 1:
        die(EXIT_USAGE, "--batch-size precisa ser >= 1")
    if args.seq_len is not None and args.seq_len < 2:
        die(EXIT_USAGE, "--seq-len precisa ser >= 2")
    if args.max_batches is not None and args.max_batches < 1:
        die(EXIT_USAGE, "--max-batches precisa ser >= 1")
    if args.log_every < 1:
        die(EXIT_USAGE, "--log-every precisa ser >= 1")
    args.splits = parse_splits(args.splits)
    args.explicit = bool(args.checkpoint) or args.work_dir is not None
    if args.work_dir is not None:
        args.checkpoint = [str(Path(args.work_dir) / "checkpoint.pt")]
    return args


def parse_splits(text: str) -> tuple[str, ...]:
    parts = [piece.strip() for piece in text.split(",") if piece.strip()]
    if not parts:
        die(EXIT_USAGE, "--splits vazio")
    unknown = [piece for piece in parts if piece not in REPORT_SPLITS]
    if unknown:
        die(
            EXIT_USAGE,
            "--splits só aceita valid e test (hold-out oficial do WikiText-2); "
            f"recebido {unknown}",
        )
    if len(parts) != len(set(parts)):
        die(EXIT_USAGE, "--splits repetido")
    return tuple(parts)


def discover_families(outputs: Path, config: ExperimentConfig) -> list[Family]:
    """Baseline denso + um MoE por λ, semente 0, na ordem do YAML.

    A pasta canônica é ``seed-0``. Se ela ainda não tiver checkpoint e o
    layout antigo tiver, o caminho antigo entra no lugar.
    """

    baseline_root = Path(outputs) / "baseline" / "wikitext"
    families = [
        Family(
            name="baseline",
            path=prefer_checkpoint(baseline_seed_dir(baseline_root, 0), baseline_root),
            expected_kind=KIND_DENSE,
            expected_lambda=None,
        )
    ]
    grid_root = Path(outputs) / "wikitext"
    for value in config.loss.lambda_grid:
        tag = lambda_tag(float(value))
        families.append(
            Family(
                name=f"lambda-{tag}",
                path=prefer_checkpoint(
                    lambda_run_dir(grid_root, float(value), 0),
                    grid_root / f"lambda-{tag}",
                ),
                expected_kind=KIND_MOE,
                expected_lambda=float(value),
            )
        )
    return families


def families_from_checkpoints(paths: list[str]) -> list[Family]:
    families = [
        Family(name=name_from_path(Path(raw)), path=Path(raw)) for raw in paths
    ]
    return _unique_names(families)


def name_from_path(path: Path) -> str:
    parent = path.parent
    if (
        is_seed_dirname(parent.name)
        and parent.parent.name == "wikitext"
        and parent.parent.parent.name == "baseline"
    ):
        return "baseline"
    if parent.name == "wikitext" and parent.parent.name == "baseline":
        return "baseline"
    if parent.name:
        return parent.name
    return path.stem


def _unique_names(families: list[Family]) -> list[Family]:
    seen: dict[str, int] = {}
    named: list[Family] = []
    for family in families:
        count = seen.get(family.name, 0) + 1
        seen[family.name] = count
        if count == 1:
            named.append(family)
            continue
        named.append(replace(family, name=f"{family.name}-{count}"))
    return named


def summary_path_for(args: argparse.Namespace, n_families: int) -> Path | None:
    if args.summary is not None:
        return Path(args.summary)
    if args.explicit and n_families == 1:
        return None
    return Path(args.outputs) / "holdout" / "summary.json"


def checkpoint_kind(blob: dict[str, Any]) -> str:
    if not isinstance(blob, dict):
        die(EXIT_USAGE, "checkpoint não é um dict")
    arch = blob.get("architecture")
    arch_kind = arch.get("kind") if isinstance(arch, dict) else None
    kind = blob.get("model_kind")
    if kind == KIND_DENSE or arch_kind == KIND_DENSE:
        if kind not in (None, KIND_DENSE) or arch_kind not in (None, KIND_DENSE):
            die(EXIT_USAGE, "checkpoint mistura model_kind denso com outro tipo")
        return KIND_DENSE
    if "lambda" in blob and isinstance(arch, dict) and "E" in arch:
        return KIND_MOE
    die(
        EXIT_USAGE,
        "não reconheci o checkpoint. O denso tem model_kind='dense'. "
        "O MoE, inclusive λ=0, tem λ e arquitetura com E.",
    )
    raise AssertionError("die não retorna")


def pack_split(config: ExperimentConfig, split: str, seq_len: int) -> PackedSplit:
    """Blocos ``[N, seq_len]`` na ordem do loader, o mesmo corte do treino.

    O nome é WikiText-2 mesmo com ``corpus.active: dev``. O embaralhamento
    da época de treino não entra: hold-out é a ordem do dataset.
    """

    if split not in {"train", "valid", "test"}:
        raise ValueError(f"split {split!r} não é train, valid ou test")
    if seq_len < 2:
        raise ValueError("seq_len precisa ser >= 2")
    documents = list(iter_documents(config, split, name="WikiText-2", streaming=False))
    encoding = load_gpt2_encoding()
    token_ids = list(iter_token_ids(documents, encoding))
    rows = list(iter_batches(token_ids, seq_len=seq_len, batch_size=64))
    if rows:
        sequences = torch.cat(rows, dim=0)
    else:
        sequences = torch.empty(0, seq_len, dtype=torch.long)
    kept = int(sequences.numel())
    return PackedSplit(
        split=split,
        sequences=sequences,
        seq_len=seq_len,
        documents=len(documents),
        tokens_in_stream=len(token_ids),
        tokens_discarded=len(token_ids) - kept,
        source_utf8_bytes=sum(len(document.encode("utf-8")) for document in documents),
        eot_token=int(encoding.eot_token),
    )


def utf8_bytes(token_ids: list[int], encoding) -> int:
    """UTF-8 do decode, pulando o EOT que o loader insere entre artigos."""

    eot = int(encoding.eot_token)
    total = 0
    run: list[int] = []
    for token in token_ids:
        if int(token) == eot:
            if run:
                total += len(encoding.decode(run).encode("utf-8"))
                run = []
            continue
        run.append(int(token))
    if run:
        total += len(encoding.decode(run).encode("utf-8"))
    return total


def scored_utf8_bytes(packed: PackedSplit, n_sequences: int) -> int:
    if n_sequences < 0 or n_sequences > int(packed.sequences.size(0)):
        die(EXIT_CRASH, "n_sequences fora do pack")
    if n_sequences == 0:
        return 0
    flat = packed.sequences[:n_sequences].reshape(-1).tolist()
    return utf8_bytes(flat, load_gpt2_encoding())


def perplexity(mean_nll: float) -> float | None:
    if not math.isfinite(mean_nll):
        return None
    if mean_nll > 700:
        return None
    value = math.exp(mean_nll)
    if not math.isfinite(value):
        return None
    return value


def bits_per_byte(nll_sum: float, n_utf8_bytes: int) -> float | None:
    if n_utf8_bytes < 1 or not math.isfinite(nll_sum):
        return None
    return (nll_sum / math.log(2)) / n_utf8_bytes


def accumulate_nll(logits: torch.Tensor, input_ids: torch.Tensor) -> tuple[float, int]:
    """Soma da NLL. A média é ``task_cross_entropy``; vezes o nº de alvos recupera a soma."""

    n_pred = int(input_ids[:, 1:].numel())
    if n_pred < 1:
        die(EXIT_CRASH, "CE de próximo token precisa de seq_len >= 2")
    mean = task_cross_entropy(logits, input_ids)
    value = float(mean.detach().to(dtype=torch.float64))
    if not math.isfinite(value):
        return value, n_pred
    return value * n_pred, n_pred


def add_dispatched(total: torch.Tensor | None, extra: torch.Tensor) -> torch.Tensor:
    piece = extra.detach().to(dtype=torch.long, device="cpu")
    if total is None:
        return piece.clone()
    if piece.shape != total.shape:
        die(
            EXIT_CRASH,
            f"dispatched {tuple(piece.shape)} não casa com {tuple(total.shape)}",
        )
    return total + piece


def iter_logical(n_seq: int, logical: int, max_logical: int | None):
    """Fatias no grão do batch do checkpoint. A última pode ser menor, como no treino."""

    if logical < 1:
        die(EXIT_USAGE, "batch lógico precisa ser >= 1")
    cursor = 0
    produced = 0
    while cursor < n_seq:
        if max_logical is not None and produced >= max_logical:
            return
        take = min(logical, n_seq - cursor)
        yield cursor, take
        cursor += take
        produced += 1


def logical_count(n_seq: int, logical: int, max_logical: int | None) -> int:
    return sum(1 for _ in iter_logical(n_seq, logical, max_logical))


def iter_forward(start: int, take: int, forward: int):
    if forward < 1:
        die(EXIT_USAGE, "batch de forward precisa ser >= 1")
    cursor = 0
    while cursor < take:
        step = min(forward, take - cursor)
        yield start + cursor, step
        cursor += step


def score_split(
    model,
    packed: PackedSplit,
    *,
    kind: str,
    device: torch.device,
    logical_batch: int,
    forward_batch: int,
    max_logical: int | None,
    log_every: int,
    emulator=None,
    config: ExperimentConfig | None = None,
    lambda_weight: float | None = None,
    reset_lru: bool = True,
    residency: ExpertResidency | None = None,
) -> dict[str, Any]:
    """Uma passagem. ``truncated`` é corte de ``--max-batches``, não o resto < seq_len.

    ``reset_lru=False`` é a curva de dose–resposta: o chamador já rodou
    ``prepare_split_budget`` (reset no B do YAML e ``set_hot_budget``).
    Um ``reset`` aqui apagaria o carimbo que a troca de teto preserva.
    Esta função não atribui ``hot_slots``. No CUDA o relógio cobre cada
    forward; ``materialize`` e ``emulator.step`` ficam fora. No CPU os
    campos de parede existem e ficam indisponíveis.
    """

    if forward_batch > logical_batch:
        forward_batch = logical_batch
    n_seq = int(packed.sequences.size(0))
    if n_seq < 1:
        die(EXIT_USAGE, f"WikiText-2 {packed.split} não fechou sequência de seq_len={packed.seq_len}")
    if int(packed.sequences.size(1)) != packed.seq_len:
        die(EXIT_CRASH, "pack com seq_len inconsistente")
    planned = logical_count(n_seq, logical_batch, max_logical)
    if kind == KIND_MOE:
        if emulator is None or config is None or lambda_weight is None:
            die(EXIT_CRASH, "MoE sem emulador na avaliação")
        if reset_lru:
            emulator.reset()
        elif int(emulator.hot_slots) < 1:
            die(EXIT_CRASH, "dose–resposta MoE sem teto quente")
        costs = level_cost_tensor(config)
        aggregation = config.loss.aggregation(training=False)
        if aggregation != "sum":
            die(EXIT_CRASH, "infer_topk_agg deixou de ser sum")
    else:
        costs = None
        aggregation = None
    model.eval()
    nll_sum = 0.0
    tokens = 0
    sequences_scored = 0
    c_mem_sum = 0.0
    entropy_sum = 0.0
    diag_weight = 0
    balance_sum = 0.0
    balance_weight = 0
    swaps = 0
    bytes_moved = 0
    promotions = 0
    demotions = 0
    logical_done = 0
    wall_times: list[float] = []
    time_wall = device.type == "cuda" and torch.cuda.is_available()
    with torch.no_grad():
        for start, take in iter_logical(n_seq, logical_batch, max_logical):
            logical_done += 1
            tally = RouterTally() if kind == KIND_MOE else None
            dispatched: torch.Tensor | None = None
            if kind == KIND_MOE:
                _park(residency)
            for micro_start, micro in iter_forward(start, take, forward_batch):
                batch = packed.sequences[micro_start : micro_start + micro].to(device)
                started = 0.0
                try:
                    if time_wall:
                        torch.cuda.synchronize(device)
                        started = time.perf_counter()
                    if kind == KIND_DENSE:
                        logits = model(batch)
                        output = None
                    else:
                        assert emulator is not None
                        output = model(batch, emulator.levels)
                        logits = output.logits
                    if time_wall:
                        torch.cuda.synchronize(device)
                        wall_times.append(time.perf_counter() - started)
                except RuntimeError as exc:
                    if is_oom(exc):
                        _release_cuda()
                        die(
                            EXIT_OOM,
                            f"OOM no split {packed.split}, batch lógico {logical_done}. "
                            "Repita com --batch-size 1 ou 2. O passo do LRU continua no "
                            f"batch do checkpoint ({logical_batch}); a PPL não muda. {exc}",
                        )
                    raise
                piece, n_pred = accumulate_nll(logits, batch)
                if not math.isfinite(piece):
                    die(EXIT_NONFINITE, f"NLL não finita no split {packed.split}")
                nll_sum += piece
                tokens += n_pred
                if kind == KIND_MOE:
                    assert (
                        output is not None
                        and config is not None
                        and costs is not None
                        and aggregation is not None
                        and tally is not None
                        and emulator is not None
                        and lambda_weight is not None
                    )
                    terms = compute_loss(
                        logits,
                        batch,
                        output.probs,
                        output.topk_index,
                        emulator.levels,
                        costs,
                        lambda_=float(lambda_weight),
                        beta=config.loss.beta,
                        aggregation=aggregation,
                    )
                    positions = int(batch.numel())
                    c_mem_sum += float(terms.c_mem.detach()) * positions
                    entropy_sum += float(terms.entropy.detach()) * positions
                    diag_weight += positions
                    tally.add(output.probs, output.topk_index)
                    dispatched = add_dispatched(dispatched, output.dispatched)
            sequences_scored += take
            if kind == KIND_MOE:
                assert emulator is not None and tally is not None and dispatched is not None
                stats = emulator.step(dispatched)
                swaps += int(stats.swaps)
                bytes_moved += int(stats.bytes_moved)
                promotions += int(stats.promotions)
                demotions += int(stats.demotions)
                positions = take * packed.seq_len
                balance_sum += tally.value() * positions
                balance_weight += positions
            if logical_done % log_every == 0 or logical_done == planned:
                mean_so_far = nll_sum / tokens
                print(
                    f"  {packed.split} batch {logical_done}/{planned}  nll={mean_so_far:.4f}",
                    flush=True,
                )
    if tokens != sequences_scored * (packed.seq_len - 1):
        die(EXIT_CRASH, "contagem de tokens previstos não fecha com as sequências")
    if logical_done != planned:
        die(EXIT_CRASH, "contagem de batches lógicos não fecha")
    mean_nll = nll_sum / tokens
    utf8 = scored_utf8_bytes(packed, sequences_scored)
    row: dict[str, Any] = {
        "perplexity": perplexity(mean_nll),
        "nll_nats": mean_nll,
        "nll_nats_sum": nll_sum,
        "bits_per_byte": bits_per_byte(nll_sum, utf8),
        "tokens_scored": tokens,
        "tokens_packed": sequences_scored * packed.seq_len,
        "sequences_scored": sequences_scored,
        "sequences_available": n_seq,
        "tokens_discarded": packed.tokens_discarded,
        "documents": packed.documents,
        "utf8_bytes": utf8,
        "source_utf8_bytes": packed.source_utf8_bytes,
        "truncated": sequences_scored < n_seq,
        "logical_batches": logical_done,
        "c_mem": None,
        "balance": None,
        "entropy": None,
        "aggregation": None,
        "swaps": None,
        "bytes_moved": None,
        "promotions": None,
        "demotions": None,
        "hot": None,
        "warm": None,
        "cold": None,
        "lru_tick_end": None,
    }
    if kind == KIND_MOE:
        assert emulator is not None
        if diag_weight < 1 or balance_weight < 1:
            die(EXIT_CRASH, "diagnóstico do MoE sem tokens")
        row.update(
            {
                "c_mem": c_mem_sum / diag_weight,
                "balance": balance_sum / balance_weight,
                "entropy": entropy_sum / diag_weight,
                "aggregation": "sum",
                "swaps": swaps,
                "bytes_moved": bytes_moved,
                "promotions": promotions,
                "demotions": demotions,
                "hot": int((emulator.levels == HOT).sum().item()),
                "warm": int((emulator.levels == WARM).sum().item()),
                "cold": int((emulator.levels == COLD).sum().item()),
                "lru_tick_end": int(emulator.tick),
            }
        )
    row.update(
        assemble_wall(
            wall_times,
            device=device,
            forward_batch=forward_batch,
            logical_batch=logical_batch,
            truncated=bool(row["truncated"]),
        )
    )
    return row


def read_checkpoint(path: Path) -> dict[str, Any]:
    try:
        blob = _torch_load(path)
    except Exception as exc:
        die(EXIT_USAGE, f"checkpoint ilegível em {path}: {exc}")
    if not isinstance(blob, dict):
        die(EXIT_USAGE, f"checkpoint {path} não é um dict")
    return blob


def validate_header(
    blob: dict[str, Any],
    config: ExperimentConfig,
    kind: str,
    family: Family,
) -> None:
    version = DENSE_CHECKPOINT_VERSION if kind == KIND_DENSE else MOE_CHECKPOINT_VERSION
    if blob.get("version") != version:
        die(
            EXIT_USAGE,
            f"versão de checkpoint {blob.get('version')!r} não é {version}",
        )
    if kind == KIND_DENSE and blob.get("corpus") != "WikiText-2":
        die(
            EXIT_USAGE,
            f"checkpoint denso é corpus={blob.get('corpus')!r}; este hold-out é WikiText-2.",
        )
    expected = dense_architecture(config) if kind == KIND_DENSE else moe_architecture(config)
    got = blob.get("architecture")
    if got != expected:
        die(
            EXIT_USAGE,
            "checkpoint não bate com a arquitetura do YAML "
            f"({'densa' if kind == KIND_DENSE else 'MoE'}). "
            "O desenho não se avalia por cima de outro. "
            f"checkpoint={_brief(got)} yaml={_brief(expected)}",
        )
    if family.expected_kind is not None and kind != family.expected_kind:
        die(
            EXIT_USAGE,
            f"{family.name}: checkpoint é {kind}, esta pasta espera {family.expected_kind}. "
            "O denso não é o MoE com λ=0.",
        )
    if kind == KIND_MOE:
        emulator_blob = blob.get("emulator")
        if not isinstance(emulator_blob, dict) or "hot_slots" not in emulator_blob:
            die(EXIT_USAGE, "checkpoint MoE sem emulator.hot_slots")
        expected_hot = config.memory.hot_slots(config.model.E)
        if as_int(emulator_blob["hot_slots"], "emulator.hot_slots") != expected_hot:
            die(
                EXIT_USAGE,
                f"checkpoint tem {emulator_blob['hot_slots']} vagas quentes; o YAML tem {expected_hot}",
            )
        if "lambda" not in blob:
            die(EXIT_USAGE, "checkpoint MoE sem λ")
        if family.expected_lambda is not None and not same_lambda(
            as_float(blob["lambda"], "lambda"), family.expected_lambda
        ):
            die(
                EXIT_USAGE,
                f"{family.name} tem λ={blob['lambda']}, esperado {family.expected_lambda}",
            )
    cursor = blob.get("cursor")
    if not isinstance(cursor, dict):
        die(EXIT_USAGE, "checkpoint sem cursor")
    for key in ("steps", "tokens", "epochs_completed", "epoch", "sequence_cursor"):
        if key not in cursor:
            die(EXIT_USAGE, f"checkpoint sem cursor.{key}")
    for key in ("seed", "seq_len", "batch_size", "lr"):
        if key not in blob:
            die(EXIT_USAGE, f"checkpoint sem {key}")


def load_model(blob: dict[str, Any], config: ExperimentConfig, kind: str, device: torch.device):
    if kind == KIND_DENSE:
        model = DenseLanguageModel.from_config(config)
        expected = count_dense_parameters(config.model)
    else:
        model = MemoryMoEModel.from_config(config)
        expected = count_moe_parameters(config.model)
    try:
        model.load_state_dict(blob["model"])
    except Exception as exc:
        die(EXIT_USAGE, f"state_dict não casa com o {kind} deste YAML: {exc}")
    blob.pop("model", None)
    model = model.to(device=device, dtype=torch.float32)
    model.eval()
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        die(EXIT_USAGE, "o modelo saiu de fp32")
    if model.num_parameters() != expected:
        die(EXIT_CRASH, f"contagem de parâmetros {model.num_parameters()} != {expected}")
    return model


def resolve_seq_len(
    blob: dict[str, Any], override: int | None, config: ExperimentConfig
) -> tuple[int, bool]:
    trained = as_int(blob["seq_len"], "seq_len")
    chosen = trained if override is None else int(override)
    if not 2 <= chosen <= config.model.max_seq_len:
        die(
            EXIT_USAGE,
            f"seq_len={chosen} fora de 2..max_seq_len={config.model.max_seq_len}",
        )
    return chosen, chosen == trained


def resolve_batches(blob: dict[str, Any], requested: int | None) -> tuple[int, int, bool]:
    logical = as_int(blob["batch_size"], "batch_size")
    if logical < 1:
        die(EXIT_USAGE, "checkpoint batch_size < 1")
    if requested is None:
        return logical, logical, False
    clamped = requested > logical
    return logical, min(requested, logical), clamped


def evaluate_family(
    family: Family,
    config: ExperimentConfig,
    args: argparse.Namespace,
    device: torch.device,
    packed_cache: dict[tuple[str, int], PackedSplit],
) -> dict[str, Any]:
    blob = read_checkpoint(family.path)
    kind = checkpoint_kind(blob)
    validate_header(blob, config, kind, family)
    seq_len, context_ok = resolve_seq_len(blob, args.seq_len, config)
    logical_batch, forward_batch, clamped = resolve_batches(blob, args.batch_size)
    if clamped:
        print(
            f"{family.name}: --batch-size {args.batch_size} limitado a {logical_batch} "
            "(batch do checkpoint); o LRU não atravessa esse grão",
            flush=True,
        )
    lambda_weight = None if kind == KIND_DENSE else as_float(blob["lambda"], "lambda")
    if kind == KIND_MOE and lambda_weight is not None:
        canonical = _canonical_lambda(config, lambda_weight)
        in_grid = canonical is not None
        if canonical is not None:
            lambda_weight = canonical
    else:
        in_grid = None
    cursor = blob["cursor"]
    architecture = blob["architecture"]
    record: dict[str, Any] = {
        "name": family.name,
        "status": "ok",
        "path": str(family.path),
        "model_kind": kind,
        "memory_path": kind == KIND_MOE,
        "lambda": lambda_weight,
        "lambda_in_grid": in_grid,
        "note": DENSE_NOTE if kind == KIND_DENSE else MOE_NOTE,
        "seed": as_int(blob["seed"], "seed"),
        "steps": as_int(cursor["steps"], "cursor.steps"),
        "epochs_completed": as_int(cursor["epochs_completed"], "cursor.epochs_completed"),
        "tokens_trained": as_int(cursor["tokens"], "cursor.tokens"),
        "epoch": as_int(cursor["epoch"], "cursor.epoch"),
        "sequence_cursor": as_int(cursor["sequence_cursor"], "cursor.sequence_cursor"),
        "checkpoint_seq_len": as_int(blob["seq_len"], "seq_len"),
        "eval_seq_len": seq_len,
        "context_matches_checkpoint": context_ok,
        "checkpoint_batch_size": logical_batch,
        "forward_batch_size": forward_batch,
        "lr": as_float(blob["lr"], "lr"),
        "architecture": architecture,
        "parameters": None,
        "residency": "lru_reset_per_split" if kind == KIND_MOE else None,
        "splits": {},
    }
    # Adam e RNG do treino não entram na PPL. Soltar antes do forward.
    blob.pop("optimizer", None)
    blob.pop("torch_rng", None)
    blob.pop("cuda_rng", None)
    model = load_model(blob, config, kind, device)
    record["parameters"] = int(model.num_parameters())
    emulator = None
    residency = None
    if kind == KIND_MOE:
        # work_dir só fica no objeto até o materialize. sync_physical não é
        # chamado: o memmap frio do treino permanece intocado. No CUDA,
        # materialize grava o home deste hold-out em holdout_lru. No CPU o
        # backend é None e esse diretório não nasce. O teto da curva continua
        # em prepare_split_budget; a residência só estaciona o storage.
        emulator = build_emulator(
            config,
            model.expert_nbytes(),
            family.path.parent / "holdout_lru",
        )
        if device.type == "cuda":
            residency = attach_residency(model, emulator, device)
    print(
        f"avaliando {family.name}  {kind}  "
        f"λ={record['lambda'] if record['lambda'] is not None else '—'}  "
        f"seq={seq_len}  batch_logico={logical_batch}  "
        f"batch_forward={forward_batch}  device={device}  "
        f"passos_treino={record['steps']}",
        flush=True,
    )
    for split in args.splits:
        packed = _cached_pack(packed_cache, config, split, seq_len)
        print(
            f"  {split}: documentos={packed.documents}  "
            f"sequências={int(packed.sequences.size(0))}  "
            f"descartados={packed.tokens_discarded}",
            flush=True,
        )
        record["splits"][split] = score_split(
            model,
            packed,
            kind=kind,
            device=device,
            logical_batch=logical_batch,
            forward_batch=forward_batch,
            max_logical=args.max_batches,
            log_every=args.log_every,
            emulator=emulator,
            config=config,
            lambda_weight=lambda_weight,
            residency=residency,
        )
    record["dose_response"] = measure_dose_response(
        kind=kind,
        model=model,
        emulator=emulator,
        config=config,
        args=args,
        device=device,
        logical_batch=logical_batch,
        forward_batch=forward_batch,
        lambda_weight=lambda_weight,
        packed_cache=packed_cache,
        seq_len=seq_len,
        residency=residency,
    )
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def _cached_pack(
    cache: dict[tuple[str, int], PackedSplit],
    config: ExperimentConfig,
    split: str,
    seq_len: int,
) -> PackedSplit:
    key = (split, seq_len)
    if key not in cache:
        try:
            cache[key] = pack_split(config, split, seq_len)
        except SystemExit:
            raise
        except Exception as exc:
            die(
                EXIT_USAGE,
                f"Não consegui ler o WikiText-2 {split}. Confira a rede, corpus.cache_dir "
                f"e corpus.paths.WikiText-2. Detalhe: {exc}",
            )
    packed = cache[key]
    if int(packed.sequences.size(0)) < 1:
        die(
            EXIT_USAGE,
            f"WikiText-2 {split} não fechou sequência de seq_len={seq_len}",
        )
    return packed


def measure_dose_response(
    *,
    kind: str,
    model,
    emulator,
    config: ExperimentConfig,
    args: argparse.Namespace,
    device: torch.device,
    logical_batch: int,
    forward_batch: int,
    lambda_weight: float | None,
    packed_cache: dict[tuple[str, int], PackedSplit],
    seq_len: int,
    residency: ExpertResidency | None = None,
) -> dict[str, Any]:
    """Curva ``eval.B_mult`` no checkpoint já carregado. Não retreina.

    O denso não tem emulador: o registro é ``n/a`` e ``set_hot_budget``
    não é chamado. No MoE, cada par (multiplicador, split) passa por
    ``prepare_split_budget`` e o mesmo ``score_split`` da referência,
    com ``reset_lru=False``. A parede e os bytes desse passe saem dali.
    """

    if kind == KIND_DENSE:
        return dose_response.dense_record(config)
    if kind != KIND_MOE:
        die(EXIT_CRASH, f"dose–resposta não conhece model_kind={kind!r}")
    if emulator is None or lambda_weight is None:
        die(EXIT_CRASH, "MoE sem emulador na dose–resposta")
    points = dose_response.budget_points(config)
    if not points:
        die(EXIT_USAGE, "eval.B_mult vazio; a dose–resposta não tem ponto")
    reference = points[0].reference_slots
    budgets: list[dict[str, Any]] = []
    for point in points:
        print(
            f"  dose–resposta {point.multiplier:g}×B "
            f"→ {point.hot_slots} vagas quentes "
            f"(referência {reference})",
            flush=True,
        )
        split_rows: dict[str, Any] = {}
        for split in args.splits:
            packed = _cached_pack(packed_cache, config, split, seq_len)
            stats = dose_response.prepare_split_budget(
                emulator, point.reference_slots, point.hot_slots
            )
            row = score_split(
                model,
                packed,
                kind=kind,
                device=device,
                logical_batch=logical_batch,
                forward_batch=forward_batch,
                max_logical=args.max_batches,
                log_every=args.log_every,
                emulator=emulator,
                config=config,
                lambda_weight=lambda_weight,
                reset_lru=False,
                residency=residency,
            )
            row["budget_reassign"] = {
                "swaps": int(stats.swaps),
                "bytes_moved": int(stats.bytes_moved),
                "promotions": int(stats.promotions),
                "demotions": int(stats.demotions),
            }
            split_rows[split] = row
        budgets.append(
            {
                "multiplier": point.multiplier,
                "hot_slots": point.hot_slots,
                "yaml_reference": point.is_reference,
                "splits": split_rows,
            }
        )
    return {
        "status": "ok",
        "reason": None,
        "reference_hot_slots": reference,
        "method": "reset_lru_at_yaml_b_then_set_hot_budget",
        "retrained": False,
        "budgets": budgets,
    }


def _canonical_lambda(config: ExperimentConfig, value: float) -> float | None:
    for item in config.loss.lambda_grid:
        if same_lambda(item, value):
            return float(item)
    return None


def missing_record(family: Family) -> dict[str, Any]:
    return {
        "name": family.name,
        "status": "missing",
        "path": str(family.path),
        "expected_model_kind": family.expected_kind,
        "expected_lambda": family.expected_lambda,
        "model_kind": None,
        "memory_path": None,
        "lambda": None,
        "context_matches_checkpoint": False,
        "splits": {},
        "dose_response": dose_response.missing_record(),
    }


def run_is_complete(run: dict[str, Any]) -> bool:
    if run.get("status") != "ok":
        return False
    if run.get("context_matches_checkpoint") is not True:
        return False
    if run.get("model_kind") == KIND_MOE and run.get("lambda_in_grid") is not True:
        return False
    splits = run.get("splits") or {}
    if "valid" not in splits or "test" not in splits:
        return False
    for row in splits.values():
        if row.get("truncated"):
            return False
        if not math.isfinite(float(row["nll_nats"])):
            return False
    return True


def protocol_is_complete(runs: list[dict[str, Any]]) -> bool:
    if not runs:
        return False
    return all(run_is_complete(run) for run in runs)


def build_payload(
    config: ExperimentConfig,
    args: argparse.Namespace,
    runs: list[dict[str, Any]],
    *,
    interrupted: bool,
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "scope": "request",
        "metric_note": METRIC_NOTE,
        "perplexity_definition": "exp(nll_nats)",
        "nll_definition": "task_cross_entropy",
        "bits_per_byte_definition": "(nll_nats_sum / ln(2)) / utf8_bytes",
        "protocol_complete": (not interrupted) and protocol_is_complete(runs),
        "interrupted": interrupted,
        "corpus": "WikiText-2",
        "source": source_label(config, "WikiText-2"),
        "dtype": "fp32",
        "torch_compile": False,
        "tokenizer": "gpt2",
        "packing": "non_overlapping_seq_len",
        "residency": "lru_reset_per_split",
        "dose_response": dose_response.curve_was_measured(runs),
        "retrained": False,
        "B_mult": [float(multiplier) for multiplier in config.eval.B_mult],
        "B_reference_slots": config.memory.hot_slots(config.model.E),
        "wall_note": WALL_NOTE,
        "splits": list(args.splits),
        "max_logical_batches": args.max_batches,
        "device": args.device,
        "config": str(config.path),
        "runs": runs,
    }


def standalone(payload_meta: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload_meta)
    body["scope"] = "run"
    body["protocol_complete"] = run_is_complete(run)
    body["interrupted"] = False
    body["runs"] = [run]
    return body


def incomplete_reason(runs: list[dict[str, Any]], args: argparse.Namespace) -> str:
    reasons: list[str] = []
    missing = [run["name"] for run in runs if run.get("status") == "missing"]
    if missing:
        reasons.append("faltam " + ", ".join(missing))
    if any(
        row.get("truncated")
        for run in runs
        if run.get("status") == "ok"
        for row in (run.get("splits") or {}).values()
    ):
        reasons.append("split cortado por --max-batches")
    if set(args.splits) != set(REPORT_SPLITS):
        reasons.append("faltou valid ou test")
    if any(run.get("context_matches_checkpoint") is False for run in runs if run.get("status") == "ok"):
        reasons.append("seq_len diferente do checkpoint")
    if any(
        run.get("model_kind") == KIND_MOE and run.get("lambda_in_grid") is not True
        for run in runs
        if run.get("status") == "ok"
    ):
        reasons.append("λ fora da grade")
    if not reasons:
        reasons.append("corrida interrompida ou NLL não finita")
    return (
        "hold-out incompleto ("
        + "; ".join(reasons)
        + "). O JSON foi gravado, mas não é o número de relatório."
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(json_ready(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            die(EXIT_CRASH, "JSON com valor não finito")
        return float(value)
    if isinstance(value, str):
        return value
    if isinstance(value, Path):
        return str(value)
    die(EXIT_CRASH, f"JSON não serializa {type(value).__name__}")
    raise AssertionError("die não retorna")


def print_table(payload: dict[str, Any]) -> None:
    print("Hold-out WikiText-2 — perplexidade de token no B do YAML", flush=True)
    for run in payload["runs"]:
        if run.get("status") != "ok":
            print(f"{run['name']}: {run.get('status')}  {run.get('path')}", flush=True)
            continue
        pieces = []
        for split, row in run["splits"].items():
            ppl = "overflow" if row["perplexity"] is None else f"{row['perplexity']:.3f}"
            pieces.append(
                f"{split} ppl={ppl} nll={row['nll_nats']:.4f} "
                f"bpb={_fmt(row['bits_per_byte'])} tokens={row['tokens_scored']} "
                f"{_wall_fmt(row)}"
            )
        lam = "λ=—" if run["lambda"] is None else f"λ={run['lambda']}"
        print(f"{run['name']}  {run['model_kind']}  {lam}  " + "  ".join(pieces), flush=True)
    if not payload["protocol_complete"]:
        print("protocolo incompleto: não use este JSON como número de relatório", flush=True)
    print(
        "parede de relatório: só com wall_report_grade=true (GTX 1060, CUDA, "
        "batch do checkpoint, split inteiro). CPU deixa os ms null. "
        "A curva de B reusa prepare_split_budget e o mesmo forward.",
        flush=True,
    )


def _fmt(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.4f}"


def _wall_fmt(row: dict[str, Any]) -> str:
    if row.get("wall_clock") != WALL_CLOCK_CUDA or row.get("wall_p50_ms") is None:
        return "wall=indisponível"
    mark = "1060" if row.get("wall_report_grade") else "não-relatório"
    return (
        f"wall_p50={row['wall_p50_ms']:.3f}ms "
        f"wall_p95={row['wall_p95_ms']:.3f}ms ({mark})"
    )


def _brief(value: Any) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        text = repr(value)
    if len(text) > 500:
        return text[:500] + "…"
    return text


def as_int(value: Any, label: str) -> int:
    if isinstance(value, torch.Tensor) and value.ndim == 0:
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        die(EXIT_USAGE, f"{label} não é inteiro")
    if isinstance(value, float) and not value.is_integer():
        die(EXIT_USAGE, f"{label} não é inteiro")
    return int(value)


def as_float(value: Any, label: str) -> float:
    if isinstance(value, torch.Tensor) and value.ndim == 0:
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        die(EXIT_USAGE, f"{label} não é numérico")
    number = float(value)
    if not math.isfinite(number):
        die(EXIT_USAGE, f"{label} não é finito")
    return number


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        die(EXIT_USAGE, "pediu CUDA, mas torch.cuda não está disponível. O padrão deste script é cpu.")
    try:
        return torch.device(name)
    except (TypeError, ValueError) as exc:
        die(EXIT_USAGE, f"device inválido: {exc}")
    raise AssertionError("die não retorna")


def _write_outputs(
    args: argparse.Namespace,
    config: ExperimentConfig,
    families: list[Family],
    runs: list[dict[str, Any]],
    *,
    interrupted: bool,
) -> list[Path]:
    payload = build_payload(config, args, runs, interrupted=interrupted)
    written: list[Path] = []
    meta = {
        key: payload[key]
        for key in (
            "protocol",
            "metric_note",
            "perplexity_definition",
            "nll_definition",
            "bits_per_byte_definition",
            "corpus",
            "source",
            "dtype",
            "torch_compile",
            "tokenizer",
            "packing",
            "residency",
            "dose_response",
            "retrained",
            "B_mult",
            "B_reference_slots",
            "wall_note",
            "splits",
            "max_logical_batches",
            "device",
            "config",
        )
    }
    by_name = {run["name"]: run for run in runs}
    for family in families:
        run = by_name.get(family.name)
        if run is None or run.get("status") != "ok":
            continue
        path = family.path.parent / "holdout.json"
        write_json(path, standalone(meta, run))
        written.append(path)
    summary = summary_path_for(args, len(families))
    if summary is not None:
        write_json(summary, payload)
        written.append(summary)
    print_table(payload)
    dose_payload = dose_response.build_payload(
        config,
        runs,
        splits=tuple(args.splits),
        max_logical_batches=args.max_batches,
    )
    dose_json, dose_table = dose_artifact_paths(args, families)
    write_json(dose_json, dose_payload)
    dose_table.parent.mkdir(parents=True, exist_ok=True)
    table_text = dose_response.render_markdown(dose_payload)
    temporary = dose_table.with_suffix(dose_table.suffix + ".tmp")
    temporary.write_text(table_text, encoding="utf-8")
    os.replace(temporary, dose_table)
    written.append(dose_json)
    written.append(dose_table)
    print(table_text, flush=True)
    for path in written:
        label = "table" if path.suffix == ".md" else "json"
        print(f"{label}={path}", flush=True)
    return written


def dose_artifact_paths(args: argparse.Namespace, families: list[Family]) -> tuple[Path, Path]:
    """JSON da curva e a tabela markdown, ao lado do resumo ou do checkpoint."""

    summary = summary_path_for(args, len(families))
    if summary is not None:
        directory = Path(summary).parent
    elif len(families) == 1:
        directory = families[0].path.parent
    else:
        directory = Path(args.outputs) / "holdout"
    return directory / "dose_response.json", directory / "dose_response.md"


def _main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    if config.corpus.report != "WikiText-2":
        die(EXIT_USAGE, f"corpus.report={config.corpus.report!r}; o hold-out é WikiText-2")
    if args.explicit:
        families = families_from_checkpoints(list(args.checkpoint))
        missing = [family for family in families if not family.path.is_file()]
        if missing:
            die(
                EXIT_USAGE,
                "não achei checkpoint: " + ", ".join(str(family.path) for family in missing),
            )
    else:
        families = discover_families(args.outputs, config)
    device = resolve_device(args.device)
    print(
        "Hold-out WikiText-2 (não treina; depois do B do YAML, dose–resposta em eval.B_mult)",
        flush=True,
    )
    print(f"fonte: {source_label(config, 'WikiText-2')}", flush=True)
    print(
        f"dtype=fp32  torch.compile=não  device={device}  splits={','.join(args.splits)}",
        flush=True,
    )
    if device.type == "cuda":
        gpu = cuda_gpu_name(device)
        grade = "relatório" if _is_gtx_1060(gpu) else "não é a GTX 1060 — não é coluna de relatório"
        print(f"parede=cuda  gpu={gpu}  {grade}", flush=True)
    else:
        print(
            "parede=indisponível (CPU). wall_p50_ms e wall_p95_ms ficam null. "
            "A coluna de relatório exige a GTX 1060 com CUDA.",
            flush=True,
        )
    runs: list[dict[str, Any]] = []
    packed_cache: dict[tuple[str, int], PackedSplit] = {}
    try:
        for family in families:
            if not family.path.is_file():
                print(f"ausente: {family.name}  {family.path}", file=sys.stderr, flush=True)
                runs.append(missing_record(family))
                continue
            runs.append(evaluate_family(family, config, args, device, packed_cache))
    except SystemExit:
        if runs:
            _write_outputs(args, config, families, runs, interrupted=True)
        raise
    _write_outputs(args, config, families, runs, interrupted=False)
    if not protocol_is_complete(runs):
        die(EXIT_INCOMPLETE, incomplete_reason(runs, args))


def main(argv: list[str] | None = None) -> None:
    try:
        _main(argv)
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
