#!/usr/bin/env python3
"""Correlação do proxy ``C_mem`` com bytes/trocas do emulador, e entropia.

Pós-processamento. Não treina, não lê checkpoint e não importa o modelo.
CPU apenas. Lê ``train.jsonl`` já gravado pelo MoE (grade ``λ``) e escreve
figuras SVG, CSV e um JSON de resumo.

O proxy e o emulador não estão na mesma escala: ``c_mem`` é custo da loss,
``bytes_moved`` e ``swaps`` são contagens do LRU por passo. Pearson mede o
alinhamento linear; Spearman (posto) mede se a ordem acompanha. O protocolo
pede correlacionar ou, no mínimo, ordenar. Os dois entram no resumo. A
entropia do roteador é curva por passo (diagnóstico de colapso).

Uso, a partir da raiz, com os jsonl da corrida WikiText-2::

    python3 experiments/plot_proxy_emulator.py
    python3 experiments/plot_proxy_emulator.py --root outputs/wikitext
    python3 experiments/plot_proxy_emulator.py \\
        --log outputs/wikitext/seed-0/lambda-0/train.jsonl \\
        --log outputs/wikitext/seed-0/lambda-0.5/train.jsonl \\
        --log outputs/wikitext/seed-0/lambda-1/train.jsonl

Sem ``--log``, o padrão junta ``seed-*/lambda-*/train.jsonl`` e o layout
antigo ``lambda-*/train.jsonl``. Se o mesmo λ da semente 0 existir nos dois,
fica ``seed-0`` (como a grade). ``--log`` explícito não sofre esse corte.

A saída cai em ``outputs/proxy-emulator/`` (gitignored, como o resto de
``outputs/``). O baseline denso tem ``c_mem`` nulo e não tem trocas: essas
linhas saem da conta. Evento que não é ``step`` (``nonfinite``) também.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from seed_layout import is_seed_dirname

EXIT_OK = 0
EXIT_USAGE = 1

DEFAULT_ROOT = ROOT / "outputs" / "wikitext"
DEFAULT_OUT = ROOT / "outputs" / "proxy-emulator"

# Okabe–Ito. Três λ cabem sem depender de cor só.
_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9")

_INTERPRETATION = (
    "Pearson mede alinhamento linear entre o proxy c_mem e a contagem do "
    "emulador no mesmo passo. Spearman mede se a ordem de c_mem acompanha "
    "bytes_moved e swaps. As escalas diferem (custo da loss contra bytes e "
    "trocas inteiros), então o posto é o diagnóstico de alinhamento. "
    "Entropia cai ao longo dos passos quando o roteador concentra massa; "
    "isso pode ser colapso em poucos experts quentes, não economia útil."
)


def lambda_tag(value: float) -> str:
    """Mesma etiqueta de ``train_wikitext.lambda_tag`` (sem importar torch)."""

    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text if text else "0"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Correlaciona c_mem do jsonl de treino com bytes/trocas do "
            "emulador e desenha a entropia do roteador. Não treina."
        )
    )
    parser.add_argument(
        "--log",
        action="append",
        type=Path,
        default=None,
        help=(
            "train.jsonl do MoE, ou a pasta que o contém. "
            "Repita para cada λ. Sem isto, procura em --root."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help=(
            "Pasta da grade. Procura seed-*/lambda-*/train.jsonl e o "
            "layout antigo lambda-*/train.jsonl. "
            f"Padrão, se --log não vier: {DEFAULT_ROOT}."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Pasta das figuras e do resumo. Padrão: {DEFAULT_OUT}.",
    )
    return parser.parse_args(argv)


def partition_logs(root: Path) -> tuple[list[Path], list[Path]]:
    """Logs do layout novo e o legado que ``seed-0`` ainda não cobriu.

    O primeiro item são os jsonl usados, em ordem de caminho. O segundo são
    os ``lambda-*/train.jsonl`` antigos ignorados porque
    ``seed-0/lambda-<λ>/train.jsonl`` também existe. ``--log`` explícito não
    passa por aqui.
    """

    if not root.is_dir():
        return [], []
    seeded = [
        path
        for path in root.glob("seed-*/lambda-*/train.jsonl")
        if path.is_file() and _is_seed_lambda_log(path)
    ]
    covered = {path.parent.name for path in seeded if path.parent.parent.name == "seed-0"}
    legacy = [
        path
        for path in root.glob("lambda-*/train.jsonl")
        if path.is_file() and path.parent.name.startswith("lambda-")
    ]
    kept: list[Path] = []
    skipped: list[Path] = []
    for path in legacy:
        if path.parent.name in covered:
            skipped.append(path)
        else:
            kept.append(path)
    return sorted(seeded + kept), sorted(skipped)


def discover_logs(root: Path) -> list[Path]:
    """``seed-*/lambda-*/train.jsonl`` e o layout antigo ainda não coberto."""

    used, _skipped = partition_logs(root)
    return used


def _is_seed_lambda_log(path: Path) -> bool:
    lam_dir = path.parent.name
    seed_dir = path.parent.parent.name
    return lam_dir.startswith("lambda-") and is_seed_dirname(seed_dir)


def resolve_log(path: Path) -> Path:
    if path.is_dir():
        candidate = path / "train.jsonl"
        if not candidate.is_file():
            raise FileNotFoundError(f"não há train.jsonl em {path}")
        return candidate
    if not path.is_file():
        raise FileNotFoundError(f"jsonl inexistente: {path}")
    return path


def _finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
    else:
        return None
    if not math.isfinite(number):
        return None
    return number


def load_log(path: Path) -> dict[str, Any]:
    """Passos utilizáveis de um jsonl.

    Linha ilegível conta em ``bad_lines``. Linha JSON que não é passo MoE
    (evento diferente de ``step``, ou ``c_mem`` / ``bytes_moved`` / ``swaps``
    ausentes ou não finitos) conta em ``skipped_rows``. O baseline denso cai
    aqui: ``c_mem`` é null e não há trocas.
    """

    rows: list[dict[str, Any]] = []
    bad_lines = 0
    skipped_rows = 0
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            bad_lines += 1
            continue
        if not isinstance(record, dict):
            skipped_rows += 1
            continue
        event = record.get("event")
        if event is not None and event != "step":
            skipped_rows += 1
            continue
        c_mem = _finite_number(record.get("c_mem"))
        bytes_moved = _finite_number(record.get("bytes_moved"))
        swaps = _finite_number(record.get("swaps"))
        if c_mem is None or bytes_moved is None or swaps is None:
            skipped_rows += 1
            continue
        step = _finite_number(record.get("step"))
        lam = _finite_number(record.get("lambda"))
        entropy = _finite_number(record.get("entropy"))
        rows.append(
            {
                "step": int(step) if step is not None else len(rows) + 1,
                "lambda": lam,
                "c_mem": c_mem,
                "bytes_moved": bytes_moved,
                "swaps": swaps,
                "entropy": entropy,
            }
        )
    return {
        "path": path,
        "rows": rows,
        "bad_lines": bad_lines,
        "skipped_rows": skipped_rows,
    }


def label_for(path: Path, rows: list[dict[str, Any]]) -> str:
    parent = path.parent.name
    grand = path.parent.parent.name
    if parent.startswith("lambda-") and is_seed_dirname(grand):
        return f"{grand}/{parent}"
    if parent.startswith("lambda-"):
        return parent
    lambdas: list[float] = []
    for row in rows:
        lam = row.get("lambda")
        if lam is not None and lam not in lambdas:
            lambdas.append(lam)
    if len(lambdas) == 1:
        return f"lambda-{lambda_tag(lambdas[0])}"
    return parent or path.name


def unique_labels(labels: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    seen: dict[str, int] = {}
    unique: list[str] = []
    for label in labels:
        if counts[label] == 1:
            unique.append(label)
            continue
        seen[label] = seen.get(label, 0) + 1
        unique.append(f"{label}#{seen[label]}")
    return unique


def average_ranks(values: list[float]) -> list[float]:
    """Posto médio (1-based). Empate divide a posição."""

    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and values[order[end + 1]] == values[order[start]]:
            end += 1
        rank = (start + end) / 2.0 + 1.0
        for cursor in range(start, end + 1):
            ranks[order[cursor]] = rank
        start = end + 1
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    """Correlação de Pearson. ``None`` se n<2 ou se alguma série é constante."""

    n = len(xs)
    if n != len(ys) or n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((item - mean_x) ** 2 for item in xs)
    var_y = sum((item - mean_y) ** 2 for item in ys)
    if var_x == 0.0 or var_y == 0.0:
        return None
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    return cov / math.sqrt(var_x * var_y)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Correlação de Spearman (Pearson dos postos médios)."""

    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return pearson(average_ranks(xs), average_ranks(ys))


def _variance_note(xs: list[float], ys: list[float], y_name: str) -> str | None:
    n = len(xs)
    if n < 2:
        return "n<2"
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    if sum((item - mean_x) ** 2 for item in xs) == 0.0:
        return "variância zero em c_mem"
    if sum((item - mean_y) ** 2 for item in ys) == 0.0:
        return f"variância zero em {y_name}"
    return None


def correlate_pair(xs: list[float], ys: list[float], y_name: str) -> dict[str, Any]:
    note = _variance_note(xs, ys, y_name)
    return {
        "n": len(xs),
        "pearson": None if note else pearson(xs, ys),
        "spearman": None if note else spearman(xs, ys),
        "note": note,
    }


def correlate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    c_mem = [row["c_mem"] for row in rows]
    return {
        "n_steps": len(rows),
        "c_mem_vs_bytes": correlate_pair(
            c_mem, [row["bytes_moved"] for row in rows], "bytes_moved"
        ),
        "c_mem_vs_swaps": correlate_pair(c_mem, [row["swaps"] for row in rows], "swaps"),
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _lambda_of(rows: list[dict[str, Any]]) -> float | None:
    present = [row["lambda"] for row in rows if row["lambda"] is not None]
    if not present:
        return None
    if all(math.isclose(item, present[0], rel_tol=0.0, abs_tol=1e-9) for item in present):
        return present[0]
    return None


def summarize_run(loaded: dict[str, Any], label: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = loaded["rows"]
    entropies = [row["entropy"] for row in rows if row["entropy"] is not None]
    corr = correlate_rows(rows)
    reason = None
    if not rows:
        reason = "nenhum passo MoE com c_mem, bytes_moved e swaps finitos"
    return {
        "label": label,
        "path": str(loaded["path"]),
        "lambda": _lambda_of(rows),
        "n_steps": len(rows),
        "bad_lines": loaded["bad_lines"],
        "skipped_rows": loaded["skipped_rows"],
        "skip_reason": reason,
        "mean_c_mem": _mean([row["c_mem"] for row in rows]),
        "mean_bytes_moved": _mean([row["bytes_moved"] for row in rows]),
        "mean_swaps": _mean([row["swaps"] for row in rows]),
        "entropy_first": entropies[0] if entropies else None,
        "entropy_last": entropies[-1] if entropies else None,
        "entropy_min": min(entropies) if entropies else None,
        "entropy_max": max(entropies) if entropies else None,
        "n_entropy": len(entropies),
        **corr,
    }


def _fmt_corr(value: float | None) -> str:
    if value is None:
        return "indefinida"
    return f"{value:.4f}"


def _fmt_num(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.10g}"


def write_steps_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "step", "lambda", "c_mem", "bytes_moved", "swaps", "entropy"])
        for run in runs:
            for row in run["rows"]:
                writer.writerow(
                    [
                        run["label"],
                        row["step"],
                        _fmt_num(row["lambda"]),
                        _fmt_num(row["c_mem"]),
                        _fmt_num(row["bytes_moved"]),
                        _fmt_num(row["swaps"]),
                        _fmt_num(row["entropy"]),
                    ]
                )


def write_correlations_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "pair", "n", "pearson", "spearman", "note"])
        for summary in summaries:
            for pair in ("c_mem_vs_bytes", "c_mem_vs_swaps"):
                block = summary[pair]
                writer.writerow(
                    [
                        summary["label"],
                        pair,
                        block["n"],
                        _fmt_num(block["pearson"]),
                        _fmt_num(block["spearman"]),
                        block["note"] or "",
                    ]
                )


def _xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _tick_label(value: float) -> str:
    if value == 0:
        return "0"
    absolute = abs(value)
    if absolute >= 10000 or absolute < 0.01:
        return f"{value:.3g}"
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text or "0"


def _padded_range(values: list[float]) -> tuple[float, float]:
    lo = min(values)
    hi = max(values)
    if hi == lo:
        pad = 1.0 if lo == 0 else abs(lo) * 0.05
        return lo - pad, hi + pad
    pad = (hi - lo) * 0.08
    return lo - pad, hi + pad


def _ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    """Passos 1/2/5 × 10^k dentro do intervalo. Evita rótulo 0,815 num eixo."""

    if count < 2 or hi == lo:
        return [lo]
    span = hi - lo
    raw = span / (count - 1)
    if raw <= 0:
        return [lo]
    magnitude = 10 ** math.floor(math.log10(raw))
    residual = raw / magnitude
    if residual <= 1.5:
        step = magnitude
    elif residual <= 3:
        step = 2 * magnitude
    elif residual <= 7:
        step = 5 * magnitude
    else:
        step = 10 * magnitude
    start = math.ceil((lo / step) - 1e-9) * step
    ticks: list[float] = []
    value = start
    for _ in range(12):
        if value > hi + step * 1e-6:
            break
        if value >= lo - step * 1e-6:
            ticks.append(0.0 if abs(value) < step * 1e-6 else value)
        value += step
    if len(ticks) < 2:
        return [lo, hi]
    return ticks


def render_svg(
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    series: list[dict[str, Any]],
    kind: str,
    footnote: str = "",
) -> str:
    """Figura SVG. ``series`` traz ``label``, ``color`` e ``points`` (x, y).

    ``kind`` é ``line`` (entropia por passo) ou ``scatter`` (proxy × emulador).
    Sem matplotlib: o jsonl de treino vira figura com a biblioteca padrão.
    """

    if kind not in {"line", "scatter"}:
        raise ValueError(f"kind desconhecido: {kind}")
    width, height = 780, 468
    left, right, top, bottom = 96, 188, 40, 78
    plot_w = width - left - right
    plot_h = height - top - bottom
    points = [point for item in series for point in item["points"]]
    if not points:
        lo_x, hi_x, lo_y, hi_y = 0.0, 1.0, 0.0, 1.0
    else:
        lo_x, hi_x = _padded_range([point[0] for point in points])
        lo_y, hi_y = _padded_range([point[1] for point in points])

    def sx(value: float) -> float:
        return left + (value - lo_x) / (hi_x - lo_x) * plot_w

    def sy(value: float) -> float:
        return top + (hi_y - value) / (hi_y - lo_y) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        f"<title>{_xml(title)}</title>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left}" y="24" font-family="sans-serif" font-size="15" '
        f'fill="#1a1a1a">{_xml(title)}</text>',
    ]
    for tick in _ticks(lo_y, hi_y):
        y = sy(tick)
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" '
            'stroke="#e6e6e6" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" '
            'font-family="sans-serif" font-size="11" fill="#333">'
            f"{_xml(_tick_label(tick))}</text>"
        )
    for tick in _ticks(lo_x, hi_x):
        x = sx(tick)
        parts.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" '
            'stroke="#f3f3f3" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.2f}" y="{top + plot_h + 18}" text-anchor="middle" '
            'font-family="sans-serif" font-size="11" fill="#333">'
            f"{_xml(_tick_label(tick))}</text>"
        )
    parts.append(
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" '
        'fill="none" stroke="#222" stroke-width="1"/>'
    )
    parts.append(
        f'<text x="{left + plot_w / 2}" y="{height - 32}" text-anchor="middle" '
        'font-family="sans-serif" font-size="12" fill="#1a1a1a">'
        f"{_xml(xlabel)}</text>"
    )
    parts.append(
        f'<text x="18" y="{top + plot_h / 2}" text-anchor="middle" '
        f'transform="rotate(-90 18 {top + plot_h / 2})" '
        'font-family="sans-serif" font-size="12" fill="#1a1a1a">'
        f"{_xml(ylabel)}</text>"
    )
    for item in series:
        color = item["color"]
        coords = item["points"]
        if kind == "line" and len(coords) >= 2:
            poly = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in coords)
            parts.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="1.8" '
                f'points="{poly}"/>'
            )
        for x, y in coords:
            radius = 2.4 if kind == "line" else 3.4
            parts.append(
                f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="{radius}" '
                f'fill="{color}" fill-opacity="0.9"/>'
            )
    legend_x = left + plot_w + 16
    parts.append(
        f'<text x="{legend_x}" y="{top + 4}" font-family="sans-serif" '
        'font-size="12" fill="#1a1a1a">série</text>'
    )
    for index, item in enumerate(series):
        y = top + 26 + index * 36
        parts.append(f'<circle cx="{legend_x + 6}" cy="{y}" r="5" fill="{item["color"]}"/>')
        parts.append(
            f'<text x="{legend_x + 18}" y="{y + 4}" font-family="sans-serif" '
            f'font-size="11" fill="#1a1a1a">{_xml(item["label"])}</text>'
        )
        detail = item.get("detail") or ""
        if detail:
            parts.append(
                f'<text x="{legend_x + 18}" y="{y + 18}" font-family="sans-serif" '
                f'font-size="10" fill="#555">{_xml(detail)}</text>'
            )
    if footnote:
        parts.append(
            f'<text x="{left}" y="{height - 12}" font-family="sans-serif" '
            f'font-size="10" fill="#555">{_xml(footnote)}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _series(
    runs: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    *,
    detail_of,
) -> list[dict[str, Any]]:
    series = []
    for index, run in enumerate(runs):
        points = []
        for row in run["rows"]:
            x = row[x_key]
            y = row[y_key]
            if x is None or y is None:
                continue
            points.append((float(x), float(y)))
        if not points:
            continue
        series.append(
            {
                "label": run["label"],
                "color": _COLORS[index % len(_COLORS)],
                "points": points,
                "detail": detail_of(run),
            }
        )
    return series


def _pair_detail(summary: dict[str, Any], key: str) -> str:
    block = summary[key]
    return f"r={_fmt_corr(block['pearson'])}  ρ={_fmt_corr(block['spearman'])}"


def write_figures(out_dir: Path, runs: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> None:
    by_label = {item["label"]: item for item in summaries}
    entropy = _series(
        runs,
        "step",
        "entropy",
        detail_of=lambda run: "",
    )
    for item in entropy:
        summary = by_label[item["label"]]
        first = summary["entropy_first"]
        last = summary["entropy_last"]
        if first is not None and last is not None:
            item["detail"] = f"{first:.3f} → {last:.3f}"
    bytes_series = _series(
        runs,
        "c_mem",
        "bytes_moved",
        detail_of=lambda run: _pair_detail(by_label[run["label"]], "c_mem_vs_bytes"),
    )
    swaps_series = _series(
        runs,
        "c_mem",
        "swaps",
        detail_of=lambda run: _pair_detail(by_label[run["label"]], "c_mem_vs_swaps"),
    )
    figures = {
        "entropy.svg": render_svg(
            title="Entropia do roteador por passo",
            xlabel="passo",
            ylabel="entropia",
            series=entropy,
            kind="line",
            footnote="Média das camadas, como no jsonl de treino. Queda forte sugere colapso.",
        ),
        "c_mem_vs_bytes.svg": render_svg(
            title="Proxy C_mem × bytes do emulador",
            xlabel="c_mem",
            ylabel="bytes_moved",
            series=bytes_series,
            kind="scatter",
            footnote="Um ponto por passo. r = Pearson, ρ = Spearman.",
        ),
        "c_mem_vs_swaps.svg": render_svg(
            title="Proxy C_mem × trocas do emulador",
            xlabel="c_mem",
            ylabel="swaps",
            series=swaps_series,
            kind="scatter",
            footnote="Um ponto por passo. r = Pearson, ρ = Spearman.",
        ),
    }
    for name, svg in figures.items():
        (out_dir / name).write_text(svg, encoding="utf-8")


def build_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    per_run = [summarize_run(run, run["label"]) for run in runs]
    pooled_rows = [row for run in runs for row in run["rows"]]
    pooled = correlate_rows(pooled_rows)
    pooled["label"] = "pooled"
    return {
        "version": 2,
        "cpu_only": True,
        "trained": False,
        "interpretation": _INTERPRETATION,
        "n_logs": len(runs),
        "n_steps": len(pooled_rows),
        "runs": per_run,
        "pooled": pooled,
    }


def collect_logs(args: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    paths: list[Path] = []
    skipped: list[Path] = []
    if args.log:
        for raw in args.log:
            paths.append(resolve_log(raw))
    root = args.root
    if root is None and not args.log:
        root = DEFAULT_ROOT
    if root is not None:
        found, legacy_skipped = partition_logs(root)
        paths.extend(found)
        skipped.extend(legacy_skipped)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique, skipped


def _print_run(summary: dict[str, Any]) -> None:
    if summary["n_steps"] == 0:
        print(
            f"{summary['label']}: sem passo utilizável "
            f"(linhas ruins={summary['bad_lines']}, ignoradas={summary['skipped_rows']})"
        )
        return
    bytes_block = summary["c_mem_vs_bytes"]
    swaps_block = summary["c_mem_vs_swaps"]
    entropy = ""
    if summary["entropy_first"] is not None and summary["entropy_last"] is not None:
        entropy = (
            f"  entropia {_fmt_corr(summary['entropy_first'])} → "
            f"{_fmt_corr(summary['entropy_last'])}"
        )
    print(
        f"{summary['label']}: n={summary['n_steps']}  "
        f"pearson(c_mem, bytes)={_fmt_corr(bytes_block['pearson'])}  "
        f"spearman={_fmt_corr(bytes_block['spearman'])}  "
        f"pearson(c_mem, trocas)={_fmt_corr(swaps_block['pearson'])}  "
        f"spearman={_fmt_corr(swaps_block['spearman'])}"
        f"{entropy}"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        paths, skipped_legacy = collect_logs(args)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE
    if not paths:
        looked = args.root if args.root is not None else DEFAULT_ROOT
        print(
            f"nenhum train.jsonl de MoE em {looked}/seed-*/lambda-*/train.jsonl "
            f"nem em {looked}/lambda-*/train.jsonl",
            file=sys.stderr,
        )
        return EXIT_USAGE

    loaded = [load_log(path) for path in paths]
    labels = unique_labels([label_for(item["path"], item["rows"]) for item in loaded])
    for item, label in zip(loaded, labels, strict=True):
        item["label"] = label

    summary = build_summary(loaded)
    summary["skipped_legacy"] = [str(path) for path in skipped_legacy]
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_steps_csv(out_dir / "steps.csv", loaded)
    write_correlations_csv(
        out_dir / "correlations.csv",
        [*summary["runs"], {**summary["pooled"], "label": "pooled"}],
    )
    write_figures(out_dir, loaded, summary["runs"])

    for path in skipped_legacy:
        print(f"layout antigo ignorado (vale seed-0): {path}")
    for run_summary in summary["runs"]:
        _print_run(run_summary)
    pooled = summary["pooled"]
    print(
        "pooled: n="
        f"{pooled['n_steps']}  "
        f"pearson(c_mem, bytes)={_fmt_corr(pooled['c_mem_vs_bytes']['pearson'])}  "
        f"spearman={_fmt_corr(pooled['c_mem_vs_bytes']['spearman'])}  "
        f"pearson(c_mem, trocas)={_fmt_corr(pooled['c_mem_vs_swaps']['pearson'])}  "
        f"spearman={_fmt_corr(pooled['c_mem_vs_swaps']['spearman'])}"
    )
    print(f"figuras em {out_dir}")
    if summary["n_steps"] == 0:
        print("nenhum passo MoE utilizável", file=sys.stderr)
        return EXIT_USAGE
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
