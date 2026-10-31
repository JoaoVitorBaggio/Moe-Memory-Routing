"""Curva de dose–resposta do hold-out. Dono: ``experiments/run_holdout_eval.py``.

Não treina e não lê checkpoint. O script de hold-out mede primeiro o ``B``
do YAML. Para cada multiplicador de ``eval.B_mult``, cada split recomeça
o LRU nesse ``B`` e ``ResidencyEmulator.set_hot_budget`` aplica o teto.
Esse ``set_hot_budget`` reatribui pelo LRU fresco e não apaga o carimbo
de uso: não há ``reset`` entre ele e a perplexidade.

O LM denso não tem residência. A curva dele é ``n/a``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from moe_memory.config import ExperimentConfig

DOSE_PROTOCOL = "wikitext-2-dose-response"
DOSE_NOTE = (
    "Perplexidade de hold-out (exp da task_cross_entropy) no checkpoint já "
    "treinado, sem retreino. Cada split recomeça o LRU no B do YAML (reset) "
    "e o multiplicador entra por set_hot_budget, que reatribui pelo LRU "
    "fresco e não apaga o carimbo de uso. Não há reset entre set_hot_budget "
    "e o forward. O bloco splits do hold-out continua sendo a medição de "
    "referência, nesse mesmo B, feita antes da curva. 1×B, quando está em "
    "eval.B_mult, repete esse B por set_hot_budget. O LM denso não tem teto "
    "quente: n/a. Cada split da curva é o score_split do hold-out, então "
    "traz swaps, bytes_moved e a parede desse forward. Número de relatório "
    "da parede: GTX 1060 (wall_report_grade). Em CPU os ms ficam null. "
    "A tabela markdown continua sendo só perplexidade. O teto não tem "
    "outro caminho: prepare_split_budget, sem atribuir hot_slots à parte."
)


@dataclass(frozen=True)
class BudgetPoint:
    multiplier: float
    hot_slots: int
    reference_slots: int

    @property
    def is_reference(self) -> bool:
        return self.hot_slots == self.reference_slots


def budget_points(config: ExperimentConfig) -> tuple[BudgetPoint, ...]:
    """Um ponto por item de ``eval.B_mult``, na ordem do YAML."""

    reference = config.memory.hot_slots(config.model.E)
    points: list[BudgetPoint] = []
    for multiplier in config.eval.B_mult:
        slots = config.memory.scaled_hot_slots(config.model.E, float(multiplier))
        points.append(BudgetPoint(float(multiplier), slots, reference))
    return tuple(points)


def prepare_split_budget(emulator: Any, reference_slots: int, hot_slots: int) -> Any:
    """LRU fresco no ``B`` do YAML e, em seguida, ``set_hot_budget``.

    ``reset`` coloca os experts ``0 .. hot_slots-1`` no quente usando o
    teto corrente. Se o split anterior deixou outro teto, a volta ao ``B``
    do YAML passa por ``set_hot_budget`` — é a API do emulador. O ``reset``
    que vem depois é o recomeço documentado do split, não a medição.

    A medição é o ``set_hot_budget`` seguinte. Ele não zera ``last_used``,
    ``last_count`` nem o tick, e não troca o store físico. Não há segundo
    ``reset``: isso apagaria o carimbo que a troca de teto acabou de
    preservar. Não cria memmap e não mexe em peso.
    """

    if reference_slots < 1 or hot_slots < 1:
        raise ValueError("teto quente precisa ser >= 1")
    if emulator.hot_slots != reference_slots:
        emulator.set_hot_budget(reference_slots)
    emulator.reset()
    if emulator.hot_slots != reference_slots:
        raise RuntimeError(
            f"reset deixou hot_slots={emulator.hot_slots}; o YAML pede {reference_slots}"
        )
    stamps = emulator.last_used
    counts = emulator.last_count
    tick = emulator.tick
    store = getattr(emulator, "_store", None)
    stats = emulator.set_hot_budget(hot_slots)
    if emulator.hot_slots != hot_slots:
        raise RuntimeError(
            f"set_hot_budget deixou hot_slots={emulator.hot_slots}; pedido {hot_slots}"
        )
    if emulator.last_used is not stamps or emulator.last_count is not counts:
        raise RuntimeError("set_hot_budget substituiu o carimbo de uso do LRU")
    if emulator.tick != tick:
        raise RuntimeError("set_hot_budget alterou o tick do LRU")
    if getattr(emulator, "_store", None) is not store:
        raise RuntimeError("set_hot_budget mexeu no store físico")
    return stats


def dense_record(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "status": "n/a",
        "reason": (
            "O LM denso não tem residência nem teto quente. "
            "Não há eixo de orçamento B; set_hot_budget não se aplica."
        ),
        "reference_hot_slots": config.memory.hot_slots(config.model.E),
        "method": None,
        "retrained": False,
        "budgets": None,
    }


def missing_record() -> dict[str, Any]:
    return {
        "status": "missing",
        "reason": "checkpoint ausente; a curva de B não foi medida",
        "reference_hot_slots": None,
        "method": None,
        "retrained": False,
        "budgets": None,
    }


def curve_was_measured(runs: list[dict[str, Any]]) -> bool:
    """Verdadeiro quando algum MoE tem pontos de ``eval.B_mult`` medidos."""

    for run in runs:
        dose = run.get("dose_response")
        if not isinstance(dose, dict) or dose.get("status") != "ok":
            continue
        budgets = dose.get("budgets")
        if isinstance(budgets, list) and budgets:
            return True
    return False


def build_payload(
    config: ExperimentConfig,
    runs: list[dict[str, Any]],
    *,
    splits: tuple[str, ...] | list[str],
    max_logical_batches: int | None,
) -> dict[str, Any]:
    points = [
        {
            "multiplier": point.multiplier,
            "hot_slots": point.hot_slots,
            "yaml_reference": point.is_reference,
        }
        for point in budget_points(config)
    ]
    slim: list[dict[str, Any]] = []
    for run in runs:
        slim.append(
            {
                "name": run.get("name"),
                "status": run.get("status"),
                "model_kind": run.get("model_kind"),
                "lambda": run.get("lambda"),
                "path": run.get("path"),
                "dose_response": run.get("dose_response"),
            }
        )
    return {
        "protocol": DOSE_PROTOCOL,
        "retrained": False,
        "metric": "perplexity",
        "perplexity_definition": "exp(nll_nats)",
        "nll_definition": "task_cross_entropy",
        "residency": "lru_reset_per_split_then_set_hot_budget",
        "dense_axis": "n/a",
        "B_reference_slots": config.memory.hot_slots(config.model.E),
        "B_mult": [point["multiplier"] for point in points],
        "points": points,
        "splits": list(splits),
        "max_logical_batches": max_logical_batches,
        "note": DOSE_NOTE,
        "runs": slim,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    """Tabela PPL × multiplicador de ``B``, um par de linhas por checkpoint."""

    points = list(payload.get("points") or [])
    splits = list(payload.get("splits") or [])
    header = ["checkpoint", "tipo", "λ", "split"]
    header.extend(_column_name(point) for point in points)
    lines = [
        "# Dose–resposta — perplexidade vs multiplicador de B",
        "",
        str(payload.get("note") or DOSE_NOTE),
        "",
        (
            f"B de referência (YAML): {payload.get('B_reference_slots')} vagas quentes. "
            "Sem retreino. Checkpoint denso: n/a. Checkpoint ausente: ausente."
        ),
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    footnote = False
    for run in payload.get("runs") or []:
        dose = run.get("dose_response") if isinstance(run, dict) else None
        if not isinstance(dose, dict):
            dose = {"status": "missing", "budgets": None}
        status = dose.get("status")
        budgets = dose.get("budgets") if status == "ok" else None
        for split in splits:
            cells = [
                str(run.get("name")),
                str(run.get("model_kind") or "—"),
                _fmt_lambda(run.get("lambda")),
                str(split),
            ]
            if status == "ok" and isinstance(budgets, list):
                for index, _point in enumerate(points):
                    cell, starred = _budget_cell(budgets, index, split)
                    footnote = footnote or starred
                    cells.append(cell)
            elif status == "n/a":
                cells.extend(["n/a"] * len(points))
            elif status == "missing":
                cells.extend(["ausente"] * len(points))
            else:
                cells.extend(["—"] * len(points))
            lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    if footnote:
        lines.append(
            "`(cortado)`: split limitado por `max_logical_batches`; não é o número de relatório."
        )
        lines.append("")
    return "\n".join(lines)


def _column_name(point: dict[str, Any]) -> str:
    return f"{_fmt_mult(float(point['multiplier']))}×B ({int(point['hot_slots'])})"


def _budget_cell(budgets: list[dict[str, Any]], index: int, split: str) -> tuple[str, bool]:
    if index >= len(budgets):
        return "—", False
    splits = budgets[index].get("splits") or {}
    row = splits.get(split)
    if not isinstance(row, dict):
        return "—", False
    perplexity = row.get("perplexity")
    if perplexity is None:
        text = "overflow"
    else:
        text = f"{float(perplexity):.4f}"
    truncated = bool(row.get("truncated"))
    if truncated:
        text += " (cortado)"
    return text, truncated


def _fmt_mult(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _fmt_lambda(value: Any) -> str:
    if value is None:
        return "—"
    return _fmt_mult(float(value))
