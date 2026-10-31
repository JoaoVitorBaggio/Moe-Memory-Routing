"""Manifesto dos quatro braços. Não é uma política de eviction.

``ResidencyEmulator`` já é LRU. O modelo 2 reusa esse emulador com λ=0:
o mesmo código, o mesmo checkpoint e a mesma classe do modelo 1. O que
muda é o rótulo de relatório. A fatia do modelo 2 confronta esse
adversário (hold-out no B de referência e dose, quando houver JSON) com
o denso e com o MoE de λ>0.

O denso não é o MoE com λ=0. Não há segunda classe de eviction. Este
módulo não treina e não mede: só nomeia os braços e copia número que já
exista.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from moe_memory.config import ExperimentConfig

SCHEMA_VERSION = 1
PROTOCOL = "protocol-arm-comparison"
HOLDOUT_PROTOCOL = "wikitext-2-holdout-ppl"
DOSE_PROTOCOL = "wikitext-2-dose-response"
HOLDOUT_RESIDENCY = "lru_reset_per_split"
DOSE_RESIDENCY = "lru_reset_per_split_then_set_hot_budget"
DOSE_METHOD = "reset_lru_at_yaml_b_then_set_hot_budget"
REPORT_SEED = 0

POLICY_CLASS = "ResidencyEmulator"
POLICY = "lru"
DOSE_HOOK = "set_hot_budget"
MODEL_CLASS_MOE = "MemoryMoEModel"
MODEL_CLASS_DENSE = "DenseLanguageModel"
CODE_PATH_LAMBDA0 = "MemoryMoEModel+ResidencyEmulator+lambda0"
CODE_PATH_LAMBDA_POS = "MemoryMoEModel+ResidencyEmulator+lambda_positive"
CODE_PATH_DENSE = "DenseLanguageModel"

SPLITS = ("valid", "test")
REFERENCE_B_MULT = 1.0

ARM_MODEL1 = "model1"
ARM_MODEL2 = "model2"
ARM_MODEL3 = "model3"
ARM_MODEL4 = "model4"

ROLE_TOPK = "moe_topk_lambda0"
ROLE_ADVERSARY = "lru_adversary"
ROLE_DENSE = "dense"
ROLE_PENALTY = "moe_lambda_positive"

LABEL_MODEL1 = "MoE top-k λ=0"
LABEL_MODEL2 = "MoE + LRU adversary"
LABEL_MODEL3 = "Denso"
LABEL_MODEL4 = "MoE λ>0"

C_MEM_WEIGHT_ZERO = "weight_zero"
C_MEM_ABSENT = "absent"
C_MEM_WEIGHTED = "weighted"

SOURCE_HOLDOUT = "holdout"
SOURCE_DOSE = "dose_response"
SOURCE_NONE = "none"
FILLED_FROM_BOTH = "holdout+dose_response"
_QUALITY_KEYS = ("perplexity", "nll_nats")

STATUS_PENDING = "pending"
STATUS_FILLED = "filled"
STATUS_INCOMPLETE = "incomplete"
STATUS_MISSING = "missing"
STATUS_NOT_APPLICABLE = "not_applicable"
STATUS_CONFLICT = "conflict"

METRIC_KEYS = (
    "perplexity",
    "nll_nats",
    "bits_per_byte",
    "swaps",
    "bytes_moved",
    "promotions",
    "demotions",
    "entropy",
    "c_mem",
)

TABLE_COLUMNS = (
    "arm_id",
    "protocol_number",
    "arm_label",
    "role",
    "in_reporting_slice",
    "model_kind",
    "model_class",
    "lambda",
    "family",
    "c_mem_in_loss",
    "policy_class",
    "policy",
    "code_path",
    "same_code_path_as",
    "metrics_shared_with",
    "checkpoint",
    "split",
    "B_mult",
    "hot_slots",
    "budget_applies",
    "source",
    "filled_from",
    "status",
    *METRIC_KEYS,
)

# Nomes que pareceriam uma corrida própria do modelo 2. O protocolo não tem.
SEPARATE_MODEL2_NAMES = frozenset(
    {
        "model2",
        "modelo2",
        "modelo-2",
        "lru",
        "lru-adversary",
        "lru_adversary",
        "adversary",
        "adversario",
    }
)

_RESIDENCY_METRIC_KEYS = ("swaps", "bytes_moved", "promotions", "demotions", "entropy", "c_mem")

_IDENTITY_NOTE = (
    "O modelo 1 e o modelo 2 são o mesmo código: MemoryMoEModel, "
    "ResidencyEmulator (LRU) e λ=0, no checkpoint da semente 0 "
    "(seed-0/lambda-0, ou o layout antigo se for o único arquivo). "
    "Não há segunda política de eviction. "
    "O modelo 2 é a fatia de relatório desse adversário — hold-out no B de "
    "referência e a curva dose_response (reset no B do YAML, depois "
    "set_hot_budget) quando o JSON existir — contra o denso e contra o MoE "
    "com λ>0. "
    "O denso é DenseLanguageModel: não é o MoE com λ=0. "
    "O modelo 4 reusa o mesmo emulador; o que muda é o λ do treino."
)

_SLICE_NOTE = (
    "A tabela que o relatório preenche nesta fatia é o modelo 2 contra o "
    "modelo 3 e o modelo 4. Os números do modelo 2 são cópia do modelo 1, "
    "não uma medição nova. Célula pending não é número."
)


class ComparisonError(ValueError):
    """O manifesto ou o JSON de entrada contradiz o protocolo."""


def lambda_tag(value: float) -> str:
    """Mesmo texto de ``experiments/seed_layout.py`` (pasta ``lambda-<tag>``)."""

    return _seed_layout().lambda_tag(value)


def same_lambda(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-6)


def checkpoint_path(outputs: str | Path, family: str) -> Path:
    """Semente 0, a mesma regra de ``run_holdout_eval.discover_families``.

    Prefere ``seed-0``. Se o checkpoint novo não existir e o layout antigo
    existir, usa o antigo. O modelo 2 não tem pasta própria.
    """

    layout = _seed_layout()
    root = Path(outputs)
    if family == "baseline":
        corpus = root / "baseline" / "wikitext"
        return layout.prefer_checkpoint(layout.baseline_seed_dir(corpus, REPORT_SEED), corpus)
    if family.startswith("lambda-"):
        grid = root / "wikitext"
        value = float(family.removeprefix("lambda-"))
        return layout.prefer_checkpoint(
            layout.lambda_run_dir(grid, value, REPORT_SEED),
            grid / family,
        )
    raise ComparisonError(f"família de checkpoint desconhecida: {family}")


def _seed_layout():
    import sys

    experiments = Path(__file__).resolve().parents[2] / "experiments"
    folder = str(experiments)
    if folder not in sys.path:
        sys.path.insert(0, folder)
    import seed_layout

    return seed_layout


def build_comparison(
    config: ExperimentConfig,
    *,
    outputs: str | Path = "outputs",
    holdout: dict[str, Any] | None = None,
    dose: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Monta o manifesto e a tabela. Hold-out e dose são opcionais.

    Sem os dois JSON, toda célula medida fica ``pending``. Isso não é
    número de relatório e este módulo não dispara treino nem eval.
    """

    outputs_path = Path(outputs)
    reference = _reference_multiplier(config)
    budgets = _budgets(config, reference)
    families = _families(config)
    arms = _arms(config, outputs_path, families)
    rows = _rows(arms, budgets, reference)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "produces_measurements": False,
        "trained_by_this_module": False,
        "evaluated_by_this_module": False,
        "config": str(config.path),
        "outputs": str(outputs_path),
        "corpus": config.corpus.report,
        "splits": list(SPLITS),
        "identity": _identity(arms),
        "reporting_slice": _reporting_slice(),
        "dose_hook": {
            "class": POLICY_CLASS,
            "method": DOSE_HOOK,
            "procedure": DOSE_METHOD,
            "protocol": DOSE_PROTOCOL,
            "residency": DOSE_RESIDENCY,
            "artifact": "holdout/dose_response.json",
            "budget_fn": "MemoryConfig.scaled_hot_slots",
            "changes_policy": False,
            "retrained": False,
            "reference_B_mult": reference,
            "budgets": budgets,
            "note": (
                "A curva vem do hold-out: dose_response true no summary e "
                "outputs/holdout/dose_response.json. Cada ponto faz reset no B "
                "do YAML e ResidencyEmulator.set_hot_budget. O ranking continua "
                "LRU. O denso é n/a. Não existe outra classe e não há retreino."
            ),
        },
        "checkpoint_seed": REPORT_SEED,
        "arms": arms,
        "columns": list(TABLE_COLUMNS),
        "rows": rows,
        "conflicts": [],
        "measurement_sources": [],
    }
    if holdout is not None:
        _apply_holdout(payload, holdout, families, budgets)
    if dose is not None:
        _apply_dose(payload, dose, families, budgets)
    _finalize(payload)
    _check_identity(payload)
    return payload


def format_summary(payload: dict[str, Any]) -> str:
    """Texto curto para o terminal. Não substitui o JSON."""

    identity = payload["identity"]
    same = "sim" if identity["model1_equals_model2"] else "não"
    second = identity["second_eviction_policy_class"]
    second_text = "nenhuma" if second is None else str(second)
    lines = [
        "Comparação dos quatro braços — não treina e não mede.",
        identity["note"],
        f"modelo 1 ≡ modelo 2: {same}; "
        f"política: {identity['eviction_policy_class']}; "
        f"segunda política: {second_text}.",
    ]
    for arm in payload["arms"]:
        shared = arm["same_code_path_as"]
        relation = f"mesmo código que {shared}" if shared else "código próprio"
        if arm["id"] == ARM_MODEL4:
            lambdas = ", ".join(str(item["lambda"]) for item in arm["variants"])
        elif arm["lambda"] is None:
            lambdas = "—"
        else:
            lambdas = str(arm["lambda"])
        policy = arm["policy_class"] if arm["policy_class"] is not None else "—"
        lines.append(
            f"{arm['id']}  {arm['label']}  λ={lambdas}  política={policy}  {relation}"
        )
    counts: dict[str, int] = {}
    for row in payload["rows"]:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    pieces = [f"{status}={counts[status]}" for status in sorted(counts)]
    lines.append("linhas: " + ", ".join(pieces))
    if payload["metrics_complete"]:
        lines.append("tabela completa com números de outro script; este runner não mediu.")
    else:
        lines.append("métricas incompletas: este JSON não é número de relatório.")
    return "\n".join(lines)


def _reference_multiplier(config: ExperimentConfig) -> float:
    refs = [float(value) for value in config.eval.B_mult if same_lambda(value, REFERENCE_B_MULT)]
    if len(refs) != 1:
        raise ComparisonError(
            "eval.B_mult precisa ter um único 1.0 (B de referência do hold-out)"
        )
    keys = [_budget_key(value) for value in config.eval.B_mult]
    if len(keys) != len(set(keys)):
        raise ComparisonError("eval.B_mult repetido")
    return refs[0]


def _budgets(config: ExperimentConfig, reference: float) -> list[dict[str, Any]]:
    budgets = []
    for multiplier in config.eval.B_mult:
        value = float(multiplier)
        budgets.append(
            {
                "B_mult": value,
                "hot_slots": config.memory.scaled_hot_slots(config.model.E, value),
                "role": "reference" if same_lambda(value, reference) else "dose",
            }
        )
    return budgets


def _families(config: ExperimentConfig) -> dict[str, dict[str, Any]]:
    zeros = [float(value) for value in config.loss.lambda_grid if same_lambda(value, 0.0)]
    positive = [float(value) for value in config.loss.lambda_grid if float(value) > 0]
    if len(zeros) != 1:
        raise ComparisonError("a grade precisa de um único λ=0 (modelo 1 e modelo 2)")
    if not positive:
        raise ComparisonError("a grade precisa de algum λ>0 (modelo 4)")
    families: dict[str, dict[str, Any]] = {}
    zero_family = f"lambda-{lambda_tag(zeros[0])}"
    families[zero_family] = {
        "family": zero_family,
        "lambda": zeros[0],
        "model_kind": "moe",
        "model_class": MODEL_CLASS_MOE,
    }
    for value in positive:
        family = f"lambda-{lambda_tag(value)}"
        if family in families:
            raise ComparisonError(f"família {family} repetida na grade")
        families[family] = {
            "family": family,
            "lambda": value,
            "model_kind": "moe",
            "model_class": MODEL_CLASS_MOE,
        }
    families["baseline"] = {
        "family": "baseline",
        "lambda": None,
        "model_kind": "dense",
        "model_class": MODEL_CLASS_DENSE,
    }
    return families


def _arms(
    config: ExperimentConfig,
    outputs: Path,
    families: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    del config
    zero = next(
        item
        for item in families.values()
        if item["lambda"] is not None and same_lambda(item["lambda"], 0.0)
    )
    positive = [
        item
        for item in families.values()
        if item["model_kind"] == "moe" and not same_lambda(item["lambda"], 0.0)
    ]
    zero_checkpoint = str(checkpoint_path(outputs, zero["family"]))

    def lambda0_arm(arm_id: str, number: int, label: str, role: str, **extra: Any) -> dict[str, Any]:
        return {
            "id": arm_id,
            "protocol_number": number,
            "label": label,
            "role": role,
            "model_kind": "moe",
            "model_class": MODEL_CLASS_MOE,
            "lambda": zero["lambda"],
            "policy_class": POLICY_CLASS,
            "policy": POLICY,
            "code_path": CODE_PATH_LAMBDA0,
            "c_mem_in_loss": C_MEM_WEIGHT_ZERO,
            "checkpoint_family": zero["family"],
            "checkpoint": zero_checkpoint,
            "variants": [_variant(outputs, zero)],
            **extra,
        }

    model1 = lambda0_arm(
        ARM_MODEL1,
        1,
        LABEL_MODEL1,
        ROLE_TOPK,
        same_code_path_as=None,
        same_eviction_class_as=None,
        metrics_shared_with=None,
        reporting_slice=False,
        note=(
            "MoE top-k com λ=0 e balanço. C_mem entra com peso zero. "
            "A residência na avaliação é o LRU do ResidencyEmulator sob B. "
            "O λ=0 da grade é este braço."
        ),
    )
    model2 = lambda0_arm(
        ARM_MODEL2,
        2,
        LABEL_MODEL2,
        ROLE_ADVERSARY,
        same_code_path_as=ARM_MODEL1,
        same_eviction_class_as=ARM_MODEL1,
        metrics_shared_with=ARM_MODEL1,
        reporting_slice=True,
        note=(
            "Adversário LRU do protocolo, como fatia de relatório. "
            "Mesmo emulador, mesmo λ=0 e mesmo checkpoint do modelo 1. "
            "Não é uma segunda política. A comparação é hold-out e dose "
            "(quando houver) contra o denso e contra λ>0."
        ),
    )
    model3 = {
        "id": ARM_MODEL3,
        "protocol_number": 3,
        "label": LABEL_MODEL3,
        "role": ROLE_DENSE,
        "model_kind": "dense",
        "model_class": MODEL_CLASS_DENSE,
        "lambda": None,
        "policy_class": None,
        "policy": None,
        "code_path": CODE_PATH_DENSE,
        "c_mem_in_loss": C_MEM_ABSENT,
        "checkpoint_family": "baseline",
        "checkpoint": str(checkpoint_path(outputs, "baseline")),
        "variants": [_variant(outputs, families["baseline"])],
        "same_code_path_as": None,
        "same_eviction_class_as": None,
        "metrics_shared_with": None,
        "reporting_slice": True,
        "note": (
            "LM denso, sem roteador e sem emulador. "
            "Não é o MoE com λ=0 e não entra em set_hot_budget."
        ),
    }
    model4 = {
        "id": ARM_MODEL4,
        "protocol_number": 4,
        "label": LABEL_MODEL4,
        "role": ROLE_PENALTY,
        "model_kind": "moe",
        "model_class": MODEL_CLASS_MOE,
        "lambda": None,
        "policy_class": POLICY_CLASS,
        "policy": POLICY,
        "code_path": CODE_PATH_LAMBDA_POS,
        "c_mem_in_loss": C_MEM_WEIGHTED,
        "checkpoint_family": None,
        "checkpoint": None,
        "variants": [_variant(outputs, item) for item in positive],
        "same_code_path_as": None,
        "same_eviction_class_as": ARM_MODEL1,
        "metrics_shared_with": None,
        "reporting_slice": True,
        "note": (
            "MoE com λ>0. O emulador é o mesmo ResidencyEmulator (LRU); "
            "o treino é que cobra C_mem. Uma linha por λ da grade acima de zero."
        ),
    }
    return [model1, model2, model3, model4]


def _variant(outputs: Path, family: dict[str, Any]) -> dict[str, Any]:
    return {
        "family": family["family"],
        "lambda": family["lambda"],
        "model_kind": family["model_kind"],
        "checkpoint": str(checkpoint_path(outputs, family["family"])),
    }


def _rows(
    arms: list[dict[str, Any]],
    budgets: list[dict[str, Any]],
    reference: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in arms:
        for variant in arm["variants"]:
            for split in SPLITS:
                for budget in budgets:
                    is_reference = same_lambda(budget["B_mult"], reference)
                    dense = arm["model_kind"] == "dense"
                    if dense and not is_reference:
                        status = STATUS_NOT_APPLICABLE
                        source = SOURCE_NONE
                        applies = False
                        hot_slots = None
                    elif dense:
                        status = STATUS_PENDING
                        source = SOURCE_HOLDOUT
                        applies = False
                        hot_slots = None
                    elif is_reference:
                        status = STATUS_PENDING
                        source = SOURCE_HOLDOUT
                        applies = True
                        hot_slots = budget["hot_slots"]
                    else:
                        status = STATUS_PENDING
                        source = SOURCE_DOSE
                        applies = True
                        hot_slots = budget["hot_slots"]
                    body = {
                        "arm_id": arm["id"],
                        "protocol_number": arm["protocol_number"],
                        "arm_label": arm["label"],
                        "role": arm["role"],
                        "in_reporting_slice": arm["id"] != ARM_MODEL1,
                        "model_kind": arm["model_kind"],
                        "model_class": arm["model_class"],
                        "lambda": variant["lambda"],
                        "family": variant["family"],
                        "c_mem_in_loss": arm["c_mem_in_loss"],
                        "policy_class": arm["policy_class"],
                        "policy": arm["policy"],
                        "code_path": arm["code_path"],
                        "same_code_path_as": arm["same_code_path_as"],
                        "metrics_shared_with": arm["metrics_shared_with"],
                        "checkpoint": variant["checkpoint"],
                        "split": split,
                        "B_mult": budget["B_mult"],
                        "hot_slots": hot_slots,
                        "budget_applies": applies,
                        "source": source,
                        "filled_from": None,
                        "status": status,
                        **_blank_metrics(),
                    }
                    rows.append(_row(body))
    return rows


def _identity(arms: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {arm["id"]: arm for arm in arms}
    model1 = by_id[ARM_MODEL1]
    model2 = by_id[ARM_MODEL2]
    return {
        "model1_equals_model2": True,
        "shared_code_path": model1["code_path"],
        "shared_checkpoint_family": model1["checkpoint_family"],
        "shared_lambda": model1["lambda"],
        "shared_checkpoint": model1["checkpoint"],
        "checkpoint_seed": REPORT_SEED,
        "checkpoint_layout": "seed-0 preferred; legacy lambda-<λ> only if it is the sole file",
        "eviction_policy_class": POLICY_CLASS,
        "eviction_algorithm": POLICY,
        "second_eviction_policy_class": None,
        "dense_is_moe_lambda0": False,
        "model4_same_eviction_class": True,
        "model2_code_path_equals_model1": model1["code_path"] == model2["code_path"],
        "note": _IDENTITY_NOTE,
    }


def _reporting_slice() -> dict[str, Any]:
    return {
        "id": ARM_MODEL2,
        "label": LABEL_MODEL2,
        "holds": ["holdout", "dose_response"],
        "against": [ARM_MODEL3, ARM_MODEL4],
        "includes_subject_metrics_from": ARM_MODEL1,
        "note": _SLICE_NOTE,
    }


def _apply_holdout(
    payload: dict[str, Any],
    holdout: dict[str, Any],
    families: dict[str, dict[str, Any]],
    budgets: list[dict[str, Any]],
) -> None:
    if not isinstance(holdout, dict):
        raise ComparisonError("hold-out não é um objeto JSON")
    if holdout.get("protocol") != HOLDOUT_PROTOCOL:
        raise ComparisonError(
            f"hold-out protocol={holdout.get('protocol')!r}; esperado {HOLDOUT_PROTOCOL}"
        )
    flag = holdout.get("dose_response", False)
    if flag not in (True, False):
        raise ComparisonError(
            "dose_response do hold-out precisa ser true ou false; "
            f"recebido {flag!r}"
        )
    if holdout.get("residency") != HOLDOUT_RESIDENCY:
        raise ComparisonError(
            "hold-out sem residência lru_reset_per_split; não há outra política"
        )
    runs = _index_runs(holdout.get("runs"), allow=set(families))
    updated: list[dict[str, Any]] = []
    for row in payload["rows"]:
        if row["source"] != SOURCE_HOLDOUT:
            updated.append(row)
            continue
        expected = families[row["family"]]
        run = runs.get(row["family"])
        updated.append(_fill_from_run(row, run, expected, filled_from=SOURCE_HOLDOUT))
    payload["rows"] = updated
    if flag is not True:
        return
    curves: dict[str, dict[str, Any]] = {}
    for family, run in runs.items():
        block = run.get("dose_response")
        if not isinstance(block, dict):
            raise ComparisonError(
                f"{family} marcado no hold-out com dose_response true, sem o bloco da curva"
            )
        _check_run_header(run, families[family], require_kind=run.get("status") == "ok")
        curves[family] = block
    _apply_curves(payload, curves, families, budgets)


def _apply_dose(
    payload: dict[str, Any],
    dose: dict[str, Any],
    families: dict[str, dict[str, Any]],
    budgets: list[dict[str, Any]],
) -> None:
    """Lê ``dose_response.json`` do hold-out (protocolo ``wikitext-2-dose-response``)."""

    if not isinstance(dose, dict):
        raise ComparisonError("dose não é um objeto JSON")
    if dose.get("protocol") != DOSE_PROTOCOL:
        raise ComparisonError(
            f"dose protocol={dose.get('protocol')!r}; esperado {DOSE_PROTOCOL}"
        )
    if dose.get("retrained") is not False:
        raise ComparisonError("a curva de dose não retreina; retrained precisa ser false")
    if dose.get("residency") != DOSE_RESIDENCY:
        raise ComparisonError(
            "a dose só pode usar o LRU com reset no B do YAML e set_hot_budget "
            f"({DOSE_RESIDENCY})"
        )
    if dose.get("dense_axis") not in (None, "n/a"):
        raise ComparisonError("o eixo do denso na dose é n/a")
    if dose.get("policy_class") not in (None, POLICY_CLASS):
        raise ComparisonError(
            f"policy_class={dose.get('policy_class')!r}; a única classe é {POLICY_CLASS}"
        )
    if dose.get("hook") not in (None, DOSE_HOOK):
        raise ComparisonError(
            f"hook={dose.get('hook')!r}; o gancho é {DOSE_HOOK}"
        )
    if dose.get("second_policy_class") not in (None,):
        raise ComparisonError(
            "a dose declara outra política de eviction; o protocolo só tem ResidencyEmulator"
        )
    runs = _index_runs(dose.get("runs"), allow=set(families))
    curves: dict[str, dict[str, Any]] = {}
    for family, run in runs.items():
        block = run.get("dose_response")
        if not isinstance(block, dict):
            raise ComparisonError(f"{family} sem bloco dose_response")
        _check_run_header(run, families[family], require_kind=run.get("status") == "ok")
        curves[family] = block
    _apply_curves(payload, curves, families, budgets)


def _apply_curves(
    payload: dict[str, Any],
    curves: dict[str, dict[str, Any]],
    families: dict[str, dict[str, Any]],
    budgets: list[dict[str, Any]],
) -> None:
    by_key = {_row_key(row): row for row in payload["rows"]}
    conflicts: list[dict[str, Any]] = list(payload["conflicts"])
    for family, curve in curves.items():
        expected = families[family]
        if expected["model_kind"] == "dense":
            _require_dense_curve(curve)
            continue
        status = curve.get("status")
        if status == "missing":
            _mark_dose_pending(by_key, family, STATUS_MISSING)
            continue
        if status != "ok":
            raise ComparisonError(f"curva de {family} com status={status!r}")
        if curve.get("retrained") is True:
            raise ComparisonError("a curva de dose não retreina")
        if curve.get("method") != DOSE_METHOD:
            raise ComparisonError(
                f"método {curve.get('method')!r} não é {DOSE_METHOD}; "
                "não há segunda política"
            )
        for budget in _dose_budgets(curve, budgets, family):
            for split in SPLITS:
                split_row = budget["splits"].get(split)
                if split_row is None:
                    continue
                targets = [
                    row
                    for row in payload["rows"]
                    if row["family"] == family
                    and row["split"] == split
                    and _same_budget(row["B_mult"], budget["B_mult"])
                    and row["status"] != STATUS_NOT_APPLICABLE
                ]
                if not targets:
                    raise ComparisonError(
                        f"dose {family} multiplier={budget['B_mult']} split={split} "
                        "não cabe na tabela"
                    )
                cell_status, metrics = _metrics_of(split_row, "moe")
                for target in targets:
                    key = _row_key(target)
                    current = by_key[key]
                    merged, conflict = _merge_dose(current, cell_status, metrics)
                    by_key[key] = merged
                    if conflict is not None:
                        conflicts.append(conflict)
    payload["rows"] = [_row(by_key[_row_key(row)]) for row in payload["rows"]]
    payload["conflicts"] = conflicts


def _require_dense_curve(curve: dict[str, Any]) -> None:
    if curve.get("status") != "n/a" or curve.get("budgets") is not None:
        raise ComparisonError(
            "o denso na dose é n/a, sem budgets: não é o MoE com λ=0 e não tem set_hot_budget"
        )
    if curve.get("method") is not None:
        raise ComparisonError("o denso não tem método de residência")
    if curve.get("retrained") is True:
        raise ComparisonError("a curva de dose não retreina")


def _mark_dose_pending(
    by_key: dict[tuple[Any, ...], dict[str, Any]],
    family: str,
    status: str,
) -> None:
    for key, row in list(by_key.items()):
        if row["family"] != family or row["source"] != SOURCE_DOSE:
            continue
        if row["status"] != STATUS_PENDING:
            continue
        by_key[key] = _with(row, _blank_metrics(), status=status, filled_from=None)


def _dose_budgets(
    curve: dict[str, Any],
    budgets: list[dict[str, Any]],
    family: str,
) -> list[dict[str, Any]]:
    raw = curve.get("budgets")
    if not isinstance(raw, list) or not raw:
        raise ComparisonError(f"{family} sem budgets na curva de dose")
    known = {_budget_key(item["B_mult"]): item for item in budgets}
    parsed = []
    seen: set[float] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ComparisonError("ponto da dose não é um objeto")
        if "multiplier" not in item or "hot_slots" not in item or "splits" not in item:
            raise ComparisonError(
                "ponto da dose precisa de multiplier, hot_slots e splits"
            )
        key = _budget_key(item["multiplier"])
        if key in seen:
            raise ComparisonError(f"multiplier={item['multiplier']} repetido na dose")
        seen.add(key)
        expected = known.get(key)
        if expected is None:
            raise ComparisonError(
                f"multiplier={item['multiplier']} não está em eval.B_mult; "
                "a dose não inventa orçamento"
            )
        hot = item["hot_slots"]
        if isinstance(hot, bool) or not isinstance(hot, int) or hot != expected["hot_slots"]:
            raise ComparisonError(
                f"hot_slots={hot} não é o teto {expected['hot_slots']} de "
                f"{expected['B_mult']}×B (scaled_hot_slots)"
            )
        if "yaml_reference" in item and bool(item["yaml_reference"]) != (
            expected["role"] == "reference"
        ):
            raise ComparisonError(
                f"yaml_reference de {item['multiplier']}×B não bate com o B do YAML"
            )
        splits = item["splits"]
        if not isinstance(splits, dict):
            raise ComparisonError("splits da dose não é um objeto")
        unknown = set(splits) - set(SPLITS)
        if unknown:
            raise ComparisonError(f"split de dose desconhecido: {sorted(unknown)}")
        parsed.append(
            {
                "B_mult": expected["B_mult"],
                "hot_slots": expected["hot_slots"],
                "splits": splits,
            }
        )
    return parsed


def _merge_dose(
    current: dict[str, Any],
    status: str,
    metrics: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Copia a dose para a linha. O modelo 2 entra porque a família é a do λ=0."""

    if current["status"] == STATUS_NOT_APPLICABLE:
        raise ComparisonError("dose não se aplica ao denso")
    if current["status"] == STATUS_CONFLICT:
        return current, None
    if current["status"] == STATUS_MISSING and status == STATUS_FILLED:
        return (
            _with(current, _blank_metrics(), status=STATUS_CONFLICT, filled_from=None),
            _conflict(current, "hold-out marcou a família ausente e a dose trouxe número"),
        )
    agree_keys = _QUALITY_KEYS if current["source"] == SOURCE_HOLDOUT else METRIC_KEYS
    if current["status"] == STATUS_FILLED and not _overlap_agrees(current, metrics, agree_keys):
        return (
            _with(current, _blank_metrics(), status=STATUS_CONFLICT, filled_from=None),
            _conflict(current, "hold-out e dose divergem"),
        )
    if status != STATUS_FILLED:
        if current["status"] == STATUS_PENDING:
            return _with(current, metrics, status=status, filled_from=SOURCE_DOSE), None
        return current, None
    if current["status"] == STATUS_FILLED:
        filled_from = (
            FILLED_FROM_BOTH if current["source"] == SOURCE_HOLDOUT else current["filled_from"]
        )
        return (
            _with(current, _metrics_of_row(current), status=STATUS_FILLED, filled_from=filled_from),
            None,
        )
    if current["status"] == STATUS_INCOMPLETE:
        # O hold-out cortado não é promovido por uma dose que só repete o pedaço.
        return current, None
    if current["status"] == STATUS_PENDING:
        return _with(current, metrics, status=STATUS_FILLED, filled_from=SOURCE_DOSE), None
    raise ComparisonError(f"status inesperado ao fundir dose: {current['status']}")


def _fill_from_run(
    row: dict[str, Any],
    run: dict[str, Any] | None,
    expected: dict[str, Any],
    *,
    filled_from: str,
) -> dict[str, Any]:
    if run is None or run.get("status") == "missing":
        return _with(row, _blank_metrics(), status=STATUS_MISSING, filled_from=None)
    _check_run_header(run, expected, require_kind=run.get("status") == "ok")
    splits = run.get("splits")
    split_row = splits.get(row["split"]) if isinstance(splits, dict) else None
    if not isinstance(split_row, dict):
        return _with(row, _blank_metrics(), status=STATUS_MISSING, filled_from=None)
    status, metrics = _metrics_of(split_row, expected["model_kind"])
    if run.get("status") != "ok":
        status = STATUS_INCOMPLETE
    if status == STATUS_FILLED:
        return _with(row, metrics, status=STATUS_FILLED, filled_from=filled_from)
    return _with(row, metrics, status=STATUS_INCOMPLETE, filled_from=filled_from)


def _metrics_of(split_row: dict[str, Any], model_kind: str) -> tuple[str, dict[str, Any]]:
    if not isinstance(split_row, dict):
        raise ComparisonError("split não é um objeto")
    metrics = {}
    for key in METRIC_KEYS:
        value = split_row.get(key)
        if value is None:
            metrics[key] = None
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ComparisonError(f"{key} não é numérico")
        if isinstance(value, float) and not math.isfinite(value):
            raise ComparisonError(f"{key} não é finito")
        metrics[key] = value
    if model_kind == "dense":
        for key in _RESIDENCY_METRIC_KEYS:
            if metrics[key] is not None:
                raise ComparisonError(
                    "o denso não tem emulador nem C_mem; não é o MoE com λ=0"
                )
    nll = metrics["nll_nats"]
    complete = (
        split_row.get("truncated") is False
        and isinstance(nll, (int, float))
        and not isinstance(nll, bool)
        and math.isfinite(float(nll))
    )
    if model_kind == "moe" and (metrics["swaps"] is None or metrics["bytes_moved"] is None):
        complete = False
    if complete:
        return STATUS_FILLED, metrics
    return STATUS_INCOMPLETE, metrics


def _check_run_header(
    run: dict[str, Any],
    expected: dict[str, Any],
    *,
    require_kind: bool,
) -> None:
    kind = run.get("model_kind")
    if require_kind and kind != expected["model_kind"]:
        raise ComparisonError(
            f"{_run_name(run)} é {kind!r}, o manifesto espera {expected['model_kind']}. "
            "O denso não é o MoE com λ=0."
        )
    if kind not in (None, expected["model_kind"]):
        raise ComparisonError(
            f"{_run_name(run)} é {kind!r}, o manifesto espera {expected['model_kind']}"
        )
    if expected["model_kind"] == "dense":
        if run.get("lambda") is not None:
            raise ComparisonError("o denso não tem λ; não é o MoE com λ=0")
        if run.get("memory_path") not in (None, False):
            raise ComparisonError("o denso não tem emulador")
        return
    if require_kind and (
        run.get("lambda") is None or not same_lambda(run["lambda"], expected["lambda"])
    ):
        raise ComparisonError(
            f"{_run_name(run)} tem λ={run.get('lambda')!r}, esperado {expected['lambda']}"
        )
    if run.get("lambda") is not None and not same_lambda(run["lambda"], expected["lambda"]):
        raise ComparisonError(
            f"{_run_name(run)} tem λ={run['lambda']}, esperado {expected['lambda']}"
        )
    if run.get("residency") not in (None, HOLDOUT_RESIDENCY):
        raise ComparisonError("residência diferente do LRU do ResidencyEmulator")
    for key in ("policy_class", "eviction_policy_class"):
        if run.get(key) not in (None, POLICY_CLASS):
            raise ComparisonError(
                f"{key}={run.get(key)!r}; a única classe é {POLICY_CLASS}"
            )
    if run.get("policy") not in (None, POLICY):
        raise ComparisonError(f"policy={run.get('policy')!r}; o algoritmo é lru")


def _index_runs(
    runs: Any,
    *,
    allow: set[str],
    dose: bool = False,
) -> dict[str, dict[str, Any]]:
    if not isinstance(runs, list):
        raise ComparisonError("runs não é uma lista")
    found: dict[str, dict[str, Any]] = {}
    for run in runs:
        if not isinstance(run, dict):
            raise ComparisonError("corrida não é um objeto")
        name = run.get("family", run.get("name"))
        if not isinstance(name, str) or not name:
            raise ComparisonError("corrida sem família")
        if name in SEPARATE_MODEL2_NAMES:
            raise ComparisonError(
                f"{name} não é uma corrida à parte: o modelo 2 reusa o checkpoint lambda-0 "
                "e o mesmo ResidencyEmulator"
            )
        if dose and name == "baseline":
            raise ComparisonError(
                "o denso não entra na dose: não tem ResidencyEmulator.set_hot_budget"
            )
        if name not in allow:
            raise ComparisonError(f"família {name} não está no manifesto")
        if name in found:
            raise ComparisonError(f"família {name} repetida")
        found[name] = run
    return found


def _finalize(payload: dict[str, Any]) -> None:
    sources: list[str] = []
    if any(row["filled_from"] in {SOURCE_HOLDOUT, FILLED_FROM_BOTH} for row in payload["rows"]):
        sources.append(SOURCE_HOLDOUT)
    if any(row["filled_from"] in {SOURCE_DOSE, FILLED_FROM_BOTH} for row in payload["rows"]):
        sources.append(SOURCE_DOSE)
    payload["measurement_sources"] = sources
    payload["holdout_ready"] = _channel_ready(payload["rows"], SOURCE_HOLDOUT)
    payload["dose_ready"] = _channel_ready(payload["rows"], SOURCE_DOSE)
    payload["metrics_complete"] = all(
        row["status"] in {STATUS_FILLED, STATUS_NOT_APPLICABLE} for row in payload["rows"]
    ) and not payload["conflicts"]
    payload["rows"] = [_row(row) for row in payload["rows"]]


def _channel_ready(rows: list[dict[str, Any]], source: str) -> bool:
    relevant = [row for row in rows if row["source"] == source]
    return bool(relevant) and all(row["status"] == STATUS_FILLED for row in relevant)


def _check_identity(payload: dict[str, Any]) -> None:
    arms = {arm["id"]: arm for arm in payload["arms"]}
    if set(arms) != {ARM_MODEL1, ARM_MODEL2, ARM_MODEL3, ARM_MODEL4}:
        raise ComparisonError("o manifesto deixou de ter os quatro braços")
    model1 = arms[ARM_MODEL1]
    model2 = arms[ARM_MODEL2]
    model3 = arms[ARM_MODEL3]
    model4 = arms[ARM_MODEL4]
    if model1["code_path"] != model2["code_path"] or model1["code_path"] != CODE_PATH_LAMBDA0:
        raise ComparisonError("modelo 1 e modelo 2 divergiram de código")
    if model1["checkpoint"] != model2["checkpoint"] or model1["checkpoint"] is None:
        raise ComparisonError("modelo 2 ganhou checkpoint próprio")
    if model1["policy_class"] != POLICY_CLASS or model2["policy_class"] != POLICY_CLASS:
        raise ComparisonError("a política deixou de ser ResidencyEmulator")
    if model1["policy"] != POLICY or model2["policy"] != POLICY or model4["policy"] != POLICY:
        raise ComparisonError("o algoritmo deixou de ser lru")
    if model4["policy_class"] != POLICY_CLASS:
        raise ComparisonError("o modelo 4 não pode trocar o emulador")
    if model3["model_class"] != MODEL_CLASS_DENSE or model3["policy_class"] is not None:
        raise ComparisonError("o denso não pode carregar emulador")
    if model3["code_path"] == model1["code_path"]:
        raise ComparisonError("o denso não pode ser o caminho do λ=0")
    identity = payload["identity"]
    if identity["second_eviction_policy_class"] is not None:
        raise ComparisonError("segunda política declarada")
    if identity["model1_equals_model2"] is not True or identity["dense_is_moe_lambda0"] is not False:
        raise ComparisonError("identidade do manifesto inconsistente")
    if payload["dose_hook"]["changes_policy"] is not False:
        raise ComparisonError("a dose não pode trocar a política")
    if payload["dose_hook"]["procedure"] != DOSE_METHOD or payload["dose_hook"]["retrained"] is not False:
        raise ComparisonError("a curva de dose deixou de ser set_hot_budget sem retreino")
    if payload.get("checkpoint_seed") != REPORT_SEED or payload["identity"]["checkpoint_seed"] != REPORT_SEED:
        raise ComparisonError("a comparação deixou de preferir a semente 0")
    if payload["produces_measurements"] is not False:
        raise ComparisonError("este manifesto não produz medição")
    for row in payload["rows"]:
        if row["arm_id"] == ARM_MODEL2 and row["metrics_shared_with"] != ARM_MODEL1:
            raise ComparisonError("linha do modelo 2 sem cópia do modelo 1")
        if row["arm_id"] == ARM_MODEL2 and row["code_path"] != CODE_PATH_LAMBDA0:
            raise ComparisonError("linha do modelo 2 com outro código")
        if set(row) != set(TABLE_COLUMNS):
            raise ComparisonError("linha fora das colunas do schema")


def _conflict(row: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "arm_id": row["arm_id"],
        "family": row["family"],
        "split": row["split"],
        "B_mult": row["B_mult"],
        "reason": reason,
    }


def _overlap_agrees(
    row: dict[str, Any],
    metrics: dict[str, Any],
    keys: tuple[str, ...] = METRIC_KEYS,
) -> bool:
    for key in keys:
        if row[key] is None or metrics[key] is None:
            continue
        if not _values_close(row[key], metrics[key]):
            return False
    return True


def _metrics_of_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in METRIC_KEYS}


def _values_close(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-6)
    return left == right


def _with(
    row: dict[str, Any],
    metrics: dict[str, Any],
    *,
    status: str,
    filled_from: str | None,
) -> dict[str, Any]:
    updated = dict(row)
    updated.update(metrics)
    updated["status"] = status
    updated["filled_from"] = filled_from
    return _row(updated)


def _blank_metrics() -> dict[str, Any]:
    return {key: None for key in METRIC_KEYS}


def _row(data: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in TABLE_COLUMNS if key not in data]
    extra = [key for key in data if key not in TABLE_COLUMNS]
    if missing or extra:
        raise ComparisonError(f"linha fora do schema: falta {missing} sobra {extra}")
    return {key: data[key] for key in TABLE_COLUMNS}


def _row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row["arm_id"], row["family"], row["split"], _budget_key(row["B_mult"]))


def _budget_key(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComparisonError(f"B_mult inválido: {value!r}")
    return round(float(value), 6)


def _same_budget(left: Any, right: Any) -> bool:
    return _budget_key(left) == _budget_key(right)


def _run_name(run: dict[str, Any]) -> str:
    name = run.get("family", run.get("name"))
    return str(name)
