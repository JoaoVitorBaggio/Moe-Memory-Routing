"""Pastas por semente da grade de λ e do baseline denso.

O YAML fecha ``eval.seeds`` (3 na partida). Os runners lançam as sementes
``0 .. seeds-1``. Cada job continua sendo um processo: um λ ou o denso,
um ``--seed``, um ``--work-dir``.

Layout novo (gitignored)::

    outputs/wikitext/seed-<n>/lambda-<λ>/
    outputs/baseline/<corpus>/seed-<n>/

A semente 0 que já está no disco no layout antigo continua lá, desde que
a pasta ``seed-0`` ainda não tenha artefato de corrida (``done.json``,
``checkpoint.pt``, ``train.jsonl`` ou ``train.log``). Se os dois existirem,
vale ``seed-0``.

    outputs/wikitext/lambda-<λ>/
    outputs/baseline/<corpus>/          # checkpoint e log soltos na raiz
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_RUN_FILES = ("done.json", "checkpoint.pt", "train.jsonl", "train.log")

LEGACY_LAYOUT_NOTE = (
    "Semente 0: se já existir artefato no layout antigo e seed-0 ainda "
    "estiver vazio, a corrida continua em outputs/wikitext/lambda-<λ>/ ou "
    "na raiz outputs/baseline/<corpus>/. Caso contrário a pasta é "
    "seed-<n>/lambda-<λ>/ ou seed-<n>/. Se os dois tiverem artefato, vale seed-0."
)


def lambda_tag(value: float) -> str:
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text if text else "0"


def seed_indices(n_seeds: int) -> list[int]:
    """Sementes ``0 .. n_seeds-1``, na ordem em que os runners lançam."""

    if isinstance(n_seeds, bool) or not isinstance(n_seeds, int):
        raise ValueError(f"eval.seeds precisa ser inteiro >= 1, recebido {n_seeds!r}")
    if n_seeds < 1:
        raise ValueError(f"eval.seeds precisa ser >= 1, recebido {n_seeds}")
    return list(range(n_seeds))


def seed_dirname(seed: int) -> str:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed precisa ser inteiro >= 0, recebido {seed!r}")
    return f"seed-{seed}"


def is_seed_dirname(name: str) -> bool:
    if not name.startswith("seed-"):
        return False
    suffix = name.removeprefix("seed-")
    return suffix.isdigit()


def has_run_artifacts(path: Path) -> bool:
    folder = Path(path)
    return any((folder / name).is_file() for name in _RUN_FILES)


def lambda_run_dir(root: Path, lam: float, seed: int) -> Path:
    """Pasta canônica de um λ, sem olhar o que já está no disco."""

    return Path(root) / seed_dirname(int(seed)) / f"lambda-{lambda_tag(lam)}"


def resolve_lambda_run_dir(root: Path, lam: float, seed: int) -> Path:
    """Canônica, ou o layout antigo da semente 0 se só ele tiver artefato."""

    canonical = lambda_run_dir(root, lam, seed)
    if int(seed) != 0 or has_run_artifacts(canonical):
        return canonical
    legacy = Path(root) / f"lambda-{lambda_tag(lam)}"
    if has_run_artifacts(legacy):
        return legacy
    return canonical


def baseline_seed_dir(corpus_root: Path, seed: int) -> Path:
    """Pasta canônica do denso, sem olhar o que já está no disco."""

    return Path(corpus_root) / seed_dirname(int(seed))


def resolve_baseline_seed_dir(corpus_root: Path, seed: int) -> Path:
    """Canônica, ou a raiz antiga da semente 0 se só ela tiver artefato."""

    root = Path(corpus_root)
    canonical = baseline_seed_dir(root, seed)
    if int(seed) != 0 or has_run_artifacts(canonical):
        return canonical
    if has_run_artifacts(root):
        return root
    return canonical


def prefer_checkpoint(canonical_dir: Path, legacy_dir: Path | None = None) -> Path:
    """Checkpoint da semente 0: pasta nova, senão a antiga, senão o caminho novo.

    Hold-out olha o arquivo, não o log. Um ``train.log`` em ``seed-0`` sem
    ``checkpoint.pt`` não esconde o checkpoint que ainda está no layout antigo.
    """

    canonical = Path(canonical_dir) / "checkpoint.pt"
    if canonical.is_file() or legacy_dir is None:
        return canonical
    legacy = Path(legacy_dir) / "checkpoint.pt"
    if legacy.is_file():
        return legacy
    return canonical


def per_seed_report(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Um item por semente, na ordem em que as corridas apareceram."""

    order: list[int] = []
    grouped: dict[int, list[dict[str, Any]]] = {}
    for run in runs:
        seed = int(run["seed"])
        if seed not in grouped:
            order.append(seed)
            grouped[seed] = []
        grouped[seed].append(run)
    report: list[dict[str, Any]] = []
    for seed in order:
        items = grouped[seed]
        statuses = [str(item["status"]) for item in items]
        if all(status == "done" for status in statuses):
            status = "done"
        elif any(status == "done" for status in statuses):
            status = "partial"
        else:
            status = "failed"
        report.append(
            {
                "seed": seed,
                "status": status,
                "work_dirs": [item["work_dir"] for item in items],
                "logs": [item.get("log") for item in items],
            }
        )
    return report
