"""systemd-run, caminhos absolutos e filtros do lote noturno."""

from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ACCENT_ROOT = Path("/home/jvbaggio/Área de trabalho/Repositórios/Moe-Memory-Routing")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "night_batch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scaffold(root: Path, night) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for rel in night.REQUIRED_SCRIPTS:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# stub\n", encoding="utf-8")
    ckpt = root / night.LSTM_CHECKPOINT
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "meta.json").write_text("{}", encoding="utf-8")


night_batch_under_test = _load("night_batch_cpu_tests")


def test_launch_injects_working_directory_with_accented_root():
    out_dir = ACCENT_ROOT / "outputs/night-paper-test"
    steps = night_batch_under_test.build_steps(out_dir, ACCENT_ROOT)
    cmd = night_batch_under_test.launch(steps[0]["cmd"], ACCENT_ROOT)
    wd = f"--working-directory={ACCENT_ROOT.resolve()}"
    assert wd in cmd
    assert cmd[cmd.index(wd) - 1] == "--wait"
    eval_path = ACCENT_ROOT.resolve() / "scripts" / "eval_ppl.py"
    assert str(eval_path) in cmd


def test_build_steps_use_absolute_paths_under_accent_root():
    out_dir = ACCENT_ROOT / "outputs/night-paper-20261113-120000"
    steps = night_batch_under_test.build_steps(out_dir, ACCENT_ROOT)
    first = steps[0]["cmd"]
    assert first[1] == str((ACCENT_ROOT / "scripts/eval_ppl.py").resolve())
    windows_flag = first.index("--windows-out") + 1
    assert first[windows_flag].startswith(str(out_dir.resolve()))


def test_only_filters_stages_and_warns_on_missing_a_for_b(capsys):
    out_dir = ROOT / "outputs/night-paper-only"
    steps = night_batch_under_test.select_steps(
        night_batch_under_test.build_steps(out_dir, ROOT),
        "etapa3",
    )
    assert {step["stage"] for step in steps} == {"b"}
    night_batch_under_test.warn_only_dependencies(steps)
    err = capsys.readouterr().err
    assert "janelas" in err and "etapa a" in err


def test_main_refuses_existing_out_dir(tmp_path: Path, monkeypatch):
    night = _load("night_batch_refuse_out")
    fake_root = tmp_path / "Área de Trabalho" / "Moe-Memory-Routing"
    _scaffold(fake_root, night)
    monkeypatch.setattr(night, "ROOT", fake_root)
    existing = fake_root / "outputs/night-paper-existente"
    existing.mkdir(parents=True)
    with pytest.raises(SystemExit) as exc:
        night.main(["--out-dir", str(existing.relative_to(fake_root))])
    assert "já existe" in str(exc.value)


def test_default_out_dir_is_dated():
    fixed = datetime(2026, 11, 13, 12, 34, 56)
    path = night_batch_under_test.default_out_dir(ROOT, fixed)
    assert path.name == "night-paper-20261113-123456"


@pytest.mark.parametrize(
    "only,expected_count",
    [
        ("etapa1,etapa2", 4),
        ("a32,b", 6),
        ("c", 1),
    ],
)
def test_only_accepts_aliases(only: str, expected_count: int):
    out_dir = ROOT / "outputs/night-paper-filter"
    steps = night_batch_under_test.select_steps(night_batch_under_test.build_steps(out_dir, ROOT), only)
    assert len(steps) == expected_count
