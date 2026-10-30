"""Sem CUDA o cronômetro não inventa milissegundo."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def test_wall_script_refuses_to_invent_a_measurement_without_cuda():
    if torch.cuda.is_available():
        pytest.skip("há GPU; a medição local seria real")
    completed = subprocess.run(
        [sys.executable, str(ROOT / "experiments" / "wall_latency.py")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "1060" in completed.stdout
    assert "medição: não executada" in completed.stdout
    assert "p50_ms" not in completed.stdout
    assert "p95_ms" not in completed.stdout
