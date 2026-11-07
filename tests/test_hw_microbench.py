"""Cotas de L e K a partir do config. Sem CUDA o script não inventa banda."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "hw_microbench",
    ROOT / "scripts" / "hw_microbench.py",
)
hw = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules["hw_microbench"] = hw
_SPEC.loader.exec_module(hw)


# Dimensões do config.json público de allenai/OLMoE-1B-7B-0924.
# A conta é aritmética do SwiGLU; o teste não baixa peso.
_OLMOE_DIMS = {
    "model_type": "olmoe",
    "hidden_size": 2048,
    "intermediate_size": 1024,
    "num_hidden_layers": 16,
    "num_experts": 64,
    "num_experts_per_tok": 8,
    "torch_dtype": "bfloat16",
    "norm_topk_prob": False,
}


def _fields(stdout: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        found[key.strip()] = value.strip()
    return found


def _tiny_config(path: Path, *, dtype: str = "float32") -> Path:
    payload = {
        "model_type": "olmoe",
        "hidden_size": 8,
        "intermediate_size": 4,
        "num_hidden_layers": 2,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "torch_dtype": dtype,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_published_olmoe_dims_match_the_swiglu_byte_count():
    expert = hw.expert_size_from_config(_OLMOE_DIMS)
    assert expert.parameters == 3 * 2048 * 1024
    assert expert.dtype_bytes == 2
    assert expert.nbytes == 12_582_912
    assert expert.native_executed_slots == 8 * 16
    assert expert.nbytes == expert.parameters * 2


def test_dtype_argument_overrides_torch_dtype_without_guessing():
    expert = hw.expert_size_from_config(_OLMOE_DIMS, dtype_bytes_override=4)
    assert expert.dtype_bytes == 4
    assert expert.dtype_bytes_source == "argument"
    assert expert.nbytes == 3 * 2048 * 1024 * 4


def test_unknown_dtype_and_foreign_model_are_refused():
    broken = dict(_OLMOE_DIMS)
    broken["torch_dtype"] = "float8"
    with pytest.raises(ValueError, match="dtype-bytes"):
        hw.expert_size_from_config(broken)
    foreign = dict(_OLMOE_DIMS)
    foreign["model_type"] = "llama"
    with pytest.raises(ValueError, match="olmoe"):
        hw.expert_size_from_config(foreign)


def test_lookahead_and_slot_bounds_follow_the_ceil_and_floor():
    assert hw.lookahead_min(2, 100, 50.0, 1.0) == 4
    assert hw.lookahead_min(1, 10, 3.0, 1.0) == 4
    assert hw.lookahead_min(0, 384, 50.0, 1.0) == 0
    assert hw.memory_slot_sum_max(1000, 384) == 2
    assert hw.memory_slot_sum_max(383, 384) == 0
    assert hw.bandwidth_slot_max(50.0, 16, 1.0, 384) == 2
    assert hw.bandwidth_slot_max(50.0, 4, 1.0, 384) == 0
    with pytest.raises(ValueError):
        hw.lookahead_min(-1, 10, 1.0, 1.0)
    with pytest.raises(ValueError):
        hw.bandwidth_slot_max(0.0, 1, 1.0, 10)


def test_config_url_rejects_weights():
    with pytest.raises(ValueError):
        hw.assert_config_url(
            "https://huggingface.co/allenai/OLMoE-1B-7B-0924/resolve/main/model.safetensors"
        )
    assert hw.OLMOE_CONFIG_URL.endswith("/config.json")
    assert "safetensors" not in hw.OLMOE_CONFIG_URL


def test_gpu_measurements_skip_without_cuda():
    if torch.cuda.is_available():
        pytest.skip("há GPU; a medição seria real")
    bandwidth = hw.measure_pinned_h2d_bandwidth(128, iters=1, warmup=0)
    proxy = hw.measure_expert_gemm_proxy(8, 4, iters=1, warmup=0)
    assert bandwidth.status == "no_cuda"
    assert bandwidth.value is None
    assert proxy.status == "no_cuda"
    assert proxy.value is None
    with pytest.raises(ValueError):
        hw.measure_pinned_h2d_bandwidth(0, iters=1, warmup=0)


def test_cpu_script_does_not_crash_and_leaves_bounds_unavailable(tmp_path: Path):
    if torch.cuda.is_available():
        pytest.skip("há GPU; a medição seria real")
    config = _tiny_config(tmp_path / "config.json")
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "hw_microbench.py"), "--config", str(config)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Traceback" not in completed.stderr
    assert "medição de banda: não executada" in completed.stdout
    assert "medição de t_tok: não executada" in completed.stdout
    fields = _fields(completed.stdout)
    assert fields["config_source"] == str(config)
    assert fields["S_e_bytes"] == "384"
    assert fields["expert_parameters"] == "96"
    assert fields["cuda"] == "false"
    assert fields["BW_bytes_per_s"] == "unavailable"
    assert fields["BW_source"] == "no_cuda"
    assert fields["t_tok_s"] == "unavailable"
    assert fields["L_min"] == "unavailable"
    assert fields["K_mem_sum_max_slots"] == "unavailable"
    assert fields["K_bw_max_slots"] == "unavailable"
    assert fields["native_executed_slots"] == "4"
    assert "safetensors" not in completed.stdout


def test_cpu_script_prints_l_min_and_k_from_arguments(tmp_path: Path):
    config = _tiny_config(tmp_path / "config.json")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "hw_microbench.py"),
            "--config",
            str(config),
            "--m",
            "2",
            "--t-tok",
            "1",
            "--bw-bytes-per-s",
            "50",
            "--m-hot-bytes",
            "1000",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    fields = _fields(completed.stdout)
    # ceil(2 * 384 / (50 * 1)) = 16; floor(1000/384) = 2; floor(50*16/384) = 2.
    assert fields["S_e_bytes"] == "384"
    assert fields["BW_source"] == "argument"
    assert fields["t_tok_source"] == "argument"
    assert fields["L_min"] == "16"
    assert fields["K_bw_L_source"] == "L_min"
    assert fields["K_bw_L"] == "16"
    assert fields["K_mem_sum_max_slots"] == "2"
    assert fields["K_mem_ratio"] == "1000/384"
    assert fields["K_bw_max_slots"] == "2"
    assert fields["native_vs_memory"] == "exceeds"
    assert fields["native_vs_bandwidth"] == "exceeds"
    assert "medição de banda: não usada" in completed.stdout


def test_weight_file_and_bad_config_fail_without_a_traceback(tmp_path: Path):
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"not a weight")
    refused = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "hw_microbench.py"), "--config", str(weights)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "Traceback" not in refused.stderr
    assert "JSON" in refused.stderr or "json" in refused.stderr

    missing = tmp_path / "config.json"
    missing.write_text(json.dumps({"model_type": "olmoe", "hidden_size": 8}), encoding="utf-8")
    broken = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "hw_microbench.py"), "--config", str(missing)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert broken.returncode != 0
    assert "Traceback" not in broken.stderr
    assert "intermediate_size" in broken.stderr
