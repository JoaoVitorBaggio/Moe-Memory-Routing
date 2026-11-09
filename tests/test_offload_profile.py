"""Perfil opt-in e traço pareado. CPU. CUDA só se existir."""

from __future__ import annotations

import json
import subprocess
import sys
import weakref
from pathlib import Path

import pytest
import torch

from moe_memory.lstm_ahead_router import TwoHorizonLstmRouter
from moe_memory.offload.bench import run_bench, run_paired
from moe_memory.offload.format import ExpertLayout
from moe_memory.offload.pool import live_nbytes
from moe_memory.offload.profile import DecodeProfile, percentile
from moe_memory.offload.runtime import OffloadRuntime
from moe_memory.offload.synthetic import write_synthetic_bundle
from moe_memory.offload.trace import (
    STORE_PREFIX_BYTES,
    ExecutionTrace,
    TraceDivergence,
    compare_traces,
    module_sha,
    store_sha,
)

ROOT = Path(__file__).resolve().parents[1]


def test_percentile_uses_nearest_rank_and_empty_is_null():
    assert percentile([], 50) is None
    assert percentile([1, 2, 3, 4], 50) == 2
    assert percentile([1, 2, 3, 4], 95) == 4


def test_profile_off_matches_profile_on_and_skips_device_sync(tmp_path: Path):
    bundle = write_synthetic_bundle(tmp_path / "model", n_layers=2, n_experts=4, seed=1)
    torch.manual_seed(2)
    router = _router()
    tokens = [1, 4, 2, 7]
    calls: list[str] = []
    original_sync = torch.cuda.synchronize

    def boom_device(*_args, **_kwargs):
        calls.append("device")
        raise AssertionError("torch.cuda.synchronize não entra no perfil")

    torch.cuda.synchronize = boom_device  # type: ignore[method-assign]
    try:
        _profile_pair(bundle, router, tokens, calls)
    finally:
        torch.cuda.synchronize = original_sync  # type: ignore[method-assign]


def _profile_pair(bundle: Path, router: TwoHorizonLstmRouter, tokens: list[int], calls: list[str]) -> None:
    with OffloadRuntime(
        bundle,
        router,
        lookahead=True,
        warm_slots=8,
        hot_slack=0,
        direct=False,
        warm_start=True,
    ) as plain:
        plain.decode(tokens)
        plain_layers = [list(row) for row in plain.executed_per_layer]
        plain_logits = plain.last_logits.detach().clone()
        plain_hot = [plan.hot_index.detach().clone() for plan in plain.plans]
    weights = {name: tensor.detach().clone() for name, tensor in router.state_dict().items()}
    profile = DecodeProfile(device=torch.device("cpu"), max_tokens=16)
    other = _router()
    other.load_state_dict(weights)
    with OffloadRuntime(
        bundle,
        other,
        lookahead=True,
        warm_slots=8,
        hot_slack=0,
        direct=False,
        warm_start=True,
        profile=profile,
    ) as traced:
        traced.decode(tokens)
        assert [list(row) for row in traced.executed_per_layer] == plain_layers
        assert torch.equal(traced.last_logits, plain_logits)
        for left, right in zip(plain_hot, traced.plans):
            assert torch.equal(left, right.hot_index)
        for row in traced.executed_per_layer:
            assert min(row) >= other.k1
    assert calls == []
    assert profile.resolve_stream_syncs == 0
    report = profile.report()
    assert report["compute_gpu"] is None
    assert report["summary"]["attention_gpu_s"] is None
    assert report["h2d"]["kind"] == "cpu_memcpy"
    assert report["h2d"]["gpu_event_s_sum"] is None
    assert report["h2d"]["additive_to_elapsed"] is False
    assert report["ssd_worker"]["additive_to_elapsed"] is False
    assert report["detail_truncated"] is False
    for row in report["per_token"]:
        parts = row["router_wall_s"] + row["schedule_enqueue_wall_s"] + row["forward_wall_s"]
        assert abs(row["token_wall_s"] - parts - row["unaccounted_wall_s"]) < 1e-6
        assert row["forward_wall_s"] + 1e-9 >= row["stall_warm_wall_s"] + row["stall_hot_wall_s"]
        assert row["attention_gpu_s"] is None
        assert row["expert_gpu_s"] is None
    summary = report["summary"]["router_wall_s"]
    assert summary["n"] == len(tokens)
    assert summary["early_p50"] is not None and summary["late_p50"] is not None


def test_profile_detail_is_bounded(tmp_path: Path):
    bundle = write_synthetic_bundle(tmp_path / "model", n_layers=2, n_experts=4, seed=2)
    torch.manual_seed(1)
    profile = DecodeProfile(device=torch.device("cpu"), max_tokens=2, max_io=1)
    with OffloadRuntime(
        bundle,
        _router(),
        lookahead=False,
        warm_slots=8,
        hot_slack=1,
        direct=False,
        profile=profile,
    ) as runtime:
        runtime.decode([1, 2, 3, 4])
    report = profile.report()
    assert len(report["per_token"]) == 2
    assert report["detail_truncated"] is True
    assert report["io_truncated"] is True
    assert report["ssd_worker"]["reads"] == 1


def test_ssd_service_time_is_not_the_stall(tmp_path: Path):
    bundle = write_synthetic_bundle(tmp_path / "model", n_layers=1, n_experts=4, hidden=8, intermediate=4, seed=3)
    torch.manual_seed(0)
    router = _router(n_layers=1, n_experts=4, k1=1, k2=1, b2=1)
    profile = DecodeProfile(device=torch.device("cpu"))
    with OffloadRuntime(
        bundle,
        router,
        lookahead=False,
        warm_slots=2,
        hot_slack=1,
        direct=False,
        n_warm_threads=1,
        profile=profile,
    ) as runtime:
        runtime.warm.store = _Slow(runtime.store, 0.05)
        runtime.decode([3, 4])
        assert runtime.stats.warm_start_s >= 0.04
    report = profile.report()
    assert report["ssd_worker"]["io_s_sum"] is not None
    assert report["ssd_worker"]["io_s_sum"] >= 0.04
    assert report["ssd_worker"]["queue_wait_s_sum"] is not None
    assert report["ssd_worker"]["queue_wait_s_sum"] >= 0.0


def test_unseeded_routers_differ_and_seeded_traces_match(tmp_path: Path):
    bundle = write_synthetic_bundle(tmp_path / "store", seed=4)
    common = dict(
        store_dir=bundle,
        dry_run=False,
        tokens=3,
        seed=4,
        l1=1,
        l2=2,
        k1=1,
        k2=2,
        b2=1,
        warm_slots=8,
        hot_slack=0,
        device="cpu",
        router_dim=16,
        trace=True,
        lookahead=True,
    )
    first = run_bench(tmp_path / "a.json", **common)
    second = run_bench(tmp_path / "b.json", **common)
    assert first["router_init"] == "global_rng"
    assert first["trace"]["tokens"] == second["trace"]["tokens"]
    assert first["trace"]["router_sha"] != second["trace"]["router_sha"]
    assert first["prefetch_enqueue_s"] is not None
    assert first["resident_load_s"] > 0
    assert first["timing_definitions"]["elapsed_s"]
    assert "load_resident" in first["timing_definitions"]["elapsed_s"]
    seeded_a = run_bench(tmp_path / "c.json", seed_router=True, **common)
    seeded_b = run_bench(tmp_path / "d.json", seed_router=True, **common)
    assert seeded_a["trace"]["fingerprint"] == seeded_b["trace"]["fingerprint"]
    assert seeded_a["profile"] is None


def test_paired_modes_share_the_execution_and_keep_phase_clocks(tmp_path: Path):
    payload = run_paired(
        tmp_path / "paired.json",
        dry_run=True,
        tokens=4,
        seed=5,
        l1=1,
        l2=2,
        k1=1,
        k2=2,
        b2=1,
        warm_slots=8,
        hot_slack=0,
        device="cpu",
        router_dim=16,
        profile=True,
        profile_max_tokens=16,
        n_warm_threads=1,
    )
    assert payload["execution_match"] is True
    assert payload["router_seeded"] is True
    assert payload["order"] == ["on-demand", "lookahead"]
    demand = payload["runs"]["on-demand"]
    ahead = payload["runs"]["lookahead"]
    assert demand["trace"]["fingerprint"] == ahead["trace"]["fingerprint"]
    assert demand["experts_executed_per_token"] == ahead["experts_executed_per_token"]
    assert demand["experts_executed_per_token"] == [demand["n_layers"] * demand["k1"]] * demand["tokens"]
    assert demand["prefetch_enqueue_s"] is None
    assert ahead["prefetch_enqueue_s"] >= 0.0
    assert demand["profile"]["compute_gpu"] is None
    assert "MemoryMax=7G" in payload["memory_run_note"]
    saved = json.loads((tmp_path / "paired.json").read_text(encoding="utf-8"))
    assert "_trace" not in saved["runs"]["on-demand"]
    for row in ahead["trace"]["steps"]:
        for layer, _expert, _weight, prefix in row:
            assert isinstance(layer, int)
            assert len(prefix) == 16


def test_trace_divergence_names_the_token_and_refuses_a_looser_budget():
    left = _manual_trace()
    right = _manual_trace()
    assert compare_traces(left, right) == left.fingerprint()
    right.steps[1][0][1] = 3
    with pytest.raises(TraceDivergence, match="token 1") as caught:
        compare_traces(left, right)
    assert caught.value.detail["token"] == 1
    other = _manual_trace()
    other.limits = dict(other.limits)
    other.limits["b2"] = 9
    with pytest.raises(TraceDivergence, match="tetos"):
        compare_traces(left, other)
    left.truncated = True
    with pytest.raises(TraceDivergence, match="truncado"):
        left.fingerprint()


def test_elapsed_covers_phases_without_replacing_tokens_per_s(tmp_path: Path):
    payload = run_bench(
        tmp_path / "one.json",
        dry_run=True,
        tokens=3,
        seed=6,
        lookahead=True,
        l1=1,
        l2=2,
        k1=1,
        k2=2,
        b2=1,
        warm_slots=8,
        device="cpu",
        profile=False,
    )
    assert payload["elapsed_s"] + 1e-6 >= (
        payload["warm_start_s"] + payload["prefetch_enqueue_s"] + payload["decode_loop_s"]
    )
    assert payload["tokens_per_s"] == pytest.approx(payload["tokens"] / payload["elapsed_s"])
    excluding = payload["tokens"] / (payload["elapsed_s"] - payload["warm_start_s"])
    assert payload["tokens_per_s_excluding_warm_start"] == pytest.approx(excluding)
    assert payload["profile"] is None
    assert payload["trace"] is None


def test_bench_script_paired_dry_run_writes_a_new_file(tmp_path: Path):
    out = tmp_path / "paired.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "bench_offload_decode.py"),
            "--dry-run",
            "--paired",
            "--profile",
            "--tokens",
            "2",
            "--out",
            str(out),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "MemoryMax=7G" in completed.stdout
    assert "phase=before-lookahead" in completed.stderr
    assert "tracked_pool_bytes=" in completed.stderr
    body = json.loads(out.read_text(encoding="utf-8"))
    assert body["execution_match"] is True
    assert body["runs"]["lookahead"]["mode"] == "lookahead"
    assert body["runs"]["on-demand"]["mode"] == "on-demand"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sem CUDA")
def test_cuda_profile_resolves_events_without_device_sync(tmp_path: Path):
    bundle = write_synthetic_bundle(tmp_path / "model", n_layers=1, n_experts=2, hidden=8, intermediate=4, seed=4)
    torch.manual_seed(0)
    device_syncs: list[str] = []
    original = torch.cuda.synchronize

    def wrapped(*args, **kwargs):
        device_syncs.append("device")
        return original(*args, **kwargs)

    torch.cuda.synchronize = wrapped  # type: ignore[method-assign]
    try:
        profile = DecodeProfile(device=torch.device("cuda"), max_tokens=8)
        router = _router(n_layers=1, n_experts=2, k1=1, k2=1, b2=1)
        with OffloadRuntime(
            bundle,
            router,
            lookahead=True,
            warm_slots=2,
            hot_slack=0,
            direct=False,
            device=torch.device("cuda"),
            profile=profile,
        ) as runtime:
            runtime.decode([1, 2])
            elapsed = runtime.stats.elapsed_s
        report = profile.report()
        assert profile.resolve_wall_s is not None
        assert report["compute_gpu"]["kind"] == "cuda_event"
        assert report["compute_gpu"]["attention_s_sum"] is not None
        assert report["h2d"]["kind"] == "cuda_event"
        assert report["h2d"]["gpu_event_s_sum"] is not None
        assert report["resolve_stream_syncs"] >= 1
        assert device_syncs == []
        assert elapsed > 0
    finally:
        torch.cuda.synchronize = original  # type: ignore[method-assign]


def test_qwen_warm_pool_arithmetic_is_one_copy_not_two():
    layout = ExpertLayout.from_dims(2048, 768, 2)
    assert layout.block_nbytes == 9_437_184
    one = 320 * layout.block_nbytes
    assert one == 3_019_898_880
    assert 2 * one == 6_039_797_760


def test_close_drops_the_pool_before_the_next_runtime(tmp_path: Path):
    bundle = write_synthetic_bundle(tmp_path / "model", n_layers=2, n_experts=4, seed=7)
    torch.manual_seed(0)
    base = live_nbytes()
    with OffloadRuntime(
        bundle,
        _router(),
        lookahead=False,
        warm_slots=8,
        hot_slack=0,
        direct=False,
        warm_start=True,
    ) as first:
        block = first.warm.pool.blocks[0]
        held = weakref.ref(block)
        del block
        first.decode([1, 2, 3])
        one = live_nbytes() - base
        assert one > 0
    assert held() is None
    assert live_nbytes() == base
    with OffloadRuntime(
        bundle,
        _router(),
        lookahead=True,
        warm_slots=8,
        hot_slack=0,
        direct=False,
        warm_start=True,
    ) as second:
        second.decode([1, 2, 3])
        assert live_nbytes() - base == one
    assert live_nbytes() == base


def test_paired_128_dry_run_releases_pools_between_modes(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    common = dict(
        dry_run=True,
        tokens=128,
        seed=0,
        l1=1,
        l2=2,
        k1=1,
        k2=2,
        b2=1,
        warm_slots=8,
        hot_slack=0,
        device="cpu",
        router_dim=16,
        n_warm_threads=1,
        profile_max_tokens=128,
    )
    control = run_paired(tmp_path / "control.json", profile=False, **common)
    traced = run_paired(tmp_path / "profile.json", profile=True, **common)
    err = capsys.readouterr().err
    assert "phase=before-lookahead" in err
    assert "phase=on-demand-before-runtime" in err
    assert "phase=lookahead-warm-allocated" in err
    for payload, label in ((control, "control"), (traced, "profile")):
        assert payload["execution_match"] is True, label
        assert payload["pool_overlap"]["second_pool_starts_with_first_still_live"] is False
        assert payload["pool_overlap"]["on_demand_pool_bytes_still_live"] == 0
        assert payload["runs"]["on-demand"]["experts_executed_per_token"] == [2] * 128
        assert payload["measurement_host"] == "synthetic-dry-run"
        assert "Não é igual ao RSS" in payload["memory_accounting"]["cgroup_memory_current_bytes"]
    assert control["runs"]["on-demand"]["profile"] is None
    assert traced["runs"]["lookahead"]["profile"]["enabled"] is True
    assert traced["runs"]["on-demand"]["trace"]["fingerprint"] == traced["runs"]["lookahead"]["trace"]["fingerprint"]


def test_store_sha_reads_only_the_prefix_of_a_huge_logical_file():
    """O store real tem ~55 GiB. Este teste não o materializa.

    O objeto mente o tamanho e recusa ``read`` sem teto e ``read_bytes``.
    Se o fingerprint voltar a ler o arquivo inteiro, a asserção dispara
    sem alocar essa RAM.
    """

    payload = bytes([index % 251 for index in range(STORE_PREFIX_BYTES + 128)])
    fake = _BoundedStore(payload, reported_size=55 * 1024**3)
    digest = store_sha(fake)  # type: ignore[arg-type]
    assert fake.read_args == [STORE_PREFIX_BYTES]
    assert fake.bytes_requested == STORE_PREFIX_BYTES
    assert fake.read_bytes_calls == 0
    assert digest == _expected_store_sha(fake.reported_size, payload[:STORE_PREFIX_BYTES])


def test_store_sha_on_a_real_file_ignores_bytes_past_the_prefix(tmp_path: Path):
    path = tmp_path / "experts.moestore"
    body = bytes([index % 251 for index in range(STORE_PREFIX_BYTES + 64)])
    path.write_bytes(body)
    original = store_sha(path)
    assert original == _expected_store_sha(len(body), body[:STORE_PREFIX_BYTES])
    mutated = bytearray(body)
    mutated[STORE_PREFIX_BYTES + 3] ^= 0xFF
    path.write_bytes(mutated)
    assert store_sha(path) == original
    mutated[0] ^= 0xFF
    path.write_bytes(mutated)
    assert store_sha(path) != original
    short = tmp_path / "short.moestore"
    short.write_bytes(b"abc")
    assert store_sha(short) == _expected_store_sha(3, b"abc")


def test_store_sha_does_not_call_path_read_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "experts.moestore"
    path.write_bytes(b"prefixo-curto")

    def boom(self: Path) -> bytes:
        del self
        raise AssertionError("Path.read_bytes lê o arquivo até o EOF")

    monkeypatch.setattr(Path, "read_bytes", boom)
    assert store_sha(path) == _expected_store_sha(len(b"prefixo-curto"), b"prefixo-curto")


def _expected_store_sha(size: int, prefix: bytes) -> str:
    import hashlib

    digest = hashlib.sha256()
    digest.update(str(size).encode())
    digest.update(prefix)
    return digest.hexdigest()


class _BoundedStore:
    """Arquivo lógico enorme. Só entrega o que ``read(n)`` pedir, com n >= 0."""

    def __init__(self, payload: bytes, reported_size: int) -> None:
        self.payload = payload
        self.reported_size = reported_size
        self.read_args: list[int] = []
        self.bytes_requested = 0
        self.read_bytes_calls = 0

    def stat(self):
        return type("Stat", (), {"st_size": self.reported_size})()

    def open(self, mode: str = "rb"):
        if "b" not in mode:
            raise AssertionError(mode)
        return self

    def read(self, n: int = -1) -> bytes:
        self.read_args.append(n)
        if n is None or n < 0:
            raise AssertionError("read sem teto materializaria o store")
        self.bytes_requested += n
        return self.payload[:n]

    def read_bytes(self) -> bytes:
        self.read_bytes_calls += 1
        raise AssertionError("read_bytes lê o arquivo até o EOF")

    def __enter__(self) -> "_BoundedStore":
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False


def _router(**overrides: int) -> TwoHorizonLstmRouter:
    spec = dict(
        vocab_size=64,
        n_layers=2,
        n_experts=4,
        l1=1,
        l2=2,
        k1=1,
        k2=2,
        b2=1,
        embed_dim=16,
        hidden_size=16,
    )
    spec.update(overrides)
    return TwoHorizonLstmRouter(**spec)


def _manual_trace() -> ExecutionTrace:
    torch.manual_seed(0)
    router = _router()
    trace = ExecutionTrace(
        limits={"k1": 1, "k2": 2, "b2": 1, "l1": 1, "l2": 2, "warm_slots": 8, "hot_slots": 2, "warm_start": True, "n_layers": 2, "n_experts": 4},
        tokens=[1, 2, 3],
        router_sha=module_sha(router),
        store_sha256="abc",
        max_tokens=8,
    )
    trace.set_initial([(0, 1), (1, 0)])
    trace.add_step([(0, 1, "00000000", "aaaaaaaaaaaaaaaa"), (1, 0, "00000000", "bbbbbbbbbbbbbbbb")])
    trace.add_step([(0, 1, "00000000", "aaaaaaaaaaaaaaaa"), (1, 2, "3f800000", "cccccccccccccccc")])
    trace.add_step([(0, 0, "00000000", "dddddddddddddddd"), (1, 2, "3f800000", "cccccccccccccccc")])
    return trace


class _Slow:
    def __init__(self, inner, delay: float) -> None:
        self.header = inner.header
        self._inner = inner
        self.delay = delay

    def read_into(self, layer: int, expert: int, dest: torch.Tensor) -> None:
        import time

        time.sleep(self.delay)
        self._inner.read_into(layer, expert, dest)
