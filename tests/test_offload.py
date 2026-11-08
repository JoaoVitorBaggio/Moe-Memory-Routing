"""Store, prefetch e stall. CPU. CUDA fica skip sem GPU."""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from moe_memory.lstm_ahead_router import TwoHorizonLstmRouter
from moe_memory.offload.bench import run_bench
from moe_memory.offload.format import (
    ALIGN,
    DTYPE_F32,
    ExpertLayout,
    StoreHeader,
    pack_header,
    unpack_header,
)
from moe_memory.offload.pool import SlotPool
from moe_memory.offload.repack import repack_model
from moe_memory.offload.runtime import OffloadRuntime
from moe_memory.offload.schedule import lookahead_plan
from moe_memory.offload.space import (
    REFERENCE_CHECKPOINT_BYTES,
    REFERENCE_FREE_BYTES,
    SpaceError,
    assess_space,
)
from moe_memory.offload.store import ExpertStore, ExpertStoreWriter
from moe_memory.offload.synthetic import write_synthetic_bundle
from moe_memory.offload.throttle import IoThrottle
from moe_memory.offload.warm import WarmTier
from moe_memory.tier_mask import plan_from_scores

ROOT = Path(__file__).resolve().parents[1]


def test_header_roundtrip_marks_alignment():
    header = StoreHeader(
        version=1,
        n_layers=2,
        n_experts=3,
        hidden=8,
        intermediate=4,
        elem_size=4,
        dtype_code=DTYPE_F32,
        align=ALIGN,
        flags=1,
        raw_nbytes=384,
        block_nbytes=ALIGN,
        index_nbytes=6 * 32,
        data_offset=ALIGN,
        n_blocks=6,
        resident_nbytes=0,
    )
    assert len(pack_header(header)) == 256
    assert unpack_header(pack_header(header)) == header


def test_block_is_aligned_and_index_roundtrips(tmp_path: Path):
    hidden, intermediate = 8, 4
    layout = ExpertLayout.from_dims(hidden, intermediate, 4)
    assert layout.raw_nbytes % ALIGN != 0
    assert layout.block_nbytes % ALIGN == 0
    order = [(0, 1), (1, 0), (0, 0), (1, 1)]
    path = tmp_path / "experts.moestore"
    writer = ExpertStoreWriter(
        path,
        n_layers=2,
        n_experts=2,
        hidden=hidden,
        intermediate=intermediate,
        dtype_code=DTYPE_F32,
        order=order,
    )
    payloads = {}
    for layer, expert in order:
        gate = bytes([(layer * 10 + expert) & 0xFF]) * layout.gate_nbytes
        up = bytes([(layer * 10 + expert + 3) & 0xFF]) * layout.up_nbytes
        down = bytes([(layer * 10 + expert + 7) & 0xFF]) * layout.down_nbytes
        writer.write(layer, expert, gate, up, down)
        payloads[(layer, expert)] = (gate, up, down)
    writer.finish()
    store = ExpertStore.open(path, direct=False)
    try:
        assert store.header.complete
        assert store.header.data_offset % ALIGN == 0
        assert store.header.block_nbytes % ALIGN == 0
        assert list(store.entries) == [(0, 0), (0, 1), (1, 0), (1, 1)]
        for key, parts in payloads.items():
            entry = store.entries[key]
            assert entry.offset % ALIGN == 0
            assert store.payload(*key) == parts
            block = store.read_block(*key)
            assert block[layout.raw_nbytes :] == bytes(layout.block_nbytes - layout.raw_nbytes)
            assert block[: layout.gate_nbytes] == parts[0]
    finally:
        store.close()
    direct = ExpertStore.open(path, direct=True)
    try:
        assert direct.payload(0, 1) == payloads[(0, 1)]
    finally:
        direct.close()


def test_incomplete_store_is_refused(tmp_path: Path):
    path = tmp_path / "experts.moestore"
    writer = ExpertStoreWriter(
        path,
        n_layers=1,
        n_experts=1,
        hidden=4,
        intermediate=4,
        dtype_code=DTYPE_F32,
        order=[(0, 0)],
    )
    layout = writer.layout
    writer.write(0, 0, bytes(layout.gate_nbytes), bytes(layout.up_nbytes), bytes(layout.down_nbytes))
    writer.abort()
    with pytest.raises(ValueError, match="completo"):
        ExpertStore.open(path, direct=False)


def test_throttle_fsyncs_in_chunks_and_sleeps_to_the_cap(tmp_path: Path):
    path = tmp_path / "blob"
    fd = path.open("wb").close() or os_open(path)
    syncs: list[int] = []
    slept: list[float] = []
    clock = {"t": 0.0}
    throttle = IoThrottle(
        fds=[fd],
        max_bytes_per_sec=1000,
        fsync_every=100,
        clock=lambda: clock["t"],
        sleep=slept.append,
        fsync=syncs.append,
        advise=lambda _fd: None,
    )
    throttle.account(250)
    assert syncs == [fd, fd]
    assert slept == pytest.approx([0.25])
    os_close(fd)


def test_reference_disk_does_not_fit_without_deleting_the_source():
    report = assess_space(
        source_bytes=REFERENCE_CHECKPOINT_BYTES,
        store_bytes=58_000_000_000,
        resident_bytes=3_000_000_000,
        free_bytes=REFERENCE_FREE_BYTES,
        largest_shard_bytes=4_000_000_000,
        delete_source=False,
        same_filesystem=True,
    )
    assert not report.fits
    assert "delete-source-after-each-shard" in report.message
    assert "61000000000" in report.message or "61.00 GB" in report.message
    assert "44.00 GB" in report.message
    with_delete = assess_space(
        source_bytes=REFERENCE_CHECKPOINT_BYTES,
        store_bytes=58_000_000_000,
        resident_bytes=3_000_000_000,
        free_bytes=REFERENCE_FREE_BYTES,
        largest_shard_bytes=4_000_000_000,
        delete_source=True,
        same_filesystem=True,
    )
    assert with_delete.fits
    other_disk = assess_space(
        source_bytes=100,
        store_bytes=500,
        resident_bytes=50,
        free_bytes=200,
        largest_shard_bytes=40,
        delete_source=True,
        same_filesystem=False,
    )
    assert not other_disk.fits


def test_repack_streams_one_expert_and_can_delete_the_shard(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    hidden, intermediate = 4, 4
    gate0 = torch.arange(intermediate * hidden, dtype=torch.float32).reshape(intermediate, hidden)
    up0 = gate0 + 1
    down0 = torch.arange(hidden * intermediate, dtype=torch.float32).reshape(hidden, intermediate) + 3
    gate1 = gate0 + 10
    up1 = up0 + 10
    down1 = down0 + 10
    embed = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    norm = torch.ones(2)
    write_safetensors(
        model / "model-00001-of-00002.safetensors",
        {
            "model.layers.0.mlp.experts.0.gate_proj.weight": gate0,
            "model.layers.0.mlp.experts.0.up_proj.weight": up0,
            "model.embed_tokens.weight": embed,
        },
    )
    write_safetensors(
        model / "model-00002-of-00002.safetensors",
        {
            "model.layers.0.mlp.experts.0.down_proj.weight": down0,
            "model.layers.0.mlp.experts.1.gate_proj.weight": gate1,
            "model.layers.0.mlp.experts.1.up_proj.weight": up1,
            "model.layers.0.mlp.experts.1.down_proj.weight": down1,
            "model.layers.0.mlp.gate.weight": torch.zeros(2, 2),
            "model.norm.weight": norm,
        },
    )
    (model / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": hidden,
                "moe_intermediate_size": intermediate,
                "num_hidden_layers": 1,
                "num_experts": 2,
                "torch_dtype": "float32",
            }
        ),
        encoding="utf-8",
    )
    (model / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.0.mlp.experts.0.gate_proj.weight": "model-00001-of-00002.safetensors",
                    "model.layers.0.mlp.experts.0.up_proj.weight": "model-00001-of-00002.safetensors",
                    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                    "model.layers.0.mlp.experts.0.down_proj.weight": "model-00002-of-00002.safetensors",
                    "model.layers.0.mlp.experts.1.gate_proj.weight": "model-00002-of-00002.safetensors",
                    "model.layers.0.mlp.experts.1.up_proj.weight": "model-00002-of-00002.safetensors",
                    "model.layers.0.mlp.experts.1.down_proj.weight": "model-00002-of-00002.safetensors",
                    "model.layers.0.mlp.gate.weight": "model-00002-of-00002.safetensors",
                    "model.norm.weight": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "store"
    with pytest.raises(SpaceError):
        repack_model(
            model,
            out,
            delete_source=False,
            free_bytes=0,
            same_filesystem=True,
            max_bytes_per_sec=None,
            log=lambda *_args, **_kwargs: None,
        )
    assert not (out / "experts.moestore").exists()
    assert (model / "model-00001-of-00002.safetensors").exists()
    result = repack_model(
        model,
        out,
        delete_source=True,
        free_bytes=10**12,
        same_filesystem=True,
        max_bytes_per_sec=None,
        fsync_every=32,
        log=lambda *_args, **_kwargs: None,
    )
    assert result.n_experts_written == 2
    assert not (model / "model-00001-of-00002.safetensors").exists()
    assert not (model / "model-00002-of-00002.safetensors").exists()
    store = ExpertStore.open(out / "experts.moestore", direct=False)
    try:
        got_gate, got_up, got_down = store.payload(0, 0)
        assert got_gate == gate0.numpy().tobytes()
        assert got_up == up0.numpy().tobytes()
        assert got_down == down0.numpy().tobytes()
        assert store.entries[(0, 0)].offset % ALIGN == 0
    finally:
        store.close()
    names = resident_names(out / "resident.bin")
    assert "model.embed_tokens.weight" in names
    assert "model.norm.weight" in names
    assert "model.layers.0.mlp.gate.weight" in names
    assert not any("experts" in name for name in names)


def test_slot_pool_evicts_only_the_named_expert():
    pool = SlotPool(2, 16)
    assert pool.reserve((0, 0)) is not None
    assert pool.reserve((0, 1)) is not None
    assert pool.reserve((1, 0)) is None
    assert pool.evict((0, 0))
    assert (0, 0) not in pool
    assert (0, 1) in pool
    assert pool.reserve((1, 0)) is not None
    assert not pool.evict((9, 9))


def test_prefetch_follows_tierplan_order_and_release_frees_the_slot(tmp_path: Path):
    bundle = write_bundle(tmp_path, n_layers=1, n_experts=4, hidden=8, intermediate=4)
    store = ExpertStore.open(bundle / "experts.moestore", direct=False)
    scores = torch.tensor(
        [
            [0.1, 0.4, 0.2, 0.3],
            [0.1, 0.05, 0.9, 0.3],
            [0.8, 0.1, 0.2, 0.05],
        ]
    ).view(1, 3, 1, 4)
    plan, _state = plan_from_scores(scores, scores.clone(), k1=1, k2=1, b2=1, l1=1, l2=2)
    entered = [plan.prefetch(0, step).warm for step in range(3)]
    assert entered == [((0, 1),), ((0, 2),), ((0, 0),)]
    warm = WarmTier(store, n_slots=3, n_threads=1)
    try:
        order = [(step, key) for step, keys in enumerate(entered) for key in keys]
        warm.sync(order)
        warm.drain()
        assert warm.read_order == [(0, 1), (0, 2), (0, 0)]
        assert warm.is_ready((0, 0))
        released = plan.prefetch(0, 2).warm_release
        assert (0, 2) in released
        warm.sync([(0, (0, 0))])
        warm.drain()
        assert (0, 1) not in warm.pool
        assert (0, 2) not in warm.pool
        assert warm.is_ready((0, 0))
    finally:
        warm.close()
        store.close()


def test_lookahead_window_matches_the_next_committed_plan():
    router = _router(n_layers=2, n_experts=4, l1=1, l2=2, k1=1, k2=2, b2=1)
    window = lookahead_plan(router, None)
    plan, state = router.plan(torch.tensor([[7]]), None)
    assert torch.equal(window.plan.warm_mask[:, 0], plan.warm_mask[:, 0])
    assert torch.equal(window.plan.hot_mask[:, 0], plan.hot_mask[:, 0])
    assert window.hot_steps == 1
    nxt = lookahead_plan(router, state)
    plan2, state2 = router.plan(torch.tensor([[8]]), state)
    assert torch.equal(nxt.plan.warm_mask[:, 0], plan2.warm_mask[:, 0])
    assert torch.equal(nxt.plan.hot_mask[:, 0], plan2.hot_mask[:, 0])
    assert state2.tier is not None and state2.tier.step == 2


def test_on_demand_stalls_and_lookahead_can_arrive_first(tmp_path: Path):
    bundle = write_bundle(tmp_path, n_layers=1, n_experts=4, hidden=8, intermediate=4)
    router = _router(n_layers=1, n_experts=4, l1=1, l2=2, k1=1, k2=1, b2=1, vocab_size=64)
    on_demand = OffloadRuntime(
        bundle,
        router,
        lookahead=False,
        warm_slots=1,
        hot_slack=1,
        direct=False,
        n_warm_threads=1,
    )
    try:
        on_demand.warm.store = SlowStore(on_demand.store, 0.08)
        on_demand.decode([3])
        assert on_demand.stats.warm_misses >= 1
        assert on_demand.stats.stall_warm_s >= 0.05
        assert on_demand.stats.ssd_bytes >= on_demand.store.header.raw_nbytes
        assert on_demand.stats.experts_executed >= 1
    finally:
        on_demand.close()

    ahead_router = _router(n_layers=1, n_experts=4, l1=1, l2=2, k1=1, k2=1, b2=1, vocab_size=64)
    ahead = OffloadRuntime(
        bundle,
        ahead_router,
        lookahead=True,
        warm_slots=4,
        hot_slack=2,
        direct=False,
        n_warm_threads=1,
    )
    try:
        ahead._prefetch_initial()
        ahead.warm.drain()
        ahead.hot.drain()
        loads = ahead.stats.warm_loads
        assert loads >= 1
        ahead.decode([3])
        assert ahead.stats.warm_misses == 0
        assert ahead.stats.ssd_bytes >= ahead.store.header.raw_nbytes
    finally:
        ahead.close()


def test_decode_logits_are_finite_and_dry_run_json_has_the_counters(tmp_path: Path):
    bundle = write_bundle(tmp_path / "model")
    router = _router(n_layers=2, n_experts=4, l1=1, l2=2, k1=1, k2=2, b2=2, vocab_size=64)
    with OffloadRuntime(
        bundle,
        router,
        lookahead=True,
        warm_slots=8,
        hot_slack=2,
        direct=False,
    ) as runtime:
        runtime.decode([1, 4, 2])
        assert runtime.last_logits is not None
        assert runtime.last_logits.shape == (1, 64)
        assert bool(torch.isfinite(runtime.last_logits).all())
        assert runtime.stats.tokens == 3
        assert runtime.stats.ram_peak_bytes > 0
        assert runtime.stats.vram_peak_bytes == 0
        assert runtime.stats.experts_executed >= 1
    out = tmp_path / "bench.json"
    payload = run_bench(
        out,
        dry_run=True,
        tokens=3,
        seed=1,
        lookahead=False,
        l1=1,
        l2=2,
        k1=1,
        k2=2,
        b2=1,
        warm_slots=8,
        hot_slack=1,
        device="cpu",
    )
    assert out.is_file()
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["mode"] == "on-demand"
    assert saved["dry_run"] is True
    for key in (
        "tokens_per_s",
        "stall_warm_s",
        "stall_hot_s",
        "ssd_bytes",
        "ram_peak_bytes",
        "vram_peak_bytes",
        "warm_misses",
        "hot_misses",
    ):
        assert key in saved
    assert "61" in payload["disk_note"] and "44" in payload["disk_note"]
    assert saved["tokens"] == 3


def test_bench_script_dry_run(tmp_path: Path):
    out = tmp_path / "out.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "bench_offload_decode.py"),
            "--dry-run",
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
    assert "61" in completed.stdout and "44" in completed.stdout
    body = json.loads(out.read_text(encoding="utf-8"))
    assert body["dry_run"] is True
    assert body["mode"] == "lookahead"
    assert body["tokens"] == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sem CUDA")
def test_hot_copy_uses_a_side_stream(tmp_path: Path):
    from moe_memory.offload.hot import HotTier

    bundle = write_bundle(tmp_path, n_layers=1, n_experts=2, hidden=8, intermediate=4)
    store = ExpertStore.open(bundle / "experts.moestore", direct=False)
    warm = WarmTier(store, n_slots=2, n_threads=1)
    hot = HotTier(
        warm,
        n_slots=2,
        layout=store.layout,
        dtype=torch.float32,
        device=torch.device("cuda"),
    )
    try:
        assert hot.stream is not None
        assert hot.stream != torch.cuda.current_stream()
        warm.sync([(0, (0, 0))])
        warm.drain()
        hot.sync([(0, (0, 0))])
        hot.drain()
        gate, _up, _down = hot.views((0, 0))
        assert gate.is_cuda
        assert gate.shape == (4, 8)
    finally:
        hot.close()
        warm.close()
        store.close()


def test_repack_splits_a_fused_expert_tensor(tmp_path: Path):
    model = tmp_path / "fused"
    model.mkdir()
    hidden, intermediate, n_experts = 4, 4, 2
    gate_up = torch.arange(n_experts * 2 * intermediate * hidden, dtype=torch.float32).reshape(
        n_experts, 2 * intermediate, hidden
    )
    down = torch.arange(n_experts * hidden * intermediate, dtype=torch.float32).reshape(
        n_experts, hidden, intermediate
    )
    write_safetensors(
        model / "model.safetensors",
        {
            "model.layers.0.mlp.experts.gate_up_proj": gate_up,
            "model.layers.0.mlp.experts.down_proj": down,
            "model.norm.weight": torch.ones(hidden),
        },
    )
    out = tmp_path / "out"
    repack_model(
        model,
        out,
        free_bytes=10**12,
        same_filesystem=True,
        max_bytes_per_sec=None,
        log=lambda *_args, **_kwargs: None,
    )
    store = ExpertStore.open(out / "experts.moestore", direct=False)
    try:
        gate, up, down_bytes = store.payload(0, 1)
        assert gate == gate_up[1, :intermediate].contiguous().numpy().tobytes()
        assert up == gate_up[1, intermediate:].contiguous().numpy().tobytes()
        assert down_bytes == down[1].contiguous().numpy().tobytes()
    finally:
        store.close()


class SlowStore:
    def __init__(self, inner: ExpertStore, delay: float) -> None:
        self.header = inner.header
        self._inner = inner
        self.delay = delay

    def read_into(self, layer: int, expert: int, dest: torch.Tensor) -> None:
        time.sleep(self.delay)
        self._inner.read_into(layer, expert, dest)


def write_bundle(path: Path, **overrides: int) -> Path:
    spec = dict(
        n_layers=2,
        n_experts=4,
        hidden=32,
        intermediate=16,
        vocab=64,
        n_heads=4,
        n_kv=2,
        head_dim=8,
        seed=0,
    )
    spec.update(overrides)
    return write_synthetic_bundle(path, **spec)


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


def write_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    header: dict[str, dict] = {}
    blobs: list[bytes] = []
    offset = 0
    for name, tensor in tensors.items():
        raw = tensor.detach().contiguous().numpy().tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)
        blobs.append(raw)
    payload = json.dumps(header).encode("utf-8")
    payload += b" " * ((8 - len(payload) % 8) % 8)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(payload)))
        handle.write(payload)
        for blob in blobs:
            handle.write(blob)


def resident_names(path: Path) -> list[str]:
    from moe_memory.offload.store import ResidentReader

    with ResidentReader(path) as reader:
        return [name for name, _code, _shape, _raw in reader]


def os_open(path: Path) -> int:
    import os

    return os.open(path, os.O_RDWR | os.O_CREAT, 0o644)


def os_close(fd: int) -> None:
    import os

    os.close(fd)
