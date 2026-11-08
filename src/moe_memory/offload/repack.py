"""Safetensors → store de experts, em streaming.

Não há ``torch.load`` do modelo. De cada shard entra só o JSON do
cabeçalho; o corpo de um expert é lido na hora de gravar o bloco e
descartado. A page cache da fonte é solta com ``DONTNEED`` quando o
shard acaba. A escrita passa pelo ``IoThrottle``.

A flag de apagar a fonte é o único jeito de caber na máquina da nota
(~61 GB de checkpoint, ~44 GB livres). O padrão é não apagar. A checagem
de espaço acontece antes de criar o store: se não cabe, nada é escrito
e nada é apagado.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from moe_memory.offload.format import (
    DTYPE_NAME,
    ELEM_SIZE,
    ExpertLayout,
    dtype_code_from_safetensors,
)
from moe_memory.offload.space import SpaceError, SpaceReport, assess_space
from moe_memory.offload.store import ExpertStoreWriter, ResidentWriter
from moe_memory.offload.throttle import IoThrottle

_SPLIT = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)
_FUSED_UP = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.gate_up_proj$")
_FUSED_DOWN = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.down_proj$")
_PROJ = {"gate_proj": "gate", "up_proj": "up", "down_proj": "down"}


@dataclass(frozen=True)
class TensorRef:
    name: str
    dtype: str
    shape: tuple[int, ...]
    shard: Path
    start: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start


@dataclass
class RepackResult:
    report: SpaceReport
    out_dir: Path
    store_path: Path
    resident_path: Path
    n_experts_written: int
    deleted_shards: tuple[str, ...]
    direct_note: str


@dataclass
class _ExpertSrc:
    layer: int
    expert: int
    gate: TensorRef | None = None
    up: TensorRef | None = None
    down: TensorRef | None = None
    gate_up: TensorRef | None = None
    down_fused: TensorRef | None = None

    def shards(self) -> set[Path]:
        found = [item.shard for item in (self.gate, self.up, self.down, self.gate_up, self.down_fused) if item is not None]
        return set(found)

    def refs(self) -> list[TensorRef]:
        if self.gate is not None:
            return [self.gate, self.up, self.down]  # type: ignore[list-item]
        return [self.gate_up, self.down_fused]  # type: ignore[list-item]


class _Fds:
    def __init__(self) -> None:
        self._fd: dict[Path, int] = {}

    def pread(self, ref: TensorRef) -> bytes:
        return self.pread_at(ref.shard, ref.start, ref.nbytes)

    def pread_at(self, shard: Path, start: int, size: int) -> bytes:
        fd = self._fd.get(shard)
        if fd is None:
            fd = os.open(shard, os.O_RDONLY)
            self._fd[shard] = fd
        blob = os.pread(fd, size, start)
        if len(blob) != size:
            raise OSError(f"leitura curta de {shard}: {len(blob)} de {size}")
        return blob

    def drop(self, shard: Path) -> None:
        fd = self._fd.pop(shard, None)
        if fd is None:
            return
        if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)

    def close(self) -> None:
        for shard in list(self._fd):
            self.drop(shard)


def repack_model(
    model_dir: Path,
    out_dir: Path,
    *,
    delete_source: bool = False,
    max_bytes_per_sec: float | None = 80 * 1024 * 1024,
    fsync_every: int = 64 * 1024 * 1024,
    free_bytes: int | None = None,
    same_filesystem: bool | None = None,
    overwrite: bool = False,
    log=print,
) -> RepackResult:
    """Converte o checkpoint. ``delete_source`` falso não apaga shard."""

    model_dir = model_dir.resolve()
    out_dir = out_dir.resolve()
    if model_dir == out_dir:
        raise ValueError("o destino é outro diretório; o repack não reescreve os shards no lugar")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"diretório do modelo não existe: {model_dir}")
    shards = list_shards(model_dir)
    config = _read_config(model_dir)
    catalog = _scan(shards)
    plan = _plan(catalog, config)
    source_bytes = sum(shard.stat().st_size for shard in shards)
    largest = max(shard.stat().st_size for shard in shards)
    layout = plan.layout
    n_blocks = plan.n_layers * plan.n_experts
    from moe_memory.offload.format import data_offset_for

    store_bytes = data_offset_for(n_blocks) + n_blocks * layout.block_nbytes
    resident_bytes = plan.resident_bytes
    parent = out_dir if out_dir.exists() else _existing_parent(out_dir)
    if free_bytes is None:
        free_bytes = shutil.disk_usage(parent).free
    if same_filesystem is None:
        same_filesystem = os.stat(parent).st_dev == os.stat(model_dir).st_dev
    report = assess_space(
        source_bytes=source_bytes,
        store_bytes=store_bytes,
        resident_bytes=resident_bytes,
        free_bytes=free_bytes,
        largest_shard_bytes=largest,
        delete_source=delete_source,
        same_filesystem=same_filesystem,
    )
    log(report.message)
    if not report.fits:
        raise SpaceError(report)
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{out_dir} não está vazio; passe overwrite para substituir a saída")
    out_dir.mkdir(parents=True, exist_ok=True)
    if config is not None:
        (out_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    order = _drain_order(plan, shards)
    throttle = IoThrottle(max_bytes_per_sec=max_bytes_per_sec, fsync_every=fsync_every)
    store: ExpertStoreWriter | None = None
    resident: ResidentWriter | None = None
    fds = _Fds()
    deleted: list[str] = []
    finished: set[str] = set()
    try:
        store = ExpertStoreWriter(
            out_dir / "experts.moestore",
            n_layers=plan.n_layers,
            n_experts=plan.n_experts,
            hidden=plan.hidden,
            intermediate=plan.intermediate,
            dtype_code=plan.dtype_code,
            order=order,
            throttle=throttle,
        )
        resident = ResidentWriter(out_dir / "resident.bin", throttle=throttle)
        refs_left = dict(plan.refcount)
        shard_left: dict[Path, int] = defaultdict(int)
        for (shard, _name), count in refs_left.items():
            if count > 0:
                shard_left[shard] += 1

        def release(ref: TensorRef) -> None:
            key = (ref.shard, ref.name)
            refs_left[key] -= 1
            if refs_left[key] < 0:
                raise RuntimeError(f"tensor {ref.name} liberado demais")
            if refs_left[key] == 0:
                shard_left[ref.shard] -= 1
                if shard_left[ref.shard] == 0:
                    _finish_shard(ref.shard, fds, throttle, delete_source, deleted, finished, log)

        written = 0
        experts_at = {shard: [] for shard in shards}
        for key in order:
            src = plan.experts[key]
            experts_at[min(src.shards(), key=shards.index)].append(key)
        residents_at: dict[Path, list[TensorRef]] = {shard: [] for shard in shards}
        for ref in plan.resident:
            residents_at[ref.shard].append(ref)

        for shard in shards:
            for key in experts_at[shard]:
                src = plan.experts[key]
                gate, up, down = _load_expert(src, fds, layout, plan.dtype_name)
                store.write(src.layer, src.expert, gate, up, down)
                del gate, up, down
                for ref in src.refs():
                    release(ref)
                written += 1
                if written % 64 == 0 or written == n_blocks:
                    log(f"experts gravados: {written}/{n_blocks}")
            for ref in residents_at[shard]:
                resident.add(ref.name, plan.dtype_code_of(ref), ref.shape, fds.pread(ref))
                release(ref)
            if shard_left.get(shard, 0) == 0:
                _finish_shard(shard, fds, throttle, delete_source, deleted, finished, log)

        if written != n_blocks:
            raise RuntimeError(f"gravação incompleta: {written} de {n_blocks}")
        resident_bytes_real = resident.nbytes
        resident.finish()
        resident = None
        store.finish(resident_nbytes=resident_bytes_real)
        store = None
    except Exception:
        if store is not None:
            store.abort()
        raise
    finally:
        fds.close()

    manifest = {
        "n_layers": plan.n_layers,
        "n_experts": plan.n_experts,
        "hidden": plan.hidden,
        "intermediate": plan.intermediate,
        "dtype": plan.dtype_name,
        "block_nbytes": layout.block_nbytes,
        "raw_nbytes": layout.raw_nbytes,
        "align": 4096,
        "n_blocks": n_blocks,
        "store_bytes": store_bytes,
        "resident_bytes": resident_bytes_real,
        "source_bytes": source_bytes,
        "deleted_source": delete_source,
        "layout": "gate|up|down, bloco alinhado a 4096",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(
        "repack completo. "
        f"store {store_bytes} bytes, residentes {resident_bytes_real} bytes, "
        f"shards apagados: {len(deleted)}."
    )
    return RepackResult(
        report=report,
        out_dir=out_dir,
        store_path=out_dir / "experts.moestore",
        resident_path=out_dir / "resident.bin",
        n_experts_written=written,
        deleted_shards=tuple(deleted),
        direct_note="a leitura do store usa O_DIRECT quando o sistema aceita; o repack em si usa pread",
    )


def list_shards(model_dir: Path) -> list[Path]:
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        names = sorted(set(weight_map.values()))
        shards = [model_dir / name for name in names]
    else:
        shards = sorted(path for path in model_dir.glob("*.safetensors") if path.is_file())
    if not shards:
        raise FileNotFoundError(f"nenhum safetensors em {model_dir}")
    missing = [str(path) for path in shards if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"shard listado e ausente: {missing[0]}")
    return shards


@dataclass
class _Plan:
    n_layers: int
    n_experts: int
    hidden: int
    intermediate: int
    dtype_code: int
    dtype_name: str
    layout: ExpertLayout
    experts: dict[tuple[int, int], _ExpertSrc]
    resident: list[TensorRef]
    resident_bytes: int
    refcount: dict[tuple[Path, str], int]

    def dtype_code_of(self, ref: TensorRef) -> int:
        code = dtype_code_from_safetensors(ref.dtype)
        if code != self.dtype_code:
            raise ValueError(f"{ref.name} está em {ref.dtype} e o store em {self.dtype_name}")
        return code


def _plan(catalog: list[TensorRef], config: dict | None) -> _Plan:
    split: dict[tuple[int, int], dict[str, TensorRef]] = defaultdict(dict)
    fused_up: dict[int, TensorRef] = {}
    fused_down: dict[int, TensorRef] = {}
    other: list[TensorRef] = []
    for ref in catalog:
        match = _SPLIT.match(ref.name)
        if match:
            layer, expert, proj = int(match.group(1)), int(match.group(2)), _PROJ[match.group(3)]
            split[(layer, expert)][proj] = ref
            continue
        up = _FUSED_UP.match(ref.name)
        if up:
            fused_up[int(up.group(1))] = ref
            continue
        down = _FUSED_DOWN.match(ref.name)
        if down:
            fused_down[int(down.group(1))] = ref
            continue
        other.append(ref)

    complete_split = {key for key, parts in split.items() if set(parts) == {"gate", "up", "down"}}
    use_fused = not complete_split and bool(fused_up)
    if complete_split and len(complete_split) != len(split):
        bad = [key for key in split if key not in complete_split]
        raise ValueError(f"expert sem as três projeções: {bad[0]}")
    if use_fused:
        if set(fused_up) != set(fused_down):
            raise ValueError("gate_up_proj e down_proj fundidos não cobrem as mesmas camadas")
        experts = {
            (layer, expert): _ExpertSrc(layer=layer, expert=expert, gate_up=fused_up[layer], down_fused=fused_down[layer])
            for layer in fused_up
            for expert in range(_fused_experts(fused_up[layer]))
        }
        ignored = set()
    else:
        if not complete_split:
            raise ValueError(
                "nenhum expert model.layers.{l}.mlp.experts.{e}.{gate,up,down}_proj.weight"
            )
        experts = {
            key: _ExpertSrc(
                layer=key[0],
                expert=key[1],
                gate=split[key]["gate"],
                up=split[key]["up"],
                down=split[key]["down"],
            )
            for key in complete_split
        }
        ignored = {ref.name for ref in fused_up.values()} | {ref.name for ref in fused_down.values()}
        other = [ref for ref in other if ref.name not in ignored]

    layers = sorted({layer for layer, _expert in experts})
    if layers != list(range(layers[-1] + 1)):
        raise ValueError(f"camadas de expert não são 0..N-1: {layers[:8]}")
    n_layers = layers[-1] + 1
    counts = {layer: sum(1 for got, _e in experts if got == layer) for layer in layers}
    if len(set(counts.values())) != 1:
        raise ValueError(f"camadas com contagens diferentes de experts: {counts}")
    n_experts = counts[0]
    hidden, intermediate, dtype_name = _dims(experts, config, use_fused)
    dtype_code = dtype_code_from_safetensors(dtype_name)
    if config is not None:
        _check_config(config, n_layers, n_experts, hidden, intermediate)
    layout = ExpertLayout.from_dims(hidden, intermediate, ELEM_SIZE[dtype_code])
    _check_shapes(experts, hidden, intermediate, use_fused, layout)
    resident = [ref for ref in other if ref.name != "__metadata__"]
    refcount: dict[tuple[Path, str], int] = defaultdict(int)
    if use_fused:
        for layer, ref in fused_up.items():
            refcount[(ref.shard, ref.name)] += n_experts
        for layer, ref in fused_down.items():
            refcount[(ref.shard, ref.name)] += n_experts
    else:
        for src in experts.values():
            for ref in src.refs():
                refcount[(ref.shard, ref.name)] += 1
    for ref in resident:
        refcount[(ref.shard, ref.name)] += 1
    return _Plan(
        n_layers=n_layers,
        n_experts=n_experts,
        hidden=hidden,
        intermediate=intermediate,
        dtype_code=dtype_code,
        dtype_name=dtype_name,
        layout=layout,
        experts=experts,
        resident=resident,
        resident_bytes=sum(ref.nbytes for ref in resident),
        refcount=dict(refcount),
    )


def _dims(experts: dict[tuple[int, int], _ExpertSrc], config: dict | None, fused: bool) -> tuple[int, int, str]:
    src = next(iter(experts.values()))
    if not fused:
        assert src.gate is not None and src.down is not None
        if len(src.gate.shape) != 2 or len(src.down.shape) != 2:
            raise ValueError(f"projeção de expert sem rank 2: {src.gate.shape}")
        intermediate, hidden = src.gate.shape
        dtype_name = src.gate.dtype
    else:
        assert src.gate_up is not None
        if len(src.gate_up.shape) != 3:
            raise ValueError(f"gate_up fundido sem rank 3: {src.gate_up.shape}")
        _experts, twice, hidden = src.gate_up.shape
        if twice % 2 != 0:
            raise ValueError("gate_up fundido não parte em gate e up")
        intermediate = twice // 2
        dtype_name = src.gate_up.dtype
    if config is not None:
        hidden = int(config.get("hidden_size", hidden))
        intermediate = int(config.get("moe_intermediate_size", config.get("intermediate_size", intermediate)))
        dtype_name = _dtype_from_config(config, dtype_name)
    return hidden, intermediate, dtype_name


def _check_config(config: dict, n_layers: int, n_experts: int, hidden: int, intermediate: int) -> None:
    if int(config.get("num_hidden_layers", n_layers)) != n_layers:
        raise ValueError("num_hidden_layers do config não fecha com os experts achados")
    if int(config.get("num_experts", n_experts)) != n_experts:
        raise ValueError("num_experts do config não fecha com os experts achados")
    if int(config["hidden_size"]) != hidden:
        raise ValueError("hidden_size do config não fecha com o peso")
    moe = int(config.get("moe_intermediate_size", intermediate))
    if moe != intermediate:
        raise ValueError("moe_intermediate_size do config não fecha com o peso")
    mlp_only = config.get("mlp_only_layers") or []
    if mlp_only:
        raise ValueError(f"camadas densas não entram neste store: {mlp_only}")


def _check_shapes(
    experts: dict[tuple[int, int], _ExpertSrc],
    hidden: int,
    intermediate: int,
    fused: bool,
    layout: ExpertLayout,
) -> None:
    del layout
    for key, src in experts.items():
        if not fused:
            assert src.gate is not None and src.up is not None and src.down is not None
            if src.gate.shape != (intermediate, hidden) or src.up.shape != (intermediate, hidden):
                raise ValueError(f"{key} gate/up não é {(intermediate, hidden)}")
            if src.down.shape != (hidden, intermediate):
                raise ValueError(f"{key} down não é {(hidden, intermediate)}")
            if src.gate.dtype != src.up.dtype or src.gate.dtype != src.down.dtype:
                raise ValueError(f"{key} mistura dtype")
        else:
            assert src.gate_up is not None and src.down_fused is not None
            if src.gate_up.shape[1:] != (2 * intermediate, hidden):
                raise ValueError(f"camada {src.layer} gate_up não é [E, {2 * intermediate}, {hidden}]")
            if src.down_fused.shape[1:] != (hidden, intermediate):
                raise ValueError(f"camada {src.layer} down fundido não é [E, {hidden}, {intermediate}]")


def _fused_experts(ref: TensorRef) -> int:
    if len(ref.shape) != 3:
        raise ValueError(f"{ref.name} fundido precisa ter rank 3")
    return int(ref.shape[0])


def _dtype_from_config(config: dict, fallback: str) -> str:
    name = str(config.get("torch_dtype", "")).lower()
    table = {
        "bfloat16": "BF16",
        "bf16": "BF16",
        "float16": "F16",
        "fp16": "F16",
        "float32": "F32",
        "fp32": "F32",
    }
    if not name:
        return fallback
    if name not in table:
        raise ValueError(f"torch_dtype {name} do config não é BF16, F16 ou F32")
    code = table[name]
    if code != fallback:
        raise ValueError(f"config pede {code} e o tensor está em {fallback}")
    return code


def _drain_order(plan: _Plan, shards: list[Path]) -> list[tuple[int, int]]:
    by_min: dict[Path, list[tuple[int, int]]] = {shard: [] for shard in shards}
    for key, src in plan.experts.items():
        first = min(src.shards(), key=shards.index)
        by_min[first].append(key)
    order: list[tuple[int, int]] = []
    for shard in shards:
        chunk = sorted(by_min[shard])
        order.extend(chunk)
    if len(order) != len(plan.experts):
        raise RuntimeError("ordem de drenagem perdeu expert")
    return order


def _load_expert(src: _ExpertSrc, fds: _Fds, layout: ExpertLayout, dtype_name: str) -> tuple[bytes, bytes, bytes]:
    if src.gate is not None:
        gate, up, down = fds.pread(src.gate), fds.pread(src.up), fds.pread(src.down)
    else:
        assert src.gate_up is not None and src.down_fused is not None
        elem = ELEM_SIZE[dtype_code_from_safetensors(dtype_name)]
        row = 2 * layout.intermediate * layout.hidden * elem
        half = layout.gate_nbytes
        base = src.gate_up.start + src.expert * row
        gate = fds.pread_at(src.gate_up.shard, base, half)
        up = fds.pread_at(src.gate_up.shard, base + half, half)
        down_off = src.down_fused.start + src.expert * layout.down_nbytes
        down = fds.pread_at(src.down_fused.shard, down_off, layout.down_nbytes)
    if len(gate) != layout.gate_nbytes or len(up) != layout.up_nbytes or len(down) != layout.down_nbytes:
        raise ValueError(f"bytes do expert {(src.layer, src.expert)} não fecham com o layout")
    return gate, up, down


def _finish_shard(
    shard: Path,
    fds: _Fds,
    throttle: IoThrottle,
    delete_source: bool,
    deleted: list[str],
    finished: set[str],
    log,
) -> None:
    if str(shard) in finished:
        return
    finished.add(str(shard))
    throttle.sync()
    fds.drop(shard)
    if not delete_source:
        return
    os.remove(shard)
    dirfd = os.open(shard.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(dirfd)
    finally:
        os.close(dirfd)
    deleted.append(str(shard))
    log(f"shard apagado depois do fsync: {shard.name}")


def _scan(shards: list[Path]) -> list[TensorRef]:
    found: list[TensorRef] = []
    for shard in shards:
        found.extend(_shard_header(shard))
    if not found:
        raise ValueError("safetensors sem tensores")
    return found


def _shard_header(path: Path) -> list[TensorRef]:
    with path.open("rb") as fh:
        raw_len = fh.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{path} não tem cabeçalho safetensors")
        (header_len,) = __import__("struct").unpack("<Q", raw_len)
        if header_len > 512 * 1024 * 1024:
            raise ValueError(f"cabeçalho de {path} grande demais")
        header = json.loads(fh.read(header_len).decode("utf-8"))
    data0 = 8 + header_len
    refs: list[TensorRef] = []
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        shape = tuple(int(dim) for dim in meta["shape"])
        refs.append(
            TensorRef(
                name=name,
                dtype=str(meta["dtype"]),
                shape=shape,
                shard=path,
                start=data0 + int(start),
                end=data0 + int(end),
            )
        )
    return refs


def _read_config(model_dir: Path) -> dict | None:
    path = model_dir / "config.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _existing_parent(path: Path) -> Path:
    cursor = path
    while not cursor.exists():
        if cursor.parent == cursor:
            raise FileNotFoundError(f"não há diretório pai para {path}")
        cursor = cursor.parent
    return cursor


def dtype_label(code: int) -> str:
    return DTYPE_NAME[code]
