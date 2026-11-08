"""Store de experts e o pacote dos pesos que ficam residentes.

O writer não acumula o modelo: cada ``write`` recebe um expert e solta
o buffer. O reader devolve um bloco de cada vez. Os dois recusam um
arquivo cujo cabeçalho não foi marcado completo — um repack interrompido
não vira silêncio.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import torch

from moe_memory.offload.direct_io import BlockFile
from moe_memory.offload.format import (
    DTYPE_BF16,
    DTYPE_F16,
    DTYPE_F32,
    ENTRY_SIZE,
    FLAG_COMPLETE,
    HEADER_SIZE,
    ExpertLayout,
    IndexEntry,
    StoreHeader,
    data_offset_for,
    pack_entry,
    pack_header,
    unpack_entry,
    unpack_header,
)
from moe_memory.offload.throttle import IoThrottle

RESIDENT_MAGIC = b"MOERESD1"
RESIDENT_VERSION = 1
_TORCH_DTYPE = {
    DTYPE_BF16: torch.bfloat16,
    DTYPE_F16: torch.float16,
    DTYPE_F32: torch.float32,
}


class ExpertStore:
    """Leitura de blocos. Uma instância, várias threads, ``pread``."""

    def __init__(self, path: str | Path, header: StoreHeader, entries: dict[tuple[int, int], IndexEntry], *, direct: bool) -> None:
        self.path = str(path)
        self.header = header
        self.entries = entries
        self.layout = header.layout
        self._file = BlockFile(self.path, direct=direct)

    @property
    def direct(self) -> bool:
        return self._file.direct

    @classmethod
    def open(cls, path: str | Path, *, direct: bool = True) -> ExpertStore:
        fd = os.open(path, os.O_RDONLY)
        try:
            blob = os.pread(fd, HEADER_SIZE, 0)
            header = unpack_header(blob)
            if not header.complete:
                raise ValueError(f"{path} não está marcado completo; o repack não terminou")
            index = os.pread(fd, header.index_nbytes, HEADER_SIZE)
        finally:
            os.close(fd)
        if len(index) != header.index_nbytes:
            raise ValueError("índice truncado")
        entries: dict[tuple[int, int], IndexEntry] = {}
        for i in range(header.n_blocks):
            entry = unpack_entry(index[i * ENTRY_SIZE : (i + 1) * ENTRY_SIZE])
            if entry.offset % header.align != 0:
                raise ValueError(f"bloco {(entry.layer, entry.expert)} fora do alinhamento")
            if entry.raw_nbytes != header.raw_nbytes:
                raise ValueError("raw_nbytes da entrada não é o do cabeçalho")
            key = entry.key()
            if key in entries:
                raise ValueError(f"entrada duplicada {key}")
            entries[key] = entry
        if len(entries) != header.n_blocks:
            raise ValueError("índice não cobre todos os experts")
        return cls(path, header, entries, direct=direct)

    def read_block(self, layer: int, expert: int) -> bytes:
        entry = self._entry(layer, expert)
        return self._file.read_at(entry.offset, self.header.block_nbytes)

    def read_into(self, layer: int, expert: int, dest: torch.Tensor) -> None:
        """Copia o bloco para um tensor ``uint8`` de ``block_nbytes``."""

        if dest.dtype != torch.uint8 or dest.numel() != self.header.block_nbytes:
            raise ValueError("destino precisa ser uint8 do tamanho do bloco")
        if not dest.is_contiguous():
            raise ValueError("destino precisa ser contíguo")
        if dest.device.type != "cpu":
            raise ValueError("a leitura do store cai numa vaga de CPU")
        entry = self._entry(layer, expert)
        size = self.header.block_nbytes
        ptr = dest.data_ptr()
        aligned = ptr % self.header.align == 0
        if self._file.direct and aligned:
            self._file.read_into(entry.offset, ptr, size, dest_aligned=True)
            return
        blob = self._file.read_at(entry.offset, size)
        dest.copy_(torch.frombuffer(bytearray(blob), dtype=torch.uint8))

    def payload(self, layer: int, expert: int) -> tuple[bytes, bytes, bytes]:
        entry = self._entry(layer, expert)
        raw = self.read_block(layer, expert)[: entry.raw_nbytes]
        gate = raw[: entry.gate_nbytes]
        up = raw[entry.gate_nbytes : entry.gate_nbytes + entry.up_nbytes]
        down = raw[entry.gate_nbytes + entry.up_nbytes :]
        return gate, up, down

    def close(self) -> None:
        self._file.close()

    def _entry(self, layer: int, expert: int) -> IndexEntry:
        key = (layer, expert)
        try:
            return self.entries[key]
        except KeyError as exc:
            raise KeyError(f"expert {key} não está no store") from exc

    def __enter__(self) -> ExpertStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class ExpertStoreWriter:
    """Grava o índice antes dos blocos e só marca completo no ``finish``."""

    def __init__(
        self,
        path: str | Path,
        *,
        n_layers: int,
        n_experts: int,
        hidden: int,
        intermediate: int,
        dtype_code: int,
        order: list[tuple[int, int]],
        throttle: IoThrottle | None = None,
    ) -> None:
        if n_layers < 1 or n_experts < 1:
            raise ValueError("camadas e experts precisam ser >= 1")
        if dtype_code not in _TORCH_DTYPE:
            raise ValueError(f"dtype_code {dtype_code} inválido")
        expected = {(layer, expert) for layer in range(n_layers) for expert in range(n_experts)}
        if set(order) != expected or len(order) != len(expected):
            raise ValueError("a ordem de escrita precisa listar cada expert uma vez")
        self.path = Path(path)
        self.layout = ExpertLayout.from_dims(hidden, intermediate, {DTYPE_BF16: 2, DTYPE_F16: 2, DTYPE_F32: 4}[dtype_code])
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.dtype_code = dtype_code
        self.order = list(order)
        self.throttle = throttle
        n_blocks = n_layers * n_experts
        self.data_offset = data_offset_for(n_blocks, 4096)
        self._offset_of: dict[tuple[int, int], int] = {}
        cursor = self.data_offset
        for key in self.order:
            self._offset_of[key] = cursor
            cursor += self.layout.block_nbytes
        self.store_bytes = cursor
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o644)
        if self.throttle is not None:
            self.throttle.add_fd(self._fd)
        header = StoreHeader(
            version=1,
            n_layers=n_layers,
            n_experts=n_experts,
            hidden=hidden,
            intermediate=intermediate,
            elem_size=self.layout.elem_size,
            dtype_code=dtype_code,
            align=4096,
            flags=0,
            raw_nbytes=self.layout.raw_nbytes,
            block_nbytes=self.layout.block_nbytes,
            index_nbytes=n_blocks * ENTRY_SIZE,
            data_offset=self.data_offset,
            n_blocks=n_blocks,
            resident_nbytes=0,
        )
        os.write(self._fd, pack_header(header))
        index = bytearray()
        for layer in range(n_layers):
            for expert in range(n_experts):
                index += pack_entry(
                    IndexEntry(
                        layer=layer,
                        expert=expert,
                        offset=self._offset_of[(layer, expert)],
                        raw_nbytes=self.layout.raw_nbytes,
                        gate_nbytes=self.layout.gate_nbytes,
                        up_nbytes=self.layout.up_nbytes,
                        down_nbytes=self.layout.down_nbytes,
                    )
                )
        os.write(self._fd, bytes(index))
        pad = self.data_offset - HEADER_SIZE - len(index)
        if pad:
            os.write(self._fd, bytes(pad))
        self._written: set[tuple[int, int]] = set()
        self._header = header

    def write(self, layer: int, expert: int, gate: bytes, up: bytes, down: bytes) -> None:
        key = (layer, expert)
        if key not in self._offset_of:
            raise KeyError(f"expert {key} fora do store")
        if key in self._written:
            raise ValueError(f"expert {key} gravado duas vezes")
        if len(gate) != self.layout.gate_nbytes or len(up) != self.layout.up_nbytes or len(down) != self.layout.down_nbytes:
            raise ValueError(
                f"projeções de {key} têm tamanho errado: "
                f"{len(gate)}, {len(up)}, {len(down)} vs "
                f"{self.layout.gate_nbytes}, {self.layout.up_nbytes}, {self.layout.down_nbytes}"
            )
        block = bytearray(self.layout.block_nbytes)
        block[: len(gate)] = gate
        block[len(gate) : len(gate) + len(up)] = up
        start = len(gate) + len(up)
        block[start : start + len(down)] = down
        offset = self._offset_of[key]
        view = memoryview(block)
        left = offset
        while view:
            n = os.pwrite(self._fd, view, left)
            if n <= 0:
                raise OSError(f"pwrite não avançou no expert {key}")
            view = view[n:]
            left += n
        self._written.add(key)
        if self.throttle is not None:
            self.throttle.account(self.layout.block_nbytes)

    def finish(self, *, resident_nbytes: int = 0) -> None:
        missing = self.n_layers * self.n_experts - len(self._written)
        if missing:
            raise RuntimeError(f"faltam {missing} experts; o store não foi marcado completo")
        if self.throttle is not None:
            self.throttle.finish()
        else:
            os.fsync(self._fd)
        done = StoreHeader(
            version=self._header.version,
            n_layers=self._header.n_layers,
            n_experts=self._header.n_experts,
            hidden=self._header.hidden,
            intermediate=self._header.intermediate,
            elem_size=self._header.elem_size,
            dtype_code=self._header.dtype_code,
            align=self._header.align,
            flags=FLAG_COMPLETE,
            raw_nbytes=self._header.raw_nbytes,
            block_nbytes=self._header.block_nbytes,
            index_nbytes=self._header.index_nbytes,
            data_offset=self._header.data_offset,
            n_blocks=self._header.n_blocks,
            resident_nbytes=resident_nbytes,
        )
        os.pwrite(self._fd, pack_header(done), 0)
        os.fsync(self._fd)
        self._drop_throttle()
        os.close(self._fd)
        self._fd = -1

    def abort(self) -> None:
        if self._fd >= 0:
            self._drop_throttle()
            os.close(self._fd)
            self._fd = -1

    def _drop_throttle(self) -> None:
        if self.throttle is None or self._fd < 0:
            return
        self.throttle.fds = [fd for fd in self.throttle.fds if fd != self._fd]


class ResidentWriter:
    """Pesos que não são expert, em sequência. O contador é remendado no fim."""

    def __init__(self, path: str | Path, throttle: IoThrottle | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "wb")
        self._fh.write(RESIDENT_MAGIC)
        self._fh.write(struct.pack("<III", RESIDENT_VERSION, 0, 0))
        self.count = 0
        self.nbytes = 0
        self.throttle = throttle
        self._names: set[str] = set()
        if throttle is not None:
            throttle.add_fd(self._fh.fileno())

    def _drop_throttle(self) -> None:
        if self.throttle is None or self._fh.closed:
            return
        fd = self._fh.fileno()
        self.throttle.fds = [got for got in self.throttle.fds if got != fd]

    def add(self, name: str, dtype_code: int, shape: tuple[int, ...], raw: bytes) -> None:
        if name in self._names:
            raise ValueError(f"tensor residente duplicado: {name}")
        if dtype_code not in _TORCH_DTYPE:
            raise ValueError(f"dtype residente inválido em {name}")
        expect = 1
        for dim in shape:
            if dim < 1:
                raise ValueError(f"shape inválido em {name}: {shape}")
            expect *= dim
        elem = {DTYPE_BF16: 2, DTYPE_F16: 2, DTYPE_F32: 4}[dtype_code]
        if len(raw) != expect * elem:
            raise ValueError(f"{name} tem {len(raw)} bytes; a shape pede {expect * elem}")
        encoded = name.encode("utf-8")
        blob = struct.pack("<I", len(encoded)) + encoded
        blob += struct.pack("<II", dtype_code, len(shape))
        blob += struct.pack("<" + "I" * len(shape), *shape)
        blob += struct.pack("<Q", len(raw)) + raw
        self._fh.write(blob)
        self._names.add(name)
        self.count += 1
        self.nbytes += len(raw)
        if self.throttle is not None:
            self.throttle.account(len(blob))

    def finish(self) -> None:
        self._fh.flush()
        if self.throttle is not None:
            self.throttle.finish()
        else:
            os.fsync(self._fh.fileno())
        self._fh.seek(len(RESIDENT_MAGIC))
        self._fh.write(struct.pack("<III", RESIDENT_VERSION, self.count, 0))
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._drop_throttle()
        self._fh.close()


class ResidentReader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fh = open(self.path, "rb")
        magic = self._fh.read(len(RESIDENT_MAGIC))
        if magic != RESIDENT_MAGIC:
            raise ValueError(f"{path} não é pacote residente")
        version, count, _reserved = struct.unpack("<III", self._fh.read(12))
        if version != RESIDENT_VERSION:
            raise ValueError(f"versão residente {version} não é suportada")
        self.count = count

    def __iter__(self):
        seen = 0
        while seen < self.count:
            (name_len,) = struct.unpack("<I", _read_exact(self._fh, 4))
            name = _read_exact(self._fh, name_len).decode("utf-8")
            dtype_code, ndim = struct.unpack("<II", _read_exact(self._fh, 8))
            shape = struct.unpack("<" + "I" * ndim, _read_exact(self._fh, 4 * ndim)) if ndim else ()
            (nbytes,) = struct.unpack("<Q", _read_exact(self._fh, 8))
            raw = _read_exact(self._fh, nbytes)
            seen += 1
            yield name, dtype_code, shape, raw
        extra = self._fh.read(1)
        if extra:
            raise ValueError("lixo depois do último tensor residente")

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> ResidentReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def torch_dtype(dtype_code: int) -> torch.dtype:
    try:
        return _TORCH_DTYPE[dtype_code]
    except KeyError as exc:
        raise ValueError(f"dtype_code {dtype_code} sem torch") from exc


def tensor_from_raw(raw: bytes, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    dest = torch.empty(shape, dtype=dtype)
    flat = dest.view(torch.uint8).reshape(-1)
    if flat.numel() != len(raw):
        raise ValueError(f"raw de {len(raw)} bytes não fecha com {shape} {dtype}")
    flat.copy_(torch.frombuffer(bytearray(raw), dtype=torch.uint8))
    return dest


def raw_from_tensor(tensor: torch.Tensor) -> tuple[int, tuple[int, ...], bytes]:
    if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"dtype {tensor.dtype} não entra no store")
    code = {torch.bfloat16: DTYPE_BF16, torch.float16: DTYPE_F16, torch.float32: DTYPE_F32}[tensor.dtype]
    contig = tensor.detach().to("cpu").contiguous()
    if contig.dtype == torch.float32:
        raw = contig.numpy().tobytes()
    elif contig.element_size() == 2:
        raw = contig.view(torch.int16).numpy().tobytes()
    else:
        raw = contig.view(torch.uint8).numpy().tobytes()
    return code, tuple(int(d) for d in contig.shape), raw


def _read_exact(fh, n: int) -> bytes:
    blob = fh.read(n)
    if len(blob) != n:
        raise ValueError(f"arquivo residente truncado: {len(blob)} de {n}")
    return blob
