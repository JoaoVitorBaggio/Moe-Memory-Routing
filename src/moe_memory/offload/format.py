"""Layout do store de experts. Um bloco por ``(camada, expert)``.

O arquivo começa com um cabeçalho de 256 bytes e um índice fixo. Os
dados começam no próximo múltiplo de 4 KiB. Cada bloco é
``gate | up | down`` e o tamanho dele também é múltiplo de 4 KiB, para
a leitura com ``O_DIRECT`` não precisar de um segundo buffer de ajuste.

O índice não carrega peso. A ordem dos blocos no arquivo pode não ser
a ordem ``(camada, expert)``: o repack grava na ordem em que consegue
soltar um shard. O índice guarda o offset.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

ALIGN = 4096
MAGIC = b"MOESTOR1"
VERSION = 1
HEADER_SIZE = 256
ENTRY_SIZE = 32
FLAG_COMPLETE = 1

DTYPE_BF16 = 1
DTYPE_F16 = 2
DTYPE_F32 = 3

DTYPE_NAME = {
    DTYPE_BF16: "BF16",
    DTYPE_F16: "F16",
    DTYPE_F32: "F32",
}
NAME_TO_DTYPE = {name: code for code, name in DTYPE_NAME.items()}
ELEM_SIZE = {
    DTYPE_BF16: 2,
    DTYPE_F16: 2,
    DTYPE_F32: 4,
}

# magic, version, n_layers, n_experts, hidden, intermediate, elem_size,
# dtype_code, align, flags, reserved,
# raw_nbytes, block_nbytes, index_nbytes, data_offset, n_blocks, resident_nbytes
_HEADER = struct.Struct("<8s10I6Q")
# layer, expert, offset, raw, gate, up, down. 4+4+8+4+4+4+4 = 32.
_ENTRY = struct.Struct("<IIQIIII")
assert _ENTRY.size == ENTRY_SIZE
assert _HEADER.size <= HEADER_SIZE


@dataclass(frozen=True)
class ExpertLayout:
    """Fatias dentro do bloco. ``block_nbytes`` inclui o padding."""

    hidden: int
    intermediate: int
    elem_size: int
    gate_nbytes: int
    up_nbytes: int
    down_nbytes: int
    raw_nbytes: int
    block_nbytes: int

    @classmethod
    def from_dims(cls, hidden: int, intermediate: int, elem_size: int, align: int = ALIGN) -> ExpertLayout:
        if hidden < 1 or intermediate < 1 or elem_size < 1:
            raise ValueError("hidden, intermediário e elem_size precisam ser >= 1")
        if align < 1 or align % 512 != 0:
            raise ValueError("alinhamento precisa ser múltiplo de 512")
        gate = intermediate * hidden * elem_size
        up = gate
        down = hidden * intermediate * elem_size
        raw = gate + up + down
        return cls(
            hidden=hidden,
            intermediate=intermediate,
            elem_size=elem_size,
            gate_nbytes=gate,
            up_nbytes=up,
            down_nbytes=down,
            raw_nbytes=raw,
            block_nbytes=align_up(raw, align),
        )


@dataclass(frozen=True)
class StoreHeader:
    version: int
    n_layers: int
    n_experts: int
    hidden: int
    intermediate: int
    elem_size: int
    dtype_code: int
    align: int
    flags: int
    raw_nbytes: int
    block_nbytes: int
    index_nbytes: int
    data_offset: int
    n_blocks: int
    resident_nbytes: int

    @property
    def complete(self) -> bool:
        return bool(self.flags & FLAG_COMPLETE)

    @property
    def layout(self) -> ExpertLayout:
        layout = ExpertLayout.from_dims(self.hidden, self.intermediate, self.elem_size, self.align)
        if layout.raw_nbytes != self.raw_nbytes or layout.block_nbytes != self.block_nbytes:
            raise ValueError("cabeçalho não fecha com o layout do expert")
        return layout


@dataclass(frozen=True)
class IndexEntry:
    layer: int
    expert: int
    offset: int
    raw_nbytes: int
    gate_nbytes: int
    up_nbytes: int
    down_nbytes: int

    def key(self) -> tuple[int, int]:
        return (self.layer, self.expert)


def align_up(n: int, align: int = ALIGN) -> int:
    if n < 0:
        raise ValueError("tamanho negativo")
    if align < 1:
        raise ValueError("alinhamento precisa ser >= 1")
    return (n + align - 1) // align * align


def data_offset_for(n_blocks: int, align: int = ALIGN) -> int:
    index_nbytes = n_blocks * ENTRY_SIZE
    return align_up(HEADER_SIZE + index_nbytes, align)


def pack_header(header: StoreHeader) -> bytes:
    if header.version != VERSION:
        raise ValueError(f"versão {header.version} não é a {VERSION}")
    raw = _HEADER.pack(
        MAGIC,
        header.version,
        header.n_layers,
        header.n_experts,
        header.hidden,
        header.intermediate,
        header.elem_size,
        header.dtype_code,
        header.align,
        header.flags,
        0,
        header.raw_nbytes,
        header.block_nbytes,
        header.index_nbytes,
        header.data_offset,
        header.n_blocks,
        header.resident_nbytes,
    )
    if len(raw) > HEADER_SIZE:
        raise RuntimeError("cabeçalho passou de 256 bytes")
    return raw + bytes(HEADER_SIZE - len(raw))


def unpack_header(blob: bytes) -> StoreHeader:
    if len(blob) < HEADER_SIZE:
        raise ValueError("cabeçalho truncado")
    fields = _HEADER.unpack_from(blob)
    magic = fields[0]
    if magic != MAGIC:
        raise ValueError("arquivo não é um store de experts (magic)")
    header = StoreHeader(
        version=fields[1],
        n_layers=fields[2],
        n_experts=fields[3],
        hidden=fields[4],
        intermediate=fields[5],
        elem_size=fields[6],
        dtype_code=fields[7],
        align=fields[8],
        flags=fields[9],
        raw_nbytes=fields[11],
        block_nbytes=fields[12],
        index_nbytes=fields[13],
        data_offset=fields[14],
        n_blocks=fields[15],
        resident_nbytes=fields[16],
    )
    if header.version != VERSION:
        raise ValueError(f"versão {header.version} do store não é suportada")
    if header.n_blocks != header.n_layers * header.n_experts:
        raise ValueError("n_blocks não é camadas × experts")
    if header.index_nbytes != header.n_blocks * ENTRY_SIZE:
        raise ValueError("tamanho do índice não fecha")
    if header.data_offset % header.align != 0 or header.block_nbytes % header.align != 0:
        raise ValueError("dados ou bloco fora do alinhamento")
    if header.dtype_code not in ELEM_SIZE or ELEM_SIZE[header.dtype_code] != header.elem_size:
        raise ValueError("dtype do store não fecha com elem_size")
    return header


def pack_entry(entry: IndexEntry) -> bytes:
    return _ENTRY.pack(
        entry.layer,
        entry.expert,
        entry.offset,
        entry.raw_nbytes,
        entry.gate_nbytes,
        entry.up_nbytes,
        entry.down_nbytes,
    )


def unpack_entry(blob: bytes) -> IndexEntry:
    if len(blob) < ENTRY_SIZE:
        raise ValueError("entrada de índice truncada")
    layer, expert, offset, raw, gate, up, down = _ENTRY.unpack_from(blob)
    return IndexEntry(
        layer=layer,
        expert=expert,
        offset=offset,
        raw_nbytes=raw,
        gate_nbytes=gate,
        up_nbytes=up,
        down_nbytes=down,
    )


def dtype_code_from_safetensors(name: str) -> int:
    if name not in NAME_TO_DTYPE:
        raise ValueError(f"dtype {name} não entra no store (use BF16, F16 ou F32)")
    return NAME_TO_DTYPE[name]
