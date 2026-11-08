"""Cabe ou não cabe no disco. A conta é explícita; o script não apaga sozinho.

Na máquina de 07/11/2026 o checkpoint em bf16 fica em torno de 61 GB e
sobram cerca de 44 GB. O store repete os corpos (~54 GiB) e ainda grava
o resto da rede. Manter a fonte e a saída ao mesmo tempo não cabe.
``--delete-source-after-each-shard`` apaga cada shard depois do fsync;
aí o pico extra fica na ordem do maior shard, não dos 61 GB.
"""

from __future__ import annotations

from dataclasses import dataclass

# Números da nota, não uma medição desta máquina. O script também imprime
# o tamanho real varrido e o espaço livre de verdade.
REFERENCE_CHECKPOINT_BYTES = 61_000_000_000
REFERENCE_FREE_BYTES = 44_000_000_000
REFERENCE_WARM_RAM_BYTES = 12 * 1024**3
_MARGIN_BYTES = 1 << 30


class SpaceError(RuntimeError):
    """A saída não cabe. ``report`` traz a conta que o CLI imprime."""

    def __init__(self, report: SpaceReport) -> None:
        super().__init__(report.message)
        self.report = report


@dataclass(frozen=True)
class SpaceReport:
    source_bytes: int
    store_bytes: int
    resident_bytes: int
    free_bytes: int
    largest_shard_bytes: int
    delete_source: bool
    same_filesystem: bool
    fits: bool
    message: str

    @property
    def output_bytes(self) -> int:
        return self.store_bytes + self.resident_bytes


def assess_space(
    *,
    source_bytes: int,
    store_bytes: int,
    resident_bytes: int,
    free_bytes: int,
    largest_shard_bytes: int,
    delete_source: bool,
    same_filesystem: bool,
) -> SpaceReport:
    """Decide se a escrita pode começar.

    Sem apagar a fonte, o destino precisa ter a saída inteira livre.
    Com a flag, e com fonte e destino no mesmo sistema de arquivos, basta
    o maior shard mais 1 GiB: cada shard é apagado depois que os bytes
    dele já estão no destino e o fsync voltou. Sistemas de arquivos
    diferentes não recuperam espaço no destino.
    """

    if min(source_bytes, store_bytes, resident_bytes, free_bytes, largest_shard_bytes) < 0:
        raise ValueError("tamanhos do relatório de disco não podem ser negativos")
    output = store_bytes + resident_bytes
    if delete_source and same_filesystem:
        need = largest_shard_bytes + _MARGIN_BYTES
        reason = (
            "com --delete-source-after-each-shard, no mesmo sistema de arquivos, "
            f"o pico extra é o maior shard ({_fmt(largest_shard_bytes)}) mais 1 GiB "
            f"de margem, {_fmt(need)} no total"
        )
    elif delete_source and not same_filesystem:
        need = output + _MARGIN_BYTES
        reason = (
            "a fonte e o destino estão em sistemas de arquivos diferentes; "
            "apagar o shard não libera espaço no destino, então a saída inteira "
            f"precisa caber: {_fmt(need)}"
        )
    else:
        need = output
        reason = (
            "sem --delete-source-after-each-shard a fonte fica no disco e a saída "
            f"precisa de {_fmt(need)} livres além dela"
        )
    fits = free_bytes >= need
    message = "\n".join(
        [
            "Espaço do repack de experts.",
            f"  fonte: {_fmt(source_bytes)}",
            f"  store de experts: {_fmt(store_bytes)}",
            f"  pesos residentes (atenção, normas, embeddings, lm_head, gate): {_fmt(resident_bytes)}",
            f"  saída somada: {_fmt(output)}",
            f"  livre agora: {_fmt(free_bytes)}",
            f"  maior shard: {_fmt(largest_shard_bytes)}",
            f"  decisão: {reason}",
            (
                "  cabe."
                if fits
                else "  não cabe. Nada foi apagado e o store não foi aberto."
            ),
            (
                "  Referência da máquina de 07/11/2026: o checkpoint tem cerca de "
                f"{_fmt(REFERENCE_CHECKPOINT_BYTES)} e sobram cerca de "
                f"{_fmt(REFERENCE_FREE_BYTES)}. Manter os dois ao mesmo tempo não cabe. "
                "A flag --delete-source-after-each-shard apaga cada shard depois do "
                "fsync; é destrutiva, e um crash no meio perde o shard já apagado "
                "(os bytes dele já foram gravados e sincronizados)."
            ),
        ]
    )
    return SpaceReport(
        source_bytes=source_bytes,
        store_bytes=store_bytes,
        resident_bytes=resident_bytes,
        free_bytes=free_bytes,
        largest_shard_bytes=largest_shard_bytes,
        delete_source=delete_source,
        same_filesystem=same_filesystem,
        fits=fits,
        message=message,
    )


def default_warm_slots(block_nbytes: int, ram_bytes: int = REFERENCE_WARM_RAM_BYTES) -> int:
    """Vagas inteiras que cabem no orçamento morno (~12 GiB na nota)."""

    if block_nbytes < 1 or ram_bytes < 0:
        raise ValueError("bloco e orçamento de RAM inválidos")
    slots = ram_bytes // block_nbytes
    if slots < 1:
        raise ValueError(
            f"um bloco de {block_nbytes} bytes não cabe no orçamento morno de {ram_bytes} bytes"
        )
    return slots


def _fmt(n: int) -> str:
    gib = n / (1024**3)
    gb = n / 1e9
    return f"{n} bytes ({gb:.2f} GB, {gib:.2f} GiB)"
