"""Recorta ``gate | up | down`` de um bloco uint8."""

from __future__ import annotations

import torch

from moe_memory.offload.format import ExpertLayout


def split_block(
    block: torch.Tensor,
    layout: ExpertLayout,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if block.dtype != torch.uint8:
        raise ValueError("o bloco da vaga é uint8")
    if block.numel() < layout.raw_nbytes:
        raise ValueError("bloco menor que o expert")
    elem = torch.empty((), dtype=dtype).element_size()
    if elem != layout.elem_size:
        raise ValueError("dtype não fecha com o elem_size do store")
    typed = block[: layout.raw_nbytes].view(dtype)
    gate_n = layout.gate_nbytes // elem
    up_n = layout.up_nbytes // elem
    down_n = layout.down_nbytes // elem
    gate = typed[:gate_n].view(layout.intermediate, layout.hidden)
    up = typed[gate_n : gate_n + up_n].view(layout.intermediate, layout.hidden)
    down = typed[gate_n + up_n : gate_n + up_n + down_n].view(layout.hidden, layout.intermediate)
    return gate, up, down
