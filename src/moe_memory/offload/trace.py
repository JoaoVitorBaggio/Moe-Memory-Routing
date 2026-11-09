"""Identidade da execução, não só a contagem.

O hash cobre tetos, pesos do LSTM, tamanho e prefixo do store, tokens,
conjunto inicial e, por chamada realmente feita no forward, camada,
expert, peso mole e 8 bytes do buffer de host. Não entra modo, byte de
SSD, stall nem tempo: lookahead e on-demand podem copiar quantidades
diferentes e ainda assim executar o mesmo expert.

Isto não é um replay oráculo. Quem compara dois modos roda os dois
runtimes com os mesmos tetos. Prefetch adiantado continua limitado a
L1/L2, e b2/k1/k2 não são afrouxados. Traço truncado não certifica.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import torch
from torch import nn


class TraceDivergence(ValueError):
    """Os dois lados não executaram a mesma coisa. O detalhe diz onde."""

    def __init__(self, message: str, detail: dict) -> None:
        super().__init__(message)
        self.detail = detail


def f32_hex(value: float) -> str:
    return struct.pack("<f", float(value)).hex()


def host_prefix(tensor: torch.Tensor, n: int = 8) -> str:
    """Oito bytes do buffer de host. Não lê a cópia da GPU."""

    flat = tensor.detach().contiguous().view(torch.uint8)
    take = min(n, int(flat.numel()))
    return bytes(flat[:take].cpu().tolist()).hex()


def module_sha(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        array = tensor.detach().contiguous().cpu().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


# O fingerprint do store é tamanho + este prefixo. Não é integridade do arquivo.
STORE_PREFIX_BYTES = 4096


def store_sha(path: Path) -> str:
    """Tamanho mais no máximo os primeiros 4 KiB.

    ``Path.read_bytes`` lê até o EOF e só então fatia: num store de ~55 GiB
    isso pede a RAM do arquivo inteiro. Aqui o ``read`` já nasce com teto.
    Arquivo menor que 4 KiB entra inteiro. Byte além do prefixo não muda o hash.
    """

    size = path.stat().st_size
    with path.open("rb") as handle:
        blob = handle.read(STORE_PREFIX_BYTES)
    if len(blob) > STORE_PREFIX_BYTES:
        raise RuntimeError(f"prefixo do store passou de {STORE_PREFIX_BYTES} bytes")
    digest = hashlib.sha256()
    digest.update(str(size).encode())
    digest.update(blob)
    return digest.hexdigest()


class ExecutionTrace:
    def __init__(
        self,
        *,
        limits: dict,
        tokens: list[int],
        router_sha: str,
        store_sha256: str,
        max_tokens: int = 512,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("teto do traço precisa ser >= 1")
        self.limits = dict(limits)
        self.tokens = list(tokens)
        self.router_sha = router_sha
        self.store_sha256 = store_sha256
        self.max_tokens = max_tokens
        self.initial: list[list[int]] = []
        self.steps: list[list[list]] = []
        self.truncated = False

    def set_initial(self, keys: list[tuple[int, int]]) -> None:
        self.initial = [[int(layer), int(expert)] for layer, expert in keys]

    def add_step(self, calls: list[tuple[int, int, str, str]]) -> None:
        if len(self.steps) >= self.max_tokens:
            self.truncated = True
            return
        self.steps.append(
            [[int(layer), int(expert), weight, prefix] for layer, expert, weight, prefix in calls]
        )

    def fingerprint(self) -> str:
        if self.truncated:
            raise TraceDivergence(
                "traço truncado não certifica a identidade da execução",
                {"truncated": True, "kept_tokens": len(self.steps), "max_tokens": self.max_tokens},
            )
        digest = hashlib.sha256()
        payload = {
            "limits": self.limits,
            "tokens": self.tokens,
            "router_sha": self.router_sha,
            "store_sha256": self.store_sha256,
            "initial": self.initial,
            "steps": self.steps,
        }
        digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
        return digest.hexdigest()

    def public(self) -> dict:
        fingerprint = None
        error = None
        try:
            fingerprint = self.fingerprint()
        except TraceDivergence as exc:
            error = str(exc)
        return {
            "fingerprint": fingerprint,
            "fingerprint_error": error,
            "truncated": self.truncated,
            "limits": self.limits,
            "tokens": self.tokens,
            "router_sha": self.router_sha,
            "store_sha256": self.store_sha256,
            "initial": self.initial,
            "steps": self.steps,
            "excludes": ["mode", "ssd_bytes", "stall", "elapsed_s", "prefetch"],
            "note": (
                "steps são as chamadas que o forward fez, com 8 bytes do "
                "buffer de host. Não é hash da cópia na GPU. O prefixo do "
                "arquivo não é o corpo inteiro do store."
            ),
        }


def compare_traces(left: ExecutionTrace, right: ExecutionTrace) -> str:
    """Devolve o fingerprint comum ou levanta ``TraceDivergence``."""

    if left.truncated or right.truncated:
        raise TraceDivergence(
            "traço truncado não certifica a identidade da execução",
            {
                "left_truncated": left.truncated,
                "right_truncated": right.truncated,
                "left_kept": len(left.steps),
                "right_kept": len(right.steps),
            },
        )
    if left.limits != right.limits:
        raise TraceDivergence(
            "tetos diferentes: a comparação não afrouxa b2, k1, k2 nem vagas",
            {"left": left.limits, "right": right.limits},
        )
    if left.router_sha != right.router_sha:
        raise TraceDivergence(
            "pesos do LSTM diferentes",
            {"left": left.router_sha, "right": right.router_sha},
        )
    if left.store_sha256 != right.store_sha256:
        raise TraceDivergence(
            "store diferente",
            {"left": left.store_sha256, "right": right.store_sha256},
        )
    if left.tokens != right.tokens:
        raise TraceDivergence(
            "tokens de entrada diferentes",
            {"left": left.tokens, "right": right.tokens},
        )
    if left.initial != right.initial:
        raise TraceDivergence(
            "conjunto inicial diferente",
            {"left": left.initial, "right": right.initial},
        )
    if len(left.steps) != len(right.steps):
        raise TraceDivergence(
            "número de tokens executados diferente",
            {"left": len(left.steps), "right": len(right.steps)},
        )
    for index, (a_step, b_step) in enumerate(zip(left.steps, right.steps)):
        if a_step != b_step:
            raise TraceDivergence(
                f"execução divergiu no token {index}",
                {"token": index, "left": a_step, "right": b_step},
            )
    return left.fingerprint()
