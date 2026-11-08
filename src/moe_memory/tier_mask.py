"""Máscara determinística dos dois níveis. A mesma no treino e na inferência.

O desenho está em ``docs/metodo-router-lstm-adiantado.md`` (extensão para
SSD) e os números de viabilidade em ``docs/medicoes-hw-fedora-qwen3.md``.
Este módulo não copia peso, não abre arquivo e não soma penalidade de
troca. A loss de quem chama continua sendo só a de linguagem.

Por sequência, o morno começa vazio e cabe ``k2`` experts em cada camada.
Em cada token entram no máximo ``b2`` experts novos, somando as camadas.
Quem entra no conjunto já cheio ocupa o lugar do de menor score, e só
se o score do novo for melhor. Vaga vazia tem prioridade sobre troca.
O quente é o top-``k1`` dentro do morno, pelo score do horizonte curto.

``refresh_n1`` e ``refresh_n2``, quando não são ``None``, seguram o
conjunto: ele só é reavaliado a cada N tokens, contados desde o começo
da sequência (o estado carrega o passo). No passo parado a contagem de
novos é zero. O teto ``b2`` vale no passo em que o morno muda.

O prefetch lista ids. O próximo passo do projeto é quem move o byte.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from moe_memory.levels import HOT, WARM


@dataclass
class TierState:
    """Conjunto já decidido, para continuar a sequência no próximo chunk.

    ``warm`` e ``hot`` são ``[B, camadas, E]`` bool. ``step`` é quantos
    tokens já passaram por esta regra.
    """

    warm: torch.Tensor
    hot: torch.Tensor
    step: int


@dataclass(frozen=True)
class TierPrefetch:
    """Ids ``(camada, expert)`` deste passo. Não há I/O.

    ``warm`` sobe para a RAM e ``hot`` para a VRAM. ``warm_release`` e
    ``hot_release`` são os que saem do nível. A ordem é camada, depois
    o índice do expert.
    """

    warm: tuple[tuple[int, int], ...]
    hot: tuple[tuple[int, int], ...]
    warm_release: tuple[tuple[int, int], ...]
    hot_release: tuple[tuple[int, int], ...]


@dataclass
class TierPlan:
    """Máscaras alinhadas à posição do token. ``soft_weights`` fica no grafo.

    O índice discreto sai de um ``detach``: a loss não atravessa a escolha
    de quem entra. O peso mole é a softmax dos scores quentes nos
    escolhidos, e essa softmax continua diferenciável.
    """

    warm_mask: torch.Tensor
    hot_mask: torch.Tensor
    entered_warm: torch.Tensor
    entered_hot: torch.Tensor
    left_warm: torch.Tensor
    left_hot: torch.Tensor
    hot_index: torch.Tensor
    hot_valid: torch.Tensor
    soft_weights: torch.Tensor
    n_new_warm: torch.Tensor
    n_new_hot: torch.Tensor
    k1: int
    k2: int
    b2: int
    l1: int
    l2: int
    refresh_n1: int | None
    refresh_n2: int | None

    def prefetch(self, batch: int, step: int) -> TierPrefetch:
        self._check_pos(batch, step)
        return TierPrefetch(
            warm=_ids(self.entered_warm[batch, step]),
            hot=_ids(self.entered_hot[batch, step]),
            warm_release=_ids(self.left_warm[batch, step]),
            hot_release=_ids(self.left_hot[batch, step]),
        )

    def resident_ids(self, batch: int, step: int, level: int) -> tuple[tuple[int, int], ...]:
        """Quem deve estar no nível neste passo. ``level`` é ``WARM`` ou ``HOT``."""

        self._check_pos(batch, step)
        if level == WARM:
            mask = self.warm_mask[batch, step]
        elif level == HOT:
            mask = self.hot_mask[batch, step]
        else:
            raise ValueError("residência consultável aqui é morno ou quente")
        return _ids(mask)

    def _check_pos(self, batch: int, step: int) -> None:
        batches, steps = self.warm_mask.shape[:2]
        if not 0 <= batch < batches:
            raise IndexError(f"batch={batch} fora de 0..{batches - 1}")
        if not 0 <= step < steps:
            raise IndexError(f"step={step} fora de 0..{steps - 1}")


def swiglu_expert_nbytes(hidden_size: int, intermediate_size: int, bytes_per_param: int = 2) -> int:
    """Bytes de um SwiGLU de três projeções, sem o gate da camada.

    A conta do Qwen3-30B-A3B na nota de 07/11/2026 é
    ``3 * 2048 * 768 * 2``. Isto só faz a aritmética; não mede o disco.
    """

    if hidden_size < 1 or intermediate_size < 1 or bytes_per_param < 1:
        raise ValueError("hidden, intermediário e bytes por parâmetro precisam ser >= 1")
    return 3 * hidden_size * intermediate_size * bytes_per_param


def k2_from_ram_budget(ram_bytes: int, expert_bytes: int, n_layers: int) -> int:
    """Experts mornos por camada que cabem inteiros em ``ram_bytes``.

    O resto da divisão (o que caberia no total e não fecha uma vaga em
    toda camada) não é redistribuído. Quem chama passa o inteiro ao
    roteador como ``k2``.
    """

    if ram_bytes < 0 or expert_bytes < 1 or n_layers < 1:
        raise ValueError("orçamento, tamanho do expert e camadas inválidos")
    return (ram_bytes // expert_bytes) // n_layers


def b2_from_token_time(t_tok_seconds: float, seconds_per_new_expert: float) -> int:
    """Quantos experts novos cabem inteiros num ``t_tok``.

    É o piso da divisão. A nota adota o lado lento da faixa medida;
    este helper não escolhe o lado, só faz o piso.
    """

    if t_tok_seconds <= 0 or seconds_per_new_expert <= 0:
        raise ValueError("t_tok e o tempo por expert precisam ser > 0")
    return int(t_tok_seconds // seconds_per_new_expert)


def plan_from_scores(
    warm_scores: torch.Tensor,
    hot_scores: torch.Tensor,
    *,
    k1: int,
    k2: int,
    b2: int,
    refresh_n1: int | None = None,
    refresh_n2: int | None = None,
    l1: int = 1,
    l2: int = 1,
    state: TierState | None = None,
) -> tuple[TierPlan, TierState]:
    """Aplica a regra dos dois níveis aos scores já alinhados à posição.

    ``warm_scores`` e ``hot_scores`` são ``[B, T, camadas, E]``. O índice
    discreto usa uma cópia sem gradiente. ``soft_weights`` lê ``hot_scores``
    vivo, então o backward chega na cabeça que produziu o score.
    """

    _validate_scores(warm_scores, hot_scores, k1, k2, b2, refresh_n1, refresh_n2)
    batch, seq, layers, experts = warm_scores.shape
    device = warm_scores.device
    warm_fixed = warm_scores.detach()
    hot_fixed = hot_scores.detach()
    if state is None:
        current_warm = torch.zeros(batch, layers, experts, dtype=torch.bool, device=device)
        current_hot = torch.zeros(batch, layers, experts, dtype=torch.bool, device=device)
        origin = 0
    else:
        current_warm, current_hot, origin = _checked_state(state, batch, layers, experts, device)

    warm_mask = torch.empty(batch, seq, layers, experts, dtype=torch.bool, device=device)
    hot_mask = torch.empty(batch, seq, layers, experts, dtype=torch.bool, device=device)
    entered_warm = torch.empty(batch, seq, layers, experts, dtype=torch.bool, device=device)
    entered_hot = torch.empty(batch, seq, layers, experts, dtype=torch.bool, device=device)
    left_warm = torch.empty(batch, seq, layers, experts, dtype=torch.bool, device=device)
    left_hot = torch.empty(batch, seq, layers, experts, dtype=torch.bool, device=device)
    hot_index = torch.zeros(batch, seq, layers, k1, dtype=torch.long, device=device)
    hot_valid = torch.zeros(batch, seq, layers, k1, dtype=torch.bool, device=device)

    for step in range(seq):
        global_step = origin + step
        before_warm = current_warm
        before_hot = current_hot
        if refresh_n2 is None or global_step % refresh_n2 == 0:
            current_warm = _admit_batch(before_warm, warm_fixed[:, step], k2, b2)
        if refresh_n1 is None or global_step % refresh_n1 == 0:
            current_hot = _fresh_hot_batch(current_warm, hot_fixed[:, step], k1)
        else:
            current_hot = _hold_hot_batch(before_hot, current_warm, hot_fixed[:, step], k1)
        warm_mask[:, step] = current_warm
        hot_mask[:, step] = current_hot
        entered_warm[:, step] = current_warm & ~before_warm
        left_warm[:, step] = before_warm & ~current_warm
        entered_hot[:, step] = current_hot & ~before_hot
        left_hot[:, step] = before_hot & ~current_hot
        index, valid = _index_batch(current_hot, hot_fixed[:, step], k1)
        hot_index[:, step] = index
        hot_valid[:, step] = valid

    plan = TierPlan(
        warm_mask=warm_mask,
        hot_mask=hot_mask,
        entered_warm=entered_warm,
        entered_hot=entered_hot,
        left_warm=left_warm,
        left_hot=left_hot,
        hot_index=hot_index,
        hot_valid=hot_valid,
        soft_weights=_soft_weights(hot_scores, hot_index, hot_valid),
        n_new_warm=entered_warm.sum(dim=-1).to(dtype=torch.long),
        n_new_hot=entered_hot.sum(dim=-1).to(dtype=torch.long),
        k1=k1,
        k2=k2,
        b2=b2,
        l1=l1,
        l2=l2,
        refresh_n1=refresh_n1,
        refresh_n2=refresh_n2,
    )
    new_state = TierState(warm=current_warm, hot=current_hot, step=origin + seq)
    return plan, new_state


def _validate_scores(
    warm_scores: torch.Tensor,
    hot_scores: torch.Tensor,
    k1: int,
    k2: int,
    b2: int,
    refresh_n1: int | None,
    refresh_n2: int | None,
) -> None:
    if warm_scores.ndim != 4 or tuple(hot_scores.shape) != tuple(warm_scores.shape):
        raise ValueError("scores mornos e quentes precisam ser [B, T, camadas, E] e ter o mesmo shape")
    if not warm_scores.is_floating_point():
        raise ValueError("scores precisam ser ponto flutuante")
    batch, seq, layers, experts = warm_scores.shape
    if min(batch, seq, layers, experts) < 1:
        raise ValueError("lote, tempo, camadas e experts precisam ser >= 1")
    if not 1 <= k1 <= k2 <= experts:
        raise ValueError(f"precisa 1 <= k1={k1} <= k2={k2} <= E={experts}")
    if b2 < 0:
        raise ValueError("b2 precisa ser >= 0")
    for name, value in (("refresh_n1", refresh_n1), ("refresh_n2", refresh_n2)):
        if value is not None and value < 1:
            raise ValueError(f"{name} precisa ser >= 1 ou None")


def _checked_state(
    state: TierState,
    batch: int,
    layers: int,
    experts: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    expected = (batch, layers, experts)
    if state.step < 0:
        raise ValueError("step do tier não pode ser negativo")
    if tuple(state.warm.shape) != expected or tuple(state.hot.shape) != expected:
        raise ValueError(f"estado do tier precisa ter shape {expected}")
    if state.warm.dtype != torch.bool or state.hot.dtype != torch.bool:
        raise ValueError("máscaras do estado precisam ser bool")
    return state.warm.to(device=device), state.hot.to(device=device), state.step


def _admit_batch(warm: torch.Tensor, scores: torch.Tensor, k2: int, b2: int) -> torch.Tensor:
    return torch.stack(
        [_admit_one(warm[item], scores[item], k2, b2) for item in range(warm.shape[0])],
        dim=0,
    )


def _admit_one(warm: torch.Tensor, scores: torch.Tensor, k2: int, b2: int) -> torch.Tensor:
    """No máximo ``b2`` experts novos, somando as camadas desta sequência."""

    current = warm.clone()
    n_layers = current.shape[0]
    for _admission in range(b2):
        best_key: tuple[int, float, int, int] | None = None
        best: tuple[int, int, int | None] | None = None
        for layer in range(n_layers):
            member = current[layer]
            outsider = ~member
            if int(outsider.sum().item()) == 0:
                continue
            cand = _best_index(scores[layer], outsider)
            cand_score = float(scores[layer, cand].item())
            n_members = int(member.sum().item())
            if n_members < k2:
                key = (0, -cand_score, layer, cand)
                proposal = (layer, cand, None)
            else:
                victim = _worst_index(scores[layer], member)
                if not _outranks(scores[layer], cand, victim):
                    continue
                margin = cand_score - float(scores[layer, victim].item())
                key = (1, -margin, layer, cand)
                proposal = (layer, cand, victim)
            if best_key is None or key < best_key:
                best_key = key
                best = proposal
        if best is None:
            break
        layer, cand, victim = best
        current[layer, cand] = True
        if victim is not None:
            current[layer, victim] = False
    return current


def _fresh_hot_batch(warm: torch.Tensor, scores: torch.Tensor, k1: int) -> torch.Tensor:
    return torch.stack(
        [_fresh_hot_one(warm[item], scores[item], k1) for item in range(warm.shape[0])],
        dim=0,
    )


def _fresh_hot_one(warm: torch.Tensor, scores: torch.Tensor, k1: int) -> torch.Tensor:
    hot = torch.zeros_like(warm)
    for layer in range(warm.shape[0]):
        chosen, valid = _topk_indices(scores[layer], warm[layer], k1)
        for slot in range(k1):
            if bool(valid[slot].item()):
                hot[layer, chosen[slot]] = True
    return hot


def _hold_hot_batch(
    previous: torch.Tensor,
    warm: torch.Tensor,
    scores: torch.Tensor,
    k1: int,
) -> torch.Tensor:
    return torch.stack(
        [
            _hold_hot_one(previous[item], warm[item], scores[item], k1)
            for item in range(warm.shape[0])
        ],
        dim=0,
    )


def _hold_hot_one(
    previous: torch.Tensor,
    warm: torch.Tensor,
    scores: torch.Tensor,
    k1: int,
) -> torch.Tensor:
    """Segura o quente. Quem saiu do morno sai daqui, e a vaga é preenchida."""

    hot = previous & warm
    for layer in range(warm.shape[0]):
        target = min(k1, int(warm[layer].sum().item()))
        have = int(hot[layer].sum().item())
        need = target - have
        if need <= 0:
            continue
        room = warm[layer] & ~hot[layer]
        chosen, valid = _topk_indices(scores[layer], room, need)
        for slot in range(need):
            if bool(valid[slot].item()):
                hot[layer, chosen[slot]] = True
    return hot


def _index_batch(hot: torch.Tensor, scores: torch.Tensor, k1: int) -> tuple[torch.Tensor, torch.Tensor]:
    indices = []
    valids = []
    for item in range(hot.shape[0]):
        index, valid = _index_one(hot[item], scores[item], k1)
        indices.append(index)
        valids.append(valid)
    return torch.stack(indices, dim=0), torch.stack(valids, dim=0)


def _index_one(hot: torch.Tensor, scores: torch.Tensor, k1: int) -> tuple[torch.Tensor, torch.Tensor]:
    index = torch.zeros(hot.shape[0], k1, dtype=torch.long, device=hot.device)
    valid = torch.zeros(hot.shape[0], k1, dtype=torch.bool, device=hot.device)
    for layer in range(hot.shape[0]):
        chosen, ok = _topk_indices(scores[layer], hot[layer], k1)
        index[layer] = chosen
        valid[layer] = ok
    return index, valid


def _topk_indices(scores: torch.Tensor, mask: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Maior score primeiro. Empate fica com o menor índice. Inválido vira 0."""

    visible = torch.where(mask, scores, torch.full_like(scores, float("-inf")))
    order = torch.argsort(-visible, stable=True)
    chosen = order[:k]
    valid = mask[chosen]
    chosen = torch.where(valid, chosen, torch.zeros_like(chosen))
    return chosen, valid


def _best_index(scores: torch.Tensor, mask: torch.Tensor) -> int:
    chosen, valid = _topk_indices(scores, mask, k=1)
    if not bool(valid[0].item()):
        raise RuntimeError("não há candidato fora do conjunto")
    return int(chosen[0].item())


def _worst_index(scores: torch.Tensor, mask: torch.Tensor) -> int:
    """Menor score. Empate sai o maior índice, para não oscilar com o melhor."""

    visible = torch.where(mask, scores, torch.full_like(scores, float("inf")))
    order = torch.argsort(visible, stable=True)
    worst = float(visible[order[0]].item())
    tied = order[visible[order] == worst]
    return int(tied.max().item())


def _outranks(scores: torch.Tensor, cand: int, victim: int) -> bool:
    cand_score = float(scores[cand].item())
    victim_score = float(scores[victim].item())
    if cand_score > victim_score:
        return True
    if cand_score < victim_score:
        return False
    return cand < victim


def _soft_weights(
    hot_scores: torch.Tensor,
    hot_index: torch.Tensor,
    hot_valid: torch.Tensor,
) -> torch.Tensor:
    gathered = torch.gather(hot_scores, -1, hot_index)
    floor = torch.finfo(hot_scores.dtype).min
    masked = gathered.masked_fill(~hot_valid, floor)
    weights = torch.softmax(masked.float(), dim=-1).to(dtype=hot_scores.dtype)
    return weights * hot_valid.to(dtype=weights.dtype)


def _ids(mask: torch.Tensor) -> tuple[tuple[int, int], ...]:
    layers, experts = torch.nonzero(mask, as_tuple=True)
    pairs = [(int(layer), int(expert)) for layer, expert in zip(layers.tolist(), experts.tolist())]
    pairs.sort()
    return tuple(pairs)
