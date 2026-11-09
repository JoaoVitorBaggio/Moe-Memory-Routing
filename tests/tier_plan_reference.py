"""Cópia congelada da máscara anterior a otimização. Oráculo dos testes.

Não é o caminho de produção. A regra é a de ``tier_mask`` antes da
seleção vetorial: argsort estável, vaga vazia antes da troca, ``b2``
global, pior empate no maior índice.
"""

from __future__ import annotations

import torch

from moe_memory.tier_mask import TierPlan, TierState, _soft_weights


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
    batch, seq, layers, experts = warm_scores.shape
    device = warm_scores.device
    warm_fixed = warm_scores.detach()
    hot_fixed = hot_scores.detach()
    if state is None:
        current_warm = torch.zeros(batch, layers, experts, dtype=torch.bool, device=device)
        current_hot = torch.zeros(batch, layers, experts, dtype=torch.bool, device=device)
        origin = 0
    else:
        current_warm = state.warm.to(device=device)
        current_hot = state.hot.to(device=device)
        origin = state.step
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
    return plan, TierState(warm=current_warm, hot=current_hot, step=origin + seq)


def _admit_batch(warm: torch.Tensor, scores: torch.Tensor, k2: int, b2: int) -> torch.Tensor:
    return torch.stack(
        [_admit_one(warm[item], scores[item], k2, b2) for item in range(warm.shape[0])],
        dim=0,
    )


def _admit_one(warm: torch.Tensor, scores: torch.Tensor, k2: int, b2: int) -> torch.Tensor:
    current = warm.clone()
    for _admission in range(b2):
        best_key: tuple[int, float, int, int] | None = None
        best: tuple[int, int, int | None] | None = None
        for layer in range(current.shape[0]):
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


def _hold_hot_batch(previous, warm, scores, k1):
    return torch.stack(
        [_hold_hot_one(previous[i], warm[i], scores[i], k1) for i in range(warm.shape[0])],
        dim=0,
    )


def _hold_hot_one(previous, warm, scores, k1):
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


def _index_batch(hot, scores, k1):
    indices = []
    valids = []
    for item in range(hot.shape[0]):
        index, valid = _index_one(hot[item], scores[item], k1)
        indices.append(index)
        valids.append(valid)
    return torch.stack(indices, dim=0), torch.stack(valids, dim=0)


def _index_one(hot, scores, k1):
    index = torch.zeros(hot.shape[0], k1, dtype=torch.long, device=hot.device)
    valid = torch.zeros(hot.shape[0], k1, dtype=torch.bool, device=hot.device)
    for layer in range(hot.shape[0]):
        chosen, ok = _topk_indices(scores[layer], hot[layer], k1)
        index[layer] = chosen
        valid[layer] = ok
    return index, valid


def _topk_indices(scores, mask, k):
    visible = torch.where(mask, scores, torch.full_like(scores, float("-inf")))
    order = torch.argsort(-visible, stable=True)
    chosen = order[:k]
    valid = mask[chosen]
    chosen = torch.where(valid, chosen, torch.zeros_like(chosen))
    return chosen, valid


def _best_index(scores, mask) -> int:
    chosen, valid = _topk_indices(scores, mask, k=1)
    if not bool(valid[0].item()):
        raise RuntimeError("não há candidato fora do conjunto")
    return int(chosen[0].item())


def _worst_index(scores, mask) -> int:
    visible = torch.where(mask, scores, torch.full_like(scores, float("inf")))
    order = torch.argsort(visible, stable=True)
    worst = float(visible[order[0]].item())
    tied = order[visible[order] == worst]
    return int(tied.max().item())


def _outranks(scores, cand: int, victim: int) -> bool:
    cand_score = float(scores[cand].item())
    victim_score = float(scores[victim].item())
    if cand_score > victim_score:
        return True
    if cand_score < victim_score:
        return False
    return cand < victim
