"""ESQUELETO do roteador LSTM adiantado.

Não é o roteador do experimento dos quatro braços. Não carrega OLMoE,
não treina e não destila roteador. O desenho está em
``docs/metodo-router-lstm-adiantado.md``.

O LSTM lê só ids de token. O estado oculto do Transformer não entra na
recorrência. No passo ``t`` as logits são as dos experts da posição
``t + lookahead``, uma fatia por camada. O top-K só escolhe o que
carregar. O peso mole é a probabilidade do LSTM naquele expert, sem
renormalizar dentro do top-K.

``HEAD_INIT_STD`` deixa a softmax perto da uniforme e, por ser maior
que zero, o primeiro backward já alcança o LSTM. Não é cota de hardware.
O tamanho do estado entra pelo construtor: quem o limita é o ``t_tok``
medido na máquina, e este módulo não escolhe esse número.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

# Desvio da projeção final. Pequeno para a softmax nascer perto da
# uniforme, e não zero para o gradiente chegar ao LSTM no primeiro passo.
# Constante do esqueleto, não medição de banda nem de t_tok.
HEAD_INIT_STD = 1e-3


@dataclass
class AheadDecision:
    """Decisão adiantada. ``logits[:, t]`` vale para a posição ``t + lookahead``."""

    logits: torch.Tensor
    probs: torch.Tensor
    topk_index: torch.Tensor
    soft_weights: torch.Tensor
    lookahead: int


class LstmAheadRouter(nn.Module):
    """ESQUELETO. Embedding de token, um LSTM, logits por camada em ``t+L``."""

    SKELETON = True

    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        n_experts: int,
        k: int,
        lookahead: int,
        embed_dim: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        if vocab_size < 1:
            raise ValueError("vocab_size precisa ser >= 1")
        if n_layers < 1:
            raise ValueError("n_layers precisa ser >= 1")
        if n_experts < 1:
            raise ValueError("n_experts precisa ser >= 1")
        if not 1 <= k <= n_experts:
            raise ValueError(f"k={k} precisa satisfazer 1 <= k <= n_experts={n_experts}")
        if lookahead < 1:
            raise ValueError("lookahead precisa ser >= 1")
        if embed_dim < 1 or hidden_size < 1:
            raise ValueError("embed_dim e hidden_size precisam ser >= 1")
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.k = k
        self.lookahead = lookahead
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, n_layers * n_experts)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.embed.weight, std=0.02)
        self.lstm.reset_parameters()
        nn.init.normal_(self.head.weight, std=HEAD_INIT_STD)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        token_ids: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[AheadDecision, tuple[torch.Tensor, torch.Tensor]]:
        """``token_ids`` é ``[B, T]`` long. Não há hidden de Transformer.

        O estado devolvido é o ``(h, c)`` do LSTM, para o passo seguinte
        de token. ``decision.logits[:, t]`` é a escolha da posição
        ``t + lookahead``.
        """

        if token_ids.ndim != 2:
            raise ValueError(f"token_ids precisa ser [B, T], recebido {tuple(token_ids.shape)}")
        if token_ids.dtype != torch.long:
            raise ValueError("token_ids precisa ser long")
        batch = token_ids.shape[0]
        state = _checked_state(state, batch=batch, hidden_size=self.hidden_size, device=token_ids.device)
        embedded = self.embed(token_ids)
        recurrent, new_state = self.lstm(embedded, state)
        logits = self.head(recurrent).view(batch, token_ids.shape[1], self.n_layers, self.n_experts)
        probs = logits.softmax(dim=-1)
        topk_index = torch.topk(logits, k=self.k, dim=-1).indices
        soft_weights = probs.gather(dim=-1, index=topk_index)
        decision = AheadDecision(
            logits=logits,
            probs=probs,
            topk_index=topk_index,
            soft_weights=soft_weights,
            lookahead=self.lookahead,
        )
        return decision, new_state

    def mix_layer(
        self,
        decision: AheadDecision,
        layer: int,
        hidden: torch.Tensor,
        experts: Sequence[nn.Module],
    ) -> torch.Tensor:
        """Aplica o peso mole da ``layer`` ao hidden do Transformer.

        ``hidden`` aqui é a entrada dos experts naquela camada. Não volta
        para o LSTM. Só os experts do top-K são chamados.
        """

        if layer < 0 or layer >= self.n_layers:
            raise ValueError(f"layer={layer} fora de 0..{self.n_layers - 1}")
        return combine_selected(
            hidden,
            experts,
            decision.topk_index[:, :, layer, :],
            decision.soft_weights[:, :, layer, :],
        )


def combine_selected(
    hidden: torch.Tensor,
    experts: Sequence[nn.Module],
    topk_index: torch.Tensor,
    soft_weights: torch.Tensor,
) -> torch.Tensor:
    """Soma ``probabilidade_do_LSTM * expert(h)`` só no top-K.

    ``hidden`` é ``[B, T, D]``. ``topk_index`` e ``soft_weights`` são
    ``[B, T, K]``. O peso não é renormalizado: a massa dos experts que
    não foram carregados fica de fora.
    """

    if hidden.ndim != 3:
        raise ValueError(f"hidden precisa ser [B, T, D], recebido {tuple(hidden.shape)}")
    if topk_index.shape != soft_weights.shape or topk_index.ndim != 3:
        raise ValueError("topk_index e soft_weights precisam ser [B, T, K] e ter o mesmo shape")
    batch, seq, dim = hidden.shape
    if topk_index.shape[0] != batch or topk_index.shape[1] != seq:
        raise ValueError(
            f"índice {tuple(topk_index.shape)} não casa com hidden {(batch, seq, dim)}"
        )
    if len(experts) < 1:
        raise ValueError("experts precisa ter ao menos um módulo")
    k = topk_index.shape[-1]
    flat = hidden.reshape(batch * seq, dim)
    flat_index = topk_index.reshape(batch * seq, k)
    flat_weights = soft_weights.reshape(batch * seq, k)
    mixed = flat.new_zeros(flat.shape)
    for expert_id, expert in enumerate(experts):
        selected = flat_index == expert_id
        if not bool(selected.any()):
            continue
        token_sel = selected.any(dim=1)
        weight = (flat_weights * selected.to(dtype=flat_weights.dtype)).sum(dim=1)
        token_index = token_sel.nonzero(as_tuple=False).flatten()
        expert_out = expert(flat.index_select(0, token_index))
        weighted = expert_out * weight.index_select(0, token_index).unsqueeze(-1)
        mixed = torch.index_add(mixed, 0, token_index, weighted)
    return mixed.view(batch, seq, dim)


def _checked_state(
    state: tuple[torch.Tensor, torch.Tensor] | None,
    *,
    batch: int,
    hidden_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if state is None:
        return None
    if len(state) != 2:
        raise ValueError("state precisa ser (h, c)")
    hidden, cell = state
    expected = (1, batch, hidden_size)
    if tuple(hidden.shape) != expected or tuple(cell.shape) != expected:
        raise ValueError(f"state precisa ter shape {expected} em h e em c")
    if hidden.device != device or cell.device != device:
        hidden = hidden.to(device=device)
        cell = cell.to(device=device)
    return hidden, cell
