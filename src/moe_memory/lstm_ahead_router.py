"""Roteadores LSTM que leem só ids de token.

``LstmAheadRouter`` é o ESQUELETO de um horizonte, o desenho do OLMoE em
``docs/metodo-router-lstm-adiantado.md``. Não carrega OLMoE, não treina
e não destila roteador. No passo ``t`` as logits são as dos experts da
posição ``t + lookahead``. O top-K só escolhe o que carregar. O peso
mole é a probabilidade do LSTM naquele expert, sem renormalizar dentro
do top-K.

``TwoHorizonLstmRouter`` é a extensão de dois níveis (SSD, 07/11/2026).
Um tronco, uma cabeça linear por camada, duas defasagens. A máscara
morno/quente está em ``tier_mask.py``. Cada sequência começa com o
conjunto inicial cheio (``k1`` por camada, fora do ``b2``): o score do
prompt, ou o do token ``BOS_ID`` se não houver prompt. Não move peso e
não entra na loss dos quatro braços.

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

# Token que pontua o conjunto inicial quando a sequência não tem prompt.
# Não entra no lote de treino nem no decode. O Qwen3 público não fixa um
# BOS único; este desenho usa o id 0.
BOS_ID = 0


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


@dataclass
class TwoHorizonDecision:
    """Scores alinhados à posição que vai consumir o expert.

    ``raw_scores[:, t]`` depende só dos tokens de índice ``<= t``.
    ``hot_scores[:, t]`` é o raw da posição ``t - l1`` (estado zero se
    ``t < l1``). ``warm_scores[:, t]`` é o raw de ``t - l2``. Quem escolhe
    o expert da posição ``u`` lê o score alinhado, não o raw.
    """

    raw_scores: torch.Tensor
    warm_scores: torch.Tensor
    hot_scores: torch.Tensor
    l1: int
    l2: int


@dataclass
class TwoHorizonState:
    """Estado para o próximo passo de token.

    ``h`` e ``c`` são o LSTM. ``raw_hist`` guarda os últimos scores crus,
    os que a defasagem ainda vai usar. ``tier`` é o conjunto morno/quente
    já aplicado; ``None`` se este estado saiu de ``forward`` e a máscara
    ainda não rodou. Inferência em fluxo chama ``plan``, que preenche
    ``tier``.
    """

    h: torch.Tensor
    c: torch.Tensor
    raw_hist: torch.Tensor
    tier: "TierState | None" = None


class TwoHorizonLstmRouter(nn.Module):
    """Um LSTM, uma cabeça por camada, horizontes ``l1`` (quente) e ``l2`` (morno).

    A cabeça da camada ``c`` lê o estado do tronco e emite ``E`` scores.
    ``L1`` e ``L2`` não acrescentam parâmetro: são deslocamentos do mesmo
    score. A decisão da posição ``t + L`` usa só tokens de índice ``<= t``.

    ``k1`` é o top do quente, dentro do morno. ``k2`` é o tamanho do morno
    por camada. ``b2`` é o teto global, por token e por sequência, de
    experts novos no morno, somando as camadas. ``refresh_n1`` e
    ``refresh_n2``, se não forem ``None``, seguram o quente e o morno
    por esse número de tokens. O orçamento ``b2`` continua valendo no
    passo em que o morno atualiza, e não cobre o conjunto inicial:
    ``plan`` semeia ``k1`` experts por camada antes do primeiro token.

    O peso mole da mistura é a softmax só sobre os experts quentes
    daquele token (a massa some 1). Isso é o ``norm_topk_prob`` do
    Qwen3, não a regra do esqueleto do OLMoE, que não renormaliza.
    """

    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        n_experts: int,
        l1: int,
        l2: int,
        k1: int,
        k2: int,
        b2: int,
        embed_dim: int,
        hidden_size: int,
        refresh_n1: int | None = None,
        refresh_n2: int | None = None,
    ) -> None:
        super().__init__()
        if vocab_size < 1:
            raise ValueError("vocab_size precisa ser >= 1")
        if n_layers < 1:
            raise ValueError("n_layers precisa ser >= 1")
        if n_experts < 1:
            raise ValueError("n_experts precisa ser >= 1")
        if l1 < 1:
            raise ValueError("l1 precisa ser >= 1")
        if l2 < l1:
            raise ValueError(f"l2={l2} é o horizonte longo e precisa ser >= l1={l1}")
        if not 1 <= k1 <= k2 <= n_experts:
            raise ValueError(f"precisa 1 <= k1={k1} <= k2={k2} <= n_experts={n_experts}")
        if b2 < 1:
            raise ValueError("b2 precisa ser >= 1")
        if embed_dim < 1 or hidden_size < 1:
            raise ValueError("embed_dim e hidden_size precisam ser >= 1")
        _check_refresh("refresh_n1", refresh_n1)
        _check_refresh("refresh_n2", refresh_n2)
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.l1 = l1
        self.l2 = l2
        self.k1 = k1
        self.k2 = k2
        self.b2 = b2
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.refresh_n1 = refresh_n1
        self.refresh_n2 = refresh_n2
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.heads = nn.ModuleList(nn.Linear(hidden_size, n_experts) for _ in range(n_layers))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.embed.weight, std=0.02)
        self.lstm.reset_parameters()
        for head in self.heads:
            nn.init.normal_(head.weight, std=HEAD_INIT_STD)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        token_ids: torch.Tensor,
        state: TwoHorizonState | None = None,
    ) -> tuple[TwoHorizonDecision, TwoHorizonState]:
        """``token_ids`` é ``[B, T]`` long. Não há hidden de Transformer."""

        if token_ids.ndim != 2:
            raise ValueError(f"token_ids precisa ser [B, T], recebido {tuple(token_ids.shape)}")
        if token_ids.dtype != torch.long:
            raise ValueError("token_ids precisa ser long")
        batch, _seq = token_ids.shape
        lstm_state = None if state is None else _checked_state(
            (state.h, state.c),
            batch=batch,
            hidden_size=self.hidden_size,
            device=token_ids.device,
        )
        hist = None if state is None else _checked_hist(state.raw_hist, batch=batch, router=self, device=token_ids.device)
        embedded = self.embed(token_ids)
        recurrent, new_lstm = self.lstm(embedded, lstm_state)
        raw = self._project(recurrent)
        initial = self._initial_scores(batch, token_ids.device, raw.dtype)
        warm_scores = _align(raw, self.l2, initial, hist)
        hot_scores = _align(raw, self.l1, initial, hist)
        combined = raw if hist is None else torch.cat([hist, raw], dim=1)
        keep = max(self.l1, self.l2)
        raw_hist = combined[:, -keep:].detach()
        hidden, cell = new_lstm
        decision = TwoHorizonDecision(
            raw_scores=raw,
            warm_scores=warm_scores,
            hot_scores=hot_scores,
            l1=self.l1,
            l2=self.l2,
        )
        # ``tier`` fica com ``plan``. ``forward`` só avança o LSTM.
        new_state = TwoHorizonState(
            h=hidden.detach(),
            c=cell.detach(),
            raw_hist=raw_hist,
            tier=None,
        )
        return decision, new_state

    def plan(
        self,
        token_ids: torch.Tensor,
        state: TwoHorizonState | None = None,
        *,
        warm_start: bool = True,
    ) -> tuple["TierPlan", TwoHorizonState]:
        """Scores defasados e a máscara dos dois níveis, a mesma da inferência.

        Sem estado prévio e com ``warm_start``, o morno e o quente nascem
        no ``initial_set`` do BOS. O token 0 não entra nesta sequência: só
        pontua. ``warm_start=False`` é o morno vazio, o comportamento
        antigo, em que o ``b2`` enche o conjunto durante a sequência.
        """

        from moe_memory.tier_mask import initial_set, plan_from_scores

        decision, new_state = self.forward(token_ids, state)
        if state is not None and state.tier is not None:
            tier_in = state.tier
        elif warm_start:
            scores = self.bos_scores(token_ids.shape[0], token_ids.device, decision.raw_scores.dtype)
            tier_in = initial_set(scores, self.k1)
        else:
            tier_in = None
        masks, tier_out = plan_from_scores(
            decision.warm_scores,
            decision.hot_scores,
            k1=self.k1,
            k2=self.k2,
            b2=self.b2,
            refresh_n1=self.refresh_n1,
            refresh_n2=self.refresh_n2,
            l1=self.l1,
            l2=self.l2,
            state=tier_in,
        )
        new_state.tier = tier_out
        return masks, new_state

    def bos_scores(
        self,
        batch: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Score cru do BOS, ``[B, camadas, E]``. Não grava o token na sequência."""

        if batch < 1:
            raise ValueError("batch do BOS precisa ser >= 1")
        if not 0 <= BOS_ID < self.vocab_size:
            raise ValueError(f"BOS id {BOS_ID} está fora do vocabulário {self.vocab_size}")
        ids = torch.full((batch, 1), BOS_ID, dtype=torch.long, device=device)
        with torch.no_grad():
            decision, _state = self.forward(ids)
        scores = decision.raw_scores[:, -1].detach()
        if dtype is not None and scores.dtype != dtype:
            scores = scores.to(dtype=dtype)
        return scores

    def initial_tier(self, batch: int = 1, device: torch.device | None = None) -> "TierState":
        """Conjunto inicial do BOS, o mesmo que ``plan`` semeia sem prompt."""

        from moe_memory.tier_mask import TierState, initial_set

        if device is None:
            device = self.embed.weight.device
        tier: TierState = initial_set(self.bos_scores(batch, device), self.k1)
        return tier

    def bootstrap(self, prompt_ids: torch.Tensor) -> TwoHorizonState:
        """Lê o prompt e devolve o LSTM com o conjunto inicial já cheio.

        O ``step`` do tier continua 0: o prompt não é token decodificado
        e não gasta ``b2``. O score do conjunto é o raw do último token
        do prompt. Sem prompt, o caminho é ``plan`` / ``initial_tier``.
        """

        from moe_memory.tier_mask import initial_set

        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        if prompt_ids.ndim != 2 or prompt_ids.numel() < 1:
            raise ValueError("prompt precisa ser [B, T] ou [T], com ao menos um token")
        if prompt_ids.dtype != torch.long:
            raise ValueError("prompt precisa ser long")
        if int(prompt_ids.min().item()) < 0 or int(prompt_ids.max().item()) >= self.vocab_size:
            raise ValueError("token do prompt fora do vocabulário")
        with torch.no_grad():
            decision, state = self.forward(prompt_ids)
        state.tier = initial_set(decision.raw_scores[:, -1].detach(), self.k1)
        return state

    def _project(self, recurrent: torch.Tensor) -> torch.Tensor:
        parts = [head(recurrent) for head in self.heads]
        return torch.stack(parts, dim=2)

    def _initial_scores(self, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        zeros = torch.zeros(batch, 1, self.hidden_size, device=device, dtype=dtype)
        return self._project(zeros).squeeze(1)


def _check_refresh(name: str, value: int | None) -> None:
    if value is not None and value < 1:
        raise ValueError(f"{name} precisa ser >= 1 ou None")


def _checked_hist(
    hist: torch.Tensor,
    *,
    batch: int,
    router: TwoHorizonLstmRouter,
    device: torch.device,
) -> torch.Tensor:
    if hist.ndim != 4:
        raise ValueError(f"raw_hist precisa ser [B, H, camadas, E], recebido {tuple(hist.shape)}")
    if hist.shape[0] != batch or hist.shape[2] != router.n_layers or hist.shape[3] != router.n_experts:
        raise ValueError(
            "raw_hist não casa com o lote ou com a grade de experts: "
            f"{tuple(hist.shape)} vs batch={batch}, camadas={router.n_layers}, E={router.n_experts}"
        )
    if hist.shape[1] > max(router.l1, router.l2):
        raise ValueError("raw_hist é mais longo que o horizonte")
    if hist.device != device:
        hist = hist.to(device=device)
    return hist


def _align(
    raw: torch.Tensor,
    horizon: int,
    initial: torch.Tensor,
    hist: torch.Tensor | None,
) -> torch.Tensor:
    """``out[:, i]`` é o raw de ``i - horizon``, ou o score do estado zero.

    ``hist`` são os raw anteriores ao chunk, do mais antigo ao mais novo.
    A posição ``t + horizon`` fica com o raw do passo ``t``, que só viu
    tokens de índice ``<= t``.
    """

    if horizon < 1:
        raise ValueError("horizonte precisa ser >= 1")
    batch, seq, _layers, _experts = raw.shape
    combined = raw if hist is None else torch.cat([hist, raw], dim=1)
    seen = 0 if hist is None else hist.shape[1]
    init = initial.unsqueeze(1).expand(batch, horizon, initial.shape[1], initial.shape[2])
    padded = torch.cat([init, combined], dim=1)
    return padded[:, seen : seen + seq]
