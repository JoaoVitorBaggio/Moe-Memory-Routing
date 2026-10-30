"""Transformer causal pequeno em que todo FFN é MoE.

A mistura dos experts é a média ponderada dos top-2 nos dois modos
(treino e inferência). Os pesos são a probabilidade do roteador
renormalizada nos dois escolhidos, então somam 1 e a escala do residual
não muda entre ``train()`` e ``eval()``.

A diferença fechada no protocolo — média ponderada no treino, soma na
inferência — é do *custo*, não do hidden. Ela está em ``loss.py``
("soma dos top-2 — custo real de acesso"). Aplicar a soma nos vetores
do FFN mudaria a escala do residual na avaliação.

Não há descarte por capacidade: todo token fica com os seus ``k``
experts. Jitter do Switch não entra no mínimo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from moe_memory.config import ExperimentConfig, ModelConfig
from moe_memory.router import MemoryAwareRouter


@dataclass(frozen=True)
class ModelOutput:
    logits: torch.Tensor
    probs: torch.Tensor
    topk_index: torch.Tensor
    dispatched: torch.Tensor


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} não é divisível por n_heads={n_heads}")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, seq, channels = hidden.shape
        query, key, value = (
            self.qkv(hidden)
            .view(batch, seq, 3, self.n_heads, self.d_head)
            .unbind(dim=2)
        )
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.d_head)
        causal = torch.ones(seq, seq, device=hidden.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(causal, float("-inf"))
        weights = scores.softmax(dim=-1)
        mixed = (weights @ value).transpose(1, 2).contiguous().view(batch, seq, channels)
        return self.out_proj(mixed)


class ExpertFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(hidden)))


class MixtureOfExperts(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int,
        k: int,
        use_level: bool,
    ) -> None:
        super().__init__()
        if not 1 <= k < n_experts:
            raise ValueError(f"k={k} precisa satisfazer 1 <= k < E={n_experts}")
        self.k = k
        self.n_experts = n_experts
        self.use_level = use_level
        self.router = MemoryAwareRouter(d_model, n_experts)
        self.experts = nn.ModuleList(ExpertFFN(d_model, d_ff) for _ in range(n_experts))

    def forward(
        self,
        hidden: torch.Tensor,
        levels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.router(hidden, levels, use_level=self.use_level)
        probs = logits.softmax(dim=-1)
        index = torch.topk(logits, k=self.k, dim=-1).indices
        picked = probs.gather(-1, index)
        gates = picked / picked.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        mixed = self._mix(hidden, index, gates)
        return mixed, probs, index

    def _mix(
        self,
        hidden: torch.Tensor,
        index: torch.Tensor,
        gates: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq, dim = hidden.shape
        flat = hidden.reshape(batch * seq, dim)
        flat_index = index.reshape(batch * seq, self.k)
        flat_gates = gates.reshape(batch * seq, self.k)
        mixed = flat.new_zeros(flat.shape)
        for expert_id, expert in enumerate(self.experts):
            selected = flat_index == expert_id
            if not bool(selected.any()):
                continue
            token_sel = selected.any(dim=1)
            weight = (flat_gates * selected.to(dtype=flat_gates.dtype)).sum(dim=1)
            token_index = token_sel.nonzero(as_tuple=False).flatten()
            expert_out = expert(flat.index_select(0, token_index))
            weighted = expert_out * weight.index_select(0, token_index).unsqueeze(-1)
            mixed = torch.index_add(mixed, 0, token_index, weighted)
        return mixed.view(batch, seq, dim)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_experts: int,
        k: int,
        use_level: bool,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = MixtureOfExperts(d_model, d_ff, n_experts, k, use_level)

    def forward(
        self,
        hidden: torch.Tensor,
        levels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = hidden + self.attn(self.ln1(hidden))
        mixed, probs, index = self.ffn(self.ln2(hidden), levels)
        return hidden + mixed, probs, index


class MemoryMoEModel(nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_experts: int,
        k: int,
        vocab_size: int,
        max_seq_len: int,
        use_level: bool,
        tie_embeddings: bool,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_experts = n_experts
        self.k = k
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.tie_embeddings = tie_embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            TransformerBlock(d_model, n_heads, d_ff, n_experts, k, use_level)
            for _ in range(n_layers)
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = None if tie_embeddings else nn.Linear(d_model, vocab_size, bias=False)
        self._init_parameters()

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> MemoryMoEModel:
        model = config.model
        return cls.from_model_config(model, use_level=config.memory.router_sees_level)

    @classmethod
    def from_model_config(cls, model: ModelConfig, *, use_level: bool) -> MemoryMoEModel:
        return cls(
            n_layers=model.n_layers,
            d_model=model.d_model,
            n_heads=model.n_heads,
            d_ff=model.expert_hidden,
            n_experts=model.E,
            k=model.k,
            vocab_size=model.vocab_size,
            max_seq_len=model.max_seq_len,
            use_level=use_level,
            tie_embeddings=model.tie_embeddings,
        )

    def forward(self, input_ids: torch.Tensor, levels: torch.Tensor) -> ModelOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids precisa ser [B, T], recebido {tuple(input_ids.shape)}")
        batch, seq = input_ids.shape
        if seq > self.max_seq_len:
            raise ValueError(f"seq_len={seq} passa de max_seq_len={self.max_seq_len}")
        if levels.shape != (self.n_layers, self.n_experts):
            raise ValueError(
                f"levels {tuple(levels.shape)} não é {(self.n_layers, self.n_experts)}"
            )
        if levels.dtype != torch.long:
            levels = levels.to(dtype=torch.long)
        levels = levels.to(device=input_ids.device)
        positions = torch.arange(seq, device=input_ids.device)
        hidden = self.tok_emb(input_ids) + self.pos_emb(positions)
        prob_layers = []
        index_layers = []
        counts = []
        for layer_id, block in enumerate(self.blocks):
            hidden, probs, index = block(hidden, levels[layer_id])
            prob_layers.append(probs)
            index_layers.append(index)
            counts.append(torch.bincount(index.reshape(-1), minlength=self.n_experts))
        hidden = self.ln_f(hidden)
        head = self.tok_emb.weight if self.tie_embeddings else self.lm_head.weight
        logits = F.linear(hidden, head)
        return ModelOutput(
            logits=logits,
            probs=torch.stack(prob_layers, dim=0),
            topk_index=torch.stack(index_layers, dim=0),
            dispatched=torch.stack(counts, dim=0).to(device="cpu"),
        )

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def expert_nbytes(self) -> int:
        expert = self.blocks[0].ffn.experts[0]
        return sum(parameter.numel() * parameter.element_size() for parameter in expert.parameters())

    def expert_parameter_vectors(self) -> list[list[torch.Tensor]]:
        vectors: list[list[torch.Tensor]] = []
        for block in self.blocks:
            layer = []
            for expert in block.ffn.experts:
                flat = torch.cat(
                    [
                        parameter.detach().reshape(-1).to(dtype=torch.float32, device="cpu")
                        for parameter in expert.parameters()
                    ]
                )
                layer.append(flat)
            vectors.append(layer)
        return vectors

    def _init_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, MemoryAwareRouter):
                nn.init.normal_(module.expert_keys, std=0.02)
        residual_std = 0.02 / math.sqrt(2 * self.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, std=residual_std)
            for expert in block.ffn.experts:
                nn.init.normal_(expert.fc2.weight, std=residual_std)
