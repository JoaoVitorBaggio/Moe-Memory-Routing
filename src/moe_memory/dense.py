"""LM denso de referência (modelo 3 do protocolo).

Não é o MoE com λ=0. λ=0 ainda roteia entre ``E`` experts, ainda tem
residência e ainda soma ``β L_balanço``; só o peso de ``C_mem`` fica em
zero. Este módulo não importa emulador, residência nem ``compute_loss``.

A largura do FFN é ``k * expert_hidden``. As duas matrizes de ``k``
experts ativos têm a mesma conta que um FFN dessa largura. O protocolo
pede parâmetros *ativos* ≈ tronco + ``k``·expert, não o total dos ``E``
experts (~48 M com o ``lm_head`` amarrado). O roteador fica só no MoE:
a diferença é pequena e o doc descreve a conta como aproximada.

O bloco reusa a atenção causal e o ``ExpertFFN`` do MoE (GELU, o mesmo
residual). Não há ``torch.compile``.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from moe_memory.config import ExperimentConfig, ModelConfig
from moe_memory.levels import N_LEVELS
from moe_memory.loss import task_cross_entropy
from moe_memory.model import CausalSelfAttention, ExpertFFN

MODEL_KIND = "dense"


def dense_ffn_width(model: ModelConfig) -> int:
    """Largura do FFN denso: ``k`` vezes o hidden de um expert."""

    width = int(model.k) * int(model.expert_hidden)
    if width < 1:
        raise ValueError(f"largura densa inválida: k={model.k} expert_hidden={model.expert_hidden}")
    return width


def _layernorm_parameters(d_model: int) -> int:
    return 2 * d_model


def _attention_parameters(d_model: int) -> int:
    qkv = d_model * (3 * d_model) + 3 * d_model
    out = d_model * d_model + d_model
    return qkv + out


def _expert_parameters(d_model: int, d_ff: int) -> int:
    fc1 = d_model * d_ff + d_ff
    fc2 = d_ff * d_model + d_model
    return fc1 + fc2


def _router_parameters(d_model: int, n_experts: int) -> int:
    """Projeção sem viés + chaves dos experts + embed dos níveis.

    O ``MemoryAwareRouter`` sempre aloca o embed de ``ℓ(e)``, mesmo quando
    o forward não soma o nível.
    """

    proj = d_model * d_model
    keys = n_experts * d_model
    levels = N_LEVELS * d_model
    return proj + keys + levels


def _embedding_parameters(model: ModelConfig) -> int:
    total = model.vocab_size * model.d_model + model.max_seq_len * model.d_model
    if not model.tie_embeddings:
        total += model.d_model * model.vocab_size
    return total


def count_dense_parameters(model: ModelConfig) -> int:
    d_ff = dense_ffn_width(model)
    block = (
        2 * _layernorm_parameters(model.d_model)
        + _attention_parameters(model.d_model)
        + _expert_parameters(model.d_model, d_ff)
    )
    return _embedding_parameters(model) + model.n_layers * block + _layernorm_parameters(model.d_model)


def count_moe_parameters(model: ModelConfig) -> int:
    block = (
        2 * _layernorm_parameters(model.d_model)
        + _attention_parameters(model.d_model)
        + _router_parameters(model.d_model, model.E)
        + model.E * _expert_parameters(model.d_model, model.expert_hidden)
    )
    return _embedding_parameters(model) + model.n_layers * block + _layernorm_parameters(model.d_model)


def count_moe_active_parameters(model: ModelConfig) -> int:
    """Tronco do MoE (com roteador) mais ``k`` experts por camada.

    Experts fora do top-k não entram. É a conta do protocolo, não o
    total armazenado.
    """

    expert = _expert_parameters(model.d_model, model.expert_hidden)
    inactive = (model.E - model.k) * expert * model.n_layers
    return count_moe_parameters(model) - inactive


def count_router_parameters(model: ModelConfig) -> int:
    return model.n_layers * _router_parameters(model.d_model, model.E)


def active_budget_gap(model: ModelConfig) -> int:
    """O que o MoE ativo tem a mais que o denso: roteador e o viés ``fc2``.

    ``k`` experts trazem ``k`` viéses na projeção de volta; o FFN denso
    tem um. As matrizes ``d_model × d_ff`` coincidem.
    """

    bias = model.n_layers * (model.k - 1) * model.d_model
    return count_router_parameters(model) + bias


def capacity_report(config: ExperimentConfig) -> dict[str, int | str | bool]:
    model = config.model
    dense = count_dense_parameters(model)
    active = count_moe_active_parameters(model)
    return {
        "kind": MODEL_KIND,
        "d_ff": dense_ffn_width(model),
        "dense_parameters": dense,
        "moe_total_parameters": count_moe_parameters(model),
        "moe_active_parameters": active,
        "active_budget_gap": active - dense,
        "router_parameters": count_router_parameters(model),
        "memory_path": False,
        "router": False,
    }


class DenseBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = ExpertFFN(d_model, d_ff)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attn(self.ln1(hidden))
        return hidden + self.ffn(self.ln2(hidden))


class DenseLanguageModel(nn.Module):
    """Transformer causal denso. O forward não recebe níveis nem experts."""

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        vocab_size: int,
        max_seq_len: int,
        tie_embeddings: bool,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers precisa ser >= 1")
        if d_ff < 1:
            raise ValueError("d_ff precisa ser >= 1")
        self.kind = MODEL_KIND
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.tie_embeddings = tie_embeddings
        self.memory_path = False
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            DenseBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = None if tie_embeddings else nn.Linear(d_model, vocab_size, bias=False)
        self._init_parameters()

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> DenseLanguageModel:
        model = config.model
        return cls(
            n_layers=model.n_layers,
            d_model=model.d_model,
            n_heads=model.n_heads,
            d_ff=dense_ffn_width(model),
            vocab_size=model.vocab_size,
            max_seq_len=model.max_seq_len,
            tie_embeddings=model.tie_embeddings,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids precisa ser [B, T], recebido {tuple(input_ids.shape)}")
        _batch, seq = input_ids.shape
        if seq > self.max_seq_len:
            raise ValueError(f"seq_len={seq} passa de max_seq_len={self.max_seq_len}")
        positions = torch.arange(seq, device=input_ids.device)
        hidden = self.tok_emb(input_ids) + self.pos_emb(positions)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.ln_f(hidden)
        head = self.tok_emb.weight if self.tie_embeddings else self.lm_head.weight
        return torch.nn.functional.linear(hidden, head)

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _init_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
        residual_std = 0.02 / math.sqrt(2 * self.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, std=residual_std)
            nn.init.normal_(block.ffn.fc2.weight, std=residual_std)


def prepare_baseline(
    config: ExperimentConfig,
    *,
    device: torch.device | None = None,
    seed: int = 0,
):
    """Modelo fp32 e AdamW. Não há emulador nem residência para amarrar."""

    device = device or torch.device("cpu")
    if config.model.dtype != "fp32":
        raise ValueError(f"dtype={config.model.dtype!r}; o baseline é fp32")
    torch.manual_seed(seed)
    model = DenseLanguageModel.from_config(config).to(device=device, dtype=torch.float32)
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("o modelo saiu de fp32")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.smoke.lr)
    return model, optimizer


def train_dense_step(
    model: DenseLanguageModel,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
) -> dict[str, float | int]:
    """Um passo de AdamW. A loss é só a CE do próximo token.

    ``total`` e ``task`` são o mesmo número: não existe ``C_mem`` nem
    ``L_balanço`` para somar.
    """

    if input_ids.ndim != 2 or input_ids.size(1) < 2:
        raise ValueError(f"input_ids precisa ser [B, T>=2], recebido {tuple(input_ids.shape)}")
    model.train()
    task = task_cross_entropy(model(input_ids), input_ids)
    optimizer.zero_grad(set_to_none=True)
    task.backward()
    grad_norm = _grad_norm(model)
    optimizer.step()
    value = float(task.detach())
    return {
        "total": value,
        "task": value,
        "grad_norm": grad_norm,
        "parameters": model.num_parameters(),
    }


def _grad_norm(model: nn.Module) -> float:
    total = torch.zeros((), dtype=torch.float32)
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        total = total + parameter.grad.detach().float().pow(2).sum()
    return float(total.sqrt())
