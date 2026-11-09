"""Qwen3-MoE cujo gate é o LSTM de dois níveis.

O bloco esparso deixa de consultar o roteador original. A escolha vem de
``TwoHorizonLstmRouter.plan``, a mesma máscara do treino e da inferência.
Cada sequência começa com o conjunto inicial cheio (``k1`` por camada,
fora do ``b2``). Não há cópia SSD/RAM: o ``TierPlan`` lista quem
entraria em cada nível.

A loss deste caminho é a entropia cruzada do próximo token. Não há termo
de troca. Os pesos base ficam congelados; treinam o LSTM e a LoRA dos
experts. Se o expert é ``nn.Linear`` e o pacote ``peft`` está instalado,
a LoRA é a dele. No expert fundido do transformers recente (projeções 3D,
sem ``nn.Linear``) a LoRA local faz a mesma conta, ``y = W(x) + (alpha/r) B(Ax)``,
com ``B = 0`` no início. ``peft`` não enxerga esse parâmetro.

O mock existe para o teste sem ``transformers``. Não é o Qwen3: é um
residual de blocos esparsos com a mesma porta (``gate`` + ``experts``).
Sequência densa: token de padding não está tratado, e avançaria o morno.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from moe_memory.lstm_ahead_router import TwoHorizonLstmRouter, TwoHorizonState, combine_selected
from moe_memory.tier_mask import TierPlan

_EXPERT_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_SPARSE_BLOCK_NAMES = frozenset({"Qwen3MoeSparseMoeBlock", "MockSparseMoeBlock"})


@dataclass
class TwoLevelOutput:
    """Logits da LM e o plano de prefetch do mesmo forward."""

    logits: torch.Tensor
    plan: TierPlan
    state: TwoHorizonState


class LoRALinear(nn.Module):
    """LoRA em volta de um ``nn.Linear`` congelado. ``B`` nasce zero."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRA local envolve nn.Linear")
        if base.bias is not None:
            raise ValueError("projeção de expert com bias não entra nesta LoRA")
        if rank < 1 or alpha <= 0:
            raise ValueError("rank >= 1 e alpha > 0")
        self.base = base
        self.base.weight.requires_grad_(False)
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(hidden, self.base.weight)
        delta = F.linear(F.linear(hidden, self.lora_A), self.lora_B)
        return base_out + self.scale * delta


class FusedExpertLoRA(nn.Module):
    """LoRA nas projeções fundidas ``gate_up_proj`` e ``down_proj``.

    O peso base fica congelado. A assinatura repete a do
    ``Qwen3MoeExperts``: hidden ``[N, D]``, índice ``[N, K]``, peso ``[N, K]``.
    """

    def __init__(self, base: nn.Module, rank: int, alpha: float) -> None:
        super().__init__()
        if rank < 1 or alpha <= 0:
            raise ValueError("rank >= 1 e alpha > 0")
        if not _is_fused_experts(base):
            raise TypeError("FusedExpertLoRA espera gate_up_proj e down_proj")
        self.base = base
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        experts = int(base.num_experts)
        hidden = int(base.hidden_dim)
        intermediate = int(base.intermediate_dim)
        self.scale = alpha / rank
        self.lora_gate_up_A = nn.Parameter(torch.empty(experts, rank, hidden))
        self.lora_gate_up_B = nn.Parameter(torch.zeros(experts, 2 * intermediate, rank))
        self.lora_down_A = nn.Parameter(torch.empty(experts, rank, intermediate))
        self.lora_down_B = nn.Parameter(torch.zeros(experts, hidden, rank))
        nn.init.kaiming_uniform_(self.lora_gate_up_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_down_A, a=math.sqrt(5))

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        mixed = hidden_states.new_zeros(hidden_states.shape)
        for expert_id in range(int(self.base.num_experts)):
            selected = top_k_index == expert_id
            if not bool(selected.any().item()):
                continue
            slot_weight = (top_k_weights * selected.to(dtype=top_k_weights.dtype)).sum(dim=1)
            token_sel = slot_weight != 0
            if not bool(token_sel.any().item()):
                continue
            token_index = token_sel.nonzero(as_tuple=False).flatten()
            current = hidden_states.index_select(0, token_index)
            gate_up = F.linear(current, self.base.gate_up_proj[expert_id])
            gate_up = gate_up + self.scale * F.linear(
                F.linear(current, self.lora_gate_up_A[expert_id]),
                self.lora_gate_up_B[expert_id],
            )
            gate, up = gate_up.chunk(2, dim=-1)
            mid = self.base.act_fn(gate) * up
            down = F.linear(mid, self.base.down_proj[expert_id])
            down = down + self.scale * F.linear(
                F.linear(mid, self.lora_down_A[expert_id]),
                self.lora_down_B[expert_id],
            )
            weighted = down * slot_weight.index_select(0, token_index).unsqueeze(-1)
            mixed = torch.index_add(mixed, 0, token_index, weighted)
        return mixed


class MockExpert(nn.Module):
    """SwiGLU mínimo, três ``nn.Linear`` sem bias, como o expert do Qwen3 antigo."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        activated = F.silu(self.gate_proj(hidden)) * self.up_proj(hidden)
        return self.down_proj(activated)


class MockSparseMoeBlock(nn.Module):
    """Porta do bloco esparso: um ``gate`` que o adapter não pode chamar."""

    def __init__(self, hidden_size: int, n_experts: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_size, n_experts, bias=False)
        self.experts = nn.ModuleList(
            MockExpert(hidden_size, intermediate_size) for _ in range(n_experts)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("o gate original não roda; o adapter troca este forward")


class MockQwen3Moe(nn.Module):
    """LM minúscula com blocos esparsos. Sem atenção e sem peso baixado.

    Serve quando ``transformers`` não está instalado, e como caminho de
    CPU ao lado do Qwen3 real. O adapter enxerga os mesmos nomes de bloco.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        n_layers: int,
        n_experts: int,
        intermediate_size: int,
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.blocks = nn.ModuleList(
            MockSparseMoeBlock(hidden_size, n_experts, intermediate_size) for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.lm_head(self.norm(hidden))


class TwoLevelQwen3(nn.Module):
    """Backbone Qwen3 (ou o mock) com o roteador externo e a LoRA nos experts."""

    def __init__(
        self,
        backbone: nn.Module,
        router: TwoHorizonLstmRouter,
        *,
        lora_rank: int = 4,
        lora_alpha: float = 8.0,
        lora_impl: str = "auto",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.router = router
        self.blocks = iter_sparse_moe_blocks(backbone)
        if len(self.blocks) != router.n_layers:
            raise ValueError(
                f"o backbone tem {len(self.blocks)} blocos esparsos e o roteador "
                f"tem {router.n_layers} camadas"
            )
        expert_count = _expert_count(self.blocks[0])
        if expert_count != router.n_experts:
            raise ValueError(
                f"o bloco tem {expert_count} experts e o roteador tem {router.n_experts}"
            )
        self.lora_mode = attach_expert_lora(backbone, lora_rank, lora_alpha, impl=lora_impl)
        self._holder: dict[str, TierPlan | None] = {"plan": None}
        for layer, block in enumerate(self.blocks):
            _patch_block(block, layer, self._holder)
        self._hf = backbone.__class__.__name__.startswith("Qwen3Moe")

    def forward(
        self,
        input_ids: torch.Tensor,
        state: TwoHorizonState | None = None,
    ) -> TwoLevelOutput:
        plan, state = self.router.plan(input_ids, state)
        self._holder["plan"] = plan
        try:
            logits = self._logits(input_ids)
        finally:
            self._holder["plan"] = None
        return TwoLevelOutput(logits=logits, plan=plan, state=state)

    def _logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self._hf:
            output = self.backbone(
                input_ids=input_ids,
                use_cache=False,
                output_router_logits=False,
            )
            return output.logits
        output = self.backbone(input_ids)
        if hasattr(output, "logits"):
            return output.logits
        return output


def iter_sparse_moe_blocks(model: nn.Module) -> list[nn.Module]:
    return [module for module in model.modules() if module.__class__.__name__ in _SPARSE_BLOCK_NAMES]


def attach_expert_lora(
    model: nn.Module,
    rank: int,
    alpha: float,
    impl: str = "auto",
) -> str:
    """Congela o que não é LoRA e pendura o delta nos experts.

    ``impl`` é ``auto`` (peft se der, senão local), ``peft`` ou ``local``.
    Expert fundido sempre usa a LoRA local: não há ``nn.Linear`` para o peft.
    """

    if impl not in {"auto", "peft", "local"}:
        raise ValueError("impl da LoRA é auto, peft ou local")
    if rank < 1 or alpha <= 0:
        raise ValueError("rank >= 1 e alpha > 0")
    blocks = iter_sparse_moe_blocks(model)
    if not blocks:
        raise ValueError("nenhum bloco MoE esparso para a LoRA")
    linear = [block for block in blocks if isinstance(block.experts, nn.ModuleList)]
    fused = [block for block in blocks if _is_fused_experts(block.experts)]
    if len(linear) + len(fused) != len(blocks):
        raise TypeError("bloco esparso sem lista de experts nem projeção fundida")
    used: list[str] = []
    if linear:
        if impl == "peft" or (impl == "auto" and _peft_available()):
            if not _peft_available():
                raise ImportError("peft não está instalado")
            _inject_peft(model, rank, alpha)
            used.append("peft")
        else:
            for block in linear:
                _wrap_linears(block, rank, alpha)
            used.append("local")
    if fused:
        for block in fused:
            block.experts = FusedExpertLoRA(block.experts, rank=rank, alpha=alpha)
        used.append("fused")
    _freeze_non_lora(model)
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("LoRA não criou parâmetro treinável")
    return "+".join(used)


def language_model_loss(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """CE do próximo token. Sem troca, sem balanceamento, sem λ."""

    if token_ids.ndim != 2 or token_ids.dtype != torch.long:
        raise ValueError("token_ids precisa ser [B, T] long")
    if token_ids.shape[1] < 2:
        raise ValueError("a CE do próximo token precisa de ao menos 2 posições")
    if logits.shape[0] != token_ids.shape[0] or logits.shape[1] != token_ids.shape[1]:
        raise ValueError(
            f"logits {tuple(logits.shape)} não casa com tokens {tuple(token_ids.shape)}"
        )
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        token_ids[:, 1:].reshape(-1),
    )


def training_step(
    module: TwoLevelQwen3,
    optimizer: torch.optim.Optimizer,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """Um passo de Adam (ou o otimizador que vier) na loss de linguagem."""

    module.train()
    optimizer.zero_grad(set_to_none=True)
    output = module(token_ids)
    loss = language_model_loss(output.logits, token_ids)
    loss.backward()
    optimizer.step()
    return loss.detach()


def transformers_qwen3_available() -> bool:
    try:
        from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM  # noqa: F401
    except ImportError:
        return False
    return True


def build_tiny_qwen3_backbone(
    *,
    vocab_size: int = 64,
    hidden_size: int = 64,
    n_layers: int = 2,
    n_experts: int = 8,
    intermediate_size: int = 64,
    backend: str = "auto",
) -> nn.Module:
    """Qwen3Moe aleatório e minúsculo, ou o mock se o pacote não estiver aí.

    ``auto`` usa o transformers quando ``Qwen3MoeForCausalLM`` importa.
    Não baixa checkpoint.
    """

    if backend not in {"auto", "mock", "transformers"}:
        raise ValueError("backend é auto, mock ou transformers")
    if min(vocab_size, hidden_size, n_layers, n_experts, intermediate_size) < 1:
        raise ValueError("dimensões do backbone minúsculo precisam ser >= 1")
    use_hf = backend == "transformers" or (backend == "auto" and transformers_qwen3_available())
    if use_hf:
        if not transformers_qwen3_available():
            raise ImportError("transformers com Qwen3Moe não está instalado")
        return _build_hf_tiny(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            n_experts=n_experts,
            intermediate_size=intermediate_size,
        )
    return MockQwen3Moe(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        n_layers=n_layers,
        n_experts=n_experts,
        intermediate_size=intermediate_size,
    )


def _build_hf_tiny(
    *,
    vocab_size: int,
    hidden_size: int,
    n_layers: int,
    n_experts: int,
    intermediate_size: int,
) -> nn.Module:
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    heads = 4 if hidden_size % 4 == 0 else 1
    if hidden_size % heads != 0:
        raise ValueError(f"hidden_size={hidden_size} não fecha com {heads} cabeças")
    kv_heads = 2 if heads % 2 == 0 else 1
    config = Qwen3MoeConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        moe_intermediate_size=intermediate_size,
        num_hidden_layers=n_layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        num_experts=n_experts,
        num_experts_per_tok=min(8, n_experts),
        decoder_sparse_step=1,
        mlp_only_layers=[],
        max_position_embeddings=256,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        norm_topk_prob=True,
        tie_word_embeddings=False,
        router_aux_loss_coef=0.0,
        output_router_logits=False,
        use_cache=False,
        initializer_range=0.02,
    )
    model = Qwen3MoeForCausalLM(config)
    model.config.output_router_logits = False
    return model


def _patch_block(block: nn.Module, layer: int, holder: dict[str, TierPlan | None]) -> None:
    def forward(hidden_states: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        del args, kwargs
        plan = holder["plan"]
        if plan is None:
            raise RuntimeError("plano de tiers ausente; o gate original não é fallback")
        if hidden_states.ndim != 3:
            raise ValueError(
                f"hidden do MoE precisa ser [B, T, D], recebido {tuple(hidden_states.shape)}"
            )
        batch, seq, _dim = hidden_states.shape
        if plan.hot_index.shape[0] != batch or plan.hot_index.shape[1] != seq:
            raise ValueError("plano e hidden não têm o mesmo lote ou o mesmo tempo")
        if layer >= plan.hot_index.shape[2]:
            raise ValueError(f"camada {layer} fora do plano")
        index = plan.hot_index[:, :, layer, :]
        weight = plan.soft_weights[:, :, layer, :]
        flat = hidden_states.reshape(batch * seq, hidden_states.shape[-1])
        mixed = _dispatch(
            block.experts,
            flat,
            index.reshape(batch * seq, index.shape[-1]),
            weight.reshape(batch * seq, weight.shape[-1]),
        )
        return mixed.view(batch, seq, hidden_states.shape[-1])

    block.forward = forward


def _dispatch(
    experts: nn.Module,
    flat: torch.Tensor,
    index: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    if isinstance(experts, FusedExpertLoRA) or _is_fused_experts(experts):
        return experts(flat, index, weight)
    if isinstance(experts, nn.ModuleList):
        hidden = flat.view(1, flat.shape[0], flat.shape[1])
        mixed = combine_selected(
            hidden,
            list(experts),
            index.view(1, index.shape[0], index.shape[1]),
            weight.view(1, weight.shape[0], weight.shape[1]),
        )
        return mixed.view(flat.shape[0], flat.shape[1])
    raise TypeError(f"experts de tipo {type(experts).__name__} não têm despacho")


def _wrap_linears(block: nn.Module, rank: int, alpha: float) -> None:
    for expert in block.experts:
        for name in _EXPERT_PROJECTIONS:
            linear = getattr(expert, name)
            if isinstance(linear, LoRALinear):
                continue
            setattr(expert, name, LoRALinear(linear, rank, alpha))


def _inject_peft(model: nn.Module, rank: int, alpha: float) -> None:
    from peft import LoraConfig, inject_adapter_in_model

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=list(_EXPERT_PROJECTIONS),
        bias="none",
        lora_dropout=0.0,
    )
    inject_adapter_in_model(config, model)


def _freeze_non_lora(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if "lora_" not in name:
            parameter.requires_grad_(False)


def _peft_available() -> bool:
    try:
        import peft  # noqa: F401
    except ImportError:
        return False
    return True


def _is_fused_experts(experts: nn.Module) -> bool:
    gate_up = getattr(experts, "gate_up_proj", None)
    down = getattr(experts, "down_proj", None)
    return isinstance(gate_up, nn.Parameter) and isinstance(down, nn.Parameter) and hasattr(experts, "num_experts")


def _expert_count(block: nn.Module) -> int:
    experts = block.experts
    if isinstance(experts, nn.ModuleList):
        return len(experts)
    if hasattr(experts, "num_experts"):
        return int(experts.num_experts)
    raise TypeError(f"não sei contar experts em {type(experts).__name__}")
