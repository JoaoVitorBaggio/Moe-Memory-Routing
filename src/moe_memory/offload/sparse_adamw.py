"""AdamW esparso com o mesmo passo do AdamW denso ``foreach=False``.

O AdamW do projeto atualiza o tensor inteiro da LoRA. A fatia de um
expert sem gradiente ainda sofre weight decay desacoplado e o termo de
momento: com gradiente zero, ``m`` e ``v`` decaem e o peso anda.

Não há forma fechada exata para a soma

``Σ_j α^{k-j} (β1^j m) / (1-β1^{s+j}) / (sqrt(β2^j v / (1-β2^{s+j})) + eps)``

porque o viés e o ``eps`` dependem do passo de um jeito que não soma.
O atraso de ``k`` passos é a mesma conta do PyTorch, repetida ``k``
vezes e vetorizada na fatia. Weight decay zero e momentos zero não
mexem no peso: essa fatia não guarda ``m`` nem ``v``.

O estado mora na CPU, como o Adam do treino. A VRAM não recebe ``m``/``v``.
"""

from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


def adamw_inplace(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: float,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    eps: float,
) -> None:
    """Um passo, na ordem do ``_single_tensor_adam`` com decay desacoplado."""

    if weight_decay != 0:
        param.mul_(1 - lr * weight_decay)
    exp_avg.lerp_(grad, 1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    step_size = lr / bias_correction1
    denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(eps)
    param.addcdiv_(exp_avg, denom, value=-step_size)


class SparseAdamW(Optimizer):
    """Fatias ``[camada, expert]`` para tensores com 3 ou mais eixos.

    O LSTM e o resto seguem o AdamW denso. O passo do tensor é único,
    igual ao do PyTorch: sobe quando alguma fatia daquele tensor tem
    gradiente. As outras fatias só são postas em dia quando vão ser
    lidas, ou no checkpoint.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0:
            raise ValueError(f"lr inválido: {lr}")
        if eps < 0:
            raise ValueError(f"eps inválido: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"betas inválidos: {betas}")
        if weight_decay < 0:
            raise ValueError(f"weight decay inválido: {weight_decay}")
        super().__init__(
            params,
            {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay},
        )
        self._moments: dict[int, dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]]] = {}
        self._through: dict[int, torch.Tensor] = {}
        self._groups: dict[int, dict] = {}
        self.last_stats = {
            "updated_tensors": 0,
            "updated_params": 0,
            "state_bytes": 0,
            "moment_slices": 0,
        }
        for group in self.param_groups:
            for param in group["params"]:
                self._groups[id(param)] = group
                if param.ndim >= 3:
                    param._sparse_adamw = self  # noqa: SLF001

    def zero_grad(self, set_to_none: bool = True) -> None:
        super().zero_grad(set_to_none=set_to_none)
        from moe_memory.offload.lm_train import _SLICE_GRADS

        _SLICE_GRADS.clear()

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        from moe_memory.offload.lm_train import _SLICE_GRADS

        queued = list(_SLICE_GRADS)
        _SLICE_GRADS.clear()
        bundled: dict[int, dict[tuple[int, int], torch.Tensor]] = {}
        params_by_id: dict[int, torch.nn.Parameter] = {}
        for param, layer, expert, grad in queued:
            params_by_id[id(param)] = param
            slot = bundled.setdefault(id(param), {})
            key = (int(layer), int(expert))
            found = slot.get(key)
            aligned = grad.detach()
            if aligned.device != param.device or aligned.dtype != param.dtype:
                aligned = aligned.to(device=param.device, dtype=param.dtype)
            if found is None:
                slot[key] = torch.zeros_like(param[key[0], key[1]])
            slot[key].add_(aligned)
        updated_tensors = 0
        updated_params = 0
        with torch.no_grad():
            seen: set[int] = set()
            for group in self.param_groups:
                beta1, beta2 = group["betas"]
                for param in group["params"]:
                    pid = id(param)
                    seen.add(pid)
                    if param.ndim >= 3:
                        grads = bundled.get(pid)
                        if not grads:
                            continue
                        step = self._advance(param)
                        for key, grad in grads.items():
                            self._catch_up_slice(param, key, step - 1, group)
                            self._apply_slice(param, key, grad, float(step), group)
                            updated_tensors += 1
                            updated_params += grad.numel()
                        continue
                    if param.grad is None:
                        continue
                    self._dense_param(param, group, beta1, beta2)
                    updated_tensors += 1
                    updated_params += param.numel()
            for pid, grads in bundled.items():
                if pid in seen:
                    continue
                param = params_by_id[pid]
                group = self._groups[pid]
                step = self._advance(param)
                for key, grad in grads.items():
                    self._catch_up_slice(param, key, step - 1, group)
                    self._apply_slice(param, key, grad, float(step), group)
                    updated_tensors += 1
                    updated_params += grad.numel()
        self.last_stats = {
            "updated_tensors": updated_tensors,
            "updated_params": updated_params,
            "state_bytes": self.state_bytes(),
            "moment_slices": self.moment_slices(),
        }
        return loss

    def catch_up_slice(self, param: torch.Tensor, layer: int, expert: int) -> None:
        """Põe a fatia no passo já concluído, antes de ela entrar no forward."""

        state = self.state.get(param)
        if not state:
            return
        group = self._groups[id(param)]
        with torch.no_grad():
            self._catch_up_slice(param, (int(layer), int(expert)), int(state["step"].item()), group)

    def catch_up_all(self) -> None:
        """Fecha o atraso de toda fatia. O checkpoint grava esse estado."""

        with torch.no_grad():
            for group in self.param_groups:
                for param in group["params"]:
                    if param.ndim < 3:
                        continue
                    state = self.state.get(param)
                    if not state:
                        continue
                    target = int(state["step"].item())
                    layers, experts = int(param.shape[0]), int(param.shape[1])
                    for layer in range(layers):
                        for expert in range(experts):
                            self._catch_up_slice(param, (layer, expert), target, group)

    def state_bytes(self) -> int:
        total = 0
        for moments in self._moments.values():
            for exp_avg, exp_avg_sq in moments.values():
                total += exp_avg.numel() * exp_avg.element_size()
                total += exp_avg_sq.numel() * exp_avg_sq.element_size()
        for through in self._through.values():
            total += through.numel() * through.element_size()
        return int(total)

    def moment_slices(self) -> int:
        return sum(len(moments) for moments in self._moments.values())

    def materialize(self, param: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``m`` e ``v`` densos. Fatia sem histórico fica em zero."""

        exp_avg = torch.zeros_like(param)
        exp_avg_sq = torch.zeros_like(param)
        for (layer, expert), (moment, second) in self._moments.get(id(param), {}).items():
            exp_avg[layer, expert].copy_(moment)
            exp_avg_sq[layer, expert].copy_(second)
        return exp_avg, exp_avg_sq

    def slice_entries(self, param: torch.Tensor):
        return self._moments.get(id(param), {}).items()

    def load_slice(
        self,
        param: torch.Tensor,
        layer: int,
        expert: int,
        *,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        through: int,
    ) -> None:
        self._ensure_through(param)
        self._through[id(param)][layer, expert] = through
        self._moments.setdefault(id(param), {})[(layer, expert)] = (exp_avg, exp_avg_sq)

    def mark_through(self, param: torch.Tensor, step: int) -> None:
        """Toda fatia sem registro fica no ``step`` gravado no checkpoint."""

        layers, experts = int(param.shape[0]), int(param.shape[1])
        through = torch.full((layers, experts), int(step), dtype=torch.int32)
        known = self._through.get(id(param))
        if known is not None:
            for (layer, expert) in self._moments.get(id(param), {}):
                through[layer, expert] = known[layer, expert]
        self._through[id(param)] = through

    def _advance(self, param: torch.Tensor) -> int:
        state = self.state[param]
        if len(state) == 0:
            state["step"] = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        state["step"] += 1
        self._ensure_through(param)
        return int(state["step"].item())

    def _ensure_through(self, param: torch.Tensor) -> torch.Tensor:
        found = self._through.get(id(param))
        if found is None:
            found = torch.zeros((int(param.shape[0]), int(param.shape[1])), dtype=torch.int32)
            self._through[id(param)] = found
        return found

    def _catch_up_slice(self, param: torch.Tensor, key: tuple[int, int], target: int, group: dict) -> None:
        if target <= 0:
            return
        through = self._ensure_through(param)
        done = int(through[key].item())
        if done >= target:
            return
        layer, expert = key
        view = param[layer, expert]
        moments = self._moments.get(id(param), {}).get(key)
        beta1, beta2 = group["betas"]
        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        eps = float(group["eps"])
        if moments is None:
            if weight_decay != 0:
                alpha = 1 - lr * weight_decay
                for _ in range(target - done):
                    view.mul_(alpha)
            through[key] = target
            return
        exp_avg, exp_avg_sq = moments
        zero = torch.zeros_like(view)
        for step in range(done + 1, target + 1):
            adamw_inplace(
                view,
                zero,
                exp_avg,
                exp_avg_sq,
                float(step),
                lr=lr,
                beta1=float(beta1),
                beta2=float(beta2),
                weight_decay=weight_decay,
                eps=eps,
            )
        through[key] = target

    def _apply_slice(
        self,
        param: torch.Tensor,
        key: tuple[int, int],
        grad: torch.Tensor,
        step: float,
        group: dict,
    ) -> None:
        layer, expert = key
        view = param[layer, expert]
        bucket = self._moments.setdefault(id(param), {})
        moments = bucket.get(key)
        if moments is None:
            moments = (torch.zeros_like(view), torch.zeros_like(view))
            bucket[key] = moments
        beta1, beta2 = group["betas"]
        adamw_inplace(
            view,
            grad,
            moments[0],
            moments[1],
            step,
            lr=float(group["lr"]),
            beta1=float(beta1),
            beta2=float(beta2),
            weight_decay=float(group["weight_decay"]),
            eps=float(group["eps"]),
        )
        self._ensure_through(param)[key] = int(step)

    def _dense_param(self, param: torch.Tensor, group: dict, beta1: float, beta2: float) -> None:
        state = self.state[param]
        if len(state) == 0:
            state["step"] = torch.tensor(0.0, dtype=torch.float32, device="cpu")
            state["exp_avg"] = torch.zeros_like(param)
            state["exp_avg_sq"] = torch.zeros_like(param)
        state["step"] += 1
        step = float(state["step"].item())
        grad = param.grad
        if grad is None:
            return
        if grad.device != param.device or grad.dtype != param.dtype:
            grad = grad.to(device=param.device, dtype=param.dtype)
        adamw_inplace(
            param,
            grad,
            state["exp_avg"],
            state["exp_avg_sq"],
            step,
            lr=float(group["lr"]),
            beta1=float(beta1),
            beta2=float(beta2),
            weight_decay=float(group["weight_decay"]),
            eps=float(group["eps"]),
        )


def catch_up_lora_slice(param: torch.Tensor, layer: int, expert: int) -> None:
    owner = getattr(param, "_sparse_adamw", None)
    if owner is None:
        return
    owner.catch_up_slice(param, layer, expert)
