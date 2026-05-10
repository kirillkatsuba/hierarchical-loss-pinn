import torch
import torch.nn as nn
from typing import Sequence, Iterable, Dict, Iterable, List, Optional, Tuple
import math


# ============================================================
# Helpers
# ============================================================

def _trainable_params(params):
    return [p for p in params if p.requires_grad]


def _flatten_grads(grads):
    valid = [g.reshape(-1) for g in grads if g is not None]
    if len(valid) == 0:
        return None
    return torch.cat(valid)


def _grad_flat(loss, params, retain_graph=True, create_graph=False):
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=True,
    )
    return _flatten_grads(grads)


def _detach_losses(losses, eps=1e-12):
    return torch.stack([L.detach().clamp_min(eps) for L in losses])

# ============================================================
# 0. Hierarchical loss 
# ============================================================

def _trainable_parameters(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def flat_gradient_from_loss(
    model: torch.nn.Module,
    loss: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Returns a detached flattened gradient vector d(loss)/d(theta).

    Important:
    - Call this before loss.backward().
    - Uses retain_graph=True because several gradient vectors are computed
      from the same forward graph.
    - Gradients are detached because loss weights should use stop-gradient logic.
    """
    params = _trainable_parameters(model)

    if len(params) == 0:
        raise ValueError("Model has no trainable parameters.")

    total_numel = sum(p.numel() for p in params)

    if not loss.requires_grad:
        device = params[0].device
        dtype = params[0].dtype
        return torch.zeros(total_numel, device=device, dtype=dtype)

    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )

    flat_parts = []
    for param, grad in zip(params, grads):
        if grad is None:
            flat_parts.append(torch.zeros_like(param).reshape(-1))
        else:
            flat_parts.append(grad.detach().reshape(-1))

    return torch.cat(flat_parts)


def orthogonal_residual(
    vector: torch.Tensor,
    basis: Iterable[torch.Tensor],
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Removes from `vector` its projections onto all vectors in `basis`.
    This is a simple Gram-Schmidt-style residual.
    """
    residual = vector

    for b in basis:
        b = b.detach()
        denom = torch.dot(b, b) + eps
        residual = residual - torch.dot(residual, b) / denom * b

    return residual


def orthogonal_fraction(
    vector: torch.Tensor,
    basis: Iterable[torch.Tensor],
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Returns ||orthogonal component of vector|| / ||vector||.

    Value is in [0, 1]:
    - close to 0: vector mostly lies in the span of previous gradients;
    - close to 1: vector contributes a new optimization direction.
    """
    vector = vector.detach()
    residual = orthogonal_residual(vector, basis, eps=eps)

    frac = torch.linalg.norm(residual) / (torch.linalg.norm(vector) + eps)
    return torch.clamp(frac, 0.0, 1.0)


class HierarchicalOrthogonalPINNLossBalancer:
    """
    Hierarchical loss balancer for PINNs.

    Loss hierarchy:
        1. data / MSE loss
        2. initial-boundary condition loss
        3. PDE residual loss

    Base hierarchical gates:
        a_mse = 1
        a_ibc = exp(-L_mse / L0_mse)
        a_pde = exp(-max(L_mse / L0_mse, L_ibc / L0_ibc))

    Optional orthogonal modulation:
        a_ibc *= orth_factor(g_ibc | g_mse)
        a_pde *= orth_factor(g_pde | g_mse, g_ibc)

    Recommended starting values after loss normalization:
        L0_mse ~ 1e-3
        L0_ibc ~ 1e-4

    If losses are not normalized, estimate L0_mse and L0_ibc from warm-up
    loss histories using `estimate_target_from_history`.
    """

    def __init__(
        self,
        L0_mse: float,
        L0_ibc: float,
        use_orthogonal_factor: bool = True,
        orthogonal_floor: float = 0.5,
        eps: float = 1e-12,
        exp_clip: float = 60.0,
    ):
        if L0_mse <= 0:
            raise ValueError("L0_mse must be positive.")
        if L0_ibc <= 0:
            raise ValueError("L0_ibc must be positive.")
        if not (0.0 <= orthogonal_floor <= 1.0):
            raise ValueError("orthogonal_floor must be in [0, 1].")

        self.L0_mse = float(L0_mse)
        self.L0_ibc = float(L0_ibc)
        self.use_orthogonal_factor = bool(use_orthogonal_factor)
        self.orthogonal_floor = float(orthogonal_floor)
        self.eps = float(eps)
        self.exp_clip = float(exp_clip)

    def _hierarchical_gates(
        self,
        loss_mse: torch.Tensor,
        loss_ibc: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = loss_mse.device
        dtype = loss_mse.dtype

        L0_mse = torch.tensor(self.L0_mse, device=device, dtype=dtype)
        L0_ibc = torch.tensor(self.L0_ibc, device=device, dtype=dtype)

        # Stop-gradient: weights depend on detached loss values.
        lmse_ratio = loss_mse.detach() / (L0_mse + self.eps)
        libc_ratio = loss_ibc.detach() / (L0_ibc + self.eps)

        lmse_ratio = torch.clamp(lmse_ratio, min=0.0, max=self.exp_clip)
        libc_ratio = torch.clamp(libc_ratio, min=0.0, max=self.exp_clip)

        a_mse = torch.ones((), device=device, dtype=dtype)
        a_ibc = torch.exp(-lmse_ratio)
        a_pde = torch.exp(-torch.maximum(lmse_ratio, libc_ratio))

        return a_mse, a_ibc, a_pde

    def _apply_orthogonal_factors(
        self,
        model: torch.nn.Module,
        loss_mse: torch.Tensor,
        loss_ibc: torch.Tensor,
        loss_pde: torch.Tensor,
        a_ibc: torch.Tensor,
        a_pde: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        g_mse = flat_gradient_from_loss(model, loss_mse, eps=self.eps)
        g_ibc = flat_gradient_from_loss(model, loss_ibc, eps=self.eps)
        g_pde = flat_gradient_from_loss(model, loss_pde, eps=self.eps)

        frac_ibc = orthogonal_fraction(
            g_ibc,
            basis=[g_mse],
            eps=self.eps,
        )

        frac_pde = orthogonal_fraction(
            g_pde,
            basis=[g_mse, g_ibc],
            eps=self.eps,
        )

        # Floor prevents complete shutdown of a component.
        factor_ibc = self.orthogonal_floor + (1.0 - self.orthogonal_floor) * frac_ibc
        factor_pde = self.orthogonal_floor + (1.0 - self.orthogonal_floor) * frac_pde

        a_ibc = a_ibc * factor_ibc.to(device=a_ibc.device, dtype=a_ibc.dtype)
        a_pde = a_pde * factor_pde.to(device=a_pde.device, dtype=a_pde.dtype)

        diagnostics = {
            "orth_frac_ibc": float(frac_ibc.detach().cpu()),
            "orth_frac_pde": float(frac_pde.detach().cpu()),
            "orth_factor_ibc": float(factor_ibc.detach().cpu()),
            "orth_factor_pde": float(factor_pde.detach().cpu()),
        }

        return a_ibc, a_pde, diagnostics

    def compute_weights(
        self,
        model: torch.nn.Module,
        loss_mse: torch.Tensor,
        loss_ibc: torch.Tensor,
        loss_pde: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Returns:
            w_mse, w_ibc, w_pde, diagnostics
        """
        a_mse, a_ibc, a_pde = self._hierarchical_gates(loss_mse, loss_ibc)

        diagnostics = {}

        if self.use_orthogonal_factor:
            a_ibc, a_pde, orth_diag = self._apply_orthogonal_factors(
                model=model,
                loss_mse=loss_mse,
                loss_ibc=loss_ibc,
                loss_pde=loss_pde,
                a_ibc=a_ibc,
                a_pde=a_pde,
            )
            diagnostics.update(orth_diag)

        weight_sum = a_mse + a_ibc + a_pde + self.eps

        w_mse = (a_mse / weight_sum).detach()
        w_ibc = (a_ibc / weight_sum).detach()
        w_pde = (a_pde / weight_sum).detach()

        diagnostics.update(
            {
                "a_mse": float(a_mse.detach().cpu()),
                "a_ibc": float(a_ibc.detach().cpu()),
                "a_pde": float(a_pde.detach().cpu()),
                "w_mse": float(w_mse.detach().cpu()),
                "w_ibc": float(w_ibc.detach().cpu()),
                "w_pde": float(w_pde.detach().cpu()),
            }
        )

        return w_mse, w_ibc, w_pde, diagnostics

    def __call__(
        self,
        model: torch.nn.Module,
        loss_mse: torch.Tensor,
        loss_ibc: torch.Tensor,
        loss_pde: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns:
            total_loss, diagnostics
        """
        w_mse, w_ibc, w_pde, diagnostics = self.compute_weights(
            model=model,
            loss_mse=loss_mse,
            loss_ibc=loss_ibc,
            loss_pde=loss_pde,
        )

        total_loss = w_mse * loss_mse + w_ibc * loss_ibc + w_pde * loss_pde

        return total_loss, diagnostics


def estimate_target_from_history(
    loss_history: List[float],
    tail_fraction: float = 0.2,
    scale: float = 2.0,
    min_value: float = 1e-12,
) -> float:
    """
    Estimates L0 from a warm-up loss history.

    Use:
        L0_mse = estimate_target_from_history(mse_warmup_history)
        L0_ibc = estimate_target_from_history(ibc_warmup_history)

    Interpretation:
        L0 is the loss level at which the next hierarchy level becomes active.
    """
    if len(loss_history) == 0:
        raise ValueError("loss_history is empty.")

    n_tail = max(1, int(len(loss_history) * tail_fraction))
    tail_values = torch.tensor(loss_history[-n_tail:], dtype=torch.float64)

    median_tail = torch.median(tail_values).item()
    return max(scale * median_tail, min_value)


# ============================================================
# 1. Vanilla PINN
# ============================================================

class VanillaPINNWeighting:
    """
    losses = [loss_data, loss_bc_ic, loss_pde]
    weights = [1, 1, 1]
    """

    def __init__(self, device=None):
        self.n_terms = 3
        self.device = device
        self.weights = None

    def __call__(self, losses: Sequence[torch.Tensor]):
        if self.weights is None:
            device = self.device or losses[0].device
            self.weights = torch.ones(self.n_terms, device=device)

        total_loss = sum(w.detach() * L for w, L in zip(self.weights, losses))
        return total_loss, self.weights.detach().clone()


# ============================================================
# 2. Learning Rate Annealing
# ============================================================

class LearningRateAnnealingWeighting:
    """
    losses = [loss_data, loss_bc_ic, loss_pde]

    loss_pde используется как reference loss.
    Его вес фиксирован: lambda_pde = 1.

    Остальные веса обновляются как:

        lambda_i_hat = max(|grad loss_pde|) / mean(|grad loss_i|)

        lambda_i = alpha * lambda_i_old + (1 - alpha) * lambda_i_hat
    """

    def __init__(
        self,
        primary_idx: int = 2,   # loss_pde
        alpha: float = 0.9,
        eps: float = 1e-12,
        device=None,
    ):
        self.n_terms = 3
        self.primary_idx = primary_idx
        self.alpha = alpha
        self.eps = eps
        self.device = device
        self.weights = None

    def __call__(
        self,
        losses: Sequence[torch.Tensor],
        model_params: Iterable[torch.nn.Parameter],
    ):
        params = _trainable_params(model_params)

        if self.weights is None:
            device = self.device or losses[0].device
            self.weights = torch.ones(self.n_terms, device=device)

        loss_ref = losses[self.primary_idx]

        g_ref = _grad_flat(
            loss_ref,
            params,
            retain_graph=True,
            create_graph=False,
        )

        if g_ref is None:
            raise RuntimeError("loss_pde does not have gradients.")

        ref_max = g_ref.detach().abs().max()

        new_weights = self.weights.clone()

        for i, loss_i in enumerate(losses):
            if i == self.primary_idx:
                new_weights[i] = 1.0
                continue

            g_i = _grad_flat(
                loss_i,
                params,
                retain_graph=True,
                create_graph=False,
            )

            if g_i is None:
                continue

            grad_mean = g_i.detach().abs().mean()
            lambda_hat = ref_max / (grad_mean + self.eps)

            new_weights[i] = (
                self.alpha * self.weights[i]
                + (1.0 - self.alpha) * lambda_hat
            )

        self.weights = new_weights.detach()

        total_loss = sum(w.detach() * L for w, L in zip(self.weights, losses))
        return total_loss, self.weights.detach().clone()


# ============================================================
# 3. SoftAdapt
# ============================================================

class SoftAdaptWeighting:
    """
    losses = [loss_data, loss_bc_ic, loss_pde]

    score_i = (L_i(t) - L_i(t-1)) / (L_i(t-1) + eps)

    lambda_i = 3 * softmax(beta * score_i)
    """

    def __init__(
        self,
        beta: float = 1.0,
        use_relative: bool = True,
        eps: float = 1e-12,
        device=None,
    ):
        self.n_terms = 3
        self.beta = beta
        self.use_relative = use_relative
        self.eps = eps
        self.device = device

        self.prev_losses = None
        self.weights = None

    @torch.no_grad()
    def update_weights(self, losses):
        L = _detach_losses(losses, self.eps)

        if self.weights is None:
            device = self.device or L.device
            self.weights = torch.ones(self.n_terms, device=device)

        if self.prev_losses is None:
            self.prev_losses = L.clone()
            return self.weights.clone()

        if self.use_relative:
            scores = (L - self.prev_losses) / (self.prev_losses + self.eps)
        else:
            scores = L - self.prev_losses

        logits = self.beta * scores
        logits = logits - logits.max()

        self.weights = self.n_terms * torch.softmax(logits, dim=0)
        self.prev_losses = L.clone()

        return self.weights.clone()

    def __call__(self, losses: Sequence[torch.Tensor]):
        weights = self.update_weights(losses)
        total_loss = sum(w.detach() * L for w, L in zip(weights, losses))
        return total_loss, weights.detach().clone()


# ============================================================
# 4. GradNorm
# ============================================================

class GradNormWeighting(nn.Module):
    """
    losses = [loss_data, loss_bc_ic, loss_pde]

    Использует обучаемые веса lambda_i.

    Нужны два optimizer'а:
        model_optimizer
        weight_optimizer
    """

    def __init__(
        self,
        alpha: float = 1.5,
        eps: float = 1e-12,
    ):
        super().__init__()

        self.n_terms = 3
        self.alpha = alpha
        self.eps = eps

        self.logits = nn.Parameter(torch.zeros(self.n_terms))

        self.register_buffer("initial_losses", torch.zeros(self.n_terms))
        self.initialized = False

    def weights(self):
        return self.n_terms * torch.softmax(self.logits, dim=0)

    @torch.no_grad()
    def initialize_if_needed(self, losses):
        if not self.initialized:
            L0 = _detach_losses(losses, self.eps).to(self.logits.device)
            self.initial_losses.copy_(L0)
            self.initialized = True

    def weighted_model_loss(self, losses):
        weights = self.weights()
        total_loss = sum(w.detach() * L for w, L in zip(weights, losses))
        return total_loss, weights.detach().clone()

    def gradnorm_loss(
        self,
        losses: Sequence[torch.Tensor],
        shared_params: Iterable[torch.nn.Parameter],
    ):
        self.initialize_if_needed(losses)

        params = _trainable_params(shared_params)

        if len(params) == 0:
            raise RuntimeError("No trainable model parameters were provided.")

        weights = self.weights()

        grad_norms = []

        for w_i, loss_i in zip(weights, losses):
            g_i = _grad_flat(
                w_i * loss_i,
                params,
                retain_graph=True,
                create_graph=True,
            )

            if g_i is None:
                grad_norms.append(torch.zeros((), device=self.logits.device))
            else:
                grad_norms.append(torch.norm(g_i, p=2))

        grad_norms = torch.stack(grad_norms)

        with torch.no_grad():
            current_losses = _detach_losses(losses, self.eps).to(self.logits.device)

            relative_losses = current_losses / self.initial_losses.clamp_min(self.eps)

            inverse_train_rates = relative_losses / relative_losses.mean().clamp_min(self.eps)

            mean_grad_norm = grad_norms.detach().mean()

            target_grad_norms = mean_grad_norm * inverse_train_rates.pow(self.alpha)

        loss_gradnorm = torch.sum(torch.abs(grad_norms - target_grad_norms))

        return loss_gradnorm

    def __call__(self, losses):
        return self.weighted_model_loss(losses)


# ============================================================
# 5. ReLoBRaLo
# ============================================================

class ReLoBRaLoWeighting:
    """
    losses = [loss_data, loss_bc_ic, loss_pde]

    Relative Loss Balancing with Random Lookback.

    lambda_i = 3 * softmax(L_i(t) / (temperature * L_i(t_ref)))
    """

    def __init__(
        self,
        alpha: float = 0.999,
        rho: float = 0.9999,
        temperature: float = 0.1,
        eps: float = 1e-12,
        device=None,
    ):
        self.n_terms = 3
        self.alpha = alpha
        self.rho = rho
        self.temperature = temperature
        self.eps = eps
        self.device = device

        self.L0 = None
        self.Lprev = None
        self.weights = None

    @torch.no_grad()
    def _balanced_weights(self, L, L_ref):
        ratios = L / (self.temperature * L_ref.clamp_min(self.eps))
        ratios = ratios - ratios.max()
        return self.n_terms * torch.softmax(ratios, dim=0)

    @torch.no_grad()
    def update_weights(self, losses):
        L = _detach_losses(losses, self.eps)

        if self.weights is None:
            device = self.device or L.device
            self.weights = torch.ones(self.n_terms, device=device)

        if self.L0 is None:
            self.L0 = L.clone()
            self.Lprev = L.clone()
            return self.weights.clone()

        lambda_prev = self._balanced_weights(L, self.Lprev)
        lambda_zero = self._balanced_weights(L, self.L0)

        keep_history = torch.rand((), device=L.device) < self.rho
        rho_t = 1.0 if keep_history else 0.0

        lambda_hist = rho_t * self.weights + (1.0 - rho_t) * lambda_zero

        self.weights = (
            self.alpha * lambda_hist
            + (1.0 - self.alpha) * lambda_prev
        )

        self.Lprev = L.clone()

        return self.weights.clone()

    def __call__(self, losses: Sequence[torch.Tensor]):
        weights = self.update_weights(losses)
        total_loss = sum(w.detach() * L for w, L in zip(weights, losses))
        return total_loss, weights.detach().clone()
    

# примеры запуска
# device = "cuda" if torch.cuda.is_available() else "cpu"

# model = PINN().to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# # Выбери один:
# balancer = VanillaPINNWeighting()
# # balancer = SoftAdaptWeighting(beta=1.0, use_relative=True)
# # balancer = ReLoBRaLoWeighting(alpha=0.999, rho=0.9999, temperature=0.1)

# for step in range(50_000):
#     optimizer.zero_grad(set_to_none=True)

#     loss_data = compute_loss_data(model)
#     loss_bc_ic = compute_loss_bc_ic(model)
#     loss_pde = compute_loss_pde(model)

#     losses = [loss_data, loss_bc_ic, loss_pde]

#     total_loss, lambdas = balancer(losses)

#     total_loss.backward()
#     optimizer.step()

#     if step % 1000 == 0:
#         print(
#             step,
#             "total_loss:", float(total_loss.detach().cpu()),
#             "lambdas:", lambdas.detach().cpu().numpy(),
#             "loss_data:", float(loss_data.detach().cpu()),
#             "loss_bc_ic:", float(loss_bc_ic.detach().cpu()),
#             "loss_pde:", float(loss_pde.detach().cpu()),
#         )

# device = "cuda" if torch.cuda.is_available() else "cpu"

# model = PINN().to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# balancer = LearningRateAnnealingWeighting(
#     primary_idx=2,   # loss_pde
#     alpha=0.9,
# )

# for step in range(50_000):
#     optimizer.zero_grad(set_to_none=True)

#     loss_data = compute_loss_data(model)
#     loss_bc_ic = compute_loss_bc_ic(model)
#     loss_pde = compute_loss_pde(model)

#     losses = [loss_data, loss_bc_ic, loss_pde]

#     total_loss, lambdas = balancer(
#         losses,
#         model_params=model.parameters(),
#     )

#     total_loss.backward()
#     optimizer.step()

#     if step % 1000 == 0:
#         print(
#             step,
#             "total_loss:", float(total_loss.detach().cpu()),
#             "lambdas:", lambdas.detach().cpu().numpy(),
#         )

# device = "cuda" if torch.cuda.is_available() else "cpu"

# model = PINN().to(device)

# model_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# balancer = GradNormWeighting(
#     alpha=1.5,
# ).to(device)

# weight_optimizer = torch.optim.Adam(
#     balancer.parameters(),
#     lr=1e-3,
# )

# for step in range(50_000):
#     loss_data = compute_loss_data(model)
#     loss_bc_ic = compute_loss_bc_ic(model)
#     loss_pde = compute_loss_pde(model)

#     losses = [loss_data, loss_bc_ic, loss_pde]

#     # 1. Update GradNorm weights
#     weight_optimizer.zero_grad(set_to_none=True)

#     loss_gradnorm = balancer.gradnorm_loss(
#         losses,
#         shared_params=model.parameters(),
#     )

#     loss_gradnorm.backward(retain_graph=True)
#     weight_optimizer.step()

#     # 2. Update PINN parameters
#     model_optimizer.zero_grad(set_to_none=True)

#     total_loss, lambdas = balancer.weighted_model_loss(losses)

#     total_loss.backward()
#     model_optimizer.step()

#     if step % 1000 == 0:
#         print(
#             step,
#             "total_loss:", float(total_loss.detach().cpu()),
#             "loss_gradnorm:", float(loss_gradnorm.detach().cpu()),
#             "lambdas:", lambdas.detach().cpu().numpy(),
#         )

# Порядок весов везде одинаковый:
# lambdas[0] -> loss_data
# lambdas[1] -> loss_bc_ic
# lambdas[2] -> loss_pde