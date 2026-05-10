import csv
import json
from datetime import datetime
from pathlib import Path

import torch

from loss_balancong_algorithms.algos import (
    GradNormWeighting,
    LearningRateAnnealingWeighting,
    ReLoBRaLoWeighting,
    SoftAdaptWeighting,
    VanillaPINNWeighting,
    HierarchicalOrthogonalPINNLossBalancer
)

BALANCER_CHOICES = (
    "grad-orth",
    "vanilla",
    "lra",
    "softadapt",
    "gradnorm",
    "relobralo",
)


def make_run_artifact_paths(save_dir, run_prefix):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = Path(save_dir) / f"{run_prefix}_{timestamp}"
    return {
        "csv": base_path.with_suffix(".csv"),
        "summary": base_path.with_name(base_path.name + "_summary.json"),
    }


class TrainingRunLogger:
    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.csv_path.open("w", newline="", encoding="utf-8")
        self._writer = None

    def log_epoch(self, metrics):
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(metrics.keys()))
            self._writer.writeheader()
        self._writer.writerow(metrics)
        self._file.flush()

    def close(self):
        self._file.close()


def save_run_summary(summary_path, summary):
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=True)


def create_balancer(args, device):
    if args.balancer == "grad-orth":
        return (
            HierarchicalOrthogonalPINNLossBalancer(
                L0_mse=1e-2,
                L0_ibc=1e-3,
                use_orthogonal_factor=True,
                orthogonal_floor=0.5,
            ), 
            None)
        # return None, None

    if args.balancer == "vanilla":
        return VanillaPINNWeighting(device=device), None

    if args.balancer == "lra":
        return (
            LearningRateAnnealingWeighting(
                primary_idx=2,
                alpha=args.lra_alpha,
                device=device,
            ),
            None,
        )

    if args.balancer == "softadapt":
        return (
            SoftAdaptWeighting(
                beta=args.softadapt_beta,
                use_relative=args.softadapt_use_relative,
                device=device,
            ),
            None,
        )

    if args.balancer == "gradnorm":
        balancer = GradNormWeighting(alpha=args.gradnorm_alpha).to(device)
        balancer_optimizer = torch.optim.Adam(
            balancer.parameters(),
            lr=args.gradnorm_lr,
        )
        return balancer, balancer_optimizer

    if args.balancer == "relobralo":
        return (
            ReLoBRaLoWeighting(
                alpha=args.relobralo_alpha,
                rho=args.relobralo_rho,
                temperature=args.relobralo_temperature,
                device=device,
            ),
            None,
        )

    raise ValueError(f"Unknown balancer: {args.balancer}")


def compute_balanced_loss(
    args,
    model,
    balancer,
    balancer_optimizer,
    loss_data,
    loss_bc_ic,
    loss_pde,
    device,
):
    losses = [loss_data, loss_bc_ic, loss_pde]

    if args.balancer == "grad-orth":
        w_data, w_bc_ic, w_pde, _ = balancer.compute_weights(
            model,
            loss_data,
            loss_bc_ic,
            loss_pde,
        )
        weights = torch.stack((w_data, w_bc_ic, w_pde)).to(device=device, dtype=loss_data.dtype)
        total_loss = weights[0] * loss_data + weights[1] * loss_bc_ic + weights[2] * loss_pde
        return total_loss, weights, None

    if args.balancer == "gradnorm":
        balancer_optimizer.zero_grad(set_to_none=True)
        gradnorm_loss = balancer.gradnorm_loss(
            losses,
            shared_params=model.parameters(),
        )
        gradnorm_loss.backward(retain_graph=True)
        balancer_optimizer.step()
        total_loss, weights = balancer.weighted_model_loss(losses)
        return total_loss, weights.detach().clone(), gradnorm_loss.detach()

    if args.balancer == "lra":
        total_loss, weights = balancer(losses, model_params=model.parameters())
        return total_loss, weights.detach().clone(), None

    total_loss, weights = balancer(losses)
    return total_loss, weights.detach().clone(), None
