import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from loss_balancong_algorithms.runtime import (
    BALANCER_CHOICES,
    compute_balanced_loss,
    create_balancer,
    make_run_artifact_paths,
    save_run_summary,
    TrainingRunLogger,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the test1 PINN with selectable loss balancing."
    )
    parser.add_argument(
        "--balancer",
        choices=BALANCER_CHOICES,
        default="grad-orth",
        help="Loss balancing strategy.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20_000,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--n-collocation",
        type=int,
        default=50,
        help="Number of collocation/data points N.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device, for example cpu, cuda, cuda:0, mps.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-3,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=250,
        help="Print losses every N epochs.",
    )
    parser.add_argument(
        "--lr-decay-every",
        type=int,
        default=3_000,
        help="Decay learning rate every N epochs. Use 0 to disable.",
    )
    parser.add_argument(
        "--lr-decay-factor",
        type=float,
        default=0.9,
        help="Learning rate decay multiplier.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=32,
        help="Hidden width of the PINN.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of PINN layers used in the notebook architecture.",
    )
    parser.add_argument(
        "--grad-orth-kappa",
        type=float,
        default=3.0,
        help="Kappa parameter for compute_weights_grad_orthogonal_autograd.",
    )
    parser.add_argument(
        "--lra-alpha",
        type=float,
        default=0.9,
        help="EMA factor for Learning Rate Annealing.",
    )
    parser.add_argument(
        "--softadapt-beta",
        type=float,
        default=1.0,
        help="Beta parameter for SoftAdapt.",
    )
    parser.add_argument(
        "--softadapt-use-relative",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use relative loss changes in SoftAdapt.",
    )
    parser.add_argument(
        "--gradnorm-alpha",
        type=float,
        default=1.5,
        help="Alpha parameter for GradNorm.",
    )
    parser.add_argument(
        "--gradnorm-lr",
        type=float,
        default=1e-3,
        help="Learning rate for GradNorm weights.",
    )
    parser.add_argument(
        "--relobralo-alpha",
        type=float,
        default=0.999,
        help="ReLoBRaLo alpha.",
    )
    parser.add_argument(
        "--relobralo-rho",
        type=float,
        default=0.9999,
        help="ReLoBRaLo random lookback retention parameter.",
    )
    parser.add_argument(
        "--relobralo-temperature",
        type=float,
        default=0.1,
        help="ReLoBRaLo softmax temperature.",
    )
    parser.add_argument(
        "--save-dir",
        default="weights_logs",
        help="Directory for saving checkpoints.",
    )
    parser.add_argument(
        "--generate-data-if-missing",
        action="store_true",
        help="Generate x_N.pt and t_N.pt if they are missing.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def thermal_conductivity_equation(t, x):
    return (
        2.0
        + torch.exp(-4.0 * (torch.pi**2) * t) * torch.sin(2.0 * torch.pi * x)
        + torch.exp(-16.0 * (torch.pi**2) * t) * torch.cos(4.0 * torch.pi * x)
    )


class PINN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, num_layers=4, output_dim=1):
        super().__init__()

        self.U = nn.Linear(input_dim, hidden_dim)
        self.V = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        u = torch.tanh(self.U(x))
        v = torch.tanh(self.V(x))
        h = u * v

        for idx, layer in enumerate(self.hidden_layers):
            h_new = torch.relu(layer(h))
            if idx % 2 == 1 and idx > 0:
                h = h + h_new
            else:
                h = h_new

        return self.output_layer(h)

def resolve_device(device_name):
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return torch.device(device_name)


def load_or_generate_points(data_dir, n_points, device, generate_if_missing):
    x_path = data_dir / f"x_{n_points}.pt"
    t_path = data_dir / f"t_{n_points}.pt"

    if x_path.exists() and t_path.exists():
        x = torch.load(x_path, map_location="cpu")
        t = torch.load(t_path, map_location="cpu")
    elif generate_if_missing:
        x = torch.rand(n_points)
        t = 0.05 * torch.rand(n_points)
        torch.save(x, x_path)
        torch.save(t, t_path)
    else:
        raise FileNotFoundError(
            f"Could not find {x_path.name} and {t_path.name}. "
            "Use --generate-data-if-missing to create them."
        )

    x = x.clone().detach().to(device).requires_grad_(True)
    t = t.clone().detach().to(device).requires_grad_(True)
    return x, t


def build_training_points(time_physics, x_physics, device):
    n_points = x_physics.shape[0]
    points = torch.stack((time_physics, x_physics), dim=-1).to(device)
    initial_points = torch.stack((torch.zeros(n_points, device=device), x_physics), dim=-1)
    periodic_points_x0 = torch.stack((time_physics, torch.zeros(n_points, device=device)), dim=-1)
    periodic_points_x1 = torch.stack((time_physics, torch.ones(n_points, device=device)), dim=-1)
    return points, initial_points, periodic_points_x0, periodic_points_x1


def compute_losses(
    model,
    time_physics,
    x_physics,
    points_for_model,
    initial_points,
    periodic_points_x0,
    periodic_points_x1,
):
    data_prediction = model(points_for_model)
    initial_prediction = model(initial_points)
    periodic_prediction_0 = model(periodic_points_x0)
    periodic_prediction_1 = model(periodic_points_x1)

    target_data = thermal_conductivity_equation(time_physics, x_physics).view(-1, 1)
    target_initial = thermal_conductivity_equation(
        torch.zeros_like(time_physics), x_physics
    ).view(-1, 1)

    loss_data = torch.mean((target_data - data_prediction) ** 2)
    loss_bc_ic = torch.mean(
        (target_initial - initial_prediction) ** 2
        + (periodic_prediction_0 - periodic_prediction_1) ** 2
    )

    pde_prediction = model(points_for_model)
    dt = torch.autograd.grad(
        pde_prediction,
        time_physics,
        torch.ones_like(pde_prediction),
        create_graph=True,
        allow_unused=False,
    )[0]
    dx = torch.autograd.grad(
        pde_prediction,
        x_physics,
        torch.ones_like(pde_prediction),
        create_graph=True,
        allow_unused=False,
    )[0]
    dx2 = torch.autograd.grad(
        dx,
        x_physics,
        torch.ones_like(dx),
        create_graph=True,
        allow_unused=False,
    )[0]
    loss_pde = torch.mean((dt - dx2) ** 2)

    return loss_data, loss_bc_ic, loss_pde

def train(args):
    set_seed(args.seed)
    device = resolve_device(args.device)

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    save_dir = base_dir / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    x_physics, time_physics = load_or_generate_points(
        data_dir=data_dir,
        n_points=args.n_collocation,
        device=device,
        generate_if_missing=args.generate_data_if_missing,
    )
    points_for_model, initial_points, periodic_points_x0, periodic_points_x1 = (
        build_training_points(time_physics, x_physics, device)
    )

    model = PINN(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    balancer, balancer_optimizer = create_balancer(args, device)

    checkpoint_name = f"test1_pinn_{args.balancer}_n{args.n_collocation}.pth"
    checkpoint_path = save_dir / checkpoint_name
    best_loss = float("inf")
    best_epoch = -1
    artifact_paths = make_run_artifact_paths(
        save_dir, f"test1_metrics_{args.balancer}_n{args.n_collocation}"
    )
    logger = TrainingRunLogger(artifact_paths["csv"])
    train_start_time = time.perf_counter()

    print(
        f"Training test1 PINN: balancer={args.balancer}, epochs={args.epochs}, "
        f"N={args.n_collocation}, device={device}"
    )

    for epoch in trange(args.epochs):
        epoch_start_time = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        if time_physics.grad is not None:
            time_physics.grad = None
        if x_physics.grad is not None:
            x_physics.grad = None

        loss_data, loss_bc_ic, loss_pde = compute_losses(
            model=model,
            time_physics=time_physics,
            x_physics=x_physics,
            points_for_model=points_for_model,
            initial_points=initial_points,
            periodic_points_x0=periodic_points_x0,
            periodic_points_x1=periodic_points_x1,
        )
        raw_total_loss = loss_data + loss_bc_ic + loss_pde
        balanced_loss, weights, gradnorm_loss = compute_balanced_loss(
            args=args,
            model=model,
            balancer=balancer,
            balancer_optimizer=balancer_optimizer,
            loss_data=loss_data,
            loss_bc_ic=loss_bc_ic,
            loss_pde=loss_pde,
            device=device,
        )

        if gradnorm_loss is not None:
            optimizer.zero_grad(set_to_none=True)
        balanced_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        balanced_value = float(balanced_loss.detach().cpu().item())
        if balanced_value < best_loss:
            best_loss = balanced_value
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)

        if args.lr_decay_every > 0 and (epoch + 1) % args.lr_decay_every == 0:
            optimizer.param_groups[0]["lr"] *= args.lr_decay_factor

        epoch_time_sec = time.perf_counter() - epoch_start_time
        elapsed_time_sec = time.perf_counter() - train_start_time
        logger.log_epoch(
            {
                "epoch": epoch,
                "epoch_time_sec": epoch_time_sec,
                "elapsed_time_sec": elapsed_time_sec,
                "lr": optimizer.param_groups[0]["lr"],
                "raw_total_loss": float(raw_total_loss.detach().cpu().item()),
                "balanced_loss": balanced_value,
                "loss_data": float(loss_data.detach().cpu().item()),
                "loss_bc_ic": float(loss_bc_ic.detach().cpu().item()),
                "loss_pde": float(loss_pde.detach().cpu().item()),
                "weight_data": float(weights[0].detach().cpu().item()),
                "weight_bc_ic": float(weights[1].detach().cpu().item()),
                "weight_pde": float(weights[2].detach().cpu().item()),
                "gradnorm_loss": (
                    None if gradnorm_loss is None else float(gradnorm_loss.detach().cpu().item())
                ),
            }
        )

        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            log_line = (
                f"epoch={epoch:6d} "
                f"raw_total={raw_total_loss.detach().cpu().item():.6e} "
                f"balanced={balanced_value:.6e} "
                f"data={loss_data.detach().cpu().item():.6e} "
                f"bc_ic={loss_bc_ic.detach().cpu().item():.6e} "
                f"pde={loss_pde.detach().cpu().item():.6e} "
                f"weights={weights.detach().cpu().numpy().tolist()}"
            )
            if gradnorm_loss is not None:
                log_line += f" gradnorm={gradnorm_loss.detach().cpu().item():.6e}"
            print(log_line)

    final_checkpoint_path = checkpoint_path.with_name(
        checkpoint_path.stem + "_last" + checkpoint_path.suffix
    )
    torch.save(model.state_dict(), final_checkpoint_path)
    total_train_time_sec = time.perf_counter() - train_start_time
    logger.close()
    save_run_summary(
        artifact_paths["summary"],
        {
            "test_name": "test1",
            "balancer": args.balancer,
            "n_collocation": args.n_collocation,
            "epochs": args.epochs,
            "device": str(device),
            "total_train_time_sec": total_train_time_sec,
            "best_balanced_loss": best_loss,
            "best_epoch": best_epoch,
            "best_checkpoint": str(checkpoint_path),
            "last_checkpoint": str(final_checkpoint_path),
            "epoch_metrics_csv": str(artifact_paths["csv"]),
            "args": vars(args),
        },
    )

    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Last checkpoint: {final_checkpoint_path}")
    print(f"Metrics CSV: {artifact_paths['csv']}")
    print(f"Run summary: {artifact_paths['summary']}")


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
