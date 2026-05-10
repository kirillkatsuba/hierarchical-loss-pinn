import argparse
import os
import random
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
MPLCONFIG_DIR = PROJECT_DIR / ".mplconfig"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from tqdm import trange

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = PROJECT_DIR
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from diffusion_equation import compute_solution
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
        description="Train the test2 PINN with selectable loss balancing."
    )
    parser.add_argument(
        "--balancer",
        choices=BALANCER_CHOICES,
        default="grad-orth",
        help="Loss balancing strategy.",
    )
    parser.add_argument("--epochs", type=int, default=20_000, help="Training epochs.")
    parser.add_argument(
        "--n-collocation",
        type=int,
        default=100,
        help="Number of sampled simulation points N.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device, for example cpu, cuda, cuda:0, mps.",
    )
    parser.add_argument("--lr", type=float, default=3e-3, help="Initial learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Model hidden width.")
    parser.add_argument("--num-layers", type=int, default=8, help="Number of model layers.")
    parser.add_argument(
        "--grad-orth-kappa",
        type=float,
        default=10.0,
        help="Kappa parameter for gradient orthogonal balancing.",
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
        help="ReLoBRaLo temperature.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=250,
        help="Print training status every N epochs.",
    )
    parser.add_argument(
        "--lr-decay-every",
        type=int,
        default=5_000,
        help="Decay learning rate every N epochs. Use 0 to disable.",
    )
    parser.add_argument(
        "--lr-decay-factor",
        type=float,
        default=0.9,
        help="Learning rate decay multiplier.",
    )
    parser.add_argument(
        "--save-dir",
        default="weights_logs",
        help="Directory for checkpoints.",
    )
    parser.add_argument(
        "--generate-indices-if-missing",
        action="store_true",
        help="Generate sim_N.npy if it does not exist.",
    )
    parser.add_argument(
        "--plot-every",
        type=int,
        default=5_000,
        help="Save validation plots every N epochs. Use 0 to disable.",
    )
    parser.add_argument(
        "--plot-pwat",
        type=float,
        default=2.0,
        help="Validation-case pwat parameter used for saved plots.",
    )
    parser.add_argument(
        "--plot-poil",
        type=float,
        default=4.0,
        help="Validation-case poil parameter used for saved plots.",
    )
    parser.add_argument(
        "--plot-kwat",
        type=float,
        default=1.0,
        help="Validation-case kwat parameter used for saved plots.",
    )
    parser.add_argument(
        "--plot-koil",
        type=float,
        default=0.3,
        help="Validation-case koil parameter used for saved plots.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name):
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return torch.device(device_name)


class PINN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, num_layers=8, output_dim=7):
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

def load_index_file(base_dir, n_points, generate_if_missing):
    index_path = base_dir / f"sim_{n_points}.npy"
    if index_path.exists():
        return np.load(index_path)
    if generate_if_missing:
        indices = np.random.randint(0, 5000, n_points)
        np.save(index_path, indices)
        return indices
    raise FileNotFoundError(
        f"Missing {index_path.name}. Use --generate-indices-if-missing to create it."
    )


def prepare_training_data(base_dir, n_points, device, generate_if_missing):
    perm = np.load(base_dir / "perm.npy")
    nx0, nx1 = perm.shape
    perm = np.reshape(perm, (nx0, nx1, 1))

    dx0 = 1.0 / nx0
    dx1 = 1.0 / nx1
    dt = 3.0 / 5000.0

    sim_indices = load_index_file(base_dir, n_points, generate_if_missing)
    sim_data = np.loadtxt(base_dir / "data_5k_200225" / "sim_5000.txt", dtype=np.float32)[
        sim_indices
    ]
    x_list = np.loadtxt(base_dir / "data_5k_200225" / "x_5000.txt", dtype=np.float32)[
        sim_indices
    ]
    y_list = np.loadtxt(base_dir / "data_5k_200225" / "y_5000.txt", dtype=np.float32)[
        sim_indices
    ]
    t_list = np.loadtxt(base_dir / "data_5k_200225" / "t_5000.txt", dtype=np.float32)[
        sim_indices
    ]

    simulation_data = torch.tensor(sim_data, dtype=torch.float32, device=device)
    x = torch.tensor(x_list * dx0, dtype=torch.float32, device=device, requires_grad=True)
    y = torch.tensor(y_list * dx1, dtype=torch.float32, device=device, requires_grad=True)
    t = torch.tensor(t_list * dt, dtype=torch.float32, device=device, requires_grad=True)

    perm_vec = torch.tensor(
        perm[list(x_list.astype(int)), list(y_list.astype(int)), -1].astype(np.float32),
        dtype=torch.float32,
        device=device,
    )
    return perm, simulation_data, x, y, t, perm_vec


def build_points(t, x, y):
    points = torch.stack((t, x, y), dim=-1)
    pres0_x1 = torch.stack((t, torch.zeros_like(x), y), dim=-1)
    pres1_x1 = torch.stack((t, torch.ones_like(x), y), dim=-1)
    swat0_x1 = torch.stack((torch.zeros_like(t), x, y), dim=-1)
    soil0_x1 = torch.stack((torch.zeros_like(t), x, y), dim=-1)
    return points, pres0_x1, pres1_x1, swat0_x1, soil0_x1


def compute_losses(model, t, x, y, perm_vec, simulation_data):
    points, pres0_x1, pres1_x1, swat0_x1, soil0_x1 = build_points(t, x, y)
    model_res = model(points)

    press0 = model(pres0_x1)[:, 0]
    press1 = model(pres1_x1)[:, 0]
    soil0 = model(soil0_x1)[:, 1]
    swat0 = model(swat0_x1)[:, 2]

    one_vector = torch.ones_like(model_res[:, 0])

    dpres_dx = torch.autograd.grad(
        model_res[:, 0], x, one_vector, create_graph=True, allow_unused=False
    )[0]
    dpres_dy = torch.autograd.grad(
        model_res[:, 0], y, one_vector, create_graph=True, allow_unused=False
    )[0]

    r1 = (
        0.1
        * torch.autograd.grad(
            model_res[:, 2], t, one_vector, create_graph=True, allow_unused=False
        )[0]
        + torch.autograd.grad(
            model_res[:, 5], x, one_vector, create_graph=True, allow_unused=False
        )[0]
        + torch.autograd.grad(
            model_res[:, 6], y, one_vector, create_graph=True, allow_unused=False
        )[0]
    )
    r2 = (
        0.1
        * torch.autograd.grad(
            model_res[:, 1], t, one_vector, create_graph=True, allow_unused=False
        )[0]
        + torch.autograd.grad(
            model_res[:, 3], x, one_vector, create_graph=True, allow_unused=False
        )[0]
        + torch.autograd.grad(
            model_res[:, 4], y, one_vector, create_graph=True, allow_unused=False
        )[0]
    )
    r3_x1 = model_res[:, 5] + perm_vec * model_res[:, 2] ** 2 * dpres_dx
    r3_x2 = model_res[:, 6] + perm_vec * model_res[:, 2] ** 2 * dpres_dy
    r4_x1 = model_res[:, 3] + (1.0 / 3.0) * perm_vec * 0.1 * model_res[:, 2] ** 4 * dpres_dx
    r4_x2 = model_res[:, 4] + (1.0 / 3.0) * perm_vec * 0.1 * model_res[:, 2] ** 4 * dpres_dy
    r5 = model_res[:, 2] + model_res[:, 1] - one_vector

    loss_pde = torch.mean(r1**2 + r2**2 + r3_x1**2 + r3_x2**2 + r4_x1**2 + r4_x2**2 + r5**2)
    loss_ic = torch.mean((press0 - one_vector) ** 2 + press1**2 + (soil0 - one_vector) ** 2 + swat0**2)
    loss_data = torch.mean(
        (model_res[:, 0] - simulation_data[:, 0]) ** 2
        + (model_res[:, 2] - simulation_data[:, 1]) ** 2
        + (model_res[:, 1] - simulation_data[:, 2]) ** 2
    )
    return loss_pde, loss_ic, loss_data


def build_validation_reference(base_dir, pwat, poil, kwat, koil):
    perm_2d = np.load(base_dir / "perm.npy").astype(np.float32)
    nx0, nx1 = perm_2d.shape
    nx2 = 1
    perm = np.reshape(perm_2d, (nx0, nx1, nx2))
    poro = 0.1 + np.zeros((nx0, nx1, nx2), dtype=np.float32)

    dx0 = 1.0 / nx0
    dx1 = 1.0 / nx1
    dx2 = 1.0 / nx2
    dt = 0.15e-1
    niter = 100
    vr = 0.3
    cache_dir = base_dir / ".validation_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_name = (
        f"test2_pwat{pwat:g}_poil{poil:g}_kwat{kwat:g}_koil{koil:g}".replace(".", "p")
        + ".npz"
    )
    cache_path = cache_dir / cache_name

    if cache_path.exists():
        cached = np.load(cache_path)
        pres = cached["pres"]
        swat = cached["swat"]
        soil = cached["soil"]
    else:
        pres, swat, soil = compute_solution(
            perm,
            poro,
            dx0,
            dx1,
            dx2,
            dt * niter,
            niter,
            pwat,
            kwat,
            poil,
            koil,
            vr,
            pmin=0.0,
            pmax=1.0,
        )
        np.savez_compressed(cache_path, pres=pres, swat=swat, soil=soil)
    x_for_model = dx0 * torch.arange(nx0, dtype=torch.float32)
    y_for_model = dx1 * torch.arange(nx1, dtype=torch.float32)
    cartesian_points = torch.cartesian_prod(x_for_model, y_for_model)
    time_for_model = torch.full((nx0 * nx1,), dt * niter, dtype=torch.float32)
    return {
        "pres": pres,
        "swat": swat,
        "soil": soil,
        "time_for_model": time_for_model,
        "cartesian_points": cartesian_points,
        "nx0": nx0,
        "nx1": nx1,
        "nx2": nx2,
    }


def save_validation_plot(model, device, reference, save_path):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model_prediction = model(
            torch.stack(
                (
                    reference["time_for_model"],
                    reference["cartesian_points"][:, 0],
                    reference["cartesian_points"][:, 1],
                ),
                dim=-1,
            ).to(device)
        )
    if was_training:
        model.train()

    nx0 = reference["nx0"]
    nx1 = reference["nx1"]
    nx2 = reference["nx2"]
    pres = reference["pres"]
    swat = reference["swat"]
    model_prediction = model_prediction.detach().cpu().numpy()
    swat_pinn = np.squeeze(model_prediction[:, 2].reshape(nx0, nx1, nx2))
    pres_pinn = np.squeeze(model_prediction[:, 0].reshape(nx0, nx1, nx2))
    swat_sim = np.squeeze(swat[:, :, :, -1])
    pres_sim = np.squeeze(pres[:, :, :, -1])

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    im = axes[0, 0].imshow(swat_sim)
    axes[0, 0].set_title("Water saturation, sim")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    fig.colorbar(im, ax=axes[0, 0])

    im = axes[0, 1].imshow(swat_pinn)
    axes[0, 1].set_title("Water saturation, PINN")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("y")
    fig.colorbar(im, ax=axes[0, 1])

    im = axes[0, 2].imshow(pres_sim)
    axes[0, 2].set_title("Pressure, sim")
    axes[0, 2].set_xlabel("x")
    axes[0, 2].set_ylabel("y")
    fig.colorbar(im, ax=axes[0, 2])

    im = axes[1, 0].imshow(pres_pinn)
    axes[1, 0].set_title("Pressure, PINN")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("y")
    fig.colorbar(im, ax=axes[1, 0])

    axes[1, 1].set_title("Water saturation at y = 0")
    axes[1, 1].scatter(np.linspace(0.0, 1.0, nx0), swat[:, 0, 0, -1], label="Simulator")
    axes[1, 1].scatter(
        np.linspace(0.0, 1.0, nx0),
        swat_pinn[:, 0],
        label="PINN",
    )
    axes[1, 1].grid()
    axes[1, 1].set_xlabel("x")
    axes[1, 1].legend()

    axes[1, 2].set_title("Pressure at y = 0")
    axes[1, 2].scatter(np.linspace(0.0, 1.0, nx0), pres[:, 0, 0, -1], label="Simulator")
    axes[1, 2].scatter(
        np.linspace(0.0, 1.0, nx0),
        pres_pinn[:, 0],
        label="PINN",
    )
    axes[1, 2].grid()
    axes[1, 2].set_xlabel("x")
    axes[1, 2].legend()

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path

def train(args):
    set_seed(args.seed)
    device = resolve_device(args.device)
    base_dir = Path(__file__).resolve().parent
    save_dir = base_dir / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    _, simulation_data, x, y, t, perm_vec = prepare_training_data(
        base_dir=base_dir,
        n_points=args.n_collocation,
        device=device,
        generate_if_missing=args.generate_indices_if_missing,
    )

    model = PINN(
        input_dim=3,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        output_dim=7,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    balancer, balancer_optimizer = create_balancer(args, device)
    validation_reference = None
    validation_plot_dir = None
    if args.plot_every > 0:
        validation_plot_dir = save_dir / f"test2_validation_{args.balancer}_n{args.n_collocation}"
        validation_plot_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = save_dir / f"test2_pinn_{args.balancer}_n{args.n_collocation}.pth"
    best_loss = float("inf")
    best_epoch = -1
    artifact_paths = make_run_artifact_paths(
        save_dir, f"test2_metrics_{args.balancer}_n{args.n_collocation}"
    )
    logger = TrainingRunLogger(artifact_paths["csv"])
    train_start_time = time.perf_counter()

    print(
        f"Training test2 PINN: balancer={args.balancer}, epochs={args.epochs}, "
        f"N={args.n_collocation}, device={device}"
    )

    for epoch in trange(args.epochs):
        epoch_start_time = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        if t.grad is not None:
            t.grad = None
        if x.grad is not None:
            x.grad = None
        if y.grad is not None:
            y.grad = None

        loss_pde, loss_ic, loss_data = compute_losses(
            model=model,
            t=t,
            x=x,
            y=y,
            perm_vec=perm_vec,
            simulation_data=simulation_data,
        )
        raw_total_loss = loss_pde + loss_ic + loss_data
        balanced_loss, weights, gradnorm_loss = compute_balanced_loss(
            args=args,
            model=model,
            balancer=balancer,
            balancer_optimizer=balancer_optimizer,
            loss_data=loss_data,
            loss_bc_ic=loss_ic,
            loss_pde=loss_pde,
            device=device,
        )

        if gradnorm_loss is not None:
            optimizer.zero_grad(set_to_none=True)
        balanced_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        current_balanced = float(balanced_loss.detach().cpu().item())
        if current_balanced < best_loss:
            best_loss = current_balanced
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
                "balanced_loss": current_balanced,
                "loss_data": float(loss_data.detach().cpu().item()),
                "loss_bc_ic": float(loss_ic.detach().cpu().item()),
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
                f"balanced={current_balanced:.6e} "
                f"pde={loss_pde.detach().cpu().item():.6e} "
                f"bc_ic={loss_ic.detach().cpu().item():.6e} "
                f"data={loss_data.detach().cpu().item():.6e} "
                f"weights={weights.detach().cpu().numpy().tolist()}"
            )
            if gradnorm_loss is not None:
                log_line += f" gradnorm={gradnorm_loss.detach().cpu().item():.6e}"
            print(log_line)

        should_plot = args.plot_every > 0 and (epoch + 1) % args.plot_every == 0
        if should_plot and validation_reference is not None:
            plot_path = validation_plot_dir / f"epoch_{epoch + 1:06d}.png"
            save_validation_plot(model, device, validation_reference, plot_path)
            print(f"Saved validation plot: {plot_path}")
        elif should_plot:
            validation_reference = build_validation_reference(
                base_dir=base_dir,
                pwat=args.plot_pwat,
                poil=args.plot_poil,
                kwat=args.plot_kwat,
                koil=args.plot_koil,
            )
            plot_path = validation_plot_dir / f"epoch_{epoch + 1:06d}.png"
            save_validation_plot(model, device, validation_reference, plot_path)
            print(f"Saved validation plot: {plot_path}")

    final_checkpoint_path = checkpoint_path.with_name(
        checkpoint_path.stem + "_last" + checkpoint_path.suffix
    )
    torch.save(model.state_dict(), final_checkpoint_path)
    total_train_time_sec = time.perf_counter() - train_start_time
    logger.close()
    save_run_summary(
        artifact_paths["summary"],
        {
            "test_name": "test2",
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
            "validation_plot_dir": None if validation_plot_dir is None else str(validation_plot_dir),
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
