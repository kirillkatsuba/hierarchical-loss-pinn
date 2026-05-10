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
import torch.nn.functional as F
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
from train_logic import compute_pde_residuals, get_batch_indices


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the test3 PINN with selectable loss balancing."
    )
    parser.add_argument(
        "--balancer",
        choices=BALANCER_CHOICES,
        default="grad-orth",
        help="Loss balancing strategy.",
    )
    parser.add_argument("--epochs", type=int, default=5_000, help="Training epochs.")
    parser.add_argument(
        "--n-collocation",
        type=int,
        default=500,
        help="Number of sampled simulation points N.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device, for example cpu, cuda, cuda:0, mps.",
    )
    parser.add_argument("--batch-size", type=int, default=150, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=3e-3, help="Initial learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Model hidden width.")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of model layers.")
    parser.add_argument(
        "--grad-orth-kappa",
        type=float,
        default=5.0,
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
        default=50,
        help="Print training status every N epochs.",
    )
    parser.add_argument(
        "--lr-decay-every",
        type=int,
        default=300,
        help="Decay learning rate every N epochs. Use 0 to disable.",
    )
    parser.add_argument(
        "--lr-decay-factor",
        type=float,
        default=0.8,
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


class ModifiedPINN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, num_layers=6, output_dim=7):
        super().__init__()

        self.U = nn.Linear(input_dim, hidden_dim)
        self.V = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        self.pressure_scale = nn.Parameter(torch.ones(1))
        self.saturation_scale = nn.Parameter(torch.ones(2))
        self.velocity_scale = nn.Parameter(torch.ones(4))
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
            h_new = torch.tanh(layer(h))
            if idx % 2 == 1 and idx > 0:
                h = h + h_new
            else:
                h = h_new

        out = self.output_layer(h)
        pressure = F.softplus(out[:, 0:1] * self.pressure_scale, beta=1, threshold=20)
        saturation_oil = torch.sigmoid(out[:, 1:2] * self.saturation_scale[0])
        saturation_wat = torch.sigmoid(out[:, 2:3] * self.saturation_scale[1])
        sat_sum = saturation_oil + saturation_wat + 1e-8
        saturation_oil = saturation_oil / sat_sum
        saturation_wat = saturation_wat / sat_sum
        velocities = torch.tanh(out[:, 3:7]) * self.velocity_scale
        return torch.cat([pressure, saturation_oil, saturation_wat, velocities], dim=1)

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
    perm = np.load(base_dir / "perm_3sigma.npy")
    nx0, nx1 = perm.shape
    perm = np.reshape(perm, (nx0, nx1, 1))

    dx0 = 1.0 / nx0
    dx1 = 1.0 / nx1
    dt = 3.0 / 5000.0

    sim_indices = load_index_file(base_dir, n_points, generate_if_missing)
    sim_data = np.loadtxt(base_dir / "data_5k_3sigma" / "sim_5000.txt", dtype=np.float32)[
        sim_indices
    ]
    x_list = np.loadtxt(base_dir / "data_5k_3sigma" / "x_5000.txt", dtype=np.float32)[
        sim_indices
    ]
    y_list = np.loadtxt(base_dir / "data_5k_3sigma" / "y_5000.txt", dtype=np.float32)[
        sim_indices
    ]
    t_list = np.loadtxt(base_dir / "data_5k_3sigma" / "t_5000.txt", dtype=np.float32)[
        sim_indices
    ]

    simulation_data = torch.tensor(sim_data, dtype=torch.float32, device=device)
    x = torch.tensor(x_list * dx0, dtype=torch.float32, device=device)
    y = torch.tensor(y_list * dx1, dtype=torch.float32, device=device)
    t = torch.tensor(t_list * dt, dtype=torch.float32, device=device)
    points = torch.stack((t, x, y), dim=-1)

    perm_vec = torch.tensor(
        perm[list(x_list.astype(int)), list(y_list.astype(int)), -1].astype(np.float32),
        dtype=torch.float32,
        device=device,
    )
    return simulation_data, points, t, x, y, perm_vec


def build_boundary_points(t, x, y):
    pres0_x1 = torch.stack((t, torch.zeros_like(x), y), dim=-1)
    pres1_x1 = torch.stack((t, torch.ones_like(x), y), dim=-1)
    swat0_x1 = torch.stack((torch.zeros_like(t), x, y), dim=-1)
    soil0_x1 = torch.stack((torch.zeros_like(t), x, y), dim=-1)
    return pres0_x1, pres1_x1, swat0_x1, soil0_x1


def build_validation_reference(base_dir, pwat, poil, kwat, koil):
    perm_2d = np.load(base_dir / "perm_3sigma.npy").astype(np.float32)
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
        f"test3_pwat{pwat:g}_poil{poil:g}_kwat{kwat:g}_koil{koil:g}".replace(".", "p")
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

    simulation_data, points, t, x, y, perm_vec = prepare_training_data(
        base_dir=base_dir,
        n_points=args.n_collocation,
        device=device,
        generate_if_missing=args.generate_indices_if_missing,
    )
    pres0_x1, pres1_x1, swat0_x1, soil0_x1 = build_boundary_points(t, x, y)

    model = ModifiedPINN(
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
        validation_plot_dir = save_dir / f"test3_validation_{args.balancer}_n{args.n_collocation}"
        validation_plot_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = save_dir / f"test3_pinn_{args.balancer}_n{args.n_collocation}.pth"
    best_dist = float("inf")
    best_epoch = -1
    artifact_paths = make_run_artifact_paths(
        save_dir, f"test3_metrics_{args.balancer}_n{args.n_collocation}"
    )
    logger = TrainingRunLogger(artifact_paths["csv"])
    train_start_time = time.perf_counter()

    print(
        f"Training test3 PINN: balancer={args.balancer}, epochs={args.epochs}, "
        f"N={args.n_collocation}, batch_size={args.batch_size}, device={device}"
    )

    for epoch in trange(args.epochs):
        epoch_start_time = time.perf_counter()
        epoch_weighted = 0.0
        epoch_pde = 0.0
        epoch_ic = 0.0
        epoch_data = 0.0
        last_weights = None
        last_gradnorm = None
        weight_data_sum = 0.0
        weight_bc_ic_sum = 0.0
        weight_pde_sum = 0.0
        gradnorm_sum = 0.0
        gradnorm_count = 0
        num_batches = 0

        for batch_indices in get_batch_indices(points.shape[0], args.batch_size, shuffle=True):
            optimizer.zero_grad(set_to_none=True)

            points_batch = points[batch_indices]
            perm_vec_batch = perm_vec[batch_indices]
            r1, r2, r3_x1, r3_x2, r4_x1, r4_x2, r5, _ = compute_pde_residuals(
                model, points_batch, perm_vec_batch
            )
            loss_pde = torch.mean(r1**2 + r2**2 + r3_x1**2 + r3_x2**2 + r4_x1**2 + r4_x2**2 + r5**2)

            bc_batch_size = min(
                max(args.batch_size // 4, 1),
                pres0_x1.shape[0],
                pres1_x1.shape[0],
                soil0_x1.shape[0],
                swat0_x1.shape[0],
            )
            bc_idx_pres0 = torch.randperm(pres0_x1.shape[0], device=device)[:bc_batch_size]
            bc_idx_pres1 = torch.randperm(pres1_x1.shape[0], device=device)[:bc_batch_size]
            bc_idx_soil0 = torch.randperm(soil0_x1.shape[0], device=device)[:bc_batch_size]
            bc_idx_swat0 = torch.randperm(swat0_x1.shape[0], device=device)[:bc_batch_size]

            press0 = model(pres0_x1[bc_idx_pres0])[:, 0]
            press1 = model(pres1_x1[bc_idx_pres1])[:, 0]
            soil0 = model(soil0_x1[bc_idx_soil0])[:, 1]
            swat0 = model(swat0_x1[bc_idx_swat0])[:, 2]
            loss_ic = torch.mean((press0 - 1.0) ** 2 + press1**2 + (soil0 - 1.0) ** 2 + swat0**2)

            data_batch_size = min(args.batch_size, simulation_data.shape[0])
            data_indices = torch.randperm(simulation_data.shape[0], device=device)[:data_batch_size]
            data_points = points[data_indices]
            sim_data_batch = simulation_data[data_indices]
            model_res_data = model(data_points)
            loss_data = torch.mean(
                (model_res_data[:, 0] - sim_data_batch[:, 0]) ** 2
                + (model_res_data[:, 2] - sim_data_batch[:, 1]) ** 2
                + (model_res_data[:, 1] - sim_data_batch[:, 2]) ** 2
            )

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

            epoch_weighted += balanced_loss.detach().cpu().item()
            epoch_pde += loss_pde.detach().cpu().item()
            epoch_ic += loss_ic.detach().cpu().item()
            epoch_data += loss_data.detach().cpu().item()
            last_weights = weights.detach().cpu().numpy().tolist()
            weight_data_sum += float(weights[0].detach().cpu().item())
            weight_bc_ic_sum += float(weights[1].detach().cpu().item())
            weight_pde_sum += float(weights[2].detach().cpu().item())
            last_gradnorm = (
                None if gradnorm_loss is None else gradnorm_loss.detach().cpu().item()
            )
            if last_gradnorm is not None:
                gradnorm_sum += last_gradnorm
                gradnorm_count += 1
            num_batches += 1

        epoch_weighted /= num_batches
        epoch_pde /= num_batches
        epoch_ic /= num_batches
        epoch_data /= num_batches
        dist = epoch_pde + epoch_ic + epoch_data
        avg_weight_data = weight_data_sum / num_batches
        avg_weight_bc_ic = weight_bc_ic_sum / num_batches
        avg_weight_pde = weight_pde_sum / num_batches
        avg_gradnorm = None if gradnorm_count == 0 else gradnorm_sum / gradnorm_count

        if dist < best_dist:
            best_dist = dist
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
                "raw_total_loss": dist,
                "balanced_loss": epoch_weighted,
                "loss_data": epoch_data,
                "loss_bc_ic": epoch_ic,
                "loss_pde": epoch_pde,
                "weight_data": avg_weight_data,
                "weight_bc_ic": avg_weight_bc_ic,
                "weight_pde": avg_weight_pde,
                "gradnorm_loss": avg_gradnorm,
            }
        )

        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            log_line = (
                f"epoch={epoch:6d} "
                f"raw_total={dist:.6e} "
                f"balanced={epoch_weighted:.6e} "
                f"pde={epoch_pde:.6e} "
                f"bc_ic={epoch_ic:.6e} "
                f"data={epoch_data:.6e} "
                f"weights={last_weights}"
            )
            if last_gradnorm is not None:
                log_line += f" gradnorm={last_gradnorm:.6e}"
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
            "test_name": "test3",
            "balancer": args.balancer,
            "n_collocation": args.n_collocation,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "device": str(device),
            "total_train_time_sec": total_train_time_sec,
            "best_raw_total_loss": best_dist,
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
