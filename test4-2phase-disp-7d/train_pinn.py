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

ROOT_DIR = PROJECT_DIR
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
from train_logic import compute_pde_residuals, get_plot_data, plot_validation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the test4 PINN with selectable loss balancing."
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
        help="Number of selected training points N.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device, for example cpu, cuda, cuda:0, mps.",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Model hidden width.")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of model layers.")
    parser.add_argument(
        "--grad-orth-kappa",
        type=float,
        default=8.0,
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
        default=20,
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
        default=0.9,
        help="Learning rate decay multiplier.",
    )
    parser.add_argument(
        "--save-dir",
        default="weights_logs",
        help="Directory for checkpoints.",
    )
    parser.add_argument(
        "--raw-sim-file",
        default="data-100-new-two-sigma/sim_100.npy",
        help="Path to the raw simulation array relative to test4 directory.",
    )
    parser.add_argument(
        "--generate-indices-if-missing",
        action="store_true",
        help="Generate train-indexes-2sigma-N.npy if it does not exist.",
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
        default=1.5,
        help="Validation-case pwat parameter used for saved plots.",
    )
    parser.add_argument(
        "--plot-poil",
        type=float,
        default=2.0,
        help="Validation-case poil parameter used for saved plots.",
    )
    parser.add_argument(
        "--plot-kwat",
        type=float,
        default=1.5,
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
    def __init__(self, input_dim=7, hidden_dim=32, num_layers=4, output_dim=7):
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
        pressure = torch.sigmoid(out[:, 0:1] * self.pressure_scale)
        saturation_oil = torch.sigmoid(out[:, 1:2] * self.saturation_scale[0])
        saturation_wat = torch.sigmoid(out[:, 2:3] * self.saturation_scale[1])
        sat_sum = saturation_oil + saturation_wat + 1e-8
        saturation_oil = saturation_oil / sat_sum
        saturation_wat = saturation_wat / sat_sum
        velocities = torch.tanh(out[:, 3:7]) * self.velocity_scale
        return torch.cat([pressure, saturation_oil, saturation_wat, velocities], dim=1)

def prepare_candidate_points(base_dir, raw_sim_file):
    sim_path = base_dir / raw_sim_file
    if not sim_path.exists():
        raise FileNotFoundError(
            f"Missing raw simulation file: {sim_path}. "
            "The notebook expects this file, but it is not present in the repository."
        )

    sim_data = np.load(sim_path).astype(np.float32)
    pwat_list = np.load(base_dir / "data-100-new-two-sigma" / "pwat_100.npy").astype(np.float32)
    poil_list = np.load(base_dir / "data-100-new-two-sigma" / "poil_100.npy").astype(np.float32)
    kwat_list = np.load(base_dir / "data-100-new-two-sigma" / "kwat_100.npy").astype(np.float32)
    koil_list = np.load(base_dir / "data-100-new-two-sigma" / "koil_100.npy").astype(np.float32)

    niter = 100
    nx = 64
    ny = 64
    dt = 3.0 / niter
    dx = 1.0 / nx
    dy = 1.0 / ny

    indexes_high = np.argwhere(sim_data[:, :, :, :, :, 1] > 0.15)
    indexes_low = np.argwhere(sim_data[:, :, :, :, :, 1] < 0.1)
    indexes_low = indexes_low[
        np.random.randint(0, len(indexes_low), max(len(indexes_high) // 2, 1))
    ]
    indexes = np.vstack([indexes_low, indexes_high])

    simulation_data = np.zeros((indexes.shape[0], 3), dtype=np.float32)
    for idx, row in enumerate(indexes):
        simulation_data[idx] = sim_data[row[0], row[1], row[2], row[3], row[4], :3]

    x_idx = indexes[:, 0]
    y_idx = indexes[:, 1]
    x_list = x_idx.astype(np.float32) * dx
    y_list = y_idx.astype(np.float32) * dy
    t_list = indexes[:, 3].astype(np.float32) * dt
    pwat_rand = pwat_list[indexes[:, -1]]
    poil_rand = poil_list[indexes[:, -1]]
    kwat_rand = kwat_list[indexes[:, -1]]
    koil_rand = koil_list[indexes[:, -1]]


    return simulation_data, x_idx, y_idx, x_list, y_list, t_list, pwat_rand, poil_rand, kwat_rand, koil_rand


def load_training_indices(base_dir, n_points, candidate_count, generate_if_missing):
    index_path = base_dir / f"train-indexes-2sigma-{n_points}.npy"
    if index_path.exists():
        return np.load(index_path)
    if generate_if_missing:
        indices = np.random.randint(0, candidate_count, n_points)
        np.save(index_path, indices)
        return indices
    raise FileNotFoundError(
        f"Missing {index_path.name}. Use --generate-indices-if-missing to create it."
    )


def prepare_training_data(base_dir, n_points, device, raw_sim_file, generate_if_missing):
    perm = np.load(base_dir / "perm_2.npy")
    nx0, nx1 = perm.shape
    perm = np.reshape(perm, (nx0, nx1, 1))

    (
        simulation_data,
        x_idx,
        y_idx,
        x_list,
        y_list,
        t_list,
        pwat_rand,
        poil_rand,
        kwat_rand,
        koil_rand,
    ) = prepare_candidate_points(base_dir, raw_sim_file)

    train_indices = load_training_indices(
        base_dir=base_dir,
        n_points=n_points,
        candidate_count=simulation_data.shape[0],
        generate_if_missing=generate_if_missing,
    )

    simulation_data_train = torch.tensor(
        simulation_data[train_indices], dtype=torch.float32, device=device
    )

    x = torch.tensor(x_list[train_indices], dtype=torch.float32, device=device)
    y = torch.tensor(y_list[train_indices], dtype=torch.float32, device=device)
    t = torch.tensor(t_list[train_indices], dtype=torch.float32, device=device)
    pwat = torch.tensor(pwat_rand[train_indices], dtype=torch.float32, device=device)
    poil = torch.tensor(poil_rand[train_indices], dtype=torch.float32, device=device)
    kwat = torch.tensor(kwat_rand[train_indices], dtype=torch.float32, device=device)
    koil = torch.tensor(koil_rand[train_indices], dtype=torch.float32, device=device)

    points = torch.stack((t, x, y, pwat, poil, kwat, koil), dim=-1)
    perm_vec = torch.tensor(
        perm[x_idx[train_indices], y_idx[train_indices], -1].astype(np.float32),
        dtype=torch.float32,
        device=device,
    )
    return simulation_data_train, points, perm_vec


def build_boundary_points(points):
    t = points[:, 0]
    x = points[:, 1]
    y = points[:, 2]
    pwat = points[:, 3]
    poil = points[:, 4]
    kwat = points[:, 5]
    koil = points[:, 6]

    pres0_x1 = torch.stack((t, torch.zeros_like(x), y, pwat, poil, kwat, koil), dim=-1)
    pres1_x1 = torch.stack((t, torch.ones_like(x), y, pwat, poil, kwat, koil), dim=-1)
    swat0_x1 = torch.stack((torch.zeros_like(t), x, y, pwat, poil, kwat, koil), dim=-1)
    soil0_x1 = torch.stack((torch.zeros_like(t), x, y, pwat, poil, kwat, koil), dim=-1)
    return pres0_x1, pres1_x1, swat0_x1, soil0_x1

def train(args):
    set_seed(args.seed)
    device = resolve_device(args.device)
    base_dir = Path(__file__).resolve().parent
    save_dir = base_dir / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    simulation_data_train, points, perm_vec = prepare_training_data(
        base_dir=base_dir,
        n_points=args.n_collocation,
        device=device,
        raw_sim_file=args.raw_sim_file,
        generate_if_missing=args.generate_indices_if_missing,
    )
    pres0_x1, pres1_x1, swat0_x1, soil0_x1 = build_boundary_points(points)

    model = ModifiedPINN(
        input_dim=7,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        output_dim=7,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    balancer, balancer_optimizer = create_balancer(args, device)
    validation_reference = None
    validation_plot_dir = None
    if args.plot_every > 0:
        validation_plot_dir = save_dir / f"test4_validation_{args.balancer}_n{args.n_collocation}"
        validation_plot_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = save_dir / f"test4_pinn_{args.balancer}_n{args.n_collocation}.pth"
    best_raw_total = float("inf")
    best_epoch = -1
    artifact_paths = make_run_artifact_paths(
        save_dir, f"test4_metrics_{args.balancer}_n{args.n_collocation}"
    )
    logger = TrainingRunLogger(artifact_paths["csv"])
    train_start_time = time.perf_counter()

    print(
        f"Training test4 PINN: balancer={args.balancer}, epochs={args.epochs}, "
        f"N={args.n_collocation}, device={device}"
    )

    for epoch in trange(args.epochs):
        epoch_start_time = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        r1, r2, r3_x1, r3_x2, r4_x1, r4_x2, r5, _ = compute_pde_residuals(
            model, points, perm_vec
        )
        loss_pde = torch.mean(r1**2 + r2**2 + r3_x1**2 + r3_x2**2 + r4_x1**2 + r4_x2**2 + r5**2)

        press0 = model(pres0_x1)[:, 0]
        press1 = model(pres1_x1)[:, 0]
        soil0 = model(soil0_x1)[:, 1]
        swat0 = model(swat0_x1)[:, 2]
        loss_ic = torch.mean((press0 - 1.0) ** 2 + press1**2 + (soil0 - 1.0) ** 2 + swat0**2)

        model_res_data = model(points)
        loss_data = torch.mean(
            (model_res_data[:, 0] - simulation_data_train[:, 0]) ** 2
            + (model_res_data[:, 2] - simulation_data_train[:, 1]) ** 2
            + (model_res_data[:, 1] - simulation_data_train[:, 2]) ** 2
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

        raw_total_value = float(raw_total_loss.detach().cpu().item())
        if raw_total_value < best_raw_total:
            best_raw_total = raw_total_value
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
                "raw_total_loss": raw_total_value,
                "balanced_loss": float(balanced_loss.detach().cpu().item()),
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
                f"raw_total={raw_total_value:.6e} "
                f"balanced={balanced_loss.detach().cpu().item():.6e} "
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
            plot_validation(
                model,
                *validation_reference,
                device=device,
                pwat=args.plot_pwat,
                poil=args.plot_poil,
                kwat=args.plot_kwat,
                koil=args.plot_koil,
                save_path=plot_path,
                show=False,
                close=True,
            )
            print(f"Saved validation plot: {plot_path}")
        elif should_plot:
            validation_reference = get_plot_data(
                pwat=args.plot_pwat,
                poil=args.plot_poil,
                kwat=args.plot_kwat,
                koil=args.plot_koil,
            )
            plot_path = validation_plot_dir / f"epoch_{epoch + 1:06d}.png"
            plot_validation(
                model,
                *validation_reference,
                device=device,
                pwat=args.plot_pwat,
                poil=args.plot_poil,
                kwat=args.plot_kwat,
                koil=args.plot_koil,
                save_path=plot_path,
                show=False,
                close=True,
            )
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
            "test_name": "test4",
            "balancer": args.balancer,
            "n_collocation": args.n_collocation,
            "epochs": args.epochs,
            "device": str(device),
            "total_train_time_sec": total_train_time_sec,
            "best_raw_total_loss": best_raw_total,
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
