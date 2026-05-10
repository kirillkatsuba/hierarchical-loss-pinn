import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

matplotlib.rcParams["image.cmap"] = "jet"


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from benchmarking.pinn_ablation_utils import load_module
from loss_balancong_algorithms.runtime import BALANCER_CHOICES


KNOWN_REPO_NAMES = {
    "hierarchical-loss-pinn",
    "ijcai26-hierarchical-loss-pinns",
}


TEST_CONFIGS = {
    "test1": {
        "dir_name": "test1-therm-conduct",
        "module_name": "test1_plot_fields_train_pinn",
        "model_class": "PINN",
        "input_dim": 2,
        "output_dim": 1,
        "profile_times": [0.0, 0.03, 0.05],
    },
    "test2": {
        "dir_name": "test2-2phase-disp-simple",
        "module_name": "test2_plot_fields_train_pinn",
        "model_class": "PINN",
        "input_dim": 3,
        "output_dim": 7,
        "default_plot_params": {
            "pwat": 2.0,
            "poil": 4.0,
            "kwat": 1.0,
            "koil": 0.3,
        },
        "reference_builder": "build_validation_reference",
    },
    "test3": {
        "dir_name": "test3-2phase-disp-perm-nonlinear",
        "module_name": "test3_plot_fields_train_pinn",
        "model_class": "ModifiedPINN",
        "input_dim": 3,
        "output_dim": 7,
        "default_plot_params": {
            "pwat": 2.0,
            "poil": 4.0,
            "kwat": 1.0,
            "koil": 0.3,
        },
        "reference_builder": "build_validation_reference",
    },
    "test4": {
        "dir_name": "test4-2phase-disp-7d",
        "module_name": "test4_plot_fields_train_pinn",
        "model_class": "ModifiedPINN",
        "input_dim": 7,
        "output_dim": 7,
        "default_plot_params": {
            "pwat": 1.5,
            "poil": 2.0,
            "kwat": 1.5,
            "koil": 0.3,
        },
        "reference_builder": "get_plot_data",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build comparison plots for Pressure and Water Saturation using the "
            "smallest available training set for each benchmark."
        )
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        default=["test1", "test2", "test3", "test4"],
        choices=sorted(TEST_CONFIGS),
        help="Benchmarks to process.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device for inference.")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "graphs"),
        help="Directory for saved plots.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Saved figure DPI.")
    return parser.parse_args()


def resolve_device(device_name):
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return torch.device(device_name)


def load_min_n_runs(test_name):
    config = TEST_CONFIGS[test_name]
    test_dir = REPO_ROOT / config["dir_name"]
    ablation_csv = test_dir / "ablation_results" / "ablation_results.csv"
    if not ablation_csv.exists():
        raise FileNotFoundError(f"Missing ablation results: {ablation_csv}")

    df = pd.read_csv(ablation_csv)
    min_n = int(df["n_collocation"].min())
    df = df[df["n_collocation"] == min_n].copy()
    if df.empty:
        raise ValueError(f"No rows found for the minimum n_collocation in {ablation_csv}")

    order = {name: idx for idx, name in enumerate(BALANCER_CHOICES)}
    df["_order"] = df["balancer"].map(lambda name: order.get(name, len(order)))
    df = df.sort_values(["_order", "balancer"]).drop(columns="_order")
    return test_dir, min_n, df


def load_summary(row, test_dir):
    summary_path = Path(row["summary_path"])
    summary_path = resolve_repo_path(summary_path, test_dir)
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary JSON: {summary_path}")
    return json.loads(summary_path.read_text())


def resolve_repo_path(path_value, fallback_base):
    path = Path(path_value)
    if not path.is_absolute():
        return fallback_base / path
    if path.exists():
        return path

    for parent in path.parents:
        if parent.name in KNOWN_REPO_NAMES:
            try:
                relative = path.relative_to(parent)
                candidate = REPO_ROOT / relative
                if candidate.exists():
                    return candidate
            except ValueError:
                continue
    return path


def get_plot_params(summary, defaults):
    args = summary.get("args", {})
    return {
        "pwat": float(args.get("plot_pwat", defaults["pwat"])),
        "poil": float(args.get("plot_poil", defaults["poil"])),
        "kwat": float(args.get("plot_kwat", defaults["kwat"])),
        "koil": float(args.get("plot_koil", defaults["koil"])),
    }


def instantiate_model(module, model_class, input_dim, hidden_dim, num_layers, output_dim, device):
    model_ctor = getattr(module, model_class)
    model = model_ctor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=output_dim,
    ).to(device)
    return model


def build_reference(test_name, module, test_dir, plot_params):
    if test_name == "test1":
        raise ValueError("test1 does not use 2D validation references")
    if test_name in {"test2", "test3"}:
        return module.build_validation_reference(
            base_dir=test_dir,
            pwat=plot_params["pwat"],
            poil=plot_params["poil"],
            kwat=plot_params["kwat"],
            koil=plot_params["koil"],
        )

    pres, swat, soil = module.get_plot_data(
        pwat=plot_params["pwat"],
        poil=plot_params["poil"],
        kwat=plot_params["kwat"],
        koil=plot_params["koil"],
    )
    nx0, nx1, nx2 = pres.shape[:3]
    return {
        "pres": pres,
        "swat": swat,
        "soil": soil,
        "nx0": nx0,
        "nx1": nx1,
        "nx2": nx2,
    }


def predict_fields(test_name, module, summary, checkpoint_path, reference, plot_params, device):
    config = TEST_CONFIGS[test_name]
    args = summary["args"]
    model = instantiate_model(
        module=module,
        model_class=config["model_class"],
        input_dim=config["input_dim"],
        hidden_dim=int(args["hidden_dim"]),
        num_layers=int(args["num_layers"]),
        output_dim=int(config["output_dim"]),
        device=device,
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        if test_name in {"test2", "test3"}:
            inputs = torch.stack(
                (
                    reference["time_for_model"],
                    reference["cartesian_points"][:, 0],
                    reference["cartesian_points"][:, 1],
                ),
                dim=-1,
            ).to(device)
        else:
            nx0 = reference["nx0"]
            nx1 = reference["nx1"]
            final_time = 0.15e-1 * 100
            time_for_model = torch.full((nx0 * nx1,), final_time, dtype=torch.float32)
            x_for_model = torch.linspace(0.0, 1.0 - 1.0 / nx0, nx0, dtype=torch.float32)
            y_for_model = torch.linspace(0.0, 1.0 - 1.0 / nx1, nx1, dtype=torch.float32)
            cartesian_points = torch.cartesian_prod(x_for_model, y_for_model)
            pwat = torch.full((nx0 * nx1,), plot_params["pwat"], dtype=torch.float32)
            poil = torch.full((nx0 * nx1,), plot_params["poil"], dtype=torch.float32)
            kwat = torch.full((nx0 * nx1,), plot_params["kwat"], dtype=torch.float32)
            koil = torch.full((nx0 * nx1,), plot_params["koil"], dtype=torch.float32)
            inputs = torch.stack(
                (
                    time_for_model,
                    cartesian_points[:, 0],
                    cartesian_points[:, 1],
                    pwat,
                    poil,
                    kwat,
                    koil,
                ),
                dim=-1,
            ).to(device)

        prediction = model(inputs).detach().cpu().numpy()

    nx0 = reference["nx0"]
    nx1 = reference["nx1"]
    nx2 = reference["nx2"]
    pressure = np.squeeze(prediction[:, 0].reshape(nx0, nx1, nx2))
    water_saturation = np.squeeze(prediction[:, 2].reshape(nx0, nx1, nx2))
    return {
        "pressure": pressure,
        "swat": water_saturation,
    }


def extract_ground_truth(reference):
    return {
        "pressure": np.squeeze(reference["pres"][:, :, :, -1]),
        "swat": np.squeeze(reference["swat"][:, :, :, -1]),
    }


def build_test1_profiles(module, times, x_grid):
    profiles = {}
    for time_value in times:
        t_tensor = torch.full_like(x_grid, float(time_value))
        with torch.no_grad():
            profiles[float(time_value)] = (
                module.thermal_conductivity_equation(t_tensor, x_grid).detach().cpu().numpy()
            )
    return profiles


def predict_test1_profiles(module, summary, checkpoint_path, times, x_grid, device):
    config = TEST_CONFIGS["test1"]
    args = summary["args"]
    model = instantiate_model(
        module=module,
        model_class=config["model_class"],
        input_dim=config["input_dim"],
        hidden_dim=int(args["hidden_dim"]),
        num_layers=int(args["num_layers"]),
        output_dim=int(config["output_dim"]),
        device=device,
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    predictions = {}
    with torch.no_grad():
        for time_value in times:
            t_tensor = torch.full_like(x_grid, float(time_value))
            inputs = torch.stack((t_tensor, x_grid), dim=-1).to(device)
            predictions[float(time_value)] = model(inputs).detach().cpu().numpy().reshape(-1)
    return predictions


def plot_fields_grid(test_name, min_n, plot_params, ground_truth, predictions, output_path, dpi):
    balancers = list(predictions)
    ncols = 1 + len(balancers)
    fig, axes = plt.subplots(2, ncols, figsize=(3.1 * ncols, 6.2), constrained_layout=True)

    pressure_fields = [ground_truth["pressure"], *[predictions[name]["pressure"] for name in balancers]]
    swat_fields = [ground_truth["swat"], *[predictions[name]["swat"] for name in balancers]]
    pressure_vmin = min(float(np.min(field)) for field in pressure_fields)
    pressure_vmax = max(float(np.max(field)) for field in pressure_fields)
    swat_vmin = min(float(np.min(field)) for field in swat_fields)
    swat_vmax = max(float(np.max(field)) for field in swat_fields)

    titles = ["Ground Truth", *balancers]
    pressure_im = None
    swat_im = None
    for col, title in enumerate(titles):
        pressure = ground_truth["pressure"] if col == 0 else predictions[title]["pressure"]
        swat = ground_truth["swat"] if col == 0 else predictions[title]["swat"]

        ax_pressure = axes[0, col]
        pressure_im = ax_pressure.imshow(
            pressure,
            cmap="jet",
            origin="lower",
            vmin=pressure_vmin,
            vmax=pressure_vmax,
        )
        ax_pressure.set_title(title)
        ax_pressure.set_xticks([])
        ax_pressure.set_yticks([])

        ax_swat = axes[1, col]
        swat_im = ax_swat.imshow(
            swat,
            cmap="jet",
            origin="lower",
            vmin=swat_vmin,
            vmax=swat_vmax,
        )
        ax_swat.set_xticks([])
        ax_swat.set_yticks([])

    axes[0, 0].set_ylabel("Pressure")
    axes[1, 0].set_ylabel("Water saturation")
    fig.colorbar(pressure_im, ax=axes[0, :], shrink=0.85, pad=0.02)
    fig.colorbar(swat_im, ax=axes[1, :], shrink=0.85, pad=0.02)
    fig.suptitle(
        (
            f"{test_name}: validation fields for the smallest training set "
            f"(n_collocation={min_n})\n"
            f"Validation setup: pwat={plot_params['pwat']}, poil={plot_params['poil']}, "
            f"kwat={plot_params['kwat']}, koil={plot_params['koil']}"
        ),
        fontsize=12,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_test1_profiles_grid(times, min_n, x_grid, ground_truth, predictions, output_path, dpi):
    balancers = list(predictions)
    ncols = 1 + len(balancers)
    nrows = len(times)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.2 * ncols, 2.3 * nrows + 1.0),
        constrained_layout=True,
    )
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    all_values = []
    for time_value in times:
        all_values.append(ground_truth[float(time_value)])
        all_values.extend(predictions[name][float(time_value)] for name in balancers)
    y_min = min(float(np.min(values)) for values in all_values)
    y_max = max(float(np.max(values)) for values in all_values)

    titles = ["Ground Truth", *balancers]
    x_np = x_grid.detach().cpu().numpy()
    for col, title in enumerate(titles):
        for row, time_value in enumerate(times):
            ax = axes[row, col]
            y_values = (
                ground_truth[float(time_value)]
                if col == 0
                else predictions[title][float(time_value)]
            )
            ax.plot(x_np, y_values, color="tab:blue", linewidth=2.0)
            ax.grid(True, alpha=0.35)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(y_min, y_max)
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f"Temperature\n$t={time_value}$")
            if row == len(times) - 1:
                ax.set_xlabel("x")

    fig.suptitle(
        f"test1: solution profiles for the smallest training set (n_collocation={min_n})",
        fontsize=12,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def process_test1(device, output_dir, dpi):
    config = TEST_CONFIGS["test1"]
    test_dir, min_n, runs_df = load_min_n_runs("test1")
    module = load_module(test_dir / "train_pinn.py", config["module_name"])
    times = [float(value) for value in config["profile_times"]]
    x_grid = torch.linspace(0.0, 1.0, 400, dtype=torch.float32)
    ground_truth = build_test1_profiles(module, times, x_grid)

    predictions = {}
    for _, row in runs_df.iterrows():
        summary = load_summary(row, test_dir)
        checkpoint_path = resolve_repo_path(row["best_checkpoint"], test_dir)
        predictions[row["balancer"]] = predict_test1_profiles(
            module=module,
            summary=summary,
            checkpoint_path=checkpoint_path,
            times=times,
            x_grid=x_grid,
            device=device,
        )

    output_path = output_dir / f"test1_profiles_min_n{min_n}.png"
    plot_test1_profiles_grid(
        times=times,
        min_n=min_n,
        x_grid=x_grid,
        ground_truth=ground_truth,
        predictions=predictions,
        output_path=output_path,
        dpi=dpi,
    )
    return output_path


def process_test(test_name, device, output_dir, dpi):
    if test_name == "test1":
        return process_test1(device=device, output_dir=output_dir, dpi=dpi)
    config = TEST_CONFIGS[test_name]
    test_dir, min_n, runs_df = load_min_n_runs(test_name)
    added_to_path = False
    if str(test_dir) not in sys.path:
        sys.path.insert(0, str(test_dir))
        added_to_path = True
    sys.modules.pop("train_logic", None)
    try:
        module = load_module(test_dir / "train_pinn.py", config["module_name"])

        first_summary = load_summary(runs_df.iloc[0], test_dir)
        plot_params = get_plot_params(first_summary, config["default_plot_params"])
        reference = build_reference(test_name, module, test_dir, plot_params)
        ground_truth = extract_ground_truth(reference)

        predictions = {}
        for _, row in runs_df.iterrows():
            summary = load_summary(row, test_dir)
            checkpoint_path = resolve_repo_path(row["best_checkpoint"], test_dir)
            predictions[row["balancer"]] = predict_fields(
                test_name=test_name,
                module=module,
                summary=summary,
                checkpoint_path=checkpoint_path,
                reference=reference,
                plot_params=plot_params,
                device=device,
            )

        output_path = output_dir / f"{test_name}_validation_fields_min_n{min_n}.png"
        plot_fields_grid(
            test_name=test_name,
            min_n=min_n,
            plot_params=plot_params,
            ground_truth=ground_truth,
            predictions=predictions,
            output_path=output_path,
            dpi=dpi,
        )
        return output_path
    finally:
        if added_to_path:
            try:
                sys.path.remove(str(test_dir))
            except ValueError:
                pass


def main():
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)

    for test_name in args.tests:
        output_path = process_test(
            test_name=test_name,
            device=device,
            output_dir=output_dir,
            dpi=args.dpi,
        )
        print(f"Saved {test_name} plot to {output_path}")


if __name__ == "__main__":
    main()
