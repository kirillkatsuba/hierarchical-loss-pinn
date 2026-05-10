import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from benchmarking.pinn_ablation_utils import (
    compute_multioutput_metrics,
    find_latest_summary,
    load_module,
    plot_convergence,
    plot_final_loss,
    plot_test_metric_comparison,
    plot_training_time,
    run_training_module,
    save_results_table,
)
from loss_balancong_algorithms.runtime import BALANCER_CHOICES


def parse_args():
    parser = argparse.ArgumentParser(description="Run ablation for test4.")
    parser.add_argument(
        "--balancers",
        nargs="+",
        default=list(BALANCER_CHOICES),
        choices=BALANCER_CHOICES,
        help="Balancers to benchmark.",
    )
    parser.add_argument(
        "--n-values",
        nargs="+",
        type=int,
        default=[100, 200, 500, 1000],
        help="Training N values to benchmark.",
    )
    parser.add_argument("--epochs", type=int, default=1000, help="Epochs per run.")
    parser.add_argument("--device", default="cpu", help="Torch device.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden width.")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of layers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--log-every", type=int, default=100, help="Logging frequency.")
    parser.add_argument(
        "--plot-every",
        type=int,
        default=5_000,
        help="Forwarded to train_pinn.py for periodic validation plots. Use 0 to disable.",
    )
    parser.add_argument("--plot-pwat", type=float, default=1.5, help="Validation pwat.")
    parser.add_argument("--plot-poil", type=float, default=2.0, help="Validation poil.")
    parser.add_argument("--plot-kwat", type=float, default=1.5, help="Validation kwat.")
    parser.add_argument("--plot-koil", type=float, default=0.3, help="Validation koil.")
    parser.add_argument(
        "--prepare-validation-gt",
        action="store_true",
        help="Precompute and cache simulator GT for validation plots before any training run starts.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=1000,
        help="Number of holdout points for evaluation.",
    )
    parser.add_argument(
        "--raw-sim-file",
        default="data-100-new-two-sigma/sim_100.npy",
        help="Raw simulation file relative to test4 directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="ablation_results",
        help="Directory for ablation plots and tables.",
    )
    parser.add_argument(
        "--extra-train-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional args forwarded to train_pinn.py.",
    )
    return parser.parse_args()


def evaluate_test4(module, checkpoint_path, device, raw_sim_file, train_n_collocation, test_size, hidden_dim, num_layers, seed):
    model = module.ModifiedPINN(
        input_dim=7,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=7,
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    base_dir = Path(module.__file__).resolve().parent
    (
        simulation_data,
        _x_idx,
        _y_idx,
        x_list,
        y_list,
        t_list,
        pwat_rand,
        poil_rand,
        kwat_rand,
        koil_rand,
    ) = module.prepare_candidate_points(base_dir, raw_sim_file)

    train_indices = module.load_training_indices(
        base_dir=base_dir,
        n_points=train_n_collocation,
        candidate_count=simulation_data.shape[0],
        generate_if_missing=False,
    )
    all_indices = np.arange(simulation_data.shape[0])
    holdout_mask = np.ones(simulation_data.shape[0], dtype=bool)
    holdout_mask[train_indices] = False
    holdout_indices = all_indices[holdout_mask]
    if holdout_indices.size == 0:
        holdout_indices = all_indices

    rng = np.random.default_rng(seed)
    rng.shuffle(holdout_indices)
    selected = holdout_indices[: min(test_size, holdout_indices.size)]

    x = torch.tensor(x_list[selected], dtype=torch.float32, device=device)
    y = torch.tensor(y_list[selected], dtype=torch.float32, device=device)
    t = torch.tensor(t_list[selected], dtype=torch.float32, device=device)
    pwat = torch.tensor(pwat_rand[selected], dtype=torch.float32, device=device)
    poil = torch.tensor(poil_rand[selected], dtype=torch.float32, device=device)
    kwat = torch.tensor(kwat_rand[selected], dtype=torch.float32, device=device)
    koil = torch.tensor(koil_rand[selected], dtype=torch.float32, device=device)
    points = torch.stack((t, x, y, pwat, poil, kwat, koil), dim=-1)

    with torch.no_grad():
        pred = model(points).detach().cpu().numpy()
    sim_tensor = torch.tensor(simulation_data[selected], dtype=torch.float32)
    y_true = torch.stack((sim_tensor[:, 0], sim_tensor[:, 2], sim_tensor[:, 1]), dim=1).numpy()
    y_pred = pred[:, [0, 1, 2]]
    return compute_multioutput_metrics(y_true, y_pred, ["Pressure", "Soil", "Swat"])


def prepare_validation_gt(module, plot_pwat, plot_poil, plot_kwat, plot_koil):
    module.get_plot_data(
        pwat=plot_pwat,
        poil=plot_poil,
        kwat=plot_kwat,
        koil=plot_koil,
    )


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    raw_sim_path = base_dir / args.raw_sim_file
    if not raw_sim_path.exists():
        raise FileNotFoundError(
            f"Missing raw simulation file for test4 ablation: {raw_sim_path}"
        )

    script_path = base_dir / "train_pinn.py"
    output_dir = base_dir / args.output_dir
    training_runs_dir = output_dir / "training_runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    training_runs_dir.mkdir(parents=True, exist_ok=True)

    module = load_module(script_path, "test4_train_pinn")
    device = torch.device(args.device)
    rows = []

    if args.prepare_validation_gt and args.plot_every > 0:
        print("Precomputing validation GT cache for test4...")
        prepare_validation_gt(
            module, args.plot_pwat, args.plot_poil, args.plot_kwat, args.plot_koil
        )
        print("Validation GT cache is ready.")

    for n_collocation in args.n_values:
        for balancer in args.balancers:
            cli_args = [
                "--balancer",
                balancer,
                "--epochs",
                str(args.epochs),
                "--n-collocation",
                str(n_collocation),
                "--device",
                args.device,
                "--lr",
                str(args.lr),
                "--seed",
                str(args.seed),
                "--hidden-dim",
                str(args.hidden_dim),
                "--num-layers",
                str(args.num_layers),
                "--log-every",
                str(args.log_every),
                "--plot-every",
                str(args.plot_every),
                "--plot-pwat",
                str(args.plot_pwat),
                "--plot-poil",
                str(args.plot_poil),
                "--plot-kwat",
                str(args.plot_kwat),
                "--plot-koil",
                str(args.plot_koil),
                "--raw-sim-file",
                args.raw_sim_file,
                "--save-dir",
                str(training_runs_dir),
                *args.extra_train_args,
            ]
            run_training_module(module, cli_args)

            prefix = f"test4_metrics_{balancer}_n{n_collocation}"
            summary_path = find_latest_summary(training_runs_dir, prefix)
            summary = json.loads(summary_path.read_text())
            run_label = f"{balancer}_n{n_collocation}"
            plot_convergence(summary["epoch_metrics_csv"], output_dir, run_label)

            test_metrics = evaluate_test4(
                module=module,
                checkpoint_path=summary["best_checkpoint"],
                device=device,
                raw_sim_file=summary["args"]["raw_sim_file"],
                train_n_collocation=n_collocation,
                test_size=args.test_size,
                hidden_dim=summary["args"]["hidden_dim"],
                num_layers=summary["args"]["num_layers"],
                seed=args.seed,
            )

            rows.append(
                {
                    "test_name": "test4",
                    "balancer": balancer,
                    "n_collocation": n_collocation,
                    "epochs": summary["epochs"],
                    "device": summary["device"],
                    "total_train_time_sec": summary["total_train_time_sec"],
                    "best_objective": summary["best_raw_total_loss"],
                    "best_epoch": summary["best_epoch"],
                    "best_checkpoint": summary["best_checkpoint"],
                    "summary_path": str(summary_path),
                    "metrics_csv_path": summary["epoch_metrics_csv"],
                    **test_metrics,
                }
            )

    results_df = pd.DataFrame(rows)
    csv_path = save_results_table(results_df, output_dir)
    plot_training_time(results_df, output_dir)
    plot_final_loss(results_df, output_dir)
    plot_test_metric_comparison(results_df, output_dir)

    print(f"Ablation results CSV: {csv_path}")
    print(f"Plots directory: {output_dir}")


if __name__ == "__main__":
    main()
