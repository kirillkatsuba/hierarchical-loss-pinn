import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from benchmarking.pinn_ablation_utils import (
    compute_scalar_metrics,
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
    parser = argparse.ArgumentParser(description="Run ablation for test1.")
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
        default=[50, 100, 200, 500],
        help="Training N values to benchmark.",
    )
    parser.add_argument("--epochs", type=int, default=1000, help="Epochs per run.")
    parser.add_argument("--device", default="cpu", help="Torch device.")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate.")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden width.")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of layers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--log-every", type=int, default=250, help="Logging frequency.")
    parser.add_argument(
        "--test-n-collocation",
        type=int,
        default=1000,
        help="Test dataset size, using data/x_N.pt and data/t_N.pt.",
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


def evaluate_test1(module, checkpoint_path, device, test_n_collocation, hidden_dim, num_layers):
    model = module.PINN(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    base_dir = Path(module.__file__).resolve().parent
    x_test = torch.load(base_dir / "data" / f"x_{test_n_collocation}.pt", map_location=device)
    t_test = torch.load(base_dir / "data" / f"t_{test_n_collocation}.pt", map_location=device)
    x_test = x_test.to(device)
    t_test = t_test.to(device)
    x_input = torch.stack((t_test, x_test), dim=-1)

    with torch.no_grad():
        y_pred = model(x_input).detach().cpu().numpy().reshape(-1)
    y_true = module.thermal_conductivity_equation(t_test, x_test).detach().cpu().numpy().reshape(-1)
    return compute_scalar_metrics(y_true, y_pred)


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    script_path = base_dir / "train_pinn.py"
    output_dir = base_dir / args.output_dir
    training_runs_dir = output_dir / "training_runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    training_runs_dir.mkdir(parents=True, exist_ok=True)

    module = load_module(script_path, "test1_train_pinn")
    device = torch.device(args.device)
    rows = []

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
                "--save-dir",
                str(training_runs_dir),
                *args.extra_train_args,
            ]
            run_training_module(module, cli_args)

            prefix = f"test1_metrics_{balancer}_n{n_collocation}"
            summary_path = find_latest_summary(training_runs_dir, prefix)
            summary = json.loads(summary_path.read_text())
            run_label = f"{balancer}_n{n_collocation}"
            plot_convergence(summary["epoch_metrics_csv"], output_dir, run_label)

            test_metrics = evaluate_test1(
                module=module,
                checkpoint_path=summary["best_checkpoint"],
                device=device,
                test_n_collocation=args.test_n_collocation,
                hidden_dim=summary["args"]["hidden_dim"],
                num_layers=summary["args"]["num_layers"],
            )

            rows.append(
                {
                    "test_name": "test1",
                    "balancer": balancer,
                    "n_collocation": n_collocation,
                    "epochs": summary["epochs"],
                    "device": summary["device"],
                    "total_train_time_sec": summary["total_train_time_sec"],
                    "best_objective": summary["best_balanced_loss"],
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
