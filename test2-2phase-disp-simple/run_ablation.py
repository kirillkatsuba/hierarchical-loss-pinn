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
    parser = argparse.ArgumentParser(description="Run ablation for test2.")
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
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate.")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden width.")
    parser.add_argument("--num-layers", type=int, default=8, help="Number of layers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--log-every", type=int, default=250, help="Logging frequency.")
    parser.add_argument(
        "--plot-every",
        type=int,
        default=5_000,
        help="Forwarded to train_pinn.py for periodic validation plots. Use 0 to disable.",
    )
    parser.add_argument("--plot-pwat", type=float, default=2.0, help="Validation pwat.")
    parser.add_argument("--plot-poil", type=float, default=4.0, help="Validation poil.")
    parser.add_argument("--plot-kwat", type=float, default=1.0, help="Validation kwat.")
    parser.add_argument("--plot-koil", type=float, default=0.3, help="Validation koil.")
    parser.add_argument(
        "--prepare-validation-gt",
        action="store_true",
        help="Precompute and cache simulator GT for validation plots before any training run starts.",
    )
    parser.add_argument(
        "--test-n-collocation",
        type=int,
        default=500,
        help="Test dataset size based on sim_N.npy.",
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


def evaluate_test2(module, checkpoint_path, device, test_n_collocation, hidden_dim, num_layers):
    model = module.PINN(
        input_dim=3,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=7,
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    base_dir = Path(module.__file__).resolve().parent
    _, simulation_data, x, y, t, _ = module.prepare_training_data(
        base_dir=base_dir,
        n_points=test_n_collocation,
        device=device,
        generate_if_missing=False,
    )
    points = torch.stack((t, x, y), dim=-1)
    with torch.no_grad():
        pred = model(points).detach().cpu().numpy()
    y_true = torch.stack(
        (simulation_data[:, 0], simulation_data[:, 2], simulation_data[:, 1]), dim=1
    ).detach().cpu().numpy()
    y_pred = pred[:, [0, 1, 2]]
    return compute_multioutput_metrics(y_true, y_pred, ["Pressure", "Soil", "Swat"])


def prepare_validation_gt(module, plot_pwat, plot_poil, plot_kwat, plot_koil):
    module.build_validation_reference(
        base_dir=Path(module.__file__).resolve().parent,
        pwat=plot_pwat,
        poil=plot_poil,
        kwat=plot_kwat,
        koil=plot_koil,
    )


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    script_path = base_dir / "train_pinn.py"
    output_dir = base_dir / args.output_dir
    training_runs_dir = output_dir / "training_runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    training_runs_dir.mkdir(parents=True, exist_ok=True)

    module = load_module(script_path, "test2_train_pinn")
    device = torch.device(args.device)
    rows = []

    if args.prepare_validation_gt and args.plot_every > 0:
        print("Precomputing validation GT cache for test2...")
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
                "--save-dir",
                str(training_runs_dir),
                *args.extra_train_args,
            ]
            run_training_module(module, cli_args)

            prefix = f"test2_metrics_{balancer}_n{n_collocation}"
            summary_path = find_latest_summary(training_runs_dir, prefix)
            summary = json.loads(summary_path.read_text())
            run_label = f"{balancer}_n{n_collocation}"
            plot_convergence(summary["epoch_metrics_csv"], output_dir, run_label)

            test_metrics = evaluate_test2(
                module=module,
                checkpoint_path=summary["best_checkpoint"],
                device=device,
                test_n_collocation=args.test_n_collocation,
                hidden_dim=summary["args"]["hidden_dim"],
                num_layers=summary["args"]["num_layers"],
            )

            rows.append(
                {
                    "test_name": "test2",
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
