import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_module(module_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_training_module(train_module, cli_args):
    old_argv = sys.argv[:]
    try:
        sys.argv = [str(train_module.__file__), *cli_args]
        args = train_module.parse_args()
        train_module.train(args)
    finally:
        sys.argv = old_argv


def find_latest_summary(save_dir, prefix):
    candidates = sorted(
        Path(save_dir).glob(f"{prefix}_*_summary.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No summary files found for prefix {prefix} in {save_dir}")
    return candidates[-1]


def compute_scalar_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - float(np.sum((y_true - y_pred) ** 2)) / denom if denom > 0 else float("nan")
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
    }


def compute_multioutput_metrics(y_true, y_pred, target_names):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    results = {}
    for idx, target_name in enumerate(target_names):
        target_metrics = compute_scalar_metrics(y_true[:, idx], y_pred[:, idx])
        for metric_name, metric_value in target_metrics.items():
            results[f"{target_name}_{metric_name}"] = metric_value
    for metric_name in ("MSE", "RMSE", "MAE", "R2"):
        metric_values = [results[f"{target}_{metric_name}"] for target in target_names]
        results[f"mean_{metric_name}"] = float(np.mean(metric_values))
    return results


def plot_convergence(csv_path, output_dir, run_label):
    df = pd.read_csv(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["epoch"], df["raw_total_loss"], label="raw_total_loss")
    ax.plot(df["epoch"], df["balanced_loss"], label="balanced_loss")
    ax.plot(df["epoch"], df["loss_data"], label="loss_data")
    ax.plot(df["epoch"], df["loss_bc_ic"], label="loss_bc_ic")
    ax.plot(df["epoch"], df["loss_pde"], label="loss_pde")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Loss convergence: {run_label}")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{run_label}_losses.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["epoch"], df["weight_data"], label="weight_data")
    ax.plot(df["epoch"], df["weight_bc_ic"], label="weight_bc_ic")
    ax.plot(df["epoch"], df["weight_pde"], label="weight_pde")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weight")
    ax.set_title(f"Weight convergence: {run_label}")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{run_label}_weights.png", dpi=200)
    plt.close(fig)


def plot_training_time(results_df, output_dir):
    df = results_df.copy()
    df["run_label"] = df["balancer"] + "\nN=" + df["n_collocation"].astype(str)
    fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.8), 6))
    ax.bar(df["run_label"], df["total_train_time_sec"])
    ax.set_ylabel("Total train time, sec")
    ax.set_title("Training time comparison")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "training_time_comparison.png", dpi=200)
    plt.close(fig)


def plot_final_loss(results_df, output_dir):
    df = results_df.copy()
    df["run_label"] = df["balancer"] + "\nN=" + df["n_collocation"].astype(str)
    fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.8), 6))
    ax.bar(df["run_label"], df["best_objective"])
    ax.set_ylabel("Best objective")
    ax.set_title("Best training objective comparison")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "best_objective_comparison.png", dpi=200)
    plt.close(fig)


def plot_test_metric_comparison(results_df, output_dir):
    output_dir = Path(output_dir)
    df = results_df.copy()
    df["run_label"] = df["balancer"] + "\nN=" + df["n_collocation"].astype(str)

    if "mean_R2" in df.columns:
        metric_pairs = [("mean_R2", "Mean R2"), ("mean_RMSE", "Mean RMSE")]
    else:
        metric_pairs = [("R2", "R2"), ("RMSE", "RMSE")]

    fig, axes = plt.subplots(1, len(metric_pairs), figsize=(7 * len(metric_pairs), 6))
    if len(metric_pairs) == 1:
        axes = [axes]
    for ax, (metric_col, title) in zip(axes, metric_pairs):
        ax.bar(df["run_label"], df[metric_col])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / "test_metric_comparison.png", dpi=200)
    plt.close(fig)

    target_r2_cols = [col for col in df.columns if col.endswith("_R2") and not col.startswith("mean_")]
    target_rmse_cols = [col for col in df.columns if col.endswith("_RMSE") and not col.startswith("mean_")]

    if target_r2_cols:
        fig, axes = plt.subplots(1, len(target_r2_cols), figsize=(7 * len(target_r2_cols), 6))
        if len(target_r2_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, target_r2_cols):
            ax.bar(df["run_label"], df[col])
            ax.set_title(col)
            ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(output_dir / "test_r2_by_target.png", dpi=200)
        plt.close(fig)

    if target_rmse_cols:
        fig, axes = plt.subplots(1, len(target_rmse_cols), figsize=(7 * len(target_rmse_cols), 6))
        if len(target_rmse_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, target_rmse_cols):
            ax.bar(df["run_label"], df[col])
            ax.set_title(col)
            ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(output_dir / "test_rmse_by_target.png", dpi=200)
        plt.close(fig)


def save_results_table(results_df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ablation_results.csv"
    results_df.to_csv(csv_path, index=False)
    return csv_path
