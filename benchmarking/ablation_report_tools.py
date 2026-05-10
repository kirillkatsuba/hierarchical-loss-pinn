import argparse
import os
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
(CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
(CACHE_DIR / "fontconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PREFERRED_PLOT_METRICS = [
    "MSE",
    "RMSE",
    "MAE",
    "R2",
    "mean_MSE",
    "mean_RMSE",
    "mean_MAE",
    "mean_R2",
    "total_train_time_sec",
    "best_objective",
]
TABLE_DEFAULT_METRICS = [
    "total_train_time_sec",
    "best_objective",
    "MSE",
    "RMSE",
    "MAE",
    "R2",
]
LOG_SCALE_METRICS = {"MSE", "RMSE", "MAE", "best_objective"}
NON_NUMERIC_COLUMNS = {
    "test_name",
    "balancer",
    "device",
    "best_checkpoint",
    "summary_path",
    "metrics_csv_path",
}
METRIC_SUFFIXES = ("_MSE", "_RMSE", "_MAE", "_R2")


def load_results(csv_path):
    df = pd.read_csv(csv_path)
    required_columns = {"balancer", "n_collocation"}
    missing = required_columns - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns in {csv_path}: {missing_str}")

    df["n_collocation"] = pd.to_numeric(df["n_collocation"], errors="raise")
    for column in df.columns:
        if column not in NON_NUMERIC_COLUMNS:
            try:
                df[column] = pd.to_numeric(df[column])
            except (TypeError, ValueError):
                pass

    return df.sort_values(["balancer", "n_collocation"]).reset_index(drop=True)


def is_metric_column(column_name):
    return column_name not in NON_NUMERIC_COLUMNS and column_name not in {"epochs", "best_epoch", "n_collocation"}


def resolve_plot_metrics(df, requested_metrics=None):
    if requested_metrics:
        return [metric for metric in requested_metrics if metric in df.columns]

    ordered_metrics = []
    for metric in PREFERRED_PLOT_METRICS:
        if metric in df.columns and metric not in ordered_metrics:
            ordered_metrics.append(metric)

    target_metrics = sorted(
        column
        for column in df.columns
        if column.endswith(METRIC_SUFFIXES) and not column.startswith("mean_")
    )
    for metric in target_metrics:
        if metric not in ordered_metrics:
            ordered_metrics.append(metric)

    other_metrics = [
        column
        for column in df.columns
        if is_metric_column(column) and column not in ordered_metrics
    ]
    for metric in other_metrics:
        ordered_metrics.append(metric)

    return ordered_metrics


def plot_metric(df, metric, output_dir):
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' is not present in the input CSV.")

    pivot_df = (
        df.pivot_table(
            index="n_collocation",
            columns="balancer",
            values=metric,
            aggfunc="mean",
        )
        .sort_index()
        .sort_index(axis=1)
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for balancer in pivot_df.columns:
        ax.plot(
            pivot_df.index,
            pivot_df[balancer],
            marker="o",
            linewidth=2,
            label=balancer,
        )

    if metric in LOG_SCALE_METRICS and (pivot_df > 0).all().all():
        ax.set_yscale("log")

    ax.set_xlabel("n_collocation")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by n_collocation")
    ax.grid(True, alpha=0.3)
    ax.legend(title="balancer")
    fig.tight_layout()
    fig.savefig(output_dir / f"{metric.lower()}_vs_n_collocation.png", dpi=200)
    plt.close(fig)


def plot_overview(df, metrics, output_dir):
    present_metrics = [metric for metric in metrics if metric in df.columns]
    if not present_metrics:
        raise ValueError("None of the requested metrics are present in the input CSV.")

    ncols = 2
    nrows = (len(present_metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.5 * nrows))
    axes = axes.flatten()

    for ax, metric in zip(axes, present_metrics):
        pivot_df = (
            df.pivot_table(
                index="n_collocation",
                columns="balancer",
                values=metric,
                aggfunc="mean",
            )
            .sort_index()
            .sort_index(axis=1)
        )
        for balancer in pivot_df.columns:
            ax.plot(
                pivot_df.index,
                pivot_df[balancer],
                marker="o",
                linewidth=2,
                label=balancer,
            )
        if metric in LOG_SCALE_METRICS and (pivot_df > 0).all().all():
            ax.set_yscale("log")
        ax.set_title(metric)
        ax.set_xlabel("n_collocation")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)

    for ax in axes[len(present_metrics):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), title="balancer")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / "ablation_overview.png", dpi=200)
    plt.close(fig)


def format_cell(value):
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def dataframe_to_markdown(df, include_index=True):
    render_df = df.reset_index() if include_index else df.copy()
    headers = [str(column) for column in render_df.columns]
    rows = [[format_cell(value) for value in row] for row in render_df.to_numpy()]

    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    header_line = "| " + " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    separator_line = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    row_lines = [
        "| " + " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator_line, *row_lines])


def save_flat_summary(df, output_dir):
    preferred_columns = [
        "balancer",
        "n_collocation",
        "total_train_time_sec",
        "best_objective",
        "best_epoch",
        "MSE",
        "RMSE",
        "MAE",
        "R2",
    ]
    selected_columns = [column for column in preferred_columns if column in df.columns]
    summary_df = df[selected_columns].copy()
    csv_path = output_dir / "ablation_summary_flat.csv"
    md_path = output_dir / "ablation_summary_flat.md"
    summary_df.to_csv(csv_path, index=False)
    md_path.write_text(dataframe_to_markdown(summary_df, include_index=False) + "\n", encoding="utf-8")
    return csv_path, md_path


def save_metric_pivot_tables(df, metrics, output_dir):
    markdown_parts = ["# Ablation Summary Tables", ""]
    created_files = []

    for metric in metrics:
        if metric not in df.columns:
            continue

        pivot_df = (
            df.pivot_table(
                index="balancer",
                columns="n_collocation",
                values=metric,
                aggfunc="mean",
            )
            .sort_index()
            .sort_index(axis=1)
        )
        pivot_df.columns = [f"n={int(value)}" for value in pivot_df.columns]

        metric_slug = metric.lower()
        csv_path = output_dir / f"{metric_slug}_pivot.csv"
        md_path = output_dir / f"{metric_slug}_pivot.md"
        pivot_df.to_csv(csv_path)
        md_path.write_text(dataframe_to_markdown(pivot_df, include_index=True) + "\n", encoding="utf-8")
        created_files.extend([csv_path, md_path])

        markdown_parts.append(f"## {metric}")
        markdown_parts.append("")
        markdown_parts.append(dataframe_to_markdown(pivot_df, include_index=True))
        markdown_parts.append("")

    report_path = output_dir / "ablation_tables.md"
    report_path.write_text("\n".join(markdown_parts), encoding="utf-8")
    created_files.append(report_path)
    return created_files


def build_plot_parser(base_dir):
    parser = argparse.ArgumentParser(
        description="Build plots for ablation results by balancer and n_collocation."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=base_dir / "ablation_results" / "ablation_results.csv",
        help="Path to ablation_results.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / "ablation_results" / "plots",
        help="Directory where plots will be saved.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Metrics to plot. By default, all available result metrics are used.",
    )
    return parser


def build_table_parser(base_dir):
    parser = argparse.ArgumentParser(
        description="Build summary tables for ablation results."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=base_dir / "ablation_results" / "ablation_results.csv",
        help="Path to ablation_results.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / "ablation_results" / "tables",
        help="Directory where summary tables will be saved.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=TABLE_DEFAULT_METRICS,
        help="Metrics to include in pivot tables.",
    )
    return parser


def run_plot_cli(base_dir):
    args = build_plot_parser(base_dir).parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.input_csv)
    present_metrics = resolve_plot_metrics(df, args.metrics)
    if not present_metrics:
        raise ValueError(
            "None of the requested metrics are present in the input CSV. "
            f"Requested: {', '.join(args.metrics or [])}"
        )

    skipped_metrics = [metric for metric in (args.metrics or []) if metric not in df.columns]
    if args.metrics and skipped_metrics:
        print("Skipped missing metrics: " + ", ".join(skipped_metrics))

    for metric in present_metrics:
        plot_metric(df, metric, args.output_dir)
    plot_overview(df, present_metrics, args.output_dir)

    print(f"Input CSV: {args.input_csv}")
    print(f"Plots saved to: {args.output_dir}")


def run_table_cli(base_dir):
    args = build_table_parser(base_dir).parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.input_csv)
    flat_csv_path, flat_md_path = save_flat_summary(df, args.output_dir)
    metric_files = save_metric_pivot_tables(df, args.metrics, args.output_dir)

    print(f"Input CSV: {args.input_csv}")
    print(f"Flat summary CSV: {flat_csv_path}")
    print(f"Flat summary Markdown: {flat_md_path}")
    print("Additional pivot tables:")
    for path in metric_files:
        print(path)
