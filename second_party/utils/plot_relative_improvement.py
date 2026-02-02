import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
import argparse
import os

# 1. Setup Style
plt.style.use(["science", "no-latex", "grid"])

# 2. Metric Definitions
task_metrics = [
    "test_ego4d_cls_top1",
    "test_charades_mAP",
    "test_ego4d_mir_avg_map",
    "test_ego4d_mir_avg_ndcg",
    "test_egtea_mean_class_acc",
    "test_egtea_top1_acc",
]
alignment_metric = ["alignment"]
all_metrics = task_metrics + alignment_metric

labels = [
    "Ego4D Cls Top-1",
    "Charades mAP",
    "Ego4D MIR Avg mAP",
    "Ego4D MIR Avg nDCG",
    "EGTEA Mean Class Acc",
    "EGTEA Top-1 Acc",
    "Alignment",
]
metric_to_label = dict(zip(all_metrics, labels))


def plot_final_clean_version(data, baseline_name, title, filename):
    if baseline_name not in data["run_name"].values:
        print(f"Baseline {baseline_name} not found in the current slice.")
        return

    # --- Data Processing ---
    baseline_vals = data.loc[data["run_name"] == baseline_name, all_metrics].iloc[0]

    improvement = data.copy()
    for m in all_metrics:
        improvement[m] = improvement[m] - baseline_vals[m]

    improvement["Mean (Tasks Only)"] = improvement[task_metrics].mean(axis=1)
    plot_df = improvement[improvement["run_name"] != baseline_name].copy()

    plot_metrics = all_metrics + ["Mean (Tasks Only)"]
    metric_order = labels + ["MEAN (Tasks Only)"]

    long = plot_df.melt(id_vars="run_name", value_vars=plot_metrics)
    long["Metric_Label"] = long["variable"].apply(
        lambda x: (
            "MEAN (Tasks Only)"
            if x == "Mean (Tasks Only)"
            else metric_to_label.get(x, x)
        )
    )

    long["Metric_Label"] = pd.Categorical(
        long["Metric_Label"], categories=metric_order, ordered=True
    )
    long = long.dropna(subset=["value"]).copy()

    # --- Figure Construction ---
    # Increased figsize slightly for better margins
    fig, ax = plt.subplots(figsize=(20, 15))

    # Alternating row backgrounds
    for i, _ in enumerate(metric_order):
        bg_color = "#f7f7f7" if i % 2 == 0 else "white"
        if i == len(metric_order) - 1:  # Highlight Mean row
            bg_color = "#fffbe6"
        ax.axhspan(i - 0.5, i + 0.5, color=bg_color, zorder=0)

    # Plot Bars
    sns.barplot(
        data=long,
        y="Metric_Label",
        x="value",
        hue="run_name",
        palette="colorblind",
        ax=ax,
        edgecolor="black",
        linewidth=0.8,
        zorder=3,
        width=0.8,
    )

    # Vertical Baseline
    ax.axvline(0, color="black", lw=1.5, zorder=4)

    # --- Large Label Sizes ---
    ax.tick_params(axis="y", labelsize=20)
    ax.tick_params(axis="x", labelsize=18)

    ax.set_title(title, fontsize=32, fontweight="bold", pad=50)
    ax.set_xlabel(
        "Relative Improvement over Baseline (%)",
        fontsize=24,
        fontweight="bold",
        labelpad=25,
    )
    ax.set_ylabel("", fontsize=22)

    # --- Dynamic Padding & Labels ---
    vals = long["value"].values
    if len(vals) > 0:
        v_min, v_max = np.min(vals), np.max(vals)
        x_range = v_max - v_min
        if x_range < 1e-9:
            x_range = 1.0
        # Increased x-padding to 60% to ensure bar labels don't hit the edge
        ax.set_xlim(v_min - (x_range * 0.6), v_max + (x_range * 0.6))

    for container in ax.containers:
        for rect in container:
            width = rect.get_width()
            if np.isnan(width) or abs(width) < 1e-4:
                continue
            offset = 0.6 if width > 0 else -0.6
            ha = "left" if width > 0 else "right"
            ax.text(
                width + offset,
                rect.get_y() + rect.get_height() / 2,
                f"{width:+.2f}%",
                va="center",
                ha=ha,
                fontsize=18,
                fontweight="bold",
                zorder=5,
            )

    # --- Legend & Layout Padding ---
    ax.legend(
        title="Dataset Variant",
        title_fontsize="20",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=True,
        fontsize=18,
        borderaxespad=0.0,
    )

    # Adding explicit padding around the subplots (left, right, top, bottom)
    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.2)

    # Save with tight layout and extra padding
    plt.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=1.0)
    print(f"Plot saved: {filename}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate Improvement Plots for LAVILA and Dual Encoder models."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="/u/dduka/project/AVION/second_party/wandb_extractor/output/combined_peak_results.csv",
        help="Path to the input CSV results file.",
    )
    parser.add_argument(
        "--lavila_baseline",
        type=str,
        default="DAIS_LAVILA_BASELINE",
        help="Name of the LAVILA baseline run.",
    )
    parser.add_argument(
        "--dual_enc_baseline",
        type=str,
        default="DAIS_DUAL_ENC_BASELINE",
        help="Name of the Dual Encoder baseline run.",
    )

    args = parser.parse_args()

    # Load Data
    if not os.path.exists(args.input):
        # Fallback for local
        args.input = "combined_peak_results.csv"

    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"Error: Could not find {args.input}")
        return

    # --- Run Plotting ---
    lavila_df = df[df["run_name"].str.contains("LAVILA", na=False)].copy()
    non_lavila_df = df[~df["run_name"].str.contains("LAVILA", na=False)].copy()

    if not lavila_df.empty:
        plot_final_clean_version(
            lavila_df,
            args.lavila_baseline,
            "LAVILA Models Improvement",
            "/u/dduka/project/AVION/images/lavila_clean.png",
        )

    if not non_lavila_df.empty:
        plot_final_clean_version(
            non_lavila_df,
            args.dual_enc_baseline,
            "Dual Encoder Improvement",
            "/u/dduka/project/AVION/images/dual_enc_clean.png",
        )


if __name__ == "__main__":
    main()
