import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np


def plot_side_by_side(ego4d_segments, egoclip_segments, output_path):
    """Plot the distribution of segment lengths for Ego4D and EgoClip side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    datasets = [
        {
            "data": ego4d_segments,
            "title": "Ego4D Segment Lengths",
            "ax": axes[0],
            "color": "blue",
        },
        {
            "data": egoclip_segments,
            "title": "EgoClip Segment Lengths",
            "ax": axes[1],
            "color": "orange",
        },
    ]

    for item in datasets:
        data = item["data"]
        ax = item["ax"]

        ax.hist(data, bins=50, edgecolor="black", alpha=0.7, color=item["color"])
        ax.set_xlabel("Segment Length (s)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(item["title"], fontsize=14)
        ax.set_yscale(
            "log"
        )  # Log scale as requested/implied by previous context or good practice for distributions

        # Add statistics
        mean_len = np.mean(data)
        median_len = np.median(data)
        std_len = np.std(data)

        stats_text = f"Mean: {mean_len:.2f}\nMedian: {median_len:.2f}\nStd: {std_len:.2f}\nCount: {len(data)}"
        ax.text(
            0.95,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


ego4d_path = "/dais/fs/scratch/dduka/databases/ego4d/ego4d_train.pkl"
egoclip_path = "/u/dduka/project/EgoVLPv2/data_root/egoclip.csv"

with open(ego4d_path, "rb") as f:
    ego4d_data = pkl.load(f)

egoclip_df = pd.read_csv(egoclip_path, on_bad_lines="skip", delimiter="\t")

ego4d_segments = [s[2] - s[1] for s in ego4d_data]

egoclip_segments = [
    s[1]["clip_end"] - s[1]["clip_start"] for s in egoclip_df.iterrows()
]

plot_side_by_side(
    ego4d_segments,
    egoclip_segments,
    "/u/dduka/project/AVION/images/egoclip_vs_ego4d.png",
)
