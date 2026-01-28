import pandas as pd
import pickle as pkl
import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set style for more appealing plots
sns.set_style("whitegrid")
sns.set_palette("husl")

# Mapping from file names to display names
NAME_MAPPING = {}


def jitter_scale_window(
    start,
    end,
    min_duration=1.0,
    max_duration=5.0,
    min_start=0.0,
    video_duration=1.0,
    scale_factor=1.0,
):
    if scale_factor == 1:
        return start, end
    c = 0.5 * (start + end)
    d = max(end - start, 1e-6)
    new_d = max(min(min(d * scale_factor, max_duration), video_duration), min_duration)
    new_start, new_end = c - new_d / 2.0, c + new_d / 2.0

    if new_end > video_duration:
        new_start -= new_end - video_duration
        new_end = video_duration
    if new_start < min_start:
        new_end += min_start - new_start
        new_start = min_start
        if new_end > video_duration:
            new_end = video_duration
    return new_start, new_end


def compute_1d_iou(seg1, seg2):
    s1, e1 = seg1
    s2, e2 = seg2
    inter = max(0, min(e1, e2) - max(s1, s2))
    union = (e1 - s1) + (e2 - s2) - inter
    return inter / union if union > 0 else 0.0


def compute_1d_iou(seg1, seg2):

    start1, end1 = seg1
    start2, end2 = seg2

    # Calculate intersection
    inter_start = max(start1, start2)
    inter_end = min(end1, end2)
    intersection = max(0, inter_end - inter_start)

    # Calculate union
    len1 = end1 - start1
    len2 = end2 - start2
    union = len1 + len2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def plot_results(results_summary, out_path="/u/dduka/project/AVION/images"):
    if not results_summary:
        print("No results to plot.")
        return

    # Extract data
    names = [r["name"] for r in results_summary]
    colors = sns.color_palette("husl", len(results_summary))
    x_pos = range(len(names))

    # --- 1. Recall Curve ---
    plt.figure(figsize=(12, 6))
    thresholds = [0.1, 0.3, 0.5, 0.7, 1.0]

    for idx, res in enumerate(results_summary):
        recalls = []
        ious = np.array(res["ious"])
        if len(ious) == 0:
            recalls = [0] * len(thresholds)
        else:
            for t in thresholds:
                recall = np.sum(ious >= t) / len(ious)
                recalls.append(recall * 100)

        plt.plot(
            thresholds,
            recalls,
            marker="o",
            linewidth=2.5,
            markersize=8,
            label=res["name"],
            color=colors[idx],
        )

    plt.title("IoU Recall Curve", fontsize=14, fontweight="bold")
    plt.xlabel("IoU Threshold", fontsize=12)
    plt.ylabel("Recall (%)", fontsize=12)
    plt.xticks(thresholds)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.ylim([-5, 105])
    plt.savefig(f"{out_path}/iou_recall_curve.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # --- 2. Average IoU Bar/Line Plot ---
    plt.figure(figsize=(12, 6))
    avg_ious = [r["avg_iou"] * 100 for r in results_summary]

    plt.plot(
        x_pos,
        avg_ious,
        marker="o",
        linewidth=2.5,
        markersize=10,
        color="#3498db",
        markerfacecolor="#e74c3c",
    )

    for i, (x, y) in enumerate(zip(x_pos, avg_ious)):
        plt.annotate(
            f"{y:.2f}%",
            xy=(x, y),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
        )

    plt.title("Average IoU Comparison", fontsize=14, fontweight="bold")
    plt.ylabel("Average IoU (%)", fontsize=12)
    plt.xticks(x_pos, names, rotation=45, ha="right")
    plt.grid(True, alpha=0.3, axis="y")
    plt.ylim([0, max(avg_ious) * 1.2 if avg_ious else 100])
    plt.savefig(f"{out_path}/iou_average_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # --- 3. Zero IoU Count Plot ---
    plt.figure(figsize=(12, 6))
    zero_counts = [r["zero_iou_count"] for r in results_summary]

    plt.plot(
        x_pos,
        zero_counts,
        marker="s",
        linewidth=2.5,
        markersize=10,
        color="#e74c3c",
        markerfacecolor="#f39c12",
    )

    for i, (x, y) in enumerate(zip(x_pos, zero_counts)):
        plt.annotate(
            f"{int(y)}",
            xy=(x, y),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
        )

    plt.title("Number of Samples with 0.0 IoU", fontsize=14, fontweight="bold")
    plt.ylabel("Count", fontsize=12)
    plt.xticks(x_pos, names, rotation=45, ha="right")
    plt.grid(True, alpha=0.3, axis="y")
    plt.ylim(
        [0, max(zero_counts) * 1.2 if zero_counts and max(zero_counts) > 0 else 10]
    )
    plt.savefig(f"{out_path}/iou_zero_counts.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print(
        "Individual plots saved: recall_curve.png, average_comparison.png, zero_counts.png"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compute IoU between manually annotated CSV and pickle data."
    )
    parser.add_argument(
        "--csv", "-c", type=str, required=True, help="Path to manually annotated CSV."
    )
    parser.add_argument(
        "--pkl",
        "-p",
        nargs="+",
        type=str,
        required=True,
        help="Path to data pickle file(s).",
    )
    parser.add_argument(
        "--names",
        "-n",
        nargs="+",
        type=str,
        default=None,
        help="Optional: Custom names for each pickle file (must match order of --pkl arguments).",
    )
    parser.add_argument(
        "--include_scaled",
        action="store_true",
        help="Include also evaluation for scaled versions of the dataset",
    )
    parser.add_argument("--min_scale", type=float, help="Minimum value of scale")
    parser.add_argument("--max_scale", type=float, help="Maximum value of scale")
    parser.add_argument(
        "--original_pkl",
        type=str,
        help="Path to the original Ego4D pickle file.",
    )
    args = parser.parse_args()

    if args.include_scaled:
        assert (
            args.min_scale is not None
            and args.max_scale is not None
            and args.original_pkl is not None
        )

    # Create dynamic name mapping if custom names provided
    if args.names:
        if len(args.names) != len(args.pkl):
            sys.exit(
                f"Error: Number of names ({len(args.names)}) must match number of pickle files ({len(args.pkl)})"
            )
        for pkl_path, custom_name in zip(args.pkl, args.names):
            pkl_name = os.path.basename(pkl_path)
            NAME_MAPPING[pkl_name] = custom_name

    # --- 1. Load and Process CSV ---
    if not os.path.exists(args.csv):
        sys.exit(f"Error: CSV not found at {args.csv}")

    print(f"Loading CSV: {args.csv}...")
    man_ann_df = pd.read_csv(args.csv)

    man_ann_df.columns = [c.strip() for c in man_ann_df.columns]

    try:
        csv_lookup = {}
        for idx, row in man_ann_df.iterrows():
            uuid = str(row.iloc[0])
            start_s = float(row.iloc[2])
            end_s = float(row.iloc[3])

            if uuid and uuid != "nan":
                csv_lookup[uuid] = (start_s, end_s)

    except Exception as e:
        sys.exit(
            f"Error processing CSV columns: {e}. Ensure columns are: uuid, video_id, start_s, end_s, caption"
        )

    print(f"Indexed {len(csv_lookup)} annotations from CSV.")

    if args.include_scaled:
        with open(args.original_pkl, "rb") as f:
            original_ego4d_data = pkl.load(f)

        # --- Updated Analysis Loop ---
        scales = np.arange(args.min_scale, args.max_scale + 0.01, 0.1)
        # Updated to include 0.1 through 1.0
        thresholds = [round(t, 1) for t in np.arange(0.1, 1.1, 0.1)]

        scaled_datasets = []
        for scale in tqdm(scales, desc="Computing scaled versions"):
            scaled_ds = []
            for sample in pkl_data:
                uuid, vid = str(sample[0]), sample[1]
                n_s, n_e = jitter_scale_window(
                    sample[2],
                    sample[3],
                    max_duration=max_dur_global,
                    video_duration=video_len_dict[vid],
                    scale_factor=scale,
                )
                scaled_ds.append((sample[0], sample[1], n_s, n_s, sample[4]))

        scaled_datasets.append(scaled_ds)

    results_summary = []

    # --- Loop over pickle files ---
    for pkl_path in args.pkl:
        pkl_name = os.path.basename(pkl_path)
        print(f"\n{'='*40}")
        print(f"Processing Pickle: {pkl_name}")
        print(f"{'='*40}")

        # --- 2. Load Pickle Data ---
        if not os.path.exists(pkl_path):
            print(f"Error: Pickle not found at {pkl_path}")
            continue

        print(f"Loading Pickle: {pkl_path}...")
        with open(pkl_path, "rb") as f:
            pkl_data = pkl.load(f)

        # --- 3. Compute IoUs ---
        iou_results = []
        matches_found = 0

        print("Computing IoUs...")

        for row in pkl_data:
            pkl_uuid = str(row[0])

            if pkl_uuid in csv_lookup:
                matches_found += 1
                csv_seg = csv_lookup[pkl_uuid]
                pkl_seg = (row[2], row[3])
                iou = compute_1d_iou(csv_seg, pkl_seg)
                iou_results.append(iou)

        # --- 4. Summary ---
        print("-" * 30)
        print(f"Total Matches Processed: {matches_found}")

        if iou_results:
            ious = np.array(iou_results)
            avg_iou = np.mean(ious)
            min_iou = np.min(ious)
            max_iou = np.max(ious)
            zero_count = len(ious) - np.count_nonzero(ious)

            print(f"Average IoU: {(100 * avg_iou):.4f}")
            print(f"Min IoU: {(100 * min_iou):.4f}")
            print(f"Max IoU: {(100 * max_iou):.4f}")
            print(f"Number of 0.0 IoU: {zero_count}")

            thresholds = [0.1, 0.3, 0.5, 0.7, 1.0]
            for t in thresholds:
                recall = np.sum(ious >= t) / len(ious)
                print(f"IoU >= {t}: {recall * 100:.2f}%")

            # Store results with mapped name
            display_name = NAME_MAPPING.get(pkl_name, pkl_name)
            results_summary.append(
                {
                    "name": display_name,
                    "ious": iou_results,
                    "avg_iou": avg_iou,
                    "zero_iou_count": zero_count,
                }
            )
        else:
            print("No matches found between CSV and Pickle UUIDs.")
            display_name = NAME_MAPPING.get(pkl_name, pkl_name)
            results_summary.append(
                {"name": display_name, "ious": [], "avg_iou": 0.0, "zero_iou_count": 0}
            )
        print("-" * 30)

    # --- 5. Plotting ---
    print("\nGenerating plots...")
    plot_results(results_summary)


if __name__ == "__main__":
    main()
