import pandas as pd
import pickle as pkl
import argparse
import sys
import os
import json
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
    """
    Scale a temporal window around its center point.

    Args:
        start: Start time of window
        end: End time of window
        min_duration: Minimum allowed duration
        max_duration: Maximum allowed duration
        min_start: Minimum allowed start time
        video_duration: Total duration of video
        scale_factor: Factor to scale the window by

    Returns:
        new_start, new_end: Adjusted window boundaries
    """
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
    """
    Compute 1D Intersection over Union between two temporal segments.

    Args:
        seg1: (start, end) tuple for first segment
        seg2: (start, end) tuple for second segment

    Returns:
        IoU value between 0 and 1
    """
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
    """Generate comparison plots for multiple datasets."""
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
        "Individual plots saved: iou_recall_curve.png, iou_average_comparison.png, iou_zero_counts.png"
    )


def plot_scaling_analysis(scale_results, out_path="/u/dduka/project/AVION/images"):
    """Generate plots showing the effect of window scaling on recall."""
    if not scale_results:
        print("No scaling results to plot.")
        return

    plt.figure(figsize=(14, 8))

    # Use viridis colormap for better differentiation
    thresholds = scale_results["thresholds"]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(thresholds)))

    for i, t in enumerate(thresholds):
        plt.plot(
            scale_results["scales"],
            scale_results[t],
            "o-",
            label=f"Recall@{t}",
            color=colors[i],
            linewidth=1.5,
            markersize=4,
        )

    # Overlay Mean IoU
    plt.plot(
        scale_results["scales"],
        scale_results["mIoU"],
        "--",
        color="red",
        alpha=0.7,
        label="Mean IoU",
        linewidth=2,
    )

    plt.title(
        "Effect of Window Scaling on Temporal Recall", fontsize=14, fontweight="bold"
    )
    plt.xlabel("Scale Factor", fontsize=12)
    plt.ylabel("Recall / IoU Score", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(title="Metric", loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()

    plt.savefig(f"{out_path}/scaling_recall_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print("Scaling analysis plot saved: scaling_recall_analysis.png")


def analyze_scaling_effect(
    csv_lookup, pkl_data, video_len_dict, min_scale, max_scale, step=0.1
):
    """
    Analyze the effect of window scaling on IoU and recall metrics.

    Args:
        csv_lookup: Dictionary mapping UUID to ground truth segments
        pkl_data: List of pickle data samples
        video_len_dict: Dictionary mapping video IDs to their durations
        min_scale: Minimum scale factor
        max_scale: Maximum scale factor
        step: Step size for scale factors

    Returns:
        Dictionary containing results for each scale and threshold
    """
    scales = np.arange(min_scale, max_scale + 0.01, step)
    thresholds = [round(t, 1) for t in np.arange(0.1, 1.1, 0.1)]

    results = {t: [] for t in thresholds}
    results["scales"] = []
    results["mIoU"] = []
    results["thresholds"] = thresholds

    max_dur_global = max(video_len_dict.values()) if video_len_dict else 1000.0

    print("\nRunning scaling analysis...")
    for scale in tqdm(scales, desc="Evaluating scale factors"):
        ious = []
        for sample in pkl_data:
            uuid, vid = str(sample[0]), sample[1]
            if uuid in csv_lookup and vid in video_len_dict:
                n_s, n_e = jitter_scale_window(
                    sample[2],
                    sample[3],
                    max_duration=max_dur_global,
                    video_duration=video_len_dict[vid],
                    scale_factor=scale,
                )
                iou = compute_1d_iou(csv_lookup[uuid], (n_s, n_e))
                ious.append(iou)

        if ious:
            results["scales"].append(scale)
            results["mIoU"].append(np.mean(ious))
            for t in thresholds:
                recall = np.mean([1 if i >= t else 0 for i in ious])
                results[t].append(recall)
        else:
            # No valid samples for this scale
            results["scales"].append(scale)
            results["mIoU"].append(0.0)
            for t in thresholds:
                results[t].append(0.0)

    return results


def print_scaling_peak_analysis(scale_results):
    """Print analysis of optimal scale factors for different metrics."""
    if not scale_results or not scale_results.get("scales"):
        print("No scaling results available.")
        return

    print("\n" + "=" * 60)
    print("PEAK PERFORMANCE ANALYSIS - OPTIMAL SCALE FACTORS")
    print("=" * 60)

    scales = scale_results["scales"]

    # Find optimal scale for mIoU
    best_miou_idx = np.argmax(scale_results["mIoU"])
    print(
        f"Optimal Scale for mIoU:      {scales[best_miou_idx]:.2f} "
        f"(Value: {scale_results['mIoU'][best_miou_idx]:.4f})"
    )

    # Print optimal scales for key thresholds
    key_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    print("\nOptimal Scales for Recall Thresholds:")
    for t in key_thresholds:
        if t in scale_results:
            best_idx = np.argmax(scale_results[t])
            best_scale = scales[best_idx]
            best_val = scale_results[t][best_idx]
            print(f"  Recall@{t:.1f}: Scale {best_scale:.2f} (Value: {best_val:.4f})")

    # Recommendation
    if 0.5 in scale_results:
        peak_05_idx = np.argmax(scale_results[0.5])
        print(f"\n{'─' * 60}")
        print(
            f"RECOMMENDATION: For general training, use Scale Factor {scales[peak_05_idx]:.2f}"
        )
        print(f"{'─' * 60}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compute IoU between manually annotated CSV and pickle data with optional scaling analysis."
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
        help="Include scaling analysis to evaluate effect of window scaling on recall",
    )
    parser.add_argument(
        "--min_scale",
        type=float,
        default=1.0,
        help="Minimum value of scale factor (default: 1.0)",
    )
    parser.add_argument(
        "--max_scale",
        type=float,
        default=5.0,
        help="Maximum value of scale factor (default: 5.0)",
    )
    parser.add_argument(
        "--scale_step",
        type=float,
        default=0.1,
        help="Step size for scale factors (default: 0.1)",
    )
    parser.add_argument(
        "--video_lengths",
        type=str,
        help="Path to JSON file containing video lengths (required for scaling analysis)",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="/u/dduka/project/AVION/images",
        help="Directory to save output plots",
    )
    args = parser.parse_args()

    # Validation
    if args.include_scaled and not args.video_lengths:
        sys.exit("Error: --video_lengths is required when --include_scaled is enabled")

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

    # --- Load video lengths if needed ---
    video_len_dict = None
    if args.include_scaled:
        if not os.path.exists(args.video_lengths):
            sys.exit(f"Error: Video lengths file not found at {args.video_lengths}")

        print(f"Loading video lengths from: {args.video_lengths}...")
        with open(args.video_lengths, "r") as f:
            video_len_dict = json.load(f)
        print(f"Loaded lengths for {len(video_len_dict)} videos.")

    results_summary = []

    # --- Loop over pickle files ---
    for pkl_path in args.pkl:
        pkl_name = os.path.basename(pkl_path)
        print(f"\n{'='*60}")
        print(f"Processing Pickle: {pkl_name}")
        print(f"{'='*60}")

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
        print("-" * 40)
        print(f"Total Matches Processed: {matches_found}")

        if iou_results:
            ious = np.array(iou_results)
            avg_iou = np.mean(ious)
            min_iou = np.min(ious)
            max_iou = np.max(ious)
            zero_count = len(ious) - np.count_nonzero(ious)

            print(f"Average IoU: {(100 * avg_iou):.4f}%")
            print(f"Min IoU: {(100 * min_iou):.4f}%")
            print(f"Max IoU: {(100 * max_iou):.4f}%")
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
        print("-" * 40)

        # --- 5. Scaling Analysis (if enabled and this is the first/only pickle) ---
        if args.include_scaled and pkl_path == args.pkl[0]:
            print("\n" + "=" * 60)
            print("SCALING ANALYSIS")
            print("=" * 60)

            scale_results = analyze_scaling_effect(
                csv_lookup,
                pkl_data,
                video_len_dict,
                args.min_scale,
                args.max_scale,
                args.scale_step,
            )

            # Print peak analysis
            print_scaling_peak_analysis(scale_results)

            # Generate scaling plots
            os.makedirs(args.output_dir, exist_ok=True)
            plot_scaling_analysis(scale_results, args.output_dir)

    # --- 6. Generate Comparison Plots ---
    if results_summary:
        print("\nGenerating comparison plots...")
        os.makedirs(args.output_dir, exist_ok=True)
        plot_results(results_summary, args.output_dir)

    print(f"\nAll plots saved to: {args.output_dir}")
    print("Analysis complete!")


if __name__ == "__main__":
    main()
