import pandas as pd
import pickle as pkl
import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt


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


def plot_results(results_summary):
    if not results_summary:
        print("No results to plot.")
        return

    # Extract data
    names = [r["name"] for r in results_summary]

    # 1. Recall Curve
    plt.figure(figsize=(10, 6))
    thresholds = [0.1, 0.3, 0.5, 0.7, 1.0]

    for res in results_summary:
        recalls = []
        ious = np.array(res["ious"])
        if len(ious) == 0:
            recalls = [0] * len(thresholds)
        else:
            for t in thresholds:
                recall = np.sum(ious >= t) / len(ious)
                recalls.append(recall * 100)

        plt.plot(thresholds, recalls, marker="o", label=res["name"])

    plt.title("IoU Recall Curve")
    plt.xlabel("IoU Threshold")
    plt.ylabel("Recall (%)")
    plt.xticks(thresholds)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("iou_recall_curve.png")
    print("Saved iou_recall_curve.png")
    plt.close()

    # 2. Average IoU Bar Chart
    plt.figure(figsize=(12, 6))
    avg_ious = [r["avg_iou"] * 100 for r in results_summary]
    bars = plt.bar(names, avg_ious, color="skyblue")

    plt.title("Average IoU Comparison")
    plt.ylabel("Average IoU (%)")
    plt.xticks(rotation=45, ha="right")

    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig("average_iou.png")
    print("Saved average_iou.png")
    plt.close()

    # 3. Zero IoU Count Bar Chart
    plt.figure(figsize=(12, 6))
    zero_counts = [r["zero_iou_count"] for r in results_summary]
    bars = plt.bar(names, zero_counts, color="salmon")

    plt.title("Number of Samples with 0.0 IoU")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig("zero_iou_count.png")
    print("Saved zero_iou_count.png")
    plt.close()

    # 4. IoU Distribution (Box Plot)
    plt.figure(figsize=(12, 6))
    data_to_plot = [r["ious"] for r in results_summary]
    plt.boxplot(
        data_to_plot, tick_labels=[n[:20] + "..." if len(n) > 20 else n for n in names]
    )  # Shorten names for x-axis if too long

    plt.title("IoU Distribution")
    plt.ylabel("IoU")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis="y")

    plt.tight_layout()
    plt.savefig("iou_distribution.png")
    print("Saved iou_distribution.png")
    plt.close()


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
    args = parser.parse_args()

    # --- 1. Load and Process CSV ---
    if not os.path.exists(args.csv):
        sys.exit(f"Error: CSV not found at {args.csv}")

    print(f"Loading CSV: {args.csv}...")
    man_ann_df = pd.read_csv(args.csv)

    # normalizing column names for safety (strip spaces, lowercase)
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

            # Store results
            results_summary.append(
                {
                    "name": pkl_name,
                    "ious": iou_results,
                    "avg_iou": avg_iou,
                    "zero_iou_count": zero_count,
                }
            )
        else:
            print("No matches found between CSV and Pickle UUIDs.")
            results_summary.append(
                {"name": pkl_name, "ious": [], "avg_iou": 0.0, "zero_iou_count": 0}
            )
        print("-" * 30)

    # --- 5. Plotting ---
    print("\nGenerating plots...")
    plot_results(results_summary)


if __name__ == "__main__":
    main()
