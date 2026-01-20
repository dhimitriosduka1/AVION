import pandas as pd
import pickle as pkl
import argparse
import sys
import os
import numpy as np


def compute_1d_iou(seg1, seg2):
    """
    Computes Intersection over Union (IoU) for two 1D segments.

    Args:
        seg1 (tuple): (start, end) of the first segment.
        seg2 (tuple): (start, end) of the second segment.
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


def main():
    parser = argparse.ArgumentParser(
        description="Compute IoU between manually annotated CSV and pickle data."
    )
    parser.add_argument(
        "--csv", "-c", type=str, required=True, help="Path to manually annotated CSV."
    )
    parser.add_argument(
        "--pkl", "-p", type=str, required=True, help="Path to data pickle file."
    )
    args = parser.parse_args()

    # --- 1. Load and Process CSV ---
    if not os.path.exists(args.csv):
        sys.exit(f"Error: CSV not found at {args.csv}")

    print(f"Loading CSV: {args.csv}...")
    # We read the CSV. We assume headers exist, but if the file has no headers,
    # you can remove 'header=0' and pass 'names=[...]' list.
    man_ann_df = pd.read_csv(args.csv)

    # normalizing column names for safety (strip spaces, lowercase)
    man_ann_df.columns = [c.strip() for c in man_ann_df.columns]

    # Based on your description, we identify columns by position if names vary,
    # or by expected names. Here we map the known order to standard keys.
    # Order: uuid, video_id, start_s, end_s, caption
    try:
        # Create a dictionary: { uuid_str: (start_float, end_float) }
        # We enforce string type for UUID to ensure matching works against pickle
        csv_lookup = {}
        for idx, row in man_ann_df.iterrows():
            # Adjust these column names if your CSV headers differ
            uuid = str(row.iloc[0])
            start_s = float(row.iloc[2])
            end_s = float(row.iloc[3])

            # Filter out invalid rows if necessary
            if uuid and uuid != "nan":
                csv_lookup[uuid] = (start_s, end_s)

    except Exception as e:
        sys.exit(
            f"Error processing CSV columns: {e}. Ensure columns are: uuid, video_id, start_s, end_s, caption"
        )

    print(f"Indexed {len(csv_lookup)} annotations from CSV.")

    # --- 2. Load Pickle Data ---
    if not os.path.exists(args.pkl):
        sys.exit(f"Error: Pickle not found at {args.pkl}")

    print(f"Loading Pickle: {args.pkl}...")
    with open(args.pkl, "rb") as f:
        pkl_data = pkl.load(f)

    # --- 3. Compute IoUs ---
    iou_results = []
    matches_found = 0

    print("Computing IoUs...")

    for row in pkl_data:
        # Structure assumption: row[0] is uuid, row[1] is segment (start, end)
        pkl_uuid = str(row[0])

        if pkl_uuid in csv_lookup:
            matches_found += 1

            # Extract segments
            csv_seg = csv_lookup[pkl_uuid]
            pkl_seg = (row[2], row[3])  # Assuming row[1] is (start, end)

            # Compute IoU
            iou = compute_1d_iou(csv_seg, pkl_seg)

            iou_results.append(
                {"uuid": pkl_uuid, "csv_seg": csv_seg, "pkl_seg": pkl_seg, "iou": iou}
            )

    # --- 4. Summary ---
    print("-" * 30)
    print(f"Total Matches Processed: {matches_found}")

    if iou_results:
        avg_iou = np.mean([r["iou"] for r in iou_results])
        print(f"Average IoU: {(100 * avg_iou):.4f}")
        print(f"Min IoU: {(100 * np.min([r['iou'] for r in iou_results])):.4f}")
        print(f"Max IoU: {(100 * np.max([r['iou'] for r in iou_results])):.4f}")
    else:
        print("No matches found between CSV and Pickle UUIDs.")
    print("-" * 30)


if __name__ == "__main__":
    main()
