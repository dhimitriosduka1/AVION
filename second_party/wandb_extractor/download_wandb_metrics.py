#!/usr/bin/env python3
import argparse
import wandb
import pandas as pd
import os


def get_state_at_peak(entity, project, run_id, main_metric):
    api = wandb.Api()
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
    except Exception as e:
        print(f"  [!] Error accessing run {run_id}: {e}")
        return None

    print(f"\n--- Exploring Run: {run_id} ---")

    # 1. Identify the actual column name
    available_keys = list(run.history(samples=1).columns)
    print(f"  [i] Available keys in history: {available_keys}")
    actual_col = next((k for k in available_keys if main_metric in k), None)

    if not actual_col:
        print(
            f"  [!] Could not find '{main_metric}'. Available: {available_keys[:10]}..."
        )
        return None

    # 2. Fetch all history data for debugging
    print(f"  [i] Fetching all history data for debugging...")
    history_df = run.history(keys=available_keys, pandas=True)  # Fetch all history data
    print(f"  [DEBUG] Full history data:\n{history_df}")

    if history_df.empty or actual_col not in history_df.columns:
        print(f"  [!] No history found for '{actual_col}'")
        return None

    valid_data = history_df.dropna(subset=[actual_col])
    if valid_data.empty:
        print(f"  [!] Metric '{actual_col}' contains only NaNs.")
        return None

    # 3. Find the peak step
    best_row_meta = valid_data.loc[valid_data[actual_col].idxmax()]
    best_step = int(best_row_meta["_step"])
    best_val = best_row_meta[actual_col]
    print(f"  [✓] Found peak {best_val:.6f} at step {best_step}")

    # 4. Fetch all metrics at the peak step explicitly
    print(f"  [i] Extracting all metrics at peak step {best_step}...")
    peak_metrics = history_df[history_df["_step"] == best_step]

    if peak_metrics.empty:
        print(f"  [!] No metrics found at step {best_step}.")
        return None

    # Ensure all metrics are present, fill missing ones with NaN
    for key in available_keys:
        if key not in peak_metrics.columns:
            peak_metrics[key] = None

    # Attach run_id and config
    peak_metrics["run_id"] = run_id
    for k, v in run.config.items():
        conf_key = f"config/{k}"
        if conf_key not in peak_metrics.columns:
            peak_metrics[conf_key] = str(v)

    return peak_metrics.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Download W&B metrics at a specific peak."
    )
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-ids", nargs="+", required=True)
    parser.add_argument("--main-metric", required=True)
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--window", type=int, default=100)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    all_best_states = []
    for rid in args.run_ids:
        state = get_state_at_peak(args.entity, args.project, rid, args.main_metric)
        if state is not None:
            all_best_states.append(state)
            # state.to_csv(
            #     os.path.join(args.output_dir, f"{rid}_peak_state.csv"), index=False
            # )

    if all_best_states:
        # 5. Robust Summary Consolidation
        summary = pd.concat(all_best_states, ignore_index=True, sort=False)

        # Clean up: remove columns that are entirely NaN
        summary = summary.dropna(axis=1, how="all")

        print("\n" + "=" * 40)
        print("PEAK STATE SUMMARY (Transposed)")
        print("=" * 40)

        # Safe check for indexing
        if "run_id" in summary.columns:
            display_df = summary.set_index("run_id").T
        else:
            display_df = summary.T

        print(display_df)

        # Add a column for the run name before filtering
        summary["run_name"] = summary["run_id"]

        # Filter the columns to save only the specified ones
        columns_to_keep = [
            "run_name",
            "_step",
            # "test_ego4d_cls_top3",
            # "test_ego4d_mir_vis_map",
            # "logit_scale",
            "test_ego4d_cls_top1",
            # "test_ego4d_mir_clip_acc",
            "test_charades_mAP",
            "test_ego4d_cls_top5",
            "test_ego4d_mir_avg_map",
            # "test_ego4d_mir_txt_ndcg",
            # "test_ego4d_mcq_Intra-video",
            "test_egtea_mean_class_acc",
            "test_egtea_top1_acc",
            "test_ego4d_mir_avg_ndcg",
            # "test_ego4d_mir_txt_map",
            # "test_ego4d_mcq_Inter-video",
            # "test_ego4d_cls_top10",
            # "test_ego4d_mir_vis_ndcg",
        ]
        summary = summary[columns_to_keep]

        # Scale metrics less than 1.0 by multiplying by 100
        for column in columns_to_keep:
            if summary[column].dtype in ["float64", "float32"]:
                summary[column] = summary[column].apply(
                    lambda x: x * 100 if x < 1.0 else x
                )

        # Round all numeric metrics to 3 decimal places
        summary = summary.round(3)

        final_path = os.path.join(args.output_dir, "combined_peak_results.csv")
        summary.to_csv(final_path, index=False)
        print(f"\n[Done] Full results saved to: {final_path}")
    else:
        print("\n[!] No data was successfully retrieved for any run IDs.")


if __name__ == "__main__":
    main()
