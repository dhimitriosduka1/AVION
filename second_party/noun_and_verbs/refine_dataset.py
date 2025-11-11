import math
import pickle
import argparse
from collections import defaultdict
from tqdm import tqdm
import wandb

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Refine GT segments by merging adjacent pseudo-label segments that match taxonomy anchors."
    )
    parser.add_argument(
        "--gt",
        default="/BS/dduka/work/projects/AVION/ego4d_train_enriched.pkl",
        help="Path to ground-truth pickle (e.g., ego4d_train.pkl)",
    )
    parser.add_argument(
        "--pseudolabels",
        default="/BS/dduka/work/projects/AVION/ego4d_train_uncovered_all.narrator_63690737.return_5_enriched_and_filtered.pkl",
        help="Path to pseudo-label pickle",
    )
    parser.add_argument(
        "--out-path",
        required=True,
        help="Desired output path",
    )
    parser.add_argument(
        "--gap",
        default=1.5,
        type=float,
        help="Gap between ground-truth and pseudo-label segments",
    )
    parser.add_argument(
        "--use-only",
        default=None,
        choices=["nouns", "verbs", None],
        help="Whether to use only nouns or verbs for the overlap check",
    )
    return parser.parse_args()


def load_data(path):
    print(f"Loading data from {path} ...")
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return None


def save_data(path, data):
    print(f"\nSaving refined data to {path} ...")
    with open(path, "wb") as f:
        pickle.dump(data, f)


def _group_videos_by_video_id(data):
    video_groups = defaultdict(list)

    for sample in data:
        video_id = sample[0]

        if video_id not in video_groups:
            video_groups[video_id] = []

        video_groups[video_id].append(sample)

    # Sort samples by start time
    for video_id in video_groups:
        video_groups[video_id].sort(key=lambda x: x[1])

    return video_groups


def _merge_data(ground_truth_data, pseudo_labels_data):
    merged_data = defaultdict(list)
    for video_id, video_data in tqdm(pseudo_labels_data.items(), desc="Merging data"):
        gt_segments = ground_truth_data[video_id]
        pseudo_segments = video_data
        if video_id not in merged_data:
            merged_data[video_id] = []

        merged_data[video_id].extend((gt_segments + pseudo_segments))

    # Sort the merged data by start time
    for video_id in merged_data:
        merged_data[video_id].sort(key=lambda x: x[1])

    return merged_data


def _resolve_idx(gt_start, gt_end, pseudo_labels_data):
    # Extract the segments from the pseudo-labels data that are within the gap of the ground truth segment
    pseudo_labels_segments = [(sample[1], sample[2]) for sample in pseudo_labels_data]
    if (gt_start, gt_end) in pseudo_labels_segments:
        return pseudo_labels_segments.index((gt_start, gt_end))

    raise ValueError(
        f"No pseudo-labels segment found for ground truth segment {gt_start} - {gt_end}"
    )


def _has_overlap(
    gt_nouns,
    gt_verbs,
    gt_noun_lemmas,
    gt_verb_lemmas,
    cand_nouns,
    cand_verbs,
    cand_noun_lemmas,
    cand_verb_lemmas,
    use_only=None
):
    if not gt_nouns or not cand_nouns:
        nouns_to_check_gt = gt_noun_lemmas
        nouns_to_check_cand = cand_noun_lemmas
    else:
        nouns_to_check_gt = gt_nouns
        nouns_to_check_cand = cand_nouns

    if not gt_verbs or not cand_verbs:
        verbs_to_check_gt = gt_verb_lemmas
        verbs_to_check_cand = cand_verb_lemmas
    else:
        verbs_to_check_gt = gt_verbs
        verbs_to_check_cand = cand_verbs

    if use_only == None:
        has_noun_overlap = bool(nouns_to_check_gt & nouns_to_check_cand)
        has_verb_overlap = bool(verbs_to_check_gt & verbs_to_check_cand)
    elif use_only == "nouns":
        has_noun_overlap = bool(nouns_to_check_gt & nouns_to_check_cand)
        has_verb_overlap = True
    elif use_only == "verbs":
        has_noun_overlap = True
        has_verb_overlap = bool(verbs_to_check_gt & verbs_to_check_cand)
    else:
        raise ValueError(f"Invalid use_only: {use_only}")

    return has_noun_overlap and has_verb_overlap


def _flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(_flatten(item))
        else:
            result.append(item)
    return result


def _expand_clip(
    merged_data,
    video_id,
    gt_start,
    gt_end,
    gt_noun_vec,
    gt_verb_vec,
    gt_noun_vec_lemmas,
    gt_verb_vec_lemmas,
    gap,
):
    # Resolve the idx of the ground truth segment in the merged data
    all_video_segments = merged_data[video_id]
    idx = _resolve_idx(gt_start, gt_end, all_video_segments)

    # Convert to set for easier processing
    gt_noun_vec = set(_flatten(gt_noun_vec))
    gt_verb_vec = set(_flatten(gt_verb_vec))
    gt_noun_vec_lemmas = set(_flatten(gt_noun_vec_lemmas))
    gt_verb_vec_lemmas = set(_flatten(gt_verb_vec_lemmas))

    start_idx = idx
    while start_idx - 1 >= 0:
        prev_segment = all_video_segments[start_idx - 1]
        current_segment = all_video_segments[start_idx]

        gap_duration = math.fabs(float(current_segment[1]) - float(prev_segment[2]))
        if gap_duration > gap:
            break

        candidate_noun_vec = set(_flatten(prev_segment[4]))
        candidate_verb_vec = set(_flatten(prev_segment[5]))
        candidate_noun_vec_lemmas = set(_flatten(prev_segment[6]))
        candidate_verb_vec_lemmas = set(_flatten(prev_segment[7]))

        if not _has_overlap(
            gt_noun_vec,
            gt_verb_vec,
            gt_noun_vec_lemmas,
            gt_verb_vec_lemmas,
            candidate_noun_vec,
            candidate_verb_vec,
            candidate_noun_vec_lemmas,
            candidate_verb_vec_lemmas,
        ):
            break

        start_idx -= 1

    end_idx = start_idx
    while end_idx + 1 < len(all_video_segments):
        next_segment = all_video_segments[end_idx + 1]
        current_segment = all_video_segments[end_idx]

        gap_duration = math.fabs(float(next_segment[1]) - float(current_segment[2]))
        if gap_duration > gap:
            break

        candidate_noun_vec = set(_flatten(next_segment[4]))
        candidate_verb_vec = set(_flatten(next_segment[5]))
        candidate_noun_vec_lemmas = set(_flatten(next_segment[6]))
        candidate_verb_vec_lemmas = set(_flatten(next_segment[7]))

        if not _has_overlap(
            gt_noun_vec,
            gt_verb_vec,
            gt_noun_vec_lemmas,
            gt_verb_vec_lemmas,
            candidate_noun_vec,
            candidate_verb_vec,
            candidate_noun_vec_lemmas,
            candidate_verb_vec_lemmas,
        ):
            break

        end_idx += 1

    return all_video_segments[start_idx][1], all_video_segments[end_idx][2]


def main(args):
    wandb.init(
        project="Thesis",
        name=f"Refine Dataset - Gap {args.gap}s",
        config={**args.__dict__},
        group=f"Refine Dataset",
    )

    ground_truth_data = load_data(args.gt)
    pseudo_labels_data = load_data(args.pseudolabels)

    total = len(ground_truth_data)
    total_pl = len(pseudo_labels_data)
    print(f"Total ground truth segments: {total}")
    print(f"Total pseudo-labels segments: {total_pl}")

    # Group videos by video_id
    print("Grouping videos by video_id...")
    ground_truth_video_groups = _group_videos_by_video_id(ground_truth_data)
    pseudo_labels_video_groups = _group_videos_by_video_id(pseudo_labels_data)

    # This will only be needed for the index resolution and clip expansion.
    merged_data = _merge_data(ground_truth_video_groups, pseudo_labels_video_groups)

    refined_data = []
    for i, data in tqdm(enumerate(ground_truth_data), desc="Refining data"):
        (
            video_id,
            gt_start,
            gt_end,
            gt_original_caption,
            gt_noun_vec,
            gt_verb_vec,
            gt_noun_vec_lemmas,
            gt_verb_vec_lemmas,
        ) = data

        # Resolve the idx of the ground truth segment in the pseudo-labels data
        new_start, new_end = _expand_clip(
            merged_data,
            video_id,
            gt_start,
            gt_end,
            gt_noun_vec,
            gt_verb_vec,
            gt_noun_vec_lemmas,
            gt_verb_vec_lemmas,
            args.gap,
        )

        refined_data.append((video_id, new_start, new_end, gt_original_caption))

        wandb.log({"progress": i / total})

    save_data(f"{args.out_path}/ego4d_train_refined_gap_{args.gap}.pkl", refined_data)

    # --- Added: Plot & log distributions and summaries (no changes to existing logic above) ---
    # Old (GT) and new (refined) segment durations
    old_durations = [
        float(end) - float(start) for (_, start, end, *_) in ground_truth_data
    ]
    new_durations = [float(end) - float(start) for (_, start, end, _) in refined_data]

    old = np.asarray(old_durations, dtype=float)
    new = np.asarray(new_durations, dtype=float)
    delta = new - old  # expansion amount

    # Overlapping histogram with log-scaled y-axis
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    ax.hist(old, bins=100, alpha=0.6, label="Old")
    ax.hist(new, bins=100, alpha=0.6, label="New")
    ax.set_yscale("log")
    ax.set_xlabel("Segment length (seconds)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title(f"Old vs New Segment Length Distribution (gap={args.gap}s)")
    ax.legend()

    # Log the figure to W&B
    wandb.log({"plots/length_distribution": wandb.Image(fig)})
    plt.close(fig)

    # Numeric summaries
    def safe_std(x):
        return float(x.std(ddof=1)) if x.size > 1 else 0.0

    summaries = {
        "counts/total": int(old.size),
        "changes/num_expanded": int((delta > 0).sum()),
        "changes/num_unchanged": int((delta == 0).sum()),
        "old/mean": float(old.mean()),
        "old/median": float(np.median(old)),
        "old/std": safe_std(old),
        "old/min": float(old.min()) if old.size else 0.0,
        "old/max": float(old.max()) if old.size else 0.0,
        "old/p25": float(np.percentile(old, 25)) if old.size else 0.0,
        "old/p75": float(np.percentile(old, 75)) if old.size else 0.0,
        "new/mean": float(new.mean()),
        "new/median": float(np.median(new)),
        "new/std": safe_std(new),
        "new/min": float(new.min()) if new.size else 0.0,
        "new/max": float(new.max()) if new.size else 0.0,
        "new/p25": float(np.percentile(new, 25)) if new.size else 0.0,
        "new/p75": float(np.percentile(new, 75)) if new.size else 0.0,
        "delta/mean": float(delta.mean()),
        "delta/median": float(np.median(delta)),
        "delta/p95": float(np.percentile(delta, 95)) if delta.size else 0.0,
    }
    wandb.log(summaries)

    # Log raw histograms as W&B histograms
    wandb.log(
        {
            "hist/old": wandb.Histogram(old),
            "hist/new": wandb.Histogram(new),
            "hist/delta": wandb.Histogram(delta),
        }
    )
    # --- End added section ---


if __name__ == "__main__":
    args = parse_args()
    main(args)
