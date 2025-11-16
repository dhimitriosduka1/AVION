import pickle
import argparse
from collections import defaultdict
from tqdm import tqdm
import wandb
import plotly.express as px


def parse_args():
    parser = argparse.ArgumentParser(
        description="Refine GT segments by merging with pseudo-labels based on IoU"
    )
    parser.add_argument(
        "--dataset",
        default="/dais/fs/scratch/dduka/databases/ego4d/ego4d_train_enriched.pkl",
        help="Path to ground-truth pickle (e.g., ego4d_train.pkl)",
    )
    parser.add_argument(
        "--pseudolabels",
        default="/dais/fs/scratch/dduka/databases/ego4d/ego4d_train_uncovered_all.narrator_63690737.return_5_enriched.pkl",
        help="Path to pseudo-label pickle",
    )
    parser.add_argument(
        "--out-path",
        default="/dais/fs/scratch/dduka/databases/ego4d/iou_nouns_and_verbs/",
        help="Desired output path",
    )
    parser.add_argument(
        "--min-iou",
        default=0.01,
        type=float,
        help="Minimum IoU threshold for merging segments",
    )
    parser.add_argument(
        "--postfix",
        type=str,
        help="Postfix for wandb logging",
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
        video_groups[video_id].append(sample)
    for video_id in video_groups:
        video_groups[video_id].sort(key=lambda x: float(x[1]))
    return video_groups


def _flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(_flatten(item))
        else:
            result.append(item)
    return result


def _merge(source, target):
    merged_data = {}
    for video_id, segments in source.items():
        merged_data[video_id] = segments + target[video_id]

    for video_id in merged_data:
        merged_data[video_id].sort(key=lambda x: float(x[1]))

    return merged_data


def _compute_iou(seg_a, seg_b):
    start_a, end_a = float(seg_a[1]), float(seg_a[2])
    start_b, end_b = float(seg_b[1]), float(seg_b[2])

    intersection_start = max(start_a, start_b)
    intersection_end = min(end_a, end_b)
    intersection = max(0.0, intersection_end - intersection_start)

    union = (end_a - start_a) + (end_b - start_b) - intersection

    if union == 0:
        return 0.0

    iou = intersection / union
    return iou


def _has_overlap(
    gt_nouns,
    gt_verbs,
    gt_noun_lemmas,
    gt_verb_lemmas,
    cand_nouns,
    cand_verbs,
    cand_noun_lemmas,
    cand_verb_lemmas,
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

    has_noun_overlap = bool(nouns_to_check_gt & nouns_to_check_cand)
    has_verb_overlap = bool(verbs_to_check_gt & verbs_to_check_cand)

    return has_noun_overlap and has_verb_overlap


def main(args):
    wandb.init(
        project="Thesis",
        name=(
            f"IoU {args.min_iou} - Noun and Verbs {args.postfix}"
            if args.postfix
            else f"IoU {args.min_iou} Noun and Verbs"
        ),
        config={**args.__dict__},
        group=f"Refine Dataset IoU",
    )

    ground_truth_data = load_data(args.dataset)
    pseudo_labels_data = load_data(args.pseudolabels)

    total = len(ground_truth_data)
    total_pl = len(pseudo_labels_data)
    print(f"Total ground truth segments: {total}")
    print(f"Total pseudo-labels segments: {total_pl}")

    print("Grouping videos by video_id...")
    ground_truth_video_groups = _group_videos_by_video_id(ground_truth_data)
    pseudo_labels_video_groups = _group_videos_by_video_id(pseudo_labels_data)

    print("Merging original and pseudo-labels...")
    data = _merge(ground_truth_video_groups, pseudo_labels_video_groups)

    results = []
    old_segments = []
    new_segments = []
    for video_id, segments in tqdm(data.items(), desc="Refining segments"):
        for idx, segment in enumerate(segments):
            # The ground truth segments have length 8
            if len(segment) == 8:
                start = float(segment[1])
                end = float(segment[2])

                old_segments.append(end - start)

                gt_nouns = set(_flatten(segment[4]))
                gt_verbs = set(_flatten(segment[5]))
                gt_noun_lemmas = set(_flatten(segment[6]))
                gt_verb_lemmas = set(_flatten(segment[7]))

                if idx > 0:
                    # Check if the left segment is a pseudo-label
                    left_segment = segments[idx - 1]
                    if len(left_segment) == 9:
                        left_nouns = set(_flatten(left_segment[5]))
                        left_verbs = set(_flatten(left_segment[6]))
                        left_noun_lemmas = set(_flatten(left_segment[7]))
                        left_verb_lemmas = set(_flatten(left_segment[8]))
                        iou = _compute_iou(segment, left_segment)
                        if iou > args.min_iou and _has_overlap(
                            gt_nouns,
                            gt_verbs,
                            gt_noun_lemmas,
                            gt_verb_lemmas,
                            left_nouns,
                            left_verbs,
                            left_noun_lemmas,
                            left_verb_lemmas,
                        ):
                            start = min(start, float(left_segment[1]))

                if idx < len(segments) - 1:
                    # Check if the right segment is a pseudo-label
                    right_segment = segments[idx + 1]
                    if len(right_segment) == 9:
                        right_nouns = set(_flatten(right_segment[5]))
                        right_verbs = set(_flatten(right_segment[6]))
                        right_noun_lemmas = set(_flatten(right_segment[7]))
                        right_verb_lemmas = set(_flatten(right_segment[8]))
                        iou = _compute_iou(segment, right_segment)
                        if iou > args.min_iou and _has_overlap(
                            gt_nouns,
                            gt_verbs,
                            gt_noun_lemmas,
                            gt_verb_lemmas,
                            right_nouns,
                            right_verbs,
                            right_noun_lemmas,
                            right_verb_lemmas,
                        ):
                            end = max(end, float(right_segment[2]))

                results.append((video_id, start, end, segment[3]))
                new_segments.append(end - start)

    print(f"Len of original ds: {len(ground_truth_data)}")
    print(f"Len of refined ds: {len(results)}")

    with open(f"{args.out_path}/ego4d_train_iou_{args.min_iou}.pkl", "wb") as f:
        pickle.dump(results, f)

    fig_old = px.histogram(
        x=old_segments,
        nbins=100,
        title="Original Distribution",
        labels={"x": "Length (seconds)", "y": "Frequency"},
        log_y=True,
    )
    fig_old.update_layout(bargap=0)

    fig_new = px.histogram(
        x=new_segments,
        nbins=100,
        title="New Distribution",
        labels={"x": "Length (seconds)", "y": "Frequency"},
        log_y=True,
    )
    fig_new.update_layout(bargap=0)

    wandb.log(
        {
            "Original Segment Distribution": fig_old,
            "New Segment Distribution": fig_new,
        }
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
