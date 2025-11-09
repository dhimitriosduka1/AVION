import pickle
import argparse
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Refine GT segments by merging adjacent pseudo-label segments that match taxonomy anchors."
    )
    parser.add_argument(
        "--gt",
        default="/BS/dduka/work/projects/AVION/ego4d_train.pkl",
        help="Path to ground-truth pickle (e.g., ego4d_train.pkl)",
    )
    parser.add_argument(
        "--pseudolabels",
        default="/BS/dduka/work/projects/AVION/ego4d_train.uncovered_all.narrator_63690737.return_5.pkl",
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
        help="Gap between ground-truth and pseudo-label segments",
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
    for video_id, video_data in pseudo_labels_data.items():
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
    gt_noun_vec = set(gt_noun_vec)
    gt_verb_vec = set(gt_verb_vec)
    gt_noun_vec_lemmas = set(gt_noun_vec_lemmas)
    gt_verb_vec_lemmas = set(gt_verb_vec_lemmas)

    start_idx = idx
    while start_idx - 1 >= 0:
        prev_segment = all_video_segments[start_idx - 1]
        current_segment = all_video_segments[start_idx]

        gap_duration = current_segment[1] - prev_segment[2]
        if gap_duration > gap:
            break

        candidate_noun_vec = set(prev_segment[3])
        candidate_verb_vec = set(prev_segment[4])
        candidate_noun_vec_lemmas = set(prev_segment[5])
        candidate_verb_vec_lemmas = set(prev_segment[6])

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

        gap_duration = next_segment[1] - current_segment[2]
        if gap_duration > gap:
            break

        candidate_noun_vec = set(next_segment[3])
        candidate_verb_vec = set(next_segment[4])
        candidate_noun_vec_lemmas = set(next_segment[5])
        candidate_verb_vec_lemmas = set(next_segment[6])

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
    merged_data = _merge_data(ground_truth_data, pseudo_labels_data)

    refined_data = []
    for _, data in ground_truth_video_groups.items():
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

    save_data(f"{args.out_path}/ego4d_train_refined.pkl", refined_data)


if __name__ == "__main__":
    args = parse_args()
    main(args)
