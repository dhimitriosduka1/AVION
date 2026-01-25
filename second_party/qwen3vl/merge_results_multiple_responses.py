from sklearn.metrics import ConfusionMatrixDisplay
import argparse
import glob
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from collections import Counter

# Example of a saved sample when only 1 caption is used
# {
#     "uuid": "10c73452-7969-4a8d-ad4b-e0b18db04911",
#     "video_id": "2d5b5f4e-8854-4d60-afb6-ccaf9814d4fb",
#     "rel_start": 14.816963737944093,
#     "rel_end": 15.797293462055904,
#     "global_start": 269.8169637379441,
#     "global_end": 270.7972934620559,
#     "caption": "#C C turns the flame down",
#     "base_offset": 255.0,
#     "padding_used": 0,
#     "model_outputs": [
#         {
#             "scene_summary": "Person rolling dough on a black surface with a wooden roller, wearing a brown dress.",
#             "caption": "#C C turns the flame down",
#             "start": 14.82,
#             "end": 15.8,
#             "confidence": 0.0,
#             "evidence": [],
#             "notes": "The action #C C turns the flame down does not occur in the video.",
#         },
#         {
#             "scene_summary": "Person rolling dough on a flat surface with a rolling pin.",
#             "caption": "#C C turns the flame down",
#             "start": 14.82,
#             "end": 15.8,
#             "confidence": 0.0,
#             "evidence": [],
#             "notes": "The action 'turns the flame down' is not visible in the video frames provided.",
#         },
#         {
#             "scene_summary": "Person rolling dough on a flat surface with a rolling pin.",
#             "caption": "#C C turns the flame down",
#             "start": 14.82,
#             "end": 15.8,
#             "confidence": 0.0,
#             "evidence": [],
#             "notes": "Action not visible in the video frames provided.",
#         },
#         {
#             "scene_summary": "Person rolling dough on a flat surface with a rolling pin.",
#             "caption": "#C C turns the flame down",
#             "start": 14.82,
#             "end": 15.8,
#             "confidence": 0.0,
#             "evidence": [],
#             "notes": "The action described does not occur in the video.",
#         },
#         {
#             "scene_summary": "Person rolling dough on a flat surface with a rolling pin in a kitchen.",
#             "caption": "#C C turns the flame down",
#             "start": 14.82,
#             "end": 15.8,
#             "confidence": 0.0,
#             "evidence": [],
#             "notes": "The action described does not occur in the video.",
#         },
#         {
#             "scene_summary": "Person rolling dough on a flat surface with a rolling pin.",
#             "caption": "#C C turns the flame down",
#             "start": 14.82,
#             "end": 15.8,
#             "confidence": 0.0,
#             "evidence": [],
#             "notes": "Action not visible in provided frames.",
#         },
#         {
#             "scene_summary": "Person rolling dough on a flat surface with a rolling pin.",
#             "caption": "#C C turns the flame down",
#             "start": 14.82,
#             "end": 15.8,
#             "confidence": 0.0,
#             "evidence": [],
#             "notes": "The action #C C turns the flame down is not visible in the video frames provided.",
#         },
#         {
#             "scene_summary": "Person rolling dough on a flat surface with a rolling pin.",
#             "caption": "#C C turns the flame down",
#             "start": 14.82,
#             "end": 15.8,
#             "confidence": 0.0,
#             "evidence": [],
#             "notes": "Action not visible in the video frames provided.",
#         },
#         {
#             "scene_summary": "Person rolling dough on a black surface with a wooden roller, wearing a brown dress.",
#             "caption": "#C C turns the flame down",
#             "start": 14.82,
#             "end": 15.8,
#             "confidence": 0.0,
#             "evidence": [],
#             "notes": "The action 'turns the flame down' is not visible in the video frames provided.",
#         },
#         {
#             "scene_summary": "Person rolling dough on a flat surface with a wooden roller.",
#             "caption": "#C C turns the flame down",
#             "start": 14.82,
#             "end": 15.8,
#             "confidence": 0.0,
#             "evidence": [],
#             "notes": "The action described does not occur in the video.",
#         },
#     ],
# }

# Error sample
# {
#     "uuid": "74d936ce-6e8d-44c5-88c1-f8be96f1054b",
#     "video_id": "6c27a2fc-51b3-4502-85b5-d3c411b17965",
#     "rel_start": 11.842945435848833,
#     "rel_end": 12.63345456415118,
#     "global_start": 1151.8429454358488,
#     "global_end": 1152.6334545641512,
#     "caption": "#C C digs garden with shovel\n",
#     "base_offset": 1140.0,
#     "padding_used": 0.0,
#     "model_output": {
#         "raw_output": '{"scene_summary": "Person digging in garden soil with shovel, surrounded by green plants.", "caption": "#C C digs garden with shovel\n", "start": 12.0, "end": 13.0, "confidence": 0.95, "evidence": ["shovel", "garden soil"], "notes": ""}',
#         "error": "Model output not valid JSON",
#     },


def reward_temporal_iou(pred_interval, gt_interval):
    """
    Temporal IoU (tIoU) for 1D intervals.
    Returns 0 to 1 score where 1 = perfect overlap.
    """
    if pred_interval is None or gt_interval is None:
        return 0.0

    start1, end1 = pred_interval
    start2, end2 = gt_interval

    # Ensure start <= end
    if start1 > end1 or start2 > end2:
        return 0.0

    # Calculate intersection
    inter_start = max(start1, start2)
    inter_end = min(end1, end2)
    intersection = max(0, inter_end - inter_start)

    # Calculate union
    duration1 = end1 - start1
    duration2 = end2 - start2
    union = duration1 + duration2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def compute_temporal_iou_distance_matrix(intervals):
    """
    Compute pairwise temporal IoU distance matrix.
    Distance = 1 - IoU, so similar segments have low distance.

    Args:
        intervals: List of (start, end) tuples

    Returns:
        np.ndarray: Symmetric distance matrix of shape (n, n)
    """
    n = len(intervals)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            iou = reward_temporal_iou(intervals[i], intervals[j])
            distance = 1.0 - iou
            dist_matrix[i, j] = distance
            dist_matrix[j, i] = distance

    return dist_matrix


def cluster_temporal_segments(model_answers, distance_threshold=0.10):
    """
    Cluster temporal segments based on IoU similarity using Agglomerative Clustering.

    Args:
        model_answers: List of temporal interval strings like "(start, end)"
        distance_threshold: Maximum distance for merging clusters (1 - min_iou).
                           Default 0.3 means segments with IoU >= 0.7 will be merged.

    Returns:
        tuple: (cluster_labels, parsed_intervals, valid_indices)
            - cluster_labels: Array of cluster labels for valid intervals
            - parsed_intervals: List of parsed (start, end) tuples
            - valid_indices: List of original indices that had valid intervals
    """
    # Parse all intervals
    parsed_intervals = []
    valid_indices = []

    for idx, ans in enumerate(model_answers):
        interval = parse_temporal(ans)
        if interval is not None and interval[0] <= interval[1]:
            parsed_intervals.append(interval)
            valid_indices.append(idx)

    if len(parsed_intervals) == 0:
        return np.array([]), [], []

    if len(parsed_intervals) == 1:
        # Only one valid interval, it's its own cluster
        return np.array([0]), parsed_intervals, valid_indices

    # Compute IoU distance matrix
    dist_matrix = compute_temporal_iou_distance_matrix(parsed_intervals)

    # Perform agglomerative clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage="average",
    ).fit(dist_matrix)

    return clustering.labels_, parsed_intervals, valid_indices


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge results from refinement script outputs"
    )
    parser.add_argument(
        "--video-len-path",
        type=str,
        required=True,
        help="Path to the video length file",
    )
    parser.add_argument(
        "--json-path",
        type=str,
        required=True,
        help="Path to the JSON file",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path for the output file",
    )
    parser.add_argument(
        "--num-of-captions",
        type=int,
        required=True,
        help="Number of captions to process",
    )
    parser.add_argument(
        "--original-ego4d-path",
        type=str,
        required=True,
        help="The path of the original Ego4D pickle file",
    )
    return parser.parse_args()


args = parse_args()

# Load the video lengths
with open(args.video_len_path, "r") as f:
    video_len_dict = json.load(f)

# Find all jsonl files matching the pattern
pattern = os.path.join(args.json_path, f"output_{args.num_of_captions}_caption*.jsonl")
jsonl_files = sorted(glob.glob(pattern))

print(f"Found {len(jsonl_files)} JSONL files matching pattern: {pattern}")

merged_results = []
for jsonl_file in jsonl_files:
    print(f"Processing - {jsonl_file}")
    with open(jsonl_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                merged_results.append(record)

print(f"Loaded {len(merged_results)} rows in total")

print(f"Loading the original Ego4D pickle file")
with open(args.original_ego4d_path, "rb") as f:
    original_captions = pickle.load(f)

    original_captions_dict = {}
    for sample in original_captions:
        original_captions_dict[sample[0]] = sample

print(f"Loaded {len(original_captions)} rows in total")

final_results = []

original_lengths = []
refined_lengths = []
nr_fallback_samples = 0

for sample in tqdm(merged_results, desc="Processing rows"):

    sample_id = sample["uuid"]
    original_sample = original_captions_dict.get(sample_id)
    original_length = original_sample[3] - original_sample[2]
    original_lengths.append(original_length)

    responses = []
    for res in sample["model_outputs"]:
        confidence = res.get("confidence", 0)
        error = res.get("error", None)

        if error or confidence < 0.9:
            continue

        start = res["start"]
        end = res["end"]

        if start > end:
            continue

        if start < 0 or end < 0:
            continue

        responses.append((start, end))

    # Here I have all the valid responses
    estimated_label = None
    if len(responses) == 0:
        # No valid was found, fallback to original
        refined_lengths.append(original_sample[3] - original_sample[2])
        final_results.append(original_sample)
        nr_fallback_samples += 1
    elif len(responses) == 1:
        start = responses[0]
        end = responses[1]

        base_offset = sample["base_offset"]
        start = max(0.0, base_offset + start)
        end = min(
            base_offset + end,
            video_len_dict[sample["video_id"]],
        )

        if end > start:
            refined_length = end - start
            refined_lengths.append(refined_length)
            final_results.append(
                (sample_id, sample["video_id"], start, end, sample["caption"])
            )
        else:
            final_results.append(original_sample)
            refined_lengths.append(original_sample[3] - original_sample[2])
            nr_fallback_samples += 1
    else:
        # Compute distance matrix
        dist_matrix = compute_temporal_iou_distance_matrix(responses)

        # Cluster the results
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.9,
            metric="precomputed",
            linkage="average",
        ).fit(dist_matrix)

        cluster_counts = Counter(clustering.labels_)

        # Find majority cluster and its representative
        majority_cluster_id, majority_count = cluster_counts.most_common(1)[0]

        # Get representative interval from majority cluster (use centroid)
        majority_intervals = [
            responses[i]
            for i, label in enumerate(clustering.labels_)
            if label == majority_cluster_id
        ]
        avg_start = np.mean([iv[0] for iv in majority_intervals])
        avg_end = np.mean([iv[1] for iv in majority_intervals])
        estimated_label = (avg_start, avg_end)

        base_offset = sample["base_offset"]
        start = max(0.0, base_offset + estimated_label[0])
        end = min(
            base_offset + estimated_label[1],
            video_len_dict[sample["video_id"]],
        )

        if end > start:
            refined_length = end - start
            refined_lengths.append(refined_length)
            final_results.append(
                (sample_id, sample["video_id"], start, end, sample["caption"])
            )
        else:
            final_results.append(original_sample)
            refined_lengths.append(original_sample[3] - original_sample[2])
            nr_fallback_samples += 1

print(f"\n{'='*60}")
print(f"MERGE RESULTS SUMMARY")
print(f"{'='*60}")
print(f"Total samples processed: {len(final_results)}")
print(
    f"Fallback to original: {nr_fallback_samples} ({100 * nr_fallback_samples / len(final_results):.2f}%)"
)
print(f"{'='*60}")

if original_lengths and refined_lengths:
    print(
        f"\nOriginal segment lengths - Mean: {np.mean(original_lengths):.2f}s, Std: {np.std(original_lengths):.2f}s"
    )
    print(
        f"Refined segment lengths  - Mean: {np.mean(refined_lengths):.2f}s, Std: {np.std(refined_lengths):.2f}s"
    )


# Generate and save two graphs next to each other showing the original and the modified distribution of the segment lengths
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Original distribution
axes[0].hist(original_lengths, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
axes[0].set_xlabel("Segment Length (seconds)")
axes[0].set_ylabel("Frequency")
axes[0].set_title(
    f"Original Segment Length Distribution\n(Mean: {np.mean(original_lengths):.2f}s, N={len(original_lengths)})"
)
axes[0].axvline(
    np.mean(original_lengths),
    color="red",
    linestyle="--",
    label=f"Mean: {np.mean(original_lengths):.2f}s",
)
axes[0].set_yscale("log")
axes[0].legend()

# Refined distribution
axes[1].hist(
    refined_lengths, bins=50, color="forestgreen", edgecolor="black", alpha=0.7
)
axes[1].set_xlabel("Segment Length (seconds)")
axes[1].set_ylabel("Frequency")
axes[1].set_title(
    f"Refined Segment Length Distribution\n(Mean: {np.mean(refined_lengths):.2f}s, N={len(refined_lengths)})"
)
axes[1].axvline(
    np.mean(refined_lengths),
    color="red",
    linestyle="--",
    label=f"Mean: {np.mean(refined_lengths):.2f}s",
)
axes[1].set_yscale("log")
axes[1].legend()

plt.tight_layout()

# Save the figure
output_dir = os.path.dirname(args.output_file)
fig_path = os.path.join(
    output_dir, f"segment_length_comparison_{args.num_of_captions}_caption.png"
)
plt.savefig(fig_path, dpi=300)
print(f"\nSaved comparison figure to: {fig_path}")
plt.close()

# Save results to output file
with open(args.output_file, "wb") as f:
    pickle.dump(final_results, f)

with open(args.output_file.replace("_with_uuid", ""), "wb") as f:
    pickle.dump([(s[1], s[2], s[3], s[4]) for s in final_results], f)

print(f"Saved {len(final_results)} merged results to: {args.output_file}")
