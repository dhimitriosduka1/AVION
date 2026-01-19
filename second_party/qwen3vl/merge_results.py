import argparse
import glob
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Example of a saved sample when only 1 caption is used
# {
#     "uuid": "45ea2955-fea6-43b2-8eba-8d814681a111",
#     "video_id": "739c2ece-23c5-458e-9500-66bcef2f8945",
#     "rel_start": 5.910828673162996,
#     "rel_end": 6.528731326836805,
#     "global_start": 3215.910828673163,
#     "global_end": 3216.528731326837,
#     "caption": "#C C cuts cloth with scissors",
#     "base_offset": 3210.0,
#     "padding_used": 0.0,
#     "model_output": {
#         "scene_summary": "Person cutting fabric on a cluttered white table with scissors.",
#         "caption": "#C C cuts cloth with scissors",
#         "start": 5.91,
#         "end": 15.0,
#         "confidence": 0.95,
#         "evidence": [
#             "scissors",
#             "cloth"
#         ],
#         "notes": ""
#     }
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
# }


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

nr_high_confidence_samples = 0
nr_fallback_samples = 0
results = []
original_lengths = []
refined_lengths = []

for sample in tqdm(merged_results, desc="Processing rows"):
    sample_id = sample["uuid"]
    error = sample["model_output"].get("error", None)
    confidence = sample["model_output"].get(
        "confidence", 0
    )  # Default to 0 if missing (error case)

    # Calculate original segment length
    original_sample = original_captions_dict.get(sample_id)
    original_length = original_sample[3] - original_sample[2]  # end - start
    original_lengths.append(original_length)

    if error or confidence < 0.9:
        # We pick the original caption from the original dataset
        results.append(original_sample)
        refined_lengths.append(original_sample[3] - original_sample[2])
        nr_fallback_samples += 1
    else:
        base_offset = sample["base_offset"]
        start = base_offset + sample["model_output"]["start"]
        end = base_offset + sample["model_output"]["end"]

        if start > end:
            results.append(original_sample)
            refined_lengths.append(original_sample[3] - original_sample[2])
            nr_fallback_samples += 1
        else:
            refined_length = end - start
            refined_lengths.append(refined_length)
            results.append(
                (sample_id, sample["video_id"], start, end, sample["caption"])
            )
            nr_high_confidence_samples += 1


# Print stats about how the process went
print(f"\n{'='*60}")
print(f"MERGE RESULTS SUMMARY")
print(f"{'='*60}")
print(f"Total samples processed: {len(merged_results)}")
print(
    f"High confidence samples (>0.9): {nr_high_confidence_samples} ({100*nr_high_confidence_samples/len(merged_results):.2f}%)"
)
print(
    f"Fallback to original: {nr_fallback_samples} ({100*nr_fallback_samples/len(merged_results):.2f}%)"
)
print(f"Final results count: {len(results)}")
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
    pickle.dump(results, f)

with open(args.output_file.replace("_with_uuid", ""), "wb") as f:
    pickle.dump([(s[1], s[2], s[3], s[4]) for s in results], f)

print(f"Saved {len(results)} merged results to: {args.output_file}")
