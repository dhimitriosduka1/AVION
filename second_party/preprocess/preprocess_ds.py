import pickle as pkl
from tqdm import tqdm
from collections import defaultdict

ORIGINAL_DATA_PATH = "/ptmp/dduka/databases/ego4d/ego4d_train_with_uuid.pkl"
PROMPT = ""


def print_debug_info(video_id, merged_result, history):
    """Prints a detailed trace of merged timestamps and captions."""
    print(f"\n[DEBUG] Merge Event in Video: {video_id}")
    for idx, s in enumerate(history):
        print(f"  Part {idx+1:2} | {s[2]:>7.2f}s - {s[3]:>7.2f}s | Caption: '{s[4]}'")

    print(f"  {'='*15} RESULT {'='*15}")
    print(
        f"  FINAL   | {merged_result[2]:>7.2f}s - {merged_result[3]:>7.2f}s | Caption: '{merged_result[4]}'"
    )
    print("-" * 60)


def load_model():
    # Placeholder for loading an external model
    pass


def should_merge_segments(merge_history):
    # Placeholder for model-based decision
    # Returns "Yes", "No", or "Unsure"
    # For now, we always merge
    return "Yes"


with open(ORIGINAL_DATA_PATH, "rb") as f:
    dataset = pkl.load(f)
    print(f"Loaded dataset with {len(dataset)} samples.")

samples_by_video_id = defaultdict(list)
for sample in dataset:
    video_id = sample[1]
    samples_by_video_id[video_id].append(sample)

for video_id, samples in samples_by_video_id.items():
    samples.sort(key=lambda x: x[2])

print(f"Organized samples into {len(samples_by_video_id)} unique video IDs.")

for video_id, samples in samples_by_video_id.items():
    seen = set()
    unique_samples = []

    for row in samples:
        identifier = (row[1], row[2], row[3], row[4])

        if identifier not in seen:
            unique_samples.append(row)
            seen.add(identifier)

    samples_by_video_id[video_id] = unique_samples

total_samples_after_dedup = sum(
    len(samples) for samples in samples_by_video_id.values()
)
print(f"Removed {len(dataset) - total_samples_after_dedup} duplicate samples.")
print(f"Dataset contains {total_samples_after_dedup} samples after deduplication.")

# Resolve overlapping captions with identical text
results = []

for video_id, samples in tqdm(samples_by_video_id.items()):
    if not samples:
        continue

    # 1. Sort samples by start time (Index 2)
    samples.sort(key=lambda x: x[2])

    # Initialize the 'active' segment
    current_merged = list(samples[0])
    merge_history = [samples[0]]

    for i in range(1, len(samples)):
        next_sample = samples[i]

        # Normalize captions for comparison
        curr_cap_norm = str(current_merged[4]).lower().strip()
        next_cap_norm = str(next_sample[4]).lower().strip()

        # Check for overlap AND normalized caption match
        if next_sample[2] <= current_merged[3] and curr_cap_norm == next_cap_norm:
            current_merged[3] = max(current_merged[3], next_sample[3])
            merge_history.append(next_sample)
        else:
            # Chain broken: Print debug if it was a multi-merge
            if len(merge_history) > 1:
                print_debug_info(video_id, current_merged, merge_history)

                # Here I need to decide if the segments should be merged or not
                vlm_decision = should_merge_segments(merge_history)

                if vlm_decision == "No":
                    # Revert to original segments if not merging
                    results.extend([tuple(s) for s in merge_history])
                else:
                    # "Yes" or "Unsure" - keep merged segment
                    results.append(tuple(current_merged))
            else:
                results.append(tuple(current_merged))

            current_merged = list(next_sample)
            merge_history = [next_sample]

    # Final segment check
    if len(merge_history) > 1:
        print_debug_info(video_id, current_merged, merge_history)

        vlm_decision = should_merge_segments(merge_history)

        if vlm_decision == "No":
            results.extend([tuple(s) for s in merge_history])
        else:
            # "Yes" or "Unsure" - keep merged segment
            results.append(tuple(current_merged))
    else:
        results.append(tuple(current_merged))
