import os
import math
import pickle as pkl
from tqdm import tqdm
from collections import defaultdict

from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

DEFAULT_MODEL_PATH = "Qwen/Qwen3-VL-8B-Instruct"
ORIGINAL_DATA_PATH = "/ptmp/dduka/databases/ego4d/ego4d_train_with_uuid.pkl"
VIDEO_ROOT = "/ptmp/dduka/databases/ego4d/video_320px_15sec/"
CHUNK_LEN_SEC = 15.0
FPS = 8
MAX_PIXELS = 360 * 420

MERGE_DECISION_PROMPT = """
TASK: Determine if overlapping video segments with identical captions should be merged.

You are reviewing multiple video segments that have the same caption: "{caption}"

Segment timestamps:
{segments_info}

Proposed merged segment: {merged_start:.2f}s to {merged_end:.2f}s

ANALYSIS:
1. Watch the video covering these segments
2. Determine if all segments show the SAME continuous action or DIFFERENT instances of the action

DECISION CRITERIA:
- Answer "Yes" if: The segments show ONE continuous action that was incorrectly split
- Answer "No" if: The segments show SEPARATE instances of the same action
- Answer "Unsure" if: Cannot determine with confidence

OUTPUT (single word only): Yes, No, or Unsure
"""


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
    print(f"Initializing vLLM model: {DEFAULT_MODEL_PATH}...")

    llm = LLM(
        model=DEFAULT_MODEL_PATH,
        tensor_parallel_size=1,
        trust_remote_code=True,
        limit_mm_per_prompt={"video": 10},
        max_model_len=32768,
        gpu_memory_utilization=0.9,
    )

    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.9,
        max_tokens=16,
        repetition_penalty=1.05,
    )

    return llm, tokenizer, sampling_params


def get_chunk_path(video_id, chunk_id):
    """Get the path to a specific video chunk."""
    return os.path.join(VIDEO_ROOT, f"{video_id}.mp4", f"{chunk_id}.mp4")


def chunk_id_from_time(t):
    """Convert a timestamp to its corresponding chunk ID."""
    return int(math.floor(t / CHUNK_LEN_SEC) * CHUNK_LEN_SEC)


def get_video_chunks_for_segments(video_id, start_time, end_time):
    """Get all video chunk paths covering the given time range."""
    first_chunk = chunk_id_from_time(start_time)
    last_chunk = chunk_id_from_time(end_time)

    chunks = []
    curr = first_chunk
    while curr <= last_chunk:
        chunk_path = get_chunk_path(video_id, int(curr))
        if os.path.exists(chunk_path):
            chunks.append(chunk_path)
        curr += CHUNK_LEN_SEC

    return chunks


def load_video_for_vlm(video_paths):
    """Load video chunks and prepare them for VLM processing."""
    video_content = []
    for path in video_paths:
        if os.path.exists(path):
            video_content.append(
                {
                    "type": "video",
                    "video": path,
                    "max_pixels": MAX_PIXELS,
                    "fps": FPS,
                }
            )
    return video_content


def should_merge_segments(
    merge_history, video_id, llm=None, tokenizer=None, sampling_params=None
):
    """
    Use VLM to decide if overlapping segments should be merged.

    Args:
        merge_history: List of segments to potentially merge
        video_id: ID of the video
        llm: vLLM model instance
        tokenizer: Tokenizer instance
        sampling_params: Sampling parameters

    Returns:
        "Yes", "No", or "Unsure"
    """
    if llm is None or tokenizer is None or sampling_params is None:
        # Fallback: merge by default if no model available
        raise ValueError("LLM, tokenizer, and sampling_params must be provided.")

    # Extract segment info
    caption = str(merge_history[0][4])
    start_times = [s[2] for s in merge_history]
    end_times = [s[3] for s in merge_history]

    merged_start = min(start_times)
    merged_end = max(end_times)

    # Build segments info string
    segments_info = "\n".join(
        f"  Segment {i+1}: {s[2]:.2f}s to {s[3]:.2f}s"
        for i, s in enumerate(merge_history)
    )

    # Get video chunks covering the entire range
    video_chunks = get_video_chunks_for_segments(video_id, merged_start, merged_end)

    if not video_chunks:
        raise ValueError(f"No video chunks found for video ID: {video_id}")

    # Build prompt
    prompt_text = MERGE_DECISION_PROMPT.format(
        caption=caption,
        segments_info=segments_info,
        merged_start=merged_start,
        merged_end=merged_end,
    )

    # Build message with video content
    video_content = load_video_for_vlm(video_chunks)

    if not video_content:
        raise ValueError(f"Failed to load video for video ID: {video_id}")

    messages = [
        {
            "role": "user",
            "content": video_content + [{"type": "text", "text": prompt_text}],
        }
    ]

    # Apply chat template
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process video inputs
        _, video_inputs = process_vision_info(messages, return_video_metadata=True)

        # Run inference
        outputs = llm.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {"video": video_inputs},
            },
            sampling_params=sampling_params,
        )

        if outputs and outputs[0].outputs:
            response = outputs[0].outputs[0].text.strip().lower()

            # Parse response
            if "yes" in response:
                return "Yes"
            elif "no" in response:
                return "No"
            else:
                return "Unsure"
        else:
            return "Unsure"

    except Exception as e:
        print(f"  [ERROR] VLM inference failed: {e}")
        return "Unsure"


# Laod the model
llm, tokenizer, sampling_params = load_model()

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
                vlm_decision = should_merge_segments(
                    merge_history, video_id, llm, tokenizer, sampling_params
                )

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

        vlm_decision = should_merge_segments(
            merge_history, video_id, llm, tokenizer, sampling_params
        )

        if vlm_decision == "No":
            results.extend([tuple(s) for s in merge_history])
        else:
            # "Yes" or "Unsure" - keep merged segment
            results.append(tuple(current_merged))
    else:
        results.append(tuple(current_merged))

# # Summary statistics
# print(f"\n{'='*60}")
# print(f"Processing Complete!")
# print(f"{'='*60}")
# print(f"Original samples: {len(dataset)}")
# print(f"After deduplication: {total_samples_after_dedup}")
# print(f"Final results: {len(results)}")
# print(f"{'='*60}")

# # Save results
# OUTPUT_PATH = ORIGINAL_DATA_PATH.replace(".pkl", "_preprocessed.pkl")
# print(f"\nSaving results to: {OUTPUT_PATH}")
# with open(OUTPUT_PATH, "wb") as f:
#     pkl.dump(results, f)
# print("Done!")
