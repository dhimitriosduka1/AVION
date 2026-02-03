import os
import math
import pickle as pkl
from tqdm import tqdm
from collections import defaultdict

from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

# --- CONFIGURATION ---
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
    return os.path.join(VIDEO_ROOT, f"{video_id}.mp4", f"{chunk_id}.mp4")


def chunk_id_from_time(t):
    return int(math.floor(t / CHUNK_LEN_SEC) * CHUNK_LEN_SEC)


def get_video_chunks_for_segments(video_id, start_time, end_time):
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
    return [
        {"type": "video", "video": path, "max_pixels": MAX_PIXELS, "fps": FPS}
        for path in video_paths
        if os.path.exists(path)
    ]


def prepare_vlm_input(candidate, tokenizer):
    """Formats the data for a single batch entry."""
    history = candidate["history"]
    video_id = candidate["video_id"]
    caption = str(history[0][4])

    start_times = [s[2] for s in history]
    end_times = [s[3] for s in history]
    merged_start, merged_end = min(start_times), max(end_times)

    segments_info = "\n".join(
        f"  Segment {i+1}: {s[2]:.2f}s to {s[3]:.2f}s" for i, s in enumerate(history)
    )
    video_chunks = get_video_chunks_for_segments(video_id, merged_start, merged_end)
    video_content = load_video_for_vlm(video_chunks)

    if not video_content:
        return None

    prompt_text = MERGE_DECISION_PROMPT.format(
        caption=caption,
        segments_info=segments_info,
        merged_start=merged_start,
        merged_end=merged_end,
    )

    messages = [
        {
            "role": "user",
            "content": video_content + [{"type": "text", "text": prompt_text}],
        }
    ]

    # Process inputs for vLLM
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    _, video_inputs = process_vision_info(messages, return_video_metadata=True)

    return {
        "prompt": prompt,
        "multi_modal_data": {"video": video_inputs},
    }


# --- MAIN EXECUTION ---

llm, tokenizer, sampling_params = load_model()

with open(ORIGINAL_DATA_PATH, "rb") as f:
    dataset = pkl.load(f)
    print(f"Loaded {len(dataset)} samples.")

# 1. Deduplication & Organization
samples_by_video_id = defaultdict(list)
for sample in dataset:
    samples_by_video_id[sample[1]].append(sample)

merge_candidates = []
processed_timeline = []

print("Analyzing timelines for merge candidates...")
for video_id, samples in tqdm(samples_by_video_id.items()):
    samples.sort(key=lambda x: x[2])
    if not samples:
        continue

    current_merged = list(samples[0])
    history = [samples[0]]

    for i in range(1, len(samples)):
        next_sample = samples[i]
        curr_cap = str(current_merged[4]).lower().strip()
        next_cap = str(next_sample[4]).lower().strip()

        if next_sample[2] <= current_merged[3] and curr_cap == next_cap:
            current_merged[3] = max(current_merged[3], next_sample[3])
            history.append(next_sample)
        else:
            if len(history) > 1:
                candidate = {
                    "video_id": video_id,
                    "history": list(history),
                    "merged_row": list(current_merged),
                    "decision": None,
                }
                merge_candidates.append(candidate)
                processed_timeline.append(candidate)
            else:
                processed_timeline.append(tuple(current_merged))

            current_merged, history = list(next_sample), [next_sample]

    # Final wrap up for video
    if len(history) > 1:
        candidate = {
            "video_id": video_id,
            "history": list(history),
            "merged_row": list(current_merged),
            "decision": None,
        }
        merge_candidates.append(candidate)
        processed_timeline.append(candidate)
    else:
        processed_timeline.append(tuple(current_merged))

# 3. Batch VLM Inference
print(f"Found {len(merge_candidates)} potential merges. Starting Batch Inference...")

# Prepare all inputs
batch_inputs = []
valid_candidates = []

for cand in tqdm(merge_candidates, desc="Preparing payloads"):
    payload = prepare_vlm_input(cand, tokenizer)
    if payload:
        batch_inputs.append(payload)
        valid_candidates.append(cand)
    else:
        cand["decision"] = "Unsure"  # Fallback if video is missing

# Run batch generation
if batch_inputs:
    outputs = llm.generate(batch_inputs, sampling_params=sampling_params)

    for i, output in enumerate(outputs):
        res = output.outputs[0].text.strip().lower()
        valid_candidates[i]["decision"] = "No" if "no" in res else "Yes"

# 4. Final Reconstruction
final_results = []
for item in processed_timeline:
    if isinstance(item, dict):  # This was a merge candidate
        if item["decision"] == "No":
            final_results.extend([tuple(s) for s in item["history"]])
        else:
            # "Yes" or "Unsure" -> keep merged
            final_results.append(tuple(item["merged_row"]))
    else:
        final_results.append(item)

# 5. Save Results
print(f"\nOriginal: {len(dataset)} | Final: {len(final_results)}")
OUTPUT_PATH = ORIGINAL_DATA_PATH.replace(".pkl", "_preprocessed_batch.pkl")

with open(OUTPUT_PATH, "wb") as f:
    pkl.dump(final_results, f)

print(f"Saved to {OUTPUT_PATH}")
