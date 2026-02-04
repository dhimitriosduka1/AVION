import os
import math
import pickle as pkl
from tqdm import tqdm
from collections import defaultdict

from vllm import LLM, SamplingParams

# from qwen_vl_utils import process_vision_info

# --- HELPERS ---
# {
#     "role": "user",
#     "content": [
#         {"type": "text", "text": "Here are two videos from a security feed."},
#         {"type": "video", "video": video_1_path},
#         {"type": "video", "video": video_2_path},
#         {"type": "text", "text": "Compare the activity levels between the first and second video."}
#     ]
# }

# --- CONFIGURATION ---
DEFAULT_MODEL_PATH = "Qwen/Qwen3-VL-8B-Instruct"
ORIGINAL_DATA_PATH = "/ptmp/dduka/databases/ego4d/ego4d_train_with_uuid.pkl"
VIDEO_ROOT = "/ptmp/dduka/databases/ego4d/video_320px_15sec/"
CHUNK_LEN_SEC = 15.0
FPS = 8
MAX_PIXELS = 360 * 420
MINI_BATCH_SIZE = 2
MAX_VIDEO_CHUNKS = 4
MAX_MODEL_LEN = 120000

PROMPT = """You are an expert in video annotation quality control. Your task is to evaluate a sequence of consecutive caption segments from a video and determine whether they should be merged into a single caption.

**Input**:  
- A list of N caption segments, each with:
  - Start and end timestamps (in seconds)
  - A short descriptive caption
  - The corresponding video footage covering the entire span of these segments

**Decision Criteria**:  
MERGE the segments if **ALL** of the following are true:  
1. The described activity is **continuous and homogeneous** (e.g., walking, scrolling, watching, stirring, holding an object).  
2. There are **no discrete, countable sub-events** (e.g., no individual clicks, throws, spoken words, card drops, tool switches).  
3. The segmentation appears to result from **technical over-splitting** (e.g., fixed-duration windows, sliding windows) rather than human-annotated event boundaries.  
4. Merging would **not lose semantically important information** (e.g., quantity, sequence order, or distinct phases).

DO NOT MERGE if **ANY** of the following apply:  
- The captions imply **repetition of distinct instances** (e.g., “again”, “another”, “next”, “#1”, “then”).  
- The action involves **separable physical or logical units** (e.g., steps in a recipe, plays in sports, UI interactions).  
- Temporal gaps or overlaps suggest **intentional segmentation** of micro-events.  
- The total duration spans a **meaningful change in context** (e.g., subject, object, goal), even if wording is similar.

**Output Format**:  
Respond ONLY with valid JSON in this exact structure:
{{
    "merge": true|false,
    "reason": "Concise justification focusing on continuity vs. discreteness of actions.",
    "confidence": 0.0-1.0
}}

Do not include any other text, explanations, or formatting.

**Caption Segments**:
{segments}
"""


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


def remove_exact_duplicates(samples_by_video_id):
    print(f"Removing exact duplicates...")
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
    return samples_by_video_id


def generate_merge_candidates(samples_by_video_id):
    print("Generating merge candidates...")

    dataset = []
    merge_candidates = []
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
                    merge_candidates.append(
                        {
                            "video_id": video_id,
                            "history": list(history),
                            "merged_row": list(current_merged),
                        }
                    )
                else:
                    dataset.append(tuple(current_merged))

                current_merged, history = list(next_sample), [next_sample]

        if len(history) > 1:
            merge_candidates.append(
                {
                    "video_id": video_id,
                    "history": list(history),
                    "merged_row": list(current_merged),
                }
            )
        else:
            dataset.append(tuple(current_merged))

    print(
        f"Found {len(merge_candidates)} potential merges. {sum(len(s['history']) for s in merge_candidates)} samples involved."
    )
    print(
        f"Shortest merge candidate has {min(len(s['history']) for s in merge_candidates)} samples."
    )
    print(
        f"Longest merge candidate has {max(len(s['history']) for s in merge_candidates)} samples."
    )
    print(
        f"Average merge candidate has {sum(len(s['history']) for s in merge_candidates) / len(merge_candidates):.2f} samples."
    )

    return dataset, merge_candidates


def prepare_prompt(candidates):
    prompts = []
    for candidate in candidates:
        segments_as_str = []
        for idx, history_item in enumerate(candidate["history"]):
            segments_as_str.append(
                f'- {idx + 1}. [{history_item[2]:.2f}s - {history_item[3]:.2f}s]: "{history_item[4]}"'
            )

        segments_formatted = "\n".join(segments_as_str)
        prompt_filled = PROMPT.format(segments=segments_formatted)

        # Resolve the video path
        chunk_paths = get_video_chunks_for_segments(
            candidate["video_id"],
            candidate["merged_row"][2],
            candidate["merged_row"][3],
        )

        final_prompt = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_filled},
                *[{"type": "video", "video": path} for path in chunk_paths],
            ],
        }

        prompts.append(final_prompt)

    return prompts


def load_model():
    llm = LLM(
        model=DEFAULT_MODEL_PATH,
        tensor_parallel_size=1,
        limit_mm_per_prompt={"video": MAX_VIDEO_CHUNKS},
        max_model_len=MAX_MODEL_LEN,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.95,
        max_tokens=256,
        repetition_penalty=1.05,
    )
    return llm, tokenizer, sampling_params


def chunk_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


if __name__ == "__main__":
    llm, tokenizer, sampling_params = load_model()

    with open(ORIGINAL_DATA_PATH, "rb") as f:
        dataset = pkl.load(f)
        print(f"Loaded {len(dataset)} samples.")

    samples_by_video_id = defaultdict(list)
    for sample in dataset:
        samples_by_video_id[sample[1]].append(sample)

    samples_by_video_id = remove_exact_duplicates(samples_by_video_id)

    # Dataset does not contain the possible merge candidates. They are stored
    # in merge_candidates for further processing.
    dataset, merge_candidates = generate_merge_candidates(samples_by_video_id)

    for candidates in tqdm(chunk_list(merge_candidates, MINI_BATCH_SIZE)):
        prompts = prepare_prompt(candidates)

        prompts = tokenizer.apply_chat_template(
            prompts,
            tokenize=False,
            add_generation_prompt=True,
        )

        print(prompts)

        outputs = llm.chat(prompts, sampling_params=sampling_params)

        for candidate, output in zip(candidates, outputs):
            response_text = output.generations[0].text.strip()
            candidate["model_response"] = response_text

        print(f"Candidates processed: {candidates}")

        break
