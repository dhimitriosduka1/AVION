import torch
import pickle as pkl
import json
import re
import os
import argparse
from vllm import LLM, SamplingParams
from collections import defaultdict
from transformers import AutoTokenizer

# --- CONFIGURATION ---
MODEL_ID = "Qwen/Qwen3-32B"
MAX_MODEL_CONTEXT = 32768
GPU_COUNT = torch.cuda.device_count()

WINDOW_SIZE = 60
OVERLAP_SIZE = 15
BATCH_SIZE = 128

INPUT_DATA_PATH = (
    "/dais/fs/scratch/dduka/databases/ego4d/ego4d_train_deduplicated_with_uuid.pkl"
)

OUTPUT_DIR = "/dais/fs/scratch/dduka/databases/ego4d/shards/"

# --- PROMPTS ---
PROMPT_SYSTEM = """You are an expert data curator for **Egocentric Video (First-Person)** datasets.
Your task is to clean noisy training data where multiple annotators have described the **same atomic action** using slightly different words and slightly offset timestamps.

**Your Goal:** Normalize these into a single temporal event while **preserving all unique text descriptions** as valuable linguistic augmentations."""

PROMPT_USER_TEMPLATE = """### TASK: Semantic Grouping & Timestamp Alignment
**CONTEXT**: 
The input contains captions from an egocentric video (camera wearer "#C"). 
Currently, the data is noisy: the same action (e.g., "#C picks up the cup") is listed multiple times with slight timestamp variations and synonym changes (e.g., "#C takes the cup").
This hurts training. We need to **merge** these into single events.

**INPUT CAPTIONS** (Format: [start, end] #C caption):
{captions_str}

The captions are sorted by start time, but may have significant overlaps.

### INSTRUCTIONS:

1. **IDENTIFY CLUSTERS**: Group captions that describe the **EXACT SAME VISUAL MOMENT**.
   - **Temporal overlap**: They must overlap significantly or be nearly consecutive.
   - **Semantic identity**: They must describe the **same atomic interaction** (e.g., "opens fridge" vs "pulls fridge door").
   - **Ignore Formatting**: Ignore differences in capitalization, punctuation, or "#C " prefix.

2. **COMPUTE UNION**: For each cluster:
   - `new_start` = Minimum start time of the group.
   - `new_end` = Maximum end time of the group.

3. **PRESERVE TEXT**: 
   - Do NOT delete any captions. 
   - Do NOT summarize text.
   - Every input caption must appear in the output, but with the **new unified timestamps** of its cluster.

**OUTPUT FORMAT**:
[["uuid_1", "uuid_2"], ["uuid_3"]]"""


def get_sliding_window_chunks(items, window_size, overlap):
    chunks = []
    i = 0
    while i < len(items):
        chunks.append(items[i : i + window_size])
        if i + window_size >= len(items):
            break
        i += window_size - overlap
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job_idx", type=int, default=0, help="Index of the current job"
    )
    parser.add_argument(
        "--num_jobs", type=int, default=1, help="Total number of jobs in array"
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading data...")
    with open(INPUT_DATA_PATH, "rb") as f:
        dataset = pkl.load(f)

    # Group by video
    grouped_by_video = defaultdict(list)
    for row in dataset:
        grouped_by_video[row[1]].append(row)

    # Sort video IDs to ensure consistency across all parallel jobs
    all_video_ids = sorted(list(grouped_by_video.keys()))

    # --- SUBSET LOGIC FOR JOB ARRAY ---
    # Divide the video list into N chunks
    chunk_size = (len(all_video_ids) + args.num_jobs - 1) // args.num_jobs
    start_idx = args.job_idx * chunk_size
    end_idx = min(start_idx + chunk_size, len(all_video_ids))

    my_video_ids = all_video_ids[start_idx:end_idx]
    print(f"Job {args.job_idx}/{args.num_jobs}: Processing {len(my_video_ids)} videos.")

    # Only keep data for videos assigned to this job
    row_lookup = {
        row[0]: list(row) for vid in my_video_ids for row in grouped_by_video[vid]
    }

    print(f"Initializing vLLM...")
    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=GPU_COUNT,
        gpu_memory_utilization=0.90,
        max_model_len=MAX_MODEL_CONTEXT,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    all_prompts = []
    for vid in my_video_ids:
        items = grouped_by_video[vid]
        items.sort(key=lambda x: x[2])  # Sort by start time
        chunks = get_sliding_window_chunks(items, WINDOW_SIZE, OVERLAP_SIZE)
        for chunk in chunks:
            caps_str = "\n".join(
                [f"ID: {c[0]} | [{c[2]}-{c[3]}] | {c[4]}" for c in chunk]
            )
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": PROMPT_SYSTEM},
                    {
                        "role": "user",
                        "content": PROMPT_USER_TEMPLATE.format(captions_str=caps_str),
                    },
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            all_prompts.append(prompt)

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=16384,
    )

    print(f"Generating for {len(all_prompts)} total windows...")
    for i in range(0, len(all_prompts), BATCH_SIZE):
        batch = all_prompts[i : i + BATCH_SIZE]
        outputs = llm.generate(batch, sampling_params, use_tqdm=True)

        print(f"Batch: {batch}")

        for output in outputs:
            raw_text = output.outputs[0].text
            clean_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
            match = re.search(r"(\[[\s\S]*\])", clean_text)

            print(f"Raw text: {raw_text}")
            print(f"Match: {match}")

            if match:
                try:
                    clusters = json.loads(match.group(1))
                    for cluster in clusters:
                        valid_uids = [uid for uid in cluster if uid in row_lookup]
                        if not valid_uids:
                            continue

                        u_start = min(row_lookup[uid][2] for uid in valid_uids)
                        u_end = max(row_lookup[uid][3] for uid in valid_uids)

                        for uid in valid_uids:
                            row_lookup[uid][2] = u_start
                            row_lookup[uid][3] = u_end
                except:
                    continue

    # Save specific shard
    shard_path = os.path.join(OUTPUT_DIR, f"shard_{args.job_idx}.pkl")
    final_dataset = list(row_lookup.values())
    with open(shard_path, "wb") as f:
        pkl.dump(final_dataset, f)

    print(f"Shard {args.job_idx} saved with {len(final_dataset)} rows.")


if __name__ == "__main__":
    main()
