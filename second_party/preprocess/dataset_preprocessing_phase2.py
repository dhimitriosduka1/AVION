import torch
import pickle as pkl
import json
import re
from vllm import LLM, SamplingParams
from collections import defaultdict
from transformers import AutoTokenizer

# --- CONFIGURATION ---
MODEL_ID = "Qwen/Qwen3-32B"
MAX_MODEL_CONTEXT = 32768
GPU_COUNT = torch.cuda.device_count()

# Sliding Window Settings
WINDOW_SIZE = 80  # Number of captions per LLM call
OVERLAP_SIZE = 20  # How many captions to repeat in the next chunk

INPUT_DATA_PATH = (
    "/dais/fs/scratch/dduka/databases/ego4d/ego4d_train_deduplicated_with_uuid.pkl"
)
OUTPUT_DATA_PATH = (
    "/dais/fs/scratch/dduka/databases/ego4d/ego4d_train_timestamp_fixed.pkl"
)

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
Return ONLY a JSON array of objects:
```json
[
    {{"uuid": "original_uuid", "start": unified_start, "end": unified_end, "caption": "original_text"}},
    ...
]
```"""


def get_sliding_window_chunks(items, window_size, overlap):
    chunks = []
    if len(items) <= window_size:
        return [items]

    i = 0
    while i < len(items):
        chunk = items[i : i + window_size]
        chunks.append(chunk)
        i += window_size - overlap
        if i + overlap >= len(items):
            break
    return chunks


def main():
    print(f"Loading data from {INPUT_DATA_PATH}...")
    with open(INPUT_DATA_PATH, "rb") as f:
        dataset = pkl.load(f)  # Expected format: [UUID, VIDEO_ID, START, END, CAPTION]

    # Group by Video ID
    grouped_by_video = defaultdict(list)
    for row in dataset:
        grouped_by_video[row[1]].append(row)

    print(f"Initializing vLLM with {GPU_COUNT} GPUs...")
    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=GPU_COUNT,
        gpu_memory_utilization=0.92,
        max_model_len=MAX_MODEL_CONTEXT,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    all_prompts = []
    metadata = []

    print("Generating overlapping chunks...")
    for vid, items in grouped_by_video.items():
        items.sort(key=lambda x: x[2])  # Sort by start time
        chunks = get_sliding_window_chunks(items, WINDOW_SIZE, OVERLAP_SIZE)

        for chunk in chunks:
            # Format: [UUID | START | END] CAPTION
            caps_str = "\n".join([f"[{c[0]} | {c[2]} | {c[3]}] {c[4]}" for c in chunk])

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
            metadata.append(vid)

    all_prompts = all_prompts[:2]  # For testing, limit to 2 prompts

    print(f"Executing batch inference ({len(all_prompts)} prompts)...")
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=16384,
    )

    outputs = llm.generate(all_prompts, sampling_params)

    temporal_reconciliation = defaultdict(list)
    uuid_to_static_data = {}

    for i, output in enumerate(outputs):
        try:
            raw_text = output.outputs[0].text

            print(f"Raw LLM Output for Video {metadata[i]}:\n{raw_text}\n{'-'*50}")

            json_match = re.search(r"```json\s*(.*?)\s*```", raw_text, re.DOTALL)
            json_str = json_match.group(1) if json_match else raw_text
            chunk_results = json.loads(json_str)

            for entry in chunk_results:
                uid = entry["uuid"]
                temporal_reconciliation[uid].append(
                    (float(entry["start"]), float(entry["end"]))
                )
                # We update the map (assuming the text/vid_id stays the same as input)
                if uid not in uuid_to_static_data:
                    # Search original dataset or use chunk data
                    uuid_to_static_data[uid] = {
                        "vid": metadata[i],
                        "cap": entry["caption"],
                    }
        except Exception as e:
            continue

    # Create Final Dataset: [UUID, VIDEO_ID, START, END, CAPTION]
    final_dataset = []
    for uid, times in temporal_reconciliation.items():
        # Handle overlaps by taking the widest bounds suggested across windows
        final_start = min(t[0] for t in times)
        final_end = max(t[1] for t in times)

        static = uuid_to_static_data[uid]
        final_dataset.append(
            [uid, static["vid"], final_start, final_end, static["cap"]]
        )

    print(f"Saving {len(final_dataset)} rows to {OUTPUT_DATA_PATH}...")
    with open(OUTPUT_DATA_PATH, "wb") as f:
        pkl.dump(final_dataset, f)


if __name__ == "__main__":
    main()
