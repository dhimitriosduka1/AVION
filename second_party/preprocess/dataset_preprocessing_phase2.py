import torch
import pickle as pkl
from vllm import LLM, SamplingParams
from collections import defaultdict
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-32B"
MAX_MODEL_CONTEXT = 32768 * 4
GPU_COUNT = torch.cuda.device_count()

INPUT_DATA_PATH = (
    "/dais/fs/scratch/dduka/databases/ego4d/ego4d_train_deduplicated_with_uuid.pkl"
)

OUTPUT_DATA_PATH = (
    "/dais/fs/scratch/dduka/databases/ego4d/ego4d_train_timestamp_fixed.pkl"
)

PROMPT_TEXT = """### TASK: Semantic Grouping & Timestamp Alignment
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

4. **OUTPUT**: Return a valid JSON array of objects.

Output the JSON immediately after in the following format:
```json
[
    {
        "start": new_start_time_in_seconds or original_start_time_in_seconds,
        "end": new_end_time_in_seconds or original_end_time_in_seconds,
        "caption": "the original caption text"
    },
...
]
```
"""


def video_captions_to_message(samples):
    captions = []
    for item in samples:
        start_timestamp = item[1]
        end_timestamp = item[2]
        caption_text = item[3]
        captions.append(f"[{start_timestamp}, {end_timestamp}] {caption_text}")
    captions_str = "\n".join(captions)
    prompt = PROMPT_TEXT.replace("{captions_str}", captions_str)

    message = [
        {
            "role": "system",
            "content": (
                "You are an expert data curator for **Egocentric Video (First-Person)** datasets. "
                "Your task is to clean noisy training data where multiple annotators have described the **same atomic action** using slightly different words and slightly offset timestamps.\n\n"
                "**Your Goal:** Normalize these into a single temporal event while **preserving all unique text descriptions** as valuable linguistic augmentations."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    return message


if __name__ == "__main__":
    print(f"Loading data from {INPUT_DATA_PATH}...")
    with open(INPUT_DATA_PATH, "rb") as f:
        dataset = pkl.load(f)
    print(f"Loaded {len(dataset)} caption segments.")

    # Grouping
    grouped = defaultdict(list)
    for item in dataset:
        grouped[item[1]].append(item)

    video_ids = []
    for vid in grouped:
        grouped[vid].sort(key=lambda x: x[2])
        video_ids.append(vid)

    print(f"Initializing {MODEL_ID}...")
    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=GPU_COUNT,
        dtype="bfloat16",
        gpu_memory_utilization=0.96,
        trust_remote_code=True,
        max_model_len=MAX_MODEL_CONTEXT,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    video_id = list(grouped.keys())[0]

    message = video_captions_to_message(grouped[video_id])

    prompt = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=16384,
    )

    outputs = llm.generate([prompt], sampling_params)
    text = outputs[0].outputs[0].text

    print(text)
