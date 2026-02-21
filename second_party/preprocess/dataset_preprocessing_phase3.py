import os
import re
import json
import torch
import pickle as pkl
from collections import defaultdict
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# --- Configuration ---
SCRATCH_CACHE_DIR = "/dais/fs/scratch/dduka/huggingface_cache"
os.environ["HF_HOME"] = SCRATCH_CACHE_DIR
os.environ["HF_HUB_CACHE"] = SCRATCH_CACHE_DIR
os.environ["TMPDIR"] = SCRATCH_CACHE_DIR

MODEL_ID = "Qwen/Qwen3-30B-A3B-Thinking-2507"
MAX_MODEL_CONTEXT = 32768
GPU_COUNT = torch.cuda.device_count()

INPUT_DATA_PATH = (
    "/dais/fs/scratch/dduka/databases/ego4d/ego4d_train_deduplicated_with_uuid.pkl"
)
VIDEO_LENGTHS_PATH = "/dais/fs/scratch/dduka/databases/ego4d/video_lengths.json"
OUTPUT_DIR = "/dais/fs/scratch/dduka/databases/ego4d/shards/"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SCRATCH_CACHE_DIR, exist_ok=True)

# --- Prompts ---
SYSTEM_PROMPT = """You are an expert egocentric video action parser. 
Process the video segments and group IDs that describe the EXACT SAME atomic action.

THINKING STEPS:
1. For every pair of segments, calculate the temporal gap: max(0, start_B - end_A).
2. If gap > 1.0s, they MUST remain separate.
3. Check if different nouns (e.g., 'sieve' vs 'drainer') refer to the same object in context.

OUTPUT FORMAT:
You must provide your internal reasoning inside <think></think> tags, followed by a JSON block.
{
    "reasoning": "Brief summary of merge decisions.",
    "clusters": [[0], [1, 2], [3]]
}"""

USER_PROMPT_TEMPLATE = """# Video Metadata
Duration: {video_duration}s
Window: {start_time} - {end_time}s

# Input Segments
[id, start_time, end_time, "caption"]
{formatted_input_list}"""


def parse_qwen_thinking_output(text):
    """Extracts internal monologue and JSON from Qwen3 Thinking models."""
    # 1. Extract reasoning from <think> tags
    thought_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    internal_thought = (
        thought_match.group(1).strip() if thought_match else "No thought trace found."
    )

    # 2. Clean the remaining text to find JSON
    clean_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    clean_text = re.sub(
        r"```json\s*|\s*```", "", clean_text
    )  # Remove markdown wrappers

    parsed_clusters = []
    reasoning_summary = ""

    try:
        # Attempt direct JSON load
        data = json.loads(clean_text)
        parsed_clusters = data.get("clusters", [])
        reasoning_summary = data.get("reasoning", internal_thought)
    except json.JSONDecodeError:
        # Fallback for messy outputs: find the first { and last }
        try:
            start_idx = clean_text.find("{")
            end_idx = clean_text.rfind("}")
            if start_idx != -1 and end_idx != -1:
                json_str = clean_text[start_idx : end_idx + 1]
                data = json.loads(json_str)
                parsed_clusters = data.get("clusters", [])
                reasoning_summary = data.get("reasoning", internal_thought)
        except Exception:
            reasoning_summary = "Parsing Error."

    return reasoning_summary, internal_thought, parsed_clusters


if __name__ == "__main__":
    print(f"🦍 Initializing Qwen3 Pipeline: {MODEL_ID}")

    # Load Metadata
    with open(VIDEO_LENGTHS_PATH, "r") as f:
        video_lengths = json.load(f)
    with open(INPUT_DATA_PATH, "rb") as f:
        dataset = pkl.load(f)

    # Grouping
    grouped_by_video = defaultdict(list)
    for row in dataset:
        grouped_by_video[row[1]].append(row)

    # Model Setup
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=GPU_COUNT,
        enable_prefix_caching=True,  # Critical for overlapping windows
        gpu_memory_utilization=0.90,
        max_model_len=MAX_MODEL_CONTEXT,
        trust_remote_code=True,
    )

    # Note: increased max_tokens to 8192 to allow for the model's 'thinking' trace
    sampling_params = SamplingParams(
        temperature=0.0, max_tokens=8192, repetition_penalty=1.05
    )

    all_prompts = []
    validation_tracker = []
    WINDOW_SIZE, OVERLAP = 20, 5
    step = WINDOW_SIZE - OVERLAP

    # Generate Windows for first 10 videos
    for v_idx, (test_video_id, raw_segments) in enumerate(grouped_by_video.items()):
        if v_idx >= 10:
            break

        segments = sorted(raw_segments, key=lambda x: x[2])
        video_dur = video_lengths.get(test_video_id, "Unknown")

        for i in range(0, len(segments), step):
            window_indices = list(range(i, min(i + WINDOW_SIZE, len(segments))))
            if not window_indices:
                continue

            formatted_lines = [
                f'[{idx}, {segments[idx][2]}, {segments[idx][3]}, "{segments[idx][4].replace("#C C ", "").replace("#C ", "").strip()}"]'
                for idx in window_indices
            ]

            user_content = USER_PROMPT_TEMPLATE.format(
                video_duration=video_dur,
                start_time=segments[window_indices[0]][2],
                end_time=segments[window_indices[-1]][3],
                formatted_input_list="\n".join(formatted_lines),
            )

            chat_prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

            all_prompts.append(chat_prompt)
            validation_tracker.append(
                {"video_id": test_video_id, "input": "\n".join(formatted_lines)}
            )

    # Inference
    print(f"🦍 Processing {len(all_prompts)} windows...")
    outputs = llm.generate(all_prompts, sampling_params)

    # Reporting
    print("\n" + "=" * 80 + "\nQWEN3 THINKING VALIDATION\n" + "=" * 80)
    for output, meta in zip(outputs, validation_tracker):
        raw_text = output.outputs[0].text
        summ, thought, clusters = parse_qwen_thinking_output(raw_text)

        print(f"\nVIDEO: {meta['video_id']}")
        print(f"PROMPT INPUT:\n{meta['input']}")
        print(f"THOUGHT TRACE (Snippet): {thought[:200]}...")
        print(f"SUMMARY: {summ}")
        print(f"CLUSTERS: {clusters}")
        print("-" * 40)
