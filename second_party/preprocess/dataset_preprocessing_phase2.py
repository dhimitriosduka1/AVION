import torch
import pickle as pkl
import json
from collections import defaultdict
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# --- Configuration ---
MODEL_ID = "openai/gpt-oss-120b"
MAX_MODEL_CONTEXT = 32768
GPU_COUNT = torch.cuda.device_count()

INPUT_DATA_PATH = (
    "/dais/fs/scratch/dduka/databases/ego4d/ego4d_train_deduplicated_with_uuid.pkl"
)
VIDEO_LENGTHS_PATH = "/dais/fs/scratch/dduka/databases/ego4d/video_lengths.json"
OUTPUT_DIR = "/dais/fs/scratch/dduka/databases/ego4d/shards/"

PROMPT_TEMPLATE_PHASE_1 = """
# Video Metadata
Total Duration: {video_duration} seconds
Domain: Egocentric (First-Person Vision)
Current Segment: {start_time} - {end_time} seconds
Current Number of Captions: {num_captions}

# Input Segments
The following is a chronologically sorted list of segments from an egocentric video. 
Format: [id, start_time, end_time, "caption"]
{formatted_input_list}

# Your Task
Your task is to process this list of egocentric video segments and group the IDs of segments that describe the exact same **atomic action**. 

Crucial Context: The captions provided are manually annotated, providing a strong semantic baseline.\n
However, their start and end timestamps are computed using a heuristic. This combination can result in\n
artificial temporal overlaps, fragmented boundaries, and concurrent segments describing the same underlying\n
action using natural human lexical variation (e.g., "chopping tomato" vs. "slicing a red vegetable").

You must group the segment IDs into discrete clusters, where each cluster represents a single, distinct atomic action.

Guidelines and requirements:
- **Domain Context:** Assume a first-person perspective (the camera wearer). Focus primarily on hand-object interactions.
- **Semantic & Temporal Clustering:** Evaluate both temporal proximity (overlapping or adjacent heuristic boundaries) and semantic similarity.\n
Group IDs together if they describe the same underlying action, accommodating for human subjective differences in the annotations.
- **Lexical Resolution:** Look past superficial lexical differences caused by different annotators describing the same event at the same time.
- **Completeness and Exclusivity:** Every ID from the input list MUST be included in exactly one cluster. Do not drop any IDs, and do not place\n
the same ID into multiple clusters. If an ID represents a standalone action that shouldn't be merged, it should be in a cluster by itself (e.g., `[4]`).
- **Reasoning First:** Think step-by-step. Analyze temporal overlaps and semantic similarities before outputting the final JSON.

# Response Format
Return a valid JSON object strictly adhering to this schema:

{
  "type": "object",
  "properties": {
    "reasoning": {
      "type": "string",
      "description": "A brief, step-by-step explanation of how the IDs were clustered based on heuristic temporal overlap and lexical variation."
    },
    "clusters": {
      "type": "array",
      "description": "A list of lists. Each inner list contains the integer IDs of the segments that should be merged together.",
      "items": {
        "type": "array",
        "items": {
          "type": "integer"
        }
      }
    }
  },
  "required": ["reasoning", "clusters"]
}
"""

# Refinement prompt, sources from the ACION100M paper
PROMPT_TEMPLATE_PHASE_2 = """Now, carefully analyze, verify, and revise the previous draft so that it is fully accurate, faithful to the provided
content, and strictly adheres to all stated guidelines and requirements."""

if __name__ == "__main__":
    print(f"Model ID: {MODEL_ID}")
    print(f"Max Model Context: {MAX_MODEL_CONTEXT}")
    print(f"GPU Count: {GPU_COUNT}")
    print(f"Input Data Path: {INPUT_DATA_PATH}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Video Lengths Path: {VIDEO_LENGTHS_PATH}")
    print(f"Prompt Template Phase 1: {PROMPT_TEMPLATE_PHASE_1}")
    print(f"Prompt Template Phase 2: {PROMPT_TEMPLATE_PHASE_2}")

    # A dictionary mapping video_id to its duration in seconds
    with open(VIDEO_LENGTHS_PATH, "r") as f:
        video_lengths = json.load(f)

    with open(INPUT_DATA_PATH, "rb") as f:
        dataset = pkl.load(f)

    grouped_by_video = defaultdict(list)
    for row in dataset:
        grouped_by_video[row[1]].append(row)

    print(f"Total videos: {len(grouped_by_video)}")

    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=GPU_COUNT,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.90,
        max_model_len=MAX_MODEL_CONTEXT,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    sampling_initial = SamplingParams(temperature=0.0, max_tokens=4096)
    sampling_refine = SamplingParams(temperature=0.1, max_tokens=4096)

    # Maybe later on I can add the sharding logic here, but for now let's just do one video to test the pipeline
    all_prompts = []
