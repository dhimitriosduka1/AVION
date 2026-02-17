import os
import re
import json
import torch
import pickle as pkl
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

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROMPT_TEMPLATE_PHASE_1 = """
Reasoning: High

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

Crucial Context: The captions provided are manually annotated, providing a strong semantic baseline.
However, their start and end timestamps are computed using a heuristic. This combination can result in
artificial temporal overlaps, fragmented boundaries, and concurrent segments describing the same underlying
action using natural human lexical variation (e.g., "chopping tomato" vs. "slicing a red vegetable").

You must group the segment IDs into discrete clusters, where each cluster represents a single, distinct atomic action.

Guidelines and requirements:
- **Domain Context:** Assume a first-person perspective (the camera wearer). Focus primarily on hand-object interactions.
- **Semantic & Temporal Clustering:** Evaluate both temporal proximity (overlapping or adjacent heuristic boundaries) and semantic similarity.
Group IDs together if they describe the same underlying action, accommodating for human subjective differences in the annotations.
- **Lexical Resolution:** Look past superficial lexical differences caused by different annotators describing the same event at the same time.
- **Completeness and Exclusivity:** Every ID from the input list MUST be included in exactly one cluster. Do not drop any IDs, and do not place
the same ID into multiple clusters. If an ID represents a standalone action that shouldn't be merged, it should be in a cluster by itself (e.g., `[4]`).
- **Reasoning First:** Think step-by-step. Analyze temporal overlaps and semantic similarities before outputting the final JSON.

# Response Format
Return a valid JSON object strictly adhering to this schema:

{{
    "type": "object",
    "properties": {{
        "reasoning": {{
            "type": "string",
            "description": "A brief, step-by-step explanation of how the IDs were clustered based on heuristic temporal overlap and lexical variation."
        }},
        "clusters": {{
            "type": "array",
            "description": "A list of lists. Each inner list contains the integer IDs of the segments that should be merged together.",
            "items": {{
                "type": "array",
                "items": {{
                    "type": "integer"
                }}
            }}
        }}
    }},
    "required": ["reasoning", "clusters"]
}}
"""

PROMPT_TEMPLATE_PHASE_2 = """Now, carefully analyze, verify, and revise the previous draft so that it is fully accurate, faithful to the provided
content, and strictly adheres to all stated guidelines and requirements."""


# --- 1. Union-Find (DSU) Implementation ---
class UnionFind:
    def __init__(self):
        self.parent = {}

    def add(self, i):
        if i not in self.parent:
            self.parent[i] = i

    def find(self, i):
        self.add(i)  # Ensure node exists
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])  # Path compression optimization
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j

    def get_clusters(self):
        clusters = defaultdict(list)
        for i in self.parent:
            clusters[self.find(i)].append(i)
        return list(clusters.values())


# --- 2. Helper for Parsing LLM JSON Output ---
def extract_clusters_from_response(text):
    """Safely extracts the 'clusters' list from the LLM's JSON output."""
    try:
        text = text.strip()
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1:
            clean_json = text[start_idx : end_idx + 1]
            data = json.loads(clean_json)
            return data.get("clusters", [])
    except Exception as e:
        print(f"Failed to parse JSON: {e}\nRaw output: {text[:200]}...")
    return []


if __name__ == "__main__":
    print(f"Model ID: {MODEL_ID}")
    print(f"Max Model Context: {MAX_MODEL_CONTEXT}")
    print(f"GPU Count: {GPU_COUNT}")
    print(f"Input Data Path: {INPUT_DATA_PATH}")

    # Load Video Lengths mapping
    with open(VIDEO_LENGTHS_PATH, "r") as f:
        video_lengths = json.load(f)

    # Load Pickled Dataset
    with open(INPUT_DATA_PATH, "rb") as f:
        dataset = pkl.load(f)

    grouped_by_video = defaultdict(list)
    for row in dataset:
        grouped_by_video[row[1]].append(row)

    print(f"Total videos: {len(grouped_by_video)}")

    # Init LLM
    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=GPU_COUNT,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.90,
        max_model_len=MAX_MODEL_CONTEXT,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Sampling parameters setup with strict STOP strings to prevent infinite looping
    sampling_initial = SamplingParams(
        temperature=0.0, max_tokens=2048 * 4, stop=["```", "}\n}"]
    )
    sampling_refine = SamplingParams(
        temperature=0.2, max_tokens=2048 * 4, stop=["```", "}\n}"]
    )

    # Process videos
    for test_video_id, raw_segments in grouped_by_video.items():
        print(f"\n============================================")
        print(f"Processing Video ID: {test_video_id}")

        # Sort chronologically
        segments = sorted(raw_segments, key=lambda x: x[2])

        # DYNAMIC INTEGER MAPPING
        int_to_uuid = {}
        uuid_to_segment = {}

        for idx, row in enumerate(segments):
            uuid = row[0]
            int_to_uuid[idx] = uuid
            uuid_to_segment[uuid] = row

        video_duration = video_lengths.get(test_video_id, "Unknown")

        # --- 3. Sliding Window Configuration ---
        WINDOW_SIZE = 40
        OVERLAP = 10
        step = WINDOW_SIZE - OVERLAP

        windows = []
        # Create windows using the new integer indices
        for i in range(0, len(segments), step):
            window_indices = list(range(i, min(i + WINDOW_SIZE, len(segments))))
            windows.append(window_indices)

        print(f"Split into {len(windows)} overlapping windows.")

        # --- 4. Prepare Phase 1 Prompts ---
        prompts_r1 = []
        for window_indices in windows:
            start_time = segments[window_indices[0]][2]
            end_time = segments[window_indices[-1]][3]
            num_captions = len(window_indices)

            # Format the list using the INTEGER ID and strictly clean the caption
            formatted_lines = []
            for idx in window_indices:
                row = segments[idx]
                clean_caption = row[4].replace("#C C ", "").replace("#C ", "").strip()
                formatted_lines.append(
                    f'[{idx}, {row[2]}, {row[3]}, "{clean_caption}"]'
                )

            formatted_list = "\n".join(formatted_lines)

            prompt = PROMPT_TEMPLATE_PHASE_1.format(
                video_duration=video_duration,
                start_time=start_time,
                end_time=end_time,
                num_captions=num_captions,
                formatted_input_list=formatted_list,
            )
            prompts_r1.append(prompt)

        # --- 5. The 3-Round Self-Refine Loop ---
        print("Starting Round 1 (Initial Draft)...")
        outputs_r1 = llm.generate(prompts_r1, sampling_initial)
        texts_r1 = [out.outputs[0].text for out in outputs_r1]

        print("Starting Round 2 (Refinement)...")
        prompts_r2 = [
            f"{p}\n\n{t}\n\n{PROMPT_TEMPLATE_PHASE_2}"
            for p, t in zip(prompts_r1, texts_r1)
        ]
        outputs_r2 = llm.generate(prompts_r2, sampling_refine)
        texts_r2 = [out.outputs[0].text for out in outputs_r2]

        print("Starting Round 3 (Final Refinement)...")
        prompts_r3 = [
            f"{p}\n\n{t}\n\n{PROMPT_TEMPLATE_PHASE_2}"
            for p, t in zip(prompts_r2, texts_r2)
        ]
        outputs_r3 = llm.generate(prompts_r3, sampling_refine)
        texts_r3 = [out.outputs[0].text for out in outputs_r3]

        # Print all three rounds of outputs for debugging
        for i in range(len(prompts_r1)):
            print(f"\n--- Window {i+1} ---")
            print("Round 1 Output:")
            print(texts_r1[i])
            print("\nRound 2 Output:")
            print(texts_r2[i])
            print("\nRound 3 Output:")
            print(texts_r3[i])

        # --- 6. Graph Merging (Union-Find) ---
        uf = UnionFind()

        for idx, final_text in enumerate(texts_r3):
            clusters = extract_clusters_from_response(final_text)
            for cluster in clusters:
                if not cluster:
                    continue

                # Single-item clusters still need to be added to the graph
                if len(cluster) == 1:
                    uf.add(cluster[0])

                # Link all IDs in the sub-cluster to the first ID
                first_id = cluster[0]
                for connected_id in cluster[1:]:
                    uf.union(first_id, connected_id)

        # --- 7. Boundary Recalculation & Save Data ---
        global_clusters = uf.get_clusters()
        final_video_actions = []

        for cluster in global_clusters:
            # Map the integer IDs back to UUIDs, then to the raw rows
            cluster_segments = [
                uuid_to_segment[int_to_uuid[cid]]
                for cid in cluster
                if cid in int_to_uuid
            ]
            if not cluster_segments:
                continue

            # Set boundaries to the absolute min and max of the cluster
            min_start = min(seg[2] for seg in cluster_segments)
            max_end = max(seg[3] for seg in cluster_segments)

            # Deduplicate the lexical variations of the captions
            unique_captions = list(set(seg[4] for seg in cluster_segments))
            original_uuids = [seg[0] for seg in cluster_segments]

            final_video_actions.append(
                {
                    "original_uuids": original_uuids,
                    "start_time": min_start,
                    "end_time": max_end,
                    "captions": unique_captions,
                }
            )

        # Sort chronologically by the new calculated start time
        final_video_actions = sorted(final_video_actions, key=lambda x: x["start_time"])

        output_file = os.path.join(OUTPUT_DIR, f"{test_video_id}_merged.json")
        with open(output_file, "w") as f:
            json.dump(final_video_actions, f, indent=4)

        print(
            f"Successfully processed and saved {len(final_video_actions)} merged actions to {output_file}"
        )

        # Break here to only test the first video. Remove this to process the entire dataset.
        break
