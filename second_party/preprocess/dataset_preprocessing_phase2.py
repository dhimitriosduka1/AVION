import torch
import pickle as pkl
import json
import re
import os
import networkx as nx
import matplotlib.pyplot as plt
from vllm import LLM, SamplingParams
from collections import defaultdict
from transformers import AutoTokenizer

# --- CONFIGURATION ---
MODEL_ID = "Qwen/Qwen3-32B"
MAX_MODEL_CONTEXT = 32768
GPU_COUNT = torch.cuda.device_count()

WINDOW_SIZE = 60
OVERLAP_SIZE = 15

INPUT_DATA_PATH = (
    "/dais/fs/scratch/dduka/databases/ego4d/ego4d_train_deduplicated_with_uuid.pkl"
)
OUTPUT_DATA_PATH = (
    "/dais/fs/scratch/dduka/databases/ego4d/ego4d_train_timestamp_fixed.pkl"
)
SAVE_DIR = "/u/dduka/project/AVION/images"

os.makedirs(SAVE_DIR, exist_ok=True)

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


def visualize_chunk(chunk_items, clusters, chunk_idx, video_id):
    plt.figure(figsize=(10, 7))
    temp_G = nx.Graph()
    chunk_uuids = [c[0] for c in chunk_items]
    temp_G.add_nodes_from(chunk_uuids)
    for cluster in clusters:
        for i in range(len(cluster) - 1):
            if cluster[i] in chunk_uuids and cluster[i + 1] in chunk_uuids:
                temp_G.add_edge(cluster[i], cluster[i + 1])
    pos = nx.spring_layout(temp_G, k=0.3, seed=42)
    components = list(nx.connected_components(temp_G))
    colors = plt.cm.get_cmap("rainbow", len(components))
    for i, nodes in enumerate(components):
        nx.draw_networkx_nodes(
            temp_G, pos, nodelist=list(nodes), node_color=[colors(i)], node_size=300
        )
    nx.draw_networkx_edges(temp_G, pos, alpha=0.3)
    nx.draw_networkx_labels(temp_G, pos, font_size=6)
    plt.title(f"Video: {video_id} | Chunk: {chunk_idx}")
    plt.axis("off")
    plt.savefig(
        os.path.join(SAVE_DIR, f"graph_{video_id}_chunk_{chunk_idx}.png"), dpi=150
    )
    plt.close()


def main():
    print(f"Loading data...")
    with open(INPUT_DATA_PATH, "rb") as f:
        dataset = pkl.load(f)

    row_lookup = {row[0]: row for row in dataset}
    grouped_by_video = defaultdict(list)
    for row in dataset:
        grouped_by_video[row[1]].append(row)

    print(f"Initializing vLLM...")
    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=GPU_COUNT,
        gpu_memory_utilization=0.92,
        max_model_len=MAX_MODEL_CONTEXT,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    all_prompts = []
    chunk_metadata = []

    for vid, items in grouped_by_video.items():
        items.sort(key=lambda x: x[2])
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
            )
            all_prompts.append(prompt)
            chunk_metadata.append({"vid": vid, "items": chunk})

    # For a full run, remove the slicing below
    all_prompts = all_prompts[:2]

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=16384,
    )
    outputs = llm.generate(all_prompts, sampling_params)

    G = nx.Graph()
    G.add_nodes_from(row_lookup.keys())

    print("Processing outputs and building graph...")
    for i, output in enumerate(outputs):
        raw_text = output.outputs[0].text

        # 1. Strip thinking blocks
        clean_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)

        # 2. Extract JSON (Finds the outermost list)
        match = re.search(r"(\[[\s\S]*\])", clean_text)
        if match:
            try:
                clusters = json.loads(match.group(1))
                if i < 20:
                    visualize_chunk(
                        chunk_metadata[i]["items"],
                        clusters,
                        i,
                        chunk_metadata[i]["vid"],
                    )

                for cluster in clusters:
                    for j in range(len(cluster) - 1):
                        u1, u2 = cluster[j], cluster[j + 1]
                        if u1 in row_lookup and u2 in row_lookup:
                            G.add_edge(u1, u2)
            except Exception as e:
                print(f"JSON error in chunk {i}: {e}")
        else:
            print(f"No valid JSON pattern in chunk {i}")

    print("Reconciling clusters into final dataset...")
    final_dataset = []
    for component in nx.connected_components(G):
        starts = [row_lookup[uid][2] for uid in component]
        ends = [row_lookup[uid][3] for uid in component]
        u_start, u_end = min(starts), max(ends)
        for uid in component:
            orig = row_lookup[uid]
            final_dataset.append([uid, orig[1], u_start, u_end, orig[4]])

    print(f"Saving {len(final_dataset)} rows to {OUTPUT_DATA_PATH}...")
    with open(OUTPUT_DATA_PATH, "wb") as f:
        pkl.dump(final_dataset, f)


if __name__ == "__main__":
    main()
