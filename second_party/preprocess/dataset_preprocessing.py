"""
Dataset Preprocessing - Semantic Caption Merging (Phase 2, Refined)

Given deduplicated egocentric video captions (output of Phase 1), this script
identifies groups of segments that are *lexically different but semantically
equivalent* - i.e. they describe the same atomic action in different words -
and emits merge-groups so that both phrasings are preserved as contrastive-
learning augmentations while their timestamps are unified.

Pipeline
--------
1.  For every video, a sliding window of segments is sent to gpt-oss-120b
    with the INITIAL prompt asking it to output merge clusters.
2.  The model's draft answer is fed back together with the original input
    in a REFINEMENT turn so the model can self-correct hallucinated merges.
3.  The validated clusters are written out.

gpt-oss-120b - specific optimisations (marked with [OPT] in comments)
----------------------------------------------------------------------
*  [OPT-1]  Model-specific stop-token IDs (199999, 200002) to avoid
            runaway generation.
*  [OPT-2]  Two-temperature strategy: T=0.7 for the creative first pass
            (catch all plausible merges) and T=0.2 for the conservative
            refinement pass (prune false positives).
*  [OPT-3]  Explicit JSON-schema block inside the prompt so the model's
            internal constrained-decoding biases activate.
*  [OPT-4]  System prompt kept intentionally short and factual - large
            instruction-tuned models perform better with concise role
            descriptions rather than verbose personas.
*  [OPT-5]  Sliding-window chunking with configurable overlap to stay
            well within the 32 768-token context limit while retaining
            cross-boundary context.
*  [OPT-6]  Batch generation via vLLM for maximum GPU throughput.
"""

import torch
import pickle as pkl
import json
import re
import os
import argparse
from vllm import LLM, SamplingParams
from collections import defaultdict
from transformers import AutoTokenizer

# ===================================================================== #
#                           CONFIGURATION                                #
# ===================================================================== #

MODEL_ID = "openai/gpt-oss-120b"
MAX_MODEL_CONTEXT = 32768
GPU_COUNT = torch.cuda.device_count()

WINDOW_SIZE = 80  # segments per prompt window
OVERLAP_SIZE = 20  # overlap between consecutive windows
BATCH_SIZE = 128  # prompts per vLLM batch call
PROXIMITY_THRESHOLD = 1.0  # seconds – segments whose gap is ≤ this are
# considered "near-consecutive" merge candidates

INPUT_DATA_PATH = (
    "/dais/fs/scratch/dduka/databases/ego4d/ego4d_train_deduplicated_with_uuid.pkl"
)
OUTPUT_DIR = "/dais/fs/scratch/dduka/databases/ego4d/shards/"

# ===================================================================== #
#                              PROMPTS                                   #
# ===================================================================== #

# ---- System prompt -------------------------------------------------- #
# [OPT-4] Kept concise – gpt-oss-120b responds better to short, direct
#          role assignments than to elaborate personas.
SYSTEM_PROMPT = (
    "You are an expert data curator for egocentric (first-person) video "
    "datasets. You identify when multiple caption segments describe the "
    "same atomic action in different words and should be grouped together."
)

# ---- JSON output schema (embedded in prompt) ------------------------ #
# [OPT-3] Providing a concrete JSON-schema nudges the model's decoder
#          towards well-formed JSON and reduces parsing failures.
RESPONSE_SCHEMA = json.dumps(
    {
        "type": "array",
        "items": {
            "type": "array",
            "items": {"type": "integer"},
            "description": (
                "A cluster of segment IDs that describe the same atomic "
                "action and should be merged. Single-element lists are "
                "allowed for segments that have no merge partner."
            ),
        },
        "description": (
            "A list of clusters. Every input ID must appear in exactly one "
            "cluster. Clusters are ordered by the earliest timestamp of "
            "their members."
        ),
    },
    indent=2,
    ensure_ascii=False,
)

# ---- Initial merge prompt (Pass 1) --------------------------------- #
TASK_TEMPLATE = """\
### TASK — Semantic Caption Merging for Contrastive Learning

You are given a chronologically ordered list of caption segments from an \
egocentric video. Each segment has an **ID**, a **[start - end]** time \
range (seconds), and a **caption** describing what the camera wearer \
("#C") is doing.

**Problem**: Some consecutive or near-consecutive segments describe the \
**exact same atomic action** but use different wording (e.g. "#C picks up \
the cup" vs "#C takes the cup"). These lexical variants are *valuable* — \
we want to keep all of them as augmentation for contrastive learning — \
but their timestamps must be unified and the segments must be grouped \
together.

**Your goal**: Produce a clustering of the IDs below so that every group \
contains segments which refer to the **same visual moment / atomic action**.

#### Merging criteria

Merge two or more segments **only when ALL** of the following hold:

1. **Temporal proximity** — the segments overlap, or the gap between the \
end of one and the start of the next is ≤ {proximity_threshold} seconds.
2. **Semantic identity** — the captions describe the **same atomic \
interaction** with the same actor(s) and object(s), even if the verb, \
phrasing, or level of detail differ.
3. **No distinct sub-events** — the captions do NOT describe a sequence \
of separate steps (e.g. "opens fridge" then "takes milk out" are two \
actions, not one).

**Do NOT merge** when:
- Captions describe **different objects or goals** even if the verb is \
the same.
- The temporal gap suggests an **intentional segmentation boundary**.
- Merging would **conflate distinct repetitions** of an action (e.g. \
"picks up cup" at 10 s and again at 25 s are two events).

#### Input segments

{captions_str}

#### Output format

Return **only** a JSON list of lists inside a markdown code block. \
Every input ID must appear in **exactly one** list. Order the outer list \
by earliest start time of each cluster.

The schema is:
```json
{response_schema}
```

Example (for illustration only):
```json
[[0, 1], [2], [3, 4, 5], [6]]
```
"""

# ---- Self-refinement prompt (Pass 2) ------------------------------- #
# Adapted from Action100M's REFINE_INSTRUCTION.  The model re-reads its
# own draft together with the original input and applies conservative
# corrections.  This step is the main defence against hallucinated merges.
REFINE_TEMPLATE = """\
### Self-Refinement — Verify & Correct

Below is your previous draft clustering for the segments listed above.

**Draft output**:
```json
{draft_output}
```

Now carefully re-examine every cluster in the draft against the original \
segments. For each cluster with ≥ 2 members, verify:

1. **Temporal proximity** — do the segments actually overlap or fall \
within {proximity_threshold} s of each other? If not, split them.
2. **Semantic identity** — do the captions truly describe the *same* \
atomic action on the *same* object? Pay close attention to object nouns \
and directional verbs ("put down" ≠ "pick up"). If they differ, split.
3. **Coverage** — is every input ID present exactly once? If any ID is \
missing or duplicated, fix it.
4. **No over-merging** — did you accidentally group a sequence of \
*different* actions just because they are temporally close? If so, \
split them.

After your analysis, output the **corrected** JSON list of lists inside \
a markdown code block. If the draft is already correct, output it \
unchanged. Do **not** add any explanation outside the code block.
"""

# ===================================================================== #
#                            UTILITIES                                   #
# ===================================================================== #


def get_sliding_window_chunks(items, window_size, overlap):
    """Split *items* into overlapping windows."""
    chunks = []
    i = 0
    while i < len(items):
        chunks.append(items[i : i + window_size])
        if i + window_size >= len(items):
            break
        i += window_size - overlap
    return chunks


def extract_json(text):
    """Extract a JSON list-of-lists from model output."""
    # Try fenced code block first
    code_block = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if code_block:
        return code_block.group(1).strip()
    # Fall back to bare list-of-lists
    matches = re.findall(r"(\[[\s\r\n]*\[[\s\S]*?\][\s\r\n]*\])", text)
    if matches:
        return matches[-1].strip()
    return None


def validate_clusters(clusters, expected_ids):
    """
    Ensure every expected ID appears exactly once. Returns a cleaned
    version or None if unrecoverable.
    """
    seen = set()
    clean = []
    for cluster in clusters:
        filtered = [
            int(idx)
            for idx in cluster
            if int(idx) in expected_ids and int(idx) not in seen
        ]
        for idx in filtered:
            seen.add(idx)
        if filtered:
            clean.append(filtered)
    # Add back any missing IDs as singletons
    for idx in expected_ids - seen:
        clean.append([idx])
    return clean


def format_captions(chunk):
    """Render a list of caption rows into the input string."""
    lines = []
    for local_id, row in enumerate(chunk):
        # row layout: [uuid, video_id, start, end, caption, ...]
        lines.append(f"ID {local_id}  [{row[2]:.2f} – {row[3]:.2f}]  {row[4]}")
    return "\n".join(lines)


# ===================================================================== #
#                          PROMPT BUILDERS                               #
# ===================================================================== #


def build_initial_prompt(captions_str, tokenizer):
    """Build the Pass-1 prompt string for vLLM."""
    user_content = TASK_TEMPLATE.format(
        captions_str=captions_str,
        proximity_threshold=PROXIMITY_THRESHOLD,
        response_schema=RESPONSE_SCHEMA,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def build_refinement_prompt(captions_str, draft_json_str, tokenizer):
    """
    Build the Pass-2 self-refinement prompt string.

    The conversation is:
        system  →  initial user turn  →  assistant draft  →  refinement user turn
    This gives the model full context to self-correct.
    """
    initial_user = TASK_TEMPLATE.format(
        captions_str=captions_str,
        proximity_threshold=PROXIMITY_THRESHOLD,
        response_schema=RESPONSE_SCHEMA,
    )
    refine_user = REFINE_TEMPLATE.format(
        draft_output=draft_json_str,
        proximity_threshold=PROXIMITY_THRESHOLD,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": initial_user},
        {"role": "assistant", "content": f"```json\n{draft_json_str}\n```"},
        {"role": "user", "content": refine_user},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ===================================================================== #
#                             MAIN LOOP                                  #
# ===================================================================== #


def main():
    parser = argparse.ArgumentParser(
        description="Phase-2 semantic caption merging with self-refinement."
    )
    parser.add_argument("--job_idx", type=int, default=0)
    parser.add_argument("--num_jobs", type=int, default=1)
    parser.add_argument(
        "--skip_refinement",
        action="store_true",
        help="Skip the self-refinement pass (faster, slightly less accurate).",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Load data -------------------------------------------------- #
    print("Loading data …")
    with open(INPUT_DATA_PATH, "rb") as f:
        dataset = pkl.load(f)

    grouped_by_video = defaultdict(list)
    for row in dataset:
        grouped_by_video[row[1]].append(row)

    all_video_ids = sorted(grouped_by_video.keys())
    chunk_size = (len(all_video_ids) + args.num_jobs - 1) // args.num_jobs
    my_video_ids = all_video_ids[
        args.job_idx * chunk_size : (args.job_idx + 1) * chunk_size
    ]

    row_lookup = {
        row[0]: list(row) for vid in my_video_ids for row in grouped_by_video[vid]
    }

    # ---- Initialise vLLM -------------------------------------------- #
    print("Initialising vLLM for gpt-oss-120b …")
    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=GPU_COUNT,
        gpu_memory_utilization=0.90,
        max_model_len=MAX_MODEL_CONTEXT,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # [OPT-1] Model-specific stop tokens to prevent runaway generation.
    # [OPT-2] Higher temperature for creative first pass.
    sampling_initial = SamplingParams(
        temperature=0.7,
        max_tokens=8192,
        stop_token_ids=[199999, 200002],
    )
    # [OPT-2] Low temperature for conservative refinement pass.
    sampling_refine = SamplingParams(
        temperature=0.2,
        max_tokens=8192,
        stop_token_ids=[199999, 200002],
    )

    # ---- Build Pass-1 prompts --------------------------------------- #
    all_prompts = []
    prompt_meta = []  # (id_mapping, captions_str) per window

    for vid in my_video_ids:
        items = sorted(grouped_by_video[vid], key=lambda x: x[2])
        chunks = get_sliding_window_chunks(items, WINDOW_SIZE, OVERLAP_SIZE)
        for chunk in chunks:
            mapping = {i: c[0] for i, c in enumerate(chunk)}
            caps_str = format_captions(chunk)
            prompt = build_initial_prompt(caps_str, tokenizer)
            all_prompts.append(prompt)
            prompt_meta.append((mapping, caps_str))

    total_windows = len(all_prompts)
    print(f"Pass 1: generating for {total_windows} windows …")

    # ---- Pass 1 — initial clustering -------------------------------- #
    # [OPT-6] Batched generation via vLLM.
    draft_results = [None] * total_windows

    for batch_start in range(0, total_windows, BATCH_SIZE):
        batch_prompts = all_prompts[batch_start : batch_start + BATCH_SIZE]
        outputs = llm.generate(batch_prompts, sampling_initial, use_tqdm=True)
        for j, output in enumerate(outputs):
            idx = batch_start + j
            raw = output.outputs[0].text
            json_str = extract_json(raw)
            draft_results[idx] = json_str

    # ---- Pass 2 — self-refinement ----------------------------------- #
    if not args.skip_refinement:
        refine_prompts = []
        refine_indices = []

        for idx, (json_str, (mapping, caps_str)) in enumerate(
            zip(draft_results, prompt_meta)
        ):
            if json_str is None:
                continue
            prompt = build_refinement_prompt(caps_str, json_str, tokenizer)
            refine_prompts.append(prompt)
            refine_indices.append(idx)

        print(
            f"Pass 2 (refinement): generating for " f"{len(refine_prompts)} windows …"
        )

        for batch_start in range(0, len(refine_prompts), BATCH_SIZE):
            batch = refine_prompts[batch_start : batch_start + BATCH_SIZE]
            batch_idx = refine_indices[batch_start : batch_start + BATCH_SIZE]
            outputs = llm.generate(batch, sampling_refine, use_tqdm=True)
            for j, output in enumerate(outputs):
                refined_json = extract_json(output.outputs[0].text)
                if refined_json is not None:
                    draft_results[batch_idx[j]] = refined_json

    # ---- Apply clusters --------------------------------------------- #
    merge_count = 0
    for idx, (mapping, _caps_str) in enumerate(prompt_meta):
        json_str = draft_results[idx]
        if json_str is None:
            continue
        try:
            clusters = json.loads(json_str)
        except (json.JSONDecodeError, TypeError) as exc:
            print(f"[WARN] JSON parse error at window {idx}: {exc}")
            continue

        expected_ids = set(mapping.keys())
        clusters = validate_clusters(clusters, expected_ids)

        for cluster in clusters:
            valid_uids = [mapping[cid] for cid in cluster if cid in mapping]
            if len(valid_uids) <= 1:
                continue

            # Unify timestamps across cluster members
            u_start = min(row_lookup[uid][2] for uid in valid_uids)
            u_end = max(row_lookup[uid][3] for uid in valid_uids)
            for uid in valid_uids:
                row_lookup[uid][2] = u_start
                row_lookup[uid][3] = u_end
            merge_count += 1

    # ---- Persist ---------------------------------------------------- #
    shard_path = os.path.join(OUTPUT_DIR, f"shard_{args.job_idx}.pkl")
    final_dataset = list(row_lookup.values())
    with open(shard_path, "wb") as f:
        pkl.dump(final_dataset, f)

    print(
        f"Shard {args.job_idx} saved — {len(final_dataset)} rows, "
        f"{merge_count} clusters merged."
    )


if __name__ == "__main__":
    main()
